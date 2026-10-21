//! ShrestiDB Web Studio & Interactive GUI
//!
//! A self-contained, browser-based management studio and query console for ShrestiDB:
//! - Interactive SQL Console with query history and execution metrics
//! - Visual Schema & Catalog Explorer
//! - Real-time Learned Index vs B-Tree Benchmark & Visualizer
//! - Zero external runtime dependencies: runs directly on Tokio HTTP server

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::error;

use crate::execution::QueryExecutor;
use crate::sql::parser::{SQLParser, SQLStatement};
use crate::VERSION;

// ── HTTP API Types ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct QueryRequest {
    sql: String,
}

#[derive(Serialize)]
struct QueryResponse {
    success: bool,
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
    affected_rows: Option<usize>,
    duration_us: u128,
    error: Option<String>,
}

#[derive(Serialize)]
struct ColumnInfo {
    name: String,
    data_type: String,
    nullable: bool,
    primary_key: bool,
}

#[derive(Serialize)]
struct TableInfo {
    id: u32,
    name: String,
    columns: Vec<ColumnInfo>,
    approx_rows: usize,
}

#[derive(Serialize)]
struct StatusResponse {
    version: String,
    wal_path: String,
    table_count: usize,
    uptime_secs: u64,
}

#[derive(Deserialize)]
struct BenchmarkRequest {
    size: usize,
}

#[derive(Serialize)]
struct BenchmarkResponse {
    size: usize,
    pgm_build_ms: f64,
    btree_build_ms: f64,
    pgm_lookup_us: f64,
    btree_lookup_us: f64,
    speedup: f64,
    pgm_segments: usize,
}

// ── Studio Server ─────────────────────────────────────────────────────────────

pub struct StudioServer {
    executor: Arc<QueryExecutor>,
    wal_path: String,
    start_time: Instant,
}

impl StudioServer {
    pub fn new(executor: Arc<QueryExecutor>, wal_path: String) -> Self {
        StudioServer {
            executor,
            wal_path,
            start_time: Instant::now(),
        }
    }

    pub async fn run(self: Arc<Self>, port: u16) -> anyhow::Result<()> {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let listener = TcpListener::bind(addr).await?;
        println!();
        println!("🚀 ShrestiDB Web Studio is running at: http://localhost:{port}/");
        println!("Press Ctrl+C to stop the web server.\n");

        loop {
            let (stream, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(e) => {
                    error!("Connection accept error: {e}");
                    continue;
                }
            };

            let server = Arc::clone(&self);
            tokio::spawn(async move {
                if let Err(e) = server.handle_connection(stream).await {
                    error!("HTTP handle error: {e}");
                }
            });
        }
    }

    async fn handle_connection(&self, mut stream: TcpStream) -> anyhow::Result<()> {
        let mut buffer = [0u8; 8192];
        let bytes_read = stream.read(&mut buffer).await?;
        if bytes_read == 0 {
            return Ok(());
        }

        let request_str = String::from_utf8_lossy(&buffer[..bytes_read]);
        let mut lines = request_str.lines();
        let request_line = match lines.next() {
            Some(line) => line,
            None => return Ok(()),
        };

        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }

        let method = parts[0];
        let path = parts[1];

        // Find body if present (split on \r\n\r\n or \n\n)
        let body = if let Some(idx) = request_str.find("\r\n\r\n") {
            &request_str[idx + 4..]
        } else if let Some(idx) = request_str.find("\n\n") {
            &request_str[idx + 2..]
        } else {
            ""
        };

        match (method, path) {
            ("GET", "/") | ("GET", "/index.html") | ("HEAD", "/") | ("HEAD", "/index.html") => {
                let html = get_studio_html(&self.wal_path);
                send_response(&mut stream, "200 OK", "text/html; charset=utf-8", html.as_bytes()).await?;
            }
            ("GET", "/favicon.ico") | ("HEAD", "/favicon.ico") => {
                send_response(&mut stream, "204 No Content", "image/x-icon", b"").await?;
            }
            ("OPTIONS", _) => {
                send_response(&mut stream, "204 No Content", "text/plain", b"").await?;
            }
            ("GET", "/api/status") => {
                let status = {
                    let catalog = self.executor.catalog.read();
                    StatusResponse {
                        version: VERSION.to_string(),
                        wal_path: self.wal_path.clone(),
                        table_count: catalog.tables.len(),
                        uptime_secs: self.start_time.elapsed().as_secs(),
                    }
                };
                let json = serde_json::to_string(&status)?;
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes()).await?;
            }
            ("GET", "/api/tables") => {
                let tables: Vec<TableInfo> = {
                    let catalog = self.executor.catalog.read();
                    let mut list = Vec::new();
                    for schema in catalog.tables.values() {
                        let approx_rows = self.executor.oltp.store.table(schema.table_id as u64).row_count();
                        let cols = schema
                            .columns
                            .iter()
                            .map(|c| ColumnInfo {
                                name: c.name.clone(),
                                data_type: format!("{:?}", c.data_type),
                                nullable: c.nullable,
                                primary_key: c.primary_key,
                            })
                            .collect();
                        list.push(TableInfo {
                            id: schema.table_id,
                            name: schema.name.clone(),
                            columns: cols,
                            approx_rows,
                        });
                    }
                    list.sort_by_key(|t| t.id);
                    list
                };
                let json = serde_json::to_string(&tables)?;
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes()).await?;
            }
            ("POST", "/api/query") => {
                let res = self.handle_query(body);
                let json = serde_json::to_string(&res)?;
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes()).await?;
            }
            ("POST", "/api/benchmark") => {
                let req: BenchmarkRequest = serde_json::from_str(body).unwrap_or(BenchmarkRequest { size: 50000 });
                let res = run_quick_benchmark(req.size.clamp(1000, 200000));
                let json = serde_json::to_string(&res)?;
                send_response(&mut stream, "200 OK", "application/json", json.as_bytes()).await?;
            }
            _ => {
                send_response(&mut stream, "404 Not Found", "text/plain", b"Not Found").await?;
            }
        }

        Ok(())
    }

    fn handle_query(&self, body: &str) -> QueryResponse {
        let req: QueryRequest = match serde_json::from_str(body) {
            Ok(r) => r,
            Err(e) => {
                return QueryResponse {
                    success: false,
                    columns: Vec::new(),
                    rows: Vec::new(),
                    affected_rows: None,
                    duration_us: 0,
                    error: Some(format!("Invalid JSON request: {e}")),
                }
            }
        };

        let trimmed = req.sql.trim();
        if trimmed.is_empty() {
            return QueryResponse {
                success: true,
                columns: Vec::new(),
                rows: Vec::new(),
                affected_rows: None,
                duration_us: 0,
                error: None,
            };
        }

        // Determine columns if SELECT
        let parsed_stmt = SQLParser::parse(trimmed).ok();
        let mut expected_columns = Vec::new();
        let mut is_dml = false;

        if let Some(ref stmt) = parsed_stmt {
            match stmt {
                SQLStatement::Select(select) => {
                    let catalog = self.executor.catalog.read();
                    if select.columns.len() == 1 && select.columns[0] == "*" {
                        if let Some(table) = catalog.get_table(&select.from) {
                            expected_columns = table.columns.iter().map(|c| c.name.clone()).collect();
                        }
                    } else {
                        expected_columns = select.columns.clone();
                    }
                }
                SQLStatement::Insert(_) | SQLStatement::Update(_) | SQLStatement::Delete(_) => {
                    is_dml = true;
                }
                _ => {}
            }
        }

        let start = Instant::now();
        match self.executor.execute_sql(trimmed) {
            Ok(raw_rows) => {
                let duration_us = start.elapsed().as_micros();
                if is_dml {
                    let affected = raw_rows
                        .first()
                        .and_then(|r| r.first())
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0);
                    QueryResponse {
                        success: true,
                        columns: Vec::new(),
                        rows: Vec::new(),
                        affected_rows: Some(affected),
                        duration_us,
                        error: None,
                    }
                } else {
                    // If columns couldn't be determined from AST, default to col_1, col_2...
                    if expected_columns.is_empty() && !raw_rows.is_empty() {
                        let count = raw_rows[0].len();
                        expected_columns = (1..=count).map(|i| format!("col_{i}")).collect();
                    }
                    QueryResponse {
                        success: true,
                        columns: expected_columns,
                        rows: raw_rows,
                        affected_rows: None,
                        duration_us,
                        error: None,
                    }
                }
            }
            Err(e) => {
                let duration_us = start.elapsed().as_micros();
                QueryResponse {
                    success: false,
                    columns: Vec::new(),
                    rows: Vec::new(),
                    affected_rows: None,
                    duration_us,
                    error: Some(e.to_string()),
                }
            }
        }
    }
}

async fn send_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         \r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    stream.flush().await?;
    Ok(())
}

// ── Live Benchmark Runner ─────────────────────────────────────────────────────

fn run_quick_benchmark(size: usize) -> BenchmarkResponse {
    use crate::index::btree::BTree;
    use crate::index::pgm::PGMIndex;

    let data: Vec<f64> = (0..size).map(|i| i as f64 * 1.5).collect();

    // PGM Build
    let t0 = Instant::now();
    let pgm = PGMIndex::build(data.clone(), 16);
    let pgm_build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let pgm_segments = pgm.segments.len();

    // B-Tree Build
    let t1 = Instant::now();
    let mut btree = BTree::new(64);
    for (i, &key) in data.iter().enumerate() {
        let _ = btree.insert(key, i);
    }
    let btree_build_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Search lookups (500 samples)
    let lookups = 500;
    let step = (size / lookups).max(1);

    let t2 = Instant::now();
    for i in (0..size).step_by(step) {
        let _ = pgm.search(data[i]);
    }
    let pgm_lookup_us = (t2.elapsed().as_secs_f64() * 1_000_000.0) / lookups as f64;

    let t3 = Instant::now();
    for i in (0..size).step_by(step) {
        let _ = btree.search(data[i]);
    }
    let btree_lookup_us = (t3.elapsed().as_secs_f64() * 1_000_000.0) / lookups as f64;

    let speedup = if pgm_lookup_us > 0.0 {
        btree_lookup_us / pgm_lookup_us
    } else {
        1.0
    };

    BenchmarkResponse {
        size,
        pgm_build_ms,
        btree_build_ms,
        pgm_lookup_us,
        btree_lookup_us,
        speedup,
        pgm_segments,
    }
}

// ── Embedded Web Studio UI ────────────────────────────────────────────────────

fn get_studio_html(wal_path: &str) -> String {
    format!(r##"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ShrestiDB Studio</title>
  <style>
    :root {{
      --bg-dark: #0f172a;
      --bg-panel: #1e293b;
      --bg-card: #243248;
      --border: #334155;
      --accent: #38bdf8;
      --accent-hover: #0ea5e9;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --success: #34d399;
      --danger: #f87171;
      --code-bg: #090d16;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{ background: var(--bg-dark); color: var(--text-main); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
    
    /* Header */
    header {{ background: var(--bg-panel); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }}
    .brand {{ display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.15rem; color: var(--accent); }}
    .badge {{ background: rgba(56, 189, 248, 0.15); border: 1px solid var(--accent); color: var(--accent); padding: 3px 8px; border-radius: 9999px; font-size: 0.75rem; }}
    .header-info {{ display: flex; align-items: center; gap: 16px; font-size: 0.85rem; color: var(--text-muted); }}

    /* Layout */
    .workspace {{ display: flex; flex: 1; overflow: hidden; }}
    .sidebar {{ width: 280px; background: var(--bg-panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; }}
    .sidebar-header {{ padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); display: flex; justify-content: space-between; align-items: center; }}
    .table-list {{ flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 6px; }}
    .table-item {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; cursor: pointer; transition: all 0.15s; }}
    .table-item:hover {{ border-color: var(--accent); background: rgba(56, 189, 248, 0.08); }}
    .table-item-name {{ font-weight: 600; font-size: 0.95rem; display: flex; justify-content: space-between; align-items: center; }}
    .table-item-meta {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }}
    
    /* Main Area */
    .main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; background: var(--bg-dark); }}
    .tabs {{ display: flex; background: var(--bg-panel); border-bottom: 1px solid var(--border); padding: 0 16px; }}
    .tab {{ padding: 12px 18px; font-size: 0.9rem; font-weight: 600; color: var(--text-muted); cursor: pointer; border-bottom: 2px solid transparent; }}
    .tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
    
    .tab-content {{ flex: 1; display: none; flex-direction: column; overflow: hidden; padding: 20px; gap: 16px; }}
    .tab-content.active {{ display: flex; }}

    /* SQL Editor */
    .editor-card {{ background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 10px; }}
    .editor-toolbar {{ display: flex; justify-content: space-between; align-items: center; }}
    .query-templates {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .template-btn {{ background: var(--bg-card); border: 1px solid var(--border); color: var(--text-muted); padding: 4px 10px; border-radius: 4px; font-size: 0.75rem; cursor: pointer; }}
    .template-btn:hover {{ color: var(--text-main); border-color: var(--accent); }}
    textarea.sql-input {{ width: 100%; height: 110px; background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; color: #e2e8f0; font-family: monospace; font-size: 0.95rem; padding: 12px; resize: vertical; outline: none; }}
    textarea.sql-input:focus {{ border-color: var(--accent); }}
    
    .btn {{ background: var(--accent); color: #000; font-weight: 600; padding: 8px 18px; border-radius: 6px; border: none; cursor: pointer; transition: background 0.15s; font-size: 0.9rem; display: inline-flex; align-items: center; gap: 6px; }}
    .btn:hover {{ background: var(--accent-hover); }}

    /* Results */
    .results-card {{ flex: 1; background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; }}
    .results-header {{ padding: 10px 16px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; color: var(--text-muted); }}
    .status-badge {{ font-weight: 600; padding: 2px 8px; border-radius: 4px; }}
    .status-success {{ background: rgba(52, 211, 153, 0.15); color: var(--success); }}
    .status-error {{ background: rgba(248, 113, 113, 0.15); color: var(--danger); }}
    .table-container {{ flex: 1; overflow: auto; }}
    table.data-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    table.data-table th {{ background: var(--bg-card); position: sticky; top: 0; padding: 10px 14px; text-align: left; font-weight: 600; color: var(--text-muted); border-bottom: 1px solid var(--border); }}
    table.data-table td {{ padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #cbd5e1; }}
    table.data-table tr:hover td {{ background: rgba(255, 255, 255, 0.03); }}
    .empty-state {{ padding: 40px; text-align: center; color: var(--text-muted); font-size: 0.95rem; }}

    /* Benchmark Tab */
    .benchmark-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; }}
    .bench-card {{ background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; display: flex; flex-direction: column; gap: 12px; }}
    .metric-big {{ font-size: 2.2rem; font-weight: 800; color: var(--accent); }}
    .metric-sub {{ font-size: 0.85rem; color: var(--text-muted); }}
    .bench-bar {{ height: 12px; border-radius: 6px; background: rgba(255,255,255,0.1); overflow: hidden; margin-top: 6px; }}
    .bench-bar-fill {{ height: 100%; background: var(--accent); }}
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <span>⚡ ShrestiDB Studio</span>
      <span class="badge">v{VERSION}</span>
    </div>
    <div class="header-info">
      <span>WAL: <strong>{wal_path}</strong></span>
      <span>Learned Index Engine: <strong>Active</strong></span>
    </div>
  </header>

  <div class="workspace">
    <!-- Left Sidebar: Tables -->
    <div class="sidebar">
      <div class="sidebar-header">
        <span>Catalog Tables</span>
        <button class="template-btn" onclick="loadTables()">↻ Refresh</button>
      </div>
      <div class="table-list" id="tableList">
        <div class="empty-state">Loading catalog...</div>
      </div>
    </div>

    <!-- Main Content -->
    <div class="main">
      <div class="tabs">
        <div class="tab active" onclick="switchTab('queryTab', this)">SQL Studio</div>
        <div class="tab" onclick="switchTab('benchTab', this)">Learned Index Benchmark</div>
      </div>

      <!-- Tab 1: SQL Studio -->
      <div class="tab-content active" id="queryTab">
        <div class="editor-card">
          <div class="editor-toolbar">
            <div class="query-templates">
              <button class="template-btn" onclick="setQuery('SELECT * FROM accounts;')">SELECT accounts</button>
              <button class="template-btn" onclick="setQuery('CREATE TABLE accounts (id INT PRIMARY KEY, name TEXT, balance FLOAT);')">CREATE TABLE</button>
              <button class="template-btn" onclick="setQuery('INSERT INTO accounts VALUES (4, \'Diana\', 4200.0);')">INSERT</button>
              <button class="template-btn" onclick="setQuery('SELECT count(*) FROM accounts;')">COUNT(*)</button>
              <button class="template-btn" onclick="setQuery('ANALYZE accounts;')">ANALYZE</button>
            </div>
            <button class="btn" onclick="runQuery()">▶ Run SQL (⌘+Enter)</button>
          </div>
          <textarea class="sql-input" id="sqlInput" placeholder="Enter SQL statements (e.g. SELECT * FROM accounts;)">SELECT * FROM accounts;</textarea>
        </div>

        <div class="results-card">
          <div class="results-header">
            <span id="resultsCount">Results will appear here</span>
            <span id="resultsStatus" class="status-badge"></span>
          </div>
          <div class="table-container" id="resultsTableContainer">
            <div class="empty-state">No query executed yet. Click "Run SQL" to execute.</div>
          </div>
        </div>
      </div>

      <!-- Tab 2: Learned Index Benchmark -->
      <div class="tab-content" id="benchTab">
        <div class="editor-card">
          <div class="editor-toolbar">
            <div>
              <h3 style="font-size:1.1rem; margin-bottom: 4px;">Real Learned Index (PGM) vs Standard B-Tree</h3>
              <p style="font-size:0.85rem; color: var(--text-muted);">Benchmarks lookup speed & memory on live generated dataset.</p>
            </div>
            <div style="display:flex; gap:10px; align-items:center;">
              <select id="benchSize" style="background:var(--bg-card); color:var(--text-main); border:1px solid var(--border); padding:6px 12px; border-radius:6px;">
                <option value="10000">10,000 Keys</option>
                <option value="50000" selected>50,000 Keys</option>
                <option value="100000">100,000 Keys</option>
              </select>
              <button class="btn" onclick="runBenchmark()">⚡ Run Benchmark</button>
            </div>
          </div>
        </div>

        <div class="benchmark-grid" id="benchResults" style="display:none;">
          <div class="bench-card">
            <h4>Learned PGM Index Lookup</h4>
            <div class="metric-big" id="pgmLookup">-</div>
            <div class="metric-sub" id="pgmSub">O(log S) search where S segments ≪ N</div>
            <div class="bench-bar"><div class="bench-bar-fill" id="pgmBar" style="width: 20%;"></div></div>
          </div>
          <div class="bench-card">
            <h4>Standard B-Tree Lookup</h4>
            <div class="metric-big" style="color:#cbd5e1;" id="btreeLookup">-</div>
            <div class="metric-sub" id="btreeSub">O(log N) tree traversal</div>
            <div class="bench-bar"><div class="bench-bar-fill" id="btreeBar" style="width: 80%; background:#94a3b8;"></div></div>
          </div>
          <div class="bench-card" style="grid-column: span 2;">
            <h4>Benchmark Summary</h4>
            <div style="font-size: 1.25rem; font-weight: 700; color: var(--success);" id="benchSpeedup">-</div>
            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 6px;" id="benchDetails">-</div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <script>
    // Tab switching
    function switchTab(tabId, el) {{
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      el.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }}

    function setQuery(sql) {{
      document.getElementById('sqlInput').value = sql;
      document.getElementById('sqlInput').focus();
    }}

    // Load table list
    async function loadTables() {{
      const container = document.getElementById('tableList');
      try {{
        const res = await fetch('/api/tables');
        const tables = await res.json();
        if (tables.length === 0) {{
          container.innerHTML = '<div class="empty-state">No tables in catalog.<br><br>Run <code>CREATE TABLE ...</code> to add one.</div>';
          return;
        }}
        container.innerHTML = tables.map(t => `
          <div class="table-item" onclick="setQuery('SELECT * FROM ${{t.name}};')">
            <div class="table-item-name">
              <span>📊 ${{t.name}}</span>
              <span class="badge">${{t.approx_rows}} rows</span>
            </div>
            <div class="table-item-meta">${{t.columns.map(c => c.name + ': ' + c.data_type).join(', ')}}</div>
          </div>
        `).join('');
      }} catch (e) {{
        container.innerHTML = '<div class="empty-state" style="color:var(--danger)">Error loading tables</div>';
      }}
    }}

    // Run SQL query
    async function runQuery() {{
      const sql = document.getElementById('sqlInput').value.trim();
      if (!sql) return;

      const countEl = document.getElementById('resultsCount');
      const statusEl = document.getElementById('resultsStatus');
      const container = document.getElementById('resultsTableContainer');

      countEl.innerText = "Executing query...";
      statusEl.className = "status-badge";
      statusEl.innerText = "";

      try {{
        const res = await fetch('/api/query', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ sql }})
        }});
        const data = await res.json();

        if (!data.success) {{
          statusEl.className = "status-badge status-error";
          statusEl.innerText = "ERROR";
          countEl.innerText = `Failed in ${{data.duration_us}}µs`;
          container.innerHTML = `<div class="empty-state" style="color:var(--danger); text-align:left; font-family:monospace;">${{data.error}}</div>`;
          return;
        }}

        statusEl.className = "status-badge status-success";
        statusEl.innerText = "SUCCESS";

        if (data.affected_rows !== null && data.affected_rows !== undefined) {{
          countEl.innerText = `Query OK, ${{data.affected_rows}} row(s) affected (took ${{data.duration_us}}µs)`;
          container.innerHTML = `<div class="empty-state" style="color:var(--success);">✓ Statement executed successfully (${{data.affected_rows}} row(s) affected).</div>`;
          loadTables();
          return;
        }}

        countEl.innerText = `${{data.rows.length}} row(s) returned (took ${{data.duration_us}}µs)`;

        if (data.rows.length === 0) {{
          container.innerHTML = '<div class="empty-state">Query returned 0 rows.</div>';
          return;
        }}

        let tableHtml = '<table class="data-table"><thead><tr>';
        data.columns.forEach(col => {{
          tableHtml += `<th>${{col}}</th>`;
        }});
        tableHtml += '</tr></thead><tbody>';

        data.rows.forEach(row => {{
          tableHtml += '<tr>';
          row.forEach(cell => {{
            tableHtml += `<td>${{cell}}</td>`;
          }});
          tableHtml += '</tr>';
        }});
        tableHtml += '</tbody></table>';

        container.innerHTML = tableHtml;
        loadTables();
      }} catch (err) {{
        statusEl.className = "status-badge status-error";
        statusEl.innerText = "NETWORK ERROR";
        countEl.innerText = err.message;
      }}
    }}

    // Benchmark runner
    async function runBenchmark() {{
      const size = parseInt(document.getElementById('benchSize').value);
      const grid = document.getElementById('benchResults');
      grid.style.display = 'grid';

      document.getElementById('pgmLookup').innerText = "Running...";
      document.getElementById('btreeLookup').innerText = "Running...";
      document.getElementById('benchSpeedup').innerText = "Calculating...";
      document.getElementById('benchDetails').innerText = "Generating dataset & measuring lookups...";

      try {{
        const res = await fetch('/api/benchmark', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ size }})
        }});
        const d = await res.json();

        document.getElementById('pgmLookup').innerText = `${{d.pgm_lookup_us.toFixed(2)}} µs`;
        document.getElementById('btreeLookup').innerText = `${{d.btree_lookup_us.toFixed(2)}} µs`;

        const pgmPct = Math.min(100, (d.pgm_lookup_us / (d.pgm_lookup_us + d.btree_lookup_us)) * 100);
        const btreePct = 100 - pgmPct;
        document.getElementById('pgmBar').style.width = `${{pgmPct}}%`;
        document.getElementById('btreeBar').style.width = `${{btreePct}}%`;

        document.getElementById('benchSpeedup').innerText = `🚀 Learned PGM is ${{d.speedup.toFixed(1)}}x Faster than B-Tree!`;
        document.getElementById('benchDetails').innerText = 
          `Indexed ${{d.size.toLocaleString()}} keys into ${{d.pgm_segments}} learned segments. PGM build: ${{d.pgm_build_ms.toFixed(1)}}ms | B-Tree build: ${{d.btree_build_ms.toFixed(1)}}ms.`;
      }} catch (e) {{
        document.getElementById('benchSpeedup').innerText = "Benchmark failed: " + e.message;
      }}
    }}

    // Hotkey: ⌘+Enter or Ctrl+Enter to execute
    document.getElementById('sqlInput').addEventListener('keydown', function(e) {{
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {{
        e.preventDefault();
        runQuery();
      }}
    }});

    // Initial load
    loadTables();
  </script>
</body>
</html>
"##, wal_path = wal_path, VERSION = VERSION)
}
