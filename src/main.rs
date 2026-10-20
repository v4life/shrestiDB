//! ShrestiDB CLI & REPL
//!
//! Command Line Interface (CLI) for the ShrestiDB learned database kernel:
//! - Interactive REPL with formatted tabular output, execution timing, and meta-commands
//! - Non-interactive SQL execution via `-e` / `--execute`
//! - Durable storage backed by Write-Ahead Logging (WAL) and MVCC

use std::io::{self, BufRead, Write};
use std::time::Instant;

use shrestidb::execution::QueryExecutor;
use shrestidb::VERSION;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

struct CliOptions {
    wal_path: String,
    execute: Option<String>,
    quiet: bool,
}

fn print_usage() {
    println!(
        r#"ShrestiDB {VERSION} — Learned Database Kernel CLI

USAGE:
    shrestidb [OPTIONS] [WAL_FILE]

ARGS:
    <WAL_FILE>              Path to WAL file (default: shrestidb.wal)

OPTIONS:
    -e, --execute <SQL>     Execute SQL statement(s) non-interactively and exit
    -w, --wal <PATH>        Path to WAL file for persistence
    -q, --quiet             Quiet mode (suppress welcome banner and logs)
    -V, --version           Print version and exit
    -h, --help              Print this help message and exit

REPL COMMANDS:
    \d, SHOW TABLES         List all tables in the database catalog
    \d <table_name>         Describe table schema (columns, types, PKs)
    DESCRIBE <table_name>   Describe table schema
    \status, STATUS         Show database and WAL status
    \h, \?, HELP            Show help and SQL syntax guide
    \q, EXIT, QUIT          Exit the interactive shell
"#
    );
}

fn parse_args() -> Result<Option<CliOptions>, i32> {
    let mut args = std::env::args().skip(1);
    let mut wal_path = "shrestidb.wal".to_string();
    let mut execute = None;
    let mut quiet = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("shrestidb {VERSION}");
                return Ok(None);
            }
            "-q" | "--quiet" => {
                quiet = true;
            }
            "-w" | "--wal" => {
                if let Some(path) = args.next() {
                    wal_path = path;
                } else {
                    eprintln!("Error: missing value for {}", arg);
                    return Err(1);
                }
            }
            s if s.starts_with("--wal=") => {
                wal_path = s.trim_start_matches("--wal=").to_string();
            }
            "-e" | "--execute" => {
                if let Some(sql) = args.next() {
                    execute = Some(sql);
                } else {
                    eprintln!("Error: missing SQL statement for {}", arg);
                    return Err(1);
                }
            }
            s if s.starts_with("--execute=") => {
                execute = Some(s.trim_start_matches("--execute=").to_string());
            }
            s if !s.starts_with('-') => {
                wal_path = s.to_string();
            }
            unknown => {
                eprintln!("Error: unrecognized option '{unknown}'. Run with --help for usage.");
                return Err(1);
            }
        }
    }

    Ok(Some(CliOptions {
        wal_path,
        execute,
        quiet,
    }))
}

/// Print tabular data in an aligned ASCII box
fn print_table(rows: &[Vec<String>]) {
    if rows.is_empty() {
        return;
    }
    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if col_count == 0 {
        return;
    }
    let mut widths = vec![0; col_count];
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if cell.len() > widths[i] {
                widths[i] = cell.len();
            }
        }
    }

    let border: String = widths
        .iter()
        .map(|w| "-".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("+");

    println!("+{border}+");
    for (row_idx, row) in rows.iter().enumerate() {
        let line: String = (0..col_count)
            .map(|i| {
                let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
                format!(" {:<width$} ", cell, width = widths[i])
            })
            .collect::<Vec<_>>()
            .join("|");
        println!("|{line}|");

        // Header separator after first row if there are multiple rows
        if row_idx == 0 && rows.len() > 1 {
            println!("+{border}+");
        }
    }
    println!("+{border}+");
}

fn handle_meta_command(cmd: &str, executor: &QueryExecutor, wal_path: &str) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return false;
    }

    let first = parts[0].to_uppercase();
    match first.as_str() {
        "\\?" | "\\H" | "HELP" => {
            println!(
                r#"ShrestiDB Interactive Shell Commands:
  \d, SHOW TABLES           List all database tables
  \d <table_name>           Display column schema for <table_name>
  DESCRIBE <table_name>     Display column schema for <table_name>
  \status, STATUS           Display database status and statistics
  \h, \?, HELP              Show this help message
  \q, EXIT, QUIT            Exit the shell

Supported SQL:
  CREATE TABLE <name> (<col> <TYPE> [PRIMARY KEY] [NOT NULL], ...);
  INSERT INTO <name> VALUES (...);
  SELECT <cols|*> FROM <name> [JOIN ...] [WHERE ...] [ORDER BY ...] [LIMIT ...];
  UPDATE <name> SET <col> = <expr> [WHERE ...];
  DELETE FROM <name> [WHERE ...];
  CREATE INDEX <idx_name> ON <name> (<col>);
  ANALYZE <name>;
"#
            );
            true
        }
        "\\D" => {
            if parts.len() > 1 {
                describe_table(parts[1], executor);
            } else {
                list_tables(executor);
            }
            true
        }
        "\\DT" => {
            list_tables(executor);
            true
        }
        "SHOW" if parts.len() > 1 && parts[1].eq_ignore_ascii_case("TABLES") => {
            list_tables(executor);
            true
        }
        "DESCRIBE" | "DESC" => {
            if parts.len() > 1 {
                describe_table(parts[1], executor);
            } else {
                println!("Usage: DESCRIBE <table_name>");
            }
            true
        }
        "\\STATUS" | "STATUS" => {
            let catalog = executor.catalog.read();
            println!("--- ShrestiDB System Status ---");
            println!("Kernel Version:  v{VERSION}");
            println!("WAL File:        {wal_path}");
            println!("Catalog Tables:  {}", catalog.tables.len());
            println!("Storage Mode:    Durable MVCC + Learned PGM/RMI Indexing");
            true
        }
        _ => false,
    }
}

fn list_tables(executor: &QueryExecutor) {
    let catalog = executor.catalog.read();
    if catalog.tables.is_empty() {
        println!("No tables found in catalog. Create one with: CREATE TABLE ...");
        return;
    }
    let mut rows = vec![vec![
        "Table ID".to_string(),
        "Table Name".to_string(),
        "Columns".to_string(),
    ]];
    let mut tables: Vec<_> = catalog.tables.values().collect();
    tables.sort_by_key(|t| t.table_id);
    for t in tables {
        rows.push(vec![
            t.table_id.to_string(),
            t.name.clone(),
            t.columns.len().to_string(),
        ]);
    }
    print_table(&rows);
    println!("({} table{})", catalog.tables.len(), if catalog.tables.len() == 1 { "" } else { "s" });
}

fn describe_table(name: &str, executor: &QueryExecutor) {
    let catalog = executor.catalog.read();
    if let Some(table) = catalog.get_table(name) {
        let mut rows = vec![vec![
            "Column".to_string(),
            "Data Type".to_string(),
            "Nullable".to_string(),
            "Primary Key".to_string(),
        ]];
        for col in &table.columns {
            rows.push(vec![
                col.name.clone(),
                format!("{:?}", col.data_type),
                if col.nullable { "YES" } else { "NO" }.to_string(),
                if col.primary_key { "YES" } else { "NO" }.to_string(),
            ]);
        }
        println!("Schema for table '{}':", table.name);
        print_table(&rows);
    } else {
        println!("Table '{}' not found.", name);
    }
}

fn execute_and_print(sql: &str, executor: &QueryExecutor) -> bool {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return true;
    }

    let start = Instant::now();
    match executor.execute_sql(trimmed) {
        Ok(rows) if rows.is_empty() => {
            let elapsed = start.elapsed();
            println!("OK (took {elapsed:.2?})");
            true
        }
        Ok(rows) => {
            let elapsed = start.elapsed();
            print_table(&rows);
            println!(
                "({} row{}, took {elapsed:.2?})",
                rows.len(),
                if rows.len() == 1 { "" } else { "s" }
            );
            true
        }
        Err(e) => {
            eprintln!("Error: {e}");
            false
        }
    }
}

fn main() -> anyhow::Result<()> {
    let opts = match parse_args() {
        Ok(Some(o)) => o,
        Ok(None) => return Ok(()),
        Err(code) => std::process::exit(code),
    };

    if !opts.quiet {
        FmtSubscriber::builder().with_max_level(Level::INFO).init();
        info!("Starting ShrestiDB v{VERSION}");
        info!("Using WAL at {}", opts.wal_path);
    }

    let executor = QueryExecutor::open(&opts.wal_path)?;

    // Non-interactive one-shot execution
    if let Some(sql) = opts.execute {
        let mut all_ok = true;
        for statement in sql.split(';') {
            let trimmed = statement.trim();
            if !trimmed.is_empty() {
                if !execute_and_print(trimmed, &executor) {
                    all_ok = false;
                }
            }
        }
        if !all_ok {
            std::process::exit(1);
        }
        return Ok(());
    }

    // Interactive REPL
    if !opts.quiet {
        println!(
            r#"
   _____ _                     _   _ _____  ____  
  / ____| |                   | | (_)  __ \|  _ \ 
 | (___ | |__  _ __ ___  ___ _| |_ _| |  | | |_) |
  \___ \| '_ \| '__/ _ \/ __|_   _| | |  | |  _ < 
  ____) | | | | | |  __/\__ \ | | | | |__| | |_) |
 |_____/|_| |_|_|  \___||___/ |_| |_|_____/|____/ 
"#
        );
        println!("ShrestiDB v{VERSION} — Learned Database Kernel Shell");
        println!("WAL persistence: {}", opts.wal_path);
        println!("Type SQL statements, '\\?' for help, or '\\q' / 'exit' to quit.\n");
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("shrestidb> ");
        stdout.flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break; // EOF (piped input ended, or Ctrl-D)
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.eq_ignore_ascii_case("exit")
            || line.eq_ignore_ascii_case("quit")
            || line == "\\q"
            || line == "\\Q"
        {
            println!("Bye!");
            break;
        }

        // Check for shell meta commands
        if handle_meta_command(line, &executor, &opts.wal_path) {
            continue;
        }

        // Execute SQL statements (supports multi-statement semicolon separated)
        for stmt in line.split(';') {
            let stmt = stmt.trim();
            if !stmt.is_empty() {
                execute_and_print(stmt, &executor);
            }
        }
    }

    Ok(())
}
