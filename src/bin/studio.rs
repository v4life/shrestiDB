//! ShrestiDB Web Studio binary
//!
//! Launches the local Web Studio & Query GUI for ShrestiDB.

use std::sync::Arc;
use shrestidb::execution::QueryExecutor;
use shrestidb::studio::StudioServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut wal_path = "shrestidb.wal".to_string();
    let mut port = 8080u16;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-p" | "--port" => {
                if let Some(p) = args.next() {
                    port = p.parse().unwrap_or(8080);
                }
            }
            "-w" | "--wal" => {
                if let Some(w) = args.next() {
                    wal_path = w;
                }
            }
            s if s.starts_with("--port=") => {
                port = s.trim_start_matches("--port=").parse().unwrap_or(8080);
            }
            s if s.starts_with("--wal=") => {
                wal_path = s.trim_start_matches("--wal=").to_string();
            }
            s if !s.starts_with('-') => {
                wal_path = s.to_string();
            }
            _ => {}
        }
    }

    println!("Initializing ShrestiDB engine...");
    let executor = Arc::new(QueryExecutor::open(&wal_path)?);
    let studio = Arc::new(StudioServer::new(executor, wal_path));

    studio.run(port).await?;

    Ok(())
}
