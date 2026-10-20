//! ShrestiDB Server Daemon
//!
//! Background server entry point for the ShrestiDB kernel.

use shrestidb::execution::QueryExecutor;
use shrestidb::VERSION;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> anyhow::Result<()> {
    FmtSubscriber::builder().with_max_level(Level::INFO).init();

    let wal_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "shrestidb.wal".to_string());

    info!("Starting ShrestiDB server v{VERSION}");
    info!("Using WAL at {wal_path}");

    let _executor = QueryExecutor::open(&wal_path)?;
    info!("ShrestiDB engine initialized successfully");

    Ok(())
}
