//! ShrestiDB CLI server
//!
//! Entry point for the ShrestiDB database kernel: a REPL that reads SQL
//! statements from stdin, runs them through `QueryExecutor::execute_sql`
//! against a durable, WAL-backed engine (see `execution::wal` and
//! `execution::oltp`), and prints the results. Data written in one run is
//! still there the next time you start it against the same WAL file.

use std::io::{self, BufRead, Write};

use shrestidb::execution::QueryExecutor;
use shrestidb::VERSION;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> anyhow::Result<()> {
    FmtSubscriber::builder().with_max_level(Level::INFO).init();

    let wal_path = std::env::args().nth(1).unwrap_or_else(|| "shrestidb.wal".to_string());
    info!("Starting ShrestiDB v{VERSION}");
    info!("Using WAL at {wal_path}");

    let executor = QueryExecutor::open(&wal_path)?;

    println!("ShrestiDB {VERSION} — type SQL statements, or 'exit' to quit.");
    println!("Data is persisted to {wal_path}.");

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
        if line.eq_ignore_ascii_case("exit") || line.eq_ignore_ascii_case("quit") {
            break;
        }

        match executor.execute_sql(line) {
            Ok(rows) if rows.is_empty() => println!("OK"),
            Ok(rows) => {
                for row in rows {
                    println!("{}", row.join(" | "));
                }
            }
            Err(e) => println!("Error: {e}"),
        }
    }

    Ok(())
}
