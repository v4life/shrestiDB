//! OLTP workload example
//!
//! Simulates online transaction processing workload.

use learned_db_kernel::execution::transaction::TransactionManager;
use learned_db_kernel::execution::catalog::{Catalog, TableSchema, Column, DataType};
use std::time::Instant;

fn main() {
    println!("=== OLTP Workload Simulation ===");
    println!();

    // Initialize transaction manager
    let tm = TransactionManager::new();
    println!("Transaction Manager initialized");
    println!();

    // Create schema
    let mut catalog = Catalog::new();
    let mut users_schema = TableSchema::new(1, "users".to_string());
    
    users_schema.add_column(Column {
        id: 1,
        name: "id".to_string(),
        data_type: DataType::Integer,
        nullable: false,
        primary_key: true,
    });
    
    users_schema.add_column(Column {
        id: 2,
        name: "name".to_string(),
        data_type: DataType::String,
        nullable: false,
        primary_key: false,
    });
    
    users_schema.add_column(Column {
        id: 3,
        name: "balance".to_string(),
        data_type: DataType::Float,
        nullable: false,
        primary_key: false,
    });

    catalog.register_table(users_schema);
    println!("Schema created: users table with 3 columns");
    println!();

    // Simulate transactions
    println!("Simulating OLTP workload...");
    let start = Instant::now();
    let num_transactions = 10000;

    for i in 0..num_transactions {
        let tx = tm.begin();
        
        // Simulate work (read/write operations)
        // In a real system, this would involve actual data access
        if i % 2 == 0 {
            tm.commit(tx);
        } else {
            tm.abort(tx);
        }
    }

    let elapsed = start.elapsed();
    let tps = num_transactions as f64 / elapsed.as_secs_f64();

    println!("Transactions executed: {}", num_transactions);
    println!("Total time: {:?}", elapsed);
    println!("Throughput: {:.0} TPS", tps);
    println!();

    println!("=== OLTP Summary ===");
    println!("Workload type: Mixed read/write");
    println!("Transactions: {}", num_transactions);
    println!("Throughput: {:.0} transactions/sec", tps);
}
