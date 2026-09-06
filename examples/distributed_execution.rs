//! Example distributed query execution
//!
//! Demonstrates distributed query processing across multiple worker nodes.

use shresti::distributed::coordinator::{QueryCoordinator, WorkerInfo};
use shresti::distributed::partitioner::{DataPartitioner, PartitionStrategy};
use shresti::distributed::distributed_plan::{
    DistributedPlan, DistributedStage,
};
use std::time::Instant;

fn main() {
    println!("=== Distributed Query Execution Example ===");
    println!();

    // Setup coordinator
    let mut coordinator = QueryCoordinator::new(1);
    println!("Query Coordinator initialized");
    println!();

    // Register worker nodes
    println!("Registering worker nodes...");
    for i in 1..=4 {
        let worker = WorkerInfo {
            id: i,
            host: format!("worker{}", i),
            port: 5430 + i as u16,
            cpu_cores: (8 * i) as usize,
            memory_gb: (16 * i) as usize,
        };
        coordinator.register_worker(worker);
        println!("  Worker {}: {} cores, {} GB RAM", i, 8 * i, 16 * i);
    }
    println!();

    println!("Cluster Resources:");
    println!("  Total workers: {}", coordinator.get_workers().len());
    println!("  Total CPU cores: {}", coordinator.total_cores());
    println!("  Total memory: {} GB", coordinator.total_memory_gb());
    println!();

    // Create data partitioner
    let mut partitioner = DataPartitioner::new(4, PartitionStrategy::Range);
    for i in 0..4 {
        use shresti::distributed::partitioner::Partition;
        let partition = Partition {
            id: i,
            start_key: (i as f64 * 250_000.0),
            end_key: ((i + 1) as f64 * 250_000.0),
            row_count: 250_000,
            size_bytes: 250_000 * 100, // Assume 100 bytes per row
        };
        partitioner.add_partition(partition);
    }
    println!("Data Distribution:");
    println!("  Total rows: {}", partitioner.total_rows());
    println!("  Total size: {} MB", partitioner.total_size_bytes() / 1_000_000);
    println!("  Partitions: {}", partitioner.num_partitions);
    println!();

    // Create distributed query plan
    let mut plan = DistributedPlan::new("q1".to_string(), 4);
    plan.add_stage(DistributedStage::PartitionedScan {
        table_id: 1,
        partitions: vec![0, 1, 2, 3],
    });
    plan.add_stage(DistributedStage::LocalFilter {
        predicate: "amount > 1000".to_string(),
    });
    plan.add_stage(DistributedStage::Shuffle {
        key: "customer_id".to_string(),
        target_partitions: vec![0, 1, 2, 3],
    });
    plan.add_stage(DistributedStage::Aggregate {
        group_by: vec!["customer_id".to_string()],
        aggregates: vec!["sum(amount)".to_string()],
    });

    println!("Distributed Query Plan:");
    println!("  Query ID: {}", plan.query_id);
    println!("  Stages: {}", plan.stages.len());
    println!("  Parallelism: {}", plan.parallelism());
    for (i, stage) in plan.stages.iter().enumerate() {
        println!("    Stage {}: {:?}", i + 1, stage);
    }
    println!();

    // Simulate execution
    println!("Simulating distributed execution...");
    let start = Instant::now();
    simulate_execution(&coordinator, &plan);
    let elapsed = start.elapsed();

    println!();
    println!("Execution Summary:");
    println!("  Total time: {:?}", elapsed);
    println!("  Estimated speedup: {:.1}x", 4.0); // 4-way parallelism
    println!();

    // Show worker selection
    println!("Task Assignment:");
    for task_type in &["scan", "aggregate", "join"] {
        if let Some(worker_id) = coordinator.select_worker_for_task(task_type, 1000) {
            let worker = &coordinator.workers[&worker_id];
            println!(
                "  {}: Worker {} ({} cores, {} GB)",
                task_type, worker_id, worker.cpu_cores, worker.memory_gb
            );
        }
    }
    println!();

    // Network communication estimate
    let shuffle_bytes = plan.estimate_shuffle_bytes(100, partitioner.total_rows());
    println!("Network Communication:");
    println!("  Estimated shuffle bytes: {} MB", shuffle_bytes / 1_000_000);
    println!("  Assuming 1Gbps network: {:.1} ms latency", shuffle_bytes as f64 / 1_000_000_000.0 * 1000.0);
}

fn simulate_execution(coordinator: &QueryCoordinator, plan: &DistributedPlan) {
    let workers = coordinator.get_workers();
    let num_stages = plan.stages.len();

    for (stage_num, stage) in plan.stages.iter().enumerate() {
        println!();
        println!("  Stage {}/{}:", stage_num + 1, num_stages);
        println!("    Type: {:?}", stage);

        // Simulate parallel execution
        let mut total_time = 0u64;
        for (i, worker) in workers.iter().enumerate() {
            // Simulate execution time based on worker capacity
            let time_ms = (1000 / (worker.cpu_cores as u64).max(1)) + (i as u64 * 50);
            total_time = total_time.max(time_ms);
            println!("    Worker {}: ~{}ms", worker.id, time_ms);
        }
        println!("    Stage time: ~{}ms", total_time);
    }
}
