//! Query coordinator for distributed execution
//!
//! Coordinates query execution across worker nodes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Worker node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: u32,
    pub host: String,
    pub port: u16,
    pub cpu_cores: usize,
    pub memory_gb: usize,
}

/// Coordinator that manages distributed query execution
pub struct QueryCoordinator {
    pub workers: HashMap<u32, WorkerInfo>,
    pub coordinator_id: u32,
}

impl QueryCoordinator {
    pub fn new(coordinator_id: u32) -> Self {
        QueryCoordinator {
            workers: HashMap::new(),
            coordinator_id,
        }
    }

    /// Register a worker node
    pub fn register_worker(&mut self, worker: WorkerInfo) {
        self.workers.insert(worker.id, worker);
    }

    /// Get list of all workers
    pub fn get_workers(&self) -> Vec<&WorkerInfo> {
        self.workers.values().collect()
    }

    /// Get total compute capacity
    pub fn total_cores(&self) -> usize {
        self.workers.values().map(|w| w.cpu_cores).sum()
    }

    /// Get total available memory
    pub fn total_memory_gb(&self) -> usize {
        self.workers.values().map(|w| w.memory_gb).sum()
    }

    /// Estimate communication cost between workers
    pub fn estimate_comm_cost(&self, _from: u32, _to: u32, _bytes: usize) -> f64 {
        // Simplified: assume 1Gbps network, 1ms latency
        // Cost = latency + bytes / bandwidth
        1.0 + (_bytes as f64 / 1_000_000_000.0) * 1000.0
    }

    /// Select best worker for task execution
    pub fn select_worker_for_task(&self, task_type: &str, _data_size_mb: usize) -> Option<u32> {
        match task_type {
            "scan" => {
                // Choose worker with most free memory
                self.workers
                    .iter()
                    .max_by_key(|(_, w)| w.memory_gb)
                    .map(|(id, _)| *id)
            }
            "aggregate" => {
                // Choose worker with most cores
                self.workers
                    .iter()
                    .max_by_key(|(_, w)| w.cpu_cores)
                    .map(|(id, _)| *id)
            }
            "join" => {
                // Choose worker with balanced resources
                self.workers
                    .iter()
                    .max_by_key(|(_, w)| w.cpu_cores + w.memory_gb / 2)
                    .map(|(id, _)| *id)
            }
            _ => self.workers.keys().next().copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = QueryCoordinator::new(1);
        assert_eq!(coordinator.coordinator_id, 1);
        assert_eq!(coordinator.workers.len(), 0);
    }

    #[test]
    fn test_worker_registration() {
        let mut coordinator = QueryCoordinator::new(1);
        let worker = WorkerInfo {
            id: 1,
            host: "localhost".to_string(),
            port: 5432,
            cpu_cores: 8,
            memory_gb: 16,
        };

        coordinator.register_worker(worker);
        assert_eq!(coordinator.workers.len(), 1);
        assert_eq!(coordinator.total_cores(), 8);
    }

    #[test]
    fn test_worker_selection() {
        let mut coordinator = QueryCoordinator::new(1);

        coordinator.register_worker(WorkerInfo {
            id: 1,
            host: "worker1".to_string(),
            port: 5432,
            cpu_cores: 4,
            memory_gb: 8,
        });

        coordinator.register_worker(WorkerInfo {
            id: 2,
            host: "worker2".to_string(),
            port: 5432,
            cpu_cores: 16,
            memory_gb: 32,
        });

        let worker = coordinator.select_worker_for_task("aggregate", 100);
        assert_eq!(worker, Some(2)); // Should select worker with more cores
    }
}
