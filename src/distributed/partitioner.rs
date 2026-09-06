//! Data partitioning strategies for distributed execution

use serde::{Deserialize, Serialize};

/// Partitioning strategy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Hash-based partitioning
    Hash,
    /// Range-based partitioning
    Range,
    /// Round-robin partitioning
    RoundRobin,
    /// Random partitioning
    Random,
}

/// Partition metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    pub id: u32,
    pub start_key: f64,
    pub end_key: f64,
    pub row_count: usize,
    pub size_bytes: usize,
}

/// Data partitioner for distributed storage
pub struct DataPartitioner {
    pub num_partitions: usize,
    pub strategy: PartitionStrategy,
    pub partitions: Vec<Partition>,
}

impl DataPartitioner {
    pub fn new(num_partitions: usize, strategy: PartitionStrategy) -> Self {
        DataPartitioner {
            num_partitions,
            strategy,
            partitions: Vec::new(),
        }
    }

    /// Compute partition ID for a key
    pub fn get_partition(&self, key: f64) -> u32 {
        match self.strategy {
            PartitionStrategy::Hash => {
                // Hash-based: use key hash mod num_partitions
                let hash = key.to_bits() as u32;
                hash % self.num_partitions as u32
            }
            PartitionStrategy::Range => {
                // Range-based: find partition by key range
                for partition in &self.partitions {
                    if key >= partition.start_key && key < partition.end_key {
                        return partition.id;
                    }
                }
                0 // Default
            }
            PartitionStrategy::RoundRobin => {
                // Round-robin: cycle through partitions
                (key as u32) % self.num_partitions as u32
            }
            PartitionStrategy::Random => {
                // Random: pseudo-random assignment
                ((key * 1.337) as u32) % self.num_partitions as u32
            }
        }
    }

    /// Add a partition
    pub fn add_partition(&mut self, partition: Partition) {
        self.partitions.push(partition);
    }

    /// Get total data size
    pub fn total_size_bytes(&self) -> usize {
        self.partitions.iter().map(|p| p.size_bytes).sum()
    }

    /// Get total row count
    pub fn total_rows(&self) -> usize {
        self.partitions.iter().map(|p| p.row_count).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_partitioning() {
        let partitioner = DataPartitioner::new(4, PartitionStrategy::Hash);
        let p1 = partitioner.get_partition(100.0);
        let p2 = partitioner.get_partition(200.0);
        assert!(p1 < 4);
        assert!(p2 < 4);
    }

    #[test]
    fn test_range_partitioning() {
        let mut partitioner = DataPartitioner::new(2, PartitionStrategy::Range);
        partitioner.add_partition(Partition {
            id: 0,
            start_key: 0.0,
            end_key: 50.0,
            row_count: 1000,
            size_bytes: 100_000,
        });

        let p = partitioner.get_partition(25.0);
        assert_eq!(p, 0);
    }
}
