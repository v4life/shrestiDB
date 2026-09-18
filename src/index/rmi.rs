//! Recursive Model Index (RMI)
//!
//! Multi-stage learned index combining hierarchical prediction models.

use crate::index::models::LinearModel;
use serde::{Deserialize, Serialize};

/// Recursive Model Index stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMIStage {
    pub models: Vec<LinearModel>,
    pub num_models: usize,
}

impl RMIStage {
    pub fn new(models: Vec<LinearModel>) -> Self {
        let num_models = models.len();
        RMIStage { models, num_models }
    }

    /// Route key through stage to next model index
    pub fn route(&self, key: f64) -> usize {
        if self.num_models == 0 {
            return 0;
        }

        let first_model = &self.models[0];
        let prediction = first_model.predict(key);

        let model_idx = (prediction as usize).clamp(0, self.num_models - 1);
        model_idx
    }
}

/// Full Recursive Model Index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMIIndex {
    pub stages: Vec<RMIStage>,
    pub keys: Vec<f64>,
    pub positions: Vec<usize>,
}

impl RMIIndex {
    pub fn new(stages: Vec<RMIStage>, keys: Vec<f64>, positions: Vec<usize>) -> Self {
        RMIIndex {
            stages,
            keys,
            positions,
        }
    }

    /// Search for a key and return predicted position
    pub fn search(&self, key: f64) -> usize {
        let mut current_prediction = key;

        for stage in &self.stages {
            let model_idx = stage.route(current_prediction);
            if model_idx < stage.models.len() {
                current_prediction = stage.models[model_idx].predict(key);
            }
        }

        (current_prediction as usize).clamp(0, self.positions.len().saturating_sub(1))
    }

    /// Find exact position with bounded error search
    pub fn find_exact(&self, key: f64, error_bound: usize) -> Option<usize> {
        let predicted_pos = self.search(key);

        let start = predicted_pos.saturating_sub(error_bound);
        let end = (predicted_pos + error_bound).min(self.keys.len());

        self.keys[start..end]
            .binary_search_by(|k| k.partial_cmp(&key).unwrap())
            .ok()
            .map(|offset| start + offset)
    }

    /// Real allocated heap memory -- see `index::pgm::PGMIndex::heap_bytes`
    /// for the accounting method this mirrors. Unlike `PGMIndex`, this
    /// keeps a full `positions: Vec<usize>` alongside `keys` (one `usize`
    /// per key, not per segment) -- a real, structural reason this
    /// construction of RMI won't show the same per-key memory advantage
    /// PGM's implicit array-index encoding gets, regardless of how
    /// accurate either one's predictions are.
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of::<RMIIndex>()
            + self.stages.capacity() * std::mem::size_of::<RMIStage>()
            + self.stages.iter().map(|s| s.models.capacity() * std::mem::size_of::<LinearModel>()).sum::<usize>()
            + self.keys.capacity() * std::mem::size_of::<f64>()
            + self.positions.capacity() * std::mem::size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmi_stage_routing() {
        let model = LinearModel::new(2.0, 0.0);
        let stage = RMIStage::new(vec![model]);

        let route = stage.route(5.0);
        assert_eq!(route, 0);
    }

    #[test]
    fn test_rmi_search() {
        let keys = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let positions = vec![0, 1, 2, 3, 4];
        let model = LinearModel::new(1.0, 0.0);
        let stage = RMIStage::new(vec![model]);

        let rmi = RMIIndex::new(vec![stage], keys, positions);
        let predicted = rmi.search(3.0);
        assert!(predicted < 5);
    }

    #[test]
    fn test_rmi_heap_bytes_grows_with_key_count() {
        let small = RMIIndex::new(vec![RMIStage::new(vec![LinearModel::new(1.0, 0.0)])], vec![1.0], vec![0]);
        let n = 100_000;
        let keys: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let positions: Vec<usize> = (0..n).collect();
        let big = RMIIndex::new(vec![RMIStage::new(vec![LinearModel::new(1.0, 0.0)])], keys, positions);
        assert!(big.heap_bytes() > small.heap_bytes());
    }
}
