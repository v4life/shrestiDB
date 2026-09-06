//! Learned version-retention scheduler.
//!
//! `MVCCTable`'s version chains never garbage-collect on their own — dead
//! versions (`end_ts` already set) just accumulate forever. Pruning them is
//! always *safe* as long as we never remove a version some active
//! transaction's snapshot could still need (see `min_active_snapshot` in
//! `oltp.rs` and `VersionChain::prune` in `mvcc_store.rs`) — that
//! correctness gate holds unconditionally and does not depend on anything
//! below.
//!
//! What's still a judgment call is *when* to bother sweeping a row: sweep
//! on every single write and cold, rarely-written rows pay a scan for no
//! benefit; sweep too rarely and a hot row's chain grows unbounded again
//! (the exact bug this module exists to avoid). Rather than picking one
//! fixed interval, this fits a small online linear model — the same
//! `LinearModel` used by the PGM/RMI indexes — on real telemetry gathered
//! from the engine's own operation: how many writes a row has taken since
//! its chain was last swept, versus how many dead versions that sweep
//! actually found. It predicts expected dead-version buildup and triggers
//! a sweep once that crosses a threshold, instead of sweeping on a fixed
//! schedule that's either wasteful or too slow for every row's actual
//! write rate.
//!
//! Before enough samples exist to fit a model, it falls back to a fixed
//! bootstrap interval — never to "never sweep."

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use crate::index::models::LinearModel;

const BOOTSTRAP_INTERVAL: u64 = 8;
const PRUNE_THRESHOLD: f64 = 3.0;
const MIN_SAMPLES_TO_FIT: usize = 4;
const RETRAIN_EVERY: usize = 5;
const MAX_SAMPLES: usize = 200;

#[derive(Debug, Default, Clone, Copy)]
struct RowStats {
    writes_since_prune: u64,
}

pub struct RetentionPredictor {
    stats: Mutex<HashMap<(u64, u64), RowStats>>,
    samples: Mutex<VecDeque<(f64, f64)>>,
    model: Mutex<Option<LinearModel>>,
}

impl RetentionPredictor {
    pub fn new() -> Self {
        RetentionPredictor {
            stats: Mutex::new(HashMap::new()),
            samples: Mutex::new(VecDeque::new()),
            model: Mutex::new(None),
        }
    }

    /// Record that a row was written (a new version pushed onto its chain).
    pub fn record_write(&self, table_id: u64, row_id: u64) {
        self.stats
            .lock()
            .unwrap()
            .entry((table_id, row_id))
            .or_default()
            .writes_since_prune += 1;
    }

    /// Predict whether this row's chain is worth sweeping right now.
    pub fn should_prune(&self, table_id: u64, row_id: u64) -> bool {
        let writes = self
            .stats
            .lock()
            .unwrap()
            .get(&(table_id, row_id))
            .map(|s| s.writes_since_prune)
            .unwrap_or(0);

        if writes == 0 {
            return false;
        }

        match &*self.model.lock().unwrap() {
            Some(m) => m.predict(writes as f64) >= PRUNE_THRESHOLD,
            None => writes >= BOOTSTRAP_INTERVAL,
        }
    }

    /// Record the outcome of an actual sweep: how many dead versions it
    /// found, given how many writes had accumulated since the row's last
    /// sweep. Feeds the online model and resets the row's counter.
    pub fn record_prune_result(&self, table_id: u64, row_id: u64, dead_versions_found: usize) {
        let writes = {
            let mut stats = self.stats.lock().unwrap();
            let entry = stats.entry((table_id, row_id)).or_default();
            let writes = entry.writes_since_prune;
            entry.writes_since_prune = 0;
            writes
        };

        let mut samples = self.samples.lock().unwrap();
        samples.push_back((writes as f64, dead_versions_found as f64));
        while samples.len() > MAX_SAMPLES {
            samples.pop_front();
        }
        let len = samples.len();
        let training_set: Vec<(f64, f64)> = samples.iter().copied().collect();
        drop(samples);

        if len >= MIN_SAMPLES_TO_FIT && len % RETRAIN_EVERY == 0 {
            let (xs, ys): (Vec<f64>, Vec<f64>) = training_set.into_iter().unzip();
            if let Some(fitted) = LinearModel::fit(&xs, &ys) {
                *self.model.lock().unwrap() = Some(fitted);
            }
        }
    }

    /// Whether the model has fit yet (for diagnostics/tests).
    pub fn has_model(&self) -> bool {
        self.model.lock().unwrap().is_some()
    }
}

impl Default for RetentionPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstraps_before_model_exists() {
        let p = RetentionPredictor::new();
        for _ in 0..7 {
            p.record_write(1, 1);
            assert!(!p.should_prune(1, 1));
        }
        p.record_write(1, 1); // 8th write
        assert!(p.should_prune(1, 1));
    }

    #[test]
    fn test_no_writes_never_prunes() {
        let p = RetentionPredictor::new();
        assert!(!p.should_prune(1, 1));
    }

    #[test]
    fn test_model_fits_and_predicts_from_varied_samples() {
        let p = RetentionPredictor::new();
        // Feed a clear linear relationship: dead_versions ~= writes, with
        // enough spread in `writes` for LinearModel::fit's OLS to have
        // non-zero variance in x.
        for writes in [2usize, 4, 6, 8, 10, 12] {
            for _ in 0..writes {
                p.record_write(1, 1);
            }
            p.record_prune_result(1, 1, writes); // reuse `writes` as the label too
        }
        assert!(p.has_model());
    }

    #[test]
    fn test_high_predicted_buildup_triggers_prune() {
        let p = RetentionPredictor::new();
        // Train a model that predicts roughly dead_versions == writes.
        let xs = [1.0, 3.0, 5.0, 7.0, 9.0];
        let ys = [1.0, 3.0, 5.0, 7.0, 9.0];
        for (x, y) in xs.iter().zip(ys.iter()) {
            for _ in 0..(*x as u64) {
                p.record_write(2, 2);
            }
            p.record_prune_result(2, 2, *y as usize);
        }
        assert!(p.has_model());

        // Row 3 has only ever written twice: predicted buildup (~2) is
        // below the threshold, so it shouldn't trigger a sweep yet.
        p.record_write(3, 3);
        p.record_write(3, 3);
        assert!(!p.should_prune(3, 3));

        // Row 4 has written well past the threshold's crossing point.
        for _ in 0..6 {
            p.record_write(4, 4);
        }
        assert!(p.should_prune(4, 4));
    }
}
