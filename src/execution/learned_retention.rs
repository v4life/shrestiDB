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
const MIN_SAMPLES_TO_FIT: usize = 30;
const RETRAIN_EVERY: usize = 20;
const MAX_SAMPLES: usize = 200;

/// A deterministic per-row bootstrap threshold in `BOOTSTRAP_INTERVAL ± 2`,
/// instead of every row always sweeping at exactly `BOOTSTRAP_INTERVAL`.
///
/// Without this, the model can never fit at all: while `model` is `None`,
/// `should_prune` (below) triggers a sweep the instant `writes_since_prune`
/// reaches `BOOTSTRAP_INTERVAL`, for every row, every time -- which means
/// the training sample `record_prune_result` feeds `LinearModel::fit` (see
/// `record_prune_result`) always has the exact same x-value. `fit` rejects
/// zero-variance input (`denominator.abs() < 1e-10`), so it returns `None`
/// forever, which keeps `should_prune` in the `None` (bootstrap) branch
/// forever -- a self-reinforcing deadlock verified empirically via
/// `mvcc_store`'s `test_retention_bloat_comparison`: `has_model()` stayed
/// `false` for the whole run under a realistic 20,000-commit skewed
/// workload driven purely through the public API, with `RetentionPredictor`
/// behaving identically to a fixed-interval-8 policy the entire time
/// despite being "wired in and running," not merely fake. Different rows
/// getting different (but each individually stable) bootstrap thresholds
/// gives `fit` real cross-row variance to learn from without ever
/// sacrificing the "always eventually sweeps" safety property (the jitter
/// range is small and centered on the original constant, not a source of
/// unbounded delay).
fn jittered_bootstrap(table_id: u64, row_id: u64) -> u64 {
    let mut h = table_id.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(row_id);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    (BOOTSTRAP_INTERVAL - 2) + (h % 5) // BOOTSTRAP_INTERVAL in {6, 7, 8, 9, 10}
}

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
            None => writes >= jittered_bootstrap(table_id, row_id),
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
        let threshold = jittered_bootstrap(1, 1);
        for _ in 0..threshold - 1 {
            p.record_write(1, 1);
            assert!(!p.should_prune(1, 1));
        }
        p.record_write(1, 1); // the threshold-th write
        assert!(p.should_prune(1, 1));
    }

    #[test]
    fn test_jittered_bootstrap_stays_within_a_small_range_of_the_original_constant() {
        for table_id in 0..5 {
            for row_id in 0..50 {
                let t = jittered_bootstrap(table_id, row_id);
                assert!(
                    (BOOTSTRAP_INTERVAL - 2..=BOOTSTRAP_INTERVAL + 2).contains(&t),
                    "jittered threshold {t} for ({table_id}, {row_id}) escaped the intended small range"
                );
            }
        }
    }

    #[test]
    fn test_jittered_bootstrap_gives_different_rows_different_thresholds() {
        // The whole point of the jitter: without real spread across rows,
        // LinearModel::fit never sees non-zero variance and the model can
        // never fit (see jittered_bootstrap's docs). A handful of distinct
        // row ids must not all land on the same threshold.
        let thresholds: std::collections::HashSet<u64> = (0..20).map(|row_id| jittered_bootstrap(1, row_id)).collect();
        assert!(thresholds.len() > 1, "20 different rows all got the identical bootstrap threshold");
    }

    /// The deadlock this predictor actually had until `jittered_bootstrap`
    /// was added: driven purely through the public API (record_write /
    /// should_prune / record_prune_result, the same three calls
    /// `oltp.rs::commit` makes), with no test hand-feeding varied samples
    /// directly, the model must eventually fit. `test_model_fits_and_
    /// predicts_from_varied_samples` above doesn't catch this -- it calls
    /// `record_prune_result` directly with a hand-picked varied `writes`
    /// sequence, never exercising the real should_prune-driven path where
    /// the bug actually lived.
    #[test]
    fn test_model_eventually_fits_under_natural_driven_usage_not_hand_fed_samples() {
        let p = RetentionPredictor::new();
        // Enough rows, each written enough times, that natural bootstrap
        // sweeps (now at varying per-row thresholds) accumulate the
        // MIN_SAMPLES_TO_FIT samples fit() needs -- nothing here injects a
        // sample directly.
        for row_id in 0..10u64 {
            for _ in 0..400 {
                p.record_write(1, row_id);
                if p.should_prune(1, row_id) {
                    p.record_prune_result(1, row_id, 1);
                }
            }
        }
        assert!(p.has_model(), "model should have fit from natural, varied-per-row bootstrap sweeps alone");
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
        // non-zero variance in x, and enough samples to clear
        // MIN_SAMPLES_TO_FIT (raised from 4 to 30 after
        // test_retention_bloat_comparison showed 4-5 noisy samples was far
        // too few to trust a 1-feature linear fit against real, confounded
        // MVCC telemetry -- see that test and jittered_bootstrap's docs).
        for i in 0..40usize {
            let writes = 2 + (i % 6) * 2; // cycles 2,4,6,8,10,12 for real spread
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
        // Train a model that predicts roughly dead_versions == writes --
        // repeated past MIN_SAMPLES_TO_FIT (see
        // test_model_fits_and_predicts_from_varied_samples's docs on why).
        let cycle = [1.0, 3.0, 5.0, 7.0, 9.0];
        for i in 0..40usize {
            let x = cycle[i % cycle.len()];
            for _ in 0..(x as u64) {
                p.record_write(2, 2);
            }
            p.record_prune_result(2, 2, x as usize); // dead_versions == writes
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
