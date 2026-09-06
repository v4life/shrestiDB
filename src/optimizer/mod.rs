//! Query Optimization Layer
//!
//! ML-driven query optimization with learned cardinality estimation and cost modeling.

pub mod cardinality;
pub mod cost_model;
pub mod join_reorder;
pub mod planner;
pub mod statistics;

pub use cardinality::LearnedCardinalityEstimator;
pub use cost_model::CostModel;
pub use join_reorder::JoinOrderer;
pub use planner::QueryPlanner;
pub use statistics::{ColumnStatistics, StatisticsCollector, TableStats};
