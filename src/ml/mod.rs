//! Machine Learning Components
//!
//! ML models for index training, cardinality estimation, and optimization.

pub mod regression;
pub mod neural;
pub mod time_series;
pub mod training;

pub use regression::LinearRegression;
pub use neural::NeuralNetwork;
pub use time_series::{ARModel, ARModel as TimeSeriesPredictor};
pub use training::ModelTrainer;
