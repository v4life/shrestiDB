//! Error types for the learned database kernel

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Page not found: {0}")]
    PageNotFound(u64),

    #[error("Buffer pool full")]
    BufferPoolFull,

    #[error("Index error: {0}")]
    IndexError(String),

    #[error("Query planning error: {0}")]
    PlanningError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Transaction error: {0}")]
    TransactionError(String),

    #[error("Type error: {0}")]
    TypeError(String),

    #[error("Parsing error: {0}")]
    ParseError(String),

    #[error("Binding error: {0}")]
    BindingError(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Key not found")]
    KeyNotFound,

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Model training error: {0}")]
    ModelError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, DatabaseError>;
