//! SQL data types

use serde::{Deserialize, Serialize};

/// SQL data type
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Integer,
    Bigint,
    Float,
    Double,
    String(usize), // String with max length
    Boolean,
    Timestamp,
    Decimal(usize, usize), // (precision, scale)
    /// Fixed-dimension floating-point vector for embeddings / similarity
    /// search — `VECTOR(384)` in SQL.
    Vector(usize),
}

impl DataType {
    /// Get the size in bytes of this type
    pub fn size_bytes(&self) -> Option<usize> {
        match self {
            DataType::Integer => Some(4),
            DataType::Bigint => Some(8),
            DataType::Float => Some(4),
            DataType::Double => Some(8),
            DataType::String(len) => Some(*len),
            DataType::Boolean => Some(1),
            DataType::Timestamp => Some(8),
            DataType::Decimal(_, _) => None, // Variable size
            DataType::Vector(dims) => Some(*dims * 8), // f64 per dimension
        }
    }

    /// Check if this type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::Integer
                | DataType::Bigint
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_, _)
        )
    }
}
