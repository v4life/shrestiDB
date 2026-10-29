//! Query operators for vectorized execution

use serde::{Deserialize, Serialize};

/// Operator trait
pub trait Operator: Send + Sync {
    fn next(&mut self) -> Option<Tuple>;
    fn close(&mut self);
}

/// Tuple representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tuple {
    pub values: Vec<Value>,
}

/// Value types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Value {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    /// Fixed-dimension floating-point vector for embeddings / similarity search.
    Vector(Vec<f64>),
    Null,
}

/// Table scan operator
pub struct ScanOperator {
    pub table_id: u32,
    pub tuples: Vec<Tuple>,
    pub position: usize,
}

impl ScanOperator {
    pub fn new(table_id: u32) -> Self {
        ScanOperator {
            table_id,
            tuples: Vec::new(),
            position: 0,
        }
    }

    pub fn add_tuple(&mut self, tuple: Tuple) {
        self.tuples.push(tuple);
    }
}

impl Operator for ScanOperator {
    fn next(&mut self) -> Option<Tuple> {
        if self.position < self.tuples.len() {
            let tuple = self.tuples[self.position].clone();
            self.position += 1;
            Some(tuple)
        } else {
            None
        }
    }

    fn close(&mut self) {
        self.tuples.clear();
    }
}

/// Filter operator
pub struct FilterOperator {
    pub child: Box<dyn Operator>,
    pub predicate: Box<dyn Fn(&Tuple) -> bool + Send + Sync>,
}

impl Operator for FilterOperator {
    fn next(&mut self) -> Option<Tuple> {
        while let Some(tuple) = self.child.next() {
            if (self.predicate)(&tuple) {
                return Some(tuple);
            }
        }
        None
    }

    fn close(&mut self) {
        self.child.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_operator() {
        let mut scan = ScanOperator::new(1);
        scan.add_tuple(Tuple {
            values: vec![Value::Integer(1)],
        });

        let result = scan.next();
        assert!(result.is_some());
    }
}
