//! Recognizes and computes simple, GROUP-BY-less aggregate functions
//! (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) over a set of rows.
//!
//! Detected from the flattened SELECT column strings `sql::parser`
//! produces (e.g. `"COUNT(*)"`, `"SUM(age)"`) — there's no expression tree
//! to recognize a function call structurally, so `parse_aggregate` is a
//! small parse of the rendered string, in the same spirit as
//! `row_codec::evaluate_predicate`. Only single-argument aggregates
//! without `GROUP BY` are supported: a `SELECT` that mixes aggregate and
//! plain columns (which needs `GROUP BY`) isn't recognized as an aggregate
//! query at all — see `optimizer::planner::plan`, which only emits an
//! `Aggregate` node when *every* projected column parses as one of these.

use crate::execution::catalog::TableSchema;
use crate::execution::operators::{Tuple, Value};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregateFn {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Parse one SELECT column string as an aggregate call, e.g.
/// `"COUNT(*)"` -> `(Count, None)`, `"SUM(age)"` -> `(Sum, Some("age"))`.
pub fn parse_aggregate(column: &str) -> Option<(AggregateFn, Option<String>)> {
    let column = column.trim();
    let open = column.find('(')?;
    let close = column.rfind(')')?;
    if close < open {
        return None;
    }

    let func = match column[..open].trim().to_uppercase().as_str() {
        "COUNT" => AggregateFn::Count,
        "SUM" => AggregateFn::Sum,
        "AVG" => AggregateFn::Avg,
        "MIN" => AggregateFn::Min,
        "MAX" => AggregateFn::Max,
        _ => return None,
    };

    let arg = column[open + 1..close].trim();
    let arg = if arg.is_empty() || arg == "*" { None } else { Some(arg.to_string()) };
    Some((func, arg))
}

/// Compute one aggregate over `tuples` (no `GROUP BY`: always a single
/// output value). `Value::Null` covers "not applicable" — an unknown
/// column, or a numeric aggregate with nothing numeric to aggregate.
pub fn compute_aggregate(
    func: AggregateFn,
    arg: Option<&str>,
    schema: &TableSchema,
    tuples: &[Tuple],
) -> Value {
    if func == AggregateFn::Count {
        return match arg {
            None => Value::Integer(tuples.len() as i64),
            Some(col) => match schema.columns.iter().position(|c| c.name == col) {
                Some(idx) => Value::Integer(
                    tuples
                        .iter()
                        .filter(|t| !matches!(t.values.get(idx), None | Some(Value::Null)))
                        .count() as i64,
                ),
                None => Value::Null,
            },
        };
    }

    let idx = match arg.and_then(|col| schema.columns.iter().position(|c| c.name == col)) {
        Some(idx) => idx,
        None => return Value::Null,
    };
    let numbers: Vec<f64> = tuples
        .iter()
        .filter_map(|t| match t.values.get(idx) {
            Some(Value::Integer(i)) => Some(*i as f64),
            Some(Value::Float(f)) => Some(*f),
            _ => None,
        })
        .collect();
    if numbers.is_empty() {
        return Value::Null;
    }

    match func {
        AggregateFn::Sum => Value::Float(numbers.iter().sum()),
        AggregateFn::Avg => Value::Float(numbers.iter().sum::<f64>() / numbers.len() as f64),
        AggregateFn::Min => Value::Float(numbers.iter().cloned().fold(f64::INFINITY, f64::min)),
        AggregateFn::Max => Value::Float(numbers.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
        AggregateFn::Count => unreachable!("handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::catalog::{Column, DataType};

    fn schema() -> TableSchema {
        let mut schema = TableSchema::new(1, "users".to_string());
        schema.add_column(Column {
            id: 1,
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: true,
        });
        schema.add_column(Column {
            id: 2,
            name: "age".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: false,
        });
        schema
    }

    fn row(id: i64, age: i64) -> Tuple {
        Tuple {
            values: vec![Value::Integer(id), Value::Integer(age)],
        }
    }

    #[test]
    fn test_parse_aggregate_variants() {
        assert_eq!(parse_aggregate("COUNT(*)"), Some((AggregateFn::Count, None)));
        assert_eq!(
            parse_aggregate("sum(age)"),
            Some((AggregateFn::Sum, Some("age".to_string())))
        );
        assert_eq!(parse_aggregate("age"), None);
        assert_eq!(parse_aggregate("id + 1"), None);
    }

    #[test]
    fn test_compute_count_star() {
        let schema = schema();
        let tuples = vec![row(1, 10), row(2, 20), row(3, 30)];
        assert_eq!(
            compute_aggregate(AggregateFn::Count, None, &schema, &tuples),
            Value::Integer(3)
        );
    }

    #[test]
    fn test_compute_sum_and_avg() {
        let schema = schema();
        let tuples = vec![row(1, 10), row(2, 20), row(3, 30)];
        assert_eq!(
            compute_aggregate(AggregateFn::Sum, Some("age"), &schema, &tuples),
            Value::Float(60.0)
        );
        assert_eq!(
            compute_aggregate(AggregateFn::Avg, Some("age"), &schema, &tuples),
            Value::Float(20.0)
        );
    }

    #[test]
    fn test_compute_min_max() {
        let schema = schema();
        let tuples = vec![row(1, 10), row(2, 20), row(3, 30)];
        assert_eq!(
            compute_aggregate(AggregateFn::Min, Some("age"), &schema, &tuples),
            Value::Float(10.0)
        );
        assert_eq!(
            compute_aggregate(AggregateFn::Max, Some("age"), &schema, &tuples),
            Value::Float(30.0)
        );
    }

    #[test]
    fn test_compute_on_empty_input_is_null_except_count() {
        let schema = schema();
        assert_eq!(compute_aggregate(AggregateFn::Count, None, &schema, &[]), Value::Integer(0));
        assert_eq!(compute_aggregate(AggregateFn::Sum, Some("age"), &schema, &[]), Value::Null);
    }

    #[test]
    fn test_compute_unknown_column_is_null() {
        let schema = schema();
        let tuples = vec![row(1, 10)];
        assert_eq!(
            compute_aggregate(AggregateFn::Sum, Some("height"), &schema, &tuples),
            Value::Null
        );
    }
}
