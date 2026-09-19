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

/// Find every aggregate call (`COUNT(*)`, `SUM(age)`, ...) in an arbitrary
/// expression string -- unlike `parse_aggregate`, which only recognizes a
/// column string that is *entirely* one aggregate call, this scans for
/// aggregate calls embedded inside a larger expression (a `HAVING` clause
/// such as `"COUNT(*) > 1"` or `"SUM(amount) > 100 AND dept = 'Eng'"`).
///
/// Each recognized call is replaced in the returned string with a bare
/// placeholder identifier (`__having_agg_0`, `__having_agg_1`, ...) safe
/// for `row_codec::tokenize_expr`/`CompiledPredicate` to treat as an
/// ordinary column reference -- that machinery splits `(`/`)` into their
/// own tokens unconditionally, so `"COUNT(*)"` would otherwise tokenize as
/// six separate tokens instead of the one atomic identifier a comparison
/// needs. The second return value maps each placeholder back to the
/// aggregate that computes its value, in the order they appear.
///
/// Doesn't handle nested aggregate calls (not valid SQL anyway) or an
/// aggregate whose argument itself contains parens; good enough for the
/// single-comparison-per-aggregate shape `HAVING` clauses actually use.
pub fn extract_calls(expr: &str) -> (String, Vec<(String, AggregateFn, Option<String>)>) {
    let chars: Vec<char> = expr.chars().collect();
    let mut rewritten = String::with_capacity(expr.len());
    let mut calls = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            // Look past optional whitespace for the call's `(` without
            // committing to it -- `substitute_placeholders` (a prepared
            // statement's `?`/`$N` substitution) rebuilds a HAVING clause
            // by rejoining tokens with a single space between every one,
            // so a placeholder-bound `HAVING COUNT(*) > ?` arrives here
            // as `"COUNT ( * ) > 1"`, not `"COUNT(*) > 1"` -- without
            // this, that would silently fail to be recognized as an
            // aggregate call at all (found and fixed together with
            // QueryExecutor::execute_prepared's missing HAVING
            // substitution arm, which had the exact same root symptom:
            // a prepared HAVING placeholder always hard-erroring).
            // `parse_aggregate` already tolerates this same whitespace
            // internally (it trims both the function name and the
            // argument), so once the call's *span* is found correctly,
            // recognizing it works unchanged.
            let mut paren_pos = i;
            while paren_pos < chars.len() && chars[paren_pos].is_whitespace() {
                paren_pos += 1;
            }
            if paren_pos < chars.len() && chars[paren_pos] == '(' {
                let mut depth = 1;
                let mut j = paren_pos + 1;
                while j < chars.len() && depth > 0 {
                    match chars[j] {
                        '(' => depth += 1,
                        ')' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                let call: String = chars[start..j].iter().collect();
                match parse_aggregate(&call) {
                    Some((func, arg)) => {
                        let placeholder = format!("__having_agg_{}", calls.len());
                        rewritten.push_str(&placeholder);
                        calls.push((placeholder, func, arg));
                    }
                    None => rewritten.push_str(&call),
                }
                i = j;
                continue;
            }
            rewritten.extend(&chars[start..i]);
            continue;
        }
        rewritten.push(chars[i]);
        i += 1;
    }

    (rewritten, calls)
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

    #[test]
    fn test_extract_calls_rewrites_a_single_aggregate() {
        let (rewritten, calls) = extract_calls("COUNT(*) > 1");
        assert_eq!(rewritten, "__having_agg_0 > 1");
        assert_eq!(calls, vec![("__having_agg_0".to_string(), AggregateFn::Count, None)]);
    }

    #[test]
    fn test_extract_calls_recognizes_a_call_with_whitespace_around_the_parens() {
        // A prepared statement's `?`/`$N` substitution
        // (row_codec::substitute_placeholders) rebuilds an expression by
        // rejoining tokens with a single space between every one, so a
        // placeholder-bound HAVING clause arrives here as
        // "COUNT ( * ) > 1", not "COUNT(*) > 1" -- must still be
        // recognized as the same call.
        let (rewritten, calls) = extract_calls("COUNT ( * ) > 1");
        assert_eq!(rewritten, "__having_agg_0 > 1");
        assert_eq!(calls, vec![("__having_agg_0".to_string(), AggregateFn::Count, None)]);
    }

    #[test]
    fn test_extract_calls_handles_multiple_aggregates_and_a_plain_column() {
        let (rewritten, calls) = extract_calls("SUM(amount) > 100 AND dept = 'Eng' AND COUNT(*) < 10");
        assert_eq!(rewritten, "__having_agg_0 > 100 AND dept = 'Eng' AND __having_agg_1 < 10");
        assert_eq!(
            calls,
            vec![
                ("__having_agg_0".to_string(), AggregateFn::Sum, Some("amount".to_string())),
                ("__having_agg_1".to_string(), AggregateFn::Count, None),
            ]
        );
    }

    #[test]
    fn test_extract_calls_on_expression_with_no_aggregates_is_unchanged() {
        let (rewritten, calls) = extract_calls("dept = 'Eng'");
        assert_eq!(rewritten, "dept = 'Eng'");
        assert!(calls.is_empty());
    }
}
