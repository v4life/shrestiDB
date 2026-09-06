//! Converts between the flattened string representations produced by
//! `sql::parser` (literal values, WHERE-clause predicates) and the typed
//! `Value`/`Tuple` rows the executor actually operates on.
//!
//! The predicate evaluator only understands a single `<column> <op>
//! <literal>` comparison (`=`, `!=`/`<>`, `<`, `<=`, `>`, `>=`) — there's no
//! expression tree to walk for `AND`/`OR`/parenthesized conditions (see
//! `sql::parser`'s doc comment: a WHERE clause is still just a rendered
//! string). `evaluate_predicate` returns `None` for anything it can't
//! parse this way, and callers treat "can't evaluate" as "don't filter the
//! row out" — a compound WHERE clause silently has no effect on results
//! rather than than erroring or (worse) dropping rows it can't check.

use crate::execution::catalog::{DataType, TableSchema};
use crate::execution::operators::{Tuple, Value};

/// Parse a literal string (as rendered by `sql::parser`, e.g. `"'Alice'"`,
/// `"30"`, `"true"`) into a typed `Value` per the target column's declared
/// type. Returns `Value::Null` if the literal doesn't parse as that type.
pub fn parse_value(raw: &str, data_type: DataType) -> Value {
    let raw = raw.trim();
    match data_type {
        DataType::Integer | DataType::Timestamp => {
            raw.parse::<i64>().map(Value::Integer).unwrap_or(Value::Null)
        }
        DataType::Float => raw.parse::<f64>().map(Value::Float).unwrap_or(Value::Null),
        DataType::Boolean => raw.parse::<bool>().map(Value::Boolean).unwrap_or(Value::Null),
        DataType::String => Value::String(strip_quotes(raw).to_string()),
    }
}

/// Render a `Value` back to a plain string (for `QueryExecutor::execute`'s
/// `Vec<Vec<String>>` result rows).
pub fn value_to_string(value: &Value) -> String {
    match value {
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s.clone(),
        Value::Boolean(b) => b.to_string(),
        Value::Null => "NULL".to_string(),
    }
}

/// Evaluate a flattened WHERE-clause predicate against one row. `None`
/// means "couldn't evaluate this" (unsupported shape, unknown column, type
/// mismatch) — see module docs for how callers should treat that.
pub fn evaluate_predicate(predicate: &str, schema: &TableSchema, tuple: &Tuple) -> Option<bool> {
    let tokens = tokenize(predicate)?;
    let [column, op, literal] = tokens.try_into().ok()?;

    let idx = schema.columns.iter().position(|c| c.name == column)?;
    let value = tuple.values.get(idx)?;
    let literal_value = parse_value(&literal, schema.columns[idx].data_type);

    compare(value, &op, &literal_value)
}

/// Split on whitespace, respecting single-quoted string literals (so
/// `name = 'John Smith'` tokenizes to 3 tokens, not 4). Returns `None`
/// unless it's exactly 3 tokens — anything else (a compound `AND`/`OR`
/// condition, a bare boolean column, ...) isn't a shape this evaluator
/// understands.
fn tokenize(predicate: &str) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for c in predicate.chars() {
        if c == '\'' {
            in_quotes = !in_quotes;
            current.push(c);
        } else if c.is_whitespace() && !in_quotes {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    if tokens.len() == 3 {
        Some(tokens)
    } else {
        None
    }
}

fn strip_quotes(s: &str) -> &str {
    s.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')).unwrap_or(s)
}

fn compare(value: &Value, op: &str, literal: &Value) -> Option<bool> {
    use std::cmp::Ordering;

    let ord = match (value, literal) {
        (Value::Integer(a), Value::Integer(b)) => a.partial_cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
        (Value::Integer(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
        (Value::Float(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)),
        (Value::String(a), Value::String(b)) => a.partial_cmp(b),
        (Value::Boolean(a), Value::Boolean(b)) => a.partial_cmp(b),
        _ => return None,
    }?;

    Some(match op {
        "=" => ord == Ordering::Equal,
        "!=" | "<>" => ord != Ordering::Equal,
        "<" => ord == Ordering::Less,
        "<=" => ord != Ordering::Greater,
        ">" => ord == Ordering::Greater,
        ">=" => ord != Ordering::Less,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::catalog::Column;

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
            name: "name".to_string(),
            data_type: DataType::String,
            nullable: false,
            primary_key: false,
        });
        schema.add_column(Column {
            id: 3,
            name: "age".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: false,
        });
        schema
    }

    fn row(id: i64, name: &str, age: i64) -> Tuple {
        Tuple {
            values: vec![
                Value::Integer(id),
                Value::String(name.to_string()),
                Value::Integer(age),
            ],
        }
    }

    #[test]
    fn test_parse_value_roundtrip() {
        assert_eq!(parse_value("42", DataType::Integer), Value::Integer(42));
        assert_eq!(parse_value("3.14", DataType::Float), Value::Float(3.14));
        assert_eq!(
            parse_value("'Alice'", DataType::String),
            Value::String("Alice".to_string())
        );
        assert_eq!(parse_value("true", DataType::Boolean), Value::Boolean(true));
    }

    #[test]
    fn test_evaluate_numeric_predicate() {
        let schema = schema();
        assert_eq!(evaluate_predicate("age > 18", &schema, &row(1, "Bob", 30)), Some(true));
        assert_eq!(evaluate_predicate("age > 18", &schema, &row(1, "Kid", 10)), Some(false));
        assert_eq!(evaluate_predicate("age >= 30", &schema, &row(1, "Bob", 30)), Some(true));
    }

    #[test]
    fn test_evaluate_string_predicate_with_spaces() {
        let schema = schema();
        let matching = row(1, "John Smith", 40);
        assert_eq!(
            evaluate_predicate("name = 'John Smith'", &schema, &matching),
            Some(true)
        );
        assert_eq!(
            evaluate_predicate("name = 'John Smith'", &schema, &row(2, "Nobody", 40)),
            Some(false)
        );
    }

    #[test]
    fn test_evaluate_unknown_column_returns_none() {
        let schema = schema();
        assert_eq!(evaluate_predicate("height > 100", &schema, &row(1, "Bob", 30)), None);
    }

    #[test]
    fn test_evaluate_compound_predicate_returns_none() {
        // AND/OR aren't a shape this evaluator understands.
        let schema = schema();
        assert_eq!(
            evaluate_predicate("age > 18 AND id = 1", &schema, &row(1, "Bob", 30)),
            None
        );
    }
}
