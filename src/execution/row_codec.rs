//! Converts between the flattened string representations produced by
//! `sql::parser` (literal values, WHERE-clause predicates) and the typed
//! `Value`/`Tuple` rows the executor actually operates on.
//!
//! `evaluate_predicate` understands a full boolean expression over
//! `<left> <op> <right>` comparisons (`=`, `!=`/`<>`, `<`, `<=`, `>`, `>=`)
//! combined with `AND`/`OR` and parentheses, at standard precedence (`AND`
//! binds tighter than `OR`) — it parses the flattened string itself (see
//! `parse_bool_expr`) rather than walking a real expression tree, since
//! there isn't one to walk (see `sql::parser`'s doc comment: a WHERE
//! clause is still just a rendered string). `right` in each comparison is
//! resolved against the schema first — if it names a real column, this is
//! a column-to-column comparison (what a `JOIN` condition like
//! `"users.id = orders.user_id"` needs, once the executor has merged both
//! sides' columns into one schema); otherwise it's parsed as a literal.
//!
//! `evaluate_predicate` returns `None` for anything it can't parse or
//! evaluate this way (an unrecognized token shape, an unknown column, a
//! type mismatch on some branch), and callers treat "can't evaluate" as
//! "don't filter the row out" — an unparseable WHERE clause silently has
//! no effect on results rather than erroring or (worse) dropping rows it
//! can't check. `execute_join` (in `execution::executor`) is the one
//! exception: it treats an unrecognized join condition as an error rather
//! than silently degrading to an unfiltered cross product — and it only
//! ever hands `split_comparison` a single comparison, never a full
//! boolean expression, since a JOIN's `ON` clause here is scoped that way.
//!
//! `evaluate_predicate` is a convenience wrapper that tokenizes and parses
//! from scratch every call — a per-row hot loop (`Filter`, `JOIN`,
//! UPDATE/DELETE's matching pass) should use `CompiledPredicate` instead
//! to pay that cost once, not once per row.
//!
//! `CompiledAssignment` is the equivalent for an `UPDATE ... SET`
//! clause's right-hand side: a bare literal (the common case) or a single
//! `<column-or-literal> <op> <column-or-literal>` arithmetic expression
//! (`+`, `-`, `*`, `/`), letting `SET balance = balance + amount` read a
//! row's own current value rather than only ever assigning a fixed
//! literal to every matched row.

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

/// An `UPDATE ... SET` assignment's right-hand side, compiled once so it
/// can be evaluated against every matching row without re-parsing the
/// expression string per row — the same reasoning as `CompiledPredicate`,
/// applied to the assignment side of an `UPDATE` instead of its `WHERE`
/// side.
///
/// Understands a bare literal (`SET age = 31`, `SET name = 'Bob'` — the
/// overwhelmingly common case, and the only shape this supported before
/// this type existed), a bare column reference (`SET age = other_age`),
/// or a single `<operand> <op> <operand>` binary arithmetic expression
/// (`+`, `-`, `*`, `/`), each operand either a column reference or a
/// literal. Every column reference is read from the row's values *as they
/// were before this statement's other assignments ran*, so
/// `SET a = b, b = a` swaps the two rather than leaving both equal to the
/// original `b`. Anything shaped differently (nested arithmetic, a
/// function call, ...) falls back to being parsed as one literal typed by
/// the target column, exactly the old behavior: silently `Value::Null`
/// rather than an error, the same soft-failure convention `parse_value`
/// already uses.
///
/// Before this type existed, `execute_update` passed every assignment's
/// raw string straight to `parse_value`, which only understands a plain
/// literal — `SET s_qty = s_qty - 1` parsed as neither a valid integer
/// nor anything else, so `parse_value` silently fell back to
/// `Value::Null`. An UPDATE using this extremely common pattern didn't
/// error; it silently wrote every matched row's column to NULL. See
/// `examples/oltp.rs`'s module doc, which surfaced this while building a
/// benchmark, for how it was worked around before this type existed.
pub struct CompiledAssignment {
    kind: AssignmentKind,
}

enum AssignmentKind {
    Literal(Value),
    ColumnRef(usize),
    Arithmetic { left: ValueOperand, op: char, right: ValueOperand },
}

enum ValueOperand {
    Column(usize),
    Literal(Value),
}

impl CompiledAssignment {
    /// `target_type` is the assigned column's declared type — used only
    /// for the plain-literal fallback path, to match `parse_value`'s
    /// existing behavior exactly (e.g. the literal `5` assigned to a
    /// `Float` column becomes `Value::Float(5.0)`, not `Value::Integer`).
    /// An arithmetic expression's own operands are typed by their own
    /// shape instead (quoted = string, otherwise int/float/bool),
    /// independent of the target column, since the two operands can
    /// legitimately have different natural types (an `Integer` column
    /// plus a `Float` literal, say).
    pub fn compile(expr: &str, schema: &TableSchema, target_type: DataType) -> CompiledAssignment {
        let tokens = tokenize_expr(expr);
        match tokens.as_slice() {
            [left, op_tok, right] => {
                if let Some(op) = arith_op(op_tok) {
                    if let (Some(left), Some(right)) = (resolve_operand(left, schema), resolve_operand(right, schema)) {
                        return CompiledAssignment { kind: AssignmentKind::Arithmetic { left, op, right } };
                    }
                }
            }
            [single] => {
                if let Some(idx) = schema.columns.iter().position(|c| &c.name == single) {
                    return CompiledAssignment { kind: AssignmentKind::ColumnRef(idx) };
                }
            }
            _ => {}
        }
        CompiledAssignment { kind: AssignmentKind::Literal(parse_value(expr, target_type)) }
    }

    /// Evaluate against `tuple`'s *pre-statement* values — see the type
    /// docs on why the caller must pass the row as it was before any of
    /// this UPDATE's other assignments were applied, not a
    /// partway-mutated working copy.
    pub fn eval(&self, tuple: &Tuple) -> Value {
        match &self.kind {
            AssignmentKind::Literal(v) => v.clone(),
            AssignmentKind::ColumnRef(idx) => tuple.values.get(*idx).cloned().unwrap_or(Value::Null),
            AssignmentKind::Arithmetic { left, op, right } => {
                let left = Self::operand_value(left, tuple);
                let right = Self::operand_value(right, tuple);
                arith(&left, *op, &right).unwrap_or(Value::Null)
            }
        }
    }

    fn operand_value(operand: &ValueOperand, tuple: &Tuple) -> Value {
        match operand {
            ValueOperand::Column(idx) => tuple.values.get(*idx).cloned().unwrap_or(Value::Null),
            ValueOperand::Literal(v) => v.clone(),
        }
    }
}

fn resolve_operand(token: &str, schema: &TableSchema) -> Option<ValueOperand> {
    if let Some(idx) = schema.columns.iter().position(|c| c.name == token) {
        return Some(ValueOperand::Column(idx));
    }
    parse_literal_value(token).map(ValueOperand::Literal)
}

fn arith_op(token: &str) -> Option<char> {
    match token {
        "+" => Some('+'),
        "-" => Some('-'),
        "*" => Some('*'),
        "/" => Some('/'),
        _ => None,
    }
}

/// Parse a literal whose type is inferred from its own shape (quoted =
/// string, otherwise the first of int/float/bool that fits) — unlike
/// `parse_value`, which is typed by an externally-known target column.
/// Used for an arithmetic assignment's operands, which don't have a
/// single target column to be typed by.
fn parse_literal_value(token: &str) -> Option<Value> {
    let token = token.trim();
    if let Some(s) = token.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
        return Some(Value::String(s.to_string()));
    }
    if let Ok(i) = token.parse::<i64>() {
        return Some(Value::Integer(i));
    }
    if let Ok(f) = token.parse::<f64>() {
        return Some(Value::Float(f));
    }
    if let Ok(b) = token.parse::<bool>() {
        return Some(Value::Boolean(b));
    }
    None
}

fn arith(left: &Value, op: char, right: &Value) -> Option<Value> {
    use Value::*;
    match (left, right) {
        (Integer(a), Integer(b)) => match op {
            '+' => Some(Integer(a + b)),
            '-' => Some(Integer(a - b)),
            '*' => Some(Integer(a * b)),
            '/' if *b != 0 => Some(Integer(a / b)),
            _ => None,
        },
        (String(a), String(b)) if op == '+' => Some(String(format!("{a}{b}"))),
        (Integer(_) | Float(_), Integer(_) | Float(_)) => {
            let a = as_f64(left)?;
            let b = as_f64(right)?;
            match op {
                '+' => Some(Float(a + b)),
                '-' => Some(Float(a - b)),
                '*' => Some(Float(a * b)),
                '/' if b != 0.0 => Some(Float(a / b)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(i) => Some(*i as f64),
        Value::Float(f) => Some(*f),
        _ => None,
    }
}

/// Evaluate a flattened WHERE/JOIN-condition predicate against one row —
/// see the module docs for exactly what shapes this understands and how
/// callers should treat `None`. Tokenizes and parses `predicate` from
/// scratch every call — fine for the many single-shot call sites (tests,
/// a one-off check), but a caller evaluating the *same* predicate against
/// many rows (a table scan's `Filter`, a `JOIN`'s nested loop, an
/// UPDATE/DELETE's matching pass) should compile it once with
/// `CompiledPredicate::compile` instead and reuse that across every row —
/// see that type's docs for why this matters in practice, not just in
/// principle.
pub fn evaluate_predicate(predicate: &str, schema: &TableSchema, tuple: &Tuple) -> Option<bool> {
    CompiledPredicate::compile(predicate)?.eval(schema, tuple)
}

/// A predicate parsed once, for callers evaluating it against many rows.
/// `evaluate_predicate` re-tokenizes and re-parses its string argument on
/// every call, which is invisible for a one-off check but dominates a hot
/// loop: a nested-loop `JOIN`'s `ON` condition, for instance, is otherwise
/// re-parsed on every single `left_rows * right_rows` pair — the
/// difference is a real, measured 350-500x slowdown relative to SQLite or
/// Postgres running the equivalent join (see `examples/vs_sqlite.rs` /
/// `examples/vs_postgres.rs`), not a theoretical one. Compile once outside
/// the loop, then call `eval` per row.
pub struct CompiledPredicate {
    expr: BoolExpr,
}

impl CompiledPredicate {
    /// `None` for anything `evaluate_predicate` would also treat as
    /// unparseable — see the module docs for exactly what that means for
    /// the caller (typically: don't filter the row out).
    pub fn compile(predicate: &str) -> Option<CompiledPredicate> {
        let tokens = tokenize_expr(predicate);
        let mut parser = ExprParser { tokens: &tokens, pos: 0 };
        let expr = parser.parse_or()?;
        if parser.pos != tokens.len() {
            return None; // trailing tokens the parser couldn't consume
        }
        Some(CompiledPredicate { expr })
    }

    pub fn eval(&self, schema: &TableSchema, tuple: &Tuple) -> Option<bool> {
        eval_bool_expr(&self.expr, schema, tuple)
    }
}

/// Split a flattened `<left> <op> <right>` comparison into its three
/// tokens. `None` for anything else (a compound `AND`/`OR` condition, a
/// bare boolean column, ...) — used where only a single comparison makes
/// sense structurally (a `JOIN` condition, the primary-key index scan),
/// not the general boolean-expression case `evaluate_predicate` handles.
pub fn split_comparison(predicate: &str) -> Option<(String, String, String)> {
    let tokens = tokenize_expr(predicate);
    let [left, op, right]: [String; 3] = tokens.try_into().ok()?;
    Some((left, op, right))
}

// ── Boolean expression grammar over comparisons ───────────────────────────
//
//   or_expr  := and_expr ("OR" and_expr)*
//   and_expr := atom ("AND" atom)*
//   atom     := "(" or_expr ")" | <token> <op> <token>
//
// Standard precedence: AND binds tighter than OR, parens override.

#[derive(Debug, Clone)]
enum BoolExpr {
    Comparison { left: String, op: String, right: String },
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
}

struct ExprParser<'a> {
    tokens: &'a [String],
    pos: usize,
}

impl<'a> ExprParser<'a> {
    fn parse_or(&mut self) -> Option<BoolExpr> {
        let mut left = self.parse_and()?;
        while self.peek_keyword("OR") {
            self.pos += 1;
            let right = self.parse_and()?;
            left = BoolExpr::Or(Box::new(left), Box::new(right));
        }
        Some(left)
    }

    fn parse_and(&mut self) -> Option<BoolExpr> {
        let mut left = self.parse_atom()?;
        while self.peek_keyword("AND") {
            self.pos += 1;
            let right = self.parse_atom()?;
            left = BoolExpr::And(Box::new(left), Box::new(right));
        }
        Some(left)
    }

    fn parse_atom(&mut self) -> Option<BoolExpr> {
        if self.peek() == Some("(") {
            self.pos += 1;
            let inner = self.parse_or()?;
            if self.peek() != Some(")") {
                return None;
            }
            self.pos += 1;
            return Some(inner);
        }

        let left = self.advance()?.clone();
        let op = self.advance()?.clone();
        if !is_comparison_op(&op) {
            return None;
        }
        let right = self.advance()?.clone();
        Some(BoolExpr::Comparison { left, op, right })
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(String::as_str)
    }

    fn peek_keyword(&self, kw: &str) -> bool {
        self.peek().is_some_and(|t| t.eq_ignore_ascii_case(kw))
    }

    fn advance(&mut self) -> Option<&String> {
        let t = self.tokens.get(self.pos);
        self.pos += 1;
        t
    }
}

fn is_comparison_op(s: &str) -> bool {
    matches!(s, "=" | "!=" | "<>" | "<" | "<=" | ">" | ">=")
}

fn eval_bool_expr(expr: &BoolExpr, schema: &TableSchema, tuple: &Tuple) -> Option<bool> {
    match expr {
        BoolExpr::Comparison { left, op, right } => eval_comparison(left, op, right, schema, tuple),
        BoolExpr::And(l, r) => Some(eval_bool_expr(l, schema, tuple)? && eval_bool_expr(r, schema, tuple)?),
        BoolExpr::Or(l, r) => Some(eval_bool_expr(l, schema, tuple)? || eval_bool_expr(r, schema, tuple)?),
    }
}

fn eval_comparison(left: &str, op: &str, right: &str, schema: &TableSchema, tuple: &Tuple) -> Option<bool> {
    let left_idx = schema.columns.iter().position(|c| c.name == left)?;
    let left_value = tuple.values.get(left_idx)?;

    let right_value = match schema.columns.iter().position(|c| c.name == right) {
        Some(right_idx) => tuple.values.get(right_idx)?.clone(),
        None => parse_value(right, schema.columns[left_idx].data_type),
    };

    compare(left_value, op, &right_value)
}

/// Tokenize a predicate: whitespace-separated words, with `'...'` string
/// literals kept intact (so `name = 'John Smith'` yields one token for the
/// literal, not two) and `(`/`)` always split into their own tokens even
/// with no surrounding whitespace.
fn tokenize_expr(predicate: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for c in predicate.chars() {
        match c {
            '\'' => {
                current.push(c);
                in_quotes = !in_quotes;
                if !in_quotes {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ if in_quotes => current.push(c),
            '(' | ')' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(c.to_string());
            }
            c if c.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
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
        schema.add_column(Column {
            id: 4,
            name: "active".to_string(),
            data_type: DataType::Boolean,
            nullable: false,
            primary_key: false,
        });
        schema
    }

    fn row(id: i64, name: &str, age: i64, active: bool) -> Tuple {
        Tuple {
            values: vec![
                Value::Integer(id),
                Value::String(name.to_string()),
                Value::Integer(age),
                Value::Boolean(active),
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
        assert_eq!(
            evaluate_predicate("age > 18", &schema, &row(1, "Bob", 30, true)),
            Some(true)
        );
        assert_eq!(
            evaluate_predicate("age > 18", &schema, &row(1, "Kid", 10, true)),
            Some(false)
        );
        assert_eq!(
            evaluate_predicate("age >= 30", &schema, &row(1, "Bob", 30, true)),
            Some(true)
        );
    }

    #[test]
    fn test_evaluate_string_predicate_with_spaces() {
        let schema = schema();
        let matching = row(1, "John Smith", 40, true);
        assert_eq!(
            evaluate_predicate("name = 'John Smith'", &schema, &matching),
            Some(true)
        );
        assert_eq!(
            evaluate_predicate("name = 'John Smith'", &schema, &row(2, "Nobody", 40, true)),
            Some(false)
        );
    }

    #[test]
    fn test_evaluate_unknown_column_returns_none() {
        let schema = schema();
        assert_eq!(
            evaluate_predicate("height > 100", &schema, &row(1, "Bob", 30, true)),
            None
        );
    }

    #[test]
    fn test_evaluate_column_to_column_comparison() {
        // Needed for JOIN conditions: "id = age" compares two columns of
        // the same row, not a column against a literal.
        let schema = schema();
        assert_eq!(
            evaluate_predicate("id = age", &schema, &row(30, "Bob", 30, true)),
            Some(true)
        );
        assert_eq!(
            evaluate_predicate("id = age", &schema, &row(1, "Bob", 30, true)),
            Some(false)
        );
    }

    #[test]
    fn test_evaluate_and() {
        let schema = schema();
        assert_eq!(
            evaluate_predicate("age > 18 AND active = true", &schema, &row(1, "Bob", 30, true)),
            Some(true)
        );
        assert_eq!(
            evaluate_predicate("age > 18 AND active = true", &schema, &row(1, "Bob", 30, false)),
            Some(false)
        );
        assert_eq!(
            evaluate_predicate("age > 18 AND active = true", &schema, &row(1, "Kid", 10, true)),
            Some(false)
        );
    }

    #[test]
    fn test_evaluate_or() {
        let schema = schema();
        assert_eq!(
            evaluate_predicate("name = 'Bob' OR name = 'Alice'", &schema, &row(1, "Alice", 30, true)),
            Some(true)
        );
        assert_eq!(
            evaluate_predicate("name = 'Bob' OR name = 'Alice'", &schema, &row(1, "Carol", 30, true)),
            Some(false)
        );
    }

    #[test]
    fn test_evaluate_and_binds_tighter_than_or() {
        // "age > 18 AND active = true OR name = 'Kid'" should parse as
        // "(age > 18 AND active = true) OR name = 'Kid'" -- a young,
        // inactive row named "Kid" still matches via the OR branch alone.
        let schema = schema();
        let predicate = "age > 18 AND active = true OR name = 'Kid'";
        assert_eq!(evaluate_predicate(predicate, &schema, &row(1, "Kid", 5, false)), Some(true));
        assert_eq!(
            evaluate_predicate(predicate, &schema, &row(1, "Other", 5, false)),
            Some(false)
        );
        assert_eq!(evaluate_predicate(predicate, &schema, &row(1, "Bob", 30, true)), Some(true));
    }

    #[test]
    fn test_evaluate_parenthesized_or_inside_and() {
        // Without the parens this reduces to (age > 18 AND name = 'Bob')
        // OR name = 'Alice', matching a very different set of rows -- this
        // proves the parens are actually honored, not just parsed and
        // discarded.
        let schema = schema();
        let predicate = "age > 18 AND (name = 'Bob' OR name = 'Alice')";
        assert_eq!(
            evaluate_predicate(predicate, &schema, &row(1, "Alice", 30, true)),
            Some(true)
        );
        assert_eq!(
            evaluate_predicate(predicate, &schema, &row(1, "Alice", 10, true)),
            Some(false) // fails the AND's left side despite matching the OR
        );
        assert_eq!(
            evaluate_predicate(predicate, &schema, &row(1, "Carol", 30, true)),
            Some(false)
        );
    }

    #[test]
    fn test_evaluate_malformed_compound_returns_none() {
        let schema = schema();
        assert_eq!(
            evaluate_predicate("age > 18 AND", &schema, &row(1, "Bob", 30, true)),
            None
        );
        assert_eq!(
            evaluate_predicate("age > 18 AND active = true )", &schema, &row(1, "Bob", 30, true)),
            None
        );
    }

    #[test]
    fn test_compiled_assignment_plain_literal() {
        // The pre-existing behavior -- a bare literal, unchanged by
        // CompiledAssignment's arithmetic support.
        let schema = schema();
        let compiled = CompiledAssignment::compile("31", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(31));
    }

    #[test]
    fn test_compiled_assignment_column_plus_literal() {
        // The case that used to silently produce Value::Null: "age + 1"
        // isn't a valid literal, so the old parse_value-only path failed
        // soft to NULL instead of computing anything.
        let schema = schema();
        let compiled = CompiledAssignment::compile("age + 1", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(31));
    }

    #[test]
    fn test_compiled_assignment_column_minus_literal() {
        let schema = schema();
        let compiled = CompiledAssignment::compile("age - 5", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(25));
    }

    #[test]
    fn test_compiled_assignment_column_times_literal() {
        let schema = schema();
        let compiled = CompiledAssignment::compile("age * 2", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(60));
    }

    #[test]
    fn test_compiled_assignment_column_to_column() {
        // "SET age = id" -- both operands are columns.
        let schema = schema();
        let compiled = CompiledAssignment::compile("id", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(7, "Bob", 30, true)), Value::Integer(7));

        let compiled = CompiledAssignment::compile("id + age", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(7, "Bob", 30, true)), Value::Integer(37));
    }

    #[test]
    fn test_compiled_assignment_division_by_zero_is_null() {
        let schema = schema();
        let compiled = CompiledAssignment::compile("age / 0", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Null);
    }

    #[test]
    fn test_compiled_assignment_unsupported_shape_falls_back_to_null() {
        // More than one operator -- not a shape this understands, so it
        // falls back to parsing the whole string as one literal, which
        // fails and produces Value::Null (parse_value's existing
        // soft-failure convention), not an error.
        let schema = schema();
        let compiled = CompiledAssignment::compile("age + 1 + 1", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Null);
    }

    #[test]
    fn test_compiled_assignment_negative_literal_still_works() {
        // A bare negative number has no operator token at all (no
        // surrounding whitespace to split on), so it must stay on the
        // plain-literal path, not be misread as a two-token expression.
        let schema = schema();
        let compiled = CompiledAssignment::compile("-5", &schema, DataType::Integer);
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(-5));
    }
}
