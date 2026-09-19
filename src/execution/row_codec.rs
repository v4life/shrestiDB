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

use crate::error::{DatabaseError, Result};
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

/// Like `parse_value`, but returns `None` instead of silently defaulting
/// to `Value::Null` when `token` doesn't actually look like a literal of
/// `data_type` -- the real fix for two real, verified bugs sharing the
/// same root cause: `CompiledAssignment::compile`'s literal fallback
/// (`UPDATE ... SET age = ABS(age)` silently wrote every matched row's
/// `age` to `NULL`; `SET name = UPPER(name)` silently wrote the literal,
/// unparsed text `"UPPER(name)"` into the column), and `INSERT`'s own
/// value parsing (`order_insert_row`, `execution::executor`) — `INSERT
/// INTO items (id, price) VALUES (1, ABS(-9.99))` silently inserted
/// `price = NULL`, verified before this existed. The explicit `NULL`
/// keyword (case-insensitive, matching SQL) is recognized first, so a
/// genuine `SET col = NULL` / `INSERT ... VALUES (..., NULL, ...)` keeps
/// working -- it's the one case `parse_value`'s ambiguous
/// "unparseable becomes Null" behavior happened to get right, kept
/// intentionally here rather than accidentally.
///
/// Stricter than `parse_value` for `String` specifically: a `String`
/// column only accepts a properly single-quoted token as a literal, not
/// arbitrary unquoted text -- `parse_value` accepted any raw text handed
/// to it (there's nothing else a `String` column's value *could* fail to
/// be), which is exactly how `UPPER(name)`'s unparsed text ended up
/// stored verbatim as a string value instead of being rejected.
pub fn parse_literal_checked(token: &str, data_type: DataType) -> Option<Value> {
    let token = token.trim();
    if token.eq_ignore_ascii_case("NULL") {
        return Some(Value::Null);
    }
    match data_type {
        DataType::Integer | DataType::Timestamp => token.parse::<i64>().ok().map(Value::Integer),
        DataType::Float => token.parse::<f64>().ok().map(Value::Float),
        DataType::Boolean => token.parse::<bool>().ok().map(Value::Boolean),
        DataType::String => token
            .strip_prefix('\'')
            .and_then(|s| s.strip_suffix('\''))
            .map(|s| Value::String(s.to_string())),
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

/// Render a bound `Value` back into the literal-string shape a parsed
/// token would have — quoted for `String`, bare otherwise — so a
/// `QueryExecutor::execute_prepared` parameter can be substituted into an
/// already-flattened statement and fed through the exact same
/// `parse_value`/`CompiledPredicate`/`CompiledAssignment` paths a literal
/// written directly in SQL would go through. The inverse of `parse_value`,
/// not of `value_to_string` (which is for display and drops the quoting a
/// re-parse would need).
///
/// Known limitation, inherited from the tokenizer this whole module
/// already relies on (see `tokenize_expr`): a quoted string's contents
/// aren't unescaped, so a bound string containing an apostrophe isn't
/// safely representable here — attempting it produces a token whose
/// embedded `'` prematurely closes the literal when re-tokenized.
/// Pre-existing, not introduced by prepared statements: hand-written SQL
/// in this engine already can't express a literal apostrophe either.
pub fn literal_repr(value: &Value) -> String {
    match value {
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => format!("'{s}'"),
        Value::Boolean(b) => b.to_string(),
        Value::Null => "NULL".to_string(),
    }
}

// ── Prepared-statement placeholders ────────────────────────────────────────
//
// `?` (positional, bound to `params` in the order every scanned/substituted
// field is visited) or `$N` (explicit 1-based index, bound directly to
// `params[N-1]` regardless of visit order). One statement must use a single
// style throughout -- `scan_placeholders` errors the moment it sees both,
// rather than guessing which the caller meant.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaceholderKind {
    Positional,
    Indexed(usize),
}

/// `None` for anything that isn't a bare, unquoted `?` or `$N` token.
/// Never true for a quoted literal that happens to contain these
/// characters — a quoted token from `tokenize_expr` always carries its
/// quote marks, so `'?'` and `?` are never confused.
fn placeholder_kind(token: &str) -> Option<PlaceholderKind> {
    if token == "?" {
        return Some(PlaceholderKind::Positional);
    }
    let digits = token.strip_prefix('$')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<usize>().ok().filter(|n| *n >= 1).map(PlaceholderKind::Indexed)
}

/// Scan one already-flattened field (a WHERE clause, a JOIN condition, an
/// UPDATE assignment's right-hand side, a single INSERT value, ...) for
/// placeholder tokens, folding them into `positional_count`/`max_indexed`
/// — running totals a caller threads across every field of one statement.
/// `Err` the moment both a `?` and a `$N` have been seen anywhere in that
/// statement.
pub fn scan_placeholders(field: &str, positional_count: &mut usize, max_indexed: &mut usize) -> Result<()> {
    for token in tokenize_expr(field) {
        match placeholder_kind(&token) {
            Some(PlaceholderKind::Positional) => *positional_count += 1,
            Some(PlaceholderKind::Indexed(n)) => *max_indexed = (*max_indexed).max(n),
            None => {}
        }
        if *positional_count > 0 && *max_indexed > 0 {
            return Err(DatabaseError::ExecutionError(
                "cannot mix ? and $N placeholders in the same prepared statement".to_string(),
            ));
        }
    }
    Ok(())
}

/// Rewrite `field`, replacing each placeholder token with the literal
/// representation (`literal_repr`) of its bound value. `next_positional`
/// is shared, mutable state across every field substituted for one
/// `execute_prepared` call, advanced only by `?` tokens — so positional
/// placeholders bind to `params` in the order fields are visited, which
/// must match the order `scan_placeholders` visited them in during
/// `prepare` (both walks are driven by the same per-statement-type field
/// list in `QueryExecutor::scan_statement_placeholders`, so this holds by
/// construction, not by convention the two have to independently honor).
pub fn substitute_placeholders(field: &str, params: &[Value], next_positional: &mut usize) -> Result<String> {
    let tokens = tokenize_expr(field);
    let mut out = Vec::with_capacity(tokens.len());
    for token in tokens {
        out.push(substitute_token(&token, params, next_positional)?);
    }
    Ok(out.join(" "))
}

/// The single-token version of `substitute_placeholders` — for a caller
/// that already knows a field is exactly one token (most usefully,
/// `split_comparison`'s `(left, op, right)`, see
/// `QueryExecutor::execute_prepared`'s `Filter` case), so there's nothing
/// left to tokenize: `token` is either a placeholder (substituted) or a
/// literal/column name already (passed through unchanged). Skipping
/// `tokenize_expr` entirely here is the point — it's the same
/// char-by-char scan `substitute_placeholders` just did on this same
/// text, redundant work `execute_prepared` used to pay for on every
/// single prepared-statement execution, not just once at `prepare` time.
pub fn substitute_token(token: &str, params: &[Value], next_positional: &mut usize) -> Result<String> {
    match placeholder_kind(token) {
        Some(PlaceholderKind::Positional) => {
            let value = params.get(*next_positional).ok_or_else(|| {
                DatabaseError::ExecutionError(format!(
                    "prepared statement expects at least {} parameter(s), got {}",
                    *next_positional + 1,
                    params.len()
                ))
            })?;
            *next_positional += 1;
            Ok(literal_repr(value))
        }
        Some(PlaceholderKind::Indexed(n)) => {
            let value = params.get(n - 1).ok_or_else(|| {
                DatabaseError::ExecutionError(format!(
                    "prepared statement references ${n} but only {} parameter(s) were given",
                    params.len()
                ))
            })?;
            Ok(literal_repr(value))
        }
        None => Ok(token.to_string()),
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
/// function call, ...) is a compile error (`compile` returns `None`) --
/// an earlier version of this fallback matched `parse_value`'s own
/// soft-failure convention and silently treated it as a literal `NULL`
/// instead: `UPDATE users SET age = ABS(age)` wrote every matched row's
/// `age` to `NULL`, and — worse — `UPDATE users SET name = UPPER(name)`
/// wrote the literal, unparsed text `"UPPER(name)"` into every matched
/// row's `name` column, both verified empirically before this was fixed.
/// The same "unsupported, not silently wrong" fix already applied to
/// `WHERE`/`HAVING`/`JOIN ON` elsewhere in this module's history.
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
    /// `target_type` is the assigned column's declared type — used for
    /// the plain-literal fallback path (e.g. the literal `5` assigned to
    /// a `Float` column becomes `Value::Float(5.0)`, not `Value::Integer`).
    /// An arithmetic expression's own operands are typed by their own
    /// shape instead (quoted = string, otherwise int/float/bool),
    /// independent of the target column, since the two operands can
    /// legitimately have different natural types (an `Integer` column
    /// plus a `Float` literal, say).
    ///
    /// `None` means `expr` isn't any of the shapes this understands --
    /// the caller's job to turn into a real error, not to substitute a
    /// value for (see this type's docs on why the old fallback of
    /// silently writing `Value::Null`, or worse, the assignment's own
    /// unparsed text, was a real, serious bug).
    pub fn compile(expr: &str, schema: &TableSchema, target_type: DataType) -> Option<CompiledAssignment> {
        let tokens = tokenize_expr(expr);
        match tokens.as_slice() {
            [left, op_tok, right] => {
                if let Some(op) = arith_op(op_tok) {
                    if let (Some(left), Some(right)) = (resolve_operand(left, schema), resolve_operand(right, schema)) {
                        return Some(CompiledAssignment { kind: AssignmentKind::Arithmetic { left, op, right } });
                    }
                }
            }
            [single] => {
                if let Some(idx) = schema.columns.iter().position(|c| &c.name == single) {
                    return Some(CompiledAssignment { kind: AssignmentKind::ColumnRef(idx) });
                }
                return parse_literal_checked(single, target_type)
                    .map(|v| CompiledAssignment { kind: AssignmentKind::Literal(v) });
            }
            _ => {}
        }
        None
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
    CompiledPredicate::compile(predicate, schema)?.eval(tuple)
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
///
/// `compile` also resolves every comparison's column name(s) to a fixed
/// index into `schema` once, at compile time, rather than `eval`
/// re-scanning `schema.columns` with a string comparison on every call —
/// the same reasoning applied one level deeper. `eval` itself does no
/// string work at all: index lookups and, where a literal was involved,
/// an already-parsed `Value`.
pub struct CompiledPredicate {
    expr: BoolExpr,
}

impl CompiledPredicate {
    /// `None` for anything `evaluate_predicate` would also treat as
    /// unparseable — see the module docs for exactly what that means for
    /// the caller (typically: don't filter the row out). This now
    /// includes a column name that doesn't resolve against `schema`: the
    /// old runtime behavior returned `None` for every row when that
    /// happened anyway (the failure doesn't depend on the row), so moving
    /// it to compile time changes nothing observable.
    pub fn compile(predicate: &str, schema: &TableSchema) -> Option<CompiledPredicate> {
        let tokens = tokenize_expr(predicate);
        let mut parser = ExprParser { tokens: &tokens, pos: 0, schema };
        let expr = parser.parse_or()?;
        if parser.pos != tokens.len() {
            return None; // trailing tokens the parser couldn't consume
        }
        Some(CompiledPredicate { expr })
    }

    /// `Some(false)` covers both a real `False` and SQL's `NULL`-driven
    /// `Unknown` (see `Tri::is_true`) — both mean "exclude this row",
    /// same as every real database. Only a genuine structural failure
    /// (an unresolvable column, an unrecognized operator) is `None`,
    /// which callers fall back to their own default for — see this
    /// type's docs.
    pub fn eval(&self, tuple: &Tuple) -> Option<bool> {
        eval_bool_expr(&self.expr, tuple).map(Tri::is_true)
    }

    /// Same as `eval`, but for a `JOIN`'s nested loop specifically:
    /// evaluates against a still-split `(left, right)` pair instead of an
    /// already-merged `Tuple`, so a pair that doesn't match never has to
    /// pay for building one — cloning and concatenating both sides'
    /// values just to immediately discard the result is real, wasted cost
    /// on every one of a join's `left_rows * right_rows` pairs that isn't
    /// a match (which is most of them: 897,000 of 900,000 in
    /// `examples/tpc_h.rs`'s join). `left_len` is `left`'s column count —
    /// the same split point `QueryExecutor::merge_schemas` used when this
    /// predicate was compiled, so a resolved index at or past it refers
    /// to `right`, not `left`.
    pub fn eval_split(&self, left: &Tuple, left_len: usize, right: &Tuple) -> Option<bool> {
        eval_bool_expr_split(&self.expr, left, left_len, right).map(Tri::is_true)
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

/// Parse a flattened `ORDER BY <expr> [ASC|DESC], ...` clause (see
/// `sql::parser::SelectStatement::order_by`'s docs — `sqlparser`'s own
/// `Display` renders the whole clause, `"ORDER BY"` keyword included, so
/// that prefix is stripped here rather than at every call site) into
/// `(column, ascending)` pairs, in priority order — the first pair breaks
/// ties using the second, and so on, standard multi-column `ORDER BY`
/// semantics.
///
/// `NULLS FIRST`/`NULLS LAST` isn't recognized: a `NULLS` token (or
/// `FIRST`/`LAST` alongside it) is simply not `DESC`, so it's silently
/// treated as ascending's absence of a direction keyword rather than
/// honored — this engine's own fixed `NULL`-ordering convention
/// (`compare_for_sort`, `NULL` always sorts before every non-`NULL`
/// value regardless of `ASC`/`DESC`) applies uniformly instead. A
/// documented scope limit, not a silent correctness gap: nothing here
/// claims to support a clause it doesn't parse.
pub fn parse_order_by(raw: &str) -> Vec<(String, bool)> {
    let raw = raw.strip_prefix("ORDER BY").unwrap_or(raw).trim();
    raw.split(',')
        .filter_map(|part| {
            let tokens: Vec<&str> = part.split_whitespace().collect();
            let (name, rest) = tokens.split_first()?;
            let ascending = !rest.iter().any(|t| t.eq_ignore_ascii_case("DESC"));
            Some(((*name).to_string(), ascending))
        })
        .collect()
}

/// Order two values for `ORDER BY` — same-type comparisons behave
/// naturally (numeric, lexicographic for strings, `false < true` for
/// booleans); a numeric comparison across `Integer`/`Float` widens to
/// `f64`, matching `compare`'s own cross-numeric-type rule. `NULL`
/// always sorts before every non-`NULL` value (SQLite's convention,
/// picked since this codebase's own comparisons throughout this session
/// have used SQLite as the reference engine) — applied the same way
/// regardless of `ASC`/`DESC` on that key, which the caller reverses the
/// *whole* ordering for, `NULL` placement included, matching how real
/// databases treat `NULL`s under `DESC` by default. A comparison across
/// otherwise-incomparable types (e.g. `String` vs `Boolean` — shouldn't
/// happen for a real column, which has one fixed `DataType`) has no
/// defined order, so it's treated as equal: a stable sort keeps those
/// rows in whatever relative order they arrived in, rather than
/// panicking or guessing.
pub fn compare_for_sort(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::Integer(x), Value::Float(y)) => (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::Float(x), Value::Integer(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal),
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Boolean(x), Value::Boolean(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
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
    /// `left_idx` is resolved once at parse time (see `ExprParser::parse_atom`),
    /// not re-looked-up on every `eval`.
    Comparison { left_idx: usize, op: String, right: ComparisonOperand },
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
}

/// A comparison's right-hand side, resolved once at parse time: either
/// another column (by index) or an already-parsed literal `Value` — never
/// a raw string needing another schema lookup or `parse_value` call
/// during `eval`.
#[derive(Debug, Clone)]
enum ComparisonOperand {
    Column(usize),
    Literal(Value),
}

struct ExprParser<'a> {
    tokens: &'a [String],
    pos: usize,
    schema: &'a TableSchema,
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

        let left_idx = self.schema.columns.iter().position(|c| c.name == left)?;
        let right = match self.schema.columns.iter().position(|c| c.name == right) {
            Some(right_idx) => ComparisonOperand::Column(right_idx),
            None => ComparisonOperand::Literal(parse_value(&right, self.schema.columns[left_idx].data_type)),
        };
        Some(BoolExpr::Comparison { left_idx, op, right })
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

fn eval_bool_expr(expr: &BoolExpr, tuple: &Tuple) -> Option<Tri> {
    match expr {
        BoolExpr::Comparison { left_idx, op, right } => eval_comparison(*left_idx, op, right, tuple),
        BoolExpr::And(l, r) => Some(eval_bool_expr(l, tuple)?.and(eval_bool_expr(r, tuple)?)),
        BoolExpr::Or(l, r) => Some(eval_bool_expr(l, tuple)?.or(eval_bool_expr(r, tuple)?)),
    }
}

/// No string work at all: `left_idx`/`right` were already resolved at
/// parse time (see `ExprParser::parse_atom`), so this is index lookups
/// and a reference comparison, not a schema scan.
fn eval_comparison(left_idx: usize, op: &str, right: &ComparisonOperand, tuple: &Tuple) -> Option<Tri> {
    let left_value = tuple.values.get(left_idx)?;
    let right_value = match right {
        ComparisonOperand::Column(idx) => tuple.values.get(*idx)?,
        ComparisonOperand::Literal(v) => v,
    };

    compare(left_value, op, right_value)
}

// ── Split (unmerged left/right) evaluation, for CompiledPredicate::eval_split ──

fn eval_bool_expr_split(expr: &BoolExpr, left: &Tuple, left_len: usize, right: &Tuple) -> Option<Tri> {
    match expr {
        BoolExpr::Comparison { left_idx, op, right: rhs } => {
            eval_comparison_split(*left_idx, op, rhs, left, left_len, right)
        }
        BoolExpr::And(l, r) => Some(
            eval_bool_expr_split(l, left, left_len, right)?.and(eval_bool_expr_split(r, left, left_len, right)?),
        ),
        BoolExpr::Or(l, r) => Some(
            eval_bool_expr_split(l, left, left_len, right)?.or(eval_bool_expr_split(r, left, left_len, right)?),
        ),
    }
}

/// A resolved index at or past `left_len` refers to `right`'s columns —
/// see `CompiledPredicate::eval_split`'s docs on why that split point is
/// safe to rely on here.
fn split_value<'a>(idx: usize, left: &'a Tuple, left_len: usize, right: &'a Tuple) -> Option<&'a Value> {
    if idx < left_len {
        left.values.get(idx)
    } else {
        right.values.get(idx - left_len)
    }
}

fn eval_comparison_split(
    left_idx: usize,
    op: &str,
    right: &ComparisonOperand,
    left: &Tuple,
    left_len: usize,
    right_tuple: &Tuple,
) -> Option<Tri> {
    let left_value = split_value(left_idx, left, left_len, right_tuple)?;
    let right_value = match right {
        ComparisonOperand::Column(idx) => split_value(*idx, left, left_len, right_tuple)?,
        ComparisonOperand::Literal(v) => v,
    };

    compare(left_value, op, right_value)
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

/// SQL's three-valued comparison/boolean logic — `Unknown` is what a
/// comparison against `NULL` produces (SQL `NULL`'s own meaning is "this
/// value is unknown", so any comparison touching it is equally unknown,
/// never simply `True` or `False`), kept distinct from `compare`/
/// `eval_bool_expr`'s `Option::None`, which means something entirely
/// different: "this engine couldn't structurally evaluate the predicate
/// at all" (an unresolvable column, an unrecognized operator). Only
/// `None` gets the existing "give up, don't filter the row out"
/// fail-open treatment (see `CompiledPredicate`'s docs) — `Unknown` is a
/// real, well-defined SQL answer, and `WHERE`/`JOIN ON` both exclude a
/// row on `Unknown` exactly like `False` (see `Tri::is_true`), never
/// `True`'s "give up and keep it" treatment. Conflating the two used to
/// mean `WHERE fk > 100` with `fk` actually `NULL` **included** that row
/// — the opposite of every real database's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tri {
    True,
    False,
    Unknown,
}

impl Tri {
    /// SQL's `AND` truth table: `False` short-circuits regardless of the
    /// other operand (even `Unknown`); otherwise `Unknown` unless both
    /// sides are `True`.
    fn and(self, other: Tri) -> Tri {
        match (self, other) {
            (Tri::False, _) | (_, Tri::False) => Tri::False,
            (Tri::True, Tri::True) => Tri::True,
            _ => Tri::Unknown,
        }
    }

    /// SQL's `OR` truth table: `True` short-circuits regardless of the
    /// other operand; otherwise `Unknown` unless both sides are `False`.
    fn or(self, other: Tri) -> Tri {
        match (self, other) {
            (Tri::True, _) | (_, Tri::True) => Tri::True,
            (Tri::False, Tri::False) => Tri::False,
            _ => Tri::Unknown,
        }
    }

    /// What `WHERE`/`JOIN ON` actually keep a row for — only a real
    /// `True`. Both `False` and `Unknown` mean "exclude", the same
    /// outcome real SQL gives a `NULL`-involving predicate.
    fn is_true(self) -> bool {
        matches!(self, Tri::True)
    }
}

fn compare(value: &Value, op: &str, literal: &Value) -> Option<Tri> {
    use std::cmp::Ordering;

    // NULL compared any way is SQL UNKNOWN, not "can't evaluate" -- still
    // validate the operator so an unrecognized one stays a real
    // structural failure (None), same as the non-NULL path below.
    if matches!(value, Value::Null) || matches!(literal, Value::Null) {
        return match op {
            "=" | "!=" | "<>" | "<" | "<=" | ">" | ">=" => Some(Tri::Unknown),
            _ => None,
        };
    }

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
        "=" => if ord == Ordering::Equal { Tri::True } else { Tri::False },
        "!=" | "<>" => if ord != Ordering::Equal { Tri::True } else { Tri::False },
        "<" => if ord == Ordering::Less { Tri::True } else { Tri::False },
        "<=" => if ord != Ordering::Greater { Tri::True } else { Tri::False },
        ">" => if ord == Ordering::Greater { Tri::True } else { Tri::False },
        ">=" => if ord != Ordering::Less { Tri::True } else { Tri::False },
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

    fn row_with_null_age(id: i64, name: &str, active: bool) -> Tuple {
        Tuple { values: vec![Value::Integer(id), Value::String(name.to_string()), Value::Null, Value::Boolean(active)] }
    }

    #[test]
    fn test_comparison_against_null_is_excluded_not_included() {
        // SQL: `NULL > 18` is UNKNOWN, and WHERE/JOIN both exclude
        // UNKNOWN exactly like FALSE -- Some(false), never Some(true) and
        // never None (a structural failure, a different thing entirely:
        // see Tri's docs). An earlier version conflated the two, so a
        // NULL-valued row was *kept* by a WHERE clause instead of
        // excluded -- the opposite of every real database's answer.
        let schema = schema();
        let bob = row_with_null_age(1, "Bob", true);
        assert_eq!(evaluate_predicate("age > 18", &schema, &bob), Some(false));
        assert_eq!(evaluate_predicate("age <= 18", &schema, &bob), Some(false));
        assert_eq!(evaluate_predicate("age = 30", &schema, &bob), Some(false));
        // `!=`/`<>` don't "catch" NULL either -- also UNKNOWN, not TRUE.
        assert_eq!(evaluate_predicate("age != 30", &schema, &bob), Some(false));
    }

    #[test]
    fn test_or_short_circuits_true_even_with_an_unknown_operand() {
        // The real three-valued-logic test: `TRUE OR UNKNOWN` must be
        // TRUE, not swallowed into "false" the way a naive "any NULL
        // means exclude the whole predicate" implementation would get
        // wrong. `id = 1` is TRUE; `age > 1000` is UNKNOWN (age is NULL).
        let schema = schema();
        let bob = row_with_null_age(1, "Bob", true);
        assert_eq!(evaluate_predicate("id = 1 OR age > 1000", &schema, &bob), Some(true));
    }

    #[test]
    fn test_and_with_an_unknown_operand_is_excluded_regardless_of_the_other_side() {
        // TRUE AND UNKNOWN = UNKNOWN (excluded); FALSE AND UNKNOWN =
        // FALSE (also excluded, via short-circuit) -- both observably
        // Some(false) at this boundary, which is correct either way.
        let schema = schema();
        let bob = row_with_null_age(1, "Bob", true);
        assert_eq!(evaluate_predicate("id = 1 AND age > 1000", &schema, &bob), Some(false));
        assert_eq!(evaluate_predicate("id = 99 AND age > 1000", &schema, &bob), Some(false));
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
        let compiled = CompiledAssignment::compile("31", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(31));
    }

    #[test]
    fn test_compiled_assignment_column_plus_literal() {
        // The case that used to silently produce Value::Null: "age + 1"
        // isn't a valid literal, so the old parse_value-only path failed
        // soft to NULL instead of computing anything.
        let schema = schema();
        let compiled = CompiledAssignment::compile("age + 1", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(31));
    }

    #[test]
    fn test_compiled_assignment_column_minus_literal() {
        let schema = schema();
        let compiled = CompiledAssignment::compile("age - 5", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(25));
    }

    #[test]
    fn test_compiled_assignment_column_times_literal() {
        let schema = schema();
        let compiled = CompiledAssignment::compile("age * 2", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(60));
    }

    #[test]
    fn test_compiled_assignment_column_to_column() {
        // "SET age = id" -- both operands are columns.
        let schema = schema();
        let compiled = CompiledAssignment::compile("id", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(7, "Bob", 30, true)), Value::Integer(7));

        let compiled = CompiledAssignment::compile("id + age", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(7, "Bob", 30, true)), Value::Integer(37));
    }

    #[test]
    fn test_compiled_assignment_division_by_zero_is_null() {
        // Compiles fine ("age / 0" is a valid arithmetic shape) -- the
        // Null comes from evaluating it, a legitimate per-value result
        // distinct from a compile-time rejection.
        let schema = schema();
        let compiled = CompiledAssignment::compile("age / 0", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Null);
    }

    #[test]
    fn test_compiled_assignment_unsupported_shape_is_a_compile_error() {
        // More than one operator -- not a shape this understands. An
        // earlier version fell back to parsing the whole string as one
        // literal, which failed and silently produced Value::Null
        // instead of an error -- the real bug this type exists to fix
        // (see its doc comment). Now a compile-time None instead.
        let schema = schema();
        assert!(CompiledAssignment::compile("age + 1 + 1", &schema, DataType::Integer).is_none());
    }

    #[test]
    fn test_compiled_assignment_function_call_is_a_compile_error_not_null_or_garbage() {
        // The two real, verified-before-the-fix bugs this type's doc
        // comment documents: a function call assigned to a numeric
        // column used to silently write NULL, and to a string column
        // used to silently write the function call's own unparsed text
        // as if it were a literal string. Both are now compile errors.
        let schema = schema();
        assert!(CompiledAssignment::compile("ABS(age)", &schema, DataType::Integer).is_none());

        let mut string_schema = TableSchema::new(1, "users".to_string());
        string_schema.add_column(Column { id: 1, name: "name".to_string(), data_type: DataType::String, nullable: false, primary_key: false });
        assert!(CompiledAssignment::compile("UPPER(name)", &string_schema, DataType::String).is_none());
    }

    #[test]
    fn test_compiled_assignment_explicit_null_literal_still_works() {
        // The one case parse_value's old ambiguous "unparseable becomes
        // Null" behavior happened to get right -- kept working
        // intentionally (see parse_literal_checked's docs), not by
        // accident, now that everything else unparseable is a hard error.
        let schema = schema();
        let compiled = CompiledAssignment::compile("NULL", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Null);
    }

    #[test]
    fn test_compiled_assignment_unquoted_text_on_string_column_is_a_compile_error() {
        // parse_value used to accept ANY raw text for a String column,
        // quoted or not -- exactly how an unsupported expression's own
        // text (e.g. "UPPER(name)") ended up stored verbatim. A String
        // literal must be properly quoted now.
        let mut string_schema = TableSchema::new(1, "users".to_string());
        string_schema.add_column(Column { id: 1, name: "name".to_string(), data_type: DataType::String, nullable: false, primary_key: false });
        assert!(CompiledAssignment::compile("Bob", &string_schema, DataType::String).is_none());
        let compiled = CompiledAssignment::compile("'Bob'", &string_schema, DataType::String).unwrap();
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::String("Bob".to_string()));
    }

    #[test]
    fn test_compiled_assignment_negative_literal_still_works() {
        // A bare negative number has no operator token at all (no
        // surrounding whitespace to split on), so it must stay on the
        // plain-literal path, not be misread as a two-token expression.
        let schema = schema();
        let compiled = CompiledAssignment::compile("-5", &schema, DataType::Integer).unwrap();
        assert_eq!(compiled.eval(&row(1, "Bob", 30, true)), Value::Integer(-5));
    }

    #[test]
    fn test_literal_repr_roundtrips_through_parse_value() {
        assert_eq!(parse_value(&literal_repr(&Value::Integer(42)), DataType::Integer), Value::Integer(42));
        assert_eq!(parse_value(&literal_repr(&Value::Float(3.5)), DataType::Float), Value::Float(3.5));
        assert_eq!(
            parse_value(&literal_repr(&Value::String("Bob".to_string())), DataType::String),
            Value::String("Bob".to_string())
        );
        assert_eq!(parse_value(&literal_repr(&Value::Boolean(true)), DataType::Boolean), Value::Boolean(true));
    }

    #[test]
    fn test_scan_placeholders_counts_positional() {
        let mut positional = 0;
        let mut indexed = 0;
        scan_placeholders("age > ? AND name = ?", &mut positional, &mut indexed).unwrap();
        assert_eq!(positional, 2);
        assert_eq!(indexed, 0);
    }

    #[test]
    fn test_scan_placeholders_tracks_max_indexed() {
        let mut positional = 0;
        let mut indexed = 0;
        scan_placeholders("age > $2 AND name = $1", &mut positional, &mut indexed).unwrap();
        assert_eq!(positional, 0);
        assert_eq!(indexed, 2);
    }

    #[test]
    fn test_scan_placeholders_rejects_mixed_styles() {
        let mut positional = 0;
        let mut indexed = 0;
        assert!(scan_placeholders("age > ? AND name = $1", &mut positional, &mut indexed).is_err());
    }

    #[test]
    fn test_scan_placeholders_ignores_quoted_question_mark() {
        // A literal '?' inside a quoted string must never be mistaken for
        // a placeholder -- tokenize_expr keeps the quote marks, so the
        // token is "'?'", not "?".
        let mut positional = 0;
        let mut indexed = 0;
        scan_placeholders("name = '?'", &mut positional, &mut indexed).unwrap();
        assert_eq!(positional, 0);
        assert_eq!(indexed, 0);
    }

    #[test]
    fn test_substitute_placeholders_positional() {
        let mut next = 0;
        let params = vec![Value::Integer(18), Value::String("Bob".to_string())];
        let out = substitute_placeholders("age > ? AND name = ?", &params, &mut next).unwrap();
        assert_eq!(out, "age > 18 AND name = 'Bob'");
        assert_eq!(next, 2);
    }

    #[test]
    fn test_substitute_placeholders_indexed_out_of_order() {
        let mut next = 0;
        let params = vec![Value::Integer(18), Value::String("Bob".to_string())];
        // $2 before $1 -- indexed placeholders don't depend on appearance
        // order the way positional ones do.
        let out = substitute_placeholders("name = $2 AND age > $1", &params, &mut next).unwrap();
        assert_eq!(out, "name = 'Bob' AND age > 18");
    }

    #[test]
    fn test_substitute_placeholders_missing_param_errors() {
        let mut next = 0;
        let params = vec![Value::Integer(18)];
        assert!(substitute_placeholders("age > ? AND name = ?", &params, &mut next).is_err());
    }

    #[test]
    fn test_substitute_token_matches_substitute_placeholders_token_by_token() {
        // substitute_token is substitute_placeholders' per-token step,
        // pulled out so a caller that already knows a field is exactly
        // one token (split_comparison's (left, op, right)) can substitute
        // without tokenizing at all -- see QueryExecutor::execute_prepared's
        // Filter case. Applying it to each of "id = ?"'s own tokens must
        // give the same result substitute_placeholders gives the whole
        // string, just without the redundant re-tokenize.
        let params = vec![Value::Integer(42)];

        let mut next_a = 0;
        let whole = substitute_placeholders("id = ?", &params, &mut next_a).unwrap();

        let mut next_b = 0;
        let left = substitute_token("id", &params, &mut next_b).unwrap();
        let op = substitute_token("=", &params, &mut next_b).unwrap();
        let right = substitute_token("?", &params, &mut next_b).unwrap();

        assert_eq!(whole, format!("{left} {op} {right}"));
        assert_eq!(next_a, next_b);
    }

    #[test]
    fn test_substitute_token_indexed_placeholder() {
        let params = vec![Value::Integer(18), Value::String("Bob".to_string())];
        let mut next = 0;
        assert_eq!(substitute_token("$2", &params, &mut next).unwrap(), "'Bob'");
        // $N never advances the positional counter -- same rule
        // substitute_placeholders itself follows.
        assert_eq!(next, 0);
    }

    #[test]
    fn test_substitute_token_non_placeholder_passes_through_unchanged() {
        let mut next = 0;
        assert_eq!(substitute_token("id", &[], &mut next).unwrap(), "id");
        assert_eq!(next, 0);
    }
}
