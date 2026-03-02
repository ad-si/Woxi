//! AST-native control flow functions.
//!
//! Switch, Piecewise, Quiet.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::syntax::Expr;
use crate::syntax::expr_to_string;

/// Switch[expr, pat1, val1, pat2, val2, ..., default?]
/// Evaluates expr, then finds first matching pattern and returns corresponding value.
/// Uses lazy evaluation — only matched branch is evaluated.
pub fn switch_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Err(InterpreterError::EvaluationError(
      "Switch called with too few arguments; at least 3 are expected.".into(),
    ));
  }

  // Evaluate the test expression
  let test = evaluate_expr_to_expr(&args[0])?;
  let test_str = crate::syntax::expr_to_string(&test);

  // Iterate pattern-value pairs
  let rest = &args[1..];
  let mut i = 0;
  while i + 1 < rest.len() {
    let pattern = &rest[i];
    let value = &rest[i + 1];

    // Check if pattern matches
    if pattern_matches(&test, pattern, &test_str) {
      return evaluate_expr_to_expr(value);
    }
    i += 2;
  }

  // Check for default (odd remaining argument)
  if i < rest.len() {
    return evaluate_expr_to_expr(&rest[i]);
  }

  // No match, no default — return unevaluated
  Ok(Expr::FunctionCall {
    name: "Switch".to_string(),
    args: args.to_vec(),
  })
}

/// Check if `test` matches `pattern`.
fn pattern_matches(test: &Expr, pattern: &Expr, _test_str: &str) -> bool {
  crate::evaluator::match_pattern(test, pattern).is_some()
}

/// Piecewise[{{val1, cond1}, {val2, cond2}, ...}] or
/// Piecewise[{{val1, cond1}, {val2, cond2}, ...}, default]
pub fn piecewise_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Piecewise expects 1 or 2 arguments".into(),
    ));
  }

  let pairs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "First argument of Piecewise must be a list of {value, condition} pairs".into(),
      ));
    }
  };

  let default = if args.len() == 2 {
    Some(&args[1])
  } else {
    None
  };

  let mut has_symbolic = false;

  for pair in pairs {
    match pair {
      Expr::List(items) if items.len() == 2 => {
        let cond = evaluate_expr_to_expr(&items[1])?;
        match &cond {
          Expr::Identifier(s) if s == "True" => {
            return evaluate_expr_to_expr(&items[0]);
          }
          Expr::Identifier(s) if s == "False" => {
            continue;
          }
          _ => {
            // Symbolic condition — can't evaluate
            has_symbolic = true;
          }
        }
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Each element of Piecewise list must be a {value, condition} pair"
            .into(),
        ));
      }
    }
  }

  if has_symbolic {
    // Return unevaluated
    return Ok(Expr::FunctionCall {
      name: "Piecewise".to_string(),
      args: args.to_vec(),
    });
  }

  // No condition was True — return default (or 0)
  match default {
    Some(d) => evaluate_expr_to_expr(d),
    None => Ok(Expr::Integer(0)),
  }
}

/// Parse the message-off specification for Quiet.
/// Returns None for All (suppress everything), Some(vec) for specific messages.
/// Each message is a (symbol, tag) pair like ("Power", "infy").
fn parse_quiet_spec(spec: &Expr) -> Option<Vec<(String, String)>> {
  match spec {
    Expr::Identifier(s) if s == "All" => None,
    Expr::Identifier(s) if s == "None" => Some(vec![]),
    Expr::List(items) => {
      let mut msgs = Vec::new();
      for item in items {
        if let Some((sym, tag)) = parse_message_name(item) {
          msgs.push((sym, tag));
        }
      }
      Some(msgs)
    }
    // Single message name
    _ => {
      if let Some((sym, tag)) = parse_message_name(spec) {
        Some(vec![(sym, tag)])
      } else {
        // Unrecognized — treat as All
        None
      }
    }
  }
}

/// Extract (symbol, tag) from a MessageName expression like Power::infy.
/// MessageName[Power, "infy"] → ("Power", "infy")
fn parse_message_name(expr: &Expr) -> Option<(String, String)> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "MessageName" && args.len() == 2 =>
    {
      let sym = match &args[0] {
        Expr::Identifier(s) => s.clone(),
        _ => expr_to_string(&args[0]),
      };
      let tag = match &args[1] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => expr_to_string(&args[1]),
      };
      Some((sym, tag))
    }
    _ => None,
  }
}

/// Check if a warning message string matches a (symbol, tag) spec.
/// Warning format: "Symbol::tag: ..."
fn message_matches(warning: &str, specs: &[(String, String)]) -> bool {
  for (sym, tag) in specs {
    let prefix = format!("{}::{}: ", sym, tag);
    if warning.starts_with(&prefix) {
      return true;
    }
    // Also match without trailing space (in case message is just "Symbol::tag:")
    let prefix2 = format!("{}::{}:", sym, tag);
    if warning.starts_with(&prefix2) {
      return true;
    }
  }
  false
}

/// Quiet[expr] — evaluate expr, suppress all messages
/// Quiet[expr, {msg1, msg2, ...}] — suppress only specific messages
/// Quiet[expr, All] — suppress all messages
/// Quiet[expr, None] — suppress nothing
/// Quiet[expr, moff, mon] — suppress moff, enable mon
pub fn quiet_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Parse the message specification
  let suppress_spec = if args.len() == 1 {
    // Quiet[expr] — suppress all
    None // None means "all"
  } else {
    // Quiet[expr, spec] or Quiet[expr, moff, mon]
    parse_quiet_spec(&args[1])
  };

  // For 3-arg form Quiet[expr, moff, mon], mon re-enables messages within moff
  // For simplicity: if moff is None (All), mon still re-enables specific messages
  // But the common case is just Quiet[expr] or Quiet[expr, {msgs}]
  let enable_spec = if args.len() == 3 {
    parse_quiet_spec(&args[2])
  } else {
    Some(vec![]) // Nothing explicitly re-enabled
  };

  let suppress_all = suppress_spec.is_none();
  let suppress_msgs = suppress_spec.unwrap_or_default();

  // Determine effective suppression:
  // - If suppress_all and enable is None (All): suppress nothing (all re-enabled)
  // - If suppress_all and enable is Some(specific): suppress all except those in enable
  // - If suppress specific: suppress those, minus any in enable

  let effective_suppress_all = if suppress_all {
    match &enable_spec {
      None => false, // Quiet[expr, All, All] = everything re-enabled
      Some(enables) => enables.is_empty(), // No re-enables: suppress all
    }
  } else {
    false
  };

  // If None suppression (suppress nothing), just evaluate normally
  if suppress_msgs.is_empty() && !suppress_all {
    return evaluate_expr_to_expr(&args[0]);
  }

  // Save warning state
  let snapshot = crate::snapshot_warnings();

  // Suppress message printing
  if effective_suppress_all {
    crate::push_quiet();
  }

  // Evaluate the expression
  let result = evaluate_expr_to_expr(&args[0]);

  // Restore quiet level
  if effective_suppress_all {
    crate::pop_quiet();
  }

  // Handle warning cleanup based on suppression spec
  if effective_suppress_all {
    // Suppress all: restore all buffers to snapshot (discard everything added during eval)
    crate::restore_warnings(snapshot);
  } else if !suppress_msgs.is_empty() {
    // Suppress specific messages: remove only matching entries from the messages buffer
    let (snap_unimpl, snap_warns, snap_msgs) = snapshot;
    // Get current messages state
    let current_msgs = crate::get_captured_messages_raw();
    // Keep only messages that were in the snapshot OR don't match the filter
    let new_msgs: Vec<String> = current_msgs
      .into_iter()
      .enumerate()
      .filter(|(i, w)| {
        *i < snap_msgs.len() || !message_matches(w, &suppress_msgs)
      })
      .map(|(_, w)| w)
      .collect();
    crate::restore_warnings((snap_unimpl, snap_warns, new_msgs));
  }

  result
}
