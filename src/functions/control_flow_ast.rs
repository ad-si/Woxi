//! AST-native control flow functions.
//!
//! Switch, Piecewise.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::syntax::Expr;

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
