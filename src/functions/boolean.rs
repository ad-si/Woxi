use pest::iterators::Pair;

use crate::{InterpreterError, Rule, evaluate_expression};

/// Helper function for boolean conversion
pub fn as_bool(s: &str) -> Option<bool> {
  match s {
    "True" => Some(true),
    "False" => Some(false),
    _ => None,
  }
}

/// Handle And[expr1, expr2, ...] - logical AND of expressions
pub fn and(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "And expects at least 2 arguments".into(),
    ));
  }
  for ap in args_pairs {
    if !as_bool(&evaluate_expression(ap.clone())?).unwrap_or(false) {
      return Ok("False".to_string());
    }
  }
  Ok("True".to_string())
}

/// Handle Or[expr1, expr2, ...] - logical OR of expressions
pub fn or(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Or expects at least 2 arguments".into(),
    ));
  }
  for ap in args_pairs {
    if as_bool(&evaluate_expression(ap.clone())?).unwrap_or(false) {
      return Ok("True".to_string());
    }
  }
  Ok("False".to_string())
}

/// Handle Xor[expr1, expr2, ...] - logical XOR of expressions
pub fn xor(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Xor expects at least 2 arguments".into(),
    ));
  }
  let mut true_cnt = 0;
  for ap in args_pairs {
    if as_bool(&evaluate_expression(ap.clone())?).unwrap_or(false) {
      true_cnt += 1;
    }
  }
  Ok(if true_cnt % 2 == 1 { "True" } else { "False" }.to_string())
}

/// Handle Not[expr] - logical negation
pub fn not(
  args_pairs: &[Pair<Rule>],
  _call_text: &str,
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    use std::io::{self, Write};
    println!(
      "\nNot::argx: Not called with {} arguments; 1 argument is expected.",
      args_pairs.len()
    );
    io::stdout().flush().ok();

    // rebuild unevaluated expression
    let mut parts = Vec::new();
    for ap in args_pairs {
      parts.push(evaluate_expression(ap.clone())?);
    }
    return Ok(format!("Not[{}]", parts.join(", ")));
  }
  let v =
    as_bool(&evaluate_expression(args_pairs[0].clone())?).unwrap_or(false);
  Ok(if v { "False" } else { "True" }.to_string())
}

/// Handle If[test, t, f, u] - conditional expression
pub fn if_condition(
  args_pairs: &[Pair<Rule>],
  call_text: &str,
) -> Result<String, InterpreterError> {
  // arity 2‥4
  if !(2..=4).contains(&args_pairs.len()) {
    use std::io::{self, Write};
    println!(
      "\nIf::argb: If called with {} arguments; between 2 and 4 arguments are expected.",
      args_pairs.len()
    );
    io::stdout().flush().ok();
    return Ok(call_text.to_string()); // return unevaluated expression
  }

  // evaluate test
  let test_str = evaluate_expression(args_pairs[0].clone())?;
  let test_val = as_bool(&test_str);

  match (test_val, args_pairs.len()) {
    (Some(true), _) => evaluate_expression(args_pairs[1].clone()),
    (Some(false), 2) => Ok("Null".to_string()),
    (Some(false), 3) => evaluate_expression(args_pairs[2].clone()),
    (Some(false), 4) => evaluate_expression(args_pairs[2].clone()),
    (_, 2) => Ok("Null".to_string()),
    (_, 3) => Ok("Null".to_string()),
    (_, 4) => evaluate_expression(args_pairs[3].clone()),
    _ => unreachable!(),
  }
}
