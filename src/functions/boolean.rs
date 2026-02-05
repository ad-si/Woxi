use pest::iterators::Pair;

use crate::{
  InterpreterError, Rule, evaluate_expression,
  evaluator::evaluate_function_call,
};

/// Fast evaluation for function arguments - avoids expression overhead for simple cases
fn eval_arg(arg: Pair<Rule>) -> Result<String, InterpreterError> {
  match arg.as_rule() {
    Rule::Expression | Rule::ExpressionNoImplicit => {
      let mut inner = arg.clone().into_inner();
      if let Some(first) = inner.next()
        && inner.next().is_none()
      {
        // Single child - check for fast paths
        match first.as_rule() {
          Rule::FunctionCall => return evaluate_function_call(first),
          Rule::Identifier => {
            let id = first.as_str();
            if let Some(crate::StoredValue::Raw(val)) =
              crate::ENV.with(|e| e.borrow().get(id).cloned())
            {
              return Ok(val);
            }
            return Ok(id.to_string());
          }
          Rule::Integer | Rule::Real | Rule::NumericValue => {
            return crate::evaluate_term(first).map(crate::format_result);
          }
          Rule::String => {
            return Ok(first.as_str().to_string());
          }
          Rule::Term => {
            // Term wraps other rules - unwrap and recurse
            if let Some(inner_term) = first.clone().into_inner().next()
              && inner_term.as_rule() == Rule::FunctionCall
            {
              return evaluate_function_call(inner_term);
            }
          }
          _ => {}
        }
      }
      evaluate_expression(arg)
    }
    Rule::FunctionCall => evaluate_function_call(arg),
    _ => evaluate_expression(arg),
  }
}

/// Helper function for boolean conversion
pub fn as_bool(s: &str) -> Option<bool> {
  match s {
    "True" => Some(true),
    "False" => Some(false),
    _ => None,
  }
}

// ============================================================================
// String-based boolean helpers (for use when values are already evaluated)
// These avoid re-parsing and are much faster than calling interpret()
// ============================================================================

/// And for already-evaluated string values
pub fn and_strs(values: &[String]) -> String {
  for v in values {
    if !as_bool(v).unwrap_or(false) {
      return "False".to_string();
    }
  }
  "True".to_string()
}

/// Or for already-evaluated string values
pub fn or_strs(values: &[String]) -> String {
  for v in values {
    if as_bool(v).unwrap_or(false) {
      return "True".to_string();
    }
  }
  "False".to_string()
}

/// SameQ for already-evaluated string values
pub fn same_q_strs(values: &[String]) -> String {
  if values.len() < 2 {
    return "True".to_string();
  }
  let first = &values[0];
  for v in values.iter().skip(1) {
    if v != first {
      return "False".to_string();
    }
  }
  "True".to_string()
}

/// UnsameQ for already-evaluated string values
pub fn unsame_q_strs(values: &[String]) -> String {
  if values.len() < 2 {
    return "False".to_string();
  }
  let first = &values[0];
  for v in values.iter().skip(1) {
    if v != first {
      return "True".to_string();
    }
  }
  "False".to_string()
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

/// Handle SameQ[expr1, expr2] - tests whether expressions are identical
pub fn same_q(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "SameQ expects at least 2 arguments".into(),
    ));
  }
  let first = evaluate_expression(args_pairs[0].clone())?;
  for ap in args_pairs.iter().skip(1) {
    let val = evaluate_expression(ap.clone())?;
    if val != first {
      return Ok("False".to_string());
    }
  }
  Ok("True".to_string())
}

/// Handle UnsameQ[expr1, expr2] - tests whether expressions are not identical
pub fn unsame_q(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "UnsameQ expects at least 2 arguments".into(),
    ));
  }
  let first = evaluate_expression(args_pairs[0].clone())?;
  for ap in args_pairs.iter().skip(1) {
    let val = evaluate_expression(ap.clone())?;
    if val != first {
      return Ok("True".to_string());
    }
  }
  Ok("False".to_string())
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

  // evaluate test using fast path
  let test_str = eval_arg(args_pairs[0].clone())?;
  let test_val = as_bool(&test_str);

  match (test_val, args_pairs.len()) {
    (Some(true), _) => eval_arg(args_pairs[1].clone()),
    (Some(false), 2) => Ok("Null".to_string()),
    (Some(false), 3) => eval_arg(args_pairs[2].clone()),
    (Some(false), 4) => eval_arg(args_pairs[2].clone()),
    (_, 2) => Ok("Null".to_string()),
    (_, 3) => Ok("Null".to_string()),
    (_, 4) => eval_arg(args_pairs[3].clone()),
    _ => unreachable!(),
  }
}

/// Handle Which[test1, value1, test2, value2, ...] - multi-way conditional
pub fn which(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // Must have an even number of arguments (test-value pairs)
  if args_pairs.len() < 2 || !args_pairs.len().is_multiple_of(2) {
    return Err(InterpreterError::EvaluationError(
      "Which expects an even number of arguments (test-value pairs)".into(),
    ));
  }

  // Process test-value pairs using fast path
  for i in (0..args_pairs.len()).step_by(2) {
    let test_str = eval_arg(args_pairs[i].clone())?;
    if let Some(true) = as_bool(&test_str) {
      return eval_arg(args_pairs[i + 1].clone());
    }
  }

  // No condition was true
  Ok("Null".to_string())
}

/// Handle Do[expr, {i, n}] or Do[expr, {i, min, max}] or Do[expr, n] - iteration with side effects
pub fn do_loop(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Do expects exactly 2 arguments".into(),
    ));
  }

  let body = &args_pairs[0];
  let iter_spec = &args_pairs[1];

  // Parse iteration specification
  let iter_spec_str = iter_spec.as_str().trim();

  // Check if it's a simple number (Do[expr, n])
  if let Ok(n) = iter_spec_str.parse::<f64>() {
    if n.fract() != 0.0 || n < 0.0 {
      return Err(InterpreterError::EvaluationError(
        "Do: iteration count must be a non-negative integer".into(),
      ));
    }
    let count = n as usize;
    for _ in 0..count {
      eval_arg(body.clone())?;
    }
    return Ok("Null".to_string());
  }

  // Check if it's a list specification {i, max} or {i, min, max}
  if iter_spec_str.starts_with('{') && iter_spec_str.ends_with('}') {
    let inner = &iter_spec_str[1..iter_spec_str.len() - 1];
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();

    match parts.len() {
      2 => {
        // {i, max} - iterate from 1 to max
        let var_name = parts[0].to_string();
        let max: i64 = parts[1].parse().map_err(|_| {
          InterpreterError::EvaluationError("Do: invalid iteration spec".into())
        })?;

        for i in 1..=max {
          // Set the loop variable
          crate::ENV.with(|e| {
            e.borrow_mut()
              .insert(var_name.clone(), crate::StoredValue::Raw(i.to_string()))
          });
          eval_arg(body.clone())?;
        }

        // Clean up loop variable
        crate::ENV.with(|e| e.borrow_mut().remove(&var_name));
      }
      3 => {
        // {i, min, max} - iterate from min to max
        let var_name = parts[0].to_string();
        let min: i64 = parts[1].parse().map_err(|_| {
          InterpreterError::EvaluationError("Do: invalid iteration spec".into())
        })?;
        let max: i64 = parts[2].parse().map_err(|_| {
          InterpreterError::EvaluationError("Do: invalid iteration spec".into())
        })?;

        for i in min..=max {
          crate::ENV.with(|e| {
            e.borrow_mut()
              .insert(var_name.clone(), crate::StoredValue::Raw(i.to_string()))
          });
          eval_arg(body.clone())?;
        }

        crate::ENV.with(|e| e.borrow_mut().remove(&var_name));
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Do: invalid iteration specification".into(),
        ));
      }
    }
  } else {
    // Try to evaluate the spec as a number
    let n = crate::evaluate_term(iter_spec.clone())?;
    if n.fract() != 0.0 || n < 0.0 {
      return Err(InterpreterError::EvaluationError(
        "Do: iteration count must be a non-negative integer".into(),
      ));
    }
    let count = n as usize;
    for _ in 0..count {
      eval_arg(body.clone())?;
    }
  }

  Ok("Null".to_string())
}

/// Handle While[test, body] - while loop
pub fn while_loop(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "While expects exactly 2 arguments".into(),
    ));
  }

  let test = &args_pairs[0];
  let body = &args_pairs[1];

  // Maximum iterations to prevent infinite loops
  const MAX_ITERATIONS: usize = 100000;
  let mut iterations = 0;

  loop {
    // Evaluate the test condition
    let test_str = evaluate_expression(test.clone())?;
    let test_val = as_bool(&test_str);

    match test_val {
      Some(true) => {
        // Execute body
        evaluate_expression(body.clone())?;
        iterations += 1;
        if iterations >= MAX_ITERATIONS {
          return Err(InterpreterError::EvaluationError(
            "While: maximum iterations exceeded".into(),
          ));
        }
      }
      Some(false) => break,
      None => {
        return Err(InterpreterError::EvaluationError(
          "While: test must evaluate to True or False".into(),
        ));
      }
    }
  }

  Ok("Null".to_string())
}
