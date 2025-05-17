use pest::iterators::Pair;

use crate::{evaluate_term, format_result, nth_prime, InterpreterError, Rule};

/// Handle Sin[x] - returns the sine of the argument
pub fn sin(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sin expects exactly 1 argument".into(),
    ));
  }
  let n = evaluate_term(args_pairs[0].clone())?;
  Ok(format_result(n.sin()))
}

/// Handle Prime[n] - returns the nth prime number
pub fn prime(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Prime expects exactly 1 argument".into(),
    ));
  }
  let n = evaluate_term(args_pairs[0].clone())?;
  if n.fract() != 0.0 || n < 1.0 {
    return Err(InterpreterError::EvaluationError(
      "Prime function argument must be a positive integer greater than 0"
        .into(),
    ));
  }
  Ok(nth_prime(n as usize).to_string())
}

/// Handle Plus[a, b, ...] - adds all arguments
pub fn plus(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ── arity check ──────────────────────────────────────────────────────
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Plus expects at least 1 argument".into(),
    ));
  }

  // ── sum all numeric arguments ────────────────────────────────────────
  let mut sum = 0.0;
  for ap in args_pairs {
    sum += evaluate_term(ap.clone())?;
  }

  // ── return formatted result ──────────────────────────────────────────
  Ok(format_result(sum))
}

/// Handle Times[a, b, ...] - multiplies all arguments
pub fn times(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ----- arity check ---------------------------------------------------
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Times expects at least 1 argument".into(),
    ));
  }

  // ----- multiply all numeric arguments --------------------------------
  let mut product = 1.0;
  for ap in args_pairs {
    product *= evaluate_term(ap.clone())?;
  }

  // ----- return formatted result ---------------------------------------
  Ok(format_result(product))
}

/// Handle Minus[a] or Minus[a, b, ...] - negates a single value or prints expressions with minus sign
pub fn minus(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ---- arity check ----------------------------------------------------
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Minus expects at least 1 argument".into(),
    ));
  }

  // ---- unary / argument-count handling ----------------------------------
  if args_pairs.len() == 1 {
    // unary Minus
    let v = evaluate_term(args_pairs[0].clone())?;
    return Ok(format_result(-v));
  }

  // ---- wrong number of arguments  ---------------------------------------
  // Print *with* a trailing newline to match shelltest's expected output,
  // and flush stdout to ensure the order is correct for shelltest.
  use std::io::{self, Write};
  println!(
    "\nMinus::argx: Minus called with {} arguments; 1 argument is expected.",
    args_pairs.len()
  );
  io::stdout().flush().ok();

  // build the pretty printing of the unevaluated expression:  "5 − 2"
  let mut pieces = Vec::new();
  for ap in args_pairs {
    pieces.push(crate::evaluate_expression(ap.clone())?); // keeps formatting (e.g. 5, 2, 3.1…)
  }
  let expr = pieces.join(" − "); // note U+2212 (minus sign) surrounded by spaces
  Ok(expr)
}

/// Handle Abs[x] - returns the absolute value of the argument
pub fn abs(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ── arity check ────────────────────────────────────────────────────────
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Abs expects exactly 1 argument".into(),
    ));
  }
  // ── evaluate argument ──────────────────────────────────────────────────
  let n = evaluate_term(args_pairs[0].clone())?;
  // ── return absolute value, formatted like other numeric outputs ───────
  Ok(format_result(n.abs()))
}

/// Handle Sign[x] - returns -1, 0, or 1 based on the sign of the argument
pub fn sign(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sign expects exactly 1 argument".into(),
    ));
  }
  let n = evaluate_term(args_pairs[0].clone())?;
  Ok(
    if n > 0.0 {
      "1"
    } else if n < 0.0 {
      "-1"
    } else {
      "0"
    }
    .to_string(),
  )
}

/// Handle Sqrt[x] - returns the square root of the argument
pub fn sqrt(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ---- arity check ----------------------------------------------------
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sqrt expects exactly 1 argument".into(),
    ));
  }
  // ---- evaluate & validate argument -----------------------------------
  let n = evaluate_term(args_pairs[0].clone())?;
  if n < 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Sqrt function argument must be non-negative".into(),
    ));
  }
  // ---- return √n, formatted like all other numeric outputs ------------
  Ok(format_result(n.sqrt()))
}

/// Handle Floor[x] - returns the greatest integer less than or equal to x
pub fn floor(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Floor expects exactly 1 argument".into(),
    ));
  }
  let n = evaluate_term(args_pairs[0].clone())?;
  let mut r = n.floor();
  if r == -0.0 {
    r = 0.0;
  }
  Ok(format_result(r))
}

/// Handle Ceiling[x] - returns the least integer greater than or equal to x
pub fn ceiling(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Ceiling expects exactly 1 argument".into(),
    ));
  }
  let n = evaluate_term(args_pairs[0].clone())?;
  let mut r = n.ceil();
  if r == -0.0 {
    r = 0.0;
  }
  Ok(format_result(r))
}

/// Handle Round[x] - rounds to the nearest integer
pub fn round(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Round expects exactly 1 argument".into(),
    ));
  }
  let n = evaluate_term(args_pairs[0].clone())?;

  // banker's rounding (half-to-even)
  let base = n.trunc();
  let frac = n - base;
  let mut r = if frac.abs() == 0.5 {
    if (base as i64) % 2 == 0 {
      base
    }
    // already even
    else if n.is_sign_positive() {
      base + 1.0
    }
    // away from zero
    else {
      base - 1.0
    }
  } else {
    n.round()
  };
  if r == -0.0 {
    r = 0.0;
  }
  Ok(format_result(r))
}
