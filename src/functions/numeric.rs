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

/// Handle Max[x1, x2, ...] or Max[{x1, x2, ...}] - returns the maximum value
pub fn max(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ── arity check ──────────────────────────────────────────────────────
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Max expects at least 1 argument".into(),
    ));
  }

  // ── collect values to compare ────────────────────────────────────────
  let values: Vec<f64> = if args_pairs.len() == 1 {
    // Check if the single argument is a list
    let first_pair = &args_pairs[0];
    if let Ok(items) = crate::functions::list::get_list_items(first_pair) {
      // It's a list - evaluate all items
      if items.is_empty() {
        // Max[{}] returns -Infinity
        return Ok("-Infinity".to_string());
      }
      items
        .iter()
        .map(|item| evaluate_term(item.clone()))
        .collect::<Result<Vec<_>, _>>()?
    } else {
      // Not a list - just a single value
      vec![evaluate_term(first_pair.clone())?]
    }
  } else {
    // Multiple arguments - evaluate each one
    args_pairs
      .iter()
      .map(|ap| evaluate_term(ap.clone()))
      .collect::<Result<Vec<_>, _>>()?
  };

  // ── find maximum ─────────────────────────────────────────────────────
  let max_val = values
    .iter()
    .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

  // ── return formatted result ──────────────────────────────────────────
  Ok(format_result(max_val))
}

/// Handle Min[x1, x2, ...] or Min[{x1, x2, ...}] - returns the minimum value
pub fn min(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ── arity check ──────────────────────────────────────────────────────
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Min expects at least 1 argument".into(),
    ));
  }

  // ── collect values to compare ────────────────────────────────────────
  let values: Vec<f64> = if args_pairs.len() == 1 {
    // Check if the single argument is a list
    let first_pair = &args_pairs[0];
    if let Ok(items) = crate::functions::list::get_list_items(first_pair) {
      // It's a list - evaluate all items
      if items.is_empty() {
        // Min[{}] returns Infinity
        return Ok("Infinity".to_string());
      }
      items
        .iter()
        .map(|item| evaluate_term(item.clone()))
        .collect::<Result<Vec<_>, _>>()?
    } else {
      // Not a list - just a single value
      vec![evaluate_term(first_pair.clone())?]
    }
  } else {
    // Multiple arguments - evaluate each one
    args_pairs
      .iter()
      .map(|ap| evaluate_term(ap.clone()))
      .collect::<Result<Vec<_>, _>>()?
  };

  // ── find minimum ─────────────────────────────────────────────────────
  let min_val = values
    .iter()
    .fold(f64::INFINITY, |acc, &x| acc.min(x));

  // ── return formatted result ──────────────────────────────────────────
  Ok(format_result(min_val))
}

/// Handle Mod[m, n] - Returns the remainder when m is divided by n
pub fn modulo(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Mod expects exactly 2 arguments".into(),
    ));
  }

  let m = evaluate_term(args_pairs[0].clone())?;
  let n = evaluate_term(args_pairs[1].clone())?;

  if n == 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Mod: division by zero".into(),
    ));
  }

  // Wolfram's Mod function uses the formula: m - n * Floor[m/n]
  let result = m - n * (m / n).floor();
  Ok(format_result(result))
}

/// Handle Power[x, y] - Returns x raised to the power y
pub fn power(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Power expects exactly 2 arguments".into(),
    ));
  }

  let base = evaluate_term(args_pairs[0].clone())?;
  let exponent = evaluate_term(args_pairs[1].clone())?;

  // Handle special cases
  if base == 0.0 && exponent < 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Power: division by zero (0^negative)".into(),
    ));
  }

  let result = base.powf(exponent);

  // Check for NaN or infinity results
  if result.is_nan() {
    return Err(InterpreterError::EvaluationError(
      "Power: result is undefined (possibly negative base with fractional exponent)".into(),
    ));
  }

  if result.is_infinite() {
    return Err(InterpreterError::EvaluationError(
      "Power: result is infinite".into(),
    ));
  }

  Ok(format_result(result))
}

/// Handle Factorial[n] - Returns the factorial of n (n!)
pub fn factorial(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Factorial expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;

  // Check if n is a non-negative integer
  if n < 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Factorial: argument must be non-negative".into(),
    ));
  }

  if n.fract() != 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Factorial: argument must be an integer".into(),
    ));
  }

  let n_int = n as u64;

  // Calculate factorial
  // For large values, we'll use f64 to avoid overflow but this means precision loss
  let mut result = 1.0_f64;
  for i in 2..=n_int {
    result *= i as f64;

    // Check for overflow to infinity
    if result.is_infinite() {
      return Err(InterpreterError::EvaluationError(
        "Factorial: result is too large".into(),
      ));
    }
  }

  Ok(format_result(result))
}

/// Handle GCD[a, b, ...] - Returns the greatest common divisor
pub fn gcd(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "GCD expects at least 1 argument".into(),
    ));
  }

  // Helper function to compute GCD of two numbers using Euclidean algorithm
  fn gcd_two(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();

    while b != 0 {
      let temp = b;
      b = a % b;
      a = temp;
    }

    a
  }

  // Evaluate all arguments and check they are integers
  let mut values = Vec::new();
  for arg in args_pairs {
    let val = evaluate_term(arg.clone())?;

    if val.fract() != 0.0 {
      return Err(InterpreterError::EvaluationError(
        "GCD: all arguments must be integers".into(),
      ));
    }

    values.push(val as i64);
  }

  // Compute GCD of all values
  let result = values.into_iter().reduce(gcd_two).unwrap();

  Ok(format_result(result as f64))
}
