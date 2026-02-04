use pest::iterators::Pair;

use crate::{InterpreterError, Rule, evaluate_term, format_result, nth_prime};

/// Check if a pair contains a Real (floating-point) number literal
fn contains_real(pair: &Pair<Rule>) -> bool {
  match pair.as_rule() {
    Rule::Real => true,
    Rule::Integer => false,
    _ => {
      // Check the string representation for a decimal point
      let s = pair.as_str();
      s.contains('.') && !s.contains("->")
    }
  }
}

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
  let max_val = values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

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
  let min_val = values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));

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

/// Try to parse an expression as a simple fraction a/b where a and b are integers
fn try_parse_fraction(pair: &Pair<Rule>) -> Option<(i64, i64)> {
  let s = pair.as_str().trim();
  // Check if it's in the form "a/b"
  if let Some(pos) = s.find('/') {
    let num_str = s[..pos].trim();
    let den_str = s[pos + 1..].trim();
    if let (Ok(num), Ok(den)) = (num_str.parse::<i64>(), den_str.parse::<i64>())
      && den != 0
    {
      return Some((num, den));
    }
  }
  None
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

  // Check if either argument contains a real (floating-point) number
  let base_is_real = contains_real(&args_pairs[0]);
  let exp_is_real = contains_real(&args_pairs[1]);
  let input_has_real = base_is_real || exp_is_real;

  // Check if exponent is a rational fraction like 1/3
  let exp_fraction = try_parse_fraction(&args_pairs[1]);

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

  // Check if exponent is a rational fraction and the result is close to an integer
  if let Some((num, den)) = exp_fraction {
    // For fractional exponents like 1/3, check if result is very close to an integer
    let rounded = result.round();
    if (result - rounded).abs() < 1e-6 {
      // Verify this is an exact root by computing rounded^(den/num) and comparing to base
      // For Power[27, 1/3]: rounded=3, we check 3^3 = 27
      let base_int = base as i64;
      let rounded_int = rounded as i64;
      if num == 1 && rounded_int > 0 {
        // Check if rounded_int^den equals base
        let check = rounded_int.pow(den as u32);
        if check == base_int {
          return Ok(format_result(rounded));
        }
      }
    }
  }

  // Handle output formatting based on input types
  if input_has_real {
    // If input contains reals, output should be a real number
    // Use trailing dot for whole numbers (e.g., 2.)
    if result.fract() == 0.0 {
      Ok(format!("{}.", result as i64))
    } else {
      // Full precision for non-integer results
      let formatted = format!("{:.16}", result);
      // Trim trailing zeros but keep at least one decimal place
      let trimmed = formatted.trim_end_matches('0');
      if trimmed.ends_with('.') {
        Ok(format!("{}0", trimmed))
      } else {
        Ok(trimmed.to_string())
      }
    }
  } else {
    // Both base and exponent are integers or simple fractions
    let base_int = base as i64;

    // Check if exponent is a negative integer
    if exponent.fract() == 0.0 && exponent < 0.0 {
      let exp_int = exponent as i64;
      // Negative integer exponent: result should be a fraction
      // Power[base, -n] = 1 / base^n
      let denominator = base_int.pow((-exp_int) as u32);
      return Ok(crate::format_fraction(1, denominator));
    }

    // Check if exponent is a positive integer
    if exponent.fract() == 0.0 && exponent >= 0.0 {
      return Ok(format_result(result));
    }

    // For fractional exponents without explicit real literals, return formatted result
    Ok(format_result(result))
  }
}

/// Handle Factorial[n] - Returns the factorial of n (n!)
pub fn factorial(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
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

/// Handle LCM[a, b, ...] - Returns the least common multiple
pub fn lcm(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "LCM expects at least 1 argument".into(),
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

  // Helper function to compute LCM of two numbers
  fn lcm_two(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
      0
    } else {
      (a.abs() / gcd_two(a, b)) * b.abs()
    }
  }

  // Evaluate all arguments and check they are integers
  let mut values = Vec::new();
  for arg in args_pairs {
    let val = evaluate_term(arg.clone())?;

    if val.fract() != 0.0 {
      return Err(InterpreterError::EvaluationError(
        "LCM: all arguments must be integers".into(),
      ));
    }

    values.push(val as i64);
  }

  // Compute LCM of all values
  let result = values.into_iter().reduce(lcm_two).unwrap();

  Ok(format_result(result as f64))
}

/// Handle Exp[x] - Returns e^x (the exponential function)
pub fn exp(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Exp expects exactly 1 argument".into(),
    ));
  }
  let x = evaluate_term(args_pairs[0].clone())?;
  Ok(format_result(x.exp()))
}

/// Handle Log[x] or Log[b, x] - Returns the natural logarithm or logarithm with base b
pub fn log(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  match args_pairs.len() {
    1 => {
      // Natural logarithm
      let x = evaluate_term(args_pairs[0].clone())?;
      if x <= 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Log: argument must be positive".into(),
        ));
      }
      Ok(format_result(x.ln()))
    }
    2 => {
      // Logarithm with base b: Log[b, x]
      let b = evaluate_term(args_pairs[0].clone())?;
      let x = evaluate_term(args_pairs[1].clone())?;
      if b <= 0.0 || b == 1.0 {
        return Err(InterpreterError::EvaluationError(
          "Log: base must be positive and not equal to 1".into(),
        ));
      }
      if x <= 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Log: argument must be positive".into(),
        ));
      }
      Ok(format_result(x.ln() / b.ln()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Log expects 1 or 2 arguments".into(),
    )),
  }
}

/// Handle Log10[x] - Returns the base-10 logarithm
pub fn log10(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Log10 expects exactly 1 argument".into(),
    ));
  }
  let x = evaluate_term(args_pairs[0].clone())?;
  if x <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Log10: argument must be positive".into(),
    ));
  }
  Ok(format_result(x.log10()))
}

/// Handle Log2[x] - Returns the base-2 logarithm
pub fn log2(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Log2 expects exactly 1 argument".into(),
    ));
  }
  let x = evaluate_term(args_pairs[0].clone())?;
  if x <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Log2: argument must be positive".into(),
    ));
  }
  Ok(format_result(x.log2()))
}

/// Handle Cos[x] - Returns the cosine of the argument
pub fn cos(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cos expects exactly 1 argument".into(),
    ));
  }
  let x = evaluate_term(args_pairs[0].clone())?;
  Ok(format_result(x.cos()))
}

/// Handle Tan[x] - Returns the tangent of the argument
pub fn tan(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Tan expects exactly 1 argument".into(),
    ));
  }
  let x = evaluate_term(args_pairs[0].clone())?;
  Ok(format_result(x.tan()))
}

/// Handle ArcSin[x] - Returns the arc sine of the argument
pub fn arcsin(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSin expects exactly 1 argument".into(),
    ));
  }
  let x = evaluate_term(args_pairs[0].clone())?;
  if !(-1.0..=1.0).contains(&x) {
    return Err(InterpreterError::EvaluationError(
      "ArcSin: argument must be in the range [-1, 1]".into(),
    ));
  }
  Ok(format_result(x.asin()))
}

/// Handle ArcCos[x] - Returns the arc cosine of the argument
pub fn arccos(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCos expects exactly 1 argument".into(),
    ));
  }
  let x = evaluate_term(args_pairs[0].clone())?;
  if !(-1.0..=1.0).contains(&x) {
    return Err(InterpreterError::EvaluationError(
      "ArcCos: argument must be in the range [-1, 1]".into(),
    ));
  }
  Ok(format_result(x.acos()))
}

/// Handle ArcTan[x] - Returns the arc tangent of the argument
pub fn arctan(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcTan expects exactly 1 argument".into(),
    ));
  }
  let x = evaluate_term(args_pairs[0].clone())?;
  Ok(format_result(x.atan()))
}

/// Handle Quotient[m, n] - Returns the integer quotient of m divided by n
pub fn quotient(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Quotient expects exactly 2 arguments".into(),
    ));
  }

  let m = evaluate_term(args_pairs[0].clone())?;
  let n = evaluate_term(args_pairs[1].clone())?;

  if n == 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Quotient: division by zero".into(),
    ));
  }

  // Wolfram's Quotient is Floor[m/n]
  let result = (m / n).floor();
  Ok(format_result(result))
}

/// Handle N[expr] - Forces numeric evaluation of an expression
pub fn numeric_eval(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() || args_pairs.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "N expects 1 or 2 arguments".into(),
    ));
  }

  // Evaluate the expression
  let result = crate::evaluate_expression(args_pairs[0].clone())?;

  // Try to parse as a number and return with decimal point
  if let Ok(val) = result.parse::<f64>() {
    // If a precision is specified, use it
    if args_pairs.len() == 2 {
      let precision = evaluate_term(args_pairs[1].clone())?;
      if precision < 1.0 {
        return Err(InterpreterError::EvaluationError(
          "N: precision must be at least 1".into(),
        ));
      }
      let prec = precision as usize;
      return Ok(format!("{:.1$}", val, prec));
    }
    // Default: return as float with enough precision
    if val.fract() == 0.0 {
      Ok(format!("{}.", val as i64))
    } else {
      Ok(format!("{}", val))
    }
  } else {
    // Can't evaluate numerically, return as-is
    Ok(result)
  }
}

/// Handle IntegerDigits[n] or IntegerDigits[n, base] - Returns list of digits
pub fn integer_digits(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() || args_pairs.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerDigits expects 1 or 2 arguments".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;
  if n.fract() != 0.0 {
    return Err(InterpreterError::EvaluationError(
      "IntegerDigits: first argument must be an integer".into(),
    ));
  }

  let n_int = n.abs() as u64;

  let base = if args_pairs.len() == 2 {
    let b = evaluate_term(args_pairs[1].clone())?;
    if b.fract() != 0.0 || b < 2.0 {
      return Err(InterpreterError::EvaluationError(
        "IntegerDigits: base must be an integer >= 2".into(),
      ));
    }
    b as u64
  } else {
    10
  };

  if n_int == 0 {
    return Ok("{0}".to_string());
  }

  let mut digits = Vec::new();
  let mut num = n_int;

  while num > 0 {
    digits.push((num % base).to_string());
    num /= base;
  }

  digits.reverse();
  Ok(format!("{{{}}}", digits.join(", ")))
}

/// Handle FromDigits[list] or FromDigits[list, base] - Constructs integer from digits
pub fn from_digits(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() || args_pairs.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FromDigits expects 1 or 2 arguments".into(),
    ));
  }

  let base: u64 = if args_pairs.len() == 2 {
    let b = evaluate_term(args_pairs[1].clone())?;
    if b.fract() != 0.0 || b < 2.0 {
      return Err(InterpreterError::EvaluationError(
        "FromDigits: base must be an integer >= 2".into(),
      ));
    }
    b as u64
  } else {
    10
  };

  let items = crate::functions::list::get_list_items(&args_pairs[0])?;
  let mut result: u64 = 0;

  for item in items {
    let digit = evaluate_term(item.clone())?;
    if digit.fract() != 0.0 || digit < 0.0 || digit >= base as f64 {
      return Err(InterpreterError::EvaluationError(format!(
        "FromDigits: invalid digit {} for base {}",
        digit, base
      )));
    }
    result = result * base + digit as u64;
  }

  Ok(format_result(result as f64))
}

/// Handle FactorInteger[n] - Returns the prime factorization of n
/// Returns a list of {prime, exponent} pairs
pub fn factor_integer(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;
  if n.fract() != 0.0 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger: argument must be an integer".into(),
    ));
  }

  let n_int = n.abs() as u64;

  if n_int == 0 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger: argument cannot be zero".into(),
    ));
  }

  if n_int == 1 {
    return Ok("{}".to_string());
  }

  let mut factors: Vec<(u64, u32)> = Vec::new();
  let mut num = n_int;

  // Handle factor of 2
  let mut count = 0;
  while num.is_multiple_of(2) {
    count += 1;
    num /= 2;
  }
  if count > 0 {
    factors.push((2, count));
  }

  // Handle odd factors
  let mut i = 3;
  while i * i <= num {
    let mut count = 0;
    while num.is_multiple_of(i) {
      count += 1;
      num /= i;
    }
    if count > 0 {
      factors.push((i, count));
    }
    i += 2;
  }

  // If there's a remaining prime factor > sqrt(n)
  if num > 1 {
    factors.push((num, 1));
  }

  // Handle negative numbers by prepending {-1, 1}
  let mut result_parts: Vec<String> = Vec::new();
  if n < 0.0 {
    result_parts.push("{-1, 1}".to_string());
  }

  for (prime, exp) in factors {
    result_parts.push(format!("{{{}, {}}}", prime, exp));
  }

  Ok(format!("{{{}}}", result_parts.join(", ")))
}

/// Handle Re[z] - returns the real part of a complex number
/// For real numbers, returns the number itself
/// For Complex[a, b], returns a
pub fn re(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Re expects exactly 1 argument".into(),
    ));
  }

  let expr = crate::evaluate_expression(args_pairs[0].clone())?;

  // Check if it's a Complex expression
  if expr.starts_with("Complex[") && expr.ends_with(']') {
    let inner = &expr[8..expr.len() - 1];
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() == 2 {
      return Ok(parts[0].trim().to_string());
    }
  }

  // For real numbers, just return the number itself
  if expr.parse::<f64>().is_ok() {
    return Ok(expr);
  }

  // Return symbolic form for other expressions
  Ok(format!("Re[{}]", expr))
}

/// Handle Im[z] - returns the imaginary part of a complex number
/// For real numbers, returns 0
/// For Complex[a, b], returns b
pub fn im(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Im expects exactly 1 argument".into(),
    ));
  }

  let expr = crate::evaluate_expression(args_pairs[0].clone())?;

  // Check if it's a Complex expression
  if expr.starts_with("Complex[") && expr.ends_with(']') {
    let inner = &expr[8..expr.len() - 1];
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() == 2 {
      return Ok(parts[1].trim().to_string());
    }
  }

  // For real numbers, imaginary part is 0
  if expr.parse::<f64>().is_ok() {
    return Ok("0".to_string());
  }

  // Return symbolic form for other expressions
  Ok(format!("Im[{}]", expr))
}

/// Handle Conjugate[z] - returns the complex conjugate
/// For real numbers, returns the number itself
/// For Complex[a, b], returns Complex[a, -b]
pub fn conjugate(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Conjugate expects exactly 1 argument".into(),
    ));
  }

  let expr = crate::evaluate_expression(args_pairs[0].clone())?;

  // Check if it's a Complex expression
  if expr.starts_with("Complex[") && expr.ends_with(']') {
    let inner = &expr[8..expr.len() - 1];
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() == 2 {
      let real = parts[0].trim();
      let imag = parts[1].trim();
      // Negate the imaginary part
      if let Ok(imag_val) = imag.parse::<f64>() {
        let neg_imag = -imag_val;
        return Ok(format!("Complex[{}, {}]", real, format_result(neg_imag)));
      }
      return Ok(format!("Complex[{}, -{}]", real, imag));
    }
  }

  // For real numbers, conjugate is the number itself
  if expr.parse::<f64>().is_ok() {
    return Ok(expr);
  }

  // Return symbolic form for other expressions
  Ok(format!("Conjugate[{}]", expr))
}

/// Handle Rationalize[x] - converts a real number to a nearby rational
/// Rationalize[x, dx] - finds rational within dx of x
pub fn rationalize(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() || args_pairs.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Rationalize expects 1 or 2 arguments".into(),
    ));
  }

  let x = evaluate_term(args_pairs[0].clone())?;

  // Default tolerance: use machine epsilon for single-arg case
  // (only rationalize exact representations)
  let tolerance = if args_pairs.len() == 2 {
    evaluate_term(args_pairs[1].clone())?
  } else {
    f64::EPSILON // very tight tolerance for exact representations only
  };

  // Handle integers
  if x.fract() == 0.0 {
    return Ok(format_result(x));
  }

  // Maximum denominator for single-argument case (matches Wolfram behavior)
  let max_denom: i64 = if args_pairs.len() == 1 {
    100000
  } else {
    i64::MAX
  };

  // Use continued fraction algorithm to find best rational approximation
  let (num, denom) = find_rational(x, tolerance, max_denom);

  // Verify the approximation is within tolerance
  let approx = num as f64 / denom as f64;
  if (approx - x).abs() >= tolerance {
    // No valid rational found within tolerance, return original number
    return Ok(format_result(x));
  }

  if denom == 1 {
    Ok(num.to_string())
  } else {
    // Return as a fraction using Times and Power for division
    Ok(format!("{}/{}", num, denom))
  }
}

/// Find best rational approximation using continued fractions
fn find_rational(x: f64, tolerance: f64, max_denom: i64) -> (i64, i64) {
  if x == 0.0 {
    return (0, 1);
  }

  let sign = if x < 0.0 { -1 } else { 1 };
  let x = x.abs();

  let mut p0: i64 = 0;
  let mut q0: i64 = 1;
  let mut p1: i64 = 1;
  let mut q1: i64 = 0;

  let mut xi = x;

  for _ in 0..50 {
    // Prevent infinite loops
    let ai = xi.floor() as i64;
    let p2 = ai * p1 + p0;
    let q2 = ai * q1 + q0;

    if q2 == 0 || q2 > max_denom {
      break;
    }

    let approx = p2 as f64 / q2 as f64;
    if (approx - x).abs() < tolerance {
      return (sign * p2, q2);
    }

    let frac = xi - ai as f64;
    if frac.abs() < 1e-15 {
      break;
    }
    xi = 1.0 / frac;

    p0 = p1;
    q0 = q1;
    p1 = p2;
    q1 = q2;
  }

  (sign * p1, q1)
}

/// Handle Arg[z] - returns the argument (phase angle) of a complex number
/// For real positive numbers, returns 0
/// For real negative numbers, returns Pi
/// For Complex[a, b], returns ArcTan[a, b]
pub fn arg(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Arg expects exactly 1 argument".into(),
    ));
  }

  let expr = crate::evaluate_expression(args_pairs[0].clone())?;

  // Check if it's a Complex expression
  if expr.starts_with("Complex[") && expr.ends_with(']') {
    let inner = &expr[8..expr.len() - 1];
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() == 2
      && let (Ok(re), Ok(im)) = (
        parts[0].trim().parse::<f64>(),
        parts[1].trim().parse::<f64>(),
      )
    {
      let angle = im.atan2(re);
      return Ok(format_result(angle));
    }
  }

  // For real numbers
  if let Ok(val) = expr.parse::<f64>() {
    if val > 0.0 {
      return Ok("0".to_string());
    } else if val < 0.0 {
      return Ok("Pi".to_string());
    } else {
      // Arg[0] is undefined, but Wolfram returns 0
      return Ok("0".to_string());
    }
  }

  // Return symbolic form for other expressions
  Ok(format!("Arg[{}]", expr))
}

/// Handle Divisors[n] - returns a sorted list of all divisors of n
pub fn divisors(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Divisors expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;

  if n.fract() != 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Divisors: argument must be an integer".into(),
    ));
  }

  let n_int = n.abs() as u64;

  if n_int == 0 {
    return Err(InterpreterError::EvaluationError(
      "Divisors: argument cannot be zero".into(),
    ));
  }

  let mut divs = Vec::new();
  let sqrt_n = (n_int as f64).sqrt() as u64;

  for i in 1..=sqrt_n {
    if n_int.is_multiple_of(i) {
      divs.push(i);
      if i != n_int / i {
        divs.push(n_int / i);
      }
    }
  }

  divs.sort();

  let result: Vec<String> = divs.iter().map(|d| d.to_string()).collect();
  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle DivisorSigma[k, n] - returns the sum of the k-th powers of divisors of n
/// DivisorSigma[0, n] counts divisors, DivisorSigma[1, n] sums divisors
pub fn divisor_sigma(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSigma expects exactly 2 arguments".into(),
    ));
  }

  let k = evaluate_term(args_pairs[0].clone())?;
  let n = evaluate_term(args_pairs[1].clone())?;

  if k.fract() != 0.0 || k < 0.0 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSigma: first argument must be a non-negative integer".into(),
    ));
  }

  if n.fract() != 0.0 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSigma: second argument must be an integer".into(),
    ));
  }

  let k_int = k as u32;
  let n_int = n.abs() as u64;

  if n_int == 0 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSigma: second argument cannot be zero".into(),
    ));
  }

  // Find all divisors and sum their k-th powers
  let sqrt_n = (n_int as f64).sqrt() as u64;
  let mut sum: u64 = 0;

  for i in 1..=sqrt_n {
    if n_int.is_multiple_of(i) {
      sum += i.pow(k_int);
      if i != n_int / i {
        sum += (n_int / i).pow(k_int);
      }
    }
  }

  Ok(format_result(sum as f64))
}

/// Handle MoebiusMu[n] - returns the Möbius function value
/// MoebiusMu[n] = 1 if n is square-free with even number of prime factors
/// MoebiusMu[n] = -1 if n is square-free with odd number of prime factors
/// MoebiusMu[n] = 0 if n has a squared prime factor
pub fn moebius_mu(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MoebiusMu expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;

  if n.fract() != 0.0 || n < 1.0 {
    return Err(InterpreterError::EvaluationError(
      "MoebiusMu: argument must be a positive integer".into(),
    ));
  }

  let mut num = n as u64;

  if num == 1 {
    return Ok("1".to_string());
  }

  let mut prime_count = 0;

  // Check for factor 2
  if num.is_multiple_of(2) {
    prime_count += 1;
    num /= 2;
    if num.is_multiple_of(2) {
      return Ok("0".to_string()); // Has squared factor
    }
  }

  // Check odd factors
  let mut i = 3u64;
  while i * i <= num {
    if num.is_multiple_of(i) {
      prime_count += 1;
      num /= i;
      if num.is_multiple_of(i) {
        return Ok("0".to_string()); // Has squared factor
      }
    }
    i += 2;
  }

  // If there's a remaining prime factor
  if num > 1 {
    prime_count += 1;
  }

  if prime_count % 2 == 0 {
    Ok("1".to_string())
  } else {
    Ok("-1".to_string())
  }
}

/// Handle EulerPhi[n] - returns Euler's totient function (count of coprimes ≤ n)
pub fn euler_phi(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EulerPhi expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;

  if n.fract() != 0.0 || n < 1.0 {
    return Err(InterpreterError::EvaluationError(
      "EulerPhi: argument must be a positive integer".into(),
    ));
  }

  let mut num = n as u64;
  let mut result = num;

  // Euler's product formula: φ(n) = n * ∏(1 - 1/p) for all prime factors p
  let mut p = 2u64;
  while p * p <= num {
    if num.is_multiple_of(p) {
      // Remove all factors of p
      while num.is_multiple_of(p) {
        num /= p;
      }
      result -= result / p;
    }
    p += 1;
  }

  // If there's a remaining prime factor
  if num > 1 {
    result -= result / num;
  }

  Ok(format_result(result as f64))
}

/// Handle CoprimeQ[a, b] - tests if two integers are coprime (GCD = 1)
pub fn coprime_q(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CoprimeQ expects exactly 2 arguments".into(),
    ));
  }

  let a = evaluate_term(args_pairs[0].clone())?;
  let b = evaluate_term(args_pairs[1].clone())?;

  if a.fract() != 0.0 || b.fract() != 0.0 {
    return Err(InterpreterError::EvaluationError(
      "CoprimeQ: arguments must be integers".into(),
    ));
  }

  let mut a_int = (a.abs() as u64).max(1);
  let mut b_int = (b.abs() as u64).max(1);

  // Calculate GCD using Euclidean algorithm
  while b_int != 0 {
    let temp = b_int;
    b_int = a_int % b_int;
    a_int = temp;
  }

  Ok(if a_int == 1 { "True" } else { "False" }.to_string())
}
