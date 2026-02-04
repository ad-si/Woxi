use pest::iterators::Pair;

use crate::{InterpreterError, Rule, evaluate_expression, evaluate_term};

/// Handle NumberQ[expr] - Tests if the expression evaluates to a number
pub fn number_q(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ----- arity check --------------------------------------------------
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NumberQ expects exactly 1 argument".into(),
    ));
  }

  // Evaluate argument to string and try to parse it as f64
  let arg_str = evaluate_expression(args_pairs[0].clone())?;
  let is_number = arg_str.parse::<f64>().is_ok();
  Ok(if is_number { "True" } else { "False" }.to_string())
}

/// Handle EvenQ[n]/OddQ[n] - Tests if a number is even or odd
pub fn even_odd_q(
  args_pairs: &[Pair<Rule>],
  func_name: &str,
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects exactly 1 argument",
      func_name
    )));
  }
  let n = evaluate_term(args_pairs[0].clone())?;
  if n.fract() != 0.0 {
    return Ok("False".to_string());
  }
  let is_even = (n as i64) % 2 == 0;
  Ok(
    if (func_name == "EvenQ" && is_even) || (func_name == "OddQ" && !is_even) {
      "True"
    } else {
      "False"
    }
    .to_string(),
  )
}

/// Handle IntegerQ[expr] - Tests if the expression evaluates to an integer
/// In Wolfram Language, IntegerQ returns True only for actual integer representations,
/// not for real numbers that happen to have no fractional part (e.g., 3.0 returns False)
pub fn integer_q(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "IntegerQ expects exactly 1 argument".into(),
    ));
  }

  // Check if the argument is syntactically an integer (not a real number like 3.0)
  fn is_integer_literal(pair: &Pair<Rule>) -> bool {
    match pair.as_rule() {
      Rule::Integer => true,
      Rule::Real => false, // Real numbers like 3.0 are not integers
      Rule::NumericValue => {
        // Check inner value
        if let Some(inner) = pair.clone().into_inner().next() {
          is_integer_literal(&inner)
        } else {
          false
        }
      }
      Rule::Expression | Rule::Term => {
        // Check if it's just wrapping a single numeric value
        let inner: Vec<_> = pair.clone().into_inner().collect();
        if inner.len() == 1 {
          is_integer_literal(&inner[0])
        } else {
          false
        }
      }
      Rule::FunctionCall => {
        // For function calls, evaluate and check the result format
        if let Ok(result) = evaluate_expression(pair.clone()) {
          // If the result contains a decimal point, it's not an integer
          !result.contains('.') && result.parse::<i64>().is_ok()
        } else {
          false
        }
      }
      _ => false,
    }
  }

  let arg = &args_pairs[0];
  let is_int = is_integer_literal(arg);
  Ok(if is_int { "True" } else { "False" }.to_string())
}

/// Handle ListQ[expr] - Tests if the expression is a list
pub fn list_q(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ListQ expects exactly 1 argument".into(),
    ));
  }

  let arg = &args_pairs[0];

  // Check if it's syntactically a list
  let is_list = match arg.as_rule() {
    Rule::List => true,
    Rule::Expression => {
      let inner: Vec<_> = arg.clone().into_inner().collect();
      if inner.len() == 1 && inner[0].as_rule() == Rule::List {
        true
      } else {
        // Try evaluating and checking if result looks like a list
        if let Ok(result) = evaluate_expression(arg.clone()) {
          result.starts_with('{') && result.ends_with('}')
        } else {
          false
        }
      }
    }
    _ => {
      // Try evaluating and checking if result looks like a list
      if let Ok(result) = evaluate_expression(arg.clone()) {
        result.starts_with('{') && result.ends_with('}')
      } else {
        false
      }
    }
  };

  Ok(if is_list { "True" } else { "False" }.to_string())
}

/// Handle StringQ[expr] - Tests if the expression is a string
pub fn string_q(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StringQ expects exactly 1 argument".into(),
    ));
  }

  let arg = &args_pairs[0];

  // Check if it's syntactically a string
  let is_string = match arg.as_rule() {
    Rule::String => true,
    Rule::Expression | Rule::Term => {
      let inner: Vec<_> = arg.clone().into_inner().collect();
      if inner.len() == 1 && inner[0].as_rule() == Rule::String {
        true
      } else {
        // Check if the raw text is a quoted string
        let text = arg.as_str().trim();
        text.starts_with('"') && text.ends_with('"')
      }
    }
    _ => {
      let text = arg.as_str().trim();
      text.starts_with('"') && text.ends_with('"')
    }
  };

  Ok(if is_string { "True" } else { "False" }.to_string())
}

/// Handle AtomQ[expr] - Tests if the expression is an atomic expression (not a compound expression)
pub fn atom_q(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AtomQ expects exactly 1 argument".into(),
    ));
  }

  let arg = &args_pairs[0];

  // Atoms are: numbers, strings, and symbols (identifiers)
  // Not atoms: lists, associations, function calls with arguments
  let is_atom = match arg.as_rule() {
    Rule::Integer | Rule::Real | Rule::String | Rule::Constant => true,
    Rule::Identifier => true,
    Rule::NumericValue => true,
    Rule::List | Rule::Association => false,
    Rule::FunctionCall => false,
    Rule::Expression | Rule::Term => {
      let inner: Vec<_> = arg.clone().into_inner().collect();
      if inner.len() == 1 {
        matches!(
          inner[0].as_rule(),
          Rule::Integer
            | Rule::Real
            | Rule::String
            | Rule::Constant
            | Rule::Identifier
            | Rule::NumericValue
        )
      } else {
        false
      }
    }
    _ => false,
  };

  Ok(if is_atom { "True" } else { "False" }.to_string())
}

/// Handle PrimeQ[n] - Tests if n is a prime number
pub fn prime_q(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PrimeQ expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;

  // Must be a positive integer greater than 1
  if n.fract() != 0.0 || n < 2.0 {
    return Ok("False".to_string());
  }

  let n = n as u64;

  // Simple primality test
  if n == 2 {
    return Ok("True".to_string());
  }
  if n.is_multiple_of(2) {
    return Ok("False".to_string());
  }

  let sqrt_n = (n as f64).sqrt() as u64;
  for i in (3..=sqrt_n).step_by(2) {
    if n.is_multiple_of(i) {
      return Ok("False".to_string());
    }
  }

  Ok("True".to_string())
}

/// Handle NumericQ[expr] - Tests if the expression has a numeric value
/// NumericQ returns True for numbers and expressions that evaluate to numbers
pub fn numeric_q(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NumericQ expects exactly 1 argument".into(),
    ));
  }

  let arg = &args_pairs[0];

  // Check if it's syntactically a string (strings are not numeric)
  match arg.as_rule() {
    Rule::String => return Ok("False".to_string()),
    Rule::Expression | Rule::Term => {
      let inner: Vec<_> = arg.clone().into_inner().collect();
      if inner.len() == 1 && inner[0].as_rule() == Rule::String {
        return Ok("False".to_string());
      }
      // Check if the raw text is a quoted string
      let text = arg.as_str().trim();
      if text.starts_with('"') && text.ends_with('"') {
        return Ok("False".to_string());
      }
    }
    _ => {
      let text = arg.as_str().trim();
      if text.starts_with('"') && text.ends_with('"') {
        return Ok("False".to_string());
      }
    }
  }

  // Check if it's syntactically a list (lists are not numeric)
  match arg.as_rule() {
    Rule::List => return Ok("False".to_string()),
    Rule::Expression => {
      let inner: Vec<_> = arg.clone().into_inner().collect();
      if inner.len() == 1 && inner[0].as_rule() == Rule::List {
        return Ok("False".to_string());
      }
    }
    _ => {}
  }

  // Evaluate the argument and check if it's a number
  let result = evaluate_expression(arg.clone())?;

  // Try to parse as a number (integer or real)
  let is_numeric = result.parse::<f64>().is_ok();

  Ok(if is_numeric { "True" } else { "False" }.to_string())
}

/// Handle Positive[x] - Tests if x is a positive number (x > 0)
pub fn positive(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Positive expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;
  Ok(if n > 0.0 { "True" } else { "False" }.to_string())
}

/// Handle Negative[x] - Tests if x is a negative number (x < 0)
pub fn negative(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Negative expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;
  Ok(if n < 0.0 { "True" } else { "False" }.to_string())
}

/// Handle NonPositive[x] - Tests if x is non-positive (x ≤ 0)
pub fn non_positive(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NonPositive expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;
  Ok(if n <= 0.0 { "True" } else { "False" }.to_string())
}

/// Handle NonNegative[x] - Tests if x is non-negative (x ≥ 0)
pub fn non_negative(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NonNegative expects exactly 1 argument".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;
  Ok(if n >= 0.0 { "True" } else { "False" }.to_string())
}

/// Handle Divisible[n, m] - Tests if n is divisible by m
pub fn divisible(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Divisible expects exactly 2 arguments".into(),
    ));
  }

  let n = evaluate_term(args_pairs[0].clone())?;
  let m = evaluate_term(args_pairs[1].clone())?;

  // Both must be integers
  if n.fract() != 0.0 || m.fract() != 0.0 {
    return Ok("False".to_string());
  }

  // Division by zero is undefined
  if m == 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Divisible: division by zero".into(),
    ));
  }

  let n = n as i64;
  let m = m as i64;
  Ok(if n % m == 0 { "True" } else { "False" }.to_string())
}
