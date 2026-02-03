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
