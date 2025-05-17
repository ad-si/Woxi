use pest::iterators::Pair;

use crate::{evaluate_expression, evaluate_term, InterpreterError, Rule};

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
  let is_even = n >= 0.0 && (n as i64) % 2 == 0;
  Ok(
    if (func_name == "EvenQ" && is_even) || (func_name == "OddQ" && !is_even) {
      "True"
    } else {
      "False"
    }
    .to_string(),
  )
}
