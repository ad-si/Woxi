use pest::iterators::Pair;

use crate::{
  InterpreterError, Rule, capture_stdout, evaluate_expression,
  evaluator::evaluate_function_call,
};

pub fn print(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // 0 args → just output a newline and return Null
  if args_pairs.is_empty() {
    println!(); // visible newline for CLI tests
    capture_stdout(""); // keep Jupyter stdout behaviour
    return Ok("Null".to_string());
  }
  // 1 arg accepted, anything else is still an error
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Print expects at most 1 argument".into(),
    ));
  }
  // Accept string, or expression wrapping a string, or any printable value
  let arg_pair = &args_pairs[0];
  let arg_str = match arg_pair.as_rule() {
    Rule::String => {
      // Get the raw string with quotes
      let raw_str = arg_pair.as_str();
      // Unescape the string content properly
      if let Ok(unescaped) = snailquote::unescape(raw_str) {
        unescaped
      } else {
        // Fall back to trimming quotes if unescaping fails
        raw_str.trim_matches('"').to_string()
      }
    }
    Rule::Expression => {
      let mut expr_inner = arg_pair.clone().into_inner();
      if let Some(first) = expr_inner.next() {
        // Check if it's a single-child expression (no more children)
        if expr_inner.next().is_none() {
          // Fast path for common cases
          match first.as_rule() {
            Rule::String => {
              // Same unescaping for string inside expression
              let raw_str = first.as_str();
              if let Ok(unescaped) = snailquote::unescape(raw_str) {
                unescaped
              } else {
                raw_str.trim_matches('"').to_string()
              }
            }
            Rule::FunctionCall => {
              // Directly evaluate function call, skipping expression overhead
              evaluate_function_call(first)?
            }
            _ => evaluate_expression(arg_pair.clone())?,
          }
        } else {
          // Multi-child expression (has operators)
          evaluate_expression(arg_pair.clone())?
        }
      } else {
        evaluate_expression(arg_pair.clone())?
      }
    }
    _ => evaluate_expression(arg_pair.clone())?,
  };
  // If the result is a quoted string, strip the quotes for display
  let display_str =
    if arg_str.starts_with('"') && arg_str.ends_with('"') && arg_str.len() >= 2
    {
      if let Ok(unescaped) = snailquote::unescape(&arg_str) {
        unescaped
      } else {
        arg_str[1..arg_str.len() - 1].to_string()
      }
    } else {
      arg_str
    };
  // Print to stdout for normal display
  println!("{}", display_str);
  // Also capture for Jupyter
  capture_stdout(&display_str);
  Ok("Null".to_string())
}
