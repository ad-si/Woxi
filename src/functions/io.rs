use pest::iterators::Pair;

use crate::{InterpreterError, Rule, capture_stdout, evaluate_expression};

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
        if first.as_rule() == Rule::String {
          // Same unescaping for string inside expression
          let raw_str = first.as_str();
          if let Ok(unescaped) = snailquote::unescape(raw_str) {
            unescaped
          } else {
            raw_str.trim_matches('"').to_string()
          }
        } else {
          evaluate_expression(arg_pair.clone())?
        }
      } else {
        evaluate_expression(arg_pair.clone())?
      }
    }
    _ => evaluate_expression(arg_pair.clone())?,
  };
  // Print to stdout for normal display
  println!("{}", arg_str);
  // Also capture for Jupyter
  capture_stdout(&arg_str);
  Ok("Null".to_string())
}
