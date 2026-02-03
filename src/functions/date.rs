use chrono::Local;
use pest::iterators::Pair;

use crate::{InterpreterError, Rule, evaluate_expression, extract_string};

/// Handle DateString[]/DateString[Now]/DateString["format"] - returns a formatted date string
pub fn date_string(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  let current_time = Local::now();
  let default_format = "%a, %d %b %Y %H:%M:%S"; // e.g., Mon, 29 Jul 2024 10:30:00

  match args_pairs.len() {
        0 => { // DateString[]
            Ok(current_time.format(default_format).to_string())
        }
        1 => { // DateString[Now] or DateString["format"]
            let arg1_eval_result = evaluate_expression(args_pairs[0].clone());
            match arg1_eval_result {
                Ok(arg1_val) if arg1_val == "CURRENT_TIME_MARKER" => { // DateString[Now]
                    Ok(current_time.format(default_format).to_string())
                }
                _ => { // DateString["format_string"] - evaluate_expression might fail if not a string
                    let format_str = extract_string(args_pairs[0].clone())?;
                    match format_str.as_str() {
                        "ISODateTime" => Ok(current_time.format("%Y-%m-%dT%H:%M:%S").to_string()),
                        _ => Ok(current_time.format(&format_str).to_string()), // Attempt direct chrono format
                    }
                }
            }
        }
        2 => { // DateString[date_spec, "format_string"]
            let date_spec_eval = evaluate_expression(args_pairs[0].clone())?;
            if date_spec_eval != "CURRENT_TIME_MARKER" {
                return Err(InterpreterError::EvaluationError(
                    "DateString: First argument currently must be Now.".into(),
                ));
            }
            // It's DateString[Now, "format_string"]
            let format_str = extract_string(args_pairs[1].clone())?;
            match format_str.as_str() {
                "ISODateTime" => Ok(current_time.format("%Y-%m-%dT%H:%M:%S").to_string()),
                _ => Ok(current_time.format(&format_str).to_string()), // Attempt direct chrono format
            }
        }
        _ => Err(InterpreterError::EvaluationError(
            "DateString: Called with invalid number of arguments. Expected 0, 1, or 2.".into(),
        )),
    }
}
