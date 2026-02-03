use pest::iterators::Pair;
use rand::Rng;

use crate::{
  InterpreterError, Rule, evaluate_expression, evaluate_term, format_result,
};

/// Handle GreaterEqual[a, b, ...] - checks if each value is greater than or equal to the next
pub fn greater_equal(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "GreaterEqual expects at least 2 arguments".into(),
    ));
  }
  let mut prev = evaluate_term(args_pairs[0].clone())?;
  for ap in args_pairs.iter().skip(1) {
    let cur = evaluate_term(ap.clone())?;
    if prev < cur {
      return Ok("False".to_string());
    }
    prev = cur;
  }
  Ok("True".to_string())
}

/// Handle LessEqual[a, b, ...] - checks if each value is less than or equal to the next
pub fn less_equal(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "LessEqual expects at least 2 arguments".into(),
    ));
  }
  let mut prev = evaluate_term(args_pairs[0].clone())?;
  for ap in args_pairs.iter().skip(1) {
    let cur = evaluate_term(ap.clone())?;
    if prev > cur {
      return Ok("False".to_string());
    }
    prev = cur;
  }
  Ok("True".to_string())
}

/// Handle Greater[a, b, ...] - checks if each value is strictly greater than the next
pub fn greater(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Greater expects at least 2 arguments".into(),
    ));
  }
  let mut prev = evaluate_term(args_pairs[0].clone())?;
  for ap in args_pairs.iter().skip(1) {
    let cur = evaluate_term(ap.clone())?;
    if prev <= cur {
      return Ok("False".to_string());
    }
    prev = cur;
  }
  Ok("True".to_string())
}

/// Handle Less[a, b, ...] - checks if each value is strictly less than the next
pub fn less(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Less expects at least 2 arguments".into(),
    ));
  }
  let mut prev = evaluate_term(args_pairs[0].clone())?;
  for ap in args_pairs.iter().skip(1) {
    let cur = evaluate_term(ap.clone())?;
    if prev >= cur {
      return Ok("False".to_string());
    }
    prev = cur;
  }
  Ok("True".to_string())
}

/// Handle Equal[a, b, ...] - checks if all values are equal
pub fn equal(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Equal expects at least 2 arguments".into(),
    ));
  }
  let ref_val = evaluate_expression(args_pairs[0].clone())?;
  for ap in args_pairs.iter().skip(1) {
    let val = evaluate_expression(ap.clone())?;
    if val != ref_val {
      return Ok("False".to_string());
    }
  }
  Ok("True".to_string())
}

/// Handle Unequal[a, b, ...] - checks if all values are different from each other
pub fn unequal(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Unequal expects at least 2 arguments".into(),
    ));
  }

  use std::collections::HashSet;
  let mut seen = HashSet::new();

  for ap in args_pairs {
    let val = evaluate_expression(ap.clone())?;
    if !seen.insert(val) {
      return Ok("False".to_string());
    }
  }

  Ok("True".to_string())
}

/// Handle Divide[a, b] - Divides the first number by the second
pub fn divide(
  args_pairs: &[Pair<Rule>],
  call_text: &str,
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    use std::io::{self, Write};
    println!(
      "\nDivide::argrx: Divide called with {} arguments; 2 arguments are expected.",
      args_pairs.len()
    );
    io::stdout().flush().ok();

    return Ok(call_text.to_string()); // return unevaluated expression
  }

  let a = evaluate_term(args_pairs[0].clone())?;
  let b = evaluate_term(args_pairs[1].clone())?;

  if b == 0.0 {
    return Err(InterpreterError::EvaluationError("Division by zero".into()));
  }

  Ok(format_result(a / b))
}

/// Handle RandomInteger[spec] or RandomInteger[{min, max}, n] - Generates random integers
pub fn random_integer(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  let mut rng = rand::thread_rng();

  match args_pairs.len() {
    0 => {
      // RandomInteger[] - returns either 0 or 1
      let random_val = rng.gen_range(0..=1);
      Ok(random_val.to_string())
    }
    1 => {
      // Either RandomInteger[n] or RandomInteger[{min, max}]
      let arg = &args_pairs[0];

      // Check if it's a list (for range) or just a number (for max)
      if arg.as_rule() == Rule::List
        || (arg.as_rule() == Rule::Expression
          && arg
            .clone()
            .into_inner()
            .next()
            .is_some_and(|p| p.as_rule() == Rule::List))
      {
        // It's a range specification: RandomInteger[{min, max}]
        let list_items = if arg.as_rule() == Rule::List {
          arg
            .clone()
            .into_inner()
            .filter(|p| p.as_str() != ",")
            .collect::<Vec<_>>()
        } else {
          arg
            .clone()
            .into_inner()
            .next()
            .unwrap()
            .into_inner()
            .filter(|p| p.as_str() != ",")
            .collect::<Vec<_>>()
        };

        if list_items.len() != 2 {
          return Err(InterpreterError::EvaluationError(
            "RandomInteger range specification must contain exactly two values"
              .into(),
          ));
        }

        let min = evaluate_term(list_items[0].clone())? as i64;
        let max = evaluate_term(list_items[1].clone())? as i64;

        if min > max {
          return Err(InterpreterError::EvaluationError(
            "RandomInteger: min value must be less than or equal to max value"
              .into(),
          ));
        }

        let random_val = rng.gen_range(min..=max);
        Ok(random_val.to_string())
      } else {
        // It's just a max value: RandomInteger[n]
        let max = evaluate_term(arg.clone())?;
        if max.fract() != 0.0 || max < 0.0 {
          return Err(InterpreterError::EvaluationError(
            "RandomInteger argument must be a non-negative integer".into(),
          ));
        }

        let max_int = max as i64;
        let random_val = if max_int == 0 {
          0
        } else {
          rng.gen_range(0..=max_int)
        };
        Ok(random_val.to_string())
      }
    }
    2 => {
      // RandomInteger[{min, max}, n] - generate n random integers in the given range
      let range = &args_pairs[0];
      let count_pair = &args_pairs[1];
      let count = evaluate_term(count_pair.clone())?;

      if count.fract() != 0.0 || count <= 0.0 {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger count must be a positive integer".into(),
        ));
      }

      // Extract min and max from the range specification
      if range.as_rule() != Rule::List
        && !(range.as_rule() == Rule::Expression
          && range
            .clone()
            .into_inner()
            .next()
            .is_some_and(|p| p.as_rule() == Rule::List))
      {
        return Err(InterpreterError::EvaluationError(
                    "First argument to RandomInteger must be a range specification {min, max}".into(),
                ));
      }

      let list_items = if range.as_rule() == Rule::List {
        range
          .clone()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect::<Vec<_>>()
      } else {
        range
          .clone()
          .into_inner()
          .next()
          .unwrap()
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .collect::<Vec<_>>()
      };

      if list_items.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger range specification must contain exactly two values"
            .into(),
        ));
      }

      let min = evaluate_term(list_items[0].clone())? as i64;
      let max = evaluate_term(list_items[1].clone())? as i64;

      if min > max {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger: min value must be less than or equal to max value"
            .into(),
        ));
      }

      // Generate the specified number of random integers
      let count = count as usize;
      let mut random_vals = Vec::with_capacity(count);

      for _ in 0..count {
        random_vals.push(rng.gen_range(min..=max).to_string());
      }

      Ok(format!("{{{}}}", random_vals.join(", ")))
    }
    _ => Err(InterpreterError::EvaluationError(
      "RandomInteger expects 0, 1, or 2 arguments".into(),
    )),
  }
}
