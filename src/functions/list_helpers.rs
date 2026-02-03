use pest::iterators::Pair;

use crate::{
  InterpreterError, Rule, evaluate_expression, evaluate_term, format_fraction,
  format_real_result, format_result, functions::boolean::as_bool, interpret,
  parse_list_string,
};

/// Check if a list item is a real (floating-point) number
fn item_is_real(pair: &Pair<Rule>) -> bool {
  let s = pair.as_str();
  // Check if the string contains a decimal point (indicating a real number)
  s.contains('.') && !s.contains("->")
}

pub fn map_list(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Map expects exactly 2 arguments".into(),
    ));
  }
  let func_pair = &args_pairs[0];
  let target_pair = &args_pairs[1];

  // Check if the second argument is an association
  if let Ok(assoc) = crate::functions::association::get_assoc_from_first_arg(&[
    target_pair.clone(),
  ]) {
    // Map over association values
    let func_src = func_pair.as_str();
    let mut mapped_pairs = Vec::new();

    for (key, value) in assoc {
      let mapped_val = if func_pair.as_rule() == Rule::AnonymousFunction
        || (func_src.contains('#') && func_src.ends_with('&'))
      {
        // Handle anonymous functions like #^2&
        let mut expr = func_src.trim_end_matches('&').to_string();
        expr = expr.replace('#', &value);
        interpret(&expr)?
      } else {
        // Handle named functions
        match func_src {
          "Sign" => {
            let num = value.parse::<f64>().map_err(|_| {
              InterpreterError::EvaluationError(
                "Map currently supports only numeric values".into(),
              )
            })?;
            let sign = if num > 0.0 {
              1.0
            } else if num < 0.0 {
              -1.0
            } else {
              0.0
            };
            format_result(sign)
          }
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "Unknown mapping function: {}",
              func_src
            )));
          }
        }
      };
      mapped_pairs.push((key, mapped_val));
    }

    // Format as association
    let disp = format!(
      "<|{}|>",
      mapped_pairs
        .iter()
        .map(|(k, v)| format!("{} -> {}", k, v))
        .collect::<Vec<_>>()
        .join(", ")
    );
    return Ok(disp);
  }

  // Original list handling
  let list_rule = target_pair.as_rule();
  let elements: Vec<_> = if list_rule == Rule::List {
    target_pair
      .clone()
      .into_inner()
      .filter(|p| p.as_str() != ",")
      .collect()
  } else if list_rule == Rule::Expression {
    let mut expr_inner = target_pair.clone().into_inner();
    if let Some(first) = expr_inner.next() {
      if first.as_rule() == Rule::List {
        first.into_inner().filter(|p| p.as_str() != ",").collect()
      } else {
        return Err(InterpreterError::EvaluationError(
          "Second argument of Map must be a list or association".into(),
        ));
      }
    } else {
      return Err(InterpreterError::EvaluationError(
        "Second argument of Map must be a list or association".into(),
      ));
    }
  } else {
    return Err(InterpreterError::EvaluationError(
      "Second argument of Map must be a list or association".into(),
    ));
  };

  let func_name_inner = func_pair.as_str();
  let mut mapped = Vec::new();

  for elem in elements {
    let elem_val_str = evaluate_expression(elem.clone())?;
    let num = elem_val_str.parse::<f64>().map_err(|_| {
      InterpreterError::EvaluationError(
        "Map currently supports only numeric list elements".into(),
      )
    })?;

    let mapped_val = match func_name_inner {
      "Sign" => {
        let sign = if num > 0.0 {
          1.0
        } else if num < 0.0 {
          -1.0
        } else {
          0.0
        };
        format_result(sign)
      }
      _ => {
        return Err(InterpreterError::EvaluationError(format!(
          "Unknown mapping function: {}",
          func_name_inner
        )));
      }
    };
    mapped.push(mapped_val);
  }
  Ok(format!("{{{}}}", mapped.join(", ")))
}

/// Handle AllTrue[list, pred] - Check if predicate is true for all elements in list
pub fn all_true(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ---------- arity ----------------------------------------------------
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "AllTrue expects exactly 2 arguments".into(),
    ));
  }

  // ---------- obtain list items (accept real list OR a value that
  // ---------- evaluates to a displayed list string) ---------------
  let list_pair = &args_pairs[0];
  let elements: Vec<String> = match list_pair.as_rule() {
    // syntactic list
    Rule::List => list_pair
      .clone()
      .into_inner()
      .filter(|p| p.as_str() != ",")
      .map(|p| evaluate_expression(p))
      .collect::<Result<_, _>>()?,
    // expression that syntactically contains a list
    Rule::Expression => {
      let mut expr_inner = list_pair.clone().into_inner();
      if let Some(first) = expr_inner.next() {
        if first.as_rule() == Rule::List && expr_inner.next().is_none() {
          first
            .into_inner()
            .filter(|p| p.as_str() != ",")
            .map(|p| evaluate_expression(p))
            .collect::<Result<_, _>>()?
        } else {
          // fall back to runtime evaluation
          let val = evaluate_expression(list_pair.clone())?;
          parse_list_string(&val).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "First argument of AllTrue must be a list".into(),
            )
          })?
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "First argument of AllTrue must be a list".into(),
        ));
      }
    }
    // any other form – evaluate, then try to parse "{…}"
    _ => {
      let val = evaluate_expression(list_pair.clone())?;
      parse_list_string(&val).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "First argument of AllTrue must be a list".into(),
        )
      })?
    }
  };

  // ---------- identify predicate --------------------------------------
  let pred_pair = &args_pairs[1];
  let pred_src = pred_pair.as_str();
  let is_slot_pred = pred_pair.as_rule() == Rule::AnonymousFunction
    || (pred_src.contains('#') && pred_src.ends_with('&'));

  // ---------- test every element --------------------------------------
  for elem_str in elements {
    let passes = if is_slot_pred {
      // substitute # and evaluate
      let mut expr = pred_src.trim_end_matches('&').to_string();
      expr = expr.replace('#', &elem_str);
      let res = interpret(&expr)?;
      as_bool(&res).unwrap_or(false)
    } else {
      match pred_src {
        "EvenQ" | "OddQ" => {
          let expr = format!("{}[{}]", pred_src, elem_str);
          let res = interpret(&expr)?;
          as_bool(&res).unwrap_or(false)
        }
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "Unknown predicate function: {}",
            pred_src
          )));
        }
      }
    };
    if !passes {
      return Ok("False".to_string());
    }
  }
  Ok("True".to_string())
}

/// Handle Select[list, pred] - Return elements in list for which predicate is true
pub fn select(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  // ----- arity ---------------------------------------------------------
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Select expects exactly 2 arguments".into(),
    ));
  }

  // ----- extract list --------------------------------------------------
  let list_pair = &args_pairs[0];
  let list_rule = list_pair.as_rule();
  let elems: Vec<_> = if list_rule == Rule::List {
    list_pair
      .clone()
      .into_inner()
      .filter(|p| p.as_str() != ",")
      .collect()
  } else if list_rule == Rule::Expression {
    let mut expr_inner = list_pair.clone().into_inner();
    if let Some(first) = expr_inner.next() {
      if first.as_rule() == Rule::List {
        first.into_inner().filter(|p| p.as_str() != ",").collect()
      } else {
        return Err(InterpreterError::EvaluationError(
          "First argument of Select must be a list".into(),
        ));
      }
    } else {
      return Err(InterpreterError::EvaluationError(
        "First argument of Select must be a list".into(),
      ));
    }
  } else {
    return Err(InterpreterError::EvaluationError(
      "First argument of Select must be a list".into(),
    ));
  };

  // ----- identify predicate -------------------------------------------
  let pred_pair = &args_pairs[1];
  let pred_src = pred_pair.as_str();
  let is_slot_pred = pred_pair.as_rule() == Rule::AnonymousFunction
    || (pred_src.contains('#') && pred_src.ends_with('&'));

  // ----- filter --------------------------------------------------------
  let mut kept = Vec::new();
  for elem in elems {
    let passes = if is_slot_pred {
      // build expression by substituting the Slot (#) with the element’s
      // evaluated value and dropping the trailing ‘&’
      let mut expr = pred_src.trim_end_matches('&').to_string();
      let elem_str = evaluate_expression(elem.clone())?;
      expr = expr.replace('#', &elem_str);
      // evaluate the resulting Wolfram-expression
      let res = interpret(&expr)?;
      as_bool(&res).unwrap_or(false)
    } else {
      match pred_src {
        "EvenQ" | "OddQ" => {
          let n = evaluate_term(elem.clone())?;
          if n.fract() != 0.0 {
            false
          } else {
            let even = (n as i64) % 2 == 0;
            if pred_src == "EvenQ" { even } else { !even }
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "Unknown predicate function: {}",
            pred_src
          )));
        }
      }
    };
    if passes {
      kept.push(evaluate_expression(elem.clone())?);
    }
  }
  Ok(format!("{{{}}}", kept.join(", ")))
}

/// Handle Flatten[list] - Flatten nested lists into a single list
pub fn flatten(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Flatten expects exactly 1 argument".into(),
    ));
  }

  // Get the list from the argument
  let list_pair = &args_pairs[0];
  let list_text = evaluate_expression(list_pair.clone())?;

  // Check if it's in the form of a list string {a, b, ...}
  if !list_text.starts_with('{') || !list_text.ends_with('}') {
    return Err(InterpreterError::EvaluationError(
      "Argument to Flatten must be a list".into(),
    ));
  }

  // Function to recursively flatten nested lists
  fn flatten_list_string(s: &str) -> Result<Vec<String>, InterpreterError> {
    // Remove outer braces and split by commas
    let inner = &s[1..s.len() - 1];
    let mut result = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    // Parse the list string manually to handle nested lists
    for (i, c) in inner.char_indices() {
      match c {
        '{' => depth += 1,
        '}' => depth -= 1,
        ',' if depth == 0 => {
          let part = inner[start..i].trim();
          if !part.is_empty() {
            if part.starts_with('{') && part.ends_with('}') {
              // Recursively flatten nested list
              let nested = flatten_list_string(part)?;
              result.extend(nested);
            } else {
              result.push(part.to_string());
            }
          }
          start = i + 1;
        }
        _ => {}
      }
    }

    // Handle the last part
    let last_part = inner[start..].trim();
    if !last_part.is_empty() {
      if last_part.starts_with('{') && last_part.ends_with('}') {
        let nested = flatten_list_string(last_part)?;
        result.extend(nested);
      } else {
        result.push(last_part.to_string());
      }
    }

    Ok(result)
  }

  let flattened = flatten_list_string(&list_text)?;
  Ok(format!("{{{}}}", flattened.join(", ")))
}

/// Handle Total[list] - Sum all elements in a list
pub fn total(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Total expects exactly 1 argument".into(),
    ));
  }

  // Get the list from the argument
  let list_pair = &args_pairs[0];
  let items = crate::functions::list::get_list_items(list_pair)?;

  let mut sum = 0.0;
  for item in items {
    let val = evaluate_term(item.clone())?;
    sum += val;
  }

  Ok(format_result(sum))
}

/// Handle Mean[list] - Calculate the arithmetic mean (average) of a list
pub fn mean(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Mean expects exactly 1 argument".into(),
    ));
  }

  // Get the list from the argument
  let list_pair = &args_pairs[0];
  let items = crate::functions::list::get_list_items(list_pair)?;

  if items.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Mean of an empty list is undefined".into(),
    ));
  }

  // Check if any item is a real number
  let has_real = items.iter().any(item_is_real);

  let mut sum = 0.0;
  let mut int_sum: i64 = 0;
  for item in &items {
    let val = evaluate_term(item.clone())?;
    sum += val;
    if !has_real && val.fract() == 0.0 {
      int_sum += val as i64;
    }
  }

  if has_real {
    // Return as real number
    let mean_val = sum / items.len() as f64;
    Ok(format_result(mean_val))
  } else {
    // Return as fraction if not an integer
    let count = items.len() as i64;
    Ok(format_fraction(int_sum, count))
  }
}

/// Handle Product[list] or Product[expr, {i, min, max}] - Multiply list elements or iterate
pub fn product(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() == 1 {
    // Product[{a, b, c}] - product of list elements
    let list_pair = &args_pairs[0];
    let items = crate::functions::list::get_list_items(list_pair)?;

    if items.is_empty() {
      // Product of empty list is 1 (multiplicative identity)
      return Ok("1".to_string());
    }

    let mut product = 1.0;
    for item in items {
      let val = evaluate_term(item.clone())?;
      product *= val;
    }

    return Ok(format_result(product));
  }

  if args_pairs.len() == 2 {
    // Product[expr, {i, min, max}] - iterator form
    let expr_pair = &args_pairs[0];
    let iter_spec_pair = &args_pairs[1];

    // Parse the iterator specification {i, min, max}
    let iter_items = crate::functions::list::get_list_items(iter_spec_pair)?;
    if iter_items.len() < 2 || iter_items.len() > 3 {
      return Err(InterpreterError::EvaluationError(
        "Product iterator specification must be {var, max} or {var, min, max}"
          .into(),
      ));
    }

    // Get the iterator variable name
    let var_name = iter_items[0].as_str().to_string();

    // Get min and max - check if they are symbolic or numeric
    let (min_str, max_str) = if iter_items.len() == 2 {
      ("1".to_string(), iter_items[1].as_str().to_string())
    } else {
      (
        iter_items[1].as_str().to_string(),
        iter_items[2].as_str().to_string(),
      )
    };

    // Check if bounds are symbolic (contain non-numeric identifiers)
    let min_is_numeric = min_str.parse::<f64>().is_ok();
    let max_is_numeric = max_str.parse::<f64>().is_ok();

    if !min_is_numeric || !max_is_numeric {
      // Symbolic case - return symbolic representation
      // For Product[i^2, {i, 1, n}], return n!^2
      // This is a specific pattern match for common cases
      let expr_text = expr_pair.as_str().trim();
      if min_str == "1" && expr_text == format!("{}^2", var_name) {
        // Product[i^2, {i, 1, n}] = (n!)^2
        return Ok(format!("{}!^2", max_str));
      }
      // For other symbolic cases, return unevaluated
      return Ok(format!(
        "Product[{}, {{{}, {}, {}}}]",
        expr_text, var_name, min_str, max_str
      ));
    }

    let min_val: f64 = min_str.parse().unwrap();
    let max_val: f64 = max_str.parse().unwrap();

    if min_val.fract() != 0.0 || max_val.fract() != 0.0 {
      return Err(InterpreterError::EvaluationError(
        "Product iterator bounds must be integers".into(),
      ));
    }

    // Get the expression text
    let expr_text = expr_pair.as_str();

    // Calculate the product
    let mut product = 1.0;
    let start = min_val as i64;
    let end = max_val as i64;

    for i in start..=end {
      // Substitute the iterator variable in the expression
      let substituted = expr_text.replace(&var_name, &i.to_string());
      let value = interpret(&substituted)?;
      let num: f64 = value.parse().map_err(|_| {
        InterpreterError::EvaluationError(format!(
          "Product: expression did not evaluate to a number: {}",
          value
        ))
      })?;
      product *= num;
    }

    return Ok(format_result(product));
  }

  Err(InterpreterError::EvaluationError(
    "Product expects 1 or 2 arguments".into(),
  ))
}

/// Handle Accumulate[list] - Returns cumulative sums of a list
pub fn accumulate(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Accumulate expects exactly 1 argument".into(),
    ));
  }

  // Get the list from the argument
  let list_pair = &args_pairs[0];
  let items = crate::functions::list::get_list_items(list_pair)?;

  if items.is_empty() {
    // Accumulate of empty list is empty list
    return Ok("{}".to_string());
  }

  // Check if any item is a real number
  let has_real = items.iter().any(item_is_real);

  let mut cumulative_sum = 0.0;
  let mut result = Vec::new();

  for item in items {
    let val = evaluate_term(item.clone())?;
    cumulative_sum += val;
    if has_real {
      result.push(format_real_result(cumulative_sum));
    } else {
      result.push(format_result(cumulative_sum));
    }
  }

  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle Differences[list] - Returns successive differences between elements
pub fn differences(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Differences expects exactly 1 argument".into(),
    ));
  }

  // Get the list from the argument
  let list_pair = &args_pairs[0];
  let items = crate::functions::list::get_list_items(list_pair)?;

  if items.len() <= 1 {
    // Differences of empty list or single element is empty list
    return Ok("{}".to_string());
  }

  let mut values: Vec<f64> = Vec::new();
  for item in items {
    let val = evaluate_term(item.clone())?;
    values.push(val);
  }

  let mut result = Vec::new();
  for i in 1..values.len() {
    let diff = values[i] - values[i - 1];
    result.push(format_result(diff));
  }

  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle Median[list] - Returns the median value of a list
pub fn median(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Median expects exactly 1 argument".into(),
    ));
  }

  // Get the list from the argument
  let list_pair = &args_pairs[0];
  let items = crate::functions::list::get_list_items(list_pair)?;

  if items.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Median of an empty list is undefined".into(),
    ));
  }

  // Check if any item is a real number
  let has_real = items.iter().any(item_is_real);

  // Evaluate all items to numbers
  let mut values: Vec<f64> = Vec::new();
  for item in items {
    let val = evaluate_term(item.clone())?;
    values.push(val);
  }

  // Sort the values
  values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

  // Calculate median
  let len = values.len();
  if len % 2 == 1 {
    // Odd length: return middle element
    let median_val = values[len / 2];
    if has_real {
      Ok(format_real_result(median_val))
    } else {
      Ok(format_result(median_val))
    }
  } else {
    // Even length: return average of two middle elements
    let mid1 = values[len / 2 - 1];
    let mid2 = values[len / 2];

    if has_real {
      let median_val = (mid1 + mid2) / 2.0;
      Ok(format_real_result(median_val))
    } else {
      // For integers, return as fraction
      let sum = (mid1 as i64) + (mid2 as i64);
      Ok(format_fraction(sum, 2))
    }
  }
}

/// Handle First[list] - Return the first element of a list
/// Handle Last[list] - Return the last element of a list
pub fn first_or_last(
  func_name: &str,
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects exactly 1 argument",
      func_name
    )));
  }
  let list_pair = &args_pairs[0];
  // Accept both List and Expression wrapping a List
  let list_rule = list_pair.as_rule();
  let items: Vec<_> = if list_rule == Rule::List {
    list_pair
      .clone()
      .into_inner()
      .filter(|p| p.as_str() != ",")
      .collect()
  } else if list_rule == Rule::Expression {
    let mut expr_inner = list_pair.clone().into_inner();
    if let Some(first) = expr_inner.next() {
      if first.as_rule() == Rule::List {
        first.into_inner().filter(|p| p.as_str() != ",").collect()
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{} function argument must be a list",
          func_name
        )));
      }
    } else {
      return Err(InterpreterError::EvaluationError(format!(
        "{} function argument must be a list",
        func_name
      )));
    }
  } else {
    return Err(InterpreterError::EvaluationError(format!(
      "{} function argument must be a list",
      func_name
    )));
  };
  let target = if func_name == "First" {
    items.first()
  } else {
    items.last()
  };
  match target {
    Some(item) => evaluate_expression(item.clone()),
    _ => Err(InterpreterError::EvaluationError("Empty list".into())),
  }
}

/// Handle Rest[list] - Return all but the first element of a list
/// Handle Most[list] - Return all but the last element of a list
pub fn rest_or_most(
  func_name: &str,
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects exactly 1 argument",
      func_name
    )));
  }
  // extract list items exactly like the First/Last implementation
  let list_pair = &args_pairs[0];
  let list_rule = list_pair.as_rule();
  let items: Vec<_> = if list_rule == Rule::List {
    list_pair
      .clone()
      .into_inner()
      .filter(|p| p.as_str() != ",")
      .collect()
  } else if list_rule == Rule::Expression {
    let mut expr_inner = list_pair.clone().into_inner();
    if let Some(first) = expr_inner.next() {
      if first.as_rule() == Rule::List {
        first.into_inner().filter(|p| p.as_str() != ",").collect()
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{} function argument must be a list",
          func_name
        )));
      }
    } else {
      return Err(InterpreterError::EvaluationError(format!(
        "{} function argument must be a list",
        func_name
      )));
    }
  } else {
    return Err(InterpreterError::EvaluationError(format!(
      "{} function argument must be a list",
      func_name
    )));
  };
  let slice: Vec<_> = if func_name == "Rest" {
    if items.len() <= 1 {
      vec![]
    } else {
      items[1..].to_vec()
    }
  } else {
    // Most
    if items.len() <= 1 {
      vec![]
    } else {
      items[..items.len() - 1].to_vec()
    }
  };
  let evaluated: Result<Vec<_>, _> =
    slice.into_iter().map(|p| evaluate_expression(p)).collect();
  Ok(format!("{{{}}}", evaluated?.join(", ")))
}

/// Handle Cases[list, pattern] - Extract elements matching a pattern
pub fn cases(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Cases expects exactly 2 arguments".into(),
    ));
  }

  let list_pair = &args_pairs[0];
  let pattern_pair = &args_pairs[1];
  let pattern = evaluate_expression(pattern_pair.clone())?;

  let items = crate::functions::list::get_list_items(list_pair)?;
  let mut result = Vec::new();

  for item in items {
    let item_str = evaluate_expression(item.clone())?;
    if item_str == pattern {
      result.push(item_str);
    }
  }

  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle DeleteCases[list, pattern] - Remove elements matching a pattern
pub fn delete_cases(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "DeleteCases expects exactly 2 arguments".into(),
    ));
  }

  let list_pair = &args_pairs[0];
  let pattern_pair = &args_pairs[1];
  let pattern = evaluate_expression(pattern_pair.clone())?;

  let items = crate::functions::list::get_list_items(list_pair)?;
  let mut result = Vec::new();

  for item in items {
    let item_str = evaluate_expression(item.clone())?;
    if item_str != pattern {
      result.push(item_str);
    }
  }

  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle MapThread[f, {list1, list2, ...}] - Apply function to corresponding elements
pub fn map_thread(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MapThread expects exactly 2 arguments".into(),
    ));
  }

  let func_pair = &args_pairs[0];
  let lists_pair = &args_pairs[1];
  let func_name = func_pair.as_str();

  // Get the outer list containing the sublists
  let outer_items = crate::functions::list::get_list_items(lists_pair)?;
  if outer_items.is_empty() {
    return Ok("{}".to_string());
  }

  // Get each sublist
  let mut sublists: Vec<Vec<String>> = Vec::new();
  for item in outer_items {
    let sublist_items = crate::functions::list::get_list_items(&item)?;
    let evaluated: Result<Vec<_>, _> = sublist_items
      .into_iter()
      .map(|p| evaluate_expression(p))
      .collect();
    sublists.push(evaluated?);
  }

  // Check all sublists have the same length
  if sublists.is_empty() {
    return Ok("{}".to_string());
  }
  let len = sublists[0].len();
  for sublist in &sublists {
    if sublist.len() != len {
      return Err(InterpreterError::EvaluationError(
        "MapThread: all lists must have the same length".into(),
      ));
    }
  }

  // Apply function to corresponding elements
  let mut result = Vec::new();
  for i in 0..len {
    let args: Vec<String> = sublists.iter().map(|sl| sl[i].clone()).collect();
    let expr = format!("{}[{}]", func_name, args.join(", "));
    let val = interpret(&expr)?;
    result.push(val);
  }

  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle Partition[list, n] - Break list into sublists of length n
pub fn partition(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Partition expects exactly 2 arguments".into(),
    ));
  }

  let list_pair = &args_pairs[0];
  let n = evaluate_term(args_pairs[1].clone())?;

  if n.fract() != 0.0 || n <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of Partition must be a positive integer".into(),
    ));
  }
  let n_int = n as usize;

  let items = crate::functions::list::get_list_items(list_pair)?;
  let evaluated: Result<Vec<_>, _> =
    items.into_iter().map(|p| evaluate_expression(p)).collect();
  let evaluated = evaluated?;

  // Partition into chunks of size n, discarding incomplete final chunk
  let mut result = Vec::new();
  for chunk in evaluated.chunks(n_int) {
    if chunk.len() == n_int {
      result.push(format!("{{{}}}", chunk.join(", ")));
    }
  }

  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle SortBy[list, f] - Sort list elements by applying function f
pub fn sort_by(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SortBy expects exactly 2 arguments".into(),
    ));
  }

  let list_pair = &args_pairs[0];
  let func_pair = &args_pairs[1];
  let func_src = func_pair.as_str();

  let items = crate::functions::list::get_list_items(list_pair)?;
  let evaluated: Result<Vec<_>, _> =
    items.into_iter().map(|p| evaluate_expression(p)).collect();
  let evaluated = evaluated?;

  // Check if it's an identity function (# &)
  let is_identity = func_src.trim() == "# &" || func_src.trim() == "#&";

  // Compute sort keys for each element
  let mut keyed: Vec<(String, f64)> = Vec::new();
  for elem in evaluated {
    let key_val = if is_identity {
      // Identity function: use the element itself as the key
      elem.parse::<f64>().map_err(|_| {
        InterpreterError::EvaluationError(
          "SortBy: element must be numeric when using identity function".into(),
        )
      })?
    } else if func_src.contains('#') && func_src.ends_with('&') {
      // Anonymous function
      let mut expr = func_src.trim_end_matches('&').to_string();
      expr = expr.replace('#', &elem);
      let res = interpret(&expr)?;
      res.parse::<f64>().map_err(|_| {
        InterpreterError::EvaluationError(
          "SortBy: function must return a numeric value".into(),
        )
      })?
    } else {
      // Named function
      let expr = format!("{}[{}]", func_src, elem);
      let res = interpret(&expr)?;
      res.parse::<f64>().map_err(|_| {
        InterpreterError::EvaluationError(
          "SortBy: function must return a numeric value".into(),
        )
      })?
    };
    keyed.push((elem, key_val));
  }

  // Sort by key
  keyed
    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

  let sorted: Vec<String> = keyed.into_iter().map(|(e, _)| e).collect();
  Ok(format!("{{{}}}", sorted.join(", ")))
}

/// Handle GroupBy[list, f] - Group elements by applying function f
pub fn group_by(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "GroupBy expects exactly 2 arguments".into(),
    ));
  }

  let list_pair = &args_pairs[0];
  let func_pair = &args_pairs[1];
  let func_src = func_pair.as_str();

  let items = crate::functions::list::get_list_items(list_pair)?;
  let evaluated: Result<Vec<_>, _> =
    items.into_iter().map(|p| evaluate_expression(p)).collect();
  let evaluated = evaluated?;

  // Group elements by key
  let mut groups: Vec<(String, Vec<String>)> = Vec::new();

  for elem in evaluated {
    // Compute the key
    let key = if func_src.contains('#') && func_src.ends_with('&') {
      // Anonymous function
      let mut expr = func_src.trim_end_matches('&').to_string();
      expr = expr.replace('#', &elem);
      interpret(&expr)?
    } else {
      // Named function
      let expr = format!("{}[{}]", func_src, elem);
      interpret(&expr)?
    };

    // Find or create group
    if let Some(group) = groups.iter_mut().find(|(k, _)| *k == key) {
      group.1.push(elem);
    } else {
      groups.push((key, vec![elem]));
    }
  }

  // Format as association
  let parts: Vec<String> = groups
    .into_iter()
    .map(|(k, v)| format!("{} -> {{{}}}", k, v.join(", ")))
    .collect();

  Ok(format!("<|{}|>", parts.join(", ")))
}

/// Handle Array[f, n] - Construct array by applying function to indices 1 through n
pub fn array(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Array expects exactly 2 arguments".into(),
    ));
  }

  let func_pair = &args_pairs[0];
  let n = evaluate_term(args_pairs[1].clone())?;

  if n.fract() != 0.0 || n <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of Array must be a positive integer".into(),
    ));
  }
  let n_int = n as usize;

  let func_src = func_pair.as_str();
  let mut result = Vec::new();

  for i in 1..=n_int {
    let val = if func_src.contains('#') && func_src.ends_with('&') {
      // Anonymous function
      let mut expr = func_src.trim_end_matches('&').to_string();
      expr = expr.replace('#', &i.to_string());
      interpret(&expr)?
    } else {
      // Named function
      let expr = format!("{}[{}]", func_src, i);
      interpret(&expr)?
    };
    result.push(val);
  }

  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle Fold[f, init, list] - Apply function cumulatively to list elements
/// Fold[f, x, {a, b, c}] -> f[f[f[x, a], b], c]
pub fn fold(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Fold expects exactly 3 arguments".into(),
    ));
  }

  let func_pair = &args_pairs[0];
  let init = evaluate_expression(args_pairs[1].clone())?;
  let list_pair = &args_pairs[2];

  // Get list elements
  let elements = get_list_elements(list_pair)?;

  let func_src = func_pair.as_str();
  let mut accumulator = init;

  for elem_str in elements {
    // Apply function: f[accumulator, element]
    let result = apply_binary_function(func_src, &accumulator, &elem_str)?;
    accumulator = result;
  }

  Ok(accumulator)
}

/// Handle FoldList[f, init, list] - Like Fold but returns list of intermediate results
/// FoldList[f, x, {a, b, c}] -> {x, f[x, a], f[f[x, a], b], f[f[f[x, a], b], c]}
pub fn fold_list(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "FoldList expects exactly 3 arguments".into(),
    ));
  }

  let func_pair = &args_pairs[0];
  let init = evaluate_expression(args_pairs[1].clone())?;
  let list_pair = &args_pairs[2];

  // Get list elements
  let elements = get_list_elements(list_pair)?;

  let func_src = func_pair.as_str();
  let mut results = vec![init.clone()];
  let mut accumulator = init;

  for elem_str in elements {
    // Apply function: f[accumulator, element]
    let result = apply_binary_function(func_src, &accumulator, &elem_str)?;
    results.push(result.clone());
    accumulator = result;
  }

  Ok(format!("{{{}}}", results.join(", ")))
}

/// Handle Nest[f, expr, n] - Apply function n times
/// Nest[f, x, 3] -> f[f[f[x]]]
pub fn nest(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Nest expects exactly 3 arguments".into(),
    ));
  }

  let func_pair = &args_pairs[0];
  let init = evaluate_expression(args_pairs[1].clone())?;
  let n = evaluate_term(args_pairs[2].clone())?;

  if n.fract() != 0.0 || n < 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Third argument of Nest must be a non-negative integer".into(),
    ));
  }
  let n_int = n as usize;

  let func_src = func_pair.as_str();
  let mut result = init;

  for _ in 0..n_int {
    result = apply_unary_function(func_src, &result)?;
  }

  Ok(result)
}

/// Handle NestList[f, expr, n] - Like Nest but returns list of intermediate results
/// NestList[f, x, 3] -> {x, f[x], f[f[x]], f[f[f[x]]]}
pub fn nest_list(
  args_pairs: &[Pair<Rule>],
) -> Result<String, InterpreterError> {
  if args_pairs.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "NestList expects exactly 3 arguments".into(),
    ));
  }

  let func_pair = &args_pairs[0];
  let init = evaluate_expression(args_pairs[1].clone())?;
  let n = evaluate_term(args_pairs[2].clone())?;

  if n.fract() != 0.0 || n < 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Third argument of NestList must be a non-negative integer".into(),
    ));
  }
  let n_int = n as usize;

  let func_src = func_pair.as_str();
  let mut results = vec![init.clone()];
  let mut current = init;

  for _ in 0..n_int {
    current = apply_unary_function(func_src, &current)?;
    results.push(current.clone());
  }

  Ok(format!("{{{}}}", results.join(", ")))
}

/// Helper function to get list elements from a pair
fn get_list_elements(
  list_pair: &Pair<Rule>,
) -> Result<Vec<String>, InterpreterError> {
  let list_rule = list_pair.as_rule();

  if list_rule == Rule::List {
    list_pair
      .clone()
      .into_inner()
      .filter(|p| p.as_str() != ",")
      .map(|p| evaluate_expression(p))
      .collect()
  } else if list_rule == Rule::Expression {
    let mut expr_inner = list_pair.clone().into_inner();
    if let Some(first) = expr_inner.next() {
      if first.as_rule() == Rule::List {
        first
          .into_inner()
          .filter(|p| p.as_str() != ",")
          .map(|p| evaluate_expression(p))
          .collect()
      } else {
        Err(InterpreterError::EvaluationError("Expected a list".into()))
      }
    } else {
      Err(InterpreterError::EvaluationError("Expected a list".into()))
    }
  } else {
    // Try to evaluate and parse the result as a list
    let list_str = evaluate_expression(list_pair.clone())?;
    if list_str.starts_with('{') && list_str.ends_with('}') {
      let inner = &list_str[1..list_str.len() - 1];
      // Simple parsing for comma-separated values
      Ok(parse_list_elements(inner))
    } else {
      Err(InterpreterError::EvaluationError("Expected a list".into()))
    }
  }
}

/// Parse comma-separated list elements (handling nested structures)
fn parse_list_elements(s: &str) -> Vec<String> {
  let mut elements = Vec::new();
  let mut current = String::new();
  let mut depth = 0;

  for c in s.chars() {
    match c {
      '{' | '[' | '(' => {
        depth += 1;
        current.push(c);
      }
      '}' | ']' | ')' => {
        depth -= 1;
        current.push(c);
      }
      ',' if depth == 0 => {
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
          elements.push(trimmed);
        }
        current.clear();
      }
      _ => current.push(c),
    }
  }

  let trimmed = current.trim().to_string();
  if !trimmed.is_empty() {
    elements.push(trimmed);
  }

  elements
}

/// Apply a binary function f[a, b]
fn apply_binary_function(
  func_src: &str,
  a: &str,
  b: &str,
) -> Result<String, InterpreterError> {
  if func_src.contains('#') && func_src.ends_with('&') {
    // Anonymous function with slots
    let mut expr = func_src.trim_end_matches('&').to_string();
    expr = expr.replace("#1", a).replace("#2", b);
    // If there's still a plain # without number, replace with first arg
    expr = expr.replace('#', a);
    interpret(&expr)
  } else {
    // Named function - return symbolic form if function is unknown
    let expr = format!("{}[{}, {}]", func_src, a, b);
    match interpret(&expr) {
      Ok(result) => Ok(result),
      Err(InterpreterError::EvaluationError(e))
        if e.starts_with("Unknown function:") =>
      {
        // Return the symbolic unevaluated form
        Ok(expr)
      }
      Err(e) => Err(e),
    }
  }
}

/// Apply a unary function f[x]
fn apply_unary_function(
  func_src: &str,
  x: &str,
) -> Result<String, InterpreterError> {
  if func_src.contains('#') && func_src.ends_with('&') {
    // Anonymous function
    let mut expr = func_src.trim_end_matches('&').to_string();
    expr = expr.replace('#', x);
    interpret(&expr)
  } else {
    // Named function - return symbolic form if function is unknown
    let expr = format!("{}[{}]", func_src, x);
    match interpret(&expr) {
      Ok(result) => Ok(result),
      Err(InterpreterError::EvaluationError(e))
        if e.starts_with("Unknown function:") =>
      {
        // Return the symbolic unevaluated form
        Ok(expr)
      }
      Err(e) => Err(e),
    }
  }
}
