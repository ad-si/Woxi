use pest::iterators::Pair;

use crate::{
  evaluate_expression, evaluate_term, format_result,
  functions::boolean::as_bool, interpret, parse_list_string, InterpreterError,
  Rule,
};

pub fn map_list(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Map expects exactly 2 arguments".into(),
    ));
  }
  let func_pair = &args_pairs[0];
  let list_pair = &args_pairs[1];

  // Accept both List and Expression wrapping a List for the second argument
  let list_rule = list_pair.as_rule();
  let elements: Vec<_> = if list_rule == Rule::List {
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
          "Second argument of Map must be a list".into(),
        ));
      }
    } else {
      return Err(InterpreterError::EvaluationError(
        "Second argument of Map must be a list".into(),
      ));
    }
  } else {
    return Err(InterpreterError::EvaluationError(
      "Second argument of Map must be a list".into(),
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
        )))
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
          )))
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
            if pred_src == "EvenQ" {
              even
            } else {
              !even
            }
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "Unknown predicate function: {}",
            pred_src
          )))
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
