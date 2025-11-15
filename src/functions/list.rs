use pest::iterators::Pair;

use crate::{
  evaluate_expression, evaluate_term, format_result, InterpreterError, Rule,
};

/// Handle List[expr1, expr2, ...] - creates a list from elements
pub fn get_list_items<'a>(
  list_pair: &'a Pair<'a, Rule>,
) -> Result<Vec<Pair<'a, Rule>>, InterpreterError> {
  let list_rule = list_pair.as_rule();
  if list_rule == Rule::List {
    Ok(
      list_pair
        .clone()
        .into_inner()
        .filter(|p| p.as_str() != ",")
        .collect(),
    )
  } else if list_rule == Rule::Expression {
    let mut expr_inner = list_pair.clone().into_inner();
    if let Some(first) = expr_inner.next() {
      if first.as_rule() == Rule::List {
        Ok(first.into_inner().filter(|p| p.as_str() != ",").collect())
      } else {
        Err(InterpreterError::EvaluationError(
          "Argument must be a list".into(),
        ))
      }
    } else {
      Err(InterpreterError::EvaluationError(
        "Argument must be a list".into(),
      ))
    }
  } else {
    Err(InterpreterError::EvaluationError(
      "Argument must be a list".into(),
    ))
  }
}

/// Handle Map[f, list] - applies function f to each element in the list
pub fn map(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Map expects exactly 2 arguments".into(),
    ));
  }
  let func_pair = &args_pairs[0];
  let list_pair = &args_pairs[1];

  // Accept both List and Expression wrapping a List for the second argument
  let elements = get_list_items(list_pair)?;

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

/// Handle First[list] - returns the first element of a list
pub fn first(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "First expects exactly 1 argument".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  match items.first() {
    Some(item) => evaluate_expression(item.clone()),
    None => Err(InterpreterError::EvaluationError("Empty list".into())),
  }
}

/// Handle Last[list] - returns the last element of a list
pub fn last(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Last expects exactly 1 argument".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  match items.last() {
    Some(item) => evaluate_expression(item.clone()),
    None => Err(InterpreterError::EvaluationError("Empty list".into())),
  }
}

/// Handle Rest[list] - returns a list containing all but the first element
pub fn rest(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Rest expects exactly 1 argument".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  let slice: Vec<_> = if items.len() <= 1 {
    vec![]
  } else {
    items[1..].to_vec()
  };

  let evaluated: Result<Vec<_>, _> =
    slice.into_iter().map(|p| evaluate_expression(p)).collect();
  Ok(format!("{{{}}}", evaluated?.join(", ")))
}

/// Handle Most[list] - returns a list containing all but the last element
pub fn most(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Most expects exactly 1 argument".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  let slice: Vec<_> = if items.len() <= 1 {
    vec![]
  } else {
    items[..items.len() - 1].to_vec()
  };

  let evaluated: Result<Vec<_>, _> =
    slice.into_iter().map(|p| evaluate_expression(p)).collect();
  Ok(format!("{{{}}}", evaluated?.join(", ")))
}

/// Handle MemberQ[list, elem] - checks if element is in the list
pub fn member_q(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MemberQ expects exactly 2 arguments".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  // evaluate element to look for
  let target = evaluate_expression(args_pairs[1].clone())?;

  // search
  for it in items {
    if evaluate_expression(it.clone())? == target {
      return Ok("True".to_string());
    }
  }
  Ok("False".to_string())
}

/// Handle Take[list, n] - returns the first n elements of a list
pub fn take(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Take expects exactly 2 arguments".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  // second argument must be a positive integer
  let n = evaluate_term(args_pairs[1].clone())?;
  if n.fract() != 0.0 || n <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of Take must be a positive integer".into(),
    ));
  }
  let k = std::cmp::min(n as usize, items.len());
  let evaluated: Result<Vec<_>, _> = items[..k]
    .iter()
    .cloned()
    .map(|p| evaluate_expression(p))
    .collect();
  Ok(format!("{{{}}}", evaluated?.join(", ")))
}

/// Handle Drop[list, n] - returns a list with the first n elements removed
pub fn drop(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Drop expects exactly 2 arguments".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  // get n
  let n = evaluate_term(args_pairs[1].clone())?;
  if n.fract() != 0.0 || n < 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of Drop must be a non-negative integer".into(),
    ));
  }
  let start = std::cmp::min(n as usize, items.len());
  let slice = items[start..].to_vec();
  let evaluated: Result<Vec<_>, _> =
    slice.into_iter().map(|p| evaluate_expression(p)).collect();
  Ok(format!("{{{}}}", evaluated?.join(", ")))
}

/// Handle Append[list, elem] - adds an element to the end of a list
pub fn append(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Append expects exactly 2 arguments".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  // evaluate existing list
  let mut evaluated: Vec<String> = items
    .into_iter()
    .map(|p| evaluate_expression(p))
    .collect::<Result<_, _>>()?;

  // evaluate new element
  let new_elem = evaluate_expression(args_pairs[1].clone())?;
  evaluated.push(new_elem);

  Ok(format!("{{{}}}", evaluated.join(", ")))
}

/// Handle Prepend[list, elem] - adds an element to the beginning of a list
pub fn prepend(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Prepend expects exactly 2 arguments".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  // evaluate existing list
  let mut evaluated: Vec<String> = items
    .into_iter()
    .map(|p| evaluate_expression(p))
    .collect::<Result<_, _>>()?;

  // evaluate new element
  let new_elem = evaluate_expression(args_pairs[1].clone())?;
  evaluated.insert(0, new_elem);

  Ok(format!("{{{}}}", evaluated.join(", ")))
}

/// Handle Part[list, i] - returns the i-th element of a list
pub fn part(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Part expects exactly 2 arguments".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  let n = evaluate_term(args_pairs[1].clone())?;
  if n.fract() != 0.0 || n <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Second argument of Part must be a positive integer".into(),
    ));
  }
  let idx = (n as usize).checked_sub(1).ok_or_else(|| {
    InterpreterError::EvaluationError("Invalid index in Part".into())
  })?;
  if idx >= items.len() {
    return Err(InterpreterError::EvaluationError(
      "Index out of bounds in Part".into(),
    ));
  }
  evaluate_expression(items[idx].clone())
}

/// Handle Length[list] - returns the number of elements in a list
pub fn length(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Length expects exactly 1 argument".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  Ok(items.len().to_string())
}

/// Handle Reverse[list] - returns a list with elements in reverse order
pub fn reverse(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Reverse expects exactly 1 argument".into(),
    ));
  }
  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  // Reverse the items and evaluate each one
  let mut reversed_items: Vec<_> = items.into_iter().collect();
  reversed_items.reverse();

  let evaluated: Result<Vec<_>, _> = reversed_items
    .into_iter()
    .map(|p| evaluate_expression(p))
    .collect();

  Ok(format!("{{{}}}", evaluated?.join(", ")))
}

/// Handle Range[n], Range[min, max], or Range[min, max, step] - generates a sequence of numbers
pub fn range(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() || args_pairs.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Range expects 1, 2, or 3 arguments".into(),
    ));
  }

  let (start, end, step) = match args_pairs.len() {
    1 => {
      // Range[n] - generates {1, 2, ..., n}
      let n = evaluate_term(args_pairs[0].clone())?;
      if n.fract() != 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Range argument must be an integer".into(),
        ));
      }
      (1.0, n, 1.0)
    }
    2 => {
      // Range[min, max] - generates {min, min+1, ..., max}
      let min = evaluate_term(args_pairs[0].clone())?;
      let max = evaluate_term(args_pairs[1].clone())?;
      if min.fract() != 0.0 || max.fract() != 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Range arguments must be integers".into(),
        ));
      }
      (min, max, 1.0)
    }
    3 => {
      // Range[min, max, step] - generates {min, min+step, ..., max}
      let min = evaluate_term(args_pairs[0].clone())?;
      let max = evaluate_term(args_pairs[1].clone())?;
      let step = evaluate_term(args_pairs[2].clone())?;
      if min.fract() != 0.0 || max.fract() != 0.0 || step.fract() != 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Range arguments must be integers".into(),
        ));
      }
      if step == 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Range step cannot be zero".into(),
        ));
      }
      (min, max, step)
    }
    _ => unreachable!(),
  };

  // Generate the sequence
  let mut result = Vec::new();
  if step > 0.0 {
    let mut current = start;
    while current <= end {
      result.push(format_result(current));
      current += step;
    }
  } else {
    let mut current = start;
    while current >= end {
      result.push(format_result(current));
      current += step;
    }
  }

  Ok(format!("{{{}}}", result.join(", ")))
}

/// Handle Join[list1, list2, ...] - concatenates multiple lists
pub fn join(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Join expects at least 1 argument".into(),
    ));
  }

  // Collect all elements from all lists
  let mut all_elements = Vec::new();

  for arg_pair in args_pairs {
    let items = get_list_items(arg_pair)?;
    for item in items {
      let evaluated = evaluate_expression(item)?;
      all_elements.push(evaluated);
    }
  }

  Ok(format!("{{{}}}", all_elements.join(", ")))
}

/// Handle Sort[list] - sorts a list in ascending order
pub fn sort(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sort expects exactly 1 argument".into(),
    ));
  }

  let list_pair = &args_pairs[0];
  let items = get_list_items(list_pair)?;

  // Evaluate all items and try to parse as numbers
  let mut values: Vec<(f64, String)> = Vec::new();
  for item in items {
    let evaluated = evaluate_expression(item)?;
    // Try to parse as number
    if let Ok(num) = evaluated.parse::<f64>() {
      values.push((num, evaluated));
    } else {
      // If not a number, return error (matching Wolfram behavior for non-numeric sorts)
      return Err(InterpreterError::EvaluationError(
        "Sort expects numeric values".into(),
      ));
    }
  }

  // Sort by numeric value
  values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

  // Extract the formatted strings
  let sorted: Vec<String> = values.into_iter().map(|(_, s)| s).collect();

  Ok(format!("{{{}}}", sorted.join(", ")))
}
