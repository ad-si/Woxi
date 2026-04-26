#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based Tally: count occurrences of each element.
/// Tally[{a, b, a, c, b, a}] -> {{a, 3}, {b, 2}, {c, 1}}
pub fn tally_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tally".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, (Expr, i128)> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key_str = crate::syntax::expr_to_string(item);
    if let Some((_, count)) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, (item.clone(), 1));
    }
  }

  let pairs: Vec<Expr> = order
    .into_iter()
    .map(|k| {
      let (expr, count) = counts.remove(&k).unwrap();
      Expr::List(vec![expr, Expr::Integer(count)])
    })
    .collect();

  Ok(Expr::List(pairs))
}

/// AST-based Tally with a custom equivalence test.
/// Tally[list, test] groups elements using `test[a, b]` to decide
/// whether `b` joins the group represented by `a` (the first element
/// seen for that group). Groups are reported in first-seen order.
pub fn tally_with_test_ast(
  list: &Expr,
  test: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tally".to_string(),
        args: vec![list.clone(), test.clone()],
      });
    }
  };

  // Representatives keep insertion order; parallel counts vector.
  let mut reps: Vec<Expr> = Vec::new();
  let mut counts: Vec<i128> = Vec::new();

  'outer: for item in items {
    for (i, rep) in reps.iter().enumerate() {
      let result = apply_func_to_two_args(test, rep, item)?;
      if matches!(result, Expr::Identifier(ref s) if s == "True") {
        counts[i] += 1;
        continue 'outer;
      }
    }
    reps.push(item.clone());
    counts.push(1);
  }

  let pairs: Vec<Expr> = reps
    .into_iter()
    .zip(counts)
    .map(|(rep, count)| Expr::List(vec![rep, Expr::Integer(count)]))
    .collect();

  Ok(Expr::List(pairs))
}

///// Counts[list] - Returns association of distinct elements with their counts
/// Counts[{a, b, a, c, b, a}] -> <|a -> 3, b -> 2, c -> 1|>
pub fn counts_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Counts".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, (Expr, i128)> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key_str = crate::syntax::expr_to_string(item);
    if let Some((_, count)) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, (item.clone(), 1));
    }
  }

  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let (expr, count) = counts.remove(&k).unwrap();
      (expr, Expr::Integer(count))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// AST-based DeleteDuplicates: remove duplicate elements.
/// DeleteDuplicates[{a, b, a, c, b}] -> {a, b, c}
/// DeleteDuplicates[list, test] uses `test[a, b]` to decide if `b`
/// collapses into an already-kept representative `a`.
pub fn delete_duplicates_ast(
  list: &Expr,
  test: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      let mut call_args = vec![list.clone()];
      if let Some(t) = test {
        call_args.push(t.clone());
      }
      return Ok(Expr::FunctionCall {
        name: "DeleteDuplicates".to_string(),
        args: call_args,
      });
    }
  };

  if let Some(test_fn) = test {
    // Custom equivalence: keep the first element from each equivalence
    // class, in first-seen order.
    let mut reps: Vec<Expr> = Vec::new();
    'outer: for item in items {
      for rep in &reps {
        let r = apply_func_to_two_args(test_fn, rep, item)?;
        if matches!(r, Expr::Identifier(ref s) if s == "True") {
          continue 'outer;
        }
      }
      reps.push(item.clone());
    }
    return Ok(Expr::List(reps));
  }

  use std::collections::HashSet;
  let mut seen: HashSet<String> = HashSet::new();
  let mut result = Vec::new();

  for item in items {
    let key_str = crate::syntax::expr_to_string(item);
    if seen.insert(key_str) {
      result.push(item.clone());
    }
  }

  Ok(Expr::List(result))
}

/// AST-based Union: combine lists and remove duplicates.
///
/// Supports `Union[l1, l2, ..., SameTest -> f]`. When `SameTest` is
/// specified, elements are combined and sorted first, then the result is
/// deduplicated by keeping the first occurrence of each equivalence class
/// under `f[#1, #2]`.
pub fn union_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  use std::collections::HashSet;

  // Separate list arguments from option rules like SameTest -> f.
  let mut list_args: Vec<&Expr> = Vec::new();
  let mut same_test: Option<&Expr> = None;
  for a in lists {
    match a {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "SameTest") =>
      {
        same_test = Some(replacement.as_ref());
        continue;
      }
      _ => {}
    }
    list_args.push(a);
  }

  let mut result = Vec::new();

  if same_test.is_none() {
    let mut seen: HashSet<String> = HashSet::new();
    for list in list_args.iter() {
      let items = match list {
        Expr::List(items) => items,
        _ => continue,
      };
      for item in items {
        let key_str = crate::syntax::expr_to_string(item);
        if seen.insert(key_str) {
          result.push(item.clone());
        }
      }
    }
  } else {
    // Collect every element from every list, without plain
    // deduplication -- equivalence is decided later by SameTest.
    for list in list_args.iter() {
      if let Expr::List(items) = list {
        for item in items {
          result.push(item.clone());
        }
      }
    }
  }

  // Union sorts its result in Mathematica
  result.sort_by(|a, b| {
    let ka = crate::syntax::expr_to_string(a);
    let kb = crate::syntax::expr_to_string(b);
    // Try numeric comparison first
    if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
      na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
    } else {
      ka.cmp(&kb)
    }
  });

  if let Some(test) = same_test {
    // Deduplicate by equivalence. Each sorted element is compared
    // against every already-kept representative; the first
    // representative that `test[item, rep]` is True for absorbs it.
    // The new candidate goes on the *left* of `test` to match
    // wolframscript: `Union[{1,2,3,4}, SameTest->Greater]` yields
    // `{1}` because `Greater[2,1]`, `Greater[3,1]`, `Greater[4,1]`
    // all hold, while `Less` keeps every element since
    // `Less[k, 1]` is False for every later `k`.
    let mut reps: Vec<Expr> = Vec::new();
    'outer: for item in result.into_iter() {
      for rep in &reps {
        let r = apply_func_to_two_args(test, &item, rep)?;
        if matches!(r, Expr::Identifier(ref s) if s == "True") {
          continue 'outer;
        }
      }
      reps.push(item);
    }
    return Ok(Expr::List(reps));
  }

  Ok(Expr::List(result))
}

/// AST-based Intersection: find common elements.
pub fn intersection_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  use std::collections::HashSet;

  // Start with elements from first list
  let first_items = match &lists[0] {
    Expr::List(items) => items,
    _ => return Ok(Expr::List(vec![])),
  };

  let mut common: HashSet<String> = first_items
    .iter()
    .map(crate::syntax::expr_to_string)
    .collect();

  // Intersect with each subsequent list
  for list in lists.iter().skip(1) {
    let items = match list {
      Expr::List(items) => items,
      _ => continue,
    };
    let list_set: HashSet<String> =
      items.iter().map(crate::syntax::expr_to_string).collect();
    common = common.intersection(&list_set).cloned().collect();
  }

  // Collect matching elements and sort canonically (like Mathematica)
  let mut result: Vec<Expr> = first_items
    .iter()
    .filter(|item| common.contains(&crate::syntax::expr_to_string(item)))
    .cloned()
    .collect();
  result.sort_by(|a, b| {
    let ord = compare_exprs(a, b);
    // compare_exprs returns 1 if a precedes b (canonical order), -1 if b precedes a
    if ord > 0 {
      std::cmp::Ordering::Less
    } else if ord < 0 {
      std::cmp::Ordering::Greater
    } else {
      std::cmp::Ordering::Equal
    }
  });
  result.dedup_by(|a, b| {
    crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
  });

  Ok(Expr::List(result))
}

/// AST-based Complement: elements in first list not in others.
pub fn complement_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  use std::collections::HashSet;

  // Extract items from any expression (List or FunctionCall)
  fn get_items(expr: &Expr) -> Option<(&[Expr], Option<&str>)> {
    match expr {
      Expr::List(items) => Some((items.as_slice(), None)),
      Expr::FunctionCall { name, args } => {
        Some((args.as_slice(), Some(name.as_str())))
      }
      _ => None,
    }
  }

  let (first_items, head_name) = match get_items(&lists[0]) {
    Some(r) => r,
    None => return Ok(Expr::List(vec![])),
  };

  // Get elements to exclude from all lists after the first
  let mut exclude: HashSet<String> = HashSet::new();
  for list in lists.iter().skip(1) {
    if let Some((items, _)) = get_items(list) {
      for item in items {
        exclude.insert(crate::syntax::expr_to_string(item));
      }
    }
  }

  // Filter first list, also remove duplicates and sort
  let mut seen = HashSet::new();
  let mut result: Vec<Expr> = first_items
    .iter()
    .filter(|item| {
      let s = crate::syntax::expr_to_string(item);
      !exclude.contains(&s) && seen.insert(s)
    })
    .cloned()
    .collect();

  // Sort by string representation (Wolfram sorts Complement output)
  result.sort_by(|a, b| {
    crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
  });

  match head_name {
    Some(name) => Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: result,
    }),
    None => Ok(Expr::List(result)),
  }
}

/// AST-based DeleteDuplicatesBy: remove duplicates by key function.
pub fn delete_duplicates_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteDuplicatesBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  use std::collections::HashSet;
  let mut seen: HashSet<String> = HashSet::new();
  let mut result = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if seen.insert(key_str) {
      result.push(item.clone());
    }
  }

  Ok(Expr::List(result))
}

/// DeleteAdjacentDuplicates[list] - removes consecutive duplicate elements
pub fn delete_adjacent_duplicates_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DeleteAdjacentDuplicates expects exactly 1 argument".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteAdjacentDuplicates".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let mut result = vec![items[0].clone()];
  for item in items.iter().skip(1) {
    if crate::syntax::expr_to_string(item)
      != crate::syntax::expr_to_string(result.last().unwrap())
    {
      result.push(item.clone());
    }
  }
  Ok(Expr::List(result))
}

/// Commonest[list] - returns the most common element(s)
/// Commonest[list, n] - returns the n most common elements
pub fn commonest_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Commonest expects 1 or 2 arguments".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Commonest".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let n = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n >= 1 => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Commonest".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    1
  };

  // Count occurrences, preserving order of first appearance
  let mut counts: Vec<(String, &Expr, usize)> = Vec::new();
  for item in items {
    let key = crate::syntax::expr_to_string(item);
    if let Some(entry) = counts.iter_mut().find(|(k, _, _)| k == &key) {
      entry.2 += 1;
    } else {
      counts.push((key, item, 1));
    }
  }

  // Sort by count descending (stable sort preserves insertion order for ties)
  counts.sort_by(|a, b| b.2.cmp(&a.2));

  // Take top n distinct count levels
  let mut result = Vec::new();
  let mut distinct_counts = 0;
  let mut last_count = 0;
  for (_, item, count) in &counts {
    if *count != last_count {
      distinct_counts += 1;
      if distinct_counts > n {
        break;
      }
      last_count = *count;
    }
    result.push((*item).clone());
  }

  Ok(Expr::List(result))
}

/// AST-based SymmetricDifference: elements in an odd number of the input lists.
/// Result is sorted and deduplicated.
pub fn symmetric_difference_ast(
  lists: &[Expr],
) -> Result<Expr, InterpreterError> {
  use std::collections::HashMap;

  if lists.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Count how many lists contain each element (dedup within each list)
  let mut membership_count: HashMap<String, (usize, Expr)> = HashMap::new();

  for list in lists {
    let items = match list {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "SymmetricDifference".to_string(),
          args: lists.to_vec(),
        });
      }
    };

    // Deduplicate within this list
    let mut seen_in_list = std::collections::HashSet::new();
    for item in items {
      let key = crate::syntax::expr_to_string(item);
      if seen_in_list.insert(key.clone()) {
        membership_count
          .entry(key)
          .and_modify(|(count, _)| *count += 1)
          .or_insert((1, item.clone()));
      }
    }
  }

  // Keep elements that appear in an odd number of lists
  let mut result: Vec<Expr> = membership_count
    .into_values()
    .filter(|(count, _)| count % 2 == 1)
    .map(|(_, expr)| expr)
    .collect();

  // Sort canonically (matching Complement/Union behavior)
  result.sort_by(|a, b| {
    crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
  });

  Ok(Expr::List(result))
}
