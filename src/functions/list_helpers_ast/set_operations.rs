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
pub fn delete_duplicates_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteDuplicates".to_string(),
        args: vec![list.clone()],
      });
    }
  };

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
pub fn union_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  use std::collections::HashSet;
  let mut seen: HashSet<String> = HashSet::new();
  let mut result = Vec::new();

  for list in lists {
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
