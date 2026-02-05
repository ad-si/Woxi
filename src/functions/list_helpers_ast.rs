//! AST-based list helper functions.
//!
//! These functions work directly with `Expr` AST nodes, avoiding the string
//! round-trips and re-parsing that the original `list_helpers.rs` functions use.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Convert an Expr to a boolean value.
/// Returns Some(true) for Identifier("True"), Some(false) for Identifier("False").
pub fn expr_to_bool(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Identifier(s) if s == "True" => Some(true),
    Expr::Identifier(s) if s == "False" => Some(false),
    _ => None,
  }
}

/// Convert a boolean to an Expr.
pub fn bool_to_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}

/// Apply a function/predicate to an argument and return the resulting Expr.
/// Uses the existing apply_function_to_arg from evaluator.
pub fn apply_func_ast(
  func: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  crate::evaluator::apply_function_to_arg(func, arg)
}

/// AST-based Map: apply function to each element of a list or association.
/// Map[f, {a, b, c}] -> {f[a], f[b], f[c]}
/// Map[f, <|a -> 1, b -> 2|>] -> <|a -> f[1], b -> f[2]|>
pub fn map_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_func_ast(func, item))
        .collect();
      Ok(Expr::List(results?))
    }
    Expr::Association(items) => {
      // Map over association applies function to values only
      let results: Result<Vec<(Expr, Expr)>, InterpreterError> = items
        .iter()
        .map(|(key, val)| {
          let new_val = apply_func_ast(func, val)?;
          Ok((key.clone(), new_val))
        })
        .collect();
      Ok(Expr::Association(results?))
    }
    _ => {
      // Not a list or association, return unevaluated
      Ok(Expr::FunctionCall {
        name: "Map".to_string(),
        args: vec![func.clone(), list.clone()],
      })
    }
  }
}

/// AST-based Select: filter elements where predicate returns True.
/// Select[{a, b, c}, pred] -> elements where pred[elem] is True
pub fn select_ast(list: &Expr, pred: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Select".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  let mut kept = Vec::new();
  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      kept.push(item.clone());
    }
  }

  Ok(Expr::List(kept))
}

/// AST-based AllTrue: check if predicate is true for all elements.
/// AllTrue[{a, b, c}, pred] -> True if pred[x] is True for all x
pub fn all_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AllTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) != Some(true) {
      return Ok(bool_to_expr(false));
    }
  }

  Ok(bool_to_expr(true))
}

/// AST-based AnyTrue: check if predicate is true for any element.
/// AnyTrue[{a, b, c}, pred] -> True if pred[x] is True for any x
pub fn any_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AnyTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      return Ok(bool_to_expr(true));
    }
  }

  Ok(bool_to_expr(false))
}

/// AST-based NoneTrue: check if predicate is false for all elements.
/// NoneTrue[{a, b, c}, pred] -> True if pred[x] is False for all x
pub fn none_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NoneTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      return Ok(bool_to_expr(false));
    }
  }

  Ok(bool_to_expr(true))
}

/// AST-based Fold/FoldList: fold a function over a list.
/// Fold[f, x, {a, b, c}] -> f[f[f[x, a], b], c]
pub fn fold_ast(
  func: &Expr,
  init: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Fold".to_string(),
        args: vec![func.clone(), init.clone(), list.clone()],
      });
    }
  };

  let mut acc = init.clone();
  for item in items {
    // Apply func[acc, item]
    acc = apply_func_to_two_args(func, &acc, item)?;
  }

  Ok(acc)
}

/// Apply a binary function to two arguments.
fn apply_func_to_two_args(
  func: &Expr,
  arg1: &Expr,
  arg2: &Expr,
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => crate::evaluator::evaluate_function_call_ast(
      name,
      &[arg1.clone(), arg2.clone()],
    ),
    Expr::Function { body } => {
      // Anonymous function with two slots
      let substituted =
        crate::syntax::substitute_slots(body, &[arg1.clone(), arg2.clone()]);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args } => {
      // Curried function: f[a] applied to (b, c) becomes f[a, b, c]
      let mut new_args = args.clone();
      new_args.push(arg1.clone());
      new_args.push(arg2.clone());
      crate::evaluator::evaluate_function_call_ast(name, &new_args)
    }
    _ => {
      let func_str = crate::syntax::expr_to_string(func);
      crate::evaluator::evaluate_function_call_ast(
        &func_str,
        &[arg1.clone(), arg2.clone()],
      )
    }
  }
}

/// AST-based CountBy: count elements by the value of a function.
/// CountBy[{a, b, c}, f] -> association of f[x] -> count
pub fn count_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CountBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, i128> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if let Some(count) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, 1);
    }
  }

  // Build association preserving order
  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let count = counts[&k];
      let key_expr = crate::syntax::string_to_expr(&k).unwrap_or(Expr::Raw(k));
      (key_expr, Expr::Integer(count))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// AST-based GroupBy: group elements by the value of a function.
/// GroupBy[{a, b, c}, f] -> association of f[x] -> {elements with that f value}
pub fn group_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "GroupBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut groups: HashMap<String, Vec<Expr>> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if let Some(group) = groups.get_mut(&key_str) {
      group.push(item.clone());
    } else {
      order.push(key_str.clone());
      groups.insert(key_str, vec![item.clone()]);
    }
  }

  // Build association preserving order
  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let items = groups.remove(&k).unwrap();
      let key_expr = crate::syntax::string_to_expr(&k).unwrap_or(Expr::Raw(k));
      (key_expr, Expr::List(items))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// AST-based SortBy: sort elements by the value of a function.
/// SortBy[{a, b, c}, f] -> elements sorted by f[x]
pub fn sort_by_ast(list: &Expr, func: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SortBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  // Compute keys for each element
  let mut keyed: Vec<(Expr, Expr)> = items
    .into_iter()
    .map(|item| {
      let key = apply_func_ast(func, &item)?;
      Ok((item, key))
    })
    .collect::<Result<_, InterpreterError>>()?;

  // Sort by key (using string representation for comparison)
  keyed.sort_by(|a, b| {
    let ka = crate::syntax::expr_to_string(&a.1);
    let kb = crate::syntax::expr_to_string(&b.1);
    // Try numeric comparison first
    if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
      na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
    } else {
      ka.cmp(&kb)
    }
  });

  Ok(Expr::List(
    keyed.into_iter().map(|(item, _)| item).collect(),
  ))
}

/// AST-based Nest: apply a function n times.
/// Nest[f, x, n] -> f[f[f[...f[x]...]]] (n times)
pub fn nest_ast(
  func: &Expr,
  init: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Nest requires non-negative count".into(),
    ));
  }

  let mut result = init.clone();
  for _ in 0..n {
    result = apply_func_ast(func, &result)?;
  }

  Ok(result)
}

/// AST-based NestList: build a list by repeatedly applying a function.
/// NestList[f, x, n] -> {x, f[x], f[f[x]], ..., f^n[x]}
pub fn nest_list_ast(
  func: &Expr,
  init: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "NestList requires non-negative count".into(),
    ));
  }

  let mut results = vec![init.clone()];
  let mut current = init.clone();
  for _ in 0..n {
    current = apply_func_ast(func, &current)?;
    results.push(current.clone());
  }

  Ok(Expr::List(results))
}

/// AST-based FixedPoint: apply function until result stops changing.
/// FixedPoint[f, x] -> fixed point of f starting from x
pub fn fixed_point_ast(
  func: &Expr,
  init: &Expr,
  max_iterations: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut current = init.clone();

  for _ in 0..max {
    let next = apply_func_ast(func, &current)?;
    let current_str = crate::syntax::expr_to_string(&current);
    let next_str = crate::syntax::expr_to_string(&next);
    if current_str == next_str {
      return Ok(current);
    }
    current = next;
  }

  Ok(current)
}

/// AST-based Cases: select elements matching a pattern.
pub fn cases_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Cases".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let mut kept = Vec::new();
  for item in items {
    // For now, use string-based pattern matching
    // This is a simplification; full pattern matching is complex
    let item_str = crate::syntax::expr_to_string(item);
    let pattern_str = crate::syntax::expr_to_string(pattern);

    // Simple pattern check
    if matches_pattern_simple(&item_str, &pattern_str) {
      kept.push(item.clone());
    }
  }

  Ok(Expr::List(kept))
}

/// Simple pattern matching for Cases.
/// This handles basic patterns like x_, _Integer, etc.
fn matches_pattern_simple(value: &str, pattern: &str) -> bool {
  // Match any value
  if pattern == "_" {
    return true;
  }

  // Named blank pattern like x_
  if pattern.ends_with('_')
    && !pattern.contains("_Integer")
    && !pattern.contains("_Real")
  {
    return true;
  }

  // Type patterns
  if pattern == "_Integer" {
    return value.parse::<i128>().is_ok();
  }
  if pattern == "_Real" {
    return value.parse::<f64>().is_ok() && value.contains('.');
  }
  if pattern == "_String" {
    return value.starts_with('"') && value.ends_with('"');
  }
  if pattern == "_List" {
    return value.starts_with('{') && value.ends_with('}');
  }

  // Literal match
  value == pattern
}

/// AST-based Position: find positions of elements matching a pattern.
pub fn position_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Position".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let pattern_str = crate::syntax::expr_to_string(pattern);
  let mut positions = Vec::new();

  for (i, item) in items.iter().enumerate() {
    let item_str = crate::syntax::expr_to_string(item);
    if matches_pattern_simple(&item_str, &pattern_str) {
      // 1-indexed
      positions.push(Expr::List(vec![Expr::Integer((i + 1) as i128)]));
    }
  }

  Ok(Expr::List(positions))
}

/// AST-based MapIndexed: apply function with index to each element.
/// MapIndexed[f, {a, b, c}] -> {f[a, {1}], f[b, {2}], f[c, {3}]}
pub fn map_indexed_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapIndexed".to_string(),
        args: vec![func.clone(), list.clone()],
      });
    }
  };

  let results: Result<Vec<Expr>, _> = items
    .iter()
    .enumerate()
    .map(|(i, item)| {
      let index = Expr::List(vec![Expr::Integer((i + 1) as i128)]);
      apply_func_to_two_args(func, item, &index)
    })
    .collect();

  Ok(Expr::List(results?))
}

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

  // Preserve order from first list
  let result: Vec<Expr> = first_items
    .iter()
    .filter(|item| common.contains(&crate::syntax::expr_to_string(item)))
    .cloned()
    .collect();

  Ok(Expr::List(result))
}

/// AST-based Complement: elements in first list not in others.
pub fn complement_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  use std::collections::HashSet;

  // Get elements to exclude from all lists after the first
  let mut exclude: HashSet<String> = HashSet::new();
  for list in lists.iter().skip(1) {
    let items = match list {
      Expr::List(items) => items,
      _ => continue,
    };
    for item in items {
      exclude.insert(crate::syntax::expr_to_string(item));
    }
  }

  // Filter first list
  let first_items = match &lists[0] {
    Expr::List(items) => items,
    _ => return Ok(Expr::List(vec![])),
  };

  let result: Vec<Expr> = first_items
    .iter()
    .filter(|item| !exclude.contains(&crate::syntax::expr_to_string(item)))
    .cloned()
    .collect();

  Ok(Expr::List(result))
}
