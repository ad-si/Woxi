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

///// Ordering[list
pub fn ordering_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Ordering expects 1 or 2 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Ordering".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut indexed: Vec<(usize, &Expr)> = items.iter().enumerate().collect();

  indexed.sort_by(|a, b| {
    let va = crate::syntax::expr_to_string(a.1);
    let vb = crate::syntax::expr_to_string(b.1);
    if let (Ok(na), Ok(nb)) = (va.parse::<f64>(), vb.parse::<f64>()) {
      na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
    } else {
      va.cmp(&vb)
    }
  });

  let mut result: Vec<Expr> = indexed
    .iter()
    .map(|(idx, _)| Expr::Integer((*idx + 1) as i128))
    .collect();

  if args.len() == 2
    && let Expr::Integer(n) = &args[1]
  {
    let n = *n;
    if n >= 0 {
      result.truncate(n as usize);
    } else {
      // Negative n: take last |n| elements (largest positions)
      let abs_n = n.unsigned_abs() as usize;
      if abs_n <= result.len() {
        result = result.split_off(result.len() - abs_n);
      }
    }
  }

  Ok(Expr::List(result))
}

/// MinimalBy[list, f] - Returns all elements that minimize f
pub fn minimal_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => return Ok(Expr::List(vec![])),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MinimalBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  let keyed: Vec<(Expr, Expr)> = items
    .iter()
    .map(|item| {
      let key = apply_func_ast(func, item)?;
      Ok((item.clone(), key))
    })
    .collect::<Result<_, InterpreterError>>()?;

  let min_key = keyed
    .iter()
    .map(|(_, k)| k)
    .min_by(|a, b| {
      let ka = crate::syntax::expr_to_string(a);
      let kb = crate::syntax::expr_to_string(b);
      if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
      } else {
        ka.cmp(&kb)
      }
    })
    .cloned();

  if let Some(min_k) = min_key {
    let min_str = crate::syntax::expr_to_string(&min_k);
    let result: Vec<Expr> = keyed
      .into_iter()
      .filter(|(_, k)| crate::syntax::expr_to_string(k) == min_str)
      .map(|(item, _)| item)
      .collect();
    Ok(Expr::List(result))
  } else {
    Ok(Expr::List(vec![]))
  }
}

/// MaximalBy[list, f] - Returns all elements that maximize f
pub fn maximal_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => return Ok(Expr::List(vec![])),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MaximalBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  let keyed: Vec<(Expr, Expr)> = items
    .iter()
    .map(|item| {
      let key = apply_func_ast(func, item)?;
      Ok((item.clone(), key))
    })
    .collect::<Result<_, InterpreterError>>()?;

  let max_key = keyed
    .iter()
    .map(|(_, k)| k)
    .max_by(|a, b| {
      let ka = crate::syntax::expr_to_string(a);
      let kb = crate::syntax::expr_to_string(b);
      if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
      } else {
        ka.cmp(&kb)
      }
    })
    .cloned();

  if let Some(max_k) = max_key {
    let max_str = crate::syntax::expr_to_string(&max_k);
    let result: Vec<Expr> = keyed
      .into_iter()
      .filter(|(_, k)| crate::syntax::expr_to_string(k) == max_str)
      .map(|(item, _)| item)
      .collect();
    Ok(Expr::List(result))
  } else {
    Ok(Expr::List(vec![]))
  }
}

/// MapAt[f, list, pos] - Apply function at specific positions
/// Supports single integer, list of integers, and negative indices
pub fn map_at_ast(
  func: &Expr,
  list: &Expr,
  pos_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapAt".to_string(),
        args: vec![func.clone(), list.clone(), pos_spec.clone()],
      });
    }
  };

  let len = items.len() as i128;

  // Collect positions to modify
  // Wolfram uses {{1}, {3}} for multiple positions (list of single-element lists)
  let positions: Vec<i128> = match pos_spec {
    Expr::Integer(n) => vec![*n],
    Expr::List(pos_list) => {
      // Each element must be a single-element list like {1}
      let mut positions = Vec::new();
      for p in pos_list {
        match p {
          Expr::List(inner) if inner.len() == 1 => {
            if let Expr::Integer(n) = &inner[0] {
              positions.push(*n);
            }
          }
          _ => {
            return Ok(Expr::FunctionCall {
              name: "MapAt".to_string(),
              args: vec![func.clone(), list.clone(), pos_spec.clone()],
            });
          }
        }
      }
      positions
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapAt".to_string(),
        args: vec![func.clone(), list.clone(), pos_spec.clone()],
      });
    }
  };

  let mut indices = std::collections::HashSet::new();
  for p in positions {
    let idx = if p < 0 {
      (len + p) as usize
    } else {
      (p - 1) as usize
    };
    if idx < items.len() {
      indices.insert(idx);
    }
  }

  let result: Result<Vec<Expr>, _> = items
    .iter()
    .enumerate()
    .map(|(i, item)| {
      if indices.contains(&i) {
        apply_func_ast(func, item)
      } else {
        Ok(item.clone())
      }
    })
    .collect();

  Ok(Expr::List(result?))
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
    if matches_pattern_ast(item, pattern) {
      kept.push(item.clone());
    } else {
      // Fall back to string matching for compatibility
      let item_str = crate::syntax::expr_to_string(item);
      let pattern_str = crate::syntax::expr_to_string(pattern);
      if matches_pattern_simple(&item_str, &pattern_str) {
        kept.push(item.clone());
      }
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

/// AST-based pattern matching for expressions.
/// Supports: Blank (_), named patterns (x_), head patterns (_Integer, _List, etc.),
/// Except, Alternatives, and literal matching.
pub fn matches_pattern_ast(expr: &Expr, pattern: &Expr) -> bool {
  match pattern {
    // Blank pattern: _ matches anything
    Expr::Pattern {
      name: _,
      head: None,
    } => true,
    // Head-constrained pattern: _Integer, _List, etc.
    Expr::Pattern {
      name: _,
      head: Some(h),
    } => get_expr_head_str(expr) == h,
    // Identifier patterns like "_", "_Integer", "_List", etc.
    Expr::Identifier(s) if s == "_" => true,
    Expr::Identifier(s) if s.starts_with('_') => {
      let head = &s[1..];
      get_expr_head_str(expr) == head
    }
    // Except[pattern] - matches anything that doesn't match the inner pattern
    Expr::FunctionCall { name, args }
      if name == "Except" && args.len() == 1 =>
    {
      !matches_pattern_ast(expr, &args[0])
    }
    // Alternatives: a | b - matches if either side matches
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      left,
      right,
    } => matches_pattern_ast(expr, left) || matches_pattern_ast(expr, right),
    // Structural matching for lists: {_, _} matches {1, 2}
    Expr::List(pat_items) => {
      if let Expr::List(expr_items) = expr {
        pat_items.len() == expr_items.len()
          && pat_items
            .iter()
            .zip(expr_items.iter())
            .all(|(p, e)| matches_pattern_ast(e, p))
      } else {
        false
      }
    }
    // Structural matching for function calls: f[_
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } => {
      if pat_name == "Except"
        || pat_name == "PatternTest"
        || pat_name == "Condition"
      {
        // Already handled above or not a structural match
        let pattern_str = crate::syntax::expr_to_string(pattern);
        let expr_str = crate::syntax::expr_to_string(expr);
        expr_str == pattern_str
      } else if let Expr::FunctionCall {
        name: expr_name,
        args: expr_args,
      } = expr
      {
        pat_name == expr_name
          && pat_args.len() == expr_args.len()
          && pat_args
            .iter()
            .zip(expr_args.iter())
            .all(|(p, e)| matches_pattern_ast(e, p))
      } else {
        false
      }
    }
    // Literal comparison
    _ => {
      let pattern_str = crate::syntax::expr_to_string(pattern);
      let expr_str = crate::syntax::expr_to_string(expr);
      expr_str == pattern_str
    }
  }
}

/// Get the head of an expression as a string
fn get_expr_head_str(expr: &Expr) -> &str {
  match expr {
    Expr::Integer(_) => "Integer",
    Expr::Real(_) => "Real",
    Expr::String(_) => "String",
    Expr::List(_) => "List",
    Expr::FunctionCall { name, .. } => name,
    Expr::Association(_) => "Association",
    _ => "Symbol",
  }
}

/// Cases with level specification: Cases[list, pattern, {level}]
pub fn cases_with_level_ast(
  list: &Expr,
  pattern: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Parse level spec: {n} means exactly level n
  let level = match level_spec {
    Expr::List(items) if items.len() == 1 => match &items[0] {
      Expr::Integer(n) => *n as usize,
      _ => 1,
    },
    _ => 1,
  };

  let mut results = Vec::new();
  collect_at_level(list, pattern, level, 0, &mut results);
  Ok(Expr::List(results))
}

/// Recursively collect elements matching pattern at a specific level
fn collect_at_level(
  expr: &Expr,
  pattern: &Expr,
  target_level: usize,
  current_level: usize,
  results: &mut Vec<Expr>,
) {
  if current_level == target_level {
    if matches_pattern_ast(expr, pattern) {
      results.push(expr.clone());
    }
    return;
  }

  // Recurse into sublists/subexpressions
  match expr {
    Expr::List(items) => {
      for item in items {
        collect_at_level(
          item,
          pattern,
          target_level,
          current_level + 1,
          results,
        );
      }
    }
    Expr::FunctionCall { args, .. } => {
      for arg in args {
        collect_at_level(
          arg,
          pattern,
          target_level,
          current_level + 1,
          results,
        );
      }
    }
    _ => {}
  }
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

/// AST-based Table: generate a table of values.
/// Table[expr, {i, min, max}] -> {expr with i=min, ..., expr with i=max}
/// Table[expr, {i, max}] -> {expr with i=1, ..., expr with i=max}
/// Table[expr, {i, {list}}] -> {expr with i=elem1, expr with i=elem2, ...}
/// Table[expr, n] -> {expr, expr, ..., expr} (n times)
/// Multi-dimensional Table: Table[expr, iter1, iter2, ...]
/// Recursively nests Table from last iterator to first.
pub fn table_multi_ast(
  body: &Expr,
  iters: &[Expr],
) -> Result<Expr, InterpreterError> {
  if iters.len() == 1 {
    return table_ast(body, &iters[0]);
  }
  // Nest: the body for the outer iterator is Table[body, inner_iters...]
  // We build from the innermost outward
  let inner_body = Expr::FunctionCall {
    name: "Table".to_string(),
    args: std::iter::once(body.clone())
      .chain(iters[1..].iter().cloned())
      .collect(),
  };
  table_ast(&inner_body, &iters[0])
}

pub fn table_ast(
  body: &Expr,
  iter_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  match iter_spec {
    Expr::Integer(n) => {
      // Simple form: Table[expr, n]
      if *n < 0 {
        return Err(InterpreterError::EvaluationError(
          "Table: count must be non-negative".into(),
        ));
      }
      let mut results = Vec::new();
      for _ in 0..*n {
        let val = crate::evaluator::evaluate_expr_to_expr(body)?;
        results.push(val);
      }
      Ok(Expr::List(results))
    }
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::List(vec![]));
      }

      // Handle {n} form (single element = just repeat count, no variable)
      if items.len() == 1 {
        let evaluated = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
        let n = expr_to_i128(&evaluated).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Table: iterator bound must be an integer".into(),
          )
        })?;
        let mut results = Vec::new();
        for _ in 0..n {
          let val = crate::evaluator::evaluate_expr_to_expr(body)?;
          results.push(val);
        }
        return Ok(Expr::List(results));
      }

      // Extract iterator variable
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Table: iterator variable must be an identifier".into(),
          ));
        }
      };

      if items.len() == 2 {
        // Check if second element is a list (iterate over list)
        let second = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        match second {
          Expr::List(list_items) => {
            // {i, {a, b, c}} form - iterate over list elements
            let mut results = Vec::new();
            for item in list_items {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &item);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
            }
            return Ok(Expr::List(results));
          }
          _ => {
            // {i, max} form - iterate from 1 to max
            let max_val = expr_to_i128(&second).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "Table: iterator bound must be an integer".into(),
              )
            })?;
            let mut results = Vec::new();
            for i in 1..=max_val {
              let substituted = crate::syntax::substitute_variable(
                body,
                &var_name,
                &Expr::Integer(i),
              );
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
            }
            return Ok(Expr::List(results));
          }
        }
      } else if items.len() >= 3 {
        // {i, min, max} or {i, min, max, step} form
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
        let min_val = expr_to_i128(&min_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Table: iterator bound must be an integer".into(),
          )
        })?;
        let max_val = expr_to_i128(&max_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Table: iterator bound must be an integer".into(),
          )
        })?;

        // Get step (default is 1)
        let step_val = if items.len() >= 4 {
          let step_expr = crate::evaluator::evaluate_expr_to_expr(&items[3])?;
          expr_to_i128(&step_expr).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Table: step must be an integer".into(),
            )
          })?
        } else {
          1
        };

        if step_val == 0 {
          return Err(InterpreterError::EvaluationError(
            "Table: step cannot be zero".into(),
          ));
        }

        let mut results = Vec::new();
        let mut i = min_val;
        if step_val > 0 {
          while i <= max_val {
            let substituted = crate::syntax::substitute_variable(
              body,
              &var_name,
              &Expr::Integer(i),
            );
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            results.push(val);
            i += step_val;
          }
        } else {
          // Negative step
          while i >= max_val {
            let substituted = crate::syntax::substitute_variable(
              body,
              &var_name,
              &Expr::Integer(i),
            );
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            results.push(val);
            i += step_val;
          }
        }
        return Ok(Expr::List(results));
      }

      Err(InterpreterError::EvaluationError(
        "Table: invalid iterator specification".into(),
      ))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Table: invalid iterator specification".into(),
    )),
  }
}

/// Helper to extract i128 from Expr
fn expr_to_i128(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::Real(f) if f.fract() == 0.0 => Some(*f as i128),
    _ => None,
  }
}

/// AST-based MapThread: apply function to corresponding elements.
/// MapThread[f, {{a, b}, {c, d}}] -> {f[a, c], f[b, d]}
pub fn map_thread_ast(
  func: &Expr,
  lists: &Expr,
) -> Result<Expr, InterpreterError> {
  let outer_items = match lists {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapThread".to_string(),
        args: vec![func.clone(), lists.clone()],
      });
    }
  };

  if outer_items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Get each sublist
  let mut sublists: Vec<Vec<Expr>> = Vec::new();
  for item in outer_items {
    match item {
      Expr::List(items) => sublists.push(items.clone()),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "MapThread: second argument must be a list of lists".into(),
        ));
      }
    }
  }

  // Check all sublists have the same length
  let len = sublists[0].len();
  for sublist in &sublists {
    if sublist.len() != len {
      return Err(InterpreterError::EvaluationError(
        "MapThread: all lists must have the same length".into(),
      ));
    }
  }

  // Apply function to corresponding elements
  let mut results = Vec::new();
  for i in 0..len {
    let args: Vec<Expr> = sublists.iter().map(|sl| sl[i].clone()).collect();
    let result = apply_func_to_n_args(func, &args)?;
    results.push(result);
  }

  Ok(Expr::List(results))
}

/// Apply a function to n arguments.
fn apply_func_to_n_args(
  func: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, args)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, args);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args: fa } => {
      let mut new_args = fa.clone();
      new_args.extend(args.iter().cloned());
      crate::evaluator::evaluate_function_call_ast(name, &new_args)
    }
    _ => {
      let func_str = crate::syntax::expr_to_string(func);
      crate::evaluator::evaluate_function_call_ast(&func_str, args)
    }
  }
}

/// AST-based Partition: break list into sublists of length n.
/// Partition[{a, b, c, d, e}, 2] -> {{a, b}, {c, d}}
pub fn partition_ast(list: &Expr, n: i128) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Partition".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  if n <= 0 {
    return Err(InterpreterError::EvaluationError(
      "Partition: size must be positive".into(),
    ));
  }

  let n_usize = n as usize;
  let mut results = Vec::new();
  for chunk in items.chunks(n_usize) {
    if chunk.len() == n_usize {
      results.push(Expr::List(chunk.to_vec()));
    }
  }

  Ok(Expr::List(results))
}

/// AST-based First: return first element of list.
pub fn first_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      if items.is_empty() {
        Err(InterpreterError::EvaluationError(
          "First: list is empty".into(),
        ))
      } else {
        Ok(items[0].clone())
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "First".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// AST-based Last: return last element of list.
pub fn last_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      if items.is_empty() {
        Err(InterpreterError::EvaluationError(
          "Last: list is empty".into(),
        ))
      } else {
        Ok(items[items.len() - 1].clone())
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Last".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// AST-based Rest: return all but first element.
pub fn rest_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      if items.is_empty() {
        Err(InterpreterError::EvaluationError(
          "Rest: list is empty".into(),
        ))
      } else {
        Ok(Expr::List(items[1..].to_vec()))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Rest".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// AST-based Most: return all but last element.
pub fn most_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      if items.is_empty() {
        Err(InterpreterError::EvaluationError(
          "Most: list is empty".into(),
        ))
      } else {
        Ok(Expr::List(items[..items.len() - 1].to_vec()))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Most".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// AST-based Take: take first n elements.
/// Returns unevaluated if n exceeds list length (to let fallback handle error).
pub fn take_ast(list: &Expr, n: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  let count = match n {
    Expr::Integer(i) => *i,
    Expr::Real(f) if f.fract() == 0.0 => *f as i128,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  let len = items.len() as i128;
  if count >= 0 {
    if count > len {
      // Print warning to stderr and return unevaluated
      let list_str = crate::syntax::expr_to_string(list);
      eprintln!();
      eprintln!(
        "Take::take: Cannot take positions 1 through {} in {}.",
        count, list_str
      );
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
    Ok(Expr::List(items[..count as usize].to_vec()))
  } else {
    if -count > len {
      // Print warning to stderr and return unevaluated
      let list_str = crate::syntax::expr_to_string(list);
      eprintln!();
      eprintln!(
        "Take::take: Cannot take positions {} through -1 in {}.",
        count, list_str
      );
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
    Ok(Expr::List(
      items[items.len() - (-count) as usize..].to_vec(),
    ))
  }
}

/// AST-based Drop: drop first n elements.
pub fn drop_ast(list: &Expr, n: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Drop".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  let count = match n {
    Expr::Integer(i) => *i,
    Expr::Real(f) if f.fract() == 0.0 => *f as i128,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Drop".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  let len = items.len() as i128;
  if count >= 0 {
    let drop = count.min(len) as usize;
    Ok(Expr::List(items[drop..].to_vec()))
  } else {
    let keep = (len + count).max(0) as usize;
    Ok(Expr::List(items[..keep].to_vec()))
  }
}

/// AST-based Flatten: flatten nested lists.
pub fn flatten_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  fn flatten_recursive(expr: &Expr, result: &mut Vec<Expr>) {
    match expr {
      Expr::List(items) => {
        for item in items {
          flatten_recursive(item, result);
        }
      }
      _ => result.push(expr.clone()),
    }
  }

  match list {
    Expr::List(_) => {
      let mut result = Vec::new();
      flatten_recursive(list, &mut result);
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Flatten".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// Flatten[list, n] - flatten a list to depth n
pub fn flatten_level_ast(
  list: &Expr,
  depth: i128,
) -> Result<Expr, InterpreterError> {
  fn flatten_to_depth(expr: &Expr, depth: i128, result: &mut Vec<Expr>) {
    if depth <= 0 {
      result.push(expr.clone());
      return;
    }
    match expr {
      Expr::List(items) => {
        for item in items {
          flatten_to_depth(item, depth - 1, result);
        }
      }
      _ => result.push(expr.clone()),
    }
  }

  match list {
    Expr::List(items) => {
      let mut result = Vec::new();
      for item in items {
        flatten_to_depth(item, depth, &mut result);
      }
      Ok(Expr::List(result))
    }
    _ => Ok(list.clone()),
  }
}

/// AST-based Reverse: reverse a list.
pub fn reverse_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut reversed = items.clone();
      reversed.reverse();
      Ok(Expr::List(reversed))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: replacement.clone(),
      replacement: pattern.clone(),
    }),
    _ => Ok(Expr::FunctionCall {
      name: "Reverse".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// AST-based Sort: sort a list.
pub fn sort_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut sorted = items.clone();
      sorted.sort_by(|a, b| {
        let ka = crate::syntax::expr_to_string(a);
        let kb = crate::syntax::expr_to_string(b);
        // Try numeric comparison first
        if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
          na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
        } else {
          ka.cmp(&kb)
        }
      });
      Ok(Expr::List(sorted))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Sort".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// AST-based Range: generate a range of numbers.
/// Range[n] -> {1, 2, ..., n}
/// Range[min, max] -> {min, ..., max}
/// Range[min, max, step] -> {min, min+step, ..., max}
pub fn range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "Range".to_string(),
      args: args.to_vec(),
    });
  }

  let (min, max, step) = if args.len() == 1 {
    let max_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (1.0, max_val, 1.0)
  } else if args.len() == 2 {
    let min_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let max_val = expr_to_f64(&args[1]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (min_val, max_val, 1.0)
  } else {
    let min_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let max_val = expr_to_f64(&args[1]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let step_val = expr_to_f64(&args[2]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (min_val, max_val, step_val)
  };

  if step == 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Range: step cannot be zero".into(),
    ));
  }

  let mut results = Vec::new();
  let mut val = min;
  if step > 0.0 {
    while val <= max + f64::EPSILON {
      results.push(f64_to_expr(val));
      val += step;
    }
  } else {
    while val >= max - f64::EPSILON {
      results.push(f64_to_expr(val));
      val += step;
    }
  }

  Ok(Expr::List(results))
}

/// Helper to extract f64 from Expr
fn expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

/// Helper to convert f64 to appropriate Expr
fn f64_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
}

/// AST-based Accumulate: cumulative sums.
pub fn accumulate_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Accumulate".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  // Check if any element is Real - result should preserve Real type
  let has_real = items.iter().any(|item| matches!(item, Expr::Real(_)));

  let mut sum = 0.0;
  let mut results = Vec::new();
  for item in items {
    if let Some(n) = expr_to_f64(item) {
      sum += n;
      if has_real {
        results.push(Expr::Real(sum));
      } else {
        results.push(f64_to_expr(sum));
      }
    } else {
      return Ok(Expr::FunctionCall {
        name: "Accumulate".to_string(),
        args: vec![list.clone()],
      });
    }
  }

  Ok(Expr::List(results))
}

/// AST-based Differences: successive differences.
pub fn differences_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Differences".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if items.len() <= 1 {
    return Ok(Expr::List(vec![]));
  }

  let mut results = Vec::new();
  for i in 1..items.len() {
    if let (Some(a), Some(b)) =
      (expr_to_f64(&items[i - 1]), expr_to_f64(&items[i]))
    {
      results.push(f64_to_expr(b - a));
    } else {
      return Ok(Expr::FunctionCall {
        name: "Differences".to_string(),
        args: vec![list.clone()],
      });
    }
  }

  Ok(Expr::List(results))
}

/// AST-based Scan: apply function to each element for side effects.
/// Returns Null but evaluates function on each element.
pub fn scan_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Scan".to_string(),
        args: vec![func.clone(), list.clone()],
      });
    }
  };

  for item in items {
    apply_func_ast(func, item)?;
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// AST-based FoldList: fold showing intermediate values.
/// FoldList[f, x, {a, b, c}] -> {x, f[x, a], f[f[x, a], b], ...}
pub fn fold_list_ast(
  func: &Expr,
  init: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FoldList".to_string(),
        args: vec![func.clone(), init.clone(), list.clone()],
      });
    }
  };

  let mut results = vec![init.clone()];
  let mut acc = init.clone();
  for item in items {
    acc = apply_func_to_two_args(func, &acc, item)?;
    results.push(acc.clone());
  }

  Ok(Expr::List(results))
}

/// AST-based FixedPointList: list of values until fixed point.
pub fn fixed_point_list_ast(
  func: &Expr,
  init: &Expr,
  max_iterations: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut results = vec![init.clone()];
  let mut current = init.clone();

  for _ in 0..max {
    let next = apply_func_ast(func, &current)?;
    let current_str = crate::syntax::expr_to_string(&current);
    let next_str = crate::syntax::expr_to_string(&next);
    results.push(next.clone());
    if current_str == next_str {
      break;
    }
    current = next;
  }

  Ok(Expr::List(results))
}

/// AST-based Transpose: transpose a matrix (list of lists).
pub fn transpose_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let rows = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Transpose".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if rows.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Get dimensions
  let num_rows = rows.len();
  let num_cols = match &rows[0] {
    Expr::List(items) => items.len(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Transpose: argument must be a matrix".into(),
      ));
    }
  };

  // Verify all rows have the same length
  for row in rows {
    if let Expr::List(items) = row {
      if items.len() != num_cols {
        return Err(InterpreterError::EvaluationError(
          "Transpose: all rows must have the same length".into(),
        ));
      }
    } else {
      return Err(InterpreterError::EvaluationError(
        "Transpose: argument must be a matrix".into(),
      ));
    }
  }

  // Build transposed matrix
  let mut result = Vec::new();
  for j in 0..num_cols {
    let mut new_row = Vec::new();
    for i in 0..num_rows {
      if let Expr::List(items) = &rows[i] {
        new_row.push(items[j].clone());
      }
    }
    result.push(Expr::List(new_row));
  }

  Ok(Expr::List(result))
}

/// AST-based Riffle: interleave elements with separator.
/// Riffle[{a, b, c}, x] -> {a, x, b, x, c}
pub fn riffle_ast(list: &Expr, sep: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Riffle".to_string(),
        args: vec![list.clone(), sep.clone()],
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // If sep is a list, interleave element-wise: Riffle[{a,b,c}, {x,y,z}] -> {a,x,b,y,c,z}
  if let Expr::List(sep_items) = sep {
    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
      result.push(item.clone());
      if i < sep_items.len() {
        result.push(sep_items[i].clone());
      }
    }
    return Ok(Expr::List(result));
  }

  let mut result = Vec::new();
  for (i, item) in items.iter().enumerate() {
    result.push(item.clone());
    if i < items.len() - 1 {
      result.push(sep.clone());
    }
  }

  Ok(Expr::List(result))
}

/// AST-based RotateLeft: rotate list left by n positions.
pub fn rotate_left_ast(list: &Expr, n: i128) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RotateLeft".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let len = items.len() as i128;
  let shift = ((n % len) + len) % len;
  let shift_usize = shift as usize;

  let mut result = items[shift_usize..].to_vec();
  result.extend_from_slice(&items[..shift_usize]);

  Ok(Expr::List(result))
}

/// AST-based RotateRight: rotate list right by n positions.
pub fn rotate_right_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  rotate_left_ast(list, -n)
}

/// AST-based PadLeft: pad list on the left to length n.
/// If n < len, truncates from the left.
pub fn pad_left_ast(
  list: &Expr,
  n: i128,
  pad: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PadLeft".to_string(),
        args: vec![list.clone(), Expr::Integer(n), pad.clone()],
      });
    }
  };

  let len = items.len() as i128;
  if n <= 0 {
    return Ok(Expr::List(vec![]));
  }

  if n < len {
    // Truncate from the left
    let skip = (len - n) as usize;
    return Ok(Expr::List(items[skip..].to_vec()));
  }

  if n == len {
    return Ok(list.clone());
  }

  let needed = (n - len) as usize;
  let mut result = vec![pad.clone(); needed];
  result.extend(items.iter().cloned());

  Ok(Expr::List(result))
}

/// AST-based PadRight: pad list on the right to length n.
/// If n < len, truncates from the right.
pub fn pad_right_ast(
  list: &Expr,
  n: i128,
  pad: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PadRight".to_string(),
        args: vec![list.clone(), Expr::Integer(n), pad.clone()],
      });
    }
  };

  let len = items.len() as i128;
  if n <= 0 {
    return Ok(Expr::List(vec![]));
  }

  if n < len {
    // Truncate from the right
    return Ok(Expr::List(items[..n as usize].to_vec()));
  }

  if n == len {
    return Ok(list.clone());
  }

  let needed = (n - len) as usize;
  let mut result = items.clone();
  result.extend(vec![pad.clone(); needed]);

  Ok(Expr::List(result))
}

/// AST-based Join: join multiple lists.
pub fn join_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut result = Vec::new();
  for list in lists {
    match list {
      Expr::List(items) => result.extend(items.iter().cloned()),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Join".to_string(),
          args: lists.to_vec(),
        });
      }
    }
  }
  Ok(Expr::List(result))
}

/// AST-based Append: append element to list.
pub fn append_ast(list: &Expr, elem: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut result = items.clone();
      result.push(elem.clone());
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Append".to_string(),
      args: vec![list.clone(), elem.clone()],
    }),
  }
}

/// AST-based Prepend: prepend element to list.
pub fn prepend_ast(list: &Expr, elem: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut result = vec![elem.clone()];
      result.extend(items.iter().cloned());
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Prepend".to_string(),
      args: vec![list.clone(), elem.clone()],
    }),
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

/// AST-based Median: calculate median of a list.
pub fn median_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Median".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if items.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Median: list is empty".into(),
    ));
  }

  // Check if all items are integers
  let all_integers = items.iter().all(|i| matches!(i, Expr::Integer(_)));

  if all_integers {
    // Sort integer values
    let mut int_values: Vec<i128> = items
      .iter()
      .filter_map(|i| {
        if let Expr::Integer(n) = i {
          Some(*n)
        } else {
          None
        }
      })
      .collect();
    int_values.sort();

    let len = int_values.len();
    if len % 2 == 1 {
      Ok(Expr::Integer(int_values[len / 2]))
    } else {
      // Average of two middle values
      let a = int_values[len / 2 - 1];
      let b = int_values[len / 2];
      let sum = a + b;
      if sum % 2 == 0 {
        Ok(Expr::Integer(sum / 2))
      } else {
        // Return as Rational
        fn gcd(a: i128, b: i128) -> i128 {
          if b == 0 { a } else { gcd(b, a % b) }
        }
        let g = gcd(sum.abs(), 2);
        Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sum / g), Expr::Integer(2 / g)],
        })
      }
    }
  } else {
    // Check if any Real inputs - preserve Real type
    let has_real = items.iter().any(|i| matches!(i, Expr::Real(_)));

    // Extract numeric values as f64
    let mut values: Vec<f64> = Vec::new();
    for item in items {
      if let Some(n) = expr_to_f64(item) {
        values.push(n);
      } else {
        return Ok(Expr::FunctionCall {
          name: "Median".to_string(),
          args: vec![list.clone()],
        });
      }
    }

    values
      .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = values.len();
    let result = if len % 2 == 1 {
      values[len / 2]
    } else {
      (values[len / 2 - 1] + values[len / 2]) / 2.0
    };

    // Preserve Real type if inputs had Real
    if has_real {
      Ok(Expr::Real(result))
    } else {
      Ok(f64_to_expr(result))
    }
  }
}

/// AST-based Count: count elements equal to pattern.
pub fn count_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Count".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let pattern_str = crate::syntax::expr_to_string(pattern);
  let count = items
    .iter()
    .filter(|item| crate::syntax::expr_to_string(item) == pattern_str)
    .count();

  Ok(Expr::Integer(count as i128))
}

/// AST-based ConstantArray: create array filled with constant.
/// ConstantArray[c, n] -> {c, c, ..., c} (n times)
/// ConstantArray[c, {n1, n2}] -> nested array
pub fn constant_array_ast(
  elem: &Expr,
  dims: &Expr,
) -> Result<Expr, InterpreterError> {
  match dims {
    Expr::Integer(n) => {
      if *n < 0 {
        return Err(InterpreterError::EvaluationError(
          "ConstantArray: dimension must be non-negative".into(),
        ));
      }
      Ok(Expr::List(vec![elem.clone(); *n as usize]))
    }
    Expr::List(dim_list) => {
      if dim_list.is_empty() {
        return Ok(elem.clone());
      }
      let first_dim = expr_to_i128(&dim_list[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "ConstantArray: dimensions must be integers".into(),
        )
      })?;
      if dim_list.len() == 1 {
        Ok(Expr::List(vec![elem.clone(); first_dim as usize]))
      } else {
        let rest_dims = Expr::List(dim_list[1..].to_vec());
        let inner = constant_array_ast(elem, &rest_dims)?;
        Ok(Expr::List(vec![inner; first_dim as usize]))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "ConstantArray".to_string(),
      args: vec![elem.clone(), dims.clone()],
    }),
  }
}

/// AST-based NestWhile: nest while condition is true.
pub fn nest_while_ast(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  max_iterations: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut current = init.clone();

  for _ in 0..max {
    let test_result = apply_func_ast(test, &current)?;
    if expr_to_bool(&test_result) != Some(true) {
      break;
    }
    current = apply_func_ast(func, &current)?;
  }

  Ok(current)
}

/// AST-based NestWhileList: like NestWhile but returns list.
pub fn nest_while_list_ast(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  max_iterations: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut results = vec![init.clone()];
  let mut current = init.clone();

  for _ in 0..max {
    let test_result = apply_func_ast(test, &current)?;
    if expr_to_bool(&test_result) != Some(true) {
      break;
    }
    current = apply_func_ast(func, &current)?;
    results.push(current.clone());
  }

  Ok(Expr::List(results))
}

/// AST-based Product: product of list elements or iterator product.
pub fn product_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Product[{a, b, c}] -> a * b * c
    let items = match &args[0] {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Product".to_string(),
          args: args.to_vec(),
        });
      }
    };

    let mut product = 1.0;
    for item in items {
      if let Some(n) = expr_to_f64(item) {
        product *= n;
      } else {
        return Ok(Expr::FunctionCall {
          name: "Product".to_string(),
          args: args.to_vec(),
        });
      }
    }
    return Ok(f64_to_expr(product));
  }

  if args.len() == 2 {
    // Product[expr, {i, min, max}] -> multiply expr for each i
    let body = &args[0];
    let iter_spec = &args[1];

    match iter_spec {
      Expr::List(items) if items.len() >= 2 => {
        let var_name = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Product".to_string(),
              args: args.to_vec(),
            });
          }
        };

        // Check for list iteration form: {i, list}
        if items.len() == 2 {
          let evaluated_second =
            crate::evaluator::evaluate_expr_to_expr(&items[1])?;
          if let Expr::List(list_items) = &evaluated_second {
            // Product[expr, {i, list}] -> iterate over list elements
            let mut product = 1.0;
            for item in list_items {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, item);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              if let Some(n) = expr_to_f64(&val) {
                product *= n;
              } else {
                return Ok(Expr::FunctionCall {
                  name: "Product".to_string(),
                  args: args.to_vec(),
                });
              }
            }
            return Ok(f64_to_expr(product));
          }
        }

        // Check if bounds are numeric
        let bounds = if items.len() == 2 {
          expr_to_i128(&items[1]).map(|max| (1i128, max))
        } else {
          match (expr_to_i128(&items[1]), expr_to_i128(&items[2])) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
          }
        };

        // If bounds are symbolic, try to compute symbolic product
        if bounds.is_none() {
          // Check for special case: Product[i^k, {i, 1, n}] = n!^k
          // where k is a constant and n is symbolic
          let min_is_one = if items.len() == 2 {
            true // {i, n} implies min = 1
          } else {
            matches!(&items[1], Expr::Integer(1))
          };

          if min_is_one {
            let n_expr = if items.len() == 2 {
              &items[1]
            } else {
              &items[2]
            };

            // Check if body is i^k where i is the variable and k is a constant
            match body {
              Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left,
                right,
              } => {
                if matches!(left.as_ref(), Expr::Identifier(name) if name == &var_name)
                {
                  // Product[i^k, {i, 1, n}] = n!^k
                  if let Some(k) = expr_to_i128(right) {
                    // Return n!^k
                    return Ok(Expr::BinaryOp {
                      op: crate::syntax::BinaryOperator::Power,
                      left: Box::new(Expr::FunctionCall {
                        name: "Factorial".to_string(),
                        args: vec![n_expr.clone()],
                      }),
                      right: Box::new(Expr::Integer(k)),
                    });
                  }
                }
              }
              Expr::Identifier(name) if name == &var_name => {
                // Product[i, {i, 1, n}] = n!
                return Ok(Expr::FunctionCall {
                  name: "Factorial".to_string(),
                  args: vec![n_expr.clone()],
                });
              }
              _ => {}
            }
          }

          // For other symbolic cases, return unevaluated
          return Ok(Expr::FunctionCall {
            name: "Product".to_string(),
            args: args.to_vec(),
          });
        }

        let (min, max) = bounds.unwrap();

        let mut product = 1.0;
        for i in min..=max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          if let Some(n) = expr_to_f64(&val) {
            product *= n;
          } else {
            return Ok(Expr::FunctionCall {
              name: "Product".to_string(),
              args: args.to_vec(),
            });
          }
        }
        return Ok(f64_to_expr(product));
      }
      _ => {}
    }
  }

  Ok(Expr::FunctionCall {
    name: "Product".to_string(),
    args: args.to_vec(),
  })
}

/// AST-based Sum: sum of list elements or iterator sum.
pub fn sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Sum[{a, b, c}] -> a + b + c (same as Total)
    return crate::functions::math_ast::total_ast(args);
  }

  if args.len() == 2 {
    // Sum[expr, {i, min, max}] -> add expr for each i
    let body = &args[0];
    let iter_spec = &args[1];

    match iter_spec {
      Expr::List(items) if items.len() >= 2 => {
        let var_name = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Sum".to_string(),
              args: args.to_vec(),
            });
          }
        };

        // Check for list iteration form: {i, list}
        if items.len() == 2 {
          let evaluated_second =
            crate::evaluator::evaluate_expr_to_expr(&items[1])?;
          if let Expr::List(list_items) = &evaluated_second {
            // Sum[expr, {i, list}] -> iterate over list elements
            let mut sum = 0.0;
            for item in list_items {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, item);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              if let Some(n) = expr_to_f64(&val) {
                sum += n;
              } else {
                return Ok(Expr::FunctionCall {
                  name: "Sum".to_string(),
                  args: args.to_vec(),
                });
              }
            }
            return Ok(f64_to_expr(sum));
          }
        }

        let (min, max) = if items.len() == 2 {
          let max_val = expr_to_i128(&items[1]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Sum: iterator bounds must be integers".into(),
            )
          })?;
          (1i128, max_val)
        } else {
          let min_val = expr_to_i128(&items[1]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Sum: iterator bounds must be integers".into(),
            )
          })?;
          let max_val = expr_to_i128(&items[2]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Sum: iterator bounds must be integers".into(),
            )
          })?;
          (min_val, max_val)
        };

        let mut sum = 0.0;
        for i in min..=max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          if let Some(n) = expr_to_f64(&val) {
            sum += n;
          } else {
            return Ok(Expr::FunctionCall {
              name: "Sum".to_string(),
              args: args.to_vec(),
            });
          }
        }
        return Ok(f64_to_expr(sum));
      }
      _ => {}
    }
  }

  Ok(Expr::FunctionCall {
    name: "Sum".to_string(),
    args: args.to_vec(),
  })
}

/// AST-based Thread: thread a function over lists.
/// Thread[f[{a, b}, {c, d}]] -> {f[a, c], f[b, d]}
pub fn thread_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => {
      // Find which args are lists
      let mut list_indices: Vec<usize> = Vec::new();
      let mut list_len: Option<usize> = None;

      for (i, arg) in args.iter().enumerate() {
        if let Expr::List(items) = arg {
          if let Some(len) = list_len {
            if items.len() != len {
              return Err(InterpreterError::EvaluationError(
                "Thread: all lists must have the same length".into(),
              ));
            }
          } else {
            list_len = Some(items.len());
          }
          list_indices.push(i);
        }
      }

      if list_indices.is_empty() {
        return Ok(expr.clone());
      }

      let len = list_len.unwrap();
      let mut results = Vec::new();

      for j in 0..len {
        let new_args: Vec<Expr> = args
          .iter()
          .enumerate()
          .map(|(i, arg)| {
            if list_indices.contains(&i) {
              if let Expr::List(items) = arg {
                items[j].clone()
              } else {
                arg.clone()
              }
            } else {
              arg.clone()
            }
          })
          .collect();
        let result =
          crate::evaluator::evaluate_function_call_ast(name, &new_args)?;
        results.push(result);
      }

      Ok(Expr::List(results))
    }
    _ => Ok(expr.clone()),
  }
}

/// AST-based Through: apply multiple functions.
/// Through[{f, g}[x]] -> {f[x], g[x]}
pub fn through_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name: _, args } if !args.is_empty() => {
      // Check if first "name" is actually a list
      // Through[{f, g}[x]] is parsed as FunctionCall with name "{f, g}"
      // This is tricky - we need to handle this case specially
      Ok(expr.clone()) // Simplified for now
    }
    _ => Ok(expr.clone()),
  }
}

/// AST-based TakeLargest: take n largest elements.
pub fn take_largest_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  // Extract numeric values with indices
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for item in items {
    if let Some(v) = expr_to_f64(item) {
      keyed.push((v, item.clone()));
    } else {
      return Ok(Expr::FunctionCall {
        name: "TakeLargest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  }

  keyed
    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

  let take = (n as usize).min(keyed.len());
  let result: Vec<Expr> =
    keyed.into_iter().take(take).map(|(_, e)| e).collect();

  Ok(Expr::List(result))
}

/// AST-based TakeSmallest: take n smallest elements.
pub fn take_smallest_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  // Extract numeric values with indices
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for item in items {
    if let Some(v) = expr_to_f64(item) {
      keyed.push((v, item.clone()));
    } else {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  }

  keyed
    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

  let take = (n as usize).min(keyed.len());
  let result: Vec<Expr> =
    keyed.into_iter().take(take).map(|(_, e)| e).collect();

  Ok(Expr::List(result))
}

/// AST-based ArrayDepth: compute depth of nested lists.
pub fn array_depth_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  fn compute_depth(expr: &Expr) -> i128 {
    match expr {
      Expr::List(items) => {
        if items.is_empty() {
          1
        } else {
          1 + items.iter().map(compute_depth).min().unwrap_or(0)
        }
      }
      _ => 0,
    }
  }

  Ok(Expr::Integer(compute_depth(list)))
}

/// AST-based TakeWhile: take elements while predicate is true.
pub fn take_while_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeWhile".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  let mut result = Vec::new();
  for item in items {
    let test_result = apply_func_ast(pred, item)?;
    if expr_to_bool(&test_result) == Some(true) {
      result.push(item.clone());
    } else {
      break;
    }
  }

  Ok(Expr::List(result))
}

/// AST-based Do: execute expression multiple times.
/// Do[expr, n] -> execute expr n times
/// Do[expr, {i, max}] -> execute with i from 1 to max
/// Do[expr, {i, min, max}] -> execute with i from min to max
pub fn do_ast(body: &Expr, iter_spec: &Expr) -> Result<Expr, InterpreterError> {
  match iter_spec {
    Expr::Integer(n) => {
      for _ in 0..*n {
        match crate::evaluator::evaluate_expr_to_expr(body) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(e) => return Err(e),
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    Expr::List(items) if items.len() == 1 => {
      // Do[body, {n}] — repeat n times without iterator variable
      let n_expr = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
      let n = expr_to_i128(&n_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Do: repeat count must be an integer".into(),
        )
      })?;
      for _ in 0..n {
        match crate::evaluator::evaluate_expr_to_expr(body) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(e) => return Err(e),
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    Expr::List(items) if items.len() >= 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Do: iterator variable must be an identifier".into(),
          ));
        }
      };

      let (min, max, step) = if items.len() == 2 {
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_val = expr_to_i128(&max_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        (1i128, max_val, 1i128)
      } else if items.len() >= 3 {
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
        let min_val = expr_to_i128(&min_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        let max_val = expr_to_i128(&max_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        let step_val = if items.len() >= 4 {
          let step_expr = crate::evaluator::evaluate_expr_to_expr(&items[3])?;
          expr_to_i128(&step_expr).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: step must be an integer".into(),
            )
          })?
        } else {
          1i128
        };
        (min_val, max_val, step_val)
      } else {
        return Err(InterpreterError::EvaluationError(
          "Do: invalid iterator specification".into(),
        ));
      };

      if step == 0 {
        return Err(InterpreterError::EvaluationError(
          "Do: step cannot be zero".into(),
        ));
      }

      let mut i = min;
      if step > 0 {
        while i <= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          match crate::evaluator::evaluate_expr_to_expr(&substituted) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(e) => return Err(e),
          }
          i += step;
        }
      } else {
        while i >= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          match crate::evaluator::evaluate_expr_to_expr(&substituted) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(e) => return Err(e),
          }
          i += step;
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Do: invalid iterator specification".into(),
    )),
  }
}

/// AST-based DeleteCases: remove elements matching pattern.
pub fn delete_cases_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  delete_cases_with_count_ast(list, pattern, None)
}

/// DeleteCases[list, pattern, levelspec, n] - delete at most n matches
pub fn delete_cases_with_count_ast(
  list: &Expr,
  pattern: &Expr,
  max_count: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteCases".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let pattern_str = crate::syntax::expr_to_string(pattern);
  let mut removed = 0i128;
  let result: Vec<Expr> = items
    .iter()
    .filter(|item| {
      if let Some(max) = max_count
        && removed >= max
      {
        return true; // keep remaining items
      }
      let item_str = crate::syntax::expr_to_string(item);
      if matches_pattern_simple(&item_str, &pattern_str) {
        removed += 1;
        false
      } else {
        true
      }
    })
    .cloned()
    .collect();

  Ok(Expr::List(result))
}

/// AST-based MinMax: return {min, max} of a list.
pub fn min_max_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MinMax".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![
      Expr::Identifier("Infinity".to_string()),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      },
    ]));
  }

  let mut min_val = f64::INFINITY;
  let mut max_val = f64::NEG_INFINITY;

  for item in items {
    if let Some(n) = expr_to_f64(item) {
      if n < min_val {
        min_val = n;
      }
      if n > max_val {
        max_val = n;
      }
    } else {
      return Ok(Expr::FunctionCall {
        name: "MinMax".to_string(),
        args: vec![list.clone()],
      });
    }
  }

  Ok(Expr::List(vec![f64_to_expr(min_val), f64_to_expr(max_val)]))
}

/// Part[list, i] or list[[i]] - Extract element at position i (1-indexed)
pub fn part_ast(list: &Expr, index: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Part".to_string(),
        args: vec![list.clone(), index.clone()],
      });
    }
  };

  match index {
    Expr::Integer(i) => {
      if *i < 1 {
        return Err(InterpreterError::EvaluationError(
          "Part: index must be a positive integer".into(),
        ));
      }
      let idx = (*i as usize) - 1;
      if idx >= items.len() {
        return Err(InterpreterError::EvaluationError(
          "Part: index out of bounds".into(),
        ));
      }
      Ok(items[idx].clone())
    }
    Expr::List(indices) => {
      // Multiple indices: Part[list, {i1, i2, ...}]
      let mut results = Vec::new();
      for idx_expr in indices {
        if let Expr::Integer(i) = idx_expr {
          if *i < 1 {
            return Err(InterpreterError::EvaluationError(
              "Part: index must be a positive integer".into(),
            ));
          }
          let idx = (*i as usize) - 1;
          if idx >= items.len() {
            return Err(InterpreterError::EvaluationError(
              "Part: index out of bounds".into(),
            ));
          }
          results.push(items[idx].clone());
        } else {
          return Err(InterpreterError::EvaluationError(
            "Part: indices must be integers".into(),
          ));
        }
      }
      Ok(Expr::List(results))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Part".to_string(),
      args: vec![list.clone(), index.clone()],
    }),
  }
}

/// Insert[list, elem, n] - Insert element at position n (1-indexed)
pub fn insert_ast(
  list: &Expr,
  elem: &Expr,
  pos: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Insert".to_string(),
        args: vec![list.clone(), elem.clone(), pos.clone()],
      });
    }
  };

  let n = match pos {
    Expr::Integer(n) => *n,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Insert: position must be an integer".into(),
      ));
    }
  };

  let len = items.len() as i128;

  // Handle positive and negative indices
  let insert_pos = if n > 0 {
    let pos = (n - 1) as usize;
    if pos > items.len() {
      return Err(InterpreterError::EvaluationError(
        "Insert: position out of bounds".into(),
      ));
    }
    pos
  } else if n < 0 {
    let pos = (len + 1 + n) as usize;
    if n < -(len + 1) {
      return Err(InterpreterError::EvaluationError(
        "Insert: position out of bounds".into(),
      ));
    }
    pos
  } else {
    return Err(InterpreterError::EvaluationError(
      "Insert: position cannot be 0".into(),
    ));
  };

  let mut result = items;
  result.insert(insert_pos, elem.clone());
  Ok(Expr::List(result))
}

/// Array[f, n] - creates a list by applying f to indices 1..n
pub fn array_ast(func: &Expr, n: i128) -> Result<Expr, InterpreterError> {
  let mut result = Vec::new();
  for i in 1..=n {
    let arg = Expr::Integer(i);
    let val = apply_func_ast(func, &arg)?;
    result.push(val);
  }
  Ok(Expr::List(result))
}

/// Gather[list] - gathers elements into sublists of identical elements
pub fn gather_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Gather expects a list argument".into(),
      ));
    }
  };
  let mut groups: Vec<Vec<Expr>> = Vec::new();
  for item in items {
    let found = groups.iter_mut().find(|g| {
      crate::syntax::expr_to_string(&g[0])
        == crate::syntax::expr_to_string(item)
    });
    if let Some(group) = found {
      group.push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// GatherBy[list, f] - gathers elements into sublists by applying f
pub fn gather_by_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "GatherBy expects a list as first argument".into(),
      ));
    }
  };
  let mut groups: Vec<(String, Vec<Expr>)> = Vec::new();
  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    let found = groups.iter_mut().find(|(k, _)| *k == key_str);
    if let Some((_, group)) = found {
      group.push(item.clone());
    } else {
      groups.push((key_str, vec![item.clone()]));
    }
  }
  Ok(Expr::List(
    groups.into_iter().map(|(_, v)| Expr::List(v)).collect(),
  ))
}

/// Split[list] - splits into sublists of identical consecutive elements
pub fn split_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Split expects a list argument".into(),
      ));
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let last_group = groups.last().unwrap();
    if crate::syntax::expr_to_string(&last_group[0])
      == crate::syntax::expr_to_string(item)
    {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// SplitBy[list, f] - splits into sublists of consecutive elements with same f value
pub fn split_by_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "SplitBy expects a list as first argument".into(),
      ));
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }
  let mut prev_key = apply_func_ast(func, &items[0])?;
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let key = apply_func_ast(func, item)?;
    if crate::syntax::expr_to_string(&key)
      == crate::syntax::expr_to_string(&prev_key)
    {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
      prev_key = key;
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// Extract[list, n] - extracts element at position n
/// Extract[list, {n1, n2, ...}] - extracts element at nested position
pub fn extract_ast(
  list: &Expr,
  index: &Expr,
) -> Result<Expr, InterpreterError> {
  match index {
    Expr::Integer(_) => part_ast(list, index),
    Expr::List(indices) => {
      // Nested extraction: Extract[expr, {i, j, ...}]
      let mut current = list.clone();
      for idx in indices {
        current = part_ast(&current, idx)?;
      }
      Ok(current)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Extract: invalid index".into(),
    )),
  }
}

/// Catenate[{list1, list2, ...}] - concatenates lists
pub fn catenate_ast(list_of_lists: &Expr) -> Result<Expr, InterpreterError> {
  let outer = match list_of_lists {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Catenate expects a list of lists".into(),
      ));
    }
  };
  let mut result = Vec::new();
  for item in outer {
    match item {
      Expr::List(inner) => result.extend(inner.clone()),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Catenate expects all elements to be lists".into(),
        ));
      }
    }
  }
  Ok(Expr::List(result))
}

/// Apply[f, list] - applies f to the elements of list (f @@ list)
pub fn apply_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Apply expects a list as second argument".into(),
      ));
    }
  };
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, &items)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, &items);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Apply: first argument must be a function".into(),
    )),
  }
}

/// Identity[x] - returns x unchanged
pub fn identity_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  Ok(arg.clone())
}

/// Outer[f, list1, list2] - generalized outer product
pub fn outer_ast(
  func: &Expr,
  list1: &Expr,
  list2: &Expr,
) -> Result<Expr, InterpreterError> {
  let items1 = match list1 {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Outer expects lists as arguments".into(),
      ));
    }
  };
  let items2 = match list2 {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Outer expects lists as arguments".into(),
      ));
    }
  };
  let mut result = Vec::new();
  for a in items1 {
    let mut row = Vec::new();
    for b in items2 {
      let val = match func {
        Expr::Identifier(name) => crate::evaluator::evaluate_function_call_ast(
          name,
          &[a.clone(), b.clone()],
        )?,
        Expr::Function { body } => {
          let substituted =
            crate::syntax::substitute_slots(body, &[a.clone(), b.clone()]);
          crate::evaluator::evaluate_expr_to_expr(&substituted)?
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Outer: first argument must be a function".into(),
          ));
        }
      };
      row.push(val);
    }
    result.push(Expr::List(row));
  }
  Ok(Expr::List(result))
}

/// Inner[f, list1, list2, g] - generalized inner product
pub fn inner_ast(
  f: &Expr,
  list1: &Expr,
  list2: &Expr,
  g: &Expr,
) -> Result<Expr, InterpreterError> {
  let items1 = match list1 {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Inner expects lists as arguments".into(),
      ));
    }
  };
  let items2 = match list2 {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Inner expects lists as arguments".into(),
      ));
    }
  };
  if items1.len() != items2.len() {
    return Err(InterpreterError::EvaluationError(
      "Inner: lists must have the same length".into(),
    ));
  }
  // Apply f pairwise, then apply g to combine
  let mut products = Vec::new();
  for (a, b) in items1.iter().zip(items2.iter()) {
    let val = match f {
      Expr::Identifier(name) => crate::evaluator::evaluate_function_call_ast(
        name,
        &[a.clone(), b.clone()],
      )?,
      Expr::Function { body } => {
        let substituted =
          crate::syntax::substitute_slots(body, &[a.clone(), b.clone()]);
        crate::evaluator::evaluate_expr_to_expr(&substituted)?
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner: first argument must be a function".into(),
        ));
      }
    };
    products.push(val);
  }
  // Apply g to combine all products
  match g {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, &products)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Inner: fourth argument must be a function".into(),
    )),
  }
}

/// ReplacePart[list, n -> val] - replaces element at position n
pub fn replace_part_ast(
  list: &Expr,
  rule: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ReplacePart expects a list as first argument".into(),
      ));
    }
  };
  let (pos, val) = match rule {
    Expr::Rule {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ReplacePart expects a rule as second argument".into(),
      ));
    }
  };
  let idx = match pos {
    Expr::Integer(n) => {
      let len = items.len() as i128;
      if *n > 0 && *n <= len {
        (*n - 1) as usize
      } else if *n < 0 && -*n <= len {
        (len + *n) as usize
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "ReplacePart: position {} out of range",
          n
        )));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ReplacePart: position must be an integer".into(),
      ));
    }
  };
  let mut result = items;
  result[idx] = val.clone();
  Ok(Expr::List(result))
}

/// AST-based Permutations: generate all permutations of a list.
/// Permutations[{a, b, c}] -> all permutations
/// Permutations[{a, b, c}, {k}] -> all permutations of length k
pub fn permutations_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let list = &args[0];
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Permutations".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let k = if args.len() >= 2 {
    // Second arg should be {k} — a list with one integer
    match &args[1] {
      Expr::List(spec) if spec.len() == 1 => match &spec[0] {
        Expr::Integer(n) => *n as usize,
        _ => items.len(),
      },
      Expr::Integer(n) => *n as usize,
      _ => items.len(),
    }
  } else {
    items.len()
  };

  let n = items.len();
  if k > n {
    return Ok(Expr::List(vec![]));
  }

  let mut result = Vec::new();
  let indices: Vec<usize> = (0..n).collect();
  generate_k_permutations(
    &indices,
    k,
    &mut vec![],
    &mut vec![false; n],
    &items,
    &mut result,
  );
  Ok(Expr::List(result))
}

/// Helper to generate k-permutations
fn generate_k_permutations(
  _indices: &[usize],
  k: usize,
  current: &mut Vec<usize>,
  used: &mut Vec<bool>,
  items: &[Expr],
  result: &mut Vec<Expr>,
) {
  if current.len() == k {
    let perm: Vec<Expr> = current.iter().map(|&i| items[i].clone()).collect();
    result.push(Expr::List(perm));
    return;
  }
  for i in 0..items.len() {
    if !used[i] {
      used[i] = true;
      current.push(i);
      generate_k_permutations(_indices, k, current, used, items, result);
      current.pop();
      used[i] = false;
    }
  }
}

/// AST-based Subsets: generate subsets of a list.
/// Subsets[{a, b, c}] -> all subsets
/// Subsets[{a, b, c}, {k}] -> all subsets of size k
pub fn subsets_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let list = &args[0];
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Subsets".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() >= 2 {
    // Subsets[list, {k}] - subsets of specific size
    match &args[1] {
      Expr::List(spec) if spec.len() == 1 => {
        if let Expr::Integer(k) = &spec[0] {
          let k = *k as usize;
          let mut result = Vec::new();
          generate_combinations(&items, k, 0, &mut vec![], &mut result);
          return Ok(Expr::List(result));
        }
      }
      _ => {}
    }
  }

  // Subsets[list] - all subsets
  let n = items.len();
  let mut result = Vec::new();
  for k in 0..=n {
    generate_combinations(&items, k, 0, &mut vec![], &mut result);
  }
  Ok(Expr::List(result))
}

/// Helper to generate combinations (subsets of size k)
fn generate_combinations(
  items: &[Expr],
  k: usize,
  start: usize,
  current: &mut Vec<Expr>,
  result: &mut Vec<Expr>,
) {
  if current.len() == k {
    result.push(Expr::List(current.clone()));
    return;
  }
  for i in start..items.len() {
    current.push(items[i].clone());
    generate_combinations(items, k, i + 1, current, result);
    current.pop();
  }
}

/// AST-based SparseArray: create a matrix from position rules.
/// SparseArray[rules, {rows, cols}, default] -> evaluates rules and creates matrix
pub fn sparse_array_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Ok(Expr::FunctionCall {
      name: "SparseArray".to_string(),
      args: args.to_vec(),
    });
  }

  let rules = &args[0];
  let dims = &args[1];
  let default = &args[2];

  // Extract dimensions
  let dim_values = match dims {
    Expr::List(items) => {
      let mut result = Vec::new();
      for item in items {
        match item {
          Expr::Integer(n) => result.push(*n as usize),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "SparseArray".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
      result
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if dim_values.len() == 2 {
    let rows = dim_values[0];
    let cols = dim_values[1];

    // Initialize matrix with default value
    let mut matrix: Vec<Vec<Expr>> = vec![vec![default.clone(); cols]; rows];

    // Process rules: {pos -> val, pos -> val, ...}
    let rule_list = match rules {
      Expr::List(items) => items.clone(),
      _ => vec![rules.clone()],
    };

    for rule in &rule_list {
      match rule {
        Expr::Rule {
          pattern,
          replacement,
        } => {
          // pattern should be {row, col} (1-indexed)
          if let Expr::List(pos) = pattern.as_ref()
            && pos.len() == 2
            && let (Expr::Integer(r), Expr::Integer(c)) = (&pos[0], &pos[1])
          {
            let ri = (*r - 1) as usize;
            let ci = (*c - 1) as usize;
            if ri < rows && ci < cols {
              matrix[ri][ci] = replacement.as_ref().clone();
            }
          }
        }
        _ => {} // skip non-rules
      }
    }

    // Convert to nested list
    let result: Vec<Expr> = matrix.into_iter().map(Expr::List).collect();
    return Ok(Expr::List(result));
  }

  // For non-2D arrays, return symbolic
  Ok(Expr::FunctionCall {
    name: "SparseArray".to_string(),
    args: args.to_vec(),
  })
}

/// Tuples[list, n] - Generate all n-tuples from elements of list (Cartesian product).
pub fn tuples_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Tuples expects exactly 2 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tuples".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = match &args[1] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tuples".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if n == 0 {
    return Ok(Expr::List(vec![Expr::List(vec![])]));
  }

  // Iterative Cartesian product
  let mut result: Vec<Vec<Expr>> = vec![vec![]];

  for _ in 0..n {
    let mut new_result = Vec::new();
    for tuple in &result {
      for item in items {
        let mut new_tuple = tuple.clone();
        new_tuple.push(item.clone());
        new_result.push(new_tuple);
      }
    }
    result = new_result;
  }

  Ok(Expr::List(result.into_iter().map(Expr::List).collect()))
}

/// Dimensions[list] - Returns the dimensions of a nested list
pub fn dimensions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Dimensions expects exactly 1 argument".into(),
    ));
  }

  fn get_dimensions(expr: &Expr) -> Vec<i128> {
    match expr {
      Expr::List(items) => {
        let mut dims = vec![items.len() as i128];
        if !items.is_empty() {
          // Check if all sub-elements are lists of the same length
          let sub_dims: Vec<Vec<i128>> =
            items.iter().map(get_dimensions).collect();
          if !sub_dims.is_empty() && sub_dims.iter().all(|d| d == &sub_dims[0])
          {
            dims.extend(sub_dims[0].iter());
          }
        }
        dims
      }
      _ => vec![],
    }
  }

  let dims = get_dimensions(&args[0]);
  Ok(Expr::List(dims.into_iter().map(Expr::Integer).collect()))
}

/// Delete[list, pos] - Delete an element at a position
pub fn delete_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Delete expects exactly 2 arguments".into(),
    ));
  }

  if let Expr::List(items) = &args[0] {
    // Collect positions to delete
    let positions: Vec<i128> = match &args[1] {
      Expr::Integer(n) => vec![*n],
      Expr::List(pos_list) => pos_list
        .iter()
        .filter_map(|p| {
          if let Expr::List(inner) = p
            && inner.len() == 1
            && let Expr::Integer(n) = &inner[0]
          {
            return Some(*n);
          }
          if let Expr::Integer(n) = p {
            Some(*n)
          } else {
            None
          }
        })
        .collect(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Delete: invalid position".into(),
        ));
      }
    };

    // Convert to 0-based indices, handling negative indices
    let len = items.len() as i128;
    let mut indices_to_remove: Vec<usize> = positions
      .iter()
      .filter_map(|&pos| {
        let idx = if pos > 0 {
          (pos - 1) as usize
        } else if pos < 0 {
          (len + pos) as usize
        } else {
          return None;
        };
        if idx < items.len() { Some(idx) } else { None }
      })
      .collect();
    indices_to_remove.sort();
    indices_to_remove.dedup();

    let result: Vec<Expr> = items
      .iter()
      .enumerate()
      .filter(|(i, _)| !indices_to_remove.contains(i))
      .map(|(_, item)| item.clone())
      .collect();
    Ok(Expr::List(result))
  } else {
    Ok(Expr::FunctionCall {
      name: "Delete".to_string(),
      args: args.to_vec(),
    })
  }
}

/// OrderedQ[list] - Tests if a list is in sorted (non-decreasing) order
pub fn ordered_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "OrderedQ expects exactly 1 argument".into(),
    ));
  }

  if let Expr::List(items) = &args[0] {
    if items.len() <= 1 {
      return Ok(Expr::Identifier("True".to_string()));
    }
    for i in 0..items.len() - 1 {
      if !expr_le(&items[i], &items[i + 1]) {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
    Ok(Expr::Identifier("True".to_string()))
  } else {
    Ok(Expr::FunctionCall {
      name: "OrderedQ".to_string(),
      args: args.to_vec(),
    })
  }
}

// ─── DeleteAdjacentDuplicates ──────────────────────────────────────

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

// ─── Commonest ─────────────────────────────────────────────────────

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

// ─── ComposeList ──────────────────────────────────────────────────

/// ComposeList[{f, g, h}, x] -> {x, f[x], g[f[x]], h[g[f[x]]]}
pub fn compose_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ComposeList expects exactly 2 arguments".into(),
    ));
  }
  let funcs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ComposeList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut result = vec![args[1].clone()];
  let mut current = args[1].clone();
  for func in funcs {
    current = apply_func_ast(func, &current)?;
    result.push(current.clone());
  }
  Ok(Expr::List(result))
}

/// Compare two Expr values for canonical ordering.
/// Returns 1 if a < b, -1 if a > b, 0 if equal (Wolfram Order convention).
pub fn compare_exprs(a: &Expr, b: &Expr) -> i64 {
  // Try numeric comparison first
  let a_num = match a {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let b_num = match b {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  if let (Some(an), Some(bn)) = (a_num, b_num) {
    return if an < bn {
      1
    } else if an > bn {
      -1
    } else {
      0
    };
  }
  // Numbers come before non-numbers
  if a_num.is_some() {
    return 1;
  }
  if b_num.is_some() {
    return -1;
  }
  // Wolfram canonical string ordering: case-insensitive first, then lowercase < uppercase
  let a_str = crate::syntax::expr_to_string(a);
  let b_str = crate::syntax::expr_to_string(b);
  wolfram_string_order(&a_str, &b_str)
}

/// Wolfram canonical string ordering: case-insensitive alphabetical, then lowercase < uppercase
fn wolfram_string_order(a: &str, b: &str) -> i64 {
  let a_chars: Vec<char> = a.chars().collect();
  let b_chars: Vec<char> = b.chars().collect();

  for (ac, bc) in a_chars.iter().zip(b_chars.iter()) {
    let al = ac.to_lowercase().next().unwrap_or(*ac);
    let bl = bc.to_lowercase().next().unwrap_or(*bc);
    if al != bl {
      // Case-insensitive comparison first
      return if al < bl { 1 } else { -1 };
    }
    // Same letter, different case: lowercase comes first
    if ac != bc {
      // lowercase < uppercase in Wolfram ordering
      if ac.is_lowercase() && bc.is_uppercase() {
        return 1;
      } else if ac.is_uppercase() && bc.is_lowercase() {
        return -1;
      }
    }
  }
  // If all compared chars are equal, shorter string comes first
  match a_chars.len().cmp(&b_chars.len()) {
    std::cmp::Ordering::Less => 1,
    std::cmp::Ordering::Greater => -1,
    std::cmp::Ordering::Equal => 0,
  }
}

/// Helper: compare two Expr values for ordering (less-or-equal)
fn expr_le(a: &Expr, b: &Expr) -> bool {
  compare_exprs(a, b) >= 0
}

/// Subsequences[list] - all contiguous subsequences
/// Subsequences[list, {n}] - contiguous subsequences of length n
/// Subsequences[list, {nmin, nmax}] - lengths in range
/// Subsequences[{a, b, c}] => {{}, {a}, {b}, {c}, {a, b}, {b, c}, {a, b, c}}
pub fn subsequences_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Subsequences expects 1 or 2 arguments".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Subsequences".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = items.len();

  // Determine min and max lengths
  let (min_len, max_len) = if args.len() == 2 {
    match &args[1] {
      Expr::List(spec) => {
        if spec.len() == 1 {
          // {n} - exactly length n
          if let Expr::Integer(k) = &spec[0] {
            let k = *k as usize;
            (k, k)
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsequences".to_string(),
              args: args.to_vec(),
            });
          }
        } else if spec.len() == 2 {
          // {nmin, nmax}
          if let (Expr::Integer(lo), Expr::Integer(hi)) = (&spec[0], &spec[1]) {
            (*lo as usize, *hi as usize)
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsequences".to_string(),
              args: args.to_vec(),
            });
          }
        } else {
          return Ok(Expr::FunctionCall {
            name: "Subsequences".to_string(),
            args: args.to_vec(),
          });
        }
      }
      Expr::Integer(k) => {
        let k = *k as usize;
        (k, k)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Subsequences".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    (0, n)
  };

  let mut result = Vec::new();
  for len in min_len..=max_len.min(n) {
    if len == 0 {
      result.push(Expr::List(vec![]));
    } else {
      for start in 0..=(n - len) {
        result.push(Expr::List(items[start..start + len].to_vec()));
      }
    }
  }
  Ok(Expr::List(result))
}
