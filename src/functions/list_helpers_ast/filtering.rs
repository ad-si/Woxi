#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based Select: filter elements where predicate returns True.
/// Select[{a, b, c}, pred] -> elements where pred[elem] is True
/// Select[{a, b, c}, pred, n] -> first n elements where pred[elem] is True
pub fn select_ast(
  list: &Expr,
  pred: &Expr,
  n: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  // Select works on any expression with arguments, preserving the head
  let (items, head_name): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
    _ => {
      let mut args = vec![list.clone(), pred.clone()];
      if let Some(limit) = n {
        args.push(limit.clone());
      }
      return Ok(Expr::FunctionCall {
        name: "Select".to_string(),
        args,
      });
    }
  };

  let limit = match n {
    Some(expr) => match expr {
      Expr::Integer(i) => Some(*i as usize),
      _ => None,
    },
    None => None,
  };

  let mut kept = Vec::new();
  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      kept.push(item.clone());
      if let Some(lim) = limit
        && kept.len() >= lim
      {
        break;
      }
    }
  }

  // Preserve the original head
  match head_name {
    Some(name) => Ok(Expr::FunctionCall { name, args: kept }),
    None => Ok(Expr::List(kept)),
  }
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
    // PatternTest: _?test or x_?test — matches if test[expr] is True
    Expr::PatternTest { name: _, test } => {
      let test_result = match test.as_ref() {
        Expr::Identifier(func_name) => {
          let call = Expr::FunctionCall {
            name: func_name.clone(),
            args: vec![expr.clone()],
          };
          crate::evaluator::evaluate_expr_to_expr(&call).ok()
        }
        Expr::Function { body } => {
          // Anonymous function: substitute slots in the body (not the Function wrapper)
          let substituted =
            crate::syntax::substitute_slots(body, &[expr.clone()]);
          crate::evaluator::evaluate_expr_to_expr(&substituted).ok()
        }
        _ => {
          // General expression used as function: call (test)[expr]
          let call_str = format!(
            "({})[{}]",
            crate::syntax::expr_to_string(test),
            crate::syntax::expr_to_string(expr)
          );
          crate::interpret(&call_str).ok().map(|r| {
            if r == "True" {
              Expr::Identifier("True".to_string())
            } else {
              Expr::Identifier("False".to_string())
            }
          })
        }
      };
      matches!(test_result, Some(Expr::Identifier(ref s)) if s == "True")
    }
    // Blank[] or Blank[h] as FunctionCall (unevaluated form)
    Expr::FunctionCall { name, args } if name == "Blank" => match args.len() {
      0 => true,
      1 => {
        if let Expr::Identifier(h) = &args[0] {
          get_expr_head_str(expr) == h
        } else {
          false
        }
      }
      _ => false,
    },
    // Pattern[name, blank] as FunctionCall (unevaluated form)
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      matches_pattern_ast(expr, &args[1])
    }
    // Identifier patterns like "_", "_Integer", "_List", etc.
    Expr::Identifier(s) if s == "_" => true,
    Expr::Identifier(s) if s.starts_with('_') => {
      let head = &s[1..];
      get_expr_head_str(expr) == head
    }
    // Except[c] - matches anything that doesn't match c
    // Except[c, pattern] - matches pattern but not c
    Expr::FunctionCall { name, args }
      if name == "Except" && (args.len() == 1 || args.len() == 2) =>
    {
      if args.len() == 2 {
        // Except[c, pattern] - matches pattern but not c
        matches_pattern_ast(expr, &args[1])
          && !matches_pattern_ast(expr, &args[0])
      } else {
        !matches_pattern_ast(expr, &args[0])
      }
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

/// Cases with level specification: Cases[list, pattern, {level}]
pub fn cases_with_level_ast(
  list: &Expr,
  pattern: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Parse level spec: {n} means exactly level n
  let level = match level_spec {
    Expr::List(items) if items.len() == 1 => {
      expr_to_i128(&items[0]).unwrap_or(1) as usize
    }
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

/// FirstPosition[list, pattern] - finds the position of the first element matching pattern
/// Returns {index} or Missing["NotFound"] if not found
pub fn first_position_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "FirstPosition expects at least 2 arguments".into(),
    ));
  }
  let default = if args.len() >= 3 {
    args[2].clone()
  } else {
    Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![Expr::String("NotFound".to_string())],
    }
  };

  fn find_first(
    expr: &Expr,
    pattern: &Expr,
    path: &mut Vec<i128>,
  ) -> Option<Vec<i128>> {
    let pattern_str = crate::syntax::expr_to_string(pattern);
    let expr_str = crate::syntax::expr_to_string(expr);
    if matches_pattern_simple(&expr_str, &pattern_str)
      || matches_pattern_ast(expr, pattern)
    {
      return Some(path.clone());
    }
    if let Expr::List(items) = expr {
      for (i, item) in items.iter().enumerate() {
        path.push((i + 1) as i128);
        if let Some(result) = find_first(item, pattern, path) {
          return Some(result);
        }
        path.pop();
      }
    }
    None
  }

  let mut path = Vec::new();
  match find_first(&args[0], &args[1], &mut path) {
    Some(indices) => {
      Ok(Expr::List(indices.into_iter().map(Expr::Integer).collect()))
    }
    None => Ok(default),
  }
}

/// AST-based Count: count elements equal to pattern.
pub fn count_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  count_ast_level(list, pattern, None)
}

pub fn count_ast_level(
  list: &Expr,
  pattern: &Expr,
  level_spec: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      let mut args = vec![list.clone(), pattern.clone()];
      if let Some(ls) = level_spec {
        args.push(ls.clone());
      }
      return Ok(Expr::FunctionCall {
        name: "Count".to_string(),
        args,
      });
    }
  };

  // Parse level spec
  let (min_level, max_level) = match level_spec {
    None => (1, 1), // Default: level 1 only
    Some(Expr::Integer(n)) => (1, *n as usize), // n means levels 1 through n
    Some(Expr::List(levels)) => {
      if levels.len() == 1 {
        if let Some(n) = expr_to_i128(&levels[0]) {
          (n as usize, n as usize) // {n} means exactly level n
        } else {
          (1, 1)
        }
      } else if levels.len() == 2 {
        let min = expr_to_i128(&levels[0]).unwrap_or(1) as usize;
        let max = expr_to_i128(&levels[1]).unwrap_or(1) as usize;
        (min, max)
      } else {
        (1, 1)
      }
    }
    _ => (1, 1),
  };

  let count = count_at_level(items, pattern, 1, min_level, max_level);
  Ok(Expr::Integer(count as i128))
}

fn count_at_level(
  items: &[Expr],
  pattern: &Expr,
  current_level: usize,
  min_level: usize,
  max_level: usize,
) -> usize {
  let mut count = 0;
  for item in items {
    if current_level >= min_level
      && current_level <= max_level
      && matches_pattern_ast(item, pattern)
    {
      count += 1;
    }
    // Recurse into sublists if we haven't reached max_level
    if current_level < max_level
      && let Expr::List(sub_items) = item
    {
      count += count_at_level(
        sub_items,
        pattern,
        current_level + 1,
        min_level,
        max_level,
      );
    }
  }
  count
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

  let mut removed = 0i128;
  let result: Vec<Expr> = items
    .iter()
    .filter(|item| {
      if let Some(max) = max_count
        && removed >= max
      {
        return true; // keep remaining items
      }
      if matches_pattern_ast(item, pattern) {
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

/// ContainsOnly[list, elems] - True if every element of list is in elems
pub fn contains_only_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ContainsOnly expects exactly 2 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ContainsOnly".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let elems = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ContainsOnly".to_string(),
        args: args.to_vec(),
      });
    }
  };

  use std::collections::HashSet;
  let allowed: HashSet<String> =
    elems.iter().map(crate::syntax::expr_to_string).collect();

  for item in list {
    if !allowed.contains(&crate::syntax::expr_to_string(item)) {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// LengthWhile[list, crit] - gives the number of contiguous elements at the start that satisfy crit
pub fn length_while_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LengthWhile expects exactly 2 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LengthWhile".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let crit = &args[1];
  let mut count: i128 = 0;
  for item in list {
    let test = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![crit.clone(), Expr::List(vec![item.clone()])],
    })?;
    match &test {
      Expr::Identifier(s) if s == "True" => count += 1,
      _ => break,
    }
  }
  Ok(Expr::Integer(count))
}

/// Pick[list, sel] - pick elements where selector is True
/// Pick[list, sel, pattern] - pick elements where selector matches pattern
pub fn pick_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Pick expects 2 or 3 arguments".into(),
    ));
  }
  let list = &args[0];
  let sel = &args[1];
  let pattern = if args.len() == 3 {
    Some(&args[2])
  } else {
    None
  };

  pick_recursive(list, sel, pattern)
}

fn pick_recursive(
  list: &Expr,
  sel: &Expr,
  pattern: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  match (list, sel) {
    (
      Expr::FunctionCall {
        name,
        args: list_args,
      },
      Expr::List(sel_items),
    ) if list_args.len() == sel_items.len() => {
      let mut result = Vec::new();
      for (item, s) in list_args.iter().zip(sel_items.iter()) {
        if let (Expr::List(_), Expr::List(_)) = (item, s) {
          let picked = pick_recursive(item, s, pattern)?;
          result.push(picked);
        } else if let (Expr::FunctionCall { .. }, Expr::List(_)) = (item, s) {
          let picked = pick_recursive(item, s, pattern)?;
          result.push(picked);
        } else if matches_selector(s, pattern) {
          result.push(item.clone());
        }
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: result,
      })
    }
    (Expr::List(list_items), Expr::List(sel_items))
      if list_items.len() == sel_items.len() =>
    {
      let mut result = Vec::new();
      for (item, s) in list_items.iter().zip(sel_items.iter()) {
        if let (Expr::List(_), Expr::List(_)) = (item, s) {
          let picked = pick_recursive(item, s, pattern)?;
          result.push(picked);
        } else if let (Expr::FunctionCall { .. }, Expr::List(_)) = (item, s) {
          let picked = pick_recursive(item, s, pattern)?;
          result.push(picked);
        } else if matches_selector(s, pattern) {
          result.push(item.clone());
        }
      }
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Pick".to_string(),
      args: if let Some(p) = pattern {
        vec![list.clone(), sel.clone(), p.clone()]
      } else {
        vec![list.clone(), sel.clone()]
      },
    }),
  }
}

/// Level[expr, levelspec] - gives a list of all subexpressions at the specified levels.
/// Level[expr, levelspec, Heads -> True] - also includes heads.
///
/// Each node has a positive level (distance from root) and a negative level (-Depth[node]).
/// Level specs: n means {1,n}, {n} means exactly n, {n1,n2} means range.
/// Positive level values refer to positive level, negative values refer to negative level.
pub fn level_ast(
  expr: &Expr,
  level_spec: &Expr,
  include_heads: bool,
) -> Result<Expr, InterpreterError> {
  let (min_level, max_level) = parse_level_spec(level_spec)?;

  let mut results = Vec::new();
  level_traverse(expr, 0, min_level, max_level, include_heads, &mut results);
  Ok(Expr::List(results))
}

/// Check if a node matches the level spec.
/// pos_level: distance from root (0 = root itself)
/// neg_level: -Depth[node] (-1 for atoms, -2 for f[atom], etc.)
fn matches_level(
  pos_level: i64,
  neg_level: i64,
  min_level: i64,
  max_level: i64,
) -> bool {
  // Check min condition
  let min_ok = if min_level >= 0 {
    pos_level >= min_level
  } else {
    neg_level >= min_level
  };

  // Check max condition
  let max_ok = if max_level >= 0 {
    pos_level <= max_level
  } else {
    neg_level <= max_level
  };

  min_ok && max_ok
}

/// Parse level spec into (min, max) raw values (positive or negative).
fn parse_level_spec(spec: &Expr) -> Result<(i64, i64), InterpreterError> {
  match spec {
    Expr::List(items) if items.len() == 1 => {
      let n = level_value(&items[0])?;
      Ok((n, n))
    }
    Expr::List(items) if items.len() == 2 => {
      let n1 = level_value(&items[0])?;
      let n2 = level_value(&items[1])?;
      Ok((n1, n2))
    }
    Expr::Identifier(s) if s == "Infinity" => Ok((1, i64::MAX)),
    _ => {
      let n = level_value(spec)?;
      Ok((1, n))
    }
  }
}

fn level_value(expr: &Expr) -> Result<i64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as i64),
    Expr::Identifier(s) if s == "Infinity" => Ok(i64::MAX),
    Expr::FunctionCall { name, args } if name == "Minus" && args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        Ok(-(*n as i64))
      } else {
        Err(InterpreterError::EvaluationError(
          "Invalid level specification".into(),
        ))
      }
    }
    _ => Err(InterpreterError::EvaluationError(
      "Invalid level specification".to_string(),
    )),
  }
}

/// Get head name for a BinaryOperator
fn binary_op_head(op: &crate::syntax::BinaryOperator) -> &'static str {
  use crate::syntax::BinaryOperator;
  match op {
    BinaryOperator::Plus | BinaryOperator::Minus => "Plus",
    BinaryOperator::Times | BinaryOperator::Divide => "Times",
    BinaryOperator::Power => "Power",
    BinaryOperator::And => "And",
    BinaryOperator::Or => "Or",
    BinaryOperator::StringJoin => "StringJoin",
    BinaryOperator::Alternatives => "Alternatives",
  }
}

/// Traverse expression tree in post-order, collecting matching elements.
/// Returns the Mathematica Depth of the expression.
fn level_traverse(
  expr: &Expr,
  pos_level: i64,
  min_level: i64,
  max_level: i64,
  include_heads: bool,
  results: &mut Vec<Expr>,
) -> i64 {
  // Helper: traverse children, emit head first if applicable, return max child depth
  let traverse_compound = |head_name: &str,
                           children: &[&Expr],
                           pos_level: i64,
                           results: &mut Vec<Expr>|
   -> i64 {
    // Head symbol is an atom (depth 1, neg_level = -1)
    if include_heads && matches_level(pos_level + 1, -1, min_level, max_level) {
      results.push(Expr::Identifier(head_name.to_string()));
    }

    let mut max_child_depth: i64 = 0;
    for child in children {
      let child_depth = level_traverse(
        child,
        pos_level + 1,
        min_level,
        max_level,
        include_heads,
        results,
      );
      max_child_depth = max_child_depth.max(child_depth);
    }
    max_child_depth
  };

  match expr {
    Expr::List(items) => {
      let children: Vec<&Expr> = items.iter().collect();
      let max_child_depth =
        traverse_compound("List", &children, pos_level, results);
      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    Expr::FunctionCall { name, args, .. } => {
      let children: Vec<&Expr> = args.iter().collect();
      let max_child_depth =
        traverse_compound(name, &children, pos_level, results);
      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    Expr::BinaryOp { op, left, right } => {
      let head = binary_op_head(op);
      let children = [left.as_ref(), right.as_ref()];
      let max_child_depth =
        traverse_compound(head, &children, pos_level, results);
      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    Expr::CurriedCall { func, args } => {
      // CurriedCall: head is the func expr, children are the args
      // For Heads->True, the head (func) is traversed as a sub-expression
      if include_heads {
        // Traverse the head expression (func) for matching sub-parts
        let _head_depth = level_traverse(
          func,
          pos_level + 1,
          min_level,
          max_level,
          include_heads,
          results,
        );
      }

      // Depth of CurriedCall is based on args only (not head), matching Mathematica behavior
      let mut max_child_depth: i64 = 0;
      for arg in args {
        let child_depth = level_traverse(
          arg,
          pos_level + 1,
          min_level,
          max_level,
          include_heads,
          results,
        );
        max_child_depth = max_child_depth.max(child_depth);
      }

      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    Expr::UnaryOp { op, operand } => {
      let head = match op {
        crate::syntax::UnaryOperator::Minus => "Times",
        crate::syntax::UnaryOperator::Not => "Not",
      };
      let children = [operand.as_ref()];
      let max_child_depth =
        traverse_compound(head, &children, pos_level, results);
      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    _ => {
      // Atom: depth 1, neg_level = -1
      if matches_level(pos_level, -1, min_level, max_level) {
        results.push(expr.clone());
      }
      1
    }
  }
}

fn matches_selector(sel: &Expr, pattern: Option<&Expr>) -> bool {
  match pattern {
    None => {
      matches!(sel, Expr::Identifier(s) if s == "True")
    }
    Some(pat) => {
      match crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "MatchQ".to_string(),
        args: vec![sel.clone(), pat.clone()],
      }) {
        Ok(Expr::Identifier(s)) => s == "True",
        _ => false,
      }
    }
  }
}
