#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based Fold/FoldList: fold a function over a list.
/// Fold[f, x, {a, b, c}] -> f[f[f[x, a], b], c]
/// Threads through any non-atomic head (e.g. Fold[f, x, g[a,b]] works).
pub fn fold_ast(
  func: &Expr,
  init: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items: &[Expr] = match list {
    Expr::List(items) => items.as_slice(),
    Expr::FunctionCall { args, .. } => args.as_slice(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Fold".to_string(),
        args: vec![func.clone(), init.clone(), list.clone()],
      });
    }
  };

  let mut acc = init.clone();
  for item in items {
    acc = apply_func_to_two_args(func, &acc, item)?;
  }

  Ok(acc)
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

/// AST-based Accumulate: cumulative sums.
/// Threads through any non-atomic head, preserving it (e.g. Accumulate[g[1,2,3]]
/// returns g[1, 3, 6]).
pub fn accumulate_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  // Determine the head to wrap the result in and the items to accumulate over.
  let (items, head): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Accumulate".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  let wrap = |elems: Vec<Expr>| -> Expr {
    match &head {
      Some(name) => Expr::FunctionCall {
        name: name.clone(),
        args: elems,
      },
      None => Expr::List(elems),
    }
  };

  if items.is_empty() {
    return Ok(wrap(Vec::new()));
  }

  // Try numeric accumulation first
  let all_numeric = items.iter().all(|item| expr_to_f64(item).is_some());

  if all_numeric {
    let has_real = items.iter().any(|item| matches!(item, Expr::Real(_)));
    let mut sum = 0.0;
    let mut results = Vec::new();
    for item in items {
      sum += expr_to_f64(item).unwrap();
      if has_real {
        results.push(Expr::Real(sum));
      } else {
        results.push(f64_to_expr(sum));
      }
    }
    Ok(wrap(results))
  } else {
    // Symbolic accumulation using Plus
    let mut results = Vec::new();
    let mut running_sum = items[0].clone();
    results.push(running_sum.clone());
    for item in &items[1..] {
      running_sum = crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[running_sum, item.clone()],
      )?;
      results.push(running_sum.clone());
    }
    Ok(wrap(results))
  }
}

/// AST-based Differences: successive differences.
pub fn differences_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  differences_n_ast(list, 1)
}

/// Differences[list, n] - n-th order differences
pub fn differences_n_ast(
  list: &Expr,
  n: usize,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Differences".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  let mut current = items;
  for _ in 0..n {
    if current.len() <= 1 {
      return Ok(Expr::List(vec![]));
    }
    let mut next = Vec::new();
    for i in 1..current.len() {
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[current[i].clone(), current[i - 1].clone()],
      )?;
      next.push(diff);
    }
    current = next;
  }

  Ok(Expr::List(current))
}

/// Differences[list, {n1, n2, ...}] - apply `ni` differences at level `i`.
///
/// At the top level we take `n1` successive differences. Each element of
/// the resulting list is then recursively passed to
/// `differences_spec_ast(_, {n2, ...})` which applies the next level of
/// differencing. A single-element spec `{n}` is equivalent to
/// `Differences[list, n]`.
pub fn differences_spec_ast(
  list: &Expr,
  spec: &[usize],
) -> Result<Expr, InterpreterError> {
  if spec.is_empty() {
    return Ok(list.clone());
  }
  let first = spec[0];
  let outer = differences_n_ast(list, first)?;
  if spec.len() == 1 {
    return Ok(outer);
  }
  let rest = &spec[1..];
  let items: Vec<Expr> = match &outer {
    Expr::List(items) => items.clone(),
    _ => return Ok(outer),
  };
  let mut result = Vec::with_capacity(items.len());
  for item in items {
    result.push(differences_spec_ast(&item, rest)?);
  }
  Ok(Expr::List(result))
}

/// AST-based Scan: apply function to each element for side effects.
/// Returns Null but evaluates function on each element.
pub fn scan_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      for item in items {
        apply_func_ast(func, item)?;
      }
    }
    _ => {
      // For any compound expression, decompose into head + children,
      // and apply func to each child for side effects.
      // E.g. Scan[f, Power[x, 2]] applies f[x] and f[2]
      use crate::functions::expr_form::{ExprForm, decompose_expr};
      match decompose_expr(list) {
        ExprForm::Composite { children, .. } => {
          for child in &children {
            apply_func_ast(func, child)?;
          }
        }
        ExprForm::Atom(_) => {
          // Atoms have no parts to scan over
        }
      }
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// AST-based FoldList: fold showing intermediate values.
/// FoldList[f, x, {a, b, c}] -> {x, f[x, a], f[f[x, a], b], ...}
/// Threads through any non-atomic head: FoldList[f, x, g[a,b]]
/// returns g[x, f[x, a], f[f[x, a], b]].
pub fn fold_list_ast(
  func: &Expr,
  init: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let (items, head): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
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

  match head {
    Some(name) => Ok(Expr::FunctionCall {
      name,
      args: results,
    }),
    None => Ok(Expr::List(results)),
  }
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

/// AST-based NestWhile: nest while condition is true.
///
/// `extra_n` corresponds to Wolfram's 6th argument: after the test stops
/// being True (or `max_iterations` is reached), apply `func` an additional
/// `n` times when `n > 0`, or return the result `|n|` iterations earlier
/// when `n < 0`.
pub fn nest_while_ast(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  max_iterations: Option<i128>,
  extra_n: i128,
) -> Result<Expr, InterpreterError> {
  // We always materialise the full history when `extra_n` could need it,
  // so that negative `extra_n` can step back through previously seen values.
  let history = nest_while_history(func, init, test, max_iterations)?;
  let len = history.len() as i128;
  let target = (len - 1) + extra_n;
  if target < 0 {
    // Asked for a step before the initial value — Wolfram returns the
    // initial value in this case as the closest available result.
    return Ok(history[0].clone());
  }
  if (target as usize) < history.len() {
    return Ok(history[target as usize].clone());
  }
  // Need to keep applying `func` past the point where the test stopped.
  let mut current = history.last().cloned().unwrap_or_else(|| init.clone());
  let extra = (target as usize) - (history.len() - 1);
  for _ in 0..extra {
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
  extra_n: i128,
) -> Result<Expr, InterpreterError> {
  let mut history = nest_while_history(func, init, test, max_iterations)?;
  if extra_n > 0 {
    // Continue applying `func` for the requested extra iterations.
    let mut current = history.last().cloned().unwrap_or_else(|| init.clone());
    for _ in 0..extra_n {
      current = apply_func_ast(func, &current)?;
      history.push(current.clone());
    }
  } else if extra_n < 0 {
    // Drop the trailing `|extra_n|` values, but never below the initial.
    let drop = (-extra_n) as usize;
    let new_len = history.len().saturating_sub(drop).max(1);
    history.truncate(new_len);
  }
  Ok(Expr::List(history))
}

/// Iterate `func` from `init` while `test` is True, returning the entire
/// history (initial value first). Stops at `max_iterations` if provided.
fn nest_while_history(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  max_iterations: Option<i128>,
) -> Result<Vec<Expr>, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut history = vec![init.clone()];
  let mut current = init.clone();
  for _ in 0..max {
    let test_result = apply_func_ast(test, &current)?;
    if expr_to_bool(&test_result) != Some(true) {
      break;
    }
    current = apply_func_ast(func, &current)?;
    history.push(current.clone());
  }
  Ok(history)
}

/// Extract the children of any expression in Wolfram canonical form.
/// Returns None for atomic expressions (Integer, Real, String, Symbol).
fn expr_children(expr: &Expr) -> Option<Vec<Expr>> {
  match expr {
    Expr::List(items) => Some(items.clone()),
    Expr::FunctionCall { args, .. } => Some(args.clone()),
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator;
      match op {
        // a - b → Plus[a, Times[-1, b]]
        BinaryOperator::Minus => Some(vec![
          left.as_ref().clone(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), right.as_ref().clone()],
          },
        ]),
        // 1/b → Power[b, -1]; a/b → Times[a, Power[b, -1]]
        BinaryOperator::Divide => {
          if matches!(left.as_ref(), Expr::Integer(1)) {
            Some(vec![right.as_ref().clone(), Expr::Integer(-1)])
          } else {
            Some(vec![
              left.as_ref().clone(),
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![right.as_ref().clone(), Expr::Integer(-1)],
              },
            ])
          }
        }
        // Plus, Times, Power, And, Or, etc.
        _ => Some(vec![left.as_ref().clone(), right.as_ref().clone()]),
      }
    }
    Expr::UnaryOp { operand, .. } => Some(vec![operand.as_ref().clone()]),
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some(vec![pattern.as_ref().clone(), replacement.as_ref().clone()]),
    Expr::Association(pairs) => Some(
      pairs
        .iter()
        .map(|(k, v)| Expr::Rule {
          pattern: Box::new(k.clone()),
          replacement: Box::new(v.clone()),
        })
        .collect(),
    ),
    // Atomic expressions have no children
    _ => None,
  }
}

/// Apply[f, list] - applies f to the elements of list (f @@ list)
pub fn apply_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  // For associations, Apply operates on values (not rules)
  let items = if let Expr::Association(pairs) = list {
    pairs.iter().map(|(_, v)| v.clone()).collect()
  } else {
    match expr_children(list) {
      Some(items) => items,
      None => {
        // Atoms have no children; Apply on an atom returns the atom unchanged
        return Ok(list.clone());
      }
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
    Expr::NamedFunction { params, body, .. } => {
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(items.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Apply: first argument must be a function".into(),
    )),
  }
}

/// Apply[f, expr, levelspec] - replace heads at specified levels.
pub fn apply_at_level_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Compute depth so we can resolve negative level values:
  //   `-k` selects subexpressions whose Depth is k, i.e. level (D - k)
  // where D = Depth[expr] (atoms have Depth 1).
  let depth = expr_depth(expr);
  let resolve = |n: i128| -> usize {
    if n < 0 {
      let k = -n;
      let pos = depth as i128 - k;
      if pos < 0 { 0 } else { pos as usize }
    } else {
      n as usize
    }
  };

  // Parse level spec. For the integer form, `Apply[f, expr, n]` is
  // equivalent to `Apply[f, expr, {1, n}]` — levels 1 through n — not
  // {0, n} as Woxi previously assumed.
  let (min_level, max_level) = match level_spec {
    Expr::Identifier(s) if s == "Infinity" => (1usize, usize::MAX),
    Expr::Integer(n) => (1usize, resolve(*n)),
    Expr::List(levels) => {
      if levels.len() == 1 {
        if let Some(n) = expr_to_i128(&levels[0]) {
          let r = resolve(n);
          (r, r)
        } else {
          (1, 1)
        }
      } else if levels.len() == 2 {
        let min = match expr_to_i128(&levels[0]) {
          Some(n) => resolve(n),
          None => 0,
        };
        let max = match &levels[1] {
          Expr::Identifier(s) if s == "Infinity" => usize::MAX,
          other => match expr_to_i128(other) {
            Some(n) => resolve(n),
            None => 1,
          },
        };
        (min, max)
      } else {
        (1, 1)
      }
    }
    _ => (1, 1),
  };

  apply_at_level_recursive(func, expr, 0, min_level, max_level)
}

/// Mathematica-style Depth: atoms have depth 1, compound expressions are
/// 1 + max(depth of children).
fn expr_depth(expr: &Expr) -> usize {
  let children: Vec<&Expr> = match expr {
    Expr::List(items) => items.iter().collect(),
    Expr::FunctionCall { args, .. } => args.iter().collect(),
    Expr::BinaryOp { left, right, .. } => vec![left.as_ref(), right.as_ref()],
    Expr::UnaryOp { operand, .. } => vec![operand.as_ref()],
    _ => return 1,
  };
  if children.is_empty() {
    return 1;
  }
  1 + children.iter().map(|c| expr_depth(c)).max().unwrap_or(0)
}

fn apply_at_level_recursive(
  func: &Expr,
  expr: &Expr,
  current_level: usize,
  min_level: usize,
  max_level: usize,
) -> Result<Expr, InterpreterError> {
  // Normalize BinaryOp / UnaryOp into a FunctionCall so Apply can descend
  // into Plus/Times/Power and other operators that the parser sometimes
  // leaves as Expr::BinaryOp rather than FunctionCall.
  let normalized: Expr = match expr {
    Expr::BinaryOp { op, left, right } => Expr::FunctionCall {
      name: binary_op_to_name(*op).to_string(),
      args: vec![(**left).clone(), (**right).clone()],
    },
    Expr::UnaryOp { op, operand } => Expr::FunctionCall {
      name: unary_op_to_name(*op).to_string(),
      args: vec![(**operand).clone()],
    },
    other => other.clone(),
  };
  let (items, is_list, head_name) = match &normalized {
    Expr::List(items) => (items.clone(), true, None),
    Expr::FunctionCall { name, args } => {
      (args.clone(), false, Some(name.clone()))
    }
    _ => return Ok(expr.clone()),
  };

  // Recurse into children first if we haven't reached max_level
  let new_items: Vec<Expr> = if current_level < max_level {
    items
      .iter()
      .map(|item| {
        apply_at_level_recursive(
          func,
          item,
          current_level + 1,
          min_level,
          max_level,
        )
      })
      .collect::<Result<Vec<_>, _>>()?
  } else {
    items
  };

  // Replace head at this level if in range
  if current_level >= min_level && current_level <= max_level {
    apply_func_as_head(func, &new_items)
  } else if is_list {
    Ok(Expr::List(new_items))
  } else if let Some(name) = head_name {
    crate::evaluator::evaluate_function_call_ast(&name, &new_items)
  } else {
    Ok(expr.clone())
  }
}

fn binary_op_to_name(op: crate::syntax::BinaryOperator) -> &'static str {
  use crate::syntax::BinaryOperator::*;
  match op {
    Plus => "Plus",
    Minus => "Subtract",
    Times => "Times",
    Divide => "Divide",
    Power => "Power",
    And => "And",
    Or => "Or",
    StringJoin => "StringJoin",
    Alternatives => "Alternatives",
  }
}

fn unary_op_to_name(op: crate::syntax::UnaryOperator) -> &'static str {
  use crate::syntax::UnaryOperator::*;
  match op {
    Minus => "Times",
    Not => "Not",
  }
}

fn apply_func_as_head(
  func: &Expr,
  items: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, items)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, items);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(items.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Apply: first argument must be a function".into(),
    )),
  }
}

/// Outer[f, list1, list2, ...] - generalized outer product.
/// Threads through any shared head (not just `List`). All arguments must have
/// the same head; if they differ, the call is returned unevaluated.
pub fn outer_ast(
  func: &Expr,
  lists: &[Expr],
) -> Result<Expr, InterpreterError> {
  outer_ast_with_levels(func, lists, &[])
}

pub fn outer_ast_with_levels(
  func: &Expr,
  lists: &[Expr],
  levels: &[usize],
) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Outer expects at least one list argument".into(),
    ));
  }

  // Determine the common head. All arguments must share the same head.
  let head = expr_head_name(&lists[0]);
  if head.is_none() {
    // First argument is atomic — apply f directly.
    return apply_func_to_n_args(func, lists);
  }
  let head = head.unwrap();
  for arg in &lists[1..] {
    match expr_head_name(arg) {
      Some(h) if h == head => {}
      _ => {
        // Mismatched or atomic — return unevaluated.
        let mut call_args = vec![func.clone()];
        call_args.extend(lists.iter().cloned());
        return Ok(Expr::FunctionCall {
          name: "Outer".to_string(),
          args: call_args,
        });
      }
    }
  }

  if levels.is_empty() {
    outer_impl(func, &lists[0], &lists[1..], &[], &head)
  } else {
    // Build per-list level specs
    let per_list_levels: Vec<usize> = if levels.len() == 1 {
      // Single level spec applies to all lists
      vec![levels[0]; lists.len()]
    } else {
      // One level per list
      let mut v = levels.to_vec();
      while v.len() < lists.len() {
        v.push(*v.last().unwrap_or(&1));
      }
      v
    };
    outer_impl_leveled(func, lists, 0, &per_list_levels, &[], &head)
  }
}

/// Return the head name for an expression that can be threaded by Outer.
fn expr_head_name(expr: &Expr) -> Option<String> {
  match expr {
    Expr::List(_) => Some("List".to_string()),
    Expr::FunctionCall { name, .. } => Some(name.clone()),
    _ => None,
  }
}

/// Get child items from a List or FunctionCall.
fn expr_items(expr: &Expr) -> Option<&[Expr]> {
  match expr {
    Expr::List(items) => Some(items.as_slice()),
    Expr::FunctionCall { args, .. } => Some(args.as_slice()),
    _ => None,
  }
}

/// Wrap items in the given head.
fn wrap_in_head(head: &str, items: Vec<Expr>) -> Expr {
  if head == "List" {
    Expr::List(items)
  } else {
    Expr::FunctionCall {
      name: head.to_string(),
      args: items,
    }
  }
}

fn outer_impl(
  func: &Expr,
  current: &Expr,
  remaining: &[Expr],
  accumulated: &[Expr],
  head: &str,
) -> Result<Expr, InterpreterError> {
  if let Some(items) = expr_items(current) {
    // Thread through child elements, wrapping the result in the common head.
    let mut results = Vec::new();
    for item in items {
      results.push(outer_impl(func, item, remaining, accumulated, head)?);
    }
    Ok(wrap_in_head(head, results))
  } else {
    // Atomic element: add to accumulated args
    let mut new_acc = accumulated.to_vec();
    new_acc.push(current.clone());
    if remaining.is_empty() {
      // All lists consumed: apply f to accumulated args
      apply_func_to_n_args(func, &new_acc)
    } else {
      // Move to next list
      outer_impl(func, &remaining[0], &remaining[1..], &new_acc, head)
    }
  }
}

/// Level-aware Outer: descend only `levels[list_idx]` levels into each list.
fn outer_impl_leveled(
  func: &Expr,
  lists: &[Expr],
  list_idx: usize,
  levels: &[usize],
  accumulated: &[Expr],
  head: &str,
) -> Result<Expr, InterpreterError> {
  if list_idx >= lists.len() {
    // All lists consumed: apply f to accumulated args
    return apply_func_to_n_args(func, accumulated);
  }
  let depth = levels[list_idx];
  outer_descend(
    func,
    &lists[list_idx],
    lists,
    list_idx,
    levels,
    accumulated,
    head,
    depth,
  )
}

fn outer_descend(
  func: &Expr,
  current: &Expr,
  lists: &[Expr],
  list_idx: usize,
  levels: &[usize],
  accumulated: &[Expr],
  head: &str,
  depth_remaining: usize,
) -> Result<Expr, InterpreterError> {
  if depth_remaining > 0
    && let Some(items) = expr_items(current)
  {
    let mut results = Vec::new();
    for item in items {
      results.push(outer_descend(
        func,
        item,
        lists,
        list_idx,
        levels,
        accumulated,
        head,
        depth_remaining - 1,
      )?);
    }
    return Ok(wrap_in_head(head, results));
  }
  // Reached target depth or hit an atom: treat as a single element
  let mut new_acc = accumulated.to_vec();
  new_acc.push(current.clone());
  outer_impl_leveled(func, lists, list_idx + 1, levels, &new_acc, head)
}

/// Inner[f, list1, list2, g] - generalized inner product
pub fn inner_ast(
  f: &Expr,
  list1: &Expr,
  list2: &Expr,
  g: &Expr,
) -> Result<Expr, InterpreterError> {
  match inner_recursive(f, list1, list2, g) {
    Err(InterpreterError::EvaluationError(msg))
      if msg.contains("incompatible dimensions") =>
    {
      let l1_len = match list1 {
        Expr::List(items) => items.len(),
        _ => 0,
      };
      let l2_len = match list2 {
        Expr::List(items) => items.len(),
        _ => 0,
      };
      crate::emit_message(&format!(
        "Inner::incom: Length {} of dimension 1 in {} is incommensurate with length {} of dimension 1 in {}.",
        l1_len,
        crate::syntax::expr_to_string(list1),
        l2_len,
        crate::syntax::expr_to_string(list2),
      ));
      Ok(Expr::FunctionCall {
        name: "Inner".to_string(),
        args: vec![f.clone(), list1.clone(), list2.clone(), g.clone()],
      })
    }
    other => other,
  }
}

fn inner_recursive(
  f: &Expr,
  a: &Expr,
  b: &Expr,
  g: &Expr,
) -> Result<Expr, InterpreterError> {
  let a_depth = list_depth(a);
  let b_depth = list_depth(b);

  if a_depth == 1 && b_depth == 1 {
    // Base case: both are flat lists - do pairwise f then combine with g
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let items_b = match b {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    if items_a.len() != items_b.len() {
      return Err(InterpreterError::EvaluationError(
        "Inner: incompatible dimensions".into(),
      ));
    }
    let mut products = Vec::new();
    for (x, y) in items_a.iter().zip(items_b.iter()) {
      let val = apply_func_to_two_args(f, x, y)?;
      products.push(val);
    }
    apply_func_to_n_args(g, &products)
  } else if a_depth > 1 {
    // Map over the first dimension of a
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let mut results = Vec::new();
    for row in items_a {
      results.push(inner_recursive(f, row, b, g)?);
    }
    Ok(Expr::List(results))
  } else {
    // a_depth == 1, b_depth > 1
    // Contract a (vector) with first dimension of b
    // Need to iterate over the inner structure of b
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let items_b = match b {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    if items_a.len() != items_b.len() {
      return Err(InterpreterError::EvaluationError(
        "Inner: incompatible dimensions".into(),
      ));
    }
    // Each items_b[k] should be a list. Extract their elements column-wise.
    let inner_len = match &items_b[0] {
      Expr::List(inner) => inner.len(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner: incompatible dimensions".into(),
        ));
      }
    };
    let mut results = Vec::new();
    for j in 0..inner_len {
      // Build a "column" vector from b: [b[0][j], b[1][j], ...]
      let mut col = Vec::new();
      for bk in items_b {
        match bk {
          Expr::List(inner) if j < inner.len() => {
            col.push(inner[j].clone());
          }
          _ => {
            return Err(InterpreterError::EvaluationError(
              "Inner: incompatible dimensions".into(),
            ));
          }
        }
      }
      let col_expr = Expr::List(col);
      results.push(inner_recursive(f, a, &col_expr, g)?);
    }
    Ok(Expr::List(results))
  }
}
