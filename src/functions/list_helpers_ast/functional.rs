#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

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

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
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
    Ok(Expr::List(results))
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
    Ok(Expr::List(results))
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

/// Apply[f, list] - applies f to the elements of list (f @@ list)
pub fn apply_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    Expr::FunctionCall { args, .. } => args.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Apply expects a list or expression as second argument".into(),
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
    Expr::NamedFunction { params, body } => {
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(items.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
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
  // Parse level spec
  let (min_level, max_level) = match level_spec {
    Expr::Integer(n) => (0, *n as usize),
    Expr::List(levels) => {
      if levels.len() == 1 {
        if let Some(n) = expr_to_i128(&levels[0]) {
          (n as usize, n as usize)
        } else {
          (1, 1)
        }
      } else if levels.len() == 2 {
        let min = expr_to_i128(&levels[0]).unwrap_or(0) as usize;
        let max = expr_to_i128(&levels[1]).unwrap_or(1) as usize;
        (min, max)
      } else {
        (1, 1)
      }
    }
    _ => (1, 1),
  };

  apply_at_level_recursive(func, expr, 0, min_level, max_level)
}

fn apply_at_level_recursive(
  func: &Expr,
  expr: &Expr,
  current_level: usize,
  min_level: usize,
  max_level: usize,
) -> Result<Expr, InterpreterError> {
  let (items, is_list) = match expr {
    Expr::List(items) => (items.clone(), true),
    Expr::FunctionCall { args, .. } => (args.clone(), false),
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
  } else if let Expr::FunctionCall { name, .. } = expr {
    crate::evaluator::evaluate_function_call_ast(name, &new_items)
  } else {
    Ok(expr.clone())
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
    Expr::NamedFunction { params, body } => {
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(items.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Apply: first argument must be a function".into(),
    )),
  }
}

/// Outer[f, list1, list2, ...] - generalized outer product
pub fn outer_ast(
  func: &Expr,
  lists: &[Expr],
) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Outer expects at least one list argument".into(),
    ));
  }
  outer_impl(func, &lists[0], &lists[1..], &[])
}

fn outer_impl(
  func: &Expr,
  current: &Expr,
  remaining: &[Expr],
  accumulated: &[Expr],
) -> Result<Expr, InterpreterError> {
  if let Expr::List(items) = current {
    // Thread through list elements
    let mut results = Vec::new();
    for item in items {
      results.push(outer_impl(func, item, remaining, accumulated)?);
    }
    Ok(Expr::List(results))
  } else {
    // Atomic element: add to accumulated args
    let mut new_acc = accumulated.to_vec();
    new_acc.push(current.clone());
    if remaining.is_empty() {
      // All lists consumed: apply f to accumulated args
      apply_func_to_n_args(func, &new_acc)
    } else {
      // Move to next list
      outer_impl(func, &remaining[0], &remaining[1..], &new_acc)
    }
  }
}

/// Inner[f, list1, list2, g] - generalized inner product
pub fn inner_ast(
  f: &Expr,
  list1: &Expr,
  list2: &Expr,
  g: &Expr,
) -> Result<Expr, InterpreterError> {
  inner_recursive(f, list1, list2, g)
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
          Expr::List(inner) => {
            if j < inner.len() {
              col.push(inner[j].clone());
            } else {
              return Err(InterpreterError::EvaluationError(
                "Inner: incompatible dimensions".into(),
              ));
            }
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
