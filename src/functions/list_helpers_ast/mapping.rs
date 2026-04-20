#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

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
      let results: Vec<Expr> =
        results?.into_iter().filter(|e| !is_nothing(e)).collect();
      Ok(Expr::List(results))
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
      // For any compound expression, decompose into head + children,
      // apply func to each child, and reconstruct.
      // E.g. Map[f, Power[x, 2]] -> Power[f[x], f[2]]
      use crate::functions::expr_form::{ExprForm, decompose_expr};
      match decompose_expr(list) {
        ExprForm::Composite { head, children } => {
          let mapped: Result<Vec<Expr>, _> = children
            .iter()
            .map(|child| apply_func_ast(func, child))
            .collect();
          crate::evaluator::evaluate_function_call_ast(&head, &mapped?)
        }
        ExprForm::Atom(_) => {
          // Atomic expression: Map[f, atom] returns atom unchanged
          Ok(list.clone())
        }
      }
    }
  }
}

/// Map[f, expr, levelspec] - apply f at specified levels
pub fn map_with_level_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Check for Infinity identifier
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) if s == "Infinity" || s == "DirectedInfinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };

  // Parse level spec: {n} = exactly level n, {min, max} = range, n = {1, n}
  // Infinity means all levels, negative levels count from leaves
  // Also handle Heads -> True option
  let (min_level, max_level, _heads) = match level_spec {
    Expr::Integer(n) => (1i64, *n as i64, false),
    _ if is_infinity(level_spec) => (1i64, i64::MAX, false),
    Expr::List(items) if items.len() == 1 => {
      if is_infinity(&items[0]) {
        (1i64, i64::MAX, false)
      } else if let Some(n) = expr_to_i128(&items[0]) {
        (n as i64, n as i64, false)
      } else {
        return Ok(Expr::FunctionCall {
          name: "Map".to_string(),
          args: vec![func.clone(), expr.clone(), level_spec.clone()],
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      let min = expr_to_i128(&items[0]).unwrap_or(0) as i64;
      let max = if is_infinity(&items[1]) {
        i64::MAX
      } else {
        expr_to_i128(&items[1]).unwrap_or(0) as i64
      };
      (min, max, false)
    }
    // Heads -> True option (Expr::Rule variant from -> syntax)
    Expr::Rule {
      pattern,
      replacement,
    } => {
      if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Heads")
        && matches!(replacement.as_ref(), Expr::Identifier(v) if v == "True")
      {
        return map_with_heads(func, expr);
      }
      return Ok(Expr::FunctionCall {
        name: "Map".to_string(),
        args: vec![func.clone(), expr.clone(), level_spec.clone()],
      });
    }
    // Also handle FunctionCall("Rule", ...) form
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      if let Expr::Identifier(s) = &args[0]
        && s == "Heads"
        && matches!(&args[1], Expr::Identifier(v) if v == "True")
      {
        return map_with_heads(func, expr);
      }
      return Ok(Expr::FunctionCall {
        name: "Map".to_string(),
        args: vec![func.clone(), expr.clone(), level_spec.clone()],
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Map".to_string(),
        args: vec![func.clone(), expr.clone(), level_spec.clone()],
      });
    }
  };

  // If any level bound is negative, we need depth-aware traversal
  if min_level < 0 || max_level < 0 {
    map_at_depth_negative(func, expr, 0, min_level, max_level)
  } else {
    map_at_depth(func, expr, 0, min_level, max_level)
  }
}

/// Compute the Wolfram-style depth of an expression.
/// Atoms have depth 1, compound expressions have depth 1 + max child depth.
fn expr_depth(expr: &Expr) -> i64 {
  match expr {
    Expr::List(items) => {
      if items.is_empty() {
        1
      } else {
        1 + items.iter().map(expr_depth).max().unwrap_or(0)
      }
    }
    Expr::FunctionCall { args, .. } => {
      if args.is_empty() {
        1
      } else {
        1 + args.iter().map(expr_depth).max().unwrap_or(0)
      }
    }
    _ => 1,
  }
}

/// Map with negative level support. Negative levels count from leaves:
/// level -1 = atoms, level -2 = expressions whose max child depth is 1, etc.
/// The negative level of a node is -depth(node).
fn map_at_depth_negative(
  func: &Expr,
  expr: &Expr,
  current_depth: i64,
  min_level: i64,
  max_level: i64,
) -> Result<Expr, InterpreterError> {
  let children = match expr {
    Expr::List(items) => Some((items.as_slice(), None::<&str>)),
    Expr::FunctionCall { name, args } => {
      Some((args.as_slice(), Some(name.as_str())))
    }
    _ => None,
  };

  let neg_level = -(expr_depth(expr));

  let result = if let Some((items, head_name)) = children {
    let mapped: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| {
        map_at_depth_negative(
          func,
          item,
          current_depth + 1,
          min_level,
          max_level,
        )
      })
      .collect();
    let mapped = mapped?;
    match head_name {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: mapped,
      },
      None => Expr::List(mapped),
    }
  } else {
    expr.clone()
  };

  // Check if this node should have f applied.
  // Each node has both a positive level (current_depth from root) and
  // a negative level (neg_level = -depth, counting from leaves).
  // A node matches if its level falls within [min_level, max_level]
  // using the appropriate sign convention for each bound.
  let meets_min = if min_level >= 0 {
    current_depth >= min_level
  } else {
    neg_level >= min_level
  };
  let meets_max = if max_level >= 0 {
    current_depth <= max_level
  } else {
    neg_level <= max_level
  };

  if meets_min && meets_max {
    apply_func_ast(func, &result)
  } else {
    Ok(result)
  }
}

/// Recursively map at specified levels (bottom-up)
fn map_at_depth(
  func: &Expr,
  expr: &Expr,
  current_depth: i64,
  min_level: i64,
  max_level: i64,
) -> Result<Expr, InterpreterError> {
  let children = match expr {
    Expr::List(items) => Some((items.as_slice(), None::<&str>)),
    Expr::FunctionCall { name, args } => {
      Some((args.as_slice(), Some(name.as_str())))
    }
    _ => None,
  };

  let result = if let Some((items, head_name)) = children {
    // Recurse into children
    let mapped: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| {
        map_at_depth(func, item, current_depth + 1, min_level, max_level)
      })
      .collect();
    let mapped = mapped?;
    match head_name {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: mapped,
      },
      None => Expr::List(mapped),
    }
  } else {
    expr.clone()
  };

  // Apply f at this depth if in range
  if current_depth >= min_level && current_depth <= max_level {
    apply_func_ast(func, &result)
  } else {
    Ok(result)
  }
}

/// Map[f, expr, Heads -> True]: apply f to head and all level-1 elements
fn map_with_heads(func: &Expr, expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(items) => {
      let mapped: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_func_ast(func, item))
        .collect();
      // Apply f to the head (List)
      let new_head =
        apply_func_ast(func, &Expr::Identifier("List".to_string()))?;
      Ok(Expr::CurriedCall {
        func: Box::new(new_head),
        args: mapped?,
      })
    }
    Expr::FunctionCall { name, args } => {
      let mapped: Result<Vec<Expr>, _> =
        args.iter().map(|item| apply_func_ast(func, item)).collect();
      // Apply f to the head
      let new_head = apply_func_ast(func, &Expr::Identifier(name.clone()))?;
      Ok(Expr::CurriedCall {
        func: Box::new(new_head),
        args: mapped?,
      })
    }
    _ => apply_func_ast(func, expr),
  }
}

/// MapAt[f, list, pos] - Apply function at specific positions
/// Supports single integer, list of integers, and negative indices
/// Helper to apply a function at a deep position within an expression.
fn map_at_deep(
  func: &Expr,
  expr: &Expr,
  path: &[i128],
) -> Result<Expr, InterpreterError> {
  if path.is_empty() {
    return apply_func_ast(func, expr);
  }

  let pos = path[0];
  let rest = &path[1..];

  let items = match expr {
    Expr::List(items) => items,
    _ => {
      return Ok(expr.clone());
    }
  };

  let len = items.len() as i128;
  let idx = if pos < 0 {
    (len + pos) as usize
  } else {
    (pos - 1) as usize
  };

  if idx >= items.len() {
    return Ok(expr.clone());
  }

  let mut new_items = items.clone();
  new_items[idx] = map_at_deep(func, &items[idx], rest)?;
  Ok(Expr::List(new_items))
}

pub fn map_at_ast(
  func: &Expr,
  list: &Expr,
  pos_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Association: <|k1 -> v1, ...|> supports integer-position MapAt that
  // transforms the value at position n (1-based; negative counts from end).
  if let Expr::Association(pairs) = list
    && let Some(n) = expr_to_i128(pos_spec)
  {
    let len = pairs.len() as i128;
    let idx = if n < 0 {
      (len + n) as usize
    } else {
      (n - 1) as usize
    };
    if idx >= pairs.len() {
      return Ok(Expr::FunctionCall {
        name: "MapAt".to_string(),
        args: vec![func.clone(), list.clone(), pos_spec.clone()],
      });
    }
    let mut new_pairs = pairs.clone();
    let new_value = apply_func_ast(func, &new_pairs[idx].1)?;
    new_pairs[idx].1 = new_value;
    return Ok(Expr::Association(new_pairs));
  }

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

  match pos_spec {
    Expr::Integer(n) => {
      // Single flat position
      let idx = if *n < 0 {
        (len + *n) as usize
      } else {
        (*n - 1) as usize
      };
      let mut new_items = items.clone();
      if idx < new_items.len() {
        new_items[idx] = apply_func_ast(func, &items[idx])?;
      }
      Ok(Expr::List(new_items))
    }
    Expr::BigInteger(_) => match expr_to_i128(pos_spec) {
      Some(n) => {
        let idx = if n < 0 {
          (len + n) as usize
        } else {
          (n - 1) as usize
        };
        let mut new_items = items.clone();
        if idx < new_items.len() {
          new_items[idx] = apply_func_ast(func, &items[idx])?;
        }
        Ok(Expr::List(new_items))
      }
      None => Ok(Expr::FunctionCall {
        name: "MapAt".to_string(),
        args: vec![func.clone(), list.clone(), pos_spec.clone()],
      }),
    },
    Expr::List(pos_list) => {
      if pos_list.is_empty() {
        return Ok(list.clone());
      }
      // Check if all elements are lists (multiple position specs) or all integers (deep position)
      let all_lists = pos_list.iter().all(|p| matches!(p, Expr::List(_)));
      let all_ints = pos_list.iter().all(|p| expr_to_i128(p).is_some());

      if all_ints {
        // Deep position: {2, 1} means position 1 within position 2
        let path: Vec<i128> =
          pos_list.iter().filter_map(expr_to_i128).collect();
        return map_at_deep(func, list, &path);
      }

      if all_lists {
        // Multiple positions: {{1}, {3}} or {{2,1}, {1,2}}
        let mut result = list.clone();
        for p in pos_list {
          if let Expr::List(inner) = p {
            if inner.len() == 1 {
              if let Some(n) = expr_to_i128(&inner[0]) {
                result = map_at_deep(func, &result, &[n])?;
              }
            } else {
              let path: Vec<i128> =
                inner.iter().filter_map(expr_to_i128).collect();
              if path.len() == inner.len() {
                result = map_at_deep(func, &result, &path)?;
              }
            }
          }
        }
        return Ok(result);
      }

      Ok(Expr::FunctionCall {
        name: "MapAt".to_string(),
        args: vec![func.clone(), list.clone(), pos_spec.clone()],
      })
    }
    // Span[start, end] or Span[start, end, step]
    Expr::FunctionCall {
      name,
      args: span_args,
    } if name == "Span" && (span_args.len() == 2 || span_args.len() == 3) => {
      let start_raw = expr_to_i128(&span_args[0]).unwrap_or(1);
      let step = if span_args.len() == 3 {
        expr_to_i128(&span_args[2]).unwrap_or(1)
      } else {
        1
      };

      // Resolve end: All means len.
      let end_raw = match &span_args[1] {
        Expr::Identifier(s) if s == "All" => len,
        other => expr_to_i128(other).unwrap_or(len),
      };

      // Normalize to 0-based indices.
      let start_idx = if start_raw < 0 {
        (len + start_raw) as usize
      } else {
        (start_raw - 1) as usize
      };
      let end_idx = if end_raw < 0 {
        (len + end_raw) as usize
      } else {
        (end_raw - 1) as usize
      };

      let step = step as usize;
      if step == 0 {
        return Err(InterpreterError::EvaluationError(
          "MapAt: Span step cannot be 0".into(),
        ));
      }

      let mut new_items = items.clone();
      let mut i = start_idx;
      while i <= end_idx && i < new_items.len() {
        new_items[i] = apply_func_ast(func, &items[i])?;
        i += step;
      }
      Ok(Expr::List(new_items))
    }
    _ => Ok(Expr::FunctionCall {
      name: "MapAt".to_string(),
      args: vec![func.clone(), list.clone(), pos_spec.clone()],
    }),
  }
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
  let results: Vec<Expr> =
    results?.into_iter().filter(|e| !is_nothing(e)).collect();
  Ok(Expr::List(results))
}

/// Detect a Heads -> True option, either as Expr::Rule or Expr::FunctionCall.
fn is_heads_true_option(e: &Expr) -> bool {
  match e {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Heads")
        && matches!(replacement.as_ref(), Expr::Identifier(v) if v == "True")
    }
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      matches!(&args[0], Expr::Identifier(s) if s == "Heads")
        && matches!(&args[1], Expr::Identifier(v) if v == "True")
    }
    _ => false,
  }
}

/// MapIndexed with level spec: MapIndexed[f, expr, levelspec]
/// Applies f to elements at the specified level, passing {position indices} as second arg.
/// Also accepts a bare 'Heads -> True' as the 3rd argument, in which case the
/// default level {1} is used but the head is included with index 0 appended.
/// As a 4th argument, 'Heads -> True' combines with the levelspec.
pub fn map_indexed_with_level_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // 'Heads -> True' as the 3rd arg ⇒ default level {1}, with heads.
  if is_heads_true_option(level_spec) {
    return map_indexed_at_depth_heads(func, expr, 0, 1, 1, &[]);
  }
  // Parse level spec: {n} = exactly level n
  let (min_level, max_level) = match level_spec {
    Expr::Integer(n) => (1i64, *n as i64),
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = expr_to_i128(&items[0]) {
        (n as i64, n as i64)
      } else {
        return Ok(Expr::FunctionCall {
          name: "MapIndexed".to_string(),
          args: vec![func.clone(), expr.clone(), level_spec.clone()],
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      let min = expr_to_i128(&items[0]).unwrap_or(0) as i64;
      let max = expr_to_i128(&items[1]).unwrap_or(0) as i64;
      (min, max)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapIndexed".to_string(),
        args: vec![func.clone(), expr.clone(), level_spec.clone()],
      });
    }
  };

  map_indexed_at_depth(func, expr, 0, min_level, max_level, &[])
}

/// MapIndexed[f, expr, levelspec, Heads -> True].
pub fn map_indexed_with_level_heads_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
  heads_opt: &Expr,
) -> Result<Expr, InterpreterError> {
  if !is_heads_true_option(heads_opt) {
    return Ok(Expr::FunctionCall {
      name: "MapIndexed".to_string(),
      args: vec![
        func.clone(),
        expr.clone(),
        level_spec.clone(),
        heads_opt.clone(),
      ],
    });
  }
  // Parse level spec the same way as the 3-arg form.
  let (min_level, max_level) = match level_spec {
    Expr::Integer(n) => (1i64, *n as i64),
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = expr_to_i128(&items[0]) {
        (n as i64, n as i64)
      } else {
        return Ok(Expr::FunctionCall {
          name: "MapIndexed".to_string(),
          args: vec![
            func.clone(),
            expr.clone(),
            level_spec.clone(),
            heads_opt.clone(),
          ],
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      let min = expr_to_i128(&items[0]).unwrap_or(0) as i64;
      let max = expr_to_i128(&items[1]).unwrap_or(0) as i64;
      (min, max)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapIndexed".to_string(),
        args: vec![
          func.clone(),
          expr.clone(),
          level_spec.clone(),
          heads_opt.clone(),
        ],
      });
    }
  };
  map_indexed_at_depth_heads(func, expr, 0, min_level, max_level, &[])
}

/// Recursively apply MapIndexed at specified depth levels, tracking position indices.
fn map_indexed_at_depth(
  func: &Expr,
  expr: &Expr,
  current_depth: i64,
  min_level: i64,
  max_level: i64,
  position: &[i128],
) -> Result<Expr, InterpreterError> {
  let children = match expr {
    Expr::List(items) => Some((items.as_slice(), true)),
    Expr::FunctionCall { name: _, args } => Some((args.as_slice(), false)),
    _ => None,
  };

  if let Some((items, is_list)) = children {
    // Recurse into children
    let mapped: Result<Vec<Expr>, _> = items
      .iter()
      .enumerate()
      .map(|(i, item)| {
        let mut child_pos = position.to_vec();
        child_pos.push((i + 1) as i128);
        map_indexed_at_depth(
          func,
          item,
          current_depth + 1,
          min_level,
          max_level,
          &child_pos,
        )
      })
      .collect();
    let mapped = mapped?;
    let result = if is_list {
      Expr::List(mapped)
    } else if let Expr::FunctionCall { name, .. } = expr {
      Expr::FunctionCall {
        name: name.clone(),
        args: mapped,
      }
    } else {
      unreachable!()
    };
    // Apply at this level if within range
    if current_depth >= min_level && current_depth <= max_level {
      let index =
        Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
      apply_func_to_two_args(func, &result, &index)
    } else {
      Ok(result)
    }
  } else {
    // Atom — apply if at the right level
    if current_depth >= min_level && current_depth <= max_level {
      let index =
        Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
      apply_func_to_two_args(func, expr, &index)
    } else {
      Ok(expr.clone())
    }
  }
}

/// MapIndexed variant that also applies f to heads, with index position 0 appended.
/// Each compound node first has its head rewritten as f[head, {position..., 0}]
/// (if within the level range), then its children are recursed with their own
/// position suffixes. The resulting head is the transformed head expression itself
/// (Mathematica style: f[List, {0}] becomes the head of the outer list).
fn map_indexed_at_depth_heads(
  func: &Expr,
  expr: &Expr,
  current_depth: i64,
  min_level: i64,
  max_level: i64,
  position: &[i128],
) -> Result<Expr, InterpreterError> {
  let (children, head_name_opt): (Option<&[Expr]>, Option<String>) = match expr
  {
    Expr::List(items) => (Some(items.as_slice()), Some("List".to_string())),
    Expr::FunctionCall { name, args } => {
      (Some(args.as_slice()), Some(name.clone()))
    }
    _ => (None, None),
  };

  if let (Some(items), Some(head_name)) = (children, head_name_opt) {
    // Recurse into children first (bottom-up).
    let mapped: Result<Vec<Expr>, _> = items
      .iter()
      .enumerate()
      .map(|(i, item)| {
        let mut child_pos = position.to_vec();
        child_pos.push((i + 1) as i128);
        map_indexed_at_depth_heads(
          func,
          item,
          current_depth + 1,
          min_level,
          max_level,
          &child_pos,
        )
      })
      .collect();
    let mapped = mapped?;

    // Transform the head at the current position with index 0 appended.
    // Heads count as level current_depth+1 (one step deeper than the node),
    // matching Mathematica's Heads->True convention.
    let head_expr = {
      let head_atom = Expr::Identifier(head_name.clone());
      let mut head_pos = position.to_vec();
      head_pos.push(0);
      let head_index =
        Expr::List(head_pos.iter().map(|&i| Expr::Integer(i)).collect());
      let head_depth = current_depth + 1;
      if head_depth >= min_level && head_depth <= max_level {
        apply_func_to_two_args(func, &head_atom, &head_index)?
      } else {
        head_atom
      }
    };

    // Re-wrap children using the transformed head as a head expression.
    // If the head is still the original symbol, use List / FunctionCall directly;
    // otherwise produce a CurriedCall so output renders as 'f[List,{0}][...]'.
    let inner_expr = match &head_expr {
      Expr::Identifier(n) if *n == "List" => Expr::List(mapped),
      Expr::Identifier(n) => Expr::FunctionCall {
        name: n.clone(),
        args: mapped,
      },
      _ => Expr::CurriedCall {
        func: Box::new(head_expr.clone()),
        args: mapped,
      },
    };

    // Apply f at this level to the wrapped expression if the level matches.
    if current_depth >= min_level && current_depth <= max_level {
      let index =
        Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
      apply_func_to_two_args(func, &inner_expr, &index)
    } else {
      Ok(inner_expr)
    }
  } else {
    // Atom: apply f if current depth is in range; heads don't apply to atoms.
    if current_depth >= min_level && current_depth <= max_level {
      let index =
        Expr::List(position.iter().map(|&i| Expr::Integer(i)).collect());
      apply_func_to_two_args(func, expr, &index)
    } else {
      Ok(expr.clone())
    }
  }
}

/// AST-based MapThread: apply function to corresponding elements.
/// MapThread[f, {{a, b}, {c, d}}] -> {f[a, c], f[b, d]}
pub fn map_thread_ast(
  func: &Expr,
  lists: &Expr,
  level: Option<usize>,
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

  let depth = level.unwrap_or(1);

  if depth <= 1 {
    // Apply function to corresponding elements
    let mut results = Vec::new();
    for i in 0..len {
      let args: Vec<Expr> = sublists.iter().map(|sl| sl[i].clone()).collect();
      let result = apply_func_to_n_args(func, &args)?;
      if !is_nothing(&result) {
        results.push(result);
      }
    }
    Ok(Expr::List(results))
  } else {
    // Recurse: thread at this level, then recurse into sublists
    let mut results = Vec::new();
    for i in 0..len {
      let inner_lists: Vec<Expr> =
        sublists.iter().map(|sl| sl[i].clone()).collect();
      let inner_arg = Expr::List(inner_lists);
      let result = map_thread_ast(func, &inner_arg, Some(depth - 1))?;
      results.push(result);
    }
    Ok(Expr::List(results))
  }
}

/// AST-based Thread: thread a function over lists.
/// Thread[f[{a, b}, {c, d}]] -> {f[a, c], f[b, d]}
pub fn thread_ast(
  expr: &Expr,
  thread_head: Option<&str>,
) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => {
      // Find which args contain the target head (List by default, or specified head)
      let mut list_indices: Vec<usize> = Vec::new();
      let mut list_len: Option<usize> = None;

      for (i, arg) in args.iter().enumerate() {
        let matching_items: Option<&Vec<Expr>> = match thread_head {
          None => {
            // Default: thread over List
            if let Expr::List(items) = arg {
              Some(items)
            } else {
              None
            }
          }
          Some(head) => {
            // Thread over specified head
            if let Expr::FunctionCall {
              name: arg_name,
              args: arg_args,
            } = arg
            {
              if arg_name == head {
                Some(arg_args)
              } else {
                None
              }
            } else if head == "List" {
              if let Expr::List(items) = arg {
                Some(items)
              } else {
                None
              }
            } else {
              None
            }
          }
        };

        if let Some(items) = matching_items {
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
              match thread_head {
                None => {
                  if let Expr::List(items) = arg {
                    items[j].clone()
                  } else {
                    arg.clone()
                  }
                }
                Some(_) => {
                  if let Expr::FunctionCall { args: arg_args, .. } = arg {
                    arg_args[j].clone()
                  } else if let Expr::List(items) = arg {
                    items[j].clone()
                  } else {
                    arg.clone()
                  }
                }
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

      // Wrap result in the thread head
      match thread_head {
        None => Ok(Expr::List(results)),
        Some(head) => {
          if head == "List" {
            Ok(Expr::List(results))
          } else {
            crate::evaluator::evaluate_function_call_ast(head, &results)
          }
        }
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      // Thread over list operands in a comparison, e.g. Thread[{a,b} >= 0] -> {a>=0, b>=0}
      let mut list_index: Option<usize> = None;
      let mut list_len: Option<usize> = None;
      for (i, op) in operands.iter().enumerate() {
        if let Expr::List(items) = op {
          match list_len {
            None => {
              list_len = Some(items.len());
              list_index = Some(i);
            }
            Some(len) if len != items.len() => {
              return Err(InterpreterError::EvaluationError(
                "Thread: all lists must have the same length".into(),
              ));
            }
            _ => {}
          }
        }
      }
      match (list_index, list_len) {
        (Some(_), Some(len)) => {
          let results: Vec<Expr> = (0..len)
            .map(|j| {
              let new_operands: Vec<Expr> = operands
                .iter()
                .map(|op| {
                  if let Expr::List(items) = op {
                    items[j].clone()
                  } else {
                    op.clone()
                  }
                })
                .collect();
              let new_cmp = Expr::Comparison {
                operands: new_operands,
                operators: operators.clone(),
              };
              crate::evaluator::evaluate_expr_to_expr(&new_cmp)
            })
            .collect::<Result<_, _>>()?;
          Ok(Expr::List(results))
        }
        _ => Ok(expr.clone()),
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      // Thread[{a,b} -> {c,d}] -> {a -> c, b -> d}
      let is_delayed = matches!(expr, Expr::RuleDelayed { .. });
      let lhs_items = match pattern.as_ref() {
        Expr::List(items) => Some(items),
        _ => None,
      };
      let rhs_items = match replacement.as_ref() {
        Expr::List(items) => Some(items),
        _ => None,
      };
      match (lhs_items, rhs_items) {
        (Some(lhs), Some(rhs)) => {
          if lhs.len() != rhs.len() {
            return Err(InterpreterError::EvaluationError(
              "Thread: all lists must have the same length".into(),
            ));
          }
          let results: Vec<Expr> = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| {
              if is_delayed {
                Expr::RuleDelayed {
                  pattern: Box::new(l.clone()),
                  replacement: Box::new(r.clone()),
                }
              } else {
                Expr::Rule {
                  pattern: Box::new(l.clone()),
                  replacement: Box::new(r.clone()),
                }
              }
            })
            .collect();
          Ok(Expr::List(results))
        }
        (Some(lhs), None) => {
          // Thread[{a,b} -> c] -> {a -> c, b -> c}
          let results: Vec<Expr> = lhs
            .iter()
            .map(|l| {
              if is_delayed {
                Expr::RuleDelayed {
                  pattern: Box::new(l.clone()),
                  replacement: replacement.clone(),
                }
              } else {
                Expr::Rule {
                  pattern: Box::new(l.clone()),
                  replacement: replacement.clone(),
                }
              }
            })
            .collect();
          Ok(Expr::List(results))
        }
        (None, Some(rhs)) => {
          // Thread[a -> {c,d}] -> {a -> c, a -> d}
          let results: Vec<Expr> = rhs
            .iter()
            .map(|r| {
              if is_delayed {
                Expr::RuleDelayed {
                  pattern: pattern.clone(),
                  replacement: Box::new(r.clone()),
                }
              } else {
                Expr::Rule {
                  pattern: pattern.clone(),
                  replacement: Box::new(r.clone()),
                }
              }
            })
            .collect();
          Ok(Expr::List(results))
        }
        (None, None) => Ok(expr.clone()),
      }
    }
    _ => Ok(expr.clone()),
  }
}

/// AST-based Through: apply multiple functions.
/// Through[{f, g}[x]] -> {f[x], g[x]}
/// Through[f[g][x]] -> f[g[x]]
/// Through[Plus[f, g][x]] -> f[x] + g[x]
pub fn through_ast(
  expr: &Expr,
  head_filter: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Through operates on CurriedCall: h[f1, f2, ...][args...]
  // It threads the args through each fi, wrapping the result in h.
  match expr {
    Expr::CurriedCall { func, args } => {
      // func is the head expression, e.g. f[g], {f, g}, Plus[f, g]
      // args are the outer arguments to thread through
      let (head_name, functions) = match func.as_ref() {
        Expr::FunctionCall { name, args: fns } => {
          (name.as_str(), fns.as_slice())
        }
        Expr::List(items) => ("List", items.as_slice()),
        _ => {
          // Not a compound head - return unevaluated
          return Ok(Expr::FunctionCall {
            name: "Through".to_string(),
            args: vec![expr.clone()],
          });
        }
      };

      // Check head filter if provided
      if let Some(filter) = head_filter
        && head_name != filter
      {
        // Head doesn't match filter - return the inner expression unchanged
        return Ok(expr.clone());
      }

      // Thread: apply each function to the outer args, then evaluate
      let threaded: Vec<Expr> = functions
        .iter()
        .map(|f| {
          let call = Expr::FunctionCall {
            name: crate::syntax::expr_to_string(f),
            args: args.clone(),
          };
          crate::evaluator::evaluate_expr_to_expr(&call).unwrap_or(call)
        })
        .collect();

      // Wrap in the head
      if head_name == "List" {
        Ok(Expr::List(threaded))
      } else {
        Ok(Expr::FunctionCall {
          name: head_name.to_string(),
          args: threaded,
        })
      }
    }
    _ => {
      if head_filter.is_some() {
        // With head filter and non-CurriedCall: return expression as-is
        Ok(expr.clone())
      } else {
        // Not a curried call - return unevaluated
        Ok(Expr::FunctionCall {
          name: "Through".to_string(),
          args: vec![expr.clone()],
        })
      }
    }
  }
}

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
