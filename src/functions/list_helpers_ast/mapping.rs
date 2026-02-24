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

/// Map[f, expr, levelspec] - apply f at specified levels
pub fn map_with_level_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Parse level spec: {n} = exactly level n, {min, max} = range, n = {1, n}
  // Also handle Heads -> True option
  let (min_level, max_level, _heads) = match level_spec {
    Expr::Integer(n) => (1i64, *n as i64, false),
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = expr_to_i128(&items[0]) {
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
      let max = expr_to_i128(&items[1]).unwrap_or(0) as i64;
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

  map_at_depth(func, expr, 0, min_level, max_level)
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

  Ok(Expr::List(results?))
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
      results.push(result);
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

      // Thread: apply each function to the outer args
      let threaded: Vec<Expr> = functions
        .iter()
        .map(|f| Expr::FunctionCall {
          name: crate::syntax::expr_to_string(f),
          args: args.clone(),
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
