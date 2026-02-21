#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based Partition: break list into sublists of length n.
/// Partition[{a, b, c, d, e}, 2] -> {{a, b}, {c, d}}
pub fn partition_ast(
  list: &Expr,
  n: i128,
  d: Option<i128>,
) -> Result<Expr, InterpreterError> {
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
  let d_usize = d.unwrap_or(n) as usize;
  if d_usize == 0 {
    return Err(InterpreterError::EvaluationError(
      "Partition: offset must be positive".into(),
    ));
  }

  let mut results = Vec::new();
  let mut i = 0;
  while i + n_usize <= items.len() {
    results.push(Expr::List(items[i..i + n_usize].to_vec()));
    i += d_usize;
  }

  Ok(Expr::List(results))
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

/// Flatten[expr, n, head] - flatten expressions with a specific head
pub fn flatten_head_ast(
  list: &Expr,
  depth: i128,
  head: &str,
) -> Result<Expr, InterpreterError> {
  fn flatten_with_head(
    expr: &Expr,
    depth: i128,
    head: &str,
    result: &mut Vec<Expr>,
  ) {
    if depth <= 0 {
      result.push(expr.clone());
      return;
    }
    match expr {
      Expr::FunctionCall { name, args } if name == head => {
        for item in args {
          flatten_with_head(item, depth - 1, head, result);
        }
      }
      Expr::List(items) if head == "List" => {
        for item in items {
          flatten_with_head(item, depth - 1, head, result);
        }
      }
      _ => result.push(expr.clone()),
    }
  }

  // The outer expression must also have the matching head
  match list {
    Expr::FunctionCall { name, args } if name == head => {
      let mut result = Vec::new();
      for item in args {
        flatten_with_head(item, depth, head, &mut result);
      }
      Ok(Expr::FunctionCall {
        name: head.to_string(),
        args: result,
      })
    }
    Expr::List(items) if head == "List" => {
      let mut result = Vec::new();
      for item in items {
        flatten_with_head(item, depth, head, &mut result);
      }
      Ok(Expr::List(result))
    }
    _ => Ok(list.clone()),
  }
}

/// Flatten[list, {{n1, n2, ...}, ...}] - generalized flatten/transpose
/// Each group specifies which levels to merge together.
/// E.g., {{2}, {1}} transposes, {{1, 2}} merges levels 1 and 2.
pub fn flatten_dims_ast(
  list: &Expr,
  dim_spec: &[Vec<usize>],
) -> Result<Expr, InterpreterError> {
  // Access element at given multi-index, returns None if out of bounds
  fn access(expr: &Expr, indices: &[usize]) -> Option<Expr> {
    if indices.is_empty() {
      return Some(expr.clone());
    }
    match expr {
      Expr::List(items) => {
        if indices[0] < items.len() {
          access(&items[indices[0]], &indices[1..])
        } else {
          None
        }
      }
      _ => None,
    }
  }

  // Get max dimension at each level (handling ragged arrays)
  fn get_max_dim(expr: &Expr, level: usize) -> usize {
    if level == 0 {
      match expr {
        Expr::List(items) => items.len(),
        _ => 0,
      }
    } else {
      match expr {
        Expr::List(items) => items
          .iter()
          .map(|item| get_max_dim(item, level - 1))
          .max()
          .unwrap_or(0),
        _ => 0,
      }
    }
  }

  let max_level = dim_spec
    .iter()
    .flat_map(|g| g.iter())
    .copied()
    .max()
    .unwrap_or(1);

  // Get max sizes at each level
  let mut max_dims = Vec::new();
  for level in 0..max_level {
    max_dims.push(get_max_dim(list, level));
  }

  // Build result by iterating over the output structure.
  // For each group in dim_spec, iterate over the corresponding dimensions.
  // For single-level groups with ragged data, only include elements that exist.
  fn build_result(
    list: &Expr,
    dim_spec: &[Vec<usize>],
    max_dims: &[usize],
    group_idx: usize,
    indices_so_far: &mut Vec<(usize, usize)>,
    max_level: usize,
  ) -> Option<Expr> {
    if group_idx >= dim_spec.len() {
      let mut orig_indices = vec![0usize; max_level];
      for &(level, idx) in indices_so_far.iter() {
        orig_indices[level] = idx;
      }
      access(list, &orig_indices)
    } else {
      let group = &dim_spec[group_idx];
      if group.len() == 1 {
        let level_0based = group[0] - 1;
        let dim_size = max_dims[level_0based];
        let mut items: Vec<Expr> = Vec::new();
        for i in 0..dim_size {
          indices_so_far.push((level_0based, i));
          if let Some(result) = build_result(
            list,
            dim_spec,
            max_dims,
            group_idx + 1,
            indices_so_far,
            max_level,
          ) {
            items.push(result);
          }
          indices_so_far.pop();
        }
        if items.is_empty() {
          None
        } else {
          Some(Expr::List(items))
        }
      } else {
        // Multiple levels merged: iterate over all combinations
        let sizes: Vec<usize> =
          group.iter().map(|&l| max_dims[l - 1]).collect();
        let total: usize = sizes.iter().product();
        let mut items: Vec<Expr> = Vec::new();
        for flat_idx in 0..total {
          let mut remainder = flat_idx;
          let mut sub_indices = Vec::new();
          for &s in sizes.iter().rev() {
            sub_indices.push(remainder % s);
            remainder /= s;
          }
          sub_indices.reverse();
          for (k, &level) in group.iter().enumerate() {
            indices_so_far.push((level - 1, sub_indices[k]));
          }
          if let Some(result) = build_result(
            list,
            dim_spec,
            max_dims,
            group_idx + 1,
            indices_so_far,
            max_level,
          ) {
            items.push(result);
          }
          for _ in 0..group.len() {
            indices_so_far.pop();
          }
        }
        if items.is_empty() {
          None
        } else {
          Some(Expr::List(items))
        }
      }
    }
  }

  match build_result(list, dim_spec, &max_dims, 0, &mut Vec::new(), max_level) {
    Some(result) => Ok(result),
    None => Ok(list.clone()),
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
    Expr::FunctionCall { name, args } => {
      let mut reversed = args.clone();
      reversed.reverse();
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: reversed,
      })
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

/// Reverse[list, n] - reverse at specific levels
pub fn reverse_level_ast(
  list: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Parse level spec: integer n means reverse at level n,
  // {n1, n2} means reverse at levels n1 through n2
  let (min_level, max_level) = match level_spec {
    Expr::Integer(n) => (*n as usize, *n as usize),
    Expr::List(items) if items.len() == 1 => {
      let n = expr_to_i128(&items[0]).unwrap_or(1) as usize;
      (n, n)
    }
    Expr::List(items) if items.len() == 2 => {
      let min = expr_to_i128(&items[0]).unwrap_or(1) as usize;
      let max = expr_to_i128(&items[1]).unwrap_or(1) as usize;
      (min, max)
    }
    _ => (1, 1),
  };

  fn reverse_at_levels(
    expr: &Expr,
    current_level: usize,
    min_level: usize,
    max_level: usize,
  ) -> Expr {
    match expr {
      Expr::List(items) => {
        // First recurse into children
        let children: Vec<Expr> = items
          .iter()
          .map(|item| {
            reverse_at_levels(item, current_level + 1, min_level, max_level)
          })
          .collect();
        // Then reverse at this level if it's in range
        if current_level >= min_level && current_level <= max_level {
          let mut reversed = children;
          reversed.reverse();
          Expr::List(reversed)
        } else {
          Expr::List(children)
        }
      }
      _ => expr.clone(),
    }
  }

  Ok(reverse_at_levels(list, 1, min_level, max_level))
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

  // If it's a 1D list (no sub-lists), return it unchanged
  if !rows.iter().any(|r| matches!(r, Expr::List(_))) {
    return Ok(list.clone());
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

  // If sep is a list, interleave with cycling
  // If sep has same length as items: full interleave {a1,s1,a2,s2,...,an,sn}
  // Otherwise: insert n-1 separators cycling: {a1,s1,a2,s2,...,an}
  if let Expr::List(sep_items) = sep {
    if sep_items.is_empty() {
      return Ok(list.clone());
    }
    let mut result = Vec::new();
    if sep_items.len() == items.len() {
      // Full interleave
      for (i, item) in items.iter().enumerate() {
        result.push(item.clone());
        result.push(sep_items[i].clone());
      }
    } else {
      // Insert n-1 separators, cycling through sep list
      for (i, item) in items.iter().enumerate() {
        result.push(item.clone());
        if i < items.len() - 1 {
          result.push(sep_items[i % sep_items.len()].clone());
        }
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
  let (items, head_name): (&[Expr], Option<&str>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RotateLeft".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  if items.is_empty() {
    return match head_name {
      Some(name) => Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: vec![],
      }),
      None => Ok(Expr::List(vec![])),
    };
  }

  let len = items.len() as i128;
  let shift = ((n % len) + len) % len;
  let shift_usize = shift as usize;

  let mut result = items[shift_usize..].to_vec();
  result.extend_from_slice(&items[..shift_usize]);

  match head_name {
    Some(name) => Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: result,
    }),
    None => Ok(Expr::List(result)),
  }
}

/// AST-based RotateRight: rotate list right by n positions.
pub fn rotate_right_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  rotate_left_ast(list, -n)
}

/// Multi-dimensional RotateLeft/RotateRight: rotate at each level.
/// RotateLeft[matrix, {r, c}] rotates rows by r, then each row by c.
pub fn rotate_multi_ast(
  list: &Expr,
  shifts: &[Expr],
  left: bool,
) -> Result<Expr, InterpreterError> {
  if shifts.is_empty() {
    return Ok(list.clone());
  }

  let first_shift = match expr_to_i128(&shifts[0]) {
    Some(n) => {
      if left {
        n
      } else {
        -n
      }
    }
    None => {
      let fn_name = if left { "RotateLeft" } else { "RotateRight" };
      return Ok(Expr::FunctionCall {
        name: fn_name.to_string(),
        args: vec![list.clone(), Expr::List(shifts.to_vec())],
      });
    }
  };

  // Rotate the top level
  let rotated = rotate_left_ast(list, first_shift)?;

  // If there are more shifts, apply them to each sublist
  if shifts.len() > 1 {
    let rest_shifts = &shifts[1..];
    match rotated {
      Expr::List(items) => {
        let mut new_items = Vec::new();
        for item in &items {
          new_items.push(rotate_multi_ast(item, rest_shifts, left)?);
        }
        Ok(Expr::List(new_items))
      }
      _ => Ok(rotated),
    }
  } else {
    Ok(rotated)
  }
}

/// AST-based PadLeft: pad list on the left to length n.
/// If n < len, truncates from the left.
/// Get cyclic padding element at a given position.
/// The padding is cyclically repeated, and the cycle is aligned so that
/// the position right after the original list corresponds to padding[0].
fn cyclic_pad_element(pad: &Expr, pos: i128, list_end: i128) -> Expr {
  match pad {
    Expr::List(items) if !items.is_empty() => {
      let cycle_len = items.len() as i128;
      let idx = ((pos - list_end) % cycle_len + cycle_len) % cycle_len;
      items[idx as usize].clone()
    }
    _ => pad.clone(),
  }
}

pub fn pad_left_ast(
  list: &Expr,
  n: i128,
  pad: &Expr,
  offset: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let (items, head_name) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PadLeft".to_string(),
        args: vec![list.clone(), Expr::Integer(n), pad.clone()],
      });
    }
  };

  if n <= 0 {
    return match head_name {
      Some(h) => Ok(Expr::FunctionCall {
        name: h.to_string(),
        args: vec![],
      }),
      None => Ok(Expr::List(vec![])),
    };
  }

  let len = items.len() as i128;
  let n_usize = n as usize;

  let result_items = if n < len {
    let skip = (len - n) as usize;
    items[skip..].to_vec()
  } else if n == len && offset.is_none() {
    items.to_vec()
  } else {
    // PadLeft: original list goes at position (n - len - m) where m = offset (default 0)
    let m = offset.unwrap_or(0).max(0);
    let start = ((n - len - m).max(0)) as usize;
    let list_end = start + items.len();
    let mut result = Vec::with_capacity(n_usize);
    for i in 0..n_usize {
      if i >= start && i < list_end {
        result.push(items[i - start].clone());
      } else {
        result.push(cyclic_pad_element(pad, i as i128, list_end as i128));
      }
    }
    result
  };

  match head_name {
    Some(h) => crate::evaluator::evaluate_function_call_ast(h, &result_items),
    None => Ok(Expr::List(result_items)),
  }
}

/// AST-based PadRight: pad list on the right to length n.
/// If n < len, truncates from the right.
pub fn pad_right_ast(
  list: &Expr,
  n: i128,
  pad: &Expr,
  offset: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let (items, head_name) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PadRight".to_string(),
        args: vec![list.clone(), Expr::Integer(n), pad.clone()],
      });
    }
  };

  if n <= 0 {
    return match head_name {
      Some(h) => Ok(Expr::FunctionCall {
        name: h.to_string(),
        args: vec![],
      }),
      None => Ok(Expr::List(vec![])),
    };
  }

  let len = items.len() as i128;
  let n_usize = n as usize;

  let result_items = if n < len {
    items[..n_usize].to_vec()
  } else if n == len && offset.is_none() {
    items.to_vec()
  } else {
    // PadRight: original list goes at position m where m = offset (default 0)
    let m = offset.unwrap_or(0).max(0);
    let start = m.min(n - len).max(0) as usize;
    let list_end = start + items.len();
    let mut result = Vec::with_capacity(n_usize);
    for i in 0..n_usize {
      if i >= start && i < list_end {
        result.push(items[i - start].clone());
      } else {
        result.push(cyclic_pad_element(pad, i as i128, list_end as i128));
      }
    }
    result
  };

  match head_name {
    Some(h) => crate::evaluator::evaluate_function_call_ast(h, &result_items),
    None => Ok(Expr::List(result_items)),
  }
}

/// AST-based Join: join multiple lists.
pub fn join_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Determine the common head
  let head: Option<&str> = match &lists[0] {
    Expr::List(_) => None, // List head
    Expr::FunctionCall { name, .. } => Some(name.as_str()),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Join".to_string(),
        args: lists.to_vec(),
      });
    }
  };

  let mut result = Vec::new();
  for list in lists {
    match (list, head) {
      (Expr::List(items), None) => result.extend(items.iter().cloned()),
      (Expr::FunctionCall { name, args }, Some(h)) if name == h => {
        result.extend(args.iter().cloned());
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Join".to_string(),
          args: lists.to_vec(),
        });
      }
    }
  }

  match head {
    None => Ok(Expr::List(result)),
    Some(h) => {
      // Re-evaluate to allow simplification (e.g., Plus combines terms)
      crate::evaluator::evaluate_function_call_ast(h, &result)
    }
  }
}

/// AST-based Append: append element to list.
pub fn append_ast(list: &Expr, elem: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut result = items.clone();
      result.push(elem.clone());
      Ok(Expr::List(result))
    }
    Expr::FunctionCall { name, args } => {
      let mut new_args = args.clone();
      new_args.push(elem.clone());
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      })
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
    Expr::FunctionCall { name, args } => {
      let mut new_args = vec![elem.clone()];
      new_args.extend(args.iter().cloned());
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Prepend".to_string(),
      args: vec![list.clone(), elem.clone()],
    }),
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
