#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// Extract key-value pair from a Rule expression for use in associations.
fn extract_rule_pair(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some((*pattern.clone(), *replacement.clone())),
    _ => None,
  }
}

/// AST-based Partition: break list into sublists of length n.
/// Partition[{a, b, c, d, e}, 2] -> {{a, b}, {c, d}}
pub fn partition_ast(
  list: &Expr,
  n: i128,
  d: Option<i128>,
  align: Option<&Expr>,
  pad: Option<&Expr>,
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
  let len = items.len();

  // Parse alignment spec {kL, kR}
  let (k_l, k_r) = if let Some(align_expr) = align {
    if let Expr::List(elems) = align_expr {
      if elems.len() == 2 {
        if let (Some(kl), Some(kr)) = (
          super::utilities::expr_to_i128(&elems[0]),
          super::utilities::expr_to_i128(&elems[1]),
        ) {
          (kl, kr)
        } else {
          (1, -1) // default: no overhang
        }
      } else {
        (1, -1)
      }
    } else {
      (1, -1)
    }
  } else {
    (1, -1) // default: Partition drops incomplete partitions
  };

  // Cyclic wrapping: {1, 1} or {-1, -1} or other alignment
  let is_cyclic = align.is_some() && pad.is_none();
  let is_default = k_l == 1 && k_r == -1;

  if is_default && !is_cyclic {
    // Standard partitioning without overhang
    let mut results = Vec::new();
    let mut i = 0;
    while i + n_usize <= len {
      results.push(Expr::List(items[i..i + n_usize].to_vec()));
      i += d_usize;
    }
    return Ok(Expr::List(results));
  }

  if is_cyclic && len > 0 {
    // Cyclic partitioning with alignment {kL, kR}
    // kL: position (1-based) in the first partition where list[0] goes
    // kR: position (1-based or negative) in the last partition where list[-1] goes
    //
    // The starting offset is -(kL-1) for positive kL, or -(n+kL) for negative kL
    let start_offset: isize = if k_l > 0 {
      -(k_l as isize - 1)
    } else {
      -(n as isize + k_l as isize)
    };

    // kR determines the position of the last element in the last partition.
    // Position where last list element must appear (0-based within partition):
    let end_pos: isize = if k_r > 0 {
      k_r as isize - 1
    } else {
      n as isize + k_r as isize
    };
    // The last partition must start at an offset such that
    // start + end_pos == len - 1 (mod alignment with d).
    // last_start + end_pos covers items[len-1]
    // => last_start = len - 1 - end_pos
    let last_start = len as isize - 1 - end_pos;

    let mut results = Vec::new();
    let mut offset = start_offset;
    while offset <= last_start {
      let mut chunk = Vec::with_capacity(n_usize);
      for j in 0..n_usize {
        let idx =
          ((offset + j as isize) % len as isize + len as isize) as usize % len;
        chunk.push(items[idx].clone());
      }
      results.push(Expr::List(chunk));
      offset += d_usize as isize;
    }
    return Ok(Expr::List(results));
  }

  // Overhang with padding (5-arg form with explicit pad)
  let mut results = Vec::new();
  let mut i = 0;
  while i + n_usize <= len {
    results.push(Expr::List(items[i..i + n_usize].to_vec()));
    i += d_usize;
  }

  if let Some(pad_expr) = pad
    && i < len
  {
    let remaining = &items[i..];
    match pad_expr {
      Expr::List(pad_elems) if pad_elems.is_empty() => {
        // {} means allow short sublists (no padding)
        results.push(Expr::List(remaining.to_vec()));
      }
      _ => {
        // Pad with the given element(s) to fill to size n
        let mut chunk = remaining.to_vec();
        let pad_items: Vec<Expr> = match pad_expr {
          Expr::List(items) => items.clone(),
          other => vec![other.clone()],
        };
        if !pad_items.is_empty() {
          let mut pad_idx = 0;
          while chunk.len() < n_usize {
            chunk.push(pad_items[pad_idx % pad_items.len()].clone());
            pad_idx += 1;
          }
        }
        results.push(Expr::List(chunk));
      }
    }
  }

  Ok(Expr::List(results))
}

/// AST-based Flatten: flatten nested lists.
pub fn flatten_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(_) => {
      let mut result = Vec::new();
      let mut stack: Vec<&Expr> = vec![list];
      while let Some(expr) = stack.pop() {
        match expr {
          Expr::List(items) => {
            // Push items in reverse so they're processed in order
            for item in items.iter().rev() {
              stack.push(item);
            }
          }
          _ => result.push(expr.clone()),
        }
      }
      Ok(Expr::List(result))
    }
    Expr::FunctionCall { name, args } => {
      // Flatten[f[f[a, b], f[c, d]]] -> f[a, b, c, d]
      // Recursively flatten subexpressions with the same head
      let mut result = Vec::new();
      flatten_same_head(name, args, &mut result);
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: result,
      })
    }
    _ => Ok(list.clone()),
  }
}

/// Helper to recursively flatten subexpressions with the same head
fn flatten_same_head(head: &str, args: &[Expr], result: &mut Vec<Expr>) {
  for arg in args {
    match arg {
      Expr::FunctionCall {
        name,
        args: sub_args,
      } if name == head => {
        flatten_same_head(head, sub_args, result);
      }
      _ => result.push(arg.clone()),
    }
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

  fn flatten_head_to_depth(
    head: &str,
    expr: &Expr,
    depth: i128,
    result: &mut Vec<Expr>,
  ) {
    if depth <= 0 {
      result.push(expr.clone());
      return;
    }
    match expr {
      Expr::FunctionCall { name, args } if name == head => {
        for arg in args {
          flatten_head_to_depth(head, arg, depth - 1, result);
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
    Expr::FunctionCall { name, args } => {
      let mut result = Vec::new();
      for arg in args {
        flatten_head_to_depth(name, arg, depth, &mut result);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: result,
      })
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

  // If the outer head matches the target head, flatten its children directly.
  // Otherwise, recurse into children to flatten matching subexpressions,
  // preserving the outer structure.
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
    Expr::FunctionCall { name, args } => {
      // Outer head doesn't match: flatten matching children into parent
      let mut result = Vec::new();
      for item in args {
        flatten_with_head(item, depth, head, &mut result);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: result,
      })
    }
    Expr::List(items) => {
      // Outer is List but target is not List: flatten matching children
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
    Expr::Association(pairs) => {
      let mut reversed = pairs.clone();
      reversed.reverse();
      Ok(Expr::Association(reversed))
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
/// Get the (rectangular) dimensions of a nested list, descending only
/// through the levels where all sibling sublists have the same length.
fn tensor_dims(expr: &Expr) -> Vec<usize> {
  match expr {
    Expr::List(items) => {
      let mut dims = vec![items.len()];
      if items.is_empty() {
        return dims;
      }
      let first = tensor_dims(&items[0]);
      // Every sibling must share the same deeper shape to count as a
      // regular tensor at this level.
      for item in items.iter().skip(1) {
        if tensor_dims(item) != first {
          return dims;
        }
      }
      dims.extend(first);
      dims
    }
    _ => vec![],
  }
}

/// Read a tensor element at the given multi-index (1-based per Wolfram).
fn tensor_get(expr: &Expr, idx: &[usize]) -> Expr {
  let mut cur = expr;
  for &i in idx {
    match cur {
      Expr::List(items) => {
        cur = &items[i - 1];
      }
      _ => return cur.clone(),
    }
  }
  cur.clone()
}

/// Build a tensor of the given shape by calling `f` with a 1-based
/// multi-index for each cell.
fn tensor_build(shape: &[usize], f: &mut dyn FnMut(&[usize]) -> Expr) -> Expr {
  let mut idx = vec![1usize; shape.len()];
  tensor_build_recursive(shape, 0, &mut idx, f)
}

fn tensor_build_recursive(
  shape: &[usize],
  level: usize,
  idx: &mut Vec<usize>,
  f: &mut dyn FnMut(&[usize]) -> Expr,
) -> Expr {
  if level == shape.len() {
    return f(idx);
  }
  let mut items = Vec::with_capacity(shape[level]);
  for i in 1..=shape[level] {
    idx[level] = i;
    items.push(tensor_build_recursive(shape, level + 1, idx, f));
  }
  Expr::List(items)
}

/// Transpose[list, {n1, n2, ..., nk}] — permute the levels of a
/// rank-k tensor so that the k-th level of `list` becomes the n_k-th
/// level of the result.
pub fn transpose_perm_ast(
  list: &Expr,
  perm: &[Expr],
) -> Result<Expr, InterpreterError> {
  let dims = tensor_dims(list);
  let rank = dims.len();
  if rank < perm.len() {
    return Err(InterpreterError::EvaluationError(
      "Transpose: permutation length exceeds tensor rank".into(),
    ));
  }
  // Parse and validate the permutation into 1-based axis indices.
  let mut sigma: Vec<usize> = Vec::with_capacity(perm.len());
  for p in perm {
    match p {
      Expr::Integer(n) if *n >= 1 && (*n as usize) <= rank => {
        sigma.push(*n as usize);
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Transpose: second argument must be a permutation of 1..n".into(),
        ));
      }
    }
  }
  // Verify it's a permutation of 1..=perm.len() (must touch each axis
  // exactly once; all remaining axes are left untouched).
  let mut seen = vec![false; rank];
  for &s in &sigma {
    if seen[s - 1] {
      return Err(InterpreterError::EvaluationError(
        "Transpose: duplicate axis in permutation".into(),
      ));
    }
    seen[s - 1] = true;
  }
  // Identity permutation → return the list unchanged.
  let is_identity =
    sigma.iter().enumerate().all(|(k, &n)| k + 1 == n) && sigma.len() == rank;
  if is_identity {
    return Ok(list.clone());
  }
  if sigma.len() != rank {
    return Err(InterpreterError::EvaluationError(
      "Transpose: permutation must cover every level of the tensor".into(),
    ));
  }

  // Compute the inverse permutation so result axis t corresponds to
  // list axis sigma^-1(t).
  let mut inv = vec![0usize; rank];
  for (k, &n) in sigma.iter().enumerate() {
    inv[n - 1] = k + 1;
  }
  let result_shape: Vec<usize> = (0..rank).map(|t| dims[inv[t] - 1]).collect();

  Ok(tensor_build(&result_shape, &mut |j| {
    // Original index i_s = j_{sigma(s)} = j_{sigma[s-1] - 1}
    let i: Vec<usize> = (0..rank).map(|s| j[sigma[s] - 1]).collect();
    tensor_get(list, &i)
  }))
}

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

/// AST-based Riffle with step spec.
/// Riffle[list, x, n] - insert x at every n-th output position while the
///   list still has elements.
/// Riffle[list, x, {a, b, s}] - insert x at positions a, a+s, a+2s, ... .
///   If b is positive, insertions stop once they would exceed b.
///   If b is negative (e.g. -1), insertions continue as long as the next
///   position equals the current output index; a negative b allows a
///   trailing insert past the last list element when the arithmetic
///   progression lands on the output-length position.
pub fn riffle_extended_ast(
  list: &Expr,
  sep: &Expr,
  spec: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Riffle".to_string(),
        args: vec![list.clone(), sep.clone(), spec.clone()],
      });
    }
  };

  // Parse the spec into (start, end_opt, step, is_simple).
  // is_simple = true for `Riffle[list, x, n]` (no explicit end).
  let (start, end_opt, step, is_simple) = match spec {
    Expr::Integer(n) => (*n, None, *n, true),
    Expr::List(spec_items) if spec_items.len() == 3 => {
      let a = match &spec_items[0] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Riffle".to_string(),
            args: vec![list.clone(), sep.clone(), spec.clone()],
          });
        }
      };
      let b = match &spec_items[1] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Riffle".to_string(),
            args: vec![list.clone(), sep.clone(), spec.clone()],
          });
        }
      };
      let s = match &spec_items[2] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Riffle".to_string(),
            args: vec![list.clone(), sep.clone(), spec.clone()],
          });
        }
      };
      (a, Some(b), s, false)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Riffle".to_string(),
        args: vec![list.clone(), sep.clone(), spec.clone()],
      });
    }
  };

  if step <= 0 || start <= 0 {
    return Err(InterpreterError::EvaluationError(
      "Riffle: start and step must be positive integers".into(),
    ));
  }

  // Collect separators: either a list (to cycle through) or a single value.
  let sep_values: Vec<Expr> = match sep {
    Expr::List(xs) if !xs.is_empty() => xs.clone(),
    Expr::List(_) => return Ok(Expr::List(items)),
    other => vec![other.clone()],
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let mut result: Vec<Expr> = Vec::new();
  let mut list_idx: usize = 0;
  let mut out_idx: i128 = 1;
  let mut next_insert: i128 = start;
  let mut insert_count: usize = 0;

  loop {
    let list_done = list_idx >= items.len();
    let want_insert = next_insert == out_idx;

    // For a positive `b`, insertions are only allowed while `next_insert <= b`.
    let insert_allowed = match end_opt {
      Some(b) if b > 0 => next_insert <= b,
      _ => true,
    };

    if is_simple {
      // Simple `Riffle[list, x, n]`: stop as soon as the list is exhausted,
      // regardless of any pending insert at the current output position.
      if list_done {
        break;
      }
      if want_insert {
        result.push(sep_values[insert_count % sep_values.len()].clone());
        insert_count += 1;
        next_insert += step;
      } else {
        result.push(items[list_idx].clone());
        list_idx += 1;
      }
      out_idx += 1;
      continue;
    }

    // `{a, b, s}` form.
    if want_insert && insert_allowed {
      result.push(sep_values[insert_count % sep_values.len()].clone());
      insert_count += 1;
      next_insert += step;
      out_idx += 1;
    } else if !list_done {
      result.push(items[list_idx].clone());
      list_idx += 1;
      out_idx += 1;
    } else {
      break;
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
    match &rotated {
      Expr::List(items) => {
        let mut new_items = Vec::new();
        for item in items {
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

/// Build a nested-list "filler" of shape `ns` filled with the scalar `pad`.
/// E.g. `build_pad_shape(x, &[3, 2])` → `{{x, x}, {x, x}, {x, x}}`.
fn build_pad_shape(pad: &Expr, ns: &[i128]) -> Expr {
  if ns.is_empty() {
    return pad.clone();
  }
  let n = ns[0].max(0) as usize;
  let inner = build_pad_shape(pad, &ns[1..]);
  Expr::List(vec![inner; n])
}

/// Multi-dimensional PadLeft with a scalar pad value. Recursively pads
/// each level and prepends "filler" shapes for missing outer rows.
pub fn pad_left_multidim(
  list: &Expr,
  ns: &[i128],
  pad: &Expr,
) -> Result<Expr, InterpreterError> {
  if ns.is_empty() {
    return Ok(list.clone());
  }
  let n = ns[0];
  let rest = &ns[1..];

  let items: Vec<Expr> = match list {
    Expr::List(items) => items.clone(),
    _ => vec![],
  };

  let mut padded_children: Vec<Expr> = Vec::with_capacity(items.len());
  for item in &items {
    padded_children.push(pad_left_multidim(item, rest, pad)?);
  }

  let len = padded_children.len() as i128;

  if n <= 0 {
    return Ok(Expr::List(vec![]));
  }
  if len >= n {
    // Truncate from the left (mirror single-dim PadLeft behavior).
    let skip = (len - n) as usize;
    return Ok(Expr::List(padded_children[skip..].to_vec()));
  }

  let filler = build_pad_shape(pad, rest);
  let needed = (n - len) as usize;
  let mut result: Vec<Expr> = vec![filler; needed];
  result.extend(padded_children);
  Ok(Expr::List(result))
}

/// Multi-dimensional PadRight with a scalar pad value.
pub fn pad_right_multidim(
  list: &Expr,
  ns: &[i128],
  pad: &Expr,
) -> Result<Expr, InterpreterError> {
  if ns.is_empty() {
    return Ok(list.clone());
  }
  let n = ns[0];
  let rest = &ns[1..];

  let items: Vec<Expr> = match list {
    Expr::List(items) => items.clone(),
    _ => vec![],
  };

  let mut padded_children: Vec<Expr> = Vec::with_capacity(items.len());
  for item in &items {
    padded_children.push(pad_right_multidim(item, rest, pad)?);
  }

  let len = padded_children.len() as i128;

  if n <= 0 {
    return Ok(Expr::List(vec![]));
  }
  if len >= n {
    return Ok(Expr::List(padded_children[..n as usize].to_vec()));
  }

  let filler = build_pad_shape(pad, rest);
  let needed = (n - len) as usize;
  padded_children.extend(vec![filler; needed]);
  Ok(Expr::List(padded_children))
}

/// AST-based PadLeft: pad list on the left to length n.
/// If n < len, truncates from the left.
/// Get cyclic padding element at a given position (0-based).
/// `anchor` controls the cycle alignment: the cycle aligns so that
/// position `anchor` corresponds to cycle index 0.
fn cyclic_pad_element(pad: &Expr, pos: i128, anchor: i128) -> Expr {
  match pad {
    Expr::List(items) if !items.is_empty() => {
      let cycle_len = items.len() as i128;
      let idx = ((pos - anchor) % cycle_len + cycle_len) % cycle_len;
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

  let result_items = if n < len && offset.is_none() {
    let skip = (len - n) as usize;
    items[skip..].to_vec()
  } else if n == len && offset.is_none() {
    items.to_vec()
  } else {
    // PadLeft positions the original list so that its last element is at
    // output index `n - 1 - m` (0-indexed), i.e. `list_start = n - len - m`.
    // Elements whose target index falls outside `[0, n)` are dropped.
    let m = offset.unwrap_or(0);
    let list_start: i128 = n - len - m;
    let list_end: i128 = list_start + len;
    let mut result = Vec::with_capacity(n_usize);
    for i in 0..n {
      if i >= list_start && i < list_end {
        result.push(items[(i - list_start) as usize].clone());
      } else {
        result.push(cyclic_pad_element(pad, i, list_end));
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

  let result_items = if n < len && offset.is_none() {
    items[..n_usize].to_vec()
  } else if n == len && offset.is_none() {
    items.to_vec()
  } else {
    // PadRight positions the original list so its first element is at
    // output index `m` (0-indexed), i.e. `list_start = m`. Elements whose
    // target index falls outside `[0, n)` are dropped.
    let m = offset.unwrap_or(0);
    let list_start: i128 = m;
    let list_end: i128 = list_start + len;
    let mut result = Vec::with_capacity(n_usize);
    for i in 0..n {
      if i >= list_start && i < list_end {
        result.push(items[(i - list_start) as usize].clone());
      } else {
        result.push(cyclic_pad_element(pad, i, list_start));
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

  // Check if all arguments are associations
  if lists.iter().all(|l| matches!(l, Expr::Association(_))) {
    let mut result: Vec<(Expr, Expr)> = Vec::new();
    for list in lists {
      if let Expr::Association(pairs) = list {
        for (k, v) in pairs {
          // Later values override earlier ones for the same key
          let key_str = crate::syntax::expr_to_string(k);
          if let Some(pos) = result
            .iter()
            .position(|(ek, _)| crate::syntax::expr_to_string(ek) == key_str)
          {
            result[pos] = (k.clone(), v.clone());
          } else {
            result.push((k.clone(), v.clone()));
          }
        }
      }
    }
    return Ok(Expr::Association(result));
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

/// Join at a specified level.
/// Join[list1, list2, ..., n] joins the lists at level n.
/// Level 1 is the default (top-level join).
/// Level 2 means descend one level and join corresponding sublists.
pub fn join_at_level_ast(
  lists: &[Expr],
  level: usize,
) -> Result<Expr, InterpreterError> {
  if level <= 1 {
    return join_ast(lists);
  }

  // For level > 1, all lists must have the same outer structure.
  // Get the items from the first list to determine the outer length.
  let first_items = match &lists[0] {
    Expr::List(items) => items,
    _ => return join_ast(lists),
  };
  let outer_len = first_items.len();

  // Verify all lists are Lists and collect their items
  let mut all_items: Vec<&Vec<Expr>> = Vec::new();
  for list in lists {
    match list {
      Expr::List(items) => all_items.push(items),
      _ => return join_ast(lists),
    }
  }

  // For each position in the outer structure, recursively join at level-1
  let mut result = Vec::with_capacity(outer_len);
  for i in 0..outer_len {
    let mut sublists = Vec::new();
    for items in &all_items {
      if i < items.len() {
        sublists.push(items[i].clone());
      }
    }
    result.push(join_at_level_ast(&sublists, level - 1)?);
  }

  Ok(Expr::List(result))
}

/// AST-based Append: append element to list.
pub fn append_ast(list: &Expr, elem: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::Association(pairs) => {
      // Append a rule to an association
      // If the key already exists, remove the old entry (new one goes to end)
      if let Some((k, v)) = extract_rule_pair(elem) {
        let key_str = crate::syntax::expr_to_string(&k);
        let mut result: Vec<(Expr, Expr)> = pairs
          .iter()
          .filter(|(ek, _)| crate::syntax::expr_to_string(ek) != key_str)
          .cloned()
          .collect();
        result.push((k, v));
        Ok(Expr::Association(result))
      } else {
        Ok(Expr::FunctionCall {
          name: "Append".to_string(),
          args: vec![list.clone(), elem.clone()],
        })
      }
    }
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
    Expr::Association(pairs) => {
      // Prepend a rule to an association
      // If the key already exists, remove the old entry (new one goes to front)
      if let Some((k, v)) = extract_rule_pair(elem) {
        let key_str = crate::syntax::expr_to_string(&k);
        let mut result = vec![(k, v)];
        result.extend(
          pairs
            .iter()
            .filter(|(ek, _)| crate::syntax::expr_to_string(ek) != key_str)
            .cloned(),
        );
        Ok(Expr::Association(result))
      } else {
        Ok(Expr::FunctionCall {
          name: "Prepend".to_string(),
          args: vec![list.clone(), elem.clone()],
        })
      }
    }
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

/// Catenate[{list1, list2, ...}] - concatenates lists or associations
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
      Expr::Association(pairs) => {
        result.extend(pairs.iter().map(|(_, v)| v.clone()));
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Catenate expects all elements to be lists or associations".into(),
        ));
      }
    }
  }
  Ok(Expr::List(result))
}
