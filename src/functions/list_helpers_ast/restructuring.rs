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
      crate::emit_message(&format!(
        "Partition::npart: The expression {} cannot be partitioned.",
        crate::syntax::format_expr(list, crate::syntax::ExprForm::Output)
      ));
      let mut full_args = vec![list.clone(), Expr::Integer(n)];
      if let Some(d) = d {
        full_args.push(Expr::Integer(d));
      }
      if let Some(a) = align {
        full_args.push(a.clone());
      }
      if let Some(p) = pad {
        full_args.push(p.clone());
      }
      return Ok(Expr::FunctionCall {
        name: "Partition".to_string(),
        args: full_args.into(),
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
      results.push(Expr::List(items[i..i + n_usize].to_vec().into()));
      i += d_usize;
    }
    return Ok(Expr::List(results.into()));
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
      results.push(Expr::List(chunk.into()));
      offset += d_usize as isize;
    }
    return Ok(Expr::List(results.into()));
  }

  // Overhang with padding (5-arg form with explicit pad): walk the same
  // offsets as the cyclic path, but fill out-of-range positions from the
  // padding instead of wrapping. A cyclic padding list is indexed by the
  // global list position mod its length (on both sides); the empty list
  // clips rows to the in-range elements.
  let start_offset: isize = if k_l > 0 {
    -(k_l as isize - 1)
  } else {
    -(n as isize + k_l as isize)
  };
  let end_pos: isize = if k_r > 0 {
    k_r as isize - 1
  } else {
    n as isize + k_r as isize
  };
  let last_start = len as isize - 1 - end_pos;

  let mut results = Vec::new();
  let mut offset = start_offset;
  while offset <= last_start {
    let mut chunk: Vec<Expr> = Vec::with_capacity(n_usize);
    for j in 0..n_usize {
      let idx = offset + j as isize;
      if idx >= 0 && (idx as usize) < len {
        chunk.push(items[idx as usize].clone());
      } else if let Some(pad_expr) = pad {
        match pad_expr {
          Expr::List(pad_elems) if pad_elems.is_empty() => {} // clip
          Expr::List(pad_elems) => {
            let m = pad_elems.len() as isize;
            let pi = (((idx % m) + m) % m) as usize;
            chunk.push(pad_elems[pi].clone());
          }
          single => chunk.push(single.clone()),
        }
      }
    }
    results.push(Expr::List(chunk.into()));
    offset += d_usize as isize;
  }
  Ok(Expr::List(results.into()))
}

pub fn partition_multi_dim_ast(
  expr: &Expr,
  sizes: &[i128],
  offsets: &[i128],
) -> Result<Expr, InterpreterError> {
  fn partition_one_dim(list: &Expr, n: usize, d: usize) -> Option<Vec<Expr>> {
    let items = match list {
      Expr::List(v) => v,
      _ => return None,
    };
    let mut chunks: Vec<Expr> = Vec::new();
    let mut i = 0;
    while i + n <= items.len() {
      chunks.push(Expr::List(items[i..i + n].to_vec().into()));
      i += d;
    }
    Some(chunks)
  }

  fn recurse(list: &Expr, sizes: &[usize], offsets: &[usize]) -> Option<Expr> {
    if sizes.is_empty() {
      return Some(list.clone());
    }
    let outer = partition_one_dim(list, sizes[0], offsets[0])?;
    let rest_sizes = &sizes[1..];
    let rest_offsets = &offsets[1..];
    if rest_sizes.is_empty() {
      return Some(Expr::List(outer.into()));
    }
    // Each outer chunk is a List whose elements we must partition along the
    // remaining axes and then transpose so the next-axis partitioning
    // happens at the right level.
    let mut recurred: Vec<Vec<Expr>> = Vec::new();
    for chunk in &outer {
      // chunk is `Expr::List(rows)`; we want to partition each row's
      // deeper dimensions, then reassemble.
      let rows = match chunk {
        Expr::List(v) => v,
        _ => return None,
      };
      let per_row: Option<Vec<Expr>> = rows
        .iter()
        .map(|r| recurse(r, rest_sizes, rest_offsets))
        .collect();
      let per_row = per_row?;
      recurred.push(per_row);
    }
    // `recurred` is indexed [outer_block][row_within_block] and each entry
    // is the tensor of sub-partitions from recurse. We need to transpose to
    // group by the next-axis block position.
    // Each per_row[row_within_block] is a List of `n_2`-sized blocks along
    // dim 2, then further partitioned below. We treat each per_row as the
    // inner partition for that "row slot", then assemble outer-axis-first.
    //
    // For the 2-dim case, per_row[r] = List of column-blocks. We want
    // outer[i] = (i-th block) = List of column-blocks, where each column
    // block itself is List of rows (size n_1). Essentially pivot so columns
    // vary in the inner axis.
    //
    // The simplest reconstruction: for each outer block, build a List of
    // items where the k-th item is taken from per_row[*][k].
    let mut result_outer: Vec<Expr> = Vec::new();
    for per_row in &recurred {
      // Number of inner blocks per row (uniform). Assume well-formed input.
      let inner_count = match per_row.first() {
        Some(Expr::List(v)) => v.len(),
        _ => return None,
      };
      let mut inner_blocks: Vec<Expr> = Vec::with_capacity(inner_count);
      for k in 0..inner_count {
        let mut gathered: Vec<Expr> = Vec::with_capacity(per_row.len());
        for row in per_row {
          if let Expr::List(items) = row
            && let Some(item) = items.get(k)
          {
            gathered.push(item.clone());
          } else {
            return None;
          }
        }
        inner_blocks.push(Expr::List(gathered.into()));
      }
      result_outer.push(Expr::List(inner_blocks.into()));
    }
    Some(Expr::List(result_outer.into()))
  }

  let sizes_u: Vec<usize> = sizes.iter().map(|&x| x as usize).collect();
  let offsets_u: Vec<usize> = offsets.iter().map(|&x| x as usize).collect();
  match recurse(expr, &sizes_u, &offsets_u) {
    Some(r) => Ok(r),
    None => {
      let mut call_args = vec![expr.clone()];
      call_args.push(Expr::List(
        sizes.iter().map(|&x| Expr::Integer(x)).collect(),
      ));
      call_args.push(Expr::List(
        offsets.iter().map(|&x| Expr::Integer(x)).collect(),
      ));
      Ok(Expr::FunctionCall {
        name: "Partition".to_string(),
        args: call_args.into(),
      })
    }
  }
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
      Ok(Expr::List(result.into()))
    }
    Expr::FunctionCall { name, args } => {
      // Flatten[f[f[a, b], f[c, d]]] -> f[a, b, c, d]
      // Recursively flatten subexpressions with the same head
      let mut result = Vec::new();
      flatten_same_head(name, args, &mut result);
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: result.into(),
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
      Ok(Expr::List(result.into()))
    }
    Expr::FunctionCall { name, args } => {
      let mut result = Vec::new();
      for arg in args {
        flatten_head_to_depth(name, arg, depth, &mut result);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: result.into(),
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
        args: result.into(),
      })
    }
    Expr::List(items) if head == "List" => {
      let mut result = Vec::new();
      for item in items {
        flatten_with_head(item, depth, head, &mut result);
      }
      Ok(Expr::List(result.into()))
    }
    Expr::FunctionCall { name, args } => {
      // Outer head doesn't match: flatten matching children into parent
      let mut result = Vec::new();
      for item in args {
        flatten_with_head(item, depth, head, &mut result);
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: result.into(),
      })
    }
    Expr::List(items) => {
      // Outer is List but target is not List: flatten matching children
      let mut result = Vec::new();
      for item in items {
        flatten_with_head(item, depth, head, &mut result);
      }
      Ok(Expr::List(result.into()))
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

  // Depth of the input (how many nested List heads there are). Must be at
  // least the highest level mentioned in `dim_spec`.
  fn depth(expr: &Expr) -> usize {
    match expr {
      Expr::List(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
      _ => 0,
    }
  }
  let expr_depth = depth(list);
  let spec_max = dim_spec
    .iter()
    .flat_map(|g| g.iter())
    .copied()
    .max()
    .unwrap_or(1);
  let max_level = expr_depth.max(spec_max).max(1);

  // Append any levels in 1..=max_level that `dim_spec` doesn't mention as
  // singleton groups at the end, so `Flatten[list, {{2}}]` behaves like
  // `Flatten[list, {{2}, {1}, {3}, …}]` (matches wolframscript).
  let mut effective_spec: Vec<Vec<usize>> = dim_spec.to_vec();
  let mut mentioned: std::collections::HashSet<usize> = effective_spec
    .iter()
    .flat_map(|g| g.iter().copied())
    .collect();
  for level in 1..=max_level {
    if !mentioned.contains(&level) {
      effective_spec.push(vec![level]);
      mentioned.insert(level);
    }
  }
  let dim_spec: &[Vec<usize>] = &effective_spec;

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
          Some(Expr::List(items.into()))
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
          Some(Expr::List(items.into()))
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
      args: vec![list.clone()].into(),
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
          Expr::List(reversed.into())
        } else {
          Expr::List(children.into())
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
  Expr::List(items.into())
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

/// Outcome of a `TensorTranspose` call: either the transposed tensor or a
/// validation failure that the dispatcher turns into the matching WL message.
pub enum TensorTransposeResult {
  Ok(Expr),
  /// Permutation moves slots beyond the tensor rank (`TensorTranspose::ttrank`).
  RankError {
    rank: usize,
  },
  /// The second argument is not a valid permutation
  /// (`TensorTranspose::symmperm`).
  SymmPerm,
}

/// `TensorTranspose[t, perm]` — transpose the levels of tensor `t` according to
/// the permutation `perm`. Unlike `Transpose`, `perm` must be a full
/// permutation of `1..rank`. With no `perm`, the first two levels are swapped.
pub fn tensor_transpose_ast(
  list: &Expr,
  perm: Option<&[Expr]>,
) -> TensorTransposeResult {
  let dims = tensor_dims(list);
  let rank = dims.len();

  // Build the partial permutation as 1-based slot indices. It may be shorter
  // than the rank, in which case the trailing levels are left untouched.
  let partial: Vec<usize> = match perm {
    Some(perm) => {
      let mut s = Vec::with_capacity(perm.len());
      for p in perm {
        match p {
          Expr::Integer(n) if *n >= 1 => s.push(*n as usize),
          _ => return TensorTransposeResult::SymmPerm,
        }
      }
      s
    }
    // Default permutation swaps the first two levels: {2, 1}.
    None => vec![2, 1],
  };

  // The given permutation must be a bijection of {1, ..., k} where k is its
  // length; otherwise it is not a valid permutation (symmperm).
  let k = partial.len();
  let mut seen = vec![false; k];
  for &s in &partial {
    if s < 1 || s > k || seen[s - 1] {
      return TensorTransposeResult::SymmPerm;
    }
    seen[s - 1] = true;
  }

  // A valid permutation that touches more slots than the tensor has (ttrank).
  if k > rank {
    return TensorTransposeResult::RankError { rank };
  }

  // Extend the partial permutation to the full rank: trailing slots are fixed.
  let sigma: Vec<usize> =
    partial.iter().copied().chain((k + 1)..=rank).collect();

  // Identity permutation → return the tensor unchanged.
  if sigma.iter().enumerate().all(|(idx, &n)| idx + 1 == n) {
    return TensorTransposeResult::Ok(list.clone());
  }

  // Result axis t corresponds to list axis sigma^-1(t).
  let mut inv = vec![0usize; rank];
  for (idx, &n) in sigma.iter().enumerate() {
    inv[n - 1] = idx + 1;
  }
  let result_shape: Vec<usize> = (0..rank).map(|t| dims[inv[t] - 1]).collect();

  TensorTransposeResult::Ok(tensor_build(&result_shape, &mut |j| {
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
        args: vec![list.clone()].into(),
      });
    }
  };

  if rows.is_empty() {
    return Ok(Expr::List(vec![].into()));
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
    result.push(Expr::List(new_row.into()));
  }

  Ok(Expr::List(result.into()))
}

/// AST-based Riffle: interleave elements with separator.
/// Riffle[{a, b, c}, x] -> {a, x, b, x, c}
/// Interleave `items` with separators placed at the given 1-based output
/// positions (sorted, distinct). A separator list cycles by insertion
/// order; any other separator expression (including `{}`) is literal.
fn riffle_build(items: &[Expr], sep: &Expr, positions: &[i128]) -> Expr {
  let sep_at = |i: usize| -> Expr {
    match sep {
      Expr::List(xs) if !xs.is_empty() => xs[i % xs.len()].clone(),
      other => other.clone(),
    }
  };
  let total = items.len() + positions.len();
  let mut result: Vec<Expr> = Vec::with_capacity(total);
  let mut item_idx = 0usize;
  let mut sep_idx = 0usize;
  for p in 1..=(total as i128) {
    if sep_idx < positions.len() && positions[sep_idx] == p {
      result.push(sep_at(sep_idx));
      sep_idx += 1;
    } else {
      result.push(items[item_idx].clone());
      item_idx += 1;
    }
  }
  Expr::List(result.into())
}

/// Unified Riffle covering the full argument space:
/// - `Riffle[list, x]` — x between elements; a separator list cycles
///   (equal length interleaves fully, with a trailing separator)
/// - `Riffle[list, x, n]` — x at every nth output position
/// - `Riffle[list, x, {imin, imax, di}]` — x at imin, imin+di, … , imax;
///   negative positions anchor to the end of the *output* list
///
/// Invalid input emits ::listrp / ::rspec / ::sepos / ::npos / ::inclen
/// like wolframscript and returns the call unevaluated.
pub fn riffle_unified_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Expr::FunctionCall {
    name: "Riffle".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  let items: &[Expr] = match &args[0] {
    Expr::List(items) => items.as_slice(),
    _ => {
      crate::emit_message(&format!(
        "Riffle::listrp: List, SparseArray object, or structured array expected at position 1 in {}.",
        show(&original())
      ));
      return Ok(original());
    }
  };
  let sep = &args[1];
  let len = items.len() as i128;

  if args.len() == 2 {
    if items.is_empty() {
      return Ok(Expr::List(vec![].into()));
    }
    // An equal-length separator list interleaves fully, separator last.
    let full = matches!(sep, Expr::List(xs) if !xs.is_empty() && xs.len() == items.len());
    let count = if full { len } else { len - 1 };
    let positions: Vec<i128> = (1..=count).map(|i| 2 * i).collect();
    return Ok(riffle_build(items, sep, &positions));
  }

  let inclen = || {
    crate::emit_message(&format!(
      "Riffle::inclen: The start and end positions and the spacing between riffled elements given in {} cannot be satisfied for the input list of length {}.",
      show(&args[2]),
      len
    ));
    Ok(original())
  };
  let rspec = || {
    crate::emit_message(&format!(
      "Riffle::rspec: The third argument {} should be a positive integer or a list with three integers.",
      show(&args[2])
    ));
    Ok(original())
  };

  match &args[2] {
    // Scalar n: x at every nth output position. The separator count k is
    // the smallest k >= 1 with floor((len + k)/n) == k; a riffle that
    // cannot place any separator consistently emits ::inclen.
    Expr::Integer(n) if *n >= 1 => {
      let n = *n;
      if n == 1 {
        return inclen();
      }
      if len == 0 {
        return Ok(Expr::List(vec![].into()));
      }
      if len == 1 {
        return Ok(args[0].clone());
      }
      let k = match (1..=len + 2).find(|k| (len + k) / n == *k) {
        Some(k) => k,
        None => return inclen(),
      };
      let positions: Vec<i128> = (1..=k).map(|i| i * n).collect();
      Ok(riffle_build(items, sep, &positions))
    }
    Expr::List(spec_items) if spec_items.len() == 3 => {
      let ints: Option<Vec<i128>> = spec_items
        .iter()
        .map(|e| match e {
          Expr::Integer(n) => Some(*n),
          _ => None,
        })
        .collect();
      let Some(ints) = ints else {
        return rspec();
      };
      let (imin, imax, di) = (ints[0], ints[1], ints[2]);
      if imin == 0 || imax == 0 {
        crate::emit_message(&format!(
          "Riffle::sepos: The start and end positions in {} should be nonzero machine-sized integers.",
          show(&args[2])
        ));
        return Ok(original());
      }
      if di == 0 {
        crate::emit_message(&format!(
          "Riffle::npos: The spacing between riffled elements given in {} should be a positive machine-sized integer.",
          show(&args[2])
        ));
        return Ok(original());
      }

      // Negative endpoints anchor to the output list, whose length depends
      // on the separator count k. A count is self-consistent when the
      // resolved range has exactly k positions, each leaving room for the
      // elements before it (p_i <= len + i). Among consistent counts the
      // largest wins, except when every count works (the range tracks the
      // output end indefinitely) — then the smallest does.
      let resolve = |e: i128, t: i128| if e > 0 { e } else { t + 1 + e };
      let consistent = |k: i128| -> Option<Vec<i128>> {
        let t = len + k;
        let lo = resolve(imin, t);
        let hi = resolve(imax, t);
        let mut positions: Vec<i128> = Vec::new();
        let mut p = lo;
        while (di > 0 && p <= hi) || (di < 0 && p >= hi) {
          positions.push(p);
          p += di;
        }
        positions.sort_unstable();
        if positions.len() as i128 != k {
          return None;
        }
        let fits = positions
          .iter()
          .enumerate()
          .all(|(i, p)| *p >= 1 && *p <= len + i as i128 + 1);
        if fits { Some(positions) } else { None }
      };
      let k_max = len + imin.abs() + imax.abs() + di.abs() + 16;
      let candidates: Vec<i128> =
        (0..=k_max).filter(|k| consistent(*k).is_some()).collect();
      let chosen = if consistent(k_max + 1).is_some() {
        candidates.first().copied()
      } else {
        candidates.last().copied()
      };
      match chosen {
        Some(k) => {
          let positions = consistent(k).unwrap();
          Ok(riffle_build(items, sep, &positions))
        }
        None => inclen(),
      }
    }
    _ => rspec(),
  }
}

/// AST-based RotateLeft: rotate list left by n positions.
pub fn rotate_left_ast(list: &Expr, n: i128) -> Result<Expr, InterpreterError> {
  let (items, head_name): (&[Expr], Option<&str>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RotateLeft".to_string(),
        args: vec![list.clone(), Expr::Integer(n)].into(),
      });
    }
  };

  if items.is_empty() {
    return match head_name {
      Some(name) => Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: vec![].into(),
      }),
      None => Ok(Expr::List(vec![].into())),
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
      args: result.into(),
    }),
    None => Ok(Expr::List(result.into())),
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
        args: vec![list.clone(), Expr::List(shifts.to_vec().into())].into(),
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
        Ok(Expr::List(new_items.into()))
      }
      _ => Ok(rotated),
    }
  } else {
    Ok(rotated)
  }
}

/// Number of levels at which *every* node of `expr` is a list (the depth a
/// list padding-specification may address). An empty list places no bound
/// on the depth below it.
fn pad_container_depth(expr: &Expr) -> usize {
  fn list_depth(e: &Expr) -> usize {
    match e {
      Expr::List(items) => {
        if items.is_empty() {
          usize::MAX
        } else {
          match items.iter().map(list_depth).min().unwrap_or(0) {
            usize::MAX => usize::MAX,
            d => d + 1,
          }
        }
      }
      _ => 0,
    }
  }
  match expr {
    Expr::List(_) => list_depth(expr),
    // A general head only counts for the top level; below it the spec
    // must address lists.
    Expr::FunctionCall { args, .. } => {
      if args.is_empty() {
        usize::MAX
      } else {
        match args.iter().map(list_depth).min().unwrap_or(0) {
          usize::MAX => usize::MAX,
          d => d + 1,
        }
      }
    }
    _ => 0,
  }
}

/// Rectangular dimensions used by the one-argument forms
/// `PadLeft[list]` / `PadRight[list]`: descend while every node at the
/// current level is a list, recording the maximum length per level.
fn ragged_dims(expr: &Expr) -> Vec<i128> {
  let mut dims = Vec::new();
  let mut level: Vec<&Expr> = vec![expr];
  while level.iter().all(|e| matches!(e, Expr::List(_))) {
    let mut max_len = 0usize;
    let mut next: Vec<&Expr> = Vec::new();
    for e in &level {
      if let Expr::List(items) = e {
        max_len = max_len.max(items.len());
        next.extend(items.iter());
      }
    }
    dims.push(max_len as i128);
    if next.is_empty() {
      break;
    }
    level = next;
  }
  dims
}

/// Padding element for column `c` (1-based) of a level of length `n` with
/// margin `m`. Cyclic padding lists align to the content edge: PadLeft
/// counts backwards from the content's last column, PadRight forwards
/// from the content's first column.
fn pad_value(pad: &Expr, c: i128, n: i128, m: i128, side_left: bool) -> Expr {
  match pad {
    Expr::List(cycle) if !cycle.is_empty() => {
      let len = cycle.len() as i128;
      let idx = if side_left {
        (c - (n - m) - 1).rem_euclid(len)
      } else {
        (c - m - 1).rem_euclid(len)
      };
      cycle[idx as usize].clone()
    }
    _ => pad.clone(),
  }
}

/// Pad one level: place `items` so they end at column `n - m` (PadLeft)
/// or start at column `m + 1` (PadRight), fill the rest with padding, and
/// recurse with the remaining spec. A negative length flips the side at
/// that level only.
fn pad_items(
  items: &[Expr],
  spec: &[i128],
  margins: &[i128],
  pad: &Expr,
  is_left: bool,
) -> Result<Vec<Expr>, InterpreterError> {
  if spec.is_empty() {
    return Ok(items.to_vec());
  }
  let n_signed = spec[0];
  let side_left = is_left != (n_signed < 0);
  let n = n_signed.saturating_abs();
  let m = margins.first().copied().unwrap_or(0);
  let rest_spec = &spec[1..];
  let rest_margins = if margins.is_empty() {
    margins
  } else {
    &margins[1..]
  };

  let len = items.len() as i128;
  let (start, end) = if side_left {
    (n - m - len + 1, n - m)
  } else {
    (m + 1, m + len)
  };

  let mut out: Vec<Expr> = Vec::with_capacity(n.max(0) as usize);
  for c in 1..=n {
    if c >= start && c <= end {
      let item = &items[(c - start) as usize];
      if rest_spec.is_empty() {
        out.push(item.clone());
      } else {
        let sub_items: &[Expr] = match item {
          Expr::List(sub) => sub.as_slice(),
          _ => &[],
        };
        out.push(Expr::List(
          pad_items(sub_items, rest_spec, rest_margins, pad, is_left)?.into(),
        ));
      }
    } else if rest_spec.is_empty() {
      out.push(pad_value(pad, c, n, m, side_left));
    } else {
      out.push(Expr::List(
        pad_items(&[], rest_spec, rest_margins, pad, is_left)?.into(),
      ));
    }
  }
  Ok(out)
}

fn pad_level(
  expr: &Expr,
  spec: &[i128],
  margins: &[i128],
  pad: &Expr,
  is_left: bool,
) -> Result<Expr, InterpreterError> {
  let (items, head): (&[Expr], Option<&str>) = match expr {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => (&[], None),
  };
  let out = pad_items(items, spec, margins, pad, is_left)?;
  match head {
    Some(h) => crate::evaluator::evaluate_function_call_ast(h, &out),
    None => Ok(Expr::List(out.into())),
  }
}

/// Unified PadLeft/PadRight engine covering the full argument space:
/// - `Pad[list]` — pad a ragged array of lists with zeros to make it full
/// - `Pad[list, n]` — scalar length; negative n pads the opposite side
/// - `Pad[list, {n1, n2, …}]` — per-level lengths
/// - `Pad[list, spec, x]` — scalar padding
/// - `Pad[list, spec, {x1, x2, …}]` — cyclic padding; `{}` returns the
///   list unchanged
/// - `Pad[list, spec, pad, m]` — margin; a scalar broadcasts to all levels
///
/// Invalid input emits `::normal` / `::ilsm` / `::level` like
/// wolframscript and returns the call unevaluated.
pub fn pad_ast(fname: &str, args: &[Expr]) -> Result<Expr, InterpreterError> {
  let is_left = fname == "PadLeft";
  let original = || Expr::FunctionCall {
    name: fname.to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);
  let ilsm = |pos: usize| {
    crate::emit_message(&format!(
      "{}::ilsm: List of machine-sized integers expected at position {} in {}.",
      fname,
      pos,
      show(&original())
    ));
  };
  let strict_int = |e: &Expr| -> Option<i128> {
    match e {
      Expr::Integer(n) => Some(*n),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i128()
      }
      _ => None,
    }
  };

  let subject = &args[0];
  if !matches!(subject, Expr::List(_) | Expr::FunctionCall { .. }) {
    crate::emit_message(&format!(
      "{}::normal: Nonatomic expression expected at position 1 in {}.",
      fname,
      show(&original())
    ));
    return Ok(original());
  }

  // One-argument ragged-fill form; non-List heads stay unevaluated.
  if args.len() == 1 {
    if !matches!(subject, Expr::List(_)) {
      return Ok(original());
    }
    let dims = ragged_dims(subject);
    return pad_level(subject, &dims, &[], &Expr::Integer(0), is_left);
  }

  // Length specification: a machine integer or a list of them.
  let (spec, spec_is_list): (Vec<i128>, bool) = match &args[1] {
    Expr::List(items) => {
      match items.iter().map(strict_int).collect::<Option<Vec<i128>>>() {
        Some(ns) => (ns, true),
        None => {
          ilsm(2);
          return Ok(original());
        }
      }
    }
    other => match strict_int(other) {
      Some(n) => (vec![n], false),
      None => {
        ilsm(2);
        return Ok(original());
      }
    },
  };

  let pad = args.get(2).cloned().unwrap_or(Expr::Integer(0));

  // Margin: a scalar broadcasts to every level of the spec.
  let margins: Vec<i128> = if args.len() >= 4 {
    match &args[3] {
      Expr::List(items) => {
        match items.iter().map(strict_int).collect::<Option<Vec<i128>>>() {
          Some(ms) => ms,
          None => {
            ilsm(4);
            return Ok(original());
          }
        }
      }
      other => match strict_int(other) {
        Some(m) => vec![m; spec.len()],
        None => {
          ilsm(4);
          return Ok(original());
        }
      },
    }
  } else {
    Vec::new()
  };

  // An empty padding list leaves the input untouched (no truncation).
  if matches!(&pad, Expr::List(items) if items.is_empty()) {
    return Ok(subject.clone());
  }

  // A list spec may not address more levels than the subject has.
  if spec_is_list {
    let depth = pad_container_depth(subject);
    if spec.len() > depth {
      // wolframscript never pluralizes the subject's level count.
      crate::emit_message(&format!(
        "{}::level: The padding specification {} involves {} levels; the list {} has only {} level.",
        fname,
        show(&args[1]),
        spec.len(),
        show(subject),
        depth
      ));
      return Ok(original());
    }
  }

  pad_level(subject, &spec, &margins, &pad, is_left)
}

/// AST-based Join: join multiple lists.
pub fn join_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Optional trailing level specification: `Join[l1, …, lk, n]` joins the
  // lists at level n (default 1). A trailing `1` is the common case (e.g.
  // `{y, m} ~Join~ 1`) and is equivalent to an ordinary join of the
  // preceding lists; drop it and continue. (Levels >= 2 are left to the
  // general path below, matching wolframscript leaving them unevaluated for
  // the shapes Woxi doesn't thread.)
  if lists.len() >= 2
    && let Expr::Integer(1) = &lists[lists.len() - 1]
  {
    return join_ast(&lists[..lists.len() - 1]);
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
        args: lists.to_vec().into(),
      });
    }
  };

  let head_label_first = head.unwrap_or("List");

  let mut result = Vec::new();
  for (i, list) in lists.iter().enumerate() {
    match (list, head) {
      (Expr::List(items), None) => result.extend(items.iter().cloned()),
      (Expr::FunctionCall { name, args }, Some(h)) if name == h => {
        result.extend(args.iter().cloned());
      }
      _ => {
        // Emit Join::heads error if the mismatched argument has an
        // explicit head (List or FunctionCall). Bare symbols/atoms don't
        // trigger the error in wolframscript.
        let other_head: Option<&str> = match list {
          Expr::List(_) => Some("List"),
          Expr::FunctionCall { name, .. } => Some(name.as_str()),
          _ => None,
        };
        if let Some(other) = other_head {
          crate::emit_message(&format!(
            "Join::heads: Heads {} and {} at positions 1 and {} are expected to be the same.",
            head_label_first,
            other,
            i + 1,
          ));
        }
        return Ok(Expr::FunctionCall {
          name: "Join".to_string(),
          args: lists.to_vec().into(),
        });
      }
    }
  }

  match head {
    None => Ok(Expr::List(result.into())),
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
  let mut all_items: Vec<&crate::ExprList> = Vec::new();
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

  Ok(Expr::List(result.into()))
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
          args: vec![list.clone(), elem.clone()].into(),
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
      args: vec![list.clone(), elem.clone()].into(),
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
          args: vec![list.clone(), elem.clone()].into(),
        })
      }
    }
    // O(log N) per call once the list has upgraded to its tree-backed
    // representation (after the first push_front). Recursive prepend
    // chains like `parseLevel` in build_summary.wls run in O(N log N)
    // rather than O(N²).
    Expr::List(items) => {
      let mut new_items = items.clone();
      new_items.push_front(elem.clone());
      Ok(Expr::List(new_items))
    }
    Expr::FunctionCall { name, args } => {
      let mut new_args = args.clone();
      new_args.push_front(elem.clone());
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Prepend".to_string(),
      args: vec![list.clone(), elem.clone()].into(),
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
  Ok(Expr::List(result.into()))
}
