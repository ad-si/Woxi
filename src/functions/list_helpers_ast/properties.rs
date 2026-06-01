#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based ArrayDepth: compute depth of nested lists.
/// Returns the depth to which the expression forms a rectangular array.
/// At each level, all elements must be lists of the same length.
pub fn array_depth_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  fn compute_depth(expr: &Expr) -> i128 {
    match expr {
      Expr::List(items) => {
        // Start with depth 1 (the outermost list counts)
        let mut depth: i128 = 1;
        // Collect the current "frontier" of elements to check
        let mut current_level: Vec<&Expr> = items.iter().collect();
        loop {
          if current_level.is_empty() {
            break;
          }
          // Check if all elements at this level are lists
          let first_len = match current_level[0] {
            Expr::List(sub) => Some(sub.len()),
            _ => break, // Not all lists — stop
          };
          if let Some(len) = first_len {
            // All must be lists with the same length
            let all_same = current_level
              .iter()
              .all(|item| matches!(item, Expr::List(sub) if sub.len() == len));
            if !all_same {
              break;
            }
            // Move to the next level
            depth += 1;
            let next_level: Vec<&Expr> = current_level
              .iter()
              .flat_map(|item| {
                if let Expr::List(sub) = item {
                  sub.iter().collect::<Vec<_>>()
                } else {
                  vec![]
                }
              })
              .collect();
            current_level = next_level;
          } else {
            break;
          }
        }
        depth
      }
      _ => 0,
    }
  }

  Ok(Expr::Integer(compute_depth(list)))
}

/// ArrayComponents[array] / [array, level] / [array, level, rules]
///
/// Replaces every distinct element of `array` with an integer index. Identical
/// elements (compared structurally) get the same index. Indices are assigned by
/// order of first appearance, using the smallest positive integer not already
/// claimed by an explicit rule. The integer `0` maps to `0` by default (it acts
/// as the background/zero element) unless overridden by a rule.
///
/// `level` (a positive integer, default Infinity) controls how deep the labeling
/// descends. Elements shallower than `level` (atoms reached before the target
/// level) are labeled as whole units; the structure above `level` is preserved.
///
/// `rules` is a rule or list of rules mapping original elements to explicit
/// labels. Elements not covered by a rule are auto-numbered with the smallest
/// positive integers not used by any rule.
pub fn array_components_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Return the expression unevaluated (matching wolframscript, which emits a
  // message and leaves ArrayComponents[...] in place).
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ArrayComponents".to_string(),
      args: args.to_vec().into(),
    })
  };

  // Argument 1 must be a list.
  if args.is_empty() || !matches!(&args[0], Expr::List(_)) {
    return unevaluated();
  }

  // Parse the level argument (default: effectively infinite).
  let target_level: i128 = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(n) if n >= 1 => n,
      _ => return unevaluated(),
    }
  } else {
    i128::MAX
  };

  // Parse the rules argument: map from structural-key -> explicit label Expr.
  // Preserve the set of labels already claimed so auto-numbering can avoid them.
  let mut rule_map: std::collections::HashMap<String, Expr> =
    std::collections::HashMap::new();
  // Default rule: integer 0 -> 0.
  rule_map.insert(
    crate::syntax::expr_to_string(&Expr::Integer(0)),
    Expr::Integer(0),
  );
  if args.len() >= 3 {
    let rule_list: Vec<Expr> = match &args[2] {
      Expr::List(rs) => rs.to_vec(),
      single => vec![single.clone()],
    };
    for r in &rule_list {
      if let Expr::Rule {
        pattern,
        replacement,
      } = r
      {
        rule_map.insert(
          crate::syntax::expr_to_string(pattern.as_ref()),
          replacement.as_ref().clone(),
        );
      }
    }
  }

  // Collect integer labels that explicit rules already claim, so that
  // auto-numbering skips them.
  let mut claimed: std::collections::HashSet<i128> =
    std::collections::HashSet::new();
  for v in rule_map.values() {
    if let Some(n) = expr_to_i128(v) {
      claimed.insert(n);
    }
  }

  // State shared across the recursive walk.
  let mut label_for_key: std::collections::HashMap<String, Expr> = rule_map;
  let mut next_index: i128 = 1;

  // Determine the next auto-assigned label: the smallest positive integer not
  // already claimed by a rule and not yet used.
  fn alloc_label(
    next_index: &mut i128,
    claimed: &std::collections::HashSet<i128>,
  ) -> Expr {
    while claimed.contains(next_index) {
      *next_index += 1;
    }
    let v = *next_index;
    *next_index += 1;
    Expr::Integer(v)
  }

  // Label a "unit" expression, reusing an existing label for identical units.
  fn label_unit(
    expr: &Expr,
    label_for_key: &mut std::collections::HashMap<String, Expr>,
    next_index: &mut i128,
    claimed: &std::collections::HashSet<i128>,
  ) -> Expr {
    let key = crate::syntax::expr_to_string(expr);
    if let Some(existing) = label_for_key.get(&key) {
      return existing.clone();
    }
    let label = alloc_label(next_index, claimed);
    label_for_key.insert(key, label.clone());
    label
  }

  // Recursively walk to `target_level`. `cur_level` is the level of `expr`
  // itself (top-level list elements are at level 1, so the outer list is at
  // level 0).
  fn walk(
    expr: &Expr,
    cur_level: i128,
    target_level: i128,
    label_for_key: &mut std::collections::HashMap<String, Expr>,
    next_index: &mut i128,
    claimed: &std::collections::HashSet<i128>,
  ) -> Expr {
    match expr {
      Expr::List(items) if cur_level < target_level => Expr::List(
        items
          .iter()
          .map(|item| {
            walk(
              item,
              cur_level + 1,
              target_level,
              label_for_key,
              next_index,
              claimed,
            )
          })
          .collect(),
      ),
      // Either an atom, or a node at the target level: label as a unit.
      _ => label_unit(expr, label_for_key, next_index, claimed),
    }
  }

  let result = walk(
    &args[0],
    0,
    target_level,
    &mut label_for_key,
    &mut next_index,
    &claimed,
  );
  Ok(result)
}

/// TensorRank[expr] - rank of a tensor (same as ArrayDepth for concrete lists,
/// but stays unevaluated for non-list expressions like symbols).
pub fn tensor_rank_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  // TensorRank[TensorContract[T, {{s, t}, …}]] = TensorRank[T] - 2 * k, where
  // k is the number of contraction pairs.
  if let Expr::FunctionCall { name, args } = expr
    && name == "TensorContract"
    && args.len() == 2
    && let Expr::List(pairs) = &args[1]
    && !pairs.is_empty()
    && pairs
      .iter()
      .all(|p| matches!(p, Expr::List(inner) if inner.len() == 2))
  {
    let inner_rank = Expr::FunctionCall {
      name: "TensorRank".to_string(),
      args: vec![args[0].clone()].into(),
    };
    let reduction = Expr::Integer(2 * pairs.len() as i128);
    let result = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), reduction].into(),
        },
        inner_rank,
      ]
      .into(),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  match expr {
    Expr::List(_) => array_depth_ast(expr),
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(_, _) => Ok(Expr::Integer(0)),
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      Ok(Expr::Integer(0))
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      Ok(Expr::Integer(0))
    }
    _ => Ok(Expr::FunctionCall {
      name: "TensorRank".to_string(),
      args: vec![expr.clone()].into(),
    }),
  }
}

/// TensorSymmetry[expr] - symmetry classification of a rank-2 tensor (matrix).
/// Returns:
///   ZeroSymmetric[{}]      — all entries are zero
///   Symmetric[{1, 2}]      — M[i,j] = M[j,i] for all i, j (and not all zero)
///   Antisymmetric[{1, 2}]  — M[i,j] = -M[j,i] for all i, j (and not all zero)
///   {}                     — generic square matrix with no special symmetry
///   TensorSymmetry[...]    — for non-matrix or non-square input, stay symbolic
/// TensorContract[T, {{i1, j1}, {i2, j2}, ...}] contracts pairs of indices
/// of the tensor T. Each pair {i, j} of 1-indexed slots is summed over the
/// matching dimension, leaving the free indices in their original order.
pub fn tensor_contract_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "TensorContract".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated());
  }
  let Expr::List(pairs_list) = &args[1] else {
    return Ok(unevaluated());
  };

  // Extract the tensor's shape and ensure it's a fully rectangular list.
  let Some(shape) = tensor_shape(&args[0]) else {
    return Ok(unevaluated());
  };
  let rank = shape.len();
  if rank == 0 {
    return Ok(unevaluated());
  }

  // Parse contraction pairs. Each must be a 2-element list of 1-indexed ints.
  let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(pairs_list.len());
  let mut used = vec![false; rank];
  for p in pairs_list.iter() {
    let Expr::List(ij) = p else {
      return Ok(unevaluated());
    };
    if ij.len() != 2 {
      return Ok(unevaluated());
    }
    let (Some(i), Some(j)) = (expr_as_pos_int(&ij[0]), expr_as_pos_int(&ij[1]))
    else {
      return Ok(unevaluated());
    };
    if i == j || i == 0 || j == 0 || i > rank || j > rank {
      return Ok(unevaluated());
    }
    if used[i - 1] || used[j - 1] {
      return Ok(unevaluated());
    }
    used[i - 1] = true;
    used[j - 1] = true;
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    if shape[a - 1] != shape[b - 1] {
      return Ok(unevaluated());
    }
    pairs.push((a, b));
  }

  // Index slots are 1-indexed; convert to 0-indexed.
  let pair_slots: Vec<(usize, usize)> =
    pairs.iter().map(|(a, b)| (a - 1, b - 1)).collect();
  let free_slots: Vec<usize> = (0..rank).filter(|i| !used[*i]).collect();

  // Build the result by iterating over all free-index combinations.
  let free_dims: Vec<usize> = free_slots.iter().map(|&s| shape[s]).collect();
  let contracted_dims: Vec<usize> =
    pair_slots.iter().map(|&(a, _)| shape[a]).collect();

  let result = build_contracted(
    &args[0],
    &shape,
    &pair_slots,
    &free_slots,
    &free_dims,
    &contracted_dims,
    rank,
  )?;
  Ok(result)
}

/// Build the result of contracting a tensor along the given paired slots.
fn build_contracted(
  tensor: &Expr,
  shape: &[usize],
  pair_slots: &[(usize, usize)],
  free_slots: &[usize],
  free_dims: &[usize],
  contracted_dims: &[usize],
  rank: usize,
) -> Result<Expr, InterpreterError> {
  // Helper: increment a multi-index `idx` in row-major order against `dims`.
  // Returns false when we've exhausted all combinations.
  fn advance(idx: &mut [usize], dims: &[usize]) -> bool {
    if dims.is_empty() {
      return false;
    }
    let mut k = dims.len();
    while k > 0 {
      k -= 1;
      idx[k] += 1;
      if idx[k] < dims[k] {
        return true;
      }
      idx[k] = 0;
    }
    false
  }

  // Sum over contracted indices for a fixed assignment of free indices.
  let sum_for_free = |free_idx: &[usize]| -> Result<Expr, InterpreterError> {
    let mut terms: Vec<Expr> = Vec::new();
    let mut c_idx = vec![0usize; contracted_dims.len()];
    loop {
      let mut full = vec![0usize; rank];
      for (k, &s) in free_slots.iter().enumerate() {
        full[s] = free_idx[k];
      }
      for (k, &(a, b)) in pair_slots.iter().enumerate() {
        full[a] = c_idx[k];
        full[b] = c_idx[k];
      }
      let elem = index_tensor(tensor, &full, shape)?;
      terms.push(elem);
      if contracted_dims.is_empty() {
        break;
      }
      if !advance(&mut c_idx, contracted_dims) {
        break;
      }
    }
    if terms.len() == 1 {
      return Ok(terms.remove(0));
    }
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    })
  };

  if free_slots.is_empty() {
    // Fully contracted → scalar.
    return sum_for_free(&[]);
  }

  // Otherwise, build nested lists matching the shape of free_dims.
  fn build_nested<F>(
    free_dims: &[usize],
    free_idx: &mut Vec<usize>,
    depth: usize,
    sum: &F,
  ) -> Result<Expr, InterpreterError>
  where
    F: Fn(&[usize]) -> Result<Expr, InterpreterError>,
  {
    if depth == free_dims.len() {
      return sum(free_idx);
    }
    let mut items = Vec::with_capacity(free_dims[depth]);
    for k in 0..free_dims[depth] {
      free_idx.push(k);
      items.push(build_nested(free_dims, free_idx, depth + 1, sum)?);
      free_idx.pop();
    }
    Ok(Expr::List(items.into()))
  }
  let mut free_idx: Vec<usize> = Vec::with_capacity(free_dims.len());
  build_nested(free_dims, &mut free_idx, 0, &sum_for_free)
}

/// Look up `tensor[idx[0], idx[1], ...]` (0-indexed) on a nested list whose
/// shape is `shape`.
fn index_tensor(
  tensor: &Expr,
  idx: &[usize],
  shape: &[usize],
) -> Result<Expr, InterpreterError> {
  let mut current = tensor.clone();
  for (depth, &k) in idx.iter().enumerate() {
    let Expr::List(items) = &current else {
      return Err(InterpreterError::EvaluationError(
        "TensorContract: irregular tensor".into(),
      ));
    };
    if items.len() != shape[depth] {
      return Err(InterpreterError::EvaluationError(
        "TensorContract: irregular tensor".into(),
      ));
    }
    if k >= items.len() {
      return Err(InterpreterError::EvaluationError(
        "TensorContract: index out of range".into(),
      ));
    }
    current = items[k].clone();
  }
  Ok(current)
}

/// Returns the shape of a fully rectangular nested-list tensor, or None.
/// A scalar (non-List) has shape `[]`.
fn tensor_shape(expr: &Expr) -> Option<Vec<usize>> {
  let Expr::List(items) = expr else {
    return Some(vec![]);
  };
  if items.is_empty() {
    return Some(vec![0]);
  }
  let mut dims = vec![items.len()];
  let inner_shape = tensor_shape(&items[0])?;
  for sub in items.iter() {
    let s = tensor_shape(sub)?;
    if s != inner_shape {
      return None;
    }
  }
  dims.extend(inner_shape);
  Some(dims)
}

fn expr_as_pos_int(e: &Expr) -> Option<usize> {
  if let Expr::Integer(n) = e
    && *n > 0
  {
    return Some(*n as usize);
  }
  None
}

pub fn tensor_symmetry_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "TensorSymmetry".to_string(),
    args: vec![expr.clone()].into(),
  };
  let Expr::List(rows) = expr else {
    return Ok(unevaluated());
  };
  let n = rows.len();
  if n == 0 {
    return Ok(unevaluated());
  }
  // Require a square matrix: every row is a list of length n.
  let mut matrix: Vec<Vec<&Expr>> = Vec::with_capacity(n);
  for row in rows.iter() {
    let Expr::List(cells) = row else {
      return Ok(unevaluated());
    };
    if cells.len() != n {
      return Ok(unevaluated());
    }
    matrix.push(cells.iter().collect());
  }

  let is_zero = |e: &Expr| {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(f) if *f == 0.0)
  };
  let mut all_zero = true;
  for row in &matrix {
    for cell in row {
      if !is_zero(cell) {
        all_zero = false;
        break;
      }
    }
    if !all_zero {
      break;
    }
  }
  if all_zero {
    return Ok(Expr::FunctionCall {
      name: "ZeroSymmetric".to_string(),
      args: vec![Expr::List(Vec::new().into())].into(),
    });
  }

  // Equality check on Expr — use canonical evaluation so e.g. (-5) == (-5).
  let eq = |a: &Expr, b: &Expr| -> bool {
    let cmp = Expr::FunctionCall {
      name: "Equal".to_string(),
      args: vec![a.clone(), b.clone()].into(),
    };
    matches!(
      crate::evaluator::evaluate_expr_to_expr(&cmp),
      Ok(Expr::Identifier(ref s)) if s == "True"
    )
  };
  let neg = |e: &Expr| -> Expr {
    let e = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), e.clone()].into(),
    };
    crate::evaluator::evaluate_expr_to_expr(&e).unwrap_or_else(|_| {
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), e.clone()].into(),
      }
    })
  };

  let mut is_symmetric = true;
  let mut is_antisymmetric = true;
  for i in 0..n {
    for j in 0..n {
      let mij = matrix[i][j];
      let mji = matrix[j][i];
      if !eq(mij, mji) {
        is_symmetric = false;
      }
      if i == j {
        if !is_zero(mij) {
          is_antisymmetric = false;
        }
      } else if !eq(mij, &neg(mji)) {
        is_antisymmetric = false;
      }
      if !is_symmetric && !is_antisymmetric {
        break;
      }
    }
    if !is_symmetric && !is_antisymmetric {
      break;
    }
  }

  let slot_list = Expr::List(vec![Expr::Integer(1), Expr::Integer(2)].into());
  if is_symmetric {
    return Ok(Expr::FunctionCall {
      name: "Symmetric".to_string(),
      args: vec![slot_list].into(),
    });
  }
  if is_antisymmetric {
    return Ok(Expr::FunctionCall {
      name: "Antisymmetric".to_string(),
      args: vec![slot_list].into(),
    });
  }
  Ok(Expr::List(Vec::new().into()))
}

/// ArrayQ[expr] - True if expr is a full array (rectangular at all levels).
pub fn array_q_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  fn is_full_array(expr: &Expr) -> bool {
    match expr {
      Expr::List(items) => {
        if items.is_empty() {
          return true;
        }
        // All items must have the same structure
        let depths: Vec<Vec<usize>> =
          items.iter().map(get_dimensions).collect();
        // All items must have the same dimensions
        depths.iter().all(|d| d == &depths[0])
      }
      _ => false,
    }
  }
  Ok(if is_full_array(expr) {
    Expr::Identifier("True".to_string())
  } else {
    Expr::Identifier("False".to_string())
  })
}

/// Get the dimensions of a rectangular array
fn get_dimensions(expr: &Expr) -> Vec<usize> {
  match expr {
    Expr::List(items) => {
      let mut dims = vec![items.len()];
      if !items.is_empty() {
        let sub_dims: Vec<Vec<usize>> =
          items.iter().map(get_dimensions).collect();
        // Check all sublists have the same dimensions
        if sub_dims.iter().all(|d| d == &sub_dims[0]) && !sub_dims[0].is_empty()
        {
          dims.extend_from_slice(&sub_dims[0]);
        }
      }
      dims
    }
    _ => vec![],
  }
}

/// VectorQ[expr] - True if expr is a list of non-list elements.
pub fn vector_q_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(items) => {
      Ok(if items.iter().all(|i| !matches!(i, Expr::List(_))) {
        Expr::Identifier("True".to_string())
      } else {
        Expr::Identifier("False".to_string())
      })
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

/// MatrixQ[expr] - True if expr is a list of equal-length lists (2D rectangular array).
pub fn matrix_q_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(rows) => {
      if rows.is_empty() {
        return Ok(Expr::Identifier("True".to_string()));
      }
      // Each row must be a list
      let mut ncols = None;
      for row in rows {
        match row {
          Expr::List(cols) => {
            if let Some(expected) = ncols {
              if cols.len() != expected {
                return Ok(Expr::Identifier("False".to_string()));
              }
            } else {
              ncols = Some(cols.len());
            }
            // Each element must not be a list (must be a scalar)
            if cols.iter().any(|c| matches!(c, Expr::List(_))) {
              return Ok(Expr::Identifier("False".to_string()));
            }
          }
          _ => return Ok(Expr::Identifier("False".to_string())),
        }
      }
      Ok(Expr::Identifier("True".to_string()))
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

/// Apply `test` to every leaf at depth `depth` of `expr` and return True only
/// if every test call returns True. Used by ArrayQ[expr, n, test].
pub fn all_leaves_pass_test(expr: &Expr, depth: usize, test: &Expr) -> bool {
  if depth == 0 {
    let result =
      crate::functions::list_helpers_ast::utilities::apply_func_ast(test, expr);
    return matches!(result, Ok(Expr::Identifier(ref s)) if s == "True");
  }
  match expr {
    Expr::List(items) => items
      .iter()
      .all(|child| all_leaves_pass_test(child, depth - 1, test)),
    _ => false,
  }
}

/// MatrixQ[m, test] - True if m is a matrix and test[elem] is True for all elements.
pub fn matrix_q_with_test_ast(
  expr: &Expr,
  test: &Expr,
) -> Result<Expr, InterpreterError> {
  // First check if it's a valid matrix structure
  let rows = match expr {
    Expr::List(rows) if !rows.is_empty() => rows,
    Expr::List(_) => return Ok(Expr::Identifier("True".to_string())),
    _ => return Ok(Expr::Identifier("False".to_string())),
  };

  let mut ncols = None;
  for row in rows {
    match row {
      Expr::List(cols) => {
        if let Some(expected) = ncols {
          if cols.len() != expected {
            return Ok(Expr::Identifier("False".to_string()));
          }
        } else {
          ncols = Some(cols.len());
        }
        // Check each element with the test function
        for elem in cols {
          if matches!(elem, Expr::List(_)) {
            return Ok(Expr::Identifier("False".to_string()));
          }
          let test_call = Expr::FunctionCall {
            name: if let Expr::Identifier(n) = test {
              n.clone()
            } else {
              // For non-identifier tests, build an application
              "__test__".to_string()
            },
            args: vec![elem.clone()].into(),
          };
          let result = if let Expr::Identifier(_) = test {
            crate::evaluator::evaluate_expr_to_expr(&test_call)?
          } else {
            // Apply test as a function
            let apply = Expr::FunctionCall {
              name: "Apply".to_string(),
              args: vec![test.clone(), Expr::List(vec![elem.clone()].into())]
                .into(),
            };
            crate::evaluator::evaluate_expr_to_expr(&apply)?
          };
          match &result {
            Expr::Identifier(s) if s == "True" => {}
            _ => return Ok(Expr::Identifier("False".to_string())),
          }
        }
      }
      _ => return Ok(Expr::Identifier("False".to_string())),
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// SymmetricMatrixQ[m] - True if m is a symmetric square matrix (m[i][j] == m[j][i]).
pub fn symmetric_matrix_q_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(rows) => {
      let n = rows.len();
      if n == 0 {
        return Ok(Expr::Identifier("False".to_string()));
      }
      // All rows must be lists of length n (square matrix)
      let mut grid: Vec<&crate::ExprList> = Vec::with_capacity(n);
      for row in rows {
        match row {
          Expr::List(cols) => {
            if cols.len() != n {
              return Ok(Expr::Identifier("False".to_string()));
            }
            grid.push(cols);
          }
          _ => return Ok(Expr::Identifier("False".to_string())),
        }
      }
      // Check symmetry: m[i][j] == m[j][i]
      for i in 0..n {
        for j in (i + 1)..n {
          let a = crate::syntax::expr_to_string(&grid[i][j]);
          let b = crate::syntax::expr_to_string(&grid[j][i]);
          if a != b {
            return Ok(Expr::Identifier("False".to_string()));
          }
        }
      }
      Ok(Expr::Identifier("True".to_string()))
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

pub fn list_depth(expr: &Expr) -> usize {
  match expr {
    Expr::List(items) => {
      if items.is_empty() {
        1
      } else {
        1 + items.iter().map(list_depth).min().unwrap_or(0)
      }
    }
    _ => 0,
  }
}

/// Dimensions[list] - Returns the dimensions of a nested list.
/// Dimensions[list, n] - Limits the analysis to at most n levels.
pub fn dimensions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Dimensions expects 1 or 2 arguments".into(),
    ));
  }

  // Parse the optional max-level argument. `None` means unlimited.
  let max_level: Option<usize> = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n >= 0 => Some(*n as usize),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Dimensions: second argument must be a non-negative integer".into(),
        ));
      }
    }
  } else {
    None
  };

  fn get_dimensions(expr: &Expr, max_level: Option<usize>) -> Vec<i128> {
    if let Some(0) = max_level {
      return vec![];
    }
    let child_max = max_level.map(|m| m - 1);
    match expr {
      Expr::List(items) => {
        let mut dims = vec![items.len() as i128];
        if !items.is_empty()
          && child_max.map(|m| m > 0).unwrap_or(true)
          && items.iter().all(|a| matches!(a, Expr::List(_)))
        {
          let sub_dims: Vec<Vec<i128>> = items
            .iter()
            .map(|it| get_dimensions(it, child_max))
            .collect();
          if sub_dims.iter().all(|d| d == &sub_dims[0]) {
            dims.extend(sub_dims[0].iter());
          }
        }
        dims
      }
      Expr::FunctionCall { name, args } => {
        let mut dims = vec![args.len() as i128];
        if !args.is_empty()
          && child_max.map(|m| m > 0).unwrap_or(true)
          && args.iter().all(
            |a| matches!(a, Expr::FunctionCall { name: n, .. } if n == name),
          )
        {
          let sub_dims: Vec<Vec<i128>> = args
            .iter()
            .map(|it| get_dimensions(it, child_max))
            .collect();
          if sub_dims.iter().all(|d| d == &sub_dims[0]) {
            dims.extend(sub_dims[0].iter());
          }
        }
        dims
      }
      // Image objects are opaque to Dimensions — use ImageDimensions instead
      _ => vec![],
    }
  }

  // Dimensions[SparseArray[Automatic, dims, default, rules]] returns the
  // array's dims directly instead of treating it as an opaque FunctionCall.
  if let Expr::FunctionCall { name, args: sa } = &args[0]
    && name == "SparseArray"
    && sa.len() == 4
    && matches!(&sa[0], Expr::Identifier(s) if s == "Automatic")
    && let Expr::List(dim_items) = &sa[1]
  {
    let dims: Vec<Expr> = match max_level {
      Some(n) => dim_items.iter().take(n).cloned().collect(),
      None => dim_items.to_vec(),
    };
    return Ok(Expr::List(dims.into()));
  }

  let dims = get_dimensions(&args[0], max_level);
  Ok(Expr::List(dims.into_iter().map(Expr::Integer).collect()))
}
