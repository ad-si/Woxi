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

/// TensorRank[expr] - rank of a tensor (same as ArrayDepth for concrete lists,
/// but stays unevaluated for non-list expressions like symbols).
pub fn tensor_rank_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
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
