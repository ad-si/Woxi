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
      args: vec![expr.clone()],
    }),
  }
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

/// SymmetricMatrixQ[m] - True if m is a symmetric square matrix (m[i][j] == m[j][i]).
pub fn symmetric_matrix_q_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(rows) => {
      let n = rows.len();
      if n == 0 {
        return Ok(Expr::Identifier("False".to_string()));
      }
      // All rows must be lists of length n (square matrix)
      let mut grid: Vec<&Vec<Expr>> = Vec::with_capacity(n);
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
      None => dim_items.clone(),
    };
    return Ok(Expr::List(dims));
  }

  let dims = get_dimensions(&args[0], max_level);
  Ok(Expr::List(dims.into_iter().map(Expr::Integer).collect()))
}
