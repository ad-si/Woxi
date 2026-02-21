#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based ArrayDepth: compute depth of nested lists.
pub fn array_depth_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  fn compute_depth(expr: &Expr) -> i128 {
    match expr {
      Expr::List(items) => {
        if items.is_empty() {
          1
        } else {
          1 + items.iter().map(compute_depth).min().unwrap_or(0)
        }
      }
      _ => 0,
    }
  }

  Ok(Expr::Integer(compute_depth(list)))
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

/// Dimensions[list] - Returns the dimensions of a nested list
pub fn dimensions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Dimensions expects exactly 1 argument".into(),
    ));
  }

  fn get_dimensions(expr: &Expr) -> Vec<i128> {
    match expr {
      Expr::List(items) => {
        let mut dims = vec![items.len() as i128];
        if !items.is_empty() {
          // Check if all sub-elements have the same dimensions
          let sub_dims: Vec<Vec<i128>> =
            items.iter().map(get_dimensions).collect();
          if !sub_dims.is_empty() && sub_dims.iter().all(|d| d == &sub_dims[0])
          {
            dims.extend(sub_dims[0].iter());
          }
        }
        dims
      }
      Expr::FunctionCall { name, args } => {
        let mut dims = vec![args.len() as i128];
        if !args.is_empty() {
          // Check if all sub-elements are function calls with the same head and dimensions
          let sub_dims: Vec<Vec<i128>> =
            args.iter().map(get_dimensions).collect();
          if !sub_dims.is_empty()
            && sub_dims.iter().all(|d| d == &sub_dims[0])
            && args.iter().all(|a| {
              matches!(a, Expr::FunctionCall { name: n, .. } if n == name)
                || matches!(a, Expr::List(_))
            })
          {
            dims.extend(sub_dims[0].iter());
          }
        }
        dims
      }
      _ => vec![],
    }
  }

  let dims = get_dimensions(&args[0]);
  Ok(Expr::List(dims.into_iter().map(Expr::Integer).collect()))
}
