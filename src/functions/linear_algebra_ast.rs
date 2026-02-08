//! AST-native linear algebra functions.
//!
//! Dot, Det, Inverse, Tr, IdentityMatrix, DiagonalMatrix, Cross.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Helper: extract a matrix (list of lists) from an Expr.
/// Returns None if it's not a valid matrix.
fn expr_to_matrix(expr: &Expr) -> Option<Vec<Vec<Expr>>> {
  if let Expr::List(rows) = expr {
    let mut matrix = Vec::new();
    let ncols = if let Expr::List(first) = rows.first()? {
      first.len()
    } else {
      return None;
    };
    for row in rows {
      if let Expr::List(cols) = row {
        if cols.len() != ncols {
          return None;
        }
        matrix.push(cols.clone());
      } else {
        return None;
      }
    }
    Some(matrix)
  } else {
    None
  }
}

/// Helper: extract a vector (flat list) from an Expr
fn expr_to_vector(expr: &Expr) -> Option<Vec<Expr>> {
  if let Expr::List(items) = expr {
    // Make sure it's not a matrix (list of lists)
    if items.iter().any(|i| matches!(i, Expr::List(_))) {
      return None;
    }
    Some(items.clone())
  } else {
    None
  }
}

/// Helper: convert matrix back to Expr
fn matrix_to_expr(matrix: Vec<Vec<Expr>>) -> Expr {
  Expr::List(matrix.into_iter().map(Expr::List).collect())
}

/// Helper: evaluate a simple arithmetic expression on Exprs
fn eval_add(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x + y),
    (Expr::Integer(x), Expr::Real(y)) | (Expr::Real(y), Expr::Integer(x)) => {
      Expr::Real(*x as f64 + y)
    }
    (Expr::Real(x), Expr::Real(y)) => Expr::Real(x + y),
    _ => {
      // Symbolic addition via evaluator
      match crate::functions::math_ast::plus_ast(&[a.clone(), b.clone()]) {
        Ok(r) => r,
        Err(_) => Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        },
      }
    }
  }
}

fn eval_mul(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x * y),
    (Expr::Integer(x), Expr::Real(y)) | (Expr::Real(y), Expr::Integer(x)) => {
      Expr::Real(*x as f64 * y)
    }
    (Expr::Real(x), Expr::Real(y)) => Expr::Real(x * y),
    _ => match crate::functions::math_ast::times_ast(&[a.clone(), b.clone()]) {
      Ok(r) => r,
      Err(_) => Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(a.clone()),
        right: Box::new(b.clone()),
      },
    },
  }
}

fn eval_sub(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x - y),
    (Expr::Integer(x), Expr::Real(y)) => Expr::Real(*x as f64 - y),
    (Expr::Real(x), Expr::Integer(y)) => Expr::Real(x - *y as f64),
    (Expr::Real(x), Expr::Real(y)) => Expr::Real(x - y),
    _ => {
      match crate::functions::math_ast::subtract_ast(&[a.clone(), b.clone()]) {
        Ok(r) => r,
        Err(_) => Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Minus,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        },
      }
    }
  }
}

/// Dot[a, b] - dot product (vector) or matrix multiply
pub fn dot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Dot expects exactly 2 arguments".into(),
    ));
  }

  // Vector . Vector → scalar
  if let (Some(va), Some(vb)) =
    (expr_to_vector(&args[0]), expr_to_vector(&args[1]))
  {
    if va.len() != vb.len() {
      return Err(InterpreterError::EvaluationError(
        "Dot: vectors have incompatible lengths".into(),
      ));
    }
    let mut sum = Expr::Integer(0);
    for (a, b) in va.iter().zip(vb.iter()) {
      sum = eval_add(&sum, &eval_mul(a, b));
    }
    return Ok(sum);
  }

  // Matrix . Vector → vector
  if let (Some(ma), Some(vb)) =
    (expr_to_matrix(&args[0]), expr_to_vector(&args[1]))
  {
    let ncols = ma.first().map(|r| r.len()).unwrap_or(0);
    if ncols != vb.len() {
      return Err(InterpreterError::EvaluationError(
        "Dot: incompatible dimensions".into(),
      ));
    }
    let mut result = Vec::new();
    for row in &ma {
      let mut sum = Expr::Integer(0);
      for (a, b) in row.iter().zip(vb.iter()) {
        sum = eval_add(&sum, &eval_mul(a, b));
      }
      result.push(sum);
    }
    return Ok(Expr::List(result));
  }

  // Matrix . Matrix → matrix
  if let (Some(ma), Some(mb)) =
    (expr_to_matrix(&args[0]), expr_to_matrix(&args[1]))
  {
    let a_cols = ma.first().map(|r| r.len()).unwrap_or(0);
    let b_rows = mb.len();
    if a_cols != b_rows {
      return Err(InterpreterError::EvaluationError(
        "Dot: incompatible matrix dimensions".into(),
      ));
    }
    let b_cols = mb.first().map(|r| r.len()).unwrap_or(0);
    let mut result = Vec::new();
    for i in 0..ma.len() {
      let mut row = Vec::new();
      for j in 0..b_cols {
        let mut sum = Expr::Integer(0);
        for k in 0..a_cols {
          sum = eval_add(&sum, &eval_mul(&ma[i][k], &mb[k][j]));
        }
        row.push(sum);
      }
      result.push(row);
    }
    return Ok(matrix_to_expr(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Dot".to_string(),
    args: args.to_vec(),
  })
}

/// Det[matrix] - determinant of a square matrix
pub fn det_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Det expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Det".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let n = matrix.len();
  if n == 0 || matrix.iter().any(|row| row.len() != n) {
    return Err(InterpreterError::EvaluationError(
      "Det: argument must be a square matrix".into(),
    ));
  }
  Ok(determinant(&matrix))
}

/// Compute determinant recursively via cofactor expansion
fn determinant(matrix: &[Vec<Expr>]) -> Expr {
  let n = matrix.len();
  if n == 1 {
    return matrix[0][0].clone();
  }
  if n == 2 {
    return eval_sub(
      &eval_mul(&matrix[0][0], &matrix[1][1]),
      &eval_mul(&matrix[0][1], &matrix[1][0]),
    );
  }
  let mut det = Expr::Integer(0);
  for j in 0..n {
    let mut minor = Vec::new();
    for i in 1..n {
      let mut row = Vec::new();
      for k in 0..n {
        if k != j {
          row.push(matrix[i][k].clone());
        }
      }
      minor.push(row);
    }
    let cofactor = eval_mul(&matrix[0][j], &determinant(&minor));
    if j % 2 == 0 {
      det = eval_add(&det, &cofactor);
    } else {
      det = eval_sub(&det, &cofactor);
    }
  }
  det
}

/// Inverse[matrix] - matrix inverse (integer matrices → rational entries)
pub fn inverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Inverse expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Inverse".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let n = matrix.len();
  if n == 0 || matrix.iter().any(|row| row.len() != n) {
    return Err(InterpreterError::EvaluationError(
      "Inverse: argument must be a square matrix".into(),
    ));
  }

  // Try integer matrix inverse via adjugate method
  let det = determinant(&matrix);
  if matches!(&det, Expr::Integer(0)) {
    return Err(InterpreterError::EvaluationError(
      "Inverse: matrix is singular".into(),
    ));
  }

  // Compute cofactor matrix
  let mut cofactors = vec![vec![Expr::Integer(0); n]; n];
  for i in 0..n {
    for j in 0..n {
      let mut minor = Vec::new();
      for ii in 0..n {
        if ii == i {
          continue;
        }
        let mut row = Vec::new();
        for jj in 0..n {
          if jj == j {
            continue;
          }
          row.push(matrix[ii][jj].clone());
        }
        minor.push(row);
      }
      let minor_det = determinant(&minor);
      if (i + j) % 2 == 0 {
        cofactors[i][j] = minor_det;
      } else {
        cofactors[i][j] = eval_sub(&Expr::Integer(0), &minor_det);
      }
    }
  }

  // Transpose cofactor matrix (adjugate) and divide by det
  let mut result = vec![vec![Expr::Integer(0); n]; n];
  for i in 0..n {
    for j in 0..n {
      result[i][j] = eval_divide(&cofactors[j][i], &det);
    }
  }

  Ok(matrix_to_expr(result))
}

/// Helper: divide two expressions, producing Rational if needed
fn eval_divide(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) if *y != 0 => {
      if x % y == 0 {
        Expr::Integer(x / y)
      } else {
        // Simplify the fraction
        let g = gcd_i128(x.abs(), y.abs());
        let (n, d) = (x / g, y / g);
        if d < 0 {
          // Keep denominator positive
          crate::functions::math_ast::make_rational_pub(-n, -d)
        } else {
          crate::functions::math_ast::make_rational_pub(n, d)
        }
      }
    }
    _ => {
      match crate::functions::math_ast::divide_ast(&[a.clone(), b.clone()]) {
        Ok(r) => r,
        Err(_) => Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        },
      }
    }
  }
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a, b);
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Tr[matrix] - trace of a square matrix
pub fn tr_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Tr expects exactly 1 argument".into(),
    ));
  }
  // Also support Tr on a plain list (sum of elements)
  if let Some(vec) = expr_to_vector(&args[0]) {
    let mut sum = Expr::Integer(0);
    for v in &vec {
      sum = eval_add(&sum, v);
    }
    return Ok(sum);
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Tr".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let n = matrix.len();
  let ncols = matrix.first().map(|r| r.len()).unwrap_or(0);
  let min_dim = n.min(ncols);
  let mut trace = Expr::Integer(0);
  for i in 0..min_dim {
    trace = eval_add(&trace, &matrix[i][i]);
  }
  Ok(trace)
}

/// IdentityMatrix[n] - n×n identity matrix
pub fn identity_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "IdentityMatrix expects exactly 1 argument".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "IdentityMatrix".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let mut matrix = Vec::new();
  for i in 0..n {
    let mut row = vec![Expr::Integer(0); n];
    row[i] = Expr::Integer(1);
    matrix.push(Expr::List(row));
  }
  Ok(Expr::List(matrix))
}

/// DiagonalMatrix[list] - diagonal matrix from a list
pub fn diagonal_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DiagonalMatrix expects exactly 1 argument".into(),
    ));
  }
  let diag = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DiagonalMatrix".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let n = diag.len();
  let mut matrix = Vec::new();
  for i in 0..n {
    let mut row = vec![Expr::Integer(0); n];
    row[i] = diag[i].clone();
    matrix.push(Expr::List(row));
  }
  Ok(Expr::List(matrix))
}

/// Cross[a, b] - cross product of two 3-vectors
pub fn cross_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Cross expects exactly 2 arguments".into(),
    ));
  }
  let va = match &args[0] {
    Expr::List(items) if items.len() == 3 => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Cross".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let vb = match &args[1] {
    Expr::List(items) if items.len() == 3 => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Cross".to_string(),
        args: args.to_vec(),
      });
    }
  };
  Ok(Expr::List(vec![
    eval_sub(&eval_mul(&va[1], &vb[2]), &eval_mul(&va[2], &vb[1])),
    eval_sub(&eval_mul(&va[2], &vb[0]), &eval_mul(&va[0], &vb[2])),
    eval_sub(&eval_mul(&va[0], &vb[1]), &eval_mul(&va[1], &vb[0])),
  ]))
}
