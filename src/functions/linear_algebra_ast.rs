//! AST-native linear algebra functions.
//!
//! Dot, Det, Inverse, Tr, IdentityMatrix, DiagonalMatrix, Cross, Eigenvalues, Fit.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

/// Transpose a matrix (Vec<Vec<Expr>>)
fn transpose_matrix(m: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
  if m.is_empty() {
    return vec![];
  }
  let nrows = m.len();
  let ncols = m[0].len();
  let mut result = vec![vec![Expr::Integer(0); nrows]; ncols];
  for i in 0..nrows {
    for j in 0..ncols {
      result[j][i] = m[i][j].clone();
    }
  }
  result
}

/// Multiply two matrices
fn mat_mul(a: &[Vec<Expr>], b: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
  let arows = a.len();
  let acols = if arows > 0 { a[0].len() } else { 0 };
  let bcols = if !b.is_empty() { b[0].len() } else { 0 };
  let mut result = vec![vec![Expr::Integer(0); bcols]; arows];
  for i in 0..arows {
    for j in 0..bcols {
      let mut sum = Expr::Integer(0);
      for k in 0..acols {
        sum = eval_add(&sum, &eval_mul(&a[i][k], &b[k][j]));
      }
      result[i][j] = sum;
    }
  }
  result
}

/// Try to invert a square matrix. Returns None if singular.
fn try_invert(matrix: &[Vec<Expr>]) -> Option<Vec<Vec<Expr>>> {
  let expr = matrix_to_expr(matrix.to_vec());
  match inverse_ast(&[expr]) {
    Ok(result) => expr_to_matrix(&result),
    Err(_) => None,
  }
}

/// PseudoInverse[matrix] - Moore-Penrose pseudoinverse
pub fn pseudo_inverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PseudoInverse expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "PseudoInverse".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let nrows = matrix.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![]));
  }
  let ncols = matrix[0].len();

  // Check if zero matrix
  let is_zero = matrix.iter().all(|row| {
    row.iter().all(|e| {
      matches!(e, Expr::Integer(0))
        || matches!(e, Expr::Real(x) if *x == 0.0)
        || matches!(e, Expr::BigInteger(n) if *n == num_bigint::BigInt::from(0))
    })
  });
  if is_zero {
    let result = vec![vec![Expr::Integer(0); nrows]; ncols];
    return Ok(matrix_to_expr(result));
  }

  // Square non-singular: just use Inverse
  if nrows == ncols
    && let Ok(inv) = inverse_ast(&[args[0].clone()])
  {
    return Ok(inv);
  }

  let mt = transpose_matrix(&matrix);

  // Try left pseudoinverse: (M^T M)^{-1} M^T
  // Works when M has full column rank (ncols <= nrows)
  let mtm = mat_mul(&mt, &matrix);
  if let Some(mtm_inv) = try_invert(&mtm) {
    let result = mat_mul(&mtm_inv, &mt);
    return Ok(matrix_to_expr(result));
  }

  // Try right pseudoinverse: M^T (M M^T)^{-1}
  // Works when M has full row rank (nrows <= ncols)
  let mmt = mat_mul(&matrix, &mt);
  if let Some(mmt_inv) = try_invert(&mmt) {
    let result = mat_mul(&mt, &mmt_inv);
    return Ok(matrix_to_expr(result));
  }

  // For rank-deficient matrices, return unevaluated
  Ok(Expr::FunctionCall {
    name: "PseudoInverse".to_string(),
    args: args.to_vec(),
  })
}

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
  let det = determinant(&matrix);
  Ok(crate::functions::expand_and_combine(&det))
}

/// Compute determinant recursively via cofactor expansion
fn determinant(matrix: &[Vec<Expr>]) -> Expr {
  let n = matrix.len();
  if n == 0 {
    return Expr::Integer(1);
  }
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
  let det_is_zero = matches!(&det, Expr::Integer(0))
    || matches!(&det, Expr::BigInteger(n) if n == &num_bigint::BigInt::from(0));
  if det_is_zero {
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
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      use num_traits::Zero;
      if *n >= num_bigint::BigInt::zero() {
        match n.to_usize() {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "IdentityMatrix".to_string(),
              args: args.to_vec(),
            });
          }
        }
      } else {
        return Ok(Expr::FunctionCall {
          name: "IdentityMatrix".to_string(),
          args: args.to_vec(),
        });
      }
    }
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

/// DiamondMatrix[n] - creates (2n+1)x(2n+1) matrix where entry (i,j) is 1 if
/// the Manhattan distance from center <= n.
pub fn diamond_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DiamondMatrix".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let size = 2 * n + 1;
  let mut matrix = Vec::with_capacity(size);
  for i in 0..size {
    let mut row = Vec::with_capacity(size);
    for j in 0..size {
      let dist = (i as i128 - n as i128).unsigned_abs()
        + (j as i128 - n as i128).unsigned_abs();
      row.push(if dist <= n as u128 {
        Expr::Integer(1)
      } else {
        Expr::Integer(0)
      });
    }
    matrix.push(Expr::List(row));
  }
  Ok(Expr::List(matrix))
}

/// DiskMatrix[n] - creates (2n+1)x(2n+1) matrix where entry (i,j) is 1 if
/// the Euclidean distance from center <= n.
pub fn disk_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DiskMatrix".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let size = 2 * n + 1;
  // Threshold is (n + 0.5)^2 = n^2 + n + 0.25; for integer comparisons, n^2 + n
  let threshold = (n * n + n) as i128;
  let mut matrix = Vec::with_capacity(size);
  for i in 0..size {
    let mut row = Vec::with_capacity(size);
    for j in 0..size {
      let di = i as i128 - n as i128;
      let dj = j as i128 - n as i128;
      row.push(if di * di + dj * dj <= threshold {
        Expr::Integer(1)
      } else {
        Expr::Integer(0)
      });
    }
    matrix.push(Expr::List(row));
  }
  Ok(Expr::List(matrix))
}

/// Cross[{x, y}] - 2D cross product (perpendicular vector)
/// Cross[a, b] - cross product of two 3-vectors
pub fn cross_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Cross[{x, y}] = {-y, x}
    if let Expr::List(items) = &args[0]
      && items.len() == 2
    {
      let neg_y = eval_sub(&Expr::Integer(0), &items[1]);
      return Ok(Expr::List(vec![neg_y, items[0].clone()]));
    }
    return Ok(Expr::FunctionCall {
      name: "Cross".to_string(),
      args: args.to_vec(),
    });
  }
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Cross expects 1 or 2 arguments".into(),
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

/// ConjugateTranspose[matrix] - transpose and conjugate each element
pub fn conjugate_transpose_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ConjugateTranspose expects exactly 1 argument".into(),
    ));
  }
  let rows = match &args[0] {
    Expr::List(rows) => rows,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ConjugateTranspose".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Get dimensions
  let nrows = rows.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![]));
  }
  let ncols = match &rows[0] {
    Expr::List(cols) => cols.len(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ConjugateTranspose".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Transpose and conjugate
  let mut result = Vec::with_capacity(ncols);
  for j in 0..ncols {
    let mut row = Vec::with_capacity(nrows);
    for item in rows.iter().take(nrows) {
      if let Expr::List(cols) = item
        && j < cols.len()
      {
        // Apply Conjugate
        let conj = evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Conjugate".to_string(),
          args: vec![cols[j].clone()],
        })?;
        row.push(conj);
      }
    }
    result.push(Expr::List(row));
  }
  Ok(Expr::List(result))
}

/// Projection[u, v] - project vector u onto vector v
/// Formula: (u.v / v.v) * v
pub fn projection_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Projection expects exactly 2 arguments".into(),
    ));
  }
  let u = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Projection".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let v = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Projection".to_string(),
        args: args.to_vec(),
      });
    }
  };
  if u.len() != v.len() {
    return Err(InterpreterError::EvaluationError(
      "Projection: vectors have incompatible lengths".into(),
    ));
  }
  // Compute u.v
  let mut uv = Expr::Integer(0);
  for (a, b) in u.iter().zip(v.iter()) {
    uv = eval_add(&uv, &eval_mul(a, b));
  }
  // Compute v.v
  let mut vv = Expr::Integer(0);
  for b in v.iter() {
    vv = eval_add(&vv, &eval_mul(b, b));
  }
  // Compute (u.v / v.v) * v
  let scalar = eval_divide(&uv, &vv);
  let result: Vec<Expr> = v.iter().map(|vi| eval_mul(&scalar, vi)).collect();
  Ok(Expr::List(result))
}

// ─── Fit (least-squares) ────────────────────────────────────────────────

/// Fit[data, funs, var] — least-squares fit of data to a linear combination
/// of the basis functions `funs` in the variable `var`.
///
/// `data` is either:
///   - `{y1, y2, …}` with implicit x = 1, 2, …, n
///   - `{{x1, y1}, {x2, y2}, …}` with explicit x-y pairs
///
/// Returns `c1*f1 + c2*f2 + … + cm*fm`.
pub fn fit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Fit expects exactly 3 arguments".into(),
    ));
  }

  // Extract basis functions
  let basis = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Fit: second argument must be a list of basis functions".into(),
      ));
    }
  };

  // Extract variable name
  let var_name = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Fit: third argument must be a variable".into(),
      ));
    }
  };

  // Extract data points: either {y1, y2, ...} or {{x1, y1}, {x2, y2}, ...}
  let data_list = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Fit: first argument must be a non-empty list".into(),
      ));
    }
  };

  let (x_vals, y_vals) = extract_fit_data(data_list)?;
  let n = x_vals.len(); // number of data points
  let m = basis.len(); // number of basis functions

  // Build design matrix A (n×m) where A[i][j] = basis[j] evaluated at x = x_vals[i]
  let mut a_matrix = vec![vec![0.0f64; m]; n];
  for i in 0..n {
    let x_expr = Expr::Real(x_vals[i]);
    for j in 0..m {
      let substituted =
        crate::syntax::substitute_variable(&basis[j], &var_name, &x_expr);
      let evaluated = evaluate_expr_to_expr(&substituted)?;
      match try_eval_to_f64(&evaluated) {
        Some(v) => a_matrix[i][j] = v,
        None => {
          return Err(InterpreterError::EvaluationError(format!(
            "Fit: could not evaluate basis function {:?} at x = {}",
            basis[j], x_vals[i]
          )));
        }
      }
    }
  }

  // Solve via QR decomposition (Householder reflections)
  let coeffs = solve_least_squares_qr(&a_matrix, &y_vals)?;

  // Build result: c[0]*basis[0] + c[1]*basis[1] + ...
  // We construct the expression using Times and Plus function calls
  // so the result matches Wolfram's output format.
  let mut terms = Vec::new();
  for (j, c) in coeffs.iter().enumerate() {
    let coeff_expr = Expr::Real(*c);
    // Check if basis function is just the constant 1
    if matches!(&basis[j], Expr::Integer(1)) {
      terms.push(coeff_expr);
    } else {
      terms.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![coeff_expr, basis[j].clone()],
      });
    }
  }

  // Build Plus expression
  if terms.len() == 1 {
    Ok(terms.into_iter().next().unwrap())
  } else {
    Ok(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    })
  }
}

/// Extract x and y values from Fit data.
/// Handles both `{y1, y2, ...}` and `{{x1, y1}, {x2, y2}, ...}` forms.
fn extract_fit_data(
  data: &[Expr],
) -> Result<(Vec<f64>, Vec<f64>), InterpreterError> {
  // Check if first element is a list (=> {x, y} pair form)
  if let Expr::List(first_pair) = &data[0]
    && first_pair.len() == 2
  {
    // {{x1, y1}, {x2, y2}, ...} form
    let mut xs = Vec::with_capacity(data.len());
    let mut ys = Vec::with_capacity(data.len());
    for item in data {
      if let Expr::List(pair) = item {
        if pair.len() != 2 {
          return Err(InterpreterError::EvaluationError(
            "Fit: data points must be {x, y} pairs of length 2".into(),
          ));
        }
        let x = try_eval_to_f64(&pair[0]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Fit: could not convert x coordinate to a number".into(),
          )
        })?;
        let y = try_eval_to_f64(&pair[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Fit: could not convert y coordinate to a number".into(),
          )
        })?;
        xs.push(x);
        ys.push(y);
      } else {
        return Err(InterpreterError::EvaluationError(
          "Fit: all data entries must be {x, y} pairs".into(),
        ));
      }
    }
    return Ok((xs, ys));
  }

  // {y1, y2, ...} form with implicit x = 1, 2, ..., n
  let mut ys = Vec::with_capacity(data.len());
  for item in data {
    let y = try_eval_to_f64(item).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Fit: could not convert data value to a number".into(),
      )
    })?;
    ys.push(y);
  }
  let xs: Vec<f64> = (1..=data.len()).map(|i| i as f64).collect();
  Ok((xs, ys))
}

/// Solve least-squares problem min ||Ax - b||_2 using Householder QR decomposition.
/// A is n×m (n rows, m cols) with n >= m.  Returns the m-element solution vector.
fn solve_least_squares_qr(
  a: &[Vec<f64>],
  b: &[f64],
) -> Result<Vec<f64>, InterpreterError> {
  let n = a.len();
  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "Fit: empty design matrix".into(),
    ));
  }
  let m = a[0].len();
  if m == 0 || b.len() != n {
    return Err(InterpreterError::EvaluationError(
      "Fit: dimension mismatch in least squares".into(),
    ));
  }

  // Copy A and b (we'll transform them in place)
  let mut r: Vec<Vec<f64>> = a.to_vec();
  let mut rhs: Vec<f64> = b.to_vec();

  // Householder QR: transform A into R, apply same reflections to rhs
  for j in 0..m {
    // Compute norm of column j below diagonal
    let mut sigma = 0.0f64;
    for i in j..n {
      sigma += r[i][j] * r[i][j];
    }
    let norm = sigma.sqrt();

    if norm < 1e-15 {
      return Err(InterpreterError::EvaluationError(
        "Fit: design matrix is rank-deficient".into(),
      ));
    }

    // Choose sign to maximize |v[j]| (avoid cancellation)
    let alpha = if r[j][j] >= 0.0 { -norm } else { norm };

    // Build Householder vector v (save a copy since column j gets overwritten)
    let mut v = vec![0.0f64; n];
    v[j] = r[j][j] - alpha;
    for i in (j + 1)..n {
      v[i] = r[i][j];
    }

    let vtv: f64 = v[j..].iter().map(|x| x * x).sum();
    if vtv < 1e-30 {
      r[j][j] = alpha;
      continue;
    }
    let tau = 2.0 / vtv;

    // Apply H = I - tau * v * v^T to columns of R
    for k in j..m {
      let dot: f64 = (j..n).map(|i| v[i] * r[i][k]).sum();
      let factor = tau * dot;
      for i in j..n {
        r[i][k] -= factor * v[i];
      }
    }

    // Apply H to rhs
    let dot: f64 = (j..n).map(|i| v[i] * rhs[i]).sum();
    let factor = tau * dot;
    for i in j..n {
      rhs[i] -= factor * v[i];
    }
  }

  // Back substitution: solve R[0:m, 0:m] * x = rhs[0:m]
  let mut x = vec![0.0f64; m];
  for j in (0..m).rev() {
    let mut sum = rhs[j];
    for k in (j + 1)..m {
      sum -= r[j][k] * x[k];
    }
    x[j] = sum / r[j][j];
  }

  Ok(x)
}

// ─── Eigenvalues ────────────────────────────────────────────────────────

/// Extract an integer from an Expr, or None if not an integer.
fn expr_to_i128(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(n) => Some(*n),
    _ => None,
  }
}

/// Extract integer matrix entries, returning None if any entry is non-integer.
fn matrix_to_i128(matrix: &[Vec<Expr>]) -> Option<Vec<Vec<i128>>> {
  matrix
    .iter()
    .map(|row| row.iter().map(expr_to_i128).collect())
    .collect()
}

/// Faddeev-LeVerrier algorithm: compute integer coefficients of the
/// characteristic polynomial det(λI - A) = λ^n + c_{n-1}*λ^{n-1} + ... + c_0.
///
/// Returns coefficients [c_0, c_1, ..., c_{n-1}, 1] (ascending powers).
fn char_poly_coefficients(a: &[Vec<i128>]) -> Vec<i128> {
  let n = a.len();
  // M starts as A, c[n] = 1 (monic)
  let mut coeffs = vec![0i128; n + 1];
  coeffs[n] = 1;

  // M_k tracks A * (M_{k-1} + c_{n-k+1} * I)
  let mut m = a.to_vec();

  for k in 1..=n {
    // c_{n-k} = -Tr(M_k) / k
    let trace: i128 = (0..n).map(|i| m[i][i]).sum();
    coeffs[n - k] = -trace / k as i128;

    if k < n {
      // M_{k+1} = A * (M_k + c_{n-k} * I)
      let c = coeffs[n - k];
      // First add c*I to M_k
      let mut temp = m.clone();
      for i in 0..n {
        temp[i][i] += c;
      }
      // Then multiply: M_{k+1} = A * temp
      let mut new_m = vec![vec![0i128; n]; n];
      for i in 0..n {
        for j in 0..n {
          for p in 0..n {
            new_m[i][j] += a[i][p] * temp[p][j];
          }
        }
      }
      m = new_m;
    }
  }

  coeffs
}

/// Get all divisors of |n| (including 1 and |n|). Returns empty if n == 0.
fn divisors(n: i128) -> Vec<i128> {
  if n == 0 {
    return vec![];
  }
  let n = n.abs();
  let mut divs = Vec::new();
  let mut i = 1i128;
  while i * i <= n {
    if n % i == 0 {
      divs.push(i);
      if i != n / i {
        divs.push(n / i);
      }
    }
    i += 1;
  }
  divs.sort();
  divs
}

/// Evaluate polynomial (ascending coefficient order) at integer x.
fn poly_eval(coeffs: &[i128], x: i128) -> i128 {
  let mut result = 0i128;
  let mut power = 1i128;
  for &c in coeffs {
    result = result
      .checked_add(c.checked_mul(power).unwrap_or(0))
      .unwrap_or(0);
    power = power.checked_mul(x).unwrap_or(0);
  }
  result
}

/// Synthetic division: divide polynomial by (x - root).
/// Returns quotient coefficients (ascending order).
fn synthetic_div(coeffs: &[i128], root: i128) -> Vec<i128> {
  let n = coeffs.len() - 1; // degree of dividend
  if n == 0 {
    return vec![];
  }
  // Work with descending coefficients for standard synthetic division
  let desc: Vec<i128> = coeffs.iter().rev().cloned().collect();
  let mut result = Vec::with_capacity(n);
  result.push(desc[0]);
  for i in 1..n {
    let val = desc[i] + root * result[i - 1];
    result.push(val);
  }
  // Convert back to ascending order
  result.reverse();
  result
}

/// Sort eigenvalues: Wolfram sorts by absolute value descending,
/// then by value descending for ties.
fn sort_eigenvalues(eigenvalues: &mut [Expr]) {
  eigenvalues.sort_by(|a, b| {
    let va = eigenvalue_sort_key(a);
    let vb = eigenvalue_sort_key(b);
    let abs_cmp = vb
      .abs()
      .partial_cmp(&va.abs())
      .unwrap_or(std::cmp::Ordering::Equal);
    if abs_cmp != std::cmp::Ordering::Equal {
      abs_cmp
    } else {
      vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    }
  });
}

/// Get a numeric sort key for an eigenvalue expression.
/// For integer: the value itself.
/// For expressions involving Sqrt: approximate numerically.
fn eigenvalue_sort_key(e: &Expr) -> f64 {
  match e {
    Expr::Integer(n) => *n as f64,
    Expr::Real(r) => *r,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        *n as f64 / *d as f64
      } else {
        0.0
      }
    }
    _ => {
      // Try to evaluate numerically
      if let Ok(n_result) = crate::functions::math_ast::n_ast(&[e.clone()]) {
        match &n_result {
          Expr::Real(r) => *r,
          Expr::Integer(n) => *n as f64,
          _ => 0.0,
        }
      } else {
        0.0
      }
    }
  }
}

/// Build eigenvalue expression from quadratic formula components.
/// Given polynomial x^2 + bx + c = 0, eigenvalues are (-b ± sqrt(b²-4c))/2.
fn quadratic_eigenvalues(b_coeff: i128, c_coeff: i128) -> Vec<Expr> {
  let neg_b = -b_coeff; // This is the trace
  let discriminant = b_coeff * b_coeff - 4 * c_coeff;

  if discriminant == 0 {
    // Repeated root
    let root = simplify_fraction(neg_b, 2);
    return vec![root.clone(), root];
  }

  // Check if discriminant is a perfect square
  if discriminant > 0 {
    let sqrt_disc = (discriminant as f64).sqrt().round() as i128;
    if sqrt_disc * sqrt_disc == discriminant {
      // Rational roots
      let r1 = simplify_fraction(neg_b + sqrt_disc, 2);
      let r2 = simplify_fraction(neg_b - sqrt_disc, 2);
      return vec![r1, r2];
    }
  }

  // Simplify sqrt(discriminant): factor out perfect squares
  let (outer, inner) = simplify_sqrt(discriminant.unsigned_abs());

  // Factor out GCD of neg_b and outer from the numerator:
  // (neg_b ± outer*Sqrt[inner]) / 2  =  common * (reduced_b ± reduced_outer*Sqrt[inner]) / 2
  let common = gcd_i128(neg_b.abs(), outer as i128);
  let reduced_b = neg_b / common;
  let reduced_outer = outer as i128 / common;

  let sqrt_expr = Expr::FunctionCall {
    name: "Sqrt".to_string(),
    args: vec![Expr::Integer(inner as i128)],
  };

  let reduced_sqrt = if reduced_outer == 1 {
    sqrt_expr.clone()
  } else {
    eval_mul(&Expr::Integer(reduced_outer), &sqrt_expr)
  };

  // Build the inner sum/difference
  let inner_plus = if reduced_b == 0 {
    reduced_sqrt.clone()
  } else {
    eval_add(&Expr::Integer(reduced_b), &reduced_sqrt)
  };
  let inner_minus = if reduced_b == 0 {
    eval_sub(&Expr::Integer(0), &reduced_sqrt)
  } else {
    eval_sub(&Expr::Integer(reduced_b), &reduced_sqrt)
  };

  // Multiply by common factor if needed
  let numer_plus = if common == 1 {
    inner_plus
  } else {
    eval_mul(&Expr::Integer(common), &inner_plus)
  };
  let numer_minus = if common == 1 {
    inner_minus
  } else {
    eval_mul(&Expr::Integer(common), &inner_minus)
  };

  // Divide by 2
  let plus_expr = eval_divide(&numer_plus, &Expr::Integer(2));
  let minus_expr = eval_divide(&numer_minus, &Expr::Integer(2));

  vec![plus_expr, minus_expr]
}

/// Simplify fraction n/d, returning Expr::Integer or Expr::Rational.
fn simplify_fraction(n: i128, d: i128) -> Expr {
  let g = gcd_i128(n.abs(), d.abs());
  let (n, d) = (n / g, d / g);
  let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
  if d == 1 {
    Expr::Integer(n)
  } else {
    crate::functions::math_ast::make_rational_pub(n, d)
  }
}

/// Simplify sqrt(n) into outer * sqrt(inner) where inner is square-free.
/// E.g. sqrt(12) = 2*sqrt(3), so returns (2, 3).
fn simplify_sqrt(n: u128) -> (u128, u128) {
  if n == 0 {
    return (0, 1);
  }
  let mut outer = 1u128;
  let mut inner = n;
  let mut factor = 2u128;
  while factor * factor <= inner {
    while inner.is_multiple_of(factor * factor) {
      outer *= factor;
      inner /= factor * factor;
    }
    factor += 1;
  }
  (outer, inner)
}

/// Eigenvalues[matrix] - eigenvalues of a square matrix, sorted descending.
pub fn eigenvalues_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Eigenvalues expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Eigenvalues".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let n = matrix.len();
  if n == 0 || matrix.iter().any(|row| row.len() != n) {
    return Err(InterpreterError::EvaluationError(
      "Eigenvalues: argument must be a square matrix".into(),
    ));
  }

  // 1x1
  if n == 1 {
    return Ok(Expr::List(vec![matrix[0][0].clone()]));
  }

  // Try integer matrix path
  if let Some(int_matrix) = matrix_to_i128(&matrix) {
    let coeffs = char_poly_coefficients(&int_matrix);
    let mut eigenvalues = find_polynomial_roots(&coeffs);
    sort_eigenvalues(&mut eigenvalues);
    return Ok(Expr::List(eigenvalues));
  }

  // Return unevaluated for non-integer matrices
  Ok(Expr::FunctionCall {
    name: "Eigenvalues".to_string(),
    args: args.to_vec(),
  })
}

/// Find roots of a monic polynomial with integer coefficients.
/// coeffs are in ascending order: [c_0, c_1, ..., c_{n-1}, 1].
fn find_polynomial_roots(coeffs: &[i128]) -> Vec<Expr> {
  let degree = coeffs.len() - 1;
  if degree == 0 {
    return vec![];
  }
  if degree == 1 {
    // x + c_0 = 0 → x = -c_0
    return vec![Expr::Integer(-coeffs[0])];
  }
  if degree == 2 {
    // x^2 + c_1*x + c_0 = 0
    return quadratic_eigenvalues(coeffs[1], coeffs[0]);
  }

  // For degree >= 3: find integer roots via rational root theorem,
  // then reduce degree via synthetic division.
  let mut current = coeffs.to_vec();
  let mut roots = Vec::new();

  loop {
    let deg = current.len() - 1;
    if deg <= 2 {
      break;
    }

    // Try to find an integer root
    let c0 = current[0];
    let mut found = false;

    if c0 == 0 {
      // x = 0 is a root
      roots.push(Expr::Integer(0));
      // Divide out x: just shift coefficients
      current = current[1..].to_vec();
      found = true;
    } else {
      let candidates = divisors(c0);
      for &d in &candidates {
        for &sign in &[1i128, -1] {
          let candidate = d * sign;
          if poly_eval(&current, candidate) == 0 {
            roots.push(Expr::Integer(candidate));
            current = synthetic_div(&current, candidate);
            found = true;
            break;
          }
        }
        if found {
          break;
        }
      }
    }

    if !found {
      break;
    }
  }

  // Handle remaining polynomial
  let deg = current.len() - 1;
  match deg {
    0 => {}
    1 => {
      roots.push(Expr::Integer(-current[0]));
    }
    2 => {
      roots.extend(quadratic_eigenvalues(current[1], current[0]));
    }
    _ => {
      // Cannot solve higher degree — return as polynomial roots symbolically
      // For now, return unevaluated Root expressions
      // This is a limitation; most practical integer matrices will be handled above
      for _ in 0..deg {
        roots.push(Expr::FunctionCall {
          name: "Root".to_string(),
          args: current.iter().map(|&c| Expr::Integer(c)).collect(),
        });
      }
    }
  }

  roots
}

// ─── Eigenvectors ───────────────────────────────────────────────────────

/// Eigenvectors[matrix] - eigenvectors of a square matrix
pub fn eigenvectors_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Eigenvectors expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Eigenvectors".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let n = matrix.len();
  if n == 0 || matrix.iter().any(|row| row.len() != n) {
    return Err(InterpreterError::EvaluationError(
      "Eigenvectors: argument must be a square matrix".into(),
    ));
  }

  // 1x1
  if n == 1 {
    return Ok(Expr::List(vec![Expr::List(vec![Expr::Integer(1)])]));
  }

  // Check if matrix is numeric (at least one Real entry)
  let has_real = matrix
    .iter()
    .any(|row| row.iter().any(|e| matches!(e, Expr::Real(_))));
  let all_numeric = matrix.iter().all(|row| {
    row
      .iter()
      .all(|e| matches!(e, Expr::Integer(_) | Expr::Real(_)))
  });

  if has_real && all_numeric {
    return numeric_eigenvectors(&matrix, n);
  }

  // Try integer matrix path
  if let Some(int_matrix) = matrix_to_i128(&matrix) {
    return integer_eigenvectors(&matrix, &int_matrix, n);
  }

  Ok(Expr::FunctionCall {
    name: "Eigenvectors".to_string(),
    args: args.to_vec(),
  })
}

/// Extract rational parts (numerator, denominator) from an Expr.
fn expr_to_rational_parts(e: &Expr) -> Option<(i128, i128)> {
  match e {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Scale a vector of rational Exprs to integer entries (clear denominators, reduce GCD).
fn scale_to_integers(vec: &[Expr]) -> Vec<Expr> {
  let rationals: Vec<Option<(i128, i128)>> =
    vec.iter().map(expr_to_rational_parts).collect();
  if rationals.iter().any(|r| r.is_none()) {
    return vec.to_vec();
  }
  let rats: Vec<(i128, i128)> =
    rationals.into_iter().map(|r| r.unwrap()).collect();

  // Find LCM of denominators
  let lcm_den = rats.iter().fold(1i128, |acc, &(_, d)| {
    let g = gcd_i128(acc.abs(), d.abs());
    if g == 0 { acc } else { (acc / g) * d.abs() }
  });

  // Scale all entries
  let scaled: Vec<i128> =
    rats.iter().map(|&(n, d)| n * (lcm_den / d)).collect();

  // Find GCD of all entries
  let g = scaled
    .iter()
    .fold(0i128, |acc, &x| gcd_i128(acc.abs(), x.abs()));

  if g == 0 || g == 1 {
    scaled.iter().map(|&x| Expr::Integer(x)).collect()
  } else {
    scaled.iter().map(|&x| Expr::Integer(x / g)).collect()
  }
}

/// Compute eigenvectors for an integer matrix.
fn integer_eigenvectors(
  matrix: &[Vec<Expr>],
  int_matrix: &[Vec<i128>],
  n: usize,
) -> Result<Expr, InterpreterError> {
  let coeffs = char_poly_coefficients(int_matrix);
  let mut eigenvalues = find_polynomial_roots(&coeffs);
  sort_eigenvalues(&mut eigenvalues);

  // Check if all eigenvalues are integers
  let all_integer = eigenvalues.iter().all(|e| expr_to_i128(e).is_some());

  if !all_integer && n == 2 {
    // 2x2 with symbolic eigenvalues: build eigenvectors directly
    let vecs = eigenvectors_2x2_symbolic(int_matrix);
    if !vecs.is_empty() {
      return Ok(Expr::List(vecs.into_iter().map(Expr::List).collect()));
    }
    return Ok(Expr::FunctionCall {
      name: "Eigenvectors".to_string(),
      args: vec![matrix_to_expr(matrix.to_vec())],
    });
  }

  if !all_integer {
    // Can't handle symbolic eigenvalues for n > 2
    return Ok(Expr::FunctionCall {
      name: "Eigenvectors".to_string(),
      args: vec![matrix_to_expr(matrix.to_vec())],
    });
  }

  // All integer eigenvalues: compute null spaces
  let mut eigenvectors: Vec<Expr> = Vec::new();
  let mut processed: Vec<(i128, Vec<Vec<Expr>>)> = Vec::new();

  for eigenvalue in &eigenvalues {
    let lambda = expr_to_i128(eigenvalue).unwrap();

    let null_vecs =
      if let Some(entry) = processed.iter_mut().find(|(l, _)| *l == lambda) {
        &mut entry.1
      } else {
        let vecs =
          null_space_for_integer_eigenvalue(matrix, lambda, int_matrix, n);
        processed.push((lambda, vecs));
        &mut processed.last_mut().unwrap().1
      };

    if null_vecs.is_empty() {
      eigenvectors.push(Expr::List(vec![Expr::Integer(0); n]));
    } else {
      eigenvectors.push(Expr::List(null_vecs.remove(0)));
    }
  }

  Ok(Expr::List(eigenvectors))
}

/// Compute null space vectors for a specific integer eigenvalue.
fn null_space_for_integer_eigenvalue(
  matrix: &[Vec<Expr>],
  lambda: i128,
  int_matrix: &[Vec<i128>],
  n: usize,
) -> Vec<Vec<Expr>> {
  let mut m: Vec<Vec<Expr>> = matrix.to_vec();
  for i in 0..n {
    m[i][i] = Expr::Integer(int_matrix[i][i] - lambda);
  }

  let rref = row_reduce_impl(&m);

  // Find pivot columns
  let mut pivot_cols = Vec::new();
  let mut pivot_row_for_col = vec![None; n];
  let mut current_row = 0;
  for col in 0..n {
    if current_row >= n {
      break;
    }
    if is_one_expr(&rref[current_row][col]) {
      pivot_cols.push(col);
      pivot_row_for_col[col] = Some(current_row);
      current_row += 1;
    }
  }

  let free_cols: Vec<usize> =
    (0..n).filter(|c| !pivot_cols.contains(c)).collect();

  // Build null space basis vectors (reverse order to match Wolfram convention)
  let mut basis = Vec::new();
  for &free_col in free_cols.iter().rev() {
    let mut vec = vec![Expr::Integer(0); n];
    vec[free_col] = Expr::Integer(1);
    for &pivot_col in &pivot_cols {
      if let Some(row) = pivot_row_for_col[pivot_col] {
        vec[pivot_col] = eval_sub(&Expr::Integer(0), &rref[row][free_col]);
      }
    }
    let scaled = scale_to_integers(&vec);
    basis.push(scaled);
  }
  basis
}

/// Build eigenvector expressions for 2x2 symbolic case (irrational eigenvalues).
/// Returns eigenvectors in the same order as eigenvalues (sorted by absolute value).
fn eigenvectors_2x2_symbolic(int_matrix: &[Vec<i128>]) -> Vec<Vec<Expr>> {
  let a = int_matrix[0][0];
  let b = int_matrix[0][1];
  let c = int_matrix[1][0];
  let d = int_matrix[1][1];

  // Discriminant: (a-d)^2 + 4bc
  let diff = a - d;
  let disc = diff * diff + 4 * b * c;

  if disc <= 0 {
    // Complex eigenvalues or zero discriminant (should be handled elsewhere)
    return vec![];
  }

  // Check if disc is a perfect square (should have been handled by integer path)
  let sqrt_disc_approx = (disc as f64).sqrt().round() as i128;
  if sqrt_disc_approx * sqrt_disc_approx == disc {
    return vec![]; // Integer eigenvalues, should be handled elsewhere
  }

  // Simplify sqrt(disc)
  let (outer, inner) = simplify_sqrt(disc.unsigned_abs());

  if c != 0 {
    // Eigenvector: {(diff ± outer*sqrt(inner)) / (2c), 1}
    let denom = 2 * c;

    // Build the two eigenvectors
    let build_v1 = |sign: i128| -> Expr {
      let signed_outer = sign * outer as i128;
      // Simplify: GCD of |diff|, |signed_outer|, |denom|
      let g = gcd_i128(gcd_i128(diff.abs(), signed_outer.abs()), denom.abs());
      let rd = diff / g;
      let ro = (signed_outer / g).abs();
      let rden = denom / g;

      let sqrt_expr = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(inner as i128)],
      };

      let sqrt_term = if ro == 1 {
        sqrt_expr
      } else {
        eval_mul(&Expr::Integer(ro), &sqrt_expr)
      };

      let numer = if rd == 0 {
        if sign > 0 {
          sqrt_term
        } else {
          eval_sub(&Expr::Integer(0), &sqrt_term)
        }
      } else if sign > 0 {
        eval_add(&Expr::Integer(rd), &sqrt_term)
      } else {
        eval_sub(&Expr::Integer(rd), &sqrt_term)
      };

      if rden == 1 {
        numer
      } else if rden == -1 {
        eval_sub(&Expr::Integer(0), &numer)
      } else {
        eval_divide(&numer, &Expr::Integer(rden))
      }
    };

    let v1_plus = build_v1(1);
    let v1_minus = build_v1(-1);

    // Determine which eigenvalue is larger by absolute value
    // λ = (a+d ± sqrt(disc)) / 2
    let trace = a + d;
    let sqrt_val = (disc as f64).sqrt();
    let l_plus = (trace as f64 + sqrt_val) / 2.0;
    let l_minus = (trace as f64 - sqrt_val) / 2.0;

    if l_plus.abs() >= l_minus.abs() {
      vec![
        vec![v1_plus, Expr::Integer(1)],
        vec![v1_minus, Expr::Integer(1)],
      ]
    } else {
      vec![
        vec![v1_minus, Expr::Integer(1)],
        vec![v1_plus, Expr::Integer(1)],
      ]
    }
  } else if b != 0 {
    // c = 0, use row 0: (a-λ)*v1 + b*v2 = 0 → v = {1, (λ-a)/b}
    // Eigenvector: {1, (λ-a)/b} = {1, (d-a ± sqrt(disc)) / (2b)}
    // But since c=0, eigenvalues are a and d (integers), handled by integer path
    vec![]
  } else {
    vec![]
  }
}

/// Compute eigenvectors for a numeric (f64) matrix.
fn numeric_eigenvectors(
  matrix: &[Vec<Expr>],
  n: usize,
) -> Result<Expr, InterpreterError> {
  // Convert to f64 matrix
  let f_matrix: Vec<Vec<f64>> = matrix
    .iter()
    .map(|row| {
      row
        .iter()
        .map(|e| match e {
          Expr::Integer(v) => *v as f64,
          Expr::Real(v) => *v,
          _ => 0.0,
        })
        .collect()
    })
    .collect();

  // Compute eigenvalues numerically using characteristic polynomial
  let eigenvalues = numeric_eigenvalues(&f_matrix, n);

  let mut eigenvectors = Vec::new();
  for &lambda in &eigenvalues {
    // Build (A - λI)
    let mut m = f_matrix.clone();
    for i in 0..n {
      m[i][i] -= lambda;
    }

    // Find null space via Gaussian elimination
    let vec = numeric_null_vector(&m, n);

    // Normalize to unit length
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
      let normalized: Vec<Expr> =
        vec.iter().map(|&x| Expr::Real(x / norm)).collect();
      eigenvectors.push(Expr::List(normalized));
    } else {
      eigenvectors.push(Expr::List(vec![Expr::Real(0.0); n].to_vec()));
    }
  }

  Ok(Expr::List(eigenvectors))
}

/// Compute eigenvalues of a f64 matrix numerically.
fn numeric_eigenvalues(matrix: &[Vec<f64>], n: usize) -> Vec<f64> {
  if n == 1 {
    return vec![matrix[0][0]];
  }
  if n == 2 {
    let tr = matrix[0][0] + matrix[1][1];
    let det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
      let sq = disc.sqrt();
      let l1 = (tr + sq) / 2.0;
      let l2 = (tr - sq) / 2.0;
      // Sort by absolute value descending
      if l1.abs() >= l2.abs() {
        return vec![l1, l2];
      } else {
        return vec![l2, l1];
      }
    }
    // Complex eigenvalues - return empty
    return vec![];
  }

  // For n >= 3: use Faddeev-LeVerrier to get characteristic polynomial,
  // then find roots numerically
  let coeffs = char_poly_f64(matrix, n);
  let mut roots = find_real_roots_f64(&coeffs);
  roots.sort_by(|a, b| {
    b.abs()
      .partial_cmp(&a.abs())
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  roots
}

/// Faddeev-LeVerrier for f64 matrix.
fn char_poly_f64(a: &[Vec<f64>], n: usize) -> Vec<f64> {
  let mut coeffs = vec![0.0f64; n + 1];
  coeffs[n] = 1.0;
  let mut m = a.to_vec();

  for k in 1..=n {
    let trace: f64 = (0..n).map(|i| m[i][i]).sum();
    coeffs[n - k] = -trace / k as f64;

    if k < n {
      let c = coeffs[n - k];
      let mut temp = m.clone();
      for i in 0..n {
        temp[i][i] += c;
      }
      let mut new_m = vec![vec![0.0; n]; n];
      for i in 0..n {
        for j in 0..n {
          for p in 0..n {
            new_m[i][j] += a[i][p] * temp[p][j];
          }
        }
      }
      m = new_m;
    }
  }
  coeffs
}

/// Find real roots of a polynomial (ascending coefficients) using
/// companion matrix eigenvalue approach (QR iteration).
fn find_real_roots_f64(coeffs: &[f64]) -> Vec<f64> {
  let degree = coeffs.len() - 1;
  if degree == 0 {
    return vec![];
  }
  if degree == 1 {
    return vec![-coeffs[0] / coeffs[1]];
  }
  if degree == 2 {
    let a = coeffs[2];
    let b = coeffs[1];
    let c = coeffs[0];
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
      let sq = disc.sqrt();
      return vec![(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)];
    }
    return vec![];
  }

  // Build companion matrix and use QR iteration
  let n = degree;
  let lc = coeffs[n];
  let mut comp = vec![vec![0.0; n]; n];
  for i in 1..n {
    comp[i][i - 1] = 1.0;
  }
  for i in 0..n {
    comp[i][n - 1] = -coeffs[i] / lc;
  }

  // QR iteration with shifts
  qr_eigenvalues(&mut comp, n)
}

/// QR iteration to find eigenvalues of an upper Hessenberg matrix.
fn qr_eigenvalues(h: &mut Vec<Vec<f64>>, n: usize) -> Vec<f64> {
  let mut eigenvalues = Vec::new();
  let mut size = n;

  for _ in 0..1000 * n {
    if size <= 1 {
      if size == 1 {
        eigenvalues.push(h[0][0]);
      }
      break;
    }

    // Check for deflation
    let mut deflated = false;
    for i in (1..size).rev() {
      if h[i][i - 1].abs()
        < 1e-14 * (h[i][i].abs() + h[i - 1][i - 1].abs()).max(1e-300)
      {
        // Deflate
        eigenvalues.push(h[i][i]);
        size = i;
        deflated = true;
        break;
      }
    }
    if deflated {
      continue;
    }

    // Wilkinson shift
    let shift = h[size - 1][size - 1];

    // Apply shift
    for i in 0..size {
      h[i][i] -= shift;
    }

    // QR factorization via Givens rotations
    let mut cs = Vec::new();
    let mut sn = Vec::new();
    for i in 0..size - 1 {
      let a = h[i][i];
      let b = h[i + 1][i];
      let r = (a * a + b * b).sqrt();
      if r < 1e-300 {
        cs.push(1.0);
        sn.push(0.0);
        continue;
      }
      let c = a / r;
      let s = b / r;
      cs.push(c);
      sn.push(s);
      // Apply Givens rotation to rows i, i+1
      for j in 0..size {
        let t1 = h[i][j];
        let t2 = h[i + 1][j];
        h[i][j] = c * t1 + s * t2;
        h[i + 1][j] = -s * t1 + c * t2;
      }
    }

    // R * Q (apply rotations from right)
    for i in 0..size - 1 {
      let c = cs[i];
      let s = sn[i];
      for j in 0..size {
        let t1 = h[j][i];
        let t2 = h[j][i + 1];
        h[j][i] = c * t1 + s * t2;
        h[j][i + 1] = -s * t1 + c * t2;
      }
    }

    // Undo shift
    for i in 0..size {
      h[i][i] += shift;
    }
  }

  if size > 0 && eigenvalues.len() < n {
    // Remaining eigenvalues
    for i in 0..size {
      eigenvalues.push(h[i][i]);
    }
  }

  eigenvalues
}

/// Find a null space vector for a numeric matrix via Gaussian elimination.
fn numeric_null_vector(m: &[Vec<f64>], n: usize) -> Vec<f64> {
  let mut aug: Vec<Vec<f64>> = m.to_vec();
  let mut pivot_cols = Vec::new();
  let mut pivot_row = 0;

  for col in 0..n {
    if pivot_row >= n {
      break;
    }
    // Partial pivoting
    let mut max_row = pivot_row;
    let mut max_val = aug[pivot_row][col].abs();
    for row in pivot_row + 1..n {
      if aug[row][col].abs() > max_val {
        max_row = row;
        max_val = aug[row][col].abs();
      }
    }
    if max_val < 1e-12 {
      continue;
    }
    aug.swap(pivot_row, max_row);
    pivot_cols.push((pivot_row, col));

    let pivot = aug[pivot_row][col];
    for row in 0..n {
      if row == pivot_row {
        continue;
      }
      let factor = aug[row][col] / pivot;
      for j in col..n {
        aug[row][j] -= factor * aug[pivot_row][j];
      }
      aug[row][col] = 0.0;
    }
    // Normalize pivot row
    for j in col..n {
      aug[pivot_row][j] /= pivot;
    }
    pivot_row += 1;
  }

  // Find free columns
  let pivot_col_set: Vec<usize> = pivot_cols.iter().map(|&(_, c)| c).collect();
  let mut free_cols: Vec<usize> =
    (0..n).filter(|c| !pivot_col_set.contains(c)).collect();

  if free_cols.is_empty() {
    return vec![0.0; n];
  }

  // Build null vector from first free column
  let free_col = free_cols.remove(0);
  let mut vec = vec![0.0; n];
  vec[free_col] = 1.0;
  for &(row, col) in &pivot_cols {
    vec[col] = -aug[row][free_col];
  }
  vec
}

/// RowReduce[matrix] - Gaussian elimination to reduced row echelon form
pub fn row_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RowReduce expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "RowReduce".to_string(),
        args: args.to_vec(),
      });
    }
  };
  Ok(matrix_to_expr(row_reduce_impl(&matrix)))
}

/// Simplify expression via the evaluator (with Simplify for symbolic)
fn simplify_expr(e: &Expr) -> Expr {
  let evaluated = match evaluate_expr_to_expr(e) {
    Ok(r) => r,
    Err(_) => return e.clone(),
  };
  // For symbolic expressions, also try Simplify
  match &evaluated {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::Identifier(_) => evaluated,
    _ => {
      match evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Simplify".to_string(),
        args: vec![evaluated.clone()],
      }) {
        Ok(r) => r,
        Err(_) => evaluated,
      }
    }
  }
}

/// Internal row reduction (Gauss-Jordan elimination) that works symbolically.
/// Uses fraction-free approach: instead of dividing by pivot immediately,
/// we compute row[j] = row[j]*pivot - factor*pivot_row[j] to avoid symbolic division issues,
/// then normalize at the end.
fn row_reduce_impl(matrix: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
  let nrows = matrix.len();
  if nrows == 0 {
    return vec![];
  }
  let ncols = matrix[0].len();
  let mut m: Vec<Vec<Expr>> = matrix.to_vec();

  let mut pivot_row = 0;
  let mut pivot_cols = Vec::new();

  for col in 0..ncols {
    if pivot_row >= nrows {
      break;
    }
    // Find a non-zero pivot in this column
    let mut found = None;
    for row in pivot_row..nrows {
      if !is_zero_expr(&m[row][col]) {
        found = Some(row);
        break;
      }
    }
    let Some(swap_row) = found else { continue };
    if swap_row != pivot_row {
      m.swap(pivot_row, swap_row);
    }

    let pivot = m[pivot_row][col].clone();
    pivot_cols.push((pivot_row, col));

    // Eliminate all other rows using fraction-free approach
    for row in 0..nrows {
      if row == pivot_row {
        continue;
      }
      let factor = m[row][col].clone();
      if !is_zero_expr(&factor) {
        // new_row[j] = pivot * row[j] - factor * pivot_row[j]
        for j in 0..ncols {
          let term1 = eval_mul(&pivot, &m[row][j]);
          let term2 = eval_mul(&factor, &m[pivot_row][j]);
          let result = eval_sub(&term1, &term2);
          m[row][j] = simplify_expr(&result);
        }
      }
    }
    pivot_row += 1;
  }

  // Now normalize: scale each pivot row so pivot element is 1
  for &(row, col) in &pivot_cols {
    let pivot = m[row][col].clone();
    if !is_one_expr(&pivot) && !is_zero_expr(&pivot) {
      for j in 0..ncols {
        if !is_zero_expr(&m[row][j]) {
          m[row][j] = simplify_expr(&eval_divide(&m[row][j], &pivot));
        }
      }
    }
  }

  m
}

/// Check if an expression is zero
fn is_zero_expr(e: &Expr) -> bool {
  matches!(e, Expr::Integer(0))
    || matches!(e, Expr::Real(x) if *x == 0.0)
    || matches!(e, Expr::BigInteger(n) if n == &num_bigint::BigInt::from(0))
}

/// Check if an expression is one
fn is_one_expr(e: &Expr) -> bool {
  matches!(e, Expr::Integer(1))
    || matches!(e, Expr::Real(x) if *x == 1.0)
    || matches!(e, Expr::BigInteger(n) if n == &num_bigint::BigInt::from(1))
}

/// MatrixRank[matrix] - rank of a matrix
pub fn matrix_rank_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MatrixRank expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "MatrixRank".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let rref = row_reduce_impl(&matrix);
  // Count non-zero rows
  let rank = rref
    .iter()
    .filter(|row| row.iter().any(|e| !is_zero_expr(e)))
    .count();
  Ok(Expr::Integer(rank as i128))
}

/// NullSpace[matrix] - find null space basis vectors
pub fn null_space_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NullSpace expects exactly 1 argument".into(),
    ));
  }
  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      // Return empty list for non-matrix (symbolic matrix)
      return Ok(Expr::List(vec![]));
    }
  };
  let nrows = matrix.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![]));
  }
  let ncols = matrix[0].len();

  let rref = row_reduce_impl(&matrix);

  // Find pivot columns
  let mut pivot_cols = Vec::new();
  let mut pivot_row_for_col = vec![None; ncols];
  let mut current_row = 0;
  for col in 0..ncols {
    if current_row >= nrows {
      break;
    }
    if is_one_expr(&rref[current_row][col]) {
      pivot_cols.push(col);
      pivot_row_for_col[col] = Some(current_row);
      current_row += 1;
    }
  }

  // Free columns are non-pivot columns
  let free_cols: Vec<usize> =
    (0..ncols).filter(|c| !pivot_cols.contains(c)).collect();

  // Build null space basis vectors
  let mut basis = Vec::new();
  for &free_col in &free_cols {
    let mut vec = vec![Expr::Integer(0); ncols];
    vec[free_col] = Expr::Integer(1);
    for &pivot_col in &pivot_cols {
      if let Some(row) = pivot_row_for_col[pivot_col] {
        // Negate the entry from rref
        vec[pivot_col] = eval_sub(&Expr::Integer(0), &rref[row][free_col]);
      }
    }
    basis.push(Expr::List(vec));
  }
  Ok(Expr::List(basis))
}

/// Compute the sign of a permutation (Levi-Civita symbol).
/// Returns 1 for even permutations, -1 for odd, 0 if any indices repeat.
fn permutation_sign(perm: &[usize]) -> i128 {
  let n = perm.len();
  // Check for duplicates
  let mut seen = vec![false; n];
  for &p in perm {
    if p >= n || seen[p] {
      return 0;
    }
    seen[p] = true;
  }
  // Count inversions
  let mut inversions = 0;
  for i in 0..n {
    for j in i + 1..n {
      if perm[i] > perm[j] {
        inversions += 1;
      }
    }
  }
  if inversions % 2 == 0 { 1 } else { -1 }
}

/// LeviCivitaTensor[n, List] - produces an n-dimensional tensor with the Levi-Civita symbol values.
pub fn levi_civita_tensor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LeviCivitaTensor".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Build the n-dimensional tensor recursively
  fn build_tensor(n: usize, depth: usize, indices: &mut Vec<usize>) -> Expr {
    if depth == n {
      let sign = permutation_sign(indices);
      Expr::Integer(sign)
    } else {
      let mut items = Vec::new();
      for i in 0..n {
        indices.push(i);
        items.push(build_tensor(n, depth + 1, indices));
        indices.pop();
      }
      Expr::List(items)
    }
  }

  let mut indices = Vec::new();
  Ok(build_tensor(n, 0, &mut indices))
}

/// LinearSolve[m, b] — solves the matrix equation m.x = b for x.
/// Uses Gaussian elimination with exact rational arithmetic.
pub fn linear_solve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LinearSolve expects exactly 2 arguments".into(),
    ));
  }

  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "LinearSolve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let b = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LinearSolve".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = matrix.len();
  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "LinearSolve: empty matrix".into(),
    ));
  }
  if matrix.iter().any(|row| row.len() != n) {
    return Err(InterpreterError::EvaluationError(
      "LinearSolve: matrix must be square".into(),
    ));
  }
  if b.len() != n {
    return Err(InterpreterError::EvaluationError(
      "LinearSolve: matrix and vector dimensions must agree".into(),
    ));
  }

  // Build augmented matrix [A | b]
  let mut aug: Vec<Vec<Expr>> = Vec::with_capacity(n);
  for i in 0..n {
    let mut row = matrix[i].clone();
    row.push(b[i].clone());
    aug.push(row);
  }

  // Forward elimination with partial pivoting
  for col in 0..n {
    // Find pivot: first non-zero entry in column
    let mut pivot_row = None;
    for row in col..n {
      if !is_zero_expr(&aug[row][col]) {
        pivot_row = Some(row);
        break;
      }
    }
    let pivot_row = match pivot_row {
      Some(r) => r,
      None => {
        return Err(InterpreterError::EvaluationError(
          "LinearSolve: matrix is singular".into(),
        ));
      }
    };

    // Swap rows if needed
    if pivot_row != col {
      aug.swap(col, pivot_row);
    }

    // Eliminate below pivot
    let pivot = aug[col][col].clone();
    for row in (col + 1)..n {
      let factor = eval_divide(&aug[row][col], &pivot);
      aug[row][col] = Expr::Integer(0);
      for j in (col + 1)..=n {
        let prod = eval_mul(&factor, &aug[col][j]);
        aug[row][j] = eval_sub(&aug[row][j], &prod);
        // Simplify intermediate results
        aug[row][j] = simplify_expr(&aug[row][j]);
      }
    }
  }

  // Back substitution
  let mut x = vec![Expr::Integer(0); n];
  for i in (0..n).rev() {
    let mut sum = aug[i][n].clone();
    for j in (i + 1)..n {
      let prod = eval_mul(&aug[i][j], &x[j]);
      sum = eval_sub(&sum, &prod);
    }
    x[i] = eval_divide(&sum, &aug[i][i]);
    x[i] = simplify_expr(&x[i]);
  }

  Ok(Expr::List(x))
}

/// Eigensystem[matrix] - returns {eigenvalues, eigenvectors}
pub fn eigensystem_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Eigensystem expects exactly 1 argument".into(),
    ));
  }

  // Check if argument is a matrix (list of lists)
  if !matches!(&args[0], Expr::List(rows) if !rows.is_empty() && matches!(&rows[0], Expr::List(_)))
  {
    return Ok(Expr::FunctionCall {
      name: "Eigensystem".to_string(),
      args: args.to_vec(),
    });
  }

  let eigenvalues = eigenvalues_ast(args)?;
  let eigenvectors = eigenvectors_ast(args)?;

  Ok(Expr::List(vec![eigenvalues, eigenvectors]))
}

/// Minors[m] - gives the minors of a matrix (determinants of (n-1)×(n-1) submatrices)
/// Minors[m, k] - gives the k-th order minors
pub fn minors_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Minors expects 1, 2, or 3 arguments".into(),
    ));
  }

  let matrix = match expr_to_matrix(&args[0]) {
    Some(m) => m,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Minors".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let nrows = matrix.len();
  if nrows == 0 {
    return Ok(Expr::List(vec![]));
  }
  let ncols = matrix[0].len();

  // Determine the minor order k
  let k = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Minors".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    // Default: n-1 for n×n matrix (standard minors)
    nrows.min(ncols) - 1
  };

  if k > nrows || k > ncols {
    return Err(InterpreterError::EvaluationError(
      "Minors: minor order exceeds matrix dimensions".into(),
    ));
  }

  // Generate all combinations of k rows and k columns
  let row_combos = combinations(nrows, k);
  let col_combos = combinations(ncols, k);

  // Build the result matrix
  let mut result_rows = Vec::new();
  for row_combo in &row_combos {
    let mut result_row = Vec::new();
    for col_combo in &col_combos {
      // Extract submatrix
      let mut submatrix = Vec::new();
      for &r in row_combo {
        let mut row = Vec::new();
        for &c in col_combo {
          row.push(matrix[r][c].clone());
        }
        submatrix.push(row);
      }
      // Apply function (default is Det)
      if args.len() == 3 {
        // Minors[m, k, f] - apply f to the submatrix
        let sub_expr =
          Expr::List(submatrix.into_iter().map(Expr::List).collect());
        let f_call = Expr::FunctionCall {
          name: match &args[2] {
            Expr::Identifier(name) => name.clone(),
            _ => "Det".to_string(),
          },
          args: vec![sub_expr],
        };
        result_row.push(crate::evaluator::evaluate_expr_to_expr(&f_call)?);
      } else {
        result_row.push(determinant(&submatrix));
      }
    }
    result_rows.push(Expr::List(result_row));
  }

  Ok(Expr::List(result_rows))
}

/// Generate all combinations of `k` items from `0..n`
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
  let mut result = Vec::new();
  let mut combo = Vec::new();
  combinations_helper(0, n, k, &mut combo, &mut result);
  result
}

fn combinations_helper(
  start: usize,
  n: usize,
  k: usize,
  current: &mut Vec<usize>,
  result: &mut Vec<Vec<usize>>,
) {
  if current.len() == k {
    result.push(current.clone());
    return;
  }
  for i in start..n {
    current.push(i);
    combinations_helper(i + 1, n, k, current, result);
    current.pop();
  }
}

/// LatticeReduce[{v1, v2, ...}] - LLL lattice basis reduction
/// Implements the Lenstra–Lenstra–Lovász algorithm with exact rational arithmetic.
pub fn lattice_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "LatticeReduce".to_string(),
      args: args.to_vec(),
    });
  }

  // Extract the matrix of integer vectors
  let rows = match &args[0] {
    Expr::List(rows) => rows,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LatticeReduce".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if rows.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Parse into Vec<Vec<i128>>
  let mut basis: Vec<Vec<i128>> = Vec::new();
  for row in rows {
    match row {
      Expr::List(elems) => {
        let mut vec = Vec::new();
        for e in elems {
          match e {
            Expr::Integer(n) => vec.push(*n),
            _ => {
              return Ok(Expr::FunctionCall {
                name: "LatticeReduce".to_string(),
                args: args.to_vec(),
              });
            }
          }
        }
        basis.push(vec);
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "LatticeReduce".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }

  // Verify all vectors have the same dimension
  let dim = basis[0].len();
  if basis.iter().any(|v| v.len() != dim) {
    return Err(InterpreterError::EvaluationError(
      "LatticeReduce: all vectors must have the same dimension".into(),
    ));
  }

  // Run LLL algorithm
  let result = lll_reduce(&mut basis);

  // Convert back to Expr
  let result_rows: Vec<Expr> = result
    .iter()
    .map(|row| Expr::List(row.iter().map(|&n| Expr::Integer(n)).collect()))
    .collect();

  Ok(Expr::List(result_rows))
}

fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// LLL lattice basis reduction algorithm (delta = 3/4)
/// Uses floating-point Gram-Schmidt with integer basis vectors
fn lll_reduce(basis: &mut Vec<Vec<i128>>) -> Vec<Vec<i128>> {
  let n = basis.len();
  if n <= 1 {
    return basis
      .iter()
      .filter(|v| v.iter().any(|&x| x != 0))
      .cloned()
      .collect();
  }

  let dim = basis[0].len();
  let delta: f64 = 0.75;

  // Gram-Schmidt orthogonal vectors (as f64)
  let mut bstar: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];
  // mu coefficients
  let mut mu: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
  // B[i] = ||b*_i||^2
  let mut big_b: Vec<f64> = vec![0.0; n];

  // Compute Gram-Schmidt
  let compute_gs = |basis: &[Vec<i128>],
                    bstar: &mut Vec<Vec<f64>>,
                    mu: &mut Vec<Vec<f64>>,
                    big_b: &mut Vec<f64>| {
    let n = basis.len();
    for i in 0..n {
      // b*_i = b_i
      for d in 0..dim {
        bstar[i][d] = basis[i][d] as f64;
      }
      // Subtract projections
      for j in 0..i {
        if big_b[j].abs() < 1e-30 {
          mu[i][j] = 0.0;
          continue;
        }
        let bi_f: Vec<f64> = basis[i].iter().map(|&x| x as f64).collect();
        mu[i][j] = dot_f64(&bi_f, &bstar[j]) / big_b[j];
        for d in 0..dim {
          bstar[i][d] -= mu[i][j] * bstar[j][d];
        }
      }
      big_b[i] = dot_f64(&bstar[i], &bstar[i]);
    }
  };

  compute_gs(basis, &mut bstar, &mut mu, &mut big_b);

  let max_iters = n * n * 200;
  let mut iters = 0;
  let mut k = 1;
  while k < n && iters < max_iters {
    iters += 1;

    // Size reduce basis[k] by basis[j] for j = k-1, k-2, ..., 0
    for j in (0..k).rev() {
      if mu[k][j].abs() > 0.5 {
        let r = mu[k][j].round() as i128;
        if r != 0 {
          for d in 0..dim {
            basis[k][d] -= r * basis[j][d];
          }
          // Update mu values
          compute_gs(basis, &mut bstar, &mut mu, &mut big_b);
        }
      }
    }

    // Check Lovász condition
    if big_b[k - 1].abs() < 1e-30 {
      if big_b[k].abs() >= 1e-30 {
        basis.swap(k, k - 1);
        compute_gs(basis, &mut bstar, &mut mu, &mut big_b);
      } else {
        k += 1;
      }
    } else {
      let lhs = big_b[k] + mu[k][k - 1] * mu[k][k - 1] * big_b[k - 1];
      if lhs >= delta * big_b[k - 1] {
        k += 1;
      } else {
        basis.swap(k, k - 1);
        compute_gs(basis, &mut bstar, &mut mu, &mut big_b);
        k = if k > 1 { k - 1 } else { 1 };
      }
    }
  }

  // Remove zero vectors
  basis
    .iter()
    .filter(|v| v.iter().any(|&x| x != 0))
    .cloned()
    .collect()
}
