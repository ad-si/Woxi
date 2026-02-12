//! AST-native linear algebra functions.
//!
//! Dot, Det, Inverse, Tr, IdentityMatrix, DiagonalMatrix, Cross, Eigenvalues.

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

/// Convert an integer eigenvalue to Expr, sorting value for ordering.
fn i128_to_eigenvalue_expr(v: i128) -> Expr {
  Expr::Integer(v)
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
