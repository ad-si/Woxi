#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::functions::math_ast::expr_to_i128;
use crate::syntax::{BinaryOperator, Expr};

/// Resultant[poly1, poly2, var] - Computes the resultant of two polynomials
/// with respect to the given variable.
///
/// The resultant is the determinant of the Sylvester matrix built from
/// the coefficients of the two polynomials.
pub fn resultant_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Resultant expects exactly 3 arguments".into(),
    ));
  }

  let poly1 = &args[0];
  let poly2 = &args[1];
  let var = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Third argument of Resultant must be a symbol".into(),
      ));
    }
  };

  // Get degrees
  let deg1 = match max_power_int(poly1, &var) {
    Some(d) => d,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Resultant".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let deg2 = match max_power_int(poly2, &var) {
    Some(d) => d,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Resultant".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let m = deg1 as usize;
  let n = deg2 as usize;

  if m == 0 && n == 0 {
    // Both constants: resultant is 1 (for nonzero constants)
    return Ok(Expr::Integer(1));
  }

  // Extract coefficients symbolically
  // coeff1[i] = coefficient of x^i in poly1, for i = 0..m
  // coeff2[i] = coefficient of x^i in poly2, for i = 0..n
  let var_expr = Expr::Identifier(var.clone());
  let mut coeff1 = Vec::with_capacity(m + 1);
  let mut coeff2 = Vec::with_capacity(n + 1);

  for i in 0..=m {
    let c = coefficient_ast(&[
      poly1.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    let c = crate::evaluator::evaluate_expr_to_expr(&c)?;
    coeff1.push(c);
  }
  for i in 0..=n {
    let c = coefficient_ast(&[
      poly2.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    let c = crate::evaluator::evaluate_expr_to_expr(&c)?;
    coeff2.push(c);
  }

  // Try integer path first for efficiency
  let int_coeffs1: Option<Vec<i128>> =
    coeff1.iter().map(expr_to_i128).collect();
  let int_coeffs2: Option<Vec<i128>> =
    coeff2.iter().map(expr_to_i128).collect();

  if let (Some(ic1), Some(ic2)) = (int_coeffs1, int_coeffs2) {
    // Integer Sylvester matrix determinant
    let size = m + n;
    if size == 0 {
      return Ok(Expr::Integer(1));
    }
    let det = sylvester_det_integer(&ic1, m, &ic2, n);
    return Ok(Expr::Integer(det));
  }

  // Symbolic path: build Sylvester matrix and compute determinant
  let size = m + n;
  if size == 0 {
    return Ok(Expr::Integer(1));
  }

  let mut matrix: Vec<Vec<Expr>> = Vec::with_capacity(size);

  // First n rows: coefficients of poly1, reversed (leading coeff first)
  for i in 0..n {
    let mut row = vec![Expr::Integer(0); size];
    for j in 0..=m {
      let col = i + m - j;
      row[col] = coeff1[j].clone();
    }
    matrix.push(row);
  }

  // Last m rows: coefficients of poly2, reversed (leading coeff first)
  for i in 0..m {
    let mut row = vec![Expr::Integer(0); size];
    for j in 0..=n {
      let col = i + n - j;
      row[col] = coeff2[j].clone();
    }
    matrix.push(row);
  }

  // Compute symbolic determinant
  let det = symbolic_determinant(&matrix);

  // Simplify the result
  let simplified = crate::evaluator::evaluate_expr_to_expr(&det)?;
  Ok(simplified)
}

/// Subresultants[poly1, poly2, var] - The principal subresultant
/// coefficients s_j for j = 0..Min[deg1, deg2]; s_0 is the resultant.
///
/// s_j is the determinant of the leading (m+n-2j) square block of the
/// matrix whose rows are x^(n-j-1)*p1 ... p1, x^(m-j-1)*p2 ... p2 over
/// the columns for degrees m+n-j-1 down to j. This single formula
/// reproduces wolframscript for every degree combination (m < n needs
/// no swap or sign factor).
pub fn subresultants_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "Subresultants".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 3 {
    return Ok(unevaluated(args));
  }
  let poly1 = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let poly2 = crate::evaluator::evaluate_expr_to_expr(&args[1])?;
  let var = match &args[2] {
    Expr::Identifier(name) => name.clone(),
    _ => return Ok(unevaluated(args)),
  };

  // A zero polynomial has no subresultant chain
  if matches!(poly1, Expr::Integer(0)) || matches!(poly2, Expr::Integer(0)) {
    return Ok(Expr::List(vec![].into()));
  }

  let m = match max_power_int(&poly1, &var) {
    Some(d) => d as usize,
    None => return Ok(unevaluated(args)),
  };
  let n = match max_power_int(&poly2, &var) {
    Some(d) => d as usize,
    None => return Ok(unevaluated(args)),
  };

  let var_expr = Expr::Identifier(var.clone());
  let mut coeff1 = Vec::with_capacity(m + 1);
  let mut coeff2 = Vec::with_capacity(n + 1);
  for i in 0..=m {
    let c = coefficient_ast(&[
      poly1.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    coeff1.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }
  for i in 0..=n {
    let c = coefficient_ast(&[
      poly2.clone(),
      var_expr.clone(),
      Expr::Integer(i as i128),
    ])?;
    coeff2.push(crate::evaluator::evaluate_expr_to_expr(&c)?);
  }

  let int_coeffs1: Option<Vec<i128>> =
    coeff1.iter().map(expr_to_i128).collect();
  let int_coeffs2: Option<Vec<i128>> =
    coeff2.iter().map(expr_to_i128).collect();
  let integer_path = int_coeffs1.is_some() && int_coeffs2.is_some();

  let d = m.min(n);
  let mut result = Vec::with_capacity(d + 1);
  for j in 0..=d {
    let size = m + n - 2 * j;
    if size == 0 {
      result.push(Expr::Integer(1));
      continue;
    }
    // Row for x^s * p: entry of degree deg is coeff[deg - s]; the kept
    // columns cover degrees m+n-j-1 down to j (col = m+n-j-1 - deg).
    let top_degree = m + n - j - 1;
    let fill_rows =
      |matrix: &mut Vec<Vec<Expr>>, coeffs: &[Expr], shifts: usize| {
        for i in 0..shifts {
          let s = shifts - 1 - i;
          let mut row = vec![Expr::Integer(0); size];
          for (k, c) in coeffs.iter().enumerate() {
            let deg = k + s;
            if deg >= j && deg <= top_degree {
              row[top_degree - deg] = c.clone();
            }
          }
          matrix.push(row);
        }
      };
    let mut matrix: Vec<Vec<Expr>> = Vec::with_capacity(size);
    fill_rows(&mut matrix, &coeff1, n - j);
    fill_rows(&mut matrix, &coeff2, m - j);

    if integer_path {
      let mut int_matrix: Vec<Vec<i128>> = matrix
        .iter()
        .map(|row| row.iter().map(|e| expr_to_i128(e).unwrap()).collect())
        .collect();
      result.push(Expr::Integer(bareiss_determinant(&mut int_matrix)));
    } else {
      let det = symbolic_determinant(&matrix);
      result.push(crate::evaluator::evaluate_expr_to_expr(&det)?);
    }
  }
  Ok(Expr::List(result.into()))
}

/// Compute the determinant of the Sylvester matrix for integer coefficients
/// using Gaussian elimination over integers (fraction-free).
fn sylvester_det_integer(
  coeffs1: &[i128],
  deg1: usize,
  coeffs2: &[i128],
  deg2: usize,
) -> i128 {
  let size = deg1 + deg2;
  if size == 0 {
    return 1;
  }

  // Build Sylvester matrix
  let mut matrix = vec![vec![0i128; size]; size];

  // First deg2 rows: coefficients of poly1 (from highest to lowest degree)
  for i in 0..deg2 {
    for j in 0..=deg1 {
      let col = i + deg1 - j;
      matrix[i][col] = coeffs1[j];
    }
  }

  // Last deg1 rows: coefficients of poly2
  for i in 0..deg1 {
    for j in 0..=deg2 {
      let col = i + deg2 - j;
      matrix[deg2 + i][col] = coeffs2[j];
    }
  }

  // Bareiss algorithm (fraction-free Gaussian elimination)
  bareiss_determinant(&mut matrix)
}

/// Bareiss algorithm for exact integer determinant computation
fn bareiss_determinant(matrix: &mut Vec<Vec<i128>>) -> i128 {
  let n = matrix.len();
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return matrix[0][0];
  }

  let mut sign = 1i128;
  let mut prev_pivot = 1i128;

  for k in 0..n {
    // Find pivot
    let mut pivot_row = None;
    for i in k..n {
      if matrix[i][k] != 0 {
        pivot_row = Some(i);
        break;
      }
    }
    let pivot_row = match pivot_row {
      Some(r) => r,
      None => return 0, // singular matrix
    };

    if pivot_row != k {
      matrix.swap(k, pivot_row);
      sign = -sign;
    }

    let pivot = matrix[k][k];

    if k + 1 < n {
      for i in (k + 1)..n {
        for j in (k + 1)..n {
          matrix[i][j] =
            (matrix[i][j] * pivot - matrix[i][k] * matrix[k][j]) / prev_pivot;
        }
      }
    }

    prev_pivot = pivot;
  }

  sign * matrix[n - 1][n - 1]
}

/// Compute symbolic determinant using cofactor expansion
fn symbolic_determinant(matrix: &[Vec<Expr>]) -> Expr {
  let n = matrix.len();
  if n == 0 {
    return Expr::Integer(1);
  }
  if n == 1 {
    return matrix[0][0].clone();
  }
  if n == 2 {
    return sym_sub(
      &sym_mul(&matrix[0][0], &matrix[1][1]),
      &sym_mul(&matrix[0][1], &matrix[1][0]),
    );
  }

  let mut det = Expr::Integer(0);
  for j in 0..n {
    // Skip zero entries to avoid unnecessary computation
    if matches!(&matrix[0][j], Expr::Integer(0)) {
      continue;
    }
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
    let cofactor = sym_mul(&matrix[0][j], &symbolic_determinant(&minor));
    if j % 2 == 0 {
      det = sym_add(&det, &cofactor);
    } else {
      det = sym_sub(&det, &cofactor);
    }
  }
  det
}

fn sym_add(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(0), _) => b.clone(),
    (_, Expr::Integer(0)) => a.clone(),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x + y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

fn sym_sub(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (_, Expr::Integer(0)) => a.clone(),
    (Expr::Integer(0), _) => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(b.clone()),
    },
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x - y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}

fn sym_mul(a: &Expr, b: &Expr) -> Expr {
  match (a, b) {
    (Expr::Integer(0), _) | (_, Expr::Integer(0)) => Expr::Integer(0),
    (Expr::Integer(1), _) => b.clone(),
    (_, Expr::Integer(1)) => a.clone(),
    (Expr::Integer(x), Expr::Integer(y)) => Expr::Integer(x * y),
    _ => Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
    },
  }
}
