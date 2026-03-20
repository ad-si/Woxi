#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::functions::math_ast::{expr_to_f64, expr_to_i128};
use crate::syntax::{BinaryOperator, Expr};

/// MinimalPolynomial[α, x] - Computes the minimal polynomial of an algebraic number α
/// in the variable x.
pub fn minimal_polynomial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MinimalPolynomial expects exactly 2 arguments".into(),
    ));
  }

  let alpha = &args[0];
  let var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of MinimalPolynomial must be a symbol".into(),
      ));
    }
  };

  // Try to compute the minimal polynomial coefficients
  match compute_minpoly_coeffs(alpha)? {
    Some(coeffs) => {
      let poly = coeffs_to_expr(&coeffs, &var);
      let simplified = crate::evaluator::evaluate_expr_to_expr(&poly)?;
      Ok(simplified)
    }
    None => {
      // Return unevaluated
      Ok(Expr::FunctionCall {
        name: "MinimalPolynomial".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute the minimal polynomial coefficients [c0, c1, ..., cn] where
/// c0 + c1*x + c2*x^2 + ... + cn*x^n = 0 and the polynomial is irreducible.
/// Returns coefficients in ascending degree order.
fn compute_minpoly_coeffs(
  expr: &Expr,
) -> Result<Option<Vec<i128>>, InterpreterError> {
  match expr {
    // Integer n → x - n
    Expr::Integer(n) => Ok(Some(vec![-*n, 1])),

    // Rational p/q → q*x - p
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
        Ok(Some(vec![-*p, *q]))
      } else {
        Ok(None)
      }
    }

    // Known constants
    Expr::Identifier(name) => match name.as_str() {
      "GoldenRatio" => Ok(Some(vec![-1, -1, 1])), // x^2 - x - 1
      "I" => Ok(Some(vec![1, 0, 1])),             // x^2 + 1
      _ => Ok(None),
    },

    // Complex I → x^2 + 1
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      if let (Expr::Integer(re), Expr::Integer(im)) = (&args[0], &args[1]) {
        if *re == 0 && *im == 1 {
          // I → x^2 + 1
          return Ok(Some(vec![1, 0, 1]));
        }
        if *re == 0 && *im == -1 {
          // -I → x^2 + 1
          return Ok(Some(vec![1, 0, 1]));
        }
        // a + b*I: minpoly of a+b*I over Q is (x-a)^2 + b^2 = x^2 - 2a*x + a^2 + b^2
        return Ok(Some(vec![re * re + im * im, -2 * re, 1]));
      }
      // General Complex[a, b] where a,b might be algebraic
      handle_complex_algebraic(&args[0], &args[1])
    }

    // Power expressions
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => handle_power(left, right),

    // Times expressions (products)
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      ..
    } => handle_times(expr),

    // Plus expressions (sums)
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let pa = compute_minpoly_coeffs(left)?;
      let pb = compute_minpoly_coeffs(right)?;
      match (pa, pb) {
        (Some(a), Some(b)) => {
          let numeric_val = expr_to_f64(expr);
          Ok(minpoly_of_sum(&a, &b, numeric_val))
        }
        _ => Ok(None),
      }
    }

    Expr::FunctionCall { name, args } if name == "Plus" => handle_plus(args),

    // Sqrt[n] → Power[n, 1/2]
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      let power_expr = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(args[0].clone()),
        right: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(2)],
        }),
      };
      compute_minpoly_coeffs(&power_expr)
    }

    // Unary minus
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      // minpoly(-α) = minpoly(α) with x replaced by -x
      let coeffs = compute_minpoly_coeffs(operand)?;
      Ok(coeffs.map(|c| negate_var_in_poly(&c)))
    }

    // FunctionCall Times/Power that weren't caught by BinaryOp
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      handle_power(&args[0], &args[1])
    }

    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      handle_times(expr)
    }

    // BinaryOp Minus
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      // a - b = a + (-b)
      let neg_right = Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new((**right).clone()),
      };
      let sum_expr = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new((**left).clone()),
        right: Box::new(neg_right),
      };
      compute_minpoly_coeffs(&sum_expr)
    }

    _ => Ok(None),
  }
}

/// Handle Power[base, exp] for minimal polynomial computation
fn handle_power(
  base: &Expr,
  exp: &Expr,
) -> Result<Option<Vec<i128>>, InterpreterError> {
  // Case: a^(p/q) where a is rational
  if let Some(a_val) = extract_rational(base)
    && let Some((p, q)) = extract_rational_pair(exp)
  {
    // α = (a_num/a_den)^(p/q)
    // α^q = (a_num/a_den)^p
    let (a_num, a_den) = a_val;
    if q > 0 && q <= 20 && p.abs() <= 20 {
      let q_us = q as u32;
      let p_abs = p.unsigned_abs() as u32;
      if p > 0 {
        // α^q = a_num^p / a_den^p → a_den^p * x^q - a_num^p = 0
        let num_pow = a_num.pow(p_abs);
        let den_pow = a_den.pow(p_abs);
        let mut coeffs = vec![0i128; q_us as usize + 1];
        coeffs[0] = -num_pow;
        coeffs[q_us as usize] = den_pow;
        return Ok(make_primitive_and_irreducible(coeffs));
      } else {
        // α = (a_num/a_den)^(-|p|/q) = (a_den/a_num)^(|p|/q)
        let num_pow = a_den.pow(p_abs);
        let den_pow = a_num.pow(p_abs);
        let mut coeffs = vec![0i128; q_us as usize + 1];
        coeffs[0] = -num_pow;
        coeffs[q_us as usize] = den_pow;
        return Ok(make_primitive_and_irreducible(coeffs));
      }
    }
  }

  // Case: algebraic^integer
  if let Some(n) = expr_to_i128(exp)
    && (2..=10).contains(&n)
  {
    let base_poly = compute_minpoly_coeffs(base)?;
    if let Some(bp) = base_poly {
      // minpoly(α^n): substitute x^(1/n) for x in p(x), then compute resultant
      // Actually: if p(α) = 0, then q(β) = Res_y(p(y), y^n - x) evaluated properly
      // Simpler: compose p with x^n replacement
      // If minpoly of α is p(x), then α^n is a root of p(x^(1/n))...
      // Better: use resultant approach
      // Res_t(p(t), t^n - x) gives a polynomial in x whose roots include α^n
      let numeric_val =
        expr_to_f64(exp).and_then(|ne| expr_to_f64(base).map(|be| be.powf(ne)));
      return Ok(minpoly_of_power(&bp, n, numeric_val));
    }
  }

  Ok(None)
}

/// Handle Times expressions
fn handle_times(expr: &Expr) -> Result<Option<Vec<i128>>, InterpreterError> {
  let factors = collect_times_factors(expr);
  if factors.len() < 2 {
    return Ok(None);
  }

  // Start with first factor
  let mut result = match compute_minpoly_coeffs(&factors[0])? {
    Some(c) => c,
    None => return Ok(None),
  };

  let mut numeric_product = expr_to_f64(&factors[0]);

  for factor in &factors[1..] {
    let fc = match compute_minpoly_coeffs(factor)? {
      Some(c) => c,
      None => return Ok(None),
    };
    let factor_val = expr_to_f64(factor);
    numeric_product = numeric_product.and_then(|a| factor_val.map(|b| a * b));
    result = match minpoly_of_product(&result, &fc, numeric_product) {
      Some(r) => r,
      None => return Ok(None),
    };
  }

  Ok(Some(result))
}

/// Handle Plus[args...] for minimal polynomial
fn handle_plus(args: &[Expr]) -> Result<Option<Vec<i128>>, InterpreterError> {
  if args.len() < 2 {
    if args.len() == 1 {
      return compute_minpoly_coeffs(&args[0]);
    }
    return Ok(None);
  }

  let mut result = match compute_minpoly_coeffs(&args[0])? {
    Some(c) => c,
    None => return Ok(None),
  };

  let mut numeric_sum = expr_to_f64(&args[0]);

  for arg in &args[1..] {
    let ac = match compute_minpoly_coeffs(arg)? {
      Some(c) => c,
      None => return Ok(None),
    };
    let arg_val = expr_to_f64(arg);
    numeric_sum = numeric_sum.and_then(|a| arg_val.map(|b| a + b));
    result = match minpoly_of_sum(&result, &ac, numeric_sum) {
      Some(r) => r,
      None => return Ok(None),
    };
  }

  Ok(Some(result))
}

/// Handle Complex[a, b] where a or b might be algebraic
fn handle_complex_algebraic(
  re: &Expr,
  im: &Expr,
) -> Result<Option<Vec<i128>>, InterpreterError> {
  // a + b*I: first get minpoly of a and b*I separately, then combine
  let re_poly = compute_minpoly_coeffs(re)?;

  // b*I has minpoly: if b is rational p/q, then (x/b)^2 + 1 = 0 → q^2*x^2 + p^2 = 0... no
  // Actually b*I: (α/(b))^2 = -1 → α^2 = -b^2 → α^2 + b^2 = 0
  // If b = p/q: q^2*x^2 + p^2 = 0
  if let (Some(re_c), Some((p, q))) = (re_poly, extract_rational(im)) {
    // minpoly of b*I is q^2*x^2 + p^2
    let im_poly = vec![p * p, 0, q * q];
    let numeric_val =
      expr_to_f64(re).and_then(|r| expr_to_f64(im).map(|i| r + i)); // approximate
    return Ok(minpoly_of_sum(&re_c, &im_poly, numeric_val));
  }

  Ok(None)
}

/// Compute minimal polynomial of α + β given minpolys of α and β.
/// Uses resultant: Res_y(p(x - y), q(y))
fn minpoly_of_sum(
  pa: &[i128],
  pb: &[i128],
  numeric_val: Option<f64>,
) -> Option<Vec<i128>> {
  let deg_a = pa.len() - 1;
  let deg_b = pb.len() - 1;
  let result_deg = deg_a * deg_b;

  if result_deg > 30 {
    return None; // Too large
  }

  // Build p(x - y) as a polynomial in y with coefficients that are polynomials in x
  // p(x - y) = sum_{k=0}^{deg_a} pa[k] * (x - y)^k
  // We need the Sylvester resultant Res_y(p(x-y), q(y))

  // Represent p(x-y) and q(y) as polynomials in y with i128 polynomial-in-x coefficients
  // p(x-y) coefficients in y: coeff of y^j in p(x-y)
  // (x-y)^k = sum_{j=0}^{k} C(k,j) * x^(k-j) * (-y)^j = sum_j C(k,j)*(-1)^j * x^(k-j) * y^j

  // Use Sylvester matrix approach with polynomial coefficients
  // Actually, let's use the integer resultant approach:
  // Build Sylvester matrix where entries are polynomials in x (as coefficient vectors)

  // Coefficients of p(x-y) as polynomial in y:
  // coeff_y_j = sum_{k>=j} pa[k] * C(k,j) * (-1)^j * x^(k-j)
  let mut p_xy_coeffs: Vec<Vec<i128>> = vec![vec![]; deg_a + 1]; // p_xy_coeffs[j] = polynomial in x for coeff of y^j

  for k in 0..=deg_a {
    for j in 0..=k {
      let binom = binomial(k as u64, j as u64) as i128;
      let sign = if j % 2 == 0 { 1i128 } else { -1i128 };
      let x_power = k - j;
      // Add pa[k] * binom * sign to coefficient of x^(x_power) in p_xy_coeffs[j]
      while p_xy_coeffs[j].len() <= x_power {
        p_xy_coeffs[j].push(0);
      }
      p_xy_coeffs[j][x_power] += pa[k] * binom * sign;
    }
  }

  // q(y) coefficients: q_coeffs[j] = pb[j] (constant polynomials in x)
  let q_coeffs: Vec<Vec<i128>> = pb.iter().map(|&c| vec![c]).collect();

  // Build Sylvester matrix and compute determinant
  // Matrix is (deg_a + deg_b) × (deg_a + deg_b)
  // First deg_b rows: shifted copies of p_xy_coeffs
  // Last deg_a rows: shifted copies of q_coeffs
  let size = deg_a + deg_b;
  if size == 0 {
    return Some(vec![1]);
  }

  let mut matrix: Vec<Vec<Vec<i128>>> = vec![vec![vec![0]; size]; size];

  // First deg_b rows: p(x-y) coefficients
  for i in 0..deg_b {
    for j in 0..=deg_a {
      let col = i + deg_a - j;
      if col < size {
        matrix[i][col] = p_xy_coeffs[j].clone();
      }
    }
  }

  // Last deg_a rows: q(y) coefficients
  for i in 0..deg_a {
    for j in 0..=deg_b {
      let col = i + deg_b - j;
      if col < size {
        matrix[deg_b + i][col] = q_coeffs[j].clone();
      }
    }
  }

  // Compute determinant of this polynomial matrix using expansion
  let det = poly_matrix_determinant(&matrix);
  let det = det?;

  // Make monic and primitive
  let result = make_primitive_monic(&det);

  // If the result is reducible, find the factor that vanishes at numeric_val
  if let Some(val) = numeric_val {
    return pick_irreducible_factor(&result, val);
  }

  Some(result)
}

/// Compute minimal polynomial of α * β given minpolys of α and β.
/// Uses resultant: Res_y(y^n * p(x/y), q(y))
fn minpoly_of_product(
  pa: &[i128],
  pb: &[i128],
  numeric_val: Option<f64>,
) -> Option<Vec<i128>> {
  let deg_a = pa.len() - 1;
  let deg_b = pb.len() - 1;
  let result_deg = deg_a * deg_b;

  if result_deg > 30 {
    return None;
  }

  // y^deg_a * p(x/y) = sum_{k=0}^{deg_a} pa[k] * x^k * y^(deg_a - k)
  // As polynomial in y: coeff of y^j = pa[deg_a - j] * x^(deg_a - j) for j=0..deg_a
  let mut p_xy_coeffs: Vec<Vec<i128>> = vec![vec![]; deg_a + 1];
  for j in 0..=deg_a {
    let k = deg_a - j;
    let mut poly_x = vec![0i128; k + 1];
    poly_x[k] = pa[k];
    p_xy_coeffs[j] = poly_x;
  }

  let q_coeffs: Vec<Vec<i128>> = pb.iter().map(|&c| vec![c]).collect();

  let size = deg_a + deg_b;
  if size == 0 {
    return Some(vec![1]);
  }

  let mut matrix: Vec<Vec<Vec<i128>>> = vec![vec![vec![0]; size]; size];

  for i in 0..deg_b {
    for j in 0..=deg_a {
      let col = i + deg_a - j;
      if col < size {
        matrix[i][col] = p_xy_coeffs[j].clone();
      }
    }
  }

  for i in 0..deg_a {
    for j in 0..=deg_b {
      let col = i + deg_b - j;
      if col < size {
        matrix[deg_b + i][col] = q_coeffs[j].clone();
      }
    }
  }

  let det = poly_matrix_determinant(&matrix)?;
  let result = make_primitive_monic(&det);

  if let Some(val) = numeric_val {
    return pick_irreducible_factor(&result, val);
  }

  Some(result)
}

/// Compute minimal polynomial of α^n given minpoly of α.
fn minpoly_of_power(
  pa: &[i128],
  n: i128,
  numeric_val: Option<f64>,
) -> Option<Vec<i128>> {
  // α^n: use Res_y(p(y), y^n - x) where p is minpoly of α
  // This gives polynomial in x whose roots include α^n
  let deg_a = pa.len() - 1;
  let n_us = n as usize;
  let result_deg = deg_a * n_us;

  if result_deg > 30 {
    return None;
  }

  // p(y) as poly in y: coefficients pa[j] (constants in x)
  let p_coeffs: Vec<Vec<i128>> = pa.iter().map(|&c| vec![c]).collect();

  // y^n - x as poly in y: coeff of y^n = 1, coeff of y^0 = -x
  let mut q_coeffs: Vec<Vec<i128>> = vec![vec![0]; n_us + 1];
  q_coeffs[0] = vec![0, -1]; // -x
  q_coeffs[n_us] = vec![1]; // 1

  let size = deg_a + n_us;
  if size == 0 {
    return Some(vec![1]);
  }

  // Sylvester matrix
  let mut matrix: Vec<Vec<Vec<i128>>> = vec![vec![vec![0]; size]; size];

  // First n rows: p(y) coefficients
  for i in 0..n_us {
    for j in 0..=deg_a {
      let col = i + deg_a - j;
      if col < size {
        matrix[i][col] = p_coeffs[j].clone();
      }
    }
  }

  // Last deg_a rows: q(y) = y^n - x
  for i in 0..deg_a {
    for j in 0..=n_us {
      let col = i + n_us - j;
      if col < size {
        matrix[n_us + i][col] = q_coeffs[j].clone();
      }
    }
  }

  let det = poly_matrix_determinant(&matrix)?;
  let result = make_primitive_monic(&det);

  if let Some(val) = numeric_val {
    return pick_irreducible_factor(&result, val);
  }

  Some(result)
}

/// Compute determinant of a matrix whose entries are polynomials (Vec<i128>).
/// Uses cofactor expansion for small matrices.
fn poly_matrix_determinant(matrix: &[Vec<Vec<i128>>]) -> Option<Vec<i128>> {
  let n = matrix.len();
  if n == 0 {
    return Some(vec![1]);
  }
  if n == 1 {
    return Some(matrix[0][0].clone());
  }
  if n == 2 {
    let a = poly_mul_i128(&matrix[0][0], &matrix[1][1]);
    let b = poly_mul_i128(&matrix[0][1], &matrix[1][0]);
    return Some(poly_sub_i128(&a, &b));
  }

  // For larger matrices, use Bareiss-like elimination with polynomial entries
  // or cofactor expansion along first row
  if n > 8 {
    return None; // Too expensive
  }

  // Cofactor expansion along first row
  let mut det = vec![0i128];
  for j in 0..n {
    if matrix[0][j].iter().all(|&c| c == 0) {
      continue;
    }
    // Build (n-1)×(n-1) minor
    let mut minor: Vec<Vec<Vec<i128>>> = Vec::new();
    for i in 1..n {
      let mut row = Vec::new();
      for k in 0..n {
        if k != j {
          row.push(matrix[i][k].clone());
        }
      }
      minor.push(row);
    }
    let minor_det = poly_matrix_determinant(&minor)?;
    let term = poly_mul_i128(&matrix[0][j], &minor_det);
    if j % 2 == 0 {
      det = poly_add_i128(&det, &term);
    } else {
      det = poly_sub_i128(&det, &term);
    }
  }
  Some(det)
}

/// Polynomial multiplication for i128 coefficients
fn poly_mul_i128(a: &[i128], b: &[i128]) -> Vec<i128> {
  if a.is_empty() || b.is_empty() {
    return vec![0];
  }
  if a.iter().all(|&c| c == 0) || b.iter().all(|&c| c == 0) {
    return vec![0];
  }
  let mut result = vec![0i128; a.len() + b.len() - 1];
  for (i, &ai) in a.iter().enumerate() {
    if ai == 0 {
      continue;
    }
    for (j, &bj) in b.iter().enumerate() {
      result[i + j] += ai * bj;
    }
  }
  result
}

/// Polynomial addition
fn poly_add_i128(a: &[i128], b: &[i128]) -> Vec<i128> {
  let len = a.len().max(b.len());
  let mut result = vec![0i128; len];
  for (i, &v) in a.iter().enumerate() {
    result[i] += v;
  }
  for (i, &v) in b.iter().enumerate() {
    result[i] += v;
  }
  // Trim trailing zeros
  while result.len() > 1 && result.last() == Some(&0) {
    result.pop();
  }
  result
}

/// Polynomial subtraction
fn poly_sub_i128(a: &[i128], b: &[i128]) -> Vec<i128> {
  let len = a.len().max(b.len());
  let mut result = vec![0i128; len];
  for (i, &v) in a.iter().enumerate() {
    result[i] += v;
  }
  for (i, &v) in b.iter().enumerate() {
    result[i] -= v;
  }
  while result.len() > 1 && result.last() == Some(&0) {
    result.pop();
  }
  result
}

/// Make polynomial primitive (divide by GCD of coefficients) and monic
fn make_primitive_monic(coeffs: &[i128]) -> Vec<i128> {
  if coeffs.is_empty() || coeffs.iter().all(|&c| c == 0) {
    return vec![0];
  }
  let mut result = coeffs.to_vec();
  // Trim trailing zeros
  while result.len() > 1 && result.last() == Some(&0) {
    result.pop();
  }
  // Divide by GCD of coefficients
  let g = result
    .iter()
    .copied()
    .filter(|&c| c != 0)
    .fold(0i128, gcd_abs);
  if g > 1 {
    for c in &mut result {
      *c /= g;
    }
  }
  // Make leading coefficient positive (monic convention)
  if let Some(&lc) = result.last()
    && lc < 0
  {
    for c in &mut result {
      *c = -*c;
    }
  }
  result
}

/// Make primitive and check irreducibility, return Some if valid
fn make_primitive_and_irreducible(coeffs: Vec<i128>) -> Option<Vec<i128>> {
  let result = make_primitive_monic(&coeffs);
  // For simple cases like x^n - a, check if this is already irreducible
  // by trying to factor
  let factors = try_factor_integer_poly(&result);
  if factors.len() == 1 {
    Some(factors[0].clone())
  } else {
    // Return the factor of smallest degree > 0
    let mut best = result;
    for f in &factors {
      if f.len() > 1 && f.len() < best.len() {
        best = f.clone();
      }
    }
    Some(make_primitive_monic(&best))
  }
}

/// Try to factor an integer polynomial, return list of irreducible factors
fn try_factor_integer_poly(coeffs: &[i128]) -> Vec<Vec<i128>> {
  if coeffs.len() <= 2 {
    return vec![coeffs.to_vec()];
  }

  // Try rational roots first (Rational Root Theorem)
  let a0 = coeffs[0].abs();
  let an = coeffs.last().copied().unwrap_or(1).abs();
  if a0 == 0 {
    // x divides the polynomial
    let reduced: Vec<i128> = coeffs[1..].to_vec();
    let mut factors = vec![vec![0, 1]]; // x
    let rest = try_factor_integer_poly(&reduced);
    factors.extend(rest);
    return factors;
  }

  // Find divisors of a0 and an
  let divs_a0 = small_divisors(a0);
  let divs_an = small_divisors(an);

  let mut remaining = coeffs.to_vec();
  let mut factors = Vec::new();

  'outer: loop {
    if remaining.len() <= 2 {
      factors.push(remaining);
      return factors;
    }

    for &p in &divs_a0 {
      for &q in &divs_an {
        for &sign in &[1i128, -1] {
          let root_num = sign * p;
          let root_den = q;
          // Check if root_num/root_den is a root
          if eval_poly_rational(&remaining, root_num, root_den) == 0 {
            // Divide by (root_den * x - root_num)
            let divisor = vec![-root_num, root_den];
            if let Some(quotient) = poly_exact_divide(&remaining, &divisor) {
              factors.push(make_primitive_monic(&divisor));
              remaining = make_primitive_monic(&quotient);
              continue 'outer;
            }
          }
        }
      }
    }
    // No more rational roots found
    break;
  }

  // Try to factor what's left using poly_gcd with derivative
  if remaining.len() > 2 {
    // Check if it has repeated factors
    let deriv = poly_derivative_i128(&remaining);
    if let Some(g) = poly_gcd(&remaining, &deriv)
      && g.len() > 1
    {
      // Has repeated factors
      if let Some(q) = poly_exact_divide(&remaining, &g) {
        let sq_free = make_primitive_monic(&q);
        if sq_free.len() > 1 && sq_free.len() < remaining.len() {
          factors.push(sq_free);
          return factors;
        }
      }
    }
  }

  factors.push(remaining);
  factors
}

/// Pick the irreducible factor of a polynomial that vanishes at the given numeric value
fn pick_irreducible_factor(coeffs: &[i128], val: f64) -> Option<Vec<i128>> {
  if coeffs.len() <= 2 {
    return Some(coeffs.to_vec());
  }

  let factors = try_factor_integer_poly(coeffs);
  if factors.len() <= 1 {
    return Some(coeffs.to_vec());
  }

  // Find the factor closest to zero at val
  let mut best = coeffs.to_vec();
  let mut best_eval = eval_poly_f64(coeffs, val).abs();

  for f in &factors {
    if f.len() <= 1 {
      continue;
    }
    let ev = eval_poly_f64(f, val).abs();
    if ev < best_eval || (ev == best_eval && f.len() < best.len()) {
      best = f.clone();
      best_eval = ev;
    }
  }

  Some(make_primitive_monic(&best))
}

/// Evaluate polynomial at f64 value
fn eval_poly_f64(coeffs: &[i128], x: f64) -> f64 {
  let mut result = 0.0f64;
  let mut xpow = 1.0f64;
  for &c in coeffs {
    result += c as f64 * xpow;
    xpow *= x;
  }
  result
}

/// Evaluate polynomial with rational root: p(num/den) * den^degree
fn eval_poly_rational(coeffs: &[i128], num: i128, den: i128) -> i128 {
  let n = coeffs.len() - 1;
  let mut result = 0i128;
  let mut num_pow = 1i128;
  let mut den_pow = den.pow(n as u32);
  for &c in coeffs {
    result += c * num_pow * den_pow;
    num_pow *= num;
    if den_pow != 0 {
      den_pow /= den;
    }
  }
  result
}

/// Polynomial derivative
fn poly_derivative_i128(coeffs: &[i128]) -> Vec<i128> {
  if coeffs.len() <= 1 {
    return vec![0];
  }
  coeffs
    .iter()
    .enumerate()
    .skip(1)
    .map(|(i, &c)| c * i as i128)
    .collect()
}

/// Replace x with -x in polynomial: negate odd-degree coefficients
fn negate_var_in_poly(coeffs: &[i128]) -> Vec<i128> {
  coeffs
    .iter()
    .enumerate()
    .map(|(i, &c)| if i % 2 == 1 { -c } else { c })
    .collect()
}

/// Extract a rational number as (numerator, denominator) from an Expr
fn extract_rational(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
        Some((*p, *q))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Extract a rational pair (p, q) from a Rational expression or integer
fn extract_rational_pair(expr: &Expr) -> Option<(i128, i128)> {
  extract_rational(expr)
}

/// Collect factors from a Times expression
fn collect_times_factors(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let mut factors = collect_times_factors(left);
      factors.extend(collect_times_factors(right));
      factors
    }
    Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
    _ => vec![expr.clone()],
  }
}

/// Binomial coefficient
fn binomial(n: u64, k: u64) -> u64 {
  if k > n {
    return 0;
  }
  if k == 0 || k == n {
    return 1;
  }
  let k = k.min(n - k);
  let mut result = 1u64;
  for i in 0..k {
    result = result * (n - i) / (i + 1);
  }
  result
}

/// GCD of absolute values
fn gcd_abs(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Small divisors of a positive integer (for root-finding)
fn small_divisors(n: i128) -> Vec<i128> {
  if n == 0 {
    return vec![1];
  }
  let n = n.abs();
  let mut divs = Vec::new();
  let limit = (n as f64).sqrt() as i128 + 1;
  for d in 1..=limit.min(1000) {
    if n % d == 0 {
      divs.push(d);
      if d != n / d {
        divs.push(n / d);
      }
    }
  }
  divs.sort();
  divs
}
