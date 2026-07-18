#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::functions::math_ast::{
  expr_to_f64, expr_to_i128, expr_to_rational, gcd as gcd_i128, is_sqrt,
};
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, unevaluated};

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
      // A minimal polynomial is always square-free. The resultant-based
      // construction can return a perfect power (e.g. (x^2+2)^2 for the
      // complex product I*Sqrt[2], where no real numeric value was available
      // to pick the irreducible factor); reduce it to the square-free part.
      let coeffs = make_square_free(&coeffs);
      let poly = coeffs_to_expr(&coeffs, &var);
      let simplified = crate::evaluator::evaluate_expr_to_expr(&poly)?;
      Ok(simplified)
    }
    None => {
      // Return unevaluated
      Ok(unevaluated("MinimalPolynomial", args))
    }
  }
}

/// MinimalPolynomial[α] — the minimal polynomial returned as a pure function
/// of `#1`, e.g. `MinimalPolynomial[Sqrt[2]]` → `-2 + #1^2 &`. Computed via the
/// two-argument form in a private dummy variable, then rewritten to `Slot[1]`.
pub fn minimal_polynomial_pure_ast(
  alpha: &Expr,
) -> Result<Expr, InterpreterError> {
  let dummy = "WoxiMinimalPolynomialDummyVar";
  let poly = minimal_polynomial_ast(&[
    alpha.clone(),
    Expr::Identifier(dummy.to_string()),
  ])?;
  // If the two-argument form could not reduce α, keep the call unevaluated.
  if matches!(&poly, Expr::FunctionCall { name, .. } if name == "MinimalPolynomial")
  {
    return Ok(unevaluated("MinimalPolynomial", &[alpha.clone()]));
  }
  Ok(Expr::Function {
    body: Box::new(replace_identifier_with_slot1(&poly, dummy)),
  })
}

/// Recursively replace every `Identifier(name)` with `Slot(1)`.
fn replace_identifier_with_slot1(expr: &Expr, name: &str) -> Expr {
  match expr {
    Expr::Identifier(n) if n == name => Expr::Slot(1),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(replace_identifier_with_slot1(left, name)),
      right: Box::new(replace_identifier_with_slot1(right, name)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_identifier_with_slot1(operand, name)),
    },
    Expr::FunctionCall { name: fname, args } => Expr::FunctionCall {
      name: fname.clone(),
      args: args
        .iter()
        .map(|a| replace_identifier_with_slot1(a, name))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| replace_identifier_with_slot1(a, name))
        .collect(),
    ),
    other => other.clone(),
  }
}

/// Compute the minimal polynomial coefficients [c0, c1, ..., cn] where
/// c0 + c1*x + c2*x^2 + ... + cn*x^n = 0 and the polynomial is irreducible.
/// Returns coefficients in ascending degree order.
fn compute_minpoly_coeffs(
  expr: &Expr,
) -> Result<Option<Vec<i128>>, InterpreterError> {
  // Expressions that are rational polynomials in a single radical m^(1/q)
  // (e.g. 2^(1/3) + 4^(1/3), where 4^(1/3) = (2^(1/3))^2) get an exact
  // treatment inside Q(m^(1/q)); the pairwise resultant composition below
  // would return a reducible degree-q^2 annihilating polynomial instead.
  if let Some(coeffs) = single_radical_minpoly(expr)? {
    return Ok(Some(coeffs));
  }
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

    // Divide: a / b = a * b^(-1). Some evaluators (e.g. Cos[Pi/5] →
    // (1 + Sqrt[5])/4) leave the quotient as a Divide BinaryOp rather than
    // normalizing it to Times[Rational[…], …]; rewrite it so handle_times
    // can take the product of the minimal polynomials.
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let recip = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new((**right).clone()),
        right: Box::new(Expr::Integer(-1)),
      };
      let product = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new((**left).clone()),
        right: Box::new(recip),
      };
      compute_minpoly_coeffs(&product)
    }

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
    expr if is_sqrt(expr).is_some() => {
      let sqrt_arg = is_sqrt(expr).unwrap();
      let power_expr = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(sqrt_arg.clone()),
        right: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
        }),
      };
      compute_minpoly_coeffs(&power_expr)
    }

    // Unary minus
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
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

    // Root objects: the defining polynomial of a canonical Root object
    // vanishes at the root, so its irreducible factor at the root's
    // numeric value is the minimal polynomial. (wolframscript keeps only
    // irreducible Root polynomials, but Woxi-built Root objects may carry
    // a reducible one.)
    Expr::FunctionCall { name, args }
      if name == "Root"
        && (args.len() == 2 || args.len() == 3)
        && matches!(&args[0], Expr::Function { .. }) =>
    {
      let Expr::Function { body } = &args[0] else {
        return Ok(None);
      };
      let var = "\u{2620}mp\u{2620}";
      let poly =
        crate::syntax::substitute_slots(body, &[Expr::Identifier(var.into())]);
      let expanded = crate::evaluator::evaluate_expr_to_expr(&poly)?;
      match crate::functions::polynomial_ast::extract_poly_coeffs(
        &expanded, var,
      ) {
        Some(c) if c.len() >= 2 => {
          let primitive = make_primitive_monic(&c);
          let nval = numeric_value_of(expr);
          Ok(Some(match nval {
            Some(val) => {
              pick_irreducible_factor(&primitive, val).unwrap_or(primitive)
            }
            None => primitive,
          }))
        }
        _ => Ok(None),
      }
    }

    // BinaryOp Minus
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      // a - b = a + (-b)
      let neg_right = Expr::UnaryOp {
        op: UnaryOperator::Minus,
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
  if let Some((a_num, a_den)) = expr_to_rational(base)
    && let Some((p, q)) = expr_to_rational(exp)
  {
    // α = (a_num/a_den)^(p/q)
    // α^q = (a_num/a_den)^p
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

  // Case: algebraic ^ negative integer (-n) → reciprocal of algebraic^n.
  // (The rational-base case above already handled rational^(negative).)
  // minpoly(1/β) is the coefficient-reversed (reciprocal) polynomial of
  // minpoly(β): e.g. Cos[Pi/4] = Sqrt[2]^(-1), minpoly(Sqrt[2]) = x^2-2,
  // reversed → 2*x^2-1.
  if let Some(n) = expr_to_i128(exp)
    && n < 0
    && -n <= 10
  {
    let k = (-n) as usize;
    let beta_poly = if k == 1 {
      compute_minpoly_coeffs(base)?
    } else {
      match compute_minpoly_coeffs(base)? {
        Some(bp) => {
          let nv = expr_to_f64(base).map(|b| b.powi(k as i32));
          minpoly_of_power(&bp, k as i128, nv)
        }
        None => None,
      }
    };
    if let Some(p) = beta_poly
      && p.first().is_some_and(|&c| c != 0)
    {
      let rev: Vec<i128> = p.iter().rev().copied().collect();
      return Ok(Some(make_primitive_monic(&rev)));
    }
  }

  // Case: algebraic ^ (1/q) — a q-th root of an algebraic number.
  // If p(t) is the minpoly of base, then α = base^(1/q) satisfies p(α^q) = 0,
  // so x^q substituted into p(t) gives a polynomial vanishing at α.
  if let Some((p, q)) = expr_to_rational(exp)
    && p == 1
    && (2..=10).contains(&q)
  {
    let base_poly = compute_minpoly_coeffs(base)?;
    if let Some(bp) = base_poly {
      let numeric_val =
        expr_to_f64(exp).and_then(|ne| expr_to_f64(base).map(|be| be.powf(ne)));
      let composed = compose_with_x_power(&bp, q as usize);
      let primitive = make_primitive_monic(&composed);
      let result = if let Some(val) = numeric_val {
        pick_irreducible_factor(&primitive, val).unwrap_or(primitive)
      } else {
        primitive
      };
      return Ok(Some(result));
    }
  }

  Ok(None)
}

/// Substitute x^k for the variable in a polynomial coefficient list.
/// `coeffs[i]` is the coefficient of t^i; the result has coeffs[i] at index i*k.
fn compose_with_x_power(coeffs: &[i128], k: usize) -> Vec<i128> {
  if coeffs.is_empty() {
    return vec![0];
  }
  let new_len = (coeffs.len() - 1) * k + 1;
  let mut out = vec![0i128; new_len];
  for (i, &c) in coeffs.iter().enumerate() {
    out[i * k] = c;
  }
  out
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
  if let (Some(re_c), Some((p, q))) = (re_poly, expr_to_rational(im)) {
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
      let binom = crate::functions::binomial_coeff(k as i128, j as i128);
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

/// Reduce a polynomial to its square-free part (remove repeated factors).
/// A minimal polynomial is always square-free, so this corrects a candidate
/// like `(x^2+2)^2` to `x^2+2`. For a square-free input, gcd(p, p') is a
/// constant and the polynomial is returned unchanged.
// ---- minimal polynomials inside a single radical extension Q(m^(1/q)) ----

/// Exact rational as a normalized (numerator, positive denominator) pair.
type Rat = (i128, i128);

fn rat_norm(n: i128, d: i128) -> Rat {
  let g = gcd_i128(n, d).max(1);
  let s = if d < 0 { -1 } else { 1 };
  (s * n / g, d.abs() / g)
}

fn rat_add(a: Rat, b: Rat) -> Rat {
  rat_norm(a.0 * b.1 + b.0 * a.1, a.1 * b.1)
}

fn rat_mul(a: Rat, b: Rat) -> Rat {
  rat_norm(a.0 * b.0, a.1 * b.1)
}

/// A rational polynomial in r = base^(1/q); `base: None` means no radical
/// has appeared yet (a plain rational constant with q = 1).
#[derive(Clone)]
struct RadPoly {
  base: Option<i128>,
  q: usize,
  coeffs: Vec<Rat>,
}

fn rad_const(c: Rat) -> RadPoly {
  RadPoly {
    base: None,
    q: 1,
    coeffs: vec![c],
  }
}

const RAD_MAX_DEGREE: usize = 24;

/// Common (base, q) for two polynomials, or None when their radicals live
/// in visibly different extensions.
fn rad_unify(a: &RadPoly, b: &RadPoly) -> Option<(Option<i128>, usize)> {
  let base = match (a.base, b.base) {
    (Some(x), Some(y)) if x != y => return None,
    (Some(x), _) => Some(x),
    (_, y) => y,
  };
  let g = gcd_i128(a.q as i128, b.q as i128) as usize;
  let l = a.q / g * b.q;
  if l > RAD_MAX_DEGREE {
    return None;
  }
  Some((base, l))
}

fn rad_reindex(p: &RadPoly, base: Option<i128>, q: usize) -> RadPoly {
  let f = q / p.q;
  let mut coeffs = vec![(0, 1); q];
  for (j, c) in p.coeffs.iter().enumerate() {
    coeffs[j * f] = *c;
  }
  RadPoly { base, q, coeffs }
}

fn rad_add_polys(a: &RadPoly, b: &RadPoly) -> Option<RadPoly> {
  let (base, q) = rad_unify(a, b)?;
  let a = rad_reindex(a, base, q);
  let b = rad_reindex(b, base, q);
  let coeffs = a
    .coeffs
    .iter()
    .zip(b.coeffs.iter())
    .map(|(&x, &y)| rat_add(x, y))
    .collect();
  Some(RadPoly { base, q, coeffs })
}

fn rad_mul_polys(a: &RadPoly, b: &RadPoly) -> Option<RadPoly> {
  let (base, q) = rad_unify(a, b)?;
  let a = rad_reindex(a, base, q);
  let b = rad_reindex(b, base, q);
  let m = base.unwrap_or(1);
  let mut coeffs: Vec<Rat> = vec![(0, 1); q];
  for (i, &ai) in a.coeffs.iter().enumerate() {
    if ai.0 == 0 {
      continue;
    }
    for (j, &bj) in b.coeffs.iter().enumerate() {
      if bj.0 == 0 {
        continue;
      }
      let idx = i + j;
      let mut term = rat_mul(ai, bj);
      for _ in 0..(idx / q) {
        term = rat_mul(term, (m, 1));
      }
      coeffs[idx % q] = rat_add(coeffs[idx % q], term);
    }
  }
  Some(RadPoly { base, q, coeffs })
}

fn rad_scale(p: &RadPoly, c: Rat) -> RadPoly {
  RadPoly {
    base: p.base,
    q: p.q,
    coeffs: p.coeffs.iter().map(|&x| rat_mul(x, c)).collect(),
  }
}

/// Write b = m^k with k maximal, via complete trial-division factorization.
/// None when b cannot be fully factored quickly (then the radical is left
/// to the generic machinery — a partially reduced base could make x^q - m
/// reducible and the characteristic polynomial no longer a pure power).
fn perfect_power_root(b: i128) -> Option<(i128, i128)> {
  let mut n = b;
  let mut factors: Vec<(i128, i128)> = Vec::new();
  let mut d = 2i128;
  while d * d <= n && d <= 1_000_000 {
    if n % d == 0 {
      let mut e = 0;
      while n % d == 0 {
        n /= d;
        e += 1;
      }
      factors.push((d, e));
    }
    d += 1;
  }
  if n > 1 {
    if d * d > n {
      factors.push((n, 1));
    } else {
      return None;
    }
  }
  let mut g = 0i128;
  for &(_, e) in &factors {
    g = gcd_i128(g, e);
  }
  let mut m = 1i128;
  for &(p, e) in &factors {
    for _ in 0..(e / g) {
      m = m.checked_mul(p)?;
    }
  }
  Some((m, g))
}

/// b^(p/q) as a RadPoly over the maximally reduced base m (so x^q - m is
/// irreducible and Q(m^(1/q)) is a field).
fn rad_radical(b: i128, p: i128, q: i128) -> Option<RadPoly> {
  if b < 2 || p < 1 || q < 2 {
    return None;
  }
  let (m, k) = perfect_power_root(b)?;
  let num = k.checked_mul(p)?;
  let g = gcd_i128(num, q).max(1);
  let (num, den) = (num / g, q / g);
  if num > 64 {
    return None;
  }
  let mut whole: Rat = (1, 1);
  for _ in 0..(num / den) {
    whole = rat_mul(whole, (m, 1));
    if whole.0 > i64::MAX as i128 {
      return None;
    }
  }
  if den == 1 {
    return Some(rad_const(whole));
  }
  if den as usize > RAD_MAX_DEGREE {
    return None;
  }
  let mut coeffs: Vec<Rat> = vec![(0, 1); den as usize];
  coeffs[(num % den) as usize] = whole;
  Some(RadPoly {
    base: Some(m),
    q: den as usize,
    coeffs,
  })
}

/// Recognize `expr` as a rational polynomial in a single radical.
fn collect_rad_poly(expr: &Expr, depth: usize) -> Option<RadPoly> {
  if depth > 32 {
    return None;
  }
  let power = |base: &Expr, exp: &Expr| -> Option<RadPoly> {
    let b = match base {
      Expr::Integer(b) => *b,
      _ => return None,
    };
    match exp {
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
          rad_radical(b, *p, *q)
        } else {
          None
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        if let (Expr::Integer(p), Expr::Integer(q)) = (&**left, &**right) {
          rad_radical(b, *p, *q)
        } else {
          None
        }
      }
      _ => None,
    }
  };
  match expr {
    Expr::Integer(n) => Some(rad_const((*n, 1))),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
        Some(rad_const(rat_norm(*p, *q)))
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      match &args[0] {
        Expr::Integer(b) => rad_radical(*b, 1, 2),
        _ => None,
      }
    }
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      let mut acc = collect_rad_poly(&args[0], depth + 1)?;
      for a in &args[1..] {
        acc = rad_add_polys(&acc, &collect_rad_poly(a, depth + 1)?)?;
      }
      Some(acc)
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      let mut acc = collect_rad_poly(&args[0], depth + 1)?;
      for a in &args[1..] {
        acc = rad_mul_polys(&acc, &collect_rad_poly(a, depth + 1)?)?;
      }
      Some(acc)
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      power(&args[0], &args[1])
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(rad_scale(&collect_rad_poly(operand, depth + 1)?, (-1, 1))),
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus => rad_add_polys(
        &collect_rad_poly(left, depth + 1)?,
        &collect_rad_poly(right, depth + 1)?,
      ),
      BinaryOperator::Minus => rad_add_polys(
        &collect_rad_poly(left, depth + 1)?,
        &rad_scale(&collect_rad_poly(right, depth + 1)?, (-1, 1)),
      ),
      BinaryOperator::Times => rad_mul_polys(
        &collect_rad_poly(left, depth + 1)?,
        &collect_rad_poly(right, depth + 1)?,
      ),
      BinaryOperator::Divide => {
        let den = collect_rad_poly(right, depth + 1)?;
        // Only division by a radical-free rational is a polynomial op.
        if den.base.is_some() || den.coeffs[0].0 == 0 {
          return None;
        }
        let inv = (den.coeffs[0].1, den.coeffs[0].0);
        Some(rad_scale(&collect_rad_poly(left, depth + 1)?, inv))
      }
      BinaryOperator::Power => power(left, right),
      _ => None,
    },
    _ => None,
  }
}

/// Minimal polynomial of a rational polynomial in one radical m^(1/q): the
/// characteristic polynomial of multiplication by the value on the basis
/// {1, r, ..., r^(q-1)} of the field Q(r) is minpoly^(q/deg), so its
/// square-free part is exactly the minimal polynomial.
fn single_radical_minpoly(
  expr: &Expr,
) -> Result<Option<Vec<i128>>, InterpreterError> {
  let Some(p) = collect_rad_poly(expr, 0) else {
    return Ok(None);
  };
  let (Some(m), q) = (p.base, p.q) else {
    return Ok(None);
  };
  if q < 2 {
    return Ok(None);
  }

  // Multiplication matrix: column i holds the coefficients of value * r^i.
  let mut mat = vec![vec![(0i128, 1i128); q]; q];
  for i in 0..q {
    for (j, &c) in p.coeffs.iter().enumerate() {
      if c.0 == 0 {
        continue;
      }
      let idx = i + j;
      let mut term = c;
      for _ in 0..(idx / q) {
        term = rat_mul(term, (m, 1));
      }
      mat[idx % q][i] = rat_add(mat[idx % q][i], term);
    }
  }

  // Characteristic polynomial det(x I - M) via the evaluator.
  let var = Expr::Identifier("MinimalPolynomial$r".to_string());
  let rat_expr = |r: Rat| {
    if r.1 == 1 {
      Expr::Integer(r.0)
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Integer(r.0)),
        right: Box::new(Expr::Integer(r.1)),
      }
    }
  };
  let mut rows = Vec::with_capacity(q);
  for (i, mat_row) in mat.iter().enumerate() {
    let mut row = Vec::with_capacity(q);
    for (j, &entry) in mat_row.iter().enumerate() {
      row.push(if i == j {
        Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(var.clone()),
          right: Box::new(rat_expr(entry)),
        }
      } else {
        rat_expr(rat_mul(entry, (-1, 1)))
      });
    }
    rows.push(Expr::List(row.into()));
  }
  let det = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Det".to_string(),
    args: vec![Expr::List(rows.into())].into(),
  })?;
  let coeff_list =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "CoefficientList".to_string(),
      args: vec![det, var].into(),
    })?;
  let Expr::List(ref items) = coeff_list else {
    return Ok(None);
  };
  let mut rats: Vec<Rat> = Vec::with_capacity(items.len());
  for it in items.iter() {
    match it {
      Expr::Integer(n) => rats.push((*n, 1)),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          rats.push(rat_norm(*n, *d));
        } else {
          return Ok(None);
        }
      }
      _ => return Ok(None),
    }
  }
  if rats.len() != q + 1 {
    return Ok(None);
  }
  let mut den_lcm = 1i128;
  for &(_, d) in &rats {
    den_lcm = den_lcm / gcd_i128(den_lcm, d).max(1) * d;
  }
  let ints: Vec<i128> = rats.iter().map(|&(n, d)| n * (den_lcm / d)).collect();
  Ok(Some(make_square_free(&make_primitive_monic(&ints))))
}

fn make_square_free(coeffs: &[i128]) -> Vec<i128> {
  if coeffs.len() <= 2 {
    return make_primitive_monic(coeffs);
  }
  let deriv = poly_derivative_i128(coeffs);
  if deriv.iter().all(|&c| c == 0) {
    return make_primitive_monic(coeffs);
  }
  if let Some(g) = poly_gcd(coeffs, &deriv)
    && g.len() > 1
    && let Some(q) = poly_exact_divide(coeffs, &g)
  {
    return make_primitive_monic(&q);
  }
  make_primitive_monic(coeffs)
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
    .fold(0i128, gcd_i128);
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

/// Numeric value of an expression via `N[...]`, used to pick the right
/// irreducible factor for Root objects (whose value `expr_to_f64` cannot
/// see directly).
fn numeric_value_of(expr: &Expr) -> Option<f64> {
  if let Some(v) = expr_to_f64(expr) {
    return Some(v);
  }
  match crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "N".to_string(),
    args: vec![expr.clone()].into(),
  }) {
    Ok(Expr::Real(v)) => Some(v),
    Ok(Expr::Integer(n)) => Some(n as f64),
    _ => None,
  }
}

/// NumberFieldSignature[a] — the signature {r1, r2} of the number field
/// Q(a): its minimal polynomial has r1 real roots and r2 complex-conjugate
/// pairs. Extra arguments are interpreted as a Root specification
/// (`NumberFieldSignature[f, k]` ≡ `NumberFieldSignature[Root[f, k]]`),
/// and non-algebraic arguments emit `nalg` and stay unevaluated — both
/// matching wolframscript.
pub fn number_field_signature_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let alpha = if args.len() == 1 {
    args[0].clone()
  } else {
    crate::evaluator::evaluate_expr_to_expr(&unevaluated("Root", args))?
  };
  if let Some(coeffs) = compute_minpoly_coeffs(&alpha)? {
    let coeffs = make_square_free(&coeffs);
    let deg = coeffs.len() - 1;
    if deg >= 1 {
      let r1 = sturm_real_root_count(&coeffs);
      return Ok(Expr::List(
        vec![
          Expr::Integer(r1 as i128),
          Expr::Integer(((deg - r1) / 2) as i128),
        ]
        .into(),
      ));
    }
  }
  crate::emit_message(&format!(
    "NumberFieldSignature::nalg: {} is not an explicit algebraic number.",
    crate::syntax::expr_to_string(&alpha)
  ));
  Ok(unevaluated("NumberFieldSignature", args))
}

/// Minimal-polynomial coefficients (ascending, primitive, positive leading
/// coefficient) of an explicit algebraic number, or None.
fn algebraic_minpoly(
  expr: &Expr,
) -> Result<Option<Vec<i128>>, InterpreterError> {
  if let Some(coeffs) = compute_minpoly_coeffs(expr)? {
    let coeffs = make_square_free(&coeffs);
    if coeffs.len() >= 2 {
      return Ok(Some(coeffs));
    }
  }
  Ok(None)
}

/// Emit `<head>::nalg` and return the unevaluated call, the shared failure
/// path of the AlgebraicNumber* functions (AlgebraicUnitQ instead returns
/// False without a message).
fn nalg_unevaluated(head: &str, args: &[Expr]) -> Expr {
  crate::emit_message(&format!(
    "{head}::nalg: {} is not an explicit algebraic number.",
    crate::syntax::expr_to_string(&args[0])
  ));
  unevaluated(head, args)
}

/// AlgebraicUnitQ[a] — True iff a is an algebraic integer whose reciprocal
/// is also an algebraic integer: the minimal polynomial is monic with
/// constant term ±1. Non-algebraic input gives False without a message.
pub fn algebraic_unit_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let is_unit = match algebraic_minpoly(&args[0])? {
    Some(c) => *c.last().unwrap() == 1 && c[0].abs() == 1,
    None => false,
  };
  Ok(Expr::Identifier(
    if is_unit { "True" } else { "False" }.to_string(),
  ))
}

/// AlgebraicNumberNorm[a] — the product of all conjugates of a over Q:
/// (-1)^n * c_0/c_n for the degree-n minimal polynomial.
pub fn algebraic_number_norm_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Some(c) = algebraic_minpoly(&args[0])? else {
    return Ok(nalg_unevaluated("AlgebraicNumberNorm", args));
  };
  let n = c.len() - 1;
  let sign = if n % 2 == 0 { 1 } else { -1 };
  crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::Integer(sign * c[0])),
    right: Box::new(Expr::Integer(*c.last().unwrap())),
  })
}

/// AlgebraicNumberTrace[a] — the sum of all conjugates of a over Q:
/// -c_(n-1)/c_n for the degree-n minimal polynomial.
pub fn algebraic_number_trace_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Some(c) = algebraic_minpoly(&args[0])? else {
    return Ok(nalg_unevaluated("AlgebraicNumberTrace", args));
  };
  let n = c.len() - 1;
  crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(Expr::Integer(-c[n - 1])),
    right: Box::new(Expr::Integer(c[n])),
  })
}

/// AlgebraicNumberDenominator[a] — the smallest positive integer d such
/// that d*a is an algebraic integer: the minimal polynomial of d*a is
/// (up to content) sum c_k d^(n-k) x^k, which is monic exactly when
/// c_n divides every c_k d^(n-k).
pub fn algebraic_number_denominator_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Some(c) = algebraic_minpoly(&args[0])? else {
    return Ok(nalg_unevaluated("AlgebraicNumberDenominator", args));
  };
  let n = c.len() - 1;
  let lc = c[n];
  for d in 1..=lc {
    let divides_all = (0..n).all(|k| {
      let mut pow = 1i128 % lc;
      for _ in 0..(n - k) {
        pow = (pow * (d % lc)) % lc;
      }
      (c[k] % lc) * pow % lc == 0
    });
    if divides_all {
      return Ok(Expr::Integer(d));
    }
  }
  Ok(Expr::Integer(lc))
}

/// Count the real roots of a square-free integer polynomial with a Sturm
/// chain over exact big-integer arithmetic (pseudo-remainders scaled by
/// positive constants so the sign structure is preserved).
fn sturm_real_root_count(coeffs: &[i128]) -> usize {
  use num_bigint::BigInt;
  use num_traits::{Signed, Zero};

  fn trim(mut v: Vec<BigInt>) -> Vec<BigInt> {
    while v.len() > 1 && v.last().is_some_and(|c| c.is_zero()) {
      v.pop();
    }
    v
  }

  use crate::functions::math_ast::gcd_bigint;

  fn content_reduce(v: &[BigInt]) -> Vec<BigInt> {
    let mut g = BigInt::ZERO;
    for c in v {
      g = gcd_bigint(&g, c);
    }
    if g > BigInt::from(1) {
      v.iter().map(|c| c / &g).collect()
    } else {
      v.to_vec()
    }
  }

  // Pseudo-remainder of a by b, scaled by the positive constant
  // |lc(b)|^(deg a - deg b + 1) so integer division stays exact and the
  // sign of the true remainder is preserved.
  fn pseudo_rem(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let lc = b.last().unwrap().clone();
    let delta = a.len() - b.len();
    let scale = lc.abs().pow(delta as u32 + 1);
    let mut r: Vec<BigInt> = a.iter().map(|c| c * &scale).collect();
    while r.len() >= b.len() && !(r.len() == 1 && r[0].is_zero()) {
      let q = r.last().unwrap() / &lc;
      let shift = r.len() - b.len();
      for (i, bc) in b.iter().enumerate() {
        let idx = shift + i;
        let sub = bc * &q;
        r[idx] -= sub;
      }
      r = trim(r);
      if r.len() < b.len() {
        break;
      }
    }
    r
  }

  let p0 = trim(coeffs.iter().map(|&c| BigInt::from(c)).collect());
  if p0.len() <= 1 {
    return 0;
  }
  let p1: Vec<BigInt> = p0
    .iter()
    .enumerate()
    .skip(1)
    .map(|(i, c)| c * BigInt::from(i as i64))
    .collect();
  let mut chain = vec![p0, trim(p1)];
  loop {
    let b = chain.last().unwrap();
    if b.len() <= 1 && b[0].is_zero() {
      chain.pop();
      break;
    }
    if b.len() == 1 {
      break;
    }
    let a = &chain[chain.len() - 2];
    let r = trim(pseudo_rem(a, b));
    if r.iter().all(|c| c.is_zero()) {
      break;
    }
    let neg: Vec<BigInt> = content_reduce(&r).iter().map(|c| -c).collect();
    chain.push(neg);
  }

  let variations = |at_neg_inf: bool| -> usize {
    let mut count = 0usize;
    let mut prev = 0i8;
    for p in &chain {
      let lc = p.last().unwrap();
      let mut s: i8 = if lc.is_zero() {
        0
      } else if lc.is_positive() {
        1
      } else {
        -1
      };
      if at_neg_inf && (p.len() - 1) % 2 == 1 {
        s = -s;
      }
      if s != 0 {
        if prev != 0 && s != prev {
          count += 1;
        }
        prev = s;
      }
    }
    count
  };

  variations(true) - variations(false)
}

/// NumberFieldDiscriminant[a] — the discriminant of the number field
/// Q(a). Rationals give 1; quadratic fields give the fundamental
/// discriminant of b^2 - 4 a c; for degree >= 3 with a monic minimal
/// polynomial the polynomial discriminant is returned when Z[a] is
/// p-maximal (Dedekind's criterion) at every prime whose square divides
/// it, and the call stays unevaluated otherwise (wolframscript computes
/// the full conductor there). Non-algebraic input emits nalg.
pub fn number_field_discriminant_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("NumberFieldDiscriminant", args));
  if args.len() != 1 {
    return unevaluated();
  }
  let ev = crate::evaluator::evaluate_expr_to_expr;
  let var = Expr::Identifier("NumberFieldDiscriminant$x".to_string());
  let mp = ev(&Expr::FunctionCall {
    name: "MinimalPolynomial".to_string(),
    args: vec![args[0].clone(), var.clone()].into(),
  })?;
  if matches!(&mp, Expr::FunctionCall { name, .. } if name == "MinimalPolynomial")
  {
    crate::emit_message(&format!(
      "NumberFieldDiscriminant::nalg: {} is not an explicit algebraic number.",
      crate::syntax::expr_to_string(&args[0])
    ));
    return unevaluated();
  }
  // Integer coefficient list, constant first.
  let coeff_expr = ev(&Expr::FunctionCall {
    name: "CoefficientList".to_string(),
    args: vec![mp.clone(), var.clone()].into(),
  })?;
  let coeffs: Vec<i128> = match &coeff_expr {
    Expr::List(items) => {
      let cs: Option<Vec<i128>> = items
        .iter()
        .map(|e| match e {
          Expr::Integer(n) => Some(*n),
          _ => None,
        })
        .collect();
      match cs {
        Some(cs) if cs.len() >= 2 => cs,
        // Degree 0 (rational input never reaches here — its minimal
        // polynomial is linear) or non-integer coefficients.
        Some(_) => return Ok(Expr::Integer(1)),
        None => return unevaluated(),
      }
    }
    _ => return unevaluated(),
  };
  let degree = coeffs.len() - 1;
  if degree <= 1 {
    return Ok(Expr::Integer(1));
  }
  if degree == 2 {
    let (c, b, a) = (coeffs[0], coeffs[1], coeffs[2]);
    let d = b.checked_mul(b).and_then(|b2| {
      a.checked_mul(c)
        .and_then(|ac| ac.checked_mul(4))
        .and_then(|f| b2.checked_sub(f))
    });
    let Some(d) = d else { return unevaluated() };
    let Some(d0) = squarefree_part(d) else {
      return unevaluated();
    };
    let fundamental = if d0.rem_euclid(4) == 1 { d0 } else { 4 * d0 };
    return Ok(Expr::Integer(fundamental));
  }
  // Degree >= 3: only the monic (algebraic-integer) case.
  if coeffs[degree] != 1 {
    return unevaluated();
  }
  let disc = ev(&Expr::FunctionCall {
    name: "Discriminant".to_string(),
    args: vec![mp, var].into(),
  })?;
  let Expr::Integer(disc) = disc else {
    return unevaluated();
  };
  if disc == 0 {
    return unevaluated();
  }
  let Some(square_primes) = primes_with_square_dividing(disc) else {
    return unevaluated();
  };
  for p in square_primes {
    match dedekind_p_maximal(&coeffs, p) {
      Some(true) => {}
      _ => return unevaluated(),
    }
  }
  Ok(Expr::Integer(disc))
}

/// The squarefree part of n (with n's sign), or None when the unfactored
/// remainder is too large to classify.
fn squarefree_part(n: i128) -> Option<i128> {
  if n == 0 {
    return None;
  }
  let sign = n.signum();
  let mut m = n.abs();
  let mut part: i128 = 1;
  let mut p: i128 = 2;
  while p * p <= m && p < 1_000_000 {
    let mut count = 0;
    while m % p == 0 {
      m /= p;
      count += 1;
    }
    if count % 2 == 1 {
      part *= p;
    }
    p += 1;
  }
  // The remainder is prime, 1, or a square of a prime > bound; larger
  // composites cannot be classified here.
  if m == 1 {
    Some(sign * part)
  } else {
    let root = (m as f64).sqrt().round() as i128;
    if root * root == m {
      Some(sign * part)
    } else if is_probable_prime(m) {
      Some(sign * part * m)
    } else {
      None
    }
  }
}

fn is_probable_prime(n: i128) -> bool {
  if n < 2 {
    return false;
  }
  for p in [2i128, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
    if n == p {
      return true;
    }
    if n % p == 0 {
      return false;
    }
  }
  // Deterministic Miller-Rabin for < 3.3e24 with these bases.
  let d = n - 1;
  let s = d.trailing_zeros();
  let d = d >> s;
  'witness: for a in [2i128, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
    let mut x = mod_pow(a, d, n);
    if x == 1 || x == n - 1 {
      continue;
    }
    for _ in 1..s {
      x = mod_mul(x, x, n);
      if x == n - 1 {
        continue 'witness;
      }
    }
    return false;
  }
  true
}

fn mod_mul(a: i128, b: i128, m: i128) -> i128 {
  // i128 multiplication can overflow for very large moduli; the values
  // here come from discriminants that already fit comfortably.
  ((a as u128).wrapping_mul(b as u128) % (m as u128)) as i128
}

fn mod_pow(mut base: i128, mut exp: i128, m: i128) -> i128 {
  let mut acc: i128 = 1;
  base %= m;
  while exp > 0 {
    if exp & 1 == 1 {
      acc = mod_mul(acc, base, m);
    }
    base = mod_mul(base, base, m);
    exp >>= 1;
  }
  acc
}

/// The primes whose square divides n, or None when n cannot be fully
/// classified by trial division.
fn primes_with_square_dividing(n: i128) -> Option<Vec<i128>> {
  let mut m = n.abs();
  let mut out = Vec::new();
  let mut p: i128 = 2;
  while p * p <= m && p < 1_000_000 {
    if m % p == 0 {
      let mut count = 0;
      while m % p == 0 {
        m /= p;
        count += 1;
      }
      if count >= 2 {
        out.push(p);
      }
    }
    p += 1;
  }
  if m == 1 {
    return Some(out);
  }
  let root = (m as f64).sqrt().round() as i128;
  if root * root == m && is_probable_prime(root) {
    out.push(root);
    Some(out)
  } else if is_probable_prime(m) {
    Some(out)
  } else {
    None
  }
}

/// Dedekind's criterion: is Z[a] p-maximal for the monic minimal
/// polynomial f (coefficients constant-first)? With f-bar = g-bar h-bar
/// (g-bar the radical of f mod p) and F = (g h - f)/p, the order is
/// p-maximal iff gcd(F-bar, g-bar, h-bar) == 1 in F_p[x].
fn dedekind_p_maximal(coeffs: &[i128], p: i128) -> Option<bool> {
  // F_p[x] helpers on constant-first coefficient vectors.
  let norm = |v: &mut Vec<i128>| {
    while v.len() > 1 && *v.last()? == 0 {
      v.pop();
    }
    Some(())
  };
  let reduce = |v: &[i128]| -> Vec<i128> {
    let mut out: Vec<i128> = v.iter().map(|c| c.rem_euclid(p)).collect();
    while out.len() > 1 && *out.last().unwrap() == 0 {
      out.pop();
    }
    out
  };
  let inv = |a: i128| -> i128 { mod_pow(a.rem_euclid(p), p - 2, p) };
  let divmod = |num: &[i128], den: &[i128]| -> (Vec<i128>, Vec<i128>) {
    let mut rem = num.to_vec();
    let dl = den.len();
    if rem.len() < dl {
      return (vec![0], rem);
    }
    let mut quo = vec![0i128; rem.len() - dl + 1];
    let lead_inv = inv(den[dl - 1]);
    for i in (0..quo.len()).rev() {
      let c = mod_mul(rem[i + dl - 1], lead_inv, p);
      quo[i] = c;
      if c != 0 {
        for (j, d) in den.iter().enumerate() {
          rem[i + j] = (rem[i + j] - mod_mul(c, *d, p)).rem_euclid(p);
        }
      }
    }
    let mut r = rem;
    r.truncate(dl - 1);
    if r.is_empty() {
      r.push(0);
    }
    while r.len() > 1 && *r.last().unwrap() == 0 {
      r.pop();
    }
    (quo, r)
  };
  let is_zero = |v: &[i128]| v.len() == 1 && v[0] == 0;
  let gcd = |a: &[i128], b: &[i128]| -> Vec<i128> {
    let (mut x, mut y) = (reduce(a), reduce(b));
    while !is_zero(&y) {
      let (_, r) = divmod(&x, &y);
      x = y;
      y = r;
    }
    // Monicize.
    if !is_zero(&x) {
      let li = inv(*x.last().unwrap());
      for c in &mut x {
        *c = mod_mul(*c, li, p);
      }
    }
    x
  };
  let derivative = |v: &[i128]| -> Vec<i128> {
    if v.len() <= 1 {
      return vec![0];
    }
    reduce(
      &v[1..]
        .iter()
        .enumerate()
        .map(|(i, c)| c.checked_mul(i as i128 + 1).unwrap_or(0))
        .collect::<Vec<_>>(),
    )
  };
  let mul = |a: &[i128], b: &[i128]| -> Vec<i128> {
    let mut out = vec![0i128; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
      for (j, y) in b.iter().enumerate() {
        out[i + j] = (out[i + j] + mod_mul(*x, *y, p)).rem_euclid(p);
      }
    }
    out
  };

  let f_bar = reduce(coeffs);
  // g-bar: the radical (product of distinct irreducible factors) of
  // f mod p. When the derivative vanishes, f is a p-th power u(x)^p and
  // rad(f) = rad(u); in F_p the p-th root just contracts exponents
  // (Frobenius fixes the coefficients).
  fn radical(
    f: &[i128],
    p: i128,
    gcd: &dyn Fn(&[i128], &[i128]) -> Vec<i128>,
    divmod: &dyn Fn(&[i128], &[i128]) -> (Vec<i128>, Vec<i128>),
    derivative: &dyn Fn(&[i128]) -> Vec<i128>,
  ) -> Option<Vec<i128>> {
    let is_zero = |v: &[i128]| v.len() == 1 && v[0] == 0;
    if f.len() <= 1 {
      return Some(vec![1]);
    }
    let d = derivative(f);
    if is_zero(&d) {
      // p-th root: keep the coefficients at exponents divisible by p.
      let mut u = Vec::new();
      for (i, c) in f.iter().enumerate() {
        if (i as i128) % p == 0 {
          u.push(*c);
        } else if *c != 0 {
          return None;
        }
      }
      return radical(&u, p, gcd, divmod, derivative);
    }
    let g = gcd(f, &d);
    // w: each factor with multiplicity not divisible by p, once.
    let (w, r) = divmod(f, &g);
    if !is_zero(&r) {
      return None;
    }
    // m: the p-th-power part — g with all w-factors stripped.
    let mut m = g;
    loop {
      let c = gcd(&m, &w);
      if c.len() <= 1 {
        break;
      }
      let (q, r) = divmod(&m, &c);
      if !is_zero(&r) {
        return None;
      }
      m = q;
    }
    let r2 = radical(&m, p, gcd, divmod, derivative)?;
    // Combine, dividing out any overlap.
    let overlap = gcd(&w, &r2);
    let (extra, r) = divmod(&r2, &overlap);
    if !is_zero(&r) {
      return None;
    }
    let mut out = vec![0i128; w.len() + extra.len() - 1];
    for (i, x) in w.iter().enumerate() {
      for (j, y) in extra.iter().enumerate() {
        out[i + j] = (out[i + j] + x * y).rem_euclid(p);
      }
    }
    while out.len() > 1 && *out.last().unwrap() == 0 {
      out.pop();
    }
    Some(out)
  }
  let g_bar = radical(&f_bar, p, &gcd, &divmod, &derivative)?;
  let (h_bar, hr) = divmod(&f_bar, &g_bar);
  if !is_zero(&hr) {
    return None;
  }
  // Lift g, h to Z[x] with coefficients in 0..p, F = (g h - f)/p.
  let gh = {
    let mut out = vec![0i128; g_bar.len() + h_bar.len() - 1];
    for (i, x) in g_bar.iter().enumerate() {
      for (j, y) in h_bar.iter().enumerate() {
        out[i + j] = out[i + j].checked_add(x.checked_mul(*y)?)?;
      }
    }
    out
  };
  let mut big_f = vec![0i128; gh.len().max(coeffs.len())];
  for (i, c) in gh.iter().enumerate() {
    big_f[i] = *c;
  }
  for (i, c) in coeffs.iter().enumerate() {
    big_f[i] = big_f[i].checked_sub(*c)?;
  }
  if big_f.iter().any(|c| c % p != 0) {
    return None;
  }
  let f_over_p: Vec<i128> = big_f.iter().map(|c| c / p).collect();
  let f_over_p_bar = reduce(&f_over_p);
  let mut acc = gcd(&f_over_p_bar, &g_bar);
  acc = gcd(&acc, &h_bar);
  // Suppress the unused-mul warning path.
  let _ = mul;
  norm(&mut acc)?;
  Some(acc.len() == 1 && acc[0] != 0)
}
