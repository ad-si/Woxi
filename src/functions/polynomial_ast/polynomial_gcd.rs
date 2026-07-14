#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, expr_to_string};

/// PolynomialGCD[p1, p2, ...] - greatest common divisor of polynomials
pub fn polynomial_gcd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialGCD expects at least 2 arguments".into(),
    ));
  }

  // Separate an optional `Modulus -> p` option from the polynomial arguments
  // (otherwise it is mistaken for a polynomial/variable, e.g. `Modulus` sorts
  // before `x` and the GCD is computed in the wrong variable).
  let mut polys: Vec<Expr> = Vec::new();
  let mut modulus: Option<i128> = None;
  for arg in args {
    if let Some(m) = extract_modulus_option(arg) {
      modulus = Some(m);
    } else {
      polys.push(arg.clone());
    }
  }
  if polys.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialGCD expects at least 2 polynomial arguments".into(),
    ));
  }

  // Detect variables from the polynomial expressions only
  let mut all_vars = std::collections::HashSet::new();
  for arg in &polys {
    collect_variables(arg, &mut all_vars);
  }

  // Remove known constants
  all_vars.remove("Pi");
  all_vars.remove("E");
  all_vars.remove("I");

  if all_vars.is_empty() {
    // All arguments are numeric - compute integer GCD
    return integer_gcd_multi(&polys);
  }

  // Sort variables alphabetically for deterministic behavior
  let mut var_list: Vec<_> = all_vars.into_iter().collect();
  var_list.sort();

  // For multivariate polynomials, use the first variable and recurse
  let var = &var_list[0];

  // With a modulus, compute the GCD over the field GF(p).
  if let Some(p) = modulus {
    return polynomial_gcd_mod(&polys, var, p);
  }

  // Fold pairwise GCD
  let mut result = crate::evaluator::evaluate_expr_to_expr(&polys[0])?;
  for arg in &polys[1..] {
    let arg_eval = crate::evaluator::evaluate_expr_to_expr(arg)?;
    result = poly_gcd_pair(&result, &arg_eval, var)?;
  }

  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Extract `Modulus -> n` (n > 1) from an option argument, in either the
/// `Expr::Rule` or `Rule[...]` form.
pub(super) fn extract_modulus_option(opt: &Expr) -> Option<i128> {
  if let Expr::Rule {
    pattern,
    replacement,
  } = opt
    && let Expr::Identifier(s) = pattern.as_ref()
    && s == "Modulus"
  {
    return crate::functions::math_ast::expr_to_i128(replacement)
      .filter(|&m| m > 1);
  }
  if let Expr::FunctionCall { name, args } = opt
    && (name == "Rule" || name == "RuleDelayed")
    && args.len() == 2
    && let Expr::Identifier(s) = &args[0]
    && s == "Modulus"
  {
    return crate::functions::math_ast::expr_to_i128(&args[1])
      .filter(|&m| m > 1);
  }
  None
}

/// PolynomialGCD over GF(p): fold the pairwise modular GCD of the coefficient
/// vectors and rebuild the resulting (monic) polynomial.
fn polynomial_gcd_mod(
  polys: &[Expr],
  var: &str,
  p: i128,
) -> Result<Expr, InterpreterError> {
  let mut acc = poly_to_coeffs_mod(&polys[0], var, p)?;
  for poly in &polys[1..] {
    let b = poly_to_coeffs_mod(poly, var, p)?;
    acc = poly_gcd_coeffs_mod(&acc, &b, p);
  }
  Ok(coeffs_to_poly(&acc, var, p))
}

/// Coefficient vector (low-to-high, reduced mod p) of a univariate polynomial.
pub(super) fn poly_to_coeffs_mod(
  poly: &Expr,
  var: &str,
  p: i128,
) -> Result<Vec<i128>, InterpreterError> {
  let expanded = expand_and_combine(poly);
  let deg = max_power_int(&expanded, var).unwrap_or(0);
  let mut coeffs = Vec::with_capacity((deg + 1) as usize);
  for i in 0..=deg {
    let c = coefficient_ast(&[
      poly.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(i),
    ])?;
    let c = crate::evaluator::evaluate_expr_to_expr(&c)?;
    let ci = crate::functions::math_ast::expr_to_i128(&c).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "PolynomialGCD modulus form requires integer coefficients".into(),
      )
    })?;
    coeffs.push(mod_norm(ci, p));
  }
  Ok(coeffs)
}

fn mod_norm(a: i128, p: i128) -> i128 {
  ((a % p) + p) % p
}

/// Modular inverse of `a` mod `p` via the extended Euclidean algorithm
/// (assumes gcd(a, p) = 1, which holds for p prime and a not a multiple of p).
fn mod_inv(a: i128, p: i128) -> i128 {
  let (mut old_r, mut r) = (mod_norm(a, p), p);
  let (mut old_s, mut s) = (1i128, 0i128);
  while r != 0 {
    let q = old_r / r;
    let tr = old_r - q * r;
    old_r = r;
    r = tr;
    let ts = old_s - q * s;
    old_s = s;
    s = ts;
  }
  mod_norm(old_s, p)
}

/// Drop high-degree zero coefficients, keeping at least `[0]`.
fn trim_high(c: &mut Vec<i128>) {
  while c.len() > 1 && *c.last().unwrap() == 0 {
    c.pop();
  }
}

fn is_zero_poly(c: &[i128]) -> bool {
  c.iter().all(|&x| x == 0)
}

/// Polynomial division `a = q*b + r` over GF(p), returning `(quotient,
/// remainder)`, coefficients low-to-high. A zero divisor yields `(0, a)`.
pub(super) fn poly_divmod_mod(
  a: &[i128],
  b: &[i128],
  p: i128,
) -> (Vec<i128>, Vec<i128>) {
  let mut r: Vec<i128> = a.iter().map(|&x| mod_norm(x, p)).collect();
  let mut b: Vec<i128> = b.iter().map(|&x| mod_norm(x, p)).collect();
  trim_high(&mut r);
  trim_high(&mut b);
  if is_zero_poly(&b) {
    return (vec![0], r);
  }
  let db = b.len();
  let inv = mod_inv(b[db - 1], p);
  let mut q = vec![0i128; if r.len() >= db { r.len() - db + 1 } else { 1 }];
  loop {
    trim_high(&mut r);
    if r.len() < db || is_zero_poly(&r) {
      break;
    }
    let dr = r.len();
    let factor = mod_norm(r[dr - 1] * inv, p);
    let shift = dr - db;
    q[shift] = mod_norm(q[shift] + factor, p);
    for i in 0..db {
      r[shift + i] = mod_norm(r[shift + i] - factor * b[i], p);
    }
  }
  trim_high(&mut q);
  trim_high(&mut r);
  (q, r)
}

/// Polynomial remainder `a mod b` over GF(p), coefficients low-to-high.
fn poly_rem_mod(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  poly_divmod_mod(a, b, p).1
}

/// Product of two coefficient vectors over GF(p) (polynomial multiplication).
fn poly_mul_mod(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  if is_zero_poly(a) || is_zero_poly(b) {
    return vec![0];
  }
  let mut res = vec![0i128; a.len() + b.len() - 1];
  for (i, &ai) in a.iter().enumerate() {
    for (j, &bj) in b.iter().enumerate() {
      res[i + j] = mod_norm(res[i + j] + ai * bj, p);
    }
  }
  trim_high(&mut res);
  res
}

/// Difference `a - b` of two coefficient vectors over GF(p).
fn poly_sub_mod(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  let n = a.len().max(b.len());
  let mut res = vec![0i128; n];
  for (i, ri) in res.iter_mut().enumerate() {
    let ai = if i < a.len() { a[i] } else { 0 };
    let bi = if i < b.len() { b[i] } else { 0 };
    *ri = mod_norm(ai - bi, p);
  }
  trim_high(&mut res);
  res
}

/// Extended Euclidean algorithm over GF(p): returns `(g, s, t)` (coefficient
/// vectors) with `s*a + t*b == g` and `g` monic.
pub(super) fn poly_extended_gcd_mod(
  a: &[i128],
  b: &[i128],
  p: i128,
) -> (Vec<i128>, Vec<i128>, Vec<i128>) {
  let mut old_r: Vec<i128> = a.iter().map(|&x| mod_norm(x, p)).collect();
  let mut r: Vec<i128> = b.iter().map(|&x| mod_norm(x, p)).collect();
  trim_high(&mut old_r);
  trim_high(&mut r);
  let mut old_s = vec![1i128];
  let mut s = vec![0i128];
  let mut old_t = vec![0i128];
  let mut t = vec![1i128];
  while !is_zero_poly(&r) {
    let (quot, rem) = poly_divmod_mod(&old_r, &r, p);
    let new_s = poly_sub_mod(&old_s, &poly_mul_mod(&quot, &s, p), p);
    let new_t = poly_sub_mod(&old_t, &poly_mul_mod(&quot, &t, p), p);
    old_r = r;
    r = rem;
    old_s = s;
    s = new_s;
    old_t = t;
    t = new_t;
  }
  // Normalize so the GCD is monic, scaling s and t by the same inverse.
  trim_high(&mut old_r);
  if !is_zero_poly(&old_r) {
    let inv = mod_inv(*old_r.last().unwrap(), p);
    let scale = |v: &[i128]| -> Vec<i128> {
      v.iter().map(|&x| mod_norm(x * inv, p)).collect()
    };
    old_r = scale(&old_r);
    old_s = scale(&old_s);
    old_t = scale(&old_t);
  }
  (old_r, old_s, old_t)
}

/// GCD of two coefficient vectors over GF(p), returned monic.
fn poly_gcd_coeffs_mod(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  let mut a: Vec<i128> = a.iter().map(|&x| mod_norm(x, p)).collect();
  let mut b: Vec<i128> = b.iter().map(|&x| mod_norm(x, p)).collect();
  trim_high(&mut a);
  trim_high(&mut b);
  while !is_zero_poly(&b) {
    let r = poly_rem_mod(&a, &b, p);
    a = b;
    b = r;
    trim_high(&mut b);
  }
  trim_high(&mut a);
  if !is_zero_poly(&a) {
    let inv = mod_inv(*a.last().unwrap(), p);
    for x in a.iter_mut() {
      *x = mod_norm(*x * inv, p);
    }
  }
  a
}

/// Rebuild a polynomial Sum c_i var^i from a coefficient vector (mod p).
pub(super) fn coeffs_to_poly(coeffs: &[i128], var: &str, p: i128) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();
  for (i, &c) in coeffs.iter().enumerate() {
    let c = mod_norm(c, p);
    if c == 0 {
      continue;
    }
    let term = if i == 0 {
      Expr::Integer(c)
    } else {
      let pow = if i == 1 {
        Expr::Identifier(var.to_string())
      } else {
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::Identifier(var.to_string()),
            Expr::Integer(i as i128),
          ]
          .into(),
        }
      };
      if c == 1 {
        pow
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(c), pow].into(),
        }
      }
    };
    terms.push(term);
  }
  if terms.is_empty() {
    return Expr::Integer(0);
  }
  let expr = if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  };
  crate::evaluator::evaluate_expr_to_expr(&expr).unwrap_or(expr)
}

/// Compute GCD of two polynomials using the Euclidean algorithm
fn poly_gcd_pair(
  p: &Expr,
  q: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let p_str = expr_to_string(p);
  let q_str = expr_to_string(q);

  // Handle zero cases
  if p_str == "0" {
    return normalize_poly_sign(q, var);
  }
  if q_str == "0" {
    return normalize_poly_sign(p, var);
  }

  // Check degrees in the main variable
  let p_deg = max_power_int(&expand_and_combine(p), var).unwrap_or(0);
  let q_deg = max_power_int(&expand_and_combine(q), var).unwrap_or(0);

  if p_deg == 0 && q_deg == 0 {
    // Both are constants w.r.t. this variable - compute numeric GCD
    return integer_gcd_two(p, q);
  }

  // Extract integer content of both polynomials
  let p_content = poly_integer_content(p, var)?;
  let q_content = poly_integer_content(q, var)?;
  let content_gcd = integer_gcd_two(&p_content, &q_content)?;

  // Make both polynomials primitive (divide by content)
  let p_prim = poly_divide_by_constant(p, &p_content)?;
  let q_prim = poly_divide_by_constant(q, &q_content)?;

  // Euclidean algorithm on primitive parts
  let mut a = p_prim;
  let mut b = q_prim;

  for _ in 0..100 {
    let b_eval = crate::evaluator::evaluate_expr_to_expr(&b)?;
    let b_str = expr_to_string(&b_eval);
    if b_str == "0" {
      break;
    }

    // Try polynomial division; if it fails, the polynomials are coprime
    // (this handles multivariate cases where symbolic coefficients cause issues)
    match poly_divide_symbolic(&a, &b_eval, var) {
      Ok((_, remainder)) => {
        let remainder = crate::evaluator::evaluate_expr_to_expr(&remainder)?;
        a = b_eval;
        b = remainder;
      }
      Err(_) => {
        // Division failed - polynomials are likely coprime in this variable
        return Ok(content_gcd);
      }
    }
  }

  // Make result primitive with positive leading coefficient
  let a = crate::evaluator::evaluate_expr_to_expr(&a)?;

  // If the result has degree 0 in the main variable and is symbolic
  // (not a pure number), the polynomials are coprime in this variable
  let a_deg = max_power_int(&expand_and_combine(&a), var).unwrap_or(0);
  if a_deg == 0 && !is_numeric_expr(&a) {
    return Ok(content_gcd);
  }

  let a_prim = make_primitive(&a, var)?;

  // Multiply by the content GCD
  let content_str = expr_to_string(&content_gcd);
  if content_str == "1" {
    return Ok(a_prim);
  }

  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![content_gcd, a_prim].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&expand_and_combine(
    &crate::evaluator::evaluate_expr_to_expr(&result)?,
  ))
}

/// Extract the integer content (GCD of all numeric coefficients) of a polynomial.
/// For polynomials with symbolic coefficients, returns 1.
fn poly_integer_content(
  poly: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let expanded = expand_and_combine(poly);
  let deg = max_power_int(&expanded, var).unwrap_or(0);

  let mut int_coeffs = Vec::new();
  for i in 0..=deg {
    let c = coefficient_ast(&[
      poly.clone(),
      Expr::Identifier(var.to_string()),
      Expr::Integer(i),
    ])?;
    let c = crate::evaluator::evaluate_expr_to_expr(&c)?;
    let c_str = expr_to_string(&c);
    if c_str != "0" {
      // Only include if it's a pure integer or rational
      match &c {
        Expr::Integer(_) | Expr::BigInteger(_) => {
          let abs = Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![c].into(),
          };
          int_coeffs.push(crate::evaluator::evaluate_expr_to_expr(&abs)?);
        }
        Expr::BinaryOp {
          op: BinaryOperator::Divide,
          ..
        } => {
          // Rational number - include it
          let abs = Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![c].into(),
          };
          int_coeffs.push(crate::evaluator::evaluate_expr_to_expr(&abs)?);
        }
        _ => {
          // Symbolic coefficient - content is 1
          return Ok(Expr::Integer(1));
        }
      }
    }
  }

  if int_coeffs.is_empty() {
    return Ok(Expr::Integer(1));
  }
  if int_coeffs.len() == 1 {
    return Ok(int_coeffs.into_iter().next().unwrap());
  }

  let mut result = int_coeffs[0].clone();
  for c in &int_coeffs[1..] {
    result = integer_gcd_two(&result, c)?;
  }
  Ok(result)
}

/// Divide a polynomial by a constant
fn poly_divide_by_constant(
  poly: &Expr,
  constant: &Expr,
) -> Result<Expr, InterpreterError> {
  let c_str = expr_to_string(constant);
  if c_str == "1" {
    return Ok(poly.clone());
  }

  let div = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![constant.clone(), Expr::Integer(-1)].into(),
      },
      poly.clone(),
    ]
    .into(),
  };
  let result = crate::evaluator::evaluate_expr_to_expr(&div)?;
  crate::evaluator::evaluate_expr_to_expr(&expand_and_combine(&result))
}

/// Make a polynomial primitive (content = 1) with positive leading coefficient
fn make_primitive(poly: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  let content = poly_integer_content(poly, var)?;
  let prim = poly_divide_by_constant(poly, &content)?;
  normalize_poly_sign(&prim, var)
}

/// Ensure polynomial has positive leading coefficient
fn normalize_poly_sign(
  poly: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  let expanded = expand_and_combine(poly);
  let deg = max_power_int(&expanded, var).unwrap_or(0);

  let lead_coeff = coefficient_ast(&[
    poly.clone(),
    Expr::Identifier(var.to_string()),
    Expr::Integer(deg),
  ])?;
  let lead_coeff = crate::evaluator::evaluate_expr_to_expr(&lead_coeff)?;

  let is_negative = match &lead_coeff {
    Expr::Integer(n) => *n < 0,
    Expr::Real(f) => *f < 0.0,
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      ..
    } => matches!(left.as_ref(), Expr::Integer(n) if *n < 0),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      ..
    } => true,
    _ => {
      let s = expr_to_string(&lead_coeff);
      s.starts_with('-')
    }
  };

  if is_negative {
    let neg = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), poly.clone()].into(),
    };
    crate::evaluator::evaluate_expr_to_expr(&neg)
  } else {
    Ok(poly.clone())
  }
}

/// Check if an expression is purely numeric (no symbolic variables)
fn is_numeric_expr(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => true,
    Expr::BinaryOp { left, right, .. } => {
      is_numeric_expr(left) && is_numeric_expr(right)
    }
    Expr::UnaryOp { operand, .. } => is_numeric_expr(operand),
    Expr::FunctionCall { name, args } => {
      // Rational numbers like Times[-1, Power[...]] are numeric
      (name == "Times"
        || name == "Plus"
        || name == "Power"
        || name == "Rational")
        && args.iter().all(is_numeric_expr)
    }
    _ => false,
  }
}

/// Compute integer GCD of multiple values
fn integer_gcd_multi(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut result = args[0].clone();
  for arg in &args[1..] {
    result = integer_gcd_two(&result, arg)?;
  }
  Ok(result)
}

/// Compute GCD of two integer/rational expressions
fn integer_gcd_two(a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  let gcd_expr = Expr::FunctionCall {
    name: "GCD".to_string(),
    args: vec![a.clone(), b.clone()].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&gcd_expr)
}
