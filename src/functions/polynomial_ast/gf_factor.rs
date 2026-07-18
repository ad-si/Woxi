//! Univariate polynomial factorization over GF(p) for the `Modulus -> p`
//! option of Factor, FactorList, IrreduciblePolynomialQ and PolynomialLCM.
//!
//! Uses char-p square-free decomposition followed by deterministic
//! Berlekamp splitting (Frobenius nullspace basis, then gcd(g, v - c)
//! sweeps over c in GF(p)), so the result is reproducible. Factors are
//! monic with coefficients normalized to [0, p) and sorted by degree,
//! then by ascending coefficient vector — matching wolframscript.

#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, bool_expr, unevaluated};

/// Keep the c-sweep in Berlekamp splitting bounded.
const MAX_MODULUS: i128 = 65_536;

// ─── GF(p) polynomial arithmetic (ascending trimmed coefficient vectors) ──

fn mod_inv(a: i128, p: i128) -> i128 {
  let (mut old_r, mut r) = (a.rem_euclid(p), p);
  let (mut old_s, mut s) = (1i128, 0i128);
  while r != 0 {
    let q = old_r / r;
    (old_r, r) = (r, old_r - q * r);
    (old_s, s) = (s, old_s - q * s);
  }
  old_s.rem_euclid(p)
}

fn poly_mul(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  if is_zero(a) || is_zero(b) {
    return vec![0];
  }
  let mut out = vec![0i128; a.len() + b.len() - 1];
  for (i, &x) in a.iter().enumerate() {
    if x == 0 {
      continue;
    }
    for (j, &y) in b.iter().enumerate() {
      out[i + j] = (out[i + j] + x * y).rem_euclid(p);
    }
  }
  trim(&mut out);
  out
}

/// Remainder of a by b (b nonzero).
fn poly_rem(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  let db = deg(b);
  let inv_lc = mod_inv(*b.last().unwrap(), p);
  let mut r = a.to_vec();
  while !is_zero(&r) && deg(&r) >= db {
    let dr = deg(&r);
    let q = (r[dr] * inv_lc).rem_euclid(p);
    for (i, &bc) in b.iter().enumerate() {
      let idx = dr - db + i;
      r[idx] = (r[idx] - q * bc).rem_euclid(p);
    }
    trim(&mut r);
    if dr == 0 {
      break;
    }
  }
  r
}

/// Exact quotient of a by b (assumes b | a).
fn poly_div_exact(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  let db = deg(b);
  let inv_lc = mod_inv(*b.last().unwrap(), p);
  let mut r = a.to_vec();
  let mut q = vec![0i128; a.len().saturating_sub(db)];
  while !is_zero(&r) && deg(&r) >= db {
    let dr = deg(&r);
    let c = (r[dr] * inv_lc).rem_euclid(p);
    q[dr - db] = c;
    for (i, &bc) in b.iter().enumerate() {
      let idx = dr - db + i;
      r[idx] = (r[idx] - c * bc).rem_euclid(p);
    }
    trim(&mut r);
    if dr == 0 {
      break;
    }
  }
  trim(&mut q);
  q
}

fn make_monic(v: &[i128], p: i128) -> Vec<i128> {
  let inv = mod_inv(*v.last().unwrap(), p);
  let mut out: Vec<i128> = v.iter().map(|&c| (c * inv).rem_euclid(p)).collect();
  trim(&mut out);
  out
}

/// Monic gcd.
fn poly_gcd(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  let mut a = a.to_vec();
  let mut b = b.to_vec();
  while !is_zero(&b) {
    let r = poly_rem(&a, &b, p);
    a = b;
    b = r;
  }
  if is_zero(&a) { a } else { make_monic(&a, p) }
}

fn derivative(v: &[i128], p: i128) -> Vec<i128> {
  if v.len() <= 1 {
    return vec![0];
  }
  let mut out: Vec<i128> = v
    .iter()
    .enumerate()
    .skip(1)
    .map(|(i, &c)| (c * i as i128).rem_euclid(p))
    .collect();
  trim(&mut out);
  out
}

/// b^e mod m over GF(p)[x].
fn poly_pow_mod(b: &[i128], e: i128, m: &[i128], p: i128) -> Vec<i128> {
  let mut result = vec![1i128];
  let mut base = poly_rem(b, m, p);
  let mut e = e;
  while e > 0 {
    if e & 1 == 1 {
      result = poly_rem(&poly_mul(&result, &base, p), m, p);
    }
    base = poly_rem(&poly_mul(&base, &base, p), m, p);
    e >>= 1;
  }
  result
}

// ─── factorization ──────────────────────────────────────────────────

/// Square-free decomposition of a monic polynomial in characteristic p:
/// list of (monic square-free part, multiplicity).
fn squarefree_decomposition(f: &[i128], p: i128) -> Vec<(Vec<i128>, usize)> {
  let mut result: Vec<(Vec<i128>, usize)> = Vec::new();
  if deg(f) == 0 {
    return result;
  }
  let fp = derivative(f, p);
  if is_zero(&fp) {
    // f = g(x^p); in GF(p) every coefficient is its own p-th root.
    let g: Vec<i128> = f.iter().step_by(p as usize).copied().collect();
    for (h, m) in squarefree_decomposition(&g, p) {
      result.push((h, m * p as usize));
    }
    return result;
  }
  let mut c = poly_gcd(f, &fp, p);
  let mut w = poly_div_exact(f, &c, p);
  let mut i = 1usize;
  while deg(&w) > 0 {
    let y = poly_gcd(&w, &c, p);
    let z = poly_div_exact(&w, &y, p);
    if deg(&z) > 0 {
      result.push((z, i));
    }
    c = poly_div_exact(&c, &y, p);
    w = y;
    i += 1;
  }
  if deg(&c) > 0 {
    let g: Vec<i128> = c.iter().step_by(p as usize).copied().collect();
    for (h, m) in squarefree_decomposition(&g, p) {
      result.push((h, m * p as usize));
    }
  }
  result
}

/// Nullspace basis of the transposed (Frobenius - identity) matrix over
/// GF(p) — the Berlekamp subalgebra of GF(p)[x]/(f).
fn berlekamp_nullspace(f: &[i128], p: i128) -> Vec<Vec<i128>> {
  let n = deg(f);
  // rows[i] = coefficients of x^(p*i) mod f
  let xp = poly_pow_mod(&[0, 1], p, f, p);
  let mut rows: Vec<Vec<i128>> = Vec::with_capacity(n);
  let mut cur = vec![1i128];
  for _ in 0..n {
    let mut row = vec![0i128; n];
    for (k, &c) in cur.iter().enumerate() {
      row[k] = c;
    }
    rows.push(row);
    cur = poly_rem(&poly_mul(&cur, &xp, p), f, p);
  }
  // v is in the subalgebra iff v * (R - I) = 0; build M = (R - I)^T and
  // find its nullspace via Gauss-Jordan.
  let mut m = vec![vec![0i128; n]; n];
  for (i, row) in rows.iter().enumerate() {
    for (j, &val) in row.iter().enumerate() {
      let mut v = val;
      if i == j {
        v -= 1;
      }
      m[j][i] = v.rem_euclid(p);
    }
  }
  let mut pivot_cols: Vec<usize> = Vec::new();
  let mut rank = 0usize;
  for col in 0..n {
    if let Some(pr) = (rank..n).find(|&r| m[r][col] != 0) {
      m.swap(rank, pr);
      let inv = mod_inv(m[rank][col], p);
      for j in 0..n {
        m[rank][j] = (m[rank][j] * inv).rem_euclid(p);
      }
      for r in 0..n {
        if r != rank && m[r][col] != 0 {
          let factor = m[r][col];
          for j in 0..n {
            m[r][j] = (m[r][j] - factor * m[rank][j]).rem_euclid(p);
          }
        }
      }
      pivot_cols.push(col);
      rank += 1;
    }
  }
  let mut basis = Vec::new();
  for col in 0..n {
    if pivot_cols.contains(&col) {
      continue;
    }
    let mut v = vec![0i128; n];
    v[col] = 1;
    for (r, &pc) in pivot_cols.iter().enumerate() {
      v[pc] = (-m[r][col]).rem_euclid(p);
    }
    trim(&mut v);
    basis.push(v);
  }
  basis
}

/// Factor a monic square-free polynomial into monic irreducibles.
fn berlekamp_factor(f: &[i128], p: i128) -> Vec<Vec<i128>> {
  if deg(f) <= 1 {
    return vec![f.to_vec()];
  }
  let basis = berlekamp_nullspace(f, p);
  let r = basis.len();
  let mut factors = vec![f.to_vec()];
  if r == 1 {
    return factors;
  }
  'outer: for v in &basis {
    if deg(v) == 0 {
      continue; // the constant basis vector never splits anything
    }
    let mut next: Vec<Vec<i128>> = Vec::new();
    for g in factors.drain(..) {
      if deg(&g) <= 1 {
        next.push(g);
        continue;
      }
      let mut remaining = g;
      for c in 0..p {
        if deg(&remaining) <= 1 {
          break;
        }
        let mut shifted = poly_rem(v, &remaining, p);
        shifted[0] = (shifted[0] - c).rem_euclid(p);
        trim(&mut shifted);
        if is_zero(&shifted) {
          continue;
        }
        let h = poly_gcd(&remaining, &shifted, p);
        if deg(&h) > 0 && deg(&h) < deg(&remaining) {
          remaining = poly_div_exact(&remaining, &h, p);
          next.push(h);
        }
      }
      next.push(remaining);
    }
    factors = next;
    if factors.len() == r {
      break 'outer;
    }
  }
  factors
}

/// Full factorization over GF(p): (unit constant, sorted monic factors
/// with multiplicities). None for the zero polynomial mod p.
pub(super) fn gf_factor_coeffs(
  coeffs: &[i128],
  p: i128,
) -> Option<(i128, Vec<(Vec<i128>, usize)>)> {
  let mut f: Vec<i128> = coeffs.iter().map(|c| c.rem_euclid(p)).collect();
  trim(&mut f);
  if is_zero(&f) {
    return None;
  }
  let constant = *f.last().unwrap();
  if deg(&f) == 0 {
    return Some((constant, vec![]));
  }
  let monic = make_monic(&f, p);
  let mut factors: Vec<(Vec<i128>, usize)> = Vec::new();
  for (part, mult) in squarefree_decomposition(&monic, p) {
    for irr in berlekamp_factor(&part, p) {
      factors.push((irr, mult));
    }
  }
  factors.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0)));
  Some((constant, factors))
}

// ─── Expr integration ───────────────────────────────────────────────

/// Recognize `expr` as a univariate integer-coefficient polynomial;
/// returns its variable name and ascending coefficients.
fn univariate_int_coeffs(
  expr: &Expr,
) -> Result<Option<(String, Vec<i128>)>, InterpreterError> {
  use crate::functions::math_ast::expr_to_i128;
  let expanded =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Expand".to_string(),
      args: vec![expr.clone()].into(),
    })?;
  let vars = crate::functions::math_ast::variables_ast(&[expanded.clone()])?;
  let Expr::List(ref vars) = vars else {
    return Ok(None);
  };
  let var = match vars.len() {
    0 => {
      // A constant: report it under a dummy variable.
      return Ok(expr_to_i128(&expanded).map(|c| (String::new(), vec![c])));
    }
    1 => match &vars[0] {
      Expr::Identifier(name) => name.clone(),
      _ => return Ok(None),
    },
    _ => return Ok(None),
  };
  let Some(deg) = super::max_power_int(&expanded, &var) else {
    return Ok(None);
  };
  let var_expr = Expr::Identifier(var.clone());
  let mut coeffs = Vec::with_capacity(deg as usize + 1);
  for i in 0..=deg {
    let c = super::coefficient_ast(&[
      expanded.clone(),
      var_expr.clone(),
      Expr::Integer(i),
    ])?;
    match expr_to_i128(&crate::evaluator::evaluate_expr_to_expr(&c)?) {
      Some(v) => coeffs.push(v),
      None => return Ok(None),
    }
  }
  Ok(Some((var, coeffs)))
}

fn gf_poly_to_expr(
  coeffs: &[i128],
  var: &str,
) -> Result<Expr, InterpreterError> {
  let var_expr = Expr::Identifier(var.to_string());
  let mut sum = Expr::Integer(0);
  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let pow = match i {
      0 => Expr::Integer(1),
      1 => var_expr.clone(),
      _ => Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(var_expr.clone()),
        right: Box::new(Expr::Integer(i as i128)),
      },
    };
    let term = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(c)),
      right: Box::new(pow),
    };
    sum = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(sum),
      right: Box::new(term),
    };
  }
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// The factored `Times` presentation used by Factor and PolynomialLCM.
fn gf_factored_expr(
  constant: i128,
  factors: &[(Vec<i128>, usize)],
  var: &str,
) -> Result<Expr, InterpreterError> {
  let mut product = if constant == 1 && !factors.is_empty() {
    Expr::Integer(1)
  } else {
    Expr::Integer(constant)
  };
  for (f, m) in factors {
    let base = gf_poly_to_expr(f, var)?;
    let factor = if *m == 1 {
      base
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(Expr::Integer(*m as i128)),
      }
    };
    product = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(product),
      right: Box::new(factor),
    };
  }
  crate::evaluator::evaluate_expr_to_expr(&product)
}

/// Factor[poly, Modulus -> p] — None when the input is not a univariate
/// integer polynomial (the caller then leaves the call unevaluated).
pub fn factor_modulus(
  expr: &Expr,
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  if !(2..=MAX_MODULUS).contains(&p) {
    return Ok(None);
  }
  let Some((var, coeffs)) = univariate_int_coeffs(expr)? else {
    return Ok(None);
  };
  let Some((constant, factors)) = gf_factor_coeffs(&coeffs, p) else {
    return Ok(Some(Expr::Integer(0)));
  };
  Ok(Some(gf_factored_expr(constant, &factors, &var)?))
}

/// FactorList[poly, Modulus -> p].
pub fn factor_list_modulus(
  expr: &Expr,
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  if !(2..=MAX_MODULUS).contains(&p) {
    return Ok(None);
  }
  let Some((var, coeffs)) = univariate_int_coeffs(expr)? else {
    return Ok(None);
  };
  let pair = |a: Expr, m: i128| Expr::List(vec![a, Expr::Integer(m)].into());
  let Some((constant, factors)) = gf_factor_coeffs(&coeffs, p) else {
    return Ok(Some(Expr::List(vec![pair(Expr::Integer(0), 1)].into())));
  };
  let mut items = vec![pair(Expr::Integer(constant), 1)];
  for (f, m) in &factors {
    items.push(pair(gf_poly_to_expr(f, &var)?, *m as i128));
  }
  Ok(Some(Expr::List(items.into())))
}

/// IrreduciblePolynomialQ[poly, Modulus -> p]: exactly one irreducible
/// factor of multiplicity 1 (constants and powers are not irreducible).
pub fn irreducible_polynomial_q_modulus(
  expr: &Expr,
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  if !(2..=MAX_MODULUS).contains(&p) {
    return Ok(None);
  }
  let Some((_, coeffs)) = univariate_int_coeffs(expr)? else {
    return Ok(None);
  };
  let irreducible = match gf_factor_coeffs(&coeffs, p) {
    Some((_, factors)) => factors.len() == 1 && factors[0].1 == 1,
    None => false,
  };
  Ok(Some(bool_expr(irreducible)))
}

/// PrimitivePolynomialQ[poly, p]: true when `poly` is primitive over GF(p),
/// i.e. the multiplicative order of x in GF(p)[x]/(poly) equals p^d - 1 (with
/// d = deg poly). That order test also implies irreducibility, so constants,
/// reducible polynomials, and polynomials with a zero constant term are not
/// primitive. Requires a prime modulus 2 <= p <= MAX_MODULUS; anything else
/// (composite p, multivariate, non-integer coefficients) stays unevaluated.
pub fn primitive_polynomial_q_modulus(
  expr: &Expr,
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  if !(2..=MAX_MODULUS).contains(&p)
    || !crate::functions::math_ast::is_prime_i128(p)
  {
    return Ok(None);
  }
  let Some((_, coeffs)) = univariate_int_coeffs(expr)? else {
    return Ok(None);
  };
  match is_primitive_over_gf(&coeffs, p) {
    Some(primitive) => Ok(Some(bool_expr(primitive))),
    None => Ok(None),
  }
}

/// Distinct prime factors of `n` by trial division.
fn distinct_prime_factors(mut n: i128) -> Vec<i128> {
  let mut factors = Vec::new();
  let mut q = 2i128;
  while q * q <= n {
    if n % q == 0 {
      factors.push(q);
      while n % q == 0 {
        n /= q;
      }
    }
    q += 1;
  }
  if n > 1 {
    factors.push(n);
  }
  factors
}

/// Core primitivity test on ascending integer coefficients. Returns None when
/// p^d overflows i128 (too large for the order arithmetic).
fn is_primitive_over_gf(coeffs: &[i128], p: i128) -> Option<bool> {
  let mut c: Vec<i128> = coeffs.iter().map(|&v| v.rem_euclid(p)).collect();
  trim(&mut c);
  let d = deg(&c);
  // Constants and the zero polynomial are never primitive.
  if d == 0 {
    return Some(false);
  }
  // A zero constant term means x divides the polynomial: reducible / not a
  // unit, so not primitive.
  if c[0] == 0 {
    return Some(false);
  }
  // Work in GF(p)[x]/(f) with f monic, so x^n is well defined.
  let f = make_monic(&c, p);
  // n = p^d - 1 is the order of the multiplicative group of GF(p^d).
  let mut pd: i128 = 1;
  for _ in 0..d {
    pd = pd.checked_mul(p)?;
  }
  let n = pd - 1;
  let x = [0i128, 1];
  let one = vec![1i128];
  // Primitive iff ord(x) = n: x^n = 1 and x^(n/q) != 1 for every prime q | n.
  // poly_pow_mod reduces x mod f first, so this also covers the degree-1 case.
  if poly_pow_mod(&x, n, &f, p) != one {
    return Some(false);
  }
  for q in distinct_prime_factors(n) {
    if poly_pow_mod(&x, n / q, &f, p) == one {
      return Some(false);
    }
  }
  Some(true)
}

/// PolynomialLCM[polys..., Modulus -> p]: the two-polynomial modular form
/// is computed over GF(p); without a modulus the pairwise Cancel-based
/// fold keeps wolframscript's factored display. The n-ary modular form is
/// left unevaluated — wolframscript's grouping of those products is
/// internal to its iterative lcm and not reproducible.
pub fn polynomial_lcm_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || unevaluated("PolynomialLCM", args);
  let mut pos: Vec<Expr> = Vec::new();
  let mut modulus: Option<i128> = None;
  for a in args {
    if let Some(p) = extract_modulus_option(a) {
      modulus = Some(p);
    } else {
      pos.push(a.clone());
    }
  }
  if let Some(p) = modulus {
    if pos.len() != 2 {
      return Ok(unevaluated());
    }
    return match polynomial_lcm_modulus(&pos[0], &pos[1], p)? {
      Some(e) => Ok(e),
      None => Ok(unevaluated()),
    };
  }
  if pos.len() < 2 {
    return Ok(unevaluated());
  }
  // LCM(p1, p2) = (p1 / gcd) * p2, kept as an unexpanded product to match
  // Wolfram's factored display (e.g. (-1 + x)*(1 + x), not -1 + x^2).
  // For more than 2 args, fold pairwise.
  let mut result = pos[0].clone();
  for arg in &pos[1..] {
    let gcd = Expr::FunctionCall {
      name: "PolynomialGCD".to_string(),
      args: vec![result.clone(), arg.clone()].into(),
    };
    let quotient = Expr::FunctionCall {
      name: "Cancel".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          result,
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![gcd, Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      }]
      .into(),
    };
    let quotient = crate::evaluator::evaluate_expr_to_expr(&quotient)
      .unwrap_or_else(|_| quotient.clone());
    let lcm = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![quotient, arg.clone()].into(),
    };
    result = crate::evaluator::evaluate_expr_to_expr(&lcm).unwrap_or(lcm);
  }
  Ok(result)
}

/// PolynomialLCM[a, b, Modulus -> p] — presented exactly as wolframscript
/// prints it: the unexpanded product of (a/gcd mod p) and (b mod p), with
/// the evaluator's canonical Times ordering (equal factors merge into a
/// power, e.g. (1 + x)^2).
fn polynomial_lcm_modulus(
  a: &Expr,
  b: &Expr,
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  if !(2..=MAX_MODULUS).contains(&p) {
    return Ok(None);
  }
  let (Some((va, ca)), Some((vb, cb))) =
    (univariate_int_coeffs(a)?, univariate_int_coeffs(b)?)
  else {
    return Ok(None);
  };
  if !va.is_empty() && !vb.is_empty() && va != vb {
    return Ok(None);
  }
  let var = if va.is_empty() { vb } else { va };
  let mut ca: Vec<i128> = ca.iter().map(|c| c.rem_euclid(p)).collect();
  let mut cb: Vec<i128> = cb.iter().map(|c| c.rem_euclid(p)).collect();
  trim(&mut ca);
  trim(&mut cb);
  if is_zero(&ca) || is_zero(&cb) {
    return Ok(Some(Expr::Integer(0)));
  }
  let g = poly_gcd(&ca, &cb, p);
  let q = poly_div_exact(&ca, &g, p);
  let product = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(gf_poly_to_expr(&q, &var)?),
    right: Box::new(gf_poly_to_expr(&cb, &var)?),
  };
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&product)?))
}
