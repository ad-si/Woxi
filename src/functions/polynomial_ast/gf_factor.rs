//! Univariate polynomial factorization over GF(p) for the `Modulus -> p`
//! option of Factor, FactorList, IrreduciblePolynomialQ and PolynomialLCM.
//!
//! Uses char-p square-free decomposition followed by deterministic
//! Berlekamp splitting (Frobenius nullspace basis, then gcd(g, v - c)
//! sweeps over c in GF(p)), so the result is reproducible. Factors are
//! monic with coefficients normalized to [0, p) and sorted by degree,
//! then by ascending coefficient vector — matching wolframscript.

use super::*;

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
    crate::evaluator::evaluate_expr_to_expr(&call1("Expand", expr.clone()))?;
  let vars =
    crate::functions::math_ast::variables_ast(std::slice::from_ref(&expanded))?;
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
  let var_expr = id_expr(var);
  let mut sum = Expr::Integer(0);
  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let pow = match i {
      0 => Expr::Integer(1),
      1 => var_expr.clone(),
      _ => pow2(var_expr.clone(), Expr::Integer(i as i128)),
    };
    let term = times2(Expr::Integer(c), pow);
    sum = plus2(sum, term);
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
      pow2(base, Expr::Integer(*m as i128))
    };
    product = times2(product, factor);
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

/// FactorSquareFree[poly, Modulus -> p]: the square-free decomposition
/// over GF(p), each part monic, with the leading coefficient in front.
pub fn factor_square_free_modulus(
  expr: &Expr,
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  if !(2..=MAX_MODULUS).contains(&p)
    || !(2..p).take_while(|d| d * d <= p).all(|d| p % d != 0)
  {
    return Ok(None);
  }
  let Some((var, coeffs)) = univariate_int_coeffs(expr)? else {
    return Ok(None);
  };
  let mut f: Vec<i128> = coeffs.iter().map(|c| c.rem_euclid(p)).collect();
  trim(&mut f);
  if is_zero(&f) {
    return Ok(Some(Expr::Integer(0)));
  }
  let constant = *f.last().unwrap();
  if deg(&f) == 0 {
    return Ok(Some(Expr::Integer(constant)));
  }
  let mut parts = squarefree_decomposition(&make_monic(&f, p), p);
  parts.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0)));
  Ok(Some(gf_factored_expr(constant, &parts, &var)?))
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
    let gcd = call("PolynomialGCD", vec![result.clone(), arg.clone()]);
    let quotient = call(
      "Cancel",
      vec![times(vec![result, pow(gcd, Expr::Integer(-1))])],
    );
    let quotient = crate::evaluator::evaluate_expr_to_expr(&quotient)
      .unwrap_or_else(|_| quotient.clone());
    let lcm = times(vec![quotient, arg.clone()]);
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
  let product = times2(gf_poly_to_expr(&q, &var)?, gf_poly_to_expr(&cb, &var)?);
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&product)?))
}

/// Together[expr, Modulus -> p] for a univariate rational function, the way
/// wolframscript does it. A single fraction is combined over Q and its
/// coefficients reduced mod p, keeping its shape: `1/(2 x)` stays, and only
/// a common GF(p) factor of numerator and denominator cancels (`(x^2 + 1)/
/// (x + 1)` is `1 + x` mod 2). A sum is combined over GF(p): every term's
/// denominator is made monic, so its constants are inverted
/// (`x + 3/(2 x)` is `(4 + x^2)/x` mod 5); the common denominator is built
/// from a gcd-free basis of the terms' square-free parts (`x (2 + 3 x +
/// x^2)`, `(1 + x) (2 + x)`), and the numerator is its leading coefficient
/// times a monic polynomial (`(6 (3 + x))/(x (4 + x))`). None when `p` is
/// not a supported prime or the input is not a univariate rational function
/// with integer coefficients (or a denominator vanishes mod p).
pub fn together_modulus(
  expr: &Expr,
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  if !(2..=MAX_MODULUS).contains(&p)
    || !(2..p).take_while(|d| d * d <= p).all(|d| p % d != 0)
  {
    return Ok(None);
  }
  match expr {
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      together_modulus_sum(args, p)
    }
    _ => together_modulus_fraction(expr, p),
  }
}

/// A term's numerator and denominator after Together over Q, as
/// coefficient vectors of one variable (`""` for a constant).
type ModFraction = (String, Vec<i128>, Vec<i128>, Expr, Expr);

fn fraction_parts(
  expr: &Expr,
) -> Result<Option<ModFraction>, InterpreterError> {
  let combined = super::together_ast(std::slice::from_ref(expr))?;
  let num = crate::evaluator::evaluate_expr_to_expr(&call1(
    "Numerator",
    combined.clone(),
  ))?;
  let den =
    crate::evaluator::evaluate_expr_to_expr(&call1("Denominator", combined))?;
  let (Some((nvar, n)), Some((dvar, d))) =
    (univariate_int_coeffs(&num)?, univariate_int_coeffs(&den)?)
  else {
    return Ok(None);
  };
  let var = match (nvar.is_empty(), dvar.is_empty()) {
    (_, true) => nvar,
    (true, false) => dvar,
    (false, false) if nvar == dvar => nvar,
    _ => return Ok(None),
  };
  Ok(Some((var, n, d, num, den)))
}

fn reduce_mod(v: &[i128], p: i128) -> Vec<i128> {
  let mut out: Vec<i128> = v.iter().map(|c| c.rem_euclid(p)).collect();
  trim(&mut out);
  out
}

/// `expr`'s own numerator and denominator when both are univariate integer
/// polynomials in one variable, without combining anything.
fn direct_fraction_parts(
  expr: &Expr,
) -> Result<Option<(String, Vec<i128>, Vec<i128>)>, InterpreterError> {
  let num =
    crate::evaluator::evaluate_expr_to_expr(&call1("Numerator", expr.clone()))?;
  let den = crate::evaluator::evaluate_expr_to_expr(&call1(
    "Denominator",
    expr.clone(),
  ))?;
  let (Some((nvar, n)), Some((dvar, d))) =
    (univariate_int_coeffs(&num)?, univariate_int_coeffs(&den)?)
  else {
    return Ok(None);
  };
  let var = match (nvar.is_empty(), dvar.is_empty()) {
    (_, true) => nvar,
    (true, false) => dvar,
    (false, false) if nvar == dvar => nvar,
    _ => return Ok(None),
  };
  Ok(Some((var, n, d)))
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  if b == 0 { a.abs() } else { gcd_i128(b, a % b) }
}

fn together_modulus_fraction(
  expr: &Expr,
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  let (var, n, d) = match direct_fraction_parts(expr)? {
    Some(parts) => parts,
    None => match fraction_parts(expr)? {
      Some((var, n, d, ..)) => (var, n, d),
      None => return Ok(None),
    },
  };
  let (n, d) = (reduce_mod(&n, p), reduce_mod(&d, p));
  if is_zero(&d) {
    return Ok(None);
  }
  if is_zero(&n) {
    return Ok(Some(Expr::Integer(0)));
  }
  let g = poly_gcd(&n, &d, p);
  let (n, d) = if deg(&g) > 0 {
    (poly_div_exact(&n, &g, p), poly_div_exact(&d, &g, p))
  } else {
    (n, d)
  };
  if deg(&d) == 0 {
    let inv = mod_inv(d[0], p);
    let n: Vec<i128> = n.iter().map(|&c| c * inv).collect();
    return Ok(Some(gf_poly_to_expr(&reduce_mod(&n, p), &var)?));
  }
  // The denominator shows its integer content outside: `2*(2 + x^2)`.
  let content = d.iter().fold(0, |acc, &c| gcd_i128(acc, c));
  let den_expr = if content > 1 {
    let primitive: Vec<i128> = d.iter().map(|&c| c / content).collect();
    times2(Expr::Integer(content), gf_poly_to_expr(&primitive, &var)?)
  } else {
    gf_poly_to_expr(&d, &var)?
  };
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&div2(
    gf_poly_to_expr(&n, &var)?,
    den_expr,
  ))?))
}

/// How often `piece` divides `poly`.
fn multiplicity(poly: &[i128], piece: &[i128], p: i128) -> usize {
  let mut count = 0;
  let mut rest = poly.to_vec();
  while deg(&rest) >= deg(piece) && is_zero(&poly_rem(&rest, piece, p)) {
    rest = poly_div_exact(&rest, piece, p);
    count += 1;
  }
  count
}

/// Add the square-free monic `q` to a gcd-free basis, splitting it and any
/// element it shares a factor with.
fn refine_basis(basis: &mut Vec<Vec<i128>>, q: Vec<i128>, p: i128) {
  let mut pending = vec![q];
  while let Some(q) = pending.pop() {
    if deg(&q) == 0 {
      continue;
    }
    match basis.iter().position(|b| deg(&poly_gcd(b, &q, p)) > 0) {
      Some(i) => {
        let b = basis.remove(i);
        let g = poly_gcd(&b, &q, p);
        pending.push(poly_div_exact(&b, &g, p));
        pending.push(poly_div_exact(&q, &g, p));
        pending.push(g);
      }
      None => basis.push(q),
    }
  }
}

fn together_modulus_sum(
  terms: &[Expr],
  p: i128,
) -> Result<Option<Expr>, InterpreterError> {
  let mut var = String::new();
  let mut fractions: Vec<(Vec<i128>, Vec<i128>)> = Vec::new();
  for term in terms {
    let Some((v, n, d, ..)) = fraction_parts(term)? else {
      return Ok(None);
    };
    if !v.is_empty() {
      if !var.is_empty() && var != v {
        return Ok(None);
      }
      var = v;
    }
    let d = reduce_mod(&d, p);
    if is_zero(&d) {
      return Ok(None);
    }
    // A monic denominator: its leading coefficient's inverse moves up.
    let inv = mod_inv(*d.last().unwrap(), p);
    let n: Vec<i128> = n.iter().map(|&c| c * inv).collect();
    fractions.push((reduce_mod(&n, p), make_monic(&d, p)));
  }
  let mut basis: Vec<Vec<i128>> = Vec::new();
  for (_, d) in &fractions {
    for (piece, _) in squarefree_decomposition(d, p) {
      refine_basis(&mut basis, piece, p);
    }
  }
  let mut den = vec![1i128];
  for b in &basis {
    let m = fractions
      .iter()
      .map(|(_, d)| multiplicity(d, b, p))
      .max()
      .unwrap_or(0);
    for _ in 0..m {
      den = poly_mul(&den, b, p);
    }
  }
  let mut num = vec![0i128];
  for (n, d) in &fractions {
    let scaled = poly_mul(n, &poly_div_exact(&den, d, p), p);
    let len = num.len().max(scaled.len());
    num.resize(len, 0);
    for (i, c) in scaled.iter().enumerate() {
      num[i] = (num[i] + c).rem_euclid(p);
    }
    trim(&mut num);
  }
  if is_zero(&num) {
    return Ok(Some(Expr::Integer(0)));
  }
  let g = poly_gcd(&num, &den, p);
  if deg(&g) > 0 {
    num = poly_div_exact(&num, &g, p);
    den = poly_div_exact(&den, &g, p);
  }
  if deg(&den) == 0 {
    return Ok(Some(gf_poly_to_expr(&num, &var)?));
  }
  let lc = *num.last().unwrap();
  let num_expr = if lc == 1 || deg(&num) == 0 {
    gf_poly_to_expr(&num, &var)?
  } else {
    times2(
      Expr::Integer(lc),
      gf_poly_to_expr(&make_monic(&num, p), &var)?,
    )
  };
  let mut den_factors = Vec::new();
  for b in &basis {
    let m = multiplicity(&den, b, p);
    if m > 0 {
      let base = gf_poly_to_expr(b, &var)?;
      den_factors.push(if m == 1 {
        base
      } else {
        pow2(base, Expr::Integer(m as i128))
      });
    }
  }
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&div2(
    num_expr,
    times(den_factors),
  ))?))
}

/// Quotient and remainder of a by b over GF(p) (b nonzero).
fn poly_divmod(a: &[i128], b: &[i128], p: i128) -> (Vec<i128>, Vec<i128>) {
  let db = deg(b);
  let inv_lc = mod_inv(*b.last().unwrap(), p);
  let mut r = a.to_vec();
  trim(&mut r);
  let mut q = vec![0i128; r.len().saturating_sub(db).max(1)];
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
  (q, r)
}

fn poly_sub(a: &[i128], b: &[i128], p: i128) -> Vec<i128> {
  let mut out = vec![0i128; a.len().max(b.len())];
  for (i, o) in out.iter_mut().enumerate() {
    let x = a.get(i).copied().unwrap_or(0);
    let y = b.get(i).copied().unwrap_or(0);
    *o = (x - y).rem_euclid(p);
  }
  trim(&mut out);
  out
}

/// The inverse of a modulo m over GF(p), when gcd(a, m) = 1.
fn poly_inverse_mod(a: &[i128], m: &[i128], p: i128) -> Option<Vec<i128>> {
  let (mut r0, mut r1) = (m.to_vec(), poly_rem(a, m, p));
  let (mut s0, mut s1) = (vec![0i128], vec![1i128]);
  while !is_zero(&r1) {
    let (q, r) = poly_divmod(&r0, &r1, p);
    let s = poly_sub(&s0, &poly_mul(&q, &s1, p), p);
    (r0, r1) = (r1, r);
    (s0, s1) = (s1, s);
  }
  if deg(&r0) != 0 {
    return None;
  }
  let inv = mod_inv(r0[0], p);
  let s: Vec<i128> = s0.iter().map(|&c| (c * inv).rem_euclid(p)).collect();
  Some(poly_rem(&s, m, p))
}

/// Apart[expr, Modulus -> p] (`square_free` false) and
/// ApartSquareFree[expr, Modulus -> p] (true): the polynomial part plus
/// one term c/f^k for each irreducible (or square-free) factor f of the
/// denominator and each power k up to its multiplicity, deg c < deg f.
pub fn apart_modulus(
  expr: &Expr,
  p: i128,
  square_free: bool,
) -> Result<Option<Expr>, InterpreterError> {
  if !(2..=MAX_MODULUS).contains(&p)
    || !(2..p).take_while(|d| d * d <= p).all(|d| p % d != 0)
  {
    return Ok(None);
  }
  let (var, n, d) = match direct_fraction_parts(expr)? {
    Some(parts) => parts,
    None => match fraction_parts(expr)? {
      Some((var, n, d, ..)) => (var, n, d),
      None => return Ok(None),
    },
  };
  let (n, d) = (reduce_mod(&n, p), reduce_mod(&d, p));
  if is_zero(&d) {
    return Ok(None);
  }
  if is_zero(&n) {
    return Ok(Some(Expr::Integer(0)));
  }
  let g = poly_gcd(&n, &d, p);
  let (n, d) = (poly_div_exact(&n, &g, p), poly_div_exact(&d, &g, p));
  // Make the denominator monic, moving its leading coefficient up.
  let lc_inv = mod_inv(*d.last().unwrap(), p);
  let n: Vec<i128> =
    reduce_mod(&n.iter().map(|&c| c * lc_inv).collect::<Vec<_>>(), p);
  let d = make_monic(&d, p);
  let (quotient, remainder) = poly_divmod(&n, &d, p);
  let mut terms: Vec<Expr> = Vec::new();
  if !is_zero(&quotient) {
    terms.push(gf_poly_to_expr(&quotient, &var)?);
  }
  if !is_zero(&remainder) {
    let parts: Vec<(Vec<i128>, usize)> = if square_free {
      squarefree_decomposition(&d, p)
    } else {
      match gf_factor_coeffs(&d, p) {
        Some((_, f)) => f,
        None => return Ok(None),
      }
    };
    for (f, e) in &parts {
      let mut power = vec![1i128];
      for _ in 0..*e {
        power = poly_mul(&power, f, p);
      }
      let cofactor = poly_div_exact(&d, &power, p);
      let Some(inv) = poly_inverse_mod(&cofactor, &power, p) else {
        return Ok(None);
      };
      // remainder / d = A / f^e + (rest) with A = remainder * cofactor⁻¹.
      let mut a = poly_rem(&poly_mul(&remainder, &inv, p), &power, p);
      // A = Σ c_k f^k: c_k / f^(e - k).
      let f_expr = gf_poly_to_expr(f, &var)?;
      let mut k = 0usize;
      while !is_zero(&a) {
        let (q, c) = poly_divmod(&a, f, p);
        if !is_zero(&c) {
          let den = if e - k == 1 {
            f_expr.clone()
          } else {
            pow2(f_expr.clone(), Expr::Integer((e - k) as i128))
          };
          terms.push(div2(gf_poly_to_expr(&c, &var)?, den));
        }
        a = q;
        k += 1;
      }
    }
  }
  if terms.is_empty() {
    return Ok(Some(Expr::Integer(0)));
  }
  Ok(Some(crate::evaluator::evaluate_expr_to_expr(&plus(terms))?))
}
