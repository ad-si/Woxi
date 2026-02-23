use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr};

use super::coefficient::collect_additive_terms;
use super::coefficient::term_var_power_and_coeff;
use super::expand::expand_and_combine;

/// Decompose[poly, x] - decompose a polynomial into a composition of simpler polynomials.
/// Returns a list {g1, g2, ..., gn} such that poly(x) = g1(g2(...gn(x)...)).
pub fn decompose_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Decompose expects exactly 2 arguments".into(),
    ));
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Decompose".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Expand and combine the expression
  let expanded = expand_and_combine(&args[0]);

  // Extract polynomial coefficients as rationals
  let coeffs = match extract_rational_poly_coeffs(&expanded, &var) {
    Some(c) => c,
    None => {
      // Not a polynomial, return unevaluated
      return Ok(Expr::FunctionCall {
        name: "Decompose".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = poly_degree(&coeffs);
  if n <= 0 {
    // Constant or linear polynomial: return {poly}
    let expr = rational_coeffs_to_expr(&coeffs, &var);
    return Ok(Expr::List(vec![expr]));
  }

  // Perform decomposition
  let decomposition = decompose_poly(&coeffs);

  // Convert each polynomial back to expression
  let exprs: Vec<Expr> = decomposition
    .iter()
    .map(|c| rational_coeffs_to_expr(c, &var))
    .collect();

  Ok(Expr::List(exprs))
}

// ──── Rational arithmetic helpers ────────────────────────────────────

type Rat = (i128, i128); // (numerator, denominator), always reduced

fn rat_gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

fn rat_reduce(n: i128, d: i128) -> Rat {
  if d == 0 {
    return (n, d);
  }
  let g = rat_gcd(n, d);
  if d < 0 {
    (-n / g, -d / g)
  } else {
    (n / g, d / g)
  }
}

fn rat_add(a: Rat, b: Rat) -> Rat {
  let num = a.0 * b.1 + b.0 * a.1;
  let den = a.1 * b.1;
  rat_reduce(num, den)
}

fn rat_sub(a: Rat, b: Rat) -> Rat {
  rat_add(a, (-b.0, b.1))
}

fn rat_mul(a: Rat, b: Rat) -> Rat {
  let num = a.0 * b.0;
  let den = a.1 * b.1;
  rat_reduce(num, den)
}

fn rat_div(a: Rat, b: Rat) -> Rat {
  rat_mul(a, (b.1, b.0))
}

fn rat_is_zero(a: Rat) -> bool {
  a.0 == 0
}

const RAT_ZERO: Rat = (0, 1);
const RAT_ONE: Rat = (1, 1);

// ──── Polynomial arithmetic (rational coefficients) ──────────────────

/// Degree of polynomial (index of last non-zero coefficient), or -1 for zero poly
fn poly_degree(p: &[Rat]) -> i32 {
  for i in (0..p.len()).rev() {
    if !rat_is_zero(p[i]) {
      return i as i32;
    }
  }
  -1
}

/// Multiply two polynomials
fn poly_mul(a: &[Rat], b: &[Rat]) -> Vec<Rat> {
  if a.is_empty() || b.is_empty() {
    return vec![];
  }
  let da = poly_degree(a);
  let db = poly_degree(b);
  if da < 0 || db < 0 {
    return vec![RAT_ZERO];
  }
  let result_len = (da + db + 1) as usize;
  let mut result = vec![RAT_ZERO; result_len + 1];
  for i in 0..=da as usize {
    for j in 0..=db as usize {
      result[i + j] = rat_add(result[i + j], rat_mul(a[i], b[j]));
    }
  }
  result
}

/// Raise polynomial to a non-negative integer power
fn poly_pow(p: &[Rat], n: usize) -> Vec<Rat> {
  if n == 0 {
    return vec![RAT_ONE];
  }
  if n == 1 {
    return p.to_vec();
  }
  // Binary exponentiation
  let mut result = vec![RAT_ONE]; // 1
  let mut base = p.to_vec();
  let mut exp = n;
  while exp > 0 {
    if exp & 1 == 1 {
      result = poly_mul(&result, &base);
    }
    base = poly_mul(&base, &base);
    exp >>= 1;
  }
  result
}

/// Subtract b from a, extending with zeros if needed
fn poly_sub(a: &[Rat], b: &[Rat]) -> Vec<Rat> {
  let len = a.len().max(b.len());
  let mut result = vec![RAT_ZERO; len];
  for i in 0..len {
    let ai = if i < a.len() { a[i] } else { RAT_ZERO };
    let bi = if i < b.len() { b[i] } else { RAT_ZERO };
    result[i] = rat_sub(ai, bi);
  }
  result
}

/// Multiply polynomial by a scalar
fn poly_scale(p: &[Rat], s: Rat) -> Vec<Rat> {
  p.iter().map(|&c| rat_mul(c, s)).collect()
}

/// Check if polynomial is a monomial (only one non-zero coefficient, the leading one)
fn is_monomial(p: &[Rat]) -> bool {
  let deg = poly_degree(p);
  if deg < 0 {
    return true; // zero is considered monomial-like
  }
  for i in 0..deg as usize {
    if !rat_is_zero(p[i]) {
      return false;
    }
  }
  true
}

// ──── Coefficient extraction ────────────────────────────────────────

/// Extract polynomial coefficients as rationals from an expression.
/// Returns coeffs[i] = (num, den) for coefficient of var^i.
fn extract_rational_poly_coeffs(expr: &Expr, var: &str) -> Option<Vec<Rat>> {
  let terms = collect_additive_terms(expr);
  let mut max_pow: i128 = 0;
  let mut term_data: Vec<(i128, Rat)> = Vec::new();

  for term in &terms {
    let (power, coeff) = term_var_power_and_coeff(term, var);
    if power < 0 {
      return None;
    }
    let rat_coeff = expr_to_rat(&crate::functions::simplify(coeff))?;
    max_pow = max_pow.max(power);
    term_data.push((power, rat_coeff));
  }

  let mut coeffs = vec![RAT_ZERO; (max_pow + 1) as usize];
  for (power, c) in term_data {
    coeffs[power as usize] = rat_add(coeffs[power as usize], c);
  }

  Some(coeffs)
}

/// Try to convert an expression to a rational number
fn expr_to_rat(expr: &Expr) -> Option<Rat> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(rat_reduce(*n, *d))
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (n, d) = expr_to_rat(operand)?;
      Some((-n, d))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let l = expr_to_rat(left)?;
      let r = expr_to_rat(right)?;
      Some(rat_mul(l, r))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let l = expr_to_rat(left)?;
      let r = expr_to_rat(right)?;
      Some(rat_div(l, r))
    }
    _ => None,
  }
}

// ──── Coefficient-to-expression conversion ──────────────────────────

/// Convert rational polynomial coefficients to an expression.
fn rational_coeffs_to_expr(coeffs: &[Rat], var: &str) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();

  for (i, &(n, d)) in coeffs.iter().enumerate() {
    if n == 0 {
      continue;
    }

    let coeff_expr = crate::functions::math_ast::make_rational(n, d);

    let var_part = if i == 0 {
      None
    } else if i == 1 {
      Some(Expr::Identifier(var.to_string()))
    } else {
      Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier(var.to_string())),
        right: Box::new(Expr::Integer(i as i128)),
      })
    };

    let term = match var_part {
      None => coeff_expr,
      Some(v) => {
        if n == d {
          // coefficient is 1
          v
        } else if n == -d {
          // coefficient is -1
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(v),
          }
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(coeff_expr),
            right: Box::new(v),
          }
        }
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Expr::Integer(0);
  }

  // Build the expression and evaluate to get canonical ordering
  let mut result = terms[0].clone();
  for term in &terms[1..] {
    result = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(result),
      right: Box::new(term.clone()),
    };
  }
  crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result)
}

// ──── Decomposition algorithm ───────────────────────────────────────

/// Find all divisors of n that are > 1 and < n, sorted from largest to smallest.
fn proper_divisors_desc(n: usize) -> Vec<usize> {
  let mut divs = Vec::new();
  for d in 2..n {
    if n.is_multiple_of(d) {
      divs.push(d);
    }
  }
  divs.sort_unstable_by(|a, b| b.cmp(a));
  divs
}

/// Try to decompose polynomial f = g(h) for a given inner degree s.
/// h is monic of degree s with h(0) = 0.
/// Returns Some((g_coeffs, h_coeffs)) if successful, None otherwise.
fn try_decompose_step(f: &[Rat], s: usize) -> Option<(Vec<Rat>, Vec<Rat>)> {
  let n = poly_degree(f) as usize;
  if n == 0 || s == 0 || !n.is_multiple_of(s) {
    return None;
  }
  let r = n / s;

  let d_r = f[n]; // leading coefficient of f = leading coefficient of g

  // Determine h's coefficients (h is monic, degree s, h(0) = 0)
  // h[s] = 1 (monic), h[0] = 0 (no constant)
  let mut h = vec![RAT_ZERO; s + 1];
  h[s] = RAT_ONE;

  // Determine h[s-1], h[s-2], ..., h[1] from top coefficients of f
  for j in 1..s {
    // Compute h^r with current partial h
    let h_pow_r = poly_pow(&h, r);
    // The target coefficient at x^{n-j}
    let target = rat_div(f.get(n - j).copied().unwrap_or(RAT_ZERO), d_r);
    let current = h_pow_r.get(n - j).copied().unwrap_or(RAT_ZERO);
    let diff = rat_sub(target, current);
    // c_{s-j} = diff / r
    let c = rat_div(diff, (r as i128, 1));
    h[s - j] = c;
  }

  // Now h is fully determined (h[0] = 0)
  // Verify: express f in the h-basis using greedy division
  let g = try_express_in_h_basis(f, &h, r)?;

  // Verify g ∘ h = f (sanity check)
  let composed = poly_compose(&g, &h);
  let diff = poly_sub(f, &composed);
  if poly_degree(&diff) >= 0 {
    return None;
  }

  // Check that g is not the identity (g(y) = y)
  if poly_degree(&g) as usize == 1 && g[1] == RAT_ONE && rat_is_zero(g[0]) {
    return None;
  }

  // For r=1 (linear outer), only accept if h is a monomial.
  // Wolfram convention: decompositions with linear outer and non-monomial inner
  // are considered trivial (just adding/scaling a constant).
  if r == 1 && !is_monomial(&h) {
    return None;
  }

  Some((g, h))
}

/// Try to express f as g(h) by computing g's coefficients via h-basis division.
/// f = g_r * h^r + g_{r-1} * h^{r-1} + ... + g_0
fn try_express_in_h_basis(f: &[Rat], h: &[Rat], r: usize) -> Option<Vec<Rat>> {
  let s = poly_degree(h) as usize;
  let mut g = vec![RAT_ZERO; r + 1];
  let mut remainder = f.to_vec();

  for k in (0..=r).rev() {
    let deg_rem = poly_degree(&remainder);
    let target_deg = (k * s) as i32;

    if deg_rem < 0 {
      // remainder is zero, all remaining g_k = 0
      break;
    }
    if deg_rem > target_deg {
      // Can't express as polynomial in h
      return None;
    }
    if deg_rem == target_deg {
      let h_k = poly_pow(h, k);
      let lc_hk = h_k.get(k * s).copied().unwrap_or(RAT_ZERO);
      if rat_is_zero(lc_hk) {
        return None;
      }
      g[k] = rat_div(remainder[deg_rem as usize], lc_hk);
      let scaled = poly_scale(&h_k, g[k]);
      remainder = poly_sub(&remainder, &scaled);
    }
    // else: deg_rem < target_deg, so g[k] = 0 (already set)
  }

  // Check remainder is zero
  if poly_degree(&remainder) >= 0 {
    return None;
  }

  Some(g)
}

/// Compose two polynomials: compute g(h(x)).
fn poly_compose(g: &[Rat], h: &[Rat]) -> Vec<Rat> {
  // Horner's method: g(h) = g_r * h^r + ... + g_0
  // = ((...((g_r * h + g_{r-1}) * h + g_{r-2}) * h + ...) * h + g_0)
  let dg = poly_degree(g);
  if dg < 0 {
    return vec![RAT_ZERO];
  }

  let mut result = vec![g[dg as usize]];
  for i in (0..dg as usize).rev() {
    result = poly_mul(&result, h);
    // Add g[i]
    if result.is_empty() {
      result = vec![g[i]];
    } else {
      result[0] = rat_add(result[0], g[i]);
    }
  }
  result
}

/// Recursively decompose a polynomial.
/// Returns a list of polynomial coefficient vectors such that
/// f(x) = p1(p2(...pn(x)...)).
fn decompose_poly(f: &[Rat]) -> Vec<Vec<Rat>> {
  let n = poly_degree(f);
  if n <= 1 {
    return vec![f.to_vec()];
  }
  let n = n as usize;

  // Try decomposition with smallest inner degree s first (Wolfram convention).
  // Also try s = n (r=1, linear outer with monomial h) last.
  let mut candidates: Vec<usize> = Vec::new();
  // Add proper divisors in ascending order (smallest s first)
  let mut divs = proper_divisors_desc(n);
  divs.reverse(); // now ascending
  candidates.extend(divs);
  // Add s = n (for r = 1, linear outer - only valid with monomial h)
  candidates.push(n);

  for s in candidates {
    let r = n / s;
    if !n.is_multiple_of(s) {
      continue;
    }
    if r == 0 || s == 0 {
      continue;
    }

    if let Some((g, h)) = try_decompose_step(f, s) {
      // Recursively decompose g and h
      let mut result = decompose_poly(&g);
      let h_decomp = decompose_poly(&h);
      result.extend(h_decomp);

      // Post-process: merge adjacent monomials
      result = merge_adjacent_monomials(result);

      return result;
    }
  }

  // No decomposition found - return as-is
  vec![f.to_vec()]
}

/// Merge adjacent monomial components.
/// If two adjacent polynomials are both monomials (a*x^p and b*x^q),
/// their composition is a*b^p * x^(p*q), which is also a monomial.
fn merge_adjacent_monomials(polys: Vec<Vec<Rat>>) -> Vec<Vec<Rat>> {
  if polys.len() <= 1 {
    return polys;
  }

  let mut result: Vec<Vec<Rat>> = Vec::new();
  for p in polys {
    if result.is_empty()
      || !is_monomial(&p)
      || !is_monomial(result.last().unwrap())
    {
      result.push(p);
    } else {
      // Compose the two monomials
      // prev = a*x^p, curr = b*x^q
      // composition: a * (b*x^q)^p = a * b^p * x^(pq)
      let prev = result.pop().unwrap();
      let dp = poly_degree(&prev) as usize;
      let dq = poly_degree(&p) as usize;
      let a = prev[dp];
      let b = p[dq];
      let new_deg = dp * dq;
      let new_coeff = rat_mul(a, rat_pow(b, dp));
      let mut merged = vec![RAT_ZERO; new_deg + 1];
      merged[new_deg] = new_coeff;
      result.push(merged);
    }
  }
  result
}

/// Rational power
fn rat_pow(base: Rat, exp: usize) -> Rat {
  if exp == 0 {
    return RAT_ONE;
  }
  let mut result = RAT_ONE;
  for _ in 0..exp {
    result = rat_mul(result, base);
  }
  result
}
