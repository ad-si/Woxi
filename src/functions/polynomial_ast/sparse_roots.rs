//! Integer roots of sparse univariate polynomials.
//!
//! `Solve[x^12345 - 2 x^777 + 1 == 0, x, Integers]` must not build a dense
//! degree-12345 polynomial or its complex roots: an integer root r != 0 of a
//! polynomial with integer coefficients divides its lowest nonzero
//! coefficient (after the factor x^k of a zero root is taken out), so the
//! candidates are that coefficient's divisors, each checked on the sparse
//! terms — first modulo two large primes, then exactly.

use super::*;
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// The terms `(exponent, coefficient)` of `expr` as a polynomial in `var`
/// with integer coefficients (rational ones are cleared by their common
/// denominator), ascending by exponent; None for anything else, including a
/// constant.
pub(super) fn sparse_integer_terms(
  expr: &Expr,
  var: &str,
) -> Option<Vec<(u64, BigInt)>> {
  let (poly, gens) = super::poly_expand::polynomial_in_symbols(expr)?;
  if gens.len() != 1 || gens[0] != var {
    return None;
  }
  let mut lcm = BigInt::one();
  for c in poly.t.values() {
    let (mut a, mut b) = (lcm.clone(), c.d.clone());
    while !b.is_zero() {
      let r = &a % &b;
      a = b;
      b = r;
    }
    let g = a;
    lcm = &lcm / g * &c.d;
  }
  let mut terms: Vec<(u64, BigInt)> = poly
    .t
    .iter()
    .map(|(m, c)| (m[0] as u64, &c.n * (&lcm / &c.d)))
    .filter(|(_, c)| !c.is_zero())
    .collect();
  terms.sort_by_key(|(e, _)| *e);
  if terms.last().is_none_or(|(e, _)| *e == 0) {
    return None;
  }
  Some(terms)
}

/// Two primes below 2^61 for the modular pre-check of a candidate root.
const PRIMES: [u64; 2] = [2_305_843_009_213_693_951, 2_305_843_009_213_693_921];

fn mod_pow(mut b: u128, mut e: u64, p: u128) -> u128 {
  let mut r = 1u128;
  b %= p;
  while e > 0 {
    if e & 1 == 1 {
      r = r * b % p;
    }
    b = b * b % p;
    e >>= 1;
  }
  r
}

fn residue(x: &BigInt, p: u64) -> u128 {
  let m = BigInt::from(p);
  let r = ((x % &m) + &m) % &m;
  r.to_u64().unwrap_or(0) as u128
}

/// The `j`-th derivative of the terms (falling-factorial coefficients).
fn derivative(terms: &[(u64, BigInt)], j: u64) -> Vec<(u64, BigInt)> {
  terms
    .iter()
    .filter(|(e, _)| *e >= j)
    .map(|(e, c)| {
      let mut f = c.clone();
      for k in 0..j {
        f *= BigInt::from(e - k);
      }
      (e - j, f)
    })
    .collect()
}

/// Whether the terms vanish at `r`: modulo the primes first, then exactly.
fn vanishes_at(terms: &[(u64, BigInt)], r: &BigInt) -> bool {
  for &p in &PRIMES {
    let pm = p as u128;
    let rr = residue(r, p);
    let sum = terms.iter().fold(0u128, |acc, (e, c)| {
      (acc + residue(c, p) * mod_pow(rr, *e, pm)) % pm
    });
    if sum != 0 {
      return false;
    }
  }
  let mut sum = BigInt::zero();
  for (e, c) in terms {
    sum += c * num_traits::pow::Pow::pow(r, *e);
  }
  sum.is_zero()
}

/// The positive divisors of `n` (n >= 1) by trial division; None when `n`
/// is too large to factor this way.
fn divisors(n: &BigInt) -> Option<Vec<u64>> {
  let mut n = n.to_u64().filter(|&v| v <= 1_000_000_000_000)?;
  let mut divs = vec![1u64];
  let mut d = 2u64;
  while d * d <= n {
    let mut k = 0;
    while n % d == 0 {
      n /= d;
      k += 1;
    }
    if k > 0 {
      let base = divs.clone();
      let mut pk = 1;
      for _ in 0..k {
        pk *= d;
        divs.extend(base.iter().map(|x| x * pk));
      }
    }
    d += 1;
  }
  if n > 1 {
    let base = divs.clone();
    divs.extend(base.iter().map(|x| x * n));
  }
  Some(divs)
}

/// The integer roots of the sparse polynomial with their multiplicities,
/// ascending; None when the lowest coefficient is too large to factor.
pub(super) fn integer_roots(
  terms: &[(u64, BigInt)],
) -> Option<Vec<(BigInt, u64)>> {
  let shift = terms.first()?.0;
  let reduced: Vec<(u64, BigInt)> =
    terms.iter().map(|(e, c)| (e - shift, c.clone())).collect();
  let mut roots = Vec::new();
  for d in divisors(&reduced[0].1.abs())? {
    for r in [-BigInt::from(d), BigInt::from(d)] {
      if vanishes_at(&reduced, &r) {
        let mut mult = 1;
        while vanishes_at(&derivative(&reduced, mult), &r) {
          mult += 1;
        }
        roots.push((r, mult));
      }
    }
  }
  if shift > 0 {
    roots.push((BigInt::zero(), shift));
  }
  roots.sort();
  Some(roots)
}

/// `lhs - rhs` of a single equation `lhs == rhs`.
pub(super) fn equation_difference(eq: &Expr) -> Option<Expr> {
  match eq {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2
      && matches!(operators.as_slice(), [ComparisonOp::Equal]) =>
    {
      Some(minus2(operands[0].clone(), operands[1].clone()))
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      Some(minus2(args[0].clone(), args[1].clone()))
    }
    _ => None,
  }
}

/// The integer roots (with multiplicity) of a single polynomial equation in
/// the lone variable `var`, for Solve and Reduce over the integers.
pub(super) fn integer_roots_of_equation(
  eq: &Expr,
  var: &str,
) -> Option<Vec<(BigInt, u64)>> {
  let diff = equation_difference(eq)?;
  integer_roots(&sparse_integer_terms(&diff, var)?)
}

// ─── Real roots of sparse polynomials ──────────────────────────────────
//
// `Reduce[x^10000 + … == 0, x, Reals]` must not build the dense polynomial
// or its 10000 complex roots. With x = ±e^t the positive (negative) roots
// of a polynomial with k terms are the zeros of the exponential sum
// F(t) = Σ c_i e^(e_i t). Its derivative F'(t) = Σ e_i c_i e^(e_i t) has
// one term fewer once the constant term drops out, so recursively the
// zeros of F' where it changes sign split the line into pieces on which F
// is monotone: each holds a zero of F exactly when F has opposite signs at
// its ends (Descartes' rule for fewnomials made constructive). Every sign
// is certified: F is evaluated in machine arithmetic together with a
// rigorous bound on the rounding error, and the call gives up (None)
// rather than guess when a value is too close to zero to decide — at a
// multiple root, or roots closer than machine precision can separate.

/// One term `c e^(exp t)` of an exponential sum, with `ln |c|` cached.
#[derive(Clone)]
struct ExpSumTerm {
  exp: u64,
  coeff: BigInt,
  ln_abs: f64,
  negative: bool,
}

impl ExpSumTerm {
  fn new(exp: u64, coeff: BigInt) -> Self {
    let ln_abs = ln_abs_bigint(&coeff);
    let negative = coeff.is_negative();
    ExpSumTerm {
      exp,
      coeff,
      ln_abs,
      negative,
    }
  }
}

/// `ln |n|` for a nonzero integer of any size.
fn ln_abs_bigint(n: &BigInt) -> f64 {
  let bits = n.bits();
  if bits <= 900 {
    n.abs().to_f64().unwrap_or(f64::INFINITY).ln()
  } else {
    let shift = bits - 64;
    let top: BigInt = n.abs() >> shift;
    top.to_f64().unwrap_or(f64::INFINITY).ln()
      + shift as f64 * std::f64::consts::LN_2
  }
}

/// `F(t)` as `(value ± error) · e^scale`, the error a rigorous bound on the
/// rounding of every step (with a safety factor for the libm functions).
fn exp_sum_eval(terms: &[ExpSumTerm], t: f64) -> (f64, f64, f64) {
  let eps = f64::EPSILON;
  let exponents: Vec<f64> = terms
    .iter()
    .map(|tm| tm.ln_abs + tm.exp as f64 * t)
    .collect();
  let scale = exponents.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let errors: Vec<f64> = terms
    .iter()
    .zip(&exponents)
    .map(|(tm, a)| {
      4.0 * eps * (tm.ln_abs.abs() + (tm.exp as f64 * t).abs() + a.abs() + 1.0)
    })
    .collect();
  let scale_error = errors.iter().copied().fold(0.0, f64::max);
  let (mut value, mut magnitude, mut error) = (0.0, 0.0, 0.0);
  for ((tm, a), e) in terms.iter().zip(&exponents).zip(&errors) {
    let z = a - scale;
    let v = z.exp();
    magnitude += v;
    error += v * ((e + scale_error + 4.0 * eps * (z.abs() + 1.0)).exp_m1());
    value += if tm.negative { -v } else { v };
  }
  error += (terms.len() as f64 + 2.0) * eps * magnitude;
  (value, 4.0 * error, scale)
}

/// The certified sign of `F(t)`, None when it is too close to zero.
fn exp_sum_sign(terms: &[ExpSumTerm], t: f64) -> Option<bool> {
  let (value, error, _) = exp_sum_eval(terms, t);
  (value.abs() > error).then_some(value > 0.0)
}

/// `ln` of an upper bound of `|F'|` on `(-∞, b]`: `Σ |e_i c_i| e^(e_i b)`.
fn exp_sum_derivative_bound_ln(terms: &[ExpSumTerm], b: f64) -> f64 {
  let parts: Vec<f64> = terms
    .iter()
    .filter(|tm| tm.exp > 0)
    .map(|tm| tm.ln_abs + (tm.exp as f64).ln() + tm.exp as f64 * b)
    .collect();
  let top = parts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  if top == f64::NEG_INFINITY {
    return top;
  }
  let sum: f64 = parts.iter().map(|p| (p - top).exp()).sum();
  top
    + sum.ln()
    + 1e-6
    + 16.0 * f64::EPSILON * (top.abs() + b.abs() * terms.len() as f64 + 1.0)
}

/// A point strictly inside `(a, b)` where `F` has a certified sign.
fn certified_inner_point(
  terms: &[ExpSumTerm],
  a: f64,
  b: f64,
) -> Option<(f64, bool)> {
  for frac in [0.5, 0.375, 0.625, 0.25, 0.75, 0.125, 0.875] {
    let m = a + (b - a) * frac;
    if m <= a || m >= b {
      continue;
    }
    if let Some(s) = exp_sum_sign(terms, m) {
      return Some((m, s));
    }
  }
  None
}

/// Intervals `[a, b]` in `t`, ascending and disjoint, each holding exactly
/// one zero of the exponential sum where it changes sign, with the sign of
/// the sum at `a`. The terms ascend by exponent, the first one being the
/// constant term (exponent 0).
fn exp_sum_sign_changes(terms: &[ExpSumTerm]) -> Option<Vec<(f64, f64, bool)>> {
  if terms.len() <= 1 {
    return Some(Vec::new());
  }
  // F' without its positive factor e^(e_1 t): one term fewer.
  let shift = terms[1].exp;
  let derivative: Vec<ExpSumTerm> = terms[1..]
    .iter()
    .map(|tm| ExpSumTerm::new(tm.exp - shift, &tm.coeff * BigInt::from(tm.exp)))
    .collect();
  let mut critical = exp_sum_sign_changes(&derivative)?;
  // Narrow each interval around a sign change of F' until F keeps one
  // sign on it: |F(a)| beats the most F can move across the interval.
  let mut piece_signs = vec![!terms[0].negative];
  for (a, b, derivative_sign_at_a) in &mut critical {
    let mut settled = None;
    for _ in 0..400 {
      let (value, error, scale) = exp_sum_eval(terms, *a);
      if value.abs() > error {
        let reach = exp_sum_derivative_bound_ln(terms, *b) + (*b - *a).ln();
        if (value.abs() - error).ln() + scale > reach {
          settled = Some(value > 0.0);
          break;
        }
      }
      let (m, s) = certified_inner_point(&derivative, *a, *b)?;
      if s == *derivative_sign_at_a {
        *a = m;
      } else {
        *b = m;
      }
    }
    piece_signs.push(settled?);
  }
  piece_signs.push(!terms[terms.len() - 1].negative);
  // Cauchy bounds put every zero of F inside (lower, upper).
  let last = terms[terms.len() - 1].ln_abs;
  let first = terms[0].ln_abs;
  let upper = terms[..terms.len() - 1]
    .iter()
    .map(|tm| tm.ln_abs - last)
    .fold(0.0, f64::max)
    + std::f64::consts::LN_2
    + 1.0;
  let lower = -(terms[1..]
    .iter()
    .map(|tm| tm.ln_abs - first)
    .fold(0.0, f64::max)
    + std::f64::consts::LN_2
    + 1.0);
  let lower = critical.first().map_or(lower, |c| lower.min(c.0));
  let upper = critical.last().map_or(upper, |c| upper.max(c.1));
  // F is monotone between the narrowed intervals.
  let mut changes = Vec::new();
  for j in 0..=critical.len() {
    let left = if j == 0 { lower } else { critical[j - 1].1 };
    let right = if j == critical.len() {
      upper
    } else {
      critical[j].0
    };
    if piece_signs[j] != piece_signs[j + 1] {
      changes.push((left, right, piece_signs[j]));
    }
  }
  Some(changes)
}

/// The sign changes of `F(t) = f(±e^t)` for the sparse integer terms
/// `(exponent, coefficient)` (ascending by exponent, nonzero
/// coefficients), the positive side first, and whether 0 is a root.
#[allow(clippy::type_complexity)]
fn sparse_root_brackets(
  terms: &[(u64, BigInt)],
) -> Option<[(Vec<ExpSumTerm>, Vec<(f64, f64, bool)>); 2]> {
  let shift = terms.first()?.0;
  if terms.iter().any(|(_, c)| c.is_zero()) {
    return None;
  }
  let side = |negate_odd: bool| {
    let sum: Vec<ExpSumTerm> = terms
      .iter()
      .map(|(e, c)| {
        let e = e - shift;
        let c = if negate_odd && e % 2 == 1 {
          -c
        } else {
          c.clone()
        };
        ExpSumTerm::new(e, c)
      })
      .collect();
    // A zero where F touches zero without changing sign is an extremum,
    // inside an interval around a sign change of F' that then never
    // settles, so it ends the search with None rather than going missing.
    let changes = exp_sum_sign_changes(&sum)?;
    Some((sum, changes))
  };
  Some([side(false)?, side(true)?])
}

/// Bisect a sign change `[a, b]` of `F` (sign `sign_a` at `a`) down to
/// machine precision.
fn refine_sign_change(terms: &[ExpSumTerm], change: (f64, f64, bool)) -> f64 {
  let (mut a, mut b, sign_a) = change;
  loop {
    let m = f64::midpoint(a, b);
    if m <= a || m >= b {
      return m;
    }
    match exp_sum_sign(terms, m) {
      Some(s) if s == sign_a => a = m,
      Some(_) => b = m,
      // Within rounding noise of the zero.
      None => return m,
    }
  }
}

/// The distinct real roots, ascending and to machine precision, of the
/// polynomial with the sparse integer terms `(exponent, coefficient)`
/// (ascending by exponent, nonzero coefficients). None when a root could
/// not be separated with certainty (e.g. a repeated real root), so the
/// caller can fall back to a dense method.
pub(crate) fn sparse_real_root_values(
  terms: &[(u64, BigInt)],
) -> Option<Vec<f64>> {
  let [(positive, up), (negative, down)] = sparse_root_brackets(terms)?;
  let mut roots: Vec<f64> = down
    .iter()
    .rev()
    .map(|c| -refine_sign_change(&negative, *c).exp())
    .collect();
  if terms[0].0 > 0 {
    roots.push(0.0);
  }
  roots.extend(up.iter().map(|c| refine_sign_change(&positive, *c).exp()));
  Some(roots)
}

/// The number of distinct real roots of the polynomial with the sparse
/// integer terms, as `sparse_real_root_values` finds them.
pub(crate) fn sparse_real_root_count(terms: &[(u64, BigInt)]) -> Option<usize> {
  let [(_, up), (_, down)] = sparse_root_brackets(terms)?;
  Some(up.len() + down.len() + usize::from(terms[0].0 > 0))
}

/// The machine value of `Root[f, k]` for the polynomial with the integer
/// coefficients `coeffs` (index = degree) when `k` names one of its real
/// roots and they can be bracketed with certainty; None otherwise, leaving
/// the dense root finder to decide. The brackets are reliable where the
/// dense root finder can fail to converge (a degree in the thousands, or
/// roots of very different sizes).
pub(crate) fn sparse_real_root_value(
  coeffs: &[BigInt],
  k: i128,
) -> Option<f64> {
  let terms: Vec<(u64, BigInt)> = coeffs
    .iter()
    .enumerate()
    .filter(|(_, c)| !c.is_zero())
    .map(|(e, c)| (e as u64, c.clone()))
    .collect();
  if k < 1 || !(2..=64).contains(&terms.len()) {
    return None;
  }
  sparse_real_root_values(&terms)?
    .get(k as usize - 1)
    .copied()
}

/// Whether `Root[f, k, …]` with a polynomial `f` over the integers names a
/// real root (`Root` numbers the real roots first, ascending). None when
/// `expr` is no such `Root` object or its real roots can't be counted.
pub(super) fn root_object_is_real(expr: &Expr) -> Option<bool> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Root" || !(args.len() == 2 || args.len() == 3) {
    return None;
  }
  let (Expr::Function { body }, Expr::Integer(k)) = (&args[0], &args[1]) else {
    return None;
  };
  // Solve lists every root of a polynomial, so the count is remembered
  // for its other `Root` objects.
  thread_local! {
    static COUNTS: std::cell::RefCell<
      std::collections::HashMap<String, Option<usize>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
  }
  let key = crate::syntax::expr_to_string(body);
  let count = if let Some(hit) = COUNTS.with(|c| c.borrow().get(&key).copied())
  {
    hit
  } else {
    let var = "\u{2620}sparse\u{2620}";
    let poly = crate::syntax::substitute_slots(body, &[id_expr(var)]);
    let count = sparse_integer_terms(&poly, var)
      .and_then(|terms| sparse_real_root_count(&terms));
    COUNTS.with(|c| {
      let mut c = c.borrow_mut();
      if c.len() >= 1024 {
        c.clear();
      }
      c.insert(key, count);
    });
    count
  }?;
  Some(*k >= 1 && (*k as usize) <= count)
}

/// Degrees above which `Reduce` over the reals and `Root` treat an integer
/// polynomial as sparse: they count its real roots directly instead of
/// factoring the dense polynomial, which takes time cubic in the degree.
pub(super) const SPARSE_ROOT_MIN_DEGREE: u64 = 500;

/// Whether the sparse integer terms (any exponent shift taken out) make a
/// polynomial `Reduce` and `Root` treat as sparse: a degree above
/// `SPARSE_ROOT_MIN_DEGREE`, at least three terms, and a degree above 4
/// as a polynomial in `x^g` for the common factor `g` of the exponents.
/// Binomials and quadratics, cubics and quartics in `x^g` are left to
/// `Solve`, which writes their roots with radicals.
pub(super) fn is_high_degree_sparse(terms: &[(u64, BigInt)]) -> bool {
  let Some((low, _)) = terms.first() else {
    return false;
  };
  let degree = terms[terms.len() - 1].0 - low;
  let mut g = 0u64;
  for (e, _) in terms {
    let (mut a, mut b) = (g, e - low);
    while b != 0 {
      (a, b) = (b, a % b);
    }
    g = a;
  }
  (3..=64).contains(&terms.len())
    && degree > SPARSE_ROOT_MIN_DEGREE
    && degree / g.max(1) > 4
}

/// Whether `f(p/q) = 0` for the sparse terms (q > 0), modulo the primes
/// first, then exactly: `Σ c_i p^(e_i) q^(n - e_i) = 0`.
fn vanishes_at_rational(
  terms: &[(u64, BigInt)],
  p: &BigInt,
  q: &BigInt,
) -> bool {
  let n = terms.last().map_or(0, |(e, _)| *e);
  for &prime in &PRIMES {
    let pm = prime as u128;
    let (pr, qr) = (residue(p, prime), residue(q, prime));
    let sum = terms.iter().fold(0u128, |acc, (e, c)| {
      (acc
        + residue(c, prime) * mod_pow(pr, *e, pm) % pm * mod_pow(qr, n - e, pm))
        % pm
    });
    if sum != 0 {
      return false;
    }
  }
  let mut sum = BigInt::zero();
  for (e, c) in terms {
    sum += c
      * num_traits::pow::Pow::pow(p, *e)
      * num_traits::pow::Pow::pow(q, n - e);
  }
  sum.is_zero()
}

/// `Reduce[f == 0, x, Reals]` for a sparse polynomial over the integers of
/// high degree (see `is_high_degree_sparse`): `x == 0` for a factor `x^m`,
/// `x == p/q` for a rational root of the rest `g`, and `x == Root[g, k]`
/// for each other real root of `g` (dividing out a rational root would
/// leave a dense cofactor). None (leaving the dense methods to decide)
/// when the roots can't be separated, e.g. at a repeated root.
pub(super) fn sparse_reals_reduce(expanded: &Expr, var: &str) -> Option<Expr> {
  let terms = sparse_integer_terms(expanded, var)?;
  if !is_high_degree_sparse(&terms) {
    return None;
  }
  let shift = terms[0].0;
  let mut reduced: Vec<(u64, BigInt)> =
    terms.iter().map(|(e, c)| (e - shift, c.clone())).collect();
  // The primitive polynomial with a positive leading coefficient, as
  // `Root` writes it.
  let content = reduced.iter().fold(BigInt::zero(), |g, (_, c)| {
    crate::functions::math_ast::gcd_bigint(&g, c).abs()
  });
  let sign = if reduced.last()?.1.is_negative() {
    -BigInt::one()
  } else {
    BigInt::one()
  };
  for (_, c) in &mut reduced {
    *c = &*c / &content * &sign;
  }
  // A rational root p/q (p dividing the constant and q the leading
  // coefficient) is written as such; the other real roots as `Root`
  // objects, numbered among all the real roots of the polynomial.
  let numerators = divisors(&reduced[0].1.abs())?;
  let denominators = divisors(&reduced.last()?.1.abs())?;
  let mut rational_roots: Vec<(i128, i128)> = Vec::new();
  for p in &numerators {
    for q in &denominators {
      let (mut a, mut b) = (*p, *q);
      while b != 0 {
        (a, b) = (b, a % b);
      }
      if a != 1 {
        continue;
      }
      for p in [-(*p as i128), *p as i128] {
        if vanishes_at_rational(&reduced, &BigInt::from(p), &BigInt::from(*q)) {
          rational_roots.push((p, *q as i128));
        }
      }
    }
  }
  let values = sparse_real_root_values(&reduced)?;
  let mut rational_indices = Vec::new();
  for (p, q) in &rational_roots {
    let r = *p as f64 / *q as f64;
    let near: Vec<usize> = (0..values.len())
      .filter(|&i| (values[i] - r).abs() <= 1e-9 * r.abs().max(1.0))
      .collect();
    let [i] = near.as_slice() else {
      return None;
    };
    rational_indices.push(*i);
  }
  let slot_terms: Vec<Expr> = reduced
    .iter()
    .map(|(e, c)| {
      let c = crate::functions::math_ast::bigint_to_expr(c.clone());
      match e {
        0 => c,
        1 => times(vec![c, Expr::Slot(1)]),
        _ => times(vec![c, pow(Expr::Slot(1), Expr::Integer(*e as i128))]),
      }
    })
    .collect();
  let body = crate::evaluator::evaluate_expr_to_expr(&plus(slot_terms)).ok()?;
  let function = Expr::Function {
    body: Box::new(body),
  };
  let x = id_expr(var);
  // The rational roots ascending, then the `Root` objects.
  let mut rationals = rational_roots.clone();
  if shift > 0 {
    rationals.push((0, 1));
  }
  rationals.sort_by(|(p1, q1), (p2, q2)| (p1 * q2).cmp(&(p2 * q1)));
  let mut equalities: Vec<Expr> = rationals
    .iter()
    .map(|(p, q)| {
      let value = if *q == 1 {
        Expr::Integer(*p)
      } else {
        call("Rational", vec![Expr::Integer(*p), Expr::Integer(*q)])
      };
      make_equality(&x, &value)
    })
    .collect();
  for k in (1..=values.len()).filter(|k| !rational_indices.contains(&(k - 1))) {
    let root = call(
      "Root",
      vec![function.clone(), Expr::Integer(k as i128), Expr::Integer(0)],
    );
    equalities.push(make_equality(&x, &root));
  }
  Some(build_or(equalities))
}

/// A polynomial of high degree that is one of low degree in `x^g` (the
/// common factor `g > 1` of its exponents, no root at zero), e.g.
/// `x^1000 - 3 x^500 + 2`: its terms as a polynomial in `y = x^g`, and `g`.
/// `Reduce` over the reals solves for `y` and takes the real `g`-th roots,
/// rather than finding all `deg` complex roots of the dense polynomial.
pub(super) fn power_substitution(
  expanded: &Expr,
  var: &str,
) -> Option<(Vec<(u64, BigInt)>, u64)> {
  let terms = sparse_integer_terms(expanded, var)?;
  if terms.first()?.0 != 0 {
    return None;
  }
  let degree = terms.last()?.0;
  let g = terms.iter().fold(0u64, |g, (e, _)| {
    let (mut a, mut b) = (g, *e);
    while b != 0 {
      (a, b) = (b, a % b);
    }
    a
  });
  (g > 1 && degree > SPARSE_ROOT_MIN_DEGREE)
    .then(|| (terms.iter().map(|(e, c)| (e / g, c.clone())).collect(), g))
}
