//! GroebnerBasis[{polys}, {vars}] — reduced Gröbner basis in
//! lexicographic order over the rationals, normalized the way
//! wolframscript prints it: primitive integer coefficients, positive
//! leading coefficient, basis sorted ascending by leading monomial.
//!
//! Exact arbitrary-precision fraction arithmetic; unrecognized
//! (non-polynomial) input or an exhausted work cap returns the call
//! unevaluated.

#[allow(unused_imports)]
use super::*;
use crate::evaluator::pattern_matching::expr_equal;
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::BTreeMap;

type Frac = (BigInt, BigInt);
type Mono = Vec<u32>;
type Poly = BTreeMap<std::cmp::Reverse<Mono>, Frac>; // leading term first

// With `Modulus -> p` (prime) the same engine runs over GF(p): every
// coefficient is kept as (v, 1) with v in [0, p), which `norm` — the
// funnel all fraction arithmetic passes through — enforces via modular
// inversion of denominators.
thread_local! {
  static GB_MODULUS: std::cell::Cell<Option<i128>> =
    const { std::cell::Cell::new(None) };
}

/// Set the thread-local modulus for the lifetime of the returned guard.
struct ModulusGuard;
impl Drop for ModulusGuard {
  fn drop(&mut self) {
    GB_MODULUS.with(|m| m.set(None));
  }
}
fn set_modulus(p: i128) -> ModulusGuard {
  GB_MODULUS.with(|m| m.set(Some(p)));
  ModulusGuard
}

fn mod_inverse(a: i128, p: i128) -> Option<i128> {
  let (mut old_r, mut r) = (a.rem_euclid(p), p);
  let (mut old_s, mut s) = (1i128, 0i128);
  if old_r == 0 {
    return None;
  }
  while r != 0 {
    let q = old_r / r;
    (old_r, r) = (r, old_r - q * r);
    (old_s, s) = (s, old_s - q * s);
  }
  if old_r != 1 {
    return None;
  }
  Some(old_s.rem_euclid(p))
}

fn big(n: i128) -> BigInt {
  BigInt::from(n)
}

fn gcd_big(a: &BigInt, b: &BigInt) -> BigInt {
  let (mut a, mut b) = (a.abs(), b.abs());
  while !b.is_zero() {
    let r = &a % &b;
    a = b;
    b = r;
  }
  a
}

fn one() -> Frac {
  (BigInt::one(), BigInt::one())
}

fn norm(f: &Frac) -> Option<Frac> {
  if f.1.is_zero() {
    return None;
  }
  if let Some(p) = GB_MODULUS.with(std::cell::Cell::get) {
    let pb = big(p);
    let den = f.1.mod_floor_i(&pb).to_i128()?;
    let inv = mod_inverse(den, p)?;
    let num = f.0.mod_floor_i(&pb);
    return Some(((num * big(inv)).mod_floor_i(&pb), BigInt::one()));
  }
  let g = gcd_big(&f.0, &f.1);
  let g = if g.is_zero() { BigInt::one() } else { g };
  let (mut n, mut d) = (&f.0 / &g, &f.1 / &g);
  if d.is_negative() {
    n = -n;
    d = -d;
  }
  Some((n, d))
}

trait ModFloor {
  fn mod_floor_i(&self, m: &BigInt) -> BigInt;
}

impl ModFloor for BigInt {
  fn mod_floor_i(&self, m: &BigInt) -> BigInt {
    let r = self % m;
    if r.is_negative() { r + m } else { r }
  }
}

fn f_add(a: &Frac, b: &Frac) -> Option<Frac> {
  norm(&(&a.0 * &b.1 + &b.0 * &a.1, &a.1 * &b.1))
}

fn f_mul(a: &Frac, b: &Frac) -> Option<Frac> {
  norm(&(&a.0 * &b.0, &a.1 * &b.1))
}

fn f_div(a: &Frac, b: &Frac) -> Option<Frac> {
  if b.0.is_zero() {
    return None;
  }
  norm(&(&a.0 * &b.1, &a.1 * &b.0))
}

fn f_neg(a: &Frac) -> Option<Frac> {
  norm(&(-&a.0, a.1.clone()))
}

fn poly_add_term(p: &mut Poly, m: Mono, c: &Frac) -> Option<()> {
  let key = std::cmp::Reverse(m);
  let existing = p
    .get(&key)
    .cloned()
    .unwrap_or((BigInt::zero(), BigInt::one()));
  let sum = if existing.1.is_one()
    && c.1.is_one()
    && GB_MODULUS.with(std::cell::Cell::get).is_none()
  {
    (&existing.0 + &c.0, BigInt::one())
  } else {
    f_add(&existing, c)?
  };
  if sum.0.is_zero() {
    p.remove(&key);
  } else {
    p.insert(key, sum);
  }
  Some(())
}

fn leading(p: &Poly) -> Option<(&Mono, Frac)> {
  p.iter().next().map(|(k, c)| (&k.0, c.clone()))
}

fn mono_divides(a: &Mono, b: &Mono) -> bool {
  a.iter().zip(b.iter()).all(|(x, y)| x <= y)
}

fn mono_div(b: &Mono, a: &Mono) -> Mono {
  b.iter().zip(a.iter()).map(|(x, y)| x - y).collect()
}

fn mono_mul(a: &Mono, b: &Mono) -> Mono {
  a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn mono_lcm(a: &Mono, b: &Mono) -> Mono {
  a.iter().zip(b.iter()).map(|(x, y)| *x.max(y)).collect()
}

/// p - c * m * q
fn poly_sub_scaled(p: &Poly, q: &Poly, c: &Frac, m: &Mono) -> Option<Poly> {
  let mut out = p.clone();
  for (key, qc) in q {
    let term_m = mono_mul(&key.0, m);
    let term_c = f_neg(&f_mul(qc, c)?)?;
    poly_add_term(&mut out, term_m, &term_c)?;
  }
  Some(out)
}

/// Fully reduce p modulo the basis (all terms, not just the leading one):
/// terms no leading monomial divides are moved to the remainder, the rest are
/// cancelled in place.
fn reduce_field(p: &Poly, basis: &[Poly]) -> Option<Poly> {
  let leads: Vec<Option<(Mono, Frac)>> = basis
    .iter()
    .map(|b| leading(b).map(|(m, c)| (m.clone(), c)))
    .collect();
  let mut current = p.clone();
  let mut remainder = Poly::new();
  while let Some((key, c)) =
    current.iter().next().map(|(k, c)| (k.clone(), c.clone()))
  {
    let divisor = leads
      .iter()
      .position(|l| l.as_ref().is_some_and(|(lm, _)| mono_divides(lm, &key.0)));
    if let Some(bi) = divisor {
      let (lm, lc) = leads[bi].as_ref()?;
      let factor = f_div(&c, lc)?;
      let shift = mono_div(&key.0, lm);
      for (bk, bc) in &basis[bi] {
        let term_c = f_neg(&f_mul(bc, &factor)?)?;
        poly_add_term(&mut current, mono_mul(&bk.0, &shift), &term_c)?;
      }
    } else {
      current.remove(&key);
      remainder.insert(key, c);
    }
  }
  Some(remainder)
}

/// `reduce` for polynomials with integer coefficients (denominators 1) over
/// Q: cancels leading terms without dividing, so no gcd runs per operation.
/// The remainder is only determined up to a nonzero scalar, which every
/// caller normalizes away.
fn reduce_integral(p: &Poly, basis: &[Poly]) -> Option<Poly> {
  let leads: Vec<Option<(Mono, BigInt)>> = basis
    .iter()
    .map(|b| leading(b).map(|(m, c)| (m.clone(), c.0)))
    .collect();
  let mut current = p.clone();
  let mut remainder = Poly::new();
  let scale = |poly: &mut Poly, k: &BigInt| {
    for c in poly.values_mut() {
      c.0 = &c.0 * k;
    }
  };
  while let Some((key, c)) =
    current.iter().next().map(|(k, c)| (k.clone(), c.0.clone()))
  {
    let divisor = leads
      .iter()
      .position(|l| l.as_ref().is_some_and(|(lm, _)| mono_divides(lm, &key.0)));
    if let Some(bi) = divisor {
      let (lm, lc) = leads[bi].as_ref()?;
      let g = gcd_big(&c, lc);
      let (mult_cur, mult_b) = (lc / &g, &c / &g);
      if !mult_cur.is_one() {
        scale(&mut current, &mult_cur);
        scale(&mut remainder, &mult_cur);
      }
      let shift = mono_div(&key.0, lm);
      for (bk, bc) in &basis[bi] {
        let term = (-(&bc.0 * &mult_b), BigInt::one());
        poly_add_term(&mut current, mono_mul(&bk.0, &shift), &term)?;
      }
    } else {
      current.remove(&key);
      remainder.insert(key, (c, BigInt::one()));
    }
  }
  Some(remainder)
}

fn reduce(p: &Poly, basis: &[Poly]) -> Option<Poly> {
  if GB_MODULUS.with(std::cell::Cell::get).is_some() {
    reduce_field(p, basis)
  } else {
    reduce_integral(&primitive_part(p), basis)
  }
}

/// Render a `Poly` (lex order, leading term first) back to an `Expr`, then
/// evaluate it so the result is canonicalized like wolframscript.
fn poly_to_expr(p: &Poly, vars: &[Expr]) -> Expr {
  if p.is_empty() {
    return Expr::Integer(0);
  }
  let mut terms: Vec<Expr> = Vec::new();
  for (key, (n, d)) in p {
    let (n, d) = (n.clone(), d.clone());
    let mono = &key.0;
    let mut factors: Vec<Expr> = Vec::new();
    for (j, &e) in mono.iter().enumerate() {
      if e == 1 {
        factors.push(vars[j].clone());
      } else if e > 1 {
        factors.push(pow2(vars[j].clone(), Expr::Integer(e as i128)));
      }
    }
    let (n_expr, d_expr) = (
      crate::functions::math_ast::bigint_to_expr(n.clone()),
      crate::functions::math_ast::bigint_to_expr(d.clone()),
    );
    let coeff = if d.is_one() {
      n_expr
    } else {
      call("Rational", vec![n_expr, d_expr])
    };
    let term = if factors.is_empty() {
      coeff
    } else {
      let mono_expr = factors.into_iter().reduce(times2).unwrap();
      if n.is_one() && d.is_one() {
        mono_expr
      } else {
        times2(coeff, mono_expr)
      }
    };
    terms.push(term);
  }
  let sum = terms.into_iter().reduce(plus2).unwrap();
  crate::evaluator::evaluate_expr_to_expr(&sum).unwrap_or(sum)
}

/// PolynomialReduce[poly, {g1, …, gk}, {x1, …, xn}] — the fast, exact
/// path: multivariate division in lexicographic order, keeping every
/// coefficient as an exact `i128` fraction. Returns `{{q1, …, qk}, r}`
/// such that `poly = q1 g1 + … + qk gk + r`. Returns `None` (so the caller
/// falls back to `polynomial_reduce_multivar_symbolic`'s slower,
/// evaluator-driven arithmetic) for non-polynomial input or a coefficient
/// that is not an exact rational number in the given variables — a free
/// parameter, or an irrational constant like `Sqrt[3]`.
///
/// A "variable" need not be a bare identifier: as in real Mathematica, any
/// expression not itself a sum/product/power (a function application like
/// `p[a]`, say) can serve as an indeterminate, matched structurally against
/// each polynomial's factors.
pub fn polynomial_reduce_multivar(
  dividend: &Expr,
  divisors: &[Expr],
  vars: &[Expr],
) -> Option<Expr> {
  if vars.is_empty() || vars.len() > 6 {
    return None;
  }
  let expand = |e: &Expr| -> Expr {
    crate::evaluator::evaluate_expr_to_expr(&call1("Expand", e.clone()))
      .unwrap_or_else(|_| e.clone())
  };

  let mut p = expr_to_poly(&expand(dividend), vars)?;
  let mut div_polys: Vec<Poly> = Vec::with_capacity(divisors.len());
  for d in divisors {
    div_polys.push(expr_to_poly(&expand(d), vars)?);
  }
  let k = div_polys.len();
  let mut quotients: Vec<Poly> = vec![Poly::new(); k];
  let mut remainder = Poly::new();

  let mut guard = 0usize;
  while let Some((lm_p, lc_p)) = leading(&p).map(|(m, c)| (m.clone(), c)) {
    guard += 1;
    if guard > 100_000 {
      return None;
    }
    let mut reduced = false;
    for i in 0..k {
      let Some((lm_i, lc_i)) =
        leading(&div_polys[i]).map(|(m, c)| (m.clone(), c))
      else {
        continue;
      };
      if mono_divides(&lm_i, &lm_p) {
        let factor = f_div(&lc_p, &lc_i)?;
        let shift = mono_div(&lm_p, &lm_i);
        poly_add_term(&mut quotients[i], shift.clone(), &factor)?;
        p = poly_sub_scaled(&p, &div_polys[i], &factor, &shift)?;
        reduced = true;
        break;
      }
    }
    if !reduced {
      // The leading term is divisible by no divisor: move it to the remainder.
      poly_add_term(&mut remainder, lm_p.clone(), &lc_p)?;
      poly_add_term(&mut p, lm_p, &f_neg(&lc_p)?)?;
    }
  }

  let q_exprs: Vec<Expr> =
    quotients.iter().map(|q| poly_to_expr(q, vars)).collect();
  let r_expr = poly_to_expr(&remainder, vars);
  Some(Expr::List(vec![Expr::List(q_exprs.into()), r_expr].into()))
}

/// Scale a polynomial over Q to coprime integer coefficients, so repeated
/// S-polynomial reduction does not accumulate ever larger denominators.
fn primitive_part(p: &Poly) -> Poly {
  let mut denom_lcm = BigInt::one();
  for c in p.values() {
    denom_lcm = &denom_lcm * (&c.1 / gcd_big(&denom_lcm, &c.1));
  }
  let mut content = BigInt::zero();
  for c in p.values() {
    content = gcd_big(&content, &(&c.0 * (&denom_lcm / &c.1)));
  }
  if content.is_zero() {
    return p.clone();
  }
  let mut out = Poly::new();
  for (k, c) in p {
    out.insert(
      k.clone(),
      (&c.0 * (&denom_lcm / &c.1) / &content, BigInt::one()),
    );
  }
  out
}

fn s_poly(a: &Poly, b: &Poly) -> Option<Poly> {
  let (lma, lca) = leading(a)?;
  let (lmb, lcb) = leading(b)?;
  let l = mono_lcm(lma, lmb);
  // a * (l/lma)/lca - b * (l/lmb)/lcb
  let mut left = Poly::new();
  let sa = mono_div(&l, lma);
  let ca = f_div(&one(), &lca)?;
  for (key, c) in a {
    poly_add_term(&mut left, mono_mul(&key.0, &sa), &f_mul(c, &ca)?)?;
  }
  let sb = mono_div(&l, lmb);
  let cb = f_div(&one(), &lcb)?;
  poly_sub_scaled(&left, b, &cb, &sb)
}

/// Extract `Modulus -> m` (m >= 0) from an option argument.
fn groebner_modulus_option(opt: &Expr) -> Option<i128> {
  let value = |v: &Expr| match v {
    Expr::Integer(m) if *m >= 0 => Some(*m),
    _ => None,
  };
  if let Expr::Rule {
    pattern,
    replacement,
  } = opt
    && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Modulus")
  {
    return value(replacement);
  }
  if let Expr::FunctionCall { name, args } = opt
    && (name == "Rule" || name == "RuleDelayed")
    && args.len() == 2
    && matches!(&args[0], Expr::Identifier(s) if s == "Modulus")
  {
    return value(&args[1]);
  }
  None
}

use crate::functions::math_ast::is_prime_i128;

pub fn groebner_basis_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let full_args = args;
  let uneval = || unevaluated("GroebnerBasis", full_args);
  // GroebnerBasis[polys, vars, Modulus -> p] computes over GF(p);
  // Modulus -> 0 is ordinary arithmetic, and composite moduli (which
  // would need strong Z/m bases) stay unevaluated.
  let mut modulus: Option<i128> = None;
  if args.len() == 3 {
    match groebner_modulus_option(&args[2]) {
      Some(0) => return groebner_basis_ast(&args[..2]),
      Some(p) if is_prime_i128(p) => modulus = Some(p),
      _ => return Ok(uneval()),
    }
  } else if args.len() != 2 {
    return Ok(uneval());
  }
  let _guard = modulus.map(set_modulus);
  let args = &args[..2];
  let polys_in: Vec<Expr> = match &args[0] {
    Expr::List(items) if !items.is_empty() => items.iter().cloned().collect(),
    single => vec![single.clone()],
  };
  // A "variable" need not be a bare identifier — `expr_to_poly` matches any
  // non-arithmetic expression (e.g. `f[x]`) structurally.
  let vars: Vec<Expr> = match &args[1] {
    Expr::List(items) => items.to_vec(),
    single => vec![single.clone()],
  };
  if vars.is_empty() || vars.len() > 6 {
    return Ok(uneval());
  }

  // Parse the input polynomials
  let mut basis: Vec<Poly> = Vec::new();
  for p in &polys_in {
    let expanded =
      crate::evaluator::evaluate_expr_to_expr(&call1("Expand", p.clone()))?;
    match expr_to_poly(&expanded, &vars) {
      Some(poly) if !poly.is_empty() => basis.push(if modulus.is_none() {
        primitive_part(&poly)
      } else {
        poly
      }),
      Some(_) => {} // zero polynomial contributes nothing
      None => return Ok(uneval()),
    }
  }
  if basis.is_empty() {
    // Over GF(p) inputs that all vanish mod p leave an empty basis;
    // over the rationals wolframscript prints {0}.
    if modulus.is_some() {
      return Ok(Expr::List(vec![].into()));
    }
    return Ok(Expr::List(vec![Expr::Integer(0)].into()));
  }

  // Buchberger's algorithm with a work cap
  let result = (|| -> Option<Vec<Poly>> {
    let mut basis = basis;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..basis.len() {
      for j in (i + 1)..basis.len() {
        pairs.push((i, j));
      }
    }
    let mut steps = 0usize;
    // Normal selection strategy: the pair whose leading monomials have the
    // lowest-degree lcm goes first, which keeps intermediate polynomials
    // small. Pairs with coprime leading monomials reduce to zero (product
    // criterion) and are skipped.
    while !pairs.is_empty() {
      let mut best = 0;
      let mut best_degree = u32::MAX;
      for (idx, &(i, j)) in pairs.iter().enumerate() {
        let (lma, _) = leading(&basis[i])?;
        let (lmb, _) = leading(&basis[j])?;
        let d: u32 = mono_lcm(lma, lmb).iter().sum();
        if d < best_degree {
          best_degree = d;
          best = idx;
        }
      }
      let (i, j) = pairs.swap_remove(best);
      let (lma, _) = leading(&basis[i])?;
      let (lmb, _) = leading(&basis[j])?;
      if lma.iter().zip(lmb.iter()).all(|(a, b)| *a == 0 || *b == 0) {
        continue;
      }
      steps += 1;
      if steps > 2000 {
        return None;
      }
      let s = s_poly(&basis[i], &basis[j])?;
      let r = reduce(&s, &basis)?;
      if !r.is_empty() {
        let r = if modulus.is_none() {
          primitive_part(&r)
        } else {
          r
        };
        let new_idx = basis.len();
        for k in 0..new_idx {
          pairs.push((k, new_idx));
        }
        basis.push(r);
      }
    }
    // Inter-reduce: drop polynomials whose leading term is divisible by
    // another's, then fully reduce each against the rest
    let mut keep: Vec<Poly> = Vec::new();
    for i in 0..basis.len() {
      let (lm_i, _) = leading(&basis[i])?;
      let redundant = basis.iter().enumerate().any(|(j, q)| {
        if i == j {
          return false;
        }
        match leading(q) {
          Some((lm_j, _)) => {
            mono_divides(lm_j, lm_i) && (lm_j != lm_i || j < i)
          }
          None => false,
        }
      });
      if !redundant {
        keep.push(basis[i].clone());
      }
    }
    let snapshot = keep.clone();
    let mut reduced: Vec<Poly> = Vec::new();
    for (i, p) in snapshot.iter().enumerate() {
      let others: Vec<Poly> = snapshot
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, q)| q.clone())
        .collect();
      let r = reduce(p, &others)?;
      if !r.is_empty() {
        reduced.push(r);
      }
    }
    Some(reduced)
  })();
  let Some(mut reduced) = result else {
    return Ok(uneval());
  };

  // Trivial ideal: any nonzero constant
  if reduced
    .iter()
    .any(|p| leading(p).is_some_and(|(m, _)| m.iter().all(|&e| e == 0)))
  {
    return Ok(Expr::List(vec![Expr::Integer(1)].into()));
  }

  // Sort ascending by leading monomial (lex)
  reduced.sort_by(|a, b| {
    let la = leading(a).map(|(m, _)| m.clone()).unwrap_or_default();
    let lb = leading(b).map(|(m, _)| m.clone()).unwrap_or_default();
    la.cmp(&lb)
  });

  // Normalize to primitive integer coefficients with a positive leading
  // coefficient, then render. Over GF(p) the basis is made monic first,
  // so the primitive/sign logic below is a no-op (content 1, lead +1).
  let mut out: Vec<Expr> = Vec::with_capacity(reduced.len());
  for p in &reduced {
    let monic;
    let p = if let Some(pm) = modulus {
      let Some((_, lc)) = leading(p) else {
        return Ok(uneval());
      };
      let Some(inv) = lc.0.to_i128().and_then(|l| mod_inverse(l, pm)) else {
        return Ok(uneval());
      };
      let mut q = Poly::new();
      for (k, c) in p {
        q.insert(
          k.clone(),
          ((&c.0 * big(inv)).mod_floor_i(&big(pm)), BigInt::one()),
        );
      }
      monic = q;
      &monic
    } else {
      p
    };
    let rendered = {
      let mut denom_lcm = BigInt::one();
      for c in p.values() {
        denom_lcm = &denom_lcm * (&c.1 / gcd_big(&denom_lcm, &c.1));
      }
      let mut nums: Vec<(Mono, BigInt)> = Vec::new();
      let mut content = BigInt::zero();
      for (key, c) in p {
        let scaled = &c.0 * (&denom_lcm / &c.1);
        content = gcd_big(&content, &scaled);
        nums.push((key.0.clone(), scaled));
      }
      let content = if content.is_zero() {
        BigInt::one()
      } else {
        content
      };
      let lead_sign = nums.first().is_some_and(|(_, c)| c.is_negative());
      let sign = BigInt::from(if lead_sign { -1 } else { 1 });
      let mut terms: Vec<Expr> = Vec::new();
      for (m, c) in &nums {
        let c = &sign * c / &content;
        let mut factors: Vec<Expr> = Vec::new();
        if !c.is_one() || m.iter().all(|&e| e == 0) {
          factors.push(crate::functions::math_ast::bigint_to_expr(c));
        }
        for (vi, &e) in m.iter().enumerate() {
          if e == 1 {
            factors.push(vars[vi].clone());
          } else if e > 1 {
            factors.push(pow2(vars[vi].clone(), Expr::Integer(e as i128)));
          }
        }
        terms.push(match factors.len() {
          1 => factors.remove(0),
          _ => times(factors),
        });
      }
      let sum = if terms.len() == 1 {
        terms.remove(0)
      } else {
        plus(terms)
      };
      crate::evaluator::evaluate_expr_to_expr(&sum).ok()
    };
    match rendered {
      Some(e) => out.push(e),
      None => return Ok(uneval()),
    }
  }
  Ok(Expr::List(out.into()))
}

/// Expanded expression -> sparse multivariate polynomial. None for
/// anything outside c * v1^e1 * ... terms with rational c.
fn expr_to_poly(expr: &Expr, vars: &[Expr]) -> Option<Poly> {
  fn split<'a>(e: &'a Expr, sign: i128, out: &mut Vec<(&'a Expr, i128)>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Plus" => {
        for a in args {
          split(a, sign, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left,
        right,
      } => {
        split(left, sign, out);
        split(right, sign, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left,
        right,
      } => {
        split(left, sign, out);
        split(right, -sign, out);
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => split(operand, -sign, out),
      other => out.push((other, sign)),
    }
  }
  let mut terms: Vec<(&Expr, i128)> = Vec::new();
  split(expr, 1, &mut terms);
  let mut poly = Poly::new();
  for (term, sign) in terms {
    let (m, c) = term_to_mono(term, vars)?;
    poly_add_term(&mut poly, m, &f_mul(&c, &(big(sign), BigInt::one()))?)?;
  }
  Some(poly)
}

fn term_to_mono(term: &Expr, vars: &[Expr]) -> Option<(Mono, Frac)> {
  let mut mono = vec![0u32; vars.len()];
  let mut coef: Frac = one();
  fn flatten<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args {
          flatten(a, out);
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        flatten(left, out);
        flatten(right, out);
      }
      other => out.push(other),
    }
  }
  let mut factors: Vec<&Expr> = Vec::new();
  flatten(term, &mut factors);
  for f in factors {
    match f {
      Expr::Integer(n) => coef = f_mul(&coef, &(big(*n), BigInt::one()))?,
      Expr::BigInteger(n) => coef = f_mul(&coef, &(n.clone(), BigInt::one()))?,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        let as_big = |e: &Expr| match e {
          Expr::Integer(n) => Some(big(*n)),
          Expr::BigInteger(n) => Some(n.clone()),
          _ => None,
        };
        coef = f_mul(&coef, &(as_big(&args[0])?, as_big(&args[1])?))?;
      }
      // A bare variable — not necessarily an `Identifier`: as in real
      // Mathematica, a function application like `p[a]` may itself be one
      // of the given "variables", matched structurally.
      _ if vars.iter().any(|w| expr_equal(f, w)) => {
        let i = vars.iter().position(|w| expr_equal(f, w)).unwrap();
        mono[i] += 1;
      }
      _ => {
        // var^k
        let (base, exp) = match f {
          Expr::FunctionCall { name, args }
            if name == "Power" && args.len() == 2 =>
          {
            (&args[0], &args[1])
          }
          Expr::BinaryOp {
            op: BinaryOperator::Power,
            left,
            right,
          } => (&**left as &Expr, &**right as &Expr),
          _ => return None,
        };
        if let Expr::Integer(k) = exp
          && *k >= 1
          && *k <= u32::MAX as i128
          && let Some(i) = vars.iter().position(|w| expr_equal(base, w))
        {
          mono[i] += *k as u32;
        } else {
          return None;
        }
      }
    }
  }
  Some((mono, coef))
}
