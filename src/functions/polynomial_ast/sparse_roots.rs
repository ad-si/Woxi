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
