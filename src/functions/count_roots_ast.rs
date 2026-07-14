//! CountRoots[f, x] / CountRoots[f, {x, a, b}] — count the real roots of a
//! univariate polynomial `f` in `x`, with multiplicity, over the reals or a
//! closed real interval [a, b] (endpoints may be ±Infinity).
//!
//! The implementation uses exact `BigInt` rational arithmetic:
//!   * Yun's algorithm splits `f` into squarefree factors `a_k` (a_k holds the
//!     distinct roots of multiplicity exactly `k`);
//!   * Sturm's theorem counts the distinct real roots of each squarefree
//!     factor in the requested interval;
//!   * the total is `Σ k · (distinct real roots of a_k in [a, b])`.
//!
//! Non-polynomial input (transcendental functions, complex rectangle bounds,
//! Real-valued bounds) returns the call unevaluated.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, unevaluated};
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

// ---------------------------------------------------------------------------
// Exact rationals backed by BigInt
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq)]
struct Rat {
  n: BigInt,
  d: BigInt, // invariant: d > 0, gcd(|n|, d) == 1
}

fn gcd_bigint(a: &BigInt, b: &BigInt) -> BigInt {
  let mut a = a.abs();
  let mut b = b.abs();
  while !b.is_zero() {
    let r = &a % &b;
    a = b;
    b = r;
  }
  a
}

impl Rat {
  fn new(mut n: BigInt, mut d: BigInt) -> Rat {
    debug_assert!(!d.is_zero());
    if d.is_negative() {
      n = -n;
      d = -d;
    }
    let g = gcd_bigint(&n, &d);
    if !g.is_zero() && !g.is_one() {
      n /= &g;
      d /= &g;
    }
    Rat { n, d }
  }

  fn from_int(n: BigInt) -> Rat {
    Rat {
      n,
      d: BigInt::one(),
    }
  }

  fn zero() -> Rat {
    Rat {
      n: BigInt::zero(),
      d: BigInt::one(),
    }
  }

  fn is_zero(&self) -> bool {
    self.n.is_zero()
  }

  /// -1, 0, or 1
  fn sign(&self) -> i32 {
    if self.n.is_zero() {
      0
    } else if self.n.is_negative() {
      -1
    } else {
      1
    }
  }

  fn add(&self, o: &Rat) -> Rat {
    Rat::new(&self.n * &o.d + &o.n * &self.d, &self.d * &o.d)
  }

  fn sub(&self, o: &Rat) -> Rat {
    Rat::new(&self.n * &o.d - &o.n * &self.d, &self.d * &o.d)
  }

  fn mul(&self, o: &Rat) -> Rat {
    Rat::new(&self.n * &o.n, &self.d * &o.d)
  }

  fn div(&self, o: &Rat) -> Rat {
    Rat::new(&self.n * &o.d, &self.d * &o.n)
  }

  fn neg(&self) -> Rat {
    Rat {
      n: -&self.n,
      d: self.d.clone(),
    }
  }
}

// ---------------------------------------------------------------------------
// Dense polynomials with ascending Rat coefficients
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Poly(Vec<Rat>); // ascending; trailing zeros trimmed; [] == zero poly

impl Poly {
  fn trim(mut self) -> Poly {
    while self.0.last().map(|c| c.is_zero()).unwrap_or(false) {
      self.0.pop();
    }
    self
  }

  fn is_zero(&self) -> bool {
    self.0.is_empty()
  }

  /// -1 for the zero polynomial, otherwise the actual degree
  fn degree(&self) -> isize {
    self.0.len() as isize - 1
  }

  fn lead(&self) -> &Rat {
    self.0.last().expect("lead of zero polynomial")
  }

  fn deriv(&self) -> Poly {
    if self.0.len() <= 1 {
      return Poly(vec![]);
    }
    let mut out = Vec::with_capacity(self.0.len() - 1);
    for (i, c) in self.0.iter().enumerate().skip(1) {
      out.push(c.mul(&Rat::from_int(BigInt::from(i))));
    }
    Poly(out).trim()
  }

  fn sub(&self, o: &Poly) -> Poly {
    let n = self.0.len().max(o.0.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
      let a = self.0.get(i).cloned().unwrap_or_else(Rat::zero);
      let b = o.0.get(i).cloned().unwrap_or_else(Rat::zero);
      out.push(a.sub(&b));
    }
    Poly(out).trim()
  }

  /// Polynomial remainder `self mod divisor` (divisor nonzero).
  fn rem(&self, divisor: &Poly) -> Poly {
    let mut r = self.clone();
    let db = divisor.degree();
    let lc = divisor.lead();
    while !r.is_zero() && r.degree() >= db {
      let dr = r.degree();
      let coef = r.lead().div(lc);
      let shift = (dr - db) as usize;
      for (i, b) in divisor.0.iter().enumerate() {
        r.0[shift + i] = r.0[shift + i].sub(&coef.mul(b));
      }
      r = r.trim();
    }
    r
  }

  /// Exact quotient `self / divisor` (assumes the division is exact).
  fn div_exact(&self, divisor: &Poly) -> Poly {
    if self.is_zero() {
      return Poly(vec![]);
    }
    let db = divisor.degree();
    let lc = divisor.lead();
    let mut r = self.clone();
    let mut quot = vec![Rat::zero(); (self.degree() - db + 1).max(0) as usize];
    while !r.is_zero() && r.degree() >= db {
      let dr = r.degree();
      let coef = r.lead().div(lc);
      let shift = (dr - db) as usize;
      quot[shift] = coef.clone();
      for (i, b) in divisor.0.iter().enumerate() {
        r.0[shift + i] = r.0[shift + i].sub(&coef.mul(b));
      }
      r = r.trim();
    }
    Poly(quot).trim()
  }

  /// Monic GCD of two polynomials.
  fn gcd(&self, o: &Poly) -> Poly {
    let mut a = self.clone();
    let mut b = o.clone();
    while !b.is_zero() {
      let r = a.rem(&b);
      a = b;
      b = r;
    }
    a.monic()
  }

  fn monic(&self) -> Poly {
    if self.is_zero() {
      return Poly(vec![]);
    }
    let lc = self.lead().clone();
    Poly(self.0.iter().map(|c| c.div(&lc)).collect())
  }

  /// Sign of the polynomial at a finite rational point.
  fn sign_at(&self, x: &Rat) -> i32 {
    // Horner evaluation over exact rationals.
    let mut acc = Rat::zero();
    for c in self.0.iter().rev() {
      acc = acc.mul(x).add(c);
    }
    acc.sign()
  }

  /// Sign as x -> +infinity (leading coefficient sign).
  fn sign_at_pos_inf(&self) -> i32 {
    if self.is_zero() {
      0
    } else {
      self.lead().sign()
    }
  }

  /// Sign as x -> -infinity (leading coeff sign times (-1)^degree).
  fn sign_at_neg_inf(&self) -> i32 {
    if self.is_zero() {
      return 0;
    }
    let s = self.lead().sign();
    if self.degree() % 2 == 0 { s } else { -s }
  }
}

// ---------------------------------------------------------------------------
// Sturm machinery
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Bound<'a> {
  NegInf,
  Finite(&'a Rat),
  PosInf,
}

/// Build the Sturm chain of a squarefree polynomial `q`.
fn sturm_chain(q: &Poly) -> Vec<Poly> {
  let mut chain = vec![q.clone(), q.deriv()];
  loop {
    let n = chain.len();
    if chain[n - 1].is_zero() {
      chain.pop();
      break;
    }
    let r = chain[n - 2].rem(&chain[n - 1]);
    if r.is_zero() {
      break;
    }
    chain.push(Poly(r.0.iter().map(|c| c.neg()).collect()));
  }
  chain
}

/// Number of sign variations of the Sturm chain at a bound.
fn variations(chain: &[Poly], at: Bound) -> i32 {
  let mut prev = 0i32;
  let mut count = 0i32;
  for p in chain {
    let s = match at {
      Bound::NegInf => p.sign_at_neg_inf(),
      Bound::PosInf => p.sign_at_pos_inf(),
      Bound::Finite(x) => p.sign_at(x),
    };
    if s == 0 {
      continue;
    }
    if prev != 0 && s != prev {
      count += 1;
    }
    prev = s;
  }
  count
}

/// Distinct real roots of squarefree `q` in the closed interval [a, b].
fn distinct_roots_in(q: &Poly, a: Bound, b: Bound) -> i32 {
  if q.degree() < 1 {
    return 0;
  }
  let chain = sturm_chain(q);
  // V(a) - V(b) counts distinct roots in the half-open interval (a, b].
  let mut count = variations(&chain, a) - variations(&chain, b);
  // Promote to the closed interval by including a finite left endpoint root.
  if let Bound::Finite(x) = a
    && q.sign_at(x) == 0
  {
    count += 1;
  }
  count.max(0)
}

// ---------------------------------------------------------------------------
// Squarefree factorization (Yun's algorithm)
// ---------------------------------------------------------------------------

/// Returns factors `(a_k, k)` where each `a_k` is the squarefree product of the
/// roots of multiplicity exactly `k`. Constant factors are omitted.
fn squarefree_factors(p: &Poly) -> Vec<(Poly, usize)> {
  let mut out = Vec::new();
  let fp = p.deriv();
  let a = p.gcd(&fp);
  let mut b = p.div_exact(&a);
  let mut c = fp.div_exact(&a);
  let mut d = c.sub(&b.deriv());
  let mut k = 1usize;
  while b.degree() >= 1 {
    let ak = b.gcd(&d);
    if ak.degree() >= 1 {
      out.push((ak.clone(), k));
    }
    b = b.div_exact(&ak);
    c = d.div_exact(&ak);
    d = c.sub(&b.deriv());
    k += 1;
    if k > 100_000 {
      break; // safety valve; never reached for real input
    }
  }
  out
}

// ---------------------------------------------------------------------------
// Coefficient extraction & bound parsing
// ---------------------------------------------------------------------------

fn rat_from_expr(e: &Expr) -> Option<Rat> {
  match e {
    Expr::Integer(n) => Some(Rat::from_int(BigInt::from(*n))),
    Expr::BigInteger(n) => Some(Rat::from_int(n.clone())),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let n = match &args[0] {
        Expr::Integer(n) => BigInt::from(*n),
        Expr::BigInteger(n) => n.clone(),
        _ => return None,
      };
      let d = match &args[1] {
        Expr::Integer(n) => BigInt::from(*n),
        Expr::BigInteger(n) => n.clone(),
        _ => return None,
      };
      if d.is_zero() {
        return None;
      }
      Some(Rat::new(n, d))
    }
    _ => None,
  }
}

/// Extract ascending rational coefficients of `f` in `var` via CoefficientList;
/// None if `f` is not a univariate polynomial with rational coefficients.
fn poly_from(f: &Expr, var: &str) -> Option<Poly> {
  let coeff_list =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "CoefficientList".to_string(),
      args: vec![f.clone(), Expr::Identifier(var.to_string())].into(),
    })
    .ok()?;
  let items = match &coeff_list {
    Expr::List(items) => items,
    _ => return None,
  };
  let mut out = Vec::with_capacity(items.len());
  for it in items {
    out.push(rat_from_expr(it)?);
  }
  Some(Poly(out).trim())
}

fn is_pos_infinity(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(s) if s == "Infinity")
}

/// Detect `-Infinity` across the forms the parser/evaluator may produce:
/// `Times[-1, Infinity]` (FunctionCall or BinaryOp) and `-Infinity` (UnaryOp).
fn is_neg_infinity(e: &Expr) -> bool {
  let is_neg = |x: &Expr| {
    matches!(x, Expr::Integer(n) if *n < 0)
      || matches!(x, Expr::Real(r) if *r < 0.0)
  };
  match e {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      is_neg(&args[0]) && is_pos_infinity(&args[1])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => is_neg(left) && is_pos_infinity(right),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_pos_infinity(operand),
    _ => false,
  }
}

enum ParsedBound {
  NegInf,
  PosInf,
  Finite(Rat),
}

fn parse_bound(e: &Expr) -> Option<ParsedBound> {
  if is_pos_infinity(e) {
    Some(ParsedBound::PosInf)
  } else if is_neg_infinity(e) {
    Some(ParsedBound::NegInf)
  } else {
    rat_from_expr(e).map(ParsedBound::Finite)
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn count_roots_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("CountRoots", args));

  if args.len() != 2 {
    return unevaluated();
  }

  // Determine the variable and the (optional) interval.
  let (var, lo, hi): (String, ParsedBound, ParsedBound) = match &args[1] {
    Expr::Identifier(v) => {
      (v.clone(), ParsedBound::NegInf, ParsedBound::PosInf)
    }
    Expr::List(items) if items.len() == 3 => {
      let v = match &items[0] {
        Expr::Identifier(v) => v.clone(),
        _ => return unevaluated(),
      };
      let lo = match parse_bound(&items[1]) {
        Some(b) => b,
        None => return unevaluated(),
      };
      let hi = match parse_bound(&items[2]) {
        Some(b) => b,
        None => return unevaluated(),
      };
      (v, lo, hi)
    }
    _ => return unevaluated(),
  };

  let poly = match poly_from(&args[0], &var) {
    Some(p) => p,
    None => return unevaluated(),
  };

  // Zero polynomial: every point is a root — not a finite count.
  if poly.is_zero() {
    return unevaluated();
  }
  // Constant nonzero polynomial: no roots.
  if poly.degree() == 0 {
    return Ok(Expr::Integer(0));
  }

  // Empty interval guard for finite bounds with lo > hi.
  if let (ParsedBound::Finite(a), ParsedBound::Finite(b)) = (&lo, &hi)
    && a.sub(b).sign() > 0
  {
    return Ok(Expr::Integer(0));
  }

  let lo_bound = match &lo {
    ParsedBound::NegInf => Bound::NegInf,
    ParsedBound::PosInf => Bound::PosInf,
    ParsedBound::Finite(r) => Bound::Finite(r),
  };
  let hi_bound = match &hi {
    ParsedBound::NegInf => Bound::NegInf,
    ParsedBound::PosInf => Bound::PosInf,
    ParsedBound::Finite(r) => Bound::Finite(r),
  };

  let mut total: i128 = 0;
  for (factor, mult) in squarefree_factors(&poly) {
    let distinct = distinct_roots_in(&factor, lo_bound, hi_bound) as i128;
    total += distinct * mult as i128;
  }

  Ok(Expr::Integer(total))
}
