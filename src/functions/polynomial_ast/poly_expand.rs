//! Fast path for `Expand` on polynomials over the rationals.
//!
//! The generic expander distributes `Expr` trees and combines like terms by
//! their printed form after every multiplication, which makes a power like
//! `(a + b + c + d - e - f)^11` (4368 monomials) take seconds. When the input
//! is built only from symbols, exact rational numbers, sums, products,
//! non-negative integer powers and division by a number, it is a polynomial,
//! so this module multiplies it out on sparse exponent-vector polynomials and
//! builds the result directly in canonical order.
//!
//! Every term is built exactly like `combine_and_build` builds it, and the
//! terms are sorted the way `Plus` orders polynomial terms — the exponent
//! vectors compared from the last generator (in canonical symbol order) to the
//! first, ascending. So the result is the expression the generic path
//! produces, except where that one is not canonical: it orders the factors of
//! a term by their printed form (`x1^2*x^3`) and can leave a coefficient
//! unnormalized (`-(-5/3)*x`).

use super::mfactor::{Mono, Poly, Q, q_to_expr};
use super::*;
use crate::functions::list_helpers_ast::wolfram_string_order;
use crate::functions::math_ast::canonical_polynomial::is_plain_symbol;
use crate::helpers::plus;
use num_bigint::BigInt;
use num_traits::{One, Signed};
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

/// Expand and combine `expr` when it is a polynomial with rational
/// coefficients in plain (non-system) symbols; None otherwise.
pub(super) fn expand_polynomial(expr: &Expr) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  let mut names = BTreeSet::new();
  if !collect_generators(expr, &mut names, None) || names.is_empty() {
    return None;
  }
  let mut gens: Vec<String> = names.into_iter().collect();
  gens.sort_by(|a, b| wolfram_string_order(a, b).cmp(&0).reverse());
  let p = to_poly(expr, gens.len(), &|e| symbol_index(e, &gens))?;
  let gens: Vec<Expr> = gens.into_iter().map(Expr::Identifier).collect();
  Some(poly_to_canonical_expr(&p, &gens))
}

/// `expand_polynomial` that also takes calls with a plain symbol head —
/// `f[x]`, `g[a, b]`, `x[1]` — as generators, opaque like Expand treats
/// them; None when there are none. The result is evaluated: each generator
/// once, and every term and the sum through `Power`, `Times` and `Plus`,
/// which put them in canonical order and apply the rules defined for them.
/// Evaluating the assembled sum instead re-evaluated every call in every
/// term it appears in.
///
/// The generic expander combines like terms by their printed form after
/// every multiplication, so expanding a product of sums of such calls (the
/// Dirac trace recursion of issue #1772) spent most of its time printing.
pub(super) fn expand_polynomial_with_kernels(expr: &Expr) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  let mut names = BTreeSet::new();
  let mut kernels = Kernels::default();
  if !collect_generators(expr, &mut names, Some(&mut kernels)) {
    return None;
  }
  if kernels.list.is_empty() {
    return None;
  }
  let symbols: Vec<String> = names.into_iter().collect();
  let lookup = |e: &Expr| match e {
    Expr::FunctionCall { .. } => kernels.index(e).map(|i| symbols.len() + i),
    _ => symbol_index(e, &symbols),
  };
  let p = to_poly(expr, symbols.len() + kernels.list.len(), &lookup)?;
  let eval = crate::evaluator::evaluate_expr_to_expr;
  let call = crate::evaluator::evaluate_function_call_ast;
  let gens: Vec<Expr> = symbols
    .into_iter()
    .map(Expr::Identifier)
    .chain(kernels.list)
    .map(|g| eval(&g).ok())
    .collect::<Option<_>>()?;
  let mut terms: Vec<Expr> = Vec::with_capacity(p.t.len());
  for (m, c) in &p.t {
    let mut factors: Vec<Expr> = Vec::new();
    if !(c.d.is_one() && c.n.is_one()) {
      factors.push(q_to_expr(c));
    }
    for (&e, g) in m.iter().zip(&gens) {
      match e {
        0 => {}
        1 => factors.push(g.clone()),
        _ => factors
          .push(call("Power", &[g.clone(), Expr::Integer(e as i128)]).ok()?),
      }
    }
    terms.push(match factors.len() {
      0 => q_to_expr(c),
      1 => factors.pop().unwrap(),
      _ => call("Times", &factors).ok()?,
    });
  }
  match terms.len() {
    0 => Some(Expr::Integer(0)),
    1 => terms.pop(),
    _ => call("Plus", &terms).ok(),
  }
}

/// The index of the plain symbol `e` among `names`.
fn symbol_index(e: &Expr, names: &[String]) -> Option<usize> {
  match e {
    Expr::Identifier(name) => names.iter().position(|g| g == name),
    _ => None,
  }
}

/// A call `expand_polynomial_with_kernels` treats as a generator.
fn is_kernel(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, .. } if is_plain_symbol(name))
}

/// The distinct opaque calls of a polynomial, in order of appearance.
#[derive(Default)]
struct Kernels {
  list: Vec<Expr>,
  by_hash: HashMap<u64, Vec<usize>>,
}

impl Kernels {
  fn index(&self, e: &Expr) -> Option<usize> {
    self
      .by_hash
      .get(&structural_hash(e))?
      .iter()
      .copied()
      .find(|&i| {
        crate::evaluator::pattern_matching::expr_equal(&self.list[i], e)
      })
  }

  fn insert(&mut self, e: &Expr) {
    if self.index(e).is_none() {
      self
        .by_hash
        .entry(structural_hash(e))
        .or_default()
        .push(self.list.len());
      self.list.push(e.clone());
    }
  }
}

/// A hash consistent with `expr_equal`: structurally equal expressions hash
/// alike.
fn structural_hash(e: &Expr) -> u64 {
  fn go(e: &Expr, h: &mut std::collections::hash_map::DefaultHasher) {
    match e {
      Expr::Integer(n) => (0u8, n).hash(h),
      Expr::Identifier(s) => (1u8, s).hash(h),
      Expr::String(s) => (2u8, s).hash(h),
      Expr::FunctionCall { name, args } => {
        (3u8, name, args.len()).hash(h);
        for a in args {
          go(a, h);
        }
      }
      Expr::List(items) => {
        (4u8, items.len()).hash(h);
        for a in items {
          go(a, h);
        }
      }
      // Rarer shapes only need to land in a consistent bucket.
      _ => 5u8.hash(h),
    }
  }
  let mut h = std::collections::hash_map::DefaultHasher::new();
  go(e, &mut h);
  h.finish()
}

/// `expr` as a polynomial over its symbols, which are returned sorted by
/// name; None unless it is a polynomial with rational coefficients in plain
/// symbols.
pub(super) fn polynomial_in_symbols(
  expr: &Expr,
) -> Option<(Poly, Vec<String>)> {
  let mut names = BTreeSet::new();
  if !collect_generators(expr, &mut names, None) {
    return None;
  }
  let gens: Vec<String> = names.into_iter().collect();
  Some((
    to_poly(expr, gens.len(), &|e| symbol_index(e, &gens))?,
    gens,
  ))
}

/// Check that `expr` is a polynomial the fast path handles, collecting the
/// names of its symbols and, when `kernels` is given, its opaque calls.
fn collect_generators(
  expr: &Expr,
  names: &mut BTreeSet<String>,
  mut kernels: Option<&mut Kernels>,
) -> bool {
  let mut go = |e: &Expr| collect_generators(e, names, kernels.as_deref_mut());
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => true,
    Expr::Identifier(name) => {
      if !is_plain_symbol(name) {
        return false;
      }
      if !names.contains(name) {
        names.insert(name.clone());
      }
      true
    }
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Rational" => rational_literal(expr).is_some(),
      "Plus" | "Times" => args.iter().all(go),
      "Power" if args.len() == 2 => {
        power_exponent(&args[1]).is_some() && go(&args[0])
      }
      _ => match kernels {
        Some(kernels) if is_kernel(expr) => {
          kernels.insert(expr);
          true
        }
        _ => false,
      },
    },
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Times => {
        go(left) && go(right)
      }
      BinaryOperator::Power => power_exponent(right).is_some() && go(left),
      BinaryOperator::Divide => {
        rational_literal(right).is_some_and(|q| !q.is_zero()) && go(left)
      }
      _ => false,
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => go(operand),
    _ => false,
  }
}

/// A non-negative machine-size integer exponent.
fn power_exponent(e: &Expr) -> Option<u32> {
  match e {
    Expr::Integer(n) if *n >= 0 => u32::try_from(*n).ok(),
    _ => None,
  }
}

/// An exact integer or rational literal.
fn rational_literal(e: &Expr) -> Option<Q> {
  match e {
    Expr::Integer(n) => Some(Q::int(BigInt::from(*n))),
    Expr::BigInteger(n) => Some(Q::int(n.clone())),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      let n = rational_literal(&args[0])?;
      let d = rational_literal(&args[1])?;
      (n.d.is_one() && d.d.is_one() && !d.is_zero()).then(|| n.div(&d))
    }
    _ => None,
  }
}

/// The index of a generator, None for any other expression.
type GenLookup<'a> = dyn Fn(&Expr) -> Option<usize> + 'a;

/// A single term `c * x^i * y^j * …` straight to its exponent vector, without
/// building and multiplying one-term polynomials. False when `expr` is not a
/// product of rational numbers and powers of the generators.
fn monomial(expr: &Expr, gens: &GenLookup, m: &mut Mono, c: &mut Q) -> bool {
  let mut symbol_power = |base: &Expr, k: u32| match gens(base) {
    Some(i) => {
      m[i] += k;
      true
    }
    None => false,
  };
  match expr {
    Expr::Identifier(_) => symbol_power(expr, 1),
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => power_exponent(right).is_some_and(|k| symbol_power(left, k)),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      power_exponent(&args[1]).is_some_and(|k| symbol_power(&args[0], k))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().all(|a| monomial(a, gens, m, c))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => monomial(left, gens, m, c) && monomial(right, gens, m, c),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      *c = c.neg();
      monomial(operand, gens, m, c)
    }
    _ => match rational_literal(expr) {
      Some(q) => {
        *c = c.mul(&q);
        true
      }
      None => symbol_power(expr, 1),
    },
  }
}

fn to_poly(expr: &Expr, nv: usize, gens: &GenLookup) -> Option<Poly> {
  let mut m: Mono = vec![0; nv];
  let mut c = Q::one();
  if monomial(expr, gens, &mut m, &mut c) {
    let mut p = Poly::zero(nv);
    p.add_term(m, c);
    return Some(p);
  }
  let sum = |items: &mut dyn Iterator<Item = &Expr>| -> Option<Poly> {
    let mut acc = Poly::zero(nv);
    for item in items {
      let mut m: Mono = vec![0; nv];
      let mut c = Q::one();
      if monomial(item, gens, &mut m, &mut c) {
        acc.add_term(m, c);
        continue;
      }
      for (m, c) in to_poly(item, nv, gens)?.t {
        acc.add_term(m, c);
      }
    }
    Some(acc)
  };
  let product = |items: &mut dyn Iterator<Item = &Expr>| -> Option<Poly> {
    let mut acc = Poly::one(nv);
    for item in items {
      acc = acc.mul(&to_poly(item, nv, gens)?);
    }
    Some(acc)
  };
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => sum(&mut args.iter()),
      "Times" => product(&mut args.iter()),
      "Power" => Some(pow(
        &to_poly(&args[0], nv, gens)?,
        power_exponent(&args[1])?,
      )),
      _ => None,
    },
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus => sum(&mut [left, right].into_iter().map(|b| &**b)),
      BinaryOperator::Minus => {
        let r = to_poly(right, nv, gens)?.scale(&Q::one().neg());
        let mut l = to_poly(left, nv, gens)?;
        for (m, c) in r.t {
          l.add_term(m, c);
        }
        Some(l)
      }
      BinaryOperator::Times => {
        product(&mut [left, right].into_iter().map(|b| &**b))
      }
      BinaryOperator::Power => {
        Some(pow(&to_poly(left, nv, gens)?, power_exponent(right)?))
      }
      BinaryOperator::Divide => Some(
        to_poly(left, nv, gens)?
          .scale(&Q::one().div(&rational_literal(right)?)),
      ),
      _ => None,
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(to_poly(operand, nv, gens)?.scale(&Q::one().neg())),
    _ => None,
  }
}

/// `p^n` by repeated multiplication with the (usually short) base, which for
/// sparse multivariate bases is cheaper than squaring the growing result.
fn pow(p: &Poly, n: u32) -> Poly {
  if let Some(r) = multinomial_pow(p, n) {
    return r;
  }
  let mut acc = Poly::one(p.nv);
  for _ in 0..n {
    acc = acc.mul(p);
  }
  acc
}

/// `p^n` by the multinomial theorem: one term per composition
/// e_1 + … + e_k = n of the exponent over the k terms of `p`, with
/// coefficient n!/(e_1!…e_k!) Π c_i^e_i. Repeated multiplication costs about
/// n^k terms for a dense k-term base (minutes for `(x + y + z)^1000`), while
/// there are only C(n + k - 1, k - 1) compositions; when terms collide
/// heavily (`(1 + x + … + x^20)^50`) the compositions far outnumber the
/// result, so that case — more than `MAX_COMPOSITIONS` — returns None.
fn multinomial_pow(p: &Poly, n: u32) -> Option<Poly> {
  const MAX_COMPOSITIONS: f64 = 2.0e6;
  let k = p.t.len();
  if k < 2 || n < 2 {
    return None;
  }
  let mut count = 1.0f64;
  for i in 1..k {
    count = count * (n as f64 + i as f64) / i as f64;
    if count > MAX_COMPOSITIONS {
      return None;
    }
  }
  let terms: Vec<(&Mono, &Q)> = p.t.iter().collect();
  // Powers of every term's numerator and denominator, 0..=n.
  let powers = |b: &BigInt| {
    let mut v = Vec::with_capacity(n as usize + 1);
    v.push(BigInt::one());
    for e in 1..=n as usize {
      let next = &v[e - 1] * b;
      v.push(next);
    }
    v
  };
  let num_pows: Vec<Vec<BigInt>> =
    terms.iter().map(|(_, c)| powers(&c.n)).collect();
  let den_pows: Vec<Vec<BigInt>> =
    terms.iter().map(|(_, c)| powers(&c.d)).collect();

  struct Walk<'a> {
    terms: &'a [(&'a Mono, &'a Q)],
    num_pows: &'a [Vec<BigInt>],
    den_pows: &'a [Vec<BigInt>],
    out: Poly,
  }
  impl Walk<'_> {
    fn go(
      &mut self,
      i: usize,
      rest: u32,
      num: &BigInt,
      den: &BigInt,
      mono: &Mono,
    ) {
      let add = |mono: &Mono, m: &Mono, e: u32| -> Mono {
        mono.iter().zip(m).map(|(a, b)| a + b * e).collect()
      };
      if i + 1 == self.terms.len() {
        // The last term takes the remaining exponent (multinomial factor 1).
        let num = num * &self.num_pows[i][rest as usize];
        let den = den * &self.den_pows[i][rest as usize];
        let m = add(mono, self.terms[i].0, rest);
        let c = if den.is_one() {
          Q::int(num)
        } else {
          Q::new(num, den)
        };
        self.out.add_term(m, c);
        return;
      }
      // binom(rest, e), updated incrementally as e grows.
      let mut binom = BigInt::one();
      for e in 0..=rest {
        if e > 0 {
          binom = binom * BigInt::from(rest - e + 1) / BigInt::from(e);
        }
        let num2 = num * &binom * &self.num_pows[i][e as usize];
        let den2 = den * &self.den_pows[i][e as usize];
        let m = add(mono, self.terms[i].0, e);
        self.go(i + 1, rest - e, &num2, &den2, &m);
      }
    }
  }
  let mut walk = Walk {
    terms: &terms,
    num_pows: &num_pows,
    den_pows: &den_pows,
    out: Poly::zero(p.nv),
  };
  walk.go(0, n, &BigInt::one(), &BigInt::one(), &vec![0; p.nv]);
  Some(walk.out)
}

/// Build the canonical expanded form of `p` over the generators.
fn poly_to_canonical_expr(p: &Poly, gens: &[Expr]) -> Expr {
  let mut monos: Vec<(&Mono, &Q)> = p.t.iter().collect();
  // Ascending from the last generator to the first; the zero vector (the
  // constant term) sorts first.
  monos.sort_by(|(a, _), (b, _)| a.iter().rev().cmp(b.iter().rev()));
  let mut terms: Vec<Expr> = monos
    .into_iter()
    .map(|(m, c)| {
      let factors: Vec<Expr> = m
        .iter()
        .zip(gens)
        .filter(|(e, _)| **e > 0)
        .map(|(&e, g)| {
          let v = g.clone();
          if e == 1 {
            v
          } else {
            pow2(v, Expr::Integer(e as i128))
          }
        })
        .collect();
      if factors.is_empty() {
        q_to_expr(c)
      } else if c.d.is_one() && c.n.is_one() {
        build_product(factors)
      } else if c.d.is_one() && c.n.is_negative() && c.n.magnitude().is_one() {
        negate_term(&build_product(factors))
      } else {
        multiply_exprs(&q_to_expr(c), &build_product(factors))
      }
    })
    .collect();
  match terms.len() {
    0 => Expr::Integer(0),
    1 => terms.pop().unwrap(),
    _ => plus(terms),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::cell::Cell;

  thread_local! {
    /// Route `expand_and_combine` through the generic expander only, to
    /// compare the fast path against it.
    pub(super) static GENERIC_ONLY: Cell<bool> = const { Cell::new(false) };
  }

  fn eval(source: &str) -> Expr {
    let parsed = crate::parse_to_expr(source).unwrap();
    crate::evaluator::evaluate_expr_to_expr(&parsed).unwrap()
  }

  fn generic(expr: &Expr) -> Expr {
    GENERIC_ONLY.set(true);
    let result = expand_and_combine(expr);
    GENERIC_ONLY.set(false);
    result
  }

  /// A small deterministic xorshift generator, so failures reproduce.
  struct Rng(u64);
  impl Rng {
    fn next(&mut self, n: u64) -> u64 {
      self.0 ^= self.0 << 13;
      self.0 ^= self.0 >> 7;
      self.0 ^= self.0 << 17;
      self.0 % n
    }
  }

  fn random_polynomial(
    rng: &mut Rng,
    depth: u32,
    atoms: &[&str],
    coeffs: &[&str],
  ) -> String {
    if depth == 0 || rng.next(4) == 0 {
      return match rng.next(3) {
        0 => coeffs[rng.next(coeffs.len() as u64) as usize].to_string(),
        _ => atoms[rng.next(atoms.len() as u64) as usize].to_string(),
      };
    }
    let mut operand = || random_polynomial(rng, depth - 1, atoms, coeffs);
    let (l, r) = (operand(), operand());
    match rng.next(5) {
      0 | 1 => format!("({l} + {r})"),
      2 => format!("({l} - {r})"),
      3 => format!("({l})*({r})"),
      _ => format!("({l})^{}", 2 + rng.next(3)),
    }
  }

  /// Structurally identical to the generic expander. Restricted to integer
  /// coefficients and single-letter symbols, where the generic path is known
  /// to be canonical (it orders `x` after `x1` by printed form, and leaves
  /// `-(-5/3)` coefficients unnormalized).
  #[test]
  fn fast_path_matches_the_generic_expander() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..1000 {
      let source = random_polynomial(
        &mut rng,
        4,
        &["a", "b", "c", "x", "y", "z"],
        &["2", "-1", "-3", "7", "100"],
      );
      let expr = eval(&source);
      let fast = expand_polynomial(&expr).map(|e| format!("{e:?}"));
      if let Some(fast) = fast {
        assert_eq!(fast, format!("{:?}", generic(&expr)), "Expand[{source}]");
      }
    }
  }

  /// With rational coefficients and multi-character symbols: the result has
  /// the input's value, and its terms are already in `Plus` order.
  #[test]
  fn fast_path_is_canonical_and_correct() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let values = "{a -> 3, b -> -2, x -> 5/7, x1 -> 11, x10 -> -13, \
                  x2 -> 2/3, zz -> 17, v -> -19}";
    for _ in 0..500 {
      let source = random_polynomial(
        &mut rng,
        4,
        &["a", "b", "x", "x1", "x10", "x2", "zz", "v"],
        &["2", "-1", "-3", "1/2", "-5/3", "100"],
      );
      let Some(fast) = expand_polynomial(&eval(&source)) else {
        continue;
      };
      // Re-sorting the evaluated terms must not move any of them.
      let terms = |e: &Expr| -> Vec<String> {
        collect_additive_terms(e)
          .iter()
          .map(|t| {
            let t = crate::evaluator::evaluate_expr_to_expr(t).unwrap();
            crate::syntax::expr_to_string(&t)
          })
          .collect()
      };
      let reevaluated = crate::evaluator::evaluate_expr_to_expr(&fast).unwrap();
      assert_eq!(terms(&fast), terms(&reevaluated), "Expand[{source}]");
      let printed = crate::syntax::expr_to_string(&fast);
      let at = |e: &str| {
        crate::syntax::expr_to_string(&eval(&format!("({e}) /. {values}")))
      };
      assert_eq!(at(&printed), at(&source), "Expand[{source}]");
    }
  }

  /// With calls of undefined functions among the atoms: once evaluated, the
  /// kernel path gives exactly what the generic expander gives.
  #[test]
  fn kernel_path_matches_the_generic_expander() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let mut expanded = 0;
    for _ in 0..500 {
      let source = random_polynomial(
        &mut rng,
        4,
        &["a", "x", "f[x]", "g[a, b]", "x[1]", "x[2]", "f[x + 1]"],
        &["2", "-1", "-3", "1/2", "7"],
      );
      let expr = eval(&source);
      let Some(fast) = expand_polynomial_with_kernels(&expr) else {
        continue;
      };
      expanded += 1;
      let generic = crate::evaluator::evaluate_expr_to_expr(&generic(&expr));
      assert_eq!(
        crate::syntax::expr_to_string(&fast),
        crate::syntax::expr_to_string(&generic.unwrap()),
        "Expand[{source}]"
      );
    }
    assert!(expanded > 400);
  }

  #[test]
  fn kernel_path_declines_other_calls() {
    for source in ["Sin[x] (1 + f[x])^2", "(1 + f[x])^(1/2)", "1/(1 + f[x])"] {
      assert!(
        expand_polynomial_with_kernels(&eval(source)).is_none(),
        "{source}"
      );
    }
  }

  #[test]
  fn fast_path_matches_on_large_powers() {
    for source in [
      "(a + b + c + d - e - f)^5",
      "(x - 2 y + 3/2 z)^6",
      "(1 + x)^12 (1 - y)^3",
      "(a b - c^2 + 3)^4",
    ] {
      let expr = eval(source);
      let fast = expand_polynomial(&expr).expect(source);
      assert_eq!(format!("{fast:?}"), format!("{:?}", generic(&expr)));
    }
  }

  #[test]
  fn fast_path_declines_non_polynomials() {
    for source in [
      "Sin[x] (1 + x)^2",
      "(1 + x)^(1/2)",
      "(I + x)^2",
      "1/(1 + x)",
    ] {
      assert!(expand_polynomial(&eval(source)).is_none(), "{source}");
    }
  }
}
