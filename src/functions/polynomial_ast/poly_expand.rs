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
use std::collections::BTreeSet;

/// Expand and combine `expr` when it is a polynomial with rational
/// coefficients in plain (non-system) symbols; None otherwise.
pub(super) fn expand_polynomial(expr: &Expr) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  let mut names = BTreeSet::new();
  if !collect_generators(expr, &mut names) || names.is_empty() {
    return None;
  }
  let mut gens: Vec<String> = names.into_iter().collect();
  gens.sort_by(|a, b| wolfram_string_order(a, b).cmp(&0).reverse());
  let p = to_poly(expr, &gens)?;
  Some(poly_to_canonical_expr(&p, &gens))
}

/// `expr` as a polynomial over its symbols, which are returned sorted by
/// name; None unless it is a polynomial with rational coefficients in plain
/// symbols.
pub(super) fn polynomial_in_symbols(
  expr: &Expr,
) -> Option<(Poly, Vec<String>)> {
  let mut names = BTreeSet::new();
  if !collect_generators(expr, &mut names) {
    return None;
  }
  let gens: Vec<String> = names.into_iter().collect();
  Some((to_poly(expr, &gens)?, gens))
}

/// Check that `expr` is a polynomial the fast path handles, collecting the
/// names of its symbols.
fn collect_generators(expr: &Expr, names: &mut BTreeSet<String>) -> bool {
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
      "Plus" | "Times" => args.iter().all(|a| collect_generators(a, names)),
      "Power" if args.len() == 2 => {
        power_exponent(&args[1]).is_some()
          && collect_generators(&args[0], names)
      }
      _ => false,
    },
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Times => {
        collect_generators(left, names) && collect_generators(right, names)
      }
      BinaryOperator::Power => {
        power_exponent(right).is_some() && collect_generators(left, names)
      }
      BinaryOperator::Divide => {
        rational_literal(right).is_some_and(|q| !q.is_zero())
          && collect_generators(left, names)
      }
      _ => false,
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => collect_generators(operand, names),
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

/// A single term `c * x^i * y^j * …` straight to its exponent vector, without
/// building and multiplying one-term polynomials. False when `expr` is not a
/// product of rational numbers and powers of the generators.
fn monomial(expr: &Expr, gens: &[String], m: &mut Mono, c: &mut Q) -> bool {
  let mut symbol_power = |base: &Expr, k: u32| match base {
    Expr::Identifier(name) => match gens.iter().position(|g| g == name) {
      Some(i) => {
        m[i] += k;
        true
      }
      None => false,
    },
    _ => false,
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
      None => false,
    },
  }
}

fn to_poly(expr: &Expr, gens: &[String]) -> Option<Poly> {
  let nv = gens.len();
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
      for (m, c) in to_poly(item, gens)?.t {
        acc.add_term(m, c);
      }
    }
    Some(acc)
  };
  let product = |items: &mut dyn Iterator<Item = &Expr>| -> Option<Poly> {
    let mut acc = Poly::one(nv);
    for item in items {
      acc = acc.mul(&to_poly(item, gens)?);
    }
    Some(acc)
  };
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => sum(&mut args.iter()),
      "Times" => product(&mut args.iter()),
      "Power" => {
        Some(pow(&to_poly(&args[0], gens)?, power_exponent(&args[1])?))
      }
      _ => None,
    },
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus => sum(&mut [left, right].into_iter().map(|b| &**b)),
      BinaryOperator::Minus => {
        let r = to_poly(right, gens)?.scale(&Q::one().neg());
        let mut l = to_poly(left, gens)?;
        for (m, c) in r.t {
          l.add_term(m, c);
        }
        Some(l)
      }
      BinaryOperator::Times => {
        product(&mut [left, right].into_iter().map(|b| &**b))
      }
      BinaryOperator::Power => {
        Some(pow(&to_poly(left, gens)?, power_exponent(right)?))
      }
      BinaryOperator::Divide => Some(
        to_poly(left, gens)?.scale(&Q::one().div(&rational_literal(right)?)),
      ),
      _ => None,
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(to_poly(operand, gens)?.scale(&Q::one().neg())),
    _ => None,
  }
}

/// `p^n` by repeated multiplication with the (usually short) base, which for
/// sparse multivariate bases is cheaper than squaring the growing result.
fn pow(p: &Poly, n: u32) -> Poly {
  let mut acc = Poly::one(p.nv);
  for _ in 0..n {
    acc = acc.mul(p);
  }
  acc
}

/// Build the canonical expanded form of `p` over the generator symbols.
fn poly_to_canonical_expr(p: &Poly, gens: &[String]) -> Expr {
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
          let v = Expr::Identifier(g.clone());
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
