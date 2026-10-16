//! Fast paths for re-evaluating polynomials that are already canonical.
//!
//! Every use of a symbol holding an expanded polynomial re-evaluates it:
//! each term goes through the general `Times` and the whole sum through the
//! general `Plus`, whose term comparison is far too heavy to sort thousands
//! of monomials. For terms that are plain monomials — an optional exact
//! coefficient times powers of distinct non-system symbols — both results
//! are known in advance: such a product is already canonical when its
//! symbols are in canonical order, and a sum of such terms with pairwise
//! different exponent vectors combines nothing and only needs sorting by
//! those vectors, compared from the last symbol to the first (the order
//! `Plus` gives polynomial terms). Anything else falls through to the
//! general code.

use super::*;
use crate::functions::list_helpers_ast::wolfram_string_order;
use std::collections::{HashMap, HashSet};

/// The base and exponent of a plain symbol or its integer power ≥ 2, in the
/// form evaluation leaves them (`BinaryOp` powers).
fn symbol_power(e: &Expr) -> Option<(&Expr, u32)> {
  let (base, exp) = match e {
    Expr::Identifier(_) => (e, 1),
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => match right.as_ref() {
      Expr::Integer(n) if *n >= 2 => (left.as_ref(), u32::try_from(*n).ok()?),
      _ => return None,
    },
    _ => return None,
  };
  match base {
    Expr::Identifier(name) if is_plain_symbol(name) => Some((base, exp)),
    _ => None,
  }
}

/// A non-system symbol without an explicit context, which therefore orders
/// by its name alone.
pub(crate) fn is_plain_symbol(name: &str) -> bool {
  !name.contains('`') && !crate::evaluator::is_builtin_symbol(name)
}

/// Canonical order of two plain symbols.
pub(crate) fn symbol_order(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  match (a, b) {
    (Expr::Identifier(a), Expr::Identifier(b)) => {
      wolfram_string_order(a, b).cmp(&0).reverse()
    }
    _ => std::cmp::Ordering::Equal,
  }
}

/// An exact rational other than 0 and 1, as a product coefficient.
fn is_coefficient(e: &Expr) -> bool {
  match e {
    Expr::Integer(n) => *n != 0 && *n != 1,
    Expr::BigInteger(_) => true,
    Expr::FunctionCall { name, args } => {
      name == "Rational"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(_))
        && matches!(&args[1], Expr::Integer(d) if *d > 1)
    }
    _ => false,
  }
}

/// The symbol powers of a canonical monomial product: an optional leading
/// coefficient, then powers of strictly increasing symbols.
fn monomial_factors(args: &[Expr]) -> Option<Vec<(&Expr, u32)>> {
  let factors = match args.first() {
    Some(c) if is_coefficient(c) => &args[1..],
    _ => args,
  };
  if factors.is_empty() || args.len() < 2 {
    return None;
  }
  let powers: Vec<(&Expr, u32)> =
    factors.iter().map(symbol_power).collect::<Option<_>>()?;
  powers
    .windows(2)
    .all(|w| symbol_order(w[0].0, w[1].0).is_lt())
    .then_some(powers)
}

/// `Times[args]` when the product is already canonical.
pub(super) fn canonical_monomial_product(args: &[Expr]) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  monomial_factors(args)?;
  Some(times(args.to_vec()))
}

fn is_sum(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, .. } if name == "Plus")
    || matches!(
      e,
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        ..
      }
    )
}

/// `Plus[args]` when every term is a canonical monomial (or one exact
/// constant) and no two terms combine.
pub(super) fn canonical_polynomial_sum(args: &[Expr]) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  // `Plus` is Flat: sums arrive nested whenever a chain is evaluated pairwise.
  if args.iter().any(is_sum) {
    let mut flat: Vec<Expr> = Vec::with_capacity(args.len() + 1);
    let mut stack: Vec<&Expr> = args.iter().rev().collect();
    while let Some(arg) = stack.pop() {
      match arg {
        Expr::FunctionCall { name, args } if name == "Plus" => {
          stack.extend(args.iter().rev());
        }
        Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left,
          right,
        } => {
          stack.push(right);
          stack.push(left);
        }
        other => flat.push(other.clone()),
      }
    }
    return canonical_polynomial_sum(&flat);
  }
  if args.len() < 2 {
    return None;
  }
  let mut constant: Option<&Expr> = None;
  let mut terms: Vec<(&Expr, Vec<(&Expr, u32)>)> = Vec::new();
  for arg in args {
    match arg {
      Expr::FunctionCall { name, args } if name == "Times" => {
        terms.push((arg, monomial_factors(args)?));
      }
      // A machine-size constant: `Plus` leaves a sum with a larger one
      // unsorted, so that case stays with the general code.
      Expr::Integer(n) if *n != 0 && n.unsigned_abs() <= (1u128 << 53) => {
        if constant.replace(arg).is_some() {
          return None;
        }
      }
      Expr::FunctionCall { name, .. }
        if name == "Rational" && is_coefficient(arg) =>
      {
        if constant.replace(arg).is_some() {
          return None;
        }
      }
      _ => terms.push((arg, vec![symbol_power(arg)?])),
    }
  }

  // Index the symbols in canonical order.
  let mut symbols: Vec<&Expr> = Vec::new();
  let mut seen: HashSet<&str> = HashSet::new();
  for (_, powers) in &terms {
    for (base, _) in powers {
      if let Expr::Identifier(name) = base
        && seen.insert(name.as_str())
      {
        symbols.push(base);
      }
    }
  }
  symbols.sort_by(|a, b| symbol_order(a, b));
  let position: HashMap<&str, usize> = symbols
    .iter()
    .enumerate()
    .filter_map(|(i, s)| match s {
      Expr::Identifier(name) => Some((name.as_str(), i)),
      _ => None,
    })
    .collect();

  let mut keyed: Vec<(Vec<u32>, &Expr)> = Vec::with_capacity(terms.len());
  let mut distinct: HashSet<Vec<u32>> = HashSet::with_capacity(terms.len());
  for (term, powers) in terms {
    let mut exponents = vec![0u32; symbols.len()];
    for (base, k) in powers {
      if let Expr::Identifier(name) = base {
        exponents[position[name.as_str()]] = k;
      }
    }
    if !distinct.insert(exponents.clone()) {
      return None; // like terms combine
    }
    keyed.push((exponents, term));
  }
  keyed.sort_by(|(a, _), (b, _)| a.iter().rev().cmp(b.iter().rev()));

  let sorted = constant
    .into_iter()
    .chain(keyed.into_iter().map(|(_, t)| t))
    .cloned()
    .collect();
  Some(plus(sorted))
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::cell::Cell;

  thread_local! {
    /// Bypass the fast paths, to compare them with the general code.
    pub(super) static GENERIC_ONLY: Cell<bool> = const { Cell::new(false) };
  }

  fn generic<T>(f: impl FnOnce() -> T) -> T {
    GENERIC_ONLY.set(true);
    let result = f();
    GENERIC_ONLY.set(false);
    result
  }

  fn eval(source: &str) -> Expr {
    let parsed = crate::parse_to_expr(source).unwrap();
    crate::evaluator::evaluate_expr_to_expr(&parsed).unwrap()
  }

  struct Rng(u64);
  impl Rng {
    fn next(&mut self, n: usize) -> usize {
      self.0 ^= self.0 << 13;
      self.0 ^= self.0 >> 7;
      self.0 ^= self.0 << 17;
      (self.0 % n as u64) as usize
    }
  }

  // No two symbols differing only in case: the general `Plus` ties them
  // (`-A^4 + a*A^2`, where the exponent vectors give `a*A^2 - A^4`).
  const SYMBOLS: [&str; 9] =
    ["a", "b", "x", "x1", "x10", "x2", "zz", "$v", "Pi"];
  const COEFFS: [&str; 9] =
    ["", "", "-", "2", "-3", "1/2", "-5/3", "2^70", "2^60"];

  fn random_monomial(rng: &mut Rng) -> String {
    let mut factors = vec![COEFFS[rng.next(COEFFS.len())].to_string()];
    for _ in 0..rng.next(4) {
      let s = SYMBOLS[rng.next(SYMBOLS.len())];
      factors.push(match rng.next(3) {
        0 => format!("{s}^{}", 2 + rng.next(3)),
        _ => s.to_string(),
      });
    }
    match factors.join(" ").trim() {
      "" | "-" => "1".to_string(),
      t => format!("({t})"),
    }
  }

  #[test]
  fn symbol_order_is_the_canonical_order() {
    let names = [
      "a", "A", "ab", "aB", "Ab", "b", "x", "x1", "x10", "x2", "$v",
    ];
    for a in names {
      for b in names {
        let (ea, eb) = (Expr::Identifier(a.into()), Expr::Identifier(b.into()));
        let canonical =
          crate::functions::list_helpers_ast::compare_exprs(&ea, &eb);
        assert_eq!(
          symbol_order(&ea, &eb),
          canonical.cmp(&0).reverse(),
          "{a} {b}"
        );
      }
    }
  }

  /// Already-evaluated factor lists, both canonical and not.
  #[test]
  fn products_match_the_general_times() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..2000 {
      let factors: Vec<Expr> = (0..=rng.next(4))
        .map(|_| eval(&random_monomial(&mut rng)))
        .flat_map(|e| match &e {
          Expr::FunctionCall { name, args } if name == "Times" => args.to_vec(),
          _ => vec![e],
        })
        .collect();
      let fast = times_ast(&factors).unwrap();
      let general = generic(|| times_ast(&factors)).unwrap();
      assert_eq!(format!("{fast:?}"), format!("{general:?}"), "{factors:?}");
    }
  }

  #[test]
  fn sums_match_the_general_plus() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..2000 {
      let mut terms: Vec<Expr> = (0..2 + rng.next(8))
        .map(|_| eval(&random_monomial(&mut rng)))
        .collect();
      if rng.next(3) == 0 {
        terms.push(eval(["7", "-1/3", "2^60", "0"][rng.next(4)]));
      }
      // A nested sum, as pairwise evaluation of a `Plus` chain passes it.
      if terms.len() > 3 && rng.next(2) == 0 {
        let tail = terms.split_off(2);
        terms.push(plus(tail));
      }
      let fast = plus_ast(&terms).unwrap();
      let general = generic(|| plus_ast(&terms)).unwrap();
      assert_eq!(format!("{fast:?}"), format!("{general:?}"), "{terms:?}");
    }
  }

  #[test]
  fn an_expanded_polynomial_reevaluates_unchanged() {
    let expanded = eval("Expand[(a + b + c + d - e - f)^4 (1 - 2 x)^3]");
    let terms = match &expanded {
      Expr::FunctionCall { name, args } if name == "Plus" => args.to_vec(),
      _ => panic!("expected a sum"),
    };
    let fast = plus_ast(&terms).unwrap();
    let general = generic(|| plus_ast(&terms)).unwrap();
    assert_eq!(format!("{fast:?}"), format!("{general:?}"));
  }
}
