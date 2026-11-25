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

/// A symbol, or a call of a head, without built-in meaning: `Times` only
/// orders it among the other factors (`a`, `g[x[1], x[2]]`).
fn is_plain_factor(e: &Expr) -> bool {
  match e {
    Expr::Identifier(name) | Expr::FunctionCall { name, .. } => {
      is_plain_symbol(name)
    }
    _ => false,
  }
}

/// A term of a plain sum: an exact number, a plain factor, or a product of
/// an optional exact coefficient and plain factors.
fn is_plain_term(e: &Expr) -> bool {
  match e {
    Expr::FunctionCall { name, args } if name == "Times" => {
      let factors = match args.first() {
        Some(c) if super::arithmetic::expr_to_coeff(c).is_some() => &args[1..],
        _ => &args[..],
      };
      !factors.is_empty() && factors.iter().all(is_plain_factor)
    }
    _ => is_plain_factor(e) || super::arithmetic::expr_to_coeff(e).is_some(),
  }
}

/// A sum of plain terms, not all of them numbers, which as a factor of a
/// product is kept whole and only ordered.
fn is_plain_sum(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, args } if name == "Plus"
    && args.iter().all(is_plain_term)
    && args.iter().any(|t| super::arithmetic::expr_to_coeff(t).is_none()))
}

/// `Times[args]` (already flattened) when, besides exact numbers, the
/// factors are distinct plain factors and plain sums: nothing combines, so the product is the
/// coefficient times the factors in canonical order. A sum with a
/// coefficient other than 1 is left to the general code, which distributes
/// `-1` over it. Products of function calls (the terms of an expanded trace
/// of `g[a, b]` metric tensors) otherwise went through every rewrite the
/// general `Times` knows.
pub(super) fn plain_factor_product(args: &[Expr]) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  let mut coeff = super::arithmetic::Coeff::Exact(1, 1);
  let mut factors: Vec<&Expr> = Vec::with_capacity(args.len());
  let mut has_sum = false;
  for arg in args {
    if let Some(c) = super::arithmetic::expr_to_coeff(arg) {
      coeff = coeff.mul(&c);
    } else if is_plain_factor(arg) {
      factors.push(arg);
    } else if is_plain_sum(arg) {
      has_sum = true;
      factors.push(arg);
    } else {
      return None;
    }
  }
  if factors.is_empty() || coeff.is_zero() || (has_sum && !coeff.is_one()) {
    return None;
  }
  factors.sort_by(|a, b| super::arithmetic::times_factor_order(a, b));
  // Equal neighbors would merge into a power.
  if !factors
    .windows(2)
    .all(|w| super::arithmetic::times_factor_order(w[0], w[1]).is_lt())
  {
    return None;
  }
  let count = factors.len();
  Some(ordered_plain_product(
    &coeff,
    factors.into_iter().cloned(),
    count,
  ))
}

/// The product of `coeff` and the `count` (at least one) `factors`, which
/// are distinct plain factors in canonical order.
fn ordered_plain_product(
  coeff: &super::arithmetic::Coeff,
  factors: impl Iterator<Item = Expr>,
  count: usize,
) -> Expr {
  let mut out: Vec<Expr> = Vec::with_capacity(count + 1);
  if !coeff.is_one() {
    out.push(coeff.to_expr());
  }
  out.extend(factors);
  if out.len() == 1 {
    out.pop().unwrap()
  } else {
    times(out)
  }
}

/// The canonical orders of a set of distinct plain factors that many
/// products share (the generators of an expanded polynomial), found once
/// instead of by sorting the factors of every product and comparing the
/// factors of every pair of terms.
pub(crate) struct PlainFactorOrders {
  /// The indices of the factors in the order `Times` puts them.
  pub(crate) times: Vec<usize>,
  /// The rank of each factor in the order `Plus` compares the factors of
  /// its terms by (see `plain_call_sum`). `None` unless every factor is a
  /// call and no two of them tie.
  pub(crate) plus_rank: Option<Vec<usize>>,
}

/// The orders of `factors`; `None` unless they are plain factors that
/// `Times` keeps apart (see `plain_factor_product`).
pub(crate) fn plain_factor_orders(
  factors: &[Expr],
) -> Option<PlainFactorOrders> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  if !factors.iter().all(is_plain_factor) {
    return None;
  }
  let order_by = |cmp: fn(&Expr, &Expr) -> std::cmp::Ordering| {
    let mut order: Vec<usize> = (0..factors.len()).collect();
    order.sort_by(|&a, &b| cmp(&factors[a], &factors[b]));
    order
      .windows(2)
      .all(|w| cmp(&factors[w[0]], &factors[w[1]]).is_lt())
      .then_some(order)
  };
  let times = order_by(super::arithmetic::times_factor_order)?;
  let plus_rank = factors
    .iter()
    .all(is_plain_call)
    .then(|| order_by(super::arithmetic::wl_arg_order))
    .flatten()
    .map(|order| {
      let mut rank = vec![0; factors.len()];
      for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
      }
      rank
    });
  Some(PlainFactorOrders { times, plus_rank })
}

/// `coefficient * factors` for an exact non-zero `coefficient` and distinct
/// plain `factors` given in canonical order (see `plain_factor_orders`): the
/// product `Times` makes of them, without sorting them again.
pub(crate) fn plain_product_in_order(
  coefficient: &Expr,
  factors: Vec<Expr>,
) -> Option<Expr> {
  let coeff = super::arithmetic::expr_to_coeff(coefficient)?;
  if coeff.is_zero() || factors.is_empty() {
    return None;
  }
  let count = factors.len();
  Some(ordered_plain_product(&coeff, factors.into_iter(), count))
}

/// A call of a head without built-in meaning.
fn is_plain_call(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, .. } if is_plain_symbol(name))
}

/// The terms of a sum whose arguments may themselves be sums (`Plus` is
/// Flat: sums arrive nested whenever a chain is evaluated pairwise).
fn sum_terms(args: &[Expr]) -> Vec<&Expr> {
  let mut flat: Vec<&Expr> = Vec::with_capacity(args.len());
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
      other => flat.push(other),
    }
  }
  flat
}

/// The factors of a product with an optional exact coefficient
/// (`4*g[a, b]*g[c, d]`), or the single factor a term that is no product
/// is.
fn term_factors(term: &Expr) -> &[Expr] {
  match term {
    Expr::FunctionCall { name, args } if name == "Times" => {
      match args.first() {
        Some(c) if super::arithmetic::expr_to_coeff(c).is_some() => &args[1..],
        _ => &args[..],
      }
    }
    _ => std::slice::from_ref(term),
  }
}

/// The order `Plus` gives `terms` when every one of them is a canonical
/// product of calls of heads without built-in meaning, as the positions of
/// the terms; `None` when one is not, or when two of them combine.
///
/// `Plus` orders such terms from their last factor back, the factors taken
/// in `wl_arg_order` (see `call_led_order`), and a term whose factors are
/// the other's last ones first. The same few calls make up thousands of
/// terms (an expanded trace of `g[a, b]` metric tensors), so each distinct
/// call is ranked once — among the others in `wl_arg_order`, and in the
/// order `Times` keeps its factors in — and the terms are checked and
/// sorted by those ranks, where comparing the calls themselves for every
/// pair of neighbors and of terms made such sums crawl.
fn plain_call_term_order(terms: &[&Expr]) -> Option<Vec<usize>> {
  use super::arithmetic::{times_factor_order, wl_arg_order};
  let first = *terms.first()?;
  // A call by the comparator the terms are ordered with, equivalence
  // classes included.
  struct Factor<'a>(&'a Expr);
  impl PartialEq for Factor<'_> {
    fn eq(&self, other: &Self) -> bool {
      self.cmp(other).is_eq()
    }
  }
  impl Eq for Factor<'_> {}
  impl PartialOrd for Factor<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
      Some(self.cmp(other))
    }
  }
  impl Ord for Factor<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
      wl_arg_order(self.0, other.0)
    }
  }
  // ExprList's interior mutability only caches derived values. The borrowed
  // expressions cannot change here, so wl_arg_order remains stable.
  #[allow(clippy::mutable_key_type)]
  let mut classes: std::collections::BTreeMap<Factor, usize> =
    std::collections::BTreeMap::new();
  // The class of each call already seen, by its head and where its
  // arguments are stored: the borrowed terms stay alive throughout, so a
  // shared call met again is the same call. No pointer is dereferenced.
  let mut by_address: rustc_hash::FxHashMap<(*const Expr, usize, &str), usize> =
    rustc_hash::FxHashMap::default();
  // The classes of the factors of all terms, one term after the other,
  // with where each term's end.
  let mut ids: Vec<usize> = Vec::with_capacity(terms.len() * 4);
  let mut ends: Vec<usize> = Vec::with_capacity(terms.len());
  for term in terms {
    let factors = term_factors(term);
    if factors.is_empty() {
      return None;
    }
    for factor in factors {
      let Expr::FunctionCall { name, args } = factor else {
        return None;
      };
      let address = (args.as_ptr(), args.len(), name.as_str());
      let id = if let Some(&id) = by_address.get(&address) {
        id
      } else {
        if !is_plain_symbol(name) {
          return None;
        }
        let next = classes.len();
        let id = *classes.entry(Factor(factor)).or_insert(next);
        by_address.insert(address, id);
        id
      };
      ids.push(id);
    }
    ends.push(ids.len());
  }
  // Rank the classes both ways.
  let mut plus_rank = vec![0; classes.len()];
  let mut members: Vec<&Expr> = vec![first; classes.len()];
  for (rank, (factor, &id)) in classes.iter().enumerate() {
    plus_rank[id] = rank;
    members[id] = factor.0;
  }
  let mut in_times_order: Vec<usize> = (0..members.len()).collect();
  in_times_order.sort_by(|&a, &b| times_factor_order(members[a], members[b]));
  let mut times_rank = vec![0; members.len()];
  let mut rank = 0;
  for (i, &id) in in_times_order.iter().enumerate() {
    if i > 0
      && times_factor_order(members[in_times_order[i - 1]], members[id]).is_lt()
    {
      rank += 1;
    }
    times_rank[id] = rank;
  }
  // Every product has to be canonical: its factors strictly ascending, as
  // `Times` leaves them (equal neighbors would merge into a power). Its
  // key is then the ranks of its factors, the last one first.
  let mut start = 0;
  for &end in &ends {
    let key = &mut ids[start..end];
    if !key.windows(2).all(|w| times_rank[w[0]] < times_rank[w[1]]) {
      return None;
    }
    for id in key.iter_mut() {
      *id = plus_rank[*id];
    }
    key.sort_unstable_by(|a, b| b.cmp(a));
    start = end;
  }
  let key = |term: usize| {
    let start = if term == 0 { 0 } else { ends[term - 1] };
    &ids[start..ends[term]]
  };
  let mut order: Vec<usize> = (0..terms.len()).collect();
  order.sort_by(|&a, &b| key(a).cmp(key(b)));
  // Equal neighbors are like terms, which combine.
  if order.windows(2).any(|w| key(w[0]) == key(w[1])) {
    return None;
  }
  Some(order)
}

/// `Plus[args]` when every term is a canonical product of calls of heads
/// without built-in meaning and no two terms combine (see
/// `plain_call_term_order`).
pub(super) fn plain_call_sum(args: &[Expr]) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  let terms = sum_terms(args);
  if terms.len() < 2 {
    return None;
  }
  let order = plain_call_term_order(&terms)?;
  Some(plus(order.into_iter().map(|i| terms[i].clone()).collect()))
}

/// `plain_call_sum` for a caller that owns the terms, which are moved into
/// the sum instead of copied (the summands of a `Sum`, each of them a long
/// sum of its own). Hands the terms back when they are not such a sum.
pub(crate) fn plain_call_sum_owned(args: Vec<Expr>) -> Result<Expr, Vec<Expr>> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return Err(args);
  }
  let order = {
    let terms = sum_terms(&args);
    if terms.len() < 2 {
      return Err(args);
    }
    match plain_call_term_order(&terms) {
      Some(order) => order,
      None => return Err(args),
    }
  };
  let mut terms: Vec<Option<Expr>> = Vec::with_capacity(order.len());
  for arg in args {
    take_sum_terms(arg, &mut terms);
  }
  Ok(plus(
    order.into_iter().filter_map(|i| terms[i].take()).collect(),
  ))
}

/// `sum_terms` by value: the terms in the same order, moved out of the sums
/// they are in (copied out of a sum that is shared with another expression).
fn take_sum_terms(mut arg: Expr, out: &mut Vec<Option<Expr>>) {
  match &mut arg {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for term in std::mem::take(args).into_inner() {
        take_sum_terms(term, out);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let left = std::mem::replace(&mut **left, Expr::Integer(0));
      let right = std::mem::replace(&mut **right, Expr::Integer(0));
      take_sum_terms(left, out);
      take_sum_terms(right, out);
    }
    _ => out.push(Some(arg)),
  }
}

/// `-1 * sum` for a sum of canonical products of calls of heads without
/// built-in meaning: `Times` distributes the `-1`, which only changes the
/// sign of every coefficient — the factors, and with them the order of the
/// terms, stay. The general code multiplies every term and adds the
/// products up one by one through all it knows.
pub(super) fn negated_plain_call_sum(args: &[Expr]) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  let ([Expr::Integer(-1), sum] | [sum, Expr::Integer(-1)]) = args else {
    return None;
  };
  let Expr::FunctionCall { name, args: terms } = sum else {
    return None;
  };
  if name != "Plus" || terms.len() < 2 {
    return None;
  }
  // The terms are checked (and put in order) only once negated; a term
  // that is not such a product makes that fail.
  let mut negated: Vec<Option<Expr>> = terms
    .iter()
    .map(|term| {
      let factors = term_factors(term);
      let coeff = match term {
        Expr::FunctionCall { name, args }
          if name == "Times" && factors.len() < args.len() =>
        {
          super::arithmetic::expr_to_coeff(&args[0])?
        }
        _ => super::arithmetic::Coeff::Exact(1, 1),
      };
      (!factors.is_empty()).then(|| {
        Some(ordered_plain_product(
          &coeff.negate(),
          factors.iter().cloned(),
          factors.len(),
        ))
      })
    })
    .collect::<Option<_>>()?;
  let order = {
    let terms: Vec<&Expr> = negated.iter().flatten().collect();
    plain_call_term_order(&terms)?
  };
  Some(plus(
    order
      .into_iter()
      .filter_map(|i| negated[i].take())
      .collect(),
  ))
}

/// `Plus[args]` when every term is a canonical monomial (or one exact
/// constant) and no two terms combine.
pub(super) fn canonical_polynomial_sum(args: &[Expr]) -> Option<Expr> {
  #[cfg(test)]
  if tests::GENERIC_ONLY.get() {
    return None;
  }
  let args = sum_terms(args);
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
        let (ea, eb) = (id_expr(a), id_expr(b));
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

  const CALLS: [&str; 8] = [
    "g[x[1], x[2]]",
    "g[x[2], x[1]]",
    "g[x[1], x[10]]",
    "g[a]",
    "f[a, b]",
    "h[]",
    "a",
    "zz",
  ];

  fn random_call_term(rng: &mut Rng) -> String {
    let mut factors = vec![COEFFS[rng.next(COEFFS.len())].to_string()];
    for _ in 0..rng.next(4) {
      factors.push(CALLS[rng.next(CALLS.len())].to_string());
    }
    match factors.join(" ").trim() {
      "" | "-" => "1".to_string(),
      t => format!("({t})"),
    }
  }

  /// Products of calls of undefined heads, of sums of those, and of
  /// coefficients, with repeated and out-of-order factors.
  #[test]
  fn call_products_match_the_general_times() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for _ in 0..2000 {
      let factors: Vec<Expr> = (0..=rng.next(4))
        .map(|_| match rng.next(4) {
          0 => eval(&format!(
            "{} + {}",
            random_call_term(&mut rng),
            random_call_term(&mut rng)
          )),
          _ => eval(&random_call_term(&mut rng)),
        })
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

  /// Sums of products of calls, with like terms, repeated factors and
  /// symbols mixed in to exercise the fallbacks too.
  #[test]
  fn call_sums_match_the_general_plus() {
    let mut rng = Rng(0x3c6e_f372_fe94_f82b);
    for _ in 0..2000 {
      let mut terms: Vec<Expr> = (0..2 + rng.next(8))
        .map(|_| eval(&random_call_term(&mut rng)))
        .collect();
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
  fn ranked_call_sums_match_the_general_plus() {
    for count in [31, 32, 64, 128] {
      let mut terms: Vec<_> = (1..=count)
        .rev()
        .map(|i| eval(&format!("g[x[1], x[2]] h[x[{i}]]")))
        .collect();
      for duplicate in [false, true] {
        if duplicate {
          terms.push(terms[0].clone());
        }
        let fast = plus_ast(&terms).unwrap();
        let general = generic(|| plus_ast(&terms)).unwrap();
        assert_eq!(format!("{fast:?}"), format!("{general:?}"));
      }
    }
  }

  /// `-1` times a sum of call products, among them sums the fast path has to
  /// leave alone (a constant or a symbol among the terms, a power).
  #[test]
  fn negated_call_sums_match_the_general_times() {
    let mut rng = Rng(0x51ed_270b_9f3c_a6d1);
    let mut negated = 0;
    for _ in 0..2000 {
      let source = (0..2 + rng.next(6))
        .map(|_| random_call_term(&mut rng))
        .collect::<Vec<_>>()
        .join(" + ");
      let sum = eval(&source);
      for args in [
        [Expr::Integer(-1), sum.clone()],
        [sum.clone(), Expr::Integer(-1)],
      ] {
        negated += usize::from(negated_plain_call_sum(&args).is_some());
        let fast = times_ast(&args).unwrap();
        let general = generic(|| times_ast(&args)).unwrap();
        assert_eq!(format!("{fast:?}"), format!("{general:?}"), "{source}");
      }
    }
    assert!(negated > 100, "only {negated} sums took the fast path");
  }

  /// The same call met in many terms is recognized by where its arguments
  /// are stored once they are shared, as `Expand` leaves them.
  #[test]
  fn sums_of_shared_calls_match_the_general_plus() {
    let mut rng = Rng(0x7f4a_7c15_9e37_79b9);
    let pool: Vec<Expr> = (1..=12)
      .map(|i| {
        let mut call = eval(&format!("g[x[{}], x[{}]]", i % 5, i));
        if let Expr::FunctionCall { args, .. } = &mut call {
          args.share();
        }
        call
      })
      .collect();
    for _ in 0..300 {
      let terms: Vec<Expr> = (0..2 + rng.next(40))
        .map(|_| {
          let mut factors = vec![eval(COEFFS[3 + rng.next(COEFFS.len() - 3)])];
          let mut next = rng.next(3);
          while next < pool.len() {
            factors.push(pool[next].clone());
            next += 1 + rng.next(5);
          }
          times_ast(&factors).unwrap()
        })
        .collect();
      let fast = plus_ast(&terms).unwrap();
      let general = generic(|| plus_ast(&terms)).unwrap();
      assert_eq!(
        crate::syntax::expr_to_string(&fast),
        crate::syntax::expr_to_string(&general)
      );
    }
  }

  /// Owned terms, nested sums among them, are moved into the same sum the
  /// borrowing path builds, and handed back when they are no such sum.
  #[test]
  fn owned_call_sums_match_the_general_plus() {
    let mut rng = Rng(0x1b87_3593_cc9e_2d51);
    let mut summed = 0;
    for _ in 0..1000 {
      let mut terms: Vec<Expr> = (0..2 + rng.next(8))
        .map(|_| eval(&random_call_term(&mut rng)))
        .collect();
      if terms.len() > 3 && rng.next(2) == 0 {
        let tail = terms.split_off(2);
        terms.push(plus(tail));
      }
      let general = generic(|| plus_ast(&terms)).unwrap();
      match plain_call_sum_owned(terms.clone()) {
        Ok(fast) => {
          summed += 1;
          assert_eq!(format!("{fast:?}"), format!("{general:?}"), "{terms:?}");
        }
        Err(back) => {
          assert_eq!(format!("{back:?}"), format!("{terms:?}"));
          assert!(plain_call_sum(&terms).is_none(), "{terms:?}");
        }
      }
    }
    assert!(summed > 10, "only {summed} sums took the fast path");
  }

  /// The orders found once for a set of factors are the orders `Times` and
  /// `Plus` give them.
  #[test]
  fn factor_orders_are_the_orders_of_times_and_plus() {
    let factors: Vec<Expr> = [
      "g[x[1], x[2]]",
      "g[x[2], x[1]]",
      "g[x[1], x[10]]",
      "h[]",
      "g[a]",
      "f[a, b]",
      "F[a]",
      "g[b]",
    ]
    .iter()
    .map(|s| eval(s))
    .collect();
    let orders = plain_factor_orders(&factors).unwrap();
    let product = generic(|| times_ast(&factors)).unwrap();
    let ordered: Vec<Expr> =
      orders.times.iter().map(|&i| factors[i].clone()).collect();
    assert_eq!(format!("{product:?}"), format!("{:?}", times(ordered)));
    let sum = generic(|| plus_ast(&factors)).unwrap();
    let mut by_rank: Vec<usize> = (0..factors.len()).collect();
    let rank = orders.plus_rank.unwrap();
    by_rank.sort_by_key(|&i| rank[i]);
    let ordered: Vec<Expr> =
      by_rank.into_iter().map(|i| factors[i].clone()).collect();
    assert_eq!(format!("{sum:?}"), format!("{:?}", plus(ordered)));
    // Symbols among the factors have no place among the calls of a sum, and
    // equal factors or ones with built-in meaning no order at all.
    let with_symbol = [eval("g[a]"), eval("zz")];
    assert!(
      plain_factor_orders(&with_symbol)
        .unwrap()
        .plus_rank
        .is_none()
    );
    assert!(plain_factor_orders(&[eval("g[a]"), eval("g[a]")]).is_none());
    assert!(plain_factor_orders(&[eval("g[a]"), eval("Sin[a]")]).is_none());
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
