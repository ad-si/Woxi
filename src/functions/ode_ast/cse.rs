//! Common-subexpression elimination for compiled `NDSolve` residuals.
//!
//! A large DAE's residuals repeat the same pieces many times over — a
//! column's vapour pressures and activity coefficients appear in every
//! equilibrium, and differentiating a bubble-point constraint copies each of
//! them into several terms of its derivative. [`share`] evaluates every
//! repeated arithmetic subtree once per evaluation: it hoists them into an
//! [`NExpr::Let`] table whose values extend the variable vector, and
//! replaces each occurrence by a variable reference.

use super::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// `expr` with its repeated arithmetic subtrees evaluated once. `base` is
/// the length of the variable vector `expr` is evaluated with: the shared
/// values are read from the slots from `base` on.
pub(super) fn share(expr: NExpr, base: usize) -> NExpr {
  let mut hashes = Vec::new();
  hash_post_order(&expr, &mut hashes);
  let mut counts: HashMap<u64, usize> = HashMap::new();
  for &(h, shareable) in &hashes {
    if shareable {
      *counts.entry(h).or_insert(0) += 1;
    }
  }
  if !counts.values().any(|&c| c >= 2) {
    return expr;
  }
  let mut state = Rewrite {
    hashes: hashes.into_iter(),
    counts,
    index: HashMap::new(),
    table: Vec::new(),
    base,
  };
  let body = state.rewrite(expr);
  if state.table.is_empty() {
    return body;
  }
  NExpr::Let {
    base,
    table: state.table,
    body: Box::new(body),
  }
}

/// The arithmetic children of a node the pass looks into; `None` for a
/// node it leaves whole (a lazily evaluated `If`, an opaque leaf).
fn arithmetic_children(e: &NExpr) -> Option<Vec<&NExpr>> {
  match e {
    NExpr::Const(_) | NExpr::Var(_) => Some(Vec::new()),
    NExpr::Add(items) | NExpr::Mul(items) => Some(items.iter().collect()),
    NExpr::Pow(a, b) | NExpr::Fn2(_, a, b) => Some(vec![a, b]),
    NExpr::Neg(a) | NExpr::Fn1(_, a) => Some(vec![a]),
    _ => None,
  }
}

/// Every node's structural hash, children before parents, with whether the
/// node may be shared (an arithmetic node with operands).
fn hash_post_order(e: &NExpr, out: &mut Vec<(u64, bool)>) -> u64 {
  let mut h = std::collections::hash_map::DefaultHasher::new();
  let Some(children) = arithmetic_children(e) else {
    // A node left whole hashes uniquely: it is never shared.
    out.len().hash(&mut h);
    u64::MAX.hash(&mut h);
    let v = h.finish();
    out.push((v, false));
    return v;
  };
  let child_hashes: Vec<u64> =
    children.iter().map(|c| hash_post_order(c, out)).collect();
  match e {
    NExpr::Const(c) => (0u8, c.to_bits()).hash(&mut h),
    NExpr::Var(i) => (1u8, *i).hash(&mut h),
    NExpr::Add(_) => 2u8.hash(&mut h),
    NExpr::Mul(_) => 3u8.hash(&mut h),
    NExpr::Pow(..) => 4u8.hash(&mut h),
    NExpr::Neg(_) => 5u8.hash(&mut h),
    NExpr::Fn1(f, _) => (6u8, *f as usize).hash(&mut h),
    NExpr::Fn2(f, ..) => (7u8, *f as usize).hash(&mut h),
    _ => unreachable!(),
  }
  child_hashes.hash(&mut h);
  let v = h.finish();
  out.push((v, !matches!(e, NExpr::Const(_) | NExpr::Var(_))));
  v
}

struct Rewrite {
  hashes: std::vec::IntoIter<(u64, bool)>,
  counts: HashMap<u64, usize>,
  /// Table entries by hash (several on a hash collision).
  index: HashMap<u64, Vec<usize>>,
  table: Vec<NExpr>,
  base: usize,
}

impl Rewrite {
  /// `e` with its children rewritten first, in the order
  /// [`hash_post_order`] visited them, and itself replaced by a table
  /// reference when it recurs.
  fn rewrite(&mut self, e: NExpr) -> NExpr {
    if arithmetic_children(&e).is_none() {
      self.hashes.next();
      return e;
    }
    let e = match e {
      NExpr::Add(items) => {
        NExpr::Add(items.into_iter().map(|c| self.rewrite(c)).collect())
      }
      NExpr::Mul(items) => {
        NExpr::Mul(items.into_iter().map(|c| self.rewrite(c)).collect())
      }
      NExpr::Pow(a, b) => {
        let a = self.rewrite(*a);
        NExpr::Pow(Box::new(a), Box::new(self.rewrite(*b)))
      }
      NExpr::Fn2(f, a, b) => {
        let a = self.rewrite(*a);
        NExpr::Fn2(f, Box::new(a), Box::new(self.rewrite(*b)))
      }
      NExpr::Neg(a) => NExpr::Neg(Box::new(self.rewrite(*a))),
      NExpr::Fn1(f, a) => NExpr::Fn1(f, Box::new(self.rewrite(*a))),
      leaf => leaf,
    };
    let (h, shareable) = self.hashes.next().unwrap_or((0, false));
    if !shareable || self.counts.get(&h).is_none_or(|&c| c < 2) {
      return e;
    }
    let slots = self.index.entry(h).or_default();
    if let Some(&k) = slots.iter().find(|&&k| same(&self.table[k], &e)) {
      return NExpr::Var(self.base + k);
    }
    let k = self.table.len();
    slots.push(k);
    self.table.push(e);
    NExpr::Var(self.base + k)
  }
}

/// Structural equality of two rewritten arithmetic nodes.
fn same(a: &NExpr, b: &NExpr) -> bool {
  match (a, b) {
    (NExpr::Const(x), NExpr::Const(y)) => x.to_bits() == y.to_bits(),
    (NExpr::Var(i), NExpr::Var(j)) => i == j,
    (NExpr::Add(x), NExpr::Add(y)) | (NExpr::Mul(x), NExpr::Mul(y)) => {
      x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
    }
    (NExpr::Pow(a1, b1), NExpr::Pow(a2, b2)) => same(a1, a2) && same(b1, b2),
    (NExpr::Neg(x), NExpr::Neg(y)) => same(x, y),
    (NExpr::Fn1(f, x), NExpr::Fn1(g, y)) => {
      std::ptr::fn_addr_eq(*f, *g) && same(x, y)
    }
    (NExpr::Fn2(f, a1, b1), NExpr::Fn2(g, a2, b2)) => {
      std::ptr::fn_addr_eq(*f, *g) && same(a1, a2) && same(b1, b2)
    }
    _ => false,
  }
}

thread_local! {
  /// Spare variable vectors for [`NExpr::Let`] evaluations, which nest.
  static LET_BUFFERS: std::cell::RefCell<Vec<Vec<f64>>> =
    const { std::cell::RefCell::new(Vec::new()) };
}

/// The value of `Let { base, table, body }` at `vars`.
pub(super) fn eval_let(
  base: usize,
  table: &[NExpr],
  body: &NExpr,
  vars: &[f64],
) -> f64 {
  let mut local = LET_BUFFERS
    .with(|b| b.borrow_mut().pop())
    .unwrap_or_default();
  local.clear();
  local.extend_from_slice(&vars[..base.min(vars.len())]);
  local.resize(base, 0.0);
  for e in table {
    let v = e.eval(&local);
    local.push(v);
  }
  let value = body.eval(&local);
  LET_BUFFERS.with(|b| b.borrow_mut().push(local));
  value
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn shared_subtrees_evaluate_once_and_agree() {
    // (x + y)^2 + Sin[x + y] * (x + y)
    let sum = || NExpr::Add(vec![NExpr::Var(0), NExpr::Var(1)]);
    let expr = NExpr::Add(vec![
      NExpr::Pow(Box::new(sum()), Box::new(NExpr::Const(2.0))),
      NExpr::Mul(vec![NExpr::Fn1(f64::sin, Box::new(sum())), sum()]),
    ]);
    let vars = [0.3, 1.1];
    let direct = expr.eval(&vars);
    let shared = share(expr, 2);
    let NExpr::Let { table, .. } = &shared else {
      panic!("nothing was shared");
    };
    assert_eq!(table.len(), 1);
    assert_eq!(shared.eval(&vars), direct);
  }
}
