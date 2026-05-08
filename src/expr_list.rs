//! Persistent / CoW-friendly wrapper around the storage backing
//! `Expr::List` and `Expr::FunctionCall { args }`.
//!
//! See `/Users/adrian/.claude/plans/plan-the-refactor-delegated-panda.md`.
//!
//! Commit 1 keeps the inner storage as `Vec<Expr>` — the only behaviour
//! change is the type name. Commit 2 swaps the inner storage to
//! `imbl::Vector<Expr>`. Commit 3 rewrites hot paths.

use std::iter::FromIterator;
use std::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};

use crate::syntax::Expr;

#[derive(Debug, Default)]
pub struct ExprList(Vec<Expr>);

impl ExprList {
  pub fn new() -> Self {
    Self(Vec::new())
  }

  pub fn with_capacity(cap: usize) -> Self {
    Self(Vec::with_capacity(cap))
  }

  pub fn push_back(&mut self, e: Expr) {
    self.0.push(e);
  }

  pub fn push_front(&mut self, e: Expr) {
    self.0.insert(0, e);
  }

  pub fn pop_front(&mut self) -> Option<Expr> {
    if self.0.is_empty() {
      None
    } else {
      Some(self.0.remove(0))
    }
  }

  /// O(log N) slice once `imbl::Vector` is in play. Returns a fresh
  /// `ExprList`. Range bounds are saturated to `[0, len]`.
  pub fn slice(&self, range: impl RangeBounds<usize>) -> ExprList {
    let len = self.0.len();
    let start = match range.start_bound() {
      std::ops::Bound::Included(&n) => n,
      std::ops::Bound::Excluded(&n) => n + 1,
      std::ops::Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
      std::ops::Bound::Included(&n) => n + 1,
      std::ops::Bound::Excluded(&n) => n,
      std::ops::Bound::Unbounded => len,
    };
    let start = start.min(len);
    let end = end.min(len);
    if start >= end {
      ExprList::new()
    } else {
      ExprList(self.0[start..end].to_vec())
    }
  }

  /// Materialize as a `Vec<Expr>`. Cheap while inner is `Vec`; once we
  /// swap to `imbl::Vector` this becomes O(N) and grep-able.
  pub fn to_vec(&self) -> Vec<Expr> {
    self.0.clone()
  }

  pub fn into_inner(self) -> Vec<Expr> {
    self.0
  }
}

// Lets every legacy call site that expected `&Vec<Expr>` or `&[Expr]`
// keep working: `items.push(x)`, `items.iter()`, `items.iter_mut()`,
// `items.first()`, `items.last()`, `items.is_empty()`, `items.len()`,
// `for x in &items`, etc. all resolve through `Deref::deref()`.

impl Deref for ExprList {
  type Target = Vec<Expr>;
  fn deref(&self) -> &Vec<Expr> {
    &self.0
  }
}

impl DerefMut for ExprList {
  fn deref_mut(&mut self) -> &mut Vec<Expr> {
    &mut self.0
  }
}

impl Clone for ExprList {
  fn clone(&self) -> Self {
    ExprList(self.0.clone())
  }
}

// `Expr` has neither `PartialEq` nor `Eq`. Element comparison goes through
// `expr_equal` (`src/evaluator/pattern_matching.rs`) which is order-sensitive
// and recursive. Don't expose `PartialEq` on `ExprList` so callers are
// forced to use `expr_equal` for the whole expression instead.

impl FromIterator<Expr> for ExprList {
  fn from_iter<I: IntoIterator<Item = Expr>>(iter: I) -> Self {
    ExprList(Vec::from_iter(iter))
  }
}

impl Extend<Expr> for ExprList {
  fn extend<I: IntoIterator<Item = Expr>>(&mut self, iter: I) {
    self.0.extend(iter);
  }
}

impl IntoIterator for ExprList {
  type Item = Expr;
  type IntoIter = std::vec::IntoIter<Expr>;
  fn into_iter(self) -> Self::IntoIter {
    self.0.into_iter()
  }
}

impl<'a> IntoIterator for &'a ExprList {
  type Item = &'a Expr;
  type IntoIter = std::slice::Iter<'a, Expr>;
  fn into_iter(self) -> Self::IntoIter {
    self.0.iter()
  }
}

impl<'a> IntoIterator for &'a mut ExprList {
  type Item = &'a mut Expr;
  type IntoIter = std::slice::IterMut<'a, Expr>;
  fn into_iter(self) -> Self::IntoIter {
    self.0.iter_mut()
  }
}

impl Index<usize> for ExprList {
  type Output = Expr;
  fn index(&self, i: usize) -> &Expr {
    &self.0[i]
  }
}

impl IndexMut<usize> for ExprList {
  fn index_mut(&mut self, i: usize) -> &mut Expr {
    &mut self.0[i]
  }
}

// Slice-style range indexing — preserved so legacy `items[a..b]` calls
// keep returning a borrowed slice rather than going through ExprList's
// owned `slice(...)` method (which copies).
impl Index<std::ops::Range<usize>> for ExprList {
  type Output = [Expr];
  fn index(&self, r: std::ops::Range<usize>) -> &[Expr] {
    &self.0[r]
  }
}

impl Index<std::ops::RangeFrom<usize>> for ExprList {
  type Output = [Expr];
  fn index(&self, r: std::ops::RangeFrom<usize>) -> &[Expr] {
    &self.0[r]
  }
}

impl Index<std::ops::RangeTo<usize>> for ExprList {
  type Output = [Expr];
  fn index(&self, r: std::ops::RangeTo<usize>) -> &[Expr] {
    &self.0[r]
  }
}

impl Index<std::ops::RangeFull> for ExprList {
  type Output = [Expr];
  fn index(&self, r: std::ops::RangeFull) -> &[Expr] {
    &self.0[r]
  }
}

impl Index<std::ops::RangeInclusive<usize>> for ExprList {
  type Output = [Expr];
  fn index(&self, r: std::ops::RangeInclusive<usize>) -> &[Expr] {
    &self.0[r]
  }
}

impl From<Vec<Expr>> for ExprList {
  fn from(v: Vec<Expr>) -> Self {
    ExprList(v)
  }
}

impl From<ExprList> for Vec<Expr> {
  fn from(l: ExprList) -> Self {
    l.0
  }
}

impl From<&ExprList> for Vec<Expr> {
  fn from(l: &ExprList) -> Self {
    l.0.clone()
  }
}

/// Shorthand constructor matching the call shape of `vec![…]`.
#[macro_export]
macro_rules! expr_list {
  () => { $crate::expr_list::ExprList::new() };
  ($($x:expr),+ $(,)?) => {
    $crate::expr_list::ExprList::from(vec![$($x),+])
  };
  ($x:expr; $n:expr) => {
    $crate::expr_list::ExprList::from(vec![$x; $n])
  };
}
