//! List storage backing `Expr::List` and `Expr::FunctionCall { args }`.
//!
//! See `/Users/adrian/.claude/plans/plan-the-refactor-delegated-panda.md`.
//!
//! # Design — hybrid Vec / Tree
//!
//! `ExprList` has two internal representations:
//!
//! - `Inner::Vec(Vec<Expr>)` — the default. Same shape and cost as the
//!   pre-refactor storage. Index, slice, iter, push_back, len, etc. are
//!   all O(1) (or O(N) where they were before). Clones are O(N).
//!
//! - `Inner::Tree(Box<TreeData>)` — only used after a `push_front` is
//!   performed. `imbl::Vector` is an RRB-tree-backed persistent vector:
//!   O(1) clone (Arc-shared backbone), O(log N) `push_front` /
//!   `push_back` / `insert`. The contiguous-slice view needed by
//!   `Deref<[Expr]>` and range indexing is materialized on demand into
//!   `OnceLock<Box<[Expr]>>` — populated on first read, invalidated on
//!   every mutating call. The `Box` keeps `Inner` the same size as
//!   `Vec<Expr>`, which matters for stack frames in recursive evaluator
//!   and parser code paths that hold `Expr` values.
//!
//! # Why hybrid?
//!
//! A pure-`imbl` design forces every `Expr::List` ever constructed to
//! pay an O(N) materialization on its first read access. That is
//! catastrophic for typical Wolfram evaluation, which builds and reads
//! thousands of small `FunctionCall` argument lists per expression.
//!
//! Most lists never see a `push_front`, so they stay in `Inner::Vec`
//! and behave exactly like before — zero overhead.
//!
//! The recursive `Prepend` chain in `parseLevel` (build_summary.wls) is
//! the case the upgrade is for: the very first `push_front` upgrades to
//! `Tree`; every subsequent `push_front` is O(log N). The result list
//! is read once at the end of the chain → one O(N) materialization for
//! the whole 1462-element accumulation.
//!
//! Total: O(N log N) for the chain (was O(N²)).

use std::iter::FromIterator;
use std::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
use std::sync::OnceLock;

use crate::syntax::Expr;

#[derive(Debug)]
struct TreeData {
  data: imbl::Vector<Expr>,
  cache: OnceLock<Box<[Expr]>>,
}

#[derive(Debug)]
enum Inner {
  Vec(Vec<Expr>),
  Tree(Box<TreeData>),
}

#[derive(Debug)]
pub struct ExprList {
  inner: Inner,
}

impl Default for ExprList {
  fn default() -> Self {
    Self::new()
  }
}

impl ExprList {
  pub fn new() -> Self {
    Self {
      inner: Inner::Vec(Vec::new()),
    }
  }

  pub fn with_capacity(cap: usize) -> Self {
    Self {
      inner: Inner::Vec(Vec::with_capacity(cap)),
    }
  }

  pub fn len(&self) -> usize {
    match &self.inner {
      Inner::Vec(v) => v.len(),
      Inner::Tree(t) => t.data.len(),
    }
  }

  pub fn is_empty(&self) -> bool {
    match &self.inner {
      Inner::Vec(v) => v.is_empty(),
      Inner::Tree(t) => t.data.is_empty(),
    }
  }

  pub fn first(&self) -> Option<&Expr> {
    match &self.inner {
      Inner::Vec(v) => v.first(),
      Inner::Tree(t) => t.data.front(),
    }
  }

  pub fn last(&self) -> Option<&Expr> {
    match &self.inner {
      Inner::Vec(v) => v.last(),
      Inner::Tree(t) => t.data.back(),
    }
  }

  pub fn get(&self, i: usize) -> Option<&Expr> {
    match &self.inner {
      Inner::Vec(v) => v.get(i),
      Inner::Tree(t) => t.data.get(i),
    }
  }

  pub fn get_mut(&mut self, i: usize) -> Option<&mut Expr> {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.get_mut(i),
      Inner::Tree(t) => t.data.get_mut(i),
    }
  }

  /// Drop the Tree variant's materialization cache and (if it was used
  /// for in-place edits via DerefMut) sync any changes back into the
  /// Tree's `data`. No-op for the Vec variant.
  fn invalidate_tree_cache(&mut self) {
    if let Inner::Tree(t) = &mut self.inner
      && let Some(slice) = t.cache.take()
    {
      t.data = slice.iter().cloned().collect();
    }
  }

  /// If currently in `Vec` mode, upgrade to `Tree`. After this call the
  /// inner is guaranteed to be `Tree`. Used by `push_front`.
  fn upgrade_to_tree(&mut self) {
    if let Inner::Vec(v) = &mut self.inner {
      let v = std::mem::take(v);
      self.inner = Inner::Tree(Box::new(TreeData {
        data: v.into_iter().collect(),
        cache: OnceLock::new(),
      }));
    }
  }

  pub fn iter(&self) -> ExprListIter<'_> {
    match &self.inner {
      Inner::Vec(v) => ExprListIter::Vec(v.iter()),
      Inner::Tree(t) => ExprListIter::Tree(Box::new(t.data.iter())),
    }
  }

  /// Mutable iteration. Forces materialization for the Tree variant.
  pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Expr> {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.iter_mut(),
      Inner::Tree(t) => {
        let _ = t.cache.get_or_init(|| {
          t.data
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice()
        });
        t.cache.get_mut().unwrap().iter_mut()
      }
    }
  }

  pub fn push_back(&mut self, e: Expr) {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.push(e),
      Inner::Tree(t) => t.data.push_back(e),
    }
  }

  /// O(log N) push_front via imbl::Vector. The `Vec → Tree` upgrade
  /// happens on the first call — that's a one-time O(N) cost; every
  /// subsequent push_front is O(log N).
  pub fn push_front(&mut self, e: Expr) {
    self.invalidate_tree_cache();
    self.upgrade_to_tree();
    if let Inner::Tree(t) = &mut self.inner {
      t.data.push_front(e);
    }
  }

  pub fn pop_back(&mut self) -> Option<Expr> {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.pop(),
      Inner::Tree(t) => t.data.pop_back(),
    }
  }

  pub fn pop_front(&mut self) -> Option<Expr> {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => {
        if v.is_empty() {
          None
        } else {
          Some(v.remove(0))
        }
      }
      Inner::Tree(t) => t.data.pop_front(),
    }
  }

  pub fn push(&mut self, e: Expr) {
    self.push_back(e);
  }

  pub fn pop(&mut self) -> Option<Expr> {
    self.pop_back()
  }

  pub fn insert(&mut self, i: usize, e: Expr) {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.insert(i, e),
      Inner::Tree(t) => t.data.insert(i, e),
    }
  }

  pub fn remove(&mut self, i: usize) -> Expr {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.remove(i),
      Inner::Tree(t) => t.data.remove(i),
    }
  }

  pub fn clear(&mut self) {
    self.inner = Inner::Vec(Vec::new());
  }

  pub fn truncate(&mut self, len: usize) {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.truncate(len),
      Inner::Tree(t) => {
        while t.data.len() > len {
          t.data.pop_back();
        }
      }
    }
  }

  pub fn extend<I: IntoIterator<Item = Expr>>(&mut self, iter: I) {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.extend(iter),
      Inner::Tree(t) => {
        for e in iter {
          t.data.push_back(e);
        }
      }
    }
  }

  pub fn resize(&mut self, new_len: usize, value: Expr) {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.resize(new_len, value),
      Inner::Tree(t) => {
        let cur = t.data.len();
        if new_len > cur {
          for _ in cur..new_len - 1 {
            t.data.push_back(value.clone());
          }
          t.data.push_back(value);
        } else {
          while t.data.len() > new_len {
            t.data.pop_back();
          }
        }
      }
    }
  }

  pub fn reverse(&mut self) {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.reverse(),
      Inner::Tree(t) => {
        let reversed: imbl::Vector<Expr> =
          t.data.iter().rev().cloned().collect();
        t.data = reversed;
      }
    }
  }

  pub fn split_off(&mut self, at: usize) -> ExprList {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => Self {
        inner: Inner::Vec(v.split_off(at)),
      },
      Inner::Tree(t) => {
        let tail = t.data.split_off(at);
        Self {
          inner: Inner::Tree(Box::new(TreeData {
            data: tail,
            cache: OnceLock::new(),
          })),
        }
      }
    }
  }

  pub fn sort_by<F>(&mut self, compare: F)
  where
    F: FnMut(&Expr, &Expr) -> std::cmp::Ordering,
  {
    self.invalidate_tree_cache();
    let mut v = self.to_vec();
    v.sort_by(compare);
    self.inner = Inner::Vec(v);
  }

  pub fn sort_by_key<K: Ord, F: FnMut(&Expr) -> K>(&mut self, f: F) {
    self.invalidate_tree_cache();
    let mut v = self.to_vec();
    v.sort_by_key(f);
    self.inner = Inner::Vec(v);
  }

  pub fn dedup_by<F>(&mut self, same_bucket: F)
  where
    F: FnMut(&mut Expr, &mut Expr) -> bool,
  {
    self.invalidate_tree_cache();
    let mut v = self.to_vec();
    v.dedup_by(same_bucket);
    self.inner = Inner::Vec(v);
  }

  pub fn retain<F: FnMut(&Expr) -> bool>(&mut self, f: F) {
    self.invalidate_tree_cache();
    let mut v = self.to_vec();
    v.retain(f);
    self.inner = Inner::Vec(v);
  }

  pub fn drain<R: RangeBounds<usize>>(
    &mut self,
    range: R,
  ) -> std::vec::IntoIter<Expr> {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.drain(range).collect::<Vec<_>>().into_iter(),
      Inner::Tree(t) => {
        let len = t.data.len();
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
        let tail = t.data.split_off(end);
        let middle = t.data.split_off(start);
        t.data.append(tail);
        let drained: Vec<Expr> = middle.into_iter().collect();
        drained.into_iter()
      }
    }
  }

  /// Slice — returns a fresh `ExprList`. O(N) for Vec, O(log N) for Tree.
  pub fn slice(&self, range: impl RangeBounds<usize>) -> ExprList {
    let len = self.len();
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
      Self::new()
    } else {
      match &self.inner {
        Inner::Vec(v) => Self {
          inner: Inner::Vec(v[start..end].to_vec()),
        },
        Inner::Tree(t) => {
          let mut copy = t.data.clone();
          let _ = copy.slice(start..end);
          Self {
            inner: Inner::Tree(Box::new(TreeData {
              data: copy,
              cache: OnceLock::new(),
            })),
          }
        }
      }
    }
  }

  /// Materialize as a `Vec<Expr>` — O(N) copy.
  pub fn to_vec(&self) -> Vec<Expr> {
    match &self.inner {
      Inner::Vec(v) => v.clone(),
      Inner::Tree(t) => t.data.iter().cloned().collect(),
    }
  }

  pub fn into_inner(self) -> Vec<Expr> {
    match self.inner {
      Inner::Vec(v) => v,
      Inner::Tree(t) => t.data.into_iter().collect(),
    }
  }

  /// Force materialization for the Tree variant; return `&[Expr]`.
  fn as_slice_internal(&self) -> &[Expr] {
    match &self.inner {
      Inner::Vec(v) => v,
      Inner::Tree(t) => t.cache.get_or_init(|| {
        t.data
          .iter()
          .cloned()
          .collect::<Vec<_>>()
          .into_boxed_slice()
      }),
    }
  }

  pub fn as_slice(&self) -> &[Expr] {
    self.as_slice_internal()
  }
}

// --- Iterator wrapper ---

/// Iteration adapter that handles both internal representations.
///
/// `imbl::vector::Iter` is parameterised by an extra
/// `P: SharedPointerKind` we don't want to expose, so we box-erase it
/// (only on the Tree branch — Vec iteration stays inline).
pub enum ExprListIter<'a> {
  Vec(std::slice::Iter<'a, Expr>),
  Tree(Box<dyn DoubleEndedIterator<Item = &'a Expr> + 'a>),
}

impl<'a> Iterator for ExprListIter<'a> {
  type Item = &'a Expr;
  fn next(&mut self) -> Option<&'a Expr> {
    match self {
      ExprListIter::Vec(it) => it.next(),
      ExprListIter::Tree(it) => it.next(),
    }
  }
  fn size_hint(&self) -> (usize, Option<usize>) {
    match self {
      ExprListIter::Vec(it) => it.size_hint(),
      ExprListIter::Tree(it) => it.size_hint(),
    }
  }
}

impl<'a> DoubleEndedIterator for ExprListIter<'a> {
  fn next_back(&mut self) -> Option<&'a Expr> {
    match self {
      ExprListIter::Vec(it) => it.next_back(),
      ExprListIter::Tree(it) => it.next_back(),
    }
  }
}

// --- Trait impls ---

impl Clone for ExprList {
  fn clone(&self) -> Self {
    match &self.inner {
      Inner::Vec(v) => Self {
        inner: Inner::Vec(v.clone()),
      },
      Inner::Tree(t) => Self {
        // O(1) Arc::clone of the imbl backbone; the cache is reset.
        inner: Inner::Tree(Box::new(TreeData {
          data: t.data.clone(),
          cache: OnceLock::new(),
        })),
      },
    }
  }
}

impl FromIterator<Expr> for ExprList {
  fn from_iter<I: IntoIterator<Item = Expr>>(iter: I) -> Self {
    Self {
      inner: Inner::Vec(Vec::from_iter(iter)),
    }
  }
}

impl Extend<Expr> for ExprList {
  fn extend<I: IntoIterator<Item = Expr>>(&mut self, iter: I) {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => v.extend(iter),
      Inner::Tree(t) => {
        for e in iter {
          t.data.push_back(e);
        }
      }
    }
  }
}

impl IntoIterator for ExprList {
  type Item = Expr;
  type IntoIter = std::vec::IntoIter<Expr>;
  fn into_iter(self) -> Self::IntoIter {
    self.into_inner().into_iter()
  }
}

impl<'a> IntoIterator for &'a ExprList {
  type Item = &'a Expr;
  type IntoIter = ExprListIter<'a>;
  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<'a> IntoIterator for &'a mut ExprList {
  type Item = &'a mut Expr;
  type IntoIter = std::slice::IterMut<'a, Expr>;
  fn into_iter(self) -> Self::IntoIter {
    self.iter_mut()
  }
}

impl Index<usize> for ExprList {
  type Output = Expr;
  fn index(&self, i: usize) -> &Expr {
    match &self.inner {
      Inner::Vec(v) => &v[i],
      Inner::Tree(t) => &t.data[i],
    }
  }
}

impl IndexMut<usize> for ExprList {
  fn index_mut(&mut self, i: usize) -> &mut Expr {
    self.invalidate_tree_cache();
    match &mut self.inner {
      Inner::Vec(v) => &mut v[i],
      Inner::Tree(t) => &mut t.data[i],
    }
  }
}

// Range indexing — go through `as_slice_internal` (free for Vec, cached
// for Tree).
macro_rules! range_index_impl {
  ($T:ty) => {
    impl Index<$T> for ExprList {
      type Output = [Expr];
      fn index(&self, r: $T) -> &[Expr] {
        &self.as_slice_internal()[r]
      }
    }
  };
}
range_index_impl!(std::ops::Range<usize>);
range_index_impl!(std::ops::RangeFrom<usize>);
range_index_impl!(std::ops::RangeTo<usize>);
range_index_impl!(std::ops::RangeFull);
range_index_impl!(std::ops::RangeInclusive<usize>);
range_index_impl!(std::ops::RangeToInclusive<usize>);

impl Deref for ExprList {
  type Target = [Expr];
  fn deref(&self) -> &[Expr] {
    self.as_slice_internal()
  }
}

impl DerefMut for ExprList {
  fn deref_mut(&mut self) -> &mut [Expr] {
    match &mut self.inner {
      Inner::Vec(v) => v,
      Inner::Tree(t) => {
        let _ = t.cache.get_or_init(|| {
          t.data
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice()
        });
        t.cache.get_mut().unwrap()
      }
    }
  }
}

impl From<Vec<Expr>> for ExprList {
  fn from(v: Vec<Expr>) -> Self {
    Self {
      inner: Inner::Vec(v),
    }
  }
}

impl From<ExprList> for Vec<Expr> {
  fn from(l: ExprList) -> Self {
    l.into_inner()
  }
}

impl From<&ExprList> for Vec<Expr> {
  fn from(l: &ExprList) -> Self {
    l.to_vec()
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
