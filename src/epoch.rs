//! The definitions epoch: a counter that changes whenever anything happens
//! that could make evaluating the same expression give a different result
//! (or a different side effect) than before.
//!
//! Evaluation results can be reused while the epoch stands still. A large
//! stored value that once evaluated to itself (`r = Expand[(a + b)^20]`) is
//! marked with the epoch at which that was verified, and reading it again
//! at the same epoch returns it as is instead of rebuilding every term —
//! the role Wolfram's evaluation timestamps play.
//!
//! The global stores evaluation reads (`ENV`, `FUNC_DEFS`, attributes,
//! options, contexts, …) are [`EpochCell`]s, whose `borrow_mut` advances
//! the epoch, so no mutation site can forget to. Observable side effects
//! (messages, printed output) advance it too: an expression whose
//! evaluation produced one must be evaluated again to produce it again.

use std::cell::{BorrowError, BorrowMutError, Cell, Ref, RefCell, RefMut};
use std::sync::atomic::{AtomicU64, Ordering};

/// Source of epoch values. Each thread draws from it, so an epoch value
/// names one stretch of one thread's state and a mark set on one thread
/// never matches on another.
static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

fn fresh_epoch() -> u64 {
  NEXT_EPOCH.fetch_add(1, Ordering::Relaxed)
}

thread_local! {
  static EPOCH: Cell<u64> = Cell::new(fresh_epoch());
  static PROTECTED_EPOCH: Cell<u64> = Cell::new(fresh_epoch());
}

/// The current epoch. Never 0, which marks a list as never verified.
pub fn current_epoch() -> u64 {
  EPOCH.with(Cell::get)
}

/// The epoch of everything but the own values of unprotected symbols. An
/// expression that mentions only protected symbols (numbers and built-in
/// functions, say) evaluates the same however those values change, so a
/// fixed point verified at this epoch stays one across the assignments of
/// a loop around it.
pub fn current_protected_epoch() -> u64 {
  PROTECTED_EPOCH.with(Cell::get)
}

/// Record that evaluation-relevant state changed (or a side effect
/// happened), invalidating every result verified so far.
pub fn bump_epoch() {
  EPOCH.with(|e| e.set(fresh_epoch()));
  PROTECTED_EPOCH.with(|e| e.set(fresh_epoch()));
}

/// Record that the own value of the symbol `name` changed: only results
/// that could mention it are invalidated when it is unprotected.
pub fn bump_epoch_for_own_value(name: &str) {
  record_own_value_assignment(name);
  if crate::evaluator::pattern_matching::is_symbol_protected(name) {
    bump_epoch();
  } else {
    EPOCH.with(|e| e.set(fresh_epoch()));
  }
}

/// How many own-value assignments the log of recent ones keeps.
const OWN_VALUE_LOG_LEN: usize = 32;

thread_local! {
  /// Counts own-value assignments, so an evaluation can tell whether one
  /// happened while it ran.
  static OWN_VALUE_SEQ: Cell<u64> = const { Cell::new(0) };
  /// The most recent own-value assignments, as (sequence number, name hash).
  static OWN_VALUE_LOG: RefCell<std::collections::VecDeque<(u64, u64)>> =
    const { RefCell::new(std::collections::VecDeque::new()) };
}

fn symbol_name_hash(name: &str) -> u64 {
  use std::hash::{Hash, Hasher};
  let mut hasher = std::collections::hash_map::DefaultHasher::new();
  name.hash(&mut hasher);
  hasher.finish()
}

fn record_own_value_assignment(name: &str) {
  let seq = OWN_VALUE_SEQ.with(|s| {
    let next = s.get() + 1;
    s.set(next);
    next
  });
  let hash = symbol_name_hash(name);
  OWN_VALUE_LOG.with(|log| {
    let mut log = log.borrow_mut();
    if log.len() == OWN_VALUE_LOG_LEN {
      log.pop_front();
    }
    log.push_back((seq, hash));
  });
}

/// The sequence number of the latest own-value assignment, to pass to
/// [`own_values_assigned_since`] later.
pub fn own_value_seq() -> u64 {
  OWN_VALUE_SEQ.with(Cell::get)
}

/// Which symbols got an own value assigned since `seq` was read.
pub enum AssignedSince {
  /// No assignment happened.
  Nothing,
  /// The symbols whose name hashes these are (see [`assigned_symbol`]).
  Symbols(Vec<u64>),
  /// More assignments than the log keeps: any symbol may have changed.
  Any,
}

pub fn own_values_assigned_since(seq: u64) -> AssignedSince {
  if own_value_seq() == seq {
    return AssignedSince::Nothing;
  }
  OWN_VALUE_LOG.with(|log| {
    let log = log.borrow();
    match log.front() {
      Some(&(oldest, _)) if oldest <= seq + 1 => AssignedSince::Symbols(
        log
          .iter()
          .filter(|(s, _)| *s > seq)
          .map(|(_, h)| *h)
          .collect(),
      ),
      _ => AssignedSince::Any,
    }
  })
}

/// Whether the symbol `name` is among the `assigned` ones.
pub fn assigned_symbol(assigned: &AssignedSince, name: &str) -> bool {
  match assigned {
    AssignedSince::Nothing => false,
    AssignedSince::Symbols(hashes) => hashes.contains(&symbol_name_hash(name)),
    AssignedSince::Any => true,
  }
}

/// A `RefCell` for state that evaluation reads. Every mutable borrow
/// advances the epoch; reads leave it alone.
pub struct EpochCell<T>(RefCell<T>);

impl<T> EpochCell<T> {
  pub const fn new(value: T) -> Self {
    Self(RefCell::new(value))
  }

  pub fn borrow(&self) -> Ref<'_, T> {
    self.0.borrow()
  }

  pub fn try_borrow(&self) -> Result<Ref<'_, T>, BorrowError> {
    self.0.try_borrow()
  }

  pub fn borrow_mut(&self) -> RefMut<'_, T> {
    bump_epoch();
    self.0.borrow_mut()
  }

  pub fn try_borrow_mut(&self) -> Result<RefMut<'_, T>, BorrowMutError> {
    bump_epoch();
    self.0.try_borrow_mut()
  }

  /// A mutable borrow for a change to the own value of the symbol `name`
  /// alone (see [`bump_epoch_for_own_value`]).
  pub fn borrow_mut_own_value(&self, name: &str) -> RefMut<'_, T> {
    bump_epoch_for_own_value(name);
    self.0.borrow_mut()
  }

  /// A mutable borrow that does not advance the epoch, for changes that
  /// leave the meaning of the state as it was (switching a stored value to
  /// a representation that is cheaper to copy).
  pub fn borrow_mut_unversioned(&self) -> RefMut<'_, T> {
    self.0.borrow_mut()
  }

  pub fn try_borrow_mut_unversioned(
    &self,
  ) -> Result<RefMut<'_, T>, BorrowMutError> {
    self.0.try_borrow_mut()
  }

  pub fn replace(&self, value: T) -> T {
    bump_epoch();
    self.0.replace(value)
  }

  pub fn take(&self) -> T
  where
    T: Default,
  {
    bump_epoch();
    self.0.take()
  }
}

impl<T: Default> Default for EpochCell<T> {
  fn default() -> Self {
    Self::new(T::default())
  }
}

thread_local! {
  static SIDE_EFFECTS: Cell<u64> = const { Cell::new(0) };
}

/// Record an observable side effect (a message, printed output). It does
/// not invalidate earlier results, but an evaluation during which one
/// happened is not marked as reusable: reading it again has to repeat it.
pub fn note_side_effect() {
  SIDE_EFFECTS.with(|c| c.set(c.get() + 1));
}

/// How many side effects have happened on this thread so far.
pub fn side_effect_count() -> u64 {
  SIDE_EFFECTS.with(Cell::get)
}
