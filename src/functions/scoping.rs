use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique symbol names
static SYMBOL_COUNTER: AtomicU64 = AtomicU64::new(1);

/// The next name `make(n)` that no symbol in use already has: a fresh
/// symbol never aliases one the session created (`$3 = 3; Unique[]` is not
/// `$3`), such as the objects a notebook's saved definitions restore.
fn fresh_name(make: impl Fn(u64) -> String) -> String {
  loop {
    let name = make(SYMBOL_COUNTER.fetch_add(1, Ordering::SeqCst));
    if !crate::evaluator::contexts::global_symbol_in_use(&name) {
      return name;
    }
  }
}

/// Generate a unique symbol name like "x$1", "x$2", etc.
pub fn unique_symbol(base_name: &str) -> String {
  fresh_name(|n| format!("{base_name}${n}"))
}

/// Generate a unique symbol from a string prefix like "xxx1", "xxx2", etc.
/// (no $ separator, used by Unique["xxx"])
pub fn unique_symbol_from_string(prefix: &str) -> String {
  fresh_name(|n| format!("{prefix}{n}"))
}
