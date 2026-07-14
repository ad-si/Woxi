use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique symbol names
static SYMBOL_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique symbol name like "x$1", "x$2", etc.
pub fn unique_symbol(base_name: &str) -> String {
  let n = SYMBOL_COUNTER.fetch_add(1, Ordering::SeqCst);
  format!("{}${}", base_name, n)
}

/// Generate a unique symbol from a string prefix like "xxx1", "xxx2", etc.
/// (no $ separator, used by Unique["xxx"])
pub fn unique_symbol_from_string(prefix: &str) -> String {
  let n = SYMBOL_COUNTER.fetch_add(1, Ordering::SeqCst);
  format!("{}{}", prefix, n)
}
