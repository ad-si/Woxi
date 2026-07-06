//! Fuzz target for the pest parser.
//!
//! Invariant: `woxi::parse` must return `Ok` or `Err` for every input —
//! it must never panic, abort, or overflow the stack.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  // Bound the input so the fuzzer explores structure instead of drowning
  // in megabyte-sized inputs (deep-nesting stack overflows still reproduce
  // well below this limit).
  if data.len() > 4096 {
    return;
  }
  if let Ok(input) = std::str::from_utf8(data) {
    let _ = woxi::parse(input);
  }
});
