//! Acceptance harness for the self-contained linear `Reduce` campaign.
//!
//! The ordinary test suite must remain independent of external solvers. The
//! ignored `oracle` module is selected only by `make test-reduce-oracle`.

use woxi::interpret;

mod reduce;

fn assert_reduces(input: &str, expected: &str) {
  assert_eq!(interpret(input).unwrap(), expected, "input: {input}");
}
