//! Acceptance tests for the exact linear `Reduce` engine (`woxi-reduce`).

use woxi::interpret;

mod reduce;

fn assert_reduces(input: &str, expected: &str) {
  assert_eq!(interpret(input).unwrap(), expected, "input: {input}");
}
