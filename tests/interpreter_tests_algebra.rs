// Symbolic algebra, calculus and linear algebra.
//
// One of the `interpreter_tests*` binaries; see `tests/interpreter_tests.rs`
// for why the suite is split. The modules live in `tests/interpreter_tests/`
// and keep their `interpreter_tests::<file>::…` test paths.
#![allow(clippy::module_inception)]

use woxi::{clear_state, interpret, interpret_with_stdout};

mod interpreter_tests {
  use super::*;

  #[allow(dead_code)]
  mod case_helpers;
  mod path_helpers;
  #[allow(unused_imports)]
  use path_helpers::*;

  mod algebra;
  mod calculus;
  mod linear_algebra;
  mod wavelets;
}
