// Numbers and arithmetic: integer and number theory, rounding, complex
// numbers, constants and special functions.
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

  mod arithmetic;
  mod financial;
  mod interval;
  mod large_number_and_memoization;
  mod math;
  mod special_functions;
}
