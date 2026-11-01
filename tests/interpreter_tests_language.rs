// Core language semantics: syntax, patterns, definitions, evaluation
// control, attributes, contexts and argument checking.
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

  mod arg_count;
  mod argument_forms;
  mod argument_forms_sweep0;
  mod argument_forms_sweep1;
  mod argument_forms_sweep2;
  mod argument_forms_sweep3;
  mod assessment;
  mod attributes;
  mod batch_wrappers;
  mod code_parser;
  mod contexts;
  mod control_flow;
  mod evaluation_reuse;
  mod function_application;
  mod function_definitions;
  mod interpret_to_expr_api;
  mod machine_specific;
  mod option_sweep0;
  mod option_sweep1;
  mod option_sweep2;
  mod option_sweep3;
  mod option_sweep3_fitting;
  mod option_sweep3_lists;
  mod option_sweep3_numbers;
  mod option_sweep3_special;
  mod option_sweep3_strings;
  mod option_sweep4_biosequence;
  mod option_sweep4_interval;
  mod option_sweep4_misc;
  mod option_sweep4_options;
  mod patterns;
  mod rosetta_script_fixes;
  mod sockets;
  mod syntax;
}
