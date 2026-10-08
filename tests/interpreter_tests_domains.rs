// Domain-specific functionality: statistics and distributions, graphs,
// geometry, dates, quantities and the curated entity data.
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

  mod astronomy;
  mod dataset;
  mod datetime;
  mod distributions;
  mod element_data;
  mod entity;
  mod example_data;
  mod geometry;
  mod graph_theory;
  mod isotope_data;
  mod knot_data;
  mod molecule;
  mod polyhedron_data;
  mod polyhedron_operations;
  mod property;
  mod quantity;
  mod statistics;
  mod tabular;
  mod timeseries;
  mod turing_machine;
  mod wxf;
}
