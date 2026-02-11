use woxi::{clear_state, interpret, interpret_with_stdout};

mod interpreter_tests {
  use super::*;

  mod algebra;
  mod arithmetic;
  mod association;
  mod calculus;
  mod control_flow;
  mod functions;
  mod io;
  mod linear_algebra;
  mod list;
  mod math;
  mod statistics;
  mod string;
  mod syntax;
}
