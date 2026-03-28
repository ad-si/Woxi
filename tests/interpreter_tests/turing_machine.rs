use super::*;

mod turing_machine {
  use super::*;

  #[test]
  fn basic_rule_2506() {
    assert_eq!(
      interpret("TuringMachine[2506, {{1}, {0, 0, 1, 0, 0}}, 3]").unwrap(),
      "{{{1, 1, 0}, {0, 0, 1, 0, 0}}, {{2, 2, 1}, {1, 0, 1, 0, 0}}, {{1, 1, 0}, {1, 1, 1, 0, 0}}, {{2, 5, -1}, {0, 1, 1, 0, 0}}}"
    );
  }

  #[test]
  fn rule_2506_small_tape() {
    assert_eq!(
      interpret("TuringMachine[2506, {{1}, {0, 1, 0}}, 2]").unwrap(),
      "{{{1, 1, 0}, {0, 1, 0}}, {{2, 2, 1}, {1, 1, 0}}, {{1, 3, 2}, {1, 0, 0}}}"
    );
  }

  #[test]
  fn zero_steps() {
    assert_eq!(
      interpret("TuringMachine[2506, {{1}, {0, 1, 0}}, 0]").unwrap(),
      "{{{1, 1, 0}, {0, 1, 0}}}"
    );
  }

  #[test]
  fn explicit_head_position() {
    assert_eq!(
      interpret("TuringMachine[2506, {{1, 3}, {0, 0, 1, 0, 0}}, 2]").unwrap(),
      "{{{1, 3, 0}, {0, 0, 1, 0, 0}}, {{2, 2, -1}, {0, 0, 0, 0, 0}}, {{1, 1, -2}, {0, 1, 0, 0, 0}}}"
    );
  }

  #[test]
  fn explicit_rule_spec_nsk() {
    assert_eq!(
      interpret("TuringMachine[{2506, 2, 2}, {{1}, {0, 0, 1, 0, 0}}, 3]")
        .unwrap(),
      "{{{1, 1, 0}, {0, 0, 1, 0, 0}}, {{2, 2, 1}, {1, 0, 1, 0, 0}}, {{1, 1, 0}, {1, 1, 1, 0, 0}}, {{2, 5, -1}, {0, 1, 1, 0, 0}}}"
    );
  }

  #[test]
  fn one_step() {
    assert_eq!(
      interpret("TuringMachine[2506, {{1}, {0, 0, 1, 0, 0}}, 1]").unwrap(),
      "{{{1, 1, 0}, {0, 0, 1, 0, 0}}, {{2, 2, 1}, {1, 0, 1, 0, 0}}}"
    );
  }

  #[test]
  fn periodic_boundary_wrapping() {
    assert_eq!(
      interpret("TuringMachine[2506, {{1}, {0, 0, 0}}, 5]").unwrap(),
      "{{{1, 1, 0}, {0, 0, 0}}, {{2, 2, 1}, {1, 0, 0}}, {{1, 1, 0}, {1, 1, 0}}, {{2, 3, -1}, {0, 1, 0}}, {{1, 2, -2}, {0, 1, 1}}, {{2, 1, -3}, {0, 0, 1}}}"
    );
  }
}
