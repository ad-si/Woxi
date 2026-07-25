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

// The documented initial-condition and one-step forms. All values verified
// against wolframscript.
mod init_and_step_forms {
  use super::*;

  // The state may be named directly, not just as a one-element list.
  #[test]
  fn a_bare_state_starts_the_head_at_one() {
    assert_eq!(
      interpret("TuringMachine[2506, {1, {0, 0, 0, 0}}, 3]").unwrap(),
      "{{{1, 1, 0}, {0, 0, 0, 0}}, {{2, 2, 1}, {1, 0, 0, 0}}, \
       {{1, 1, 0}, {1, 1, 0, 0}}, {{2, 4, -1}, {0, 1, 0, 0}}}"
    );
    // `{state, position}` still works.
    assert_eq!(
      interpret("TuringMachine[2506, {{1, 2}, {0, 0, 0, 0}}, 2]").unwrap(),
      "{{{1, 2, 0}, {0, 0, 0, 0}}, {{2, 3, 1}, {0, 1, 0, 0}}, \
       {{1, 2, 0}, {0, 1, 1, 0}}}"
    );
  }

  // `{cells, background}` is an infinite tape: the head never wraps, and the
  // reported tape is exactly the region the run touched.
  #[test]
  fn an_infinite_tape_reports_the_visited_window() {
    assert_eq!(
      interpret("TuringMachine[2506, {1, {{}, 0}}, 0]").unwrap(),
      "{{{1, 1, 0}, {0}}}"
    );
    assert_eq!(
      interpret("TuringMachine[2506, {1, {{}, 0}}, 1]").unwrap(),
      "{{{1, 1, 0}, {0, 0}}, {{2, 2, 1}, {1, 0}}}"
    );
    // Moving left extends the window leftwards, shifting every position.
    assert_eq!(
      interpret("TuringMachine[2506, {1, {{}, 0}}, 3]").unwrap(),
      "{{{1, 2, 0}, {0, 0, 0}}, {{2, 3, 1}, {0, 1, 0}}, \
       {{1, 2, 0}, {0, 1, 1}}, {{2, 1, -1}, {0, 0, 1}}}"
    );
    assert_eq!(
      interpret("TuringMachine[2506, {1, {{}, 0}}, 4]").unwrap(),
      "{{{1, 3, 0}, {0, 0, 0, 0}}, {{2, 4, 1}, {0, 0, 1, 0}}, \
       {{1, 3, 0}, {0, 0, 1, 1}}, {{2, 2, -1}, {0, 0, 0, 1}}, \
       {{1, 1, -2}, {0, 1, 0, 1}}}"
    );
  }

  // Initial cells sit at the head and to its right, so they stay in the
  // window even when the head only ever moves left.
  #[test]
  fn an_infinite_tape_may_carry_initial_cells() {
    assert_eq!(
      interpret("TuringMachine[2506, {1, {{1, 0, 1}, 0}}, 3]").unwrap(),
      "{{{1, 3, 0}, {0, 0, 1, 0, 1}}, {{2, 2, -1}, {0, 0, 0, 0, 1}}, \
       {{1, 1, -2}, {0, 1, 0, 0, 1}}, {{2, 2, -1}, {1, 1, 0, 0, 1}}}"
    );
  }

  // Without a step count only one step runs, and just the new state comes
  // back — keeping the `{cells, background}` tape form when it had one.
  #[test]
  fn the_one_step_forms() {
    assert_eq!(
      interpret("TuringMachine[2506, {1, {0, 0, 0, 0}}]").unwrap(),
      "{{2, 2, 1}, {1, 0, 0, 0}}"
    );
    assert_eq!(
      interpret("TuringMachine[2506, {1, {{}, 0}}]").unwrap(),
      "{{2, 2, 1}, {{1, 0}, 0}}"
    );
    // The operator form does the same.
    assert_eq!(
      interpret("TuringMachine[2506][{1, {0, 0, 0, 0}}]").unwrap(),
      "{{2, 2, 1}, {1, 0, 0, 0}}"
    );
    assert_eq!(
      interpret("TuringMachine[2506][{1, {{}, 0}}]").unwrap(),
      "{{2, 2, 1}, {{1, 0}, 0}}"
    );
  }

  #[test]
  fn a_non_integer_time_reports_tspec() {
    assert_eq!(
      interpret("TuringMachine[2506, {1, {{}, 0}}, {3}]").unwrap(),
      "TuringMachine[2506, {1, {{}, 0}}, {3}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "TuringMachine::tspec: The time specification {3} must be an integer >= 0."
      )),
      "expected tspec message, got {msgs:?}"
    );
  }
}
