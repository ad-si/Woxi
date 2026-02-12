use super::*;

mod cellular_automaton {
  use super::*;

  #[test]
  fn rule_30_expanding_small() {
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 3]").unwrap(),
      "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0, 0}, {0, 1, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 1, 1}}"
    );
  }

  #[test]
  fn rule_90_expanding() {
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, 3]").unwrap(),
      "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, 0, 1}}"
    );
  }

  #[test]
  fn rule_110_trims_background_columns() {
    assert_eq!(
      interpret("CellularAutomaton[110, {{1}, 0}, 3]").unwrap(),
      "{{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 1, 1}, {1, 1, 0, 1}}"
    );
  }

  #[test]
  fn rule_2_trims_right() {
    assert_eq!(
      interpret("CellularAutomaton[2, {{1}, 0}, 3]").unwrap(),
      "{{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}}"
    );
  }

  #[test]
  fn periodic_boundary() {
    assert_eq!(
      interpret("CellularAutomaton[30, {1, 0, 0, 0, 1}, 2]").unwrap(),
      "{{1, 0, 0, 0, 1}, {0, 1, 0, 1, 1}, {0, 1, 0, 1, 0}}"
    );
  }

  #[test]
  fn periodic_rule_30() {
    assert_eq!(
      interpret("CellularAutomaton[30, {1, 0, 0, 0, 1}, 3]").unwrap(),
      "{{1, 0, 0, 0, 1}, {0, 1, 0, 1, 1}, {0, 1, 0, 1, 0}, {1, 1, 0, 1, 1}}"
    );
  }

  #[test]
  fn zero_steps() {
    assert_eq!(
      interpret("CellularAutomaton[30, {1, 0, 0, 0, 1}, 0]").unwrap(),
      "{{1, 0, 0, 0, 1}}"
    );
  }

  #[test]
  fn zero_steps_expanding() {
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 0]").unwrap(),
      "{{1}}"
    );
  }

  #[test]
  fn rule_254_expanding() {
    assert_eq!(
      interpret("CellularAutomaton[254, {{1}, 0}, 3]").unwrap(),
      "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1, 1}}"
    );
  }

  #[test]
  fn symbolic_return_for_non_integer_rule() {
    assert_eq!(
      interpret("CellularAutomaton[x, {{1}, 0}, 3]").unwrap(),
      "CellularAutomaton[x, {{1}, 0}, 3]"
    );
  }
}
