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

  #[test]
  fn step_spec_list_t_gives_all_steps() {
    // {t} is identical to the bare `t` form: all steps 0 through t.
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, {3}]").unwrap(),
      "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0, 0}, \
       {0, 1, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 1, 1}}"
    );
  }

  #[test]
  fn step_spec_single_state_wrapped() {
    // {{t}} returns a list holding the step-t state.
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, {{3}}]").unwrap(),
      "{{1, 1, 0, 1, 1, 1, 1}}"
    );
  }

  #[test]
  fn step_spec_range() {
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, {{1, 3}}]").unwrap(),
      "{{0, 0, 1, 1, 1, 0, 0}, {0, 1, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 1, 1}}"
    );
  }

  #[test]
  fn step_spec_range_with_increment() {
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{0, 4, 2}}]").unwrap(),
      "{{0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 1, 0, 0}, \
       {1, 0, 0, 0, 0, 0, 0, 0, 1}}"
    );
  }

  #[test]
  fn general_rule_spec_matches_elementary() {
    // {n, k} and {n, k, r} with k = 2, r = 1 are the elementary rules.
    assert_eq!(
      interpret(
        "CellularAutomaton[{30, 2}, {{1}, 0}, 3] === \
         CellularAutomaton[30, {{1}, 0}, 3]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "CellularAutomaton[{30, 2, 1}, {{1}, 0}, 3] === \
         CellularAutomaton[30, {{1}, 0}, 3]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn weighted_rule_spec_matches_elementary() {
    // The documented equivalence: elementary n == {n, {2, {4, 2, 1}}, 1}.
    assert_eq!(
      interpret(
        "CellularAutomaton[{110, {2, {4, 2, 1}}, 1}, {{1}, 0}, 3] === \
         CellularAutomaton[110, {{1}, 0}, 3]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn two_dimensional_growth_rule() {
    // Outer totalistic 5-neighbor code 942: a dead cell with exactly one
    // (or all four) live orthogonal neighbors is born, live cells survive.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, 1]"
      )
      .unwrap(),
      "{{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}}"
    );
  }

  #[test]
  fn two_dimensional_growth_rule_steps() {
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {2}]"
      )
      .unwrap(),
      "{{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}, \
       {0, 0, 0, 0, 0}}, {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, \
       {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}}, {{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, \
       {1, 1, 1, 1, 1}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}}}"
    );
  }

  #[test]
  fn two_dimensional_range_step_spec_dimensions() {
    // The documentation example: states at steps 10, 20, 30 share the
    // jointly trimmed 61x61 extent of the step-30 pattern.
    assert_eq!(
      interpret(
        "Dimensions /@ CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, \
         {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}, {{10, 30, 10}}]"
      )
      .unwrap(),
      "{{61, 61}, {61, 61}, {61, 61}}"
    );
  }

  #[test]
  fn game_of_life_blinker() {
    // Conway's Game of Life as the weighted 9-neighbor rule 224: a
    // horizontal blinker flips to a vertical one.
    assert_eq!(
      interpret(
        "CellularAutomaton[{224, {2, {{2, 2, 2}, {2, 1, 2}, {2, 2, 2}}}, \
         {1, 1}}, {{{1, 1, 1}}, 0}, 1]"
      )
      .unwrap(),
      "{{{0, 0, 0}, {1, 1, 1}, {0, 0, 0}}, {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}}}"
    );
  }

  #[test]
  fn two_dimensional_cyclic_init() {
    // A bare matrix init evolves on a fixed grid with cyclic boundaries.
    assert_eq!(
      interpret(
        "CellularAutomaton[{224, {2, {{2, 2, 2}, {2, 1, 2}, {2, 2, 2}}}, \
         {1, 1}}, {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, \
         {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}, {1}]"
      )
      .unwrap(),
      "{{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}, \
       {0, 0, 0, 0, 0}}, {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, \
       {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}}}"
    );
  }

  #[test]
  fn invalid_two_dimensional_rule_stays_symbolic() {
    // An even-sized weight matrix is not a valid neighborhood.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2}, {2, 1}}}, {1, 1}}, {{{1}}, 0}, 1]"
      )
      .unwrap(),
      "CellularAutomaton[{942, {2, {{0, 2}, {2, 1}}}, {1, 1}}, {{{1}}, 0}, 1]"
    );
  }

  #[test]
  fn arrayplot_over_two_dimensional_states() {
    // The documentation example renders one ArrayPlot per returned state.
    assert_eq!(
      interpret(
        "ArrayPlot /@ CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, \
         {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}, {{10, 30, 10}}]"
      )
      .unwrap(),
      "{-Graphics-, -Graphics-, -Graphics-}"
    );
  }

  #[test]
  fn arrayplot_over_two_dimensional_states_renders_combined_svg() {
    // In visual hosts (playground, studio) the list of ArrayPlots is
    // combined into one SVG with a nested SVG per state.
    clear_state();
    let result = interpret_with_stdout(
      "ArrayPlot /@ CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, \
       {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}, {{10, 30, 10}}]",
    )
    .unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.expect("list of ArrayPlots should render");
    assert_eq!(
      svg.matches("<svg x=").count(),
      3,
      "expected one nested SVG per returned state"
    );
  }
}
