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
  fn explicit_offset_neighborhood_matches_range() {
    for (offsets, r) in
      [("{{-1}, {0}, {1}}", 1), ("{{-2}, {-1}, {0}, {1}, {2}}", 2)]
    {
      let a = interpret(&format!(
        "CellularAutomaton[{{150, 2, {offsets}}}, {{{{1, 0, 1}}, 0}}, {{4, All}}]"
      ))
      .unwrap();
      let b = interpret(&format!(
        "CellularAutomaton[{{150, 2, {r}}}, {{{{1, 0, 1}}, 0}}, {{4, All}}]"
      ))
      .unwrap();
      assert_eq!(a, b);
      assert!(a.starts_with("{{") && !a.contains("CellularAutomaton"));
    }
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
  fn two_dimensional_windowed_single_step_is_bare() {
    // `{{t}, xspec, yspec}` (2D) restricts to a single time step; unlike
    // the bare `{{t}}` form it returns that state directly, not wrapped in
    // a length-1 list — the shape `ArrayPlot` needs to plot it straight
    // from `CellularAutomaton[rule, init, {{t}, All, All}]`.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{{1}}, All, All}]"
      )
      .unwrap(),
      "{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}"
    );
  }

  #[test]
  fn two_dimensional_windowed_multi_step_stays_a_list() {
    // `{t1, t2}` is the multi-step tspec — steps t1 through t2, one matrix
    // per step, exactly like the un-windowed `1` form. (The single-brace
    // `{1}` next to it is step 1 *alone*, still wrapped in a list.)
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{0, 1}, All, All}]"
      )
      .unwrap(),
      "{{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}}"
    );
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{1}, All, All}]"
      )
      .unwrap(),
      "{{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}}"
    );
  }

  #[test]
  fn two_dimensional_row_window_restricts_rows() {
    // An explicit row window keeps only the named rows, all columns.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{{1}}, {0, 0}, All}]"
      )
      .unwrap(),
      "{{1, 1, 1}}"
    );
  }

  #[test]
  fn two_dimensional_row_window_may_reach_past_the_affected_region() {
    // A row window wider than the rule's natural growth reads the
    // (all-zero) background above and below the affected rows.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{{1}}, {-2, 2}, All}]"
      )
      .unwrap(),
      "{{0, 0, 0}, {0, 1, 0}, {1, 1, 1}, {0, 1, 0}, {0, 0, 0}}"
    );
  }

  #[test]
  fn two_dimensional_column_window_restricts_columns() {
    // The `yspec` axis restricts columns the same way `xspec` restricts
    // rows; here the middle column alone still shows the vertical arm.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{{1}}, All, {0, 0}}]"
      )
      .unwrap(),
      "{{1}, {1}, {1}}"
    );
  }

  #[test]
  fn a_one_axis_window_on_a_two_dimensional_rule_windows_the_rows() {
    // `{tspec, xspec}` is legal on a 2D rule: the one window axis is the
    // *rows*, and the columns default to `All`.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{{1}}, All}]"
      )
      .unwrap(),
      "{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}"
    );
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{{1}}, {0, 0}}]"
      )
      .unwrap(),
      "{{1, 1, 1}}"
    );
    // Three window axes are one too many even for a 2D rule.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{{1}}, All, All, All}]"
      )
      .unwrap(),
      "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
       {1, 1}}, {{{1}}, 0}, {{{1}}, All, All, All}]"
    );
  }

  #[test]
  fn a_two_axis_window_is_invalid_on_a_one_dimensional_rule() {
    // `{tspec, xspec, yspec}` (two window axes) only makes sense for a 2D
    // rule.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{{1}}, All, All}]").unwrap(),
      "CellularAutomaton[90, {{1}, 0}, {{{1}}, All, All}]"
    );
  }

  #[test]
  fn the_leading_element_of_a_step_spec_is_always_the_tspec() {
    // `{{t1, t2}, xspec}` is the `{t1, t2}` tspec (steps t1 through t2)
    // with a space window, never a nested `{tspec, xspec}` window.
    assert_eq!(
      interpret(
        "CellularAutomaton[{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, \
         {1, 1}}, {{{1}}, 0}, {{0, 1}, All, All}]"
      )
      .unwrap(),
      "{{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{0, 1}, All}]").unwrap(),
      "{{0, 1, 0}, {1, 0, 1}}"
    );
    // Conversely `{1, 3}` is tspec 1 (steps 0 through 1) windowed to cells
    // 0 through 3 — *not* steps 1 through 3.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {1, 3}]").unwrap(),
      "{{1, 0, 0, 0}, {0, 1, 0, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{1, 3}}]").unwrap(),
      "{{0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 0, 0, 1, 0}, \
       {1, 0, 1, 0, 1, 0, 1}}"
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

// The `{tspec, xspec}` step specification restricts the returned cells, and
// the 2-argument / operator forms run a single step. All values verified
// against wolframscript.
// The third argument's grammar: it is either a bare `t` or the list
// `{tspec, xspec[, yspec]}`, whose *leading* element is always the tspec.
// Woxi used to read `{t}` as "steps 0 through t" and `{t1, t2}` as a
// `{tspec, xspec}` window, which made `{{1}, All, All}` return two states
// where wolframscript returns one, and rejected `{{0, 1}, All}` outright.
mod time_specifications {
  use super::*;

  const R2D: &str =
    "{942, {2, {{0, 2, 0}, {2, 1, 2}, {0, 2, 0}}}, {1, 1}}, {{{1}}, 0}";

  #[test]
  fn a_bare_t_and_a_one_element_spec_list_both_mean_steps_0_through_t() {
    clear_state();
    let all = "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 1, 0, 0}, \
               {0, 1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, 0, 1}}";
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, 3]").unwrap(),
      all
    );
    // `{3}` is the spec *list* `{tspec}` with tspec = 3, not the `{t}`
    // tspec — same answer, by a different route.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {3}]").unwrap(),
      all
    );
  }

  #[test]
  fn a_braced_t_selects_that_step_alone_but_keeps_the_list() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{3}}]").unwrap(),
      "{{1, 0, 1, 0, 1, 0, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{3}, All}]").unwrap(),
      "{{1, 0, 1, 0, 1, 0, 1}}"
    );
  }

  #[test]
  fn a_doubly_braced_t_returns_the_state_bare_with_or_without_a_window() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{{3}}}]").unwrap(),
      "{1, 0, 1, 0, 1, 0, 1}"
    );
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{{3}}, All}]").unwrap(),
      "{1, 0, 1, 0, 1, 0, 1}"
    );
    assert_eq!(
      interpret(&format!("CellularAutomaton[{R2D}, {{{{{{1}}}}}}]")).unwrap(),
      "{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}"
    );
  }

  #[test]
  fn a_two_or_three_element_tspec_is_a_step_range() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{1, 3}}]").unwrap(),
      "{{0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 0, 0, 1, 0}, \
       {1, 0, 1, 0, 1, 0, 1}}"
    );
    // `{t1, t2, dt}` keeps every dt-th step.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{0, 4, 2}}]").unwrap(),
      "{{0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 1, 0, 0}, \
       {1, 0, 0, 0, 0, 0, 0, 0, 1}}"
    );
    // Written without the extra braces those are window specs instead: the
    // three-element one names two space windows, which a 1D rule cannot
    // take.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {0, 4, 2}]").unwrap(),
      "CellularAutomaton[90, {{1}, 0}, {0, 4, 2}]"
    );
  }

  #[test]
  fn the_single_state_form_takes_exactly_one_step_number() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{{1, 3}}}]").unwrap(),
      "CellularAutomaton[90, {{1}, 0}, {{{1, 3}}}]"
    );
  }

  #[test]
  fn a_one_dimensional_rule_takes_at_most_one_space_window() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{{1}}, All, All}]").unwrap(),
      "CellularAutomaton[90, {{1}, 0}, {{{1}}, All, All}]"
    );
  }
}

mod step_and_cell_specs {
  use super::*;

  #[test]
  fn all_keeps_every_affected_cell() {
    clear_state();
    // `{t, All}` matches the plain `t` form for both init kinds.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {3, All}]").unwrap(),
      "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 1, 0, 0}, \
       {0, 1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, 0, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {0, 0, 0, 1, 0, 0, 0}, {2, All}]")
        .unwrap(),
      "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0, 0}, {0, 1, 1, 0, 0, 1, 0}}"
    );
  }

  // Offset 0 is the first cell of the initial condition.
  #[test]
  fn a_cell_range_windows_the_evolution() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {3, {-2, 2}}]").unwrap(),
      "{{0, 0, 1, 0, 0}, {0, 1, 0, 1, 0}, {1, 0, 0, 0, 1}, {0, 1, 0, 1, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {3, {0, 0}}]").unwrap(),
      "{{1}, {0}, {0}, {0}}"
    );
    // A multi-cell init still puts offset 0 on its first cell.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1, 1, 1}, 0}, {2, {0, 2}}]").unwrap(),
      "{{1, 1, 1}, {1, 0, 1}, {1, 0, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[90, {{1, 1, 1}, 0}, {2, {-1, 3}}]").unwrap(),
      "{{0, 1, 1, 1, 0}, {1, 1, 0, 1, 1}, {1, 1, 0, 1, 1}}"
    );
  }

  // A window wider than the affected region reads the background.
  #[test]
  fn a_window_may_reach_past_the_affected_region() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {2, {-5, 5}}]").unwrap(),
      "{{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, \
       {0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0}, \
       {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}}"
    );
  }

  // A cyclic init wraps around instead.
  #[test]
  fn a_cyclic_window_wraps() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[30, {0, 0, 0, 1, 0, 0, 0}, {2, {0, 2}}]")
        .unwrap(),
      "{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {0, 0, 0, 1, 0, 0, 0}, {2, {-1, 1}}]")
        .unwrap(),
      "{{0, 0, 0}, {0, 0, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn a_cell_range_may_carry_a_step() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {3, {-2, 2, 2}}]").unwrap(),
      "{{0, 1, 0}, {0, 0, 0}, {1, 0, 1}, {0, 0, 0}}"
    );
  }

  // A bare integer xspec runs from the origin out to that offset.
  #[test]
  fn an_integer_cell_spec_runs_from_the_origin() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {3, 2}]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {3, -2}]").unwrap(),
      "{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {0, 1, 0}}"
    );
  }

  #[test]
  fn one_dimensional_windowed_single_step_is_bare() {
    clear_state();
    // `{{{t}}, xspec}` (1D) names the explicit single state t; like the 2D
    // `{{{t}}, All, All}` case it returns that state bare, not wrapped in
    // a length-1 list.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{{3}}, {-2, 2}}]").unwrap(),
      "{0, 1, 0, 1, 0}"
    );
  }

  #[test]
  fn a_windowed_spec_that_merely_resolves_to_one_state_still_wraps() {
    clear_state();
    // `{0, All}` names steps "0 through 0" via the bare-`t` form (t = 0),
    // not the explicit single-state `{{t}}` form — even though it
    // resolves to exactly one state, it must keep returning a length-1
    // list, matching bare `t` always returning a list (`zero_steps` in the
    // `cellular_automaton` module) and the documented `{t, All} == t`
    // equivalence.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {0, {-2, 2}}]").unwrap(),
      "{{0, 0, 1, 0, 0}}"
    );
    // Likewise a `{t1, t2}` range that happens to collapse to one step.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{5, 5}, {-2, 2}}]").unwrap(),
      "{{0, 0, 0, 0, 0}}"
    );
    // The single-state form `{{t}}` takes exactly one step number, so
    // `{{5, 5}}` is not a tspec at all and the call stays unevaluated.
    assert_eq!(
      interpret("CellularAutomaton[90, {{1}, 0}, {{{5, 5}}, {-2, 2}}]")
        .unwrap(),
      "CellularAutomaton[90, {{1}, 0}, {{{5, 5}}, {-2, 2}}]"
    );
  }

  #[test]
  fn a_huge_explicit_window_is_rejected_before_allocating() {
    clear_state();
    // A cyclic init never grows past its own size, so nothing earlier
    // bounds an explicit window's length against it; the window's size
    // must be checked arithmetically before any per-cell allocation, or
    // this would try to collect roughly a billion indices.
    assert_eq!(
      interpret("CellularAutomaton[90, {1, 0, 1}, {{{1}}, {0, 1000000000}}]")
        .unwrap(),
      "CellularAutomaton[90, {1, 0, 1}, {{{1}}, {0, 1000000000}}]"
    );
  }

  // Without a step count only one step runs, and the result is the new state
  // alone — as a `{cells, {background}}` pair for a background init.
  #[test]
  fn the_two_argument_form_runs_one_step() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[30, {0, 0, 0, 1, 0, 0, 0}]").unwrap(),
      "{0, 0, 1, 1, 1, 0, 0}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}]").unwrap(),
      "{{1, 1, 1}, {0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {{1, 0, 1}, 0}]").unwrap(),
      "{{1, 1, 0, 1, 1}, {0}}"
    );
  }

  #[test]
  fn the_operator_form_runs_one_step() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[30][{0, 0, 1, 0, 0}]").unwrap(),
      "{0, 1, 1, 1, 0}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30][{{1}, 0}]").unwrap(),
      "{{1, 1, 1}, {0}}"
    );
  }

  #[test]
  fn a_sparse_array_init_is_its_dense_list() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[30, SparseArray[{3 -> 1}, 5], {2, All}]")
        .unwrap(),
      interpret("CellularAutomaton[30, {0, 0, 1, 0, 0}, {2, All}]").unwrap()
    );
    assert_eq!(
      interpret("CellularAutomaton[30, SparseArray[{3 -> 1}, 5], {{1}}]")
        .unwrap(),
      "{{0, 1, 1, 1, 0}}"
    );
  }

  // Half-integer ranges, offset-list neighborhoods, the explicit `All`
  // window, rule-number bounds and the legacy four-argument form. All
  // values verified against wolframscript.

  #[test]
  fn a_half_integer_range_reaches_one_cell_further_left() {
    clear_state();
    // r = 1/2 reads offsets -1 and 0: rule 4 (left cell 1, own cell 0)
    // moves a cell right, rule 2 keeps an isolated cell in place.
    assert_eq!(
      interpret("CellularAutomaton[{4, 2, 1/2}, {0, 0, 1, 0, 0, 0}, 2]")
        .unwrap(),
      "{{0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{2, 2, 1/2}, {{1}, 0}, 3]").unwrap(),
      "{{1}, {1}, {1}, {1}}"
    );
    // r = 3/2 reads offsets -2 through 1; rule 43690 copies offset 1, so
    // the pattern moves left.
    assert_eq!(
      interpret(
        "CellularAutomaton[{43690, 2, 3/2}, {1, 0, 0, 1, 1, 0, 1, 0}, 3]"
      )
      .unwrap(),
      "{{1, 0, 0, 1, 1, 0, 1, 0}, {0, 0, 1, 1, 0, 1, 0, 1}, \
       {0, 1, 1, 0, 1, 0, 1, 0}, {1, 1, 0, 1, 0, 1, 0, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{6, 2, 3/2}, {{1}, 0}, 2]").unwrap(),
      "{{0, 0, 1}, {0, 1, 1}, {1, 0, 0}}"
    );
  }

  #[test]
  fn half_integer_ranges_take_weights_and_two_dimensions() {
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[{6, {2, 1}, 1/2}, {{1}, 0}, 3]").unwrap(),
      "{{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{6, {2, {1, 2}}, 1/2}, {{1}, 0}, 3]")
        .unwrap(),
      "{{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 1, 1, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{6, 2, {1/2, 1/2}}, {{1, 0}, {0, 0}}, 1]")
        .unwrap(),
      "{{{1, 0}, {0, 0}}, {{1, 1}, {0, 0}}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{2, 2, {1/2, 5}}, {{{1}}, 0}, 1]").unwrap(),
      "{{{0, 0, 0, 0, 0, 1}}, {{1, 0, 0, 0, 0, 0}}}"
    );
  }

  #[test]
  fn an_offset_list_weights_its_cells_in_list_order() {
    clear_state();
    // The first offset gets the highest positional weight, so swapping
    // the offsets reverses the direction of travel.
    assert_eq!(
      interpret("CellularAutomaton[{2, 2, {{1}, {-1}}}, {{1}, 0}, 2]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{2, 2, {{-1}, {1}}}, {{1}, 0}, 2]").unwrap(),
      "{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}"
    );
    assert_eq!(
      interpret(
        "CellularAutomaton[{2, {2, {1, 3}}, {{-1}, {1}}}, {{1}, 0}, 2]"
      )
      .unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{6, 2, {{-1}, {1}}}, {0, 1, 0, 0, 1}, 2]")
        .unwrap(),
      "{{0, 1, 0, 0, 1}, {0, 0, 1, 1, 0}, {0, 1, 1, 1, 1}}"
    );
    // Two-dimensional offsets take a flat weight list.
    assert_eq!(
      interpret(
        "CellularAutomaton[{6, {2, {1, 2}}, {{-1, 0}, {0, 1}}}, {{{1}}, 0}, 1]"
      )
      .unwrap(),
      "{{{0, 1}, {0, 0}}, {{1, 0}, {0, 1}}}"
    );
  }

  #[test]
  fn an_explicit_all_keeps_the_whole_affected_region() {
    clear_state();
    // Without a window only cells that ever leave the background are kept;
    // `All` keeps every cell the rule could have reached, which for a
    // lopsided neighborhood extends further on one side.
    assert_eq!(
      interpret("CellularAutomaton[{2, 2, 1}, {{1}, 0}, 3]").unwrap(),
      "{{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{2, 2, 1}, {{1}, 0}, {3, All}]").unwrap(),
      "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, \
       {0, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{2, 2, 1/2}, {{1}, 0}, {3, All}]").unwrap(),
      "{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[{2, 2, {{1}, {2}}}, {{1}, 0}, {2, All}]")
        .unwrap(),
      "{{0, 0, 0, 0, 1}, {0, 0, 1, 0, 0}, {1, 0, 0, 0, 0}}"
    );
    // Per axis in 2D: the rows are `All`, the unnamed columns are trimmed.
    let identity = "{2, {2, {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}}, {1, 1}}";
    assert_eq!(
      interpret(&format!(
        "CellularAutomaton[{identity}, {{{{{{1}}}}, 0}}, {{1, All, All}}]"
      ))
      .unwrap(),
      "{{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}}"
    );
    assert_eq!(
      interpret(&format!(
        "CellularAutomaton[{identity}, {{{{{{1}}}}, 0}}, {{1, All}}]"
      ))
      .unwrap(),
      "{{{0}, {1}, {0}}, {{0}, {1}, {0}}}"
    );
  }

  #[test]
  fn the_four_argument_form_reads_its_offsets_relative_to_t() {
    clear_state();
    let full = "{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0, 0}, \
                {0, 1, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 1, 1}}";
    for offs in ["{All, All}", "All", "{All}"] {
      assert_eq!(
        interpret(&format!("CellularAutomaton[30, {{{{1}}, 0}}, 3, {offs}]"))
          .unwrap(),
        full
      );
    }
    // A non-negative offt is a modern tspec, even beyond t.
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 3, {2, 3}]").unwrap(),
      "{{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 1, 0}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 3, {{{1}}}]").unwrap(),
      "{1, 1, 1}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 2, {{5}}]").unwrap(),
      "{{1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1}}"
    );
    // Negative steps count back from t: -n alone is the last n steps.
    let last_two = "{{0, 1, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 1, 1, 1}}";
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 3, -2]").unwrap(),
      last_two
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 3, {-2}]").unwrap(),
      last_two
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 3, {{-1}}]").unwrap(),
      "{{1, 1, 0, 1, 1, 1, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 3, {{-3, -2}}]").unwrap(),
      "{{0, 1, 1, 1, 0}, {1, 1, 0, 0, 1}}"
    );
    assert_eq!(
      interpret("CellularAutomaton[30, {{1}, 0}, 3, {All, -2}]").unwrap(),
      "{{0, 0, 1}, {0, 1, 1}, {1, 1, 0}, {1, 0, 1}}"
    );
  }

  #[test]
  fn the_reversible_rule_demonstration_call_evaluates() {
    // From the "Reversible Two-Color Cellular Automaton Rules"
    // Demonstration: `CellularAutomaton[{rule, 2, 3/2}, init, n, {All, All}]`.
    clear_state();
    assert_eq!(
      interpret(
        "CellularAutomaton[{43690, 2, 3/2}, {{1, 0, 1}, 0}, 3, {All, All}]"
      )
      .unwrap(),
      "{{0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0}, \
       {0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, \
       {0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}}"
    );
  }

  #[test]
  fn malformed_specs_emit_wolfram_messages() {
    let cases = [
      (
        "CellularAutomaton[30, {{1}, 0}, {3}, {All, All}]",
        "CellularAutomaton::offg4: Since four arguments are used, the third \
         argument {3} must be a non-negative machine integer, and the fourth \
         argument {All, All} must be an integer or {offt, offx1, offx2, ... \
         offxk}, where offt is present and k does not exceed 1, the number of \
         spatial dimensions.",
      ),
      (
        "CellularAutomaton[30, {{1}, 0}, 3, {-9}]",
        "CellularAutomaton::offtm4: Step size 3 and temporal offset \
         specification -9 request step -5, but the lowest step is 0.",
      ),
      (
        "CellularAutomaton[{6, 2, 0.5}, {{1}, 0}, 3]",
        "CellularAutomaton::rspec: The neighbor specification at position 3 \
         of rule specification {6, 2, 0.5} should be a radius",
      ),
      (
        "CellularAutomaton[{6, 2, {{0}, {0}}}, {{1}, 0}, 1]",
        "CellularAutomaton::offrep: The offsets at position 3 in rule \
         specification {6, 2, {{0}, {0}}}",
      ),
      (
        "CellularAutomaton[{2, 2, {{0, 1}, {1}}}, {{1}, 0}, 1]",
        "CellularAutomaton::offfo: Because rule specification \
         {2, 2, {{0, 1}, {1}}} has no specified order",
      ),
      (
        "CellularAutomaton[{6, {2, {1, 2, 4, 8, 16}}}, {{1}, 0}, 1]",
        "CellularAutomaton::wtspec: The weight specification at position \
         {2, 2} of rule specification {6, {2, {1, 2, 4, 8, 16}}} should be 1 \
         or an array of non-negative machine integers with dimensions {3}.",
      ),
      (
        "CellularAutomaton[{2, {2, {1, 2, 3}}, {{0}, {1}}}, {{1}, 0}, 1]",
        "CellularAutomaton::wtlen: The length of the weight list {1, 2, 3} \
         does not match the number of cells in the neighborhood {{0}, {1}}.",
      ),
      (
        "CellularAutomaton[300, {{1}, 0}, 1]",
        "CellularAutomaton::rsize: The specified rule number 300 is greater \
         than the largest possible rule number (255).",
      ),
      (
        "CellularAutomaton[{16, {2, 1}}, {{1}, 0}, 1]",
        "CellularAutomaton::rsize: The specified rule number 16 is greater \
         than the largest possible rule number (15).",
      ),
    ];
    for (input, message) in cases {
      clear_state();
      let result = interpret_with_stdout(input).unwrap();
      assert_eq!(result.result, input, "{input}");
      assert!(
        result.warnings.iter().any(|w| w.contains(message)),
        "{input}: {:?}",
        result.warnings
      );
    }
    // The largest rule number itself is fine.
    clear_state();
    assert_eq!(
      interpret("CellularAutomaton[{15, {2, 1}}, {{1}, 0}, 1]").unwrap(),
      "{{1}, {1}}"
    );
  }
}
