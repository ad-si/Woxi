//! Regressions found while making Wolfram Demonstrations notebooks run:
//! semantic bugs that surfaced as generic Part/list/argument messages.
use super::*;

mod internal_messages {
  use super::*;

  // The elimination in Solve/Reduce tries the branch where a denominator
  // vanishes and meets `1/0` there. That is the solver's own bookkeeping:
  // the message must not be generated at all, so `Check` does not fire, it
  // is not in `$MessageList`, and a front end (which shows the captured
  // messages) reports nothing.
  const SYSTEM: &str = "Solve[{8.314 == (22.4*p)/a + (22.4*p)/b, \
    0.0036639754618530535*a == (0.009871668311944718*p)^(2/7), \
    2269.12 == 44.8*p}, {a, b, p}]";

  #[test]
  fn solve_elimination_raises_no_messages() {
    clear_state();
    let result = interpret_with_stdout(SYSTEM).unwrap();
    assert!(
      result.result.starts_with("{{a -> 223.89"),
      "{}",
      result.result
    );
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
  }

  #[test]
  fn solve_elimination_does_not_trip_check() {
    clear_state();
    assert_eq!(
      interpret(&format!("Head[Check[{SYSTEM}, failed]]")).unwrap(),
      "List"
    );
    assert_eq!(interpret(&format!("{SYSTEM}; $MessageList")).unwrap(), "{}");
  }

  #[test]
  fn user_division_by_zero_still_reported() {
    clear_state();
    let result = interpret_with_stdout("1/0").unwrap();
    assert_eq!(result.result, "ComplexInfinity");
    assert!(result.warnings.iter().any(|w| w.contains("Power::infy")));
  }
}

mod element_alternatives {
  use super::*;

  #[test]
  fn list_becomes_flat_alternatives() {
    clear_state();
    assert_eq!(
      interpret("ToString[FullForm[Element[{a, 2, b, c}, Integers]]]").unwrap(),
      "Element[Alternatives[a, b, c], Integers]"
    );
    assert_eq!(
      interpret("Element[{a, 2, b}, Integers]").unwrap(),
      "Element[a | b, Integers]"
    );
    assert_eq!(
      interpret("Element[{a, 2.5, b}, Integers]").unwrap(),
      "False"
    );
  }

  // A long chain used to be a nested binary `a | (b | (c | …))` as deep as
  // the list was long, and re-evaluating the stored value hit
  // $RecursionLimit (an integer program over 1640 variables).
  #[test]
  fn long_variable_list_reevaluates() {
    clear_state();
    assert_eq!(
      interpret(
        "c = Thread[Array[z, 1640] \\[Element] Integers]; {Head[c], Length[c[[1]]]}"
      )
      .unwrap(),
      "{Element, 1640}"
    );
  }
}

mod count_rule_pattern {
  use super::*;

  // Count has no transformation-rule form (unlike Cases): a rule is a
  // pattern that matches rules.
  #[test]
  fn rule_is_a_pattern() {
    clear_state();
    assert_eq!(
      interpret(
        "{Count[{a -> 1, b -> 0}, _ -> 1], Count[{a :> 1, b -> 0}, _ :> 1], \
         Count[{f[a -> 1], b -> 1}, _ -> 1, Infinity]}"
      )
      .unwrap(),
      "{1, 1, 2}"
    );
    assert_eq!(
      interpret("Cases[{a -> 1, b -> 0}, _ -> 1]").unwrap(),
      "{1, 1}"
    );
  }
}

mod integer_programs {
  use super::*;

  // An inexact integer program is solved in machine precision (as in
  // Wolfram); the exact search over the rationals of decimal inputs took
  // minutes on a few hundred variables. The continuous variables of a mixed
  // problem still come out at the exact vertex.
  #[test]
  fn mixed_linear_programming_keeps_exact_vertex() {
    clear_state();
    assert_eq!(
      interpret(
        "LinearProgramming[{1., 2.5}, {{1, 2}, {3, 1}}, {3.5, 2}, {0, 0}, {Integers, Reals}]"
      )
      .unwrap(),
      "{3, 0.25}"
    );
    assert_eq!(
      interpret(
        "LinearProgramming[{1.5, 2}, {{1, 1}}, {{2.5, 1}}, {{0, 3}, {0, 3}}, {Integers, Reals}]"
      )
      .unwrap(),
      "{2, 0.5}"
    );
  }

  // The facility-location program of the "Facilities Location Problem"
  // Demonstration: 600 binary variables x[i, j], y[j].
  #[test]
  fn facility_location_over_indexed_variables() {
    clear_state();
    let result = interpret(
      "n = 24; pts = Table[{Cos[2. Pi i/n] (1 + i/10.), Sin[2. Pi i/n]}, {i, n}]; \
       var = Flatten@Join[Table[x[i, j], {i, n}, {j, n}], Table[y[j], {j, n}]]; \
       obj = Sum[EuclideanDistance[pts[[i]], pts[[j]]]*x[i, j], {i, n}, {j, n}]; \
       cons = Flatten@{Thread[0 <= var <= 1], Thread[var \\[Element] Integers], \
         Table[x[i, j] <= y[j], {i, n}, {j, n}], Sum[y[j], {j, n}] <= 3, \
         Table[Sum[x[i, j], {j, n}] == 1, {i, n}]}; \
       sol = NMinimize[{obj, cons}, var]; \
       {sol[[1]], Cases[sol[[2]], HoldPattern[y[j_] -> 1] :> j], Count[sol[[2]], _ -> 1]}",
    )
    .unwrap();
    assert_eq!(result, "{18.96359150724976, {4, 11, 22}, 27}");
  }
}

mod combinatorica_tableaux {
  use super::*;

  #[test]
  fn tableaux_of_a_shape_in_package_order() {
    clear_state();
    assert_eq!(
      interpret(
        "Quiet@Get[\"Combinatorica`\"]; \
         {Tableaux[{2, 2}], Tableaux[{3, 1}], Combinatorica`Tableaux[{1}]}"
      )
      .unwrap(),
      "{{{{1, 3}, {2, 4}}, {{1, 2}, {3, 4}}}, \
       {{{1, 3, 4}, {2}}, {{1, 2, 4}, {3}}, {{1, 2, 3}, {4}}}, {{{1}}}}"
    );
    assert_eq!(
      interpret("Take[Combinatorica`Tableaux[{3, 2, 1}], 3]").unwrap(),
      "{{{1, 4, 6}, {2, 5}, {3}}, {{1, 3, 6}, {2, 5}, {4}}, \
       {{1, 2, 6}, {3, 5}, {4}}}"
    );
  }

  #[test]
  fn tableaux_of_every_partition() {
    clear_state();
    assert_eq!(
      interpret("Combinatorica`Tableaux[3]").unwrap(),
      "{{{1, 2, 3}}, {{1, 3}, {2}}, {{1, 2}, {3}}, {{1}, {2}, {3}}}"
    );
  }

  #[test]
  fn number_of_tableaux_by_hook_lengths() {
    clear_state();
    assert_eq!(
      interpret(
        "{Combinatorica`NumberOfTableaux[{3, 2, 1}], \
         Combinatorica`NumberOfTableaux[{4, 3, 1}], \
         Combinatorica`NumberOfTableaux[5], \
         Length[Combinatorica`Tableaux[{4, 3, 1}]]}"
      )
      .unwrap(),
      "{16, 70, 26, 70}"
    );
  }

  #[test]
  fn non_shapes_stay_unevaluated() {
    clear_state();
    assert_eq!(
      interpret("Combinatorica`Tableaux[{2, 3}]").unwrap(),
      "Combinatorica`Tableaux[{2, 3}]"
    );
  }
}

mod upvalue_pattern_variables {
  use super::*;

  // A pattern variable inside a compound argument of an upvalue
  // (`R[tag_]`) is bound to that part of the actual argument, also in a
  // held position: it used to become `R[x][[1]]`, so `axis[tag] =.` reported
  // Unset::norep (the SU2 spinor map Demonstration's object deletion).
  #[test]
  fn held_positions_see_the_matched_part() {
    clear_state();
    assert_eq!(
      interpret(
        "R /: R[tag_].f[] := Hold[tag]; R /: g[R[t_]] := Hold[t]; \
         {R[x].f[], g[R[y]]}"
      )
      .unwrap(),
      "{Hold[x], Hold[y]}"
    );
  }

  #[test]
  fn unset_through_an_upvalue_method() {
    clear_state();
    let result = interpret_with_stdout(
      "Module[{axis, key}, axis[_] := {}; \
         R /: new[R[a_]] := (key = Unique[]; axis[key] = a; Return[R[key]]); \
         R /: R[tag_].delete[] := (axis[tag] =.;); \
         R /: R[tag_].get[] := axis[tag]]; \
       r = new[R[{1, 2, 3}]]; {r.get[], r.delete[]; r.get[]}",
    )
    .unwrap();
    assert_eq!(result.result, "{{1, 2, 3}, {}}");
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
  }

  #[test]
  fn sequences_and_conditions() {
    clear_state();
    assert_eq!(
      interpret(
        "R /: h[R[a_, b__]] := {a, {b}}; R /: h2[R[a_, b___, c_]] := {a, {b}, c}; \
         tagG /: fH[tagG[x_]] := x + 1 /; x > 2; \
         S /: k[S[x_], S[x_]] := same; S /: k2[y_, S[x_]] := {x, y} /; x > y; \
         T /: m[T[x_]] /; x > 0 := pos; \
         {h[R[1, 2, 3]], h2[R[1, 2]], h[R[1]], fH[tagG[7]], fH[tagG[1]], \
          k[S[1], S[1]], k[S[1], S[2]], k2[1, S[2]], k2[3, S[2]], m[T[1]], m[T[-1]]}"
      )
      .unwrap(),
      "{{1, {2, 3}}, {1, {}, 2}, h[R[1]], 8, fH[tagG[1]], same, k[S[1], S[2]], \
       {2, 1}, k2[3, S[2]], pos, m[T[-1]]}"
    );
  }
}

mod unset_literal_values {
  use super::*;

  // `f[a] = 1` is kept apart from the pattern rules (the memoization
  // cache); `f[a] =.` has to find it there, with its arguments evaluated
  // as Set evaluates them.
  #[test]
  fn unset_removes_a_literal_value() {
    clear_state();
    assert_eq!(interpret("f[a] = 1; f[a] =.; f[a]").unwrap(), "f[a]");
    assert_eq!(
      interpret(
        "f[_] := 0; f[a] = 1; f[a] =.; k = Unique[]; g[k] = 1; g[k] =.; \
         {f[a], DownValues[f], DownValues[g]}"
      )
      .unwrap(),
      "{0, {HoldPattern[f[_]] :> 0}, {}}"
    );
  }

  #[test]
  fn unset_keeps_held_arguments_and_reports_missing() {
    clear_state();
    assert_eq!(
      interpret(
        "SetAttributes[h, HoldAll]; k = 1; h[k] = 5; h[k] =.; {h[k], DownValues[h]}"
      )
      .unwrap(),
      "{h[k], {}}"
    );
    let result = interpret_with_stdout("q[a] =.").unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(result.warnings.iter().any(|w| w.contains("Unset::norep")));
  }
}

mod upvalue_redefinition {
  use super::*;

  // A rule whose left-hand side differs only in pattern names replaces the
  // earlier one, as for down-values; the FullForm `Pattern[x, Blank[]]` a
  // SaveDefinitions dump writes is the same as `x_` (the SU2 spinor map
  // Demonstration replays its objects' methods that way).
  #[test]
  fn same_lhs_up_to_pattern_names_replaces() {
    clear_state();
    assert_eq!(
      interpret(
        "G /: G[t_].m[] := 1; G /: G[s_].m[] := 2; \
         U /: new[U[p_]] := 1; UpSetDelayed[new[U[Pattern[q$, Blank[]]]], 4]; \
         {UpValues[G], G[1].m[], Length[UpValues[U]], new[U[0]]}"
      )
      .unwrap(),
      "{{HoldPattern[G[s_] . m[]] :> 2}, 2, 1, 4}"
    );
  }

  #[test]
  fn shared_pattern_variables_still_distinguish_rules() {
    clear_state();
    assert_eq!(
      interpret(
        "S /: k[S[x_], S[x_]] := same; S /: k[S[x_], S[y_]] := diff; \
         {Length[UpValues[S]], k[S[1], S[1]], k[S[1], S[2]]}"
      )
      .unwrap(),
      "{2, same, diff}"
    );
  }
}

mod unique_fresh_names {
  use super::*;

  // Unique never hands out a symbol already in use: one with a value, or
  // one merely read in (the objects `$22`, `$23`, … a SaveDefinitions dump
  // restores).
  #[test]
  fn skips_symbols_in_use() {
    clear_state();
    let result = interpret(
      "$3 = 3; z = {$4, $2}; u = Table[Unique[], {4}]; \
       {Intersection[u, {$2, $3, $4}], Length[Union[u]]}",
    )
    .unwrap();
    assert_eq!(result, "{{}, 4}");
  }
}

mod upset_fullform_pattern_test {
  use super::*;

  // `r_h?test` written out as `PatternTest[Pattern[r, Blank[h]], test]` (a
  // SaveDefinitions dump of the potential-flow Demonstrations) still names
  // the tag `h`; it used to report UpSetDelayed::nosym.
  #[test]
  fn fullform_pattern_test_is_tagged_by_its_head() {
    clear_state();
    assert_eq!(
      interpret(
        "TwoDimQ[_] := True; \
         UpSetDelayed[ArcTan[PatternTest[Pattern[r, Blank[CV]], TwoDimQ]], at[r]]; \
         {ArcTan[CV[1, 2]], Length[UpValues[CV]]}"
      )
      .unwrap(),
      "{at[CV[1, 2]], 1}"
    );
  }
}

mod upvalue_sequence_slots {
  use super::*;

  // An upvalue's `v___h` / `a__` slot takes a run of arguments, also an
  // empty one: `Cross[u_CV, v___CV]` used to need exactly two arguments,
  // so the potential-flow Demonstrations' `Cross[ud]` stayed unevaluated.
  #[test]
  fn sequence_slots_take_runs_of_arguments() {
    clear_state();
    assert_eq!(
      interpret(
        "CV /: g[u_CV, v___CV] := gg[u, {v}]; CV /: h[a__, u_CV] := hh[{a}, u]; \
         {g[CV[1]], g[CV[1], CV[2], CV[3]], g[CV[1], 2], h[1, 2, CV[0]], h[CV[0]]}"
      )
      .unwrap(),
      "{gg[CV[1], {}], gg[CV[1], {CV[2], CV[3]}], g[CV[1], 2], hh[{1, 2}, CV[0]], h[CV[0]]}"
    );
  }
}

mod literal_rule_ordering {
  use super::*;

  // A rule with literal arguments is tried before a rule it is strictly
  // more specific than, also when that one has a literal argument too:
  // `f[0, 1]` before `f[l_, 1]`, `MPTPlot[{}, …, None, opts]` before
  // `MPTPlot[list_, …, None, opts]` (ModernPortfolioTheory).
  #[test]
  fn more_specific_literal_rule_goes_first() {
    clear_state();
    assert_eq!(
      interpret(
        "h1[l_, None] := gen; h1[{}, None] := empty; \
         h4[l_, 1] := gen; h4[0, 1] := zero; \
         f[l_, None, o___] := gen; f[{}, None, o___] := empty; \
         m[0, 1] := a; m[0, y_] := b; \
         {h1[{}, None], h1[{1}, None], h4[0, 1], h4[2, 1], f[{}, None], \
          f[{1}, None], m[0, 1], m[0, 2]}"
      )
      .unwrap(),
      "{empty, gen, zero, gen, empty, gen, a, b}"
    );
  }
}

mod function_interpolation_samples_held_expression {
  use super::*;

  // FunctionInterpolation samples its expression as written, with the
  // variable set to each number: a function defined only for numbers (as
  // the ballistics Demonstration's NDSolve-based F[Cd, …]) is interpolated
  // instead of evaluated once at the symbol.
  #[test]
  fn samples_numeric_only_function() {
    clear_state();
    assert_eq!(
      interpret(
        "k = 0; f[c_?NumericQ] := (k++; c^2); f[c_] := 0; \
         G = FunctionInterpolation[f[c], {c, 0, 1}]; \
         e = Sin[x]; H = FunctionInterpolation[e, {x, 0, 1}]; \
         x = 5; J = FunctionInterpolation[x^2, {x, 0, 2}]; \
         {G[0.5], k > 0, H[0.5], J[1.5], x}"
      )
      .unwrap(),
      "{0.25, True, 0.479425538604203, 2.25, 5}"
    );
  }
}

mod make_expression_boxes {
  use super::*;

  // A number standing in a box row is itself, and MakeExpression threads
  // over a list of boxes (SpreadsheetLikeSolution… reads its model's
  // variables back this way).
  #[test]
  fn numbers_in_boxes_and_lists_of_boxes() {
    clear_state();
    assert_eq!(
      interpret(
        "{ToExpression[RowBox[{\"g\", \"[\", 1, \"]\"}]], \
         MakeExpression[{RowBox[{\"g\", \"[\", 1, \"]\"}], RowBox[{\"g\", \"[\", 2, \"]\"}]}, StandardForm], \
         ToExpression[RowBox[{\"x\", \"+\", -2}]], MakeExpression[{}, StandardForm]}"
      )
      .unwrap(),
      "{g[1], {HoldComplete[g[1]], HoldComplete[g[2]]}, -2 + x, {}}"
    );
  }
}

mod solve_options_and_bad_domain {
  use super::*;

  // `WorkingPrecision -> p` numericizes the solution, and a third argument
  // that is no domain is a variable to eliminate (Solve::bdomv) — the
  // galactic-center ellipse Demonstration passes `Null` there.
  #[test]
  fn working_precision_and_null_domain() {
    clear_state();
    let result = interpret_with_stdout(
      "Solve[{a^2 == 2, b == 3}, {a, b}, Null, WorkingPrecision -> 20]",
    )
    .unwrap();
    assert_eq!(
      result.result,
      "{{a -> -1.41421356237309504880168872420969807857`20., b -> 3.`20.}, \
       {a -> 1.41421356237309504880168872420969807857`20., b -> 3.`20.}}"
    );
    assert!(result.warnings.iter().any(|w| w.contains("Solve::bdomv")));
    assert_eq!(
      interpret("Precision[a /. First[Solve[a^2 == 2, a, Reals, WorkingPrecision -> 30]]]")
        .unwrap(),
      "30."
    );
  }
}

mod filter_rules_patterns {
  use super::*;

  // FilterRules matches the left-hand sides with patterns (`Except[…]`,
  // `x[_]`), takes rules among the keys by their names, and treats an
  // option name as the same written as a string or a symbol.
  #[test]
  fn patterns_rules_and_string_names() {
    clear_state();
    assert_eq!(
      interpret(
        "{FilterRules[{a -> 1, b -> 2}, \"a\"], FilterRules[{a -> 1, b -> 2}, {a -> 5}], \
         FilterRules[{a -> 1, b :> 2, c -> 3}, Except[a]], \
         FilterRules[{x[1] -> 1, x[2] -> 2, y -> 3}, x[_]], FilterRules[{a -> 1}, {}], \
         FilterRules[{n -> {1}, Indexed[x, {2}] -> 3, c -> 4}, \
           Except[Alternatives[Indexed[x, {2}], Indexed[y, {2}]]]]}"
      )
      .unwrap(),
      "{{a -> 1}, {a -> 1}, {b :> 2, c -> 3}, {x[1] -> 1, x[2] -> 2}, {}, {n -> {1}, c -> 4}}"
    );
  }
}

mod turing_machine_head_displacement {
  use super::*;

  // An `{s, pos, x}` head spec starts the reported displacement at `x`
  // (FractalDimensionVersusTimeComplexityInTuringMachines); the whole
  // triple used to be taken for the state, so the machine never moved.
  #[test]
  fn three_element_head_spec() {
    clear_state();
    assert_eq!(
      interpret(
        "TuringMachine[{91290, 3, 2}, {{1, 2, 0}, {{1, 1, Null}, 0}}, 3]"
      )
      .unwrap(),
      "{{{1, 4, 0}, {0, 0, 1, 1, Null}}, {{1, 3, -1}, {0, 0, 1, 0, Null}}, \
       {{1, 2, -2}, {0, 0, 0, 0, Null}}, {{2, 1, -3}, {0, 0, 0, 0, Null}}}"
    );
    assert_eq!(
      interpret("TuringMachine[{91290, 3, 2}, {{1, 2, 5}, {1, 1, 0, 1}}, 2]")
        .unwrap(),
      "{{{1, 2, 5}, {1, 1, 0, 1}}, {{1, 1, 4}, {1, 0, 0, 1}}, {{1, 4, 3}, {0, 0, 0, 1}}}"
    );
  }
}
