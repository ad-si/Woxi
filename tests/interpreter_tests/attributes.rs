use super::*;

mod set_attributes {
  use super::*;

  #[test]
  fn listable_threads_over_list() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Listable]; f[x_] := x * 2; f[{1, 2, 3}]")
        .unwrap(),
      "{2, 4, 6}"
    );
  }

  #[test]
  fn listable_with_conditions() {
    clear_state();
    assert_eq!(
        interpret(r#"SetAttributes[f, Listable]; f[n_ /; Mod[n, 3] == 0] := "Fizz"; f[n_] := n; f[{1, 2, 3, 4, 5, 6}]"#).unwrap(),
        r#"{1, 2, Fizz, 4, 5, Fizz}"#
      );
  }

  #[test]
  fn listable_single_value_unchanged() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Listable]; f[x_] := x + 1; f[5]").unwrap(),
      "6"
    );
  }

  #[test]
  fn listable_nested_lists() {
    assert_eq!(
      interpret("{{1, 2}, {3, 4}} + {5, 6}").unwrap(),
      "{{6, 7}, {9, 10}}"
    );
  }
}

mod flat_attribute {
  use super::*;

  #[test]
  fn flat_flattens_nested_calls() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Flat]; f[a, f[b, c]]").unwrap(),
      "f[a, b, c]"
    );
  }

  #[test]
  fn flat_flattens_left_nested() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Flat]; f[f[a, b], c]").unwrap(),
      "f[a, b, c]"
    );
  }

  #[test]
  fn flat_no_effect_without_attribute() {
    clear_state();
    assert_eq!(interpret("g[a, g[b, c]]").unwrap(), "g[a, g[b, c]]");
  }
}

mod orderless_attribute {
  use super::*;

  #[test]
  fn orderless_sorts_symbols() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Orderless]; f[c, a, b]").unwrap(),
      "f[a, b, c]"
    );
  }

  #[test]
  fn orderless_sorts_numbers() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Orderless]; f[3, 1, 2]").unwrap(),
      "f[1, 2, 3]"
    );
  }

  #[test]
  fn orderless_numbers_before_symbols() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Orderless]; f[b, 1, a, 3]").unwrap(),
      "f[1, 3, a, b]"
    );
  }

  #[test]
  fn orderless_no_effect_without_attribute() {
    clear_state();
    assert_eq!(interpret("g[c, a, b]").unwrap(), "g[c, a, b]");
  }

  #[test]
  fn orderless_compound_expressions_after_symbols() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Orderless]; f[c, a, b, a + b, 3, 1.0]")
        .unwrap(),
      "f[1., 3, a, b, a + b, c]"
    );
  }

  #[test]
  fn orderless_equality() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Orderless]; f[a, b] == f[b, a]").unwrap(),
      "True"
    );
  }
}

mod flat_and_orderless {
  use super::*;

  #[test]
  fn flat_orderless_combined() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, {Flat, Orderless}]; f[b, f[a, c]]").unwrap(),
      "f[a, b, c]"
    );
  }

  #[test]
  fn flat_subsequence_replace_all() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Flat]; f[a, b, c] /. f[a, b] -> d").unwrap(),
      "f[d, c]"
    );
  }

  #[test]
  fn flat_subsequence_replace_all_end() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, Flat]; f[a, b, c] /. f[b, c] -> d").unwrap(),
      "f[a, d]"
    );
  }

  #[test]
  fn orderless_subset_replace_all() {
    clear_state();
    // With Flat+Orderless, f[a, c] matches non-contiguous subset of f[a, b, c]
    assert_eq!(
      interpret(
        "SetAttributes[f, {Flat, Orderless}]; f[a, b, c] /. f[a, c] -> d"
      )
      .unwrap(),
      "f[b, d]"
    );
  }

  #[test]
  fn orderless_subset_replace_all_reversed() {
    clear_state();
    // Pattern f[c, a] should also match (Orderless allows reordering)
    assert_eq!(
      interpret(
        "SetAttributes[f, {Flat, Orderless}]; f[a, b, c] /. f[c, a] -> d"
      )
      .unwrap(),
      "f[b, d]"
    );
  }
}

mod one_identity_attribute {
  use super::*;

  #[test]
  fn one_identity_basic_match() {
    // With OneIdentity, a /. f[x_:0, u_] -> {u} matches a as f[0, a]
    assert_eq!(
      interpret("SetAttributes[f, OneIdentity]; a /. f[x_:0, u_] -> {u}")
        .unwrap(),
      "{a}"
    );
  }

  #[test]
  fn one_identity_with_default_binding() {
    // The default value should be bound to the optional pattern variable
    assert_eq!(
      interpret("SetAttributes[f, OneIdentity]; a /. f[x_:0, u_] -> {x, u}")
        .unwrap(),
      "{0, a}"
    );
  }

  #[test]
  fn one_identity_no_match_without_attribute() {
    // Without OneIdentity, the pattern should not match
    assert_eq!(interpret("a /. f[x_:0, u_] -> {u}").unwrap(), "a");
  }

  #[test]
  fn one_identity_requires_default_in_pattern() {
    // OneIdentity alone doesn't make a bare `a` match `f[u_]` — the
    // pattern must include an Optional/default slot (e.g. `x_:0`) so
    // OneIdentity has somewhere to fold the missing arguments. With
    // just `f[u_]`, `a` stays unmatched.
    assert_eq!(
      interpret("SetAttributes[f, OneIdentity]; a /. f[u_] -> {u}").unwrap(),
      "a"
    );
  }

  #[test]
  fn one_identity_direct_function_call_still_matches() {
    // Direct function calls should still match normally
    assert_eq!(
      interpret(
        "SetAttributes[f, OneIdentity]; f[3, a] /. f[x_:0, u_] -> {x, u}"
      )
      .unwrap(),
      "{3, a}"
    );
  }

  #[test]
  fn one_identity_times_system_default() {
    // a_.*x_^n_. in function definition: Times has OneIdentity,
    // so x^2 should match with a=1 (Default[Times])
    assert_eq!(
      interpret("f[a_.*x_^n_.] := {a, x, n}; f[y^2]").unwrap(),
      "{1, y, 2}"
    );
  }

  #[test]
  fn one_identity_times_and_power_system_default() {
    // When only a variable is passed, both Times and Power OneIdentity
    // should fill in defaults: a=1 (Default[Times]), n=1 (Default[Power,2])
    assert_eq!(
      interpret("f[a_.*x_^n_.] := {a, x, n}; f[y]").unwrap(),
      "{1, y, 1}"
    );
  }

  #[test]
  fn one_identity_power_system_default() {
    // 3*y should match a_.*x_^n_. with a=y, x=3, n=1 (Orderless Times matching)
    assert_eq!(
      interpret("f[a_.*x_^n_.] := {a, x, n}; f[3*y]").unwrap(),
      "{y, 3, 1}"
    );
  }

  #[test]
  fn one_identity_times_explicit_values() {
    // 3*y^2 should match a_.*x_^n_. with a=y^2, x=3, n=1 (Orderless Times matching)
    assert_eq!(
      interpret("f[a_.*x_^n_.] := {a, x, n}; f[3*y^2]").unwrap(),
      "{y^2, 3, 1}"
    );
  }

  #[test]
  fn one_identity_integration_pattern() {
    // Regression test for GitHub issue #57
    assert_eq!(
      interpret("Int[a_.*x_^n_.,x_Symbol] := a*x^(n+1)/(n+1); Int[x^2, x]")
        .unwrap(),
      "x^3/3"
    );
  }

  #[test]
  fn one_identity_plus_system_default() {
    // Plus has OneIdentity with Default[Plus]=0
    assert_eq!(interpret("g[a_. + b_] := {a, b}; g[x]").unwrap(), "{0, x}");
  }
}

mod protect_unprotect {
  use super::*;

  #[test]
  fn protect_blocks_simple_assignment() {
    clear_state();
    assert_eq!(interpret("Protect[p]; p = 2; p").unwrap(), "p");
  }

  #[test]
  fn set_protected_constant_returns_rhs() {
    // `Pi = 4` should emit `Set::wrsym` and return 4 (the RHS), matching
    // wolframscript. Pi parses as `Expr::Constant("Pi")`, so the
    // simple-identifier path needs to accept both Identifier and
    // Constant variants.
    assert_eq!(interpret("Pi = 4").unwrap(), "4");
  }

  #[test]
  fn clear_protected_constant_returns_null() {
    // `Pi = 4; Clear[Pi]` emits warnings on both statements but the
    // final result is Null (from Clear). `interpret` uses "\0" as the
    // sentinel for Null so the CLI can suppress it cleanly.
    assert_eq!(interpret("Pi = 4; Clear[Pi]").unwrap(), "\0");
  }

  #[test]
  fn protect_blocks_part_assignment() {
    clear_state();
    assert_eq!(
      interpret("A = {1, 2, 3}; Protect[A]; A[[2]] = 4; A").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn protect_returns_symbol_list() {
    clear_state();
    assert_eq!(interpret("Protect[x]").unwrap(), "{x}");
  }

  #[test]
  fn unprotect_removes_protection() {
    clear_state();
    assert_eq!(
      interpret("Protect[x]; Unprotect[x]; x = 5; x").unwrap(),
      "5"
    );
  }

  #[test]
  fn unprotect_returns_symbol_if_was_protected() {
    clear_state();
    assert_eq!(interpret("Protect[x]; Unprotect[x]").unwrap(), "{x}");
  }

  #[test]
  fn unprotect_returns_empty_if_not_protected() {
    clear_state();
    assert_eq!(interpret("Unprotect[x]").unwrap(), "{}");
  }

  #[test]
  fn unprotect_builtin_protected_symbol_returns_list() {
    // Sin has Protected via builtin attributes; Unprotect should report it.
    assert_eq!(interpret("Unprotect[Sin]").unwrap(), "{Sin}");
  }

  #[test]
  fn unprotect_multiple_builtins_returns_all() {
    assert_eq!(interpret("Unprotect[Cos, Tan]").unwrap(), "{Cos, Tan}");
  }

  #[test]
  fn protected_via_attributes_assignment() {
    clear_state();
    assert_eq!(
      interpret("Attributes[p] = {Protected}; p = 2; p").unwrap(),
      "p"
    );
  }

  #[test]
  fn set_attributes_on_protected_symbol() {
    clear_state();
    // SetAttributes can add attributes even when symbol is Protected
    assert_eq!(
      interpret(
        "Attributes[p] = {Protected}; SetAttributes[p, Flat]; Attributes[p]"
      )
      .unwrap(),
      "{Flat, Protected}"
    );
  }

  #[test]
  fn unprotect_pi_drops_protected_from_attributes() {
    // `Unprotect[Pi]` should remove `Protected` from Pi's reported
    // attributes, even though it's a builtin attribute. Pi parses as
    // `Expr::Constant("Pi")` rather than `Expr::Identifier`, so the
    // handler must accept both variants.
    assert_eq!(
      interpret("Unprotect[Pi]; Attributes[Pi]").unwrap(),
      "{Constant, ReadProtected}"
    );
  }

  #[test]
  fn protect_pi_restores_protected() {
    assert_eq!(
      interpret("Unprotect[Pi]; Protect[Pi]; Attributes[Pi]").unwrap(),
      "{Constant, Protected, ReadProtected}"
    );
  }

  #[test]
  fn clear_all_pi_drops_all_builtin_attributes() {
    // ClearAll should wipe both user and builtin attributes.
    assert_eq!(
      interpret("Unprotect[Pi]; ClearAll[Pi]; Attributes[Pi]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn clear_pi_keeps_remaining_builtin_attributes() {
    // Clear (without "All") doesn't remove attributes — only OwnValues.
    assert_eq!(
      interpret("Unprotect[Pi]; Clear[Pi]; Attributes[Pi]").unwrap(),
      "{Constant, ReadProtected}"
    );
  }

  #[test]
  fn unprotect_blocked_by_locked() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[p, {Protected, Locked}]; Unprotect[p]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn override_builtin_function_with_user_rule() {
    // Regression test: user-defined rules should take precedence over built-in
    // implementations when the function is Unprotected and a matching rule exists.
    clear_state();
    assert_eq!(
      interpret(
        "Unprotect[PolynomialQ]; PolynomialQ[u_List, x_Symbol] := Foo[u, x]; \
         Protect[PolynomialQ]; PolynomialQ[{x + 2}, x]"
      )
      .unwrap(),
      "Foo[{2 + x}, x]"
    );
  }

  #[test]
  fn override_builtin_falls_through_to_builtin() {
    // When user rule doesn't match, built-in should still work
    clear_state();
    assert_eq!(
      interpret(
        "Unprotect[PolynomialQ]; PolynomialQ[u_List, x_Symbol] := Foo[u, x]; \
         Protect[PolynomialQ]; PolynomialQ[x^2 + 1, x]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod attributes_assignment {
  use super::*;

  #[test]
  fn set_attributes_via_assignment() {
    clear_state();
    assert_eq!(
      interpret("ClearAll[f]; Attributes[f] = {Listable}; Attributes[f]")
        .unwrap(),
      "{Listable}"
    );
  }

  #[test]
  fn set_attributes_via_set_delayed() {
    clear_state();
    assert_eq!(
      interpret("ClearAll[f]; Attributes[f] := {Flat}; Attributes[f]").unwrap(),
      "{Flat}"
    );
  }

  #[test]
  fn set_attributes_via_set_delayed_returns_null() {
    // `Attributes[f] := {...}` is SetDelayed; its direct result should be
    // Null (no visible output), not the RHS — matching wolframscript.
    // Regression for mathics symbols.py:241.
    clear_state();
    assert_eq!(interpret("Attributes[r] := {Orderless}").unwrap(), "\0");
  }

  #[test]
  fn set_attributes_with_symbol() {
    clear_state();
    assert_eq!(
      interpret(
        "ClearAll[f]; Attributes[f] = Symbol[\"Listable\"]; Attributes[f]"
      )
      .unwrap(),
      "{Listable}"
    );
  }

  #[test]
  fn set_attributes_invalid_returns_failed() {
    clear_state();
    assert_eq!(interpret("Attributes[f] := {a + b}").unwrap(), "$Failed");
  }

  #[test]
  fn set_attributes_replaces_existing() {
    clear_state();
    assert_eq!(
      interpret("ClearAll[f]; Attributes[f] = {Flat}; Attributes[f] = {Listable}; Attributes[f]")
        .unwrap(),
      "{Listable}"
    );
  }

  #[test]
  fn clear_attributes_list_form() {
    clear_state();
    assert_eq!(
      interpret("ClearAll[f]; SetAttributes[f, Flat]; ClearAttributes[{f}, {Flat}]; Attributes[f]")
        .unwrap(),
      "{}"
    );
  }

  #[test]
  fn set_attributes_list_form() {
    clear_state();
    assert_eq!(
      interpret("ClearAll[f]; SetAttributes[{f}, {Flat}]; Attributes[f]")
        .unwrap(),
      "{Flat}"
    );
  }

  #[test]
  fn locked_prevents_modification() {
    clear_state();
    assert_eq!(
      interpret(
        "ClearAll[lock]; Attributes[lock] = {Flat, Locked}; Attributes[lock]"
      )
      .unwrap(),
      "{Flat, Locked}"
    );
  }

  #[test]
  fn locked_assignment_returns_value() {
    clear_state();
    assert_eq!(
      interpret(
        "ClearAll[lock]; Attributes[lock] = {Flat, Locked}; Attributes[lock] = {}"
      )
      .unwrap(),
      "{}"
    );
  }
}

// HoldAllComplete suppresses UpValues lookup (in addition to holding all
// args and disabling Sequence flattening / Evaluate). A symbol's upvalue
// is normally consulted when the surrounding head sees it, but with
// HoldAllComplete on the head the upvalue stays dormant.
mod hold_all_complete_blocks_upvalues {
  use super::*;

  #[test]
  fn upvalue_normally_fires() {
    clear_state();
    assert_eq!(interpret("ClearAll[g, a]; g[a] ^= 3; g[a]").unwrap(), "3");
  }

  #[test]
  fn upvalue_blocked_when_head_has_hold_all_complete() {
    clear_state();
    assert_eq!(
      interpret(
        "ClearAll[f, a]; SetAttributes[f, HoldAllComplete]; f[a] ^= 3; f[a]"
      )
      .unwrap(),
      "f[a]"
    );
  }

  #[test]
  fn hold_all_complete_also_keeps_sequence_unsplattered() {
    clear_state();
    assert_eq!(
      interpret(
        "ClearAll[f]; SetAttributes[f, HoldAllComplete]; f[Sequence[a, b]]"
      )
      .unwrap(),
      "f[Sequence[a, b]]"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn not_option_q_1() {
    assert_case(r#"NotOptionQ[x]"#, r#"NotOptionQ[x]"#);
  }
  #[test]
  fn not_option_q_2() {
    assert_case(r#"NotOptionQ[x]; NotOptionQ[2]"#, r#"NotOptionQ[2]"#);
  }
  #[test]
  fn not_option_q_3() {
    assert_case(
      r#"NotOptionQ[x]; NotOptionQ[2]; NotOptionQ["abc"]"#,
      r#"NotOptionQ["abc"]"#,
    );
  }
  #[test]
  fn not_option_q_4() {
    assert_case(
      r#"NotOptionQ[x]; NotOptionQ[2]; NotOptionQ["abc"]; NotOptionQ[a -> True]"#,
      r#"NotOptionQ[a -> True]"#,
    );
  }
  #[test]
  fn option_q_1() {
    assert_case(r#"OptionQ[a -> True]"#, r#"True"#);
  }
  #[test]
  fn option_q_2() {
    assert_case(r#"OptionQ[a -> True]; OptionQ[a :> True]"#, r#"True"#);
  }
  #[test]
  fn option_q_3() {
    assert_case(
      r#"OptionQ[a -> True]; OptionQ[a :> True]; OptionQ[{a -> True}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn option_q_4() {
    assert_case(
      r#"OptionQ[a -> True]; OptionQ[a :> True]; OptionQ[{a -> True}]; OptionQ[{a :> True}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn option_q_5() {
    assert_case(
      r#"OptionQ[a -> True]; OptionQ[a :> True]; OptionQ[{a -> True}]; OptionQ[{a :> True}]; OptionQ[{a -> True, {b->1, "c"->2}}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn option_q_6() {
    assert_case(
      r#"OptionQ[a -> True]; OptionQ[a :> True]; OptionQ[{a -> True}]; OptionQ[{a :> True}]; OptionQ[{a -> True, {b->1, "c"->2}}]; OptionQ[{a -> True, {b->1, c}}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn options_1() {
    assert_case(r#"Options[f] = {n -> 2}"#, r#"{n -> 2}"#);
  }
  #[test]
  fn options_2() {
    assert_case(r#"Options[f] = {n -> 2}; Options[f]"#, r#"{n -> 2}"#);
  }
  #[test]
  fn f_1() {
    assert_case(
      r#"Options[f] = {n -> 2}; Options[f]; f[x_, OptionsPattern[f]] := x ^ OptionValue[n]; f[x]"#,
      r#"x ^ 2"#,
    );
  }
  #[test]
  fn f_2() {
    assert_case(
      r#"Options[f] = {n -> 2}; Options[f]; f[x_, OptionsPattern[f]] := x ^ OptionValue[n]; f[x]; f[x, n -> 3]"#,
      r#"x ^ 3"#,
    );
  }
  #[test]
  fn options_3() {
    assert_case(
      r#"Options[MySetting] = {"foo" -> 5, "bar" -> 6}"#,
      r#"{"foo" -> 5, "bar" -> 6}"#,
    );
  }
  #[test]
  fn option_value_1() {
    assert_case(
      r#"Options[MySetting] = {"foo" -> 5, "bar" -> 6}; OptionValue[MySetting, "bar"]"#,
      r#"6"#,
    );
  }
  #[test]
  fn set_options() {
    assert_case(
      r#"SetOptions[Plot]"#,
      r#"{AlignmentPoint -> Center, AspectRatio -> GoldenRatio^(-1), Axes -> True, AxesLabel -> None, AxesOrigin -> Automatic, AxesStyle -> {}, Background -> None, BaselinePosition -> Automatic, BaseStyle -> {}, ClippingStyle -> None, ColorFunction -> Automatic, ColorFunctionScaling -> True, ColorOutput -> Automatic, ContentSelectable -> Automatic, CoordinatesToolOptions -> Automatic, DisplayFunction :> $DisplayFunction, Epilog -> {}, Evaluated -> Automatic, EvaluationMonitor -> None, Exclusions -> Automatic, ExclusionsStyle -> None, Filling -> None, FillingStyle -> Automatic, FormatType :> TraditionalForm, Frame -> False, FrameLabel -> None, FrameStyle -> {}, FrameTicks -> Automatic, FrameTicksStyle -> {}, GridLines -> None, GridLinesStyle -> {}, ImageMargins -> 0., ImagePadding -> All, ImageSize -> Automatic, ImageSizeRaw -> Automatic, IntervalMarkers -> Automatic, IntervalMarkersStyle -> Automatic, LabelingSize -> Automatic, LabelStyle -> {}, MaxRecursion -> Automatic, Mesh -> None, MeshFunctions -> {#1 & }, MeshShading -> None, MeshStyle -> Automatic, Method -> Automatic, PerformanceGoal :> $PerformanceGoal, PlotHighlighting -> Automatic, PlotInteractivity :> $PlotInteractivity, PlotLabel -> None, PlotLabels -> None, PlotLayout -> Automatic, PlotLegends -> None, PlotPoints -> Automatic, PlotRange -> {Full, Automatic}, PlotRangeClipping -> True, PlotRangePadding -> Automatic, PlotRegion -> Automatic, PlotStyle -> Automatic, PlotTheme :> $PlotTheme, PreserveImageOptions -> Automatic, Prolog -> {}, RegionFunction -> (True & ), RotateLabel -> True, ScalingFunctions -> None, TargetUnits -> Automatic, Ticks -> Automatic, TicksStyle -> {}, WorkingPrecision -> MachinePrecision}"#,
    );
  }
  #[test]
  fn attributes_1() {
    assert_case(
      r#"Attributes[Plus]"#,
      r#"{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"#,
    );
  }
  #[test]
  fn attributes_2() {
    assert_case(
      r#"Attributes[Plus]; Attributes["Plus"]"#,
      r#"{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"#,
    );
  }
  #[test]
  fn attributes_3() {
    assert_case(r#"SetAttributes[f, Flat]; Attributes[f]"#, r#"{Flat}"#);
  }
  #[test]
  fn attributes_4() {
    assert_case(
      r#"SetAttributes[f, Flat]; Attributes[f]; ClearAttributes[f, Flat]; Attributes[f]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn attributes_5() {
    assert_case(
      r#"SetAttributes[f, Flat]; Attributes[f]; ClearAttributes[f, Flat]; Attributes[f]; ClearAttributes[{f}, {Flat}]; Attributes[f]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn attributes_6() {
    assert_case(
      r#"Attributes[E]"#,
      r#"{Constant, Protected, ReadProtected}"#,
    );
  }
  #[test]
  fn f_3() {
    assert_case(r#"SetAttributes[f, Flat]; f[a, f[b, c]]"#, r#"f[a, b, c]"#);
  }
  #[test]
  fn f_4() {
    assert_case(
      r#"SetAttributes[f, Flat]; f[a, f[b, c]]; f[a, b, c] /. f[a, b] -> d"#,
      r#"f[d, c]"#,
    );
  }
  #[test]
  fn attributes_7() {
    assert_case(r#"Attributes[Function]"#, r#"{HoldAll, Protected}"#);
  }
  #[test]
  fn f_5() {
    assert_case(
      r#"SetAttributes[f, HoldAllComplete]; f[a] ^= 3; f[a]"#,
      r#"f[a]"#,
    );
  }
  #[test]
  fn f_6() {
    assert_case(
      r#"SetAttributes[f, HoldAllComplete]; f[a] ^= 3; f[a]; f[Sequence[a, b]]"#,
      r#"f[Sequence[a, b]]"#,
    );
  }
  #[test]
  fn attributes_8() {
    assert_case(
      r#"Attributes[Set]"#,
      r#"{HoldFirst, Protected, SequenceHold}"#,
    );
  }
  #[test]
  fn attributes_9() {
    assert_case(r#"Attributes[If]"#, r#"{HoldRest, Protected}"#);
  }
  #[test]
  fn f_7() {
    assert_case(
      r#"SetAttributes[f, Listable]; f[{1, 2, 3}, {4, 5, 6}]"#,
      r#"{f[1, 4], f[2, 5], f[3, 6]}"#,
    );
  }
  #[test]
  fn f_8() {
    assert_case(
      r#"SetAttributes[f, Listable]; f[{1, 2, 3}, {4, 5, 6}]; f[{1, 2, 3}, 4]"#,
      r#"{f[1, 4], f[2, 4], f[3, 4]}"#,
    );
  }
  #[test]
  fn list_literal() {
    assert_case(
      r#"SetAttributes[f, Listable]; f[{1, 2, 3}, {4, 5, 6}]; f[{1, 2, 3}, 4]; {{1, 2}, {3, 4}} + {5, 6}"#,
      r#"{{6, 7}, {9, 10}}"#,
    );
  }
  #[test]
  fn n_1() {
    assert_case(
      r#"N[f[2, 3]]; SetAttributes[f, NHoldAll]; N[f[2, 3]]"#,
      r#"f[2, 3]"#,
    );
  }
  #[test]
  fn attributes_10() {
    assert_case(
      r#"Attributes[Sqrt]"#,
      r#"{Listable, NumericFunction, Protected}"#,
    );
  }
  #[test]
  fn numeric_q_1() {
    assert_case(r#"Attributes[Sqrt]; NumericQ[Sqrt[1]]"#, r#"True"#);
  }
  #[test]
  fn numeric_q_2() {
    assert_case(
      r#"Attributes[Sqrt]; NumericQ[Sqrt[1]]; NumericQ[a]=True; NumericQ[Sqrt[a]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn numeric_q_3() {
    assert_case(
      r#"Attributes[Sqrt]; NumericQ[Sqrt[1]]; NumericQ[a]=True; NumericQ[Sqrt[a]]; NumericQ[a]=False; NumericQ[Sqrt[a]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn greater_1() {
    assert_case(
      r#"a /. f[x_:0, u_] -> {u}; SetAttributes[f, OneIdentity]; a /. f[x_:0, u_] -> {u}"#,
      r#"{a}"#,
    );
  }
  #[test]
  fn greater_2() {
    assert_case(
      r#"a /. f[x_:0, u_] -> {u}; SetAttributes[f, OneIdentity]; a /. f[x_:0, u_] -> {u}; a /. f[u_] -> {u}"#,
      r#"a"#,
    );
  }
  #[test]
  fn f_9() {
    assert_case(
      r#"a /. f[x_:0, u_] -> {u}; SetAttributes[f, OneIdentity]; a /. f[x_:0, u_] -> {u}; a /. f[u_] -> {u}; f[a]"#,
      r#"f[a]"#,
    );
  }
  #[test]
  fn f_10() {
    assert_case(
      r#"SetAttributes[f, Orderless]; f[c, a, b, a + b, 3, 1.0]"#,
      r#"f[1., 3, a, b, a + b, c]"#,
    );
  }
  #[test]
  fn f_11() {
    assert_case(
      r#"SetAttributes[f, Orderless]; f[c, a, b, a + b, 3, 1.0]; f[a, b] == f[b, a]"#,
      r#"True"#,
    );
  }
  #[test]
  fn f_12() {
    assert_case(
      r#"SetAttributes[f, Orderless]; f[c, a, b, a + b, 3, 1.0]; f[a, b] == f[b, a]; SetAttributes[f, Flat]; f[a, b, c] /. f[a, c] -> d"#,
      r#"f[b, d]"#,
    );
  }
  #[test]
  fn f_13() {
    assert_case(
      r#"f[Sequence[a, b]]; SetAttributes[f, SequenceHold]; f[Sequence[a, b]]"#,
      r#"f[Sequence[a, b]]"#,
    );
  }
  #[test]
  fn plus() {
    assert_case(
      r#"f[Sequence[a, b]]; SetAttributes[f, SequenceHold]; f[Sequence[a, b]]; s = Sequence[a, b]; s; Plus[s]"#,
      r#"a + b"#,
    );
  }
  #[test]
  fn attributes_11() {
    assert_case(r#"SetAttributes[f, Flat]; Attributes[f]"#, r#"{Flat}"#);
  }
  #[test]
  fn attributes_12() {
    assert_case(
      r#"SetAttributes[f, Flat]; Attributes[f]; SetAttributes[{f, g}, {Flat, Orderless}]; Attributes[g]"#,
      r#"{Flat, Orderless}"#,
    );
  }
  #[test]
  fn attributes_13() {
    assert_case(r#"Attributes[Hold]"#, r#"{HoldAll, Protected}"#);
  }
  #[test]
  fn attributes_14() {
    assert_case(
      r#"Attributes[HoldComplete]"#,
      r#"{HoldAllComplete, Protected}"#,
    );
  }
  #[test]
  fn attributes_15() {
    assert_case(
      r#"HoldForm[1 + 2 + 3]; Attributes[HoldForm]"#,
      r#"{HoldAll, Protected}"#,
    );
  }
  #[test]
  fn f_14() {
    assert_case(r#"SetAttributes[f, HoldAll]; f[1 + 2]"#, r#"f[1 + 2]"#);
  }
  #[test]
  fn f_15() {
    assert_case(
      r#"SetAttributes[f, HoldAll]; f[1 + 2]; f[Evaluate[1 + 2]]"#,
      r#"f[3]"#,
    );
  }
  #[test]
  fn hold() {
    assert_case(
      r#"SetAttributes[f, HoldAll]; f[1 + 2]; f[Evaluate[1 + 2]]; Hold[Evaluate[1 + 2]]"#,
      r#"Hold[3]"#,
    );
  }
  #[test]
  fn hold_complete() {
    assert_case(
      r#"SetAttributes[f, HoldAll]; f[1 + 2]; f[Evaluate[1 + 2]]; Hold[Evaluate[1 + 2]]; HoldComplete[Evaluate[1 + 2]]"#,
      r#"HoldComplete[Evaluate[1 + 2]]"#,
    );
  }
  #[test]
  fn evaluate() {
    // Multi-arg `Evaluate[a, b]` returns `Sequence[a, b]`, which splices
    // into surrounding hold contexts and CompoundExpression. The trailing
    // `Evaluate[Sequence[1, 2]]` becomes `Sequence[1, 2]` and the
    // outer `;`-chain (CompoundExpression-style) keeps just the last
    // spliced element, matching wolframscript's `2`.
    assert_case(
      r#"SetAttributes[f, HoldAll]; f[1 + 2]; f[Evaluate[1 + 2]]; Hold[Evaluate[1 + 2]]; HoldComplete[Evaluate[1 + 2]]; Evaluate[Sequence[1, 2]]"#,
      r#"2"#,
    );
  }
  #[test]
  fn attributes_16() {
    assert_case(
      r#"Sqrt[Unevaluated[x]]; Length[Unevaluated[1+2+3+4]]; Attributes[Unevaluated]"#,
      r#"{HoldAllComplete, Protected}"#,
    );
  }
  #[test]
  fn f_16() {
    assert_case(
      r#"Sqrt[Unevaluated[x]]; Length[Unevaluated[1+2+3+4]]; Attributes[Unevaluated]; f[Unevaluated[x]]"#,
      r#"f[Unevaluated[x]]"#,
    );
  }
  #[test]
  fn f_17() {
    assert_case(
      r#"Sqrt[Unevaluated[x]]; Length[Unevaluated[1+2+3+4]]; Attributes[Unevaluated]; f[Unevaluated[x]]; Attributes[f] = {Flat}; f[a, Unevaluated[f[b, c]]]"#,
      r#"f[a, Unevaluated[b], Unevaluated[c]]"#,
    );
  }
  #[test]
  fn g_1() {
    assert_case(
      r#"Sqrt[Unevaluated[x]]; Length[Unevaluated[1+2+3+4]]; Attributes[Unevaluated]; f[Unevaluated[x]]; Attributes[f] = {Flat}; f[a, Unevaluated[f[b, c]]]; g[a, Sequence[Unevaluated[b], Unevaluated[c]]]"#,
      r#"g[a, Unevaluated[b], Unevaluated[c]]"#,
    );
  }
  #[test]
  fn g_2() {
    assert_case(
      r#"Sqrt[Unevaluated[x]]; Length[Unevaluated[1+2+3+4]]; Attributes[Unevaluated]; f[Unevaluated[x]]; Attributes[f] = {Flat}; f[a, Unevaluated[f[b, c]]]; g[a, Sequence[Unevaluated[b], Unevaluated[c]]]; g[Unevaluated[Sequence[a, b, c]]]"#,
      r#"g[Unevaluated[Sequence[a, b, c]]]"#,
    );
  }
  #[test]
  fn attributes_17() {
    assert_case(
      r#"f[x, Sequence[a, b], y]; Attributes[Set]"#,
      r#"{HoldFirst, Protected, SequenceHold}"#,
    );
  }
  #[test]
  fn n_2() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]; N[a, 20] = 11; N[a + b, 20]; N[f[a, b]]; SetAttributes[f, NHoldAll]; N[f[a, b]]"#,
      r#"f[a, b]"#,
    );
  }
  #[test]
  fn n_3() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]; N[a, 20] = 11; N[a + b, 20]; N[f[a, b]]; SetAttributes[f, NHoldAll]; N[f[a, b]]; N[c, p_?(#>10&)] := p; N[c, 3]"#,
      r#"c"#,
    );
  }
  #[test]
  fn n_4() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]; N[a, 20] = 11; N[a + b, 20]; N[f[a, b]]; SetAttributes[f, NHoldAll]; N[f[a, b]]; N[c, p_?(#>10&)] := p; N[c, 3]; N[c, 11]"#,
      r#"11."#,
    );
  }
  #[test]
  fn sort_1() {
    assert_case(
      r#"Sort[{4, 1.0, a, 3+I}]; Sort[{items___, item_, OptionsPattern[], item_symbol, item_?test}, PatternsOrderedQ]"#,
      r#"{items___, item_, OptionsPattern[], item_symbol, (item_)?test}"#,
    );
  }
  #[test]
  fn sort_2() {
    assert_case(
      r#"Sort[{4, 1.0, a, 3+I}]; Sort[{items___, item_, OptionsPattern[], item_symbol, item_?test}, PatternsOrderedQ]; Sort[{a, b/;t}, PatternsOrderedQ]"#,
      r#"{a, b /; t}"#,
    );
  }
  #[test]
  fn sort_3() {
    assert_case(
      r#"Sort[{4, 1.0, a, 3+I}]; Sort[{items___, item_, OptionsPattern[], item_symbol, item_?test}, PatternsOrderedQ]; Sort[{a, b/;t}, PatternsOrderedQ]; Sort[{2+c_, 1+b__}, PatternsOrderedQ]"#,
      r#"{2 + (c_), 1 + (b__)}"#,
    );
  }
  #[test]
  fn sort_4() {
    assert_case(
      r#"Sort[{4, 1.0, a, 3+I}]; Sort[{items___, item_, OptionsPattern[], item_symbol, item_?test}, PatternsOrderedQ]; Sort[{a, b/;t}, PatternsOrderedQ]; Sort[{2+c_, 1+b__}, PatternsOrderedQ]; Sort[{x_ + n_*y_, x_ + y_}, PatternsOrderedQ]"#,
      r#"{(x_) + (n_)*(y_), (x_) + (y_)}"#,
    );
  }
  #[test]
  fn attributes_18() {
    assert_case(
      r#"Attributes[SetDelayed]"#,
      r#"{HoldAll, Protected, SequenceHold}"#,
    );
  }
  #[test]
  fn set_1() {
    assert_case(r#"Attributes[SetDelayed]; a = 1"#, r#"1"#);
  }
  #[test]
  fn symbol_literal_1() {
    assert_case(r#"Attributes[SetDelayed]; a = 1; x := a; x"#, r#"1"#);
  }
  #[test]
  fn set_2() {
    assert_case(r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2"#, r#"2"#);
  }
  #[test]
  fn symbol_literal_2() {
    assert_case(
      r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2; x"#,
      r#"2"#,
    );
  }
  #[test]
  fn f_18() {
    assert_case(
      r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2; x; f[x_] := p[x] /; x>0; f[3]"#,
      r#"p[3]"#,
    );
  }
  #[test]
  fn f_19() {
    assert_case(
      r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2; x; f[x_] := p[x] /; x>0; f[3]; f[-3]"#,
      r#"f[-3]"#,
    );
  }
  #[test]
  fn attributes_19() {
    assert_case(
      r#"ConditionalExpression[a, False]; Attributes[Undefined]"#,
      r#"{Protected}"#,
    );
  }
  #[test]
  fn f_20() {
    assert_case(
      r#"a + b + c /. a + b -> t; a + 2 + b + c + x * y /. n_Integer + s__Symbol + rest_ -> {n, s, rest}; f[a, b, c, d] /. f[first_, rest___] -> {first, {rest}}; f[4] /. f[x_?(# > 0&)] -> x ^ 2; f[4] /. f[x_] /; x > 0 -> x ^ 2; f[a, b, c, d] /. f[start__, end__] -> {{start}, {end}}; f[a] /. f[x_, y_:3] -> {x, y}; f[y, a->3] /. f[x_, OptionsPattern[{a->2, b->5}]] -> {x, OptionValue[a], OptionValue[b]}"#,
      r#"{y, 3, 5}"#,
    );
  }
  #[test]
  fn attributes_20() {
    assert_case(
      r#"Attributes[RuleDelayed]"#,
      r#"{HoldRest, Protected, SequenceHold}"#,
    );
  }
  #[test]
  fn f_21() {
    assert_case(
      r#"f[x_, OptionsPattern[{n->2}]] := x ^ OptionValue[n]; f[x]"#,
      r#"x ^ 2"#,
    );
  }
  #[test]
  fn f_22() {
    assert_case(
      r#"f[x_, OptionsPattern[{n->2}]] := x ^ OptionValue[n]; f[x]; f[x, n->3]"#,
      r#"x ^ 3"#,
    );
  }
  #[test]
  fn greater_3() {
    assert_case(
      r#"f[x_, OptionsPattern[{n->2}]] := x ^ OptionValue[n]; f[x]; f[x, n->3]; e = f[x, n:>a]"#,
      r#"x ^ a"#,
    );
  }
  #[test]
  fn symbol_literal_3() {
    assert_case(
      r#"f[x_, OptionsPattern[{n->2}]] := x ^ OptionValue[n]; f[x]; f[x, n->3]; e = f[x, n:>a]; a = 5; e"#,
      r#"x ^ 5"#,
    );
  }
  #[test]
  fn f_23() {
    assert_case(
      r#"f[x_, OptionsPattern[{n->2}]] := x ^ OptionValue[n]; f[x]; f[x, n->3]; e = f[x, n:>a]; a = 5; e; f[x, {{{n->4}}}]"#,
      r#"x ^ 4"#,
    );
  }
  #[test]
  fn f2() {
    assert_case(
      r#"f1[y]; Options[f2]:={s->12};f2[x_,opt:OptionsPattern[]]:=x^OptionValue[s]; f2[y]"#,
      r#"y ^ 12"#,
    );
  }
  #[test]
  fn f3() {
    assert_case(
      r#"f1[y]; f2[y]; Options[f3]:={a->12};f3[x_,opt:OptionsPattern[{a:>4}]]:=x^OptionValue[a]; f3[y]"#,
      r#"y ^ 4"#,
    );
  }
  #[test]
  fn f4() {
    assert_case(
      r#"f1[y]; f2[y]; f3[y]; Options[f4]:={a->12};f4[x_,OptionsPattern[{a:>4}]]:=x^OptionValue[a]; f4[y]"#,
      r#"y ^ 4"#,
    );
  }
  #[test]
  fn option_value_2() {
    assert_case(
      r#"f1[y]; f2[y]; f3[y]; f4[y]; Options[F]:={a->89,b->37}; OptionValue[F, a]"#,
      r#"89"#,
    );
  }
  #[test]
  fn f_24() {
    assert_case(
      r#"f[x_, OptionsPattern[f]] := x ^ OptionValue["m"];Options[f] = {"m" -> 7};f[x]"#,
      r#"x ^ 7"#,
    );
  }
  #[test]
  fn greater_4() {
    assert_case(
      r#"f[x_, OptionsPattern[f]] := x ^ OptionValue["m"];Options[f] = {"m" -> 7};f[x]; f /: Options[f] = {a -> b}"#,
      r#"{a -> b}"#,
    );
  }
  #[test]
  fn options_4() {
    assert_case(
      r#"f[x_, OptionsPattern[f]] := x ^ OptionValue["m"];Options[f] = {"m" -> 7};f[x]; f /: Options[f] = {a -> b}; Options[f]"#,
      r#"{a -> b}"#,
    );
  }
  #[test]
  fn set_attributes_1() {
    assert_case(r#"SetAttributes[F, OneIdentity]"#, r#"Null"#);
  }
  #[test]
  fn set_attributes_2() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_attributes_3() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_attributes_4() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_attributes_5() {
    assert_case(r#"SetAttributes[F, OneIdentity]"#, r#"Null"#);
  }
  #[test]
  fn set_attributes_6() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_attributes_7() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_attributes_8() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_attributes_9() {
    assert_case(r#"SetAttributes[F, OneIdentity]"#, r#"Null"#);
  }
  #[test]
  fn set_attributes_10() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_attributes_11() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_attributes_12() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn attributes_21() {
    assert_case(
      r#"Attributes[Pi]"#,
      r#"{Constant, Protected, ReadProtected}"#,
    );
  }
  #[test]
  fn attributes_22() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]"#,
      r#"{Constant, ReadProtected}"#,
    );
  }
}
