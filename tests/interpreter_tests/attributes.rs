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

  // Regression tests for https://github.com/ad-si/Woxi/issues/396 —
  // `C` is the generated-parameter symbol of DSolve / RSolve / Reduce /
  // Solve (`C[1]`, `C[2]`, …) and therefore a Protected built-in.
  #[test]
  fn c_has_builtin_attributes() {
    clear_state();
    assert_eq!(
      interpret("Attributes[C]").unwrap(),
      "{NHoldAll, Protected, ReadProtected}"
    );
  }

  #[test]
  fn set_c_is_rejected() {
    clear_state();
    let result = interpret_with_stdout("C = 12").unwrap();
    assert_eq!(result.result, "12");
    assert!(
      result.warnings[0].contains("Set::wrsym: Symbol C is Protected."),
      "unexpected warnings: {:?}",
      result.warnings
    );
    // The assignment must not have taken effect.
    assert_eq!(interpret("C").unwrap(), "C");
  }

  #[test]
  fn set_delayed_c_is_rejected() {
    clear_state();
    let result = interpret_with_stdout("C := 12").unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result.warnings[0].contains("SetDelayed::wrsym: Symbol C is Protected."),
      "unexpected warnings: {:?}",
      result.warnings
    );
  }

  #[test]
  fn unprotect_c_allows_assignment() {
    clear_state();
    assert_eq!(interpret("Unprotect[C]; C = 12; C").unwrap(), "12");
  }

  #[test]
  fn c_stays_inert_as_generated_parameter() {
    clear_state();
    let result = interpret_with_stdout("C[1] + C[2]").unwrap();
    assert_eq!(result.result, "C[1] + C[2]");
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
  }

  #[test]
  fn set_delayed_protected_constant_returns_failed() {
    // `Pi := 12` takes the same OwnValue path as `C := 12`; wolframscript
    // returns `$Failed` from a rejected SetDelayed (Set returns its RHS).
    clear_state();
    let result = interpret_with_stdout("Pi := 12").unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result.warnings[0].contains("SetDelayed::wrsym: Symbol Pi is Protected."),
      "unexpected warnings: {:?}",
      result.warnings
    );
  }

  #[test]
  fn set_downvalue_on_protected_builtin_is_rejected() {
    clear_state();
    let result = interpret_with_stdout("Sin[1] = 5").unwrap();
    assert_eq!(result.result, "5");
    assert!(
      result.warnings[0]
        .contains("Set::write: Tag Sin in Sin[1] is Protected."),
      "unexpected warnings: {:?}",
      result.warnings
    );
    assert_eq!(interpret("Sin[1] // N").unwrap(), "0.8414709848078965");
  }

  #[test]
  fn set_downvalue_on_user_protected_symbol_is_rejected() {
    clear_state();
    let result = interpret_with_stdout("Protect[bar]; bar[1] = 3").unwrap();
    assert_eq!(result.result, "3");
    assert!(
      result.warnings[0]
        .contains("Set::write: Tag bar in bar[1] is Protected."),
      "unexpected warnings: {:?}",
      result.warnings
    );
  }

  #[test]
  fn set_delayed_downvalue_on_user_protected_symbol_is_rejected() {
    clear_state();
    let result = interpret_with_stdout("Protect[bar]; bar[2] := 4").unwrap();
    assert_eq!(result.result, "$Failed");
    assert!(
      result.warnings[0]
        .contains("SetDelayed::write: Tag bar in bar[2] is Protected."),
      "unexpected warnings: {:?}",
      result.warnings
    );
    assert_eq!(interpret("DownValues[bar]").unwrap(), "{}");
  }

  #[test]
  fn set_on_times_lhs_is_rejected_not_fatal() {
    // `2 x = 5` parses as `Set[Times[2, x], 5]` since `=` binds looser than
    // implicit multiplication. wolframscript rejects this with a message
    // (not a fatal error) and returns the right-hand side, leaving `x`
    // unassigned. Regression for a Times/Plus/Power-headed Set target
    // aborting the whole interpretation instead of just failing softly.
    clear_state();
    let result = interpret_with_stdout("2 x = 5").unwrap();
    assert_eq!(result.result, "5");
    assert!(
      result.warnings[0].contains("Set::write: Tag Times in 2*x is Protected."),
      "unexpected warnings: {:?}",
      result.warnings
    );
    assert_eq!(interpret("x").unwrap(), "x");
  }

  #[test]
  fn set_after_null_statement_without_semicolon_assigns_target() {
    // A `While`/`For`/etc. statement followed by another statement with no
    // separating `;` parses as implicit multiplication: `While[...] y = v`
    // becomes `Set[Times[While[...], y], v]`. wolframscript still runs the
    // `While` for its side effects and, since it evaluates to `Null`
    // (contributing no coefficient), assigns `v` to `y` directly rather
    // than rejecting the Times-headed target. This is the exact shape a
    // missing `;` between two Module statements produces.
    clear_state();
    assert_eq!(
      interpret(
        "Module[{n = 0, total}, While[n < 3, n = n + 1] total = n * 10; total]"
      )
      .unwrap(),
      "30"
    );
  }

  #[test]
  fn set_after_null_statement_runs_side_effects_of_both_factors() {
    // Same shape as above, but confirms the `While` loop's side effects
    // (not just the final assignment) actually ran — a naive fix that
    // only patched the error message without evaluating the held Times
    // factors would leave `acc` empty.
    clear_state();
    assert_eq!(
      interpret(concat!(
        "Module[{acc = {}, i = 1, len},",
        "  While[i <= 3, acc = Append[acc, i]; i++] len = Length[acc];",
        "  {acc, len}]"
      ))
      .unwrap(),
      "{{1, 2, 3}, 3}"
    );
  }

  #[test]
  fn clear_protected_symbol_keeps_definitions() {
    clear_state();
    let result =
      interpret_with_stdout("foo = 1; Protect[foo]; Clear[foo]; foo").unwrap();
    assert_eq!(result.result, "1");
    assert!(
      result
        .warnings
        .iter()
        .any(|w| w.contains("Clear::wrsym: Symbol foo is Protected.")),
      "unexpected warnings: {:?}",
      result.warnings
    );
  }

  #[test]
  fn clear_all_protected_builtin_keeps_attributes() {
    clear_state();
    let result =
      interpret_with_stdout("ClearAll[Sin]; Attributes[Sin]").unwrap();
    assert_eq!(result.result, "{Listable, NumericFunction, Protected}");
    assert!(
      result
        .warnings
        .iter()
        .any(|w| w.contains("ClearAll::wrsym: Symbol Sin is Protected.")),
      "unexpected warnings: {:?}",
      result.warnings
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

  // `Sin[x_] := y` attempts to install a DownValue on a
  // built-in Protected symbol. wolframscript emits
  // `SetDelayed::write` and returns `$Failed`. Regression for
  // mathics 1-Manual `Dice[a___] + Dice[b___] := …` row.
  #[test]
  fn set_delayed_on_protected_builtin_fails() {
    clear_state();
    // The CLI prints `$Failed`; the underlying message goes to
    // stdout. Just check the return value here.
    assert_eq!(interpret("Sin[x_] := y").unwrap(), "$Failed");
  }

  #[test]
  fn set_delayed_on_protected_binary_op_fails() {
    clear_state();
    assert_eq!(
      interpret("Dice[a___] + Dice[b___] := Dice[Sequence @@ {a, b}]").unwrap(),
      "$Failed"
    );
  }

  // NValues / Messages / Format / Default / Options are
  // "redirected per-symbol" definitions: wolframscript permits
  // them even though the head is Protected.
  #[test]
  fn n_value_assignment_allowed_despite_protected() {
    clear_state();
    assert_eq!(interpret("N[c, p_?(#>10&)] := p; N[c, 11]").unwrap(), "11");
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
  // The built-in option lists of the core symbolic and numeric functions, each
  // transcribed from wolframscript. Options carrying a `$…` global stay delayed.
  #[test]
  fn builtin_options_of_computation_heads() {
    assert_case(r#"Options[D]"#, r#"{NonConstants -> {}}"#);
    assert_case(r#"Options[Root]"#, r#"{ExactRootIsolation -> False}"#);
    assert_case(
      r#"Options[Total]"#,
      r#"{AllowedHeads -> Automatic, Method -> Automatic}"#,
    );
    assert_case(
      r#"Options[StringSplit]"#,
      r#"{IgnoreCase -> False, MetaCharacters -> None}"#,
    );
    assert_case(
      r#"Options[StringCases]"#,
      r#"{IgnoreCase -> False, MetaCharacters -> None, Overlaps -> False}"#,
    );
    assert_case(
      r#"Options[Refine]"#,
      r#"{Assumptions :> $Assumptions, TimeConstraint -> 30}"#,
    );
    assert_case(
      r#"Options[Nearest]"#,
      r#"{DistanceFunction -> Automatic, Method -> Automatic, WorkingPrecision -> Automatic}"#,
    );
    assert_case(
      r#"Options[Interpolation]"#,
      r#"{InterpolationOrder -> 3, Method -> Automatic, PeriodicInterpolation -> False}"#,
    );
    assert_case(
      r#"Options[Factor]"#,
      r#"{Extension -> None, GaussianIntegers -> False, Modulus -> 0, Trig -> False}"#,
    );
    assert_case(
      r#"Options[Series]"#,
      r#"{Analytic -> True, Assumptions :> $Assumptions, SeriesTermGoal -> Automatic}"#,
    );
    assert_case(
      r#"Options[Limit]"#,
      r#"{Analytic -> False, Assumptions :> $Assumptions, Direction -> Reals, GenerateConditions -> Automatic, Method -> Automatic, PerformanceGoal :> $PerformanceGoal}"#,
    );
    assert_case(
      r#"Options[Sum]"#,
      r#"{Assumptions :> $Assumptions, GenerateConditions -> False, GeneratedParameters -> None, Method -> Automatic, Regularization -> None, VerifyConvergence -> True}"#,
    );
    assert_case(
      r#"Options[Solve]"#,
      r#"{Assumptions :> $Assumptions, Cubics -> Automatic, GeneratedParameters -> C, InverseFunctions -> Automatic, MaxExtraConditions -> 0, MaxRoots -> Infinity, Method -> Automatic, Modulus -> 0, Quartics -> Automatic, VerifySolutions -> Automatic, WorkingPrecision -> Infinity}"#,
    );
    assert_case(
      r#"Options[FindRoot]"#,
      r#"{AccuracyGoal -> Automatic, Compiled -> Automatic, DampingFactor -> 1, Evaluated -> True, EvaluationMonitor -> None, Jacobian -> Automatic, MaxIterations -> 100, Method -> Automatic, PrecisionGoal -> Automatic, StepMonitor -> None, WorkingPrecision -> MachinePrecision}"#,
    );
    // Simplify and FullSimplify differ only in their time budget.
    assert_case(
      r#"Options[Simplify]"#,
      r#"{Assumptions :> $Assumptions, ComplexityFunction -> Automatic, ExcludedForms -> {}, TimeConstraint -> 300, TransformationFunctions -> Automatic, Trig -> True}"#,
    );
    assert_case(
      r#"Options[FullSimplify]"#,
      r#"{Assumptions :> $Assumptions, ComplexityFunction -> Automatic, ExcludedForms -> {}, TimeConstraint -> Infinity, TransformationFunctions -> Automatic, Trig -> True}"#,
    );
    // A symbol with no options still reports none.
    assert_case(r#"Options[Sin]"#, r#"{}"#);
  }

  // SetOptions[f, name -> value] replaces that entry in Options[f], keeping the
  // original order, and returns the updated list.
  #[test]
  fn set_options_updates_and_returns_the_list() {
    assert_case(
      r#"SetOptions[Total, Method -> "X"]"#,
      r#"{AllowedHeads -> Automatic, Method -> X}"#,
    );
    // The change sticks.
    assert_case(
      r#"SetOptions[Total, Method -> "X"]; Options[Total]"#,
      r#"{AllowedHeads -> Automatic, Method -> X}"#,
    );
    // And OptionValue reads it back.
    assert_case(
      r#"SetOptions[Total, Method -> 7]; OptionValue[Total, Method]"#,
      r#"7"#,
    );
    // Several options at once, in any order.
    assert_case(
      r#"SetOptions[Total, Method -> 1, AllowedHeads -> 2]"#,
      r#"{AllowedHeads -> 2, Method -> 1}"#,
    );
    // Options may be gathered into a list.
    assert_case(
      r#"SetOptions[Total, {Method -> "Y"}]"#,
      r#"{AllowedHeads -> Automatic, Method -> Y}"#,
    );
    // A string names the same option as the symbol, and is stored as one.
    assert_case(
      r#"SetOptions[StringSplit, "IgnoreCase" -> True]"#,
      r#"{IgnoreCase -> True, MetaCharacters -> None}"#,
    );
    // A delayed rule stays delayed.
    assert_case(
      r#"SetOptions[Total, Method :> qq]"#,
      r#"{AllowedHeads -> Automatic, Method :> qq}"#,
    );
    // Untouched entries keep their delayed defaults.
    assert_case(
      r#"SetOptions[Simplify, TimeConstraint -> 5, ComplexityFunction -> f]"#,
      r#"{Assumptions :> $Assumptions, ComplexityFunction -> f, ExcludedForms -> {}, TimeConstraint -> 5, TransformationFunctions -> Automatic, Trig -> True}"#,
    );
    // With no options to set, the current list comes back unchanged.
    assert_case(
      r#"SetOptions[Total]"#,
      r#"{AllowedHeads -> Automatic, Method -> Automatic}"#,
    );
  }

  // A name that is not already an option of f refuses the whole call, so
  // nothing is changed even when other names in the same call are valid.
  #[test]
  fn set_options_rejects_unknown_names_atomically() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout(
      r#"SetOptions[Total, Method -> 5]; SetOptions[Total, Bogus -> 1, Method -> 9]"#,
    )
    .unwrap();
    assert_eq!(r.result, "SetOptions[Total, Bogus -> 1, Method -> 9]");
    assert!(
      r.warnings.iter().any(
        |w| w == "SetOptions::optnf: Bogus is not a known option for Total."
      ),
      "expected optnf message, got {:?}",
      r.warnings
    );
    // Method kept the value the earlier call gave it.
    assert_case(
      r#"SetOptions[Total, Method -> 5]; SetOptions[Total, Bogus -> 1, Method -> 9]; Options[Total]"#,
      r#"{AllowedHeads -> Automatic, Method -> 5}"#,
    );
    // A symbol with no options at all has no known names either.
    let r = interpret_with_stdout(r#"SetOptions[foo, a -> 1]"#).unwrap();
    assert_eq!(r.result, "SetOptions[foo, a -> 1]");
    assert!(
      r.warnings
        .iter()
        .any(|w| w == "SetOptions::optnf: a is not a known option for foo."),
      "expected optnf message, got {:?}",
      r.warnings
    );
  }

  #[test]
  fn set_options_rejects_bad_arguments() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout(r#"SetOptions[5, Method -> 1]"#).unwrap();
    assert_eq!(r.result, "SetOptions[5, Method -> 1]");
    assert!(
      r.warnings.iter().any(
        |w| w == "SetOptions::sstm: Argument 5 is not a symbol or a stream."
      ),
      "expected sstm message, got {:?}",
      r.warnings
    );
    let r = interpret_with_stdout(r#"SetOptions[Total, Method]"#).unwrap();
    assert_eq!(r.result, "SetOptions[Total, Method]");
    assert!(
      r.warnings.iter().any(
        |w| w == "SetOptions::rep: Method is not a valid replacement rule."
      ),
      "expected rep message, got {:?}",
      r.warnings
    );
  }

  // OptionValue[f, name] reads the option out of Options[f] — the built-in
  // defaults as well as a user-set list.
  #[test]
  fn option_value_reads_builtin_defaults() {
    assert_case(r#"OptionValue[Plot, Axes]"#, r#"True"#);
    assert_case(r#"OptionValue[Plot, Frame]"#, r#"False"#);
    assert_case(r#"OptionValue[Plot, PlotRange]"#, r#"{Full, Automatic}"#);
    assert_case(r#"OptionValue[Integrate, Assumptions]"#, r#"True"#);
    assert_case(r#"OptionValue[Position, Heads]"#, r#"True"#);
    assert_case(r#"OptionValue[Replace, Heads]"#, r#"False"#);
    assert_case(r#"OptionValue[ExpandAll, Modulus]"#, r#"0"#);
    // A string names the same option as the symbol does.
    assert_case(r#"OptionValue[Plot, "Axes"]"#, r#"True"#);
  }

  // A list of names gives a list of values, mixing symbols and strings.
  #[test]
  fn option_value_list_of_names() {
    assert_case(r#"OptionValue[Plot, {Axes, Frame}]"#, r#"{True, False}"#);
    assert_case(r#"OptionValue[Plot, {Axes, "Frame"}]"#, r#"{True, False}"#);
    assert_case(
      r#"OptionValue[Integrate, {Assumptions, GenerateConditions}]"#,
      r#"{True, Automatic}"#,
    );
    assert_case(r#"OptionValue[Plot, {}]"#, r#"{}"#);
  }

  // An explicit rule list overrides the defaults name by name, and an empty one
  // falls back to them entirely.
  #[test]
  fn option_value_explicit_rules_override_defaults() {
    assert_case(r#"OptionValue[Plot, {Frame -> True}, Frame]"#, r#"True"#);
    assert_case(r#"OptionValue[Plot, {}, Axes]"#, r#"True"#);
    assert_case(
      r#"OptionValue[Plot, {Axes -> 7}, {Axes, Frame}]"#,
      r#"{7, False}"#,
    );
    // A bare rule stands in for a one-element list.
    assert_case(r#"OptionValue[Plot, Frame -> True, Frame]"#, r#"True"#);
    assert_case(r#"OptionValue[Plot, Frame :> True, Frame]"#, r#"True"#);
  }

  // A rule list in place of the head is itself the option list.
  #[test]
  fn option_value_rules_without_a_head() {
    assert_case(r#"OptionValue[{Frame -> True}, Frame]"#, r#"True"#);
    assert_case(
      r#"OptionValue[{Frame -> True, Axes -> False}, {Frame, Axes}]"#,
      r#"{True, False}"#,
    );
  }

  // A fourth argument wraps each value in that head; a missing option is
  // returned bare.
  #[test]
  fn option_value_wrapper_head() {
    assert_case(
      r#"OptionValue[Plot, {}, Axes, Automatic]"#,
      r#"Automatic[True]"#,
    );
    assert_case(
      r#"OptionValue[Plot, {Frame -> True}, Frame, Hold]"#,
      r#"Hold[True]"#,
    );
    assert_case(
      r#"OptionValue[Plot, {}, {Axes, Frame}, Hold]"#,
      r#"{Hold[True], Hold[False]}"#,
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
  // A mathematical constant is numeric whichever way it reached the
  // product. Regression: `Pi 2` leaves `Pi` as a plain symbol (while `2 Pi`
  // keeps the constant node), so `NumericQ[Pi 2]` answered False — which in
  // turn stopped `Re[2 E^(Pi I/25)]` from reducing.
  #[test]
  fn numeric_q_constants_in_a_product() {
    assert_case(r#"NumericQ[Pi 2]"#, r#"True"#);
    assert_case(r#"NumericQ[2 Pi]"#, r#"True"#);
    assert_case(r#"NumericQ[E 2]"#, r#"True"#);
    assert_case(r#"NumericQ[Degree 2]"#, r#"True"#);
    assert_case(r#"NumericQ[Pi I]"#, r#"True"#);
    assert_case(r#"NumericQ[E^(Pi I/25)]"#, r#"True"#);
    // A symbol with no numeric value is still not numeric, and Infinity
    // is not numeric in the Wolfram Language.
    assert_case(r#"NumericQ[Pi x]"#, r#"False"#);
    assert_case(r#"NumericQ[Infinity]"#, r#"False"#);
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
  fn sequence_hold_contrast() {
    // With SequenceHold, Sequence args are NOT spliced; without it they are.
    assert_case(
      r#"SetAttributes[g, SequenceHold]; {g[Sequence[1, 2]], h[Sequence[1, 2]]}"#,
      r#"{g[Sequence[1, 2]], h[1, 2]}"#,
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
  fn attributes_hold_complete_form() {
    assert_case(
      r#"Attributes[HoldCompleteForm]"#,
      r#"{HoldAllComplete, Protected}"#,
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
  fn hold_complete_form() {
    // HoldCompleteForm is HoldForm's HoldAllComplete sibling: it keeps its
    // argument unevaluated for display, and — unlike HoldForm — an inner
    // `Evaluate` does not break through the hold.
    assert_case(r#"HoldCompleteForm[2 + 3]"#, r#"HoldCompleteForm[2 + 3]"#);
    assert_case(
      r#"HoldCompleteForm[Evaluate[1 + 2]]"#,
      r#"HoldCompleteForm[Evaluate[1 + 2]]"#,
    );
    assert_case(r#"HoldForm[Evaluate[1 + 2]]"#, r#"HoldForm[3]"#);
    // HoldAllComplete also carries SequenceHold, so a `Sequence` argument
    // keeps its wrapper instead of splicing.
    assert_case(
      r#"HoldCompleteForm[Sequence[1, 2]]"#,
      r#"HoldCompleteForm[Sequence[1, 2]]"#,
    );
    // …and `Unevaluated` is not stripped inside it.
    assert_case(
      r#"HoldCompleteForm[Unevaluated[1 + 1]]"#,
      r#"HoldCompleteForm[Unevaluated[1 + 1]]"#,
    );
    // An assigned symbol keeps its name inside the wrapper.
    assert_case(r#"x = 2; HoldCompleteForm[x]"#, r#"HoldCompleteForm[x]"#);
    // Nested and wrapped forms keep the wrapper in the output, like
    // `Hold[HoldForm[…]]` does.
    assert_case(
      r#"Hold[HoldCompleteForm[1 + 2]]"#,
      r#"Hold[HoldCompleteForm[1 + 2]]"#,
    );
    assert_case(
      r#"FullForm[HoldCompleteForm[1 + 2]]"#,
      r#"FullForm[HoldCompleteForm[1 + 2]]"#,
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

// The NHold attribute family — which argument slots N may numericize.
// All outputs verified against wolframscript. (NHoldAll already had
// coverage above; NHoldFirst/NHoldRest did not.)
mod nhold_attributes {
  use super::*;

  #[test]
  fn nhold_first_and_rest() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[g, NHoldFirst]; N[g[2, 1/3, 1/4]]").unwrap(),
      "g[2, 0.3333333333333333, 0.25]"
    );
    clear_state();
    assert_eq!(
      interpret("SetAttributes[h, NHoldRest]; N[h[2, 1/3, 1/4]]").unwrap(),
      "h[2., 1/3, 1/4]"
    );
    // The first argument is held even when it is the only one.
    clear_state();
    assert_eq!(
      interpret("SetAttributes[g, NHoldFirst]; N[g[1/3]]").unwrap(),
      "g[1/3]"
    );
  }

  #[test]
  fn held_arguments_are_not_descended_into() {
    clear_state();
    assert_eq!(
      interpret(
        "SetAttributes[f, NHoldAll]; SetAttributes[g, NHoldFirst]; \
         N[f[g[1/3]]]"
      )
      .unwrap(),
      "f[g[1/3]]"
    );
    // The protection is per-subtree: siblings still numericize.
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, NHoldAll]; N[{f[1/2], 1/2}]").unwrap(),
      "{f[1/2], 0.5}"
    );
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, NHoldAll]; N[f[1/2] + 1/2]").unwrap(),
      "0.5 + f[1/2]"
    );
  }

  #[test]
  fn precision_and_clearing() {
    // The precision form holds too; numericized parts carry the
    // requested precision.
    clear_state();
    assert_eq!(
      interpret("SetAttributes[f, NHoldAll]; N[f[2, 1/3], 20]").unwrap(),
      "f[2, 1/3]"
    );
    clear_state();
    assert_eq!(
      interpret("SetAttributes[h, NHoldRest]; N[h[2, 1/3], 20]").unwrap(),
      "h[2.`20., 1/3]"
    );
    // ClearAttributes restores normal numericization.
    clear_state();
    assert_eq!(
      interpret(
        "SetAttributes[f, NHoldAll]; ClearAttributes[f, NHoldAll]; N[f[1/3]]"
      )
      .unwrap(),
      "f[0.3333333333333333]"
    );
    // NHold combines with other attributes.
    clear_state();
    assert_eq!(
      interpret("SetAttributes[k, {NHoldFirst, Listable}]; Attributes[k]")
        .unwrap(),
      "{Listable, NHoldFirst}"
    );
  }
}

// Taking a part out of a held expression lifts it out of the wrapper that was
// suppressing evaluation, so the extracted piece evaluates. Woxi used to
// return the raw subexpression, so `Hold[1 + 1][[1]]` came back as `1 + 1`.
mod parts_of_held_expressions {
  use super::*;

  //   wolframscript -code 'Hold[1 + 1][[1]]'
  #[test]
  fn part_of_hold_evaluates() {
    clear_state();
    assert_eq!(interpret("Hold[1 + 1][[1]]").unwrap(), "2");
    assert_eq!(interpret("Hold[1 + 1, 2 + 2][[2]]").unwrap(), "4");
    // Indices reaching through the held expression evaluate the leaf.
    assert_eq!(interpret("Hold[{1 + 1, 2 + 2}][[1, 2]]").unwrap(), "4");
    // HoldForm and HoldComplete behave the same way.
    assert_eq!(interpret("HoldForm[1 + 1][[1]]").unwrap(), "2");
    assert_eq!(interpret("HoldComplete[1 + 1][[1]]").unwrap(), "2");
  }

  //   wolframscript -code 'First[Hold[1 + 1]]'
  #[test]
  fn first_and_last_of_hold_evaluate() {
    clear_state();
    assert_eq!(interpret("First[Hold[1 + 1]]").unwrap(), "2");
    assert_eq!(interpret("Last[Hold[1 + 1, 2 + 2]]").unwrap(), "4");
    // The default argument is only reached when there is no element at all.
    assert_eq!(interpret("First[Hold[1 + 1, 2 + 2], \"d\"]").unwrap(), "2");
  }

  //   wolframscript -code 'Extract[Hold[1 + 1, 2 + 2], {2}]'
  #[test]
  fn extract_of_hold_evaluates() {
    clear_state();
    assert_eq!(interpret("Extract[Hold[1 + 1, 2 + 2], {1}]").unwrap(), "2");
    assert_eq!(interpret("Extract[Hold[1 + 1, 2 + 2], {2}]").unwrap(), "4");
    assert_eq!(interpret("Extract[Hold[1 + 1], 1]").unwrap(), "2");
    assert_eq!(interpret("Extract[Hold[f[1 + 1]], {1, 1}]").unwrap(), "2");
    // A list of positions extracts each one.
    assert_eq!(
      interpret("Extract[Hold[1 + 1, 2 + 2], {{1}, {2}}]").unwrap(),
      "{2, 4}"
    );
    // The wrapper of the three-argument form is applied first, and then the
    // whole thing evaluates — unless the wrapper itself holds.
    assert_eq!(interpret("Extract[Hold[1 + 1], {1}, f]").unwrap(), "f[2]");
    assert_eq!(
      interpret("Extract[Hold[1 + 1], {1}, HoldForm]").unwrap(),
      "HoldForm[1 + 1]"
    );
  }

  // User-defined hold attributes count too, not just the built-in wrappers.
  //   wolframscript -code 'SetAttributes[hh, HoldAll]; hh[1 + 1][[1]]'
  #[test]
  fn part_of_user_hold_attribute_evaluates() {
    clear_state();
    assert_eq!(
      interpret("SetAttributes[hh, HoldAll]; hh[1 + 1][[1]]").unwrap(),
      "2"
    );
    clear_state();
    assert_eq!(
      interpret("SetAttributes[hh, HoldAll]; First[hh[1 + 1]]").unwrap(),
      "2"
    );
  }

  // Results that keep the holding head stay held, since evaluating them just
  // re-applies the same hold.
  //   wolframscript -code 'Hold[1 + 1, 2 + 2][[{1, 2}]]'
  #[test]
  fn re_wrapped_parts_stay_held() {
    clear_state();
    assert_eq!(
      interpret("Hold[1 + 1, 2 + 2][[{1, 2}]]").unwrap(),
      "Hold[1 + 1, 2 + 2]"
    );
    assert_eq!(
      interpret("Rest[Hold[1 + 1, 2 + 2]]").unwrap(),
      "Hold[2 + 2]"
    );
    assert_eq!(
      interpret("Most[Hold[1 + 1, 2 + 2]]").unwrap(),
      "Hold[1 + 1]"
    );
    assert_eq!(
      interpret("Take[Hold[1 + 1, 2 + 2], 1]").unwrap(),
      "Hold[1 + 1]"
    );
  }

  // An empty held expression has nothing to extract: the call is reported and
  // returned as is, rather than being re-evaluated (which repeated the
  // message).
  //   wolframscript -code 'First[Hold[]]'
  #[test]
  fn empty_hold_reports_once() {
    clear_state();
    assert_eq!(interpret("First[Hold[]]").unwrap(), "First[Hold[]]");
    assert_eq!(interpret("Last[Hold[]]").unwrap(), "Last[Hold[]]");
    // A default still wins over the message.
    assert_eq!(interpret("First[Hold[], 1 + 1]").unwrap(), "2");
  }

  // Parts of ordinary expressions are unaffected.
  #[test]
  fn parts_of_unheld_expressions_unchanged() {
    clear_state();
    assert_eq!(interpret("{1 + 1, 3}[[1]]").unwrap(), "2");
    assert_eq!(
      interpret("Extract[{Hold[1 + 1]}, {1}]").unwrap(),
      "Hold[1 + 1]"
    );
    assert_eq!(interpret("First[{Hold[1 + 1]}]").unwrap(), "Hold[1 + 1]");
  }
}

// The reported attributes of the built-ins, and the behaviour that follows from
// them.
mod builtin_attribute_table {
  use super::*;

  #[test]
  fn hold_attributes_match_wolframscript() {
    // Switch evaluates its first argument and holds the rest.
    assert_eq!(
      interpret("Attributes[Switch]").unwrap(),
      "{HoldRest, Protected}"
    );
    // Sum and Product hold their body and iterator.
    assert_eq!(
      interpret("Attributes[Sum]").unwrap(),
      "{HoldAll, Protected, ReadProtected}"
    );
    assert_eq!(
      interpret("Attributes[Product]").unwrap(),
      "{HoldAll, Protected, ReadProtected}"
    );
    // First and Last hold their default.
    assert_eq!(
      interpret("Attributes[First]").unwrap(),
      "{HoldRest, Protected}"
    );
    assert_eq!(
      interpret("Attributes[Last]").unwrap(),
      "{HoldRest, Protected}"
    );
    // Catch holds only the expression; Throw holds nothing.
    assert_eq!(
      interpret("Attributes[Catch]").unwrap(),
      "{HoldFirst, Protected}"
    );
    assert_eq!(interpret("Attributes[Throw]").unwrap(), "{Protected}");
    assert_eq!(
      interpret("Attributes[Pattern]").unwrap(),
      "{HoldFirst, Protected}"
    );
    assert_eq!(
      interpret("Attributes[SetAttributes]").unwrap(),
      "{HoldFirst, Protected}"
    );
    assert_eq!(
      interpret("Attributes[Association]").unwrap(),
      "{HoldAllComplete, Protected}"
    );
  }

  #[test]
  fn structural_attributes_match_wolframscript() {
    assert_eq!(
      interpret("Attributes[Join]").unwrap(),
      "{Flat, OneIdentity, Protected}"
    );
    assert_eq!(
      interpret("Attributes[StringJoin]").unwrap(),
      "{Flat, OneIdentity, Protected}"
    );
    assert_eq!(
      interpret("Attributes[Union]").unwrap(),
      "{Flat, OneIdentity, Protected, ReadProtected}"
    );
    assert_eq!(
      interpret("Attributes[Intersection]").unwrap(),
      "{Flat, OneIdentity, Protected, ReadProtected}"
    );
    assert_eq!(
      interpret("Attributes[Part]").unwrap(),
      "{NHoldRest, Protected, ReadProtected}"
    );
    assert_eq!(
      interpret("Attributes[Slot]").unwrap(),
      "{NHoldAll, Protected}"
    );
    // List and Symbol cannot be unprotected at all.
    assert_eq!(
      interpret("Attributes[List]").unwrap(),
      "{Locked, Protected}"
    );
    assert_eq!(
      interpret("Attributes[Symbol]").unwrap(),
      "{Locked, Protected}"
    );
  }

  #[test]
  fn symbols_that_were_missing_from_the_table() {
    for sym in [
      "Sequence",
      "Insert",
      "Delete",
      "Return",
      "Blank",
      "Verbatim",
      "Options",
      "OptionValue",
      "Evaluate",
      "Indeterminate",
    ] {
      assert_eq!(
        interpret(&format!("Attributes[{sym}]")).unwrap(),
        "{Protected}",
        "for {sym}"
      );
    }
    assert_eq!(
      interpret("Attributes[Limit]").unwrap(),
      "{Protected, ReadProtected}"
    );
    assert_eq!(
      interpret("Attributes[Missing]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  // Attributes is Listable: a list of symbols gives one list per symbol.
  #[test]
  fn attributes_threads_over_a_list() {
    assert_eq!(interpret("Attributes[{}]").unwrap(), "{}");
    assert_eq!(
      interpret("Attributes[{Plus, Hold}]").unwrap(),
      "{{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}, \
       {HoldAll, Protected}}"
    );
  }

  // First and Last only evaluate their default when they use it, and then they
  // do evaluate it.
  #[test]
  fn the_default_is_evaluated_only_when_used() {
    assert_eq!(interpret("First[{}, 2 + 2]").unwrap(), "4");
    assert_eq!(interpret("Last[{}, 2 + 2]").unwrap(), "4");
    assert_eq!(interpret("First[{1, 2}, 99]").unwrap(), "1");
    assert_eq!(interpret("Last[{1, 2}, 99]").unwrap(), "2");
    // The unused default is never evaluated, so nothing is printed here —
    // while a used one is.
    let unused =
      interpret_with_stdout("First[{1, 2}, Print[\"unused\"]]").unwrap();
    assert!(!unused.stdout.contains("unused"), "got {unused:?}");
    let used = interpret_with_stdout("First[{}, Print[\"used\"]]").unwrap();
    assert!(used.stdout.contains("used"), "got {used:?}");
  }
}

// Wolfram protects every `System`` symbol apart from a fixed set it leaves
// open so user code can attach its own definitions. Built-ins without an
// explicit entry in the attribute table therefore default to Protected.
mod default_protection {
  use super::*;

  #[test]
  fn builtins_without_a_table_entry_are_protected() {
    for name in ["Partition", "Array", "Identity", "Red", "Chop"] {
      assert_eq!(
        interpret(&format!("Attributes[{name}]")).unwrap(),
        "{Protected}",
        "{name}"
      );
    }
  }

  #[test]
  fn assignment_to_such_a_builtin_is_blocked() {
    clear_state();
    // wolframscript: Set::wrsym plus the RHS as the result.
    let out = interpret_with_stdout("BernoulliB = 1").unwrap();
    assert_eq!(out.result, "1");
    assert!(
      out
        .warnings
        .iter()
        .any(|w| w == "Set::wrsym: Symbol BernoulliB is Protected."),
      "got {out:?}"
    );
    assert_eq!(interpret("BernoulliB[2]").unwrap(), "1/6");
    clear_state();
  }

  #[test]
  fn builtins_whose_csv_row_has_no_description_are_protected_too() {
    // Whether functions.csv carries a description for a symbol says nothing
    // about its protection, so the fallback keys off the name alone.
    for name in ["GraphLayout", "EdgeWeight"] {
      assert_eq!(
        interpret(&format!("Attributes[{name}]")).unwrap(),
        "{Protected}",
        "{name}"
      );
    }
    clear_state();
    let out = interpret_with_stdout("CityData = 1").unwrap();
    assert!(
      out
        .warnings
        .iter()
        .any(|w| w == "Set::wrsym: Symbol CityData is Protected."),
      "got {out:?}"
    );
    clear_state();
  }

  #[test]
  fn assert_matches_wolframscript() {
    // Assert is HoldAllComplete and unprotected, unlike the neighbouring
    // HoldAll builtins it used to be grouped with.
    assert_eq!(
      interpret("Attributes[Assert]").unwrap(),
      "{HoldAllComplete}"
    );
  }

  #[test]
  fn symbols_wolfram_leaves_unprotected_have_no_attributes() {
    for name in [
      "Wedge",
      "CircleTimes",
      "Subset",
      "Star",
      "Square",
      "Tilde",
      "Derivative",
      "VonMisesDistribution",
    ] {
      let attrs = interpret(&format!("Attributes[{name}]")).unwrap();
      assert!(!attrs.contains("Protected"), "{name} got {attrs}");
    }
  }

  #[test]
  fn unprotected_operator_symbols_accept_user_definitions() {
    clear_state();
    // Wolfram leaves the operator symbols unprotected precisely so that they
    // can be given a meaning; `tests/scripts/metaprogramming.wls` relies on it.
    assert_eq!(
      interpret(
        "CircleTimes[x_, y_] := Mod[x, 10] Mod[y, 10]; 14 \\[CircleTimes] 13"
      )
      .unwrap(),
      "12"
    );
    clear_state();
    assert_eq!(
      interpret("CirclePlus[x_, y_] := x + y; 4 \\[CirclePlus] 0").unwrap(),
      "4"
    );
    clear_state();
  }
}
