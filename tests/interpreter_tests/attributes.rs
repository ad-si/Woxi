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
