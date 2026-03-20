use super::*;

mod user_defined_functions {
  use super::*;

  #[test]
  fn function_with_multiple_calls() {
    // Regression test: ensure function calls with different arguments
    // return different results (not cached incorrectly)
    assert_eq!(
      interpret("f[a_, b_, c_] := {a, b, c}; f[1, 2, 3]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn function_calls_are_not_incorrectly_cached() {
    // This test ensures that consecutive calls to a user-defined function
    // with different arguments return correct results
    assert_eq!(
      interpret(
        "g[a_, b_, c_] := a + b + c; x = g[1, 2, 3]; y = g[4, 5, 6]; {x, y}"
      )
      .unwrap(),
      "{6, 15}"
    );
  }

  #[test]
  fn map_apply_with_user_function() {
    // Test @@@ with user-defined function
    assert_eq!(
      interpret("f[a_, b_, c_] := a + b + c; f @@@ {{1, 2, 3}, {4, 5, 6}}")
        .unwrap(),
      "{6, 15}"
    );
  }

  #[test]
  fn user_function_with_if_lazy_evaluation() {
    // Test that If only evaluates the selected branch in user-defined functions
    // If both branches were evaluated, First[{}] would error
    assert_eq!(
      interpret("f[x_] := If[x > 0, 1, First[{}]]; f[5]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("f[x_] := If[x > 0, First[{}], 2]; f[-5]").unwrap(),
      "2"
    );
  }
}

mod conditional_definitions {
  use super::*;

  #[test]
  fn single_condition() {
    clear_state();
    assert_eq!(
      interpret(
        "f[n_ /; n > 0] := \"positive\"; f[n_] := \"non-positive\"; f[3]"
      )
      .unwrap(),
      "positive"
    );
  }

  #[test]
  fn single_condition_fallback() {
    clear_state();
    assert_eq!(
      interpret(
        "f[n_ /; n > 0] := \"positive\"; f[n_] := \"non-positive\"; f[-1]"
      )
      .unwrap(),
      "non-positive"
    );
  }

  #[test]
  fn multiple_conditions_fizzbuzz() {
    // Regression: multiple SetDelayed definitions with conditions overwrote each other
    clear_state();
    assert_eq!(
        interpret(r#"f[n_ /; Mod[n, 15] == 0] := "FizzBuzz"; f[n_ /; Mod[n, 3] == 0] := "Fizz"; f[n_ /; Mod[n, 5] == 0] := "Buzz"; f[n_] := n; f[3]"#).unwrap(),
        "Fizz"
      );
    assert_eq!(interpret("f[5]").unwrap(), "Buzz");
    assert_eq!(interpret("f[15]").unwrap(), "FizzBuzz");
    assert_eq!(interpret("f[7]").unwrap(), "7");
  }

  #[test]
  fn conditions_tried_in_order() {
    // Definitions are tried in the order they were defined
    clear_state();
    assert_eq!(
        interpret(r#"g[n_ /; n > 10] := "big"; g[n_ /; n > 0] := "small"; g[n_] := "zero or negative"; g[20]"#).unwrap(),
        "big"
      );
    assert_eq!(interpret("g[5]").unwrap(), "small");
    assert_eq!(interpret("g[0]").unwrap(), "zero or negative");
  }
}

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

mod replace_all_top_level {
  use super::*;

  #[test]
  fn replace_all_matches_whole_list_first() {
    // ReplaceAll should match the whole expression first before descending
    assert_eq!(interpret("{a, b, c} /. x_ -> {x}").unwrap(), "{{a, b, c}}");
  }

  #[test]
  fn replace_all_descends_when_top_level_fails() {
    // When the top-level doesn't match a specific pattern, descend into elements
    assert_eq!(interpret("{1, 2, 3} /. 2 -> x").unwrap(), "{1, x, 3}");
  }

  #[test]
  fn replace_all_descends_into_function_args() {
    // Should replace inside function call arguments
    assert_eq!(interpret("f[a, b, c] /. b -> x").unwrap(), "f[a, x, c]");
  }

  #[test]
  fn replace_all_descends_into_binary_op_divide() {
    // ReplaceAll must recurse into BinaryOp::Divide nodes
    assert_eq!(interpret("(a/b) /. {a -> 1, b -> 2}").unwrap(), "1/2");
    assert_eq!(interpret("(Sin[x]/Cos[x]) /. x -> 0").unwrap(), "0");
  }

  #[test]
  fn replace_all_descends_into_binary_op_power() {
    // ReplaceAll must recurse into BinaryOp::Power nodes
    assert_eq!(interpret("x^2 /. x -> 3").unwrap(), "9");
  }

  #[test]
  fn replace_all_descends_into_unary_op() {
    // ReplaceAll must recurse into UnaryOp (negation)
    assert_eq!(interpret("(-x) /. x -> 5").unwrap(), "-5");
  }

  #[test]
  fn replace_all_descends_into_nested_division_in_plus() {
    // Regression: ReplaceAll failed to substitute inside Divide within Plus
    assert_eq!(interpret("(x/y + x) /. {x -> 1, y -> 2}").unwrap(), "3/2");
  }

  #[test]
  fn replace_all_normalize_with_division() {
    // Regression: Normalize produces BinaryOp::Divide that ReplaceAll must descend into
    assert_eq!(
      interpret("(Normalize[{Cos[x] - Sin[x], Cos[x]}] /. x -> 0)[[1]]")
        .unwrap(),
      "1/Sqrt[2]"
    );
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

mod memory {
  use super::*;

  #[test]
  fn max_memory_used_returns_positive_integer() {
    let result = interpret("MaxMemoryUsed[]").unwrap();
    let val: i128 = result.parse().expect("should be an integer");
    assert!(val > 0, "MaxMemoryUsed should be positive: {}", val);
  }

  #[test]
  fn memory_in_use_returns_positive_integer() {
    let result = interpret("MemoryInUse[]").unwrap();
    let val: i128 = result.parse().expect("should be an integer");
    assert!(val > 0, "MemoryInUse should be positive: {}", val);
  }

  #[test]
  fn max_memory_at_least_memory_in_use() {
    // Evaluate MemoryInUse first, then MaxMemoryUsed second.
    // Peak RSS can only grow, so querying it after current ensures peak >= current.
    let result =
      interpret("With[{c = MemoryInUse[]}, MaxMemoryUsed[] >= c]").unwrap();
    assert_eq!(result, "True");
  }
}

mod context {
  use super::*;

  #[test]
  fn context_no_args() {
    // Context[] returns current context
    assert_eq!(interpret("Context[]").unwrap(), "Global`");
  }

  #[test]
  fn context_builtin_symbol() {
    // Built-in symbols are in System` context
    assert_eq!(interpret("Context[Plus]").unwrap(), "System`");
  }

  #[test]
  fn context_user_symbol() {
    // User-defined symbols are in Global` context
    assert_eq!(interpret("Context[x]").unwrap(), "Global`");
  }

  #[test]
  fn context_string_arg() {
    // String argument also works
    assert_eq!(interpret("Context[\"Plus\"]").unwrap(), "System`");
  }

  #[test]
  fn context_string_user() {
    // "myVar" doesn't exist as a symbol, so Context returns unevaluated
    assert_eq!(interpret("Context[\"myVar\"]").unwrap(), "Context[myVar]");
  }

  #[test]
  fn context_matches_dollar_context() {
    assert_eq!(interpret("Context[] === $Context").unwrap(), "True");
  }
}

mod anonymous_function_call {
  use super::*;

  #[test]
  fn identity_anonymous() {
    // #&[1] should return 1
    assert_eq!(interpret("#&[1]").unwrap(), "1");
  }

  #[test]
  fn power_anonymous() {
    // #^2&[{1, 2, 3}] should map squaring
    assert_eq!(interpret("#^2 &[{1, 2, 3}]").unwrap(), "{1, 4, 9}");
  }

  #[test]
  fn anonymous_with_addition() {
    assert_eq!(interpret("#+10&[5]").unwrap(), "15");
  }
}

mod function_name_substitution {
  use super::*;

  #[test]
  fn pass_function_as_argument() {
    clear_state();
    assert_eq!(
      interpret_with_stdout("g[f_] := f[]; g[Print[\"Hello\"] &]")
        .unwrap()
        .stdout,
      "Hello\n"
    );
  }

  #[test]
  fn repeat_with_anonymous_function() {
    clear_state();
    assert_eq!(
      interpret_with_stdout(
        "repeat[f_, n_] := Do[f[], {n}]; repeat[Print[\"hi\"] &, 3]"
      )
      .unwrap()
      .stdout,
      "hi\nhi\nhi\n"
    );
  }
}

mod function_head {
  use super::*;

  #[test]
  fn function_one_arg_is_pure_function() {
    // Function[body] is equivalent to body &
    assert_eq!(interpret("Function[# + 1][5]").unwrap(), "6");
  }

  #[test]
  fn function_one_arg_with_multiple_slots() {
    assert_eq!(interpret("Function[#1 + #2][3, 4]").unwrap(), "7");
  }

  #[test]
  fn function_named_param_single() {
    assert_eq!(interpret("Function[x, x + 1][5]").unwrap(), "6");
  }

  #[test]
  fn function_named_param_power() {
    assert_eq!(interpret("Function[x, x^2][3]").unwrap(), "9");
  }

  #[test]
  fn function_named_param_multi() {
    assert_eq!(interpret("Function[{x, y}, x + y][3, 4]").unwrap(), "7");
  }

  #[test]
  fn function_named_param_multiply() {
    assert_eq!(interpret("Function[{x, y}, x*y][3, 4]").unwrap(), "12");
  }

  #[test]
  fn function_named_identity() {
    assert_eq!(interpret("Function[x, x][10]").unwrap(), "10");
  }

  #[test]
  fn function_display_one_arg() {
    // Function[body] displays as body &
    assert_eq!(interpret("Function[# + 1]").unwrap(), "#1 + 1 & ");
  }

  #[test]
  fn function_display_named_single() {
    assert_eq!(
      interpret("Function[x, x + 1]").unwrap(),
      "Function[x, x + 1]"
    );
  }

  #[test]
  fn function_display_named_multi() {
    assert_eq!(
      interpret("Function[{x, y}, x + y]").unwrap(),
      "Function[{x, y}, x + y]"
    );
  }

  #[test]
  fn function_assigned_to_variable() {
    clear_state();
    assert_eq!(interpret("f = Function[x, x + 1]; f[10]").unwrap(), "11");
  }

  #[test]
  fn function_multi_param_assigned() {
    clear_state();
    assert_eq!(
      interpret("g = Function[{x, y}, x^2 + y^2]; g[3, 4]").unwrap(),
      "25"
    );
  }

  #[test]
  fn function_in_map() {
    assert_eq!(
      interpret("Map[Function[x, x^2], {1, 2, 3, 4}]").unwrap(),
      "{1, 4, 9, 16}"
    );
  }

  #[test]
  fn function_in_select() {
    assert_eq!(
      interpret("Select[{1, 2, 3, 4, 5}, Function[x, OddQ[x]]]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn function_in_apply() {
    assert_eq!(
      interpret("Apply[Function[{x, y}, x + y], {10, 20}]").unwrap(),
      "30"
    );
  }

  #[test]
  fn function_holdall_body() {
    // Function should not evaluate the body prematurely
    assert_eq!(interpret("Function[x, OddQ[x]][3]").unwrap(), "True");
  }

  #[test]
  fn function_head_is_function() {
    assert_eq!(interpret("Head[Function[x, x + 1]]").unwrap(), "Function");
  }
}

mod set_delayed {
  use super::*;

  #[test]
  fn list_pattern_destructuring() {
    clear_state();
    assert_eq!(
      interpret("swap[{a_Integer, b_Integer}] := {b, a}; swap[{1, 2}]")
        .unwrap(),
      "{2, 1}"
    );
  }

  #[test]
  fn list_pattern_with_computation() {
    clear_state();
    assert_eq!(
      interpret("f[{x_Integer, y_Integer}] := x + y; f[{3, 4}]").unwrap(),
      "7"
    );
  }

  #[test]
  fn literal_arg_priority_over_blank() {
    // f[1] := 1 (literal) should take priority over f[x_] := x + 1 (blank)
    clear_state();
    assert_eq!(interpret("f[1] := 1; f[x_] := x + 1; f[1]").unwrap(), "1");
    assert_eq!(interpret("f[5]").unwrap(), "6");
  }

  #[test]
  fn literal_arg_priority_over_pattern_test() {
    // Literal match should take priority even when defined before PatternTest
    clear_state();
    assert_eq!(
      interpret("g[0] := 0; g[x_ ? EvenQ] := x / 2; g[x_] := 3 x + 1; g[0]")
        .unwrap(),
      "0"
    );
    assert_eq!(interpret("g[4]").unwrap(), "2");
    assert_eq!(interpret("g[3]").unwrap(), "10");
  }

  mod structural_patterns {
    use super::*;

    #[test]
    fn reciprocal_pattern_does_not_match_plain_variable() {
      // Regression: 1/x_ should NOT match plain x
      clear_state();
      assert_eq!(
        interpret("Int[1/x_, x_Symbol] := Log[x]; Int[x, x]").unwrap(),
        "Int[x, x]"
      );
    }

    #[test]
    fn reciprocal_pattern_matches_reciprocal() {
      clear_state();
      assert_eq!(
        interpret("Int[1/x_, x_Symbol] := Log[x]; Int[1/y, y]").unwrap(),
        "Log[y]"
      );
    }

    #[test]
    fn simple_reciprocal_pattern() {
      clear_state();
      assert_eq!(interpret("f[1/x_] := Log[x]; f[1/y]").unwrap(), "Log[y]");
    }

    #[test]
    fn sum_pattern() {
      clear_state();
      assert_eq!(interpret("f[a_ + b_] := a * b; f[x + y]").unwrap(), "x*y");
    }

    #[test]
    fn sum_pattern_does_not_match_non_sum() {
      clear_state();
      assert_eq!(interpret("f[a_ + b_] := a * b; f[3]").unwrap(), "f[3]");
    }
  }
}

mod down_values {
  use super::*;

  #[test]
  fn basic_down_value() {
    clear_state();
    assert_eq!(interpret("f[0] = 42; f[0]").unwrap(), "42");
  }

  #[test]
  fn multiple_down_values() {
    clear_state();
    assert_eq!(interpret("f[0] = 0; f[1] = 1; f[0]").unwrap(), "0");
    assert_eq!(interpret("f[1]").unwrap(), "1");
  }

  #[test]
  fn down_value_with_pattern() {
    clear_state();
    assert_eq!(
      interpret(
        "g[0] = 0; g[1] = 1; g[n_Integer] := g[n - 1] + g[n - 2]; g[5]"
      )
      .unwrap(),
      "5"
    );
  }

  #[test]
  fn memoized_down_value() {
    clear_state();
    assert_eq!(
      interpret(
        "h[0] = 0; h[1] = 1; h[n_Integer] := h[n] = h[n - 1] + h[n - 2]; h[10]"
      )
      .unwrap(),
      "55"
    );
  }
}

mod pattern_matching {
  use super::*;

  mod blank_pattern {
    use super::*;

    #[test]
    fn simple_blank_matches_any() {
      // x_ matches any expression
      assert_eq!(interpret("5 /. x_ :> 10").unwrap(), "10");
      // Strings are displayed without quotes at top level (Wolfram behavior)
      assert_eq!(interpret("\"hello\" /. x_ :> \"world\"").unwrap(), "world");
    }

    #[test]
    fn blank_with_replacement_using_variable() {
      // The matched value can be used in replacement
      // Note: expressions in replacement need parentheses
      assert_eq!(interpret("5 /. x_ :> (x + 1)").unwrap(), "6");
      assert_eq!(interpret("3 /. n_ :> (n * 2)").unwrap(), "6");
    }

    #[test]
    fn blank_on_list_elements() {
      // Pattern applies to each element in a list
      // Note: expressions in replacement need parentheses
      assert_eq!(
        interpret("{1, 2, 3} /. x_ :> (x + 10)").unwrap(),
        "{11, 12, 13}"
      );
    }
  }

  mod anonymous_blank_in_set {
    use super::*;

    #[test]
    fn anonymous_blank_downvalue() {
      // f[_] = value — anonymous Blank pattern via Set should match any argument
      assert_eq!(
        interpret("ProductQ[_] = False; ProductQ[4]").unwrap(),
        "False"
      );
    }

    #[test]
    fn anonymous_blank_downvalue_multiple_args() {
      assert_eq!(interpret("h[_, _] = True; h[1, 2]").unwrap(), "True");
    }

    #[test]
    fn named_blank_downvalue_via_set() {
      // Named pattern in Set should also work
      assert_eq!(interpret("sq[x_] = x^2; sq[5]").unwrap(), "25");
    }
  }

  mod blank_sequence_pattern {
    use super::*;

    #[test]
    fn blank_sequence_in_set_delayed() {
      // u__ (BlankSequence) matches one or more arguments
      assert_eq!(
        interpret("HalfIntegerQ[u__] := False; HalfIntegerQ[1/2]").unwrap(),
        "False"
      );
    }

    #[test]
    fn blank_sequence_with_body_reference() {
      // Named BlankSequence used in the body
      assert_eq!(interpret("g[u__] := u; g[42]").unwrap(), "42");
    }

    #[test]
    fn blank_null_sequence_in_set_delayed() {
      // u___ (BlankNullSequence) also matches single arguments
      assert_eq!(interpret("f[u___] := u; f[7]").unwrap(), "7");
    }

    #[test]
    fn double_underscore_with_head() {
      // x__Integer — BlankSequence with head constraint
      assert_eq!(interpret("f[x__Integer] := x + 1; f[5]").unwrap(), "6");
    }

    #[test]
    fn blank_sequence_multi_arg_length() {
      // f[x__] := Length[{x}] should match multiple args and wrap in Sequence
      assert_eq!(
        interpret("f[x__] := Length[{x}]; {f[x, y, z], f[]}").unwrap(),
        "{3, f[]}"
      );
    }

    #[test]
    fn blank_sequence_single_arg() {
      // Single argument should bind directly without Sequence wrapper
      assert_eq!(interpret("g[x__] := x + 1; g[5]").unwrap(), "6");
    }

    #[test]
    fn blank_null_sequence_zero_args() {
      // BlankNullSequence matches zero arguments
      assert_eq!(interpret("h[x___] := Length[{x}]; h[]").unwrap(), "0");
    }

    #[test]
    fn matchq_blank_sequence_basic() {
      // Anonymous __ matches one or more args inside function patterns
      assert_eq!(interpret("MatchQ[f[1, 2, 3], f[__]]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[f[1], f[__]]").unwrap(), "True");
      // Must match at least one
      assert_eq!(interpret("MatchQ[f[], f[__]]").unwrap(), "False");
    }

    #[test]
    fn matchq_blank_null_sequence() {
      // ___ matches zero or more
      assert_eq!(interpret("MatchQ[f[], f[___]]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[f[1], f[___]]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[f[1, 2], f[___]]").unwrap(), "True");
    }

    #[test]
    fn matchq_blank_sequence_with_head() {
      // __Integer matches one or more Integer args
      assert_eq!(
        interpret("MatchQ[f[1, 2, 3], f[__Integer]]").unwrap(),
        "True"
      );
      // Fails when any arg is not Integer
      assert_eq!(
        interpret("MatchQ[f[1, 2, x], f[__Integer]]").unwrap(),
        "False"
      );
    }

    #[test]
    fn replace_all_with_blank_sequence() {
      // Named x__ in ReplaceAll binds to Sequence
      assert_eq!(
        interpret("f[1, 2, 3] /. f[x__] :> {x}").unwrap(),
        "{1, 2, 3}"
      );
    }

    #[test]
    fn replace_all_blank_sequence_named_sum() {
      assert_eq!(
        interpret("{f[1, 2], f[3, 4, 5]} /. f[x__] :> Plus[x]").unwrap(),
        "{3, 12}"
      );
    }

    #[test]
    fn cases_with_blank_sequence() {
      assert_eq!(
        interpret("Cases[{f[1, 2], f[3], g[4, 5]}, f[__]]").unwrap(),
        "{f[1, 2], f[3]}"
      );
    }

    #[test]
    fn count_with_blank_sequence() {
      assert_eq!(
        interpret("Count[{f[1], f[2, 3], g[4]}, f[__]]").unwrap(),
        "2"
      );
    }

    #[test]
    fn position_with_blank_sequence() {
      assert_eq!(
        interpret("Position[{f[1], f[2, 3], g[4]}, f[__]]").unwrap(),
        "{{1}, {2}}"
      );
    }

    #[test]
    fn multiple_blank_sequences_in_definition() {
      // f[x__, y__] splits args: first gets minimum, rest goes to second
      assert_eq!(
        interpret("f[x__, y__] := {{x}, {y}}; f[1, 2, 3]").unwrap(),
        "{{1}, {2, 3}}"
      );
    }

    #[test]
    fn blank_sequence_pattern_test() {
      // __?IntegerQ matches one or more integers
      assert_eq!(
        interpret("MatchQ[f[1, 2, 3], f[__?IntegerQ]]").unwrap(),
        "True"
      );
      assert_eq!(
        interpret("MatchQ[f[1, 2, x], f[__?IntegerQ]]").unwrap(),
        "False"
      );
    }

    #[test]
    fn anonymous_blank_in_matchq() {
      // Standalone _ matches any single expression
      assert_eq!(interpret("MatchQ[42, _]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[{1, 2}, {_, _}]").unwrap(), "True");
    }

    #[test]
    fn anonymous_blank_with_head() {
      // _Integer matches integer, _Symbol matches symbol
      assert_eq!(interpret("MatchQ[42, _Integer]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[x, _Integer]").unwrap(), "False");
      assert_eq!(interpret("MatchQ[x, _Symbol]").unwrap(), "True");
    }
  }

  mod conditional_pattern {
    use super::*;

    #[test]
    fn condition_true_matches() {
      assert_eq!(
        interpret("6 /. x_ /; Mod[x, 2] == 0 :> \"even\"").unwrap(),
        "even"
      );
    }

    #[test]
    fn condition_false_no_match() {
      assert_eq!(
        interpret("5 /. x_ /; Mod[x, 2] == 0 :> \"even\"").unwrap(),
        "5"
      );
    }

    #[test]
    fn conditional_with_function_call() {
      assert_eq!(
        interpret("3 /. i_ /; Mod[i, 3] == 0 :> \"Fizz\"").unwrap(),
        "Fizz"
      );
      assert_eq!(
        interpret("5 /. i_ /; Mod[i, 5] == 0 :> \"Buzz\"").unwrap(),
        "Buzz"
      );
    }

    #[test]
    fn conditional_on_list() {
      assert_eq!(
        interpret("{1, 2, 3, 4} /. x_ /; x > 2 :> 0").unwrap(),
        "{1, 2, 0, 0}"
      );
    }
  }

  mod pattern_test {
    use super::*;

    #[test]
    fn pattern_test_matches() {
      assert_eq!(interpret("4 /. x_?EvenQ :> \"even\"").unwrap(), "even");
    }

    #[test]
    fn pattern_test_no_match() {
      assert_eq!(interpret("3 /. x_?EvenQ :> \"even\"").unwrap(), "3");
    }

    #[test]
    fn pattern_test_on_list() {
      assert_eq!(
        interpret("{1, 2, 3, 4} /. x_?EvenQ :> 0").unwrap(),
        "{1, 0, 3, 0}"
      );
    }

    #[test]
    fn pattern_test_with_oddq() {
      assert_eq!(
        interpret("{1, 2, 3, 4} /. x_?OddQ :> 0").unwrap(),
        "{0, 2, 0, 4}"
      );
    }

    #[test]
    fn anonymous_blank_pattern_test() {
      // _?EvenQ without a named variable
      assert_eq!(interpret("Count[{1, 2, 3, 4, 5}, _?EvenQ]").unwrap(), "2");
    }

    #[test]
    fn anonymous_blank_pattern_test_with_anonymous_function() {
      // _?(func &) with parenthesized anonymous function
      assert_eq!(
        interpret("Count[{1, 2, 3, 4, 5}, _?(MemberQ[{2, 3, 5}, #] &)]")
          .unwrap(),
        "3"
      );
    }

    #[test]
    fn pattern_test_cases() {
      assert_eq!(
        interpret("Cases[{1, \"a\", 2, \"b\", 3}, _?StringQ]").unwrap(),
        "{a, b}"
      );
    }

    #[test]
    fn pattern_test_anonymous_function_replace_all() {
      assert_eq!(
        interpret("{1, 2, 3, 4, 5} /. x_?(# > 3 &) -> 0").unwrap(),
        "{1, 2, 3, 0, 0}"
      );
    }

    #[test]
    fn pattern_test_named_with_anonymous_function() {
      assert_eq!(
        interpret("{1, 2, 3, 4} /. x_?(EvenQ[#] &) :> x^2").unwrap(),
        "{1, 4, 3, 16}"
      );
    }
  }

  mod multiple_rules {
    use super::*;

    #[test]
    fn list_of_rules_applied_in_order() {
      // First matching rule wins
      // Note: strings inside lists still show quotes (only top-level strings are unquoted)
      assert_eq!(
        interpret(
          "{1, 2, 3} /. {x_ /; x == 1 :> \"one\", x_ /; x == 2 :> \"two\"}"
        )
        .unwrap(),
        "{one, two, 3}"
      );
    }

    #[test]
    fn fizzbuzz_style_rules() {
      // Test the FizzBuzz pattern
      assert_eq!(
          interpret("15 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "FizzBuzz"
        );
      assert_eq!(
          interpret("9 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "Fizz"
        );
      assert_eq!(
          interpret("10 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "Buzz"
        );
      assert_eq!(
          interpret("7 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "7"
        );
    }
  }

  mod structural_pattern {
    use super::*;

    #[test]
    fn power_pattern_matches_all() {
      // x^n_ matches any power of x, binding n to the exponent
      assert_eq!(
        interpret("{x^2, x^3, x^4} /. x^n_ :> f[n]").unwrap(),
        "{f[2], f[3], f[4]}"
      );
    }

    #[test]
    fn power_pattern_with_condition() {
      // x^n_ /; EvenQ[n] matches only even powers
      assert_eq!(
        interpret("{x^2, x^3, x^4} /. x^n_ /; EvenQ[n] :> f[n]").unwrap(),
        "{f[2], x^3, f[4]}"
      );
    }

    #[test]
    fn power_pattern_non_matching() {
      // Pattern doesn't match non-power expressions
      assert_eq!(
        interpret("{x, x^2, y^3} /. x^n_ :> f[n]").unwrap(),
        "{x, f[2], y^3}"
      );
    }

    #[test]
    fn function_call_pattern() {
      // f[n_] pattern matching within replacement rules
      assert_eq!(
        interpret("{f[1], f[2], g[3]} /. f[n_] :> n^2").unwrap(),
        "{1, 4, g[3]}"
      );
    }
  }
}

mod paren_anonymous_function {
  use super::*;

  #[test]
  fn paren_anonymous_with_comparison() {
    // (# === "")& is an anonymous function testing for empty string
    // Uses postfix @ operator since direct call syntax is not supported
    assert_eq!(interpret("(# === \"\")& @ \"hello\"").unwrap(), "False");
    assert_eq!(interpret("(# === \"\")& @ \"\"").unwrap(), "True");
  }

  #[test]
  fn paren_anonymous_with_arithmetic() {
    assert_eq!(interpret("(# + 1)& @ 5").unwrap(), "6");
    assert_eq!(interpret("(# * 2 + 3)& @ 4").unwrap(), "11");
  }

  #[test]
  fn paren_anonymous_in_map() {
    assert_eq!(interpret("Map[(# + 1)&, {1, 2, 3}]").unwrap(), "{2, 3, 4}");
  }

  #[test]
  fn paren_anonymous_with_if() {
    assert_eq!(interpret("(If[# > 0, #, 0])& @ 5").unwrap(), "5");
    assert_eq!(interpret("(If[# > 0, #, 0])& @ -3").unwrap(), "0");
  }

  #[test]
  fn paren_anonymous_in_postfix() {
    // If[# === "", i, #]& @ "hello" should return "hello" (strings displayed without quotes)
    assert_eq!(
      interpret("(If[# === \"\", \"empty\", #])& @ \"hello\"").unwrap(),
      "hello"
    );
    assert_eq!(
      interpret("(If[# === \"\", \"empty\", #])& @ \"\"").unwrap(),
      "empty"
    );
  }

  #[test]
  fn paren_anonymous_with_compound_expression() {
    // (expr1; expr2)& should create an anonymous function with CompoundExpression body
    assert_eq!(
      interpret("Reap[Nest[(Sow[#]; 3*# + 1) &, 7, 5]]").unwrap(),
      "{1822, {{7, 22, 67, 202, 607}}}"
    );
  }

  #[test]
  fn paren_anonymous_direct_call() {
    // (expr)&[args] should call the anonymous function directly
    assert_eq!(interpret("(# + 1) &[5]").unwrap(), "6");
    assert_eq!(interpret("(# * 2 + 3) &[4]").unwrap(), "11");
    assert_eq!(interpret("(#1 + #2) &[3, 4]").unwrap(), "7");
  }

  #[test]
  fn function_anonymous_direct_call() {
    // If[...]&[args] should call the anonymous function directly
    assert_eq!(interpret("If[# > 0, #, 0] &[5]").unwrap(), "5");
    assert_eq!(interpret("If[# > 0, #, 0] &[-3]").unwrap(), "0");
  }

  #[test]
  fn list_anonymous_direct_call() {
    // {expr}&[args] should call the anonymous function directly
    assert_eq!(interpret("{#, #^2} &[3]").unwrap(), "{3, 9}");
    assert_eq!(interpret("{#1, #2, #1 + #2} &[2, 5]").unwrap(), "{2, 5, 7}");
  }

  #[test]
  fn paren_call_with_ampersand_inside() {
    // (expr &)[args] — anonymous function with & inside parens, called with bracket args
    assert_eq!(interpret("(# + 1 &)[5]").unwrap(), "6");
    assert_eq!(interpret("(#^2 &)[3]").unwrap(), "9");
    assert_eq!(interpret("(#1 + #2 &)[3, 4]").unwrap(), "7");
  }

  #[test]
  fn paren_call_derivative_anonymous() {
    // (D[#, x] &)[expr] — derivative as anonymous function with & inside parens
    assert_eq!(
      interpret("(D[#, x] &)[x^3 + Sin[x]]").unwrap(),
      "3*x^2 + Cos[x]"
    );
  }

  #[test]
  fn paren_call_if_anonymous() {
    // (If[...] &)[args] — If as anonymous function with & inside parens
    assert_eq!(interpret("(If[# > 0, #, -#] &)[5]").unwrap(), "5");
    assert_eq!(interpret("(If[# > 0, #, -#] &)[-5]").unwrap(), "5");
  }

  #[test]
  fn paren_call_chained() {
    // (expr)[a][b] — chained calls on parenthesized expression
    assert_eq!(interpret("(# &)[#^2 &][3]").unwrap(), "9");
  }
}

mod part_anonymous_function {
  use super::*;

  #[test]
  fn slot_part_simple() {
    // #[[n]]& extracts the nth element
    assert_eq!(interpret("#[[2]] &[{3, 4, 5}]").unwrap(), "4");
    assert_eq!(interpret("#[[1]] &[{10, 20, 30}]").unwrap(), "10");
    assert_eq!(interpret("#[[3]] &[{10, 20, 30}]").unwrap(), "30");
  }

  #[test]
  fn slot_part_in_list_anonymous() {
    // {#[[1]], #[[2]]}& is a list anonymous function with Part extracts
    assert_eq!(interpret("{#[[2]], #[[1]]} &[{3, 4}]").unwrap(), "{4, 3}");
  }

  #[test]
  fn slot_part_with_arithmetic() {
    // #[[1]] + #[[2]]& performs arithmetic on parts
    assert_eq!(interpret("#[[1]] + #[[2]] &[{3, 4}]").unwrap(), "7");
    assert_eq!(interpret("#[[1]] * #[[2]] &[{5, 6}]").unwrap(), "30");
  }

  #[test]
  fn slot_part_in_nest() {
    // Fibonacci via Nest with Part-based anonymous function
    assert_eq!(
      interpret("Nest[{#[[2]], #[[1]] + #[[2]]} &, {0, 1}, 10]").unwrap(),
      "{55, 89}"
    );
  }

  #[test]
  fn slot_part_in_map() {
    // Map with Part-based anonymous function
    assert_eq!(
      interpret("Map[#[[1]] &, {{1, 2}, {3, 4}, {5, 6}}]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn slot_part_without_anonymous_function() {
    // Part extraction on slots in evaluated contexts (no &)
    assert_eq!(interpret("{1, 2, 3}[[2]]").unwrap(), "2");
    assert_eq!(interpret("x = {10, 20, 30}; x[[3]]").unwrap(), "30");
  }
}

mod postfix_with_anonymous_function {
  use super::*;

  #[test]
  fn postfix_at_with_simple_anonymous() {
    assert_eq!(interpret("#^2& @ 3").unwrap(), "9");
    assert_eq!(interpret("#+1& @ 5").unwrap(), "6");
  }

  #[test]
  fn postfix_at_with_function_anonymous() {
    assert_eq!(interpret("Sqrt[#]& @ 16").unwrap(), "4");
  }

  #[test]
  fn postfix_at_with_string_result() {
    // Anonymous function that returns a string (strings displayed without quotes)
    assert_eq!(
      interpret("If[# > 0, \"positive\", \"non-positive\"]& @ 5").unwrap(),
      "positive"
    );
    assert_eq!(
      interpret("If[# > 0, \"positive\", \"non-positive\"]& @ -3").unwrap(),
      "non-positive"
    );
  }

  #[test]
  fn postfix_at_preserves_string_arg() {
    // When the argument is a string, it should be preserved
    assert_eq!(interpret("StringLength[#]& @ \"hello\"").unwrap(), "5");
  }
}

mod prefix_application_associativity {
  use super::*;

  #[test]
  fn right_associative_chaining() {
    // f @ g @ x should be f[g[x]] (right-associative)
    assert_eq!(
      interpret("Double[x_] := x * 2; Double @ Sin @ (Pi/2)").unwrap(),
      "2"
    );
  }

  #[test]
  fn single_prefix() {
    assert_eq!(interpret("Sqrt @ 16").unwrap(), "4");
  }
}

mod operator_precedence_at_map_apply {
  use super::*;

  #[test]
  fn apply_looser_than_map() {
    // @@ binds looser than /@ — f @@ g /@ h = Apply[f, Map[g, h]]
    assert_eq!(
      interpret("StringJoin @@ ToString /@ IntegerDigits[50, 2]").unwrap(),
      "110010"
    );
  }

  #[test]
  fn prefix_at_tighter_than_map() {
    // @ binds tighter than /@ — f @ g /@ h = Map[f[g], h]
    assert_eq!(
      interpret("FullForm[Hold[f @ g /@ h]]").unwrap(),
      "Hold[Map[f[g], h]]"
    );
  }

  #[test]
  fn prefix_at_tighter_than_apply() {
    // @ binds tighter than @@ — f @ g @@ h = Apply[f[g], h]
    assert_eq!(
      interpret("FullForm[Hold[f @ g @@ h]]").unwrap(),
      "Hold[Apply[f[g], h]]"
    );
  }

  #[test]
  fn anon_func_map_continuation() {
    // f & /@ {1, 2} — & then /@ should work even for single-term body
    assert_eq!(
      interpret("FullForm[Hold[f & /@ g @ h]]").unwrap(),
      "Hold[Map[Function[f], g[h]]]"
    );
  }

  #[test]
  fn anon_func_single_term_with_map() {
    // x & /@ {1, 2} — single identifier before & with Map continuation
    assert_eq!(interpret("42 & /@ {1, 2, 3}").unwrap(), "{42, 42, 42}");
  }
}

mod variable_as_function_head {
  use super::*;

  #[test]
  fn variable_holding_function_name() {
    clear_state();
    assert_eq!(
      interpret("t = Flatten; t @ {{1, 2}, {3, 4}}").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn variable_in_apply() {
    clear_state();
    assert_eq!(interpret("f = Plus; f @@ {1, 2, 3}").unwrap(), "6");
  }

  #[test]
  fn variable_in_map() {
    clear_state();
    assert_eq!(
      interpret("f = ToString; f /@ {1, 2, 3}").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn variable_in_function_call() {
    clear_state();
    assert_eq!(interpret("f = Length; f[{1, 2, 3}]").unwrap(), "3");
  }
}

mod expression_level_anonymous_function {
  use super::*;

  #[test]
  fn multi_operator_body() {
    // #^2 + 1 & — body has multiple operators, not just Slot op Term
    assert_eq!(interpret("Map[#^2 + 1 &, {3, 0}]").unwrap(), "{10, 1}");
    assert_eq!(interpret("Map[# * 2 - 3 &, {5, 10}]").unwrap(), "{7, 17}");
  }

  #[test]
  fn replace_all_body() {
    // # /. {rules} & — body contains ReplaceAll with conditional patterns
    assert_eq!(
      interpret(
        "Map[# /. {n_ /; EvenQ[n] :> n/2, n_ /; OddQ[n] :> 3 n + 1} &, {27, 6}]"
      )
      .unwrap(),
      "{82, 3}"
    );
  }

  #[test]
  fn replace_all_simple_rule() {
    // # /. rule & — body contains ReplaceAll with a single rule
    assert_eq!(
      interpret("Map[# /. x_ /; x > 3 :> 0 &, {1, 2, 5, 4, 3}]").unwrap(),
      "{1, 2, 0, 0, 3}"
    );
  }

  #[test]
  fn nestlist_collatz() {
    // Full Collatz sequence via NestList with ReplaceAll in anonymous function
    assert_eq!(
      interpret("NestList[# /. {n_ /; EvenQ[n] :> n/2, n_ /; OddQ[n] :> 3 n + 1} &, 27, 10]")
        .unwrap(),
      "{27, 82, 41, 124, 62, 31, 94, 47, 142, 71, 214}"
    );
  }

  #[test]
  fn replace_all_variable_rules() {
    // # /. r & — RHS of /. is a variable holding rules
    assert_eq!(
      interpret("r = {x -> 1, y -> 2}; {x, y, z} /. r").unwrap(),
      "{1, 2, z}"
    );
    // In anonymous function context
    assert_eq!(
      interpret("r = {x_ /; EvenQ[x] :> x/2}; Map[# /. r &, {4, 7}]").unwrap(),
      "{2, 7}"
    );
    // Nest with variable rules
    assert_eq!(
      interpret("r = {a -> b, b -> c}; Nest[# /. r &, a, 2]").unwrap(),
      "c"
    );
  }

  #[test]
  fn replace_repeated_variable_rules() {
    // # //. r — RHS of //. is a variable holding rules
    assert_eq!(interpret("r = {a -> b, b -> c}; a //. r").unwrap(), "c");
  }

  #[test]
  fn postfix_application_body() {
    // body // func & — body contains postfix application
    assert_eq!(
      interpret("Map[# // Abs &, {-3, 2, -1}]").unwrap(),
      "{3, 2, 1}"
    );
  }

  #[test]
  fn existing_forms_still_work() {
    // Simple: Slot op Term (uses SimpleAnonymousFunction)
    assert_eq!(interpret("Map[# + 1 &, {5}]").unwrap(), "{6}");
    // Function call (uses FunctionAnonymousFunction)
    assert_eq!(interpret("Map[Sin[#] &, {0}]").unwrap(), "{0}");
    // Paren (uses ParenAnonymousFunction)
    assert_eq!(interpret("(# + 1) &[5]").unwrap(), "6");
    // List (uses ListAnonymousFunction)
    assert_eq!(interpret("{#, #^2} &[3]").unwrap(), "{3, 9}");
  }
}

mod leaf_count {
  use super::*;

  #[test]
  fn leaf_count_sum_expr() {
    assert_eq!(interpret("LeafCount[1 + x + y^a]").unwrap(), "6");
  }

  #[test]
  fn leaf_count_function_call() {
    assert_eq!(interpret("LeafCount[f[x, y]]").unwrap(), "3");
  }

  #[test]
  fn leaf_count_list() {
    assert_eq!(interpret("LeafCount[{1, 2, 3}]").unwrap(), "4");
  }

  #[test]
  fn leaf_count_atom() {
    assert_eq!(interpret("LeafCount[42]").unwrap(), "1");
  }

  #[test]
  fn leaf_count_symbol() {
    assert_eq!(interpret("LeafCount[x]").unwrap(), "1");
  }
}

mod byte_count {
  use super::*;

  #[test]
  fn byte_count_integer() {
    // i128 = 16 bytes
    assert_eq!(interpret("ByteCount[42]").unwrap(), "16");
  }

  #[test]
  fn byte_count_real() {
    // machine real: 16 bytes (Wolfram's representation)
    assert_eq!(interpret("ByteCount[3.14]").unwrap(), "16");
  }

  #[test]
  fn byte_count_string() {
    // 32-byte header; "hello" (5 chars) fits within header
    assert_eq!(interpret("ByteCount[\"hello\"]").unwrap(), "32");
  }

  #[test]
  fn byte_count_empty_string() {
    // empty string: 32-byte header
    assert_eq!(interpret("ByteCount[\"\"]").unwrap(), "32");
  }

  #[test]
  fn byte_count_symbol_is_zero() {
    // Symbols are shared, so 0 bytes
    assert_eq!(interpret("ByteCount[x]").unwrap(), "0");
  }

  #[test]
  fn byte_count_list() {
    // {1, 2, 3}: 40 base + 3*8 slots + 3*16 integers = 112
    assert_eq!(interpret("ByteCount[{1, 2, 3}]").unwrap(), "112");
  }

  #[test]
  fn byte_count_nested_list() {
    // {{1,2},{3,4}}: 40 base + 2*8 slots + 2*(40+2*8+2*16) = 232
    assert_eq!(interpret("ByteCount[{{1, 2}, {3, 4}}]").unwrap(), "232");
  }

  #[test]
  fn byte_count_function_call() {
    // f[x, y]: 40 base + 2*8 slots + 2*0 symbols = 56
    assert_eq!(interpret("ByteCount[f[x, y]]").unwrap(), "56");
  }

  #[test]
  fn byte_count_larger_is_more() {
    // A larger list should have a larger byte count
    let small = interpret("ByteCount[{1, 2}]").unwrap();
    let large = interpret("ByteCount[{1, 2, 3, 4, 5}]").unwrap();
    assert!(
      large.parse::<i128>().unwrap() > small.parse::<i128>().unwrap(),
      "Larger list should have larger byte count"
    );
  }
}

mod free_q {
  use super::*;

  #[test]
  fn free_q_head_matching_plus() {
    // Plus is the head of a+b inside a^(a+b)
    assert_eq!(interpret("FreeQ[{1, 2, a^(a+b)}, Plus]").unwrap(), "False");
  }

  #[test]
  fn free_q_flat_subsequence() {
    // a+b is a subsequence of a+b+c (Plus is Flat)
    assert_eq!(interpret("FreeQ[a+b+c, a+b]").unwrap(), "False");
  }

  #[test]
  fn free_q_flat_subsequence_bc() {
    assert_eq!(interpret("FreeQ[a+b+c, b+c]").unwrap(), "False");
  }

  #[test]
  fn free_q_head_list() {
    assert_eq!(interpret("FreeQ[{1,2,3}, List]").unwrap(), "False");
  }

  #[test]
  fn free_q_head_plus_direct() {
    assert_eq!(interpret("FreeQ[a+b+c, Plus]").unwrap(), "False");
  }

  #[test]
  fn free_q_non_flat_no_subset() {
    // f is NOT Flat, so f[a,c] is NOT a sub-expression of f[a,b,c]
    assert_eq!(interpret("FreeQ[f[a,b,c], f[a,c]]").unwrap(), "True");
  }

  #[test]
  fn free_q_symbol_as_element() {
    // Plus appears as a literal element in the list
    assert_eq!(interpret("FreeQ[{Plus, 1}, Plus]").unwrap(), "False");
  }

  #[test]
  fn free_q_basic_true() {
    assert_eq!(interpret("FreeQ[{1, 2, 3}, 4]").unwrap(), "True");
  }

  #[test]
  fn free_q_basic_false() {
    assert_eq!(interpret("FreeQ[{1, 2, 3}, 2]").unwrap(), "False");
  }
}

mod subsets {
  use super::*;

  #[test]
  fn all_subsets() {
    assert_eq!(
      interpret("Subsets[{a, b, c}]").unwrap(),
      "{{}, {a}, {b}, {c}, {a, b}, {a, c}, {b, c}, {a, b, c}}"
    );
  }

  #[test]
  fn max_size_integer() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d}, 2]").unwrap(),
      "{{}, {a}, {b}, {c}, {d}, {a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}}"
    );
  }

  #[test]
  fn exact_size() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d}, {2}]").unwrap(),
      "{{a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}}"
    );
  }

  #[test]
  fn exact_size_with_max_count() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d, e}, {3}, 5]").unwrap(),
      "{{a, b, c}, {a, b, d}, {a, b, e}, {a, c, d}, {a, c, e}}"
    );
  }

  #[test]
  fn size_range_with_step() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d}, {0, 4, 2}]").unwrap(),
      "{{}, {a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}, {a, b, c, d}}"
    );
  }

  #[test]
  fn all_with_max_count() {
    assert_eq!(
      interpret("Subsets[Range[5], All, 5]").unwrap(),
      "{{}, {1}, {2}, {3}, {4}}"
    );
  }

  #[test]
  fn part_spec_single() {
    assert_eq!(
      interpret("Subsets[Range[5], All, {25}]").unwrap(),
      "{{2, 4, 5}}"
    );
  }

  #[test]
  fn part_spec_range_reverse() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d}, All, {15, 1, -2}]").unwrap(),
      "{{b, c, d}, {a, b, d}, {c, d}, {b, c}, {a, c}, {d}, {b}, {}}"
    );
  }
}

mod append_prepend {
  use super::*;

  #[test]
  fn append_to_list() {
    assert_eq!(interpret("Append[{1, 2, 3}, 4]").unwrap(), "{1, 2, 3, 4}");
  }

  #[test]
  fn append_to_function_call() {
    assert_eq!(interpret("Append[f[a, b], c]").unwrap(), "f[a, b, c]");
  }

  #[test]
  fn append_list_element() {
    assert_eq!(
      interpret("Append[{a, b}, {c, d}]").unwrap(),
      "{a, b, {c, d}}"
    );
  }

  #[test]
  fn prepend_to_list() {
    assert_eq!(interpret("Prepend[{1, 2, 3}, 0]").unwrap(), "{0, 1, 2, 3}");
  }

  #[test]
  fn prepend_to_function_call() {
    assert_eq!(interpret("Prepend[f[a, b], c]").unwrap(), "f[c, a, b]");
  }
}

mod drop_extended {
  use super::*;

  #[test]
  fn drop_range() {
    assert_eq!(
      interpret("Drop[{a, b, c, d, e}, {2, -2}]").unwrap(),
      "{a, e}"
    );
  }

  #[test]
  fn drop_single_index() {
    assert_eq!(
      interpret("Drop[{a, b, c, d, e}, {3}]").unwrap(),
      "{a, b, d, e}"
    );
  }

  #[test]
  fn drop_zero() {
    assert_eq!(interpret("Drop[{a, b, c, d}, 0]").unwrap(), "{a, b, c, d}");
  }
}

mod partition_extended {
  use super::*;

  #[test]
  fn partition_with_stride() {
    assert_eq!(
      interpret("Partition[{a, b, c, d, e, f}, 3, 1]").unwrap(),
      "{{a, b, c}, {b, c, d}, {c, d, e}, {d, e, f}}"
    );
  }

  #[test]
  fn partition_with_stride_2() {
    assert_eq!(
      interpret("Partition[{a, b, c, d, e}, 2, 1]").unwrap(),
      "{{a, b}, {b, c}, {c, d}, {d, e}}"
    );
  }
}

mod block_map {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e, h}, 2]").unwrap(),
      "{g[{a, b}], g[{c, d}], g[{e, h}]}"
    );
  }

  #[test]
  fn with_total() {
    assert_eq!(
      interpret("BlockMap[Total, {1, 2, 3, 4, 5, 6}, 2]").unwrap(),
      "{3, 7, 11}"
    );
  }

  #[test]
  fn block_size_3() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e, h}, 3]").unwrap(),
      "{g[{a, b, c}], g[{d, e, h}]}"
    );
  }

  #[test]
  fn drops_remainder() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e}, 3]").unwrap(),
      "{g[{a, b, c}]}"
    );
  }

  #[test]
  fn with_offset_1() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e}, 3, 1]").unwrap(),
      "{g[{a, b, c}], g[{b, c, d}], g[{c, d, e}]}"
    );
  }

  #[test]
  fn with_offset_2() {
    assert_eq!(
      interpret("BlockMap[g, {a, b, c, d, e, h, i}, 3, 2]").unwrap(),
      "{g[{a, b, c}], g[{c, d, e}], g[{e, h, i}]}"
    );
  }
}

mod downsample {
  use super::*;

  #[test]
  fn by_two() {
    assert_eq!(
      interpret("Downsample[{a, b, c, d, e, f, g, h}, 2]").unwrap(),
      "{a, c, e, g}"
    );
  }

  #[test]
  fn by_three() {
    assert_eq!(
      interpret("Downsample[{a, b, c, d, e, f, g, h}, 3]").unwrap(),
      "{a, d, g}"
    );
  }

  #[test]
  fn with_offset() {
    assert_eq!(
      interpret("Downsample[{a, b, c, d, e, f, g, h}, 2, 2]").unwrap(),
      "{b, d, f, h}"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("Downsample[{1, 2, 3, 4, 5, 6}, 2]").unwrap(),
      "{1, 3, 5}"
    );
  }
}

mod square_matrix_q {
  use super::*;

  #[test]
  fn square() {
    assert_eq!(
      interpret("SquareMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn rectangular() {
    assert_eq!(
      interpret("SquareMatrixQ[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn vector() {
    assert_eq!(interpret("SquareMatrixQ[{1, 2, 3}]").unwrap(), "False");
  }

  #[test]
  fn three_by_three() {
    assert_eq!(
      interpret("SquareMatrixQ[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "True"
    );
  }
}

mod contains_any {
  use super::*;

  #[test]
  fn has_common() {
    assert_eq!(interpret("ContainsAny[{a, b, c}, {b, d}]").unwrap(), "True");
  }

  #[test]
  fn no_common() {
    assert_eq!(
      interpret("ContainsAny[{a, b, c}, {d, e}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("ContainsAny[{1, 2, 3}, {3, 4, 5}]").unwrap(),
      "True"
    );
  }
}

mod contains_none {
  use super::*;

  #[test]
  fn no_common() {
    assert_eq!(
      interpret("ContainsNone[{a, b, c}, {d, e}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn has_common() {
    assert_eq!(
      interpret("ContainsNone[{a, b, c}, {b, d}]").unwrap(),
      "False"
    );
  }
}

mod factorial_power {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("FactorialPower[10, 3]").unwrap(), "720");
  }

  #[test]
  fn zero_power() {
    assert_eq!(interpret("FactorialPower[5, 0]").unwrap(), "1");
  }

  #[test]
  fn one_power() {
    assert_eq!(interpret("FactorialPower[5, 1]").unwrap(), "5");
  }

  #[test]
  fn with_step() {
    assert_eq!(interpret("FactorialPower[10, 3, 2]").unwrap(), "480");
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("FactorialPower[n, 3]").unwrap(),
      "FactorialPower[n, 3]"
    );
  }
}

mod machine_number_q {
  use super::*;

  #[test]
  fn real_is_true() {
    assert_eq!(interpret("MachineNumberQ[1.5]").unwrap(), "True");
  }

  #[test]
  fn integer_is_false() {
    assert_eq!(interpret("MachineNumberQ[1]").unwrap(), "False");
  }

  #[test]
  fn rational_is_false() {
    assert_eq!(interpret("MachineNumberQ[1/3]").unwrap(), "False");
  }

  #[test]
  fn string_is_false() {
    assert_eq!(interpret("MachineNumberQ[\"hello\"]").unwrap(), "False");
  }
}

mod text_string {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("TextString[42]").unwrap(), "42");
  }

  #[test]
  fn list() {
    assert_eq!(interpret("TextString[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn string_passthrough() {
    assert_eq!(interpret("TextString[\"hello\"]").unwrap(), "hello");
  }
}

mod string_partition {
  use super::*;

  #[test]
  fn basic_partition() {
    assert_eq!(
      interpret("StringPartition[\"abcdefghij\", 3]").unwrap(),
      "{abc, def, ghi}"
    );
  }

  #[test]
  fn non_divisible_drops_remainder() {
    assert_eq!(
      interpret("StringPartition[\"abcde\", 2]").unwrap(),
      "{ab, cd}"
    );
  }

  #[test]
  fn with_offset_1() {
    assert_eq!(
      interpret("StringPartition[\"abcdefghij\", 3, 1]").unwrap(),
      "{abc, bcd, cde, def, efg, fgh, ghi, hij}"
    );
  }

  #[test]
  fn with_offset_2() {
    assert_eq!(
      interpret("StringPartition[\"abcdefghij\", 3, 2]").unwrap(),
      "{abc, cde, efg, ghi}"
    );
  }

  #[test]
  fn single_char_partition() {
    assert_eq!(
      interpret("StringPartition[\"abc\", 1]").unwrap(),
      "{a, b, c}"
    );
  }
}

mod power_range {
  use super::*;

  #[test]
  fn default_factor_10() {
    assert_eq!(
      interpret("PowerRange[1, 1000, 10]").unwrap(),
      "{1, 10, 100, 1000}"
    );
  }

  #[test]
  fn factor_2() {
    assert_eq!(
      interpret("PowerRange[2, 32, 2]").unwrap(),
      "{2, 4, 8, 16, 32}"
    );
  }

  #[test]
  fn two_arg_default_factor() {
    assert_eq!(interpret("PowerRange[1, 100]").unwrap(), "{1, 10, 100}");
  }

  #[test]
  fn fractional_factor() {
    assert_eq!(
      interpret("PowerRange[1, 1/27, 1/3]").unwrap(),
      "{1, 1/3, 1/9, 1/27}"
    );
  }
}

mod color_q {
  use super::*;

  #[test]
  fn rgb_color_is_true() {
    assert_eq!(interpret("ColorQ[RGBColor[1, 0, 0]]").unwrap(), "True");
  }

  #[test]
  fn named_color_is_true() {
    assert_eq!(interpret("ColorQ[Red]").unwrap(), "True");
  }

  #[test]
  fn integer_is_false() {
    assert_eq!(interpret("ColorQ[42]").unwrap(), "False");
  }

  #[test]
  fn string_is_false() {
    assert_eq!(interpret("ColorQ[\"hello\"]").unwrap(), "False");
  }
}

mod polynomial_quotient_remainder {
  use super::*;

  #[test]
  fn basic_division() {
    assert_eq!(
      interpret("PolynomialQuotientRemainder[x^3 + 2x + 1, x + 1, x]").unwrap(),
      "{3 - x + x^2, -2}"
    );
  }

  #[test]
  fn exact_division() {
    assert_eq!(
      interpret("PolynomialQuotientRemainder[x^2 - 1, x - 1, x]").unwrap(),
      "{1 + x, 0}"
    );
  }

  #[test]
  fn linear_by_linear() {
    assert_eq!(
      interpret("PolynomialQuotientRemainder[2x + 3, x + 1, x]").unwrap(),
      "{2, 1}"
    );
  }
}

mod heaviside_pi {
  use super::*;

  #[test]
  fn zero_is_one() {
    assert_eq!(interpret("HeavisidePi[0]").unwrap(), "1");
  }

  #[test]
  fn inside_is_one() {
    assert_eq!(interpret("HeavisidePi[1/4]").unwrap(), "1");
  }

  #[test]
  fn outside_is_zero() {
    assert_eq!(interpret("HeavisidePi[-1]").unwrap(), "0");
  }

  #[test]
  fn at_boundary_unevaluated() {
    assert_eq!(interpret("HeavisidePi[1/2]").unwrap(), "HeavisidePi[1/2]");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("HeavisidePi[x]").unwrap(), "HeavisidePi[x]");
  }
}

mod prime_omega {
  use super::*;

  #[test]
  fn one_has_zero() {
    assert_eq!(interpret("PrimeOmega[1]").unwrap(), "0");
  }

  #[test]
  fn prime_has_one() {
    assert_eq!(interpret("PrimeOmega[7]").unwrap(), "1");
  }

  #[test]
  fn composite_with_multiplicity() {
    // 12 = 2^2 * 3, so PrimeOmega = 2 + 1 = 3
    assert_eq!(interpret("PrimeOmega[12]").unwrap(), "3");
  }

  #[test]
  fn larger_number() {
    // 100 = 2^2 * 5^2, so PrimeOmega = 2 + 2 = 4
    assert_eq!(interpret("PrimeOmega[100]").unwrap(), "4");
  }
}

mod prime_nu {
  use super::*;

  #[test]
  fn one_has_zero() {
    assert_eq!(interpret("PrimeNu[1]").unwrap(), "0");
  }

  #[test]
  fn prime_has_one() {
    assert_eq!(interpret("PrimeNu[7]").unwrap(), "1");
  }

  #[test]
  fn composite_distinct_factors() {
    // 12 = 2^2 * 3, so PrimeNu = 2 (distinct primes: 2, 3)
    assert_eq!(interpret("PrimeNu[12]").unwrap(), "2");
  }

  #[test]
  fn three_distinct_factors() {
    // 60 = 2^2 * 3 * 5, so PrimeNu = 3
    assert_eq!(interpret("PrimeNu[60]").unwrap(), "3");
  }
}

mod contains_all {
  use super::*;

  #[test]
  fn all_present() {
    assert_eq!(
      interpret("ContainsAll[{1, 2, 3, 4, 5}, {2, 4}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn not_all_present() {
    assert_eq!(
      interpret("ContainsAll[{1, 2, 3}, {2, 4}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn empty_subset() {
    assert_eq!(interpret("ContainsAll[{1, 2, 3}, {}]").unwrap(), "True");
  }
}

mod missing_q {
  use super::*;

  #[test]
  fn missing_is_true() {
    assert_eq!(interpret("MissingQ[Missing[]]").unwrap(), "True");
  }

  #[test]
  fn missing_with_reason_is_true() {
    assert_eq!(interpret("MissingQ[Missing[\"reason\"]]").unwrap(), "True");
  }

  #[test]
  fn integer_is_false() {
    assert_eq!(interpret("MissingQ[42]").unwrap(), "False");
  }
}

mod hilbert_matrix {
  use super::*;

  #[test]
  fn two_by_two() {
    assert_eq!(
      interpret("HilbertMatrix[2]").unwrap(),
      "{{1, 1/2}, {1/2, 1/3}}"
    );
  }

  #[test]
  fn three_by_three() {
    assert_eq!(
      interpret("HilbertMatrix[3]").unwrap(),
      "{{1, 1/2, 1/3}, {1/2, 1/3, 1/4}, {1/3, 1/4, 1/5}}"
    );
  }
}

mod toeplitz_matrix {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("ToeplitzMatrix[{1, 2, 3}]").unwrap(),
      "{{1, 2, 3}, {2, 1, 2}, {3, 2, 1}}"
    );
  }

  #[test]
  fn two_by_two() {
    assert_eq!(
      interpret("ToeplitzMatrix[{a, b}]").unwrap(),
      "{{a, b}, {b, a}}"
    );
  }
}

mod mantissa_exponent {
  use super::*;

  #[test]
  fn real_value() {
    assert_eq!(
      interpret("MantissaExponent[350.12]").unwrap(),
      "{0.35012, 3}"
    );
  }

  #[test]
  fn integer_value() {
    assert_eq!(interpret("MantissaExponent[100]").unwrap(), "{1/10, 3}");
  }

  #[test]
  fn negative_real() {
    assert_eq!(interpret("MantissaExponent[-42.5]").unwrap(), "{-0.425, 2}");
  }
}

mod heaviside_lambda {
  use super::*;

  #[test]
  fn zero_is_one() {
    assert_eq!(interpret("HeavisideLambda[0]").unwrap(), "1");
  }

  #[test]
  fn inside_is_value() {
    assert_eq!(interpret("HeavisideLambda[1/3]").unwrap(), "2/3");
  }

  #[test]
  fn at_boundary_is_zero() {
    assert_eq!(interpret("HeavisideLambda[1]").unwrap(), "0");
  }

  #[test]
  fn outside_real_is_zero() {
    assert_eq!(interpret("HeavisideLambda[1.5]").unwrap(), "0.");
  }

  #[test]
  fn negative_inside() {
    assert_eq!(interpret("HeavisideLambda[-0.5]").unwrap(), "0.5");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("HeavisideLambda[x]").unwrap(),
      "HeavisideLambda[x]"
    );
  }
}

mod mean_deviation {
  use super::*;

  #[test]
  fn basic_list() {
    assert_eq!(interpret("MeanDeviation[{1, 2, 3, 4, 5}]").unwrap(), "6/5");
  }

  #[test]
  fn even_list() {
    assert_eq!(interpret("MeanDeviation[{2, 4, 6, 8}]").unwrap(), "2");
  }

  #[test]
  fn identical_values() {
    assert_eq!(interpret("MeanDeviation[{5, 5, 5}]").unwrap(), "0");
  }
}

mod reverse_extended {
  use super::*;

  #[test]
  fn syntax_q_valid() {
    assert_eq!(interpret("SyntaxQ[\"1 + 2\"]").unwrap(), "True");
  }

  #[test]
  fn syntax_q_invalid() {
    assert_eq!(interpret("SyntaxQ[\"1 + \"]").unwrap(), "False");
  }

  #[test]
  fn interquartile_range() {
    assert_eq!(
      interpret("InterquartileRange[{1, 2, 3, 4, 5, 6, 7, 8}]").unwrap(),
      "4"
    );
  }

  #[test]
  fn interquartile_range_odd() {
    assert_eq!(
      interpret("InterquartileRange[{1, 3, 5, 7, 9}]").unwrap(),
      "5"
    );
  }

  #[test]
  fn unitary_matrix_q_identity() {
    assert_eq!(
      interpret("UnitaryMatrixQ[{{1, 0}, {0, 1}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn unitary_matrix_q_permutation() {
    assert_eq!(
      interpret("UnitaryMatrixQ[{{0, 1}, {1, 0}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn unitary_matrix_q_non_unitary() {
    assert_eq!(
      interpret("UnitaryMatrixQ[{{1, 0}, {0, 2}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn reflection_matrix_axis() {
    assert_eq!(
      interpret("ReflectionMatrix[{1, 0}]").unwrap(),
      "{{-1, 0}, {0, 1}}"
    );
  }

  #[test]
  fn reflection_matrix_diagonal() {
    assert_eq!(
      interpret("ReflectionMatrix[{1, 1}]").unwrap(),
      "{{0, -1}, {-1, 0}}"
    );
  }

  #[test]
  fn reverse_function_call() {
    assert_eq!(interpret("Reverse[x[a, b, c]]").unwrap(), "x[c, b, a]");
  }

  #[test]
  fn reverse_list() {
    assert_eq!(interpret("Reverse[{1, 2, 3, 4}]").unwrap(), "{4, 3, 2, 1}");
  }
}

mod first_last_extended {
  use super::*;

  #[test]
  fn first_with_default_nonempty() {
    assert_eq!(interpret("First[{a, b, c}, default]").unwrap(), "a");
  }

  #[test]
  fn first_with_default_empty() {
    assert_eq!(interpret("First[{}, default]").unwrap(), "default");
  }

  #[test]
  fn last_with_default_nonempty() {
    assert_eq!(interpret("Last[{a, b, c}, default]").unwrap(), "c");
  }

  #[test]
  fn last_with_default_empty() {
    assert_eq!(interpret("Last[{}, default]").unwrap(), "default");
  }

  #[test]
  fn first_of_function_call() {
    assert_eq!(interpret("First[f[a, b]]").unwrap(), "a");
  }

  #[test]
  fn last_of_function_call() {
    assert_eq!(interpret("Last[f[a, b]]").unwrap(), "b");
  }
}

mod array_predicates {
  use super::*;

  #[test]
  fn array_q_true() {
    assert_eq!(interpret("ArrayQ[{{1, 2}, {3, 4}}]").unwrap(), "True");
  }

  #[test]
  fn array_q_false() {
    assert_eq!(interpret("ArrayQ[{{1, 2}, {3}}]").unwrap(), "False");
  }

  #[test]
  fn matrix_q_true() {
    assert_eq!(interpret("MatrixQ[{{1, 2}, {3, 4}}]").unwrap(), "True");
  }

  #[test]
  fn matrix_q_false() {
    assert_eq!(interpret("MatrixQ[{1, 2, 3}]").unwrap(), "False");
  }

  #[test]
  fn vector_q_true() {
    assert_eq!(interpret("VectorQ[{1, 2, 3}]").unwrap(), "True");
  }

  #[test]
  fn vector_q_false() {
    assert_eq!(interpret("VectorQ[{{1}, {2}}]").unwrap(), "False");
  }
}

mod dimensions_extended {
  use super::*;

  #[test]
  fn function_call_head() {
    assert_eq!(interpret("Dimensions[f[f[a, b, c]]]").unwrap(), "{1, 3}");
  }
}

mod transpose_extended {
  use super::*;

  #[test]
  fn one_d_list() {
    assert_eq!(interpret("Transpose[{a, b, c}]").unwrap(), "{a, b, c}");
  }
}

mod product_extended {
  use super::*;

  #[test]
  fn symbolic_body() {
    assert_eq!(
      interpret("Product[f[i], {i, 1, 7}]").unwrap(),
      "f[1]*f[2]*f[3]*f[4]*f[5]*f[6]*f[7]"
    );
  }

  #[test]
  fn with_step() {
    // Product[k, {k, 1, 6, 2}] = 1 * 3 * 5 = 15
    assert_eq!(interpret("Product[k, {k, 1, 6, 2}]").unwrap(), "15");
  }
}

mod real_sign {
  use super::*;

  #[test]
  fn negative_real() {
    assert_eq!(interpret("RealSign[-3.]").unwrap(), "-1");
  }

  #[test]
  fn positive_integer() {
    assert_eq!(interpret("RealSign[5]").unwrap(), "1");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("RealSign[0]").unwrap(), "0");
  }

  #[test]
  fn negative_integer() {
    assert_eq!(interpret("RealSign[-7]").unwrap(), "-1");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("RealSign[3/4]").unwrap(), "1");
  }

  #[test]
  fn complex_stays_symbolic() {
    assert_eq!(
      interpret("RealSign[2. + 3. I]").unwrap(),
      "RealSign[2. + 3.*I]"
    );
  }

  #[test]
  fn symbolic_stays() {
    assert_eq!(interpret("RealSign[x]").unwrap(), "RealSign[x]");
  }
}

mod between {
  use super::*;

  #[test]
  fn basic_in_range() {
    assert_eq!(interpret("Between[6, {4, 10}]").unwrap(), "True");
  }

  #[test]
  fn out_of_range() {
    assert_eq!(interpret("Between[2, {4, 10}]").unwrap(), "False");
  }

  #[test]
  fn operator_form() {
    assert_eq!(interpret("Between[{4, 10}][6]").unwrap(), "True");
  }

  #[test]
  fn symbolic_constants() {
    assert_eq!(interpret("Between[2, {E, Pi}]").unwrap(), "False");
  }

  #[test]
  fn multiple_ranges() {
    assert_eq!(interpret("Between[5, {{1, 2}, {4, 6}}]").unwrap(), "True");
  }

  #[test]
  fn multiple_ranges_no_match() {
    assert_eq!(interpret("Between[3, {{1, 2}, {4, 6}}]").unwrap(), "False");
  }

  #[test]
  fn boundary_values() {
    assert_eq!(interpret("Between[4, {4, 10}]").unwrap(), "True");
    assert_eq!(interpret("Between[10, {4, 10}]").unwrap(), "True");
  }
}

mod alternatives {
  use super::*;

  #[test]
  fn replace_all_with_alternatives() {
    assert_eq!(
      interpret("a + b + c + d /. (a | b) -> t").unwrap(),
      "c + d + 2*t"
    );
  }

  #[test]
  fn replace_all_single_match() {
    assert_eq!(interpret("{a, b, c} /. (a | c) -> x").unwrap(), "{x, b, x}");
  }

  #[test]
  fn match_q_with_alternatives() {
    assert_eq!(interpret("MatchQ[5, _Integer | _String]").unwrap(), "True");
  }

  #[test]
  fn match_q_no_match() {
    assert_eq!(
      interpret("MatchQ[5.0, _Integer | _String]").unwrap(),
      "False"
    );
  }

  #[test]
  fn cases_with_alternatives() {
    assert_eq!(
      interpret("Cases[{1, \"a\", 2, \"b\", 3}, _Integer | _String]").unwrap(),
      "{1, a, 2, b, 3}"
    );
  }

  #[test]
  fn replace_with_three_alternatives() {
    assert_eq!(
      interpret("{a, b, c, d} /. (a | b | c) -> x").unwrap(),
      "{x, x, x, d}"
    );
  }

  #[test]
  fn alternatives_function_call_form() {
    // Alternatives[a, b, c] displays as a | b | c
    assert_eq!(interpret("Alternatives[a, b, c]").unwrap(), "a | b | c");
  }

  #[test]
  fn alternatives_single_arg() {
    assert_eq!(interpret("Alternatives[a]").unwrap(), "Alternatives[a]");
  }

  #[test]
  fn alternatives_flattening() {
    // Alternatives is Flat: nested Alternatives are flattened
    assert_eq!(
      interpret("Alternatives[Alternatives[a, b], c]").unwrap(),
      "(a | b) | c"
    );
  }

  #[test]
  fn alternatives_attributes() {
    assert_eq!(
      interpret("Attributes[Alternatives]").unwrap(),
      "{Protected}"
    );
  }

  #[test]
  fn alternatives_head() {
    assert_eq!(interpret("Head[a | b | c]").unwrap(), "Alternatives");
    assert_eq!(
      interpret("Head[Alternatives[a, b]]").unwrap(),
      "Alternatives"
    );
  }

  #[test]
  fn alternatives_match_q_function_form() {
    // MatchQ with Alternatives as FunctionCall
    assert_eq!(
      interpret("MatchQ[1, Alternatives[1, 2, 3]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[5, Alternatives[1, 2, 3]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn alternatives_cases_function_form() {
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4, 5}, Alternatives[1, 3, 5]]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn alternatives_string_replace() {
    assert_eq!(
      interpret("StringReplace[\"abcabc\", \"a\" | \"b\" -> \"x\"]").unwrap(),
      "xxcxxc"
    );
  }

  #[test]
  fn alternatives_string_cases() {
    assert_eq!(
      interpret("StringCases[\"the cat sat on the mat\", \"cat\" | \"mat\"]")
        .unwrap(),
      "{cat, mat}"
    );
  }

  #[test]
  fn alternatives_precedence_over_rule() {
    // | binds tighter than -> so "a" | "b" -> "x" is Rule[Alternatives["a","b"], "x"]
    assert_eq!(interpret("Head[\"a\" | \"b\" -> \"x\"]").unwrap(), "Rule");
  }
}

mod filter_rules {
  use super::*;

  #[test]
  fn single_key() {
    assert_eq!(
      interpret("FilterRules[{x -> 100, y -> 1000}, x]").unwrap(),
      "{x -> 100}"
    );
  }

  #[test]
  fn key_list() {
    assert_eq!(
      interpret("FilterRules[{x -> 100, y -> 1000, z -> 10000}, {a, b, x, z}]")
        .unwrap(),
      "{x -> 100, z -> 10000}"
    );
  }

  #[test]
  fn no_match() {
    assert_eq!(
      interpret("FilterRules[{x -> 1, y -> 2}, {a, b}]").unwrap(),
      "{}"
    );
  }
}

mod operate {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Operate[p, f[a, b]]").unwrap(), "p[f][a, b]");
  }

  #[test]
  fn level_1() {
    assert_eq!(interpret("Operate[p, f[a, b], 1]").unwrap(), "p[f][a, b]");
  }

  #[test]
  fn level_0() {
    assert_eq!(
      interpret("Operate[p, f[a][b][c], 0]").unwrap(),
      "p[f[a][b][c]]"
    );
  }
}

mod reverse_sort {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("ReverseSort[{c, b, d, a}]").unwrap(),
      "{d, c, b, a}"
    );
  }

  #[test]
  fn with_less() {
    assert_eq!(
      interpret("ReverseSort[{1, 2, 0, 3}, Less]").unwrap(),
      "{3, 2, 1, 0}"
    );
  }

  #[test]
  fn with_greater() {
    assert_eq!(
      interpret("ReverseSort[{1, 2, 0, 3}, Greater]").unwrap(),
      "{0, 1, 2, 3}"
    );
  }
}

mod sort_with_comparator {
  use super::*;

  #[test]
  fn greater() {
    assert_eq!(
      interpret("Sort[{1, 2, 0, 3}, Greater]").unwrap(),
      "{3, 2, 1, 0}"
    );
  }

  #[test]
  fn less() {
    assert_eq!(interpret("Sort[{3, 1, 2}, Less]").unwrap(), "{1, 2, 3}");
  }
}

mod quartiles {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("Quartiles[Range[25]]").unwrap(),
      "{27/4, 13, 77/4}"
    );
  }

  #[test]
  fn small_list() {
    assert_eq!(
      interpret("Quartiles[{1, 2, 3, 4, 5}]").unwrap(),
      "{7/4, 3, 17/4}"
    );
  }
}

mod compile {
  use super::*;

  #[test]
  fn compile_basic() {
    clear_state();
    assert_eq!(
      interpret("cf = Compile[{x, y}, x + 2 y]; cf[2.5, 4.3]").unwrap(),
      "11.1"
    );
  }

  #[test]
  fn compile_with_typed_specs() {
    clear_state();
    assert_eq!(
      interpret("cf = Compile[{{x, _Real}}, Sin[x]]; cf[1.4]").unwrap(),
      "0.9854497299884601"
    );
  }

  #[test]
  fn compile_complex_body() {
    clear_state();
    assert_eq!(
      interpret("cf = Compile[{{x, _Real}, {y, _Integer}}, If[x == 0.0 && y <= 0, 0.0, Sin[x ^ y] + 1 / Min[x, 0.5]] + 0.5]; cf[3.5, 2]").unwrap(),
      "2.1888806450188727"
    );
  }

  #[test]
  fn compile_forces_numerical() {
    clear_state();
    // Compile forces numerical evaluation even for integer inputs
    assert_eq!(interpret("sqr = Compile[{x}, x x]; sqr[2]").unwrap(), "4.");
  }

  #[test]
  fn compile_head() {
    clear_state();
    assert_eq!(
      interpret("sqr = Compile[{x}, x x]; Head[sqr]").unwrap(),
      "CompiledFunction"
    );
  }

  #[test]
  fn compile_display() {
    clear_state();
    // CompiledFunction display should show the variable list and body
    let result = interpret("Compile[{x, y}, x + y]").unwrap();
    assert!(result.starts_with("CompiledFunction["));
  }
}

mod expression {
  use super::*;

  #[test]
  fn evaluates_to_self() {
    assert_eq!(interpret("Expression").unwrap(), "Expression");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Expression]").unwrap(), "Symbol");
  }
}

mod quit {
  use super::*;

  // Note: Quit[] calls std::process::exit() so we can't directly test
  // successful exits in unit tests. These tests verify edge cases.

  #[test]
  fn quit_too_many_args_unevaluated() {
    assert_eq!(interpret("Quit[1, 2]").unwrap(), "Quit[1, 2]");
  }

  #[test]
  fn exit_too_many_args_unevaluated() {
    assert_eq!(interpret("Exit[1, 2]").unwrap(), "Exit[1, 2]");
  }

  #[test]
  fn quit_before_terminates() {
    // Quit[] should prevent subsequent expressions from being evaluated.
    // In a CompoundExpression, the process exits so "after" is never printed.
    // We can't test process::exit in unit tests, so just verify that
    // Quit appears as a valid function call.
    assert_eq!(interpret("Head[Quit]").unwrap(), "Symbol");
  }
}

mod names {
  use super::*;

  #[test]
  fn empty_env() {
    assert_eq!(interpret("Names[]").unwrap(), "{}");
  }

  #[test]
  fn lists_variables() {
    assert_eq!(interpret("x = 1; y = 2; Names[\"*\"]").unwrap(), "{x, y}");
  }

  #[test]
  fn lists_functions() {
    assert_eq!(interpret("f[x_] := x^2; Names[\"*\"]").unwrap(), "{f}");
  }

  #[test]
  fn pattern_filter() {
    assert_eq!(
      interpret("abc = 1; abd = 2; xyz = 3; Names[\"ab*\"]").unwrap(),
      "{abc, abd}"
    );
  }

  #[test]
  fn no_match() {
    assert_eq!(interpret("a = 1; Names[\"z*\"]").unwrap(), "{}");
  }
}

mod unique {
  use super::*;

  #[test]
  fn unique_no_args() {
    // Unique[] generates $nnn
    let result = interpret("Unique[]").unwrap();
    assert!(result.starts_with('$'));
    let num: u64 = result[1..].parse().unwrap();
    assert!(num > 0);
  }

  #[test]
  fn unique_with_symbol() {
    // Unique[x] generates x$nnn
    let result = interpret("Unique[x]").unwrap();
    assert!(result.starts_with("x$"));
    let num: u64 = result[2..].parse().unwrap();
    assert!(num > 0);
  }

  #[test]
  fn unique_with_string() {
    // Unique["hello"] generates hellonnn
    let result = interpret("Unique[\"hello\"]").unwrap();
    assert!(result.starts_with("hello"));
    let num_str = &result[5..];
    let num: u64 = num_str.parse().unwrap();
    assert!(num > 0);
  }

  #[test]
  fn unique_list() {
    // Unique[{a, b}] generates list of unique symbols
    let result = interpret("Unique[{a, b}]").unwrap();
    assert!(result.starts_with('{'));
    assert!(result.contains("a$"));
    assert!(result.contains("b$"));
  }

  #[test]
  fn unique_successive_different() {
    // Two calls to Unique give different symbols
    let result = interpret("{Unique[x], Unique[x]}").unwrap();
    // Parse the two symbols
    let inner = &result[1..result.len() - 1]; // strip { }
    let parts: Vec<&str> = inner.split(", ").collect();
    assert_eq!(parts.len(), 2);
    assert_ne!(parts[0], parts[1]);
  }

  #[test]
  fn unique_symbolic_unevaluated() {
    // Non-symbol, non-string, non-list args return unevaluated
    assert_eq!(interpret("Unique[1]").unwrap(), "Unique[1]");
  }
}

mod entity {
  use super::*;

  #[test]
  fn entity_preserves_string_args() {
    // Entity preserves string quotes in output
    assert_eq!(
      interpret("Entity[\"Country\", \"France\"]").unwrap(),
      "Entity[\"Country\", \"France\"]"
    );
  }

  #[test]
  fn entity_single_arg() {
    assert_eq!(
      interpret("Entity[\"Country\"]").unwrap(),
      "Entity[\"Country\"]"
    );
  }

  #[test]
  fn entity_head() {
    assert_eq!(
      interpret("Head[Entity[\"Country\", \"France\"]]").unwrap(),
      "Entity"
    );
  }

  #[test]
  fn entity_attributes() {
    assert_eq!(
      interpret("Attributes[Entity]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  #[test]
  fn entity_stays_symbolic() {
    // Entity expressions are inert - they evaluate to themselves
    assert_eq!(
      interpret("Entity[\"City\", \"Paris\"]").unwrap(),
      "Entity[\"City\", \"Paris\"]"
    );
  }

  #[test]
  fn entity_mixed_args() {
    // Entity with mixed arg types preserves strings but evaluates others
    assert_eq!(
      interpret("Entity[\"Planet\", \"Mars\"]").unwrap(),
      "Entity[\"Planet\", \"Mars\"]"
    );
  }
}

mod image_size {
  use super::*;

  #[test]
  fn image_size_is_symbol() {
    // ImageSize evaluates to itself as a symbol
    assert_eq!(interpret("ImageSize").unwrap(), "ImageSize");
  }

  #[test]
  fn image_size_attributes() {
    assert_eq!(interpret("Attributes[ImageSize]").unwrap(), "{Protected}");
  }

  #[test]
  fn image_size_head() {
    assert_eq!(interpret("Head[ImageSize]").unwrap(), "Symbol");
  }

  #[test]
  fn image_size_in_rule() {
    // ImageSize used as option name in a Rule
    assert_eq!(interpret("ImageSize -> 300").unwrap(), "ImageSize -> 300");
  }

  #[test]
  fn image_size_in_list_of_rules() {
    assert_eq!(
      interpret("{ImageSize -> 400, PlotRange -> All}").unwrap(),
      "{ImageSize -> 400, PlotRange -> All}"
    );
  }
}

mod font_size {
  use super::*;

  #[test]
  fn font_size_is_symbol() {
    assert_eq!(interpret("FontSize").unwrap(), "FontSize");
  }

  #[test]
  fn font_size_attributes() {
    assert_eq!(interpret("Attributes[FontSize]").unwrap(), "{Protected}");
  }

  #[test]
  fn font_size_head() {
    assert_eq!(interpret("Head[FontSize]").unwrap(), "Symbol");
  }

  #[test]
  fn font_size_in_rule() {
    assert_eq!(interpret("FontSize -> 14").unwrap(), "FontSize -> 14");
  }

  #[test]
  fn font_size_in_style() {
    assert_eq!(
      interpret("Style[\"hello\", FontSize -> 24]").unwrap(),
      "Style[hello, FontSize -> 24]"
    );
  }
}

mod reals {
  use super::*;

  #[test]
  fn reals_evaluates_to_itself() {
    assert_eq!(interpret("Reals").unwrap(), "Reals");
  }

  #[test]
  fn reals_head() {
    assert_eq!(interpret("Head[Reals]").unwrap(), "Symbol");
  }

  #[test]
  fn reals_attributes() {
    assert_eq!(
      interpret("Attributes[Reals]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  #[test]
  fn element_integer_in_reals() {
    assert_eq!(interpret("Element[5, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_rational_in_reals() {
    assert_eq!(interpret("Element[3/7, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_real_in_reals() {
    assert_eq!(interpret("Element[2.5, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_pi_in_reals() {
    assert_eq!(interpret("Element[Pi, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_complex_not_in_reals() {
    assert_eq!(interpret("Element[2 + 3 I, Reals]").unwrap(), "False");
  }
}

mod font_family {
  use super::*;

  #[test]
  fn font_family_evaluates_to_itself() {
    assert_eq!(interpret("FontFamily").unwrap(), "FontFamily");
  }

  #[test]
  fn font_family_head() {
    assert_eq!(interpret("Head[FontFamily]").unwrap(), "Symbol");
  }

  #[test]
  fn font_family_attributes() {
    assert_eq!(interpret("Attributes[FontFamily]").unwrap(), "{Protected}");
  }

  #[test]
  fn font_family_in_rule() {
    assert_eq!(
      interpret("FontFamily -> \"Helvetica\"").unwrap(),
      "FontFamily -> Helvetica"
    );
  }

  #[test]
  fn font_family_in_style() {
    assert_eq!(
      interpret("Style[\"hello\", FontFamily -> \"Arial\"]").unwrap(),
      "Style[hello, FontFamily -> Arial]"
    );
  }
}

mod thick {
  use super::*;

  #[test]
  fn thick_evaluates_to_itself() {
    assert_eq!(interpret("Thick").unwrap(), "Thickness[Large]");
  }

  #[test]
  fn thick_head() {
    assert_eq!(interpret("Head[Thick]").unwrap(), "Thickness");
  }

  #[test]
  fn thick_attributes() {
    assert_eq!(
      interpret("Attributes[Thick]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  #[test]
  fn thick_in_graphics_directive_list() {
    // Thick should be usable in a Graphics directive list
    assert_eq!(
      interpret("Graphics[{Thick, Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn thick_in_plot_style() {
    // Thick can be used as a PlotStyle option value
    assert_eq!(
      interpret("Plot[Sin[x], {x, 0, 1}, PlotStyle -> Thick]").unwrap(),
      "-Graphics-"
    );
  }
}

mod dashed {
  use super::*;

  #[test]
  fn dashed_evaluates() {
    assert_eq!(interpret("Dashed").unwrap(), "Dashing[{Small, Small}]");
  }

  #[test]
  fn dotted_evaluates() {
    assert_eq!(interpret("Dotted").unwrap(), "Dashing[{0, Small}]");
  }

  #[test]
  fn dot_dashed_evaluates() {
    assert_eq!(
      interpret("DotDashed").unwrap(),
      "Dashing[{0, Small, Small, Small}]"
    );
  }

  #[test]
  fn dashed_head() {
    assert_eq!(interpret("Head[Dashed]").unwrap(), "Dashing");
  }

  #[test]
  fn dashed_in_graphics() {
    assert_eq!(
      interpret("Graphics[{Dashed, Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn dotted_in_graphics() {
    assert_eq!(
      interpret("Graphics[{Dotted, Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn dot_dashed_in_graphics() {
    assert_eq!(
      interpret("Graphics[{DotDashed, Line[{{0, 0}, {1, 1}}]}]").unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn dashing_with_named_sizes() {
    assert_eq!(
      interpret("Dashing[{Small, Small}]").unwrap(),
      "Dashing[{Small, Small}]"
    );
  }
}

mod base_style {
  use super::*;

  #[test]
  fn base_style_evaluates_to_itself() {
    assert_eq!(interpret("BaseStyle").unwrap(), "BaseStyle");
  }

  #[test]
  fn base_style_head() {
    assert_eq!(interpret("Head[BaseStyle]").unwrap(), "Symbol");
  }

  #[test]
  fn base_style_attributes() {
    assert_eq!(interpret("Attributes[BaseStyle]").unwrap(), "{Protected}");
  }

  #[test]
  fn base_style_in_rule() {
    assert_eq!(
      interpret("BaseStyle -> {FontSize -> 14}").unwrap(),
      "BaseStyle -> {FontSize -> 14}"
    );
  }

  #[test]
  fn base_style_in_graphics() {
    assert_eq!(
      interpret("Graphics[{Disk[]}, BaseStyle -> {Red}]").unwrap(),
      "-Graphics-"
    );
  }
}

mod activate {
  use super::*;

  #[test]
  fn basic_plus() {
    assert_eq!(interpret("Activate[Inactive[Plus][1, 2, 3]]").unwrap(), "6");
  }

  #[test]
  fn with_sin() {
    assert_eq!(interpret("Activate[Inactive[Sin][Pi/2]]").unwrap(), "1");
  }

  #[test]
  fn nested_in_expression() {
    assert_eq!(
      interpret("Activate[a + Inactive[Plus][1, 2]]").unwrap(),
      "3 + a"
    );
  }

  #[test]
  fn with_filter() {
    // Only activate Plus, not Times
    assert_eq!(
      interpret("Activate[Inactive[Plus][1, 2] + Inactive[Times][3, 4], Plus]")
        .unwrap(),
      "3 + Inactive[Times][3, 4]"
    );
  }

  #[test]
  fn inactive_preserved_without_activate() {
    assert_eq!(
      interpret("Inactive[Plus][1, 2]").unwrap(),
      "Inactive[Plus][1, 2]"
    );
  }

  #[test]
  fn with_integrate() {
    assert_eq!(
      interpret("Activate[Inactive[Integrate][x^2, x]]").unwrap(),
      "x^3/3"
    );
  }
}

mod area {
  use super::*;

  #[test]
  fn unit_disk() {
    assert_eq!(interpret("Area[Disk[]]").unwrap(), "Pi");
  }

  #[test]
  fn disk_with_radius() {
    assert_eq!(interpret("Area[Disk[{0, 0}, 5]]").unwrap(), "25*Pi");
  }

  #[test]
  fn elliptical_disk() {
    assert_eq!(interpret("Area[Disk[{0, 0}, {3, 2}]]").unwrap(), "6*Pi");
  }

  #[test]
  fn unit_rectangle() {
    assert_eq!(interpret("Area[Rectangle[]]").unwrap(), "1");
  }

  #[test]
  fn rectangle_with_bounds() {
    assert_eq!(interpret("Area[Rectangle[{0, 0}, {3, 4}]]").unwrap(), "12");
  }

  #[test]
  fn triangle() {
    assert_eq!(
      interpret("Area[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn polygon() {
    assert_eq!(
      interpret("Area[Polygon[{{0, 0}, {4, 0}, {4, 3}, {0, 3}}]]").unwrap(),
      "12"
    );
  }

  #[test]
  fn circle_undefined() {
    assert_eq!(interpret("Area[Circle[]]").unwrap(), "Undefined");
  }

  #[test]
  fn symbolic_radius() {
    assert_eq!(interpret("Area[Disk[{0, 0}, r]]").unwrap(), "Pi*r^2");
  }

  #[test]
  fn default_triangle() {
    assert_eq!(interpret("Area[Triangle[]]").unwrap(), "1/2");
  }
}

mod arc_length {
  use super::*;

  #[test]
  fn unit_circle() {
    assert_eq!(interpret("ArcLength[Circle[]]").unwrap(), "2*Pi");
  }

  #[test]
  fn circle_with_radius() {
    assert_eq!(interpret("ArcLength[Circle[{0, 0}, r]]").unwrap(), "2*Pi*r");
  }

  #[test]
  fn circle_numeric_radius() {
    assert_eq!(interpret("ArcLength[Circle[{0, 0}, 5]]").unwrap(), "10*Pi");
  }

  #[test]
  fn line_two_points() {
    assert_eq!(interpret("ArcLength[Line[{{0, 0}, {3, 4}}]]").unwrap(), "5");
  }

  #[test]
  fn line_multi_segment() {
    assert_eq!(
      interpret("ArcLength[Line[{{0, 0}, {1, 0}, {1, 1}}]]").unwrap(),
      "2"
    );
  }

  #[test]
  fn polygon_undefined() {
    assert_eq!(
      interpret("ArcLength[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn disk_undefined() {
    assert_eq!(interpret("ArcLength[Disk[]]").unwrap(), "Undefined");
  }

  #[test]
  fn triangle_undefined() {
    assert_eq!(
      interpret("ArcLength[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "Undefined"
    );
  }
}

mod perimeter {
  use super::*;

  #[test]
  fn unit_square_polygon() {
    assert_eq!(
      interpret("Perimeter[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "4"
    );
  }

  #[test]
  fn rectangle() {
    assert_eq!(
      interpret("Perimeter[Rectangle[{0, 0}, {3, 4}]]").unwrap(),
      "14"
    );
  }

  #[test]
  fn unit_rectangle() {
    assert_eq!(interpret("Perimeter[Rectangle[]]").unwrap(), "4");
  }

  #[test]
  fn triangle() {
    assert_eq!(
      interpret("Perimeter[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "2 + Sqrt[2]"
    );
  }

  #[test]
  fn disk() {
    assert_eq!(interpret("Perimeter[Disk[{0, 0}, r]]").unwrap(), "2*Pi*r");
  }

  #[test]
  fn unit_disk() {
    assert_eq!(interpret("Perimeter[Disk[]]").unwrap(), "2*Pi");
  }

  #[test]
  fn circle() {
    assert_eq!(
      interpret("Perimeter[Circle[{0, 0}, 3]]").unwrap(),
      "Undefined"
    );
  }
}

mod region_centroid {
  use super::*;

  #[test]
  fn point() {
    assert_eq!(
      interpret("RegionCentroid[Point[{3, 4}]]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn unit_disk() {
    assert_eq!(interpret("RegionCentroid[Disk[]]").unwrap(), "{0, 0}");
  }

  #[test]
  fn disk_with_center() {
    assert_eq!(
      interpret("RegionCentroid[Disk[{3, 4}, 2]]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn disk_symbolic() {
    assert_eq!(
      interpret("RegionCentroid[Disk[{a, b}, r]]").unwrap(),
      "{a, b}"
    );
  }

  #[test]
  fn unit_rectangle() {
    assert_eq!(
      interpret("RegionCentroid[Rectangle[]]").unwrap(),
      "{1/2, 1/2}"
    );
  }

  #[test]
  fn rectangle_with_bounds() {
    assert_eq!(
      interpret("RegionCentroid[Rectangle[{0, 0}, {2, 3}]]").unwrap(),
      "{1, 3/2}"
    );
  }

  #[test]
  fn rectangle_symbolic() {
    assert_eq!(
      interpret("RegionCentroid[Rectangle[{a, b}, {c, d}]]").unwrap(),
      "{(a + c)/2, (b + d)/2}"
    );
  }

  #[test]
  fn triangle_basic() {
    assert_eq!(
      interpret("RegionCentroid[Triangle[{{0, 0}, {1, 0}, {0, 1}}]]").unwrap(),
      "{1/3, 1/3}"
    );
  }

  #[test]
  fn polygon_square() {
    assert_eq!(
      interpret("RegionCentroid[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]")
        .unwrap(),
      "{1/2, 1/2}"
    );
  }

  #[test]
  fn polygon_trapezoid() {
    assert_eq!(
      interpret("RegionCentroid[Polygon[{{0,0},{2,0},{3,1},{1,1}}]]").unwrap(),
      "{3/2, 1/2}"
    );
  }

  #[test]
  fn line_two_points() {
    assert_eq!(
      interpret("RegionCentroid[Line[{{0, 0}, {1, 1}}]]").unwrap(),
      "{1/2, 1/2}"
    );
  }

  #[test]
  fn ball_3d() {
    assert_eq!(
      interpret("RegionCentroid[Ball[{1, 2, 3}, 5]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn circle_center() {
    assert_eq!(
      interpret("RegionCentroid[Circle[{2, 3}, 1]]").unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn unevaluated_unknown() {
    assert_eq!(
      interpret("RegionCentroid[foo]").unwrap(),
      "RegionCentroid[foo]"
    );
  }
}

mod triangle {
  use super::*;

  #[test]
  fn default_form() {
    assert_eq!(
      interpret("Triangle[]").unwrap(),
      "Triangle[{{0, 0}, {1, 0}, {0, 1}}]"
    );
  }

  #[test]
  fn explicit_vertices() {
    assert_eq!(
      interpret("Triangle[{{0, 0}, {3, 0}, {0, 4}}]").unwrap(),
      "Triangle[{{0, 0}, {3, 0}, {0, 4}}]"
    );
  }

  #[test]
  fn area() {
    assert_eq!(
      interpret("Area[Triangle[{{0, 0}, {3, 0}, {0, 4}}]]").unwrap(),
      "6"
    );
  }
}

mod plus_minus {
  use super::*;

  #[test]
  fn unary() {
    assert_eq!(interpret("PlusMinus[3]").unwrap(), "\u{00B1}3");
  }

  #[test]
  fn binary() {
    assert_eq!(interpret("PlusMinus[a, b]").unwrap(), "a \u{00B1} b");
  }
}

mod circle_times {
  use super::*;

  #[test]
  fn binary() {
    assert_eq!(interpret("CircleTimes[a, b]").unwrap(), "a \u{2297} b");
  }

  #[test]
  fn ternary() {
    assert_eq!(
      interpret("CircleTimes[a, b, c]").unwrap(),
      "a \u{2297} b \u{2297} c"
    );
  }
}

mod wedge {
  use super::*;

  #[test]
  fn binary() {
    assert_eq!(interpret("Wedge[a, b]").unwrap(), "a \u{22C0} b");
  }
}

mod del {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Del[f]").unwrap(), "\u{2207}f");
  }
}

mod complete_graph {
  use super::*;

  #[test]
  fn vertices() {
    assert_eq!(
      interpret("VertexList[CompleteGraph[4]]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn edge_count_3() {
    assert_eq!(
      interpret("Length[EdgeList[CompleteGraph[3]]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn edge_count_4() {
    assert_eq!(
      interpret("Length[EdgeList[CompleteGraph[4]]]").unwrap(),
      "6"
    );
  }

  #[test]
  fn adjacency_matrix() {
    assert_eq!(
      interpret("AdjacencyMatrix[CompleteGraph[3]]").unwrap(),
      "{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}"
    );
  }
}

mod adjacency_matrix {
  use super::*;

  #[test]
  fn directed_cycle() {
    assert_eq!(
      interpret("AdjacencyMatrix[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}"
    );
  }

  #[test]
  fn directed_chain() {
    assert_eq!(
      interpret("AdjacencyMatrix[Graph[{a -> b, b -> c}]]").unwrap(),
      "{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}}"
    );
  }

  #[test]
  fn single_edge() {
    assert_eq!(
      interpret("AdjacencyMatrix[Graph[{1 -> 2}]]").unwrap(),
      "{{0, 1}, {0, 0}}"
    );
  }

  #[test]
  fn self_loop() {
    assert_eq!(
      interpret("AdjacencyMatrix[Graph[{1 -> 1, 1 -> 2}]]").unwrap(),
      "{{1, 1}, {0, 0}}"
    );
  }
}

mod dispatch {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("Dispatch[{a -> 1, b -> 2}]").unwrap(),
      "Dispatch[{a -> 1, b -> 2}]"
    );
  }

  #[test]
  fn replace_all() {
    assert_eq!(
      interpret("{a, b, c} /. Dispatch[{a -> 1, b -> 2}]").unwrap(),
      "{1, 2, c}"
    );
  }

  #[test]
  fn replace_repeated() {
    assert_eq!(interpret("a /. Dispatch[{a -> b, b -> c}]").unwrap(), "b");
  }
}

mod cycles {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("Cycles[{{1, 2, 3}}]").unwrap(),
      "Cycles[{{1, 2, 3}}]"
    );
  }

  #[test]
  fn permutation_list_single_cycle() {
    assert_eq!(
      interpret("PermutationList[Cycles[{{1, 3, 2}}]]").unwrap(),
      "{3, 1, 2}"
    );
  }

  #[test]
  fn permutation_list_transposition() {
    assert_eq!(
      interpret("PermutationList[Cycles[{{1, 2}}]]").unwrap(),
      "{2, 1}"
    );
  }

  #[test]
  fn permutation_list_identity() {
    assert_eq!(interpret("PermutationList[Cycles[{}]]").unwrap(), "{}");
  }

  #[test]
  fn permutation_list_two_cycles() {
    assert_eq!(
      interpret("PermutationList[Cycles[{{1, 2}, {3, 4}}]]").unwrap(),
      "{2, 1, 4, 3}"
    );
  }

  #[test]
  fn permutation_list_with_length() {
    assert_eq!(
      interpret("PermutationList[Cycles[{{1, 2}}], 4]").unwrap(),
      "{2, 1, 3, 4}"
    );
  }
}

mod exists {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Exists[x, x > 0]").unwrap(), "Exists[x, x > 0]");
  }

  #[test]
  fn with_list_vars() {
    assert_eq!(
      interpret("Exists[{x, y}, x + y > 0]").unwrap(),
      "Exists[{x, y}, x + y > 0]"
    );
  }

  #[test]
  fn with_condition() {
    assert_eq!(
      interpret("Exists[x, x > 0 && x < 1, x^2 < 1]").unwrap(),
      "Exists[x, x > 0 && x < 1, x^2 < 1]"
    );
  }

  #[test]
  fn for_all() {
    assert_eq!(
      interpret("ForAll[x, x^2 >= 0]").unwrap(),
      "ForAll[x, x^2 >= 0]"
    );
  }
}

mod pattern_constructs {
  use super::*;

  #[test]
  fn pattern_sequence() {
    assert_eq!(
      interpret("PatternSequence[a, b]").unwrap(),
      "PatternSequence[a, b]"
    );
  }

  #[test]
  fn start_of_string() {
    assert_eq!(interpret("StartOfString").unwrap(), "StartOfString");
  }

  #[test]
  fn end_of_string() {
    assert_eq!(interpret("EndOfString").unwrap(), "EndOfString");
  }

  #[test]
  fn whitespace() {
    assert_eq!(interpret("Whitespace").unwrap(), "Whitespace");
  }
}

mod moving_average {
  use super::*;

  #[test]
  fn basic_integers() {
    assert_eq!(
      interpret("MovingAverage[{1, 2, 3, 4, 5}, 3]").unwrap(),
      "{2, 3, 4}"
    );
  }

  #[test]
  fn window_two() {
    assert_eq!(
      interpret("MovingAverage[{1, 2, 3, 4, 5, 6}, 2]").unwrap(),
      "{3/2, 5/2, 7/2, 9/2, 11/2}"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("MovingAverage[{a, b, c, d}, 2]").unwrap(),
      "{(a + b)/2, (b + c)/2, (c + d)/2}"
    );
  }

  #[test]
  fn window_equals_length() {
    assert_eq!(interpret("MovingAverage[{1, 2, 3}, 3]").unwrap(), "{2}");
  }
}

mod adjacency_graph_from_matrix {
  use super::*;

  #[test]
  fn undirected_symmetric() {
    assert_eq!(
      interpret("AdjacencyGraph[{{0, 1, 1}, {1, 0, 0}, {1, 0, 0}}]").unwrap(),
      "Graph[{1, 2, 3}, {UndirectedEdge[1, 2], UndirectedEdge[1, 3]}]"
    );
  }

  #[test]
  fn directed_asymmetric() {
    assert_eq!(
      interpret("AdjacencyGraph[{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}]").unwrap(),
      "Graph[{1, 2, 3}, {DirectedEdge[1, 2], DirectedEdge[2, 3], DirectedEdge[3, 1]}]"
    );
  }

  #[test]
  fn with_named_vertices() {
    let result = interpret(
      "AdjacencyGraph[{\"a\", \"b\", \"c\"}, {{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}]",
    )
    .unwrap();
    assert!(result.contains("UndirectedEdge[a, b]"));
    assert!(result.contains("UndirectedEdge[b, c]"));
  }
}

mod circle_plus {
  use super::*;

  #[test]
  fn binary() {
    assert_eq!(interpret("CirclePlus[a, b]").unwrap(), "a \u{2295} b");
  }

  #[test]
  fn ternary() {
    assert_eq!(
      interpret("CirclePlus[a, b, c]").unwrap(),
      "a \u{2295} b \u{2295} c"
    );
  }
}

mod piecewise_expand {
  use super::*;

  #[test]
  fn min_two_args() {
    assert_eq!(
      interpret("PiecewiseExpand[Min[x, y]]").unwrap(),
      "Piecewise[{{x, x - y <= 0}}, y]"
    );
  }

  #[test]
  fn max_two_args() {
    assert_eq!(
      interpret("PiecewiseExpand[Max[x, y]]").unwrap(),
      "Piecewise[{{x, x - y >= 0}}, y]"
    );
  }

  #[test]
  fn unit_step() {
    assert_eq!(
      interpret("PiecewiseExpand[UnitStep[x]]").unwrap(),
      "Piecewise[{{1, x >= 0}}, 0]"
    );
  }

  #[test]
  fn clip_default() {
    assert_eq!(
      interpret("PiecewiseExpand[Clip[x]]").unwrap(),
      "Piecewise[{{-1, x < -1}, {1, x > 1}}, x]"
    );
  }

  #[test]
  fn clip_custom_bounds() {
    assert_eq!(
      interpret("PiecewiseExpand[Clip[x, {0, 10}]]").unwrap(),
      "Piecewise[{{0, x < 0}, {10, x > 10}}, x]"
    );
  }

  #[test]
  fn unsupported_returns_unchanged() {
    // Non-expandable functions pass through unchanged
    assert_eq!(interpret("PiecewiseExpand[Sin[x]]").unwrap(), "Sin[x]");
  }
}

mod circle_points {
  use super::*;

  #[test]
  fn single_point() {
    assert_eq!(interpret("CirclePoints[1]").unwrap(), "{{0, 1}}");
  }

  #[test]
  fn two_points() {
    assert_eq!(interpret("CirclePoints[2]").unwrap(), "{{1, 0}, {-1, 0}}");
  }

  #[test]
  fn three_points() {
    let result = interpret("CirclePoints[3]").unwrap();
    assert!(result.contains("Sqrt[3]/2") && result.contains("-1/2"));
  }

  #[test]
  fn four_points() {
    let result = interpret("CirclePoints[4]").unwrap();
    assert!(result.contains("1/Sqrt[2]"));
  }

  #[test]
  fn six_points() {
    let result = interpret("CirclePoints[6]").unwrap();
    assert!(result.contains("{1, 0}") && result.contains("{-1, 0}"));
  }
}

mod parallel_map {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("ParallelMap[f, {1, 2, 3}]").unwrap(),
      "{f[1], f[2], f[3]}"
    );
  }

  #[test]
  fn with_function() {
    assert_eq!(
      interpret("ParallelMap[#^2 &, {1, 2, 3}]").unwrap(),
      "{1, 4, 9}"
    );
  }
}

mod path_graph {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("PathGraph[{1, 2, 3, 4}]").unwrap(),
      "Graph[{1, 2, 3, 4}, {UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[3, 4]}]"
    );
  }

  #[test]
  fn with_symbols() {
    assert_eq!(
      interpret("PathGraph[{a, b, c}]").unwrap(),
      "Graph[{a, b, c}, {UndirectedEdge[a, b], UndirectedEdge[b, c]}]"
    );
  }
}

mod vertex_count {
  use super::*;

  #[test]
  fn complete_graph() {
    assert_eq!(interpret("VertexCount[CompleteGraph[5]]").unwrap(), "5");
  }

  #[test]
  fn path_graph() {
    assert_eq!(interpret("VertexCount[PathGraph[{1, 2, 3}]]").unwrap(), "3");
  }
}

mod edge_count {
  use super::*;

  #[test]
  fn complete_graph() {
    assert_eq!(interpret("EdgeCount[CompleteGraph[5]]").unwrap(), "10");
  }

  #[test]
  fn path_graph() {
    assert_eq!(
      interpret("EdgeCount[PathGraph[{1, 2, 3, 4}]]").unwrap(),
      "3"
    );
  }
}

mod vertex_degree {
  use super::*;

  #[test]
  fn all_degrees() {
    assert_eq!(
      interpret("VertexDegree[CompleteGraph[4]]").unwrap(),
      "{3, 3, 3, 3}"
    );
  }

  #[test]
  fn single_vertex() {
    assert_eq!(interpret("VertexDegree[CompleteGraph[5], 1]").unwrap(), "4");
  }

  #[test]
  fn path_graph_degrees() {
    assert_eq!(
      interpret("VertexDegree[PathGraph[{1, 2, 3, 4}]]").unwrap(),
      "{1, 2, 2, 1}"
    );
  }
}

mod graph_embedding {
  use super::*;

  #[test]
  fn complete_graph_3() {
    assert_eq!(
      interpret("GraphEmbedding[CompleteGraph[3]]").unwrap(),
      "{{-0.8660254037844388, -0.5}, {0.8660254037844384, -0.5}, {0., 1.}}"
    );
  }

  #[test]
  fn complete_graph_4() {
    assert_eq!(
      interpret("GraphEmbedding[CompleteGraph[4]]").unwrap(),
      "{{-1., 0.}, {0., -1.}, {1., 0.}, {0., 1.}}"
    );
  }

  #[test]
  fn directed_graph() {
    assert_eq!(
      interpret("GraphEmbedding[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "{{-0.8660254037844388, -0.5}, {0.8660254037844384, -0.5}, {0., 1.}}"
    );
  }

  #[test]
  fn single_edge() {
    assert_eq!(
      interpret("GraphEmbedding[Graph[{1 -> 2}]]").unwrap(),
      "{{0., -1.}, {0., 1.}}"
    );
  }

  #[test]
  fn path_graph() {
    assert_eq!(
      interpret("GraphEmbedding[PathGraph[{1, 2, 3, 4}]]").unwrap(),
      "{{-1., 0.}, {0., -1.}, {1., 0.}, {0., 1.}}"
    );
  }

  #[test]
  fn non_graph_unevaluated() {
    assert_eq!(
      interpret("GraphEmbedding[42]").unwrap(),
      "GraphEmbedding[42]"
    );
  }

  #[test]
  fn with_method_argument() {
    // With explicit "CircularEmbedding" method, same result
    assert_eq!(
      interpret("GraphEmbedding[CompleteGraph[3], \"CircularEmbedding\"]")
        .unwrap(),
      "{{-0.8660254037844388, -0.5}, {0.8660254037844384, -0.5}, {0., 1.}}"
    );
  }

  #[test]
  fn length_matches_vertex_count() {
    assert_eq!(
      interpret("Length[GraphEmbedding[CompleteGraph[5]]]").unwrap(),
      "5"
    );
  }

  #[test]
  fn each_coordinate_is_pair() {
    assert_eq!(
      interpret("Length[GraphEmbedding[CompleteGraph[3]][[1]]]").unwrap(),
      "2"
    );
  }
}

mod bezier_function {
  use super::*;

  #[test]
  fn quadratic_at_half() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}][0.5]").unwrap(),
      "{1., 0.5}"
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}][0]").unwrap(),
      "{0., 0.}"
    );
  }

  #[test]
  fn at_one() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}][1]").unwrap(),
      "{2., 0.}"
    );
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0},{3,2}}][0.25]").unwrap(),
      "{0.75, 0.453125}"
    );
  }

  #[test]
  fn one_dimensional() {
    assert_eq!(
      interpret("BezierFunction[{{0},{1},{4}}][0.5]").unwrap(),
      "{1.5}"
    );
  }

  #[test]
  fn with_rational_parameter() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}][1/3]").unwrap(),
      "{0.6666666666666667, 0.4444444444444445}"
    );
  }

  #[test]
  fn unevaluated_symbolic() {
    assert_eq!(
      interpret("BezierFunction[{{0,0},{1,1},{2,0}}]").unwrap(),
      "BezierFunction[{{0, 0}, {1, 1}, {2, 0}}]"
    );
  }

  #[test]
  fn non_numeric_argument() {
    // With symbolic argument, returns unevaluated
    let result = interpret("BezierFunction[{{0,0},{1,1},{2,0}}][t]").unwrap();
    assert!(result.contains("BezierFunction"));
  }
}

mod elliptic_theta_prime {
  use super::*;

  #[test]
  fn theta1() {
    assert_eq!(
      interpret("EllipticThetaPrime[1, 0.5, 0.1]").unwrap(),
      "0.9846106693769313"
    );
  }

  #[test]
  fn theta2() {
    assert_eq!(
      interpret("EllipticThetaPrime[2, 0.5, 0.1]").unwrap(),
      "-0.5728609100292524"
    );
  }

  #[test]
  fn theta3() {
    assert_eq!(
      interpret("EllipticThetaPrime[3, 0.5, 0.1]").unwrap(),
      "-0.33731583355805805"
    );
  }

  #[test]
  fn theta4() {
    assert_eq!(
      interpret("EllipticThetaPrime[4, 0.5, 0.1]").unwrap(),
      "0.33586095767513946"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("EllipticThetaPrime[1, z, q]").unwrap(),
      "EllipticThetaPrime[1, z, q]"
    );
  }
}

mod bessel_j_zero {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("BesselJZero[0, 1]").unwrap(), "BesselJZero[0, 1]");
  }

  #[test]
  fn numeric_j0_1() {
    assert_eq!(
      interpret("N[BesselJZero[0, 1]]").unwrap(),
      "2.404825557695773"
    );
  }

  #[test]
  fn numeric_j0_2() {
    assert_eq!(
      interpret("N[BesselJZero[0, 2]]").unwrap(),
      "5.520078110286309"
    );
  }

  #[test]
  fn numeric_j0_3() {
    assert_eq!(
      interpret("N[BesselJZero[0, 3]]").unwrap(),
      "8.653727912911014"
    );
  }

  #[test]
  fn numeric_j1_1() {
    assert_eq!(
      interpret("N[BesselJZero[1, 1]]").unwrap(),
      "3.8317059702075125"
    );
  }

  #[test]
  fn is_actual_zero() {
    // BesselJ at the zero should be approximately 0
    assert_eq!(
      interpret("Chop[BesselJ[0, N[BesselJZero[0, 1]]]]").unwrap(),
      "0"
    );
  }
}

mod file_exists_q {
  use super::*;

  #[test]
  fn existing_path() {
    assert_eq!(interpret("FileExistsQ[\"/tmp\"]").unwrap(), "True");
  }

  #[test]
  fn nonexistent_path() {
    assert_eq!(
      interpret("FileExistsQ[\"/nonexistent_path_xyz\"]").unwrap(),
      "False"
    );
  }
}

mod unit_vector {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("UnitVector[3, 2]").unwrap(), "{0, 1, 0}");
  }

  #[test]
  fn first_element() {
    assert_eq!(interpret("UnitVector[5, 1]").unwrap(), "{1, 0, 0, 0, 0}");
  }

  #[test]
  fn shorthand() {
    assert_eq!(interpret("UnitVector[2]").unwrap(), "{0, 1}");
  }

  #[test]
  fn shorthand_first() {
    assert_eq!(interpret("UnitVector[1]").unwrap(), "{1, 0}");
  }
}

mod permute {
  use super::*;

  #[test]
  fn list_form() {
    assert_eq!(
      interpret("Permute[{a, b, c, d}, {3, 1, 4, 2}]").unwrap(),
      "{b, d, a, c}"
    );
  }

  #[test]
  fn cycles_form() {
    assert_eq!(
      interpret("Permute[{a, b, c}, Cycles[{{1, 3, 2}}]]").unwrap(),
      "{b, c, a}"
    );
  }

  #[test]
  fn identity() {
    assert_eq!(
      interpret("Permute[{a, b, c}, {1, 2, 3}]").unwrap(),
      "{a, b, c}"
    );
  }
}

mod delete_file {
  use super::*;

  #[test]
  fn delete_existing() {
    assert_eq!(
      interpret(
        "WriteString[\"/tmp/test_delete_woxi.txt\", \"hello\"]; DeleteFile[\"/tmp/test_delete_woxi.txt\"]; FileExistsQ[\"/tmp/test_delete_woxi.txt\"]"
      )
      .unwrap(),
      "False"
    );
  }
}

mod delete_missing {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("DeleteMissing[{1, Missing[], 3, Missing[\"x\"], 5}]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn no_missing() {
    assert_eq!(interpret("DeleteMissing[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn all_missing() {
    assert_eq!(
      interpret("DeleteMissing[{Missing[], Missing[]}]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("DeleteMissing[{}]").unwrap(), "{}");
  }
}

mod angle_bracket {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("AngleBracket[a, b, c]").unwrap(),
      "\u{2329} a, b, c \u{232A}"
    );
  }

  #[test]
  fn single() {
    assert_eq!(interpret("AngleBracket[x]").unwrap(), "\u{2329} x \u{232A}");
  }
}

mod face_grids {
  use super::*;

  #[test]
  fn symbol() {
    assert_eq!(interpret("FaceGrids").unwrap(), "FaceGrids");
  }

  #[test]
  fn with_args() {
    assert_eq!(
      interpret("FaceGrids[1, 2, 3]").unwrap(),
      "FaceGrids[1, 2, 3]"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(interpret("Attributes[FaceGrids]").unwrap(), "{Protected}");
  }

  #[test]
  fn face_grids_style_symbol() {
    assert_eq!(interpret("FaceGridsStyle").unwrap(), "FaceGridsStyle");
  }
}

mod padding {
  use super::*;

  #[test]
  fn symbol() {
    assert_eq!(interpret("Padding").unwrap(), "Padding");
  }

  #[test]
  fn with_args() {
    assert_eq!(interpret("Padding[1, 2, 3]").unwrap(), "Padding[1, 2, 3]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Padding]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod library_function_load {
  use super::*;

  #[test]
  fn symbol() {
    assert_eq!(
      interpret("LibraryFunctionLoad").unwrap(),
      "LibraryFunctionLoad"
    );
  }

  #[test]
  fn with_args() {
    assert_eq!(
      interpret("LibraryFunctionLoad[a, b, c, d]").unwrap(),
      "LibraryFunctionLoad[a, b, c, d]"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[LibraryFunctionLoad]").unwrap(),
      "{Protected}"
    );
  }
}

mod point_legend {
  use super::*;

  #[test]
  fn symbol() {
    assert_eq!(interpret("PointLegend").unwrap(), "PointLegend");
  }

  #[test]
  fn with_args() {
    assert_eq!(interpret("PointLegend[x, y]").unwrap(), "PointLegend[x, y]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[PointLegend]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod legend_label {
  use super::*;

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(
      interpret("LegendLabel[\"test\"]").unwrap(),
      "LegendLabel[test]"
    );
  }

  #[test]
  fn with_none() {
    assert_eq!(interpret("LegendLabel[None]").unwrap(), "LegendLabel[None]");
  }

  #[test]
  fn bare_symbol() {
    assert_eq!(interpret("LegendLabel").unwrap(), "LegendLabel");
  }
}

mod cells {
  use super::*;

  #[test]
  fn symbol() {
    assert_eq!(interpret("Cells").unwrap(), "Cells");
  }

  #[test]
  fn with_args() {
    assert_eq!(interpret("Cells[a, b]").unwrap(), "Cells[a, b]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Cells]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod hypergeometric_0f1_regularized {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("Hypergeometric0F1Regularized[a, z]").unwrap(),
      "Hypergeometric0F1Regularized[a, z]"
    );
  }

  #[test]
  fn zero_at_a_zero_z_zero() {
    // 1/Gamma(0) = 0
    assert_eq!(
      interpret("Hypergeometric0F1Regularized[0, 0]").unwrap(),
      "0"
    );
  }

  #[test]
  fn one_at_positive_integer_z_zero() {
    assert_eq!(
      interpret("Hypergeometric0F1Regularized[1, 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Hypergeometric0F1Regularized[2, 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn numeric_a1() {
    // Hypergeometric0F1Regularized[1, 1.0] ≈ 2.279585302336067
    let result = interpret("Hypergeometric0F1Regularized[1, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.279585302336067).abs() < 1e-10);
  }

  #[test]
  fn numeric_a2() {
    // Hypergeometric0F1Regularized[2, 1.0] ≈ 1.5906368546373288
    let result = interpret("Hypergeometric0F1Regularized[2, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.5906368546373288).abs() < 1e-10);
  }

  #[test]
  fn numeric_a3() {
    // Hypergeometric0F1Regularized[3, 2.0] ≈ 0.9287588901146092
    let result = interpret("Hypergeometric0F1Regularized[3, 2.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.9287588901146092).abs() < 1e-10);
  }

  #[test]
  fn numeric_a_half() {
    // Hypergeometric0F1Regularized[0.5, 1.0] ≈ 2.122591620177637
    let result = interpret("Hypergeometric0F1Regularized[0.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.122591620177637).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_z() {
    // Hypergeometric0F1Regularized[3, -2.0] ≈ 0.2397640410755054
    let result = interpret("Hypergeometric0F1Regularized[3, -2.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.2397640410755054).abs() < 1e-10);
  }

  #[test]
  fn numeric_a_zero_z_nonzero() {
    // Hypergeometric0F1Regularized[0, 1.0] ≈ 1.5906368546373288
    let result = interpret("Hypergeometric0F1Regularized[0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.5906368546373288).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_z() {
    // Hypergeometric0F1Regularized[1, 5.0] ≈ 17.05777785336906
    let result = interpret("Hypergeometric0F1Regularized[1, 5.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 17.05777785336906).abs() < 1e-8);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Hypergeometric0F1Regularized]").unwrap(),
      "{Listable, NumericFunction, Protected}"
    );
  }
}

mod struve_h {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("StruveH[n, z]").unwrap(), "StruveH[n, z]");
  }

  #[test]
  fn zero_arg_integer_order() {
    assert_eq!(interpret("StruveH[0, 0]").unwrap(), "0");
    assert_eq!(interpret("StruveH[1, 0]").unwrap(), "0");
    assert_eq!(interpret("StruveH[2, 0]").unwrap(), "0");
  }

  #[test]
  fn numeric_order_0() {
    // StruveH[0, 1.0] ≈ 0.5686566270482879
    let result = interpret("StruveH[0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.5686566270482879).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_1() {
    // StruveH[1, 1.0] ≈ 0.1984573362019444
    let result = interpret("StruveH[1, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.1984573362019444).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_2() {
    // StruveH[2, 1.0] ≈ 0.040464636144794626
    let result = interpret("StruveH[2, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.040464636144794626).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_z() {
    // StruveH[0, 10.0] ≈ 0.11874368368750424
    let result = interpret("StruveH[0, 10.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.11874368368750424).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_z_order_1() {
    // StruveH[1, 10.0] ≈ 0.8918324920945468
    let result = interpret("StruveH[1, 10.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8918324920945468).abs() < 1e-10);
  }

  #[test]
  fn numeric_fractional_order() {
    // StruveH[0.5, 3.0] ≈ 0.9167076867564138
    let result = interpret("StruveH[0.5, 3.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.9167076867564138).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_fractional_order() {
    // StruveH[-0.5, 3.0] ≈ 0.06500818287737578
    let result = interpret("StruveH[-0.5, 3.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.06500818287737578).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_integer_order() {
    // StruveH[-1, 1.0] ≈ 0.43816243616563694
    let result = interpret("N[StruveH[-1, 1.0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.43816243616563694).abs() < 1e-10);
  }

  #[test]
  fn numeric_high_order() {
    // StruveH[5, 10.0] ≈ 7.644815648083951
    let result = interpret("StruveH[5, 10.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 7.644815648083951).abs() < 1e-8);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[StruveH]").unwrap(),
      "{Listable, NumericFunction, Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod struve_l {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("StruveL[n, z]").unwrap(), "StruveL[n, z]");
  }

  #[test]
  fn zero_arg_integer_order() {
    assert_eq!(interpret("StruveL[0, 0]").unwrap(), "0");
    assert_eq!(interpret("StruveL[1, 0]").unwrap(), "0");
    assert_eq!(interpret("StruveL[2, 0]").unwrap(), "0");
  }

  #[test]
  fn numeric_order_0() {
    // StruveL[0, 1.0] ≈ 0.7102431859378909
    let result = interpret("StruveL[0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.7102431859378909).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_1() {
    // StruveL[1, 1.0] ≈ 0.22676438105580865
    let result = interpret("StruveL[1, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.22676438105580865).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_2() {
    // StruveL[2, 1.0] ≈ 0.044507833037079836
    let result = interpret("StruveL[2, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.044507833037079836).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_z() {
    // StruveL[0, 2.5] ≈ 3.0112116937373057
    let result = interpret("StruveL[0, 2.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 3.0112116937373057).abs() < 1e-10);
  }

  #[test]
  fn numeric_half_order() {
    // StruveL[1/2, 1.0] ≈ 0.4333156537901021
    let result = interpret("StruveL[0.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.4333156537901021).abs() < 1e-10);
  }

  #[test]
  fn numeric_neg_half_order() {
    // StruveL[-1/2, 1.0] ≈ 0.9376748882454876
    let result = interpret("StruveL[-0.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.9376748882454876).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_z() {
    // StruveL[0, -1.0] ≈ -0.7102431859378909
    let result = interpret("StruveL[0, -1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-0.7102431859378909)).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_integer_order() {
    // StruveL[-1, 1.0] ≈ 0.86338415342339
    let result = interpret("StruveL[-1.0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.86338415342339).abs() < 1e-10);
  }

  #[test]
  fn numeric_small_z() {
    // StruveL[0, 0.5] ≈ 0.32724069939418077
    let result = interpret("StruveL[0, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.32724069939418077).abs() < 1e-10);
  }

  #[test]
  fn numeric_n_pi() {
    // N[StruveL[0, Pi]] ≈ 5.256595137877723
    let result = interpret("N[StruveL[0, Pi]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 5.256595137877723).abs() < 1e-8);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[StruveL]").unwrap(),
      "{Listable, NumericFunction, Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod generating_function {
  use super::*;

  #[test]
  fn constant_sequence() {
    assert_eq!(
      interpret("GeneratingFunction[1, n, x]").unwrap(),
      "(1 - x)^(-1)"
    );
  }

  #[test]
  fn constant_a() {
    assert_eq!(
      interpret("GeneratingFunction[a, n, x]").unwrap(),
      "a/(1 - x)"
    );
  }

  #[test]
  fn identity_sequence() {
    assert_eq!(
      interpret("GeneratingFunction[n, n, x]").unwrap(),
      "x/(1 - x)^2"
    );
  }

  #[test]
  fn n_squared() {
    assert_eq!(
      interpret("GeneratingFunction[n^2, n, x]").unwrap(),
      "(-x - x^2)/(-1 + x)^3"
    );
  }

  #[test]
  fn n_cubed() {
    assert_eq!(
      interpret("GeneratingFunction[n^3, n, x]").unwrap(),
      "(x + 4*x^2 + x^3)/(-1 + x)^4"
    );
  }

  #[test]
  fn n_to_4() {
    assert_eq!(
      interpret("GeneratingFunction[n^4, n, x]").unwrap(),
      "(-x - 11*x^2 - 11*x^3 - x^4)/(-1 + x)^5"
    );
  }

  #[test]
  fn exponential_2n() {
    assert_eq!(
      interpret("GeneratingFunction[2^n, n, x]").unwrap(),
      "(1 - 2*x)^(-1)"
    );
  }

  #[test]
  fn alternating() {
    assert_eq!(
      interpret("GeneratingFunction[(-1)^n, n, x]").unwrap(),
      "(1 + x)^(-1)"
    );
  }

  #[test]
  fn exponential_a_n() {
    assert_eq!(
      interpret("GeneratingFunction[a^n, n, x]").unwrap(),
      "(1 - a*x)^(-1)"
    );
  }

  #[test]
  fn reciprocal_factorial() {
    assert_eq!(
      interpret("GeneratingFunction[1/Factorial[n], n, x]").unwrap(),
      "E^x"
    );
  }

  #[test]
  fn reciprocal_n_plus_1() {
    assert_eq!(
      interpret("GeneratingFunction[1/(n+1), n, x]").unwrap(),
      "-(Log[1 - x]/x)"
    );
  }

  #[test]
  fn binomial_n_2() {
    assert_eq!(
      interpret("GeneratingFunction[Binomial[n, 2], n, x]").unwrap(),
      "x^2/(1 - x)^3"
    );
  }

  #[test]
  fn binomial_2n_n() {
    assert_eq!(
      interpret("GeneratingFunction[Binomial[2*n, n], n, x]").unwrap(),
      "1/Sqrt[1 - 4*x]"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("GeneratingFunction[f[n], n, x]").unwrap(),
      "GeneratingFunction[f[n], n, x]"
    );
  }

  #[test]
  fn shifted_sequence() {
    assert_eq!(
      interpret("GeneratingFunction[f[n + 1], n, x]").unwrap(),
      "(GeneratingFunction[f[n], n, x] - f[0])/x"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[GeneratingFunction]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod before {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Before[3]").unwrap(), "Before[3]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Before[3]]").unwrap(), "Before");
  }

  #[test]
  fn string_arg() {
    assert_eq!(interpret("Before[\"cat\"]").unwrap(), "Before[cat]");
  }
}

#[cfg(test)]
mod complexity_function {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("ComplexityFunction[3]").unwrap(),
      "ComplexityFunction[3]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[ComplexityFunction]").unwrap(), "Symbol");
  }

  #[test]
  fn multiple_args() {
    assert_eq!(
      interpret("ComplexityFunction[x, y]").unwrap(),
      "ComplexityFunction[x, y]"
    );
  }
}

#[cfg(test)]
mod compilation_options {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("CompilationOptions[1, 2, 3]").unwrap(),
      "CompilationOptions[1, 2, 3]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[CompilationOptions]").unwrap(), "Symbol");
  }

  #[test]
  fn no_args() {
    assert_eq!(
      interpret("CompilationOptions[]").unwrap(),
      "CompilationOptions[]"
    );
  }
}

#[cfg(test)]
mod absolute_dashing {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("AbsoluteDashing[{1, 2}]").unwrap(),
      "AbsoluteDashing[{1, 2}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[AbsoluteDashing]").unwrap(), "Symbol");
  }

  #[test]
  fn three_element_list() {
    assert_eq!(
      interpret("AbsoluteDashing[{1, 2, 3}]").unwrap(),
      "AbsoluteDashing[{1, 2, 3}]"
    );
  }
}

#[cfg(test)]
mod data_reversed {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("DataReversed[1, 2]").unwrap(),
      "DataReversed[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[DataReversed]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod axes_edge {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("AxesEdge[1, 2]").unwrap(), "AxesEdge[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[AxesEdge]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod tagging_rules {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("TaggingRules[1, 2]").unwrap(),
      "TaggingRules[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TaggingRules]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod rationals {
  use super::*;

  #[test]
  fn unevaluated_with_args() {
    assert_eq!(interpret("Rationals[1, 2]").unwrap(), "Rationals[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Rationals]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod file_function {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("File[1, 2]").unwrap(), "File[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[File]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod bode_plot {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("BodePlot[x]").unwrap(), "BodePlot[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[BodePlot]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod delaunay_mesh {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("DelaunayMesh[x]").unwrap(), "DelaunayMesh[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[DelaunayMesh]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod complex_region_plot {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("ComplexRegionPlot[x]").unwrap(),
      "ComplexRegionPlot[x]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[ComplexRegionPlot]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod accounting_form {
  use super::*;

  #[test]
  fn unevaluated_real() {
    assert_eq!(
      interpret("AccountingForm[123.45]").unwrap(),
      "AccountingForm[123.45]"
    );
  }

  #[test]
  fn unevaluated_integer() {
    assert_eq!(
      interpret("AccountingForm[42]").unwrap(),
      "AccountingForm[42]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[AccountingForm[123.45]]").unwrap(),
      "AccountingForm"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[AccountingForm]").unwrap(),
      "{NHoldRest, Protected}"
    );
  }
}

#[cfg(test)]
mod session_time {
  use super::*;

  #[test]
  fn returns_real() {
    assert_eq!(interpret("Head[SessionTime[]]").unwrap(), "Real");
  }

  #[test]
  fn non_negative() {
    let result = interpret("SessionTime[] >= 0").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn monotonically_increasing() {
    let result =
      interpret("t1 = SessionTime[]; t2 = SessionTime[]; t2 >= t1").unwrap();
    assert_eq!(result, "True");
  }
}

#[cfg(test)]
mod function_interpolation {
  use super::*;

  #[test]
  fn returns_interpolating_function() {
    assert_eq!(
      interpret("Head[FunctionInterpolation[Sin[x], {x, 0, 6.28}]]").unwrap(),
      "InterpolatingFunction"
    );
  }

  #[test]
  fn evaluate_at_zero() {
    let result =
      interpret("FunctionInterpolation[Sin[x], {x, 0, 6.28}][0.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(val.abs() < 0.01, "Expected ~0 at x=0, got {}", val);
  }

  #[test]
  fn evaluate_accuracy() {
    let result =
      interpret("Abs[FunctionInterpolation[Sin[x], {x, 0, 2*Pi}][1.0] - Sin[1.0]] < 0.001")
        .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn evaluate_polynomial() {
    let result =
      interpret("Abs[FunctionInterpolation[x^2, {x, 0, 5}][3.0] - 9.0] < 0.01")
        .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[FunctionInterpolation]").unwrap(),
      "{HoldAll, Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod cmyk_color {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("CMYKColor[0.5, 0.2, 0.8, 0.1]").unwrap(),
      "CMYKColor[0.5, 0.2, 0.8, 0.1]"
    );
  }

  #[test]
  fn integer_args() {
    assert_eq!(
      interpret("CMYKColor[1, 0, 0, 0]").unwrap(),
      "CMYKColor[1, 0, 0, 0]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[CMYKColor[0.5, 0.2, 0.8, 0.1]]").unwrap(),
      "CMYKColor"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[CMYKColor]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod net_graph {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("NetGraph[{1, 2}, {1 -> 2}]").unwrap(),
      "NetGraph[{1, 2}, {1 -> 2}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[NetGraph[{1, 2}, {1 -> 2}]]").unwrap(),
      "NetGraph"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[NetGraph]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod scaling_transform {
  use super::*;

  #[test]
  fn basic_2d() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3}]").unwrap(),
      "TransformationFunction[{{2, 0, 0}, {0, 3, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn basic_1d() {
    assert_eq!(
      interpret("ScalingTransform[{2}]").unwrap(),
      "TransformationFunction[{{2, 0}, {0, 1}}]"
    );
  }

  #[test]
  fn basic_3d() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3, 4}]").unwrap(),
      "TransformationFunction[{{2, 0, 0, 0}, {0, 3, 0, 0}, {0, 0, 4, 0}, {0, 0, 0, 1}}]"
    );
  }

  #[test]
  fn with_center() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3}, {1, 1}]").unwrap(),
      "TransformationFunction[{{2, 0, -1}, {0, 3, -2}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn apply_basic() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3}][{1, 1}]").unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn apply_1d() {
    assert_eq!(interpret("ScalingTransform[{2}][{5}]").unwrap(), "{10}");
  }

  #[test]
  fn apply_with_center() {
    assert_eq!(
      interpret("ScalingTransform[{2, 3}, {1, 1}][{2, 2}]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("ScalingTransform[{sx, sy}]").unwrap(),
      "TransformationFunction[{{sx, 0, 0}, {0, sy, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ScalingTransform]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod connected_components {
  use super::*;

  #[test]
  fn undirected_two_components() {
    let result =
      interpret("ConnectedComponents[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[4, 5]}]]")
        .unwrap();
    assert_eq!(result, "{{1, 2, 3}, {4, 5}}");
  }

  #[test]
  fn undirected_single_component() {
    let result = interpret(
      "ConnectedComponents[Graph[{UndirectedEdge[a, b], UndirectedEdge[c, d], UndirectedEdge[b, c]}]]",
    )
    .unwrap();
    assert_eq!(result, "{{a, b, c, d}}");
  }

  #[test]
  fn directed_no_cycles() {
    // Each vertex is its own SCC when there are no cycles
    let result =
      interpret("ConnectedComponents[Graph[{1 -> 2, 2 -> 3, 4 -> 5}]]")
        .unwrap();
    // Should have 5 singleton components
    assert!(result.contains("{1}"));
    assert!(result.contains("{2}"));
    assert!(result.contains("{3}"));
    assert!(result.contains("{4}"));
    assert!(result.contains("{5}"));
  }

  #[test]
  fn directed_cycle() {
    let result =
      interpret("ConnectedComponents[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]")
        .unwrap();
    // All three vertices form one SCC
    assert!(
      result.contains("1") && result.contains("2") && result.contains("3")
    );
    // Should be a single component
    assert_eq!(result.matches('{').count(), 2); // outer { + one inner {
  }

  #[test]
  fn directed_mixed() {
    let result = interpret(
      "ConnectedComponents[Graph[{1 -> 2, 2 -> 1, 3 -> 4, 4 -> 3, 1 -> 3}]]",
    )
    .unwrap();
    // {1,2} and {3,4} are separate SCCs (1->3 doesn't create a cycle between them)
    assert!(result.contains("1") && result.contains("2"));
    assert!(result.contains("3") && result.contains("4"));
  }

  #[test]
  fn complete_graph() {
    let result = interpret("ConnectedComponents[CompleteGraph[4]]").unwrap();
    assert_eq!(result, "{{1, 2, 3, 4}}");
  }

  #[test]
  fn unevaluated_non_graph() {
    assert_eq!(
      interpret("ConnectedComponents[foo]").unwrap(),
      "ConnectedComponents[foo]"
    );
  }
}

#[cfg(test)]
mod proportional {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Proportional[a, b]").unwrap(), "a \u{221D} b");
  }

  #[test]
  fn multiple_args() {
    assert_eq!(
      interpret("Proportional[1, 2, 3]").unwrap(),
      "1 \u{221D} 2 \u{221D} 3"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[Proportional[a, b]]").unwrap(),
      "Proportional"
    );
  }
}

#[cfg(test)]
mod decimal_form {
  use super::*;

  #[test]
  fn unevaluated_real() {
    assert_eq!(
      interpret("DecimalForm[1234567.89]").unwrap(),
      "DecimalForm[1.23456789*^6]"
    );
  }

  #[test]
  fn unevaluated_with_digits() {
    assert_eq!(
      interpret("DecimalForm[1234567.89, 9]").unwrap(),
      "DecimalForm[1.23456789*^6, 9]"
    );
  }

  #[test]
  fn unevaluated_small_real() {
    assert_eq!(
      interpret("DecimalForm[0.000004]").unwrap(),
      "DecimalForm[4.*^-6]"
    );
  }

  #[test]
  fn unevaluated_integer() {
    assert_eq!(interpret("DecimalForm[42]").unwrap(), "DecimalForm[42]");
  }

  #[test]
  fn unevaluated_rational() {
    assert_eq!(interpret("DecimalForm[1/3]").unwrap(), "DecimalForm[1/3]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[DecimalForm[3.14]]").unwrap(), "DecimalForm");
  }
}

#[cfg(test)]
mod monomial_list {
  use super::*;

  #[test]
  fn basic_two_variables() {
    assert_eq!(
      interpret("MonomialList[x^2 + 3*x*y + y^3, {x, y}]").unwrap(),
      "{x^2, 3*x*y, y^3}"
    );
  }

  #[test]
  fn three_terms_two_variables() {
    assert_eq!(
      interpret("MonomialList[x^3 + 2*x^2*y + y^2, {x, y}]").unwrap(),
      "{x^3, 2*x^2*y, y^2}"
    );
  }

  #[test]
  fn three_variables() {
    assert_eq!(
      interpret("MonomialList[a + b + c, {a, b, c}]").unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn constant_polynomial() {
    assert_eq!(interpret("MonomialList[5, {x}]").unwrap(), "{5}");
  }

  #[test]
  fn expansion_needed() {
    assert_eq!(
      interpret("MonomialList[(x + y)^3, {x, y}]").unwrap(),
      "{x^3, 3*x^2*y, 3*x*y^2, y^3}"
    );
  }

  #[test]
  fn single_variable() {
    assert_eq!(interpret("MonomialList[x, {x}]").unwrap(), "{x}");
  }

  #[test]
  fn single_variable_polynomial() {
    assert_eq!(
      interpret("MonomialList[x^3 + 2*x + 1, {x}]").unwrap(),
      "{x^3, 2*x, 1}"
    );
  }
}

#[cfg(test)]
mod airy_bi {
  use super::*;

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("AiryBi[x]").unwrap(), "AiryBi[x]");
  }

  #[test]
  fn numeric_zero() {
    let result = interpret("N[AiryBi[0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.6149266274460007).abs() < 1e-10);
  }

  #[test]
  fn numeric_positive() {
    let result = interpret("N[AiryBi[1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.2074235949528715).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result = interpret("N[AiryBi[-1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.10399738949694468).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_positive() {
    let result = interpret("N[AiryBi[10]]").unwrap();
    let val: f64 = result.replace("*^", "e").parse().unwrap();
    assert!((val - 4.556411535482249e8).abs() / 4.556411535482249e8 < 1e-6);
  }

  #[test]
  fn numeric_large_negative() {
    let result = interpret("N[AiryBi[-10]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-0.3146798296438386)).abs() < 1e-6);
  }

  #[test]
  fn direct_real_input() {
    let result = interpret("AiryBi[0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8542770431031554).abs() < 1e-10);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[AiryBi]").unwrap(),
      "{Listable, NumericFunction, Protected}"
    );
  }
}

#[cfg(test)]
mod square_wave {
  use super::*;

  #[test]
  fn positive_first_half() {
    assert_eq!(interpret("SquareWave[0.1]").unwrap(), "1");
  }

  #[test]
  fn positive_at_zero() {
    assert_eq!(interpret("SquareWave[0]").unwrap(), "1");
  }

  #[test]
  fn negative_second_half() {
    assert_eq!(interpret("SquareWave[0.5]").unwrap(), "-1");
  }

  #[test]
  fn negative_at_0_7() {
    assert_eq!(interpret("SquareWave[0.7]").unwrap(), "-1");
  }

  #[test]
  fn periodic_wrapping() {
    assert_eq!(interpret("SquareWave[1.3]").unwrap(), "1");
  }

  #[test]
  fn negative_argument() {
    assert_eq!(interpret("SquareWave[-0.3]").unwrap(), "-1");
  }

  #[test]
  fn exact_rational_input() {
    assert_eq!(interpret("SquareWave[1/4]").unwrap(), "1");
    assert_eq!(interpret("SquareWave[1/2]").unwrap(), "-1");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("SquareWave[x]").unwrap(), "SquareWave[x]");
  }

  #[test]
  fn multi_level_two_values() {
    assert_eq!(interpret("SquareWave[{1/3, 2/3}, 0.2]").unwrap(), "2/3");
    assert_eq!(interpret("SquareWave[{1/3, 2/3}, 0.5]").unwrap(), "1/3");
    assert_eq!(interpret("SquareWave[{1/3, 2/3}, 0.8]").unwrap(), "1/3");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[SquareWave]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod parabolic_cylinder_d {
  use super::*;

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(
      interpret("ParabolicCylinderD[n, x]").unwrap(),
      "ParabolicCylinderD[n, x]"
    );
  }

  #[test]
  fn numeric_d0_0() {
    let result = interpret("N[ParabolicCylinderD[0, 0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
  }

  #[test]
  fn numeric_d1_0() {
    let result = interpret("N[ParabolicCylinderD[1, 0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(val.abs() < 1e-10);
  }

  #[test]
  fn numeric_d2_1_5() {
    let result = interpret("ParabolicCylinderD[2, 1.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.7122285309136538).abs() < 1e-8);
  }

  #[test]
  fn numeric_negative_order() {
    let result = interpret("ParabolicCylinderD[-1, 2.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.15501307659733082).abs() < 1e-8);
  }

  #[test]
  fn numeric_half_integer_order() {
    let result = interpret("ParabolicCylinderD[0.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8422032440698396).abs() < 1e-10);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ParabolicCylinderD]").unwrap(),
      "{Listable, NumericFunction, Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod positive_reals {
  use super::*;

  #[test]
  fn symbol_passthrough() {
    assert_eq!(interpret("PositiveReals").unwrap(), "PositiveReals");
  }

  #[test]
  fn element_positive_integer() {
    assert_eq!(interpret("Element[3, PositiveReals]").unwrap(), "True");
  }

  #[test]
  fn element_negative_integer() {
    assert_eq!(interpret("Element[-3, PositiveReals]").unwrap(), "False");
  }

  #[test]
  fn element_zero() {
    assert_eq!(interpret("Element[0, PositiveReals]").unwrap(), "False");
  }

  #[test]
  fn element_positive_real() {
    assert_eq!(interpret("Element[2.5, PositiveReals]").unwrap(), "True");
  }

  #[test]
  fn element_positive_rational() {
    assert_eq!(interpret("Element[1/3, PositiveReals]").unwrap(), "True");
  }
}

#[cfg(test)]
mod positive_integers {
  use super::*;

  #[test]
  fn symbol_passthrough() {
    assert_eq!(interpret("PositiveIntegers").unwrap(), "PositiveIntegers");
  }

  #[test]
  fn element_positive() {
    assert_eq!(interpret("Element[3, PositiveIntegers]").unwrap(), "True");
  }

  #[test]
  fn element_negative() {
    assert_eq!(interpret("Element[-3, PositiveIntegers]").unwrap(), "False");
  }

  #[test]
  fn element_zero() {
    assert_eq!(interpret("Element[0, PositiveIntegers]").unwrap(), "False");
  }

  #[test]
  fn element_real_not_integer() {
    assert_eq!(
      interpret("Element[3.5, PositiveIntegers]").unwrap(),
      "False"
    );
  }

  #[test]
  fn element_non_negative_integers() {
    assert_eq!(
      interpret("Element[0, NonNegativeIntegers]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Element[5, NonNegativeIntegers]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Element[-1, NonNegativeIntegers]").unwrap(),
      "False"
    );
  }

  #[test]
  fn element_negative_reals() {
    assert_eq!(interpret("Element[-2.5, NegativeReals]").unwrap(), "True");
    assert_eq!(interpret("Element[1, NegativeReals]").unwrap(), "False");
  }
}

#[cfg(test)]
mod discriminant {
  use super::*;

  #[test]
  fn quadratic_monic() {
    assert_eq!(
      interpret("Discriminant[x^2 + b*x + c, x]").unwrap(),
      "b^2 - 4*c"
    );
  }

  #[test]
  fn quadratic_general() {
    assert_eq!(
      interpret("Discriminant[a*x^2 + b*x + c, x]").unwrap(),
      "b^2 - 4*a*c"
    );
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("Discriminant[x^3 + p*x + q, x]").unwrap(),
      "-4*p^3 - 27*q^2"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(interpret("Discriminant[x^2 - 4, x]").unwrap(), "16");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Discriminant]").unwrap(),
      "{Listable, Protected}"
    );
  }
}

#[cfg(test)]
mod triangle_wave {
  use super::*;

  #[test]
  fn integer_zero() {
    assert_eq!(interpret("TriangleWave[0]").unwrap(), "0");
  }

  #[test]
  fn integer_one() {
    assert_eq!(interpret("TriangleWave[1]").unwrap(), "0");
  }

  #[test]
  fn integer_negative() {
    assert_eq!(interpret("TriangleWave[-1]").unwrap(), "0");
  }

  #[test]
  fn integer_large() {
    assert_eq!(interpret("TriangleWave[100]").unwrap(), "0");
  }

  #[test]
  fn float_quarter() {
    assert_eq!(interpret("TriangleWave[0.25]").unwrap(), "1.");
  }

  #[test]
  fn float_half() {
    assert_eq!(interpret("TriangleWave[0.5]").unwrap(), "0.");
  }

  #[test]
  fn float_three_quarter() {
    assert_eq!(interpret("TriangleWave[0.75]").unwrap(), "-1.");
  }

  #[test]
  fn float_negative() {
    assert_eq!(interpret("TriangleWave[-0.25]").unwrap(), "-1.");
  }

  #[test]
  fn rational_quarter() {
    assert_eq!(interpret("TriangleWave[1/4]").unwrap(), "1");
  }

  #[test]
  fn rational_third() {
    assert_eq!(interpret("TriangleWave[1/3]").unwrap(), "2/3");
  }

  #[test]
  fn rational_three_eighths() {
    assert_eq!(interpret("TriangleWave[3/8]").unwrap(), "1/2");
  }

  #[test]
  fn periodic_wrapping() {
    assert_eq!(interpret("TriangleWave[2]").unwrap(), "0");
  }

  #[test]
  fn two_arg_scaling() {
    assert_eq!(interpret("TriangleWave[{2, 5}, 0.25]").unwrap(), "5.");
  }

  #[test]
  fn two_arg_unit() {
    assert_eq!(interpret("TriangleWave[{-1, 1}, 0.25]").unwrap(), "1.");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("TriangleWave[x]").unwrap(), "TriangleWave[x]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[TriangleWave]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod sawtooth_wave {
  use super::*;

  #[test]
  fn integer_zero() {
    assert_eq!(interpret("SawtoothWave[0]").unwrap(), "0");
  }

  #[test]
  fn integer_one() {
    assert_eq!(interpret("SawtoothWave[1]").unwrap(), "0");
  }

  #[test]
  fn integer_negative() {
    assert_eq!(interpret("SawtoothWave[-3]").unwrap(), "0");
  }

  #[test]
  fn rational_quarter() {
    assert_eq!(interpret("SawtoothWave[1/4]").unwrap(), "1/4");
  }

  #[test]
  fn rational_third() {
    assert_eq!(interpret("SawtoothWave[1/3]").unwrap(), "1/3");
  }

  #[test]
  fn rational_three_quarters() {
    assert_eq!(interpret("SawtoothWave[3/4]").unwrap(), "3/4");
  }

  #[test]
  fn float_quarter() {
    assert_eq!(interpret("SawtoothWave[0.25]").unwrap(), "0.25");
  }

  #[test]
  fn float_half() {
    assert_eq!(interpret("SawtoothWave[0.5]").unwrap(), "0.5");
  }

  #[test]
  fn float_periodic() {
    assert_eq!(interpret("SawtoothWave[1.5]").unwrap(), "0.5");
  }

  #[test]
  fn float_negative() {
    assert_eq!(interpret("SawtoothWave[-0.3]").unwrap(), "0.7");
  }

  #[test]
  fn two_arg_scaled() {
    assert_eq!(interpret("SawtoothWave[{0, 10}, 0.3]").unwrap(), "3.");
  }

  #[test]
  fn two_arg_negative_range() {
    assert_eq!(interpret("SawtoothWave[{-1, 1}, 0.25]").unwrap(), "-0.5");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("SawtoothWave[x]").unwrap(), "SawtoothWave[x]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[SawtoothWave]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

#[cfg(test)]
mod long_right_arrow {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("LongRightArrow[x, y]").unwrap(), "x \u{27F6} y");
  }

  #[test]
  fn three_args() {
    assert_eq!(
      interpret("LongRightArrow[1, 2, 3]").unwrap(),
      "1 \u{27F6} 2 \u{27F6} 3"
    );
  }

  #[test]
  fn single_arg_unevaluated() {
    assert_eq!(interpret("LongRightArrow[x]").unwrap(), "LongRightArrow[x]");
  }

  #[test]
  fn no_args_unevaluated() {
    assert_eq!(interpret("LongRightArrow[]").unwrap(), "LongRightArrow[]");
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[LongRightArrow[x, y]]").unwrap(),
      "LongRightArrow"
    );
  }
}

#[cfg(test)]
mod geo_grid_position {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("GeoGridPosition[1, 2]").unwrap(),
      "GeoGridPosition[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[GeoGridPosition]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod opener_view {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("OpenerView[1, 2]").unwrap(), "OpenerView[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[OpenerView]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod ellipsoid_function {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("Ellipsoid[{0, 0}, {1, 2}]").unwrap(),
      "Ellipsoid[{0, 0}, {1, 2}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Ellipsoid]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod max_step_size {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("MaxStepSize[1, 2]").unwrap(), "MaxStepSize[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[MaxStepSize]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod radio_button_bar {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("RadioButtonBar[1, 2]").unwrap(),
      "RadioButtonBar[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[RadioButtonBar]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod step_monitor {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("StepMonitor[1, 2]").unwrap(), "StepMonitor[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[StepMonitor]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod thumbnail {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Thumbnail[x]").unwrap(), "Thumbnail[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Thumbnail]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod transformed_region {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("TransformedRegion[x, y]").unwrap(),
      "TransformedRegion[x, y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TransformedRegion]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod hypoexponential_distribution {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("HypoexponentialDistribution[{1, 2, 3}]").unwrap(),
      "HypoexponentialDistribution[{1, 2, 3}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[HypoexponentialDistribution]").unwrap(),
      "Symbol"
    );
  }
}

#[cfg(test)]
mod formula_lookup {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("FormulaLookup[x]").unwrap(), "FormulaLookup[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[FormulaLookup]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod geometric_distribution {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("GeometricDistribution[1/3]").unwrap(),
      "GeometricDistribution[1/3]"
    );
  }

  #[test]
  fn mean() {
    assert_eq!(interpret("Mean[GeometricDistribution[1/3]]").unwrap(), "2");
  }

  #[test]
  fn variance() {
    assert_eq!(
      interpret("Variance[GeometricDistribution[1/3]]").unwrap(),
      "6"
    );
  }

  #[test]
  fn pdf_at_5() {
    assert_eq!(
      interpret("PDF[GeometricDistribution[1/3], 5]").unwrap(),
      "32/729"
    );
  }

  #[test]
  fn cdf_at_5() {
    assert_eq!(
      interpret("CDF[GeometricDistribution[1/3], 5]").unwrap(),
      "665/729"
    );
  }

  #[test]
  fn pdf_at_0() {
    assert_eq!(
      interpret("PDF[GeometricDistribution[1/3], 0]").unwrap(),
      "1/3"
    );
  }

  #[test]
  fn cdf_at_0() {
    assert_eq!(
      interpret("CDF[GeometricDistribution[1/3], 0]").unwrap(),
      "1/3"
    );
  }

  #[test]
  fn standard_deviation() {
    assert_eq!(
      interpret("StandardDeviation[GeometricDistribution[1/3]]").unwrap(),
      "Sqrt[6]"
    );
  }
}

#[cfg(test)]
mod group_orbits {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("GroupOrbits[x, y]").unwrap(), "GroupOrbits[x, y]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[GroupOrbits]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod region_bounds {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("RegionBounds[x]").unwrap(), "RegionBounds[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[RegionBounds]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod tensor_contract {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("TensorContract[1, 2]").unwrap(),
      "TensorContract[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TensorContract]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod clear_system_cache {
  use super::*;

  #[test]
  fn returns_null() {
    assert_eq!(interpret("ClearSystemCache[]").unwrap(), "Null");
  }

  #[test]
  fn with_arg_returns_null() {
    assert_eq!(interpret("ClearSystemCache[\"Numeric\"]").unwrap(), "Null");
  }
}

#[cfg(test)]
mod locator_auto_create {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("LocatorAutoCreate[1, 2]").unwrap(),
      "LocatorAutoCreate[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[LocatorAutoCreate]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod legend_function {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("LegendFunction[1, 2]").unwrap(),
      "LegendFunction[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[LegendFunction]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod raster_size {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("RasterSize[1, 2]").unwrap(), "RasterSize[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[RasterSize]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod transformed_field {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("TransformedField[x, y, z]").unwrap(),
      "TransformedField[x, y, z]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TransformedField]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod jacobi_zeta {
  use super::*;

  #[test]
  fn zero_phi() {
    assert_eq!(interpret("JacobiZeta[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn zero_m() {
    assert_eq!(interpret("JacobiZeta[0.5, 0]").unwrap(), "0.");
  }

  #[test]
  fn numeric() {
    let result = interpret("JacobiZeta[0.5, 0.3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.06715126391766499).abs() < 1e-10);
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("JacobiZeta[x, m]").unwrap(), "JacobiZeta[x, m]");
  }
}

#[cfg(test)]
mod elliptic_e_incomplete {
  use super::*;

  #[test]
  fn two_arg_numeric() {
    let result = interpret("EllipticE[0.5, 0.3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.49399114472896827).abs() < 1e-10);
  }

  #[test]
  fn two_arg_zero_phi() {
    assert_eq!(interpret("EllipticE[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("EllipticE[x, m]").unwrap(), "EllipticE[x, m]");
  }
}

#[cfg(test)]
mod select_first {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("SelectFirst[{1, 2, 3, 4, 5}, EvenQ]").unwrap(),
      "2"
    );
  }

  #[test]
  fn not_found() {
    assert_eq!(
      interpret("SelectFirst[{1, 3, 5}, EvenQ]").unwrap(),
      "Missing[NotFound]"
    );
  }

  #[test]
  fn with_default() {
    assert_eq!(
      interpret("SelectFirst[{1, 3, 5}, EvenQ, \"none\"]").unwrap(),
      "none"
    );
  }

  #[test]
  fn with_pure_function() {
    assert_eq!(interpret("SelectFirst[{1, 2, 3, 4}, (#>2&)]").unwrap(), "3");
  }
}

#[cfg(test)]
mod conditioned {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Conditioned[1, 2]").unwrap(), "Conditioned[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Conditioned]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod geo_projection {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("GeoProjection[x, y]").unwrap(),
      "GeoProjection[x, y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[GeoProjection]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod sound_volume {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("SoundVolume[x, y]").unwrap(), "SoundVolume[x, y]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[SoundVolume]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod gradient_filter {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("GradientFilter[x, y]").unwrap(),
      "GradientFilter[x, y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[GradientFilter]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod region_nearest {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("RegionNearest[x, y]").unwrap(),
      "RegionNearest[x, y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[RegionNearest]").unwrap(), "Symbol");
  }
}

#[cfg(test)]
mod take_drop {
  use super::*;

  #[test]
  fn positive_n() {
    assert_eq!(
      interpret("TakeDrop[{a, b, c, d, e}, 3]").unwrap(),
      "{{a, b, c}, {d, e}}"
    );
  }

  #[test]
  fn negative_n() {
    assert_eq!(
      interpret("TakeDrop[{a, b, c, d, e}, -2]").unwrap(),
      "{{d, e}, {a, b, c}}"
    );
  }

  #[test]
  fn range_spec() {
    assert_eq!(
      interpret("TakeDrop[{a, b, c, d, e}, {2, 4}]").unwrap(),
      "{{b, c, d}, {a, e}}"
    );
  }

  #[test]
  fn zero() {
    assert_eq!(
      interpret("TakeDrop[{a, b, c, d, e}, 0]").unwrap(),
      "{{}, {a, b, c, d, e}}"
    );
  }
}

#[cfg(test)]
mod array_rules {
  use super::*;

  #[test]
  fn matrix() {
    assert_eq!(
      interpret("ArrayRules[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {_, _} -> 0}"
    );
  }

  #[test]
  fn vector() {
    assert_eq!(
      interpret("ArrayRules[{5, 0, 3, 0, 1}]").unwrap(),
      "{{1} -> 5, {3} -> 3, {5} -> 1, {_} -> 0}"
    );
  }

  #[test]
  fn all_zeros() {
    assert_eq!(interpret("ArrayRules[{0, 0, 0}]").unwrap(), "{{_} -> 0}");
  }

  #[test]
  fn custom_default() {
    assert_eq!(
      interpret("ArrayRules[{1, x, 1}, 1]").unwrap(),
      "{{2} -> x, {_} -> 1}"
    );
  }
}

// Batch tests for unevaluated wrappers
#[cfg(test)]
mod batch_unevaluated_wrappers {
  use super::*;

  #[test]
  fn tilde_tilde() {
    assert_eq!(interpret("TildeTilde[x]").unwrap(), "TildeTilde[x]");
  }
  #[test]
  fn notebook_close() {
    assert_eq!(interpret("NotebookClose[x]").unwrap(), "NotebookClose[x]");
  }
  #[test]
  fn failure() {
    assert_eq!(interpret("Failure[x]").unwrap(), "Failure[x]");
  }
  #[test]
  fn time_value() {
    assert_eq!(interpret("TimeValue[x]").unwrap(), "TimeValue[x]");
  }
  #[test]
  fn line_indent() {
    assert_eq!(interpret("LineIndent[x]").unwrap(), "LineIndent[x]");
  }
  #[test]
  fn layered_graph_plot() {
    assert_eq!(
      interpret("LayeredGraphPlot[x]").unwrap(),
      "LayeredGraphPlot[x]"
    );
  }
  #[test]
  fn word_character() {
    assert_eq!(interpret("WordCharacter[x]").unwrap(), "WordCharacter[x]");
  }
  #[test]
  fn reflection_transform() {
    assert_eq!(
      interpret("ReflectionTransform[x]").unwrap(),
      "ReflectionTransform[x]"
    );
  }
  #[test]
  fn bspline_basis() {
    assert_eq!(
      interpret("BSplineBasis[x, y]").unwrap(),
      "BSplineBasis[x, y]"
    );
  }
  #[test]
  fn parameter_mixture_distribution() {
    assert_eq!(
      interpret("ParameterMixtureDistribution[x, y]").unwrap(),
      "ParameterMixtureDistribution[x, y]"
    );
  }
  #[test]
  fn binary_read_list() {
    assert_eq!(
      interpret("BinaryReadList[x, y]").unwrap(),
      "BinaryReadList[x, y]"
    );
  }
  #[test]
  fn total_layer() {
    assert_eq!(interpret("TotalLayer[x]").unwrap(), "$Failed");
  }
  #[test]
  fn find_distribution_parameters() {
    assert_eq!(
      interpret("FindDistributionParameters[x, y]").unwrap(),
      "FindDistributionParameters[x, y]"
    );
  }
  #[test]
  fn find_path() {
    assert_eq!(interpret("FindPath[x, y, z]").unwrap(), "FindPath[x, y, z]");
  }
  #[test]
  fn find_peaks() {
    assert_eq!(interpret("FindPeaks[x]").unwrap(), "FindPeaks[x]");
  }
  #[test]
  fn nprobability() {
    assert_eq!(
      interpret("NProbability[x, y]").unwrap(),
      "NProbability[x, y]"
    );
  }
  #[test]
  fn net_encoder() {
    assert_eq!(interpret("NetEncoder[x]").unwrap(), "$Failed");
  }
  #[test]
  fn permutation_product() {
    assert_eq!(interpret("PermutationProduct[x]").unwrap(), "x");
  }
  #[test]
  fn syntax_information() {
    assert_eq!(interpret("SyntaxInformation[x]").unwrap(), "{}");
  }
}

#[cfg(test)]
mod cauchy_distribution {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("CauchyDistribution[0, 1]").unwrap(),
      "CauchyDistribution[0, 1]"
    );
  }

  #[test]
  fn pdf_at_zero() {
    assert_eq!(
      interpret("PDF[CauchyDistribution[0, 1], 0]").unwrap(),
      "1/Pi"
    );
  }

  #[test]
  fn cdf_at_zero() {
    assert_eq!(
      interpret("CDF[CauchyDistribution[0, 1], 0]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn cdf_at_one_numeric() {
    let result = interpret("N[CDF[CauchyDistribution[0, 1], 1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.75).abs() < 1e-10);
  }

  #[test]
  fn mean_indeterminate() {
    assert_eq!(
      interpret("Mean[CauchyDistribution[0, 1]]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn variance_indeterminate() {
    assert_eq!(
      interpret("Variance[CauchyDistribution[0, 1]]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn pdf_with_params_numeric() {
    let result = interpret("N[PDF[CauchyDistribution[2, 3], 0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 3.0 / (13.0 * std::f64::consts::PI);
    assert!((val - expected).abs() < 1e-10);
  }
}

#[cfg(test)]
mod batch_unevaluated_wrappers_2 {
  use super::*;

  #[test]
  fn dedekind_eta() {
    assert_eq!(interpret("DedekindEta[x]").unwrap(), "DedekindEta[x]");
  }
  #[test]
  fn pixel_value_positions() {
    assert_eq!(
      interpret("PixelValuePositions[x, y]").unwrap(),
      "PixelValuePositions[x, y]"
    );
  }
  #[test]
  fn weights() {
    assert_eq!(interpret("Weights[x]").unwrap(), "Weights[x]");
  }
  #[test]
  fn whitespace_character() {
    assert_eq!(
      interpret("WhitespaceCharacter[x]").unwrap(),
      "WhitespaceCharacter[x]"
    );
  }
  #[test]
  fn bar_chart_3d() {
    assert_eq!(interpret("BarChart3D[x]").unwrap(), "BarChart3D[x]");
  }
  #[test]
  fn vertical_slider() {
    assert_eq!(interpret("VerticalSlider[x]").unwrap(), "VerticalSlider[x]");
  }
  #[test]
  fn cycle_graph() {
    assert_eq!(interpret("CycleGraph[x]").unwrap(), "CycleGraph[x]");
  }
  #[test]
  fn over_dot() {
    assert_eq!(interpret("OverDot[x]").unwrap(), "OverDot[x]");
  }
  #[test]
  fn max_plot_points() {
    assert_eq!(interpret("MaxPlotPoints[x]").unwrap(), "MaxPlotPoints[x]");
  }
  #[test]
  fn launch_kernels() {
    assert_eq!(interpret("LaunchKernels[x]").unwrap(), "{}");
  }
  #[test]
  fn permutation_cycles() {
    assert_eq!(
      interpret("PermutationCycles[x]").unwrap(),
      "PermutationCycles[x]"
    );
  }
  #[test]
  fn animation_repetitions() {
    assert_eq!(
      interpret("AnimationRepetitions[x]").unwrap(),
      "AnimationRepetitions[x]"
    );
  }
  #[test]
  fn arma_process() {
    assert_eq!(interpret("ARMAProcess[x]").unwrap(), "ARMAProcess[x]");
  }
  #[test]
  fn file_name_take() {
    assert_eq!(interpret("FileNameTake[x]").unwrap(), "FileNameTake[x]");
  }
  #[test]
  fn undo_tracked_variables() {
    assert_eq!(
      interpret("UndoTrackedVariables[x]").unwrap(),
      "UndoTrackedVariables[x]"
    );
  }
  #[test]
  fn vector_color_function() {
    assert_eq!(
      interpret("VectorColorFunction[x]").unwrap(),
      "VectorColorFunction[x]"
    );
  }
  #[test]
  fn notebook_get() {
    assert_eq!(interpret("NotebookGet[x]").unwrap(), "NotebookGet[x]");
  }
  #[test]
  fn visible() {
    assert_eq!(interpret("Visible[x]").unwrap(), "Visible[x]");
  }
  #[test]
  fn truncated_distribution() {
    assert_eq!(
      interpret("TruncatedDistribution[x, y]").unwrap(),
      "TruncatedDistribution[x, y]"
    );
  }

  // ─── DiscreteUniformDistribution ───────────────────────────────────
  #[test]
  fn discrete_uniform_distribution_pdf() {
    assert_eq!(
      interpret("PDF[DiscreteUniformDistribution[{1, 10}], 5]").unwrap(),
      "1/10"
    );
  }
  #[test]
  fn discrete_uniform_distribution_pdf_outside() {
    assert_eq!(
      interpret("PDF[DiscreteUniformDistribution[{1, 10}], 0]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("PDF[DiscreteUniformDistribution[{1, 10}], 11]").unwrap(),
      "0"
    );
  }
  #[test]
  fn discrete_uniform_distribution_cdf() {
    assert_eq!(
      interpret("CDF[DiscreteUniformDistribution[{1, 10}], 5]").unwrap(),
      "1/2"
    );
  }
  #[test]
  fn discrete_uniform_distribution_cdf_edges() {
    assert_eq!(
      interpret("CDF[DiscreteUniformDistribution[{1, 10}], 0]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("CDF[DiscreteUniformDistribution[{1, 10}], 10]").unwrap(),
      "1"
    );
  }
  #[test]
  fn discrete_uniform_distribution_mean() {
    assert_eq!(
      interpret("Mean[DiscreteUniformDistribution[{1, 10}]]").unwrap(),
      "11/2"
    );
  }
  #[test]
  fn discrete_uniform_distribution_variance() {
    assert_eq!(
      interpret("Variance[DiscreteUniformDistribution[{1, 10}]]").unwrap(),
      "33/4"
    );
  }

  // ─── PositiveDefiniteMatrixQ ───────────────────────────────────────
  #[test]
  fn positive_definite_matrix_q_true() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{2, -1}, {-1, 2}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn positive_definite_matrix_q_false() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{1, 2}, {2, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn positive_definite_matrix_q_identity() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[IdentityMatrix[3]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn positive_definite_matrix_q_zero() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{0}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn positive_definite_matrix_q_scalar() {
    assert_eq!(interpret("PositiveDefiniteMatrixQ[{{5}}]").unwrap(), "True");
  }
  #[test]
  fn positive_definite_matrix_q_diagonal() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]")
        .unwrap(),
      "True"
    );
  }

  // ─── MovingMap ─────────────────────────────────────────────────────
  #[test]
  fn moving_map_total() {
    assert_eq!(
      interpret("MovingMap[Total, {1, 2, 3, 4, 5}, 2]").unwrap(),
      "{6, 9, 12}"
    );
  }
  #[test]
  fn moving_map_mean() {
    assert_eq!(
      interpret("MovingMap[Mean, {1, 2, 3, 4, 5}, 3]").unwrap(),
      "{5/2, 7/2}"
    );
  }

  // ─── Unevaluated batch ────────────────────────────────────────────
  #[test]
  fn notebook_find() {
    assert_eq!(interpret("NotebookFind[x]").unwrap(), "NotebookFind[x]");
  }
  #[test]
  fn elementwise_layer() {
    assert_eq!(interpret("ElementwiseLayer[x]").unwrap(), "$Failed");
  }
  #[test]
  fn classifier_measurements() {
    assert_eq!(
      interpret("ClassifierMeasurements[x, y]").unwrap(),
      "ClassifierMeasurements[x, y]"
    );
  }
  #[test]
  fn estimated_process() {
    assert_eq!(
      interpret("EstimatedProcess[x]").unwrap(),
      "EstimatedProcess[x]"
    );
  }
  #[test]
  fn highlight_mesh() {
    assert_eq!(interpret("HighlightMesh[x]").unwrap(), "HighlightMesh[x]");
  }
  #[test]
  fn animator() {
    assert_eq!(interpret("Animator[x]").unwrap(), "Animator[x]");
  }
  #[test]
  fn auto_scroll() {
    assert_eq!(interpret("AutoScroll[x]").unwrap(), "AutoScroll[x]");
  }
  #[test]
  fn confidence_level() {
    assert_eq!(
      interpret("ConfidenceLevel[x]").unwrap(),
      "ConfidenceLevel[x]"
    );
  }
  #[test]
  fn coefficient_rules() {
    assert_eq!(
      interpret("CoefficientRules[x, y]").unwrap(),
      "CoefficientRules[x, y]"
    );
  }
  #[test]
  fn create_palette() {
    assert_eq!(interpret("CreatePalette[x]").unwrap(), "$Failed");
  }
  #[test]
  fn thinning() {
    assert_eq!(interpret("Thinning[x]").unwrap(), "Thinning[x]");
  }
  #[test]
  fn net_decoder() {
    assert_eq!(interpret("NetDecoder[x]").unwrap(), "$Failed");
  }
  #[test]
  fn erosion() {
    assert_eq!(interpret("Erosion[x]").unwrap(), "Erosion[x]");
  }
  #[test]
  fn tolerance() {
    assert_eq!(interpret("Tolerance[x]").unwrap(), "Tolerance[x]");
  }
  #[test]
  fn net_initialize() {
    assert_eq!(interpret("NetInitialize[x]").unwrap(), "NetInitialize[x]");
  }
  #[test]
  fn boundary_mesh_region() {
    assert_eq!(
      interpret("BoundaryMeshRegion[x]").unwrap(),
      "BoundaryMeshRegion[x]"
    );
  }
  #[test]
  fn geometric_brownian_motion_process() {
    assert_eq!(
      interpret("GeometricBrownianMotionProcess[x]").unwrap(),
      "GeometricBrownianMotionProcess[x]"
    );
  }
  #[test]
  fn boolean_convert() {
    assert_eq!(interpret("BooleanConvert[x]").unwrap(), "x");
  }
  #[test]
  fn select_components() {
    assert_eq!(
      interpret("SelectComponents[x]").unwrap(),
      "SelectComponents[x]"
    );
  }
  #[test]
  fn mesh_cell_style() {
    assert_eq!(interpret("MeshCellStyle[x]").unwrap(), "MeshCellStyle[x]");
  }
  #[test]
  fn notebook_put() {
    assert_eq!(interpret("NotebookPut[x]").unwrap(), "NotebookPut[x]");
  }
  #[test]
  fn text_sentences() {
    assert_eq!(interpret("TextSentences[x]").unwrap(), "TextSentences[x]");
  }
  #[test]
  fn polynomial_reduce() {
    assert_eq!(
      interpret("PolynomialReduce[x, y]").unwrap(),
      "PolynomialReduce[x, y]"
    );
  }
  #[test]
  fn cumulant_unevaluated() {
    assert_eq!(interpret("Cumulant[x]").unwrap(), "Cumulant[x]");
  }
  #[test]
  fn three_j_symbol() {
    assert_eq!(
      interpret("ThreeJSymbol[x, y]").unwrap(),
      "ThreeJSymbol[x, y]"
    );
  }
  #[test]
  fn copy_file() {
    assert_eq!(interpret("CopyFile[x]").unwrap(), "CopyFile[x]");
  }
  #[test]
  fn create_directory() {
    assert_eq!(interpret("CreateDirectory[x]").unwrap(), "$Failed");
  }

  // ─── DiscreteDelta ─────────────────────────────────────────────────
  #[test]
  fn discrete_delta_zero() {
    assert_eq!(interpret("DiscreteDelta[0]").unwrap(), "1");
  }
  #[test]
  fn discrete_delta_nonzero() {
    assert_eq!(interpret("DiscreteDelta[1]").unwrap(), "0");
  }
  #[test]
  fn discrete_delta_multiple_zeros() {
    assert_eq!(interpret("DiscreteDelta[0, 0]").unwrap(), "1");
  }
  #[test]
  fn discrete_delta_mixed() {
    assert_eq!(interpret("DiscreteDelta[0, 1]").unwrap(), "0");
  }
  #[test]
  fn discrete_delta_no_args() {
    assert_eq!(interpret("DiscreteDelta[]").unwrap(), "1");
  }
  #[test]
  fn discrete_delta_symbolic() {
    assert_eq!(interpret("DiscreteDelta[x]").unwrap(), "DiscreteDelta[x]");
  }

  // ─── Unevaluated batch 5 ──────────────────────────────────────────
  #[test]
  fn magnify() {
    assert_eq!(interpret("Magnify[x]").unwrap(), "Magnify[x]");
  }
  #[test]
  fn script_baseline_shifts() {
    assert_eq!(
      interpret("ScriptBaselineShifts[x]").unwrap(),
      "ScriptBaselineShifts[x]"
    );
  }
  #[test]
  fn line_spacing() {
    assert_eq!(interpret("LineSpacing[x]").unwrap(), "LineSpacing[x]");
  }
  #[test]
  fn function_range() {
    assert_eq!(
      interpret("FunctionRange[x, y]").unwrap(),
      "FunctionRange[x, y]"
    );
  }
  #[test]
  fn vectors() {
    assert_eq!(interpret("Vectors[x]").unwrap(), "Vectors[x, Complexes]");
  }
  #[test]
  fn sector_origin() {
    assert_eq!(interpret("SectorOrigin[x]").unwrap(), "SectorOrigin[x]");
  }
  #[test]
  fn max_training_rounds() {
    assert_eq!(
      interpret("MaxTrainingRounds[x]").unwrap(),
      "MaxTrainingRounds[x]"
    );
  }
  #[test]
  fn polar_axes() {
    assert_eq!(interpret("PolarAxes[x]").unwrap(), "PolarAxes[x]");
  }
  #[test]
  fn polynomial_gcd() {
    assert_eq!(
      interpret("PolynomialGCD[x, y]").unwrap(),
      "PolynomialGCD[x, y]"
    );
  }
  #[test]
  fn system_dialog_input() {
    assert_eq!(
      interpret("SystemDialogInput[x]").unwrap(),
      "SystemDialogInput[x]"
    );
  }
  #[test]
  fn ar_process() {
    assert_eq!(interpret("ARProcess[x]").unwrap(), "ARProcess[x]");
  }
  #[test]
  fn discrete_wavelet_transform() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[x]").unwrap(),
      "DiscreteWaveletTransform[x]"
    );
  }
  #[test]
  fn relation_graph() {
    assert_eq!(interpret("RelationGraph[x]").unwrap(), "RelationGraph[x]");
  }
  #[test]
  fn image_partition() {
    assert_eq!(interpret("ImagePartition[x]").unwrap(), "ImagePartition[x]");
  }
  #[test]
  fn petersen_graph() {
    assert_eq!(interpret("PetersenGraph[x]").unwrap(), "PetersenGraph[x]");
  }
  #[test]
  fn r_solve_value() {
    assert_eq!(
      interpret("RSolveValue[x, y, z]").unwrap(),
      "RSolveValue[x, y, z]"
    );
  }
  #[test]
  fn feature_extraction() {
    assert_eq!(
      interpret("FeatureExtraction[x]").unwrap(),
      "FeatureExtraction[x]"
    );
  }
  #[test]
  fn graph_distance() {
    assert_eq!(
      interpret("GraphDistance[x, y]").unwrap(),
      "GraphDistance[x, y]"
    );
  }
  #[test]
  fn cell_style() {
    assert_eq!(interpret("CellStyle[x]").unwrap(), "CellStyle[x]");
  }
  #[test]
  fn directory_q() {
    assert_eq!(interpret("DirectoryQ[x]").unwrap(), "False");
  }
  #[test]
  fn image_identify() {
    assert_eq!(interpret("ImageIdentify[x]").unwrap(), "ImageIdentify[x]");
  }
  #[test]
  fn asymptotic() {
    assert_eq!(interpret("Asymptotic[x, y]").unwrap(), "Asymptotic[x, y]");
  }
  #[test]
  fn coordinate_transform() {
    assert_eq!(
      interpret("CoordinateTransform[x, y]").unwrap(),
      "CoordinateTransform[x, y]"
    );
  }
  #[test]
  fn window_margins() {
    assert_eq!(interpret("WindowMargins[x]").unwrap(), "WindowMargins[x]");
  }
  #[test]
  fn affine_transform() {
    assert_eq!(
      interpret("AffineTransform[x]").unwrap(),
      "AffineTransform[x]"
    );
  }
  #[test]
  fn radio_button() {
    assert_eq!(interpret("RadioButton[x]").unwrap(), "RadioButton[x]");
  }
  #[test]
  fn legend_markers() {
    assert_eq!(interpret("LegendMarkers[x]").unwrap(), "LegendMarkers[x]");
  }
  #[test]
  fn powers_representations() {
    assert_eq!(
      interpret("PowersRepresentations[x, y, z]").unwrap(),
      "PowersRepresentations[x, y, z]"
    );
  }
  #[test]
  fn show_string_characters() {
    assert_eq!(
      interpret("ShowStringCharacters[x]").unwrap(),
      "ShowStringCharacters[x]"
    );
  }

  // ─── FlattenAt ─────────────────────────────────────────────────────
  #[test]
  fn flatten_at_single() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, d}, 2]").unwrap(),
      "{a, b, c, d}"
    );
  }
  #[test]
  fn flatten_at_negative() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, d}, -2]").unwrap(),
      "{a, b, c, d}"
    );
  }
  #[test]
  fn flatten_at_first() {
    assert_eq!(
      interpret("FlattenAt[{{a, b}, {c, d}, {e, f}}, 2]").unwrap(),
      "{{a, b}, c, d, {e, f}}"
    );
  }

  // ─── InversePermutation ────────────────────────────────────────────
  #[test]
  fn inverse_permutation_basic() {
    assert_eq!(
      interpret("InversePermutation[{3, 1, 2}]").unwrap(),
      "{2, 3, 1}"
    );
  }
  #[test]
  fn inverse_permutation_4() {
    assert_eq!(
      interpret("InversePermutation[{2, 4, 1, 3}]").unwrap(),
      "{3, 1, 4, 2}"
    );
  }

  // ─── Unevaluated batch 6 ──────────────────────────────────────────
  #[test]
  fn nd_eigensystem() {
    assert_eq!(
      interpret("NDEigensystem[x, y]").unwrap(),
      "NDEigensystem[x, y]"
    );
  }
  #[test]
  fn texture_coordinate_function() {
    assert_eq!(
      interpret("TextureCoordinateFunction[x]").unwrap(),
      "TextureCoordinateFunction[x]"
    );
  }
  #[test]
  fn find_distribution() {
    assert_eq!(
      interpret("FindDistribution[x]").unwrap(),
      "FindDistribution[x]"
    );
  }
  #[test]
  fn text_cases() {
    assert_eq!(interpret("TextCases[x, y]").unwrap(), "TextCases[x, y]");
  }
  #[test]
  fn multicolumn() {
    assert_eq!(interpret("Multicolumn[x]").unwrap(), "Multicolumn[x]");
  }
  #[test]
  fn record() {
    assert_eq!(interpret("Record[x]").unwrap(), "Record[x]");
  }
  #[test]
  fn whittaker_m() {
    assert_eq!(
      interpret("WhittakerM[x, y, z]").unwrap(),
      "WhittakerM[x, y, z]"
    );
  }
  #[test]
  fn interpretation_box() {
    assert_eq!(
      interpret("InterpretationBox[x]").unwrap(),
      "InterpretationBox[x]"
    );
  }
  #[test]
  fn include_pods() {
    assert_eq!(interpret("IncludePods[x]").unwrap(), "IncludePods[x]");
  }
  #[test]
  fn rule_plot() {
    assert_eq!(interpret("RulePlot[x]").unwrap(), "RulePlot[x]");
  }
  #[test]
  fn mathieu_group_m11() {
    assert_eq!(
      interpret("MathieuGroupM11[x]").unwrap(),
      "MathieuGroupM11[x]"
    );
  }
  #[test]
  fn trig() {
    assert_eq!(interpret("Trig[x]").unwrap(), "Trig[x]");
  }
  #[test]
  fn overlaps() {
    assert_eq!(interpret("Overlaps[x]").unwrap(), "Overlaps[x]");
  }
  #[test]
  fn ito_process() {
    assert_eq!(interpret("ItoProcess[x]").unwrap(), "ItoProcess[x]");
  }
  #[test]
  fn net_model() {
    assert_eq!(interpret("NetModel[x]").unwrap(), "$Failed");
  }
  #[test]
  fn rotation_action() {
    assert_eq!(interpret("RotationAction[x]").unwrap(), "RotationAction[x]");
  }
  #[test]
  fn ket() {
    assert_eq!(interpret("Ket[x]").unwrap(), "Ket[x]");
  }
  #[test]
  fn discrete_markov_process() {
    assert_eq!(
      interpret("DiscreteMarkovProcess[x, y]").unwrap(),
      "DiscreteMarkovProcess[x, y]"
    );
  }
  #[test]
  fn boundary_discretize_graphics() {
    assert_eq!(
      interpret("BoundaryDiscretizeGraphics[x]").unwrap(),
      "BoundaryDiscretizeGraphics[x]"
    );
  }
  #[test]
  fn trading_chart() {
    assert_eq!(interpret("TradingChart[x]").unwrap(), "TradingChart[x]");
  }
  #[test]
  fn find_max_value() {
    assert_eq!(
      interpret("FindMaxValue[x, y]").unwrap(),
      "FindMaxValue[x, y]"
    );
  }
  #[test]
  fn form_page() {
    assert_eq!(interpret("FormPage[x]").unwrap(), "FormPage[x]");
  }
  #[test]
  fn nearest_neighbor_graph() {
    assert_eq!(
      interpret("NearestNeighborGraph[x]").unwrap(),
      "NearestNeighborGraph[x]"
    );
  }
  #[test]
  fn file_print() {
    assert_eq!(interpret("FilePrint[x]").unwrap(), "FilePrint[x]");
  }
  #[test]
  fn distribute_definitions() {
    assert_eq!(interpret("DistributeDefinitions[x]").unwrap(), "{}");
  }
  #[test]
  fn riemann_siegel_z() {
    assert_eq!(interpret("RiemannSiegelZ[x]").unwrap(), "RiemannSiegelZ[x]");
  }
  #[test]
  fn batch_normalization_layer() {
    assert_eq!(interpret("BatchNormalizationLayer[x]").unwrap(), "$Failed");
  }
  #[test]
  fn chart_base_style() {
    assert_eq!(interpret("ChartBaseStyle[x]").unwrap(), "ChartBaseStyle[x]");
  }

  // ─── Unevaluated batch 7 ──────────────────────────────────────────
  #[test]
  fn moon_phase() {
    assert_eq!(interpret("MoonPhase[x]").unwrap(), "MoonPhase[x]");
  }
  #[test]
  fn hazard_function() {
    assert_eq!(
      interpret("HazardFunction[x, y]").unwrap(),
      "HazardFunction[x, y]"
    );
  }
  #[test]
  fn content_size() {
    assert_eq!(interpret("ContentSize[x]").unwrap(), "ContentSize[x]");
  }
  #[test]
  fn horner_form() {
    assert_eq!(interpret("HornerForm[x]").unwrap(), "x");
  }
  #[test]
  fn word_boundary() {
    assert_eq!(interpret("WordBoundary[x]").unwrap(), "WordBoundary[x]");
  }
  #[test]
  fn n_expectation() {
    assert_eq!(
      interpret("NExpectation[x, y]").unwrap(),
      "NExpectation[x, y]"
    );
  }
  #[test]
  fn mouseover() {
    assert_eq!(interpret("Mouseover[x, y]").unwrap(), "Mouseover[x, y]");
  }
  #[test]
  fn rectangle_chart() {
    assert_eq!(interpret("RectangleChart[x]").unwrap(), "RectangleChart[x]");
  }
  #[test]
  fn affine_state_space_model() {
    assert_eq!(
      interpret("AffineStateSpaceModel[x]").unwrap(),
      "AffineStateSpaceModel[x]"
    );
  }
  #[test]
  fn log_likelihood() {
    assert_eq!(
      interpret("LogLikelihood[x, y]").unwrap(),
      "LogLikelihood[x, y]"
    );
  }
  #[test]
  fn span_from_above() {
    assert_eq!(interpret("SpanFromAbove[x]").unwrap(), "SpanFromAbove[x]");
  }
  #[test]
  fn min_value() {
    assert_eq!(interpret("MinValue[x, y]").unwrap(), "MinValue[x, y]");
  }
  #[test]
  fn sub_plus() {
    assert_eq!(interpret("SubPlus[x]").unwrap(), "SubPlus[x]");
  }
  #[test]
  fn extension() {
    assert_eq!(interpret("Extension[x]").unwrap(), "Extension[x]");
  }
  #[test]
  fn weighted_adjacency_graph() {
    assert_eq!(
      interpret("WeightedAdjacencyGraph[x]").unwrap(),
      "WeightedAdjacencyGraph[x]"
    );
  }
  #[test]
  fn cell_frame() {
    assert_eq!(interpret("CellFrame[x]").unwrap(), "CellFrame[x]");
  }
  #[test]
  fn compiled() {
    assert_eq!(interpret("Compiled[x]").unwrap(), "Compiled[x]");
  }
  #[test]
  fn audio_generator() {
    assert_eq!(interpret("AudioGenerator[x]").unwrap(), "AudioGenerator[x]");
  }
  #[test]
  fn underlined() {
    assert_eq!(interpret("Underlined[x]").unwrap(), "Underlined[x]");
  }
  #[test]
  fn fourier_coefficient() {
    assert_eq!(
      interpret("FourierCoefficient[x, y, z]").unwrap(),
      "FourierCoefficient[x, y, z]"
    );
  }
  #[test]
  fn overscript() {
    assert_eq!(interpret("Overscript[x, y]").unwrap(), "Overscript[x, y]");
  }
  #[test]
  fn primes() {
    assert_eq!(interpret("Primes[x]").unwrap(), "Primes[x]");
  }
  #[test]
  fn community_graph_plot() {
    assert_eq!(
      interpret("CommunityGraphPlot[x]").unwrap(),
      "CommunityGraphPlot[x]"
    );
  }
  #[test]
  fn random_prime() {
    assert_eq!(interpret("RandomPrime[x]").unwrap(), "RandomPrime[x]");
  }
  #[test]
  fn super_dagger() {
    assert_eq!(interpret("SuperDagger[x]").unwrap(), "SuperDagger[x]");
  }
  #[test]
  fn re_im_plot() {
    assert_eq!(interpret("ReImPlot[x, y]").unwrap(), "ReImPlot[x, y]");
  }
  #[test]
  fn exponent_function() {
    assert_eq!(
      interpret("ExponentFunction[x]").unwrap(),
      "ExponentFunction[x]"
    );
  }
  #[test]
  fn softmax_layer() {
    assert_eq!(interpret("SoftmaxLayer[x]").unwrap(), "$Failed");
  }
  #[test]
  fn product_distribution() {
    assert_eq!(
      interpret("ProductDistribution[x]").unwrap(),
      "ProductDistribution[x]"
    );
  }
  #[test]
  fn toggler_bar() {
    assert_eq!(interpret("TogglerBar[x]").unwrap(), "TogglerBar[x]");
  }

  // ─── TakeList ──────────────────────────────────────────────────────
  #[test]
  fn take_list_basic() {
    assert_eq!(
      interpret("TakeList[{a, b, c, d, e, f}, {2, 3, 1}]").unwrap(),
      "{{a, b}, {c, d, e}, {f}}"
    );
  }
  #[test]
  fn take_list_equal_parts() {
    assert_eq!(
      interpret("TakeList[{1, 2, 3, 4}, {2, 2}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  // ─── MultiplicativeOrder ───────────────────────────────────────────
  #[test]
  fn multiplicative_order_basic() {
    assert_eq!(interpret("MultiplicativeOrder[2, 7]").unwrap(), "3");
  }
  #[test]
  fn multiplicative_order_3_10() {
    assert_eq!(interpret("MultiplicativeOrder[3, 10]").unwrap(), "4");
  }
  #[test]
  fn multiplicative_order_10_7() {
    assert_eq!(interpret("MultiplicativeOrder[10, 7]").unwrap(), "6");
  }

  // ─── Unevaluated batch 8 ──────────────────────────────────────────
  #[test]
  fn region_dimension() {
    assert_eq!(
      interpret("RegionDimension[x]").unwrap(),
      "RegionDimension[x]"
    );
  }
  #[test]
  fn feature_extractor() {
    assert_eq!(
      interpret("FeatureExtractor[x]").unwrap(),
      "FeatureExtractor[x]"
    );
  }
  #[test]
  fn arg_max() {
    assert_eq!(interpret("ArgMax[x, y]").unwrap(), "ArgMax[x, y]");
  }
  #[test]
  fn vertex_normals() {
    assert_eq!(interpret("VertexNormals[x]").unwrap(), "VertexNormals[x]");
  }
  #[test]
  fn correlation_function() {
    assert_eq!(
      interpret("CorrelationFunction[x, y]").unwrap(),
      "CorrelationFunction[x, y]"
    );
  }
  #[test]
  fn bell_y() {
    assert_eq!(interpret("BellY[x, y, z]").unwrap(), "BellY[x, y, z]");
  }
  #[test]
  fn parallel_do() {
    assert_eq!(interpret("ParallelDo[x]").unwrap(), "Null");
  }
  #[test]
  fn barnes_g() {
    assert_eq!(interpret("BarnesG[x]").unwrap(), "BarnesG[x]");
  }
  #[test]
  fn url() {
    assert_eq!(interpret("URL[x]").unwrap(), "URL[x]");
  }
  #[test]
  fn find_geometric_transform() {
    assert_eq!(
      interpret("FindGeometricTransform[x, y]").unwrap(),
      "FindGeometricTransform[x, y]"
    );
  }
  #[test]
  fn deployed() {
    assert_eq!(interpret("Deployed[x]").unwrap(), "Deployed[x]");
  }
  #[test]
  fn dirichlet_distribution() {
    assert_eq!(
      interpret("DirichletDistribution[x]").unwrap(),
      "DirichletDistribution[x]"
    );
  }
  #[test]
  fn riemann_siegel_theta() {
    assert_eq!(
      interpret("RiemannSiegelTheta[x]").unwrap(),
      "RiemannSiegelTheta[x]"
    );
  }
  #[test]
  fn random_instance() {
    assert_eq!(interpret("RandomInstance[x]").unwrap(), "RandomInstance[x]");
  }
  #[test]
  fn trig_factor() {
    assert_eq!(interpret("TrigFactor[x]").unwrap(), "x");
  }
  #[test]
  fn pooling_layer() {
    assert_eq!(interpret("PoolingLayer[x]").unwrap(), "$Failed");
  }
  #[test]
  fn notebook_delete() {
    assert_eq!(interpret("NotebookDelete[x]").unwrap(), "NotebookDelete[x]");
  }
  #[test]
  fn find_formula() {
    assert_eq!(interpret("FindFormula[x, y]").unwrap(), "FindFormula[x, y]");
  }
  #[test]
  fn graph_3d() {
    assert_eq!(interpret("Graph3D[x]").unwrap(), "Graph3D[x]");
  }
  #[test]
  fn whittaker_w() {
    assert_eq!(
      interpret("WhittakerW[x, y, z]").unwrap(),
      "WhittakerW[x, y, z]"
    );
  }
  #[test]
  fn max_detect() {
    assert_eq!(interpret("MaxDetect[x]").unwrap(), "MaxDetect[x]");
  }
  #[test]
  fn geometric_scene() {
    assert_eq!(interpret("GeometricScene[x]").unwrap(), "GeometricScene[x]");
  }
  #[test]
  fn parallelize() {
    assert_eq!(interpret("Parallelize[x]").unwrap(), "x");
  }
  #[test]
  fn clustering_components() {
    assert_eq!(
      interpret("ClusteringComponents[x]").unwrap(),
      "ClusteringComponents[x]"
    );
  }
  #[test]
  fn bernoulli_graph_distribution() {
    assert_eq!(
      interpret("BernoulliGraphDistribution[x, y]").unwrap(),
      "BernoulliGraphDistribution[x, y]"
    );
  }
  #[test]
  fn mandelbrot_set_plot() {
    assert_eq!(
      interpret("MandelbrotSetPlot[x]").unwrap(),
      "MandelbrotSetPlot[x]"
    );
  }
  #[test]
  fn language() {
    assert_eq!(interpret("Language[x]").unwrap(), "Language[x]");
  }
  #[test]
  fn sequence_cases() {
    assert_eq!(
      interpret("SequenceCases[x, y]").unwrap(),
      "SequenceCases[x, y]"
    );
  }
  #[test]
  fn time_constraint() {
    assert_eq!(interpret("TimeConstraint[x]").unwrap(), "TimeConstraint[x]");
  }
  #[test]
  fn double_right_tee() {
    assert_eq!(interpret("DoubleRightTee[x]").unwrap(), "DoubleRightTee[x]");
  }
  #[test]
  fn matrices() {
    assert_eq!(interpret("Matrices[x]").unwrap(), "Matrices[x]");
  }
  #[test]
  fn joined_curve() {
    assert_eq!(interpret("JoinedCurve[x]").unwrap(), "JoinedCurve[x]");
  }
  #[test]
  fn run_process() {
    assert_eq!(interpret("RunProcess[x]").unwrap(), "RunProcess[x]");
  }
  #[test]
  fn starting_step_size() {
    assert_eq!(
      interpret("StartingStepSize[x]").unwrap(),
      "StartingStepSize[x]"
    );
  }
  #[test]
  fn default_button() {
    assert_eq!(interpret("DefaultButton[x]").unwrap(), "DefaultButton[x]");
  }
  #[test]
  fn trigger() {
    assert_eq!(interpret("Trigger[x]").unwrap(), "Trigger[x]");
  }
  #[test]
  fn geo_marker() {
    assert_eq!(interpret("GeoMarker[x]").unwrap(), "GeoMarker[x]");
  }
  #[test]
  fn content_selectable() {
    assert_eq!(
      interpret("ContentSelectable[x]").unwrap(),
      "ContentSelectable[x]"
    );
  }

  // ─── LaplaceDistribution ──────────────────────────────────────────
  #[test]
  fn laplace_distribution_pdf() {
    assert_eq!(
      interpret("PDF[LaplaceDistribution[0, 1], 0]").unwrap(),
      "1/2"
    );
  }
  #[test]
  fn laplace_distribution_cdf() {
    assert_eq!(
      interpret("CDF[LaplaceDistribution[0, 1], 0]").unwrap(),
      "1/2"
    );
  }
  #[test]
  fn laplace_distribution_mean() {
    assert_eq!(interpret("Mean[LaplaceDistribution[2, 3]]").unwrap(), "2");
  }
  #[test]
  fn laplace_distribution_variance() {
    assert_eq!(
      interpret("Variance[LaplaceDistribution[2, 3]]").unwrap(),
      "18"
    );
  }

  // ─── RayleighDistribution ─────────────────────────────────────────
  #[test]
  fn rayleigh_distribution_pdf() {
    assert_eq!(
      interpret("N[PDF[RayleighDistribution[1], 1]]").unwrap(),
      interpret("N[1/Sqrt[E]]").unwrap()
    );
  }
  #[test]
  fn rayleigh_distribution_cdf() {
    assert_eq!(
      interpret("N[CDF[RayleighDistribution[1], 1]]").unwrap(),
      interpret("N[1 - 1/Sqrt[E]]").unwrap()
    );
  }
  #[test]
  fn rayleigh_distribution_mean() {
    assert_eq!(
      interpret("Mean[RayleighDistribution[s]]").unwrap(),
      "Sqrt[Pi/2]*s"
    );
  }
  #[test]
  fn rayleigh_distribution_variance() {
    assert_eq!(
      interpret("Variance[RayleighDistribution[s]]").unwrap(),
      "s^2*(2 - Pi/2)"
    );
  }

  // ─── Unevaluated batch 9 ──────────────────────────────────────────
  #[test]
  fn export_form() {
    assert_eq!(interpret("ExportForm[x, y]").unwrap(), "ExportForm[x, y]");
  }
  #[test]
  fn parallel_submit() {
    assert_eq!(interpret("ParallelSubmit[x]").unwrap(), "ParallelSubmit[x]");
  }
  #[test]
  fn application() {
    assert_eq!(interpret("Application[x]").unwrap(), "Application[x]");
  }
  #[test]
  fn find_file() {
    assert_eq!(interpret("FindFile[x]").unwrap(), "FindFile[x]");
  }
  #[test]
  fn distance_transform() {
    assert_eq!(
      interpret("DistanceTransform[x]").unwrap(),
      "DistanceTransform[x]"
    );
  }
  #[test]
  fn timeline_plot() {
    assert_eq!(interpret("TimelinePlot[x]").unwrap(), "TimelinePlot[x]");
  }
  #[test]
  fn dialog_input() {
    assert_eq!(interpret("DialogInput[x]").unwrap(), "$Failed");
  }
  #[test]
  fn pass_events_down() {
    assert_eq!(interpret("PassEventsDown[x]").unwrap(), "PassEventsDown[x]");
  }
  #[test]
  fn circle_dot() {
    assert_eq!(interpret("CircleDot[x]").unwrap(), "CircleDot[x]");
  }
  #[test]
  fn vector_scaling() {
    assert_eq!(interpret("VectorScaling[x]").unwrap(), "VectorScaling[x]");
  }
  #[test]
  fn find_generating_function() {
    assert_eq!(
      interpret("FindGeneratingFunction[x, y]").unwrap(),
      "FindGeneratingFunction[x, y]"
    );
  }
  #[test]
  fn associate_to() {
    assert_eq!(interpret("AssociateTo[x, y]").unwrap(), "AssociateTo[x, y]");
  }
  #[test]
  fn histogram_distribution() {
    assert_eq!(
      interpret("HistogramDistribution[x]").unwrap(),
      "HistogramDistribution[x]"
    );
  }
  #[test]
  fn gaussian_matrix() {
    assert_eq!(interpret("GaussianMatrix[x]").unwrap(), "GaussianMatrix[x]");
  }
  #[test]
  fn text_recognize() {
    assert_eq!(interpret("TextRecognize[x]").unwrap(), "TextRecognize[x]");
  }
  #[test]
  fn number_signs() {
    assert_eq!(interpret("NumberSigns[x]").unwrap(), "NumberSigns[x]");
  }
  #[test]
  fn weierstrass_zeta() {
    assert_eq!(
      interpret("WeierstrassZeta[x, y]").unwrap(),
      "WeierstrassZeta[x, y]"
    );
  }
  #[test]
  fn list_surface_plot_3d() {
    assert_eq!(
      interpret("ListSurfacePlot3D[x]").unwrap(),
      "ListSurfacePlot3D[x]"
    );
  }
  #[test]
  fn f_ratio_distribution() {
    assert_eq!(
      interpret("FRatioDistribution[x, y]").unwrap(),
      "FRatioDistribution[x, y]"
    );
  }
  #[test]
  fn date_value() {
    assert_eq!(interpret("DateValue[x]").unwrap(), "DateValue[x]");
  }
  #[test]
  fn density_plot_3d() {
    assert_eq!(
      interpret("DensityPlot3D[x, y]").unwrap(),
      "DensityPlot3D[x, y]"
    );
  }
  #[test]
  fn geo_region_value_plot() {
    assert_eq!(
      interpret("GeoRegionValuePlot[x]").unwrap(),
      "GeoRegionValuePlot[x]"
    );
  }
  #[test]
  fn max_extra_conditions() {
    assert_eq!(
      interpret("MaxExtraConditions[x]").unwrap(),
      "MaxExtraConditions[x]"
    );
  }
  #[test]
  fn time_series_model_fit() {
    assert_eq!(
      interpret("TimeSeriesModelFit[x]").unwrap(),
      "TimeSeriesModelFit[x]"
    );
  }
  #[test]
  fn pane_selector() {
    assert_eq!(interpret("PaneSelector[x]").unwrap(), "PaneSelector[x]");
  }
  #[test]
  fn url_execute() {
    assert_eq!(interpret("URLExecute[x]").unwrap(), "URLExecute[x]");
  }
  #[test]
  fn sequence_position() {
    assert_eq!(
      interpret("SequencePosition[x, y]").unwrap(),
      "SequencePosition[x, y]"
    );
  }
  #[test]
  fn file_base_name() {
    assert_eq!(interpret("FileBaseName[x]").unwrap(), "FileBaseName[x]");
  }
  #[test]
  fn coordinates_tool_options() {
    assert_eq!(
      interpret("CoordinatesToolOptions[x]").unwrap(),
      "CoordinatesToolOptions[x]"
    );
  }
  #[test]
  fn color_combine() {
    assert_eq!(interpret("ColorCombine[x]").unwrap(), "ColorCombine[x]");
  }
  #[test]
  fn highlighted() {
    assert_eq!(interpret("Highlighted[x]").unwrap(), "Highlighted[x]");
  }
  #[test]
  fn text_grid() {
    assert_eq!(interpret("TextGrid[x]").unwrap(), "TextGrid[x]");
  }
  #[test]
  fn numeric_function() {
    assert_eq!(
      interpret("NumericFunction[x]").unwrap(),
      "NumericFunction[x]"
    );
  }
  #[test]
  fn scrollbars() {
    assert_eq!(interpret("Scrollbars[x]").unwrap(), "Scrollbars[x]");
  }
  #[test]
  fn color_setter() {
    assert_eq!(interpret("ColorSetter[x]").unwrap(), "ColorSetter[x]");
  }
  #[test]
  fn distance_matrix() {
    assert_eq!(interpret("DistanceMatrix[x]").unwrap(), "DistanceMatrix[x]");
  }
  #[test]
  fn inverse_wavelet_transform() {
    assert_eq!(
      interpret("InverseWaveletTransform[x]").unwrap(),
      "InverseWaveletTransform[x]"
    );
  }
  #[test]
  fn tree_graph() {
    assert_eq!(interpret("TreeGraph[x]").unwrap(), "TreeGraph[x]");
  }

  // ─── DuplicateFreeQ ───────────────────────────────────────────────
  #[test]
  fn duplicate_free_q_true() {
    assert_eq!(interpret("DuplicateFreeQ[{1, 2, 3}]").unwrap(), "True");
  }
  #[test]
  fn duplicate_free_q_false() {
    assert_eq!(interpret("DuplicateFreeQ[{1, 2, 1}]").unwrap(), "False");
  }
  #[test]
  fn duplicate_free_q_empty() {
    assert_eq!(interpret("DuplicateFreeQ[{}]").unwrap(), "True");
  }

  // ─── Unevaluated batch 10 ─────────────────────────────────────────
  #[test]
  fn set_shared_variable() {
    assert_eq!(interpret("SetSharedVariable[x]").unwrap(), "Null");
  }
  #[test]
  fn pade_approximant() {
    assert_eq!(
      interpret("PadeApproximant[x, y]").unwrap(),
      "PadeApproximant[x, y]"
    );
  }
  #[test]
  fn filling_transform() {
    assert_eq!(
      interpret("FillingTransform[x]").unwrap(),
      "FillingTransform[x]"
    );
  }
  #[test]
  fn sampling_period() {
    assert_eq!(interpret("SamplingPeriod[x]").unwrap(), "SamplingPeriod[x]");
  }
  #[test]
  fn find_cycle() {
    assert_eq!(interpret("FindCycle[x]").unwrap(), "FindCycle[x]");
  }
  #[test]
  fn time_series_forecast() {
    assert_eq!(
      interpret("TimeSeriesForecast[x]").unwrap(),
      "TimeSeriesForecast[x]"
    );
  }
  #[test]
  fn cube() {
    assert_eq!(interpret("Cube[x]").unwrap(), "Cube[x]");
  }
  #[test]
  fn characteristic_function() {
    assert_eq!(
      interpret("CharacteristicFunction[x]").unwrap(),
      "CharacteristicFunction[x]"
    );
  }
  #[test]
  fn permutation_replace() {
    assert_eq!(
      interpret("PermutationReplace[x, y]").unwrap(),
      "PermutationReplace[x, y]"
    );
  }
  #[test]
  fn discrete_variables() {
    assert_eq!(
      interpret("DiscreteVariables[x]").unwrap(),
      "DiscreteVariables[x]"
    );
  }
  #[test]
  fn strip_on_input() {
    assert_eq!(interpret("StripOnInput[x]").unwrap(), "StripOnInput[x]");
  }
  #[test]
  fn standardize() {
    assert_eq!(interpret("Standardize[x]").unwrap(), "Standardize[x]");
  }
  #[test]
  fn sub_minus() {
    assert_eq!(interpret("SubMinus[x]").unwrap(), "SubMinus[x]");
  }
  #[test]
  fn corner_neighbors() {
    assert_eq!(
      interpret("CornerNeighbors[x]").unwrap(),
      "CornerNeighbors[x]"
    );
  }
  #[test]
  fn triangular_distribution() {
    assert_eq!(
      interpret("TriangularDistribution[x]").unwrap(),
      "TriangularDistribution[x]"
    );
  }
  #[test]
  fn real_exponent() {
    assert_eq!(interpret("RealExponent[x]").unwrap(), "RealExponent[x]");
  }
  #[test]
  fn color_quantize() {
    assert_eq!(interpret("ColorQuantize[x]").unwrap(), "ColorQuantize[x]");
  }
  #[test]
  fn binary_write() {
    assert_eq!(interpret("BinaryWrite[x]").unwrap(), "BinaryWrite[x]");
  }
  #[test]
  fn checkbox_bar() {
    assert_eq!(interpret("CheckboxBar[x]").unwrap(), "CheckboxBar[x]");
  }
  #[test]
  fn tooltip_delay() {
    assert_eq!(interpret("TooltipDelay[x]").unwrap(), "TooltipDelay[x]");
  }
  #[test]
  fn random_permutation() {
    assert_eq!(
      interpret("RandomPermutation[x]").unwrap(),
      "RandomPermutation[x]"
    );
  }
  #[test]
  fn watershed_components() {
    assert_eq!(
      interpret("WatershedComponents[x]").unwrap(),
      "WatershedComponents[x]"
    );
  }
  #[test]
  fn factorial_moment() {
    assert_eq!(
      interpret("FactorialMoment[x]").unwrap(),
      "FactorialMoment[x]"
    );
  }
  #[test]
  fn view_center() {
    assert_eq!(interpret("ViewCenter[x]").unwrap(), "ViewCenter[x]");
  }
  #[test]
  fn quantile_plot() {
    assert_eq!(interpret("QuantilePlot[x]").unwrap(), "QuantilePlot[x]");
  }
  #[test]
  fn fourier_sin_series() {
    assert_eq!(
      interpret("FourierSinSeries[x, y, z]").unwrap(),
      "FourierSinSeries[x, y, z]"
    );
  }
  #[test]
  fn mathieu_characteristic_a() {
    assert_eq!(
      interpret("MathieuCharacteristicA[x, y]").unwrap(),
      "MathieuCharacteristicA[x, y]"
    );
  }
  #[test]
  fn file_type() {
    assert_eq!(interpret("FileType[x]").unwrap(), "FileType[x]");
  }
  #[test]
  fn resource_object() {
    assert_eq!(interpret("ResourceObject[x]").unwrap(), "$Failed");
  }
  #[test]
  fn stieltjes_gamma() {
    assert_eq!(interpret("StieltjesGamma[x]").unwrap(), "StieltjesGamma[x]");
  }
  #[test]
  fn polar_ticks() {
    assert_eq!(interpret("PolarTicks[x]").unwrap(), "PolarTicks[x]");
  }
  #[test]
  fn beckmann_distribution() {
    assert_eq!(
      interpret("BeckmannDistribution[x]").unwrap(),
      "BeckmannDistribution[x]"
    );
  }
  #[test]
  fn first_case() {
    assert_eq!(interpret("FirstCase[x, y]").unwrap(), "Missing[NotFound]");
  }
  #[test]
  fn weierstrass_sigma() {
    assert_eq!(
      interpret("WeierstrassSigma[x, y]").unwrap(),
      "WeierstrassSigma[x, y]"
    );
  }
  #[test]
  fn mathieu_c() {
    assert_eq!(interpret("MathieuC[x, y, z]").unwrap(), "MathieuC[x, y, z]");
  }
  #[test]
  fn string_replace_part() {
    assert_eq!(
      interpret("StringReplacePart[x, y, z]").unwrap(),
      "StringReplacePart[x, y, z]"
    );
  }
  #[test]
  fn meta_information() {
    assert_eq!(
      interpret("MetaInformation[x]").unwrap(),
      "MetaInformation[x]"
    );
  }
  #[test]
  fn notebook_save() {
    assert_eq!(interpret("NotebookSave[x]").unwrap(), "NotebookSave[x]");
  }
  #[test]
  fn list_contour_plot_3d() {
    assert_eq!(
      interpret("ListContourPlot3D[x]").unwrap(),
      "ListContourPlot3D[x]"
    );
  }

  // ─── Haversine ─────────────────────────────────────────────────────
  #[test]
  fn haversine_zero() {
    assert_eq!(interpret("Haversine[0]").unwrap(), "0");
  }
  #[test]
  fn haversine_pi() {
    assert_eq!(interpret("Haversine[Pi]").unwrap(), "1");
  }
  #[test]
  fn haversine_half_pi() {
    assert_eq!(interpret("Haversine[Pi/2]").unwrap(), "1/2");
  }
  #[test]
  fn inverse_haversine_zero() {
    assert_eq!(interpret("InverseHaversine[0]").unwrap(), "0");
  }
  #[test]
  fn inverse_haversine_one() {
    assert_eq!(interpret("InverseHaversine[1]").unwrap(), "Pi");
  }

  // ─── Unevaluated batch 11 ─────────────────────────────────────────
  #[test]
  fn resampling_method() {
    assert_eq!(
      interpret("ResamplingMethod[x]").unwrap(),
      "ResamplingMethod[x]"
    );
  }
  #[test]
  fn angular_gauge() {
    assert_eq!(interpret("AngularGauge[x]").unwrap(), "AngularGauge[x]");
  }
  #[test]
  fn copy_to_clipboard() {
    assert_eq!(interpret("CopyToClipboard[x]").unwrap(), "$Failed");
  }
  #[test]
  fn color_replace() {
    assert_eq!(interpret("ColorReplace[x]").unwrap(), "ColorReplace[x]");
  }
  #[test]
  fn graph_plot_3d() {
    assert_eq!(interpret("GraphPlot3D[x]").unwrap(), "GraphPlot3D[x]");
  }
  #[test]
  fn button_function() {
    assert_eq!(interpret("ButtonFunction[x]").unwrap(), "ButtonFunction[x]");
  }
  #[test]
  fn system_options() {
    assert_eq!(interpret("SystemOptions[x]").unwrap(), "{}");
  }
  #[test]
  fn sunday() {
    assert_eq!(interpret("Sunday[x]").unwrap(), "Sunday[x]");
  }
  #[test]
  fn frobenius_solve() {
    assert_eq!(
      interpret("FrobeniusSolve[x, y]").unwrap(),
      "FrobeniusSolve[x, y]"
    );
  }
  #[test]
  fn print_temporary() {
    assert_eq!(interpret("PrintTemporary[x]").unwrap(), "Null");
  }
  #[test]
  fn image_value() {
    assert_eq!(interpret("ImageValue[x]").unwrap(), "ImageValue[x]");
  }
  #[test]
  fn generated_parameters() {
    assert_eq!(
      interpret("GeneratedParameters[x]").unwrap(),
      "GeneratedParameters[x]"
    );
  }
  #[test]
  fn plot_region() {
    assert_eq!(interpret("PlotRegion[x]").unwrap(), "PlotRegion[x]");
  }
  #[test]
  fn matrix_log() {
    assert_eq!(interpret("MatrixLog[x]").unwrap(), "MatrixLog[x]");
  }
  #[test]
  fn density_histogram() {
    assert_eq!(
      interpret("DensityHistogram[x]").unwrap(),
      "DensityHistogram[x]"
    );
  }
  #[test]
  fn distribution_chart() {
    assert_eq!(
      interpret("DistributionChart[x]").unwrap(),
      "DistributionChart[x]"
    );
  }
  #[test]
  fn inverse_z_transform() {
    assert_eq!(
      interpret("InverseZTransform[x, y, z]").unwrap(),
      "InverseZTransform[x, y, z]"
    );
  }
  #[test]
  fn incidence_matrix() {
    assert_eq!(
      interpret("IncidenceMatrix[x]").unwrap(),
      "IncidenceMatrix[x]"
    );
  }
  #[test]
  fn notebooks() {
    assert_eq!(interpret("Notebooks[x]").unwrap(), "Notebooks[x]");
  }
  #[test]
  fn z_transform() {
    assert_eq!(
      interpret("ZTransform[x, y, z]").unwrap(),
      "ZTransform[x, y, z]"
    );
  }
  #[test]
  fn least_squares() {
    assert_eq!(
      interpret("LeastSquares[x, y]").unwrap(),
      "LeastSquares[x, y]"
    );
  }
  #[test]
  fn feature_types() {
    assert_eq!(interpret("FeatureTypes[x]").unwrap(), "FeatureTypes[x]");
  }
  #[test]
  fn covariance_function() {
    assert_eq!(
      interpret("CovarianceFunction[x, y]").unwrap(),
      "CovarianceFunction[x, y]"
    );
  }
  #[test]
  fn xyz_color() {
    assert_eq!(interpret("XYZColor[x]").unwrap(), "XYZColor[x]");
  }
  #[test]
  fn graph_highlight_style() {
    assert_eq!(
      interpret("GraphHighlightStyle[x]").unwrap(),
      "GraphHighlightStyle[x]"
    );
  }
  #[test]
  fn image_trim() {
    assert_eq!(interpret("ImageTrim[x]").unwrap(), "ImageTrim[x]");
  }
  #[test]
  fn setting() {
    assert_eq!(interpret("Setting[x]").unwrap(), "x");
  }
  #[test]
  fn b_spline_surface() {
    assert_eq!(interpret("BSplineSurface[x]").unwrap(), "BSplineSurface[x]");
  }
  #[test]
  fn singular_value_list() {
    assert_eq!(
      interpret("SingularValueList[x]").unwrap(),
      "SingularValueList[x]"
    );
  }
  #[test]
  fn morphological_binarize() {
    assert_eq!(
      interpret("MorphologicalBinarize[x]").unwrap(),
      "MorphologicalBinarize[x]"
    );
  }
  #[test]
  fn vertex_weight() {
    assert_eq!(interpret("VertexWeight[x]").unwrap(), "VertexWeight[x]");
  }
  #[test]
  fn single_letter_italics() {
    assert_eq!(
      interpret("SingleLetterItalics[x]").unwrap(),
      "SingleLetterItalics[x]"
    );
  }
  #[test]
  fn polar_grid_lines() {
    assert_eq!(interpret("PolarGridLines[x]").unwrap(), "PolarGridLines[x]");
  }
  #[test]
  fn root_approximant() {
    assert_eq!(
      interpret("RootApproximant[x]").unwrap(),
      "RootApproximant[x]"
    );
  }
  #[test]
  fn abs_arg() {
    assert_eq!(interpret("AbsArg[x]").unwrap(), "{Abs[x], Arg[x]}");
  }
  #[test]
  fn interpretation() {
    assert_eq!(interpret("Interpretation[x]").unwrap(), "Interpretation[x]");
  }
  #[test]
  fn symmetric_group() {
    assert_eq!(interpret("SymmetricGroup[x]").unwrap(), "SymmetricGroup[x]");
  }
  #[test]
  fn databin() {
    assert_eq!(interpret("Databin[x]").unwrap(), "Databin[x]");
  }
  #[test]
  fn inverse_erf() {
    assert_eq!(interpret("InverseErf[x]").unwrap(), "InverseErf[x]");
  }
  #[test]
  fn smooth_density_histogram() {
    assert_eq!(
      interpret("SmoothDensityHistogram[x]").unwrap(),
      "SmoothDensityHistogram[x]"
    );
  }
  #[test]
  fn net_extract() {
    assert_eq!(interpret("NetExtract[x]").unwrap(), "NetExtract[x]");
  }
  #[test]
  fn hankel_h1() {
    assert_eq!(interpret("HankelH1[x, y]").unwrap(), "HankelH1[x, y]");
  }
  #[test]
  fn friday() {
    assert_eq!(interpret("Friday[x]").unwrap(), "Friday[x]");
  }
  #[test]
  fn cloud_import() {
    assert_eq!(interpret("CloudImport[x]").unwrap(), "CloudImport[x]");
  }
  #[test]
  fn temporary() {
    assert_eq!(interpret("Temporary[x]").unwrap(), "Temporary[x]");
  }
  #[test]
  fn service_connect() {
    assert_eq!(interpret("ServiceConnect[x]").unwrap(), "ServiceConnect[x]");
  }
  #[test]
  fn nonlinear_state_space_model() {
    assert_eq!(
      interpret("NonlinearStateSpaceModel[x]").unwrap(),
      "NonlinearStateSpaceModel[x]"
    );
  }
  #[test]
  fn closing() {
    assert_eq!(interpret("Closing[x]").unwrap(), "Closing[x]");
  }
  #[test]
  fn default_duration() {
    assert_eq!(
      interpret("DefaultDuration[x]").unwrap(),
      "DefaultDuration[x]"
    );
  }
  #[test]
  fn from_polar_coordinates_symbolic() {
    assert_eq!(
      interpret("FromPolarCoordinates[{r, theta}]").unwrap(),
      "{r*Cos[theta], r*Sin[theta]}"
    );
  }
  #[test]
  fn from_polar_coordinates_numeric() {
    assert_eq!(
      interpret("FromPolarCoordinates[{2, Pi/4}]").unwrap(),
      "{2/Sqrt[2], 2/Sqrt[2]}"
    );
  }
  #[test]
  fn to_polar_coordinates_symbolic() {
    assert_eq!(
      interpret("ToPolarCoordinates[{x, y}]").unwrap(),
      "{Sqrt[x^2 + y^2], ArcTan[x, y]}"
    );
  }
  #[test]
  fn to_polar_coordinates_numeric() {
    assert_eq!(
      interpret("ToPolarCoordinates[{1, 1}]").unwrap(),
      "{Sqrt[2], ArcTan[1, 1]}"
    );
  }
  #[test]
  fn from_spherical_coordinates() {
    assert_eq!(
      interpret("FromSphericalCoordinates[{r, theta, phi}]").unwrap(),
      "{r*Cos[phi]*Sin[theta], r*Sin[phi]*Sin[theta], r*Cos[theta]}"
    );
  }
  #[test]
  fn to_spherical_coordinates() {
    assert_eq!(
      interpret("ToSphericalCoordinates[{x, y, z}]").unwrap(),
      "{Sqrt[x^2 + y^2 + z^2], ArcTan[z, Sqrt[x^2 + y^2]], ArcTan[x, y]}"
    );
  }
  #[test]
  fn qr_decomposition_identity() {
    assert_eq!(
      interpret("QRDecomposition[{{1, 0}, {0, 1}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}}"
    );
  }
  #[test]
  fn qr_decomposition_3x3() {
    assert_eq!(
      interpret(
        "QRDecomposition[{{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}}]"
      )
      .unwrap(),
      "{{{6/7, 3/7, -2/7}, {-69/175, 158/175, 6/35}, {-58/175, 6/175, -33/35}}, {{14, 21, -14}, {0, 175, -70}, {0, 0, 35}}}"
    );
  }
  #[test]
  fn continued_fraction_k_basic() {
    assert_eq!(
      interpret("ContinuedFractionK[k, {k, 1, 3}]").unwrap(),
      "7/10"
    );
  }
  #[test]
  fn continued_fraction_k_constant() {
    assert_eq!(
      interpret("ContinuedFractionK[1, {i, 1, 10}]").unwrap(),
      "55/89"
    );
  }
  #[test]
  fn continued_fraction_k_five() {
    assert_eq!(
      interpret("ContinuedFractionK[k, {k, 1, 5}]").unwrap(),
      "157/225"
    );
  }
  #[test]
  fn counts_by_odd_q() {
    assert_eq!(
      interpret("CountsBy[{1, 2, 3, 4, 5}, OddQ]").unwrap(),
      "<|True -> 3, False -> 2|>"
    );
  }
  #[test]
  fn counts_by_string_length() {
    assert_eq!(
      interpret(
        "CountsBy[{\"a\", \"bb\", \"c\", \"dd\", \"eee\"}, StringLength]"
      )
      .unwrap(),
      "<|1 -> 2, 2 -> 2, 3 -> 1|>"
    );
  }
  #[test]
  fn find_linear_recurrence_fibonacci() {
    assert_eq!(
      interpret("FindLinearRecurrence[{1, 1, 2, 3, 5, 8, 13}]").unwrap(),
      "{1, 1}"
    );
  }
  #[test]
  fn find_linear_recurrence_powers_of_2() {
    assert_eq!(
      interpret("FindLinearRecurrence[{1, 2, 4, 8, 16, 32}]").unwrap(),
      "{2}"
    );
  }
  #[test]
  fn sss_triangle_345() {
    assert_eq!(
      interpret("SSSTriangle[3, 4, 5]").unwrap(),
      "Triangle[{{0, 0}, {5, 0}, {16/5, 12/5}}]"
    );
  }
  #[test]
  fn sss_triangle_equilateral() {
    assert_eq!(
      interpret("SSSTriangle[1, 1, 1]").unwrap(),
      "Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}]"
    );
  }
  #[test]
  fn fold_pair_list_add_mul() {
    assert_eq!(
      interpret("FoldPairList[{#1 + #2, #1*#2}&, 1, {1, 2, 3}]").unwrap(),
      "{2, 3, 5}"
    );
  }
  #[test]
  fn fold_pair_list_sub() {
    assert_eq!(
      interpret("FoldPairList[{#1 + #2, #1 - #2}&, 0, {1, 2, 3}]").unwrap(),
      "{1, 1, 0}"
    );
  }
  #[test]
  fn join_across_basic() {
    assert_eq!(
      interpret(
        "JoinAcross[{<|\"a\" -> 1, \"b\" -> 2|>}, {<|\"a\" -> 1, \"c\" -> 3|>}, \"a\"]"
      )
      .unwrap(),
      "{<|a -> 1, b -> 2, c -> 3|>}"
    );
  }
  #[test]
  fn join_across_multi() {
    assert_eq!(
      interpret(
        "JoinAcross[{<|\"a\" -> 1, \"b\" -> 2|>, <|\"a\" -> 2, \"b\" -> 3|>}, {<|\"a\" -> 1, \"c\" -> 10|>, <|\"a\" -> 2, \"c\" -> 20|>}, \"a\"]"
      )
      .unwrap(),
      "{<|a -> 1, b -> 2, c -> 10|>, <|a -> 2, b -> 3, c -> 20|>}"
    );
  }
  #[test]
  fn exponential_moving_average_real() {
    assert_eq!(
      interpret("ExponentialMovingAverage[{1, 2, 3, 4, 5}, 0.5]").unwrap(),
      "{1, 1.5, 2.25, 3.125, 4.0625}"
    );
  }
  #[test]
  fn exponential_moving_average_rational() {
    assert_eq!(
      interpret("ExponentialMovingAverage[{1, 2, 3, 4, 5}, 1/3]").unwrap(),
      "{1, 4/3, 17/9, 70/27, 275/81}"
    );
  }
  #[test]
  fn letter_counts_basic() {
    assert_eq!(
      interpret("LetterCounts[\"hello world\"]").unwrap(),
      "<|l -> 3, o -> 2, d -> 1, r -> 1, w -> 1, e -> 1, h -> 1|>"
    );
  }
  #[test]
  fn word_counts_basic() {
    assert_eq!(
      interpret("WordCounts[\"the cat sat on the mat\"]").unwrap(),
      "<|the -> 2, mat -> 1, on -> 1, sat -> 1, cat -> 1|>"
    );
  }
  #[test]
  fn orthogonal_matrix_q_identity() {
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{1, 0}, {0, 1}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn orthogonal_matrix_q_rotation() {
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{0, -1}, {1, 0}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn orthogonal_matrix_q_non_orthogonal() {
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{1, 1}, {0, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn circle_through_basic() {
    assert_eq!(
      interpret("CircleThrough[{{0, 0}, {1, 0}, {0, 1}}]").unwrap(),
      "Circle[{1/2, 1/2}, 1/Sqrt[2]]"
    );
  }
  #[test]
  fn circle_through_unit() {
    assert_eq!(
      interpret("CircleThrough[{{1, 0}, {-1, 0}, {0, 1}}]").unwrap(),
      "Circle[{0, 0}, 1]"
    );
  }
  #[test]
  fn numerical_sort_basic() {
    assert_eq!(
      interpret("NumericalSort[{\"b3\", \"a1\", \"c2\", \"a10\"}]").unwrap(),
      "{a1, a10, b3, c2}"
    );
  }
  #[test]
  fn numerical_sort_numbers() {
    assert_eq!(
      interpret("NumericalSort[{\"file10\", \"file2\", \"file1\"}]").unwrap(),
      "{file1, file2, file10}"
    );
  }
  #[test]
  fn from_coefficient_rules_basic() {
    assert_eq!(
      interpret("FromCoefficientRules[{{0} -> 1, {1} -> 3, {2} -> 5}, x]")
        .unwrap(),
      "1 + 3*x + 5*x^2"
    );
  }
  // PolynomialExtendedGCD skipped - requires polynomial GCD infrastructure
  #[test]
  fn count_distinct_basic() {
    assert_eq!(interpret("CountDistinct[{1, 2, 3, 2, 1, 4}]").unwrap(), "4");
  }
  #[test]
  fn count_distinct_strings() {
    assert_eq!(
      interpret("CountDistinct[{\"a\", \"b\", \"a\", \"c\"}]").unwrap(),
      "3"
    );
  }
  #[test]
  fn diagonalizable_matrix_q_true() {
    assert_eq!(
      interpret("DiagonalizableMatrixQ[{{1, 0}, {0, 2}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn diagonalizable_matrix_q_false() {
    assert_eq!(
      interpret("DiagonalizableMatrixQ[{{1, 1}, {0, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn positive_semidefinite_matrix_q_true() {
    assert_eq!(
      interpret("PositiveSemidefiniteMatrixQ[{{1, 0}, {0, 1}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn positive_semidefinite_matrix_q_false() {
    assert_eq!(
      interpret("PositiveSemidefiniteMatrixQ[{{1, 2}, {2, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn symmetric_polynomial_1() {
    assert_eq!(
      interpret("SymmetricPolynomial[1, {a, b, c}]").unwrap(),
      "a + b + c"
    );
  }
  #[test]
  fn symmetric_polynomial_2() {
    assert_eq!(
      interpret("SymmetricPolynomial[2, {a, b, c}]").unwrap(),
      "a*b + a*c + b*c"
    );
  }
  #[test]
  fn symmetric_polynomial_3() {
    assert_eq!(
      interpret("SymmetricPolynomial[3, {a, b, c}]").unwrap(),
      "a*b*c"
    );
  }
  #[test]
  fn adjugate_2x2() {
    assert_eq!(
      interpret("Adjugate[{{1, 2}, {3, 4}}]").unwrap(),
      "{{4, -2}, {-3, 1}}"
    );
  }
  #[test]
  fn adjugate_3x3() {
    assert_eq!(
      interpret("Adjugate[{{1, 2, 3}, {0, 4, 5}, {1, 0, 6}}]").unwrap(),
      "{{24, -12, -2}, {5, 3, -5}, {-4, 2, 4}}"
    );
  }
  #[test]
  fn coordinate_bounds_basic() {
    assert_eq!(
      interpret("CoordinateBounds[{{1, 5}, {3, 2}, {-1, 7}}]").unwrap(),
      "{{-1, 3}, {2, 7}}"
    );
  }
  #[test]
  fn glaisher_symbolic() {
    assert_eq!(interpret("Glaisher").unwrap(), "Glaisher");
  }
  #[test]
  fn glaisher_numeric() {
    assert_eq!(interpret("N[Glaisher]").unwrap(), "1.2824271291006226");
  }
  #[test]
  fn nminvalue_basic() {
    assert_eq!(interpret("NMinValue[x^2 + 3*x + 2, x]").unwrap(), "-0.25");
  }
  #[test]
  fn nmaxvalue_basic() {
    assert_eq!(interpret("NMaxValue[-x^2 + 3*x + 2, x]").unwrap(), "4.25");
  }
  #[test]
  fn find_arg_min_basic() {
    assert_eq!(interpret("FindArgMin[x^2 + 3*x + 2, x]").unwrap(), "{-1.5}");
  }
  #[test]
  fn find_arg_max_basic() {
    assert_eq!(interpret("FindArgMax[-x^2 + 3*x + 2, x]").unwrap(), "{1.5}");
  }
  #[test]
  fn string_replace_list_basic() {
    assert_eq!(
      interpret("StringReplaceList[\"abcabc\", \"a\" -> \"X\"]").unwrap(),
      "{Xbcabc, abcXbc}"
    );
  }
  #[test]
  fn string_replace_list_overlap() {
    assert_eq!(
      interpret("StringReplaceList[\"aaa\", \"aa\" -> \"X\"]").unwrap(),
      "{Xa, aX}"
    );
  }
  #[test]
  fn chessboard_distance_basic() {
    assert_eq!(
      interpret("ChessboardDistance[{1, 2}, {3, 5}]").unwrap(),
      "3"
    );
  }
  #[test]
  fn chessboard_distance_3d() {
    assert_eq!(
      interpret("ChessboardDistance[{1, 2, 3}, {4, 6, 5}]").unwrap(),
      "4"
    );
  }
  #[test]
  fn negative_definite_matrix_q_true() {
    assert_eq!(
      interpret("NegativeDefiniteMatrixQ[{{-2, 0}, {0, -3}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn negative_definite_matrix_q_false() {
    assert_eq!(
      interpret("NegativeDefiniteMatrixQ[{{-1, 0}, {0, 0}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn negative_semidefinite_matrix_q_true() {
    assert_eq!(
      interpret("NegativeSemidefiniteMatrixQ[{{-1, 0}, {0, 0}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn negative_semidefinite_matrix_q_false() {
    assert_eq!(
      interpret("NegativeSemidefiniteMatrixQ[{{1, 0}, {0, -1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn sequence_count_basic() {
    assert_eq!(
      interpret("SequenceCount[{1, 2, 3, 1, 2, 3, 1}, {1, 2}]").unwrap(),
      "2"
    );
  }
  #[test]
  fn sequence_count_no_match() {
    assert_eq!(interpret("SequenceCount[{1, 2, 3}, {4, 5}]").unwrap(), "0");
  }
  #[test]
  fn chebyshev_distance_basic() {
    assert_eq!(interpret("ChebyshevDistance[{1, 2}, {3, 5}]").unwrap(), "3");
  }
  #[test]
  fn hermitian_matrix_q_real_symmetric() {
    assert_eq!(
      interpret("HermitianMatrixQ[{{1, 0}, {0, 2}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn hermitian_matrix_q_false() {
    assert_eq!(
      interpret("HermitianMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn normal_matrix_q_diagonal() {
    assert_eq!(
      interpret("NormalMatrixQ[{{1, 0}, {0, 2}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn normal_matrix_q_false() {
    assert_eq!(
      interpret("NormalMatrixQ[{{1, 1}, {0, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn parallel_sum_basic() {
    assert_eq!(interpret("ParallelSum[i^2, {i, 1, 5}]").unwrap(), "55");
  }
  #[test]
  fn parallel_product_basic() {
    assert_eq!(interpret("ParallelProduct[i, {i, 1, 5}]").unwrap(), "120");
  }
  #[test]
  fn mangoldt_lambda_prime() {
    assert_eq!(interpret("MangoldtLambda[7]").unwrap(), "Log[7]");
  }
  #[test]
  fn mangoldt_lambda_prime_power() {
    assert_eq!(interpret("MangoldtLambda[8]").unwrap(), "Log[2]");
  }
  #[test]
  fn mangoldt_lambda_composite() {
    assert_eq!(interpret("MangoldtLambda[6]").unwrap(), "0");
  }
  #[test]
  fn mangoldt_lambda_one() {
    assert_eq!(interpret("MangoldtLambda[1]").unwrap(), "0");
  }
  #[test]
  fn liouville_lambda_basic() {
    assert_eq!(interpret("LiouvilleLambda[6]").unwrap(), "1");
  }
  #[test]
  fn liouville_lambda_prime() {
    assert_eq!(interpret("LiouvilleLambda[7]").unwrap(), "-1");
  }
  #[test]
  fn liouville_lambda_prime_power() {
    assert_eq!(interpret("LiouvilleLambda[8]").unwrap(), "-1");
  }
  #[test]
  fn bray_curtis_distance() {
    assert_eq!(
      interpret("BrayCurtisDistance[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "3/7"
    );
  }
  #[test]
  fn canberra_distance() {
    assert_eq!(
      interpret("CanberraDistance[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "143/105"
    );
  }
  #[test]
  fn cosine_distance_orthogonal() {
    assert_eq!(interpret("CosineDistance[{1, 0}, {0, 1}]").unwrap(), "1");
  }
  #[test]
  fn key_sort_by_basic() {
    assert_eq!(
      interpret(
        "KeySortBy[<|\"ba\" -> 2, \"a\" -> 1, \"ccc\" -> 3|>, StringLength]"
      )
      .unwrap(),
      "<|a -> 1, ba -> 2, ccc -> 3|>"
    );
  }
  #[test]
  fn max_filter_basic() {
    assert_eq!(
      interpret("MaxFilter[{1, 5, 2, 8, 3}, 1]").unwrap(),
      "{5, 5, 8, 8, 8}"
    );
  }
  #[test]
  fn min_filter_basic() {
    assert_eq!(
      interpret("MinFilter[{1, 5, 2, 8, 3}, 1]").unwrap(),
      "{1, 1, 2, 2, 3}"
    );
  }
  #[test]
  fn upsample_basic() {
    assert_eq!(
      interpret("Upsample[{a, b, c}, 2]").unwrap(),
      "{a, 0, b, 0, c, 0}"
    );
  }
  #[test]
  fn downsample_basic() {
    assert_eq!(
      interpret("Downsample[{a, b, c, d, e, f}, 2]").unwrap(),
      "{a, c, e}"
    );
  }
  #[test]
  fn euler_angles_identity() {
    assert_eq!(
      interpret("EulerAngles[{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]").unwrap(),
      "{0, 0, 0}"
    );
  }
  #[test]
  fn trimmed_mean_basic() {
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 100}, 0.25]").unwrap(),
      "5/2"
    );
  }
  #[test]
  fn trimmed_mean_larger() {
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]").unwrap(),
      "11/2"
    );
  }
  #[test]
  fn winsorized_mean_basic() {
    assert_eq!(
      interpret("WinsorizedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]")
        .unwrap(),
      "11/2"
    );
  }
  #[test]
  fn trimmed_variance_basic() {
    assert_eq!(
      interpret("TrimmedVariance[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]")
        .unwrap(),
      "7/2"
    );
  }
  #[test]
  fn winsorized_variance_basic() {
    assert_eq!(
      interpret("WinsorizedVariance[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]")
        .unwrap(),
      "85/18"
    );
  }

  // EqualTo operator form
  #[test]
  fn equal_to_true() {
    assert_eq!(interpret("EqualTo[5][5]").unwrap(), "True");
  }
  #[test]
  fn equal_to_false() {
    assert_eq!(interpret("EqualTo[5][3]").unwrap(), "False");
  }
  #[test]
  fn equal_to_symbolic() {
    assert_eq!(interpret("EqualTo[5][x]").unwrap(), "x == 5");
  }

  // GreaterThan, LessThan, etc.
  #[test]
  fn greater_than_false() {
    assert_eq!(interpret("GreaterThan[5][3]").unwrap(), "False");
  }
  #[test]
  fn greater_than_true() {
    assert_eq!(interpret("GreaterThan[5][7]").unwrap(), "True");
  }
  #[test]
  fn less_than_true() {
    assert_eq!(interpret("LessThan[5][3]").unwrap(), "True");
  }
  #[test]
  fn less_than_false() {
    assert_eq!(interpret("LessThan[5][7]").unwrap(), "False");
  }
  #[test]
  fn greater_equal_than_true() {
    assert_eq!(interpret("GreaterEqualThan[3][3]").unwrap(), "True");
  }
  #[test]
  fn greater_equal_than_false() {
    assert_eq!(interpret("GreaterEqualThan[5][3]").unwrap(), "False");
  }
  #[test]
  fn less_equal_than_true() {
    assert_eq!(interpret("LessEqualThan[3][3]").unwrap(), "True");
  }
  #[test]
  fn less_equal_than_false() {
    assert_eq!(interpret("LessEqualThan[3][5]").unwrap(), "False");
  }
  #[test]
  fn unequal_to_true() {
    assert_eq!(interpret("UnequalTo[5][3]").unwrap(), "True");
  }
  #[test]
  fn unequal_to_false() {
    assert_eq!(interpret("UnequalTo[5][5]").unwrap(), "False");
  }

  // FileNameDrop
  #[test]
  fn file_name_drop_default() {
    assert_eq!(interpret("FileNameDrop[\"a/b/c/d.txt\"]").unwrap(), "a/b/c");
  }
  #[test]
  fn file_name_drop_positive() {
    assert_eq!(
      interpret("FileNameDrop[\"a/b/c/d.txt\", 1]").unwrap(),
      "b/c/d.txt"
    );
  }
  #[test]
  fn file_name_drop_positive_2() {
    assert_eq!(
      interpret("FileNameDrop[\"a/b/c/d.txt\", 2]").unwrap(),
      "c/d.txt"
    );
  }
  #[test]
  fn file_name_drop_negative() {
    assert_eq!(
      interpret("FileNameDrop[\"a/b/c/d.txt\", -1]").unwrap(),
      "a/b/c"
    );
  }
  #[test]
  fn file_name_drop_negative_2() {
    assert_eq!(
      interpret("FileNameDrop[\"a/b/c/d.txt\", -2]").unwrap(),
      "a/b"
    );
  }

  // FromDMS
  #[test]
  fn from_dms_three() {
    assert_eq!(interpret("FromDMS[{40, 26, 46}]").unwrap(), "72803/1800");
  }
  #[test]
  fn from_dms_two() {
    assert_eq!(interpret("FromDMS[{40, 26}]").unwrap(), "1213/30");
  }
  #[test]
  fn from_dms_one() {
    assert_eq!(interpret("FromDMS[46]").unwrap(), "46");
  }

  // NArgMin / NArgMax
  #[test]
  fn nargmin_basic() {
    assert_eq!(interpret("NArgMin[x^2 + 3x + 1, x]").unwrap(), "-1.5");
  }
  #[test]
  fn nargmax_basic() {
    assert_eq!(interpret("NArgMax[-x^2 + 3x + 1, x]").unwrap(), "1.5");
  }

  // AddSides / SubtractSides / MultiplySides / DivideSides / ApplySides
  #[test]
  fn add_sides_basic() {
    assert_eq!(interpret("AddSides[x == 2, 3]").unwrap(), "3 + x == 5");
  }
  #[test]
  fn subtract_sides_basic() {
    assert_eq!(interpret("SubtractSides[x + 3 == 5, 3]").unwrap(), "x == 2");
  }
  #[test]
  fn multiply_sides_basic() {
    assert_eq!(interpret("MultiplySides[x == 2, 3]").unwrap(), "3*x == 6");
  }
  #[test]
  fn divide_sides_basic() {
    assert_eq!(interpret("DivideSides[2x == 6, 2]").unwrap(), "x == 3");
  }
  #[test]
  fn apply_sides_basic() {
    assert_eq!(interpret("ApplySides[f, x == y]").unwrap(), "f[x] == f[y]");
  }

  // DayCount
  #[test]
  fn day_count_basic() {
    assert_eq!(
      interpret("DayCount[{2020, 1, 1}, {2020, 12, 31}]").unwrap(),
      "365"
    );
  }
  #[test]
  fn day_count_same_month() {
    assert_eq!(
      interpret("DayCount[{2023, 1, 1}, {2023, 1, 31}]").unwrap(),
      "30"
    );
  }

  // ArrayResample
  #[test]
  fn array_resample_downsample() {
    assert_eq!(
      interpret("ArrayResample[{1, 2, 3, 4, 5}, 3]").unwrap(),
      "{1, 3, 5}"
    );
  }
  #[test]
  fn array_resample_upsample() {
    assert_eq!(
      interpret("ArrayResample[{1, 2, 3}, 5]").unwrap(),
      "{1, 3/2, 2, 5/2, 3}"
    );
  }
  #[test]
  fn array_resample_exact() {
    assert_eq!(
      interpret("ArrayResample[{10, 20, 30}, 5]").unwrap(),
      "{10, 15, 20, 25, 30}"
    );
  }

  // IntersectingQ
  #[test]
  fn intersecting_q_true() {
    assert_eq!(
      interpret("IntersectingQ[{1, 2, 3}, {3, 4, 5}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn intersecting_q_false() {
    assert_eq!(
      interpret("IntersectingQ[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn intersecting_q_empty() {
    assert_eq!(interpret("IntersectingQ[{}, {1}]").unwrap(), "False");
  }

  // AlternatingFactorial
  #[test]
  fn alternating_factorial_0() {
    assert_eq!(interpret("AlternatingFactorial[0]").unwrap(), "0");
  }
  #[test]
  fn alternating_factorial_1() {
    assert_eq!(interpret("AlternatingFactorial[1]").unwrap(), "1");
  }
  #[test]
  fn alternating_factorial_3() {
    assert_eq!(interpret("AlternatingFactorial[3]").unwrap(), "5");
  }
  #[test]
  fn alternating_factorial_5() {
    assert_eq!(interpret("AlternatingFactorial[5]").unwrap(), "101");
  }
  #[test]
  fn alternating_factorial_10() {
    assert_eq!(interpret("AlternatingFactorial[10]").unwrap(), "3301819");
  }

  // AlphabeticOrder
  #[test]
  fn alphabetic_order_less() {
    assert_eq!(
      interpret("AlphabeticOrder[\"apple\", \"banana\"]").unwrap(),
      "1"
    );
  }
  #[test]
  fn alphabetic_order_greater() {
    assert_eq!(
      interpret("AlphabeticOrder[\"banana\", \"apple\"]").unwrap(),
      "-1"
    );
  }
  #[test]
  fn alphabetic_order_equal() {
    assert_eq!(
      interpret("AlphabeticOrder[\"apple\", \"apple\"]").unwrap(),
      "0"
    );
  }

  // BinaryDistance
  #[test]
  fn binary_distance_same() {
    assert_eq!(
      interpret("BinaryDistance[{1, 0, 1}, {1, 0, 1}]").unwrap(),
      "0"
    );
  }
  #[test]
  fn binary_distance_different() {
    assert_eq!(
      interpret("BinaryDistance[{1, 0, 1, 1}, {1, 1, 0, 1}]").unwrap(),
      "1"
    );
  }

  // SquaresR
  #[test]
  fn squares_r_2_5() {
    assert_eq!(interpret("SquaresR[2, 5]").unwrap(), "8");
  }
  #[test]
  fn squares_r_2_25() {
    assert_eq!(interpret("SquaresR[2, 25]").unwrap(), "12");
  }
  #[test]
  fn squares_r_2_0() {
    assert_eq!(interpret("SquaresR[2, 0]").unwrap(), "1");
  }
  #[test]
  fn squares_r_1_4() {
    assert_eq!(interpret("SquaresR[1, 4]").unwrap(), "2");
  }
  #[test]
  fn squares_r_1_3() {
    assert_eq!(interpret("SquaresR[1, 3]").unwrap(), "0");
  }
  #[test]
  fn squares_r_4_5() {
    assert_eq!(interpret("SquaresR[4, 5]").unwrap(), "48");
  }

  // HankelMatrix
  #[test]
  fn hankel_matrix_basic() {
    assert_eq!(
      interpret("HankelMatrix[{1, 2, 3}]").unwrap(),
      "{{1, 2, 3}, {2, 3, 0}, {3, 0, 0}}"
    );
  }
  #[test]
  fn hankel_matrix_two_args() {
    assert_eq!(
      interpret("HankelMatrix[{1, 2, 3}, {3, 4, 5}]").unwrap(),
      "{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}"
    );
  }

  // HadamardMatrix
  #[test]
  fn hadamard_matrix_2() {
    assert_eq!(interpret("HadamardMatrix[2]").unwrap(), "{{1, 1}, {1, -1}}");
  }
  #[test]
  fn hadamard_matrix_4() {
    assert_eq!(
      interpret("HadamardMatrix[4]").unwrap(),
      "{{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}}"
    );
  }

  // PrimitiveRootList
  #[test]
  fn primitive_root_list_7() {
    assert_eq!(interpret("PrimitiveRootList[7]").unwrap(), "{3, 5}");
  }
  #[test]
  fn primitive_root_list_13() {
    assert_eq!(interpret("PrimitiveRootList[13]").unwrap(), "{2, 6, 7, 11}");
  }

  // DMSList (inverse of FromDMS)
  #[test]
  fn dms_list_basic() {
    assert_eq!(interpret("DMSList[72803/1800]").unwrap(), "{40, 26, 46}");
  }

  // WordCount
  #[test]
  fn word_count_basic() {
    assert_eq!(
      interpret("WordCount[\"hello world foo bar\"]").unwrap(),
      "4"
    );
  }
  #[test]
  fn word_count_empty() {
    assert_eq!(interpret("WordCount[\"\"]").unwrap(), "0");
  }

  // CenterArray
  #[test]
  fn center_array_basic() {
    assert_eq!(
      interpret("CenterArray[{a, b, c}, 7]").unwrap(),
      "{0, 0, a, b, c, 0, 0}"
    );
  }
  #[test]
  fn center_array_smaller() {
    assert_eq!(interpret("CenterArray[{a, b, c}, 2]").unwrap(), "{a, b}");
  }

  // ScalingMatrix
  #[test]
  fn scaling_matrix_2d() {
    assert_eq!(
      interpret("ScalingMatrix[{2, 3}]").unwrap(),
      "{{2, 0}, {0, 3}}"
    );
  }
  #[test]
  fn scaling_matrix_3d() {
    assert_eq!(
      interpret("ScalingMatrix[{1, 2, 3}]").unwrap(),
      "{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}"
    );
  }

  // ReverseSortBy
  #[test]
  fn reverse_sort_by_basic() {
    assert_eq!(
      interpret("ReverseSortBy[{3, 1, 2, 5, 4}, Identity]").unwrap(),
      "{5, 4, 3, 2, 1}"
    );
  }

  // CorrelationDistance
  #[test]
  fn correlation_distance_perfect() {
    assert_eq!(
      interpret("CorrelationDistance[{1, 2, 3}, {2, 4, 6}]").unwrap(),
      "0"
    );
  }

  // PowerModList
  #[test]
  fn power_mod_list_basic() {
    assert_eq!(
      interpret("PowerModList[2, 5, 7]").unwrap(),
      "{2, 4, 1, 2, 4}"
    );
  }
  #[test]
  fn power_mod_list_3() {
    assert_eq!(interpret("PowerModList[3, 4, 5]").unwrap(), "{3, 4, 2, 1}");
  }

  // AntisymmetricMatrixQ
  #[test]
  fn antisymmetric_matrix_q_true() {
    assert_eq!(
      interpret("AntisymmetricMatrixQ[{{0, 1, -2}, {-1, 0, 3}, {2, -3, 0}}]")
        .unwrap(),
      "True"
    );
  }
  #[test]
  fn antisymmetric_matrix_q_false() {
    assert_eq!(
      interpret("AntisymmetricMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }

  // ShearingMatrix
  #[test]
  fn shearing_matrix_2d() {
    assert_eq!(
      interpret("ShearingMatrix[2, {1, 0}, {0, 1}]").unwrap(),
      "{{1, 2}, {0, 1}}"
    );
  }

  // DiagonalMatrixQ
  #[test]
  fn diagonal_matrix_q_true() {
    assert_eq!(
      interpret("DiagonalMatrixQ[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn diagonal_matrix_q_false() {
    assert_eq!(
      interpret("DiagonalMatrixQ[{{1, 2}, {0, 3}}]").unwrap(),
      "False"
    );
  }

  // UpperTriangularMatrixQ
  #[test]
  fn upper_triangular_q_true() {
    assert_eq!(
      interpret("UpperTriangularMatrixQ[{{1, 2, 3}, {0, 4, 5}, {0, 0, 6}}]")
        .unwrap(),
      "True"
    );
  }
  #[test]
  fn upper_triangular_q_false() {
    assert_eq!(
      interpret("UpperTriangularMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }

  // LowerTriangularMatrixQ
  #[test]
  fn lower_triangular_q_true() {
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{{1, 0, 0}, {2, 3, 0}, {4, 5, 6}}]")
        .unwrap(),
      "True"
    );
  }
  #[test]
  fn lower_triangular_q_false() {
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }

  // KroneckerSymbol
  #[test]
  fn kronecker_symbol_basic() {
    assert_eq!(interpret("KroneckerSymbol[2, 7]").unwrap(), "1");
  }
  #[test]
  fn kronecker_symbol_neg() {
    assert_eq!(interpret("KroneckerSymbol[3, 7]").unwrap(), "-1");
  }
  #[test]
  fn kronecker_symbol_zero() {
    assert_eq!(interpret("KroneckerSymbol[7, 7]").unwrap(), "0");
  }

  // NormalizedSquaredEuclideanDistance
  #[test]
  fn normalized_sqeuclidean_same() {
    assert_eq!(
      interpret("NormalizedSquaredEuclideanDistance[{1, 2, 3}, {1, 2, 3}]")
        .unwrap(),
      "0"
    );
  }

  // CrossMatrix
  #[test]
  fn cross_matrix_basic() {
    assert_eq!(
      interpret("CrossMatrix[{1, 0, 0}]").unwrap(),
      "{{0, 0, 0}, {0, 0, -1}, {0, 1, 0}}"
    );
  }
  #[test]
  fn cross_matrix_general() {
    assert_eq!(
      interpret("CrossMatrix[{a, b, c}]").unwrap(),
      "{{0, -c, b}, {c, 0, -a}, {-b, a, 0}}"
    );
  }

  // FourierMatrix
  #[test]
  fn fourier_matrix_1() {
    assert_eq!(interpret("FourierMatrix[1]").unwrap(), "{{1}}");
  }
  #[test]
  fn fourier_matrix_2() {
    // FourierMatrix[2] entry (2,2) = E^(I*Pi)/Sqrt[2]
    assert_eq!(
      interpret("FourierMatrix[2]").unwrap(),
      "{{1/Sqrt[2], 1/Sqrt[2]}, {1/Sqrt[2], E^(I*Pi)/Sqrt[2]}}"
    );
  }

  // Symmetrize
  #[test]
  fn symmetrize_symmetric() {
    assert_eq!(
      interpret("Symmetrize[{{1, 2}, {2, 3}}]").unwrap(),
      "{{1, 2}, {2, 3}}"
    );
  }
  #[test]
  fn symmetrize_asymmetric() {
    assert_eq!(
      interpret("Symmetrize[{{1, 2}, {4, 3}}]").unwrap(),
      "{{1, 3}, {3, 3}}"
    );
  }

  // DisjointQ
  #[test]
  fn disjoint_q_true() {
    assert_eq!(
      interpret("DisjointQ[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn disjoint_q_false() {
    assert_eq!(
      interpret("DisjointQ[{1, 2, 3}, {3, 4, 5}]").unwrap(),
      "False"
    );
  }

  // CoordinateBoundsArray
  #[test]
  fn coordinate_bounds_array_basic() {
    assert_eq!(
      interpret("CoordinateBoundsArray[{{0, 1}, {0, 2}}]").unwrap(),
      "{{0, 1}, {0, 2}}"
    );
  }
  #[test]
  fn coordinate_bounds_array_padded() {
    assert_eq!(
      interpret("CoordinateBoundsArray[{{0, 10}, {0, 20}}, 1]").unwrap(),
      "{{-1, 11}, {-1, 21}}"
    );
  }

  // FindPermutation
  #[test]
  fn find_permutation_basic() {
    assert_eq!(
      interpret("FindPermutation[{a, b, c}, {b, c, a}]").unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
  }
  #[test]
  fn find_permutation_identity() {
    assert_eq!(
      interpret("FindPermutation[{a, b, c}, {a, b, c}]").unwrap(),
      "Cycles[{}]"
    );
  }

  // KeyMemberQ
  #[test]
  fn key_member_q_true() {
    assert_eq!(
      interpret("KeyMemberQ[<|\"a\" -> 1, \"b\" -> 2|>, \"a\"]").unwrap(),
      "True"
    );
  }
  #[test]
  fn key_member_q_false() {
    assert_eq!(
      interpret("KeyMemberQ[<|\"a\" -> 1, \"b\" -> 2|>, \"c\"]").unwrap(),
      "False"
    );
  }

  // PermutationOrder
  #[test]
  fn permutation_order_identity() {
    assert_eq!(interpret("PermutationOrder[{1, 2, 3}]").unwrap(), "1");
  }
  #[test]
  fn permutation_order_swap() {
    assert_eq!(interpret("PermutationOrder[{2, 1, 3}]").unwrap(), "2");
  }
  #[test]
  fn permutation_order_cycle3() {
    assert_eq!(interpret("PermutationOrder[{2, 3, 1}]").unwrap(), "3");
  }

  // PermutationPower
  #[test]
  fn permutation_power_identity() {
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, 3]").unwrap(),
      "{1, 2, 3}"
    );
  }
  #[test]
  fn permutation_power_square() {
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, 2]").unwrap(),
      "{3, 1, 2}"
    );
  }

  // PermutationLength
  #[test]
  fn permutation_length_identity() {
    assert_eq!(interpret("PermutationLength[{1, 2, 3}]").unwrap(), "0");
  }
  #[test]
  fn permutation_length_swap() {
    assert_eq!(interpret("PermutationLength[{2, 1, 3}]").unwrap(), "2");
  }

  // PermutationListQ
  #[test]
  fn permutation_list_q_true() {
    assert_eq!(interpret("PermutationListQ[{2, 3, 1}]").unwrap(), "True");
  }
  #[test]
  fn permutation_list_q_false() {
    assert_eq!(interpret("PermutationListQ[{1, 1, 2}]").unwrap(), "False");
  }

  // FoldWhileList
  #[test]
  fn fold_while_list_basic() {
    assert_eq!(
      interpret("FoldWhileList[Plus, 0, {1, 2, 3, 4, 5}, Function[# < 10]]")
        .unwrap(),
      "{0, 1, 3, 6}"
    );
  }

  // PermutationCyclesQ
  #[test]
  fn permutation_cycles_q_true() {
    assert_eq!(
      interpret("PermutationCyclesQ[Cycles[{{1, 3}, {2, 4}}]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn permutation_cycles_q_false() {
    assert_eq!(interpret("PermutationCyclesQ[{1, 2, 3}]").unwrap(), "False");
  }

  // PermutationSupport
  #[test]
  fn permutation_support_basic() {
    assert_eq!(
      interpret("PermutationSupport[{2, 1, 3}]").unwrap(),
      "{1, 2}"
    );
  }
  #[test]
  fn permutation_support_identity() {
    assert_eq!(interpret("PermutationSupport[{1, 2, 3}]").unwrap(), "{}");
  }

  // PermutationMax
  #[test]
  fn permutation_max_basic() {
    assert_eq!(interpret("PermutationMax[{2, 1, 3, 4}]").unwrap(), "2");
  }

  // PermutationMin
  #[test]
  fn permutation_min_basic() {
    assert_eq!(interpret("PermutationMin[{2, 1, 3, 4}]").unwrap(), "1");
  }

  // Splice
  #[test]
  fn splice_basic() {
    assert_eq!(interpret("{1, Splice[{2, 3}], 4}").unwrap(), "{1, 2, 3, 4}");
  }

  // SubsetMap
  #[test]
  fn subset_map_basic() {
    assert_eq!(
      interpret("SubsetMap[Reverse, {a, b, c, d, e}, {2, 4}]").unwrap(),
      "{a, d, c, b, e}"
    );
  }

  // Assert
  #[test]
  fn assert_true() {
    assert_eq!(interpret("Assert[1 + 1 == 2]").unwrap(), "Null");
  }
  #[test]
  fn assert_false() {
    assert!(interpret("Assert[1 + 1 == 3]").is_err());
  }

  // StarGraph
  #[test]
  fn star_graph_basic() {
    assert_eq!(
      interpret("StarGraph[4]").unwrap(),
      "Graph[{1, 2, 3, 4}, {UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[1, 4]}]"
    );
  }

  // CirculantGraph
  #[test]
  fn circulant_graph_basic() {
    let result = interpret("CirculantGraph[4, {1}]").unwrap();
    assert!(result.starts_with("Graph[{1, 2, 3, 4}"));
  }

  // KaryTree
  #[test]
  fn kary_tree_binary() {
    let result = interpret("KaryTree[7]").unwrap();
    assert!(result.starts_with("Graph[{1, 2, 3, 4, 5, 6, 7}"));
    assert!(result.contains("UndirectedEdge[1, 2]"));
    assert!(result.contains("UndirectedEdge[1, 3]"));
  }

  // HypercubeGraph
  #[test]
  fn hypercube_graph_2() {
    let result = interpret("HypercubeGraph[2]").unwrap();
    assert!(result.starts_with("Graph[{1, 2, 3, 4}"));
  }

  // EdgeQ
  #[test]
  fn edge_q_true() {
    assert_eq!(
      interpret("EdgeQ[CompleteGraph[3], UndirectedEdge[1, 2]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn edge_q_false() {
    assert_eq!(
      interpret("EdgeQ[StarGraph[3], UndirectedEdge[2, 3]]").unwrap(),
      "False"
    );
  }

  // Booleans domain
  #[test]
  fn booleans_element() {
    assert_eq!(interpret("Element[True, Booleans]").unwrap(), "True");
  }
}
