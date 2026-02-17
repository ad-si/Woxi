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
    assert_eq!(interpret("Function[# + 1]").unwrap(), "#1 + 1&");
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

mod reverse_extended {
  use super::*;

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
      "2*t + c + d"
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
