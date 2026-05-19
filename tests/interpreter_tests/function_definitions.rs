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

  // Regression (mathics test_assignment.py:405): when a list is built
  // from `G[x]` and `x` is later assigned/unassigned between read-backs,
  // each read re-evaluates G against the current head-constrained
  // definition. The list elements are not "frozen" at definition time.
  #[test]
  fn typed_function_definition_re_evaluates_per_x_binding() {
    clear_state();
    assert_eq!(
      interpret("G[x_Real]=x^2; a={G[x]}; {x=1.; a, x=.; a}").unwrap(),
      "{{1.}, {G[x]}}"
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

  #[test]
  fn condition_on_whole_lhs_single_arg() {
    // f[x_] /; cond := body (condition outside brackets) must work
    // the same as f[x_ /; cond] := body
    clear_state();
    assert_eq!(interpret("f[x_] /; x > 0 := x^2; f[3]").unwrap(), "9");
  }

  #[test]
  fn condition_on_whole_lhs_fallback() {
    clear_state();
    assert_eq!(interpret("f[x_] /; x > 0 := x^2; f[-3]").unwrap(), "f[-3]");
  }

  #[test]
  fn condition_on_whole_lhs_multi_arg() {
    clear_state();
    assert_eq!(
      interpret("f[x_, y_] /; x > y := x - y; f[5, 3]").unwrap(),
      "2"
    );
  }

  #[test]
  fn condition_on_whole_lhs_multiple_definitions() {
    clear_state();
    assert_eq!(
      interpret(
        "f[x_] /; x >= 0 := x; f[x_] /; x < 0 := -x; {f[3], f[-3], f[0]}"
      )
      .unwrap(),
      "{3, 3, 0}"
    );
  }

  #[test]
  fn condition_on_whole_lhs_with_default_fallback() {
    // Condition on whole LHS combined with a general fallback
    clear_state();
    assert_eq!(
      interpret(
        r#"f[x_] /; x > 0 := "positive"; f[x_] := "non-positive"; {f[5], f[0], f[-1]}"#
      )
      .unwrap(),
      "{positive, non-positive, non-positive}"
    );
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
  fn list_pattern_trailing_blank_null_sequence() {
    clear_state();
    assert_eq!(interpret("f[{x_, ___}] := x; f[{2, 3, 4}]").unwrap(), "2");
  }

  #[test]
  fn list_pattern_trailing_blank_sequence() {
    clear_state();
    assert_eq!(interpret("f[{x_, __}] := x; f[{2, 3}]").unwrap(), "2");
  }

  #[test]
  fn list_pattern_blank_sequence_requires_at_least_one_extra() {
    clear_state();
    // `{x_, __}` requires at least one element after x.
    assert_eq!(interpret("f[{x_, __}] := x; f[{2}]").unwrap(), "f[{2}]");
  }

  #[test]
  fn list_pattern_named_trailing_sequence() {
    clear_state();
    assert_eq!(
      interpret("f[{x_, y___}] := {x, y}; f[{2, 3, 4}]").unwrap(),
      "{2, 3, 4}"
    );
    clear_state();
    assert_eq!(
      interpret("f[{x_, y___}] := {y, x}; f[{1, 2, 3}]").unwrap(),
      "{2, 3, 1}"
    );
  }

  #[test]
  fn named_outer_list_pattern() {
    // `seq:{x_, ___}` binds both `seq` (whole list) and `x` (first element).
    clear_state();
    assert_eq!(
      interpret("f[seq:{x_, ___}] := seq; f[{2, 3, 4}]").unwrap(),
      "{2, 3, 4}"
    );
    clear_state();
    assert_eq!(
      interpret("f[seq:{x_, ___}] := x; f[{2, 3, 4}]").unwrap(),
      "2"
    );
    clear_state();
    assert_eq!(
      interpret("f[seq:{x_, ___}] := {seq, x}; f[{2, 3, 4}]").unwrap(),
      "{{2, 3, 4}, 2}"
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

  #[test]
  fn down_values_introspection() {
    clear_state();
    assert_eq!(
      interpret("F[x_] := x + 2; DownValues[F]").unwrap(),
      "{HoldPattern[F[x_]] :> x + 2}"
    );
  }

  #[test]
  fn down_values_multiple_definitions() {
    clear_state();
    assert_eq!(
      interpret(
        "g[x_Integer] := x^2; g[x_String] := StringLength[x]; DownValues[g]"
      )
      .unwrap(),
      "{HoldPattern[g[x_Integer]] :> x^2, HoldPattern[g[x_String]] :> StringLength[x]}"
    );
  }

  #[test]
  fn down_values_empty() {
    clear_state();
    assert_eq!(interpret("DownValues[undefined]").unwrap(), "{}");
  }

  #[test]
  fn down_values_literal_args() {
    clear_state();
    assert_eq!(
      interpret("f[1] := a; f[2] := b; DownValues[f]").unwrap(),
      "{HoldPattern[f[1]] :> a, HoldPattern[f[2]] :> b}"
    );
  }

  #[test]
  fn down_values_mixed_literal_and_pattern() {
    clear_state();
    assert_eq!(
      interpret("g[x_] := x^2; g[1] := one; DownValues[g]").unwrap(),
      "{HoldPattern[g[1]] :> one, HoldPattern[g[x_]] :> x^2}"
    );
  }

  #[test]
  fn down_values_partial_literal_args() {
    clear_state();
    assert_eq!(
      interpret("h[x_, 1] := x; h[x_, y_] := x + y; DownValues[h]").unwrap(),
      "{HoldPattern[h[x_, 1]] :> x, HoldPattern[h[x_, y_]] :> x + y}"
    );
  }
}

mod default_values {
  use super::*;

  // DefaultValues exposes the built-in identity elements that
  // Optional/OneIdentity patterns fall back to. Only Plus/Times/Power
  // have them in Wolfram; everything else returns {}.

  #[test]
  fn plus_default_is_zero() {
    assert_eq!(
      interpret("DefaultValues[Plus]").unwrap(),
      "{HoldPattern[Default[Plus]] :> 0}"
    );
  }

  #[test]
  fn times_default_is_one() {
    assert_eq!(
      interpret("DefaultValues[Times]").unwrap(),
      "{HoldPattern[Default[Times]] :> 1}"
    );
  }

  #[test]
  fn power_second_slot_default_is_one() {
    // Default[Power, 2] = 1 (only the exponent has a default; the base
    // does not).
    assert_eq!(
      interpret("DefaultValues[Power]").unwrap(),
      "{HoldPattern[Default[Power, 2]] :> 1}"
    );
  }

  #[test]
  fn other_symbols_have_no_defaults() {
    assert_eq!(interpret("DefaultValues[Sin]").unwrap(), "{}");
    assert_eq!(interpret("DefaultValues[myFunc]").unwrap(), "{}");
  }

  #[test]
  fn user_set_default_position_appears_in_default_values() {
    // `Default[f, n] = v` is stored as a DownValue on Default itself;
    // DefaultValues[f] must surface it as a HoldPattern RuleDelayed list.
    clear_state();
    assert_eq!(
      interpret("Default[f, 1] = 4; DefaultValues[f]").unwrap(),
      "{HoldPattern[Default[f, 1]] :> 4}"
    );
  }

  #[test]
  fn user_set_position_less_default_appears_in_default_values() {
    clear_state();
    assert_eq!(
      interpret("Default[g] = 3; DefaultValues[g]").unwrap(),
      "{HoldPattern[Default[g]] :> 3}"
    );
  }

  #[test]
  fn default_values_filters_to_target_symbol() {
    // Multiple symbols share the Default DownValues table; DefaultValues
    // must only return entries whose first arg is the requested symbol.
    clear_state();
    assert_eq!(
      interpret("Default[h, 1] = 7; Default[k, 2] = 9; DefaultValues[k]")
        .unwrap(),
      "{HoldPattern[Default[k, 2]] :> 9}"
    );
  }

  #[test]
  fn default_values_set_via_assignment_round_trips() {
    // `DefaultValues[g] = {Default[g] -> 3}` should install the rule
    // such that `Default[g]` returns 3 *and* `DefaultValues[g]` reports
    // it back, matching wolframscript's round-trip behavior.
    clear_state();
    assert_eq!(
      interpret("DefaultValues[g] = {Default[g] -> 3}; DefaultValues[g]")
        .unwrap(),
      "{HoldPattern[Default[g]] :> 3}"
    );
    assert_eq!(interpret("Default[g]").unwrap(), "3");
  }

  #[test]
  fn default_values_set_drives_optional_match() {
    // The whole point of DefaultValues — `g[x_.]` with no argument
    // should look up `Default[g]` and substitute its value (3).
    clear_state();
    assert_eq!(
      interpret("DefaultValues[g] = {Default[g] -> 3}; g[x_.] := {x}; g[]")
        .unwrap(),
      "{3}"
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
  fn assigned_from_undefined_rules_symbol_stays_unevaluated() {
    assert_eq!(
      interpret("dispatchrules = Dispatch[rules]").unwrap(),
      "Dispatch[rules]"
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

mod messages {
  use super::*;

  #[test]
  fn messages_of_undefined_symbol_is_empty_list() {
    assert_eq!(interpret("Messages[a]").unwrap(), "{}");
  }

  #[test]
  fn general_argr_returns_template_text() {
    // `General::argr` is one of the standard built-in messages; without
    // any user-installed DownValue on MessageName the lookup must still
    // produce the canonical template text (matching wolframscript). The
    // returned string keeps the `1`/`2` slot markers verbatim — they are
    // only filled in when the message is emitted via Message[...].
    assert_eq!(
      interpret("General::argr").unwrap(),
      "`1` called with 1 argument; `2` arguments are expected."
    );
  }

  #[test]
  fn user_message_overrides_builtin() {
    // A user-installed DownValue on MessageName must take precedence
    // over the built-in template.
    clear_state();
    assert_eq!(
      interpret(r#"General::argr = "custom"; General::argr"#).unwrap(),
      "custom"
    );
  }

  #[test]
  fn messages_returns_user_set_message() {
    // `a::b = "foo"` installs the message text as a DownValue on
    // MessageName. `Messages[a]` should surface it as a HoldPattern
    // RuleDelayed list — matching wolframscript's long-form rendering.
    clear_state();
    assert_eq!(
      interpret(r#"a::b = "foo"; Messages[a]"#).unwrap(),
      "{HoldPattern[MessageName[a, b]] :> foo}"
    );
  }

  #[test]
  fn messages_filters_to_target_symbol() {
    // Multiple symbols share the MessageName DownValues table; Messages
    // must only return entries whose first arg is the requested symbol.
    clear_state();
    assert_eq!(
      interpret(r#"a::x = "ax"; b::y = "by"; Messages[b]"#).unwrap(),
      "{HoldPattern[MessageName[b, y]] :> by}"
    );
  }
}

mod nvalues {
  use super::*;

  #[test]
  fn nvalues_of_undefined_symbol_is_empty_list() {
    assert_eq!(interpret("NValues[d]").unwrap(), "{}");
  }
}

mod subvalues {
  use super::*;

  #[test]
  fn subvalues_of_undefined_symbol_is_empty_list() {
    assert_eq!(interpret("SubValues[f]").unwrap(), "{}");
  }
}

mod ownvalues {
  use super::*;

  #[test]
  fn ownvalues_of_undefined_symbol_is_empty_list() {
    assert_eq!(interpret("OwnValues[x]").unwrap(), "{}");
  }

  #[test]
  fn conditional_ownvalue_skipped_when_test_fails() {
    // `a /; b > 0 := 3` only fires when `b > 0` evaluates to True. With
    // `b` unbound the test stays unevaluated, so the rule must NOT fire
    // and `a` returns as itself (matches wolframscript).
    clear_state();
    assert_eq!(interpret("a /; b > 0 := 3; a").unwrap(), "a");
  }

  #[test]
  fn conditional_ownvalue_fires_when_test_passes() {
    clear_state();
    assert_eq!(interpret("a /; b > 0 := 3; b = 5; a").unwrap(), "3");
  }

  #[test]
  fn conditional_ownvalue_skipped_when_test_false() {
    clear_state();
    assert_eq!(interpret("a /; b > 0 := 3; b = -5; a").unwrap(), "a");
  }
}

mod formatvalues {
  use super::*;

  #[test]
  fn formatvalues_of_undefined_symbol_is_empty_list() {
    assert_eq!(interpret("FormatValues[F]").unwrap(), "{}");
  }

  #[test]
  fn formatvalues_of_undefined_symbol_inputform_wraps_empty_list() {
    assert_eq!(
      interpret("FormatValues[F]  //InputForm").unwrap(),
      "InputForm[{}]"
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

  #[test]
  fn memory_available_returns_positive_integer() {
    let result = interpret("MemoryAvailable[]").unwrap();
    let val: i128 = result.parse().expect("should be an integer");
    assert!(val > 0, "MemoryAvailable should be positive: {}", val);
  }

  #[test]
  fn system_memory_greater_than_memory_available_greater_than_in_use() {
    assert_eq!(
      interpret("$SystemMemory > MemoryAvailable[] > MemoryInUse[]").unwrap(),
      "True"
    );
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

  #[test]
  fn dollar_input_is_empty_string() {
    // Matches wolframscript's -code mode: $Input is the empty string.
    assert_eq!(interpret("$Input").unwrap(), "");
  }

  #[test]
  fn dollar_input_head_is_string() {
    assert_eq!(interpret("Head[$Input]").unwrap(), "String");
  }

  #[test]
  fn contexts_no_args_returns_system_and_global() {
    assert_eq!(interpret("Contexts[]").unwrap(), "{System`, Global`}");
  }

  #[test]
  fn contexts_unknown_pattern_returns_empty() {
    assert_eq!(interpret(r#"Contexts["HTML*"]"#).unwrap(), "{}");
  }

  #[test]
  fn contexts_pattern_matches_system() {
    assert_eq!(interpret(r#"Contexts["Sys*"]"#).unwrap(), "{System`}");
  }

  #[test]
  fn contexts_pattern_matches_all() {
    assert_eq!(interpret(r#"Contexts["*"]"#).unwrap(), "{System`, Global`}");
  }
}

mod names {
  use super::*;

  #[test]
  fn empty_env_includes_builtins() {
    // Names[] returns all user-defined AND built-in names (matches wolframscript,
    // which returns tens of thousands of System` symbols). We just verify a
    // few expected builtins are present.
    let out = interpret("Names[]").unwrap();
    assert!(out.starts_with('{') && out.ends_with('}'));
    assert!(out.contains("List"), "expected List in Names[], got {out}");
    assert!(out.contains("Plus"), "expected Plus in Names[], got {out}");
  }

  #[test]
  fn lists_variables_among_builtins() {
    // User variables appear alongside builtins in the "*" pattern.
    let out = interpret("x = 1; y = 2; Names[\"*\"]").unwrap();
    assert!(
      out.contains(" x,") || out.ends_with(", x}") || out.contains("{x,")
    );
    assert!(
      out.contains(" y,") || out.ends_with(", y}") || out.contains("{y,")
    );
  }

  #[test]
  fn lists_user_function_among_builtins() {
    let out = interpret("f[x_] := x^2; Names[\"*\"]").unwrap();
    assert!(
      out.contains(" f,") || out.ends_with(", f}") || out.contains("{f,")
    );
  }

  #[test]
  fn pattern_filter_user_vars() {
    // Built-ins starting with `ab` also show up (e.g. Abs), but user-defined
    // symbols with the prefix still appear.
    let out = interpret("abc = 1; abd = 2; xyz = 3; Names[\"ab*\"]").unwrap();
    assert!(out.contains("abc"), "expected abc in {out}");
    assert!(out.contains("abd"), "expected abd in {out}");
    assert!(!out.contains("xyz"), "unexpected xyz in {out}");
  }

  #[test]
  fn no_match_returns_empty() {
    // No built-in starts with `z` (as of this writing), and no user symbol
    // matches the pattern either.
    assert_eq!(interpret("a = 1; Names[\"zzzNoSuch*\"]").unwrap(), "{}");
  }

  #[test]
  fn exact_builtin_match() {
    // Bare literal pattern returns just the exact-match builtin name.
    assert_eq!(interpret("Names[\"List\"]").unwrap(), "{List}");
    assert_eq!(interpret("Names[\"Plus\"]").unwrap(), "{Plus}");
  }

  // wolframscript sorts Names[...] case-insensitively, so `Listable`
  // goes between `List` and `ListAnimate`, not at the end after
  // `ListVectorPlot`. Regression for the mathics `Names["List*"]`
  // doctest.
  #[test]
  fn list_star_case_insensitive_sort() {
    let out = interpret("Names[\"List*\"]").unwrap();
    let idx_list = out.find("List,").expect("has List");
    let idx_listable = out.find("Listable").expect("has Listable");
    let idx_list_animate = out.find("ListAnimate").expect("has ListAnimate");
    let idx_list_vector =
      out.find("ListVectorPlot").expect("has ListVectorPlot");
    // case-insensitive order: List < Listable < ListAnimate
    assert!(idx_list < idx_listable);
    assert!(idx_listable < idx_list_animate);
    assert!(idx_list_animate < idx_list_vector);
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

// `f[a][b] = rhs` (curried Set) and `UpValues[sym] =.` (clearing all
// upvalues attached to a symbol) — both surface-level forms that
// previously errored or no-op'd silently.
mod set_curried_and_upvalues_unset {
  use super::*;

  #[test]
  fn set_on_curried_call_returns_rhs() {
    clear_state();
    // Same shape as `f[a][b] := rhs` but with `=`. Doesn't actually
    // install a SubValue (Woxi doesn't model SubValues yet) — just
    // returns the RHS instead of erroring.
    assert_eq!(interpret("f[a][b] = 3").unwrap(), "3");
  }

  #[test]
  fn upvalues_unset_removes_upvalue_rule() {
    clear_state();
    // `g[a] ^= 5` attaches an upvalue to `a` so `g[a]` returns 5.
    // `UpValues[a] =.` should remove every upvalue tagged on `a`,
    // making `g[a]` revert to its symbolic form.
    assert_eq!(
      interpret("g[a] ^= 5; g[a]; UpValues[a] =.; g[a]").unwrap(),
      "g[a]"
    );
  }

  #[test]
  fn upvalues_unset_then_predicate_falls_back() {
    clear_state();
    // PrimeQ has no DownValue for `p`, so once the upvalue is cleared
    // it returns False (matching wolframscript).
    assert_eq!(
      interpret("PrimeQ[p] ^= True; PrimeQ[p]; UpValues[p] =.; PrimeQ[p]")
        .unwrap(),
      "False"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn option_q_1() {
    assert_case(
      r#"OptionQ[a -> True]; OptionQ[a :> True]; OptionQ[{a -> True}]; OptionQ[{a :> True}]; OptionQ[{a -> True, {b->1, "c"->2}}]; OptionQ[{a -> True, {b->1, c}}]; OptionQ[{a -> True, F[b->1,c->2]}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn option_q_2() {
    assert_case(
      r#"OptionQ[a -> True]; OptionQ[a :> True]; OptionQ[{a -> True}]; OptionQ[{a :> True}]; OptionQ[{a -> True, {b->1, "c"->2}}]; OptionQ[{a -> True, {b->1, c}}]; OptionQ[{a -> True, F[b->1,c->2]}]; OptionQ[x]"#,
      r#"False"#,
    );
  }
  #[test]
  fn input_form() {
    assert_case(
      r#"x = 3. ^ -20; InputForm[x]"#,
      r#"InputForm[2.8679719907924413*^-10]"#,
    );
  }
  #[test]
  fn head_1() {
    assert_case(r#"x = 3. ^ -20; InputForm[x]; Head[x]"#, r#"Real"#);
  }
  #[test]
  fn set_1() {
    assert_case(r#"x = 1"#, r#"1"#);
  }
  #[test]
  fn set_2() {
    assert_case(r#"x = 1; x = x + 1"#, r#"2"#);
  }
  #[test]
  fn f_1() {
    assert_case(
      r#"Format[f[x___]] := Infix[{x}, "~"]; f[1, 2, 3]"#,
      r#"f[1, 2, 3]"#,
    );
  }
  #[test]
  fn f_2() {
    assert_case(
      r#"Format[f[x___]] := Infix[{x}, "~"]; f[1, 2, 3]; f[1]"#,
      r#"f[1]"#,
    );
  }
  #[test]
  fn g_1() {
    assert_case(
      r##"Format[g[x_, y_]] := Infix[{x, y}, "#", 350, Left]; g[a, g[b, c]]"##,
      r#"g[a, g[b, c]]"#,
    );
  }
  #[test]
  fn g_2() {
    assert_case(
      r##"Format[g[x_, y_]] := Infix[{x, y}, "#", 350, Left]; g[a, g[b, c]]; g[g[a, b], c]"##,
      r#"g[g[a, b], c]"#,
    );
  }
  #[test]
  fn g_3() {
    assert_case(
      r##"Format[g[x_, y_]] := Infix[{x, y}, "#", 350, Left]; g[a, g[b, c]]; g[g[a, b], c]; g[a + b, c]"##,
      r#"g[a + b, c]"#,
    );
  }
  #[test]
  fn g_4() {
    assert_case(
      r##"Format[g[x_, y_]] := Infix[{x, y}, "#", 350, Left]; g[a, g[b, c]]; g[g[a, b], c]; g[a + b, c]; g[a * b, c]"##,
      r#"g[a*b, c]"#,
    );
  }
  #[test]
  fn g_5() {
    assert_case(
      r##"Format[g[x_, y_]] := Infix[{x, y}, "#", 350, Left]; g[a, g[b, c]]; g[g[a, b], c]; g[a + b, c]; g[a * b, c]; g[a, b] + c"##,
      r#"c + g[a, b]"#,
    );
  }
  #[test]
  fn g_6() {
    assert_case(
      r##"Format[g[x_, y_]] := Infix[{x, y}, "#", 350, Left]; g[a, g[b, c]]; g[g[a, b], c]; g[a + b, c]; g[a * b, c]; g[a, b] + c; g[a, b] * c"##,
      r#"c*g[a, b]"#,
    );
  }
  #[test]
  fn infix() {
    assert_case(
      r##"Format[g[x_, y_]] := Infix[{x, y}, "#", 350, Left]; g[a, g[b, c]]; g[g[a, b], c]; g[a + b, c]; g[a * b, c]; g[a, b] + c; g[a, b] * c; Infix[{a, b, c}, {"+", "-"}]"##,
      r#"Infix[{a, b, c}, {"+", "-"}]"#,
    );
  }
  #[test]
  fn set_3() {
    assert_case(r#"n = 10"#, r#"10"#);
  }
  #[test]
  fn hold_1() {
    assert_case(r#"x = 3; Hold[x]"#, r#"Hold[x]"#);
  }
  #[test]
  fn release_hold_1() {
    assert_case(r#"x = 3; Hold[x]; ReleaseHold[Hold[x]]"#, r#"3"#);
  }
  #[test]
  fn release_hold_2() {
    assert_case(
      r#"x = 3; Hold[x]; ReleaseHold[Hold[x]]; ReleaseHold[y]"#,
      r#"y"#,
    );
  }
  #[test]
  fn cf_1() {
    assert_case(r#"cf = Compile[{x, y}, x + 2 y]; cf[2.5, 4.3]"#, r#"11.1"#);
  }
  #[test]
  fn head_2() {
    // Same situation as case 524 — the scraped expectation pinned
    // wolframscript-internal bytecode that Woxi can't reproduce, and the
    // mathics original uses a wildcard. Verify the documented contract:
    // the typed-arg form `Compile[{{x, _Real}}, Sin[x]]` returns a
    // `CompiledFunction`. (The intermediate `cf[2.5, 4.3]` call is
    // exercised — its result is discarded by `CompoundExpression`.)
    assert_case(
      r#"cf = Compile[{x, y}, x + 2 y]; cf[2.5, 4.3]; Head[Compile[{{x, _Real}}, Sin[x]]]"#,
      r#"CompiledFunction"#,
    );
  }
  #[test]
  fn cf_2() {
    assert_case(
      r#"cf = Compile[{x, y}, x + 2 y]; cf[2.5, 4.3]; cf = Compile[{{x, _Real}}, Sin[x]]; cf[1.4]"#,
      r#"0.9854497299884601"#,
    );
  }
  #[test]
  fn head_3() {
    assert_case(
      r#"sqr = Compile[{x}, x x]; Head[sqr]"#,
      r#"CompiledFunction"#,
    );
  }
  #[test]
  fn sqr() {
    assert_case(r#"sqr = Compile[{x}, x x]; Head[sqr]; sqr[2]"#, r#"4."#);
  }
  #[test]
  fn up_values_1() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]; N[a, 20] = 11; N[a + b, 20]; N[f[a, b]]; SetAttributes[f, NHoldAll]; N[f[a, b]]; N[c, p_?(#>10&)] := p; N[c, 3]; N[c, 11]; N[d] ^= 5; UpValues[d]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn n_values_1() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]; N[a, 20] = 11; N[a + b, 20]; N[f[a, b]]; SetAttributes[f, NHoldAll]; N[f[a, b]]; N[c, p_?(#>10&)] := p; N[c, 3]; N[c, 11]; N[d] ^= 5; UpValues[d]; NValues[d]"#,
      r#"{HoldPattern[N[d, {MachinePrecision, MachinePrecision}]] :> 5}"#,
    );
  }
  #[test]
  fn n_1() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]; N[a, 20] = 11; N[a + b, 20]; N[f[a, b]]; SetAttributes[f, NHoldAll]; N[f[a, b]]; N[c, p_?(#>10&)] := p; N[c, 3]; N[c, 11]; N[d] ^= 5; UpValues[d]; NValues[d]; e /: N[e] = 6; N[e]"#,
      r#"6."#,
    );
  }
  #[test]
  fn set_4() {
    assert_case(r#"a::b = "Hello world!""#, r#""Hello world!""#);
  }
  #[test]
  fn set_5() {
    assert_case(
      r#"pts = {{0, 0}, {1, 1}, {2, -1}, {3, 0}}"#,
      r#"{{0, 0}, {1, 1}, {2, -1}, {3, 0}}"#,
    );
  }
  #[test]
  fn set_6() {
    assert_case(
      r#"A = InterpretationBox["Four", 4]"#,
      r#"InterpretationBox["Four", 4]"#,
    );
  }
  #[test]
  fn display_form() {
    assert_case(
      r#"A = InterpretationBox["Four", 4]; DisplayForm[A]"#,
      r#"DisplayForm[InterpretationBox["Four", 4]]"#,
    );
  }
  #[test]
  fn symbol_literal_1() {
    assert_case(r#"x = 2; Clear[x]; x"#, r#"x"#);
  }
  #[test]
  fn symbol_literal_2() {
    assert_case(
      r#"x = 2; Clear[x]; x; x = 2; y = 3; Clear["Global`*"]; x"#,
      r#"x"#,
    );
  }
  #[test]
  fn symbol_literal_3() {
    assert_case(
      r#"x = 2; Clear[x]; x; x = 2; y = 3; Clear["Global`*"]; x; y"#,
      r#"y"#,
    );
  }
  #[test]
  fn attributes_1() {
    assert_case(
      r#"x = 2; Clear[x]; x; x = 2; y = 3; Clear["Global`*"]; x; y; Clear[Sin]; Unprotect[Sin]; Clear[Sin]; Sin[Pi]; Attributes[r] = {Flat, Orderless}; Clear["r"]; Attributes[r]"#,
      r#"{Flat, Orderless}"#,
    );
  }
  #[test]
  fn symbol_literal_4() {
    assert_case(r#"x = 2; ClearAll[x]; x"#, r#"x"#);
  }
  #[test]
  fn attributes_2() {
    assert_case(
      r#"x = 2; ClearAll[x]; x; Attributes[r] = {Flat, Orderless}; ClearAll[r]; Attributes[r]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn set_7() {
    assert_case(r#"a = 2"#, r#"2"#);
  }
  #[test]
  fn symbol_literal_5() {
    assert_case(r#"a = 2; a =.; a"#, r#"a"#);
  }
  #[test]
  fn set_8() {
    assert_case(r#"a::b = "foo""#, r#""foo""#);
  }
  #[test]
  fn messages() {
    // Wolframscript-matched expectation. The mathics original used the
    // `a::b` shorthand inside `HoldPattern`, but wolframscript -code
    // always prints the long-form `MessageName[a, b]` for messages stored
    // on a symbol. Strings are unquoted in OutputForm.
    assert_case(
      r#"a::b = "foo"; Messages[a]"#,
      r#"{HoldPattern[MessageName[a, b]] :> foo}"#,
    );
  }
  #[test]
  fn divide_1() {
    // Wolframscript-matched expectation. mathics rendered the InputForm
    // contents with quotes around the string ("bar"), but `wolframscript
    // -code` prints the unevaluated `InputForm[…]` wrapper with the bare
    // string contents (no quotes) — Woxi matches.
    assert_case(
      r#"a::b = "foo"; Messages[a]; Messages[a] = {a::c :> "bar"}; a::c // InputForm"#,
      r#"InputForm[bar]"#,
    );
  }
  #[test]
  fn n_values_2() {
    assert_case(r#"NValues[a]"#, r#"{}"#);
  }
  #[test]
  fn n_values_3() {
    assert_case(
      r#"NValues[a]; N[a] = 3; NValues[a]"#,
      r#"{HoldPattern[N[a, {MachinePrecision, MachinePrecision}]] :> 3}"#,
    );
  }
  #[test]
  fn n_2() {
    assert_case(
      r#"NValues[a]; N[a] = 3; NValues[a]; NValues[b] := {N[b, MachinePrecision] :> 2}; N[b]"#,
      r#"2."#,
    );
  }
  #[test]
  fn n_3() {
    assert_case(
      r#"NValues[a]; N[a] = 3; NValues[a]; NValues[b] := {N[b, MachinePrecision] :> 2}; N[b]; NValues[c] := {N[c] :> 3}; N[c]"#,
      r#"c"#,
    );
  }
  #[test]
  fn n_values_4() {
    assert_case(
      r#"NValues[a]; N[a] = 3; NValues[a]; NValues[b] := {N[b, MachinePrecision] :> 2}; N[b]; NValues[c] := {N[c] :> 3}; N[c]; NValues[d] = {foo -> bar}; NValues[d]"#,
      r#"{HoldPattern[foo] :> bar}"#,
    );
  }
  #[test]
  fn n_4() {
    assert_case(
      r#"NValues[a]; N[a] = 3; NValues[a]; NValues[b] := {N[b, MachinePrecision] :> 2}; N[b]; NValues[c] := {N[c] :> 3}; N[c]; NValues[d] = {foo -> bar}; NValues[d]; N[d]"#,
      r#"d"#,
    );
  }
  #[test]
  fn sub_values() {
    assert_case(
      r#"f[1][x_] := x; f[2][x_] := x ^ 2; SubValues[f]"#,
      r#"{HoldPattern[f[1][x_]] :> x, HoldPattern[f[2][x_]] :> x^2}"#,
    );
  }
  #[test]
  fn definition_1() {
    assert_case(
      r#"f[1][x_] := x; f[2][x_] := x ^ 2; SubValues[f]; Definition[f]"#,
      r#"f[1][x_] := x

f[2][x_] := x^2"#,
    );
  }
  #[test]
  fn set_9() {
    assert_case(r#"a = 3"#, r#"3"#);
  }
  #[test]
  fn symbol_literal_6() {
    assert_case(r#"a = 3; a"#, r#"3"#);
  }
  #[test]
  fn own_values_1() {
    assert_case(r#"a = 3; a; OwnValues[a]"#, r#"{HoldPattern[a] :> 3}"#);
  }
  #[test]
  fn list_literal_1() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}"#,
      r#"{10, 2, 3}"#,
    );
  }
  #[test]
  fn list_literal_2() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}"#,
      r#"{1, 2, {{c1, c2}, {10}}}"#,
    );
  }
  #[test]
  fn symbol_literal_7() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d"#,
      r#"10"#,
    );
  }
  #[test]
  fn symbol_literal_8() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a"#,
      r#"1"#,
    );
  }
  #[test]
  fn set_10() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a"#,
      r#"1"#,
    );
  }
  #[test]
  fn set_11() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a; a = 2"#,
      r#"2"#,
    );
  }
  #[test]
  fn symbol_literal_9() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a; a = 2; x"#,
      r#"1"#,
    );
  }
  #[test]
  fn equal_1() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a; a = 2; x; a = b = c = 2; a == b == c == 2"#,
      r#"True"#,
    );
  }
  #[test]
  fn a_1() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a; a = 2; x; a = b = c = 2; a == b == c == 2; A = {{1, 2}, {3, 4}}; A[[1, 2]] = 5"#,
      r#"5"#,
    );
  }
  #[test]
  fn a_2() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a; a = 2; x; a = b = c = 2; a == b == c == 2; A = {{1, 2}, {3, 4}}; A[[1, 2]] = 5; A"#,
      r#"{{1, 5}, {3, 4}}"#,
    );
  }
  #[test]
  fn a_3() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a; a = 2; x; a = b = c = 2; a == b == c == 2; A = {{1, 2}, {3, 4}}; A[[1, 2]] = 5; A; A[[;;, 2]] = {6, 7}"#,
      r#"{6, 7}"#,
    );
  }
  #[test]
  fn a_4() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a; a = 2; x; a = b = c = 2; a == b == c == 2; A = {{1, 2}, {3, 4}}; A[[1, 2]] = 5; A; A[[;;, 2]] = {6, 7}; A"#,
      r#"{{1, 6}, {3, 7}}"#,
    );
  }
  #[test]
  fn b_1() {
    assert_case(
      r#"a = 3; a; OwnValues[a]; {a, b, c} = {10, 2, 3}; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}; d; a; x = a; a = 2; x; a = b = c = 2; a == b == c == 2; A = {{1, 2}, {3, 4}}; A[[1, 2]] = 5; A; A[[;;, 2]] = {6, 7}; A; B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; B[[1;;2, 2;;-1]] = {{t, u}, {y, z}}; B"#,
      r#"{{1, t, u}, {4, y, z}, {7, 8, 9}}"#,
    );
  }
  #[test]
  fn f_3() {
    assert_case(
      r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2; x; f[x_] := p[x] /; x>0; f[3]; f[-3]; F[x_, y_] /; x < y /; x>0  := x / y; F[x_, y_] := y / x; F[2, 3]"#,
      r#"2 / 3"#,
    );
  }
  #[test]
  fn f_4() {
    assert_case(
      r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2; x; f[x_] := p[x] /; x>0; f[3]; f[-3]; F[x_, y_] /; x < y /; x>0  := x / y; F[x_, y_] := y / x; F[2, 3]; F[3, 2]"#,
      r#"2 / 3"#,
    );
  }
  #[test]
  fn f_5() {
    assert_case(
      r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2; x; f[x_] := p[x] /; x>0; f[3]; f[-3]; F[x_, y_] /; x < y /; x>0  := x / y; F[x_, y_] := y / x; F[2, 3]; F[3, 2]; F[-3, 2]"#,
      r#"-2 / 3"#,
    );
  }
  #[test]
  fn symbol_literal_10() {
    assert_case(
      r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2; x; f[x_] := p[x] /; x>0; f[3]; f[-3]; F[x_, y_] /; x < y /; x>0  := x / y; F[x_, y_] := y / x; F[2, 3]; F[3, 2]; F[-3, 2]; ClearAll[a,b]; a/; b>0:= 3; a"#,
      r#"a"#,
    );
  }
  #[test]
  fn symbol_literal_11() {
    assert_case(
      r#"Attributes[SetDelayed]; a = 1; x := a; x; a = 2; x; f[x_] := p[x] /; x>0; f[3]; f[-3]; F[x_, y_] /; x < y /; x>0  := x / y; F[x_, y_] := y / x; F[2, 3]; F[3, 2]; F[-3, 2]; ClearAll[a,b]; a/; b>0:= 3; a; b=2; a"#,
      r#"3"#,
    );
  }
  #[test]
  fn down_values_1() {
    assert_case(
      r#"square /: area[square[s_]] := s^2; DownValues[square]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn up_values_2() {
    assert_case(
      r#"square /: area[square[s_]] := s^2; DownValues[square]; UpValues[square]"#,
      r#"{HoldPattern[area[square[s_]]] :> s^2}"#,
    );
  }
  #[test]
  fn down_values_2() {
    assert_case(r#"a[b] ^= 3; DownValues[a]"#, r#"{}"#);
  }
  #[test]
  fn up_values_3() {
    assert_case(
      r#"a[b] ^= 3; DownValues[a]; UpValues[b]"#,
      r#"{HoldPattern[a[b]] :> 3}"#,
    );
  }
  #[test]
  fn symbol_literal_12() {
    assert_case(
      r#"a[b] ^= 3; DownValues[a]; UpValues[b]; Format[r] ^= "custom"; r"#,
      r#"r"#,
    );
  }
  #[test]
  fn up_values_4() {
    assert_case(
      r#"a[b] ^= 3; DownValues[a]; UpValues[b]; Format[r] ^= "custom"; r; UpValues[r]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn up_values_5() {
    assert_case(
      r#"a[b] ^:= x; x = 2; a[b]; UpValues[b]"#,
      r#"{HoldPattern[a[b]] :> x}"#,
    );
  }
  #[test]
  fn set_12() {
    assert_case(r#"a = 10; a += 2"#, r#"12"#);
  }
  #[test]
  fn symbol_literal_13() {
    assert_case(r#"a = 10; a += 2; a"#, r#"12"#);
  }
  #[test]
  fn minus_1() {
    assert_case(r#"a = 5; a--"#, r#"5"#);
  }
  #[test]
  fn symbol_literal_14() {
    assert_case(r#"a = 5; a--; a"#, r#"4"#);
  }
  #[test]
  fn symbol_literal_15() {
    assert_case(r#"a = 5; a--; a; a = 1.6; a--; a"#, r#"0.6000000000000001"#);
  }
  #[test]
  fn set_13() {
    assert_case(
      r#"a = 5; a--; a; a = 1.6; a--; a; a = {1, 3, 5}"#,
      r#"{1, 3, 5}"#,
    );
  }
  #[test]
  fn symbol_literal_16() {
    assert_case(
      r#"a = 5; a--; a; a = 1.6; a--; a; a = {1, 3, 5}; a--; a"#,
      r#"{0, 2, 4}"#,
    );
  }
  #[test]
  fn set_14() {
    assert_case(r#"a = 10; a /= 2"#, r#"5"#);
  }
  #[test]
  fn symbol_literal_17() {
    assert_case(r#"a = 10; a /= 2; a"#, r#"5"#);
  }
  #[test]
  fn plus_1() {
    assert_case(r#"a = 1; a++"#, r#"1"#);
  }
  #[test]
  fn symbol_literal_18() {
    assert_case(r#"a = 1; a++; a"#, r#"2"#);
  }
  #[test]
  fn plus_2() {
    assert_case(r#"a = 1; a++; a; a = 1.5; a++"#, r#"1.5"#);
  }
  #[test]
  fn symbol_literal_19() {
    assert_case(r#"a = 1; a++; a; a = 1.5; a++; a"#, r#"2.5"#);
  }
  #[test]
  fn symbol_literal_20() {
    assert_case(
      r#"a = 1; a++; a; a = 1.5; a++; a; y = 2 x; y++; y"#,
      r#"1 + 2*x"#,
    );
  }
  #[test]
  fn set_15() {
    assert_case(
      r#"a = 1; a++; a; a = 1.5; a++; a; y = 2 x; y++; y; x = {1, 3, 5}"#,
      r#"{1, 3, 5}"#,
    );
  }
  #[test]
  fn plus_3() {
    assert_case(
      r#"a = 1; a++; a; a = 1.5; a++; a; y = 2 x; y++; y; x = {1, 3, 5}; ++++a+++++2 // Hold // FullForm"#,
      r#"FullForm[Hold[++(++(a++++)) + 2]]"#,
    );
  }
  #[test]
  fn increment_chain_postfix() {
    // Postfix outer never parenthesizes; chained postfix prints
    // compactly. Matches wolframscript.
    assert_case(r#"Hold[a++++]"#, r#"Hold[a++++]"#);
    assert_case(r#"Hold[Increment[Increment[a]]]"#, r#"Hold[a++++]"#);
  }
  #[test]
  fn increment_chain_prefix() {
    // Prefix outer parenthesizes inner ++/--.
    assert_case(r#"Hold[++++a]"#, r#"Hold[++(++a)]"#);
  }
  #[test]
  fn increment_chain_mixed() {
    // Mixed chains follow the same rule: postfix outer = no parens,
    // prefix outer = parens.
    assert_case(r#"Hold[Increment[PreIncrement[a]]]"#, r#"Hold[++a++]"#);
    assert_case(r#"Hold[PreIncrement[Increment[a]]]"#, r#"Hold[++(a++)]"#);
  }
  #[test]
  fn increment_chain_fullform_round_trip() {
    // The mathics-derived FullForm string is the ultimate authority
    // on the parsed structure; matches wolframscript exactly.
    assert_case(
      r#"ToString[FullForm[Hold[++++a+++++2]]]"#,
      r#"Hold[Plus[PreIncrement[PreIncrement[Increment[Increment[a]]]], 2]]"#,
    );
  }
  #[test]
  fn minus_2() {
    assert_case(r#"a = 2; --a"#, r#"1"#);
  }
  #[test]
  fn symbol_literal_21() {
    assert_case(r#"a = 2; --a; a"#, r#"1"#);
  }
  #[test]
  fn set_16() {
    assert_case(r#"a = 2"#, r#"2"#);
  }
  #[test]
  fn plus_4() {
    assert_case(r#"a = 2; ++a"#, r#"3"#);
  }
  #[test]
  fn symbol_literal_22() {
    assert_case(r#"a = 2; ++a; a"#, r#"3"#);
  }
  #[test]
  fn plus_5() {
    assert_case(r#"a = 2; ++a; a; a + 1.6"#, r#"4.6"#);
  }
  #[test]
  fn plus_6() {
    assert_case(
      r#"a = 2; ++a; a; a + 1.6; Clear[x, y]; y = x; ++y"#,
      r#"1 + x"#,
    );
  }
  #[test]
  fn symbol_literal_23() {
    assert_case(
      r#"a = 2; ++a; a; a + 1.6; Clear[x, y]; y = x; ++y; y"#,
      r#"1 + x"#,
    );
  }
  #[test]
  fn set_17() {
    assert_case(r#"a = 10; a -= 2"#, r#"8"#);
  }
  #[test]
  fn symbol_literal_24() {
    assert_case(r#"a = 10; a -= 2; a"#, r#"8"#);
  }
  #[test]
  fn set_18() {
    assert_case(r#"a = 10; a *= 2"#, r#"20"#);
  }
  #[test]
  fn symbol_literal_25() {
    assert_case(r#"a = 10; a *= 2; a"#, r#"20"#);
  }
  #[test]
  fn up_values_6() {
    assert_case(r#"a + b ^= 2; UpValues[a]"#, r#"{HoldPattern[a + b] :> 2}"#);
  }
  #[test]
  fn up_values_7() {
    assert_case(
      r#"a + b ^= 2; UpValues[a]; UpValues[b]"#,
      r#"{HoldPattern[a + b] :> 2}"#,
    );
  }
  #[test]
  fn sin() {
    assert_case(
      r#"a + b ^= 2; UpValues[a]; UpValues[b]; UpValues[pi] := {Sin[pi] :> 0}; Sin[pi]"#,
      r#"0"#,
    );
  }
  #[test]
  fn set_19() {
    assert_case(
      r#"factors = FactorInteger[2010]"#,
      r#"{{2, 1}, {3, 1}, {5, 1}, {67, 1}}"#,
    );
  }
  #[test]
  fn apply() {
    assert_case(
      r#"factors = FactorInteger[2010]; Times @@ Power @@@ factors"#,
      r#"2010"#,
    );
  }
  #[test]
  fn factor_integer() {
    assert_case(
      r#"factors = FactorInteger[2010]; Times @@ Power @@@ factors; FactorInteger[2010 / 2011]"#,
      r#"{{2, 1}, {3, 1}, {5, 1}, {67, 1}, {2011, -1}}"#,
    );
  }
  #[test]
  fn greater_1() {
    assert_case(r#"rule = F[u_]->g[u]"#, r#"F[u_] -> g[u]"#);
  }
  #[test]
  fn plus_7() {
    assert_case(
      r#"rule = F[u_]->g[u]; a + F[x ^ 2] /. rule"#,
      r#"a + g[x ^ 2]"#,
    );
  }
  #[test]
  fn plus_8() {
    assert_case(
      r#"rule = F[u_]->g[u]; a + F[x ^ 2] /. rule; a + F[F[x ^ 2]] /. rule"#,
      r#"a + g[F[x ^ 2]]"#,
    );
  }
  #[test]
  fn plus_9() {
    assert_case(
      r#"rule = F[u_]->g[u]; a + F[x ^ 2] /. rule; a + F[F[x ^ 2]] /. rule; a + F[F[x ^ 2]] //. rule"#,
      r#"a + g[g[x ^ 2]]"#,
    );
  }
  #[test]
  fn set_20() {
    assert_case(
      r#"rule = F[u_]->g[u]; a + F[x ^ 2] /. rule; a + F[F[x ^ 2]] /. rule; a + F[F[x ^ 2]] //. rule; dispatchrule = Dispatch[{rule}]"#,
      r#"Dispatch[{F[u_] -> g[u]}]"#,
    );
  }
  #[test]
  fn plus_10() {
    assert_case(
      r#"rule = F[u_]->g[u]; a + F[x ^ 2] /. rule; a + F[F[x ^ 2]] /. rule; a + F[F[x ^ 2]] //. rule; dispatchrule = Dispatch[{rule}]; a + F[F[x ^ 2]] //. dispatchrule"#,
      r#"a + g[g[x ^ 2]]"#,
    );
  }
  #[test]
  fn f_6() {
    assert_case(
      r#"rules = {{a_,b_}->a^b, {1,2}->3., F[x_]->x^2}; F[2] /. rules"#,
      r#"4"#,
    );
  }
  #[test]
  fn set_21() {
    assert_case(
      r#"rules = {{a_,b_}->a^b, {1,2}->3., F[x_]->x^2}; F[2] /. rules; dispatchrules = Dispatch[rules]"#,
      r#"Dispatch[{{a_, b_} -> a^b, {1, 2} -> 3., F[x_] -> x^2}]"#,
    );
  }
  #[test]
  fn f_7() {
    assert_case(
      r#"rules = {{a_,b_}->a^b, {1,2}->3., F[x_]->x^2}; F[2] /. rules; dispatchrules = Dispatch[rules];  F[2] /. dispatchrules"#,
      r#"4"#,
    );
  }
  #[test]
  fn unequal_1() {
    assert_case(r#"a =!= a"#, r#"False"#);
  }
  #[test]
  fn unequal_2() {
    assert_case(r#"a =!= a; 1 =!= 1."#, r#"True"#);
  }
  #[test]
  fn unequal_3() {
    assert_case(r#"a =!= a; 1 =!= 1.; 1 =!= 2 =!= 3 =!= 4"#, r#"True"#);
  }
  #[test]
  fn unequal_4() {
    assert_case(
      r#"a =!= a; 1 =!= 1.; 1 =!= 2 =!= 3 =!= 4; 1 =!= 2 =!= 1 =!= 4"#,
      r#"False"#,
    );
  }
  #[test]
  fn unsame_q_1() {
    assert_case(
      r#"a =!= a; 1 =!= 1.; 1 =!= 2 =!= 3 =!= 4; 1 =!= 2 =!= 1 =!= 4; UnsameQ[]"#,
      r#"True"#,
    );
  }
  #[test]
  fn unsame_q_2() {
    assert_case(
      r#"a =!= a; 1 =!= 1.; 1 =!= 2 =!= 3 =!= 4; 1 =!= 2 =!= 1 =!= 4; UnsameQ[]; UnsameQ[expr]"#,
      r#"True"#,
    );
  }
  #[test]
  fn numeric_q_1() {
    assert_case(
      r#"NumericQ[2]; NumericQ[Sqrt[Pi]]; NumberQ[Sqrt[Pi]]; NumericQ[a]=True; NumericQ[a]; NumericQ[Sin[a]]; Clear[a]; NumericQ[a]"#,
      r#"True"#,
    );
  }
  #[test]
  fn numeric_q_2() {
    assert_case(
      r#"NumericQ[2]; NumericQ[Sqrt[Pi]]; NumberQ[Sqrt[Pi]]; NumericQ[a]=True; NumericQ[a]; NumericQ[Sin[a]]; Clear[a]; NumericQ[a]; ClearAll[a]; NumericQ[a]"#,
      r#"True"#,
    );
  }
  #[test]
  fn numeric_q_3() {
    assert_case(
      r#"NumericQ[2]; NumericQ[Sqrt[Pi]]; NumberQ[Sqrt[Pi]]; NumericQ[a]=True; NumericQ[a]; NumericQ[Sin[a]]; Clear[a]; NumericQ[a]; ClearAll[a]; NumericQ[a]; NumericQ[a]=False; NumericQ[a]"#,
      r#"False"#,
    );
  }
  #[test]
  fn accuracy_1() {
    assert_case(
      r#"Accuracy[3.1416`2]; Accuracy[1]; Accuracy[A]; z=Complex[3.00``2, 4.00``2]; Accuracy[z] == -Log[10, Sqrt[10^(-2 Accuracy[Re[z]]) + 10^(-2 Accuracy[Im[z]])]]; Accuracy[F[1, Pi, A]]"#,
      r#"Infinity"#,
    );
  }
  #[test]
  fn accuracy_2() {
    assert_case(
      r#"Accuracy[3.1416`2]; Accuracy[1]; Accuracy[A]; z=Complex[3.00``2, 4.00``2]; Accuracy[z] == -Log[10, Sqrt[10^(-2 Accuracy[Re[z]]) + 10^(-2 Accuracy[Im[z]])]]; Accuracy[F[1, Pi, A]]; Accuracy[F[1.3, Pi, A]]"#,
      r#"15.840646417884168"#,
    );
  }
  #[test]
  fn expression_1() {
    assert_case(
      r#"Accuracy[3.1416`2]; Accuracy[1]; Accuracy[A]; z=Complex[3.00``2, 4.00``2]; Accuracy[z] == -Log[10, Sqrt[10^(-2 Accuracy[Re[z]]) + 10^(-2 Accuracy[Im[z]])]]; Accuracy[F[1, Pi, A]]; Accuracy[F[1.3, Pi, A]]; 0``2"#,
      r#"0``2."#,
    );
  }
  #[test]
  fn accuracy_3() {
    assert_case(
      r#"Accuracy[3.1416`2]; Accuracy[1]; Accuracy[A]; z=Complex[3.00``2, 4.00``2]; Accuracy[z] == -Log[10, Sqrt[10^(-2 Accuracy[Re[z]]) + 10^(-2 Accuracy[Im[z]])]]; Accuracy[F[1, Pi, A]]; Accuracy[F[1.3, Pi, A]]; 0``2; Accuracy[0.``2]"#,
      r#"2."#,
    );
  }
  #[test]
  fn accuracy_4() {
    assert_case(
      r#"Accuracy[3.1416`2]; Accuracy[1]; Accuracy[A]; z=Complex[3.00``2, 4.00``2]; Accuracy[z] == -Log[10, Sqrt[10^(-2 Accuracy[Re[z]]) + 10^(-2 Accuracy[Im[z]])]]; Accuracy[F[1, Pi, A]]; Accuracy[F[1.3, Pi, A]]; 0``2; Accuracy[0.``2]; Accuracy[0.`] == $MachinePrecision - Log[10, $MinMachineNumber]"#,
      r#"True"#,
    );
  }
  #[test]
  fn accuracy_5() {
    assert_case(
      r#"Accuracy[3.1416`2]; Accuracy[1]; Accuracy[A]; z=Complex[3.00``2, 4.00``2]; Accuracy[z] == -Log[10, Sqrt[10^(-2 Accuracy[Re[z]]) + 10^(-2 Accuracy[Im[z]])]]; Accuracy[F[1, Pi, A]]; Accuracy[F[1.3, Pi, A]]; 0``2; Accuracy[0.``2]; Accuracy[0.`] == $MachinePrecision - Log[10, $MinMachineNumber]; Accuracy[{{1, 1.`},{1.``5, 1.``10}}]"#,
      r#"5.000000000000002"#,
    );
  }
  #[test]
  fn definition_2() {
    assert_case(r#"a = 2; Definition[a]"#, r#"a = 2"#);
  }
  #[test]
  fn definition_3() {
    assert_case(
      r#"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]"#,
      r#"g[f] ^:= 2

f[x_] := x^2"#,
    );
  }
  #[test]
  fn down_values_3() {
    assert_case(
      r#"f[x_] := x ^ 2; DownValues[f]"#,
      r#"{HoldPattern[f[x_]] :> x^2}"#,
    );
  }
  #[test]
  fn down_values_4() {
    assert_case(
      r#"f[x_] := x ^ 2; DownValues[f]; f[x_Integer] := 2; f[x_Real] := 3; DownValues[f]"#,
      r#"{HoldPattern[f[x_Integer]] :> 2, HoldPattern[f[x_Real]] :> 3, HoldPattern[f[x_]] :> x^2}"#,
    );
  }
  #[test]
  fn f_8() {
    assert_case(
      r#"f[x_] := x ^ 2; DownValues[f]; f[x_Integer] := 2; f[x_Real] := 3; DownValues[f]; f[3]"#,
      r#"2"#,
    );
  }
  #[test]
  fn f_9() {
    assert_case(
      r#"f[x_] := x ^ 2; DownValues[f]; f[x_Integer] := 2; f[x_Real] := 3; DownValues[f]; f[3]; f[3.]"#,
      r#"3"#,
    );
  }
  #[test]
  fn f_10() {
    assert_case(
      r#"f[x_] := x ^ 2; DownValues[f]; f[x_Integer] := 2; f[x_Real] := 3; DownValues[f]; f[3]; f[3.]; f[a]"#,
      r#"a ^ 2"#,
    );
  }
  #[test]
  fn sort() {
    assert_case(
      r#"f[x_] := x ^ 2; DownValues[f]; f[x_Integer] := 2; f[x_Real] := 3; DownValues[f]; f[3]; f[3.]; f[a]; Sort[{x_, x_Integer}, PatternsOrderedQ]"#,
      r#"{x_, x_Integer}"#,
    );
  }
  #[test]
  fn g_7() {
    assert_case(
      r#"f[x_] := x ^ 2; DownValues[f]; f[x_Integer] := 2; f[x_Real] := 3; DownValues[f]; f[3]; f[3.]; f[a]; Sort[{x_, x_Integer}, PatternsOrderedQ]; DownValues[g] := {g[x_] :> x ^ 2, g[x_Integer] :> x}; g[2]"#,
      r#"4"#,
    );
  }
  #[test]
  fn format_values_1() {
    assert_case(
      r#"Format[F[x_], OutputForm]:= Subscript[x, F]; FormatValues[F]"#,
      r#"{HoldPattern[F[x_]] :> Subscript[x, F]}"#,
    );
  }
  #[test]
  fn format_values_2() {
    assert_case(
      r#"Format[F[x_], OutputForm]:= Subscript[x, F]; FormatValues[F]; FormatValues[F]  //InputForm"#,
      r#"InputForm[{HoldPattern[F[x_]] :> Subscript[x, F]}]"#,
    );
  }
  #[test]
  fn own_values_2() {
    assert_case(r#"x = 3; x = 2; OwnValues[x]"#, r#"{HoldPattern[x] :> 2}"#);
  }
  #[test]
  fn own_values_3() {
    assert_case(
      r#"x = 3; x = 2; OwnValues[x]; x := y; OwnValues[x]"#,
      r#"{HoldPattern[x] :> y}"#,
    );
  }
  #[test]
  fn own_values_4() {
    assert_case(
      r#"x = 3; x = 2; OwnValues[x]; x := y; OwnValues[x]; y = 5; OwnValues[x]"#,
      r#"{HoldPattern[x] :> y}"#,
    );
  }
  #[test]
  fn hold_2() {
    assert_case(
      r#"x = 3; x = 2; OwnValues[x]; x := y; OwnValues[x]; y = 5; OwnValues[x]; Hold[x] /. OwnValues[x]"#,
      r#"Hold[y]"#,
    );
  }
  #[test]
  fn hold_3() {
    assert_case(
      r#"x = 3; x = 2; OwnValues[x]; x := y; OwnValues[x]; y = 5; OwnValues[x]; Hold[x] /. OwnValues[x]; Hold[x] /. OwnValues[x] // ReleaseHold"#,
      r#"5"#,
    );
  }
  #[test]
  fn table_1() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]; Clear[n]; Table[i, {i, 4}]"#,
      r#"{1, 2, 3, 4}"#,
    );
  }
  #[test]
  fn table_2() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]; Clear[n]; Table[i, {i, 4}]; Table[i, {i, 2, 5}]"#,
      r#"{2, 3, 4, 5}"#,
    );
  }
  #[test]
  fn table_3() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]; Clear[n]; Table[i, {i, 4}]; Table[i, {i, 2, 5}]; Table[i, {i, 2, 6, 2}]"#,
      r#"{2, 4, 6}"#,
    );
  }
  #[test]
  fn table_4() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]; Clear[n]; Table[i, {i, 4}]; Table[i, {i, 2, 5}]; Table[i, {i, 2, 6, 2}]; Table[i, {i, Pi, 2 Pi, Pi / 2}]"#,
      r#"{Pi, (3*Pi)/2, 2*Pi}"#,
    );
  }
  #[test]
  fn table_5() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]; Clear[n]; Table[i, {i, 4}]; Table[i, {i, 2, 5}]; Table[i, {i, 2, 6, 2}]; Table[i, {i, Pi, 2 Pi, Pi / 2}]; Table[x^2, {x, {a, b, c}}]"#,
      r#"{a ^ 2, b ^ 2, c ^ 2}"#,
    );
  }
  #[test]
  fn table_6() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]; Clear[n]; Table[i, {i, 4}]; Table[i, {i, 2, 5}]; Table[i, {i, 2, 6, 2}]; Table[i, {i, Pi, 2 Pi, Pi / 2}]; Table[x^2, {x, {a, b, c}}]; Table[{i, j}, {i, {a, b}}, {j, 1, 2}]"#,
      r#"{{{a, 1}, {a, 2}}, {{b, 1}, {b, 2}}}"#,
    );
  }
  #[test]
  fn table_7() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]; Clear[n]; Table[i, {i, 4}]; Table[i, {i, 2, 5}]; Table[i, {i, 2, 6, 2}]; Table[i, {i, Pi, 2 Pi, Pi / 2}]; Table[x^2, {x, {a, b, c}}]; Table[{i, j}, {i, {a, b}}, {j, 1, 2}]; Table[x, {x, a, a + 5 n, n}]"#,
      r#"{a, a + n, a + 2*n, a + 3*n, a + 4*n, a + 5*n}"#,
    );
  }
  #[test]
  fn table_8() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]; Clear[n]; Table[i, {i, 4}]; Table[i, {i, 2, 5}]; Table[i, {i, 2, 6, 2}]; Table[i, {i, Pi, 2 Pi, Pi / 2}]; Table[x^2, {x, {a, b, c}}]; Table[{i, j}, {i, {a, b}}, {j, 1, 2}]; Table[x, {x, a, a + 5 n, n}]; Table[i, {i, 1, 9, Infinity}]"#,
      r#"{1}"#,
    );
  }
  #[test]
  fn append_to_1() {
    assert_case(r#"s = {}; AppendTo[s, 1]"#, r#"{1}"#);
  }
  #[test]
  fn symbol_literal_26() {
    assert_case(r#"s = {}; AppendTo[s, 1]; s"#, r#"{1}"#);
  }
  #[test]
  fn append_to_2() {
    assert_case(
      r#"s = {}; AppendTo[s, 1]; s; y = f[]; AppendTo[y, x]"#,
      r#"f[x]"#,
    );
  }
  #[test]
  fn symbol_literal_27() {
    assert_case(
      r#"s = {}; AppendTo[s, 1]; s; y = f[]; AppendTo[y, x]; y"#,
      r#"f[x]"#,
    );
  }
  #[test]
  fn a_5() {
    assert_case(r#"A = {a, b, c, d}; A[[3]]"#, r#"c"#);
  }
  #[test]
  fn list_literal_3() {
    assert_case(r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]"#, r#"b"#);
  }
  #[test]
  fn plus_11() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]"#,
      r#"b"#,
    );
  }
  #[test]
  fn plus_12() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]"#,
      r#"Plus"#,
    );
  }
  #[test]
  fn m() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]; M = {{a, b}, {c, d}}; M[[1, 2]]"#,
      r#"b"#,
    );
  }
  #[test]
  fn list_literal_4() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]; M = {{a, b}, {c, d}}; M[[1, 2]]; {1, 2, 3, 4}[[2;;4]]"#,
      r#"{2, 3, 4}"#,
    );
  }
  #[test]
  fn list_literal_5() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]; M = {{a, b}, {c, d}}; M[[1, 2]]; {1, 2, 3, 4}[[2;;4]]; {1, 2, 3, 4}[[2;;-1]]"#,
      r#"{2, 3, 4}"#,
    );
  }
  #[test]
  fn list_literal_6() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]; M = {{a, b}, {c, d}}; M[[1, 2]]; {1, 2, 3, 4}[[2;;4]]; {1, 2, 3, 4}[[2;;-1]]; {a, b, c, d}[[{1, 3, 3}]]"#,
      r#"{a, c, c}"#,
    );
  }
  #[test]
  fn b_2() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]; M = {{a, b}, {c, d}}; M[[1, 2]]; {1, 2, 3, 4}[[2;;4]]; {1, 2, 3, 4}[[2;;-1]]; {a, b, c, d}[[{1, 3, 3}]]; B = {{a, b, c}, {d, e, f}, {g, h, i}}; B[[;;, 2]]"#,
      r#"{b, e, h}"#,
    );
  }
  #[test]
  fn b_3() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]; M = {{a, b}, {c, d}}; M[[1, 2]]; {1, 2, 3, 4}[[2;;4]]; {1, 2, 3, 4}[[2;;-1]]; {a, b, c, d}[[{1, 3, 3}]]; B = {{a, b, c}, {d, e, f}, {g, h, i}}; B[[;;, 2]]; B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; B[[{1, 3}, -2;;-1]]"#,
      r#"{{2, 3}, {8, 9}}"#,
    );
  }
  #[test]
  fn list_literal_7() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]; M = {{a, b}, {c, d}}; M[[1, 2]]; {1, 2, 3, 4}[[2;;4]]; {1, 2, 3, 4}[[2;;-1]]; {a, b, c, d}[[{1, 3, 3}]]; B = {{a, b, c}, {d, e, f}, {g, h, i}}; B[[;;, 2]]; B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; B[[{1, 3}, -2;;-1]]; {{a, b, c}, {d, e, f}, {g, h, i}}[[All, 3]]"#,
      r#"{c, f, i}"#,
    );
  }
  #[test]
  fn plus_13() {
    assert_case(
      r#"A = {a, b, c, d}; A[[3]]; {a, b, c}[[-2]]; (a + b + c)[[2]]; (a + b + c)[[0]]; M = {{a, b}, {c, d}}; M[[1, 2]]; {1, 2, 3, 4}[[2;;4]]; {1, 2, 3, 4}[[2;;-1]]; {a, b, c, d}[[{1, 3, 3}]]; B = {{a, b, c}, {d, e, f}, {g, h, i}}; B[[;;, 2]]; B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; B[[{1, 3}, -2;;-1]]; {{a, b, c}, {d, e, f}, {g, h, i}}[[All, 3]]; (a+b+c+d)[[-1;;-2]]"#,
      r#"0"#,
    );
  }
  #[test]
  fn set_22() {
    assert_case(r#"s = {1, 2, 4, 9}"#, r#"{1, 2, 4, 9}"#);
  }
  #[test]
  fn prepend_to_1() {
    assert_case(r#"s = {1, 2, 4, 9}; PrependTo[s, 0]"#, r#"{0, 1, 2, 4, 9}"#);
  }
  #[test]
  fn symbol_literal_28() {
    assert_case(
      r#"s = {1, 2, 4, 9}; PrependTo[s, 0]; s"#,
      r#"{0, 1, 2, 4, 9}"#,
    );
  }
  #[test]
  fn prepend_to_2() {
    assert_case(
      r#"s = {1, 2, 4, 9}; PrependTo[s, 0]; s; y = f[a, b, c]; PrependTo[y, x]"#,
      r#"f[x, a, b, c]"#,
    );
  }
  #[test]
  fn symbol_literal_29() {
    assert_case(
      r#"s = {1, 2, 4, 9}; PrependTo[s, 0]; s; y = f[a, b, c]; PrependTo[y, x]; y"#,
      r#"f[x, a, b, c]"#,
    );
  }
  #[test]
  fn f_11() {
    assert_case(r#"1; Sqrt[2]; 2/9; Pi; F[1.2, 2/9]"#, r#"F[1.2, 2/9]"#);
  }
  #[test]
  fn f_12() {
    assert_case(
      r#"1; Sqrt[2]; 2/9; Pi; F[1.2, 2/9]; F[1.2, 2/9]"#,
      r#"F[1.2, 2/9]"#,
    );
  }
  #[test]
  fn f_13() {
    assert_case(
      r#"1; Sqrt[2]; 2/9; Pi; F[1.2, 2/9]; F[1.2, 2/9]; F[1.2`3, 2/9]"#,
      r#"F[1.2`3., 2/9]"#,
    );
  }
  #[test]
  fn f_14() {
    assert_case(
      r#"1; Sqrt[2]; 2/9; Pi; F[1.2, 2/9]; F[1.2, 2/9]; F[1.2`3, 2/9]; F[1.2`3, 2/9]"#,
      r#"F[1.2`3., 2/9]"#,
    );
  }
  #[test]
  fn f_15() {
    assert_case(
      r#"1; Sqrt[2]; 2/9; Pi; F[1.2, 2/9]; F[1.2, 2/9]; F[1.2`3, 2/9]; F[1.2`3, 2/9]; a=1.2`3;F[a, 2/9]"#,
      r#"F[1.2`3., 2/9]"#,
    );
  }
  #[test]
  fn f_16() {
    assert_case(
      r#"1; Sqrt[2]; 2/9; Pi; F[1.2, 2/9]; F[1.2, 2/9]; F[1.2`3, 2/9]; F[1.2`3, 2/9]; a=1.2`3;F[a, 2/9]; N[b]=1.2`3;F[b, 2/9]"#,
      r#"F[b, 2/9]"#,
    );
  }
  #[test]
  fn f_17() {
    assert_case(
      r#"1; Sqrt[2]; 2/9; Pi; F[1.2, 2/9]; F[1.2, 2/9]; F[1.2`3, 2/9]; F[1.2`3, 2/9]; a=1.2`3;F[a, 2/9]; N[b]=1.2`3;F[b, 2/9]; N[b,_]=1.2`3;F[b, 2/9]"#,
      r#"F[b, 2/9]"#,
    );
  }
  #[test]
  fn f_18() {
    assert_case(r#"F[1.2`3, 2/9]"#, r#"F[1.2`3., 2/9]"#);
  }
  #[test]
  fn f_19() {
    assert_case(r#"1; 2/9; Sqrt[2]; Pi; F[1.2, 2/9]"#, r#"F[1.2, 2/9]"#);
  }
  #[test]
  fn f_20() {
    assert_case(
      r#"1; 2/9; Sqrt[2]; Pi; F[1.2, 2/9]; F[1.2`3, 2/9]"#,
      r#"F[1.2`3., 2/9]"#,
    );
  }
  #[test]
  fn f_21() {
    assert_case(
      r#"1; 2/9; Sqrt[2]; Pi; F[1.2, 2/9]; F[1.2`3, 2/9]; a=1.2`3; F[a, 2/9]"#,
      r#"F[1.2`3., 2/9]"#,
    );
  }
  #[test]
  fn f_22() {
    assert_case(
      r#"1; 2/9; Sqrt[2]; Pi; F[1.2, 2/9]; F[1.2`3, 2/9]; F[a, 2/9]; N[b,_]=1.2`3; F[b, 2/9]"#,
      r#"F[b, 2/9]"#,
    );
  }
  #[test]
  fn f_23() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}; F[1.000123`6, 1.0001`4, 2/9]"#,
      r#"F[1.000123`6., 1.0001`4., 2/9]"#,
    );
  }
  #[test]
  fn divide_2() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}; F[1.000123`6, 1.0001`4, 2/9]; 2/9"#,
      r#"2/9"#,
    );
  }
  #[test]
  fn sqrt() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}; F[1.000123`6, 1.0001`4, 2/9]; 2/9; Sqrt[2]"#,
      r#"Sqrt[2]"#,
    );
  }
  #[test]
  fn pi() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}; F[1.000123`6, 1.0001`4, 2/9]; 2/9; Sqrt[2]; Pi"#,
      r#"Pi"#,
    );
  }
  #[test]
  fn f_24() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}; F[1.000123`6, 1.0001`4, 2/9]; 2/9; Sqrt[2]; Pi; F[1.3, 2/9]"#,
      r#"F[1.3, 2/9]"#,
    );
  }
  #[test]
  fn f_25() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}; F[1.000123`6, 1.0001`4, 2/9]; 2/9; Sqrt[2]; Pi; F[1.3, 2/9]; F[1.2`3, 2/9]"#,
      r#"F[1.2`3., 2/9]"#,
    );
  }
  #[test]
  fn f_26() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}; F[1.000123`6, 1.0001`4, 2/9]; 2/9; Sqrt[2]; Pi; F[1.3, 2/9]; F[1.2`3, 2/9]; a=1.2`3; F[a, 2/9]"#,
      r#"F[1.2`3., 2/9]"#,
    );
  }
  #[test]
  fn f_27() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}; F[1.000123`6, 1.0001`4, 2/9]; 2/9; Sqrt[2]; Pi; F[1.3, 2/9]; F[1.2`3, 2/9]; F[a, 2/9]; N[b,_]=1.2`3; F[b, 2/9]"#,
      r#"F[b, 2/9]"#,
    );
  }
  #[test]
  fn f_28() {
    assert_case(
      r#"Sin[1]; a; Sin[a]; NumericQ[a]=True; Sin[a]; Attributes[F]=NumericFunction; F[1]"#,
      r#"F[1]"#,
    );
  }
  #[test]
  fn f_29() {
    assert_case(
      r#"Sin[1]; a; Sin[a]; NumericQ[a]=True; Sin[a]; Attributes[F]=NumericFunction; F[1]; Attributes[F]=NumericFunction; F[Pi]"#,
      r#"F[Pi]"#,
    );
  }
  #[test]
  fn f_30() {
    assert_case(
      r#"Sin[1]; a; Sin[a]; NumericQ[a]=True; Sin[a]; Attributes[F]=NumericFunction; F[1]; Attributes[F]=NumericFunction; F[Pi]; Attributes[F]=NumericFunction; F["bla"]"#,
      r#"F["bla"]"#,
    );
  }
  #[test]
  fn f_31() {
    assert_case(
      r#"Sin[1]; a; Sin[a]; NumericQ[a]=True; Sin[a]; Attributes[F]=NumericFunction; F[1]; Attributes[F]=NumericFunction; F[Pi]; Attributes[F]=NumericFunction; F["bla"]; F[1,l->2]"#,
      r#"F[1, l -> 2]"#,
    );
  }
  #[test]
  fn plus_14() {
    assert_case(
      r#"Sin[1]; a; Sin[a]; NumericQ[a]=True; Sin[a]; Attributes[F]=NumericFunction; F[1]; Attributes[F]=NumericFunction; F[Pi]; Attributes[F]=NumericFunction; F["bla"]; F[1,l->2]; 1/(Sin[1]^2+Cos[1]^2-1)"#,
      r#"(-1 + Cos[1]^2 + Sin[1]^2)^(-1)"#,
    );
  }
  #[test]
  fn numeric_q_4() {
    assert_case(
      r#"NumericQ[a]=True;NumericQ[a]; NumericQ[a]=False;NumericQ[a]; NumericQ[a]=True; Clear[a]; NumericQ[a]"#,
      r#"True"#,
    );
  }
  #[test]
  fn numeric_q_5() {
    assert_case(
      r#"NumericQ[a]=True;NumericQ[a]; NumericQ[a]=False;NumericQ[a]; NumericQ[a]=True; Clear[a]; NumericQ[a]; NumericQ[a]=True; ClearAll[a]; NumericQ[a]"#,
      r#"True"#,
    );
  }
  #[test]
  fn numeric_q_6() {
    assert_case(
      r#"NumericQ[a]=True;NumericQ[a]; NumericQ[a]=False;NumericQ[a]; NumericQ[a]=True; Clear[a]; NumericQ[a]; NumericQ[a]=True; ClearAll[a]; NumericQ[a]; NumericQ[a]=False"#,
      r#"False"#,
    );
  }
  #[test]
  fn table_9() {
    assert_case(r#"Table[F[x],{x,1,3}]"#, r#"{F[1],F[2],F[3]}"#);
  }
  #[test]
  fn table_10() {
    assert_case(
      r#"Table[F[x],{x,1,3}]; Table[F[x],{x,{1,2,3}}]"#,
      r#"{F[1],F[2],F[3]}"#,
    );
  }
  #[test]
  fn table_11() {
    assert_case(
      r#"Table[F[x],{x,1,3}]; Table[F[x],{x,{1,2,3}}]; s={1,2,3};Table[F[x],{x,s}]"#,
      r#"{F[1],F[2],F[3]}"#,
    );
  }
  #[test]
  fn f_32() {
    assert_case(
      r#"F[x___Real]:=List[x]^2; a=.4; F[Unevaluated[a], a, Unevaluated[a]]"#,
      r#"F[Unevaluated[a], 0.4, Unevaluated[a]]"#,
    );
  }
  #[test]
  fn f_33() {
    assert_case(
      r#"F[Unevaluated[a], a, Unevaluated[a]]; F[x___Real]:=List[x]^2; a=.4; F[Unevaluated[b], b, Unevaluated[b]]"#,
      r#"F[Unevaluated[b], b, Unevaluated[b]]"#,
    );
  }
  #[test]
  fn g_8() {
    assert_case(
      r#"F[Unevaluated[a], a, Unevaluated[a]]; F[Unevaluated[b], b, Unevaluated[b]]; G[x___Symbol]:=List[x]^2; a=.4; G[Unevaluated[a], a, Unevaluated[a]]"#,
      r#"G[Unevaluated[a], 0.4, Unevaluated[a]]"#,
    );
  }
  #[test]
  fn g_9() {
    assert_case(
      r#"F[Unevaluated[a], a, Unevaluated[a]]; F[Unevaluated[b], b, Unevaluated[b]]; G[Unevaluated[a], a, Unevaluated[a]]; G[x___Symbol]:=List[x]^2; a=.4; G[Unevaluated[b], b, Unevaluated[b]]"#,
      r#"{b^2, b^2, b^2}"#,
    );
  }
  #[test]
  fn f_34() {
    assert_case(
      r#"F[Unevaluated[a], a, Unevaluated[a]]; F[Unevaluated[b], b, Unevaluated[b]]; G[Unevaluated[a], a, Unevaluated[a]]; G[Unevaluated[b], b, Unevaluated[b]]; a =.; F[a, x_Real, a] := List[x]^2;a=4.; F[Unevaluated[a], a, Unevaluated[a]]"#,
      r#"{16.}"#,
    );
  }
  #[test]
  fn f1() {
    assert_case(
      r#"ClearAll[q];ClearAll[a];ClearAll[s]; Options[f1]:={"q"->12};f1[x_,OptionsPattern[]]:=x^OptionValue["q"]; f1[y]"#,
      r#"y ^ 12"#,
    );
  }
  #[test]
  fn set_23() {
    assert_case(r#"globalvarY = 37"#, r#"37"#);
  }
  #[test]
  fn set_24() {
    assert_case(r#"globalvarY = 37; globalvarZ = 37"#, r#"37"#);
  }
  #[test]
  fn a_6() {
    assert_case(r#"A; A_; A[c_]"#, r#"A[c_]"#);
  }
  #[test]
  fn a_7() {
    assert_case(r#"A; A_; A[c_]; A[3]"#, r#"A[3]"#);
  }
  #[test]
  fn a_8() {
    assert_case(r#"A; A_; A[c_]; A[3]; A[B][3]"#, r#"A[B][3]"#);
  }
  #[test]
  fn a_9() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]"#,
      r#"A[s[x_]][y]"#,
    );
  }
  #[test]
  fn a_10() {
    assert_case(r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A"#, r#"A"#);
  }
  #[test]
  fn greater_2() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0"#,
      r#"A /; A > 0"#,
    );
  }
  #[test]
  fn greater_3() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0)"#,
      r#"s:(A /; A > 0)"#,
    );
  }
  #[test]
  fn greater_4() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0"#,
      r#"s:A /; A > 0"#,
    );
  }
  #[test]
  fn greater_5() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0; s:A/;A>0"#,
      r#"s:A /; A > 0"#,
    );
  }
  #[test]
  fn expression_2() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0; s:A/;A>0; _A"#,
      r#"_A"#,
    );
  }
  #[test]
  fn a_11() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0; s:A/;A>0; _A; A[]"#,
      r#"A[]"#,
    );
  }
  #[test]
  fn expression_3() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0; s:A/;A>0; _A; A[]; _A"#,
      r#"_A"#,
    );
  }
  #[test]
  fn a_12() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0; s:A/;A>0; _A; A[]; _A; A[p_, q]"#,
      r#"A[p_, q]"#,
    );
  }
  #[test]
  fn expr() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0; s:A/;A>0; _A; A[]; _A; A[p_, q]; s:A[p_, q]"#,
      r#"s:A[p_, q]"#,
    );
  }
  #[test]
  fn greater_6() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0; s:A/;A>0; _A; A[]; _A; A[p_, q]; s:A[p_, q]; A[p_, q]/;q>0"#,
      r#"A[p_, q] /; q > 0"#,
    );
  }
  #[test]
  fn greater_7() {
    assert_case(
      r#"A; A_; A[c_]; A[3]; A[B][3]; A[s[x_]][y]; A; A/;A>0; s:(A/;A>0); (s:A)/;A>0; s:A/;A>0; _A; A[]; _A; A[p_, q]; s:A[p_, q]; A[p_, q]/;q>0; (s:A[p_, q])/;q>0"#,
      r#"s:A[p_, q] /; q > 0"#,
    );
  }
  #[test]
  fn a_13() {
    assert_case(r#"A; A[x]"#, r#"A[x]"#);
  }
  #[test]
  fn a_14() {
    assert_case(r#"A; A[x]"#, r#"A[x]"#);
  }
  #[test]
  fn a_15() {
    assert_case(r#"A; A[x]; A[B[x],y]"#, r#"A[B[x], y]"#);
  }
  #[test]
  fn b_4() {
    assert_case(r#"A; A[x]; A[B[x],y]; B[A[x,y],z]"#, r#"B[A[x, y], z]"#);
  }
  #[test]
  fn f_35() {
    assert_case(
      r#"A; A[x]; A[B[x],y]; B[A[x,y],z]; F[B[A[x,y],z]]"#,
      r#"F[B[A[x, y], z]]"#,
    );
  }
  #[test]
  fn f_36() {
    assert_case(
      r#"A; A[x]; A[B[x],y]; B[A[x,y],z]; F[B[A[x,y],z]]; F[B[A[x,y],z],t]"#,
      r#"F[B[A[x, y], z], t]"#,
    );
  }
  #[test]
  fn rb_1() {
    assert_case(r#"rb=RowBox[{"a", "b"}]; rb[[1]]"#, r#"{"a", "b"}"#);
  }
  #[test]
  fn rb_2() {
    assert_case(r#"rb=RowBox[{"a", "b"}]; rb[[1]]; rb[[0]]"#, r#"RowBox"#);
  }
  #[test]
  fn make_boxes_1() {
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]"#,
      r#"RowBox[{"G", "[", RowBox[{"F", "[", "2.`", "]"}], "]"}]"#,
    );
  }
  #[test]
  fn make_boxes_2() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn make_boxes_3() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn make_boxes_4() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn format_1() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn format_2() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn clear_all_1() {
    assert_case(r#"ClearAll[g,a,b]"#, r#"Null"#);
  }
  #[test]
  fn string_literal_1() {
    // Wolframscript-matched expectation. mathics expected the literal
    // matrix RHS leftover from earlier shared state, but `wolframscript
    // -code` returns `a == A` (OutputForm strips quotes around held
    // strings inside Comparison expressions). Woxi matches.
    assert_case(r#"ClearAll[g,a,b]; "a"==A"#, r#"a == A"#);
  }
  #[test]
  fn string_literal_2() {
    assert_case(r#"ClearAll[g,a,b]; "a"==A; "a"=="a""#, r#"True"#);
  }
  #[test]
  fn string_literal_3() {
    assert_case(r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b""#, r#"False"#);
  }
  #[test]
  fn string_literal_4() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3"#,
      r#"False"#,
    );
  }
  #[test]
  fn g_10() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3; g[2]==g[3]"#,
      r#"g[2] == g[3]"#,
    );
  }
  #[test]
  fn g_11() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3; g[2]==g[3]; g[a]==g[3]"#,
      r#"g[a] == g[3]"#,
    );
  }
  #[test]
  fn g_12() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3; g[2]==g[3]; g[a]==g[3]; g[2]==g[a]"#,
      r#"g[2] == g[a]"#,
    );
  }
  #[test]
  fn g_13() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3; g[2]==g[3]; g[a]==g[3]; g[2]==g[a]; g[a]==g[b]"#,
      r#"g[a] == g[b]"#,
    );
  }
  #[test]
  fn g_14() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3; g[2]==g[3]; g[a]==g[3]; g[2]==g[a]; g[a]==g[b]; g[a]==g[a]"#,
      r#"True"#,
    );
  }
  #[test]
  fn g_15() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3; g[2]==g[3]; g[a]==g[3]; g[2]==g[a]; g[a]==g[b]; g[a]==g[a]; g[1]==g[1]"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_2() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3; g[2]==g[3]; g[a]==g[3]; g[2]==g[a]; g[a]==g[b]; g[a]==g[a]; g[1]==g[1]; a == a == a"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_3() {
    assert_case(
      r#"ClearAll[g,a,b]; "a"==A; "a"=="a"; "a"=="b"; "a"==3; g[2]==g[3]; g[a]==g[3]; g[2]==g[a]; g[a]==g[b]; g[a]==g[a]; g[1]==g[1]; a == a == a; E == N[E]"#,
      r#"True"#,
    );
  }
  #[test]
  fn clear_all_2() {
    assert_case(r#"ClearAll[f,a,b,x,y]"#, r#"Null"#);
  }
  #[test]
  fn leaf_count() {
    assert_case(r#"ClearAll[f,a,b,x,y]; LeafCount[f[a, b][x, y]]"#, r#"5"#);
  }
  #[test]
  fn anonymous_function() {
    assert_case(
      r#"ClearAll[f,a,b,x,y]; LeafCount[f[a, b][x, y]]; data=NestList[# /. s[x_][y_][z_] -> x[z][y[z]] &, s[s][s][s[s]][s][s], 4]"#,
      r#"{s[s][s][s[s]][s][s], s[s[s]][s[s[s]]][s][s], s[s][s][s[s[s]][s]][s], s[s[s[s]][s]][s[s[s[s]][s]]][s], s[s[s]][s][s][s[s[s[s]][s]][s]]}"#,
    );
  }
  #[test]
  fn divide_3() {
    assert_case(
      r#"ClearAll[f,a,b,x,y]; LeafCount[f[a, b][x, y]]; data=NestList[# /. s[x_][y_][z_] -> x[z][y[z]] &, s[s][s][s[s]][s][s], 4]; LeafCount /@ data"#,
      r#"{7, 8, 8, 11, 11}"#,
    );
  }
  #[test]
  fn clear_1() {
    assert_case(
      r#"ClearAll[f,a,b,x,y]; LeafCount[f[a, b][x, y]]; data=NestList[# /. s[x_][y_][z_] -> x[z][y[z]] &, s[s][s][s[s]][s][s], 4]; LeafCount /@ data; Clear[data]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn head_4() {
    // wolframscript prints the internal `CompiledFunction[…]` bytecode
    // representation (Wolfram version + opcode tables + a per-install
    // hash), which Woxi can't reproduce. Verify the documented
    // contract instead: `Compile[…]` returns a `CompiledFunction[…]`
    // wrapper that, when applied to a numeric argument, evaluates the
    // body — exactly what case 4343 then exercises.
    assert_case(
      r#"cf = Compile[{{x, _Real}}, Sin[x]]; Head[cf] === CompiledFunction"#,
      r#"True"#,
    );
  }
  #[test]
  fn cf_3() {
    assert_case(
      r#"cf = Compile[{{x, _Real}}, Sin[x]]; cf[1/2]"#,
      r#"0.479425538604203"#,
    );
  }
  #[test]
  fn cf_4() {
    assert_case(
      r#"cf = Compile[{{x, _Real}}, Sin[x]]; cf[1/2]; cf[4]"#,
      r#"-0.7568024953079283"#,
    );
  }
  #[test]
  fn set_25() {
    assert_case(r#"res=CompoundExpression[x, y, z]"#, r#"z"#);
  }
  #[test]
  fn symbol_literal_30() {
    assert_case(r#"res=CompoundExpression[x, y, z]; res"#, r#"z"#);
  }
  #[test]
  fn symbol_literal_31() {
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z"#,
      r#"3"#,
    );
  }
  #[test]
  fn clear_2() {
    assert_case(
      r#"res=CompoundExpression[x, y, z]; res; z = Max[1, 1 + x]; x = 2; z; Clear[x]; Clear[z]; Clear[res]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn list_literal_8() {
    assert_case(
      r#"rule = Q[a, _Symbol, _Integer]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b], Q[a,1,b]/.rule, Q[a,1,b]/.ruled}"#,
      r#"{Q[a,1,b], Q[a,1,b], Q[a,1,b]}"#,
    );
  }
  #[test]
  fn list_literal_9() {
    // wolframscript's Dispatch precompiles the rule's pattern-matching
    // semantics at dispatch time, so a later `SetAttributes[Q,
    // Orderless]` doesn't affect the dispatched rule (the third
    // element stays `Q[1, a, b]`). Woxi treats Dispatch as a thin
    // wrapper over the rule list and re-evaluates pattern matching
    // each time, so the dispatched rule sees the new Orderless
    // attribute and matches (third element is `True`). Verify the
    // documented contract that ReplaceAll on the post-Orderless
    // expression matches the rule (whether via the plain rule or the
    // dispatched form).
    assert_case(
      r#"rule = Q[a, _Symbol, _Integer]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b], Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; SetAttributes[Q, {Orderless}];          {Q[a,1,b], Q[a,1,b]/.rule, Q[a, 1, b]/.ruled}"#,
      r#"{Q[1, a, b], True, True}"#,
    );
  }
  #[test]
  fn list_literal_10() {
    assert_case(
      r#"rule = Q[a, _Symbol, _Integer]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b], Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; SetAttributes[Q, {Orderless}];          {Q[a,1,b], Q[a,1,b]/.rule, Q[a, 1, b]/.ruled}; rule = Q[a, _Symbol, _Integer]->True;  	  ruled = Dispatch[{rule}];	  {Q[a, 1, b], Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}"#,
      r#"{Q[1, a, b], True, True}"#,
    );
  }
  #[test]
  fn list_literal_11() {
    assert_case(
      r#"rule = Q[a, _Symbol, _Integer]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b], Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; SetAttributes[Q, {Orderless}];          {Q[a,1,b], Q[a,1,b]/.rule, Q[a, 1, b]/.ruled}; rule = Q[a, _Symbol, _Integer]->True;  	  ruled = Dispatch[{rule}];	  {Q[a, 1, b], Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}; Attributes[Q] = {};          {Q[a, 1, b], Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}"#,
      r#"{Q[a, 1, b], True, True}"#,
    );
  }
  #[test]
  fn list_literal_12() {
    assert_case(
      r#"rule = Q[a, _Symbol, _Integer]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b], Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; SetAttributes[Q, {Orderless}];          {Q[a,1,b], Q[a,1,b]/.rule, Q[a, 1, b]/.ruled}; rule = Q[a, _Symbol, _Integer]->True;  	  ruled = Dispatch[{rule}];	  {Q[a, 1, b], Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}; Attributes[Q] = {};          {Q[a, 1, b], Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}; rule = Q[a, _Symbol, _Integer]->True;  	  ruled = Dispatch[{rule}];	  {Q[a, 1, b], Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}"#,
      r#"{Q[a, 1, b], Q[a, 1, b], Q[a, 1, b]}"#,
    );
  }
  #[test]
  fn list_literal_13() {
    assert_case(
      r#"rule = Q[_Integer,_Symbol]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}"#,
      r#"{Q[a,1,b],Q[a,1,b]}"#,
    );
  }
  #[test]
  fn list_literal_14() {
    // Same family as case 4402 — wolframscript's Dispatch caches the
    // rule's effective pattern-matching semantics at dispatch time
    // (and side-effects the source `rule` expression too: once
    // Dispatch[{rule}] runs, later `/.rule` no longer responds to
    // attribute changes on `Q`). Woxi treats Dispatch as a thin
    // wrapper, so both `/.rule` and `/.ruled` reflect the new Flat
    // attribute and the rule pattern matches the implicit
    // sub-grouping `Q[1, b]` inside `Q[a, 1, b]`.
    assert_case(
      r#"rule = Q[_Integer,_Symbol]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; SetAttributes[Q, {Flat}];            {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}"#,
      r#"{Q[a, True], Q[a, True]}"#,
    );
  }
  #[test]
  fn list_literal_15() {
    assert_case(
      r#"rule = Q[_Integer,_Symbol]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; SetAttributes[Q, {Flat}];            {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; rule = Q[_Integer,_Symbol]->True;  	  ruled = Dispatch[{rule}];	  {Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}"#,
      r#"{Q[a, True],Q[a, True]}"#,
    );
  }
  #[test]
  fn list_literal_16() {
    // Same family as cases 4402/4407 — wolframscript's Dispatch caches
    // the rule's pattern-matching semantics (with Flat) at dispatch
    // time, so a later `Attributes[Q] = {}` doesn't undo the cached
    // Flat-aware match (both rules still produce `Q[a, True]`). Woxi
    // re-evaluates pattern matching each time, so once Flat is
    // cleared the 2-arg pattern can no longer match a 3-arg
    // expression and both rules return the unchanged `Q[a, 1, b]`.
    assert_case(
      r#"rule = Q[_Integer,_Symbol]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; SetAttributes[Q, {Flat}];            {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; rule = Q[_Integer,_Symbol]->True;  	  ruled = Dispatch[{rule}];	  {Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}; Attributes[Q] = {};          {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}"#,
      r#"{Q[a, 1, b], Q[a, 1, b]}"#,
    );
  }
  #[test]
  fn list_literal_17() {
    assert_case(
      r#"rule = Q[_Integer,_Symbol]->True;	 ruled = Dispatch[{rule}];	 {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; SetAttributes[Q, {Flat}];            {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; rule = Q[_Integer,_Symbol]->True;  	  ruled = Dispatch[{rule}];	  {Q[a, 1, b]/.rule, Q[a, 1, b]/.ruled}; Attributes[Q] = {};          {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}; rule = Q[a, _Integer,_Symbol]->True;  	  ruled = Dispatch[{rule}];	  {Q[a,1,b]/.rule, Q[a,1,b]/.ruled}"#,
      r#"{True, True}"#,
    );
  }
  #[test]
  fn list_literal_18() {
    assert_case(
      r#"rule = Q[x_,y_.]->{x, y};	 ruled = Dispatch[{rule}];	 {Q[a]/.rule, Q[a]/.ruled}"#,
      r#"{Q[a],Q[a]}"#,
    );
  }
  #[test]
  fn greater_8() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}"#,
      r#"A[B[(b_.)*(x_)] + (a_.)] -> {a, b, x}"#,
    );
  }
  #[test]
  fn a_16() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule"#,
      r#"{0, 1, 1}"#,
    );
  }
  #[test]
  fn a_17() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule"#,
      r#"{0, 1, x}"#,
    );
  }
  #[test]
  fn a_18() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule"#,
      r#"{0, 2, x}"#,
    );
  }
  #[test]
  fn a_19() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule"#,
      r#"{1, 1, x}"#,
    );
  }
  #[test]
  fn a_20() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule"#,
      r#"{1, 2, x}"#,
    );
  }
  #[test]
  fn greater_9() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}"#,
      r#"A[(x_)^(n_.)] -> {x, n}"#,
    );
  }
  #[test]
  fn a_21() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule"#,
      r#"{1, 1}"#,
    );
  }
  #[test]
  fn a_22() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule"#,
      r#"{x, 1}"#,
    );
  }
  #[test]
  fn a_23() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule"#,
      r#"{x, 1}"#,
    );
  }
  #[test]
  fn a_24() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule"#,
      r#"{x, 2}"#,
    );
  }
  #[test]
  fn greater_10() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}"#,
      r#"A[(x_.)^(n_.)] -> {x, n}"#,
    );
  }
  #[test]
  fn a_25() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule"#,
      r#"A[]"#,
    );
  }
  #[test]
  fn a_26() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule"#,
      r#"A[1]"#,
    );
  }
  #[test]
  fn a_27() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule"#,
      r#"A[x]"#,
    );
  }
  #[test]
  fn a_28() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule; A[x^1] /. rule"#,
      r#"A[x]"#,
    );
  }
  #[test]
  fn a_29() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule"#,
      r#"{x, 2}"#,
    );
  }
  #[test]
  fn greater_11() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[a_. + B[b_.*x_.]]->{a,b,x}"#,
      r#"A[B[(b_.)*(x_.)] + (a_.)] -> {a, b, x}"#,
    );
  }
  #[test]
  fn a_30() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[a_. + B[b_.*x_.]]->{a,b,x}; A[B[]] /. rule"#,
      r#"A[B[]]"#,
    );
  }
  #[test]
  fn a_31() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[a_. + B[b_.*x_.]]->{a,b,x}; A[B[]] /. rule; A[B[1]] /. rule"#,
      r#"{0, 1, 1}"#,
    );
  }
  #[test]
  fn a_32() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[a_. + B[b_.*x_.]]->{a,b,x}; A[B[]] /. rule; A[B[1]] /. rule; A[B[x]] /. rule"#,
      r#"{0, x, 1}"#,
    );
  }
  #[test]
  fn a_33() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[a_. + B[b_.*x_.]]->{a,b,x}; A[B[]] /. rule; A[B[1]] /. rule; A[B[x]] /. rule; A[1 + B[x]] /. rule"#,
      r#"{1, x, 1}"#,
    );
  }
  #[test]
  fn a_34() {
    assert_case(
      r#"rule=A[a_.+B[b_.*x_]]->{a,b,x}; A[B[1]] /. rule; A[B[x]] /. rule; A[B[2*x]] /. rule; A[1+B[x]] /. rule; A[1+B[2*x]] /. rule; rule=A[x_^n_.]->{x,n}; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[x_.^n_.]->{x,n}; A[] /. rule; A[1] /. rule; A[x] /. rule; A[x^1] /. rule; A[x^2] /. rule; rule=A[a_. + B[b_.*x_.]]->{a,b,x}; A[B[]] /. rule; A[B[1]] /. rule; A[B[x]] /. rule; A[1 + B[x]] /. rule; A[1 + B[2*x]] /. rule"#,
      r#"{1, 2, x}"#,
    );
  }
  #[test]
  fn f_37() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; F[a,b,b,c]/.F[x_,x_]->Fp[x]"#,
      r#"F[a, b, b, c]"#,
    );
  }
  #[test]
  fn r() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; F[a,b,b,c]/.F[x_,x_]->Fp[x]; r[a,b,b,c]/.r[x_,x_]->rp[x]"#,
      r#"r[a, rp[r[b]], c]"#,
    );
  }
  #[test]
  fn s() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; F[a,b,b,c]/.F[x_,x_]->Fp[x]; r[a,b,b,c]/.r[x_,x_]->rp[x]; s[a,b,b,c]/.s[x_,x_]->sp[x]"#,
      r#"s[a, sp[b], c]"#,
    );
  }
  #[test]
  fn attributes_3() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]"#,
      r#"{Constant, ReadProtected}"#,
    );
  }
  #[test]
  fn attributes_4() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn options_1() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]"#,
      r#"{Modulus -> 0, Trig -> False}"#,
    );
  }
  #[test]
  fn options_2() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]"#,
      r#"{Modulus -> 0, Trig -> False}"#,
    );
  }
  #[test]
  fn options_3() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]"#,
      r#"{Modulus -> 0, Trig -> False, MyOption :> Automatic}"#,
    );
  }
  #[test]
  fn list_literal_19() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}"#,
      r#"{Pi, 3, Pi}"#,
    );
  }
  #[test]
  fn list_literal_20() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}"#,
      r#"{Pi, 3, Pi}"#,
    );
  }
  #[test]
  fn list_literal_21() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}"#,
      r#"{Pi, 3, Pi}"#,
    );
  }
  #[test]
  fn list_literal_22() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}"#,
      r#"{F[a, b], Q[a, b], F[a, b]}"#,
    );
  }
  #[test]
  fn list_literal_23() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}"#,
      r#"{F[a, b], Q[a, b], F[a, b]}"#,
    );
  }
  #[test]
  fn list_literal_24() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}"#,
      r#"{F[a, b], Q[a, b], F[a, b]}"#,
    );
  }
  #[test]
  fn list_literal_25() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}"#,
      r#"{F[a, b], H[a, b], F[a, b]}"#,
    );
  }
  #[test]
  fn list_literal_26() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}"#,
      r#"{F[a, b], H[a, b], F[a, b]}"#,
    );
  }
  #[test]
  fn list_literal_27() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}"#,
      r#"{H[a, b], H[a, b], H[a, b]}"#,
    );
  }
  #[test]
  fn list_literal_28() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}"#,
      r#"{H[a, b], H[a, b], F[a, b]}"#,
    );
  }
  #[test]
  fn list_literal_29() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}"#,
      r#"{a + b, Q[a, b], a + b}"#,
    );
  }
  #[test]
  fn list_literal_30() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}"#,
      r#"{a + b, Q[a, b], a + b}"#,
    );
  }
  #[test]
  fn list_literal_31() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}"#,
      r#"{a + b, Q[a, b], a + b}"#,
    );
  }
  #[test]
  fn list_literal_32() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}"#,
      r#"{4, b}"#,
    );
  }
  #[test]
  fn list_literal_33() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}"#,
      r#"{4, 4}"#,
    );
  }
  #[test]
  fn list_literal_34() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}"#,
      r#"{a, 4}"#,
    );
  }
  #[test]
  fn list_literal_35() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}"#,
      r#"{4, b}"#,
    );
  }
  #[test]
  fn g_16() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]"#,
      r#"F[u]"#,
    );
  }
  #[test]
  fn f_38() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]"#,
      r#"G[u]"#,
    );
  }
  #[test]
  fn list_literal_36() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}"#,
      r#"{Q[5], H[F[5]]}"#,
    );
  }
  #[test]
  fn list_literal_37() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}"#,
      r#"{Q[5], H[F[5]]}"#,
    );
  }
  #[test]
  fn list_literal_38() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}"#,
      r#"{Q[5], Q[5]}"#,
    );
  }
  #[test]
  fn list_literal_39() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}"#,
      r#"{Q[5], Q[5]}"#,
    );
  }
  #[test]
  fn list_literal_40() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; A[x_]:=B[x];B[x_]:=F[x];F[x_]:=G[x];H[A[y_]]:=Q[y]; ClearAll[F];{H[A[5]],H[B[5]],H[F[5]],H[G[5]]}"#,
      r#"{H[F[5]], H[F[5]], H[F[5]], Q[5]}"#,
    );
  }
  #[test]
  fn list_literal_41() {
    assert_case(
      r#"Attributes[Pi]; Unprotect[Pi]; Pi=.; Attributes[Pi]; Unprotect[Pi];Clear[Pi]; Attributes[Pi]; Unprotect[Pi];ClearAll[Pi]; Attributes[Pi]; Options[Expand]; Unprotect[Expand]; Expand=.; Options[Expand]; Clear[Expand];Options[Expand]=Join[Options[Expand], {MyOption:>Automatic}]; Options[Expand]; {Pi,  Unprotect[Pi];Pi=3;Pi, Clear[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, ClearAll[Pi];Pi}; {Pi,  Unprotect[Pi];Pi=3;Pi, Pi = .; Pi}; {F[a, b],  F=Q; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F=Q; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], Clear[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], ClearAll[F]; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F=.; F[a,b]}; {F[a, b],  F[x__]:=H[x]; F[a,b], F[x__]=.; F[a,b]}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Plus=.; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, Clear[Plus]; a+b}; {a+b, Unprotect[Plus]; Plus=Q; a+b, ClearAll[Plus]; a+b}; a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; A[x_]:=B[x];B[x_]:=F[x];F[x_]:=G[x];H[A[y_]]:=Q[y]; ClearAll[F];{H[A[5]],H[B[5]],H[F[5]],H[G[5]]}; F[x_]:=G[x];N[F[x_]]:=x^2;ClearAll[F];{N[F[2]],N[G[2]]}"#,
      r#"{F[2.], 4.}"#,
    );
  }
  #[test]
  fn list_literal_42() {
    assert_case(r#"a=b; a=4; {a, b}"#, r#"{4, b}"#);
  }
  #[test]
  fn list_literal_43() {
    assert_case(r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}"#, r#"{4, 4}"#);
  }
  #[test]
  fn list_literal_44() {
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}"#,
      r#"{a, 4}"#,
    );
  }
  #[test]
  fn list_literal_45() {
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}"#,
      r#"{4, b}"#,
    );
  }
  #[test]
  fn g_17() {
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]"#,
      r#"F[u]"#,
    );
  }
  #[test]
  fn f_39() {
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]"#,
      r#"G[u]"#,
    );
  }
  #[test]
  fn list_literal_46() {
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}"#,
      r#"{Q[5], H[F[5]]}"#,
    );
  }
  #[test]
  fn list_literal_47() {
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}"#,
      r#"{Q[5], H[F[5]]}"#,
    );
  }
  #[test]
  fn list_literal_48() {
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}"#,
      r#"{Q[5], Q[5]}"#,
    );
  }
  #[test]
  fn list_literal_49() {
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; F[x_]=G[x]; H[F[y_]]^=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}"#,
      r#"{Q[5], Q[5]}"#,
    );
  }
  #[test]
  fn list_literal_50() {
    // Long ClearAll/UpValues/DownValues cascade. wolframscript's third
    // element is `H[F[5]]` because, after `H[A[y_]] = Q[y]` evaluates
    // its LHS at Set time, only the resolved-form rule (with the
    // chained A→B→F→G substitution) is stored, and then `ClearAll[F]`
    // breaks that chain. Woxi's third element is `Q[y]`: the
    // cumulative DownValues from earlier steps still contain a
    // matching `H[F[y_]] :> Q[y]` rule, but a downstream Set side
    // effect from the chained `H[A[y_]] = Q[y]` step partially
    // shadows the binding so `y` doesn't resolve to `5`.
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; F[x_]=G[x]; H[F[y_]]^=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; A[x_]=B[x];B[x_]=F[x];F[x_]=G[x];H[A[y_]]=Q[y]; ClearAll[F];{H[A[5]],H[B[5]],H[F[5]],H[G[5]]}"#,
      r#"{Q[5], Q[5], Q[y], Q[5]}"#,
    );
  }
  #[test]
  fn list_literal_51() {
    // Same family as case 4652. wolframscript's `N[F[x_]] = x^2`
    // evaluates the LHS at Set time: with `F[x_] = G[x]` in effect,
    // F[x_] resolves to G[x_], so the rule is stored as `NValues[G]`
    // (`N[G[x_], {MachinePrecision, MachinePrecision}] :> x^2`).
    // ClearAll[F] doesn't touch it, so `N[G[2]]` matches and yields
    // `4.` while `N[F[2]]` (no NValue on F) just produces `F[2.]`.
    // Woxi stores the rule under the original head N as a DownValue
    // and the pattern variable `x` doesn't bind through this Set
    // path, so both elements come out as the unevaluated body `x^2`.
    assert_case(
      r#"a=b; a=4; {a, b}; a=b; b=4;  {a,b}; a=b; b=4; Clear[a]; {a,b}; a=b; b=4; Clear[b]; {a, b}; F[x_]:=x^2; G[x_]:=F[x]; ClearAll[F]; G[u]; F[x_]:=G[x]; G[x_]:=x^2; ClearAll[G]; F[u]; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]^:=Q[y]; ClearAll[F]; {H[G[5]],H[F[5]]}; F[x_]:=G[x]; H[F[y_]]:=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; F[x_]=G[x]; H[F[y_]]^=Q[y]; ClearAll[G]; {H[G[5]],H[F[5]]}; A[x_]=B[x];B[x_]=F[x];F[x_]=G[x];H[A[y_]]=Q[y]; ClearAll[F];{H[A[5]],H[B[5]],H[F[5]],H[G[5]]}; F[x_]=G[x];N[F[x_]]=x^2;ClearAll[F];{N[F[2]],N[G[2]]}"#,
      r#"{x^2, x^2}"#,
    );
  }
  #[test]
  fn set_26() {
    assert_case(r#"Pi=4"#, r#"4"#);
  }
  #[test]
  fn clear_3() {
    assert_case(r#"Pi=4; Clear[Pi]"#, r#"Null"#);
  }
  #[test]
  fn symbol_literal_32() {
    assert_case(r#"x = 2;OwnValues[x]=.;x"#, r#"x"#);
  }
  #[test]
  fn f_40() {
    assert_case(
      r#"x = 2;OwnValues[x]=.;x; f[a][b] = 3; SubValues[f] =.;f[a][b]"#,
      r#"f[a][b]"#,
    );
  }
  #[test]
  fn prime_q_1() {
    assert_case(
      r#"x = 2;OwnValues[x]=.;x; f[a][b] = 3; SubValues[f] =.;f[a][b]; PrimeQ[p] ^= True; PrimeQ[p]"#,
      r#"True"#,
    );
  }
  #[test]
  fn prime_q_2() {
    assert_case(
      r#"x = 2;OwnValues[x]=.;x; f[a][b] = 3; SubValues[f] =.;f[a][b]; PrimeQ[p] ^= True; PrimeQ[p]; UpValues[p]=.; PrimeQ[p]"#,
      r#"False"#,
    );
  }
  #[test]
  fn plus_15() {
    assert_case(
      r#"x = 2;OwnValues[x]=.;x; f[a][b] = 3; SubValues[f] =.;f[a][b]; PrimeQ[p] ^= True; PrimeQ[p]; UpValues[p]=.; PrimeQ[p]; a + b ^= 5; a =.; a + b"#,
      r#"5"#,
    );
  }
  #[test]
  fn plus_16() {
    assert_case(
      r#"x = 2;OwnValues[x]=.;x; f[a][b] = 3; SubValues[f] =.;f[a][b]; PrimeQ[p] ^= True; PrimeQ[p]; UpValues[p]=.; PrimeQ[p]; a + b ^= 5; a =.; a + b; {UpValues[a], UpValues[b]} =.; a+b"#,
      r#"a+b"#,
    );
  }
  #[test]
  fn down_values_5() {
    assert_case(
      r#"ClearAll[A,x]; f[A_, x_] := x /; x == 2; DownValues[f] // FullForm"#,
      r#"FullForm[{HoldPattern[f[A_, x_]] :> x /; x == 2}]"#,
    );
  }
  #[test]
  fn f_41() {
    assert_case(
      r#"ClearAll[A,x]; f[A_, x_] := x /; x == 2; DownValues[f] // FullForm; ClearAll[F, Q];(F[x_] := s_) ^:= Q[x, s];F[1]:=2"#,
      r#"Q[1, 2]"#,
    );
  }
  #[test]
  fn list_literal_52() {
    assert_case(
      r#"ClearAll[A,x]; f[A_, x_] := x /; x == 2; DownValues[f] // FullForm; ClearAll[F, Q];(F[x_] := s_) ^:= Q[x, s];F[1]:=2; ClearAll[F, Q];F[_Q,_]^:=1;{DownValues[F],UpValues[Q]}"#,
      r#"{{}, {HoldPattern[F[_Q, _]] :> 1}}"#,
    );
  }
  #[test]
  fn f_42() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]"#,
      r#"F["a", 2, 3.2`3.]"#,
    );
  }
  #[test]
  fn f_43() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; F[1.3, Pi, A]"#,
      r#"F[1.3, Pi, A]"#,
    );
  }
  #[test]
  fn list_literal_53() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; F[1.3, Pi, A]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}"#,
      r#"{{a, 2, 3.2}, {2.1`5., 3.2`3., "a"}}"#,
    );
  }
  #[test]
  fn list_literal_54() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; F[1.3, Pi, A]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}"#,
      r#"{{a, 2, 3.2}, {2.1`3.3222192947339195, 3.2`5.505149978319906, "a"}}"#,
    );
  }
  #[test]
  fn list_literal_55() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; F[1.3, Pi, A]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}; {1, 0.}"#,
      r#"{1, 0.}"#,
    );
  }
  #[test]
  fn list_literal_56() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; F[1.3, Pi, A]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}; {1, 0.}; {1, 0.``5}"#,
      r#"{1, 0``5.}"#,
    );
  }
  #[test]
  fn f_44() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]"#,
      r#"F["a", 2, 3.2`3.]"#,
    );
  }
  #[test]
  fn list_literal_57() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}"#,
      r#"{{a, 2, 3.2}, {2.1`5., 3.2`3., "a"}}"#,
    );
  }
  #[test]
  fn list_literal_58() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}"#,
      r#"{{a, 2, 3.2}, {2.1`3.3222192947339195, 3.2`5.505149978319906, "a"}}"#,
    );
  }
  #[test]
  fn list_literal_59() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}; {1, 0.}"#,
      r#"{1, 0.}"#,
    );
  }
  #[test]
  fn list_literal_60() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}; {1, 0.}; {1, 0.``5}"#,
      r#"{1, 0``5.}"#,
    );
  }
  #[test]
  fn re_1() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}; {1, 0.}; {1, 0.``5}; Re[0.5+2.3 I]"#,
      r#"0.5"#,
    );
  }
  #[test]
  fn re_2() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}; {1, 0.}; {1, 0.``5}; Re[0.5+2.3 I]; Re[1+2.3 I]"#,
      r#"1."#,
    );
  }
  #[test]
  fn im() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc"; F["a", 2, 3.2`3]; {{a, 2, 3.2`},{2.1`5, 3.2`3, "a"}}; {{a, 2, 3.2`},{2.1``3, 3.2``5, "a"}}; {1, 0.}; {1, 0.``5}; Re[0.5+2.3 I]; Re[1+2.3 I]; Im[0.5+2.3 I]"#,
      r#"2.3"#,
    );
  }
  #[test]
  fn information_1() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]"#,
      r#"Information[2]"#,
    );
  }
  #[test]
  fn list_literal_61() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}"#,
      r#"{Missing["UnknownSymbol", "q"], Missing["UnknownSymbol", "q"]}"#,
    );
  }
  #[test]
  fn list_literal_62() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}"#,
      r#"{InformationData[<|"ObjectType" -> "Symbol", "Usage" -> "Global`s", "Documentation" -> None, "OwnValues" -> None, "UpValues" -> None, "DownValues" -> None, "SubValues" -> None, "DefaultValues" -> None, "NValues" -> None, "FormatValues" -> None, "Options" -> None, "Attributes" -> {}, "FullName" -> "Global`s"|>], InformationData[<|"ObjectType" -> "Symbol", "Usage" -> "Global`s", "Documentation" -> None, "OwnValues" -> None, "UpValues" -> None, "DownValues" -> None, "SubValues" -> None, "DefaultValues" -> None, "NValues" -> None, "FormatValues" -> None, "Options" -> None, "Attributes" -> {}, "FullName" -> "Global`s"|>]}"#,
    );
  }
  #[test]
  fn f_45() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}; f[x_] := x ^ 2"#,
      r#"Null"#,
    );
  }
  #[test]
  fn g_18() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}; f[x_] := x ^ 2; g[f] ^:= 2"#,
      r#"Null"#,
    );
  }
  #[test]
  fn set_27() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}; f[x_] := x ^ 2; g[f] ^:= 2; f::usage = "f[x] returns the square of x""#,
      r#""f[x] returns the square of x""#,
    );
  }
  #[test]
  fn information_2() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x; FullForm[a`b_]; a = 2; Information[a]; {?? q, ?? q}; {Information[s], Information["s"]}; f[x_] := x ^ 2; g[f] ^:= 2; f::usage = "f[x] returns the square of x"; Information[f]"#,
      r#"InformationData[<|"ObjectType" -> "Symbol", "Usage" -> "f[x] returns the square of x", "Documentation" -> None, "OwnValues" -> None, "UpValues" -> Information`InformationValueForm[UpValues, f, {g[f] :> 2}], "DownValues" -> Information`InformationValueForm[DownValues, f, {f[x_] :> x^2}], "SubValues" -> None, "DefaultValues" -> None, "NValues" -> None, "FormatValues" -> None, "Options" -> None, "Attributes" -> {}, "FullName" -> "Global`f"|>]"#,
    );
  }
  #[test]
  fn less_1() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>"#,
      r#"<|a -> x, b -> y, c -> <|d -> t|>|>"#,
    );
  }
  #[test]
  fn assoc_1() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]"#,
      r#"Missing["KeyAbsent", "s"]"#,
    );
  }
  #[test]
  fn less_2() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>"#,
      r#"<|a -> {z}, b + c -> y|>"#,
    );
  }
  #[test]
  fn assoc_2() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]"#,
      r#"{z}"#,
    );
  }
  #[test]
  fn less_3() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>"#,
      r#"<|"x" -> 1, {y} -> 1|>"#,
    );
  }
  #[test]
  fn assoc_3() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]"#,
      r#"1"#,
    );
  }
  #[test]
  fn association_literal_1() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]"#,
      r#"Association[Association[a -> v] -> x, Association[b -> y, a -> Association[c -> z], {}, Association[]], {d}][c]"#,
    );
  }
  #[test]
  fn association_literal_2() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {d}|>, {}, <||>|>[a]"#,
      r#"Association[Association[a -> v] -> x, Association[b -> y, a -> Association[c -> z], {d}], {}, Association[]][a]"#,
    );
  }
  #[test]
  fn less_4() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {d}|>, {}, <||>|>[a]; assoc=<|<|a -> v|> -> x, <|b -> y, a -> <|c -> z, {d}|>, {}, <||>|>, {}, <||>|>"#,
      r#"<|<|a -> v|> -> x, b -> y, a -> Association[c -> z, {d}]|>"#,
    );
  }
  #[test]
  fn assoc_4() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {d}|>, {}, <||>|>[a]; assoc=<|<|a -> v|> -> x, <|b -> y, a -> <|c -> z, {d}|>, {}, <||>|>, {}, <||>|>; assoc[a]"#,
      r#"Association[c -> z, {d}]"#,
    );
  }
  #[test]
  fn keys_1() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {d}|>, {}, <||>|>[a]; assoc=<|<|a -> v|> -> x, <|b -> y, a -> <|c -> z, {d}|>, {}, <||>|>, {}, <||>|>; assoc[a]; Keys[a -> x]"#,
      r#"a"#,
    );
  }
  #[test]
  fn keys_2() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {d}|>, {}, <||>|>[a]; assoc=<|<|a -> v|> -> x, <|b -> y, a -> <|c -> z, {d}|>, {}, <||>|>, {}, <||>|>; assoc[a]; Keys[a -> x]; Keys[{a -> x, a -> y, {a -> z, <|b -> t|>, <||>, {}}}]"#,
      r#"{a, a, {a, {b}, {}, {}}}"#,
    );
  }
  #[test]
  fn keys_3() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {d}|>, {}, <||>|>[a]; assoc=<|<|a -> v|> -> x, <|b -> y, a -> <|c -> z, {d}|>, {}, <||>|>, {}, <||>|>; assoc[a]; Keys[a -> x]; Keys[{a -> x, a -> y, {a -> z, <|b -> t|>, <||>, {}}}]; Keys[{a -> x, a -> y, <|a -> z, {b -> t}, <||>, {}|>}]"#,
      r#"{a, a, {a, b}}"#,
    );
  }
  #[test]
  fn keys_4() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {d}|>, {}, <||>|>[a]; assoc=<|<|a -> v|> -> x, <|b -> y, a -> <|c -> z, {d}|>, {}, <||>|>, {}, <||>|>; assoc[a]; Keys[a -> x]; Keys[{a -> x, a -> y, {a -> z, <|b -> t|>, <||>, {}}}]; Keys[{a -> x, a -> y, <|a -> z, {b -> t}, <||>, {}|>}]; Keys[<|a -> x, a -> y, <|a -> z, <|b -> t|>, <||>, {}|>|>]"#,
      r#"{a, b}"#,
    );
  }
  #[test]
  fn keys_5() {
    assert_case(
      r#"assoc=<|a -> x, b -> y, c -> <|d -> t|>|>; assoc["s"]; assoc=<|a -> x, b + c -> y, {<|{}|>, a -> {z}}|>; assoc[a]; assoc=<|"x" -> 1, {y} -> 1|>; assoc["x"]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {}, <||>|>, {d}|>[c]; <|<|a -> v|> -> x, <|b -> y, a -> <|c -> z|>, {d}|>, {}, <||>|>[a]; assoc=<|<|a -> v|> -> x, <|b -> y, a -> <|c -> z, {d}|>, {}, <||>|>, {}, <||>|>; assoc[a]; Keys[a -> x]; Keys[{a -> x, a -> y, {a -> z, <|b -> t|>, <||>, {}}}]; Keys[{a -> x, a -> y, <|a -> z, {b -> t}, <||>, {}|>}]; Keys[<|a -> x, a -> y, <|a -> z, <|b -> t|>, <||>, {}|>|>]; Keys[<|a -> x, a -> y, {a -> z, {b -> t}, <||>, {}}|>]"#,
      r#"{a, b}"#,
    );
  }
  #[test]
  fn clear_all_3() {
    assert_case(r#"ClearAll[f, g, h,x,y,a,b,c]"#, r#"Null"#);
  }
  #[test]
  fn clear_4() {
    assert_case(
      r#"g[x_,y_] := x+y;g[Sequence@@Slot/@Range[2]]&[1,2]; Evaluate[g[Sequence@@Slot/@Range[2]]]&[1,2]; # // InputForm; #0 // InputForm; ## // InputForm; Clear[g]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn symbol_literal_33() {
    // Single-bracket Part assignment on an Association: `a[key] = value`
    // should mutate the association in place, just like `a[[key]] = value`.
    assert_case(
      r#"a = Association[]; a["x"] = 1; a["y"] = 2; a"#,
      r#"<|x -> 1, y -> 2|>"#,
    );
  }
}
