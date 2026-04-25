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
