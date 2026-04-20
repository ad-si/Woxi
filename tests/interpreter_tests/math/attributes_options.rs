use super::*;

mod attributes {
  use super::*;

  #[test]
  fn plus() {
    assert_eq!(
      interpret("Attributes[Plus]").unwrap(),
      "{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"
    );
  }

  #[test]
  fn hold() {
    assert_eq!(
      interpret("Attributes[Hold]").unwrap(),
      "{HoldAll, Protected}"
    );
  }

  #[test]
  fn if_func() {
    assert_eq!(
      interpret("Attributes[If]").unwrap(),
      "{HoldRest, Protected}"
    );
  }

  #[test]
  fn set_func() {
    assert_eq!(
      interpret("Attributes[Set]").unwrap(),
      "{HoldFirst, Protected, SequenceHold}"
    );
  }

  #[test]
  fn and_func() {
    assert_eq!(
      interpret("Attributes[And]").unwrap(),
      "{Flat, HoldAll, OneIdentity, Protected}"
    );
  }

  #[test]
  fn constant_e() {
    assert_eq!(
      interpret("Attributes[E]").unwrap(),
      "{Constant, Protected, ReadProtected}"
    );
  }

  #[test]
  fn sin_func() {
    assert_eq!(
      interpret("Attributes[Sin]").unwrap(),
      "{Listable, NumericFunction, Protected}"
    );
  }

  #[test]
  fn unknown_func() {
    assert_eq!(interpret("Attributes[unknownfunc]").unwrap(), "{}");
  }

  #[test]
  fn string_arg() {
    assert_eq!(
      interpret("Attributes[\"Plus\"]").unwrap(),
      "{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"
    );
  }

  #[test]
  fn non_symbol_arg_returns_unevaluated() {
    assert_eq!(
      interpret("Attributes[a + b + c]").unwrap(),
      "Attributes[a + b + c]"
    );
  }

  #[test]
  fn hold_complete() {
    assert_eq!(
      interpret("Attributes[HoldComplete]").unwrap(),
      "{HoldAllComplete, Protected}"
    );
  }

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("Attributes[Unevaluated]").unwrap(),
      "{HoldAllComplete, Protected}"
    );
  }
}

mod hold_attributes {
  use super::*;

  #[test]
  fn hold_first() {
    assert_eq!(
      interpret("SetAttributes[f, HoldFirst]; f[5 + 6, 5 + 6]").unwrap(),
      "f[5 + 6, 11]"
    );
  }

  #[test]
  fn hold_all() {
    assert_eq!(
      interpret("SetAttributes[g, HoldAll]; g[5 + 6, 5 + 6]").unwrap(),
      "g[5 + 6, 5 + 6]"
    );
  }

  #[test]
  fn hold_rest() {
    assert_eq!(
      interpret("SetAttributes[h, HoldRest]; h[5 + 6, 5 + 6]").unwrap(),
      "h[11, 5 + 6]"
    );
  }

  #[test]
  fn hold_first_single_arg() {
    assert_eq!(
      interpret("SetAttributes[f, HoldFirst]; f[3 + 4]").unwrap(),
      "f[3 + 4]"
    );
  }

  #[test]
  fn hold_rest_single_arg() {
    assert_eq!(
      interpret("SetAttributes[h, HoldRest]; h[3 + 4]").unwrap(),
      "h[7]"
    );
  }

  #[test]
  fn hold_all_three_args() {
    assert_eq!(
      interpret("SetAttributes[g, HoldAll]; g[1 + 2, 3 + 4, 5 + 6]").unwrap(),
      "g[1 + 2, 3 + 4, 5 + 6]"
    );
  }

  #[test]
  fn hold_first_three_args() {
    assert_eq!(
      interpret("SetAttributes[f, HoldFirst]; f[1 + 2, 3 + 4, 5 + 6]").unwrap(),
      "f[1 + 2, 7, 11]"
    );
  }

  #[test]
  fn hold_rest_three_args() {
    assert_eq!(
      interpret("SetAttributes[h, HoldRest]; h[1 + 2, 3 + 4, 5 + 6]").unwrap(),
      "h[3, 3 + 4, 5 + 6]"
    );
  }

  #[test]
  fn and_short_circuit() {
    // And with HoldAll should still short-circuit
    assert_eq!(interpret("And[False, Print[\"no\"]]").unwrap(), "False");
  }

  #[test]
  fn or_short_circuit() {
    // Or with HoldAll should still short-circuit
    assert_eq!(interpret("Or[True, Print[\"no\"]]").unwrap(), "True");
  }

  #[test]
  fn and_basic() {
    assert_eq!(interpret("And[True, True]").unwrap(), "True");
    assert_eq!(interpret("And[True, False]").unwrap(), "False");
  }

  #[test]
  fn and_zero_args() {
    // And[] is the identity element: True
    assert_eq!(interpret("And[]").unwrap(), "True");
  }

  #[test]
  fn and_single_arg() {
    assert_eq!(interpret("And[True]").unwrap(), "True");
    assert_eq!(interpret("And[False]").unwrap(), "False");
    assert_eq!(interpret("And[x]").unwrap(), "x");
  }

  #[test]
  fn or_basic() {
    assert_eq!(interpret("Or[False, False]").unwrap(), "False");
    assert_eq!(interpret("Or[False, True]").unwrap(), "True");
  }

  #[test]
  fn or_zero_args() {
    // Or[] is the identity element: False
    assert_eq!(interpret("Or[]").unwrap(), "False");
  }

  #[test]
  fn or_single_arg() {
    assert_eq!(interpret("Or[True]").unwrap(), "True");
    assert_eq!(interpret("Or[False]").unwrap(), "False");
    assert_eq!(interpret("Or[x]").unwrap(), "x");
  }
}

mod options {
  use super::*;

  #[test]
  fn set_and_get() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f]").unwrap(),
      "{a -> 1, b -> 2}"
    );
  }

  #[test]
  fn get_specific_option() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, a]").unwrap(),
      "{a -> 1}"
    );
  }

  #[test]
  fn get_second_option() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, b]").unwrap(),
      "{b -> 2}"
    );
  }

  #[test]
  fn unknown_function() {
    assert_eq!(interpret("Options[unknownfunc]").unwrap(), "{}");
  }

  #[test]
  fn option_not_found() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, c]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn overwrite_options() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> 1}; Options[f] = {a -> 10, b -> 20}; Options[f]"
      )
      .unwrap(),
      "{a -> 10, b -> 20}"
    );
  }

  #[test]
  fn single_rule() {
    assert_eq!(
      interpret("Options[g] = {x -> 42}; Options[g]").unwrap(),
      "{x -> 42}"
    );
  }

  #[test]
  fn multiple_functions() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> 1}; Options[g] = {b -> 2}; {Options[f], Options[g]}"
      )
      .unwrap(),
      "{{a -> 1}, {b -> 2}}"
    );
  }

  #[test]
  fn builtin_plot_options() {
    // Options[Plot] should return the built-in default options
    let result = interpret("Options[Plot]").unwrap();
    assert!(result.starts_with(
      "{AlignmentPoint -> Center, AspectRatio -> GoldenRatio^(-1)"
    ));
    assert!(result.contains("PlotRange -> {Full, Automatic}"));
    assert!(result.contains("RegionFunction -> (True & )"));
    assert!(result.contains("WorkingPrecision -> MachinePrecision}"));
  }

  #[test]
  fn builtin_plot_specific_option() {
    assert_eq!(
      interpret("Options[Plot, PlotRange]").unwrap(),
      "{PlotRange -> {Full, Automatic}}"
    );
  }

  #[test]
  fn builtin_plot_option_not_found() {
    assert_eq!(interpret("Options[Plot, NonExistentOption]").unwrap(), "{}");
  }
}

mod options_pattern {
  use super::*;

  #[test]
  fn bare_symbolic() {
    assert_eq!(interpret("OptionsPattern[]").unwrap(), "OptionsPattern[]");
  }

  #[test]
  fn bare_with_defaults_symbolic() {
    assert_eq!(
      interpret("OptionsPattern[{a -> 1}]").unwrap(),
      "OptionsPattern[{a -> 1}]"
    );
  }
}

mod option_value {
  use super::*;

  #[test]
  fn basic_with_explicit_option() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> a0, b -> b0}; f[x_, OptionsPattern[]] := {x, OptionValue[a]}; f[7, a -> uuu]"
      )
      .unwrap(),
      "{7, uuu}"
    );
  }

  #[test]
  fn default_option_value() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> a0, b -> b0}; f[x_, OptionsPattern[]] := {x, OptionValue[a]}; f[7]"
      )
      .unwrap(),
      "{7, a0}"
    );
  }

  #[test]
  fn multiple_options() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> a0, b -> b0}; f[x_, OptionsPattern[]] := {x, OptionValue[a], OptionValue[b]}; f[7, b -> bbb]"
      )
      .unwrap(),
      "{7, a0, bbb}"
    );
  }

  #[test]
  fn override_all_options() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> 1, b -> 2}; f[x_, OptionsPattern[]] := {x, OptionValue[a], OptionValue[b]}; f[0, a -> 10, b -> 20]"
      )
      .unwrap(),
      "{0, 10, 20}"
    );
  }

  #[test]
  fn no_options_defined() {
    assert_eq!(
      interpret(
        "g[x_, OptionsPattern[]] := {x, OptionValue[a]}; g[5, a -> 10]"
      )
      .unwrap(),
      "{5, 10}"
    );
  }

  #[test]
  fn options_pattern_no_args() {
    assert_eq!(
      interpret(
        "Options[h] = {x -> 42}; h[OptionsPattern[]] := OptionValue[x]; h[]"
      )
      .unwrap(),
      "42"
    );
  }

  #[test]
  fn inline_defaults_basic() {
    assert_eq!(
      interpret(
        "f[x_, OptionsPattern[{a -> a0, b -> b0}]] := {x, OptionValue[a]}; f[7]"
      )
      .unwrap(),
      "{7, a0}"
    );
  }

  #[test]
  fn inline_defaults_with_override() {
    assert_eq!(
      interpret(
        "f[x_, OptionsPattern[{a -> a0, b -> b0}]] := {x, OptionValue[a]}; f[7, a -> uuu]"
      )
      .unwrap(),
      "{7, uuu}"
    );
  }

  #[test]
  fn inline_defaults_multiple_option_values() {
    assert_eq!(
      interpret(
        "f[x_, OptionsPattern[{a -> a0, b -> b0}]] := {x, OptionValue[a], OptionValue[b]}; f[7, b -> bbb]"
      )
      .unwrap(),
      "{7, a0, bbb}"
    );
  }

  #[test]
  fn inline_defaults_override_options() {
    // OptionsPattern[{...}] inline defaults take priority over Options[f]
    assert_eq!(
      interpret(
        "Options[f] = {a -> fromOptions}; f[x_, OptionsPattern[{a -> fromPattern, b -> b0}]] := {x, OptionValue[a], OptionValue[b]}; f[7]"
      )
      .unwrap(),
      "{7, fromPattern, b0}"
    );
  }

  #[test]
  fn inline_defaults_no_args_pattern() {
    assert_eq!(
      interpret("h[OptionsPattern[{x -> 42}]] := OptionValue[x]; h[]").unwrap(),
      "42"
    );
  }

  #[test]
  fn option_value_two_arg_from_options() {
    assert_eq!(
      interpret(
        "Options[MySetting] = {\"bar\" -> 6}; OptionValue[MySetting, \"bar\"]"
      )
      .unwrap(),
      "6"
    );
  }

  #[test]
  fn option_value_two_arg_not_found_returns_name_string() {
    assert_eq!(
      interpret("OptionValue[MySetting, \"baz\"]").unwrap(),
      "baz"
    );
  }

  #[test]
  fn option_value_two_arg_not_found_returns_name_symbol() {
    assert_eq!(interpret("OptionValue[MySetting, bar]").unwrap(), "bar");
  }

  #[test]
  fn option_value_three_arg_from_explicit_opts() {
    assert_eq!(
      interpret("OptionValue[MySetting, {\"bar\" -> 9}, \"bar\"]").unwrap(),
      "9"
    );
  }
}
