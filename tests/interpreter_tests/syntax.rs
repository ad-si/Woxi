use super::*;

mod errors {
  use super::*;

  #[test]
  fn invalid_input() {
    match interpret("1 + ") {
      Err(woxi::InterpreterError::ParseError(_)) => (),
      _ => panic!("Expected a ParseError"),
    }
  }
}

mod postfix_application {
  use super::*;

  #[test]
  fn postfix_with_identifier() {
    // x // f is equivalent to f[x]
    assert_eq!(interpret("4 // Sqrt").unwrap(), "2");
    assert_eq!(interpret("16 // Sqrt").unwrap(), "4");
  }

  #[test]
  fn postfix_with_list() {
    assert_eq!(interpret("{1, 2, 3} // Length").unwrap(), "3");
    assert_eq!(interpret("{1, 2, 3} // First").unwrap(), "1");
    assert_eq!(interpret("{1, 2, 3} // Last").unwrap(), "3");
  }

  #[test]
  fn postfix_with_function_call() {
    // x // Map[f] is equivalent to Map[f][x] which is Map[f, x]
    assert_eq!(interpret("{1, 4, 9} // Map[Sqrt]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn chained_postfix() {
    // x // f // g is equivalent to g[f[x]]
    assert_eq!(interpret("16 // Sqrt // Sqrt").unwrap(), "2");
  }

  #[test]
  fn postfix_after_replace_all() {
    // (expr /. rules) // f
    assert_eq!(
      interpret("{1, 2, 3} /. x_ /; x > 1 :> 0 // Length").unwrap(),
      "3"
    );
  }

  #[test]
  fn postfix_after_operator_chain() {
    // (1 + 2) // ToString is ToString[Plus[1, 2]]
    assert_eq!(interpret("1 + 2 // ToString").unwrap(), "3");
    // Map operator followed by postfix
    assert_eq!(interpret("Sqrt /@ {1, 4, 9} // Length").unwrap(), "3");
  }

  #[test]
  fn postfix_after_map_with_anonymous_function() {
    // Which[...]& /@ Range[1, 5] // Map[Print] - pattern from fizzbuzz_5
    assert_eq!(interpret("(# + 1)& /@ {1, 2, 3} // Length").unwrap(), "3");
  }

  #[test]
  fn postfix_with_trailing_ampersand() {
    // In Wolfram, & binds tighter than //, so "x // f &" means "(f &)[x]"
    // (f &) is Function[f] which always returns f regardless of argument
    assert_eq!(interpret("5 // Sqrt &").unwrap(), "Sqrt");
    // Chained postfix where only last has &
    assert_eq!(
      interpret("{3, 1, 2} // Sort // Length &").unwrap(),
      "Length"
    );
  }

  #[test]
  fn postfix_ampersand_with_function_call() {
    // "x // f[#, 2] &" means "(f[#, 2] &)[x]" = f[x, 2]
    assert_eq!(interpret("5 // Power[#, 2] &").unwrap(), "25");
  }

  #[test]
  fn nestlist_with_postfix_ampersand() {
    // Original bug: NestList[... // Flatten &, {10}, 10] should give constant {10}
    // because (Flatten &) is Function[Flatten] which always returns Flatten
    assert_eq!(
      interpret(
        "NestList[# /. x_ /; x > 1 :> {x - 1, x - 2} // Flatten &, {10}, 10] // Last // Total"
      )
      .unwrap(),
      "10"
    );
  }
}

mod trailing_semicolon {
  use super::*;

  #[test]
  fn trailing_semicolon_returns_null() {
    // expr; is CompoundExpression[expr, Null] — result is Null
    assert_eq!(interpret("1 + 2;").unwrap(), "Null");
  }

  #[test]
  fn trailing_semicolon_with_print() {
    // Print[1]; should still execute Print, result is Null
    assert_eq!(interpret("Print[1];").unwrap(), "Null");
  }

  #[test]
  fn trailing_semicolon_with_postfix() {
    // {1,2,3} // Map[Print]; should print and return Null
    assert_eq!(interpret("{1,2,3} // Map[Print];").unwrap(), "Null");
  }

  #[test]
  fn no_trailing_semicolon_shows_result() {
    // Without trailing ;, result should be shown
    assert_eq!(interpret("1 + 2").unwrap(), "3");
  }

  #[test]
  fn compound_expression_with_trailing_semicolon() {
    // x = 5; x + 1; should return Null
    assert_eq!(interpret("x = 5; x + 1;").unwrap(), "Null");
  }

  #[test]
  fn compound_expression_without_trailing_semicolon() {
    // x = 5; x + 1 should show the final result
    assert_eq!(interpret("x = 5; x + 1").unwrap(), "6");
  }
}

mod unary_minus_parsing {
  use super::*;

  #[test]
  fn negative_identifier_in_parens() {
    assert_eq!(interpret("(-x)").unwrap(), "-x");
  }

  #[test]
  fn negative_power_in_parens() {
    // (-x^2) should be -(x^2), not (-x)^2
    assert_eq!(interpret("(-x^2)").unwrap(), "-x^2");
  }

  #[test]
  fn negative_expr_plus_constant() {
    assert_eq!(interpret("(-x + 3)").unwrap(), "3 - x");
  }

  #[test]
  fn e_to_negative_x_squared() {
    assert_eq!(interpret("E^(-x^2)").unwrap(), "E^(-x^2)");
  }

  #[test]
  fn negative_power_exponent() {
    assert_eq!(interpret("x^(-2)").unwrap(), "x^(-2)");
  }
}

mod plus_formatting {
  use super::*;

  #[test]
  fn plus_with_negative_term() {
    // a + (-b) should format as a - b
    let result = interpret("Integrate[Sin[x], x]").unwrap();
    assert_eq!(result, "-Cos[x]");
  }

  #[test]
  fn integrate_sum_formatting() {
    // Integrate[x^2 + Sin[x], x] should format nicely
    let result = interpret("Integrate[x^2 + Sin[x], x]").unwrap();
    assert_eq!(result, "x^3/3 - Cos[x]");
  }

  #[test]
  fn plus_term_ordering_polynomial_first() {
    // Polynomial terms should come before transcendental functions
    let result = interpret("x^2 + Sin[x]").unwrap();
    assert_eq!(result, "x^2 + Sin[x]");
  }

  #[test]
  fn plus_term_ordering_alphabetical() {
    // Same priority terms should be alphabetical
    let result = interpret("c + a + b").unwrap();
    assert_eq!(result, "a + b + c");
  }

  #[test]
  fn plus_times_before_identifier() {
    // Times[a, b] should come before c alphabetically (a < c)
    let result = interpret("FullForm[a b + c]").unwrap();
    assert_eq!(result, "Plus[Times[a, b], c]");
  }
}

mod subtraction_without_spaces {
  use super::*;

  #[test]
  fn n_minus_1_in_function_body() {
    // Regression: n-1 (without spaces) was parsed as implicit multiplication n*(-1)
    clear_state();
    assert_eq!(interpret("f[n_] := n-1; f[99]").unwrap(), "98");
  }

  #[test]
  fn subtraction_in_nested_function_call() {
    // Regression: ToString[n-1] was evaluating n-1 as -(n) instead of n minus 1
    clear_state();
    assert_eq!(interpret("f[n_] := ToString[n-1]; f[99]").unwrap(), "98");
  }

  #[test]
  fn subtraction_in_string_join() {
    clear_state();
    assert_eq!(
      interpret(
        r#"f[n_] := ToString[n] <> " minus 1 is " <> ToString[n-1]; f[10]"#
      )
      .unwrap(),
      "10 minus 1 is 9"
    );
  }

  #[test]
  fn implicit_times_then_minus_constant_fraction() {
    // Regression: `2 Pi - Pi/4` must parse as subtraction, not implicit multiplication by -Pi.
    assert_eq!(
      interpret("FullForm[2 Pi - Pi/4]").unwrap(),
      "Plus[Times[-1, Times[Pi, Power[4, -1]]], Times[2, Pi]]"
    );
  }

  #[test]
  fn tostring_input_form() {
    // ToString[expr, InputForm] — strings are quoted, fractions single-line
    assert_eq!(
      interpret(r#"ToString["hello", InputForm]"#).unwrap(),
      r#""hello""#
    );
    assert_eq!(interpret("ToString[1/3, InputForm]").unwrap(), "1/3");
    assert_eq!(
      interpret(r#"ToString[{1, "a", x^2}, InputForm]"#).unwrap(),
      r#"{1, "a", x^2}"#
    );
    assert_eq!(interpret("ToString[x + y, InputForm]").unwrap(), "x + y");
    // Without InputForm, strings are unquoted
    assert_eq!(interpret(r#"ToString["hello"]"#).unwrap(), "hello");
  }

  #[test]
  fn negative_numbers_still_work() {
    assert_eq!(interpret("{-1, -2, -3}").unwrap(), "{-1, -2, -3}");
    assert_eq!(interpret("-1 + 3").unwrap(), "2");
  }
}

mod newline_statements {
  use super::*;

  #[test]
  fn multiline_assignments() {
    clear_state();
    assert_eq!(interpret("x = 5\ny = 10\nx + y").unwrap(), "15");
  }

  #[test]
  fn multiline_with_blank_lines() {
    clear_state();
    assert_eq!(interpret("x = 42\n\nx").unwrap(), "42");
  }

  #[test]
  fn multiline_preserves_continuation() {
    clear_state();
    // A function definition spanning lines should still work
    assert_eq!(interpret("f[x_] :=\n  x + 1\nf[5]").unwrap(), "6");
  }
}

mod full_form {
  use super::*;

  #[test]
  fn full_form_plus() {
    assert_eq!(interpret("FullForm[x + y + z]").unwrap(), "Plus[x, y, z]");
  }

  #[test]
  fn full_form_times() {
    assert_eq!(interpret("FullForm[x y z]").unwrap(), "Times[x, y, z]");
  }

  #[test]
  fn full_form_list() {
    assert_eq!(interpret("FullForm[{1, 2, 3}]").unwrap(), "List[1, 2, 3]");
  }

  #[test]
  fn full_form_power() {
    assert_eq!(interpret("FullForm[x^2]").unwrap(), "Power[x, 2]");
  }

  #[test]
  fn full_form_complex() {
    assert_eq!(
      interpret("FullForm[a b + c]").unwrap(),
      "Plus[Times[a, b], c]"
    );
  }
}

mod construct {
  use super::*;

  #[test]
  fn construct_basic() {
    assert_eq!(interpret("Construct[f, a, b, c]").unwrap(), "f[a, b, c]");
  }

  #[test]
  fn construct_single_arg() {
    assert_eq!(interpret("Construct[f, a]").unwrap(), "f[a]");
  }

  #[test]
  fn construct_with_fold() {
    assert_eq!(
      interpret("Fold[Construct, f, {a, b, c}]").unwrap(),
      "f[a][b][c]"
    );
  }
}

mod rule_display {
  use super::*;

  #[test]
  fn rule_display() {
    assert_eq!(interpret("Rule[a, b]").unwrap(), "a -> b");
  }

  #[test]
  fn rule_arrow_syntax() {
    assert_eq!(interpret("a -> b").unwrap(), "a -> b");
  }

  #[test]
  fn rule_evaluates_arguments() {
    assert_eq!(interpret("Rule[1 + 2, 3 + 4]").unwrap(), "3 -> 7");
  }

  #[test]
  fn rule_head() {
    assert_eq!(interpret("Head[Rule[x, y]]").unwrap(), "Rule");
    assert_eq!(interpret("Head[x -> y]").unwrap(), "Rule");
  }

  #[test]
  fn rule_function_form_equals_arrow() {
    assert_eq!(interpret("Rule[a, b] === (a -> b)").unwrap(), "True");
  }

  #[test]
  fn rule_function_call_in_replace_all() {
    assert_eq!(interpret("f[a, b] /. Rule[a, 1]").unwrap(), "f[1, b]");
  }

  #[test]
  fn rule_sequence_hold() {
    // Rule has SequenceHold: Sequence should not be spliced
    assert_eq!(
      interpret("Rule[Sequence[a, b], c]").unwrap(),
      "Sequence[a, b] -> c"
    );
  }

  #[test]
  fn rule_in_list() {
    assert_eq!(interpret("{a -> 1, b -> 2}").unwrap(), "{a -> 1, b -> 2}");
  }

  #[test]
  fn rule_replace_all_with_list() {
    assert_eq!(interpret("f[x, y] /. {x -> 1, y -> 2}").unwrap(), "f[1, 2]");
  }

  #[test]
  fn rule_with_patterns() {
    assert_eq!(
      interpret("{1, 2, 3} /. x_Integer -> x^2").unwrap(),
      "{1, 4, 9}"
    );
  }

  #[test]
  fn rule_map() {
    assert_eq!(
      interpret("Map[Rule[#, #^2] &, {1, 2, 3}]").unwrap(),
      "{1 -> 1, 2 -> 4, 3 -> 9}"
    );
  }

  #[test]
  fn rule_attributes() {
    assert_eq!(
      interpret("Attributes[Rule]").unwrap(),
      "{Protected, SequenceHold}"
    );
  }

  #[test]
  fn rule_delayed_display() {
    assert_eq!(interpret("RuleDelayed[a, b]").unwrap(), "a :> b");
  }
}

mod blank_function {
  use super::*;

  #[test]
  fn blank_no_args_displays_as_underscore() {
    assert_eq!(interpret("Blank[]").unwrap(), "_");
  }

  #[test]
  fn blank_with_head_displays_as_underscore_head() {
    assert_eq!(interpret("Blank[Integer]").unwrap(), "_Integer");
    assert_eq!(interpret("Blank[String]").unwrap(), "_String");
    assert_eq!(interpret("Blank[List]").unwrap(), "_List");
    assert_eq!(interpret("Blank[Symbol]").unwrap(), "_Symbol");
  }

  #[test]
  fn blank_head_is_blank() {
    assert_eq!(interpret("Head[Blank[]]").unwrap(), "Blank");
    assert_eq!(interpret("Head[Blank[Integer]]").unwrap(), "Blank");
  }

  #[test]
  fn blank_matchq_any() {
    assert_eq!(interpret("MatchQ[42, Blank[]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[\"hello\", Blank[]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[{1, 2}, Blank[]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[f[x], Blank[]]").unwrap(), "True");
  }

  #[test]
  fn blank_matchq_with_head() {
    assert_eq!(interpret("MatchQ[42, Blank[Integer]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[42, Blank[String]]").unwrap(), "False");
    assert_eq!(
      interpret("MatchQ[\"hello\", Blank[String]]").unwrap(),
      "True"
    );
    assert_eq!(interpret("MatchQ[symbol, Blank[Symbol]]").unwrap(), "True");
  }

  #[test]
  fn blank_in_cases() {
    assert_eq!(
      interpret("Cases[{1, \"a\", 2, \"b\"}, Blank[Integer]]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("Cases[{1, \"a\", 2, \"b\"}, Blank[String]]").unwrap(),
      "{a, b}"
    );
  }

  #[test]
  fn blank_in_replace_all() {
    assert_eq!(
      interpret("{1, x, 2.5, \"hello\"} /. Blank[Integer] -> 0").unwrap(),
      "{0, x, 2.5, hello}"
    );
  }
}

mod pattern_function {
  use super::*;

  #[test]
  fn pattern_blank_displays_as_name_underscore() {
    assert_eq!(interpret("Pattern[x, Blank[]]").unwrap(), "x_");
  }

  #[test]
  fn pattern_blank_head_displays_as_name_underscore_head() {
    assert_eq!(
      interpret("Pattern[x, Blank[Integer]]").unwrap(),
      "x_Integer"
    );
    assert_eq!(interpret("Pattern[y, Blank[String]]").unwrap(), "y_String");
  }

  #[test]
  fn pattern_head_is_pattern() {
    assert_eq!(interpret("Head[Pattern[x, Blank[]]]").unwrap(), "Pattern");
  }

  #[test]
  fn pattern_matchq() {
    assert_eq!(
      interpret("MatchQ[42, Pattern[x, Blank[Integer]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[\"hi\", Pattern[x, Blank[Integer]]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn pattern_equals_shorthand() {
    assert_eq!(interpret("Pattern[x, Blank[]] === x_").unwrap(), "True");
  }

  #[test]
  fn pattern_in_replace_all() {
    assert_eq!(
      interpret("f[a, b] /. Pattern[x, Blank[]] -> x^2").unwrap(),
      "f[a, b]^2"
    );
  }
}

mod none_symbol {
  use super::*;

  #[test]
  fn none_evaluates_to_itself() {
    assert_eq!(interpret("None").unwrap(), "None");
  }

  #[test]
  fn none_head_is_symbol() {
    assert_eq!(interpret("Head[None]").unwrap(), "Symbol");
  }

  #[test]
  fn none_is_protected() {
    assert_eq!(interpret("Attributes[None]").unwrap(), "{Protected}");
  }
}

mod rule_delayed {
  use super::*;

  #[test]
  fn rule_delayed_display() {
    assert_eq!(interpret("x :> x^2").unwrap(), "x :> x^2");
  }

  #[test]
  fn rule_delayed_function_call_form() {
    assert_eq!(interpret("RuleDelayed[x, x^2]").unwrap(), "x :> x^2");
  }

  #[test]
  fn rule_delayed_head() {
    assert_eq!(interpret("Head[x :> x^2]").unwrap(), "RuleDelayed");
    assert_eq!(
      interpret("Head[RuleDelayed[x, x^2]]").unwrap(),
      "RuleDelayed"
    );
  }

  #[test]
  fn rule_delayed_attributes() {
    assert_eq!(
      interpret("Attributes[RuleDelayed]").unwrap(),
      "{HoldRest, Protected, SequenceHold}"
    );
  }

  #[test]
  fn rule_delayed_with_replace_all() {
    assert_eq!(
      interpret("{1, 2, 3} /. x_Integer :> x^2").unwrap(),
      "{1, 4, 9}"
    );
  }

  #[test]
  fn rule_delayed_function_call_with_replace_all() {
    assert_eq!(
      interpret("{1, 2, 3} /. RuleDelayed[x_Integer, x^2]").unwrap(),
      "{1, 4, 9}"
    );
  }

  #[test]
  fn rule_delayed_holds_rhs() {
    // RuleDelayed should not evaluate the RHS prematurely
    assert_eq!(interpret("RuleDelayed[x, 1 + 1]").unwrap(), "x :> 1 + 1");
  }
}

mod false_symbol {
  use super::*;

  #[test]
  fn false_evaluates_to_itself() {
    assert_eq!(interpret("False").unwrap(), "False");
  }

  #[test]
  fn false_head_is_symbol() {
    assert_eq!(interpret("Head[False]").unwrap(), "Symbol");
  }

  #[test]
  fn false_is_protected() {
    assert_eq!(interpret("Attributes[False]").unwrap(), "{Protected}");
  }

  #[test]
  fn not_false_is_true() {
    assert_eq!(interpret("Not[False]").unwrap(), "True");
  }

  #[test]
  fn false_in_list() {
    assert_eq!(
      interpret("{False, True, False}").unwrap(),
      "{False, True, False}"
    );
  }
}

mod plot_range_symbol {
  use super::*;

  #[test]
  fn plot_range_evaluates_to_itself() {
    assert_eq!(interpret("PlotRange").unwrap(), "PlotRange");
  }

  #[test]
  fn plot_range_head_is_symbol() {
    assert_eq!(interpret("Head[PlotRange]").unwrap(), "Symbol");
  }

  #[test]
  fn plot_range_attributes() {
    assert_eq!(
      interpret("Attributes[PlotRange]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod all_symbol {
  use super::*;

  #[test]
  fn all_evaluates_to_itself() {
    assert_eq!(interpret("All").unwrap(), "All");
  }

  #[test]
  fn all_head_is_symbol() {
    assert_eq!(interpret("Head[All]").unwrap(), "Symbol");
  }

  #[test]
  fn all_is_protected() {
    assert_eq!(interpret("Attributes[All]").unwrap(), "{Protected}");
  }
}

mod subscript_function {
  use super::*;

  #[test]
  fn subscript_basic() {
    assert_eq!(interpret("Subscript[x, 1]").unwrap(), "Subscript[x, 1]");
  }

  #[test]
  fn subscript_head() {
    assert_eq!(interpret("Head[Subscript[x, 1]]").unwrap(), "Subscript");
  }

  #[test]
  fn subscript_attributes() {
    assert_eq!(interpret("Attributes[Subscript]").unwrap(), "{NHoldRest}");
  }

  #[test]
  fn subscript_multi_index() {
    assert_eq!(
      interpret("Subscript[x, 1, 2]").unwrap(),
      "Subscript[x, 1, 2]"
    );
  }

  #[test]
  fn subscript_fullform() {
    assert_eq!(
      interpret("FullForm[Subscript[x, 1]]").unwrap(),
      "Subscript[x, 1]"
    );
  }
}

mod automatic_symbol {
  use super::*;

  #[test]
  fn automatic_evaluates_to_itself() {
    assert_eq!(interpret("Automatic").unwrap(), "Automatic");
  }

  #[test]
  fn automatic_head_is_symbol() {
    assert_eq!(interpret("Head[Automatic]").unwrap(), "Symbol");
  }

  #[test]
  fn automatic_is_protected() {
    assert_eq!(interpret("Attributes[Automatic]").unwrap(), "{Protected}");
  }

  #[test]
  fn automatic_in_list() {
    assert_eq!(interpret("{Automatic, None}").unwrap(), "{Automatic, None}");
  }
}

mod true_symbol {
  use super::*;

  #[test]
  fn true_evaluates_to_itself() {
    assert_eq!(interpret("True").unwrap(), "True");
  }

  #[test]
  fn true_head_is_symbol() {
    assert_eq!(interpret("Head[True]").unwrap(), "Symbol");
  }

  #[test]
  fn true_is_protected() {
    assert_eq!(interpret("Attributes[True]").unwrap(), "{Protected}");
  }

  #[test]
  fn not_true_is_false() {
    assert_eq!(interpret("Not[True]").unwrap(), "False");
  }

  #[test]
  fn true_in_list() {
    assert_eq!(
      interpret("{True, False, True}").unwrap(),
      "{True, False, True}"
    );
  }
}

mod slot_function {
  use super::*;

  #[test]
  fn slot_displays_as_hash() {
    assert_eq!(interpret("Slot[1]").unwrap(), "#1");
    assert_eq!(interpret("Slot[2]").unwrap(), "#2");
  }

  #[test]
  fn slot_head() {
    assert_eq!(interpret("Head[Slot[1]]").unwrap(), "Slot");
    assert_eq!(interpret("Head[#]").unwrap(), "Slot");
  }

  #[test]
  fn slot_equals_hash() {
    assert_eq!(interpret("Slot[1] === #").unwrap(), "True");
  }

  #[test]
  fn slot_in_function() {
    assert_eq!(interpret("Function[Slot[1]^2][5]").unwrap(), "25");
  }

  #[test]
  fn slot_multi_arg_function() {
    assert_eq!(interpret("Function[Slot[1] + Slot[2]][3, 4]").unwrap(), "7");
  }

  #[test]
  fn slot_hash_syntax() {
    assert_eq!(interpret("#^2 &[3]").unwrap(), "9");
  }
}

mod set_delayed {
  use super::*;

  #[test]
  fn function_form_defines_function() {
    assert_eq!(interpret("SetDelayed[h[x_], x^3]; h[4]").unwrap(), "64");
  }

  #[test]
  fn colon_equals_syntax() {
    assert_eq!(interpret("f[x_] := x^2; f[3]").unwrap(), "9");
  }

  #[test]
  fn function_form_simple_variable() {
    assert_eq!(
      interpret("Clear[myvar]; SetDelayed[myvar, 42]; myvar").unwrap(),
      "42"
    );
  }

  #[test]
  fn delayed_evaluation() {
    // SetDelayed evaluates the RHS each time
    assert_eq!(
      interpret("n = 0; f[x_] := (n = n + 1; x + n); {f[10], f[10]}").unwrap(),
      "{11, 12}"
    );
  }
}

mod compound_expression {
  use super::*;

  #[test]
  fn function_form_returns_last() {
    assert_eq!(interpret("CompoundExpression[1, 2, 3]").unwrap(), "3");
  }

  #[test]
  fn function_form_single_arg() {
    assert_eq!(interpret("CompoundExpression[42]").unwrap(), "42");
  }

  #[test]
  fn function_form_no_args_returns_null() {
    assert_eq!(interpret("CompoundExpression[]").unwrap(), "Null");
  }

  #[test]
  fn function_form_with_assignments() {
    assert_eq!(
      interpret("CompoundExpression[a = 2, b = 3, a + b]").unwrap(),
      "5"
    );
  }

  #[test]
  fn semicolon_syntax() {
    assert_eq!(interpret("a = 2; b = 3; a + b").unwrap(), "5");
  }

  #[test]
  fn function_form_with_side_effects() {
    // Side effects execute sequentially
    assert_eq!(
      interpret("CompoundExpression[x = 10, x = x + 1, x]").unwrap(),
      "11"
    );
  }
}

mod hold_form {
  use super::*;

  #[test]
  fn hold_form_unevaluated() {
    assert_eq!(interpret("HoldForm[1 + 1]").unwrap(), "HoldForm[1 + 1]");
  }
}

mod numerator {
  use super::*;

  #[test]
  fn rational() {
    assert_eq!(interpret("Numerator[3/4]").unwrap(), "3");
  }

  #[test]
  fn integer() {
    assert_eq!(interpret("Numerator[5]").unwrap(), "5");
  }

  #[test]
  fn negative_rational() {
    assert_eq!(interpret("Numerator[-3/4]").unwrap(), "-3");
  }

  #[test]
  fn whole_number_rational() {
    assert_eq!(interpret("Numerator[6/3]").unwrap(), "2");
  }
}

mod denominator {
  use super::*;

  #[test]
  fn rational() {
    assert_eq!(interpret("Denominator[3/4]").unwrap(), "4");
  }

  #[test]
  fn integer() {
    assert_eq!(interpret("Denominator[5]").unwrap(), "1");
  }

  #[test]
  fn negative_rational() {
    assert_eq!(interpret("Denominator[-3/4]").unwrap(), "4");
  }

  #[test]
  fn reduced_form() {
    assert_eq!(interpret("Denominator[6/4]").unwrap(), "2");
  }
}

mod symbolic_ordering {
  use super::*;

  #[test]
  fn numbers_before_symbols() {
    assert_eq!(interpret("cow + 5").unwrap(), "5 + cow");
  }

  #[test]
  fn numeric_terms_combined() {
    assert_eq!(interpret("cow + 5 + 10").unwrap(), "15 + cow");
  }

  #[test]
  fn multiple_symbolic_terms_sorted() {
    assert_eq!(interpret("z + a + 3").unwrap(), "3 + a + z");
  }
}

mod power_formatting {
  use super::*;

  #[test]
  fn power_exponent_with_plus_gets_parens() {
    // D[x^n, x] = n*x^(-1 + n)
    assert_eq!(interpret("D[x^n, x]").unwrap(), "n*x^(-1 + n)");
  }

  #[test]
  fn power_plus_base_gets_parens() {
    assert_eq!(interpret("(1 + x)^(-1)").unwrap(), "(1 + x)^(-1)");
    assert_eq!(interpret("(-1 + x)^(-1)").unwrap(), "(-1 + x)^(-1)");
    assert_eq!(interpret("(2 + x)^3").unwrap(), "(2 + x)^3");
  }
}

mod table_form {
  use super::*;

  #[test]
  fn returns_unevaluated() {
    // TableForm is a display wrapper — returns unevaluated in text mode
    // (matches wolframscript behavior)
    assert_eq!(
      interpret("TableForm[{1, 2, 3}]").unwrap(),
      "TableForm[{1, 2, 3}]"
    );
    assert_eq!(
      interpret("TableForm[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "TableForm[{{1, 2, 3}, {4, 5, 6}}]"
    );
    assert_eq!(interpret("TableForm[5]").unwrap(), "TableForm[5]");
    assert_eq!(interpret("TableForm[x]").unwrap(), "TableForm[x]");
  }

  #[test]
  fn evaluates_arguments() {
    // Arguments are evaluated, but TableForm wrapper remains
    assert_eq!(
      interpret("TableForm[Table[i^2, {i, 3}]]").unwrap(),
      "TableForm[{1, 4, 9}]"
    );
    assert_eq!(
      interpret("TableForm[{1 + 1, 2 + 2}]").unwrap(),
      "TableForm[{2, 4}]"
    );
  }

  #[test]
  fn postfix_notation() {
    assert_eq!(
      interpret("{1, 2, 3} // TableForm").unwrap(),
      "TableForm[{1, 2, 3}]"
    );
  }
}

mod row {
  use super::*;

  #[test]
  fn no_separator() {
    // Row[{exprs...}] concatenates elements with no separator
    assert_eq!(interpret("Row[{1, 2, 3}]").unwrap(), "123");
    assert_eq!(interpret("Row[{a, b, c}]").unwrap(), "abc");
  }

  #[test]
  fn with_separator() {
    // Row[{exprs...}, sep] joins elements with sep between them
    assert_eq!(interpret(r#"Row[{1, 2, 3}, ", "]"#).unwrap(), "1, 2, 3");
    assert_eq!(interpret(r#"Row[{a, b, c}, "+"]"#).unwrap(), "a+b+c");
    assert_eq!(interpret(r#"Row[{x, y, z}, " | "]"#).unwrap(), "x | y | z");
  }

  #[test]
  fn evaluates_arguments() {
    // Arguments inside the list are evaluated before display
    assert_eq!(interpret("Row[{1 + 1, 2 + 2}]").unwrap(), "24");
    assert_eq!(interpret(r#"Row[{1 + 1, 2 + 2}, " "]"#).unwrap(), "2 4");
  }

  #[test]
  fn single_element() {
    assert_eq!(interpret("Row[{42}]").unwrap(), "42");
    assert_eq!(interpret(r#"Row[{42}, ", "]"#).unwrap(), "42");
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("Row[{}]").unwrap(), "");
    assert_eq!(interpret(r#"Row[{}, ", "]"#).unwrap(), "");
  }

  #[test]
  fn with_strings() {
    assert_eq!(
      interpret(r#"Row[{"Hello", " ", "World"}]"#).unwrap(),
      "Hello World"
    );
  }

  #[test]
  fn postfix_notation() {
    assert_eq!(interpret("{1, 2, 3} // Row").unwrap(), "123");
  }

  #[test]
  fn non_list_arg_stays_symbolic() {
    // Row[x] where x is not a list stays unevaluated
    assert_eq!(interpret("Row[x]").unwrap(), "Row[x]");
    assert_eq!(interpret("Row[5]").unwrap(), "Row[5]");
  }
}

mod sequence {
  use super::*;

  #[test]
  fn basic_flattening() {
    assert_eq!(interpret("f[Sequence[a, b]]").unwrap(), "f[a, b]");
  }

  #[test]
  fn in_middle_of_args() {
    assert_eq!(
      interpret("f[x, Sequence[a, b], y]").unwrap(),
      "f[x, a, b, y]"
    );
  }

  #[test]
  fn in_list() {
    assert_eq!(interpret("{a, Sequence[b, c], d}").unwrap(), "{a, b, c, d}");
  }

  #[test]
  fn empty_sequence() {
    assert_eq!(interpret("f[a, Sequence[], b]").unwrap(), "f[a, b]");
  }

  #[test]
  fn hold_flattens_sequence() {
    assert_eq!(
      interpret("Hold[a, Sequence[b, c], d]").unwrap(),
      "Hold[a, b, c, d]"
    );
  }
}

mod hold {
  use super::*;

  #[test]
  fn hold_prevents_evaluation() {
    assert_eq!(interpret("Hold[1 + 2]").unwrap(), "Hold[1 + 2]");
  }

  #[test]
  fn hold_form_prevents_evaluation() {
    assert_eq!(
      interpret("HoldForm[1 + 2 + 3]").unwrap(),
      "HoldForm[1 + 2 + 3]"
    );
  }

  #[test]
  fn hold_complete_prevents_evaluation() {
    assert_eq!(
      interpret("HoldComplete[Evaluate[1 + 2]]").unwrap(),
      "HoldComplete[Evaluate[1 + 2]]"
    );
  }

  #[test]
  fn release_hold_basic() {
    assert_eq!(interpret("ReleaseHold[Hold[1 + 2]]").unwrap(), "3");
  }

  #[test]
  fn release_hold_form() {
    assert_eq!(interpret("ReleaseHold[HoldForm[1 + 2]]").unwrap(), "3");
  }

  #[test]
  fn release_hold_non_hold() {
    assert_eq!(interpret("ReleaseHold[5]").unwrap(), "5");
    assert_eq!(interpret("ReleaseHold[x]").unwrap(), "x");
  }
}

mod deeply_nested_lists {
  use super::*;

  #[test]
  fn nested_list_in_function_in_list() {
    // Regression test: deeply nested lists inside function calls inside lists
    // previously caused exponential backtracking in the PEG parser
    assert_eq!(interpret("{f[{1, {{{1}}}}]}").unwrap(), "{f[{1, {{{1}}}}]}");
  }

  #[test]
  fn deeply_nested_braces() {
    assert_eq!(
      interpret("{f[{1, {{{{{{1}}}}}}}]}").unwrap(),
      "{f[{1, {{{{{{1}}}}}}}]}"
    );
  }

  #[test]
  fn nested_lists_still_evaluate() {
    // Ensure lists still evaluate correctly after grammar optimization
    assert_eq!(interpret("{1 + 1, {2 + 2}}").unwrap(), "{2, {4}}");
    assert_eq!(interpret("{{1, 2}, {3, 4}}[[1]]").unwrap(), "{1, 2}");
    assert_eq!(interpret("{#, #^2}&[3]").unwrap(), "{3, 9}");
  }

  #[test]
  fn replacement_rules_in_lists() {
    // Ensure replacement rules still work in lists after grammar optimization
    assert_eq!(interpret("{x -> 1, y -> 2}").unwrap(), "{x -> 1, y -> 2}");
    assert_eq!(interpret("x /. {x -> 5}").unwrap(), "5");
  }
}

mod grid {
  use super::*;

  #[test]
  fn basic_2x2() {
    clear_state();
    let svg =
      interpret("ExportString[Grid[{{a, b}, {c, d}}], \"SVG\"]").unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">d</text>"));
  }

  #[test]
  fn one_dimensional_list() {
    clear_state();
    let svg = interpret("ExportString[Grid[{a, b, c}], \"SVG\"]").unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">c</text>"));
  }

  #[test]
  fn arguments_are_evaluated() {
    clear_state();
    let svg =
      interpret("ExportString[Grid[{{1+1, 2+2}, {3+3, 4+4}}], \"SVG\"]")
        .unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">2</text>"));
    assert!(svg.contains(">8</text>"));
  }

  #[test]
  fn with_options() {
    clear_state();
    let svg = interpret(
      "ExportString[Grid[{{1, 2}, {3, 4}}, Alignment -> Center], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
  }

  #[test]
  fn single_element() {
    clear_state();
    let svg = interpret("ExportString[Grid[{{x}}], \"SVG\"]").unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">x</text>"));
  }

  #[test]
  fn postfix_form() {
    clear_state();
    let svg =
      interpret("ExportString[{{a, b}, {c, d}} // Grid, \"SVG\"]").unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">a</text>"));
  }

  #[test]
  fn frame_all() {
    clear_state();
    let svg = interpret(
      "ExportString[Grid[{{a, b, c}, {x, y^2, z^3}}, Frame -> All], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<line"), "Frame -> All should produce lines");
  }

  #[test]
  fn frame_all_with_other_options() {
    clear_state();
    let svg = interpret(
      "ExportString[Grid[{{1, 2}, {3, 4}}, Alignment -> Center, Frame -> All], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<line"), "Frame -> All should produce lines");
  }
}
