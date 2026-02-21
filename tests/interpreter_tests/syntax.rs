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

mod tree_form {
  use super::*;

  #[test]
  fn tree_form_simple() {
    assert_eq!(interpret("TreeForm[f[x, y]]").unwrap(), "TreeForm[f[x, y]]");
  }

  #[test]
  fn tree_form_expression() {
    assert_eq!(
      interpret("TreeForm[a + b^2 + c^3 + d]").unwrap(),
      "TreeForm[a + b^2 + c^3 + d]"
    );
  }

  #[test]
  fn tree_form_evaluates_argument() {
    assert_eq!(interpret("TreeForm[1 + 2]").unwrap(), "TreeForm[3]");
  }

  #[test]
  fn tree_form_with_depth() {
    assert_eq!(interpret("TreeForm[f[x], 2]").unwrap(), "TreeForm[f[x], 2]");
  }

  #[test]
  fn tree_form_no_args() {
    assert_eq!(interpret("TreeForm[]").unwrap(), "TreeForm[]");
  }

  #[test]
  fn tree_form_head() {
    assert_eq!(interpret("Head[TreeForm[f[x]]]").unwrap(), "TreeForm");
  }

  #[test]
  fn tree_form_in_list() {
    assert_eq!(
      interpret("{TreeForm[f[x]], TreeForm[g[y]]}").unwrap(),
      "{TreeForm[f[x]], TreeForm[g[y]]}"
    );
  }
}

mod digit_block {
  use super::*;

  #[test]
  fn digit_block_standalone() {
    assert_eq!(interpret("DigitBlock").unwrap(), "DigitBlock");
  }

  #[test]
  fn digit_block_head() {
    assert_eq!(interpret("Head[DigitBlock]").unwrap(), "Symbol");
  }

  #[test]
  fn digit_block_as_option() {
    assert_eq!(
      interpret("NumberForm[123, DigitBlock -> 3]").unwrap(),
      "NumberForm[123, DigitBlock -> 3]"
    );
  }
}

mod cubics {
  use super::*;

  #[test]
  fn cubics_standalone() {
    assert_eq!(interpret("Cubics").unwrap(), "Cubics");
  }

  #[test]
  fn cubics_head() {
    assert_eq!(interpret("Head[Cubics]").unwrap(), "Symbol");
  }

  #[test]
  fn cubics_as_option() {
    // Cubics used as an option value in a rule
    assert_eq!(interpret("Cubics -> True").unwrap(), "Cubics -> True");
  }
}

mod page_width {
  use super::*;

  #[test]
  fn page_width_standalone() {
    assert_eq!(interpret("PageWidth").unwrap(), "PageWidth");
  }

  #[test]
  fn page_width_head() {
    assert_eq!(interpret("Head[PageWidth]").unwrap(), "Symbol");
  }

  #[test]
  fn page_width_as_option() {
    assert_eq!(interpret("PageWidth -> 80").unwrap(), "PageWidth -> 80");
  }
}

mod constant {
  use super::*;

  #[test]
  fn constant_standalone() {
    assert_eq!(interpret("Constant").unwrap(), "Constant");
  }

  #[test]
  fn constant_head() {
    assert_eq!(interpret("Head[Constant]").unwrap(), "Symbol");
  }

  #[test]
  fn constant_as_attribute() {
    // Pi has the Constant attribute
    assert_eq!(
      interpret("MemberQ[Attributes[Pi], Constant]").unwrap(),
      "True"
    );
  }
}

mod catalan_constant {
  use super::*;

  #[test]
  fn catalan_standalone() {
    assert_eq!(interpret("Catalan").unwrap(), "Catalan");
  }

  #[test]
  fn catalan_numeric() {
    let result: f64 = interpret("N[Catalan]").unwrap().parse().unwrap();
    assert!((result - 0.915965594177219).abs() < 1e-10);
  }

  #[test]
  fn catalan_head() {
    assert_eq!(interpret("Head[Catalan]").unwrap(), "Symbol");
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

mod plot_style_symbol {
  use super::*;

  #[test]
  fn plot_style_evaluates_to_itself() {
    assert_eq!(interpret("PlotStyle").unwrap(), "PlotStyle");
  }

  #[test]
  fn plot_style_head_is_symbol() {
    assert_eq!(interpret("Head[PlotStyle]").unwrap(), "Symbol");
  }

  #[test]
  fn plot_style_is_protected() {
    assert_eq!(interpret("Attributes[PlotStyle]").unwrap(), "{Protected}");
  }
}

mod condition_function {
  use super::*;

  #[test]
  fn condition_display() {
    assert_eq!(interpret("Condition[x_, x > 0]").unwrap(), "x_ /; x > 0");
  }

  #[test]
  fn condition_head() {
    assert_eq!(
      interpret("Head[Condition[x_, x > 0]]").unwrap(),
      "Condition"
    );
  }

  #[test]
  fn condition_attributes() {
    assert_eq!(
      interpret("Attributes[Condition]").unwrap(),
      "{HoldAll, Protected}"
    );
  }

  #[test]
  fn condition_holds_args() {
    // Condition should not evaluate its arguments
    assert_eq!(
      interpret("Condition[x_, 1 + 1 == 2]").unwrap(),
      "x_ /; 1 + 1 == 2"
    );
  }
}

mod axes_label_symbol {
  use super::*;

  #[test]
  fn axes_label_evaluates_to_itself() {
    assert_eq!(interpret("AxesLabel").unwrap(), "AxesLabel");
  }

  #[test]
  fn axes_label_head_is_symbol() {
    assert_eq!(interpret("Head[AxesLabel]").unwrap(), "Symbol");
  }

  #[test]
  fn axes_label_is_protected() {
    assert_eq!(interpret("Attributes[AxesLabel]").unwrap(), "{Protected}");
  }
}

mod show_function {
  use super::*;

  #[test]
  fn show_evaluates_to_itself() {
    assert_eq!(interpret("Show").unwrap(), "Show");
  }

  #[test]
  fn show_head_is_symbol() {
    assert_eq!(interpret("Head[Show]").unwrap(), "Symbol");
  }

  #[test]
  fn show_attributes() {
    assert_eq!(
      interpret("Attributes[Show]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  #[test]
  fn show_head_with_arg() {
    assert_eq!(interpret("Head[Show[1]]").unwrap(), "Show");
  }
}

mod pattern_test_function {
  use super::*;

  #[test]
  fn pattern_test_display_named() {
    assert_eq!(
      interpret("PatternTest[x_, IntegerQ]").unwrap(),
      "(x_)?IntegerQ"
    );
  }

  #[test]
  fn pattern_test_display_blank() {
    assert_eq!(
      interpret("PatternTest[Blank[], IntegerQ]").unwrap(),
      "_?IntegerQ"
    );
  }

  #[test]
  fn pattern_test_head() {
    assert_eq!(
      interpret("Head[PatternTest[x_, IntegerQ]]").unwrap(),
      "PatternTest"
    );
  }

  #[test]
  fn pattern_test_attributes() {
    assert_eq!(
      interpret("Attributes[PatternTest]").unwrap(),
      "{HoldRest, Protected}"
    );
  }

  #[test]
  fn pattern_test_with_cases() {
    assert_eq!(
      interpret("Cases[{1, 2.5, 3, \"a\"}, _?IntegerQ]").unwrap(),
      "{1, 3}"
    );
  }
}

mod blank_null_sequence {
  use super::*;

  #[test]
  fn blank_null_sequence_display() {
    assert_eq!(interpret("BlankNullSequence[]").unwrap(), "___");
  }

  #[test]
  fn blank_null_sequence_with_head() {
    assert_eq!(
      interpret("BlankNullSequence[Integer]").unwrap(),
      "___Integer"
    );
  }

  #[test]
  fn blank_null_sequence_head() {
    assert_eq!(
      interpret("Head[BlankNullSequence[]]").unwrap(),
      "BlankNullSequence"
    );
  }

  #[test]
  fn blank_null_sequence_is_protected() {
    assert_eq!(
      interpret("Attributes[BlankNullSequence]").unwrap(),
      "{Protected}"
    );
  }

  #[test]
  fn blank_null_sequence_syntax() {
    assert_eq!(interpret("___").unwrap(), "___");
  }
}

mod plot_label_symbol {
  use super::*;

  #[test]
  fn plot_label_evaluates_to_itself() {
    assert_eq!(interpret("PlotLabel").unwrap(), "PlotLabel");
  }

  #[test]
  fn plot_label_is_protected() {
    assert_eq!(interpret("Attributes[PlotLabel]").unwrap(), "{Protected}");
  }
}

mod axes_symbol {
  use super::*;

  #[test]
  fn axes_evaluates_to_itself() {
    assert_eq!(interpret("Axes").unwrap(), "Axes");
  }

  #[test]
  fn axes_is_protected() {
    assert_eq!(interpret("Attributes[Axes]").unwrap(), "{Protected}");
  }
}

mod aspect_ratio_symbol {
  use super::*;

  #[test]
  fn aspect_ratio_evaluates_to_itself() {
    assert_eq!(interpret("AspectRatio").unwrap(), "AspectRatio");
  }

  #[test]
  fn aspect_ratio_is_protected() {
    assert_eq!(interpret("Attributes[AspectRatio]").unwrap(), "{Protected}");
  }
}

mod message_name_function {
  use super::*;

  #[test]
  fn message_name_basic() {
    assert_eq!(
      interpret("MessageName[f, \"usage\"]").unwrap(),
      "MessageName[f, usage]"
    );
  }

  #[test]
  fn message_name_head() {
    assert_eq!(
      interpret("Head[MessageName[f, \"usage\"]]").unwrap(),
      "MessageName"
    );
  }

  #[test]
  fn message_name_attributes() {
    assert_eq!(
      interpret("Attributes[MessageName]").unwrap(),
      "{HoldFirst, Protected, ReadProtected}"
    );
  }
}

mod plot3d_function {
  use super::*;

  #[test]
  fn plot3d_evaluates_to_itself() {
    assert_eq!(interpret("Plot3D").unwrap(), "Plot3D");
  }

  #[test]
  fn plot3d_attributes() {
    assert_eq!(
      interpret("Attributes[Plot3D]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod increment_function {
  use super::*;

  #[test]
  fn increment_postfix_returns_old_value() {
    assert_eq!(interpret("x = 5; x++").unwrap(), "5");
  }

  #[test]
  fn increment_postfix_modifies_variable() {
    assert_eq!(interpret("x = 5; x++; x").unwrap(), "6");
  }

  #[test]
  fn increment_function_call() {
    assert_eq!(interpret("x = 10; Increment[x]").unwrap(), "10");
    assert_eq!(interpret("x = 10; Increment[x]; x").unwrap(), "11");
  }

  #[test]
  fn increment_attributes() {
    assert_eq!(
      interpret("Attributes[Increment]").unwrap(),
      "{HoldFirst, Protected, ReadProtected}"
    );
  }

  #[test]
  fn increment_multiple_times() {
    assert_eq!(interpret("x = 0; x++; x++; x++; x").unwrap(), "3");
  }
}

mod decrement_function {
  use super::*;

  #[test]
  fn decrement_postfix_returns_old_value() {
    assert_eq!(interpret("x = 5; x--").unwrap(), "5");
  }

  #[test]
  fn decrement_postfix_modifies_variable() {
    assert_eq!(interpret("x = 5; x--; x").unwrap(), "4");
  }

  #[test]
  fn decrement_function_call() {
    assert_eq!(interpret("x = 10; Decrement[x]").unwrap(), "10");
    assert_eq!(interpret("x = 10; Decrement[x]; x").unwrap(), "9");
  }

  #[test]
  fn decrement_attributes() {
    assert_eq!(
      interpret("Attributes[Decrement]").unwrap(),
      "{HoldFirst, Protected, ReadProtected}"
    );
  }
}

mod pre_increment_function {
  use super::*;

  #[test]
  fn pre_increment_returns_new_value() {
    assert_eq!(interpret("x = 5; ++x").unwrap(), "6");
  }

  #[test]
  fn pre_increment_modifies_variable() {
    assert_eq!(interpret("x = 5; ++x; x").unwrap(), "6");
  }

  #[test]
  fn pre_increment_function_call() {
    assert_eq!(interpret("x = 10; PreIncrement[x]").unwrap(), "11");
    assert_eq!(interpret("x = 10; PreIncrement[x]; x").unwrap(), "11");
  }

  #[test]
  fn pre_increment_attributes() {
    assert_eq!(
      interpret("Attributes[PreIncrement]").unwrap(),
      "{HoldFirst, Protected, ReadProtected}"
    );
  }
}

mod pre_decrement_function {
  use super::*;

  #[test]
  fn pre_decrement_returns_new_value() {
    assert_eq!(interpret("x = 5; --x").unwrap(), "4");
  }

  #[test]
  fn pre_decrement_modifies_variable() {
    assert_eq!(interpret("x = 5; --x; x").unwrap(), "4");
  }

  #[test]
  fn pre_decrement_function_call() {
    assert_eq!(interpret("x = 10; PreDecrement[x]").unwrap(), "9");
    assert_eq!(interpret("x = 10; PreDecrement[x]; x").unwrap(), "9");
  }

  #[test]
  fn pre_decrement_attributes() {
    assert_eq!(
      interpret("Attributes[PreDecrement]").unwrap(),
      "{HoldFirst, Protected, ReadProtected}"
    );
  }
}

mod max_iterations_symbol {
  use super::*;

  #[test]
  fn max_iterations_attributes() {
    assert_eq!(
      interpret("Attributes[MaxIterations]").unwrap(),
      "{Protected}"
    );
  }
}

mod accuracy_goal_symbol {
  use super::*;

  #[test]
  fn accuracy_goal_attributes() {
    assert_eq!(
      interpret("Attributes[AccuracyGoal]").unwrap(),
      "{Protected}"
    );
  }
}

mod general_symbol {
  use super::*;

  #[test]
  fn general_attributes() {
    assert_eq!(interpret("Attributes[General]").unwrap(), "{Protected}");
  }
}

mod default_symbol {
  use super::*;

  #[test]
  fn default_attributes() {
    assert_eq!(interpret("Attributes[Default]").unwrap(), "{Protected}");
  }
}

mod number_symbol {
  use super::*;

  #[test]
  fn number_attributes() {
    assert_eq!(interpret("Attributes[Number]").unwrap(), "{Protected}");
  }
}

mod flat_symbol {
  use super::*;

  #[test]
  fn flat_attributes() {
    assert_eq!(interpret("Attributes[Flat]").unwrap(), "{Protected}");
  }
}

mod read_protected_symbol {
  use super::*;

  #[test]
  fn read_protected_attributes() {
    assert_eq!(
      interpret("Attributes[ReadProtected]").unwrap(),
      "{Protected}"
    );
  }
}

mod protected_symbol {
  use super::*;

  #[test]
  fn protected_attributes() {
    assert_eq!(interpret("Attributes[Protected]").unwrap(), "{Protected}");
  }
}

mod hold_rest_symbol {
  use super::*;

  #[test]
  fn hold_rest_attributes() {
    assert_eq!(interpret("Attributes[HoldRest]").unwrap(), "{Protected}");
  }
}

mod off_function {
  use super::*;

  #[test]
  fn off_returns_null() {
    assert_eq!(interpret("Off[f]").unwrap(), "Null");
  }

  #[test]
  fn off_attributes() {
    assert_eq!(
      interpret("Attributes[Off]").unwrap(),
      "{HoldAll, Protected}"
    );
  }
}

mod remove_function {
  use super::*;

  #[test]
  fn remove_returns_null() {
    assert_eq!(interpret("Remove[x]").unwrap(), "Null");
  }

  #[test]
  fn remove_attributes() {
    assert_eq!(
      interpret("Attributes[Remove]").unwrap(),
      "{HoldAll, Locked, Protected}"
    );
  }
}

mod set_options_function {
  use super::*;

  #[test]
  fn set_options_returns_null() {
    assert_eq!(interpret("SetOptions[f, a -> 1]").unwrap(), "Null");
  }

  #[test]
  fn set_options_attributes() {
    assert_eq!(interpret("Attributes[SetOptions]").unwrap(), "{Protected}");
  }
}

mod clear_attributes_function {
  use super::*;

  #[test]
  fn clear_attributes_works() {
    assert_eq!(
      interpret("SetAttributes[g, Listable]; ClearAttributes[g, Listable]; Attributes[g]")
        .unwrap(),
      "{}"
    );
  }

  #[test]
  fn clear_attributes_attributes() {
    assert_eq!(
      interpret("Attributes[ClearAttributes]").unwrap(),
      "{HoldFirst, Protected}"
    );
  }
}

mod list_plot_3d_function {
  use super::*;

  #[test]
  fn list_plot_3d_attributes() {
    assert_eq!(
      interpret("Attributes[ListPlot3D]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod input_function {
  use super::*;

  #[test]
  fn input_attributes() {
    assert_eq!(
      interpret("Attributes[Input]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod add_to_function {
  use super::*;

  #[test]
  fn add_to_works() {
    assert_eq!(interpret("x = 5; x += 3; x").unwrap(), "8");
  }

  #[test]
  fn add_to_attributes() {
    assert_eq!(
      interpret("Attributes[AddTo]").unwrap(),
      "{HoldFirst, Protected}"
    );
  }
}

mod subtract_from_function {
  use super::*;

  #[test]
  fn subtract_from_attributes() {
    assert_eq!(
      interpret("Attributes[SubtractFrom]").unwrap(),
      "{HoldFirst, Protected}"
    );
  }
}

mod times_by_function {
  use super::*;

  #[test]
  fn times_by_attributes() {
    assert_eq!(
      interpret("Attributes[TimesBy]").unwrap(),
      "{HoldFirst, Protected}"
    );
  }
}

mod divide_by_function {
  use super::*;

  #[test]
  fn divide_by_attributes() {
    assert_eq!(
      interpret("Attributes[DivideBy]").unwrap(),
      "{HoldFirst, Protected}"
    );
  }
}

mod golden_ratio_symbol {
  use super::*;

  #[test]
  fn golden_ratio_evaluates_to_itself() {
    assert_eq!(interpret("GoldenRatio").unwrap(), "GoldenRatio");
  }

  #[test]
  fn golden_ratio_numeric() {
    assert!(interpret("N[GoldenRatio]").unwrap().starts_with("1.61803"));
  }
}

mod complex_symbol {
  use super::*;

  #[test]
  fn complex_function_call() {
    assert_eq!(interpret("Complex[3, 4]").unwrap(), "3 + 4*I");
  }

  #[test]
  fn complex_is_head() {
    assert_eq!(interpret("Head[3 + 4 I]").unwrap(), "Complex");
  }

  #[test]
  fn complex_attributes() {
    assert_eq!(interpret("Attributes[Complex]").unwrap(), "{Protected}");
  }
}

mod hold_all_symbol {
  use super::*;

  #[test]
  fn hold_all_attributes() {
    assert_eq!(interpret("Attributes[HoldAll]").unwrap(), "{Protected}");
  }
}

mod listable_symbol {
  use super::*;

  #[test]
  fn listable_attributes() {
    assert_eq!(interpret("Attributes[Listable]").unwrap(), "{Protected}");
  }
}

mod hold_first_symbol {
  use super::*;

  #[test]
  fn hold_first_attributes() {
    assert_eq!(interpret("Attributes[HoldFirst]").unwrap(), "{Protected}");
  }
}

mod begin_end_package {
  use super::*;

  #[test]
  fn begin_returns_null() {
    assert_eq!(interpret("Begin[\"Private`\"]").unwrap(), "Null");
  }

  #[test]
  fn end_returns_null() {
    assert_eq!(interpret("End[]").unwrap(), "Null");
  }

  #[test]
  fn begin_package_returns_null() {
    assert_eq!(interpret("BeginPackage[\"MyPkg`\"]").unwrap(), "Null");
  }

  #[test]
  fn end_package_returns_null() {
    assert_eq!(interpret("EndPackage[]").unwrap(), "Null");
  }
}

mod break_function {
  use super::*;

  #[test]
  fn break_attributes() {
    assert_eq!(interpret("Attributes[Break]").unwrap(), "{Protected}");
  }
}

mod lighting_symbol {
  use super::*;

  #[test]
  fn lighting_attributes() {
    assert_eq!(interpret("Attributes[Lighting]").unwrap(), "{Protected}");
  }
}

mod modulus_symbol {
  use super::*;

  #[test]
  fn modulus_attributes() {
    assert_eq!(interpret("Attributes[Modulus]").unwrap(), "{Protected}");
  }
}

mod unset_function {
  use super::*;

  #[test]
  fn unset_removes_variable() {
    assert_eq!(interpret("x = 5; Unset[x]; x").unwrap(), "x");
  }

  #[test]
  fn unset_syntax() {
    assert_eq!(interpret("x = 5; x =.; x").unwrap(), "x");
  }

  #[test]
  fn unset_returns_null() {
    assert_eq!(interpret("x = 5; Unset[x]").unwrap(), "Null");
  }

  #[test]
  fn unset_removes_function_definition() {
    assert_eq!(interpret("f[x_] := x^2; f[3]").unwrap(), "9");
    // After unset, f should no longer be defined
    // (this tests function call form)
  }

  #[test]
  fn unset_attributes() {
    assert_eq!(
      interpret("Attributes[Unset]").unwrap(),
      "{HoldFirst, Protected, ReadProtected}"
    );
  }
}

mod repeated_null_function {
  use super::*;

  #[test]
  fn repeated_null_is_inert() {
    assert_eq!(
      interpret("RepeatedNull[x_, 3]").unwrap(),
      "RepeatedNull[x_, 3]"
    );
  }

  #[test]
  fn repeated_null_attributes() {
    assert_eq!(
      interpret("Attributes[RepeatedNull]").unwrap(),
      "{Protected}"
    );
  }
}

mod view_point_symbol {
  use super::*;

  #[test]
  fn view_point_attributes() {
    assert_eq!(interpret("Attributes[ViewPoint]").unwrap(), "{Protected}");
  }
}

mod box_ratios_symbol {
  use super::*;

  #[test]
  fn box_ratios_attributes() {
    assert_eq!(interpret("Attributes[BoxRatios]").unwrap(), "{Protected}");
  }
}

mod display_function_symbol {
  use super::*;

  #[test]
  fn display_function_attributes() {
    assert_eq!(
      interpret("Attributes[DisplayFunction]").unwrap(),
      "{Protected}"
    );
  }
}

mod right_symbol {
  use super::*;

  #[test]
  fn right_evaluates_to_itself() {
    assert_eq!(interpret("Right").unwrap(), "Right");
  }

  #[test]
  fn right_attributes() {
    assert_eq!(interpret("Attributes[Right]").unwrap(), "{Protected}");
  }
}

mod top_symbol {
  use super::*;

  #[test]
  fn top_evaluates_to_itself() {
    assert_eq!(interpret("Top").unwrap(), "Top");
  }

  #[test]
  fn top_attributes() {
    assert_eq!(interpret("Attributes[Top]").unwrap(), "{Protected}");
  }
}

mod bottom_symbol {
  use super::*;

  #[test]
  fn bottom_evaluates_to_itself() {
    assert_eq!(interpret("Bottom").unwrap(), "Bottom");
  }

  #[test]
  fn bottom_attributes() {
    assert_eq!(interpret("Attributes[Bottom]").unwrap(), "{Protected}");
  }
}

mod above_symbol {
  use super::*;

  #[test]
  fn above_evaluates_to_itself() {
    assert_eq!(interpret("Above").unwrap(), "Above");
  }

  #[test]
  fn above_attributes() {
    assert_eq!(interpret("Attributes[Above]").unwrap(), "{Protected}");
  }
}

mod working_precision_symbol {
  use super::*;

  #[test]
  fn working_precision_attributes() {
    assert_eq!(
      interpret("Attributes[WorkingPrecision]").unwrap(),
      "{Protected}"
    );
  }
}

mod information_function {
  use super::*;

  #[test]
  fn information_attributes() {
    assert_eq!(
      interpret("Attributes[Information]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod message_function {
  use super::*;

  #[test]
  fn message_returns_null() {
    assert_eq!(interpret("Message[f, \"test\"]").unwrap(), "Null");
  }

  #[test]
  fn message_attributes() {
    assert_eq!(
      interpret("Attributes[Message]").unwrap(),
      "{HoldFirst, Protected}"
    );
  }
}

mod non_commutative_multiply {
  use super::*;

  #[test]
  fn ncm_function_call() {
    assert_eq!(
      interpret("NonCommutativeMultiply[a, b, c]").unwrap(),
      "a**b**c"
    );
  }

  #[test]
  fn ncm_attributes() {
    assert_eq!(
      interpret("Attributes[NonCommutativeMultiply]").unwrap(),
      "{Flat, OneIdentity, Protected}"
    );
  }

  #[test]
  fn ncm_two_args() {
    assert_eq!(interpret("NonCommutativeMultiply[x, y]").unwrap(), "x**y");
  }
}

mod superscript_function {
  use super::*;

  #[test]
  fn superscript_is_inert() {
    assert_eq!(interpret("Superscript[x, 2]").unwrap(), "Superscript[x, 2]");
  }

  #[test]
  fn superscript_attributes() {
    assert_eq!(
      interpret("Attributes[Superscript]").unwrap(),
      "{NHoldRest, ReadProtected}"
    );
  }
}

mod repeated_function {
  use super::*;

  #[test]
  fn repeated_is_inert() {
    assert_eq!(interpret("Repeated[x_, 3]").unwrap(), "Repeated[x_, 3]");
  }

  #[test]
  fn repeated_attributes() {
    assert_eq!(interpret("Attributes[Repeated]").unwrap(), "{Protected}");
  }
}

mod number_form {
  use super::*;

  #[test]
  fn number_form_is_inert() {
    assert_eq!(
      interpret("NumberForm[3.14159, 4]").unwrap(),
      "NumberForm[3.14159, 4]"
    );
  }

  #[test]
  fn number_form_attributes() {
    assert_eq!(
      interpret("Attributes[NumberForm]").unwrap(),
      "{NHoldRest, Protected}"
    );
  }
}

mod slot_sequence {
  use super::*;

  #[test]
  fn slot_sequence_display() {
    assert_eq!(interpret("##").unwrap(), "##1");
    assert_eq!(interpret("##2").unwrap(), "##2");
  }

  #[test]
  fn slot_sequence_function_call_form() {
    assert_eq!(interpret("SlotSequence[1]").unwrap(), "##1");
    assert_eq!(interpret("SlotSequence[2]").unwrap(), "##2");
  }

  #[test]
  fn slot_sequence_in_plus() {
    assert_eq!(interpret("f = Plus[##] &; f[1, 2, 3]").unwrap(), "6");
  }

  #[test]
  fn slot_sequence_in_list() {
    assert_eq!(interpret("g = {##} &; g[a, b, c]").unwrap(), "{a, b, c}");
  }

  #[test]
  fn slot_sequence_from_position() {
    assert_eq!(
      interpret("h = {##2} &; h[a, b, c, d]").unwrap(),
      "{b, c, d}"
    );
  }

  #[test]
  fn slot_sequence_attributes() {
    assert_eq!(
      interpret("Attributes[SlotSequence]").unwrap(),
      "{NHoldAll, Protected}"
    );
  }

  #[test]
  fn slot_sequence_with_slot() {
    assert_eq!(
      interpret("f = {#1, ##2} &; f[x, y, z]").unwrap(),
      "{x, y, z}"
    );
  }
}

mod left_symbol {
  use super::*;

  #[test]
  fn left_evaluates_to_itself() {
    assert_eq!(interpret("Left").unwrap(), "Left");
  }

  #[test]
  fn left_attributes() {
    assert_eq!(interpret("Attributes[Left]").unwrap(), "{Protected}");
  }
}

mod real_symbol {
  use super::*;

  #[test]
  fn real_evaluates_to_itself() {
    assert_eq!(interpret("Real").unwrap(), "Real");
  }

  #[test]
  fn real_attributes() {
    assert_eq!(interpret("Attributes[Real]").unwrap(), "{Protected}");
  }

  #[test]
  fn real_is_head_of_floats() {
    assert_eq!(interpret("Head[3.14]").unwrap(), "Real");
    assert_eq!(interpret("Head[0.0]").unwrap(), "Real");
  }

  #[test]
  fn match_real_pattern() {
    assert_eq!(interpret("MatchQ[3.14, _Real]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[5, _Real]").unwrap(), "False");
  }
}

mod ticks_symbol {
  use super::*;

  #[test]
  fn ticks_evaluates_to_itself() {
    assert_eq!(interpret("Ticks").unwrap(), "Ticks");
  }

  #[test]
  fn ticks_attributes() {
    assert_eq!(interpret("Attributes[Ticks]").unwrap(), "{Protected}");
  }
}

mod boxed_symbol {
  use super::*;

  #[test]
  fn boxed_evaluates_to_itself() {
    assert_eq!(interpret("Boxed").unwrap(), "Boxed");
  }

  #[test]
  fn boxed_attributes() {
    assert_eq!(interpret("Attributes[Boxed]").unwrap(), "{Protected}");
  }
}

mod scaled_function {
  use super::*;

  #[test]
  fn scaled_evaluates_to_itself() {
    assert_eq!(interpret("Scaled").unwrap(), "Scaled");
  }

  #[test]
  fn scaled_function_call_is_inert() {
    assert_eq!(
      interpret("Scaled[{0.5, 0.5}]").unwrap(),
      "Scaled[{0.5, 0.5}]"
    );
  }

  #[test]
  fn scaled_attributes() {
    assert_eq!(interpret("Attributes[Scaled]").unwrap(), "{Protected}");
  }
}

mod plot_points_symbol {
  use super::*;

  #[test]
  fn plot_points_evaluates_to_itself() {
    assert_eq!(interpret("PlotPoints").unwrap(), "PlotPoints");
  }

  #[test]
  fn plot_points_attributes() {
    assert_eq!(interpret("Attributes[PlotPoints]").unwrap(), "{Protected}");
  }
}

mod needs_function {
  use super::*;

  #[test]
  fn needs_returns_null() {
    assert_eq!(interpret("Needs[\"SomePackage`\"]").unwrap(), "Null");
  }

  #[test]
  fn needs_attributes() {
    assert_eq!(interpret("Attributes[Needs]").unwrap(), "{Protected}");
  }
}

mod center_symbol {
  use super::*;

  #[test]
  fn center_evaluates_to_itself() {
    assert_eq!(interpret("Center").unwrap(), "Center");
  }

  #[test]
  fn center_attributes() {
    assert_eq!(interpret("Attributes[Center]").unwrap(), "{Protected}");
  }
}

mod rational_symbol {
  use super::*;

  #[test]
  fn rational_evaluates_to_itself() {
    assert_eq!(interpret("Rational").unwrap(), "Rational");
  }

  #[test]
  fn rational_attributes() {
    assert_eq!(interpret("Attributes[Rational]").unwrap(), "{Protected}");
  }

  #[test]
  fn rational_is_head_of_fractions() {
    assert_eq!(interpret("Head[1/3]").unwrap(), "Rational");
    assert_eq!(interpret("Head[2/5]").unwrap(), "Rational");
  }

  #[test]
  fn rational_function_call_creates_fraction() {
    assert_eq!(interpret("Rational[1, 3]").unwrap(), "1/3");
    assert_eq!(interpret("Rational[2, 4]").unwrap(), "1/2");
    assert_eq!(interpret("Rational[3, 1]").unwrap(), "3");
  }

  #[test]
  fn match_rational_pattern() {
    assert_eq!(interpret("MatchQ[1/2, _Rational]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[5, _Rational]").unwrap(), "False");
  }
}

mod mesh_symbol {
  use super::*;

  #[test]
  fn mesh_evaluates_to_itself() {
    assert_eq!(interpret("Mesh").unwrap(), "Mesh");
  }

  #[test]
  fn mesh_attributes() {
    assert_eq!(interpret("Attributes[Mesh]").unwrap(), "{Protected}");
  }
}

mod string_symbol {
  use super::*;

  #[test]
  fn string_evaluates_to_itself() {
    assert_eq!(interpret("String").unwrap(), "String");
  }

  #[test]
  fn string_attributes() {
    assert_eq!(interpret("Attributes[String]").unwrap(), "{Protected}");
  }

  #[test]
  fn string_is_head_of_strings() {
    assert_eq!(interpret("Head[\"hello\"]").unwrap(), "String");
  }

  #[test]
  fn string_function_call_is_inert() {
    assert_eq!(interpret("String[1, 2]").unwrap(), "String[1, 2]");
  }

  #[test]
  fn match_string_pattern() {
    assert_eq!(interpret("MatchQ[\"hello\", _String]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[5, _String]").unwrap(), "False");
  }
}

mod optional_function {
  use super::*;

  #[test]
  fn optional_with_default() {
    assert_eq!(interpret("Optional[x_, 0]").unwrap(), "x_:0");
  }

  #[test]
  fn optional_with_head_and_default() {
    assert_eq!(interpret("Optional[x_Integer, 5]").unwrap(), "x_Integer:5");
  }

  #[test]
  fn optional_without_default() {
    assert_eq!(interpret("Optional[x_]").unwrap(), "x_.");
  }

  #[test]
  fn optional_attributes() {
    assert_eq!(interpret("Attributes[Optional]").unwrap(), "{Protected}");
  }

  #[test]
  fn optional_syntax_with_colon() {
    assert_eq!(interpret("f[x_ : 0] := x; f[]").unwrap(), "0");
    assert_eq!(interpret("f[x_ : 0] := x; f[5]").unwrap(), "5");
  }
}

mod integer_symbol {
  use super::*;

  #[test]
  fn integer_evaluates_to_itself() {
    assert_eq!(interpret("Integer").unwrap(), "Integer");
  }

  #[test]
  fn integer_attributes() {
    assert_eq!(interpret("Attributes[Integer]").unwrap(), "{Protected}");
  }

  #[test]
  fn integer_is_head_of_integers() {
    assert_eq!(interpret("Head[5]").unwrap(), "Integer");
    assert_eq!(interpret("Head[0]").unwrap(), "Integer");
    assert_eq!(interpret("Head[-3]").unwrap(), "Integer");
  }

  #[test]
  fn integer_function_call_is_inert() {
    assert_eq!(interpret("Integer[3]").unwrap(), "Integer[3]");
    assert_eq!(interpret("Integer[x]").unwrap(), "Integer[x]");
  }

  #[test]
  fn match_integer_pattern() {
    assert_eq!(interpret("MatchQ[5, _Integer]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[1/2, _Integer]").unwrap(), "False");
    assert_eq!(interpret("MatchQ[3.0, _Integer]").unwrap(), "False");
  }
}

mod matrix_form {
  use super::*;

  #[test]
  fn matrix_form_basic() {
    assert_eq!(
      interpret("MatrixForm[{{1, 2}, {3, 4}}]").unwrap(),
      "MatrixForm[{{1, 2}, {3, 4}}]"
    );
  }

  #[test]
  fn matrix_form_head() {
    assert_eq!(
      interpret("Head[MatrixForm[{{1, 2}, {3, 4}}]]").unwrap(),
      "MatrixForm"
    );
  }

  #[test]
  fn matrix_form_attributes() {
    assert_eq!(
      interpret("Attributes[MatrixForm]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  #[test]
  fn matrix_form_single_list() {
    assert_eq!(
      interpret("MatrixForm[{1, 2, 3}]").unwrap(),
      "MatrixForm[{1, 2, 3}]"
    );
  }
}

mod out_function {
  use super::*;

  #[test]
  fn out_evaluates_to_itself() {
    assert_eq!(interpret("Out").unwrap(), "Out");
  }

  #[test]
  fn out_head_is_symbol() {
    assert_eq!(interpret("Head[Out]").unwrap(), "Symbol");
  }

  #[test]
  fn out_attributes() {
    assert_eq!(
      interpret("Attributes[Out]").unwrap(),
      "{Listable, NHoldFirst, Protected}"
    );
  }

  #[test]
  fn out_with_index_is_inert() {
    assert_eq!(interpret("Out[1]").unwrap(), "Out[1]");
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

mod tag_set_delayed {
  use super::*;

  #[test]
  fn basic_upvalue() {
    // g /: f[g[x_]] := fg[x] — defines an upvalue for g
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := fg[x]; {f[g[2]], f[h[2]]}").unwrap(),
      "{fg[2], f[h[2]]}"
    );
  }

  #[test]
  fn multi_arg_upvalue() {
    // Upvalue with multiple arguments
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_], y_] := fg[x, y]; f[g[2], 3]").unwrap(),
      "fg[2, 3]"
    );
  }

  #[test]
  fn multiple_upvalues_same_tag() {
    // Multiple upvalue definitions for the same tag
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := fg[x]; g /: h[g[x_]] := hg[x]; {f[g[3]], h[g[5]], f[5]}").unwrap(),
      "{fg[3], hg[5], f[5]}"
    );
  }

  #[test]
  fn overwrite_upvalue() {
    // Later definitions take priority
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := fg[x]; g /: f[g[x_]] := fg2[x]; {f[g[2]]}")
        .unwrap(),
      "{fg2[2]}"
    );
  }

  #[test]
  fn tag_set_evaluated_rhs() {
    // TagSet (=) evaluates the RHS
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] = fg[x]; {f[g[2]], f[h[2]]}").unwrap(),
      "{fg[2], f[h[2]]}"
    );
  }

  #[test]
  fn functional_form() {
    // TagSetDelayed[tag, lhs, rhs] as function call
    clear_state();
    assert_eq!(
      interpret("TagSetDelayed[g, f[g[x_]], fg[x]]; {f[g[2]], f[h[2]]}")
        .unwrap(),
      "{fg[2], f[h[2]]}"
    );
  }

  #[test]
  fn upvalue_non_matching_head() {
    // The upvalue should not fire when the argument head doesn't match
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := fg[x]; f[h[2]]").unwrap(),
      "f[h[2]]"
    );
  }

  #[test]
  fn upvalue_with_computation() {
    // Upvalue body performs computation
    clear_state();
    assert_eq!(
      interpret("myType /: combine[myType[x_], myType[y_]] := myType[x + y]; combine[myType[3], myType[5]]").unwrap(),
      "myType[8]"
    );
  }

  #[test]
  fn upvalue_returns_null() {
    // TagSetDelayed returns Null
    clear_state();
    assert_eq!(interpret("g /: f[g[x_]] := fg[x]").unwrap(), "Null");
  }

  #[test]
  fn clear_all_removes_upvalues() {
    // ClearAll should remove upvalues
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := fg[x]; ClearAll[g]; f[g[2]]").unwrap(),
      "f[g[2]]"
    );
  }
}

mod upset {
  use super::*;

  #[test]
  fn basic_upset() {
    clear_state();
    assert_eq!(interpret("f[g] ^= 5; f[g]").unwrap(), "5");
  }

  #[test]
  fn upset_returns_value() {
    // UpSet returns the evaluated RHS
    clear_state();
    assert_eq!(interpret("f[g] ^= 1 + 2").unwrap(), "3");
  }

  #[test]
  fn upset_evaluates_rhs() {
    // RHS is evaluated before storing
    clear_state();
    assert_eq!(interpret("f[g] ^= 2 + 3; f[g]").unwrap(), "5");
  }

  #[test]
  fn upset_multiple_symbols() {
    // UpSet stores for all symbols in arguments
    clear_state();
    assert_eq!(interpret("f[g, h] ^= 10; f[g, h]").unwrap(), "10");
  }

  #[test]
  fn upset_with_nested_function() {
    // Tag is extracted from the head of nested function call
    clear_state();
    assert_eq!(interpret("f[g[x_]] ^= x^2; f[g[3]]").unwrap(), "9");
  }

  #[test]
  fn upset_stores_upvalue() {
    // UpValues should contain a definition
    clear_state();
    let result = interpret("f[g] ^= 5; UpValues[g]").unwrap();
    assert!(result.contains(":> 5"));
  }

  #[test]
  fn upset_attributes() {
    assert_eq!(
      interpret("Attributes[UpSet]").unwrap(),
      "{HoldFirst, Protected, SequenceHold}"
    );
  }
}

mod upset_delayed {
  use super::*;

  #[test]
  fn upset_delayed_basic() {
    clear_state();
    assert_eq!(interpret("f[g] ^:= 5; f[g]").unwrap(), "5");
  }

  #[test]
  fn upset_delayed_returns_null() {
    // UpSetDelayed returns Null (unlike UpSet which returns evaluated RHS)
    clear_state();
    assert_eq!(interpret("f[g] ^:= 1 + 2").unwrap(), "Null");
  }

  #[test]
  fn upset_delayed_does_not_evaluate_rhs() {
    // RHS is not evaluated at definition time, but at use time
    clear_state();
    assert_eq!(
      interpret("n = 0; f[g] ^:= (n = n + 1); f[g]; f[g]; n").unwrap(),
      "2"
    );
  }

  #[test]
  fn upset_delayed_with_pattern() {
    clear_state();
    assert_eq!(interpret("f[g[x_]] ^:= x^2; f[g[4]]").unwrap(), "16");
  }

  #[test]
  fn upset_delayed_multiple_symbols() {
    clear_state();
    assert_eq!(interpret("f[g, h] ^:= 10; f[g, h]").unwrap(), "10");
  }

  #[test]
  fn upset_delayed_stores_upvalue() {
    clear_state();
    let result = interpret("f[g] ^:= 5; UpValues[g]").unwrap();
    assert!(result.contains(":> 5"));
  }

  #[test]
  fn upset_delayed_attributes() {
    assert_eq!(
      interpret("Attributes[UpSetDelayed]").unwrap(),
      "{HoldAll, Protected, SequenceHold}"
    );
  }
}
