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
    assert_eq!(interpret("1 + 2;").unwrap(), "\0");
  }

  #[test]
  fn trailing_semicolon_with_print() {
    // Print[1]; should still execute Print, result is Null
    assert_eq!(interpret("Print[1];").unwrap(), "\0");
  }

  #[test]
  fn trailing_semicolon_with_postfix() {
    // {1,2,3} // Map[Print]; should print and return Null
    assert_eq!(interpret("{1,2,3} // Map[Print];").unwrap(), "\0");
  }

  #[test]
  fn no_trailing_semicolon_shows_result() {
    // Without trailing ;, result should be shown
    assert_eq!(interpret("1 + 2").unwrap(), "3");
  }

  #[test]
  fn compound_expression_with_trailing_semicolon() {
    // x = 5; x + 1; should return Null
    assert_eq!(interpret("x = 5; x + 1;").unwrap(), "\0");
  }

  #[test]
  fn compound_expression_without_trailing_semicolon() {
    // x = 5; x + 1 should show the final result
    assert_eq!(interpret("x = 5; x + 1").unwrap(), "6");
  }

  #[test]
  fn null_symbol_uses_sentinel() {
    // The Null symbol should use the "\0" sentinel so visual contexts
    // (Studio, JupyterLite) can suppress it without confusing it
    // with the string "Null".
    assert_eq!(interpret("Clear[x]").unwrap(), "\0");
  }

  #[test]
  fn string_null_is_not_suppressed() {
    // The string "Null" must remain as "Null", not be suppressed
    assert_eq!(interpret(r#""Null""#).unwrap(), "Null");
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

  #[test]
  fn plus_term_ordering_reverse_lex() {
    // Wolfram sorts polynomial terms by reverse-lex variable ordering
    assert_eq!(interpret("x^2 + 2*b*x + b^2").unwrap(), "b^2 + 2*b*x + x^2");
  }

  #[test]
  fn plus_term_ordering_ascending_degree() {
    // For single-variable polynomials, ascending degree order
    assert_eq!(interpret("3*x^2 + 6*x + 2").unwrap(), "2 + 6*x + 3*x^2");
  }

  #[test]
  fn plus_term_ordering_multivar() {
    // Multi-variable terms: reverse-lex order
    assert_eq!(
      interpret("a*c + b*c + a*d + b*d").unwrap(),
      "a*c + b*c + a*d + b*d"
    );
  }

  #[test]
  fn plus_term_ordering_with_division() {
    // Terms with 1/z should sort by the variable z
    assert_eq!(
      interpret("x/Sqrt[5] + y^2 + 1/z").unwrap(),
      "x/Sqrt[5] + y^2 + z^(-1)"
    );
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
    // The result is simplified: 2*Pi - Pi/4 = (8*Pi - Pi)/4 = 7*Pi/4
    assert_eq!(
      interpret("FullForm[2 Pi - Pi/4]").unwrap(),
      "Times[Rational[7, 4], Pi]"
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
    // In InputForm, Inequality always uses the head form (even with same operators)
    assert_eq!(
      interpret(
        "ToString[Inequality[0, LessEqual, x, LessEqual, 1], InputForm]"
      )
      .unwrap(),
      "Inequality[0, LessEqual, x, LessEqual, 1]"
    );
    assert_eq!(
      interpret("ToString[Inequality[a, Less, b, Less, c], InputForm]")
        .unwrap(),
      "Inequality[a, Less, b, Less, c]"
    );
    // Mixed operators also use Inequality[] head in InputForm
    assert_eq!(
      interpret("ToString[Inequality[a, LessEqual, b, Less, c], InputForm]")
        .unwrap(),
      "Inequality[a, LessEqual, b, Less, c]"
    );
    // Chained comparison with same operators uses infix in InputForm
    assert_eq!(
      interpret("ToString[0 <= x <= 1, InputForm]").unwrap(),
      "0 <= x <= 1"
    );
    assert_eq!(
      interpret("ToString[a < b < c, InputForm]").unwrap(),
      "a < b < c"
    );
    // Chained comparison with mixed operators uses Inequality head in InputForm
    assert_eq!(
      interpret("ToString[Inequality[a, LessEqual, b, Less, c], InputForm]")
        .unwrap(),
      "Inequality[a, LessEqual, b, Less, c]"
    );
  }

  #[test]
  fn tostring_input_form_negative_coefficients() {
    // Negative coefficients in Plus should render as "- N*..." not "+ -N*..."
    assert_eq!(
      interpret("ToString[Expand[Resultant[x^2 + a*x + b, x^2 + c*x + d, x]], InputForm]").unwrap(),
      "b^2 - a*b*c + b*c^2 + a^2*d - 2*b*d - a*c*d + d^2"
    );
    assert_eq!(
      interpret(
        "ToString[Expand[InterpolatingPolynomial[{0, 1, 8, 27}, x]], InputForm]"
      )
      .unwrap(),
      "-1 + 3*x - 3*x^2 + x^3"
    );
    assert_eq!(
      interpret("ToString[Discriminant[x^2 + b*x + c, x], InputForm]").unwrap(),
      "b^2 - 4*c"
    );
    assert_eq!(
      interpret("ToString[Discriminant[a*x^2 + b*x + c, x], InputForm]")
        .unwrap(),
      "b^2 - 4*a*c"
    );
    assert_eq!(
      interpret("ToString[Discriminant[x^3 + p*x + q, x], InputForm]").unwrap(),
      "-4*p^3 - 27*q^2"
    );
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

  // Newlines inside function-call brackets are ignored by the parser, so
  // `Sin[ \n 0 ]` parses as `Sin[0]` regardless of where the newlines are.
  #[test]
  fn function_call_leading_newline() {
    assert_eq!(
      interpret("Hold[Sin[\n0]] // FullForm").unwrap(),
      "Hold[Sin[0]]"
    );
  }

  #[test]
  fn function_call_multiple_leading_newlines() {
    assert_eq!(
      interpret("Hold[Sin[\n\n0]] // FullForm").unwrap(),
      "Hold[Sin[0]]"
    );
  }

  #[test]
  fn function_call_trailing_newline() {
    assert_eq!(
      interpret("Hold[Sin[0\n]] // FullForm").unwrap(),
      "Hold[Sin[0]]"
    );
  }

  // A CompoundExpression separator followed by a newline-separated tail
  // parses as `CompoundExpression[a, b]` inside the surrounding call.
  #[test]
  fn function_call_compound_expression_across_newlines() {
    assert_eq!(
      interpret("Hold[f[a;\nb]] // FullForm").unwrap(),
      "Hold[f[CompoundExpression[a, b]]]"
    );
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
  fn full_form_times_with_number() {
    // Regression test for https://github.com/ad-si/Woxi/issues/71
    assert_eq!(interpret("FullForm[5*x]").unwrap(), "Times[5, x]");
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

  #[test]
  fn full_form_complex_number() {
    assert_eq!(interpret("FullForm[2 + 3*I]").unwrap(), "Complex[2, 3]");
  }

  #[test]
  fn full_form_imaginary_unit() {
    assert_eq!(interpret("FullForm[I]").unwrap(), "Complex[0, 1]");
  }

  #[test]
  fn full_form_complex_rational() {
    assert_eq!(
      interpret("FullForm[1/2 + 3/4*I]").unwrap(),
      "Complex[Rational[1, 2], Rational[3, 4]]"
    );
  }

  #[test]
  fn full_form_division() {
    assert_eq!(
      interpret("FullForm[a/b]").unwrap(),
      "Times[a, Power[b, -1]]"
    );
  }

  #[test]
  fn full_form_reciprocal() {
    assert_eq!(interpret("FullForm[1/z]").unwrap(), "Power[z, -1]");
  }

  /// Division is canonicalized to Times[a, Power[b, -1]]
  #[test]
  fn full_form_division_canonical() {
    assert_eq!(
      interpret("FullForm[x/y]").unwrap(),
      "Times[x, Power[y, -1]]"
    );
  }

  #[test]
  fn full_form_sqrt() {
    assert_eq!(
      interpret("FullForm[Sqrt[5]]").unwrap(),
      "Power[5, Rational[1, 2]]"
    );
  }

  #[test]
  fn full_form_complex_expression() {
    // x/Sqrt[5] canonicalizes to Times[Power[5, Rational[-1, 2]], x]
    assert_eq!(
      interpret("FullForm[x/Sqrt[5] + y^2 + 1/z]").unwrap(),
      "Plus[Times[Power[5, Rational[-1, 2]], x], Power[y, 2], Power[z, -1]]"
    );
  }

  // Issue #97: Sqrt[x] should canonicalize to Power[x, Rational[1, 2]]
  #[test]
  fn head_of_sqrt() {
    assert_eq!(interpret("Head[Sqrt[x]]").unwrap(), "Power");
  }

  #[test]
  fn sqrt_identical_to_power_half() {
    assert_eq!(interpret("Sqrt[x] === Power[x, 1/2]").unwrap(), "True");
  }

  #[test]
  fn sqrt_parts() {
    assert_eq!(
      interpret("{Sqrt[x][[0]], Sqrt[x][[1]], Sqrt[x][[2]]}").unwrap(),
      "{Power, x, 1/2}"
    );
  }

  #[test]
  fn sqrt_of_triply_nested_reciprocal() {
    assert_eq!(
      interpret("Sqrt[1/(1+1/(1+1/a))]").unwrap(),
      "Sqrt[(1 + (1 + a^(-1))^(-1))^(-1)]"
    );
  }

  // Log[b, x] should canonicalize to Log[x]/Log[b]
  #[test]
  fn head_of_log_two_arg() {
    assert_eq!(interpret("Head[Log[2, x]]").unwrap(), "Times");
  }

  #[test]
  fn log_two_arg_identical_to_quotient() {
    assert_eq!(interpret("Log[2, x] === Log[x]/Log[2]").unwrap(), "True");
  }

  // CubeRoot[x] should canonicalize to Surd[x, 3]
  #[test]
  fn head_of_cube_root() {
    assert_eq!(interpret("Head[CubeRoot[x]]").unwrap(), "Surd");
  }

  #[test]
  fn cube_root_identical_to_surd() {
    assert_eq!(interpret("CubeRoot[x] === Surd[x, 3]").unwrap(), "True");
  }

  #[test]
  fn full_form_no_canonicalization_regression() {
    // Regression test for issue #91: FullForm must not canonicalize Divide.
    // Pasting FullForm output back should give the same behavior as the original.
    let full_form = interpret("FullForm[1/((1 + x) (5 + x))]").unwrap();
    let apart_original = interpret("Apart[1/((1 + x) (5 + x))]").unwrap();
    let apart_from_full_form =
      interpret(&format!("Apart[{}]", full_form)).unwrap();
    assert_eq!(
      apart_original, apart_from_full_form,
      "Apart of FullForm output should match Apart of original"
    );
  }

  #[test]
  fn full_form_no_svg_output() {
    // Regression: FullForm results must be plain text (no SVG) in the playground
    use woxi::interpret_with_stdout;
    let result = interpret_with_stdout("FullForm[1/z]").unwrap();
    assert!(
      result.output_svg.is_none(),
      "FullForm should not produce SVG output"
    );
    assert_eq!(result.result, "Power[z, -1]");
  }
}

mod tree_form {
  use super::*;

  #[test]
  fn tree_form_simple() {
    // TreeForm stays as wrapper in OutputForm (matching wolframscript)
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
    // 1 + 2 evaluates to 3, then wrapped
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
  fn replace_all_function_form_reevaluates_result() {
    // ReplaceAll[...] (function-call form) should re-evaluate the result
    // after substitution so e.g. 1 + 2 becomes 3. Regression: previously
    // only the /. operator did this, while the function form returned
    // "{2, 1 + 2, 2 + 2}" unchanged.
    assert_eq!(
      interpret("ReplaceAll[{x, x + 1, x + 2}, x -> 2]").unwrap(),
      "{2, 3, 4}"
    );
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
  fn replace_all_head_list_to_sequence() {
    // List -> Sequence should flatten nested lists into function args
    assert_eq!(
      interpret("f[{{a, b}, {c, d}, {a}}] /. List -> Sequence").unwrap(),
      "f[a, b, c, d, a]"
    );
  }

  #[test]
  fn replace_all_head_list_to_function() {
    // List -> f should turn {a, b} into f[a, b]
    assert_eq!(interpret("{a, b} /. List -> g").unwrap(), "g[a, b]");
  }

  #[test]
  fn replace_all_head_list_to_plus() {
    // List -> Plus should sum the elements
    assert_eq!(interpret("{1, 2, 3} /. List -> Plus").unwrap(), "6");
  }

  #[test]
  fn replace_all_head_function_call() {
    // f -> g should replace function head
    assert_eq!(interpret("f[a, b] /. f -> g").unwrap(), "g[a, b]");
  }

  #[test]
  fn replace_all_list_as_argument() {
    // List as a symbol argument should be replaced
    assert_eq!(interpret("g[List, a] /. List -> f").unwrap(), "g[f, a]");
  }

  #[test]
  fn replace_all_multi_rule_with_head() {
    // Multi-rule should replace both args and head
    assert_eq!(
      interpret("{a, b} /. {a -> 1, List -> f}").unwrap(),
      "f[1, b]"
    );
    assert_eq!(interpret("f[a] /. {a -> 1, f -> g}").unwrap(), "g[1]");
  }

  #[test]
  fn replace_all_nested_list_to_sequence() {
    // Nested lists should all be replaced
    assert_eq!(
      interpret("f[{a, b}] /. List -> Sequence").unwrap(),
      "f[a, b]"
    );
  }

  #[test]
  fn rule_with_patterns() {
    assert_eq!(
      interpret("{1, 2, 3} /. x_Integer -> x^2").unwrap(),
      "{1, 4, 9}"
    );
  }

  #[test]
  fn blank_type_replace_list() {
    // Multiple typed-Blank rules applied to a list with mixed types.
    // Matches wolframscript.
    assert_eq!(
      interpret(r#"{42, 1.0, x} /. {_Integer -> "integer", _Real -> "real"} // InputForm"#)
        .unwrap(),
      "InputForm[{integer, real, x}]"
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

  // Regression test for https://github.com/ad-si/Woxi/issues/96
  // Pattern variable names must not leak across bindings in rules.
  #[test]
  fn rule_pattern_variable_no_leak() {
    assert_eq!(
      interpret("f[a + 1, b + 2] /. f[u_, a_] -> {u, a}").unwrap(),
      "{1 + a, 2 + b}"
    );
  }

  #[test]
  fn rule_pattern_variable_no_leak_single_rule() {
    assert_eq!(interpret("{a + 1} /. {a_} -> a^2").unwrap(), "(1 + a)^2");
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

  #[test]
  fn pattern_variable_binding_consistency() {
    // Same named pattern variable must bind to the same value
    // f[x_, x_] should match f[a, a] but not f[a, b]
    assert_eq!(interpret("f[a, a] /. f[x_, x_] -> yes").unwrap(), "yes");
    assert_eq!(interpret("f[a, b] /. f[x_, x_] -> yes").unwrap(), "f[a, b]");
  }

  #[test]
  fn pattern_variable_no_match_sqrt_vs_symbol() {
    // Regression test for issue #65:
    // x_ bound to Symbol x should not match Sqrt[x]
    assert_eq!(
      interpret(
        "Int[1/(x_*(a_+b_.*x_)),x_Symbol] := \
         -Log[(a+b*x)/x]/a /; FreeQ[{a,b},x]; \
         Int[1/(Sqrt[x]*(a + b*x)), x]"
      )
      .unwrap(),
      "Int[1/(Sqrt[x]*(a + b*x)), x]"
    );
    // But it should still match when x_ consistently binds to x
    assert_eq!(
      interpret(
        "Int[1/(x_*(a_+b_.*x_)),x_Symbol] := \
         -Log[(a+b*x)/x]/a /; FreeQ[{a,b},x]; \
         Int[1/(x*(a + b*x)), x]"
      )
      .unwrap(),
      "-(Log[(a + b*x)/x]/a)"
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
    assert_eq!(
      interpret("Attributes[False]").unwrap(),
      "{Locked, Protected}"
    );
  }

  #[test]
  fn not_false_is_true() {
    assert_eq!(interpret("Not[False]").unwrap(), "True");
  }

  #[test]
  fn not_prefix_after_and_operator() {
    // !q must parse correctly after &&
    assert_eq!(interpret("True && !False").unwrap(), "True");
  }

  #[test]
  fn not_prefix_after_or_operator() {
    // !q must parse correctly after ||
    assert_eq!(interpret("False || !False").unwrap(), "True");
  }

  #[test]
  fn not_prefix_symbolic_after_and() {
    assert_eq!(interpret("p && !q").unwrap(), "p &&  !q");
  }

  #[test]
  fn boolean_minimize_with_not_prefix() {
    // p && q || p && !q simplifies to p
    assert_eq!(
      interpret("BooleanMinimize[p && q || p && !q]").unwrap(),
      "p"
    );
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

mod condition_pattern_matching {
  use super::*;

  #[test]
  fn matchq_with_condition_true() {
    assert_eq!(interpret("MatchQ[4, x_ /; x > 3]").unwrap(), "True");
  }

  #[test]
  fn setdelayed_with_condition_applies_when_true() {
    // f[x_] := p[x] /; x>0 — conditional definition applies for positive x.
    clear_state();
    assert_eq!(interpret("f[x_] := p[x] /; x>0; f[3]").unwrap(), "p[3]");
  }

  #[test]
  fn matchq_with_condition_false() {
    assert_eq!(interpret("MatchQ[2, x_ /; x > 3]").unwrap(), "False");
  }

  #[test]
  fn matchq_with_trivial_condition() {
    assert_eq!(interpret("MatchQ[4, _ /; True]").unwrap(), "True");
  }

  #[test]
  fn matchq_with_false_condition() {
    assert_eq!(interpret("MatchQ[4, _ /; False]").unwrap(), "False");
  }

  #[test]
  fn matchq_with_evenq_condition() {
    assert_eq!(interpret("MatchQ[4, x_ /; EvenQ[x]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[3, x_ /; EvenQ[x]]").unwrap(), "False");
  }

  #[test]
  fn cases_with_condition() {
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4, 5}, x_ /; x > 3]").unwrap(),
      "{4, 5}"
    );
  }

  #[test]
  fn cases_with_condition_and_rule_delayed() {
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4, 5}, x_ /; x > 3 :> x^2]").unwrap(),
      "{16, 25}"
    );
  }

  #[test]
  fn cases_with_condition_evenq_rule() {
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4, 5}, x_ /; EvenQ[x] :> x^2]").unwrap(),
      "{4, 16}"
    );
  }

  #[test]
  fn matchq_blank_sequence_with_condition() {
    // BlankSequence with Condition: x__ /; test should work with multiple matched elements
    assert_eq!(
      interpret("MatchQ[f[1, 2, 3], f[x__Integer /; Total[{x}] > 5]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[1, 2, 3], f[x__Integer /; Total[{x}] > 10]]")
        .unwrap(),
      "False"
    );
    assert_eq!(
      interpret("MatchQ[f[1, 2, 3], f[x__ /; Length[{x}] > 2]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[1, 2, 3], f[x__ /; Length[{x}] > 5]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn cases_blank_sequence_with_condition() {
    assert_eq!(
      interpret(
        "Cases[{f[1, 2, 3], f[4, 5, 6]}, f[x__Integer /; Total[{x}] > 10]]"
      )
      .unwrap(),
      "{f[4, 5, 6]}"
    );
  }

  #[test]
  fn replace_all_blank_sequence_with_condition() {
    assert_eq!(
      interpret("f[1, 2, 3] /. f[x__Integer /; Total[{x}] > 5] :> Total[{x}]")
        .unwrap(),
      "6"
    );
    assert_eq!(
      interpret("f[1, 2, 3] /. f[x__Integer /; Total[{x}] > 10] :> Total[{x}]")
        .unwrap(),
      "f[1, 2, 3]"
    );
  }

  #[test]
  fn replace_all_with_condition() {
    assert_eq!(
      interpret("{1, 2, 3, 4, 5} /. x_ /; x > 3 :> x^2").unwrap(),
      "{1, 2, 3, 16, 25}"
    );
  }

  #[test]
  fn replace_all_with_condition_in_list() {
    assert_eq!(
      interpret("ReplaceAll[{1, 2, 3, 4, 5}, {x_ /; EvenQ[x] :> x^2}]")
        .unwrap(),
      "{1, 4, 3, 16, 5}"
    );
  }

  #[test]
  fn condition_with_head_constraint() {
    assert_eq!(
      interpret("MatchQ[42, x_Integer /; x > 10]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[5, x_Integer /; x > 10]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("MatchQ[\"hello\", x_Integer /; x > 10]").unwrap(),
      "False"
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

  #[test]
  fn pattern_test_with_head_match_q() {
    assert_eq!(
      interpret("MatchQ[3, _Integer?NonNegative]").unwrap(),
      "True"
    );
  }

  #[test]
  fn pattern_test_with_head_no_match_head() {
    // 3.5 is Real, not Integer — head doesn't match
    assert_eq!(
      interpret("MatchQ[3.5, _Integer?NonNegative]").unwrap(),
      "False"
    );
  }

  #[test]
  fn pattern_test_with_head_no_match_test() {
    // -3 is Integer but not NonNegative — test fails
    assert_eq!(
      interpret("MatchQ[-3, _Integer?NonNegative]").unwrap(),
      "False"
    );
  }

  #[test]
  fn pattern_test_with_head_in_function_def() {
    assert_eq!(
      interpret("f[n_Integer?NonNegative] := n + 1; f[3]").unwrap(),
      "4"
    );
  }

  #[test]
  fn pattern_test_with_head_function_no_match() {
    assert_eq!(
      interpret("g[n_Integer?NonNegative] := n + 1; g[-1]").unwrap(),
      "g[-1]"
    );
  }

  #[test]
  fn pattern_test_with_head_display() {
    assert_eq!(
      interpret("Hold[n_Integer?NonNegative]").unwrap(),
      "Hold[n_Integer?NonNegative]"
    );
  }

  #[test]
  fn pattern_test_with_head_fullform() {
    assert_eq!(
      interpret("FullForm[Hold[_Integer?NonNegative]]").unwrap(),
      "Hold[PatternTest[Blank[Integer], NonNegative]]"
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

  #[test]
  fn blank_sequence_display() {
    assert_eq!(interpret("BlankSequence[]").unwrap(), "__");
  }

  #[test]
  fn blank_sequence_with_head() {
    assert_eq!(interpret("BlankSequence[Integer]").unwrap(), "__Integer");
  }

  #[test]
  fn blank_sequence_head() {
    assert_eq!(interpret("Head[BlankSequence[]]").unwrap(), "BlankSequence");
  }

  #[test]
  fn blank_sequence_syntax() {
    assert_eq!(interpret("__").unwrap(), "__");
  }

  #[test]
  fn anonymous_blank_with_head_syntax() {
    assert_eq!(interpret("_Integer").unwrap(), "_Integer");
    assert_eq!(interpret("__Integer").unwrap(), "__Integer");
    assert_eq!(interpret("___Integer").unwrap(), "___Integer");
  }

  #[test]
  fn head_of_anonymous_blanks() {
    assert_eq!(interpret("Head[_]").unwrap(), "Blank");
    assert_eq!(interpret("Head[__]").unwrap(), "BlankSequence");
    assert_eq!(interpret("Head[___]").unwrap(), "BlankNullSequence");
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

  #[test]
  fn message_name_double_colon_syntax() {
    // `a::b` parses as MessageName[a, "b"].
    assert_eq!(interpret("a::b").unwrap(), "MessageName[a, b]");
  }

  #[test]
  fn message_name_set_returns_rhs() {
    clear_state();
    assert_eq!(interpret("freshMsgA::usage = \"hello\"").unwrap(), "hello");
  }

  #[test]
  fn message_name_lookup_after_set() {
    clear_state();
    assert_eq!(
      interpret("freshMsgB::tag = \"val\"; freshMsgB::tag").unwrap(),
      "val"
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
    // Wolfram: Plot3D has HoldAll (like Plot, ListPlot, etc.).
    // (A fresh kernel shows only {Protected, ReadProtected}, but the
    // symbol auto-upgrades to HoldAll on first mention.)
    assert_eq!(
      interpret("Attributes[Plot3D]").unwrap(),
      "{HoldAll, Protected, ReadProtected}"
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
  fn increment_symbolic_expression() {
    // y holds `2 x`; y++ adds 1 to it, yielding `1 + 2*x`.
    assert_eq!(interpret("y = 2 x; y++; y").unwrap(), "1 + 2*x");
  }

  #[test]
  fn increment_multiple_times() {
    assert_eq!(interpret("x = 0; x++; x++; x++; x").unwrap(), "3");
  }

  #[test]
  fn postfix_increment_unset_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("freshIncA++").unwrap(), "freshIncA++");
  }

  #[test]
  fn prefix_increment_unset_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("++freshPreIncA").unwrap(), "++freshPreIncA");
  }

  #[test]
  fn postfix_decrement_unset_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("freshDecA--").unwrap(), "freshDecA--");
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

  #[test]
  fn decrement_real_value_matches_machine_precision() {
    // 1.6 - 1 in IEEE double is 0.6000000000000001 (matches wolframscript).
    assert_eq!(interpret("a = 1.6; a--; a").unwrap(), "0.6000000000000001");
  }
}

mod pre_increment_function {
  use super::*;

  #[test]
  fn pre_increment_returns_new_value() {
    assert_eq!(interpret("x = 5; ++x").unwrap(), "6");
  }

  #[test]
  fn pre_increment_symbolic_expression() {
    // y holds `x`; ++y adds 1 to it, yielding `1 + x`.
    assert_eq!(interpret("y = x; ++y").unwrap(), "1 + x");
  }

  #[test]
  fn pre_increment_real_then_add() {
    // a = 2.; after ++a, a holds 3.; 3. + 1.6 = 4.6.
    assert_eq!(interpret("a = 2.; ++a; a + 1.6").unwrap(), "4.6");
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

  #[test]
  fn pre_decrement_part() {
    assert_eq!(interpret("pos = {1, 2}; --pos[[1]]").unwrap(), "0");
    assert_eq!(interpret("pos = {10, 20}; --pos[[2]]").unwrap(), "19");
  }

  #[test]
  fn pre_increment_part() {
    assert_eq!(interpret("pos = {1, 2}; ++pos[[1]]").unwrap(), "2");
    assert_eq!(interpret("pos = {10, 20}; ++pos[[2]]").unwrap(), "21");
  }

  #[test]
  fn post_increment_part() {
    // Post-increment returns old value
    assert_eq!(interpret("pos = {1, 2}; pos[[1]]++").unwrap(), "1");
    assert_eq!(
      interpret("pos = {1, 2}; pos[[1]]++; pos").unwrap(),
      "{2, 2}"
    );
  }

  #[test]
  fn post_decrement_part() {
    // Post-decrement returns old value
    assert_eq!(interpret("pos = {1, 2}; pos[[1]]--").unwrap(), "1");
    assert_eq!(
      interpret("pos = {1, 2}; pos[[1]]--; pos").unwrap(),
      "{0, 2}"
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

mod traditional_form {
  use super::*;

  #[test]
  fn wraps_expression() {
    assert_eq!(
      interpret("TraditionalForm[x + y]").unwrap(),
      "TraditionalForm[x + y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[TraditionalForm[x]]").unwrap(),
      "TraditionalForm"
    );
  }

  #[test]
  fn evaluates_argument() {
    assert_eq!(
      interpret("TraditionalForm[1 + 2]").unwrap(),
      "TraditionalForm[3]"
    );
  }

  #[test]
  fn nested_expression() {
    assert_eq!(
      interpret("TraditionalForm[Sin[Pi/4]]").unwrap(),
      "TraditionalForm[1/Sqrt[2]]"
    );
  }

  #[test]
  fn to_string() {
    assert_eq!(
      interpret("ToString[TraditionalForm[x + y]]").unwrap(),
      "DisplayForm[FormBox[RowBox[{x, +, y}], TraditionalForm]]"
    );
  }

  #[test]
  fn to_string_evaluates() {
    assert_eq!(
      interpret("ToString[TraditionalForm[1 + 2]]").unwrap(),
      "DisplayForm[FormBox[3, TraditionalForm]]"
    );
  }

  #[test]
  fn polynomial() {
    assert_eq!(
      interpret("TraditionalForm[6 + 6 x^2 - 12 x]").unwrap(),
      "TraditionalForm[6 - 12*x + 6*x^2]"
    );
  }
}

mod batch_symbols {
  use super::*;

  #[test]
  fn vertex_labels() {
    assert_eq!(interpret("VertexLabels").unwrap(), "VertexLabels");
  }

  #[test]
  fn plot_theme() {
    assert_eq!(interpret("PlotTheme").unwrap(), "PlotTheme");
  }

  #[test]
  fn exclusions() {
    assert_eq!(interpret("Exclusions").unwrap(), "Exclusions");
  }

  #[test]
  fn center_dot() {
    assert_eq!(interpret("CenterDot").unwrap(), "CenterDot");
  }

  #[test]
  fn spacer() {
    assert_eq!(interpret("Spacer[10]").unwrap(), "Spacer[10]");
  }

  #[test]
  fn control_placement() {
    assert_eq!(interpret("ControlPlacement").unwrap(), "ControlPlacement");
  }

  #[test]
  fn item_size() {
    assert_eq!(interpret("ItemSize").unwrap(), "ItemSize");
  }

  #[test]
  fn tracked_symbols() {
    assert_eq!(interpret("TrackedSymbols").unwrap(), "TrackedSymbols");
  }

  #[test]
  fn plot_markers() {
    assert_eq!(interpret("PlotMarkers").unwrap(), "PlotMarkers");
  }

  #[test]
  fn mesh_functions() {
    assert_eq!(interpret("MeshFunctions").unwrap(), "MeshFunctions");
  }

  #[test]
  fn baseline() {
    assert_eq!(interpret("Baseline").unwrap(), "Baseline");
  }

  #[test]
  fn ticks_style() {
    assert_eq!(interpret("TicksStyle").unwrap(), "TicksStyle");
  }
}

mod thin_symbol {
  use super::*;

  #[test]
  fn evaluates_to_thickness_tiny() {
    assert_eq!(interpret("Thin").unwrap(), "Thickness[Tiny]");
  }
}

mod unit_system_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("UnitSystem").unwrap(), "UnitSystem");
  }
}

mod filling_style_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("FillingStyle").unwrap(), "FillingStyle");
  }
}

mod color_space_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("ColorSpace").unwrap(), "ColorSpace");
  }
}

mod image_padding_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("ImagePadding").unwrap(), "ImagePadding");
  }
}

mod quantity_variable_function {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("QuantityVariable[\"x\", \"Length\"]").unwrap(),
      "QuantityVariable[x, Length]"
    );
  }
}

mod interleaving_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("Interleaving").unwrap(), "Interleaving");
  }
}

mod interpolation_order_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(
      interpret("InterpolationOrder").unwrap(),
      "InterpolationOrder"
    );
  }
}

mod plot_range_padding_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("PlotRangePadding").unwrap(), "PlotRangePadding");
  }
}

mod plain_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("Plain").unwrap(), "Plain");
  }
}

mod distributed_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("Distributed").unwrap(), "Distributed");
  }
}

mod entity_property_function {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("EntityProperty[\"Country\", \"Population\"]").unwrap(),
      "EntityProperty[Country, Population]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[EntityProperty[x, y]]").unwrap(),
      "EntityProperty"
    );
  }
}

mod font_weight_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("FontWeight").unwrap(), "FontWeight");
  }
}

mod control_type_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("ControlType").unwrap(), "ControlType");
  }
}

mod labeled_function {
  use super::*;

  #[test]
  fn wraps_expression() {
    assert_eq!(
      interpret("Labeled[x, \"label\"]").unwrap(),
      "Labeled[x, label]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Labeled[x, y]]").unwrap(), "Labeled");
  }
}

mod entity_value_function {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("EntityValue[x, y]").unwrap(), "EntityValue[x, y]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[EntityValue[x, y]]").unwrap(), "EntityValue");
  }
}

mod item_function {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Item[x]").unwrap(), "Item[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Item[x]]").unwrap(), "Item");
  }

  #[test]
  fn with_options() {
    assert_eq!(
      interpret("Item[\"hello\", Background -> Red]").unwrap(),
      "Item[hello, Background -> RGBColor[1, 0, 0]]"
    );
  }
}

mod cell_function {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Cell[\"hello\"]").unwrap(), "Cell[hello]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Cell[\"hello\"]]").unwrap(), "Cell");
  }

  #[test]
  fn with_style() {
    assert_eq!(
      interpret("Cell[TextData[\"test\"], \"Input\"]").unwrap(),
      "Cell[TextData[test], Input]"
    );
  }
}

mod test_id_symbol {
  use super::*;

  #[test]
  fn evaluates_to_itself() {
    assert_eq!(interpret("TestID").unwrap(), "TestID");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TestID]").unwrap(), "Symbol");
  }
}

mod inherited_symbol {
  use super::*;

  #[test]
  fn inherited_evaluates_to_itself() {
    assert_eq!(interpret("Inherited").unwrap(), "Inherited");
  }

  #[test]
  fn inherited_head() {
    assert_eq!(interpret("Head[Inherited]").unwrap(), "Symbol");
  }

  #[test]
  fn inherited_identity() {
    assert_eq!(interpret("Inherited === Inherited").unwrap(), "True");
  }
}

mod off_function {
  use super::*;

  #[test]
  fn off_returns_null() {
    assert_eq!(interpret("Off[f]").unwrap(), "\0");
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
    assert_eq!(interpret("Remove[x]").unwrap(), "\0");
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
  fn set_options_returns_unevaluated() {
    // SetOptions is not implemented - returns unevaluated (matching wolframscript)
    assert_eq!(
      interpret("SetOptions[f, a -> 1]").unwrap(),
      "SetOptions[f, a -> 1]"
    );
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

  #[test]
  fn add_to_part() {
    clear_state();
    assert_eq!(
      interpret("x = {1, 2, 3}; x[[2]] += 9; x").unwrap(),
      "{1, 11, 3}"
    );
  }

  #[test]
  fn subtract_from_part() {
    clear_state();
    assert_eq!(
      interpret("x = {10, 20, 30}; x[[1]] -= 3; x").unwrap(),
      "{7, 20, 30}"
    );
  }

  #[test]
  fn times_by_part() {
    clear_state();
    assert_eq!(
      interpret("x = {2, 3, 4}; x[[3]] *= 5; x").unwrap(),
      "{2, 3, 20}"
    );
  }

  #[test]
  fn divide_by_part() {
    clear_state();
    assert_eq!(
      interpret("x = {10, 20, 30}; x[[2]] /= 4; x").unwrap(),
      "{10, 5, 30}"
    );
  }

  #[test]
  fn add_to_part_in_function_def() {
    // Parsing test: F[x_] := x[[2]] += 9 should parse without error
    clear_state();
    // FunctionDefinition returns "\0" (suppressed Null)
    assert!(interpret("F[x_] := x[[2]] += 9").is_ok());
  }

  #[test]
  fn add_to_rvalue_error() {
    // AddTo on uninitialized variable should return the variable unchanged
    clear_state();
    assert_eq!(
      interpret("AddTo[freshAddVar, 3]; freshAddVar").unwrap(),
      "freshAddVar"
    );
  }

  #[test]
  fn add_to_uninitialized_returns_unevaluated() {
    // Matches Mathematica: `a += 2` with unset `a` keeps the AddTo form.
    clear_state();
    assert_eq!(interpret("freshAddToA += 2").unwrap(), "freshAddToA += 2");
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

  #[test]
  fn subtract_from_rvalue_error() {
    clear_state();
    assert_eq!(
      interpret("SubtractFrom[freshSubVar, 2]; freshSubVar").unwrap(),
      "freshSubVar"
    );
  }

  #[test]
  fn subtract_from_uninitialized_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("freshSubA -= 2").unwrap(), "freshSubA -= 2");
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

  #[test]
  fn times_by_rvalue_error() {
    clear_state();
    assert_eq!(
      interpret("TimesBy[freshTimVar, 3]; freshTimVar").unwrap(),
      "freshTimVar"
    );
  }

  #[test]
  fn times_by_uninitialized_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("freshTimA *= 3").unwrap(), "freshTimA *= 3");
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

  #[test]
  fn divide_by_rvalue_error() {
    clear_state();
    assert_eq!(
      interpret("DivideBy[freshDivVar, 2]; freshDivVar").unwrap(),
      "freshDivVar"
    );
  }

  #[test]
  fn divide_by_uninitialized_returns_unevaluated() {
    clear_state();
    assert_eq!(interpret("freshDivA /= 2").unwrap(), "freshDivA /= 2");
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

  #[test]
  fn match_complex_pattern() {
    assert_eq!(interpret("MatchQ[2I, _Complex]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[4 - I, _Complex]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[I, _Complex]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[3 + 2I, _Complex]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[5, _Complex]").unwrap(), "False");
    assert_eq!(interpret("MatchQ[3.14, _Complex]").unwrap(), "False");
  }

  #[test]
  fn cases_complex_pattern() {
    assert_eq!(
      interpret("Cases[{1, 2I, 3, 4 - I, 5}, _Complex]").unwrap(),
      "{2*I, 4 - I}"
    );
  }

  #[test]
  fn depth_complex_is_atom() {
    assert_eq!(interpret("Depth[1 + 2 I]").unwrap(), "1");
    assert_eq!(interpret("Depth[3 + 4I]").unwrap(), "1");
    assert_eq!(interpret("Depth[I]").unwrap(), "1");
    assert_eq!(interpret("Depth[2I]").unwrap(), "1");
  }

  #[test]
  fn atom_q_complex() {
    assert_eq!(interpret("AtomQ[2 + I]").unwrap(), "True");
    assert_eq!(interpret("AtomQ[3 + 4I]").unwrap(), "True");
    assert_eq!(interpret("AtomQ[I]").unwrap(), "True");
    assert_eq!(
      interpret("Map[AtomQ, {2, 2.1, 1/2, 2 + I}]").unwrap(),
      "{True, True, True, True}"
    );
  }

  #[test]
  fn atom_q_with_base_literal() {
    // 2^^101 is the binary literal 5, which is an atom.
    assert_eq!(
      interpret("Map[AtomQ, {2, 2.1, 1/2, 2 + I, 2^^101}]").unwrap(),
      "{True, True, True, True, True}"
    );
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
  fn begin_returns_context() {
    // Begin returns the context string (matching wolframscript)
    assert_eq!(interpret("Begin[\"Private`\"]").unwrap(), "Private`");
  }

  #[test]
  fn end_returns_context() {
    // End[] returns the context that was set by Begin[]
    assert_eq!(interpret("Begin[\"Private`\"]; End[]").unwrap(), "Private`");
  }

  #[test]
  fn begin_package_returns_context() {
    assert_eq!(interpret("BeginPackage[\"MyPkg`\"]").unwrap(), "MyPkg`");
  }

  #[test]
  fn end_package_returns_null() {
    // EndPackage[] returns Null (matching wolframscript)
    assert_eq!(interpret("EndPackage[]").unwrap(), "\0");
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
    assert_eq!(interpret("x = 5; Unset[x]").unwrap(), "\0");
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

  #[test]
  fn unset_pattern_without_prior_def_returns_failed() {
    // Mathematica: 'f[x_] =.' with no prior definition emits an
    // Unset::norep warning and returns $Failed.
    clear_state();
    assert_eq!(interpret("freshUnsetF[x_] =.").unwrap(), "$Failed");
  }

  #[test]
  fn unset_pattern_after_set_delayed_returns_null() {
    clear_state();
    assert_eq!(
      interpret("freshUnsetG[x_] := x^2; freshUnsetG[x_] =.").unwrap(),
      "\0"
    );
  }

  #[test]
  fn unset_pattern_removes_definition() {
    // After 'f[x_] =.' the downvalue is gone and f[5] stays symbolic.
    clear_state();
    assert_eq!(
      interpret("freshUnsetH[x_] := x^2; freshUnsetH[x_] =.; freshUnsetH[5]")
        .unwrap(),
      "freshUnsetH[5]"
    );
  }

  #[test]
  fn unset_threads_over_list() {
    // '{a, {b}} =.' should thread, returning {Null, {Null}}.
    clear_state();
    assert_eq!(interpret("{a, {b}} =.").unwrap(), "{Null, {Null}}");
  }

  #[test]
  fn unset_threads_removes_each_ownvalue() {
    clear_state();
    assert_eq!(
      interpret("a = 1; b = 2; {a, {b}} =.; {a, b}").unwrap(),
      "{a, b}"
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

  #[test]
  fn double_question_mark_parses() {
    clear_state();
    let result = interpret("??Sin").unwrap();
    // ??symbol parses as Information[symbol, "Full"] which includes attributes
    assert!(result.contains("Attributes"));
    assert!(result.contains("FullName -> System`Sin"));
    assert!(result.contains("True]"));
  }

  #[test]
  fn single_question_mark_parses() {
    clear_state();
    let result = interpret("?Sin").unwrap();
    assert!(result.contains("Name -> Sin"));
    assert!(result.contains("False]"));
  }
}

mod message_function {
  use super::*;

  #[test]
  fn message_returns_unevaluated() {
    assert_eq!(
      interpret("Message[f, \"test\"]").unwrap(),
      "Message[f, test]"
    );
  }

  #[test]
  fn message_with_message_name_returns_null() {
    clear_state();
    assert_eq!(interpret("Message[freshMsgA::tag]").unwrap(), "\0");
  }

  #[test]
  fn message_with_defined_text_returns_null() {
    clear_state();
    assert_eq!(
      interpret("freshMsgB::tag = \"hi\"; Message[freshMsgB::tag]").unwrap(),
      "\0"
    );
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
  fn needs_returns_failed() {
    // Needs returns $Failed when package is not found (matching wolframscript)
    assert_eq!(interpret("Needs[\"SomePackage`\"]").unwrap(), "$Failed");
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

  // Regression tests for https://github.com/ad-si/Woxi/issues/83
  #[test]
  fn head_of_reciprocal_is_power() {
    assert_eq!(interpret("Head[1/x]").unwrap(), "Power");
    assert_eq!(interpret("Head[1/(2*x - 3)]").unwrap(), "Power");
  }

  #[test]
  fn head_of_reciprocal_via_variable() {
    clear_state();
    assert_eq!(interpret("y = 1/(2*x - 3); Head[y]").unwrap(), "Power");
  }

  #[test]
  fn head_of_symbolic_division_is_times() {
    assert_eq!(interpret("Head[2/x]").unwrap(), "Times");
    assert_eq!(interpret("Head[x/(2*y)]").unwrap(), "Times");
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

  #[test]
  fn optional_default_dot_syntax_parses() {
    // x_. is Optional[Pattern[x, Blank[]]] — system-determined default
    assert_eq!(interpret("x_.").unwrap(), "x_.");
  }

  #[test]
  fn optional_default_dot_with_head_is_syntax_error() {
    // x_Integer. is a syntax error in Wolfram Language (only x_. and _. are valid)
    assert!(interpret("x_Integer.").is_err());
  }

  #[test]
  fn optional_default_dot_anonymous_parses() {
    // _. is Optional[Blank[]] — anonymous system-determined default
    assert_eq!(interpret("_.").unwrap(), "_.");
  }

  #[test]
  fn optional_default_dot_in_expression() {
    // m_. can appear in expressions like Power patterns
    assert_eq!(interpret("x_^m_.").unwrap(), "(x_)^(m_.)");
  }

  #[test]
  fn optional_default_dot_in_function_definition() {
    // The original failing expression should parse without error
    assert_eq!(
      interpret(
        "Int[x_^m_., x_Symbol] := x^(m + 1)/(m + 1) /; FreeQ[m, x] && NeQ[m, -1]"
      )
      .unwrap(),
      "\0"
    );
  }
}

mod condition_operator {
  use super::*;

  #[test]
  fn condition_in_set_delayed() {
    // f[x_] := body /; condition
    assert_eq!(interpret("g[x_] := x^2 /; x > 0; g[3]").unwrap(), "9");
  }

  #[test]
  fn condition_in_set_delayed_rejects_when_false() {
    // When condition is false, definition should not match
    assert_eq!(interpret("g[x_] := x^2 /; x > 0; g[-3]").unwrap(), "g[-3]");
  }

  #[test]
  fn condition_with_multiple_conditions() {
    // Multiple conditions via &&
    assert_eq!(
      interpret("h[x_] := x + 1 /; x > 0 && x < 10; h[5]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("h[x_] := x + 1 /; x > 0 && x < 10; h[15]").unwrap(),
      "h[15]"
    );
  }

  #[test]
  fn chained_conditions_with_fallback() {
    // Chained /; with later unconditional definition — when the first
    // condition fails (3 < 2 is false), the fallback applies: y/x = 2/3.
    clear_state();
    assert_eq!(
      interpret(
        "F[x_, y_] /; x < y /; x>0 := x / y; \
         F[x_, y_] := y / x; \
         F[3, 2]"
      )
      .unwrap(),
      "2/3"
    );
  }

  #[test]
  fn chained_conditions_fallback_negative_arg() {
    // For F[-3, 2], first condition x>0 fails so fallback y/x = 2/-3 = -2/3.
    clear_state();
    assert_eq!(
      interpret(
        "F[x_, y_] /; x < y /; x>0 := x / y; \
         F[x_, y_] := y / x; \
         F[-3, 2]"
      )
      .unwrap(),
      "-2/3"
    );
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

  #[test]
  fn matrix_form_from_array_4x3() {
    assert_eq!(
      interpret("Array[a,{4,3}]//MatrixForm").unwrap(),
      "MatrixForm[{{a[1, 1], a[1, 2], a[1, 3]}, {a[2, 1], a[2, 2], a[2, 3]}, {a[3, 1], a[3, 2], a[3, 3]}, {a[4, 1], a[4, 2], a[4, 3]}}]"
    );
  }

  #[test]
  fn matrix_form_2x2_symbols() {
    assert_eq!(
      interpret("MatrixForm[{{a,b},{c,d}}]").unwrap(),
      "MatrixForm[{{a, b}, {c, d}}]"
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

  #[test]
  fn subsuperscript_stays_symbolic() {
    assert_eq!(
      interpret("Subsuperscript[a, p, q]").unwrap(),
      "Subsuperscript[a, p, q]"
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
  fn undefined_is_protected() {
    assert_eq!(interpret("Attributes[Undefined]").unwrap(), "{Protected}");
  }

  #[test]
  fn composition_attributes() {
    assert_eq!(
      interpret("Attributes[Composition]").unwrap(),
      "{Flat, OneIdentity, Protected}"
    );
  }

  #[test]
  fn hold_pattern_attributes() {
    assert_eq!(
      interpret("Attributes[HoldPattern]").unwrap(),
      "{HoldAll, Protected}"
    );
  }

  #[test]
  fn make_boxes_attributes() {
    assert_eq!(
      interpret("Attributes[MakeBoxes]").unwrap(),
      "{HoldAllComplete}"
    );
  }

  #[test]
  fn level_is_protected() {
    assert_eq!(interpret("Attributes[Level]").unwrap(), "{Protected}");
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
    assert_eq!(
      interpret("Attributes[True]").unwrap(),
      "{Locked, Protected}"
    );
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

  // Regression tests for `&` precedence: `&` is 90 in Wolfram, which is
  // tighter than Set (40) but looser than Equal (290) and all arithmetic
  // infix operators, so both of these bindings must be preserved.

  #[test]
  fn amp_binds_whole_equality_with_function_call_operands() {
    // Function call on both sides of == followed by `&`: the whole
    // equality must become the Function body.
    assert_eq!(
      interpret("Head[Mod[#1, 3] == Mod[#2, 3] &]").unwrap(),
      "Function"
    );
    assert_eq!(
      interpret("(Mod[#1, 3] == Mod[#2, 3] &)[1, 4]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("(Mod[#1, 3] == Mod[#2, 3] &)[1, 5]").unwrap(),
      "False"
    );
  }

  #[test]
  fn amp_binds_whole_inequality_with_function_call_operands() {
    assert_eq!(
      interpret("(Sort[#1] == Sort[#2] &)[{1, 2}, {2, 1}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn amp_binds_tighter_than_set_on_plain_body() {
    // `f = body &` must parse as Set[f, Function[body]], not
    // Function[Set[f, body]].
    assert_eq!(interpret("f = Sqrt[#] &; f[16]").unwrap(), "4");
  }

  #[test]
  fn amp_binds_tighter_than_set_delayed_on_plain_body() {
    assert_eq!(interpret("g := #^2 &; g[7]").unwrap(), "49");
  }

  #[test]
  fn amp_binds_tighter_than_set_with_function_call_body() {
    assert_eq!(interpret("h = Plus[##] &; h[1, 2, 3, 4]").unwrap(), "10");
  }

  #[test]
  fn destructuring_assignment() {
    assert_eq!(
      interpret("Clear[a, b, c]; {a, b, c} = {10, 2, 3}").unwrap(),
      "{10, 2, 3}"
    );
    assert_eq!(
      interpret("Clear[a, b, c]; {a, b, c} = {10, 2, 3}; {a, b, c}").unwrap(),
      "{10, 2, 3}"
    );
  }

  #[test]
  fn destructuring_assignment_nested() {
    assert_eq!(
      interpret(
        "Clear[a, b, c, d]; {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {3}}}; {a, b, c, d}"
      )
      .unwrap(),
      "{1, 2, {c1, c2}, 3}"
    );
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
    assert_eq!(interpret("CompoundExpression[]").unwrap(), "\0");
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
    // HoldForm prevents evaluation but displays without the wrapper
    assert_eq!(interpret("HoldForm[1 + 1]").unwrap(), "1 + 1");
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

  #[test]
  fn symbolic_fraction() {
    assert_eq!(interpret("Numerator[x/y]").unwrap(), "x");
  }

  #[test]
  fn multi_factor_fraction() {
    assert_eq!(interpret("Numerator[(a*b)/(c*d)]").unwrap(), "a*b");
  }

  #[test]
  fn mixed_rational_symbolic_fraction() {
    assert_eq!(interpret("Numerator[(3*x^2)/(7*y)]").unwrap(), "3*x^2");
  }

  #[test]
  fn fraction_with_power_in_denom() {
    assert_eq!(interpret("Numerator[a/b^2]").unwrap(), "a");
  }

  #[test]
  fn product_without_denominator() {
    assert_eq!(interpret("Numerator[a*b*c]").unwrap(), "a*b*c");
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

  #[test]
  fn symbolic_fraction() {
    assert_eq!(interpret("Denominator[x/y]").unwrap(), "y");
  }

  #[test]
  fn multi_factor_fraction() {
    assert_eq!(interpret("Denominator[(a*b)/(c*d)]").unwrap(), "c*d");
  }

  #[test]
  fn mixed_rational_symbolic_fraction() {
    assert_eq!(interpret("Denominator[(3*x^2)/(7*y)]").unwrap(), "7*y");
  }

  #[test]
  fn fraction_with_power_in_denom() {
    assert_eq!(interpret("Denominator[a/b^2]").unwrap(), "b^2");
  }

  #[test]
  fn product_without_denominator() {
    assert_eq!(interpret("Denominator[a*b*c]").unwrap(), "1");
  }
}

mod unknown_function_no_args {
  use super::*;

  #[test]
  fn undefined_symbol_called_with_no_args_stays_symbolic() {
    assert_eq!(interpret("A[]").unwrap(), "A[]");
  }

  #[test]
  fn undefined_symbol_with_blank_and_symbol_args_stays_symbolic() {
    assert_eq!(interpret("A[p_, q]").unwrap(), "A[p_, q]");
  }

  #[test]
  fn undefined_curried_subvalue_call_stays_symbolic() {
    assert_eq!(interpret("A[x][t]").unwrap(), "A[x][t]");
  }

  #[test]
  fn undefined_symbol_with_blank_arg_and_symbol_tag_stays_symbolic() {
    assert_eq!(interpret("S[x_, A]").unwrap(), "S[x_, A]");
  }

  #[test]
  fn undefined_symbol_with_blank_and_typed_blank_arg_stays_symbolic() {
    assert_eq!(interpret("S[x_, _A]").unwrap(), "S[x_, _A]");
  }

  #[test]
  fn typed_blank_called_with_no_args_stays_symbolic() {
    assert_eq!(interpret("_A[]").unwrap(), "_A[]");
  }

  #[test]
  fn typed_blank_stays_symbolic() {
    assert_eq!(interpret("_A").unwrap(), "_A");
  }

  #[test]
  fn bare_condition_on_undefined_symbol_stays_symbolic() {
    assert_eq!(interpret("A/;A>0").unwrap(), "A /; A > 0");
  }

  #[test]
  fn condition_on_undefined_function_call_stays_symbolic() {
    assert_eq!(interpret("A[p_, q]/;q>0").unwrap(), "A[p_, q] /; q > 0");
  }

  #[test]
  fn display_form_with_interpretation_box_pattern_stays_symbolic() {
    assert_eq!(
      interpret("DisplayForm[boxexpr_InterpretationBox]").unwrap(),
      "DisplayForm[boxexpr_InterpretationBox]"
    );
  }

  #[test]
  fn n_of_undefined_function_with_pattern_unwraps_n() {
    assert_eq!(interpret("N[A[s_]]").unwrap(), "A[s_]");
  }

  #[test]
  fn exponential_e_named_char_evaluates_to_e() {
    assert_eq!(interpret("\\[ExponentialE]").unwrap(), "E");
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
    // HoldForm prevents evaluation but displays without the wrapper
    assert_eq!(interpret("HoldForm[1 + 2 + 3]").unwrap(), "1 + 2 + 3");
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

  #[test]
  fn traditional_form_grid() {
    clear_state();
    assert_eq!(
      interpret("TraditionalForm[Grid[{{1, 2}, {3, 4}}, Frame -> All]]")
        .unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn traditional_form_grid_svg_content() {
    clear_state();
    let svg = interpret(
      "ExportString[TraditionalForm[Grid[{{a, b}, {c, d}}, Frame -> All]], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">a</text>"));
    assert!(svg.contains(">d</text>"));
    assert!(svg.contains("<line"), "Frame -> All should produce lines");
  }

  #[test]
  fn traditional_form_grid_with_table() {
    clear_state();
    assert_eq!(
      interpret(
        "f[x_] := x^2; values = Table[{i, f[i]}, {i, 1, 10, 1}]; PrependTo[values, {\"x\", \"x^2\"}]; TraditionalForm[Grid[values, Frame -> All]]"
      )
      .unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn traditional_form_list_renders_as_matrix() {
    clear_state();
    let result =
      interpret_with_stdout("TraditionalForm[{{1, 2}, {3, 4}}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    // Should render all elements
    for val in ["1", "2", "3", "4"] {
      assert!(
        svg.contains(&format!(">{val}</text>")),
        "Missing value {val} in TraditionalForm matrix SVG"
      );
    }
  }

  #[test]
  fn traditional_form_1d_list_renders_as_column_vector() {
    clear_state();
    let result = interpret_with_stdout("TraditionalForm[{a, b, c}]").unwrap();
    assert_eq!(result.result, "-Graphics-");
    let svg = result.graphics.unwrap();
    for val in ["a", "b", "c"] {
      assert!(
        svg.contains(&format!(">{val}</text>")),
        "Missing value {val} in TraditionalForm column SVG"
      );
    }
  }
}

mod text_grid {
  use super::*;

  #[test]
  fn basic_2x2() {
    clear_state();
    let svg = interpret(
      "ExportString[TextGrid[{{\"item 1\", \"item 2\"}, {\"item 3\", \"item 4\"}}], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">item 1</text>"));
    assert!(svg.contains(">item 4</text>"));
  }

  #[test]
  fn renders_as_graphics() {
    clear_state();
    assert_eq!(
      interpret(
        "TextGrid[{{\"item 1\", \"item 2\"}, {\"item 3\", \"item 4\"}}, Frame -> All]"
      )
      .unwrap(),
      "-Graphics-"
    );
  }

  #[test]
  fn frame_all() {
    clear_state();
    let svg = interpret(
      "ExportString[TextGrid[{{\"a\", \"b\"}, {\"c\", \"d\"}}, Frame -> All], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<line"), "Frame -> All should produce lines");
  }

  #[test]
  fn with_numeric_data() {
    clear_state();
    let svg = interpret(
      "ExportString[TextGrid[{{1, 2}, {3, 4}}, Frame -> All], \"SVG\"]",
    )
    .unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">1</text>"));
    assert!(svg.contains(">4</text>"));
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
  fn tag_set_returns_rhs() {
    // TagSet returns the evaluated RHS (unlike TagSetDelayed which returns Null)
    clear_state();
    assert_eq!(interpret("g /: f[g[x_]] = 1 + 2").unwrap(), "3");
  }

  #[test]
  fn tag_set_functional_form_returns_rhs() {
    // TagSet[tag, lhs, rhs] also returns evaluated RHS
    clear_state();
    assert_eq!(interpret("TagSet[g, f[g[x_]], 1 + 2]").unwrap(), "3");
  }

  #[test]
  fn tag_set_attributes() {
    assert_eq!(
      interpret("Attributes[TagSet]").unwrap(),
      "{HoldAll, Protected, SequenceHold}"
    );
  }

  #[test]
  fn tag_set_delayed_attributes() {
    assert_eq!(
      interpret("Attributes[TagSetDelayed]").unwrap(),
      "{HoldAll, Protected, SequenceHold}"
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
    assert_eq!(interpret("g /: f[g[x_]] := fg[x]").unwrap(), "\0");
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

  #[test]
  fn binary_op_plus_upvalue() {
    // Dist /: Dist[u_,v_]+Dist[w_,v_] := Dist[u+w,v]
    // LHS is a BinaryOp (Plus), not a FunctionCall
    clear_state();
    assert_eq!(
      interpret(
        "Dist /: Dist[u_,v_]+Dist[w_,v_] := Dist[u+w,v]; Dist[a,b]+Dist[c,b]"
      )
      .unwrap(),
      "Dist[a + c, b]"
    );
  }

  #[test]
  fn binary_op_plus_upvalue_no_match() {
    // When the repeated pattern variable doesn't match, the rule should not fire
    clear_state();
    assert_eq!(
      interpret(
        "Dist /: Dist[u_,v_]+Dist[w_,v_] := Dist[u+w,v]; Dist[a,b]+Dist[c,d]"
      )
      .unwrap(),
      "Dist[a, b] + Dist[c, d]"
    );
  }

  #[test]
  fn binary_op_plus_upvalue_numeric() {
    // Numeric arguments should also work with upvalue on Plus
    clear_state();
    assert_eq!(
      interpret(
        "Dist /: Dist[u_,v_]+Dist[w_,v_] := Dist[u+w,v]; Dist[1,x]+Dist[2,x]"
      )
      .unwrap(),
      "Dist[3, x]"
    );
  }

  #[test]
  fn binary_op_times_upvalue() {
    // Upvalue on Times (BinaryOp::Times)
    clear_state();
    assert_eq!(
      interpret("Foo /: Foo[x_] * Foo[y_] := Foo[x * y]; Foo[a] * Foo[b]")
        .unwrap(),
      "Foo[a*b]"
    );
  }

  #[test]
  fn binary_op_upvalue_normal_arith_unaffected() {
    // Normal arithmetic should not be affected by upvalue definitions
    clear_state();
    assert_eq!(
      interpret("Dist /: Dist[u_,v_]+Dist[w_,v_] := Dist[u+w,v]; 2 + 3")
        .unwrap(),
      "5"
    );
  }

  #[test]
  fn upvalues_display_basic() {
    // UpValues should display the original pattern and body
    clear_state();
    assert_eq!(
      interpret("f /: g[f[x_]] := x^2; UpValues[f]").unwrap(),
      "{HoldPattern[g[f[x_]]] :> x^2}"
    );
  }

  #[test]
  fn upvalues_display_multi_arg() {
    clear_state();
    assert_eq!(
      interpret("f /: g[f[x_], y_] := x^2 + y; UpValues[f]").unwrap(),
      "{HoldPattern[g[f[x_], y_]] :> x^2 + y}"
    );
  }

  #[test]
  fn upvalues_display_multiple_rules() {
    clear_state();
    assert_eq!(
      interpret("f /: g[f[x_]] := x^2; f /: h[f[y_]] := y + 1; UpValues[f]")
        .unwrap(),
      "{HoldPattern[g[f[x_]]] :> x^2, HoldPattern[h[f[y_]]] :> y + 1}"
    );
  }

  #[test]
  fn upvalues_display_binary_op() {
    clear_state();
    assert_eq!(
      interpret("f /: f + g := fg; UpValues[f]").unwrap(),
      "{HoldPattern[f + g] :> fg}"
    );
  }

  #[test]
  fn upvalues_empty() {
    clear_state();
    assert_eq!(interpret("UpValues[x]").unwrap(), "{}");
  }

  #[test]
  fn upvalue_literal_symbol_plus() {
    // x /: x + y_ := f[y] — x is a literal symbol, y_ is a pattern
    clear_state();
    assert_eq!(
      interpret("ClearAll[x,y,f]; x /: x + y_ := f[y]; x + 1").unwrap(),
      "f[1]"
    );
  }

  #[test]
  fn upvalue_literal_symbol_plus_symbolic() {
    // Symbolic argument should also work
    clear_state();
    assert_eq!(
      interpret("ClearAll[x,y,f,a]; x /: x + y_ := f[y]; x + a").unwrap(),
      "f[a]"
    );
  }

  #[test]
  fn upvalue_with_condition_on_lhs() {
    // x /: x + y_ /; y > -2 := f[y] — condition on the LHS
    clear_state();
    assert_eq!(
      interpret("ClearAll[x,y,f]; x /: x + y_ /; y > -2 := f[y]; x + 1")
        .unwrap(),
      "f[1]"
    );
  }

  #[test]
  fn upvalue_with_condition_no_match() {
    // Condition not satisfied — rule should not fire
    clear_state();
    assert_eq!(
      interpret("ClearAll[x,y,f]; x /: x + y_ /; y > 5 := f[y]; x + 1")
        .unwrap(),
      "1 + x"
    );
  }

  #[test]
  fn upvalue_multiple_conditions_ordering() {
    // Multiple upvalue rules with conditions — first matching rule wins
    clear_state();
    assert_eq!(
      interpret(
        "x /: x + y_ /; y > -2 := f[y]; \
         x /: x + y_ /; y < 2 := g[y]; \
         {x + 1, x + (-3), x + 5}"
      )
      .unwrap(),
      "{f[1], g[-3], f[5]}"
    );
  }

  #[test]
  fn upvalue_condition_on_body() {
    // Condition can also be on the body side (rhs /;)
    clear_state();
    assert_eq!(
      interpret(
        "ClearAll[x,y,f]; x /: x + y_ := f[y] /; y > 0; {x + 1, x + (-1)}"
      )
      .unwrap(),
      "{f[1], -1 + x}"
    );
  }

  #[test]
  fn upvalues_display_with_condition() {
    // UpValues display should include the condition
    clear_state();
    assert_eq!(
      interpret("ClearAll[x,y,f]; x /: x + y_ /; y > -2 := f[y]; UpValues[x]")
        .unwrap(),
      "{HoldPattern[x + (y_) /; y > -2] :> f[y]}"
    );
  }

  #[test]
  fn upvalue_redefinition_replaces() {
    // Redefining an upvalue with the same LHS should replace, not duplicate
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := x^2; g /: f[g[x_]] := x^3; UpValues[g]")
        .unwrap(),
      "{HoldPattern[f[g[x_]]] :> x^3}"
    );
    // The new definition should be used for evaluation
    assert_eq!(interpret("f[g[3]]").unwrap(), "27");
  }

  #[test]
  fn hold_pattern_prevents_plus_evaluation() {
    // HoldPattern keeps `x + x` from evaluating to `2*x`.
    assert_eq!(
      interpret("HoldPattern[x + x]").unwrap(),
      "HoldPattern[x + x]"
    );
  }

  #[test]
  fn hold_pattern_is_transparent_for_matching() {
    // ReplaceAll should treat HoldPattern[x] -> t identically to x -> t.
    assert_eq!(interpret("x /. HoldPattern[x] -> t").unwrap(), "t");
  }
}

mod tag_unset {
  use super::*;

  #[test]
  fn basic_tag_unset_syntax() {
    // g /: f[g[x_]] =. removes the upvalue
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := x^2; g /: f[g[x_]] =.; f[g[3]]").unwrap(),
      "f[g[3]]"
    );
  }

  #[test]
  fn tag_unset_clears_upvalues() {
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := x^2; g /: f[g[x_]] =.; UpValues[g]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn tag_unset_functional_form() {
    // TagUnset[g, f[g[x_]]] as functional form
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := x^2; TagUnset[g, f[g[x_]]]; f[g[3]]")
        .unwrap(),
      "f[g[3]]"
    );
  }

  #[test]
  fn tag_unset_preserves_other_upvalues() {
    // Removing one upvalue should not affect others
    clear_state();
    assert_eq!(
      interpret(
        "g /: f[g[x_]] := x^2; g /: h[g[x_]] := x + 1; g /: f[g[x_]] =.; h[g[5]]"
      )
      .unwrap(),
      "6"
    );
  }

  #[test]
  fn tag_unset_with_tag_set() {
    // TagUnset should also remove TagSet (not just TagSetDelayed) definitions
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] = x^2; g /: f[g[x_]] =.; f[g[3]]").unwrap(),
      "f[g[3]]"
    );
  }

  #[test]
  fn tag_unset_returns_null() {
    // TagUnset should suppress output (return Null)
    clear_state();
    assert_eq!(
      interpret("g /: f[g[x_]] := x^2; g /: f[g[x_]] =.").unwrap(),
      "\0"
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
  fn upset_with_binary_op_lhs_returns_rhs() {
    // 'a + b ^= 2' parses as UpSet[a+b, 2]; the Plus LHS should normalize
    // to Plus[a, b] and UpSet should return 2.
    clear_state();
    assert_eq!(interpret("a + b ^= 2").unwrap(), "2");
  }

  #[test]
  fn upset_with_binary_op_lhs_applies_rule() {
    clear_state();
    assert_eq!(interpret("a + b ^= 2; a + b").unwrap(), "2");
  }

  #[test]
  fn upset_with_binary_op_lhs_stores_upvalue_on_a() {
    clear_state();
    assert_eq!(
      interpret("a + b ^= 2; UpValues[a]").unwrap(),
      "{HoldPattern[a + b] :> 2}"
    );
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
    assert_eq!(interpret("f[g] ^:= 1 + 2").unwrap(), "\0");
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

mod name_q {
  use super::*;

  #[test]
  fn builtin_symbol() {
    assert_eq!(interpret("NameQ[\"Plus\"]").unwrap(), "True");
  }

  #[test]
  fn undefined_symbol() {
    assert_eq!(interpret("NameQ[\"asdfNotDefined\"]").unwrap(), "False");
  }

  #[test]
  fn user_defined() {
    assert_eq!(interpret("x = 5; NameQ[\"x\"]").unwrap(), "True");
  }

  #[test]
  fn attributes() {
    assert_eq!(interpret("Attributes[NameQ]").unwrap(), "{Protected}");
  }
}

mod share {
  use super::*;

  #[test]
  fn returns_zero() {
    assert_eq!(interpret("Share[x]").unwrap(), "0");
  }

  #[test]
  fn no_args_returns_zero() {
    assert_eq!(interpret("Share[]").unwrap(), "0");
  }

  #[test]
  fn attributes() {
    assert_eq!(interpret("Attributes[Share]").unwrap(), "{Protected}");
  }
}

mod delimiters {
  use super::*;

  #[test]
  fn evaluates_to_self() {
    assert_eq!(interpret("Delimiters").unwrap(), "Delimiters");
  }

  #[test]
  fn attributes() {
    assert_eq!(interpret("Attributes[Delimiters]").unwrap(), "{Protected}");
  }
}

mod precedence_form {
  use super::*;

  #[test]
  fn evaluates_to_self() {
    assert_eq!(
      interpret("PrecedenceForm[x + y, 10]").unwrap(),
      "PrecedenceForm[x + y, 10]"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[PrecedenceForm]").unwrap(),
      "{Protected}"
    );
  }
}

mod skeleton {
  use super::*;

  #[test]
  fn displays_as_angle_brackets() {
    assert_eq!(interpret("Skeleton[5]").unwrap(), "<<5>>");
  }

  #[test]
  fn displays_with_one() {
    assert_eq!(interpret("Skeleton[1]").unwrap(), "<<1>>");
  }

  #[test]
  fn displays_with_ten() {
    assert_eq!(interpret("Skeleton[10]").unwrap(), "<<10>>");
  }

  #[test]
  fn no_args_returns_unevaluated() {
    assert_eq!(interpret("Skeleton[]").unwrap(), "Skeleton[]");
  }

  #[test]
  fn attributes() {
    assert_eq!(interpret("Attributes[Skeleton]").unwrap(), "{}");
  }
}

mod string_skeleton {
  use super::*;

  #[test]
  fn displays_as_angle_brackets() {
    assert_eq!(interpret("StringSkeleton[5]").unwrap(), "<<5>>");
  }

  #[test]
  fn displays_with_string() {
    assert_eq!(interpret("StringSkeleton[\"abc\"]").unwrap(), "<<abc>>");
  }

  #[test]
  fn no_args_returns_unevaluated() {
    assert_eq!(interpret("StringSkeleton[]").unwrap(), "StringSkeleton[]");
  }

  #[test]
  fn no_builtin_attributes() {
    assert_eq!(interpret("Attributes[StringSkeleton]").unwrap(), "{}");
  }
}

mod total_width {
  use super::*;

  #[test]
  fn evaluates_to_self() {
    assert_eq!(interpret("TotalWidth").unwrap(), "TotalWidth");
  }

  #[test]
  fn attributes() {
    assert_eq!(interpret("Attributes[TotalWidth]").unwrap(), "{Protected}");
  }
}

mod unevaluated {
  use super::*;

  #[test]
  fn holds_argument() {
    assert_eq!(
      interpret("Unevaluated[1 + 2]").unwrap(),
      "Unevaluated[1 + 2]"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Unevaluated]").unwrap(),
      "{HoldAllComplete, Protected}"
    );
  }
}

mod v2_option_symbols {
  use super::*;

  #[test]
  fn word_attributes() {
    assert_eq!(interpret("Attributes[Word]").unwrap(), "{Protected}");
  }

  #[test]
  fn frame_attributes() {
    assert_eq!(interpret("Attributes[Frame]").unwrap(), "{Protected}");
  }

  #[test]
  fn background_attributes() {
    assert_eq!(interpret("Attributes[Background]").unwrap(), "{Protected}");
  }

  #[test]
  fn axes_style_attributes() {
    assert_eq!(interpret("Attributes[AxesStyle]").unwrap(), "{Protected}");
  }

  #[test]
  fn color_function_attributes() {
    assert_eq!(
      interpret("Attributes[ColorFunction]").unwrap(),
      "{Protected}"
    );
  }

  #[test]
  fn axes_origin_attributes() {
    assert_eq!(interpret("Attributes[AxesOrigin]").unwrap(), "{Protected}");
  }

  #[test]
  fn frame_style_attributes() {
    assert_eq!(interpret("Attributes[FrameStyle]").unwrap(), "{Protected}");
  }

  #[test]
  fn grid_lines_attributes() {
    assert_eq!(interpret("Attributes[GridLines]").unwrap(), "{Protected}");
  }

  #[test]
  fn epilog_attributes() {
    assert_eq!(interpret("Attributes[Epilog]").unwrap(), "{Protected}");
  }

  #[test]
  fn frame_ticks_attributes() {
    assert_eq!(interpret("Attributes[FrameTicks]").unwrap(), "{Protected}");
  }

  #[test]
  fn absolute_point_size_attributes() {
    assert_eq!(
      interpret("Attributes[AbsolutePointSize]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod tableform_headings {
  use super::*;

  #[test]
  fn tableform_with_options_stays_symbolic() {
    // In text mode, TableForm with options stays symbolic (evaluated args)
    assert_eq!(
      interpret("TableForm[{{1, 2}, {3, 4}}, TableHeadings -> {{\"a\", \"b\"}, {\"x\", \"y\"}}]").unwrap(),
      "TableForm[{{1, 2}, {3, 4}}, TableHeadings -> {{a, b}, {x, y}}]"
    );
  }

  #[test]
  fn tableform_single_arg_stays_symbolic() {
    // TableForm with just data stays symbolic in text mode
    assert_eq!(
      interpret("TableForm[{{1, 2}, {3, 4}}]").unwrap(),
      "TableForm[{{1, 2}, {3, 4}}]"
    );
  }
}

mod boolean_table {
  use super::*;

  #[test]
  fn or_two_vars() {
    assert_eq!(
      interpret("BooleanTable[p || q, {p, q}]").unwrap(),
      "{True, True, True, False}"
    );
  }

  #[test]
  fn and_two_vars() {
    assert_eq!(
      interpret("BooleanTable[p && q, {p, q}]").unwrap(),
      "{True, False, False, False}"
    );
  }

  #[test]
  fn not_single_var() {
    assert_eq!(
      interpret("BooleanTable[Not[p], {p}]").unwrap(),
      "{False, True}"
    );
  }

  #[test]
  fn implies_two_vars() {
    assert_eq!(
      interpret("BooleanTable[Implies[p, q], {p, q}]").unwrap(),
      "{True, False, True, True}"
    );
  }

  #[test]
  fn xor_two_vars() {
    assert_eq!(
      interpret("BooleanTable[Xor[p, q], {p, q}]").unwrap(),
      "{False, True, True, False}"
    );
  }

  #[test]
  fn equivalent_two_vars() {
    assert_eq!(
      interpret("BooleanTable[Equivalent[p, q], {p, q}]").unwrap(),
      "{True, False, False, True}"
    );
  }

  #[test]
  fn and_three_vars() {
    assert_eq!(
      interpret("BooleanTable[p && q && r, {p, q, r}]").unwrap(),
      "{True, False, False, False, False, False, False, False}"
    );
  }

  #[test]
  fn constant_true() {
    assert_eq!(
      interpret("BooleanTable[True, {p, q}]").unwrap(),
      "{True, True, True, True}"
    );
  }
}

mod framed {
  use super::*;

  #[test]
  fn framed_symbolic() {
    assert_eq!(interpret("Framed[x]").unwrap(), "Framed[x]");
  }

  #[test]
  fn framed_evaluates_args() {
    assert_eq!(interpret("Framed[1 + 2]").unwrap(), "Framed[3]");
  }

  #[test]
  fn nestlist_framed() {
    assert_eq!(
      interpret("NestList[Framed, x, 3]").unwrap(),
      "{x, Framed[x], Framed[Framed[x]], Framed[Framed[Framed[x]]]}"
    );
  }
}

mod plus_rendering {
  use super::*;

  #[test]
  fn negative_times_coefficient() {
    // Regression test: Plus with negative Times coefficient should use " - "
    assert_eq!(
      interpret("Plus[1, Times[-2, x], Power[x, 2]]").unwrap(),
      "1 - 2*x + x^2"
    );
  }

  #[test]
  fn negative_integer_term() {
    assert_eq!(interpret("1 + (-3) + x").unwrap(), "-2 + x");
  }

  #[test]
  fn multiple_negative_terms() {
    assert_eq!(interpret("a - 3*b - 5*c").unwrap(), "a - 3*b - 5*c");
  }
}

mod tilde_infix {
  use super::*;

  #[test]
  fn basic_tilde_infix() {
    // a ~f~ b means f[a, b]
    assert_eq!(interpret("a ~f~ b").unwrap(), "f[a, b]");
  }

  #[test]
  fn tilde_infix_evaluates() {
    assert_eq!(interpret("1 ~Plus~ 2").unwrap(), "3");
  }

  #[test]
  fn tilde_infix_join() {
    assert_eq!(
      interpret("{1, 2, 3} ~Join~ {4, 5}").unwrap(),
      "{1, 2, 3, 4, 5}"
    );
  }

  #[test]
  fn tilde_infix_left_associative() {
    // a ~f~ b ~g~ c means g[f[a, b], c]
    assert_eq!(
      interpret("FullForm[Hold[a ~f~ b ~g~ c]]").unwrap(),
      "Hold[g[f[a, b], c]]"
    );
  }

  #[test]
  fn tilde_infix_precedence_plus() {
    // ~f~ binds tighter than +
    assert_eq!(
      interpret("FullForm[Hold[a + b ~f~ c]]").unwrap(),
      "Hold[Plus[a, f[b, c]]]"
    );
  }

  #[test]
  fn tilde_infix_precedence_times() {
    // ~f~ binds tighter than *
    assert_eq!(
      interpret("FullForm[Hold[a * b ~f~ c]]").unwrap(),
      "Hold[Times[a, f[b, c]]]"
    );
  }

  #[test]
  fn tilde_infix_precedence_power() {
    // ~f~ binds tighter than ^ (right-associative)
    assert_eq!(
      interpret("FullForm[Hold[a^2 ~f~ c]]").unwrap(),
      "Hold[Power[a, f[2, c]]]"
    );
  }

  #[test]
  fn tilde_infix_precedence_prefix_at() {
    // @ binds tighter than ~f~
    assert_eq!(
      interpret("FullForm[Hold[g @ a ~f~ b]]").unwrap(),
      "Hold[f[g[a], b]]"
    );
  }

  #[test]
  fn tilde_infix_precedence_apply() {
    // ~f~ binds tighter than @@
    assert_eq!(
      interpret("FullForm[Hold[g @@ a ~f~ b]]").unwrap(),
      "Hold[Apply[g, f[a, b]]]"
    );
  }

  #[test]
  fn tilde_infix_does_not_conflict_with_string_expression() {
    // ~~ (StringExpression) should still work
    assert_eq!(
      interpret(r#"FullForm[Hold["a" ~~ "b"]]"#).unwrap(),
      r#"Hold[StringExpression["a", "b"]]"#
    );
  }

  #[test]
  fn tilde_infix_caesar_cipher() {
    // End-to-end test from the issue file
    assert_eq!(
      interpret(r#"caesarDecode[text_, n_] := StringReplace[text, Thread[CharacterRange["A","Z"] -> RotateLeft[CharacterRange["A","Z"], -n]] ~Join~ Thread[CharacterRange["a","z"] -> RotateLeft[CharacterRange["a","z"], -n]]]; caesarDecode["Khoor Zruog", 3]"#).unwrap(),
      "Hello World"
    );
  }
}

mod line_continuation {
  use super::*;

  #[test]
  fn backslash_newline_in_definition() {
    // Backslash at end of line continues the expression on the next line
    assert_eq!(interpret("f[x_] :=\\\n  x^2\nf[5]").unwrap(), "25");
  }

  #[test]
  fn backslash_newline_in_expression() {
    assert_eq!(interpret("1 +\\\n2 +\\\n3").unwrap(), "6");
  }

  #[test]
  fn backslash_newline_preserves_function_def() {
    assert_eq!(
      interpret(
        "ImaginaryQ[u_] :=\\\n  Head[u]===Complex && Re[u]===0\nImaginaryQ[3 I]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn backslash_newline_not_in_strings() {
    // Backslash inside strings should NOT be treated as line continuation
    assert_eq!(interpret(r#""hello\nworld""#).unwrap(), r#"hello\nworld"#);
  }
}

mod structural_pattern_consistency {
  use super::*;

  #[test]
  fn structural_binding_must_not_conflict_with_positional() {
    // Issue #73: x_ in structural pattern matched -1, but x positionally is Symbol x.
    // The pattern should NOT match because of the inconsistent binding for x.
    assert_eq!(
      interpret(
        "f[g_^(a_.+b_.*x_), x_Symbol] := {g,a,b,x} /; FreeQ[{a,b,g},x]; f[y^-1, x]"
      )
      .unwrap(),
      "f[y^(-1), x]"
    );
  }

  #[test]
  fn structural_binding_consistent_with_positional() {
    // When structural pattern variables don't conflict with positional params,
    // the match should succeed normally.
    assert_eq!(
      interpret(
        "g[f_^(a_.+b_.*y_), x_Symbol] := {f,a,b,y,x} /; FreeQ[{a,b,f,y},x]; g[z^(2+3*w), x]"
      )
      .unwrap(),
      "{z, 3*w, 1, 2, x}"
    );
  }

  #[test]
  fn integrate_pattern_no_false_match() {
    // The original issue case: Int[1/(Sqrt[x]*(a+b*x)), x] should not match
    // a rule where x in the structural pattern binds to -1.
    assert_eq!(
      interpret(
        "Int[f_^(a_.+b_.*x_), x_Symbol] := {f,a,b,x} /; FreeQ[{a,b,f},x]; Int[1/(Sqrt[x]*(a+b*x)), x]"
      )
      .unwrap(),
      "Int[1/(Sqrt[x]*(a + b*x)), x]"
    );
  }
}

mod two_way_rule {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("TwoWayRule[a, b]").unwrap(), "a <-> b");
  }

  #[test]
  fn numeric() {
    assert_eq!(interpret("TwoWayRule[1, 2]").unwrap(), "1 <-> 2");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TwoWayRule[a, b]]").unwrap(), "TwoWayRule");
  }
}

mod batch_inert_symbols_2 {
  use super::*;

  #[test]
  fn dividers() {
    assert_eq!(interpret("Dividers[x]").unwrap(), "Dividers[x]");
  }

  #[test]
  fn locator() {
    assert_eq!(interpret("Locator[x]").unwrap(), "Locator[x]");
  }

  #[test]
  fn input_field() {
    assert_eq!(interpret("InputField[x]").unwrap(), "InputField[x]");
  }

  #[test]
  fn region_function() {
    assert_eq!(interpret("RegionFunction[x]").unwrap(), "RegionFunction[x]");
  }

  #[test]
  fn color_function_scaling() {
    assert_eq!(
      interpret("ColorFunctionScaling[x]").unwrap(),
      "ColorFunctionScaling[x]"
    );
  }

  #[test]
  fn initialization() {
    assert_eq!(interpret("Initialization[x]").unwrap(), "Initialization[x]");
  }

  #[test]
  fn save_definitions() {
    assert_eq!(
      interpret("SaveDefinitions[x]").unwrap(),
      "SaveDefinitions[x]"
    );
  }

  #[test]
  fn around() {
    assert_eq!(interpret("Around[5, 0.3]").unwrap(), "Around[5., 0.3]");
  }

  #[test]
  fn specularity() {
    assert_eq!(
      interpret("Specularity[White, 10]").unwrap(),
      "Specularity[GrayLevel[1], 10]"
    );
  }

  #[test]
  fn status_area() {
    assert_eq!(interpret("StatusArea[x, y]").unwrap(), "StatusArea[x, y]");
  }

  #[test]
  fn pane() {
    assert_eq!(interpret("Pane[x]").unwrap(), "Pane[x]");
  }

  #[test]
  fn plot_labels() {
    assert_eq!(interpret("PlotLabels[x]").unwrap(), "PlotLabels[x]");
  }

  #[test]
  fn inactive() {
    assert_eq!(
      interpret("Inactive[Plus][2, 3]").unwrap(),
      "Inactive[Plus][2, 3]"
    );
  }

  #[test]
  fn geo_position() {
    assert_eq!(
      interpret("GeoPosition[{40, -74}]").unwrap(),
      "GeoPosition[{40, -74}]"
    );
  }

  #[test]
  fn baseline_position() {
    assert_eq!(
      interpret("BaselinePosition[x]").unwrap(),
      "BaselinePosition[x]"
    );
  }

  #[test]
  fn image_scaled() {
    assert_eq!(
      interpret("ImageScaled[{0.5, 0.5}]").unwrap(),
      "ImageScaled[{0.5, 0.5}]"
    );
  }

  #[test]
  fn dirichlet_condition() {
    assert_eq!(
      interpret("DirichletCondition[u[x] == 0, x == 0]").unwrap(),
      "DirichletCondition[u[x] == 0, x == 0]"
    );
  }

  #[test]
  fn boundary_style() {
    assert_eq!(interpret("BoundaryStyle[x]").unwrap(), "BoundaryStyle[x]");
  }

  #[test]
  fn entity_class() {
    assert_eq!(interpret("EntityClass[x, y]").unwrap(), "EntityClass[x, y]");
  }

  #[test]
  fn default_label_style() {
    assert_eq!(
      interpret("DefaultLabelStyle[x]").unwrap(),
      "DefaultLabelStyle[x]"
    );
  }
}

mod rotation_matrix {
  use super::*;

  #[test]
  fn symbolic_2d() {
    assert_eq!(
      interpret("RotationMatrix[theta]").unwrap(),
      "{{Cos[theta], -Sin[theta]}, {Sin[theta], Cos[theta]}}"
    );
  }

  #[test]
  fn pi_over_4() {
    assert_eq!(
      interpret("RotationMatrix[Pi/4]").unwrap(),
      "{{1/Sqrt[2], -(1/Sqrt[2])}, {1/Sqrt[2], 1/Sqrt[2]}}"
    );
  }

  #[test]
  fn pi_over_2_3d() {
    assert_eq!(
      interpret("RotationMatrix[Pi/2, {0, 0, 1}]").unwrap(),
      "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"
    );
  }
}

mod batch_inert_symbols_3 {
  use super::*;

  #[test]
  fn performance_goal() {
    assert_eq!(
      interpret("PerformanceGoal[x]").unwrap(),
      "PerformanceGoal[x]"
    );
  }

  #[test]
  fn vertex_list() {
    assert_eq!(interpret("VertexList[x]").unwrap(), "VertexList[x]");
  }

  #[test]
  fn chart_labels() {
    assert_eq!(interpret("ChartLabels[x]").unwrap(), "ChartLabels[x]");
  }

  #[test]
  fn text_cell() {
    assert_eq!(interpret("TextCell[x]").unwrap(), "TextCell[x]");
  }

  #[test]
  fn plot_range_clipping() {
    assert_eq!(
      interpret("PlotRangeClipping[x]").unwrap(),
      "PlotRangeClipping[x]"
    );
  }

  #[test]
  fn rotation_transform() {
    assert_eq!(
      interpret("RotationTransform[x]").unwrap(),
      "TransformationFunction[{{Cos[x], -Sin[x], 0}, {Sin[x], Cos[x], 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn data_range() {
    assert_eq!(interpret("DataRange[x]").unwrap(), "DataRange[x]");
  }

  #[test]
  fn cell_baseline() {
    assert_eq!(interpret("CellBaseline[x]").unwrap(), "CellBaseline[x]");
  }

  #[test]
  fn animation_running() {
    assert_eq!(
      interpret("AnimationRunning[x]").unwrap(),
      "AnimationRunning[x]"
    );
  }

  #[test]
  fn selected_notebook() {
    assert_eq!(
      interpret("SelectedNotebook[x]").unwrap(),
      "SelectedNotebook[x]"
    );
  }

  #[test]
  fn geometric_transformation() {
    assert_eq!(
      interpret("GeometricTransformation[x, y]").unwrap(),
      "GeometricTransformation[x, y]"
    );
  }

  #[test]
  fn cloud_export() {
    assert_eq!(interpret("CloudExport[x]").unwrap(), "CloudExport[x]");
  }
}

mod right_composition {
  use super::*;

  #[test]
  fn display_two_args() {
    assert_eq!(interpret("RightComposition[f, g]").unwrap(), "f /* g");
  }

  #[test]
  fn display_three_args() {
    assert_eq!(
      interpret("RightComposition[f, g, h]").unwrap(),
      "f /* g /* h"
    );
  }

  #[test]
  fn apply_two_functions() {
    assert_eq!(interpret("RightComposition[f, g][x]").unwrap(), "g[f[x]]");
  }

  #[test]
  fn apply_three_functions() {
    assert_eq!(
      interpret("RightComposition[f, g, h][x]").unwrap(),
      "h[g[f[x]]]"
    );
  }

  #[test]
  fn single_function() {
    assert_eq!(interpret("RightComposition[f][x]").unwrap(), "f[x]");
  }

  #[test]
  fn empty_composition() {
    assert_eq!(interpret("RightComposition[][x]").unwrap(), "x");
  }

  #[test]
  fn with_numeric_values() {
    assert_eq!(
      interpret("RightComposition[# + 1 &, #^2 &][3]").unwrap(),
      "16"
    );
  }
}

mod composition_operator_parsing {
  use super::*;

  #[test]
  fn at_star_basic() {
    assert_eq!(interpret("f @* g").unwrap(), "f @* g");
  }

  #[test]
  fn at_star_apply() {
    assert_eq!(interpret("(f @* g)[x]").unwrap(), "f[g[x]]");
  }

  #[test]
  fn at_star_three_functions() {
    assert_eq!(interpret("(f @* g @* h)[x]").unwrap(), "f[g[h[x]]]");
  }

  #[test]
  fn at_star_with_builtins() {
    assert_eq!(interpret("(StringLength @* ToString)[12345]").unwrap(), "5");
  }

  #[test]
  fn slash_star_basic() {
    assert_eq!(interpret("f /* g").unwrap(), "f /* g");
  }

  #[test]
  fn slash_star_apply() {
    assert_eq!(interpret("(f /* g)[x]").unwrap(), "g[f[x]]");
  }

  #[test]
  fn slash_star_three_functions() {
    assert_eq!(interpret("(f /* g /* h)[x]").unwrap(), "h[g[f[x]]]");
  }

  #[test]
  fn slash_star_with_pure_functions() {
    assert_eq!(interpret("((# + 1 &) /* (#^2 &))[3]").unwrap(), "16");
  }
}

mod parallel_table {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("ParallelTable[i^2, {i, 5}]").unwrap(),
      "{1, 4, 9, 16, 25}"
    );
  }

  #[test]
  fn multi_dim() {
    assert_eq!(
      interpret("ParallelTable[i + j, {i, 2}, {j, 2}]").unwrap(),
      "{{2, 3}, {3, 4}}"
    );
  }
}

mod sinc {
  use super::*;

  #[test]
  fn sinc_zero() {
    assert_eq!(interpret("Sinc[0]").unwrap(), "1");
  }

  #[test]
  fn sinc_pi_half() {
    assert_eq!(interpret("Sinc[Pi/2]").unwrap(), "2/Pi");
  }

  #[test]
  fn sinc_pi() {
    assert_eq!(interpret("Sinc[Pi]").unwrap(), "0");
  }

  #[test]
  fn sinc_symbolic() {
    assert_eq!(interpret("Sinc[x]").unwrap(), "Sinc[x]");
  }

  #[test]
  fn sinc_numeric() {
    assert_eq!(interpret("Sinc[1.0]").unwrap(), "0.8414709848078965");
  }
}

mod reim {
  use super::*;

  #[test]
  fn reim_complex() {
    assert_eq!(interpret("ReIm[3 + 4 I]").unwrap(), "{3, 4}");
  }

  #[test]
  fn reim_real() {
    assert_eq!(interpret("ReIm[5]").unwrap(), "{5, 0}");
  }

  #[test]
  fn reim_pure_imaginary() {
    assert_eq!(interpret("ReIm[3 I]").unwrap(), "{0, 3}");
  }
}

mod complex_expand {
  use super::*;

  #[test]
  fn sin_complex() {
    // ComplexExpand[Sin[x + I*y]] = Cosh[y]*Sin[x] + I*Cos[x]*Sinh[y]
    let result = interpret("ComplexExpand[Sin[x + I*y]]").unwrap();
    // Both term orderings are valid
    assert!(
      result == "Cosh[y]*Sin[x] + I*Cos[x]*Sinh[y]"
        || result == "I*Cos[x]*Sinh[y] + Cosh[y]*Sin[x]",
      "Got: {}",
      result
    );
  }

  #[test]
  fn cos_complex() {
    assert_eq!(
      interpret("ComplexExpand[Cos[x + I*y]]").unwrap(),
      "Cos[x]*Cosh[y] - I*Sin[x]*Sinh[y]"
    );
  }

  #[test]
  fn exp_complex() {
    assert_eq!(
      interpret("ComplexExpand[Exp[x + I*y]]").unwrap(),
      "E^x*Cos[y] + I*E^x*Sin[y]"
    );
  }

  #[test]
  fn abs_complex() {
    assert_eq!(
      interpret("ComplexExpand[Abs[x + I*y]]").unwrap(),
      "Sqrt[x^2 + y^2]"
    );
  }
}

mod abs_arg {
  use super::*;

  #[test]
  fn abs_arg_complex() {
    assert_eq!(interpret("AbsArg[1 + I]").unwrap(), "{Sqrt[2], Pi/4}");
  }

  #[test]
  fn abs_arg_positive_real() {
    assert_eq!(interpret("AbsArg[2]").unwrap(), "{2, 0}");
  }

  #[test]
  fn abs_arg_negative_real() {
    assert_eq!(interpret("AbsArg[-3]").unwrap(), "{3, Pi}");
  }

  #[test]
  fn abs_arg_pure_imaginary() {
    assert_eq!(interpret("AbsArg[I]").unwrap(), "{1, Pi/2}");
  }

  #[test]
  fn abs_arg_negative_imaginary() {
    assert_eq!(interpret("AbsArg[-I]").unwrap(), "{1, -1/2*Pi}");
  }

  #[test]
  fn abs_arg_zero() {
    assert_eq!(interpret("AbsArg[0]").unwrap(), "{0, 0}");
  }

  #[test]
  fn abs_arg_float() {
    assert_eq!(interpret("AbsArg[3.5]").unwrap(), "{3.5, 0}");
    assert_eq!(interpret("AbsArg[-2.5]").unwrap(), "{2.5, Pi}");
  }
}

mod characteristic_polynomial {
  use super::*;

  #[test]
  fn two_by_two() {
    assert_eq!(
      interpret("CharacteristicPolynomial[{{a, b}, {c, d}}, x]").unwrap(),
      "-(b*c) + a*d - a*x - d*x + x^2"
    );
  }

  #[test]
  fn identity_matrix() {
    assert_eq!(
      interpret("CharacteristicPolynomial[{{1, 0}, {0, 1}}, x]").unwrap(),
      "1 - 2*x + x^2"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("CharacteristicPolynomial[{{2, 1}, {0, 3}}, x]").unwrap(),
      "6 - 5*x + x^2"
    );
  }
}

mod boolean_minimize {
  use super::*;

  #[test]
  fn minimize_true() {
    clear_state();
    assert_eq!(interpret("BooleanMinimize[True]").unwrap(), "True");
  }

  #[test]
  fn minimize_false() {
    clear_state();
    assert_eq!(interpret("BooleanMinimize[False]").unwrap(), "False");
  }

  #[test]
  fn minimize_tautology() {
    clear_state();
    assert_eq!(interpret("BooleanMinimize[a || Not[a]]").unwrap(), "True");
  }

  #[test]
  fn minimize_contradiction() {
    clear_state();
    assert_eq!(interpret("BooleanMinimize[a && Not[a]]").unwrap(), "False");
  }

  #[test]
  fn minimize_absorption() {
    clear_state();
    assert_eq!(interpret("BooleanMinimize[a || (a && b)]").unwrap(), "a");
  }

  #[test]
  fn minimize_complementary() {
    clear_state();
    // (a && b) || (a && !b) → a
    assert_eq!(
      interpret("BooleanMinimize[And[a, b] || And[a, Not[b]]]").unwrap(),
      "a"
    );
  }

  #[test]
  fn minimize_extract_b() {
    clear_state();
    // (a && b) || (!a && b) → b
    assert_eq!(
      interpret("BooleanMinimize[And[a, b] || And[Not[a], b]]").unwrap(),
      "b"
    );
  }

  #[test]
  fn minimize_identity() {
    clear_state();
    assert_eq!(interpret("BooleanMinimize[a && b]").unwrap(), "a && b");
  }

  #[test]
  fn minimize_implies() {
    clear_state();
    // Implies[a, b] → !a || b
    let result = interpret("BooleanMinimize[Implies[a, b]]").unwrap();
    assert!(
      result == " !a || b" || result == "b ||  !a",
      "Got: {}",
      result
    );
  }

  #[test]
  fn minimize_xor() {
    clear_state();
    // Xor[a, b] → (a && !b) || (!a && b)
    let result = interpret("BooleanMinimize[Xor[a, b]]").unwrap();
    assert!(result.contains("&&"), "Got: {}", result);
    assert!(result.contains("||"), "Got: {}", result);
  }

  #[test]
  fn minimize_single_var() {
    clear_state();
    assert_eq!(interpret("BooleanMinimize[a || a]").unwrap(), "a");
  }

  #[test]
  fn minimize_single_not_var() {
    clear_state();
    assert_eq!(
      interpret("BooleanMinimize[Not[a] || Not[a]]").unwrap(),
      " !a"
    );
  }
}

mod recursion_limit {
  use super::*;

  #[test]
  fn mutually_recursive_protected_symbol_rules_no_stack_overflow() {
    // Regression test for https://github.com/ad-si/Woxi/issues/99
    // Mutually recursive rules on protected symbols caused stack overflow
    clear_state();
    let result = interpret(
      "Unprotect[ArcSec, ArcCos]; \
       ArcCos[1/u_] := ArcSec[u]; \
       ArcSec[1/u_] := ArcCos[u]; \
       f[ArcSec[x_]] := 0",
    );
    // Should not stack overflow — returns Null from SetDelayed
    assert!(result.is_ok());
  }

  #[test]
  fn mutually_recursive_rules_symbolic_arg() {
    // ArcSec[y] with mutual recursion should not stack overflow
    clear_state();
    let result = interpret(
      "Unprotect[ArcSec, ArcCos]; \
       ArcCos[1/u_] := ArcSec[u]; \
       ArcSec[1/u_] := ArcCos[u]; \
       ArcSec[y]",
    );
    assert!(result.is_ok());
    // Should return ArcSec[y] unevaluated (recursion limit prevents infinite loop)
    assert_eq!(result.unwrap(), "ArcSec[y]");
  }

  #[test]
  fn concrete_value_still_works_with_recursive_rules() {
    // Concrete numeric values should still evaluate correctly
    clear_state();
    let result = interpret(
      "Unprotect[ArcSec, ArcCos]; \
       ArcCos[1/u_] := ArcSec[u]; \
       ArcSec[1/u_] := ArcCos[u]; \
       ArcSec[2]",
    );
    assert_eq!(result.unwrap(), "Pi/3");
  }
}

mod unicode_operators {
  use super::*;

  #[test]
  fn less_equal() {
    assert_eq!(interpret("3 ≤ 5").unwrap(), "True");
    assert_eq!(interpret("5 ≤ 3").unwrap(), "False");
    assert_eq!(interpret("3 ≤ 3").unwrap(), "True");
    assert_eq!(interpret("x ≤ 5").unwrap(), "x <= 5");
  }

  #[test]
  fn greater_equal() {
    assert_eq!(interpret("5 ≥ 3").unwrap(), "True");
    assert_eq!(interpret("3 ≥ 5").unwrap(), "False");
    assert_eq!(interpret("3 ≥ 3").unwrap(), "True");
    assert_eq!(interpret("x ≥ 5").unwrap(), "x >= 5");
  }

  #[test]
  fn equal_unicode() {
    assert_eq!(interpret("1 ⩵ 1").unwrap(), "True");
    assert_eq!(interpret("1 ⩵ 2").unwrap(), "False");
    assert_eq!(interpret("x ⩵ 5").unwrap(), "x == 5");
  }

  #[test]
  fn not_equal() {
    assert_eq!(interpret("3 ≠ 5").unwrap(), "True");
    assert_eq!(interpret("3 ≠ 3").unwrap(), "False");
    assert_eq!(interpret("x ≠ 5").unwrap(), "x != 5");
  }

  #[test]
  fn rule_arrow() {
    assert_eq!(interpret("{1, 2, 3} /. x_ → x^2").unwrap(), "{1, 4, 9}");
    assert_eq!(interpret("a → b").unwrap(), "a -> b");
  }

  #[test]
  fn infinity_symbol() {
    assert_eq!(interpret("∞").unwrap(), "Infinity");
    assert_eq!(interpret("∞ + 1").unwrap(), "Infinity");
    assert_eq!(interpret("-∞").unwrap(), "-Infinity");
  }

  #[test]
  fn rule_in_association() {
    assert_eq!(interpret("<|\"a\" → 1, \"b\" → 2|>[\"a\"]").unwrap(), "1");
  }

  #[test]
  fn comparison_chain() {
    assert_eq!(interpret("1 ≤ 2 ≤ 3").unwrap(), "True");
    assert_eq!(interpret("1 ≤ 3 ≥ 2").unwrap(), "True");
  }
}

// Regression tests for `&` (Function) precedence. `&` has very low precedence
// in Wolfram — it binds looser than any infix operator. Inner-term rules must
// not consume the `&` greedily when there are preceding operators, otherwise
// `a + b &` would mis-parse as `a + (b &)` instead of `(a + b) &`.
mod anonymous_function_precedence {
  use super::*;

  #[test]
  fn amp_after_function_call_with_logical_operator() {
    // `# > 10 && PrimeQ[#] &[11]` should parse as
    // `Function[And[Greater[#, 10], PrimeQ[#]]][11]` = True.
    assert_eq!(interpret("# > 10 && PrimeQ[#] &[11]").unwrap(), "True");
    assert_eq!(
      interpret("FullForm[Hold[# > 10 && PrimeQ[#] &[11]]]").unwrap(),
      "Hold[Function[And[Greater[Slot[1], 10], PrimeQ[Slot[1]]]][11]]"
    );
  }

  #[test]
  fn amp_after_function_call_with_plus() {
    // `a + PrimeQ[#] &[11]` should parse as `Function[a + PrimeQ[#]][11]`.
    assert_eq!(
      interpret("FullForm[Hold[a + PrimeQ[#] &[11]]]").unwrap(),
      "Hold[Function[Plus[a, PrimeQ[Slot[1]]]][11]]"
    );
  }

  #[test]
  fn amp_after_list_with_plus() {
    // `a + {1, 2} &[5]` should parse as `Function[a + {1, 2}][5]`.
    assert_eq!(
      interpret("FullForm[Hold[a + {1, 2} &[5]]]").unwrap(),
      "Hold[Function[Plus[a, List[1, 2]]][5]]"
    );
  }

  #[test]
  fn amp_after_parenthesized_with_plus() {
    // `a + (b) &[5]` should parse as `Function[a + b][5]`.
    assert_eq!(
      interpret("FullForm[Hold[a + (b) &[5]]]").unwrap(),
      "Hold[Function[Plus[a, b]][5]]"
    );
  }

  #[test]
  fn amp_after_part_extract_with_plus() {
    // `a + f[{1,2,3}][[1]] &[5]` should wrap the whole Plus.
    assert_eq!(
      interpret("FullForm[Hold[a + f[{1,2,3}][[1]] &[5]]]").unwrap(),
      "Hold[Function[Plus[a, Part[f[List[1, 2, 3]], 1]]][5]]"
    );
  }

  #[test]
  fn amp_after_slot_part_extract_with_plus() {
    // `a + #[[1]] &[{5, 6}]` should wrap the whole Plus.
    assert_eq!(
      interpret("FullForm[Hold[a + #[[1]] &[{5, 6}]]]").unwrap(),
      "Hold[Function[Plus[a, Part[Slot[1], 1]]][List[5, 6]]]"
    );
  }

  #[test]
  fn simple_direct_call_forms_still_work() {
    // Standalone `f[x] &[y]` cases (no preceding operator) still work.
    assert_eq!(interpret("PrimeQ[#] &[11]").unwrap(), "True");
    assert_eq!(interpret("(# + 1 &)[5]").unwrap(), "6");
    assert_eq!(interpret("{#, #^2} &[3]").unwrap(), "{3, 9}");
  }

  #[test]
  fn amp_inside_select() {
    // The predicate form used with Select: full function body wrapped.
    assert_eq!(
      interpret("Select[Range[20], # > 10 && PrimeQ[#] &]").unwrap(),
      "{11, 13, 17, 19}"
    );
  }
}
