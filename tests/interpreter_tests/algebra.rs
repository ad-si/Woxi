use super::*;

mod polynomial_q {
  use super::*;

  #[test]
  fn basic_polynomial() {
    assert_eq!(interpret("PolynomialQ[x^2 + 1, x]").unwrap(), "True");
    assert_eq!(
      interpret("PolynomialQ[3*x^3 + 2*x + 1, x]").unwrap(),
      "True"
    );
  }

  #[test]
  fn constant_is_polynomial() {
    assert_eq!(interpret("PolynomialQ[5, x]").unwrap(), "True");
  }

  #[test]
  fn variable_is_polynomial() {
    assert_eq!(interpret("PolynomialQ[x, x]").unwrap(), "True");
  }

  #[test]
  fn non_polynomial() {
    assert_eq!(interpret("PolynomialQ[Sin[x], x]").unwrap(), "False");
    assert_eq!(interpret("PolynomialQ[1/x, x]").unwrap(), "False");
  }

  #[test]
  fn multivariate() {
    assert_eq!(interpret("PolynomialQ[x^2 + y, x]").unwrap(), "True");
  }
}

mod exponent {
  use super::*;

  #[test]
  fn basic_exponent() {
    assert_eq!(interpret("Exponent[x^3 + x, x]").unwrap(), "3");
    assert_eq!(interpret("Exponent[x^2 + 3*x + 2, x]").unwrap(), "2");
  }

  #[test]
  fn constant_exponent() {
    assert_eq!(interpret("Exponent[5, x]").unwrap(), "0");
  }

  #[test]
  fn linear_exponent() {
    assert_eq!(interpret("Exponent[3*x + 1, x]").unwrap(), "1");
  }
}

mod coefficient {
  use super::*;

  #[test]
  fn quadratic_coefficients() {
    assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 2]").unwrap(), "1");
    assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 1]").unwrap(), "3");
    assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 0]").unwrap(), "2");
  }

  #[test]
  fn default_power_is_one() {
    assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x]").unwrap(), "3");
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("Coefficient[a*x^2 + b*x + c, x, 2]").unwrap(),
      "a"
    );
    assert_eq!(
      interpret("Coefficient[a*x^2 + b*x + c, x, 1]").unwrap(),
      "b"
    );
    assert_eq!(
      interpret("Coefficient[a*x^2 + b*x + c, x, 0]").unwrap(),
      "c"
    );
  }

  #[test]
  fn zero_coefficient() {
    assert_eq!(interpret("Coefficient[x^2 + 1, x, 1]").unwrap(), "0");
  }
}

mod expand {
  use super::*;

  #[test]
  fn simple_product() {
    assert_eq!(
      interpret("Expand[(x + 1)*(x + 2)]").unwrap(),
      "2 + 3*x + x^2"
    );
  }

  #[test]
  fn square() {
    assert_eq!(interpret("Expand[(x + 1)^2]").unwrap(), "1 + 2*x + x^2");
  }

  #[test]
  fn cube() {
    assert_eq!(
      interpret("Expand[(x + 1)^3]").unwrap(),
      "1 + 3*x + 3*x^2 + x^3"
    );
  }

  #[test]
  fn distribute() {
    assert_eq!(interpret("Expand[x*(x + 1)]").unwrap(), "x + x^2");
  }

  #[test]
  fn already_expanded() {
    assert_eq!(interpret("Expand[x^2 + 3*x + 2]").unwrap(), "2 + 3*x + x^2");
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("Expand[5]").unwrap(), "5");
  }

  #[test]
  fn difference_of_squares() {
    assert_eq!(interpret("Expand[(x + 2)*(x - 2)]").unwrap(), "-4 + x^2");
  }

  #[test]
  fn multivariate_two_vars() {
    assert_eq!(interpret("Expand[(x + y)^2]").unwrap(), "x^2 + 2*x*y + y^2");
  }

  #[test]
  fn multivariate_four_vars() {
    assert_eq!(
      interpret("Expand[(a + b)*(c + d)]").unwrap(),
      "a*c + b*c + a*d + b*d"
    );
  }

  #[test]
  fn multivariate_with_constant() {
    assert_eq!(
      interpret("Expand[(x + y + 1)^2]").unwrap(),
      "1 + 2*x + x^2 + 2*y + 2*x*y + y^2"
    );
  }
}

mod simplify {
  use super::*;

  #[test]
  fn combine_like_terms() {
    assert_eq!(interpret("Simplify[x + x]").unwrap(), "2*x");
  }

  #[test]
  fn combine_powers() {
    assert_eq!(interpret("Simplify[x*x]").unwrap(), "x^2");
  }

  #[test]
  fn cancel_division() {
    assert_eq!(interpret("Simplify[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
  }

  #[test]
  fn trivial() {
    assert_eq!(interpret("Simplify[5]").unwrap(), "5");
    assert_eq!(interpret("Simplify[x]").unwrap(), "x");
  }
}

mod factor {
  use super::*;

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("Factor[x^2 + 3*x + 2]").unwrap(),
      "(1 + x)*(2 + x)"
    );
  }

  #[test]
  fn difference_of_squares() {
    assert_eq!(interpret("Factor[x^2 - 4]").unwrap(), "(-2 + x)*(2 + x)");
  }

  #[test]
  fn with_common_factor() {
    assert_eq!(
      interpret("Factor[2*x^2 + 6*x + 4]").unwrap(),
      "2*(1 + x)*(2 + x)"
    );
  }

  #[test]
  fn irreducible() {
    assert_eq!(interpret("Factor[x^2 + 1]").unwrap(), "1 + x^2");
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("Factor[x^3 - 1]").unwrap(),
      "(-1 + x)*(1 + x + x^2)"
    );
  }

  #[test]
  fn linear() {
    assert_eq!(interpret("Factor[2*x + 4]").unwrap(), "2*(2 + x)");
  }

  #[test]
  fn cyclotomic_x6_minus_1() {
    assert_eq!(
      interpret("Factor[x^6 - 1]").unwrap(),
      "(-1 + x)*(1 + x)*(1 - x + x^2)*(1 + x + x^2)"
    );
  }

  #[test]
  fn cyclotomic_x12_minus_1() {
    assert_eq!(
      interpret("Factor[x^12 - 1]").unwrap(),
      "(-1 + x)*(1 + x)*(1 + x^2)*(1 - x + x^2)*(1 + x + x^2)*(1 - x^2 + x^4)"
    );
  }

  #[test]
  fn cyclotomic_x100_minus_1() {
    assert_eq!(
      interpret("Factor[x^100 - 1]").unwrap(),
      "(-1 + x)*(1 + x)*(1 + x^2)*(1 - x + x^2 - x^3 + x^4)*(1 + x + x^2 + x^3 + x^4)*(1 - x^2 + x^4 - x^6 + x^8)*(1 - x^5 + x^10 - x^15 + x^20)*(1 + x^5 + x^10 + x^15 + x^20)*(1 - x^10 + x^20 - x^30 + x^40)"
    );
  }

  #[test]
  fn irreducible_x4_plus_1() {
    assert_eq!(interpret("Factor[x^4 + 1]").unwrap(), "1 + x^4");
  }

  #[test]
  fn cyclotomic_x4_minus_1() {
    assert_eq!(
      interpret("Factor[x^4 - 1]").unwrap(),
      "(-1 + x)*(1 + x)*(1 + x^2)"
    );
  }
}

mod cancel {
  use super::*;

  #[test]
  fn cancel_simple() {
    assert_eq!(interpret("Cancel[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
  }

  #[test]
  fn cancel_cubic() {
    assert_eq!(interpret("Cancel[(x^3 - x)/(x^2 - 1)]").unwrap(), "x");
  }

  #[test]
  fn cancel_symbolic_common_factor() {
    assert_eq!(interpret("Cancel[(a*b)/(a*c)]").unwrap(), "b/c");
  }

  #[test]
  fn cancel_symbolic_powers() {
    assert_eq!(interpret("Cancel[(a^2*b)/(a*b^2)]").unwrap(), "a/b");
  }

  #[test]
  fn cancel_numeric_content() {
    assert_eq!(interpret("Cancel[(2*x)/(4*x)]").unwrap(), "1/2");
  }

  #[test]
  fn cancel_mixed_symbolic_and_poly() {
    assert_eq!(interpret("Cancel[(a*b*x)/(a*c*x^2)]").unwrap(), "b/(c*x)");
  }

  #[test]
  fn cancel_quadratic() {
    assert_eq!(
      interpret("Cancel[(x^2 + 2*x + 1)/(x + 1)]").unwrap(),
      "1 + x"
    );
  }
}

mod expand_all {
  use super::*;

  #[test]
  fn expand_all_basic() {
    assert_eq!(
      interpret("ExpandAll[x*(x + 1)^2]").unwrap(),
      "x + 2*x^2 + x^3"
    );
  }
}

mod collect_tests {
  use super::*;

  #[test]
  fn collect_basic() {
    assert_eq!(interpret("Collect[x*y + x*z, x]").unwrap(), "x*(y + z)");
  }

  #[test]
  fn collect_symbolic_coefficients() {
    // Coefficients with variables before the collect variable go first
    assert_eq!(
      interpret("Collect[a*x^2 + b*x + c*x^2 + d*x, x]").unwrap(),
      "(b + d)*x + (a + c)*x^2"
    );
  }

  #[test]
  fn collect_mixed_coefficients() {
    // Coefficient (2+y) has y > x alphabetically, so x goes first
    assert_eq!(
      interpret("Collect[x*y + 2*x + 3*y + 6, x]").unwrap(),
      "6 + 3*y + x*(2 + y)"
    );
  }

  #[test]
  fn collect_with_constant_term() {
    assert_eq!(
      interpret("Collect[a*x^2 + b*x^2 + c*x + d*x + e, x]").unwrap(),
      "e + (c + d)*x + (a + b)*x^2"
    );
  }
}

mod together {
  use super::*;

  #[test]
  fn together_basic() {
    assert_eq!(interpret("Together[1/x + 1/y]").unwrap(), "(x + y)/(x*y)");
  }

  #[test]
  fn together_symbolic_fractions() {
    assert_eq!(
      interpret("Together[a/b + c/d]").unwrap(),
      "(b*c + a*d)/(b*d)"
    );
  }

  #[test]
  fn together_subtracted_fractions() {
    assert_eq!(
      interpret("Together[1/(x-1) - 1/(x+1)]").unwrap(),
      "2/((-1 + x)*(1 + x))"
    );
  }

  #[test]
  fn together_added_fractions_with_binomial_denominators() {
    assert_eq!(
      interpret("Together[1/(x-1) + 1/(x+1)]").unwrap(),
      "(2*x)/((-1 + x)*(1 + x))"
    );
  }
}

mod apart {
  use super::*;

  #[test]
  fn apart_basic() {
    assert_eq!(
      interpret("Apart[1/(x^2 - 1)]").unwrap(),
      "1/(2*(-1 + x)) - 1/(2*(1 + x))"
    );
  }

  #[test]
  fn apart_x2_plus_1_over_x3_minus_x() {
    assert_eq!(
      interpret("Apart[(x^2 + 1)/(x^3 - x)]").unwrap(),
      "(-1 + x)^(-1) - x^(-1) + (1 + x)^(-1)"
    );
  }

  #[test]
  fn apart_two_linear_factors() {
    assert_eq!(
      interpret("Apart[1/((x - 1)*(x - 2))]").unwrap(),
      "(-2 + x)^(-1) - (-1 + x)^(-1)"
    );
  }

  #[test]
  fn apart_three_linear_factors() {
    assert_eq!(
      interpret("Apart[1/((x-1)*(x-2)*(x-3))]").unwrap(),
      "1/(2*(-3 + x)) - (-2 + x)^(-1) + 1/(2*(-1 + x))"
    );
  }
}

mod switch {
  use super::*;

  #[test]
  fn basic_match() {
    assert_eq!(interpret("Switch[2, 1, a, 2, b, 3, c]").unwrap(), "b");
  }

  #[test]
  fn first_match() {
    assert_eq!(interpret("Switch[1, 1, a, 2, b, 3, c]").unwrap(), "a");
  }

  #[test]
  fn no_match_returns_unevaluated() {
    assert_eq!(
      interpret("Switch[4, 1, a, 2, b, 3, c]").unwrap(),
      "Switch[4, 1, a, 2, b, 3, c]"
    );
  }

  #[test]
  fn wildcard_match() {
    assert_eq!(interpret("Switch[4, 1, a, _, c]").unwrap(), "c");
  }

  #[test]
  fn evaluated_expression() {
    assert_eq!(interpret("Switch[1 + 1, 1, a, 2, b, 3, c]").unwrap(), "b");
  }
}

mod piecewise {
  use super::*;

  #[test]
  fn first_true() {
    assert_eq!(interpret("Piecewise[{{1, True}}]").unwrap(), "1");
  }

  #[test]
  fn second_true() {
    assert_eq!(
      interpret("Piecewise[{{1, False}, {2, True}}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn default_value() {
    assert_eq!(
      interpret("Piecewise[{{1, False}, {2, False}}, 42]").unwrap(),
      "42"
    );
  }

  #[test]
  fn no_match_default_zero() {
    assert_eq!(interpret("Piecewise[{{1, False}}]").unwrap(), "0");
  }

  #[test]
  fn with_conditions() {
    clear_state();
    assert_eq!(
      interpret("x = 5; Piecewise[{{1, x < 0}, {2, x >= 0}}]").unwrap(),
      "2"
    );
  }
}

mod match_q {
  use super::*;

  #[test]
  fn head_matching() {
    assert_eq!(interpret("MatchQ[{1, 2, 3}, _List]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[42, _Integer]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[3.14, _Real]").unwrap(), "True");
    assert_eq!(interpret(r#"MatchQ["hello", _String]"#).unwrap(), "True");
  }

  #[test]
  fn head_mismatch() {
    assert_eq!(interpret("MatchQ[1, _String]").unwrap(), "False");
    assert_eq!(interpret("MatchQ[1, _List]").unwrap(), "False");
    assert_eq!(interpret(r#"MatchQ["x", _Integer]"#).unwrap(), "False");
  }

  #[test]
  fn blank_matches_anything() {
    assert_eq!(interpret("MatchQ[42, _]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[{1, 2}, _]").unwrap(), "True");
    assert_eq!(interpret(r#"MatchQ["x", _]"#).unwrap(), "True");
  }

  #[test]
  fn literal_matching() {
    assert_eq!(interpret("MatchQ[42, 42]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[42, 43]").unwrap(), "False");
    assert_eq!(interpret("MatchQ[x, x]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[x, y]").unwrap(), "False");
  }
}

mod replace_all_after_operators {
  use super::*;

  #[test]
  fn replace_all_after_plus() {
    assert_eq!(interpret("x + y /. x -> 1").unwrap(), "1 + y");
  }

  #[test]
  fn replace_all_after_times() {
    assert_eq!(interpret("x * y /. x -> 2").unwrap(), "2*y");
  }

  #[test]
  fn replace_all_after_power() {
    assert_eq!(interpret("x^2 + y /. x -> 3").unwrap(), "9 + y");
  }

  #[test]
  fn replace_all_multiple_operators() {
    assert_eq!(interpret("x + y + z /. {x -> 1, y -> 2}").unwrap(), "3 + z");
  }

  #[test]
  fn replace_all_times_multiple_vars() {
    assert_eq!(interpret("x * y * z /. {x -> 2, y -> 3}").unwrap(), "6*z");
  }

  #[test]
  fn replace_all_with_implicit_times() {
    assert_eq!(interpret("2 x + 3 y /. {x -> 1, y -> 2}").unwrap(), "8");
  }

  #[test]
  fn replace_repeated_after_plus() {
    assert_eq!(interpret("x + y //. x -> 1").unwrap(), "1 + y");
  }

  #[test]
  fn replace_repeated_after_times() {
    assert_eq!(interpret("x * y //. x -> 2").unwrap(), "2*y");
  }

  #[test]
  fn replace_all_after_comparison() {
    assert_eq!(interpret("x > y /. x -> 3").unwrap(), "3 > y");
  }

  #[test]
  fn replace_all_all_vars_replaced() {
    assert_eq!(interpret("x + y /. {x -> 10, y -> 20}").unwrap(), "30");
  }

  #[test]
  fn replace_all_list_of_rules_simultaneous() {
    // Rules should be applied simultaneously, not sequentially
    // 2->1 and 1->0 should NOT chain (2 becomes 1, not 0)
    assert_eq!(
      interpret("{1,1,2,2,2,2} /. {2 -> 1, 1 -> 0}").unwrap(),
      "{0, 0, 1, 1, 1, 1}"
    );
  }

  #[test]
  fn replace_all_list_of_rules_first_match_wins() {
    // x->y should match, then y->z should NOT apply to the result
    assert_eq!(interpret("x /. {x -> y, y -> z}").unwrap(), "y");
  }

  #[test]
  fn replace_all_list_of_rules_no_match() {
    // No rule matches, original expression returned
    assert_eq!(
      interpret("{3, 4, 5} /. {1 -> a, 2 -> b}").unwrap(),
      "{3, 4, 5}"
    );
  }

  #[test]
  fn replace_all_list_of_rules_partial_match() {
    // Only some elements match
    assert_eq!(
      interpret("{a, b, c} /. {a -> x, b -> y}").unwrap(),
      "{x, y, c}"
    );
  }

  #[test]
  fn replace_all_list_of_rules_swap() {
    // Swap a and b simultaneously
    assert_eq!(
      interpret("{a, b, a, b} /. {a -> b, b -> a}").unwrap(),
      "{b, a, b, a}"
    );
  }
}

mod replace_all_expression_rhs {
  use super::*;

  #[test]
  fn rule_delayed_with_division() {
    assert_eq!(
      interpret("{0, 0, 1, 1, 1, 1} /. {any_Integer :> any / 2}").unwrap(),
      "{0, 0, 1/2, 1/2, 1/2, 1/2}"
    );
  }

  #[test]
  fn rule_delayed_with_addition() {
    assert_eq!(
      interpret("{1, 2, 3} /. {x_ :> x + 10}").unwrap(),
      "{11, 12, 13}"
    );
  }

  #[test]
  fn rule_with_expression_rhs() {
    assert_eq!(
      interpret("{1, 2, 3} /. x_Integer -> x + 10").unwrap(),
      "{11, 12, 13}"
    );
  }

  #[test]
  fn rule_delayed_with_power() {
    assert_eq!(interpret("5 /. x_Integer :> x^2").unwrap(), "25");
  }
}

mod replace_all_head_constraint {
  use super::*;

  #[test]
  fn integer_head_matches() {
    assert_eq!(
      interpret("{0, 0, 1, 1, 1, 1} /. {any_Integer :> any / 2}").unwrap(),
      "{0, 0, 1/2, 1/2, 1/2, 1/2}"
    );
  }

  #[test]
  fn integer_head_skips_non_integers() {
    assert_eq!(
      interpret(r#"{1, 2.5, "hello", x} /. a_Integer :> a + 100"#).unwrap(),
      "{101, 2.5, hello, x}"
    );
  }

  #[test]
  fn string_head_matches() {
    assert_eq!(
      interpret(r#"{1, "hi", "bye"} /. s_String :> StringLength[s]"#).unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn real_head_matches() {
    assert_eq!(
      interpret("{1, 2.5, 3.0} /. x_Real :> x * 10").unwrap(),
      "{1, 25., 30.}"
    );
  }

  #[test]
  fn head_no_match_returns_unchanged() {
    assert_eq!(
      interpret(r#""hello" /. x_Integer :> x^2"#).unwrap(),
      "hello"
    );
  }

  #[test]
  fn multiple_head_rules() {
    assert_eq!(
      interpret("{1, 2.5, 3} /. {a_Integer :> a + 100, b_Real :> b * 10}")
        .unwrap(),
      "{101, 25., 103}"
    );
  }
}

mod replace_repeated {
  use super::*;

  #[test]
  fn replace_repeated_applies_multiple_times() {
    assert_eq!(interpret("f[f[f[f[2]]]] //. f[2] -> 2").unwrap(), "2");
  }

  #[test]
  fn replace_repeated_simple() {
    assert_eq!(interpret("f[f[2]] //. f[2] -> 2").unwrap(), "2");
  }
}

mod replace_repeated_operator_form {
  use super::*;

  #[test]
  fn operator_form_works() {
    // ReplaceRepeated[rule][expr] should work like expr //. rule
    let result =
      interpret("ReplaceRepeated[f[2] -> 2][f[f[f[f[2]]]]]").unwrap();
    assert_eq!(result, "2");
  }

  #[test]
  fn infix_form_works() {
    let result = interpret("f[f[f[2]]] //. f[2] -> 2").unwrap();
    assert_eq!(result, "2");
  }
}

mod symbolic_equal {
  use super::*;

  #[test]
  fn numeric_equal() {
    assert_eq!(interpret("1 == 1").unwrap(), "True");
    assert_eq!(interpret("1 == 2").unwrap(), "False");
  }

  #[test]
  fn identical_symbols() {
    assert_eq!(interpret("x == x").unwrap(), "True");
  }

  #[test]
  fn different_symbols_stay_symbolic() {
    assert_eq!(interpret("x == y").unwrap(), "x == y");
  }

  #[test]
  fn symbolic_expression_vs_number() {
    assert_eq!(interpret("x + 1 == 0").unwrap(), "1 + x == 0");
  }

  #[test]
  fn same_q_always_evaluates() {
    assert_eq!(interpret("x === x").unwrap(), "True");
    assert_eq!(interpret("x === y").unwrap(), "False");
  }

  #[test]
  fn unequal_symbolic() {
    assert_eq!(interpret("x != x").unwrap(), "False");
    assert_eq!(interpret("x != y").unwrap(), "x != y");
  }

  #[test]
  fn unsame_q_always_evaluates() {
    assert_eq!(interpret("x =!= x").unwrap(), "False");
    assert_eq!(interpret("x =!= y").unwrap(), "True");
  }
}

mod solve {
  use super::*;

  #[test]
  fn linear_equation() {
    assert_eq!(interpret("Solve[x - 5 == 0, x]").unwrap(), "{{x -> 5}}");
    assert_eq!(interpret("Solve[2*x + 6 == 0, x]").unwrap(), "{{x -> -3}}");
    assert_eq!(interpret("Solve[3*x + 9 == 0, x]").unwrap(), "{{x -> -3}}");
  }

  #[test]
  fn quadratic_integer_roots() {
    assert_eq!(
      interpret("Solve[x^2 + 3x - 10 == 0, x]").unwrap(),
      "{{x -> -5}, {x -> 2}}"
    );
  }

  #[test]
  fn quadratic_symmetric() {
    assert_eq!(
      interpret("Solve[x^2 - 4 == 0, x]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }

  #[test]
  fn quadratic_with_leading_coeff() {
    assert_eq!(
      interpret("Solve[2*x^2 - 8 == 0, x]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }

  #[test]
  fn quadratic_repeated_root() {
    assert_eq!(
      interpret("Solve[x^2 + 2*x + 1 == 0, x]").unwrap(),
      "{{x -> -1}, {x -> -1}}"
    );
  }

  #[test]
  fn quadratic_complex_roots() {
    assert_eq!(
      interpret("Solve[x^2 + 1 == 0, x]").unwrap(),
      "{{x -> -I}, {x -> I}}"
    );
  }

  #[test]
  fn quadratic_irrational_roots() {
    assert_eq!(
      interpret("Solve[x^2 - 5 == 0, x]").unwrap(),
      "{{x -> -Sqrt[5]}, {x -> Sqrt[5]}}"
    );
  }

  #[test]
  fn quadratic_golden_ratio() {
    assert_eq!(
      interpret("Solve[x^2 + x - 1 == 0, x]").unwrap(),
      "{{x -> (-1 - Sqrt[5])/2}, {x -> (-1 + Sqrt[5])/2}}"
    );
  }

  #[test]
  fn quadratic_general() {
    assert_eq!(
      interpret("Solve[x^2 - 2*x - 3 == 0, x]").unwrap(),
      "{{x -> -1}, {x -> 3}}"
    );
  }

  #[test]
  fn trivial_x_equals_zero() {
    assert_eq!(interpret("Solve[x == 0, x]").unwrap(), "{{x -> 0}}");
  }

  #[test]
  fn tautology() {
    assert_eq!(interpret("Solve[x == x, x]").unwrap(), "{{}}");
  }

  #[test]
  fn contradiction() {
    assert_eq!(interpret("Solve[2 == 3, x]").unwrap(), "{}");
  }

  #[test]
  fn rational_solution() {
    assert_eq!(interpret("Solve[2*x - 1 == 0, x]").unwrap(), "{{x -> 1/2}}");
  }
}
