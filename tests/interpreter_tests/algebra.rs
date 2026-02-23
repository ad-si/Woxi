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

  #[test]
  fn pythagorean_identity() {
    assert_eq!(interpret("Simplify[Sin[x]^2 + Cos[x]^2]").unwrap(), "1");
  }

  #[test]
  fn pythagorean_with_coefficient() {
    assert_eq!(interpret("Simplify[2*Sin[x]^2 + 2*Cos[x]^2]").unwrap(), "2");
  }

  #[test]
  fn pythagorean_with_extra_terms() {
    assert_eq!(interpret("Simplify[Sin[y]^2 + Cos[y]^2 + 1]").unwrap(), "2");
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

  #[test]
  fn repeated_root_squared() {
    assert_eq!(interpret("Factor[x^2 + 2*x + 1]").unwrap(), "(1 + x)^2");
  }

  #[test]
  fn repeated_root_cubed() {
    assert_eq!(
      interpret("Factor[x^3 + 3*x^2 + 3*x + 1]").unwrap(),
      "(1 + x)^3"
    );
  }
}

mod factor_list {
  use super::*;

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("FactorList[x^3 - 1]").unwrap(),
      "{{1, 1}, {-1 + x, 1}, {1 + x + x^2, 1}}"
    );
  }

  #[test]
  fn with_numeric_coefficient() {
    assert_eq!(
      interpret("FactorList[2*x^2 + 4*x + 2]").unwrap(),
      "{{2, 1}, {1 + x, 2}}"
    );
  }

  #[test]
  fn irreducible() {
    assert_eq!(
      interpret("FactorList[x^2 + 1]").unwrap(),
      "{{1, 1}, {1 + x^2, 1}}"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("FactorList[6]").unwrap(), "{{6, 1}}");
  }

  #[test]
  fn quartic() {
    assert_eq!(
      interpret("FactorList[x^4 - 1]").unwrap(),
      "{{1, 1}, {-1 + x, 1}, {1 + x, 1}, {1 + x^2, 1}}"
    );
  }

  #[test]
  fn repeated_root() {
    assert_eq!(
      interpret("FactorList[x^3 + 3*x^2 + 3*x + 1]").unwrap(),
      "{{1, 1}, {1 + x, 3}}"
    );
  }

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("FactorList[x^2 + 3*x + 2]").unwrap(),
      "{{1, 1}, {1 + x, 1}, {2 + x, 1}}"
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

  #[test]
  fn head_constrained_blank() {
    // _Head pattern matches expressions with matching head
    assert_eq!(
      interpret("Switch[C[1], _C, matched, _, other]").unwrap(),
      "matched"
    );
  }

  #[test]
  fn head_constrained_blank_no_match() {
    // _Head pattern doesn't match different head
    assert_eq!(
      interpret("Switch[D[1], _C, matched, _, other]").unwrap(),
      "other"
    );
  }

  #[test]
  fn head_constrained_blank_integer() {
    assert_eq!(
      interpret("Switch[42, _Integer, num, _String, str]").unwrap(),
      "num"
    );
  }

  #[test]
  fn named_pattern() {
    // x_ matches anything — Switch does NOT bind pattern variables
    assert_eq!(interpret("Switch[42, x_, x + 1]").unwrap(), "1 + x");
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

  #[test]
  fn operator_form() {
    assert_eq!(interpret("MatchQ[_Integer][123]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[_String][123]").unwrap(), "False");
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

mod replace_all_conditional_multi_rules {
  use super::*;

  #[test]
  fn conditional_pattern_single_rule() {
    assert_eq!(interpret("27 /. n_ /; OddQ[n] :> 3 n + 1").unwrap(), "82");
  }

  #[test]
  fn conditional_pattern_multi_rules_scalar() {
    // Multi-rule ReplaceAll with conditional patterns on a scalar
    assert_eq!(
      interpret("6 /. {n_ /; EvenQ[n] :> n/2, n_ /; OddQ[n] :> 3 n + 1}")
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn conditional_pattern_multi_rules_list() {
    assert_eq!(
      interpret("{27, 6} /. {n_ /; EvenQ[n] :> n/2, n_ /; OddQ[n] :> 3 n + 1}")
        .unwrap(),
      "{82, 3}"
    );
  }

  #[test]
  fn nested_list_multi_rules() {
    // Multi-rule ReplaceAll should recurse into nested lists
    assert_eq!(
      interpret("{C[1], {X, 4, Y, C[1]}} /. {X -> a, Y -> b}").unwrap(),
      "{C[1], {a, 4, b, C[1]}}"
    );
  }

  #[test]
  fn nested_list_swap() {
    assert_eq!(
      interpret("{{a, b}, {c, d}} /. {a -> 1, d -> 4}").unwrap(),
      "{{1, b}, {c, 4}}"
    );
  }
}

mod replace_all_variable_rhs {
  use super::*;

  #[test]
  fn variable_holding_rules() {
    clear_state();
    assert_eq!(
      interpret("r = {x -> 1, y -> 2}; {x, y, z} /. r").unwrap(),
      "{1, 2, z}"
    );
  }

  #[test]
  fn variable_holding_conditional_rules() {
    clear_state();
    assert_eq!(
      interpret("r = {x_ /; EvenQ[x] :> x/2}; Map[# /. r &, {4, 7}]").unwrap(),
      "{2, 7}"
    );
  }

  #[test]
  fn variable_in_anonymous_function() {
    clear_state();
    assert_eq!(
      interpret("r = {a -> b, b -> c}; Nest[# /. r &, a, 2]").unwrap(),
      "c"
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

  #[test]
  fn constant_variable_rejected() {
    assert_eq!(
      interpret("Solve[x + E == 0, E]").unwrap(),
      "Solve[E + x == 0, E]"
    );
  }
}

mod roots {
  use super::*;

  #[test]
  fn roots_linear() {
    assert_eq!(interpret("Roots[x == 5, x]").unwrap(), "x == 5");
  }

  #[test]
  fn roots_quadratic_integer() {
    let result = interpret("Roots[x^2 - 4 == 0, x]").unwrap();
    assert_eq!(result, "x == -2 || x == 2");
  }

  #[test]
  fn roots_quadratic_symbolic() {
    let result = interpret("Roots[a*x^2 + b*x + c == 0, x]").unwrap();
    assert!(result.contains("||"), "Should have two roots: {}", result);
    assert!(result.contains("Sqrt"), "Should have Sqrt: {}", result);
  }

  #[test]
  fn roots_no_solution() {
    // x^2 + 1 == 0 has complex roots
    let result = interpret("Roots[x^2 + 1 == 0, x]").unwrap();
    assert!(
      result.contains("||"),
      "Should return complex roots: {}",
      result
    );
  }

  #[test]
  fn roots_quadratic_repeated() {
    let result = interpret("Roots[x^2 - 2*x + 1 == 0, x]").unwrap();
    assert_eq!(result, "x == 1");
  }
}

mod eliminate {
  use super::*;

  #[test]
  fn eliminate_single_variable_linear() {
    // Eliminate y from {x == 2 + y, y == z}
    let result = interpret("Eliminate[{x == 2 + y, y == z}, y]").unwrap();
    assert_eq!(result, "-2 + x == z");
  }

  #[test]
  fn eliminate_single_from_two_linear() {
    // Eliminate a from {x == a + b, y == a - b}
    let result = interpret("Eliminate[{x == a + b, y == a - b}, a]").unwrap();
    assert_eq!(result, "y == -2*b + x");
  }

  #[test]
  fn eliminate_to_constant() {
    // Eliminate y from {x + y == 3, x - y == 1}
    let result = interpret("Eliminate[{x + y == 3, x - y == 1}, y]").unwrap();
    assert_eq!(result, "-3 + 2*x == 1");
  }

  #[test]
  fn eliminate_with_product() {
    // Eliminate x from {a == x + y, b == x*y}
    let result = interpret("Eliminate[{a == x + y, b == x*y}, x]").unwrap();
    assert_eq!(result, "b == a*y - y^2");
  }

  #[test]
  fn eliminate_variable_not_found() {
    // If variable doesn't appear, equation is returned unchanged
    let result = interpret("Eliminate[{x == 2}, y]").unwrap();
    assert_eq!(result, "x == 2");
  }

  #[test]
  fn eliminate_single_equation() {
    // With a single equation and the variable in it, eliminating gives True
    let result = interpret("Eliminate[{x == 2}, x]").unwrap();
    assert_eq!(result, "True");
  }
}

mod to_rules {
  use super::*;

  #[test]
  fn to_rules_single_equation() {
    // x == 5 → {{x -> 5}}
    assert_eq!(interpret("ToRules[x == 5]").unwrap(), "{{x -> 5}}");
  }

  #[test]
  fn to_rules_or_conditions() {
    // x == -2 || x == 2 → {{x -> -2}, {x -> 2}}
    assert_eq!(
      interpret("ToRules[x == -2 || x == 2]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }

  #[test]
  fn to_rules_from_roots() {
    // Convert Roots output to Solve-style rules
    assert_eq!(
      interpret("ToRules[Roots[x^2 - 4 == 0, x]]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }

  #[test]
  fn to_rules_and_conditions() {
    // x == 1 && y == 2 → {{x -> 1, y -> 2}}
    assert_eq!(
      interpret("ToRules[x == 1 && y == 2]").unwrap(),
      "{{x -> 1, y -> 2}}"
    );
  }

  #[test]
  fn to_rules_true() {
    assert_eq!(interpret("ToRules[True]").unwrap(), "{{}}");
  }

  #[test]
  fn to_rules_false() {
    assert_eq!(interpret("ToRules[False]").unwrap(), "{}");
  }
}

mod reduce {
  use super::*;

  // ── Trivial cases ──

  #[test]
  fn reduce_true() {
    assert_eq!(interpret("Reduce[True, x]").unwrap(), "True");
  }

  #[test]
  fn reduce_false() {
    assert_eq!(interpret("Reduce[False, x]").unwrap(), "False");
  }

  // ── Linear equations ──

  #[test]
  fn linear_equation() {
    assert_eq!(interpret("Reduce[2*x + 3 == 7, x]").unwrap(), "x == 2");
  }

  #[test]
  fn linear_equation_negative() {
    assert_eq!(interpret("Reduce[x - 5 == 0, x]").unwrap(), "x == 5");
  }

  #[test]
  fn trivial_equation() {
    assert_eq!(interpret("Reduce[x == 5, x]").unwrap(), "x == 5");
  }

  // ── Quadratic equations ──

  #[test]
  fn quadratic_integer_roots() {
    assert_eq!(
      interpret("Reduce[x^2 - 4 == 0, x]").unwrap(),
      "x == -2 || x == 2"
    );
  }

  #[test]
  fn quadratic_two_roots() {
    assert_eq!(
      interpret("Reduce[x^2 + x - 6 == 0, x]").unwrap(),
      "x == -3 || x == 2"
    );
  }

  #[test]
  fn quadratic_repeated_root() {
    assert_eq!(
      interpret("Reduce[x^2 + 2*x + 1 == 0, x]").unwrap(),
      "x == -1"
    );
  }

  #[test]
  fn quadratic_irrational_roots() {
    assert_eq!(
      interpret("Reduce[x^2 - 3 == 0, x]").unwrap(),
      "x == -Sqrt[3] || x == Sqrt[3]"
    );
  }

  #[test]
  fn quadratic_complex_roots() {
    assert_eq!(
      interpret("Reduce[x^2 + 1 == 0, x]").unwrap(),
      "x == -I || x == I"
    );
  }

  // ── Higher degree equations (via factoring) ──

  #[test]
  fn cubic_equation() {
    assert_eq!(
      interpret("Reduce[x^3 - 3*x^2 + 2*x == 0, x]").unwrap(),
      "x == 0 || x == 1 || x == 2"
    );
  }

  #[test]
  fn cubic_factored() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x - 2)*(x - 3) == 0, x]").unwrap(),
      "x == 1 || x == 2 || x == 3"
    );
  }

  #[test]
  fn quartic_factored() {
    assert_eq!(
      interpret("Reduce[x*(x - 1)*(x - 2)*(x - 3) == 0, x]").unwrap(),
      "x == 0 || x == 1 || x == 2 || x == 3"
    );
  }

  // ── Domain filtering ──

  #[test]
  fn complex_roots_over_reals() {
    assert_eq!(
      interpret("Reduce[x^2 + 1 == 0, x, Reals]").unwrap(),
      "False"
    );
  }

  #[test]
  fn complex_roots_over_integers() {
    assert_eq!(
      interpret("Reduce[x^2 + 1 == 0, x, Integers]").unwrap(),
      "False"
    );
  }

  #[test]
  fn real_roots_over_reals() {
    assert_eq!(
      interpret("Reduce[x^2 - 1 == 0, x, Reals]").unwrap(),
      "x == -1 || x == 1"
    );
  }

  // ── Or (disjunction) ──

  #[test]
  fn reduce_or() {
    assert_eq!(
      interpret("Reduce[x == 1 || x == 2, x]").unwrap(),
      "x == 1 || x == 2"
    );
  }

  // ── Simple inequalities ──

  #[test]
  fn simple_inequality_gt() {
    assert_eq!(interpret("Reduce[x > 0, x]").unwrap(), "x > 0");
  }

  // ── Quadratic inequalities ──

  #[test]
  fn quadratic_inequality_less() {
    assert_eq!(
      interpret("Reduce[x^2 < 4, x]").unwrap(),
      "Inequality[-2, Less, x, Less, 2]"
    );
  }

  #[test]
  fn quadratic_inequality_greater() {
    assert_eq!(
      interpret("Reduce[x^2 - 1 > 0, x]").unwrap(),
      "x < -1 || x > 1"
    );
  }

  #[test]
  fn quadratic_inequality_geq() {
    assert_eq!(
      interpret("Reduce[x^2 - 1 >= 0, x]").unwrap(),
      "x <= -1 || x >= 1"
    );
  }

  #[test]
  fn factored_inequality_less() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x + 2) < 0, x]").unwrap(),
      "Inequality[-2, Less, x, Less, 1]"
    );
  }

  #[test]
  fn factored_inequality_leq() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x + 2) <= 0, x]").unwrap(),
      "Inequality[-2, LessEqual, x, LessEqual, 1]"
    );
  }

  #[test]
  fn factored_inequality_greater() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x + 2) > 0, x]").unwrap(),
      "x < -2 || x > 1"
    );
  }

  // ── Always true / always false inequalities ──

  #[test]
  fn always_true_inequality() {
    assert_eq!(
      interpret("Reduce[x^2 + 1 > 0, x]").unwrap(),
      "Element[x, Reals]"
    );
  }

  #[test]
  fn always_true_with_reals_domain() {
    assert_eq!(interpret("Reduce[x^2 + 1 > 0, x, Reals]").unwrap(), "True");
  }

  #[test]
  fn always_false_inequality() {
    assert_eq!(interpret("Reduce[x^2 + 1 < 0, x]").unwrap(), "False");
  }

  // ── And (conjunction) ──

  #[test]
  fn equation_with_inequality_constraint() {
    assert_eq!(interpret("Reduce[x^2 == 9 && x > 0, x]").unwrap(), "x == 3");
  }

  #[test]
  fn combined_inequalities() {
    assert_eq!(
      interpret("Reduce[x > 0 && x < 5 && x > 3, x]").unwrap(),
      "Inequality[3, Less, x, Less, 5]"
    );
  }

  #[test]
  fn combined_two_bounds() {
    assert_eq!(
      interpret("Reduce[x > 2 && x < 10, x]").unwrap(),
      "Inequality[2, Less, x, Less, 10]"
    );
  }

  #[test]
  fn mixed_equation_inequality() {
    assert_eq!(
      interpret("Reduce[x^2 <= 4 && x > 0, x]").unwrap(),
      "Inequality[0, Less, x, LessEqual, 2]"
    );
  }

  // ── Multi-variable systems ──

  #[test]
  fn two_variable_linear_system() {
    assert_eq!(
      interpret("Reduce[x + y == 5 && x - y == 1, {x, y}]").unwrap(),
      "x == 3 && y == 2"
    );
  }

  #[test]
  fn two_variable_list_input() {
    assert_eq!(
      interpret("Reduce[{x + y == 5, x - y == 1}, {x, y}]").unwrap(),
      "x == 3 && y == 2"
    );
  }

  // ── Cubic with expanded polynomial ──

  #[test]
  fn cubic_expanded() {
    assert_eq!(
      interpret("Reduce[x^3 - 6 x^2 + 11 x - 6 == 0, x]").unwrap(),
      "x == 1 || x == 2 || x == 3"
    );
  }
}

mod find_root {
  use super::*;

  #[test]
  fn polynomial_root() {
    assert_eq!(
      interpret("FindRoot[x^2 - 2, {x, 1}]").unwrap(),
      "{x -> 1.4142135623730951}"
    );
  }

  #[test]
  fn polynomial_root_negative_start() {
    assert_eq!(
      interpret("FindRoot[x^2 - 2, {x, -1}]").unwrap(),
      "{x -> -1.4142135623730951}"
    );
  }

  #[test]
  fn equation_form() {
    assert_eq!(
      interpret("FindRoot[Cos[x] == x, {x, 0}]").unwrap(),
      "{x -> 0.7390851332151607}"
    );
  }

  #[test]
  fn transcendental() {
    assert_eq!(
      interpret("FindRoot[Sin[x] + Exp[x], {x, 0}]").unwrap(),
      "{x -> -0.5885327439818611}"
    );
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("FindRoot[x^3 - x - 1, {x, 1}]").unwrap(),
      "{x -> 1.324717957244746}"
    );
  }

  #[test]
  fn exponential() {
    assert_eq!(
      interpret("FindRoot[Exp[x] - 2, {x, 0}]").unwrap(),
      "{x -> 0.6931471805599453}"
    );
  }

  #[test]
  fn trivial() {
    assert_eq!(interpret("FindRoot[x, {x, 5}]").unwrap(), "{x -> 0.}");
  }

  #[test]
  fn quadratic_larger_start() {
    assert_eq!(
      interpret("FindRoot[x^2 - 10^5 x + 1 == 0, {x, 10^6}]").unwrap(),
      "{x -> 99999.99999000001}"
    );
  }
}

mod replace {
  use super::*;

  #[test]
  fn simple_match() {
    assert_eq!(interpret("Replace[x, x -> 2]").unwrap(), "2");
  }

  #[test]
  fn with_rule_list() {
    assert_eq!(interpret("Replace[x, {x -> 2}]").unwrap(), "2");
  }

  #[test]
  fn no_subexpression_match() {
    assert_eq!(interpret("Replace[1 + x, {x -> 2}]").unwrap(), "1 + x");
  }

  #[test]
  fn multiple_rule_sets() {
    assert_eq!(
      interpret("Replace[x, {{x -> 1}, {x -> 2}}]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn first_matching_rule() {
    assert_eq!(interpret("Replace[x, {x -> 10, y -> 20}]").unwrap(), "10");
  }

  #[test]
  fn pattern_match() {
    assert_eq!(interpret("Replace[42, n_Integer -> n + 1]").unwrap(), "43");
  }
}

mod distribute {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("Distribute[f[a + b, c]]").unwrap(),
      "f[a, c] + f[b, c]"
    );
  }

  #[test]
  fn both_sums() {
    assert_eq!(
      interpret("Distribute[f[a + b, c + d]]").unwrap(),
      "f[a, c] + f[a, d] + f[b, c] + f[b, d]"
    );
  }

  #[test]
  fn times_over_plus() {
    assert_eq!(
      interpret("Distribute[(a + b)(c + d)]").unwrap(),
      "a*c + b*c + a*d + b*d"
    );
  }

  #[test]
  fn three_terms() {
    assert_eq!(
      interpret("Distribute[f[a + b + c, d]]").unwrap(),
      "f[a, d] + f[b, d] + f[c, d]"
    );
  }

  #[test]
  fn three_args() {
    let result = interpret("Distribute[f[a + b, c + d, e + g]]").unwrap();
    assert!(result.contains("f[a, c, e]"));
    assert!(result.contains("f[b, d, g]"));
  }

  #[test]
  fn no_distribution_needed() {
    assert_eq!(interpret("Distribute[f[a, b]]").unwrap(), "f[a, b]");
  }

  #[test]
  fn with_head_restriction() {
    assert_eq!(
      interpret("Distribute[(a + b)(c + d), Plus, Times]").unwrap(),
      "a*c + b*c + a*d + b*d"
    );
  }

  #[test]
  fn atom_input() {
    assert_eq!(interpret("Distribute[x]").unwrap(), "x");
  }
}

mod polynomial_remainder {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("PolynomialRemainder[x^3 + 2x + 1, x^2 + 1, x]").unwrap(),
      "1 + x"
    );
  }

  #[test]
  fn exact_division() {
    assert_eq!(
      interpret("PolynomialRemainder[x (x^2 + 1), x^2 + 1, x]").unwrap(),
      "0"
    );
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("PolynomialRemainder[a x^2 + b x + c, x + 1, x]").unwrap(),
      "a - b + c"
    );
  }

  #[test]
  fn lower_degree_dividend() {
    assert_eq!(
      interpret("PolynomialRemainder[x + 1, x^2, x]").unwrap(),
      "1 + x"
    );
  }

  #[test]
  fn quotient_basic() {
    assert_eq!(
      interpret("PolynomialQuotient[x^3 + 2x + 1, x^2 + 1, x]").unwrap(),
      "x"
    );
  }

  #[test]
  fn quotient_and_remainder_consistency() {
    // p = q * quotient + remainder
    let q =
      interpret("PolynomialQuotient[x^4 + 3x^2 + x, x^2 + 1, x]").unwrap();
    let r =
      interpret("PolynomialRemainder[x^4 + 3x^2 + x, x^2 + 1, x]").unwrap();
    // q should be x^2 + 2, r should be x - 2
    assert_eq!(q, "2 + x^2");
    assert_eq!(r, "-2 + x");
  }
}

mod solve_always {
  use super::*;

  #[test]
  fn linear_single_variable() {
    assert_eq!(
      interpret("SolveAlways[a*x + b == 0, x]").unwrap(),
      "{{b -> 0, a -> 0}}"
    );
  }

  #[test]
  fn quadratic_with_offsets() {
    assert_eq!(
      interpret("SolveAlways[(a - 2)*x^2 + (b + 1)*x + c == 0, x]").unwrap(),
      "{{c -> 0, b -> -1, a -> 2}}"
    );
  }

  #[test]
  fn matching_polynomial() {
    assert_eq!(
      interpret("SolveAlways[a*x^2 + b*x + c == 3*x^2 - 5*x + 7, x]").unwrap(),
      "{{c -> 7, b -> -5, a -> 3}}"
    );
  }

  #[test]
  fn trivially_true() {
    assert_eq!(interpret("SolveAlways[0 == 0, x]").unwrap(), "{{}}");
  }

  #[test]
  fn impossible_equation() {
    assert_eq!(interpret("SolveAlways[x + 1 == 0, x]").unwrap(), "{}");
  }

  #[test]
  fn no_parameters() {
    assert_eq!(
      interpret("SolveAlways[3*x^2 + 5*x + 7 == 0, x]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn multivariate() {
    assert_eq!(
      interpret("SolveAlways[(a - 2)*x + (b + 1)*y + c == 0, {x, y}]").unwrap(),
      "{{c -> 0, b -> -1, a -> 2}}"
    );
  }

  #[test]
  fn multivariate_all_zero() {
    assert_eq!(
      interpret("SolveAlways[a*x + b*y + c == 0, {x, y}]").unwrap(),
      "{{c -> 0, b -> 0, a -> 0}}"
    );
  }

  #[test]
  fn list_form_single_var() {
    assert_eq!(
      interpret("SolveAlways[a*x + b == 0, {x}]").unwrap(),
      "{{b -> 0, a -> 0}}"
    );
  }

  #[test]
  fn quadratic_cross_terms() {
    assert_eq!(
      interpret("SolveAlways[a*x^2 + b*x*y + c*y^2 == 0, {x, y}]").unwrap(),
      "{{c -> 0, b -> 0, a -> 0}}"
    );
  }
}

mod factor_terms {
  use super::*;

  #[test]
  fn basic_integer_gcd() {
    assert_eq!(
      interpret("FactorTerms[3 + 6 x + 3 x^2]").unwrap(),
      "3*(1 + 2*x + x^2)"
    );
  }

  #[test]
  fn factor_from_all_terms() {
    assert_eq!(
      interpret("FactorTerms[6 x + 9 x^2]").unwrap(),
      "3*(2*x + 3*x^2)"
    );
  }

  #[test]
  fn simple_factoring() {
    assert_eq!(interpret("FactorTerms[5 + 10 x]").unwrap(), "5*(1 + 2*x)");
  }

  #[test]
  fn gcd_one_no_change() {
    assert_eq!(interpret("FactorTerms[x + x^2]").unwrap(), "x + x^2");
  }

  #[test]
  fn single_term() {
    assert_eq!(interpret("FactorTerms[7 x]").unwrap(), "7*x");
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("FactorTerms[42]").unwrap(), "42");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("FactorTerms[0]").unwrap(), "0");
  }

  #[test]
  fn variable_only() {
    assert_eq!(interpret("FactorTerms[x]").unwrap(), "x");
  }

  #[test]
  fn negative_gcd() {
    assert_eq!(
      interpret("FactorTerms[-6*x - 9*x^2]").unwrap(),
      "-3*(2*x + 3*x^2)"
    );
  }

  #[test]
  fn negative_gcd_with_constant() {
    assert_eq!(interpret("FactorTerms[-3 - 6*x]").unwrap(), "-3*(1 + 2*x)");
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("FactorTerms[2 a + 4 b + 6 c]").unwrap(),
      "2*(a + 2*b + 3*c)"
    );
  }

  #[test]
  fn no_common_factor() {
    assert_eq!(interpret("FactorTerms[a + b]").unwrap(), "a + b");
  }

  #[test]
  fn rational_coefficients() {
    assert_eq!(
      interpret("FactorTerms[2/3 + (4/3)*x]").unwrap(),
      "(2*(1 + 2*x))/3"
    );
  }

  #[test]
  fn with_variable_argument() {
    assert_eq!(
      interpret("FactorTerms[3 + 3 a + 6 a x + 6 x + 12 a x^2 + 12 x^2, x]")
        .unwrap(),
      "3*(1 + a)*(1 + 2*x + 4*x^2)"
    );
  }

  #[test]
  fn threads_over_list() {
    assert_eq!(
      interpret("FactorTerms[{6 + 12 x, 4 + 8 x}]").unwrap(),
      "{6*(1 + 2*x), 4*(1 + 2*x)}"
    );
  }
}

mod cyclotomic {
  use super::*;

  #[test]
  fn phi_1() {
    assert_eq!(interpret("Cyclotomic[1, x]").unwrap(), "-1 + x");
  }

  #[test]
  fn phi_2() {
    assert_eq!(interpret("Cyclotomic[2, x]").unwrap(), "1 + x");
  }

  #[test]
  fn phi_3() {
    assert_eq!(interpret("Cyclotomic[3, x]").unwrap(), "1 + x + x^2");
  }

  #[test]
  fn phi_4() {
    assert_eq!(interpret("Cyclotomic[4, x]").unwrap(), "1 + x^2");
  }

  #[test]
  fn phi_5() {
    assert_eq!(
      interpret("Cyclotomic[5, x]").unwrap(),
      "1 + x + x^2 + x^3 + x^4"
    );
  }

  #[test]
  fn phi_6() {
    assert_eq!(interpret("Cyclotomic[6, x]").unwrap(), "1 - x + x^2");
  }

  #[test]
  fn phi_8() {
    assert_eq!(interpret("Cyclotomic[8, x]").unwrap(), "1 + x^4");
  }

  #[test]
  fn phi_12() {
    assert_eq!(interpret("Cyclotomic[12, x]").unwrap(), "1 - x^2 + x^4");
  }

  #[test]
  fn phi_0() {
    assert_eq!(interpret("Cyclotomic[0, x]").unwrap(), "1");
  }

  #[test]
  fn numeric_evaluation() {
    assert_eq!(interpret("Cyclotomic[1, 3]").unwrap(), "2");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Cyclotomic]").unwrap(),
      "{Listable, Protected}"
    );
  }
}

mod expand_denominator {
  use super::*;

  #[test]
  fn basic_square() {
    assert_eq!(
      interpret("ExpandDenominator[(1 + x)^2/(1 + y)^2]").unwrap(),
      "(1 + x)^2/(1 + 2*y + y^2)"
    );
  }

  #[test]
  fn cubic_denominator() {
    assert_eq!(
      interpret("ExpandDenominator[(a + b)/(c + d)^3]").unwrap(),
      "(a + b)/(c^3 + 3*c^2*d + 3*c*d^2 + d^3)"
    );
  }

  #[test]
  fn no_denominator() {
    assert_eq!(interpret("ExpandDenominator[x + 1]").unwrap(), "1 + x");
  }

  #[test]
  fn simple_fraction() {
    assert_eq!(
      interpret("ExpandDenominator[x/(y+z)]").unwrap(),
      "x/(y + z)"
    );
  }

  #[test]
  fn sum_of_fractions() {
    assert_eq!(
      interpret("ExpandDenominator[a/(1+x)^2 + b/(1+y)^2]").unwrap(),
      "a/(1 + 2*x + x^2) + b/(1 + 2*y + y^2)"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ExpandDenominator]").unwrap(),
      "{Protected}"
    );
  }
}

mod resultant {
  use super::*;

  #[test]
  fn basic_integer() {
    assert_eq!(interpret("Resultant[x^2 + 1, x^3 - 1, x]").unwrap(), "2");
  }

  #[test]
  fn common_root() {
    // x^2 - 1 and x^3 - 1 share x=1 as a root
    assert_eq!(interpret("Resultant[x^2 - 1, x^3 - 1, x]").unwrap(), "0");
  }

  #[test]
  fn linear_polynomials() {
    assert_eq!(interpret("Resultant[2*x + 3, 4*x - 1, x]").unwrap(), "-14");
  }

  #[test]
  fn symbolic_linear() {
    assert_eq!(
      interpret("Resultant[a*x + b, c*x + d, x]").unwrap(),
      "-(b*c) + a*d"
    );
  }

  #[test]
  fn symbolic_quadratic_expanded() {
    assert_eq!(
      interpret("Expand[Resultant[x^2 + a*x + b, x^2 + c*x + d, x]]").unwrap(),
      "b^2 - a*b*c + b*c^2 + a^2*d - 2*b*d - a*c*d + d^2"
    );
  }

  #[test]
  fn quadratic_common_root() {
    // x^2 - 5x + 6 = (x-2)(x-3), x^2 - 3x + 2 = (x-1)(x-2), share x=2
    assert_eq!(
      interpret("Resultant[x^2 - 5*x + 6, x^2 - 3*x + 2, x]").unwrap(),
      "0"
    );
  }

  #[test]
  fn zero_polynomial() {
    assert_eq!(interpret("Resultant[x, x^2, x]").unwrap(), "0");
  }

  #[test]
  fn constant_and_polynomial() {
    assert_eq!(interpret("Resultant[3, x + 1, x]").unwrap(), "3");
  }

  #[test]
  fn symbolic_stays_unevaluated() {
    assert_eq!(interpret("Resultant[f, g, x]").unwrap(), "1");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Resultant]").unwrap(),
      "{Listable, Protected}"
    );
  }
}
