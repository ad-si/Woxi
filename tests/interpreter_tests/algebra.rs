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

  #[test]
  fn multivariate_list_of_vars() {
    // PolynomialQ with a list of variables
    assert_eq!(interpret("PolynomialQ[x + y^2, {x, y}]").unwrap(), "True");
    assert_eq!(
      interpret("PolynomialQ[x^2 + 2*x*y + y^2, {x, y}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("PolynomialQ[Sin[x] + y, {x, y}]").unwrap(),
      "False"
    );
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

  #[test]
  fn rational_exponent() {
    assert_eq!(interpret("Exponent[b*x^(3/2), x]").unwrap(), "3/2");
    assert_eq!(interpret("Exponent[x^(1/2) + x^(5/2), x]").unwrap(), "5/2");
    assert_eq!(interpret("Exponent[x^(1/3) + x^(2/3), x]").unwrap(), "2/3");
  }

  #[test]
  fn rational_exponent_min() {
    assert_eq!(
      interpret("Exponent[x^(1/2) + x^(5/2), x, Min]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn exponent_list() {
    assert_eq!(interpret("Exponent[-4x, x, List]").unwrap(), "{1}");
    assert_eq!(
      interpret("Exponent[x^3 + 2x^2 - 5x + 1, x, List]").unwrap(),
      "{0, 1, 2, 3}"
    );
    assert_eq!(interpret("Exponent[x^2 + 1, x, List]").unwrap(), "{0, 2}");
    assert_eq!(interpret("Exponent[5, x, List]").unwrap(), "{0}");
  }

  #[test]
  fn exponent_list_in_map() {
    assert_eq!(
      interpret(
        "u = -4x; Map[Function[Coefficient[u,x,#]*x^#], Exponent[u,x,List]]"
      )
      .unwrap(),
      "{-4*x}"
    );
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

  #[test]
  fn monomial_second_argument() {
    // Coefficient[expr, x^n] should mean "coefficient of x^n" — the
    // same result as Coefficient[expr, x, n].
    assert_eq!(interpret("Coefficient[(x + 1)^5, x^3]").unwrap(), "10");
    assert_eq!(interpret("Coefficient[3 x^2 + 5 x, x^2]").unwrap(), "3");
    assert_eq!(interpret("Coefficient[(x + y)^4, x^2]").unwrap(), "6*y^2");
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

  #[test]
  fn expand_inside_equal() {
    assert_eq!(
      interpret("Expand[(x - 1)*(x^2 + 1) == 0]").unwrap(),
      "-1 + x - x^2 + x^3 == 0"
    );
  }

  #[test]
  fn expand_inside_comparison() {
    assert_eq!(
      interpret("Expand[(a + b)^2 == (c + d)^2]").unwrap(),
      "a^2 + 2*a*b + b^2 == c^2 + 2*c*d + d^2"
    );
  }

  #[test]
  fn expand_inside_inequality() {
    assert_eq!(
      interpret("Expand[(x + 1)^2 > x]").unwrap(),
      "1 + 2*x + x^2 > x"
    );
  }

  #[test]
  fn inequality_numeric_evaluation() {
    assert_eq!(
      interpret("Inequality[1, Less, 3, Less, 5]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Inequality[1, Less, 0, Less, 5]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Inequality[1, Less, x, Less, 5] /. x -> 3").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Inequality[1, Less, x, Less, 5] /. x -> 0").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Inequality[1, LessEqual, 1, LessEqual, 5]").unwrap(),
      "True"
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

  /// Regression test for issue #93: Simplify must handle canonical
  /// Times[Power[]] form identically to Divide form
  #[test]
  fn simplify_canonical_division_form() {
    assert_eq!(interpret("Simplify[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
    assert_eq!(
      interpret("Simplify[(x^2 - 1) * (x - 1)^-1]").unwrap(),
      "1 + x"
    );
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

  #[test]
  fn combine_like_denominator_fractions() {
    assert_eq!(interpret("Simplify[a/x + b/x]").unwrap(), "(a + b)/x");
  }

  #[test]
  fn combine_like_denominator_with_extra() {
    assert_eq!(
      interpret("Simplify[a/x + b/x + c/y]").unwrap(),
      "(a + b)/x + c/y"
    );
  }

  #[test]
  fn combine_fractions_different_denominators() {
    assert_eq!(
      interpret(
        "Simplify[k*q/(2*a^4*(1 + s)^(3/2)) + k*q*(1 + s)^(9/4)/(2*a^4)]"
      )
      .unwrap(),
      "(k*q*(1 + (1 + s)^(15/4)))/(2*a^4*(1 + s)^(3/2))"
    );
  }

  #[test]
  fn trig_polynomial_power_reduction() {
    // Simplify[D[Sin[x]^10, {x, 4}]] should use double-angle forms
    assert_eq!(
      interpret("Simplify[D[Sin[x]^10, {x, 4}]]").unwrap(),
      "10*(141 + 238*Cos[2*x] + 125*Cos[4*x])*Sin[x]^6"
    );
  }

  #[test]
  fn trig_polynomial_simple() {
    assert_eq!(
      interpret("Simplify[3*Cos[x]^2*Sin[x]^2 + Sin[x]^4]").unwrap(),
      "(2 + Cos[2*x])*Sin[x]^2"
    );
  }

  #[test]
  fn trig_polynomial_cos_dominant() {
    assert_eq!(
      interpret("Simplify[2*Cos[x]^4 - Cos[x]^2*Sin[x]^2]").unwrap(),
      "((1 + 3*Cos[2*x])*Cos[x]^2)/2"
    );
  }

  #[test]
  fn equation_algebraically_equal() {
    // x^2 - y^2 == (x+y)(x-y) is always True
    assert_eq!(
      interpret("Simplify[x^2 - y^2 == (x + y)(x - y)]").unwrap(),
      "True"
    );
  }

  #[test]
  fn equation_expansion_equal() {
    assert_eq!(
      interpret("Simplify[(a + b)^2 == a^2 + 2 a b + b^2]").unwrap(),
      "True"
    );
  }

  #[test]
  fn equation_constant_difference_false() {
    // x^2 == x^2 + 1 is always False
    assert_eq!(interpret("Simplify[x^2 == x^2 + 1]").unwrap(), "False");
  }

  #[test]
  fn equation_symbolic_stays_unevaluated() {
    assert_eq!(interpret("Simplify[x == y]").unwrap(), "x == y");
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

mod factor_multivariate {
  use super::*;

  #[test]
  fn bivariate_perfect_square() {
    assert_eq!(interpret("Factor[x^2 + 2*x*y + y^2]").unwrap(), "(x + y)^2");
  }

  #[test]
  fn bivariate_difference_of_squares() {
    assert_eq!(interpret("Factor[x^2 - y^2]").unwrap(), "(x - y)*(x + y)");
  }

  #[test]
  fn common_variable_factor() {
    assert_eq!(interpret("Factor[x^2*y + x*y^2]").unwrap(), "x*y*(x + y)");
  }

  #[test]
  fn perfect_cube() {
    assert_eq!(
      interpret("Factor[x^3 + 3*x^2*y + 3*x*y^2 + y^3]").unwrap(),
      "(x + y)^3"
    );
  }

  #[test]
  fn target_expression() {
    assert_eq!(
      interpret("Factor[Expand[Expand[(x + y)^2 + 9(2 + x)(x + y)]^3]]")
        .unwrap(),
      "(x + y)^3*(18 + 10*x + y)^3"
    );
  }

  #[test]
  fn irreducible_sum_of_squares() {
    assert_eq!(interpret("Factor[x^2 + y^2]").unwrap(), "x^2 + y^2");
  }

  #[test]
  fn with_numeric_gcd() {
    assert_eq!(
      interpret("Factor[6*x^2 + 12*x*y + 6*y^2]").unwrap(),
      "6*(x + y)^2"
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

  /// Regression test for issue #93: Cancel must handle canonical
  /// Times[Power[]] form identically to Divide form
  #[test]
  fn cancel_canonical_times_power_form() {
    // Both forms must produce the same result
    assert_eq!(interpret("Cancel[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
    assert_eq!(
      interpret("Cancel[(x^2 - 1) * (x - 1)^-1]").unwrap(),
      "1 + x"
    );
  }
}

mod expand_modulus {
  use super::*;

  #[test]
  fn modulus_3() {
    assert_eq!(
      interpret("Expand[(1 + a)^12, Modulus -> 3]").unwrap(),
      "1 + a^3 + a^9 + a^12"
    );
  }

  #[test]
  fn modulus_4() {
    assert_eq!(
      interpret("Expand[(1 + a)^12, Modulus -> 4]").unwrap(),
      "1 + 2*a^2 + 3*a^4 + 3*a^8 + 2*a^10 + a^12"
    );
  }

  #[test]
  fn modulus_simple() {
    assert_eq!(
      interpret("Expand[(1 + a)^3, Modulus -> 3]").unwrap(),
      "1 + a^3"
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

  #[test]
  fn collect_product_coefficient_canonical_ordering() {
    // When the coefficient is a product (not a sum), factors should be
    // flattened and sorted in canonical order (alphabetical).
    assert_eq!(
      interpret("Collect[a x^2 + b x^2 y + c x y, x]").unwrap(),
      "c*x*y + x^2*(a + b*y)"
    );
  }

  #[test]
  fn collect_two_variables_shared_factor_ascending_powers() {
    // Regression: when collecting by {x, y} and every grouped term shares
    // the y factor, terms should be ordered by ascending power of x and
    // each term's Plus coefficient should appear *before* the monomial
    // factors (Plus * x^k * y), matching wolframscript exactly.
    assert_eq!(
      interpret(
        "Collect[a^2 y + 2 a b y + b^2 y + 2 a x y + 2 b x y + x^2 y + c^2 x^2 y + 2 c d x^2 y + d^2 x^2 y, {x, y}]"
      )
      .unwrap(),
      "(a^2 + 2*a*b + b^2)*y + (2*a + 2*b)*x*y + (1 + c^2 + 2*c*d + d^2)*x^2*y"
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

  #[test]
  fn prefix_apply() {
    // Piecewise @ {{...}} should work the same as Piecewise[{{...}}]
    assert_eq!(
      interpret("Piecewise @ {{1, True}, {2, False}}").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Piecewise @ {{1, False}, {2, True}}").unwrap(),
      "2"
    );
  }

  #[test]
  fn prefix_apply_with_conditions() {
    clear_state();
    assert_eq!(
      interpret("x = 3; Piecewise @ {{1, x < 0}, {2, x >= 0}}").unwrap(),
      "2"
    );
  }

  #[test]
  fn with_chained_inequality() {
    clear_state();
    assert_eq!(
      interpret("x = 0.5; Piecewise[{{1, 0 <= x < 1}, {2, 1 <= x < 2}}]")
        .unwrap(),
      "1"
    );
    assert_eq!(
      interpret("x = 1.5; Piecewise[{{1, 0 <= x < 1}, {2, 1 <= x < 2}}]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn via_table_prefix_apply() {
    assert_eq!(
      interpret("With[{l = {1, 2, 1}}, Piecewise @ Table[{(-1)^i, Accumulate[Prepend[l, 0]][[i]] <= t < Accumulate[Prepend[l, 0]][[i + 1]]}, {i, 1, Length[l]}] /. t -> 0.5]").unwrap(),
      "-1"
    );
    assert_eq!(
      interpret("With[{l = {1, 2, 1}}, Piecewise @ Table[{(-1)^i, Accumulate[Prepend[l, 0]][[i]] <= t < Accumulate[Prepend[l, 0]][[i + 1]]}, {i, 1, Length[l]}] /. t -> 1.5]").unwrap(),
      "1"
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

  #[test]
  fn repeated_pattern_variable_must_match() {
    // Regression: matches_pattern_ast ignored Pattern names, so both `a_`
    // positions in {a_, b_, a_} would match independently.
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {a_, b_, a_}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2, 1}, {a_, b_, a_}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn repeated_pattern_variable_in_function_args() {
    // Pattern variable `x_` used twice must match the same value
    assert_eq!(interpret("MatchQ[f[1, 1], f[x_, x_]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[f[1, 2], f[x_, x_]]").unwrap(), "False");
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
  fn reciprocal_product_equal() {
    // Power[Times[...], -1] should distribute and match 1/(...) form
    assert_eq!(interpret("1/(a*b) == (a*b)^-1").unwrap(), "True");
    assert_eq!(interpret("1/(x*y) == (x*y)^-1").unwrap(), "True");
    assert_eq!(
      interpret("1/(Sqrt[x]*(a + b*x)) == (Sqrt[x]*(a + b*x))^-1").unwrap(),
      "True"
    );
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

  #[test]
  fn fractional_power_equation() {
    // Physics: find extrema of E-field on axis of ring charge
    // Common constant factors (2*k*q) are factored out before the quadratic formula
    assert_eq!(
      interpret(
        "Solve[(2*k*q*(a^2 + x^2)^(3/2) - 6*k*q*x^2*(a^2 + x^2)^(1/2))/(a^2 + x^2)^3 == 0, x]"
      )
      .unwrap(),
      "{{x -> -(a/Sqrt[2])}, {x -> a/Sqrt[2]}}"
    );
  }

  #[test]
  fn symbolic_quadratic_simple() {
    assert_eq!(
      interpret("Solve[a^2 - 2*x^2 == 0, x]").unwrap(),
      "{{x -> -(a/Sqrt[2])}, {x -> a/Sqrt[2]}}"
    );
  }

  #[test]
  fn symbolic_quadratic_general() {
    assert_eq!(
      interpret("Solve[a*x^2 + b*x + c == 0, x]").unwrap(),
      "{{x -> (-b - Sqrt[b^2 - 4*a*c])/(2*a)}, {x -> (-b + Sqrt[b^2 - 4*a*c])/(2*a)}}"
    );
  }

  #[test]
  fn quartic_factor_based() {
    assert_eq!(
      interpret("Solve[x^4 - x == 0, x]").unwrap(),
      "{{x -> 0}, {x -> 1}, {x -> -(-1)^(1/3)}, {x -> (-1)^(2/3)}}"
    );
  }

  #[test]
  fn cubic_factor_based() {
    assert_eq!(
      interpret("Solve[x^3 - 1 == 0, x]").unwrap(),
      "{{x -> 1}, {x -> -(-1)^(1/3)}, {x -> (-1)^(2/3)}}"
    );
  }

  #[test]
  fn cubic_integer_roots_sorted() {
    // Solutions must be sorted ascending, matching Wolfram Language
    assert_eq!(
      interpret("Solve[x^3 - 6 x^2 + 11 x - 6 == 0, x]").unwrap(),
      "{{x -> 1}, {x -> 2}, {x -> 3}}"
    );
  }

  #[test]
  fn quartic_integer_roots_sorted() {
    // Solutions must be sorted ascending, matching Wolfram Language
    assert_eq!(
      interpret("Solve[x^4 - 10 x^2 + 9 == 0, x]").unwrap(),
      "{{x -> -3}, {x -> -1}, {x -> 1}, {x -> 3}}"
    );
  }

  #[test]
  fn cyclotomic_phi3() {
    assert_eq!(
      interpret("Solve[x^2 + x + 1 == 0, x]").unwrap(),
      "{{x -> -(-1)^(1/3)}, {x -> (-1)^(2/3)}}"
    );
  }

  #[test]
  fn cyclotomic_phi6() {
    assert_eq!(
      interpret("Solve[x^2 - x + 1 == 0, x]").unwrap(),
      "{{x -> (-1)^(1/3)}, {x -> -(-1)^(2/3)}}"
    );
  }

  #[test]
  fn nonlinear_system() {
    assert_eq!(
      interpret("Solve[{3 x ^ 2 - 3 y == 0, 3 y ^ 2 - 3 x == 0}, {x, y}]")
        .unwrap(),
      "{{x -> 0, y -> 0}, {x -> 1, y -> 1}, {x -> -(-1)^(1/3), y -> (-1)^(2/3)}, {x -> (-1)^(2/3), y -> -(-1)^(1/3)}}"
    );
  }

  #[test]
  fn linear_system_with_and_in_list() {
    // Solve[{eq1 && eq2}, {x, y}] should behave like
    // Solve[{eq1, eq2}, {x, y}] — the conjunction inside the list is
    // flattened into individual equations.
    assert_eq!(
      interpret(
        "{x, y} /. Solve[{2 x + y == 12 && x + 4 y == 34}, {x, y}] // First"
      )
      .unwrap(),
      "{2, 8}"
    );
  }

  #[test]
  fn underdetermined_quadratic_cubic() {
    // Single equation with two variables: solve for lowest-degree variable
    assert_eq!(
      interpret("Solve[x^2 - y^3 == 1, {x, y}]").unwrap(),
      "{{x -> -Sqrt[1 + y^3]}, {x -> Sqrt[1 + y^3]}}"
    );
  }

  #[test]
  fn underdetermined_quintic_quadratic() {
    // Prefers y (degree 2) over x (degree 5)
    assert_eq!(
      interpret("Solve[x^5 - y^2 == 1, {x, y}]").unwrap(),
      "{{y -> -Sqrt[-1 + x^5]}, {y -> Sqrt[-1 + x^5]}}"
    );
  }

  #[test]
  fn solve_pure_cubic_symbolic() {
    assert_eq!(
      interpret("Solve[y^3 == a, y]").unwrap(),
      "{{y -> a^(1/3)}, {y -> -((-1)^(1/3)*a^(1/3))}, {y -> (-1)^(2/3)*a^(1/3)}}"
    );
  }

  #[test]
  fn solve_pure_quintic_symbolic() {
    assert_eq!(
      interpret("Solve[y^5 == a, y]").unwrap(),
      "{{y -> a^(1/5)}, {y -> -((-1)^(1/5)*a^(1/5))}, {y -> (-1)^(2/5)*a^(1/5)}, {y -> -((-1)^(3/5)*a^(1/5))}, {y -> (-1)^(4/5)*a^(1/5)}}"
    );
  }

  #[test]
  fn solve_sqrt_equation() {
    assert_eq!(interpret("Solve[Sqrt[x] == 3, x]").unwrap(), "{{x -> 9}}");
  }

  #[test]
  fn solve_sqrt_nested() {
    assert_eq!(
      interpret("Solve[Sqrt[x + 1] == 4, x]").unwrap(),
      "{{x -> 15}}"
    );
  }

  #[test]
  fn solve_log_equation() {
    assert_eq!(interpret("Solve[Log[x] == 2, x]").unwrap(), "{{x -> E^2}}");
  }

  #[test]
  fn solve_exp_equation() {
    // Matches wolframscript: returns the full complex solution with
    // ConditionalExpression, covering all integer branches.
    assert_eq!(
      interpret("Solve[Exp[x] == 1, x]").unwrap(),
      "{{x -> ConditionalExpression[(2*I)*Pi*C[1], Element[C[1], Integers]]}}"
    );
  }

  #[test]
  fn solve_log_with_linear_inner() {
    // Matches wolframscript's preferred form: (-1 + E^3)/2 over
    // -((1 - E^3)/2).
    assert_eq!(
      interpret("Solve[Log[2*x + 1] == 3, x]").unwrap(),
      "{{x -> (-1 + E^3)/2}}"
    );
  }
}

mod rsolve {
  use super::*;

  #[test]
  fn constant_coeff_second_order() {
    assert_eq!(
      interpret("RSolve[{a[n + 2] == a[n], a[0] == 1, a[1] == 4}, a, n]")
        .unwrap(),
      "{{a -> Function[{n}, (5 - 3*(-1)^n)/2]}}"
    );
  }
}

mod full_simplify {
  use super::*;

  #[test]
  fn algebraic_factoring() {
    assert_eq!(
      interpret("FullSimplify[x^2 + 2*x + 1]").unwrap(),
      "(1 + x)^2"
    );
  }

  #[test]
  fn trig_identity() {
    assert_eq!(interpret("FullSimplify[Sin[x]^2 + Cos[x]^2]").unwrap(), "1");
  }

  #[test]
  fn trig_ratio_cot() {
    assert_eq!(interpret("Simplify[Cos[x]/Sin[x]]").unwrap(), "Cot[x]");
  }

  #[test]
  fn trig_ratio_tan() {
    assert_eq!(interpret("Simplify[Sin[x]/Cos[x]]").unwrap(), "Tan[x]");
  }

  #[test]
  fn trig_ratio_with_arg() {
    assert_eq!(
      interpret("Simplify[Sin[2*x]/Cos[2*x]]").unwrap(),
      "Tan[2*x]"
    );
  }

  #[test]
  fn trig_with_symbolic_coefficients() {
    assert_eq!(
      interpret(
        "FullSimplify[{a^2*((-1 + Sin[theta])^2 + Cos[theta]^2), a^2*((1 + Sin[theta])^2 + Cos[theta]^2)}]"
      )
      .unwrap(),
      "{-2*a^2*(-1 + Sin[theta]), 2*a^2*(1 + Sin[theta])}"
    );
  }

  #[test]
  fn numeric_factoring() {
    assert_eq!(interpret("FullSimplify[3*x + 6]").unwrap(), "3*(2 + x)");
  }

  #[test]
  fn trivial() {
    assert_eq!(interpret("FullSimplify[5]").unwrap(), "5");
    assert_eq!(interpret("FullSimplify[x]").unwrap(), "x");
  }

  #[test]
  fn combine_like_denominator_fractions() {
    assert_eq!(interpret("FullSimplify[a/x + b/x]").unwrap(), "(a + b)/x");
  }

  #[test]
  fn abs_quotient() {
    // Abs[a]/Abs[b] → Abs[a/b] with expansion
    assert_eq!(
      interpret("FullSimplify[Abs[1 + x^3]/Abs[x]]").unwrap(),
      "Abs[x^(-1) + x^2]"
    );
  }

  #[test]
  fn abs_product() {
    // Abs[a]*Abs[b] → Abs[a*b]
    assert_eq!(
      interpret("FullSimplify[Abs[x]*Abs[y]]").unwrap(),
      "Abs[x*y]"
    );
  }

  #[test]
  fn combine_fractions_different_denominators() {
    assert_eq!(
      interpret(
        "FullSimplify[k*q/(2*a^4*(1 + s)^(3/2)) + k*q*(1 + s)^(9/4)/(2*a^4)]"
      )
      .unwrap(),
      "(k*q*(1 + (1 + s)^(15/4)))/(2*a^4*(1 + s)^(3/2))"
    );
  }

  // Regression: FullSimplify should partially factor sums whose terms split
  // into variable-disjoint groups. `1 + c^2 + 2 c d + d^2` has a constant
  // `1` plus a group connected by c,d that factors to `(c+d)^2`.
  #[test]
  fn partial_factor_disjoint_groups() {
    assert_eq!(
      interpret("FullSimplify[1 + c^2 + 2 c d + d^2]").unwrap(),
      "1 + (c + d)^2"
    );
  }

  // Regression: FullSimplify should recursively re-simplify inside a factored
  // product. After pulling `y` out of the sum, the remaining polynomial in x
  // should be collected by x with each coefficient factored in turn.
  #[test]
  fn nested_factor_after_common_factor() {
    assert_eq!(
      interpret(
        "FullSimplify[(a^2 + 2 a b + b^2) y + (2 a + 2 b) x y + (1 + c^2 + 2 c d + d^2) x^2 y]"
      )
      .unwrap(),
      "((a + b)^2 + 2*(a + b)*x + (1 + (c + d)^2)*x^2)*y"
    );
  }

  // Regression: FullSimplify should collect a multi-variable polynomial by a
  // chosen variable and factor each collected coefficient.
  #[test]
  fn collect_and_factor_coefficients() {
    assert_eq!(
      interpret(
        "FullSimplify[a^2 + 2 a b + b^2 + 2 a x + 2 b x + x^2 + c^2 x^2 + 2 c d x^2 + d^2 x^2]"
      )
      .unwrap(),
      "(a + b)^2 + 2*(a + b)*x + (1 + (c + d)^2)*x^2"
    );
  }
}

mod simplify_assumptions {
  use super::*;

  #[test]
  fn simplify_with_assumptions_option() {
    assert_eq!(
      interpret("Simplify[x + x, Assumptions -> x > 0]").unwrap(),
      "2*x"
    );
  }

  #[test]
  fn full_simplify_with_assumptions_option() {
    assert_eq!(
      interpret("FullSimplify[x + x, Assumptions -> x > 0]").unwrap(),
      "2*x"
    );
  }

  // Regression: Simplify should accept a direct assumption (not only
  // `Assumptions -> val`), and `Assuming[...]` should propagate to a nested
  // `Simplify[...]` call via `$Assumptions`.
  #[test]
  fn simplify_with_direct_assumption() {
    assert_eq!(interpret("Simplify[Sqrt[x^2], x > 0]").unwrap(), "x");
  }

  #[test]
  fn simplify_power_one_half_with_assumption() {
    // (x^2)^(1/2) is the unevaluated Power form of Sqrt[x^2].
    assert_eq!(interpret("Simplify[(x^2)^(1/2), x > 0]").unwrap(), "x");
  }

  #[test]
  fn assuming_propagates_to_simplify() {
    assert_eq!(
      interpret("Assuming[x > 0, Simplify[Sqrt[x^2]]]").unwrap(),
      "x"
    );
  }

  #[test]
  fn assuming_combines_with_inner_simplify_assumption() {
    // Outer Assuming and inner direct assumption should combine via And.
    assert_eq!(
      interpret("Assuming[x > 0, Simplify[Sqrt[x^2] + Sqrt[y^2], y > 0]]")
        .unwrap(),
      "x + y"
    );
  }

  #[test]
  fn assuming_propagates_to_simplify_multi_var() {
    assert_eq!(
      interpret("Assuming[x > 0 && y > 0, Simplify[Sqrt[x^2] + Sqrt[y^2]]]")
        .unwrap(),
      "x + y"
    );
  }
}

// Regression: Simplify should collapse nested continued-fraction-like
// expressions by combining inner fractions, not leave them in the
// `1 + (1 + x^(-1))^(-1)` form.
mod simplify_continued_fractions {
  use super::*;

  #[test]
  fn single_level_nested_inverse() {
    // 1/(1 + 1/x) → x/(1 + x)
    assert_eq!(interpret("Simplify[1/(1 + 1/x)]").unwrap(), "x/(1 + x)");
  }

  #[test]
  fn plus_with_single_level_nested_inverse() {
    // 1 + 1/(1 + 1/x) → 1 + x/(1 + x)
    assert_eq!(
      interpret("Simplify[1 + 1/(1 + 1/x)]").unwrap(),
      "1 + x/(1 + x)"
    );
  }

  #[test]
  fn two_level_nested_inverse() {
    // 1/(1 + 1/(1 + 1/x)) → (1 + x)/(1 + 2 x)
    assert_eq!(
      interpret("Simplify[1/(1 + 1/(1 + 1/x))]").unwrap(),
      "(1 + x)/(1 + 2*x)"
    );
  }

  #[test]
  fn plus_with_two_level_nested_inverse() {
    // 1 + 1/(1 + 1/(1 + 1/x)) → (2 + 3 x)/(1 + 2 x)
    assert_eq!(
      interpret("Simplify[1 + 1/(1 + 1/(1 + 1/x))]").unwrap(),
      "(2 + 3*x)/(1 + 2*x)"
    );
  }

  #[test]
  fn nested_inverse_with_symbolic_coefficients() {
    // a / (b + c/d) → a d / (c + b d)
    assert_eq!(
      interpret("Simplify[a/(b + c/d)]").unwrap(),
      "(a*d)/(c + b*d)"
    );
  }

  #[test]
  fn together_continued_fraction() {
    // Same input — Together alone should also combine fully.
    assert_eq!(
      interpret("Together[1 + 1/(1 + 1/(1 + 1/x))]").unwrap(),
      "(2 + 3*x)/(1 + 2*x)"
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
    assert_eq!(result, "x == 2 || x == -2");
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
    // Double root: x == 1 repeated twice (matching Wolfram behavior)
    let result = interpret("Roots[x^2 - 2*x + 1 == 0, x]").unwrap();
    assert_eq!(result, "x == 1 || x == 1");
  }
}

mod eliminate {
  use super::*;

  #[test]
  fn eliminate_single_variable_linear() {
    // Eliminate y from {x == 2 + y, y == z}
    let result = interpret("Eliminate[{x == 2 + y, y == z}, y]").unwrap();
    assert_eq!(result, "2 + z == x");
  }

  #[test]
  fn eliminate_single_from_two_linear() {
    // Eliminate a from {x == a + b, y == a - b}
    let result = interpret("Eliminate[{x == a + b, y == a - b}, a]").unwrap();
    assert_eq!(result, "x - y == 2*b");
  }

  #[test]
  fn eliminate_to_constant() {
    // Eliminate y from {x + y == 3, x - y == 1}
    let result = interpret("Eliminate[{x + y == 3, x - y == 1}, y]").unwrap();
    assert_eq!(result, "x == 2");
  }

  #[test]
  fn eliminate_with_product() {
    // Eliminate x from {a == x + y, b == x*y}
    let result = interpret("Eliminate[{a == x + y, b == x*y}, x]").unwrap();
    assert_eq!(result, "-b + a*y - y^2 == 0");
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
    // x == 5 → {x -> 5}
    assert_eq!(interpret("ToRules[x == 5]").unwrap(), "{x -> 5}");
  }

  #[test]
  fn to_rules_or_conditions() {
    // x == -2 || x == 2 → Sequence[{x -> -2}, {x -> 2}] (displays as "{x -> -2}{x -> 2}")
    assert_eq!(
      interpret("ToRules[x == -2 || x == 2]").unwrap(),
      "{x -> -2}{x -> 2}"
    );
  }

  #[test]
  fn to_rules_from_roots() {
    // Convert Roots output to Solve-style rules
    assert_eq!(
      interpret("ToRules[Roots[x^2 - 4 == 0, x]]").unwrap(),
      "{x -> 2}{x -> -2}"
    );
  }

  #[test]
  fn to_rules_and_conditions() {
    // x == 1 && y == 2 → {x -> 1, y -> 2}
    assert_eq!(
      interpret("ToRules[x == 1 && y == 2]").unwrap(),
      "{x -> 1, y -> 2}"
    );
  }

  #[test]
  fn to_rules_true() {
    assert_eq!(interpret("ToRules[True]").unwrap(), "{}");
  }

  #[test]
  fn to_rules_false() {
    // ToRules[False] returns Sequence[] (empty, matches Wolfram behavior)
    // When wrapped in ToString[(ToRules[False]), InputForm] → "InputForm"
    assert_eq!(interpret("ToRules[False]").unwrap(), "");
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
    assert_eq!(interpret("Reduce[x^2 < 4, x]").unwrap(), "-2 < x < 2");
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
      "-2 < x < 1"
    );
  }

  #[test]
  fn factored_inequality_leq() {
    assert_eq!(
      interpret("Reduce[(x - 1)*(x + 2) <= 0, x]").unwrap(),
      "-2 <= x <= 1"
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
      "3 < x < 5"
    );
  }

  #[test]
  fn combined_two_bounds() {
    assert_eq!(
      interpret("Reduce[x > 2 && x < 10, x]").unwrap(),
      "2 < x < 10"
    );
  }

  #[test]
  fn mixed_equation_inequality() {
    assert_eq!(
      interpret("Reduce[x^2 <= 4 && x > 0, x]").unwrap(),
      "0 < x <= 2"
    );
  }

  // ── Reduce InputForm: chained inequalities use Inequality[] head ──

  #[test]
  fn reduce_quadratic_inequality_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[x^2 < 4, x], InputForm]").unwrap(),
      "Inequality[-2, Less, x, Less, 2]"
    );
  }

  #[test]
  fn reduce_factored_inequality_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[(x - 1)*(x + 2) < 0, x], InputForm]").unwrap(),
      "Inequality[-2, Less, x, Less, 1]"
    );
  }

  #[test]
  fn reduce_factored_inequality_leq_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[(x - 1)*(x + 2) <= 0, x], InputForm]")
        .unwrap(),
      "Inequality[-2, LessEqual, x, LessEqual, 1]"
    );
  }

  #[test]
  fn reduce_combined_inequalities_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[x > 0 && x < 5 && x > 3, x], InputForm]")
        .unwrap(),
      "Inequality[3, Less, x, Less, 5]"
    );
  }

  #[test]
  fn reduce_two_bounds_input_form() {
    assert_eq!(
      interpret("ToString[Reduce[x > 2 && x < 10, x], InputForm]").unwrap(),
      "Inequality[2, Less, x, Less, 10]"
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

  // ── Multi-variable nonlinear ──

  #[test]
  fn reduce_two_var_nonlinear() {
    assert_eq!(
      interpret("Reduce[x^2 - y^3 == 1, {x, y}]").unwrap(),
      "y == (-1 + x^2)^(1/3) || y == -((-1)^(1/3)*(-1 + x^2)^(1/3)) || y == (-1)^(2/3)*(-1 + x^2)^(1/3)"
    );
  }

  #[test]
  fn reduce_two_var_solve_for_lower_degree() {
    assert_eq!(
      interpret("Reduce[x^5 - y^2 == 1, {x, y}]").unwrap(),
      "y == -Sqrt[-1 + x^5] || y == Sqrt[-1 + x^5]"
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

  #[test]
  fn quadratic_ineq_with_sign_constraint_negative() {
    // x^2 > 4 && x < 0  →  x < -2
    assert_eq!(interpret("Reduce[x^2 > 4 && x < 0, x]").unwrap(), "x < -2");
  }

  #[test]
  fn quadratic_ineq_with_sign_constraint_positive() {
    // x^2 > 4 && x > 0  →  x > 2
    assert_eq!(interpret("Reduce[x^2 > 4 && x > 0, x]").unwrap(), "x > 2");
  }

  #[test]
  fn quadratic_ineq_with_range_constraint() {
    // x^2 > 4 && x > 0 && x < 10  →  2 < x < 10
    assert_eq!(
      interpret("Reduce[x^2 > 4 && x > 0 && x < 10, x]").unwrap(),
      "2 < x < 10"
    );
  }
}

mod nsolve {
  use super::*;

  #[test]
  fn linear_equation() {
    assert_eq!(interpret("NSolve[x - 5 == 0, x]").unwrap(), "{{x -> 5.}}");
  }

  #[test]
  fn quadratic_integer_roots() {
    assert_eq!(
      interpret("NSolve[x^2 - 4 == 0, x]").unwrap(),
      "{{x -> -2.}, {x -> 2.}}"
    );
  }

  #[test]
  fn quadratic_irrational_roots() {
    assert_eq!(
      interpret("NSolve[x^2 + x - 1 == 0, x]").unwrap(),
      "{{x -> -1.618033988749895}, {x -> 0.6180339887498948}}"
    );
  }

  #[test]
  fn quadratic_complex_roots() {
    assert_eq!(
      interpret("NSolve[x^2 + 1 == 0, x]").unwrap(),
      "{{x -> 0. - 1.*I}, {x -> 0. + 1.*I}}"
    );
  }

  #[test]
  fn cubic_roots() {
    assert_eq!(
      interpret("NSolve[x^3 - 3*x^2 + 2*x == 0, x]").unwrap(),
      "{{x -> 0.}, {x -> 1.}, {x -> 2.}}"
    );
  }

  #[test]
  fn with_user_defined_function() {
    assert_eq!(
      interpret("f[x_] := x^2 + x + 1; NSolve[f[b] - 2 == 0, b]").unwrap(),
      "{{b -> -1.618033988749895}, {b -> 0.6180339887498948}}"
    );
  }

  #[test]
  fn rational_solution() {
    assert_eq!(
      interpret("NSolve[2*x - 1 == 0, x]").unwrap(),
      "{{x -> 0.5}}"
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

  #[test]
  fn bessel_j_root() {
    // FindRoot should work with BesselJ using numerical derivatives
    let result = interpret("FindRoot[BesselJ[0,x], {x,10.5}]").unwrap();
    assert!(
      result.starts_with("{x -> 18.07106396"),
      "Expected root near 18.071..., got: {}",
      result
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

  #[test]
  fn negative_levelspec_leaves() {
    // {-1} matches all atoms (Depth = 1).
    assert_eq!(
      interpret("Replace[f[1, g[2, h[3]]], _Integer -> 0, {-1}]").unwrap(),
      "f[0, g[0, h[0]]]"
    );
  }

  #[test]
  fn negative_levelspec_leaves_in_list() {
    assert_eq!(
      interpret("Replace[{1, {2, {3, 4}}}, _Integer -> 0, {-1}]").unwrap(),
      "{0, {0, {0, 0}}}"
    );
  }

  #[test]
  fn negative_levelspec_subtree_depth_two() {
    // {-2} matches subtrees with Depth = 2 (e.g. h[3] in this expression).
    assert_eq!(
      interpret("Replace[f[1, g[2, h[3]]], h[_] -> X, {-2}]").unwrap(),
      "f[1, g[2, X]]"
    );
  }

  #[test]
  fn negative_levelspec_no_match_at_wrong_depth() {
    // {-3} of f[1,g[2,h[3]]] only matches the subtree g[2,h[3]] (Depth = 3).
    assert_eq!(
      interpret("Replace[f[1, g[2, h[3]]], _ -> X, {-3}]").unwrap(),
      "f[1, X]"
    );
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

  #[test]
  fn distribute_over_list() {
    assert_eq!(
      interpret("Distribute[f[{a, b}, {c, d}], List]").unwrap(),
      "{f[a, c], f[a, d], f[b, c], f[b, d]}"
    );
  }

  #[test]
  fn distribute_nested_lists() {
    assert_eq!(
      interpret("Distribute[{{1, 2}, {3, 4}}, List]").unwrap(),
      "{{1, 3}, {1, 4}, {2, 3}, {2, 4}}"
    );
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

mod solve_expression_target {
  use super::*;

  #[test]
  fn solve_for_function_call() {
    assert_eq!(
      interpret("Solve[f[x + y] == 3, f[x + y]]").unwrap(),
      "{{f[x + y] -> 3}}"
    );
  }
}

mod solve_with_domain {
  use super::*;

  #[test]
  fn reals_no_complex() {
    assert_eq!(interpret("Solve[x^2 == -1, x, Reals]").unwrap(), "{}");
  }

  #[test]
  fn reals_with_solutions() {
    assert_eq!(
      interpret("Solve[x^2 == 1, x, Reals]").unwrap(),
      "{{x -> -1}, {x -> 1}}"
    );
  }

  #[test]
  fn integers_filters_non_integer() {
    assert_eq!(
      interpret("Solve[-4 - 4 x + x^4 + x^5 == 0, x, Integers]").unwrap(),
      "{{x -> -1}}"
    );
  }

  #[test]
  fn integers_no_solutions() {
    assert_eq!(interpret("Solve[x^4 == 4, x, Integers]").unwrap(), "{}");
  }
}

mod solve_always {
  use super::*;

  #[test]
  fn linear_single_variable() {
    assert_eq!(
      interpret("SolveAlways[a*x + b == 0, x]").unwrap(),
      "{{a -> 0, b -> 0}}"
    );
  }

  #[test]
  fn quadratic_with_offsets() {
    assert_eq!(
      interpret("SolveAlways[(a - 2)*x^2 + (b + 1)*x + c == 0, x]").unwrap(),
      "{{a -> 2, b -> -1, c -> 0}}"
    );
  }

  #[test]
  fn matching_polynomial() {
    assert_eq!(
      interpret("SolveAlways[a*x^2 + b*x + c == 3*x^2 - 5*x + 7, x]").unwrap(),
      "{{a -> 3, b -> -5, c -> 7}}"
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
      "{{a -> 2, b -> -1, c -> 0}}"
    );
  }

  #[test]
  fn multivariate_all_zero() {
    assert_eq!(
      interpret("SolveAlways[a*x + b*y + c == 0, {x, y}]").unwrap(),
      "{{a -> 0, b -> 0, c -> 0}}"
    );
  }

  #[test]
  fn list_form_single_var() {
    assert_eq!(
      interpret("SolveAlways[a*x + b == 0, {x}]").unwrap(),
      "{{a -> 0, b -> 0}}"
    );
  }

  #[test]
  fn quadratic_cross_terms() {
    assert_eq!(
      interpret("SolveAlways[a*x^2 + b*x*y + c*y^2 == 0, {x, y}]").unwrap(),
      "{{a -> 0, b -> 0, c -> 0}}"
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

mod expand_numerator {
  use super::*;

  #[test]
  fn basic_square() {
    assert_eq!(
      interpret("ExpandNumerator[(1 + x)^2/(1 + y)^2]").unwrap(),
      "(1 + 2*x + x^2)/(1 + y)^2"
    );
  }

  #[test]
  fn cubic_numerator() {
    assert_eq!(
      interpret("ExpandNumerator[(a + b)^3/(c + d)]").unwrap(),
      "(a^3 + 3*a^2*b + 3*a*b^2 + b^3)/(c + d)"
    );
  }

  #[test]
  fn no_fraction() {
    assert_eq!(interpret("ExpandNumerator[x + 1]").unwrap(), "1 + x");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ExpandNumerator]").unwrap(),
      "{Protected}"
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

mod power_expand {
  use super::*;

  #[test]
  fn product_power() {
    assert_eq!(interpret("PowerExpand[(a*b)^s]").unwrap(), "a^s*b^s");
  }

  #[test]
  fn product_power_three_factors() {
    assert_eq!(interpret("PowerExpand[(a*b*c)^s]").unwrap(), "a^s*b^s*c^s");
  }

  #[test]
  fn nested_power() {
    assert_eq!(interpret("PowerExpand[(a^r)^s]").unwrap(), "a^(r*s)");
  }

  #[test]
  fn quotient_power() {
    assert_eq!(interpret("PowerExpand[(x/y)^n]").unwrap(), "x^n/y^n");
  }

  #[test]
  fn integer_exponent() {
    assert_eq!(interpret("PowerExpand[(x*y)^2]").unwrap(), "x^2*y^2");
  }

  #[test]
  fn numeric_factor_power() {
    assert_eq!(interpret("PowerExpand[(2*x)^n]").unwrap(), "2^n*x^n");
  }

  #[test]
  fn compound_powers_in_product() {
    assert_eq!(
      interpret("PowerExpand[(x^a*y^b)^c]").unwrap(),
      "x^(a*c)*y^(b*c)"
    );
  }

  #[test]
  fn sqrt_product() {
    assert_eq!(
      interpret("PowerExpand[Sqrt[a*b]]").unwrap(),
      "Sqrt[a]*Sqrt[b]"
    );
  }

  #[test]
  fn sqrt_power_product() {
    assert_eq!(interpret("PowerExpand[Sqrt[x^2*y^4]]").unwrap(), "x*y^2");
  }

  #[test]
  fn sqrt_squared() {
    assert_eq!(interpret("PowerExpand[Sqrt[x^2]]").unwrap(), "x");
  }

  #[test]
  fn fractional_power() {
    assert_eq!(interpret("PowerExpand[(x^2)^(1/3)]").unwrap(), "x^(2/3)");
  }

  #[test]
  fn sqrt_three_factors() {
    assert_eq!(
      interpret("PowerExpand[(a*b*c)^(1/2)]").unwrap(),
      "Sqrt[a]*Sqrt[b]*Sqrt[c]"
    );
  }

  #[test]
  fn half_power_compound() {
    assert_eq!(
      interpret("PowerExpand[(x^2*y^3)^(1/2)]").unwrap(),
      "x*y^(3/2)"
    );
  }

  #[test]
  fn log_product() {
    assert_eq!(
      interpret("PowerExpand[Log[a*b]]").unwrap(),
      "Log[a] + Log[b]"
    );
  }

  #[test]
  fn log_product_three() {
    assert_eq!(
      interpret("PowerExpand[Log[a*b*c]]").unwrap(),
      "Log[a] + Log[b] + Log[c]"
    );
  }

  #[test]
  fn log_power() {
    assert_eq!(interpret("PowerExpand[Log[a^b]]").unwrap(), "b*Log[a]");
  }

  #[test]
  fn log_quotient() {
    assert_eq!(
      interpret("PowerExpand[Log[a/b]]").unwrap(),
      "Log[a] - Log[b]"
    );
  }

  #[test]
  fn log_sqrt() {
    assert_eq!(interpret("PowerExpand[Log[Sqrt[x]]]").unwrap(), "Log[x]/2");
  }

  #[test]
  fn log_e_power() {
    assert_eq!(interpret("PowerExpand[Log[E^x]]").unwrap(), "x");
  }

  #[test]
  fn log_compound_product_powers() {
    assert_eq!(
      interpret("PowerExpand[Log[x^2*y^3]]").unwrap(),
      "2*Log[x] + 3*Log[y]"
    );
  }

  #[test]
  fn log_symbolic_powers() {
    assert_eq!(
      interpret("PowerExpand[Log[x^a*y^b*z^c]]").unwrap(),
      "a*Log[x] + b*Log[y] + c*Log[z]"
    );
  }

  #[test]
  fn exp_log_identity() {
    assert_eq!(interpret("PowerExpand[Exp[x*Log[y]]]").unwrap(), "y^x");
  }

  #[test]
  fn sum_passthrough() {
    assert_eq!(interpret("PowerExpand[(a+b)^n]").unwrap(), "(a + b)^n");
  }

  #[test]
  fn log_sum_passthrough() {
    assert_eq!(interpret("PowerExpand[Log[a+b]]").unwrap(), "Log[a + b]");
  }

  #[test]
  fn atom_passthrough() {
    assert_eq!(interpret("PowerExpand[x]").unwrap(), "x");
    assert_eq!(interpret("PowerExpand[5]").unwrap(), "5");
  }

  #[test]
  fn additive_passthrough() {
    assert_eq!(interpret("PowerExpand[x+y]").unwrap(), "x + y");
  }

  #[test]
  fn thread_over_list() {
    assert_eq!(
      interpret("PowerExpand[{(a*b)^s, Log[a*b], Sqrt[x*y]}]").unwrap(),
      "{a^s*b^s, Log[a] + Log[b], Sqrt[x]*Sqrt[y]}"
    );
  }

  #[test]
  fn nested_in_function() {
    assert_eq!(
      interpret("PowerExpand[Sin[(a*b)^s]]").unwrap(),
      "Sin[a^s*b^s]"
    );
  }

  #[test]
  fn sum_of_powers() {
    assert_eq!(
      interpret("PowerExpand[(x+y)^n + (a*b)^s]").unwrap(),
      "a^s*b^s + (x + y)^n"
    );
  }

  #[test]
  fn with_assumptions() {
    // Second argument (Assumptions) is accepted
    assert_eq!(
      interpret("PowerExpand[(a*b)^s, Assumptions -> a > 0]").unwrap(),
      "a^s*b^s"
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

mod factor_square_free {
  use super::*;

  #[test]
  fn basic_repeated_factor() {
    assert_eq!(
      interpret("FactorSquareFree[x^5 - x^4 - x + 1]").unwrap(),
      "(-1 + x)^2*(1 + x + x^2 + x^3)"
    );
  }

  #[test]
  fn with_x_factor() {
    assert_eq!(
      interpret("FactorSquareFree[x^4 - 2*x^3 + x^2]").unwrap(),
      "(-1 + x)^2*x^2"
    );
  }

  #[test]
  fn with_integer_content() {
    assert_eq!(
      interpret("FactorSquareFree[12*x^3 + 36*x^2 + 36*x + 12]").unwrap(),
      "12*(1 + x)^3"
    );
  }

  #[test]
  fn square_free_unchanged() {
    assert_eq!(interpret("FactorSquareFree[x^6 - 1]").unwrap(), "-1 + x^6");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[FactorSquareFree]").unwrap(),
      "{Listable, Protected}"
    );
  }
}

mod factor_terms_list {
  use super::*;

  #[test]
  fn common_integer_factor() {
    assert_eq!(
      interpret("FactorTermsList[6*x^2 - 12*x + 6]").unwrap(),
      "{6, 1 - 2*x + x^2}"
    );
  }

  #[test]
  fn no_common_factor() {
    assert_eq!(
      interpret("FactorTermsList[x^2 + 2*x + 1]").unwrap(),
      "{1, 1 + 2*x + x^2}"
    );
  }

  #[test]
  fn negative_leading() {
    assert_eq!(
      interpret("FactorTermsList[-3*x^2 + 6*x - 9]").unwrap(),
      "{-3, 3 - 2*x + x^2}"
    );
  }

  #[test]
  fn constant_input() {
    assert_eq!(interpret("FactorTermsList[5]").unwrap(), "{5, 1}");
  }

  #[test]
  fn no_numeric_content() {
    assert_eq!(
      interpret("FactorTermsList[x^3 + x]").unwrap(),
      "{1, x + x^3}"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[FactorTermsList]").unwrap(),
      "{Protected}"
    );
  }
}

mod refine {
  use super::*;

  #[test]
  fn sqrt_x_squared_positive() {
    assert_eq!(interpret("Refine[Sqrt[x^2], x > 0]").unwrap(), "x");
  }

  #[test]
  fn sqrt_y_squared_positive() {
    assert_eq!(interpret("Refine[Sqrt[y^2], y > 0]").unwrap(), "y");
  }

  #[test]
  fn abs_x_positive() {
    assert_eq!(interpret("Refine[Abs[x], x > 0]").unwrap(), "x");
  }

  #[test]
  fn abs_y_positive() {
    assert_eq!(interpret("Refine[Abs[y], y > 0]").unwrap(), "y");
  }

  #[test]
  fn sqrt_x_squared_no_assumption() {
    // Without assumptions, Sqrt[x^2] should stay as Sqrt[x^2]
    assert_eq!(interpret("Sqrt[x^2]").unwrap(), "Sqrt[x^2]");
  }

  #[test]
  fn sqrt_integer_squared() {
    // Sqrt[4] = 2, Sqrt[9] = 3 (exact integers, no assumptions needed)
    assert_eq!(interpret("Sqrt[4]").unwrap(), "2");
    assert_eq!(interpret("Sqrt[9]").unwrap(), "3");
  }

  #[test]
  fn sqrt_known_non_negative_squared() {
    // Sqrt[(positive_constant)^2] should simplify without Refine
    assert_eq!(interpret("Sqrt[Pi^2]").unwrap(), "Pi");
  }

  #[test]
  fn refine_nested_expression() {
    assert_eq!(
      interpret("Refine[Sqrt[x^2] + Sqrt[y^2], x > 0 && y > 0]").unwrap(),
      "x + y"
    );
  }

  #[test]
  fn refine_no_simplification_needed() {
    // Expression that doesn't benefit from assumptions
    assert_eq!(interpret("Refine[x + 1, x > 0]").unwrap(), "1 + x");
  }

  #[test]
  fn sqrt_x_squared_x_gt_positive() {
    // x > 2 implies x > 0, so Sqrt[x^2] → x
    assert_eq!(interpret("Refine[Sqrt[x^2], x > 2]").unwrap(), "x");
  }

  #[test]
  fn sqrt_x_squared_x_ge_positive() {
    // x >= 5 implies x > 0, so Sqrt[x^2] → x
    assert_eq!(interpret("Refine[Sqrt[x^2], x >= 5]").unwrap(), "x");
  }

  #[test]
  fn sqrt_x_squared_positive_lt_x() {
    // 3 < x implies x > 0, so Sqrt[x^2] → x
    assert_eq!(interpret("Refine[Sqrt[x^2], 3 < x]").unwrap(), "x");
  }

  #[test]
  fn sqrt_x_squared_positive_le_x() {
    // 1 <= x implies x > 0, so Sqrt[x^2] → x
    assert_eq!(interpret("Refine[Sqrt[x^2], 1 <= x]").unwrap(), "x");
  }

  #[test]
  fn abs_x_gt_positive() {
    // x > 7 implies x > 0, so Abs[x] → x
    assert_eq!(interpret("Refine[Abs[x], x > 7]").unwrap(), "x");
  }

  // --- Single argument ---

  #[test]
  fn single_arg_symbol() {
    assert_eq!(interpret("Refine[x]").unwrap(), "x");
  }

  #[test]
  fn single_arg_numeric() {
    assert_eq!(interpret("Refine[Abs[2]]").unwrap(), "2");
  }

  // --- Negative variable assumptions ---

  #[test]
  fn abs_x_negative() {
    assert_eq!(interpret("Refine[Abs[x], x < 0]").unwrap(), "-x");
  }

  #[test]
  fn sqrt_x_squared_negative() {
    assert_eq!(interpret("Refine[Sqrt[x^2], x < 0]").unwrap(), "-x");
  }

  // --- General (x^n)^(1/m) simplification ---

  #[test]
  fn cube_root_x_cubed_positive() {
    assert_eq!(interpret("Refine[(x^3)^(1/3), x >= 0]").unwrap(), "x");
  }

  #[test]
  fn fifth_root_x_fifth_positive() {
    assert_eq!(interpret("Refine[(x^5)^(1/5), x >= 0]").unwrap(), "x");
  }

  #[test]
  fn fourth_root_x_fourth_positive() {
    assert_eq!(interpret("Refine[(x^4)^(1/4), x >= 0]").unwrap(), "x");
  }

  #[test]
  fn fourth_root_x_fourth_negative() {
    // x^4 is always positive, (x^4)^(1/4) = |x| = -x when x < 0
    assert_eq!(interpret("Refine[(x^4)^(1/4), x < 0]").unwrap(), "-x");
  }

  #[test]
  fn sixth_power_cube_root_positive() {
    // (x^6)^(1/3) = x^2 when x >= 0
    assert_eq!(interpret("Refine[(x^6)^(1/3), x >= 0]").unwrap(), "x^2");
  }

  #[test]
  fn sixth_power_cube_root_negative() {
    // (x^6)^(1/3) = x^2 when x < 0 (x^6 always positive, result is |x|^2 = x^2)
    assert_eq!(interpret("Refine[(x^6)^(1/3), x < 0]").unwrap(), "x^2");
  }

  // --- Sign function ---

  #[test]
  fn sign_x_positive() {
    assert_eq!(interpret("Refine[Sign[x], x > 0]").unwrap(), "1");
  }

  #[test]
  fn sign_x_negative() {
    assert_eq!(interpret("Refine[Sign[x], x < 0]").unwrap(), "-1");
  }

  #[test]
  fn sign_x_gt_5() {
    // x > 5 implies positive
    assert_eq!(interpret("Refine[Sign[x], x > 5]").unwrap(), "1");
  }

  // --- Arg function ---

  #[test]
  fn arg_x_positive() {
    assert_eq!(interpret("Refine[Arg[x], x > 0]").unwrap(), "0");
  }

  #[test]
  fn arg_x_negative() {
    assert_eq!(interpret("Refine[Arg[x], x < 0]").unwrap(), "Pi");
  }

  // --- Re/Im with Element assumptions ---

  #[test]
  fn re_x_real() {
    assert_eq!(interpret("Refine[Re[x], Element[x, Reals]]").unwrap(), "x");
  }

  #[test]
  fn im_x_real() {
    assert_eq!(interpret("Refine[Im[x], Element[x, Reals]]").unwrap(), "0");
  }

  // --- Floor/Ceiling with Element[x, Integers] ---

  #[test]
  fn floor_x_integer() {
    assert_eq!(
      interpret("Refine[Floor[x], Element[x, Integers]]").unwrap(),
      "x"
    );
  }

  #[test]
  fn ceiling_x_integer() {
    assert_eq!(
      interpret("Refine[Ceiling[x], Element[x, Integers]]").unwrap(),
      "x"
    );
  }

  // --- Inequality simplification under assumptions ---

  #[test]
  fn x_gt_0_given_x_gt_1() {
    assert_eq!(interpret("Refine[x > 0, x > 1]").unwrap(), "True");
  }

  #[test]
  fn x_lt_0_given_x_gt_1() {
    assert_eq!(interpret("Refine[x < 0, x > 1]").unwrap(), "False");
  }

  #[test]
  fn x_geq_0_given_x_gt_0() {
    assert_eq!(interpret("Refine[x >= 0, x > 0]").unwrap(), "True");
  }

  #[test]
  fn same_inequality() {
    assert_eq!(interpret("Refine[x > 0, x > 0]").unwrap(), "True");
  }

  // --- Compound assumptions ---

  #[test]
  fn abs_sum_compound() {
    assert_eq!(
      interpret("Refine[Abs[x] + Abs[y], x > 0 && y > 0]").unwrap(),
      "x + y"
    );
  }

  #[test]
  fn abs_product_positive() {
    assert_eq!(
      interpret("Refine[Abs[x*y], x > 0 && y > 0]").unwrap(),
      "x*y"
    );
  }

  // --- Positive var implied by Element and inequality ---

  #[test]
  fn abs_x_reals_nonneg() {
    assert_eq!(
      interpret("Refine[Abs[x], Element[x, Reals] && x >= 0]").unwrap(),
      "x"
    );
  }

  // --- Element with Alternatives pattern ---

  #[test]
  fn element_alternatives_reals() {
    assert_eq!(
      interpret("Refine[Re[a + b I], Element[a | b, Reals]]").unwrap(),
      "a"
    );
  }

  #[test]
  fn element_alternatives_integers() {
    assert_eq!(
      interpret("Refine[Floor[x], Element[x, Integers]]").unwrap(),
      "x"
    );
  }

  // --- Sqrt[x^2] with x ∈ Reals → Abs[x] ---

  #[test]
  fn sqrt_x_squared_real() {
    assert_eq!(
      interpret("Refine[Sqrt[x^2], Element[x, Reals]]").unwrap(),
      "Abs[x]"
    );
  }

  // --- Power rules ---

  #[test]
  fn power_of_power_bounded_exp() {
    assert_eq!(interpret("Refine[(a^b)^c, -1 < b < 1]").unwrap(), "a^(b*c)");
  }

  #[test]
  fn combine_power_product_positive_bases() {
    assert_eq!(
      interpret("Refine[a^p b^p, a > 0 && b > 0]").unwrap(),
      "(a*b)^p"
    );
  }

  // --- Log simplifications ---

  #[test]
  fn log_negative_var() {
    assert_eq!(
      interpret("Refine[Log[x], x < 0]").unwrap(),
      "I*Pi + Log[-x]"
    );
  }

  #[test]
  fn log_power_bounded_exp() {
    assert_eq!(
      interpret("Refine[Log[x^p], -1 < p < 1]").unwrap(),
      "p*Log[x]"
    );
  }

  // --- Trig with integer multiples of Pi ---

  #[test]
  fn sin_k_pi_integer() {
    assert_eq!(
      interpret("Refine[Sin[k Pi], Element[k, Integers]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cos_x_plus_k_pi_integer() {
    assert_eq!(
      interpret("Refine[Cos[x + k Pi], Element[k, Integers]]").unwrap(),
      "(-1)^k*Cos[x]"
    );
  }

  // --- ArcTan[Tan[x]] ---

  #[test]
  fn arctan_tan_in_range() {
    assert_eq!(
      interpret("Refine[ArcTan[Tan[x]], -Pi/2 < Re[x] < Pi/2]").unwrap(),
      "x"
    );
  }

  // --- Algebraic comparisons ---

  #[test]
  fn equation_by_substitution() {
    assert_eq!(
      interpret("Refine[a^2 - b^2 + 1 == 0, a + b == 0]").unwrap(),
      "False"
    );
  }

  #[test]
  fn quadratic_form_nonneg() {
    assert_eq!(
      interpret("Refine[a^2 - a b + b^2 >= 0, Element[a | b, Reals]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn sign_positive_definite() {
    assert_eq!(
      interpret("Refine[Sign[x^2 - x y + y^2 + 1], Element[x | y, Reals]]")
        .unwrap(),
      "1"
    );
  }

  // --- Element membership ---

  #[test]
  fn element_real_positive_division() {
    assert_eq!(
      interpret(
        "Refine[Element[(2 x + x^p)/(x Gamma[x + 2]), Reals], x > 0 && p > 0]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn element_integer_floor_power() {
    assert_eq!(
      interpret("Refine[Element[2 k^3 Floor[x]^k, Integers], Element[k, Integers] && k > 0 && Element[x, Reals]]").unwrap(),
      "True"
    );
  }

  // --- Floor/Ceiling with compound expressions ---

  #[test]
  fn floor_integer_linear() {
    assert_eq!(
      interpret("Refine[Floor[2 a + 1], Element[a, Integers]]").unwrap(),
      "1 + 2*a"
    );
  }

  #[test]
  fn ceiling_bounded_var() {
    assert_eq!(interpret("Refine[Ceiling[x], 2 < x <= 3]").unwrap(), "3");
  }

  // --- FractionalPart and Mod ---

  #[test]
  fn fractional_part_negative_with_mod() {
    assert_eq!(
      interpret("Refine[FractionalPart[a], a < 0 && Mod[a, 1] == 1/3]")
        .unwrap(),
      "-2/3"
    );
  }

  #[test]
  fn mod_from_integer_element() {
    assert_eq!(
      interpret("Refine[Mod[a, 4], Element[(a + 3)/4, Integers]]").unwrap(),
      "1"
    );
  }

  // --- Assuming + Refine ---

  #[test]
  fn assuming_refine_compound_comparison() {
    assert_eq!(
      interpret("Assuming[x >= 0 && y < 0, Refine[x - y > 0]]").unwrap(),
      "True"
    );
  }

  // --- Nonnegative variable handling ---

  #[test]
  fn sqrt_x_squared_nonneg() {
    assert_eq!(interpret("Refine[Sqrt[x^2], x >= 0]").unwrap(), "x");
  }

  #[test]
  fn inequality_false_under_sum_of_squares_constraint() {
    // (x-1)^2 + (y-2)^2 >= 2 when x^2 + y^2 <= 1, so < 3/2 is False
    assert_eq!(
      interpret("Refine[(x - 1)^2 + (y - 2)^2 < 3/2, x^2 + y^2 <= 1]").unwrap(),
      "False"
    );
  }
}

mod simplify_solve_verification {
  use super::*;

  #[test]
  fn simplify_quadratic_formula_substitution() {
    // Substituting quadratic formula roots back into polynomial should give 0
    assert_eq!(
      interpret(
        "sol = Solve[a x^2 + b x + c == 0, x]; Simplify[a x^2 + b x + c /. sol]"
      )
      .unwrap(),
      "{0, 0}"
    );
  }

  #[test]
  fn simplify_threads_over_list() {
    assert_eq!(interpret("Simplify[{1 + 1, 2 + 3}]").unwrap(), "{2, 5}");
  }

  #[test]
  fn together_fraction_power() {
    // Together should correctly handle (sum/product)^n terms
    assert_eq!(
      interpret("Together[a*(x/a)^2 + x]").unwrap(),
      "(a*x + x^2)/a"
    );
  }
}

mod expand_fraction_power {
  use super::*;

  #[test]
  fn expand_fraction_squared() {
    // (x+y)^2/z^2 expanded, displaying negative exponents as fractions
    assert_eq!(
      interpret("Expand[((x + y)/z)^2]").unwrap(),
      "x^2/z^2 + (2*x*y)/z^2 + y^2/z^2"
    );
  }

  #[test]
  fn expand_product_power() {
    assert_eq!(interpret("Expand[(2*a)^2]").unwrap(), "4*a^2");
  }

  #[test]
  fn expand_product_power_three() {
    assert_eq!(interpret("Expand[(3*x)^3]").unwrap(), "27*x^3");
  }
}

mod root {
  use super::*;

  #[test]
  fn sqrt_2_first_root() {
    assert_eq!(interpret("Root[#^2 - 2 &, 1]").unwrap(), "-Sqrt[2]");
  }

  #[test]
  fn sqrt_2_second_root() {
    assert_eq!(interpret("Root[#^2 - 2 &, 2]").unwrap(), "Sqrt[2]");
  }

  #[test]
  fn linear() {
    assert_eq!(interpret("Root[# &, 1]").unwrap(), "0");
  }

  #[test]
  fn quadratic_integer_roots() {
    assert_eq!(interpret("Root[#^2 - 3*# + 2 &, 1]").unwrap(), "1");
    assert_eq!(interpret("Root[#^2 - 3*# + 2 &, 2]").unwrap(), "2");
  }

  #[test]
  fn complex_roots() {
    assert_eq!(interpret("Root[#^2 + 1 &, 1]").unwrap(), "-I");
    assert_eq!(interpret("Root[#^2 + 1 &, 2]").unwrap(), "I");
  }

  #[test]
  fn fourth_roots_of_unity_minus_one() {
    // x^4 - 1: roots are -1, 1, -I, I
    assert_eq!(interpret("Root[#^4 - 1 &, 1]").unwrap(), "-1");
    assert_eq!(interpret("Root[#^4 - 1 &, 2]").unwrap(), "1");
    assert_eq!(interpret("Root[#^4 - 1 &, 3]").unwrap(), "-I");
    assert_eq!(interpret("Root[#^4 - 1 &, 4]").unwrap(), "I");
  }

  #[test]
  fn cubic_with_real_root() {
    // x^3 - 1: real root is 1
    assert_eq!(interpret("Root[#^3 - 1 &, 1]").unwrap(), "1");
  }

  #[test]
  fn numerical_value() {
    let result = interpret("N[Root[#^2 - 2 &, 1]]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - (-1.414213562373095)).abs() < 1e-10,
      "Expected -1.414..., got {}",
      val
    );
  }

  #[test]
  fn out_of_range_error() {
    assert!(interpret("Root[#^2 - 1 &, 3]").is_err());
  }
}

mod polynomial_mod {
  use super::*;

  #[test]
  fn basic_cubic() {
    assert_eq!(
      interpret("PolynomialMod[x^3 + 2x + 1, 3]").unwrap(),
      "1 + 2*x + x^3"
    );
  }

  #[test]
  fn all_coefficients_zero() {
    assert_eq!(interpret("PolynomialMod[6x^2 + 9x + 12, 3]").unwrap(), "0");
  }

  #[test]
  fn mod_one() {
    assert_eq!(interpret("PolynomialMod[x + y + z, 1]").unwrap(), "0");
  }

  #[test]
  fn constant_polynomial() {
    assert_eq!(interpret("PolynomialMod[7, 3]").unwrap(), "1");
  }

  #[test]
  fn zero_polynomial() {
    assert_eq!(interpret("PolynomialMod[0, 5]").unwrap(), "0");
  }

  #[test]
  fn with_large_coefficients() {
    assert_eq!(
      interpret("PolynomialMod[5x^2 + 3x + 7, 4]").unwrap(),
      "3 + 3*x + x^2"
    );
  }

  #[test]
  fn multivariate() {
    assert_eq!(interpret("PolynomialMod[3x + 4y + 5, 3]").unwrap(), "2 + y");
  }

  #[test]
  fn symbolic_modulus_unevaluated() {
    assert_eq!(interpret("PolynomialMod[x^2, m]").unwrap(), "x^2");
  }
}

mod interpolating_polynomial {
  use super::*;

  #[test]
  fn quadratic_explicit_points() {
    // Through (1,1),(2,4),(3,9) → x^2
    assert_eq!(
      interpret("Expand[InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x]]")
        .unwrap(),
      "x^2"
    );
  }

  #[test]
  fn quadratic_implicit_points() {
    // {1,4,9} at x=1,2,3 → x^2
    assert_eq!(
      interpret("Expand[InterpolatingPolynomial[{1,4,9}, x]]").unwrap(),
      "x^2"
    );
  }

  #[test]
  fn linear_two_points() {
    // Through (0,0),(1,1) → x
    assert_eq!(
      interpret("InterpolatingPolynomial[{{0,0},{1,1}}, x]").unwrap(),
      "x"
    );
  }

  #[test]
  fn constant_one_point() {
    // Single point → constant
    assert_eq!(
      interpret("InterpolatingPolynomial[{{5,3}}, x]").unwrap(),
      "3"
    );
  }

  #[test]
  fn cubic_values() {
    // {0,1,8,27} at x=1,2,3,4 should give (x-1)^3
    let result =
      interpret("Expand[InterpolatingPolynomial[{0, 1, 8, 27}, x]]").unwrap();
    assert_eq!(result, "-1 + 3*x - 3*x^2 + x^3");
  }

  #[test]
  fn newton_form_structure() {
    // InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x]
    // Newton form: 1 + (x-1)*(1 + 3*(x-2)) = 1 + (x-1)*(1 + 3x - 6) = 1 + (-1+x)*(3*(-1+x) - 2)
    let _result =
      interpret("InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x]").unwrap();
    // Just verify it evaluates to correct values
    let at1 =
      interpret("InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x] /. x -> 1")
        .unwrap();
    let at2 =
      interpret("InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x] /. x -> 2")
        .unwrap();
    let at3 =
      interpret("InterpolatingPolynomial[{{1,1},{2,4},{3,9}}, x] /. x -> 3")
        .unwrap();
    assert_eq!(at1, "1");
    assert_eq!(at2, "4");
    assert_eq!(at3, "9");
  }

  #[test]
  fn linear_three_collinear() {
    // Through (0,0),(1,2),(2,4) → 2x
    assert_eq!(
      interpret("Expand[InterpolatingPolynomial[{{0,0},{1,2},{2,4}}, x]]")
        .unwrap(),
      "2*x"
    );
  }

  #[test]
  fn non_list_unevaluated() {
    assert_eq!(
      interpret("InterpolatingPolynomial[5, x]").unwrap(),
      "InterpolatingPolynomial[5, x]"
    );
  }
}

mod minimal_polynomial {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("MinimalPolynomial[3, x]").unwrap(), "-3 + x");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("MinimalPolynomial[1/3, x]").unwrap(), "-1 + 3*x");
  }

  #[test]
  fn sqrt_2() {
    assert_eq!(
      interpret("MinimalPolynomial[Sqrt[2], x]").unwrap(),
      "-2 + x^2"
    );
  }

  #[test]
  fn sqrt_3() {
    assert_eq!(
      interpret("MinimalPolynomial[Sqrt[3], x]").unwrap(),
      "-3 + x^2"
    );
  }

  #[test]
  fn cube_root_2() {
    assert_eq!(
      interpret("MinimalPolynomial[2^(1/3), x]").unwrap(),
      "-2 + x^3"
    );
  }

  #[test]
  fn golden_ratio() {
    assert_eq!(
      interpret("MinimalPolynomial[GoldenRatio, x]").unwrap(),
      "-1 - x + x^2"
    );
  }

  #[test]
  fn imaginary_unit() {
    assert_eq!(interpret("MinimalPolynomial[I, x]").unwrap(), "1 + x^2");
  }

  #[test]
  fn sum_of_square_roots() {
    assert_eq!(
      interpret("MinimalPolynomial[Sqrt[2] + Sqrt[3], x]").unwrap(),
      "1 - 10*x^2 + x^4"
    );
  }

  #[test]
  fn scaled_sqrt() {
    assert_eq!(
      interpret("MinimalPolynomial[2*Sqrt[3], x]").unwrap(),
      "-12 + x^2"
    );
  }

  #[test]
  fn one_plus_sqrt_2() {
    assert_eq!(
      interpret("MinimalPolynomial[1 + Sqrt[2], x]").unwrap(),
      "-1 - 2*x + x^2"
    );
  }

  #[test]
  fn wrong_arg_count() {
    // Single argument returns unevaluated or error
    assert!(
      interpret("MinimalPolynomial[Sqrt[2]]").is_err()
        || interpret("MinimalPolynomial[Sqrt[2]]")
          .unwrap()
          .contains("MinimalPolynomial")
    );
  }

  #[test]
  fn negative_sqrt() {
    assert_eq!(
      interpret("MinimalPolynomial[-Sqrt[2], x]").unwrap(),
      "-2 + x^2"
    );
  }
}

mod find_instance {
  use super::*;

  #[test]
  fn simple_equation() {
    assert_eq!(
      interpret("FindInstance[x^2 == 4, x]").unwrap(),
      "{{x -> 2}}"
    );
  }

  #[test]
  fn multiple_solutions() {
    assert_eq!(
      interpret("FindInstance[x^2 == 4, x, 3]").unwrap(),
      "{{x -> -2}, {x -> 2}}"
    );
  }

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("FindInstance[x^2 - 5 x + 6 == 0, x]").unwrap(),
      "{{x -> 3}}"
    );
  }

  #[test]
  fn two_variable_equation() {
    assert_eq!(
      interpret("FindInstance[x^2 + y^2 == 1, {x, y}]").unwrap(),
      "{{x -> -1, y -> 0}}"
    );
  }

  #[test]
  fn integer_domain_inequality() {
    assert_eq!(
      interpret("FindInstance[x > 3 && x < 5, x, Integers]").unwrap(),
      "{{x -> 4}}"
    );
  }

  #[test]
  fn no_solution() {
    assert_eq!(
      interpret("FindInstance[x^2 == -1, x, Reals]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn single_var_in_list() {
    assert_eq!(
      interpret("FindInstance[x^2 == 9, {x}]").unwrap(),
      "{{x -> 3}}"
    );
  }
}

mod find_sequence_function {
  use super::*;

  #[test]
  fn linear_sequence() {
    assert_eq!(
      interpret("FindSequenceFunction[{1, 2, 3, 4, 5}, n]").unwrap(),
      "n"
    );
  }

  #[test]
  fn squares() {
    assert_eq!(
      interpret("FindSequenceFunction[{1, 4, 9, 16, 25}, n]").unwrap(),
      "n^2"
    );
  }

  #[test]
  fn cubes() {
    assert_eq!(
      interpret("FindSequenceFunction[{1, 8, 27, 64, 125}, n]").unwrap(),
      "n^3"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(
      interpret("FindSequenceFunction[{5, 5, 5, 5}, n]").unwrap(),
      "5"
    );
  }

  #[test]
  fn powers_of_2() {
    assert_eq!(
      interpret("FindSequenceFunction[{2, 4, 8, 16, 32}, n]").unwrap(),
      "2^n"
    );
  }

  #[test]
  fn powers_of_3() {
    assert_eq!(
      interpret("FindSequenceFunction[{3, 9, 27, 81, 243}, n]").unwrap(),
      "3^n"
    );
  }

  #[test]
  fn factorial() {
    assert_eq!(
      interpret("FindSequenceFunction[{1, 2, 6, 24, 120}, n]").unwrap(),
      "n!"
    );
  }

  #[test]
  fn triangular_numbers() {
    // (n*(1+n))/2 expanded form
    assert_eq!(
      interpret("FindSequenceFunction[{1, 3, 6, 10, 15}, n]").unwrap(),
      "n/2 + n^2/2"
    );
  }

  #[test]
  fn formula_is_correct() {
    // Verify the found formula gives correct values by substitution
    assert_eq!(
      interpret("FindSequenceFunction[{1, 4, 9, 16, 25}, n] /. n -> 6")
        .unwrap(),
      "36"
    );
  }
}

mod horner_form {
  use super::*;

  #[test]
  fn basic_univariate() {
    assert_eq!(
      interpret("HornerForm[11 x^3 - 4 x^2 + 7 x + 2]").unwrap(),
      "2 + x*(7 + x*(-4 + 11*x))"
    );
  }

  #[test]
  fn explicit_variable() {
    assert_eq!(
      interpret("HornerForm[a + b x + c x^2, x]").unwrap(),
      "a + x*(b + c*x)"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("HornerForm[5]").unwrap(), "5");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("HornerForm[0]").unwrap(), "0");
  }

  #[test]
  fn single_variable() {
    assert_eq!(interpret("HornerForm[x]").unwrap(), "x");
  }

  #[test]
  fn monomial() {
    assert_eq!(interpret("HornerForm[x^3]").unwrap(), "x^3");
  }

  #[test]
  fn linear_polynomial() {
    assert_eq!(interpret("HornerForm[x + 2]").unwrap(), "2 + x");
  }

  #[test]
  fn non_polynomial() {
    assert_eq!(interpret("HornerForm[Sin[x]]").unwrap(), "Sin[x]");
  }

  #[test]
  fn degree_four() {
    assert_eq!(
      interpret("HornerForm[x^4 + 2 x^3 - x + 5]").unwrap(),
      "5 + x*(-1 + x^2*(2 + x))"
    );
  }

  #[test]
  fn univariate_in_a() {
    assert_eq!(
      interpret("HornerForm[a^2 + 3 a + 1]").unwrap(),
      "1 + a*(3 + a)"
    );
  }

  #[test]
  fn multivariate_explicit_x() {
    assert_eq!(
      interpret("HornerForm[x^2 + 2 x y + y^2, x]").unwrap(),
      "y^2 + x*(x + 2*y)"
    );
  }

  #[test]
  fn multivariate_explicit_y() {
    assert_eq!(
      interpret("HornerForm[x^2 + 2 x y + y^2, y]").unwrap(),
      "x^2 + y*(2*x + y)"
    );
  }

  #[test]
  fn unrelated_variable() {
    assert_eq!(
      interpret("HornerForm[3 x^2 + 2 x + 1, y]").unwrap(),
      "1 + 2*x + 3*x^2"
    );
  }

  #[test]
  fn rational_function() {
    let result =
      interpret("HornerForm[(11 x^3 - 4 x^2 + 7 x + 2)/(x^2 - 3 x + 1)]")
        .unwrap();
    assert_eq!(result, "(2 + x*(7 + x*(-4 + 11*x)))/(1 + (-3 + x)*x)");
  }

  #[test]
  fn all_numeric_coefficients() {
    assert_eq!(
      interpret("HornerForm[1 + 2 x + 3 x^2 + 4 x^3]").unwrap(),
      "1 + x*(2 + x*(3 + 4*x))"
    );
  }

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("HornerForm[x^2 + 5 x + 6]").unwrap(),
      "6 + x*(5 + x)"
    );
  }
}

mod function_expand {
  use super::*;

  #[test]
  fn pochhammer() {
    assert_eq!(
      interpret("FunctionExpand[Pochhammer[a, n]]").unwrap(),
      "Gamma[a + n]/Gamma[a]"
    );
  }

  #[test]
  fn beta() {
    assert_eq!(
      interpret("FunctionExpand[Beta[a, b]]").unwrap(),
      "(Gamma[a]*Gamma[b])/Gamma[a + b]"
    );
  }

  #[test]
  fn binomial_n_2() {
    assert_eq!(
      interpret("FunctionExpand[Binomial[n, 2]]").unwrap(),
      "((-1 + n)*n)/2"
    );
  }

  #[test]
  fn haversine() {
    assert_eq!(
      interpret("FunctionExpand[Haversine[x]]").unwrap(),
      "(1 - Cos[x])/2"
    );
  }

  #[test]
  fn inverse_haversine() {
    assert_eq!(
      interpret("FunctionExpand[InverseHaversine[x]]").unwrap(),
      "2*ArcSin[Sqrt[x]]"
    );
  }

  #[test]
  fn sinc() {
    assert_eq!(interpret("FunctionExpand[Sinc[x]]").unwrap(), "Sin[x]/x");
  }

  #[test]
  fn chebyshev_t() {
    assert_eq!(
      interpret("FunctionExpand[ChebyshevT[n, x]]").unwrap(),
      "Cos[n*ArcCos[x]]"
    );
  }

  #[test]
  fn chebyshev_u() {
    assert_eq!(
      interpret("FunctionExpand[ChebyshevU[n, x]]").unwrap(),
      "Sin[(1 + n)*ArcCos[x]]/(Sqrt[1 - x]*Sqrt[1 + x])"
    );
  }

  #[test]
  fn fibonacci() {
    assert_eq!(
      interpret("FunctionExpand[Fibonacci[n]]").unwrap(),
      "(((1 + Sqrt[5])/2)^n - (2/(1 + Sqrt[5]))^n*Cos[n*Pi])/Sqrt[5]"
    );
  }

  #[test]
  fn lucas_l() {
    assert_eq!(
      interpret("FunctionExpand[LucasL[n]]").unwrap(),
      "((1 + Sqrt[5])/2)^n + (2/(1 + Sqrt[5]))^n*Cos[n*Pi]"
    );
  }

  #[test]
  fn gamma_half() {
    assert_eq!(interpret("FunctionExpand[Gamma[1/2]]").unwrap(), "Sqrt[Pi]");
  }

  #[test]
  fn passthrough() {
    // Functions without expansion rules pass through
    assert_eq!(interpret("FunctionExpand[Sin[x]]").unwrap(), "Sin[x]");
  }
}

mod to_radicals {
  use super::*;

  #[test]
  fn quadratic() {
    assert_eq!(
      interpret("ToRadicals[Root[#^2 - 3 &, 1]]").unwrap(),
      "-Sqrt[3]"
    );
    assert_eq!(
      interpret("ToRadicals[Root[#^2 - 3 &, 2]]").unwrap(),
      "Sqrt[3]"
    );
  }

  #[test]
  fn quadratic_with_linear() {
    assert_eq!(
      interpret("ToRadicals[Root[#^2 + 3# + 1 &, 1]]").unwrap(),
      "(-3 - Sqrt[5])/2"
    );
    assert_eq!(
      interpret("ToRadicals[Root[#^2 + 3# + 1 &, 2]]").unwrap(),
      "(-3 + Sqrt[5])/2"
    );
  }

  #[test]
  fn cubic_pure() {
    assert_eq!(
      interpret("ToRadicals[Root[#^3 - 2 &, 1]]").unwrap(),
      "2^(1/3)"
    );
  }

  #[test]
  fn quartic_pure() {
    assert_eq!(
      interpret("ToRadicals[Root[#^4 - 2 &, 1]]").unwrap(),
      "-2^(1/4)"
    );
    assert_eq!(
      interpret("ToRadicals[Root[#^4 - 2 &, 2]]").unwrap(),
      "2^(1/4)"
    );
  }

  #[test]
  fn quintic_pure() {
    assert_eq!(
      interpret("ToRadicals[Root[#^5 - 2 &, 1]]").unwrap(),
      "2^(1/5)"
    );
  }

  #[test]
  fn sixth_root() {
    assert_eq!(
      interpret("ToRadicals[Root[#^6 - 2 &, 1]]").unwrap(),
      "-2^(1/6)"
    );
  }
}
