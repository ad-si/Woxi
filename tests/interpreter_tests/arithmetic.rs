use super::*;

mod arithmetic {
  use super::*;

  mod integer {
    use super::*;

    #[test]
    fn addition() {
      assert_eq!(interpret("1 + 2").unwrap(), "3");
      assert_eq!(interpret("1 + 2 + 3").unwrap(), "6");
      assert_eq!(interpret("(1 + 2) + 3").unwrap(), "6");
      assert_eq!(interpret("1 + (2 + 3)").unwrap(), "6");
      assert_eq!(interpret("(1 + 2 + 3)").unwrap(), "6");
    }

    #[test]
    fn subtraction() {
      assert_eq!(interpret("3 - 1").unwrap(), "2");
      assert_eq!(interpret("7 - 3 - 1").unwrap(), "3");
    }

    #[test]
    fn multiple_operations() {
      assert_eq!(interpret("1 + 2 - 3 + 4").unwrap(), "4");
    }

    #[test]
    fn negative_numbers() {
      assert_eq!(interpret("-1 + 3").unwrap(), "2");
    }

    #[test]
    fn multiplication() {
      assert_eq!(interpret("3 * 4").unwrap(), "12");
    }

    #[test]
    fn complex_multiplication() {
      assert_eq!(interpret("2 * 3 + 4 * 5").unwrap(), "26");
    }

    #[test]
    fn division() {
      assert_eq!(interpret("10 / 2").unwrap(), "5");
    }

    #[test]
    fn division_repeating_decimal() {
      // Wolfram keeps this as a fraction 10/3
      assert_eq!(interpret("10 / 3").unwrap(), "10/3");
    }

    #[test]
    fn complex_division() {
      assert_eq!(interpret("10 / 2 + 3 / 3").unwrap(), "6");
    }

    #[test]
    fn integer_plus_rational_stays_exact() {
      // Must not convert to float (was returning 4.6)
      assert_eq!(interpret("1 + 18/5").unwrap(), "23/5");
      assert_eq!(interpret("2 + 1/3").unwrap(), "7/3");
      assert_eq!(interpret("1/3 + 1/6").unwrap(), "1/2");
      assert_eq!(interpret("1/3 + 2/3").unwrap(), "1");
      assert_eq!(interpret("5 + 1/2 + 1/3").unwrap(), "35/6");
    }
  }

  mod float {
    use super::*;

    #[test]
    fn addition() {
      assert_eq!(interpret("1.5 + 2.7").unwrap(), "4.2");
    }

    #[test]
    fn subtraction() {
      assert_eq!(interpret("3.5 - 1.2").unwrap(), "2.3");
    }

    #[test]
    fn multiple_operations() {
      assert_eq!(interpret("1.1 + 2.2 - 3.3 + 4.4").unwrap(), "4.4");
    }

    #[test]
    fn multiplication() {
      assert_eq!(interpret("1.5 * 2.5").unwrap(), "3.75");
    }

    #[test]
    fn complex_multiplication() {
      assert_eq!(interpret("1.5 * 2.0 + 3.0 * 1.5").unwrap(), "7.5");
    }

    #[test]
    fn division() {
      assert_eq!(interpret("9.6 / 3").unwrap(), "3.1999999999999997");
    }

    #[test]
    fn complex_division() {
      assert_eq!(interpret("9.6 / 3 + 3.0 / 3").unwrap(), "4.199999999999999");
    }

    #[test]
    fn addition_ieee754_precision() {
      // Must preserve IEEE 754 representation, not round
      assert_eq!(interpret("0.1 + 0.2").unwrap(), "0.30000000000000004");
    }

    #[test]
    fn division_repeating() {
      assert_eq!(interpret("1.0 / 3.0").unwrap(), "0.3333333333333333");
    }

    #[test]
    fn sqrt_real() {
      assert_eq!(interpret("Sqrt[2.0]").unwrap(), "1.4142135623730951");
    }

    #[test]
    fn whole_number_real() {
      // Whole-number reals keep trailing dot
      assert_eq!(interpret("1.0").unwrap(), "1.");
      assert_eq!(interpret("100.0").unwrap(), "100.");
    }

    #[test]
    fn real_type_preserved_in_addition() {
      assert_eq!(interpret("2.0 + 3.0").unwrap(), "5.");
      assert_eq!(interpret("Head[2.0 + 3.0]").unwrap(), "Real");
    }

    #[test]
    fn real_type_preserved_in_subtraction() {
      assert_eq!(interpret("6.0 - 3.0").unwrap(), "3.");
      assert_eq!(interpret("Head[6.0 - 3.0]").unwrap(), "Real");
    }

    #[test]
    fn real_type_preserved_in_multiplication() {
      assert_eq!(interpret("2.0 * 3.0").unwrap(), "6.");
      assert_eq!(interpret("Head[2.0 * 3.0]").unwrap(), "Real");
    }

    #[test]
    fn real_type_preserved_in_negation() {
      assert_eq!(interpret("-3.0").unwrap(), "-3.");
      assert_eq!(interpret("Head[-3.0]").unwrap(), "Real");
    }
  }

  mod head_of_comparisons {
    use super::*;

    // Regression: Head of comparison expressions used to return "Comparison"
    // instead of the actual operator name (Equal, Less, etc.).
    #[test]
    fn head_equal() {
      assert_eq!(interpret("Head[x == y]").unwrap(), "Equal");
    }

    #[test]
    fn head_less() {
      assert_eq!(interpret("Head[x < y]").unwrap(), "Less");
    }

    #[test]
    fn head_greater_equal() {
      assert_eq!(interpret("Head[x >= y]").unwrap(), "GreaterEqual");
    }

    #[test]
    fn head_unequal() {
      assert_eq!(interpret("Head[x != y]").unwrap(), "Unequal");
    }

    #[test]
    fn head_uniform_chain() {
      // Uniform chain: all operators the same → head is that operator
      assert_eq!(interpret("Head[1 < x < 10]").unwrap(), "Less");
    }

    #[test]
    fn head_mixed_chain() {
      // Mixed chain: different operators → head is Inequality
      assert_eq!(interpret("Head[1 < x <= 10]").unwrap(), "Inequality");
    }

    // FullForm of a uniform comparison chain collapses to the operator head,
    // matching Wolfram (a > b > c prints as Greater[a, b, c], not Inequality).
    // wolframscript's REPL keeps the `FullForm[…]` wrapper around comparison
    // expressions; the bare head form is reachable via `ToString[…]`.
    #[test]
    fn full_form_uniform_chain_greater() {
      assert_eq!(
        interpret("a > b > c // FullForm").unwrap(),
        "FullForm[a > b > c]"
      );
      assert_eq!(
        interpret("ToString[a > b > c // FullForm]").unwrap(),
        "Greater[a, b, c]"
      );
    }

    #[test]
    fn full_form_mixed_chain_stays_inequality() {
      assert_eq!(
        interpret("a < b <= c // FullForm").unwrap(),
        "FullForm[Inequality[a, Less, b, LessEqual, c]]"
      );
      assert_eq!(
        interpret("ToString[a < b <= c // FullForm]").unwrap(),
        "Inequality[a, Less, b, LessEqual, c]"
      );
    }

    #[test]
    fn inequality_display_output_form() {
      // wolframscript keeps the Inequality head in script-mode output
      // (this previously asserted the chained -2 < x < 2 form)
      assert_eq!(
        interpret("Inequality[-2, Less, x, Less, 2]").unwrap(),
        "Inequality[-2, Less, x, Less, 2]"
      );
    }

    #[test]
    fn inequality_less_equal_display() {
      assert_eq!(
        interpret("Inequality[0, LessEqual, x, LessEqual, 1]").unwrap(),
        "Inequality[0, LessEqual, x, LessEqual, 1]"
      );
    }
  }

  mod times_simplification {
    use super::*;

    #[test]
    fn zero_times_symbol() {
      assert_eq!(interpret("0*x").unwrap(), "0");
      assert_eq!(interpret("x*0").unwrap(), "0");
    }

    #[test]
    fn one_times_symbol() {
      assert_eq!(interpret("1*x").unwrap(), "x");
      assert_eq!(interpret("x*1").unwrap(), "x");
    }

    #[test]
    fn times_function_zero() {
      assert_eq!(interpret("Times[0, x]").unwrap(), "0");
      assert_eq!(interpret("Times[x, 0]").unwrap(), "0");
      assert_eq!(interpret("Times[0, x, y]").unwrap(), "0");
    }

    #[test]
    fn times_function_one() {
      assert_eq!(interpret("Times[1, x]").unwrap(), "x");
      assert_eq!(interpret("Times[x, 1]").unwrap(), "x");
    }

    #[test]
    fn times_coefficient() {
      assert_eq!(interpret("Times[2, x]").unwrap(), "2*x");
      assert_eq!(interpret("Times[-1, x]").unwrap(), "-x");
    }

    #[test]
    fn zero_times_list() {
      assert_eq!(interpret("0*{a, b, c}").unwrap(), "{0, 0, 0}");
    }

    #[test]
    fn one_times_list() {
      assert_eq!(interpret("1*{a, b, c}").unwrap(), "{a, b, c}");
    }
  }

  mod times_canonical_ordering {
    use super::*;

    #[test]
    fn symbols_sorted_alphabetically() {
      assert_eq!(interpret("z*a*m*b").unwrap(), "a*b*m*z");
    }

    #[test]
    fn number_first_then_symbols() {
      assert_eq!(interpret("x*3").unwrap(), "3*x");
      assert_eq!(interpret("b*a*5").unwrap(), "5*a*b");
    }

    #[test]
    fn transcendental_after_polynomial() {
      assert_eq!(interpret("Sin[x]*a*Cos[x]").unwrap(), "a*Cos[x]*Sin[x]");
      assert_eq!(interpret("Log[x]*x*Cos[y]").unwrap(), "x*Cos[y]*Log[x]");
    }

    #[test]
    fn times_function_sorted() {
      assert_eq!(interpret("Times[z, a, m, b]").unwrap(), "a*b*m*z");
    }

    #[test]
    fn rational_preserved_in_product() {
      assert_eq!(interpret("Rational[1, 3]*Sin[x]").unwrap(), "Sin[x]/3");
    }

    #[test]
    fn derivative_sorted() {
      assert_eq!(interpret("D[Sin[x]^2, x]").unwrap(), "2*Cos[x]*Sin[x]");
    }

    // When two reciprocal factors share the same sort key (e.g. y vs (x-y)),
    // the additive base sorts FIRST when it contains the negation of the
    // bare identifier — matching Wolfram. Regression for `Apart` output.
    #[test]
    fn reciprocal_additive_with_negated_ident_sorts_first() {
      assert_eq!(interpret("y^-1 * (x - y)^-1").unwrap(), "1/((x - y)*y)");
      assert_eq!(interpret("(x - y)^-1 * y^-1").unwrap(), "1/((x - y)*y)");
    }

    #[test]
    fn reciprocal_additive_without_negated_ident_keeps_default_order() {
      // `x + y` does not contain `-y`, so the shorter `y` factor stays first.
      assert_eq!(interpret("y^-1 * (x + y)^-1").unwrap(), "1/(y*(x + y))");
    }

    #[test]
    fn apart_partial_fraction_canonical_factor_order() {
      assert_eq!(
        interpret("Apart[1 / (x^2 - y^2), x]").unwrap(),
        "1/(2*(x - y)*y) - 1/(2*y*(x + y))"
      );
    }

    // Bare-identifier vs additive: Wolfram's `Order[Plus[2,a,b], z]` is 1
    // because Plus's highest-key arg `b` is alphabetically smaller than
    // `z`, so the additive sorts first in `Times`. This applies only to
    // bare identifiers — Power[x, n] vs additive keeps its existing
    // ordering (`x^x*(1 + Log[x])`, not `(1 + Log[x])*x^x`).
    #[test]
    fn additive_before_bare_identifier_when_key_is_smaller() {
      assert_eq!(interpret("(2 + a + b)*z").unwrap(), "(2 + a + b)*z");
      assert_eq!(interpret("z*(2 + a + b)").unwrap(), "(2 + a + b)*z");
    }

    #[test]
    fn bare_identifier_keeps_first_when_additive_key_is_larger() {
      assert_eq!(interpret("a*(b + c)").unwrap(), "a*(b + c)");
    }

    #[test]
    fn bare_identifier_keeps_first_when_keys_tie() {
      // Plus[x, y].sort_key = "y" ties with `y`; bare identifier wins
      // (matches `Order[x+y, y] = -1`).
      assert_eq!(interpret("y*(x + y)").unwrap(), "y*(x + y)");
    }

    #[test]
    fn power_factor_keeps_existing_order_against_additive() {
      // D[x^x, x] → x^x*(1 + Log[x]). Power[x, x] is NOT a bare
      // identifier, so the additive doesn't sort first even though
      // sort_key("Log[x]") < sort_key("x") would suggest it should.
      assert_eq!(interpret("D[x^x, x]").unwrap(), "x^x*(1 + Log[x])");
    }

    // BigFloat factors sort BEFORE the imaginary unit (matches Wolfram:
    // `N[Pi, 30]*I` displays as `<BigFloat>*I`, not `I*<BigFloat>`).
    // Machine-precision Reals continue to use the `0. + r*I` Re/Im
    // split that wolframscript prints for them.
    #[test]
    fn bigfloat_sorts_before_imaginary_unit() {
      assert!(
        interpret("N[Pi, 30] * I").unwrap().ends_with("*I"),
        "N[Pi, 30] * I should display BigFloat first then *I"
      );
      assert!(
        !interpret("N[Pi, 30] * I").unwrap().starts_with("I*"),
        "N[Pi, 30] * I must not display I*BigFloat"
      );
    }

    #[test]
    fn machine_real_keeps_re_im_split() {
      assert_eq!(interpret("3.5 I").unwrap(), "0. + 3.5*I");
    }
  }

  mod combine_like_bases {
    use super::*;

    #[test]
    fn function_call_squared() {
      assert_eq!(interpret("f[x]*f[x]").unwrap(), "f[x]^2");
    }

    #[test]
    fn function_call_cubed() {
      assert_eq!(interpret("g[x, y]*g[x, y]*g[x, y]").unwrap(), "g[x, y]^3");
    }

    #[test]
    fn tan_squared() {
      assert_eq!(interpret("Tan[x]*Tan[x]").unwrap(), "Tan[x]^2");
    }

    #[test]
    fn plus_expr_squared() {
      assert_eq!(interpret("(a + b)*(a + b)").unwrap(), "(a + b)^2");
    }

    #[test]
    fn unary_minus_in_times() {
      // -Sin[x]*Sin[x] should simplify to -Sin[x]^2
      assert_eq!(interpret("Times[-Sin[x], Sin[x]]").unwrap(), "-Sin[x]^2");
    }

    // Named numeric constants (Pi, E, Degree, …) raised to a
    // *concrete* integer exponent sort like polynomial variables
    // in Plus, so `Pi^2 + 3*Pi` lands as `3*Pi + Pi^2` (ascending
    // exponent, low first). Symbolic exponents (e.g. `E^x`) still
    // route through the old term-sort-key path. Regression for
    // mathics mathics3.py `E^2 + 3E` row.
    #[test]
    fn plus_pi_orders_by_concrete_exponent() {
      assert_eq!(interpret("Pi^2 + 3 Pi").unwrap(), "3*Pi + Pi^2");
    }

    #[test]
    fn plus_e_orders_by_concrete_exponent() {
      assert_eq!(interpret("E^2 + 3 E").unwrap(), "3*E + E^2");
    }
  }

  // `n^(p/q)` for a perfect-power base reduces to its primitive base: the
  // primes of n collapse to a smaller base sharing one fractional exponent.
  mod radical_perfect_power_base {
    use super::*;

    #[test]
    fn cube_root_of_square() {
      // 100 = 10^2, so 100^(1/3) = 10^(2/3).
      assert_eq!(interpret("100^(1/3)").unwrap(), "10^(2/3)");
      assert_eq!(interpret("36^(1/3)").unwrap(), "6^(2/3)");
    }

    #[test]
    fn fourth_root_collapses_to_sqrt() {
      // 100^(1/4) = (10^2)^(1/4) = 10^(1/2).
      assert_eq!(interpret("100^(1/4)").unwrap(), "Sqrt[10]");
    }

    #[test]
    fn fourth_root_of_cube_power() {
      // 1000 = 10^3, so 1000^(1/4) = 10^(3/4).
      assert_eq!(interpret("1000^(1/4)").unwrap(), "10^(3/4)");
      assert_eq!(interpret("216^(1/4)").unwrap(), "6^(3/4)");
    }

    // Squarefree bases have no perfect-power reduction and stay as written
    // (the simplifier must not loop here).
    #[test]
    fn squarefree_base_unchanged() {
      assert_eq!(interpret("6^(1/3)").unwrap(), "6^(1/3)");
      assert_eq!(interpret("6^(2/3)").unwrap(), "6^(2/3)");
      assert_eq!(interpret("10^(2/3)").unwrap(), "10^(2/3)");
    }

    // Extraction still composes with the reduction.
    #[test]
    fn extraction_then_combine() {
      // 1000000 = 10^6, 1000000^(1/4) = 10^(3/2) = 10*Sqrt[10].
      assert_eq!(interpret("1000000^(1/4)").unwrap(), "10*Sqrt[10]");
      // 72 = 8*9, 72^(1/3) = 2*9^(1/3) = 2*3^(2/3).
      assert_eq!(interpret("72^(1/3)").unwrap(), "2*3^(2/3)");
    }

    // The motivating case: the p-norm.
    #[test]
    fn norm_p3() {
      assert_eq!(interpret("Norm[{1, 2, 3, 4}, 3]").unwrap(), "10^(2/3)");
    }
  }
}

mod real_number_formatting {
  use super::*;

  #[test]
  fn power_with_decimal_exponent() {
    // In Wolfram, 0.5 is Real so result is Real (2.)
    assert_eq!(interpret("Power[4, 0.5]").unwrap(), "2.");
  }

  #[test]
  fn accumulate_preserves_real() {
    assert_eq!(interpret("Accumulate[{1.5, 2.5}]").unwrap(), "{1.5, 4.}");
  }

  #[test]
  fn division_preserves_real_type() {
    assert_eq!(interpret("10.0 / 2").unwrap(), "5.");
  }

  #[test]
  fn integer_division_stays_integer() {
    assert_eq!(interpret("10 / 2").unwrap(), "5");
  }

  #[test]
  fn trailing_dot_float_standalone() {
    assert_eq!(interpret("1.").unwrap(), "1.");
  }

  #[test]
  fn trailing_dot_float_in_function_call() {
    assert_eq!(interpret("Head[1.]").unwrap(), "Real");
  }

  #[test]
  fn trailing_dot_float_in_list() {
    assert_eq!(interpret("{1., 2., 3.}").unwrap(), "{1., 2., 3.}");
  }

  #[test]
  fn trailing_dot_float_arithmetic() {
    assert_eq!(interpret("1. + 2.").unwrap(), "3.");
  }

  #[test]
  fn trailing_dot_float_equals_explicit() {
    assert_eq!(interpret("1. == 1.0").unwrap(), "True");
  }
}

mod plus_numeric_contagion {
  use super::*;

  // Regression (mathics test_comparison.py:570): a machine Real among
  // the Plus terms numerifies named numeric constants (Pi, E, ...) and
  // any numeric constants embedded inside otherwise-symbolic summands
  // (e.g. `Pi*I`), matching wolframscript. Bare variables stay symbolic.
  #[test]
  fn real_plus_pi_numerifies() {
    assert_eq!(interpret("2. + Pi").unwrap(), "5.141592653589793");
  }

  #[test]
  fn real_plus_e_numerifies() {
    assert_eq!(interpret("2. + E").unwrap(), "4.718281828459045");
  }

  #[test]
  fn real_plus_sqrt2_numerifies() {
    assert_eq!(interpret("2. + Sqrt[2]").unwrap(), "3.414213562373095");
  }

  #[test]
  fn real_plus_pi_times_i_numerifies_pi_only() {
    // Pi is numerified but I stays symbolic, matching wolframscript.
    assert_eq!(interpret("2.+ Pi I").unwrap(), "2. + 3.141592653589793*I");
  }

  #[test]
  fn real_plus_pi_plus_symbol() {
    // Pi numerified into the Real sum; bare symbol stays.
    assert_eq!(interpret("2. + Pi + a").unwrap(), "5.141592653589793 + a");
  }

  #[test]
  fn real_plus_symbol_stays_symbolic() {
    // No numeric constants in the symbolic part — nothing to numerify.
    assert_eq!(interpret("2. + a").unwrap(), "2. + a");
  }

  // Regression: numeric contagion must not over-eagerly numerify
  // bare integer exponents or unrelated integer summands inside the
  // symbolic part.
  #[test]
  fn real_plus_polynomial_keeps_exact_exponent() {
    assert_eq!(interpret("Plus[2., (x-3)^2]").unwrap(), "2. + (-3 + x)^2");
  }
}

mod big_integer {
  use super::*;

  #[test]
  fn power_exceeding_i128() {
    assert_eq!(
      interpret("2^127").unwrap(),
      "170141183460469231731687303715884105728"
    );
  }

  #[test]
  fn power_200() {
    assert_eq!(
      interpret("2^200").unwrap(),
      "1606938044258990275541962092341162602522202993782792835301376"
    );
  }

  #[test]
  fn big_power_minus_one_fits_i128() {
    // 2^127 - 1 = i128::MAX, should convert back to Integer
    assert_eq!(
      interpret("2^127 - 1").unwrap(),
      "170141183460469231731687303715884105727"
    );
  }

  #[test]
  fn big_power_addition() {
    assert_eq!(
      interpret("2^127 + 1").unwrap(),
      "170141183460469231731687303715884105729"
    );
  }

  #[test]
  fn big_power_multiplication() {
    assert_eq!(
      interpret("2^127 * 2").unwrap(),
      "340282366920938463463374607431768211456"
    );
  }

  #[test]
  fn large_i128_subtraction() {
    // 2^67 fits in i128 but exceeds f64 precision (> 2^53)
    assert_eq!(interpret("2^67 - 1").unwrap(), "147573952589676412927");
  }

  #[test]
  fn large_i128_addition() {
    assert_eq!(interpret("2^67 + 1").unwrap(), "147573952589676412929");
  }

  #[test]
  fn large_i128_multiplication() {
    assert_eq!(interpret("2^67 * 3").unwrap(), "442721857769029238784");
  }

  #[test]
  fn large_i128_sum_of_two() {
    assert_eq!(interpret("10^20 + 10^20").unwrap(), "200000000000000000000");
  }

  #[test]
  fn factor_integer_large() {
    assert_eq!(
      interpret("FactorInteger[2^67 - 1]").unwrap(),
      "{{193707721, 1}, {761838257287, 1}}"
    );
  }

  #[test]
  fn factor_integer_bigint_2_128_minus_1() {
    assert_eq!(
      interpret("FactorInteger[2^128 - 1]").unwrap(),
      "{{3, 1}, {5, 1}, {17, 1}, {257, 1}, {641, 1}, {65537, 1}, {274177, 1}, {6700417, 1}, {67280421310721, 1}}"
    );
  }

  #[test]
  fn factor_integer_bigint_power_of_2() {
    assert_eq!(interpret("FactorInteger[2^200]").unwrap(), "{{2, 200}}");
  }

  #[test]
  fn factor_integer_negative_bigint() {
    assert_eq!(
      interpret("FactorInteger[-(2^128 - 1)]").unwrap(),
      "{{-1, 1}, {3, 1}, {5, 1}, {17, 1}, {257, 1}, {641, 1}, {65537, 1}, {274177, 1}, {6700417, 1}, {67280421310721, 1}}"
    );
  }

  #[test]
  fn factor_integer_bigint_mersenne_prime() {
    assert_eq!(
      interpret("FactorInteger[2^127 - 1]").unwrap(),
      "{{170141183460469231731687303715884105727, 1}}"
    );
  }

  #[test]
  fn head_of_big_integer() {
    assert_eq!(interpret("Head[2^128]").unwrap(), "Integer");
  }

  #[test]
  fn integer_q_big_integer() {
    assert_eq!(interpret("IntegerQ[2^128]").unwrap(), "True");
  }

  #[test]
  fn even_q_big_integer() {
    assert_eq!(interpret("EvenQ[2^128]").unwrap(), "True");
  }

  #[test]
  fn square_free_q_true() {
    assert_eq!(interpret("SquareFreeQ[10]").unwrap(), "True");
  }

  #[test]
  fn square_free_q_false() {
    assert_eq!(interpret("SquareFreeQ[12]").unwrap(), "False");
  }

  #[test]
  fn square_free_q_one() {
    assert_eq!(interpret("SquareFreeQ[1]").unwrap(), "True");
  }

  #[test]
  fn square_free_q_zero() {
    assert_eq!(interpret("SquareFreeQ[0]").unwrap(), "False");
  }

  #[test]
  fn square_free_q_negative() {
    assert_eq!(interpret("SquareFreeQ[-12]").unwrap(), "False");
  }

  #[test]
  fn perfect_number_q() {
    // Perfect numbers: sum of proper divisors equals the number
    assert_eq!(interpret("PerfectNumberQ[6]").unwrap(), "True");
    assert_eq!(interpret("PerfectNumberQ[28]").unwrap(), "True");
    assert_eq!(interpret("PerfectNumberQ[496]").unwrap(), "True");
    assert_eq!(interpret("PerfectNumberQ[12]").unwrap(), "False");
    assert_eq!(interpret("PerfectNumberQ[1]").unwrap(), "False");
    assert_eq!(interpret("PerfectNumberQ[0]").unwrap(), "False");
    assert_eq!(
      interpret("Select[Range[500], PerfectNumberQ]").unwrap(),
      "{6, 28, 496}"
    );
  }

  // wolframscript reports any non-(positive-integer) argument as not a
  // perfect number, rather than leaving the call unevaluated.
  #[test]
  fn perfect_number_q_non_integer_is_false() {
    assert_eq!(interpret("PerfectNumberQ[6.0]").unwrap(), "False");
    assert_eq!(interpret("PerfectNumberQ[28.0]").unwrap(), "False");
    assert_eq!(interpret("PerfectNumberQ[Pi]").unwrap(), "False");
    assert_eq!(interpret("PerfectNumberQ[3/2]").unwrap(), "False");
    assert_eq!(interpret("PerfectNumberQ[x]").unwrap(), "False");
    assert_eq!(interpret("PerfectNumberQ[-6]").unwrap(), "False");
  }

  #[test]
  fn square_free_q_prime() {
    assert_eq!(interpret("SquareFreeQ[7]").unwrap(), "True");
  }

  #[test]
  fn square_free_q_product_of_distinct_primes() {
    assert_eq!(interpret("SquareFreeQ[30]").unwrap(), "True");
  }

  #[test]
  fn square_free_q_perfect_square() {
    assert_eq!(interpret("SquareFreeQ[49]").unwrap(), "False");
  }

  // A rational p/q is square-free when both numerator and denominator are
  // square-free; an explicit real number is never square-free. Verified
  // against wolframscript.
  #[test]
  fn square_free_q_rational_and_real() {
    assert_eq!(interpret("SquareFreeQ[3/2]").unwrap(), "True");
    assert_eq!(interpret("SquareFreeQ[12/5]").unwrap(), "False");
    assert_eq!(interpret("SquareFreeQ[8/9]").unwrap(), "False");
    assert_eq!(interpret("SquareFreeQ[18/25]").unwrap(), "False");
    assert_eq!(interpret("SquareFreeQ[-3/2]").unwrap(), "True");
    assert_eq!(interpret("SquareFreeQ[5.0]").unwrap(), "False");
    assert_eq!(interpret("SquareFreeQ[2.5]").unwrap(), "False");
  }

  #[test]
  fn odd_q_big_integer() {
    assert_eq!(interpret("OddQ[2^128 + 1]").unwrap(), "True");
  }

  #[test]
  fn number_q_big_integer() {
    assert_eq!(interpret("NumberQ[2^128]").unwrap(), "True");
  }

  #[test]
  fn number_q_complex() {
    assert_eq!(interpret("NumberQ[3 + 2 I]").unwrap(), "True");
    assert_eq!(interpret("NumberQ[I]").unwrap(), "True");
  }

  #[test]
  fn number_q_rational() {
    assert_eq!(interpret("NumberQ[3/4]").unwrap(), "True");
  }

  #[test]
  fn number_q_symbolic() {
    assert_eq!(interpret("NumberQ[x]").unwrap(), "False");
  }

  #[test]
  fn divisible_big_integer() {
    assert_eq!(interpret("Divisible[2^128, 4]").unwrap(), "True");
    assert_eq!(interpret("Divisible[2^128 + 1, 2]").unwrap(), "False");
  }

  #[test]
  fn composite_q_big_integer() {
    assert_eq!(interpret("CompositeQ[2^128]").unwrap(), "True");
  }

  // CompositeQ, DigitSum, IntegerReverse, DigitCount are Listable: they
  // thread element-wise over a list argument (previously CompositeQ wrongly
  // returned a single False for the whole list).
  #[test]
  fn number_theory_predicates_thread() {
    assert_eq!(
      interpret("CompositeQ[{4, 5, 6}]").unwrap(),
      "{True, False, True}"
    );
    assert_eq!(interpret("DigitSum[{12, 123}]").unwrap(), "{3, 6}");
    assert_eq!(interpret("IntegerReverse[{12, 34}]").unwrap(), "{21, 43}");
    assert_eq!(
      interpret("DigitCount[{12, 123}, 10]").unwrap(),
      "{{1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0, 0, 0, 0}}"
    );
    // Scalar forms unchanged.
    assert_eq!(interpret("CompositeQ[7]").unwrap(), "False");
    assert_eq!(interpret("DigitSum[123]").unwrap(), "6");
  }

  // DigitSum[n, MixedRadix[{...}]] sums the mixed-radix digits. A radix of 1 is
  // valid (e.g. MixedRadix[{60, 60, 1}]) and must not be rejected.
  #[test]
  fn digit_sum_mixed_radix() {
    assert_eq!(
      interpret("DigitSum[123, MixedRadix[{60, 60, 1}]]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("DigitSum[3723, MixedRadix[{60, 60, 1}]]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("DigitSum[3661, MixedRadix[{24, 60, 60}]]").unwrap(),
      "3"
    );
    // The leading place absorbs overflow; sum still matches the digit list.
    assert_eq!(
      interpret("DigitSum[100000, MixedRadix[{60, 60, 1}]]").unwrap(),
      "113"
    );
    assert_eq!(
      interpret("DigitSum[0, MixedRadix[{60, 60, 1}]]").unwrap(),
      "0"
    );
    // Consistent with summing the mixed-radix digits.
    assert_eq!(
      interpret(
        "DigitSum[100000, MixedRadix[{60, 60, 1}]] == Total[IntegerDigits[100000, MixedRadix[{60, 60, 1}]]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn nest_with_big_integer() {
    assert_eq!(
      interpret("Nest[#+1&, 2^128, 3]").unwrap(),
      "340282366920938463463374607431768211459"
    );
  }

  #[test]
  fn fibonacci_big_integer() {
    assert_eq!(
      interpret("Fibonacci[200]").unwrap(),
      "280571172992510140037611932413038677189525"
    );
  }

  #[test]
  fn factorial_big_integer() {
    // 50! is larger than i128
    assert_eq!(
      interpret("Factorial[50]").unwrap(),
      "30414093201713378043612608166064768844377641568960512000000000000"
    );
  }

  // An integer-valued real index yields the exact factorial rounded to a
  // machine real, not the float-Gamma approximation (which drifted, e.g.
  // 120.00000000000021). Verified against wolframscript.
  #[test]
  fn factorial_integer_valued_real_is_exact() {
    assert_eq!(interpret("Factorial[5.0]").unwrap(), "120.");
    assert_eq!(interpret("Factorial[0.0]").unwrap(), "1.");
    assert_eq!(interpret("Factorial[1.0]").unwrap(), "1.");
    assert_eq!(interpret("Factorial[10.0]").unwrap(), "3.6288*^6");
    assert_eq!(
      interpret("Factorial[20.0]").unwrap(),
      "2.43290200817664*^18"
    );
  }

  // `a! b!` is implicit multiplication of two factorials. Each factor
  // in an ImplicitTimes accepts a trailing `!` / `!!` postfix, matching
  // Wolfram's binding (Factorial > implicit Times > Power).
  #[test]
  fn factorial_implicit_times_two_symbols() {
    assert_eq!(interpret("a! b!").unwrap(), "a!*b!");
  }

  #[test]
  fn factorial_implicit_times_two_integers() {
    assert_eq!(interpret("5! 3!").unwrap(), "720");
  }

  #[test]
  fn factorial_in_power_exponent_binds_tighter() {
    // `a^b!` parses as `a^(b!)` because Factorial binds tighter than Power.
    assert_eq!(
      interpret("ToString[FullForm[Hold[a^b!]]]").unwrap(),
      "Hold[Power[a, Factorial[b]]]"
    );
  }

  #[test]
  fn factorial_quotient_of_factorials() {
    // The Catalan-number style expression `(2 n)!/((n + 1)! n!)`
    // exercises factorials nested in parentheses and implicit times.
    assert_eq!(interpret("(2 n)!/((n + 1)! n!) /. n -> 5").unwrap(), "42");
  }

  #[test]
  fn digit_count_big_integer() {
    assert_eq!(interpret("DigitCount[2^128, 10, 3]").unwrap(), "7");
  }

  #[test]
  fn part_of_list_with_big_integer() {
    assert_eq!(
      interpret("{2^128, 2^129}[[1]]").unwrap(),
      "340282366920938463463374607431768211456"
    );
  }

  #[test]
  fn bit_length_big_integer() {
    assert_eq!(interpret("BitLength[2^128]").unwrap(), "129");
  }

  // Regression: f64 conversion silently rounded 1-ULP differences
  // above ~2^53 to a tie, so `2^60 < 2^60 + 1` returned False.
  #[test]
  fn big_integer_strict_less_one_ulp() {
    assert_eq!(interpret("2^60 < 2^60 + 1").unwrap(), "True");
    assert_eq!(interpret("2^10000 < 2^10000 + 1").unwrap(), "True");
  }

  #[test]
  fn big_integer_equal_one_ulp() {
    assert_eq!(interpret("2^60 == 2^60 + 1").unwrap(), "False");
    assert_eq!(interpret("2^10000 == 2^10000 + 1").unwrap(), "False");
  }

  #[test]
  fn big_integer_unequal_one_ulp() {
    assert_eq!(interpret("2^60 != 2^60 + 1").unwrap(), "True");
  }

  // Regression: storing a BigInteger via Set used to round-trip through
  // `expr_to_string` + `string_to_expr`'s f64 fallback, collapsing
  // `2^200` to a `Real`. After the fix `Head[a]` is `Integer`.
  #[test]
  fn big_integer_set_preserves_head() {
    assert_eq!(interpret("a = 2^200; Head[a]").unwrap(), "Integer");
  }

  // Regression: compound destructuring `{a, b} = {2^60, 2^60 + 1}`
  // collapsed both bindings to the same f64 when comparing for
  // equality.
  #[test]
  fn big_integer_destructured_comparison() {
    assert_eq!(
      interpret("{a, b} = {2^10000, 2^10000 + 1}; {a == b, a < b, a <= b}")
        .unwrap(),
      "{False, True, True}"
    );
  }
}

mod nary_power {
  use super::*;

  // Power folds right-associatively: Power[a, b, c] = a^(b^c).
  #[test]
  fn three_args_numeric() {
    assert_eq!(interpret("Power[2, 3, 2]").unwrap(), "512");
  }

  #[test]
  fn three_args_symbolic() {
    assert_eq!(interpret("Power[x, y, z]").unwrap(), "x^y^z");
  }

  #[test]
  fn four_args() {
    assert_eq!(interpret("Power[2, 3, 2, 1]").unwrap(), "512");
  }

  #[test]
  fn one_arg_is_identity() {
    assert_eq!(interpret("Power[x]").unwrap(), "x");
  }

  #[test]
  fn zero_args_is_one() {
    assert_eq!(interpret("Power[]").unwrap(), "1");
  }

  #[test]
  fn two_args_unchanged() {
    assert_eq!(interpret("Power[2, 3]").unwrap(), "8");
  }
}

mod sequence_in_operators {
  use super::*;

  // Sequence operands splice into the n-ary form of arithmetic operators.
  #[test]
  fn plus_two_sequences() {
    assert_eq!(interpret("Sequence[1, 2] + Sequence[3, 4]").unwrap(), "10");
  }

  #[test]
  fn plus_mixed() {
    assert_eq!(interpret("1 + Sequence[2, 3, 4]").unwrap(), "10");
  }

  #[test]
  fn times_sequences() {
    assert_eq!(interpret("Sequence[2, 3] * Sequence[4, 5]").unwrap(), "120");
  }

  #[test]
  fn times_scalar() {
    assert_eq!(interpret("2 Sequence[3, 4]").unwrap(), "24");
  }

  // Subtraction and division map onto Plus/Times after splicing.
  #[test]
  fn minus_splices_into_plus() {
    assert_eq!(interpret("Sequence[1, 2] - 3").unwrap(), "0");
  }

  #[test]
  fn divide_splices_into_times() {
    assert_eq!(interpret("Sequence[10, 20]/2").unwrap(), "100");
  }

  #[test]
  fn power_splices() {
    assert_eq!(interpret("x^Sequence[2, 3]").unwrap(), "x^8");
  }

  #[test]
  fn symbolic_plus() {
    assert_eq!(interpret("a + Sequence[b, c]").unwrap(), "a + b + c");
  }
}

mod power_with_negative_exponent {
  use super::*;

  #[test]
  fn power_negative_one_exponent() {
    assert_eq!(interpret("Power[2, -1]").unwrap(), "1/2");
  }

  #[test]
  fn power_negative_two_exponent() {
    assert_eq!(interpret("Power[3, -2]").unwrap(), "1/9");
  }

  #[test]
  fn rational_to_rational_power_negative_half() {
    // (5/2)^(-1/2) → Sqrt[2/5], exact symbolic (not a numeric decimal).
    assert_eq!(interpret("(5/2)^(-1/2)").unwrap(), "Sqrt[2/5]");
  }

  #[test]
  fn rational_to_rational_power_perfect_cube_root() {
    assert_eq!(interpret("(8/27)^(1/3)").unwrap(), "2/3");
  }

  #[test]
  fn rational_to_rational_power_stays_unevaluated_when_no_simplification() {
    // (5/2)^(1/3) has no exact simplification — keep unevaluated.
    assert_eq!(interpret("(5/2)^(1/3)").unwrap(), "(5/2)^(1/3)");
  }

  #[test]
  fn rational_to_rational_power_three_halves() {
    assert_eq!(interpret("(2/3)^(3/2)").unwrap(), "(2*Sqrt[2/3])/3");
  }

  #[test]
  fn rational_to_rational_power_minus_three_halves() {
    assert_eq!(interpret("(2/3)^(-3/2)").unwrap(), "(3*Sqrt[3/2])/2");
  }
}

mod power_of_i {
  use super::*;

  #[test]
  fn i_squared() {
    assert_eq!(interpret("I^2").unwrap(), "-1");
  }

  #[test]
  fn i_cubed() {
    assert_eq!(interpret("I^3").unwrap(), "-I");
  }

  #[test]
  fn i_fourth() {
    assert_eq!(interpret("I^4").unwrap(), "1");
  }

  #[test]
  fn i_negative_one() {
    assert_eq!(interpret("I^(-1)").unwrap(), "-I");
  }

  #[test]
  fn i_negative_two() {
    assert_eq!(interpret("I^(-2)").unwrap(), "-1");
  }

  #[test]
  fn x_to_zero() {
    assert_eq!(interpret("a^0").unwrap(), "1");
    assert_eq!(interpret("5^0").unwrap(), "1");
  }
}

mod negative_base_fractional_exponent {
  use super::*;

  #[test]
  fn n_neg1_one_third() {
    // N[(-1)^(1/3)] should give a complex number, not Indeterminate
    let result = interpret("N[(-1)^(1/3)]").unwrap();
    assert!(
      result.contains("I"),
      "Expected complex result, got: {}",
      result
    );
    assert!(
      result.contains("0.5"),
      "Expected real part ~0.5, got: {}",
      result
    );
  }

  #[test]
  fn n_neg1_two_thirds() {
    let result = interpret("N[(-1)^(2/3)]").unwrap();
    assert!(
      result.contains("I"),
      "Expected complex result, got: {}",
      result
    );
  }

  #[test]
  fn float_negative_base_fractional_exp() {
    // (-1.0)^0.5 should give I, not Indeterminate
    let result = interpret("(-1.0)^0.5").unwrap();
    assert!(
      result.contains("I"),
      "Expected complex result, got: {}",
      result
    );
  }

  #[test]
  fn negative_base_power_not_indeterminate() {
    // Negative base with fractional exponent must not return Indeterminate
    let result = interpret("(-2.0)^0.5").unwrap();
    assert_ne!(result, "Indeterminate");
    assert!(
      result.contains("I"),
      "Expected complex result, got: {}",
      result
    );
  }
}

mod subtract_function {
  use super::*;

  #[test]
  fn subtract_basic() {
    assert_eq!(interpret("Subtract[5, 2]").unwrap(), "3");
  }

  #[test]
  fn subtract_negative_result() {
    assert_eq!(interpret("Subtract[2, 5]").unwrap(), "-3");
  }

  #[test]
  fn subtract_distributes_minus_over_plus() {
    // a - (b - c) should distribute: a + (-b) + c
    assert_eq!(interpret("a - (b - c)").unwrap(), "a - b + c");
  }

  #[test]
  fn subtract_distributes_minus_over_sum() {
    // x - (a + b + c) = x - a - b - c
    assert_eq!(interpret("x - (a + b + c)").unwrap(), "-a - b - c + x");
  }

  #[test]
  fn negate_sum() {
    // -(a + b) = -a - b
    assert_eq!(interpret("-(a + b)").unwrap(), "-a - b");
  }

  #[test]
  fn subtract_nested() {
    // a - (b - (c - d)) = a - b + c - d
    assert_eq!(interpret("a - (b - (c - d))").unwrap(), "a - b + c - d");
  }
}

mod multiplication_formatting {
  use super::*;

  #[test]
  fn times_no_spaces() {
    assert_eq!(interpret("Times[2, x]").unwrap(), "2*x");
  }

  #[test]
  fn power_no_spaces() {
    assert_eq!(interpret("Power[x, 2]").unwrap(), "x^2");
  }

  #[test]
  fn negated_division_formatting() {
    // Wolfram displays -(a/b) not -a/b or (-a)/b
    assert_eq!(interpret("-a/b").unwrap(), "-(a/b)");
    assert_eq!(interpret("{-a/b}").unwrap(), "{-(a/b)}");
  }

  #[test]
  fn negated_product_formatting() {
    // Wolfram displays -(a*b) not -a*b
    assert_eq!(interpret("-a*b").unwrap(), "-(a*b)");
    assert_eq!(interpret("{-a*b}").unwrap(), "{-(a*b)}");
  }
}

mod implicit_times_in_function_body {
  use super::*;

  #[test]
  fn numeric_times_variable() {
    assert_eq!(interpret("f[x_] := 2 x; f[5]").unwrap(), "10");
  }

  #[test]
  fn numeric_times_function_call() {
    assert_eq!(
      interpret("L[n_] := 2 Fibonacci[n + 1] - 1; L[5]").unwrap(),
      "15"
    );
  }

  #[test]
  fn numeric_times_slot() {
    assert_eq!(interpret("3 # &[5]").unwrap(), "15");
  }

  #[test]
  fn numeric_times_slot_in_function_arg() {
    assert_eq!(interpret("f[3 # + 1]").unwrap(), "f[1 + 3*#1]");
  }

  #[test]
  fn numeric_times_slot_in_list() {
    assert_eq!(interpret("{3 #}").unwrap(), "{3*#1}");
  }

  #[test]
  fn numeric_times_slot_in_anonymous_function() {
    // The anonymous function is passed as an argument, not called directly
    assert_eq!(
      interpret("Map[If[EvenQ[#], #/2, 3 # + 1] &, {4, 5}]").unwrap(),
      "{2, 16}"
    );
  }
}

mod exact_value_returns {
  use super::*;

  #[test]
  fn sin_pi_half_returns_integer() {
    // Sin[Pi/2] should return 1 (Integer), not 1. (Real)
    assert_eq!(interpret("Sin[Pi/2]").unwrap(), "1");
  }

  #[test]
  fn cos_zero_returns_integer() {
    assert_eq!(interpret("Cos[0]").unwrap(), "1");
  }

  #[test]
  fn power_cube_root_returns_integer() {
    // Power[27, 1/3] should return 3 (Integer) when result is exact
    assert_eq!(interpret("Power[27, 1/3]").unwrap(), "3");
  }

  #[test]
  fn power_square_root_returns_integer() {
    assert_eq!(interpret("Power[16, 1/2]").unwrap(), "4");
  }

  #[test]
  fn large_float_power_scientific_notation() {
    // 1.05^1578 is ~2.73e33, must not overflow to i64::MAX
    let result = interpret("1.05^1578").unwrap();
    assert!(
      result.contains("*^"),
      "Expected scientific notation, got: {}",
      result
    );
    assert!(result.starts_with("2.73346"));
  }

  #[test]
  fn large_float_multiplication_scientific_notation() {
    // 50 * 1.05^1578 should produce scientific notation
    let result = interpret("50 * (1 + 0.05)^1578").unwrap();
    assert!(result.contains("*^35"), "Expected *^35, got: {}", result);
  }

  #[test]
  fn real_scientific_notation_formatting() {
    assert_eq!(interpret("1000000.").unwrap(), "1.*^6");
    assert_eq!(interpret("999999.").unwrap(), "999999.");
    assert_eq!(interpret("0.000001").unwrap(), "1.*^-6");
    assert_eq!(interpret("0.00001").unwrap(), "0.00001");
  }

  #[test]
  fn real_scientific_notation_in_list() {
    assert_eq!(
      interpret("{1000000., 0.000001}").unwrap(),
      "{1.*^6, 1.*^-6}"
    );
  }

  #[test]
  fn parse_scientific_notation_literal() {
    // *^ notation should be parseable as a number literal
    assert_eq!(interpret("2.7*^7").unwrap(), "2.7*^7");
    assert_eq!(interpret("1.5*^-6").unwrap(), "1.5*^-6");
    assert_eq!(interpret("-3.14*^10").unwrap(), "-3.14*^10");
    assert_eq!(interpret("1.*^3").unwrap(), "1000.");
  }

  // Parsing `1.09*^12` as `mantissa * 10^exp` in f64 introduces rounding
  // (1.09 * 1e12 = 1090000000000.0001). Parsing the full literal as
  // `1.09e12` in one step yields the exact nearest f64, matching
  // wolframscript's display `1.09*^12`.
  #[test]
  fn scientific_literal_is_exact_to_nearest_f64() {
    assert_eq!(interpret("1.09*^12").unwrap(), "1.09*^12");
    assert_eq!(
      interpret("Complex[1.09*^12, 3.]").unwrap(),
      "1.09*^12 + 3.*I"
    );
  }

  #[test]
  fn scientific_notation_arithmetic() {
    // Arithmetic with *^ notation should work
    assert_eq!(interpret("2.5*^7 + 3.0*^7").unwrap(), "5.5*^7");
    assert_eq!(interpret("1.*^6 * 2").unwrap(), "2.*^6");
  }

  #[test]
  fn scientific_notation_in_compound_expression() {
    // *^ results from intermediate computations should be usable in subsequent expressions
    assert_eq!(interpret("x = 1.*^6; x * 2").unwrap(), "2.*^6");
  }

  #[test]
  fn integer_scientific_notation() {
    // Integer *^ with positive exponent produces Integer
    assert_eq!(interpret("5*^3").unwrap(), "5000");
    assert_eq!(interpret("5*^0").unwrap(), "5");
    assert_eq!(interpret("1*^6").unwrap(), "1000000");
    assert_eq!(interpret("Head[5*^3]").unwrap(), "Integer");

    // Integer *^ with negative exponent produces Rational
    assert_eq!(interpret("5*^-13").unwrap(), "1/2000000000000");
    assert_eq!(interpret("Head[5*^-13]").unwrap(), "Rational");
    assert_eq!(interpret("12*^-3").unwrap(), "3/250");
    assert_eq!(interpret("100*^-2").unwrap(), "1");

    // Negative mantissa
    assert_eq!(interpret("-5*^3").unwrap(), "-5000");
    assert_eq!(interpret("-5*^-13").unwrap(), "-1/2000000000000");
    assert_eq!(interpret("-2/3 // Head").unwrap(), "Rational");
    assert_eq!(interpret("-2/3").unwrap(), "-2/3");

    // Works in Quantity
    assert_eq!(
      interpret("Quantity[5*^-13, \"Bars\"]").unwrap(),
      "Quantity[1/2000000000000, Bars]"
    );
  }

  #[test]
  fn mean_returns_rational() {
    // Mean[{0, 0, 0, 10}] = 10/4 = 5/2
    assert_eq!(interpret("Mean[{0, 0, 0, 10}]").unwrap(), "5/2");
  }

  #[test]
  fn mean_returns_integer_when_exact() {
    // Mean[{2, 4, 6}] = 12/3 = 4
    assert_eq!(interpret("Mean[{2, 4, 6}]").unwrap(), "4");
  }

  #[test]
  fn median_even_count_returns_rational() {
    // Median[{1, 2, 3, 4}] = (2+3)/2 = 5/2
    assert_eq!(interpret("Median[{1, 2, 3, 4}]").unwrap(), "5/2");
  }

  #[test]
  fn median_odd_count_returns_integer() {
    // Median[{1, 2, 3}] = 2
    assert_eq!(interpret("Median[{1, 2, 3}]").unwrap(), "2");
  }

  #[test]
  fn median_preserves_real_type() {
    // Median of reals should return real
    assert_eq!(interpret("Median[{1.5, 2.5, 3.5, 4.5}]").unwrap(), "3.");
  }

  // On an association, statistics operate on the values.
  #[test]
  fn median_association() {
    assert_eq!(
      interpret("Median[<|a -> 1, b -> 2, c -> 3|>]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Median[<|a -> 1, b -> 2, c -> 3, d -> 4|>]").unwrap(),
      "5/2"
    );
  }

  #[test]
  fn min_max_association() {
    assert_eq!(
      interpret("MinMax[<|a -> 3, b -> 1, c -> 5|>]").unwrap(),
      "{1, 5}"
    );
  }

  #[test]
  fn variance_and_stddev_association() {
    assert_eq!(
      interpret("Variance[<|a -> 1, b -> 2, c -> 3|>]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("StandardDeviation[<|a -> 2, b -> 4, c -> 6|>]").unwrap(),
      "2"
    );
  }

  #[test]
  fn variance_symbolic_list() {
    // Symbolic Variance[list] == Covariance[list, list], using the
    // Conjugate-based deviation product. wolframscript factors the n == 2 case
    // and leaves n >= 3 expanded — Woxi must match both forms exactly.
    assert_eq!(
      interpret("Variance[{a, b}]").unwrap(),
      "((a - b)*(Conjugate[a] - Conjugate[b]))/2"
    );
    assert_eq!(
      interpret("Variance[{a, b, c}]").unwrap(),
      "((2*a - b - c)*Conjugate[a] + (-a + 2*b - c)*Conjugate[b] + \
       (-a - b + 2*c)*Conjugate[c])/6"
    );
    // Mixed symbolic/numeric entries keep the factored n == 2 shape.
    assert_eq!(
      interpret("Variance[{x, 2}]").unwrap(),
      "((-2 + x)*(-2 + Conjugate[x]))/2"
    );
    // Numeric paths are unaffected by the symbolic delegation.
    assert_eq!(interpret("Variance[{1, 2, 3, 4, 5}]").unwrap(), "5/2");
    assert_eq!(interpret("Variance[{2 + I, 3 - I}]").unwrap(), "5/2");
  }

  #[test]
  fn quantile_association() {
    assert_eq!(
      interpret("Quantile[<|a -> 1, b -> 2, c -> 3, d -> 4|>, 1/2]").unwrap(),
      "2"
    );
  }

  #[test]
  fn quotient_positive() {
    assert_eq!(interpret("Quotient[23, 7]").unwrap(), "3");
  }

  #[test]
  fn quotient_negative_dividend() {
    // Floor division, not truncation
    assert_eq!(interpret("Quotient[-23, 7]").unwrap(), "-4");
  }

  #[test]
  fn quotient_negative_divisor() {
    assert_eq!(interpret("Quotient[23, -7]").unwrap(), "-4");
  }

  #[test]
  fn quotient_remainder_basic() {
    assert_eq!(interpret("QuotientRemainder[23, 7]").unwrap(), "{3, 2}");
  }

  #[test]
  fn quotient_remainder_negative() {
    assert_eq!(interpret("QuotientRemainder[-23, 7]").unwrap(), "{-4, 5}");
  }

  #[test]
  fn quotient_remainder_negative_divisor() {
    assert_eq!(interpret("QuotientRemainder[23, -7]").unwrap(), "{-4, -5}");
  }

  // Gaussian Quotient = Round[m/n] (component-wise round-half-to-even); the
  // result displays as `a + b*I`, not a raw Complex[...] literal.
  #[test]
  fn quotient_gaussian_integer() {
    assert_eq!(interpret("Quotient[7 + 3 I, 2]").unwrap(), "4 + 2*I");
    assert_eq!(interpret("Quotient[1 + I, 2]").unwrap(), "0");
    assert_eq!(interpret("Quotient[3 + I, 2]").unwrap(), "2");
    assert_eq!(interpret("Quotient[1 + 6 I, 2 + I]").unwrap(), "2 + 2*I");
    assert_eq!(interpret("Quotient[5, 2 + I]").unwrap(), "2 - I");
  }

  #[test]
  fn quotient_remainder_gaussian() {
    assert_eq!(
      interpret("QuotientRemainder[7 + 3 I, 2]").unwrap(),
      "{4 + 2*I, -1 - I}"
    );
    assert_eq!(
      interpret("QuotientRemainder[1 + I, 2]").unwrap(),
      "{0, 1 + I}"
    );
  }

  #[test]
  fn quotient_with_offset_positive() {
    // Quotient[n, m, d] = Floor[(n - d) / m]
    // (17 - 3)/5 = 2.8, Floor = 2
    assert_eq!(interpret("Quotient[17, 5, 3]").unwrap(), "2");
  }

  #[test]
  fn quotient_with_offset_one() {
    assert_eq!(interpret("Quotient[17, 5, 1]").unwrap(), "3");
  }

  #[test]
  fn quotient_with_offset_negative_dividend() {
    // (-17 - 3)/5 = -4, Floor = -4
    assert_eq!(interpret("Quotient[-17, 5, 3]").unwrap(), "-4");
  }

  #[test]
  fn quotient_with_offset_invariant() {
    // n == Quotient[n, m, d] * m + Mod[n, m, d]
    assert_eq!(
      interpret("17 - (Quotient[17, 5, 3] * 5 + Mod[17, 5, 3])").unwrap(),
      "0"
    );
  }

  #[test]
  fn quotient_with_offset_rationals() {
    // Quotient[17/2, 3/2, 1/2] = Floor[((17/2) - (1/2)) / (3/2)]
    //                         = Floor[8 / (3/2)]
    //                         = Floor[16/3] = 5
    assert_eq!(interpret("Quotient[17/2, 3/2, 1/2]").unwrap(), "5");
  }
}

mod infinity_arithmetic {
  use super::*;

  #[test]
  fn infinity_plus_finite() {
    assert_eq!(interpret("Infinity + 100").unwrap(), "Infinity");
  }

  #[test]
  fn neg_infinity_plus_finite() {
    assert_eq!(interpret("-Infinity + 100").unwrap(), "-Infinity");
  }

  #[test]
  fn real_coefficient_times_infinity() {
    // A positive real coefficient × Infinity collapses to Infinity;
    // a negative one to -Infinity (matching the integer/rational cases).
    assert_eq!(interpret("2.5 Infinity").unwrap(), "Infinity");
    assert_eq!(interpret("-2.5 Infinity").unwrap(), "-Infinity");
    assert_eq!(interpret("2.5 (-Infinity)").unwrap(), "-Infinity");
  }

  #[test]
  fn positive_constant_times_infinity() {
    // Known-positive constants and algebraic factors are absorbed.
    assert_eq!(interpret("Pi Infinity").unwrap(), "Infinity");
    assert_eq!(interpret("E Infinity").unwrap(), "Infinity");
    assert_eq!(interpret("GoldenRatio Infinity").unwrap(), "Infinity");
    assert_eq!(interpret("Degree Infinity").unwrap(), "Infinity");
    assert_eq!(interpret("Sqrt[2] Infinity").unwrap(), "Infinity");
    assert_eq!(interpret("-Pi Infinity").unwrap(), "-Infinity");
  }

  #[test]
  fn symbolic_coefficient_times_infinity_unchanged() {
    // A factor of unknown sign keeps the product symbolic.
    assert_eq!(interpret("a Infinity").unwrap(), "a*Infinity");
  }

  #[test]
  fn e_to_infinity() {
    assert_eq!(interpret("E^Infinity").unwrap(), "Infinity");
  }

  #[test]
  fn e_to_neg_infinity() {
    assert_eq!(interpret("E^(-Infinity)").unwrap(), "0");
  }

  #[test]
  fn exp_infinity() {
    assert_eq!(interpret("Exp[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn exp_neg_infinity() {
    assert_eq!(interpret("Exp[-Infinity]").unwrap(), "0");
  }

  #[test]
  fn infinity_to_positive_power() {
    assert_eq!(interpret("Infinity^2").unwrap(), "Infinity");
    assert_eq!(interpret("Infinity^10").unwrap(), "Infinity");
  }

  #[test]
  fn infinity_to_negative_power() {
    assert_eq!(interpret("Infinity^(-1)").unwrap(), "0");
    assert_eq!(interpret("Infinity^(-3)").unwrap(), "0");
  }

  #[test]
  fn neg_infinity_to_integer_power() {
    assert_eq!(interpret("(-Infinity)^2").unwrap(), "Infinity");
    assert_eq!(interpret("(-Infinity)^3").unwrap(), "-Infinity");
    assert_eq!(interpret("(-Infinity)^(-1)").unwrap(), "0");
  }

  #[test]
  fn base_to_infinity() {
    assert_eq!(interpret("2^Infinity").unwrap(), "Infinity");
    assert_eq!(interpret("Power[1/2, Infinity]").unwrap(), "0");
    assert_eq!(interpret("1^Infinity").unwrap(), "Indeterminate");
  }

  #[test]
  fn base_to_neg_infinity() {
    assert_eq!(interpret("2^(-Infinity)").unwrap(), "0");
    assert_eq!(interpret("Power[1/2, -Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn infinity_divide_infinity() {
    assert_eq!(interpret("Infinity / Infinity").unwrap(), "Indeterminate");
  }

  #[test]
  fn complex_infinity_power() {
    assert_eq!(interpret("ComplexInfinity^2").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("ComplexInfinity^(-1)").unwrap(), "0");
  }

  #[test]
  fn infinity_power_zero_is_indeterminate() {
    assert_eq!(interpret("Infinity^0").unwrap(), "Indeterminate");
  }

  #[test]
  fn sqrt_infinity() {
    assert_eq!(interpret("Sqrt[Infinity]").unwrap(), "Infinity");
    assert_eq!(
      interpret("Sqrt[ComplexInfinity]").unwrap(),
      "ComplexInfinity"
    );
  }

  #[test]
  fn infinity_times_zero() {
    assert_eq!(interpret("Infinity * 0").unwrap(), "Indeterminate");
    assert_eq!(interpret("0 * Infinity").unwrap(), "Indeterminate");
    assert_eq!(interpret("-Infinity * 0").unwrap(), "Indeterminate");
    assert_eq!(interpret("0 * (-Infinity)").unwrap(), "Indeterminate");
    assert_eq!(interpret("ComplexInfinity * 0").unwrap(), "Indeterminate");
    assert_eq!(interpret("0 * ComplexInfinity").unwrap(), "Indeterminate");
  }

  #[test]
  fn division_by_zero() {
    // n/0 → ComplexInfinity
    assert_eq!(interpret("1/0").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("5/0").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("(-1)/0").unwrap(), "ComplexInfinity");
    // 0/0 → Indeterminate (0 * ComplexInfinity)
    assert_eq!(interpret("0/0").unwrap(), "Indeterminate");
  }

  #[test]
  fn power_zero_negative() {
    // 0^(-n) → ComplexInfinity
    assert_eq!(interpret("Power[0, -1]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Power[0, -2]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn one_to_any_power() {
    // 1^x → 1 for any x
    assert_eq!(interpret("1^x").unwrap(), "1");
    assert_eq!(interpret("1^(2 + 3 I)").unwrap(), "1");
    assert_eq!(interpret("1^100").unwrap(), "1");
  }

  #[test]
  fn integral_with_infinity_boundary() {
    assert_eq!(
      interpret("Integrate[x^2 Exp[-x], {x, 0, Infinity}]").unwrap(),
      "2"
    );
  }
}

mod infinity_comparisons {
  use super::*;

  #[test]
  fn less_than_infinity() {
    assert_eq!(interpret("5 < Infinity").unwrap(), "True");
    assert_eq!(interpret("0 < Infinity").unwrap(), "True");
    assert_eq!(interpret("-100 < Infinity").unwrap(), "True");
    assert_eq!(interpret("Infinity < 5").unwrap(), "False");
    assert_eq!(interpret("Infinity < Infinity").unwrap(), "False");
    assert_eq!(interpret("-Infinity < Infinity").unwrap(), "True");
    assert_eq!(interpret("-Infinity < 0").unwrap(), "True");
  }

  #[test]
  fn greater_than_infinity() {
    assert_eq!(interpret("Infinity > 5").unwrap(), "True");
    assert_eq!(interpret("Infinity > 0").unwrap(), "True");
    assert_eq!(interpret("Greater[Infinity, -100]").unwrap(), "True");
    assert_eq!(interpret("5 > Infinity").unwrap(), "False");
    assert_eq!(interpret("Infinity > Infinity").unwrap(), "False");
    assert_eq!(interpret("Greater[Infinity, -Infinity]").unwrap(), "True");
    assert_eq!(interpret("Greater[0, -Infinity]").unwrap(), "True");
  }

  #[test]
  fn less_equal_infinity() {
    assert_eq!(interpret("5 <= Infinity").unwrap(), "True");
    assert_eq!(interpret("Infinity <= Infinity").unwrap(), "True");
    assert_eq!(interpret("Infinity <= 5").unwrap(), "False");
    assert_eq!(interpret("-Infinity <= Infinity").unwrap(), "True");
    assert_eq!(
      interpret("LessEqual[-Infinity, -Infinity]").unwrap(),
      "True"
    );
  }

  #[test]
  fn greater_equal_infinity() {
    assert_eq!(interpret("Infinity >= 5").unwrap(), "True");
    assert_eq!(interpret("Infinity >= Infinity").unwrap(), "True");
    assert_eq!(interpret("5 >= Infinity").unwrap(), "False");
    assert_eq!(
      interpret("GreaterEqual[Infinity, -Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("GreaterEqual[-Infinity, -Infinity]").unwrap(),
      "True"
    );
  }

  #[test]
  fn infinity_in_if_condition() {
    clear_state();
    assert_eq!(
      interpret("If[5 < Infinity, \"yes\", \"no\"]").unwrap(),
      "yes"
    );
    assert_eq!(
      interpret("If[Infinity < 5, \"yes\", \"no\"]").unwrap(),
      "no"
    );
  }
}

mod length_atoms {
  use super::*;

  #[test]
  fn rational_length_zero() {
    assert_eq!(interpret("Length[1/3]").unwrap(), "0");
  }

  #[test]
  fn integer_length_zero() {
    assert_eq!(interpret("Length[42]").unwrap(), "0");
  }
}

mod complement_tests {
  use super::*;

  #[test]
  fn complement_with_function_head() {
    assert_eq!(
      interpret("Complement[f[z, y, x, w], f[x], f[x, z]]").unwrap(),
      "f[w, y]"
    );
  }
}

mod negative_symbolic {
  use super::*;

  #[test]
  fn negative_symbolic_stays_unevaluated() {
    clear_state();
    assert_eq!(interpret("Negative[a + b]").unwrap(), "Negative[a + b]");
  }

  #[test]
  fn negative_known_value() {
    assert_eq!(interpret("Negative[-5]").unwrap(), "True");
  }
}

mod n_constants {
  use super::*;

  #[test]
  fn euler_gamma() {
    let r = interpret("N[EulerGamma]").unwrap();
    assert!(r.starts_with("0.577215"), "got: {}", r);
  }

  #[test]
  fn golden_ratio() {
    let r = interpret("N[GoldenRatio]").unwrap();
    assert!(r.starts_with("1.61803"), "got: {}", r);
  }

  #[test]
  fn catalan() {
    let r = interpret("N[Catalan]").unwrap();
    assert!(r.starts_with("0.91596"), "got: {}", r);
  }

  #[test]
  fn glaisher() {
    let r = interpret("N[Glaisher]").unwrap();
    assert!(r.starts_with("1.28242"), "got: {}", r);
  }

  #[test]
  fn khinchin() {
    let r = interpret("N[Khinchin]").unwrap();
    assert!(r.starts_with("2.68545"), "got: {}", r);
  }

  #[test]
  fn machine_precision() {
    let r = interpret("N[MachinePrecision]").unwrap();
    assert!(r.starts_with("15.9545"), "got: {}", r);
  }

  #[test]
  fn machine_precision_exact() {
    // Full double-precision value matches wolframscript exactly.
    assert_eq!(
      interpret("N[MachinePrecision]").unwrap(),
      "15.954589770191003"
    );
  }
}

mod symbol_function {
  use super::*;

  #[test]
  fn symbol_creates_identifier() {
    clear_state();
    assert_eq!(interpret("Symbol[\"x\"] + Symbol[\"x\"]").unwrap(), "2*x");
  }

  #[test]
  fn symbol_name() {
    clear_state();
    assert_eq!(interpret("SymbolName[x]").unwrap(), "x");
  }
}

mod composition_flatten {
  use super::*;

  #[test]
  fn flatten_nested() {
    clear_state();
    assert_eq!(
      interpret("Composition[f, Composition[g, h]]").unwrap(),
      "f @* g @* h"
    );
  }

  // Composition binds tighter than prefix application: `f @* g @ x` parses as
  // `(f @* g) @ x`, i.e. f[g[x]].
  #[test]
  fn binds_tighter_than_apply() {
    clear_state();
    assert_eq!(interpret("f @* g @ x").unwrap(), "f[g[x]]");
    assert_eq!(
      interpret("Identity @* Reverse @ {1, 2, 3}").unwrap(),
      "{3, 2, 1}"
    );
    assert_eq!(
      interpret("Reverse @* Sort @ {3, 1, 2}").unwrap(),
      "{3, 2, 1}"
    );
    assert_eq!(interpret("Length @* Reverse @ {1, 2, 3, 4}").unwrap(), "4");
  }
}

mod rotate_nonlist {
  use super::*;

  #[test]
  fn rotate_right_function_head() {
    clear_state();
    assert_eq!(
      interpret("RotateRight[x[a, b, c], 2]").unwrap(),
      "x[b, c, a]"
    );
  }

  #[test]
  fn rotate_left_function_head() {
    clear_state();
    assert_eq!(
      interpret("RotateLeft[x[a, b, c], 1]").unwrap(),
      "x[b, c, a]"
    );
  }
}

mod exp_function {
  use super::*;

  #[test]
  fn exp_zero() {
    assert_eq!(interpret("Exp[0]").unwrap(), "1");
  }

  #[test]
  fn exp_one() {
    assert_eq!(interpret("Exp[1]").unwrap(), "E");
  }

  #[test]
  fn exp_integer() {
    assert_eq!(interpret("Exp[2]").unwrap(), "E^2");
  }

  #[test]
  fn exp_negative_integer() {
    assert_eq!(interpret("Exp[-1]").unwrap(), "E^(-1)");
  }

  #[test]
  fn exp_symbol() {
    assert_eq!(interpret("Exp[x]").unwrap(), "E^x");
  }

  #[test]
  fn exp_sum() {
    assert_eq!(interpret("Exp[y + z]").unwrap(), "E^(y + z)");
  }

  #[test]
  fn exp_real() {
    assert_eq!(interpret("Exp[1.0]").unwrap(), "2.718281828459045");
  }
}

mod power_of_power {
  use super::*;

  #[test]
  fn power_of_power_both_positive_integers() {
    assert_eq!(interpret("(y^2)^3").unwrap(), "y^6");
  }

  #[test]
  fn power_of_power_negative_outer() {
    assert_eq!(interpret("(x^3)^(-2)").unwrap(), "x^(-6)");
  }

  #[test]
  fn power_of_power_negative_inner() {
    assert_eq!(interpret("(x^(-1))^3").unwrap(), "x^(-3)");
  }

  #[test]
  fn power_of_power_both_negative() {
    assert_eq!(interpret("(x^(-2))^(-3)").unwrap(), "x^6");
  }

  #[test]
  fn power_of_power_outer_zero() {
    assert_eq!(interpret("(x^2)^0").unwrap(), "1");
  }

  #[test]
  fn power_of_power_outer_one() {
    assert_eq!(interpret("(x^2)^1").unwrap(), "x^2");
  }

  #[test]
  fn power_of_power_numeric_base() {
    assert_eq!(interpret("(2^3)^2").unwrap(), "64");
  }

  #[test]
  fn power_of_rational_raised_to_real() {
    // (a^(1/2))^3. → a^1.5 — real outer exponent combines with numeric inner.
    assert_eq!(interpret("(a^(1/2))^3.").unwrap(), "a^1.5");
  }

  #[test]
  fn power_of_real_raised_to_real() {
    // (a^0.3)^3. → a^0.9 (float imprecision expected, matches wolframscript).
    assert_eq!(interpret("(a^(.3))^3.").unwrap(), "a^0.8999999999999999");
  }

  #[test]
  fn power_of_integer_raised_to_real() {
    // (a^2)^3. stays unevaluated: Wolfram only combines exponents through a
    // Real outer when the inner exponent satisfies |p| < 1 (branch safety).
    assert_eq!(interpret("(a^2)^3.").unwrap(), "(a^2)^3.");
  }
}

mod power_combining {
  use super::*;

  #[test]
  fn same_base_add_exponents() {
    assert_eq!(interpret("x^2 * x^3").unwrap(), "x^5");
  }

  #[test]
  fn bare_times_bare() {
    assert_eq!(interpret("x * x").unwrap(), "x^2");
  }

  #[test]
  fn bare_times_power() {
    assert_eq!(interpret("x * x^2").unwrap(), "x^3");
  }

  #[test]
  fn negative_exponent_combining() {
    assert_eq!(interpret("x^(-1) * x^2").unwrap(), "x");
  }

  #[test]
  fn exponents_cancel_to_zero() {
    assert_eq!(interpret("x^3 * x^(-3)").unwrap(), "1");
  }

  #[test]
  fn three_factors_same_base() {
    assert_eq!(interpret("x * x^2 * x^3").unwrap(), "x^6");
  }

  #[test]
  fn different_bases_no_combining() {
    assert_eq!(interpret("x^2 * y^3").unwrap(), "x^2*y^3");
  }

  #[test]
  fn mixed_bases_partial_combining() {
    assert_eq!(interpret("x^2 * y * x^3").unwrap(), "x^5*y");
  }

  #[test]
  fn sqrt_combining_numeric() {
    assert_eq!(interpret("Sqrt[2] * Sqrt[3]").unwrap(), "Sqrt[6]");
  }

  #[test]
  fn sqrt_same_base_gives_base() {
    assert_eq!(interpret("Sqrt[x] * Sqrt[x]").unwrap(), "x");
  }

  #[test]
  fn sqrt_display_from_power() {
    assert_eq!(interpret("6^(1/2)").unwrap(), "Sqrt[6]");
  }

  #[test]
  fn cube_root_combining_numeric() {
    assert_eq!(interpret("2^(1/3) * 3^(1/3)").unwrap(), "6^(1/3)");
  }

  #[test]
  fn sqrt_of_perfect_square_simplifies() {
    assert_eq!(interpret("Sqrt[4]").unwrap(), "2");
  }

  #[test]
  fn coefficient_times_power() {
    assert_eq!(interpret("3 * x^2 * x^3").unwrap(), "3*x^5");
  }
}

mod sqrt_negative {
  use super::*;

  #[test]
  fn sqrt_neg_1() {
    assert_eq!(interpret("Sqrt[-1]").unwrap(), "I");
  }

  #[test]
  fn sqrt_neg_4() {
    assert_eq!(interpret("Sqrt[-4]").unwrap(), "2*I");
  }

  #[test]
  fn sqrt_neg_9() {
    assert_eq!(interpret("Sqrt[-9]").unwrap(), "3*I");
  }

  #[test]
  fn sqrt_neg_2() {
    assert_eq!(interpret("Sqrt[-2]").unwrap(), "I*Sqrt[2]");
  }

  #[test]
  fn sqrt_neg_12() {
    assert_eq!(interpret("Sqrt[-12]").unwrap(), "(2*I)*Sqrt[3]");
  }

  #[test]
  fn sqrt_equals_i() {
    assert_eq!(interpret("I == Sqrt[-1]").unwrap(), "True");
  }
}

mod sqrt_power {
  use super::*;

  #[test]
  fn perfect_root_of_large_integer() {
    // Regression: the perfect-root extraction cast the i128 base to u64 (and
    // used imprecise f64), so roots of large bases were 1 / garbage.
    // (2^90 as u64 == 0 made (2^90)^(1/3) return 1.)
    assert_eq!(interpret("(2^90)^(1/3)").unwrap(), "1073741824");
    assert_eq!(interpret("(10^30)^(1/3)").unwrap(), "10000000000");
    // BigInteger base (beyond i128).
    assert_eq!(
      interpret("(10^60)^(1/2)").unwrap(),
      "1000000000000000000000000000000"
    );
    assert_eq!(interpret("Sqrt[10^40]").unwrap(), "100000000000000000000");
    // Edge cases and non-perfect roots are unaffected.
    assert_eq!(interpret("0^(1/3)").unwrap(), "0");
    assert_eq!(interpret("1^(1/5)").unwrap(), "1");
    assert_eq!(interpret("100^(1/3)").unwrap(), "10^(2/3)");
    assert_eq!(interpret("Sqrt[12]").unwrap(), "2*Sqrt[3]");
  }

  #[test]
  fn sqrt_partial_square_factor_of_bigint() {
    // Regression: Sqrt of a large/BigInteger non-square pulled out no perfect
    // square factor (stayed Sqrt[10^41] etc.). It now extracts it, matching
    // wolframscript. This also fixes Norm of large-integer vectors.
    assert_eq!(
      interpret("Sqrt[10^41]").unwrap(),
      "100000000000000000000*Sqrt[10]"
    );
    assert_eq!(
      interpret("Sqrt[8*10^40]").unwrap(),
      "200000000000000000000*Sqrt[2]"
    );
    assert_eq!(
      interpret("Sqrt[12*10^30]").unwrap(),
      "2000000000000000*Sqrt[3]"
    );
    assert_eq!(
      interpret("Norm[{10^20, 10^20, 10^20}]").unwrap(),
      "100000000000000000000*Sqrt[3]"
    );
    // Skewness of a large-integer list is now value-correct (its central
    // moment m2^(3/2) = Sqrt[2744*10^90/729] now extracts its square factor),
    // rather than the garbage integer it used to give.
    let skew = interpret("N[Skewness[{10^15, 2*10^15, 4*10^15}]]").unwrap();
    assert!(
      skew.starts_with("0.38180177"),
      "skewness should be ~0.3818, got {skew}"
    );
  }

  #[test]
  fn times_large_integer_rational_symbolic_cancels() {
    // Regression: the >2^53 ("needs BigInt") branch of times_ast folded only
    // Integer/BigInteger factors and pushed a Rational coefficient into the
    // symbolic args, so `large_int * (1/large_int) * sym` stayed unreduced as
    // `(N*sym)/N`. It now folds the rational into a BigInt fraction and cancels.
    assert_eq!(
      interpret("Times[15000000000000000, Rational[1, 10000000000000000], x]")
        .unwrap(),
      "(3*x)/2"
    );
    assert_eq!(interpret("10^20*x/10^20").unwrap(), "x");
    assert_eq!(
      interpret("Times[10^18, Rational[2, 3], x, y]").unwrap(),
      "(2000000000000000000*x*y)/3"
    );
    // BigInt × Real coefficient still collapses to a single Real.
    assert_eq!(interpret("2.5*10^20").unwrap(), "2.5*^20");
  }

  #[test]
  fn sqrt_rational_numerator_exceeding_u64() {
    // Regression: Sqrt of a Rational whose numerator fit i128 but exceeded u64
    // truncated via `as u64`, giving a garbage radicand. The u64 fast path is
    // now guarded so such inputs take the BigInt extraction path.
    assert_eq!(
      interpret("Sqrt[7*10^30/3]").unwrap(),
      "1000000000000000*Sqrt[7/3]"
    );
    assert_eq!(
      interpret("Sqrt[7*10^40/3]").unwrap(),
      "100000000000000000000*Sqrt[7/3]"
    );
  }

  #[test]
  fn sqrt_absorb_coefficient_no_overflow_panic() {
    // Regression: the `(1/d)*Sqrt[p/q]` absorption optimization computed `d*d`
    // unchecked, panicking with "attempt to multiply with overflow" for a large
    // denominator. It now uses checked arithmetic and skips on overflow.
    assert_eq!(
      interpret("150000000000000000000*(100000000000000000000*Sqrt[7/3])^(-1)")
        .unwrap(),
      "(3*Sqrt[3/7])/2"
    );
    // Small absorption case is unchanged.
    assert_eq!(interpret("(1/3)*Sqrt[15/11]").unwrap(), "Sqrt[5/33]");
  }

  #[test]
  fn sqrt_squared() {
    assert_eq!(interpret("Sqrt[a]^2").unwrap(), "a");
  }

  #[test]
  fn sqrt_cubed() {
    assert_eq!(interpret("Sqrt[3]^2").unwrap(), "3");
  }

  #[test]
  fn sqrt_to_fourth() {
    assert_eq!(interpret("Sqrt[2]^4").unwrap(), "4");
  }

  #[test]
  fn sqrt_neg1_squared() {
    assert_eq!(interpret("Sqrt[-1]^2").unwrap(), "-1");
  }
}

mod expand_complex {
  use super::*;

  #[test]
  fn expand_conjugate_product() {
    assert_eq!(interpret("Expand[(3+I)*(3-I)]").unwrap(), "10");
  }

  #[test]
  fn expand_i_squared() {
    assert_eq!(interpret("Expand[(1+I)^2]").unwrap(), "2*I");
  }
}

mod radical_simplification {
  use super::*;

  #[test]
  fn cube_root_4() {
    assert_eq!(interpret("4^(1/3)").unwrap(), "2^(2/3)");
  }

  #[test]
  fn cube_root_8() {
    assert_eq!(interpret("8^(1/3)").unwrap(), "2");
  }

  #[test]
  fn cube_root_27() {
    assert_eq!(interpret("27^(1/3)").unwrap(), "3");
  }

  #[test]
  fn sqrt_12() {
    assert_eq!(interpret("12^(1/2)").unwrap(), "2*Sqrt[3]");
  }

  #[test]
  fn sqrt_18() {
    assert_eq!(interpret("18^(1/2)").unwrap(), "3*Sqrt[2]");
  }

  #[test]
  fn sqrt_72() {
    assert_eq!(interpret("72^(1/2)").unwrap(), "6*Sqrt[2]");
  }

  #[test]
  fn fifth_root_32() {
    assert_eq!(interpret("32^(1/5)").unwrap(), "2");
  }

  #[test]
  fn sixth_root_irreducible() {
    assert_eq!(interpret("6^(1/3)").unwrap(), "6^(1/3)");
  }
}

mod division_flattening {
  use super::*;

  #[test]
  fn nested_division_flattens() {
    assert_eq!(interpret("a/b/c").unwrap(), "a/(b*c)");
  }

  #[test]
  fn division_by_division() {
    assert_eq!(interpret("a/(b/c)").unwrap(), "(a*c)/b");
  }

  #[test]
  fn simple_symbolic_division() {
    assert_eq!(interpret("a/b").unwrap(), "a/b");
  }

  #[test]
  fn real_division_two_by_nine() {
    assert_eq!(interpret("2./9.").unwrap(), "0.2222222222222222");
  }

  #[test]
  fn division_by_one_simplifies() {
    assert_eq!(interpret("x/1").unwrap(), "x");
    assert_eq!(interpret("Log[x]/1").unwrap(), "Log[x]");
    assert_eq!(interpret("Sin[x]/1").unwrap(), "Sin[x]");
    assert_eq!(interpret("(a + b)/1").unwrap(), "a + b");
    assert_eq!(interpret("2/1").unwrap(), "2");
  }

  #[test]
  fn numeric_division_unchanged() {
    assert_eq!(interpret("2/3").unwrap(), "2/3");
  }

  #[test]
  fn reciprocal() {
    assert_eq!(interpret("1/x").unwrap(), "x^(-1)");
  }

  #[test]
  fn rational_direct_evaluation() {
    // Already simplified
    assert_eq!(interpret("Rational[4, 3]").unwrap(), "4/3");
    // GCD simplification
    assert_eq!(interpret("Rational[6, 4]").unwrap(), "3/2");
    // Integer when denom = 1
    assert_eq!(interpret("Rational[4, 2]").unwrap(), "2");
    // Zero numerator
    assert_eq!(interpret("Rational[0, 5]").unwrap(), "0");
    // Sign normalization
    assert_eq!(interpret("Rational[-3, -5]").unwrap(), "3/5");
    // Division by zero
    assert_eq!(interpret("Rational[3, 0]").unwrap(), "ComplexInfinity");
    // Identity cases
    assert_eq!(interpret("Rational[1, 1]").unwrap(), "1");
    assert_eq!(interpret("Rational[-7, 1]").unwrap(), "-7");
    // Arithmetic with direct Rational
    assert_eq!(interpret("Rational[1, 3] + Rational[1, 6]").unwrap(), "1/2");
  }

  #[test]
  fn rational_division() {
    // Rational / Rational
    assert_eq!(interpret("(1/3) / (1/4)").unwrap(), "4/3");
    // Rational / Integer
    assert_eq!(interpret("(2/3) / 5").unwrap(), "2/15");
    // Integer / Rational
    assert_eq!(interpret("5 / (2/3)").unwrap(), "15/2");
    // Simplification
    assert_eq!(interpret("(2/3) / (4/9)").unwrap(), "3/2");
  }
}

mod round_with_step {
  use super::*;

  #[test]
  fn round_real_step_gives_real() {
    assert_eq!(interpret("Round[0.04, 0.1]").unwrap(), "0.");
  }

  #[test]
  fn round_pi_real_step() {
    assert_eq!(interpret("Round[Pi, .5]").unwrap(), "3.");
  }

  #[test]
  fn round_rational_step() {
    assert_eq!(interpret("Round[2.6, 1/3]").unwrap(), "8/3");
  }

  #[test]
  fn round_symbolic_step() {
    assert_eq!(interpret("Round[10, Pi]").unwrap(), "3*Pi");
  }
}

// Rounding a real whose magnitude exceeds the i128 range must produce an exact
// BigInteger (the float's exact integer value), not saturate to i128::MAX.
mod round_large_reals {
  use super::*;

  const TEN_POW_40: &str = "10000000000000000303786028427003666890752";

  #[test]
  fn round_one_arg() {
    assert_eq!(interpret("Round[10.0^40]").unwrap(), TEN_POW_40);
    assert_eq!(
      interpret("Round[-1.0*^40]").unwrap(),
      "-10000000000000000303786028427003666890752"
    );
  }

  #[test]
  fn round_integer_step() {
    assert_eq!(interpret("Round[10.0^40, 1]").unwrap(), TEN_POW_40);
    assert_eq!(interpret("Round[1.0*^40, 2]").unwrap(), TEN_POW_40);
  }

  #[test]
  fn floor_ceiling_integer_part() {
    assert_eq!(interpret("Floor[10.0^40]").unwrap(), TEN_POW_40);
    assert_eq!(interpret("Ceiling[10.0^40]").unwrap(), TEN_POW_40);
    assert_eq!(interpret("IntegerPart[10.0^40]").unwrap(), TEN_POW_40);
  }

  // Floor/Ceiling/Round/IntegerPart applied to an already-integer-valued
  // argument (one of the same four, single-arg) is idempotent and returns the
  // inner expression, matching wolframscript.
  #[test]
  fn nested_rounding_idempotent() {
    assert_eq!(interpret("Floor[Floor[x]]").unwrap(), "Floor[x]");
    assert_eq!(interpret("Ceiling[Ceiling[x]]").unwrap(), "Ceiling[x]");
    assert_eq!(interpret("Round[Round[x]]").unwrap(), "Round[x]");
    assert_eq!(
      interpret("IntegerPart[IntegerPart[x]]").unwrap(),
      "IntegerPart[x]"
    );
    assert_eq!(interpret("Floor[Ceiling[x]]").unwrap(), "Ceiling[x]");
    assert_eq!(interpret("Ceiling[Floor[x]]").unwrap(), "Floor[x]");
    assert_eq!(interpret("Round[Floor[x]]").unwrap(), "Floor[x]");
    assert_eq!(interpret("Floor[Round[x]]").unwrap(), "Round[x]");
    assert_eq!(
      interpret("Floor[IntegerPart[x]]").unwrap(),
      "IntegerPart[x]"
    );
    // The two-argument inner form is not integer-valued, so it stays nested.
    assert_eq!(
      interpret("Floor[Floor[x, 2]]").unwrap(),
      "Floor[Floor[x, 2]]"
    );
  }
}

mod logistic_sigmoid {
  use super::*;

  #[test]
  fn logistic_sigmoid_real() {
    assert_eq!(
      interpret("LogisticSigmoid[0.5]").unwrap(),
      "0.6224593312018546"
    );
  }

  #[test]
  fn logistic_sigmoid_zero() {
    assert_eq!(interpret("LogisticSigmoid[0]").unwrap(), "1/2");
  }

  // The ±Infinity limits are 1 and 0. Exact non-zero arguments (integers,
  // rationals, I, symbols) stay symbolic and are NOT numericized — only Real
  // (and inexact-complex) inputs evaluate numerically. Per wolframscript.
  #[test]
  fn logistic_sigmoid_infinity_and_exact() {
    assert_eq!(interpret("LogisticSigmoid[Infinity]").unwrap(), "1");
    assert_eq!(interpret("LogisticSigmoid[-Infinity]").unwrap(), "0");
    assert_eq!(
      interpret("LogisticSigmoid[2]").unwrap(),
      "LogisticSigmoid[2]"
    );
    assert_eq!(
      interpret("LogisticSigmoid[-2]").unwrap(),
      "LogisticSigmoid[-2]"
    );
    assert_eq!(
      interpret("LogisticSigmoid[I]").unwrap(),
      "LogisticSigmoid[I]"
    );
  }
}

mod expand_threading {
  use super::*;

  #[test]
  fn expand_over_list() {
    assert_eq!(
      interpret("Expand[{4 (x + y), 2 (x + y) -> 4 (x + y)}]").unwrap(),
      "{4*x + 4*y, 2*x + 2*y -> 4*x + 4*y}"
    );
  }

  #[test]
  fn from_digits_symbolic() {
    assert_eq!(
      interpret("FromDigits[{a, b, c}, 5]").unwrap(),
      "5*(5*a + b) + c"
    );
  }

  #[test]
  fn binomial_real() {
    let result = interpret("Binomial[10.5, 3.2]").unwrap();
    assert!(result.starts_with("165.286"));
  }

  #[test]
  fn binomial_large_result_uses_bigint() {
    // Regression: small arguments can still produce a result far beyond i128
    // (Binomial[1000, 500] has 300 digits). It must not overflow/panic.
    assert_eq!(
      interpret("IntegerLength[Binomial[1000, 500]]").unwrap(),
      "300"
    );
    assert_eq!(interpret("Binomial[50, 25]").unwrap(), "126410606437752");
    // Exact 300-digit value matching wolframscript.
    assert_eq!(
      interpret("Binomial[1000, 500]").unwrap(),
      "270288240945436569515614693625975275496152008446548287007392875\
106625428705522193898612483924502370165362606085021546104802209750\
050679917549894219699518475423665484263751733356162464079737887344\
364574161119497604571044985756287880514600994219426752366915856603\
136862602484428109296905863799821216320"
    );
  }

  #[test]
  fn multinomial_large_result_uses_bigint() {
    // Multinomial accumulates products of binomials; a large case must also
    // use BigInt rather than overflowing i128.
    assert_eq!(interpret("Multinomial[2, 3, 4]").unwrap(), "1260");
    assert_eq!(
      interpret("IntegerLength[Multinomial[100, 100, 100]]").unwrap(),
      "141"
    );
  }

  #[test]
  fn string_repeat_max_length() {
    assert_eq!(
      interpret("StringRepeat[\"abc\", 10, 7]").unwrap(),
      "abcabca"
    );
  }

  #[test]
  fn euclidean_distance_scalar() {
    assert_eq!(interpret("EuclideanDistance[-7, 5]").unwrap(), "12");
  }

  #[test]
  fn euclidean_distance_vector() {
    assert_eq!(
      interpret("EuclideanDistance[{-1, -1}, {1, 1}]").unwrap(),
      "2*Sqrt[2]"
    );
  }

  #[test]
  fn manhattan_distance_scalar() {
    assert_eq!(interpret("ManhattanDistance[-7, 5]").unwrap(), "12");
  }

  #[test]
  fn manhattan_distance_vector() {
    assert_eq!(interpret("ManhattanDistance[{1, 2}, {3, 4}]").unwrap(), "4");
  }

  #[test]
  fn chessboard_distance_scalar() {
    assert_eq!(interpret("ChessboardDistance[-7, 5]").unwrap(), "12");
  }

  #[test]
  fn chessboard_distance_vector() {
    assert_eq!(
      interpret("ChessboardDistance[{-1, -1}, {1, 1}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn chessboard_distance_symbolic() {
    assert_eq!(
      interpret("ChessboardDistance[{a, b, c}, {x, y, z}]").unwrap(),
      "Max[Abs[a - x], Abs[b - y], Abs[c - z]]"
    );
  }

  #[test]
  fn bray_curtis_distance_scalar() {
    assert_eq!(interpret("BrayCurtisDistance[-7, 5]").unwrap(), "6");
  }

  #[test]
  fn bray_curtis_distance_vector() {
    assert_eq!(
      interpret("BrayCurtisDistance[{-1, -1}, {10, 10}]").unwrap(),
      "11/9"
    );
  }

  #[test]
  fn canberra_distance_scalar() {
    assert_eq!(interpret("CanberraDistance[-7, 5]").unwrap(), "1");
  }

  #[test]
  fn canberra_distance_vector() {
    assert_eq!(
      interpret("CanberraDistance[{-1, -1}, {1, 1}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn warping_distance() {
    // Dynamic time warping with |a - b| local cost; result is a Real.
    assert_eq!(
      interpret("WarpingDistance[{1, 2, 3}, {1, 2, 3}]").unwrap(),
      "0."
    );
    assert_eq!(
      interpret("WarpingDistance[{1, 2, 3}, {2, 3, 4}]").unwrap(),
      "2."
    );
    // Warping stretches a value over several positions for free.
    assert_eq!(
      interpret("WarpingDistance[{0, 1, 2}, {0, 0, 1, 2}]").unwrap(),
      "0."
    );
    assert_eq!(
      interpret("WarpingDistance[{1, 2, 3, 4}, {1, 4}]").unwrap(),
      "2."
    );
  }

  #[test]
  fn warping_distance_non_numeric_warns() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout("WarpingDistance[{1, 2}, {x, 3}]").unwrap();
    assert_eq!(r.result, "WarpingDistance[{1, 2}, {x, 3}]");
    assert!(r.warnings[0].contains(
      "WarpingDistance::invarg: Expecting a real-valued numeric or Boolean \
       vector or matrix instead of {x, 3}."
    ));
  }

  #[test]
  fn string_functions_conditional_patterns() {
    // A `/;`-conditioned string pattern (`x_ /; test`) must evaluate its test
    // per candidate match. These all returned no-match before the fix.
    assert_eq!(
      interpret(
        "StringReplace[\"aAbB\", x_ /; LowerCaseQ[x] :> ToUpperCase[x]]"
      )
      .unwrap(),
      "AABB"
    );
    assert_eq!(
      interpret("StringReplace[\"aAbB\", x_ /; LowerCaseQ[x] :> \"_\"]")
        .unwrap(),
      "_A_B"
    );
    assert_eq!(
      interpret("StringCases[\"aAbB\", x_ /; LowerCaseQ[x]]").unwrap(),
      "{a, b}"
    );
    assert_eq!(
      interpret("StringCount[\"aAbB\", x_ /; LowerCaseQ[x]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("StringPosition[\"a1b2c3\", d_ /; DigitQ[d]]").unwrap(),
      "{{2, 2}, {4, 4}, {6, 6}}"
    );
    assert_eq!(
      interpret("StringSplit[\"aXbXc\", x_ /; UpperCaseQ[x]]").unwrap(),
      "{a, b, c}"
    );
    // The conditional limit form still honours the max-replacements count.
    assert_eq!(
      interpret(
        "StringReplace[\"aAbB\", x_ /; LowerCaseQ[x] :> ToUpperCase[x], 1]"
      )
      .unwrap(),
      "AAbB"
    );
  }

  #[test]
  fn string_replace_with_limit() {
    assert_eq!(
      interpret("StringReplace[\"xyxyxyyyxxxyyxy\", \"xy\" -> \"A\", 2]")
        .unwrap(),
      "AAxyyyxxxyyxy"
    );
  }

  #[test]
  fn string_replace_rule_list_with_limit() {
    assert_eq!(
      interpret("StringReplace[\"abba\", {\"a\" -> \"A\", \"b\" -> \"B\"}, 2]")
        .unwrap(),
      "ABba"
    );
  }

  #[test]
  fn string_replace_list_of_strings() {
    assert_eq!(
      interpret(
        "StringReplace[{\"xyxyxxy\", \"yxyxyxxxyyxy\"}, \"xy\" -> \"A\"]"
      )
      .unwrap(),
      "{AAxA, yAAxxAyA}"
    );
  }

  #[test]
  fn squared_euclidean_distance_scalar() {
    assert_eq!(interpret("SquaredEuclideanDistance[-7, 5]").unwrap(), "144");
  }

  #[test]
  fn squared_euclidean_distance_vector() {
    assert_eq!(
      interpret("SquaredEuclideanDistance[{-1, -1}, {1, 1}]").unwrap(),
      "8"
    );
  }

  #[test]
  fn factorial2_negative_odd() {
    assert_eq!(interpret("Factorial2[-1]").unwrap(), "1");
    assert_eq!(interpret("Factorial2[-3]").unwrap(), "-1");
    assert_eq!(interpret("Factorial2[-5]").unwrap(), "1/3");
    assert_eq!(interpret("Factorial2[-7]").unwrap(), "-1/15");
  }

  // Factorial2 should render with the `!!` suffix (not as `Factorial2[x]`),
  // matching wolframscript. Parentheses wrap Plus/Times operands so the
  // suffix binds to the whole expression.
  #[test]
  fn factorial2_renders_with_double_bang_suffix() {
    assert_eq!(interpret("Factorial2[x]").unwrap(), "x!!");
    assert_eq!(interpret("I!! + 1").unwrap(), "1 + I!!");
    assert_eq!(interpret("(a + b)!!").unwrap(), "(a + b)!!");
  }

  // Real-valued Factorial2 uses the analytic continuation
  // `x!! = 2^(x/2 + (1 - Cos[Pi x])/4) Gamma[x/2 + 1] / Pi^((1 - Cos[Pi x])/4)`
  // so non-integer inputs now evaluate numerically instead of staying symbolic.
  #[test]
  fn factorial2_real_non_integer() {
    // 3.14!! ≈ 3.3477, to 6 significant figures.
    let s = interpret("Factorial2[3.14]").unwrap();
    assert!(s.starts_with("3.347742585544"), "got {}", s);
  }

  #[test]
  fn factorial2_real_integer() {
    // 3.0!! should evaluate, even if subject to 1-ULP error.
    let s = interpret("Factorial2[3.0]").unwrap();
    assert!(s.starts_with("3.0"), "got {}", s);
  }

  #[test]
  fn cases_except_two_arg() {
    assert_eq!(
      interpret("Cases[{a, 0, b, 1, c, 2, 3}, Except[1, _Integer]]").unwrap(),
      "{0, 2, 3}"
    );
  }

  #[test]
  fn delete_cases_symbol() {
    assert_eq!(
      interpret("DeleteCases[{a, b, 1, c, 2, 3}, _Symbol]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn delete_cases_alternatives() {
    assert_eq!(
      interpret("DeleteCases[{a, 1, 2.5, \"string\"}, _Integer|_Real]")
        .unwrap(),
      "{a, string}"
    );
  }

  #[test]
  fn string_insert_list_positions() {
    assert_eq!(
      interpret("StringInsert[\"adac\", \"he\", {1, 5}]").unwrap(),
      "headache"
    );
  }

  #[test]
  fn string_insert_list_strings() {
    assert_eq!(
      interpret("StringInsert[{\"something\", \"sometimes\"}, \" \", 5]")
        .unwrap(),
      "{some thing, some times}"
    );
  }

  #[test]
  fn norm_symbolic_vector() {
    assert_eq!(
      interpret("Norm[{x, y, z}]").unwrap(),
      "Sqrt[Abs[x]^2 + Abs[y]^2 + Abs[z]^2]"
    );
  }

  #[test]
  fn norm_infinity_symbolic() {
    assert_eq!(
      interpret("Norm[{x, y, z}, Infinity]").unwrap(),
      "Max[Abs[x], Abs[y], Abs[z]]"
    );
  }

  #[test]
  fn norm_infinity_numeric() {
    assert_eq!(interpret("Norm[{-100, 2, 3, 4}, Infinity]").unwrap(), "100");
  }

  #[test]
  fn norm_complex_scalar() {
    assert_eq!(interpret("Norm[1 + I]").unwrap(), "Sqrt[2]");
  }

  #[test]
  fn norm_exact_real_scalar_kept_exact() {
    // Real numeric scalars give an exact Abs, not a machine float.
    assert_eq!(interpret("Norm[Pi]").unwrap(), "Pi");
    assert_eq!(interpret("Norm[-Pi]").unwrap(), "Pi");
    assert_eq!(interpret("Norm[2/3]").unwrap(), "2/3");
    assert_eq!(interpret("Norm[-2/3]").unwrap(), "2/3");
    assert_eq!(interpret("Norm[Sqrt[2]]").unwrap(), "Sqrt[2]");
  }

  #[test]
  fn norm_symbolic_scalar_unevaluated() {
    // A non-numeric scalar leaves Norm unevaluated (real-ness unknown).
    assert_eq!(interpret("Norm[x]").unwrap(), "Norm[x]");
    assert_eq!(interpret("Norm[a]").unwrap(), "Norm[a]");
    assert_eq!(interpret("Norm[a + b I]").unwrap(), "Norm[a + I*b]");
  }

  #[test]
  fn norm_numeric_complex_scalar_evaluates() {
    assert_eq!(interpret("Norm[3 + 4 I]").unwrap(), "5");
    assert_eq!(interpret("Norm[2 - 3 I]").unwrap(), "Sqrt[13]");
    assert_eq!(interpret("Norm[3 I]").unwrap(), "3");
  }

  #[test]
  fn norm_symbolic_list_with_sin() {
    // Known-real scalar elements (integers, Sin[k] for integer k, etc.)
    // should stay exact/symbolic — not collapse to a machine float.
    assert_eq!(
      interpret("myFunction[x_] := x Sin[x]; Norm[Array[myFunction, 5]]")
        .unwrap(),
      "Sqrt[Sin[1]^2 + 4*Sin[2]^2 + 9*Sin[3]^2 + 16*Sin[4]^2 + 25*Sin[5]^2]"
    );
  }

  #[test]
  fn norm_exact_integer_list() {
    assert_eq!(interpret("Norm[{1, 2, 3}]").unwrap(), "Sqrt[14]");
  }

  #[test]
  fn norm_list_with_pi() {
    assert_eq!(interpret("Norm[{1, 2, Pi}]").unwrap(), "Sqrt[5 + Pi^2]");
  }

  #[test]
  fn norm_list_with_real_is_numeric() {
    assert_eq!(
      interpret("Norm[{1.0, 2, 3}]").unwrap(),
      "3.7416573867739413"
    );
  }

  #[test]
  fn norm_vector_general_p_integer() {
    // General p-norm: (Sum Abs[x]^p)^(1/p). Numeric inputs collapse.
    assert_eq!(interpret("Norm[{3, 4}, 4]").unwrap(), "337^(1/4)");
    assert_eq!(interpret("Norm[{3, 4}, 3]").unwrap(), "91^(1/3)");
  }

  #[test]
  fn norm_vector_general_p_symbolic_entries() {
    assert_eq!(
      interpret("Norm[{a, b}, 3]").unwrap(),
      "(Abs[a]^3 + Abs[b]^3)^(1/3)"
    );
  }

  #[test]
  fn norm_vector_symbolic_p() {
    assert_eq!(
      interpret("Norm[{a, b, c}, p]").unwrap(),
      "(Abs[a]^p + Abs[b]^p + Abs[c]^p)^p^(-1)"
    );
  }

  #[test]
  fn norm_vector_general_p_real() {
    assert_eq!(
      interpret("Norm[{1.0, 2.0}, 3]").unwrap(),
      "2.080083823051904"
    );
  }

  #[test]
  fn norm_matrix_spectral() {
    // Default matrix norm is the spectral norm (largest singular value).
    assert_eq!(
      interpret("Norm[{{1, 2}, {3, 4}}]").unwrap(),
      "Sqrt[15 + Sqrt[221]]"
    );
    assert_eq!(interpret("Norm[{{3, 0}, {0, 4}}]").unwrap(), "4");
  }

  #[test]
  fn norm_matrix_one_is_max_column_sum() {
    assert_eq!(interpret("Norm[{{1, 2}, {3, 4}}, 1]").unwrap(), "6");
    assert_eq!(interpret("Norm[{{1, -2}, {-3, 4}}, 1]").unwrap(), "6");
  }

  #[test]
  fn norm_matrix_infinity_is_max_row_sum() {
    assert_eq!(interpret("Norm[{{1, 2}, {3, 4}}, Infinity]").unwrap(), "7");
  }

  #[test]
  fn norm_matrix_frobenius() {
    assert_eq!(
      interpret(r#"Norm[{{1, 2}, {3, 4}}, "Frobenius"]"#).unwrap(),
      "Sqrt[30]"
    );
  }

  #[test]
  fn string_replace_operator_form() {
    assert_eq!(
      interpret("StringReplace[\"y\" -> \"ies\"][\"city\"]").unwrap(),
      "cities"
    );
  }

  #[test]
  fn ceiling_complex() {
    assert_eq!(interpret("Ceiling[1.3 + 0.7 I]").unwrap(), "2 + I");
  }

  #[test]
  fn floor_complex() {
    assert_eq!(interpret("Floor[1.5 + 2.7 I]").unwrap(), "1 + 2*I");
  }

  #[test]
  fn integer_part_complex() {
    // IntegerPart applies componentwise, truncating toward zero, and always
    // yields integer components regardless of input exactness.
    assert_eq!(interpret("IntegerPart[2.5 + 3.5 I]").unwrap(), "2 + 3*I");
    assert_eq!(interpret("IntegerPart[-2.5 - 3.5 I]").unwrap(), "-2 - 3*I");
    assert_eq!(interpret("IntegerPart[5/2 + 7/2 I]").unwrap(), "2 + 3*I");
    assert_eq!(interpret("IntegerPart[2 + 3 I]").unwrap(), "2 + 3*I");
  }

  #[test]
  fn fractional_part_complex() {
    // Exact rational complex keeps exact rational components.
    assert_eq!(
      interpret("FractionalPart[5/2 + 3/2 I]").unwrap(),
      "1/2 + I/2"
    );
    assert_eq!(interpret("FractionalPart[2 + 3 I]").unwrap(), "0");
    // Inexact complex: forming a complex from a Real promotes both parts, so
    // an integer-valued component prints as `0.`.
    assert_eq!(
      interpret("FractionalPart[1.5 + 2.7 I]").unwrap(),
      "0.5 + 0.7000000000000002*I"
    );
    assert_eq!(
      interpret("FractionalPart[2 + 2.5 I]").unwrap(),
      "0. + 0.5*I"
    );
    assert_eq!(
      interpret("FractionalPart[1.5 + 3 I]").unwrap(),
      "0.5 + 0.*I"
    );
  }

  #[test]
  fn rounding_symbolic_complex() {
    // Floor/Ceiling/Round/IntegerPart of a complex with symbolic-real parts
    // (Pi + E I, Sqrt[2] + Sqrt[3] I) apply componentwise.
    assert_eq!(interpret("Floor[Pi + E I]").unwrap(), "3 + 2*I");
    assert_eq!(interpret("Ceiling[Pi + E I]").unwrap(), "4 + 3*I");
    assert_eq!(interpret("Round[Pi + E I]").unwrap(), "3 + 3*I");
    assert_eq!(interpret("IntegerPart[Pi + E I]").unwrap(), "3 + 2*I");
    assert_eq!(interpret("Floor[Sqrt[2] + Sqrt[3] I]").unwrap(), "1 + I");
    assert_eq!(interpret("Floor[E I]").unwrap(), "2*I");
    // Fully symbolic (unknown sign) stays unevaluated, like wolframscript.
    assert_eq!(interpret("Floor[x + y I]").unwrap(), "Floor[x + I*y]");
  }

  #[test]
  fn rounding_of_infinity() {
    // Floor/Ceiling/Round/IntegerPart leave (un)signed infinities unchanged.
    assert_eq!(interpret("Floor[Infinity]").unwrap(), "Infinity");
    assert_eq!(interpret("Ceiling[Infinity]").unwrap(), "Infinity");
    assert_eq!(interpret("Round[Infinity]").unwrap(), "Infinity");
    assert_eq!(interpret("IntegerPart[Infinity]").unwrap(), "Infinity");

    assert_eq!(interpret("Floor[-Infinity]").unwrap(), "-Infinity");
    assert_eq!(interpret("Ceiling[-Infinity]").unwrap(), "-Infinity");
    assert_eq!(interpret("Round[-Infinity]").unwrap(), "-Infinity");
    assert_eq!(interpret("IntegerPart[-Infinity]").unwrap(), "-Infinity");

    assert_eq!(
      interpret("Floor[ComplexInfinity]").unwrap(),
      "ComplexInfinity"
    );
    assert_eq!(
      interpret("Round[ComplexInfinity]").unwrap(),
      "ComplexInfinity"
    );
    assert_eq!(
      interpret("IntegerPart[ComplexInfinity]").unwrap(),
      "ComplexInfinity"
    );
  }

  #[test]
  fn rounding_of_infinity_two_arg() {
    // The two-argument forms also pass the infinity through.
    assert_eq!(interpret("Floor[Infinity, 2]").unwrap(), "Infinity");
    assert_eq!(interpret("Ceiling[-Infinity, 2]").unwrap(), "-Infinity");
    assert_eq!(interpret("Round[Infinity, 5]").unwrap(), "Infinity");
  }

  #[test]
  fn fractional_part_of_infinity() {
    // FractionalPart of an infinity is the full unit interval in that
    // direction.
    assert_eq!(
      interpret("FractionalPart[Infinity]").unwrap(),
      "Interval[{0, 1}]"
    );
    assert_eq!(
      interpret("FractionalPart[-Infinity]").unwrap(),
      "Interval[{-1, 0}]"
    );
    assert_eq!(
      interpret("FractionalPart[ComplexInfinity]").unwrap(),
      "Interval[{0, 1}]"
    );
    // 2 Infinity collapses to Infinity first.
    assert_eq!(
      interpret("FractionalPart[2 Infinity]").unwrap(),
      "Interval[{0, 1}]"
    );
  }

  #[test]
  fn rounding_of_indeterminate() {
    // Floor/Ceiling/Round/IntegerPart/FractionalPart of Indeterminate stay
    // Indeterminate.
    assert_eq!(interpret("Floor[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(
      interpret("Ceiling[Indeterminate]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(interpret("Round[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(
      interpret("IntegerPart[Indeterminate]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("FractionalPart[Indeterminate]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn integer_length_negative_base() {
    assert_eq!(
      interpret("IntegerLength[3, -2]").unwrap(),
      "IntegerLength[3, -2]"
    );
  }

  #[test]
  fn power_mod_zero_modulus() {
    assert_eq!(interpret("PowerMod[5, 2, 0]").unwrap(), "PowerMod[5, 2, 0]");
  }

  #[test]
  fn power_mod_non_invertible() {
    assert_eq!(
      interpret("PowerMod[0, -1, 2]").unwrap(),
      "PowerMod[0, -1, 2]"
    );
  }

  // These functions are Listable and thread over any list argument.
  #[test]
  fn power_mod_threads_over_lists() {
    assert_eq!(interpret("PowerMod[{2, 3}, 2, 5]").unwrap(), "{4, 4}");
    assert_eq!(interpret("PowerMod[2, {2, 3}, 5]").unwrap(), "{4, 3}");
    assert_eq!(interpret("PowerMod[{2, 3}, {2, 3}, 5]").unwrap(), "{4, 2}");
  }

  #[test]
  fn bit_set_clear_thread_over_lists() {
    assert_eq!(interpret("BitSet[{4, 8}, 1]").unwrap(), "{6, 10}");
    assert_eq!(interpret("BitClear[{15, 7}, 0]").unwrap(), "{14, 6}");
  }

  #[test]
  fn jacobi_symbol_threads_over_lists() {
    assert_eq!(interpret("JacobiSymbol[{3, 5}, 7]").unwrap(), "{-1, -1}");
    assert_eq!(interpret("JacobiSymbol[3, {7, 11}]").unwrap(), "{-1, 1}");
  }

  #[test]
  fn integer_exponent_threads_over_lists() {
    assert_eq!(
      interpret("IntegerExponent[{100, 1000}, 10]").unwrap(),
      "{2, 3}"
    );
    assert_eq!(
      interpret("IntegerExponent[1000, {2, 5}]").unwrap(),
      "{3, 3}"
    );
  }

  #[test]
  fn newly_listable_functions_report_attributes() {
    for f in [
      "PowerMod",
      "BitSet",
      "BitClear",
      "JacobiSymbol",
      "IntegerExponent",
    ] {
      assert_eq!(
        interpret(&format!("Attributes[{f}]")).unwrap(),
        "{Listable, Protected}",
        "{f} should be Listable and Protected"
      );
    }
  }

  #[test]
  fn mod_zero_modulus() {
    assert_eq!(interpret("Mod[5, 0]").unwrap(), "Indeterminate");
  }

  #[test]
  fn mod_real_operand_keeps_real_result() {
    // When an operand is an inexact machine real, Mod returns a Real even for a
    // whole-number result (regression: Woxi returned Integer 0).
    assert_eq!(interpret("Mod[1.5, 0.5]").unwrap(), "0.");
    assert_eq!(interpret("Mod[10.0, 5.0]").unwrap(), "0.");
    assert_eq!(interpret("Mod[5, 2.0]").unwrap(), "1.");
    assert_eq!(interpret("Mod[6.0, 3]").unwrap(), "0.");
    assert_eq!(interpret("Mod[1.0, 1.0]").unwrap(), "0.");
    // Non-whole real results are unaffected.
    assert_eq!(interpret("Mod[5.5, 2]").unwrap(), "1.5");
    assert_eq!(interpret("Mod[-5.5, 2]").unwrap(), "0.5");
    // 3-argument form behaves the same.
    assert_eq!(interpret("Mod[6.0, 3, 0]").unwrap(), "0.");
    assert_eq!(interpret("Mod[10, 3.0, 1]").unwrap(), "1.");
    // Pure-integer Mod still yields an Integer.
    assert_eq!(interpret("Mod[17, 5]").unwrap(), "2");
    assert_eq!(interpret("Mod[6, 3]").unwrap(), "0");
  }

  #[test]
  fn coth_zero() {
    assert_eq!(interpret("Coth[0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn arccosh_zero() {
    assert_eq!(interpret("ArcCosh[0]").unwrap(), "I/2*Pi");
  }

  #[test]
  fn arccosh_one() {
    assert_eq!(interpret("ArcCosh[1]").unwrap(), "0");
  }

  #[test]
  fn arccoth_zero() {
    assert_eq!(interpret("ArcCoth[0]").unwrap(), "I/2*Pi");
  }

  #[test]
  fn arccoth_one() {
    assert_eq!(interpret("ArcCoth[1]").unwrap(), "Infinity");
  }

  #[test]
  fn arcsech_zero() {
    assert_eq!(interpret("ArcSech[0]").unwrap(), "Infinity");
  }

  #[test]
  fn arcsech_one() {
    assert_eq!(interpret("ArcSech[1]").unwrap(), "0");
  }

  #[test]
  fn arcsech_half() {
    // ArcSech[0.5] = ArcCosh[2] ≈ 1.3169578969248166. The last ULP is
    // platform-dependent (system libm differs across OSes; Linux CI gives
    // ...168), so compare numerically rather than by exact string.
    let val: f64 = interpret("ArcSech[0.5]").unwrap().parse().unwrap();
    assert!((val - 1.3169578969248166).abs() < 1e-12);
  }

  #[test]
  fn arccoth_half() {
    assert_eq!(
      interpret("ArcCoth[0.5]").unwrap(),
      "0.5493061443340549 - 1.5707963267948966*I"
    );
  }

  #[test]
  fn arccos_half() {
    assert_eq!(interpret("ArcCos[1/2]").unwrap(), "Pi/3");
  }

  #[test]
  fn arccos_neg_half() {
    assert_eq!(interpret("ArcCos[-1/2]").unwrap(), "(2*Pi)/3");
  }

  #[test]
  fn arccos_sqrt2_over_2() {
    assert_eq!(interpret("ArcCos[Sqrt[2]/2]").unwrap(), "Pi/4");
  }

  // Regression (mathics test_trig.py:20): `ArcCos[1/2 Sqrt[2]]` is the
  // same value as `ArcCos[Sqrt[2]/2]`. mathics expects `1/4 Pi` (its
  // own rendering of `Times[Rational[1,4], Pi]`); Woxi (and
  // wolframscript) render it as `Pi/4`.
  #[test]
  fn arccos_half_sqrt2() {
    assert_eq!(interpret("ArcCos[1/2 Sqrt[2]]").unwrap(), "Pi/4");
  }

  // Regression (mathics test_trig.py:21): the negative variant.
  #[test]
  fn arccos_neg_half_sqrt2() {
    assert_eq!(interpret("ArcCos[-1/2 Sqrt[2]]").unwrap(), "(3*Pi)/4");
  }

  // Regression (mathics test_trig.py:22): `ArcCos[1/2 Sqrt[3]]` → Pi/6.
  #[test]
  fn arccos_half_sqrt3() {
    assert_eq!(interpret("ArcCos[1/2 Sqrt[3]]").unwrap(), "Pi/6");
  }

  // Regression (mathics test_trig.py:23): the negative variant.
  #[test]
  fn arccos_neg_half_sqrt3() {
    assert_eq!(interpret("ArcCos[-1/2 Sqrt[3]]").unwrap(), "(5*Pi)/6");
  }

  // Regression (mathics test_basic.py:48): `E^(I Pi/4)` stays in
  // symbolic exponent form rather than being expanded to roots of
  // unity. mathics's expected `E ^ (I / 4 Pi)` is the same value
  // rendered with a different surface form.
  #[test]
  fn e_to_imaginary_pi_quarter() {
    assert_eq!(interpret("E^(I Pi/4)").unwrap(), "E^(I/4*Pi)");
  }

  #[test]
  fn arccos_neg_sqrt2_over_2() {
    assert_eq!(interpret("ArcCos[-Sqrt[2]/2]").unwrap(), "(3*Pi)/4");
  }

  #[test]
  fn arccos_sqrt3_over_2() {
    assert_eq!(interpret("ArcCos[Sqrt[3]/2]").unwrap(), "Pi/6");
  }

  // ArcCos[±I·Infinity] → DirectedInfinity[∓I] (matches wolframscript).
  // Regression for mathics test_trig.py:15.
  #[test]
  fn arccos_imaginary_infinity_positive() {
    assert_eq!(
      interpret("ArcCos[I Infinity]").unwrap(),
      "DirectedInfinity[-I]"
    );
  }

  #[test]
  fn arccos_imaginary_infinity_negative() {
    assert_eq!(
      interpret("ArcCos[-I Infinity]").unwrap(),
      "DirectedInfinity[I]"
    );
  }

  #[test]
  fn arcsin_half() {
    assert_eq!(interpret("ArcSin[1/2]").unwrap(), "Pi/6");
  }

  #[test]
  fn arcsin_neg_half() {
    assert_eq!(interpret("ArcSin[-1/2]").unwrap(), "-1/6*Pi");
  }

  #[test]
  fn arcsin_sqrt2_over_2() {
    assert_eq!(interpret("ArcSin[Sqrt[2]/2]").unwrap(), "Pi/4");
  }

  #[test]
  fn arcsin_sqrt3_over_2() {
    assert_eq!(interpret("ArcSin[Sqrt[3]/2]").unwrap(), "Pi/3");
  }

  #[test]
  fn complex_iterated_i() {
    assert_eq!(interpret("Complex[1, Complex[0, 1]]").unwrap(), "0");
  }

  #[test]
  fn complex_iterated_one_plus_i() {
    assert_eq!(interpret("Complex[1, Complex[1, 1]]").unwrap(), "I");
  }

  #[test]
  fn complex_negative_imag() {
    assert_eq!(interpret("Complex[1, -2]").unwrap(), "1 - 2*I");
  }

  #[test]
  fn indeterminate_propagation() {
    assert_eq!(interpret("Sin[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Cos[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Tan[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Exp[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Log[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Sqrt[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Sinh[Indeterminate]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Cosh[Indeterminate]").unwrap(), "Indeterminate");
    // Arithmetic propagation
    assert_eq!(interpret("Indeterminate + 1").unwrap(), "Indeterminate");
    assert_eq!(interpret("Indeterminate * 5").unwrap(), "Indeterminate");
    assert_eq!(interpret("ComplexInfinity + 1").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("ComplexInfinity * 3").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("ComplexInfinity * 0").unwrap(), "Indeterminate");
  }

  #[test]
  fn complex_infinity_trig() {
    assert_eq!(interpret("Sin[ComplexInfinity]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Cos[ComplexInfinity]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Tan[ComplexInfinity]").unwrap(), "Indeterminate");
  }

  #[test]
  fn log_zero_real() {
    assert_eq!(interpret("Log[0.]").unwrap(), "Indeterminate");
  }

  #[test]
  fn log_negative_real() {
    assert_eq!(interpret("Log[-1.]").unwrap(), "0. + 3.141592653589793*I");
  }

  #[test]
  fn log_negative_integer() {
    assert_eq!(interpret("Log[-1]").unwrap(), "I*Pi");
    assert_eq!(interpret("Log[-2]").unwrap(), "I*Pi + Log[2]");
    assert_eq!(interpret("Log[-3]").unwrap(), "I*Pi + Log[3]");
  }

  #[test]
  fn log_negative_expr() {
    assert_eq!(interpret("Log[-E]").unwrap(), "1 + I*Pi");
  }

  #[test]
  fn log_negative_rational() {
    assert_eq!(interpret("Log[-5/2]").unwrap(), "I*Pi + Log[5/2]");
    assert_eq!(interpret("Log[-1/3]").unwrap(), "I*Pi - Log[3]");
  }

  #[test]
  fn log_pure_imaginary() {
    // Log[c I] = Log[Abs[c]] + Sign[c]*I*Pi/2 for exact rational c.
    assert_eq!(interpret("Log[2 I]").unwrap(), "I/2*Pi + Log[2]");
    assert_eq!(interpret("Log[3 I]").unwrap(), "I/2*Pi + Log[3]");
    assert_eq!(interpret("Log[-2 I]").unwrap(), "(-1/2*I)*Pi + Log[2]");
    assert_eq!(interpret("Log[I/2]").unwrap(), "I/2*Pi - Log[2]");
    assert_eq!(interpret("Log[-I/3]").unwrap(), "(-1/2*I)*Pi - Log[3]");
  }

  #[test]
  fn log_imaginary_symbolic_coefficient_unevaluated() {
    // A symbolic-real coefficient (Pi, Sqrt[2], a) stays unevaluated, as in
    // Wolfram, since Sign/Abs of it can't be determined here.
    assert_eq!(interpret("Log[Pi I]").unwrap(), "Log[I*Pi]");
    assert_eq!(interpret("Log[Sqrt[2] I]").unwrap(), "Log[I*Sqrt[2]]");
    assert_eq!(interpret("Log[a I]").unwrap(), "Log[I*a]");
  }

  #[test]
  fn sin_complex_float() {
    assert_eq!(
      interpret("Sin[1.0 + I]").unwrap(),
      "1.2984575814159773 + 0.6349639147847361*I"
    );
  }

  #[test]
  fn cos_complex_float() {
    assert_eq!(
      interpret("Cos[1.0 + I]").unwrap(),
      "0.8337300251311491 - 0.9888977057628651*I"
    );
  }

  #[test]
  fn conjugate_real_constants() {
    assert_eq!(interpret("Conjugate[Pi]").unwrap(), "Pi");
    assert_eq!(interpret("Conjugate[E]").unwrap(), "E");
    assert_eq!(
      interpret("{Conjugate[Pi], Conjugate[E]}").unwrap(),
      "{Pi, E}"
    );
  }

  // Conjugate of any real-valued (NumericQ) expression is the expression
  // itself, not just exact constants. Verified against wolframscript.
  #[test]
  fn conjugate_real_valued_expressions() {
    assert_eq!(interpret("Conjugate[Sqrt[2]]").unwrap(), "Sqrt[2]");
    assert_eq!(interpret("Conjugate[Pi^2]").unwrap(), "Pi^2");
    assert_eq!(interpret("Conjugate[Log[2]]").unwrap(), "Log[2]");
    assert_eq!(interpret("Conjugate[Sin[2]]").unwrap(), "Sin[2]");
    assert_eq!(interpret("Conjugate[2^(1/3)]").unwrap(), "2^(1/3)");
    assert_eq!(
      interpret("Conjugate[Sqrt[2] + Pi]").unwrap(),
      "Sqrt[2] + Pi"
    );
  }

  // Complex-valued expressions must still be conjugated, not treated as real.
  #[test]
  fn conjugate_complex_still_works() {
    assert_eq!(interpret("Conjugate[3 + 4 I]").unwrap(), "3 - 4*I");
    assert_eq!(interpret("Conjugate[I]").unwrap(), "-I");
    assert_eq!(interpret("Conjugate[Sqrt[-2]]").unwrap(), "-I*Sqrt[2]");
    assert_eq!(
      interpret("Conjugate[a + b I]").unwrap(),
      "Conjugate[a] - I*Conjugate[b]"
    );
  }

  // Re/Im of a real-valued (NumericQ) expression: the expression and 0.
  #[test]
  fn re_im_real_valued_expressions() {
    assert_eq!(interpret("Re[Sqrt[2]]").unwrap(), "Sqrt[2]");
    assert_eq!(interpret("Im[Sqrt[2]]").unwrap(), "0");
    assert_eq!(interpret("Re[Pi^2]").unwrap(), "Pi^2");
    assert_eq!(interpret("Im[Log[2]]").unwrap(), "0");
    assert_eq!(interpret("Re[Sqrt[2] + Pi]").unwrap(), "Sqrt[2] + Pi");
    assert_eq!(interpret("Im[Sqrt[2] + Pi]").unwrap(), "0");
  }

  // Re/Im still extract parts of genuine complex values.
  #[test]
  fn re_im_complex_still_works() {
    assert_eq!(interpret("Re[3 + 4 I]").unwrap(), "3");
    assert_eq!(interpret("Im[3 + 4 I]").unwrap(), "4");
    assert_eq!(interpret("Im[2 I]").unwrap(), "2");
    assert_eq!(interpret("Re[a]").unwrap(), "Re[a]");
  }

  // Re/Im pull real coefficients out of products: Re[c x] = c Re[x].
  #[test]
  fn re_im_pull_real_coefficient() {
    assert_eq!(interpret("Re[2 a]").unwrap(), "2*Re[a]");
    assert_eq!(interpret("Im[3 b]").unwrap(), "3*Im[b]");
    assert_eq!(interpret("Re[2 b I]").unwrap(), "-2*Im[b]");
  }

  // Re/Im split a symbolic Plus along its explicit-I terms; the non-I terms
  // stay grouped. Verified against wolframscript.
  #[test]
  fn re_im_distribute_over_plus() {
    assert_eq!(interpret("Re[a + b I]").unwrap(), "-Im[b] + Re[a]");
    assert_eq!(interpret("Im[a + b I]").unwrap(), "Im[a] + Re[b]");
    assert_eq!(interpret("Re[x + y + z I]").unwrap(), "-Im[z] + Re[x + y]");
    assert_eq!(interpret("Re[a - b I]").unwrap(), "Im[b] + Re[a]");
    assert_eq!(interpret("Re[2 a + 3 b I]").unwrap(), "-3*Im[b] + 2*Re[a]");
    assert_eq!(interpret("Im[a + b I + c]").unwrap(), "Im[a + c] + Re[b]");
  }

  // A plain Plus with no imaginary terms is not distributed.
  #[test]
  fn re_im_plain_plus_not_distributed() {
    assert_eq!(interpret("Re[a + b]").unwrap(), "Re[a + b]");
    assert_eq!(interpret("Im[a + b]").unwrap(), "Im[a + b]");
  }

  // Arg of a real-valued expression: 0 for positive, Pi for negative.
  #[test]
  fn arg_real_valued_expressions() {
    assert_eq!(interpret("Arg[Sqrt[2]]").unwrap(), "0");
    assert_eq!(interpret("Arg[Log[2]]").unwrap(), "0");
    assert_eq!(interpret("Arg[Pi^2]").unwrap(), "0");
    assert_eq!(interpret("Arg[Pi - 4]").unwrap(), "Pi");
    assert_eq!(interpret("Arg[-Sqrt[2]]").unwrap(), "Pi");
  }

  // Arg of a complex number with surd parts: Arg[z] = ArcTan[Re[z], Im[z]].
  #[test]
  fn arg_surd_complex() {
    assert_eq!(interpret("Arg[1 + I Sqrt[3]]").unwrap(), "Pi/3");
    assert_eq!(interpret("Arg[Sqrt[3] + I]").unwrap(), "Pi/6");
    assert_eq!(interpret("Arg[-Sqrt[3] + I]").unwrap(), "(5*Pi)/6");
    assert_eq!(interpret("Arg[I Sqrt[3]]").unwrap(), "Pi/2");
    assert_eq!(interpret("Arg[-1 - I Sqrt[3]]").unwrap(), "(-2*Pi)/3");
  }

  #[test]
  fn product_log_special_values() {
    assert_eq!(interpret("ProductLog[0]").unwrap(), "0");
    assert_eq!(interpret("ProductLog[E]").unwrap(), "1");
  }

  // ─── Trig with exact complex arguments ────────────────────────────

  #[test]
  fn sin_pure_imaginary() {
    assert_eq!(interpret("Sin[I]").unwrap(), "I*Sinh[1]");
    assert_eq!(interpret("Sin[2*I]").unwrap(), "I*Sinh[2]");
    assert_eq!(interpret("Sin[-I]").unwrap(), "-I*Sinh[1]");
  }

  #[test]
  fn cos_pure_imaginary() {
    assert_eq!(interpret("Cos[I]").unwrap(), "Cosh[1]");
    assert_eq!(interpret("Cos[2*I]").unwrap(), "Cosh[2]");
    assert_eq!(interpret("Cos[-I]").unwrap(), "Cosh[1]");
  }

  #[test]
  fn tan_pure_imaginary() {
    assert_eq!(interpret("Tan[I]").unwrap(), "I*Tanh[1]");
    assert_eq!(interpret("Tan[-I]").unwrap(), "-I*Tanh[1]");
  }

  #[test]
  fn sin_exact_complex_unevaluated() {
    // Non-Pi-fraction real part: stays unevaluated
    assert_eq!(interpret("Sin[1 + I]").unwrap(), "Sin[1 + I]");
    assert_eq!(interpret("Sin[1/2 + I]").unwrap(), "Sin[1/2 + I]");
  }

  #[test]
  fn sin_pi_fraction_complex() {
    // Pi-fraction real part: decomposes symbolically
    assert_eq!(interpret("Sin[Pi + I]").unwrap(), "-I*Sinh[1]");
  }

  // ─── Exact fifth/tenth-angle trig values ──────────────────────────
  #[test]
  fn cos_fifth_angles() {
    assert_eq!(interpret("Cos[Pi/5]").unwrap(), "(1 + Sqrt[5])/4");
    assert_eq!(interpret("Cos[2 Pi/5]").unwrap(), "(-1 + Sqrt[5])/4");
    assert_eq!(interpret("Cos[Pi/10]").unwrap(), "Sqrt[5/8 + Sqrt[5]/8]");
    assert_eq!(interpret("Cos[3 Pi/10]").unwrap(), "Sqrt[5/8 - Sqrt[5]/8]");
  }

  #[test]
  fn sin_fifth_angles() {
    assert_eq!(interpret("Sin[Pi/5]").unwrap(), "Sqrt[5/8 - Sqrt[5]/8]");
    assert_eq!(interpret("Sin[2 Pi/5]").unwrap(), "Sqrt[5/8 + Sqrt[5]/8]");
    assert_eq!(interpret("Sin[Pi/10]").unwrap(), "(-1 + Sqrt[5])/4");
    assert_eq!(interpret("Sin[3 Pi/10]").unwrap(), "(1 + Sqrt[5])/4");
  }

  #[test]
  fn sin_fifth_angles_second_quadrant() {
    // sin(Pi - x) = sin(x): these stay positive, so they match exactly.
    assert_eq!(interpret("Sin[3 Pi/5]").unwrap(), "Sqrt[5/8 + Sqrt[5]/8]");
    assert_eq!(interpret("Sin[4 Pi/5]").unwrap(), "Sqrt[5/8 - Sqrt[5]/8]");
    assert_eq!(interpret("Sin[7 Pi/10]").unwrap(), "(1 + Sqrt[5])/4");
    assert_eq!(interpret("Sin[9 Pi/10]").unwrap(), "(-1 + Sqrt[5])/4");
  }

  #[test]
  fn fifth_angle_gcd_reduction() {
    // 2*Pi/10 reduces to Pi/5.
    assert_eq!(interpret("Sin[2 Pi/10]").unwrap(), "Sqrt[5/8 - Sqrt[5]/8]");
    assert_eq!(interpret("Cos[2 Pi/10]").unwrap(), "(1 + Sqrt[5])/4");
  }

  // ─── First-octant canonicalization (no radical form) ──────────────
  // Wolfram folds Sin/Cos of k*Pi/n to the first octant [0, Pi/4] using
  // co-function identities, keeping the reference angle symbolic when there
  // is no nice radical form.
  #[test]
  fn eighth_angles_canonical() {
    // Pi/8 itself is already first-octant: stays unevaluated.
    assert_eq!(interpret("Sin[Pi/8]").unwrap(), "Sin[Pi/8]");
    assert_eq!(interpret("Cos[Pi/8]").unwrap(), "Cos[Pi/8]");
    // 3Pi/8 > Pi/4 → co-function of Pi/8.
    assert_eq!(interpret("Sin[3 Pi/8]").unwrap(), "Cos[Pi/8]");
    assert_eq!(interpret("Cos[3 Pi/8]").unwrap(), "Sin[Pi/8]");
    // Second/third quadrant signs.
    assert_eq!(interpret("Sin[5 Pi/8]").unwrap(), "Cos[Pi/8]");
    assert_eq!(interpret("Cos[5 Pi/8]").unwrap(), "-Sin[Pi/8]");
    assert_eq!(interpret("Sin[7 Pi/8]").unwrap(), "Sin[Pi/8]");
    assert_eq!(interpret("Cos[7 Pi/8]").unwrap(), "-Cos[Pi/8]");
  }

  #[test]
  fn octant_reduction_other_denominators() {
    assert_eq!(interpret("Sin[3 Pi/7]").unwrap(), "Cos[Pi/14]");
    assert_eq!(interpret("Cos[5 Pi/7]").unwrap(), "-Sin[(3*Pi)/14]");
    assert_eq!(interpret("Sin[5 Pi/16]").unwrap(), "Cos[(3*Pi)/16]");
    assert_eq!(interpret("Cos[7 Pi/16]").unwrap(), "Sin[Pi/16]");
    // Already first-octant: unchanged.
    assert_eq!(interpret("Sin[2 Pi/9]").unwrap(), "Sin[(2*Pi)/9]");
  }

  #[test]
  fn octant_reduction_negative_and_periodic() {
    assert_eq!(interpret("Sin[-3 Pi/8]").unwrap(), "-Cos[Pi/8]");
    assert_eq!(interpret("Sin[9 Pi/8]").unwrap(), "-Sin[Pi/8]");
    assert_eq!(interpret("Cos[9 Pi/8]").unwrap(), "-Cos[Pi/8]");
  }

  #[test]
  fn octant_reduction_full_period() {
    // Angles beyond 2*Pi reduce to the canonical first-octant form.
    assert_eq!(interpret("Sin[17 Pi/8]").unwrap(), "Sin[Pi/8]");
    assert_eq!(interpret("Cos[17 Pi/8]").unwrap(), "Cos[Pi/8]");
    assert_eq!(interpret("Tan[9 Pi/8]").unwrap(), "Tan[Pi/8]");
  }

  // ─── Tan/Cot/Sec/Csc first-octant canonicalization ────────────────
  #[test]
  fn tan_cot_octant_reduction() {
    // 3Pi/8 > Pi/4: Tan <-> Cot co-function.
    assert_eq!(interpret("Tan[3 Pi/8]").unwrap(), "Cot[Pi/8]");
    assert_eq!(interpret("Cot[3 Pi/8]").unwrap(), "Tan[Pi/8]");
    // Already first-octant: unchanged.
    assert_eq!(interpret("Tan[Pi/8]").unwrap(), "Tan[Pi/8]");
    assert_eq!(interpret("Cot[2 Pi/9]").unwrap(), "Cot[(2*Pi)/9]");
    // Other denominators / signs.
    assert_eq!(interpret("Tan[2 Pi/7]").unwrap(), "Cot[(3*Pi)/14]");
    assert_eq!(interpret("Tan[5 Pi/8]").unwrap(), "-Cot[Pi/8]");
  }

  #[test]
  fn sec_csc_octant_reduction() {
    // 3Pi/8 > Pi/4: Sec <-> Csc co-function.
    assert_eq!(interpret("Sec[3 Pi/8]").unwrap(), "Csc[Pi/8]");
    assert_eq!(interpret("Csc[3 Pi/8]").unwrap(), "Sec[Pi/8]");
    // Quadrant sign + co-function.
    assert_eq!(interpret("Sec[5 Pi/7]").unwrap(), "-Csc[(3*Pi)/14]");
    assert_eq!(interpret("Csc[5 Pi/16]").unwrap(), "Sec[(3*Pi)/16]");
    // Already first-octant: unchanged.
    assert_eq!(interpret("Sec[2 Pi/9]").unwrap(), "Sec[(2*Pi)/9]");
  }

  // ─── Sec/Csc exact values at twelfths ─────────────────────────────
  // Wolfram evaluates the twelfth-angle Sec/Csc to a radical (unlike the
  // eighth angles, which it leaves symbolic). These follow from
  //   Sin[Pi/12]  = (Sqrt[3]-1)/(2 Sqrt[2]),
  //   Sin[5Pi/12] = (Sqrt[3]+1)/(2 Sqrt[2]).
  #[test]
  fn sec_csc_twelfths_exact() {
    assert_eq!(interpret("Csc[Pi/12]").unwrap(), "Sqrt[2]*(1 + Sqrt[3])");
    assert_eq!(interpret("Csc[5 Pi/12]").unwrap(), "Sqrt[2]*(-1 + Sqrt[3])");
    assert_eq!(interpret("Sec[Pi/12]").unwrap(), "Sqrt[2]*(-1 + Sqrt[3])");
    assert_eq!(interpret("Sec[5 Pi/12]").unwrap(), "Sqrt[2]*(1 + Sqrt[3])");
  }

  // ─── Reciprocal trig/hyperbolic products ──────────────────────────
  // A function times its own reciprocal of the same argument collapses:
  // Sin[x] Csc[x] -> 1, Sin[x]^2 Csc[x] -> Sin[x], Csc[x]/Sin[x] -> Csc[x]^2.
  #[test]
  fn reciprocal_trig_products_collapse() {
    assert_eq!(interpret("Sin[x] Csc[x]").unwrap(), "1");
    assert_eq!(interpret("Cos[x] Sec[x]").unwrap(), "1");
    assert_eq!(interpret("Tan[x] Cot[x]").unwrap(), "1");
    assert_eq!(interpret("Sinh[x] Csch[x]").unwrap(), "1");
    assert_eq!(interpret("2 Sin[x] Csc[x]").unwrap(), "2");
    assert_eq!(interpret("Sin[x]^2 Csc[x]").unwrap(), "Sin[x]");
    assert_eq!(interpret("Sin[x]^3 Csc[x]^2").unwrap(), "Sin[x]");
    assert_eq!(interpret("Csc[x]/Sin[x]").unwrap(), "Csc[x]^2");
    assert_eq!(
      interpret("Csc[x] Sin[x]^2 Cos[x]").unwrap(),
      "Cos[x]*Sin[x]"
    );
  }

  // Boundaries that must NOT collapse: different arguments, lone reciprocals,
  // and cross-function quotients (Wolfram's Cos/Sin -> Cot is a separate,
  // form-divergent canonicalization that Woxi does not perform).
  #[test]
  fn reciprocal_trig_non_pairs_unchanged() {
    assert_eq!(interpret("Sin[x] Csc[y]").unwrap(), "Csc[y]*Sin[x]");
    assert_eq!(interpret("Csc[x]").unwrap(), "Csc[x]");
    assert_eq!(interpret("Cos[x]/Sin[x]").unwrap(), "Cos[x]/Sin[x]");
  }

  // ─── Hyperbolic parity ────────────────────────────────────────────

  #[test]
  fn sinh_negative_arg() {
    assert_eq!(interpret("Sinh[-1]").unwrap(), "-Sinh[1]");
    assert_eq!(interpret("Sinh[-2]").unwrap(), "-Sinh[2]");
    assert_eq!(interpret("Sinh[-x]").unwrap(), "-Sinh[x]");
  }

  #[test]
  fn cosh_negative_arg() {
    assert_eq!(interpret("Cosh[-1]").unwrap(), "Cosh[1]");
    assert_eq!(interpret("Cosh[-x]").unwrap(), "Cosh[x]");
  }

  #[test]
  fn tanh_negative_arg() {
    assert_eq!(interpret("Tanh[-1]").unwrap(), "-Tanh[1]");
    assert_eq!(interpret("Tanh[-x]").unwrap(), "-Tanh[x]");
  }

  // ─── Hyperbolic of Log ────────────────────────────────────────────
  // f[Log[u]] reduces to a rational function of u, since E^Log[u] = u.

  #[test]
  fn sinh_of_log_rational() {
    assert_eq!(interpret("Sinh[Log[2]]").unwrap(), "3/4");
    assert_eq!(interpret("Sinh[Log[5]]").unwrap(), "12/5");
    assert_eq!(interpret("Sinh[Log[2/3]]").unwrap(), "-5/12");
  }

  #[test]
  fn cosh_of_log_rational() {
    assert_eq!(interpret("Cosh[Log[2]]").unwrap(), "5/4");
    assert_eq!(interpret("Cosh[Log[3]]").unwrap(), "5/3");
    assert_eq!(interpret("Cosh[Log[2/3]]").unwrap(), "13/12");
  }

  #[test]
  fn tanh_coth_of_log_rational() {
    assert_eq!(interpret("Tanh[Log[2]]").unwrap(), "3/5");
    assert_eq!(interpret("Coth[Log[3]]").unwrap(), "5/4");
  }

  #[test]
  fn sech_csch_of_log_rational() {
    assert_eq!(interpret("Sech[Log[2]]").unwrap(), "4/5");
    assert_eq!(interpret("Csch[Log[2]]").unwrap(), "4/3");
  }

  #[test]
  fn hyperbolic_of_log_symbolic() {
    assert_eq!(interpret("Sinh[Log[x]]").unwrap(), "(-1 + x^2)/(2*x)");
    assert_eq!(interpret("Cosh[Log[x]]").unwrap(), "(1 + x^2)/(2*x)");
    assert_eq!(interpret("Tanh[Log[x]]").unwrap(), "(-1 + x^2)/(1 + x^2)");
    assert_eq!(interpret("Coth[Log[x]]").unwrap(), "(1 + x^2)/(-1 + x^2)");
    assert_eq!(interpret("Sech[Log[x]]").unwrap(), "(2*x)/(1 + x^2)");
    assert_eq!(interpret("Csch[Log[x]]").unwrap(), "(2*x)/(-1 + x^2)");
  }

  // Integer powers inside the Log still reduce (Log[x^2]).
  #[test]
  fn sinh_of_log_integer_power() {
    assert_eq!(interpret("Sinh[Log[x^2]]").unwrap(), "(-1 + x^4)/(2*x^2)");
    assert_eq!(interpret("Sinh[Log[4]]").unwrap(), "15/8");
  }

  // Sign extraction composes with the reduction.
  #[test]
  fn sinh_of_negated_log() {
    assert_eq!(interpret("Sinh[-Log[2]]").unwrap(), "-3/4");
    assert_eq!(interpret("Tanh[Log[1/2]]").unwrap(), "-3/5");
  }

  // wolframscript leaves a non-unit Log coefficient unevaluated, as does
  // Woxi: only a bare Log[u] triggers the reduction.
  #[test]
  fn scaled_log_not_reduced() {
    assert_eq!(interpret("Sinh[2 Log[2]]").unwrap(), "Sinh[2*Log[2]]");
  }

  // ─── N[trig, prec] with complex arguments ─────────────────────────

  #[test]
  fn n_sin_i_100_digits() {
    let result = interpret("N[Sin[I], 100]").unwrap();
    // Should start with correct digits and end with *I
    assert!(
      result.starts_with("1.175201193643801456882381850595600815"),
      "N[Sin[I],100] should have correct digits, got: {}",
      result
    );
    assert!(
      result.contains("*I"),
      "N[Sin[I],100] should be purely imaginary, got: {}",
      result
    );
    assert!(
      !result.contains("+") && !result.contains("0.`"),
      "N[Sin[I],100] should not have a real part, got: {}",
      result
    );
  }

  #[test]
  fn n_cos_i_100_digits() {
    let result = interpret("N[Cos[I], 100]").unwrap();
    assert!(
      result.starts_with("1.543080634815243778477905620757061682"),
      "N[Cos[I],100] digits wrong, got: {}",
      result
    );
    assert!(
      !result.contains("I"),
      "N[Cos[I],100] should be purely real, got: {}",
      result
    );
  }

  #[test]
  fn n_sin_complex_100_digits() {
    let result = interpret("N[Sin[1 + I], 100]").unwrap();
    // Both real and imaginary parts should be present
    assert!(
      result.starts_with("1.298457581415977294826042365807815"),
      "N[Sin[1+I],100] real part wrong, got: {}",
      result
    );
    assert!(
      result.contains("0.634963914784736108255082202991509"),
      "N[Sin[1+I],100] imaginary part wrong, got: {}",
      result
    );
    assert!(
      result.contains("*I"),
      "N[Sin[1+I],100] should have imaginary part, got: {}",
      result
    );
  }

  // Wolfram caps the digit count AFTER the decimal point per bit budget.
  // For prec 40 (192-bit equivalent), that cap is 58 fractional digits, so
  // a value ≥ 1 (one integer digit) shows 59 sig digits while a value < 1
  // shows 58 — and the last shown digit is rounded, not truncated.
  #[test]
  fn n_sqrt2_prec40_rounds_last_digit() {
    assert_eq!(
      interpret("N[Sqrt[2], 40]").unwrap(),
      "1.4142135623730950488016887242096980785696718753769480731767`40."
    );
  }

  #[test]
  fn n_sin1_prec40_caps_at_58_fraction_digits() {
    assert_eq!(
      interpret("N[Sin[1], 40]").unwrap(),
      "0.8414709848078965066525023216302989996225630607983710656728`40."
    );
  }

  #[test]
  fn n_sqrt2_prec4_shows_20_significant_digits() {
    // Bit budget for prec ≤ 8 is 64 bits → 19 fractional digits cap; with
    // a single integer digit that yields the 20-significant-digit form
    // Wolfram displays for any prec in this band.
    assert_eq!(
      interpret("N[Sqrt[2], 4]").unwrap(),
      "1.4142135623730950488`4."
    );
  }
}

/// Regression tests for unary minus after operators (issues 2 & 5)
mod unary_minus_after_operator {
  use super::*;

  #[test]
  fn times_minus_var() {
    // a * -b should parse as a * (-b)
    assert_eq!(interpret("2 * -3").unwrap(), "-6");
  }

  #[test]
  fn times_minus_expr() {
    // a*(b + c)*-d
    assert_eq!(
      interpret("a*(b + c)*-d /. {a -> 2, b -> 3, c -> 4, d -> 5}").unwrap(),
      "-70"
    );
  }

  #[test]
  fn assign_minus_map() {
    // y = -f /@ list should parse (unary minus before /@ )
    let result = interpret("f[x_] := x^2; y = -f /@ {1,2,3}; y").unwrap();
    // -f mapped over list: (-f)[1] etc. — depends on evaluation
    assert!(
      result.contains("{"),
      "Expected list result, got: {}",
      result
    );
  }

  #[test]
  fn minus_power_precedence() {
    // a * -b^2 should be a * (-(b^2)), not a * (-b)^2
    assert_eq!(interpret("a * -b^2 /. {a -> 2, b -> 3}").unwrap(), "-18");
  }

  #[test]
  fn plus_minus() {
    // a + -b should be a - b
    assert_eq!(interpret("5 + -3").unwrap(), "2");
  }

  #[test]
  fn negative_base_power_parens() {
    // (-1)^n should display with parentheses around -1
    assert_eq!(interpret("Power[-1, n]").unwrap(), "(-1)^n");
  }

  #[test]
  fn negative_imaginary_base_power_parens() {
    // A -I base must be parenthesized: `-I^k` reparses as `-(I^k)`, a
    // different (wrong) value. wolframscript keeps (-I)^k.
    assert_eq!(interpret("(-I)^k").unwrap(), "(-I)^k");
    assert_eq!(interpret("(-I)^I").unwrap(), "(-I)^I");
    assert_eq!(interpret("(-I)^n").unwrap(), "(-I)^n");
    // Integer exponents still fold to a concrete value.
    assert_eq!(interpret("(-I)^2").unwrap(), "-1");
    assert_eq!(interpret("(-I)^3").unwrap(), "I");
    // A non-unit imaginary coefficient was already parenthesized.
    assert_eq!(interpret("(-2 I)^k").unwrap(), "(-2*I)^k");
  }

  #[test]
  fn rational_exponent_parens() {
    // (-1)^(1/3) should have parens around the Rational exponent
    assert_eq!(interpret("Power[-1, 1/3]").unwrap(), "(-1)^(1/3)");
  }

  #[test]
  fn neg1_rational_power_simplification() {
    // ((-1)^(1/3))^2 = (-1)^(2/3)
    assert_eq!(interpret("((-1)^(1/3))^2").unwrap(), "(-1)^(2/3)");
  }

  #[test]
  fn neg_neg1_rational_power_simplification() {
    // (-(-1)^(1/3))^2 = (-1)^(2/3)
    assert_eq!(interpret("(-(-1)^(1/3))^2").unwrap(), "(-1)^(2/3)");
  }

  #[test]
  fn neg1_rational_power_simplification_4_3() {
    // ((-1)^(2/3))^2 = -(-1)^(1/3)
    assert_eq!(interpret("((-1)^(2/3))^2").unwrap(), "-(-1)^(1/3)");
  }

  #[test]
  fn neg1_half_power_is_i() {
    assert_eq!(interpret("(-1)^(1/2)").unwrap(), "I");
  }

  #[test]
  fn negative_base_cube_root() {
    assert_eq!(interpret("(-8)^(1/3)").unwrap(), "2*(-1)^(1/3)");
    assert_eq!(interpret("(-27)^(1/3)").unwrap(), "3*(-1)^(1/3)");
  }

  #[test]
  fn negative_base_sqrt() {
    assert_eq!(interpret("(-4)^(1/2)").unwrap(), "2*I");
    assert_eq!(interpret("(-9)^(1/2)").unwrap(), "3*I");
  }
}

/// Regression test for @ prefix application in ReplaceAll context (issue 3)
mod prefix_apply_in_replace_all {
  use super::*;

  #[test]
  fn first_at_list_in_replace_all() {
    assert_eq!(interpret("x /. First@{x -> 42}").unwrap(), "42");
  }

  #[test]
  fn last_at_list_in_replace_all() {
    assert_eq!(interpret("x /. Last@{x -> 10, x -> 20}").unwrap(), "20");
  }
}

/// Regression test for multi-line expression continuation (issue 4)
mod multiline_continuation {
  use super::*;

  #[test]
  fn division_across_lines() {
    assert_eq!(interpret("a = 6 /\n  2\na").unwrap(), "3");
  }

  #[test]
  fn plus_across_lines() {
    assert_eq!(interpret("a = 3 +\n  4\na").unwrap(), "7");
  }

  #[test]
  fn times_across_lines() {
    assert_eq!(interpret("a = 3 *\n  4\na").unwrap(), "12");
  }
}

mod midpoint {
  use super::*;

  #[test]
  fn two_points_2d() {
    assert_eq!(interpret("Midpoint[{{0, 0}, {4, 6}}]").unwrap(), "{2, 3}");
  }

  #[test]
  fn two_points_3d() {
    assert_eq!(
      interpret("Midpoint[{{1, 2, 3}, {5, 6, 7}}]").unwrap(),
      "{3, 4, 5}"
    );
  }

  #[test]
  fn symbolic_points() {
    assert_eq!(
      interpret("Midpoint[{{a, b}, {c, d}}]").unwrap(),
      "{(a + c)/2, (b + d)/2}"
    );
  }

  #[test]
  fn with_line() {
    assert_eq!(
      interpret("Midpoint[Line[{{0, 0}, {4, 6}}]]").unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn scalar_midpoint_unevaluated() {
    // Midpoint[{2, 8}] returns unevaluated because {2, 8} are scalars, not points
    assert_eq!(interpret("Midpoint[{2, 8}]").unwrap(), "Midpoint[{2, 8}]");
  }
}

mod qfactorial {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("QFactorial[0, q]").unwrap(), "1");
  }

  #[test]
  fn one() {
    assert_eq!(interpret("QFactorial[1, q]").unwrap(), "1");
  }

  #[test]
  fn three_half() {
    assert_eq!(interpret("QFactorial[3, 1/2]").unwrap(), "21/8");
  }

  #[test]
  fn four_half() {
    assert_eq!(interpret("QFactorial[4, 1/2]").unwrap(), "315/64");
  }

  #[test]
  fn series_at_zero_n3() {
    assert_eq!(
      interpret("Series[QFactorial[3, q], {q, 0, 3}]").unwrap(),
      "SeriesData[q, 0, {1, 2, 2, 1}, 0, 4, 1]"
    );
  }

  #[test]
  fn series_at_zero_n5() {
    assert_eq!(
      interpret("Series[QFactorial[5, q], {q, 0, 5}]").unwrap(),
      "SeriesData[q, 0, {1, 4, 9, 15, 20, 22}, 0, 6, 1]"
    );
  }

  #[test]
  fn series_at_zero_n10() {
    // Audit case: must not time out.
    assert_eq!(
      interpret("Series[QFactorial[10, q], {q, 0, 10}]").unwrap(),
      "SeriesData[q, 0, {1, 9, 44, 155, 440, 1068, 2298, 4489, 8095, 13640, 21670}, 0, 11, 1]"
    );
  }

  #[test]
  fn series_partial_order() {
    // Lower truncation than the polynomial degree.
    assert_eq!(
      interpret("Series[QFactorial[5, q], {q, 0, 2}]").unwrap(),
      "SeriesData[q, 0, {1, 4, 9}, 0, 3, 1]"
    );
  }
}

mod plus_term_ordering {
  use super::*;

  #[test]
  fn monomial_before_multi_var_sqrt() {
    // When Sqrt has more variables than the monomial, monomial comes first
    assert_eq!(
      interpret("Plus[Sqrt[b^2 - 4*a*c], b]").unwrap(),
      "b + Sqrt[b^2 - 4*a*c]"
    );
    assert_eq!(
      interpret("Plus[-b, -Sqrt[b^2 - 4*a*c]]").unwrap(),
      "-b - Sqrt[b^2 - 4*a*c]"
    );
  }

  #[test]
  fn monomial_before_multi_var_power() {
    assert_eq!(interpret("Plus[(a+c)^2, b]").unwrap(), "b + (a + c)^2");
    assert_eq!(interpret("Plus[Sqrt[a*c], b]").unwrap(), "b + Sqrt[a*c]");
  }

  #[test]
  fn compound_before_monomial_when_all_vars_earlier() {
    // When all compound vars are alphabetically before the monomial
    assert_eq!(interpret("Plus[(a+b)^2, c]").unwrap(), "(a + b)^2 + c");
    assert_eq!(
      interpret("Plus[5*(5*a + b), c]").unwrap(),
      "5*(5*a + b) + c"
    );
  }

  #[test]
  fn single_var_sqrt_before_later_monomial() {
    assert_eq!(interpret("Plus[Sqrt[a], b]").unwrap(), "Sqrt[a] + b");
    assert_eq!(interpret("Plus[Sqrt[x], x]").unwrap(), "Sqrt[x] + x");
    assert_eq!(interpret("Plus[Sqrt[b], b]").unwrap(), "Sqrt[b] + b");
  }

  // A real-variable term (bare symbol or power) always sorts before an
  // indexed-variable function call like C[1] or x[3], regardless of head
  // name. Indexed-vs-indexed still orders by name. Verified against
  // wolframscript; surfaces in DSolve/RSolve results (`x^2 + C[1]`).
  #[test]
  fn real_var_before_indexed_function_call() {
    assert_eq!(interpret("C[1] + x").unwrap(), "x + C[1]");
    assert_eq!(interpret("C[1] + x^2").unwrap(), "x^2 + C[1]");
    assert_eq!(interpret("C[1] + x + x^2").unwrap(), "x + x^2 + C[1]");
    assert_eq!(interpret("f[1] + x").unwrap(), "x + f[1]");
    assert_eq!(interpret("2 x + C[1]").unwrap(), "2*x + C[1]");
    // A bare symbol precedes an indexed call even with the same head/name.
    assert_eq!(interpret("x[2] + x").unwrap(), "x + x[2]");
    assert_eq!(interpret("z + a[1]").unwrap(), "z + a[1]");
    // Indexed-vs-indexed orders by name (unchanged).
    assert_eq!(interpret("b[1] + a[1]").unwrap(), "a[1] + b[1]");
    assert_eq!(interpret("C[1] + x[1]").unwrap(), "C[1] + x[1]");
  }
}

// A machine-real scalar times a complex number folds into a single machine
// complex number (wolframscript collapses the inexact product). Integer
// coefficients already folded; this covers Real/constant scalars. Verified
// against wolframscript.
mod real_scalar_times_complex {
  use super::*;

  #[test]
  fn real_scalar_folds() {
    assert_eq!(interpret("2.*(3 + 4 I)").unwrap(), "6. + 8.*I");
    assert_eq!(interpret("0.5*(2 + 4 I)").unwrap(), "1. + 2.*I");
  }

  #[test]
  fn inexact_constant_numerifies() {
    assert_eq!(
      interpret("Pi*(2. + I)").unwrap(),
      "6.283185307179586 + 3.141592653589793*I"
    );
  }

  #[test]
  fn complex_product_with_real_part() {
    // A product of two complex numbers that is real keeps the `+ 0.*I`.
    assert_eq!(interpret("I*(2.*I)").unwrap(), "-2. + 0.*I");
    assert_eq!(interpret("(1. + I)*(1 - I)").unwrap(), "2. + 0.*I");
  }

  #[test]
  fn exact_product_stays_symbolic() {
    // No inexact factor: the exact form is preserved, not numerified.
    assert_eq!(interpret("Pi*(2 + I)").unwrap(), "(2 + I)*Pi");
    // A symbolic factor blocks folding entirely.
    assert_eq!(interpret("x*(2. + I)").unwrap(), "(2. + 1.*I)*x");
  }

  #[test]
  fn cube_root_of_unity_numerifies() {
    // N of -(-1)^(1/3) must distribute the -1. into the complex value.
    assert_eq!(
      interpret("N[-(-1)^(1/3)]").unwrap(),
      "-0.5000000000000001 - 0.8660254037844386*I"
    );
  }
}

mod complex_division {
  use super::*;

  #[test]
  fn basic_complex_over_complex() {
    assert_eq!(interpret("(2 + 3 I) / (1 - I)").unwrap(), "-1/2 + (5*I)/2");
  }

  #[test]
  fn complex_over_complex_2() {
    assert_eq!(
      interpret("(3 + 4 I) / (1 + 2 I)").unwrap(),
      "11/5 - (2*I)/5"
    );
  }

  #[test]
  fn one_over_complex() {
    assert_eq!(interpret("1/(2 + I)").unwrap(), "2/5 - I/5");
  }

  #[test]
  fn imaginary_over_complex() {
    assert_eq!(interpret("I / (1 + I)").unwrap(), "1/2 + I/2");
  }

  #[test]
  fn complex_self_division() {
    assert_eq!(interpret("(1 + I) / (1 + I)").unwrap(), "1");
  }

  #[test]
  fn one_over_i() {
    assert_eq!(interpret("1/I").unwrap(), "-I");
  }

  #[test]
  fn plus_sort_total_order_with_opaque_fn() {
    // Regression test for #107: sorting Plus terms with a mix of
    // constants, polynomials, and opaque function calls must not panic.
    let result = interpret(
      "Simplify[Int[(2+3*x)^2*(4+5*x)^3,x] - (2+3*x)^3*(4+5*x)^3/(3*8) \
       + (3*(3*4-2*5))/(3*8)*((2*(2 + 3*x)^3)/405 \
       + ((2 + 3*x)^3*(4 + 5*x)^2)/15 + ((2 + 3*x)^3*(4 + 5*x))/45)]",
    );
    assert!(
      result.is_ok(),
      "should not panic with total order violation"
    );
  }
}

mod min_max_identity {
  use super::*;

  #[test]
  fn min_zero_args() {
    // Min[] is the identity element: Infinity
    assert_eq!(interpret("Min[]").unwrap(), "Infinity");
  }

  #[test]
  fn max_zero_args() {
    // Max[] is the identity element: -Infinity
    assert_eq!(interpret("Max[]").unwrap(), "-Infinity");
  }

  #[test]
  fn min_single_arg() {
    assert_eq!(interpret("Min[3]").unwrap(), "3");
  }

  #[test]
  fn max_single_arg() {
    assert_eq!(interpret("Max[5]").unwrap(), "5");
  }

  #[test]
  fn max_min_deduplicate_symbolic_args() {
    // Max/Min are idempotent: duplicate symbolic arguments are removed.
    assert_eq!(interpret("Max[a, b, a]").unwrap(), "Max[a, b]");
    assert_eq!(interpret("Min[a, b, a]").unwrap(), "Min[a, b]");
    assert_eq!(interpret("Max[a, a, a]").unwrap(), "a");
    assert_eq!(interpret("Max[a, b, c, b]").unwrap(), "Max[a, b, c]");
    // Numeric duplicates collapse into the best value alongside symbols.
    assert_eq!(interpret("Max[2, a, 2, a]").unwrap(), "Max[2, a]");
  }

  #[test]
  fn min_max_basic() {
    assert_eq!(
      interpret("MinMax[{3, 1, 4, 1, 5, 9, 2, 6}]").unwrap(),
      "{1, 9}"
    );
  }

  #[test]
  fn min_max_with_integer_expansion() {
    assert_eq!(
      interpret("MinMax[{3, 1, 4, 1, 5, 9, 2, 6}, 1]").unwrap(),
      "{0, 10}"
    );
  }

  #[test]
  fn min_max_with_rational_expansion() {
    assert_eq!(
      interpret("MinMax[{3, 1, 4, 1, 5, 9, 2, 6}, 1/2]").unwrap(),
      "{1/2, 19/2}"
    );
  }

  #[test]
  fn min_max_with_asymmetric_expansion() {
    assert_eq!(
      interpret("MinMax[{3, 1, 4, 1, 5, 9, 2, 6}, {1, 2}]").unwrap(),
      "{0, 11}"
    );
  }

  #[test]
  fn min_max_with_scaled_expansion() {
    assert_eq!(
      interpret("MinMax[{1.0, 2.0, 3.0}, Scaled[0.1]]").unwrap(),
      "{0.8, 3.2}"
    );
  }
}

mod exp_log_identity {
  use super::*;

  #[test]
  fn exp_log_simplifies() {
    // E^Log[x] = x (inverse function identity)
    assert_eq!(interpret("Exp[Log[x]]").unwrap(), "x");
    assert_eq!(interpret("E^Log[x]").unwrap(), "x");
  }

  #[test]
  fn exp_log_numeric() {
    assert_eq!(interpret("E^Log[5]").unwrap(), "5");
    assert_eq!(interpret("Exp[Log[42]]").unwrap(), "42");
  }

  #[test]
  fn exp_log_compound() {
    assert_eq!(interpret("E^Log[a + b]").unwrap(), "a + b");
  }

  #[test]
  fn exp_n_log_x() {
    // E^(n*Log[x]) = x^n (generalized inverse)
    assert_eq!(interpret("Exp[2 Log[x]]").unwrap(), "x^2");
    assert_eq!(interpret("Exp[3 Log[2]]").unwrap(), "8");
    assert_eq!(interpret("Exp[a Log[x]]").unwrap(), "x^a");
  }

  // E^(Plus[...]) pulls each Log term out as a factor, keeping the rest under
  // a single E^(rest). Verified against wolframscript.
  #[test]
  fn exp_of_log_sum() {
    assert_eq!(interpret("Exp[Log[x] + Log[y]]").unwrap(), "x*y");
    assert_eq!(interpret("Exp[Log[x] - Log[y]]").unwrap(), "x/y");
    assert_eq!(interpret("Exp[Log[x] + Log[y] + z]").unwrap(), "E^z*x*y");
    assert_eq!(interpret("E^(Log[5] + Log[3])").unwrap(), "15");
  }

  #[test]
  fn exp_of_log_plus_constant() {
    assert_eq!(interpret("Exp[Log[2] + 1]").unwrap(), "2*E");
    assert_eq!(interpret("Exp[Log[x] + a]").unwrap(), "E^a*x");
    assert_eq!(interpret("Exp[Log[2] + Log[3] + 1]").unwrap(), "6*E");
  }

  // A Plus exponent with no Log term is left untouched.
  #[test]
  fn exp_of_plain_sum_unchanged() {
    assert_eq!(interpret("Exp[a + b]").unwrap(), "E^(a + b)");
    assert_eq!(interpret("Exp[1 + a]").unwrap(), "E^(1 + a)");
  }

  #[test]
  fn exp_half_log_x() {
    assert_eq!(interpret("Exp[Log[x] / 2]").unwrap(), "Sqrt[x]");
  }
}

mod rationalize {
  use super::*;

  #[test]
  fn rationalize_decimal_fractions() {
    // Wolfram converts machine-precision numbers to exact decimal fractions.
    // Numbers close to simple fractions (denom ≤ 3) are returned unchanged as Real.
    assert_eq!(interpret("Rationalize[0.333333]").unwrap(), "0.333333");
    assert_eq!(
      interpret("Rationalize[0.333333]; Rationalize[0.142857]").unwrap(),
      "142857/1000000"
    );
  }

  #[test]
  fn rationalize_pi_approx() {
    assert_eq!(interpret("Rationalize[3.14159]").unwrap(), "314159/100000");
  }

  #[test]
  fn exact_fraction() {
    assert_eq!(interpret("Rationalize[0.25]").unwrap(), "1/4");
    assert_eq!(interpret("Rationalize[0.5]").unwrap(), "1/2");
    assert_eq!(interpret("Rationalize[1.5]").unwrap(), "3/2");
  }

  #[test]
  fn one_tenth() {
    assert_eq!(interpret("Rationalize[0.1]").unwrap(), "1/10");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("Rationalize[0.0]").unwrap(), "0");
  }

  #[test]
  fn with_tolerance() {
    assert_eq!(interpret("Rationalize[0.333333, 0.001]").unwrap(), "1/3");
    // Rationalize finds smallest-denominator rational within tolerance
    assert_eq!(interpret("Rationalize[3.14159, 0.001]").unwrap(), "201/64");
  }

  #[test]
  fn integer_input() {
    assert_eq!(interpret("Rationalize[5]").unwrap(), "5");
  }

  // Rationalize acts on the real and imaginary parts of a complex number.
  #[test]
  fn complex() {
    assert_eq!(
      interpret("Rationalize[2.5 + 3.5 I]").unwrap(),
      "5/2 + (7*I)/2"
    );
    assert_eq!(interpret("Rationalize[0.1 + 0.2 I]").unwrap(), "1/10 + I/5");
    assert_eq!(
      interpret("Rationalize[1.5 - 2.5 I]").unwrap(),
      "3/2 - (5*I)/2"
    );
    // A Gaussian integer is already exact.
    assert_eq!(interpret("Rationalize[3 + 4 I]").unwrap(), "3 + 4*I");
  }

  // The tolerance form applies to each component.
  #[test]
  fn complex_with_tolerance() {
    assert_eq!(
      interpret("Rationalize[2.5 + 3.5 I, 0.01]").unwrap(),
      "5/2 + (7*I)/2"
    );
  }

  // All-or-nothing: if either part has no nearby rational, both stay real.
  #[test]
  fn complex_all_or_nothing() {
    assert_eq!(
      interpret("Rationalize[Pi + 0.5 I]").unwrap(),
      "3.141592653589793 + 0.5*I"
    );
    assert_eq!(
      interpret("Rationalize[0.333333 I]").unwrap(),
      "0. + 0.333333*I"
    );
  }

  // Rationalize threads element-wise over lists (WL has no Listable
  // attribute for it but threads via built-in list handling).
  #[test]
  fn list_threads() {
    assert_eq!(
      interpret("Rationalize[{1.5, 2.7}]").unwrap(),
      "{3/2, 27/10}"
    );
    assert_eq!(
      interpret("Rationalize[{0.5, 0.25, 0.1}]").unwrap(),
      "{1/2, 1/4, 1/10}"
    );
    // Exact and symbolic elements pass through unchanged.
    assert_eq!(
      interpret("Rationalize[{Pi, 0.5, 3}]").unwrap(),
      "{Pi, 1/2, 3}"
    );
  }

  // The tolerance argument is carried into each element.
  #[test]
  fn list_with_tolerance() {
    assert_eq!(
      interpret("Rationalize[{1.5, 2.7}, 0.1]").unwrap(),
      "{3/2, 8/3}"
    );
  }

  // Nested lists thread recursively.
  #[test]
  fn list_nested() {
    assert_eq!(
      interpret("Rationalize[{{1.5, 0.5}, {0.25, 0.1}}]").unwrap(),
      "{{3/2, 1/2}, {1/4, 1/10}}"
    );
  }

  // List elements may themselves be complex.
  #[test]
  fn list_complex_elements() {
    assert_eq!(
      interpret("Rationalize[{2.5 + 3.5 I, 0.1}]").unwrap(),
      "{5/2 + (7*I)/2, 1/10}"
    );
  }

  // Rationalize is not Listable in WL (it threads via built-in handling).
  #[test]
  fn not_listable_attribute() {
    assert_eq!(interpret("Attributes[Rationalize]").unwrap(), "{Protected}");
  }
}

mod biginteger_division {
  use super::*;

  #[test]
  fn factorial_ratio_34_33() {
    assert_eq!(interpret("34!/33!").unwrap(), "34");
  }

  #[test]
  fn factorial_ratio_100_99() {
    assert_eq!(interpret("100!/99!").unwrap(), "100");
  }

  #[test]
  fn factorial_ratio_100_98() {
    assert_eq!(interpret("100!/98!").unwrap(), "9900");
  }

  #[test]
  fn factorial_ratio_50_48() {
    assert_eq!(interpret("50!/48!").unwrap(), "2450");
  }

  #[test]
  fn big_integer_fraction_reduces() {
    // 200! / 199! should reduce to 200
    assert_eq!(interpret("200!/199!").unwrap(), "200");
  }
}

mod rational_overflow {
  use super::*;

  #[test]
  fn plus_three_large_rationals_no_panic() {
    // Regression: previously panicked with "attempt to multiply with
    // overflow" in plus_ast's i128 rational sum. The sum path must now
    // promote to BigInt on overflow.
    assert_eq!(
      interpret(
        "Rational[1,99999999999999999] + Rational[1,99999999999999998] + Rational[1,99999999999999997]"
      )
      .unwrap(),
      "29999999999999998800000000000000011/\
       999999999999999940000000000000001099999999999999994"
        .replace(|c: char| c.is_whitespace(), "")
    );
  }

  #[test]
  fn plus_symbolic_with_large_rationals_no_panic() {
    // Same overflow path, but with a symbolic term forcing the second
    // rational-sum branch (~ line 218 before the fix).
    let out = interpret(
      "Rational[1,99999999999999999] + Rational[1,99999999999999998] + x",
    )
    .unwrap();
    assert!(out.contains(" + x"), "unexpected output: {out}");
  }
}

mod real_digits_base {
  use super::*;

  #[test]
  fn base_140() {
    assert_eq!(interpret("RealDigits[220, 140]").unwrap(), "{{1, 80}, 2}");
  }

  #[test]
  fn base_7_rational() {
    assert_eq!(interpret("RealDigits[1/2, 7]").unwrap(), "{{{3}}, 0}");
    assert_eq!(interpret("RealDigits[3/2, 7]").unwrap(), "{{1, {3}}, 1}");
  }

  #[test]
  fn base_6_terminating() {
    assert_eq!(interpret("RealDigits[3/2, 6]").unwrap(), "{{1, 3}, 1}");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("RealDigits[0]").unwrap(), "{{0}, 1}");
  }
}

mod number_digit {
  use super::*;

  #[test]
  fn positive_position() {
    assert_eq!(interpret("NumberDigit[210.345, 2]").unwrap(), "2");
    assert_eq!(interpret("NumberDigit[210.345, 1]").unwrap(), "1");
    assert_eq!(interpret("NumberDigit[210.345, 0]").unwrap(), "0");
  }

  #[test]
  fn negative_position() {
    assert_eq!(interpret("NumberDigit[210.345, -1]").unwrap(), "3");
    assert_eq!(interpret("NumberDigit[210.345, -2]").unwrap(), "4");
    assert_eq!(interpret("NumberDigit[210.345, -3]").unwrap(), "5");
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn filter_rules_1() {
    assert_case(r#"FilterRules[{x -> 100, y -> 1000}, x]"#, r#"{x -> 100}"#);
  }
  #[test]
  fn filter_rules_2() {
    assert_case(
      r#"FilterRules[{x -> 100, y -> 1000}, x]; FilterRules[{x -> 100, y -> 1000, z -> 10000}, {a, b, x, z}]"#,
      r#"{x -> 100, z -> 10000}"#,
    );
  }
  #[test]
  fn power_1() {
    assert_case(r#"I^2"#, r#"-1"#);
  }
  #[test]
  fn plus_1() {
    assert_case(r#"I^2; (3+I)*(3-I)"#, r#"10"#);
  }
  #[test]
  fn trace_evaluation() {
    assert_case(r#"TraceEvaluation[(x + x)^2]"#, r#"TraceEvaluation[4*x^2]"#);
  }
  #[test]
  fn get_environment_1() {
    assert_case(
      r#"SetEnvironment["FOO" -> "bar"]; GetEnvironment["FOO"]"#,
      r#""FOO" -> "bar""#,
    );
  }
  #[test]
  fn get_environment_2() {
    assert_case(
      r#"SetEnvironment["FOO" -> "bar"]; GetEnvironment["FOO"]; SetEnvironment[{"FOO" -> "baz", "A" -> "B"}]; GetEnvironment["FOO"]"#,
      r#""FOO" -> "baz""#,
    );
  }
  #[test]
  fn greater_1() {
    assert_case(r#"a /. f[x_:0, u_] -> {u}"#, r#"a"#);
  }
  #[test]
  fn divide_1() {
    assert_case(r#"b // a"#, r#"a[b]"#);
  }
  #[test]
  fn divide_2() {
    assert_case(r#"b // a; c // b // a"#, r#"a[b[c]]"#);
  }
  #[test]
  fn precedence_form() {
    assert_case(
      r#"PrecedenceForm[x/y, 12] - z"#,
      r#"-z + PrecedenceForm[x/y, 12]"#,
    );
  }
  #[test]
  fn contexts() {
    assert_case(r#"Contexts[]; Contexts["HTML*"]"#, r#"{}"#);
  }
  #[test]
  fn hold_form() {
    assert_case(r#"HoldForm[1 + 2 + 3]"#, r#"HoldForm[1 + 2 + 3]"#);
  }
  #[test]
  fn real_sign_1() {
    assert_case(r#"RealSign[-3.]"#, r#"-1"#);
  }
  #[test]
  fn real_sign_2() {
    assert_case(
      r#"RealSign[-3.]; RealSign[2. + 3. I]"#,
      r#"RealSign[2. + 3.*I]"#,
    );
  }
  #[test]
  fn integer_literal_1() {
    assert_case(r#"42"#, r#"42"#);
  }
  #[test]
  fn power_2() {
    assert_case(r#"\(x \^ 2\)"#, r#"SuperscriptBox["x", "2"]"#);
  }
  #[test]
  fn expression_1() {
    assert_case(r#"\(x \^ 2\); \(x \_ 2\)"#, r#"SubscriptBox["x", "2"]"#);
  }
  #[test]
  fn plus_2() {
    assert_case(
      r#"\(x \^ 2\); \(x \_ 2\); \( a \+ b \% c\)"#,
      r#"UnderoverscriptBox["a", "b", "c"]"#,
    );
  }
  #[test]
  fn anonymous_function() {
    assert_case(
      r#"\(x \^ 2\); \(x \_ 2\); \( a \+ b \% c\); \( a \& b \% c\)"#,
      r#"UnderoverscriptBox["a", "c", "b"]"#,
    );
  }

  // Box-form `\/` is the FractionBox operator. It binds tighter
  // than the surrounding RowBox so `\(x \/ y + z\)` parses as
  // `RowBox[{FractionBox["x", "y"], "+", "z"}]`. Regression for
  // mathics makeboxes_tests.yaml `\(x \/ y + z\)` row.
  #[test]
  fn box_form_fraction_simple() {
    assert_case(r#"\(x \/ y\)"#, r#"FractionBox["x", "y"]"#);
  }

  #[test]
  fn box_form_fraction_in_rowbox() {
    assert_case(
      r#"\(x \/ y + z\)"#,
      r#"RowBox[{FractionBox["x", "y"], "+", "z"}]"#,
    );
  }

  // `\`` is the box-form FormBox operator. `<form> \` <body>`
  // wraps the body in `FormBox[body, form]`, where the body is
  // the entire remaining chain (as a RowBox when multiple tokens).
  // Regression for mathics makeboxes_tests.yaml `FormBox` row.
  #[test]
  fn box_form_form_box() {
    assert_case(
      r#"\(TraditionalForm \` a + b\)"#,
      r#"FormBox[RowBox[{"a", "+", "b"}], TraditionalForm]"#,
    );
  }

  // A balanced `(...)` group inside box-notation is taken as a
  // single unit by the surrounding operator. wolframscript:
  // `\(x \/ (y + z)\)` → `FractionBox["x", RowBox[{"(", RowBox[{"y",
  // "+", "z"}], ")"}]]`. Regression for mathics
  // makeboxes_tests.yaml `FractionBox bracket` row.
  #[test]
  fn box_form_fraction_with_paren_group() {
    assert_case(
      r#"\(x \/ (y + z)\)"#,
      r#"FractionBox["x", RowBox[{"(", RowBox[{"y", "+", "z"}], ")"}]]"#,
    );
  }
  #[test]
  fn span_inside_parens_then_span() {
    // wolframscript: (a;;b);;c -> Span[Span[a, b], c] (OutputForm keeps head)
    assert_case(r#"(a;;b);;c"#, r#"Span[Span[a, b], c]"#);
    assert_case(r#"(a;;b)"#, r#"Span[a, b]"#);
  }
  #[test]
  fn function_arrow_named_char() {
    // \[Function] is wolfram's binary function-arrow operator. Right
    // operand absorbs +, ==, ->, /; (precedence below all of these).
    assert_case(r#"x \[Function] y"#, r#"Function[x, y]"#);
    assert_case(r#"x \[Function] y + 1"#, r#"Function[x, y + 1]"#);
    assert_case(r#"x \[Function] y == 1"#, r#"Function[x, y == 1]"#);
    assert_case(r#"x \[Function] y -> 1"#, r#"Function[x, y -> 1]"#);
    assert_case(r#"(x \[Function] x^2)[5]"#, r#"25"#);
  }
  #[test]
  fn function_arrow_ascii() {
    // `|->` is the ASCII spelling of the \[Function] operator.
    assert_case(r#"x |-> x^2"#, r#"Function[x, x^2]"#);
    assert_case(r#"{x, y} |-> x + y"#, r#"Function[{x, y}, x + y]"#);
    assert_case(r#"(x |-> x^2)[5]"#, r#"25"#);
    assert_case(r#"({x, y} |-> x + y)[3, 4]"#, r#"7"#);
    assert_case(r#"Map[x |-> x + 1, {1, 2, 3}]"#, r#"{2, 3, 4}"#);
    // The right operand absorbs lower-precedence operators, like \[Function].
    assert_case(r#"x |-> y + 1"#, r#"Function[x, y + 1]"#);
  }
  #[test]
  fn ranked_max() {
    assert_case(r#"RankedMax[{482, 17, 181, -12}, 2]"#, r#"181"#);
  }
  #[test]
  fn ranked_min() {
    assert_case(r#"RankedMin[{482, 17, 181, -12}, 2]"#, r#"17"#);
  }
  #[test]
  fn take_largest() {
    assert_case(r#"TakeLargest[{100, -1, 50, 10}, 2]"#, r#"{100, 50}"#);
  }
  #[test]
  fn take_smallest() {
    assert_case(r#"TakeSmallest[{100, -1, 50, 10}, 2]"#, r#"{-1, 10}"#);
  }
  #[test]
  fn take_largest_smallest_upto() {
    // UpTo[n] takes min(n, available) elements without erroring.
    assert_case(r#"TakeLargest[{1, 2, 3}, UpTo[5]]"#, r#"{3, 2, 1}"#);
    assert_case(r#"TakeLargest[{5, 2, 8, 1, 9}, UpTo[3]]"#, r#"{9, 8, 5}"#);
    assert_case(r#"TakeSmallest[{5, 2, 8, 1}, UpTo[2]]"#, r#"{1, 2}"#);
    // UpTo within range behaves like the plain count.
    assert_case(r#"TakeLargest[{1, 2, 3}, UpTo[2]]"#, r#"{3, 2}"#);
    // The By variants honor UpTo too.
    assert_case(r#"TakeLargestBy[{1, 2, 3}, # &, UpTo[5]]"#, r#"{3, 2, 1}"#);
    assert_case(r#"TakeSmallestBy[{5, 2, 8, 1}, # &, UpTo[2]]"#, r#"{1, 2}"#);
  }
  #[test]
  fn take_largest_smallest_association() {
    // On an association, rank by value; the result is an association sorted
    // by value (descending for largest, ascending for smallest).
    assert_case(
      r#"TakeSmallest[<|"a" -> 3, "b" -> 1, "c" -> 2|>, 2]"#,
      r#"<|b -> 1, c -> 2|>"#,
    );
    assert_case(
      r#"TakeLargest[<|"a" -> 3, "b" -> 1, "c" -> 2|>, 2]"#,
      r#"<|a -> 3, c -> 2|>"#,
    );
    // Ties keep their original order.
    assert_case(
      r#"TakeLargest[<|"a" -> 3, "b" -> 1, "c" -> 3|>, 2]"#,
      r#"<|a -> 3, c -> 3|>"#,
    );
    // UpTo clamps to the number of entries.
    assert_case(
      r#"TakeSmallest[<|"a" -> 3, "b" -> 1, "c" -> 2|>, UpTo[5]]"#,
      r#"<|b -> 1, c -> 2, a -> 3|>"#,
    );
  }
  #[test]
  fn a() {
    assert_case(r#"a[b] ^:= x; x = 2; a[b]"#, r#"2"#);
  }
  #[test]
  fn set_1() {
    assert_case(r#"a + b ^= 2"#, r#"2"#);
  }
  #[test]
  fn coefficient_arrays_1() {
    assert_case(
      r#"CoefficientArrays[1 + x^3, x]"#,
      r#"{1, SparseArray[Automatic, {1}, 0, {1, {{0, 0}, {}}, {}}], SparseArray[Automatic, {1, 1}, 0, {1, {{0, 0}, {}}, {}}], SparseArray[Automatic, {1, 1, 1}, 0, {1, {{0, 1}, {{1, 1}}}, {1}}]}"#,
    );
  }
  #[test]
  fn coefficient_arrays_2() {
    assert_case(
      r#"CoefficientArrays[1 + x^3, x]; CoefficientArrays[1 + x y+ x^3, {x, y}]"#,
      r#"{1, SparseArray[Automatic, {2}, 0, {1, {{0, 0}, {}}, {}}], SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 1}, {{2}}}, {1}}], SparseArray[Automatic, {2, 2, 2}, 0, {1, {{0, 1, 1}, {{1, 1}}}, {1}}]}"#,
    );
  }
  #[test]
  fn coefficient_arrays_3() {
    assert_case(
      r#"CoefficientArrays[1 + x^3, x]; CoefficientArrays[1 + x y+ x^3, {x, y}]; CoefficientArrays[{1 + x^2, x y}, {x, y}]"#,
      r#"{SparseArray[Automatic, {2}, 0, {1, {{0, 1}, {{1}}}, {0 + 1}}], SparseArray[Automatic, {2, 2}, 0, {1, {{0, 0, 0}, {}}, {}}], SparseArray[Automatic, {2, 2, 2}, 0, {1, {{0, 1, 2}, {{1, 1}, {1, 2}}}, {1, 1}}]}"#,
    );
  }
  #[test]
  fn expand_all() {
    assert_case(
      r#"ExpandAll[(a + b) ^ 2 / (c + d)^2]"#,
      r#"a^2/(c^2 + 2*c*d + d^2) + (2*a*b)/(c^2 + 2*c*d + d^2) + b^2/(c^2 + 2*c*d + d^2)"#,
    );
  }
  #[test]
  fn expand_denominator() {
    assert_case(
      r#"ExpandDenominator[(a + b) ^ 2 / ((c + d)^2 (e + f))]"#,
      r#"(a + b)^2/(c^2*e + 2*c*d*e + d^2*e + c^2*f + 2*c*d*f + d^2*f)"#,
    );
  }
  #[test]
  fn factor_terms_list_1() {
    assert_case(r#"FactorTermsList[2 x^2 - 2]"#, r#"{2, -1 + x ^ 2}"#);
  }
  #[test]
  fn factor_terms_list_2() {
    assert_case(
      r#"FactorTermsList[2 x^2 - 2]; FactorTermsList[x^2 - 2 x + 1]"#,
      r#"{1, 1 - 2*x + x^2}"#,
    );
  }
  #[test]
  fn set_2() {
    assert_case(
      r#"FactorTermsList[2 x^2 - 2]; FactorTermsList[x^2 - 2 x + 1]; f = 3 (-1 + 2 x) (-1 + y) (1 - a)"#,
      r#"3*(1 - a)*(-1 + 2*x)*(-1 + y)"#,
    );
  }
  #[test]
  fn factor_terms_list_3() {
    assert_case(
      r#"FactorTermsList[2 x^2 - 2]; FactorTermsList[x^2 - 2 x + 1]; f = 3 (-1 + 2 x) (-1 + y) (1 - a); FactorTermsList[f]"#,
      r#"{-3, -1 + a + 2*x - 2*a*x + y - a*y - 2*x*y + 2*a*x*y}"#,
    );
  }
  #[test]
  fn factor_terms_list_4() {
    assert_case(
      r#"FactorTermsList[2 x^2 - 2]; FactorTermsList[x^2 - 2 x + 1]; f = 3 (-1 + 2 x) (-1 + y) (1 - a); FactorTermsList[f]; FactorTermsList[f, x]"#,
      r#"{-3, 1 - a - y + a*y, -1 + 2*x}"#,
    );
  }
  #[test]
  fn power_expand_1() {
    assert_case(r#"PowerExpand[(a ^ b) ^ c]"#, r#"a^(b*c)"#);
  }
  #[test]
  fn power_expand_2() {
    assert_case(
      r#"PowerExpand[(a ^ b) ^ c]; PowerExpand[(a * b) ^ c]"#,
      r#"a^c*b^c"#,
    );
  }
  #[test]
  fn power_expand_3() {
    assert_case(
      r#"PowerExpand[(a ^ b) ^ c]; PowerExpand[(a * b) ^ c]; PowerExpand[(x ^ 2) ^ (1/2)]"#,
      r#"x"#,
    );
  }
  #[test]
  fn variables_1() {
    assert_case(r#"Variables[a x^2 + b x + c]"#, r#"{c, b, x, a}"#);
  }
  #[test]
  fn variables_2() {
    assert_case(
      r#"Variables[a x^2 + b x + c]; Variables[{a + b x, c y^2 + x/2}]"#,
      r#"{a, b, c, x, y}"#,
    );
  }
  #[test]
  fn variables_of_non_polynomial_is_empty() {
    // Relational and logical expressions are not polynomials, so Variables
    // returns {} (rather than wrapping the whole expression as one variable).
    assert_case(r#"Variables[2 x == 6]"#, r#"{}"#);
    assert_case(r#"Variables[x > 2]"#, r#"{}"#);
    assert_case(r#"Variables[x != 2]"#, r#"{}"#);
    assert_case(r#"Variables[x^2 + 1 == 0 && y < 3]"#, r#"{}"#);
    // Polynomials and transcendental kernels are unaffected.
    assert_case(r#"Variables[x^2 + 1]"#, r#"{x}"#);
    assert_case(r#"Variables[Sin[x] + y]"#, r#"{y, Sin[x]}"#);
  }
  #[test]
  fn divide_3() {
    assert_case(r#"1 / Overflow[]"#, r#"Underflow[]"#);
  }
  #[test]
  fn times_1() {
    assert_case(r#"1 / Overflow[]; 5 * Underflow[]"#, r#"Underflow[]"#);
  }
  #[test]
  fn divide_4() {
    assert_case(r#"1 / Overflow[]; 5 * Underflow[]; % // N"#, r#"Out[0]"#);
  }
  #[test]
  fn minus_1() {
    assert_case(
      r#"1 / Overflow[]; 5 * Underflow[]; % // N; 1 - Underflow[]"#,
      r#"Underflow[] + 1"#,
    );
  }
  #[test]
  fn divide_5() {
    assert_case(
      r#"1 / Overflow[]; 5 * Underflow[]; % // N; 1 - Underflow[]; % // N"#,
      r#"Out[0]"#,
    );
  }
  #[test]
  fn discrete_limit_1() {
    assert_case(r#"DiscreteLimit[n/(n + 1), n -> Infinity]"#, r#"1"#);
  }
  #[test]
  fn discrete_limit_2() {
    assert_case(
      r#"DiscreteLimit[n/(n + 1), n -> Infinity]; DiscreteLimit[f[n], n -> Infinity]"#,
      r#"DiscreteLimit[f[n], n -> Infinity]"#,
    );
  }
  #[test]
  fn complex_expand() {
    assert_case(
      r#"ComplexExpand[3^(I x)]"#,
      r#"Cos[x*Log[3]] + I*Sin[x*Log[3]]"#,
    );
  }
  #[test]
  fn mantissa_exponent_1() {
    assert_case(r#"MantissaExponent[2.5*10^20]"#, r#"{0.25, 21}"#);
  }
  #[test]
  fn mantissa_exponent_2() {
    assert_case(
      r#"MantissaExponent[2.5*10^20]; MantissaExponent[125.24]"#,
      r#"{0.12524, 3}"#,
    );
  }
  #[test]
  fn mantissa_exponent_3() {
    assert_case(
      r#"MantissaExponent[2.5*10^20]; MantissaExponent[125.24]; MantissaExponent[125., 2]"#,
      r#"{0.9765625, 7}"#,
    );
  }
  #[test]
  fn mantissa_exponent_4() {
    assert_case(
      r#"MantissaExponent[2.5*10^20]; MantissaExponent[125.24]; MantissaExponent[125., 2]; MantissaExponent[10, b]"#,
      r#"MantissaExponent[10, b]"#,
    );
  }
  #[test]
  fn standard_form_1() {
    // mathics rendered the contents to box-syntax RowBox markup;
    // wolframscript -code returns the unevaluated wrapper
    // `StandardForm[a + b*c]` verbatim. Woxi matches wolframscript.
    assert_case(r#"StandardForm[a + b * c]"#, r#"StandardForm[a + b*c]"#);
  }
  #[test]
  fn standard_form_2() {
    // mathics rendered the contents to box-syntax markup; wolframscript -code
    // returns the unevaluated wrapper `StandardForm[A string]` verbatim
    // (strings print without quotes in OutputForm). Woxi matches.
    assert_case(
      r#"StandardForm[a + b * c]; StandardForm["A string"]"#,
      r#"StandardForm[A string]"#,
    );
  }
  #[test]
  fn expr() {
    assert_case(
      r#"StandardForm[a + b * c]; StandardForm["A string"]; f'[x]"#,
      r#"Derivative[1][f][x]"#,
    );
  }
  #[test]
  fn greater_2() {
    assert_case(r#"a + b + c /. a + b -> t"#, r#"c + t"#);
  }
  #[test]
  fn greater_3() {
    assert_case(
      r#"a + b + c /. a + b -> t; a + 2 + b + c + x * y /. n_Integer + s__Symbol + rest_ -> {n, s, rest}"#,
      r#"{2, a, b, c, x*y}"#,
    );
  }
  #[test]
  fn f_1() {
    assert_case(
      r#"a + b + c /. a + b -> t; a + 2 + b + c + x * y /. n_Integer + s__Symbol + rest_ -> {n, s, rest}; f[a, b, c, d] /. f[first_, rest___] -> {first, {rest}}"#,
      r#"{a, {b, c, d}}"#,
    );
  }
  #[test]
  fn greater_4() {
    assert_case(r#"a+b+c /. c->d"#, r#"a + b + d"#);
  }
  #[test]
  fn g() {
    assert_case(
      r#"a+b+c /. c->d; g[a+b+c,a]/.g[x_+y_,x_]->{x,y}"#,
      r#"{a, b + c}"#,
    );
  }
  #[test]
  fn list_literal_1() {
    assert_case(
      r#"a+b+c /. c->d; g[a+b+c,a]/.g[x_+y_,x_]->{x,y}; {a, b} /. {{a->x, b->y}, {a->u, b->v}}"#,
      r#"{{x, y}, {u, v}}"#,
    );
  }
  #[test]
  fn list_literal_2() {
    assert_case(
      r#"a+b+c /. c->d; g[a+b+c,a]/.g[x_+y_,x_]->{x,y}; {a, b} /. {{a->x, b->y}, {a->u, b->v}}; {a, b} /. {{{a->x, b->y}, {a->w, b->z}}, {a->u, b->v}}"#,
      r#"{{{x, y}, {w, z}}, {u, v}}"#,
    );
  }
  #[test]
  fn greater_5() {
    assert_case(r#"a+b+c //. c->d"#, r#"a + b + d"#);
  }
  #[test]
  fn list_literal_3() {
    assert_case(r#"a+b+c /. c->d; {x,x^2,y} /. x->3"#, r#"{3, 9, y}"#);
  }
  #[test]
  fn greater_6() {
    assert_case(r#"a+b+c+d/.(a|b)->t"#, r#"c + d + 2*t"#);
  }
  #[test]
  fn divide_6() {
    assert_case(r#"a_Integer.. // FullForm"#, r#"FullForm[(a_Integer)..]"#);
  }
  #[test]
  fn divide_7() {
    assert_case(
      r#"a_Integer.. // FullForm; 0..1 // FullForm"#,
      r#"FullForm[(0)..]"#,
    );
  }
  #[test]
  fn divide_8() {
    assert_case(
      r#"a___Integer... // FullForm"#,
      r#"FullForm[(a___Integer)...]"#,
    );
  }
  #[test]
  fn f_2() {
    assert_case(
      r#"a___Integer... // FullForm; f[x] /. f[x, 0...] -> t"#,
      r#"t"#,
    );
  }
  #[test]
  fn divide_9() {
    assert_case(
      r#"f[x_, y_:1] := {x, y}; f[x_, y_: 1] := {x, y}; f[a, 2]; f[a]; y : 1 // FullForm"#,
      r#"FullForm[y:1]"#,
    );
  }
  #[test]
  fn divide_10() {
    assert_case(
      r#"f[x_, y_:1] := {x, y}; f[x_, y_: 1] := {x, y}; f[a, 2]; f[a]; y : 1 // FullForm; y_ : 1 // FullForm"#,
      r#"FullForm[y_:1]"#,
    );
  }
  #[test]
  fn greater_7() {
    assert_case(r#"f[3] /. f[x_] /; x>0 -> t"#, r#"t"#);
  }
  #[test]
  fn greater_8() {
    assert_case(
      r#"f[3] /. f[x_] /; x>0 -> t; f[-3] /. f[x_] /; x>0 -> t"#,
      r#"f[-3]"#,
    );
  }
  #[test]
  fn f_3() {
    assert_case(
      r#"f[3] /. f[x_] /; x>0 -> t; f[-3] /. f[x_] /; x>0 -> t; f[x_] := p[x] /; x>0; f[3]"#,
      r#"p[3]"#,
    );
  }
  #[test]
  fn f_4() {
    assert_case(
      r#"f[3] /. f[x_] /; x>0 -> t; f[-3] /. f[x_] /; x>0 -> t; f[x_] := p[x] /; x>0; f[3]; f[-3]"#,
      r#"f[-3]"#,
    );
  }
  #[test]
  fn bray_curtis_distance_1() {
    assert_case(r#"BrayCurtisDistance[-7, 5]"#, r#"6"#);
  }
  #[test]
  fn bray_curtis_distance_2() {
    assert_case(
      r#"BrayCurtisDistance[-7, 5]; BrayCurtisDistance[{-1, -1}, {10, 10}]"#,
      r#"11 / 9"#,
    );
  }
  #[test]
  fn canberra_distance_1() {
    assert_case(r#"CanberraDistance[-7, 5]"#, r#"1"#);
  }
  #[test]
  fn canberra_distance_2() {
    assert_case(
      r#"CanberraDistance[-7, 5]; CanberraDistance[{-1, -1}, {1, 1}]"#,
      r#"2"#,
    );
  }
  #[test]
  fn chessboard_distance_1() {
    assert_case(r#"ChessboardDistance[-7, 5]"#, r#"12"#);
  }
  #[test]
  fn chessboard_distance_2() {
    assert_case(
      r#"ChessboardDistance[-7, 5]; ChessboardDistance[{-1, -1}, {1, 1}]"#,
      r#"2"#,
    );
  }
  #[test]
  fn euclidean_distance_1() {
    assert_case(r#"EuclideanDistance[-7, 5]"#, r#"12"#);
  }
  #[test]
  fn euclidean_distance_2() {
    assert_case(
      r#"EuclideanDistance[-7, 5]; EuclideanDistance[{-1, -1}, {1, 1}]"#,
      r#"2*Sqrt[2]"#,
    );
  }
  #[test]
  fn euclidean_distance_3() {
    assert_case(
      r#"EuclideanDistance[-7, 5]; EuclideanDistance[{-1, -1}, {1, 1}]; EuclideanDistance[{a, b}, {c, d}]"#,
      r#"Sqrt[Abs[a - c] ^ 2 + Abs[b - d] ^ 2]"#,
    );
  }
  #[test]
  fn manhattan_distance_1() {
    assert_case(r#"ManhattanDistance[-7, 5]"#, r#"12"#);
  }
  #[test]
  fn manhattan_distance_2() {
    assert_case(
      r#"ManhattanDistance[-7, 5]; ManhattanDistance[{-1, -1}, {1, 1}]"#,
      r#"4"#,
    );
  }
  #[test]
  fn squared_euclidean_distance_1() {
    assert_case(r#"SquaredEuclideanDistance[-7, 5]"#, r#"144"#);
  }
  #[test]
  fn squared_euclidean_distance_2() {
    assert_case(
      r#"SquaredEuclideanDistance[-7, 5]; SquaredEuclideanDistance[{-1, -1}, {1, 1}]"#,
      r#"8"#,
    );
  }
  #[test]
  fn file_type_1() {
    assert_case(r#"FileType["ExampleData/sunflowers.jpg"]"#, r#"None"#);
  }
  #[test]
  fn file_type_2() {
    assert_case(
      r#"FileType["ExampleData/sunflowers.jpg"]; FileType["ExampleData"]"#,
      r#"None"#,
    );
  }
  #[test]
  fn file_type_3() {
    assert_case(
      r#"FileType["ExampleData/sunflowers.jpg"]; FileType["ExampleData"]; FileType["ExampleData/nonexistent"]"#,
      r#"None"#,
    );
  }
  #[test]
  fn curl() {
    assert_case(r#"Curl[{y, -x}, {x, y}]"#, r#"-2"#);
  }
  #[test]
  fn curl_2d_scalar() {
    // Curl[s, {x, y}] for a scalar s returns {-D[s, y], D[s, x]}
    // (perpendicular gradient). Wolframscript:
    //   {-Derivative[0, 1][v][x, y], Derivative[1, 0][v][x, y]}
    assert_case(
      r#"Curl[v[x, y], {x, y}]"#,
      r#"{-Derivative[0, 1][v][x, y], Derivative[1, 0][v][x, y]}"#,
    );
    // Closed-form scalar: Curl[x*y, {x, y}] = {-x, y}.
    assert_case(r#"Curl[x*y, {x, y}]"#, r#"{-x, y}"#);
  }
  #[test]
  fn symbol_name() {
    assert_case(r#"SymbolName[x] // InputForm"#, r#"InputForm["x"]"#);
  }
  #[test]
  fn divide_11() {
    assert_case(r#"30 / 5"#, r#"6"#);
  }
  #[test]
  fn divide_12() {
    assert_case(r#"30 / 5; 1 / 8"#, r#"1 / 8"#);
  }
  #[test]
  fn minus_2() {
    assert_case(r#"-a // FullForm"#, r#"FullForm[-a]"#);
  }
  #[test]
  fn minus_3() {
    assert_case(r#"-a // FullForm; -(x - 2/3)"#, r#"2 / 3 - x"#);
  }
  #[test]
  fn plus_3() {
    assert_case(r#"1 + 2"#, r#"3"#);
  }
  #[test]
  fn plus_4() {
    assert_case(r#"1 + 2; a + b + a"#, r#"2*a + b"#);
  }
  #[test]
  fn plus_5() {
    assert_case(r#"1 + 2; a + b + a; a + a + 3 * a"#, r#"5*a"#);
  }
  #[test]
  fn plus_6() {
    assert_case(
      r#"1 + 2; a + b + a; a + a + 3 * a; a + b + 4.5 + a + b + a + 2 + 1.5 b"#,
      r#"6.5 + 3*a + 3.5*b"#,
    );
  }
  #[test]
  fn divide_13() {
    assert_case(r#"4 ^ (1/2)"#, r#"2"#);
  }
  #[test]
  fn divide_14() {
    assert_case(r#"4 ^ (1/2); 4 ^ (1/3)"#, r#"2 ^ (2 / 3)"#);
  }
  #[test]
  fn power_3() {
    assert_case(
      r#"4 ^ (1/2); 4 ^ (1/3); 3^123"#,
      r#"48519278097689642681155855396759336072749841943521979872827"#,
    );
  }
  #[test]
  fn divide_15() {
    assert_case(
      r#"4 ^ (1/2); 4 ^ (1/3); 3^123; (y ^ 2) ^ (1/2)"#,
      r#"Sqrt[y ^ 2]"#,
    );
  }
  #[test]
  fn power_4() {
    assert_case(
      r#"4 ^ (1/2); 4 ^ (1/3); 3^123; (y ^ 2) ^ (1/2); (y ^ 2) ^ 3"#,
      r#"y ^ 6"#,
    );
  }
  #[test]
  fn minus_4() {
    assert_case(r#"5 - 3"#, r#"2"#);
  }
  #[test]
  fn minus_5() {
    assert_case(r#"5 - 3; a - b // FullForm"#, r#"FullForm[a - b]"#);
  }
  #[test]
  fn minus_6() {
    assert_case(r#"5 - 3; a - b // FullForm; a - b - c"#, r#"a - b - c"#);
  }
  #[test]
  fn minus_7() {
    assert_case(
      r#"5 - 3; a - b // FullForm; a - b - c; a - (b - c)"#,
      r#"a - b + c"#,
    );
  }
  #[test]
  fn times_2() {
    assert_case(r#"10 * 2"#, r#"20"#);
  }
  #[test]
  fn expression_2() {
    assert_case(r#"10 * 2; 10 2"#, r#"20"#);
  }
  #[test]
  fn times_3() {
    assert_case(r#"10 * 2; 10 2; a * a"#, r#"a ^ 2"#);
  }
  #[test]
  fn minus_8() {
    assert_case(r#"10 * 2; 10 2; a * a; x ^ 10 * x ^ -2"#, r#"x ^ 8"#);
  }
  #[test]
  fn list_literal_4() {
    assert_case(
      r#"10 * 2; 10 2; a * a; x ^ 10 * x ^ -2; {1, 2, 3} * 4"#,
      r#"{4, 8, 12}"#,
    );
  }
  #[test]
  fn leaf_count_1() {
    assert_case(r#"LeafCount[1 + x + y^a]"#, r#"6"#);
  }
  #[test]
  fn leaf_count_2() {
    assert_case(r#"LeafCount[1 + x + y^a]; LeafCount[f[x, y]]"#, r#"3"#);
  }
  #[test]
  fn leaf_count_3() {
    assert_case(
      r#"LeafCount[1 + x + y^a]; LeafCount[f[x, y]]; LeafCount[{1 / 3, 1 + I}]"#,
      r#"7"#,
    );
  }
  #[test]
  fn level_1() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]"#,
      r#"{a, b, 3, 2, x, 2}"#,
    );
  }
  #[test]
  fn level_2() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]"#,
      r#"{{a}, {{a}}, {{{a}}}}"#,
    );
  }
  #[test]
  fn level_3() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]; Level[{{{{a}}}}, -4]"#,
      r#"{{{{a}}}}"#,
    );
  }
  #[test]
  fn level_4() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]; Level[{{{{a}}}}, -4]; Level[{{{{a}}}}, -5]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn level_5() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]; Level[{{{{a}}}}, -4]; Level[{{{{a}}}}, -5]; Level[h0[h1[h2[h3[a]]]], {0, -1}]"#,
      r#"{a, h3[a], h2[h3[a]], h1[h2[h3[a]]], h0[h1[h2[h3[a]]]]}"#,
    );
  }
  #[test]
  fn lighter() {
    assert_case(r#"Lighter[Orange, 1/4]"#, r#"RGBColor[1, 0.625, 1/4]"#);
  }
  #[test]
  fn take_largest_by_1() {
    assert_case(
      r#"TakeLargestBy[{{1, -1}, {10, 100}, {23, 7, 8}, {5, 1}}, Total, 2]"#,
      r#"{{10, 100}, {23, 7, 8}}"#,
    );
  }
  #[test]
  fn take_largest_by_2() {
    assert_case(
      r#"TakeLargestBy[{{1, -1}, {10, 100}, {23, 7, 8}, {5, 1}}, Total, 2]; TakeLargestBy[{"abc", "ab", "x"}, StringLength, 1]"#,
      r#"{"abc"}"#,
    );
  }
  #[test]
  fn take_smallest_by_1() {
    assert_case(
      r#"TakeSmallestBy[{{1, -1}, {10, 100}, {23, 7, 8}, {5, 1}}, Total, 2]"#,
      r#"{{1, -1}, {5, 1}}"#,
    );
  }
  #[test]
  fn take_smallest_by_2() {
    assert_case(
      r#"TakeSmallestBy[{{1, -1}, {10, 100}, {23, 7, 8}, {5, 1}}, Total, 2]; TakeSmallestBy[{"abc", "ab", "x"}, StringLength, 1]"#,
      r#"{"x"}"#,
    );
  }
  #[test]
  fn extract_1() {
    assert_case(r#"Extract[a + b + c, {2}]"#, r#"b"#);
  }
  #[test]
  fn extract_2() {
    assert_case(
      r#"Extract[a + b + c, {2}]; Extract[{{a, b}, {c, d}}, {{1}, {2, 2}}]"#,
      r#"{{a, b}, d}"#,
    );
  }
  #[test]
  fn extract_association_key() {
    // Key[k] and a bare string key extract the value at that key.
    assert_case(r#"Extract[<|"a" -> 1, "b" -> 2|>, Key["a"]]"#, r#"1"#);
    assert_case(r#"Extract[<|"a" -> 1, "b" -> 2|>, "b"]"#, r#"2"#);
    assert_case(r#"Extract[<|a -> 1, b -> 2|>, Key[b]]"#, r#"2"#);
  }
  #[test]
  fn divide_16() {
    assert_case(r#";; // FullForm"#, r#"FullForm[Span[1, All]]"#);
  }
  #[test]
  fn divide_17() {
    assert_case(
      r#";; // FullForm; 1;;4;;2 // FullForm"#,
      r#"FullForm[Span[1, 4, 2]]"#,
    );
  }
  #[test]
  fn minus_9() {
    assert_case(
      r#";; // FullForm; 1;;4;;2 // FullForm; 2;;-2 // FullForm"#,
      r#"FullForm[Span[2, -2]]"#,
    );
  }
  #[test]
  fn divide_18() {
    // mathics rendered the FullForm as `1 ;; 3` (using the Span shorthand);
    // wolframscript -code returns the explicit head `Span[1, 3]`, which is
    // what Woxi also produces.
    assert_case(
      r#";; // FullForm; 1;;4;;2 // FullForm; 2;;-2 // FullForm; ;;3 // FullForm"#,
      r#"FullForm[Span[1, 3]]"#,
    );
  }
  #[test]
  fn file_name_depth_1() {
    assert_case(r#"FileNameDepth["a/b/c"]"#, r#"3"#);
  }
  #[test]
  fn file_name_depth_2() {
    assert_case(r#"FileNameDepth["a/b/c"]; FileNameDepth["a/b/c/"]"#, r#"3"#);
  }
  #[test]
  fn file_name_split() {
    assert_case(
      r#"FileNameSplit["example/path/file.txt"]"#,
      r#"{"example", "path", "file.txt"}"#,
    );
  }
  #[test]
  fn string_literal_1() {
    assert_case(r#""a" ~~ "b" // FullForm"#, r#"FullForm["ab"]"#);
  }
  #[test]
  fn file_information() {
    assert_case(r#"FileInformation["ExampleData/sunflowers.jpg"]"#, r#"{}"#);
  }
  #[test]
  fn find_file() {
    assert_case(r#"FindFile["ExampleData/sunflowers.jpg"]"#, r#"$Failed"#);
  }
  #[test]
  fn integer_literal_2() {
    assert_case(r#"1"#, r#"1"#);
  }
  #[test]
  fn integer_literal_3() {
    assert_case(r#"1"#, r#"1"#);
  }
  #[test]
  fn divide_19() {
    assert_case(r#"1; 2/9"#, r#"2/9"#);
  }
  #[test]
  fn integer_literal_4() {
    assert_case(r#"1"#, r#"1"#);
  }
  #[test]
  fn list_literal_5() {
    assert_case(
      r#"1; {1, 1.}; {1.000123`6, 1.0001`4, 2/9}"#,
      r#"{1.000123`6., 1.0001`4., 2/9}"#,
    );
  }
  #[test]
  fn integer_literal_5() {
    assert_case(r#"2"#, r#"2"#);
  }
  #[test]
  fn plus_7() {
    assert_case(r#"2; 3+2 I"#, r#"3 + 2*I"#);
  }
  #[test]
  fn divide_20() {
    assert_case(r#"2; 3+2 I; 2/9"#, r#"2/9"#);
  }
  #[test]
  fn string_literal_2() {
    assert_case(r#"2; 3+2 I; 2/9;  "hi!""#, r#""hi!""#);
  }
  #[test]
  fn infinity() {
    assert_case(r#"2; 3+2 I; 2/9;  "hi!"; Infinity"#, r#"Infinity"#);
  }
  #[test]
  fn plus_8() {
    assert_case(
      r#"\(c (1 + x)\)"#,
      r#"RowBox[{"c", RowBox[{"(", RowBox[{"1", "+", "x"}], ")"}]}]"#,
    );
  }
  #[test]
  fn power_5() {
    assert_case(r#"\(c (1 + x)\); \!\(x \^ 2\)"#, r#"x ^ 2"#);
  }
  #[test]
  fn divide_21() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes"#,
      r#"FractionBox["a", "b"]"#,
    );
  }
  #[test]
  fn integer_literal_6() {
    assert_case(r#"1"#, r#"1"#);
  }
  #[test]
  fn times_4() {
    assert_case(r#"1; x; 2*x"#, r#"2*x"#);
  }
  #[test]
  fn plus_9() {
    assert_case(r#"1; x; 2*x; 1+x"#, r#"1 + x"#);
  }
  #[test]
  fn plus_10() {
    assert_case(r#"1; x; 2*x; 1+x; 1+2*x"#, r#"1 + 2*x"#);
  }
  #[test]
  fn integer_literal_7() {
    assert_case(r#"1; x; 2*x; 1+x; 1+2*x; 1"#, r#"1"#);
  }
  #[test]
  fn symbol_literal_1() {
    assert_case(r#"1; x; 2*x; 1+x; 1+2*x; 1; x"#, r#"x"#);
  }
  #[test]
  fn power_6() {
    assert_case(r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1"#, r#"x"#);
  }
  #[test]
  fn power_7() {
    assert_case(r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2"#, r#"x^2"#);
  }
  #[test]
  fn integer_literal_8() {
    assert_case(r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1"#, r#"1"#);
  }
  #[test]
  fn symbol_literal_2() {
    assert_case(r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x"#, r#"x"#);
  }
  #[test]
  fn power_8() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1"#,
      r#"x"#,
    );
  }
  #[test]
  fn power_9() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1; x^2"#,
      r#"x^2"#,
    );
  }
  #[test]
  fn integer_literal_9() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1; x^2; 1"#,
      r#"1"#,
    );
  }
  #[test]
  fn symbol_literal_3() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1; x^2; 1; x"#,
      r#"x"#,
    );
  }
  #[test]
  fn plus_11() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1; x^2; 1; x; 1+x"#,
      r#"1 + x"#,
    );
  }
  #[test]
  fn plus_12() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1; x^2; 1; x; 1+x; 1+2*x"#,
      r#"1 + 2*x"#,
    );
  }
  #[test]
  fn integer_literal_10() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1; x^2; 1; x; 1+x; 1+2*x; 1"#,
      r#"1"#,
    );
  }
  #[test]
  fn symbol_literal_4() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1; x^2; 1; x; 1+x; 1+2*x; 1; x"#,
      r#"x"#,
    );
  }
  #[test]
  fn times_5() {
    assert_case(
      r#"1; x; 2*x; 1+x; 1+2*x; 1; x; x^1; x^2; 1; x; x^1; x^2; 1; x; 1+x; 1+2*x; 1; x; 2*x"#,
      r#"2*x"#,
    );
  }
  #[test]
  fn greater_9() {
    assert_case(r#"a + b /. x_ + y_ -> {x, y}"#, r#"{a, b}"#);
  }
  #[test]
  fn integer_literal_11() {
    assert_case(r#"0"#, r#"0"#);
  }
  #[test]
  fn minus_10() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2"#,
      r#"0."#,
    );
  }
  #[test]
  fn minus_11() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20"#,
      r#"0."#,
    );
  }
  #[test]
  fn minus_12() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2"#,
      r#"0``2."#,
    );
  }
  #[test]
  fn minus_13() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20"#,
      r#"0``20."#,
    );
  }
  #[test]
  fn integer_literal_12() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10"#,
      r#"10"#,
    );
  }
  #[test]
  fn real_literal_1() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10."#,
      r#"10."#,
    );
  }
  #[test]
  fn real_literal_2() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00"#,
      r#"10."#,
    );
  }
  #[test]
  fn expression_3() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`"#,
      r#"10."#,
    );
  }
  #[test]
  fn expression_4() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2"#,
      r#"10.`2."#,
    );
  }
  #[test]
  fn expression_5() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20"#,
      r#"10.`20."#,
    );
  }
  #[test]
  fn real_literal_3() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000"#,
      r#"10.`21."#,
    );
  }
  #[test]
  fn expression_6() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2"#,
      r#"10.`3."#,
    );
  }
  #[test]
  fn expression_7() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20"#,
      r#"10.`21."#,
    );
  }
  #[test]
  fn expression_8() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I"#,
      r#"0. + 1.*I"#,
    );
  }
  #[test]
  fn plus_13() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I"#,
      r#"0.4 + 2.4*I"#,
    );
  }
  #[test]
  fn plus_14() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I; 2 + 3 I"#,
      r#"2 + 3*I"#,
    );
  }
  #[test]
  fn string_literal_3() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20; 1. I;  0.4 + 2.4 I; 2 + 3 I; "abc""#,
      r#""abc""#,
    );
  }
  #[test]
  fn integer_literal_13() {
    assert_case(r#"0"#, r#"0"#);
  }
  #[test]
  fn minus_14() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2"#,
      r#"0."#,
    );
  }
  #[test]
  fn minus_15() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20"#,
      r#"0."#,
    );
  }
  #[test]
  fn minus_16() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2"#,
      r#"0``2."#,
    );
  }
  #[test]
  fn minus_17() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20"#,
      r#"0``20."#,
    );
  }
  #[test]
  fn integer_literal_14() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10"#,
      r#"10"#,
    );
  }
  #[test]
  fn real_literal_4() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10."#,
      r#"10."#,
    );
  }
  #[test]
  fn real_literal_5() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00"#,
      r#"10."#,
    );
  }
  #[test]
  fn expression_9() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`"#,
      r#"10."#,
    );
  }
  #[test]
  fn expression_10() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2"#,
      r#"10.`2."#,
    );
  }
  #[test]
  fn expression_11() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20"#,
      r#"10.`20."#,
    );
  }
  #[test]
  fn real_literal_6() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000"#,
      r#"10.`21."#,
    );
  }
  #[test]
  fn expression_12() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2"#,
      r#"10.`3."#,
    );
  }
  #[test]
  fn expression_13() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20"#,
      r#"10.`21."#,
    );
  }
  #[test]
  fn plus_15() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I"#,
      r#"0.4 + 2.4*I"#,
    );
  }
  #[test]
  fn plus_16() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I"#,
      r#"2 + 3*I"#,
    );
  }
  #[test]
  fn string_literal_4() {
    assert_case(
      r#"0; 0.; 0.00; 0.00`; 0.00`2; 0.00`20; 0.00000000000000000000; 0.``2; 0.``20; -0.`2; -0.`20; -0.``2; -0.``20; 10; 10.; 10.00; 10.00`; 10.00`2; 10.00`20; 10.00000000000000000000; 10.``2; 10.``20;  0.4 + 2.4 I; 2 + 3 I; "abc""#,
      r#""abc""#,
    );
  }
  #[test]
  fn plus_17() {
    assert_case(r#"1. + 2. + 3."#, r#"6."#);
  }
  #[test]
  fn plus_18() {
    assert_case(r#"1. + 2. + 3.; 1 + 2/3 + 3/5"#, r#"34 / 15"#);
  }
  #[test]
  fn plus_19() {
    assert_case(
      r#"1. + 2. + 3.; 1 + 2/3 + 3/5; 1 - 2/3 + 3/5"#,
      r#"14 / 15"#,
    );
  }
  #[test]
  fn plus_20() {
    assert_case(
      r#"1. + 2. + 3.; 1 + 2/3 + 3/5; 1 - 2/3 + 3/5; 1. - 2/3 + 3/5"#,
      r#"0.9333333333333333"#,
    );
  }
  #[test]
  fn plus_21() {
    assert_case(
      r#"1. + 2. + 3.; 1 + 2/3 + 3/5; 1 - 2/3 + 3/5; 1. - 2/3 + 3/5; 1 - 2/3 + 2 I"#,
      r#"1/3 + 2*I"#,
    );
  }
  #[test]
  fn plus_22() {
    assert_case(
      r#"1. + 2. + 3.; 1 + 2/3 + 3/5; 1 - 2/3 + 3/5; 1. - 2/3 + 3/5; 1 - 2/3 + 2 I; 1. - 2/3 + 2 I"#,
      r#"0.33333333333333337 + 2.*I"#,
    );
  }
  #[test]
  fn plus_23() {
    assert_case(
      r#"1. + 2. + 3.; 1 + 2/3 + 3/5; 1 - 2/3 + 3/5; 1. - 2/3 + 3/5; 1 - 2/3 + 2 I; 1. - 2/3 + 2 I; a + 2 a + 3 a q"#,
      r#"3*a + 3*a*q"#,
    );
  }
  #[test]
  fn plus_24() {
    assert_case(
      r#"1. + 2. + 3.; 1 + 2/3 + 3/5; 1 - 2/3 + 3/5; 1. - 2/3 + 3/5; 1 - 2/3 + 2 I; 1. - 2/3 + 2 I; a + 2 a + 3 a q; a - 2 a + 3 a q"#,
      r#"-a + 3*a*q"#,
    );
  }
  #[test]
  fn plus_25() {
    assert_case(
      r#"1. + 2. + 3.; 1 + 2/3 + 3/5; 1 - 2/3 + 3/5; 1. - 2/3 + 3/5; 1 - 2/3 + 2 I; 1. - 2/3 + 2 I; a + 2 a + 3 a q; a - 2 a + 3 a q; a - (5+ a+ 2 b) + 3 a q"#,
      r#"-5 - 2*b + 3*a*q"#,
    );
  }
  #[test]
  fn plus_26() {
    assert_case(
      r#"1. + 2. + 3.; 1 + 2/3 + 3/5; 1 - 2/3 + 3/5; 1. - 2/3 + 3/5; 1 - 2/3 + 2 I; 1. - 2/3 + 2 I; a + 2 a + 3 a q; a - 2 a + 3 a q; a - (5+ a+ 2 b) + 3 a q; a - 2 (5+ a+ 2 b) + 3 a q"#,
      r#"a - 2*(5 + a + 2*b) + 3*a*q"#,
    );
  }
  #[test]
  fn times_6() {
    assert_case(r#"1.  2.  3.; 1 * 2/3 * 3/5"#, r#"2 / 5"#);
  }
  #[test]
  fn minus_18() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5)"#,
      r#"-2 / 5"#,
    );
  }
  #[test]
  fn minus_19() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5)"#,
      r#"-0.39999999999999997"#,
    );
  }
  #[test]
  fn minus_20() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I)"#,
      r#"(-4*I)/3"#,
    );
  }
  #[test]
  fn minus_21() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I)"#,
      r#"0. - 1.3333333333333333*I"#,
    );
  }
  #[test]
  fn expression_14() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q)"#,
      r#"6*a^3*q"#,
    );
  }
  #[test]
  fn power_10() {
    assert_case(r#"2^0"#, r#"1"#);
  }
  #[test]
  fn divide_22() {
    assert_case(r#"2^0; (2/3)^0"#, r#"1"#);
  }
  #[test]
  fn power_11() {
    assert_case(r#"2^0; (2/3)^0; 2.^0"#, r#"1."#);
  }
  #[test]
  fn power_12() {
    assert_case(r#"2^0; (2/3)^0; 2.^0; 2^1"#, r#"2"#);
  }
  #[test]
  fn divide_23() {
    assert_case(r#"2^0; (2/3)^0; 2.^0; 2^1; (2/3)^1"#, r#"2 / 3"#);
  }
  #[test]
  fn power_13() {
    assert_case(r#"2^0; (2/3)^0; 2.^0; 2^1; (2/3)^1; 2.^1"#, r#"2."#);
  }
  #[test]
  fn power_14() {
    assert_case(r#"2^0; (2/3)^0; 2.^0; 2^1; (2/3)^1; 2.^1; 2^(3)"#, r#"8"#);
  }
  #[test]
  fn divide_24() {
    assert_case(
      r#"2^0; (2/3)^0; 2.^0; 2^1; (2/3)^1; 2.^1; 2^(3); (1/2)^3"#,
      r#"1 / 8"#,
    );
  }
  #[test]
  fn minus_22() {
    assert_case(
      r#"2^0; (2/3)^0; 2.^0; 2^1; (2/3)^1; 2.^1; 2^(3); (1/2)^3; 2^(-3)"#,
      r#"1 / 8"#,
    );
  }
  #[test]
  fn minus_23() {
    assert_case(
      r#"2^0; (2/3)^0; 2.^0; 2^1; (2/3)^1; 2.^1; 2^(3); (1/2)^3; 2^(-3); (1/2)^(-3)"#,
      r#"8"#,
    );
  }
  #[test]
  fn minus_24() {
    assert_case(r#"(-I)^(2/3)"#, r#"-(-1)^(2/3)"#);
  }
  // I^(p/q) = (-1)^(p/(2q)), Wolfram's canonical form for the imaginary unit.
  #[test]
  fn imaginary_unit_rational_power() {
    assert_case(r#"I^(1/2)"#, r#"(-1)^(1/4)"#);
    assert_case(r#"I^(1/3)"#, r#"(-1)^(1/6)"#);
    assert_case(r#"I^(2/3)"#, r#"(-1)^(1/3)"#);
    assert_case(r#"I^(3/2)"#, r#"(-1)^(3/4)"#);
    assert_case(r#"I^(5/2)"#, r#"-(-1)^(1/4)"#);
    assert_case(r#"I^(-1/2)"#, r#"-(-1)^(3/4)"#);
  }
  #[test]
  fn sqrt_of_imaginary_unit() {
    assert_case(r#"Sqrt[I]"#, r#"(-1)^(1/4)"#);
    assert_case(r#"Sqrt[-I]"#, r#"-(-1)^(3/4)"#);
    // Round-trips: (Sqrt[I])^2 = I.
    assert_case(r#"Sqrt[I]^2"#, r#"I"#);
  }
  #[test]
  fn power_15() {
    assert_case(r#"(a^"w")^2"#, r#"a^(2*"w")"#);
  }
  #[test]
  fn divide_25() {
    assert_case(r#"(a^(1/2))^3."#, r#"a ^ 1.5"#);
  }
  #[test]
  fn power_16() {
    assert_case(r#"(a^(.3))^3."#, r#"a^0.8999999999999999"#);
  }
  #[test]
  fn harmonic_number() {
    assert_case(r#"HarmonicNumber[-1.5]"#, r#"0.6137056388801093"#);
  }
  #[test]
  fn integer_literal_15() {
    assert_case(r#"1234567890"#, r#"1234567890"#);
  }
  #[test]
  fn integer_literal_16() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890"#,
      r#"-1234567890"#,
    );
  }
  #[test]
  fn integer_literal_17() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890"#,
      r#"-1234567890"#,
    );
  }
  #[test]
  fn integer_literal_18() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890"#,
      r#"-1234567890"#,
    );
  }
  #[test]
  fn integer_literal_19() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890"#,
      r#"-1234567890"#,
    );
  }
  #[test]
  fn integer_literal_20() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890"#,
      r#"-1234567890"#,
    );
  }
  #[test]
  fn integer_literal_21() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890"#,
      r#"-9934567890"#,
    );
  }
  #[test]
  fn integer_literal_22() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321"#,
      r#"12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_23() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890"#,
      r#"-1234567890"#,
    );
  }
  #[test]
  fn integer_literal_24() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321"#,
      r#"12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_25() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321"#,
      r#"12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_26() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329"#,
      r#"92345678900987654329"#,
    );
  }
  #[test]
  fn integer_literal_27() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321"#,
      r#"12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_28() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321"#,
      r#"12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_29() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321"#,
      r#"12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_30() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321"#,
      r#"12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_31() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321"#,
      r#"-12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_32() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321"#,
      r#"-12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_33() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321"#,
      r#"-12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_34() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321"#,
      r#"-12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_35() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321"#,
      r#"-12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_36() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321"#,
      r#"-12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_37() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -99345678900987654321"#,
      r#"-99345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_38() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -99345678900987654321; -12345678900987654321"#,
      r#"-12345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_39() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -99345678900987654321; -12345678900987654321; -99345678900987654321"#,
      r#"-99345678900987654321"#,
    );
  }
  #[test]
  fn integer_literal_40() {
    assert_case(
      r#"1234567890; 1234567890; 1234567890; 1234567890; 9934567890; 1234567890; 1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -1234567890; -9934567890; 12345678900987654321; -1234567890; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 92345678900987654329; 12345678900987654321; 12345678900987654321; 12345678900987654321; 12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -12345678900987654321; -99345678900987654321; -12345678900987654321; -99345678900987654321; -12345678900987654321"#,
      r#"-12345678900987654321"#,
    );
  }
}

mod set_precision {
  use woxi::interpret;

  #[test]
  fn integer_to_bigfloat() {
    assert_eq!(interpret("SetPrecision[3, 20]").unwrap(), "3.`20.");
  }

  #[test]
  fn rational_to_bigfloat() {
    assert_eq!(interpret("SetPrecision[3/4, 20]").unwrap(), "0.75`20.");
  }

  #[test]
  fn real_expands_to_binary_form() {
    // 2.1 in IEEE 754 double is 2.10000000000000008881784197001252323389...
    // wolframscript emits 38 fractional digits at precision 20.
    assert_eq!(
      interpret("SetPrecision[2.1, 20]").unwrap(),
      "2.10000000000000008881784197001252323389`20."
    );
  }

  #[test]
  fn list_maps_elementwise() {
    assert_eq!(
      interpret("SetPrecision[{1, 2, 3}, 20]").unwrap(),
      "{1.`20., 2.`20., 3.`20.}"
    );
  }

  #[test]
  fn symbolic_sum_converts_only_numeric_leaves() {
    // a + 2*b + 3*c → a + 2.`20.*b + 3.`20.*c
    assert_eq!(
      interpret("SetPrecision[a + 2*b + 3*c, 20]").unwrap(),
      "a + 2.`20.*b + 3.`20.*c"
    );
  }

  // A symbolic real constant numericizes to the requested precision (like
  // N[c, p]) rather than passing through unchanged. A bare symbol still
  // passes through. Verified against wolframscript.
  #[test]
  fn symbolic_constant_numericizes() {
    assert_eq!(
      interpret("SetPrecision[Pi, 5]").unwrap(),
      "3.1415926535897932385`5."
    );
    assert_eq!(
      interpret("SetPrecision[GoldenRatio, 10]").unwrap(),
      "1.61803398874989484820458683436563811772`10."
    );
    // A bare symbol is unaffected.
    assert_eq!(interpret("SetPrecision[x, 10]").unwrap(), "x");
  }

  #[test]
  fn machine_precision_demotes_numbers() {
    // SetPrecision[a + 2*b + 3*c, MachinePrecision] → a + 2.*b + 3.*c
    assert_eq!(
      interpret("SetPrecision[a + 2*b + 3*c, MachinePrecision]").unwrap(),
      "a + 2.*b + 3.*c"
    );
  }

  #[test]
  fn machine_precision_integer() {
    assert_eq!(
      interpret("SetPrecision[3, MachinePrecision]").unwrap(),
      "3."
    );
  }
}

mod set_accuracy {
  use woxi::interpret;

  // SetAccuracy[x, a] gives x at precision a + Log10[|x|]. An integer power of
  // ten makes the precision tag exact: 100 -> precision 3 + 2 = 5.
  #[test]
  fn integer_power_of_ten() {
    assert_eq!(interpret("SetAccuracy[100, 3]").unwrap(), "100.`5.");
  }

  // x = 1 has Log10 = 0, so accuracy and precision coincide.
  #[test]
  fn unit_value_precision_equals_accuracy() {
    assert_eq!(interpret("SetAccuracy[1.0, 20]").unwrap(), "1.`20.");
  }

  // A machine real keeps its full binary expansion at the derived precision.
  #[test]
  fn real_value() {
    assert_eq!(
      interpret("SetAccuracy[123.456, 2]").unwrap(),
      "123.4560000000000030695`4.091512201627772"
    );
  }

  // Zero carries the accuracy directly in the `0``a` form.
  #[test]
  fn zero_uses_accuracy_form() {
    assert_eq!(interpret("SetAccuracy[0, 5]").unwrap(), "0``5.");
  }

  // Maps elementwise; each element's precision reflects its own magnitude.
  #[test]
  fn list_maps_elementwise() {
    assert_eq!(
      interpret("SetAccuracy[{1, 2, 3}, 4]").unwrap(),
      "{1.`4., 2.`4.301029995663981, 3.`4.477121254719663}"
    );
  }

  // Numeric leaves convert; bare symbols pass through.
  #[test]
  fn symbolic_sum_converts_only_numeric_leaves() {
    assert_eq!(
      interpret("SetAccuracy[a + 100*b, 3]").unwrap(),
      "a + 100.`5.*b"
    );
  }
}

mod number_digit_extended {
  use super::*;

  #[test]
  fn exact_rationals_and_constants() {
    // Previously unevaluated: rationals and symbolic numeric constants
    assert_eq!(interpret("NumberDigit[5/7, -2]").unwrap(), "1");
    assert_eq!(interpret("NumberDigit[1/3, -10]").unwrap(), "3");
    assert_eq!(interpret("NumberDigit[1/7, -7]").unwrap(), "1");
    assert_eq!(interpret("NumberDigit[-5/7, -1]").unwrap(), "7");
    assert_eq!(interpret("NumberDigit[Pi, -3]").unwrap(), "1");
    assert_eq!(interpret("NumberDigit[Pi, 0]").unwrap(), "3");
    assert_eq!(interpret("NumberDigit[E, -4]").unwrap(), "2");
    assert_eq!(interpret("NumberDigit[Sqrt[2], -5]").unwrap(), "1");
    // Deep digits via arbitrary precision
    assert_eq!(interpret("NumberDigit[Pi, -50]").unwrap(), "0");
  }

  #[test]
  fn base_form() {
    assert_eq!(interpret("NumberDigit[255, 1, 16]").unwrap(), "15");
    assert_eq!(interpret("NumberDigit[255, 0, 16]").unwrap(), "15");
    assert_eq!(interpret("NumberDigit[10, 2, 2]").unwrap(), "0");
    assert_eq!(interpret("NumberDigit[2^200, 60, 2]").unwrap(), "0");
    assert_eq!(
      interpret("NumberDigit[100, 1, 1]").unwrap(),
      "NumberDigit[100, 1, 1]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "NumberDigit::rbase: Base 1 is not a real number greater than 1."
      )),
      "expected rbase message, got {:?}",
      msgs
    );
  }

  #[test]
  fn list_positions() {
    assert_eq!(
      interpret("NumberDigit[123.456, {2, 1, 0, -1}]").unwrap(),
      "{1, 2, 3, 4}"
    );
    assert_eq!(
      interpret("NumberDigit[2/3, {-1, -2, -3}]").unwrap(),
      "{6, 6, 6}"
    );
  }

  #[test]
  fn machine_precision_indeterminate() {
    // Digits finer than one ulp of the machine real are Indeterminate
    assert_eq!(interpret("NumberDigit[123.456, -13]").unwrap(), "0");
    assert_eq!(
      interpret("NumberDigit[123.456, -14]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(interpret("NumberDigit[0.1, -17]").unwrap(), "Indeterminate");
    assert_eq!(
      interpret("NumberDigit[1000000.0, -10]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(interpret("NumberDigit[1000000.0, -9]").unwrap(), "0");
  }

  #[test]
  fn messages() {
    // Non-integer position: badspec (previously a hard error)
    assert_eq!(
      interpret("NumberDigit[123.456, 1.5]").unwrap(),
      "NumberDigit[123.456, 1.5]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "NumberDigit::badspec: Argument 1.5 at position 2 should be an integer or a list of integers."
      )),
      "expected badspec message, got {:?}",
      msgs
    );
    assert_eq!(interpret("NumberDigit[x, 1]").unwrap(), "NumberDigit[x, 1]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs
        .iter()
        .any(|m| m
          .contains("NumberDigit::num: Argument x should be a number.")),
      "expected num message, got {:?}",
      msgs
    );
  }
}

// Mod[m, n] with exact real (but non-rational) operands must stay exact,
// following Mod[m, n] = m - n*Floor[m/n], rather than collapsing to a
// machine-precision real. Regression for `Mod[GoldenRatio, 1]` returning
// 0.618... instead of `-1 + GoldenRatio`.
mod mod_exact_symbolic {
  use super::*;

  #[test]
  fn golden_ratio_mod_one() {
    assert_eq!(
      interpret("Mod[GoldenRatio, 1]").unwrap(),
      "-1 + GoldenRatio"
    );
  }

  #[test]
  fn pi_mod_integer() {
    assert_eq!(interpret("Mod[Pi, 1]").unwrap(), "-3 + Pi");
    assert_eq!(interpret("Mod[Pi, 2]").unwrap(), "-2 + Pi");
  }

  #[test]
  fn e_mod_one() {
    assert_eq!(interpret("Mod[E, 1]").unwrap(), "-2 + E");
    assert_eq!(interpret("Mod[2 E, 1]").unwrap(), "-5 + 2*E");
  }

  #[test]
  fn sqrt_mod_one() {
    assert_eq!(interpret("Mod[Sqrt[2], 1]").unwrap(), "-1 + Sqrt[2]");
    assert_eq!(interpret("Mod[Sqrt[10], 2]").unwrap(), "-2 + Sqrt[10]");
  }

  #[test]
  fn integer_mod_irrational() {
    assert_eq!(interpret("Mod[5, Pi]").unwrap(), "5 - Pi");
  }

  #[test]
  fn negative_irrational_dividend() {
    assert_eq!(interpret("Mod[-Pi, 2]").unwrap(), "4 - Pi");
    assert_eq!(interpret("Mod[-E, 1]").unwrap(), "3 - E");
  }

  // Exact cancellation: the floor is taken after the symbolic quotient
  // simplifies, so multiples of the modulus reduce to exact zero / fraction.
  #[test]
  fn commensurate_irrationals_cancel() {
    assert_eq!(interpret("Mod[2 Pi, Pi]").unwrap(), "0");
    assert_eq!(interpret("Mod[3 Pi/2, Pi]").unwrap(), "Pi/2");
    assert_eq!(interpret("Mod[5 Pi, 2 Pi]").unwrap(), "Pi");
    assert_eq!(interpret("Mod[10 E, E]").unwrap(), "0");
  }

  // Modulus larger than the dividend leaves it unchanged (Floor = 0).
  #[test]
  fn dividend_below_modulus() {
    assert_eq!(interpret("Mod[GoldenRatio, 2]").unwrap(), "GoldenRatio");
    assert_eq!(interpret("Mod[EulerGamma, 1]").unwrap(), "EulerGamma");
  }

  // Machine-real operands keep the numeric (inexact) result.
  #[test]
  fn machine_real_stays_numeric() {
    assert_eq!(interpret("Mod[10.5, 3]").unwrap(), "1.5");
    assert_eq!(interpret("Mod[Pi, 2.0]").unwrap(), "1.1415926535897931");
  }

  // Purely symbolic (non-numeric) arguments stay unevaluated.
  #[test]
  fn non_numeric_symbol_unevaluated() {
    assert_eq!(interpret("Mod[a, 1]").unwrap(), "Mod[a, 1]");
    assert_eq!(interpret("Mod[x, 2]").unwrap(), "Mod[x, 2]");
  }

  // Three-argument offset form also stays exact.
  #[test]
  fn three_arg_exact() {
    assert_eq!(interpret("Mod[Pi, 2, -1]").unwrap(), "-4 + Pi");
    assert_eq!(interpret("Mod[Pi, 1, 1]").unwrap(), "-2 + Pi");
    assert_eq!(interpret("Mod[7 Pi/2, Pi, 1]").unwrap(), "Pi/2");
  }
}

// Division-by-zero behavior for Mod / Quotient / QuotientRemainder.
// Verified against wolframscript (InputForm).
mod zero_divisor {
  use super::*;

  #[test]
  fn real_division_by_zero_does_not_error() {
    // Regression: real / 0 used to raise a hard "Division by zero" error
    // instead of returning ComplexInfinity (or Indeterminate for 0./0) with
    // the Power::infy message, matching wolframscript's `/`-operator behaviour.
    assert_eq!(interpret("1.5/0").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Divide[1.5, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("0./0").unwrap(), "Indeterminate");
    // Integer forms keep working.
    assert_eq!(interpret("5/0").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("0/0").unwrap(), "Indeterminate");
    assert_eq!(interpret("a/0").unwrap(), "ComplexInfinity");
    // The Power::infy reciprocal message is emitted (not a thrown error).
    let r = interpret_with_stdout("1.5/0").unwrap();
    assert_eq!(r.result, "ComplexInfinity");
    assert!(
      r.warnings
        .iter()
        .any(|w| w.contains("Power::infy: Infinite expression")),
      "expected Power::infy message, got: {:?}",
      r.warnings
    );
    // 0./0 names the real dividend in the indeterminate message.
    let r = interpret_with_stdout("0./0").unwrap();
    assert_eq!(r.result, "Indeterminate");
    assert!(r.warnings.iter().any(|w| w.contains(
      "Infinity::indet: Indeterminate expression 0. ComplexInfinity encountered."
    )));
  }

  #[test]
  fn explicit_divide_head_uses_divide_infy_message() {
    // wolframscript distinguishes the explicit `Divide[]` head from the `/`
    // operator when a *numeric* numerator is divided by zero: the head reports
    // `Divide::infy` (and keeps the literal numerator in the 2D fraction),
    // whereas `/` desugars to `n*(1/0)` and reports `Power::infy` with a `1/0`
    // box. Verified against wolframscript.
    let pfx = "Divide::infy: Infinite expression ";
    let lead = " ".repeat(pfx.len());

    let r = interpret_with_stdout("Divide[5, 0]").unwrap();
    assert_eq!(r.result, "ComplexInfinity");
    let expected = format!("{lead}5\n{pfx}- encountered.\n{lead}0");
    assert!(
      r.warnings.iter().any(|w| *w == expected),
      "Divide[5,0] message mismatch, got: {:?}",
      r.warnings
    );

    // Negative / real numerators widen the dash run to the numerator width.
    let r = interpret_with_stdout("Divide[-3, 0]").unwrap();
    assert_eq!(r.result, "ComplexInfinity");
    let expected = format!("{lead}-3\n{pfx}-- encountered.\n{lead}0");
    assert!(
      r.warnings.iter().any(|w| *w == expected),
      "Divide[-3,0] message mismatch, got: {:?}",
      r.warnings
    );

    // 0/0 via the head is the indeterminate form with the Divide::indet tag.
    let r = interpret_with_stdout("Divide[0, 0]").unwrap();
    assert_eq!(r.result, "Indeterminate");
    let ipfx = "Divide::indet: Indeterminate expression ";
    let ilead = " ".repeat(ipfx.len());
    let expected = format!("{ilead}0\n{ipfx}- encountered.\n{ilead}0");
    assert!(
      r.warnings.iter().any(|w| *w == expected),
      "Divide[0,0] message mismatch, got: {:?}",
      r.warnings
    );

    // A symbolic numerator decays to `x*(1/0)` and keeps the Power::infy form.
    let r = interpret_with_stdout("Divide[x, 0]").unwrap();
    assert_eq!(r.result, "ComplexInfinity");
    assert!(
      r.warnings
        .iter()
        .any(|w| w.contains("Power::infy: Infinite expression")),
      "Divide[x,0] should fall through to Power::infy, got: {:?}",
      r.warnings
    );
    assert!(
      !r.warnings.iter().any(|w| w.contains("Divide::infy")),
      "Divide[x,0] should not emit Divide::infy, got: {:?}",
      r.warnings
    );

    // The `/` operator stays Power::infy even with a numeric numerator.
    let r = interpret_with_stdout("5/0").unwrap();
    assert_eq!(r.result, "ComplexInfinity");
    assert!(
      r.warnings
        .iter()
        .any(|w| w.contains("Power::infy: Infinite expression")),
      "5/0 operator should emit Power::infy, got: {:?}",
      r.warnings
    );
    assert!(
      !r.warnings.iter().any(|w| w.contains("Divide::infy")),
      "5/0 operator should not emit Divide::infy, got: {:?}",
      r.warnings
    );
  }

  #[test]
  fn power_zero_negative_renders_full_box() {
    // Regression: `Power[0, -1]` previously printed the `1` numerator with a
    // mis-indented box and dropped the `0` denominator. It must render the full
    // `1/0` fraction, and `Power[0, -n]` (n != 1) must render `0` with the
    // exponent as a superscript. Verified against wolframscript.
    let pfx = "Power::infy: Infinite expression ";
    let lead = " ".repeat(pfx.len());

    let r = interpret_with_stdout("Power[0, -1]").unwrap();
    assert_eq!(r.result, "ComplexInfinity");
    let expected = format!("{lead}1\n{pfx}- encountered.\n{lead}0");
    assert!(
      r.warnings.iter().any(|w| *w == expected),
      "Power[0,-1] message mismatch, got: {:?}",
      r.warnings
    );

    // 0^-2 → superscript `-2` above the base `0`, a two-line box (no fraction
    // denominator line).
    let r = interpret_with_stdout("Power[0, -2]").unwrap();
    assert_eq!(r.result, "ComplexInfinity");
    let sup_lead = " ".repeat(pfx.len() + 1);
    let expected = format!("{sup_lead}-2\n{pfx}0   encountered.");
    assert!(
      r.warnings.iter().any(|w| *w == expected),
      "Power[0,-2] superscript box mismatch, got: {:?}",
      r.warnings
    );
  }

  #[test]
  fn infinity_indeterminate_forms_emit_message() {
    // Indeterminate forms involving Infinity return Indeterminate AND emit the
    // Infinity::indet message (Woxi previously returned Indeterminate silently).
    let plus_cases = ["Infinity - Infinity", "2 Infinity - Infinity"];
    for input in plus_cases {
      let r = interpret_with_stdout(input).unwrap();
      assert_eq!(r.result, "Indeterminate", "result for {input}");
      assert!(
        r.warnings.iter().any(|w| w.contains(
          "Infinity::indet: Indeterminate expression -Infinity + Infinity \
           encountered."
        )),
        "missing -Infinity + Infinity message for {input}: {:?}",
        r.warnings
      );
    }
    let times_cases = ["0 * Infinity", "Infinity / Infinity", "Infinity*0"];
    for input in times_cases {
      let r = interpret_with_stdout(input).unwrap();
      assert_eq!(r.result, "Indeterminate", "result for {input}");
      assert!(
        r.warnings.iter().any(|w| w.contains(
          "Infinity::indet: Indeterminate expression 0 Infinity encountered."
        )),
        "missing 0 Infinity message for {input}: {:?}",
        r.warnings
      );
    }
    // ComplexInfinity names itself in the message.
    let r = interpret_with_stdout("0 * ComplexInfinity").unwrap();
    assert_eq!(r.result, "Indeterminate");
    assert!(r.warnings.iter().any(|w| w.contains(
      "Infinity::indet: Indeterminate expression 0 ComplexInfinity encountered."
    )));
    // Indeterminate propagation (already-Indeterminate operand) does NOT
    // re-emit the message.
    let r = interpret_with_stdout("Indeterminate + 1").unwrap();
    assert_eq!(r.result, "Indeterminate");
    assert!(!r.warnings.iter().any(|w| w.contains("Infinity::indet")));
  }

  #[test]
  fn mod_by_zero_is_indeterminate() {
    assert_eq!(interpret("Mod[3, 0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Mod[0, 0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Mod[3.5, 0]").unwrap(), "Indeterminate");
    // Symbolic dividend is still Indeterminate, not left as Mod[a, 0].
    assert_eq!(interpret("Mod[a, 0]").unwrap(), "Indeterminate");
  }

  #[test]
  fn mod3_by_zero_is_indeterminate() {
    assert_eq!(interpret("Mod[3, 0, 2]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Mod[a, 0, 2]").unwrap(), "Indeterminate");
  }

  #[test]
  fn quotient_by_zero_is_complex_infinity() {
    assert_eq!(interpret("Quotient[3, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Quotient[-5, 0]").unwrap(), "ComplexInfinity");
    // Symbolic dividend is treated as non-zero -> ComplexInfinity.
    assert_eq!(interpret("Quotient[a, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn quotient_zero_over_zero_is_indeterminate() {
    assert_eq!(interpret("Quotient[0, 0]").unwrap(), "Indeterminate");
  }

  #[test]
  fn quotient3_by_zero() {
    // Effective numerator (n - d): non-zero -> ComplexInfinity, zero -> Indeterminate.
    assert_eq!(interpret("Quotient[3, 0, 1]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Quotient[5, 0, 1]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Quotient[1, 0, 1]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Quotient[0, 0, 0]").unwrap(), "Indeterminate");
  }

  #[test]
  fn quotient_remainder_by_zero_stays_symbolic() {
    assert_eq!(
      interpret("QuotientRemainder[3, 0]").unwrap(),
      "QuotientRemainder[3, 0]"
    );
    assert_eq!(
      interpret("QuotientRemainder[0, 0]").unwrap(),
      "QuotientRemainder[0, 0]"
    );
    assert_eq!(
      interpret("QuotientRemainder[a, 0]").unwrap(),
      "QuotientRemainder[a, 0]"
    );
  }
}

// Binomial[k, k] = 1 (any k) and LCM[] identity, verified against wolframscript.
mod binomial_diagonal_and_lcm_empty {
  use super::*;

  #[test]
  fn binomial_diagonal_symbolic() {
    assert_eq!(interpret("Binomial[n, n]").unwrap(), "1");
    assert_eq!(interpret("Binomial[n + 1, n + 1]").unwrap(), "1");
    assert_eq!(interpret("Binomial[2 n, 2 n]").unwrap(), "1");
    assert_eq!(interpret("Binomial[a, a]").unwrap(), "1");
    assert_eq!(interpret("Binomial[Pi, Pi]").unwrap(), "1");
  }

  #[test]
  fn binomial_diagonal_exact_numeric() {
    assert_eq!(interpret("Binomial[5, 5]").unwrap(), "1");
    assert_eq!(interpret("Binomial[1/2, 1/2]").unwrap(), "1");
    assert_eq!(interpret("Binomial[-3, -3]").unwrap(), "1");
  }

  #[test]
  fn binomial_diagonal_inexact() {
    // Machine reals yield 1. (Real), not the exact integer.
    assert_eq!(interpret("Binomial[0.5, 0.5]").unwrap(), "1.");
    assert_eq!(interpret("Binomial[2.5, 2.5]").unwrap(), "1.");
  }

  #[test]
  fn binomial_offdiagonal_unaffected() {
    assert_eq!(interpret("Binomial[10, 3]").unwrap(), "120");
    assert_eq!(interpret("Binomial[n, 2]").unwrap(), "((-1 + n)*n)/2");
    assert_eq!(interpret("Binomial[n, 0]").unwrap(), "1");
  }

  #[test]
  fn lcm_empty_stays_symbolic() {
    // LCM[] is left unevaluated (unlike GCD[], which is 0).
    assert_eq!(interpret("LCM[]").unwrap(), "LCM[]");
    assert_eq!(interpret("GCD[]").unwrap(), "0");
    assert_eq!(interpret("LCM[4, 6]").unwrap(), "12");
  }
}

// Abs pulls real-constant factors out of a product (as their magnitude),
// drops |I| = 1, and is idempotent. All verified against wolframscript.
mod abs_simplifications {
  use super::*;

  #[test]
  fn abs_negative_real_constant_stays_exact() {
    // Regression: Abs[-Pi] used to numericize to 3.14159...
    assert_eq!(interpret("Abs[-Pi]").unwrap(), "Pi");
    assert_eq!(interpret("Abs[-Sqrt[2]]").unwrap(), "Sqrt[2]");
  }

  #[test]
  fn abs_pulls_real_coefficient() {
    assert_eq!(interpret("Abs[2 x]").unwrap(), "2*Abs[x]");
    assert_eq!(interpret("Abs[-2 x]").unwrap(), "2*Abs[x]");
    assert_eq!(interpret("Abs[-3 x y]").unwrap(), "3*Abs[x*y]");
    assert_eq!(interpret("Abs[-Pi x]").unwrap(), "Pi*Abs[x]");
    assert_eq!(interpret("Abs[x/2]").unwrap(), "Abs[x]/2");
  }

  #[test]
  fn abs_negation_removed() {
    assert_eq!(interpret("Abs[-x]").unwrap(), "Abs[x]");
    assert_eq!(interpret("Abs[-a]").unwrap(), "Abs[a]");
  }

  #[test]
  fn abs_imaginary_unit_dropped() {
    assert_eq!(interpret("Abs[I x]").unwrap(), "Abs[x]");
    assert_eq!(interpret("Abs[-I x]").unwrap(), "Abs[x]");
  }

  #[test]
  fn abs_idempotent() {
    assert_eq!(interpret("Abs[Abs[x]]").unwrap(), "Abs[x]");
  }

  // Abs[base^exp] = Abs[base]^exp for a real numeric exponent.
  #[test]
  fn abs_of_power() {
    assert_eq!(interpret("Abs[x^2]").unwrap(), "Abs[x]^2");
    assert_eq!(interpret("Abs[x^3]").unwrap(), "Abs[x]^3");
    assert_eq!(interpret("Abs[x^(1/2)]").unwrap(), "Sqrt[Abs[x]]");
    assert_eq!(interpret("Abs[x^-1]").unwrap(), "Abs[x]^(-1)");
    // Composes with the negation pull: Abs[-x^2] = Abs[x]^2.
    assert_eq!(interpret("Abs[-x^2]").unwrap(), "Abs[x]^2");
    // A symbolic exponent stays unevaluated.
    assert_eq!(interpret("Abs[x^n]").unwrap(), "Abs[x^n]");
  }

  // Fully symbolic products and genuine complex values are unchanged.
  #[test]
  fn abs_unaffected_cases() {
    assert_eq!(interpret("Abs[a x]").unwrap(), "Abs[a*x]");
    assert_eq!(interpret("Abs[x]").unwrap(), "Abs[x]");
    assert_eq!(interpret("Abs[3 + 4 I]").unwrap(), "5");
    assert_eq!(interpret("Abs[2 Pi]").unwrap(), "2*Pi");
  }

  // Abs of a real-valued numeric sum stays exact: a negative value is negated,
  // a non-negative value is returned unchanged (previously floatified).
  #[test]
  fn abs_real_sum_stays_exact() {
    assert_eq!(interpret("Abs[Sqrt[2] - 3]").unwrap(), "3 - Sqrt[2]");
    assert_eq!(interpret("Abs[Pi - 3]").unwrap(), "-3 + Pi");
    assert_eq!(interpret("Abs[E - 3]").unwrap(), "3 - E");
    assert_eq!(interpret("Abs[2 Pi - 7]").unwrap(), "7 - 2*Pi");
    assert_eq!(interpret("Abs[Sqrt[2] + 1]").unwrap(), "1 + Sqrt[2]");
    assert_eq!(interpret("Abs[-3 - Sqrt[2]]").unwrap(), "3 + Sqrt[2]");
    // A Real-leaf argument still collapses to a Real.
    assert_eq!(interpret("Abs[2.0 - Pi]").unwrap(), "1.1415926535897931");
  }

  // The Sqrt of a complex modulus must pull out perfect-square factors rather
  // than leaving e.g. Sqrt[8]; wolframscript returns 2 Sqrt[2].
  #[test]
  fn abs_complex_modulus_simplifies_sqrt() {
    assert_eq!(interpret("Abs[2 + 2 I]").unwrap(), "2*Sqrt[2]");
    assert_eq!(interpret("Abs[2 + 4 I]").unwrap(), "2*Sqrt[5]");
    assert_eq!(interpret("Abs[-3 - 6 I]").unwrap(), "3*Sqrt[5]");
    // A non-square modulus stays under the radical.
    assert_eq!(interpret("Abs[3 + I]").unwrap(), "Sqrt[10]");
    // Rational components.
    assert_eq!(interpret("Abs[1/2 + 1/2 I]").unwrap(), "1/Sqrt[2]");
  }
}

// Sign is multiplicative: it pulls the sign of each real-constant factor out
// of a product (positive → 1, negative → -1) and Sign[I] = I, keeping the rest
// under a single Sign. All verified against wolframscript.
mod sign_simplifications {
  use super::*;

  #[test]
  fn sign_negation() {
    assert_eq!(interpret("Sign[-x]").unwrap(), "-Sign[x]");
    assert_eq!(interpret("Sign[-a]").unwrap(), "-Sign[a]");
  }

  #[test]
  fn sign_pulls_real_coefficient() {
    assert_eq!(interpret("Sign[2 x]").unwrap(), "Sign[x]");
    assert_eq!(interpret("Sign[-2 x]").unwrap(), "-Sign[x]");
    assert_eq!(interpret("Sign[-3 x y]").unwrap(), "-Sign[x*y]");
    assert_eq!(interpret("Sign[-Pi x]").unwrap(), "-Sign[x]");
    assert_eq!(interpret("Sign[x/2]").unwrap(), "Sign[x]");
  }

  #[test]
  fn sign_imaginary_factor() {
    assert_eq!(interpret("Sign[I x]").unwrap(), "I*Sign[x]");
    assert_eq!(interpret("Sign[-I x]").unwrap(), "-I*Sign[x]");
  }

  #[test]
  fn sign_idempotent() {
    assert_eq!(interpret("Sign[Sign[x]]").unwrap(), "Sign[x]");
  }

  // Sign[base^exp] = Sign[base]^exp for a real numeric exponent.
  #[test]
  fn sign_of_power() {
    assert_eq!(interpret("Sign[x^2]").unwrap(), "Sign[x]^2");
    assert_eq!(interpret("Sign[x^3]").unwrap(), "Sign[x]^3");
    assert_eq!(interpret("Sign[x^(1/2)]").unwrap(), "Sqrt[Sign[x]]");
    assert_eq!(interpret("Sign[x^n]").unwrap(), "Sign[x^n]");
  }

  // Fully symbolic products and genuine complex values are unchanged.
  #[test]
  fn sign_unaffected_cases() {
    assert_eq!(interpret("Sign[a b]").unwrap(), "Sign[a*b]");
    assert_eq!(interpret("Sign[a]").unwrap(), "Sign[a]");
    assert_eq!(interpret("Sign[-Pi]").unwrap(), "-1");
    assert_eq!(interpret("Sign[2 I]").unwrap(), "I");
  }
}

mod non_negative_non_positive {
  use super::*;

  // Real-valued numeric expressions are decided by sign, matching
  // Positive/Negative (previously NonNegative/NonPositive stayed unevaluated).
  #[test]
  fn non_negative_numeric_expressions() {
    assert_eq!(interpret("NonNegative[Pi - 3]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[3 - Pi]").unwrap(), "False");
    assert_eq!(interpret("NonNegative[Sqrt[2] - 2]").unwrap(), "False");
    assert_eq!(interpret("NonNegative[Sin[1]]").unwrap(), "True");
  }

  #[test]
  fn non_positive_numeric_expressions() {
    assert_eq!(interpret("NonPositive[Pi - 3]").unwrap(), "False");
    assert_eq!(interpret("NonPositive[3 - Pi]").unwrap(), "True");
    assert_eq!(interpret("NonPositive[Sqrt[2] - 2]").unwrap(), "True");
    assert_eq!(interpret("NonPositive[Cos[3]]").unwrap(), "True");
  }

  #[test]
  fn complex_is_neither() {
    assert_eq!(interpret("NonNegative[2 + 3 I]").unwrap(), "False");
    assert_eq!(interpret("NonPositive[2 + 3 I]").unwrap(), "False");
    assert_eq!(interpret("NonNegative[I]").unwrap(), "False");
  }

  #[test]
  fn exact_and_symbolic_unchanged() {
    assert_eq!(interpret("NonNegative[0]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[1/3]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[-1/3]").unwrap(), "False");
    assert_eq!(interpret("NonNegative[x]").unwrap(), "NonNegative[x]");
    assert_eq!(interpret("NonPositive[x]").unwrap(), "NonPositive[x]");
  }
}

// Trig/hyperbolic of an imaginary argument I*z reduce to the counterpart
// function (Cos[I z] = Cosh[z], Sin[I z] = I Sinh[z], …). Verified against
// wolframscript.
mod trig_of_inverse_trig {
  use super::*;

  // Diagonal inverse identities.
  #[test]
  fn same_function_collapses() {
    assert_eq!(interpret("Sin[ArcSin[x]]").unwrap(), "x");
    assert_eq!(interpret("Cos[ArcCos[x]]").unwrap(), "x");
    assert_eq!(interpret("Tan[ArcTan[x]]").unwrap(), "x");
  }

  // Diagonal inverse identities for the reciprocal functions.
  #[test]
  fn reciprocal_same_function_collapses() {
    assert_eq!(interpret("Sec[ArcSec[x]]").unwrap(), "x");
    assert_eq!(interpret("Csc[ArcCsc[x]]").unwrap(), "x");
    assert_eq!(interpret("Cot[ArcCot[x]]").unwrap(), "x");
    // Also holds for explicit numeric arguments.
    assert_eq!(interpret("Cot[ArcCot[5]]").unwrap(), "5");
  }

  // Cross combinations auto-simplify to algebraic forms.
  #[test]
  fn sin_of_arccos_arctan() {
    assert_eq!(interpret("Sin[ArcCos[x]]").unwrap(), "Sqrt[1 - x^2]");
    assert_eq!(interpret("Sin[ArcTan[x]]").unwrap(), "x/Sqrt[1 + x^2]");
  }

  #[test]
  fn cos_of_arcsin_arctan() {
    assert_eq!(interpret("Cos[ArcSin[x]]").unwrap(), "Sqrt[1 - x^2]");
    assert_eq!(interpret("Cos[ArcTan[x]]").unwrap(), "1/Sqrt[1 + x^2]");
  }

  #[test]
  fn tan_of_arcsin_arccos() {
    assert_eq!(interpret("Tan[ArcSin[x]]").unwrap(), "x/Sqrt[1 - x^2]");
    assert_eq!(interpret("Tan[ArcCos[x]]").unwrap(), "Sqrt[1 - x^2]/x");
  }

  // With a rational argument the algebraic form collapses to a number:
  // the radicand (Sqrt[16/25]) and the resulting quotient are evaluated.
  #[test]
  fn rational_argument_collapses() {
    assert_eq!(interpret("Cos[ArcSin[3/5]]").unwrap(), "4/5");
    assert_eq!(interpret("Sin[ArcCos[4/5]]").unwrap(), "3/5");
    assert_eq!(interpret("Cos[ArcSin[5/13]]").unwrap(), "12/13");
    assert_eq!(interpret("Tan[ArcSin[3/5]]").unwrap(), "3/4");
    assert_eq!(interpret("Tan[ArcCos[3/5]]").unwrap(), "4/3");
    assert_eq!(interpret("Sin[ArcTan[3/4]]").unwrap(), "3/5");
    assert_eq!(interpret("Sec[ArcTan[3/4]]").unwrap(), "5/4");
  }

  // Compound arguments expand the inner square: (3 x)^2 -> 9 x^2.
  #[test]
  fn compound_argument_expands() {
    assert_eq!(
      interpret("Sin[ArcTan[3 x]]").unwrap(),
      "(3*x)/Sqrt[1 + 9*x^2]"
    );
    assert_eq!(interpret("Cos[ArcTan[2 y]]").unwrap(), "1/Sqrt[1 + 4*y^2]");
  }

  // Reciprocal trig (Sec/Csc/Cot) of inverse trig.
  #[test]
  fn sec_of_inverse() {
    assert_eq!(interpret("Sec[ArcSin[x]]").unwrap(), "1/Sqrt[1 - x^2]");
    assert_eq!(interpret("Sec[ArcCos[x]]").unwrap(), "x^(-1)");
    assert_eq!(interpret("Sec[ArcTan[x]]").unwrap(), "Sqrt[1 + x^2]");
  }

  #[test]
  fn csc_of_inverse() {
    assert_eq!(interpret("Csc[ArcSin[x]]").unwrap(), "x^(-1)");
    assert_eq!(interpret("Csc[ArcCos[x]]").unwrap(), "1/Sqrt[1 - x^2]");
    assert_eq!(interpret("Csc[ArcTan[x]]").unwrap(), "Sqrt[1 + x^2]/x");
  }

  #[test]
  fn cot_of_inverse() {
    assert_eq!(interpret("Cot[ArcSin[x]]").unwrap(), "Sqrt[1 - x^2]/x");
    assert_eq!(interpret("Cot[ArcCos[x]]").unwrap(), "x/Sqrt[1 - x^2]");
    assert_eq!(interpret("Cot[ArcTan[x]]").unwrap(), "x^(-1)");
  }

  // The bare reciprocal canonicalizes like wolframscript: x^(-1) for an
  // atom, 1/(2 x) for a product, x^(-2) for x^2.
  #[test]
  fn reciprocal_compound_forms() {
    assert_eq!(interpret("Csc[ArcSin[2 x]]").unwrap(), "1/(2*x)");
    assert_eq!(interpret("Csc[ArcSin[x^2]]").unwrap(), "x^(-2)");
    assert_eq!(interpret("Cot[ArcTan[2 x]]").unwrap(), "1/(2*x)");
  }
}

mod hyperbolic_of_inverse_hyperbolic {
  use super::*;

  #[test]
  fn same_function_collapses() {
    assert_eq!(interpret("Sinh[ArcSinh[x]]").unwrap(), "x");
    assert_eq!(interpret("Cosh[ArcCosh[x]]").unwrap(), "x");
    assert_eq!(interpret("Tanh[ArcTanh[x]]").unwrap(), "x");
  }

  // Diagonal inverse identities for the reciprocal hyperbolic functions.
  #[test]
  fn reciprocal_same_function_collapses() {
    assert_eq!(interpret("Coth[ArcCoth[x]]").unwrap(), "x");
    assert_eq!(interpret("Sech[ArcSech[x]]").unwrap(), "x");
    assert_eq!(interpret("Csch[ArcCsch[x]]").unwrap(), "x");
    assert_eq!(interpret("Coth[ArcCoth[3]]").unwrap(), "3");
  }

  #[test]
  fn clean_sqrt_forms() {
    assert_eq!(interpret("Sinh[ArcTanh[x]]").unwrap(), "x/Sqrt[1 - x^2]");
    assert_eq!(interpret("Cosh[ArcSinh[x]]").unwrap(), "Sqrt[1 + x^2]");
    assert_eq!(interpret("Cosh[ArcTanh[x]]").unwrap(), "1/Sqrt[1 - x^2]");
    assert_eq!(interpret("Tanh[ArcSinh[x]]").unwrap(), "x/Sqrt[1 + x^2]");
  }

  // ArcCosh uses wolframscript's branch-cut form Sqrt[(-1+x)/(1+x)] (1+x).
  #[test]
  fn arccosh_branch_form() {
    assert_eq!(
      interpret("Sinh[ArcCosh[x]]").unwrap(),
      "Sqrt[(-1 + x)/(1 + x)]*(1 + x)"
    );
    assert_eq!(
      interpret("Tanh[ArcCosh[x]]").unwrap(),
      "(Sqrt[(-1 + x)/(1 + x)]*(1 + x))/x"
    );
  }

  #[test]
  fn compound_argument_expands() {
    assert_eq!(
      interpret("Tanh[ArcSinh[2 x]]").unwrap(),
      "(2*x)/Sqrt[1 + 4*x^2]"
    );
  }
}

mod imaginary_argument_trig {
  use super::*;

  #[test]
  fn trig_of_imaginary() {
    assert_eq!(interpret("Cos[I x]").unwrap(), "Cosh[x]");
    assert_eq!(interpret("Sin[I x]").unwrap(), "I*Sinh[x]");
    assert_eq!(interpret("Tan[I x]").unwrap(), "I*Tanh[x]");
    assert_eq!(interpret("Cot[I x]").unwrap(), "-I*Coth[x]");
    assert_eq!(interpret("Sec[I x]").unwrap(), "Sech[x]");
    assert_eq!(interpret("Csc[I x]").unwrap(), "-I*Csch[x]");
  }

  #[test]
  fn hyperbolic_of_imaginary() {
    assert_eq!(interpret("Cosh[I x]").unwrap(), "Cos[x]");
    assert_eq!(interpret("Sinh[I x]").unwrap(), "I*Sin[x]");
    assert_eq!(interpret("Tanh[I x]").unwrap(), "I*Tan[x]");
  }

  #[test]
  fn imaginary_with_coefficient() {
    assert_eq!(interpret("Cos[2 I x]").unwrap(), "Cosh[2*x]");
    assert_eq!(interpret("Sin[3 I x]").unwrap(), "I*Sinh[3*x]");
    assert_eq!(interpret("Sin[-I x]").unwrap(), "-I*Sinh[x]");
    assert_eq!(interpret("Cos[I a b]").unwrap(), "Cosh[a*b]");
    // Bare I → z = 1.
    assert_eq!(interpret("Cos[I]").unwrap(), "Cosh[1]");
    assert_eq!(interpret("Sin[I]").unwrap(), "I*Sinh[1]");
  }

  // Real / even-odd arguments are unaffected by the reduction.
  #[test]
  fn real_arguments_unaffected() {
    assert_eq!(interpret("Cos[-x]").unwrap(), "Cos[x]");
    assert_eq!(interpret("Sin[-x]").unwrap(), "-Sin[x]");
    assert_eq!(interpret("Cos[x]").unwrap(), "Cos[x]");
    assert_eq!(interpret("Cos[Pi/3]").unwrap(), "1/2");
  }
}

// Times[Rational[-1, d], Plus[...]] renders by negating each summand, matching
// wolframscript: (-1/2)(a + b) → (-a - b)/2 (not the prior buggy -1/2*a + b).
mod negative_half_times_sum_display {
  use super::*;

  #[test]
  fn negates_each_summand() {
    assert_eq!(interpret("(-1/2)*(a + b)").unwrap(), "(-a - b)/2");
    assert_eq!(interpret("(-1/2)*(a + b + c)").unwrap(), "(-a - b - c)/2");
    assert_eq!(interpret("-1*((a + b)/2)").unwrap(), "(-a - b)/2");
  }

  // Already-negative summands collapse the double negation.
  #[test]
  fn collapses_double_negation() {
    assert_eq!(interpret("(-1/2)*(2 - x)").unwrap(), "(-2 + x)/2");
    assert_eq!(interpret("(-1/3)*(2 - x)").unwrap(), "(-2 + x)/3");
    assert_eq!(interpret("(-1/2)*(a - b)").unwrap(), "(-a + b)/2");
    assert_eq!(interpret("(-1/2)*(2 + x)").unwrap(), "(-2 - x)/2");
  }

  // A non-sum factor or a numerator other than -1 are unaffected.
  #[test]
  fn other_cases_unchanged() {
    assert_eq!(interpret("Times[Rational[-1, 2], x]").unwrap(), "-1/2*x");
    assert_eq!(interpret("(-3/2)*(a + b)").unwrap(), "(-3*(a + b))/2");
    assert_eq!(interpret("(1/2)*(a + b)").unwrap(), "(a + b)/2");
  }

  // Threaded[...] broadcasts against the trailing axes of the other operand.
  mod threaded {
    use super::*;

    // A bare Threaded stays symbolic.
    #[test]
    fn bare_is_symbolic() {
      assert_eq!(
        interpret("Threaded[{1, 2, 3}]").unwrap(),
        "Threaded[{1, 2, 3}]"
      );
    }

    // A matrix plus a Threaded vector adds the vector to every row.
    #[test]
    fn matrix_plus_vector() {
      assert_eq!(
        interpret("{{1, 2, 3}, {4, 5, 6}} + Threaded[{10, 20, 30}]").unwrap(),
        "{{11, 22, 33}, {14, 25, 36}}"
      );
    }

    // The same broadcasting applies to multiplication.
    #[test]
    fn matrix_times_vector() {
      assert_eq!(
        interpret("{{1, 2}, {3, 4}} * Threaded[{10, 100}]").unwrap(),
        "{{10, 200}, {30, 400}}"
      );
    }

    // Subtraction works through the additive path.
    #[test]
    fn matrix_minus_vector() {
      assert_eq!(
        interpret("{{1, 2, 3}, {4, 5, 6}} - Threaded[{1, 1, 1}]").unwrap(),
        "{{0, 1, 2}, {3, 4, 5}}"
      );
    }

    // A same-rank list combines element-wise and drops the wrapper.
    #[test]
    fn vector_same_rank() {
      assert_eq!(
        interpret("{1, 2, 3} + Threaded[{10, 20, 30}]").unwrap(),
        "{11, 22, 33}"
      );
    }

    // A scalar keeps the wrapper: Threaded[scalar op inner].
    #[test]
    fn scalar_keeps_wrapper() {
      assert_eq!(
        interpret("Threaded[{1, 2}] + 5").unwrap(),
        "Threaded[{6, 7}]"
      );
      assert_eq!(
        interpret("Threaded[{1, 2}] * 3").unwrap(),
        "Threaded[{3, 6}]"
      );
    }

    // Two Threaded operands combine their contents and keep the wrapper.
    #[test]
    fn both_threaded() {
      assert_eq!(
        interpret("Threaded[{1, 2}] + Threaded[{3, 4}]").unwrap(),
        "Threaded[{4, 6}]"
      );
    }

    // A three-way mix of matrix, Threaded vector, and scalar.
    #[test]
    fn three_way_mix() {
      assert_eq!(
        interpret("{{1, 2}, {3, 4}} + Threaded[{10, 20}] + 100").unwrap(),
        "{{111, 122}, {113, 124}}"
      );
    }

    // When the non-Threaded operand is shallower than the Threaded content,
    // WL reports Threaded::thrdts and leaves the expression unevaluated.
    #[test]
    fn too_shallow_is_unevaluated() {
      assert_eq!(
        interpret("Head[{1, 2} + Threaded[{{10, 20}, {30, 40}}]]").unwrap(),
        "Plus"
      );
    }
  }
}

/// Regression tests for issue #136: the canonical Times ordering of a
/// factor against a Plus factor must follow Wolfram's top-term rule for
/// function calls and powers, not just bare identifiers.
mod times_factor_vs_sum_ordering {
  use super::*;

  // A factor sorts before `1 + factor` regardless of its head.
  #[test]
  fn factor_before_shifted_sum() {
    assert_eq!(interpret("n(n+1)").unwrap(), "n*(1 + n)");
    assert_eq!(interpret("f[x](f[x]+1)").unwrap(), "f[x]*(1 + f[x])");
    assert_eq!(
      interpret("f[x,y](1+f[x,y])").unwrap(),
      "f[x, y]*(1 + f[x, y])"
    );
    assert_eq!(
      interpret("Sin[x](Cos[x]+Sin[x])").unwrap(),
      "Sin[x]*(Cos[x] + Sin[x])"
    );
    assert_eq!(
      interpret("Cos[x](1+Cos[x])(2+Cos[x])").unwrap(),
      "Cos[x]*(1 + Cos[x])*(2 + Cos[x])"
    );
    assert_eq!(
      interpret("f'[x](1+f'[x])").unwrap(),
      "Derivative[1][f][x]*(1 + Derivative[1][f][x])"
    );
  }

  // The sum's highest term decides the order across different heads.
  #[test]
  fn sum_top_term_decides() {
    assert_eq!(interpret("g[x](f[x]+1)").unwrap(), "(1 + f[x])*g[x]");
    assert_eq!(interpret("f[x](g[x]+1)").unwrap(), "f[x]*(1 + g[x])");
    assert_eq!(interpret("f[y](1+f[x])").unwrap(), "(1 + f[x])*f[y]");
    assert_eq!(
      interpret("Sin[x](1+Cos[x])").unwrap(),
      "(1 + Cos[x])*Sin[x]"
    );
    assert_eq!(
      interpret("Log[x](1+Sin[x])").unwrap(),
      "Log[x]*(1 + Sin[x])"
    );
    assert_eq!(interpret("h[x](1+Sin[x])").unwrap(), "h[x]*(1 + Sin[x])");
  }

  // An atom sorts before a sum whose top term is a function call.
  #[test]
  fn atom_before_function_call_sum() {
    assert_eq!(interpret("x(1+f[x])").unwrap(), "x*(1 + f[x])");
    assert_eq!(interpret("x(1-f[x])").unwrap(), "x*(1 - f[x])");
    assert_eq!(interpret("x(f[x]-x)").unwrap(), "x*(-x + f[x])");
    assert_eq!(interpret("f[x](x+1)").unwrap(), "(1 + x)*f[x]");
  }

  // Negated top terms and `Plus[negative, base]` pull the sum forward.
  #[test]
  fn negated_forms_put_sum_first() {
    assert_eq!(interpret("f[x](f[x]-1)").unwrap(), "(-1 + f[x])*f[x]");
    assert_eq!(interpret("f[x](2-f[x])").unwrap(), "(2 - f[x])*f[x]");
    assert_eq!(interpret("g[x](f[x]-g[x])").unwrap(), "(f[x] - g[x])*g[x]");
    assert_eq!(interpret("x^2(x-1)").unwrap(), "(-1 + x)*x^2");
    assert_eq!(interpret("f[x]^2(f[x]-1)").unwrap(), "(-1 + f[x])*f[x]^2");
    // ...but not with a coefficient on the top term, extra terms, or a
    // lower power of the shared base.
    assert_eq!(interpret("x(11x-4)").unwrap(), "x*(-4 + 11*x)");
    assert_eq!(interpret("b(-1+a+b)").unwrap(), "b*(-1 + a + b)");
    assert_eq!(interpret("x(x^2-3)").unwrap(), "x*(-3 + x^2)");
  }

  // Powers compare by base; exponents only matter on a tie.
  #[test]
  fn powers_compare_by_base() {
    assert_eq!(interpret("f[x]^2(1+f[x])").unwrap(), "f[x]^2*(1 + f[x])");
    assert_eq!(interpret("f[x](1+f[x]^2)").unwrap(), "f[x]*(1 + f[x]^2)");
    assert_eq!(interpret("f[x](2-f[x]^2)").unwrap(), "f[x]*(2 - f[x]^2)");
    assert_eq!(interpret("x^3(1+x^2)").unwrap(), "x^3*(1 + x^2)");
    assert_eq!(interpret("y(1+x^2)").unwrap(), "(1 + x^2)*y");
    assert_eq!(interpret("y(x-y^2)").unwrap(), "y*(x - y^2)");
  }
}
