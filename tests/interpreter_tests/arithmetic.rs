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

  #[test]
  fn odd_q_big_integer() {
    assert_eq!(interpret("OddQ[2^128 + 1]").unwrap(), "True");
  }

  #[test]
  fn number_q_big_integer() {
    assert_eq!(interpret("NumberQ[2^128]").unwrap(), "True");
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
    assert_eq!(interpret("Sqrt[-12]").unwrap(), "2*I*Sqrt[3]");
  }

  #[test]
  fn sqrt_equals_i() {
    assert_eq!(interpret("I == Sqrt[-1]").unwrap(), "True");
  }
}

mod sqrt_power {
  use super::*;

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
