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
}
