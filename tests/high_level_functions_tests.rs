use woxi::interpret;

mod high_level_functions_tests {
  use super::*;

  mod evenq_tests {
    use super::*;
    #[test]
    fn test_for_negative() {
      assert_eq!(interpret("EvenQ[-2]").unwrap(), "True",);
      assert_eq!(interpret("EvenQ[-1]").unwrap(), "False",);
    }
    #[test]
    fn test_for_zero() {
      assert_eq!(interpret("EvenQ[0]").unwrap(), "True",);
    }
    #[test]
    fn test_for_positive() {
      assert_eq!(interpret("EvenQ[1]").unwrap(), "False",);
      assert_eq!(interpret("EvenQ[2]").unwrap(), "True",);
      assert_eq!(interpret("EvenQ[3]").unwrap(), "False",);
    }
    #[test]
    fn test_for_float() {
      assert_eq!(interpret("EvenQ[1.2]").unwrap(), "False",);
      assert_eq!(interpret("EvenQ[1.3]").unwrap(), "False",);
    }
  }

  mod oddq_tests {
    use super::*;
    #[test]
    fn test_for_negative() {
      assert_eq!(interpret("OddQ[-1]").unwrap(), "True");
    }
    #[test]
    fn test_for_zero() {
      assert_eq!(interpret("OddQ[0]").unwrap(), "False");
    }
    #[test]
    fn test_for_positive() {
      assert_eq!(interpret("OddQ[1]").unwrap(), "True");
      assert_eq!(interpret("OddQ[2]").unwrap(), "False");
    }
    #[test]
    fn test_for_float() {
      assert_eq!(interpret("OddQ[1.2]").unwrap(), "False");
      assert_eq!(interpret("OddQ[1.3]").unwrap(), "False");
    }
  }

  // #[test]
  // fn test_first_function() {
  //   assert_eq!(interpret("First[{1, 2, 3}]").unwrap(), "1");
  //   assert_eq!(interpret("First[{a, b, c}]").unwrap(), "a");
  //   assert_eq!(interpret("First[{True, False, False}]").unwrap(), "True");
  // }

  mod prime_function {
    use super::*;

    #[test]
    fn test_prime_function() {
      assert_eq!(interpret("Prime[1]").unwrap(), "2");
      assert_eq!(interpret("Prime[2]").unwrap(), "3");
      assert_eq!(interpret("Prime[3]").unwrap(), "5");
      assert_eq!(interpret("Prime[4]").unwrap(), "7");
      assert_eq!(interpret("Prime[5]").unwrap(), "11");
      assert_eq!(interpret("Prime[100]").unwrap(), "541");
    }

    #[test]
    fn test_prime_function_invalid_input() {
      // Wolfram returns unevaluated for invalid inputs
      assert_eq!(interpret("Prime[0]").unwrap(), "Prime[0]");
      assert_eq!(interpret("Prime[-1]").unwrap(), "Prime[-1]");
      assert_eq!(interpret("Prime[1.5]").unwrap(), "Prime[1.5]");
    }
  }

  // ─── Hyperbolic Trig Functions ────────────────────────────────────
  mod sinh_tests {
    use super::*;
    #[test]
    fn test_sinh_zero() {
      assert_eq!(interpret("Sinh[0]").unwrap(), "0");
    }
    #[test]
    fn test_sinh_real() {
      assert_eq!(interpret("Sinh[1.0]").unwrap(), "1.1752011936438014");
    }
    #[test]
    fn test_sinh_symbolic() {
      assert_eq!(interpret("Sinh[x]").unwrap(), "Sinh[x]");
    }
  }

  mod cosh_tests {
    use super::*;
    #[test]
    fn test_cosh_zero() {
      assert_eq!(interpret("Cosh[0]").unwrap(), "1");
    }
    #[test]
    fn test_cosh_real() {
      assert_eq!(interpret("Cosh[1.0]").unwrap(), "1.5430806348152437");
    }
    #[test]
    fn test_cosh_symbolic() {
      assert_eq!(interpret("Cosh[x]").unwrap(), "Cosh[x]");
    }
  }

  mod tanh_tests {
    use super::*;
    #[test]
    fn test_tanh_zero() {
      assert_eq!(interpret("Tanh[0]").unwrap(), "0");
    }
    #[test]
    fn test_tanh_real() {
      assert_eq!(interpret("Tanh[1.0]").unwrap(), "0.7615941559557649");
    }
  }

  mod sech_tests {
    use super::*;
    #[test]
    fn test_sech_zero() {
      assert_eq!(interpret("Sech[0]").unwrap(), "1");
    }
    #[test]
    fn test_sech_real() {
      // Sech[1.0] = 1/Cosh[1.0]
      assert_eq!(interpret("Sech[1.0]").unwrap(), "0.6480542736638855");
    }
  }

  mod arcsinh_tests {
    use super::*;
    #[test]
    fn test_arcsinh_zero() {
      assert_eq!(interpret("ArcSinh[0]").unwrap(), "0");
    }
    #[test]
    fn test_arcsinh_real() {
      assert_eq!(interpret("ArcSinh[1.0]").unwrap(), "0.881373587019543");
    }
  }

  mod arccosh_tests {
    use super::*;
    #[test]
    fn test_arccosh_one() {
      assert_eq!(interpret("ArcCosh[1]").unwrap(), "0");
    }
    #[test]
    fn test_arccosh_real() {
      assert_eq!(interpret("ArcCosh[2.0]").unwrap(), "1.3169578969248166");
    }
  }

  mod arctanh_tests {
    use super::*;
    #[test]
    fn test_arctanh_zero() {
      assert_eq!(interpret("ArcTanh[0]").unwrap(), "0");
    }
    #[test]
    fn test_arctanh_real() {
      let result: f64 = interpret("ArcTanh[0.5]").unwrap().parse().unwrap();
      assert!((result - 0.5493061443340549).abs() < 1e-15);
    }
  }

  // ─── String Functions ──────────────────────────────────────────────
  mod capitalize_tests {
    use super::*;
    #[test]
    fn test_capitalize() {
      assert_eq!(
        interpret(r#"Capitalize["hello world"]"#).unwrap(),
        "Hello world"
      );
    }
    #[test]
    fn test_capitalize_empty() {
      assert_eq!(interpret(r#"Capitalize[""]"#).unwrap(), "");
    }
    #[test]
    fn test_capitalize_already() {
      assert_eq!(interpret(r#"Capitalize["Hello"]"#).unwrap(), "Hello");
    }
  }

  mod decapitalize_tests {
    use super::*;
    #[test]
    fn test_decapitalize() {
      assert_eq!(
        interpret(r#"Decapitalize["Hello World"]"#).unwrap(),
        "hello World"
      );
    }
    #[test]
    fn test_decapitalize_empty() {
      assert_eq!(interpret(r#"Decapitalize[""]"#).unwrap(), "");
    }
  }

  mod string_insert_tests {
    use super::*;
    #[test]
    fn test_string_insert() {
      assert_eq!(
        interpret(r#"StringInsert["abcdef", "X", 3]"#).unwrap(),
        "abXcdef"
      );
    }
    #[test]
    fn test_string_insert_start() {
      assert_eq!(interpret(r#"StringInsert["abc", "X", 1]"#).unwrap(), "Xabc");
    }
    #[test]
    fn test_string_insert_negative() {
      assert_eq!(
        interpret(r#"StringInsert["abc", "X", -1]"#).unwrap(),
        "abcX"
      );
    }
  }

  mod string_delete_tests {
    use super::*;
    #[test]
    fn test_string_delete() {
      assert_eq!(interpret(r#"StringDelete["abcabc", "b"]"#).unwrap(), "acac");
    }
    #[test]
    fn test_string_delete_none() {
      assert_eq!(interpret(r#"StringDelete["abc", "x"]"#).unwrap(), "abc");
    }
  }

  // ─── List Functions ────────────────────────────────────────────────
  mod delete_adjacent_duplicates_tests {
    use super::*;
    #[test]
    fn test_basic() {
      assert_eq!(
        interpret("DeleteAdjacentDuplicates[{1,1,2,3,3,3,2,2}]").unwrap(),
        "{1, 2, 3, 2}"
      );
    }
    #[test]
    fn test_no_duplicates() {
      assert_eq!(
        interpret("DeleteAdjacentDuplicates[{1,2,3}]").unwrap(),
        "{1, 2, 3}"
      );
    }
    #[test]
    fn test_empty() {
      assert_eq!(interpret("DeleteAdjacentDuplicates[{}]").unwrap(), "{}");
    }
  }

  mod commonest_tests {
    use super::*;
    #[test]
    fn test_commonest_single() {
      assert_eq!(interpret("Commonest[{a,b,a,c,b,a}]").unwrap(), "{a}");
    }
    #[test]
    fn test_commonest_n() {
      assert_eq!(interpret("Commonest[{a,b,a,c,b,a}, 2]").unwrap(), "{a, b}");
    }
    #[test]
    fn test_commonest_numeric() {
      assert_eq!(interpret("Commonest[{1,2,2,3,3,3}]").unwrap(), "{3}");
    }
  }

  mod compose_list_tests {
    use super::*;
    #[test]
    fn test_compose_list() {
      assert_eq!(
        interpret("ComposeList[{f,g,h}, x]").unwrap(),
        "{x, f[x], g[f[x]], h[g[f[x]]]}"
      );
    }
  }

  // ─── Number Theory ─────────────────────────────────────────────────
  mod integer_digits_tests {
    use super::*;
    #[test]
    fn test_base10() {
      assert_eq!(
        interpret("IntegerDigits[12345]").unwrap(),
        "{1, 2, 3, 4, 5}"
      );
    }
    #[test]
    fn test_zero() {
      assert_eq!(interpret("IntegerDigits[0]").unwrap(), "{0}");
    }
    #[test]
    fn test_negative() {
      assert_eq!(interpret("IntegerDigits[-123]").unwrap(), "{1, 2, 3}");
    }
    #[test]
    fn test_base2() {
      assert_eq!(
        interpret("IntegerDigits[255, 2]").unwrap(),
        "{1, 1, 1, 1, 1, 1, 1, 1}"
      );
    }
    #[test]
    fn test_base16() {
      assert_eq!(interpret("IntegerDigits[255, 16]").unwrap(), "{15, 15}");
    }
    #[test]
    fn test_padding() {
      assert_eq!(
        interpret("IntegerDigits[42, 10, 5]").unwrap(),
        "{0, 0, 0, 4, 2}"
      );
    }
    #[test]
    fn test_truncation() {
      assert_eq!(
        interpret("IntegerDigits[12345, 10, 3]").unwrap(),
        "{3, 4, 5}"
      );
    }
    #[test]
    fn test_zero_padding() {
      assert_eq!(
        interpret("IntegerDigits[0, 10, 5]").unwrap(),
        "{0, 0, 0, 0, 0}"
      );
    }
    #[test]
    fn test_negative_padding() {
      assert_eq!(
        interpret("IntegerDigits[-42, 10, 5]").unwrap(),
        "{0, 0, 0, 4, 2}"
      );
    }
    #[test]
    fn test_base2_padding() {
      assert_eq!(
        interpret("IntegerDigits[5, 2, 8]").unwrap(),
        "{0, 0, 0, 0, 0, 1, 0, 1}"
      );
    }
  }

  mod digit_count_tests {
    use super::*;
    #[test]
    fn test_digit_count_base10() {
      assert_eq!(
        interpret("DigitCount[1234]").unwrap(),
        "{1, 1, 1, 1, 0, 0, 0, 0, 0, 0}"
      );
    }
    #[test]
    fn test_digit_count_base2() {
      assert_eq!(interpret("DigitCount[255, 2]").unwrap(), "{8, 0}");
    }
    #[test]
    fn test_digit_count_specific_digit() {
      assert_eq!(interpret("DigitCount[1234, 10, 1]").unwrap(), "1");
    }
  }

  mod digit_sum_tests {
    use super::*;
    #[test]
    fn test_digit_sum_base10() {
      assert_eq!(interpret("DigitSum[1234]").unwrap(), "10");
    }
    #[test]
    fn test_digit_sum_base16() {
      assert_eq!(interpret("DigitSum[255, 16]").unwrap(), "30");
    }
    #[test]
    fn test_digit_sum_zero() {
      assert_eq!(interpret("DigitSum[0]").unwrap(), "0");
    }
  }

  mod continued_fraction_tests {
    use super::*;
    #[test]
    fn test_rational() {
      assert_eq!(interpret("ContinuedFraction[3/7]").unwrap(), "{0, 2, 3}");
    }
    #[test]
    fn test_integer() {
      assert_eq!(interpret("ContinuedFraction[5]").unwrap(), "{5}");
    }
    #[test]
    fn test_pi_5terms() {
      assert_eq!(
        interpret("ContinuedFraction[Pi, 5]").unwrap(),
        "{3, 7, 15, 1, 292}"
      );
    }
    #[test]
    fn test_rational_22_7() {
      assert_eq!(interpret("ContinuedFraction[22/7]").unwrap(), "{3, 7}");
    }
    #[test]
    fn test_pi_30terms() {
      assert_eq!(
        interpret("ContinuedFraction[Pi, 30]").unwrap(),
        "{3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, 1, 84, 2, 1, 1, 15, 3, 13, 1, 4}"
      );
    }
    #[test]
    fn test_e_40terms() {
      assert_eq!(
        interpret("ContinuedFraction[E, 40]").unwrap(),
        "{2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, 1, 1, 10, 1, 1, 12, 1, 1, 14, 1, 1, 16, 1, 1, 18, 1, 1, 20, 1, 1, 22, 1, 1, 24, 1, 1, 26, 1}"
      );
    }
  }

  mod from_continued_fraction_tests {
    use super::*;
    #[test]
    fn test_pi_approximation() {
      assert_eq!(
        interpret("FromContinuedFraction[{3, 7, 15, 1, 292, 1}]").unwrap(),
        "104348/33215"
      );
    }
    #[test]
    fn test_simple() {
      assert_eq!(
        interpret("FromContinuedFraction[{1, 2, 3}]").unwrap(),
        "10/7"
      );
    }
    #[test]
    fn test_single_element() {
      assert_eq!(interpret("FromContinuedFraction[{2}]").unwrap(), "2");
    }
    #[test]
    fn test_empty_list() {
      assert_eq!(interpret("FromContinuedFraction[{}]").unwrap(), "Infinity");
    }
    #[test]
    fn test_fibonacci_convergent() {
      assert_eq!(
        interpret("FromContinuedFraction[{1, 1, 1, 1, 1, 1, 1, 1}]").unwrap(),
        "34/21"
      );
    }
    #[test]
    fn test_starting_with_zero() {
      assert_eq!(
        interpret("FromContinuedFraction[{0, 1, 1, 1, 1, 1}]").unwrap(),
        "5/8"
      );
    }
    #[test]
    fn test_roundtrip() {
      assert_eq!(
        interpret("FromContinuedFraction[ContinuedFraction[355/113]]").unwrap(),
        "355/113"
      );
    }
  }

  mod lucas_l_tests {
    use super::*;
    #[test]
    fn test_lucas_l_10() {
      assert_eq!(interpret("LucasL[10]").unwrap(), "123");
    }
    #[test]
    fn test_lucas_l_0() {
      assert_eq!(interpret("LucasL[0]").unwrap(), "2");
    }
    #[test]
    fn test_lucas_l_1() {
      assert_eq!(interpret("LucasL[1]").unwrap(), "1");
    }
  }

  mod chinese_remainder_tests {
    use super::*;
    #[test]
    fn test_basic() {
      assert_eq!(
        interpret("ChineseRemainder[{1,2,3},{3,5,7}]").unwrap(),
        "52"
      );
    }
  }

  mod divisor_sum_tests {
    use super::*;
    #[test]
    fn test_basic() {
      assert_eq!(interpret("DivisorSum[12, #^2&]").unwrap(), "210");
    }
  }

  // ─── Combinatorics ─────────────────────────────────────────────────
  mod bernoulli_b_tests {
    use super::*;
    #[test]
    fn test_bernoulli_0() {
      assert_eq!(interpret("BernoulliB[0]").unwrap(), "1");
    }
    #[test]
    fn test_bernoulli_1() {
      assert_eq!(interpret("BernoulliB[1]").unwrap(), "-1/2");
    }
    #[test]
    fn test_bernoulli_2() {
      assert_eq!(interpret("BernoulliB[2]").unwrap(), "1/6");
    }
    #[test]
    fn test_bernoulli_odd() {
      assert_eq!(interpret("BernoulliB[3]").unwrap(), "0");
    }
    #[test]
    fn test_bernoulli_10() {
      assert_eq!(interpret("BernoulliB[10]").unwrap(), "5/66");
    }
    #[test]
    fn test_bernoulli_4() {
      assert_eq!(interpret("BernoulliB[4]").unwrap(), "-1/30");
    }
  }

  mod euler_e_tests {
    use super::*;
    #[test]
    fn test_euler_e_0() {
      assert_eq!(interpret("EulerE[0]").unwrap(), "1");
    }
    #[test]
    fn test_euler_e_1() {
      assert_eq!(interpret("EulerE[1]").unwrap(), "0");
    }
    #[test]
    fn test_euler_e_2() {
      assert_eq!(interpret("EulerE[2]").unwrap(), "-1");
    }
    #[test]
    fn test_euler_e_odd() {
      assert_eq!(interpret("EulerE[3]").unwrap(), "0");
      assert_eq!(interpret("EulerE[5]").unwrap(), "0");
      assert_eq!(interpret("EulerE[7]").unwrap(), "0");
    }
    #[test]
    fn test_euler_e_4() {
      assert_eq!(interpret("EulerE[4]").unwrap(), "5");
    }
    #[test]
    fn test_euler_e_6() {
      assert_eq!(interpret("EulerE[6]").unwrap(), "-61");
    }
    #[test]
    fn test_euler_e_8() {
      assert_eq!(interpret("EulerE[8]").unwrap(), "1385");
    }
    #[test]
    fn test_euler_e_10() {
      assert_eq!(interpret("EulerE[10]").unwrap(), "-50521");
    }
    #[test]
    fn test_euler_e_20() {
      assert_eq!(interpret("EulerE[20]").unwrap(), "370371188237525");
    }
    #[test]
    fn test_euler_e_table() {
      assert_eq!(
        interpret("Table[EulerE[k], {k, 0, 10}]").unwrap(),
        "{1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521}"
      );
    }
    #[test]
    fn test_euler_e_negative_arg() {
      assert_eq!(interpret("EulerE[-1]").unwrap(), "EulerE[-1]");
    }
    #[test]
    fn test_euler_e_rational_arg() {
      assert_eq!(interpret("EulerE[1/2]").unwrap(), "EulerE[1/2]");
    }
    #[test]
    fn test_euler_polynomial_0() {
      assert_eq!(interpret("EulerE[0, z]").unwrap(), "1");
    }
    #[test]
    fn test_euler_polynomial_1() {
      assert_eq!(interpret("EulerE[1, z]").unwrap(), "-1/2 + z");
    }
    #[test]
    fn test_euler_polynomial_2() {
      assert_eq!(interpret("EulerE[2, z]").unwrap(), "-z + z^2");
    }
    #[test]
    fn test_euler_polynomial_3() {
      assert_eq!(interpret("EulerE[3, z]").unwrap(), "1/4 - (3*z^2)/2 + z^3");
    }
    #[test]
    fn test_euler_polynomial_5() {
      assert_eq!(
        interpret("EulerE[5, z]").unwrap(),
        "-1/2 + (5*z^2)/2 - (5*z^4)/2 + z^5"
      );
    }
    #[test]
    fn test_euler_polynomial_numeric_eval() {
      assert_eq!(interpret("EulerE[3, 5]").unwrap(), "351/4");
      assert_eq!(interpret("EulerE[4, -2]").unwrap(), "30");
      assert_eq!(interpret("EulerE[6, 1/3]").unwrap(), "-602/729");
    }
    #[test]
    fn test_euler_polynomial_at_half() {
      assert_eq!(interpret("EulerE[3, 1/2]").unwrap(), "0");
      assert_eq!(interpret("EulerE[5, 1/2]").unwrap(), "0");
    }
    #[test]
    fn test_euler_polynomial_at_0_and_1() {
      assert_eq!(interpret("EulerE[2, 0]").unwrap(), "0");
      assert_eq!(interpret("EulerE[2, 1]").unwrap(), "0");
      assert_eq!(interpret("EulerE[4, 0]").unwrap(), "0");
      assert_eq!(interpret("EulerE[4, 1]").unwrap(), "0");
    }
    #[test]
    fn test_euler_polynomial_invalid_first_arg() {
      assert_eq!(interpret("EulerE[-1, z]").unwrap(), "EulerE[-1, z]");
      assert_eq!(interpret("EulerE[1/2, z]").unwrap(), "EulerE[1/2, z]");
    }
  }

  mod catalan_number_tests {
    use super::*;
    #[test]
    fn test_catalan_5() {
      assert_eq!(interpret("CatalanNumber[5]").unwrap(), "42");
    }
    #[test]
    fn test_catalan_0() {
      assert_eq!(interpret("CatalanNumber[0]").unwrap(), "1");
    }
    #[test]
    fn test_catalan_1() {
      assert_eq!(interpret("CatalanNumber[1]").unwrap(), "1");
    }
    #[test]
    fn test_catalan_10() {
      assert_eq!(interpret("CatalanNumber[10]").unwrap(), "16796");
    }
  }

  mod stirling_s1_tests {
    use super::*;
    #[test]
    fn test_stirling_s1_5_2() {
      assert_eq!(interpret("StirlingS1[5, 2]").unwrap(), "-50");
    }
    #[test]
    fn test_stirling_s1_boundary() {
      assert_eq!(interpret("StirlingS1[0, 0]").unwrap(), "1");
      assert_eq!(interpret("StirlingS1[3, 0]").unwrap(), "0");
      assert_eq!(interpret("StirlingS1[3, 4]").unwrap(), "0");
    }
    #[test]
    fn test_stirling_s1_large() {
      assert_eq!(
        interpret("StirlingS1[50, 1]").unwrap(),
        "-608281864034267560872252163321295376887552831379210240000000000"
      );
    }
  }

  mod stirling_s2_tests {
    use super::*;
    #[test]
    fn test_stirling_s2_5_2() {
      assert_eq!(interpret("StirlingS2[5, 2]").unwrap(), "15");
    }
    #[test]
    fn test_stirling_s2_boundary() {
      assert_eq!(interpret("StirlingS2[0, 0]").unwrap(), "1");
      assert_eq!(interpret("StirlingS2[3, 0]").unwrap(), "0");
    }
  }

  // ─── FrobeniusNumber ─────────────────────────────────────────────────
  mod frobenius_number_tests {
    use super::*;
    #[test]
    fn test_frobenius_two_coprime() {
      assert_eq!(interpret("FrobeniusNumber[{2, 5}]").unwrap(), "3");
      assert_eq!(interpret("FrobeniusNumber[{3, 7}]").unwrap(), "11");
    }
    #[test]
    fn test_frobenius_three_elements() {
      assert_eq!(interpret("FrobeniusNumber[{3, 5, 7}]").unwrap(), "4");
      assert_eq!(interpret("FrobeniusNumber[{7, 13, 23}]").unwrap(), "45");
      assert_eq!(interpret("FrobeniusNumber[{6, 9, 20}]").unwrap(), "43");
    }
    #[test]
    fn test_frobenius_with_one() {
      assert_eq!(interpret("FrobeniusNumber[{1, 5}]").unwrap(), "-1");
      assert_eq!(interpret("FrobeniusNumber[{1, 2, 3}]").unwrap(), "-1");
    }
    #[test]
    fn test_frobenius_non_coprime() {
      assert_eq!(interpret("FrobeniusNumber[{2, 4}]").unwrap(), "Infinity");
      assert_eq!(interpret("FrobeniusNumber[{6, 9}]").unwrap(), "Infinity");
    }
    #[test]
    fn test_frobenius_single_element() {
      assert_eq!(interpret("FrobeniusNumber[{5}]").unwrap(), "Infinity");
      assert_eq!(interpret("FrobeniusNumber[{1}]").unwrap(), "-1");
    }
    #[test]
    fn test_frobenius_unevaluated() {
      assert_eq!(
        interpret("FrobeniusNumber[{-2, 5}]").unwrap(),
        "FrobeniusNumber[{-2, 5}]"
      );
      assert_eq!(
        interpret("FrobeniusNumber[{}]").unwrap(),
        "FrobeniusNumber[{}]"
      );
    }
  }

  // ─── Catch/Throw ───────────────────────────────────────────────────
  mod catch_throw_tests {
    use super::*;
    #[test]
    fn test_catch_with_throw() {
      assert_eq!(interpret("Catch[1 + Throw[2]]").unwrap(), "2");
    }
    #[test]
    fn test_catch_no_throw() {
      assert_eq!(interpret("Catch[1 + 2]").unwrap(), "3");
    }
    #[test]
    fn test_catch_with_tag() {
      assert_eq!(
        interpret(r#"Catch[Throw["hello", "tag"], "tag"]"#).unwrap(),
        "hello"
      );
    }
    #[test]
    fn test_throw_value() {
      assert_eq!(interpret("Catch[Throw[42]]").unwrap(), "42");
    }
    #[test]
    fn test_nested_catch() {
      assert_eq!(interpret("Catch[Catch[Throw[1, a], b], a]").unwrap(), "1");
    }
  }

  // ─── IntegerPartitions ──────────────────────────────────────────────
  mod integer_partitions_tests {
    use super::*;
    #[test]
    fn test_integer_partitions_5() {
      assert_eq!(
        interpret("IntegerPartitions[5]").unwrap(),
        "{{5}, {4, 1}, {3, 2}, {3, 1, 1}, {2, 2, 1}, {2, 1, 1, 1}, {1, 1, 1, 1, 1}}"
      );
    }
    #[test]
    fn test_integer_partitions_0() {
      assert_eq!(interpret("IntegerPartitions[0]").unwrap(), "{{}}");
    }
    #[test]
    fn test_integer_partitions_1() {
      assert_eq!(interpret("IntegerPartitions[1]").unwrap(), "{{1}}");
    }
    #[test]
    fn test_integer_partitions_negative() {
      assert_eq!(interpret("IntegerPartitions[-1]").unwrap(), "{}");
    }
    #[test]
    fn test_integer_partitions_max_length() {
      assert_eq!(
        interpret("IntegerPartitions[10, 3]").unwrap(),
        "{{10}, {9, 1}, {8, 2}, {8, 1, 1}, {7, 3}, {7, 2, 1}, {6, 4}, {6, 3, 1}, {6, 2, 2}, {5, 5}, {5, 4, 1}, {5, 3, 2}, {4, 4, 2}, {4, 3, 3}}"
      );
    }
    #[test]
    fn test_integer_partitions_exact_length() {
      assert_eq!(
        interpret("IntegerPartitions[5, {3}]").unwrap(),
        "{{3, 1, 1}, {2, 2, 1}}"
      );
    }
    #[test]
    fn test_integer_partitions_length_range() {
      assert_eq!(
        interpret("IntegerPartitions[5, {2, 3}]").unwrap(),
        "{{4, 1}, {3, 2}, {3, 1, 1}, {2, 2, 1}}"
      );
    }
    #[test]
    fn test_integer_partitions_restricted_elements() {
      assert_eq!(
        interpret("IntegerPartitions[10, All, {1, 2, 5}]").unwrap(),
        "{{5, 5}, {5, 2, 2, 1}, {5, 2, 1, 1, 1}, {5, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 1, 1}, {2, 2, 2, 1, 1, 1, 1}, {2, 2, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}"
      );
    }
    #[test]
    fn test_integer_partitions_length_and_elements() {
      assert_eq!(
        interpret("IntegerPartitions[6, {3}, {1, 2, 3}]").unwrap(),
        "{{3, 2, 1}, {2, 2, 2}}"
      );
    }
    #[test]
    fn test_integer_partitions_20_length() {
      assert_eq!(interpret("IntegerPartitions[20] // Length").unwrap(), "627");
    }
    #[test]
    fn test_integer_partitions_3_max2() {
      assert_eq!(
        interpret("IntegerPartitions[3, 2]").unwrap(),
        "{{3}, {2, 1}}"
      );
    }
  }

  mod save_tests {
    use super::*;
    use woxi::interpret_with_stdout;

    #[test]
    fn test_save_variable_to_stdout() {
      let result =
        interpret_with_stdout(r#"a = 5; Save["stdout", a]"#).unwrap();
      assert_eq!(result.result, "Null");
      assert!(result.stdout.contains("a = 5"));
    }

    #[test]
    fn test_save_function_to_stdout() {
      let result =
        interpret_with_stdout(r#"f[x_] := x + 1; Save["stdout", f]"#).unwrap();
      assert_eq!(result.result, "Null");
      assert!(result.stdout.contains("f[x_] := x + 1"));
    }

    #[test]
    fn test_save_multiple_definitions() {
      let result = interpret_with_stdout(
        r#"f[x_] := x + 1; f[x_, y_] := x + y; Save["stdout", f]"#,
      )
      .unwrap();
      assert!(result.stdout.contains("f[x_] := x + 1"));
      assert!(result.stdout.contains("f[x_, y_] := x + y"));
    }

    #[test]
    fn test_save_multiple_symbols() {
      let result = interpret_with_stdout(
        r#"f[x_] := x + 1; a = 5; Save["stdout", {f, a}]"#,
      )
      .unwrap();
      assert!(result.stdout.contains("f[x_] := x + 1"));
      assert!(result.stdout.contains("a = 5"));
    }

    #[test]
    fn test_save_with_attributes() {
      let result = interpret_with_stdout(
        r#"Attributes[h] = {Listable}; h[x_] := x^3; Save["stdout", h]"#,
      )
      .unwrap();
      assert!(result.stdout.contains("Attributes[h] = {Listable}"));
      assert!(result.stdout.contains("h[x_] := x^3"));
    }

    #[test]
    fn test_save_with_options() {
      let result = interpret_with_stdout(
        r#"Options[h] = {Method -> "Default"}; h[x_] := x^2; Save["stdout", h]"#,
      )
      .unwrap();
      assert!(result.stdout.contains("h[x_] := x^2"));
      assert!(
        result
          .stdout
          .contains(r#"Options[h] = {Method -> "Default"}"#)
      );
    }

    #[test]
    fn test_save_with_head_constraint() {
      let result =
        interpret_with_stdout(r#"g[x_Integer] := x^2; Save["stdout", g]"#)
          .unwrap();
      assert!(result.stdout.contains("g[x_Integer] := x^2"));
    }

    #[test]
    fn test_save_with_default_value() {
      let result =
        interpret_with_stdout(r#"f[x_, y_:3] := x + y; Save["stdout", f]"#)
          .unwrap();
      assert!(result.stdout.contains("f[x_, y_:3] := x + y"));
    }

    #[test]
    fn test_save_literal_dispatch() {
      let result =
        interpret_with_stdout(r#"f[1] = 42; Save["stdout", f]"#).unwrap();
      assert!(result.stdout.contains("f[1] = 42"));
    }

    #[test]
    fn test_save_undefined_symbol() {
      let result =
        interpret_with_stdout(r#"Save["stdout", undefined]"#).unwrap();
      assert_eq!(result.result, "Null");
    }

    #[test]
    fn test_save_string_symbol_name() {
      let result =
        interpret_with_stdout(r#"a = 5; Save["stdout", "a"]"#).unwrap();
      assert!(result.stdout.contains("a = 5"));
    }

    #[test]
    fn test_save_to_file() {
      let result =
        interpret(r#"a = 5; Save["/tmp/woxi_save_test.wl", a]"#).unwrap();
      assert_eq!(result, "Null");
      let content = std::fs::read_to_string("/tmp/woxi_save_test.wl").unwrap();
      assert!(content.contains("a = 5"));
      std::fs::remove_file("/tmp/woxi_save_test.wl").ok();
    }

    #[test]
    fn test_save_roundtrip() {
      let result = interpret(
        r#"f[x_] := x + 1; Save["/tmp/woxi_save_rt.wl", f]; Clear[f]; Get["/tmp/woxi_save_rt.wl"]; f[3]"#,
      )
      .unwrap();
      assert_eq!(result, "4");
      std::fs::remove_file("/tmp/woxi_save_rt.wl").ok();
    }

    #[test]
    fn test_save_multiple_to_file() {
      let result = interpret(
        r#"f[x_] := x^2; a = 10; Save["/tmp/woxi_save_multi.wl", {f, a}]; Clear[f]; Clear[a]; Get["/tmp/woxi_save_multi.wl"]; {f[3], a}"#,
      )
      .unwrap();
      assert_eq!(result, "{9, 10}");
      std::fs::remove_file("/tmp/woxi_save_multi.wl").ok();
    }

    #[test]
    fn test_save_string_value() {
      let result =
        interpret_with_stdout(r#"a = "hello world"; Save["stdout", a]"#)
          .unwrap();
      assert!(result.stdout.contains(r#"a = "hello world""#));
    }

    #[test]
    fn test_save_list_value() {
      let result =
        interpret_with_stdout(r#"b = {1, 2, 3}; Save["stdout", b]"#).unwrap();
      assert!(result.stdout.contains("b = {1, 2, 3}"));
    }

    #[test]
    fn test_save_returns_null() {
      assert_eq!(
        interpret(r#"a = 5; Save["/tmp/woxi_save_null.wl", a]"#).unwrap(),
        "Null"
      );
      std::fs::remove_file("/tmp/woxi_save_null.wl").ok();
    }

    #[test]
    fn test_save_attributes() {
      assert_eq!(
        interpret(r#"Attributes[Save]"#).unwrap(),
        "{HoldRest, Protected}"
      );
    }
  }

  mod format_tests {
    use super::*;

    #[test]
    fn format_number() {
      assert_eq!(interpret("Format[3.14159]").unwrap(), "3.14159");
    }

    #[test]
    fn format_integer() {
      assert_eq!(interpret("Format[42]").unwrap(), "42");
    }

    #[test]
    fn format_rational() {
      assert_eq!(interpret("Format[1/3]").unwrap(), "1/3");
    }

    #[test]
    fn format_symbol() {
      assert_eq!(interpret("Format[Pi]").unwrap(), "Pi");
    }

    #[test]
    fn format_expression() {
      assert_eq!(interpret("Format[x + y]").unwrap(), "x + y");
    }

    #[test]
    fn format_string() {
      assert_eq!(interpret(r#"Format["hello"]"#).unwrap(), "hello");
    }

    #[test]
    fn format_list() {
      assert_eq!(interpret("Format[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
    }

    #[test]
    fn format_with_output_form() {
      assert_eq!(interpret("Format[x^2 + 1, OutputForm]").unwrap(), "1 + x^2");
    }

    #[test]
    fn format_with_input_form() {
      assert_eq!(interpret("Format[x + y, InputForm]").unwrap(), "x + y");
    }

    #[test]
    fn format_attributes() {
      assert_eq!(interpret("Attributes[Format]").unwrap(), "{Protected}");
    }
  }

  mod definition_tests {
    use super::*;

    #[test]
    fn definition_user_function() {
      assert_eq!(
        interpret("f[x_] := x^2; Definition[f]").unwrap(),
        "f[x_] := x^2"
      );
    }

    #[test]
    fn definition_variable() {
      assert_eq!(interpret("a = 5; Definition[a]").unwrap(), "a = 5");
    }

    #[test]
    fn definition_undefined() {
      assert_eq!(interpret("Definition[noSuchThing]; 42").unwrap(), "42");
    }

    #[test]
    fn definition_builtin_attributes() {
      assert_eq!(
        interpret("Definition[Plus]").unwrap(),
        "Attributes[Plus] = {Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"
      );
    }

    #[test]
    fn definition_with_head_constraint() {
      assert_eq!(
        interpret("f[x_Integer] := x^2; Definition[f]").unwrap(),
        "f[x_Integer] := x^2"
      );
    }

    #[test]
    fn definition_multiple_rules() {
      assert_eq!(
        interpret("f[x_] := x^2; f[0] = 42; Definition[f]").unwrap(),
        "f[0] = 42\n \nf[x_] := x^2"
      );
    }

    #[test]
    fn definition_with_user_attributes() {
      assert_eq!(
        interpret("f[x_] := x^2; SetAttributes[f, Listable]; Definition[f]")
          .unwrap(),
        "Attributes[f] = {Listable}\n \nf[x_] := x^2"
      );
    }

    #[test]
    fn definition_attributes() {
      assert_eq!(
        interpret("Attributes[Definition]").unwrap(),
        "{HoldAll, Protected}"
      );
    }
  }
}
