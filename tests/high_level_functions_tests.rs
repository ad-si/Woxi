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
      assert!(interpret("Prime[0]").is_err());
      assert!(interpret("Prime[-1]").is_err());
      assert!(interpret("Prime[1.5]").is_err());
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
}
