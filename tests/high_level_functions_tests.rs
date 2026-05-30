use woxi::interpret;

mod high_level_functions_tests {
  use super::*;

  mod entropy_tests {
    use super::*;

    #[test]
    fn test_basic_symbolic() {
      // Shannon entropy in nats; exact symbolic form.
      assert_eq!(
        interpret("Entropy[{a,b,b,c,c,c}]").unwrap(),
        "(-2*Log[2] - 3*Log[3])/6 + Log[6]"
      );
    }

    #[test]
    fn test_two_classes() {
      assert_eq!(interpret("Entropy[{a,a,b,b}]").unwrap(), "-Log[2] + Log[4]");
    }

    #[test]
    fn test_all_distinct() {
      assert_eq!(interpret("Entropy[{1,2,3,4}]").unwrap(), "Log[4]");
    }

    #[test]
    fn test_single_element() {
      assert_eq!(interpret("Entropy[{x}]").unwrap(), "0");
    }

    #[test]
    fn test_all_same() {
      assert_eq!(interpret("Entropy[{1,1,1}]").unwrap(), "0");
    }

    #[test]
    fn test_empty_list() {
      assert_eq!(interpret("Entropy[{}]").unwrap(), "0");
    }

    #[test]
    fn test_explicit_base_2() {
      assert_eq!(
        interpret("Entropy[2,{a,b,b,c,c,c}]").unwrap(),
        "(-2 - (3*Log[3])/Log[2])/6 + Log[6]/Log[2]"
      );
    }

    #[test]
    fn test_base_e_matches_default() {
      assert_eq!(
        interpret("Entropy[E,{a,b,b,c,c,c}]").unwrap(),
        "(-2*Log[2] - 3*Log[3])/6 + Log[6]"
      );
    }

    #[test]
    fn test_numeric() {
      assert_eq!(
        interpret("N[Entropy[{a,b,b,c,c,c}]]").unwrap(),
        "1.0114042647073518"
      );
    }
  }

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

  mod trig_negation_identities {
    // Regression: Sin[-x], Cos[-x], Tan[-x], etc. used to return unchanged
    // instead of applying the odd/even symmetries.
    use super::*;

    #[test]
    fn sin_is_odd() {
      assert_eq!(interpret("Sin[-x]").unwrap(), "-Sin[x]");
      assert_eq!(interpret("Sin[-Pi/6]").unwrap(), "-1/2");
    }

    #[test]
    fn cos_is_even() {
      assert_eq!(interpret("Cos[-x]").unwrap(), "Cos[x]");
      assert_eq!(interpret("Cos[-Pi/3]").unwrap(), "1/2");
    }

    #[test]
    fn tan_is_odd() {
      assert_eq!(interpret("Tan[-x]").unwrap(), "-Tan[x]");
    }

    #[test]
    fn sec_is_even() {
      assert_eq!(interpret("Sec[-x]").unwrap(), "Sec[x]");
    }

    #[test]
    fn csc_is_odd() {
      assert_eq!(interpret("Csc[-x]").unwrap(), "-Csc[x]");
    }

    #[test]
    fn cot_is_odd() {
      assert_eq!(interpret("Cot[-x]").unwrap(), "-Cot[x]");
    }

    #[test]
    fn arcsin_is_odd() {
      assert_eq!(interpret("ArcSin[-x]").unwrap(), "-ArcSin[x]");
    }

    #[test]
    fn arctan_is_odd() {
      assert_eq!(interpret("ArcTan[-x]").unwrap(), "-ArcTan[x]");
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
    #[test]
    fn test_arccosh_bigfloat_zero() {
      // ArcCosh[0``α] = (Pi/2 at α-digit accuracy) * I — wolframscript
      // returns the high-precision Pi/2 imaginary value.
      let s = interpret("ArcCosh[0``38.]").unwrap();
      assert!(
        s.starts_with("1.5707963267948966192313216916397514")
          && s.ends_with("*I"),
        "got: {}",
        s
      );
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
    #[test]
    fn test_capitalize_threads_list() {
      assert_eq!(
        interpret(r#"Capitalize[{"abc", "def"}]"#).unwrap(),
        "{Abc, Def}"
      );
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
    #[test]
    fn test_decapitalize_threads_list() {
      assert_eq!(
        interpret(r#"Decapitalize[{"ABC", "DEF"}]"#).unwrap(),
        "{aBC, dEF}"
      );
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

  mod string_backtick_escape_tests {
    use super::*;
    #[test]
    fn test_backtick_escape_length() {
      // `\`` parses as a single (private-use) character, not as `\` + ` ``.
      assert_eq!(interpret(r#"StringLength["\`"]"#).unwrap(), "1");
    }
    #[test]
    fn test_backtick_escape_renders_as_escape() {
      // The parsed character renders back as `\`` in OutputForm
      // (matching wolframscript).
      assert_eq!(interpret(r#"Characters["\`"]"#).unwrap(), "{\\`}");
    }
    #[test]
    fn test_backtick_escape_distinct_from_literal_backtick() {
      // U+F7CD (parsed `\``) is distinct from U+0060 (literal backtick).
      assert_eq!(interpret(r#"ToCharacterCode["\`"]"#).unwrap(), "{63437}");
      assert_eq!(interpret(r#"ToCharacterCode["`"]"#).unwrap(), "{96}");
    }
    #[test]
    fn test_mixed_backslash_backtick_space() {
      assert_eq!(interpret(r#"StringLength["\\\` "]"#).unwrap(), "3");
    }
  }

  mod string_box_escape_tests {
    use super::*;
    #[test]
    fn test_box_open_close_length() {
      // `\(` and `\)` each parse as a single private-use codepoint.
      assert_eq!(interpret(r#"StringLength["\(A\)"]"#).unwrap(), "3");
    }
    #[test]
    fn test_box_open_close_codepoints() {
      // Wolfram maps `\(` → U+F7C9 (63433), `\)` → U+F7C0 (63424).
      assert_eq!(
        interpret(r#"ToCharacterCode["\(A\)"]"#).unwrap(),
        "{63433, 65, 63424}"
      );
    }
    #[test]
    fn test_box_start_sep_codepoints() {
      // `\!` → U+F7C1 (63425), `\*` → U+F7C8 (63432).
      assert_eq!(interpret(r#"ToCharacterCode["\!"]"#).unwrap(), "{63425}");
      assert_eq!(interpret(r#"ToCharacterCode["\*"]"#).unwrap(), "{63432}");
    }
    #[test]
    fn test_box_chars_render_as_escapes() {
      // The parsed codepoints render back as `\(`, `\)` etc. in OutputForm.
      assert_eq!(
        interpret(r#"FromCharacterCode[{63433, 65, 63424}]"#).unwrap(),
        "\\(A\\)"
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
    #[test]
    fn test_string_delete_threads_list() {
      assert_eq!(
        interpret(r#"StringDelete[{"abcabc", "xbyb"}, "b"]"#).unwrap(),
        "{acac, xy}"
      );
    }
    #[test]
    fn test_string_delete_list_of_patterns() {
      assert_eq!(
        interpret(r#"StringDelete["hello world", {"l", "o"}]"#).unwrap(),
        "he wrd"
      );
    }
  }

  mod word_boundary_tests {
    use super::*;
    #[test]
    fn test_word_boundary_basic() {
      assert_eq!(
        interpret(r#"StringReplace["the cat", WordBoundary -> "|"]"#).unwrap(),
        "|the| |cat|"
      );
    }
    #[test]
    fn test_word_boundary_multiple_words() {
      assert_eq!(
        interpret(r#"StringReplace["the cat sat", WordBoundary -> "X"]"#)
          .unwrap(),
        "XtheX XcatX XsatX"
      );
    }
    #[test]
    fn test_word_boundary_single_char() {
      assert_eq!(
        interpret(r#"StringReplace["a", WordBoundary -> "|"]"#).unwrap(),
        "|a|"
      );
    }
    #[test]
    fn test_word_boundary_empty_string() {
      assert_eq!(
        interpret(r#"StringReplace["", WordBoundary -> "|"]"#).unwrap(),
        ""
      );
    }
    #[test]
    fn test_word_boundary_punctuation() {
      // Digits and letters are word characters; the dot is not.
      assert_eq!(
        interpret(r#"StringReplace["the.cat", WordBoundary -> "|"]"#).unwrap(),
        "|the|.|cat|"
      );
      assert_eq!(
        interpret(r#"StringReplace["one2three", WordBoundary -> "|"]"#).unwrap(),
        "|one2three|"
      );
    }
    #[test]
    fn test_word_boundary_in_string_expression() {
      assert_eq!(
        interpret(r#"StringReplace["foo bar", WordBoundary ~~ "b" -> "X"]"#)
          .unwrap(),
        "foo Xar"
      );
    }
    #[test]
    fn test_word_boundary_string_cases() {
      assert_eq!(
        interpret(r#"StringCases["the cat", WordBoundary ~~ LetterCharacter]"#)
          .unwrap(),
        "{t, c}"
      );
    }
    #[test]
    fn test_word_boundary_max_replacements() {
      assert_eq!(
        interpret(r#"StringReplace["the cat sat", WordBoundary -> "|", 3]"#)
          .unwrap(),
        "|the| |cat sat"
      );
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
    #[test]
    fn test_commonest_first_appearance_order() {
      // When n covers more than one count tier, the chosen elements
      // come back in first-appearance order, not count order.
      assert_eq!(
        interpret("Commonest[{b, a, c, 2, a, b, 1, 2}, 4]").unwrap(),
        "{b, a, c, 2}"
      );
      assert_eq!(
        interpret("Commonest[{b, a, c, 2, a, b, 1, 2}, 5]").unwrap(),
        "{b, a, c, 2, 1}"
      );
    }
    #[test]
    fn test_commonest_up_to() {
      // UpTo[n] returns at most n distinct elements (input has 5 distinct
      // values, so UpTo[6] returns all 5 in first-appearance order).
      assert_eq!(
        interpret("Commonest[{b, a, c, 2, a, b, 1, 2}, UpTo[6]]").unwrap(),
        "{b, a, c, 2, 1}"
      );
      assert_eq!(
        interpret("Commonest[{b, a, c, 2, a, b, 1, 2}, UpTo[3]]").unwrap(),
        "{b, a, 2}"
      );
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
    #[test]
    fn test_digit_sum_mixed_radix_hms() {
      // wolframscript: DigitSum[102341, MixedRadix[{24, 60, 60}]] = 71.
      // 102341 = 1*86400 + 4*3600 + 25*60 + 41 → digits {1, 4, 25, 41}.
      assert_eq!(
        interpret("DigitSum[102341, MixedRadix[{24, 60, 60}]]").unwrap(),
        "71"
      );
    }
    #[test]
    fn test_digit_sum_mixed_radix_simple() {
      // 42 with bases {2, 5}: digits {4, 0, 2}, sum 6.
      assert_eq!(interpret("DigitSum[42, MixedRadix[{2, 5}]]").unwrap(), "6");
    }
    #[test]
    fn test_digit_sum_mixed_radix_extra_base() {
      // Bases {24, 60, 60, 1000} for 102341: digits {1, 42, 341}, sum 384.
      assert_eq!(
        interpret("DigitSum[102341, MixedRadix[{24, 60, 60, 1000}]]").unwrap(),
        "384"
      );
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

    #[test]
    fn test_sqrt_70_periodic() {
      // ContinuedFraction[Sqrt[70]] should expose the periodic
      // expansion: {8, {2, 1, 2, 1, 2, 16}}.
      assert_eq!(
        interpret("ContinuedFraction[Sqrt[70]]").unwrap(),
        "{8, {2, 1, 2, 1, 2, 16}}"
      );
    }

    #[test]
    fn test_sqrt_2_periodic() {
      assert_eq!(interpret("ContinuedFraction[Sqrt[2]]").unwrap(), "{1, {2}}");
    }

    #[test]
    fn test_sqrt_3_periodic() {
      assert_eq!(
        interpret("ContinuedFraction[Sqrt[3]]").unwrap(),
        "{1, {1, 2}}"
      );
    }

    #[test]
    fn test_quadratic_irrational() {
      // ContinuedFraction[(p + q·√d)/r] picks up the periodic expansion.
      // Regression for mathics numbers/integer.py:?? (ContinuedFraction
      // docs).
      assert_eq!(
        interpret("ContinuedFraction[(1 + 2 Sqrt[3])/5]").unwrap(),
        "{0, 1, {8, 3, 34, 3}}"
      );
      // Purely periodic: golden ratio needs a prefix term.
      assert_eq!(
        interpret("ContinuedFraction[(1 + Sqrt[5])/2]").unwrap(),
        "{1, {1}}"
      );
      assert_eq!(
        interpret("ContinuedFraction[(-1 + Sqrt[5])/2]").unwrap(),
        "{0, {1}}"
      );
    }

    #[test]
    fn test_sqrt_of_perfect_square() {
      // Perfect-square input resolves to the Integer case via Sqrt.
      assert_eq!(interpret("ContinuedFraction[Sqrt[25]]").unwrap(), "{5}");
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

  mod convergents_tests {
    use super::*;

    #[test]
    fn from_list() {
      assert_eq!(
        interpret("Convergents[{1, 2, 3, 4}]").unwrap(),
        "{1, 3/2, 10/7, 43/30}"
      );
    }

    #[test]
    fn pi_approximations() {
      assert_eq!(
        interpret("Convergents[{3, 7, 15, 1}]").unwrap(),
        "{3, 22/7, 333/106, 355/113}"
      );
    }

    #[test]
    fn five_terms() {
      assert_eq!(
        interpret("Convergents[{1, 2, 3, 4, 5}]").unwrap(),
        "{1, 3/2, 10/7, 43/30, 225/157}"
      );
    }

    #[test]
    fn from_number() {
      assert_eq!(
        interpret("Convergents[Pi, 5]").unwrap(),
        "{3, 22/7, 333/106, 355/113, 103993/33102}"
      );
    }

    #[test]
    fn single_element() {
      assert_eq!(interpret("Convergents[{5}]").unwrap(), "{5}");
    }

    #[test]
    fn sqrt2() {
      assert_eq!(
        interpret("Convergents[Sqrt[2], 5]").unwrap(),
        "{1, 3/2, 7/5, 17/12, 41/29}"
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
    #[test]
    fn test_bernoulli_poly_0() {
      assert_eq!(interpret("BernoulliB[0, z]").unwrap(), "1");
    }
    #[test]
    fn test_bernoulli_poly_1() {
      assert_eq!(interpret("BernoulliB[1, z]").unwrap(), "-1/2 + z");
    }
    #[test]
    fn test_bernoulli_poly_2() {
      assert_eq!(interpret("BernoulliB[2, z]").unwrap(), "1/6 - z + z^2");
    }
    #[test]
    fn test_bernoulli_poly_3() {
      assert_eq!(
        interpret("BernoulliB[3, z]").unwrap(),
        "z/2 - (3*z^2)/2 + z^3"
      );
    }
    #[test]
    fn test_bernoulli_poly_table() {
      assert_eq!(
        interpret("Table[BernoulliB[k, z], {k, 0, 3}]").unwrap(),
        "{1, -1/2 + z, 1/6 - z + z^2, z/2 - (3*z^2)/2 + z^3}"
      );
    }
    #[test]
    fn test_bernoulli_poly_numeric() {
      assert_eq!(interpret("BernoulliB[2, 3]").unwrap(), "37/6");
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

  // ─── FrobeniusSolve ──────────────────────────────────────────────────
  mod frobenius_solve_tests {
    use super::*;
    #[test]
    fn test_frobenius_solve_basic() {
      assert_eq!(
        interpret("FrobeniusSolve[{2, 3, 4}, 10]").unwrap(),
        "{{0, 2, 1}, {1, 0, 2}, {2, 2, 0}, {3, 0, 1}, {5, 0, 0}}"
      );
      assert_eq!(
        interpret("FrobeniusSolve[{2, 3, 5}, 7]").unwrap(),
        "{{1, 0, 1}, {2, 1, 0}}"
      );
    }
    #[test]
    fn test_frobenius_solve_no_solution() {
      assert_eq!(interpret("FrobeniusSolve[{2, 4}, 7]").unwrap(), "{}");
    }
    #[test]
    fn test_frobenius_solve_zero_target() {
      assert_eq!(interpret("FrobeniusSolve[{2, 4}, 0]").unwrap(), "{{0, 0}}");
      assert_eq!(interpret("FrobeniusSolve[{3}, 0]").unwrap(), "{{0}}");
    }
    #[test]
    fn test_frobenius_solve_count_large() {
      // Same input as the user's example - count must match wolframscript.
      assert_eq!(
        interpret(
          "Length[FrobeniusSolve[{230, 306, 392, 410, 574, 780, 750, 850}, 10000]]"
        )
        .unwrap(),
        "4674"
      );
    }
    #[test]
    fn test_frobenius_solve_limit() {
      assert_eq!(
        interpret("FrobeniusSolve[{2, 3, 4}, 10, 2]").unwrap(),
        "{{0, 2, 1}, {1, 0, 2}}"
      );
      assert_eq!(
        interpret("FrobeniusSolve[{2, 3, 4}, 10, All]").unwrap(),
        "{{0, 2, 1}, {1, 0, 2}, {2, 2, 0}, {3, 0, 1}, {5, 0, 0}}"
      );
    }
    #[test]
    fn test_frobenius_solve_bad_input() {
      assert_eq!(
        interpret("FrobeniusSolve[{}, 5]").unwrap(),
        "FrobeniusSolve[{}, 5]"
      );
      assert_eq!(
        interpret("FrobeniusSolve[{0, 3}, 6]").unwrap(),
        "FrobeniusSolve[{0, 3}, 6]"
      );
      assert_eq!(
        interpret("FrobeniusSolve[{2, -3}, 6]").unwrap(),
        "FrobeniusSolve[{2, -3}, 6]"
      );
      assert_eq!(
        interpret("FrobeniusSolve[{2, 3, 4}, 10, 0]").unwrap(),
        "FrobeniusSolve[{2, 3, 4}, 10, 0]"
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

  // ─── Scan ────────────────────────────────────────────────────────────
  mod scan_tests {
    use super::*;
    #[test]
    fn test_scan_list() {
      // Scan returns Null and applies function for side effects
      assert_eq!(interpret("Scan[Print, {1, 2, 3}]").unwrap(), "\0");
    }
    #[test]
    fn test_scan_non_list_expression() {
      // Scan should work on any expression, not just lists
      assert_eq!(interpret("Scan[Print, f[a, b, c]]").unwrap(), "\0");
    }
    #[test]
    fn test_scan_power_expression() {
      // Scan over Power[x, -1] should iterate over parts x and -1
      assert_eq!(interpret("Scan[Print, Power[x, -1]]").unwrap(), "\0");
    }
    #[test]
    fn test_scan_with_throw_in_non_list() {
      // Regression test for issue #75:
      // Throw inside Scan on a non-list expression must propagate to Catch
      assert_eq!(
        interpret(
          "FFunctionOfExpnQ[u_] := Catch[Scan[Function[Throw[False]],u];True]; FFunctionOfExpnQ[1/x]"
        )
        .unwrap(),
        "False"
      );
    }
    #[test]
    fn test_scan_with_throw_in_list() {
      // Throw inside Scan on a list must also propagate to Catch
      assert_eq!(
        interpret("Catch[Scan[Function[Throw[False]], {1, 2, 3}]; True]")
          .unwrap(),
        "False"
      );
    }
    #[test]
    fn test_scan_atom() {
      // Scan on an atom (no parts) should return Null without error
      assert_eq!(interpret("Scan[Print, 42]").unwrap(), "\0");
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
    #[test]
    fn test_integer_partitions_max_results() {
      assert_eq!(
        interpret("IntegerPartitions[5, All, All, 3]").unwrap(),
        "{{5}, {4, 1}, {3, 2}}"
      );
    }
    #[test]
    fn test_integer_partitions_signed_elements() {
      assert_eq!(
        interpret("IntegerPartitions[4, {2}, {-1, 0, 1, 4, 5}]").unwrap(),
        "{{5, -1}, {4, 0}}"
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
      assert_eq!(result.result, "\0");
      assert!(result.stdout.contains("a = 5"));
    }

    #[test]
    fn test_save_function_to_stdout() {
      let result =
        interpret_with_stdout(r#"f[x_] := x + 1; Save["stdout", f]"#).unwrap();
      assert_eq!(result.result, "\0");
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
      assert_eq!(result.result, "\0");
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
      assert_eq!(result, "\0");
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
        "\0"
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
      assert_eq!(
        interpret("Format[x^2 + 1, OutputForm]").unwrap(),
        "     2\n1 + x"
      );
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
        "Attributes[Plus] = {Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}\n \nDefault[Plus] := 0"
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

  mod default_tests {
    use super::*;

    #[test]
    fn default_plus() {
      assert_eq!(interpret("Default[Plus]").unwrap(), "0");
    }

    #[test]
    fn default_times() {
      assert_eq!(interpret("Default[Times]").unwrap(), "1");
    }

    #[test]
    fn default_unevaluated() {
      assert_eq!(interpret("Default[Power]").unwrap(), "Default[Power]");
    }
  }

  mod decompose_tests {
    use super::*;
    #[test]
    fn test_basic_composition() {
      assert_eq!(interpret("Decompose[x^2 + 1, x]").unwrap(), "{1 + x, x^2}");
    }
    #[test]
    fn test_polynomial_in_x_squared() {
      assert_eq!(
        interpret("Decompose[x^4 + x^2, x]").unwrap(),
        "{x + x^2, x^2}"
      );
    }
    #[test]
    fn test_three_level_decomposition() {
      assert_eq!(
        interpret("Decompose[(x^2 + x)^4 + 1, x]").unwrap(),
        "{1 + x, x^4, x + x^2}"
      );
    }
    #[test]
    fn test_cubic_plus_one() {
      assert_eq!(interpret("Decompose[x^3 + 1, x]").unwrap(), "{1 + x, x^3}");
    }
    #[test]
    fn test_indecomposable() {
      assert_eq!(
        interpret("Decompose[x^4 + x^3 + x^2 + x, x]").unwrap(),
        "{x + x^2 + x^3 + x^4}"
      );
    }
    #[test]
    fn test_monomial_indecomposable() {
      assert_eq!(interpret("Decompose[x^6, x]").unwrap(), "{x^6}");
    }
    #[test]
    fn test_constant() {
      assert_eq!(interpret("Decompose[5, x]").unwrap(), "{5}");
    }
    #[test]
    fn test_linear() {
      assert_eq!(interpret("Decompose[x, x]").unwrap(), "{x}");
    }
    #[test]
    fn test_squared_composition() {
      assert_eq!(
        interpret("Decompose[x^4 + 2*x^3 + x^2, x]").unwrap(),
        "{x^2, x + x^2}"
      );
    }
    #[test]
    fn test_shifted_power() {
      assert_eq!(
        interpret("Decompose[(x + 1)^4, x]").unwrap(),
        "{1 + 2*x + x^2, 2*x + x^2}"
      );
    }
    #[test]
    fn test_sixth_power_shifted() {
      assert_eq!(
        interpret("Decompose[(x + 1)^6, x]").unwrap(),
        "{1 + 3*x + 3*x^2 + x^3, 2*x + x^2}"
      );
    }
    #[test]
    fn test_non_monic() {
      assert_eq!(
        interpret("Decompose[2*x^4 + 3*x^2 + 1, x]").unwrap(),
        "{1 + 3*x + 2*x^2, x^2}"
      );
    }
    #[test]
    fn test_x8_x4_1() {
      assert_eq!(
        interpret("Decompose[x^8 + x^4 + 1, x]").unwrap(),
        "{1 + x + x^2, x^4}"
      );
    }
    #[test]
    fn test_with_negative_coeffs() {
      assert_eq!(
        interpret("Decompose[x^6 - 2*x^5 + x^4, x]").unwrap(),
        "{x^2, -x^2 + x^3}"
      );
      assert_eq!(
        interpret("Decompose[x^4 - 2*x^3 + 3*x^2 - 2*x + 1, x]").unwrap(),
        "{1 + 2*x + x^2, -x + x^2}"
      );
    }
    #[test]
    fn test_rational_coefficients() {
      assert_eq!(
        interpret("Decompose[(x^2 + 1/2)^2, x]").unwrap(),
        "{1/4 + x + x^2, x^2}"
      );
    }
    #[test]
    fn test_cubic_indecomposable() {
      assert_eq!(
        interpret("Decompose[x^3 + 3*x^2 + 3*x + 1, x]").unwrap(),
        "{1 + 3*x + 3*x^2 + x^3}"
      );
    }
    #[test]
    fn test_x6_plus_1() {
      assert_eq!(interpret("Decompose[x^6 + 1, x]").unwrap(), "{1 + x, x^6}");
    }
    #[test]
    fn test_binomial_cubed_in_x2() {
      assert_eq!(
        interpret("Decompose[x^6 + 3*x^4 + 3*x^2 + 1, x]").unwrap(),
        "{1 + 3*x + 3*x^2 + x^3, x^2}"
      );
    }
  }

  mod factor_square_free_list_tests {
    use super::*;

    #[test]
    fn test_zero() {
      assert_eq!(interpret("FactorSquareFreeList[0]").unwrap(), "{{0, 1}}");
    }

    #[test]
    fn test_integer() {
      assert_eq!(interpret("FactorSquareFreeList[5]").unwrap(), "{{5, 1}}");
    }

    #[test]
    fn test_negative_integer() {
      assert_eq!(interpret("FactorSquareFreeList[-7]").unwrap(), "{{-7, 1}}");
    }

    #[test]
    fn test_single_variable() {
      assert_eq!(
        interpret("FactorSquareFreeList[x]").unwrap(),
        "{{1, 1}, {x, 1}}"
      );
    }

    #[test]
    fn test_x_power() {
      assert_eq!(
        interpret("FactorSquareFreeList[x^6]").unwrap(),
        "{{1, 1}, {x, 6}}"
      );
    }

    #[test]
    fn test_perfect_square() {
      assert_eq!(
        interpret("FactorSquareFreeList[(1+x)^2]").unwrap(),
        "{{1, 1}, {1 + x, 2}}"
      );
    }

    #[test]
    fn test_mixed_multiplicities() {
      // x^5-x^3-x^2+1 = (x-1)^2 * (x^3+2x^2+2x+1)
      assert_eq!(
        interpret("FactorSquareFreeList[x^5 - x^3 - x^2 + 1]").unwrap(),
        "{{1, 1}, {-1 + x, 2}, {1 + 2*x + 2*x^2 + x^3, 1}}"
      );
    }

    #[test]
    fn test_with_numeric_content() {
      // 12*x^4-12*x^2 = 12*x^2*(x^2-1)
      assert_eq!(
        interpret("FactorSquareFreeList[12*x^4 - 12*x^2]").unwrap(),
        "{{12, 1}, {-1 + x^2, 1}, {x, 2}}"
      );
    }

    #[test]
    fn test_three_distinct_multiplicities() {
      // 8*x^5*(x+1)^3*(x-2)^2
      assert_eq!(
        interpret("FactorSquareFreeList[8*x^5*(x+1)^3*(x-2)^2]").unwrap(),
        "{{8, 1}, {-2 + x, 2}, {x, 5}, {1 + x, 3}}"
      );
    }

    #[test]
    fn test_ordering_by_constant_term() {
      // (x-3)^2*(x+2)*(x-5)^3 - sorted by constant term: -5, -3, 2
      assert_eq!(
        interpret("FactorSquareFreeList[(x-3)^2*(x+2)*(x-5)^3]").unwrap(),
        "{{1, 1}, {-5 + x, 3}, {-3 + x, 2}, {2 + x, 1}}"
      );
    }

    #[test]
    fn test_irreducible_square_free() {
      // x^4-1 is already square-free
      assert_eq!(
        interpret("FactorSquareFreeList[x^4 - 1]").unwrap(),
        "{{1, 1}, {-1 + x^4, 1}}"
      );
    }

    #[test]
    fn test_linear_polynomial() {
      assert_eq!(
        interpret("FactorSquareFreeList[3*x + 6]").unwrap(),
        "{{3, 1}, {2 + x, 1}}"
      );
    }

    #[test]
    fn test_quadratic_with_square_factor() {
      // (x^2+3)*(x^2-2)^2
      assert_eq!(
        interpret("FactorSquareFreeList[(x^2+3)*(x^2-2)^2]").unwrap(),
        "{{1, 1}, {-2 + x^2, 2}, {3 + x^2, 1}}"
      );
    }

    #[test]
    fn test_x_times_linear_squared() {
      // x^3*(x+1)^2*(x-1)
      assert_eq!(
        interpret("FactorSquareFreeList[x^3*(x+1)^2*(x-1)]").unwrap(),
        "{{1, 1}, {-1 + x, 1}, {x, 3}, {1 + x, 2}}"
      );
    }

    #[test]
    fn test_negative_leading() {
      assert_eq!(
        interpret("FactorSquareFreeList[-x^2 + 1]").unwrap(),
        "{{-1, 1}, {-1 + x^2, 1}}"
      );
    }
  }

  mod inverse_jacobi_tests {
    use super::*;

    // ─── InverseJacobiSN ─────────────────────────────────
    #[test]
    fn sn_zero() {
      assert_eq!(interpret("InverseJacobiSN[0, m]").unwrap(), "0");
    }

    #[test]
    fn sn_one() {
      assert_eq!(interpret("InverseJacobiSN[1, m]").unwrap(), "EllipticK[m]");
    }

    #[test]
    fn sn_m_zero() {
      assert_eq!(interpret("InverseJacobiSN[x, 0]").unwrap(), "ArcSin[x]");
    }

    #[test]
    fn sn_m_one() {
      assert_eq!(interpret("InverseJacobiSN[x, 1]").unwrap(), "ArcTanh[x]");
    }

    #[test]
    fn sn_numeric() {
      let r = interpret("InverseJacobiSN[0.5, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.5306368995398673).abs() < 1e-6);
    }

    // ─── InverseJacobiCN ─────────────────────────────────
    #[test]
    fn cn_one() {
      assert_eq!(interpret("InverseJacobiCN[1, m]").unwrap(), "0");
    }

    #[test]
    fn cn_zero() {
      assert_eq!(interpret("InverseJacobiCN[0, m]").unwrap(), "EllipticK[m]");
    }

    #[test]
    fn cn_m_zero() {
      assert_eq!(interpret("InverseJacobiCN[x, 0]").unwrap(), "ArcCos[x]");
    }

    #[test]
    fn cn_m_one() {
      assert_eq!(interpret("InverseJacobiCN[x, 1]").unwrap(), "ArcSech[x]");
    }

    #[test]
    fn cn_numeric() {
      let r = interpret("InverseJacobiCN[0.5, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 1.0991352230920428).abs() < 1e-6);
    }

    // ─── InverseJacobiDN ─────────────────────────────────
    #[test]
    fn dn_one() {
      assert_eq!(interpret("InverseJacobiDN[1, m]").unwrap(), "0");
    }

    #[test]
    fn dn_m_one() {
      assert_eq!(interpret("InverseJacobiDN[x, 1]").unwrap(), "ArcSech[x]");
    }

    #[test]
    fn dn_numeric() {
      let r = interpret("InverseJacobiDN[0.9, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.9566256006832332).abs() < 1e-5);
    }

    // ─── InverseJacobiCD ─────────────────────────────────
    #[test]
    fn cd_one() {
      assert_eq!(interpret("InverseJacobiCD[1, m]").unwrap(), "0");
    }

    #[test]
    fn cd_zero() {
      assert_eq!(interpret("InverseJacobiCD[0, m]").unwrap(), "EllipticK[m]");
    }

    #[test]
    fn cd_m_zero() {
      assert_eq!(interpret("InverseJacobiCD[x, 0]").unwrap(), "ArcCos[x]");
    }

    #[test]
    fn cd_numeric() {
      let r = interpret("InverseJacobiCD[0.5, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 1.1832525486389236).abs() < 1e-5);
    }

    // ─── InverseJacobiSC ─────────────────────────────────
    #[test]
    fn sc_zero() {
      assert_eq!(interpret("InverseJacobiSC[0, m]").unwrap(), "0");
    }

    #[test]
    fn sc_m_zero() {
      assert_eq!(interpret("InverseJacobiSC[x, 0]").unwrap(), "ArcTan[x]");
    }

    #[test]
    fn sc_m_one() {
      assert_eq!(interpret("InverseJacobiSC[x, 1]").unwrap(), "ArcSinh[x]");
    }

    #[test]
    fn sc_numeric() {
      let r = interpret("InverseJacobiSC[0.5, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.4685566171679445).abs() < 1e-6);
    }

    // ─── InverseJacobiCS ─────────────────────────────────
    #[test]
    fn cs_m_zero() {
      assert_eq!(interpret("InverseJacobiCS[x, 0]").unwrap(), "ArcCot[x]");
    }

    #[test]
    fn cs_numeric() {
      let r = interpret("InverseJacobiCS[2.0, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.4685566171679443).abs() < 1e-6);
    }

    // ─── InverseJacobiSD ─────────────────────────────────
    #[test]
    fn sd_zero() {
      assert_eq!(interpret("InverseJacobiSD[0, m]").unwrap(), "0");
    }

    #[test]
    fn sd_m_zero() {
      assert_eq!(interpret("InverseJacobiSD[x, 0]").unwrap(), "ArcSin[x]");
    }

    #[test]
    fn sd_m_one() {
      assert_eq!(interpret("InverseJacobiSD[x, 1]").unwrap(), "ArcSinh[x]");
    }

    #[test]
    fn sd_numeric() {
      let r = interpret("InverseJacobiSD[0.5, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.5094709002418752).abs() < 1e-5);
    }

    // ─── InverseJacobiDS ─────────────────────────────────
    #[test]
    fn ds_m_zero() {
      assert_eq!(interpret("InverseJacobiDS[x, 0]").unwrap(), "ArcCsc[x]");
    }

    #[test]
    fn ds_m_one() {
      assert_eq!(interpret("InverseJacobiDS[x, 1]").unwrap(), "ArcCsch[x]");
    }

    #[test]
    fn ds_numeric() {
      let r = interpret("InverseJacobiDS[2.0, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.5094709002418752).abs() < 1e-5);
    }

    // ─── InverseJacobiNS ─────────────────────────────────
    #[test]
    fn ns_m_zero() {
      assert_eq!(interpret("InverseJacobiNS[x, 0]").unwrap(), "ArcCsc[x]");
    }

    #[test]
    fn ns_m_one() {
      assert_eq!(interpret("InverseJacobiNS[x, 1]").unwrap(), "ArcCoth[x]");
    }

    #[test]
    fn ns_numeric() {
      let r = interpret("InverseJacobiNS[2.0, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.5306368995398673).abs() < 1e-6);
    }

    // ─── InverseJacobiNC ─────────────────────────────────
    #[test]
    fn nc_one() {
      assert_eq!(interpret("InverseJacobiNC[1, m]").unwrap(), "0");
    }

    #[test]
    fn nc_m_zero() {
      assert_eq!(interpret("InverseJacobiNC[x, 0]").unwrap(), "ArcSec[x]");
    }

    #[test]
    fn nc_m_one() {
      assert_eq!(interpret("InverseJacobiNC[x, 1]").unwrap(), "ArcCosh[x]");
    }

    #[test]
    fn nc_numeric() {
      let r = interpret("InverseJacobiNC[2.0, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 1.0991352230920428).abs() < 1e-5);
    }

    // ─── InverseJacobiND ─────────────────────────────────
    #[test]
    fn nd_one() {
      assert_eq!(interpret("InverseJacobiND[1, m]").unwrap(), "0");
    }

    #[test]
    fn nd_m_one() {
      assert_eq!(interpret("InverseJacobiND[x, 1]").unwrap(), "ArcCosh[x]");
    }

    #[test]
    fn nd_numeric() {
      let r = interpret("InverseJacobiND[1.05, 0.5]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.45324704270437755).abs() < 1e-5);
    }

    // ─── InverseJacobiDC ─────────────────────────────────
    #[test]
    fn dc_one() {
      assert_eq!(interpret("InverseJacobiDC[1, m]").unwrap(), "0");
    }

    #[test]
    fn dc_numeric() {
      let r = interpret("InverseJacobiDC[1.2, 0.3]").unwrap();
      let val: f64 = r.parse().unwrap();
      assert!((val - 0.6849087487827934).abs() < 1e-5);
    }
  }

  mod full_definition_tests {
    use super::*;

    #[test]
    fn basic_function() {
      let r = interpret("Clear[f]; f[x_] := x^2; FullDefinition[f]").unwrap();
      assert_eq!(r, "f[x_] := x^2");
    }

    #[test]
    fn with_dependency() {
      let r = interpret(
        "Clear[f, g]; f[x_] := x^2; g[x_] := f[x] + 1; FullDefinition[g]",
      )
      .unwrap();
      assert!(
        r.contains("g[x_] := f[x] + 1"),
        "should contain g's def: {}",
        r
      );
      assert!(r.contains("f[x_] := x^2"), "should contain f's def: {}", r);
    }

    #[test]
    fn chain_dependencies() {
      let r = interpret(
        "Clear[f, g, h]; f[x_] := x^2; g[x_] := f[x] + 1; h[x_] := g[x] + f[x]; FullDefinition[h]",
      )
      .unwrap();
      assert!(
        r.contains("h[x_] := g[x] + f[x]"),
        "should contain h: {}",
        r
      );
      assert!(r.contains("g[x_] := f[x] + 1"), "should contain g: {}", r);
      assert!(r.contains("f[x_] := x^2"), "should contain f: {}", r);
    }

    #[test]
    fn no_dependencies() {
      let r = interpret("Clear[f]; f[x_] := x^2; FullDefinition[f]").unwrap();
      assert_eq!(r, "f[x_] := x^2");
    }

    #[test]
    fn undefined_symbol() {
      let r = interpret("Clear[xyz]; FullDefinition[xyz]").unwrap();
      assert_eq!(r, "");
    }

    #[test]
    fn builtin_symbol() {
      let r = interpret("FullDefinition[Sin]").unwrap();
      assert!(
        r.contains("Attributes[Sin]"),
        "should show built-in attrs: {}",
        r
      );
    }

    #[test]
    fn attributes() {
      assert_eq!(
        interpret("Attributes[FullDefinition]").unwrap(),
        "{HoldAll, Protected}"
      );
    }

    // Regression test for https://github.com/ad-si/Woxi/issues/96
    // Pattern variable names must not leak into other matched arguments.
    #[test]
    fn set_delayed_pattern_variable_no_leak() {
      assert_eq!(
        interpret("f[u_, a_] := {u, a}; f[a + 1, 42]").unwrap(),
        "{1 + a, 42}"
      );
    }

    #[test]
    fn set_delayed_pattern_variable_no_leak_reversed() {
      assert_eq!(
        interpret("g[a_, u_] := {a, u}; g[42, a + 1]").unwrap(),
        "{42, 1 + a}"
      );
    }

    #[test]
    fn set_delayed_variable_dependency() {
      let r = interpret("Clear[a, b]; a := 5; b := a + 1; FullDefinition[b]")
        .unwrap();
      assert!(r.contains("b = a + 1"), "should contain b: {}", r);
      assert!(r.contains("a = 5"), "should contain a: {}", r);
    }
  }

  mod matrix_power_tests {
    use super::*;

    #[test]
    fn power_zero_is_identity() {
      assert_eq!(
        interpret("MatrixPower[{{1, 2}, {3, 4}}, 0]").unwrap(),
        "{{1, 0}, {0, 1}}"
      );
    }

    #[test]
    fn power_one() {
      assert_eq!(
        interpret("MatrixPower[{{1, 2}, {3, 4}}, 1]").unwrap(),
        "{{1, 2}, {3, 4}}"
      );
    }

    #[test]
    fn power_two() {
      assert_eq!(
        interpret("MatrixPower[{{1, 2}, {3, 4}}, 2]").unwrap(),
        "{{7, 10}, {15, 22}}"
      );
    }

    #[test]
    fn power_three() {
      assert_eq!(
        interpret("MatrixPower[{{1, 2}, {3, 4}}, 3]").unwrap(),
        "{{37, 54}, {81, 118}}"
      );
    }

    #[test]
    fn power_negative_one() {
      assert_eq!(
        interpret("MatrixPower[{{1, 2}, {3, 4}}, -1]").unwrap(),
        "{{-2, 1}, {3/2, -1/2}}"
      );
    }

    #[test]
    fn power_negative_two() {
      assert_eq!(
        interpret("MatrixPower[{{1, 2}, {3, 4}}, -2]").unwrap(),
        "{{11/2, -5/2}, {-15/4, 7/4}}"
      );
    }

    #[test]
    fn diagonal_matrix() {
      assert_eq!(
        interpret("MatrixPower[{{2, 0}, {0, 3}}, 5]").unwrap(),
        "{{32, 0}, {0, 243}}"
      );
    }

    #[test]
    fn identity_matrix_power() {
      assert_eq!(
        interpret("MatrixPower[{{1, 0}, {0, 1}}, 100]").unwrap(),
        "{{1, 0}, {0, 1}}"
      );
    }

    #[test]
    fn symbolic_returns_unevaluated() {
      assert_eq!(interpret("MatrixPower[m, 3]").unwrap(), "MatrixPower[m, 3]");
    }

    #[test]
    fn attributes() {
      assert_eq!(
        interpret("Attributes[MatrixPower]").unwrap(),
        "{NonThreadable, Protected}"
      );
    }
  }

  // Regression tests for issue #76:
  // Pattern matching with structural patterns involving Power and Sqrt
  mod structural_pattern_matching_tests {
    use super::*;

    #[test]
    fn test_inverted_times_sqrt_matches_pattern() {
      // (Sqrt[x]*(a + b*x))^-1 should match 1/((a_.+b_.*x_)*Sqrt[c_.+d_.*x_])
      assert_eq!(
        interpret(
          "Int[1/((a_.+b_.*x_)*Sqrt[c_.+d_.*x_]),x_Symbol] := False; \
           Int[(Sqrt[x]*(a + b*x))^-1, x]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn test_direct_form_matches_pattern() {
      // 1/((a+b*x)*Sqrt[x]) should also match the same pattern
      assert_eq!(
        interpret(
          "f76a[1/((a_.+b_.*x_)*Sqrt[c_.+d_.*x_]),x_Symbol] := {a,b,c,d,x}; \
           f76a[1/((p+q*r)*Sqrt[r]), r]"
        )
        .unwrap(),
        "{p, q, 0, 1, r}"
      );
    }

    #[test]
    fn test_inverted_form_extracts_bindings() {
      // (Sqrt[r]*(p+q*r))^-1 should match and extract correct bindings
      assert_eq!(
        interpret(
          "f76b[1/((a_.+b_.*x_)*Sqrt[c_.+d_.*x_]), x_Symbol] := {a,b,c,d,x}; \
           f76b[(Sqrt[r]*(p+q*r))^-1, r]"
        )
        .unwrap(),
        "{p, q, 0, 1, r}"
      );
    }

    #[test]
    fn test_power_times_distribution() {
      // Power[Times[a, b], -1] should distribute to Times[Power[a,-1], Power[b,-1]].
      // wolframscript keeps the FullForm wrapper at the top level; the
      // bare head form is reachable via `ToString[…]`.
      assert_eq!(
        interpret("FullForm[(a*Sqrt[x])^-1]").unwrap(),
        "FullForm[1/(a*Sqrt[x])]"
      );
      assert_eq!(
        interpret("ToString[FullForm[(a*Sqrt[x])^-1]]").unwrap(),
        "Times[Power[a, -1], Power[x, Rational[-1, 2]]]"
      );
    }

    #[test]
    fn test_divide_canonical_form() {
      // 1/(a*b) should produce canonical Times[Power[...]] form matching (a*b)^-1
      assert_eq!(interpret("(a*b)^-1 === 1/(a*b)").unwrap(), "True");
    }
  }

  mod apart_tests {
    use super::*;

    #[test]
    fn test_apart_basic() {
      assert_eq!(
        interpret("Apart[1/((1 + x) (5 + x))]").unwrap(),
        "1/(4*(1 + x)) - 1/(4*(5 + x))"
      );
    }

    #[test]
    fn test_apart_canonical_times_power_form() {
      // Apart should handle Times[Power[...,-1], ...] form (issue #91)
      assert_eq!(
        interpret("Apart[Times[Power[Plus[1, x], -1], Power[Plus[5, x], -1]]]")
          .unwrap(),
        "1/(4*(1 + x)) - 1/(4*(5 + x))"
      );
    }

    #[test]
    fn test_apart_fullform_roundtrip() {
      // Using the result of FullForm should give the same Apart result
      assert_eq!(
        interpret("Apart[1/((1 + x) (5 + x))]").unwrap(),
        interpret("Apart[Times[Power[Plus[1, x], -1], Power[Plus[5, x], -1]]]")
          .unwrap()
      );
    }

    #[test]
    fn test_apart_simple_fraction() {
      assert_eq!(
        interpret("Apart[(5 + 2*x)/(1 + x)]").unwrap(),
        "2 + 3/(1 + x)"
      );
    }

    #[test]
    fn test_apart_no_denominator() {
      // Non-fractions should be returned as-is
      assert_eq!(interpret("Apart[x + 1]").unwrap(), "1 + x");
    }
  }

  mod trace_tests {
    use super::*;

    #[test]
    fn trace_of_simple_arithmetic() {
      // Trace wraps each step in HoldForm and the wrapper is preserved on
      // print, matching `wolframscript -code 'Trace[1 + 2]'` →
      // `{HoldForm[1 + 2], HoldForm[3]}`.
      assert_eq!(
        interpret("Trace[1 + 2]").unwrap(),
        "{HoldForm[1 + 2], HoldForm[3]}"
      );
    }

    #[test]
    fn trace_of_times() {
      assert_eq!(
        interpret("Trace[3 * 4]").unwrap(),
        "{HoldForm[3*4], HoldForm[12]}"
      );
    }

    #[test]
    fn trace_of_idempotent_expr_returns_empty_list() {
      // When evaluation doesn't change the expression, Trace returns {},
      // matching wolframscript.
      assert_eq!(interpret("Trace[x]").unwrap(), "{}");
    }
  }

  mod stack_tests {
    use super::*;
    use woxi::clear_state;

    #[test]
    fn test_stack_at_top_level() {
      clear_state();
      // At top level, Stack[] returns {} (the Stack entry itself is stripped
      // to match wolframscript).
      assert_eq!(interpret("Stack[]").unwrap(), "{}");
    }

    #[test]
    fn test_stack_in_nested_calls() {
      clear_state();
      // The trailing 'Stack' entry is stripped; caller chain remains.
      assert_eq!(
        interpret("f[] := Stack[]; g[] := f[]; g[]").unwrap(),
        "{g, f}"
      );
    }

    #[test]
    fn test_stack_trace_on_message() {
      clear_state();
      // Part out-of-range emits a message with a stack trace
      let result =
        interpret("f[x_] := {1, 2}[[x]]; g[x_] := f[x]; g[10]").unwrap();
      // The expression returns unevaluated
      assert_eq!(result, "{1, 2}[[10]]");
    }

    #[test]
    fn test_stack_trace_on_evaluation_error() {
      clear_state();
      // Division by zero returns ComplexInfinity (matching Wolfram)
      let result = interpret("f[x_] := 1/x; g[x_] := f[x]; g[0]").unwrap();
      assert_eq!(result, "ComplexInfinity");
    }

    #[test]
    fn test_stack_empty_after_successful_eval() {
      clear_state();
      // After successful evaluation, Stack[] is empty (the outer Stack
      // entry is stripped).
      assert_eq!(interpret("f[x_] := x + 1; f[5]").unwrap(), "6");
      assert_eq!(interpret("Stack[]").unwrap(), "{}");
    }
  }

  mod unicode_constants_tests {
    use super::*;

    #[test]
    fn test_unicode_pi_is_pi() {
      assert_eq!(interpret("π").unwrap(), "Pi");
    }

    #[test]
    fn test_unicode_pi_in_expression() {
      assert_eq!(interpret("Sin[π]").unwrap(), "0");
    }

    #[test]
    fn test_unicode_pi_numeric() {
      assert_eq!(interpret("N[π]").unwrap(), "3.141592653589793");
    }

    #[test]
    fn test_unicode_pi_implicit_multiply() {
      assert_eq!(interpret("2 π").unwrap(), "2*Pi");
    }

    #[test]
    fn test_unicode_exponential_e() {
      assert_eq!(interpret("ℯ").unwrap(), "E");
    }

    #[test]
    fn test_unicode_exponential_e_numeric() {
      assert_eq!(interpret("N[ℯ]").unwrap(), "2.718281828459045");
    }

    #[test]
    fn test_unicode_degree() {
      assert_eq!(interpret("°").unwrap(), "Degree");
    }

    #[test]
    fn test_unicode_degree_numeric() {
      assert_eq!(interpret("N[180 °]").unwrap(), "3.141592653589793");
    }

    #[test]
    fn test_unicode_infinity() {
      assert_eq!(interpret("∞").unwrap(), "Infinity");
    }

    #[test]
    fn test_unicode_infinity_in_expression() {
      assert_eq!(interpret("1/∞").unwrap(), "0");
    }

    #[test]
    fn test_unicode_imaginary_i() {
      assert_eq!(interpret("ⅈ").unwrap(), "I");
    }

    #[test]
    fn test_unicode_imaginary_i_squared() {
      assert_eq!(interpret("ⅈ^2").unwrap(), "-1");
    }
  }

  mod style_grid_tests {
    use super::*;

    #[test]
    fn test_style_grid_renders() {
      // CLI `interpret(...)` keeps Grid[...] symbolic to match wolframscript;
      // the SVG render only happens in visual mode (notebook / playground)
      // or via ExportString. Style[..., directives] unwraps to its first
      // arg in CLI output.
      assert_eq!(
        interpret("Style[Grid[{{a, b}, {c, d}}], Bold, Red]").unwrap(),
        "Grid[{{a, b}, {c, d}}]"
      );
    }

    #[test]
    fn test_style_grid_bold_propagates() {
      // All cells should have font-weight="bold"
      let svg =
        interpret("ExportString[Style[Grid[{{a, b}, {c, d}}], Bold], \"SVG\"]")
          .unwrap();
      assert!(
        svg.contains("font-weight=\"bold\""),
        "Expected bold text in SVG: {}",
        svg
      );
      // Count bold attributes — should be 4 (one per cell)
      let bold_count = svg.matches("font-weight=\"bold\"").count();
      assert_eq!(bold_count, 4, "All 4 cells should be bold");
    }

    #[test]
    fn test_style_grid_color_propagates() {
      // All cells should have red fill when Style[..., Red]
      let svg =
        interpret("ExportString[Style[Grid[{{a, b}, {c, d}}], Red], \"SVG\"]")
          .unwrap();
      let red_count = svg.matches("fill=\"rgb(255,0,0)\"").count();
      assert_eq!(red_count, 4, "All 4 cells should be red");
    }

    #[test]
    fn test_style_grid_inner_style_overrides_color() {
      // Inner Style[e, Green] overrides outer Red for that cell
      let svg = interpret(
        "ExportString[Style[Grid[{{a, Style[b, Green]}, {c, d}}], Red], \"SVG\"]",
      )
      .unwrap();
      let red_count = svg.matches("fill=\"rgb(255,0,0)\"").count();
      assert_eq!(red_count, 3, "3 cells should be red");
      assert!(
        svg.contains("fill=\"rgb(0,255,0)\""),
        "Cell b should be green"
      );
    }

    #[test]
    fn test_style_grid_inner_style_inherits_bold() {
      // Inner Style[e, Italic, Green] should still inherit Bold from outer
      let svg = interpret(
        "ExportString[Style[Grid[{{a, Style[b, Italic, Green]}}], Bold, Red], \"SVG\"]",
      )
      .unwrap();
      // Cell a should be bold+red
      assert!(svg.contains("font-weight=\"bold\""));
      // Cell b should be bold (inherited) + italic (inner) + green (inner)
      assert!(
        svg.contains("font-style=\"italic\""),
        "Cell b should be italic"
      );
      assert!(
        svg.contains("fill=\"rgb(0,255,0)\""),
        "Cell b should be green"
      );
      // Both cells should have bold
      let bold_count = svg.matches("font-weight=\"bold\"").count();
      assert_eq!(bold_count, 2, "Both cells should inherit bold");
    }

    #[test]
    fn test_style_grid_frame_dividers_use_style_color() {
      // Frame and divider lines should use the Style color
      let svg = interpret(
        "ExportString[Style[Grid[{{a, b}, {c, d}}, Frame -> True, Dividers -> All], Red], \"SVG\"]",
      )
      .unwrap();
      // Lines should have red stroke
      assert!(
        svg.contains("stroke=\"rgb(255,0,0)\""),
        "Frame lines should be red: {}",
        svg
      );
    }

    #[test]
    fn test_style_grid_italic() {
      // Style[Grid[...], Italic] should make all cells italic
      let svg =
        interpret("ExportString[Style[Grid[{{x, y}}], Italic], \"SVG\"]")
          .unwrap();
      let italic_count = svg.matches("font-style=\"italic\"").count();
      assert_eq!(italic_count, 2, "Both cells should be italic");
    }
  }

  mod graphics_text_style_tests {
    use super::*;

    #[test]
    fn test_text_style_font_size_rule() {
      // Style[..., FontSize -> 24] should set font-size="24" on the text
      let svg = interpret(
        "ExportString[Graphics[Text[Style[42, FontSize -> 24]]], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("font-size=\"24\""),
        "Expected font-size=\"24\" in SVG: {}",
        svg
      );
    }

    #[test]
    fn test_text_style_font_family_rule() {
      // Style[..., FontFamily -> "Consolas"] should set font-family="Consolas"
      let svg = interpret(
        "ExportString[Graphics[Text[Style[7, FontFamily -> \"Consolas\"]]], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("font-family=\"Consolas\""),
        "Expected font-family=\"Consolas\" in SVG: {}",
        svg
      );
    }

    #[test]
    fn test_text_style_font_weight_medium() {
      // Style[..., FontWeight -> "Medium"] should set font-weight="medium"
      let svg = interpret(
        "ExportString[Graphics[Text[Style[7, FontWeight -> \"Medium\"]]], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("font-weight=\"medium\""),
        "Expected font-weight=\"medium\" in SVG: {}",
        svg
      );
    }

    #[test]
    fn test_text_style_font_weight_bold_rule() {
      // Style[..., FontWeight -> "Bold"] should map to font-weight="bold"
      let svg = interpret(
        "ExportString[Graphics[Text[Style[7, FontWeight -> \"Bold\"]]], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("font-weight=\"bold\""),
        "Expected font-weight=\"bold\" in SVG: {}",
        svg
      );
    }

    #[test]
    fn test_text_style_combined_font_options() {
      // FontSize/FontFamily/FontWeight should all apply together
      let svg = interpret(
        "ExportString[Graphics[Text[Style[1, FontSize -> 30, FontFamily -> \"Consolas\", FontWeight -> \"Medium\"]]], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("font-size=\"30\""),
        "missing font-size: {}",
        svg
      );
      assert!(
        svg.contains("font-family=\"Consolas\""),
        "missing font-family: {}",
        svg
      );
      assert!(
        svg.contains("font-weight=\"medium\""),
        "missing font-weight: {}",
        svg
      );
    }
  }

  mod graphics_grid_frame_tests {
    use super::*;

    #[test]
    fn test_graphics_grid_frame_all_draws_lines() {
      // GraphicsGrid[..., Frame -> All] should draw inner column and row
      // dividers as well as an outer border.
      let svg = interpret(
        "ExportString[GraphicsGrid[Table[Graphics@Text[i j],{i,3},{j,3}],Frame->All],\"SVG\"]",
      )
      .unwrap();
      // 4 outer borders + 2 inner row + 2 inner column = 8 line elements
      let line_count = svg.matches("<line ").count();
      assert_eq!(
        line_count, 8,
        "Frame -> All on a 3x3 grid should produce 8 lines (4 outer + 2 inner-row + 2 inner-col): {}",
        svg
      );
    }

    #[test]
    fn test_graphics_grid_frame_none_draws_nothing() {
      let svg = interpret(
        "ExportString[GraphicsGrid[Table[Graphics@Text[i],{i,3},{j,3}]],\"SVG\"]",
      )
      .unwrap();
      assert!(
        !svg.contains("<line "),
        "No Frame option should produce no frame lines: {}",
        svg
      );
    }

    #[test]
    fn test_graphics_grid_frame_all_zero_default_spacing() {
      // With Frame -> All and no explicit Spacings, cells should touch
      // (zero gap) so frame lines align with cell boundaries.
      let svg = interpret(
        "ExportString[GraphicsGrid[Table[Graphics@Text[i j],{i,3},{j,3}],Frame->All],\"SVG\"]",
      )
      .unwrap();
      // Total grid width should be exactly 3 * 360 = 1080 (no gaps)
      assert!(
        svg.contains("width=\"1080\""),
        "Frame -> All should yield zero default spacing: {}",
        svg
      );
    }

    #[test]
    fn test_graphics_grid_table_cells_use_per_cell_imagesize() {
      // GraphicsGrid is held by the dispatcher, so a Table[...] arg
      // is unevaluated when graphics_grid_ast runs. The implementation
      // must still inject a per-cell ImageSize (1080/n) so each cell
      // renders at a size proportional to the column count. Otherwise
      // a 10-column grid balloons to ~3600 wide and inline pixel font
      // sizes (e.g. FontSize -> 24) become illegible after viewport
      // scaling.
      let svg = interpret(
        "ExportString[GraphicsGrid[Table[Graphics@Text@Style[i j,FontSize->24],{i,10},{j,10}],Frame->All],\"SVG\"]",
      )
      .unwrap();
      // Outer SVG should be 1080-wide (10 * 108, no gaps because Frame->All).
      let header_end = svg.find('\n').unwrap_or(svg.len()).min(200);
      assert!(
        svg.contains("width=\"1080\""),
        "10-col grid should use per-cell width 108, not natural 360: {}",
        &svg[..header_end]
      );
      // Each cell svg should be at the per-cell width (108), not 360.
      assert!(
        svg.contains("width=\"108\""),
        "Cell SVGs should be 108px wide for 10-col grid: {}",
        &svg[..svg.find("</svg>").unwrap_or(svg.len()).min(400)]
      );
    }
  }

  mod graphics_frame_tests {
    use super::*;

    #[test]
    fn test_graphics_frame_true_renders_rect() {
      let svg =
        interpret("ExportString[Graphics[Disk[], Frame -> True], \"SVG\"]")
          .unwrap();
      // Should have a frame rect element
      assert!(
        svg.contains("<rect") && svg.contains("fill=\"none\""),
        "Frame should render a border rect: {}",
        svg
      );
    }

    #[test]
    fn test_graphics_frame_true_has_tick_labels() {
      let svg =
        interpret("ExportString[Graphics[Disk[], Frame -> True], \"SVG\"]")
          .unwrap();
      // Should have tick label text elements
      assert!(
        svg.contains("<text"),
        "Frame should have tick labels: {}",
        svg
      );
    }

    #[test]
    fn test_graphics_frame_true_has_margins() {
      let svg =
        interpret("ExportString[Graphics[Disk[], Frame -> True], \"SVG\"]")
          .unwrap();
      // Should have a translate for margins
      assert!(
        svg.contains("translate(50,10)"),
        "Frame should offset for margin: {}",
        svg
      );
    }

    #[test]
    fn test_graphics_frame_with_plot_range() {
      let svg = interpret(
        "ExportString[Graphics[Disk[], PlotRange -> 6, Frame -> True], \"SVG\"]",
      )
      .unwrap();
      // Ticks should show values up to 6
      assert!(
        svg.contains(">6<") || svg.contains("\">6</"),
        "Frame ticks should reach PlotRange bounds: {}",
        svg
      );
      assert!(
        svg.contains(">-6<") || svg.contains("\">-6</"),
        "Frame ticks should reach negative PlotRange bounds: {}",
        svg
      );
    }

    #[test]
    fn test_graphics_no_frame_by_default() {
      let svg = interpret("ExportString[Graphics[Disk[]], \"SVG\"]").unwrap();
      // Should NOT have frame border or tick labels
      assert!(
        !svg.contains("translate(50"),
        "No frame margin by default: {}",
        svg
      );
    }

    #[test]
    fn test_plot_range_single_number() {
      // PlotRange -> n should be equivalent to PlotRange -> {{-n, n}, {-n, n}}
      let svg = interpret(
        "ExportString[Graphics[Disk[], PlotRange -> 3, Frame -> True], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("\">-3</") || svg.contains(">-3<"),
        "PlotRange -> 3 should give range -3 to 3: {}",
        svg
      );
    }
  }

  mod string_split_delimiter_tests {
    use super::*;

    #[test]
    fn test_split_preserves_interior_empty_strings() {
      assert_eq!(
        interpret(r#"StringSplit["one:two::three", ":"]"#).unwrap(),
        "{one, two, , three}"
      );
    }

    #[test]
    fn test_split_trims_leading_trailing_empty_strings() {
      assert_eq!(
        interpret(r#"StringSplit["::a::b::", ":"]"#).unwrap(),
        "{a, , b}"
      );
    }

    #[test]
    fn test_split_no_empty_strings() {
      assert_eq!(
        interpret(r#"StringSplit["a:b:c", ":"]"#).unwrap(),
        "{a, b, c}"
      );
    }

    #[test]
    fn test_split_all_delimiters() {
      assert_eq!(interpret(r#"StringSplit["::", ":"]"#).unwrap(), "{}");
    }

    #[test]
    fn test_split_multiple_consecutive() {
      assert_eq!(
        interpret(r#"StringSplit["a::b:::c", ":"]"#).unwrap(),
        "{a, , b, , , c}"
      );
    }

    #[test]
    fn test_split_max_parts_2() {
      assert_eq!(
        interpret(r#"StringSplit["a:b:c:d", ":", 2]"#).unwrap(),
        "{a, b:c:d}"
      );
    }

    #[test]
    fn test_split_max_parts_3() {
      assert_eq!(
        interpret(r#"StringSplit["a:b:c:d", ":", 3]"#).unwrap(),
        "{a, b, c:d}"
      );
    }

    #[test]
    fn test_split_max_parts_1() {
      assert_eq!(
        interpret(r#"StringSplit["a:b:c", ":", 1]"#).unwrap(),
        "{a:b:c}"
      );
    }
  }

  mod associate_to_tests {
    use super::*;

    #[test]
    fn test_associate_to_adds_key() {
      assert_eq!(
        interpret(r#"x = <|"a" -> 1|>; AssociateTo[x, "b" -> 2]; x"#).unwrap(),
        "<|a -> 1, b -> 2|>"
      );
    }

    #[test]
    fn test_associate_to_updates_existing_key() {
      assert_eq!(
        interpret(r#"x = <|"a" -> 1|>; AssociateTo[x, "a" -> 5]; x"#).unwrap(),
        "<|a -> 5|>"
      );
    }

    #[test]
    fn test_associate_to_attributes() {
      assert_eq!(
        interpret("Attributes[AssociateTo]").unwrap(),
        "{HoldFirst, Protected}"
      );
    }
  }

  mod key_drop_from_tests {
    use super::*;

    #[test]
    fn test_key_drop_from_removes_key() {
      assert_eq!(
        interpret(r#"x = <|"a" -> 1, "b" -> 2|>; KeyDropFrom[x, "a"]; x"#)
          .unwrap(),
        "<|b -> 2|>"
      );
    }

    #[test]
    fn test_key_drop_from_nonexistent_key() {
      assert_eq!(
        interpret(r#"x = <|"a" -> 1|>; KeyDropFrom[x, "z"]; x"#).unwrap(),
        "<|a -> 1|>"
      );
    }

    #[test]
    fn test_key_drop_from_attributes() {
      assert_eq!(
        interpret("Attributes[KeyDropFrom]").unwrap(),
        "{HoldFirst, Protected}"
      );
    }
  }

  mod member_q_level_spec_tests {
    use super::*;

    #[test]
    fn test_member_q_level_2() {
      assert_eq!(
        interpret("MemberQ[{{1, 2}, {3, 4}}, 3, {2}]").unwrap(),
        "True"
      );
    }

    #[test]
    fn test_member_q_level_1_not_found() {
      assert_eq!(
        interpret("MemberQ[{{1, 2}, {3, 4}}, 3, {1}]").unwrap(),
        "False"
      );
    }

    #[test]
    fn test_member_q_infinity() {
      assert_eq!(
        interpret("MemberQ[{1, {2, {3}}}, 3, Infinity]").unwrap(),
        "True"
      );
    }

    #[test]
    fn test_member_q_basic() {
      assert_eq!(interpret("MemberQ[{1, 2, 3}, 2]").unwrap(), "True");
      assert_eq!(interpret("MemberQ[{1, 2, 3}, 5]").unwrap(), "False");
    }

    #[test]
    fn test_member_q_pattern_at_level() {
      assert_eq!(
        interpret("MemberQ[{{1, 2}, {3, 4}}, _Integer, {2}]").unwrap(),
        "True"
      );
    }
  }

  mod operator_form_tests {
    use super::*;

    #[test]
    fn test_count_operator_form() {
      assert_eq!(
        interpret(r#"Count[_Integer][{1, "a", 2, "b"}]"#).unwrap(),
        "2"
      );
    }

    #[test]
    fn test_position_operator_form() {
      assert_eq!(
        interpret(r#"Position[_Integer][{1, "a", 2}]"#).unwrap(),
        "{{1}, {3}}"
      );
    }

    #[test]
    fn test_delete_cases_operator_form() {
      assert_eq!(
        interpret(r#"DeleteCases[_Integer][{1, "a", 2}]"#).unwrap(),
        "{a}"
      );
    }

    #[test]
    fn test_match_q_operator_form() {
      assert_eq!(interpret("MatchQ[{__Integer}][{1, 2, 3}]").unwrap(), "True");
    }

    #[test]
    fn test_string_cases_operator_form() {
      assert_eq!(
        interpret(r#"StringCases[DigitCharacter..]["abc123"]"#).unwrap(),
        "{123}"
      );
    }
  }

  mod collect_multi_variable_tests {
    use super::*;

    #[test]
    fn test_collect_two_variables() {
      assert_eq!(
        interpret("Collect[a*x + b*x + c*y + d*y, {x, y}]").unwrap(),
        "(a + b)*x + (c + d)*y"
      );
    }

    #[test]
    fn test_collect_three_variables() {
      assert_eq!(
        interpret("Collect[a*x + b*y + c*z + d*x + e*y, {x, y, z}]").unwrap(),
        "(a + d)*x + (b + e)*y + c*z"
      );
    }

    #[test]
    fn test_collect_single_var_in_list() {
      assert_eq!(
        interpret("Collect[a*x + b*x + c, {x}]").unwrap(),
        "c + (a + b)*x"
      );
    }
  }

  mod re_im_symbolic_tests {
    use super::*;

    #[test]
    fn test_re_i_times_x() {
      assert_eq!(interpret("Re[I*x]").unwrap(), "-Im[x]");
    }

    #[test]
    fn test_im_i_times_x() {
      assert_eq!(interpret("Im[I*x]").unwrap(), "Re[x]");
    }

    #[test]
    fn test_re_i_times_product() {
      assert_eq!(interpret("Re[I*y*z]").unwrap(), "-Im[y*z]");
    }

    #[test]
    fn test_im_i_times_product() {
      assert_eq!(interpret("Im[I*y*z]").unwrap(), "Re[y*z]");
    }

    #[test]
    fn test_re_numeric_complex() {
      assert_eq!(interpret("Re[3 + 4 I]").unwrap(), "3");
    }

    #[test]
    fn test_im_numeric_complex() {
      assert_eq!(interpret("Im[3 + 4 I]").unwrap(), "4");
    }
  }

  mod from_digits_realdigits_tests {
    use super::*;

    #[test]
    fn test_from_digits_realdigits_format() {
      assert_eq!(
        interpret("FromDigits[{{1, 2, 3, 4, 5, 6}, 3}]").unwrap(),
        "15432/125"
      );
    }

    #[test]
    fn test_from_digits_integer_exponent() {
      assert_eq!(interpret("FromDigits[{{1, 2, 3}, 3}]").unwrap(), "123");
    }

    #[test]
    fn test_from_digits_small_exponent() {
      assert_eq!(interpret("FromDigits[{{5}, 0}]").unwrap(), "1/2");
    }

    #[test]
    fn test_from_digits_roundtrip() {
      assert_eq!(
        interpret("FromDigits[RealDigits[123.456]]").unwrap(),
        "15432/125"
      );
    }

    #[test]
    fn test_from_digits_normal_list() {
      assert_eq!(interpret("FromDigits[{1, 2, 3}]").unwrap(), "123");
    }
  }

  mod association_dedup_tests_actual {
    use super::*;

    #[test]
    fn test_association_duplicate_keys_last_wins() {
      assert_eq!(
        interpret(r#"<|"a" -> 1, "b" -> 2, "a" -> 3|>"#).unwrap(),
        "<|a -> 3, b -> 2|>"
      );
    }

    #[test]
    fn test_association_constructor_dedup() {
      assert_eq!(
        interpret(r#"Association["a" -> 1, "b" -> 2, "a" -> 3]"#).unwrap(),
        "<|a -> 3, b -> 2|>"
      );
    }

    #[test]
    fn test_association_list_constructor_dedup() {
      assert_eq!(
        interpret(r#"Association[{"a" -> 1, "b" -> 2, "a" -> 3}]"#).unwrap(),
        "<|a -> 3, b -> 2|>"
      );
    }

    #[test]
    fn test_association_no_duplicates() {
      assert_eq!(
        interpret(r#"<|"a" -> 1, "b" -> 2|>"#).unwrap(),
        "<|a -> 1, b -> 2|>"
      );
    }

    #[test]
    fn test_association_triple_duplicate() {
      assert_eq!(
        interpret(r#"<|"x" -> 1, "x" -> 2, "x" -> 3|>"#).unwrap(),
        "<|x -> 3|>"
      );
    }
  }

  mod association_rule_delayed_tests {
    use super::*;

    #[test]
    fn test_literal_assoc_preserves_rule_delayed() {
      // `<|key :> value|>` keeps the held value and renders `:>`.
      assert_eq!(
        interpret("<|a -> 1, b :> Association[p->3, q->4]|>").unwrap(),
        "<|a -> 1, b :> Association[p -> 3, q -> 4]|>"
      );
    }

    #[test]
    fn test_constructor_assoc_preserves_rule_delayed() {
      // The `Association[...]` constructor preserves `:>` too — the held
      // RHS does not get evaluated to its `<|...|>` short form.
      assert_eq!(
        interpret("Association[a -> 1, b :> Association[p->3, q->4]]").unwrap(),
        "<|a -> 1, b :> Association[p -> 3, q -> 4]|>"
      );
    }

    #[test]
    fn test_map_at_level_one_keeps_rule_delayed() {
      // Mapping F over an association applies F to each value while
      // keeping the original Rule/RuleDelayed kind on each entry.
      assert_eq!(
        interpret("Map[F, Association[a -> 1, b :> 2]]").unwrap(),
        "<|a -> F[1], b :> F[2]|>"
      );
    }

    #[test]
    fn test_map_at_level_two_descends_into_held_value() {
      // At level {2} F descends into the held replacement of the `:>`
      // entry and is applied to that expression's children.
      assert_eq!(
        interpret("Map[F, Association[a -> 1, b :> Q[p->3, q->4]], {2}]")
          .unwrap(),
        "<|a -> 1, b :> Q[F[p -> 3], F[q -> 4]]|>"
      );
    }
  }

  mod pdf_export_tests {
    use super::*;

    #[test]
    fn test_export_pdf_creates_valid_file() {
      let tmp = std::env::temp_dir().join("woxi_test_export.pdf");
      let path = tmp.display().to_string();
      let result =
        interpret(&format!("Export[\"{path}\", (x^2 + 3)/7, \"PDF\"]"))
          .unwrap();
      assert_eq!(result, path);
      let bytes = std::fs::read(&tmp).unwrap();
      assert!(bytes.starts_with(b"%PDF"), "File should be a valid PDF");
      std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_pdf_by_extension() {
      let tmp = std::env::temp_dir().join("woxi_test_ext.pdf");
      let path = tmp.display().to_string();
      let result = interpret(&format!("Export[\"{path}\", x + 1]")).unwrap();
      assert_eq!(result, path);
      let bytes = std::fs::read(&tmp).unwrap();
      assert!(bytes.starts_with(b"%PDF"), "File should be a valid PDF");
      std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_pdf_integer() {
      let tmp = std::env::temp_dir().join("woxi_test_int.pdf");
      let path = tmp.display().to_string();
      let result =
        interpret(&format!("Export[\"{path}\", 42, \"PDF\"]")).unwrap();
      assert_eq!(result, path);
      let bytes = std::fs::read(&tmp).unwrap();
      assert!(bytes.starts_with(b"%PDF"));
      assert!(bytes.len() > 100, "PDF should have substantial content");
      std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_pdf_symbolic_expression() {
      let tmp = std::env::temp_dir().join("woxi_test_sym.pdf");
      let path = tmp.display().to_string();
      let result =
        interpret(&format!("Export[\"{path}\", Sin[x] + Cos[y], \"PDF\"]"))
          .unwrap();
      assert_eq!(result, path);
      let bytes = std::fs::read(&tmp).unwrap();
      assert!(bytes.starts_with(b"%PDF"));
      std::fs::remove_file(&tmp).ok();
    }
  }

  mod pause_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_pause_zero_returns_null() {
      assert_eq!(interpret("Pause[0]").unwrap(), "\0");
    }

    #[test]
    fn test_pause_integer_returns_null() {
      assert_eq!(interpret("Pause[0]; 42").unwrap(), "42");
    }

    #[test]
    fn test_pause_real_returns_null() {
      assert_eq!(interpret("Pause[0.05]").unwrap(), "\0");
    }

    #[test]
    fn test_pause_actually_waits() {
      let start = Instant::now();
      interpret("Pause[0.3]").unwrap();
      let elapsed = start.elapsed().as_secs_f64();
      assert!(
        elapsed >= 0.3,
        "Pause[0.3] should wait at least 0.3s, waited {elapsed}s"
      );
    }

    #[test]
    fn test_pause_uses_wall_clock_time() {
      let result = interpret("AbsoluteTiming[Pause[0.2]]").unwrap();
      assert!(result.starts_with('{') && result.ends_with('}'));
      assert!(result.contains(", Null}"));
      let inner = &result[1..result.len() - 1];
      let comma = inner.find(',').unwrap();
      let elapsed: f64 = inner[..comma].trim().parse().unwrap();
      assert!(
        elapsed >= 0.2,
        "AbsoluteTiming[Pause[0.2]] should report >= 0.2s, got {elapsed}"
      );
    }

    #[test]
    fn test_pause_negative_stays_unevaluated() {
      // wolframscript rejects negative durations with Pause::numnm and
      // leaves the call unevaluated.
      assert_eq!(interpret("Pause[-1]").unwrap(), "Pause[-1]");
    }

    #[test]
    fn test_pause_non_numeric_stays_unevaluated() {
      assert_eq!(interpret("Pause[a]").unwrap(), "Pause[a]");
    }

    #[test]
    fn test_pause_in_compound_expression() {
      assert_eq!(interpret("(Pause[0]; 1 + 2)").unwrap(), "3");
    }

    #[test]
    fn test_pause_symbolic_arg_returns_null() {
      assert_eq!(interpret("Pause[Sqrt[0]]").unwrap(), "\0");
    }
  }

  mod option_value_tests {
    use super::*;

    #[test]
    fn test_option_value_lookup_resolves_specified_option() {
      assert_eq!(
        interpret("f[a->3] /. f[OptionsPattern[{}]] -> {OptionValue[a]}")
          .unwrap(),
        "{3}"
      );
    }

    #[test]
    fn test_option_value_missing_falls_back_to_symbol() {
      // Inside an OptionsPattern context, an unbound name resolves to
      // the bare symbol (matches Wolfram).
      assert_eq!(
        interpret("f[a->3] /. f[OptionsPattern[{}]] -> {OptionValue[b]}")
          .unwrap(),
        "{b}"
      );
    }

    #[test]
    fn test_option_value_missing_string_arg_falls_back_to_string() {
      // wolframscript preserves the original String form when the option
      // name is unresolved — OptionValue["b"] returns the string "b" (not
      // the bare symbol). OutputForm strips the surrounding quotes for
      // display, so `interpret` returns `{b}` here while
      // `ToString[..., InputForm]` round-trips to `{"b"}`.
      assert_eq!(
        interpret("f[a->3] /. f[OptionsPattern[{}]] :> {OptionValue[\"b\"]}")
          .unwrap(),
        "{b}"
      );
      assert_eq!(
        interpret(
          "ToString[f[a->3] /. f[OptionsPattern[{}]] :> {OptionValue[\"b\"]}, InputForm]"
        )
        .unwrap(),
        "{\"b\"}"
      );
    }

    #[test]
    fn test_option_value_outside_context_stays_unevaluated() {
      assert_eq!(interpret("OptionValue[b]").unwrap(), "OptionValue[b]");
    }

    #[test]
    fn test_option_value_non_symbol_arg_in_context_returns_argument() {
      // OptionValue[a+b] inside OptionsPattern resolves to the bare
      // expression a+b (not a symbol lookup), matching Wolfram.
      assert_eq!(
        interpret("f[a->3] /. f[OptionsPattern[{}]] -> {OptionValue[a+b]}")
          .unwrap(),
        "{a + b}"
      );
    }

    #[test]
    fn test_option_value_non_symbol_arg_outside_context_unevaluated() {
      assert_eq!(interpret("OptionValue[a+b]").unwrap(), "OptionValue[a + b]");
    }
  }

  mod expand_canonical_order_tests {
    use super::*;

    #[test]
    fn test_expand_symbolic_power_sorts_by_base() {
      assert_eq!(interpret("Expand[a^n + b]").unwrap(), "a^n + b");
      assert_eq!(interpret("Expand[c + b + a^n]").unwrap(), "a^n + b + c");
    }

    #[test]
    fn test_expand_symbolic_power_after_numeric_power() {
      assert_eq!(interpret("Expand[a^n + a^2]").unwrap(), "a^2 + a^n");
      assert_eq!(interpret("Expand[a^n + a]").unwrap(), "a + a^n");
    }

    #[test]
    fn test_expand_bare_call_before_times_product_with_same_call() {
      // `C[1] + (-1)^n*C[1]` keeps the bare `C[1]` first when both terms
      // share the same FunctionCall factor — Wolfram's canonical Plus
      // ordering treats the bare call as "shorter" / lower.
      assert_eq!(
        interpret("Expand[C[1] - (-1)^n*C[1]]").unwrap(),
        "C[1] - (-1)^n*C[1]"
      );
      assert_eq!(
        interpret("Expand[(-1)^n + C[1] - (-1)^n*C[1]]").unwrap(),
        "(-1)^n + C[1] - (-1)^n*C[1]"
      );
    }

    #[test]
    fn test_rsolve_initial_condition_canonical_form() {
      // RSolve relies on the Expand ordering above for its readable output.
      assert_eq!(
        interpret("RSolve[{a[n + 2] == a[n], a[0] == 1}, a, n]").unwrap(),
        "{{a -> Function[{n}, (-1)^n + C[1] - (-1)^n*C[1]]}}"
      );
    }
  }

  mod import_ppm_tests {
    use std::io::Write;
    use woxi::interpret_with_stdout;

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
      let path = std::env::temp_dir().join(format!(
        "woxi_ppm_test_{}_{}",
        std::process::id(),
        name
      ));
      let mut f = std::fs::File::create(&path).unwrap();
      f.write_all(bytes).unwrap();
      path
    }

    #[test]
    fn test_import_ppm_invalid_emits_fmterr() {
      // File present but contents aren't a Netpbm stream — wolframscript
      // prints `Import::fmterr: Cannot import data as PPM format.` and
      // returns `$Failed`. Woxi must match.
      let path = write_tmp("invalid.ppm", b"image");
      let code = format!(r#"Import["{}","PPM"]"#, path.to_string_lossy());
      let r = interpret_with_stdout(&code).unwrap();
      assert_eq!(r.result, "$Failed");
      assert!(
        r.stdout
          .contains("Import::fmterr: Cannot import data as PPM format."),
        "stdout was: {:?}",
        r.stdout
      );
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_import_ppm_missing_file_emits_nffil() {
      let path = std::env::temp_dir().join(format!(
        "woxi_ppm_does_not_exist_{}.ppm",
        std::process::id()
      ));
      let _ = std::fs::remove_file(&path);
      let code = format!(r#"Import["{}","PPM"]"#, path.to_string_lossy());
      let r = interpret_with_stdout(&code).unwrap();
      assert_eq!(r.result, "$Failed");
      assert!(
        r.stdout.contains("Import::nffil:"),
        "stdout was: {:?}",
        r.stdout
      );
    }

    #[test]
    fn test_import_ppm_by_extension_detects_invalid() {
      // No explicit format argument; .ppm extension alone must trigger
      // the Netpbm path so the error message still appears.
      let path = write_tmp("invalid_ext.ppm", b"not a ppm");
      let code = format!(r#"Import["{}"]"#, path.to_string_lossy());
      let r = interpret_with_stdout(&code).unwrap();
      assert_eq!(r.result, "$Failed");
      assert!(
        r.stdout
          .contains("Import::fmterr: Cannot import data as PPM format."),
        "stdout was: {:?}",
        r.stdout
      );
      let _ = std::fs::remove_file(&path);
    }
  }

  mod cholesky_decomposition_tests {
    use super::*;

    #[test]
    fn test_basic_integer() {
      // A = U^*.U with U upper triangular; exact integer result.
      assert_eq!(
        interpret("CholeskyDecomposition[{{4,2},{2,2}}]").unwrap(),
        "{{2, 1}, {0, 1}}"
      );
    }

    #[test]
    fn test_radical_diagonal() {
      assert_eq!(
        interpret("CholeskyDecomposition[{{4,2},{2,3}}]").unwrap(),
        "{{2, 1}, {0, Sqrt[2]}}"
      );
    }

    #[test]
    fn test_three_by_three_integer() {
      assert_eq!(
        interpret(
          "CholeskyDecomposition[{{25,15,-5},{15,18,0},{-5,0,11}}]"
        )
        .unwrap(),
        "{{5, 3, -1}, {0, 3, 1}, {0, 0, 3}}"
      );
    }

    #[test]
    fn test_four_by_four_radicals() {
      assert_eq!(
        interpret(
          "CholeskyDecomposition[{{6,3,4,8},{3,6,5,1},{4,5,10,7},{8,1,7,25}}]"
        )
        .unwrap(),
        "{{Sqrt[6], Sqrt[3/2], 2*Sqrt[2/3], 4*Sqrt[2/3]}, \
         {0, 3/Sqrt[2], Sqrt[2], -Sqrt[2]}, \
         {0, 0, 4/Sqrt[3], 11/(4*Sqrt[3])}, \
         {0, 0, 0, Sqrt[157]/4}}"
      );
    }

    #[test]
    fn test_numeric_real_zero_below_diagonal() {
      // Machine-precision input: the strictly-lower zero is `0.`.
      assert_eq!(
        interpret("CholeskyDecomposition[{{2.0,1.0},{1.0,2.0}}]").unwrap(),
        "{{1.4142135623730951, 0.7071067811865475}, {0., 1.224744871391589}}"
      );
    }

    #[test]
    fn test_one_by_one() {
      assert_eq!(
        interpret("CholeskyDecomposition[{{2}}]").unwrap(),
        "{{Sqrt[2]}}"
      );
    }

    #[test]
    fn test_not_positive_definite_returns_unevaluated() {
      // wolframscript emits npdef and returns the expression unevaluated.
      assert_eq!(
        interpret("CholeskyDecomposition[{{1,2},{2,1}}]").unwrap(),
        "CholeskyDecomposition[{{1, 2}, {2, 1}}]"
      );
    }

    #[test]
    fn test_non_square_returns_unevaluated() {
      assert_eq!(
        interpret("CholeskyDecomposition[{{1,2,3},{4,5,6}}]").unwrap(),
        "CholeskyDecomposition[{{1, 2, 3}, {4, 5, 6}}]"
      );
    }
  }
}
