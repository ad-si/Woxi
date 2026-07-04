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

    // When the base-rebased logs land on exact powers of the base they must
    // collapse to integers/rationals, not stay as Log[k]/Log[b] ratios.
    #[test]
    fn test_explicit_base_2_two_equal_categories() {
      assert_eq!(interpret("Entropy[2, {1, 1, 2, 2}]").unwrap(), "1");
    }

    #[test]
    fn test_explicit_base_2_four_distinct() {
      assert_eq!(interpret("Entropy[2, {a, b, c, d}]").unwrap(), "2");
    }

    #[test]
    fn test_explicit_base_3_three_equal_categories() {
      assert_eq!(
        interpret("Entropy[3, {a, a, a, b, b, b, c, c, c}]").unwrap(),
        "1"
      );
    }

    // Non-power category counts keep the ratio form (matches wolframscript).
    #[test]
    fn test_explicit_base_2_non_power() {
      assert_eq!(
        interpret("Entropy[2, {a, a, b, b, c, c}]").unwrap(),
        "-1 + Log[6]/Log[2]"
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
    #[test]
    fn test_capitalize_all_words() {
      assert_eq!(
        interpret(r#"Capitalize["the cat AND dog", "AllWords"]"#).unwrap(),
        "The Cat AND Dog"
      );
    }
    #[test]
    fn test_capitalize_first_word() {
      assert_eq!(
        interpret(r#"Capitalize["the quick brown fox", "FirstWord"]"#).unwrap(),
        "The quick brown fox"
      );
    }
    #[test]
    fn test_capitalize_long_words() {
      // Words with more than three letters are capitalized.
      assert_eq!(
        interpret(r#"Capitalize["the quick brown fox", "LongWords"]"#).unwrap(),
        "the Quick Brown fox"
      );
    }
    #[test]
    fn test_capitalize_preserves_punctuation() {
      assert_eq!(
        interpret(r#"Capitalize["a-b c.d", "AllWords"]"#).unwrap(),
        "A-b C.d"
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
    #[test]
    fn test_string_delete_digit_character() {
      assert_eq!(
        interpret(r#"StringDelete["a1b2c3", DigitCharacter]"#).unwrap(),
        "abc"
      );
    }
    #[test]
    fn test_string_delete_letter_character() {
      assert_eq!(
        interpret(r#"StringDelete["a1b2c3", LetterCharacter]"#).unwrap(),
        "123"
      );
    }
    #[test]
    fn test_string_delete_number_string_pattern() {
      assert_eq!(
        interpret(r#"StringDelete["one1two2", NumberString]"#).unwrap(),
        "onetwo"
      );
    }
    #[test]
    fn test_string_delete_mixed_pattern_list() {
      assert_eq!(
        interpret(r#"StringDelete["a1b2c3", {DigitCharacter, "a"}]"#).unwrap(),
        "bc"
      );
    }
    #[test]
    fn test_string_delete_ignore_case() {
      assert_eq!(
        interpret(r#"StringDelete["abcABC", "abc", IgnoreCase -> True]"#)
          .unwrap(),
        ""
      );
    }
    #[test]
    fn test_string_delete_pattern_threads_list() {
      assert_eq!(
        interpret(r#"StringDelete[{"a1", "b2"}, DigitCharacter]"#).unwrap(),
        "{a, b}"
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
        interpret(r#"StringReplace["one2three", WordBoundary -> "|"]"#)
          .unwrap(),
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

    #[test]
    fn test_rational_truncates_to_n_terms() {
      // The exact CF of 649/200 is {3, 4, 12, 4}; requesting 2 terms truncates.
      assert_eq!(
        interpret("ContinuedFraction[649/200, 2]").unwrap(),
        "{3, 4}"
      );
      assert_eq!(
        interpret("ContinuedFraction[649/200, 4]").unwrap(),
        "{3, 4, 12, 4}"
      );
    }

    #[test]
    fn test_incomp_warning_when_fewer_terms() {
      // Requesting more terms than the finite CF has emits the incomp warning,
      // matching wolframscript, while still returning the available terms.
      woxi::clear_state();
      assert_eq!(
        interpret("ContinuedFraction[649/200, 8]").unwrap(),
        "{3, 4, 12, 4}"
      );
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().any(|m| m.contains(
          "ContinuedFraction::incomp: Warning: ContinuedFraction terminated before 8 terms."
        )),
        "expected incomp warning, got {msgs:?}"
      );
    }

    #[test]
    fn test_incomp_warning_for_integer() {
      woxi::clear_state();
      assert_eq!(interpret("ContinuedFraction[5, 3]").unwrap(), "{5}");
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().any(|m| m.contains(
          "ContinuedFraction::incomp: Warning: ContinuedFraction terminated before 3 terms."
        )),
        "expected incomp warning, got {msgs:?}"
      );
    }

    #[test]
    fn test_real_incomp_warning() {
      // The machine float 3.245 justifies only 3 terms; requesting 4 warns.
      woxi::clear_state();
      assert_eq!(
        interpret("ContinuedFraction[3.245, 4]").unwrap(),
        "{3, 4, 12}"
      );
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().any(|m| m.contains(
          "ContinuedFraction::incomp: Warning: ContinuedFraction terminated before 4 terms."
        )),
        "expected incomp warning, got {msgs:?}"
      );
    }

    #[test]
    fn test_no_warning_when_enough_terms() {
      // Exactly n terms available: no warning.
      woxi::clear_state();
      assert_eq!(interpret("ContinuedFraction[3.245, 2]").unwrap(), "{3, 4}");
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs
          .iter()
          .all(|m| !m.contains("ContinuedFraction::incomp")),
        "unexpected incomp warning, got {msgs:?}"
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

    // The three-argument form returns the smallest solution >= d.
    #[test]
    fn test_offset() {
      assert_eq!(
        interpret("ChineseRemainder[{1, 2}, {3, 5}, 10]").unwrap(),
        "22"
      );
      assert_eq!(
        interpret("ChineseRemainder[{1, 2}, {3, 5}, -10]").unwrap(),
        "-8"
      );
      // d already at or below the base solution returns the base solution.
      assert_eq!(
        interpret("ChineseRemainder[{2, 3}, {3, 5}, 8]").unwrap(),
        "8"
      );
    }

    // Incompatible congruences have no solution: wolframscript leaves the
    // call unevaluated (rather than erroring).
    #[test]
    fn test_no_solution_unevaluated() {
      assert_eq!(
        interpret("ChineseRemainder[{2, 3}, {4, 6}]").unwrap(),
        "ChineseRemainder[{2, 3}, {4, 6}]"
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
      // The count-limited form returns *which* m solutions in an undocumented
      // internal order that differs from the full enumeration, so assert the
      // conformance-stable properties instead: exactly m solutions are
      // returned and every one satisfies the Frobenius equation.
      assert_eq!(
        interpret("Length[FrobeniusSolve[{2, 3, 4}, 10, 2]]").unwrap(),
        "2"
      );
      assert_eq!(
        interpret(
          "Union[Map[{2, 3, 4} . # &, FrobeniusSolve[{2, 3, 4}, 10, 2]]]"
        )
        .unwrap(),
        "{10}"
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

  mod get_tests {
    use woxi::interpret_with_stdout;

    // `Get` on a missing file returns `$Failed` and, like wolframscript,
    // prints the `Get::noopen` message to stdout (with a leading blank line)
    // so it is captured by `interpret_with_stdout` (snapshots/playground/Jupyter).
    #[test]
    fn test_get_missing_file_message_to_stdout() {
      let result = interpret_with_stdout(r#"Get["does_not_exist.m"]"#).unwrap();
      assert_eq!(result.result, "$Failed");
      assert_eq!(
        result.stdout,
        "\nGet::noopen: Cannot open does_not_exist.m.\n"
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
    fn test_scientific_form_to_string() {
      // ScientificForm renders as `mantissa × 10` with the exponent placed as a
      // superscript on the line above. Default precision is 6 significant figs.
      assert_eq!(
        interpret("ToString[ScientificForm[12345.678]]").unwrap(),
        "            4\n1.23457 \u{00d7} 10"
      );
      // Trailing zeros in the mantissa are dropped (123.45 -> 1.2345).
      assert_eq!(
        interpret("ToString[ScientificForm[123.45]]").unwrap(),
        "           2\n1.2345 \u{00d7} 10"
      );
      // Negative exponents.
      assert_eq!(
        interpret("ToString[ScientificForm[0.000012345]]").unwrap(),
        "           -5\n1.2345 \u{00d7} 10"
      );
      // Negative mantissa.
      assert_eq!(
        interpret("ToString[ScientificForm[-9876.5]]").unwrap(),
        "            3\n-9.8765 \u{00d7} 10"
      );
      // Explicit precision: ScientificForm[x, n] uses n significant figures.
      assert_eq!(
        interpret("ToString[ScientificForm[123.45, 3]]").unwrap(),
        "         2\n1.23 \u{00d7} 10"
      );
      // Exponent 0 collapses to just the mantissa (no `× 10`).
      assert_eq!(interpret("ToString[ScientificForm[5.0]]").unwrap(), "5.");
      // Rounding that bumps the mantissa to 10 renormalizes the exponent.
      assert_eq!(
        interpret("ToString[ScientificForm[9999.995]]").unwrap(),
        "       4\n1. \u{00d7} 10"
      );
      // Integers are shown unchanged.
      assert_eq!(interpret("ToString[ScientificForm[123]]").unwrap(), "123");
    }

    #[test]
    fn test_engineering_form_to_string() {
      // EngineeringForm is like ScientificForm but the exponent is forced to a
      // multiple of 3, so the mantissa lies in [1, 1000).
      assert_eq!(
        interpret("ToString[EngineeringForm[12345.678]]").unwrap(),
        "            3\n12.3457 \u{00d7} 10"
      );
      // 1.2345e8 -> 123.45 × 10^6 (exponent stepped to a multiple of 3).
      assert_eq!(
        interpret("ToString[EngineeringForm[1.2345*^8]]").unwrap(),
        "           6\n123.45 \u{00d7} 10"
      );
      // Negative exponent, also a multiple of 3.
      assert_eq!(
        interpret("ToString[EngineeringForm[0.000012345]]").unwrap(),
        "           -6\n12.345 \u{00d7} 10"
      );
      // Negative mantissa.
      assert_eq!(
        interpret("ToString[EngineeringForm[-9876.5]]").unwrap(),
        "            3\n-9.8765 \u{00d7} 10"
      );
      // Values with exponent 0 collapse to just the mantissa.
      assert_eq!(
        interpret("ToString[EngineeringForm[123.45]]").unwrap(),
        "123.45"
      );
      assert_eq!(interpret("ToString[EngineeringForm[5.0]]").unwrap(), "5.");
      // Explicit precision.
      assert_eq!(
        interpret("ToString[EngineeringForm[123.45, 3]]").unwrap(),
        "123."
      );
      // Mantissa fills all three integer digits when needed.
      assert_eq!(
        interpret("ToString[EngineeringForm[999999.0]]").unwrap(),
        "            3\n999.999 \u{00d7} 10"
      );
      // Integers are shown unchanged.
      assert_eq!(interpret("ToString[EngineeringForm[123]]").unwrap(), "123");
    }

    #[test]
    fn test_accounting_form_never_scientific() {
      // AccountingForm never uses scientific notation: a large Real that the
      // default OutputForm would render as `1.23457*^6` is shown in full
      // decimal instead (regression — Woxi used to emit `1.23457*^6`).
      assert_eq!(
        interpret("ToString[AccountingForm[1234567.]]").unwrap(),
        "1234567."
      );
      // Fractional digits beyond the displayed precision round into the integer.
      assert_eq!(
        interpret("ToString[AccountingForm[1234567.89]]").unwrap(),
        "1234568."
      );
      // Very large magnitudes keep every integer digit.
      assert_eq!(
        interpret("ToString[AccountingForm[12345678901234.]]").unwrap(),
        "12345678901234."
      );
      // Powers of ten do not collapse to `1.*^6`.
      assert_eq!(
        interpret("ToString[AccountingForm[1000000.]]").unwrap(),
        "1000000."
      );
      // Negatives are parenthesised, still without scientific notation.
      assert_eq!(
        interpret("ToString[AccountingForm[-1234567.]]").unwrap(),
        "(1234567.)"
      );
    }

    #[test]
    fn test_accounting_form_reqsigz_warning() {
      // Requesting fewer significant figures than the number has integer
      // digits pads the trailing digits with zeros and warns, matching
      // wolframscript (`AccountingForm[1234.5678, 3]` -> `1230.`).
      woxi::clear_state();
      assert_eq!(
        interpret("ToString[AccountingForm[1234.5678, 3]]").unwrap(),
        "1230."
      );
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().any(|m| m.contains(
          "AccountingForm::reqsigz: Requested number precision is lower \
           than number of digits shown; padding with zeros."
        )),
        "expected reqsigz warning, got {msgs:?}"
      );
    }

    #[test]
    fn test_accounting_form_no_reqsigz_when_precision_sufficient() {
      // When the requested precision is at least the integer-digit count, no
      // warning fires (4 figures for the 4-digit 1234.5678 -> `1235.`).
      woxi::clear_state();
      assert_eq!(
        interpret("ToString[AccountingForm[1234.5678, 4]]").unwrap(),
        "1235."
      );
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        !msgs.iter().any(|m| m.contains("reqsigz")),
        "did not expect reqsigz warning, got {msgs:?}"
      );
    }

    #[test]
    fn test_column_form_renders_stacked() {
      // ColumnForm is the legacy display directive: it stacks elements one per
      // line. Unlike Column, it renders at top level and under every form
      // (including InputForm).
      assert_eq!(
        interpret("ToString[ColumnForm[{1, 2, 3}]]").unwrap(),
        "1\n2\n3"
      );
      assert_eq!(
        interpret("ToString[ColumnForm[{a, b, c}]]").unwrap(),
        "a\nb\nc"
      );
      // Bare top-level ColumnForm renders too (Column stays symbolic).
      assert_eq!(interpret("ColumnForm[{1, 2, 3}]").unwrap(), "1\n2\n3");
      // Trailing alignment arguments do not affect the text.
      assert_eq!(
        interpret("ToString[ColumnForm[{1, 2, 3}, Center]]").unwrap(),
        "1\n2\n3"
      );
      // ColumnForm renders even under InputForm (Column does not).
      assert_eq!(
        interpret("ToString[ColumnForm[{1, 2, 3}], InputForm]").unwrap(),
        "1\n2\n3"
      );
      // An empty list renders as an empty string.
      assert_eq!(interpret("ToString[ColumnForm[{}]]").unwrap(), "");
      // Regression: Column stays symbolic under InputForm.
      assert_eq!(
        interpret("ToString[Column[{1, 2, 3}], InputForm]").unwrap(),
        "Column[{1, 2, 3}]"
      );
    }

    #[test]
    fn test_matrix_form_renders_grid() {
      // MatrixForm renders an aligned text grid: every cell padded to a single
      // uniform width (the widest cell), left-aligned, three-space separators,
      // blank line between rows.
      assert_eq!(
        interpret("ToString[MatrixForm[{{1, 2}, {3, 4}}]]").unwrap(),
        "1   2\n\n3   4"
      );
      // Uniform width comes from the widest cell anywhere (here \"444\" -> 3).
      assert_eq!(
        interpret("ToString[MatrixForm[{{1, 22, 3}, {444, 5, 6}}]]").unwrap(),
        "1     22    3\n\n444   5     6"
      );
      // A single wide cell widens every column.
      assert_eq!(
        interpret("ToString[MatrixForm[{{1, 2}, {3, 4444}}]]").unwrap(),
        "1      2\n\n3      4444"
      );
      // A flat vector renders as a single column, one element per row.
      assert_eq!(
        interpret("ToString[MatrixForm[{1, 2, 3}]]").unwrap(),
        "1\n\n2\n\n3"
      );
      // Reals and negatives are rendered with their normal output form.
      assert_eq!(
        interpret("ToString[MatrixForm[{{1, -2}, {3.5, 4}}]]").unwrap(),
        "1     -2\n\n3.5   4"
      );
      // MatrixForm keeps its symbolic head under InputForm.
      assert_eq!(
        interpret("ToString[MatrixForm[{{1, 2}, {3, 4}}], InputForm]").unwrap(),
        "MatrixForm[{{1, 2}, {3, 4}}]"
      );
    }

    #[test]
    fn test_number_form_digit_block() {
      // DigitBlock -> n groups integer-part digits into blocks of n from the
      // right, separated by commas by default.
      assert_eq!(
        interpret("ToString[NumberForm[1234567, DigitBlock -> 3]]").unwrap(),
        "1,234,567"
      );
      // Block size 2.
      assert_eq!(
        interpret("ToString[NumberForm[12345678, DigitBlock -> 2]]").unwrap(),
        "12,34,56,78"
      );
      // Negative integers keep the sign before the grouped digits.
      assert_eq!(
        interpret("ToString[NumberForm[-1234567, DigitBlock -> 3]]").unwrap(),
        "-1,234,567"
      );
      // Fewer digits than a block: no separator added.
      assert_eq!(
        interpret("ToString[NumberForm[12, DigitBlock -> 3]]").unwrap(),
        "12"
      );
      // Reals: integer part grouped with commas, fractional part untouched
      // when shorter than a block.
      assert_eq!(
        interpret("ToString[NumberForm[1234.567, DigitBlock -> 3]]").unwrap(),
        "1,234.57"
      );
      assert_eq!(
        interpret("ToString[NumberForm[12345.6789, DigitBlock -> 2]]").unwrap(),
        "1,23,45.7"
      );
      // Fractional-part digits group from the left with spaces by default.
      assert_eq!(
        interpret("ToString[NumberForm[1.23456789, 9, DigitBlock -> 3]]")
          .unwrap(),
        "1.234 567 89"
      );
      assert_eq!(
        interpret("ToString[NumberForm[0.123456, 6, DigitBlock -> 2]]")
          .unwrap(),
        "0.12 34 56"
      );
      // NumberSeparator as a single string applies to both sides.
      assert_eq!(
        interpret(
          "ToString[NumberForm[1234567, DigitBlock -> 3, NumberSeparator -> \".\"]]"
        )
        .unwrap(),
        "1.234.567"
      );
      // NumberSeparator as {intSep, fracSep}.
      assert_eq!(
        interpret(
          "ToString[NumberForm[12345.6789, DigitBlock -> 3, NumberSeparator -> {\".\", \"_\"}]]"
        )
        .unwrap(),
        "12.345.7"
      );
      // DigitBlock keeps the symbolic head under InputForm.
      assert_eq!(
        interpret("ToString[NumberForm[1234567, DigitBlock -> 3], InputForm]")
          .unwrap(),
        "NumberForm[1234567, DigitBlock -> 3]"
      );
    }

    #[test]
    fn test_padded_form() {
      // A list pads each element to the same field width (n+1), right-aligned,
      // and renders as a brace list.
      assert_eq!(
        interpret("ToString[PaddedForm[{1, 22, 333}, 4]]").unwrap(),
        "{    1,    22,   333}"
      );
      // Negatives keep the sign within the reserved column.
      assert_eq!(
        interpret("ToString[PaddedForm[{1, -22, 333}, 4]]").unwrap(),
        "{    1,   -22,   333}"
      );
      assert_eq!(
        interpret("ToString[PaddedForm[{10, 200, 3}, 5]]").unwrap(),
        "{    10,    200,      3}"
      );
      // Scalar padding still works.
      assert_eq!(interpret("ToString[PaddedForm[12, 4]]").unwrap(), "   12");
      // Regression: an integer with a {n, f} spec stays an integer (no spurious
      // decimals) and is padded to width n+1.
      assert_eq!(
        interpret("ToString[PaddedForm[1, {4, 1}]]").unwrap(),
        "    1"
      );
      assert_eq!(
        interpret("ToString[PaddedForm[22, {4, 1}]]").unwrap(),
        "   22"
      );
      assert_eq!(
        interpret("ToString[PaddedForm[123, {2, 1}]]").unwrap(),
        " 123"
      );
      // Regression: when the number has more digits than n, the field widens
      // but still reserves a sign column.
      assert_eq!(interpret("ToString[PaddedForm[123, 2]]").unwrap(), " 123");
      assert_eq!(
        interpret("ToString[PaddedForm[12345, 3]]").unwrap(),
        " 12345"
      );
      // A real with a {n, f} spec is shown with exactly f decimals.
      assert_eq!(
        interpret("ToString[PaddedForm[1.5, {4, 2}]]").unwrap(),
        "  1.50"
      );
      // PaddedForm keeps its symbolic head under InputForm.
      assert_eq!(
        interpret("ToString[PaddedForm[{1, 22, 333}, 4], InputForm]").unwrap(),
        "PaddedForm[{1, 22, 333}, 4]"
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

  mod graphics_rotate_tests {
    use super::*;

    // Extract the four "x,y" vertex pairs from the first <polygon> element.
    fn polygon_points(svg: &str) -> Vec<(f64, f64)> {
      let start = svg.find("<polygon").expect("no polygon");
      let seg = &svg[start..];
      let pstart = seg.find("points=\"").expect("no points") + 8;
      let pend = seg[pstart..].find('"').unwrap() + pstart;
      seg[pstart..pend]
        .split_whitespace()
        .map(|pair| {
          let (x, y) = pair.split_once(',').unwrap();
          (x.parse::<f64>().unwrap(), y.parse::<f64>().unwrap())
        })
        .collect()
    }

    #[test]
    fn test_rotate_rectangle_becomes_tilted_polygon() {
      // A rotated rectangle is no longer axis-aligned: it must render as a
      // polygon whose edges are not parallel to the axes.
      let svg = interpret(
        "ExportString[Graphics[Rotate[Rectangle[{0, 0}, {2, 1}], 0.5]], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("<polygon"),
        "rotated rectangle should be a polygon: {svg}"
      );
      let pts = polygon_points(&svg);
      assert_eq!(pts.len(), 4, "expected 4 corners: {svg}");
      // No edge is purely horizontal or vertical once rotated.
      let axis_aligned = pts.windows(2).any(|w| {
        (w[0].0 - w[1].0).abs() < 1e-6 || (w[0].1 - w[1].1).abs() < 1e-6
      });
      assert!(!axis_aligned, "rotated rect edges must be tilted: {svg}");
    }

    #[test]
    fn test_rotate_by_zero_keeps_axis_alignment() {
      // Rotating by 0 must leave the shape axis-aligned (every edge stays
      // horizontal or vertical), even though a rectangle is emitted as a
      // polygon once wrapped in Rotate.
      let svg = interpret(
        "ExportString[Graphics[Rotate[Rectangle[{0, 0}, {2, 1}], 0]], \"SVG\"]",
      )
      .unwrap();
      let pts = polygon_points(&svg);
      assert_eq!(pts.len(), 4);
      for w in pts.windows(2) {
        assert!(
          (w[0].0 - w[1].0).abs() < 1e-6 || (w[0].1 - w[1].1).abs() < 1e-6,
          "edges must stay axis-aligned at angle 0: {svg}"
        );
      }
    }

    #[test]
    fn test_rotate_line_about_point() {
      // Rotating the horizontal segment {{1,0},{2,0}} by Pi/2 about {0,0}
      // maps (1,0)->(0,1) and (2,0)->(0,2): the line becomes vertical.
      let svg = interpret(
        "ExportString[Graphics[Rotate[Line[{{1, 0}, {2, 0}}], Pi/2, {0, 0}], PlotRange -> {{-1, 3}, {-1, 3}}], \"SVG\"]",
      )
      .unwrap();
      let start = svg.find("<polyline").expect("no polyline");
      let seg = &svg[start..];
      let pstart = seg.find("points=\"").unwrap() + 8;
      let pend = seg[pstart..].find('"').unwrap() + pstart;
      let xs: Vec<f64> = seg[pstart..pend]
        .split_whitespace()
        .map(|p| p.split_once(',').unwrap().0.parse::<f64>().unwrap())
        .collect();
      // A vertical segment has both endpoints at the same x pixel.
      assert!(
        (xs[0] - xs[1]).abs() < 1e-3,
        "rotated line should be vertical: {svg}"
      );
    }

    #[test]
    fn test_rotate_disk_moves_center_keeps_radius() {
      // Disk at {1,0}, r=0.25 rotated Pi/2 about origin -> center {0,1}.
      let svg = interpret(
        "ExportString[Graphics[Rotate[Disk[{1, 0}, 0.25], Pi/2, {0, 0}], PlotRange -> {{-1, 1}, {-1, 1}}], \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("<ellipse"), "disk should render: {svg}");
      // rx and ry unchanged by rotation (still a circle).
      let e = &svg[svg.find("<ellipse").unwrap()..];
      let rx = e[e.find("rx=\"").unwrap() + 4..]
        .split('"')
        .next()
        .unwrap()
        .parse::<f64>()
        .unwrap();
      let ry = e[e.find("ry=\"").unwrap() + 4..]
        .split('"')
        .next()
        .unwrap()
        .parse::<f64>()
        .unwrap();
      assert!((rx - ry).abs() < 1e-6, "disk stays circular: {svg}");
    }

    #[test]
    fn test_rectangle_reversed_corners_render() {
      // Wolfram accepts corners in any order. A reversed pair must still
      // produce a positive-size rect, not a dropped negative-height one.
      let svg = interpret(
        "ExportString[Graphics[Rectangle[{0, 2.}, {0.5, 1.5}]], \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("<rect"), "should render a rect: {svg}");
      let r = &svg[svg.find("<rect").unwrap()..];
      let h = r[r.find("height=\"").unwrap() + 8..]
        .split('"')
        .next()
        .unwrap()
        .parse::<f64>()
        .unwrap();
      assert!(h > 0.0, "height must be positive, got {h}: {svg}");
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

  mod string_count_option_tests {
    use super::*;

    #[test]
    fn test_default_is_non_overlapping() {
      assert_eq!(interpret(r#"StringCount["aaaa", "aa"]"#).unwrap(), "2");
      assert_eq!(
        interpret(r#"StringCount["aaaa", "aa", Overlaps -> False]"#).unwrap(),
        "2"
      );
    }

    #[test]
    fn test_overlaps_true_counts_every_start() {
      assert_eq!(
        interpret(r#"StringCount["aaaa", "aa", Overlaps -> True]"#).unwrap(),
        "3"
      );
      assert_eq!(
        interpret(r#"StringCount["abababab", "aba", Overlaps -> True]"#)
          .unwrap(),
        "3"
      );
    }

    #[test]
    fn test_ignore_case_option() {
      assert_eq!(
        interpret(r#"StringCount["AAaa", "aa", IgnoreCase -> True]"#).unwrap(),
        "2"
      );
    }

    #[test]
    fn test_overlaps_and_ignore_case_combined() {
      assert_eq!(
        interpret(
          r#"StringCount["AAAA", "aa", Overlaps -> True, IgnoreCase -> True]"#
        )
        .unwrap(),
        "3"
      );
    }

    #[test]
    fn test_overlaps_threads_over_list() {
      assert_eq!(
        interpret(r#"StringCount[{"aaa", "aaaa"}, "aa", Overlaps -> True]"#)
          .unwrap(),
        "{2, 3}"
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

    #[test]
    fn test_member_q_rational_complex_atoms() {
      // A Rational / Complex has no members: its parts are not searched.
      assert_eq!(interpret("MemberQ[1/2, 1]").unwrap(), "False");
      assert_eq!(interpret("MemberQ[1 + 2 I, 2]").unwrap(), "False");
      // The whole atom is still a level-1 member of a containing list.
      assert_eq!(interpret("MemberQ[{1/2, 3}, 1/2]").unwrap(), "True");
    }

    #[test]
    fn test_free_q_rational_complex_atoms() {
      // Internal numerator/denominator and real/imaginary parts are not
      // free-standing subexpressions.
      assert_eq!(interpret("FreeQ[1/2, 2]").unwrap(), "True");
      assert_eq!(interpret("FreeQ[1/2, 1]").unwrap(), "True");
      assert_eq!(interpret("FreeQ[3 + x/2, 2]").unwrap(), "True");
      assert_eq!(interpret("FreeQ[1 + 2 I, 2]").unwrap(), "True");
      assert_eq!(interpret("FreeQ[x + 2 I, 2]").unwrap(), "True");
      // The head symbol and the whole atom still match.
      assert_eq!(interpret("FreeQ[1/2, Rational]").unwrap(), "False");
      assert_eq!(interpret("FreeQ[1 + 2 I, Complex]").unwrap(), "False");
      assert_eq!(interpret("FreeQ[1/2, 1/2]").unwrap(), "False");
      assert_eq!(interpret("FreeQ[{1/2, 3}, 3]").unwrap(), "False");
      // Ordinary searches are unaffected.
      assert_eq!(interpret("FreeQ[x^2 + 1, x]").unwrap(), "False");
      assert_eq!(interpret("FreeQ[Sin[y], x]").unwrap(), "True");
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

    #[test]
    fn test_punctuation_character() {
      // Letters, digits and spaces are not punctuation; ASCII punctuation is.
      assert_eq!(
        interpret(
          r#"StringMatchQ[{"a", "1", " ", ".", "!", ",", "?", "@"}, PunctuationCharacter]"#
        )
        .unwrap(),
        "{False, False, False, True, True, True, True, True}"
      );
      // ASCII symbols Wolfram treats as punctuation.
      assert_eq!(
        interpret(
          r#"StringMatchQ[{"+", "=", "<", ">", "$", "^", "|", "~"}, PunctuationCharacter]"#
        )
        .unwrap(),
        "{True, True, True, True, True, True, True, True}"
      );
      // StringCases extracts punctuation characters.
      assert_eq!(
        interpret(r#"StringCases["Hello, world!", PunctuationCharacter]"#)
          .unwrap(),
        "{,, !}"
      );
      // Unicode dash (U+2013) is punctuation.
      assert_eq!(
        interpret(
          r#"StringMatchQ[FromCharacterCode[{8211}], PunctuationCharacter]"#
        )
        .unwrap(),
        "True"
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

    #[test]
    fn test_map_at_level_descends_into_operator_forms() {
      // Power (x^2 / Sqrt[x]) and other operator/special forms are stored as
      // BinaryOp/Comparison/Rule etc.; leveled Map must descend into their
      // canonical FullForm, like Level and Depth.
      assert_eq!(interpret("Map[f, x^2, {-1}]").unwrap(), "f[x]^f[2]");
      assert_eq!(interpret("Map[f, Sqrt[x], {-1}]").unwrap(), "f[x]^f[1/2]");
      assert_eq!(interpret("Map[f, 2^x^y, {-1}]").unwrap(), "f[2]^f[x]^f[y]");
      assert_eq!(interpret("Map[f, x^2, 2]").unwrap(), "f[x]^f[2]");
      assert_eq!(
        interpret("Map[f, x^2 + y^3, {2}]").unwrap(),
        "f[x]^f[2] + f[y]^f[3]"
      );
      // Comparison and Rule heads render infix after mapping.
      assert_eq!(interpret("Map[f, a == b, {-1}]").unwrap(), "f[a] == f[b]");
      assert_eq!(interpret("Map[f, a < b, {-1}]").unwrap(), "f[a] < f[b]");
      assert_eq!(interpret("Map[f, a -> b, {-1}]").unwrap(), "f[a] -> f[b]");
      // Plain lists/calls and atoms are unaffected.
      assert_eq!(interpret("Map[f, {a, b}, {-1}]").unwrap(), "{f[a], f[b]}");
      assert_eq!(interpret("Map[f, 1/2, {-1}]").unwrap(), "f[1/2]");
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

  mod wav_export_tests {
    use super::*;

    /// Assert the file at `tmp` is a mono 16-bit PCM WAV at the given
    /// sample rate and return the number of sample frames.
    fn assert_wav(tmp: &std::path::Path, expected_rate: u32) -> u32 {
      let bytes = std::fs::read(tmp).unwrap();
      assert!(bytes.starts_with(b"RIFF"), "missing RIFF header");
      assert_eq!(&bytes[8..12], b"WAVE", "missing WAVE tag");
      let rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
      assert_eq!(rate, expected_rate, "unexpected sample rate");
      let data_len = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
      assert_eq!(bytes.len(), 44 + data_len as usize, "truncated data chunk");
      data_len / 2
    }

    #[test]
    fn test_export_wav_play() {
      // wolframscript writes 16-bit mono PCM at Play's 8000 Hz default.
      let tmp = std::env::temp_dir().join("woxi_test_play.wav");
      let path = tmp.display().to_string();
      let result = interpret(&format!(
        "Export[\"{path}\", Play[Sin[440 2 Pi t], {{t, 0, 1}}]]"
      ))
      .unwrap();
      assert_eq!(result, path);
      let frames = assert_wav(&tmp, 8000);
      assert_eq!(frames, 8000, "1 second at 8000 Hz");
      // First PCM samples, byte-verified against wolframscript's export
      // (samples start at t = 1/8000, positive amplitudes scale by 32767).
      let bytes = std::fs::read(&tmp).unwrap();
      assert_eq!(&bytes[44..48], &[0x5b, 0x2b, 0x96, 0x51]);
      std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_wav_sound() {
      let tmp = std::env::temp_dir().join("woxi_test_sound.wav");
      let path = tmp.display().to_string();
      let result = interpret(&format!(
        "Export[\"{path}\", Sound[{{Play[Sin[220 2 Pi t], {{t, 0, 1}}], \
         Play[Sin[330 2 Pi t], {{t, 0, 1}}]}}]]"
      ))
      .unwrap();
      assert_eq!(result, path);
      // Two 1-second segments are concatenated.
      let frames = assert_wav(&tmp, 8000);
      assert_eq!(frames, 16000, "2 seconds at 8000 Hz");
      std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_wav_audio_samples() {
      let tmp = std::env::temp_dir().join("woxi_test_audio_samples.wav");
      let path = tmp.display().to_string();
      // 800 samples keep the symbolic Table cheap enough for debug-build CI
      // (8000 samples exceeded nextest's 20s timeout on the runner).
      let result = interpret(&format!(
        "Export[\"{path}\", Audio[Table[Sin[2 Pi 440 n/8000], {{n, 0, 799}}], \
         SampleRate -> 8000]]"
      ))
      .unwrap();
      assert_eq!(result, path);
      let frames = assert_wav(&tmp, 8000);
      assert_eq!(frames, 800, "0.1 seconds at 8000 Hz");
      std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_wav_explicit_format() {
      // The format can be given explicitly instead of via the extension.
      let tmp = std::env::temp_dir().join("woxi_test_explicit_wav.bin");
      let path = tmp.display().to_string();
      let result = interpret(&format!(
        "Export[\"{path}\", Play[Sin[440 2 Pi t], {{t, 0, 1}}], \"WAV\"]"
      ))
      .unwrap();
      assert_eq!(result, path);
      assert_wav(&tmp, 8000);
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
    fn test_absolute_timing_simple_expression() {
      // Regression for issue #128: `AbsoluteTiming[1 + 1]` panicked in the
      // WASM playground because `std::time::Instant::now()` aborts on
      // wasm32-unknown-unknown. The result must be `{<time>, 2}`.
      let result = interpret("AbsoluteTiming[1 + 1]").unwrap();
      assert!(result.starts_with('{') && result.ends_with('}'));
      assert!(result.ends_with(", 2}"));
      let inner = &result[1..result.len() - 1];
      let comma = inner.find(',').unwrap();
      let elapsed: f64 = inner[..comma].trim().parse().unwrap();
      assert!(
        elapsed >= 0.0,
        "elapsed time must be non-negative, got {elapsed}"
      );
    }

    #[test]
    fn test_timing_simple_expression() {
      let result = interpret("Timing[1 + 1]").unwrap();
      assert!(result.starts_with('{') && result.ends_with(", 2}"));
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

  mod file_template_tests {
    use std::io::Write;
    use woxi::{interpret, interpret_with_stdout};

    fn write_tmp(name: &str, contents: &str) -> std::path::PathBuf {
      let path = std::env::temp_dir().join(format!(
        "woxi_filetemplate_{}_{}",
        std::process::id(),
        name
      ));
      let mut f = std::fs::File::create(&path).unwrap();
      f.write_all(contents.as_bytes()).unwrap();
      path
    }

    #[test]
    fn test_file_template_renders_as_template_object() {
      // FileTemplate reads the file and produces a TemplateObject whose
      // rendered form matches wolframscript exactly.
      let path = write_tmp("object.txt", "Hello `name`, welcome to `place`!");
      let code = format!(r#"FileTemplate["{}"]"#, path.to_string_lossy());
      assert_eq!(
        interpret(&code).unwrap(),
        "TemplateObject[{Hello , TemplateSlot[name], , welcome to , \
         TemplateSlot[place], !}, CombinerFunction -> StringJoin, \
         InsertionFunction -> TextString, MetaInformation -> <||>]"
      );
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_template_head_is_template_object() {
      let path = write_tmp("head.txt", "x = `x`");
      let code = format!(r#"Head[FileTemplate["{}"]]"#, path.to_string_lossy());
      assert_eq!(interpret(&code).unwrap(), "TemplateObject");
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_template_numbered_slots() {
      let path = write_tmp("numbered.txt", "a `1` b `2`");
      let code = format!(r#"FileTemplate["{}"]"#, path.to_string_lossy());
      assert_eq!(
        interpret(&code).unwrap(),
        "TemplateObject[{a , TemplateSlot[1],  b , TemplateSlot[2]}, \
         CombinerFunction -> StringJoin, InsertionFunction -> TextString, \
         MetaInformation -> <||>]"
      );
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_template_no_slots() {
      let path = write_tmp("plain.txt", "plain text");
      let code = format!(r#"FileTemplate["{}"]"#, path.to_string_lossy());
      assert_eq!(
        interpret(&code).unwrap(),
        "TemplateObject[{plain text}, CombinerFunction -> StringJoin, \
         InsertionFunction -> TextString, MetaInformation -> <||>]"
      );
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_template_apply_on_file_template_association() {
      let path = write_tmp("apply.txt", "Hello `name`, welcome to `place`!");
      let code = format!(
        r#"TemplateApply[FileTemplate["{}"], <|"name" -> "Bob", "place" -> "Earth"|>]"#,
        path.to_string_lossy()
      );
      assert_eq!(interpret(&code).unwrap(), "Hello Bob, welcome to Earth!");
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_template_apply_on_file_template_numbered_list() {
      let path = write_tmp("applynum.txt", "a `1` b `2`");
      let code = format!(
        r#"TemplateApply[FileTemplate["{}"], {{"X", "Y"}}]"#,
        path.to_string_lossy()
      );
      assert_eq!(interpret(&code).unwrap(), "a X b Y");
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_template_with_file_wrapper() {
      let path = write_tmp("wrapper.txt", "v = `v`");
      let code =
        format!(r#"Head[FileTemplate[File["{}"]]]"#, path.to_string_lossy());
      assert_eq!(interpret(&code).unwrap(), "TemplateObject");
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_template_missing_file_fails() {
      let path = std::env::temp_dir().join(format!(
        "woxi_filetemplate_missing_{}.txt",
        std::process::id()
      ));
      let _ = std::fs::remove_file(&path);
      let code = format!(r#"FileTemplate["{}"]"#, path.to_string_lossy());
      let r = interpret_with_stdout(&code).unwrap();
      assert_eq!(r.result, "$Failed");
      assert!(
        r.stdout.contains("StringTemplate::fnfnd:"),
        "stdout was: {:?}",
        r.stdout
      );
    }
  }

  mod xml_template_tests {
    use std::io::Write;
    use woxi::{interpret, interpret_with_stdout};

    #[test]
    fn test_xml_template_renders_with_html_fragment() {
      // XMLTemplate uses InsertionFunction -> HTMLFragment (vs TextString).
      assert_eq!(
        interpret(r#"XMLTemplate["Hi `name`"]"#).unwrap(),
        "TemplateObject[{Hi , TemplateSlot[name]}, CombinerFunction -> \
         StringJoin, InsertionFunction -> HTMLFragment, MetaInformation -> <||>]"
      );
    }

    #[test]
    fn test_xml_template_head_is_template_object() {
      assert_eq!(
        interpret(r#"Head[XMLTemplate["Hi `a`"]]"#).unwrap(),
        "TemplateObject"
      );
    }

    #[test]
    fn test_xml_template_expression_renders() {
      // `<* expr *>` becomes TemplateExpression, with `#a` -> TemplateSlot[a].
      assert_eq!(
        interpret(r#"XMLTemplate["Range of `a`: <* Range[#a] *>."]"#).unwrap(),
        "TemplateObject[{Range of , TemplateSlot[a], : , \
         TemplateExpression[Range[TemplateSlot[a]]], .}, CombinerFunction -> \
         StringJoin, InsertionFunction -> HTMLFragment, MetaInformation -> <||>]"
      );
    }

    #[test]
    fn test_xml_template_apply_slot_and_expression() {
      // The documentation example: a slot plus an embedded expression.
      assert_eq!(
        interpret(
          r#"TemplateApply[XMLTemplate["Range of `a`: <* Range[#a] *>."], Association["a" -> 5]]"#
        )
        .unwrap(),
        "Range of 5: {1, 2, 3, 4, 5}."
      );
    }

    #[test]
    fn test_xml_template_apply_numbered_expression() {
      // Numbered slot references `#1`, `#2` inside an expression.
      assert_eq!(
        interpret(r#"TemplateApply[XMLTemplate["<* #1 + #2 *>"], {3, 4}]"#)
          .unwrap(),
        "7"
      );
    }

    #[test]
    fn test_xml_template_two_arg_binds_args() {
      assert_eq!(
        interpret(r#"XMLTemplate["Hi `a`", <|"a" -> 1|>]"#).unwrap(),
        "TemplateObject[{Hi , TemplateSlot[a]}, <|a -> 1|>, CombinerFunction \
         -> StringJoin, InsertionFunction -> HTMLFragment, MetaInformation -> \
         <||>]"
      );
    }

    #[test]
    fn test_xml_template_from_file() {
      let path = std::env::temp_dir()
        .join(format!("woxi_xmltemplate_{}.xml", std::process::id()));
      let mut f = std::fs::File::create(&path).unwrap();
      f.write_all(b"Hello `name`").unwrap();
      let code =
        format!(r#"Head[XMLTemplate[File["{}"]]]"#, path.to_string_lossy());
      assert_eq!(interpret(&code).unwrap(), "TemplateObject");
      let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_xml_template_missing_file_fails() {
      let path = std::env::temp_dir().join(format!(
        "woxi_xmltemplate_missing_{}.xml",
        std::process::id()
      ));
      let _ = std::fs::remove_file(&path);
      let code = format!(r#"XMLTemplate[File["{}"]]"#, path.to_string_lossy());
      let r = interpret_with_stdout(&code).unwrap();
      assert_eq!(r.result, "$Failed");
      assert!(
        r.stdout.contains("XMLTemplate::fnfnd:"),
        "stdout was: {:?}",
        r.stdout
      );
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
        interpret("CholeskyDecomposition[{{25,15,-5},{15,18,0},{-5,0,11}}]")
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

  mod shearing_transform_tests {
    use super::*;

    #[test]
    fn test_basic_2d() {
      // ShearingTransform[Pi/4, {1,0}, {0,1}] → unit horizontal shear.
      assert_eq!(
        interpret("ShearingTransform[Pi/4, {1,0}, {0,1}]").unwrap(),
        "TransformationFunction[{{1, 1, 0}, {0, 1, 0}, {0, 0, 1}}]"
      );
    }

    #[test]
    fn test_apply_to_point() {
      // Applying the shear maps {2,3} → {2 + 3, 3} = {5, 3}.
      assert_eq!(
        interpret("ShearingTransform[Pi/4, {1,0}, {0,1}][{2,3}]").unwrap(),
        "{5, 3}"
      );
    }

    #[test]
    fn test_direction_is_normalized() {
      // Non-unit direction/normal vectors are normalized: same as {1,0},{0,1}.
      assert_eq!(
        interpret("ShearingTransform[Pi/4, {2,0}, {0,3}]").unwrap(),
        "TransformationFunction[{{1, 1, 0}, {0, 1, 0}, {0, 0, 1}}]"
      );
    }

    #[test]
    fn test_perpendicular_direction_and_normal() {
      // e={1,1}, n={1,-1} are perpendicular; both normalized by Sqrt[2].
      assert_eq!(
        interpret("ShearingTransform[Pi/4, {1,1}, {1,-1}]").unwrap(),
        "TransformationFunction[{{3/2, -1/2, 0}, {1/2, 1/2, 0}, {0, 0, 1}}]"
      );
    }

    #[test]
    fn test_only_perpendicular_component_used() {
      // e={3,4} has a component along n={0,1}; only the perpendicular part
      // ({3,0}→{1,0}) drives the shear, giving Tan[Pi/6] = 1/Sqrt[3].
      assert_eq!(
        interpret("ShearingTransform[Pi/6, {3,4}, {0,1}]").unwrap(),
        "TransformationFunction[{{1, 1/Sqrt[3], 0}, {0, 1, 0}, {0, 0, 1}}]"
      );
    }

    #[test]
    fn test_symbolic_3d() {
      assert_eq!(
        interpret("ShearingTransform[Pi/3, {1,2,0}, {0,0,1}]").unwrap(),
        "TransformationFunction[{{1, 0, Sqrt[3/5], 0}, \
         {0, 1, 2*Sqrt[3/5], 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}]"
      );
    }

    #[test]
    fn test_symbolic_angle() {
      assert_eq!(
        interpret("ShearingTransform[t, {1,0}, {0,1}]").unwrap(),
        "TransformationFunction[{{1, Tan[t], 0}, {0, 1, 0}, {0, 0, 1}}]"
      );
    }

    #[test]
    fn test_inexact_input_promotes_matrix() {
      assert_eq!(
        interpret("ShearingTransform[0.5, {1,0}, {0,1}]").unwrap(),
        "TransformationFunction[{{1., 0.5463024898437905, 0.}, \
         {0., 1., 0.}, {0., 0., 1.}}]"
      );
    }
  }

  mod polynomial_extended_gcd_tests {
    use super::*;

    #[test]
    fn test_basic() {
      // {g, {s, t}} with s*p + t*q == g, g monic.
      assert_eq!(
        interpret("PolynomialExtendedGCD[x^2 - 1, x^2 - 3 x + 2, x]").unwrap(),
        "{-1 + x, {1/3, -1/3}}"
      );
    }

    #[test]
    fn test_linear_gcd() {
      assert_eq!(
        interpret("PolynomialExtendedGCD[x^2 + 7 x + 6, x^2 - 5 x - 6, x]")
          .unwrap(),
        "{1 + x, {1/12, -1/12}}"
      );
    }

    #[test]
    fn test_polynomial_cofactor() {
      assert_eq!(
        interpret("PolynomialExtendedGCD[x^4 - 1, x^3 - 1, x]").unwrap(),
        "{-1 + x, {1, -x}}"
      );
    }

    #[test]
    fn test_divisor_pair() {
      assert_eq!(
        interpret("PolynomialExtendedGCD[(x - 1)^2, x - 1, x]").unwrap(),
        "{-1 + x, {0, 1}}"
      );
    }

    #[test]
    fn test_coprime_collapses_denominator() {
      // Coprime: g == 1, cofactors share a common denominator that
      // wolframscript collapses into a single factor.
      assert_eq!(
        interpret("PolynomialExtendedGCD[x^2 + 1, x + 1, x]").unwrap(),
        "{1, {1/2, (1 - x)/2}}"
      );
    }

    #[test]
    fn test_coprime_quadratic_cofactor() {
      assert_eq!(
        interpret("PolynomialExtendedGCD[x^3 + 1, x^2 + 1, x]").unwrap(),
        "{1, {(1 + x)/2, (1 - x - x^2)/2}}"
      );
    }

    #[test]
    fn test_content_is_normalized_to_monic() {
      // PolynomialGCD keeps content; the extended GCD is monic.
      assert_eq!(
        interpret("PolynomialExtendedGCD[2 x^2 - 2, 2 x^2 - 6 x + 4, x]")
          .unwrap(),
        "{-1 + x, {1/6, -1/6}}"
      );
    }

    #[test]
    fn test_constant_arguments() {
      // Over a field both are units: g == 1, s == 1/p1, t == 0.
      assert_eq!(
        interpret("PolynomialExtendedGCD[6, 9, x]").unwrap(),
        "{1, {1/6, 0}}"
      );
    }

    #[test]
    fn test_zero_argument() {
      assert_eq!(
        interpret("PolynomialExtendedGCD[x^2 - 1, 0, x]").unwrap(),
        "{-1 + x^2, {1, 0}}"
      );
      assert_eq!(
        interpret("PolynomialExtendedGCD[0, x - 1, x]").unwrap(),
        "{-1 + x, {0, 1}}"
      );
    }
  }

  mod graph_distance_tests {
    use super::*;

    #[test]
    fn test_path_distance() {
      assert_eq!(
        interpret("GraphDistance[Graph[{1,2,3,4},{1<->2,2<->3,3<->4}],1,4]")
          .unwrap(),
        "3"
      );
    }

    #[test]
    fn test_self_distance_is_zero() {
      assert_eq!(
        interpret("GraphDistance[Graph[{1,2,3,4},{1<->2,2<->3,3<->4}],1,1]")
          .unwrap(),
        "0"
      );
    }

    #[test]
    fn test_complete_graph() {
      assert_eq!(
        interpret("GraphDistance[CompleteGraph[4],1,3]").unwrap(),
        "1"
      );
    }

    // CompleteKaryTree[n] / CompleteKaryTree[n, k]: complete k-ary tree with
    // n levels (depth n-1, default branching factor k=2). Vertices are
    // numbered breadth-first. Outputs verified against wolframscript.
    #[test]
    fn test_complete_kary_tree() {
      // The bare Graph summary (`Graph[<7>, <6>]`) is a Woxi-specific display;
      // assert the size via VertexCount/EdgeCount, which both engines agree on.
      // The vertex/edge structure itself is checked by the assertions below.
      assert_eq!(
        interpret("{VertexCount[#], EdgeCount[#]} &[CompleteKaryTree[3]]")
          .unwrap(),
        "{7, 6}"
      );
      assert_eq!(
        interpret(
          "g = CompleteKaryTree[3]; Apply[List, {VertexList[g], EdgeList[g]}, {2}]"
        )
        .unwrap(),
        "{{1, 2, 3, 4, 5, 6, 7}, \
         {{1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 6}, {3, 7}}}"
      );
      // Branching factor k = 3, n = 3 levels -> 13 vertices.
      assert_eq!(
        interpret(
          "g = CompleteKaryTree[3, 3]; \
           Apply[List, {VertexList[g], EdgeList[g]}, {2}]"
        )
        .unwrap(),
        "{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, \
         {{1, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6}, {2, 7}, \
         {3, 8}, {3, 9}, {3, 10}, {4, 11}, {4, 12}, {4, 13}}}"
      );
      // k = 1 degenerates to a path on n vertices.
      assert_eq!(
        interpret(
          "g = CompleteKaryTree[4, 1]; \
           Apply[List, {VertexList[g], EdgeList[g]}, {2}]"
        )
        .unwrap(),
        "{{1, 2, 3, 4}, {{1, 2}, {2, 3}, {3, 4}}}"
      );
      // Single level: one isolated vertex, no edges.
      assert_eq!(
        interpret("g = CompleteKaryTree[1]; {VertexList[g], EdgeList[g]}")
          .unwrap(),
        "{{1}, {}}"
      );
    }

    // WheelGraph[n]: hub vertex 1 joined to rim vertices 2..n, plus a cycle
    // on the rim. Vertex/edge ordering verified against wolframscript.
    #[test]
    fn test_wheel_graph_default_render() {
      // `Graph[<5>, <8>]` is a Woxi-specific summary; assert the size via
      // VertexCount/EdgeCount (the structure is checked in the test below).
      assert_eq!(
        interpret("{VertexCount[#], EdgeCount[#]} &[WheelGraph[5]]").unwrap(),
        "{5, 8}"
      );
    }

    #[test]
    fn test_wheel_graph_vertices_and_edges() {
      assert_eq!(
        interpret(
          "g = WheelGraph[6]; Apply[List, {VertexList[g], EdgeList[g]}, {2}]"
        )
        .unwrap(),
        "{{1, 2, 3, 4, 5, 6}, \
         {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, \
         {2, 3}, {2, 6}, {3, 4}, {4, 5}, {5, 6}}}"
      );
    }

    #[test]
    fn test_wheel_graph_n4() {
      assert_eq!(
        interpret(
          "g = WheelGraph[4]; Apply[List, {VertexList[g], EdgeList[g]}, {2}]"
        )
        .unwrap(),
        "{{1, 2, 3, 4}, {{1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}}}"
      );
    }

    // Degenerate small cases must mirror wolframscript exactly.
    #[test]
    fn test_wheel_graph_n3_double_edge() {
      assert_eq!(
        interpret(
          "g = WheelGraph[3]; Apply[List, {VertexList[g], EdgeList[g]}, {2}]"
        )
        .unwrap(),
        "{{1, 2, 3}, {{1, 2}, {1, 3}, {2, 3}, {2, 3}}}"
      );
    }

    #[test]
    fn test_wheel_graph_n2_self_loop() {
      assert_eq!(
        interpret(
          "g = WheelGraph[2]; Apply[List, {VertexList[g], EdgeList[g]}, {2}]"
        )
        .unwrap(),
        "{{1, 2}, {{1, 2}, {2, 2}}}"
      );
    }

    #[test]
    fn test_wheel_graph_n1() {
      assert_eq!(
        interpret("g = WheelGraph[1]; {VertexList[g], EdgeList[g]}").unwrap(),
        "{{1}, {}}"
      );
    }

    // FindCycle: returns cycles in a (di)graph. Each cycle is a list of edges.
    // Output forms below are verified to match wolframscript exactly. Cycle
    // edges are rendered with Apply[List, ..., {2}] (so {a, b} stands for the
    // edge a->b) since the raw DirectedEdge/UndirectedEdge glyphs are awkward
    // to assert on.
    mod find_cycle_tests {
      use super::*;

      #[test]
      fn test_directed_triangle_default() {
        // Default returns a single cycle.
        assert_eq!(
          interpret("Apply[List, FindCycle[{1->2, 2->3, 3->1}], {2}]").unwrap(),
          "{{{1, 2}, {2, 3}, {3, 1}}}"
        );
      }

      #[test]
      fn test_no_cycle_returns_empty() {
        assert_eq!(interpret("FindCycle[{1->2, 2->3, 3->4}]").unwrap(), "{}");
      }

      #[test]
      fn test_self_loop_not_a_cycle() {
        // A length-1 self-loop is not reported as a cycle.
        assert_eq!(
          interpret("FindCycle[{1->1}, Infinity, All]").unwrap(),
          "{}"
        );
      }

      #[test]
      fn test_directed_edge_heads() {
        assert_eq!(
          interpret("Head /@ FindCycle[{1->2, 2->3, 3->1}][[1]]").unwrap(),
          "{DirectedEdge, DirectedEdge, DirectedEdge}"
        );
      }

      #[test]
      fn test_all_cycles_sorted_by_length() {
        // Three nested cycles through vertex 1, all of length 2/3/4, returned
        // shortest first.
        assert_eq!(
          interpret(
            "Apply[List, FindCycle[\
               {1->2, 2->3, 3->1, 3->4, 4->1, 1->5, 5->1}, Infinity, All], \
             {2}]"
          )
          .unwrap(),
          "{{{1, 5}, {5, 1}}, {{1, 2}, {2, 3}, {3, 1}}, \
           {{1, 2}, {2, 3}, {3, 4}, {4, 1}}}"
        );
      }

      #[test]
      fn test_count_limits_number_of_cycles() {
        // Default (no count) returns the first cycle found by DFS.
        assert_eq!(
          interpret(
            "Apply[List, \
               FindCycle[{1->2, 2->3, 3->1, 3->4, 4->1, 1->5, 5->1}], {2}]"
          )
          .unwrap(),
          "{{{1, 2}, {2, 3}, {3, 1}}}"
        );
      }

      #[test]
      fn test_disjoint_cycles_reverse_root_order() {
        // Two independent 2-cycles: All lists them in reverse vertex order.
        assert_eq!(
          interpret(
            "Apply[List, \
               FindCycle[{1->2, 2->1, 3->4, 4->3}, Infinity, All], {2}]"
          )
          .unwrap(),
          "{{{3, 4}, {4, 3}}, {{1, 2}, {2, 1}}}"
        );
      }

      #[test]
      fn test_kspec_max_length() {
        // A single integer second argument bounds the maximum cycle length.
        assert_eq!(
          interpret(
            "Apply[List, \
               FindCycle[{1->2, 2->3, 3->1, 1->4, 4->1}, 2, All], {2}]"
          )
          .unwrap(),
          "{{{1, 4}, {4, 1}}}"
        );
      }

      #[test]
      fn test_undirected_triangle() {
        // A two-way (undirected) cycle uses UndirectedEdge in the result.
        assert_eq!(
          interpret(
            "Apply[List, FindCycle[{1<->2, 2<->3, 3<->1}, Infinity, All], {2}]"
          )
          .unwrap(),
          "{{{1, 2}, {2, 3}, {3, 1}}}"
        );
        assert_eq!(
          interpret("Head /@ FindCycle[{1<->2, 2<->3, 3<->1}][[1]]").unwrap(),
          "{UndirectedEdge, UndirectedEdge, UndirectedEdge}"
        );
      }

      #[test]
      fn test_single_undirected_edge_is_not_a_cycle() {
        // One undirected edge cannot be traversed back along itself.
        assert_eq!(
          interpret("FindCycle[{1<->2}, Infinity, All]").unwrap(),
          "{}"
        );
      }

      #[test]
      fn test_findcycle_on_graph_object() {
        // Accepts a Graph[...] wrapper and honours its vertex order.
        assert_eq!(
          interpret(
            "Apply[List, \
               FindCycle[Graph[{2, 3, 1}, {1->2, 2->3, 3->1}], Infinity, All], \
             {2}]"
          )
          .unwrap(),
          "{{{2, 3}, {3, 1}, {1, 2}}}"
        );
      }
    }

    #[test]
    fn test_relation_graph() {
      // Asymmetric relation → directed graph with an edge for every ordered
      // pair (i, j) where the relation holds.
      assert_eq!(
        interpret("Apply[List, EdgeList[RelationGraph[Less, {1, 2, 3}]], {1}]")
          .unwrap(),
        "{{1, 2}, {1, 3}, {2, 3}}"
      );
      assert_eq!(
        interpret("Head /@ EdgeList[RelationGraph[Less, {1, 2, 3}]]").unwrap(),
        "{DirectedEdge, DirectedEdge, DirectedEdge}"
      );
      assert_eq!(
        interpret("DirectedGraphQ[RelationGraph[Less, {1, 2, 3}]]").unwrap(),
        "True"
      );

      // Symmetric relation → undirected graph with one edge per unordered pair.
      assert_eq!(
        interpret(
          "Apply[List, EdgeList[RelationGraph[#1 != #2 &, {1, 2, 3}]], {1}]"
        )
        .unwrap(),
        "{{1, 2}, {1, 3}, {2, 3}}"
      );
      assert_eq!(
        interpret("Head /@ EdgeList[RelationGraph[#1 != #2 &, {1, 2, 3}]]")
          .unwrap(),
        "{UndirectedEdge, UndirectedEdge, UndirectedEdge}"
      );
      assert_eq!(
        interpret("DirectedGraphQ[RelationGraph[#1 != #2 &, {1, 2, 3}]]")
          .unwrap(),
        "False"
      );

      // Symmetric relation that holds on the diagonal → undirected self-loops.
      assert_eq!(
        interpret(
          "Apply[List, EdgeList[RelationGraph[#1 == #2 &, {1, 2, 3}]], {1}]"
        )
        .unwrap(),
        "{{1, 1}, {2, 2}, {3, 3}}"
      );

      // Asymmetric relation that holds on the diagonal → directed self-loops,
      // edges enumerated in lexicographic (i, j) order.
      assert_eq!(
        interpret(
          "Apply[List, EdgeList[RelationGraph[#1 >= #2 &, {1, 2, 3}]], {1}]"
        )
        .unwrap(),
        "{{1, 1}, {2, 1}, {2, 2}, {3, 1}, {3, 2}, {3, 3}}"
      );

      // Vertices are preserved verbatim, including non-integer vertices.
      assert_eq!(
        interpret("VertexList[RelationGraph[Less, {1, 2, 3}]]").unwrap(),
        "{1, 2, 3}"
      );
      assert_eq!(
        interpret("EdgeList[RelationGraph[False &, {1, 2, 3}]]").unwrap(),
        "{}"
      );
    }

    #[test]
    fn test_vertex_index() {
      // 1-based position of a vertex in VertexList order.
      assert_eq!(
        interpret("VertexIndex[Graph[{1 -> 2, 2 -> 3, 3 -> 1}], 2]").unwrap(),
        "2"
      );
      // Insertion order is preserved, not sorted.
      assert_eq!(
        interpret("VertexIndex[Graph[{3 -> 1, 1 -> 2}], 3]").unwrap(),
        "1"
      );
      // String vertices.
      assert_eq!(
        interpret(
          "VertexIndex[Graph[{\"a\" -> \"b\", \"b\" -> \"c\"}], \"c\"]"
        )
        .unwrap(),
        "3"
      );
      // CompleteGraph form.
      assert_eq!(interpret("VertexIndex[CompleteGraph[4], 3]").unwrap(), "3");
      // Invalid vertex returns unevaluated.
      assert_eq!(
        interpret("VertexIndex[Graph[{1 -> 2, 2 -> 3, 3 -> 1}], 5]").unwrap(),
        "VertexIndex[Graph[<3>, <3>], 5]"
      );
    }

    #[test]
    fn test_vertex_delete() {
      // Single vertex: removes the vertex and every incident edge.
      assert_eq!(
        interpret(
          "g = VertexDelete[Graph[{1 -> 2, 2 -> 3, 3 -> 1}], 2]; \
           {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{1, 3}, {3 \u{F3D5} 1}}"
      );
      // A list deletes multiple vertices.
      assert_eq!(
        interpret(
          "g = VertexDelete[Graph[{1 -> 2, 2 -> 3, 3 -> 1, 1 -> 3}], \
           {1, 2}]; {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{3}, {}}"
      );
      // Undirected edges are handled too.
      assert_eq!(
        interpret(
          "g = VertexDelete[Graph[{1 <-> 2, 2 <-> 3, 3 <-> 4, 4 <-> 1}], \
           3]; {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{1, 2, 4}, {1 \u{F3D4} 2, 4 \u{F3D4} 1}}"
      );
      // Explicit vertex/edge form keeps isolated vertices.
      assert_eq!(
        interpret(
          "g = VertexDelete[Graph[{1, 2, 3, 4}, {1 -> 2, 2 -> 3}], 2]; \
           {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{1, 3, 4}, {}}"
      );
      // A singleton list behaves like a single vertex.
      assert_eq!(
        interpret(
          "g = VertexDelete[Graph[{1 -> 2, 2 -> 3, 3 -> 1}], {2}]; \
           {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{1, 3}, {3 \u{F3D5} 1}}"
      );
      // Invalid vertex leaves the expression unevaluated.
      assert_eq!(
        interpret("VertexDelete[Graph[{1 -> 2, 2 -> 3}], 5]").unwrap(),
        "VertexDelete[Graph[<3>, <2>], 5]"
      );
      // If any vertex in the list is invalid, nothing is deleted.
      assert_eq!(
        interpret("VertexDelete[Graph[{1 -> 2, 2 -> 3, 3 -> 1}], {2, 9}]")
          .unwrap(),
        "VertexDelete[Graph[<3>, <3>], {2, 9}]"
      );
    }

    #[test]
    fn test_edge_delete() {
      // Single directed edge: removes the edge, keeps all vertices.
      assert_eq!(
        interpret(
          "g = EdgeDelete[Graph[{1 -> 2, 2 -> 3, 3 -> 1}], 1 -> 2]; \
           {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{1, 2, 3}, {2 \u{F3D5} 3, 3 \u{F3D5} 1}}"
      );
      // A list deletes multiple edges (one occurrence each).
      assert_eq!(
        interpret(
          "EdgeList[EdgeDelete[Graph[{1 -> 2, 2 -> 3, 3 -> 1}], \
           {1 -> 2, 2 -> 3}]]"
        )
        .unwrap(),
        "{3 \u{F3D5} 1}"
      );
      // Duplicate edges: only one matching occurrence is removed.
      assert_eq!(
        interpret(
          "EdgeList[EdgeDelete[Graph[{1 -> 2, 1 -> 2, 2 -> 3}], 1 -> 2]]"
        )
        .unwrap(),
        "{1 \u{F3D5} 2, 2 \u{F3D5} 3}"
      );
      // Undirected edges match regardless of endpoint order.
      assert_eq!(
        interpret("EdgeList[EdgeDelete[Graph[{1 <-> 2, 2 <-> 3}], 2 <-> 1]]")
          .unwrap(),
        "{2 \u{F3D4} 3}"
      );
      // DirectedEdge form is accepted as the edge argument.
      assert_eq!(
        interpret(
          "EdgeList[EdgeDelete[Graph[{1 -> 2, 2 -> 3}], DirectedEdge[1, 2]]]"
        )
        .unwrap(),
        "{2 \u{F3D5} 3}"
      );
      // Isolated vertices are preserved.
      assert_eq!(
        interpret(
          "VertexList[EdgeDelete[Graph[{1, 2, 3, 4}, {1 -> 2, 2 -> 3}], \
           1 -> 2]]"
        )
        .unwrap(),
        "{1, 2, 3, 4}"
      );
      // A non-existent edge leaves the expression unevaluated.
      assert_eq!(
        interpret("EdgeDelete[Graph[{1 -> 2, 2 -> 3}], 1 -> 3]").unwrap(),
        "EdgeDelete[Graph[<3>, <2>], 1 -> 3]"
      );
      // An undirected edge does not match a directed one.
      assert_eq!(
        interpret("EdgeDelete[Graph[{1 -> 2, 2 -> 3}], 1 <-> 2]").unwrap(),
        "EdgeDelete[Graph[<3>, <2>], 1 <-> 2]"
      );
    }

    #[test]
    fn test_edge_add() {
      // Single undirected edge introducing a new vertex: edge appended,
      // new vertex added at the end of the vertex list.
      assert_eq!(
        interpret(
          "g = EdgeAdd[CompleteGraph[3], 1 <-> 4]; \
           {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{1, 2, 3, 4}, {1 \u{F3D4} 2, 1 \u{F3D4} 3, 2 \u{F3D4} 3, \
         1 \u{F3D4} 4}}"
      );
      // A list adds several edges, adding new vertices in order.
      assert_eq!(
        interpret(
          "g = EdgeAdd[CompleteGraph[3], {3 <-> 4, 4 <-> 5}]; \
           {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{1, 2, 3, 4, 5}, {1 \u{F3D4} 2, 1 \u{F3D4} 3, 2 \u{F3D4} 3, \
         3 \u{F3D4} 4, 4 \u{F3D4} 5}}"
      );
      // Directed edge (Rule form) is stored as DirectedEdge.
      assert_eq!(
        interpret(
          "g = EdgeAdd[Graph[{1 -> 2, 2 -> 3}], 3 -> 1]; \
           {VertexList[g], EdgeList[g]}"
        )
        .unwrap(),
        "{{1, 2, 3}, {1 \u{F3D5} 2, 2 \u{F3D5} 3, 3 \u{F3D5} 1}}"
      );
      // Re-adding an existing edge yields a multigraph (edge kept twice).
      assert_eq!(
        interpret("EdgeList[EdgeAdd[CompleteGraph[3], 1 <-> 2]]").unwrap(),
        "{1 \u{F3D4} 2, 1 \u{F3D4} 3, 2 \u{F3D4} 3, 1 \u{F3D4} 2}"
      );
      // DirectedEdge / UndirectedEdge spellings are accepted.
      assert_eq!(
        interpret("EdgeList[EdgeAdd[Graph[{1 -> 2}], DirectedEdge[2, 3]]]")
          .unwrap(),
        "{1 \u{F3D5} 2, 2 \u{F3D5} 3}"
      );
      // An invalid edge argument leaves the expression unevaluated.
      assert_eq!(
        interpret("EdgeAdd[CompleteGraph[3], 5]").unwrap(),
        "EdgeAdd[Graph[<3>, <3>], 5]"
      );
    }

    #[test]
    fn test_single_source_distances() {
      // Two-arg form returns distances to every vertex, in VertexList order.
      assert_eq!(
        interpret("GraphDistance[Graph[{1,2,3,4},{1<->2,2<->3,3<->4}],1]")
          .unwrap(),
        "{0, 1, 2, 3}"
      );
    }

    #[test]
    fn test_unreachable_is_infinity() {
      assert_eq!(
        interpret("GraphDistance[Graph[{1,2,3,4},{1<->2}],1,4]").unwrap(),
        "Infinity"
      );
      assert_eq!(
        interpret("GraphDistance[Graph[{1,2,3,4},{1<->2}],3]").unwrap(),
        "{Infinity, Infinity, 0, Infinity}"
      );
    }

    #[test]
    fn test_directed_edges_honoured() {
      assert_eq!(
        interpret("GraphDistance[Graph[{1,2,3},{1->2,2->3}],1,3]").unwrap(),
        "2"
      );
      // No path against the edge direction.
      assert_eq!(
        interpret("GraphDistance[Graph[{1,2,3},{1->2,2->3}],3,1]").unwrap(),
        "Infinity"
      );
    }

    #[test]
    fn test_string_vertices() {
      assert_eq!(
        interpret(
          "GraphDistance[Graph[{\"a\",\"b\",\"c\"},{\"a\"<->\"b\",\"b\"<->\"c\"}],\"a\",\"c\"]"
        )
        .unwrap(),
        "2"
      );
    }
  }

  mod graph_distance_matrix_tests {
    use super::*;

    #[test]
    fn test_complete_graph() {
      assert_eq!(
        interpret("GraphDistanceMatrix[CompleteGraph[3]]").unwrap(),
        "{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}"
      );
    }

    #[test]
    fn test_path_graph() {
      assert_eq!(
        interpret("GraphDistanceMatrix[PathGraph[{1,2,3,4}]]").unwrap(),
        "{{0, 1, 2, 3}, {1, 0, 1, 2}, {2, 1, 0, 1}, {3, 2, 1, 0}}"
      );
    }

    #[test]
    fn test_cycle_graph() {
      assert_eq!(
        interpret("GraphDistanceMatrix[CycleGraph[4]]").unwrap(),
        "{{0, 1, 2, 1}, {1, 0, 1, 2}, {2, 1, 0, 1}, {1, 2, 1, 0}}"
      );
    }

    #[test]
    fn test_directed_edges_yield_infinity() {
      assert_eq!(
        interpret("GraphDistanceMatrix[Graph[{1->2,2->3}]]").unwrap(),
        "{{0, 1, 2}, {Infinity, 0, 1}, {Infinity, Infinity, 0}}"
      );
    }

    #[test]
    fn test_disconnected_components() {
      assert_eq!(
        interpret("GraphDistanceMatrix[Graph[{1,2,3},{1<->2}]]").unwrap(),
        "{{0, 1, Infinity}, {1, 0, Infinity}, {Infinity, Infinity, 0}}"
      );
    }
  }

  mod dawson_f_tests {
    use super::*;

    fn approx(code: &str, expected: f64) {
      let out = interpret(code).unwrap();
      let val: f64 = out
        .parse()
        .unwrap_or_else(|_| panic!("not a number: {out} (from {code})"));
      assert!(
        (val - expected).abs() < 1e-12,
        "{code} => {val}, expected ~{expected}"
      );
    }

    #[test]
    fn test_dawson_zero() {
      // DawsonF[0] is exact.
      assert_eq!(interpret("DawsonF[0]").unwrap(), "0");
    }

    #[test]
    fn test_dawson_infinity() {
      // DawsonF[Infinity] = 0.
      assert_eq!(interpret("DawsonF[Infinity]").unwrap(), "0");
    }

    #[test]
    fn test_dawson_symbolic() {
      assert_eq!(interpret("DawsonF[x]").unwrap(), "DawsonF[x]");
    }

    #[test]
    fn test_dawson_odd_symmetry() {
      // DawsonF[-2] stays symbolic as -DawsonF[2] (matches wolframscript).
      assert_eq!(interpret("DawsonF[-2]").unwrap(), "-DawsonF[2]");
    }

    #[test]
    fn test_dawson_numeric_one() {
      // wolframscript: N[DawsonF[1]] == 0.5380795069127684
      approx("N[DawsonF[1]]", 0.538_079_506_912_768_4);
    }

    #[test]
    fn test_dawson_numeric_two() {
      // wolframscript: N[DawsonF[2]] == 0.3013403889237921
      approx("N[DawsonF[2]]", 0.301_340_388_923_792_1);
    }

    #[test]
    fn test_dawson_numeric_three() {
      // wolframscript: N[DawsonF[3]] == 0.17827103061055843
      approx("N[DawsonF[3]]", 0.178_271_030_610_558_43);
    }

    #[test]
    fn test_dawson_numeric_half() {
      // wolframscript: DawsonF[0.5] == 0.4244363835020223
      approx("DawsonF[0.5]", 0.424_436_383_502_022_3);
    }

    #[test]
    fn test_dawson_numeric_negative_real() {
      // wolframscript: N[DawsonF[-1.5]] == -0.4282490710853987
      approx("DawsonF[-1.5]", -0.428_249_071_085_398_7);
    }

    #[test]
    fn test_dawson_numeric_large() {
      // wolframscript: N[DawsonF[10]] == 0.05025384718759881
      approx("N[DawsonF[10]]", 0.050_253_847_187_598_81);
    }

    #[test]
    fn test_dawson_listable() {
      // DawsonF threads over lists (Listable attribute).
      let out = interpret("DawsonF[{0.5, 1.0, 2.0}]").unwrap();
      assert!(out.starts_with('{') && out.ends_with('}'), "{out}");
      let inner = &out[1..out.len() - 1];
      let nums: Vec<f64> =
        inner.split(", ").map(|s| s.parse().unwrap()).collect();
      let expected = [
        0.424_436_383_502_022_3,
        0.538_079_506_912_768_4,
        0.301_340_388_923_792_1,
      ];
      assert_eq!(nums.len(), 3);
      for (a, b) in nums.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
      }
    }
  }

  mod partitions_p {
    use super::*;

    #[test]
    fn test_basic_values() {
      assert_eq!(interpret("PartitionsP[0]").unwrap(), "1");
      assert_eq!(interpret("PartitionsP[1]").unwrap(), "1");
      assert_eq!(interpret("PartitionsP[10]").unwrap(), "42");
      assert_eq!(interpret("PartitionsP[100]").unwrap(), "190569292");
    }

    #[test]
    fn test_large_value_bigint() {
      // p(200) overflows i64; result must be exact.
      assert_eq!(interpret("PartitionsP[200]").unwrap(), "3972999029388");
      assert_eq!(
        interpret("PartitionsP[1000]").unwrap(),
        "24061467864032622473692149727991"
      );
    }

    #[test]
    fn test_negative_returns_zero() {
      assert_eq!(interpret("PartitionsP[-3]").unwrap(), "0");
      assert_eq!(interpret("PartitionsP[-5]").unwrap(), "0");
    }

    #[test]
    fn test_listable() {
      // PartitionsP has the Listable attribute and threads over lists.
      assert_eq!(interpret("PartitionsP[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
      assert_eq!(interpret("PartitionsP[{4, 5, 6}]").unwrap(), "{5, 7, 11}");
      // Nested lists thread recursively.
      assert_eq!(
        interpret("PartitionsP[{{1, 2}, {3, 4}}]").unwrap(),
        "{{1, 2}, {3, 5}}"
      );
      assert_eq!(interpret("PartitionsP[{}]").unwrap(), "{}");
    }

    #[test]
    fn test_attributes() {
      assert_eq!(
        interpret("Attributes[PartitionsP]").unwrap(),
        "{Listable, Protected}"
      );
    }

    #[test]
    fn test_symbolic_and_real_unevaluated() {
      // Non-integer numeric and symbolic args stay unevaluated.
      assert_eq!(interpret("PartitionsP[n]").unwrap(), "PartitionsP[n]");
      assert_eq!(interpret("PartitionsP[3.5]").unwrap(), "PartitionsP[3.5]");
    }
  }

  mod nminimize_constrained {
    use super::*;

    /// Parse a `{value, {var -> v, ...}}` result into (value, [v, ...]).
    fn parse_result(out: &str) -> (f64, Vec<f64>) {
      // value is the first comma-separated token at depth 1.
      let inner = out.trim();
      let inner = inner.strip_prefix('{').unwrap();
      let inner = inner.strip_suffix('}').unwrap();
      // Split off the leading value before ", {".
      let split = inner.find(", {").unwrap();
      // Woxi prints scientific notation with `*^` (e.g. 1.25*^-8).
      let to_f64 =
        |s: &str| -> f64 { s.trim().replace("*^", "e").parse().unwrap() };
      let value = to_f64(&inner[..split]);
      let rules = &inner[split + 3..inner.len() - 1];
      let vals: Vec<f64> = rules
        .split(", ")
        .map(|r| to_f64(r.split("-> ").nth(1).unwrap()))
        .collect();
      (value, vals)
    }

    #[test]
    fn test_xy_product_constraint() {
      // Minimize x^2 + y^2 subject to x y >= 1 inside a box.
      // wolframscript: {2., {x -> 1., y -> 1.}} (x=y=-1 is an equally
      // valid global minimum).
      let out = interpret(
        "NMinimize[{x^2 + y^2, x y >= 1 && -3 <= x <= 3 && -3 <= y <= 3}, {x, y}]",
      )
      .unwrap();
      let (value, vals) = parse_result(&out);
      assert!((value - 2.0).abs() < 1e-4, "value {value}");
      // |x| ~ |y| ~ 1 and x*y ~ 1.
      assert!((vals[0].abs() - 1.0).abs() < 1e-3, "x {}", vals[0]);
      assert!((vals[1].abs() - 1.0).abs() < 1e-3, "y {}", vals[1]);
      assert!(vals[0] * vals[1] >= 1.0 - 1e-3, "constraint {vals:?}");
    }

    #[test]
    fn test_disk_constraint_minimize() {
      // wolframscript: {-9., {x -> 0., y -> -2.}}
      let out =
        interpret("NMinimize[{x^2 - (y - 1)^2, x^2 + y^2 <= 4}, {x, y}]")
          .unwrap();
      let (value, vals) = parse_result(&out);
      assert!((value + 9.0).abs() < 1e-4, "value {value}");
      assert!(vals[0].abs() < 1e-3, "x {}", vals[0]);
      assert!((vals[1] + 2.0).abs() < 1e-3, "y {}", vals[1]);
    }

    #[test]
    fn test_disk_constraint_maximize() {
      // wolframscript: {3.5, {x -> ±1.9365, y -> 0.5}}
      let out =
        interpret("NMaximize[{x^2 - (y - 1)^2, x^2 + y^2 <= 4}, {x, y}]")
          .unwrap();
      let (value, vals) = parse_result(&out);
      assert!((value - 3.5).abs() < 1e-3, "value {value}");
      assert!((vals[0].abs() - 1.9365).abs() < 1e-2, "x {}", vals[0]);
      assert!((vals[1] - 0.5).abs() < 1e-2, "y {}", vals[1]);
    }

    #[test]
    fn test_unconstrained_still_works() {
      // Single-variable case must keep matching wolframscript.
      let out = interpret("NMinimize[x^4 - 3*x^2 - x, x]").unwrap();
      let (value, vals) = parse_result(&out);
      assert!((value + 3.513_905).abs() < 1e-4, "value {value}");
      assert!((vals[0] - 1.300_84).abs() < 1e-3, "x {}", vals[0]);
    }

    #[test]
    fn test_exact_optimum_no_float_noise() {
      // An objective with an exact optimum (here (x-1)^2 minimized to 0 at
      // x->1) must report the clean value, not the local optimizer's
      // tolerance-level float noise. wolframscript: {0., {x -> 1.}}.
      // Regression for `{2.1*^-25, {x -> 0.9999999999995413}}`.
      assert_eq!(
        interpret("NMinimize[(x - 1)^2, x]").unwrap(),
        "{0., {x -> 1.}}"
      );
      // Same for NMaximize. wolframscript: {0., {x -> 1.}}.
      assert_eq!(
        interpret("NMaximize[-(x - 1)^2, x]").unwrap(),
        "{0., {x -> 1.}}"
      );
      // Multivariate exact optimum. wolframscript: {0., {x -> 0., y -> 0.}}.
      assert_eq!(
        interpret("NMinimize[x^2 + y^2, {x, y}]").unwrap(),
        "{0., {x -> 0., y -> 0.}}"
      );
      // A nonzero exact optimum at a shifted location.
      // wolframscript: {2., {x -> 3.}}.
      assert_eq!(
        interpret("NMinimize[(x - 3)^2 + 2, x]").unwrap(),
        "{2., {x -> 3.}}"
      );
    }

    #[test]
    fn test_equality_sphere_three_vars() {
      // Minimize x+y+z on the unit sphere x^2+y^2+z^2==1. The coupling
      // equality constraint forces the penalty optimizer; the minimum is
      // -Sqrt[3] at x=y=z=-1/Sqrt[3]. Regression for a convergence bug that
      // returned a feasible-but-suboptimal -1.685.
      let out =
        interpret("NMinimize[{x + y + z, x^2 + y^2 + z^2 == 1}, {x, y, z}]")
          .unwrap();
      let (value, vals) = parse_result(&out);
      assert!((value + 3.0_f64.sqrt()).abs() < 1e-3, "value {value}");
      for v in &vals {
        assert!((v + 1.0 / 3.0_f64.sqrt()).abs() < 1e-2, "var {v}");
      }
    }

    #[test]
    fn test_rosenbrock_box() {
      // The Rosenbrock function has a curved valley; coordinate-wise descent
      // stalls partway, so a simplex polish is needed to reach the minimum 0
      // at (1, 1). Regression for the optimizer stalling at ~3.9e-4.
      let out = interpret(
        "NMinimize[{100 (y - x^2)^2 + (1 - x)^2, -2 <= x <= 2 && -2 <= y <= 2}, {x, y}]",
      )
      .unwrap();
      let (value, vals) = parse_result(&out);
      assert!(value.abs() < 1e-6, "value {value}");
      assert!((vals[0] - 1.0).abs() < 1e-3, "x {}", vals[0]);
      assert!((vals[1] - 1.0).abs() < 1e-3, "y {}", vals[1]);
    }

    #[test]
    fn test_equality_linear_sum() {
      // wolframscript: {3., {x1 -> 1., x2 -> 1., x3 -> 1.}}
      let out = interpret(
        "NMinimize[{x1^2 + x2^2 + x3^2, x1 + x2 + x3 == 3}, {x1, x2, x3}]",
      )
      .unwrap();
      let (value, vals) = parse_result(&out);
      assert!((value - 3.0).abs() < 1e-3, "value {value}");
      for v in &vals {
        assert!((v - 1.0).abs() < 1e-2, "var {v}");
      }
    }
  }

  mod geographics_tests {
    use super::*;

    #[test]
    fn test_geographics_returns_graphics() {
      // GeoGraphics renders as an SVG image and echoes the -Graphics-
      // placeholder in CLI output, but keeps its own head (see
      // test_geographics_head_is_geographics), matching wolframscript.
      let expr = "GeoGraphics[{Red, PointSize[Large], \
                  GeoMarker[GeoPosition[{-26.2041, 28.0473}]]}]";
      assert_eq!(interpret(expr).unwrap(), "-Graphics-");
    }

    #[test]
    fn test_geographics_head_is_geographics() {
      // Unlike Plot/Graphics, wolframscript keeps GeoGraphics as its own head
      // rather than reducing it to a Graphics object.
      assert_eq!(
        interpret(
          "Head[GeoGraphics[{Red, PointSize[Large], \
           GeoMarker[GeoPosition[{-26.2041, 28.0473}]]}]]"
        )
        .unwrap(),
        "GeoGraphics"
      );
    }

    #[test]
    fn test_geomarker_alone_stays_symbolic() {
      // GeoMarker on its own is an inert primitive (no "unimplemented" warning).
      assert_eq!(
        interpret("GeoMarker[GeoPosition[{-26.2041, 28.0473}]]").unwrap(),
        "GeoMarker[GeoPosition[{-26.2041, 28.0473}]]"
      );
    }

    // ── Numeric geodesy ──────────────────────────────────────────────────
    // Geographiclib (GRS80, matching WL's ITRF00 model) agrees with WL to
    // ~12 significant figures; the displayed last few digits diverge because
    // WL uses its own geodesic. These assert Woxi's deterministic output.

    #[test]
    fn test_geodistance_kilometers() {
      assert_eq!(
        interpret("GeoDistance[{40, -100}, {34, -118}]").unwrap(),
        "Quantity[1731.0149683233299, Kilometers]"
      );
      // GeoPosition wrappers resolve identically to bare {lat, lon}.
      assert_eq!(
        interpret(
          "GeoDistance[GeoPosition[{40, -100}], GeoPosition[{34, -118}]]"
        )
        .unwrap(),
        "Quantity[1731.0149683233299, Kilometers]"
      );
      // Coincident points are exactly zero. WL reports sub-kilometer
      // distances in meters, so zero comes back as "Meters".
      assert_eq!(
        interpret("GeoDistance[{40, -100}, {40, -100}]").unwrap(),
        "Quantity[0., Meters]"
      );
      // Sub-kilometer distances stay in meters too (magnitude has
      // platform-dependent ULP noise, so assert only the chosen unit).
      assert_eq!(
        interpret("QuantityUnit[GeoDistance[{0, 0}, {0, 0.005}]]").unwrap(),
        "Meters"
      );
    }

    #[test]
    fn test_geodirection_angulardegrees() {
      // Due east along the equator is exactly 90 degrees.
      assert_eq!(
        interpret("GeoDirection[{0, 0}, {0, 10}]").unwrap(),
        "Quantity[90., AngularDegrees]"
      );
      // Round to 9 decimals: the raw 17th significant digit is platform-dependent
      // ULP noise in the geographiclib float math (e.g. ...235 vs ...232 on Linux).
      assert_eq!(
        interpret(
          "N[Round[QuantityMagnitude[GeoDirection[{40, -100}, {34, -118}]], 10^-9]]"
        )
        .unwrap(),
        "-106.938074214"
      );
    }

    #[test]
    fn test_geodestination_returns_geoposition() {
      // GeoDestination[pos, {distance_meters, azimuth_degrees}].
      assert_eq!(
        interpret("GeoDestination[{40, -100}, {100000, 45}]").unwrap(),
        "GeoPosition[{40.63380067529133, -99.1641722386888}]"
      );
    }

    #[test]
    fn test_geodestination_accepts_quantity_distance_and_azimuth() {
      // A length Quantity for the distance resolves to meters: 100 km is
      // identical to a bare 100000, and Miles convert too. An
      // "AngularDegrees" Quantity for the bearing resolves to plain degrees.
      // (wolframscript accepts all of these forms.)
      let bare = interpret("GeoDestination[{40, -100}, {100000, 45}]").unwrap();
      assert_eq!(
        interpret(
          "GeoDestination[{40, -100}, {Quantity[100, \"Kilometers\"], 45}]"
        )
        .unwrap(),
        bare
      );
      assert_eq!(
        interpret(
          "GeoDestination[{40, -100}, {Quantity[100, \"Kilometers\"], Quantity[45, \"AngularDegrees\"]}]"
        )
        .unwrap(),
        bare
      );
      // GeoPosition wrapper with a Quantity distance resolves the same way.
      assert_eq!(
        interpret(
          "GeoDestination[GeoPosition[{40, -100}], {Quantity[100, \"Kilometers\"], 45}]"
        )
        .unwrap(),
        bare
      );
      // A Miles distance must NOT match the kilometer result.
      assert_ne!(
        interpret("GeoDestination[{40, -100}, {Quantity[100, \"Miles\"], 45}]")
          .unwrap(),
        bare
      );
    }

    #[test]
    fn test_geolength_of_path() {
      assert_eq!(
        interpret("GeoLength[GeoPath[{{40, -100}, {34, -118}}]]").unwrap(),
        "Quantity[1731.0149683233299, Kilometers]"
      );
      // A bare list of positions works too; multi-segment sums each leg.
      assert_eq!(
        interpret("GeoLength[GeoPath[{{0, 0}, {0, 10}, {0, 20}}]]").unwrap(),
        interpret("GeoLength[GeoPath[{{0, 0}, {0, 20}}]]").unwrap()
      );
    }

    #[test]
    fn test_geobounds() {
      assert_eq!(
        interpret(
          "GeoBounds[{GeoPosition[{40, -100}], GeoPosition[{34, -118}]}]"
        )
        .unwrap(),
        "{{34., 40.}, {-118., -100.}}"
      );
    }

    #[test]
    fn test_geopath_stays_symbolic() {
      assert_eq!(
        interpret("GeoPath[{{1, 2}, {3, 4}}]").unwrap(),
        "GeoPath[{{1, 2}, {3, 4}}]"
      );
    }

    #[test]
    fn test_geoantipode() {
      // Latitude negated; longitude shifted 180° into (-180, 180].
      // Exact integers in -> exact integers out (matching wolframscript).
      assert_eq!(interpret("GeoAntipode[{40, -100}]").unwrap(), "{-40, 80}");
      // Longitude 0 maps to +180 (boundary stays positive).
      assert_eq!(interpret("GeoAntipode[{0, 0}]").unwrap(), "{0, 180}");
      // Eastern longitude wraps past +180 to a negative value.
      assert_eq!(interpret("GeoAntipode[{40, 100}]").unwrap(), "{-40, -80}");
      assert_eq!(interpret("GeoAntipode[{-30, 170}]").unwrap(), "{30, -10}");
      // Real coordinates round-trip as reals.
      assert_eq!(
        interpret("GeoAntipode[{40.5, -100.5}]").unwrap(),
        "{-40.5, 79.5}"
      );
      // GeoPosition wrapper is preserved.
      assert_eq!(
        interpret("GeoAntipode[GeoPosition[{40, -100}]]").unwrap(),
        "GeoPosition[{-40, 80}]"
      );
    }

    // ── Tier 1 rendering primitives ──────────────────────────────────────

    #[test]
    fn test_geo_primitives_render_graphics() {
      for prim in [
        "GeoPath[{{40, -100}, {34, -118}}]",
        "GeoPolygon[{{40, -100}, {45, -90}, {35, -95}}]",
        "GeoCircle[{40, -100}, Quantity[500, \"Kilometers\"]]",
        "GeoDisk[{40, -100}, Quantity[300, \"Kilometers\"]]",
      ] {
        assert_eq!(
          interpret(&format!("Head[GeoGraphics[{prim}]]")).unwrap(),
          "GeoGraphics",
          "{prim} should render with head GeoGraphics"
        );
      }
    }

    #[test]
    fn test_geogridlines_emits_lines() {
      let plain =
        interpret("ExportString[GeoGraphics[Point[{40, -100}]], \"SVG\"]")
          .unwrap();
      assert!(!plain.contains("<line"), "no graticule by default");
      let grid = interpret(
        "ExportString[GeoGraphics[Point[{40, -100}], \
         GeoGridLines -> Automatic], \"SVG\"]",
      )
      .unwrap();
      assert!(grid.contains("<line"), "GeoGridLines should add lines");
    }

    #[test]
    fn test_geoprojection_mercator_changes_height() {
      // The whole world is 2:1 in equirectangular but ~1:1 in web Mercator.
      let equi = interpret(
        "ExportString[GeoGraphics[Point[{0, 0}], GeoRange -> \"World\", \
         ImageSize -> 360], \"SVG\"]",
      )
      .unwrap();
      assert!(
        equi.contains("height=\"180\""),
        "equirectangular world is 2:1"
      );
      let merc = interpret(
        "ExportString[GeoGraphics[Point[{0, 0}], GeoRange -> \"World\", \
         GeoProjection -> \"Mercator\", ImageSize -> 360], \"SVG\"]",
      )
      .unwrap();
      assert!(
        !merc.contains("height=\"180\""),
        "Mercator world is taller than 2:1"
      );
    }

    #[test]
    fn test_geocircle_disk_fill_differs() {
      // GeoDisk is filled (fill-opacity); GeoCircle is an unfilled outline.
      let disk = interpret(
        "ExportString[GeoGraphics[GeoDisk[{40, -100}, \
         Quantity[300, \"Kilometers\"]]], \"SVG\"]",
      )
      .unwrap();
      assert!(disk.contains("fill-opacity"), "GeoDisk should be filled");
    }

    // ── Tier 2: named countries ──────────────────────────────────────────

    #[test]
    fn test_geonearest_containment() {
      // Points inside countries resolve to the canonical country entity.
      assert_eq!(
        interpret("GeoNearest[\"Country\", GeoPosition[{48.85, 2.35}]]")
          .unwrap(),
        "{Entity[Country, France]}"
      );
      assert_eq!(
        interpret("GeoNearest[\"Country\", GeoPosition[{40, -100}]]").unwrap(),
        "{Entity[Country, United States]}"
      );
      assert_eq!(
        interpret("GeoNearest[\"Country\", GeoPosition[{35.7, 139.7}]]")
          .unwrap(),
        "{Entity[Country, Japan]}"
      );
    }

    #[test]
    fn test_geonearest_abbreviated_name_fixup() {
      // Natural Earth's "Dem. Rep. Congo" resolves to the canonical name.
      assert_eq!(
        interpret("GeoNearest[\"Country\", GeoPosition[{-2, 23}]]").unwrap(),
        "{Entity[Country, Democratic Republic of the Congo]}"
      );
    }

    #[test]
    fn test_geographics_entity_highlight() {
      // A named country renders as a highlighted region over the basemap.
      assert_eq!(
        interpret("Head[GeoGraphics[Entity[\"Country\", \"France\"]]]")
          .unwrap(),
        "GeoGraphics"
      );
      let svg = interpret(
        "ExportString[GeoGraphics[Entity[\"Country\", \"France\"]], \"SVG\"]",
      )
      .unwrap();
      // The highlight uses a semi-transparent even-odd fill.
      assert!(
        svg.contains("fill-rule=\"evenodd\" fill-opacity=\"0.65\""),
        "country highlight fill missing"
      );
    }

    #[test]
    fn test_georegionvalueplot_renders() {
      assert_eq!(
        interpret(
          "Head[GeoRegionValuePlot[{Entity[\"Country\", \"France\"] -> 10, \
           Entity[\"Country\", \"Germany\"] -> 20}]]"
        )
        .unwrap(),
        "Graphics"
      );
    }

    #[test]
    fn test_georegionvalueplot_has_legend() {
      // The choropleth carries a color-scale legend with min/max tick labels.
      let svg = interpret(
        "ExportString[GeoRegionValuePlot[{Entity[\"Country\", \"France\"] -> 5, \
         Entity[\"Country\", \"Germany\"] -> 20}], \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("<text"), "legend tick labels missing");
      assert!(svg.contains(">20<"), "legend max label missing");
      assert!(svg.contains(">5<"), "legend min label missing");
      // The canvas is widened to hold the legend strip (default map width 360).
      assert!(
        svg.contains("width=\"438\""),
        "canvas not widened for legend"
      );
      // Map content is clipped to the map box so nothing draws behind the
      // legend strip.
      assert!(
        svg.contains("clipPath id=\"geomap\"")
          && svg.contains("clip-path=\"url(#geomap)\""),
        "map content is not clipped to the map box"
      );
    }

    #[test]
    fn test_geographics_svg_has_map_and_marker() {
      let svg = interpret(
        "ExportString[GeoGraphics[{Red, PointSize[Large], \
         GeoMarker[GeoPosition[{-26.2041, 28.0473}]]}], \"SVG\"]",
      )
      .unwrap();
      // Ocean background present.
      assert!(
        svg.contains("rgb(214,232,245)"),
        "ocean missing: {}",
        &svg[..200]
      );
      // The 1:110m countries blob contributes hundreds of land paths.
      let land_paths = svg.matches("rgb(245,242,233)").count();
      assert!(
        land_paths > 100,
        "expected many country paths, got {land_paths}"
      );
      // Red marker pin rendered.
      assert!(svg.contains("fill=\"rgb(255,0,0)\""), "red pin missing");
    }

    /// Regression: the teardrop pin's bulb arc must sweep *over the top* of the
    /// circle. A wrong sweep flag (`0 1 1`) made SVG pick the opposite arc,
    /// rendering two "horns" and a downward dart instead of a round bulb.
    #[test]
    fn test_geographics_pin_bulb_arc_over_top() {
      let svg = interpret(
        "ExportString[GeoGraphics[{Red, \
         GeoMarker[GeoPosition[{-26.2041, 28.0473}]]}], \"SVG\"]",
      )
      .unwrap();
      // Isolate the red pin path.
      let pin = svg
        .split("fill=\"rgb(255,0,0)\"")
        .next()
        .and_then(|s| s.rfind("M ").map(|i| s[i..].to_string()))
        .expect("red pin path missing");
      // Major arc (large-arc-flag 1) with sweep-flag 0 → bulb over the tip.
      assert!(
        pin.contains(" 0 1 0 "),
        "pin arc has wrong flags (expected '0 1 0'): {pin}"
      );
      // The white hole sits at the bulb center, above the tip (smaller y).
      let (_tx, ty) = red_pin_tip(&svg);
      let hole = svg
        .rsplit("<circle")
        .find(|c| c.contains("fill=\"white\""))
        .expect("white hole missing");
      let cy: f64 = hole
        .split("cy=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .and_then(|s| s.parse().ok())
        .expect("hole cy missing");
      assert!(cy < ty, "bulb center {cy} should be above tip {ty}");
    }

    /// Parse the pin tip `(x, y)` from the `M <x> <y>` that starts the red
    /// marker path.
    fn red_pin_tip(svg: &str) -> (f64, f64) {
      let pin = svg
        .split("fill=\"rgb(255,0,0)\"")
        .next()
        .and_then(|s| s.rfind("M ").map(|i| s[i..].to_string()))
        .unwrap_or_default();
      let nums: Vec<f64> = pin
        .trim_start_matches("M ")
        .split_whitespace()
        .take(2)
        .map(|t| t.parse().unwrap())
        .collect();
      (nums[0], nums[1])
    }

    #[test]
    fn test_geographics_autozoom_single_centered() {
      // A single marker auto-zooms to a square regional view with the marker
      // at the image center (default width 360 → 360x360).
      let svg = interpret(
        "ExportString[GeoGraphics[{Red, \
         GeoMarker[GeoPosition[{-26.2041, 28.0473}]]}], \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("width=\"360\""), "{}", &svg[..120]);
      assert!(svg.contains("height=\"360\""), "{}", &svg[..120]);
      let (x, y) = red_pin_tip(&svg);
      assert!((x - 180.0).abs() < 1.0, "pin x not centered: {x}");
      assert!((y - 180.0).abs() < 1.0, "pin y not centered: {y}");
    }

    #[test]
    fn test_geomarker_city_entity_resolves_via_keshvar() {
      // Entity["City", {city, region, country}] is resolved through the keshvar
      // gazetteer. It has no city coordinates, so Munich resolves to the center
      // of its administrative subdivision (Bavaria); the red pin must render.
      let svg = interpret(
        "ExportString[GeoGraphics[{Red, \
         GeoMarker[Entity[\"City\", {\"Munich\", \"Bavaria\", \"Germany\"}]]}], \
         \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("rgb(214,232,245)"), "ocean missing");
      assert!(svg.contains("fill=\"rgb(255,0,0)\""), "red pin missing");
    }

    #[test]
    fn test_geomarker_city_entity_centered_with_radius_range() {
      // The requested example: GeoRange -> Quantity[50, "Kilometers"] frames a
      // 50 km disk around the Munich marker, so the pin sits at the center of
      // the (square 360x360) image.
      let svg = interpret(
        "ExportString[GeoGraphics[{Red, \
         GeoMarker[Entity[\"City\", {\"Munich\", \"Bavaria\", \"Germany\"}]]}, \
         GeoRange -> Quantity[50, \"Kilometers\"]], \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("width=\"360\""), "{}", &svg[..120]);
      assert!(svg.contains("height=\"360\""), "{}", &svg[..120]);
      let (x, y) = red_pin_tip(&svg);
      assert!((x - 180.0).abs() < 1.0, "pin x not centered: {x}");
      assert!((y - 180.0).abs() < 1.0, "pin y not centered: {y}");
    }

    #[test]
    fn test_geomarker_country_entity_resolves_to_center() {
      // Entity["Country", name] inside a GeoMarker resolves to the country's
      // center (no subdivision given).
      let svg = interpret(
        "ExportString[GeoGraphics[{Blue, \
         GeoMarker[Entity[\"Country\", \"France\"]]}], \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("fill=\"rgb(0,0,255)\""), "blue pin missing");
    }

    #[test]
    fn test_georange_quantity_radius_smaller_than_full_country() {
      // A 50 km radius range zooms in much tighter than the auto-fit view, so
      // the longitude span (image width / sx) is well under a degree.
      let svg = interpret(
        "ExportString[GeoGraphics[{Red, GeoMarker[GeoPosition[{48.14, 11.58}]]}, \
         GeoRange -> Quantity[50, \"Kilometers\"]], \"SVG\"]",
      )
      .unwrap();
      // 50 km ~ 0.45 deg of latitude, so a 360 px image spans < 1.5 deg of
      // longitude; the Munich marker and a point 1 deg east must be far apart.
      let (x0, _) = red_pin_tip(&svg);
      assert!((x0 - 180.0).abs() < 1.0, "center pin off: {x0}");
      assert!(svg.contains("height=\"360\""), "expected square 50km view");
    }

    #[test]
    fn test_geographics_world_projection() {
      // GeoRange -> "World" forces the whole-globe equirectangular 2:1 view;
      // Johannesburg then projects to x≈208, y≈116 at 360x180.
      let svg = interpret(
        "ExportString[GeoGraphics[{Red, \
         GeoMarker[GeoPosition[{-26.2041, 28.0473}]]}, GeoRange -> \"World\"], \
         \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("width=\"360\""), "{}", &svg[..120]);
      assert!(svg.contains("height=\"180\""), "{}", &svg[..120]);
      let (x, y) = red_pin_tip(&svg);
      assert!((x - 208.0).abs() < 1.0, "pin x off: {x}");
      assert!((y - 116.2).abs() < 1.0, "pin y off: {y}");
    }

    #[test]
    fn test_geographics_autozoom_multi_fits_all() {
      // Three European markers (Paris, London, Berlin) auto-zoom to a regional
      // view; every pin tip must land inside the image bounds.
      let svg = interpret(
        "ExportString[GeoGraphics[{Red, \
         GeoMarker[GeoPosition[{48.8566, 2.3522}]], \
         GeoMarker[GeoPosition[{51.5074, -0.1278}]], \
         GeoMarker[GeoPosition[{52.52, 13.405}]]}], \"SVG\"]",
      )
      .unwrap();
      let pins = svg.matches("fill=\"rgb(255,0,0)\"").count();
      assert_eq!(pins, 3, "expected 3 pins, got {pins}");
      // Not a whole-world view (zoomed in): far-away land would be clipped.
      assert!(
        !svg.contains("height=\"180\""),
        "should not be 2:1 world view"
      );
      // Every pin tip is within the image.
      let height: f64 = svg
        .split("height=\"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .unwrap()
        .parse()
        .unwrap();
      for chunk in svg.split("<path d=\"M ").skip(1) {
        if !chunk.contains("fill=\"rgb(255,0,0)\"") {
          continue;
        }
        let nums: Vec<f64> = chunk
          .split_whitespace()
          .take(2)
          .map(|t| t.parse().unwrap())
          .collect();
        assert!(
          nums[0] >= 0.0 && nums[0] <= 360.0,
          "pin x out of bounds: {}",
          nums[0]
        );
        assert!(
          nums[1] >= 0.0 && nums[1] <= height,
          "pin y out of bounds: {}",
          nums[1]
        );
      }
    }

    #[test]
    fn test_geographics_imagesize() {
      // ImageSize sets the map width; with GeoRange -> "World" the height is
      // half the width (2:1 equirectangular).
      let svg = interpret(
        "ExportString[GeoGraphics[{GeoMarker[GeoPosition[{0, 0}]]}, \
         ImageSize -> 600, GeoRange -> \"World\"], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("width=\"600\""),
        "width not honored: {}",
        &svg[..120]
      );
      assert!(
        svg.contains("height=\"300\""),
        "height not honored: {}",
        &svg[..120]
      );
    }

    #[test]
    fn test_geographics_bare_latlon_and_point() {
      // Bare {lat, lon} positions and Point primitives also render.
      let svg = interpret(
        "ExportString[GeoGraphics[{Blue, Point[{48.8566, 2.3522}]}], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("<circle"),
        "point not rendered: missing circle"
      );
      assert!(svg.contains("fill=\"rgb(0,0,255)\""), "blue point missing");
    }

    // ── GeoHistogram ─────────────────────────────────────────────────────

    #[test]
    fn test_geohistogram_head_and_placeholder() {
      // Like GeoGraphics, GeoHistogram renders as an SVG image (echoing the
      // -Graphics- placeholder in CLI output) but keeps the GeoGraphics head.
      let data = "{GeoPosition[{40, -100}], GeoPosition[{41, -101}], \
                  GeoPosition[{34, -118}]}";
      assert_eq!(
        interpret(&format!("GeoHistogram[{data}]")).unwrap(),
        "-Graphics-"
      );
      assert_eq!(
        interpret(&format!("Head[GeoHistogram[{data}]]")).unwrap(),
        "GeoGraphics"
      );
    }

    #[test]
    fn test_geohistogram_denser_bin_is_darker() {
      // Two clusters: three near-coincident points and one lone point. The
      // dense bin gets the darkest heat color (t = 1); the lone point's bin
      // gets the t = 1/3 color. Bare {lat, lon} pairs are accepted.
      let svg = interpret(
        "ExportString[GeoHistogram[{{40, -100}, {40.2, -100.1}, \
         {40.1, -100.2}, {34, -118}}], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("fill=\"rgb(140,26,33)\" fill-opacity=\"0.7\""),
        "densest bin missing its full-scale heat color"
      );
      // heat_color(1/3) = (0.837, 0.653, 0.443) → rgb(213,167,113)
      assert!(
        svg.contains("fill=\"rgb(213,167,113)\" fill-opacity=\"0.7\""),
        "sparse bin missing its scaled heat color"
      );
      // Default bins are hexagons: every bin path has 6 vertices (M + 5 L).
      for chunk in svg.split("<path d=\"").skip(1) {
        if !chunk.contains("fill-opacity=\"0.7\"") {
          continue;
        }
        let path = chunk.split('"').next().unwrap();
        let verts = path.matches(['M', 'L']).count();
        assert_eq!(verts, 6, "hexagon bin should have 6 vertices: {path}");
      }
    }

    #[test]
    fn test_geohistogram_rectangle_and_triangle_bins() {
      // "Rectangle" bins are 4-vertex tiles, "Triangle" bins 3-vertex tiles.
      let rect = interpret(
        "ExportString[GeoHistogram[{{40, -100}, {41, -101}, {34, -118}}, \
         \"Rectangle\"], \"SVG\"]",
      )
      .unwrap();
      let tri = interpret(
        "ExportString[GeoHistogram[{{40, -100}, {41, -101}, {34, -118}}, \
         \"Triangle\"], \"SVG\"]",
      )
      .unwrap();
      for (svg, expected) in [(&rect, 4), (&tri, 3)] {
        let mut bins = 0;
        for chunk in svg.split("<path d=\"").skip(1) {
          if !chunk.contains("fill-opacity=\"0.7\"") {
            continue;
          }
          bins += 1;
          let path = chunk.split('"').next().unwrap();
          let verts = path.matches(['M', 'L']).count();
          assert_eq!(verts, expected, "bin should have {expected} vertices");
        }
        assert!(bins > 0, "no bins rendered");
      }
    }

    #[test]
    fn test_geohistogram_association_weights() {
      // <|pos -> w, …|> weights each location; the max-normalized colors put
      // the w=10 bin at full scale and the w=1 bin at t = 0.1.
      let svg = interpret(
        "ExportString[GeoHistogram[<|GeoPosition[{40, -100}] -> 10, \
         GeoPosition[{34, -118}] -> 1|>], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("fill=\"rgb(140,26,33)\""),
        "heavy bin not at full scale"
      );
      // heat_color(0.1) = (0.937, 0.847, 0.553) → rgb(239,216,141)
      assert!(
        svg.contains("fill=\"rgb(239,216,141)\""),
        "light bin not at t = 0.1"
      );
    }

    #[test]
    fn test_geohistogram_weighted_data() {
      // WeightedData[locs, wts] canonicalizes to
      // WeightedData[Automatic, {locs, wts}] before GeoHistogram sees it.
      let svg = interpret(
        "ExportString[GeoHistogram[WeightedData[{{40, -100}, {34, -118}}, \
         {5, 1}]], \"SVG\"]",
      )
      .unwrap();
      assert!(
        svg.contains("fill=\"rgb(140,26,33)\""),
        "weight-5 bin not at full scale"
      );
      // heat_color(0.2) = (0.894, 0.764, 0.506) → rgb(228,195,129)
      assert!(
        svg.contains("fill=\"rgb(228,195,129)\""),
        "weight-1 bin not at t = 0.2"
      );
    }

    #[test]
    fn test_geohistogram_bin_count_spec() {
      // A numeric bspec sets the tile count across the data: more bins across
      // means smaller tiles, so 20 bins yield at least as many non-empty
      // tiles as 2 bins for spread-out data.
      let data = "{{40, -100}, {42, -104}, {44, -108}, {46, -112}, \
                  {48, -116}, {50, -120}}";
      let count_bins = |svg: &str| svg.matches("fill-opacity=\"0.7\"").count();
      let coarse =
        interpret(&format!("ExportString[GeoHistogram[{data}, 2], \"SVG\"]"))
          .unwrap();
      let fine =
        interpret(&format!("ExportString[GeoHistogram[{data}, 20], \"SVG\"]"))
          .unwrap();
      assert!(
        count_bins(&fine) > count_bins(&coarse),
        "20 bins across should produce more tiles than 2 ({} vs {})",
        count_bins(&fine),
        count_bins(&coarse)
      );
    }

    #[test]
    fn test_geohistogram_quantity_bin_size() {
      // Quantity bin diameters are honored: 50 km tiles separate two points
      // ~110 km apart into two bins.
      let svg = interpret(
        "ExportString[GeoHistogram[{{40, -100}, {41, -100}}, \
         Quantity[50, \"Kilometers\"]], \"SVG\"]",
      )
      .unwrap();
      assert_eq!(
        svg.matches("fill-opacity=\"0.7\"").count(),
        2,
        "50 km tiles should separate points 1° of latitude apart"
      );
    }

    #[test]
    fn test_geohistogram_plot_legends() {
      // No legend by default; PlotLegends -> Automatic adds the color-scale
      // bar (widening the canvas) with the max count as its top tick label.
      let plain = interpret(
        "ExportString[GeoHistogram[{{40, -100}, {40.1, -100.1}, \
         {34, -118}}], \"SVG\"]",
      )
      .unwrap();
      assert!(!plain.contains("<text"), "no legend expected by default");
      let legended = interpret(
        "ExportString[GeoHistogram[{{40, -100}, {40.1, -100.1}, {34, -118}}, \
         PlotLegends -> Automatic], \"SVG\"]",
      )
      .unwrap();
      assert!(legended.contains("<text"), "legend tick labels missing");
      assert!(legended.contains(">2<"), "legend max label missing");
      assert!(legended.contains(">0<"), "legend min label missing");
    }

    #[test]
    fn test_geohistogram_height_specs_accepted() {
      // The hspec argument ("Count"/"Probability"/"Intensity"/"PDF") is
      // accepted; with equal-size bins and max-normalized colors, all four
      // shade the map identically.
      let base = interpret(
        "ExportString[GeoHistogram[{{40, -100}, {41, -101}}, Automatic, \
         \"Count\"], \"SVG\"]",
      )
      .unwrap();
      for hspec in ["Probability", "Intensity", "PDF"] {
        let svg = interpret(&format!(
          "ExportString[GeoHistogram[{{{{40, -100}}, {{41, -101}}}}, \
           Automatic, \"{hspec}\"], \"SVG\"]"
        ))
        .unwrap();
        assert_eq!(svg, base, "{hspec} should shade like Count");
      }
    }

    #[test]
    fn test_geohistogram_options_passthrough() {
      // GeoGraphics options (ImageSize, GeoRange, GeoProjection, …) pass
      // through to the underlying map.
      let svg = interpret(
        "ExportString[GeoHistogram[{{40, -100}, {41, -101}}, \
         ImageSize -> 500, GeoRange -> \"World\"], \"SVG\"]",
      )
      .unwrap();
      assert!(svg.contains("width=\"500\""), "ImageSize not honored");
      assert!(
        svg.contains("height=\"250\""),
        "world view should be 2:1 at width 500"
      );
    }

    #[test]
    fn test_geohistogram_single_point_renders() {
      // A single location still produces a (one-bin) map instead of a
      // degenerate view.
      let svg = interpret(
        "ExportString[GeoHistogram[{GeoPosition[{48.8566, 2.3522}]}], \
         \"SVG\"]",
      )
      .unwrap();
      assert_eq!(
        svg.matches("fill-opacity=\"0.7\"").count(),
        1,
        "expected exactly one bin"
      );
    }

    #[test]
    fn test_geohistogram_invalid_data_stays_symbolic() {
      // Non-positional data leaves the expression unevaluated, matching the
      // GeoRegionValuePlot behavior for unusable input.
      assert_eq!(interpret("GeoHistogram[5]").unwrap(), "GeoHistogram[5]");
    }
  }

  mod dihedral_angle_tests {
    use super::*;

    #[test]
    fn test_orthogonal_half_planes() {
      // Edge along z; directions along x and y are perpendicular.
      assert_eq!(
        interpret("DihedralAngle[{{0,0,0},{0,0,1}},{{1,0,0},{0,1,0}}]")
          .unwrap(),
        "Pi/2"
      );
    }

    #[test]
    fn test_pi_over_four() {
      assert_eq!(
        interpret("DihedralAngle[{{0,0,0},{1,0,0}},{{0,1,0},{0,1,1}}]")
          .unwrap(),
        "Pi/4"
      );
    }

    #[test]
    fn test_exact_arccos_and_nonzero_base_point() {
      // Edge not through the origin; directions yield an exact ArcCos form.
      assert_eq!(
        interpret("DihedralAngle[{{1,1,0},{1,1,2}},{{2,1,0},{1,2,1}}]")
          .unwrap(),
        "ArcCos[4/5]"
      );
    }

    #[test]
    fn test_opposite_directions_give_pi() {
      assert_eq!(
        interpret("DihedralAngle[{{0,0,0},{1,0,0}},{{0,1,0},{0,-1,0}}]")
          .unwrap(),
        "Pi"
      );
    }

    #[test]
    fn test_numeric() {
      assert_eq!(
        interpret("N[DihedralAngle[{{1,1,0},{1,1,2}},{{2,1,0},{1,2,1}}]]")
          .unwrap(),
        "0.6435011087932843"
      );
    }

    #[test]
    fn test_2d_unevaluated() {
      // DihedralAngle is only defined for 3D vectors in Wolfram.
      assert_eq!(
        interpret("DihedralAngle[{{0,0},{1,0}},{{0,1},{1,1}}]").unwrap(),
        "DihedralAngle[{{0, 0}, {1, 0}}, {{0, 1}, {1, 1}}]"
      );
    }

    #[test]
    fn test_4d_unevaluated() {
      assert_eq!(
        interpret("DihedralAngle[{{0,0,0,0},{1,0,0,0}},{{0,1,0,0},{0,0,1,0}}]")
          .unwrap(),
        "DihedralAngle[{{0, 0, 0, 0}, {1, 0, 0, 0}}, \
         {{0, 1, 0, 0}, {0, 0, 1, 0}}]"
      );
    }
  }

  mod mean_degree_connectivity_tests {
    use super::*;

    #[test]
    fn test_regular_cycle() {
      // 2-regular: only degree-2 vertices, all neighbors degree 2.
      assert_eq!(
        interpret("MeanDegreeConnectivity[CycleGraph[5]]").unwrap(),
        "{0, 0, 2}"
      );
    }

    #[test]
    fn test_path_mixed_degrees() {
      // Endpoints (deg 1) neighbor a deg-2 vertex; interior (deg 2)
      // neighbors average to 3/2.
      assert_eq!(
        interpret("MeanDegreeConnectivity[PathGraph[Range[4]]]").unwrap(),
        "{0, 2, 3/2}"
      );
    }

    #[test]
    fn test_star() {
      assert_eq!(
        interpret("MeanDegreeConnectivity[StarGraph[4]]").unwrap(),
        "{0, 3, 0, 1}"
      );
    }

    #[test]
    fn test_complete() {
      assert_eq!(
        interpret("MeanDegreeConnectivity[CompleteGraph[4]]").unwrap(),
        "{0, 0, 0, 3}"
      );
    }

    #[test]
    fn test_edgeless() {
      assert_eq!(
        interpret("MeanDegreeConnectivity[Graph[{1,2,3},{}]]").unwrap(),
        "{0}"
      );
    }

    #[test]
    fn test_isolated_vertex_present() {
      assert_eq!(
        interpret("MeanDegreeConnectivity[Graph[{1,2,3},{1<->2}]]").unwrap(),
        "{0, 1}"
      );
    }
  }

  mod geometric_transformation_tests {
    use super::*;

    #[test]
    fn test_rotation_function_to_affine_pair() {
      // A TransformationFunction normalizes to {linearMatrix, translation}.
      assert_eq!(
        interpret(
          "GeometricTransformation[Point[{1,1}],RotationTransform[Pi]]"
        )
        .unwrap(),
        "GeometricTransformation[Point[{1, 1}], {{{-1, 0}, {0, -1}}, {0, 0}}]"
      );
    }

    #[test]
    fn test_translation_function() {
      assert_eq!(
        interpret(
          "GeometricTransformation[Point[{1,1}],TranslationTransform[{3,4}]]"
        )
        .unwrap(),
        "GeometricTransformation[Point[{1, 1}], {{{1, 0}, {0, 1}}, {3, 4}}]"
      );
    }

    #[test]
    fn test_scaling_function() {
      assert_eq!(
        interpret(
          "GeometricTransformation[Point[{1,1}],ScalingTransform[{2,3}]]"
        )
        .unwrap(),
        "GeometricTransformation[Point[{1, 1}], {{{2, 0}, {0, 3}}, {0, 0}}]"
      );
    }

    #[test]
    fn test_affine_function() {
      assert_eq!(
        interpret(
          "GeometricTransformation[{Point[{1,1}]},AffineTransform[{{1,2},{3,4}}]]"
        )
        .unwrap(),
        "GeometricTransformation[{Point[{1, 1}]}, {{{1, 2}, {3, 4}}, {0, 0}}]"
      );
    }

    #[test]
    fn test_plain_matrix_unchanged() {
      // A bare matrix is not a TransformationFunction and is left as-is.
      assert_eq!(
        interpret("GeometricTransformation[Point[{1,1}],{{2,0},{0,2}}]")
          .unwrap(),
        "GeometricTransformation[Point[{1, 1}], {{2, 0}, {0, 2}}]"
      );
    }

    #[test]
    fn test_explicit_affine_pair_unchanged() {
      assert_eq!(
        interpret(
          "GeometricTransformation[Point[{1,1}],{{{2,0},{0,2}},{5,6}}]"
        )
        .unwrap(),
        "GeometricTransformation[Point[{1, 1}], {{{2, 0}, {0, 2}}, {5, 6}}]"
      );
    }
  }

  mod rotation_transform_3d_tests {
    use super::*;

    #[test]
    fn test_z_axis_quarter_turn() {
      assert_eq!(
        interpret("TransformationMatrix[RotationTransform[Pi/2,{0,0,1}]]")
          .unwrap(),
        "{{0, -1, 0, 0}, {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}"
      );
    }

    #[test]
    fn test_x_axis_quarter_turn() {
      assert_eq!(
        interpret("TransformationMatrix[RotationTransform[Pi/2,{1,0,0}]]")
          .unwrap(),
        "{{1, 0, 0, 0}, {0, 0, -1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}"
      );
    }

    #[test]
    fn test_symbolic_angle() {
      assert_eq!(
        interpret("TransformationMatrix[RotationTransform[t,{0,0,1}]]")
          .unwrap(),
        "{{Cos[t], -Sin[t], 0, 0}, {Sin[t], Cos[t], 0, 0}, \
         {0, 0, 1, 0}, {0, 0, 0, 1}}"
      );
    }

    #[test]
    fn test_axis_normalized_when_not_unit() {
      // {0,0,2} normalizes to the same rotation as {0,0,1}.
      assert_eq!(
        interpret("TransformationMatrix[RotationTransform[Pi/2,{0,0,2}]]")
          .unwrap(),
        "{{0, -1, 0, 0}, {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}"
      );
    }

    #[test]
    fn test_diagonal_axis_radical_form() {
      // Rotation about {1,1,1} yields the exact (1 ± Sqrt[3])/3 entries.
      assert_eq!(
        interpret("TransformationMatrix[RotationTransform[Pi/2,{1,1,1}]]")
          .unwrap(),
        "{{1/3, (1 - Sqrt[3])/3, (1 + Sqrt[3])/3, 0}, \
         {(1 + Sqrt[3])/3, 1/3, (1 - Sqrt[3])/3, 0}, \
         {(1 - Sqrt[3])/3, (1 + Sqrt[3])/3, 1/3, 0}, {0, 0, 0, 1}}"
      );
    }

    #[test]
    fn test_apply_to_vector() {
      assert_eq!(
        interpret("RotationTransform[Pi/2,{0,0,1}][{1,0,0}]").unwrap(),
        "{0, 1, 0}"
      );
    }

    #[test]
    fn test_apply_to_vector_radical() {
      assert_eq!(
        interpret("RotationTransform[Pi/2,{1,1,1}][{1,0,0}]").unwrap(),
        "{1/3, (1 + Sqrt[3])/3, (1 - Sqrt[3])/3}"
      );
    }

    #[test]
    fn test_symbolic_axis_unevaluated() {
      // A symbolic axis is intractable in wolframscript; leave unevaluated.
      assert_eq!(
        interpret("RotationTransform[Pi/2,{a,b,c}]").unwrap(),
        "RotationTransform[Pi/2, {a, b, c}]"
      );
    }

    #[test]
    fn test_geometric_transformation_3d_pair() {
      // The TransformationFunction normalizes to the {matrix, translation}
      // affine pair (combines with GeometricTransformation handling).
      assert_eq!(
        interpret(
          "GeometricTransformation[Point[{1,1,1}],RotationTransform[Pi,{0,0,1}]]"
        )
        .unwrap(),
        "GeometricTransformation[Point[{1, 1, 1}], \
         {{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}}, {0, 0, 0}}]"
      );
    }
  }

  mod rotation_matrix_3d_tests {
    use super::*;

    #[test]
    fn test_diagonal_axis_normalized() {
      // Regression: the axis must be normalized. The un-normalized {1,1,1}
      // would (wrongly) give {{1,0,2},{2,1,0},{0,2,1}}.
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{1,1,1}]").unwrap(),
        "{{1/3, (1 - Sqrt[3])/3, (1 + Sqrt[3])/3}, \
         {(1 + Sqrt[3])/3, 1/3, (1 - Sqrt[3])/3}, \
         {(1 - Sqrt[3])/3, (1 + Sqrt[3])/3, 1/3}}"
      );
    }

    #[test]
    fn test_unit_z_axis() {
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{0,0,1}]").unwrap(),
        "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn test_nonunit_z_axis_normalized() {
      // {0,0,2} must give the same matrix as the unit z axis.
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{0,0,2}]").unwrap(),
        "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn test_nonunit_x_axis_normalized() {
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{2,0,0}]").unwrap(),
        "{{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}"
      );
    }

    #[test]
    fn test_symbolic_angle() {
      assert_eq!(
        interpret("RotationMatrix[t,{0,0,1}]").unwrap(),
        "{{Cos[t], -Sin[t], 0}, {Sin[t], Cos[t], 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn test_apply_to_vector() {
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{1,1,1}].{1,0,0}").unwrap(),
        "{1/3, (1 + Sqrt[3])/3, (1 - Sqrt[3])/3}"
      );
    }

    #[test]
    fn test_symbolic_axis_unevaluated() {
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{a,b,c}]").unwrap(),
        "RotationMatrix[Pi/2, {a, b, c}]"
      );
    }

    #[test]
    fn test_2d_unchanged() {
      // The 2D single-argument form is unaffected.
      assert_eq!(
        interpret("RotationMatrix[Pi/3]").unwrap(),
        "{{1/2, -1/2*Sqrt[3]}, {Sqrt[3]/2, 1/2}}"
      );
    }
  }

  mod rotation_matrix_plane_tests {
    use super::*;

    #[test]
    fn test_two_vector_basis() {
      // RotationMatrix[{u, v}] rotates the direction of u to v.
      assert_eq!(
        interpret("RotationMatrix[{{1,0,0},{0,1,0}}]").unwrap(),
        "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"
      );
      assert_eq!(
        interpret("RotationMatrix[{{1,0,0},{0,0,1}}]").unwrap(),
        "{{0, 0, -1}, {0, 1, 0}, {1, 0, 0}}"
      );
    }

    #[test]
    fn test_two_vector_rational() {
      // Clean rational result (orthogonal vectors of equal norm).
      assert_eq!(
        interpret("RotationMatrix[{{1,2,2},{2,-2,1}}]").unwrap(),
        "{{4/9, 8/9, -1/9}, {-4/9, 1/9, -8/9}, {-7/9, 4/9, 4/9}}"
      );
    }

    #[test]
    fn test_two_vector_radical() {
      assert_eq!(
        interpret("RotationMatrix[{{1,0,0},{1,1,0}}]").unwrap(),
        "{{1/Sqrt[2], -(1/Sqrt[2]), 0}, {1/Sqrt[2], 1/Sqrt[2], 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn test_plane_form_orthonormal() {
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{{1,0,0},{0,1,0}}]").unwrap(),
        "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn test_plane_form_radical_angle() {
      assert_eq!(
        interpret("RotationMatrix[Pi/3,{{1,0,0},{0,1,0}}]").unwrap(),
        "{{1/2, -1/2*Sqrt[3], 0}, {Sqrt[3]/2, 1/2, 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn test_plane_form_orthonormalizes() {
      // The second vector is orthogonalized against the first, so {1,1,0}
      // gives the same plane as {0,1,0}.
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{{1,0,0},{1,1,0}}]").unwrap(),
        "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn test_plane_form_non_axis_aligned() {
      assert_eq!(
        interpret("RotationMatrix[Pi/2,{{0,0,1},{0,1,0}}]").unwrap(),
        "{{1, 0, 0}, {0, 0, 1}, {0, -1, 0}}"
      );
    }

    #[test]
    fn test_plane_form_symbolic_angle() {
      assert_eq!(
        interpret("RotationMatrix[t,{{1,0,0},{0,1,0}}]").unwrap(),
        "{{Cos[t], -Sin[t], 0}, {Sin[t], Cos[t], 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn test_2d_two_vector_unchanged() {
      assert_eq!(
        interpret("RotationMatrix[{{1,0},{0,1}}]").unwrap(),
        "{{0, -1}, {1, 0}}"
      );
    }
  }

  mod integer_name_words_tests {
    use super::*;

    #[test]
    fn test_words_spells_out_large_numbers() {
      // The "Words" form spells every group (vs. the default's digit groups
      // for n >= 1000), joining non-zero groups with ", ".
      assert_eq!(
        interpret("IntegerName[1234567, \"Words\"]").unwrap(),
        "one million, two hundred thirty\u{2010}four thousand, \
         five hundred sixty\u{2010}seven"
      );
      assert_eq!(
        interpret("IntegerName[1234, \"Words\"]").unwrap(),
        "one thousand, two hundred thirty\u{2010}four"
      );
      assert_eq!(
        interpret("IntegerName[1000, \"Words\"]").unwrap(),
        "one thousand"
      );
      assert_eq!(
        interpret("IntegerName[2000000, \"Words\"]").unwrap(),
        "two million"
      );
      assert_eq!(
        interpret("IntegerName[1000000000, \"Words\"]").unwrap(),
        "one billion"
      );
      // Zero groups are skipped, but the comma still separates the rest.
      assert_eq!(
        interpret("IntegerName[1000005, \"Words\"]").unwrap(),
        "one million, five"
      );
      assert_eq!(
        interpret("IntegerName[-1234, \"Words\"]").unwrap(),
        "negative one thousand, two hundred thirty\u{2010}four"
      );
    }

    #[test]
    fn test_words_small_numbers_and_other_forms_unchanged() {
      // Small numbers match the default; "Words" of them is identical.
      assert_eq!(
        interpret("IntegerName[42, \"Words\"]").unwrap(),
        "forty\u{2010}two"
      );
      assert_eq!(interpret("IntegerName[0, \"Words\"]").unwrap(), "zero");
      // The default form keeps digit groups for large numbers.
      assert_eq!(
        interpret("IntegerName[1234567]").unwrap(),
        "1 million 234 thousand 567"
      );
      // Ordinal form is unaffected.
      assert_eq!(
        interpret("IntegerName[123, \"Ordinal\"]").unwrap(),
        "one hundred twenty-third"
      );
    }

    #[test]
    fn test_list_qualifier_selects_form() {
      // The qualifier may be a list {language, form} or {form}.
      assert_eq!(
        interpret("IntegerName[15, {\"English\", \"Ordinal\"}]").unwrap(),
        "fifteenth"
      );
      assert_eq!(
        interpret("IntegerName[42, {\"Ordinal\"}]").unwrap(),
        "forty-second"
      );
      assert_eq!(
        interpret("IntegerName[1234, {\"English\", \"Words\"}]").unwrap(),
        "one thousand, two hundred thirty\u{2010}four"
      );
      // A bare language list keeps the default cardinal name.
      assert_eq!(
        interpret("IntegerName[15, {\"English\"}]").unwrap(),
        "fifteen"
      );
      // German cardinal and ordinal names (morphemes joined by U+00AD).
      assert_eq!(
        interpret("IntegerName[15, {\"German\", \"Ordinal\"}]").unwrap(),
        "fünfzehnte"
      );
      assert_eq!(
        interpret("IntegerName[21, \"German\"]").unwrap(),
        "ein\u{00AD}und\u{00AD}zwanzig"
      );
      assert_eq!(
        interpret("IntegerName[1234, {\"German\", \"Ordinal\"}]").unwrap(),
        "ein\u{00AD}tausend\u{00AD}zwei\u{00AD}hundert\u{00AD}vier\u{00AD}und\u{00AD}dreißigste"
      );
      assert_eq!(
        interpret("IntegerName[-5, {\"German\", \"Ordinal\"}]").unwrap(),
        "minus fünfte"
      );
      // German long-scale names (Million/Milliarde/Billion/Billiarde). The
      // multiplier "1" agrees with the feminine noun as "eine"; the noun is
      // singular only for a multiplier of exactly 1.
      assert_eq!(
        interpret("IntegerName[1000000, \"German\"]").unwrap(),
        "eine Million"
      );
      assert_eq!(
        interpret("IntegerName[2000000, \"German\"]").unwrap(),
        "zwei Millionen"
      );
      assert_eq!(
        interpret("IntegerName[101000000, \"German\"]").unwrap(),
        "ein\u{00AD}hundert\u{00AD}eine Millionen"
      );
      assert_eq!(
        interpret("IntegerName[1234567, \"German\"]").unwrap(),
        "eine Million zwei\u{00AD}hundert\u{00AD}vier\u{00AD}und\u{00AD}dreißig\u{00AD}\
         tausend\u{00AD}fünf\u{00AD}hundert\u{00AD}sieben\u{00AD}und\u{00AD}sechzig"
      );
      assert_eq!(
        interpret("IntegerName[1000000000, \"German\"]").unwrap(),
        "eine Milliarde"
      );
      assert_eq!(
        interpret("IntegerName[2002000000, \"German\"]").unwrap(),
        "zwei Milliarden zwei Millionen"
      );
      assert_eq!(
        interpret("IntegerName[1000000000000, \"German\"]").unwrap(),
        "eine Billion"
      );
      assert_eq!(
        interpret("IntegerName[1000000000000000, \"German\"]").unwrap(),
        "eine Billiarde"
      );
      assert_eq!(
        interpret("IntegerName[-1000000, \"German\"]").unwrap(),
        "minus eine Million"
      );
      // Ordinal long-scale: the "-ste" suffix attaches to the last morpheme.
      assert_eq!(
        interpret("IntegerName[1000000, {\"German\", \"Ordinal\"}]").unwrap(),
        "eine Millionste"
      );
      assert_eq!(
        interpret("IntegerName[2000000, {\"German\", \"Ordinal\"}]").unwrap(),
        "zwei Millionenste"
      );
      // At or beyond 10^18 wolframscript cannot spell German; stays unevaluated.
      assert_eq!(
        interpret("IntegerName[1000000000000000000, \"German\"]").unwrap(),
        "IntegerName[1000000000000000000, German]"
      );
    }
  }

  mod geometric_test_tests {
    use super::*;

    // --- Point relations --------------------------------------------------

    #[test]
    fn collinear_points() {
      assert_eq!(
        interpret("GeometricTest[{{2, 3}, {4, 6}, {-2, -3}}, \"Collinear\"]")
          .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "GeometricTest[{{0, 0}, {1, 0}, {2, 0}, {3, 0}}, \"Collinear\"]"
        )
        .unwrap(),
        "True"
      );
    }

    #[test]
    fn non_collinear_points() {
      assert_eq!(
        interpret("GeometricTest[{{0, 0}, {1, 1}, {2, 3}}, \"Collinear\"]")
          .unwrap(),
        "False"
      );
    }

    #[test]
    fn distinct_points() {
      assert_eq!(
        interpret("GeometricTest[{{0, 0}, {1, 1}, {2, 3}}, \"Distinct\"]")
          .unwrap(),
        "True"
      );
      assert_eq!(
        interpret("GeometricTest[{{0, 0}, {1, 1}, {0, 0}}, \"Distinct\"]")
          .unwrap(),
        "False"
      );
    }

    // --- Line relations ---------------------------------------------------

    #[test]
    fn parallel_lines() {
      assert_eq!(
        interpret(
          "GeometricTest[{InfiniteLine[{{0, 0}, {1, 1}}], \
           InfiniteLine[{{0, 1}, {1, 2}}]}, \"Parallel\"]"
        )
        .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "GeometricTest[{InfiniteLine[{{0, 0}, {1, 1}}], \
           InfiniteLine[{{0, 0}, {1, -1}}]}, \"Parallel\"]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn perpendicular_lines() {
      assert_eq!(
        interpret(
          "GeometricTest[{InfiniteLine[{{0, 0}, {1, 1}}], \
           InfiniteLine[{{0, 0}, {1, -1}}]}, \"Perpendicular\"]"
        )
        .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "GeometricTest[{InfiniteLine[{{0, 0}, {1, 0}}], \
           InfiniteLine[{{0, 0}, {1, 1}}]}, \"Perpendicular\"]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn concurrent_lines() {
      // Three lines through the common point (1, 1).
      assert_eq!(
        interpret(
          "GeometricTest[{InfiniteLine[{{0, 0}, {1, 1}}], \
           InfiniteLine[{{0, 2}, {1, 1}}], \
           InfiniteLine[{{2, 0}, {1, 1}}]}, \"Concurrent\"]"
        )
        .unwrap(),
        "True"
      );
      // Two parallel lines are never concurrent.
      assert_eq!(
        interpret(
          "GeometricTest[{InfiniteLine[{{0, 0}, {1, 0}}], \
           InfiniteLine[{{0, 1}, {1, 1}}]}, \"Concurrent\"]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn horizontal_and_vertical_lines() {
      assert_eq!(
        interpret(
          "GeometricTest[InfiniteLine[{{0, 0}, {1, 0}}], \"Horizontal\"]"
        )
        .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "GeometricTest[InfiniteLine[{{0, 0}, {1, 0}}], \"Vertical\"]"
        )
        .unwrap(),
        "False"
      );
      assert_eq!(
        interpret(
          "GeometricTest[InfiniteLine[{{0, 0}, {0, 1}}], \"Vertical\"]"
        )
        .unwrap(),
        "True"
      );
    }

    // --- Polygon predicates ----------------------------------------------

    #[test]
    fn convex_polygon() {
      assert_eq!(
        interpret(
          "GeometricTest[Polygon[{{0, 0}, {5, 1}, {4, 4}, {-2, 0}}], \"Convex\"]"
        )
        .unwrap(),
        "True"
      );
      // Self-intersecting vertex order is not convex.
      assert_eq!(
        interpret(
          "GeometricTest[Polygon[{{0, 0}, {2, 0}, {0, 2}, {2, 2}}], \"Convex\"]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn equilateral_and_regular_triangle() {
      let eq = "Triangle[{{0, 0}, {2, 0}, {1, 1.7320508075688772}}]";
      assert_eq!(
        interpret(&format!("GeometricTest[{eq}, \"Equilateral\"]")).unwrap(),
        "True"
      );
      assert_eq!(
        interpret(&format!("GeometricTest[{eq}, \"Regular\"]")).unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "GeometricTest[Triangle[{{0, 0}, {1, 0}, {0, 1}}], \"Equilateral\"]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn square_is_regular_rectangle() {
      let sq = "Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}]";
      assert_eq!(
        interpret(&format!("GeometricTest[{sq}, \"Regular\"]")).unwrap(),
        "True"
      );
      assert_eq!(
        interpret(&format!("GeometricTest[{sq}, \"Rectangle\"]")).unwrap(),
        "True"
      );
      assert_eq!(
        interpret(&format!("GeometricTest[{sq}, \"Equiangular\"]")).unwrap(),
        "True"
      );
    }

    #[test]
    fn parallelogram_and_rectangle() {
      assert_eq!(
        interpret(
          "GeometricTest[Polygon[{{0, 0}, {3, 0}, {4, 2}, {1, 2}}], \
           \"Parallelogram\"]"
        )
        .unwrap(),
        "True"
      );
      // A slanted parallelogram is not a rectangle.
      assert_eq!(
        interpret(
          "GeometricTest[Polygon[{{0, 0}, {3, 0}, {4, 2}, {1, 2}}], \
           \"Rectangle\"]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn orientation() {
      assert_eq!(
        interpret(
          "GeometricTest[Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}], \
           \"Counterclockwise\"]"
        )
        .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "GeometricTest[Polygon[{{0, 0}, {0, 2}, {2, 2}, {2, 0}}], \
           \"Clockwise\"]"
        )
        .unwrap(),
        "True"
      );
    }

    #[test]
    fn simple_polygon() {
      assert_eq!(
        interpret(
          "GeometricTest[Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}], \"Simple\"]"
        )
        .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "GeometricTest[Polygon[{{0, 0}, {2, 2}, {2, 0}, {0, 2}}], \"Simple\"]"
        )
        .unwrap(),
        "False"
      );
    }

    // --- Object relations -------------------------------------------------

    #[test]
    fn congruent_triangles() {
      assert_eq!(
        interpret(
          "GeometricTest[{Triangle[{{0, 0}, {3, 0}, {0, 4}}], \
           Triangle[{{1, 1}, {1, 5}, {4, 1}}]}, \"Congruent\"]"
        )
        .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "GeometricTest[{Triangle[{{0, 0}, {3, 0}, {0, 4}}], \
           Triangle[{{0, 0}, {6, 0}, {0, 8}}]}, \"Congruent\"]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn similar_triangles() {
      assert_eq!(
        interpret(
          "GeometricTest[{Triangle[{{0, 0}, {3, 0}, {0, 4}}], \
           Triangle[{{0, 0}, {6, 0}, {0, 8}}]}, \"Similar\"]"
        )
        .unwrap(),
        "True"
      );
    }

    // --- Multiple properties and fallbacks --------------------------------

    #[test]
    fn multiple_properties_are_anded() {
      let sq = "Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}]";
      assert_eq!(
        interpret(&format!("GeometricTest[{sq}, \"Convex\", \"Rectangle\"]"))
          .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(&format!(
          "GeometricTest[{sq}, \"Convex\", \"Equilateral\", \"Rectangle\"]"
        ))
        .unwrap(),
        "True"
      );
    }

    #[test]
    fn symbolic_input_stays_unevaluated() {
      // Symbolic coordinates yield algebraic conditions in Wolfram; here the
      // call is simply left unevaluated.
      assert_eq!(
        interpret("GeometricTest[{{a, b}, {c, d}, {e, f}}, \"Collinear\"]")
          .unwrap(),
        "GeometricTest[{{a, b}, {c, d}, {e, f}}, Collinear]"
      );
    }

    #[test]
    fn unknown_property_stays_unevaluated() {
      assert_eq!(
        interpret("GeometricTest[{{0, 0}, {1, 1}}, \"Bogus\"]").unwrap(),
        "GeometricTest[{{0, 0}, {1, 1}}, Bogus]"
      );
    }
  }

  // ParallelDo has no real parallel kernels in Woxi, so it iterates
  // sequentially exactly like Do (matching the rest of the Parallel*
  // family). It returns Null and holds its body, so side-effecting bodies
  // like Print[i] are not evaluated once before iteration begins.
  //
  // Note: in wolframscript the parallel kernels run in isolated memory, so
  // master-kernel side effects are NOT observable and Print output is
  // reordered with per-kernel headers. Only the return value (Null) and the
  // attributes match verbatim; the stdout-capture tests below lock in
  // Woxi's deterministic sequential behavior.
  mod parallel_do_tests {
    use super::*;
    use woxi::interpret_with_stdout;

    #[test]
    fn returns_null() {
      // `interpret` renders the Null return value as the "\0" sentinel.
      assert_eq!(interpret("ParallelDo[i, {i, 3}]").unwrap(), "\0");
    }

    #[test]
    fn returns_null_in_list() {
      // Wraps the Null so it is observable in the result (matches WS).
      assert_eq!(
        interpret("{ParallelDo[i, {i, 3}], 42}").unwrap(),
        "{Null, 42}"
      );
    }

    #[test]
    fn attributes_match_wolframscript() {
      assert_eq!(
        interpret("Attributes[ParallelDo]").unwrap(),
        "{Protected, ReadProtected}"
      );
    }

    #[test]
    fn holds_body_no_premature_evaluation() {
      // Before the fix, Print[i] was evaluated once eagerly (printing the
      // bare symbol `i`) before ParallelDo saw its held argument.
      let r = interpret_with_stdout("ParallelDo[Print[i], {i, 3}]").unwrap();
      assert_eq!(r.result, "\0");
      assert_eq!(r.stdout, "1\n2\n3\n");
    }

    #[test]
    fn iterates_with_imin_imax_step() {
      let r =
        interpret_with_stdout("ParallelDo[Print[i], {i, 2, 6, 2}]").unwrap();
      assert_eq!(r.stdout, "2\n4\n6\n");
    }

    #[test]
    fn iterates_over_explicit_list() {
      let r =
        interpret_with_stdout("ParallelDo[Print[k], {k, {a, b, c}}]").unwrap();
      assert_eq!(r.stdout, "a\nb\nc\n");
    }

    #[test]
    fn multiple_iterators() {
      let r =
        interpret_with_stdout("ParallelDo[Print[10 i + j], {i, 2}, {j, 2}]")
          .unwrap();
      assert_eq!(r.stdout, "11\n12\n21\n22\n");
    }
  }

  // TimeValue[Annuity[pmt, tspan, q], i, t]: a level annuity paying pmt at the
  // end of every interval q over a total time span tspan, valued with interest
  // rate i per unit time at time t. There are tspan/q payments and the
  // effective per-interval rate is (1+i)^q - 1. Results are rounded to the
  // cent so the assertions are immune to last-digit floating-point noise
  // (verified identical to wolframscript after the same rounding).
  mod time_value_annuity_tests {
    use super::*;

    #[test]
    fn future_value_semiannual_payments() {
      // 20 semiannual payments of 1000 over 10 years at 6%/yr, valued at t=10.
      assert_eq!(
        interpret("Round[TimeValue[Annuity[1000, 10, 1/2], 0.06, 10], 0.01]")
          .unwrap(),
        "26751.25"
      );
    }

    #[test]
    fn present_value_quarterly_payments() {
      // 20 quarterly payments of 500 over 5 years at 8%/yr, valued at t=0.
      assert_eq!(
        interpret("Round[TimeValue[Annuity[500, 5, 1/4], 0.08, 0], 0.01]")
          .unwrap(),
        "8221.14"
      );
    }

    #[test]
    fn present_value_monthly_payments() {
      // 36 monthly payments of 100 over 3 years at 8%/yr, valued at t=0.
      assert_eq!(
        interpret("Round[TimeValue[Annuity[100, 3, 1/12], 0.08, 0], 0.01]")
          .unwrap(),
        "3204.33"
      );
    }

    #[test]
    fn present_value_semiannual_payments() {
      assert_eq!(
        interpret("Round[TimeValue[Annuity[1000, 10, 1/2], 0.06, 0], 0.01]")
          .unwrap(),
        "14937.76"
      );
    }

    #[test]
    fn payment_interval_one_matches_two_arg_form() {
      // Annuity[pmt, tspan, 1] is the q = 1 specialization of the 2-arg form.
      let three_arg =
        interpret("Round[TimeValue[Annuity[1000, 10, 1], 0.06, 10], 0.01]")
          .unwrap();
      let two_arg =
        interpret("Round[TimeValue[Annuity[1000, 10], 0.06, 10], 0.01]")
          .unwrap();
      assert_eq!(three_arg, two_arg);
      assert_eq!(three_arg, "13180.79");
    }

    // Annuity[{p, ip, fp}, tspan (, q)]: a list first argument adds an initial
    // payment ip at time 0 and a final payment fp at time tspan on top of the
    // level payment p. A length-2 list omits fp. Previously Woxi threaded the
    // list through the annuity formula and returned a list of values.

    #[test]
    fn list_payment_with_initial() {
      // {100, 200}: level 100 for 2 periods plus a 200 payment at time 0.
      assert_eq!(
        interpret("Round[TimeValue[Annuity[{100, 200}, 2], 0.05, 0], 0.01]")
          .unwrap(),
        "385.94"
      );
    }

    #[test]
    fn list_payment_with_initial_and_final() {
      // {100, 50, 25}: level 100 for 4 periods, 50 at time 0, 25 at time 4.
      assert_eq!(
        interpret("Round[TimeValue[Annuity[{100, 50, 25}, 4], 0.05, 0], 0.01]")
          .unwrap(),
        "425.16"
      );
    }

    #[test]
    fn list_payment_with_interval() {
      // List payment composed with a semiannual payment interval q = 1/2.
      assert_eq!(
        interpret(
          "Round[TimeValue[Annuity[{100, 50, 25}, 4, 1/2], 0.05, 0], 0.01]"
        )
        .unwrap(),
        "788.51"
      );
    }

    #[test]
    fn list_payment_final_only_discounted() {
      // {0, 0, 500}: a single 500 payment at time 3, discounted to time 0.
      assert_eq!(
        interpret("Round[TimeValue[Annuity[{0, 0, 500}, 3], 0.05, 0], 0.01]")
          .unwrap(),
        "431.92"
      );
    }

    #[test]
    fn list_payment_future_value() {
      // {100, 200} valued at time 2 rather than time 0.
      assert_eq!(
        interpret("Round[TimeValue[Annuity[{100, 200}, 2], 0.05, 2], 0.01]")
          .unwrap(),
        "425.5"
      );
    }

    #[test]
    fn invalid_list_length_stays_unevaluated() {
      // Lists of length other than 2 or 3 are not valid annuity payment
      // specs and stay unevaluated (matching wolframscript).
      assert_eq!(
        interpret("TimeValue[Annuity[{100}, 2], 0.05, 0]").unwrap(),
        "TimeValue[Annuity[{100}, 2], 0.05, 0]"
      );
      assert_eq!(
        interpret("TimeValue[Annuity[{1, 2, 3, 4}, 2], 0.05, 0]").unwrap(),
        "TimeValue[Annuity[{1, 2, 3, 4}, 2], 0.05, 0]"
      );
    }
  }

  // TimeValue[Cashflow[...], i, t]. Cashflow[{c0, c1, ...}] places amount c_k
  // at time k; Cashflow[{{t0,c0}, {t1,c1}, ...}] places amount c_k at the
  // explicit time t_k. Its value at time t is Sum[c_k * (1+i)^(t - t_k)].
  // Previously the {time, amount}-pair form was fed through the scalar formula
  // and returned a list of values. Rounded to the cent so the assertions are
  // immune to last-digit float noise (verified identical to wolframscript).
  mod time_value_cashflow_tests {
    use super::*;

    #[test]
    fn explicit_time_amount_pairs() {
      // 100 at time 1 and 300 at time 3, valued at time 0.
      assert_eq!(
        interpret(
          "Round[TimeValue[Cashflow[{{1, 100}, {3, 300}}], 0.05, 0], 0.01]"
        )
        .unwrap(),
        "354.39"
      );
    }

    #[test]
    fn pairs_including_time_zero() {
      assert_eq!(
        interpret(
          "Round[TimeValue[Cashflow[{{0, 100}, {1, 100}, {2, 100}}], 0.05, 0], \
           0.01]"
        )
        .unwrap(),
        "285.94"
      );
    }

    #[test]
    fn single_pair_discounted() {
      // A single 500 payment at time 2, discounted to time 0.
      assert_eq!(
        interpret("Round[TimeValue[Cashflow[{{2, 500}}], 0.05, 0], 0.01]")
          .unwrap(),
        "453.51"
      );
    }

    #[test]
    fn pairs_future_value() {
      // Same pairs valued at time 3 rather than time 0.
      assert_eq!(
        interpret(
          "Round[TimeValue[Cashflow[{{1, 100}, {3, 300}}], 0.05, 3], 0.01]"
        )
        .unwrap(),
        "410.25"
      );
    }

    #[test]
    fn scalar_amounts_still_use_index_times() {
      // Regression guard: the plain-amount form is unchanged (amounts at
      // times 0, 1, 2).
      assert_eq!(
        interpret("Round[TimeValue[Cashflow[{100, 200, 300}], 0.05, 0], 0.01]")
          .unwrap(),
        "562.59"
      );
    }
  }
}
