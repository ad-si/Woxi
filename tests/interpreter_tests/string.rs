use super::*;

mod string_join_with_list {
  use super::*;

  #[test]
  fn string_join_list_of_strings() {
    assert_eq!(
      interpret("StringJoin[{\"a\", \"b\", \"c\"}]").unwrap(),
      "abc"
    );
  }

  #[test]
  fn string_join_empty_list() {
    assert_eq!(interpret("StringJoin[{}]").unwrap(), "");
  }

  #[test]
  fn string_join_multiple_args() {
    assert_eq!(
      interpret("StringJoin[\"hello\", \" \", \"world\"]").unwrap(),
      "hello world"
    );
  }

  #[test]
  fn string_join_with_table_result() {
    // StringJoin with a Table that returns strings
    assert_eq!(
      interpret("StringJoin[Table[\"x\", {i, 3}]]").unwrap(),
      "xxx"
    );
  }
}

mod string_split_list_delimiters {
  use super::*;

  #[test]
  fn multiple_delimiters() {
    assert_eq!(
      interpret("StringSplit[\"a!===b=!=c\", {\"==\", \"!=\", \"=\"}]")
        .unwrap(),
      "{a, , b, , c}"
    );
  }

  #[test]
  fn single_delimiter_in_list() {
    assert_eq!(
      interpret("StringSplit[\"a,b,c\", {\",\"}]").unwrap(),
      "{a, b, c}"
    );
  }
}

mod string_split_single_arg {
  use super::*;

  #[test]
  fn split_by_whitespace() {
    assert_eq!(
      interpret("StringSplit[\"Wolfram Language is incredible\"]").unwrap(),
      "{Wolfram, Language, is, incredible}"
    );
  }

  #[test]
  fn split_multiple_spaces() {
    assert_eq!(
      interpret("StringSplit[\"  hello   world  \"]").unwrap(),
      "{hello, world}"
    );
  }

  #[test]
  fn split_single_word() {
    assert_eq!(interpret("StringSplit[\"hello\"]").unwrap(), "{hello}");
  }

  #[test]
  fn split_empty_string() {
    assert_eq!(interpret("StringSplit[\"\"]").unwrap(), "{}");
  }

  #[test]
  fn split_with_tabs_and_newlines() {
    assert_eq!(
      interpret("StringSplit[\"a\\tb\\nc\"]").unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn map_string_reverse_over_split() {
    assert_eq!(
      interpret(
        "StringReverse /@ StringSplit[\"Wolfram Language is incredible\"]"
      )
      .unwrap(),
      "{marfloW, egaugnaL, si, elbidercni}"
    );
  }
}

mod string_split_regex {
  use super::*;

  #[test]
  fn split_by_regex_whitespace() {
    assert_eq!(
      interpret(
        r#"StringSplit["hello  world  foo", RegularExpression["\\s+"]]"#
      )
      .unwrap(),
      "{hello, world, foo}"
    );
  }

  #[test]
  fn split_by_regex_non_word() {
    assert_eq!(
      interpret(
        r#"StringSplit["Four score and seven", RegularExpression["\\W+"]]"#
      )
      .unwrap(),
      "{Four, score, and, seven}"
    );
  }

  #[test]
  fn split_by_regex_with_ignore_case() {
    assert_eq!(
      interpret(r#"StringSplit["helloXworldxfoo", RegularExpression["x"], IgnoreCase -> True]"#).unwrap(),
      "{hello, world, foo}"
    );
  }

  #[test]
  fn split_by_regex_ignore_case_false() {
    assert_eq!(
      interpret(r#"StringSplit["helloXworldxfoo", RegularExpression["x"], IgnoreCase -> False]"#).unwrap(),
      "{helloXworld, foo}"
    );
  }

  #[test]
  fn split_by_regex_digits() {
    assert_eq!(
      interpret(r#"StringSplit["abc123def456ghi", RegularExpression["\\d+"]]"#)
        .unwrap(),
      "{abc, def, ghi}"
    );
  }

  #[test]
  fn gettysburg_example() {
    assert_eq!(
      interpret(r#"SortBy[StringSplit["Four score and seven years ago our fathers brought forth", RegularExpression["\\W+"]], StringLength]"#).unwrap(),
      "{ago, and, our, Four, forth, score, seven, years, brought, fathers}"
    );
  }
}

mod string_replace {
  use super::*;

  #[test]
  fn single_rule() {
    assert_eq!(
      interpret(r#"StringReplace["hello world", "world" -> "planet"]"#)
        .unwrap(),
      "hello planet"
    );
  }

  #[test]
  fn list_of_rules() {
    assert_eq!(
        interpret(r#"StringReplace["hello world", {"hello" -> "goodbye", "world" -> "planet"}]"#)
          .unwrap(),
        "goodbye planet"
      );
  }

  #[test]
  fn replace_all_occurrences() {
    assert_eq!(
      interpret(r#"StringReplace["abcabc", "a" -> "x"]"#).unwrap(),
      "xbcxbc"
    );
  }

  #[test]
  fn replace_with_empty() {
    assert_eq!(
      interpret(r#"StringReplace["hello", "l" -> ""]"#).unwrap(),
      "heo"
    );
  }

  #[test]
  fn no_match() {
    assert_eq!(
      interpret(r#"StringReplace["hello", "xyz" -> "abc"]"#).unwrap(),
      "hello"
    );
  }
}

mod to_character_code {
  use super::*;

  #[test]
  fn basic_string() {
    assert_eq!(
      interpret(r#"ToCharacterCode["Hello"]"#).unwrap(),
      "{72, 101, 108, 108, 111}"
    );
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret(r#"ToCharacterCode[""]"#).unwrap(), "{}");
  }

  #[test]
  fn single_char() {
    assert_eq!(interpret(r#"ToCharacterCode["A"]"#).unwrap(), "{65}");
  }

  #[test]
  fn digits() {
    assert_eq!(
      interpret(r#"ToCharacterCode["0123"]"#).unwrap(),
      "{48, 49, 50, 51}"
    );
  }
}

mod from_character_code {
  use super::*;

  #[test]
  fn list_of_codes() {
    assert_eq!(
      interpret("FromCharacterCode[{72, 101, 108, 108, 111}]").unwrap(),
      "Hello"
    );
  }

  #[test]
  fn single_code() {
    assert_eq!(interpret("FromCharacterCode[65]").unwrap(), "A");
  }

  #[test]
  fn roundtrip() {
    assert_eq!(
      interpret(r#"FromCharacterCode[ToCharacterCode["Test"]]"#).unwrap(),
      "Test"
    );
  }
}

mod character_range {
  use super::*;

  #[test]
  fn lowercase() {
    assert_eq!(
      interpret(r#"CharacterRange["a", "z"]"#).unwrap(),
      "{a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z}"
    );
  }

  #[test]
  fn uppercase() {
    assert_eq!(
      interpret(r#"CharacterRange["A", "F"]"#).unwrap(),
      "{A, B, C, D, E, F}"
    );
  }

  #[test]
  fn digits() {
    assert_eq!(
      interpret(r#"CharacterRange["0", "9"]"#).unwrap(),
      "{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}"
    );
  }

  #[test]
  fn empty_range() {
    assert_eq!(interpret(r#"CharacterRange["z", "a"]"#).unwrap(), "{}");
  }

  #[test]
  fn single_char() {
    assert_eq!(interpret(r#"CharacterRange["m", "m"]"#).unwrap(), "{m}");
  }
}

mod letter_q {
  use super::*;

  #[test]
  fn all_letters() {
    assert_eq!(interpret("LetterQ[\"abc\"]").unwrap(), "True");
  }

  #[test]
  fn with_digits() {
    assert_eq!(interpret("LetterQ[\"ab3\"]").unwrap(), "False");
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret("LetterQ[\"\"]").unwrap(), "True");
  }

  #[test]
  fn uppercase() {
    assert_eq!(interpret("LetterQ[\"ABC\"]").unwrap(), "True");
  }
}

mod upper_case_q {
  use super::*;

  #[test]
  fn all_upper() {
    assert_eq!(interpret("UpperCaseQ[\"ABC\"]").unwrap(), "True");
  }

  #[test]
  fn mixed() {
    assert_eq!(interpret("UpperCaseQ[\"AbC\"]").unwrap(), "False");
  }

  #[test]
  fn all_lower() {
    assert_eq!(interpret("UpperCaseQ[\"abc\"]").unwrap(), "False");
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret("UpperCaseQ[\"\"]").unwrap(), "True");
  }
}

mod lower_case_q {
  use super::*;

  #[test]
  fn all_lower() {
    assert_eq!(interpret("LowerCaseQ[\"abc\"]").unwrap(), "True");
  }

  #[test]
  fn mixed() {
    assert_eq!(interpret("LowerCaseQ[\"AbC\"]").unwrap(), "False");
  }

  #[test]
  fn all_upper() {
    assert_eq!(interpret("LowerCaseQ[\"ABC\"]").unwrap(), "False");
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret("LowerCaseQ[\"\"]").unwrap(), "True");
  }
}

mod digit_q {
  use super::*;

  #[test]
  fn all_digits() {
    assert_eq!(interpret("DigitQ[\"123\"]").unwrap(), "True");
  }

  #[test]
  fn with_letters() {
    assert_eq!(interpret("DigitQ[\"12a\"]").unwrap(), "False");
  }

  #[test]
  fn empty() {
    assert_eq!(interpret("DigitQ[\"\"]").unwrap(), "True");
  }
}

mod alphabet {
  use super::*;

  #[test]
  fn basic() {
    let result = interpret("Alphabet[]").unwrap();
    assert!(result.starts_with("{a, b, c"));
    assert!(result.ends_with("x, y, z}"));
  }

  #[test]
  fn length() {
    assert_eq!(interpret("Length[Alphabet[]]").unwrap(), "26");
  }
}

mod operator_form {
  use super::*;

  #[test]
  fn string_starts_q_curried() {
    assert_eq!(
      interpret("StringStartsQ[\"He\"][\"Hello\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn string_starts_q_curried_false() {
    assert_eq!(
      interpret("StringStartsQ[\"Wo\"][\"Hello\"]").unwrap(),
      "False"
    );
  }

  #[test]
  fn string_ends_q_curried() {
    assert_eq!(interpret("StringEndsQ[\"lo\"][\"Hello\"]").unwrap(), "True");
  }

  #[test]
  fn string_contains_q_curried() {
    assert_eq!(
      interpret("StringContainsQ[\"ell\"][\"Hello\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn member_q_curried() {
    assert_eq!(interpret("MemberQ[2][{1, 2, 3}]").unwrap(), "True");
  }

  #[test]
  fn member_q_curried_false() {
    assert_eq!(interpret("MemberQ[5][{1, 2, 3}]").unwrap(), "False");
  }

  #[test]
  fn select_with_curried_string_starts_q() {
    assert_eq!(
      interpret(
        "Select[{\"apple\", \"avocado\", \"banana\"}, StringStartsQ[\"a\"]]"
      )
      .unwrap(),
      "{apple, avocado}"
    );
  }
}

mod string_patterns {
  use super::*;

  #[test]
  fn repeated_parsing() {
    // Repeated[x] displays as x..
    assert_eq!(
      interpret("Repeated[DigitCharacter]").unwrap(),
      "DigitCharacter.."
    );
    // RepeatedNull[x] displays as x...
    assert_eq!(
      interpret("RepeatedNull[DigitCharacter]").unwrap(),
      "DigitCharacter..."
    );
  }

  #[test]
  fn repeated_shorthand_parsing() {
    // .. parses as Repeated, ... parses as RepeatedNull
    assert_eq!(interpret("DigitCharacter ..").unwrap(), "DigitCharacter..");
    assert_eq!(
      interpret("DigitCharacter ...").unwrap(),
      "DigitCharacter..."
    );
    assert_eq!(
      interpret("LetterCharacter ..").unwrap(),
      "LetterCharacter.."
    );
  }

  #[test]
  fn repeated_head() {
    assert_eq!(interpret("Head[DigitCharacter ..]").unwrap(), "Repeated");
    assert_eq!(
      interpret("Head[DigitCharacter ...]").unwrap(),
      "RepeatedNull"
    );
  }

  #[test]
  fn string_cases_digit_character_repeated() {
    assert_eq!(
      interpret("StringCases[\"The year is 2025\", DigitCharacter ..]")
        .unwrap(),
      "{2025}"
    );
    assert_eq!(
      interpret("StringCases[\"abc123def456\", DigitCharacter ..]").unwrap(),
      "{123, 456}"
    );
  }

  #[test]
  fn string_cases_single_digit_character() {
    assert_eq!(
      interpret("StringCases[\"abc123\", DigitCharacter]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn string_cases_letter_character_repeated() {
    assert_eq!(
      interpret("StringCases[\"abc123def456\", LetterCharacter ..]").unwrap(),
      "{abc, def}"
    );
  }

  #[test]
  fn string_cases_whitespace_character_repeated() {
    assert_eq!(
      interpret("StringCases[\"hello world foo\", WhitespaceCharacter ..]")
        .unwrap(),
      "{ ,  }"
    );
  }

  #[test]
  fn string_cases_word_character_repeated() {
    assert_eq!(
      interpret("StringCases[\"abc123\", WordCharacter ..]").unwrap(),
      "{abc123}"
    );
  }

  #[test]
  fn string_cases_no_matches() {
    assert_eq!(
      interpret("StringCases[\"hello\", DigitCharacter ..]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn string_cases_literal_still_works() {
    assert_eq!(
      interpret("StringCases[\"abcabc\", \"bc\"]").unwrap(),
      "{bc, bc}"
    );
  }
}

mod string_part {
  use super::*;

  #[test]
  fn single_index() {
    assert_eq!(interpret(r#"StringPart["abcdefgh", 3]"#).unwrap(), "c");
  }

  #[test]
  fn negative_index() {
    assert_eq!(interpret(r#"StringPart["abcdefgh", -2]"#).unwrap(), "g");
  }

  #[test]
  fn first_char() {
    assert_eq!(interpret(r#"StringPart["abcdefgh", 1]"#).unwrap(), "a");
  }

  #[test]
  fn last_char() {
    assert_eq!(interpret(r#"StringPart["abcdefgh", -1]"#).unwrap(), "h");
  }

  #[test]
  fn list_of_indices() {
    assert_eq!(
      interpret(r#"StringPart["abcdefgh", {3, 5}]"#).unwrap(),
      "{c, e}"
    );
  }

  #[test]
  fn list_with_negative() {
    assert_eq!(
      interpret(r#"StringPart["abcdefgh", {1, -1}]"#).unwrap(),
      "{a, h}"
    );
  }
}

mod string_take_drop {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret(r#"StringTakeDrop["Hello World", 5]"#).unwrap(),
      "{Hello,  World}"
    );
  }

  #[test]
  fn take_all() {
    assert_eq!(interpret(r#"StringTakeDrop["abc", 3]"#).unwrap(), "{abc, }");
  }

  #[test]
  fn negative() {
    assert_eq!(
      interpret(r#"StringTakeDrop["Hello World", -5]"#).unwrap(),
      "{World, Hello }"
    );
  }
}

mod hamming_distance {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret(r#"HammingDistance["karolin", "kathrin"]"#).unwrap(),
      "3"
    );
  }

  #[test]
  fn binary_strings() {
    assert_eq!(
      interpret(r#"HammingDistance["1011101", "1001001"]"#).unwrap(),
      "2"
    );
  }

  #[test]
  fn identical() {
    assert_eq!(interpret(r#"HammingDistance["abc", "abc"]"#).unwrap(), "0");
  }

  #[test]
  fn completely_different() {
    assert_eq!(interpret(r#"HammingDistance["abc", "xyz"]"#).unwrap(), "3");
  }
}

mod character_counts {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret(r#"CharacterCounts["hello"]"#).unwrap(),
      "<|l -> 2, o -> 1, e -> 1, h -> 1|>"
    );
  }

  #[test]
  fn sorted_by_frequency() {
    assert_eq!(
      interpret(r#"CharacterCounts["aababcabcd"]"#).unwrap(),
      "<|a -> 4, b -> 3, c -> 2, d -> 1|>"
    );
  }

  #[test]
  fn single_char() {
    assert_eq!(
      interpret(r#"CharacterCounts["aaa"]"#).unwrap(),
      "<|a -> 3|>"
    );
  }
}

mod remove_diacritics {
  use super::*;

  #[test]
  fn basic_accents() {
    assert_eq!(
      interpret("RemoveDiacritics[\"caf\u{00e9}\"]").unwrap(),
      "cafe"
    );
  }

  #[test]
  fn plain_ascii() {
    assert_eq!(interpret(r#"RemoveDiacritics["hello"]"#).unwrap(), "hello");
  }

  #[test]
  fn umlaut() {
    assert_eq!(
      interpret("RemoveDiacritics[\"\u{00fc}ber\"]").unwrap(),
      "uber"
    );
  }
}

mod string_rotate_left {
  use super::*;

  #[test]
  fn rotate_by_2() {
    assert_eq!(
      interpret(r#"StringRotateLeft["abcdef", 2]"#).unwrap(),
      "cdefab"
    );
  }

  #[test]
  fn default_rotation() {
    assert_eq!(
      interpret(r#"StringRotateLeft["abcdef"]"#).unwrap(),
      "bcdefa"
    );
  }

  #[test]
  fn rotate_full_cycle() {
    assert_eq!(
      interpret(r#"StringRotateLeft["abcdef", 6]"#).unwrap(),
      "abcdef"
    );
  }

  #[test]
  fn negative_rotation() {
    // Negative rotation = rotate right
    assert_eq!(
      interpret(r#"StringRotateLeft["abcdef", -1]"#).unwrap(),
      "fabcde"
    );
  }
}

mod string_rotate_right {
  use super::*;

  #[test]
  fn rotate_by_2() {
    assert_eq!(
      interpret(r#"StringRotateRight["abcdef", 2]"#).unwrap(),
      "efabcd"
    );
  }

  #[test]
  fn default_rotation() {
    assert_eq!(
      interpret(r#"StringRotateRight["abcdef"]"#).unwrap(),
      "fabcde"
    );
  }
}

mod alphabetic_sort {
  use super::*;

  #[test]
  fn case_insensitive() {
    assert_eq!(
      interpret(r#"AlphabeticSort[{"Banana", "apple", "Cherry"}]"#).unwrap(),
      "{apple, Banana, Cherry}"
    );
  }

  #[test]
  fn already_sorted() {
    assert_eq!(
      interpret(r#"AlphabeticSort[{"a", "b", "c"}]"#).unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn reverse_order() {
    assert_eq!(
      interpret(r#"AlphabeticSort[{"c", "b", "a"}]"#).unwrap(),
      "{a, b, c}"
    );
  }
}

mod hash {
  use super::*;

  #[test]
  fn md5_hex_string() {
    assert_eq!(
      interpret(r#"Hash["hello", "MD5", "HexString"]"#).unwrap(),
      "5d41402abc4b2a76b9719d911017c592"
    );
  }

  #[test]
  fn sha256_hex_string() {
    assert_eq!(
      interpret(r#"Hash["hello", "SHA256", "HexString"]"#).unwrap(),
      "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
  }

  #[test]
  fn sha1_hex_string() {
    assert_eq!(
      interpret(r#"Hash["hello", "SHA", "HexString"]"#).unwrap(),
      "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d"
    );
  }

  #[test]
  fn md5_integer() {
    assert_eq!(
      interpret(r#"Hash["hello", "MD5"]"#).unwrap(),
      "123957004363873451094272536567338222994"
    );
  }

  #[test]
  fn default_returns_unevaluated() {
    assert_eq!(interpret(r#"Hash["hello"]"#).unwrap(), "Hash[hello]");
  }
}

mod string_riffle_extended {
  use super::*;

  #[test]
  fn left_sep_right() {
    assert_eq!(
      interpret(r#"StringRiffle[{"a", "b", "c"}, {"[", "|", "]"}]"#).unwrap(),
      "[a|b|c]"
    );
  }

  #[test]
  fn braces() {
    assert_eq!(
      interpret(r#"StringRiffle[{"a", "b", "c"}, {"{", ", ", "}"}]"#).unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn basic_still_works() {
    assert_eq!(
      interpret(r#"StringRiffle[{"a", "b", "c"}, ", "]"#).unwrap(),
      "a, b, c"
    );
  }
}

mod palindrome_q {
  use super::*;

  #[test]
  fn string_palindrome() {
    assert_eq!(interpret(r#"PalindromeQ["racecar"]"#).unwrap(), "True");
  }

  #[test]
  fn string_not_palindrome() {
    assert_eq!(interpret(r#"PalindromeQ["hello"]"#).unwrap(), "False");
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret(r#"PalindromeQ[""]"#).unwrap(), "True");
  }

  #[test]
  fn single_char_string() {
    assert_eq!(interpret(r#"PalindromeQ["a"]"#).unwrap(), "True");
  }

  #[test]
  fn list_palindrome() {
    assert_eq!(interpret("PalindromeQ[{1, 2, 3, 2, 1}]").unwrap(), "True");
  }

  #[test]
  fn list_not_palindrome() {
    assert_eq!(interpret("PalindromeQ[{1, 2, 3}]").unwrap(), "False");
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("PalindromeQ[{}]").unwrap(), "True");
  }

  #[test]
  fn integer_palindrome() {
    assert_eq!(interpret("PalindromeQ[12321]").unwrap(), "True");
  }

  #[test]
  fn integer_not_palindrome() {
    assert_eq!(interpret("PalindromeQ[12345]").unwrap(), "False");
  }

  #[test]
  fn single_digit() {
    assert_eq!(interpret("PalindromeQ[7]").unwrap(), "True");
  }

  #[test]
  fn list_with_symbols() {
    assert_eq!(interpret("PalindromeQ[{a, b, a}]").unwrap(), "True");
  }
}

mod string_drop_list_spec {
  use super::*;

  #[test]
  fn drop_single_char() {
    assert_eq!(interpret(r#"StringDrop["abcde", {2}]"#).unwrap(), "acde");
  }

  #[test]
  fn drop_range() {
    assert_eq!(interpret(r#"StringDrop["abcde", {2,3}]"#).unwrap(), "ade");
  }

  #[test]
  fn drop_reversed_range() {
    assert_eq!(interpret(r#"StringDrop["abcd",{3,2}]"#).unwrap(), "abcd");
  }

  #[test]
  fn drop_zero() {
    assert_eq!(interpret(r#"StringDrop["abcd",0]"#).unwrap(), "abcd");
  }
}

mod string_take_extended {
  use super::*;

  #[test]
  fn take_zero() {
    assert_eq!(interpret(r#"StringTake["abcde", 0]"#).unwrap(), "");
  }

  #[test]
  fn take_with_step() {
    assert_eq!(
      interpret(r#"StringTake["abcdefgh", {1, 5, 2}]"#).unwrap(),
      "ace"
    );
  }

  #[test]
  fn take_list_of_strings() {
    assert_eq!(
      interpret(r#"StringTake[{"abcdef", "stuv", "xyzw"}, -2]"#).unwrap(),
      "{ef, uv, zw}"
    );
  }

  #[test]
  fn take_all() {
    assert_eq!(interpret(r#"StringTake["abcdef", All]"#).unwrap(), "abcdef");
  }

  #[test]
  fn take_single_char() {
    assert_eq!(interpret(r#"StringTake["abcde", {2}]"#).unwrap(), "b");
  }

  #[test]
  fn take_range() {
    assert_eq!(interpret(r#"StringTake["abcd", {2,3}]"#).unwrap(), "bc");
  }
}

mod string_match_q_patterns {
  use super::*;

  #[test]
  fn digit_character() {
    assert_eq!(
      interpret(r#"StringMatchQ["1", DigitCharacter]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn repeated_digit_character() {
    assert_eq!(
      interpret(r#"StringMatchQ["123245", Repeated[DigitCharacter]]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn word_character_repeated() {
    assert_eq!(
      interpret(r#"StringMatchQ["abc123DEF", Repeated[WordCharacter]]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn number_string() {
    assert_eq!(
      interpret(r#"StringMatchQ["1234", NumberString]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["1234.5", NumberString]"#).unwrap(),
      "True"
    );
  }
}

mod string_split_edge {
  use super::*;

  #[test]
  fn split_x_by_x() {
    assert_eq!(
      interpret(r#"StringSplit["x", "x"]"#).unwrap(),
      "{}"
    );
  }

  #[test]
  fn split_filters_empty() {
    assert_eq!(
      interpret(r#"StringSplit["xxax", "x"]"#).unwrap(),
      "{a}"
    );
  }
}

mod integer_string_tests {
  use super::*;

  #[test]
  fn negative_drops_sign() {
    assert_eq!(interpret("IntegerString[-500]").unwrap(), "500");
  }

  #[test]
  fn truncate_to_length() {
    assert_eq!(interpret("IntegerString[12345, 10, 3]").unwrap(), "345");
  }
}
