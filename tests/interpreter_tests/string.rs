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
