use super::*;

mod string_length_arg_errors {
  use super::*;

  #[test]
  fn non_string_identifier_returns_unevaluated() {
    // Matches wolframscript: StringLength[x] stays unevaluated with a
    // StringLength::string message; it does NOT return the length of the
    // identifier's name.
    assert_eq!(interpret("StringLength[x]").unwrap(), "StringLength[x]");
  }

  #[test]
  fn plain_string_still_works() {
    assert_eq!(interpret(r#"StringLength["abc"]"#).unwrap(), "3");
  }

  #[test]
  fn list_of_strings_threads() {
    assert_eq!(
      interpret(r#"StringLength[{"a", "bb", "ccc"}]"#).unwrap(),
      "{1, 2, 3}"
    );
  }
}

mod string_join_arg_errors {
  use super::*;

  #[test]
  fn non_string_operand_returns_unevaluated() {
    // Matches wolframscript: "Debian" <> 6 emits StringJoin::string and
    // returns StringJoin[Debian, 6]. Previously Woxi coerced and produced
    // "Debian6".
    assert_eq!(
      interpret(r#""Debian" <> 6"#).unwrap(),
      "StringJoin[Debian, 6]"
    );
  }

  #[test]
  fn non_string_operand_message_points_at_bad_arg() {
    // Regression: the StringJoin::string warning must name the 1-based
    // position of the first non-string argument and render the call in
    // infix form — matching wolframscript's `position 2 in U<>2`.
    let _ = interpret(r#""U" <> 2"#).unwrap();
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "StringJoin::string: String expected at position 2 in U<>2."
      )),
      "expected infix `U<>2` message, got {:?}",
      msgs
    );
  }

  #[test]
  fn plain_string_chain_still_works() {
    assert_eq!(interpret(r#""a" <> "b" <> "c""#).unwrap(), "abc");
  }
}

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
  fn string_join_infix_then_input_form() {
    // Infix <> and postfix // InputForm — InputForm stays unevaluated and
    // wraps the concatenated string.
    assert_eq!(
      interpret(r#""a" <> "b" <> "c" // InputForm"#).unwrap(),
      "InputForm[abc]"
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

  #[test]
  fn string_join_flattens_lists_in_multi_arg() {
    // StringJoin should flatten list arguments when mixed with strings
    assert_eq!(
      interpret(r#"StringJoin["x", {"a", "b"}, "y"]"#).unwrap(),
      "xaby"
    );
  }

  #[test]
  fn string_join_flattens_nested_lists() {
    assert_eq!(
      interpret(r#"StringJoin[{"a", {"b", "c"}}, "d"]"#).unwrap(),
      "abcd"
    );
  }

  #[test]
  fn string_join_hello_world_with_nested_list_and_tail() {
    assert_eq!(
      interpret(r#"StringJoin[{"Hello", " ", {"world"}}, "!"]"#).unwrap(),
      "Hello world!"
    );
  }

  #[test]
  fn string_join_in_rule_rhs() {
    // StringJoin (<>) must parse correctly in the RHS of Rule inside a list
    assert_eq!(
      interpret(r#"StringReplace["hello", "ello" -> "i" <> " there"]"#)
        .unwrap(),
      "hi there"
    );
  }

  #[test]
  fn string_join_in_rule_delayed_rhs() {
    // StringJoin (<>) must parse correctly in the RHS of RuleDelayed inside a list
    assert_eq!(
      interpret(r#"{"abc"} /. x_String :> "(" <> x <> ")""#).unwrap(),
      "{(abc)}"
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
  fn split_by_whitespace_character_keeps_empty_runs() {
    assert_eq!(
      interpret("StringSplit[\"  abc    123  \", WhitespaceCharacter]")
        .unwrap(),
      "{abc, , , , 123}"
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

  #[test]
  fn string_reverse_threads_list() {
    assert_eq!(
      interpret(r#"StringReverse[{"abc", "def"}]"#).unwrap(),
      "{cba, fed}"
    );
  }

  #[test]
  fn string_reverse_threads_nested_list() {
    assert_eq!(
      interpret(r#"StringReverse[{{"ab", "cd"}, {"ef"}}]"#).unwrap(),
      "{{ba, dc}, {fe}}"
    );
  }

  #[test]
  fn characters_single_string() {
    assert_eq!(interpret(r#"Characters["abc"]"#).unwrap(), "{a, b, c}");
  }

  #[test]
  fn characters_threads_list() {
    assert_eq!(
      interpret(r#"Characters[{"abc", "de"}]"#).unwrap(),
      "{{a, b, c}, {d, e}}"
    );
  }

  #[test]
  fn characters_empty_string() {
    assert_eq!(interpret(r#"Characters[""]"#).unwrap(), "{}");
  }

  #[test]
  fn to_upper_case_single_string() {
    assert_eq!(interpret(r#"ToUpperCase["abc"]"#).unwrap(), "ABC");
  }

  #[test]
  fn to_upper_case_threads_list() {
    assert_eq!(
      interpret(r#"ToUpperCase[{"abc", "def"}]"#).unwrap(),
      "{ABC, DEF}"
    );
  }

  #[test]
  fn to_lower_case_single_string() {
    assert_eq!(interpret(r#"ToLowerCase["ABC"]"#).unwrap(), "abc");
  }

  #[test]
  fn to_lower_case_threads_list() {
    assert_eq!(
      interpret(r#"ToLowerCase[{"ABC", "Def"}]"#).unwrap(),
      "{abc, def}"
    );
  }

  #[test]
  fn to_upper_case_threads_nested_list() {
    assert_eq!(
      interpret(r#"ToUpperCase[{{"ab", "cd"}, "ef"}]"#).unwrap(),
      "{{AB, CD}, EF}"
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

  // StartOfLine and EndOfLine are zero-width anchors — splitting by them
  // produces segments that start / end with a newline depending on which
  // side of the anchor the newline falls on.
  #[test]
  fn split_by_end_of_line() {
    assert_eq!(
      interpret("StringSplit[\"abc\\ndef\\nhij\", EndOfLine]").unwrap(),
      "{abc, \ndef, \nhij}"
    );
  }

  #[test]
  fn split_by_start_of_line() {
    assert_eq!(
      interpret("StringSplit[\"abc\\ndef\\nhij\", StartOfLine]").unwrap(),
      "{abc\n, def\n, hij}"
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
  fn alternatives_pattern() {
    // "1" | "2" matches either "1" or "2" — replace each with "X".
    assert_eq!(
      interpret(r#"StringReplace["0123 3210", "1" | "2" -> "X"]"#).unwrap(),
      "0XX3 3XX0"
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

  #[test]
  fn named_pattern_rule_delayed() {
    // RuleDelayed with named pattern variable and function application
    assert_eq!(
      interpret(r#"StringReplace["hello world", " " ~~ x_ :> ToUpperCase[x]]"#)
        .unwrap(),
      "helloWorld"
    );
  }

  #[test]
  fn named_pattern_rule() {
    // Rule with named pattern variable — substitutes matched string
    assert_eq!(
      interpret(r#"StringReplace["hello world", " " ~~ x_ -> x]"#).unwrap(),
      "helloworld"
    );
  }

  #[test]
  fn named_pattern_multiple_matches() {
    // Multiple matches with delayed replacement
    assert_eq!(
      interpret(r#"StringReplace["the cat sat", " " ~~ x_ :> ToUpperCase[x]]"#)
        .unwrap(),
      "theCatSat"
    );
  }

  #[test]
  fn named_character_in_pattern_and_target() {
    // \[CirclePlus] is a named character in both the subject and the pattern;
    // wolframscript rewrites the Unicode ⊕ glyph to the literal "x".
    assert_eq!(
      interpret(
        r#"StringReplace["product: A \[CirclePlus] B", "\[CirclePlus]" -> "x"]"#
      )
      .unwrap(),
      "product: A x B"
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

  #[test]
  fn greek_named_chars() {
    // \[Alpha]\[Beta]\[Gamma] → Unicode code points 945, 946, 947.
    assert_eq!(
      interpret(r#"ToCharacterCode["\[Alpha]\[Beta]\[Gamma]"]"#).unwrap(),
      "{945, 946, 947}"
    );
  }

  // With an explicit "UTF8" encoding, multi-byte characters are returned
  // as their underlying byte sequence (two bytes for ä).
  #[test]
  fn utf8_encoding_returns_bytes() {
    assert_eq!(
      interpret(r#"ToCharacterCode["ä", "UTF8"]"#).unwrap(),
      "{195, 164}"
    );
  }

  // With an ASCII-compatible single-byte encoding like ISO8859-1, the
  // codepoint itself fits and is returned directly.
  #[test]
  fn iso8859_1_encoding_single_byte() {
    assert_eq!(
      interpret(r#"ToCharacterCode["ä", "ISO8859-1"]"#).unwrap(),
      "{228}"
    );
  }

  // A list containing a non-string is a type error — wolframscript emits
  // ToCharacterCode::strse and returns the call unchanged.
  #[test]
  fn mixed_list_stays_unevaluated() {
    assert_eq!(
      interpret(r#"ToCharacterCode[{"ab", x}]"#).unwrap(),
      "ToCharacterCode[{ab, x}]"
    );
  }

  #[test]
  fn non_string_single_arg_stays_unevaluated() {
    assert_eq!(
      interpret("ToCharacterCode[42]").unwrap(),
      "ToCharacterCode[42]"
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

  // A second CharacterEncoding argument is accepted — for ASCII-compatible
  // encodings like ISO8859-1, the result is identical to the single-arg form
  // because the codepoints are already Unicode.
  #[test]
  fn with_character_encoding_option() {
    assert_eq!(
      interpret(r#"FromCharacterCode[228, "ISO8859-1"]"#).unwrap(),
      "ä"
    );
    assert_eq!(
      interpret(r#"FromCharacterCode[{228, 246}, "UTF-8"]"#).unwrap(),
      "äö"
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

  #[test]
  fn german_is_plain_latin() {
    // wolframscript's Alphabet["German"] returns the plain 26-letter alphabet
    // (no ä/ö/ü/ß). Regression for mathics atomic/strings.py:235.
    assert_eq!(
      interpret("Alphabet[\"German\"]").unwrap(),
      "{a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, \
       x, y, z}"
    );
  }

  #[test]
  fn spanish_has_enye() {
    assert_eq!(
      interpret("Alphabet[\"Spanish\"]").unwrap(),
      "{a, b, c, d, e, f, g, h, i, j, k, l, m, n, ñ, o, p, q, r, s, t, u, v, \
       w, x, y, z}"
    );
  }

  #[test]
  fn russian_differs_from_cyrillic() {
    // wolframscript's Cyrillic list is a superset of Russian's (covers
    // Ukrainian, Serbian, …). Regression for mathics atomic/strings.py:239,
    // whose "EXPECTED: True" does not match wolframscript.
    assert_eq!(
      interpret("Alphabet[\"Russian\"] == Alphabet[\"Cyrillic\"]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Length[Alphabet[\"Cyrillic\"]]").unwrap(),
      "49"
    );
  }
}

mod from_letter_number {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("FromLetterNumber[5]").unwrap(), "e");
  }

  #[test]
  fn first_letter() {
    assert_eq!(interpret("FromLetterNumber[1]").unwrap(), "a");
  }

  #[test]
  fn last_letter() {
    assert_eq!(interpret("FromLetterNumber[26]").unwrap(), "z");
  }

  #[test]
  fn list_input() {
    assert_eq!(
      interpret("FromLetterNumber[{1, 2, 3}]").unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn zero_gives_space() {
    assert_eq!(interpret("FromLetterNumber[0]").unwrap(), " ");
  }

  #[test]
  fn out_of_range_gives_space() {
    assert_eq!(interpret("FromLetterNumber[27]").unwrap(), " ");
  }

  #[test]
  fn negative_wraps() {
    assert_eq!(interpret("FromLetterNumber[-1]").unwrap(), "z");
  }

  #[test]
  fn negative_first() {
    assert_eq!(interpret("FromLetterNumber[-26]").unwrap(), "a");
  }

  #[test]
  fn negative_out_of_range() {
    assert_eq!(interpret("FromLetterNumber[-27]").unwrap(), " ");
  }

  #[test]
  fn full_negative_range() {
    assert_eq!(
      interpret("Table[FromLetterNumber[i], {i, -5, -1}]").unwrap(),
      "{v, w, x, y, z}"
    );
  }
}

mod letter_number {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("LetterNumber[\"e\"]").unwrap(), "5");
  }

  #[test]
  fn first_letter() {
    assert_eq!(interpret("LetterNumber[\"a\"]").unwrap(), "1");
  }

  #[test]
  fn last_letter() {
    assert_eq!(interpret("LetterNumber[\"z\"]").unwrap(), "26");
  }

  #[test]
  fn uppercase() {
    assert_eq!(interpret("LetterNumber[\"A\"]").unwrap(), "1");
  }

  #[test]
  fn non_letter() {
    assert_eq!(interpret("LetterNumber[\"1\"]").unwrap(), "0");
  }

  #[test]
  fn multi_char_string() {
    assert_eq!(
      interpret("LetterNumber[\"hello\"]").unwrap(),
      "{8, 5, 12, 12, 15}"
    );
  }

  #[test]
  fn list_input() {
    assert_eq!(
      interpret("LetterNumber[{\"a\", \"z\"}]").unwrap(),
      "{1, 26}"
    );
  }

  #[test]
  fn greek_beta() {
    // \[Beta] is the Greek lowercase beta (β). Its position is 2.
    assert_eq!(
      interpret("LetterNumber[\"\\[Beta]\", \"Greek\"]").unwrap(),
      "2"
    );
  }

  #[test]
  fn greek_alpha_omega() {
    assert_eq!(interpret("LetterNumber[\"α\", \"Greek\"]").unwrap(), "1");
    assert_eq!(interpret("LetterNumber[\"ω\", \"Greek\"]").unwrap(), "24");
  }

  #[test]
  fn greek_final_sigma_same_as_sigma() {
    // Both ς (final sigma) and σ (sigma) map to position 18.
    assert_eq!(interpret("LetterNumber[\"σ\", \"Greek\"]").unwrap(), "18");
    assert_eq!(interpret("LetterNumber[\"ς\", \"Greek\"]").unwrap(), "18");
  }

  #[test]
  fn greek_uppercase_normalizes() {
    assert_eq!(interpret("LetterNumber[\"Β\", \"Greek\"]").unwrap(), "2");
    assert_eq!(interpret("LetterNumber[\"Ω\", \"Greek\"]").unwrap(), "24");
  }

  #[test]
  fn greek_non_letter_returns_zero() {
    assert_eq!(interpret("LetterNumber[\"a\", \"Greek\"]").unwrap(), "0");
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
  fn member_q_with_blank_pattern() {
    assert_eq!(interpret("MemberQ[{1, 2, 3}, _Integer]").unwrap(), "True");
  }

  #[test]
  fn member_q_with_blank_pattern_no_match() {
    assert_eq!(interpret("MemberQ[{1, 2, 3}, _String]").unwrap(), "False");
  }

  #[test]
  fn member_q_with_string_pattern() {
    assert_eq!(
      interpret(r#"MemberQ[{1, "a", 2}, _String]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn member_q_with_head_pattern() {
    assert_eq!(
      interpret("MemberQ[{f[1], g[2], h[3]}, _f]").unwrap(),
      "True"
    );
  }

  #[test]
  fn member_q_with_condition_pattern() {
    assert_eq!(
      interpret("MemberQ[{1, 2, 3, 4, 5}, _?(# > 3 &)]").unwrap(),
      "True"
    );
  }

  #[test]
  fn member_q_with_condition_pattern_no_match() {
    assert_eq!(
      interpret("MemberQ[{1, 2, 3}, _?(# > 10 &)]").unwrap(),
      "False"
    );
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

mod ignore_case {
  use super::*;

  #[test]
  fn string_contains_q_ignore_case() {
    assert_eq!(
      interpret(
        "StringContainsQ[\"Hello World\", \"world\", IgnoreCase -> True]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn string_contains_q_case_sensitive() {
    assert_eq!(
      interpret("StringContainsQ[\"Hello World\", \"world\"]").unwrap(),
      "False"
    );
  }

  #[test]
  fn string_starts_q_ignore_case() {
    assert_eq!(
      interpret("StringStartsQ[\"Hello\", \"hello\", IgnoreCase -> True]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn string_ends_q_ignore_case() {
    assert_eq!(
      interpret("StringEndsQ[\"Hello\", \"ELLO\", IgnoreCase -> True]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn string_starts_q_threads_list() {
    assert_eq!(
      interpret(r#"StringStartsQ[{"hello", "world"}, "he"]"#).unwrap(),
      "{True, False}"
    );
  }

  #[test]
  fn string_ends_q_threads_list() {
    assert_eq!(
      interpret(r#"StringEndsQ[{"hello", "world"}, "d"]"#).unwrap(),
      "{False, True}"
    );
  }

  #[test]
  fn string_contains_q_threads_list() {
    assert_eq!(
      interpret(r#"StringContainsQ[{"hello", "world", "xy"}, "o"]"#).unwrap(),
      "{True, True, False}"
    );
  }

  #[test]
  fn string_starts_q_threads_list_ignore_case() {
    assert_eq!(
      interpret(
        r#"StringStartsQ[{"Hello", "world"}, "HE", IgnoreCase -> True]"#
      )
      .unwrap(),
      "{True, False}"
    );
  }

  #[test]
  fn string_match_q_ignore_case() {
    assert_eq!(
      interpret("StringMatchQ[\"Hello\", \"hello\", IgnoreCase -> True]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn string_match_q_threads_over_list() {
    assert_eq!(
      interpret(r#"StringMatchQ[{"abc", "ab1", "abcd"}, "abc"]"#).unwrap(),
      "{True, False, False}"
    );
    assert_eq!(
      interpret(
        r#"StringMatchQ[{"abc", "ab1", "abcd"}, RegularExpression["[a-z]+"]]"#
      )
      .unwrap(),
      "{True, False, True}"
    );
    // IgnoreCase option still threads.
    assert_eq!(
      interpret(r#"StringMatchQ[{"ABC", "ab1"}, "abc", IgnoreCase -> True]"#)
        .unwrap(),
      "{True, False}"
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

  #[test]
  fn string_cases_rule() {
    assert_eq!(interpret(r#"StringCases["abc", "a" -> 1]"#).unwrap(), "{1}");
  }

  #[test]
  fn string_cases_shortest_blank_sequence() {
    assert_eq!(
      interpret(r#"StringCases["aabaaab", Shortest["a" ~~ __ ~~ "b"]]"#)
        .unwrap(),
      "{aab, aaab}"
    );
  }

  #[test]
  fn string_cases_longest_blank_sequence() {
    assert_eq!(
      interpret(r#"StringCases["aabaaab", Longest["a" ~~ __ ~~ "b"]]"#)
        .unwrap(),
      "{aabaaab}"
    );
  }

  #[test]
  fn string_cases_shortest_regex_quantifier() {
    // Shortest applied to a regex `+` quantifier makes it non-greedy.
    assert_eq!(
      interpret(
        r#"StringCases["aabaaab", Shortest[RegularExpression["a+b"]]]"#
      )
      .unwrap(),
      "{aab, aaab}"
    );
  }

  #[test]
  fn string_cases_shortest_named_capture() {
    assert_eq!(
      interpret(
        r#"StringCases["-abc- def -uvw- xyz", Shortest["-" ~~ x__ ~~ "-"] -> x]"#
      )
      .unwrap(),
      "{abc, uvw}"
    );
  }

  #[test]
  fn string_cases_rule_list_with_max() {
    assert_eq!(
      interpret(r#"StringCases["abba", {"a" -> 10, "b" -> 20}, 2]"#).unwrap(),
      "{10, 20}"
    );
  }

  #[test]
  fn string_cases_rule_list_all_matches() {
    assert_eq!(
      interpret(r#"StringCases["abba", {"a" -> 10, "b" -> 20}]"#).unwrap(),
      "{10, 20, 20, 10}"
    );
  }

  #[test]
  fn string_cases_rule_list_longest_non_overlapping() {
    // When rules are tried in order, the first matching rule wins at each
    // position and the scan advances past the matched text.
    assert_eq!(
      interpret(r#"StringCases["aabb", {"aa" -> 100, "bb" -> 200}]"#).unwrap(),
      "{100, 200}"
    );
  }

  #[test]
  fn string_cases_regular_expression() {
    assert_eq!(
      interpret(r#"StringCases["cat bat mat", RegularExpression["[a-z]at"]]"#)
        .unwrap(),
      "{cat, bat, mat}"
    );
  }

  #[test]
  fn string_cases_regular_expression_digits() {
    assert_eq!(
      interpret(r#"StringCases["abc123def456", RegularExpression["[0-9]+"]]"#)
        .unwrap(),
      "{123, 456}"
    );
  }

  #[test]
  fn string_cases_max_count() {
    assert_eq!(interpret(r#"StringCases["abc", _, 2]"#).unwrap(), "{a, b}");
  }

  #[test]
  fn string_cases_max_count_literal() {
    assert_eq!(
      interpret(r#"StringCases["aabbbcc", "b", 2]"#).unwrap(),
      "{b, b}"
    );
  }

  #[test]
  fn string_cases_max_count_one() {
    assert_eq!(
      interpret(r#"StringCases["the cat sat on the mat", "at", 1]"#).unwrap(),
      "{at}"
    );
  }

  #[test]
  fn string_cases_max_count_pattern() {
    assert_eq!(
      interpret(r#"StringCases["abc123def456", DigitCharacter.., 1]"#).unwrap(),
      "{123}"
    );
  }

  #[test]
  fn string_cases_max_count_infinity() {
    assert_eq!(
      interpret(r#"StringCases["hello", _, Infinity]"#).unwrap(),
      "{h, e, l, l, o}"
    );
  }

  #[test]
  fn string_cases_max_count_exceeds_matches() {
    // When max count exceeds available matches, return all matches
    assert_eq!(interpret(r#"StringCases["ab", _, 10]"#).unwrap(), "{a, b}");
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

  #[test]
  fn ignore_case_option() {
    // HammingDistance with IgnoreCase -> True treats cases as equal.
    // Matches wolframscript.
    assert_eq!(
      interpret(r#"HammingDistance["TIME", "dime", IgnoreCase -> True]"#)
        .unwrap(),
      "1"
    );
    // Without IgnoreCase, all four differ (uppercase vs lowercase).
    assert_eq!(
      interpret(r#"HammingDistance["TIME", "dime"]"#).unwrap(),
      "4"
    );
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
  fn default_returns_integer() {
    // Default Hash uses Expression type (SipHash on InputForm)
    let result = interpret(r#"Hash["hello"]"#).unwrap();
    assert!(
      result.parse::<u64>().is_ok(),
      "Hash[\"hello\"] should return an integer, got: {}",
      result
    );
  }

  #[test]
  fn unknown_type_returns_unevaluated() {
    assert_eq!(
      interpret(r#"Hash[{a, b, c}, "xyzstr"]"#).unwrap(),
      "Hash[{a, b, c}, xyzstr]"
    );
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

  #[test]
  fn nested_default_separators() {
    assert_eq!(
      interpret(r#"StringRiffle[{{"a", "b"}, {"c", "d"}}]"#).unwrap(),
      "a b\nc d"
    );
  }

  #[test]
  fn nested_explicit_separators() {
    assert_eq!(
      interpret(r#"StringRiffle[{{"a", "b"}, {"c", "d"}}, "\n", " "]"#)
        .unwrap(),
      "a b\nc d"
    );
  }

  #[test]
  fn nested_three_by_two() {
    assert_eq!(
      interpret(r#"StringRiffle[{{"a","b","c"},{"d","e","f"}}, "; ", ", "]"#)
        .unwrap(),
      "a, b, c; d, e, f"
    );
  }

  #[test]
  fn nested_with_brackets_on_outer() {
    assert_eq!(
      interpret(
        r#"StringRiffle[{{"a", "b"}, {"c", "d"}}, {"[", "|", "]"}, "-"]"#
      )
      .unwrap(),
      "[a-b|c-d]"
    );
  }

  #[test]
  fn nested_with_integers() {
    assert_eq!(
      interpret(r#"StringRiffle[{{1, 2, 3}, {4, 5, 6}}, "\n", " "]"#).unwrap(),
      "1 2 3\n4 5 6"
    );
  }

  #[test]
  fn triple_nested_default_separators() {
    assert_eq!(
      interpret(
        r#"StringRiffle[{{{"a","b"},{"c","d"}},{{"e","f"},{"g","h"}}}]"#
      )
      .unwrap(),
      "a b\nc d\n\ne f\ng h"
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

  #[test]
  fn drop_threads_list() {
    assert_eq!(
      interpret(r#"StringDrop[{"abcde", "fghij"}, 2]"#).unwrap(),
      "{cde, hij}"
    );
  }

  #[test]
  fn drop_threads_list_negative() {
    assert_eq!(
      interpret(r#"StringDrop[{"abcde", "fghij"}, -2]"#).unwrap(),
      "{abc, fgh}"
    );
  }

  #[test]
  fn drop_threads_list_single_index() {
    assert_eq!(
      interpret(r#"StringDrop[{"abcde", "fghij"}, {2}]"#).unwrap(),
      "{acde, fhij}"
    );
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

  #[test]
  fn take_up_to_within_length() {
    assert_eq!(interpret(r#"StringTake["Hello", UpTo[3]]"#).unwrap(), "Hel");
  }

  #[test]
  fn take_up_to_exceeds_length() {
    assert_eq!(
      interpret(r#"StringTake["Hello", UpTo[10]]"#).unwrap(),
      "Hello"
    );
  }

  #[test]
  fn take_list_of_ranges() {
    assert_eq!(
      interpret(r#"StringTake["abcdef", {{1, 3}, {4, 6}}]"#).unwrap(),
      "{abc, def}"
    );
  }

  #[test]
  fn take_list_of_mixed_subspecs() {
    assert_eq!(
      interpret(r#"StringTake["abcdefghij", {{1, 3}, {5}, {7, 9}}]"#).unwrap(),
      "{abc, e, ghi}"
    );
  }

  #[test]
  fn take_list_of_one_range() {
    assert_eq!(
      interpret(r#"StringTake["abcdefghij", {{2, 4}}]"#).unwrap(),
      "{bcd}"
    );
  }

  #[test]
  fn take_list_of_negative_range() {
    assert_eq!(
      interpret(r#"StringTake["abcdefghij", {{-4, -1}}]"#).unwrap(),
      "{ghij}"
    );
  }

  #[test]
  fn take_list_of_stepped_range() {
    assert_eq!(
      interpret(r#"StringTake["abcdefghij", {{1, 8, 2}}]"#).unwrap(),
      "{aceg}"
    );
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
  fn repeated_with_count() {
    // Repeated["a", 3] means 1 to 3 repetitions
    assert_eq!(
      interpret(r#"StringMatchQ["aaa", Repeated["a", 3]]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["aa", Repeated["a", 3]]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["a", Repeated["a", 3]]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["aaaa", Repeated["a", 3]]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn repeated_with_range() {
    assert_eq!(
      interpret(r#"StringMatchQ["aaa", Repeated["a", {2, 4}]]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["a", Repeated["a", {2, 4}]]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn repeated_with_list_count() {
    assert_eq!(
      interpret(r#"StringMatchQ["aaa", Repeated["a", {3}]]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn word_character_repeated() {
    assert_eq!(
      interpret(r#"StringMatchQ["abc123DEF", Repeated[WordCharacter]]"#)
        .unwrap(),
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

  #[test]
  fn wildcard_star() {
    assert_eq!(interpret(r#"StringMatchQ["Hello", "H*"]"#).unwrap(), "True");
    assert_eq!(
      interpret(r#"StringMatchQ["Hello", "*llo"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["Hello", "H*o"]"#).unwrap(),
      "True"
    );
    assert_eq!(interpret(r#"StringMatchQ["Hello", "*"]"#).unwrap(), "True");
    assert_eq!(
      interpret(r#"StringMatchQ["Hello", "X*"]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn wildcard_at() {
    assert_eq!(interpret(r#"StringMatchQ["abc", "a@c"]"#).unwrap(), "True");
    assert_eq!(interpret(r#"StringMatchQ["aXc", "a@c"]"#).unwrap(), "False");
  }
}

mod string_expression {
  use super::*;

  #[test]
  fn parse_standalone() {
    assert_eq!(
      interpret(r#""a" ~~ __"#).unwrap(),
      r#"StringExpression[a, __]"#
    );
  }

  #[test]
  fn string_match_q_with_prefix_pattern() {
    assert_eq!(
      interpret(r#"StringMatchQ["apple", "a" ~~ __]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["banana", "a" ~~ __]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn string_match_q_with_suffix_pattern() {
    assert_eq!(
      interpret(r#"StringMatchQ["hello", __ ~~ "lo"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["hello", __ ~~ "xyz"]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn string_match_q_with_blank() {
    // _ matches exactly one character
    assert_eq!(
      interpret(r#"StringMatchQ["ab", "a" ~~ _]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["a", "a" ~~ _]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn string_match_q_with_blank_null_sequence() {
    // ___ matches zero or more characters
    assert_eq!(
      interpret(r#"StringMatchQ["a", "a" ~~ ___]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["abc", "a" ~~ ___]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn three_part_pattern() {
    assert_eq!(
      interpret(r#"StringMatchQ["abc", "a" ~~ _ ~~ "c"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["axc", "a" ~~ _ ~~ "c"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["axxc", "a" ~~ _ ~~ "c"]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn select_with_string_expression() {
    assert_eq!(
      interpret(
        r#"Select[{apple, banana, pear, apricot}, StringMatchQ[ToString[#], "a" ~~ __] &]"#
      )
      .unwrap(),
      "{apple, apricot}"
    );
  }

  #[test]
  fn with_character_classes() {
    assert_eq!(
      interpret(r#"StringMatchQ["a1", LetterCharacter ~~ DigitCharacter]"#)
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["1a", LetterCharacter ~~ DigitCharacter]"#)
        .unwrap(),
      "False"
    );
  }

  #[test]
  fn flat_associativity() {
    // String literals in ~~ are concatenated (Wolfram behavior)
    assert_eq!(interpret(r#""a" ~~ "b" ~~ "c""#).unwrap(), r#"abc"#);
  }
}

mod string_split_edge {
  use super::*;

  #[test]
  fn split_x_by_x() {
    assert_eq!(interpret(r#"StringSplit["x", "x"]"#).unwrap(), "{}");
  }

  #[test]
  fn split_filters_empty() {
    assert_eq!(interpret(r#"StringSplit["xxax", "x"]"#).unwrap(), "{a}");
  }

  #[test]
  fn split_threads_list_default() {
    assert_eq!(
      interpret(r#"StringSplit[{"ab cd", "ef gh"}]"#).unwrap(),
      "{{ab, cd}, {ef, gh}}"
    );
  }

  #[test]
  fn split_threads_list_with_delim() {
    assert_eq!(
      interpret(r#"StringSplit[{"a-b-c", "x-y"}, "-"]"#).unwrap(),
      "{{a, b, c}, {x, y}}"
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

mod compress {
  use super::*;

  #[test]
  fn compress_returns_string() {
    let result = interpret("Compress[42]").unwrap();
    assert!(result.starts_with("1:eJx"));
  }

  #[test]
  fn uncompress_roundtrip_integer() {
    assert_eq!(interpret("Uncompress[Compress[42]]").unwrap(), "42");
  }

  #[test]
  fn uncompress_roundtrip_string() {
    assert_eq!(
      interpret("Uncompress[Compress[\"hello world\"]]").unwrap(),
      "hello world"
    );
  }

  #[test]
  fn uncompress_roundtrip_list() {
    assert_eq!(
      interpret("Uncompress[Compress[{1, 2, 3}]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn uncompress_roundtrip_symbolic() {
    assert_eq!(
      interpret("Uncompress[Compress[x^2 + y Sin[x] + 10 Log[15]]]").unwrap(),
      "x^2 + 10*Log[15] + y*Sin[x]"
    );
  }

  #[test]
  fn uncompress_via_variable() {
    clear_state();
    assert_eq!(
      interpret("c = Compress[\"Mathics3 is cool\"]; Uncompress[c]").unwrap(),
      "Mathics3 is cool"
    );
  }
}

mod sequence_form {
  use super::*;

  // SequenceForm concatenates the printed forms of its arguments, showing
  // strings without quotes and rendering numbers as their input-form text.
  #[test]
  fn mixed_strings_and_numbers() {
    assert_eq!(
      interpret(r#"SequenceForm["[", "x = ", 56, "]"]"#).unwrap(),
      "[x = 56]"
    );
  }

  #[test]
  fn only_symbols() {
    assert_eq!(interpret("SequenceForm[a, b, c]").unwrap(), "abc");
  }

  #[test]
  fn only_numbers() {
    assert_eq!(interpret("SequenceForm[1, 2, 3]").unwrap(), "123");
  }
}

mod string_form {
  use super::*;

  #[test]
  fn display_with_placeholder() {
    // StringForm displays the formatted string in OutputForm
    assert_eq!(
      interpret("StringForm[\"The value is ``.\", 5]").unwrap(),
      "The value is 5."
    );
  }

  #[test]
  fn to_string_single_placeholder() {
    assert_eq!(
      interpret("ToString[StringForm[\"The value is ``.\", 5]]").unwrap(),
      "The value is 5."
    );
  }

  #[test]
  fn to_string_multiple_placeholders() {
    assert_eq!(
      interpret("ToString[StringForm[\"x=`` and y=``.\", 5, 10]]").unwrap(),
      "x=5 and y=10."
    );
  }

  #[test]
  fn to_string_indexed_placeholders() {
    assert_eq!(
      interpret("ToString[StringForm[\"`2` is `1`.\", \"dog\", \"big\"]]")
        .unwrap(),
      "big is dog."
    );
  }

  #[test]
  fn to_string_no_placeholders() {
    assert_eq!(
      interpret("ToString[StringForm[\"hello\"]]").unwrap(),
      "hello"
    );
  }

  #[test]
  fn to_string_with_list_arg() {
    assert_eq!(
      interpret("ToString[StringForm[\"x=``\", {1, 2, 3}]]").unwrap(),
      "x={1, 2, 3}"
    );
  }

  #[test]
  fn to_string_with_symbolic_arg() {
    assert_eq!(
      interpret("ToString[StringForm[\"x=``\", Pi]]").unwrap(),
      "x=Pi"
    );
  }

  #[test]
  fn to_string_three_sequential() {
    assert_eq!(
      interpret("ToString[StringForm[\"`` + `` = ``\", 1, 2, 3]]").unwrap(),
      "1 + 2 = 3"
    );
  }

  #[test]
  fn display_indexed_placeholders() {
    assert_eq!(
      interpret("StringForm[\"`1` plus `2` is `3`\", 1, 2, 3]").unwrap(),
      "1 plus 2 is 3"
    );
  }

  // Out-of-range indexed placeholders keep the `n` literal in the output
  // (instead of silently blanking it). Matches wolframscript/mathics.
  #[test]
  fn out_of_range_positive_index_kept_literal() {
    assert_eq!(interpret("StringForm[\"`2` bla\", a]").unwrap(), "`2` bla");
  }

  #[test]
  fn out_of_range_negative_index_kept_literal() {
    assert_eq!(
      interpret("StringForm[\"`-1` bla\", a]").unwrap(),
      "`-1` bla"
    );
  }

  #[test]
  fn out_of_range_sequential_placeholder_kept_literal() {
    // `` with no argument to pull from: keep the two backticks literal.
    assert_eq!(interpret("StringForm[\"x=``\"]").unwrap(), "x=``");
  }

  #[test]
  fn sequential_placeholder_resumes_from_last_numbered() {
    // `` picks up from the most recently used numbered slot + 1, not from
    // its own independent counter. `1` was the most recent; so `` -> arg 2.
    assert_eq!(
      interpret("StringForm[\"`2` bla `1` blub `` bla `3`\", a, b, c]")
        .unwrap(),
      "b bla a blub b bla c"
    );
  }

  #[test]
  fn escaped_backquote_kept_literal() {
    // \` inside the template is a literal backslash + backtick sequence;
    // Woxi/wolframscript keep both bytes verbatim in the output.
    assert_eq!(
      interpret(r#"StringForm["`` is Global\`a", a]"#).unwrap(),
      "a is Global\\`a"
    );
  }
}

mod tex_form {
  use super::*;

  #[test]
  fn simple_addition() {
    // Wolfram canonical order for same-degree terms
    assert_eq!(
      interpret("ToString[x^2 + y^2, TeXForm]").unwrap(),
      "x^2+y^2"
    );
  }

  #[test]
  fn sqrt() {
    assert_eq!(
      interpret("ToString[Sqrt[x], TeXForm]").unwrap(),
      "\\sqrt{x}"
    );
  }

  #[test]
  fn fraction() {
    assert_eq!(interpret("ToString[a/b, TeXForm]").unwrap(), "\\frac{a}{b}");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("ToString[3/4, TeXForm]").unwrap(), "\\frac{3}{4}");
  }

  #[test]
  fn sin() {
    assert_eq!(interpret("ToString[Sin[x], TeXForm]").unwrap(), "\\sin (x)");
  }

  #[test]
  fn log() {
    assert_eq!(interpret("ToString[Log[x], TeXForm]").unwrap(), "\\log (x)");
  }

  #[test]
  fn pi() {
    assert_eq!(interpret("ToString[Pi, TeXForm]").unwrap(), "\\pi");
  }

  #[test]
  fn infinity() {
    assert_eq!(interpret("ToString[Infinity, TeXForm]").unwrap(), "\\infty");
  }

  #[test]
  fn list() {
    assert_eq!(
      interpret("ToString[{a, b, c}, TeXForm]").unwrap(),
      "\\{a,b,c\\}"
    );
  }

  #[test]
  fn string() {
    assert_eq!(
      interpret("ToString[\"hello\", TeXForm]").unwrap(),
      "\\text{hello}"
    );
  }

  #[test]
  fn real_number() {
    assert_eq!(interpret("ToString[2.5, TeXForm]").unwrap(), "2.5");
  }

  #[test]
  fn power() {
    // Single-character exponents use no braces (Wolfram behavior)
    assert_eq!(interpret("ToString[x^n, TeXForm]").unwrap(), "x^n");
  }

  #[test]
  fn multiplication() {
    assert_eq!(interpret("ToString[x*y, TeXForm]").unwrap(), "x y");
  }

  #[test]
  fn subtraction() {
    assert_eq!(interpret("ToString[x - z, TeXForm]").unwrap(), "x-z");
  }

  #[test]
  fn negation() {
    assert_eq!(interpret("ToString[-x, TeXForm]").unwrap(), "-x");
  }

  #[test]
  fn abs() {
    // Wolfram uses simple bar notation for Abs
    assert_eq!(interpret("ToString[Abs[x], TeXForm]").unwrap(), "| x|");
  }

  #[test]
  fn single_letter_function_call_bare() {
    // Single-letter user functions render bare (not wrapped in \text{}),
    // matching wolframscript: ToString[f[x], TeXForm] = f(x).
    assert_eq!(interpret("ToString[f[x], TeXForm]").unwrap(), "f(x)");
  }

  #[test]
  fn integrate_of_user_function() {
    assert_eq!(
      interpret("ToString[Integrate[f[x],x], TeXForm]").unwrap(),
      "\\int f(x) \\, dx"
    );
  }

  #[test]
  fn definite_integrate_single_char_bound() {
    // Single-character bounds render without braces (\int_a^b) — matches
    // wolframscript's TeX convention.
    assert_eq!(
      interpret("ToString[Integrate[F[x], {x, a, g[b]}], TeXForm]").unwrap(),
      "\\int_a^{g(b)} F(x) \\, dx"
    );
  }

  #[test]
  fn definite_integrate_multi_char_bound() {
    // Multi-character bounds still use braces to disambiguate the
    // sub-/super-script scope. Multi-letter identifiers also pick up
    // \text{...} for safety against implicit-product confusion.
    assert_eq!(
      interpret("ToString[Integrate[F[x], {x, a1, b2}], TeXForm]").unwrap(),
      "\\int_{\\text{a1}}^{\\text{b2}} F(x) \\, dx"
    );
  }

  #[test]
  fn multi_letter_function_uses_text() {
    // Multi-letter user functions still use \text{} to avoid ambiguity with products.
    assert_eq!(
      interpret("ToString[myFunc[x], TeXForm]").unwrap(),
      "\\text{myFunc}(x)"
    );
  }

  #[test]
  fn hold_form_is_transparent() {
    // HoldForm is a display wrapper — TeXForm renders its content directly.
    assert_eq!(
      interpret("ToString[TeXForm[HoldForm[Sqrt[a^3]]]]").unwrap(),
      "\\sqrt{a^3}"
    );
  }

  #[test]
  fn output_form_then_tex_wraps_in_text() {
    // OutputForm renders its content to a textual form first; TeXForm then
    // wraps that as \text{…}, matching wolframscript.
    assert_eq!(
      interpret("ToString[b // OutputForm // TeXForm]").unwrap(),
      "\\text{b}"
    );
  }
}

mod to_expression {
  use super::*;

  #[test]
  fn single_arg_parses_and_evaluates() {
    assert_eq!(interpret("ToExpression[\"2+3\"]").unwrap(), "5");
  }

  #[test]
  fn two_args_accepts_form() {
    // Woxi's parser is form-agnostic, so the form arg is accepted but ignored.
    assert_eq!(interpret("ToExpression[\"2 3\", InputForm]").unwrap(), "6");
  }

  #[test]
  fn three_args_applies_head() {
    // The third argument is applied to the evaluated expression.
    assert_eq!(
      interpret("ToExpression[\"{2, 3, 1}\", InputForm, Max]").unwrap(),
      "3"
    );
  }
}

mod base_form {
  use super::*;

  #[test]
  fn binary() {
    // BaseForm stays as wrapper in OutputForm (matching wolframscript)
    assert_eq!(interpret("BaseForm[123, 2]").unwrap(), "BaseForm[123, 2]");
  }

  #[test]
  fn hexadecimal() {
    assert_eq!(interpret("BaseForm[255, 16]").unwrap(), "BaseForm[255, 16]");
  }

  #[test]
  fn octal() {
    assert_eq!(interpret("BaseForm[8, 8]").unwrap(), "BaseForm[8, 8]");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("BaseForm[0, 2]").unwrap(), "BaseForm[0, 2]");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("BaseForm[-42, 2]").unwrap(), "BaseForm[-42, 2]");
  }

  #[test]
  fn base_36() {
    assert_eq!(interpret("BaseForm[35, 36]").unwrap(), "BaseForm[35, 36]");
  }

  #[test]
  fn real_binary() {
    assert_eq!(interpret("BaseForm[0.5, 2]").unwrap(), "BaseForm[0.5, 2]");
  }

  #[test]
  fn real_integer_part() {
    assert_eq!(interpret("BaseForm[8., 2]").unwrap(), "BaseForm[8., 2]");
  }

  #[test]
  fn large_integer() {
    assert_eq!(interpret("BaseForm[256, 16]").unwrap(), "BaseForm[256, 16]");
  }

  #[test]
  fn unevaluated_symbolic() {
    assert_eq!(interpret("BaseForm[x, 2]").unwrap(), "BaseForm[x, 2]");
  }
}

mod c_form {
  use super::*;

  #[test]
  fn polynomial() {
    // CForm wraps in OutputForm, matching wolframscript
    assert_eq!(
      interpret("CForm[x^2 + 2 x + 1]").unwrap(),
      "CForm[1 + 2*x + x^2]"
    );
  }

  #[test]
  fn trig_functions() {
    assert_eq!(
      interpret("CForm[Sin[x] + Cos[y]]").unwrap(),
      "CForm[Cos[y] + Sin[x]]"
    );
  }

  #[test]
  fn pi_constant() {
    assert_eq!(interpret("CForm[Pi]").unwrap(), "CForm[Pi]");
  }

  #[test]
  fn e_constant() {
    assert_eq!(interpret("CForm[E]").unwrap(), "CForm[E]");
  }

  #[test]
  fn sqrt() {
    assert_eq!(interpret("CForm[Sqrt[x]]").unwrap(), "CForm[Sqrt[x]]");
  }

  #[test]
  fn division() {
    assert_eq!(interpret("CForm[1/x]").unwrap(), "CForm[x^(-1)]");
  }

  #[test]
  fn division_to_string() {
    // ToString[CForm[1/x], InputForm] produces C division notation
    assert_eq!(interpret("ToString[CForm[1/x], InputForm]").unwrap(), "1/x");
  }

  #[test]
  fn to_string_form() {
    // ToString[expr, CForm] produces the actual C representation
    assert_eq!(
      interpret("ToString[x^2 + 1, CForm]").unwrap(),
      "1 + Power(x,2)"
    );
  }

  #[test]
  fn exp_function() {
    // CForm wraps, Exp[x] evaluates to E^x
    assert_eq!(interpret("CForm[Exp[x]]").unwrap(), "CForm[E^x]");
  }
}

mod tex_form_standalone {
  use super::*;

  #[test]
  fn wraps_in_output_form() {
    // TeXForm wraps in OutputForm, matching wolframscript
    assert_eq!(interpret("TeXForm[1 + x^2]").unwrap(), "TeXForm[1 + x^2]");
  }

  #[test]
  fn to_string_extracts_tex() {
    assert_eq!(interpret("ToString[TeXForm[1 + x^2]]").unwrap(), "x^2+1");
  }

  #[test]
  fn pi_constant() {
    assert_eq!(interpret("TeXForm[Pi]").unwrap(), "TeXForm[Pi]");
  }

  #[test]
  fn to_string_pi() {
    assert_eq!(interpret("ToString[TeXForm[Pi]]").unwrap(), "\\pi");
  }

  #[test]
  fn sqrt() {
    assert_eq!(interpret("TeXForm[Sqrt[x]]").unwrap(), "TeXForm[Sqrt[x]]");
  }

  #[test]
  fn negative_power_reciprocal() {
    assert_eq!(
      interpret("ToString[TeXForm[Power[a,-1]]]").unwrap(),
      "\\frac{1}{a}"
    );
  }

  #[test]
  fn negative_power_squared() {
    assert_eq!(
      interpret("ToString[TeXForm[Power[a,-2]]]").unwrap(),
      "\\frac{1}{a^2}"
    );
  }

  #[test]
  fn negative_power_compound_base() {
    assert_eq!(
      interpret("ToString[TeXForm[Power[x+1,-2]]]").unwrap(),
      "\\frac{1}{(x+1)^2}"
    );
  }

  #[test]
  fn negative_power_compound_base_reciprocal() {
    assert_eq!(
      interpret("ToString[TeXForm[Power[x+1,-1]]]").unwrap(),
      "\\frac{1}{x+1}"
    );
  }

  #[test]
  fn times_with_negative_power() {
    assert_eq!(
      interpret("ToString[TeXForm[2*Power[a,-1]]]").unwrap(),
      "\\frac{2}{a}"
    );
  }

  #[test]
  fn times_symbolic_over_symbolic() {
    assert_eq!(
      interpret("ToString[TeXForm[a*Power[b,-1]]]").unwrap(),
      "\\frac{a}{b}"
    );
  }

  #[test]
  fn times_multiple_denom_factors() {
    assert_eq!(
      interpret("ToString[TeXForm[a*Power[b,-1]*Power[c,-1]]]").unwrap(),
      "\\frac{a}{b c}"
    );
  }

  #[test]
  fn negative_symbolic_power_unchanged() {
    assert_eq!(
      interpret("ToString[TeXForm[Power[a,-n]]]").unwrap(),
      "a^{-n}"
    );
  }
}

mod fortran_form {
  use super::*;

  #[test]
  fn wraps_in_output_form() {
    // FortranForm wraps in OutputForm, matching wolframscript
    assert_eq!(
      interpret("FortranForm[1 + x^2]").unwrap(),
      "FortranForm[1 + x^2]"
    );
  }

  #[test]
  fn to_string_extracts_fortran() {
    assert_eq!(
      interpret("ToString[FortranForm[1 + x^2]]").unwrap(),
      "1 + x**2"
    );
  }

  #[test]
  fn power() {
    assert_eq!(interpret("ToString[x^2, FortranForm]").unwrap(), "x**2");
  }

  #[test]
  fn multiplication() {
    assert_eq!(interpret("ToString[x*y, FortranForm]").unwrap(), "x*y");
  }

  #[test]
  fn sqrt() {
    assert_eq!(
      interpret("ToString[Sqrt[x], FortranForm]").unwrap(),
      "Sqrt(x)"
    );
  }

  #[test]
  fn trig() {
    assert_eq!(
      interpret("ToString[Sin[x], FortranForm]").unwrap(),
      "Sin(x)"
    );
  }

  #[test]
  fn list() {
    assert_eq!(
      interpret("ToString[{1, 2, 3}, FortranForm]").unwrap(),
      "List(1,2,3)"
    );
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("ToString[3/4, FortranForm]").unwrap(), "0.75");
  }

  #[test]
  fn addition() {
    assert_eq!(
      interpret("ToString[x + y + z, FortranForm]").unwrap(),
      "x + y + z"
    );
  }

  #[test]
  fn negation() {
    // -x evaluates to Times[-1, x]
    assert_eq!(interpret("ToString[-x, FortranForm]").unwrap(), "-x");
  }

  #[test]
  fn division() {
    assert_eq!(interpret("ToString[x/y, FortranForm]").unwrap(), "x/y");
  }

  #[test]
  fn polynomial() {
    assert_eq!(
      interpret("ToString[x^2 + x + 1, FortranForm]").unwrap(),
      "1 + x + x**2"
    );
  }

  #[test]
  fn to_string_form() {
    // ToString[expr, FortranForm] produces the Fortran representation
    assert_eq!(
      interpret("ToString[x^2 + 1, FortranForm]").unwrap(),
      "1 + x**2"
    );
  }

  #[test]
  fn exp_function() {
    assert_eq!(
      interpret("FortranForm[Exp[x]]").unwrap(),
      "FortranForm[E^x]"
    );
  }

  #[test]
  fn real_number() {
    assert_eq!(interpret("ToString[2.5, FortranForm]").unwrap(), "2.5");
  }
}

mod to_boxes {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("ToBoxes[42]").unwrap(), "42");
  }

  #[test]
  fn symbol() {
    assert_eq!(interpret("ToBoxes[x]").unwrap(), "x");
  }

  #[test]
  fn string_literal() {
    assert_eq!(interpret("ToBoxes[\"hello\"]").unwrap(), "\"hello\"");
  }

  #[test]
  fn plus() {
    assert_eq!(interpret("ToBoxes[x + y]").unwrap(), "RowBox[{x, +, y}]");
  }

  #[test]
  fn subtraction() {
    assert_eq!(interpret("ToBoxes[x - y]").unwrap(), "RowBox[{x, -, y}]");
  }

  #[test]
  fn negation() {
    assert_eq!(interpret("ToBoxes[-x]").unwrap(), "RowBox[{-, x}]");
  }

  #[test]
  fn times() {
    assert_eq!(interpret("ToBoxes[x * y]").unwrap(), "RowBox[{x,  , y}]");
  }

  #[test]
  fn division() {
    assert_eq!(interpret("ToBoxes[x / y]").unwrap(), "FractionBox[x, y]");
  }

  #[test]
  fn power() {
    assert_eq!(
      interpret("ToBoxes[a + b^2]").unwrap(),
      "RowBox[{a, +, SuperscriptBox[b, 2]}]"
    );
  }

  #[test]
  fn sqrt() {
    assert_eq!(interpret("ToBoxes[Sqrt[x]]").unwrap(), "SqrtBox[x]");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("ToBoxes[2/3]").unwrap(), "FractionBox[2, 3]");
  }

  #[test]
  fn list() {
    assert_eq!(
      interpret("ToBoxes[{1, 2, 3}]").unwrap(),
      "RowBox[{{, RowBox[{1, ,, 2, ,, 3}], }}]"
    );
  }

  #[test]
  fn function_call() {
    assert_eq!(
      interpret("ToBoxes[f[x, y]]").unwrap(),
      "RowBox[{f, [, RowBox[{x, ,, y}], ]}]"
    );
  }

  #[test]
  fn function_call_single_arg() {
    assert_eq!(interpret("ToBoxes[f[x]]").unwrap(), "RowBox[{f, [, x, ]}]");
  }

  #[test]
  fn function_call_no_args() {
    assert_eq!(interpret("ToBoxes[f[]]").unwrap(), "RowBox[{f, [, ]}]");
  }

  #[test]
  fn boolean() {
    assert_eq!(interpret("ToBoxes[True]").unwrap(), "True");
  }

  #[test]
  fn evaluated_expression() {
    // ToBoxes evaluates its argument first
    assert_eq!(interpret("ToBoxes[1 + 2]").unwrap(), "3");
  }

  #[test]
  fn subscript_box() {
    assert_eq!(
      interpret("ToBoxes[Subscript[x, 0]]").unwrap(),
      "SubscriptBox[x, 0]"
    );
    assert_eq!(
      interpret("ToBoxes[Subscript[a, b]]").unwrap(),
      "SubscriptBox[a, b]"
    );
  }

  #[test]
  fn subsuperscript_box() {
    // Power[Subscript[x, sub], exp] → SubsuperscriptBox
    assert_eq!(
      interpret("ToBoxes[Subscript[a, b]^c]").unwrap(),
      "SubsuperscriptBox[a, b, c]"
    );
    assert_eq!(
      interpret("ToBoxes[Subscript[x, 0]^n]").unwrap(),
      "SubsuperscriptBox[x, 0, n]"
    );
    // Special rational exponents still use SqrtBox/FractionBox with SubscriptBox
    assert_eq!(
      interpret("ToBoxes[Subscript[a, b]^(1/2)]").unwrap(),
      "SqrtBox[SubscriptBox[a, b]]"
    );
    assert_eq!(
      interpret("ToBoxes[Subscript[a, b]^(-1/2)]").unwrap(),
      "FractionBox[1, SqrtBox[SubscriptBox[a, b]]]"
    );
  }

  #[test]
  fn subsuperscript_box_unevaluated() {
    // SubsuperscriptBox stays unevaluated as a symbolic head
    assert_eq!(
      interpret("SubsuperscriptBox[\"x\", \"0\", \"n\"]").unwrap(),
      "SubsuperscriptBox[x, 0, n]"
    );
    assert_eq!(
      interpret("Head[SubsuperscriptBox[\"x\", \"0\", \"n\"]]").unwrap(),
      "SubsuperscriptBox"
    );
  }

  // Graphics / Graphics3D get dedicated box wrappers, so Head[ToBoxes[...]]
  // returns the specific *Box head rather than the generic RowBox.
  #[test]
  fn graphics_box_head() {
    assert_eq!(
      interpret("Head[ToBoxes[Graphics[{Circle[]}]]]").unwrap(),
      "GraphicsBox"
    );
  }

  #[test]
  fn graphics3d_box_head() {
    assert_eq!(
      interpret("Head[ToBoxes[Graphics3D[{Polygon[]}]]]").unwrap(),
      "Graphics3DBox"
    );
  }
}

mod make_boxes {
  use super::*;

  #[test]
  fn make_boxes_integer() {
    assert_eq!(interpret("MakeBoxes[42]").unwrap(), "42");
  }

  #[test]
  fn make_boxes_symbol() {
    assert_eq!(interpret("MakeBoxes[x]").unwrap(), "x");
  }

  #[test]
  fn make_boxes_power() {
    assert_eq!(interpret("MakeBoxes[x^2]").unwrap(), "SuperscriptBox[x, 2]");
  }

  #[test]
  fn make_boxes_plus() {
    assert_eq!(interpret("MakeBoxes[a + b]").unwrap(), "RowBox[{a, +, b}]");
  }

  #[test]
  fn make_boxes_times() {
    assert_eq!(interpret("MakeBoxes[a b]").unwrap(), "RowBox[{a,  , b}]");
  }

  #[test]
  fn make_boxes_fraction() {
    assert_eq!(interpret("MakeBoxes[a/b]").unwrap(), "FractionBox[a, b]");
  }

  #[test]
  fn make_boxes_sqrt() {
    assert_eq!(interpret("MakeBoxes[Sqrt[x]]").unwrap(), "SqrtBox[x]");
  }

  #[test]
  fn make_boxes_list() {
    assert_eq!(
      interpret("MakeBoxes[{1, 2, 3}]").unwrap(),
      "RowBox[{{, RowBox[{1, ,, 2, ,, 3}], }}]"
    );
  }

  #[test]
  fn make_boxes_function_call() {
    assert_eq!(
      interpret("MakeBoxes[f[x, y]]").unwrap(),
      "RowBox[{f, [, RowBox[{x, ,, y}], ]}]"
    );
  }

  #[test]
  fn make_boxes_holds_argument() {
    // MakeBoxes should NOT evaluate its argument
    assert_eq!(interpret("MakeBoxes[1 + 2]").unwrap(), "RowBox[{1, +, 2}]");
  }

  #[test]
  fn make_boxes_rational() {
    assert_eq!(interpret("MakeBoxes[2/3]").unwrap(), "FractionBox[2, 3]");
  }

  #[test]
  fn make_boxes_subscript() {
    assert_eq!(
      interpret("MakeBoxes[Subscript[x, 0]]").unwrap(),
      "SubscriptBox[x, 0]"
    );
  }

  #[test]
  fn make_boxes_subsuperscript() {
    assert_eq!(
      interpret("MakeBoxes[Subscript[a, b]^c]").unwrap(),
      "SubsuperscriptBox[a, b, c]"
    );
  }
}

mod raw_boxes {
  use super::*;

  #[test]
  fn raw_boxes_identity() {
    // RawBoxes wraps its content without modification
    assert_eq!(
      interpret(r#"RawBoxes[SuperscriptBox["x", "2"]]"#).unwrap(),
      "RawBoxes[SuperscriptBox[x, 2]]"
    );
  }

  #[test]
  fn raw_boxes_with_make_boxes() {
    // RawBoxes[MakeBoxes[...]] should work end-to-end
    assert_eq!(
      interpret("RawBoxes[MakeBoxes[x^2]]").unwrap(),
      "RawBoxes[SuperscriptBox[x, 2]]"
    );
  }
}

mod display_form {
  use super::*;

  #[test]
  fn display_form_identity() {
    // DisplayForm wraps its content without modification
    assert_eq!(
      interpret(r#"DisplayForm[SuperscriptBox["x", "2"]]"#).unwrap(),
      "DisplayForm[SuperscriptBox[x, 2]]"
    );
  }

  #[test]
  fn display_form_head() {
    assert_eq!(
      interpret(r#"Head[DisplayForm[SuperscriptBox["x", "2"]]]"#).unwrap(),
      "DisplayForm"
    );
  }

  #[test]
  fn display_form_with_make_boxes() {
    assert_eq!(
      interpret("DisplayForm[MakeBoxes[x^2]]").unwrap(),
      "DisplayForm[SuperscriptBox[x, 2]]"
    );
  }

  #[test]
  fn display_form_subscript_box() {
    assert_eq!(
      interpret(r#"DisplayForm[SubscriptBox["a", "i"]]"#).unwrap(),
      "DisplayForm[SubscriptBox[a, i]]"
    );
  }

  #[test]
  fn display_form_row_box() {
    assert_eq!(
      interpret(
        r#"DisplayForm[RowBox[{SubscriptBox["a", "1"], SubscriptBox["b", "2"]}]]"#
      )
      .unwrap(),
      "DisplayForm[RowBox[{SubscriptBox[a, 1], SubscriptBox[b, 2]}]]"
    );
  }

  #[test]
  fn display_form_fraction_box() {
    assert_eq!(
      interpret(r#"DisplayForm[FractionBox["x", "y"]]"#).unwrap(),
      "DisplayForm[FractionBox[x, y]]"
    );
  }

  #[test]
  fn display_form_sqrt_box() {
    assert_eq!(
      interpret(r#"DisplayForm[SqrtBox["x"]]"#).unwrap(),
      "DisplayForm[SqrtBox[x]]"
    );
  }

  #[test]
  fn display_form_complex_expression() {
    // RawBoxes[MakeBoxes[...]] // DisplayForm — end-to-end
    assert_eq!(
      interpret("DisplayForm[MakeBoxes[a + b]]").unwrap(),
      "DisplayForm[RowBox[{a, +, b}]]"
    );
  }
}

mod template_apply {
  use super::*;

  #[test]
  fn basic_list() {
    assert_eq!(
      interpret(r#"TemplateApply["Hello `1`", {"World"}]"#).unwrap(),
      "Hello World"
    );
  }

  #[test]
  fn multiple_slots() {
    assert_eq!(
      interpret(r#"TemplateApply["`1` + `2` = `3`", {1, 2, 3}]"#).unwrap(),
      "1 + 2 = 3"
    );
  }

  #[test]
  fn repeated_slot() {
    assert_eq!(
      interpret(r#"TemplateApply["`1` and `1`", {"x"}]"#).unwrap(),
      "x and x"
    );
  }

  #[test]
  fn no_slots() {
    assert_eq!(
      interpret(r#"TemplateApply["no slots here", {}]"#).unwrap(),
      "no slots here"
    );
  }

  #[test]
  fn association_args() {
    assert_eq!(
      interpret(
        r#"TemplateApply["Hi `name`, you are `age`", <|"name" -> "Alice", "age" -> 30|>]"#
      )
      .unwrap(),
      "Hi Alice, you are 30"
    );
  }

  #[test]
  fn non_string_template() {
    assert_eq!(interpret(r#"TemplateApply[42, {1}]"#).unwrap(), "42");
  }

  #[test]
  fn positional_slots() {
    // Double backtick `` is a positional slot
    assert_eq!(
      interpret(r#"TemplateApply["Hello ``!", {"World"}]"#).unwrap(),
      "Hello World!"
    );
    assert_eq!(
      interpret(r#"TemplateApply["`` + `` = ``", {1, 2, 3}]"#).unwrap(),
      "1 + 2 = 3"
    );
  }
}

mod dictionary_word_q {
  use super::*;

  #[test]
  fn common_word() {
    assert_eq!(interpret(r#"DictionaryWordQ["dolphin"]"#).unwrap(), "True");
  }

  #[test]
  fn nonsense_word() {
    assert_eq!(
      interpret(r#"DictionaryWordQ["beltalowda"]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn case_insensitive() {
    assert_eq!(interpret(r#"DictionaryWordQ["Hello"]"#).unwrap(), "True");
  }

  #[test]
  fn all_caps() {
    assert_eq!(interpret(r#"DictionaryWordQ["HELLO"]"#).unwrap(), "True");
  }

  #[test]
  fn empty_string() {
    assert_eq!(interpret(r#"DictionaryWordQ[""]"#).unwrap(), "True");
  }

  #[test]
  fn multi_word_phrase() {
    assert_eq!(
      interpret(r#"DictionaryWordQ["ice cream"]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn single_letter() {
    assert_eq!(interpret(r#"DictionaryWordQ["a"]"#).unwrap(), "True");
  }

  #[test]
  fn non_string_argument() {
    assert_eq!(
      interpret(r#"DictionaryWordQ[123]"#).unwrap(),
      "DictionaryWordQ[123]"
    );
  }
}

mod url_encode_tests {
  use super::*;

  #[test]
  fn url_encode_simple_string() {
    assert_eq!(
      interpret(r#"URLEncode["Hello, World!"]"#).unwrap(),
      "Hello%2C%20World%21"
    );
  }

  #[test]
  fn url_encode_spaces() {
    assert_eq!(
      interpret(r#"URLEncode["hello world"]"#).unwrap(),
      "hello%20world"
    );
  }

  #[test]
  fn url_encode_special_chars() {
    assert_eq!(
      interpret(r#"URLEncode["a=1&b=2"]"#).unwrap(),
      "a%3D1%26b%3D2"
    );
  }

  #[test]
  fn url_encode_none() {
    assert_eq!(interpret("URLEncode[None]").unwrap(), "");
  }

  #[test]
  fn url_encode_integer() {
    assert_eq!(interpret("URLEncode[1]").unwrap(), "1");
  }

  #[test]
  fn url_encode_real() {
    assert_eq!(interpret("URLEncode[1.3]").unwrap(), "1.3");
  }

  #[test]
  fn url_encode_threads_over_list() {
    assert_eq!(
      interpret(r#"URLEncode[{"a", "b c"}]"#).unwrap(),
      "{a, b%20c}"
    );
  }

  #[test]
  fn url_encode_unreserved_chars() {
    assert_eq!(
      interpret(r#"URLEncode["abc123-._~"]"#).unwrap(),
      "abc123-._~"
    );
  }
}

mod url_decode_tests {
  use super::*;

  #[test]
  fn url_decode_simple() {
    assert_eq!(
      interpret(r#"URLDecode["Hello%2C%20World%21"]"#).unwrap(),
      "Hello, World!"
    );
  }

  #[test]
  fn url_decode_special_chars() {
    assert_eq!(
      interpret(r#"URLDecode["a%3D1%26b%3D2"]"#).unwrap(),
      "a=1&b=2"
    );
  }

  #[test]
  fn url_decode_roundtrip() {
    assert_eq!(
      interpret(r#"URLDecode[URLEncode["test string!@#"]]"#).unwrap(),
      "test string!@#"
    );
  }
}

mod string_trim {
  use super::*;

  #[test]
  fn trim_whitespace() {
    assert_eq!(interpret(r#"StringTrim["  hello  "]"#).unwrap(), "hello");
  }

  #[test]
  fn trim_with_pattern_removes_one_occurrence() {
    // StringTrim removes only one occurrence of the pattern from each end
    assert_eq!(
      interpret(r#"StringTrim["xxxhelloxxx", "x"]"#).unwrap(),
      "xxhelloxx"
    );
    assert_eq!(
      interpret(r#"StringTrim["xxxhelloxxx", "xx"]"#).unwrap(),
      "xhellox"
    );
    assert_eq!(
      interpret(r#"StringTrim["   hello   ", " "]"#).unwrap(),
      "  hello  "
    );
  }

  #[test]
  fn trim_with_repeated_pattern() {
    // StringTrim with Repeated pattern strips all matching from each end
    assert_eq!(
      interpret(r#"StringTrim["xxxhelloxxx", "x"..]"#).unwrap(),
      "hello"
    );
  }

  #[test]
  fn trim_with_whitespace_pattern() {
    assert_eq!(
      interpret(r#"StringTrim["  hello  ", Whitespace]"#).unwrap(),
      "hello"
    );
  }

  #[test]
  fn trim_with_digit_pattern() {
    assert_eq!(
      interpret(r#"StringTrim["123hello456", DigitCharacter..]"#).unwrap(),
      "hello"
    );
  }

  #[test]
  fn trim_threads_list() {
    assert_eq!(
      interpret(r#"StringTrim[{"  abc  ", " def "}]"#).unwrap(),
      "{abc, def}"
    );
  }

  #[test]
  fn trim_threads_list_with_pattern() {
    assert_eq!(
      interpret(r#"StringTrim[{"xxhixx", "xyx"}, "x"]"#).unwrap(),
      "{xhix, y}"
    );
  }
}

mod longest_common_subsequence_tests {
  use woxi::interpret;

  #[test]
  fn basic() {
    // Wolfram's LongestCommonSubsequence finds the longest common substring (contiguous)
    assert_eq!(
      interpret(r#"LongestCommonSubsequence["ABCDE", "ACDBE"]"#).unwrap(),
      "CD"
    );
  }

  #[test]
  fn identical_strings() {
    assert_eq!(
      interpret(r#"LongestCommonSubsequence["abc", "abc"]"#).unwrap(),
      "abc"
    );
  }

  #[test]
  fn no_common() {
    assert_eq!(
      interpret(r#"LongestCommonSubsequence["abc", "xyz"]"#).unwrap(),
      ""
    );
  }

  #[test]
  fn longest_common_substring() {
    // Wolfram's LongestCommonSubsequence finds contiguous common substring
    assert_eq!(
      interpret(r#"LongestCommonSubsequence["abcdef", "acbcf"]"#).unwrap(),
      "bc"
    );
  }
}

mod string_count_patterns {
  use woxi::interpret;

  #[test]
  fn count_with_regex() {
    assert_eq!(
      interpret(r#"StringCount["hello world", RegularExpression["[aeiou]"]]"#)
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn count_with_digit_character() {
    assert_eq!(
      interpret(r#"StringCount["abc123def456", DigitCharacter]"#).unwrap(),
      "6"
    );
  }

  #[test]
  fn count_plain_string() {
    assert_eq!(interpret(r#"StringCount["abcabc", "a"]"#).unwrap(), "2");
  }

  #[test]
  fn count_list_of_patterns_single_chars() {
    // A list of patterns is treated as Alternatives.
    assert_eq!(
      interpret(r#"StringCount["abcabc", {"a", "b"}]"#).unwrap(),
      "4"
    );
  }

  #[test]
  fn count_list_of_patterns_multi_char() {
    assert_eq!(
      interpret(r#"StringCount["abcabcabc", {"ab", "bc"}]"#).unwrap(),
      "3"
    );
  }

  #[test]
  fn count_list_of_patterns_non_overlapping_chars() {
    assert_eq!(
      interpret(r#"StringCount["abcabcabc", {"a", "c"}]"#).unwrap(),
      "6"
    );
  }

  #[test]
  fn count_threads_over_list_of_strings() {
    assert_eq!(
      interpret(r#"StringCount[{"abc", "abcabc", "xyz"}, "a"]"#).unwrap(),
      "{1, 2, 0}"
    );
  }

  #[test]
  fn count_threads_over_list_of_strings_with_list_pattern() {
    assert_eq!(
      interpret(r#"StringCount[{"abc", "abcabc"}, {"a", "b"}]"#).unwrap(),
      "{2, 4}"
    );
  }
}

mod string_starts_ends_patterns {
  use woxi::interpret;

  #[test]
  fn starts_q_letter_character() {
    assert_eq!(
      interpret(r#"StringStartsQ["hello", LetterCharacter]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringStartsQ["123hello", LetterCharacter]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn ends_q_digit_character() {
    assert_eq!(
      interpret(r#"StringEndsQ["hello123", DigitCharacter]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringEndsQ["hello", DigitCharacter]"#).unwrap(),
      "False"
    );
  }
}

mod string_contains_free_patterns {
  use woxi::interpret;

  #[test]
  fn contains_q_digit_character() {
    assert_eq!(
      interpret(r#"StringContainsQ["hello123", DigitCharacter]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringContainsQ["hello", DigitCharacter]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn free_q_regex() {
    assert_eq!(
      interpret(r#"StringFreeQ["hello123", RegularExpression["[0-9]"]]"#)
        .unwrap(),
      "False"
    );
    assert_eq!(
      interpret(r#"StringFreeQ["hello", RegularExpression["[0-9]"]]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn free_q_threads_over_list() {
    // StringFreeQ threads over a list of strings (matches wolframscript).
    assert_eq!(
      interpret(r#"StringFreeQ[{"g", "a", "laxy", "universe", "sun"}, "u"]"#)
        .unwrap(),
      "{True, True, True, False, False}"
    );
  }

  #[test]
  fn free_q_ignore_case() {
    assert_eq!(
      interpret(r#"StringFreeQ["Mathics", "MA", IgnoreCase -> True]"#).unwrap(),
      "False"
    );
    assert_eq!(
      interpret(r#"StringFreeQ["Mathics", "MA"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringFreeQ["Mathics", "XX", IgnoreCase -> True]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringFreeQ[{"abc", "ABC"}, "a", IgnoreCase -> True]"#)
        .unwrap(),
      "{False, False}"
    );
  }

  #[test]
  fn split_digit_pattern() {
    assert_eq!(
      interpret(r#"StringSplit["abc123def456", DigitCharacter..]"#).unwrap(),
      "{abc, def}"
    );
  }

  #[test]
  fn string_pad_right_single_char() {
    assert_eq!(
      interpret(r#"StringPadRight["hi", 5, "0"]"#).unwrap(),
      "hi000"
    );
  }

  #[test]
  fn string_pad_right_multi_char() {
    assert_eq!(
      interpret(r#"StringPadRight["hi", 10, "xy"]"#).unwrap(),
      "hixyxyxyxy"
    );
    assert_eq!(
      interpret(r#"StringPadRight["x", 6, "abc"]"#).unwrap(),
      "xbcabc"
    );
    assert_eq!(
      interpret(r#"StringPadRight["hi", 10, "abc"]"#).unwrap(),
      "hicabcabca"
    );
  }

  #[test]
  fn string_pad_right_default() {
    assert_eq!(interpret(r#"StringPadRight["hi", 5]"#).unwrap(), "hi   ");
  }

  #[test]
  fn string_pad_right_truncate() {
    assert_eq!(interpret(r#"StringPadRight["hello", 3]"#).unwrap(), "hel");
  }

  #[test]
  fn string_pad_left_single_char() {
    assert_eq!(
      interpret(r#"StringPadLeft["hi", 5, "0"]"#).unwrap(),
      "000hi"
    );
  }

  #[test]
  fn string_pad_left_multi_char() {
    assert_eq!(
      interpret(r#"StringPadLeft["hi", 10, "xy"]"#).unwrap(),
      "xyxyxyxyhi"
    );
    assert_eq!(
      interpret(r#"StringPadLeft["x", 6, "abc"]"#).unwrap(),
      "abcabx"
    );
    assert_eq!(
      interpret(r#"StringPadLeft["hi", 10, "abc"]"#).unwrap(),
      "cabcabcahi"
    );
  }

  #[test]
  fn string_pad_left_default() {
    assert_eq!(interpret(r#"StringPadLeft["hi", 5]"#).unwrap(), "   hi");
  }

  #[test]
  fn string_pad_left_truncate() {
    assert_eq!(interpret(r#"StringPadLeft["hello", 3]"#).unwrap(), "llo");
  }

  #[test]
  fn string_pad_left_list_default() {
    assert_eq!(
      interpret(r#"StringPadLeft[{"a", "bc", "def"}, 5]"#).unwrap(),
      "{    a,    bc,   def}"
    );
  }

  #[test]
  fn string_pad_left_list_with_pad() {
    assert_eq!(
      interpret(r#"StringPadLeft[{"a", "bc", "def"}, 5, "*"]"#).unwrap(),
      "{****a, ***bc, **def}"
    );
  }

  #[test]
  fn string_pad_left_list_multi_char_pad() {
    assert_eq!(
      interpret(r#"StringPadLeft[{"a", "bc"}, 6, "xy"]"#).unwrap(),
      "{xyxyxa, xyxybc}"
    );
  }

  #[test]
  fn string_pad_right_list_default() {
    assert_eq!(
      interpret(r#"StringPadRight[{"a", "bc", "def"}, 5]"#).unwrap(),
      "{a    , bc   , def  }"
    );
  }

  #[test]
  fn string_pad_right_list_with_pad() {
    assert_eq!(
      interpret(r#"StringPadRight[{"a", "bc", "def"}, 5, "-"]"#).unwrap(),
      "{a----, bc---, def--}"
    );
  }

  #[test]
  fn string_pad_left_list_truncate() {
    assert_eq!(
      interpret(r#"StringPadLeft[{"hello", "hi"}, 3]"#).unwrap(),
      "{llo,  hi}"
    );
  }
}

mod string_position_alternatives {
  use super::*;

  #[test]
  fn string_position_list_of_alternatives() {
    // Matches of "a" and "b" should interleave and be sorted by position.
    assert_eq!(
      interpret(r#"StringPosition["abcabc", {"a", "b"}]"#).unwrap(),
      "{{1, 1}, {2, 2}, {4, 4}, {5, 5}}"
    );
  }

  #[test]
  fn string_position_alternatives_mixed_lengths() {
    // "a" has length 1, "bc" has length 2.
    assert_eq!(
      interpret(r#"StringPosition["abcabc", {"a", "bc"}]"#).unwrap(),
      "{{1, 1}, {2, 3}, {4, 4}, {5, 6}}"
    );
  }

  #[test]
  fn string_position_alternatives_with_limit() {
    assert_eq!(
      interpret(r#"StringPosition["abcabcabc", {"bc", "ab"}, 1]"#).unwrap(),
      "{{1, 2}}"
    );
  }

  #[test]
  fn string_position_alternatives_sorted_by_position() {
    assert_eq!(
      interpret(r#"StringPosition["abcdefabc", {"d", "a"}]"#).unwrap(),
      "{{1, 1}, {4, 4}, {7, 7}}"
    );
  }

  #[test]
  fn string_position_threads_over_list_of_strings() {
    assert_eq!(
      interpret(r#"StringPosition[{"abcabc", "xyabc"}, "b"]"#).unwrap(),
      "{{{2, 2}, {5, 5}}, {{4, 4}}}"
    );
  }

  #[test]
  fn string_position_threads_list_of_strings_with_alternatives() {
    assert_eq!(
      interpret(r#"StringPosition[{"abcabc", "xyabc"}, {"a", "b"}]"#).unwrap(),
      "{{{1, 1}, {2, 2}, {4, 4}, {5, 5}}, {{3, 3}, {4, 4}}}"
    );
  }

  #[test]
  fn string_position_with_regex() {
    assert_eq!(
      interpret(r#"StringPosition["hello", RegularExpression["l+"]]"#).unwrap(),
      "{{3, 4}, {4, 4}}"
    );
  }

  #[test]
  fn string_position_with_regex_single_char() {
    assert_eq!(
      interpret(r#"StringPosition["hello", RegularExpression["l"]]"#).unwrap(),
      "{{3, 3}, {4, 4}}"
    );
  }

  #[test]
  fn string_position_with_regex_overlapping() {
    assert_eq!(
      interpret(r#"StringPosition["aabaa", RegularExpression["a+"]]"#).unwrap(),
      "{{1, 2}, {2, 2}, {4, 5}, {5, 5}}"
    );
  }
}

mod edit_distance_options {
  use super::*;

  #[test]
  fn ignore_case_option() {
    // EditDistance with IgnoreCase treats upper/lower as equal (matches wolframscript).
    assert_eq!(
      interpret(r#"EditDistance["time", "Thyme", IgnoreCase -> True]"#)
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn list_of_items() {
    // EditDistance accepts lists and compares elementwise by equality.
    assert_eq!(
      interpret("EditDistance[{1, E, 2, Pi}, {1, E, Pi, 2}]").unwrap(),
      "2"
    );
  }
}

mod damerau_levenshtein_distance {
  use super::*;

  #[test]
  fn basic_substitution_insertion() {
    assert_eq!(
      interpret(r#"DamerauLevenshteinDistance["kitten", "kitchen"]"#).unwrap(),
      "2"
    );
  }

  #[test]
  fn deletion() {
    assert_eq!(
      interpret(r#"DamerauLevenshteinDistance["abc", "ac"]"#).unwrap(),
      "1"
    );
  }

  #[test]
  fn adjacent_transposition_is_one() {
    // DL distinguishes itself from plain Levenshtein by treating a swap of
    // adjacent characters as cost 1 (Levenshtein would say 2).
    assert_eq!(
      interpret(r#"DamerauLevenshteinDistance["abc", "acb"]"#).unwrap(),
      "1"
    );
  }

  #[test]
  fn mixed_insertion_transposition() {
    assert_eq!(
      interpret(r#"DamerauLevenshteinDistance["azbc", "abxyc"]"#).unwrap(),
      "3"
    );
  }

  #[test]
  fn case_sensitive() {
    assert_eq!(
      interpret(r#"DamerauLevenshteinDistance["time", "Thyme"]"#).unwrap(),
      "3"
    );
  }

  #[test]
  fn ignore_case_option() {
    assert_eq!(
      interpret(
        r#"DamerauLevenshteinDistance["time", "Thyme", IgnoreCase -> True]"#
      )
      .unwrap(),
      "2"
    );
  }

  #[test]
  fn list_arguments_transposition() {
    assert_eq!(
      interpret("DamerauLevenshteinDistance[{1, E, 2, Pi}, {1, E, Pi, 2}]")
        .unwrap(),
      "1"
    );
  }
}

mod string_position_anchors {
  use super::*;

  // EndOfString anchors a match to the end of the input string.
  #[test]
  fn match_end_of_string() {
    assert_eq!(
      interpret(
        r#"StringMatchQ[#, __ ~~ "e" ~~ EndOfString] &/@ {"apple", "banana", "artichoke"}"#
      )
      .unwrap(),
      "{True, False, True}"
    );
  }

  // StartOfString anchors a match to the beginning of the input string.
  #[test]
  fn match_start_of_string() {
    assert_eq!(
      interpret(
        r#"StringMatchQ[#, StartOfString ~~ "a" ~~ __] &/@ {"apple", "banana", "artichoke"}"#
      )
      .unwrap(),
      "{True, False, True}"
    );
  }

  // StartOfLine anchors to the start of each line in multiline input. The
  // anchor must inspect the original string, not a positional slice — otherwise
  // every position after a line break would spuriously match.
  #[test]
  fn replace_start_of_line_does_not_match_middle() {
    assert_eq!(
      interpret(r#"StringReplace["abab", StartOfLine ~~ "a" -> "X"]"#).unwrap(),
      "Xbab"
    );
  }

  // WordBoundary (\b) matches between a word and non-word character.
  #[test]
  fn replace_with_word_boundary() {
    assert_eq!(
      interpret(
        r#"StringReplace["apple banana orange artichoke", "e" ~~ WordBoundary -> "E"]"#
      )
      .unwrap(),
      "applE banana orangE artichokE"
    );
  }

  // Except[pattern] matches a single non-matching character — used here to
  // strip everything that isn't a letter.
  #[test]
  fn replace_except_letter_character() {
    assert_eq!(
      interpret(
        r#"StringReplace["Hello world!", Except[LetterCharacter] -> ""]"#
      )
      .unwrap(),
      "Helloworld"
    );
  }
}
