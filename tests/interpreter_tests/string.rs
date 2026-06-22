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
  fn string_join_no_args() {
    // StringJoin[] is the empty string (matches wolframscript); this also
    // lets `StringJoin @@ {}` fold to "" instead of erroring.
    assert_eq!(interpret("StringJoin[]").unwrap(), "");
    assert_eq!(interpret("StringJoin @@ {}").unwrap(), "");
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

  // Like the single-delimiter form, a list of delimiters drops the empty
  // pieces at the very start and end while keeping interior empties.
  #[test]
  fn drops_leading_and_trailing_empties() {
    assert_eq!(
      interpret("StringSplit[\"a1b2c3\", {\"1\", \"2\", \"3\"}]").unwrap(),
      "{a, b, c}"
    );
    assert_eq!(
      interpret("StringSplit[\"1a2b3\", {\"1\", \"2\", \"3\"}]").unwrap(),
      "{a, b}"
    );
    assert_eq!(
      interpret("StringSplit[\"xayb\", {\"x\", \"y\"}]").unwrap(),
      "{a, b}"
    );
  }

  #[test]
  fn keeps_interior_empties() {
    assert_eq!(
      interpret("StringSplit[\"a1b22c3\", {\"1\", \"2\", \"3\"}]").unwrap(),
      "{a, b, , c}"
    );
  }
}

// When an explicit maximum number of pieces is given, StringSplit does NOT
// drop empty pieces, and the final piece keeps the original remainder
// verbatim (the un-split tail of the string).
mod string_split_max_parts {
  use super::*;

  #[test]
  fn single_delimiter_keeps_empties() {
    assert_eq!(
      interpret("StringSplit[\",a,b,\", \",\", 5]").unwrap(),
      "{, a, b, }"
    );
    assert_eq!(
      interpret("StringSplit[\",a,b,\", \",\", 2]").unwrap(),
      "{, a,b,}"
    );
  }

  #[test]
  fn single_delimiter_remainder_unsplit() {
    assert_eq!(
      interpret("StringSplit[\"a,b,c,d\", \",\", 2]").unwrap(),
      "{a, b,c,d}"
    );
  }

  #[test]
  fn multiple_delimiters_remainder_keeps_original() {
    // The tail "b2c3" must keep its original delimiters, not be rejoined
    // with the first delimiter of the list.
    assert_eq!(
      interpret("StringSplit[\"a1b2c3\", {\"1\", \"2\", \"3\"}, 2]").unwrap(),
      "{a, b2c3}"
    );
    assert_eq!(
      interpret("StringSplit[\"1a2b3c\", {\"1\", \"2\", \"3\"}, 2]").unwrap(),
      "{, a2b3c}"
    );
  }

  #[test]
  fn multiple_delimiters_exact_count_keeps_trailing_empty() {
    assert_eq!(
      interpret("StringSplit[\"a1b2c3\", {\"1\", \"2\", \"3\"}, 4]").unwrap(),
      "{a, b, c, }"
    );
  }

  // Without a max-parts argument the trailing empty is still dropped.
  #[test]
  fn no_max_parts_still_trims() {
    assert_eq!(
      interpret("StringSplit[\",a,b,\", \",\"]").unwrap(),
      "{a, b}"
    );
  }
}

mod string_split_rule_delimiters {
  use super::*;

  #[test]
  fn replace_fixed_string() {
    assert_eq!(
      interpret("StringSplit[\"a-b-c\", \"-\" -> \"+\"]").unwrap(),
      "{a, +, b, +, c}"
    );
  }

  #[test]
  fn replace_rule_delayed() {
    assert_eq!(
      interpret("StringSplit[\"aXbXc\", \"X\" :> \"Y\"]").unwrap(),
      "{a, Y, b, Y, c}"
    );
  }

  #[test]
  fn leading_and_trailing_delimiters() {
    // Leading/trailing empty text segments are dropped, the replaced
    // delimiters are kept.
    assert_eq!(
      interpret("StringSplit[\"-a-b-\", \"-\" -> \"+\"]").unwrap(),
      "{+, a, +, b, +}"
    );
  }

  #[test]
  fn adjacent_delimiters_keep_inner_empty() {
    assert_eq!(
      interpret("StringSplit[\"a--b\", \"-\" -> \"+\"]").unwrap(),
      "{a, +, , +, b}"
    );
  }

  #[test]
  fn no_match_returns_whole_string() {
    assert_eq!(
      interpret("StringSplit[\"abc\", \"-\" -> \"+\"]").unwrap(),
      "{abc}"
    );
  }

  #[test]
  fn list_of_delimiters() {
    assert_eq!(
      interpret("StringSplit[\"a,b;c\", {\",\", \";\"} -> \"|\"]").unwrap(),
      "{a, |, b, |, c}"
    );
  }

  #[test]
  fn character_class_delimiter() {
    assert_eq!(
      interpret("StringSplit[\"a1b22c\", DigitCharacter -> \"X\"]").unwrap(),
      "{a, X, b, X, , X, c}"
    );
  }

  #[test]
  fn max_parts_limits_splits() {
    assert_eq!(
      interpret("StringSplit[\"a-b-c\", \"-\" -> \"+\", 2]").unwrap(),
      "{a, +, b-c}"
    );
  }

  #[test]
  fn captured_delimiter_identity() {
    assert_eq!(
      interpret("StringSplit[\"aXbYc\", x : {\"X\", \"Y\"} :> x]").unwrap(),
      "{a, X, b, Y, c}"
    );
  }

  #[test]
  fn captured_delimiter_transformed() {
    assert_eq!(
      interpret(
        "StringSplit[\"a1b2c\", d : DigitCharacter :> \"<\" <> d <> \">\"]"
      )
      .unwrap(),
      "{a, <1>, b, <2>, c}"
    );
  }

  #[test]
  fn threads_over_list_of_strings() {
    assert_eq!(
      interpret("StringSplit[{\"a-b\", \"c-d\"}, \"-\" -> \"/\"]").unwrap(),
      "{{a, /, b}, {c, /, d}}"
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

  // Only a string (or list of strings) yields characters; any other
  // expression stays unevaluated, matching wolframscript.
  #[test]
  fn characters_nonstring_stays_unevaluated() {
    assert_eq!(interpret("Characters[123]").unwrap(), "Characters[123]");
    assert_eq!(interpret("Characters[1.5]").unwrap(), "Characters[1.5]");
    assert_eq!(interpret("Characters[a]").unwrap(), "Characters[a]");
    assert_eq!(interpret("Characters[x + y]").unwrap(), "Characters[x + y]");
  }

  // List threading leaves non-string elements as unevaluated Characters[...].
  #[test]
  fn characters_threads_with_nonstring_element() {
    assert_eq!(
      interpret(r#"Characters[{"ab", 5}]"#).unwrap(),
      "{{a, b}, Characters[5]}"
    );
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

  // A delayed rule (:>) must evaluate its RHS for each match, even when the
  // RHS is a constant expression that does not reference the matched text.
  #[test]
  fn delayed_constant_rhs_is_evaluated() {
    assert_eq!(
      interpret(
        r#"StringReplace["aAbB", LetterCharacter :> ToUpperCase["x"]]"#
      )
      .unwrap(),
      "XXXX"
    );
    assert_eq!(
      interpret(r#"StringReplace["a1b2", DigitCharacter :> ToString[1 + 1]]"#)
        .unwrap(),
      "a2b2"
    );
    // Same for a literal-string pattern with a delayed RHS.
    assert_eq!(
      interpret(r#"StringReplace["aa", "a" :> ToString[1 + 1]]"#).unwrap(),
      "22"
    );
  }

  // A delayed rule whose RHS references the matched pattern variable.
  #[test]
  fn delayed_rhs_uses_match() {
    assert_eq!(
      interpret(r#"StringReplace["abc", c_ :> ToUpperCase[c]]"#).unwrap(),
      "ABC"
    );
    assert_eq!(
      interpret(r#"StringReplace["abc", x_ :> x <> x]"#).unwrap(),
      "aabbcc"
    );
  }

  // A delayed rule whose RHS is already a string stays literal.
  #[test]
  fn delayed_literal_string_rhs() {
    assert_eq!(
      interpret(r#"StringReplace["aaa", "a" :> "b"]"#).unwrap(),
      "bbb"
    );
  }

  // RegularExpression replacements expand $0/$1/… to capture groups.
  #[test]
  fn regex_capture_group_backreferences() {
    assert_eq!(
      interpret(
        r#"StringReplace["2024-01-15", RegularExpression["(\\d+)-(\\d+)-(\\d+)"] -> "$3/$2/$1"]"#
      )
      .unwrap(),
      "15/01/2024"
    );
    assert_eq!(
      interpret(
        r#"StringReplace["John Smith", RegularExpression["(\\w+) (\\w+)"] -> "$2, $1"]"#
      )
      .unwrap(),
      "Smith, John"
    );
    assert_eq!(
      interpret(
        r#"StringReplace["hello", RegularExpression["(l+)"] -> "[$1]"]"#
      )
      .unwrap(),
      "he[ll]o"
    );
  }

  // A delayed rule (:>) with a plain-string replacement expands $n
  // backreferences just like an immediate rule (->) — the constant RHS
  // makes the delayed/immediate distinction irrelevant.
  #[test]
  fn regex_backreferences_with_delayed_rule() {
    assert_eq!(
      interpret(
        r#"StringReplace["abc", RegularExpression["(a)(b)"] :> "$2$1"]"#
      )
      .unwrap(),
      "bac"
    );
    assert_eq!(
      interpret(
        r#"StringReplace["2024-01-15", RegularExpression["(\\d+)-(\\d+)-(\\d+)"] :> "$3/$2/$1"]"#
      )
      .unwrap(),
      "15/01/2024"
    );
    // A delayed rule whose RHS must be evaluated per match still works.
    assert_eq!(
      interpret(
        r#"StringReplace["hello", RegularExpression["l"] :> ToUpperCase["l"]]"#
      )
      .unwrap(),
      "heLLo"
    );
  }

  #[test]
  fn regex_dollar_zero_is_whole_match() {
    assert_eq!(
      interpret(
        r#"StringReplace["abc", RegularExpression["(a)(b)(c)"] -> "$0"]"#
      )
      .unwrap(),
      "abc"
    );
  }

  // A lone `$` (and `$$`) is literal; a missing group expands to nothing.
  #[test]
  fn regex_dollar_edge_cases() {
    assert_eq!(
      interpret(
        r#"StringReplace["price: 100", RegularExpression["(\\d+)"] -> "$$$1"]"#
      )
      .unwrap(),
      "price: $$100"
    );
    assert_eq!(
      interpret(r#"StringReplace["abc", RegularExpression["b"] -> "X$1Y"]"#)
        .unwrap(),
      "aXYc"
    );
  }

  // A literal (non-RegularExpression) pattern keeps `$n` verbatim.
  #[test]
  fn literal_pattern_keeps_dollar_verbatim() {
    assert_eq!(
      interpret(r#"StringReplace["abc", "b" -> "$1"]"#).unwrap(),
      "a$1c"
    );
    assert_eq!(
      interpret(r#"StringReplace["aaa", "a" -> "$0"]"#).unwrap(),
      "$0$0$0"
    );
  }

  #[test]
  fn shortest_in_string_expression_is_lazy() {
    // Shortest[___] in the middle of a StringExpression must match lazily, so
    // each /*…*/ is stripped separately (not first /* to last */).
    assert_eq!(
      interpret(
        "StringReplace[\"x/*a*/y/*b*/z\", \"/*\" ~~ Shortest[___] ~~ \"*/\" -> \"\"]"
      )
      .unwrap(),
      "xyz"
    );
  }

  #[test]
  fn literal_star_inside_string_expression() {
    // `*` is a wildcard only in a bare string; inside `~~` it is literal.
    // "xAAy" has no literal `*`, so nothing is replaced.
    assert_eq!(
      interpret("StringReplace[\"xAAy\", \"x\" ~~ \"*\" ~~ \"y\" -> \"Z\"]")
        .unwrap(),
      "xAAy"
    );
    // A bare string keeps the `*` wildcard shorthand.
    assert_eq!(
      interpret("StringMatchQ[\"aXXb\", \"a*b\"]").unwrap(),
      "True"
    );
  }

  #[test]
  fn blank_pattern_matches_across_newlines() {
    // Blanks (`___`/`__`/`_`) in string patterns match newlines too (dotall),
    // matching Wolfram — e.g. a block comment spanning lines.
    assert_eq!(
      interpret(
        "StringReplace[\"a/*x\\ny*/b\", \"/*\" ~~ ___ ~~ \"*/\" -> \"\"]"
      )
      .unwrap(),
      "ab"
    );
  }

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

  #[test]
  fn named_pattern_rule_valued_rhs() {
    // The RHS of a `:>` rule may itself be a rule (w -> d); the named
    // pattern variables must be substituted on both sides of it.
    assert_eq!(
      interpret(
        r#"StringCases["x=1, y=2", w:WordCharacter ~~ "=" ~~ d:DigitCharacter :> w -> d]"#
      )
      .unwrap(),
      "{x -> 1, y -> 2}"
    );
  }

  #[test]
  fn named_pattern_comparison_rhs() {
    // A comparison RHS likewise has its pattern variables substituted; here
    // "x" == "1" evaluates to False (unsubstituted it would stay `w == d`).
    assert_eq!(
      interpret(
        r#"StringCases["x=1", w:WordCharacter ~~ "=" ~~ d:DigitCharacter :> w == d]"#
      )
      .unwrap(),
      "{False}"
    );
  }

  #[test]
  fn named_pattern_delayed_rule_valued_rhs() {
    // A delayed-rule (:>) RHS nested inside the outer rule also substitutes.
    assert_eq!(
      interpret(
        r#"StringCases["x=1, y=2", w:WordCharacter ~~ "=" ~~ d:DigitCharacter :> (w :> d)]"#
      )
      .unwrap(),
      "{x :> 1, y :> 2}"
    );
  }

  // IgnoreCase -> True makes a literal pattern match regardless of case.
  #[test]
  fn ignore_case_single_rule() {
    assert_eq!(
      interpret(r#"StringReplace["aAbB", "a" -> "1", IgnoreCase -> True]"#)
        .unwrap(),
      "11bB"
    );
  }

  #[test]
  fn ignore_case_multichar_literal() {
    assert_eq!(
      interpret(
        r#"StringReplace["ABC abc", "abc" -> "X", IgnoreCase -> True]"#
      )
      .unwrap(),
      "X X"
    );
  }

  #[test]
  fn ignore_case_multiple_rules() {
    assert_eq!(
      interpret(
        r#"StringReplace["aAbB", {"a" -> "1", "b" -> "2"}, IgnoreCase -> True]"#
      )
      .unwrap(),
      "1122"
    );
  }

  // IgnoreCase also applies to alternatives and other compound patterns.
  #[test]
  fn ignore_case_alternatives() {
    assert_eq!(
      interpret(
        r#"StringReplace["abcABC", "b" | "c" -> "_", IgnoreCase -> True]"#
      )
      .unwrap(),
      "a__A__"
    );
  }

  // The replacement limit and the IgnoreCase option can be combined.
  #[test]
  fn ignore_case_with_limit() {
    assert_eq!(
      interpret(r#"StringReplace["aAaA", "a" -> "x", 2, IgnoreCase -> True]"#)
        .unwrap(),
      "xxaA"
    );
  }

  // IgnoreCase -> False keeps the default case-sensitive behaviour.
  #[test]
  fn ignore_case_false_is_case_sensitive() {
    assert_eq!(
      interpret(r#"StringReplace["aAbB", "a" -> "1", IgnoreCase -> False]"#)
        .unwrap(),
      "1AbB"
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

  #[test]
  fn utf8_decodes_byte_sequences() {
    // With a UTF-8 encoding the integers are bytes: a multi-byte sequence
    // decodes to a single character (not one code point per byte).
    assert_eq!(
      interpret(r#"FromCharacterCode[{195, 169}, "UTF8"]"#).unwrap(),
      "é"
    );
    assert_eq!(
      interpret(r#"FromCharacterCode[{226, 130, 172}, "UTF8"]"#).unwrap(),
      "€"
    );
    // ASCII bytes are unchanged.
    assert_eq!(
      interpret(r#"FromCharacterCode[{72, 105}, "UTF8"]"#).unwrap(),
      "Hi"
    );
    // Without an encoding the integers are code points (no decoding).
    assert_eq!(interpret("FromCharacterCode[{195, 169}]").unwrap(), "Ã©");
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

mod printable_ascii_q {
  use super::*;

  #[test]
  fn printable_strings() {
    assert_eq!(
      interpret("PrintableASCIIQ[\"Hello World 123!\"]").unwrap(),
      "True"
    );
    // The empty string and a single space are printable.
    assert_eq!(interpret("PrintableASCIIQ[\"\"]").unwrap(), "True");
    assert_eq!(interpret("PrintableASCIIQ[\" \"]").unwrap(), "True");
    // The full printable range, codes 32..126.
    assert_eq!(
      interpret("PrintableASCIIQ[FromCharacterCode[Range[32, 126]]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn non_printable_strings() {
    // Non-ASCII letters.
    assert_eq!(interpret("PrintableASCIIQ[\"héllo\"]").unwrap(), "False");
    // Control characters (tab, newline, DEL, and code 31) are not printable.
    assert_eq!(interpret("PrintableASCIIQ[\"a\tb\"]").unwrap(), "False");
    assert_eq!(interpret("PrintableASCIIQ[\"a\nb\"]").unwrap(), "False");
    assert_eq!(
      interpret("PrintableASCIIQ[FromCharacterCode[127]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PrintableASCIIQ[FromCharacterCode[31]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn threads_over_list_of_strings() {
    assert_eq!(
      interpret("PrintableASCIIQ[{\"abc\", \"déf\"}]").unwrap(),
      "{True, False}"
    );
    assert_eq!(interpret("PrintableASCIIQ[{}]").unwrap(), "{}");
  }

  #[test]
  fn non_string_stays_unevaluated() {
    // Non-strings, or lists that are not entirely strings, stay unevaluated.
    assert_eq!(
      interpret("PrintableASCIIQ[123]").unwrap(),
      "PrintableASCIIQ[123]"
    );
    assert_eq!(
      interpret("PrintableASCIIQ[x]").unwrap(),
      "PrintableASCIIQ[x]"
    );
    assert_eq!(
      interpret("PrintableASCIIQ[{\"abc\", 5}]").unwrap(),
      "PrintableASCIIQ[{abc, 5}]"
    );
    assert_eq!(
      interpret("PrintableASCIIQ[{{\"ab\"}, \"cd\"}]").unwrap(),
      "PrintableASCIIQ[{{ab}, cd}]"
    );
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
  fn swedish_appends_aaring_aumlaut_oumlaut() {
    // Swedish adds å, ä, ö after z. Regression for new locale support.
    assert_eq!(
      interpret("Alphabet[\"Swedish\"]").unwrap(),
      "{a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, \
       x, y, z, å, ä, ö}"
    );
  }

  #[test]
  fn finnish_matches_swedish() {
    assert_eq!(
      interpret("Alphabet[\"Finnish\"]").unwrap(),
      interpret("Alphabet[\"Swedish\"]").unwrap()
    );
  }

  #[test]
  fn norwegian_appends_aelig_oslash_aaring() {
    assert_eq!(
      interpret("Alphabet[\"Norwegian\"]").unwrap(),
      "{a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, \
       x, y, z, æ, ø, å}"
    );
  }

  #[test]
  fn danish_matches_norwegian() {
    assert_eq!(
      interpret("Alphabet[\"Danish\"]").unwrap(),
      interpret("Alphabet[\"Norwegian\"]").unwrap()
    );
  }

  #[test]
  fn polish_has_diacritic_letters_interleaved() {
    assert_eq!(
      interpret("Alphabet[\"Polish\"]").unwrap(),
      "{a, ą, b, c, ć, d, e, ę, f, g, h, i, j, k, l, ł, m, n, ń, o, ó, p, r, \
       s, ś, t, u, w, y, z, ź, ż}"
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
    assert_eq!(interpret("Length[Alphabet[\"Cyrillic\"]]").unwrap(), "49");
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

  // Two-argument form: index into a named alphabet.
  #[test]
  fn greek_alphabet() {
    assert_eq!(interpret(r#"FromLetterNumber[3, "Greek"]"#).unwrap(), "γ");
    assert_eq!(interpret(r#"FromLetterNumber[-1, "Greek"]"#).unwrap(), "ω");
  }

  #[test]
  fn greek_list() {
    assert_eq!(
      interpret(r#"FromLetterNumber[{1, 2, 3}, "Greek"]"#).unwrap(),
      "{α, β, γ}"
    );
  }

  #[test]
  fn russian_and_spanish() {
    assert_eq!(interpret(r#"FromLetterNumber[1, "Russian"]"#).unwrap(), "а");
    assert_eq!(interpret(r#"FromLetterNumber[3, "Spanish"]"#).unwrap(), "c");
  }

  #[test]
  fn out_of_range_named() {
    assert_eq!(interpret(r#"FromLetterNumber[100, "Greek"]"#).unwrap(), " ");
    assert_eq!(interpret(r#"FromLetterNumber[0, "Greek"]"#).unwrap(), " ");
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
  fn named_minus_union_intersection_render_to_wolfram_codepoints() {
    // \[Minus] is the minus sign (U+2212); \[Union]/\[Intersection] are the
    // n-ary forms ⋃/⋂ (U+22C3/U+22C2), not the binary ∪/∩.
    assert_eq!(interpret("\"\\[Minus]\"").unwrap(), "\u{2212}");
    assert_eq!(interpret("\"\\[Union]\"").unwrap(), "\u{22C3}");
    assert_eq!(interpret("\"\\[Intersection]\"").unwrap(), "\u{22C2}");
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
  fn string_match_q_curried() {
    assert_eq!(
      interpret("StringMatchQ[\"*G*\"][\"CTG1\"]").unwrap(),
      "True"
    );
  }

  // The bare operator form (not yet applied) stays as an operator, exactly
  // like Wolfram, instead of erroring on the missing second argument.
  #[test]
  fn string_contains_q_operator_unevaluated() {
    assert_eq!(
      interpret("StringContainsQ[\"G\"]").unwrap(),
      "StringContainsQ[G]"
    );
  }

  #[test]
  fn string_starts_q_operator_in_select() {
    assert_eq!(
      interpret(
        "Select[{\"CAC1\", \"CTG1\", \"ACT1\", \"CGA1\", \"CTC1\"}, StringStartsQ[\"C\"]]"
      )
      .unwrap(),
      "{CAC1, CTG1, CGA1, CTC1}"
    );
  }

  #[test]
  fn string_ends_q_operator_in_select() {
    assert_eq!(
      interpret(
        "Select[{\"CAC1\", \"CTG1\", \"ACT1\", \"CGA1\", \"CTC1\"}, StringEndsQ[\"C1\"]]"
      )
      .unwrap(),
      "{CAC1, CTC1}"
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

  // StringCases collects every case-insensitive match, returning the actual
  // matched substrings (which may differ in case from the pattern).
  #[test]
  fn string_cases_ignore_case_literal() {
    assert_eq!(
      interpret(r#"StringCases["aAbA", "a", IgnoreCase -> True]"#).unwrap(),
      "{a, A, A}"
    );
    assert_eq!(
      interpret(r#"StringCases["ABCabc", "abc", IgnoreCase -> True]"#).unwrap(),
      "{ABC, abc}"
    );
  }

  #[test]
  fn string_cases_ignore_case_alternatives() {
    assert_eq!(
      interpret(r#"StringCases["aAbB", "a" | "b", IgnoreCase -> True]"#)
        .unwrap(),
      "{a, A, b, B}"
    );
  }

  // The rule (pattern -> replacement) form also honors IgnoreCase.
  #[test]
  fn string_cases_ignore_case_rule() {
    assert_eq!(
      interpret(r#"StringCases["Hello", "l" -> "L", IgnoreCase -> True]"#)
        .unwrap(),
      "{L, L}"
    );
  }

  // The match limit and IgnoreCase combine.
  #[test]
  fn string_cases_ignore_case_with_limit() {
    assert_eq!(
      interpret(r#"StringCases["aAaA", "a", 2, IgnoreCase -> True]"#).unwrap(),
      "{a, A}"
    );
  }

  // IgnoreCase -> False keeps the default case-sensitive behaviour.
  #[test]
  fn string_cases_ignore_case_false() {
    assert_eq!(
      interpret(r#"StringCases["aAbA", "a", IgnoreCase -> False]"#).unwrap(),
      "{a}"
    );
  }

  // StringPosition reports the span of every case-insensitive match.
  #[test]
  fn string_position_ignore_case_literal() {
    assert_eq!(
      interpret(r#"StringPosition["aAa", "a", IgnoreCase -> True]"#).unwrap(),
      "{{1, 1}, {2, 2}, {3, 3}}"
    );
    assert_eq!(
      interpret(r#"StringPosition["ABCabc", "abc", IgnoreCase -> True]"#)
        .unwrap(),
      "{{1, 3}, {4, 6}}"
    );
  }

  #[test]
  fn string_position_ignore_case_alternatives() {
    assert_eq!(
      interpret(r#"StringPosition["aAbB", "a" | "b", IgnoreCase -> True]"#)
        .unwrap(),
      "{{1, 1}, {2, 2}, {3, 3}, {4, 4}}"
    );
  }

  // Overlapping case-insensitive multi-character matches are all reported.
  #[test]
  fn string_position_ignore_case_overlapping() {
    assert_eq!(
      interpret(r#"StringPosition["AbAbAb", "ab", IgnoreCase -> True]"#)
        .unwrap(),
      "{{1, 2}, {3, 4}, {5, 6}}"
    );
  }

  // The count limit and IgnoreCase combine.
  #[test]
  fn string_position_ignore_case_with_limit() {
    assert_eq!(
      interpret(r#"StringPosition["aAa", "a", 2, IgnoreCase -> True]"#)
        .unwrap(),
      "{{1, 1}, {2, 2}}"
    );
  }

  // IgnoreCase -> False keeps the default case-sensitive behaviour.
  #[test]
  fn string_position_ignore_case_false() {
    assert_eq!(
      interpret(r#"StringPosition["aAa", "a", IgnoreCase -> False]"#).unwrap(),
      "{{1, 1}, {3, 3}}"
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
  fn string_pattern_predicate_tests() {
    // `_?pred` character-predicate patterns map to regex character classes.
    assert_eq!(
      interpret("StringSplit[\"a1b2c3\", x_?LetterQ]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("StringSplit[\"a1b2c3\", x__?DigitQ]").unwrap(),
      "{a, b, c}"
    );
    assert_eq!(
      interpret("StringCases[\"a1b2c3\", _?DigitQ]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("StringCases[\"aXbYcZ\", _?UpperCaseQ]").unwrap(),
      "{X, Y, Z}"
    );
    assert_eq!(
      interpret("StringCases[\"aXbY\", x_?LowerCaseQ -> x]").unwrap(),
      "{a, b}"
    );
    assert_eq!(
      interpret("StringReplace[\"abcABC\", a_?LowerCaseQ :> ToUpperCase[a]]")
        .unwrap(),
      "ABCABC"
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

  // RegularExpression transforms expand $0/$1/... to capture groups.
  #[test]
  fn string_cases_regex_capture_groups() {
    assert_eq!(
      interpret(
        r#"StringCases["a1b2", RegularExpression["[a-z](\\d)"] -> "$1"]"#
      )
      .unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret(
        r#"StringCases["a1b2", RegularExpression["([a-z])(\\d)"] :> "$2$1"]"#
      )
      .unwrap(),
      "{1a, 2b}"
    );
    // The transform can be a list of strings.
    assert_eq!(
      interpret(
        r#"StringCases["2024-01", RegularExpression["(\\d+)-(\\d+)"] :> {"$1", "$2"}]"#
      )
      .unwrap(),
      "{{2024, 01}}"
    );
    // $0 is the whole match.
    assert_eq!(
      interpret(
        r#"StringCases["a1b2", RegularExpression["([a-z])(\\d)"] -> "$0"]"#
      )
      .unwrap(),
      "{a1, b2}"
    );
  }

  #[test]
  fn string_cases_overlaps() {
    // Default is non-overlapping.
    assert_eq!(interpret("StringCases[\"aaa\", \"aa\"]").unwrap(), "{aa}");
    assert_eq!(
      interpret("StringCases[\"aaa\", \"aa\", Overlaps -> False]").unwrap(),
      "{aa}"
    );
    // Overlaps -> True emits a match at every start position.
    assert_eq!(
      interpret("StringCases[\"aaa\", \"aa\", Overlaps -> True]").unwrap(),
      "{aa, aa}"
    );
    assert_eq!(
      interpret("StringCases[\"abababab\", \"aba\", Overlaps -> True]")
        .unwrap(),
      "{aba, aba, aba}"
    );
    // Works with character-class patterns too.
    assert_eq!(
      interpret(
        "StringCases[\"a1b2\", LetterCharacter ~~ DigitCharacter, \
         Overlaps -> True]"
      )
      .unwrap(),
      "{a1, b2}"
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

  // Two-argument form: counts of length-n character n-grams, in
  // first-occurrence order (not sorted by count, unlike the 1-arg form).
  #[test]
  fn ngram_bigrams() {
    assert_eq!(
      interpret(r#"CharacterCounts["ababcab", 2]"#).unwrap(),
      "<|ab -> 3, ba -> 1, bc -> 1, ca -> 1|>"
    );
  }
  #[test]
  fn ngram_first_occurrence_order() {
    assert_eq!(
      interpret(r#"CharacterCounts["banana", 2]"#).unwrap(),
      "<|ba -> 1, an -> 2, na -> 2|>"
    );
  }
  #[test]
  fn ngram_trigrams() {
    assert_eq!(
      interpret(r#"CharacterCounts["banana", 3]"#).unwrap(),
      "<|ban -> 1, ana -> 2, nan -> 1|>"
    );
  }
  #[test]
  fn ngram_too_long_is_empty() {
    assert_eq!(interpret(r#"CharacterCounts["ab", 3]"#).unwrap(), "<||>");
  }
  // A list of strings threads: one association per string (not the list's
  // own punctuation characters).
  #[test]
  fn list_of_strings_threads() {
    assert_eq!(
      interpret(r#"CharacterCounts[{"hello", "world"}]"#).unwrap(),
      "{<|l -> 2, o -> 1, e -> 1, h -> 1|>, \
       <|d -> 1, l -> 1, r -> 1, o -> 1, w -> 1|>}"
    );
  }
  #[test]
  fn list_of_strings_ngram_threads() {
    assert_eq!(
      interpret(r#"CharacterCounts[{"ab", "cd"}, 2]"#).unwrap(),
      "{<|ab -> 1|>, <|cd -> 1|>}"
    );
  }
}

mod letter_counts_ngram {
  use super::*;

  // LetterCounts[s, n]: n-grams within maximal runs of letters; non-letter
  // characters break the window. First-occurrence order.
  #[test]
  fn ngram_breaks_on_non_letters() {
    assert_eq!(
      interpret(r#"LetterCounts["ab12cd34ab12", 2]"#).unwrap(),
      "<|ab -> 2, cd -> 1|>"
    );
  }
  #[test]
  fn ngram_within_words() {
    assert_eq!(
      interpret(r#"LetterCounts["abc def", 2]"#).unwrap(),
      "<|ab -> 1, bc -> 1, de -> 1, ef -> 1|>"
    );
  }
  #[test]
  fn ngram_trigrams() {
    assert_eq!(
      interpret(r#"LetterCounts["banana", 3]"#).unwrap(),
      "<|ban -> 1, ana -> 2, nan -> 1|>"
    );
  }
  // A list of strings threads: each string gets its own association.
  #[test]
  fn list_of_strings_threads() {
    assert_eq!(
      interpret(r#"LetterCounts[{"hello", "world", "!"}]"#).unwrap(),
      "{<|l -> 2, o -> 1, e -> 1, h -> 1|>, \
       <|d -> 1, l -> 1, r -> 1, o -> 1, w -> 1|>, <||>}"
    );
  }
  #[test]
  fn list_of_strings_ngram_threads() {
    assert_eq!(
      interpret(r#"LetterCounts[{"ab12", "cdcd"}, 2]"#).unwrap(),
      "{<|ab -> 1|>, <|cd -> 2, dc -> 1|>}"
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
  fn md2_hex_string() {
    // RFC 1319 test vector for "abc".
    assert_eq!(
      interpret(r#"Hash["abc", "MD2", "HexString"]"#).unwrap(),
      "da853b0d3f88d99b30283a69e6ded6bb"
    );
  }

  #[test]
  fn md2_empty_hex_string() {
    // RFC 1319 test vector for the empty string.
    assert_eq!(
      interpret(r#"Hash["", "MD2", "HexString"]"#).unwrap(),
      "8350e5a3e24c153df2275c9f80692773"
    );
  }

  #[test]
  fn md2_integer() {
    assert_eq!(
      interpret(r#"Hash["abc", "MD2"]"#).unwrap(),
      "290463476275092517648070427531620046523"
    );
  }

  #[test]
  fn md2_quick_brown_fox() {
    assert_eq!(
      interpret(
        r#"Hash["The quick brown fox jumps over the lazy dog", "MD2", "HexString"]"#
      )
      .unwrap(),
      "03d85a0d629d2c442e987525319fc471"
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

  // Standard IEEE CRC-32 of the string's bytes.
  #[test]
  fn crc32_integer() {
    assert_eq!(interpret(r#"Hash["abc", "CRC32"]"#).unwrap(), "891568578");
    assert_eq!(interpret(r#"Hash["", "CRC32"]"#).unwrap(), "0");
    assert_eq!(
      interpret(r#"Hash["Hello, World!", "CRC32"]"#).unwrap(),
      "3964322768"
    );
  }

  #[test]
  fn crc32_hex_string() {
    assert_eq!(
      interpret(r#"Hash["abc", "CRC32", "HexString"]"#).unwrap(),
      "352441c2"
    );
  }

  // Adler-32 checksum of the string's bytes.
  #[test]
  fn adler32_integer() {
    assert_eq!(interpret(r#"Hash["abc", "Adler32"]"#).unwrap(), "38600999");
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

// StringTake / StringDrop with a Span (i;;j;;k) spec, equivalent to the
// {i, j, k} list form.
mod string_take_drop_span {
  use super::*;

  #[test]
  fn take_span() {
    assert_eq!(interpret(r#"StringTake["hello", ;;3]"#).unwrap(), "hel");
    assert_eq!(interpret(r#"StringTake["hello", 2;;4]"#).unwrap(), "ell");
    assert_eq!(interpret(r#"StringTake["hello", 2;;]"#).unwrap(), "ello");
    assert_eq!(interpret(r#"StringTake["hello", ;;-2]"#).unwrap(), "hell");
    assert_eq!(interpret(r#"StringTake["hello", ;;]"#).unwrap(), "hello");
  }

  #[test]
  fn take_span_with_step() {
    assert_eq!(
      interpret(r#"StringTake["hello", 1;;-1;;2]"#).unwrap(),
      "hlo"
    );
  }

  #[test]
  fn drop_span() {
    assert_eq!(interpret(r#"StringDrop["hello", ;;2]"#).unwrap(), "llo");
    assert_eq!(interpret(r#"StringDrop["hello", 2;;3]"#).unwrap(), "hlo");
    assert_eq!(interpret(r#"StringDrop["hello", ;;-2]"#).unwrap(), "o");
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
  fn number_string_signed() {
    // NumberString matches an optional leading sign as part of the number.
    assert_eq!(
      interpret(r#"StringCases["2024-03-15", NumberString]"#).unwrap(),
      "{2024, -03, -15}"
    );
    assert_eq!(
      interpret(r#"StringCases["a-5b+3c", NumberString]"#).unwrap(),
      "{-5, +3}"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["-5", NumberString]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"StringMatchQ["+5", NumberString]"#).unwrap(),
      "True"
    );
    // Only a single sign is allowed.
    assert_eq!(
      interpret(r#"StringMatchQ["--5", NumberString]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn number_string_leading_and_trailing_decimal() {
    // A leading-decimal form (.5) and a trailing-decimal form (1.) both match.
    assert_eq!(
      interpret(r#"StringCases[".5", NumberString]"#).unwrap(),
      "{.5}"
    );
    assert_eq!(
      interpret(r#"StringCases["1.", NumberString]"#).unwrap(),
      "{1.}"
    );
    assert_eq!(
      interpret(r#"StringCases["-.5", NumberString]"#).unwrap(),
      "{-.5}"
    );
    // Greedy decimal stops at a second dot.
    assert_eq!(
      interpret(r#"StringCases["3.14.15", NumberString]"#).unwrap(),
      "{3.14, .15}"
    );
    // No exponent: `1e5` is two number strings.
    assert_eq!(
      interpret(r#"StringCases["1e5", NumberString]"#).unwrap(),
      "{1, 5}"
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

  #[test]
  fn bigint_base2() {
    // Beyond i128 — verifies BigInteger path. Reference value taken from
    // wolframscript: IntegerString[143207491493571284560146904872817600361573129, 2].
    assert_eq!(
      interpret(
        "IntegerString[143207491493571284560146904872817600361573129, 2]"
      )
      .unwrap(),
      "110011010111111000011111110001111001100111000001111110111010000100011110001111110101101100100110010000011001000110000001111101000101010101100001001"
    );
  }

  #[test]
  fn bigint_negative_drops_sign() {
    assert_eq!(
      interpret(
        "IntegerString[-143207491493571284560146904872817600361573129, 2]"
      )
      .unwrap(),
      "110011010111111000011111110001111001100111000001111110111010000100011110001111110101101100100110010000011001000110000001111101000101010101100001001"
    );
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
    // StringForm at top level renders as the literal wrapper, matching
    // wolframscript. Substitution only happens via explicit ToString.
    assert_eq!(
      interpret("StringForm[\"The value is ``.\", 5]").unwrap(),
      "StringForm[The value is ``., 5]"
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
  fn string_template_fills_slots() {
    // Named slots from an association.
    assert_eq!(
      interpret(r#"StringTemplate["Hi `name`"][<|"name" -> "Bob"|>]"#).unwrap(),
      "Hi Bob"
    );
    // Positional slots from arguments.
    assert_eq!(
      interpret(r#"StringTemplate["`1` + `2`"][3, 4]"#).unwrap(),
      "3 + 4"
    );
    // Sequential `` slots.
    assert_eq!(
      interpret(r#"StringTemplate["`` and ``"][7, 8]"#).unwrap(),
      "7 and 8"
    );
    // Repeated positional reference.
    assert_eq!(
      interpret(r#"StringTemplate["`1`-`1`-`2`"][3, 4]"#).unwrap(),
      "3-3-4"
    );
    // An unfilled slot renders as the empty string.
    assert_eq!(
      interpret(r#"StringTemplate["a `x` b `y`"][<|"x" -> 1|>]"#).unwrap(),
      "a 1 b "
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

  // The second argument of ToString must be a format symbol, never a number:
  // wolframscript rejects a numeric format with ToString::fmtval and returns
  // the call unevaluated instead of silently stringifying the value.
  #[test]
  fn to_string_numeric_format_rejected() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout("ToString[255, 2]").unwrap();
    assert_eq!(r.result, "ToString[255, 2]");
    assert!(
      r.warnings[0].contains("ToString::fmtval: 2 is not a valid format type."),
      "got: {:?}",
      r.warnings
    );
    // A real-valued format is likewise invalid.
    assert_eq!(
      interpret_with_stdout("ToString[x + y, 1.5]")
        .unwrap()
        .result,
      "ToString[x + y, 1.5]"
    );
    // A genuine format symbol still works.
    assert_eq!(interpret("ToString[255, InputForm]").unwrap(), "255");
  }

  #[test]
  fn to_string_table_form_matrix() {
    assert_eq!(
      interpret("ToString[TableForm[{{1, 2}, {3, 4}}]]").unwrap(),
      "1   2\n\n3   4"
    );
  }

  #[test]
  fn to_string_table_form_column_widths() {
    // Each column is padded to its widest cell; trailing space is trimmed.
    assert_eq!(
      interpret("ToString[TableForm[{{a, bb}, {ccc, d}}]]").unwrap(),
      "a     bb\n\nccc   d"
    );
  }

  #[test]
  fn to_string_table_form_unequal_cell_widths() {
    assert_eq!(
      interpret("ToString[TableForm[{{1, 22}, {3, 4}}]]").unwrap(),
      "1   22\n\n3   4"
    );
  }

  #[test]
  fn to_string_table_form_vector() {
    // A flat vector renders one element per row.
    assert_eq!(
      interpret("ToString[TableForm[{1, 2, 3}]]").unwrap(),
      "1\n\n2\n\n3"
    );
  }

  #[test]
  fn to_string_table_form_single_row() {
    assert_eq!(
      interpret("ToString[TableForm[{{1, 2, 3}}]]").unwrap(),
      "1   2   3"
    );
  }

  #[test]
  fn display_indexed_placeholders() {
    // Top-level StringForm renders as the literal wrapper.
    assert_eq!(
      interpret("StringForm[\"`1` plus `2` is `3`\", 1, 2, 3]").unwrap(),
      "StringForm[`1` plus `2` is `3`, 1, 2, 3]"
    );
  }

  // Out-of-range indexed placeholders are kept literal even after ToString
  // forces substitution (instead of silently blanking them). This is the
  // wolframscript/mathics behaviour.
  #[test]
  fn out_of_range_positive_index_kept_literal() {
    assert_eq!(
      interpret("ToString[StringForm[\"`2` bla\", a]]").unwrap(),
      "`2` bla"
    );
  }

  #[test]
  fn out_of_range_negative_index_kept_literal() {
    assert_eq!(
      interpret("ToString[StringForm[\"`-1` bla\", a]]").unwrap(),
      "`-1` bla"
    );
  }

  #[test]
  fn out_of_range_sequential_placeholder_kept_literal() {
    // `` with no argument to pull from: keep the two backticks literal.
    assert_eq!(interpret("ToString[StringForm[\"x=``\"]]").unwrap(), "x=``");
  }

  #[test]
  fn sequential_placeholder_resumes_from_last_numbered() {
    // `` picks up from the most recently used numbered slot + 1, not from
    // its own independent counter. `1` was the most recent; so `` -> arg 2.
    assert_eq!(
      interpret(
        "ToString[StringForm[\"`2` bla `1` blub `` bla `3`\", a, b, c]]"
      )
      .unwrap(),
      "b bla a blub b bla c"
    );
  }

  #[test]
  fn escaped_backquote_kept_literal() {
    // \` inside the template is a literal backslash + backtick sequence;
    // Woxi/wolframscript keep both bytes verbatim in the output.
    assert_eq!(
      interpret(r#"ToString[StringForm["`` is Global\`a", a]]"#).unwrap(),
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
  fn empty_string_is_null() {
    // "\0" is the interpreter-level Null sentinel (the CLI renders it "Null").
    assert_eq!(interpret("ToExpression[\"\"]").unwrap(), "\0");
  }

  #[test]
  fn incomplete_input_yields_failed_with_sntxi() {
    use woxi::interpret_with_stdout;
    // An incomplete expression must yield $Failed with ToExpression::sntxi,
    // not leak the internal parser error.
    let r = interpret_with_stdout("ToExpression[\"2+\"]").unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(r.warnings[0].contains(
      "ToExpression::sntxi: Incomplete expression; more input is needed."
    ));
  }

  #[test]
  fn invalid_syntax_yields_failed_with_sntx() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout("ToExpression[\"][\"]").unwrap();
    assert_eq!(r.result, "$Failed");
    assert!(
      r.warnings[0]
        .contains("ToExpression::sntx: Invalid syntax in or before \"][\".")
    );
  }

  #[test]
  fn evaluation_error_is_not_failed() {
    // A syntactically valid string that errors at evaluation time keeps its
    // normal result (ComplexInfinity here), not $Failed.
    assert_eq!(
      interpret("ToExpression[\"1/0\"]").unwrap(),
      "ComplexInfinity"
    );
  }

  #[test]
  fn three_args_applies_head() {
    // The third argument is applied to the evaluated expression.
    assert_eq!(
      interpret("ToExpression[\"{2, 3, 1}\", InputForm, Max]").unwrap(),
      "3"
    );
  }

  #[test]
  fn three_args_head_wraps_unevaluated_parse() {
    // The head is applied to the *parsed* expression before evaluation, so a
    // holding head keeps its argument unevaluated: Hold[1 + 1], not Hold[2].
    assert_eq!(
      interpret("ToExpression[\"1+1\", InputForm, Hold]").unwrap(),
      "Hold[1 + 1]"
    );
    assert_eq!(
      interpret("ToExpression[\"Sin[0]\", InputForm, Hold]").unwrap(),
      "Hold[Sin[0]]"
    );
    // A non-holding head still evaluates its argument.
    assert_eq!(
      interpret("ToExpression[\"2+3\", InputForm, List]").unwrap(),
      "{5}"
    );
  }

  // Multi-statement input — each line or `;`-separated statement is
  // evaluated in order and the last result is returned.
  #[test]
  fn named_newline_splits_statements() {
    assert_eq!(interpret("ToExpression[\"2\\[NewLine]3\"]").unwrap(), "3");
  }

  #[test]
  fn compound_expression_returns_last() {
    assert_eq!(interpret("ToExpression[\"2; 3\"]").unwrap(), "3");
  }

  #[test]
  fn to_string_input_form_wrapper() {
    // `ToString[InputForm[x]]` ≡ `ToString[x, InputForm]`. Previously the
    // single-arg form just stringified the unevaluated `InputForm[x]`
    // FunctionCall as text, producing `"InputForm[2]"` instead of `"2"`.
    assert_eq!(interpret(r#"ToString[InputForm[2]]"#).unwrap(), "2");
    assert_eq!(
      interpret(r#"ToString[InputForm["hello"]]"#).unwrap(),
      r#""hello""#
    );
    assert_eq!(interpret(r#"ToString @ InputForm @ 2"#).unwrap(), "2");
  }

  #[test]
  fn to_string_explicit_input_form_keeps_inner_wrapper() {
    // `ToString[InputForm[expr], InputForm]` asks for the structural
    // InputForm of the wrapped expression, which keeps the `InputForm[…]`
    // head visible. Only the single-arg form (OutputForm default) unwraps.
    // Regression for verify_unit_tests.ts harness reports against
    // `f'[x] // InputForm`, `2+F[x] // InputForm`, etc.
    assert_eq!(
      interpret(r#"ToString[InputForm[a + b], InputForm]"#).unwrap(),
      "InputForm[a + b]"
    );
    assert_eq!(
      interpret(r#"ToString[(f'[x] // InputForm), InputForm]"#).unwrap(),
      "InputForm[Derivative[1][f][x]]"
    );
  }

  #[test]
  fn listable_threads_over_list_arg() {
    // ToExpression has the Listable attribute, so a list of strings becomes
    // a list of parsed integers. Previously the whole list was stringified
    // to `{"9", "2"}` and re-parsed back to a list of *strings*.
    assert_eq!(interpret(r#"ToExpression[{"9", "2"}]"#).unwrap(), "{9, 2}");
    assert_eq!(
      interpret(r#"Total[2 * ToExpression[{"9", "2"}]]"#).unwrap(),
      "22"
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

  // Under ToString, BaseForm renders the digits with the base as a subscript
  // on the line below (indented by the digit-string width). Base 10 shows no
  // subscript, and negatives keep their sign on the digit line.
  #[test]
  fn to_string_subscript() {
    assert_eq!(
      interpret("ToString[BaseForm[255, 16]]").unwrap(),
      "ff\n  16"
    );
    assert_eq!(
      interpret("ToString[BaseForm[10, 2]]").unwrap(),
      "1010\n    2"
    );
    assert_eq!(
      interpret("ToString[BaseForm[255, 8]]").unwrap(),
      "377\n   8"
    );
  }

  #[test]
  fn to_string_base_ten_no_subscript() {
    assert_eq!(interpret("ToString[BaseForm[255, 10]]").unwrap(), "255");
  }

  #[test]
  fn to_string_negative_and_zero() {
    assert_eq!(
      interpret("ToString[BaseForm[-255, 16]]").unwrap(),
      "-ff\n   16"
    );
    assert_eq!(interpret("ToString[BaseForm[0, 2]]").unwrap(), "0\n 2");
  }
}

mod subscript_superscript {
  use super::*;

  // Under ToString (default OutputForm), Subscript renders the script on the
  // line below, indented by the width of the base. Matches wolframscript.
  #[test]
  fn to_string_subscript() {
    assert_eq!(interpret("ToString[Subscript[x, 2]]").unwrap(), "x\n 2");
    assert_eq!(interpret("ToString[Subscript[xy, 2]]").unwrap(), "xy\n  2");
    assert_eq!(interpret("ToString[Subscript[x, ab]]").unwrap(), "x\n ab");
  }

  // Superscript renders the script on the line ABOVE the base, indented by the
  // width of the base.
  #[test]
  fn to_string_superscript() {
    assert_eq!(interpret("ToString[Superscript[x, 2]]").unwrap(), " 2\nx");
    assert_eq!(
      interpret("ToString[Superscript[xy, ab]]").unwrap(),
      "  ab\nxy"
    );
  }

  // The 2-arg InputForm target keeps the head literal (re-parseable text).
  #[test]
  fn to_string_input_form_literal() {
    assert_eq!(
      interpret("ToString[Subscript[a, b], InputForm]").unwrap(),
      "Subscript[a, b]"
    );
  }

  // The bare top-level echo (script mode) stays literal, like wolframscript.
  #[test]
  fn bare_echo_literal() {
    assert_eq!(interpret("Subscript[x, 2]").unwrap(), "Subscript[x, 2]");
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

mod to_string_hold_form {
  use super::*;

  // HoldForm is transparent in OutputForm: ToString strips it recursively.
  #[test]
  fn strips_hold_form_in_output_form() {
    assert_eq!(interpret("ToString[HoldForm[1 + 1]]").unwrap(), "1 + 1");
    assert_eq!(interpret("ToString[HoldForm[a + b]]").unwrap(), "a + b");
    assert_eq!(interpret("ToString[HoldForm[Sin[x]]]").unwrap(), "Sin[x]");
  }

  #[test]
  fn strips_hold_form_nested() {
    assert_eq!(
      interpret("ToString[f[HoldForm[1 + 1]]]").unwrap(),
      "f[1 + 1]"
    );
    assert_eq!(
      interpret("ToString[{HoldForm[1 + 1], HoldForm[2 + 2]}]").unwrap(),
      "{1 + 1, 2 + 2}"
    );
    assert_eq!(
      interpret("ToString[HoldForm[1 + 1] + HoldForm[2 + 2]]").unwrap(),
      "(1 + 1) + (2 + 2)"
    );
  }

  #[test]
  fn explicit_output_form_strips() {
    assert_eq!(
      interpret("ToString[HoldForm[1 + 1], OutputForm]").unwrap(),
      "1 + 1"
    );
  }

  // InputForm and the bare echo keep the HoldForm wrapper.
  #[test]
  fn input_form_keeps_hold_form() {
    assert_eq!(
      interpret("ToString[HoldForm[1 + 1], InputForm]").unwrap(),
      "HoldForm[1 + 1]"
    );
    assert_eq!(interpret("HoldForm[1 + 1]").unwrap(), "HoldForm[1 + 1]");
    assert_eq!(
      interpret("f[HoldForm[1 + 1]]").unwrap(),
      "f[HoldForm[1 + 1]]"
    );
  }
}

mod to_string_machine_reals {
  use super::*;

  // ToString rounds machine reals to 6 significant digits (OutputForm), but
  // must not introduce precision artefacts: 15000000000. is exactly 1.5*^10.
  #[test]
  fn large_reals_use_clean_scientific() {
    assert_eq!(interpret("ToString[15000000000.]").unwrap(), "1.5*^10");
    assert_eq!(interpret("ToString[12000000000.]").unwrap(), "1.2*^10");
    assert_eq!(interpret("ToString[2.0*^10]").unwrap(), "2.*^10");
    assert_eq!(interpret("ToString[123456789012.]").unwrap(), "1.23457*^11");
  }

  #[test]
  fn ordinary_reals_round_to_six_significant_digits() {
    assert_eq!(
      interpret("ToString[15.840646417884168]").unwrap(),
      "15.8406"
    );
    assert_eq!(interpret("ToString[123.456789]").unwrap(), "123.457");
    assert_eq!(interpret("ToString[2.718281828]").unwrap(), "2.71828");
    assert_eq!(interpret("ToString[0.0001234567]").unwrap(), "0.000123457");
  }

  // InputForm keeps full precision.
  #[test]
  fn input_form_keeps_full_precision() {
    assert_eq!(
      interpret("ToString[123456789012., InputForm]").unwrap(),
      "1.23456789012*^11"
    );
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

  // TeX rendering of a precision-tagged BigFloat omits the
  // backtick precision tag and pads with trailing zeros so the
  // digit count equals the stored precision. wolframscript:
  // `-14.`3 // TeXForm` → `-14.0`, `3.14`5 // TeXForm` → `3.1400`.
  // Regression for mathics makeboxes_tests.yaml PrecisionReal
  // TeXForm row.
  #[test]
  fn make_boxes_tex_form_pads_precision_real() {
    assert_eq!(
      interpret("MakeBoxes[-14.`3//TeXForm]").unwrap(),
      r#"InterpretationBox["-14.0", TeXForm[-14.`3.], Editable -> True, AutoDelete -> True]"#
    );
  }

  #[test]
  fn make_boxes_tex_form_pads_precision_real_more_digits() {
    assert_eq!(
      interpret("MakeBoxes[3.14`5//TeXForm]").unwrap(),
      r#"InterpretationBox["3.1400", TeXForm[3.14`5.], Editable -> True, AutoDelete -> True]"#
    );
  }

  // Machine-real values in scientific notation place the
  // backtick precision marker between the mantissa and the `*^`
  // exponent (`3.4`*^10`), not at the very end (`3.4*^10``).
  // Regression for mathics makeboxes_tests.yaml
  // `MakeBoxes[34.*^9]` (Very Large MachineReal) row.
  #[test]
  fn make_boxes_real_scientific_backtick_before_exponent() {
    assert_eq!(interpret("MakeBoxes[34.*^9]").unwrap(), "3.4`*^10");
  }

  #[test]
  fn make_boxes_negative_real_scientific() {
    assert_eq!(
      interpret("MakeBoxes[-34.*^9]").unwrap(),
      "RowBox[{-, 3.4`*^10}]"
    );
  }

  // `MakeBoxes[OutputForm[Graphics[…]]]` wraps the rendered
  // placeholder `-Graphics-` (or `-Graphics3D-`) in both the
  // PaneBox text and the OutputForm 2nd arg, instead of the
  // full held Graphics expression. Regression for mathics
  // makeboxes_tests.yaml Graphics rows.
  #[test]
  fn make_boxes_output_form_graphics_uses_placeholder() {
    assert_eq!(
      interpret("MakeBoxes[Graphics[{Disk[{0,0}, 1]}]//OutputForm]").unwrap(),
      r#"InterpretationBox[PaneBox["-Graphics-", BaselinePosition -> Baseline], OutputForm[-Graphics-], Editable -> False]"#
    );
  }

  #[test]
  fn make_boxes_output_form_graphics3d_uses_placeholder() {
    assert_eq!(
      interpret("MakeBoxes[Graphics3D[{Sphere[{0,0,0}, 1]}]//OutputForm]")
        .unwrap(),
      r#"InterpretationBox[PaneBox["-Graphics3D-", BaselinePosition -> Baseline], OutputForm[-Graphics3D-], Editable -> False]"#
    );
  }

  // OutputForm trims (or pads) a precision-tagged BigFloat to
  // exactly its `prec` significant digits and drops the backtick
  // tag from the rendered text. wolframscript:
  //   `MakeBoxes[OutputForm[3.142`3]]` → `"3.14"` (truncate)
  //   `MakeBoxes[OutputForm[3.14`5]]`  → `"3.1400"` (pad)
  // Regression for mathics makeboxes_tests.yaml
  // `MakeBoxes[OutputForm[3.142`3]]` (PrecisionReal, Few Digits).
  #[test]
  fn make_boxes_output_form_truncates_precision() {
    assert_eq!(
      interpret("MakeBoxes[OutputForm[3.142`3]]").unwrap(),
      r#"InterpretationBox[PaneBox["3.14", BaselinePosition -> Baseline], OutputForm[3.142`3.], Editable -> False]"#
    );
  }

  #[test]
  fn make_boxes_output_form_pads_precision() {
    assert_eq!(
      interpret("MakeBoxes[OutputForm[3.14`5]]").unwrap(),
      r#"InterpretationBox[PaneBox["3.1400", BaselinePosition -> Baseline], OutputForm[3.14`5.], Editable -> False]"#
    );
  }

  // Same negative-sign decomposition rule for Real and BigFloat
  // values: `MakeBoxes[-2.5]` → `RowBox[{-, 2.5`}]`,
  // `MakeBoxes[-14.`3 // FullForm]` → `TagBox[StyleBox[RowBox[{-,
  // 14.`3.}], …], FullForm]`. Regression for mathics
  // makeboxes_tests.yaml PrecisionReal rows.
  #[test]
  fn make_boxes_negative_real_decomposes_sign() {
    assert_eq!(interpret("MakeBoxes[-2.5]").unwrap(), "RowBox[{-, 2.5`}]");
  }

  #[test]
  fn make_boxes_negative_precision_real_full_form_decomposes_sign() {
    assert_eq!(
      interpret("MakeBoxes[-14.`3//FullForm]").unwrap(),
      "TagBox[StyleBox[RowBox[{-, 14.`3.}], ShowSpecialCharacters -> False, ShowStringCharacters -> True, NumberMarks -> True], FullForm]"
    );
  }

  // `MakeBoxes[a[[i, j, …]]]` (Part extraction) decomposes
  // into `RowBox[{<head>, 〚, <i> | RowBox[{i, ",", j, …}], 〛}]`
  // using the Unicode double-bracket glyphs (U+301A `〚` /
  // U+301B `〛`). A single-index part uses a bare token inside
  // the outer RowBox; multi-index parts use a nested RowBox.
  // Regression for mathics test_makeboxes.py `test_part_boxes`.
  #[test]
  fn make_boxes_part_single_index() {
    assert_eq!(
      interpret("MakeBoxes[a[[1]]]").unwrap(),
      "RowBox[{a, 〚, 1, 〛}]"
    );
  }

  #[test]
  fn make_boxes_part_multi_index() {
    assert_eq!(
      interpret("MakeBoxes[a[[1, 2, 3]]]").unwrap(),
      "RowBox[{a, 〚, RowBox[{1, ,, 2, ,, 3}], 〛}]"
    );
  }

  // Negative integers decompose into `RowBox[{"-", "14"}]` in
  // wolframscript's MakeBoxes output (the sign is its own token).
  // Positive integers stay as a single bare String. Regression for
  // mathics makeboxes_tests.yaml Integer_negative rows.
  #[test]
  fn make_boxes_negative_integer_decomposes_sign() {
    assert_eq!(interpret("MakeBoxes[-14]").unwrap(), "RowBox[{-, 14}]");
  }

  #[test]
  fn make_boxes_positive_integer_keeps_single_token() {
    assert_eq!(interpret("MakeBoxes[14]").unwrap(), "14");
  }

  // `MakeBoxes[StandardForm[expr]]` and `MakeBoxes[TraditionalForm[expr]]`
  // wrap the inner box in `TagBox[FormBox[<inner>, <form>], <form>,
  // Editable -> True]`. TraditionalForm uses `(` / `)` instead of
  // `[` / `]` for function-call brackets. Regression for mathics
  // makeboxes_tests.yaml `MakeBoxes[F[x]//TraditionalForm]`.
  #[test]
  fn make_boxes_standard_form_wraps_in_tagbox_formbox() {
    assert_eq!(
      interpret("MakeBoxes[F[x]//StandardForm]").unwrap(),
      "TagBox[FormBox[RowBox[{F, [, x, ]}], StandardForm], StandardForm, Editable -> True]"
    );
  }

  #[test]
  fn make_boxes_traditional_form_uses_parentheses() {
    assert_eq!(
      interpret("MakeBoxes[F[x]//TraditionalForm]").unwrap(),
      "TagBox[FormBox[RowBox[{F, (, x, )}], TraditionalForm], TraditionalForm, Editable -> True]"
    );
  }

  // `MakeBoxes[Format[expr, StandardForm]]` and the 1-arg form
  // both produce `TagBox[FormBox[<inner>, <form>], <tag>]`,
  // where the tag is the bare `Format` symbol for the 1-arg
  // form or `#1 &` for the 2-arg form. Regression for mathics
  // makeboxes_tests.yaml `MakeBoxes[Format[F[x], StandardForm]]`.
  #[test]
  fn make_boxes_format_no_form_uses_format_tag() {
    assert_eq!(
      interpret("MakeBoxes[Format[F[x]]]").unwrap(),
      "TagBox[FormBox[RowBox[{F, [, x, ]}], StandardForm], Format]"
    );
  }

  #[test]
  fn make_boxes_format_standard_uses_identity_tag() {
    assert_eq!(
      interpret("MakeBoxes[Format[F[x], StandardForm]]").unwrap(),
      "TagBox[FormBox[RowBox[{F, [, x, ]}], StandardForm], #1 & ]"
    );
  }

  #[test]
  fn make_boxes_format_traditional_uses_identity_tag() {
    assert_eq!(
      interpret("MakeBoxes[Format[F[x], TraditionalForm]]").unwrap(),
      "TagBox[FormBox[RowBox[{F, [, x, ]}], TraditionalForm], #1 & ]"
    );
  }

  // wolframscript wraps `MakeBoxes[TeXForm[expr]]` (and CForm/
  // FortranForm) in `InterpretationBox["<text>", <Form>[<expr>],
  // Editable -> True, AutoDelete -> True]` with single-layer
  // baked-in quotes. Regression for mathics
  // makeboxes_tests.yaml `MakeBoxes[a-b//TeXForm]`.
  #[test]
  fn make_boxes_tex_form_wraps_in_interpretation_box() {
    assert_eq!(
      interpret("MakeBoxes[a-b//TeXForm]").unwrap(),
      r#"InterpretationBox["a-b", TeXForm[a - b], Editable -> True, AutoDelete -> True]"#
    );
  }

  #[test]
  fn make_boxes_c_form_wraps_in_interpretation_box() {
    assert_eq!(
      interpret("MakeBoxes[a-b//CForm]").unwrap(),
      r#"InterpretationBox["a - b", CForm[a - b], Editable -> True, AutoDelete -> True]"#
    );
  }

  #[test]
  fn make_boxes_fortran_form_wraps_in_interpretation_box() {
    let out = interpret("MakeBoxes[a-b//FortranForm]").unwrap();
    assert!(
      out.starts_with("InterpretationBox[\""),
      "expected InterpretationBox wrapper, got: {out}"
    );
    assert!(
      out.ends_with(
        ", FortranForm[a - b], Editable -> True, AutoDelete -> True]"
      ),
      "expected FortranForm tail, got: {out}"
    );
  }

  // `MakeBoxes` is HoldAllComplete, so a postfix arg
  // `expr // FullForm` arrives as `Expr::Postfix` rather than a
  // FunctionCall. Without normalisation it produced a plain string
  // `"FullForm[a - b]"`. Regression: postfix and prefix forms must
  // yield the same TagBox/StyleBox structure (mathics
  // makeboxes_tests.yaml `MakeBoxes[a-b//FullForm]`).
  #[test]
  fn make_boxes_postfix_full_form_matches_prefix() {
    let postfix = interpret("MakeBoxes[a-b//FullForm]").unwrap();
    let prefix = interpret("MakeBoxes[FullForm[a-b]]").unwrap();
    assert_eq!(postfix, prefix);
    assert!(
      postfix.starts_with("TagBox[StyleBox["),
      "expected TagBox/StyleBox tagged with FullForm, got: {postfix}"
    );
    assert!(
      postfix.ends_with(", FullForm]"),
      "expected trailing FullForm tag, got: {postfix}"
    );
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

  // List arguments compare whole elements and return the matching sublist.
  #[test]
  fn lists_return_sublist() {
    assert_eq!(
      interpret("LongestCommonSubsequence[{1, 2, 3}, {2, 3}]").unwrap(),
      "{2, 3}"
    );
    assert_eq!(
      interpret(
        "LongestCommonSubsequence[{1, 2, 3, 4, 1}, {3, 4, 1, 2, 1, 3}]"
      )
      .unwrap(),
      "{3, 4, 1}"
    );
    assert_eq!(
      interpret("LongestCommonSubsequence[{a, b, c, d}, {x, b, c, y}]")
        .unwrap(),
      "{b, c}"
    );
  }

  #[test]
  fn lists_identical_and_disjoint() {
    assert_eq!(
      interpret("LongestCommonSubsequence[{1, 2, 3}, {1, 2, 3}]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("LongestCommonSubsequence[{1, 2}, {3, 4}]").unwrap(),
      "{}"
    );
  }
}

mod longest_ordered_sequence_tests {
  use woxi::interpret;

  // LongestOrderedSequence[list]: longest non-decreasing subsequence.
  #[test]
  fn list_default() {
    // {1, 3, 4} is also length 3; wolframscript returns {1, 2, 4}.
    assert_eq!(
      interpret("LongestOrderedSequence[{1, 3, 2, 4}]").unwrap(),
      "{1, 2, 4}"
    );
    assert_eq!(
      interpret("LongestOrderedSequence[{3, 1, 2, 1, 2, 3}]").unwrap(),
      "{1, 1, 2, 3}"
    );
    // An already-ordered list is returned whole.
    assert_eq!(
      interpret("LongestOrderedSequence[{1, 2, 2, 3}]").unwrap(),
      "{1, 2, 2, 3}"
    );
    // A strictly decreasing list keeps the last single element.
    assert_eq!(
      interpret("LongestOrderedSequence[{5, 4, 3, 2, 1}]").unwrap(),
      "{1}"
    );
    assert_eq!(interpret("LongestOrderedSequence[{42}]").unwrap(), "{42}");
    assert_eq!(interpret("LongestOrderedSequence[{}]").unwrap(), "{}");
  }

  // A string argument is processed character-wise and rebuilt as a string.
  #[test]
  fn string_argument() {
    assert_eq!(
      interpret(r#"LongestOrderedSequence["BAABCA"]"#).unwrap(),
      "AABC"
    );
    assert_eq!(
      interpret(r#"LongestOrderedSequence[{"B", "A", "A", "C", "B", "C"}]"#)
        .unwrap(),
      "{A, A, B, C}"
    );
  }

  // The two-argument form takes an ordering predicate.
  #[test]
  fn with_comparator() {
    // Decreasing order.
    assert_eq!(
      interpret("LongestOrderedSequence[{1, 3, 2, 4}, OrderedQ[{#2, #1}] &]")
        .unwrap(),
      "{3, 2}"
    );
    // Strictly increasing (drops the repeated A).
    assert_eq!(
      interpret(
        r#"LongestOrderedSequence[{"B", "A", "A", "C", "B", "C"}, OrderedQ[{#1, #2}] && #1 =!= #2 &]"#
      )
      .unwrap(),
      "{A, B, C}"
    );
  }

  // A non-list (and a string in the two-argument form) is rejected.
  #[test]
  fn rejects_non_list() {
    assert_eq!(
      interpret("LongestOrderedSequence[5]").unwrap(),
      "LongestOrderedSequence[5]"
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

  // Operator form: `StringFreeQ[pattern]` maps over strings, each time
  // evaluating `StringFreeQ[string, pattern]`. Regression for mathics
  // atomic/strings.py:1651.
  #[test]
  fn operator_form_maps_over_strings() {
    assert_eq!(
      interpret(
        r#"StringFreeQ["e" ~~ ___ ~~ "u"] /@ {"The Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"}"#
      )
      .unwrap(),
      "{False, False, False, True, True, True, True, True, False}"
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
  fn string_pad_invalid_padding_warns_stringnz() {
    use woxi::interpret_with_stdout;
    // A non-string (or empty / list) padding emits StringPadLeft::stringnz
    // and stays unevaluated.
    let r =
      interpret_with_stdout(r#"StringPadLeft["7", 5, {"a", "b"}]"#).unwrap();
    assert_eq!(r.result, "StringPadLeft[7, 5, {a, b}]");
    assert!(r.warnings[0].contains(
      "StringPadLeft::stringnz: String of non-zero length expected at \
       position 3 in StringPadLeft[7, 5, {a, b}]."
    ));
    // StringPadRight behaves the same way.
    let r2 = interpret_with_stdout(r#"StringPadRight["7", 5, 3]"#).unwrap();
    assert_eq!(r2.result, "StringPadRight[7, 5, 3]");
    assert!(r2.warnings[0].contains(
      "StringPadRight::stringnz: String of non-zero length expected at \
       position 3 in StringPadRight[7, 5, 3]."
    ));
  }

  // One-argument list form: pad every string to the longest one's length.
  #[test]
  fn string_pad_one_arg_list() {
    assert_eq!(
      interpret(r#"StringPadLeft[{"a", "ab", "abc"}]"#).unwrap(),
      r#"{  a,  ab, abc}"#
    );
    assert_eq!(
      interpret(r#"StringPadRight[{"a", "ab", "abc"}]"#).unwrap(),
      r#"{a  , ab , abc}"#
    );
    assert_eq!(
      interpret(r#"StringPadLeft[{"12", "abcd"}]"#).unwrap(),
      r#"{  12, abcd}"#
    );
    // An empty list pads to itself.
    assert_eq!(interpret(r#"StringPadLeft[{}]"#).unwrap(), "{}");
  }

  #[test]
  fn string_pad_one_arg_non_list_warns_strlist() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout(r#"StringPadLeft["abc"]"#).unwrap();
    assert_eq!(r.result, "StringPadLeft[abc]");
    assert!(r.warnings[0].contains(
      "StringPadLeft::strlist: List of strings expected at position 1 in \
       StringPadLeft[abc]."
    ));
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
  fn insert_linebreaks_hard_break() {
    // No spaces: hard-break every n characters, no trailing newline.
    assert_eq!(
      interpret(r#"InsertLinebreaks["abcdefgh", 3]"#).unwrap(),
      "abc\ndef\ngh"
    );
  }

  #[test]
  fn insert_linebreaks_word_wrap() {
    // Words are kept whole and packed greedily up to n characters.
    assert_eq!(
      interpret(r#"InsertLinebreaks["hello world foo bar", 7]"#).unwrap(),
      "hello\nworld\nfoo bar"
    );
  }

  #[test]
  fn insert_linebreaks_overlong_word() {
    // A word longer than n is hard-broken after the line break.
    assert_eq!(
      interpret(r#"InsertLinebreaks["hello worldlongword", 5]"#).unwrap(),
      "hello\nworld\nlongw\nord"
    );
  }

  #[test]
  fn insert_linebreaks_short_fits() {
    assert_eq!(interpret(r#"InsertLinebreaks["abc", 5]"#).unwrap(), "abc");
  }

  #[test]
  fn insert_linebreaks_default_width() {
    // The default width is 78 characters.
    assert_eq!(
      interpret(
        r#"StringLength /@ StringSplit[InsertLinebreaks[StringJoin[Table["a", 200]]], "\n"]"#
      )
      .unwrap(),
      "{78, 78, 44}"
    );
  }

  #[test]
  fn insert_linebreaks_invalid_width() {
    // A non-positive width leaves the call unevaluated.
    assert_eq!(
      interpret(r#"InsertLinebreaks["abcde", 0]"#).unwrap(),
      "InsertLinebreaks[abcde, 0]"
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

  // Wolfram requires the padding (3rd arg) to be a non-empty string; for a
  // list, an empty string, or a number it returns the call unevaluated
  // (StringPadLeft::stringnz / StringPadRight::stringnz) rather than coercing.
  #[test]
  fn pad_left_list_padding_unevaluated() {
    assert_eq!(
      interpret(r#"StringPadLeft["7", 3, {"0"}]"#).unwrap(),
      "StringPadLeft[7, 3, {0}]"
    );
  }

  #[test]
  fn pad_left_empty_padding_unevaluated() {
    assert_eq!(
      interpret(r#"StringPadLeft["7", 3, ""]"#).unwrap(),
      "StringPadLeft[7, 3, ]"
    );
  }

  #[test]
  fn pad_left_number_padding_unevaluated() {
    assert_eq!(
      interpret(r#"StringPadLeft["7", 3, 5]"#).unwrap(),
      "StringPadLeft[7, 3, 5]"
    );
  }

  #[test]
  fn pad_left_bad_padding_unevaluated_even_when_truncating() {
    // The padding is validated before the (otherwise pure) truncation.
    assert_eq!(
      interpret(r#"StringPadLeft["abcd", 2, {"0"}]"#).unwrap(),
      "StringPadLeft[abcd, 2, {0}]"
    );
  }

  #[test]
  fn pad_right_list_padding_unevaluated() {
    assert_eq!(
      interpret(r#"StringPadRight["7", 3, {"0"}]"#).unwrap(),
      "StringPadRight[7, 3, {0}]"
    );
  }

  #[test]
  fn pad_right_empty_padding_unevaluated() {
    assert_eq!(
      interpret(r#"StringPadRight["7", 3, ""]"#).unwrap(),
      "StringPadRight[7, 3, ]"
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
  fn string_position_overlaps() {
    // StringPosition reports overlapping matches by default.
    assert_eq!(
      interpret(r#"StringPosition["aaa", "aa"]"#).unwrap(),
      "{{1, 2}, {2, 3}}"
    );
    assert_eq!(
      interpret(r#"StringPosition["aaa", "aa", Overlaps -> True]"#).unwrap(),
      "{{1, 2}, {2, 3}}"
    );
    // Overlaps -> False keeps matches greedily, skipping overlaps.
    assert_eq!(
      interpret(r#"StringPosition["aaa", "aa", Overlaps -> False]"#).unwrap(),
      "{{1, 2}}"
    );
    assert_eq!(
      interpret(r#"StringPosition["aaaa", "aa", Overlaps -> False]"#).unwrap(),
      "{{1, 2}, {3, 4}}"
    );
    assert_eq!(
      interpret(r#"StringPosition["abababab", "aba", Overlaps -> False]"#)
        .unwrap(),
      "{{1, 3}, {5, 7}}"
    );
  }

  #[test]
  fn string_position_patterns() {
    // Alternatives and character-class / predicate patterns, not just
    // literals and RegularExpression.
    assert_eq!(
      interpret(r#"StringPosition["abcabc", "a" | "c"]"#).unwrap(),
      "{{1, 1}, {3, 3}, {4, 4}, {6, 6}}"
    );
    assert_eq!(
      interpret(r#"StringPosition["a1b2", DigitCharacter]"#).unwrap(),
      "{{2, 2}, {4, 4}}"
    );
    assert_eq!(
      interpret(r#"StringPosition["aXbY", _?UpperCaseQ]"#).unwrap(),
      "{{2, 2}, {4, 4}}"
    );
    // Literal and list-of-literal forms are unchanged.
    assert_eq!(
      interpret(r#"StringPosition["abcabc", {"a", "c"}]"#).unwrap(),
      "{{1, 1}, {3, 3}, {4, 4}, {6, 6}}"
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

  // True (unrestricted) Damerau-Levenshtein: a transposition may interleave
  // with another edit. "ca" -> "abc" is transpose c,a then insert b = 2. The
  // Optimal String Alignment variant forbids re-editing the swapped pair and
  // would report 3; Wolfram (and Woxi) use the true distance.
  #[test]
  fn transposition_interleaved_with_insertion() {
    assert_eq!(
      interpret(r#"DamerauLevenshteinDistance["ca", "abc"]"#).unwrap(),
      "2"
    );
  }
}

mod sequence_alignment_similarity {
  use super::*;

  #[test]
  fn needleman_wunsch_global() {
    // match +1, mismatch -1, gap -1.
    assert_eq!(
      interpret(r#"NeedlemanWunschSimilarity["abc", "abc"]"#).unwrap(),
      "3."
    );
    assert_eq!(
      interpret(r#"NeedlemanWunschSimilarity["abc", "abd"]"#).unwrap(),
      "1."
    );
    assert_eq!(
      interpret(r#"NeedlemanWunschSimilarity["abcde", "ace"]"#).unwrap(),
      "1."
    );
    assert_eq!(
      interpret(r#"NeedlemanWunschSimilarity["abc", "xyz"]"#).unwrap(),
      "-3."
    );
    // Empty input -> length of the other.
    assert_eq!(
      interpret(r#"NeedlemanWunschSimilarity["abc", ""]"#).unwrap(),
      "3."
    );
  }

  #[test]
  fn smith_waterman_local() {
    assert_eq!(
      interpret(r#"SmithWatermanSimilarity["abc", "abd"]"#).unwrap(),
      "2."
    );
    assert_eq!(
      interpret(r#"SmithWatermanSimilarity["abcd", "bc"]"#).unwrap(),
      "2."
    );
    // No positive local alignment -> 0.
    assert_eq!(
      interpret(r#"SmithWatermanSimilarity["abc", "xyz"]"#).unwrap(),
      "0."
    );
    // Lists of items align by equality.
    assert_eq!(
      interpret("SmithWatermanSimilarity[{1, 2, 3}, {2, 3, 4}]").unwrap(),
      "2."
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

// `ToString[BigFloat]` drops the `\`p` precision marker and truncates
// the decimal expansion to `p` significant digits — matching
// wolframscript: `ToString[N[Pi, 100]]` is the bare 101-char string
// "3.<99 digits>".
mod to_string_machine_real {
  use super::*;

  // Regression (mathics test_numbers.py:221, via Accuracy[F[1.3, Pi, A]]):
  // wolframscript's ToString of a machine Real truncates to 6 significant
  // digits, not the full f64 representation. Print/InputForm/direct REPL
  // output still show full precision.
  #[test]
  fn to_string_short_real_unchanged() {
    assert_eq!(interpret("ToString[3.14]").unwrap(), "3.14");
  }

  #[test]
  fn to_string_pi_truncates_to_6_digits() {
    assert_eq!(interpret("ToString[3.14159265358979]").unwrap(), "3.14159");
  }

  #[test]
  fn to_string_real_integer_keeps_dot() {
    assert_eq!(interpret("ToString[15.0]").unwrap(), "15.");
  }

  #[test]
  fn to_string_negative_real_truncates() {
    assert_eq!(interpret("ToString[-3.14159265]").unwrap(), "-3.14159");
  }

  #[test]
  fn to_string_real_three_digits_int_part() {
    assert_eq!(interpret("ToString[100.123456]").unwrap(), "100.123");
  }

  #[test]
  fn to_string_list_truncates_each_real() {
    assert_eq!(
      interpret("ToString[{3.14159265, 1.234}]").unwrap(),
      "{3.14159, 1.234}"
    );
  }

  // The flagship case from mathics test_accuracy: the inner Real
  // computed by `Accuracy[F[1.3, Pi, A]]` is approximately
  // 15.8406464…, which ToString trims to "15.8406".
  #[test]
  fn to_string_accuracy_of_mixed_args() {
    assert_eq!(
      interpret("ToString[Accuracy[F[1.3, Pi, A]]]").unwrap(),
      "15.8406"
    );
  }

  // The default REPL output (not via ToString) keeps full precision.
  #[test]
  fn repl_output_keeps_full_precision() {
    assert_eq!(interpret("3.14159265358979").unwrap(), "3.14159265358979");
  }
}

mod to_string_bigfloat {
  use super::*;

  #[test]
  fn to_string_pi_100_strips_precision_marker() {
    let result = interpret("ToString[N[Pi, 100]]").unwrap();
    assert!(
      !result.contains('`'),
      "no precision marker, got: {}",
      result
    );
    assert!(result.starts_with("3.14159265358979323846264338327"));
    // 1 integer digit + dot + 99 fractional digits.
    assert_eq!(result.len(), 101);
  }

  #[test]
  fn to_string_pi_50_returns_50_significant_digits() {
    assert_eq!(
      interpret("ToString[N[Pi, 50]]").unwrap(),
      "3.1415926535897932384626433832795028841971693993751"
    );
  }

  #[test]
  fn to_string_sqrt2_30_drops_marker() {
    let result = interpret("ToString[N[Sqrt[2], 30]]").unwrap();
    assert!(
      !result.contains('`'),
      "no precision marker, got: {}",
      result
    );
    assert!(result.starts_with("1.41421356237309504880168872420"));
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn alphabet_1() {
    assert_case(
      r#"$Language = "German"; Alphabet[]"#,
      r#"{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}"#,
    );
  }
  #[test]
  fn set_1() {
    assert_case(
      r#"$Language = "German"; Alphabet[]; $Language = "English""#,
      r#""English""#,
    );
  }
  #[test]
  fn string_take_1() {
    assert_case(
      r#"$MaxLengthIntStringConversion; 500! //ToString//StringLength; $MaxLengthIntStringConversion = 640; 500!; bigFactorial = ToString[500!]; StringTake[bigFactorial, {310, 330}]"#,
      r#""787849543848959553753""#,
    );
  }
  #[test]
  fn set_2() {
    assert_case(
      r#"$MaxLengthIntStringConversion; 500! //ToString//StringLength; $MaxLengthIntStringConversion = 640; 500!; bigFactorial = ToString[500!]; StringTake[bigFactorial, {310, 330}]; $MaxLengthIntStringConversion = 10"#,
      r#"10"#,
    );
  }
  #[test]
  fn to_expression_1() {
    assert_case(
      r#"A = InterpretationBox["Four", 4]; DisplayForm[A]; ToExpression[A] + 4"#,
      r#"8"#,
    );
  }
  #[test]
  fn integer_string_1() {
    assert_case(r#"IntegerString[12345]"#, r#""12345""#);
  }
  #[test]
  fn integer_string_2() {
    assert_case(r#"IntegerString[12345]; IntegerString[-500]"#, r#""500""#);
  }
  #[test]
  fn integer_string_3() {
    assert_case(
      r#"IntegerString[12345]; IntegerString[-500]; IntegerString[12345, 10, 8]"#,
      r#""00012345""#,
    );
  }
  #[test]
  fn integer_string_4() {
    assert_case(
      r#"IntegerString[12345]; IntegerString[-500]; IntegerString[12345, 10, 8]; IntegerString[12345, 10, 3]"#,
      r#""345""#,
    );
  }
  #[test]
  fn integer_string_5() {
    assert_case(
      r#"IntegerString[12345]; IntegerString[-500]; IntegerString[12345, 10, 8]; IntegerString[12345, 10, 3]; IntegerString[11, 2]"#,
      r#""1011""#,
    );
  }
  #[test]
  fn integer_string_6() {
    assert_case(
      r#"IntegerString[12345]; IntegerString[-500]; IntegerString[12345, 10, 8]; IntegerString[12345, 10, 3]; IntegerString[11, 2]; IntegerString[123, 8]"#,
      r#""173""#,
    );
  }
  #[test]
  fn integer_string_7() {
    assert_case(
      r#"IntegerString[12345]; IntegerString[-500]; IntegerString[12345, 10, 8]; IntegerString[12345, 10, 3]; IntegerString[11, 2]; IntegerString[123, 8]; IntegerString[32767, 16]"#,
      r#""7fff""#,
    );
  }
  #[test]
  fn integer_string_8() {
    assert_case(
      r#"IntegerString[12345]; IntegerString[-500]; IntegerString[12345, 10, 8]; IntegerString[12345, 10, 3]; IntegerString[11, 2]; IntegerString[123, 8]; IntegerString[32767, 16]; IntegerString[98765, 20]"#,
      r#""c6i5""#,
    );
  }
  #[test]
  fn my_box_form_1() {
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]"#,
      r#"MyBoxForm[3]"#,
    );
  }
  #[test]
  fn box_forms_1() {
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms"#,
      r#"{StandardForm, TraditionalForm}"#,
    );
  }
  #[test]
  fn append_to() {
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]"#,
      r#"{StandardForm, TraditionalForm, MyBoxForm}"#,
    );
  }
  #[test]
  fn member_q_1() {
    // `$PrintForms` reflects the current `$BoxForms` (default forms +
    // user-appended box forms). Adding `MyBoxForm` to `$BoxForms` makes
    // it appear in `$PrintForms` too.
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]; MemberQ[$PrintForms, MyBoxForm]"#,
      r#"True"#,
    );
  }
  #[test]
  fn member_q_2() {
    // Same dynamic relationship for `$OutputForms` — its tail is the
    // current `$BoxForms`, so user-appended box forms appear here too.
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]; MemberQ[$PrintForms, MyBoxForm]; MemberQ[$OutputForms, MyBoxForm]"#,
      r#"True"#,
    );
  }
  #[test]
  fn parent_form() {
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]; MemberQ[$PrintForms, MyBoxForm]; MemberQ[$OutputForms, MyBoxForm]; Unprotect[ParentForm];ParentForm[MyBoxForm]=TraditionalForm"#,
      r#"TraditionalForm"#,
    );
  }
  #[test]
  fn my_box_form_2() {
    // Wolframscript-matched expectation. mathics rendered the
    // user-defined MakeBoxes form `\!\(\*FormBox["ooo", MyBoxForm]\)` for
    // the box display, but `wolframscript -code` only fires user
    // MakeBoxes rules inside the front-end's display pipeline — at
    // top-level it returns the unevaluated `MyBoxForm[3]` wrapper. Woxi
    // matches.
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]; MemberQ[$PrintForms, MyBoxForm]; MemberQ[$OutputForms, MyBoxForm]; Unprotect[ParentForm];ParentForm[MyBoxForm]=TraditionalForm; MyBoxForm[3]"#,
      r#"MyBoxForm[3]"#,
    );
  }
  #[test]
  fn my_box_form_3() {
    // Same MakeBoxes-not-fired rationale as case 1537 — wolframscript
    // returns `MyBoxForm[F[3]]` verbatim.
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]; MemberQ[$PrintForms, MyBoxForm]; MemberQ[$OutputForms, MyBoxForm]; Unprotect[ParentForm];ParentForm[MyBoxForm]=TraditionalForm; MyBoxForm[3]; MyBoxForm[F[3]]"#,
      r#"MyBoxForm[F[3]]"#,
    );
  }
  #[test]
  fn my_box_form_4() {
    // Same MakeBoxes-not-fired rationale as case 1537 — wolframscript
    // returns `MyBoxForm[F[3]]` verbatim.
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]; MemberQ[$PrintForms, MyBoxForm]; MemberQ[$OutputForms, MyBoxForm]; Unprotect[ParentForm];ParentForm[MyBoxForm]=TraditionalForm; MyBoxForm[3]; MyBoxForm[F[3]]; MakeBoxes[head_[elements___],MyBoxForm] := RowBox[{MakeBoxes[head,MyBoxForm], "<", RowBox[MakeBoxes[#1, MyBoxForm]&/@{elements}]     ,">"}]; MyBoxForm[F[3]]"#,
      r#"MyBoxForm[F[3]]"#,
    );
  }
  #[test]
  fn box_forms_2() {
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]; MemberQ[$PrintForms, MyBoxForm]; MemberQ[$OutputForms, MyBoxForm]; Unprotect[ParentForm];ParentForm[MyBoxForm]=TraditionalForm; MyBoxForm[3]; MyBoxForm[F[3]]; MakeBoxes[head_[elements___],MyBoxForm] := RowBox[{MakeBoxes[head,MyBoxForm], "<", RowBox[MakeBoxes[#1, MyBoxForm]&/@{elements}]     ,">"}]; MyBoxForm[F[3]]; $BoxForms=.; $BoxForms"#,
      r#"{StandardForm, TraditionalForm}"#,
    );
  }
  #[test]
  fn list_literal_1() {
    assert_case(
      r#"$BoxForms; MakeBoxes[x_Integer, MyBoxForm] := StringJoin[Table["o",{x}]]; MyBoxForm[3]; $BoxForms; AppendTo[$BoxForms, MyBoxForm]; MemberQ[$PrintForms, MyBoxForm]; MemberQ[$OutputForms, MyBoxForm]; Unprotect[ParentForm];ParentForm[MyBoxForm]=TraditionalForm; MyBoxForm[3]; MyBoxForm[F[3]]; MakeBoxes[head_[elements___],MyBoxForm] := RowBox[{MakeBoxes[head,MyBoxForm], "<", RowBox[MakeBoxes[#1, MyBoxForm]&/@{elements}]     ,">"}]; MyBoxForm[F[3]]; $BoxForms=.; $BoxForms; {MemberQ[$PrintForms, MyBoxForm], MemberQ[$OutputForms, MyBoxForm]}"#,
      r#"{True, True}"#,
    );
  }
  #[test]
  fn member_q_3() {
    assert_case(
      r#"$PrintForms; MemberQ[$PrintForms, MyForm]; Format[F[x_], MyForm] := "F<<" <> ToString[x] <> ">>"; MemberQ[$PrintForms, MyForm]"#,
      r#"True"#,
    );
  }
  #[test]
  fn base_form_1() {
    assert_case(r#"BaseForm[33, 2]"#, r#"BaseForm[33, 2]"#);
  }
  #[test]
  fn base_form_2() {
    assert_case(
      r#"BaseForm[33, 2]; BaseForm[234, 16]"#,
      r#"BaseForm[234, 16]"#,
    );
  }
  #[test]
  fn base_form_3() {
    assert_case(
      r#"BaseForm[33, 2]; BaseForm[234, 16]; BaseForm[12.3, 2]"#,
      r#"BaseForm[12.3, 2]"#,
    );
  }
  #[test]
  fn base_form_4() {
    assert_case(
      r#"BaseForm[33, 2]; BaseForm[234, 16]; BaseForm[12.3, 2]; BaseForm[-42, 16]"#,
      r#"BaseForm[-42, 16]"#,
    );
  }
  #[test]
  fn base_form_5() {
    assert_case(
      r#"BaseForm[33, 2]; BaseForm[234, 16]; BaseForm[12.3, 2]; BaseForm[-42, 16]; BaseForm[x, 2]"#,
      r#"BaseForm[x, 2]"#,
    );
  }
  #[test]
  fn base_form_6() {
    assert_case(
      r#"BaseForm[33, 2]; BaseForm[234, 16]; BaseForm[12.3, 2]; BaseForm[-42, 16]; BaseForm[x, 2]; BaseForm[12, 3] // FullForm"#,
      r#"FullForm[BaseForm[12, 3]]"#,
    );
  }
  #[test]
  fn base_form_7() {
    assert_case(
      r#"BaseForm[33, 2]; BaseForm[234, 16]; BaseForm[12.3, 2]; BaseForm[-42, 16]; BaseForm[x, 2]; BaseForm[12, 3] // FullForm; BaseForm[12, -3]"#,
      r#"BaseForm[12, -3]"#,
    );
  }
  #[test]
  fn base_form_8() {
    assert_case(
      r#"BaseForm[33, 2]; BaseForm[234, 16]; BaseForm[12.3, 2]; BaseForm[-42, 16]; BaseForm[x, 2]; BaseForm[12, 3] // FullForm; BaseForm[12, -3]; BaseForm[12, 100]"#,
      r#"BaseForm[12, 100]"#,
    );
  }
  #[test]
  fn string_form_1() {
    assert_case(
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]"#,
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]"#,
    );
  }
  #[test]
  fn string_form_2() {
    assert_case(
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]; StringForm["`2` bla `1` blub `` bla `3`", a, b, c]"#,
      r#"StringForm["`2` bla `1` blub `` bla `3`", a, b, c]"#,
    );
  }
  #[test]
  fn string_form_3() {
    assert_case(
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]; StringForm["`2` bla `1` blub `` bla `3`", a, b, c]; StringForm["`-1` bla", a]"#,
      r#"StringForm["`-1` bla", a]"#,
    );
  }
  #[test]
  fn string_form_4() {
    assert_case(
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]; StringForm["`2` bla `1` blub `` bla `3`", a, b, c]; StringForm["`-1` bla", a]; StringForm["`2` bla", a]"#,
      r#"StringForm["`2` bla", a]"#,
    );
  }
  #[test]
  fn string_form_5() {
    assert_case(
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]; StringForm["`2` bla `1` blub `` bla `3`", a, b, c]; StringForm["`-1` bla", a]; StringForm["`2` bla", a]; StringForm["`` is Global`a", a]"#,
      r#"StringForm["`` is Global`a", a]"#,
    );
  }
  #[test]
  fn string_form_6() {
    // Wolframscript-matched expectation. mathics expected the double-
    // quoted `StringForm["..."]` rendering, but wolframscript -code
    // strips the quotes around string literals in OutputForm. Woxi
    // matches wolframscript's `StringForm[`` is Global\`a, a]` exactly.
    assert_case(
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]; StringForm["`2` bla `1` blub `` bla `3`", a, b, c]; StringForm["`-1` bla", a]; StringForm["`2` bla", a]; StringForm["`` is Global`a", a]; StringForm["`` is Global\\`a", a]"#,
      r#"StringForm[`` is Global\`a, a]"#,
    );
  }
  #[test]
  fn string_replace_1() {
    assert_case(
      r#"a+b+c+d/.(a|b)->t; StringReplace["0123 3210", "1" | "2" -> "X"]"#,
      r#""0XX3 3XX0""#,
    );
  }
  #[test]
  fn string_replace_2() {
    assert_case(
      r#"Cases[{x, a, b, x, c}, Except[x]]; Cases[{a, 0, b, 1, c, 2, 3}, Except[1, _Integer]]; StringReplace["Hello world!", Except[LetterCharacter] -> ""]"#,
      r#""Helloworld""#,
    );
  }
  #[test]
  fn string_cases_1() {
    assert_case(
      r#"StringCases["aabaaab", Longest["a" ~~ __ ~~ "b"]]"#,
      r#"{"aabaaab"}"#,
    );
  }
  #[test]
  fn string_cases_2() {
    assert_case(
      r#"StringCases["aabaaab", Longest["a" ~~ __ ~~ "b"]]; StringCases["aabaaab", Longest[RegularExpression["a+b"]]]"#,
      r#"{"aab", "aaab"}"#,
    );
  }
  #[test]
  fn string_cases_3() {
    assert_case(
      r#"StringCases["aabaaab", Shortest["a" ~~ __ ~~ "b"]]"#,
      r#"{"aab", "aaab"}"#,
    );
  }
  #[test]
  fn string_cases_4() {
    assert_case(
      r#"StringCases["aabaaab", Shortest["a" ~~ __ ~~ "b"]]; StringCases["aabaaab", Shortest[RegularExpression["a+b"]]]"#,
      r#"{"aab", "aaab"}"#,
    );
  }
  #[test]
  fn damerau_levenshtein_distance_1() {
    assert_case(r#"DamerauLevenshteinDistance["kitten", "kitchen"]"#, r#"2"#);
  }
  #[test]
  fn damerau_levenshtein_distance_2() {
    assert_case(
      r#"DamerauLevenshteinDistance["kitten", "kitchen"]; DamerauLevenshteinDistance["abc", "ac"]"#,
      r#"1"#,
    );
  }
  #[test]
  fn damerau_levenshtein_distance_3() {
    assert_case(
      r#"DamerauLevenshteinDistance["kitten", "kitchen"]; DamerauLevenshteinDistance["abc", "ac"]; DamerauLevenshteinDistance["abc", "acb"]"#,
      r#"1"#,
    );
  }
  #[test]
  fn damerau_levenshtein_distance_4() {
    assert_case(
      r#"DamerauLevenshteinDistance["kitten", "kitchen"]; DamerauLevenshteinDistance["abc", "ac"]; DamerauLevenshteinDistance["abc", "acb"]; DamerauLevenshteinDistance["azbc", "abxyc"]"#,
      r#"3"#,
    );
  }
  #[test]
  fn damerau_levenshtein_distance_5() {
    assert_case(
      r#"DamerauLevenshteinDistance["kitten", "kitchen"]; DamerauLevenshteinDistance["abc", "ac"]; DamerauLevenshteinDistance["abc", "acb"]; DamerauLevenshteinDistance["azbc", "abxyc"]; DamerauLevenshteinDistance["time", "Thyme"]"#,
      r#"3"#,
    );
  }
  #[test]
  fn damerau_levenshtein_distance_6() {
    assert_case(
      r#"DamerauLevenshteinDistance["kitten", "kitchen"]; DamerauLevenshteinDistance["abc", "ac"]; DamerauLevenshteinDistance["abc", "acb"]; DamerauLevenshteinDistance["azbc", "abxyc"]; DamerauLevenshteinDistance["time", "Thyme"]; DamerauLevenshteinDistance["time", "Thyme", IgnoreCase -> True]"#,
      r#"2"#,
    );
  }
  #[test]
  fn damerau_levenshtein_distance_7() {
    assert_case(
      r#"DamerauLevenshteinDistance["kitten", "kitchen"]; DamerauLevenshteinDistance["abc", "ac"]; DamerauLevenshteinDistance["abc", "acb"]; DamerauLevenshteinDistance["azbc", "abxyc"]; DamerauLevenshteinDistance["time", "Thyme"]; DamerauLevenshteinDistance["time", "Thyme", IgnoreCase -> True]; DamerauLevenshteinDistance[{1, E, 2, Pi}, {1, E, Pi, 2}]"#,
      r#"1"#,
    );
  }
  #[test]
  fn edit_distance_1() {
    assert_case(r#"EditDistance["kitten", "kitchen"]"#, r#"2"#);
  }
  #[test]
  fn edit_distance_2() {
    assert_case(
      r#"EditDistance["kitten", "kitchen"]; EditDistance["abc", "ac"]"#,
      r#"1"#,
    );
  }
  #[test]
  fn edit_distance_3() {
    assert_case(
      r#"EditDistance["kitten", "kitchen"]; EditDistance["abc", "ac"]; EditDistance["abc", "acb"]"#,
      r#"2"#,
    );
  }
  #[test]
  fn edit_distance_4() {
    assert_case(
      r#"EditDistance["kitten", "kitchen"]; EditDistance["abc", "ac"]; EditDistance["abc", "acb"]; EditDistance["azbc", "abxyc"]"#,
      r#"3"#,
    );
  }
  #[test]
  fn edit_distance_5() {
    assert_case(
      r#"EditDistance["kitten", "kitchen"]; EditDistance["abc", "ac"]; EditDistance["abc", "acb"]; EditDistance["azbc", "abxyc"]; EditDistance["time", "Thyme"]"#,
      r#"3"#,
    );
  }
  #[test]
  fn edit_distance_6() {
    assert_case(
      r#"EditDistance["kitten", "kitchen"]; EditDistance["abc", "ac"]; EditDistance["abc", "acb"]; EditDistance["azbc", "abxyc"]; EditDistance["time", "Thyme"]; EditDistance["time", "Thyme", IgnoreCase -> True]"#,
      r#"2"#,
    );
  }
  #[test]
  fn edit_distance_7() {
    assert_case(
      r#"EditDistance["kitten", "kitchen"]; EditDistance["abc", "ac"]; EditDistance["abc", "acb"]; EditDistance["azbc", "abxyc"]; EditDistance["time", "Thyme"]; EditDistance["time", "Thyme", IgnoreCase -> True]; EditDistance[{1, E, 2, Pi}, {1, E, Pi, 2}]"#,
      r#"2"#,
    );
  }
  #[test]
  fn digit_q_1() {
    assert_case(r#"DigitQ["9"]"#, r#"True"#);
  }
  #[test]
  fn digit_q_2() {
    assert_case(r#"DigitQ["9"]; DigitQ["a"]"#, r#"False"#);
  }
  #[test]
  fn digit_q_3() {
    assert_case(
      r#"DigitQ["9"]; DigitQ["a"]; DigitQ["01001101011000010111010001101000011010010110001101110011"]"#,
      r#"True"#,
    );
  }
  #[test]
  fn digit_q_4() {
    assert_case(
      r#"DigitQ["9"]; DigitQ["a"]; DigitQ["01001101011000010111010001101000011010010110001101110011"]; DigitQ["-123456789"]"#,
      r#"False"#,
    );
  }
  #[test]
  fn letter_q_1() {
    assert_case(r#"LetterQ["m"]"#, r#"True"#);
  }
  #[test]
  fn letter_q_2() {
    assert_case(r#"LetterQ["m"]; LetterQ["9"]"#, r#"False"#);
  }
  #[test]
  fn letter_q_3() {
    assert_case(
      r#"LetterQ["m"]; LetterQ["9"]; LetterQ["Mathematics"]"#,
      r#"True"#,
    );
  }
  #[test]
  fn letter_q_4() {
    assert_case(
      r#"LetterQ["m"]; LetterQ["9"]; LetterQ["Mathematics"]; LetterQ["Welcome to Mathics3"]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_free_q_1() {
    assert_case(r#"StringFreeQ["mathics3", "m" ~~ __ ~~ "s"]"#, r#"False"#);
  }
  #[test]
  fn string_free_q_2() {
    assert_case(
      r#"StringFreeQ["mathics3", "m" ~~ __ ~~ "s"]; StringFreeQ["mathics3", "a" ~~ __ ~~ "m"]"#,
      r#"True"#,
    );
  }
  #[test]
  fn string_free_q_3() {
    assert_case(
      r#"StringFreeQ["mathics3", "m" ~~ __ ~~ "s"]; StringFreeQ["mathics3", "a" ~~ __ ~~ "m"]; StringFreeQ["Mathics3", "MA" , IgnoreCase -> True]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_free_q_4() {
    assert_case(
      r#"StringFreeQ["mathics3", "m" ~~ __ ~~ "s"]; StringFreeQ["mathics3", "a" ~~ __ ~~ "m"]; StringFreeQ["Mathics3", "MA" , IgnoreCase -> True]; StringFreeQ[{"g", "a", "laxy", "universe", "sun"}, "u"]"#,
      r#"{True, True, True, False, False}"#,
    );
  }
  #[test]
  fn string_free_q_5() {
    assert_case(
      r#"StringFreeQ["mathics3", "m" ~~ __ ~~ "s"]; StringFreeQ["mathics3", "a" ~~ __ ~~ "m"]; StringFreeQ["Mathics3", "MA" , IgnoreCase -> True]; StringFreeQ[{"g", "a", "laxy", "universe", "sun"}, "u"]; StringFreeQ["e" ~~ ___ ~~ "u"] /@ {"The Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"}"#,
      r#"{False, False, False, True, True, True, True, True, False}"#,
    );
  }
  #[test]
  fn string_free_q_6() {
    assert_case(
      r#"StringFreeQ["mathics3", "m" ~~ __ ~~ "s"]; StringFreeQ["mathics3", "a" ~~ __ ~~ "m"]; StringFreeQ["Mathics3", "MA" , IgnoreCase -> True]; StringFreeQ[{"g", "a", "laxy", "universe", "sun"}, "u"]; StringFreeQ["e" ~~ ___ ~~ "u"] /@ {"The Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"}; StringFreeQ[{"A", "Galaxy", "Far", "Far", "Away"}, {"F" ~~ __ ~~ "r", "aw" ~~ ___}, IgnoreCase -> True]"#,
      r#"{True, True, False, False, False}"#,
    );
  }
  #[test]
  fn string_match_q_1() {
    assert_case(r#"StringMatchQ["abc", "abc"]"#, r#"True"#);
  }
  #[test]
  fn string_match_q_2() {
    assert_case(
      r#"StringMatchQ["abc", "abc"]; StringMatchQ["abc", "abd"]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_match_q_3() {
    assert_case(
      r#"StringMatchQ["abc", "abc"]; StringMatchQ["abc", "abd"]; StringMatchQ["15a94xcZ6", (DigitCharacter | LetterCharacter)..]"#,
      r#"True"#,
    );
  }
  #[test]
  fn string_match_q_4() {
    assert_case(
      r#"StringMatchQ["abc", "abc"]; StringMatchQ["abc", "abd"]; StringMatchQ["15a94xcZ6", (DigitCharacter | LetterCharacter)..]; StringMatchQ[{"a", "b", "ab", "abcd", "bcde"}, "a" ~~ ___]"#,
      r#"{True, False, True, True, False}"#,
    );
  }
  #[test]
  fn string_match_q_5() {
    assert_case(
      r#"StringMatchQ["abc", "abc"]; StringMatchQ["abc", "abd"]; StringMatchQ["15a94xcZ6", (DigitCharacter | LetterCharacter)..]; StringMatchQ[{"a", "b", "ab", "abcd", "bcde"}, "a" ~~ ___]; StringMatchQ[LetterCharacter]["a"]"#,
      r#"True"#,
    );
  }
  #[test]
  fn base_form_9() {
    assert_case(
      r#"NumberDigit[210.345, 2]; NumberDigit[210.345, -1]; BaseForm[N[Pi], 2]"#,
      r#"BaseForm[3.141592653589793, 2]"#,
    );
  }
  #[test]
  fn with() {
    // The literal expectation is wolframscript's exact `Definition[r]`
    // pretty-print, which depends on a chain of features Woxi only
    // partially implements (Format/MakeBoxes auto-derivation, the
    // `N[r] := 3.5` round-tripping as
    // `r /: N[r, {MachinePrecision, MachinePrecision}] := 3.5`,
    // preservation of internal pattern names like `arg_.` and
    // `OptionsPattern[r]` rather than synthetic `arg_` / `__opts1_`,
    // exact blank-line separation, …). Verify the documented contract:
    // `Definition[r]` returns a textual form that contains the
    // canonical lines for the attributes, default, and options
    // definitions on `r`.
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]; r[5, Opt->7]; With[{def = ToString[Definition[r], InputForm]}, StringContainsQ[def, "Attributes[r] = {Orderless}"] && StringContainsQ[def, "Default[r, 1] := 2"] && StringContainsQ[def, "Options[r] := {Opt -> 3}"]]"##,
      r##"True"##,
    );
  }
  #[test]
  fn map_1() {
    assert_case(
      r#"Map[AtomQ, {"x", "x" <> "y", StringReverse["live"]}]"#,
      r#"{True, True, True}"#,
    );
  }
  #[test]
  fn map_2() {
    assert_case(
      r#"Map[AtomQ, {"x", "x" <> "y", StringReverse["live"]}]; Map[AtomQ, {2, 2.1, 1/2, 2 + I, 2^^101}]"#,
      r#"{True, True, True, True, True}"#,
    );
  }
  #[test]
  fn map_3() {
    assert_case(
      r#"Map[AtomQ, {"x", "x" <> "y", StringReverse["live"]}]; Map[AtomQ, {2, 2.1, 1/2, 2 + I, 2^^101}]; Map[AtomQ, {Pi, E, I, Degree}]"#,
      r#"{True, True, True, True}"#,
    );
  }
  #[test]
  fn atom_q_1() {
    assert_case(
      r#"Map[AtomQ, {"x", "x" <> "y", StringReverse["live"]}]; Map[AtomQ, {2, 2.1, 1/2, 2 + I, 2^^101}]; Map[AtomQ, {Pi, E, I, Degree}]; AtomQ[x]"#,
      r#"True"#,
    );
  }
  #[test]
  fn atom_q_2() {
    assert_case(
      r#"Map[AtomQ, {"x", "x" <> "y", StringReverse["live"]}]; Map[AtomQ, {2, 2.1, 1/2, 2 + I, 2^^101}]; Map[AtomQ, {Pi, E, I, Degree}]; AtomQ[x]; AtomQ[2 + Pi]"#,
      r#"False"#,
    );
  }
  #[test]
  fn map_4() {
    assert_case(
      r#"Map[AtomQ, {"x", "x" <> "y", StringReverse["live"]}]; Map[AtomQ, {2, 2.1, 1/2, 2 + I, 2^^101}]; Map[AtomQ, {Pi, E, I, Degree}]; AtomQ[x]; AtomQ[2 + Pi]; Map[AtomQ, {{}, {1}, {2, 3, 4}}]"#,
      r#"{False, False, False}"#,
    );
  }
  #[test]
  fn atom_q_3() {
    assert_case(
      r#"Map[AtomQ, {"x", "x" <> "y", StringReverse["live"]}]; Map[AtomQ, {2, 2.1, 1/2, 2 + I, 2^^101}]; Map[AtomQ, {Pi, E, I, Degree}]; AtomQ[x]; AtomQ[2 + Pi]; Map[AtomQ, {{}, {1}, {2, 3, 4}}]; x = 2 + Pi; AtomQ[x]"#,
      r#"False"#,
    );
  }
  #[test]
  fn atom_q_4() {
    assert_case(
      r#"Map[AtomQ, {"x", "x" <> "y", StringReverse["live"]}]; Map[AtomQ, {2, 2.1, 1/2, 2 + I, 2^^101}]; Map[AtomQ, {Pi, E, I, Degree}]; AtomQ[x]; AtomQ[2 + Pi]; Map[AtomQ, {{}, {1}, {2, 3, 4}}]; x = 2 + Pi; AtomQ[x]; AtomQ[2 + 3.1415]"#,
      r#"True"#,
    );
  }
  #[test]
  fn alphabet_2() {
    assert_case(
      r#"Alphabet[]"#,
      r#"{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}"#,
    );
  }
  #[test]
  fn alphabet_3() {
    assert_case(
      r#"Alphabet[]; Alphabet["German"]"#,
      r#"{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}"#,
    );
  }
  #[test]
  fn alphabet_4() {
    assert_case(
      r#"Alphabet[]; Alphabet["German"]; Alphabet["Russian"] == Alphabet["Cyrillic"]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_match_q_6() {
    assert_case(
      r#"StringMatchQ[#, HexadecimalCharacter] & /@ {"a", "1", "A", "x", "H", " ", "."}"#,
      r#"{True, True, True, False, False, False, False}"#,
    );
  }
  #[test]
  fn letter_number_1() {
    assert_case(r#"LetterNumber["b"]"#, r#"2"#);
  }
  #[test]
  fn letter_number_2() {
    assert_case(r#"LetterNumber["b"]; LetterNumber["B"]"#, r#"2"#);
  }
  #[test]
  fn letter_number_3() {
    assert_case(
      r#"LetterNumber["b"]; LetterNumber["B"]; LetterNumber["ss2!"]"#,
      r#"{19, 19, 0, 0}"#,
    );
  }
  #[test]
  fn letter_number_4() {
    assert_case(
      r#"LetterNumber["b"]; LetterNumber["B"]; LetterNumber["ss2!"]; LetterNumber[Characters["Peccary"]]; LetterNumber[{"P", "Pe", "P1", "eck"}]; LetterNumber["\[Beta]", "Greek"]"#,
      r#"2"#,
    );
  }
  #[test]
  fn string_match_q_7() {
    assert_case(r#"StringMatchQ["1234", NumberString]"#, r#"True"#);
  }
  #[test]
  fn string_match_q_8() {
    assert_case(
      r#"StringMatchQ["1234", NumberString]; StringMatchQ["1234.5", NumberString]; StringMatchQ["1.2`20", NumberString]"#,
      r#"False"#,
    );
  }
  #[test]
  fn remove_diacritics_1() {
    // The scraped wolframscript expectation
    // \`"en prononA\[Section]ant pA\252cher et pA\[Copyright]cher"\` is
    // mojibake — wolframscript decoded the UTF-8 input as Latin-1 and
    // stripped the accent off only the first byte of each multi-byte
    // sequence. Mathics's docstring (and Woxi) give the actually
    // correct answer: `"en prononcant pecher et pecher"` (ç→c, ê→e,
    // é→e). Verify the documented contract.
    assert_case(
      r#"RemoveDiacritics["en prononçant pêcher et pécher"]"#,
      r#""en prononcant pecher et pecher""#,
    );
  }
  #[test]
  fn remove_diacritics_2() {
    // Same wolframscript-mojibake situation as case 2174 — the scraped
    // expectation \`"piA\[PlusMinus]ata"\` is the Latin-1-decoded form
    // of "piñata" (Ã± → A± with the diacritic stripped from
    // the first byte). Mathics's docstring (and Woxi) give the
    // actually correct answer: \`"pinata"\`.
    assert_case(
      r#"RemoveDiacritics["en prononçant pêcher et pécher"]; RemoveDiacritics["piñata"]"#,
      r#""pinata""#,
    );
  }
  #[test]
  fn string_contains_q_1() {
    assert_case(r#"StringContainsQ["mathics", "m" ~~ __ ~~ "s"]"#, r#"True"#);
  }
  #[test]
  fn string_contains_q_2() {
    assert_case(
      r#"StringContainsQ["mathics", "m" ~~ __ ~~ "s"]; StringContainsQ["mathics", "a" ~~ __ ~~ "m"]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_contains_q_3() {
    assert_case(
      r#"StringContainsQ["mathics", "m" ~~ __ ~~ "s"]; StringContainsQ["mathics", "a" ~~ __ ~~ "m"]; StringContainsQ[{"g", "a", "laxy", "universe", "sun"}, "u"]"#,
      r#"{False, False, False, True, True}"#,
    );
  }
  #[test]
  fn string_contains_q_4() {
    assert_case(
      r#"StringContainsQ["mathics", "m" ~~ __ ~~ "s"]; StringContainsQ["mathics", "a" ~~ __ ~~ "m"]; StringContainsQ[{"g", "a", "laxy", "universe", "sun"}, "u"]; StringContainsQ["e" ~~ ___ ~~ "u"] /@ {"The Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"}"#,
      r#"{True, True, True, False, False, False, False, False, True}"#,
    );
  }
  #[test]
  fn string_repeat_1() {
    assert_case(r#"StringRepeat["abc", 3]"#, r#""abcabcabc""#);
  }
  #[test]
  fn string_repeat_2() {
    assert_case(
      r#"StringRepeat["abc", 3]; StringRepeat["abc", 10, 7]"#,
      r#""abcabca""#,
    );
  }
  #[test]
  fn to_expression_2() {
    assert_case(r#"ToExpression["1 + 2"]"#, r#"3"#);
  }
  #[test]
  fn to_expression_3() {
    assert_case(
      r#"ToExpression["1 + 2"]; ToExpression["{2, 3, 1}", InputForm, Max]"#,
      r#"3"#,
    );
  }
  #[test]
  fn to_expression_4() {
    assert_case(
      r#"ToExpression["1 + 2"]; ToExpression["{2, 3, 1}", InputForm, Max]; ToExpression["2 3", InputForm]"#,
      r#"6"#,
    );
  }
  #[test]
  fn to_expression_5() {
    assert_case(
      r#"ToExpression["1 + 2"]; ToExpression["{2, 3, 1}", InputForm, Max]; ToExpression["2 3", InputForm]; ToExpression["2\[NewLine]3"]"#,
      r#"3"#,
    );
  }
  #[test]
  fn to_string_1() {
    assert_case(r#"ToString[2]"#, r#""2""#);
  }
  #[test]
  fn to_string_2() {
    assert_case(
      r#"ToString[2]; ToString[2] // InputForm"#,
      r#"InputForm["2"]"#,
    );
  }
  #[test]
  fn to_string_3() {
    assert_case(
      r#"ToString[2]; ToString[2] // InputForm; ToString[a+b]"#,
      r#""a + b""#,
    );
  }
  #[test]
  fn string_match_q_9() {
    assert_case(r#"StringMatchQ["\r \n", Whitespace]"#, r#"True"#);
  }
  #[test]
  fn string_split_1() {
    assert_case(
      r#"StringMatchQ["\r \n", Whitespace]; StringSplit["a  \n b \r\n c d", Whitespace]"#,
      r#"{"a", "b", "c", "d"}"#,
    );
  }
  #[test]
  fn string_replace_3() {
    assert_case(
      r#"StringMatchQ["\r \n", Whitespace]; StringSplit["a  \n b \r\n c d", Whitespace]; StringReplace[" this has leading and trailing whitespace \n ", (StartOfString ~~ Whitespace) | (Whitespace ~~ EndOfString) -> ""] <> " removed" // FullForm"#,
      r#"FullForm["this has leading and trailing whitespace removed"]"#,
    );
  }
  #[test]
  fn set_3() {
    // Wolframscript-matched expectation. mathics expected the InputForm
    // `ByteArray["ARkD"]` (base64 payload), but wolframscript -code shows
    // the compact `ByteArray[<n>]` length notation, which is what Woxi
    // also produces. Use ToString or InputForm to recover the base64
    // serialization.
    assert_case(r#"A=ByteArray[{1, 25, 3}]"#, r#"ByteArray[<3>]"#);
  }
  #[test]
  fn a() {
    assert_case(r#"A=ByteArray[{1, 25, 3}]; A[[2]]"#, r#"25"#);
  }
  #[test]
  fn normal() {
    assert_case(
      r#"A=ByteArray[{1, 25, 3}]; A[[2]]; Normal[A]"#,
      r#"{1, 25, 3}"#,
    );
  }
  #[test]
  fn to_string_4() {
    assert_case(
      r#"A=ByteArray[{1, 25, 3}]; A[[2]]; Normal[A]; ToString[A]"#,
      r#""ByteArray[<3>]""#,
    );
  }
  #[test]
  fn byte_array() {
    assert_case(
      r#"A=ByteArray[{1, 25, 3}]; A[[2]]; Normal[A]; ToString[A]; ByteArray["ARkD"]"#,
      r#"ByteArray[<3>]"#,
    );
  }
  #[test]
  fn string_match_q_10() {
    assert_case(r#"StringMatchQ["1", DigitCharacter]"#, r#"True"#);
  }
  #[test]
  fn string_match_q_11() {
    assert_case(
      r#"StringMatchQ["1", DigitCharacter]; StringMatchQ["a", DigitCharacter]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_match_q_12() {
    assert_case(
      r#"StringMatchQ["1", DigitCharacter]; StringMatchQ["a", DigitCharacter]; StringMatchQ["12", DigitCharacter]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_match_q_13() {
    assert_case(
      r#"StringMatchQ["1", DigitCharacter]; StringMatchQ["a", DigitCharacter]; StringMatchQ["12", DigitCharacter]; StringMatchQ["123245", DigitCharacter..]"#,
      r#"True"#,
    );
  }
  #[test]
  fn string_replace_4() {
    assert_case(
      r#"StringReplace["aba\nbba\na\nab", "a" ~~ EndOfLine -> "c"]"#,
      r#""abc
bbc
c
ab""#,
    );
  }
  #[test]
  fn string_split_2() {
    assert_case(
      r#"StringReplace["aba\nbba\na\nab", "a" ~~ EndOfLine -> "c"]; StringSplit["abc\ndef\nhij", EndOfLine]"#,
      r#"{"abc", "
def", "
hij"}"#,
    );
  }
  #[test]
  fn string_match_q_14() {
    assert_case(
      r#"StringMatchQ[#, __ ~~ "e" ~~ EndOfString] &/@ {"apple", "banana", "artichoke"}"#,
      r#"{True, False, True}"#,
    );
  }
  #[test]
  fn string_replace_5() {
    assert_case(
      r#"StringMatchQ[#, __ ~~ "e" ~~ EndOfString] &/@ {"apple", "banana", "artichoke"}; StringReplace["aab\nabb", "b" ~~ EndOfString -> "c"]"#,
      r#""aab
abc""#,
    );
  }
  #[test]
  fn string_match_q_15() {
    assert_case(
      r#"StringMatchQ[#, LetterCharacter] & /@ {"a", "1", "A", " ", "."}"#,
      r#"{True, False, True, False, False}"#,
    );
  }
  #[test]
  fn string_match_q_16() {
    assert_case(
      r#"StringMatchQ[#, LetterCharacter] & /@ {"a", "1", "A", " ", "."}; StringMatchQ["\[Lambda]", LetterCharacter]"#,
      r#"True"#,
    );
  }
  #[test]
  fn string_replace_6() {
    assert_case(
      r#"StringReplace["aba\nbba\na\nab", StartOfLine ~~ "a" -> "c"]"#,
      r#""cba
bba
c
cb""#,
    );
  }
  #[test]
  fn string_split_3() {
    assert_case(
      r#"StringReplace["aba\nbba\na\nab", StartOfLine ~~ "a" -> "c"]; StringSplit["abc\ndef\nhij", StartOfLine]"#,
      r#"{"abc
", "def
", "hij"}"#,
    );
  }
  #[test]
  fn string_match_q_17() {
    assert_case(
      r#"StringMatchQ[#, StartOfString ~~ "a" ~~ __] &/@ {"apple", "banana", "artichoke"}"#,
      r#"{True, False, True}"#,
    );
  }
  #[test]
  fn string_replace_7() {
    assert_case(
      r#"StringMatchQ[#, StartOfString ~~ "a" ~~ __] &/@ {"apple", "banana", "artichoke"}; StringReplace["aba\nabb", StartOfString ~~ "a" -> "c"]"#,
      r#""cba
abb""#,
    );
  }
  #[test]
  fn string_cases_5() {
    assert_case(r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]"#, r#"{"axb"}"#);
  }
  #[test]
  fn string_cases_6() {
    assert_case(
      r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]; StringCases["axbaxxb", "a" ~~ x__ ~~ "b"]"#,
      r#"{"axbaxxb"}"#,
    );
  }
  #[test]
  fn string_cases_7() {
    assert_case(
      r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]; StringCases["axbaxxb", "a" ~~ x__ ~~ "b"]; StringCases["axbaxxb", Shortest["a" ~~ x__ ~~ "b"]]"#,
      r#"{"axb", "axxb"}"#,
    );
  }
  #[test]
  fn string_cases_8() {
    assert_case(
      r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]; StringCases["axbaxxb", "a" ~~ x__ ~~ "b"]; StringCases["axbaxxb", Shortest["a" ~~ x__ ~~ "b"]]; StringCases["-abc- def -uvw- xyz", Shortest["-" ~~ x__ ~~ "-"] -> x]"#,
      r#"{"abc", "uvw"}"#,
    );
  }
  #[test]
  fn string_cases_9() {
    assert_case(
      r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]; StringCases["axbaxxb", "a" ~~ x__ ~~ "b"]; StringCases["axbaxxb", Shortest["a" ~~ x__ ~~ "b"]]; StringCases["-abc- def -uvw- xyz", Shortest["-" ~~ x__ ~~ "-"] -> x]; StringCases["-öhi- -abc- -.-", "-" ~~ x : WordCharacter .. ~~ "-" -> x]"#,
      r#"{"abc"}"#,
    );
  }
  #[test]
  fn string_cases_10() {
    assert_case(
      r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]; StringCases["axbaxxb", "a" ~~ x__ ~~ "b"]; StringCases["axbaxxb", Shortest["a" ~~ x__ ~~ "b"]]; StringCases["-abc- def -uvw- xyz", Shortest["-" ~~ x__ ~~ "-"] -> x]; StringCases["-öhi- -abc- -.-", "-" ~~ x : WordCharacter .. ~~ "-" -> x]; StringCases["abc-abc xyz-uvw", Shortest[x : WordCharacter .. ~~ "-" ~~ x_] -> x]"#,
      r#"{"abc"}"#,
    );
  }
  #[test]
  fn string_cases_11() {
    assert_case(
      r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]; StringCases["axbaxxb", "a" ~~ x__ ~~ "b"]; StringCases["axbaxxb", Shortest["a" ~~ x__ ~~ "b"]]; StringCases["-abc- def -uvw- xyz", Shortest["-" ~~ x__ ~~ "-"] -> x]; StringCases["-öhi- -abc- -.-", "-" ~~ x : WordCharacter .. ~~ "-" -> x]; StringCases["abc-abc xyz-uvw", Shortest[x : WordCharacter .. ~~ "-" ~~ x_] -> x]; StringCases["abba", {"a" -> 10, "b" -> 20}, 2]"#,
      r#"{10, 20}"#,
    );
  }
  #[test]
  fn string_cases_12() {
    // The scraped expectation \`{"a", "\[CapitalATilde]", "1", "2",
    // "3"}\` — the \`\\[CapitalATilde]\` (\`Ã\`) — is more
    // wolframscript UTF-8-as-Latin-1 mojibake (cf. cases 2174/2175):
    // the bytes for \`ä\` (\`0xC3 0xB1\` interpreted as \`Ã ¤\`)
    // produce a stray \`Ã\` that Wolfram's ASCII-only \`WordCharacter\`
    // matches. Wolfram itself documents \`WordCharacter\` as ASCII-
    // only (\`StringMatchQ["ä", WordCharacter]\` → False). With proper
    // UTF-8, \`StringCases["a#ä_123", WordCharacter]\` gives
    // \`{a, 1, 2, 3}\`.
    assert_case(
      r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]; StringCases["axbaxxb", "a" ~~ x__ ~~ "b"]; StringCases["axbaxxb", Shortest["a" ~~ x__ ~~ "b"]]; StringCases["-abc- def -uvw- xyz", Shortest["-" ~~ x__ ~~ "-"] -> x]; StringCases["-öhi- -abc- -.-", "-" ~~ x : WordCharacter .. ~~ "-" -> x]; StringCases["abc-abc xyz-uvw", Shortest[x : WordCharacter .. ~~ "-" ~~ x_] -> x]; StringCases["abba", {"a" -> 10, "b" -> 20}, 2]; StringCases["a#ä_123", WordCharacter]"#,
      r#"{"a", "1", "2", "3"}"#,
    );
  }
  #[test]
  fn string_cases_13() {
    // Same wolframscript-mojibake situation as case 2779. The scraped
    // \`{"a", "\\[CapitalATilde]"}\` is the Latin-1 leftover of \`ä\`
    // — Wolfram's \`LetterCharacter\` does match Unicode letters
    // (unlike \`WordCharacter\`), but the input got mis-decoded as
    // Latin-1 first. Mathics's docstring (and Woxi) give the actually
    // correct \`{"a", "ä"}\`.
    assert_case(
      r#"StringCases["axbaxxb", "a" ~~ x_ ~~ "b"]; StringCases["axbaxxb", "a" ~~ x__ ~~ "b"]; StringCases["axbaxxb", Shortest["a" ~~ x__ ~~ "b"]]; StringCases["-abc- def -uvw- xyz", Shortest["-" ~~ x__ ~~ "-"] -> x]; StringCases["-öhi- -abc- -.-", "-" ~~ x : WordCharacter .. ~~ "-" -> x]; StringCases["abc-abc xyz-uvw", Shortest[x : WordCharacter .. ~~ "-" ~~ x_] -> x]; StringCases["abba", {"a" -> 10, "b" -> 20}, 2]; StringCases["a#ä_123", WordCharacter]; StringCases["a#ä_123", LetterCharacter]"#,
      r#"{"a", "ä"}"#,
    );
  }
  #[test]
  fn string_match_q_18() {
    assert_case(r#"StringMatchQ["\n", WhitespaceCharacter]"#, r#"True"#);
  }
  #[test]
  fn string_split_4() {
    assert_case(
      r#"StringMatchQ["\n", WhitespaceCharacter]; StringSplit["a\nb\r\nc\rd", WhitespaceCharacter]"#,
      r#"{"a", "b", "", "c", "d"}"#,
    );
  }
  #[test]
  fn string_match_q_19() {
    assert_case(
      r#"StringMatchQ["\n", WhitespaceCharacter]; StringSplit["a\nb\r\nc\rd", WhitespaceCharacter]; StringMatchQ[" \n", WhitespaceCharacter]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_match_q_20() {
    assert_case(
      r#"StringMatchQ["\n", WhitespaceCharacter]; StringSplit["a\nb\r\nc\rd", WhitespaceCharacter]; StringMatchQ[" \n", WhitespaceCharacter]; StringMatchQ[" \n", Whitespace]"#,
      r#"True"#,
    );
  }
  #[test]
  fn string_replace_8() {
    assert_case(
      r#"StringReplace["apple banana orange artichoke", "e" ~~ WordBoundary -> "E"]"#,
      r#""applE banana orangE artichokE""#,
    );
  }
  #[test]
  fn string_match_q_21() {
    assert_case(
      r#"StringMatchQ[#, WordCharacter] &/@ {"1", "a", "A", ",", " "}"#,
      r#"{True, True, True, False, False}"#,
    );
  }
  #[test]
  fn string_match_q_22() {
    assert_case(
      r#"StringMatchQ[#, WordCharacter] &/@ {"1", "a", "A", ",", " "}; StringMatchQ["abc123DEF", WordCharacter..]"#,
      r#"True"#,
    );
  }
  #[test]
  fn string_match_q_23() {
    assert_case(
      r#"StringMatchQ[#, WordCharacter] &/@ {"1", "a", "A", ",", " "}; StringMatchQ["abc123DEF", WordCharacter..]; StringMatchQ["$b;123", WordCharacter..]"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_insert_1() {
    assert_case(r#"StringInsert["noting", "h", 4]"#, r#""nothing""#);
  }
  #[test]
  fn string_insert_2() {
    assert_case(
      r#"StringInsert["noting", "h", 4]; StringInsert["note", "d", -1]"#,
      r#""noted""#,
    );
  }
  #[test]
  fn string_insert_3() {
    assert_case(
      r#"StringInsert["noting", "h", 4]; StringInsert["note", "d", -1]; StringInsert["here", "t", -5]"#,
      r#""there""#,
    );
  }
  #[test]
  fn string_insert_4() {
    assert_case(
      r#"StringInsert["noting", "h", 4]; StringInsert["note", "d", -1]; StringInsert["here", "t", -5]; StringInsert["adac", "he", {1, 5}]"#,
      r#""headache""#,
    );
  }
  #[test]
  fn string_insert_5() {
    assert_case(
      r#"StringInsert["noting", "h", 4]; StringInsert["note", "d", -1]; StringInsert["here", "t", -5]; StringInsert["adac", "he", {1, 5}]; StringInsert[{"something", "sometimes"}, " ", 5]"#,
      r#"{"some thing", "some times"}"#,
    );
  }
  #[test]
  fn string_insert_6() {
    assert_case(
      r#"StringInsert["noting", "h", 4]; StringInsert["note", "d", -1]; StringInsert["here", "t", -5]; StringInsert["adac", "he", {1, 5}]; StringInsert[{"something", "sometimes"}, " ", 5]; StringInsert["1234567890123456", ".", Range[-16, -4, 3]]"#,
      r#""1.234.567.890.123.456""#,
    );
  }
  // The inserted text (position 2) must be a single string; a list or other
  // expression there stays unevaluated (WL emits StringInsert::string).
  #[test]
  fn string_insert_nonstring_snew_list() {
    assert_case(
      r#"StringInsert["abc", {"X", "Y"}, {1, 3}]"#,
      r#"StringInsert[abc, {X, Y}, {1, 3}]"#,
    );
  }
  #[test]
  fn string_insert_nonstring_snew_integer() {
    assert_case(r#"StringInsert["abc", 5, 2]"#, r#"StringInsert[abc, 5, 2]"#);
  }
  #[test]
  fn string_insert_nonstring_snew_with_list_first_arg() {
    // The check fires before the list-of-strings first-argument form, so the
    // message reports the whole original call (single result, not per element).
    assert_case(
      r#"StringInsert[{"ab", "cd"}, {"X", "Y"}, 2]"#,
      r#"StringInsert[{ab, cd}, {X, Y}, 2]"#,
    );
  }
  #[test]
  fn string_join_1() {
    assert_case(r#"StringJoin["a", "b", "c"]"#, r#""abc""#);
  }
  #[test]
  fn string_literal_1() {
    assert_case(
      r#"StringJoin["a", "b", "c"]; "a" <> "b" <> "c" // InputForm"#,
      r#"InputForm["abc"]"#,
    );
  }
  #[test]
  fn string_join_2() {
    assert_case(
      r#"StringJoin["a", "b", "c"]; "a" <> "b" <> "c" // InputForm; StringJoin[{"a", "b"}] // InputForm"#,
      r#"InputForm["ab"]"#,
    );
  }
  #[test]
  fn string_length_1() {
    assert_case(r#"StringLength["abc"]"#, r#"3"#);
  }
  #[test]
  fn string_length_2() {
    assert_case(
      r#"StringLength["abc"]; StringLength[{"a", "bc"}]"#,
      r#"{1, 2}"#,
    );
  }
  #[test]
  fn string_position_1() {
    assert_case(
      r#"StringPosition["123ABCxyABCzzzABCABC", "ABC"]"#,
      r#"{{4, 6}, {9, 11}, {15, 17}, {18, 20}}"#,
    );
  }
  #[test]
  fn string_position_2() {
    assert_case(
      r#"StringPosition["123ABCxyABCzzzABCABC", "ABC"]; StringPosition["123ABCxyABCzzzABCABC", "ABC", 2]"#,
      r#"{{4, 6}, {9, 11}}"#,
    );
  }
  #[test]
  fn string_replace_9() {
    assert_case(
      r#"StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A"]"#,
      r#""AAAyyxxAyA""#,
    );
  }
  #[test]
  fn string_replace_10() {
    assert_case(
      r#"StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A"]; StringReplace["xyzwxyzwxxyzxyzw", {"xyz" -> "A", "w" -> "BCD"}]"#,
      r#""ABCDABCDxAABCD""#,
    );
  }
  #[test]
  fn string_replace_11() {
    assert_case(
      r#"StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A"]; StringReplace["xyzwxyzwxxyzxyzw", {"xyz" -> "A", "w" -> "BCD"}]; StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A", 2]"#,
      r#""AAxyyyxxxyyxy""#,
    );
  }
  #[test]
  fn string_replace_12() {
    assert_case(
      r#"StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A"]; StringReplace["xyzwxyzwxxyzxyzw", {"xyz" -> "A", "w" -> "BCD"}]; StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A", 2]; StringReplace["abba", {"a" -> "A", "b" -> "B"}, 2]"#,
      r#""ABba""#,
    );
  }
  #[test]
  fn string_replace_13() {
    assert_case(
      r#"StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A"]; StringReplace["xyzwxyzwxxyzxyzw", {"xyz" -> "A", "w" -> "BCD"}]; StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A", 2]; StringReplace["abba", {"a" -> "A", "b" -> "B"}, 2]; StringReplace[{"xyxyxxy", "yxyxyxxxyyxy"}, "xy" -> "A"]"#,
      r#"{"AAxA", "yAAxxAyA"}"#,
    );
  }
  #[test]
  fn string_replace_14() {
    assert_case(
      r#"StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A"]; StringReplace["xyzwxyzwxxyzxyzw", {"xyz" -> "A", "w" -> "BCD"}]; StringReplace["xyxyxyyyxxxyyxy", "xy" -> "A", 2]; StringReplace["abba", {"a" -> "A", "b" -> "B"}, 2]; StringReplace[{"xyxyxxy", "yxyxyxxxyyxy"}, "xy" -> "A"]; StringReplace["y" -> "ies"]["city"]"#,
      r#""cities""#,
    );
  }
  #[test]
  fn string_reverse() {
    assert_case(r#"StringReverse["live"]"#, r#""evil""#);
  }
  #[test]
  fn string_riffle_1() {
    assert_case(
      r#"StringRiffle[{"a", "b", "c", "d", "e"}]"#,
      r#""a b c d e""#,
    );
  }
  #[test]
  fn string_riffle_2() {
    assert_case(
      r#"StringRiffle[{"a", "b", "c", "d", "e"}]; StringRiffle[{"a", "b", "c", "d", "e"}, ", "]"#,
      r#""a, b, c, d, e""#,
    );
  }
  #[test]
  fn string_riffle_3() {
    assert_case(
      r#"StringRiffle[{"a", "b", "c", "d", "e"}]; StringRiffle[{"a", "b", "c", "d", "e"}, ", "]; StringRiffle[{"a", "b", "c", "d", "e"}, {"(", " ", ")"}]"#,
      r#""(a b c d e)""#,
    );
  }
  #[test]
  fn string_split_5() {
    assert_case(r#"StringSplit["abc,123", ","]"#, r#"{"abc", "123"}"#);
  }
  #[test]
  fn string_split_6() {
    assert_case(
      r#"StringSplit["abc,123", ","]; StringSplit["  abc    123  "]"#,
      r#"{"abc", "123"}"#,
    );
  }
  #[test]
  fn string_split_7() {
    assert_case(
      r#"StringSplit["abc,123", ","]; StringSplit["  abc    123  "]; StringSplit["  abc    123  ", WhitespaceCharacter]"#,
      r#"{"abc", "", "", "", "123"}"#,
    );
  }
  #[test]
  fn string_split_8() {
    assert_case(
      r#"StringSplit["abc,123", ","]; StringSplit["  abc    123  "]; StringSplit["  abc    123  ", WhitespaceCharacter]; StringSplit["abc,123.456", {",", "."}]"#,
      r#"{"abc", "123", "456"}"#,
    );
  }
  #[test]
  fn string_split_9() {
    assert_case(
      r#"StringSplit["abc,123", ","]; StringSplit["  abc    123  "]; StringSplit["  abc    123  ", WhitespaceCharacter]; StringSplit["abc,123.456", {",", "."}]; StringSplit["a  b    c", RegularExpression[" +"]]"#,
      r#"{"a", "b", "c"}"#,
    );
  }
  #[test]
  fn string_split_list_with_pattern() {
    // A list of delimiters that contains a pattern (not just literals).
    assert_case(r#"StringSplit["a1b2", {DigitCharacter}]"#, r#"{"a", "b"}"#);
    assert_case(
      r#"StringSplit["a1b2c3", {LetterCharacter}]"#,
      r#"{"1", "2", "3"}"#,
    );
    // Mixed literal + pattern delimiters keep interior empties.
    assert_case(
      r#"StringSplit["x1y22z", {DigitCharacter, "y"}]"#,
      r#"{"x", "", "", "", "z"}"#,
    );
  }
  #[test]
  fn string_split_10() {
    assert_case(
      r#"StringSplit["abc,123", ","]; StringSplit["  abc    123  "]; StringSplit["  abc    123  ", WhitespaceCharacter]; StringSplit["abc,123.456", {",", "."}]; StringSplit["a  b    c", RegularExpression[" +"]]; StringSplit[{"a  b", "c  d"}, RegularExpression[" +"]]"#,
      r#"{{"a", "b"}, {"c", "d"}}"#,
    );
  }
  #[test]
  fn string_split_11() {
    assert_case(
      r#"StringSplit["abc,123", ","]; StringSplit["  abc    123  "]; StringSplit["  abc    123  ", WhitespaceCharacter]; StringSplit["abc,123.456", {",", "."}]; StringSplit["a  b    c", RegularExpression[" +"]]; StringSplit[{"a  b", "c  d"}, RegularExpression[" +"]]; StringSplit["x", "x"]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn string_split_12() {
    assert_case(
      r#"StringSplit["abc,123", ","]; StringSplit["  abc    123  "]; StringSplit["  abc    123  ", WhitespaceCharacter]; StringSplit["abc,123.456", {",", "."}]; StringSplit["a  b    c", RegularExpression[" +"]]; StringSplit[{"a  b", "c  d"}, RegularExpression[" +"]]; StringSplit["x", "x"]; StringSplit["12312123", "12"..]"#,
      r#"{"3", "3"}"#,
    );
  }
  #[test]
  fn string_take_2() {
    assert_case(r#"StringTake["abcde", 2]"#, r#""ab""#);
  }
  #[test]
  fn string_take_3() {
    assert_case(r#"StringTake["abcde", 2]; StringTake["abcde", 0]"#, r#""""#);
  }
  #[test]
  fn string_take_4() {
    assert_case(
      r#"StringTake["abcde", 2]; StringTake["abcde", 0]; StringTake["abcde", -2]"#,
      r#""de""#,
    );
  }
  #[test]
  fn string_take_5() {
    assert_case(
      r#"StringTake["abcde", 2]; StringTake["abcde", 0]; StringTake["abcde", -2]; StringTake["abcde", {2}]"#,
      r#""b""#,
    );
  }
  #[test]
  fn string_take_6() {
    assert_case(
      r#"StringTake["abcde", 2]; StringTake["abcde", 0]; StringTake["abcde", -2]; StringTake["abcde", {2}]; StringTake["abcd", {2,3}]"#,
      r#""bc""#,
    );
  }
  #[test]
  fn string_take_7() {
    assert_case(
      r#"StringTake["abcde", 2]; StringTake["abcde", 0]; StringTake["abcde", -2]; StringTake["abcde", {2}]; StringTake["abcd", {2,3}]; StringTake["abcdefgh", {1, 5, 2}]"#,
      r#""ace""#,
    );
  }
  #[test]
  fn string_take_8() {
    assert_case(
      r#"StringTake["abcde", 2]; StringTake["abcde", 0]; StringTake["abcde", -2]; StringTake["abcde", {2}]; StringTake["abcd", {2,3}]; StringTake["abcdefgh", {1, 5, 2}]; StringTake[{"abcdef", "stuv", "xyzw"}, -2]"#,
      r#"{"ef", "uv", "zw"}"#,
    );
  }
  #[test]
  fn string_take_9() {
    assert_case(
      r#"StringTake["abcde", 2]; StringTake["abcde", 0]; StringTake["abcde", -2]; StringTake["abcde", {2}]; StringTake["abcd", {2,3}]; StringTake["abcdefgh", {1, 5, 2}]; StringTake[{"abcdef", "stuv", "xyzw"}, -2]; StringTake["abcdef", All]"#,
      r#""abcdef""#,
    );
  }
  #[test]
  fn string_join_3() {
    assert_case(
      r#"StringJoin["a", StringTrim["  \tb\n "], "c"]"#,
      r#""abc""#,
    );
  }
  #[test]
  fn string_trim() {
    assert_case(
      r#"StringJoin["a", StringTrim["  \tb\n "], "c"]; StringTrim["ababaxababyaabab", RegularExpression["(ab)+"]]"#,
      r#""axababya""#,
    );
  }
  #[test]
  fn string_split_13() {
    assert_case(
      r#"StringSplit["1.23, 4.56  7.89", RegularExpression["(\\s|,)+"]]"#,
      r#"{"1.23", "4.56", "7.89"}"#,
    );
  }
  #[test]
  fn regular_expression() {
    assert_case(
      r#"StringSplit["1.23, 4.56  7.89", RegularExpression["(\\s|,)+"]]; RegularExpression["[abc]"]"#,
      r#"RegularExpression["[abc]"]"#,
    );
  }
  #[test]
  fn characters_1() {
    assert_case(r#"Characters["abc"]"#, r#"{"a", "b", "c"}"#);
  }
  #[test]
  fn character_range_1() {
    assert_case(
      r#"CharacterRange["a", "e"]"#,
      r#"{"a", "b", "c", "d", "e"}"#,
    );
  }
  #[test]
  fn character_range_2() {
    assert_case(
      r#"CharacterRange["a", "e"]; CharacterRange["b", "a"]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn lower_case_q_1() {
    assert_case(r#"LowerCaseQ["abc"]"#, r#"True"#);
  }
  #[test]
  fn lower_case_q_2() {
    assert_case(r#"LowerCaseQ["abc"]; LowerCaseQ[""]"#, r#"True"#);
  }
  #[test]
  fn to_lower_case() {
    assert_case(r#"ToLowerCase["New York"]"#, r#""new york""#);
  }
  #[test]
  fn to_upper_case() {
    assert_case(r#"ToUpperCase["New York"]"#, r#""NEW YORK""#);
  }
  #[test]
  fn upper_case_q_1() {
    assert_case(r#"UpperCaseQ["ABC"]"#, r#"True"#);
  }
  #[test]
  fn upper_case_q_2() {
    assert_case(r#"UpperCaseQ["ABC"]; UpperCaseQ[""]"#, r#"True"#);
  }
  #[test]
  fn to_character_code_1() {
    assert_case(r#"ToCharacterCode["abc"]"#, r#"{97, 98, 99}"#);
  }
  #[test]
  fn from_character_code_1() {
    assert_case(r#"FromCharacterCode[100]"#, r#""d""#);
  }
  #[test]
  fn from_character_code_2() {
    // Wolframscript-matched expectation. The mathics original used the
    // named-character notation `"\[ADoubleDot]"`, but wolframscript and
    // Woxi both emit the actual UTF-8 codepoint `ä` for character 228.
    assert_case(
      r#"FromCharacterCode[100]; FromCharacterCode[228, "ISO8859-1"]"#,
      "\u{e4}",
    );
  }
  #[test]
  fn from_character_code_3() {
    assert_case(
      r#"FromCharacterCode[100]; FromCharacterCode[228, "ISO8859-1"]; FromCharacterCode[{100, 101, 102}]"#,
      r#""def""#,
    );
  }
  #[test]
  fn from_character_code_invalid_arg_returns_unevaluated() {
    // Non-integer / non-list args (e.g. an unbound symbol via `%` in a
    // script) must not raise a hard error. Wolframscript emits the
    // FromCharacterCode::intnm message and returns the call unevaluated;
    // Woxi must do the same so chained sequences keep flowing.
    assert_case(r#"FromCharacterCode[xyz]"#, r#"FromCharacterCode[xyz]"#);
  }
  #[test]
  fn string_position_unbound_string_returns_unevaluated() {
    // StringPosition[<unbound>, "uranium"] previously coerced the symbol
    // to its name and searched there, returning `{}`. Wolframscript emits
    // StringPosition::strse and returns the call unevaluated; Woxi must
    // do the same.
    assert_case(
      r#"StringPosition[data, "uranium"]"#,
      r#"StringPosition[data, "uranium"]"#,
    );
  }
  #[test]
  fn unequal() {
    assert_case(
      r#"System`Convert`B64Dump`B64Decode["R!="]"#,
      r#"System`Convert`B64Dump`B64Decode["R!="]"#,
    );
  }
  #[test]
  fn expr_1() {
    assert_case(
      r#"System`Convert`B64Dump`B64Encode["Hello world"]"#,
      r#"System`Convert`B64Dump`B64Encode["Hello world"]"#,
    );
  }
  #[test]
  fn expr_2() {
    assert_case(
      r#"System`Convert`B64Dump`B64Encode["Hello world"]; System`Convert`B64Dump`B64Decode[%]"#,
      r#"System`Convert`B64Dump`B64Decode[Out[0]]"#,
    );
  }
  #[test]
  fn make_boxes_1() {
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]"#,
      r#"RowBox[{"G","[","F[3.002]","]"}]"#,
    );
  }
  #[test]
  fn make_boxes_2() {
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]"#,
      // wolframscript: single layer of baked-in quotes,
      // second arg wraps the original expression in OutputForm.
      r#"InterpretationBox[PaneBox["G[F[3.002]]", BaselinePosition -> Baseline], OutputForm[G[F[3.002]]], Editable -> False]"#,
    );
  }
  #[test]
  fn format_1() {
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_]] := {"Formatted f", {x}, "Standard"}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn make_boxes_3() {
    // mathics's expected output uses InputForm-style box rendering
    // (every box element wrapped in quotes, inner quotes escaped);
    // wolframscript's REPL uses unquoted box-element strings, with
    // user-supplied strings retaining their original quotes. Match
    // wolframscript here — `Format[F[x_]] := {…}` causes the inner
    // F[3.002] to render via the formatted list of strings.
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_]] := {"Formatted f", {x}, "Standard"}; MakeBoxes[G[F[3.002]], StandardForm]"#,
      r#"RowBox[{G, [, RowBox[{{, RowBox[{"Formatted f", ,, RowBox[{{, 3.002`, }}], ,, "Standard"}], }}], ]}]"#,
    );
  }
  #[test]
  fn make_boxes_4() {
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_]] := {"Formatted f", {x}, "Standard"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]"#,
      // wolframscript form: single layer of baked-in quotes;
      // second arg wraps the *original* expression in OutputForm.
      // The Format rule still applies inside the rendered string.
      r#"InterpretationBox[PaneBox["G[{Formatted f, {3.002}, Standard}]", BaselinePosition -> Baseline], OutputForm[G[F[3.002]]], Editable -> False]"#,
    );
  }
  #[test]
  fn format_2() {
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_]] := {"Formatted f", {x}, "Standard"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_], StandardForm] :=  {"Formatted f", {x}, "Standard"};Format[F[x_], OutputForm] :=  {"Formatted f", {x}, "Output"}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn make_boxes_5() {
    // Same family as case 3674. After also defining the form-specific
    // `Format[F[x_], StandardForm]` and `Format[F[x_], OutputForm]`
    // rules, the StandardForm box rendering of `G[F[3.002]]` should
    // still apply the StandardForm-tagged rule (or fall through to the
    // 1-arg rule, which has the same body). Match wolframscript's
    // unquoted box-element style.
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_]] := {"Formatted f", {x}, "Standard"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_], StandardForm] :=  {"Formatted f", {x}, "Standard"};Format[F[x_], OutputForm] :=  {"Formatted f", {x}, "Output"}; MakeBoxes[G[F[3.002]], StandardForm]"#,
      r#"RowBox[{G, [, RowBox[{{, RowBox[{"Formatted f", ,, RowBox[{{, 3.002`, }}], ,, "Standard"}], }}], ]}]"#,
    );
  }
  #[test]
  fn make_boxes_6() {
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_]] := {"Formatted f", {x}, "Standard"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_], StandardForm] :=  {"Formatted f", {x}, "Standard"};Format[F[x_], OutputForm] :=  {"Formatted f", {x}, "Output"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]"#,
      // wolframscript form: single layer of baked-in quotes;
      // second arg wraps the *original* expression in OutputForm.
      r#"InterpretationBox[PaneBox["G[{Formatted f, {3.002}, Output}]", BaselinePosition -> Baseline], OutputForm[G[F[3.002]]], Editable -> False]"#,
    );
  }
  #[test]
  fn make_boxes_7() {
    // Same family as case 3674. After ClearAll[F] removes the Format
    // rules but the user `MakeBoxes[F[x_], fmt_] := …` rule is still
    // in effect, so `G[F[2.]]` boxes via the user's MakeBoxes for F
    // (returning the literal string `"F[2.]"`). Match wolframscript's
    // unquoted box-element style.
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_]] := {"Formatted f", {x}, "Standard"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_], StandardForm] :=  {"Formatted f", {x}, "Standard"};Format[F[x_], OutputForm] :=  {"Formatted f", {x}, "Output"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; ClearAll[F]; MakeBoxes[G[F[2.]], StandardForm]"#,
      r#"RowBox[{G, [, F[2.], ]}]"#,
    );
  }
  #[test]
  fn make_boxes_8() {
    assert_case(
      r#"MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_] := "F[" <> ToString[x] <> "]";MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_]] := {"Formatted f", {x}, "Standard"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; Format[F[x_], StandardForm] :=  {"Formatted f", {x}, "Standard"};Format[F[x_], OutputForm] :=  {"Formatted f", {x}, "Output"}; MakeBoxes[G[F[3.002]], StandardForm]; MakeBoxes[OutputForm[G[F[3.002]]], StandardForm]; ClearAll[F]; MakeBoxes[G[F[2.]], StandardForm]; MakeBoxes[F[x_], fmt_]=.; MakeBoxes[G[F[2.]], StandardForm]"#,
      r#"RowBox[{"G", "[", RowBox[{"F", "[", "2.`", "]"}], "]"}]"#,
    );
  }
  #[test]
  fn to_string_5() {
    // mathics's expected output is the InputForm rendering of the
    // resulting String (literal `\!\(\*RowBox[…]\)` escape syntax).
    // wolframscript's REPL prints the same String in OutputForm, where
    // the box-escape characters render as `DisplayForm[RowBox[…]]`.
    // Match wolframscript: Format[G[x___], StandardForm] applies first
    // (yielding `{"Standard", GG[F[1., "l"], .2]}`), then the inner
    // GG / F sub-expressions box via the user MakeBoxes rules; the
    // Format[F[x_, y_], StandardForm] rule rewrites `F[1., "l"]` into
    // `{F[1.], "Standard"}` before user MakeBoxes for F runs on F[1.].
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]"#,
      r#"DisplayForm[RowBox[{{, RowBox[{"Standard", ,, RowBox[{GG, <<, RowBox[{RowBox[{{, RowBox[{RowBox[{F, <~, RowBox[{1.`}], ~>}], ,, "Standard"}], }}], 0.2`}], >>}]}], }}]]"#,
    );
  }
  #[test]
  fn to_string_6() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]"#,
      r#""G[F[1.`, \"l\"], 0.2`]""#,
    );
  }
  #[test]
  fn to_string_7() {
    // Wolframscript-matched expectation. mathics quoted the returned
    // String as `"G[F[1., \"l\"], 0.2]"`, but `wolframscript -code`
    // prints `ToString[…, InputForm]`'s String result without surrounding
    // quotes. Woxi matches.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]"#,
      r#"G[F[1., "l"], 0.2]"#,
    );
  }
  #[test]
  fn to_string_8() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]"#,
      r#""G[F[1., l], 0.2]""#,
    );
  }
  #[test]
  fn format_3() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn format_4() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn format_5() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn format_6() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn format_7() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}"#,
      r#"Null"#,
    );
  }
  #[test]
  fn to_string_9() {
    // Same family as case 3686. After also defining InputForm /
    // OutputForm / FullForm Format rules, the StandardForm ToString
    // call still picks the StandardForm-tagged Format rule (and falls
    // back to the same `DisplayForm[RowBox[...]]` shape that case 3686
    // produces). Match wolframscript's display.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]"#,
      r#"DisplayForm[RowBox[{{, RowBox[{"Standard", ,, RowBox[{GG, <<, RowBox[{RowBox[{{, RowBox[{RowBox[{F, <~, RowBox[{1.`}], ~>}], ,, "Standard"}], }}], 0.2`}], >>}]}], }}]]"#,
    );
  }
  #[test]
  fn to_string_10() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]"#,
      r#""G[F[1.`, \"l\"], 0.2`]""#,
    );
  }
  #[test]
  fn to_string_11() {
    // mathics's expected wraps the resulting String's InputForm with
    // outer quotes; wolframscript's REPL prints the unquoted contents
    // (since OutputForm strips outer string quotes). Match
    // wolframscript: ToString[…, InputForm] applies the user
    // `Format[…, InputForm]` rules and produces the formatted list.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]"#,
      r#"{"In", GG[{F[1.], "In"}, 0.2]}"#,
    );
  }
  #[test]
  fn to_string_12() {
    // Same family as case 3697 — `ToString[…, OutputForm]` applies the
    // user `Format[…, OutputForm]` rules. Match wolframscript's REPL
    // display (no outer string quotes).
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]"#,
      r#"{Out, GG[{F[1.], Out}, 0.2]}"#,
    );
  }
  #[test]
  fn make_boxes_9() {
    // Same family as case 3686 — direct `MakeBoxes[…, StandardForm]`
    // (no ToString wrapper) yields the box AST that wolframscript
    // prints with unquoted box-element strings (only the user
    // `"Standard"` strings keep their quotes).
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]"#,
      r#"RowBox[{{, RowBox[{"Standard", ,, RowBox[{GG, <<, RowBox[{RowBox[{{, RowBox[{RowBox[{F, <~, RowBox[{1.`}], ~>}], ,, "Standard"}], }}], 0.2`}], >>}]}], }}]"#,
    );
  }
  #[test]
  fn make_boxes_10() {
    // mathics's expectation wraps the formatted text in extra
    // InputForm-quoting and replaces the InputForm's interpretation
    // arg with the formatted shape; wolframscript keeps the original
    // expression `G[F[1., l], 0.2]` as the interpretation arg and
    // shows the formatted text unquoted. Match wolframscript.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]"#,
      r#"InterpretationBox[StyleBox[{"In", GG[{F[1.], "In"}, 0.2]}, ShowStringCharacters -> True, NumberMarks -> True], InputForm[G[F[1., l], 0.2]], Editable -> True, AutoDelete -> True]"#,
    );
  }
  #[test]
  fn make_boxes_11() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]"#,
      // wolframscript form: single layer of baked-in quotes;
      // second arg is `OutputForm[<original-expr>]` rather than
      // the formatted body. (`G[F[1., "l"], .2]` has no
      // OutputForm Format rule applied yet at MakeBoxes time
      // because the rule operates inside expr_to_output_form_2d.)
      r#"InterpretationBox[PaneBox["{Out, GG[{F[1.], Out}, 0.2]}", BaselinePosition -> Baseline], OutputForm[G[F[1., l], 0.2]], Editable -> False]"#,
    );
  }
  #[test]
  fn to_string_13() {
    // mathics's expected was a typo-ridden raw string ("\	ext" is
    // backslash-tab-ext). wolframscript's actual TeX rendering of the
    // box AST keeps the user MakeBoxes delimiters and translates `~`
    // to TeX's `\sim ` macro. Match wolframscript.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]"#,
      r#"G<F<\sim 1.\text{l}\sim >0.2>"#,
    );
  }
  #[test]
  fn to_string_14() {
    // mathics's expected used a literal tab (`\	ext`) in place of
    // `\text` due to a Python escaping bug. wolframscript renders
    // `TeXForm[InputForm[expr]]` as `\text{<input-form-text>}` with
    // the formatted shape inside and `{`/`}` escaped using `$\{$` /
    // `$\}$`. Match wolframscript.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]"#,
      r#"\text{$\{$In, GG[$\{$F[1.], In$\}$, 0.2]$\}$}"#,
    );
  }
  #[test]
  fn clear_all() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]"#,
      r#"Null"#,
    );
  }
  #[test]
  fn to_string_15() {
    // After `ClearAll[F, G, GG]` the Format rules (stored under
    // FORMAT_VALUES[F/G/GG]) are removed, but the user MakeBoxes
    // rules (stored under FUNC_DEFS[MakeBoxes]) survive — wolframscript
    // does the same. The resulting StandardForm box AST therefore
    // still uses the user delimiters (`<`, `>`, `<~`, `~>`) but
    // skips the Format substitution.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]"#,
      r#"DisplayForm[RowBox[{G, <, RowBox[{RowBox[{F, <~, RowBox[{1.`, "l"}], ~>}], 0.2`}], >}]]"#,
    );
  }
  #[test]
  fn to_string_16() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]"#,
      r#""G[F[1.`, \"l\"], 0.2`]""#,
    );
  }
  #[test]
  fn to_string_17() {
    // Wolframscript-matched expectation. The mathics original quoted the
    // returned string as `"G[F[1., \"l\"], 0.2]"`, but `wolframscript -code`
    // prints `ToString[…, InputForm]`'s String result without surrounding
    // quotes. Woxi matches.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]"#,
      r#"G[F[1., "l"], 0.2]"#,
    );
  }
  #[test]
  fn to_string_18() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]"#,
      r#""G[F[1., l], 0.2]""#,
    );
  }
  #[test]
  fn make_boxes_12() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[F[x__], fmt_]=."#,
      r#"Null"#,
    );
  }
  #[test]
  fn make_boxes_13() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[F[x__], fmt_]=.; MakeBoxes[G[x___], fmt_]=."#,
      r#"Null"#,
    );
  }
  #[test]
  fn make_boxes_14() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[F[x__], fmt_]=.; MakeBoxes[G[x___], fmt_]=.; MakeBoxes[GG[x___], fmt_]=."#,
      r#"Null"#,
    );
  }
  #[test]
  fn to_string_19() {
    // After also unsetting the user MakeBoxes definitions
    // (`MakeBoxes[F[x__], fmt_]=.` etc.), the StandardForm rendering
    // falls back to the default `head[args]` boxing. Match
    // wolframscript's REPL display.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[F[x__], fmt_]=.; MakeBoxes[G[x___], fmt_]=.; MakeBoxes[GG[x___], fmt_]=.; ToString[G[F[1., "l"], .2], StandardForm]"#,
      r#"DisplayForm[RowBox[{G, [, RowBox[{RowBox[{F, [, RowBox[{1.`, ,, "l"}], ]}], ,, 0.2`}], ]}]]"#,
    );
  }
  #[test]
  fn to_string_20() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[F[x__], fmt_]=.; MakeBoxes[G[x___], fmt_]=.; MakeBoxes[GG[x___], fmt_]=.; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]"#,
      r#""G[F[1.`, \"l\"], 0.2`]""#,
    );
  }
  #[test]
  fn to_string_21() {
    // Wolframscript-matched expectation. mathics quoted the returned
    // String as `"G[F[1., \"l\"], 0.2]"`, but `wolframscript -code`
    // prints `ToString[…, InputForm]`'s String result without surrounding
    // quotes. Woxi matches.
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[F[x__], fmt_]=.; MakeBoxes[G[x___], fmt_]=.; MakeBoxes[GG[x___], fmt_]=.; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]"#,
      r#"G[F[1., "l"], 0.2]"#,
    );
  }
  #[test]
  fn to_string_22() {
    assert_case(
      r#"MakeBoxes[F[x__], fmt_] :=  RowBox[{"F", "<~", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], "~>"}]; MakeBoxes[G[x___], fmt_] := RowBox[{"G", "<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">"}]; MakeBoxes[GG[x___], fmt_] := RowBox[{"GG", "<<", RowBox[MakeBoxes[#1, fmt] & /@ List[x]], ">>"}]; Format[F[x_, y_], StandardForm] := {F[x], "Standard"}; Format[G[x___], StandardForm] :=  {"Standard", GG[x]}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; Format[F[x_, y_], InputForm] := {F[x], "In"}; Format[G[x___], InputForm] :=  {"In", GG[x]}; Format[F[x_, y_], OutputForm] := {F[x], "Out"}; Format[G[x___], OutputForm] :=  {"Out", GG[x]}; Format[F[x_, y_], FullForm] := {F[x], "full"}; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[G[F[1., "l"], .2], StandardForm]; MakeBoxes[InputForm[G[F[1., "l"], .2]], StandardForm]; MakeBoxes[OutputForm[G[F[1., "l"], .2]], StandardForm]; ToString[TeXForm[G[F[1., "l"], .2]]]; ToString[TeXForm[InputForm[G[F[1., "l"], .2]]]]; ClearAll[F, G, GG]; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]; MakeBoxes[F[x__], fmt_]=.; MakeBoxes[G[x___], fmt_]=.; MakeBoxes[GG[x___], fmt_]=.; ToString[G[F[1., "l"], .2], StandardForm]; ToString[FullForm[G[F[1., "l"], .2]]]; ToString[G[F[1., "l"], .2], InputForm]; ToString[G[F[1., "l"], .2], OutputForm]"#,
      r#""G[F[1., l], 0.2]""#,
    );
  }
  #[test]
  fn string_literal_2() {
    assert_case(r#""Hola""#, r#""Hola""#);
  }
  #[test]
  fn base_form_10() {
    assert_case(r#"BaseForm[0, 2]"#, r#"BaseForm[0, 2]"#);
  }
  #[test]
  fn base_form_11() {
    assert_case(r#"BaseForm[0, 2]; BaseForm[0.0, 2]"#, r#"BaseForm[0., 2]"#);
  }
  #[test]
  fn base_form_12() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]"#,
      r#"BaseForm[3.1415926535897932384626433832795028841971693993751058209749`30., 16]"#,
    );
  }
  #[test]
  fn input_form_1() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]"#,
      r#"InputForm[2*x^2 + 4*z!]"#,
    );
  }
  #[test]
  fn input_form_2() {
    // mathics quoted the embedded string and double-escaped the backslash;
    // wolframscript -code (OutputForm) emits the literal escape `\$` with
    // no surrounding quotes since string contents render verbatim inside
    // the InputForm wrapper. Woxi matches.
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]"#,
      r#"InputForm[\$]"#,
    );
  }
  #[test]
  fn number_form_1() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]"#,
      r#"NumberForm[Pi, 20]"#,
    );
  }
  #[test]
  fn number_form_2() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]"#,
      r#"NumberForm[2/3, 10]"#,
    );
  }
  #[test]
  fn number_form_3() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]"#,
      r#"NumberForm[3.141592653589793]"#,
    );
  }
  #[test]
  fn number_form_4() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]"#,
      r#"NumberForm[3.1415926535897932384626433832795028842`20.]"#,
    );
  }
  #[test]
  fn number_form_5() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]"#,
      r#"NumberForm[14310983091809]"#,
    );
  }
  // ToString[NumberForm[x, n]] formats x to n significant figures.
  #[test]
  fn to_string_number_form_significant_digits() {
    assert_case("ToString[NumberForm[3.14159, 3]]", "3.14");
    assert_case("ToString[NumberForm[3.14159, 5]]", "3.1416");
  }
  #[test]
  fn to_string_number_form_trailing_dot() {
    assert_case("ToString[NumberForm[3.0, 3]]", "3.");
    assert_case("ToString[NumberForm[100.0, 5]]", "100.");
    assert_case("ToString[NumberForm[0.0, 3]]", "0.");
  }
  #[test]
  fn to_string_number_form_rounds_integer_part() {
    assert_case("ToString[NumberForm[1234.5678, 2]]", "1200.");
  }
  #[test]
  fn to_string_number_form_small_and_negative() {
    assert_case("ToString[NumberForm[0.00123456, 3]]", "0.00123");
    assert_case("ToString[NumberForm[-3.14159, 3]]", "-3.14");
  }
  #[test]
  fn to_string_number_form_integer_unchanged() {
    // An integer argument is shown unchanged, ignoring the precision.
    assert_case("ToString[NumberForm[2, 3]]", "2");
    assert_case("ToString[NumberForm[1234567, 3]]", "1234567");
  }
  // NumberForm[x] with no precision spec renders like NumberForm[x, 6]
  // (the machine-precision default of 6 significant figures).
  #[test]
  fn to_string_number_form_default_precision() {
    assert_case("ToString[NumberForm[3.14159]]", "3.14159");
    assert_case("ToString[NumberForm[12345.678]]", "12345.7");
    assert_case("ToString[NumberForm[0.0001234]]", "0.0001234");
    assert_case("ToString[NumberForm[-3.5]]", "-3.5");
    assert_case("ToString[NumberForm[100.0]]", "100.");
    assert_case("ToString[NumberForm[42]]", "42");
  }
  // NumberForm[x, {n, f}] shows exactly f digits after the decimal point.
  #[test]
  fn to_string_number_form_fixed_decimals() {
    assert_case("ToString[NumberForm[3.14159, {5, 2}]]", "3.14");
    assert_case("ToString[NumberForm[1234.5678, {6, 2}]]", "1234.57");
  }
  #[test]
  fn to_string_number_form_fixed_pads_zeros() {
    assert_case("ToString[NumberForm[3.0, {5, 2}]]", "3.00");
    assert_case("ToString[NumberForm[3.1, {5, 3}]]", "3.100");
    assert_case("ToString[NumberForm[0.5, {4, 2}]]", "0.50");
  }
  #[test]
  fn to_string_number_form_fixed_zero_decimals() {
    assert_case("ToString[NumberForm[3.14159, {4, 0}]]", "3.");
  }
  #[test]
  fn to_string_number_form_fixed_negative() {
    assert_case("ToString[NumberForm[-2.5, {6, 3}]]", "-2.500");
  }
  #[test]
  fn set_4() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000"#,
      r#"0``28."#,
    );
  }
  #[test]
  fn number_form_6() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]"#,
      r#"NumberForm[{0., 0``28.}, 10]"#,
    );
  }
  #[test]
  fn number_form_7() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]"#,
      r#"NumberForm[{0., 0``28.}, {10, 4}]"#,
    );
  }
  #[test]
  fn unset() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=."#,
      r#"Null"#,
    );
  }
  #[test]
  fn number_form_8() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]"#,
      r#"NumberForm[1., 10]"#,
    );
  }
  #[test]
  fn number_form_9() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]"#,
      r#"NumberForm[1.`24., 10]"#,
    );
  }
  #[test]
  fn number_form_10() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]"#,
      r#"NumberForm[1., {10, 8}]"#,
    );
  }
  #[test]
  fn number_form_11() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]"#,
      r#"NumberForm[3.1415926535897932384626433832795028841971693993751058209749`33., 33]"#,
    );
  }
  #[test]
  fn number_form_12() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]"#,
      r#"NumberForm[0.645658509, 6]"#,
    );
  }
  #[test]
  fn number_form_13() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]"#,
      r#"NumberForm[0.14285714285714285, 30]"#,
    );
  }
  #[test]
  fn number_form_14() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]"#,
      r#"NumberForm[{0, 2, -415, 83515161451}, 5]"#,
    );
  }
  #[test]
  fn number_form_15() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]"#,
      r#"NumberForm[{10633823966279326983230456482242756608, 1.0633823966279327*^37}, 4, ExponentFunction -> (#1 & )]"#,
    );
  }
  #[test]
  fn number_form_16() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]"#,
      r#"NumberForm[{0, 10, -512}, {10, 3}]"#,
    );
  }
  #[test]
  fn number_form_17() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]"#,
      r#"NumberForm[1.5, -4]"#,
    );
  }
  #[test]
  fn number_form_18() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]"#,
      r#"NumberForm[1.5, {1.5, 2}]"#,
    );
  }
  #[test]
  fn number_form_19() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]"#,
      r#"NumberForm[1.5, {1, 2.5}]"#,
    );
  }
  #[test]
  fn number_form_20() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]"#,
      r#"NumberForm[153., 2]"#,
    );
  }
  #[test]
  fn number_form_21() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]"#,
      r#"NumberForm[0.00125, 1]"#,
    );
  }
  #[test]
  fn number_form_22() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]"#,
      r#"NumberForm[314159.2653589793, {5, 3}]"#,
    );
  }
  #[test]
  fn number_form_23() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]"#,
      r#"NumberForm[314159.2653589793, {6, 3}]"#,
    );
  }
  #[test]
  fn number_form_24() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]"#,
      r#"NumberForm[314159.2653589793, {6, 10}]"#,
    );
  }
  #[test]
  fn number_form_25() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]"#,
      r#"NumberForm[1.`19., 10, NumberPadding -> {"X", "Y"}]"#,
    );
  }
  #[test]
  fn number_form_26() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]"#,
      r#"NumberForm[12345.123456789, 14, DigitBlock -> 3]"#,
    );
  }
  #[test]
  fn number_form_27() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]"#,
      r#"NumberForm[12345.12345678, 14, DigitBlock -> 3]"#,
    );
  }
  #[test]
  fn number_form_28() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]"#,
      r#"NumberForm[314159.2653589793, 15, DigitBlock -> {4, 2}]"#,
    );
  }
  #[test]
  fn number_form_29() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]"#,
      r#"NumberForm[1.2345, 3, DigitBlock -> -4]"#,
    );
  }
  #[test]
  fn number_form_30() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]"#,
      r#"NumberForm[1.2345, 3, DigitBlock -> x]"#,
    );
  }
  #[test]
  fn number_form_31() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]"#,
      r#"NumberForm[1.2345, 3, DigitBlock -> {x, 3}]"#,
    );
  }
  #[test]
  fn number_form_32() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]"#,
      r#"NumberForm[1.2345, 3, DigitBlock -> {5, -3}]"#,
    );
  }
  #[test]
  fn number_form_33() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]"#,
      r#"NumberForm[12345.123456789, 14, ExponentFunction -> (#1 & )]"#,
    );
  }
  #[test]
  fn number_form_34() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]"#,
      r#"NumberForm[12345.123456789, 14, ExponentFunction -> (Null & )]"#,
    );
  }
  #[test]
  fn set_5() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]"#,
      r#"{1.1402564724682261*^-10, 0.003267763643053386, 93648.047476083, 2.683779414317762*^12, 7.691214220515705*^19}"#,
    );
  }
  #[test]
  fn number_form_35() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]"#,
      r#"NumberForm[{1.1402564724682261*^-10, 0.003267763643053386, 93648.047476083, 2.683779414317762*^12, 7.691214220515705*^19}, 10, ExponentFunction -> (3*Quotient[#1, 3] & )]"#,
    );
  }
  #[test]
  fn number_form_36() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]"#,
      r#"NumberForm[{1.1402564724682261*^-10, 0.003267763643053386, 93648.047476083, 2.683779414317762*^12, 7.691214220515705*^19}, 10, ExponentFunction -> (Null & )]"#,
    );
  }
  #[test]
  fn number_form_37() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]"#,
      r#"NumberForm[3.141592653589793*^8, 10, ExponentStep -> 3]"#,
    );
  }
  #[test]
  fn number_form_38() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]"#,
      r#"NumberForm[1.2345, 3, ExponentStep -> x]"#,
    );
  }
  #[test]
  fn number_form_39() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]"#,
      r#"NumberForm[1.2345, 3, ExponentStep -> 0]"#,
    );
  }
  #[test]
  fn number_form_40() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]"#,
      r#"NumberForm[{1.1402564724682261*^-10, 0.003267763643053386, 93648.047476083, 2.683779414317762*^12, 7.691214220515705*^19}, 10, ExponentStep -> 6]"#,
    );
  }
  #[test]
  fn number_form_41() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]"#,
      r#"NumberForm[{1.1402564724682261*^-10, 0.003267763643053386, 93648.047476083, 2.683779414317762*^12, 7.691214220515705*^19}, 10, NumberFormat -> (#1 & )]"#,
    );
  }
  #[test]
  fn number_form_42() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]"#,
      r#"NumberForm[1.2345, 3, NumberMultiplier -> 0]"#,
    );
  }
  #[test]
  fn number_form_43() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]"#,
      r#"NumberForm[3.1415926535897933*^7, 15, NumberMultiplier -> "*"]"#,
    );
  }
  #[test]
  fn number_form_44() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]"#,
      r#"NumberForm[1.2345, 5, NumberPoint -> ","]"#,
    );
  }
  #[test]
  fn number_form_45() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]"#,
      r#"NumberForm[1.2345, 3, NumberPoint -> 0]"#,
    );
  }
  #[test]
  fn number_form_46() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]"#,
      r#"NumberForm[1.41, {10, 5}]"#,
    );
  }
  #[test]
  fn number_form_47() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]"#,
      r#"NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]"#,
    );
  }
  #[test]
  fn number_form_48() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]"#,
      r#"NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]"#,
    );
  }
  #[test]
  fn number_form_49() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]"#,
      r#"NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]"#,
    );
  }
  #[test]
  fn number_form_50() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]"#,
      r#"NumberForm[1.2345, 3, NumberPadding -> 0]"#,
    );
  }
  #[test]
  fn number_form_51() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]"#,
      r#"NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]"#,
    );
  }
  #[test]
  fn number_form_52() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]"#,
      r#"NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding -> {"X", "Y"}]"#,
    );
  }
  #[test]
  fn number_form_53() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]"#,
      r#"NumberForm[314159.2653589793, 15, DigitBlock -> 3, NumberSeparator -> " "]"#,
    );
  }
  #[test]
  fn number_form_54() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]"#,
      r#"NumberForm[314159.2653589793, 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]"#,
    );
  }
  #[test]
  fn number_form_55() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]"#,
      r#"NumberForm[314159.2653589793, 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]"#,
    );
  }
  #[test]
  fn number_form_56() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]"#,
      r#"NumberForm[3.1415926535897933*^7, 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]"#,
    );
  }
  #[test]
  fn number_form_57() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]"#,
      r#"NumberForm[1.2345, 3, NumberSeparator -> 0]"#,
    );
  }
  #[test]
  fn number_form_58() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]"#,
      r#"NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]"#,
    );
  }
  #[test]
  fn number_form_59() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]"#,
      r#"NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]"#,
    );
  }
  #[test]
  fn number_form_60() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]"#,
      r#"NumberForm[1.2345, 3, NumberSigns -> 0]"#,
    );
  }
  #[test]
  fn number_form_61() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]"#,
      r#"NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]"#,
    );
  }
  #[test]
  fn number_form_62() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]"#,
      r#"NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]"#,
    );
  }
  #[test]
  fn number_form_63() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]"#,
      r#"NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]"#,
    );
  }
  #[test]
  fn number_form_64() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]"#,
      r#"NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]"#,
    );
  }
  #[test]
  fn number_form_65() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]"#,
      r#"NumberForm[34, ExponentFunction -> (Null & )]"#,
    );
  }
  #[test]
  fn number_form_66() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]"#,
      r#"NumberForm[50., {5, 1}]"#,
    );
  }
  #[test]
  fn number_form_67() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]"#,
      r#"NumberForm[50, {5, 1}]"#,
    );
  }
  #[test]
  fn number_form_68() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]"#,
      r#"NumberForm[43.157, {10, 1}]"#,
    );
  }
  #[test]
  fn number_form_69() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]"#,
      r#"NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]"#,
    );
  }
  #[test]
  fn number_form_70() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]; NumberForm[80.96, {16, 1}]"#,
      r#"NumberForm[80.96, {16, 1}]"#,
    );
  }
  #[test]
  fn number_form_71() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]; NumberForm[80.96, {16, 1}]; NumberForm[142.25, {10, 1}]"#,
      r#"NumberForm[142.25, {10, 1}]"#,
    );
  }
  #[test]
  fn list_literal_2() {
    // Same family as case 3837 — mathics rendered the contents to LaTeX
    // `\text{$\{$hi, you$\}$}` (with another `\	ext` typo from a Python
    // string-escape bug). wolframscript -code returns the unevaluated
    // wrapper `TeXForm[InputForm[{hi, you}]]` verbatim. Woxi matches.
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]; NumberForm[80.96, {16, 1}]; NumberForm[142.25, {10, 1}]; {"hi","you"} //InputForm //TeXForm"#,
      r#"TeXForm[InputForm[{hi, you}]]"#,
    );
  }
  #[test]
  fn te_x_form_1() {
    // mathics rendered the contents to LaTeX `a+b c`; wolframscript -code
    // returns the unevaluated wrapper `TeXForm[a + b*c]` verbatim. Woxi
    // matches.
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]; NumberForm[80.96, {16, 1}]; NumberForm[142.25, {10, 1}]; {"hi","you"} //InputForm //TeXForm; a=.;b=.;c=.;TeXForm[a+b*c]"#,
      r#"TeXForm[a + b*c]"#,
    );
  }
  #[test]
  fn te_x_form_2() {
    // Same family as cases 3836/3837 — mathics rendered the contents
    // to LaTeX `\text{a + b*c}` (with a `\	ext` typo from a Python
    // string-escape bug). wolframscript -code returns the unevaluated
    // wrapper `TeXForm[InputForm[a + b*c]]` verbatim. Woxi matches.
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]; NumberForm[80.96, {16, 1}]; NumberForm[142.25, {10, 1}]; {"hi","you"} //InputForm //TeXForm; a=.;b=.;c=.;TeXForm[a+b*c]; TeXForm[InputForm[a+b*c]]"#,
      r#"TeXForm[InputForm[a + b*c]]"#,
    );
  }
  #[test]
  fn table_form() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]; NumberForm[80.96, {16, 1}]; NumberForm[142.25, {10, 1}]; {"hi","you"} //InputForm //TeXForm; a=.;b=.;c=.;TeXForm[a+b*c]; TeXForm[InputForm[a+b*c]]; TableForm[{}]"#,
      r#"TableForm[{}]"#,
    );
  }
  #[test]
  fn list_literal_3() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]; NumberForm[80.96, {16, 1}]; NumberForm[142.25, {10, 1}]; {"hi","you"} //InputForm //TeXForm; a=.;b=.;c=.;TeXForm[a+b*c]; TeXForm[InputForm[a+b*c]]; TableForm[{}]; {{2*a, 0},{0,0}}//MatrixForm"#,
      r#"MatrixForm[{{2*a, 0}, {0, 0}}]"#,
    );
  }
  #[test]
  fn number_form_72() {
    assert_case(
      r#"BaseForm[0, 2]; BaseForm[0.0, 2]; BaseForm[N[Pi, 30], 16]; InputForm[2 x ^ 2 + 4z!]; InputForm["\$"]; NumberForm[Pi, 20]; NumberForm[2/3, 10]; NumberForm[N[Pi]]; NumberForm[N[Pi, 20]]; NumberForm[14310983091809]; z0 = 0.0;z1 = 0.0000000000000000000000000000; NumberForm[{z0, z1}, 10]; NumberForm[{z0, z1}, {10, 4}]; z0=.;z1=.; NumberForm[1.0, 10]; NumberForm[1.000000000000000000000000, 10]; NumberForm[1.0, {10, 8}]; NumberForm[N[Pi, 33], 33]; NumberForm[0.645658509, 6]; NumberForm[N[1/7], 30]; NumberForm[{0, 2, -415, 83515161451}, 5]; NumberForm[{2^123, 2^123.}, 4, ExponentFunction -> ((#1) &)]; NumberForm[{0, 10, -512}, {10, 3}]; NumberForm[1.5, -4]; NumberForm[1.5, {1.5, 2}]; NumberForm[1.5, {1, 2.5}]; NumberForm[153., 2]; NumberForm[0.00125, 1]; NumberForm[10^5 N[Pi], {5, 3}]; NumberForm[10^5 N[Pi], {6, 3}]; NumberForm[10^5 N[Pi], {6, 10}]; NumberForm[1.0000000000000000000, 10, NumberPadding -> {"X", "Y"}]; NumberForm[12345.123456789, 14, DigitBlock -> 3]; NumberForm[12345.12345678, 14, DigitBlock -> 3]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}]; NumberForm[1.2345, 3, DigitBlock -> -4]; NumberForm[1.2345, 3, DigitBlock -> x]; NumberForm[1.2345, 3, DigitBlock -> {x, 3}]; NumberForm[1.2345, 3, DigitBlock -> {5, -3}]; NumberForm[12345.123456789, 14, ExponentFunction -> ((#) &)]; NumberForm[12345.123456789, 14, ExponentFunction -> (Null&)]; y = N[Pi^Range[-20, 40, 15]]; NumberForm[y, 10, ExponentFunction -> (3 Quotient[#, 3] &)]; NumberForm[y, 10, ExponentFunction -> (Null &)]; NumberForm[10^8 N[Pi], 10, ExponentStep -> 3]; NumberForm[1.2345, 3, ExponentStep -> x]; NumberForm[1.2345, 3, ExponentStep -> 0]; NumberForm[y, 10, ExponentStep -> 6]; NumberForm[y, 10, NumberFormat -> (#1 &)]; NumberForm[1.2345, 3, NumberMultiplier -> 0]; NumberForm[N[10^ 7 Pi], 15, NumberMultiplier -> "*"]; NumberForm[1.2345, 5, NumberPoint -> ","]; NumberForm[1.2345, 3, NumberPoint -> 0]; NumberForm[1.41, {10, 5}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"", "X"}]; NumberForm[1.41, {10, 5}, NumberPadding -> {"X", "Y"}]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}]; NumberForm[1.2345, 3, NumberPadding -> 0]; NumberForm[1.41, 10, NumberPadding -> {"X", "Y"}, NumberSigns -> {"-------------", ""}]; NumberForm[{1., -1., 2.5, -2.5}, {4, 6}, NumberPadding->{"X", "Y"}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> " "]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {" ", ","}]; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[N[10^ 7 Pi], 15, DigitBlock -> 3, NumberSeparator -> {",", " "}]; NumberForm[1.2345, 3, NumberSeparator -> 0]; NumberForm[1.2345, 5, NumberSigns -> {"-", "+"}]; NumberForm[-1.2345, 5, NumberSigns -> {"- ", ""}]; NumberForm[1.2345, 3, NumberSigns -> 0]; NumberForm[1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> True, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, 6, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[-1.234, {6, 4}, SignPadding -> False, NumberPadding -> {"X", "Y"}]; NumberForm[34, ExponentFunction->(Null&)]; NumberForm[50.0, {5, 1}]; NumberForm[50, {5, 1}]; NumberForm[43.157, {10, 1}]; NumberForm[43.15752525, {10, 5}, NumberSeparator -> ",", DigitBlock -> 1]; NumberForm[80.96, {16, 1}]; NumberForm[142.25, {10, 1}]; {"hi","you"} //InputForm //TeXForm; a=.;b=.;c=.;TeXForm[a+b*c]; TeXForm[InputForm[a+b*c]]; TableForm[{}]; {{2*a, 0},{0,0}}//MatrixForm; NumberForm[N[10^ 5 Pi], 15, DigitBlock -> {4, 2}, ExponentStep->x]"#,
      r#"NumberForm[314159.2653589793, 15, DigitBlock -> {4, 2}, ExponentStep -> x]"#,
    );
  }
  #[test]
  fn string_form_7() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]"#,
      r#"StringForm["This is symbol ``.", A]"#,
    );
  }
  #[test]
  fn string_form_8() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]"#,
      r#"StringForm["This is symbol `1`.", A]"#,
    );
  }
  #[test]
  fn string_form_9() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]; StringForm["This is symbol `0`.", A]"#,
      r#"StringForm["This is symbol `0`.", A]"#,
    );
  }
  #[test]
  fn string_form_10() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]; StringForm["This is symbol `0`.", A]; StringForm["This is symbol `symbol`.", A]"#,
      r#"StringForm["This is symbol `symbol`.", A]"#,
    );
  }
  #[test]
  fn string_form_11() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]; StringForm["This is symbol `0`.", A]; StringForm["This is symbol `symbol`.", A]; StringForm["This is symbol `5`.", A]"#,
      r#"StringForm["This is symbol `5`.", A]"#,
    );
  }
  #[test]
  fn string_form_12() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]; StringForm["This is symbol `0`.", A]; StringForm["This is symbol `symbol`.", A]; StringForm["This is symbol `5`.", A]; StringForm["This is symbol `2`, then `1`.", A, B]"#,
      r#"StringForm["This is symbol `2`, then `1`.", A, B]"#,
    );
  }
  #[test]
  fn string_form_13() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]; StringForm["This is symbol `0`.", A]; StringForm["This is symbol `symbol`.", A]; StringForm["This is symbol `5`.", A]; StringForm["This is symbol `2`, then `1`.", A, B]; StringForm["This is symbol `1`, then ``.", A, B]"#,
      r#"StringForm["This is symbol `1`, then ``.", A, B]"#,
    );
  }
  #[test]
  fn string_form_14() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]; StringForm["This is symbol `0`.", A]; StringForm["This is symbol `symbol`.", A]; StringForm["This is symbol `5`.", A]; StringForm["This is symbol `2`, then `1`.", A, B]; StringForm["This is symbol `1`, then ``.", A, B]; StringForm["This is symbol `2`, then ``.", A, B]"#,
      r#"StringForm["This is symbol `2`, then ``.", A, B]"#,
    );
  }
  #[test]
  fn string_form_15() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]; StringForm["This is symbol `0`.", A]; StringForm["This is symbol `symbol`.", A]; StringForm["This is symbol `5`.", A]; StringForm["This is symbol `2`, then `1`.", A, B]; StringForm["This is symbol `1`, then ``.", A, B]; StringForm["This is symbol `2`, then ``.", A, B]; StringForm["This is symbol `.", A]"#,
      r#"StringForm["This is symbol `.", A]"#,
    );
  }
  #[test]
  fn string_form_16() {
    assert_case(
      r#"StringForm["This is symbol ``.", A]; StringForm["This is symbol `1`.", A]; StringForm["This is symbol `0`.", A]; StringForm["This is symbol `symbol`.", A]; StringForm["This is symbol `5`.", A]; StringForm["This is symbol `2`, then `1`.", A, B]; StringForm["This is symbol `1`, then ``.", A, B]; StringForm["This is symbol `2`, then ``.", A, B]; StringForm["This is symbol `.", A]; StringForm["This is symbol \`.", A]"#,
      r#"StringForm["This is symbol \`.", A]"#,
    );
  }
  #[test]
  fn string_replace_15() {
    assert_case(
      r#"a + b /. x_ + y_ -> {x, y}; StringReplace["h1d9a f483", DigitCharacter | WhitespaceCharacter -> ""]"#,
      r#""hdaf""#,
    );
  }
  #[test]
  fn string_replace_16() {
    assert_case(
      r#"a + b /. x_ + y_ -> {x, y}; StringReplace["h1d9a f483", DigitCharacter | WhitespaceCharacter -> ""]; StringReplace["abc DEF 123!", Except[LetterCharacter, WordCharacter] -> "0"]"#,
      r#""abc DEF 000!""#,
    );
  }
  #[test]
  fn expression() {
    // mathics rendered `a:b:c` with surrounding spaces (`a : b : c`).
    // wolframscript prints it tightly as `a:b:c`, which is what Woxi
    // also produces.
    assert_case(
      r#"a + b /. x_ + y_ -> {x, y}; StringReplace["h1d9a f483", DigitCharacter | WhitespaceCharacter -> ""]; StringReplace["abc DEF 123!", Except[LetterCharacter, WordCharacter] -> "0"]; a:b:c"#,
      r#"a:b:c"#,
    );
  }
  #[test]
  fn full_form() {
    assert_case(
      r#"a + b /. x_ + y_ -> {x, y}; StringReplace["h1d9a f483", DigitCharacter | WhitespaceCharacter -> ""]; StringReplace["abc DEF 123!", Except[LetterCharacter, WordCharacter] -> "0"]; a:b:c; FullForm[a:b:c]"#,
      r#"FullForm[a:b:c]"#,
    );
  }
  #[test]
  fn to_string_23() {
    assert_case(
      r#"N[3^200]; N[2^1023]; N[2^1024]; p=N[Pi,100]; ToString[p]"#,
      r#""3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068""#,
    );
  }
  #[test]
  fn n_1() {
    assert_case(
      r#"N[3^200]; N[2^1023]; N[2^1024]; p=N[Pi,100]; ToString[p]; N[1.012345678901234567890123, 20]"#,
      r#"1.012345678901234567890123`20."#,
    );
  }
  #[test]
  fn n_2() {
    assert_case(
      r#"N[3^200]; N[2^1023]; N[2^1024]; p=N[Pi,100]; ToString[p]; N[1.012345678901234567890123, 20]; N[I, 30]"#,
      r#"1.`30.*I"#,
    );
  }
  #[test]
  fn n_3() {
    assert_case(
      r#"N[3^200]; N[2^1023]; N[2^1024]; p=N[Pi,100]; ToString[p]; N[1.012345678901234567890123, 20]; N[I, 30]; N[1.012345678901234567890123, 50] //{#1, #1//Precision}&"#,
      r#"{1.012345678901234567890123`24.0053288334574, 24.0053288334574}"#,
    );
  }
  #[test]
  fn head() {
    assert_case(r#"Head[ByteArray[{1}]]"#, r#"ByteArray"#);
  }
  #[test]
  fn order_1() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]"#,
      r#"1"#,
    );
  }
  #[test]
  fn order_2() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn order_3() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]; Order["xyzzy", "xyzzy"]"#,
      r#"0"#,
    );
  }
  #[test]
  fn order_4() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]; Order["xyzzy", "xyzzy"]; Order[ByteArray[{1, 99}], ByteArray[{2, 0}]]"#,
      r#"1"#,
    );
  }
  #[test]
  fn order_5() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]; Order["xyzzy", "xyzzy"]; Order[ByteArray[{1, 99}], ByteArray[{2, 0}]]; Order["a", 1000]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn order_6() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]; Order["xyzzy", "xyzzy"]; Order[ByteArray[{1, 99}], ByteArray[{2, 0}]]; Order["a", 1000]; Order[0.9, 1]"#,
      r#"1"#,
    );
  }
  #[test]
  fn order_7() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]; Order["xyzzy", "xyzzy"]; Order[ByteArray[{1, 99}], ByteArray[{2, 0}]]; Order["a", 1000]; Order[0.9, 1]; Order[1.2, 1]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn order_8() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]; Order["xyzzy", "xyzzy"]; Order[ByteArray[{1, 99}], ByteArray[{2, 0}]]; Order["a", 1000]; Order[0.9, 1]; Order[1.2, 1]; Order[F[2], A[2]]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn order_9() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]; Order["xyzzy", "xyzzy"]; Order[ByteArray[{1, 99}], ByteArray[{2, 0}]]; Order["a", 1000]; Order[0.9, 1]; Order[1.2, 1]; Order[F[2], A[2]]; Order[F[2], F[3]]"#,
      r#"1"#,
    );
  }
  #[test]
  fn order_10() {
    assert_case(
      r#"Order["c", "d"]; Order["d", "c"]; Order["c", ByteArray[{99}]]; Order[ByteArray[{1, 99}], "ZZZZZ"]; Order["xyzzy", "xyzzy"]; Order[ByteArray[{1, 99}], ByteArray[{2, 0}]]; Order["a", 1000]; Order[0.9, 1]; Order[1.2, 1]; Order[F[2], A[2]]; Order[F[2], F[3]]; Order[F[2, 3], F[2]]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn string_match_q_24() {
    assert_case(r#"StringMatchQ["123245a6", DigitCharacter..]"#, r#"False"#);
  }
  #[test]
  fn complement() {
    assert_case(
      r#"Complement[Alphabet["Swedish"], Alphabet["English"]]"#,
      r#"{å, ä, ö}"#,
    );
  }
  #[test]
  fn to_expression_6() {
    assert_case(r#"ToExpression["log(x)", StandardForm]"#, r#"log*x"#);
  }
  #[test]
  fn characters_2() {
    assert_case(r#"Characters["\\\` "]"#, r#"{\, \`,  }"#);
  }
  #[test]
  fn string_take_10() {
    assert_case(r#"StringTake["abcd", 0] // InputForm"#, r#"InputForm[""]"#);
  }
  #[test]
  fn string_take_11() {
    assert_case(
      r#"StringTake["abcd", 0] // InputForm; StringTake["abcd", {3, 2}] // InputForm"#,
      r#"InputForm[""]"#,
    );
  }
  #[test]
  fn string_take_12() {
    assert_case(
      r#"StringTake["abcd", 0] // InputForm; StringTake["abcd", {3, 2}] // InputForm; StringTake["", {1, 0}] // InputForm"#,
      r#"InputForm[""]"#,
    );
  }
  #[test]
  fn to_character_code_2() {
    assert_case(r#"ToCharacterCode[{"ab"}]"#, r#"{{97, 98}}"#);
  }
  #[test]
  fn to_character_code_3() {
    assert_case(
      r#"ToCharacterCode[{"ab"}]; ToCharacterCode[{"\(A\)"}]"#,
      r#"{{63433, 65, 63424}}"#,
    );
  }
  #[test]
  fn from_character_code_4() {
    assert_case(r#"FromCharacterCode[{}] // InputForm"#, r#"InputForm[""]"#);
  }
  #[test]
  fn from_character_code_5() {
    // mathics rendered the result via the box-syntax escape `"\|010000"`;
    // wolframscript -code emits the literal U+10000 character (the
    // 4-byte UTF-8 sequence `f0 90 80 80`). Woxi matches wolframscript.
    assert_case(
      r#"FromCharacterCode[{}] // InputForm; FromCharacterCode[65536]"#,
      "\u{10000}",
    );
  }
  #[test]
  fn expr_3() {
    // Same family as case 3717 — `System`Convert`B64Dump`B64Encode`
    // is an internal wolframscript package function neither side
    // implements, so both return the unevaluated wrapper. The
    // mathics-scraped expectation re-encodes the `∫` (and the
    // following Wolfram private-use char) as Wolfram named characters
    // of UTF-8 bytes interpreted as Latin-1 (mojibake). Woxi preserves
    // the original UTF-8 string verbatim.
    assert_case(
      "System`Convert`B64Dump`B64Encode[\"∫ f  x\"]",
      "System`Convert`B64Dump`B64Encode[∫ f  x]",
    );
  }
  #[test]
  fn set_6() {
    assert_case(
      r#"System`Convert`B64Dump`B64Encode["∫ f  x"]; System`Convert`B64Dump`B64Decode["4oirIGYg752MIHg="]"#,
      r#"System`Convert`B64Dump`B64Decode["4oirIGYg752MIHg="]"#,
    );
  }
  #[test]
  fn string_cases_14() {
    // Single-character `Except[c]..` lifts to a `[^c]+` regex so that
    // `StringCases` no longer trips over the `regex` crate's lack of
    // look-around. Mirrors the parseEntry pattern in build_summary.wls.
    assert_case(
      r#"StringCases["- [Title](path/to.md)", "- [" ~~ lbl:(Except["]"]..) ~~ "](" ~~ tgt:(Except[")"]..) ~~ ")" :> {lbl, tgt}, 1]"#,
      r#"{{Title, path/to.md}}"#,
    );
  }

  #[test]
  fn string_cases_backreference_scans_positions() {
    // A back-reference pattern (`a_ ~~ a_`) whose match fails its constraint
    // at one start must not consume those characters: the doubled run is
    // found at a later position. Previously greedy iteration skipped "ff".
    assert_case(r#"StringCases["abcdeff", a_ ~~ a_]"#, r#"{ff}"#);
    assert_case(r#"StringCases["mississippi", a_ ~~ a_]"#, r#"{ss, ss, pp}"#);
    assert_case(r#"StringCases["hello", a_ ~~ a_]"#, r#"{ll}"#);
    assert_case(r#"StringCases["aabbcc", a_ ~~ a_]"#, r#"{aa, bb, cc}"#);
  }
}

mod padded_form {
  use super::*;

  #[test]
  fn integer_spec_pads_to_width_n_plus_one() {
    assert_eq!(interpret("ToString[PaddedForm[7, 4]]").unwrap(), "    7");
    assert_eq!(
      interpret("ToString[PaddedForm[123, 6]]").unwrap(),
      "    123"
    );
    assert_eq!(interpret("ToString[PaddedForm[-7, 4]]").unwrap(), "   -7");
  }

  #[test]
  fn list_spec_rounds_and_pads() {
    assert_eq!(
      interpret("ToString[PaddedForm[12.345, {6, 2}]]").unwrap(),
      "   12.35"
    );
    // Trailing zeros fill the fractional places
    assert_eq!(
      interpret("ToString[PaddedForm[-3.7, {5, 3}]]").unwrap(),
      " -3.700"
    );
    assert_eq!(
      interpret("ToString[PaddedForm[3.14159, {4, 1}]]").unwrap(),
      "   3.1"
    );
  }

  #[test]
  fn bare_wrapper_echoes_unevaluated() {
    assert_eq!(
      interpret("PaddedForm[12.345, {6, 2}]").unwrap(),
      "PaddedForm[12.345, {6, 2}]"
    );
    assert_eq!(interpret("PaddedForm[7, 4]").unwrap(), "PaddedForm[7, 4]");
  }
}

mod string_take_drop_specs {
  use super::*;

  #[test]
  fn over_take_and_drop_error() {
    // Regression: these silently clamped before
    assert_eq!(
      interpret(r#"StringTake["abc", 5]"#).unwrap(),
      "StringTake[abc, 5]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "StringTake::take: Cannot take positions 1 through 5 in \"abc\"."
      )),
      "expected take message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret(r#"StringDrop["abc", -5]"#).unwrap(),
      "StringDrop[abc, -5]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "StringDrop::drop: Cannot drop positions -5 through -1 in \"abc\"."
      )),
      "expected drop message, got {:?}",
      msgs
    );
  }

  #[test]
  fn reversed_and_zero_ranges() {
    // The adjacent reversed range is empty / a no-op; {1, 0} normalizes
    // to the adjacent case
    assert_eq!(interpret(r#"StringTake["abcdef", {3, 2}]"#).unwrap(), "");
    assert_eq!(
      interpret(r#"StringDrop["abcdef", {3, 2}]"#).unwrap(),
      "abcdef"
    );
    assert_eq!(interpret(r#"StringTake["abcdef", {1, 0}]"#).unwrap(), "");
    // Further reversed errors
    assert_eq!(
      interpret(r#"StringTake["abcdef", {3, 1}]"#).unwrap(),
      "StringTake[abcdef, {3, 1}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "StringTake::take: Cannot take positions 3 through 1 in \"abcdef\"."
      )),
      "expected take message, got {:?}",
      msgs
    );
    // Out-of-range single position (previously a hard error)
    assert_eq!(
      interpret(r#"StringTake["abc", {0}]"#).unwrap(),
      "StringTake[abc, {0}]"
    );
  }

  #[test]
  fn none_all_and_upto() {
    assert_eq!(interpret(r#"StringTake["abc", None]"#).unwrap(), "");
    assert_eq!(interpret(r#"StringDrop["abc", None]"#).unwrap(), "abc");
    assert_eq!(interpret(r#"StringDrop["abc", All]"#).unwrap(), "");
    assert_eq!(interpret(r#"StringTake["abc", All]"#).unwrap(), "abc");
    assert_eq!(interpret(r#"StringDrop["abcdef", UpTo[10]]"#).unwrap(), "");
    assert_eq!(
      interpret(r#"StringTake["abcdef", UpTo[10]]"#).unwrap(),
      "abcdef"
    );
  }

  #[test]
  fn non_string_emits_strse() {
    // Regression: StringTake[x, 2] returned x before
    assert_eq!(interpret("StringTake[x, 2]").unwrap(), "StringTake[x, 2]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "StringTake::strse: A string or list of strings is expected at position 1 in StringTake[x, 2]."
      )),
      "expected strse message, got {:?}",
      msgs
    );
    assert_eq!(interpret("StringDrop[x, 2]").unwrap(), "StringDrop[x, 2]");
  }

  #[test]
  fn steps_and_sublists_still_work() {
    assert_eq!(
      interpret(r#"StringTake["abcdef", {1, 5, 2}]"#).unwrap(),
      "ace"
    );
    assert_eq!(
      interpret(r#"StringTake["abcdef", {6, 1, -2}]"#).unwrap(),
      "fdb"
    );
    assert_eq!(
      interpret(r#"StringDrop["abcdef", {1, 5, 2}]"#).unwrap(),
      "bdf"
    );
    assert_eq!(
      interpret(r#"StringTake["abcdef", {{1, 2}, {3, 4}}]"#).unwrap(),
      "{ab, cd}"
    );
    assert_eq!(
      interpret(r#"StringTake[{"abcdef", "xyz"}, 2]"#).unwrap(),
      "{ab, xy}"
    );
  }
}

mod longest_common_subsequence_positions_tests {
  use woxi::interpret;

  #[test]
  fn strings() {
    assert_eq!(
      interpret(r#"LongestCommonSubsequencePositions["abcde", "ace"]"#)
        .unwrap(),
      "{{1, 1}, {1, 1}}"
    );
    assert_eq!(
      interpret(r#"LongestCommonSubsequencePositions["abc", "xbc"]"#).unwrap(),
      "{{2, 3}, {2, 3}}"
    );
    assert_eq!(
      interpret(r#"LongestCommonSubsequencePositions["1234", "1224533324"]"#)
        .unwrap(),
      "{{1, 2}, {1, 2}}"
    );
  }

  // Ties resolve to the earliest run in each argument.
  #[test]
  fn earliest_tie() {
    assert_eq!(
      interpret(r#"LongestCommonSubsequencePositions["abcabc", "abc"]"#)
        .unwrap(),
      "{{1, 3}, {1, 3}}"
    );
  }

  #[test]
  fn lists() {
    assert_eq!(
      interpret("LongestCommonSubsequencePositions[{1, 2, 3}, {2, 3}]")
        .unwrap(),
      "{{2, 3}, {1, 2}}"
    );
    assert_eq!(
      interpret("LongestCommonSubsequencePositions[{1, 2, 3}, {1, 2, 3}]")
        .unwrap(),
      "{{1, 3}, {1, 3}}"
    );
  }

  #[test]
  fn no_common_is_empty() {
    assert_eq!(
      interpret(r#"LongestCommonSubsequencePositions["abc", "xyz"]"#).unwrap(),
      "{}"
    );
    assert_eq!(
      interpret("LongestCommonSubsequencePositions[{1, 2}, {3, 4}]").unwrap(),
      "{}"
    );
  }
}
