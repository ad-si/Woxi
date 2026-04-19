use super::*;

mod pattern_matching {
  use super::*;

  mod blank_pattern {
    use super::*;

    #[test]
    fn simple_blank_matches_any() {
      // x_ matches any expression
      assert_eq!(interpret("5 /. x_ :> 10").unwrap(), "10");
      // Strings are displayed without quotes at top level (Wolfram behavior)
      assert_eq!(interpret("\"hello\" /. x_ :> \"world\"").unwrap(), "world");
    }

    #[test]
    fn blank_with_replacement_using_variable() {
      // The matched value can be used in replacement
      // Note: expressions in replacement need parentheses
      assert_eq!(interpret("5 /. x_ :> (x + 1)").unwrap(), "6");
      assert_eq!(interpret("3 /. n_ :> (n * 2)").unwrap(), "6");
    }

    #[test]
    fn blank_on_list_elements() {
      // Pattern applies to each element in a list
      // Note: expressions in replacement need parentheses
      assert_eq!(
        interpret("{1, 2, 3} /. x_ :> (x + 10)").unwrap(),
        "{11, 12, 13}"
      );
    }
  }

  mod anonymous_blank_in_set {
    use super::*;

    #[test]
    fn anonymous_blank_downvalue() {
      // f[_] = value — anonymous Blank pattern via Set should match any argument
      assert_eq!(
        interpret("ProductQ[_] = False; ProductQ[4]").unwrap(),
        "False"
      );
    }

    #[test]
    fn anonymous_blank_downvalue_multiple_args() {
      assert_eq!(interpret("h[_, _] = True; h[1, 2]").unwrap(), "True");
    }

    #[test]
    fn named_blank_downvalue_via_set() {
      // Named pattern in Set should also work
      assert_eq!(interpret("sq[x_] = x^2; sq[5]").unwrap(), "25");
    }
  }

  mod verbatim_pattern {
    use super::*;

    #[test]
    fn verbatim_matches_literal_integer() {
      assert_eq!(interpret("MatchQ[42, Verbatim[42]]").unwrap(), "True");
    }

    #[test]
    fn verbatim_matches_literal_pattern() {
      // Verbatim[_Integer] should match the literal _Integer pattern object
      assert_eq!(
        interpret("MatchQ[_Integer, Verbatim[_Integer]]").unwrap(),
        "True"
      );
    }

    #[test]
    fn verbatim_does_not_match_different_head() {
      assert_eq!(
        interpret("MatchQ[_Real, Verbatim[_Integer]]").unwrap(),
        "False"
      );
    }

    #[test]
    fn verbatim_in_cases() {
      assert_eq!(
        interpret("Cases[{1, _Integer, 2, _String}, Verbatim[_Integer]]")
          .unwrap(),
        "{_Integer}"
      );
    }

    #[test]
    fn verbatim_does_not_treat_blank_as_pattern() {
      // Verbatim[_] should only match a literal Blank, not any expression
      assert_eq!(interpret("MatchQ[42, Verbatim[_]]").unwrap(), "False");
      assert_eq!(interpret("MatchQ[_, Verbatim[_]]").unwrap(), "True");
    }

    #[test]
    fn verbatim_blank_as_replace_lhs() {
      // `_ /. Verbatim[_]->t` replaces a literal Blank with t.
      assert_eq!(interpret("_ /. Verbatim[_]->t").unwrap(), "t");
    }

    #[test]
    fn blank_pattern_matches_any_expr_as_rule_lhs() {
      // `_ -> t` used as a rule matches any expression, so `x /. _->t` → t.
      assert_eq!(interpret("x /. _->t").unwrap(), "t");
    }
  }

  mod blank_sequence_pattern {
    use super::*;

    #[test]
    fn blank_sequence_in_set_delayed() {
      // u__ (BlankSequence) matches one or more arguments
      assert_eq!(
        interpret("HalfIntegerQ[u__] := False; HalfIntegerQ[1/2]").unwrap(),
        "False"
      );
    }

    #[test]
    fn blank_sequence_with_body_reference() {
      // Named BlankSequence used in the body
      assert_eq!(interpret("g[u__] := u; g[42]").unwrap(), "42");
    }

    #[test]
    fn blank_null_sequence_in_set_delayed() {
      // u___ (BlankNullSequence) also matches single arguments
      assert_eq!(interpret("f[u___] := u; f[7]").unwrap(), "7");
    }

    #[test]
    fn double_underscore_with_head() {
      // x__Integer — BlankSequence with head constraint
      assert_eq!(interpret("f[x__Integer] := x + 1; f[5]").unwrap(), "6");
    }

    #[test]
    fn blank_sequence_multi_arg_length() {
      // f[x__] := Length[{x}] should match multiple args and wrap in Sequence
      assert_eq!(
        interpret("f[x__] := Length[{x}]; {f[x, y, z], f[]}").unwrap(),
        "{3, f[]}"
      );
    }

    #[test]
    fn blank_sequence_single_arg() {
      // Single argument should bind directly without Sequence wrapper
      assert_eq!(interpret("g[x__] := x + 1; g[5]").unwrap(), "6");
    }

    #[test]
    fn blank_null_sequence_zero_args() {
      // BlankNullSequence matches zero arguments
      assert_eq!(interpret("h[x___] := Length[{x}]; h[]").unwrap(), "0");
    }

    #[test]
    fn matchq_blank_sequence_basic() {
      // Anonymous __ matches one or more args inside function patterns
      assert_eq!(interpret("MatchQ[f[1, 2, 3], f[__]]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[f[1], f[__]]").unwrap(), "True");
      // Must match at least one
      assert_eq!(interpret("MatchQ[f[], f[__]]").unwrap(), "False");
    }

    #[test]
    fn matchq_blank_null_sequence() {
      // ___ matches zero or more
      assert_eq!(interpret("MatchQ[f[], f[___]]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[f[1], f[___]]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[f[1, 2], f[___]]").unwrap(), "True");
    }

    #[test]
    fn matchq_blank_sequence_with_head() {
      // __Integer matches one or more Integer args
      assert_eq!(
        interpret("MatchQ[f[1, 2, 3], f[__Integer]]").unwrap(),
        "True"
      );
      // Fails when any arg is not Integer
      assert_eq!(
        interpret("MatchQ[f[1, 2, x], f[__Integer]]").unwrap(),
        "False"
      );
    }

    #[test]
    fn replace_all_with_blank_sequence() {
      // Named x__ in ReplaceAll binds to Sequence
      assert_eq!(
        interpret("f[1, 2, 3] /. f[x__] :> {x}").unwrap(),
        "{1, 2, 3}"
      );
    }

    #[test]
    fn replace_all_blank_sequence_named_sum() {
      assert_eq!(
        interpret("{f[1, 2], f[3, 4, 5]} /. f[x__] :> Plus[x]").unwrap(),
        "{3, 12}"
      );
    }

    #[test]
    fn cases_with_blank_sequence() {
      assert_eq!(
        interpret("Cases[{f[1, 2], f[3], g[4, 5]}, f[__]]").unwrap(),
        "{f[1, 2], f[3]}"
      );
    }

    #[test]
    fn count_with_blank_sequence() {
      assert_eq!(
        interpret("Count[{f[1], f[2, 3], g[4]}, f[__]]").unwrap(),
        "2"
      );
    }

    #[test]
    fn position_with_blank_sequence() {
      assert_eq!(
        interpret("Position[{f[1], f[2, 3], g[4]}, f[__]]").unwrap(),
        "{{1}, {2}}"
      );
    }

    #[test]
    fn position_nested() {
      // Position searches all levels by default
      assert_eq!(
        interpret("Position[{{1, 2}, {3, 4}}, 3]").unwrap(),
        "{{2, 1}}"
      );
      assert_eq!(
        interpret("Position[{1, {2, {3, 4}}, 5}, 3]").unwrap(),
        "{{2, 2, 1}}"
      );
      assert_eq!(
        interpret("Position[{{a, b}, {c, a}}, a]").unwrap(),
        "{{1, 1}, {2, 2}}"
      );
    }

    #[test]
    fn position_with_integer_levelspec() {
      // Position[expr, pat, n] only returns matches at depth <= n.
      assert_eq!(
        interpret("Position[{a, {a, b}, {a, {a, b}}}, a, 2]").unwrap(),
        "{{1}, {2, 1}, {3, 1}}"
      );
    }

    #[test]
    fn position_with_exact_levelspec() {
      // Position[expr, pat, {n}] returns only matches at exact depth n.
      assert_eq!(
        interpret("Position[{a, {a, b}, {a, {a, b}}}, a, {2}]").unwrap(),
        "{{2, 1}, {3, 1}}"
      );
    }

    #[test]
    fn position_with_infinity_levelspec() {
      // Explicit Infinity matches the default behaviour.
      assert_eq!(
        interpret("Position[{a, {a, b}, {a, {a, b}}}, a, Infinity]").unwrap(),
        "{{1}, {2, 1}, {3, 1}, {3, 2, 1}}"
      );
    }

    #[test]
    fn position_with_pattern_and_levelspec() {
      // Pattern-based match across a nested list, limited to depth 2.
      assert_eq!(
        interpret("Position[{{1, 2, 3}, {4, 5, 6}}, _Integer, 2]").unwrap(),
        "{{1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {2, 3}}"
      );
    }

    #[test]
    fn position_with_max_count() {
      // Position[expr, pat, levelspec, n] returns at most n positions
      // (in scan order).
      assert_eq!(
        interpret("Position[{a, b, a, c, a, b, a}, a, 1, 2]").unwrap(),
        "{{1}, {3}}"
      );
      assert_eq!(
        interpret("Position[{a, b, a, c, a, b, a}, a, {1}, 3]").unwrap(),
        "{{1}, {3}, {5}}"
      );
    }

    #[test]
    fn position_with_max_count_nested() {
      // The 4-arg form should also stop early in nested structures.
      assert_eq!(
        interpret("Position[{1, {2, 3}, {4, {5, 6}}}, _Integer, Infinity, 4]")
          .unwrap(),
        "{{1}, {2, 1}, {2, 2}, {3, 1}}"
      );
    }

    #[test]
    fn position_with_max_count_zero() {
      // n = 0 should always produce the empty list.
      assert_eq!(
        interpret("Position[{1, {2, 3}, {4, {5, 6}}}, _Integer, Infinity, 0]")
          .unwrap(),
        "{}"
      );
    }

    #[test]
    fn position_with_max_count_no_match() {
      // No match should still return the empty list, regardless of n.
      assert_eq!(
        interpret("Position[{a, b, c}, x, Infinity, 5]").unwrap(),
        "{}"
      );
    }

    #[test]
    fn position_with_infinite_max_count() {
      // n = Infinity behaves the same as the 3-arg form.
      assert_eq!(
        interpret("Position[{a, b, a, c, a}, a, Infinity, Infinity]").unwrap(),
        "{{1}, {3}, {5}}"
      );
    }

    #[test]
    fn position_with_max_count_more_than_matches() {
      // If n exceeds the number of matches, all matches are returned.
      assert_eq!(
        interpret("Position[{a, b, a, c, a}, a, Infinity, 100]").unwrap(),
        "{{1}, {3}, {5}}"
      );
    }

    #[test]
    fn multiple_blank_sequences_in_definition() {
      // f[x__, y__] splits args: first gets minimum, rest goes to second
      assert_eq!(
        interpret("f[x__, y__] := {{x}, {y}}; f[1, 2, 3]").unwrap(),
        "{{1}, {2, 3}}"
      );
    }

    #[test]
    fn blank_sequence_pattern_test() {
      // __?IntegerQ matches one or more integers
      assert_eq!(
        interpret("MatchQ[f[1, 2, 3], f[__?IntegerQ]]").unwrap(),
        "True"
      );
      assert_eq!(
        interpret("MatchQ[f[1, 2, x], f[__?IntegerQ]]").unwrap(),
        "False"
      );
    }

    #[test]
    fn blank_vs_blank_sequence_specificity() {
      // Blank (u_) should take priority over BlankSequence (u__) for single args
      // Regression test for https://github.com/ad-si/Woxi/issues/95
      assert_eq!(
        interpret("f[u_] := \"single\"; f[u__] := \"multi\"; f[a]").unwrap(),
        "single"
      );
      assert_eq!(interpret("f[a, b]").unwrap(), "multi");
    }

    #[test]
    fn blank_vs_blank_sequence_specificity_reversed_definition() {
      // Even when BlankSequence is defined first, Blank should match single arg
      clear_state();
      assert_eq!(
        interpret("g[u__] := \"multi\"; g[u_] := \"single\"; g[a]").unwrap(),
        "single"
      );
      assert_eq!(interpret("g[a, b, c]").unwrap(), "multi");
    }

    #[test]
    fn blank_vs_blank_sequence_zeroq_issue_95() {
      // The exact example from issue #95
      clear_state();
      assert_eq!(
        interpret(
          "ZeroQ[u_] := PossibleZeroQ[u]; \
           ZeroQ[u__] := Catch[Scan[Function[If[ZeroQ[#],Null,Throw[False]]],{u}];True]; \
           ZeroQ[1*a-0*b]"
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn anonymous_blank_in_matchq() {
      // Standalone _ matches any single expression
      assert_eq!(interpret("MatchQ[42, _]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[{1, 2}, {_, _}]").unwrap(), "True");
    }

    #[test]
    fn anonymous_blank_with_head() {
      // _Integer matches integer, _Symbol matches symbol
      assert_eq!(interpret("MatchQ[42, _Integer]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[x, _Integer]").unwrap(), "False");
      assert_eq!(interpret("MatchQ[x, _Symbol]").unwrap(), "True");
    }
  }

  mod conditional_pattern {
    use super::*;

    #[test]
    fn condition_true_matches() {
      assert_eq!(
        interpret("6 /. x_ /; Mod[x, 2] == 0 :> \"even\"").unwrap(),
        "even"
      );
    }

    #[test]
    fn condition_false_no_match() {
      assert_eq!(
        interpret("5 /. x_ /; Mod[x, 2] == 0 :> \"even\"").unwrap(),
        "5"
      );
    }

    #[test]
    fn conditional_with_function_call() {
      assert_eq!(
        interpret("3 /. i_ /; Mod[i, 3] == 0 :> \"Fizz\"").unwrap(),
        "Fizz"
      );
      assert_eq!(
        interpret("5 /. i_ /; Mod[i, 5] == 0 :> \"Buzz\"").unwrap(),
        "Buzz"
      );
    }

    #[test]
    fn conditional_on_list() {
      assert_eq!(
        interpret("{1, 2, 3, 4} /. x_ /; x > 2 :> 0").unwrap(),
        "{1, 2, 0, 0}"
      );
    }
  }

  mod pattern_test {
    use super::*;

    #[test]
    fn pattern_test_matches() {
      assert_eq!(interpret("4 /. x_?EvenQ :> \"even\"").unwrap(), "even");
    }

    #[test]
    fn pattern_test_no_match() {
      assert_eq!(interpret("3 /. x_?EvenQ :> \"even\"").unwrap(), "3");
    }

    #[test]
    fn pattern_test_on_list() {
      assert_eq!(
        interpret("{1, 2, 3, 4} /. x_?EvenQ :> 0").unwrap(),
        "{1, 0, 3, 0}"
      );
    }

    #[test]
    fn pattern_test_with_oddq() {
      assert_eq!(
        interpret("{1, 2, 3, 4} /. x_?OddQ :> 0").unwrap(),
        "{0, 2, 0, 4}"
      );
    }

    #[test]
    fn anonymous_blank_pattern_test() {
      // _?EvenQ without a named variable
      assert_eq!(interpret("Count[{1, 2, 3, 4, 5}, _?EvenQ]").unwrap(), "2");
    }

    #[test]
    fn anonymous_blank_pattern_test_with_anonymous_function() {
      // _?(func &) with parenthesized anonymous function
      assert_eq!(
        interpret("Count[{1, 2, 3, 4, 5}, _?(MemberQ[{2, 3, 5}, #] &)]")
          .unwrap(),
        "3"
      );
    }

    #[test]
    fn pattern_test_cases() {
      assert_eq!(
        interpret("Cases[{1, \"a\", 2, \"b\", 3}, _?StringQ]").unwrap(),
        "{a, b}"
      );
    }

    #[test]
    fn pattern_test_anonymous_function_replace_all() {
      assert_eq!(
        interpret("{1, 2, 3, 4, 5} /. x_?(# > 3 &) -> 0").unwrap(),
        "{1, 2, 3, 0, 0}"
      );
    }

    #[test]
    fn pattern_test_named_with_anonymous_function() {
      assert_eq!(
        interpret("{1, 2, 3, 4} /. x_?(EvenQ[#] &) :> x^2").unwrap(),
        "{1, 4, 3, 16}"
      );
    }
  }

  mod multiple_rules {
    use super::*;

    #[test]
    fn list_of_rules_applied_in_order() {
      // First matching rule wins
      // Note: strings inside lists still show quotes (only top-level strings are unquoted)
      assert_eq!(
        interpret(
          "{1, 2, 3} /. {x_ /; x == 1 :> \"one\", x_ /; x == 2 :> \"two\"}"
        )
        .unwrap(),
        "{one, two, 3}"
      );
    }

    #[test]
    fn fizzbuzz_style_rules() {
      // Test the FizzBuzz pattern
      assert_eq!(
          interpret("15 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "FizzBuzz"
        );
      assert_eq!(
          interpret("9 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "Fizz"
        );
      assert_eq!(
          interpret("10 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "Buzz"
        );
      assert_eq!(
          interpret("7 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "7"
        );
    }
  }

  mod structural_pattern {
    use super::*;

    #[test]
    fn power_pattern_matches_all() {
      // x^n_ matches any power of x, binding n to the exponent
      assert_eq!(
        interpret("{x^2, x^3, x^4} /. x^n_ :> f[n]").unwrap(),
        "{f[2], f[3], f[4]}"
      );
    }

    #[test]
    fn power_pattern_with_condition() {
      // x^n_ /; EvenQ[n] matches only even powers
      assert_eq!(
        interpret("{x^2, x^3, x^4} /. x^n_ /; EvenQ[n] :> f[n]").unwrap(),
        "{f[2], x^3, f[4]}"
      );
    }

    #[test]
    fn power_pattern_non_matching() {
      // Pattern doesn't match non-power expressions
      assert_eq!(
        interpret("{x, x^2, y^3} /. x^n_ :> f[n]").unwrap(),
        "{x, f[2], y^3}"
      );
    }

    #[test]
    fn function_call_pattern() {
      // f[n_] pattern matching within replacement rules
      assert_eq!(
        interpret("{f[1], f[2], g[3]} /. f[n_] :> n^2").unwrap(),
        "{1, 4, g[3]}"
      );
    }
  }
}

mod alternatives {
  use super::*;

  #[test]
  fn replace_all_with_alternatives() {
    assert_eq!(
      interpret("a + b + c + d /. (a | b) -> t").unwrap(),
      "c + d + 2*t"
    );
  }

  #[test]
  fn replace_all_single_match() {
    assert_eq!(interpret("{a, b, c} /. (a | c) -> x").unwrap(), "{x, b, x}");
  }

  #[test]
  fn match_q_with_alternatives() {
    assert_eq!(interpret("MatchQ[5, _Integer | _String]").unwrap(), "True");
  }

  #[test]
  fn match_q_no_match() {
    assert_eq!(
      interpret("MatchQ[5.0, _Integer | _String]").unwrap(),
      "False"
    );
  }

  #[test]
  fn cases_with_alternatives() {
    assert_eq!(
      interpret("Cases[{1, \"a\", 2, \"b\", 3}, _Integer | _String]").unwrap(),
      "{1, a, 2, b, 3}"
    );
  }

  #[test]
  fn replace_with_three_alternatives() {
    assert_eq!(
      interpret("{a, b, c, d} /. (a | b | c) -> x").unwrap(),
      "{x, x, x, d}"
    );
  }

  #[test]
  fn alternatives_function_call_form() {
    // Alternatives[a, b, c] displays as a | b | c
    assert_eq!(interpret("Alternatives[a, b, c]").unwrap(), "a | b | c");
  }

  #[test]
  fn alternatives_single_arg() {
    assert_eq!(interpret("Alternatives[a]").unwrap(), "Alternatives[a]");
  }

  #[test]
  fn alternatives_flattening() {
    // Alternatives is Flat: nested Alternatives are flattened
    assert_eq!(
      interpret("Alternatives[Alternatives[a, b], c]").unwrap(),
      "(a | b) | c"
    );
  }

  #[test]
  fn alternatives_attributes() {
    assert_eq!(
      interpret("Attributes[Alternatives]").unwrap(),
      "{Protected}"
    );
  }

  #[test]
  fn alternatives_head() {
    assert_eq!(interpret("Head[a | b | c]").unwrap(), "Alternatives");
    assert_eq!(
      interpret("Head[Alternatives[a, b]]").unwrap(),
      "Alternatives"
    );
  }

  #[test]
  fn alternatives_match_q_function_form() {
    // MatchQ with Alternatives as FunctionCall
    assert_eq!(
      interpret("MatchQ[1, Alternatives[1, 2, 3]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[5, Alternatives[1, 2, 3]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn alternatives_cases_function_form() {
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4, 5}, Alternatives[1, 3, 5]]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn alternatives_string_replace() {
    assert_eq!(
      interpret("StringReplace[\"abcabc\", \"a\" | \"b\" -> \"x\"]").unwrap(),
      "xxcxxc"
    );
  }

  #[test]
  fn alternatives_string_cases() {
    assert_eq!(
      interpret("StringCases[\"the cat sat on the mat\", \"cat\" | \"mat\"]")
        .unwrap(),
      "{cat, mat}"
    );
  }

  #[test]
  fn alternatives_precedence_over_rule() {
    // | binds tighter than -> so "a" | "b" -> "x" is Rule[Alternatives["a","b"], "x"]
    assert_eq!(interpret("Head[\"a\" | \"b\" -> \"x\"]").unwrap(), "Rule");
  }
}

mod pattern_constructs {
  use super::*;

  #[test]
  fn pattern_sequence() {
    assert_eq!(
      interpret("PatternSequence[a, b]").unwrap(),
      "PatternSequence[a, b]"
    );
  }

  #[test]
  fn start_of_string() {
    assert_eq!(interpret("StartOfString").unwrap(), "StartOfString");
  }

  #[test]
  fn end_of_string() {
    assert_eq!(interpret("EndOfString").unwrap(), "EndOfString");
  }

  #[test]
  fn whitespace() {
    assert_eq!(interpret("Whitespace").unwrap(), "Whitespace");
  }
}

mod repeated_pattern {
  use super::*;

  #[test]
  fn matchq_repeated_literal() {
    assert_eq!(
      interpret("MatchQ[f[a, a, a], f[Repeated[a]]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn matchq_repeated_single_element() {
    assert_eq!(interpret("MatchQ[f[a], f[Repeated[a]]]").unwrap(), "True");
  }

  #[test]
  fn matchq_repeated_mismatch() {
    assert_eq!(
      interpret("MatchQ[f[a, b, a], f[Repeated[a]]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn matchq_repeated_blank_integer() {
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {Repeated[_Integer]}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn matchq_repeated_blank_mixed_types() {
    assert_eq!(
      interpret(r#"MatchQ[{1, "a", 2}, {Repeated[_Integer]}]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn matchq_repeated_with_exact_count() {
    assert_eq!(
      interpret("MatchQ[{1, 2}, {Repeated[_Integer, {2}]}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn matchq_repeated_count_mismatch() {
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {Repeated[_Integer, {2}]}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn matchq_repeated_with_range() {
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {Repeated[_Integer, {2, 4}]}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn matchq_repeated_range_too_few() {
    assert_eq!(
      interpret("MatchQ[{1}, {Repeated[_Integer, {2, 4}]}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn matchq_repeated_null_empty() {
    assert_eq!(
      interpret("MatchQ[{}, {RepeatedNull[_Integer]}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn matchq_repeated_empty_fails() {
    assert_eq!(
      interpret("MatchQ[{}, {Repeated[_Integer]}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn matchq_repeated_null_with_elements() {
    assert_eq!(
      interpret("MatchQ[{1, 2}, {RepeatedNull[_Integer]}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn replace_all_with_repeated() {
    assert_eq!(
      interpret("ReplaceAll[f[1, 2, 3], f[Repeated[_Integer]] :> \"ints\"]")
        .unwrap(),
      "ints"
    );
  }

  #[test]
  fn replace_all_repeated_no_match() {
    assert_eq!(
      interpret(r#"ReplaceAll[f[1, "a", 3], f[Repeated[_Integer]] :> "ints"]"#)
        .unwrap(),
      r#"f[1, a, 3]"#
    );
  }

  #[test]
  fn cases_with_repeated() {
    assert_eq!(
      interpret("Cases[{f[1, 2], f[a, b], g[1, 2, 3]}, f[Repeated[_Integer]]]")
        .unwrap(),
      "{f[1, 2]}"
    );
  }

  #[test]
  fn postfix_repeated_in_matchq() {
    assert_eq!(interpret("MatchQ[f[a, a, a], f[a..]]").unwrap(), "True");
  }

  #[test]
  fn postfix_repeated_null_in_matchq() {
    assert_eq!(interpret("MatchQ[f[], f[a...]]").unwrap(), "True");
  }

  #[test]
  fn repeated_with_string_expression() {
    assert_eq!(
      interpret(
        r#"StringMatchQ["abc123", LetterCharacter.. ~~ DigitCharacter..]"#
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn repeated_with_string_expression_no_match() {
    assert_eq!(
      interpret(
        r#"StringMatchQ["123abc", LetterCharacter.. ~~ DigitCharacter..]"#
      )
      .unwrap(),
      "False"
    );
  }

  #[test]
  fn repeated_combined_with_other_patterns() {
    assert_eq!(
      interpret("MatchQ[f[1, 2, 3, x], f[Repeated[_Integer], _Symbol]]")
        .unwrap(),
      "True"
    );
  }
}

mod replace_all_top_level {
  use super::*;

  #[test]
  fn replace_all_matches_whole_list_first() {
    // ReplaceAll should match the whole expression first before descending
    assert_eq!(interpret("{a, b, c} /. x_ -> {x}").unwrap(), "{{a, b, c}}");
  }

  #[test]
  fn replace_all_descends_when_top_level_fails() {
    // When the top-level doesn't match a specific pattern, descend into elements
    assert_eq!(interpret("{1, 2, 3} /. 2 -> x").unwrap(), "{1, x, 3}");
  }

  #[test]
  fn replace_all_descends_into_function_args() {
    // Should replace inside function call arguments
    assert_eq!(interpret("f[a, b, c] /. b -> x").unwrap(), "f[a, x, c]");
  }

  #[test]
  fn replace_all_descends_into_binary_op_divide() {
    // ReplaceAll must recurse into BinaryOp::Divide nodes
    assert_eq!(interpret("(a/b) /. {a -> 1, b -> 2}").unwrap(), "1/2");
    assert_eq!(interpret("(Sin[x]/Cos[x]) /. x -> 0").unwrap(), "0");
  }

  #[test]
  fn replace_all_descends_into_binary_op_power() {
    // ReplaceAll must recurse into BinaryOp::Power nodes
    assert_eq!(interpret("x^2 /. x -> 3").unwrap(), "9");
  }

  #[test]
  fn replace_all_descends_into_unary_op() {
    // ReplaceAll must recurse into UnaryOp (negation)
    assert_eq!(interpret("(-x) /. x -> 5").unwrap(), "-5");
  }

  #[test]
  fn replace_all_descends_into_nested_division_in_plus() {
    // Regression: ReplaceAll failed to substitute inside Divide within Plus
    assert_eq!(interpret("(x/y + x) /. {x -> 1, y -> 2}").unwrap(), "3/2");
  }

  #[test]
  fn replace_all_normalize_with_division() {
    // Regression: Normalize produces BinaryOp::Divide that ReplaceAll must descend into
    assert_eq!(
      interpret("(Normalize[{Cos[x] - Sin[x], Cos[x]}] /. x -> 0)[[1]]")
        .unwrap(),
      "1/Sqrt[2]"
    );
  }
}

mod replace_with_levels {
  use super::*;

  #[test]
  fn replace_at_level_2() {
    assert_eq!(
      interpret("Replace[{1, {2, {3}}}, x_Integer :> x^2, {2}]").unwrap(),
      "{1, {4, {3}}}"
    );
  }

  #[test]
  fn replace_at_level_1() {
    assert_eq!(
      interpret("Replace[{1, {2, {3}}}, x_Integer :> x + 10, {1}]").unwrap(),
      "{11, {2, {3}}}"
    );
  }

  #[test]
  fn replace_at_level_0() {
    // At level 0, only the whole expression is checked
    assert_eq!(
      interpret("Replace[{1, 2, 3}, x_Integer :> x^2]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn replace_at_level_range() {
    assert_eq!(
      interpret("Replace[{1, {2, {3}}}, x_Integer :> x^2, {1, 2}]").unwrap(),
      "{1, {4, {3}}}"
    );
  }

  #[test]
  fn replace_all_levels() {
    assert_eq!(
      interpret("Replace[{1, {2, {3}}}, x_Integer :> x^2, {1, 3}]").unwrap(),
      "{1, {4, {9}}}"
    );
  }

  #[test]
  fn replace_at_level_in_function_call() {
    assert_eq!(
      interpret("Replace[f[a, g[b]], x_ :> h[x], {1}]").unwrap(),
      "f[h[a], h[g[b]]]"
    );
  }

  #[test]
  fn replace_with_rule_at_level() {
    assert_eq!(
      interpret("Replace[{a, {b, {c}}}, x_Symbol :> ToString[x], {2}]")
        .unwrap(),
      "{a, {b, {c}}}"
    );
  }
}

mod filter_rules {
  use super::*;

  #[test]
  fn single_key() {
    assert_eq!(
      interpret("FilterRules[{x -> 100, y -> 1000}, x]").unwrap(),
      "{x -> 100}"
    );
  }

  #[test]
  fn key_list() {
    assert_eq!(
      interpret("FilterRules[{x -> 100, y -> 1000, z -> 10000}, {a, b, x, z}]")
        .unwrap(),
      "{x -> 100, z -> 10000}"
    );
  }

  #[test]
  fn no_match() {
    assert_eq!(
      interpret("FilterRules[{x -> 1, y -> 2}, {a, b}]").unwrap(),
      "{}"
    );
  }
}

mod exists {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Exists[x, x > 0]").unwrap(), "Exists[x, x > 0]");
  }

  #[test]
  fn with_list_vars() {
    assert_eq!(
      interpret("Exists[{x, y}, x + y > 0]").unwrap(),
      "Exists[{x, y}, x + y > 0]"
    );
  }

  #[test]
  fn with_condition() {
    assert_eq!(
      interpret("Exists[x, x > 0 && x < 1, x^2 < 1]").unwrap(),
      "Exists[x, x > 0 && x < 1, x^2 < 1]"
    );
  }

  #[test]
  fn for_all() {
    assert_eq!(
      interpret("ForAll[x, x^2 >= 0]").unwrap(),
      "ForAll[x, x^2 >= 0]"
    );
  }
}

mod conditioned {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Conditioned[1, 2]").unwrap(), "Conditioned[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Conditioned]").unwrap(), "Symbol");
  }
}

mod between {
  use super::*;

  #[test]
  fn basic_in_range() {
    assert_eq!(interpret("Between[6, {4, 10}]").unwrap(), "True");
  }

  #[test]
  fn out_of_range() {
    assert_eq!(interpret("Between[2, {4, 10}]").unwrap(), "False");
  }

  #[test]
  fn operator_form() {
    assert_eq!(interpret("Between[{4, 10}][6]").unwrap(), "True");
  }

  #[test]
  fn symbolic_constants() {
    assert_eq!(interpret("Between[2, {E, Pi}]").unwrap(), "False");
  }

  #[test]
  fn multiple_ranges() {
    assert_eq!(interpret("Between[5, {{1, 2}, {4, 6}}]").unwrap(), "True");
  }

  #[test]
  fn multiple_ranges_no_match() {
    assert_eq!(interpret("Between[3, {{1, 2}, {4, 6}}]").unwrap(), "False");
  }

  #[test]
  fn boundary_values() {
    assert_eq!(interpret("Between[4, {4, 10}]").unwrap(), "True");
    assert_eq!(interpret("Between[10, {4, 10}]").unwrap(), "True");
  }
}

mod free_q {
  use super::*;

  #[test]
  fn free_q_head_matching_plus() {
    // Plus is the head of a+b inside a^(a+b)
    assert_eq!(interpret("FreeQ[{1, 2, a^(a+b)}, Plus]").unwrap(), "False");
  }

  #[test]
  fn free_q_flat_subsequence() {
    // a+b is a subsequence of a+b+c (Plus is Flat)
    assert_eq!(interpret("FreeQ[a+b+c, a+b]").unwrap(), "False");
  }

  #[test]
  fn free_q_flat_subsequence_bc() {
    assert_eq!(interpret("FreeQ[a+b+c, b+c]").unwrap(), "False");
  }

  #[test]
  fn free_q_head_list() {
    assert_eq!(interpret("FreeQ[{1,2,3}, List]").unwrap(), "False");
  }

  #[test]
  fn free_q_head_plus_direct() {
    assert_eq!(interpret("FreeQ[a+b+c, Plus]").unwrap(), "False");
  }

  #[test]
  fn free_q_non_flat_no_subset() {
    // f is NOT Flat, so f[a,c] is NOT a sub-expression of f[a,b,c]
    assert_eq!(interpret("FreeQ[f[a,b,c], f[a,c]]").unwrap(), "True");
  }

  #[test]
  fn free_q_symbol_as_element() {
    // Plus appears as a literal element in the list
    assert_eq!(interpret("FreeQ[{Plus, 1}, Plus]").unwrap(), "False");
  }

  #[test]
  fn free_q_basic_true() {
    assert_eq!(interpret("FreeQ[{1, 2, 3}, 4]").unwrap(), "True");
  }

  #[test]
  fn free_q_basic_false() {
    assert_eq!(interpret("FreeQ[{1, 2, 3}, 2]").unwrap(), "False");
  }

  #[test]
  fn free_q_with_blank_pattern() {
    assert_eq!(interpret("FreeQ[{1, 2, x, 3}, _Symbol]").unwrap(), "False");
  }

  #[test]
  fn free_q_with_blank_pattern_all_integers() {
    // {1, 2, 3} still contains a Symbol: the head "List" is a Symbol.
    assert_eq!(interpret("FreeQ[{1, 2, 3}, _Symbol]").unwrap(), "False");
    // An integer atom is truly free of symbols
    assert_eq!(interpret("FreeQ[1, _Symbol]").unwrap(), "True");
    assert_eq!(interpret("FreeQ[3.14, _Symbol]").unwrap(), "True");
    assert_eq!(interpret(r#"FreeQ["hello", _Symbol]"#).unwrap(), "True");
  }

  #[test]
  fn free_q_with_integer_pattern() {
    assert_eq!(interpret("FreeQ[{1, 2, 3}, _Integer]").unwrap(), "False");
  }

  #[test]
  fn free_q_with_string_pattern() {
    assert_eq!(
      interpret(r#"FreeQ[{1, "a", 3}, _String]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn free_q_with_string_pattern_true() {
    assert_eq!(interpret("FreeQ[{1, 2, 3}, _String]").unwrap(), "True");
  }

  #[test]
  fn free_q_with_nested_pattern_in_plus() {
    // The form is Plus[x_, y_, z_] — a FunctionCall containing patterns.
    // FreeQ must detect the nested pattern and do pattern matching.
    assert_eq!(interpret("FreeQ[a+b+c, x_+y_+z_]").unwrap(), "False");
  }

  #[test]
  fn free_q_with_nested_pattern_no_match() {
    // Pattern with 4 blanks can't match Plus with 3 operands (non-Flat
    // pattern matching). The expression is free of the pattern.
    assert_eq!(interpret("FreeQ[a+b, x_+y_+z_+w_]").unwrap(), "True");
  }
}

mod select_first {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("SelectFirst[{1, 2, 3, 4, 5}, EvenQ]").unwrap(),
      "2"
    );
  }

  #[test]
  fn not_found() {
    assert_eq!(
      interpret("SelectFirst[{1, 3, 5}, EvenQ]").unwrap(),
      "Missing[NotFound]"
    );
  }

  #[test]
  fn with_default() {
    assert_eq!(
      interpret("SelectFirst[{1, 3, 5}, EvenQ, \"none\"]").unwrap(),
      "none"
    );
  }

  #[test]
  fn with_pure_function() {
    assert_eq!(interpret("SelectFirst[{1, 2, 3, 4}, (#>2&)]").unwrap(), "3");
  }
}
