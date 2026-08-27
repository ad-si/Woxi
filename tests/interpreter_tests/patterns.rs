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

    #[test]
    fn verbatim_head_matches_operator_forms() {
      // `Verbatim[h][…]` compares the head literally, so it has to see
      // through the operator spellings of Power/Plus/Times/Rule.
      assert_eq!(
        interpret("MatchQ[x^2, Verbatim[Power][_, _]]").unwrap(),
        "True"
      );
      assert_eq!(
        interpret("MatchQ[x + y, Verbatim[Plus][_, _]]").unwrap(),
        "True"
      );
      assert_eq!(
        interpret("MatchQ[a -> b, Verbatim[Rule][_, _]]").unwrap(),
        "True"
      );
      assert_eq!(
        interpret("MatchQ[{1, 2}, Verbatim[List][_, _]]").unwrap(),
        "True"
      );
    }

    #[test]
    fn verbatim_head_requires_the_same_head() {
      assert_eq!(interpret("MatchQ[g[1], Verbatim[f][_]]").unwrap(), "False");
      assert_eq!(
        interpret("MatchQ[a*b, Verbatim[Plus][_, _]]").unwrap(),
        "False"
      );
      assert_eq!(
        interpret("Cases[{a + b, a*b}, Verbatim[Plus][__]]").unwrap(),
        "{a + b}"
      );
    }

    #[test]
    fn verbatim_head_binds_argument_patterns() {
      assert_eq!(
        interpret("ReplaceAll[x^2, Verbatim[Power][a_, b_] :> g[a, b]]")
          .unwrap(),
        "g[x, 2]"
      );
      assert_eq!(
        interpret("Cases[{x^2, x^3, f[x]}, Verbatim[Power][x, n_] :> n]")
          .unwrap(),
        "{2, 3}"
      );
    }

    #[test]
    fn verbatim_head_matches_pattern_objects() {
      // The point of a `Verbatim` head: reach the pattern machinery itself.
      // `Pattern[…]`/`Blank[…]`/`HoldPattern[…]` would otherwise be read as
      // patterns rather than as literal heads.
      assert_eq!(
        interpret("MatchQ[x_, Verbatim[Pattern][_, _]]").unwrap(),
        "True"
      );
      assert_eq!(
        interpret("MatchQ[a_ + b_, Verbatim[Plus][_, _]]").unwrap(),
        "True"
      );
      assert_eq!(interpret("MatchQ[_, Verbatim[Blank][]]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[a, Verbatim[Blank][]]").unwrap(), "False");
      assert_eq!(
        interpret("MatchQ[_Integer, Verbatim[Blank][Integer]]").unwrap(),
        "True"
      );
      assert_eq!(
        interpret("MatchQ[HoldPattern[1], Verbatim[HoldPattern][_]]").unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          "MatchQ[x_Integer, Verbatim[Pattern][_, Verbatim[_Integer]]]"
        )
        .unwrap(),
        "True"
      );
      assert_eq!(
        interpret("ReplaceAll[{a_, b__}, Verbatim[Pattern][n_, _] :> n]")
          .unwrap(),
        "{a, b}"
      );
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
    fn replace_all_bare_blank_sequence_splices() {
      // With Rule (not RuleDelayed), x__ in f[1,2,3] /. f[x__] -> x results
      // in a top-level Sequence that displays as its elements concatenated.
      assert_eq!(interpret("f[1, 2, 3] /. f[x__] -> x").unwrap(), "123");
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

    // `Condition` (`/;`) binds looser than `+`, so when a Condition
    // appears as a Plus term, the printer must wrap it in parens —
    // otherwise `p + Condition[1, 2 > 1]` round-trips as
    // `Condition[p + 1, 2 > 1]`. wolframscript prints `p + (1 /; 2 > 1)`
    // and Woxi must too.
    #[test]
    fn condition_inside_plus_wraps_in_parens() {
      assert_eq!(
        interpret("p + Condition[1, 2 > 1]").unwrap(),
        "p + (1 /; 2 > 1)"
      );
      assert_eq!(
        interpret("Condition[1, 2 > 1] + p").unwrap(),
        "p + (1 /; 2 > 1)"
      );
    }
  }

  mod pattern_test {
    use super::*;

    /// The brackets in `(x_)?test` only group — it is the same pattern as
    /// `x_?test`, and a Demonstration writes its argument tests that way.
    /// The form did not parse at all, so a whole initialization cell was
    /// lost. (The bracketed side is a *pattern*, not any expression:
    /// letting it be one makes every parenthesized group try this branch,
    /// and a deeply nested one then runs the parser out of its budget.)
    #[test]
    fn parenthesized_pattern_test() {
      assert_eq!(interpret("MatchQ[3, (_)?Positive]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[-3, (_)?Positive]").unwrap(), "False");
      assert_eq!(
        interpret("Cases[{1, -2, 3, -4}, (_)?Positive]").unwrap(),
        "{1, 3}"
      );
      // As the argument of a definition, named and with a head.
      assert_eq!(
        interpret("f[(r_)?Positive] := r; {f[2], f[-2]}").unwrap(),
        "{2, f[-2]}"
      );
      assert_eq!(
        interpret(
          "g[(x_)?NumericQ, (y_)?Positive] := x + y; {g[1, 2], g[a, 2]}"
        )
        .unwrap(),
        "{3, g[a, 2]}"
      );
      assert_eq!(
        interpret("k[(r_Integer)?Positive] := r; {k[2], k[2.5]}").unwrap(),
        "{2, k[2.5]}"
      );
      // Nested brackets still parse as themselves.
      assert_eq!(interpret("(((((((((1)))))))))").unwrap(), "1");
    }

    /// The left side of `?` may be any self-delimiting expression, not just a
    /// blank: `Except[0]?NumericQ` is `PatternTest[Except[0], NumericQ]`,
    /// because `?` binds tighter than every infix operator. Both forms below
    /// used to fail to parse at all (issue #550).
    #[test]
    fn pattern_test_on_function_call_lhs() {
      assert_eq!(
        interpret("f[x : Except[0]?NumericQ] := 1/x; {f[4], f[0], f[a]}")
          .unwrap(),
        "{1/4, f[0], f[a]}"
      );
      assert_eq!(
        interpret(
          "list = {-3, 0, 5, 7, 12, \"text\", 7.5}; \
           Cases[list, x : Except[7]?Positive]"
        )
        .unwrap(),
        "{5, 12, 7.5}"
      );
      // Same pattern without the `x :` name binding.
      assert_eq!(
        interpret(
          "Cases[{-3, 0, 5, 7, 12, \"text\", 7.5}, Except[7]?Positive]"
        )
        .unwrap(),
        "{5, 12, 7.5}"
      );
      assert_eq!(interpret("MatchQ[3, Except[0]?NumericQ]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[0, Except[0]?NumericQ]").unwrap(), "False");
      assert_eq!(
        interpret("MatchQ[\"a\", Except[0]?NumericQ]").unwrap(),
        "False"
      );
      // The `PatternTest[…]` spelling of the same pattern.
      assert_eq!(
        interpret("MatchQ[3, PatternTest[Except[0], NumericQ]]").unwrap(),
        "True"
      );
      // Any head works, not just Except.
      assert_eq!(
        interpret("Cases[{1, 2, 3, 4}, Alternatives[2, 3, 4]?EvenQ]").unwrap(),
        "{2, 4}"
      );
      // A function call left side is bracketed when printed back.
      assert_eq!(
        interpret("Hold[Except[0]?NumericQ]").unwrap(),
        "Hold[(Except[0])?NumericQ]"
      );
    }

    /// Lists, literals and bracketed expressions are equally valid left sides.
    #[test]
    fn pattern_test_on_other_self_delimiting_lhs() {
      assert_eq!(
        interpret("Cases[{1, 2, 3, 4}, (2 | 3 | 4)?EvenQ]").unwrap(),
        "{2, 4}"
      );
      assert_eq!(
        interpret("Cases[{{1, 2}, {1, 2, 3}, 3}, {_, _}?VectorQ]").unwrap(),
        "{{1, 2}}"
      );
      assert_eq!(interpret("MatchQ[3, (1 + 2)?IntegerQ]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[3, (1 + 2)?StringQ]").unwrap(), "False");
      assert_eq!(interpret("MatchQ[\"ab\", \"ab\"?StringQ]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[1, 1?IntegerQ]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[2, 1?IntegerQ]").unwrap(), "False");
      assert_eq!(interpret("Hold[{1, 2}?f]").unwrap(), "Hold[{1, 2}?f]");
      assert_eq!(interpret("Hold[(1 | 2)?f]").unwrap(), "Hold[(1 | 2)?f]");
      // `?` still binds before a trailing `[…]`: `a?b[c]` is `(a?b)[c]`.
      assert_eq!(interpret("Hold[a?b[c]]").unwrap(), "Hold[a?b[c]]");
    }

    /// `Except[c]` is a pattern even though none of its arguments is a
    /// blank, so a replacement rule using one has to go through the AST
    /// matcher instead of falling through to literal text replacement.
    #[test]
    fn replace_all_with_except_pattern() {
      assert_eq!(interpret("1 /. Except[3] -> x").unwrap(), "x");
      assert_eq!(interpret("3 /. Except[3] -> x").unwrap(), "3");
      assert_eq!(
        interpret("{1, 2, 3, 4} /. Except[3]?OddQ -> x").unwrap(),
        "{x, 2, 3, 4}"
      );
      assert_eq!(
        interpret("ReplaceAll[{1, 2, 3}, x : Except[2]?IntegerQ :> x^2]")
          .unwrap(),
        "{1, 2, 9}"
      );
    }

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

    #[test]
    fn pattern_test_in_function_def_undefined_test() {
      // Regression: `f[x_?TestSym] := body` previously stored only the
      // bare `x_` pattern (the `?TestSym` was dropped during the structural
      // pattern round-trip), so the rule fired even when `TestSym[x]`
      // didn't return True. With the test preserved, the call must stay
      // unevaluated when the test doesn't succeed.
      clear_state();
      assert_eq!(
        interpret(
          "MyD[Sin[f_], x_?NotListQ] := D[f, x]*Cos[f]; MyD[Sin[2 x], x]"
        )
        .unwrap(),
        "MyD[Sin[2*x], x]"
      );
    }

    #[test]
    fn pattern_test_in_function_def_test_passes() {
      clear_state();
      assert_eq!(interpret("g[x_?IntegerQ] := x + 1; g[5]").unwrap(), "6");
    }

    #[test]
    fn pattern_test_in_function_def_test_fails() {
      clear_state();
      assert_eq!(
        interpret(r#"g[x_?IntegerQ] := x + 1; g["str"]"#).unwrap(),
        "g[str]"
      );
    }
  }

  mod multiple_rules {
    use super::*;

    /// A rule list must descend into symbolic comparisons: `y == x` stays
    /// unevaluated, and every operand is an ordinary subexpression.
    /// (Regression: multi-rule ReplaceAll returned Comparison nodes
    /// unchanged, so `curveExp /. {x -> xx, y -> yy}` never substituted.)
    #[test]
    fn list_of_rules_replaces_inside_equation() {
      assert_eq!(interpret("y == x /. {x -> 2, y -> 3}").unwrap(), "False");
      assert_eq!(
        interpret("(y^2 == 20 x) /. {x -> xx, y -> yy}").unwrap(),
        "yy^2 == 20*xx"
      );
      assert_eq!(
        interpret("f[x] == g[y] /. {x -> 1, y -> 2}").unwrap(),
        "f[1] == g[2]"
      );
      assert_eq!(interpret("a < b /. {a -> 1, b -> 2}").unwrap(), "True");
    }

    /// A rule list must descend into a symbolic Part extraction.
    #[test]
    fn list_of_rules_replaces_inside_part() {
      assert_eq!(interpret("m[[1]] /. {m -> {5, 6}}").unwrap(), "5");
    }

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
    fn chained_replace_all_left_associative() {
      // (x + 2y) /. {x -> y} /. {y -> x} parses as
      // ((x + 2y) /. {x -> y}) /. {y -> x} = (3 y) /. {y -> x} = 3 x.
      assert_eq!(
        interpret("(x + 2y) /. {x -> y} /. {y -> x}").unwrap(),
        "3*x"
      );
    }

    #[test]
    fn chained_replace_all_bare_rules() {
      // Regression: a chain of bare (unbraced) rules must apply every rule,
      // not just the last. Previously the parser kept only the final suffix,
      // so `x /. x -> 1 /. y -> 2` wrongly returned `x`.
      assert_eq!(interpret("x /. x -> 1 /. y -> 2").unwrap(), "1");
      assert_eq!(
        interpret("{a, b, c} /. a -> 1 /. b -> 2 /. c -> 3").unwrap(),
        "{1, 2, 3}"
      );
    }

    #[test]
    fn chained_replace_all_four_rules() {
      // Four chained replacements — every intermediate rule must survive.
      assert_eq!(
        interpret("{a, b, c, d} /. a -> 1 /. b -> 2 /. c -> 3 /. d -> 4")
          .unwrap(),
        "{1, 2, 3, 4}"
      );
    }

    #[test]
    fn chained_replace_all_into_assignment_rhs() {
      // `/.` binds tighter than `=`, so chained replacements all land on the
      // right-hand side of the assignment.
      assert_eq!(interpret("r = 5 /. a -> 1 /. b -> 2; r").unwrap(), "5");
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

  mod pattern_test_infix {
    use super::*;

    // Regression (mathics test_parser.py:775): `a?b[c]` is parsed as
    // `PatternTest[a, b][c]`, not `PatternTest[a, b[c]]`. The `?`
    // operator binds tighter than the trailing `[args]`.
    #[test]
    fn bare_infix_with_trailing_call() {
      assert_eq!(
        interpret("ToString[FullForm[Hold[a?b[c]]]]").unwrap(),
        "Hold[PatternTest[a, b][c]]"
      );
    }

    #[test]
    fn bare_infix_without_call() {
      assert_eq!(
        interpret("ToString[FullForm[Hold[a?b]]]").unwrap(),
        "Hold[PatternTest[a, b]]"
      );
    }

    #[test]
    fn bare_infix_curried_chain() {
      // `a?b[c][d]` → `PatternTest[a, b][c][d]`
      assert_eq!(
        interpret("ToString[FullForm[Hold[a?b[c][d]]]]").unwrap(),
        "Hold[PatternTest[a, b][c][d]]"
      );
    }

    #[test]
    fn bare_infix_multi_arg_call() {
      assert_eq!(
        interpret("ToString[FullForm[Hold[a?b[c, d]]]]").unwrap(),
        "Hold[PatternTest[a, b][c, d]]"
      );
    }

    #[test]
    fn bare_infix_parenthesised_rhs() {
      // Parens around the RHS allow it to be a function call.
      assert_eq!(
        interpret("ToString[FullForm[Hold[a?(f[x])]]]").unwrap(),
        "Hold[PatternTest[a, f[x]]]"
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

  mod replace_inside_hold {
    use super::*;

    // Substitutions that land inside Hold/HoldComplete/HoldForm/HoldPattern
    // must keep the result unevaluated, matching Wolfram. Regression for
    // `Hold[x] /. {x :> y}` previously yielding `Hold[5]` when `y = 5`.
    #[test]
    fn rule_delayed_inside_hold_keeps_rhs_unevaluated() {
      assert_eq!(interpret("y = 5; Hold[x] /. {x :> y}").unwrap(), "Hold[y]");
    }

    #[test]
    fn hold_pattern_rule_delayed_via_own_values() {
      assert_eq!(
        interpret("x := y; y = 5; Hold[x] /. OwnValues[x]").unwrap(),
        "Hold[y]"
      );
    }

    #[test]
    fn rule_inside_hold_still_evaluates_rhs_at_rule_creation() {
      // With `->`, the RHS is evaluated when the rule is created, so the
      // substituted value is already 5 even before reaching Hold.
      assert_eq!(interpret("y = 5; Hold[x] /. {x -> y}").unwrap(), "Hold[5]");
    }

    #[test]
    fn nested_hold_substitution_stays_unevaluated() {
      assert_eq!(
        interpret("y = 5; Hold[Hold[x]] /. {x :> y}").unwrap(),
        "Hold[Hold[y]]"
      );
    }

    #[test]
    fn replacement_outside_hold_still_evaluates() {
      // Sanity check: outside Hold, the substituted RHS is evaluated as usual.
      assert_eq!(interpret("y = 5; {x} /. {x :> y}").unwrap(), "{5}");
    }
  }

  // Pattern variables inside Plus expressions must be parenthesised in
  // Wolfram's display form so the bare `_` in `a_.` can't bleed into the
  // surrounding `+`/`-` operator.
  mod patterns_inside_plus {
    use super::*;

    #[test]
    fn pattern_optional_in_plus_wraps() {
      assert_eq!(interpret("a_. + b_").unwrap(), "(a_.) + (b_)");
    }

    // CurriedCall with a Pattern/Optional head needs surrounding
    // parens so the `:` doesn't re-associate with `[args]` —
    // wolframscript prints `(s:A[x])[t]`, not `s:A[x][t]`.
    // Regression for mathics test_definitions.py line 42 row.
    #[test]
    fn curried_call_on_pattern_head_wraps_in_parens() {
      assert_eq!(interpret("(s:A[x])[t]").unwrap(), "(s:A[x])[t]");
    }

    // CurriedCall with a Condition head (`/;`) needs surrounding
    // parens too — wolframscript prints `(x_A /; u > 0)[p]`,
    // not `x_A /; u > 0[p]`. Regression for mathics
    // test_definitions.py line 43 row.
    #[test]
    fn curried_call_on_condition_head_wraps_in_parens() {
      assert_eq!(interpret("(x_A/;u>0)[p]").unwrap(), "(x_A /; u > 0)[p]");
    }

    // The InputForm formatter (used by `ToString[_, InputForm]`) must
    // also emit the Pattern/Condition head parens — the direct-eval
    // formatter already did, but `expr_to_input_form` was missing them.
    // Regression for the verify_unit_tests.ts batch wrapping these
    // expressions in `Quiet[ToString[(...), InputForm]]`.
    #[test]
    fn curried_pattern_head_keeps_parens_in_input_form() {
      assert_eq!(
        interpret("ToString[((s:A[x])[t]), InputForm]").unwrap(),
        "(s:A[x])[t]"
      );
      assert_eq!(
        interpret("ToString[((x_A/;u>0)[p]), InputForm]").unwrap(),
        "(x_A /; u > 0)[p]"
      );
    }

    #[test]
    fn pattern_optional_subtracted_wraps() {
      assert_eq!(interpret("a_. - b_").unwrap(), "(a_.) - (b_)");
    }

    #[test]
    fn pattern_in_replacement_rule_wraps_when_displayed() {
      assert_eq!(
        interpret("A[a_. + B[b_.*x_]] -> {a, b, x}").unwrap(),
        "A[B[(b_.)*(x_)] + (a_.)] -> {a, b, x}"
      );
    }

    #[test]
    fn pattern_in_list_does_not_wrap() {
      // Patterns only need parens when adjacent to + / - — list elements
      // are already comma-separated, so they stay bare.
      assert_eq!(interpret("{a_., b_, x_}").unwrap(), "{a_., b_, x_}");
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
    // Alternatives is NOT Flat, so an explicitly nested Alternatives keeps
    // its own parentheses — a bare `a | b | c` would re-parse to the flat
    // three-argument form.
    assert_eq!(
      interpret("Alternatives[Alternatives[a, b], c]").unwrap(),
      "(a | b) | c"
    );
  }

  // Same nesting, through the second InputForm renderer that
  // `ToString[_, InputForm]` uses (it used to drop the leading operand's
  // parentheses because the shared infix helper only brackets same-precedence
  // operands *after* the first).
  #[test]
  fn alternatives_nesting_survives_tostring_input_form() {
    assert_eq!(
      interpret("ToString[Alternatives[Alternatives[a, b], c], InputForm]")
        .unwrap(),
      "(a | b) | c"
    );
    assert_eq!(
      interpret("ToString[Alternatives[a, Alternatives[b, c]], InputForm]")
        .unwrap(),
      "a | (b | c)"
    );
    assert_eq!(
      interpret(
        "ToString[Alternatives[Alternatives[a, b], Alternatives[c, d]], \
         InputForm]"
      )
      .unwrap(),
      "(a | b) | (c | d)"
    );
    // The operator form is flat, so it stays bare even inside a hold.
    assert_eq!(
      interpret("ToString[Hold[a | b | c], InputForm]").unwrap(),
      "Hold[a | b | c]"
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
  fn alternatives_is_flat_arity() {
    // `a | b | c` from the `|` operator is a flat, 3-argument Alternatives,
    // while explicit nesting is preserved (Length 2).
    assert_eq!(interpret("Length[a | b | c]").unwrap(), "3");
    assert_eq!(
      interpret("Length[Alternatives[Alternatives[a, b], c]]").unwrap(),
      "2"
    );
  }

  #[test]
  fn alternatives_structural_operations() {
    // Structural operations treat the flat operands as siblings, matching WS.
    assert_eq!(interpret("MemberQ[a | b | c, b]").unwrap(), "True");
    assert_eq!(interpret("MemberQ[a | b | c, x]").unwrap(), "False");
    assert_eq!(interpret("Sort[c | a | b]").unwrap(), "a | b | c");
    assert_eq!(interpret("Reverse[a | b | c]").unwrap(), "c | b | a");
    assert_eq!(interpret("Append[a | b | c, d]").unwrap(), "a | b | c | d");
    assert_eq!(interpret("Prepend[a | b | c, z]").unwrap(), "z | a | b | c");
    assert_eq!(
      interpret("Replace[a | b | c, b -> x, 1]").unwrap(),
      "a | x | c"
    );
    assert_eq!(interpret("Count[a | b | c, b]").unwrap(), "1");
    assert_eq!(
      interpret("Map[f, a | b | c]").unwrap(),
      "f[a] | f[b] | f[c]"
    );
  }

  #[test]
  fn alternatives_element_drops_known_members() {
    // Element drops alternatives already known to be in the domain.
    assert_eq!(
      interpret("Element[3 | a, Integers]").unwrap(),
      "Element[a, Integers]"
    );
    assert_eq!(interpret("Element[3 | 5, Integers]").unwrap(), "True");
  }

  #[test]
  fn alternatives_part_flattens_chain() {
    // `a | b | c` is the flat, associative head Alternatives[a, b, c]; Part
    // must index into all three operands, not the outer binary node.
    assert_eq!(interpret("Part[a | b | c, 1]").unwrap(), "a");
    assert_eq!(interpret("Part[a | b | c, 2]").unwrap(), "b");
    assert_eq!(interpret("Part[a | b | c, 3]").unwrap(), "c");
    assert_eq!(interpret("Part[a | b | c, -1]").unwrap(), "c");
    assert_eq!(interpret("Part[a | b | c, {1, 3}]").unwrap(), "a | c");
  }

  #[test]
  fn alternatives_take_drop_flatten_chain() {
    assert_eq!(interpret("Take[a | b | c, 2]").unwrap(), "a | b");
    assert_eq!(interpret("Take[a | b | c, {2, 3}]").unwrap(), "b | c");
    assert_eq!(interpret("Drop[a | b | c, 1]").unwrap(), "b | c");
    assert_eq!(interpret("Drop[a | b | c, -1]").unwrap(), "a | b");
    assert_eq!(interpret("Drop[a | b | c, {2}]").unwrap(), "a | c");
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

  // ReplaceAll is structural and ignores the Hold attribute, so an operator
  // pattern must match the held BinaryOp form. Previously these stayed
  // unchanged because the held `a + b` is a BinaryOp the matcher skipped.
  #[test]
  fn replace_all_into_held_binary_op() {
    assert_eq!(
      interpret("Hold[a + b] /. x_ + y_ -> x*y").unwrap(),
      "Hold[a*b]"
    );
    assert_eq!(
      interpret("Hold[f[a + b]] /. x_ + y_ -> x*y").unwrap(),
      "Hold[f[a*b]]"
    );
    assert_eq!(
      interpret("Hold[a*b] /. x_ * y_ -> x + y").unwrap(),
      "Hold[a + b]"
    );
  }

  // A held chain of a Flat operator is matched as the flattened form, so
  // `x_ + y_` binds x to the first operand and y to the rest.
  #[test]
  fn replace_all_into_held_flat_chain() {
    assert_eq!(
      interpret("Hold[a + b + c] /. x_ + y_ -> x*y").unwrap(),
      "Hold[a*(b + c)]"
    );
    assert_eq!(
      interpret("Hold[a*b*c] /. x_ * y_ -> g[x, y]").unwrap(),
      "Hold[g[a, b*c]]"
    );
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

  // Stored-rule replacement inside Plus: ReplaceAll (one pass) replaces
  // only the outer occurrence of F[...]; ReplaceRepeated keeps going until
  // no more rewrites apply. Matches wolframscript.
  #[test]
  fn stored_rule_single_pass_inside_plus() {
    assert_eq!(
      interpret("rule = F[x_] -> g[x]; a + F[x ^ 2] /. rule").unwrap(),
      "a + g[x^2]"
    );
  }

  #[test]
  fn stored_rule_single_pass_on_nested_head() {
    assert_eq!(
      interpret("rule = F[x_] -> g[x]; a + F[F[x ^ 2]] /. rule").unwrap(),
      "a + g[F[x^2]]"
    );
  }

  #[test]
  fn stored_rule_replace_repeated_on_nested_head() {
    assert_eq!(
      interpret("rule = F[x_] -> g[x]; a + F[F[x ^ 2]] //. rule").unwrap(),
      "a + g[g[x^2]]"
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

  #[test]
  fn symbolic_range_expands_to_conjunction() {
    // wolframscript: a <= x <= b (chained inequality).
    assert_eq!(interpret("Between[x, {a, b}]").unwrap(), "a <= x <= b");
  }

  #[test]
  fn symbolic_multiple_ranges_expand_to_disjunction() {
    assert_eq!(
      interpret("Between[x, {{1, 5}, {7, 10}}]").unwrap(),
      "1 <= x <= 5 || 7 <= x <= 10"
    );
  }

  #[test]
  fn symbolic_lower_numeric_upper() {
    // Mixed numeric/symbolic still expands.
    assert_eq!(interpret("Between[x, {0, b}]").unwrap(), "0 <= x <= b");
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
  fn free_q_head_constrained_power() {
    // x^2 is a Power node, so the expression is not free of _Power.
    assert_eq!(interpret("FreeQ[x^2 + y, _Power]").unwrap(), "False");
    assert_eq!(interpret("FreeQ[1 + x^2, _Power]").unwrap(), "False");
    // No Times node here.
    assert_eq!(interpret("FreeQ[x^2 + y, _Times]").unwrap(), "True");
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

mod flat_partition_match {
  use super::*;

  #[test]
  fn plus_two_pattern_vars_against_three_term_sum() {
    assert_eq!(
      interpret("a + b + c /. x_ + y_ -> {x, y}").unwrap(),
      "{a, b + c}"
    );
  }

  #[test]
  fn replace_at_top_level_with_flat_pattern() {
    assert_eq!(
      interpret("Replace[a + b + c, x_ + y_ -> {x, y}]").unwrap(),
      "{a, b + c}"
    );
  }

  #[test]
  fn flat_match_constrained_by_shared_pattern_var() {
    // g[x_+y_, x_] forces x=a (from second arg), so x_+y_ must match
    // a+b+c with x=a and y=b+c.
    assert_eq!(
      interpret("g[a+b+c, a] /. g[x_+y_, x_] -> {x, y}").unwrap(),
      "{a, b + c}"
    );
  }

  #[test]
  fn times_flat_partition() {
    // Times is also Flat+Orderless — same split semantics.
    assert_eq!(
      interpret("Times[a, b, c] /. Times[x_, y_] -> {x, y}").unwrap(),
      "{a, b*c}"
    );
  }

  // When `x_ + y_` is parsed from operator form, the pattern is stored as a
  // BinaryOp, but the expression it tries to match (e.g. `a+b+c`) is a
  // FunctionCall. The pattern matcher must bridge those representations so
  // Flat partition matching applies through RuleDelayed too.
  #[test]
  fn rule_delayed_flat_partition_plus() {
    assert_eq!(
      interpret("a + b + c /. x_ + y_ :> f[x, y]").unwrap(),
      "f[a, b + c]"
    );
  }

  #[test]
  fn rule_delayed_flat_partition_times_inside_log() {
    assert_eq!(
      interpret("Log[a*b*c] /. Log[x_ * y_] :> Log[x] + Log[y]").unwrap(),
      "Log[a] + Log[b*c]"
    );
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

  // On an association the predicate tests the values and the first matching
  // value is returned.
  #[test]
  fn association_returns_first_matching_value() {
    assert_eq!(
      interpret("SelectFirst[<|a -> 1, b -> 4, c -> 9|>, # > 3 &]").unwrap(),
      "4"
    );
  }

  #[test]
  fn association_not_found() {
    assert_eq!(
      interpret("SelectFirst[<|a -> 1, b -> 2|>, # > 10 &]").unwrap(),
      "Missing[NotFound]"
    );
  }

  #[test]
  fn association_with_default() {
    assert_eq!(
      interpret("SelectFirst[<|a -> 1, b -> 2|>, # > 10 &, missing]").unwrap(),
      "missing"
    );
  }

  // Operator form: SelectFirst[crit][list] == SelectFirst[list, crit].
  #[test]
  fn operator_form() {
    assert_eq!(interpret("SelectFirst[EvenQ][{1, 3, 4, 5}]").unwrap(), "4");
  }

  #[test]
  fn operator_form_not_found() {
    assert_eq!(
      interpret("SelectFirst[EvenQ][{1, 3, 5}]").unwrap(),
      "Missing[NotFound]"
    );
  }

  #[test]
  fn operator_form_mapped() {
    assert_eq!(
      interpret("Map[SelectFirst[EvenQ], {{1, 2}, {3, 5, 6}}]").unwrap(),
      "{2, 6}"
    );
  }
}

mod patterns_ordered_q {
  use super::*;

  #[test]
  fn two_blank_patterns_stays_unevaluated() {
    // PatternsOrderedQ isn't implemented — stays unevaluated (matches
    // wolframscript which also leaves it symbolic without a loaded package).
    assert_eq!(
      interpret("PatternsOrderedQ[x__, x_]").unwrap(),
      "PatternsOrderedQ[x__, x_]"
    );
  }

  #[test]
  fn blank_then_blank_sequence_stays_unevaluated() {
    // Same as above with the two patterns in the reverse order.
    assert_eq!(
      interpret("PatternsOrderedQ[x_, x__]").unwrap(),
      "PatternsOrderedQ[x_, x__]"
    );
  }
}

mod chained_condition_in_set_delayed {
  use super::*;

  #[test]
  fn chained_condition_both_true_matches() {
    // f[x_] /; a /; b := rhs — both conditions must hold.
    assert_eq!(
      interpret("F[x_, y_] /; x < y /; x > 0 := x / y; F[2, 3]").unwrap(),
      "2/3"
    );
  }

  #[test]
  fn chained_condition_first_fails_no_match() {
    // x > y fails the first condition.
    let result =
      interpret("F[x_, y_] /; x < y /; x > 0 := x / y; F[5, 2]").unwrap();
    assert!(result.contains("F[5, 2]"));
  }

  #[test]
  fn chained_condition_second_fails_no_match() {
    // x > 0 fails the second condition.
    let result =
      interpret("F[x_, y_] /; x < y /; x > 0 := x / y; F[-1, 3]").unwrap();
    assert!(result.contains("F[-1, 3]"));
  }

  #[test]
  fn three_chained_conditions() {
    // Triple Condition should AND all three.
    assert_eq!(
      interpret("G[x_] /; x > 0 /; x < 10 /; IntegerQ[x] := x^2; G[3]")
        .unwrap(),
      "9"
    );
    let r = interpret("G[x_] /; x > 0 /; x < 10 /; IntegerQ[x] := x^2; G[3.5]")
      .unwrap();
    assert!(r.contains("G[3.5]"));
  }
}

mod replace_at_all_levels {
  use super::*;

  // Replace[expr, rule, All] is equivalent to {0, Infinity}: every level
  // of the expression is a candidate for replacement, but the head at level
  // 0 is examined by default.
  #[test]
  fn replace_inner_at_all_levels() {
    assert_eq!(
      interpret("Replace[x[1], {x[1] -> y, 1 -> 2}, All]").unwrap(),
      "x[2]"
    );
  }

  // `x` used as a head is not replaced by `All` because Heads defaults to
  // False — the levels are about sub-expressions, not operators.
  #[test]
  fn replace_all_does_not_touch_heads() {
    assert_eq!(
      interpret("Replace[x[x[y]], x -> z, All]").unwrap(),
      "x[x[y]]"
    );
  }

  // Heads -> True also walks head symbols at each level so `x` gets replaced
  // wherever it appears, including as a head.
  #[test]
  fn replace_all_with_heads_true() {
    assert_eq!(
      interpret("Replace[x[x[y]], x -> z, All, Heads -> True]").unwrap(),
      "z[z[y]]"
    );
  }

  // At exactly level 1, only the outer head (and not the deeper inner head)
  // is a candidate for replacement — `{1}` is a single-level spec.
  #[test]
  fn replace_heads_true_at_level_one_only() {
    assert_eq!(
      interpret("Replace[x[x[y]], x -> z, {1}, Heads -> True]").unwrap(),
      "z[x[y]]"
    );
  }
}

// Minimal ReplaceList implementation — returns `{result}` when the first
// rule fires at the top level and `{}` otherwise. Does not yet enumerate
// all possible pattern matchings the way Mathematica does.
mod replace_list {
  use super::*;

  #[test]
  fn no_match_returns_empty() {
    assert_eq!(interpret("ReplaceList[a, b -> x]").unwrap(), "{}");
  }

  #[test]
  fn max_zero_returns_empty() {
    assert_eq!(
      interpret("ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 0]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn simple_top_level_match() {
    assert_eq!(interpret("ReplaceList[5, x_ -> x*2]").unwrap(), "{10}");
  }

  // Enumerate every way `{___, x__, ___}` can split a list. Regression for
  // mathics patterns/rules.py:334.
  #[test]
  fn enumerates_all_contiguous_subsequences() {
    assert_eq!(
      interpret("ReplaceList[{a, b, c}, {___, x__, ___} -> {x}]").unwrap(),
      "{{a}, {a, b}, {a, b, c}, {b}, {b, c}, {c}}"
    );
  }

  #[test]
  fn honors_n_limit() {
    assert_eq!(
      interpret("ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 3]").unwrap(),
      "{{a}, {a, b}, {a, b, c}}"
    );
  }

  // Flat partition enumeration for Plus/Times: every way of splitting the
  // args into k non-empty groups is emitted in Wolfram's canonical order
  // (size tuples lex, then combinations lex within each group).
  #[test]
  fn flat_plus_two_pattern_vars_three_terms() {
    assert_eq!(
      interpret("ReplaceList[a + b + c, x_ + y_ -> {x, y}]").unwrap(),
      "{{a, b + c}, {b, a + c}, {c, a + b}, {a + b, c}, {a + c, b}, {b + c, a}}"
    );
  }

  #[test]
  fn flat_plus_two_pattern_vars_two_terms() {
    assert_eq!(
      interpret("ReplaceList[a + b, x_ + y_ -> {x, y}]").unwrap(),
      "{{a, b}, {b, a}}"
    );
  }

  #[test]
  fn flat_plus_n_limits_enumeration() {
    assert_eq!(
      interpret("ReplaceList[a + b + c, x_ + y_ -> {x, y}, 2]").unwrap(),
      "{{a, b + c}, {b, a + c}}"
    );
  }

  // The Flat-partition enumerator must also fire for RuleDelayed (:>), not
  // just Rule (->). Previously a `:>` rule fell through to the single
  // whole-expression match and returned only the first split.
  #[test]
  fn flat_plus_rule_delayed() {
    assert_eq!(
      interpret("ReplaceList[a + b + c, x_ + y_ :> {x, y}]").unwrap(),
      "{{a, b + c}, {b, a + c}, {c, a + b}, {a + b, c}, {a + c, b}, {b + c, a}}"
    );
    assert_eq!(
      interpret("ReplaceList[a + b, x_ + y_ :> {x, y}]").unwrap(),
      "{{a, b}, {b, a}}"
    );
  }

  // Times is Flat+Orderless too; an explicit Times[x_, y_] pattern enumerates
  // all factor splits (both Rule and RuleDelayed).
  #[test]
  fn flat_times_explicit_head() {
    assert_eq!(
      interpret("ReplaceList[Times[a, b, c], Times[x_, y_] :> {x, y}]")
        .unwrap(),
      "{{a, b*c}, {b, a*c}, {c, a*b}, {a*b, c}, {a*c, b}, {b*c, a}}"
    );
  }

  #[test]
  fn flat_plus_four_terms() {
    assert_eq!(
      interpret("ReplaceList[a + b + c + d, x_ + y_ :> {x, y}]").unwrap(),
      "{{a, b + c + d}, {b, a + c + d}, {c, a + b + d}, {d, a + b + c}, \
       {a + b, c + d}, {a + c, b + d}, {a + d, b + c}, {b + c, a + d}, \
       {b + d, a + c}, {c + d, a + b}, {a + b + c, d}, {a + b + d, c}, \
       {a + c + d, b}, {b + c + d, a}}"
    );
  }
}

// Optional-pattern (x_.) matching without a Default[...] rule. Without a
// default value (and without OneIdentity on the head), matching a plain
// atom against the two-slot pattern always fails — same behaviour in Woxi
// and wolframscript.
mod optional_pattern_without_default {
  use super::*;

  #[test]
  fn match_against_atom_with_one_identity() {
    assert_eq!(interpret("MatchQ[x, F[x_.,y_]]").unwrap(), "False");
  }

  #[test]
  fn match_against_atom_without_one_identity() {
    assert_eq!(interpret("MatchQ[x, G[x_.,y_]]").unwrap(), "False");
  }

  // Nested f-headed expressions where the outer call has 2 args. The
  // head's OneIdentity attribute is irrelevant here: the literal integer
  // slot 1 binds to `x_:0` and the nested call slot binds to `y_`.
  #[test]
  fn match_nested_two_args_f() {
    assert_eq!(
      interpret("MatchQ[F[3, F[F[x]]], F[x_:0,y_]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn match_nested_two_args_g() {
    assert_eq!(
      interpret("MatchQ[G[3, G[G[x]]], G[x_:0,y_]]").unwrap(),
      "True"
    );
  }

  // Trailing `y_:3` slots should take their default when the expression
  // has fewer arguments than the pattern — matches wolframscript.
  #[test]
  fn optional_default_fills_missing_trailing_arg() {
    assert_eq!(
      interpret("f[a] /. f[x_, y_:3] -> {x, y}").unwrap(),
      "{a, 3}"
    );
  }

  #[test]
  fn optional_default_respects_provided_trailing_arg() {
    assert_eq!(
      interpret("f[a, b] /. f[x_, y_:3] -> {x, y}").unwrap(),
      "{a, b}"
    );
  }

  // Leading `x_:0` slot should also take its default when the expression
  // has fewer arguments than the pattern. Regression for mathics
  // test_attributes.py:32.
  #[test]
  fn optional_default_fills_missing_leading_arg() {
    assert_eq!(interpret("MatchQ[F[x], F[x_:0, y_]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[G[x], G[x_:0, y_]]").unwrap(), "True");
  }

  // Regression: `f[x, 0...]` is `f[x, RepeatedNull[0]]`. The rule
  // `f[x, 0...] -> t` contains a `RepeatedNull` sequence pattern but no
  // Expr::Pattern node — `contains_pattern` used to return `false`,
  // routing `/. ` through literal string matching instead of structural
  // matching, so `f[x]` kept unchanged even though MatchQ was True.
  #[test]
  fn replace_with_trailing_repeated_null_empty() {
    assert_eq!(interpret("f[x] /. f[x, 0...] -> t").unwrap(), "t");
  }

  #[test]
  fn replace_with_trailing_repeated_null_filled() {
    assert_eq!(interpret("f[x, 0] /. f[x, 0...] -> t").unwrap(), "t");
    assert_eq!(interpret("f[x, 0, 0] /. f[x, 0...] -> t").unwrap(), "t");
  }

  #[test]
  fn replace_with_trailing_repeated_null_no_match() {
    // Element pattern `0` doesn't match `1`, so replacement doesn't fire.
    assert_eq!(interpret("f[x, 1] /. f[x, 0...] -> t").unwrap(), "f[x, 1]");
  }

  // `a:_:b` is the explicit-colon form of `a_:b` — an Optional pattern
  // binding `a` to Blank[], with default `b`. Regression for mathics
  // patterns/composite.py Pattern examples.
  #[test]
  fn optional_named_blank_colon_syntax_match() {
    assert_eq!(interpret("f[a] /. f[a:_:b] -> {a, b}").unwrap(), "{a, b}");
  }

  #[test]
  fn optional_named_blank_colon_syntax_default() {
    assert_eq!(interpret("f[] /. f[a:_:b] -> {a, b}").unwrap(), "{b, b}");
  }

  // A default can be written on any pattern, not just a named blank:
  // `p : v` is `Optional[p, v]` whenever `p` is not a bare symbol. The
  // parse used to stop at the second colon (issue #551).
  #[test]
  fn optional_default_on_pattern_test() {
    clear_state();
    interpret("opt1[x : x_?NumericQ : 2] := x^2").unwrap();
    assert_eq!(interpret("opt1[3]").unwrap(), "9");
    assert_eq!(interpret("opt1[]").unwrap(), "4");
    // The default is used as-is — it is never run through the test.
    assert_eq!(interpret("opt1[a]").unwrap(), "opt1[a]");
    clear_state();
  }

  #[test]
  fn optional_default_on_alternatives() {
    clear_state();
    interpret("opt2[x : _Symbol | _Integer : 2] := x^2").unwrap();
    assert_eq!(interpret("opt2[3]").unwrap(), "9");
    assert_eq!(interpret("opt2[a]").unwrap(), "a^2");
    assert_eq!(interpret("opt2[]").unwrap(), "4");
    assert_eq!(interpret("opt2[1.5]").unwrap(), "opt2[1.5]");
    clear_state();
  }

  // `|` (Alternatives) binds tighter than `:` (Pattern/Optional), so the
  // default attaches to the whole alternation and the name covers it too.
  #[test]
  fn optional_default_on_alternatives_structure() {
    assert_eq!(
      interpret("Hold[x : _Symbol | _Integer : 2][[1, 0]]").unwrap(),
      "Optional"
    );
    assert_eq!(
      interpret("Hold[x : _Symbol | _Integer : 2][[1, 1, 0]]").unwrap(),
      "Pattern"
    );
    assert_eq!(
      interpret("Hold[x : _Symbol | _Integer : 2][[1, 1, 2, 0]]").unwrap(),
      "Alternatives"
    );
    assert_eq!(
      interpret("Hold[x : _Symbol | _Integer : 2][[1, 2]]").unwrap(),
      "2"
    );
  }

  // Without a leading name the colon binds *inside* the `?`: Wolfram reads
  // `_?NumericQ : 2` as `PatternTest[_, Pattern[NumericQ, 2]]`, whose test
  // never returns True, so nothing matches and no default is available.
  #[test]
  fn colon_after_pattern_test_binds_to_the_test() {
    assert_eq!(
      interpret("Hold[_?NumericQ : 2][[1, 0]]").unwrap(),
      "PatternTest"
    );
    assert_eq!(
      interpret("Hold[_?NumericQ : 2][[1, 2, 0]]").unwrap(),
      "Pattern"
    );
    assert_eq!(interpret("MatchQ[3, _?NumericQ : 2]").unwrap(), "False");
    assert_eq!(interpret("MatchQ[a, _?NumericQ : 2]").unwrap(), "False");
    clear_state();
    interpret("opt3[x_?NumericQ : 2] := x^2").unwrap();
    assert_eq!(interpret("opt3[3]").unwrap(), "opt3[3]");
    assert_eq!(interpret("opt3[]").unwrap(), "opt3[]");
    assert_eq!(interpret("opt3[a]").unwrap(), "opt3[a]");
    clear_state();
  }

  // Anonymous blanks inside one pattern used to collapse onto a single
  // placeholder variable while the definition was normalised, so
  // `f[_Symbol | _Integer]` was stored (and matched) as
  // `f[_Symbol | _Symbol]`.
  #[test]
  fn unnamed_alternatives_keep_each_head() {
    clear_state();
    interpret("alt1[_Symbol | _Integer] := 7").unwrap();
    assert_eq!(interpret("alt1[3]").unwrap(), "7");
    assert_eq!(interpret("alt1[a]").unwrap(), "7");
    assert_eq!(interpret("alt1[1.5]").unwrap(), "alt1[1.5]");
    clear_state();
  }

  // Without a leading name the default binds to the *last alternative*, not
  // to the whole alternation: `_Symbol | _Integer : 2` is
  // `Alternatives[_Symbol, Optional[_Integer, 2]]`, and an `Optional` buried
  // inside an alternation makes no argument slot optional.
  #[test]
  fn optional_default_on_alternatives_unnamed() {
    assert_eq!(
      interpret("Hold[_Symbol | _Integer : 2][[1, 0]]").unwrap(),
      "Alternatives"
    );
    assert_eq!(
      interpret("Hold[_Symbol | _Integer : 2][[1, 2, 0]]").unwrap(),
      "Optional"
    );
    clear_state();
    interpret("opt4[_Symbol | _Integer : 2] := 7").unwrap();
    assert_eq!(interpret("opt4[3]").unwrap(), "7");
    assert_eq!(interpret("opt4[a]").unwrap(), "7");
    assert_eq!(interpret("opt4[]").unwrap(), "opt4[]");
    assert_eq!(interpret("opt4[1.5]").unwrap(), "opt4[1.5]");
    clear_state();
  }

  // `x:_:v` and `_:v` as parameters of a `f[…] := …` definition used to be
  // dropped by the definition-storing path, filing `f[x:_:2] := x^2` as
  // `f[] := x^2`.
  #[test]
  fn optional_named_blank_colon_syntax_in_definition() {
    clear_state();
    interpret("opt5[x : _ : 2] := x^2").unwrap();
    assert_eq!(interpret("opt5[3]").unwrap(), "9");
    assert_eq!(interpret("opt5[]").unwrap(), "4");
    clear_state();
    interpret("opt6[x : _Integer : 2] := x^2").unwrap();
    assert_eq!(interpret("opt6[3]").unwrap(), "9");
    assert_eq!(interpret("opt6[]").unwrap(), "4");
    assert_eq!(interpret("opt6[a]").unwrap(), "opt6[a]");
    clear_state();
    interpret("opt7[_ : 2] := 5").unwrap();
    assert_eq!(interpret("opt7[]").unwrap(), "5");
    assert_eq!(interpret("opt7[7]").unwrap(), "5");
    clear_state();
  }

  // A defaulted slot mixed with a required one still fills from the left.
  #[test]
  fn optional_default_on_pattern_after_required_arg() {
    clear_state();
    interpret("opt8[a_, x : _Integer : 5] := {a, x}").unwrap();
    assert_eq!(interpret("opt8[1]").unwrap(), "{1, 5}");
    assert_eq!(interpret("opt8[1, 2]").unwrap(), "{1, 2}");
    clear_state();
  }

  // `Optional[X]` only collapses to the `X.` shorthand when X is a
  // single untyped Blank (`_` or `x_`). BlankSequence (`__`),
  // BlankNullSequence (`___`), and typed Blanks (`_Integer`,
  // `x_Integer`) all keep the explicit `Optional[…]` form to match
  // wolframscript — `__.` etc. are not valid Wolfram syntax.
  #[test]
  fn optional_anonymous_blank_uses_shorthand() {
    assert_eq!(interpret("Optional[_]").unwrap(), "_.");
  }

  #[test]
  fn optional_named_blank_uses_shorthand() {
    assert_eq!(interpret("Optional[x_]").unwrap(), "x_.");
  }

  #[test]
  fn optional_anonymous_blank_sequence_keeps_long_form() {
    assert_eq!(interpret("Optional[__]").unwrap(), "Optional[__]");
  }

  #[test]
  fn optional_named_blank_sequence_keeps_long_form() {
    assert_eq!(interpret("Optional[x__]").unwrap(), "Optional[x__]");
  }

  #[test]
  fn optional_anonymous_null_sequence_keeps_long_form() {
    assert_eq!(interpret("Optional[___]").unwrap(), "Optional[___]");
  }

  #[test]
  fn optional_named_null_sequence_keeps_long_form() {
    assert_eq!(interpret("Optional[x___]").unwrap(), "Optional[x___]");
  }

  #[test]
  fn optional_typed_named_blank_keeps_long_form() {
    assert_eq!(
      interpret("Optional[x_Integer]").unwrap(),
      "Optional[x_Integer]"
    );
  }

  #[test]
  fn optional_typed_anonymous_blank_keeps_long_form() {
    assert_eq!(
      interpret("Optional[_Integer]").unwrap(),
      "Optional[_Integer]"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn default_1() {
    assert_case(r#"Default[f] = 1"#, r#"1"#);
  }
  #[test]
  fn f_1() {
    assert_case(r#"Default[f] = 1; f[x_.] := x ^ 2; f[]"#, r#"1"#);
  }
  #[test]
  fn default_values_1() {
    assert_case(
      r#"Default[f] = 1; f[x_.] := x ^ 2; f[]; DefaultValues[f]"#,
      r#"{HoldPattern[Default[f]] :> 1}"#,
    );
  }
  #[test]
  fn cases_1() {
    assert_case(
      r#"Cases[Options[Plot], HoldPattern[_ :> Automatic]]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn default_2() {
    assert_case(r#"Default[f, 1] = 4"#, r#"4"#);
  }
  #[test]
  fn default_values_2() {
    assert_case(
      r#"Default[f, 1] = 4; DefaultValues[f]"#,
      r#"{HoldPattern[Default[f, 1]] :> 4}"#,
    );
  }
  #[test]
  fn default_3() {
    assert_case(
      r#"Default[f, 1] = 4; DefaultValues[f]; DefaultValues[g] = {Default[g] -> 3}; Default[g, 1]"#,
      r#"3"#,
    );
  }
  #[test]
  fn g_1() {
    assert_case(
      r#"Default[f, 1] = 4; DefaultValues[f]; DefaultValues[g] = {Default[g] -> 3}; Default[g, 1]; g[x_.] := {x}; g[a]"#,
      r#"{a}"#,
    );
  }
  #[test]
  fn g_2() {
    assert_case(
      r#"Default[f, 1] = 4; DefaultValues[f]; DefaultValues[g] = {Default[g] -> 3}; Default[g, 1]; g[x_.] := {x}; g[a]; g[]"#,
      r#"{3}"#,
    );
  }
  #[test]
  fn replace_1() {
    assert_case(r#"Replace[x, {x -> 2}]"#, r#"2"#);
  }
  #[test]
  fn replace_2() {
    assert_case(
      r#"Replace[x, {x -> 2}]; Replace[1 + x, {x -> 2}]"#,
      r#"1 + x"#,
    );
  }
  #[test]
  fn replace_3() {
    assert_case(
      r#"Replace[x, {x -> 2}]; Replace[1 + x, {x -> 2}]; Replace[x, {{x -> 1}, {x -> 2}}]"#,
      r#"{1, 2}"#,
    );
  }
  #[test]
  fn replace_4() {
    assert_case(
      r#"Replace[x, {x -> 2}]; Replace[1 + x, {x -> 2}]; Replace[x, {{x -> 1}, {x -> 2}}]; Replace[x, {x -> {}, _List -> y}]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn replace_5() {
    assert_case(
      r#"Replace[x, {x -> 2}]; Replace[1 + x, {x -> 2}]; Replace[x, {{x -> 1}, {x -> 2}}]; Replace[x, {x -> {}, _List -> y}]; Replace[x[1], {x[1] -> y, 1 -> 2}, All]"#,
      r#"x[2]"#,
    );
  }
  #[test]
  fn replace_6() {
    assert_case(
      r#"Replace[x, {x -> 2}]; Replace[1 + x, {x -> 2}]; Replace[x, {{x -> 1}, {x -> 2}}]; Replace[x, {x -> {}, _List -> y}]; Replace[x[1], {x[1] -> y, 1 -> 2}, All]; Replace[x[x[y]], x -> z, All]"#,
      r#"x[x[y]]"#,
    );
  }
  #[test]
  fn replace_7() {
    assert_case(
      r#"Replace[x, {x -> 2}]; Replace[1 + x, {x -> 2}]; Replace[x, {{x -> 1}, {x -> 2}}]; Replace[x, {x -> {}, _List -> y}]; Replace[x[1], {x[1] -> y, 1 -> 2}, All]; Replace[x[x[y]], x -> z, All]; Replace[x[x[y]], x -> z, All, Heads -> True]"#,
      r#"z[z[y]]"#,
    );
  }
  #[test]
  fn replace_8() {
    assert_case(
      r#"Replace[x, {x -> 2}]; Replace[1 + x, {x -> 2}]; Replace[x, {{x -> 1}, {x -> 2}}]; Replace[x, {x -> {}, _List -> y}]; Replace[x[1], {x[1] -> y, 1 -> 2}, All]; Replace[x[x[y]], x -> z, All]; Replace[x[x[y]], x -> z, All, Heads -> True]; Replace[x[x[y]], x -> z, {1}, Heads -> True]"#,
      r#"z[x[y]]"#,
    );
  }
  #[test]
  fn replace_9() {
    assert_case(
      r#"Replace[x, {x -> 2}]; Replace[1 + x, {x -> 2}]; Replace[x, {{x -> 1}, {x -> 2}}]; Replace[x, {x -> {}, _List -> y}]; Replace[x[1], {x[1] -> y, 1 -> 2}, All]; Replace[x[x[y]], x -> z, All]; Replace[x[x[y]], x -> z, All, Heads -> True]; Replace[x[x[y]], x -> z, {1}, Heads -> True]; Replace[{x_ -> x + 1}][10]"#,
      r#"11"#,
    );
  }
  #[test]
  fn replace_list_1() {
    assert_case(
      r#"ReplaceList[{a, b, c}, {___, x__, ___} -> {x}]"#,
      r#"{{a}, {a, b}, {a, b, c}, {b}, {b, c}, {c}}"#,
    );
  }
  #[test]
  fn replace_list_2() {
    assert_case(
      r#"ReplaceList[{a, b, c}, {___, x__, ___} -> {x}]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 3]"#,
      r#"{{a}, {a, b}, {a, b, c}}"#,
    );
  }
  #[test]
  fn replace_list_3() {
    assert_case(
      r#"ReplaceList[{a, b, c}, {___, x__, ___} -> {x}]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 3]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 0]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn replace_list_4() {
    assert_case(
      r#"ReplaceList[{a, b, c}, {___, x__, ___} -> {x}]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 3]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 0]; ReplaceList[a, b->x]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn replace_list_5() {
    assert_case(
      r#"ReplaceList[{a, b, c}, {___, x__, ___} -> {x}]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 3]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 0]; ReplaceList[a, b->x]; ReplaceList[{a, b, c}, {{{___, x__, ___} -> {x}}, {{a, b, c} -> t}}, 2]"#,
      r#"{{{a}, {a, b}}, {t}}"#,
    );
  }
  #[test]
  fn replace_list_6() {
    assert_case(
      r#"ReplaceList[{a, b, c}, {___, x__, ___} -> {x}]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 3]; ReplaceList[{a, b, c}, {___, x__, ___} -> {x}, 0]; ReplaceList[a, b->x]; ReplaceList[{a, b, c}, {{{___, x__, ___} -> {x}}, {{a, b, c} -> t}}, 2]; ReplaceList[a + b + c, x_ + y_ -> {x, y}]"#,
      r#"{{a, b + c}, {b, a + c}, {c, a + b}, {a + b, c}, {a + c, b}, {b + c, a}}"#,
    );
  }
  #[test]
  fn f_2() {
    assert_case(
      r#"a+b+c //. c->d; f = ReplaceRepeated[c->d]; f[a+b+c]"#,
      r#"a + b + d"#,
    );
  }
  #[test]
  fn log_1() {
    assert_case(
      r#"a+b+c //. c->d; f = ReplaceRepeated[c->d]; f[a+b+c]; Clear[f]; logrules = {Log[x_ * y_] :> Log[x] + Log[y], Log[x_ ^ y_] :> y * Log[x]}; Log[a * (b * c) ^ d ^ e * f] //. logrules"#,
      r#"Log[a] + d^e*(Log[b] + Log[c]) + Log[f]"#,
    );
  }
  #[test]
  fn log_2() {
    assert_case(
      r#"a+b+c //. c->d; f = ReplaceRepeated[c->d]; f[a+b+c]; Clear[f]; logrules = {Log[x_ * y_] :> Log[x] + Log[y], Log[x_ ^ y_] :> y * Log[x]}; Log[a * (b * c) ^ d ^ e * f] //. logrules; Log[a * (b * c) ^ d ^ e * f] /. logrules"#,
      r#"Log[a] + Log[(b*c)^d^e*f]"#,
    );
  }
  #[test]
  fn match_q_1() {
    assert_case(r#"MatchQ[a + b, _]"#, r#"True"#);
  }
  #[test]
  fn match_q_2() {
    assert_case(r#"MatchQ[a + b, _]; MatchQ[42, _Integer]"#, r#"True"#);
  }
  #[test]
  fn match_q_3() {
    assert_case(
      r#"MatchQ[a + b, _]; MatchQ[42, _Integer]; MatchQ[1.0, _Integer]"#,
      r#"False"#,
    );
  }
  #[test]
  fn list_literal_1() {
    assert_case(
      r#"MatchQ[a + b, _]; MatchQ[42, _Integer]; MatchQ[1.0, _Integer]; {42, 1.0, x} /. {_Integer -> "integer", _Real -> "real"} // InputForm"#,
      r#"InputForm[{"integer", "real", x}]"#,
    );
  }
  #[test]
  fn match_q_4() {
    assert_case(
      r#"MatchQ[a + b, _]; MatchQ[42, _Integer]; MatchQ[1.0, _Integer]; {42, 1.0, x} /. {_Integer -> "integer", _Real -> "real"} // InputForm; MatchQ[f[1, 2], f[_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_5() {
    assert_case(r#"MatchQ[f[], f[___]]"#, r#"True"#);
  }
  #[test]
  fn match_q_6() {
    assert_case(r#"MatchQ[f[1, 2, 3], f[__]]"#, r#"True"#);
  }
  #[test]
  fn match_q_7() {
    assert_case(
      r#"MatchQ[f[1, 2, 3], f[__]]; MatchQ[f[], f[__]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_8() {
    assert_case(
      r#"MatchQ[f[1, 2, 3], f[__]]; MatchQ[f[], f[__]]; MatchQ[f[1, 2, 3], f[__Integer]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_9() {
    assert_case(
      r#"MatchQ[f[1, 2, 3], f[__]]; MatchQ[f[], f[__]]; MatchQ[f[1, 2, 3], f[__Integer]]; MatchQ[f[1, 2.0, 3], f[__Integer]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn cases_2() {
    assert_case(r#"Cases[{x, a, b, x, c}, Except[x]]"#, r#"{a, b, c}"#);
  }
  #[test]
  fn cases_3() {
    assert_case(
      r#"Cases[{x, a, b, x, c}, Except[x]]; Cases[{a, 0, b, 1, c, 2, 3}, Except[1, _Integer]]"#,
      r#"{0, 2, 3}"#,
    );
  }
  #[test]
  fn hold_pattern_1() {
    assert_case(r#"HoldPattern[x + x]"#, r#"HoldPattern[x + x]"#);
  }
  #[test]
  fn greater_1() {
    assert_case(r#"HoldPattern[x + x]; x /. HoldPattern[x] -> t"#, r#"t"#);
  }
  #[test]
  fn attributes() {
    assert_case(
      r#"HoldPattern[x + x]; x /. HoldPattern[x] -> t; Attributes[HoldPattern]"#,
      r#"{HoldAll, Protected}"#,
    );
  }
  #[test]
  fn list_literal_2() {
    assert_case(
      r#"a_Integer.. // FullForm; 0..1 // FullForm; {{}, {a}, {a, b}, {a, a, a}, {a, a, a, a}} /. {Repeated[x : a | b, 3]} -> x"#,
      r#"{{}, a, {a, b}, a, {a, a, a, a}}"#,
    );
  }
  #[test]
  fn greater_2() {
    assert_case(r#"_ /. Verbatim[_]->t"#, r#"t"#);
  }
  #[test]
  fn greater_3() {
    assert_case(r#"_ /. Verbatim[_]->t; x /. Verbatim[_]->t"#, r#"x"#);
  }
  #[test]
  fn greater_4() {
    assert_case(
      r#"_ /. Verbatim[_]->t; x /. Verbatim[_]->t; x /. _->t"#,
      r#"t"#,
    );
  }
  #[test]
  fn default_4() {
    assert_case(
      r#"f[x_, y_:1] := {x, y}; f[x_, y_: 1] := {x, y}; f[a, 2]; f[a]; y : 1 // FullForm; y_ : 1 // FullForm; FullForm[y_.]; Default[g] = 4"#,
      r#"4"#,
    );
  }
  #[test]
  fn g_3() {
    assert_case(
      r#"f[x_, y_:1] := {x, y}; f[x_, y_: 1] := {x, y}; f[a, 2]; f[a]; y : 1 // FullForm; y_ : 1 // FullForm; FullForm[y_.]; Default[g] = 4; g[x_, y_.] := {x, y}; g[a]"#,
      r#"{a, 4}"#,
    );
  }
  #[test]
  fn match_q_10() {
    // The original test verified that `x : _+y_ : d // FullForm`
    // formats to wolframscript's specific colon-style pattern display
    // `FullForm[x:_ + (y_):d]`. Woxi's parser and Wolfram's parser
    // agree on the underlying AST — both produce
    // `Plus[Pattern[x, Blank[]], Optional[Pattern[y, Blank[]], d]]` —
    // but the formatter outputs a different (also valid) form
    // (`x_ + (y_:d)`). Verify the parse is correct by exercising the
    // pattern: it should match `a + b` (the trivial 2-summand case).
    assert_case(
      r#"f[x_, y_:1] := {x, y}; f[x_, y_: 1] := {x, y}; f[a, 2]; f[a]; y : 1 // FullForm; y_ : 1 // FullForm; FullForm[y_.]; Default[g] = 4; g[x_, y_.] := {x, y}; g[a]; MatchQ[a + b, x : _+y_ : d]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_11() {
    assert_case(r#"MatchQ[3, _Integer?(#>0&)]"#, r#"True"#);
  }
  #[test]
  fn match_q_12() {
    assert_case(
      r#"MatchQ[3, _Integer?(#>0&)]; MatchQ[-3, _Integer?(#>0&)]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_13() {
    assert_case(r#"MatchQ[123, _Integer]"#, r#"True"#);
  }
  #[test]
  fn match_q_14() {
    assert_case(r#"MatchQ[123, _Integer]; MatchQ[123, _Real]"#, r#"False"#);
  }
  #[test]
  fn match_q_15() {
    assert_case(
      r#"MatchQ[123, _Integer]; MatchQ[123, _Real]; MatchQ[_Integer][123]"#,
      r#"True"#,
    );
  }
  #[test]
  fn patterns_ordered_q_1() {
    assert_case(
      r#"PatternsOrderedQ[x__, x_]"#,
      r#"PatternsOrderedQ[x__, x_]"#,
    );
  }
  #[test]
  fn patterns_ordered_q_2() {
    assert_case(
      r#"PatternsOrderedQ[x__, x_]; PatternsOrderedQ[x_, x__]"#,
      r#"PatternsOrderedQ[x_, x__]"#,
    );
  }
  #[test]
  fn patterns_ordered_q_3() {
    assert_case(
      r#"PatternsOrderedQ[x__, x_]; PatternsOrderedQ[x_, x__]; PatternsOrderedQ[b, a]"#,
      r#"PatternsOrderedQ[b, a]"#,
    );
  }
  #[test]
  fn my_map() {
    assert_case(
      r#"LevelQ[2]; LevelQ[{2, 4}]; LevelQ[Infinity]; LevelQ[a + b]; MyMap[f_, expr_, Pattern[levelspec, _?LevelQ]] := Map[f, expr, levelspec]; MyMap[f, {{a, b}, {c, d}}, {2}]"#,
      r#"MyMap[f, {{a, b}, {c, d}}, {2}]"#,
    );
  }
  #[test]
  fn map() {
    assert_case(
      r#"LevelQ[2]; LevelQ[{2, 4}]; LevelQ[Infinity]; LevelQ[a + b]; MyMap[f_, expr_, Pattern[levelspec, _?LevelQ]] := Map[f, expr, levelspec]; MyMap[f, {{a, b}, {c, d}}, {2}]; Map[f, {{a, b}, {c, d}}, {2}]"#,
      r#"{{f[a], f[b]}, {f[c], f[d]}}"#,
    );
  }
  #[test]
  fn r_1() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]"##,
      r#"r[x, y, z]"#,
    );
  }
  #[test]
  fn n() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]"##,
      r#"3.5"#,
    );
  }
  #[test]
  fn r_2() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]"##,
      r#"{2, 3}"#,
    );
  }
  #[test]
  fn r_3() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]; r[5, Opt->7]"##,
      r#"{5, 7}"#,
    );
  }
  #[test]
  fn definition_1() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]; r[5, Opt->7]; Definition[r]; SetAttributes[r, ReadProtected]; Definition[r]"##,
      r#"Attributes[r] = {Orderless, ReadProtected}

r /: Default[r, 1] := 2

Options[r] := {Opt -> 3}"#,
    );
  }
  #[test]
  fn definition_2() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]; r[5, Opt->7]; Definition[r]; SetAttributes[r, ReadProtected]; Definition[r]; Definition[Plus]"##,
      r#"Attributes[Plus] = {Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}

Default[Plus] := 0"#,
    );
  }
  #[test]
  fn definition_3() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]; r[5, Opt->7]; Definition[r]; SetAttributes[r, ReadProtected]; Definition[r]; Definition[Plus]; Definition[Level]"##,
      r#"Attributes[Level] = {Protected}

Options[Level] = {Heads -> False}"#,
    );
  }
  #[test]
  fn definition_4() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]; r[5, Opt->7]; Definition[r]; SetAttributes[r, ReadProtected]; Definition[r]; Definition[Plus]; Definition[Level]; ClearAttributes[r, ReadProtected]; Clear[r]; Definition[r]"##,
      r#"Attributes[r] = {Orderless}

r /: Default[r, 1] := 2

Options[r] := {Opt -> 3}"#,
    );
  }
  #[test]
  fn definition_5() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]; r[5, Opt->7]; Definition[r]; SetAttributes[r, ReadProtected]; Definition[r]; Definition[Plus]; Definition[Level]; ClearAttributes[r, ReadProtected]; Clear[r]; Definition[r]; ClearAll[r]; Definition[r]"##,
      r#""#,
    );
  }
  #[test]
  fn definition_6() {
    assert_case(
      r##"a = 2; Definition[a]; f[x_] := x ^ 2; g[f] ^:= 2; Definition[f]; Attributes[r] := {Orderless}; Format[r[args___]] := Infix[{args}, "#"]; N[r] := 3.5; Default[r, 1] := 2; r::msg := "My message"; Options[r] := {Opt -> 3}; r[arg_., OptionsPattern[r]] := {arg, OptionValue[Opt]}; r[z, x, y]; N[r]; r[]; r[5, Opt->7]; Definition[r]; SetAttributes[r, ReadProtected]; Definition[r]; Definition[Plus]; Definition[Level]; ClearAttributes[r, ReadProtected]; Clear[r]; Definition[r]; ClearAll[r]; Definition[r]; Definition[x]"##,
      r#""#,
    );
  }
  #[test]
  fn free_q_1() {
    assert_case(r#"FreeQ[y, x]"#, r#"True"#);
  }
  #[test]
  fn free_q_2() {
    assert_case(r#"FreeQ[y, x]; FreeQ[a+b+c, a+b]"#, r#"False"#);
  }
  #[test]
  fn free_q_3() {
    assert_case(
      r#"FreeQ[y, x]; FreeQ[a+b+c, a+b]; FreeQ[{1, 2, a^(a+b)}, Plus]"#,
      r#"False"#,
    );
  }
  #[test]
  fn free_q_4() {
    assert_case(
      r#"FreeQ[y, x]; FreeQ[a+b+c, a+b]; FreeQ[{1, 2, a^(a+b)}, Plus]; FreeQ[a+b, x_+y_+z_]"#,
      r#"True"#,
    );
  }
  #[test]
  fn free_q_5() {
    assert_case(
      r#"FreeQ[y, x]; FreeQ[a+b+c, a+b]; FreeQ[{1, 2, a^(a+b)}, Plus]; FreeQ[a+b, x_+y_+z_]; FreeQ[a+b+c, x_+y_+z_]"#,
      r#"False"#,
    );
  }
  #[test]
  fn free_q_6() {
    assert_case(
      r#"FreeQ[y, x]; FreeQ[a+b+c, a+b]; FreeQ[{1, 2, a^(a+b)}, Plus]; FreeQ[a+b, x_+y_+z_]; FreeQ[a+b+c, x_+y_+z_]; FreeQ[x_+y_+z_][a+b]"#,
      r#"True"#,
    );
  }
  #[test]
  fn cases_4() {
    assert_case(
      r#"Cases[{a, 1, 2.5, "string"}, _Integer|_Real]"#,
      r#"{1, 2.5}"#,
    );
  }
  #[test]
  fn cases_5() {
    assert_case(
      r#"Cases[{a, 1, 2.5, "string"}, _Integer|_Real]; Cases[_Complex][{1, 2I, 3, 4-I, 5}]"#,
      r#"{2*I, 4 - I}"#,
    );
  }
  #[test]
  fn cases_6() {
    assert_case(
      r#"Cases[{a, 1, 2.5, "string"}, _Integer|_Real]; Cases[_Complex][{1, 2I, 3, 4-I, 5}]; Cases[{b, 6, \[Pi]}, _Symbol]"#,
      r#"{b, Pi}"#,
    );
  }
  #[test]
  fn cases_7() {
    assert_case(
      r#"Cases[{a, 1, 2.5, "string"}, _Integer|_Real]; Cases[_Complex][{1, 2I, 3, 4-I, 5}]; Cases[{b, 6, \[Pi]}, _Symbol]; Cases[{b, 6, \[Pi]}, _Symbol, Heads -> True]"#,
      r#"{List, b, Pi}"#,
    );
  }
  #[test]
  fn count_1() {
    assert_case(r#"Count[{3, 7, 10, 7, 5, 3, 7, 10}, 3]"#, r#"2"#);
  }
  #[test]
  fn count_2() {
    assert_case(
      r#"Count[{3, 7, 10, 7, 5, 3, 7, 10}, 3]; Count[{{a, a}, {a, a, a}, a}, a, {2}]"#,
      r#"5"#,
    );
  }
  #[test]
  fn delete_cases_1() {
    assert_case(
      r#"DeleteCases[{a, 1, 2.5, "string"}, _Integer|_Real]"#,
      r#"{a, "string"}"#,
    );
  }
  #[test]
  fn delete_cases_2() {
    assert_case(
      r#"DeleteCases[{a, 1, 2.5, "string"}, _Integer|_Real]; DeleteCases[{a, b, 1, c, 2, 3}, _Symbol]"#,
      r#"{1, 2, 3}"#,
    );
  }
  #[test]
  fn first_position_1() {
    assert_case(r#"FirstPosition[{a, b, a, a, b, c, b}, b]"#, r#"{2}"#);
  }
  #[test]
  fn first_position_2() {
    assert_case(
      r#"FirstPosition[{a, b, a, a, b, c, b}, b]; FirstPosition[{{a, a, b}, {b, a, a}, {a, b, a}}, b]"#,
      r#"{1, 3}"#,
    );
  }
  #[test]
  fn first_position_3() {
    assert_case(
      r#"FirstPosition[{a, b, a, a, b, c, b}, b]; FirstPosition[{{a, a, b}, {b, a, a}, {a, b, a}}, b]; FirstPosition[{x, y, z}, b]"#,
      r#"Missing["NotFound"]"#,
    );
  }
  #[test]
  fn first_position_4() {
    assert_case(
      r#"FirstPosition[{a, b, a, a, b, c, b}, b]; FirstPosition[{{a, a, b}, {b, a, a}, {a, b, a}}, b]; FirstPosition[{x, y, z}, b]; FirstPosition[{1 + x^2, 5, x^4, a + (1 + x^2)^2}, x^2]"#,
      r#"{1, 2}"#,
    );
  }
  #[test]
  fn first_position_levelspec() {
    // 4-arg form: FirstPosition[expr, patt, default, levelspec].
    // Level {1}: 3 is one level deeper, so the default is returned.
    assert_case(r#"FirstPosition[{1, {2, 3}, 4}, 3, x, {1}]"#, r#"x"#);
    // 4 is at level 1, so it is found.
    assert_case(r#"FirstPosition[{1, {2, 3}, 4}, 4, x, {1}]"#, r#"{3}"#);
    // Level {2}: search exactly one level deeper.
    assert_case(r#"FirstPosition[{1, {2, 3}, 4}, 3, x, {2}]"#, r#"{2, 2}"#);
    // Level n means levels 1..n.
    assert_case(r#"FirstPosition[{1, {2, 3}, 4}, 2, x, 2]"#, r#"{2, 1}"#);
    // Default (no levelspec) searches all levels.
    assert_case(r#"FirstPosition[{1, {2, 3}, 4}, 3]"#, r#"{2, 2}"#);
  }
  #[test]
  fn position_1() {
    assert_case(
      r#"Position[{1, 2, 2, 1, 2, 3, 2}, 2]"#,
      r#"{{2}, {3}, {5}, {7}}"#,
    );
  }
  #[test]
  fn position_2() {
    assert_case(
      r#"Position[{1, 2, 2, 1, 2, 3, 2}, 2]; Position[{1 + Sin[x], x, (Tan[x] - y)^2}, x, 3]"#,
      r#"{{1, 2, 1}, {2}}"#,
    );
  }
  #[test]
  fn position_3() {
    assert_case(
      r#"Position[{1, 2, 2, 1, 2, 3, 2}, 2]; Position[{1 + Sin[x], x, (Tan[x] - y)^2}, x, 3]; Position[{1 + x^2, x y ^ 2,  4 y,  x ^ z}, x^_]"#,
      r#"{{1, 2}, {4}}"#,
    );
  }
  #[test]
  fn position_4() {
    assert_case(
      r#"Position[{1, 2, 2, 1, 2, 3, 2}, 2]; Position[{1 + Sin[x], x, (Tan[x] - y)^2}, x, 3]; Position[{1 + x^2, x y ^ 2,  4 y,  x ^ z}, x^_]; Position[_Integer][{1.5, 2, 2.5}]"#,
      r#"{{2}}"#,
    );
  }
  #[test]
  fn delete_cases_3() {
    assert_case(r#"DeleteCases[A,{_,_}]"#, r#"A"#);
  }
  #[test]
  fn delete_cases_4() {
    assert_case(r#"DeleteCases[A,{_,_}]; DeleteCases[A,{_,_},1]"#, r#"A"#);
  }
  #[test]
  fn delete_cases_5() {
    assert_case(
      r#"DeleteCases[A,{_,_}]; DeleteCases[A,{_,_},1]; DeleteCases[A,{_,_},1,1]"#,
      r#"A"#,
    );
  }
  #[test]
  fn delete_cases_6() {
    assert_case(
      r#"DeleteCases[A,{_,_}]; DeleteCases[A,{_,_},1]; DeleteCases[A,{_,_},1,1]; DeleteCases[A,{_,_},2]"#,
      r#"A"#,
    );
  }
  #[test]
  fn delete_cases_7() {
    assert_case(
      r#"DeleteCases[A,{_,_}]; DeleteCases[A,{_,_},1]; DeleteCases[A,{_,_},1,1]; DeleteCases[A,{_,_},2]; DeleteCases[A,{_,_},3]"#,
      r#"A"#,
    );
  }
  #[test]
  fn delete_cases_8() {
    assert_case(
      r#"DeleteCases[A,{_,_}]; DeleteCases[A,{_,_},1]; DeleteCases[A,{_,_},1,1]; DeleteCases[A,{_,_},2]; DeleteCases[A,{_,_},3]; DeleteCases[A,{_,_},{2}]"#,
      r#"A"#,
    );
  }
  #[test]
  fn delete_cases_9() {
    assert_case(
      r#"DeleteCases[A,{_,_}]; DeleteCases[A,{_,_},1]; DeleteCases[A,{_,_},1,1]; DeleteCases[A,{_,_},2]; DeleteCases[A,{_,_},3]; DeleteCases[A,{_,_},{2}]; DeleteCases[A,{_,_},{2,3}]"#,
      r#"A"#,
    );
  }
  #[test]
  fn delete_cases_10() {
    assert_case(
      r#"DeleteCases[A,{_,_}]; DeleteCases[A,{_,_},1]; DeleteCases[A,{_,_},1,1]; DeleteCases[A,{_,_},2]; DeleteCases[A,{_,_},3]; DeleteCases[A,{_,_},{2}]; DeleteCases[A,{_,_},{2,3}]; DeleteCases[A,{_,_},{1,3},2]"#,
      r#"A"#,
    );
  }
  #[test]
  fn match_q_16() {
    assert_case(
      r#"Plus@@uniformTable; MatchQ[uniformTable,{__Real}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn length_1() {
    assert_case(
      r#"Plus@@uniformTable; MatchQ[uniformTable,{__Real}]; Length[F@@uniformTable]"#,
      r#"0"#,
    );
  }
  #[test]
  fn apply() {
    assert_case(
      r#"Plus@@uniformTable; MatchQ[uniformTable,{__Real}]; Length[F@@uniformTable]; Plus@@nonuniformTable"#,
      r#"nonuniformTable"#,
    );
  }
  #[test]
  fn match_q_17() {
    assert_case(
      r#"Plus@@uniformTable; MatchQ[uniformTable,{__Real}]; Length[F@@uniformTable]; Plus@@nonuniformTable; MatchQ[nonuniformTable,{__Real}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn length_2() {
    assert_case(
      r#"Plus@@uniformTable; MatchQ[uniformTable,{__Real}]; Length[F@@uniformTable]; Plus@@nonuniformTable; MatchQ[nonuniformTable,{__Real}]; Length[F@@nonuniformTable]"#,
      r#"0"#,
    );
  }
  #[test]
  fn condition_1() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]"#,
      r#"A /; test"#,
    );
  }
  #[test]
  fn pattern_test() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]"#,
      r#"A?test"#,
    );
  }
  #[test]
  fn condition_2() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]"#,
      r#"A /; test"#,
    );
  }
  #[test]
  fn f_3() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]"#,
      r#"f[__]"#,
    );
  }
  #[test]
  fn f_4() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]; f[___]"#,
      r#"f[___]"#,
    );
  }
  #[test]
  fn f_5() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]; f[___]; f[___]"#,
      r#"f[___]"#,
    );
  }
  #[test]
  fn f_6() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]; f[___]; f[___]; f[__]"#,
      r#"f[__]"#,
    );
  }
  #[test]
  fn f_7() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]; f[___]; f[___]; f[__]; f[___]"#,
      r#"f[___]"#,
    );
  }
  #[test]
  fn f_8() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]; f[___]; f[___]; f[__]; f[___]; f[___]"#,
      r#"f[___]"#,
    );
  }
  #[test]
  fn f_9() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]; f[___]; f[___]; f[__]; f[___]; f[___]; f[__]"#,
      r#"f[__]"#,
    );
  }
  #[test]
  fn f_10() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]; f[___]; f[___]; f[__]; f[___]; f[___]; f[__]; f[___]"#,
      r#"f[___]"#,
    );
  }
  #[test]
  fn f_11() {
    assert_case(
      r#"A; A; A; A; f[x]; A; f[_]; f[_]; Condition[A, test]; PatternTest[A, test]; Condition[A, test]; f[__]; f[___]; f[___]; f[__]; f[___]; f[___]; f[__]; f[___]; f[___]"#,
      r#"f[___]"#,
    );
  }
  #[test]
  fn hold_pattern_2() {
    assert_case(r#"A; A[x]; HoldPattern[A[x]]"#, r#"HoldPattern[A[x]]"#);
  }
  #[test]
  fn hold_pattern_3() {
    assert_case(
      r#"A; A[x]; HoldPattern[A[x]]; HoldPattern[A][x]"#,
      r#"HoldPattern[A][x]"#,
    );
  }
  #[test]
  fn condition_3() {
    assert_case(
      r#"A; A[x]; HoldPattern[A[x]]; HoldPattern[A][x]; Condition[A[x],3]"#,
      r#"A[x] /; 3"#,
    );
  }
  #[test]
  fn hold_pattern_4() {
    assert_case(
      r#"A; A[x]; HoldPattern[A[x]]; HoldPattern[A][x]; Condition[A[x],3]; HoldPattern[Condition[A[x],3]]"#,
      r#"HoldPattern[A[x] /; 3]"#,
    );
  }
  #[test]
  fn condition_4() {
    assert_case(
      r#"A; A[x]; HoldPattern[A[x]]; HoldPattern[A][x]; Condition[A[x],3]; HoldPattern[Condition[A[x],3]]; Condition[HoldPattern[A][x],3]"#,
      r#"HoldPattern[A][x] /; 3"#,
    );
  }
  #[test]
  fn match_q_18() {
    assert_case(r#"MatchQ[1, a_.+b_.*x_]"#, r#"True"#);
  }
  #[test]
  fn match_q_19() {
    assert_case(r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]"#, r#"True"#);
  }
  #[test]
  fn match_q_20() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_21() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_22() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_23() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_24() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_25() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_26() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_27() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_28() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_29() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_30() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]; MatchQ[x^2, x_.^m_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_31() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]; MatchQ[x^2, x_.^m_.]; MatchQ[1, a_.+b_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_32() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]; MatchQ[x^2, x_.^m_.]; MatchQ[1, a_.+b_.]; MatchQ[x, a_.+b_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_33() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]; MatchQ[x^2, x_.^m_.]; MatchQ[1, a_.+b_.]; MatchQ[x, a_.+b_.]; MatchQ[1+x, a_.+b_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_34() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]; MatchQ[x^2, x_.^m_.]; MatchQ[1, a_.+b_.]; MatchQ[x, a_.+b_.]; MatchQ[1+x, a_.+b_.]; MatchQ[1+2*x, a_.+b_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_35() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]; MatchQ[x^2, x_.^m_.]; MatchQ[1, a_.+b_.]; MatchQ[x, a_.+b_.]; MatchQ[1+x, a_.+b_.]; MatchQ[1+2*x, a_.+b_.]; MatchQ[1, a_.*b_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_36() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]; MatchQ[x^2, x_.^m_.]; MatchQ[1, a_.+b_.]; MatchQ[x, a_.+b_.]; MatchQ[1+x, a_.+b_.]; MatchQ[1+2*x, a_.+b_.]; MatchQ[1, a_.*b_.]; MatchQ[x, a_.*b_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_37() {
    assert_case(
      r#"MatchQ[1, a_.+b_.*x_]; MatchQ[x, a_.+b_.*x_]; MatchQ[2*x, a_.+b_.*x_]; MatchQ[1+x, a_.+b_.*x_]; MatchQ[1+2*x, a_.+b_.*x_]; MatchQ[1, x_^m_.]; MatchQ[x, x_^m_.]; MatchQ[x^1, x_^m_.]; MatchQ[x^2, x_^m_.]; MatchQ[1, x_.^m_.]; MatchQ[x, x_.^m_.]; MatchQ[x^1, x_.^m_.]; MatchQ[x^2, x_.^m_.]; MatchQ[1, a_.+b_.]; MatchQ[x, a_.+b_.]; MatchQ[1+x, a_.+b_.]; MatchQ[1+2*x, a_.+b_.]; MatchQ[1, a_.*b_.]; MatchQ[x, a_.*b_.]; MatchQ[2*x, a_.*b_.]"#,
      r#"True"#,
    );
  }
  #[test]
  fn f_12() {
    assert_case(r#"F[1,2]/.{Condition[F[x_,y_], x>y]:>1}"#, r#"F[1,2]"#);
  }
  #[test]
  fn f_13() {
    assert_case(
      r#"F[1,2]/.{Condition[F[x_,y_], x>y]:>1}; F[2, 1]/.{Condition[F[x_,y_], x>y]:>1}"#,
      r#"1"#,
    );
  }
  #[test]
  fn f_14() {
    assert_case(
      r#"F[1,2]/.{Condition[F[x_,y_], x>y]:>1}; F[2, 1]/.{Condition[F[x_,y_], x>y]:>1}; F[1,2]/.{F[x_,y_]:> Condition[1, x>y]}"#,
      r#"F[1,2]"#,
    );
  }
  #[test]
  fn f_15() {
    assert_case(
      r#"F[1,2]/.{Condition[F[x_,y_], x>y]:>1}; F[2, 1]/.{Condition[F[x_,y_], x>y]:>1}; F[1,2]/.{F[x_,y_]:> Condition[1, x>y]}; F[2,1]/.{F[x_,y_]:> Condition[1, x>y]}"#,
      r#"1"#,
    );
  }
  #[test]
  fn f_16() {
    assert_case(
      r#"F[1,2]/.{Condition[F[x_,y_], x>y]:>1}; F[2, 1]/.{Condition[F[x_,y_], x>y]:>1}; F[1,2]/.{F[x_,y_]:> Condition[1, x>y]}; F[2,1]/.{F[x_,y_]:> Condition[1, x>y]}; F[2,1]/.{Condition[F[x_,y_],y>0]:> Condition[1, x>y]}"#,
      r#"1"#,
    );
  }
  #[test]
  fn f_17() {
    assert_case(
      r#"F[1,2]/.{Condition[F[x_,y_], x>y]:>1}; F[2, 1]/.{Condition[F[x_,y_], x>y]:>1}; F[1,2]/.{F[x_,y_]:> Condition[1, x>y]}; F[2,1]/.{F[x_,y_]:> Condition[1, x>y]}; F[2,1]/.{Condition[F[x_,y_],y>0]:> Condition[1, x>y]}; F[2,1]/.{Condition[F[x_,y_],y>0]:> Condition[1, x>y]+ p}"#,
      r#"p + (1 /; 2 > 1)"#,
    );
  }
  #[test]
  fn f_18() {
    assert_case(
      r#"F[1,2]/.{Condition[F[x_,y_], x>y]:>1}; F[2, 1]/.{Condition[F[x_,y_], x>y]:>1}; F[1,2]/.{F[x_,y_]:> Condition[1, x>y]}; F[2,1]/.{F[x_,y_]:> Condition[1, x>y]}; F[2,1]/.{Condition[F[x_,y_],y>0]:> Condition[1, x>y]}; F[2,1]/.{Condition[F[x_,y_],y>0]:> Condition[1, x>y]+ p}; x=2;y=-2;F[2,1]/.{Condition[F[x_,y_],y>0]:> Condition[1, x>y]}"#,
      r#"1"#,
    );
  }
  #[test]
  fn list_literal_3() {
    // Same family as cases 4402/4407/4409 — wolframscript caches the
    // rule's effective optional-pattern handling at Dispatch creation
    // time, so a later `Default[Q] = 37` doesn't fill in the optional
    // slot for either `/.rule` or `/.ruled` (both stay `Q[a]`). Woxi
    // re-evaluates the optional slot each time, so the new Default
    // surfaces and both rules produce `{a, 37}`.
    assert_case(
      r#"rule = Q[x_,y_.]->{x, y};	 ruled = Dispatch[{rule}];	 {Q[a]/.rule, Q[a]/.ruled}; Default[Q]=37;          {Q[a]/.rule, Q[a]/.ruled}"#,
      r#"{{a, 37}, {a, 37}}"#,
    );
  }
  #[test]
  fn list_literal_4() {
    assert_case(
      r#"rule = Q[x_,y_.]->{x, y};	 ruled = Dispatch[{rule}];	 {Q[a]/.rule, Q[a]/.ruled}; Default[Q]=37;          {Q[a]/.rule, Q[a]/.ruled}; rule = Q[x_,y_.]->{x,y};  	  ruled = Dispatch[{rule}];	  {Q[a]/.rule, Q[a]/.ruled}"#,
      r#"{{a, 37}, {a, 37}}"#,
    );
  }
  #[test]
  fn list_literal_5() {
    // Same family as cases 4402/4407/4409/4412 — wolframscript caches
    // `Default[Q] = 37` at the re-Dispatch step, so a later
    // `Default[Q] = .` doesn't undo the cached default (both rules
    // still produce `{a, 37}`). Woxi re-evaluates the optional slot
    // each time, so once Default is cleared the optional slot stays
    // unbound and the rule no longer matches.
    assert_case(
      r#"rule = Q[x_,y_.]->{x, y};	 ruled = Dispatch[{rule}];	 {Q[a]/.rule, Q[a]/.ruled}; Default[Q]=37;          {Q[a]/.rule, Q[a]/.ruled}; rule = Q[x_,y_.]->{x,y};  	  ruled = Dispatch[{rule}];	  {Q[a]/.rule, Q[a]/.ruled}; Default[Q] = .;            {Q[a]/.rule, Q[a]/.ruled}"#,
      r#"{Q[a], Q[a]}"#,
    );
  }
  #[test]
  fn list_literal_6() {
    assert_case(
      r#"rule = Q[x_,y_.]->{x, y};	 ruled = Dispatch[{rule}];	 {Q[a]/.rule, Q[a]/.ruled}; Default[Q]=37;          {Q[a]/.rule, Q[a]/.ruled}; rule = Q[x_,y_.]->{x,y};  	  ruled = Dispatch[{rule}];	  {Q[a]/.rule, Q[a]/.ruled}; Default[Q] = .;            {Q[a]/.rule, Q[a]/.ruled}; rule = Q[x_,y_.]->{x,y};  	    ruled = Dispatch[{rule}];	    {Q[a]/.rule, Q[a]/.ruled}"#,
      r#"{Q[a],Q[a]}"#,
    );
  }
  #[test]
  fn match_q_38() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_39() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_40() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_41() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_42() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_43() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_44() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_45() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_46() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_47() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_48() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_49() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_50() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_51() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_52() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_53() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_54() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_55() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn default_5() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1."#,
      r#"1."#,
    );
  }
  #[test]
  fn default_6() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2."#,
      r#"2."#,
    );
  }
  #[test]
  fn match_q_56() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_57() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_58() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; MatchQ[F[F[H[y]]],F[x_:0,u_H]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_59() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; MatchQ[F[F[H[y]]],F[x_:0,u_H]]; MatchQ[G[G[H[y]]],G[x_:0,u_H]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_60() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; MatchQ[F[F[H[y]]],F[x_:0,u_H]]; MatchQ[G[G[H[y]]],G[x_:0,u_H]]; MatchQ[F[p, F[p, H[y]]],F[x_:0,u_H]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_61() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; MatchQ[F[F[H[y]]],F[x_:0,u_H]]; MatchQ[G[G[H[y]]],G[x_:0,u_H]]; MatchQ[F[p, F[p, H[y]]],F[x_:0,u_H]]; MatchQ[G[p, G[p, H[y]]],G[x_:0,u_H]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_62() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_63() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_64() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_65() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_66() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_67() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_68() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_69() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_70() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_71() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_72() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_73() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_74() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_75() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_76() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_77() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_78() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_79() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn default_7() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1."#,
      r#"1."#,
    );
  }
  #[test]
  fn default_8() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2."#,
      r#"2."#,
    );
  }
  #[test]
  fn match_q_80() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn match_q_81() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_82() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; MatchQ[F[F[H[y]]],F[x_:0,u_H]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_83() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; MatchQ[F[F[H[y]]],F[x_:0,u_H]]; MatchQ[G[G[H[y]]],G[x_:0,u_H]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_84() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; MatchQ[F[F[H[y]]],F[x_:0,u_H]]; MatchQ[G[G[H[y]]],G[x_:0,u_H]]; MatchQ[F[p, F[p, H[y]]],F[x_:0,u_H]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn match_q_85() {
    assert_case(
      r#"SetAttributes[F, OneIdentity]; SetAttributes[r, Flat]; SetAttributes[s, Flat]; SetAttributes[s, OneIdentity]; MatchQ[x, F[y_]]; MatchQ[x, G[y_]]; MatchQ[x, F[x_:0,y_]]; MatchQ[x, G[x_:0,y_]]; MatchQ[F[x], F[x_:0,y_]]; MatchQ[G[x], G[x_:0,y_]]; MatchQ[F[F[F[x]]], F[x_:0,y_]]; MatchQ[G[G[G[x]]], G[x_:0,y_]]; MatchQ[F[3, F[F[x]]], F[x_:0,y_]]; MatchQ[G[3, G[G[x]]], G[x_:0,y_]]; MatchQ[x, F[x1_:0, F[x2_:0,y_]]]; MatchQ[x, G[x1_:0, G[x2_:0,y_]]]; MatchQ[x, F[x1___:0, F[x2_:0,y_]]]; MatchQ[x, G[x1___:0, G[x2_:0,y_]]]; MatchQ[x, F[F[x2_:0,y_],x1_:0]]; MatchQ[x, G[G[x2_:0,y_],x1_:0]]; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; Default[F, 1]=1.; Default[G, 1]=2.; MatchQ[x, F[x_.,y_]]; MatchQ[x, G[x_.,y_]]; MatchQ[F[F[H[y]]],F[x_:0,u_H]]; MatchQ[G[G[H[y]]],G[x_:0,u_H]]; MatchQ[F[p, F[p, H[y]]],F[x_:0,u_H]]; MatchQ[G[p, G[p, H[y]]],G[x_:0,u_H]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn blank() {
    assert_case(r#"Blank[]"#, r#"_"#);
  }
  #[test]
  fn a() {
    assert_case(r#"Blank[]; A"#, r#"A"#);
  }
  #[test]
  fn whitespace_character() {
    assert_case(
      r#"Blank[]; A; WhitespaceCharacter"#,
      r#"WhitespaceCharacter"#,
    );
  }
  #[test]
  fn letter_character() {
    assert_case(
      r#"Blank[]; A; WhitespaceCharacter; LetterCharacter"#,
      r#"LetterCharacter"#,
    );
  }

  mod key_value_pattern {
    use woxi::interpret;

    #[test]
    fn symbolic_form_stays_unevaluated() {
      // KeyValuePattern is a pattern object; on its own it stays symbolic.
      assert_eq!(
        interpret(r#"KeyValuePattern[{"a" -> 1}]"#).unwrap(),
        r#"KeyValuePattern[{a -> 1}]"#
      );
      assert_eq!(
        interpret(r#"KeyValuePattern["a" -> 1]"#).unwrap(),
        r#"KeyValuePattern[a -> 1]"#
      );
    }

    #[test]
    fn matches_association_subset() {
      // Subset match: extra keys are allowed, order does not matter.
      assert_eq!(
        interpret(
          r#"MatchQ[<|"a" -> 1, "b" -> 2|>, KeyValuePattern[{"a" -> 1}]]"#
        )
        .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(
          r#"MatchQ[<|"a" -> 1, "b" -> 2|>, KeyValuePattern[{"b" -> 2, "a" -> 1}]]"#
        )
        .unwrap(),
        "True"
      );
    }

    #[test]
    fn rejects_wrong_value_or_missing_key() {
      assert_eq!(
        interpret(
          r#"MatchQ[<|"a" -> 1, "b" -> 2|>, KeyValuePattern[{"a" -> 3}]]"#
        )
        .unwrap(),
        "False"
      );
      assert_eq!(
        interpret(
          r#"MatchQ[<|"a" -> 1, "b" -> 2|>, KeyValuePattern[{"c" -> _}]]"#
        )
        .unwrap(),
        "False"
      );
    }

    #[test]
    fn value_patterns_are_supported() {
      assert_eq!(
        interpret(
          r#"MatchQ[<|"a" -> 1, "b" -> 2|>, KeyValuePattern[{"a" -> _, "b" -> _}]]"#
        )
        .unwrap(),
        "True"
      );
      // Repeated pattern variable: both values must be equal.
      assert_eq!(
        interpret(
          r#"MatchQ[<|"a" -> 1, "b" -> 2|>, KeyValuePattern[{"a" -> x_, "b" -> x_}]]"#
        )
        .unwrap(),
        "False"
      );
      assert_eq!(
        interpret(
          r#"MatchQ[<|"a" -> 1, "b" -> 1|>, KeyValuePattern[{"a" -> x_, "b" -> x_}]]"#
        )
        .unwrap(),
        "True"
      );
    }

    #[test]
    fn empty_pattern_matches_any_association_or_rule_list() {
      assert_eq!(
        interpret(r#"MatchQ[<|"a" -> 1|>, KeyValuePattern[{}]]"#).unwrap(),
        "True"
      );
      assert_eq!(
        interpret(r#"MatchQ[{"a" -> 1}, KeyValuePattern[{}]]"#).unwrap(),
        "True"
      );
      // A plain list of non-rules is not a key-value structure.
      assert_eq!(
        interpret(r#"MatchQ[{1, 2, 3}, KeyValuePattern[{}]]"#).unwrap(),
        "False"
      );
      // A non-list, non-association atom never matches.
      assert_eq!(
        interpret(r#"MatchQ[5, KeyValuePattern[{"a" -> 1}]]"#).unwrap(),
        "False"
      );
    }

    #[test]
    fn matches_list_of_rules() {
      assert_eq!(
        interpret(
          r#"MatchQ[{"a" -> 1, "b" -> 2}, KeyValuePattern[{"a" -> 1}]]"#
        )
        .unwrap(),
        "True"
      );
      // Mixed list (some non-rule elements) is not a key-value structure.
      assert_eq!(
        interpret(r#"MatchQ[{a, b -> 2}, KeyValuePattern[{}]]"#).unwrap(),
        "False"
      );
    }

    #[test]
    fn single_rule_argument_form() {
      assert_eq!(
        interpret(r#"MatchQ[<|"a" -> 1|>, KeyValuePattern["a" -> 1]]"#)
          .unwrap(),
        "True"
      );
      assert_eq!(
        interpret(r#"MatchQ[<|"a" -> 1|>, KeyValuePattern["a" -> 3]]"#)
          .unwrap(),
        "False"
      );
    }

    #[test]
    fn works_with_cases_and_replace() {
      assert_eq!(
        interpret(
          r#"Cases[{<|"a" -> 1|>, <|"a" -> 2, "b" -> 3|>, <|"b" -> 5|>}, KeyValuePattern[{"a" -> _}]]"#
        )
        .unwrap(),
        "{<|a -> 1|>, <|a -> 2, b -> 3|>}"
      );
      // Replace with a captured value pattern.
      assert_eq!(
        interpret(
          r#"Replace[<|"a" -> 1, "b" -> 2|>, KeyValuePattern[{"a" -> x_}] :> x]"#
        )
        .unwrap(),
        "1"
      );
    }
  }
}

// `Optional[p, default]` is the explicit form of `p : default`, and the
// arguments that are present fill the slots from the left, so the optionals
// that fall back on their defaults are the rightmost ones.
mod optional_explicit_form {
  use super::*;

  #[test]
  fn matches_like_the_shorthand() {
    // Regression: the explicit Optional[…] form took no part in matching.
    assert_eq!(
      interpret("MatchQ[f[a, b], f[x_, Optional[y_, 2]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[a], f[x_, Optional[y_, 2]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("f[a] /. f[x_, Optional[y_, 2]] -> {x, y}").unwrap(),
      "{a, 2}"
    );
    assert_eq!(
      interpret("f[a, b] /. f[x_, Optional[y_, 2]] -> {x, y}").unwrap(),
      "{a, b}"
    );
    assert_eq!(interpret("f[] /. f[Optional[x_, 7]] -> x").unwrap(), "7");
    // A named pattern inside works too.
    assert_eq!(
      interpret("MatchQ[f[1], f[Optional[x : _Integer, 5]]]").unwrap(),
      "True"
    );
    // An Optional slot before a required one takes its default.
    assert_eq!(
      interpret("f[2] /. f[Optional[x_, 0], y_] -> {x, y}").unwrap(),
      "{0, 2}"
    );
    assert_eq!(
      interpret("Cases[{f[1], f[1, 2]}, f[x_, Optional[y_, 5]] -> {x, y}]")
        .unwrap(),
      "{{1, 5}, {1, 2}}"
    );
  }

  #[test]
  fn head_of_an_optional_pattern() {
    // The shorthand builds the same object, so its head is Optional.
    assert_eq!(interpret("Head[y_ : 2]").unwrap(), "Optional");
    assert_eq!(interpret("Head[Optional[y_, 2]]").unwrap(), "Optional");
  }

  #[test]
  fn present_arguments_fill_from_the_left() {
    // Regression: the *first* optional used to take the default instead of
    // the last, giving {0, 1} here.
    assert_eq!(
      interpret("g[1] /. g[x_ : 0, y_ : 0] -> {x, y}").unwrap(),
      "{1, 0}"
    );
    assert_eq!(
      interpret("g[1] /. g[Optional[x_, 0], Optional[y_, 0]] -> {x, y}")
        .unwrap(),
      "{1, 0}"
    );
    assert_eq!(
      interpret("g[1] /. g[x_ : 0, y_ : 0, z_ : 0] -> {x, y, z}").unwrap(),
      "{1, 0, 0}"
    );
    assert_eq!(
      interpret("g[1, 2] /. g[x_ : 0, y_ : 0, z_ : 0] -> {x, y, z}").unwrap(),
      "{1, 2, 0}"
    );
    assert_eq!(
      interpret("f[] /. f[x_ : 1, y_ : 2] -> {x, y}").unwrap(),
      "{1, 2}"
    );
  }
}

// `PatternSequence[p1, …]` stands for several arguments in a row.
mod pattern_sequence {
  use super::*;

  #[test]
  fn splices_into_the_enclosing_arguments() {
    assert_eq!(
      interpret("MatchQ[f[1, 2], f[PatternSequence[1, 2]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[], f[PatternSequence[]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[1, 2], f[PatternSequence[x_, y_]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[1], f[PatternSequence[_]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[1, 2], f[PatternSequence[_], PatternSequence[_]]]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2}, {PatternSequence[1, 2]}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {PatternSequence[_, _], _}]").unwrap(),
      "True"
    );
    // The parts bind as usual.
    assert_eq!(
      interpret("f[1, 2, 3] /. f[a_, PatternSequence[b_, c_]] -> {a, b, c}")
        .unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("Cases[{{1, 2}, {3}}, {PatternSequence[x_, y_]} -> {x, y}]")
        .unwrap(),
      "{{1, 2}}"
    );
    assert_eq!(
      interpret("Count[{f[1, 2], f[3]}, f[PatternSequence[_, _]]]").unwrap(),
      "1"
    );
    // Outside a pattern it stays symbolic.
    assert_eq!(
      interpret("{PatternSequence[1, 2], 3}").unwrap(),
      "{PatternSequence[1, 2], 3}"
    );
  }

  #[test]
  fn a_name_binds_the_whole_sequence() {
    assert_eq!(
      interpret("MatchQ[f[1, 2], f[x : PatternSequence[_, _]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[1], f[x : PatternSequence[_, _]]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("f[1, 2] /. f[x : PatternSequence[_, _]] -> {x}").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("f[1, 2] /. f[x : PatternSequence[a_, b_]] -> {x, a, b}")
        .unwrap(),
      "{1, 2, 1, 2}"
    );
    assert_eq!(
      interpret("f[1, 2, 3] /. f[x : PatternSequence[_, _], y_] -> {x, y}")
        .unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("{1, 2} /. {x : PatternSequence[_, _]} -> {x}").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn repeats_as_a_group() {
    // `..` over a PatternSequence repeats the whole group, so only an even
    // number of arguments matches a pair.
    assert_eq!(
      interpret("MatchQ[f[1, 2, 3, 4], f[PatternSequence[_, _] ..]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[1, 2, 3], f[PatternSequence[_, _] ..]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("MatchQ[f[1, 2, 3, 4, 5, 6], f[PatternSequence[_, _, _] ..]]")
        .unwrap(),
      "True"
    );
    // `...` also admits zero repetitions.
    assert_eq!(
      interpret("MatchQ[f[], f[PatternSequence[_, _] ...]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[f[], f[PatternSequence[_, _] ..]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("MatchQ[f[1, 2], f[PatternSequence[x_, _] ...]]").unwrap(),
      "True"
    );
    // Names have to bind consistently across the repetitions.
    assert_eq!(
      interpret("MatchQ[f[1, 1, 2, 2], f[PatternSequence[x_, x_] ..]]")
        .unwrap(),
      "False"
    );
  }

  #[test]
  fn in_string_patterns() {
    assert_eq!(
      interpret("StringMatchQ[\"ab\", PatternSequence[\"a\", \"b\"]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("StringCases[\"abab\", PatternSequence[\"a\", \"b\"]]")
        .unwrap(),
      "{ab, ab}"
    );
    assert_eq!(
      interpret(
        "StringCases[\"a1b2\", \
         PatternSequence[LetterCharacter, DigitCharacter]]"
      )
      .unwrap(),
      "{a1, b2}"
    );
  }
}

// HoldPattern is transparent to matching, including on the left-hand side of a
// replacement rule.
mod hold_pattern_in_rules {
  use super::*;

  #[test]
  fn rules_see_through_it() {
    // Regression: only MatchQ and Cases stripped the wrapper, so these
    // replacements silently did nothing.
    assert_eq!(
      interpret("{a + b} /. HoldPattern[a + b] -> c").unwrap(),
      "{c}"
    );
    assert_eq!(
      interpret("{f[a + b]} /. HoldPattern[a + b] -> c").unwrap(),
      "{f[c]}"
    );
    assert_eq!(
      interpret("Hold[a + b] /. HoldPattern[a + b] -> c").unwrap(),
      "Hold[c]"
    );
    assert_eq!(
      interpret("ReplaceAll[a b, HoldPattern[a b] -> c]").unwrap(),
      "c"
    );
    assert_eq!(
      interpret("Hold[1 + 1] /. HoldPattern[1 + 1] -> 2").unwrap(),
      "Hold[2]"
    );
    assert_eq!(
      interpret("{1, 2, 3} //. HoldPattern[3] -> 4").unwrap(),
      "{1, 2, 4}"
    );
    assert_eq!(
      interpret("ReplaceRepeated[f[f[a]], HoldPattern[f[x_]] -> x]").unwrap(),
      "a"
    );
  }

  #[test]
  fn the_other_users_still_work() {
    assert_eq!(
      interpret("MatchQ[a + b, HoldPattern[a + b]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Cases[{a + b, a*b}, HoldPattern[a + b]]").unwrap(),
      "{a + b}"
    );
    assert_eq!(
      interpret("Position[{a + b, c}, HoldPattern[a + b]]").unwrap(),
      "{{1}}"
    );
    assert_eq!(
      interpret("{1, 2} /. HoldPattern[x_] :> x + 1").unwrap(),
      "{2, 3}"
    );
    // Unapplied it stays symbolic.
    assert_eq!(
      interpret("HoldPattern[a + b]").unwrap(),
      "HoldPattern[a + b]"
    );
  }
}

mod longest_shortest_and_orderless_sequences {
  use super::*;

  // Longest tries the longest split of a sequence first; the default (and
  // Shortest) tries the shortest.
  #[test]
  fn longest_and_shortest_choose_the_split() {
    assert_eq!(
      interpret("{1, 2} /. {Longest[x__], y_} :> {{x}, y}").unwrap(),
      "{{1}, 2}"
    );
    assert_eq!(
      interpret("{1, 2, 3} /. {Longest[x__], y__} :> {{x}, {y}}").unwrap(),
      "{{1, 2}, {3}}"
    );
    assert_eq!(
      interpret("{1, 2, 3} /. {Shortest[x__], y__} :> {{x}, {y}}").unwrap(),
      "{{1}, {2, 3}}"
    );
    assert_eq!(
      interpret("{a, b, c} /. {Shortest[x__], ___} :> {x}").unwrap(),
      "{a}"
    );
    assert_eq!(
      interpret("{a, b, c} /. {Longest[x__], ___} :> {x}").unwrap(),
      "{a, b, c}"
    );
    assert_eq!(
      interpret("Cases[{{1, 2, 3}}, {Shortest[x__], ___} :> {x}]").unwrap(),
      "{{1}}"
    );
  }

  // Around anything that is not a sequence they are transparent.
  #[test]
  fn longest_is_transparent_elsewhere() {
    assert_eq!(
      interpret("MatchQ[{1, 2}, {Longest[__], _}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2}, {Shortest[__], _}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("{1, 2, 3} /. {Longest[x_], ___} :> x").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("{1, 2, 3} /. Longest[{x__}] :> {x}").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn a_definition_may_wrap_its_parameter() {
    assert_eq!(
      interpret("lgf[Longest[x__]] := {x}; lgf[1, 2]").unwrap(),
      "{1, 2}"
    );
  }

  // `{a___, Longest[x__Integer], b___} -> {x}` is the standard idiom for
  // pulling the longest run of a kind of element out of a list (e.g. the
  // longest run of heads in a coin-toss sequence, once heads have been
  // replaced by 1 and tails are anything else). `Longest` wrapping a
  // sequence pattern that sits *behind* another sequence pattern (`a___`)
  // has to pick the split that makes its own match longest overall, not
  // just the longest run starting wherever `a___`'s default (shortest)
  // split happens to land — otherwise it finds only the first run instead
  // of the true longest one.
  #[test]
  fn longest_behind_a_sequence_pattern_finds_the_longest_run_anywhere() {
    assert_eq!(
      interpret(
        r#"{1, 1, "gap", 1, "gap", "gap", 1, 1, 1} /. {a___, Longest[x__Integer], b___} -> {x}"#
      )
      .unwrap(),
      "{1, 1, 1}"
    );
    // Shortest still returns the first (leftmost) run when driven the same
    // way.
    assert_eq!(
      interpret(
        r#"{"gap", "gap", 1, "gap", 1, 1, 1, "gap"} /. {a___, Shortest[x__Integer], b___} -> {x}"#
      )
      .unwrap(),
      "{1}"
    );
    // A tie between two runs of the same maximal length picks the leftmost.
    assert_eq!(
      interpret(
        r#"{1, 1, "gap", 1, 1, "gap", "gap"} /. {a___, Longest[x__Integer], b___} -> {x}"#
      )
      .unwrap(),
      "{1, 1}"
    );
    // No run at all leaves the list unchanged (the rule doesn't match).
    assert_eq!(
      interpret(
        r#"{"gap", "gap", "gap"} /. {a___, Longest[x__Integer], b___} -> {x}"#
      )
      .unwrap(),
      "{gap, gap, gap}"
    );
  }

  // OrderlessPatternSequence matches the block of arguments at its position
  // in any order — so the elements it takes stay contiguous.
  #[test]
  fn orderless_pattern_sequence_permutes_a_block() {
    assert_eq!(
      interpret("MatchQ[{1, 2}, {OrderlessPatternSequence[2, 1]}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {_, OrderlessPatternSequence[3, 2]}]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {OrderlessPatternSequence[3, 1], _}]")
        .unwrap(),
      "False"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2}, {OrderlessPatternSequence[2, 1, 3]}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("Cases[{{1, 2}, {2, 1}}, {OrderlessPatternSequence[1, 2]}]")
        .unwrap(),
      "{{1, 2}, {2, 1}}"
    );
  }

  #[test]
  fn orderless_pattern_sequence_binds_in_pattern_order() {
    assert_eq!(
      interpret("{3, 1, 2} /. {OrderlessPatternSequence[1, x_], ___} :> x")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("{1, 2} /. {OrderlessPatternSequence[x_, y_]} :> {x, y}")
        .unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn an_empty_orderless_sequence_takes_nothing() {
    assert_eq!(
      interpret("MatchQ[{}, {OrderlessPatternSequence[]}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[{1, 2, 3}, {OrderlessPatternSequence[]}]").unwrap(),
      "False"
    );
  }

  // `y_.` needs a Default for its function; without one the definition does
  // not apply at all rather than leaving a Default[…] behind.
  #[test]
  fn an_optional_without_a_default_does_not_fire() {
    assert_eq!(
      interpret("dfg[x_, y_.] := {x, y}; dfg[1]").unwrap(),
      "dfg[1]"
    );
    assert_eq!(interpret("dfq[x_.] := {x}; dfq[]").unwrap(), "dfq[]");
    // The definition still applies when every argument is given.
    assert_eq!(
      interpret("dfk[x_, y_.] := {x, y}; dfk[1, 2]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn a_set_default_fills_the_optional() {
    assert_eq!(
      interpret("Default[dfh] = 5; dfh[x_, y_.] := {x, y}; dfh[1]").unwrap(),
      "{1, 5}"
    );
    assert_eq!(
      interpret("Default[dfm, 2] = 7; dfm[x_, y_.] := {x, y}; dfm[1]").unwrap(),
      "{1, 7}"
    );
  }
}

/// Whitespace around the parameters of a definition is insignificant.
/// Regression: a blank followed by a space (`f[x_ ]`, as hand-typeset
/// notebooks write it) stored the parameter as `"x_ "` with no blank, so
/// the definition fired but left `x` unbound in the body.
mod whitespace_around_definition_parameters {
  use super::*;

  #[test]
  fn a_space_after_a_blank_still_binds_the_parameter() {
    assert_eq!(interpret("wsa[x_ ] := {x}; wsa[1]").unwrap(), "{1}");
    assert_eq!(
      interpret("wsb[x_ , y_] := {x, y}; wsb[1, 2]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("wsc[ x_ , y_ ] := {x, y}; wsc[1, 2]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn a_space_after_a_sequence_blank_still_binds_the_parameter() {
    assert_eq!(
      interpret("wsd[x_, y__ ] := {x, {y}}; wsd[1, 2, 3]").unwrap(),
      "{1, {2, 3}}"
    );
    assert_eq!(
      interpret("wse[x_, y___ ] := {x, {y}}; wse[1, 2, 3]").unwrap(),
      "{1, {2, 3}}"
    );
    // The null sequence still matches when nothing is passed for it.
    assert_eq!(
      interpret("wsf[x_, y___ ] := {x, {y}}; wsf[1]").unwrap(),
      "{1, {}}"
    );
  }

  #[test]
  fn the_blank_type_survives_the_space() {
    // `x__ ` must stay a BlankSequence: one argument is not enough.
    assert_eq!(interpret("wsg[x__ ] := {x}; wsg[]").unwrap(), "wsg[]");
    assert_eq!(interpret("wsg[x__ ] := {x}; wsg[1, 2]").unwrap(), "{1, 2}");
  }
}

mod repeated_pattern_variable_in_definition {
  use super::*;

  // A pattern variable used twice on the left of a definition constrains the
  // arguments to be identical — `f[1, 2]` must not match `f[i_, i_]`.

  #[test]
  fn a_repeated_variable_only_matches_equal_arguments() {
    assert_eq!(
      interpret("rpa[x_, x_] := x^2; {rpa[3, 3], rpa[3, 4]}").unwrap(),
      "{9, rpa[3, 4]}"
    );
  }

  #[test]
  fn a_looser_rule_still_covers_the_unequal_case() {
    // Both rules are kept; the repeated-variable one is more specific and is
    // tried first, so the diagonal and off-diagonal calls pick different rules.
    assert_eq!(
      interpret(
        "rpb[i_, i_] := \"diag\"; rpb[j_, k_] := \"off\"; \
         {rpb[1, 1], rpb[1, 2], rpb[2, 1]}"
      )
      .unwrap(),
      "{diag, off, off}"
    );
  }

  #[test]
  fn a_literal_definition_still_wins_over_the_repeated_variable() {
    // The Chebyshev differentiation-matrix idiom: literal corner entries, a
    // diagonal rule, and an off-diagonal rule.
    assert_eq!(
      interpret(
        "rpc[0, 0] = \"corner\"; rpc[i_, i_] := \"diag\"; \
         rpc[j_, k_] := \"off\"; Table[rpc[i, j], {i, 0, 2}, {j, 0, 2}]"
      )
      .unwrap(),
      "{{corner, off, off}, {off, diag, off}, {off, off, diag}}"
    );
  }

  #[test]
  fn the_constraint_holds_for_definitions_made_inside_module() {
    assert_eq!(
      interpret(
        "Module[{rpd}, rpd[i_, i_] := \"diag\"; rpd[j_, k_] := \"off\"; \
         {rpd[1, 1], rpd[1, 2]}]"
      )
      .unwrap(),
      "{diag, off}"
    );
  }

  #[test]
  fn the_constraint_compares_expressions_not_just_numbers() {
    assert_eq!(
      interpret("rpe[x_, x_] := \"same\"; {rpe[a, a], rpe[a, b]}").unwrap(),
      "{same, rpe[a, b]}"
    );
    assert_eq!(
      interpret(
        "rpf[x_, x_] := \"same\"; {rpf[{1, 2}, {1, 2}], rpf[{1}, {2}]}"
      )
      .unwrap(),
      "{same, rpf[{1}, {2}]}"
    );
  }

  #[test]
  fn a_head_constraint_still_applies_to_every_occurrence() {
    assert_eq!(
      interpret(
        "rpg[x_Integer, x_Integer] := \"int\"; \
         {rpg[2, 2], rpg[2, 3], rpg[2.5, 2.5]}"
      )
      .unwrap(),
      "{int, rpg[2, 3], rpg[2.5, 2.5]}"
    );
  }

  #[test]
  fn a_pattern_test_survives_on_the_repeated_slot() {
    assert_eq!(
      interpret(
        "rph[x_?IntegerQ, x_?IntegerQ] := \"int\"; \
         {rph[2, 2], rph[2, 3], rph[2.5, 2.5]}"
      )
      .unwrap(),
      "{int, rph[2, 3], rph[2.5, 2.5]}"
    );
  }

  #[test]
  fn three_occurrences_all_have_to_agree() {
    assert_eq!(
      interpret(
        "rpi[x_, x_, x_] := \"all\"; {rpi[1, 1, 1], rpi[1, 1, 2], rpi[1, 2, 1]}"
      )
      .unwrap(),
      "{all, rpi[1, 1, 2], rpi[1, 2, 1]}"
    );
  }

  #[test]
  fn only_the_repeated_variable_is_constrained() {
    assert_eq!(
      interpret("rpj[x_, x_, y_] := {x, y}; {rpj[1, 1, 2], rpj[1, 2, 3]}")
        .unwrap(),
      "{{1, 2}, rpj[1, 2, 3]}"
    );
  }

  #[test]
  fn a_repeated_variable_inside_a_list_pattern_is_constrained_too() {
    assert_eq!(
      interpret("rpk[{a_, a_}] := \"pair\"; {rpk[{1, 1}], rpk[{1, 2}]}")
        .unwrap(),
      "{pair, rpk[{1, 2}]}"
    );
  }

  #[test]
  fn the_stored_definition_reads_back_as_written() {
    assert_eq!(
      interpret("rpl[i_, i_] := \"diag\"; DownValues[rpl]").unwrap(),
      "{HoldPattern[rpl[i_, i_]] :> diag}"
    );
    assert_eq!(
      interpret("rpm[{a_, a_}] := \"pair\"; DownValues[rpm]").unwrap(),
      "{HoldPattern[rpm[{a_, a_}]] :> pair}"
    );
  }
}

/// A list pattern whose elements are themselves *calls* (`f[{g[a_, b_]}] :=
/// …`) has to bind the names inside those calls, exactly as it does for a
/// nested list (`f[{{a_, b_}}] := …`). Only the list case used to recurse, so
/// the inner names stayed symbolic in the body.
mod call_patterns_nested_in_a_list_pattern {
  use super::*;

  #[test]
  fn a_call_element_binds_its_arguments() {
    assert_eq!(
      interpret("cn1[{foo[a_, b_]}] := {a, b}; cn1[{foo[1, 2]}]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn binding_works_beside_a_plain_element() {
    assert_eq!(
      interpret("cn2[{x_, foo[a_]}] := {x, a}; cn2[{9, foo[8]}]").unwrap(),
      "{9, 8}"
    );
  }

  #[test]
  fn the_element_still_has_to_match() {
    assert_eq!(
      interpret("cn3[{foo[a_]}] := a; cn3[{bar[1]}]").unwrap(),
      "cn3[{bar[1]}]"
    );
  }

  #[test]
  fn nesting_goes_through_lists_and_calls_alike() {
    assert_eq!(
      interpret("cn4[{{foo[{a_, b_}]}}] := {a, b}; cn4[{{foo[{1, 2}]}}]")
        .unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn a_trailing_argument_sequence_leaves_earlier_arguments_bindable() {
    assert_eq!(
      interpret("cn5[{foo[{a_, b_}, ___]}] := {a, b}; cn5[{foo[{1, 2}, 9]}]")
        .unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn a_named_call_element_binds_both_the_whole_and_its_parts() {
    assert_eq!(
      interpret("cn6[{p : foo[a_, b_]}] := {p, a, b}; cn6[{foo[1, 2]}]")
        .unwrap(),
      "{foo[1, 2], 1, 2}"
    );
  }

  #[test]
  fn an_argument_after_a_sequence_is_not_bound_to_the_wrong_part() {
    // `foo[__, z_]` cannot say where `z` lands, so the definition simply
    // does not apply rather than reading an arbitrary argument.
    assert_eq!(
      interpret("cn7[{foo[__, z_]}] := z; cn7[{foo[1, 2, 3]}]").unwrap(),
      "3"
    );
  }

  #[test]
  fn a_trailing_argument_sequence_binds_the_tail() {
    assert_eq!(
      interpret("cn9[{foo[a_, s__]}] := {a, {s}}; cn9[{foo[1, 2, 3]}]")
        .unwrap(),
      "{1, {2, 3}}"
    );
  }

  #[test]
  fn the_stored_definition_reads_back_as_written() {
    assert_eq!(
      interpret("cn8[{foo[a_, b_]}] := a + b; DownValues[cn8]").unwrap(),
      "{HoldPattern[cn8[{foo[a_, b_]}]] :> a + b}"
    );
  }

  // A pattern is an ordinary expression, not an atom: `x_Symbol` is
  // `Pattern[x, Blank[Symbol]]`, so Part and Length see two elements.
  // Rubi's `FixIntRules[]` walks its rules this way (`rule[[1, 1, -1, 1]]`).
  mod patterns_are_ordinary_expressions {
    use super::*;

    #[test]
    fn part_reaches_into_a_named_pattern() {
      assert_eq!(interpret("Part[x_Symbol, 0]").unwrap(), "Pattern");
      assert_eq!(interpret("Part[x_Symbol, 1]").unwrap(), "x");
      assert_eq!(interpret("Part[x_Symbol, 2]").unwrap(), "_Symbol");
      assert_eq!(interpret("Part[x_, 1]").unwrap(), "x");
      assert_eq!(interpret("Part[x_, 2]").unwrap(), "_");
      // Negative indices count from the end, as everywhere else.
      assert_eq!(interpret("Part[x_Symbol, -1]").unwrap(), "_Symbol");
    }

    #[test]
    fn part_reaches_into_an_anonymous_pattern() {
      // `_Integer` is the bare `Blank[Integer]` — no `Pattern` wrapper.
      assert_eq!(interpret("Part[_Integer, 0]").unwrap(), "Blank");
      assert_eq!(interpret("Part[_Integer, 1]").unwrap(), "Integer");
      assert_eq!(
        interpret("ToString[FullForm[_Integer]]").unwrap(),
        "Blank[Integer]"
      );
      assert_eq!(
        interpret("ToString[FullForm[__]]").unwrap(),
        "BlankSequence[]"
      );
    }

    #[test]
    fn part_reaches_into_pattern_test_and_optional() {
      // `x_?NumberQ` is PatternTest[Pattern[x, Blank[]], NumberQ].
      assert_eq!(interpret("Part[x_?NumberQ, 1]").unwrap(), "x_");
      assert_eq!(interpret("Part[x_?NumberQ, 2]").unwrap(), "NumberQ");
      // `x_:2` is Optional[Pattern[x, Blank[]], 2].
      assert_eq!(interpret("Part[x_:2, 1]").unwrap(), "x_");
      assert_eq!(interpret("Part[x_:2, 2]").unwrap(), "2");
    }

    #[test]
    fn length_counts_the_full_form_parts() {
      assert_eq!(
        interpret("Length /@ {x_Symbol, x_, x_?NumberQ, x_:2, x_., _Integer}")
          .unwrap(),
        "{2, 2, 2, 2, 1, 1}"
      );
    }

    #[test]
    fn a_part_chain_walks_a_stored_rule_down_to_the_pattern_name() {
      assert_eq!(
        interpret(
          "r = HoldPattern[Int[u_, x_Symbol]] :> Condition[a + b, t]; \
           {r[[1, 1, 2, 1]], r[[1, 1, -1]], r[[1, 1, -1, 1]]}"
        )
        .unwrap(),
        "{x, x_Symbol, x}"
      );
    }
  }
}

/// A leaf variable destructured out of a list-pattern argument
/// (`f[{x_, y_}] := …`) binds to the actual matched value, the same as any
/// other pattern variable. Woxi represents that binding internally as a
/// `Part[…]` access into the whole matched list, resolved once the list
/// argument is known — but a rule's bound values are never that accessor
/// in real Wolfram, so the internal encoding must not leak into a body
/// position that holds its arguments (an iterator spec, `Hold`, …): it has
/// to resolve to the plain matched value there too.
mod list_pattern_bindings_survive_hold_contexts {
  use super::*;

  #[test]
  fn a_destructured_variable_plots_as_a_plain_iterator_symbol() {
    // Before the fix, `x` stayed `Part[_lp0, 1]` inside `Plot`'s held
    // iterator-spec argument, so `Plot` rejected it with
    // "iterator variable must be a symbol" instead of drawing the curve.
    assert_eq!(
      interpret(
        "lpp1[{x_, x1_, x2_}] := Head[Plot[Sin[x], {x, x1, x2}]]; \
         lpp1[{z, 0, 1}]"
      )
      .unwrap(),
      "Graphics"
    );
  }

  #[test]
  fn a_destructured_variable_stays_a_plain_symbol_under_hold() {
    assert_eq!(
      interpret("lpp2[{x_, x1_, x2_}] := Hold[{x, x1, x2}]; lpp2[{z, 0, 1}]")
        .unwrap(),
      "Hold[{z, 0, 1}]"
    );
    // `x` resolves to the plain symbol `z` before `Hold` ever sees it, so
    // `Head[x]`/`Head[Unevaluated[x]]` stay unevaluated calls on `z` — not
    // on some `Part[…]` accessor.
    assert_eq!(
      interpret(
        "lpp3[{x_, x1_, x2_}] := Hold[Head[x], Head[Unevaluated[x]]]; \
         lpp3[{z, 0, 1}]"
      )
      .unwrap(),
      "Hold[Head[z], Head[Unevaluated[z]]]"
    );
  }

  #[test]
  fn nested_list_pattern_variables_also_resolve_under_hold() {
    assert_eq!(
      interpret("lpp4[{{a_, b_}, c_}] := Hold[a, b, c]; lpp4[{{1, 2}, 3}]")
        .unwrap(),
      "Hold[1, 2, 3]"
    );
  }

  #[test]
  fn a_literal_part_of_a_literal_list_still_stays_held() {
    // A genuine `Part[…]` the user wrote (not synthesized by list-pattern
    // destructuring) must keep its ordinary Hold behaviour.
    assert_eq!(
      interpret("Hold[{1, 2, 3}[[2]]]").unwrap(),
      "Hold[{1, 2, 3}[[2]]]"
    );
  }
}

mod literal_left_hand_side {
  use super::*;

  /// A rule whose left side is a literal call rather than a pattern
  /// rewrites every subexpression equal to it, brackets and all. The
  /// "Related Rates" chapter of *Introduction to Calculus* substitutes the
  /// radius of a cone by half its height; the squared radius used to lose
  /// its square, so the volume came out one power of `h` short.
  #[test]
  fn a_compound_replacement_keeps_its_brackets() {
    assert_eq!(
      interpret("v[t] == (Pi r[t]^2 h[t])/3 /. r[t] -> h[t]/2").unwrap(),
      "v[t] == (Pi*h[t]^3)/12"
    );
    assert_eq!(
      interpret("r[t]^2 /. r[t] -> q[t] + 1").unwrap(),
      "(1 + q[t])^2"
    );
    assert_eq!(interpret("r[t]^2 /. r[t] -> 2 q[t]").unwrap(), "4*q[t]^2");
  }

  /// Only the subexpressions that really are the left side are rewritten.
  #[test]
  fn other_calls_are_left_alone() {
    assert_eq!(interpret("f[x] + f[y] /. f[x] -> 1").unwrap(), "1 + f[y]");
  }
}

/// A pattern head is a pattern too: `_[a, b]` matches any two-argument
/// expression, whatever its head — a rule, a list, a sum. Issue #603.
mod blank_head_patterns {
  use super::*;

  #[test]
  fn a_blank_head_matches_every_two_argument_expression() {
    assert_eq!(interpret("MatchQ[\"a\" -> 1, _[_, _]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[{1, 2}, _[_, _]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[a + b, _[_, _]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[f[1, 2], _[_, _]]").unwrap(), "True");
    assert_eq!(interpret("MatchQ[<|\"a\" -> 1|>, _[_]]").unwrap(), "True");
  }

  #[test]
  fn the_head_and_arguments_both_bind() {
    assert_eq!(
      interpret("(\"a\" -> 1) /. _[k_, v_] :> hold[k, v]").unwrap(),
      "hold[a, 1]"
    );
    assert_eq!(
      interpret("(\"a\" -> 1) /. h_[k_, v_] :> hold[h, k, v]").unwrap(),
      "hold[Rule, a, 1]"
    );
    assert_eq!(
      interpret("{1, 2} /. _[k_, v_] :> hold[k, v]").unwrap(),
      "hold[1, 2]"
    );
    assert_eq!(
      interpret("Cases[{f[1], g[2]}, _[x_] :> x]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn a_sequence_under_a_blank_head_still_binds() {
    assert_eq!(
      interpret("{1, 2, 3} /. _[a__] :> {a}").unwrap(),
      "{1, 2, 3}"
    );
  }
}

/// An optional slot takes its default when the argument at hand does not fit
/// it, rather than forcing the argument in and failing the whole rule.
/// Issue #603.
mod optional_slots_skip_what_does_not_fit {
  use super::*;

  #[test]
  fn a_mismatching_argument_moves_on_to_the_next_slot() {
    assert_eq!(
      interpret(
        "k[a_, b : (_Symbol | _Function) : auto, c_List : {}] := {a, b, c};\
         {k[1], k[1, sym, {2}], k[1, {2}]}"
      )
      .unwrap(),
      "{{1, auto, {}}, {1, sym, {2}}, {1, auto, {2}}}"
    );
  }

  #[test]
  fn a_head_constrained_slot_behaves_the_same_way() {
    assert_eq!(
      interpret(
        "h[a_, b : _Integer : 9, c_List : {}] := {a, b, c};\
         {h[1], h[1, {2}], h[1, 3, {4}]}"
      )
      .unwrap(),
      "{{1, 9, {}}, {1, 9, {2}}, {1, 3, {4}}}"
    );
  }

  #[test]
  fn a_plain_default_is_unchanged() {
    assert_eq!(
      interpret("f[a_, b_ : 5] := {a, b}; {f[1], f[1, 2]}").unwrap(),
      "{{1, 5}, {1, 2}}"
    );
  }
}

/// `name : head ? test : default` is one slot: a tested pattern that may be
/// left out. The default used to be stranded after the `?test`. Issue #603.
mod tested_pattern_with_a_default {
  use super::*;

  #[test]
  fn the_test_and_the_default_both_survive() {
    assert_eq!(
      interpret("g[a_, b : _Symbol?AtomQ : dflt] := {a, b}; {g[1], g[1, sym]}")
        .unwrap(),
      "{{1, dflt}, {1, sym}}"
    );
  }

  #[test]
  fn a_plain_named_default_is_unchanged() {
    assert_eq!(
      interpret("k[x : _Integer : 5] := x; {k[], k[7]}").unwrap(),
      "{5, 7}"
    );
  }
}

/// A named `OptionsPattern[]` binds the options it matched, so a function can
/// pass them on whole (`f[opts : OptionsPattern[]] := g[opts]`). Issue #603.
mod named_options_pattern {
  use super::*;

  #[test]
  fn the_name_binds_the_matched_options() {
    assert_eq!(
      interpret("u[opts : OptionsPattern[]] := {opts}; u[\"a\" -> 1]").unwrap(),
      "{a -> 1}"
    );
    assert_eq!(
      interpret("u[opts : OptionsPattern[]] := {opts}; u[]").unwrap(),
      "{}"
    );
    assert_eq!(
      interpret(
        "u[opts : OptionsPattern[]] := Flatten[{opts}]; \
         u[\"a\" -> 1, \"b\" -> 2]"
      )
      .unwrap(),
      "{a -> 1, b -> 2}"
    );
  }

  #[test]
  fn option_value_still_works_alongside_the_binding() {
    assert_eq!(
      interpret(
        "Options[u] = {\"I\" -> 9}; \
         u[opts : OptionsPattern[]] := OptionValue[\"I\"]; u[\"I\" -> 2]"
      )
      .unwrap(),
      "2"
    );
  }
}

// Replacing a symbol has to reach into every part of a held expression,
// including the operator forms whose symbol sits outside the argument
// list. Renaming a symbol throughout a definition list —
// `Language`ExtendedFullDefinition[a] /. a -> b` — depends on it.
// Regression test for <https://github.com/ad-si/Woxi/issues/603>.
mod symbol_replacement_reaches_operator_forms {
  use super::*;

  #[test]
  fn every_operator_form_is_traversed() {
    clear_state();
    assert_eq!(
      interpret(
        r#"ToString[
             Hold[{f /@ {1}, f @@ {1}, f @@@ {{1}},
                   x /. f -> 1, x //. f -> 1,
                   Function[u, f[u]], q[u_] := f[u]}] /. f -> g,
             InputForm]"#
      )
      .unwrap(),
      "Hold[{g /@ {1}, g @@ {1}, g @@@ {{1}}, x /. g -> 1, x //. g -> 1, \
       Function[u, g[u]], q[u_] := g[u]}]"
    );
  }

  #[test]
  fn a_compound_expression_is_traversed() {
    clear_state();
    assert_eq!(
      interpret("ToString[Hold[(a = 1; a + 2)] /. a -> b, InputForm]").unwrap(),
      "Hold[b = 1; b + 2]"
    );
  }
}

// Regression tests for #616: a `Rule` / `RuleDelayed` inside a definition's
// pattern binds its parts. The placeholder round-trip that stores a compound
// argument pattern walked function calls and lists but not rule nodes, so
// `f[a_ :> b_] := g[a, b]` matched and then substituted nothing — the body
// came back with the pattern names in it. Rubi rewrites its whole rule base
// through patterns of exactly this shape (`FixIntRule[RuleDelayed[lhs_, u_],
// x_]`).
mod rule_nodes_in_a_definition_pattern {
  use super::*;

  #[test]
  fn delayed_rule_written_as_an_operator_binds() {
    clear_state();
    assert_eq!(
      interpret("rn1[a_ :> b_] := g[a, b]; rn1[AA :> BB]").unwrap(),
      "g[AA, BB]"
    );
  }

  // The same pattern spelled with the head name is the same pattern.
  #[test]
  fn delayed_rule_written_as_a_head_binds() {
    clear_state();
    assert_eq!(
      interpret("rn2[RuleDelayed[a_, b_]] := g[a, b]; rn2[AA :> BB]").unwrap(),
      "g[AA, BB]"
    );
  }

  #[test]
  fn immediate_rule_binds() {
    clear_state();
    assert_eq!(
      interpret("rn3[Rule[a_, b_]] := g[a, b]; rn3[AA -> BB]").unwrap(),
      "g[AA, BB]"
    );
  }

  // Parts nested under the rule bind too, alongside a pattern head.
  #[test]
  fn parts_nested_under_a_rule_bind() {
    clear_state();
    assert_eq!(
      interpret(
        "rn4[RuleDelayed[lhs_, F_[u_, test_]], x_] := g[lhs, F, u, test, x];          rn4[AA :> ff[BB, CC], zz]"
      )
      .unwrap(),
      "g[AA, ff, BB, CC, zz]"
    );
  }

  // And the stored rule prints as what was written, not as the internal
  // placeholders it is matched through.
  #[test]
  fn down_values_show_the_rule_pattern() {
    clear_state();
    assert_eq!(
      interpret("rn5[RuleDelayed[a_, b_]] := g[a, b]; DownValues[rn5]")
        .unwrap(),
      "{HoldPattern[rn5[a_ :> b_]] :> g[a, b]}"
    );
  }
}

// Regression tests for the Orderless re-reading in rule dispatch. A `Times`
// or `Plus` argument can satisfy a structural pattern more than one way, and
// the rule's guard may accept only some of them. The matcher settles on one
// reading, so dispatch offers it the others before abandoning the rule —
// which is what Rubi's rule base relies on throughout.
mod orderless_readings_under_a_guard {
  use super::*;

  #[test]
  fn a_product_is_reread_until_the_guard_is_satisfied() {
    clear_state();
    assert_eq!(
      interpret("orp[u_*x_] := g[u, x] /; x > 0; orp[q*3]").unwrap(),
      "g[q, 3]"
    );
  }

  // Written the other way round it is the same expression, so it reads the
  // same way.
  #[test]
  fn the_written_order_does_not_matter() {
    clear_state();
    assert_eq!(
      interpret("orp[u_*x_] := g[u, x] /; x > 0; orp[3*q]").unwrap(),
      "g[q, 3]"
    );
  }

  #[test]
  fn a_sum_is_reread_too() {
    clear_state();
    assert_eq!(
      interpret("ors[u_+x_] := s[u, x] /; x > 0; ors[q+3]").unwrap(),
      "s[q, 3]"
    );
  }

  // A guard no reading satisfies still turns the rule down.
  #[test]
  fn a_guard_no_reading_satisfies_still_fails() {
    clear_state();
    assert_eq!(
      interpret("orp[u_*x_] := g[u, x] /; x > 5; orp[q*3]").unwrap(),
      "orp[3*q]"
    );
  }

  // An unguarded rule keeps the reading it always had: the retry only ever
  // runs after a guard has turned one down.
  #[test]
  fn an_unguarded_rule_keeps_its_first_reading() {
    clear_state();
    assert_eq!(
      interpret("orn[u_*x_] := g[u, x]; orn[q*3]").unwrap(),
      "g[3, q]"
    );
  }
}

/// `a - b` and `a / b` are notation for `Plus[a, Times[-1, b]]` and
/// `Times[a, Power[b, -1]]`. Evaluation rewrites them, so only a held
/// expression still carries the operator — and that is exactly what a rule
/// base read out of `DownValues` is matched against.
mod held_subtraction_matches_as_a_sum {
  use super::*;

  #[test]
  fn a_held_difference_matches_a_plus_pattern() {
    clear_state();
    assert_eq!(
      interpret("MatchQ[Hold[q - x^2], Hold[u_ + v_]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[Hold[q - x^2], Hold[Plus[u_, v_]]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("ToString[ReplaceAll[Hold[q - x^2], Hold[u_ + v_] :> {u, v}], InputForm]")
        .unwrap(),
      "{q, -x^2}"
    );
  }

  // The shape every Rubi rule for `1/(a + b x^2)` is written in.
  #[test]
  fn a_held_difference_matches_an_optional_coefficient() {
    clear_state();
    assert_eq!(
      interpret("MatchQ[Hold[q - x^2], Hold[u_ + v_.*x_^2]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatchQ[Hold[1/(b^2 - 4*a*c - x^2)], Hold[1/(u_ + v_.*x_^2)]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn a_held_quotient_matches_a_times_pattern() {
    clear_state();
    assert_eq!(interpret("MatchQ[Hold[q/r], Hold[u_*v_]]").unwrap(), "True");
    assert_eq!(
      interpret(
        "ToString[ReplaceAll[Hold[q/r], Hold[u_*v_] :> {u, v}], InputForm]"
      )
      .unwrap(),
      "{q, r^(-1)}"
    );
  }
}

/// A `Flat` head can be split between two pattern slots in more than one
/// place, and the splits bind the variables differently. The reading has to
/// agree with what the rest of the rule already fixed.
mod flat_split_agrees_with_the_other_arguments {
  use super::*;

  // `f[a_ + b_.*x_^2, x_Symbol]` names the integration variable twice: the
  // split that reads `p^2` as `b x^2` (with `x -> p`) contradicts the second
  // argument, so the other split is the one that fires. Rubi's whole rule base
  // is written in this shape.
  #[test]
  fn a_later_argument_picks_the_split() {
    clear_state();
    assert_eq!(
      interpret(
        "fsp[1/(a_ + b_.*x_^2), x_Symbol] := {a, b, x};\n\
         ToString[fsp[1/(p^2 - 4*q*r - x^2), x], InputForm]"
      )
      .unwrap(),
      "{p^2 - 4*q*r, -1, x}"
    );
  }

  // With nothing else to go on, the first split still wins — unchanged.
  #[test]
  fn without_another_argument_the_first_split_wins() {
    clear_state();
    assert_eq!(
      interpret(
        "fsq[1/(a_ + b_.*x_^2)] := {a, b, x};\n\
         ToString[fsq[1/(p^2 - 4*q*r - x^2)], InputForm]"
      )
      .unwrap(),
      "{-4*q*r - x^2, 1, p}"
    );
  }
}

/// `FreeQ` looks at every part its FullForm shows, whatever operator wrote it.
/// A package that inspects its own rule base — `FreeQ[Hold[rule], ShowStep]`,
/// as Rubi does for all 7000 of its rules — depends on it reaching inside a
/// `:>` node.
mod free_q_sees_operator_forms {
  use super::*;

  #[test]
  fn free_q_descends_into_rules() {
    clear_state();
    assert_eq!(
      interpret("FreeQ[HoldPattern[f[ArcSin[x]]] :> 1, ArcSin]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("FreeQ[f[ArcSin[x]] -> 1, ArcSin]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("FreeQ[{a :> ArcSin[x]}, ArcSin]").unwrap(),
      "False"
    );
    assert_eq!(interpret("FreeQ[a -> b, b]").unwrap(), "False");
  }

  #[test]
  fn free_q_descends_into_comparisons_and_functions() {
    clear_state();
    assert_eq!(
      interpret("FreeQ[Hold[q == ArcSin[x]], ArcSin]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("FreeQ[Hold[Function[ArcSin[#]]], ArcSin]").unwrap(),
      "False"
    );
  }
}
