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
