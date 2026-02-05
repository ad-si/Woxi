use woxi::interpret;

mod interpreter_tests {
  use super::*;

  mod arithmetic {
    use super::*;

    mod integer {
      use super::*;

      #[test]
      fn addition() {
        assert_eq!(interpret("1 + 2").unwrap(), "3");
        assert_eq!(interpret("1 + 2 + 3").unwrap(), "6");
        assert_eq!(interpret("(1 + 2) + 3").unwrap(), "6");
        assert_eq!(interpret("1 + (2 + 3)").unwrap(), "6");
        assert_eq!(interpret("(1 + 2 + 3)").unwrap(), "6");
      }

      #[test]
      fn subtraction() {
        assert_eq!(interpret("3 - 1").unwrap(), "2");
        assert_eq!(interpret("7 - 3 - 1").unwrap(), "3");
      }

      #[test]
      fn multiple_operations() {
        assert_eq!(interpret("1 + 2 - 3 + 4").unwrap(), "4");
      }

      #[test]
      fn negative_numbers() {
        assert_eq!(interpret("-1 + 3").unwrap(), "2");
      }

      #[test]
      fn multiplication() {
        assert_eq!(interpret("3 * 4").unwrap(), "12");
      }

      #[test]
      fn complex_multiplication() {
        assert_eq!(interpret("2 * 3 + 4 * 5").unwrap(), "26");
      }

      #[test]
      fn division() {
        assert_eq!(interpret("10 / 2").unwrap(), "5");
      }

      #[test]
      fn division_repeating_decimal() {
        // TODO: Should be kept as the fraction 10/3
        assert_eq!(interpret("10 / 3").unwrap(), "3.3333333333");
      }

      #[test]
      fn complex_division() {
        assert_eq!(interpret("10 / 2 + 3 / 3").unwrap(), "6");
      }
    }

    mod float {
      use super::*;

      #[test]
      fn addition() {
        assert_eq!(interpret("1.5 + 2.7").unwrap(), "4.2");
      }

      #[test]
      fn subtraction() {
        assert_eq!(interpret("3.5 - 1.2").unwrap(), "2.3");
      }

      #[test]
      fn multiple_operations() {
        assert_eq!(interpret("1.1 + 2.2 - 3.3 + 4.4").unwrap(), "4.4");
      }

      #[test]
      fn multiplication() {
        assert_eq!(interpret("1.5 * 2.5").unwrap(), "3.75");
      }

      #[test]
      fn complex_multiplication() {
        assert_eq!(interpret("1.5 * 2.0 + 3.0 * 1.5").unwrap(), "7.5");
      }

      #[test]
      fn division() {
        // TODO: Should be 3.2
        assert_eq!(interpret("9.6 / 3").unwrap(), "3.2");
      }

      #[test]
      fn complex_division() {
        // TODO: Should be 4.2
        assert_eq!(interpret("9.6 / 3 + 3.0 / 3").unwrap(), "4.2");
      }
    }
  }

  mod errors {
    use super::*;

    #[test]
    fn invalid_input() {
      match interpret("1 + ") {
        Err(woxi::InterpreterError::ParseError(_)) => (),
        _ => panic!("Expected a ParseError"),
      }
    }
  }

  mod postfix_application {
    use super::*;

    #[test]
    fn postfix_with_identifier() {
      // x // f is equivalent to f[x]
      assert_eq!(interpret("4 // Sqrt").unwrap(), "2");
      assert_eq!(interpret("16 // Sqrt").unwrap(), "4");
    }

    #[test]
    fn postfix_with_list() {
      assert_eq!(interpret("{1, 2, 3} // Length").unwrap(), "3");
      assert_eq!(interpret("{1, 2, 3} // First").unwrap(), "1");
      assert_eq!(interpret("{1, 2, 3} // Last").unwrap(), "3");
    }

    #[test]
    fn postfix_with_function_call() {
      // x // Map[f] is equivalent to Map[f][x] which is Map[f, x]
      assert_eq!(interpret("{1, 4, 9} // Map[Sqrt]").unwrap(), "{1, 2, 3}");
    }

    #[test]
    fn chained_postfix() {
      // x // f // g is equivalent to g[f[x]]
      assert_eq!(interpret("16 // Sqrt // Sqrt").unwrap(), "2");
    }

    #[test]
    fn postfix_after_replace_all() {
      // (expr /. rules) // f
      assert_eq!(
        interpret("{1, 2, 3} /. x_ /; x > 1 :> 0 // Length").unwrap(),
        "3"
      );
    }
  }

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
  }

  mod divisible {
    use super::*;

    #[test]
    fn divisible_true() {
      assert_eq!(interpret("Divisible[10, 2]").unwrap(), "True");
      assert_eq!(interpret("Divisible[15, 3]").unwrap(), "True");
      assert_eq!(interpret("Divisible[15, 5]").unwrap(), "True");
      assert_eq!(interpret("Divisible[100, 10]").unwrap(), "True");
    }

    #[test]
    fn divisible_false() {
      assert_eq!(interpret("Divisible[10, 3]").unwrap(), "False");
      assert_eq!(interpret("Divisible[7, 2]").unwrap(), "False");
      assert_eq!(interpret("Divisible[11, 5]").unwrap(), "False");
    }

    #[test]
    fn divisible_by_one() {
      assert_eq!(interpret("Divisible[5, 1]").unwrap(), "True");
      assert_eq!(interpret("Divisible[0, 1]").unwrap(), "True");
    }

    #[test]
    fn divisible_zero() {
      assert_eq!(interpret("Divisible[0, 5]").unwrap(), "True");
    }

    #[test]
    fn divisible_non_integer() {
      assert_eq!(interpret("Divisible[5.5, 2]").unwrap(), "False");
      assert_eq!(interpret("Divisible[10, 2.5]").unwrap(), "False");
    }

    #[test]
    fn divisible_division_by_zero() {
      assert!(interpret("Divisible[5, 0]").is_err());
    }
  }

  mod paren_anonymous_function {
    use super::*;

    #[test]
    fn paren_anonymous_with_comparison() {
      // (# === "")& is an anonymous function testing for empty string
      // Uses postfix @ operator since direct call syntax is not supported
      assert_eq!(interpret("(# === \"\")& @ \"hello\"").unwrap(), "False");
      assert_eq!(interpret("(# === \"\")& @ \"\"").unwrap(), "True");
    }

    #[test]
    fn paren_anonymous_with_arithmetic() {
      assert_eq!(interpret("(# + 1)& @ 5").unwrap(), "6");
      assert_eq!(interpret("(# * 2 + 3)& @ 4").unwrap(), "11");
    }

    #[test]
    fn paren_anonymous_in_map() {
      assert_eq!(interpret("Map[(# + 1)&, {1, 2, 3}]").unwrap(), "{2, 3, 4}");
    }

    #[test]
    fn paren_anonymous_with_if() {
      assert_eq!(interpret("(If[# > 0, #, 0])& @ 5").unwrap(), "5");
      assert_eq!(interpret("(If[# > 0, #, 0])& @ -3").unwrap(), "0");
    }

    #[test]
    fn paren_anonymous_in_postfix() {
      // If[# === "", i, #]& @ "hello" should return "hello" (strings displayed without quotes)
      assert_eq!(
        interpret("(If[# === \"\", \"empty\", #])& @ \"hello\"").unwrap(),
        "hello"
      );
      assert_eq!(
        interpret("(If[# === \"\", \"empty\", #])& @ \"\"").unwrap(),
        "empty"
      );
    }
  }

  mod table_with_list_iterator {
    use super::*;

    #[test]
    fn table_iterate_over_list() {
      // Table[expr, {x, {a, b, c}}] iterates x over the list elements
      assert_eq!(
        interpret("Table[x^2, {x, {1, 2, 3}}]").unwrap(),
        "{1, 4, 9}"
      );
    }

    #[test]
    fn table_iterate_over_nested_list() {
      // Iterate over list of pairs
      assert_eq!(
        interpret("Table[First[pair], {pair, {{1, 2}, {3, 4}, {5, 6}}}]")
          .unwrap(),
        "{1, 3, 5}"
      );
    }

    #[test]
    fn table_iterate_over_strings() {
      assert_eq!(
        interpret("Table[StringLength[s], {s, {\"a\", \"bb\", \"ccc\"}}]")
          .unwrap(),
        "{1, 2, 3}"
      );
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
    fn string_join_with_table_result() {
      // StringJoin with a Table that returns strings
      assert_eq!(
        interpret("StringJoin[Table[\"x\", {i, 3}]]").unwrap(),
        "xxx"
      );
    }
  }

  mod postfix_with_anonymous_function {
    use super::*;

    #[test]
    fn postfix_at_with_simple_anonymous() {
      assert_eq!(interpret("#^2& @ 3").unwrap(), "9");
      assert_eq!(interpret("#+1& @ 5").unwrap(), "6");
    }

    #[test]
    fn postfix_at_with_function_anonymous() {
      assert_eq!(interpret("Sqrt[#]& @ 16").unwrap(), "4");
    }

    #[test]
    fn postfix_at_with_string_result() {
      // Anonymous function that returns a string (strings displayed without quotes)
      assert_eq!(
        interpret("If[# > 0, \"positive\", \"non-positive\"]& @ 5").unwrap(),
        "positive"
      );
      assert_eq!(
        interpret("If[# > 0, \"positive\", \"non-positive\"]& @ -3").unwrap(),
        "non-positive"
      );
    }

    #[test]
    fn postfix_at_preserves_string_arg() {
      // When the argument is a string, it should be preserved
      assert_eq!(interpret("StringLength[#]& @ \"hello\"").unwrap(), "5");
    }
  }

  mod user_defined_functions {
    use super::*;

    #[test]
    fn function_with_multiple_calls() {
      // Regression test: ensure function calls with different arguments
      // return different results (not cached incorrectly)
      assert_eq!(
        interpret("f[a_, b_, c_] := {a, b, c}; f[1, 2, 3]").unwrap(),
        "{1, 2, 3}"
      );
    }

    #[test]
    fn function_calls_are_not_incorrectly_cached() {
      // This test ensures that consecutive calls to a user-defined function
      // with different arguments return correct results
      assert_eq!(
        interpret(
          "g[a_, b_, c_] := a + b + c; x = g[1, 2, 3]; y = g[4, 5, 6]; {x, y}"
        )
        .unwrap(),
        "{6, 15}"
      );
    }

    #[test]
    fn map_apply_with_user_function() {
      // Test @@@ with user-defined function
      assert_eq!(
        interpret("f[a_, b_, c_] := a + b + c; f @@@ {{1, 2, 3}, {4, 5, 6}}")
          .unwrap(),
        "{6, 15}"
      );
    }

    #[test]
    fn user_function_with_if_lazy_evaluation() {
      // Test that If only evaluates the selected branch in user-defined functions
      // If both branches were evaluated, First[{}] would error
      assert_eq!(
        interpret("f[x_] := If[x > 0, 1, First[{}]]; f[5]").unwrap(),
        "1"
      );
      assert_eq!(
        interpret("f[x_] := If[x > 0, First[{}], 2]; f[-5]").unwrap(),
        "2"
      );
    }
  }

  mod part_extraction {
    use super::*;

    #[test]
    fn nested_list_part_via_variable() {
      // Test that Part extraction works correctly for nested lists stored in variables
      assert_eq!(interpret("x = {{a, b}, {c, d}}; x[[1]]").unwrap(), "{a, b}");
      assert_eq!(interpret("x = {{a, b}, {c, d}}; x[[2]]").unwrap(), "{c, d}");
    }

    #[test]
    fn deeply_nested_list_part() {
      assert_eq!(
        interpret("x = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}; x[[1]]").unwrap(),
        "{{1, 2}, {3, 4}}"
      );
    }
  }

  mod length_function {
    use super::*;

    #[test]
    fn length_with_variable() {
      // Test that Length works with lists stored in variables
      assert_eq!(interpret("x = {1, 2, 3}; Length[x]").unwrap(), "3");
      assert_eq!(interpret("x = {}; Length[x]").unwrap(), "0");
    }

    #[test]
    fn length_with_nested_list_variable() {
      assert_eq!(
        interpret("x = {{a, b}, {c, d}, {e, f}}; Length[x]").unwrap(),
        "3"
      );
    }
  }
}
