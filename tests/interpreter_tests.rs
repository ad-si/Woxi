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
        // Wolfram keeps this as a fraction 10/3
        assert_eq!(interpret("10 / 3").unwrap(), "10/3");
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
        // ToString[9.6 / 3] in Wolfram returns 3.2
        assert_eq!(interpret("9.6 / 3").unwrap(), "3.2");
      }

      #[test]
      fn complex_division() {
        // ToString[9.6 / 3 + 3.0 / 3] in Wolfram returns 4.2
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
      // Wolfram returns unevaluated for non-exact numbers
      assert_eq!(interpret("Divisible[5.5, 2]").unwrap(), "Divisible[5.5, 2]");
      assert_eq!(
        interpret("Divisible[10, 2.5]").unwrap(),
        "Divisible[10, 2.5]"
      );
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

  mod full_form {
    use super::*;

    #[test]
    fn full_form_plus() {
      assert_eq!(interpret("FullForm[x + y + z]").unwrap(), "Plus[x, y, z]");
    }

    #[test]
    fn full_form_times() {
      assert_eq!(interpret("FullForm[x y z]").unwrap(), "Times[x, y, z]");
    }

    #[test]
    fn full_form_list() {
      assert_eq!(interpret("FullForm[{1, 2, 3}]").unwrap(), "List[1, 2, 3]");
    }

    #[test]
    fn full_form_power() {
      assert_eq!(interpret("FullForm[x^2]").unwrap(), "Power[x, 2]");
    }

    #[test]
    fn full_form_complex() {
      assert_eq!(
        interpret("FullForm[a b + c]").unwrap(),
        "Plus[Times[a, b], c]"
      );
    }
  }

  mod construct {
    use super::*;

    #[test]
    fn construct_basic() {
      assert_eq!(interpret("Construct[f, a, b, c]").unwrap(), "f[a, b, c]");
    }

    #[test]
    fn construct_single_arg() {
      assert_eq!(interpret("Construct[f, a]").unwrap(), "f[a]");
    }

    #[test]
    fn construct_with_fold() {
      assert_eq!(
        interpret("Fold[Construct, f, {a, b, c}]").unwrap(),
        "f[a][b][c]"
      );
    }
  }

  mod association_ast {
    use super::*;

    #[test]
    fn keys_basic() {
      assert_eq!(
        interpret("Keys[<|\"a\" -> 1, \"b\" -> 2|>]").unwrap(),
        "{a, b}"
      );
    }

    #[test]
    fn values_basic() {
      assert_eq!(
        interpret("Values[<|\"a\" -> 1, \"b\" -> 2|>]").unwrap(),
        "{1, 2}"
      );
    }

    #[test]
    fn keys_with_variable() {
      assert_eq!(
        interpret("h = <|\"x\" -> 10, \"y\" -> 20|>; Keys[h]").unwrap(),
        "{x, y}"
      );
    }

    #[test]
    fn values_with_variable() {
      assert_eq!(
        interpret("h = <|\"x\" -> 10, \"y\" -> 20|>; Values[h]").unwrap(),
        "{10, 20}"
      );
    }

    #[test]
    fn key_exists_q_true() {
      assert_eq!(
        interpret("h = <|\"a\" -> 1|>; KeyExistsQ[h, \"a\"]").unwrap(),
        "True"
      );
    }

    #[test]
    fn key_exists_q_false() {
      assert_eq!(
        interpret("h = <|\"a\" -> 1|>; KeyExistsQ[h, \"b\"]").unwrap(),
        "False"
      );
    }

    #[test]
    fn key_drop_from() {
      assert_eq!(
        interpret("KeyDropFrom[<|\"a\" -> 1, \"b\" -> 2|>, \"a\"]").unwrap(),
        "<|b -> 2|>"
      );
    }

    #[test]
    fn part_extraction() {
      assert_eq!(
        interpret("h = <|\"Green\" -> 2, \"Red\" -> 1|>; h[[\"Green\"]]")
          .unwrap(),
        "2"
      );
    }

    #[test]
    fn map_over_association() {
      assert_eq!(
        interpret("Map[#^2&, <|\"a\" -> 2, \"b\" -> 3|>]").unwrap(),
        "<|a -> 4, b -> 9|>"
      );
    }
  }

  mod list_threading {
    use super::*;

    #[test]
    fn list_plus_scalar() {
      assert_eq!(interpret("{1, 2, 3} + 10").unwrap(), "{11, 12, 13}");
    }

    #[test]
    fn scalar_plus_list() {
      assert_eq!(interpret("10 + {1, 2, 3}").unwrap(), "{11, 12, 13}");
    }

    #[test]
    fn list_plus_list() {
      assert_eq!(
        interpret("{1, 2, 3} + {10, 20, 30}").unwrap(),
        "{11, 22, 33}"
      );
    }

    #[test]
    fn list_times_scalar() {
      assert_eq!(interpret("{1, 2, 3} * 2").unwrap(), "{2, 4, 6}");
    }

    #[test]
    fn list_power_scalar() {
      assert_eq!(interpret("{1, 2, 3}^2").unwrap(), "{1, 4, 9}");
    }
  }

  mod table_with_step {
    use super::*;

    #[test]
    fn table_positive_step() {
      assert_eq!(
        interpret("Table[i, {i, 1, 10, 2}]").unwrap(),
        "{1, 3, 5, 7, 9}"
      );
    }

    #[test]
    fn table_negative_step() {
      assert_eq!(
        interpret("Table[i, {i, 10, 1, -2}]").unwrap(),
        "{10, 8, 6, 4, 2}"
      );
    }

    #[test]
    fn table_step_of_three() {
      assert_eq!(interpret("Table[i, {i, 0, 9, 3}]").unwrap(), "{0, 3, 6, 9}");
    }
  }

  mod union_sorting {
    use super::*;

    #[test]
    fn union_sorts_elements() {
      assert_eq!(interpret("Union[{3, 1, 2}]").unwrap(), "{1, 2, 3}");
    }

    #[test]
    fn union_removes_duplicates_and_sorts() {
      assert_eq!(interpret("Union[{3, 1, 2, 1, 3}]").unwrap(), "{1, 2, 3}");
    }
  }

  mod subtract_function {
    use super::*;

    #[test]
    fn subtract_basic() {
      assert_eq!(interpret("Subtract[5, 2]").unwrap(), "3");
    }

    #[test]
    fn subtract_negative_result() {
      assert_eq!(interpret("Subtract[2, 5]").unwrap(), "-3");
    }
  }

  mod power_with_negative_exponent {
    use super::*;

    #[test]
    fn power_negative_one_exponent() {
      assert_eq!(interpret("Power[2, -1]").unwrap(), "1/2");
    }

    #[test]
    fn power_negative_two_exponent() {
      assert_eq!(interpret("Power[3, -2]").unwrap(), "1/9");
    }
  }

  mod real_number_formatting {
    use super::*;

    #[test]
    fn power_with_decimal_exponent() {
      // In Wolfram, 0.5 is Real so result is Real (2.)
      assert_eq!(interpret("Power[4, 0.5]").unwrap(), "2.");
    }

    #[test]
    fn accumulate_preserves_real() {
      assert_eq!(interpret("Accumulate[{1.5, 2.5}]").unwrap(), "{1.5, 4.}");
    }

    #[test]
    fn division_preserves_real_type() {
      assert_eq!(interpret("10.0 / 2").unwrap(), "5.");
    }

    #[test]
    fn integer_division_stays_integer() {
      assert_eq!(interpret("10 / 2").unwrap(), "5");
    }
  }

  mod replace_repeated {
    use super::*;

    #[test]
    fn replace_repeated_applies_multiple_times() {
      assert_eq!(interpret("f[f[f[f[2]]]] //. f[2] -> 2").unwrap(), "2");
    }

    #[test]
    fn replace_repeated_simple() {
      assert_eq!(interpret("f[f[2]] //. f[2] -> 2").unwrap(), "2");
    }
  }

  mod symbolic_ordering {
    use super::*;

    #[test]
    fn numbers_before_symbols() {
      assert_eq!(interpret("cow + 5").unwrap(), "5 + cow");
    }

    #[test]
    fn numeric_terms_combined() {
      assert_eq!(interpret("cow + 5 + 10").unwrap(), "15 + cow");
    }

    #[test]
    fn multiple_symbolic_terms_sorted() {
      assert_eq!(interpret("z + a + 3").unwrap(), "3 + a + z");
    }
  }

  mod power_formatting {
    use super::*;

    #[test]
    fn power_exponent_with_plus_gets_parens() {
      // D[x^n, x] = n*x^(-1 + n)
      assert_eq!(interpret("D[x^n, x]").unwrap(), "n*x^(-1 + n)");
    }
  }

  mod integrate_with_sum {
    use super::*;

    #[test]
    fn integrate_polynomial() {
      assert_eq!(interpret("Integrate[x^2, x]").unwrap(), "x^3/3");
    }

    #[test]
    fn integrate_sin() {
      assert_eq!(interpret("Integrate[Sin[x], x]").unwrap(), "-Cos[x]");
    }

    #[test]
    fn integrate_sum_of_terms() {
      // The ordering may differ from Mathematica but the result is correct
      let result = interpret("Integrate[x^2 + Sin[x], x]").unwrap();
      // Accept either ordering
      assert!(
        result == "x^3/3 - Cos[x]" || result == "-Cos[x] + x^3/3",
        "Got: {}",
        result
      );
    }
  }

  mod multiplication_formatting {
    use super::*;

    #[test]
    fn times_no_spaces() {
      assert_eq!(interpret("Times[2, x]").unwrap(), "2*x");
    }

    #[test]
    fn power_no_spaces() {
      assert_eq!(interpret("Power[x, 2]").unwrap(), "x^2");
    }
  }

  // Regression tests for bug fixes
  mod exact_value_returns {
    use super::*;

    #[test]
    fn sin_pi_half_returns_integer() {
      // Sin[Pi/2] should return 1 (Integer), not 1. (Real)
      assert_eq!(interpret("Sin[Pi/2]").unwrap(), "1");
    }

    #[test]
    fn cos_zero_returns_integer() {
      assert_eq!(interpret("Cos[0]").unwrap(), "1");
    }

    #[test]
    fn power_cube_root_returns_integer() {
      // Power[27, 1/3] should return 3 (Integer) when result is exact
      assert_eq!(interpret("Power[27, 1/3]").unwrap(), "3");
    }

    #[test]
    fn power_square_root_returns_integer() {
      assert_eq!(interpret("Power[16, 1/2]").unwrap(), "4");
    }

    #[test]
    fn mean_returns_rational() {
      // Mean[{0, 0, 0, 10}] = 10/4 = 5/2
      assert_eq!(interpret("Mean[{0, 0, 0, 10}]").unwrap(), "5/2");
    }

    #[test]
    fn mean_returns_integer_when_exact() {
      // Mean[{2, 4, 6}] = 12/3 = 4
      assert_eq!(interpret("Mean[{2, 4, 6}]").unwrap(), "4");
    }

    #[test]
    fn median_even_count_returns_rational() {
      // Median[{1, 2, 3, 4}] = (2+3)/2 = 5/2
      assert_eq!(interpret("Median[{1, 2, 3, 4}]").unwrap(), "5/2");
    }

    #[test]
    fn median_odd_count_returns_integer() {
      // Median[{1, 2, 3}] = 2
      assert_eq!(interpret("Median[{1, 2, 3}]").unwrap(), "2");
    }

    #[test]
    fn median_preserves_real_type() {
      // Median of reals should return real
      assert_eq!(interpret("Median[{1.5, 2.5, 3.5, 4.5}]").unwrap(), "3.");
    }
  }

  mod minus_wrong_arity {
    use super::*;

    #[test]
    fn minus_single_arg_negates() {
      assert_eq!(interpret("Minus[5]").unwrap(), "-5");
    }

    #[test]
    fn minus_two_args_returns_unevaluated() {
      // Minus[5, 2] should print warning and return 5 − 2
      let result = interpret("Minus[5, 2]").unwrap();
      assert_eq!(result, "5 − 2");
    }
  }

  mod part_out_of_bounds {
    use super::*;

    #[test]
    fn part_returns_unevaluated_on_out_of_bounds() {
      // {1, 2, 3}[[5]] should return unevaluated Part expression
      let result = interpret("{1, 2, 3}[[5]]").unwrap();
      assert_eq!(result, "{1, 2, 3}[[5]]");
    }

    #[test]
    fn part_negative_index_out_of_bounds() {
      let result = interpret("{1, 2}[[-5]]").unwrap();
      assert_eq!(result, "{1, 2}[[-5]]");
    }
  }

  mod take_out_of_bounds {
    use super::*;

    #[test]
    fn take_returns_unevaluated_on_out_of_bounds() {
      // Take[{1, 2, 3}, 5] should return unevaluated
      let result = interpret("Take[{1, 2, 3}, 5]").unwrap();
      assert_eq!(result, "Take[{1, 2, 3}, 5]");
    }

    #[test]
    fn take_negative_out_of_bounds() {
      let result = interpret("Take[{1, 2}, -5]").unwrap();
      assert_eq!(result, "Take[{1, 2}, -5]");
    }
  }

  mod association_part_assignment {
    use super::*;

    #[test]
    fn association_update_existing_key() {
      let result = interpret(
        r#"myHash = <|"A" -> 1, "B" -> 2|>; myHash[["A"]] = 5; myHash"#,
      )
      .unwrap();
      assert_eq!(result, "<|A -> 5, B -> 2|>");
    }

    #[test]
    fn association_add_new_key() {
      let result =
        interpret(r#"myHash = <|"A" -> 1|>; myHash[["B"]] = 2; myHash"#)
          .unwrap();
      assert_eq!(result, "<|A -> 1, B -> 2|>");
    }
  }

  mod association_nested_access {
    use super::*;

    #[test]
    fn nested_access_two_levels() {
      let result = interpret(
        r#"assoc = <|"outer" -> <|"inner" -> 8|>|>; assoc["outer", "inner"]"#,
      )
      .unwrap();
      assert_eq!(result, "8");
    }

    #[test]
    fn single_key_access() {
      let result = interpret(r#"assoc = <|"a" -> 1|>; assoc["a"]"#).unwrap();
      assert_eq!(result, "1");
    }
  }

  mod replace_repeated_operator_form {
    use super::*;

    #[test]
    fn operator_form_works() {
      // ReplaceRepeated[rule][expr] should work like expr //. rule
      let result =
        interpret("ReplaceRepeated[f[2] -> 2][f[f[f[f[2]]]]]").unwrap();
      assert_eq!(result, "2");
    }

    #[test]
    fn infix_form_works() {
      let result = interpret("f[f[f[2]]] //. f[2] -> 2").unwrap();
      assert_eq!(result, "2");
    }
  }

  mod real_precision {
    use super::*;

    #[test]
    fn full_precision_when_needed() {
      // Power[1.5, 2.5] needs full precision
      assert_eq!(interpret("Power[1.5, 2.5]").unwrap(), "2.7556759606310752");
    }

    #[test]
    fn short_precision_when_clean() {
      // Simple addition should round cleanly
      assert_eq!(interpret("1.5 + 2.7").unwrap(), "4.2");
    }
  }
}
