use woxi::{clear_state, interpret, interpret_with_stdout};

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
        assert_eq!(interpret("9.6 / 3").unwrap(), "3.1999999999999997");
      }

      #[test]
      fn complex_division() {
        assert_eq!(
          interpret("9.6 / 3 + 3.0 / 3").unwrap(),
          "4.199999999999999"
        );
      }

      #[test]
      fn addition_ieee754_precision() {
        // Must preserve IEEE 754 representation, not round
        assert_eq!(interpret("0.1 + 0.2").unwrap(), "0.30000000000000004");
      }

      #[test]
      fn division_repeating() {
        assert_eq!(interpret("1.0 / 3.0").unwrap(), "0.3333333333333333");
      }

      #[test]
      fn sqrt_real() {
        assert_eq!(interpret("Sqrt[2.0]").unwrap(), "1.4142135623730951");
      }

      #[test]
      fn whole_number_real() {
        // Whole-number reals keep trailing dot
        assert_eq!(interpret("1.0").unwrap(), "1.");
        assert_eq!(interpret("100.0").unwrap(), "100.");
      }

      #[test]
      fn real_type_preserved_in_addition() {
        assert_eq!(interpret("2.0 + 3.0").unwrap(), "5.");
        assert_eq!(interpret("Head[2.0 + 3.0]").unwrap(), "Real");
      }

      #[test]
      fn real_type_preserved_in_subtraction() {
        assert_eq!(interpret("6.0 - 3.0").unwrap(), "3.");
        assert_eq!(interpret("Head[6.0 - 3.0]").unwrap(), "Real");
      }

      #[test]
      fn real_type_preserved_in_multiplication() {
        assert_eq!(interpret("2.0 * 3.0").unwrap(), "6.");
        assert_eq!(interpret("Head[2.0 * 3.0]").unwrap(), "Real");
      }

      #[test]
      fn real_type_preserved_in_negation() {
        assert_eq!(interpret("-3.0").unwrap(), "-3.");
        assert_eq!(interpret("Head[-3.0]").unwrap(), "Real");
      }
    }

    mod times_simplification {
      use super::*;

      #[test]
      fn zero_times_symbol() {
        assert_eq!(interpret("0*x").unwrap(), "0");
        assert_eq!(interpret("x*0").unwrap(), "0");
      }

      #[test]
      fn one_times_symbol() {
        assert_eq!(interpret("1*x").unwrap(), "x");
        assert_eq!(interpret("x*1").unwrap(), "x");
      }

      #[test]
      fn times_function_zero() {
        assert_eq!(interpret("Times[0, x]").unwrap(), "0");
        assert_eq!(interpret("Times[x, 0]").unwrap(), "0");
        assert_eq!(interpret("Times[0, x, y]").unwrap(), "0");
      }

      #[test]
      fn times_function_one() {
        assert_eq!(interpret("Times[1, x]").unwrap(), "x");
        assert_eq!(interpret("Times[x, 1]").unwrap(), "x");
      }

      #[test]
      fn times_coefficient() {
        assert_eq!(interpret("Times[2, x]").unwrap(), "2*x");
        assert_eq!(interpret("Times[-1, x]").unwrap(), "-x");
      }

      #[test]
      fn zero_times_list() {
        assert_eq!(interpret("0*{a, b, c}").unwrap(), "{0, 0, 0}");
      }

      #[test]
      fn one_times_list() {
        assert_eq!(interpret("1*{a, b, c}").unwrap(), "{a, b, c}");
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

    #[test]
    fn postfix_after_operator_chain() {
      // (1 + 2) // ToString is ToString[Plus[1, 2]]
      assert_eq!(interpret("1 + 2 // ToString").unwrap(), "3");
      // Map operator followed by postfix
      assert_eq!(interpret("Sqrt /@ {1, 4, 9} // Length").unwrap(), "3");
    }

    #[test]
    fn postfix_after_map_with_anonymous_function() {
      // Which[...]& /@ Range[1, 5] // Map[Print] - pattern from fizzbuzz_5
      assert_eq!(interpret("(# + 1)& /@ {1, 2, 3} // Length").unwrap(), "3");
    }
  }

  mod trailing_semicolon {
    use super::*;

    #[test]
    fn trailing_semicolon_returns_null() {
      // expr; is CompoundExpression[expr, Null] — result is Null
      assert_eq!(interpret("1 + 2;").unwrap(), "Null");
    }

    #[test]
    fn trailing_semicolon_with_print() {
      // Print[1]; should still execute Print, result is Null
      assert_eq!(interpret("Print[1];").unwrap(), "Null");
    }

    #[test]
    fn trailing_semicolon_with_postfix() {
      // {1,2,3} // Map[Print]; should print and return Null
      assert_eq!(interpret("{1,2,3} // Map[Print];").unwrap(), "Null");
    }

    #[test]
    fn no_trailing_semicolon_shows_result() {
      // Without trailing ;, result should be shown
      assert_eq!(interpret("1 + 2").unwrap(), "3");
    }

    #[test]
    fn compound_expression_with_trailing_semicolon() {
      // x = 5; x + 1; should return Null
      assert_eq!(interpret("x = 5; x + 1;").unwrap(), "Null");
    }

    #[test]
    fn compound_expression_without_trailing_semicolon() {
      // x = 5; x + 1 should show the final result
      assert_eq!(interpret("x = 5; x + 1").unwrap(), "6");
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

    #[test]
    fn paren_anonymous_with_compound_expression() {
      // (expr1; expr2)& should create an anonymous function with CompoundExpression body
      assert_eq!(
        interpret("Reap[Nest[(Sow[#]; 3*# + 1) &, 7, 5]]").unwrap(),
        "{1822, {{7, 22, 67, 202, 607}}}"
      );
    }

    #[test]
    fn paren_anonymous_direct_call() {
      // (expr)&[args] should call the anonymous function directly
      assert_eq!(interpret("(# + 1) &[5]").unwrap(), "6");
      assert_eq!(interpret("(# * 2 + 3) &[4]").unwrap(), "11");
      assert_eq!(interpret("(#1 + #2) &[3, 4]").unwrap(), "7");
    }

    #[test]
    fn function_anonymous_direct_call() {
      // If[...]&[args] should call the anonymous function directly
      assert_eq!(interpret("If[# > 0, #, 0] &[5]").unwrap(), "5");
      assert_eq!(interpret("If[# > 0, #, 0] &[-3]").unwrap(), "0");
    }

    #[test]
    fn list_anonymous_direct_call() {
      // {expr}&[args] should call the anonymous function directly
      assert_eq!(interpret("{#, #^2} &[3]").unwrap(), "{3, 9}");
      assert_eq!(interpret("{#1, #2, #1 + #2} &[2, 5]").unwrap(), "{2, 5, 7}");
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

  mod big_integer {
    use super::*;

    #[test]
    fn power_exceeding_i128() {
      assert_eq!(
        interpret("2^127").unwrap(),
        "170141183460469231731687303715884105728"
      );
    }

    #[test]
    fn power_200() {
      assert_eq!(
        interpret("2^200").unwrap(),
        "1606938044258990275541962092341162602522202993782792835301376"
      );
    }

    #[test]
    fn big_power_minus_one_fits_i128() {
      // 2^127 - 1 = i128::MAX, should convert back to Integer
      assert_eq!(
        interpret("2^127 - 1").unwrap(),
        "170141183460469231731687303715884105727"
      );
    }

    #[test]
    fn big_power_addition() {
      assert_eq!(
        interpret("2^127 + 1").unwrap(),
        "170141183460469231731687303715884105729"
      );
    }

    #[test]
    fn big_power_multiplication() {
      assert_eq!(
        interpret("2^127 * 2").unwrap(),
        "340282366920938463463374607431768211456"
      );
    }

    #[test]
    fn large_i128_subtraction() {
      // 2^67 fits in i128 but exceeds f64 precision (> 2^53)
      assert_eq!(interpret("2^67 - 1").unwrap(), "147573952589676412927");
    }

    #[test]
    fn large_i128_addition() {
      assert_eq!(interpret("2^67 + 1").unwrap(), "147573952589676412929");
    }

    #[test]
    fn large_i128_multiplication() {
      assert_eq!(interpret("2^67 * 3").unwrap(), "442721857769029238784");
    }

    #[test]
    fn large_i128_sum_of_two() {
      assert_eq!(interpret("10^20 + 10^20").unwrap(), "200000000000000000000");
    }

    #[test]
    fn factor_integer_large() {
      assert_eq!(
        interpret("FactorInteger[2^67 - 1]").unwrap(),
        "{{193707721, 1}, {761838257287, 1}}"
      );
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

    #[test]
    fn power_plus_base_gets_parens() {
      assert_eq!(interpret("(1 + x)^(-1)").unwrap(), "(1 + x)^(-1)");
      assert_eq!(interpret("(-1 + x)^(-1)").unwrap(), "(-1 + x)^(-1)");
      assert_eq!(interpret("(2 + x)^3").unwrap(), "(2 + x)^3");
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

  mod definite_integrals {
    use super::*;

    #[test]
    fn gaussian_integral_full() {
      // ∫_{-∞}^{∞} E^(-x^2) dx = Sqrt[Pi]
      assert_eq!(
        interpret("Integrate[E^(-x^2), {x, -Infinity, Infinity}]").unwrap(),
        "Sqrt[Pi]"
      );
    }

    #[test]
    fn gaussian_integral_with_coefficient() {
      // ∫_{-∞}^{∞} E^(-2x^2) dx = Sqrt[Pi/2]
      assert_eq!(
        interpret("Integrate[E^(-2*x^2), {x, -Infinity, Infinity}]").unwrap(),
        "Sqrt[Pi/2]"
      );
    }

    #[test]
    fn half_gaussian_integral() {
      // ∫_0^{∞} E^(-x^2) dx = Sqrt[Pi]/2
      assert_eq!(
        interpret("Integrate[E^(-x^2), {x, 0, Infinity}]").unwrap(),
        "Sqrt[Pi]/2"
      );
    }

    #[test]
    fn definite_integral_polynomial() {
      // ∫_0^1 x^2 dx = 1/3
      assert_eq!(interpret("Integrate[x^2, {x, 0, 1}]").unwrap(), "1/3");
    }

    #[test]
    fn definite_integral_constant() {
      // ∫_0^3 5 dx = 15
      assert_eq!(interpret("Integrate[5, {x, 0, 3}]").unwrap(), "15");
    }
  }

  mod unary_minus_parsing {
    use super::*;

    #[test]
    fn negative_identifier_in_parens() {
      assert_eq!(interpret("(-x)").unwrap(), "-x");
    }

    #[test]
    fn negative_power_in_parens() {
      // (-x^2) should be -(x^2), not (-x)^2
      assert_eq!(interpret("(-x^2)").unwrap(), "-x^2");
    }

    #[test]
    fn negative_expr_plus_constant() {
      assert_eq!(interpret("(-x + 3)").unwrap(), "3 - x");
    }

    #[test]
    fn e_to_negative_x_squared() {
      assert_eq!(interpret("E^(-x^2)").unwrap(), "E^(-x^2)");
    }

    #[test]
    fn negative_power_exponent() {
      assert_eq!(interpret("x^(-2)").unwrap(), "x^(-2)");
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

    #[test]
    fn negated_division_formatting() {
      // Wolfram displays -(a/b) not -a/b or (-a)/b
      assert_eq!(interpret("-a/b").unwrap(), "-(a/b)");
      assert_eq!(interpret("{-a/b}").unwrap(), "{-(a/b)}");
    }

    #[test]
    fn negated_product_formatting() {
      // Wolfram displays -(a*b) not -a*b
      assert_eq!(interpret("-a*b").unwrap(), "-(a*b)");
      assert_eq!(interpret("{-a*b}").unwrap(), "{-(a*b)}");
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

  mod variance {
    use super::*;

    #[test]
    fn variance_integers() {
      assert_eq!(interpret("Variance[{1, 2, 3}]").unwrap(), "1");
      assert_eq!(interpret("Variance[{1, 2, 3, 4, 5}]").unwrap(), "5/2");
    }

    #[test]
    fn variance_reals() {
      let result = interpret("Variance[{1.0, 2.0, 3.0}]").unwrap();
      // Should be 1 or close to it
      let val: f64 = result.parse().unwrap();
      assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn variance_four_integers() {
      assert_eq!(interpret("Variance[{2, 4, 6, 8}]").unwrap(), "20/3");
    }
  }

  mod standard_deviation {
    use super::*;

    #[test]
    fn stddev_integers() {
      assert_eq!(interpret("StandardDeviation[{1, 2, 3}]").unwrap(), "1");
      assert_eq!(
        interpret("StandardDeviation[{1, 2, 3, 4, 5}]").unwrap(),
        "Sqrt[5/2]"
      );
    }

    #[test]
    fn stddev_reals() {
      let result = interpret("StandardDeviation[{1.0, 2.0, 3.0}]").unwrap();
      let val: f64 = result.parse().unwrap();
      assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stddev_perfect_square_variance() {
      // Variance of {1,3,5,7} = 20/3, not perfect square
      // Variance of {0, 2} = 2, StdDev = Sqrt[2]
      assert_eq!(interpret("StandardDeviation[{0, 2}]").unwrap(), "Sqrt[2]");
    }
  }

  mod geometric_mean {
    use super::*;

    #[test]
    fn geometric_mean_perfect_result() {
      assert_eq!(interpret("GeometricMean[{2, 8}]").unwrap(), "4");
    }

    #[test]
    fn geometric_mean_reals() {
      let result = interpret("GeometricMean[{1.0, 2.0, 3.0}]").unwrap();
      let val: f64 = result.parse().unwrap();
      assert!((val - 1.8171205928321397).abs() < 1e-10);
    }
  }

  mod harmonic_mean {
    use super::*;

    #[test]
    fn harmonic_mean_integers() {
      assert_eq!(interpret("HarmonicMean[{1, 2, 3}]").unwrap(), "18/11");
    }

    #[test]
    fn harmonic_mean_reals() {
      let result = interpret("HarmonicMean[{1.0, 2.0, 3.0}]").unwrap();
      let val: f64 = result.parse().unwrap();
      assert!((val - 1.6363636363636365).abs() < 1e-10);
    }

    #[test]
    fn harmonic_mean_equal_elements() {
      assert_eq!(interpret("HarmonicMean[{5, 5, 5}]").unwrap(), "5");
    }
  }

  mod root_mean_square {
    use super::*;

    #[test]
    fn rms_integers() {
      assert_eq!(
        interpret("RootMeanSquare[{1, 2, 3}]").unwrap(),
        "Sqrt[14/3]"
      );
    }

    #[test]
    fn rms_perfect_result() {
      // RMS of {3, 4} = Sqrt[(9+16)/2] = Sqrt[25/2]
      // RMS of {1, 1} = 1
      assert_eq!(interpret("RootMeanSquare[{1, 1}]").unwrap(), "1");
    }

    #[test]
    fn rms_reals() {
      let result = interpret("RootMeanSquare[{1.0, 2.0, 3.0}]").unwrap();
      let val: f64 = result.parse().unwrap();
      assert!((val - 2.160246899469287).abs() < 1e-10);
    }
  }

  mod integer_length {
    use super::*;

    #[test]
    fn integer_length_basic() {
      assert_eq!(interpret("IntegerLength[12345]").unwrap(), "5");
      assert_eq!(interpret("IntegerLength[0]").unwrap(), "0");
      assert_eq!(interpret("IntegerLength[1]").unwrap(), "1");
      assert_eq!(interpret("IntegerLength[-99]").unwrap(), "2");
    }

    #[test]
    fn integer_length_base() {
      assert_eq!(interpret("IntegerLength[255, 16]").unwrap(), "2");
      assert_eq!(interpret("IntegerLength[100, 10]").unwrap(), "3");
      assert_eq!(interpret("IntegerLength[255, 2]").unwrap(), "8");
      assert_eq!(interpret("IntegerLength[12345, 2]").unwrap(), "14");
    }
  }

  mod rescale {
    use super::*;

    #[test]
    fn rescale_basic() {
      assert_eq!(interpret("Rescale[5, {0, 10}]").unwrap(), "1/2");
      assert_eq!(interpret("Rescale[0, {0, 10}]").unwrap(), "0");
      assert_eq!(interpret("Rescale[10, {0, 10}]").unwrap(), "1");
    }

    #[test]
    fn rescale_with_target_range() {
      assert_eq!(interpret("Rescale[5, {0, 10}, {-1, 1}]").unwrap(), "0");
    }

    #[test]
    fn rescale_list() {
      assert_eq!(
        interpret("Rescale[{1, 2, 3, 4, 5}]").unwrap(),
        "{0, 1/4, 1/2, 3/4, 1}"
      );
    }

    #[test]
    fn rescale_boundary() {
      assert_eq!(interpret("Rescale[3, {1, 5}]").unwrap(), "1/2");
    }
  }

  mod normalize {
    use super::*;

    #[test]
    fn normalize_integers_perfect() {
      assert_eq!(interpret("Normalize[{3, 4}]").unwrap(), "{3/5, 4/5}");
    }

    #[test]
    fn normalize_integers_irrational() {
      assert_eq!(
        interpret("Normalize[{1, 1, 1}]").unwrap(),
        "{1/Sqrt[3], 1/Sqrt[3], 1/Sqrt[3]}"
      );
    }

    #[test]
    fn normalize_zero_vector() {
      assert_eq!(interpret("Normalize[{0, 0, 0}]").unwrap(), "{0, 0, 0}");
    }

    #[test]
    fn normalize_three_elements() {
      assert_eq!(
        interpret("Normalize[{1, 2, 2}]").unwrap(),
        "{1/3, 2/3, 2/3}"
      );
    }
  }

  mod subsequences {
    use super::*;

    #[test]
    fn subsequences_all() {
      assert_eq!(
        interpret("Subsequences[{a, b, c}]").unwrap(),
        "{{}, {a}, {b}, {c}, {a, b}, {b, c}, {a, b, c}}"
      );
    }

    #[test]
    fn subsequences_fixed_length() {
      assert_eq!(
        interpret("Subsequences[{a, b, c}, {2}]").unwrap(),
        "{{a, b}, {b, c}}"
      );
    }

    #[test]
    fn subsequences_length_range() {
      assert_eq!(
        interpret("Subsequences[{a, b, c, d}, {2, 3}]").unwrap(),
        "{{a, b}, {b, c}, {c, d}, {a, b, c}, {b, c, d}}"
      );
    }

    #[test]
    fn subsequences_empty() {
      assert_eq!(interpret("Subsequences[{}]").unwrap(), "{{}}");
    }

    #[test]
    fn subsequences_zero_length() {
      assert_eq!(interpret("Subsequences[{a, b}, {0, 0}]").unwrap(), "{{}}");
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

  mod date_string {
    use super::*;

    #[test]
    fn date_string_returns_string() {
      // DateString should return a string (not cause parse error)
      let result = interpret("StringQ[DateString[]]").unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn date_string_with_now() {
      let result = interpret("StringQ[DateString[Now]]").unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn date_string_iso_format() {
      // ISODateTime format should contain T separator
      let result =
        interpret("StringContainsQ[DateString[Now, \"ISODateTime\"], \"T\"]")
          .unwrap();
      assert_eq!(result, "True");
    }
  }

  mod create_file {
    use super::*;

    #[test]
    fn create_file_returns_string() {
      // CreateFile should return a string path (not cause parse error)
      let result = interpret("StringQ[CreateFile[]]").unwrap();
      assert_eq!(result, "True");
    }

    #[test]
    fn create_file_path_exists() {
      // The returned path should be a non-empty string
      let result = interpret("StringLength[CreateFile[]] > 0").unwrap();
      assert_eq!(result, "True");
    }
  }

  mod symbolic_product {
    use super::*;

    #[test]
    fn product_i_squared_symbolic() {
      // Product[i^2, {i, 1, n}] = n!^2
      assert_eq!(interpret("Product[i^2, {i, 1, n}]").unwrap(), "n!^2");
    }

    #[test]
    fn product_i_symbolic() {
      // Product[i, {i, 1, n}] = n!
      assert_eq!(interpret("Product[i, {i, 1, n}]").unwrap(), "n!");
    }

    #[test]
    fn product_numeric() {
      // Numeric bounds should still work
      assert_eq!(interpret("Product[i^2, {i, 1, 6}]").unwrap(), "518400");
    }

    #[test]
    fn factorial_formatting() {
      // Factorial[n] should display as n!
      assert_eq!(interpret("Factorial[n]").unwrap(), "n!");
      assert_eq!(interpret("Factorial[5]").unwrap(), "120");
    }
  }

  mod anonymous_function_call {
    use super::*;

    #[test]
    fn identity_anonymous() {
      // #&[1] should return 1
      assert_eq!(interpret("#&[1]").unwrap(), "1");
    }

    #[test]
    fn power_anonymous() {
      // #^2&[{1, 2, 3}] should map squaring
      assert_eq!(interpret("#^2 &[{1, 2, 3}]").unwrap(), "{1, 4, 9}");
    }

    #[test]
    fn anonymous_with_addition() {
      assert_eq!(interpret("#+10&[5]").unwrap(), "15");
    }
  }

  mod prefix_application_associativity {
    use super::*;

    #[test]
    fn right_associative_chaining() {
      // f @ g @ x should be f[g[x]] (right-associative)
      assert_eq!(
        interpret("Double[x_] := x * 2; Double @ Sin @ (Pi/2)").unwrap(),
        "2"
      );
    }

    #[test]
    fn single_prefix() {
      assert_eq!(interpret("Sqrt @ 16").unwrap(), "4");
    }
  }

  mod sign_predicates {
    use super::*;

    #[test]
    fn positive() {
      assert_eq!(interpret("Positive[5]").unwrap(), "True");
      assert_eq!(interpret("Positive[-3]").unwrap(), "False");
      assert_eq!(interpret("Positive[0]").unwrap(), "False");
    }

    #[test]
    fn negative() {
      assert_eq!(interpret("Negative[-5]").unwrap(), "True");
      assert_eq!(interpret("Negative[3]").unwrap(), "False");
      assert_eq!(interpret("Negative[0]").unwrap(), "False");
    }

    #[test]
    fn non_positive() {
      assert_eq!(interpret("NonPositive[-5]").unwrap(), "True");
      assert_eq!(interpret("NonPositive[0]").unwrap(), "True");
      assert_eq!(interpret("NonPositive[3]").unwrap(), "False");
    }

    #[test]
    fn non_negative() {
      assert_eq!(interpret("NonNegative[5]").unwrap(), "True");
      assert_eq!(interpret("NonNegative[0]").unwrap(), "True");
      assert_eq!(interpret("NonNegative[-3]").unwrap(), "False");
    }
  }

  mod if_function_extended {
    use super::*;

    #[test]
    fn if_four_args_default() {
      // If[non-boolean, true-branch, false-branch, default]
      // Non-boolean condition should return default (4th arg)
      assert_eq!(interpret("If[\"x\", 1, 0, 2]").unwrap(), "2");
    }

    #[test]
    fn if_four_args_true() {
      assert_eq!(interpret("If[True, 1, 0, 2]").unwrap(), "1");
    }

    #[test]
    fn if_four_args_false() {
      assert_eq!(interpret("If[False, 1, 0, 2]").unwrap(), "0");
    }
  }

  mod plus_formatting {
    use super::*;

    #[test]
    fn plus_with_negative_term() {
      // a + (-b) should format as a - b
      let result = interpret("Integrate[Sin[x], x]").unwrap();
      assert_eq!(result, "-Cos[x]");
    }

    #[test]
    fn integrate_sum_formatting() {
      // Integrate[x^2 + Sin[x], x] should format nicely
      let result = interpret("Integrate[x^2 + Sin[x], x]").unwrap();
      assert_eq!(result, "x^3/3 - Cos[x]");
    }

    #[test]
    fn plus_term_ordering_polynomial_first() {
      // Polynomial terms should come before transcendental functions
      let result = interpret("x^2 + Sin[x]").unwrap();
      assert_eq!(result, "x^2 + Sin[x]");
    }

    #[test]
    fn plus_term_ordering_alphabetical() {
      // Same priority terms should be alphabetical
      let result = interpret("c + a + b").unwrap();
      assert_eq!(result, "a + b + c");
    }

    #[test]
    fn plus_times_before_identifier() {
      // Times[a, b] should come before c alphabetically (a < c)
      let result = interpret("FullForm[a b + c]").unwrap();
      assert_eq!(result, "Plus[Times[a, b], c]");
    }
  }

  mod subtraction_without_spaces {
    use super::*;

    #[test]
    fn n_minus_1_in_function_body() {
      // Regression: n-1 (without spaces) was parsed as implicit multiplication n*(-1)
      clear_state();
      assert_eq!(interpret("f[n_] := n-1; f[99]").unwrap(), "98");
    }

    #[test]
    fn subtraction_in_nested_function_call() {
      // Regression: ToString[n-1] was evaluating n-1 as -(n) instead of n minus 1
      clear_state();
      assert_eq!(interpret("f[n_] := ToString[n-1]; f[99]").unwrap(), "98");
    }

    #[test]
    fn subtraction_in_string_join() {
      clear_state();
      assert_eq!(
        interpret(
          r#"f[n_] := ToString[n] <> " minus 1 is " <> ToString[n-1]; f[10]"#
        )
        .unwrap(),
        "10 minus 1 is 9"
      );
    }

    #[test]
    fn tostring_input_form() {
      // ToString[expr, InputForm] — strings are quoted, fractions single-line
      assert_eq!(
        interpret(r#"ToString["hello", InputForm]"#).unwrap(),
        r#""hello""#
      );
      assert_eq!(interpret("ToString[1/3, InputForm]").unwrap(), "1/3");
      assert_eq!(
        interpret(r#"ToString[{1, "a", x^2}, InputForm]"#).unwrap(),
        r#"{1, "a", x^2}"#
      );
      assert_eq!(interpret("ToString[x + y, InputForm]").unwrap(), "x + y");
      // Without InputForm, strings are unquoted
      assert_eq!(interpret(r#"ToString["hello"]"#).unwrap(), "hello");
    }

    #[test]
    fn negative_numbers_still_work() {
      assert_eq!(interpret("{-1, -2, -3}").unwrap(), "{-1, -2, -3}");
      assert_eq!(interpret("-1 + 3").unwrap(), "2");
    }
  }

  mod conditional_definitions {
    use super::*;

    #[test]
    fn single_condition() {
      clear_state();
      assert_eq!(
        interpret(
          "f[n_ /; n > 0] := \"positive\"; f[n_] := \"non-positive\"; f[3]"
        )
        .unwrap(),
        "positive"
      );
    }

    #[test]
    fn single_condition_fallback() {
      clear_state();
      assert_eq!(
        interpret(
          "f[n_ /; n > 0] := \"positive\"; f[n_] := \"non-positive\"; f[-1]"
        )
        .unwrap(),
        "non-positive"
      );
    }

    #[test]
    fn multiple_conditions_fizzbuzz() {
      // Regression: multiple SetDelayed definitions with conditions overwrote each other
      clear_state();
      assert_eq!(
        interpret(r#"f[n_ /; Mod[n, 15] == 0] := "FizzBuzz"; f[n_ /; Mod[n, 3] == 0] := "Fizz"; f[n_ /; Mod[n, 5] == 0] := "Buzz"; f[n_] := n; f[3]"#).unwrap(),
        "Fizz"
      );
      assert_eq!(interpret("f[5]").unwrap(), "Buzz");
      assert_eq!(interpret("f[15]").unwrap(), "FizzBuzz");
      assert_eq!(interpret("f[7]").unwrap(), "7");
    }

    #[test]
    fn conditions_tried_in_order() {
      // Definitions are tried in the order they were defined
      clear_state();
      assert_eq!(
        interpret(r#"g[n_ /; n > 10] := "big"; g[n_ /; n > 0] := "small"; g[n_] := "zero or negative"; g[20]"#).unwrap(),
        "big"
      );
      assert_eq!(interpret("g[5]").unwrap(), "small");
      assert_eq!(interpret("g[0]").unwrap(), "zero or negative");
    }
  }

  mod set_attributes {
    use super::*;

    #[test]
    fn listable_threads_over_list() {
      clear_state();
      assert_eq!(
        interpret("SetAttributes[f, Listable]; f[x_] := x * 2; f[{1, 2, 3}]")
          .unwrap(),
        "{2, 4, 6}"
      );
    }

    #[test]
    fn listable_with_conditions() {
      clear_state();
      assert_eq!(
        interpret(r#"SetAttributes[f, Listable]; f[n_ /; Mod[n, 3] == 0] := "Fizz"; f[n_] := n; f[{1, 2, 3, 4, 5, 6}]"#).unwrap(),
        r#"{1, 2, Fizz, 4, 5, Fizz}"#
      );
    }

    #[test]
    fn listable_single_value_unchanged() {
      clear_state();
      assert_eq!(
        interpret("SetAttributes[f, Listable]; f[x_] := x + 1; f[5]").unwrap(),
        "6"
      );
    }
  }

  mod replace_all_after_operators {
    use super::*;

    #[test]
    fn replace_all_after_plus() {
      assert_eq!(interpret("x + y /. x -> 1").unwrap(), "1 + y");
    }

    #[test]
    fn replace_all_after_times() {
      assert_eq!(interpret("x * y /. x -> 2").unwrap(), "2*y");
    }

    #[test]
    fn replace_all_after_power() {
      assert_eq!(interpret("x^2 + y /. x -> 3").unwrap(), "9 + y");
    }

    #[test]
    fn replace_all_multiple_operators() {
      assert_eq!(interpret("x + y + z /. {x -> 1, y -> 2}").unwrap(), "3 + z");
    }

    #[test]
    fn replace_all_times_multiple_vars() {
      assert_eq!(interpret("x * y * z /. {x -> 2, y -> 3}").unwrap(), "6*z");
    }

    #[test]
    fn replace_all_with_implicit_times() {
      assert_eq!(interpret("2 x + 3 y /. {x -> 1, y -> 2}").unwrap(), "8");
    }

    #[test]
    fn replace_repeated_after_plus() {
      assert_eq!(interpret("x + y //. x -> 1").unwrap(), "1 + y");
    }

    #[test]
    fn replace_repeated_after_times() {
      assert_eq!(interpret("x * y //. x -> 2").unwrap(), "2*y");
    }

    #[test]
    fn replace_all_after_comparison() {
      assert_eq!(interpret("x > y /. x -> 3").unwrap(), "3 > y");
    }

    #[test]
    fn replace_all_all_vars_replaced() {
      assert_eq!(interpret("x + y /. {x -> 10, y -> 20}").unwrap(), "30");
    }
  }

  mod compound_assignment {
    use super::*;

    #[test]
    fn add_to() {
      clear_state();
      assert_eq!(interpret("x = 5; x += 3; x").unwrap(), "8");
    }

    #[test]
    fn add_to_return_value() {
      clear_state();
      assert_eq!(interpret("x = 10; x += 7").unwrap(), "17");
    }

    #[test]
    fn subtract_from() {
      clear_state();
      assert_eq!(interpret("x = 10; x -= 3; x").unwrap(), "7");
    }

    #[test]
    fn times_by() {
      clear_state();
      assert_eq!(interpret("x = 5; x *= 4; x").unwrap(), "20");
    }

    #[test]
    fn divide_by() {
      clear_state();
      assert_eq!(interpret("x = 20; x /= 4; x").unwrap(), "5");
    }

    #[test]
    fn chained_compound_assignment() {
      clear_state();
      assert_eq!(interpret("x = 1; x += 2; x *= 3; x -= 1; x").unwrap(), "8");
    }
  }

  mod fibonacci_builtin {
    use super::*;

    #[test]
    fn fibonacci_zero() {
      assert_eq!(interpret("Fibonacci[0]").unwrap(), "0");
    }

    #[test]
    fn fibonacci_one() {
      assert_eq!(interpret("Fibonacci[1]").unwrap(), "1");
    }

    #[test]
    fn fibonacci_small() {
      assert_eq!(interpret("Fibonacci[9]").unwrap(), "34");
      assert_eq!(interpret("Fibonacci[10]").unwrap(), "55");
    }

    #[test]
    fn fibonacci_negative() {
      assert_eq!(interpret("Fibonacci[-1]").unwrap(), "1");
      assert_eq!(interpret("Fibonacci[-2]").unwrap(), "-1");
      assert_eq!(interpret("Fibonacci[-3]").unwrap(), "2");
      assert_eq!(interpret("Fibonacci[-4]").unwrap(), "-3");
    }

    #[test]
    fn fibonacci_symbolic() {
      assert_eq!(interpret("Fibonacci[x]").unwrap(), "Fibonacci[x]");
    }
  }

  mod down_values {
    use super::*;

    #[test]
    fn basic_down_value() {
      clear_state();
      assert_eq!(interpret("f[0] = 42; f[0]").unwrap(), "42");
    }

    #[test]
    fn multiple_down_values() {
      clear_state();
      assert_eq!(interpret("f[0] = 0; f[1] = 1; f[0]").unwrap(), "0");
      assert_eq!(interpret("f[1]").unwrap(), "1");
    }

    #[test]
    fn down_value_with_pattern() {
      clear_state();
      assert_eq!(
        interpret(
          "g[0] = 0; g[1] = 1; g[n_Integer] := g[n - 1] + g[n - 2]; g[5]"
        )
        .unwrap(),
        "5"
      );
    }

    #[test]
    fn memoized_down_value() {
      clear_state();
      assert_eq!(
        interpret("h[0] = 0; h[1] = 1; h[n_Integer] := h[n] = h[n - 1] + h[n - 2]; h[10]").unwrap(),
        "55"
      );
    }
  }

  mod block_scoping {
    use super::*;

    #[test]
    fn basic_block() {
      clear_state();
      assert_eq!(interpret("Block[{x = 5}, x + 1]").unwrap(), "6");
    }

    #[test]
    fn block_restores_variables() {
      clear_state();
      assert_eq!(interpret("x = 10; Block[{x = 5}, x]; x").unwrap(), "10");
    }

    #[test]
    fn block_uninitialized_var() {
      clear_state();
      assert_eq!(interpret("Block[{x}, x]").unwrap(), "x");
    }

    #[test]
    fn block_multiple_vars() {
      clear_state();
      assert_eq!(interpret("Block[{x = 3, y = 4}, x + y]").unwrap(), "7");
    }
  }

  mod for_loop {
    use super::*;

    #[test]
    fn basic_for() {
      clear_state();
      assert_eq!(
        interpret("s = 0; For[i = 1, i <= 5, i++, s += i]; s").unwrap(),
        "15"
      );
    }

    #[test]
    fn for_returns_null() {
      clear_state();
      assert_eq!(interpret("For[i = 0, i < 3, i++, i]").unwrap(), "Null");
    }
  }

  mod while_loop {
    use super::*;

    #[test]
    fn basic_while() {
      clear_state();
      assert_eq!(interpret("i = 0; While[i < 5, i++]; i").unwrap(), "5");
    }

    #[test]
    fn while_with_assignment() {
      clear_state();
      assert_eq!(
        interpret("n = 0; While[n < 10, n = n + 3]; n").unwrap(),
        "12"
      );
    }

    #[test]
    fn while_returns_null() {
      clear_state();
      assert_eq!(interpret("i = 0; While[i < 3, i++]").unwrap(), "Null");
    }

    #[test]
    fn while_with_break() {
      clear_state();
      assert_eq!(
        interpret("i = 0; While[True, i++; If[i >= 5, Break[]]]; i").unwrap(),
        "5"
      );
    }

    #[test]
    fn while_in_module() {
      clear_state();
      assert_eq!(
        interpret("Module[{i = 0, s = 0}, While[i < 5, s += i; i++]; s]")
          .unwrap(),
        "10"
      );
    }

    #[test]
    fn while_false_condition() {
      clear_state();
      assert_eq!(interpret("While[False, Print[1]]").unwrap(), "Null");
    }
  }

  mod return_value {
    use super::*;

    #[test]
    fn return_in_block() {
      clear_state();
      // Return propagates through Block; at top level it becomes symbolic Return[val] (like wolframscript)
      assert_eq!(interpret("Block[{}, Return[42]]").unwrap(), "Return[42]");
    }

    #[test]
    fn return_in_module() {
      clear_state();
      // At top level, uncaught Return[] becomes symbolic Return[val] (like wolframscript)
      assert_eq!(
        interpret("Module[{x = 10}, Return[x + 1]]").unwrap(),
        "Return[11]"
      );
    }

    #[test]
    fn return_in_block_inside_function() {
      clear_state();
      assert_eq!(
        interpret("f[] := Block[{}, Return[42]]; f[]").unwrap(),
        "42"
      );
    }

    #[test]
    fn return_in_module_inside_function() {
      clear_state();
      assert_eq!(
        interpret("g[] := Module[{x = 10}, Return[x + 1]]; g[]").unwrap(),
        "11"
      );
    }
  }

  mod set_delayed {
    use super::*;

    #[test]
    fn list_pattern_destructuring() {
      clear_state();
      assert_eq!(
        interpret("swap[{a_Integer, b_Integer}] := {b, a}; swap[{1, 2}]")
          .unwrap(),
        "{2, 1}"
      );
    }

    #[test]
    fn list_pattern_with_computation() {
      clear_state();
      assert_eq!(
        interpret("f[{x_Integer, y_Integer}] := x + y; f[{3, 4}]").unwrap(),
        "7"
      );
    }
  }

  mod newline_statements {
    use super::*;

    #[test]
    fn multiline_assignments() {
      clear_state();
      assert_eq!(interpret("x = 5\ny = 10\nx + y").unwrap(), "15");
    }

    #[test]
    fn multiline_with_blank_lines() {
      clear_state();
      assert_eq!(interpret("x = 42\n\nx").unwrap(), "42");
    }

    #[test]
    fn multiline_preserves_continuation() {
      clear_state();
      // A function definition spanning lines should still work
      assert_eq!(interpret("f[x_] :=\n  x + 1\nf[5]").unwrap(), "6");
    }
  }

  // ─── Polynomial Functions ─────────────────────────────────────────

  mod polynomial_q {
    use super::*;

    #[test]
    fn basic_polynomial() {
      assert_eq!(interpret("PolynomialQ[x^2 + 1, x]").unwrap(), "True");
      assert_eq!(
        interpret("PolynomialQ[3*x^3 + 2*x + 1, x]").unwrap(),
        "True"
      );
    }

    #[test]
    fn constant_is_polynomial() {
      assert_eq!(interpret("PolynomialQ[5, x]").unwrap(), "True");
    }

    #[test]
    fn variable_is_polynomial() {
      assert_eq!(interpret("PolynomialQ[x, x]").unwrap(), "True");
    }

    #[test]
    fn non_polynomial() {
      assert_eq!(interpret("PolynomialQ[Sin[x], x]").unwrap(), "False");
      assert_eq!(interpret("PolynomialQ[1/x, x]").unwrap(), "False");
    }

    #[test]
    fn multivariate() {
      assert_eq!(interpret("PolynomialQ[x^2 + y, x]").unwrap(), "True");
    }
  }

  mod exponent {
    use super::*;

    #[test]
    fn basic_exponent() {
      assert_eq!(interpret("Exponent[x^3 + x, x]").unwrap(), "3");
      assert_eq!(interpret("Exponent[x^2 + 3*x + 2, x]").unwrap(), "2");
    }

    #[test]
    fn constant_exponent() {
      assert_eq!(interpret("Exponent[5, x]").unwrap(), "0");
    }

    #[test]
    fn linear_exponent() {
      assert_eq!(interpret("Exponent[3*x + 1, x]").unwrap(), "1");
    }
  }

  mod coefficient {
    use super::*;

    #[test]
    fn quadratic_coefficients() {
      assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 2]").unwrap(), "1");
      assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 1]").unwrap(), "3");
      assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x, 0]").unwrap(), "2");
    }

    #[test]
    fn default_power_is_one() {
      assert_eq!(interpret("Coefficient[x^2 + 3*x + 2, x]").unwrap(), "3");
    }

    #[test]
    fn symbolic_coefficients() {
      assert_eq!(
        interpret("Coefficient[a*x^2 + b*x + c, x, 2]").unwrap(),
        "a"
      );
      assert_eq!(
        interpret("Coefficient[a*x^2 + b*x + c, x, 1]").unwrap(),
        "b"
      );
      assert_eq!(
        interpret("Coefficient[a*x^2 + b*x + c, x, 0]").unwrap(),
        "c"
      );
    }

    #[test]
    fn zero_coefficient() {
      assert_eq!(interpret("Coefficient[x^2 + 1, x, 1]").unwrap(), "0");
    }
  }

  mod expand {
    use super::*;

    #[test]
    fn simple_product() {
      assert_eq!(
        interpret("Expand[(x + 1)*(x + 2)]").unwrap(),
        "2 + 3*x + x^2"
      );
    }

    #[test]
    fn square() {
      assert_eq!(interpret("Expand[(x + 1)^2]").unwrap(), "1 + 2*x + x^2");
    }

    #[test]
    fn cube() {
      assert_eq!(
        interpret("Expand[(x + 1)^3]").unwrap(),
        "1 + 3*x + 3*x^2 + x^3"
      );
    }

    #[test]
    fn distribute() {
      assert_eq!(interpret("Expand[x*(x + 1)]").unwrap(), "x + x^2");
    }

    #[test]
    fn already_expanded() {
      assert_eq!(interpret("Expand[x^2 + 3*x + 2]").unwrap(), "2 + 3*x + x^2");
    }

    #[test]
    fn constant() {
      assert_eq!(interpret("Expand[5]").unwrap(), "5");
    }

    #[test]
    fn difference_of_squares() {
      assert_eq!(interpret("Expand[(x + 2)*(x - 2)]").unwrap(), "-4 + x^2");
    }

    #[test]
    fn multivariate_two_vars() {
      assert_eq!(interpret("Expand[(x + y)^2]").unwrap(), "x^2 + 2*x*y + y^2");
    }

    #[test]
    fn multivariate_four_vars() {
      assert_eq!(
        interpret("Expand[(a + b)*(c + d)]").unwrap(),
        "a*c + b*c + a*d + b*d"
      );
    }

    #[test]
    fn multivariate_with_constant() {
      assert_eq!(
        interpret("Expand[(x + y + 1)^2]").unwrap(),
        "1 + 2*x + x^2 + 2*y + 2*x*y + y^2"
      );
    }
  }

  mod simplify {
    use super::*;

    #[test]
    fn combine_like_terms() {
      assert_eq!(interpret("Simplify[x + x]").unwrap(), "2*x");
    }

    #[test]
    fn combine_powers() {
      assert_eq!(interpret("Simplify[x*x]").unwrap(), "x^2");
    }

    #[test]
    fn cancel_division() {
      assert_eq!(interpret("Simplify[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
    }

    #[test]
    fn trivial() {
      assert_eq!(interpret("Simplify[5]").unwrap(), "5");
      assert_eq!(interpret("Simplify[x]").unwrap(), "x");
    }
  }

  mod factor {
    use super::*;

    #[test]
    fn quadratic() {
      assert_eq!(
        interpret("Factor[x^2 + 3*x + 2]").unwrap(),
        "(1 + x)*(2 + x)"
      );
    }

    #[test]
    fn difference_of_squares() {
      assert_eq!(interpret("Factor[x^2 - 4]").unwrap(), "(-2 + x)*(2 + x)");
    }

    #[test]
    fn with_common_factor() {
      assert_eq!(
        interpret("Factor[2*x^2 + 6*x + 4]").unwrap(),
        "2*(1 + x)*(2 + x)"
      );
    }

    #[test]
    fn irreducible() {
      assert_eq!(interpret("Factor[x^2 + 1]").unwrap(), "1 + x^2");
    }

    #[test]
    fn cubic() {
      assert_eq!(
        interpret("Factor[x^3 - 1]").unwrap(),
        "(-1 + x)*(1 + x + x^2)"
      );
    }

    #[test]
    fn linear() {
      assert_eq!(interpret("Factor[2*x + 4]").unwrap(), "2*(2 + x)");
    }

    #[test]
    fn cyclotomic_x6_minus_1() {
      assert_eq!(
        interpret("Factor[x^6 - 1]").unwrap(),
        "(-1 + x)*(1 + x)*(1 - x + x^2)*(1 + x + x^2)"
      );
    }

    #[test]
    fn cyclotomic_x12_minus_1() {
      assert_eq!(
        interpret("Factor[x^12 - 1]").unwrap(),
        "(-1 + x)*(1 + x)*(1 + x^2)*(1 - x + x^2)*(1 + x + x^2)*(1 - x^2 + x^4)"
      );
    }

    #[test]
    fn cyclotomic_x100_minus_1() {
      assert_eq!(
        interpret("Factor[x^100 - 1]").unwrap(),
        "(-1 + x)*(1 + x)*(1 + x^2)*(1 - x + x^2 - x^3 + x^4)*(1 + x + x^2 + x^3 + x^4)*(1 - x^2 + x^4 - x^6 + x^8)*(1 - x^5 + x^10 - x^15 + x^20)*(1 + x^5 + x^10 + x^15 + x^20)*(1 - x^10 + x^20 - x^30 + x^40)"
      );
    }

    #[test]
    fn irreducible_x4_plus_1() {
      assert_eq!(interpret("Factor[x^4 + 1]").unwrap(), "1 + x^4");
    }

    #[test]
    fn cyclotomic_x4_minus_1() {
      assert_eq!(
        interpret("Factor[x^4 - 1]").unwrap(),
        "(-1 + x)*(1 + x)*(1 + x^2)"
      );
    }
  }

  // ─── Control Flow ─────────────────────────────────────────────────

  mod switch {
    use super::*;

    #[test]
    fn basic_match() {
      assert_eq!(interpret("Switch[2, 1, a, 2, b, 3, c]").unwrap(), "b");
    }

    #[test]
    fn first_match() {
      assert_eq!(interpret("Switch[1, 1, a, 2, b, 3, c]").unwrap(), "a");
    }

    #[test]
    fn no_match_returns_unevaluated() {
      assert_eq!(
        interpret("Switch[4, 1, a, 2, b, 3, c]").unwrap(),
        "Switch[4, 1, a, 2, b, 3, c]"
      );
    }

    #[test]
    fn wildcard_match() {
      assert_eq!(interpret("Switch[4, 1, a, _, c]").unwrap(), "c");
    }

    #[test]
    fn evaluated_expression() {
      assert_eq!(interpret("Switch[1 + 1, 1, a, 2, b, 3, c]").unwrap(), "b");
    }
  }

  mod piecewise {
    use super::*;

    #[test]
    fn first_true() {
      assert_eq!(interpret("Piecewise[{{1, True}}]").unwrap(), "1");
    }

    #[test]
    fn second_true() {
      assert_eq!(
        interpret("Piecewise[{{1, False}, {2, True}}]").unwrap(),
        "2"
      );
    }

    #[test]
    fn default_value() {
      assert_eq!(
        interpret("Piecewise[{{1, False}, {2, False}}, 42]").unwrap(),
        "42"
      );
    }

    #[test]
    fn no_match_default_zero() {
      assert_eq!(interpret("Piecewise[{{1, False}}]").unwrap(), "0");
    }

    #[test]
    fn with_conditions() {
      clear_state();
      assert_eq!(
        interpret("x = 5; Piecewise[{{1, x < 0}, {2, x >= 0}}]").unwrap(),
        "2"
      );
    }
  }

  // ─── List Operations ──────────────────────────────────────────────

  mod append_to {
    use super::*;

    #[test]
    fn basic() {
      clear_state();
      assert_eq!(
        interpret("x = {1, 2, 3}; AppendTo[x, 4]").unwrap(),
        "{1, 2, 3, 4}"
      );
    }

    #[test]
    fn updates_variable() {
      clear_state();
      assert_eq!(
        interpret("x = {1, 2}; AppendTo[x, 3]; x").unwrap(),
        "{1, 2, 3}"
      );
    }
  }

  mod prepend_to {
    use super::*;

    #[test]
    fn basic() {
      clear_state();
      assert_eq!(
        interpret("x = {1, 2, 3}; PrependTo[x, 0]").unwrap(),
        "{0, 1, 2, 3}"
      );
    }

    #[test]
    fn updates_variable() {
      clear_state();
      assert_eq!(
        interpret("x = {2, 3}; PrependTo[x, 1]; x").unwrap(),
        "{1, 2, 3}"
      );
    }
  }

  mod tuples {
    use super::*;

    #[test]
    fn pairs() {
      assert_eq!(
        interpret("Tuples[{a, b}, 2]").unwrap(),
        "{{a, a}, {a, b}, {b, a}, {b, b}}"
      );
    }

    #[test]
    fn triples() {
      assert_eq!(
        interpret("Tuples[{0, 1}, 3]").unwrap(),
        "{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}}"
      );
    }

    #[test]
    fn singles() {
      assert_eq!(
        interpret("Tuples[{a, b, c}, 1]").unwrap(),
        "{{a}, {b}, {c}}"
      );
    }

    #[test]
    fn empty_tuple() {
      assert_eq!(interpret("Tuples[{a, b}, 0]").unwrap(), "{{}}");
    }
  }

  mod match_q {
    use super::*;

    #[test]
    fn head_matching() {
      assert_eq!(interpret("MatchQ[{1, 2, 3}, _List]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[42, _Integer]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[3.14, _Real]").unwrap(), "True");
      assert_eq!(interpret(r#"MatchQ["hello", _String]"#).unwrap(), "True");
    }

    #[test]
    fn head_mismatch() {
      assert_eq!(interpret("MatchQ[1, _String]").unwrap(), "False");
      assert_eq!(interpret("MatchQ[1, _List]").unwrap(), "False");
      assert_eq!(interpret(r#"MatchQ["x", _Integer]"#).unwrap(), "False");
    }

    #[test]
    fn blank_matches_anything() {
      assert_eq!(interpret("MatchQ[42, _]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[{1, 2}, _]").unwrap(), "True");
      assert_eq!(interpret(r#"MatchQ["x", _]"#).unwrap(), "True");
    }

    #[test]
    fn literal_matching() {
      assert_eq!(interpret("MatchQ[42, 42]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[42, 43]").unwrap(), "False");
      assert_eq!(interpret("MatchQ[x, x]").unwrap(), "True");
      assert_eq!(interpret("MatchQ[x, y]").unwrap(), "False");
    }
  }

  mod counts {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(
        interpret("Counts[{a, b, a, c, b, a}]").unwrap(),
        "<|a -> 3, b -> 2, c -> 1|>"
      );
    }

    #[test]
    fn integers() {
      assert_eq!(
        interpret("Counts[{1, 2, 1, 3, 2, 1}]").unwrap(),
        "<|1 -> 3, 2 -> 2, 3 -> 1|>"
      );
    }

    #[test]
    fn single_element() {
      assert_eq!(interpret("Counts[{x}]").unwrap(), "<|x -> 1|>");
    }

    #[test]
    fn all_same() {
      assert_eq!(interpret("Counts[{a, a, a}]").unwrap(), "<|a -> 3|>");
    }
  }

  mod clip {
    use super::*;

    #[test]
    fn clip_above() {
      assert_eq!(interpret("Clip[15, {0, 10}]").unwrap(), "10");
    }

    #[test]
    fn clip_below() {
      assert_eq!(interpret("Clip[-5, {0, 10}]").unwrap(), "0");
    }

    #[test]
    fn clip_within() {
      assert_eq!(interpret("Clip[5, {0, 10}]").unwrap(), "5");
    }

    #[test]
    fn clip_default() {
      // Clip[x
      assert_eq!(interpret("Clip[1.5]").unwrap(), "1");
      assert_eq!(interpret("Clip[-0.5]").unwrap(), "-0.5");
      assert_eq!(interpret("Clip[0.5]").unwrap(), "0.5");
      assert_eq!(interpret("Clip[5]").unwrap(), "1");
      assert_eq!(interpret("Clip[-5]").unwrap(), "-1");
    }

    #[test]
    fn clip_boundaries() {
      assert_eq!(interpret("Clip[0, {0, 10}]").unwrap(), "0");
      assert_eq!(interpret("Clip[10, {0, 10}]").unwrap(), "10");
    }
  }

  mod ordering {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(interpret("Ordering[{3, 1, 2}]").unwrap(), "{2, 3, 1}");
    }

    #[test]
    fn with_limit() {
      assert_eq!(interpret("Ordering[{3, 1, 2}, 2]").unwrap(), "{2, 3}");
    }

    #[test]
    fn already_sorted() {
      assert_eq!(interpret("Ordering[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
    }

    #[test]
    fn reverse_sorted() {
      assert_eq!(interpret("Ordering[{3, 2, 1}]").unwrap(), "{3, 2, 1}");
    }

    #[test]
    fn single_element() {
      assert_eq!(interpret("Ordering[{5}]").unwrap(), "{1}");
    }

    #[test]
    fn strings() {
      assert_eq!(interpret("Ordering[{c, a, b}]").unwrap(), "{2, 3, 1}");
    }
  }

  mod minimal_by {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(
        interpret("MinimalBy[{-3, 1, 2, -1}, Abs]").unwrap(),
        "{1, -1}"
      );
    }

    #[test]
    fn single_min() {
      assert_eq!(
        interpret("MinimalBy[{5, 3, 7, 1, 4}, Identity]").unwrap(),
        "{1}"
      );
    }

    #[test]
    fn with_anonymous_function() {
      assert_eq!(
        interpret("MinimalBy[{10, 21, 32, 43}, Mod[#, 10] &]").unwrap(),
        "{10}"
      );
    }
  }

  mod maximal_by {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(interpret("MaximalBy[{-3, 1, 2, -1}, Abs]").unwrap(), "{-3}");
    }

    #[test]
    fn string_length() {
      assert_eq!(
        interpret(r#"MaximalBy[{"abc", "x", "ab"}, StringLength]"#).unwrap(),
        "{abc}"
      );
    }

    #[test]
    fn single_max() {
      assert_eq!(
        interpret("MaximalBy[{5, 3, 7, 1, 4}, Identity]").unwrap(),
        "{7}"
      );
    }
  }

  mod map_at {
    use super::*;

    #[test]
    fn single_position() {
      assert_eq!(
        interpret("MapAt[f, {a, b, c, d}, 2]").unwrap(),
        "{a, f[b], c, d}"
      );
    }

    #[test]
    fn negative_position() {
      assert_eq!(
        interpret("MapAt[f, {a, b, c, d}, -1]").unwrap(),
        "{a, b, c, f[d]}"
      );
    }

    #[test]
    fn multiple_positions() {
      // Wolfram uses {{1}, {3}} for multiple positions
      assert_eq!(
        interpret("MapAt[f, {a, b, c, d}, {{1}, {3}}]").unwrap(),
        "{f[a], b, f[c], d}"
      );
    }

    #[test]
    fn first_and_last() {
      assert_eq!(
        interpret("MapAt[f, {a, b, c}, {{1}, {-1}}]").unwrap(),
        "{f[a], b, f[c]}"
      );
    }

    #[test]
    fn with_anonymous_function() {
      assert_eq!(
        interpret("MapAt[# + 1 &, {10, 20, 30}, 2]").unwrap(),
        "{10, 21, 30}"
      );
    }
  }

  mod key_sort {
    use super::*;

    #[test]
    fn string_keys() {
      assert_eq!(
        interpret("KeySort[<|c -> 3, a -> 1, b -> 2|>]").unwrap(),
        "<|a -> 1, b -> 2, c -> 3|>"
      );
    }

    #[test]
    fn integer_keys() {
      assert_eq!(
        interpret("KeySort[<|3 -> c, 1 -> a, 2 -> b|>]").unwrap(),
        "<|1 -> a, 2 -> b, 3 -> c|>"
      );
    }

    #[test]
    fn already_sorted() {
      assert_eq!(
        interpret("KeySort[<|a -> 1, b -> 2|>]").unwrap(),
        "<|a -> 1, b -> 2|>"
      );
    }
  }

  mod key_value_map {
    use super::*;

    #[test]
    fn named_function() {
      assert_eq!(
        interpret("KeyValueMap[f, <|a -> 1, b -> 2|>]").unwrap(),
        "{f[a, 1], f[b, 2]}"
      );
    }

    #[test]
    fn list_function() {
      assert_eq!(
        interpret("KeyValueMap[List, <|a -> 1, b -> 2|>]").unwrap(),
        "{{a, 1}, {b, 2}}"
      );
    }

    #[test]
    fn anonymous_function() {
      assert_eq!(
        interpret("KeyValueMap[#2 &, <|x -> 10, y -> 20|>]").unwrap(),
        "{10, 20}"
      );
    }
  }

  mod random_choice {
    use super::*;

    #[test]
    fn single_choice() {
      let result = interpret("RandomChoice[{a, b, c}]").unwrap();
      assert!(result == "a" || result == "b" || result == "c");
    }

    #[test]
    fn multiple_choices() {
      assert_eq!(
        interpret("Length[RandomChoice[{1, 2, 3}, 10]]").unwrap(),
        "10"
      );
    }

    #[test]
    fn single_element_list() {
      assert_eq!(interpret("RandomChoice[{x}]").unwrap(), "x");
      assert_eq!(interpret("RandomChoice[{x}, 3]").unwrap(), "{x, x, x}");
    }
  }

  mod random_sample {
    use super::*;

    #[test]
    fn sample_count() {
      assert_eq!(
        interpret("Length[RandomSample[{1, 2, 3, 4, 5}, 3]]").unwrap(),
        "3"
      );
    }

    #[test]
    fn full_permutation() {
      assert_eq!(interpret("Length[RandomSample[{a, b, c}]]").unwrap(), "3");
    }

    #[test]
    fn sample_one() {
      let result = interpret("RandomSample[{a, b, c}, 1]").unwrap();
      assert!(result == "{a}" || result == "{b}" || result == "{c}");
    }

    #[test]
    fn no_duplicates() {
      // RandomSample should return distinct elements
      assert_eq!(
        interpret("Length[DeleteDuplicates[RandomSample[{1, 2, 3, 4, 5}, 5]]]")
          .unwrap(),
        "5"
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

  mod constant_array {
    use super::*;

    #[test]
    fn simple_integer() {
      assert_eq!(interpret("ConstantArray[0, 3]").unwrap(), "{0, 0, 0}");
    }

    #[test]
    fn symbol() {
      assert_eq!(interpret("ConstantArray[x, 4]").unwrap(), "{x, x, x, x}");
    }

    #[test]
    fn nested_dimensions() {
      assert_eq!(
        interpret("ConstantArray[0, {2, 3}]").unwrap(),
        "{{0, 0, 0}, {0, 0, 0}}"
      );
    }

    #[test]
    fn zero_length() {
      assert_eq!(interpret("ConstantArray[1, 0]").unwrap(), "{}");
    }
  }

  mod from_digits {
    use super::*;

    #[test]
    fn base_10() {
      assert_eq!(interpret("FromDigits[{1, 2, 3}]").unwrap(), "123");
    }

    #[test]
    fn base_2() {
      assert_eq!(interpret("FromDigits[{1, 0, 1}, 2]").unwrap(), "5");
    }

    #[test]
    fn base_16() {
      assert_eq!(interpret("FromDigits[{1, 0}, 16]").unwrap(), "16");
    }

    #[test]
    fn single_digit() {
      assert_eq!(interpret("FromDigits[{7}]").unwrap(), "7");
    }

    #[test]
    fn empty_list() {
      assert_eq!(interpret("FromDigits[{}]").unwrap(), "0");
    }
  }

  mod complement {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(
        interpret("Complement[{1, 2, 3, 4}, {2, 4}]").unwrap(),
        "{1, 3}"
      );
    }

    #[test]
    fn no_overlap() {
      assert_eq!(
        interpret("Complement[{1, 2, 3}, {4, 5}]").unwrap(),
        "{1, 2, 3}"
      );
    }

    #[test]
    fn complete_overlap() {
      assert_eq!(interpret("Complement[{1, 2}, {1, 2}]").unwrap(), "{}");
    }

    #[test]
    fn multiple_exclusion_lists() {
      assert_eq!(
        interpret("Complement[{1, 2, 3, 4, 5}, {2}, {4}]").unwrap(),
        "{1, 3, 5}"
      );
    }
  }

  mod count {
    use super::*;

    #[test]
    fn count_integer() {
      assert_eq!(interpret("Count[{1, 2, 3, 2, 1}, 2]").unwrap(), "2");
    }

    #[test]
    fn count_zero_matches() {
      assert_eq!(interpret("Count[{1, 2, 3}, 4]").unwrap(), "0");
    }

    #[test]
    fn count_symbol() {
      assert_eq!(interpret("Count[{a, b, a, c, a}, a]").unwrap(), "3");
    }
  }

  mod sort_by {
    use super::*;

    #[test]
    fn sort_by_abs() {
      assert_eq!(
        interpret("SortBy[{-3, 1, -2, 4}, Abs]").unwrap(),
        "{1, -2, -3, 4}"
      );
    }

    #[test]
    fn sort_by_length() {
      assert_eq!(
        interpret(r#"SortBy[{{1, 2, 3}, {1}, {1, 2}}, Length]"#).unwrap(),
        "{{1}, {1, 2}, {1, 2, 3}}"
      );
    }

    #[test]
    fn sort_by_anonymous_function() {
      assert_eq!(
        interpret("SortBy[{5, 1, 3, 2, 4}, (0 - #) &]").unwrap(),
        "{5, 4, 3, 2, 1}"
      );
    }
  }

  mod floor {
    use super::*;

    #[test]
    fn positive_float() {
      assert_eq!(interpret("Floor[3.7]").unwrap(), "3");
    }

    #[test]
    fn negative_float() {
      assert_eq!(interpret("Floor[-2.3]").unwrap(), "-3");
    }

    #[test]
    fn integer_unchanged() {
      assert_eq!(interpret("Floor[5]").unwrap(), "5");
    }

    #[test]
    fn zero() {
      assert_eq!(interpret("Floor[0]").unwrap(), "0");
    }

    #[test]
    fn symbolic() {
      assert_eq!(interpret("Floor[x]").unwrap(), "Floor[x]");
    }
  }

  mod run {
    use super::*;

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn run_echo() {
      assert_eq!(interpret(r#"Run["true"]"#).unwrap(), "0");
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

  mod integer_string {
    use super::*;

    #[test]
    fn base_10() {
      assert_eq!(interpret("IntegerString[42]").unwrap(), "42");
    }

    #[test]
    fn base_16() {
      assert_eq!(interpret("IntegerString[255, 16]").unwrap(), "ff");
    }

    #[test]
    fn base_2() {
      assert_eq!(interpret("IntegerString[255, 2]").unwrap(), "11111111");
    }

    #[test]
    fn with_padding() {
      assert_eq!(interpret("IntegerString[255, 16, 4]").unwrap(), "00ff");
    }

    #[test]
    fn zero() {
      assert_eq!(interpret("IntegerString[0]").unwrap(), "0");
    }

    #[test]
    fn zero_base_2() {
      assert_eq!(interpret("IntegerString[0, 2]").unwrap(), "0");
    }
  }

  mod round {
    use super::*;

    #[test]
    fn round_integer() {
      assert_eq!(interpret("Round[3]").unwrap(), "3");
    }

    #[test]
    fn round_real() {
      assert_eq!(interpret("Round[2.6]").unwrap(), "3");
    }

    #[test]
    fn round_two_args() {
      assert_eq!(interpret("Round[3.14159, 0.01]").unwrap(), "3.14");
    }

    #[test]
    fn round_to_tens() {
      assert_eq!(interpret("Round[37, 10]").unwrap(), "40");
    }
  }

  mod degree_constant {
    use super::*;

    #[test]
    fn degree_symbolic() {
      assert_eq!(interpret("Degree").unwrap(), "Degree");
    }

    #[test]
    fn degree_numeric() {
      assert_eq!(interpret("N[Degree]").unwrap(), "0.017453292519943295");
    }

    #[test]
    fn degree_arithmetic() {
      assert_eq!(interpret("Sin[30 Degree]").unwrap(), "1/2");
    }

    #[test]
    fn sin_exact_values() {
      assert_eq!(interpret("Sin[0]").unwrap(), "0");
      assert_eq!(interpret("Sin[Pi/6]").unwrap(), "1/2");
      assert_eq!(interpret("Sin[Pi/4]").unwrap(), "1/Sqrt[2]");
      assert_eq!(interpret("Sin[Pi/3]").unwrap(), "Sqrt[3]/2");
      assert_eq!(interpret("Sin[Pi/2]").unwrap(), "1");
      assert_eq!(interpret("Sin[Pi]").unwrap(), "0");
      assert_eq!(interpret("Sin[2 Pi]").unwrap(), "0");
    }

    #[test]
    fn sin_exact_negative() {
      assert_eq!(interpret("Sin[210 Degree]").unwrap(), "-1/2");
      assert_eq!(interpret("Sin[270 Degree]").unwrap(), "-1");
    }

    #[test]
    fn cos_exact_values() {
      assert_eq!(interpret("Cos[0]").unwrap(), "1");
      assert_eq!(interpret("Cos[Pi/6]").unwrap(), "Sqrt[3]/2");
      assert_eq!(interpret("Cos[Pi/4]").unwrap(), "1/Sqrt[2]");
      assert_eq!(interpret("Cos[Pi/3]").unwrap(), "1/2");
      assert_eq!(interpret("Cos[Pi/2]").unwrap(), "0");
      assert_eq!(interpret("Cos[Pi]").unwrap(), "-1");
      assert_eq!(interpret("Cos[2 Pi]").unwrap(), "1");
    }

    #[test]
    fn cos_exact_negative() {
      assert_eq!(interpret("Cos[120 Degree]").unwrap(), "-1/2");
      assert_eq!(interpret("Cos[180 Degree]").unwrap(), "-1");
    }

    #[test]
    fn tan_exact_values() {
      assert_eq!(interpret("Tan[0]").unwrap(), "0");
      assert_eq!(interpret("Tan[Pi/6]").unwrap(), "1/Sqrt[3]");
      assert_eq!(interpret("Tan[Pi/4]").unwrap(), "1");
      assert_eq!(interpret("Tan[Pi/3]").unwrap(), "Sqrt[3]");
      assert_eq!(interpret("Tan[Pi]").unwrap(), "0");
    }

    #[test]
    fn tan_exact_negative() {
      assert_eq!(interpret("Tan[120 Degree]").unwrap(), "-Sqrt[3]");
      assert_eq!(interpret("Tan[135 Degree]").unwrap(), "-1");
    }

    #[test]
    fn tan_complex_infinity() {
      assert_eq!(interpret("Tan[Pi/2]").unwrap(), "ComplexInfinity");
      assert_eq!(interpret("Tan[90 Degree]").unwrap(), "ComplexInfinity");
      assert_eq!(interpret("Tan[270 Degree]").unwrap(), "ComplexInfinity");
      assert_eq!(interpret("Tan[450 Degree]").unwrap(), "ComplexInfinity");
      assert_eq!(interpret("Tan[3 Pi/2]").unwrap(), "ComplexInfinity");
    }

    #[test]
    fn sin_degree_all_quadrants() {
      assert_eq!(interpret("Sin[45 Degree]").unwrap(), "1/Sqrt[2]");
      assert_eq!(interpret("Sin[90 Degree]").unwrap(), "1");
      assert_eq!(interpret("Sin[120 Degree]").unwrap(), "Sqrt[3]/2");
      assert_eq!(interpret("Sin[135 Degree]").unwrap(), "1/Sqrt[2]");
      assert_eq!(interpret("Sin[150 Degree]").unwrap(), "1/2");
      assert_eq!(interpret("Sin[180 Degree]").unwrap(), "0");
      assert_eq!(interpret("Sin[360 Degree]").unwrap(), "0");
    }
  }

  mod e_constant {
    use super::*;

    #[test]
    fn e_symbolic() {
      assert_eq!(interpret("E").unwrap(), "E");
    }

    #[test]
    fn e_numeric() {
      assert_eq!(interpret("N[E]").unwrap(), "2.718281828459045");
    }

    #[test]
    fn e_comparison() {
      assert_eq!(interpret("E > 2").unwrap(), "True");
    }
  }

  mod pi_symbolic {
    use super::*;

    #[test]
    fn pi_stays_symbolic() {
      assert_eq!(interpret("Pi").unwrap(), "Pi");
    }

    #[test]
    fn pi_comparison() {
      assert_eq!(interpret("Pi > 3").unwrap(), "True");
    }

    #[test]
    fn pi_less() {
      assert_eq!(interpret("Pi < 4").unwrap(), "True");
    }
  }

  mod clear_all {
    use super::*;

    #[test]
    fn clear_variable() {
      assert_eq!(interpret("x = 5; ClearAll[x]; x").unwrap(), "x");
    }

    #[test]
    fn clear_multiple() {
      assert_eq!(
        interpret("a = 1; b = 2; ClearAll[a, b]; {a, b}").unwrap(),
        "{a, b}"
      );
    }

    #[test]
    fn clear_preserves_others() {
      assert_eq!(
        interpret("a = 1; b = 2; ClearAll[a]; {a, b}").unwrap(),
        "{a, 2}"
      );
    }
  }

  mod dimensions {
    use super::*;

    #[test]
    fn dimensions_2d() {
      assert_eq!(
        interpret("Dimensions[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
        "{2, 3}"
      );
    }

    #[test]
    fn dimensions_1d() {
      assert_eq!(interpret("Dimensions[{1, 2, 3}]").unwrap(), "{3}");
    }

    #[test]
    fn dimensions_3d() {
      assert_eq!(
        interpret("Dimensions[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}]").unwrap(),
        "{2, 2, 2}"
      );
    }

    #[test]
    fn dimensions_ragged() {
      assert_eq!(interpret("Dimensions[{{1, 2}, {3}}]").unwrap(), "{2}");
    }
  }

  mod delete {
    use super::*;

    #[test]
    fn delete_positive() {
      assert_eq!(interpret("Delete[{a, b, c, d}, 2]").unwrap(), "{a, c, d}");
    }

    #[test]
    fn delete_negative() {
      assert_eq!(interpret("Delete[{a, b, c, d}, -1]").unwrap(), "{a, b, c}");
    }

    #[test]
    fn delete_multiple() {
      assert_eq!(
        interpret("Delete[{a, b, c, d, e}, {{1}, {3}}]").unwrap(),
        "{b, d, e}"
      );
    }
  }

  mod ordered_q {
    use super::*;

    #[test]
    fn ordered_sorted() {
      assert_eq!(interpret("OrderedQ[{1, 2, 3}]").unwrap(), "True");
    }

    #[test]
    fn ordered_unsorted() {
      assert_eq!(interpret("OrderedQ[{3, 1, 2}]").unwrap(), "False");
    }

    #[test]
    fn ordered_equal() {
      assert_eq!(interpret("OrderedQ[{1, 1, 2}]").unwrap(), "True");
    }

    #[test]
    fn ordered_strings() {
      assert_eq!(
        interpret("OrderedQ[{\"a\", \"b\", \"c\"}]").unwrap(),
        "True"
      );
    }

    #[test]
    fn ordered_empty() {
      assert_eq!(interpret("OrderedQ[{}]").unwrap(), "True");
    }
  }

  mod composition {
    use super::*;

    #[test]
    fn composition_apply() {
      assert_eq!(
        interpret("Composition[StringLength, ToString][12345]").unwrap(),
        "5"
      );
    }

    #[test]
    fn composition_symbolic() {
      assert_eq!(interpret("Composition[f, g]").unwrap(), "f @* g");
    }

    #[test]
    fn composition_variable() {
      assert_eq!(
        interpret("f = Composition[StringLength, ToString]; f[12345]").unwrap(),
        "5"
      );
    }
  }

  mod value_q {
    use super::*;

    #[test]
    fn value_q_defined() {
      assert_eq!(interpret("x = 5; ValueQ[x]").unwrap(), "True");
    }

    #[test]
    fn value_q_undefined() {
      assert_eq!(interpret("ValueQ[undefined]").unwrap(), "False");
    }

    #[test]
    fn value_q_cleared() {
      assert_eq!(interpret("x = 5; ClearAll[x]; ValueQ[x]").unwrap(), "False");
    }
  }

  mod rule_display {
    use super::*;

    #[test]
    fn rule_display() {
      assert_eq!(interpret("Rule[a, b]").unwrap(), "a -> b");
    }

    #[test]
    fn rule_delayed_display() {
      assert_eq!(interpret("RuleDelayed[a, b]").unwrap(), "a :> b");
    }
  }

  mod hold_form {
    use super::*;

    #[test]
    fn hold_form_unevaluated() {
      assert_eq!(interpret("HoldForm[1 + 1]").unwrap(), "HoldForm[1 + 1]");
    }
  }

  mod numerator {
    use super::*;

    #[test]
    fn rational() {
      assert_eq!(interpret("Numerator[3/4]").unwrap(), "3");
    }

    #[test]
    fn integer() {
      assert_eq!(interpret("Numerator[5]").unwrap(), "5");
    }

    #[test]
    fn negative_rational() {
      assert_eq!(interpret("Numerator[-3/4]").unwrap(), "-3");
    }

    #[test]
    fn whole_number_rational() {
      assert_eq!(interpret("Numerator[6/3]").unwrap(), "2");
    }
  }

  mod denominator {
    use super::*;

    #[test]
    fn rational() {
      assert_eq!(interpret("Denominator[3/4]").unwrap(), "4");
    }

    #[test]
    fn integer() {
      assert_eq!(interpret("Denominator[5]").unwrap(), "1");
    }

    #[test]
    fn negative_rational() {
      assert_eq!(interpret("Denominator[-3/4]").unwrap(), "4");
    }

    #[test]
    fn reduced_form() {
      assert_eq!(interpret("Denominator[6/4]").unwrap(), "2");
    }
  }

  mod binomial {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(interpret("Binomial[10, 3]").unwrap(), "120");
    }

    #[test]
    fn zero_k() {
      assert_eq!(interpret("Binomial[5, 0]").unwrap(), "1");
    }

    #[test]
    fn k_equals_n() {
      assert_eq!(interpret("Binomial[5, 5]").unwrap(), "1");
    }

    #[test]
    fn k_greater_than_n() {
      assert_eq!(interpret("Binomial[3, 5]").unwrap(), "0");
    }

    #[test]
    fn negative_n() {
      assert_eq!(interpret("Binomial[-3, 2]").unwrap(), "6");
    }

    #[test]
    fn zero_zero() {
      assert_eq!(interpret("Binomial[0, 0]").unwrap(), "1");
    }

    #[test]
    fn choose_one() {
      assert_eq!(interpret("Binomial[7, 1]").unwrap(), "7");
    }

    #[test]
    fn large_n() {
      assert_eq!(interpret("Binomial[20, 10]").unwrap(), "184756");
    }
  }

  mod multinomial {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(interpret("Multinomial[2, 3, 4]").unwrap(), "1260");
    }

    #[test]
    fn two_args() {
      // Multinomial[a, b] = Binomial[a+b, a]
      assert_eq!(interpret("Multinomial[3, 4]").unwrap(), "35");
    }

    #[test]
    fn single_arg() {
      assert_eq!(interpret("Multinomial[5]").unwrap(), "1");
    }

    #[test]
    fn empty() {
      assert_eq!(interpret("Multinomial[]").unwrap(), "1");
    }

    #[test]
    fn all_ones() {
      // Multinomial[1,1,1] = 3! / (1!*1!*1!) = 6
      assert_eq!(interpret("Multinomial[1, 1, 1]").unwrap(), "6");
    }
  }

  mod power_mod {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(interpret("PowerMod[2, 10, 1000]").unwrap(), "24");
    }

    #[test]
    fn modular_inverse() {
      assert_eq!(interpret("PowerMod[3, -1, 7]").unwrap(), "5");
    }

    #[test]
    fn large_exponent() {
      assert_eq!(interpret("PowerMod[2, 100, 13]").unwrap(), "3");
    }

    #[test]
    fn base_zero() {
      assert_eq!(interpret("PowerMod[0, 5, 7]").unwrap(), "0");
    }

    #[test]
    fn exponent_zero() {
      assert_eq!(interpret("PowerMod[5, 0, 7]").unwrap(), "1");
    }

    #[test]
    fn negative_base() {
      assert_eq!(interpret("PowerMod[-2, 3, 7]").unwrap(), "6");
    }

    #[test]
    fn large_modulus() {
      // 2^127 - 1 is a Mersenne prime; by Fermat's little theorem, 2^(p-1) ≡ 1 (mod p)
      // so 2^p ≡ 2 (mod p), meaning PowerMod[2, 2^127-1, 2^127-1] = 2
      assert_eq!(interpret("PowerMod[2, 2^127 - 1, 2^127 - 1]").unwrap(), "2");
    }
  }

  mod prime_pi {
    use super::*;

    #[test]
    fn small_values() {
      assert_eq!(interpret("PrimePi[1]").unwrap(), "0");
      assert_eq!(interpret("PrimePi[2]").unwrap(), "1");
      assert_eq!(interpret("PrimePi[10]").unwrap(), "4");
      assert_eq!(interpret("PrimePi[100]").unwrap(), "25");
    }

    #[test]
    fn zero() {
      assert_eq!(interpret("PrimePi[0]").unwrap(), "0");
    }

    #[test]
    fn negative() {
      assert_eq!(interpret("PrimePi[-5]").unwrap(), "0");
    }

    #[test]
    fn at_prime() {
      // PrimePi[7] = 4 (primes: 2, 3, 5, 7)
      assert_eq!(interpret("PrimePi[7]").unwrap(), "4");
    }
  }

  mod next_prime {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(interpret("NextPrime[7]").unwrap(), "11");
    }

    #[test]
    fn from_composite() {
      assert_eq!(interpret("NextPrime[10]").unwrap(), "11");
    }

    #[test]
    fn from_one() {
      assert_eq!(interpret("NextPrime[1]").unwrap(), "2");
    }

    #[test]
    fn from_two() {
      assert_eq!(interpret("NextPrime[2]").unwrap(), "3");
    }

    #[test]
    fn from_zero() {
      assert_eq!(interpret("NextPrime[0]").unwrap(), "2");
    }

    #[test]
    fn from_negative() {
      assert_eq!(interpret("NextPrime[-5]").unwrap(), "-3");
    }

    #[test]
    fn big_integer_10_pow_100() {
      assert_eq!(
        interpret("NextPrime[10^100]").unwrap(),
        "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000267"
      );
    }

    #[test]
    fn big_integer_10_pow_50() {
      assert_eq!(
        interpret("NextPrime[10^50]").unwrap(),
        "100000000000000000000000000000000000000000000000151"
      );
    }

    #[test]
    fn big_integer_result_is_prime() {
      assert_eq!(interpret("PrimeQ[NextPrime[10^100]]").unwrap(), "True");
    }

    #[test]
    fn prime_q_big_integer() {
      assert_eq!(interpret("PrimeQ[10^100]").unwrap(), "False");
    }

    #[test]
    fn prime_q_big_prime() {
      assert_eq!(
        interpret("PrimeQ[10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000267]").unwrap(),
        "True"
      );
    }
  }

  mod bit_length {
    use super::*;

    #[test]
    fn basic_values() {
      assert_eq!(interpret("BitLength[0]").unwrap(), "0");
      assert_eq!(interpret("BitLength[1]").unwrap(), "1");
      assert_eq!(interpret("BitLength[2]").unwrap(), "2");
      assert_eq!(interpret("BitLength[7]").unwrap(), "3");
      assert_eq!(interpret("BitLength[8]").unwrap(), "4");
      assert_eq!(interpret("BitLength[255]").unwrap(), "8");
      assert_eq!(interpret("BitLength[256]").unwrap(), "9");
    }

    #[test]
    fn negative() {
      assert_eq!(interpret("BitLength[-1]").unwrap(), "0");
      assert_eq!(interpret("BitLength[-255]").unwrap(), "8");
    }
  }

  mod integer_part {
    use super::*;

    #[test]
    fn positive_float() {
      assert_eq!(interpret("IntegerPart[3.7]").unwrap(), "3");
    }

    #[test]
    fn negative_float() {
      assert_eq!(interpret("IntegerPart[-3.7]").unwrap(), "-3");
    }

    #[test]
    fn integer() {
      assert_eq!(interpret("IntegerPart[5]").unwrap(), "5");
    }

    #[test]
    fn rational() {
      assert_eq!(interpret("IntegerPart[7/3]").unwrap(), "2");
    }

    #[test]
    fn negative_rational() {
      assert_eq!(interpret("IntegerPart[-7/3]").unwrap(), "-2");
    }
  }

  mod fractional_part {
    use super::*;

    #[test]
    fn integer() {
      assert_eq!(interpret("FractionalPart[5]").unwrap(), "0");
    }

    #[test]
    fn rational() {
      assert_eq!(interpret("FractionalPart[7/3]").unwrap(), "1/3");
    }

    #[test]
    fn negative_rational() {
      assert_eq!(interpret("FractionalPart[-7/3]").unwrap(), "-1/3");
    }
  }

  mod chop {
    use super::*;

    #[test]
    fn small_number() {
      assert_eq!(interpret("Chop[0.00000000001]").unwrap(), "0");
    }

    #[test]
    fn normal_number() {
      assert_eq!(interpret("Chop[1.5]").unwrap(), "1.5");
    }

    #[test]
    fn zero() {
      assert_eq!(interpret("Chop[0]").unwrap(), "0");
    }

    #[test]
    fn negative_small() {
      assert_eq!(interpret("Chop[-0.00000000001]").unwrap(), "0");
    }

    #[test]
    fn list() {
      assert_eq!(
        interpret("Chop[{0.00000000001, 1.5, -0.000000000001}]").unwrap(),
        "{0, 1.5, 0}"
      );
    }

    #[test]
    fn custom_tolerance() {
      assert_eq!(interpret("Chop[0.05, 0.1]").unwrap(), "0");
      assert_eq!(interpret("Chop[0.5, 0.1]").unwrap(), "0.5");
    }
  }

  mod cube_root {
    use super::*;

    #[test]
    fn perfect_cube() {
      assert_eq!(interpret("CubeRoot[8]").unwrap(), "2");
    }

    #[test]
    fn negative_perfect_cube() {
      assert_eq!(interpret("CubeRoot[-27]").unwrap(), "-3");
    }

    #[test]
    fn zero() {
      assert_eq!(interpret("CubeRoot[0]").unwrap(), "0");
    }

    #[test]
    fn one() {
      assert_eq!(interpret("CubeRoot[1]").unwrap(), "1");
    }

    #[test]
    fn large_cube() {
      assert_eq!(interpret("CubeRoot[1000]").unwrap(), "10");
    }

    #[test]
    fn non_perfect_cube_symbolic() {
      // Non-perfect cubes return n^(1/3)
      assert_eq!(interpret("CubeRoot[2]").unwrap(), "2^(1/3)");
    }
  }

  mod subdivide {
    use super::*;

    #[test]
    fn unit_interval() {
      assert_eq!(interpret("Subdivide[4]").unwrap(), "{0, 1/4, 1/2, 3/4, 1}");
    }

    #[test]
    fn two_parts() {
      assert_eq!(interpret("Subdivide[2]").unwrap(), "{0, 1/2, 1}");
    }

    #[test]
    fn one_part() {
      assert_eq!(interpret("Subdivide[1]").unwrap(), "{0, 1}");
    }

    #[test]
    fn custom_range() {
      assert_eq!(
        interpret("Subdivide[0, 10, 5]").unwrap(),
        "{0, 2, 4, 6, 8, 10}"
      );
    }

    #[test]
    fn two_arg_form() {
      assert_eq!(
        interpret("Subdivide[10, 5]").unwrap(),
        "{0, 2, 4, 6, 8, 10}"
      );
    }
  }

  mod nothing {
    use super::*;

    #[test]
    fn filters_from_list() {
      assert_eq!(interpret("{1, Nothing, 3}").unwrap(), "{1, 3}");
    }

    #[test]
    fn multiple_nothing() {
      assert_eq!(
        interpret("{Nothing, 1, Nothing, 2, Nothing}").unwrap(),
        "{1, 2}"
      );
    }

    #[test]
    fn all_nothing() {
      assert_eq!(interpret("{Nothing, Nothing}").unwrap(), "{}");
    }

    #[test]
    fn nothing_standalone() {
      assert_eq!(interpret("Nothing").unwrap(), "Nothing");
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

  // ─── Linear Algebra ─────────────────────────────────────────────────

  mod dot {
    use super::*;

    #[test]
    fn vector_dot_product() {
      assert_eq!(interpret("Dot[{1, 2, 3}, {4, 5, 6}]").unwrap(), "32");
    }

    #[test]
    fn matrix_vector() {
      assert_eq!(
        interpret("Dot[{{1, 0}, {0, 1}}, {5, 6}]").unwrap(),
        "{5, 6}"
      );
    }

    #[test]
    fn matrix_matrix() {
      assert_eq!(
        interpret("Dot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}]").unwrap(),
        "{{19, 22}, {43, 50}}"
      );
    }

    #[test]
    fn dot_simple() {
      assert_eq!(interpret("Dot[{1, 2}, {3, 4}]").unwrap(), "11");
    }

    #[test]
    fn infix_dot_vector() {
      // Infix . operator should evaluate the same as Dot[]
      assert_eq!(interpret("{1, 2, 3} . {4, 5, 6}").unwrap(), "32");
    }

    #[test]
    fn infix_dot_matrix_vector() {
      assert_eq!(interpret("{{1, 2}, {3, 4}} . {5, 6}").unwrap(), "{17, 39}");
    }

    #[test]
    fn infix_dot_matrix_matrix() {
      assert_eq!(
        interpret("{{1, 2}, {3, 4}} . {{5, 6}, {7, 8}}").unwrap(),
        "{{19, 22}, {43, 50}}"
      );
    }

    #[test]
    fn infix_dot_symbolic() {
      // Symbolic dot returns unevaluated in infix form
      assert_eq!(interpret("a . b").unwrap(), "a . b");
    }

    #[test]
    fn dot_display_infix() {
      // Dot[a, b] should display as infix a . b
      assert_eq!(interpret("Dot[a, b]").unwrap(), "a . b");
    }

    #[test]
    fn identity_matrix_dot_symbolic_vector() {
      assert_eq!(
        interpret("Dot[IdentityMatrix[2], {x, y}]").unwrap(),
        "{x, y}"
      );
      assert_eq!(
        interpret("Dot[IdentityMatrix[3], {a, b, c}]").unwrap(),
        "{a, b, c}"
      );
    }

    #[test]
    fn matrix_dot_symbolic_vector() {
      assert_eq!(
        interpret("Dot[{{2, 0}, {0, 3}}, {x, y}]").unwrap(),
        "{2*x, 3*y}"
      );
    }
  }

  mod det {
    use super::*;

    #[test]
    fn det_2x2() {
      assert_eq!(interpret("Det[{{1, 2}, {3, 4}}]").unwrap(), "-2");
    }

    #[test]
    fn det_3x3() {
      assert_eq!(
        interpret("Det[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
        "6"
      );
    }

    #[test]
    fn det_identity() {
      assert_eq!(interpret("Det[{{1, 0}, {0, 1}}]").unwrap(), "1");
    }
  }

  mod inverse {
    use super::*;

    #[test]
    fn inverse_2x2() {
      assert_eq!(
        interpret("Inverse[{{1, 2}, {3, 4}}]").unwrap(),
        "{{-2, 1}, {3/2, -1/2}}"
      );
    }

    #[test]
    fn inverse_identity() {
      assert_eq!(
        interpret("Inverse[{{1, 0}, {0, 1}}]").unwrap(),
        "{{1, 0}, {0, 1}}"
      );
    }

    #[test]
    fn inverse_symbolic_2x2() {
      assert_eq!(
        interpret("Inverse[{{a, b}, {c, d}}]").unwrap(),
        "{{d/(-(b*c) + a*d), -(b/(-(b*c) + a*d))}, {-(c/(-(b*c) + a*d)), a/(-(b*c) + a*d)}}"
      );
    }

    #[test]
    fn inverse_times_identity() {
      // A . Inverse[A] should give identity matrix for numeric matrices
      assert_eq!(
        interpret("{{1, 2}, {3, 4}} . Inverse[{{1, 2}, {3, 4}}]").unwrap(),
        "{{1, 0}, {0, 1}}"
      );
    }
  }

  mod tr {
    use super::*;

    #[test]
    fn trace_2x2() {
      assert_eq!(interpret("Tr[{{1, 2}, {3, 4}}]").unwrap(), "5");
    }

    #[test]
    fn trace_3x3() {
      assert_eq!(
        interpret("Tr[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
        "6"
      );
    }

    #[test]
    fn trace_vector() {
      assert_eq!(interpret("Tr[{1, 2, 3}]").unwrap(), "6");
    }
  }

  mod identity_matrix {
    use super::*;

    #[test]
    fn identity_3() {
      assert_eq!(
        interpret("IdentityMatrix[3]").unwrap(),
        "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
      );
    }

    #[test]
    fn identity_1() {
      assert_eq!(interpret("IdentityMatrix[1]").unwrap(), "{{1}}");
    }
  }

  mod diagonal_matrix {
    use super::*;

    #[test]
    fn diagonal_basic() {
      assert_eq!(
        interpret("DiagonalMatrix[{1, 2, 3}]").unwrap(),
        "{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}"
      );
    }
  }

  mod cross {
    use super::*;

    #[test]
    fn cross_basic() {
      assert_eq!(
        interpret("Cross[{1, 2, 3}, {4, 5, 6}]").unwrap(),
        "{-3, 6, -3}"
      );
    }

    #[test]
    fn cross_unit_vectors() {
      assert_eq!(
        interpret("Cross[{1, 0, 0}, {0, 1, 0}]").unwrap(),
        "{0, 0, 1}"
      );
    }

    #[test]
    fn cross_symbolic() {
      // Negated products should be parenthesized: -(c*e) not -c*e
      let result = interpret("Cross[{a, b, c}, {d, e, f}]").unwrap();
      assert!(result.contains("-(c*e)"), "Expected -(c*e) in {}", result);
      assert!(result.contains("b*f"), "Expected b*f in {}", result);
      assert!(result.contains("-(b*d)"), "Expected -(b*d) in {}", result);
      assert!(result.contains("a*e"), "Expected a*e in {}", result);
    }

    #[test]
    fn cross_unit_vectors_yz() {
      assert_eq!(
        interpret("Cross[{0, 1, 0}, {0, 0, 1}]").unwrap(),
        "{1, 0, 0}"
      );
    }

    #[test]
    fn cross_unit_vectors_zx() {
      assert_eq!(
        interpret("Cross[{0, 0, 1}, {1, 0, 0}]").unwrap(),
        "{0, 1, 0}"
      );
    }

    #[test]
    fn cross_symbolic_return() {
      // Non-3-vectors should return unevaluated
      assert_eq!(
        interpret("Cross[{1, 2}, {3, 4}]").unwrap(),
        "Cross[{1, 2}, {3, 4}]"
      );
    }
  }

  // ─── CAS / Algebra ──────────────────────────────────────────────────

  mod cancel {
    use super::*;

    #[test]
    fn cancel_simple() {
      assert_eq!(interpret("Cancel[(x^2 - 1)/(x - 1)]").unwrap(), "1 + x");
    }

    #[test]
    fn cancel_cubic() {
      assert_eq!(interpret("Cancel[(x^3 - x)/(x^2 - 1)]").unwrap(), "x");
    }

    #[test]
    fn cancel_symbolic_common_factor() {
      assert_eq!(interpret("Cancel[(a*b)/(a*c)]").unwrap(), "b/c");
    }

    #[test]
    fn cancel_symbolic_powers() {
      assert_eq!(interpret("Cancel[(a^2*b)/(a*b^2)]").unwrap(), "a/b");
    }

    #[test]
    fn cancel_numeric_content() {
      assert_eq!(interpret("Cancel[(2*x)/(4*x)]").unwrap(), "1/2");
    }

    #[test]
    fn cancel_mixed_symbolic_and_poly() {
      assert_eq!(interpret("Cancel[(a*b*x)/(a*c*x^2)]").unwrap(), "b/(c*x)");
    }

    #[test]
    fn cancel_quadratic() {
      assert_eq!(
        interpret("Cancel[(x^2 + 2*x + 1)/(x + 1)]").unwrap(),
        "1 + x"
      );
    }
  }

  mod expand_all {
    use super::*;

    #[test]
    fn expand_all_basic() {
      assert_eq!(
        interpret("ExpandAll[x*(x + 1)^2]").unwrap(),
        "x + 2*x^2 + x^3"
      );
    }
  }

  mod collect_tests {
    use super::*;

    #[test]
    fn collect_basic() {
      assert_eq!(interpret("Collect[x*y + x*z, x]").unwrap(), "x*(y + z)");
    }

    #[test]
    fn collect_symbolic_coefficients() {
      // Coefficients with variables before the collect variable go first
      assert_eq!(
        interpret("Collect[a*x^2 + b*x + c*x^2 + d*x, x]").unwrap(),
        "(b + d)*x + (a + c)*x^2"
      );
    }

    #[test]
    fn collect_mixed_coefficients() {
      // Coefficient (2+y) has y > x alphabetically, so x goes first
      assert_eq!(
        interpret("Collect[x*y + 2*x + 3*y + 6, x]").unwrap(),
        "6 + 3*y + x*(2 + y)"
      );
    }

    #[test]
    fn collect_with_constant_term() {
      assert_eq!(
        interpret("Collect[a*x^2 + b*x^2 + c*x + d*x + e, x]").unwrap(),
        "e + (c + d)*x + (a + b)*x^2"
      );
    }
  }

  mod together {
    use super::*;

    #[test]
    fn together_basic() {
      assert_eq!(interpret("Together[1/x + 1/y]").unwrap(), "(x + y)/(x*y)");
    }

    #[test]
    fn together_symbolic_fractions() {
      assert_eq!(
        interpret("Together[a/b + c/d]").unwrap(),
        "(b*c + a*d)/(b*d)"
      );
    }

    #[test]
    fn together_subtracted_fractions() {
      assert_eq!(
        interpret("Together[1/(x-1) - 1/(x+1)]").unwrap(),
        "2/((-1 + x)*(1 + x))"
      );
    }

    #[test]
    fn together_added_fractions_with_binomial_denominators() {
      assert_eq!(
        interpret("Together[1/(x-1) + 1/(x+1)]").unwrap(),
        "(2*x)/((-1 + x)*(1 + x))"
      );
    }
  }

  mod apart {
    use super::*;

    #[test]
    fn apart_basic() {
      assert_eq!(
        interpret("Apart[1/(x^2 - 1)]").unwrap(),
        "1/(2*(-1 + x)) - 1/(2*(1 + x))"
      );
    }

    #[test]
    fn apart_x2_plus_1_over_x3_minus_x() {
      assert_eq!(
        interpret("Apart[(x^2 + 1)/(x^3 - x)]").unwrap(),
        "(-1 + x)^(-1) - x^(-1) + (1 + x)^(-1)"
      );
    }

    #[test]
    fn apart_two_linear_factors() {
      assert_eq!(
        interpret("Apart[1/((x - 1)*(x - 2))]").unwrap(),
        "(-2 + x)^(-1) - (-1 + x)^(-1)"
      );
    }

    #[test]
    fn apart_three_linear_factors() {
      assert_eq!(
        interpret("Apart[1/((x-1)*(x-2)*(x-3))]").unwrap(),
        "1/(2*(-3 + x)) - (-2 + x)^(-1) + 1/(2*(-1 + x))"
      );
    }
  }

  // ─── Calculus ───────────────────────────────────────────────────────

  mod limit {
    use super::*;

    #[test]
    fn limit_sin_x_over_x() {
      assert_eq!(interpret("Limit[Sin[x]/x, x -> 0]").unwrap(), "1");
    }

    #[test]
    fn limit_direct_substitution() {
      assert_eq!(interpret("Limit[x^2, x -> 3]").unwrap(), "9");
    }

    #[test]
    fn limit_compound_interest() {
      assert_eq!(interpret("Limit[(1 + 1/n)^n, n -> Infinity]").unwrap(), "E");
    }

    #[test]
    fn limit_compound_interest_general() {
      assert_eq!(
        interpret("Limit[(1 + 2/n)^n, n -> Infinity]").unwrap(),
        "E^2"
      );
    }

    #[test]
    fn limit_one_over_n() {
      assert_eq!(interpret("Limit[1/n, n -> Infinity]").unwrap(), "0");
    }

    #[test]
    fn limit_n_to_infinity() {
      assert_eq!(interpret("Limit[n, n -> Infinity]").unwrap(), "Infinity");
    }
  }

  mod differentiate_plus_times {
    use super::*;

    #[test]
    fn d_x_plus_one() {
      assert_eq!(interpret("D[x + 1, x]").unwrap(), "1");
    }

    #[test]
    fn d_x_squared_plus_x() {
      assert_eq!(interpret("D[x^2 + x, x]").unwrap(), "1 + 2*x");
    }

    #[test]
    fn d_log_one_plus_t() {
      assert_eq!(interpret("D[Log[1 + t], t]").unwrap(), "1/(1 + t)");
    }
  }

  mod series {
    use super::*;

    #[test]
    fn series_exp() {
      assert_eq!(
        interpret("Series[Exp[x], {x, 0, 3}]").unwrap(),
        "SeriesData[x, 0, {1, 1, 1/2, 1/6}, 0, 4, 1]"
      );
    }

    #[test]
    fn series_exp_order4() {
      assert_eq!(
        interpret("Series[Exp[x], {x, 0, 4}]").unwrap(),
        "SeriesData[x, 0, {1, 1, 1/2, 1/6, 1/24}, 0, 5, 1]"
      );
    }

    #[test]
    fn series_sin_strips_leading_zero() {
      assert_eq!(
        interpret("Series[Sin[x], {x, 0, 5}]").unwrap(),
        "SeriesData[x, 0, {1, 0, -1/6, 0, 1/120}, 1, 6, 1]"
      );
    }

    #[test]
    fn series_cos_no_leading_zero() {
      assert_eq!(
        interpret("Series[Cos[x], {x, 0, 6}]").unwrap(),
        "SeriesData[x, 0, {1, 0, -1/2, 0, 1/24, 0, -1/720}, 0, 7, 1]"
      );
    }

    #[test]
    fn series_log_around_1() {
      assert_eq!(
        interpret("Series[Log[x], {x, 1, 3}]").unwrap(),
        "SeriesData[x, 1, {1, -1/2, 1/3}, 1, 4, 1]"
      );
    }

    #[test]
    fn series_geometric() {
      assert_eq!(
        interpret("Series[1/(1 - x), {x, 0, 3}]").unwrap(),
        "SeriesData[x, 0, {1, 1, 1, 1}, 0, 4, 1]"
      );
    }

    #[test]
    fn series_zero_returns_zero() {
      assert_eq!(interpret("Series[0, {x, 0, 3}]").unwrap(), "0");
    }
  }

  // ─── Association Operations ─────────────────────────────────────────

  mod association_thread {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(
        interpret("AssociationThread[{a, b}, {1, 2}]").unwrap(),
        "<|a -> 1, b -> 2|>"
      );
    }
  }

  mod association_map {
    use super::*;

    #[test]
    fn reverse_on_association() {
      assert_eq!(
        interpret("AssociationMap[Reverse, <|a -> 1, b -> 2|>]").unwrap(),
        "<|1 -> a, 2 -> b|>"
      );
    }

    #[test]
    fn symbolic_function_on_association() {
      assert_eq!(
        interpret("AssociationMap[f, <|x -> 10, y -> 20|>]").unwrap(),
        "Association[f[x -> 10], f[y -> 20]]"
      );
    }

    #[test]
    fn on_list() {
      assert_eq!(
        interpret("AssociationMap[StringLength, {\"cat\", \"horse\", \"ox\"}]")
          .unwrap(),
        "<|cat -> 3, horse -> 5, ox -> 2|>"
      );
    }
  }

  mod reverse_rule {
    use super::*;

    #[test]
    fn reverse_rule() {
      assert_eq!(interpret("Reverse[a -> b]").unwrap(), "b -> a");
    }

    #[test]
    fn reverse_rule_numeric() {
      assert_eq!(interpret("Reverse[1 -> 2]").unwrap(), "2 -> 1");
    }
  }

  mod merge {
    use super::*;

    #[test]
    fn merge_with_total() {
      assert_eq!(
        interpret("Merge[{<|a -> 1|>, <|a -> 2, b -> 3|>}, Total]").unwrap(),
        "<|a -> 3, b -> 3|>"
      );
    }
  }

  mod key_map {
    use super::*;

    #[test]
    fn key_map_basic() {
      assert_eq!(
        interpret("KeyMap[f, <|a -> 1, b -> 2|>]").unwrap(),
        "<|f[a] -> 1, f[b] -> 2|>"
      );
    }
  }

  mod key_select {
    use super::*;

    #[test]
    fn key_select_even() {
      assert_eq!(
        interpret("KeySelect[<|1 -> a, 2 -> b, 3 -> c|>, EvenQ]").unwrap(),
        "<|2 -> b|>"
      );
    }

    #[test]
    fn key_select_string_starts_q() {
      assert_eq!(
        interpret(
          "KeySelect[<|\"apple\" -> 1, \"ax\" -> 2, \"banana\" -> 3|>, StringStartsQ[\"a\"]]"
        )
        .unwrap(),
        "<|apple -> 1, ax -> 2|>"
      );
    }

    #[test]
    fn key_select_string_ends_q() {
      assert_eq!(
        interpret(
          "KeySelect[<|\"hello\" -> 1, \"world\" -> 2, \"foo\" -> 3|>, StringEndsQ[\"o\"]]"
        )
        .unwrap(),
        "<|hello -> 1, foo -> 3|>"
      );
    }

    #[test]
    fn key_select_anonymous_function() {
      assert_eq!(
        interpret("KeySelect[<|-3 -> x, 0 -> y, 5 -> z|>, # > 0 &]").unwrap(),
        "<|5 -> z|>"
      );
    }
  }

  mod key_take {
    use super::*;

    #[test]
    fn key_take_basic() {
      assert_eq!(
        interpret("KeyTake[<|a -> 1, b -> 2, c -> 3|>, {a, c}]").unwrap(),
        "<|a -> 1, c -> 3|>"
      );
    }
  }

  mod key_drop {
    use super::*;

    #[test]
    fn key_drop_basic() {
      assert_eq!(
        interpret("KeyDrop[<|a -> 1, b -> 2, c -> 3|>, {a}]").unwrap(),
        "<|b -> 2, c -> 3|>"
      );
    }
  }

  // ─── Utility Functions ──────────────────────────────────────────────

  mod reap_sow {
    use super::*;

    #[test]
    fn reap_sow_basic() {
      assert_eq!(
        interpret("Reap[Sow[1]; Sow[2]; 42]").unwrap(),
        "{42, {{1, 2}}}"
      );
    }

    #[test]
    fn reap_without_sow() {
      assert_eq!(interpret("Reap[42]").unwrap(), "{42, {}}");
    }

    #[test]
    fn sow_returns_value() {
      assert_eq!(interpret("Sow[7]").unwrap(), "7");
    }

    #[test]
    fn sow_with_tag() {
      assert_eq!(
        interpret(r#"Reap[Sow[1, "a"]; Sow[2, "b"]; Sow[3, "a"]]"#).unwrap(),
        "{3, {{1, 3}, {2}}}"
      );
    }

    #[test]
    fn reap_with_single_tag_pattern() {
      assert_eq!(
        interpret(r#"Reap[Sow[1, "a"]; Sow[2, "b"]; Sow[3, "a"], "a"]"#)
          .unwrap(),
        "{3, {{1, 3}}}"
      );
    }

    #[test]
    fn reap_with_tag_list() {
      assert_eq!(
        interpret(r#"Reap[Sow[1, "a"]; Sow[2, "b"]; Sow[3, "a"], {"a", "b"}]"#)
          .unwrap(),
        "{3, {{{1, 3}}, {{2}}}}"
      );
    }

    #[test]
    fn reap_tagged_with_do_loop() {
      assert_eq!(
        interpret(r#"Reap[Do[If[EvenQ[i], Sow[i, "even"], Sow[i, "odd"]], {i, 6}], {"even", "odd"}]"#).unwrap(),
        "{Null, {{{2, 4, 6}}, {{1, 3, 5}}}}"
      );
    }

    #[test]
    fn reap_mixed_tagged_untagged() {
      assert_eq!(
        interpret(r#"Reap[Sow[1]; Sow[2, "a"]; Sow[3]]"#).unwrap(),
        "{3, {{1, 3}, {2}}}"
      );
    }

    #[test]
    fn reap_with_none_tag() {
      assert_eq!(
        interpret("Reap[Sow[1]; Sow[2]; Sow[3], None]").unwrap(),
        "{3, {{1, 2, 3}}}"
      );
    }

    #[test]
    fn reap_tag_list_with_no_matches() {
      assert_eq!(
        interpret(r#"Reap[Sow[1, "a"], {"b"}]"#).unwrap(),
        "{1, {{}}}"
      );
    }
  }

  mod unitize {
    use super::*;

    #[test]
    fn unitize_list() {
      assert_eq!(
        interpret("Unitize[{0, 1, -3, 0, 5}]").unwrap(),
        "{0, 1, 1, 0, 1}"
      );
    }

    #[test]
    fn unitize_zero() {
      assert_eq!(interpret("Unitize[0]").unwrap(), "0");
    }

    #[test]
    fn unitize_nonzero() {
      assert_eq!(interpret("Unitize[42]").unwrap(), "1");
    }
  }

  mod ramp {
    use super::*;

    #[test]
    fn ramp_list() {
      assert_eq!(
        interpret("Ramp[{-2, -1, 0, 1, 2}]").unwrap(),
        "{0, 0, 0, 1, 2}"
      );
    }

    #[test]
    fn ramp_negative() {
      assert_eq!(interpret("Ramp[-5]").unwrap(), "0");
    }

    #[test]
    fn ramp_positive() {
      assert_eq!(interpret("Ramp[3]").unwrap(), "3");
    }

    #[test]
    fn ramp_negative_real_returns_real_zero() {
      assert_eq!(interpret("Ramp[-0.1]").unwrap(), "0.");
    }

    #[test]
    fn ramp_positive_real() {
      assert_eq!(interpret("Ramp[3.2]").unwrap(), "3.2");
    }

    #[test]
    fn ramp_map_with_reals() {
      assert_eq!(
        interpret("Map[Ramp, {-5, 3.2, -0.1, 7}]").unwrap(),
        "{0, 3.2, 0., 7}"
      );
    }
  }

  mod kronecker_delta {
    use super::*;

    #[test]
    fn equal() {
      assert_eq!(interpret("KroneckerDelta[1, 1]").unwrap(), "1");
    }

    #[test]
    fn unequal() {
      assert_eq!(interpret("KroneckerDelta[1, 2]").unwrap(), "0");
    }

    #[test]
    fn three_equal() {
      assert_eq!(interpret("KroneckerDelta[3, 3, 3]").unwrap(), "1");
    }

    #[test]
    fn three_unequal() {
      assert_eq!(interpret("KroneckerDelta[1, 2, 1]").unwrap(), "0");
    }

    #[test]
    fn no_args() {
      assert_eq!(interpret("KroneckerDelta[]").unwrap(), "1");
    }

    #[test]
    fn single_zero() {
      assert_eq!(interpret("KroneckerDelta[0]").unwrap(), "1");
    }

    #[test]
    fn single_nonzero() {
      assert_eq!(interpret("KroneckerDelta[3]").unwrap(), "0");
    }

    #[test]
    fn symbolic_single() {
      assert_eq!(interpret("KroneckerDelta[x]").unwrap(), "KroneckerDelta[x]");
    }

    #[test]
    fn symbolic_pair() {
      assert_eq!(
        interpret("KroneckerDelta[i, j]").unwrap(),
        "KroneckerDelta[i, j]"
      );
    }

    #[test]
    fn table_identity_matrix() {
      assert_eq!(
        interpret("Table[KroneckerDelta[i, j], {i, 3}, {j, 3}]").unwrap(),
        "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
      );
    }
  }

  mod unit_step {
    use super::*;

    #[test]
    fn positive() {
      assert_eq!(interpret("UnitStep[1]").unwrap(), "1");
    }

    #[test]
    fn zero() {
      assert_eq!(interpret("UnitStep[0]").unwrap(), "1");
    }

    #[test]
    fn negative() {
      assert_eq!(interpret("UnitStep[-1]").unwrap(), "0");
    }

    #[test]
    fn list() {
      assert_eq!(interpret("UnitStep[{-1, 0, 1}]").unwrap(), "{0, 1, 1}");
    }

    #[test]
    fn multi_arg_all_positive() {
      assert_eq!(interpret("UnitStep[1, 2, 3]").unwrap(), "1");
    }

    #[test]
    fn multi_arg_with_negative() {
      assert_eq!(interpret("UnitStep[1, 2, -1]").unwrap(), "0");
    }

    #[test]
    fn multi_arg_all_zero() {
      assert_eq!(interpret("UnitStep[0, 0, 0]").unwrap(), "1");
    }

    #[test]
    fn list_mixed() {
      assert_eq!(
        interpret("UnitStep[{-3, -1, 0, 1, 3}]").unwrap(),
        "{0, 0, 1, 1, 1}"
      );
    }
  }

  mod list_function {
    use super::*;

    #[test]
    fn basic() {
      assert_eq!(interpret("List[1, 2, 3]").unwrap(), "{1, 2, 3}");
    }

    #[test]
    fn empty() {
      assert_eq!(interpret("List[]").unwrap(), "{}");
    }

    #[test]
    fn nested() {
      assert_eq!(interpret("List[List[1, 2], 3]").unwrap(), "{{1, 2}, 3}");
    }
  }

  mod implicit_times_in_function_body {
    use super::*;

    #[test]
    fn numeric_times_variable() {
      assert_eq!(interpret("f[x_] := 2 x; f[5]").unwrap(), "10");
    }

    #[test]
    fn numeric_times_function_call() {
      assert_eq!(
        interpret("L[n_] := 2 Fibonacci[n + 1] - 1; L[5]").unwrap(),
        "15"
      );
    }

    #[test]
    fn numeric_times_slot() {
      assert_eq!(interpret("3 # &[5]").unwrap(), "15");
    }

    #[test]
    fn numeric_times_slot_in_function_arg() {
      assert_eq!(interpret("f[3 # + 1]").unwrap(), "f[1 + 3*#]");
    }

    #[test]
    fn numeric_times_slot_in_list() {
      assert_eq!(interpret("{3 #}").unwrap(), "{3*#}");
    }

    #[test]
    fn numeric_times_slot_in_anonymous_function() {
      // The anonymous function is passed as an argument, not called directly
      assert_eq!(
        interpret("Map[If[EvenQ[#], #/2, 3 # + 1] &, {4, 5}]").unwrap(),
        "{2, 16}"
      );
    }
  }

  mod do_single_iter {
    use super::*;

    #[test]
    fn repeat_n_times() {
      clear_state();
      assert_eq!(
        interpret_with_stdout("Do[Print[\"hello\"], {3}]")
          .unwrap()
          .stdout,
        "hello\nhello\nhello\n"
      );
    }
  }

  mod while_single_arg {
    use super::*;

    #[test]
    fn do_while_pattern() {
      clear_state();
      assert_eq!(
        interpret_with_stdout(
          "value = 0; While[value++; Print[value]; Mod[value,6]!=0]"
        )
        .unwrap()
        .stdout,
        "1\n2\n3\n4\n5\n6\n"
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

  mod function_name_substitution {
    use super::*;

    #[test]
    fn pass_function_as_argument() {
      clear_state();
      assert_eq!(
        interpret_with_stdout("g[f_] := f[]; g[Print[\"Hello\"] &]")
          .unwrap()
          .stdout,
        "Hello\n"
      );
    }

    #[test]
    fn repeat_with_anonymous_function() {
      clear_state();
      assert_eq!(
        interpret_with_stdout(
          "repeat[f_, n_] := Do[f[], {n}]; repeat[Print[\"hi\"] &, 3]"
        )
        .unwrap()
        .stdout,
        "hi\nhi\nhi\n"
      );
    }
  }

  mod nest_while_list {
    use super::*;

    #[test]
    fn basic_increment() {
      assert_eq!(
        interpret("NestWhileList[# + 1 &, 0, # < 5 &]").unwrap(),
        "{0, 1, 2, 3, 4, 5}"
      );
    }

    #[test]
    fn halving() {
      assert_eq!(
        interpret("NestWhileList[# / 2 &, 64, EvenQ]").unwrap(),
        "{64, 32, 16, 8, 4, 2, 1}"
      );
    }

    #[test]
    fn collatz_sequence() {
      assert_eq!(
        interpret("NestWhileList[If[EvenQ[#], #/2, 3 # + 1] &, 27, # > 1 &]")
          .unwrap(),
        "{27, 82, 41, 124, 62, 31, 94, 47, 142, 71, 214, 107, 322, 161, 484, 242, 121, 364, 182, 91, 274, 137, 412, 206, 103, 310, 155, 466, 233, 700, 350, 175, 526, 263, 790, 395, 1186, 593, 1780, 890, 445, 1336, 668, 334, 167, 502, 251, 754, 377, 1132, 566, 283, 850, 425, 1276, 638, 319, 958, 479, 1438, 719, 2158, 1079, 3238, 1619, 4858, 2429, 7288, 3644, 1822, 911, 2734, 1367, 4102, 2051, 6154, 3077, 9232, 4616, 2308, 1154, 577, 1732, 866, 433, 1300, 650, 325, 976, 488, 244, 122, 61, 184, 92, 46, 23, 70, 35, 106, 53, 160, 80, 40, 20, 10, 5, 16, 8, 4, 2, 1}"
      );
    }

    #[test]
    fn collatz_short() {
      assert_eq!(
        interpret("NestWhileList[If[EvenQ[#], #/2, 3 # + 1] &, 6, # > 1 &]")
          .unwrap(),
        "{6, 3, 10, 5, 16, 8, 4, 2, 1}"
      );
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

  mod plot {
    use super::*;

    #[test]
    fn plot_returns_graphics() {
      clear_state();
      assert_eq!(
        interpret("Plot[Sin[x], {x, 0, 2 Pi}]").unwrap(),
        "-Graphics-"
      );
    }

    #[test]
    fn plot_produces_svg() {
      clear_state();
      let result =
        interpret_with_stdout("Plot[Sin[x], {x, 0, 2 Pi}]").unwrap();
      assert_eq!(result.result, "-Graphics-");
      assert!(result.graphics.is_some());
      let svg = result.graphics.unwrap();
      assert!(svg.starts_with("<svg"));
      assert!(svg.contains("</svg>"));
      assert!(svg.contains("<path"));
    }

    #[test]
    fn plot_cos() {
      clear_state();
      let result =
        interpret_with_stdout("Plot[Cos[x], {x, 0, 2 Pi}]").unwrap();
      assert_eq!(result.result, "-Graphics-");
      assert!(result.graphics.is_some());
    }

    #[test]
    fn plot_polynomial() {
      clear_state();
      let result =
        interpret_with_stdout("Plot[x^2, {x, -2, 2}]").unwrap();
      assert_eq!(result.result, "-Graphics-");
      assert!(result.graphics.is_some());
    }

    #[test]
    fn plot_expression() {
      clear_state();
      let result =
        interpret_with_stdout("Plot[x^2 - 1, {x, -3, 3}]").unwrap();
      assert_eq!(result.result, "-Graphics-");
      assert!(result.graphics.is_some());
    }

    #[test]
    fn plot_with_numeric_range() {
      clear_state();
      let result =
        interpret_with_stdout("Plot[Sin[x], {x, 0, 6}]").unwrap();
      assert_eq!(result.result, "-Graphics-");
      assert!(result.graphics.is_some());
    }

    #[test]
    fn plot_unevaluated_with_one_arg() {
      clear_state();
      // Like Wolfram, Plot with wrong args returns unevaluated
      assert_eq!(
        interpret("Plot[Sin[x]]").unwrap(),
        "Plot[Sin[x]]"
      );
    }

    #[test]
    fn plot_error_bad_iterator() {
      clear_state();
      assert!(interpret("Plot[Sin[x], 5]").is_err());
    }

    #[test]
    fn plot_no_graphics_for_non_plot() {
      clear_state();
      let result = interpret_with_stdout("1 + 2").unwrap();
      assert_eq!(result.result, "3");
      assert!(result.graphics.is_none());
    }
  }

  mod echo {
    use super::*;

    #[test]
    fn single_arg_returns_value() {
      clear_state();
      let result = interpret_with_stdout("Echo[42]").unwrap();
      assert_eq!(result.result, "42");
      assert_eq!(result.stdout, ">> 42\n");
    }

    #[test]
    fn single_arg_with_expression() {
      clear_state();
      let result = interpret_with_stdout("Echo[2 + 3]").unwrap();
      assert_eq!(result.result, "5");
      assert_eq!(result.stdout, ">> 5\n");
    }

    #[test]
    fn two_args_custom_label() {
      clear_state();
      let result = interpret_with_stdout("Echo[2 + 3, \"result: \"]").unwrap();
      assert_eq!(result.result, "5");
      assert_eq!(result.stdout, ">> result:  5\n");
    }

    #[test]
    fn three_args_with_function() {
      clear_state();
      let result =
        interpret_with_stdout("Echo[{1, 2, 3}, \"length: \", Length]").unwrap();
      assert_eq!(result.result, "{1, 2, 3}");
      assert_eq!(result.stdout, ">> length:  3\n");
    }

    #[test]
    fn map_echo() {
      clear_state();
      let result = interpret_with_stdout("Map[Echo, {1, 2, 3}]").unwrap();
      assert_eq!(result.result, "{1, 2, 3}");
      assert_eq!(result.stdout, ">> 1\n>> 2\n>> 3\n");
    }

    #[test]
    fn echo_with_list() {
      clear_state();
      let result = interpret_with_stdout("Echo[{a, b, c}]").unwrap();
      assert_eq!(result.result, "{a, b, c}");
      assert_eq!(result.stdout, ">> {a, b, c}\n");
    }

    #[test]
    fn echo_with_string() {
      clear_state();
      let result = interpret_with_stdout("Echo[\"hello\"]").unwrap();
      assert_eq!(result.result, "hello");
      assert_eq!(result.stdout, ">> hello\n");
    }
  }

  mod sum {
    use super::*;

    #[test]
    fn finite_sum_integers() {
      assert_eq!(interpret("Sum[i, {i, 1, 10}]").unwrap(), "55");
    }

    #[test]
    fn finite_sum_squares() {
      assert_eq!(interpret("Sum[i^2, {i, 1, 5}]").unwrap(), "55");
    }

    #[test]
    fn finite_sum_with_explicit_min() {
      assert_eq!(interpret("Sum[i, {i, 5, 10}]").unwrap(), "45");
    }

    #[test]
    fn infinite_sum_zeta_2() {
      assert_eq!(interpret("Sum[1/n^2, {n, 1, Infinity}]").unwrap(), "Pi^2/6");
    }

    #[test]
    fn infinite_sum_zeta_4() {
      assert_eq!(
        interpret("Sum[1/n^4, {n, 1, Infinity}]").unwrap(),
        "Pi^4/90"
      );
    }

    #[test]
    fn infinite_sum_zeta_6() {
      assert_eq!(
        interpret("Sum[1/n^6, {n, 1, Infinity}]").unwrap(),
        "Pi^6/945"
      );
    }

    #[test]
    fn infinite_sum_zeta_8() {
      assert_eq!(
        interpret("Sum[1/n^8, {n, 1, Infinity}]").unwrap(),
        "Pi^8/9450"
      );
    }

    #[test]
    fn infinite_sum_zeta_10() {
      assert_eq!(
        interpret("Sum[1/n^10, {n, 1, Infinity}]").unwrap(),
        "Pi^10/93555"
      );
    }

    #[test]
    fn infinite_sum_zeta_12() {
      assert_eq!(
        interpret("Sum[1/n^12, {n, 1, Infinity}]").unwrap(),
        "(691*Pi^12)/638512875"
      );
    }

    #[test]
    fn infinite_sum_zeta_odd_returns_zeta() {
      assert_eq!(
        interpret("Sum[1/n^3, {n, 1, Infinity}]").unwrap(),
        "Zeta[3]"
      );
      assert_eq!(
        interpret("Sum[1/n^5, {n, 1, Infinity}]").unwrap(),
        "Zeta[5]"
      );
    }

    #[test]
    fn infinite_sum_negative_power_form() {
      // n^(-2) form should also work
      assert_eq!(
        interpret("Sum[n^(-2), {n, 1, Infinity}]").unwrap(),
        "Pi^2/6"
      );
    }

    #[test]
    fn infinite_sum_harmonic_unevaluated() {
      // Sum[1/n, {n, 1, Infinity}] diverges — should return unevaluated
      assert_eq!(
        interpret("Sum[1/n, {n, 1, Infinity}]").unwrap(),
        "Sum[1/n, {n, 1, Infinity}]"
      );
    }
  }
}
