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
      assert_eq!(
        interpret("{1, 4, 9} // Map[Sqrt]").unwrap(),
        "{1, 2, 3}"
      );
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
        assert_eq!(interpret("\"hello\" /. x_ :> \"world\"").unwrap(), "\"world\"");
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
          "\"even\""
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
          "\"Fizz\""
        );
        assert_eq!(
          interpret("5 /. i_ /; Mod[i, 5] == 0 :> \"Buzz\"").unwrap(),
          "\"Buzz\""
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
        assert_eq!(
          interpret("4 /. x_?EvenQ :> \"even\"").unwrap(),
          "\"even\""
        );
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
        assert_eq!(
          interpret("{1, 2, 3} /. {x_ /; x == 1 :> \"one\", x_ /; x == 2 :> \"two\"}").unwrap(),
          "{\"one\", \"two\", 3}"
        );
      }

      #[test]
      fn fizzbuzz_style_rules() {
        // Test the FizzBuzz pattern
        assert_eq!(
          interpret("15 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "\"FizzBuzz\""
        );
        assert_eq!(
          interpret("9 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "\"Fizz\""
        );
        assert_eq!(
          interpret("10 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "\"Buzz\""
        );
        assert_eq!(
          interpret("7 /. {i_ /; Mod[i, 15] == 0 :> \"FizzBuzz\", i_ /; Mod[i, 3] == 0 :> \"Fizz\", i_ /; Mod[i, 5] == 0 :> \"Buzz\"}").unwrap(),
          "7"
        );
      }
    }
  }
}
