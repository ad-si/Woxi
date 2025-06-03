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
}
