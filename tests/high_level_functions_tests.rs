use woxi::interpret;

mod high_level_functions_tests {
  use super::*;

  mod evenq_tests {
    use super::*;
    #[test]
    fn test_for_negative() {
      assert_eq!(interpret("EvenQ[-2]").unwrap(), "True",);
      assert_eq!(interpret("EvenQ[-1]").unwrap(), "False",);
    }
    #[test]
    fn test_for_zero() {
      assert_eq!(interpret("EvenQ[0]").unwrap(), "True",);
    }
    #[test]
    fn test_for_positive() {
      assert_eq!(interpret("EvenQ[1]").unwrap(), "False",);
      assert_eq!(interpret("EvenQ[2]").unwrap(), "True",);
      assert_eq!(interpret("EvenQ[3]").unwrap(), "False",);
    }
    #[test]
    fn test_for_float() {
      assert_eq!(interpret("EvenQ[1.2]").unwrap(), "False",);
      assert_eq!(interpret("EvenQ[1.3]").unwrap(), "False",);
    }
  }

  mod oddq_tests {
    use super::*;
    #[test]
    fn test_for_negative() {
      assert_eq!(interpret("OddQ[-1]").unwrap(), "True");
    }
    #[test]
    fn test_for_zero() {
      assert_eq!(interpret("OddQ[0]").unwrap(), "False");
    }
    #[test]
    fn test_for_positive() {
      assert_eq!(interpret("OddQ[1]").unwrap(), "True");
      assert_eq!(interpret("OddQ[2]").unwrap(), "False");
    }
    #[test]
    fn test_for_float() {
      assert_eq!(interpret("OddQ[1.2]").unwrap(), "False");
      assert_eq!(interpret("OddQ[1.3]").unwrap(), "False");
    }
  }

  // #[test]
  // fn test_first_function() {
  //   assert_eq!(interpret("First[{1, 2, 3}]").unwrap(), "1");
  //   assert_eq!(interpret("First[{a, b, c}]").unwrap(), "a");
  //   assert_eq!(interpret("First[{True, False, False}]").unwrap(), "True");
  // }

  mod prime_function {
    use super::*;

    #[test]
    fn test_prime_function() {
      assert_eq!(interpret("Prime[1]").unwrap(), "2");
      assert_eq!(interpret("Prime[2]").unwrap(), "3");
      assert_eq!(interpret("Prime[3]").unwrap(), "5");
      assert_eq!(interpret("Prime[4]").unwrap(), "7");
      assert_eq!(interpret("Prime[5]").unwrap(), "11");
      assert_eq!(interpret("Prime[100]").unwrap(), "541");
    }

    #[test]
    fn test_prime_function_invalid_input() {
      assert!(interpret("Prime[0]").is_err());
      assert!(interpret("Prime[-1]").is_err());
      assert!(interpret("Prime[1.5]").is_err());
    }
  }

  // ─── Catch/Throw ───────────────────────────────────────────────────
  mod catch_throw_tests {
    use super::*;
    #[test]
    fn test_catch_with_throw() {
      assert_eq!(interpret("Catch[1 + Throw[2]]").unwrap(), "2");
    }
    #[test]
    fn test_catch_no_throw() {
      assert_eq!(interpret("Catch[1 + 2]").unwrap(), "3");
    }
    #[test]
    fn test_catch_with_tag() {
      assert_eq!(
        interpret(r#"Catch[Throw["hello", "tag"], "tag"]"#).unwrap(),
        "hello"
      );
    }
    #[test]
    fn test_throw_value() {
      assert_eq!(interpret("Catch[Throw[42]]").unwrap(), "42");
    }
    #[test]
    fn test_nested_catch() {
      assert_eq!(interpret("Catch[Catch[Throw[1, a], b], a]").unwrap(), "1");
    }
  }
}
