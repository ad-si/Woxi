use woxi::interpret;

mod high_level_functions_tests {
  use super::*;

  mod evenq_tests {
    use super::*;

    #[test]
    fn test_evenq_function() {
      assert_eq!(interpret("EvenQ[-1]").unwrap(), "False");
      assert_eq!(interpret("EvenQ[0]").unwrap(), "True");
      assert_eq!(interpret("EvenQ[1]").unwrap(), "False");
      assert_eq!(interpret("EvenQ[1.2]").unwrap(), "False");
      assert_eq!(interpret("EvenQ[2]").unwrap(), "True");
      assert_eq!(interpret("EvenQ[3]").unwrap(), "False");
    }
  }

  mod oddq_tests {
    use super::*;

    #[test]
    fn test_oddq_function() {
      assert_eq!(interpret("OddQ[-1]").unwrap(), "True");
      assert_eq!(interpret("OddQ[0]").unwrap(), "False");
      assert_eq!(interpret("OddQ[1]").unwrap(), "True");
      assert_eq!(interpret("OddQ[1.3]").unwrap(), "False");
      assert_eq!(interpret("OddQ[2]").unwrap(), "False");
      assert_eq!(interpret("OddQ[3]").unwrap(), "True");
    }
  }

  mod prime_tests {
    use super::*;

    #[test]
    fn test_prime_function() {
      assert_eq!(interpret("Prime[1]").unwrap(), "2");
      assert_eq!(interpret("Prime[2]").unwrap(), "3");
      assert_eq!(interpret("Prime[3]").unwrap(), "5");
      assert_eq!(interpret("Prime[4]").unwrap(), "7");
      assert_eq!(interpret("Prime[5]").unwrap(), "11");
    }
  }

  mod group_by_tests {
    #[test]
    #[ignore]
    fn test_group_by_function() {
      // let result = interpret("GroupBy[{1, 2, 3, 4, 5}, EvenQ]").unwrap();
      // assert_eq!(result, "<|False -> {1, 3, 5}, True -> {2, 4}|>");
    }
  }
}
