use woxi::interpret;

mod high_level_functions_tests {
  use super::*;

  mod prime_tests {
    use super::*;

    #[test]
    fn test_prime_function() {
      assert_eq!(interpret("Prime[1]").unwrap(), 2.0);
      assert_eq!(interpret("Prime[2]").unwrap(), 3.0);
      assert_eq!(interpret("Prime[3]").unwrap(), 5.0);
      assert_eq!(interpret("Prime[4]").unwrap(), 7.0);
      assert_eq!(interpret("Prime[5]").unwrap(), 11.0);
    }
  }

  mod group_by_tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_group_by_function() {
      // let result = interpret("GroupBy[{1, 2, 3, 4, 5}, EvenQ]").unwrap();
      // assert_eq!(result, "<|False -> {1, 3, 5}, True -> {2, 4}|>");
    }
  }
}
