use woxi::interpret;

mod list_tests {
  use super::*;

  #[test]
  fn parse() {
    assert_eq!(interpret("{1, 2, 3}").unwrap(), "{1, 2, 3}");
    assert_eq!(interpret("{a, b, c}").unwrap(), "{a, b, c}");
    assert_eq!(
      interpret("{True, False, True}").unwrap(),
      "{True, False, True}"
    );
  }

  #[test]
  fn first() {
    assert_eq!(interpret("First[{1, 2, 3}]").unwrap(), "1");
    assert_eq!(interpret("First[{a, b, c}]").unwrap(), "a");
    assert_eq!(interpret("First[{True, False, False}]").unwrap(), "True");
  }

  #[test]
  fn last() {
    assert_eq!(interpret("Last[{1, 2, 3}]").unwrap(), "3");
    assert_eq!(interpret("Last[{a, b, c}]").unwrap(), "c");
    assert_eq!(interpret("Last[{True, True, False}]").unwrap(), "False");
  }
}
