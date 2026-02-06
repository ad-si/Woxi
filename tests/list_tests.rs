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

  #[test]
  fn gather() {
    assert_eq!(
      interpret("Gather[{1, 1, 2, 2, 1}]").unwrap(),
      "{{1, 1, 1}, {2, 2}}"
    );
    assert_eq!(
      interpret("Gather[{a, b, a, c, b}]").unwrap(),
      "{{a, a}, {b, b}, {c}}"
    );
    assert_eq!(interpret("Gather[{}]").unwrap(), "{}");
  }

  #[test]
  fn gather_by() {
    assert_eq!(
      interpret("GatherBy[{1, 2, 3, 4, 5}, EvenQ]").unwrap(),
      "{{1, 3, 5}, {2, 4}}"
    );
    assert_eq!(
      interpret("GatherBy[{-2, -1, 0, 1, 2}, Sign]").unwrap(),
      "{{-2, -1}, {0}, {1, 2}}"
    );
  }

  #[test]
  fn split() {
    assert_eq!(
      interpret("Split[{1, 1, 2, 2, 3}]").unwrap(),
      "{{1, 1}, {2, 2}, {3}}"
    );
    assert_eq!(
      interpret("Split[{a, a, b, c, c, c}]").unwrap(),
      "{{a, a}, {b}, {c, c, c}}"
    );
    assert_eq!(interpret("Split[{}]").unwrap(), "{}");
  }

  #[test]
  fn split_by() {
    assert_eq!(
      interpret("SplitBy[{1, 2, 3, 4, 5}, EvenQ]").unwrap(),
      "{{1}, {2}, {3}, {4}, {5}}"
    );
  }

  #[test]
  fn extract() {
    assert_eq!(interpret("Extract[{a, b, c, d}, 2]").unwrap(), "b");
    assert_eq!(
      interpret("Extract[{a, {b1, b2, b3}, c, d}, {2, 3}]").unwrap(),
      "b3"
    );
  }

  #[test]
  fn catenate() {
    assert_eq!(
      interpret("Catenate[{{1, 2}, {3, 4}}]").unwrap(),
      "{1, 2, 3, 4}"
    );
    assert_eq!(
      interpret("Catenate[{{a, b}, {c}, {d, e, f}}]").unwrap(),
      "{a, b, c, d, e, f}"
    );
  }

  #[test]
  fn apply() {
    assert_eq!(interpret("Apply[Plus, {1, 2, 3}]").unwrap(), "6");
    assert_eq!(interpret("Apply[Times, {2, 3, 4}]").unwrap(), "24");
  }

  #[test]
  fn identity() {
    assert_eq!(interpret("Identity[5]").unwrap(), "5");
    assert_eq!(interpret("Identity[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn outer() {
    assert_eq!(
      interpret("Outer[Times, {1, 2}, {3, 4}]").unwrap(),
      "{{3, 4}, {6, 8}}"
    );
    assert_eq!(
      interpret("Outer[Plus, {1, 2}, {10, 20}]").unwrap(),
      "{{11, 21}, {12, 22}}"
    );
  }

  #[test]
  fn inner() {
    assert_eq!(
      interpret("Inner[Times, {1, 2, 3}, {4, 5, 6}, Plus]").unwrap(),
      "32"
    );
    assert_eq!(
      interpret("Inner[Plus, {1, 2}, {3, 4}, Times]").unwrap(),
      "24"
    );
  }

  #[test]
  fn replace_part() {
    assert_eq!(
      interpret("ReplacePart[{a, b, c}, 2 -> x]").unwrap(),
      "{a, x, c}"
    );
    assert_eq!(
      interpret("ReplacePart[{1, 2, 3, 4}, 1 -> 0]").unwrap(),
      "{0, 2, 3, 4}"
    );
    assert_eq!(
      interpret("ReplacePart[{a, b, c}, -1 -> z]").unwrap(),
      "{a, b, z}"
    );
  }

  #[test]
  fn array() {
    assert_eq!(interpret("Array[#^2 &, 3]").unwrap(), "{1, 4, 9}");
    assert_eq!(interpret("Array[# + 1 &, 4]").unwrap(), "{2, 3, 4, 5}");
  }

  #[test]
  fn through_two_args() {
    // Through[{f, g}, h] returns {f, g} when head doesn't match h
    assert_eq!(interpret("Through[{Sin, Cos}, 0]").unwrap(), "{Sin, Cos}");
    assert_eq!(
      interpret("Through[{Abs, Sign}, -5]").unwrap(),
      "{Abs, Sign}"
    );
  }

  #[test]
  fn delete_cases_with_count() {
    // DeleteCases[list, pattern, levelspec, n] deletes at most n matches
    assert_eq!(
      interpret("DeleteCases[{a, b, a, c, a}, a, 1, 1]").unwrap(),
      "{b, a, c, a}"
    );
    assert_eq!(
      interpret("DeleteCases[{a, b, a, c, a}, a, 1, 2]").unwrap(),
      "{b, c, a}"
    );
    // Without count, deletes all
    assert_eq!(
      interpret("DeleteCases[{a, b, a, c, a}, a]").unwrap(),
      "{b, c}"
    );
  }

  #[test]
  fn flatten_with_level() {
    assert_eq!(
      interpret("Flatten[{{{1, 2}}, {{3, 4}}}, 1]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
    assert_eq!(interpret("Flatten[{{{1}}, {{2}}}, 2]").unwrap(), "{1, 2}");
    // Flatten[list, 0] returns list unchanged
    assert_eq!(
      interpret("Flatten[{{1, 2}, {3, 4}}, 0]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }
}
