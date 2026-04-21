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
  fn gather_by_nested_funcs() {
    // GatherBy with a list of functions does a nested gather: first by
    // f1, then each resulting sublist is gathered by f2, and so on.
    assert_eq!(
      interpret("GatherBy[{1, 2, 3, 4, 5, 6}, {OddQ, PrimeQ}]").unwrap(),
      "{{{1}, {3, 5}}, {{2}, {4, 6}}}"
    );
    assert_eq!(
      interpret("GatherBy[Range[10], {EvenQ, PrimeQ}]").unwrap(),
      "{{{1, 9}, {3, 5, 7}}, {{2}, {4, 6, 8, 10}}}"
    );
    // A list with a single function behaves like GatherBy[list, f1] but
    // wraps the result in an extra level.
    assert_eq!(
      interpret("GatherBy[{{1,3},{2,2},{1,5},{2,1},{3,6}}, {First}]").unwrap(),
      "{{{1, 3}, {1, 5}}, {{2, 2}, {2, 1}}, {{3, 6}}}"
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
  fn split_by_list_of_funcs_single() {
    // Singleton list of funcs behaves like SplitBy[list, f] wrapped one extra level.
    assert_eq!(
      interpret("SplitBy[{1, 1, 2, 2, 3, 3, 3, 4}, {OddQ}]").unwrap(),
      "{{1, 1}, {2, 2}, {3, 3, 3}, {4}}"
    );
  }

  #[test]
  fn split_by_list_of_funcs_nested() {
    assert_eq!(
      interpret("SplitBy[{1, 1, 2, 2, 3, 3, 3, 4}, {OddQ, EvenQ}]").unwrap(),
      "{{{1, 1}}, {{2, 2}}, {{3, 3, 3}}, {{4}}}"
    );
  }

  #[test]
  fn split_by_list_of_funcs_with_structured_items() {
    assert_eq!(
      interpret("SplitBy[{{1, a}, {1, b}, {2, a}, {2, b}}, {First, Last}]")
        .unwrap(),
      "{{{{1, a}}, {{1, b}}}, {{{2, a}}, {{2, b}}}}"
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
  fn catenate_associations() {
    assert_eq!(
      interpret("Catenate[{<|a -> 1|>, <|b -> 2|>}]").unwrap(),
      "{1, 2}"
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

  #[test]
  fn flatten_single_level_spec_preserves_others() {
    // Flatten[m, {{2}}] for 3D m: unmentioned old levels 1 and 3 are
    // appended as singleton groups, so the result is equivalent to
    // Flatten[m, {{2}, {1}, {3}}]. Regression for mathics
    // test_structure.py:37.
    assert_eq!(
      interpret("m = {{{1, 2}, {3}}, {{4}, {5, 6}}}; Flatten[m, {{2}}]")
        .unwrap(),
      "{{{1, 2}, {4}}, {{3}, {5, 6}}}"
    );
  }

  #[test]
  fn flatten_with_custom_head() {
    // Flatten[f[g[a, b], g[c, d]], Infinity, g] splices g children into f
    assert_eq!(
      interpret("Flatten[f[g[a, b], g[c, d]], Infinity, g]").unwrap(),
      "f[a, b, c, d]"
    );
  }

  #[test]
  fn flatten_with_custom_head_in_list() {
    assert_eq!(
      interpret("Flatten[{f[a, b], f[c, d]}, Infinity, f]").unwrap(),
      "{a, b, c, d}"
    );
  }

  #[test]
  fn flatten_with_custom_head_nested() {
    assert_eq!(
      interpret("Flatten[f[g[a, b], g[c, g[d, e]]], Infinity, g]").unwrap(),
      "f[a, b, c, d, e]"
    );
  }

  #[test]
  fn flatten_with_custom_head_mixed() {
    // h[b] is not g, so it stays
    assert_eq!(
      interpret("Flatten[f[g[a], h[b], g[c]], Infinity, g]").unwrap(),
      "f[a, h[b], c]"
    );
  }

  // --- SubsetPosition ---
  #[test]
  fn subset_position_literal() {
    assert_eq!(
      interpret("SubsetPosition[{a, b, c, d, e, f}, {b, d}]").unwrap(),
      "{{2, 4}}"
    );
  }

  #[test]
  fn subset_position_pattern() {
    assert_eq!(
      interpret("SubsetPosition[{a, b, c, d, e, f}, {_, c}]").unwrap(),
      "{{1, 3}, {2, 3}, {4, 3}, {5, 3}, {6, 3}}"
    );
  }

  #[test]
  fn subset_position_all_pairs() {
    assert_eq!(
      interpret("SubsetPosition[{1, 2, 3, 4, 5}, {_, _}]").unwrap(),
      "{{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 5}, {3, 4}, {3, 5}, {4, 5}}"
    );
  }

  #[test]
  fn subset_position_pattern_test() {
    assert_eq!(
      interpret("SubsetPosition[{1, 2, 3, 4, 5}, {_?(# > 3 &)}]").unwrap(),
      "{{4}, {5}}"
    );
  }

  // --- SubsetCases ---
  #[test]
  fn subset_cases_basic() {
    assert_eq!(
      interpret("SubsetCases[{1, 2, 3, 4, 5}, {_, _}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  #[test]
  fn subset_cases_triples() {
    assert_eq!(
      interpret("SubsetCases[{1, 2, 3, 4, 5, 6}, {_, _, _}]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6}}"
    );
  }

  #[test]
  fn subset_cases_with_pattern() {
    assert_eq!(
      interpret("SubsetCases[{1, 2, 3, 4, 5}, {_?(# > 3 &)}]").unwrap(),
      "{{4}, {5}}"
    );
  }

  #[test]
  fn subset_cases_with_condition() {
    assert_eq!(
      interpret("SubsetCases[{a, b, c, d}, {x_, y_} /; OrderedQ[{x, y}]]")
        .unwrap(),
      "{{a, b}, {c, d}}"
    );
  }

  #[test]
  fn subset_cases_with_limit() {
    assert_eq!(
      interpret("SubsetCases[{1, 2, 3, 4, 5}, {_, _}, 1]").unwrap(),
      "{{1, 2}}"
    );
  }

  // --- SubsetCount ---
  #[test]
  fn subset_count_basic() {
    assert_eq!(interpret("SubsetCount[{a, b, c, d}, {a, c}]").unwrap(), "1");
  }

  #[test]
  fn subset_count_pairs() {
    assert_eq!(
      interpret("SubsetCount[{1, 2, 3, 4, 5}, {_, _}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn subset_count_pattern_test() {
    assert_eq!(
      interpret("SubsetCount[{1, 2, 3, 4, 5}, {_?(# > 3 &)}]").unwrap(),
      "2"
    );
  }
}
