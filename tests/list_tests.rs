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

  // --- FlattenAt ---
  #[test]
  fn flatten_at_positive_index() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, 2]").unwrap(),
      "{a, b, c, {d, e}, {f}}"
    );
  }

  #[test]
  fn flatten_at_negative_index() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, -1]").unwrap(),
      "{a, {b, c}, {d, e}, f}"
    );
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, -2]").unwrap(),
      "{a, {b, c}, d, e, {f}}"
    );
  }

  #[test]
  fn flatten_at_singleton_position() {
    // {2} is a length-1 position vector — same as integer 2
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, {2}]").unwrap(),
      "{a, b, c, {d, e}, {f}}"
    );
  }

  #[test]
  fn flatten_at_list_of_positions() {
    // {{2}, {4}} flattens at positions 2 and 4
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, {{2}, {4}}]").unwrap(),
      "{a, b, c, {d, e}, f}"
    );
  }

  #[test]
  fn flatten_at_list_of_positions_with_negative() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, {d, e}, {f}}, {{-1}, {2}}]").unwrap(),
      "{a, b, c, {d, e}, f}"
    );
  }

  #[test]
  fn flatten_at_nested_position() {
    // {2, 1} is a single nested position — flatten expr[[2, 1]]
    assert_eq!(
      interpret("FlattenAt[{a, {{b, c}, {d, e}}, {f}}, {2, 1}]").unwrap(),
      "{a, {b, c, {d, e}}, {f}}"
    );
  }

  #[test]
  fn flatten_at_list_of_nested_positions() {
    assert_eq!(
      interpret(
        "FlattenAt[{{a, b}, {c, {d, e}}, {f, {g, h}}}, {{2, 2}, {3, 2}}]"
      )
      .unwrap(),
      "{{a, b}, {c, d, e}, {f, g, h}}"
    );
  }

  #[test]
  fn flatten_at_non_list_head() {
    assert_eq!(
      interpret("FlattenAt[f[a, g[b, c], d], 2]").unwrap(),
      "f[a, b, c, d]"
    );
  }

  #[test]
  fn flatten_at_operator_form() {
    assert_eq!(
      interpret("FlattenAt[2][{a, {b, c}, {d, e}, {f}}]").unwrap(),
      "{a, b, c, {d, e}, {f}}"
    );
  }

  #[test]
  fn flatten_at_empty_position() {
    // FlattenAt[expr, {}] is a no-op
    assert_eq!(interpret("FlattenAt[{a, b}, {}]").unwrap(), "{a, b}");
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

  #[test]
  fn range_imax() {
    // Range[imax] generates {1, 2, ..., imax}
    assert_eq!(interpret("Range[5]").unwrap(), "{1, 2, 3, 4, 5}");
    assert_eq!(interpret("Range[1]").unwrap(), "{1}");
    assert_eq!(interpret("Range[0]").unwrap(), "{}");
    // Non-integer upper bound: step defaults to 1, min defaults to 1
    // so the elements are integers (matches wolframscript).
    assert_eq!(interpret("Range[3.5]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn range_imin_imax() {
    // Range[imin, imax] generates {imin, ..., imax}
    assert_eq!(interpret("Range[2, 8]").unwrap(), "{2, 3, 4, 5, 6, 7, 8}");
    assert_eq!(interpret("Range[5, 5]").unwrap(), "{5}");
    // Empty when imin > imax and step is positive (the default).
    assert_eq!(interpret("Range[5, 1]").unwrap(), "{}");
    // Integer min with Real (non-integer) max -> integer elements.
    assert_eq!(interpret("Range[1, 3.5]").unwrap(), "{1, 2, 3}");
    // Real min with integer max -> Real elements.
    assert_eq!(interpret("Range[1.0, 5]").unwrap(), "{1., 2., 3., 4., 5.}");
    // Negative bounds.
    assert_eq!(interpret("Range[-2, 2]").unwrap(), "{-2, -1, 0, 1, 2}");
  }

  #[test]
  fn range_imin_imax_step() {
    // Range[imin, imax, di] uses step di.
    assert_eq!(interpret("Range[1, 10, 2]").unwrap(), "{1, 3, 5, 7, 9}");
    // Negative step iterates downward.
    assert_eq!(
      interpret("Range[10, 1, -1]").unwrap(),
      "{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}"
    );
    // Fractional rational step preserves exact rationals.
    assert_eq!(
      interpret("Range[1, 2, 1/4]").unwrap(),
      "{1, 5/4, 3/2, 7/4, 2}"
    );
    // Real step -> Real outputs.
    assert_eq!(
      interpret("Range[0, 1, 0.25]").unwrap(),
      "{0., 0.25, 0.5, 0.75, 1.}"
    );
    // Step that overshoots -> single element.
    assert_eq!(interpret("Range[1, 5, 10]").unwrap(), "{1}");
  }

  #[test]
  fn range_zero_step_errors() {
    assert!(interpret("Range[1, 5, 0]").is_err());
  }

  #[test]
  fn range_symbolic_step_with_numeric_ratio() {
    // Range[a, b, (b - a)/n]: step is symbolic but (max - min)/step
    // simplifies to a numeric value, so the count is determined.
    // Verify count and endpoints rather than exact string formatting.
    assert_eq!(interpret("Length[Range[a, b, (b - a)/4]]").unwrap(), "5");
    assert_eq!(interpret("First[Range[a, b, (b - a)/4]]").unwrap(), "a");
    assert_eq!(interpret("Last[Range[a, b, (b - a)/4]]").unwrap(), "b");

    assert_eq!(interpret("Length[Range[a, b, (b - a)/3]]").unwrap(), "4");
    assert_eq!(interpret("First[Range[a, b, (b - a)/3]]").unwrap(), "a");
    assert_eq!(interpret("Last[Range[a, b, (b - a)/3]]").unwrap(), "b");

    // Mixed numeric/symbolic ratio: (b - a) / ((b - a)/2.5) = 2.5,
    // so count = floor(2.5) + 1 = 3 elements.
    assert_eq!(interpret("Length[Range[a, b, (b - a)/2.5]]").unwrap(), "3");
    assert_eq!(interpret("First[Range[a, b, (b - a)/2.5]]").unwrap(), "a");

    // Adjacent differences should equal the step.
    assert_eq!(
      interpret("Simplify[Range[a, b, (b - a)/4][[2]] - a]").unwrap(),
      interpret("Simplify[(b - a)/4]").unwrap()
    );
  }

  #[test]
  fn range_irrational_ratio() {
    // When (max - min)/step simplifies to an irrational positive real
    // like Pi (~3.14), Range should still enumerate using floor(k) + 1.
    assert_eq!(interpret("Length[Range[a, b, (b - a)/Pi]]").unwrap(), "4");
    assert_eq!(interpret("First[Range[a, b, (b - a)/Pi]]").unwrap(), "a");

    // E ~ 2.718 -> 3 elements
    assert_eq!(interpret("Length[Range[a, b, (b - a)/E]]").unwrap(), "3");

    // GoldenRatio ~ 1.618 -> 2 elements
    assert_eq!(
      interpret("Length[Range[a, b, (b - a)/GoldenRatio]]").unwrap(),
      "2"
    );

    // Sqrt[2] ~ 1.414 -> 2 elements
    assert_eq!(
      interpret("Length[Range[a, b, (b - a)/Sqrt[2]]]").unwrap(),
      "2"
    );
  }

  #[test]
  fn inverse_permutation_list() {
    assert_eq!(
      interpret("InversePermutation[{2, 3, 1}]").unwrap(),
      "{3, 1, 2}"
    );
    assert_eq!(
      interpret("InversePermutation[{2, 3, 1, 4}]").unwrap(),
      "{3, 1, 2, 4}"
    );
    assert_eq!(
      interpret("InversePermutation[{1, 2, 3}]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn inverse_permutation_cycles() {
    // Basic two-cycle inversion: each cycle is reversed, then canonicalised
    // so its smallest element comes first and cycles are sorted by their
    // smallest element.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{3, 2, 5, 1}, {4, 7}}]]").unwrap(),
      "Cycles[{{1, 5, 2, 3}, {4, 7}}]"
    );
    // Involutions are their own inverse.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{1, 2}, {3, 4}}]]").unwrap(),
      "Cycles[{{1, 2}, {3, 4}}]"
    );
    // Empty cycle list.
    assert_eq!(
      interpret("InversePermutation[Cycles[{}]]").unwrap(),
      "Cycles[{}]"
    );
    // Single longer cycle requiring rotation.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{2, 3, 5, 1, 4}}]]").unwrap(),
      "Cycles[{{1, 5, 3, 2, 4}}]"
    );
    // Two-element cycles in canonical order are unchanged.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{5, 4, 3, 2, 1}}]]").unwrap(),
      "Cycles[{{1, 2, 3, 4, 5}}]"
    );
    // Multiple cycles get sorted by smallest element after inversion.
    assert_eq!(
      interpret("InversePermutation[Cycles[{{1, 3, 2}, {7, 6, 8}}]]").unwrap(),
      "Cycles[{{1, 2, 3}, {6, 7, 8}}]"
    );
  }

  #[test]
  fn permutation_power_list() {
    // List-form permutations (regression coverage for existing behaviour).
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, 3]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, 2]").unwrap(),
      "{3, 1, 2}"
    );
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, -1]").unwrap(),
      "{3, 1, 2}"
    );
  }

  #[test]
  fn permutation_power_cycles() {
    // Sixth power of a permutation with cycle lengths 3 and 4. The
    // 3-cycle returns to identity (6 mod 3 = 0) and is dropped; the
    // 4-cycle becomes its square (two swaps).
    assert_eq!(
      interpret("PermutationPower[Cycles[{{4, 2, 5}, {6, 3, 1, 7}}], 6]")
        .unwrap(),
      "Cycles[{{1, 6}, {3, 7}}]"
    );
    // Inverse squared on the same permutation.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{4, 2, 5}, {6, 3, 1, 7}}], -2]")
        .unwrap(),
      "Cycles[{{1, 6}, {2, 5, 4}, {3, 7}}]"
    );
    // Zeroth power: identity.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2, 3}}], 0]").unwrap(),
      "Cycles[{}]"
    );
    // Cube of a 3-cycle is identity.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2, 3}}], 3]").unwrap(),
      "Cycles[{}]"
    );
    // Square of a 3-cycle is its inverse.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2, 3}}], 2]").unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
    // Negative power of a 3-cycle.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2, 3}}], -1]").unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
    // Odd power of a 2-cycle is itself.
    assert_eq!(
      interpret("PermutationPower[Cycles[{{1, 2}}], 5]").unwrap(),
      "Cycles[{{1, 2}}]"
    );
    // Any power of the identity is the identity.
    assert_eq!(
      interpret("PermutationPower[Cycles[{}], 5]").unwrap(),
      "Cycles[{}]"
    );
  }

  #[test]
  fn harmonic_mean_symbolic() {
    // n / Plus[1/x1, ..., 1/xn], with reciprocals printed as x^(-1).
    assert_eq!(
      interpret("HarmonicMean[{a, b, c, d}]").unwrap(),
      "4/(a^(-1) + b^(-1) + c^(-1) + d^(-1))"
    );
    assert_eq!(
      interpret("HarmonicMean[{a, b}]").unwrap(),
      "2/(a^(-1) + b^(-1))"
    );
    // Singleton symbolic list collapses to the single element.
    assert_eq!(interpret("HarmonicMean[{a}]").unwrap(), "a");
    // Mixed numeric and symbolic: numeric reciprocals combine into a rational.
    assert_eq!(
      interpret("HarmonicMean[{2, 4, x}]").unwrap(),
      "3/(3/4 + x^(-1))"
    );
  }

  #[test]
  fn trimmed_mean_asymmetric() {
    // TrimmedMean[list, {f1, f2}]: floor(f1*n) smallest and floor(f2*n)
    // largest elements are removed before averaging.
    assert_eq!(
      interpret("TrimmedMean[{-10, 1, 1, 1, 1, 20}, {0.2, 0}]").unwrap(),
      "24/5"
    );
    assert_eq!(
      interpret("TrimmedMean[{-10, 1, 1, 1, 1, 20}, {0, 0.2}]").unwrap(),
      "-6/5"
    );
    // Drop 1 from start (floor(0.1*10)=1) and 3 from end (floor(0.3*10)=3).
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0.1, 0.3}]")
        .unwrap(),
      "9/2"
    );
    // {0, 0} drops nothing and falls back to the plain mean.
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 4, 5}, {0, 0}]").unwrap(),
      "3"
    );
  }

  #[test]
  fn trimmed_mean_default_fraction() {
    // TrimmedMean[list] defaults to the 5% trimmed mean, i.e. f=0.05.
    // For n=10 with f=0.05, floor(0.5)=0 so nothing is trimmed.
    assert_eq!(
      interpret("TrimmedMean[{-10, 1, 1, 1, 1, 20}]").unwrap(),
      "7/3"
    );
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]").unwrap(),
      "11/2"
    );
  }

  #[test]
  fn trimmed_mean_floor_rounding() {
    // Regression: 0.05 * 10 = 0.5 should floor to 0 (not round to 1),
    // matching wolframscript. Without this fix, an outlier at the end of
    // the list would be incorrectly dropped and the mean shifted.
    assert_eq!(
      interpret("TrimmedMean[{0, 1, 2, 3, 4, 5, 6, 7, 8, 100}, 0.05]").unwrap(),
      "68/5"
    );
    // 0.025 * 20 = 0.5 → floor = 0 → no trimming
    assert_eq!(interpret("TrimmedMean[Range[20], 0.025]").unwrap(), "21/2");
  }

  #[test]
  fn take_list_negative_counts() {
    // Negative counts take from the end of the remaining slice and
    // remove the taken elements from it.
    assert_eq!(
      interpret("TakeList[{a, b, c, d, e, f, g, h}, {-2, -3, -1}]").unwrap(),
      "{{g, h}, {d, e, f}, {c}}"
    );
    // Mixed signs interleave front/back takes.
    assert_eq!(
      interpret("TakeList[{a, b, c, d, e, f, g, h}, {-2, 3, -1}]").unwrap(),
      "{{g, h}, {a, b, c}, {f}}"
    );
  }

  #[test]
  fn take_list_custom_head() {
    // Custom heads are preserved on each sublist.
    assert_eq!(
      interpret("TakeList[h[a, b, c, d, e, f, g], {2, 3, 1}]").unwrap(),
      "{h[a, b], h[c, d, e], h[f]}"
    );
  }

  #[test]
  fn take_list_all_and_upto() {
    // All takes everything remaining.
    assert_eq!(
      interpret("TakeList[{a, b, c, d, e, f, g, h}, {2, 3, All}]").unwrap(),
      "{{a, b}, {c, d, e}, {f, g, h}}"
    );
    // UpTo[n] takes min(n, remaining); an UpTo past the end yields an
    // empty trailing sublist instead of an error.
    assert_eq!(
      interpret("TakeList[Range[12], {2, 3, UpTo[10], UpTo[5]}]").unwrap(),
      "{{1, 2}, {3, 4, 5}, {6, 7, 8, 9, 10, 11, 12}, {}}"
    );
  }

  #[test]
  fn threshold_default() {
    // Threshold[data] uses 10^-10 as the default threshold and replaces
    // elements with |x| <= threshold by zero. The zero substituted in is
    // an integer 0 for integer/rational inputs.
    assert_eq!(
      interpret("Threshold[{1, 10^(-1), 10^(-2), 10^(-8), 10^(-11)}]").unwrap(),
      "{1, 1/10, 1/100, 1/100000000, 0}"
    );
  }

  #[test]
  fn threshold_numeric_list() {
    // |x| <= t triggers replacement (note the boundary case: |2| <= 2 → 0).
    assert_eq!(
      interpret("Threshold[{-3, 1, -2, 0, 2, -1, 0, 1, -3, 3, 2}, 3/2]")
        .unwrap(),
      "{-3, 0, -2, 0, 2, 0, 0, 0, -3, 3, 2}"
    );
    assert_eq!(
      interpret("Threshold[{-3, 1, -2, 0, 2, -1, 0, 1, -3, 3, 2}, 2]").unwrap(),
      "{-3, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0}"
    );
    // Real input: zeros come back as 0. (Real), not 0 (Integer).
    assert_eq!(
      interpret("Threshold[{1.5, -0.5, 0.1, -2.0, 3.0}, 1.0]").unwrap(),
      "{1.5, 0., 0., -2., 3.}"
    );
  }

  #[test]
  fn threshold_rectangular_array() {
    // Nested numeric lists are thresholded element-wise.
    assert_eq!(
      interpret("Threshold[{{1, 2}, {3, 4}}, 2]").unwrap(),
      "{{0, 0}, {3, 4}}"
    );
  }

  #[test]
  fn threshold_non_numeric_returns_unevaluated() {
    // A non-numeric element should leave the call unevaluated rather
    // than coerce or silently drop the symbol.
    assert_eq!(
      interpret("Threshold[{1, 2, x}, 1]").unwrap(),
      "Threshold[{1, 2, x}, 1]"
    );
  }

  #[test]
  fn indexed_concrete_vector() {
    // Indexed[list, i] behaves like Part[list, i] for a concrete list.
    assert_eq!(interpret("Indexed[{a, b}, 1]").unwrap(), "a");
    assert_eq!(interpret("Indexed[{a, b}, 2]").unwrap(), "b");
    // Negative indices count from the end.
    assert_eq!(interpret("Indexed[{a, b, c}, -1]").unwrap(), "c");
  }

  #[test]
  fn indexed_concrete_matrix() {
    // Indexed[expr, {i, j, ...}] descends one level per index.
    assert_eq!(
      interpret("Indexed[{{a, b, c}, {d, e, f}}, {1, 2}]").unwrap(),
      "b"
    );
    assert_eq!(
      interpret("Indexed[{{a, b, c}, {d, e, f}}, {2, 3}]").unwrap(),
      "f"
    );
    // A one-element index list still indexes one level.
    assert_eq!(
      interpret("Indexed[{{a, b, c}, {d, e, f}}, {2}]").unwrap(),
      "{d, e, f}"
    );
    assert_eq!(interpret("Indexed[{a, b, c}, {2}]").unwrap(), "b");
  }

  #[test]
  fn indexed_out_of_range_returns_unevaluated() {
    // Out-of-range indices leave the call unevaluated (matching
    // wolframscript's Indexed::partw message).
    assert_eq!(
      interpret("Indexed[{a, b}, 3]").unwrap(),
      "Indexed[{a, b}, 3]"
    );
    // Zero is rejected (Indexed::ind: not a nonzero integer).
    assert_eq!(
      interpret("Indexed[{a, b}, 0]").unwrap(),
      "Indexed[{a, b}, 0]"
    );
  }

  #[test]
  fn indexed_symbolic_normalises_to_list_index() {
    // A non-list argument cannot be indexed concretely, so the call
    // stays symbolic — but the integer index is canonicalised into a
    // singleton list, matching wolframscript.
    assert_eq!(interpret("Indexed[x, 1]").unwrap(), "Indexed[x, {1}]");
  }

  #[test]
  fn center_array_scalar() {
    // Scalar input is centered in a length-n vector; padding split
    // floor((n-k)/2) left, ceil((n-k)/2) right (so for even n a scalar
    // sits just left of middle).
    assert_eq!(interpret("CenterArray[x, 5]").unwrap(), "{0, 0, x, 0, 0}");
    assert_eq!(interpret("CenterArray[x, 4]").unwrap(), "{0, x, 0, 0}");
    assert_eq!(
      interpret("CenterArray[x, 6]").unwrap(),
      "{0, 0, x, 0, 0, 0}"
    );
    // Custom padding element.
    assert_eq!(
      interpret("CenterArray[x, 5, y]").unwrap(),
      "{y, y, x, y, y}"
    );
  }

  #[test]
  fn center_array_no_arg_defaults_to_1() {
    // CenterArray[n] is shorthand for CenterArray[1, n].
    assert_eq!(interpret("CenterArray[5]").unwrap(), "{0, 0, 1, 0, 0}");
  }

  #[test]
  fn center_array_list_in_vector() {
    // Asymmetric padding when (n - k) is odd: 1 left, 2 right.
    assert_eq!(
      interpret("CenterArray[{a, b}, 5]").unwrap(),
      "{0, a, b, 0, 0}"
    );
    // Symmetric padding when (n - k) is even.
    assert_eq!(
      interpret("CenterArray[{a, b, c}, 7]").unwrap(),
      "{0, 0, a, b, c, 0, 0}"
    );
  }

  #[test]
  fn center_array_multi_dim_scalar() {
    // 2D specification with a scalar value.
    assert_eq!(
      interpret("CenterArray[x, {5, 5}]").unwrap(),
      "{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, x, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}"
    );
    // Even dims: scalar sits at row 2, col 2 (1-indexed).
    assert_eq!(
      interpret("CenterArray[x, {4, 4}]").unwrap(),
      "{{0, 0, 0, 0}, {0, x, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}"
    );
  }

  #[test]
  fn center_array_multi_dim_block() {
    // 2D centered block.
    assert_eq!(
      interpret("CenterArray[{{a, b}, {c, d}}, {4, 4}]").unwrap(),
      "{{0, 0, 0, 0}, {0, a, b, 0}, {0, c, d, 0}, {0, 0, 0, 0}}"
    );
  }

  #[test]
  fn frechet_mean_symbolic() {
    // Mean exists only when shape > 1; the Piecewise default is Infinity.
    assert_eq!(
      interpret("Mean[FrechetDistribution[a, b]]").unwrap(),
      "Piecewise[{{b*Gamma[1 - a^(-1)], 1 < a}}, Infinity]"
    );
  }

  #[test]
  fn frechet_variance_symbolic() {
    // Variance exists only when shape > 2.
    assert_eq!(
      interpret("Variance[FrechetDistribution[a, b]]").unwrap(),
      "Piecewise[{{b^2*(Gamma[1 - 2/a] - Gamma[1 - a^(-1)]^2), a > 2}}, Infinity]"
    );
  }

  #[test]
  fn frechet_median_symbolic() {
    // Median is always defined: b * Log[2]^(-1/a).
    assert_eq!(
      interpret("Median[FrechetDistribution[a, b]]").unwrap(),
      "b/Log[2]^a^(-1)"
    );
  }

  #[test]
  fn frechet_mean_numeric_branches() {
    // Concrete a > 1 collapses the Piecewise into the first branch.
    assert_eq!(
      interpret("Mean[FrechetDistribution[2, 3]]").unwrap(),
      "3*Sqrt[Pi]"
    );
    // a ≤ 1 falls through to the default branch (Infinity).
    assert_eq!(
      interpret("Mean[FrechetDistribution[0.5, 1.5]]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn frechet_variance_numeric_branches() {
    // a ≤ 2 → Infinity.
    assert_eq!(
      interpret("Variance[FrechetDistribution[2, 3]]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn half_normal_mean_and_variance_symbolic() {
    // Mean = 1/theta; rendered with the negative-exponent form.
    assert_eq!(
      interpret("Mean[HalfNormalDistribution[a]]").unwrap(),
      "a^(-1)"
    );
    // Variance = (Pi - 2) / (2 theta^2).
    assert_eq!(
      interpret("Variance[HalfNormalDistribution[a]]").unwrap(),
      "(-2 + Pi)/(2*a^2)"
    );
  }

  #[test]
  fn half_normal_median_symbolic() {
    // Median = Sqrt[Pi] * InverseErf[1/2] / theta.
    assert_eq!(
      interpret("Median[HalfNormalDistribution[a]]").unwrap(),
      "(Sqrt[Pi]*InverseErf[1/2])/a"
    );
  }

  #[test]
  fn half_normal_numeric_values() {
    // Concrete parameters should collapse to exact rationals.
    assert_eq!(interpret("Mean[HalfNormalDistribution[2]]").unwrap(), "1/2");
    assert_eq!(
      interpret("Variance[HalfNormalDistribution[2]]").unwrap(),
      "(-2 + Pi)/8"
    );
  }

  #[test]
  fn hypoexponential_mean_symbolic() {
    // Mean = sum_i 1/lambda_i. Woxi renders reciprocals as x^(-1).
    assert_eq!(
      interpret("Mean[HypoexponentialDistribution[{a, b, c}]]").unwrap(),
      "a^(-1) + b^(-1) + c^(-1)"
    );
    assert_eq!(
      interpret("Mean[HypoexponentialDistribution[{a, b}]]").unwrap(),
      "a^(-1) + b^(-1)"
    );
  }

  #[test]
  fn hypoexponential_variance_symbolic() {
    // Variance of independent exponentials is sum_i 1/lambda_i^2.
    assert_eq!(
      interpret("Variance[HypoexponentialDistribution[{a, b, c}]]").unwrap(),
      "a^(-2) + b^(-2) + c^(-2)"
    );
  }

  #[test]
  fn hypoexponential_numeric_values() {
    // Concrete rates collapse to a single rational.
    assert_eq!(
      interpret("Mean[HypoexponentialDistribution[{1, 2, 3}]]").unwrap(),
      "11/6"
    );
    assert_eq!(
      interpret("Mean[HypoexponentialDistribution[{1, 2}]]").unwrap(),
      "3/2"
    );
  }

  #[test]
  fn inverse_gaussian_mean_variance() {
    // Mean = mu; Variance = mu^3 / lambda.
    assert_eq!(
      interpret("Mean[InverseGaussianDistribution[m, l]]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("Variance[InverseGaussianDistribution[m, l]]").unwrap(),
      "m^3/l"
    );
    // Concrete parameters.
    assert_eq!(
      interpret("Mean[InverseGaussianDistribution[2, 3]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Variance[InverseGaussianDistribution[2, 3]]").unwrap(),
      "8/3"
    );
  }

  #[test]
  fn logistic_mean_variance_median() {
    // Mean = mu; Variance = beta^2 * Pi^2 / 3; Median = mu.
    assert_eq!(interpret("Mean[LogisticDistribution[m, b]]").unwrap(), "m");
    assert_eq!(
      interpret("Variance[LogisticDistribution[m, b]]").unwrap(),
      "(b^2*Pi^2)/3"
    );
    assert_eq!(
      interpret("Median[LogisticDistribution[m, b]]").unwrap(),
      "m"
    );
    // Zero-arg form defaults to mu = 0, beta = 1.
    assert_eq!(interpret("Mean[LogisticDistribution[]]").unwrap(), "0");
    assert_eq!(
      interpret("Variance[LogisticDistribution[]]").unwrap(),
      "Pi^2/3"
    );
  }

  #[test]
  fn extreme_value_mean_variance_median_symbolic() {
    // Mean = a + b*EulerGamma; Variance = (b^2 Pi^2)/6;
    // Median = a - b*Log[Log[2]].
    assert_eq!(
      interpret("Mean[ExtremeValueDistribution[a, b]]").unwrap(),
      "a + b*EulerGamma"
    );
    assert_eq!(
      interpret("Variance[ExtremeValueDistribution[a, b]]").unwrap(),
      "(b^2*Pi^2)/6"
    );
    assert_eq!(
      interpret("Median[ExtremeValueDistribution[a, b]]").unwrap(),
      "a - b*Log[Log[2]]"
    );
  }

  #[test]
  fn extreme_value_mean_variance_median_numeric() {
    // Concrete parameters collapse to closed form.
    assert_eq!(
      interpret("Mean[ExtremeValueDistribution[1, 2]]").unwrap(),
      "1 + 2*EulerGamma"
    );
    assert_eq!(
      interpret("Variance[ExtremeValueDistribution[1, 2]]").unwrap(),
      "(2*Pi^2)/3"
    );
    assert_eq!(
      interpret("Median[ExtremeValueDistribution[1, 2]]").unwrap(),
      "1 - 2*Log[Log[2]]"
    );
  }

  #[test]
  fn inverse_gamma_mean_variance_median_symbolic() {
    // Mean exists only for a > 1; the Piecewise default is Indeterminate.
    assert_eq!(
      interpret("Mean[InverseGammaDistribution[a, b]]").unwrap(),
      "Piecewise[{{b/(-1 + a), a > 1}}, Indeterminate]"
    );
    // Variance exists only for a > 2.
    assert_eq!(
      interpret("Variance[InverseGammaDistribution[a, b]]").unwrap(),
      "Piecewise[{{b^2/((-2 + a)*(-1 + a)^2), a > 2}}, Indeterminate]"
    );
    // Median is defined for all a > 0 via the regularized inverse gamma.
    assert_eq!(
      interpret("Median[InverseGammaDistribution[a, b]]").unwrap(),
      "b/InverseGammaRegularized[a, 1/2]"
    );
  }

  #[test]
  fn inverse_gamma_mean_variance_numeric_branches() {
    // a = 3 > 2, so both Mean and Variance collapse to the first branch.
    assert_eq!(
      interpret("Mean[InverseGammaDistribution[3, 2]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Variance[InverseGammaDistribution[3, 2]]").unwrap(),
      "1"
    );
    // a = 1/2 < 1 → Mean falls through to the Indeterminate default.
    assert_eq!(
      interpret("Mean[InverseGammaDistribution[1/2, 2]]").unwrap(),
      "Indeterminate"
    );
    // 1 < a = 3/2 < 2 → Variance is Indeterminate even though Mean is finite.
    assert_eq!(
      interpret("Variance[InverseGammaDistribution[3/2, 2]]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn inverse_gamma_median_numeric() {
    // Median stays symbolic because InverseGammaRegularized is itself
    // symbolic for these arguments.
    assert_eq!(
      interpret("Median[InverseGammaDistribution[3, 2]]").unwrap(),
      "2/InverseGammaRegularized[3, 1/2]"
    );
  }

  #[test]
  fn gompertz_makeham_mean_and_median_symbolic() {
    // Gompertz (two-arg) form: Mean = (E^xi*Gamma[0, xi])/lambda;
    // Median = Log[1 + Log[2]/xi]/lambda.
    assert_eq!(
      interpret("Mean[GompertzMakehamDistribution[l, x]]").unwrap(),
      "(E^x*Gamma[0, x])/l"
    );
    assert_eq!(
      interpret("Median[GompertzMakehamDistribution[l, x]]").unwrap(),
      "Log[1 + Log[2]/x]/l"
    );
  }

  #[test]
  fn gompertz_makeham_mean_and_median_numeric() {
    // Concrete parameters keep the closed form unevaluated where
    // wolframscript also keeps it symbolic (Gamma[0, 3], Log[2]/3).
    assert_eq!(
      interpret("Mean[GompertzMakehamDistribution[2, 3]]").unwrap(),
      "(E^3*Gamma[0, 3])/2"
    );
    assert_eq!(
      interpret("Median[GompertzMakehamDistribution[2, 3]]").unwrap(),
      "Log[1 + Log[2]/3]/2"
    );
  }

  #[test]
  fn extreme_value_zero_arg_form() {
    // Zero-arg form defaults to a = 0, b = 1.
    assert_eq!(
      interpret("ExtremeValueDistribution[]").unwrap(),
      "ExtremeValueDistribution[0, 1]"
    );
    assert_eq!(
      interpret("Mean[ExtremeValueDistribution[]]").unwrap(),
      "EulerGamma"
    );
    assert_eq!(
      interpret("Variance[ExtremeValueDistribution[]]").unwrap(),
      "Pi^2/6"
    );
    assert_eq!(
      interpret("Median[ExtremeValueDistribution[]]").unwrap(),
      "-Log[Log[2]]"
    );
  }

  #[test]
  fn harmonic_mean_matrix() {
    // List-of-lists input → column-wise harmonic mean.
    assert_eq!(
      interpret("HarmonicMean[{{1, 2}, {5, 10}, {5, 2}, {4, 8}}]").unwrap(),
      "{80/33, 160/49}"
    );
    assert_eq!(
      interpret("HarmonicMean[{{a, b}, {c, d}}]").unwrap(),
      "{2/(a^(-1) + c^(-1)), 2/(b^(-1) + d^(-1))}"
    );
  }
}
