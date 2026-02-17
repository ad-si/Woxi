use super::*;

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

  #[test]
  fn table_symbolic_pi_step() {
    assert_eq!(
      interpret("Table[θ, {θ, 0, 2 Pi - Pi/4, Pi/4}]").unwrap(),
      "{0, Pi/4, 2*Pi/4, 3*Pi/4, 4*Pi/4, 5*Pi/4, 6*Pi/4, 7*Pi/4}"
    );
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

mod tuples_extended {
  use super::*;

  #[test]
  fn list_of_lists() {
    assert_eq!(
      interpret("Tuples[{{a, b}, {1, 2, 3}}]").unwrap(),
      "{{a, 1}, {a, 2}, {a, 3}, {b, 1}, {b, 2}, {b, 3}}"
    );
  }

  #[test]
  fn function_head() {
    assert_eq!(
      interpret("Tuples[f[a, b, c], 2]").unwrap(),
      "{f[a, a], f[a, b], f[a, c], f[b, a], f[b, b], f[b, c], f[c, a], f[c, b], f[c, c]}"
    );
  }

  #[test]
  fn list_of_function_heads() {
    assert_eq!(
      interpret("Tuples[{f[a, b], g[c, d]}]").unwrap(),
      "{{a, c}, {a, d}, {b, c}, {b, d}}"
    );
  }
}

mod map_thread_extended {
  use super::*;

  #[test]
  fn with_level() {
    assert_eq!(
      interpret("MapThread[f, {{{a, b}, {c, d}}, {{e, f}, {g, h}}}, 2]")
        .unwrap(),
      "{{f[a, e], f[b, f]}, {f[c, g], f[d, h]}}"
    );
  }
}

mod inner_extended {
  use super::*;

  #[test]
  fn matrix_and_or() {
    assert_eq!(
      interpret("Inner[And, {{False, False}, {False, True}}, {{True, False}, {True, True}}, Or]")
        .unwrap(),
      "{{False, False}, {True, True}}"
    );
  }

  #[test]
  fn nested_lists() {
    assert_eq!(
      interpret("Inner[f, {{{a, b}}, {{c, d}}}, {{1}, {2}}, g]").unwrap(),
      "{{{g[f[a, 1], f[b, 2]]}}, {{g[f[c, 1], f[d, 2]]}}}"
    );
  }
}

mod outer_extended {
  use super::*;

  #[test]
  fn matrix_times() {
    assert_eq!(
      interpret("Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]").unwrap(),
      "{{{{a, 2*a}, {3*a, 4*a}}, {{b, 2*b}, {3*b, 4*b}}}, {{{c, 2*c}, {3*c, 4*c}}, {{d, 2*d}, {3*d, 4*d}}}}"
    );
  }

  #[test]
  fn three_lists() {
    assert_eq!(
      interpret("Outer[f, {a, b}, {x, y, z}, {1, 2}]").unwrap(),
      "{{{f[a, x, 1], f[a, x, 2]}, {f[a, y, 1], f[a, y, 2]}, {f[a, z, 1], f[a, z, 2]}}, {{f[b, x, 1], f[b, x, 2]}, {f[b, y, 1], f[b, y, 2]}, {f[b, z, 1], f[b, z, 2]}}}"
    );
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

  #[test]
  fn sort_by_string_length_tiebreaker() {
    assert_eq!(
      interpret(
        r#"SortBy[{"Four", "score", "seven", "years", "forth"}, StringLength]"#
      )
      .unwrap(),
      "{Four, forth, score, seven, years}"
    );
  }

  #[test]
  fn sort_by_string_length_three_char() {
    assert_eq!(
      interpret(r#"SortBy[{"and", "ago", "our"}, StringLength]"#).unwrap(),
      "{ago, and, our}"
    );
  }
}

mod sort_canonical {
  use super::*;

  #[test]
  fn sort_strings_case_insensitive() {
    assert_eq!(
      interpret(r#"Sort[{"Four", "score", "seven", "years", "forth"}]"#)
        .unwrap(),
      "{forth, Four, score, seven, years}"
    );
  }

  #[test]
  fn sort_strings_lowercase_before_uppercase() {
    assert_eq!(
      interpret(r#"Sort[{"abc", "ABC", "Abc", "aBc"}]"#).unwrap(),
      "{abc, aBc, Abc, ABC}"
    );
  }

  #[test]
  fn sort_numbers() {
    assert_eq!(
      interpret("Sort[{3, 1, 4, 1, 5, 9}]").unwrap(),
      "{1, 1, 3, 4, 5, 9}"
    );
  }

  #[test]
  fn sort_mixed_with_complex() {
    assert_eq!(
      interpret("Sort[{4, 1.0, a, 3+I}]").unwrap(),
      "{1., 3 + I, 4, a}"
    );
  }

  #[test]
  fn sort_complex_same_real_part() {
    assert_eq!(
      interpret("Sort[{3+I, 3, 3-I}]").unwrap(),
      "{3, 3 - I, 3 + I}"
    );
  }

  #[test]
  fn sort_pure_imaginary() {
    assert_eq!(interpret("Sort[{I, -I, 1, -1}]").unwrap(), "{-1, -I, I, 1}");
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

  #[test]
  fn count_with_head_pattern() {
    assert_eq!(interpret("Count[{1, a, 2, b, 3}, _Integer]").unwrap(), "3");
  }

  #[test]
  fn count_with_symbol_pattern() {
    assert_eq!(interpret("Count[{1, a, 2, b, 3}, _Symbol]").unwrap(), "2");
  }

  #[test]
  fn count_with_blank() {
    assert_eq!(interpret("Count[{1, 2, 3}, _]").unwrap(), "3");
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

  #[test]
  fn part_all_flat_list() {
    // Part[list, All] returns the list as-is
    assert_eq!(interpret("{a, b, c}[[All]]").unwrap(), "{a, b, c}");
  }

  #[test]
  fn part_all_column_extraction() {
    // Part[matrix, All, i] extracts column i
    assert_eq!(
      interpret("{{1, 2}, {3, 4}, {5, 6}}[[All, 1]]").unwrap(),
      "{1, 3, 5}"
    );
    assert_eq!(
      interpret("{{1, 2}, {3, 4}, {5, 6}}[[All, 2]]").unwrap(),
      "{2, 4, 6}"
    );
  }

  #[test]
  fn part_row_then_all() {
    // Part[matrix, i, All] returns the whole row
    assert_eq!(
      interpret("{{1, 2, 3}, {4, 5, 6}}[[1, All]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn part_all_all_identity() {
    // Part[matrix, All, All] returns the matrix as-is
    assert_eq!(
      interpret("{{a, b, c}, {d, e, f}}[[All, All]]").unwrap(),
      "{{a, b, c}, {d, e, f}}"
    );
  }

  #[test]
  fn part_all_3d_array() {
    // Part[3d, All, All, 1] extracts first element at deepest level
    assert_eq!(
      interpret("{{{a, b}, {c, d}}, {{e, f}, {g, h}}}[[All, All, 1]]").unwrap(),
      "{{a, c}, {e, g}}"
    );
  }

  #[test]
  fn part_all_on_function_call() {
    assert_eq!(interpret("f[a, b, c][[All]]").unwrap(), "f[a, b, c]");
  }

  #[test]
  fn part_all_with_variable() {
    assert_eq!(
      interpret("x = {{1, 2}, {3, 4}}; x[[All, 1]]").unwrap(),
      "{1, 3}"
    );
  }

  #[test]
  fn all_evaluates_to_itself() {
    assert_eq!(interpret("All").unwrap(), "All");
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

  #[test]
  fn length_of_atoms() {
    // Atoms have Length 0
    assert_eq!(interpret("Length[42]").unwrap(), "0");
    assert_eq!(interpret("Length[3.14]").unwrap(), "0");
    assert_eq!(interpret(r#"Length["hello"]"#).unwrap(), "0");
    assert_eq!(interpret("Length[x]").unwrap(), "0");
    assert_eq!(interpret("Length[True]").unwrap(), "0");
  }

  #[test]
  fn length_of_function_call() {
    // Length counts top-level arguments of any head
    assert_eq!(interpret("Length[f[a, b, c]]").unwrap(), "3");
    assert_eq!(interpret("Length[f[]]").unwrap(), "0");
    assert_eq!(interpret("Length[g[x]]").unwrap(), "1");
    assert_eq!(interpret("Length[Plus[a, b, c]]").unwrap(), "3");
  }

  #[test]
  fn length_of_symbolic_expressions() {
    // Symbolic arithmetic expressions
    assert_eq!(interpret("Length[a + b]").unwrap(), "2");
    assert_eq!(interpret("Length[a + b + c]").unwrap(), "3");
    assert_eq!(interpret("Length[a * b * c]").unwrap(), "3");
    assert_eq!(interpret("Length[a^b]").unwrap(), "2");
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

mod take_multi_dim {
  use super::*;

  #[test]
  fn take_all() {
    assert_eq!(
      interpret("Take[{1, 2, 3, 4, 5}, All]").unwrap(),
      "{1, 2, 3, 4, 5}"
    );
  }

  #[test]
  fn take_all_column() {
    assert_eq!(
      interpret("Take[{{1, 3}, {5, 7}}, All, {1}]").unwrap(),
      "{{1}, {5}}"
    );
  }

  #[test]
  fn take_multi_dim_range() {
    assert_eq!(
      interpret("Take[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 2, {1, 2}]").unwrap(),
      "{{1, 2}, {4, 5}}"
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

  #[test]
  fn constant_positive() {
    assert_eq!(interpret("UnitStep[Pi]").unwrap(), "1");
    assert_eq!(interpret("UnitStep[E]").unwrap(), "1");
    assert_eq!(interpret("UnitStep[Infinity]").unwrap(), "1");
  }

  #[test]
  fn constant_negative() {
    assert_eq!(interpret("UnitStep[-Pi]").unwrap(), "0");
    assert_eq!(interpret("UnitStep[-E]").unwrap(), "0");
    assert_eq!(interpret("UnitStep[-Infinity]").unwrap(), "0");
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
      interpret(r#"Reap[Sow[1, "a"]; Sow[2, "b"]; Sow[3, "a"], "a"]"#).unwrap(),
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

mod random_real {
  use super::*;

  #[test]
  fn no_args() {
    let result: f64 = interpret("RandomReal[]").unwrap().parse().unwrap();
    assert!(result >= 0.0 && result < 1.0);
  }

  #[test]
  fn with_max() {
    let result: f64 = interpret("RandomReal[5]").unwrap().parse().unwrap();
    assert!(result >= 0.0 && result < 5.0);
  }

  #[test]
  fn with_range() {
    let result: f64 = interpret("RandomReal[{2, 5}]").unwrap().parse().unwrap();
    assert!(result >= 2.0 && result < 5.0);
  }

  #[test]
  fn list_with_max() {
    assert_eq!(interpret("Length[RandomReal[1, 10]]").unwrap(), "10");
  }

  #[test]
  fn list_with_range() {
    assert_eq!(interpret("Length[RandomReal[{0, 1}, 50]]").unwrap(), "50");
  }

  #[test]
  fn list_values_in_range() {
    // All values should be between 3 and 7
    assert_eq!(
      interpret("AllTrue[RandomReal[{3, 7}, 100], (# >= 3 && # < 7) &]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn mean_of_random_reals() {
    // Mean of RandomReal[1, 1000] should be a real number
    let result: f64 = interpret("Mean[RandomReal[1, 1000]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!(result > 0.0 && result < 1.0);
  }
}

mod random_integer {
  use super::*;

  #[test]
  fn no_args() {
    let result: i128 = interpret("RandomInteger[]").unwrap().parse().unwrap();
    assert!(result == 0 || result == 1);
  }

  #[test]
  fn with_max() {
    let result: i128 = interpret("RandomInteger[10]").unwrap().parse().unwrap();
    assert!(result >= 0 && result <= 10);
  }

  #[test]
  fn list_form() {
    assert_eq!(interpret("Length[RandomInteger[10, 5]]").unwrap(), "5");
  }
}

mod distributions {
  use super::*;

  #[test]
  fn uniform_distribution_inert() {
    assert_eq!(
      interpret("UniformDistribution[{0, 1}]").unwrap(),
      "UniformDistribution[{0, 1}]"
    );
  }

  #[test]
  fn normal_distribution_default() {
    assert_eq!(
      interpret("NormalDistribution[]").unwrap(),
      "NormalDistribution[0, 1]"
    );
  }

  #[test]
  fn normal_distribution_with_params() {
    assert_eq!(
      interpret("NormalDistribution[5, 2]").unwrap(),
      "NormalDistribution[5, 2]"
    );
  }
}

mod random_variate {
  use super::*;

  #[test]
  fn uniform_single() {
    let result: f64 = interpret("RandomVariate[UniformDistribution[{0, 1}]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!(result >= 0.0 && result < 1.0);
  }

  #[test]
  fn uniform_list() {
    assert_eq!(
      interpret("Length[RandomVariate[UniformDistribution[{0, 1}], 100]]")
        .unwrap(),
      "100"
    );
  }

  #[test]
  fn uniform_values_in_range() {
    assert_eq!(
      interpret(
        "AllTrue[RandomVariate[UniformDistribution[{3, 7}], 100], (# >= 3 && # < 7) &]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn normal_single() {
    let result: f64 = interpret("RandomVariate[NormalDistribution[]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!(result.is_finite());
  }

  #[test]
  fn normal_list() {
    assert_eq!(
      interpret("Length[RandomVariate[NormalDistribution[0, 1], 50]]").unwrap(),
      "50"
    );
  }

  #[test]
  fn normal_with_params() {
    // Mean of 1000 samples from N(100, 1) should be near 100
    let result: f64 =
      interpret("Mean[RandomVariate[NormalDistribution[100, 1], 1000]]")
        .unwrap()
        .parse()
        .unwrap();
    assert!(result > 95.0 && result < 105.0);
  }

  #[test]
  fn combined_mean_stddev() {
    // The original failing expression
    let result = interpret(
      "data = RandomVariate[UniformDistribution[{0, 1}], 1000]; {Mean[data], StandardDeviation[data]}"
    ).unwrap();
    assert!(result.starts_with('{'));
    assert!(result.ends_with('}'));
  }
}

mod select {
  use super::*;

  #[test]
  fn basic_predicate() {
    assert_eq!(
      interpret("Select[{1, 2, 3, 4, 5}, OddQ]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn even_q() {
    assert_eq!(
      interpret("Select[{1, 2, 3, 4, 5}, EvenQ]").unwrap(),
      "{2, 4}"
    );
  }

  #[test]
  fn anonymous_function() {
    assert_eq!(
      interpret("Select[{1, 4, 2, 3}, # > 1 &]").unwrap(),
      "{4, 2, 3}"
    );
  }

  #[test]
  fn with_limit() {
    assert_eq!(
      interpret("Select[{1, 2, 3, 4, 5}, OddQ, 2]").unwrap(),
      "{1, 3}"
    );
  }

  #[test]
  fn with_limit_exceeding() {
    assert_eq!(
      interpret("Select[{1, 2, 3, 4, 5}, EvenQ, 10]").unwrap(),
      "{2, 4}"
    );
  }

  #[test]
  fn with_limit_one() {
    assert_eq!(
      interpret("Select[{1, 2, 3, 4, 5}, # > 3 &, 1]").unwrap(),
      "{4}"
    );
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("Select[{}, EvenQ]").unwrap(), "{}");
  }

  #[test]
  fn no_matches() {
    assert_eq!(interpret("Select[{1, 3, 5}, EvenQ]").unwrap(), "{}");
  }

  #[test]
  fn operator_form() {
    assert_eq!(interpret("Select[EvenQ][{1, 2, 3, 4}]").unwrap(), "{2, 4}");
  }

  #[test]
  fn with_prime_q() {
    assert_eq!(
      interpret("Select[Range[20], PrimeQ]").unwrap(),
      "{2, 3, 5, 7, 11, 13, 17, 19}"
    );
  }

  #[test]
  fn string_select() {
    assert_eq!(
      interpret(
        "Select[{\"apple\", \"avocado\", \"banana\"}, StringStartsQ[\"a\"]]"
      )
      .unwrap(),
      "{apple, avocado}"
    );
  }

  #[test]
  fn negative_numbers() {
    assert_eq!(
      interpret("Select[{-3, -1, 0, 2, 5}, # > 0 &]").unwrap(),
      "{2, 5}"
    );
  }
}

mod subset_q {
  use super::*;

  #[test]
  fn empty_sets() {
    assert_eq!(interpret("SubsetQ[{}, {}]").unwrap(), "True");
  }

  #[test]
  fn subset_true() {
    assert_eq!(interpret("SubsetQ[{1, 2, 3}, {1, 2}]").unwrap(), "True");
  }

  #[test]
  fn subset_false() {
    assert_eq!(interpret("SubsetQ[{1, 2}, {1, 2, 3}]").unwrap(), "False");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("SubsetQ[{a, b, c}, {a, c}]").unwrap(), "True");
  }
}

mod option_q {
  use super::*;

  #[test]
  fn rule() {
    assert_eq!(interpret("OptionQ[a -> True]").unwrap(), "True");
  }

  #[test]
  fn rule_delayed() {
    assert_eq!(interpret("OptionQ[a :> True]").unwrap(), "True");
  }

  #[test]
  fn list_of_rules() {
    assert_eq!(interpret("OptionQ[{a -> 1, b -> 2}]").unwrap(), "True");
  }

  #[test]
  fn not_option() {
    assert_eq!(interpret("OptionQ[3]").unwrap(), "False");
  }
}

mod fold_list {
  use super::*;

  #[test]
  fn two_arg_times() {
    assert_eq!(
      interpret("FoldList[Times, {1, 2, 3}]").unwrap(),
      "{1, 2, 6}"
    );
  }

  #[test]
  fn two_arg_plus() {
    assert_eq!(
      interpret("FoldList[Plus, {1, 2, 3, 4}]").unwrap(),
      "{1, 3, 6, 10}"
    );
  }

  #[test]
  fn two_arg_fold() {
    assert_eq!(interpret("Fold[Plus, {1, 2, 3, 4}]").unwrap(), "10");
  }
}

mod split_with_test {
  use super::*;

  #[test]
  fn less() {
    assert_eq!(
      interpret("Split[{1, 5, 6, 3, 6, 1, 6, 3, 4, 5, 4}, Less]").unwrap(),
      "{{1, 5, 6}, {3, 6}, {1, 6}, {3, 4, 5}, {4}}"
    );
  }
}

mod first_position {
  use super::*;

  #[test]
  fn simple() {
    assert_eq!(interpret("FirstPosition[{a, b, c, d}, c]").unwrap(), "{3}");
  }

  #[test]
  fn nested() {
    assert_eq!(
      interpret("FirstPosition[{{a, a, b}, {b, a, a}, {a, b, a}}, b]").unwrap(),
      "{1, 3}"
    );
  }

  #[test]
  fn not_found() {
    assert_eq!(
      interpret("FirstPosition[{a, b, c}, z]").unwrap(),
      "Missing[NotFound]"
    );
  }
}

mod ranked {
  use super::*;

  #[test]
  fn ranked_max() {
    assert_eq!(
      interpret("RankedMax[{482, 17, 181, -12}, 2]").unwrap(),
      "181"
    );
  }

  #[test]
  fn ranked_min() {
    assert_eq!(
      interpret("RankedMin[{482, 17, 181, -12}, 2]").unwrap(),
      "17"
    );
  }
}

mod quantile {
  use super::*;

  #[test]
  fn single() {
    assert_eq!(
      interpret("Quantile[{1, 2, 3, 4, 5, 6, 7}, 1/4]").unwrap(),
      "2"
    );
  }

  #[test]
  fn multiple() {
    assert_eq!(
      interpret("Quantile[{1, 2, 3, 4, 5, 6, 7}, {1/4, 3/4}]").unwrap(),
      "{2, 6}"
    );
  }
}

mod range {
  use super::*;

  #[test]
  fn range_single() {
    assert_eq!(interpret("Range[5]").unwrap(), "{1, 2, 3, 4, 5}");
  }

  #[test]
  fn range_min_max() {
    assert_eq!(interpret("Range[3, 7]").unwrap(), "{3, 4, 5, 6, 7}");
  }

  #[test]
  fn range_with_step() {
    assert_eq!(interpret("Range[1, 10, 2]").unwrap(), "{1, 3, 5, 7, 9}");
  }

  #[test]
  fn range_negative_step() {
    assert_eq!(interpret("Range[5, 1, -1]").unwrap(), "{5, 4, 3, 2, 1}");
  }

  #[test]
  fn range_rational_step() {
    assert_eq!(
      interpret("Range[0, 2, 1/3]").unwrap(),
      "{0, 1/3, 2/3, 1, 4/3, 5/3, 2}"
    );
  }

  #[test]
  fn range_rational_step_half() {
    assert_eq!(
      interpret("Range[0, 2, 1/2]").unwrap(),
      "{0, 1/2, 1, 3/2, 2}"
    );
  }

  #[test]
  fn range_from_zero() {
    assert_eq!(interpret("Range[0]").unwrap(), "{}");
  }

  #[test]
  fn range_negative_to_positive() {
    assert_eq!(interpret("Range[-2, 2]").unwrap(), "{-2, -1, 0, 1, 2}");
  }
}

mod intersection_sorting {
  use super::*;

  #[test]
  fn intersection_sorts_result() {
    assert_eq!(interpret("Intersection[{c, b, a}]").unwrap(), "{a, b, c}");
  }

  #[test]
  fn intersection_sorts_numeric() {
    assert_eq!(
      interpret("Intersection[{1000, 100, 10, 1}, {1, 5, 10, 15}]").unwrap(),
      "{1, 10}"
    );
  }
}

mod contains_only {
  use super::*;

  #[test]
  fn contains_only_true() {
    assert_eq!(
      interpret("ContainsOnly[{1, 2, 3}, {1, 2, 3, 4}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn contains_only_false() {
    assert_eq!(
      interpret("ContainsOnly[{1, 2, 3}, {1, 2}]").unwrap(),
      "False"
    );
  }
}

mod length_while {
  use super::*;

  #[test]
  fn length_while_basic() {
    assert_eq!(
      interpret("LengthWhile[{1, 2, 3, 4, 5}, # < 3 &]").unwrap(),
      "2"
    );
  }

  #[test]
  fn length_while_none() {
    assert_eq!(interpret("LengthWhile[{1, 2, 3}, # > 5 &]").unwrap(), "0");
  }

  #[test]
  fn length_while_all() {
    assert_eq!(interpret("LengthWhile[{1, 2, 3}, # < 5 &]").unwrap(), "3");
  }
}

mod take_largest_by {
  use super::*;

  #[test]
  fn take_largest_by_abs() {
    assert_eq!(
      interpret("TakeLargestBy[{-1, -2, -3, 4, 5}, Abs, 2]").unwrap(),
      "{5, 4}"
    );
  }
}

mod take_smallest_by {
  use super::*;

  #[test]
  fn take_smallest_by_abs() {
    assert_eq!(
      interpret("TakeSmallestBy[{-1, -2, -3, 4, 5}, Abs, 2]").unwrap(),
      "{-1, -2}"
    );
  }
}

mod pick {
  use super::*;

  #[test]
  fn pick_basic() {
    assert_eq!(
      interpret("Pick[{a, b, c}, {False, True, False}]").unwrap(),
      "{b}"
    );
  }

  #[test]
  fn pick_nested() {
    assert_eq!(
      interpret("Pick[f[g[1, 2], h[3, 4]], {{True, False}, {False, True}}]")
        .unwrap(),
      "f[g[1], h[4]]"
    );
  }

  #[test]
  fn pick_with_pattern() {
    assert_eq!(
      interpret("Pick[{a, b, c, d, e}, {1, 2, 3.5, 4, 5.5}, _Integer]")
        .unwrap(),
      "{a, b, d}"
    );
  }
}

mod rest_nonlist {
  use super::*;

  #[test]
  fn rest_plus() {
    assert_eq!(interpret("Rest[a + b + c]").unwrap(), "b + c");
  }

  #[test]
  fn rest_error_atomic() {
    assert!(interpret("Rest[x]").is_err());
  }

  #[test]
  fn rest_error_empty() {
    assert!(interpret("Rest[{}]").is_err());
  }
}

mod level {
  use super::*;

  #[test]
  fn level_atoms() {
    assert_eq!(
      interpret("Level[a + b ^ 3 * f[2 x ^ 2], {-1}]").unwrap(),
      "{a, b, 3, 2, x, 2}"
    );
  }

  #[test]
  fn level_positive() {
    assert_eq!(
      interpret("Level[{{{{a}}}}, 3]").unwrap(),
      "{{a}, {{a}}, {{{a}}}}"
    );
  }

  #[test]
  fn level_negative() {
    assert_eq!(interpret("Level[{{{{a}}}}, -4]").unwrap(), "{{{{a}}}}");
  }

  #[test]
  fn level_negative_empty() {
    assert_eq!(interpret("Level[{{{{a}}}}, -5]").unwrap(), "{}");
  }

  #[test]
  fn level_range_with_zero() {
    assert_eq!(
      interpret("Level[h0[h1[h2[h3[a]]]], {0, -1}]").unwrap(),
      "{a, h3[a], h2[h3[a]], h1[h2[h3[a]]], h0[h1[h2[h3[a]]]]}"
    );
  }

  #[test]
  fn level_heads_list() {
    assert_eq!(
      interpret("Level[{{{{a}}}}, 3, Heads -> True]").unwrap(),
      "{List, List, List, {a}, {{a}}, {{{a}}}}"
    );
  }

  #[test]
  fn level_heads_expr() {
    assert_eq!(
      interpret("Level[x^2 + y^3, 3, Heads -> True]").unwrap(),
      "{Plus, Power, x, 2, x^2, Power, y, 3, y^3}"
    );
  }

  #[test]
  fn level_curried_heads() {
    assert_eq!(
      interpret("Level[f[g[h]][x], {-1}, Heads -> True]").unwrap(),
      "{f, g, h, x}"
    );
  }

  #[test]
  fn level_curried_heads_range() {
    assert_eq!(
      interpret("Level[f[g[h]][x], {-2, -1}, Heads -> True]").unwrap(),
      "{f, g, h, g[h], x, f[g[h]][x]}"
    );
  }
}

mod median_extended {
  use super::*;

  #[test]
  fn median_matrix() {
    assert_eq!(
      interpret("Median[{{100, 1, 10, 50}, {-1, 1, -2, 2}}]").unwrap(),
      "{99/2, 1, 4, 26}"
    );
  }
}

mod linear_recurrence {
  use super::*;

  #[test]
  fn fibonacci_sequence() {
    assert_eq!(
      interpret("LinearRecurrence[{1, 1}, {1, 1}, 10]").unwrap(),
      "{1, 1, 2, 3, 5, 8, 13, 21, 34, 55}"
    );
  }

  #[test]
  fn range_slice() {
    assert_eq!(
      interpret("LinearRecurrence[{1, 1}, {1, 1}, {3, 5}]").unwrap(),
      "{2, 3, 5}"
    );
  }

  #[test]
  fn single_element() {
    assert_eq!(
      interpret("LinearRecurrence[{1, 1}, {1, 1}, {6}]").unwrap(),
      "{8}"
    );
  }

  #[test]
  fn tribonacci() {
    assert_eq!(
      interpret("LinearRecurrence[{1, 1, 1}, {1, 1, 1}, 8]").unwrap(),
      "{1, 1, 1, 3, 5, 9, 17, 31}"
    );
  }
}

mod range_real {
  use super::*;

  #[test]
  fn range_real_start() {
    assert_eq!(interpret("Range[1.0, 2.3]").unwrap(), "{1., 2.}");
  }

  #[test]
  fn range_real_step() {
    assert_eq!(interpret("Range[1.0, 2.3, .5]").unwrap(), "{1., 1.5, 2.}");
  }
}

mod delete_deep {
  use super::*;

  #[test]
  fn delete_multi_part_index() {
    assert_eq!(
      interpret("Delete[{{a, b}, {c, d}}, {2, 1}]").unwrap(),
      "{{a, b}, {d}}"
    );
  }

  #[test]
  fn delete_multiple_positions() {
    assert_eq!(
      interpret("Delete[{a, b, c, d}, {{1}, {3}}]").unwrap(),
      "{b, d}"
    );
  }
}

mod extract_multi {
  use super::*;

  #[test]
  fn extract_multiple_positions() {
    assert_eq!(
      interpret("Extract[{{a, b}, {c, d}}, {{1}, {2, 2}}]").unwrap(),
      "{{a, b}, d}"
    );
  }
}

mod matrix_constructors {
  use super::*;

  #[test]
  fn diamond_matrix_2() {
    assert_eq!(
      interpret("DiamondMatrix[2]").unwrap(),
      "{{0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}}"
    );
  }

  #[test]
  fn disk_matrix_2() {
    assert_eq!(
      interpret("DiskMatrix[2]").unwrap(),
      "{{0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}}"
    );
  }
}

mod part_list_index {
  use super::*;

  #[test]
  fn part_with_list_index() {
    assert_eq!(interpret("{a, b, c, d}[[{1, 3, 3}]]").unwrap(), "{a, c, c}");
  }

  #[test]
  fn part_with_list_index_function_call() {
    assert_eq!(
      interpret("Part[{a, b, c, d}, {1, 3, 3}]").unwrap(),
      "{a, c, c}"
    );
  }
}

mod join_non_list {
  use super::*;

  #[test]
  fn join_plus_heads() {
    assert_eq!(
      interpret("Join[a + b, c + d, e + f]").unwrap(),
      "a + b + c + d + e + f"
    );
  }

  #[test]
  fn most_non_list_head() {
    assert_eq!(interpret("Most[a + b + c]").unwrap(), "a + b");
  }

  #[test]
  fn riffle_cycling_separator() {
    assert_eq!(
      interpret("Riffle[{a, b, c, d, e, f}, {x, y, z}]").unwrap(),
      "{a, x, b, y, c, z, d, x, e, y, f}"
    );
  }

  #[test]
  fn riffle_same_length_lists() {
    assert_eq!(
      interpret("Riffle[{a, b, c}, {x, y, z}]").unwrap(),
      "{a, x, b, y, c, z}"
    );
  }

  #[test]
  fn thread_with_head() {
    assert_eq!(
      interpret("Thread[f[a + b + c], Plus]").unwrap(),
      "f[a] + f[b] + f[c]"
    );
  }

  #[test]
  fn map_at_deep_position() {
    assert_eq!(
      interpret("MapAt[0&, {{1, 1}, {1, 1}}, {2, 1}]").unwrap(),
      "{{1, 1}, {0, 1}}"
    );
  }

  #[test]
  fn map_at_multiple_deep_positions() {
    assert_eq!(
      interpret("MapAt[f, {a, {b, c}}, {{2, 1}}]").unwrap(),
      "{a, {f[b], c}}"
    );
  }

  #[test]
  fn rotate_left_multi_dim() {
    assert_eq!(
      interpret("RotateLeft[{{a, b, c}, {d, e, f}, {g, h, i}}, {1, 2}]")
        .unwrap(),
      "{{f, d, e}, {i, g, h}, {c, a, b}}"
    );
  }

  #[test]
  fn rotate_right_multi_dim() {
    assert_eq!(
      interpret("RotateRight[{{a, b, c}, {d, e, f}, {g, h, i}}, {1, 2}]")
        .unwrap(),
      "{{h, i, g}, {b, c, a}, {e, f, d}}"
    );
  }

  #[test]
  fn levi_civita_tensor_list() {
    assert_eq!(
      interpret("LeviCivitaTensor[3, List]").unwrap(),
      "{{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}, {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}}, {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}}"
    );
  }

  #[test]
  fn count_at_level() {
    assert_eq!(
      interpret("Count[{{a, a}, {a, a, a}, a}, a, {2}]").unwrap(),
      "5"
    );
  }

  #[test]
  fn apply_at_level_0() {
    assert_eq!(interpret("Apply[f, {a, b, c}, {0}]").unwrap(), "f[a, b, c]");
  }

  #[test]
  fn apply_at_level_1() {
    assert_eq!(
      interpret("Apply[f, {{1, 2}, {3, 4}}, {1}]").unwrap(),
      "{f[1, 2], f[3, 4]}"
    );
  }

  #[test]
  fn apply_at_level_1_mixed() {
    assert_eq!(
      interpret("Apply[f, {a + b, g[c, d, e * f], 3}, {1}]").unwrap(),
      "{f[a, b], f[c, d, e*f], 3}"
    );
  }

  #[test]
  fn reverse_level_1() {
    assert_eq!(
      interpret("Reverse[{{1, 2}, {3, 4}}, 1]").unwrap(),
      "{{3, 4}, {1, 2}}"
    );
  }

  #[test]
  fn reverse_level_2() {
    assert_eq!(
      interpret("Reverse[{{1, 2}, {3, 4}}, 2]").unwrap(),
      "{{2, 1}, {4, 3}}"
    );
  }

  #[test]
  fn reverse_level_1_2() {
    assert_eq!(
      interpret("Reverse[{{1, 2}, {3, 4}}, {1, 2}]").unwrap(),
      "{{4, 3}, {2, 1}}"
    );
  }

  #[test]
  fn pad_left_non_list_head() {
    assert_eq!(
      interpret("PadLeft[x[a, b, c], 5]").unwrap(),
      "x[0, 0, a, b, c]"
    );
  }

  #[test]
  fn pad_right_non_list_head() {
    assert_eq!(
      interpret("PadRight[x[a, b, c], 5]").unwrap(),
      "x[a, b, c, 0, 0]"
    );
  }

  #[test]
  fn flatten_with_head() {
    assert_eq!(
      interpret("Flatten[f[a, f[b, f[c, d]], e], Infinity, f]").unwrap(),
      "f[a, b, c, d, e]"
    );
  }

  #[test]
  fn permutations_up_to_length() {
    assert_eq!(
      interpret("Permutations[{1, 2, 3}, 2]").unwrap(),
      "{{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}"
    );
  }
}
