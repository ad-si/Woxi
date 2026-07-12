use super::*;

mod unique {
  use super::*;

  #[test]
  fn unique_no_args() {
    // Unique[] generates $nnn
    let result = interpret("Unique[]").unwrap();
    assert!(result.starts_with('$'));
    let num: u64 = result[1..].parse().unwrap();
    assert!(num > 0);
  }

  #[test]
  fn unique_with_symbol() {
    // Unique[x] generates x$nnn
    let result = interpret("Unique[x]").unwrap();
    assert!(result.starts_with("x$"));
    let num: u64 = result[2..].parse().unwrap();
    assert!(num > 0);
  }

  #[test]
  fn unique_with_string() {
    // Unique["hello"] generates hellonnn
    let result = interpret("Unique[\"hello\"]").unwrap();
    assert!(result.starts_with("hello"));
    let num_str = &result[5..];
    let num: u64 = num_str.parse().unwrap();
    assert!(num > 0);
  }

  #[test]
  fn unique_list() {
    // Unique[{a, b}] generates list of unique symbols
    let result = interpret("Unique[{a, b}]").unwrap();
    assert!(result.starts_with('{'));
    assert!(result.contains("a$"));
    assert!(result.contains("b$"));
  }

  #[test]
  fn unique_successive_different() {
    // Two calls to Unique give different symbols
    let result = interpret("{Unique[x], Unique[x]}").unwrap();
    // Parse the two symbols
    let inner = &result[1..result.len() - 1]; // strip { }
    let parts: Vec<&str> = inner.split(", ").collect();
    assert_eq!(parts.len(), 2);
    assert_ne!(parts[0], parts[1]);
  }

  #[test]
  fn unique_symbolic_unevaluated() {
    // Non-symbol, non-string, non-list args return unevaluated
    assert_eq!(interpret("Unique[1]").unwrap(), "Unique[1]");
  }
}

// ─── Option symbols ──────────────────────────────────────────────────────────

#[test]
fn excluded_forms_is_symbol() {
  assert_eq!(interpret("ExcludedForms").unwrap(), "ExcludedForms");
}

#[test]
fn excluded_forms_as_option() {
  assert_eq!(
    interpret("ExcludedForms -> {a, b}").unwrap(),
    "ExcludedForms -> {a, b}"
  );
}

// ─── PDE terms ───────────────────────────────────────────────────────────────

#[test]
fn diffusion_pde_term_basic() {
  assert_eq!(
    interpret("DiffusionPDETerm[{u, x}]").unwrap(),
    "DiffusionPDETerm[{u, x}]"
  );
}

#[test]
fn diffusion_pde_term_with_coefficient() {
  // DiffusionPDETerm with 2 args evaluates to 0 (outside solver context)
  assert_eq!(interpret("DiffusionPDETerm[{u, x}, c]").unwrap(), "0");
}

#[test]
fn diffusion_pde_term_with_params() {
  assert_eq!(
    interpret(r#"DiffusionPDETerm[{u, x}, c, "RegionSymmetry"]"#).unwrap(),
    "DiffusionPDETerm[{u, x}, c, RegionSymmetry]"
  );
}

// ─── SequenceAlignment ───────────────────────────────────────────────────────

#[test]
fn sequence_alignment_simple_substitution() {
  assert_eq!(
    interpret(r#"SequenceAlignment["abcde", "abxde"]"#).unwrap(),
    "{ab, {c, x}, de}"
  );
}

#[test]
fn sequence_alignment_identical() {
  assert_eq!(
    interpret(r#"SequenceAlignment["abc", "abc"]"#).unwrap(),
    "{abc}"
  );
}

#[test]
fn sequence_alignment_completely_different() {
  assert_eq!(
    interpret(r#"SequenceAlignment["abc", "xyz"]"#).unwrap(),
    "{{abc, xyz}}"
  );
}

#[test]
fn sequence_alignment_insertion() {
  assert_eq!(
    interpret(r#"SequenceAlignment["ac", "abc"]"#).unwrap(),
    "{a, {, b}, c}"
  );
}

#[test]
fn sequence_alignment_empty() {
  assert_eq!(
    interpret(r#"SequenceAlignment["abc", ""]"#).unwrap(),
    "{{abc, }}"
  );
}

// List inputs keep elements as sublists rather than concatenating them.
#[test]
fn sequence_alignment_list_deletion() {
  assert_eq!(
    interpret("SequenceAlignment[{1, 2, 3}, {1, 3}]").unwrap(),
    "{{1}, {{2}, {}}, {3}}"
  );
}

#[test]
fn sequence_alignment_list_substitution() {
  assert_eq!(
    interpret("SequenceAlignment[{a, b, c}, {a, x, c}]").unwrap(),
    "{{a}, {{b}, {x}}, {c}}"
  );
}

#[test]
fn sequence_alignment_list_identical() {
  assert_eq!(
    interpret("SequenceAlignment[{1, 2, 3}, {1, 2, 3}]").unwrap(),
    "{{1, 2, 3}}"
  );
}

#[test]
fn sequence_alignment_list_disjoint() {
  assert_eq!(
    interpret("SequenceAlignment[{1, 2, 3}, {4, 5, 6}]").unwrap(),
    "{{{1, 2, 3}, {4, 5, 6}}}"
  );
}

// ─── StationaryDistribution ─────────────────────────────────────────────────

#[test]
fn stationary_distribution_inert() {
  assert_eq!(
    interpret("StationaryDistribution[proc]").unwrap(),
    "StationaryDistribution[proc]"
  );
}

#[test]
fn stationary_distribution_with_markov() {
  assert_eq!(
    interpret(
      "StationaryDistribution[DiscreteMarkovProcess[{1, 0}, {{0.5, 0.5}, {0.3, 0.7}}]]"
    )
    .unwrap(),
    "StationaryDistribution[DiscreteMarkovProcess[{1, 0}, {{0.5, 0.5}, {0.3, 0.7}}]]"
  );
}

mod leaf_count {
  use super::*;

  #[test]
  fn leaf_count_sum_expr() {
    assert_eq!(interpret("LeafCount[1 + x + y^a]").unwrap(), "6");
  }

  #[test]
  fn leaf_count_function_call() {
    assert_eq!(interpret("LeafCount[f[x, y]]").unwrap(), "3");
  }

  #[test]
  fn leaf_count_list() {
    assert_eq!(interpret("LeafCount[{1, 2, 3}]").unwrap(), "4");
  }

  #[test]
  fn leaf_count_atom() {
    assert_eq!(interpret("LeafCount[42]").unwrap(), "1");
  }

  #[test]
  fn leaf_count_symbol() {
    assert_eq!(interpret("LeafCount[x]").unwrap(), "1");
  }
}

mod byte_count {
  use super::*;

  #[test]
  fn byte_count_integer() {
    // i128 = 16 bytes
    assert_eq!(interpret("ByteCount[42]").unwrap(), "16");
  }

  #[test]
  fn byte_count_real() {
    // machine real: 16 bytes (Wolfram's representation)
    assert_eq!(interpret("ByteCount[3.14]").unwrap(), "16");
  }

  #[test]
  fn byte_count_string() {
    // 32-byte header; "hello" (5 chars) fits within header
    assert_eq!(interpret("ByteCount[\"hello\"]").unwrap(), "32");
  }

  #[test]
  fn byte_count_empty_string() {
    // empty string: 32-byte header
    assert_eq!(interpret("ByteCount[\"\"]").unwrap(), "32");
  }

  #[test]
  fn byte_count_symbol_is_zero() {
    // Symbols are shared, so 0 bytes
    assert_eq!(interpret("ByteCount[x]").unwrap(), "0");
  }

  #[test]
  fn byte_count_list() {
    // {1, 2, 3}: 40 base + 3*8 slots + 3*16 integers = 112
    assert_eq!(interpret("ByteCount[{1, 2, 3}]").unwrap(), "112");
  }

  #[test]
  fn byte_count_nested_list() {
    // {{1,2},{3,4}}: 40 base + 2*8 slots + 2*(40+2*8+2*16) = 232
    assert_eq!(interpret("ByteCount[{{1, 2}, {3, 4}}]").unwrap(), "232");
  }

  #[test]
  fn byte_count_function_call() {
    // f[x, y]: 40 base + 2*8 slots + 2*0 symbols = 56
    assert_eq!(interpret("ByteCount[f[x, y]]").unwrap(), "56");
  }

  #[test]
  fn byte_count_larger_is_more() {
    // A larger list should have a larger byte count
    let small = interpret("ByteCount[{1, 2}]").unwrap();
    let large = interpret("ByteCount[{1, 2, 3, 4, 5}]").unwrap();
    assert!(
      large.parse::<i128>().unwrap() > small.parse::<i128>().unwrap(),
      "Larger list should have larger byte count"
    );
  }
}

mod subsets {
  use super::*;

  #[test]
  fn all_subsets() {
    assert_eq!(
      interpret("Subsets[{a, b, c}]").unwrap(),
      "{{}, {a}, {b}, {c}, {a, b}, {a, c}, {b, c}, {a, b, c}}"
    );
  }

  #[test]
  fn max_size_integer() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d}, 2]").unwrap(),
      "{{}, {a}, {b}, {c}, {d}, {a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}}"
    );
  }

  #[test]
  fn exact_size() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d}, {2}]").unwrap(),
      "{{a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}}"
    );
  }

  #[test]
  fn exact_size_with_max_count() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d, e}, {3}, 5]").unwrap(),
      "{{a, b, c}, {a, b, d}, {a, b, e}, {a, c, d}, {a, c, e}}"
    );
  }

  #[test]
  fn size_range_with_step() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d}, {0, 4, 2}]").unwrap(),
      "{{}, {a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}, {a, b, c, d}}"
    );
  }

  #[test]
  fn all_with_max_count() {
    assert_eq!(
      interpret("Subsets[Range[5], All, 5]").unwrap(),
      "{{}, {1}, {2}, {3}, {4}}"
    );
  }

  #[test]
  fn part_spec_single() {
    assert_eq!(
      interpret("Subsets[Range[5], All, {25}]").unwrap(),
      "{{2, 4, 5}}"
    );
  }

  #[test]
  fn part_spec_range_reverse() {
    assert_eq!(
      interpret("Subsets[{a, b, c, d}, All, {15, 1, -2}]").unwrap(),
      "{{b, c, d}, {a, b, d}, {c, d}, {b, c}, {a, c}, {d}, {b}, {}}"
    );
  }
}

mod append_prepend {
  use super::*;

  #[test]
  fn append_to_list() {
    assert_eq!(interpret("Append[{1, 2, 3}, 4]").unwrap(), "{1, 2, 3, 4}");
  }

  #[test]
  fn append_to_function_call() {
    assert_eq!(interpret("Append[f[a, b], c]").unwrap(), "f[a, b, c]");
  }

  #[test]
  fn append_list_element() {
    assert_eq!(
      interpret("Append[{a, b}, {c, d}]").unwrap(),
      "{a, b, {c, d}}"
    );
  }

  #[test]
  fn prepend_to_list() {
    assert_eq!(interpret("Prepend[{1, 2, 3}, 0]").unwrap(), "{0, 1, 2, 3}");
  }

  #[test]
  fn prepend_to_function_call() {
    assert_eq!(interpret("Prepend[f[a, b], c]").unwrap(), "f[c, a, b]");
  }
}

mod drop_extended {
  use super::*;

  #[test]
  fn drop_range() {
    assert_eq!(
      interpret("Drop[{a, b, c, d, e}, {2, -2}]").unwrap(),
      "{a, e}"
    );
  }

  #[test]
  fn drop_single_index() {
    assert_eq!(
      interpret("Drop[{a, b, c, d, e}, {3}]").unwrap(),
      "{a, b, d, e}"
    );
  }

  #[test]
  fn drop_zero() {
    assert_eq!(interpret("Drop[{a, b, c, d}, 0]").unwrap(), "{a, b, c, d}");
  }

  #[test]
  fn drop_none_columns() {
    // Drop[list, None, n] drops n columns from each row
    assert_eq!(
      interpret("Drop[{{a, b, c}, {d, e, f}}, None, 1]").unwrap(),
      "{{b, c}, {e, f}}"
    );
  }

  #[test]
  fn drop_none_columns_negative() {
    assert_eq!(
      interpret("Drop[{{a, b, c}, {d, e, f}}, None, -1]").unwrap(),
      "{{a, b}, {d, e}}"
    );
  }

  #[test]
  fn drop_none_columns_single() {
    // Drop[list, None, {n}] drops the nth column
    assert_eq!(
      interpret("Drop[{{a, b, c}, {d, e, f}}, None, {2}]").unwrap(),
      "{{a, c}, {d, f}}"
    );
  }

  #[test]
  fn drop_stepped_range() {
    // Drop[list, {m, n, s}] drops elements m, m+s, m+2s, ..., up to n.
    // Drop[{1,2,3,4,5}, {1,5,2}] -> drops positions 1, 3, 5 -> {2, 4}
    assert_eq!(
      interpret("Drop[{1, 2, 3, 4, 5}, {1, 5, 2}]").unwrap(),
      "{2, 4}"
    );
  }

  #[test]
  fn drop_stepped_range_interior() {
    // Drop[{a..g}, {2,6,2}] -> drops positions 2, 4, 6 (b, d, f)
    assert_eq!(
      interpret("Drop[{a, b, c, d, e, f, g}, {2, 6, 2}]").unwrap(),
      "{a, c, e, g}"
    );
  }

  #[test]
  fn drop_stepped_range_negative_end() {
    // Drop[{a..g}, {2,-2,2}] -> drops positions 2, 4, 6 (b, d, f)
    assert_eq!(
      interpret("Drop[{a, b, c, d, e, f, g}, {2, -2, 2}]").unwrap(),
      "{a, c, e, g}"
    );
  }

  #[test]
  fn drop_stepped_range_step_three() {
    // Drop[{a..g}, {1,-1,3}] -> drops positions 1, 4, 7 (a, d, g)
    assert_eq!(
      interpret("Drop[{a, b, c, d, e, f, g}, {1, -1, 3}]").unwrap(),
      "{b, c, e, f}"
    );
  }

  #[test]
  fn drop_2d_range_rows_and_columns() {
    // Drop rows 2..3 and columns 2..3 from a 4x4 matrix, leaving the four
    // corners {{11, 14}, {41, 44}}. Matches wolframscript.
    let expr = "Drop[{{11,12,13,14},{21,22,23,24},\
                      {31,32,33,34},{41,42,43,44}}, {2, 3}, {2, 3}]";
    assert_eq!(interpret(expr).unwrap(), "{{11, 14}, {41, 44}}");
  }
}

mod partition_extended {
  use super::*;

  #[test]
  fn partition_with_stride() {
    assert_eq!(
      interpret("Partition[{a, b, c, d, e, f}, 3, 1]").unwrap(),
      "{{a, b, c}, {b, c, d}, {c, d, e}, {d, e, f}}"
    );
  }

  #[test]
  fn partition_with_stride_2() {
    assert_eq!(
      interpret("Partition[{a, b, c, d, e}, 2, 1]").unwrap(),
      "{{a, b}, {b, c}, {c, d}, {d, e}}"
    );
  }

  // Multi-dimensional Partition[tensor, {n1, n2}, d]. Regression for
  // mathics rearrange.py:1001.
  #[test]
  fn partition_matrix_2x2_stride_1() {
    assert_eq!(
      interpret(
        "Partition[{{11, 12, 13}, {21, 22, 23}, {31, 32, 33}}, {2, 2}, 1]"
      )
      .unwrap(),
      "{{{{11, 12}, {21, 22}}, {{12, 13}, {22, 23}}}, {{{21, 22}, {31, 32}}, \
       {{22, 23}, {32, 33}}}}"
    );
  }

  #[test]
  fn partition_matrix_2x2_default_stride() {
    assert_eq!(
      interpret("Partition[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {2, 2}]")
        .unwrap(),
      "{{{{1, 2}, {4, 5}}}}"
    );
  }
}

mod downsample {
  use super::*;

  #[test]
  fn by_two() {
    assert_eq!(
      interpret("Downsample[{a, b, c, d, e, f, g, h}, 2]").unwrap(),
      "{a, c, e, g}"
    );
  }

  #[test]
  fn by_three() {
    assert_eq!(
      interpret("Downsample[{a, b, c, d, e, f, g, h}, 3]").unwrap(),
      "{a, d, g}"
    );
  }

  #[test]
  fn with_offset() {
    assert_eq!(
      interpret("Downsample[{a, b, c, d, e, f, g, h}, 2, 2]").unwrap(),
      "{b, d, f, h}"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("Downsample[{1, 2, 3, 4, 5, 6}, 2]").unwrap(),
      "{1, 3, 5}"
    );
  }
}

mod square_matrix_q {
  use super::*;

  #[test]
  fn square() {
    assert_eq!(
      interpret("SquareMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn rectangular() {
    assert_eq!(
      interpret("SquareMatrixQ[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn vector() {
    assert_eq!(interpret("SquareMatrixQ[{1, 2, 3}]").unwrap(), "False");
  }

  #[test]
  fn three_by_three() {
    assert_eq!(
      interpret("SquareMatrixQ[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "True"
    );
  }
}

mod contains_any {
  use super::*;

  #[test]
  fn has_common() {
    assert_eq!(interpret("ContainsAny[{a, b, c}, {b, d}]").unwrap(), "True");
  }

  #[test]
  fn no_common() {
    assert_eq!(
      interpret("ContainsAny[{a, b, c}, {d, e}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("ContainsAny[{1, 2, 3}, {3, 4, 5}]").unwrap(),
      "True"
    );
  }
}

mod contains_none {
  use super::*;

  #[test]
  fn no_common() {
    assert_eq!(
      interpret("ContainsNone[{a, b, c}, {d, e}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn has_common() {
    assert_eq!(
      interpret("ContainsNone[{a, b, c}, {b, d}]").unwrap(),
      "False"
    );
  }
}

mod machine_number_q {
  use super::*;

  #[test]
  fn real_is_true() {
    assert_eq!(interpret("MachineNumberQ[1.5]").unwrap(), "True");
  }

  #[test]
  fn integer_is_false() {
    assert_eq!(interpret("MachineNumberQ[1]").unwrap(), "False");
  }

  #[test]
  fn rational_is_false() {
    assert_eq!(interpret("MachineNumberQ[1/3]").unwrap(), "False");
  }

  #[test]
  fn string_is_false() {
    assert_eq!(interpret("MachineNumberQ[\"hello\"]").unwrap(), "False");
  }

  #[test]
  fn machine_complex_is_true() {
    assert_eq!(interpret("MachineNumberQ[1.5 + 2.3 I]").unwrap(), "True");
  }

  #[test]
  fn integer_complex_is_false() {
    assert_eq!(interpret("MachineNumberQ[1 + 2 I]").unwrap(), "False");
  }
}

mod text_string {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("TextString[42]").unwrap(), "42");
  }

  #[test]
  fn list() {
    assert_eq!(interpret("TextString[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn string_passthrough() {
    assert_eq!(interpret("TextString[\"hello\"]").unwrap(), "hello");
  }
}

mod string_partition {
  use super::*;

  #[test]
  fn basic_partition() {
    assert_eq!(
      interpret("StringPartition[\"abcdefghij\", 3]").unwrap(),
      "{abc, def, ghi}"
    );
  }

  #[test]
  fn non_divisible_drops_remainder() {
    // Wolfram drops the trailing partial substring
    assert_eq!(
      interpret("StringPartition[\"abcde\", 2]").unwrap(),
      "{ab, cd}"
    );
  }

  #[test]
  fn with_offset_1() {
    assert_eq!(
      interpret("StringPartition[\"abcdefghij\", 3, 1]").unwrap(),
      "{abc, bcd, cde, def, efg, fgh, ghi, hij}"
    );
  }

  #[test]
  fn with_offset_2() {
    assert_eq!(
      interpret("StringPartition[\"abcdefghij\", 3, 2]").unwrap(),
      "{abc, cde, efg, ghi}"
    );
  }

  #[test]
  fn single_char_partition() {
    assert_eq!(
      interpret("StringPartition[\"abc\", 1]").unwrap(),
      "{a, b, c}"
    );
  }

  // An invalid block size or offset leaves the call unevaluated (matching
  // wolframscript) rather than raising an evaluation error.
  #[test]
  fn invalid_block_size_stays_unevaluated() {
    assert_eq!(
      interpret("StringPartition[\"abc\", 0]").unwrap(),
      "StringPartition[abc, 0]"
    );
    assert_eq!(
      interpret("StringPartition[\"abc\", -2]").unwrap(),
      "StringPartition[abc, -2]"
    );
    // A non-integer / symbolic block size stays unevaluated too.
    assert_eq!(
      interpret("StringPartition[\"abc\", x]").unwrap(),
      "StringPartition[abc, x]"
    );
    assert_eq!(
      interpret("StringPartition[\"abc\", {3, 2}]").unwrap(),
      "StringPartition[abc, {3, 2}]"
    );
  }

  #[test]
  fn invalid_offset_stays_unevaluated() {
    assert_eq!(
      interpret("StringPartition[\"abc\", 3, 0]").unwrap(),
      "StringPartition[abc, 3, 0]"
    );
    assert_eq!(
      interpret("StringPartition[\"abc\", 3, -1]").unwrap(),
      "StringPartition[abc, 3, -1]"
    );
  }
}

mod color_q {
  use super::*;

  #[test]
  fn rgb_color_is_true() {
    assert_eq!(interpret("ColorQ[RGBColor[1, 0, 0]]").unwrap(), "True");
  }

  #[test]
  fn named_color_is_true() {
    assert_eq!(interpret("ColorQ[Red]").unwrap(), "True");
  }

  #[test]
  fn integer_is_false() {
    assert_eq!(interpret("ColorQ[42]").unwrap(), "False");
  }

  #[test]
  fn string_is_false() {
    assert_eq!(interpret("ColorQ[\"hello\"]").unwrap(), "False");
  }
}

mod contains_all {
  use super::*;

  #[test]
  fn all_present() {
    assert_eq!(
      interpret("ContainsAll[{1, 2, 3, 4, 5}, {2, 4}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn not_all_present() {
    assert_eq!(
      interpret("ContainsAll[{1, 2, 3}, {2, 4}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn empty_subset() {
    assert_eq!(interpret("ContainsAll[{1, 2, 3}, {}]").unwrap(), "True");
  }

  // The Contains* family accepts a SameTest -> f option, applied as
  // f[a_elem, b_elem].
  #[test]
  fn contains_same_test() {
    assert_eq!(
      interpret("ContainsAll[{1, 2, 3}, {1, 2}, SameTest -> Equal]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("ContainsAll[{1, 2, 3}, {1, 5}, SameTest -> Equal]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("ContainsAny[{1, 2, 3}, {5, 2}, SameTest -> Equal]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("ContainsNone[{1, 2, 3}, {5, 6}, SameTest -> Equal]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("ContainsExactly[{1, 2}, {2, 1}, SameTest -> Equal]").unwrap(),
      "True"
    );
    // Asymmetric test confirms the f[a, b] argument order.
    assert_eq!(
      interpret("ContainsAll[{5}, {3}, SameTest -> Greater]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("ContainsAny[{3}, {5}, SameTest -> Greater]").unwrap(),
      "False"
    );
  }

  // Operator (curried) forms: f[list2][list1] == f[list1, list2].
  #[test]
  fn contains_operator_forms() {
    assert_eq!(
      interpret("ContainsExactly[{a, b, c}][{c, b, a}]").unwrap(),
      "True"
    );
    // ContainsAll[{a,b,c}][{a,b}] == ContainsAll[{a,b}, {a,b,c}] -> {a,b}
    // does not contain c.
    assert_eq!(
      interpret("ContainsAll[{a, b, c}][{a, b}]").unwrap(),
      "False"
    );
    assert_eq!(interpret("ContainsAny[{a, b}][{b, z}]").unwrap(), "True");
    assert_eq!(interpret("ContainsNone[{x, y}][{a, b}]").unwrap(), "True");
    // ContainsOnly[{a,b}][{a,b,c}] == ContainsOnly[{a,b,c}, {a,b}] -> c is
    // not among the allowed elements.
    assert_eq!(
      interpret("ContainsOnly[{a, b}][{a, b, c}]").unwrap(),
      "False"
    );
  }

  // Operator forms compose with Map/Select.
  #[test]
  fn contains_operator_forms_mapped() {
    assert_eq!(
      interpret("Map[ContainsAll[{1, 2, 3}], {{1, 2}, {1, 4}}]").unwrap(),
      "{False, False}"
    );
    assert_eq!(
      interpret("Select[{{1, 2}, {5, 6}}, ContainsAny[{1, 9}]]").unwrap(),
      "{{1, 2}}"
    );
  }
}

mod missing_q {
  use super::*;

  #[test]
  fn missing_is_true() {
    assert_eq!(interpret("MissingQ[Missing[]]").unwrap(), "True");
  }

  #[test]
  fn missing_with_reason_is_true() {
    assert_eq!(interpret("MissingQ[Missing[\"reason\"]]").unwrap(), "True");
  }

  #[test]
  fn integer_is_false() {
    assert_eq!(interpret("MissingQ[42]").unwrap(), "False");
  }
}

mod failure_q {
  use super::*;

  // $Failed and $Aborted are failure indicators.
  #[test]
  fn failed_and_aborted_are_true() {
    assert_eq!(interpret("FailureQ[$Failed]").unwrap(), "True");
    assert_eq!(interpret("FailureQ[$Aborted]").unwrap(), "True");
  }

  // Any Failure[...] object is a failure, including the bare head.
  #[test]
  fn failure_object_is_true() {
    assert_eq!(
      interpret("FailureQ[Failure[\"err\", <||>]]").unwrap(),
      "True"
    );
    assert_eq!(interpret("FailureQ[Failure[]]").unwrap(), "True");
  }

  // Missing[...] is explicitly NOT a failure.
  #[test]
  fn missing_is_false() {
    assert_eq!(interpret("FailureQ[Missing[]]").unwrap(), "False");
    assert_eq!(
      interpret("FailureQ[Missing[\"NotFound\"]]").unwrap(),
      "False"
    );
  }

  // Other indicators and ordinary values are not failures.
  #[test]
  fn other_values_are_false() {
    assert_eq!(interpret("FailureQ[$Canceled]").unwrap(), "False");
    assert_eq!(interpret("FailureQ[5]").unwrap(), "False");
    assert_eq!(interpret("FailureQ[\"string\"]").unwrap(), "False");
    assert_eq!(interpret("FailureQ[{1, 2, 3}]").unwrap(), "False");
  }
}

mod hilbert_matrix {
  use super::*;

  #[test]
  fn two_by_two() {
    assert_eq!(
      interpret("HilbertMatrix[2]").unwrap(),
      "{{1, 1/2}, {1/2, 1/3}}"
    );
  }

  #[test]
  fn three_by_three() {
    assert_eq!(
      interpret("HilbertMatrix[3]").unwrap(),
      "{{1, 1/2, 1/3}, {1/2, 1/3, 1/4}, {1/3, 1/4, 1/5}}"
    );
  }

  #[test]
  fn rectangular_three_by_five() {
    // HilbertMatrix[{m, n}] gives the m×n Hilbert matrix.
    assert_eq!(
      interpret("HilbertMatrix[{3, 5}]").unwrap(),
      "{{1, 1/2, 1/3, 1/4, 1/5}, {1/2, 1/3, 1/4, 1/5, 1/6}, {1/3, 1/4, 1/5, 1/6, 1/7}}"
    );
  }

  #[test]
  fn rectangular_two_by_three() {
    assert_eq!(
      interpret("HilbertMatrix[{2, 3}]").unwrap(),
      "{{1, 1/2, 1/3}, {1/2, 1/3, 1/4}}"
    );
  }

  #[test]
  fn rectangular_taller_than_wide() {
    // Rectangular form should also work for m > n.
    assert_eq!(
      interpret("HilbertMatrix[{4, 2}]").unwrap(),
      "{{1, 1/2}, {1/2, 1/3}, {1/3, 1/4}, {1/4, 1/5}}"
    );
  }
}

mod toeplitz_matrix {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("ToeplitzMatrix[{1, 2, 3}]").unwrap(),
      "{{1, 2, 3}, {2, 1, 2}, {3, 2, 1}}"
    );
  }

  #[test]
  fn two_by_two() {
    assert_eq!(
      interpret("ToeplitzMatrix[{a, b}]").unwrap(),
      "{{a, b}, {b, a}}"
    );
  }

  #[test]
  fn integer_n() {
    // ToeplitzMatrix[n] gives the n×n integer Toeplitz matrix with
    // first row/column 1..n. Entry (i, j) = |i - j| + 1.
    assert_eq!(
      interpret("ToeplitzMatrix[4]").unwrap(),
      "{{1, 2, 3, 4}, {2, 1, 2, 3}, {3, 2, 1, 2}, {4, 3, 2, 1}}"
    );
  }

  #[test]
  fn integer_one() {
    assert_eq!(interpret("ToeplitzMatrix[1]").unwrap(), "{{1}}");
  }

  #[test]
  fn integer_three() {
    assert_eq!(
      interpret("ToeplitzMatrix[3]").unwrap(),
      "{{1, 2, 3}, {2, 1, 2}, {3, 2, 1}}"
    );
  }

  // ToeplitzMatrix[col, row]: col is the first column, row the first row.
  #[test]
  fn column_and_row() {
    assert_eq!(
      interpret("ToeplitzMatrix[{1, 2, 3}, {1, 4, 5}]").unwrap(),
      "{{1, 4, 5}, {2, 1, 4}, {3, 2, 1}}"
    );
    // Different lengths give a rectangular matrix (len(col) x len(row)).
    assert_eq!(
      interpret("ToeplitzMatrix[{1, 2, 3, 4}, {1, 5}]").unwrap(),
      "{{1, 5}, {2, 1}, {3, 2}, {4, 3}}"
    );
    // Symbolic entries.
    assert_eq!(
      interpret("ToeplitzMatrix[{a, b}, {a, c, d}]").unwrap(),
      "{{a, c, d}, {b, a, c}}"
    );
  }
}

mod fourier_dct_matrix {
  use super::*;

  // FourierDCTMatrix[n] defaults to the type-2 discrete cosine transform
  // matrix. Entry (i, j) = Sqrt[1/n] Cos[Pi (2i-1)(j-1)/(2n)].
  #[test]
  fn default_type_two() {
    assert_eq!(
      interpret("FourierDCTMatrix[3]").unwrap(),
      "{{1/Sqrt[3], 1/2, 1/(2*Sqrt[3])}, {1/Sqrt[3], 0, -(1/Sqrt[3])}, \
       {1/Sqrt[3], -1/2, 1/(2*Sqrt[3])}}"
    );
  }

  #[test]
  fn default_two_by_two() {
    assert_eq!(
      interpret("FourierDCTMatrix[2]").unwrap(),
      "{{1/Sqrt[2], 1/2}, {1/Sqrt[2], -1/2}}"
    );
  }

  #[test]
  fn one_by_one() {
    assert_eq!(interpret("FourierDCTMatrix[1]").unwrap(), "{{1}}");
  }

  // Keeps radicals that stay symbolic in wolframscript, e.g. Cos[Pi/8].
  #[test]
  fn default_four_keeps_symbolic_cos() {
    assert_eq!(
      interpret("FourierDCTMatrix[4]").unwrap(),
      "{{1/2, Cos[Pi/8]/2, 1/(2*Sqrt[2]), Sin[Pi/8]/2}, \
       {1/2, Sin[Pi/8]/2, -1/2*1/Sqrt[2], -1/2*Cos[Pi/8]}, \
       {1/2, -1/2*Sin[Pi/8], -1/2*1/Sqrt[2], Cos[Pi/8]/2}, \
       {1/2, -1/2*Cos[Pi/8], 1/(2*Sqrt[2]), -1/2*Sin[Pi/8]}}"
    );
  }

  // FourierDCTMatrix[n, 1] — DCT type 1.
  #[test]
  fn type_one() {
    assert_eq!(
      interpret("FourierDCTMatrix[3, 1]").unwrap(),
      "{{1/2, 1/2, 1/2}, {1, 0, -1}, {1/2, -1/2, 1/2}}"
    );
    assert_eq!(
      interpret("FourierDCTMatrix[4, 1]").unwrap(),
      "{{1/Sqrt[6], 1/Sqrt[6], 1/Sqrt[6], 1/Sqrt[6]}, \
       {Sqrt[2/3], 1/Sqrt[6], -(1/Sqrt[6]), -Sqrt[2/3]}, \
       {Sqrt[2/3], -(1/Sqrt[6]), -(1/Sqrt[6]), Sqrt[2/3]}, \
       {1/Sqrt[6], -(1/Sqrt[6]), 1/Sqrt[6], -(1/Sqrt[6])}}"
    );
  }

  // FourierDCTMatrix[n, 2] is the same as the default.
  #[test]
  fn type_two_matches_default() {
    assert_eq!(
      interpret("FourierDCTMatrix[3, 2]").unwrap(),
      interpret("FourierDCTMatrix[3]").unwrap()
    );
  }

  // FourierDCTMatrix[n, 3] — DCT type 3.
  #[test]
  fn type_three() {
    assert_eq!(
      interpret("FourierDCTMatrix[3, 3]").unwrap(),
      "{{1/Sqrt[3], 1/Sqrt[3], 1/Sqrt[3]}, {1, 0, -1}, \
       {1/Sqrt[3], -2/Sqrt[3], 1/Sqrt[3]}}"
    );
  }

  // FourierDCTMatrix[n, 4] — DCT type 4, exact nested radicals.
  #[test]
  fn type_four() {
    assert_eq!(
      interpret("FourierDCTMatrix[3, 4]").unwrap(),
      "{{(1 + Sqrt[3])/(2*Sqrt[3]), 1/Sqrt[3], (-1 + Sqrt[3])/(2*Sqrt[3])}, \
       {1/Sqrt[3], -(1/Sqrt[3]), -(1/Sqrt[3])}, \
       {(-1 + Sqrt[3])/(2*Sqrt[3]), -(1/Sqrt[3]), (1 + Sqrt[3])/(2*Sqrt[3])}}"
    );
    assert_eq!(
      interpret("FourierDCTMatrix[2, 4]").unwrap(),
      "{{Cos[Pi/8], Sin[Pi/8]}, {Sin[Pi/8], -Cos[Pi/8]}}"
    );
  }

  // Non-integer size or out-of-range type stays unevaluated, matching
  // wolframscript.
  #[test]
  fn stays_unevaluated_for_bad_args() {
    assert_eq!(
      interpret("FourierDCTMatrix[3, 5]").unwrap(),
      "FourierDCTMatrix[3, 5]"
    );
  }
}

mod discrete_hilbert_transform {
  use super::*;

  // Even length: clean integer/half results (WL chops vanishing bins to the
  // exact integer 0, keeping the rest as machine reals).
  #[test]
  fn even_length_clean_reals() {
    assert_eq!(
      interpret("DiscreteHilbertTransform[{1, 2, 3, 4}]").unwrap(),
      "{1., -1., -1., 1.}"
    );
    assert_eq!(
      interpret("DiscreteHilbertTransform[{1, 0, 0, 0}]").unwrap(),
      "{0, 0.5, 0, -0.5}"
    );
  }

  // Vanishing transforms come back as exact integer zeros.
  #[test]
  fn vanishing_gives_integer_zero() {
    assert_eq!(interpret("DiscreteHilbertTransform[{2}]").unwrap(), "{0}");
    assert_eq!(
      interpret("DiscreteHilbertTransform[{1, -1}]").unwrap(),
      "{0, 0}"
    );
    assert_eq!(
      interpret("DiscreteHilbertTransform[{1, 1, 1, 1}]").unwrap(),
      "{0, 0, 0, 0}"
    );
  }

  // Odd length, irrational entries — rounded to sidestep last-digit float
  // drift; the rounded rationals match wolframscript exactly.
  #[test]
  fn odd_length_rounded() {
    assert_eq!(
      interpret("Round[DiscreteHilbertTransform[{1, 0, 0}], 10^-10]").unwrap(),
      "{0, 1443375673/2500000000, -1443375673/2500000000}"
    );
  }

  // Even length, irrational entries — rounded for the same reason.
  #[test]
  fn even_length_rounded() {
    assert_eq!(
      interpret("Round[DiscreteHilbertTransform[{1, 2, 3, 4, 5, 6}], 10^-10]")
        .unwrap(),
      "{1443375673/625000000, -1443375673/1250000000, -1443375673/1250000000, \
       -1443375673/1250000000, -1443375673/1250000000, 1443375673/625000000}"
    );
  }

  // Non-numeric argument stays unevaluated.
  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("DiscreteHilbertTransform[{a, b, c}]").unwrap(),
      "DiscreteHilbertTransform[{a, b, c}]"
    );
  }
}

// Regression: a residual Integer(1) left behind when radicals cancel inside a
// product (Sqrt[1/3]*Sqrt[3] = 1) must be dropped, not printed as `*1`.
mod times_radical_cancellation {
  use super::*;

  #[test]
  fn rational_coefficient_survives_cleanly() {
    assert_eq!(interpret("(-1/2)*Sqrt[1/3]*Sqrt[3]").unwrap(), "-1/2");
  }

  #[test]
  fn symbolic_factor_survives_cleanly() {
    assert_eq!(interpret("x*Sqrt[1/3]*Sqrt[3]").unwrap(), "x");
    assert_eq!(interpret("(-1/2)*Sqrt[1/3]*Sqrt[3]*y").unwrap(), "-1/2*y");
  }
}

mod mantissa_exponent {
  use super::*;

  #[test]
  fn real_value() {
    assert_eq!(
      interpret("MantissaExponent[350.12]").unwrap(),
      "{0.35012, 3}"
    );
  }

  #[test]
  fn real_value_2_5() {
    // MantissaExponent[2.5] = {0.25, 1} since 2.5 = 0.25 * 10^1.
    assert_eq!(interpret("MantissaExponent[2.5]").unwrap(), "{0.25, 1}");
  }

  #[test]
  fn integer_value() {
    assert_eq!(interpret("MantissaExponent[100]").unwrap(), "{1/10, 3}");
  }

  #[test]
  fn negative_real() {
    assert_eq!(interpret("MantissaExponent[-42.5]").unwrap(), "{-0.425, 2}");
  }

  // Regression: for a negative exponent the mantissa must be scaled by the
  // EXACT positive power (value * base^|e|) rather than divided by the inexact
  // fraction base^e, so MantissaExponent[0.0012] -> {0.12, -2}, not
  // {0.11999999999999998, -2}.
  #[test]
  fn negative_exponent_no_float_noise() {
    assert_eq!(interpret("MantissaExponent[0.0012]").unwrap(), "{0.12, -2}");
  }

  #[test]
  fn real_times_large_integer_power() {
    // Regression: `2.5 * 10^20` used to stay as an unevaluated Times node
    // because 10^20 exceeded the machine-integer threshold, leaving
    // MantissaExponent unable to see a single Real argument.
    assert_eq!(
      interpret("MantissaExponent[2.5*10^20]").unwrap(),
      "{0.25, 21}"
    );
  }
}

mod reverse_extended {
  use super::*;

  #[test]
  fn syntax_q_valid() {
    assert_eq!(interpret("SyntaxQ[\"1 + 2\"]").unwrap(), "True");
  }

  #[test]
  fn syntax_q_invalid() {
    assert_eq!(interpret("SyntaxQ[\"1 + \"]").unwrap(), "False");
  }

  #[test]
  fn interquartile_range() {
    assert_eq!(
      interpret("InterquartileRange[{1, 2, 3, 4, 5, 6, 7, 8}]").unwrap(),
      "4"
    );
  }

  #[test]
  fn interquartile_range_odd() {
    assert_eq!(
      interpret("InterquartileRange[{1, 3, 5, 7, 9}]").unwrap(),
      "5"
    );
  }

  #[test]
  fn unitary_matrix_q_identity() {
    assert_eq!(
      interpret("UnitaryMatrixQ[{{1, 0}, {0, 1}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn unitary_matrix_q_permutation() {
    assert_eq!(
      interpret("UnitaryMatrixQ[{{0, 1}, {1, 0}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn unitary_matrix_q_non_unitary() {
    assert_eq!(
      interpret("UnitaryMatrixQ[{{1, 0}, {0, 2}}]").unwrap(),
      "False"
    );
  }

  // Unitarity tests ConjugateTranspose[m].m == Id — the plain transpose
  // previously declared complex unitary matrices non-unitary. Orthogonal
  // keeps the plain transpose, so {{0, I}, {I, 0}} is unitary but NOT
  // orthogonal. All outputs verified against wolframscript 15.0.
  #[test]
  fn unitary_matrix_q_complex_conjugate_transpose() {
    assert_eq!(
      interpret("UnitaryMatrixQ[{{0, I}, {I, 0}}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "UnitaryMatrixQ[{{1/Sqrt[2], I/Sqrt[2]}, {I/Sqrt[2], 1/Sqrt[2]}}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("UnitaryMatrixQ[{{1, 1}, {0, 1}}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{0, I}, {I, 0}}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{0, -1}, {1, 0}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn reflection_matrix_axis() {
    assert_eq!(
      interpret("ReflectionMatrix[{1, 0}]").unwrap(),
      "{{-1, 0}, {0, 1}}"
    );
  }

  #[test]
  fn reflection_matrix_diagonal() {
    assert_eq!(
      interpret("ReflectionMatrix[{1, 1}]").unwrap(),
      "{{0, -1}, {-1, 0}}"
    );
  }

  // ScalingMatrix[s, v] scales by factor s along the direction of v:
  // entry[i, j] = δ_ij + (s - 1) v_i v_j / (v . v).
  #[test]
  fn scaling_matrix_axis() {
    assert_eq!(
      interpret("ScalingMatrix[2, {1, 0}]").unwrap(),
      "{{2, 0}, {0, 1}}"
    );
  }

  #[test]
  fn scaling_matrix_diagonal_direction() {
    assert_eq!(
      interpret("ScalingMatrix[2, {1, 1}]").unwrap(),
      "{{3/2, 1/2}, {1/2, 3/2}}"
    );
  }

  #[test]
  fn scaling_matrix_non_unit_direction() {
    assert_eq!(
      interpret("ScalingMatrix[2, {3, 4}]").unwrap(),
      "{{34/25, 12/25}, {12/25, 41/25}}"
    );
  }

  #[test]
  fn scaling_matrix_symbolic_factor() {
    assert_eq!(
      interpret("ScalingMatrix[s, {1, 1}]").unwrap(),
      "{{(1 + s)/2, (-1 + s)/2}, {(-1 + s)/2, (1 + s)/2}}"
    );
  }

  #[test]
  fn scaling_matrix_3d() {
    assert_eq!(
      interpret("ScalingMatrix[3, {1, 1, 0}]").unwrap(),
      "{{2, 1, 0}, {1, 2, 0}, {0, 0, 1}}"
    );
  }

  // The list form is a diagonal matrix of the per-axis scale factors.
  #[test]
  fn scaling_matrix_list_form() {
    assert_eq!(
      interpret("ScalingMatrix[{2, 3, 4}]").unwrap(),
      "{{2, 0, 0}, {0, 3, 0}, {0, 0, 4}}"
    );
  }

  #[test]
  fn reverse_function_call() {
    assert_eq!(interpret("Reverse[x[a, b, c]]").unwrap(), "x[c, b, a]");
  }

  #[test]
  fn reverse_list() {
    assert_eq!(interpret("Reverse[{1, 2, 3, 4}]").unwrap(), "{4, 3, 2, 1}");
  }
}

mod first_last_extended {
  use super::*;

  #[test]
  fn first_with_default_nonempty() {
    assert_eq!(interpret("First[{a, b, c}, default]").unwrap(), "a");
  }

  #[test]
  fn first_with_default_empty() {
    assert_eq!(interpret("First[{}, default]").unwrap(), "default");
  }

  #[test]
  fn last_with_default_nonempty() {
    assert_eq!(interpret("Last[{a, b, c}, default]").unwrap(), "c");
  }

  #[test]
  fn last_with_default_empty() {
    assert_eq!(interpret("Last[{}, default]").unwrap(), "default");
  }

  #[test]
  fn first_of_function_call() {
    assert_eq!(interpret("First[f[a, b]]").unwrap(), "a");
  }

  #[test]
  fn last_of_function_call() {
    assert_eq!(interpret("Last[f[a, b]]").unwrap(), "b");
  }

  // NumericArray / ByteArray wrap a single list payload. First / Last should
  // index into that inner list rather than into the wrapper's args tuple.
  #[test]
  fn first_of_numeric_array_is_first_element() {
    assert_eq!(interpret("First[NumericArray[{1, 2, 3}]]").unwrap(), "1");
  }

  #[test]
  fn last_of_numeric_array_is_last_element() {
    assert_eq!(interpret("Last[NumericArray[{1, 2, 3}]]").unwrap(), "3");
  }

  // Regression (mathics test_binary.py:430-437): for a 2-D NumericArray the
  // wrapper itself, and First/Last/ToString of it, render the abbreviated
  // `NumericArray[<dim,...>, type]` form rather than spelling out the
  // payload list — matching wolframscript exactly. The mathics-side
  // expectation (`<Integer64, 2×2>`) reflects mathics's own default type
  // inference (Integer64) and short-form display, not wolframscript output.
  #[test]
  fn numeric_array_2d_display() {
    assert_eq!(
      interpret("NumericArray[{{1,2},{3,4}}]").unwrap(),
      "NumericArray[<2,2>, UnsignedInteger8]"
    );
  }

  #[test]
  fn tostring_numeric_array_2d() {
    assert_eq!(
      interpret("ToString[NumericArray[{{1,2},{3,4}}]]").unwrap(),
      "NumericArray[<2,2>, UnsignedInteger8]"
    );
  }

  #[test]
  fn first_of_numeric_array_2d_returns_row_array() {
    assert_eq!(
      interpret("First[NumericArray[{{1,2}, {3,4}}]]").unwrap(),
      "NumericArray[<2>, UnsignedInteger8]"
    );
  }

  #[test]
  fn last_of_numeric_array_2d_returns_row_array() {
    assert_eq!(
      interpret("Last[NumericArray[{{1,2}, {3,4}}]]").unwrap(),
      "NumericArray[<2>, UnsignedInteger8]"
    );
  }
}

mod array_predicates {
  use super::*;

  #[test]
  fn array_q_true() {
    assert_eq!(interpret("ArrayQ[{{1, 2}, {3, 4}}]").unwrap(), "True");
  }

  #[test]
  fn array_q_false() {
    assert_eq!(interpret("ArrayQ[{{1, 2}, {3}}]").unwrap(), "False");
  }

  #[test]
  fn array_q_with_depth() {
    assert_eq!(interpret("ArrayQ[{{1, 2}, {3, 4}}, 2]").unwrap(), "True");
  }

  #[test]
  fn array_q_with_wrong_depth() {
    assert_eq!(interpret("ArrayQ[{{1, 2}, {3, 4}}, 1]").unwrap(), "False");
  }

  #[test]
  fn array_q_vector_depth_1() {
    assert_eq!(interpret("ArrayQ[{1, 2, 3}, 1]").unwrap(), "True");
  }

  #[test]
  fn matrix_q_true() {
    assert_eq!(interpret("MatrixQ[{{1, 2}, {3, 4}}]").unwrap(), "True");
  }

  #[test]
  fn matrix_q_false() {
    assert_eq!(interpret("MatrixQ[{1, 2, 3}]").unwrap(), "False");
  }

  #[test]
  fn vector_q_true() {
    assert_eq!(interpret("VectorQ[{1, 2, 3}]").unwrap(), "True");
  }

  #[test]
  fn vector_q_false() {
    assert_eq!(interpret("VectorQ[{{1}, {2}}]").unwrap(), "False");
  }

  // Two-argument form VectorQ[expr, test]

  #[test]
  fn vector_q_test_number_all_pass() {
    assert_eq!(interpret("VectorQ[{1, 2, 3}, NumberQ]").unwrap(), "True");
  }

  #[test]
  fn vector_q_test_number_one_fails() {
    assert_eq!(interpret("VectorQ[{1, 2, x}, NumberQ]").unwrap(), "False");
  }

  #[test]
  fn vector_q_test_listq_allows_nested() {
    // With an explicit test, list elements are allowed if they pass it.
    assert_eq!(interpret("VectorQ[{{1}, {2, 3}}, ListQ]").unwrap(), "True");
  }

  #[test]
  fn vector_q_test_positive() {
    assert_eq!(interpret("VectorQ[{1, 2, 3}, Positive]").unwrap(), "True");
    assert_eq!(interpret("VectorQ[{1, -2, 3}, Positive]").unwrap(), "False");
  }

  #[test]
  fn vector_q_test_pure_function() {
    assert_eq!(interpret("VectorQ[{1, 2, 3}, #>0&]").unwrap(), "True");
  }

  #[test]
  fn vector_q_test_empty_is_true() {
    // Vacuously true for an empty list.
    assert_eq!(interpret("VectorQ[{}, NumberQ]").unwrap(), "True");
  }

  #[test]
  fn vector_q_test_non_list_is_false() {
    assert_eq!(interpret("VectorQ[5, NumberQ]").unwrap(), "False");
  }
}

mod dimensions_extended {
  use super::*;

  #[test]
  fn function_call_head() {
    assert_eq!(interpret("Dimensions[f[f[a, b, c]]]").unwrap(), "{1, 3}");
  }

  #[test]
  fn mixed_function_and_list_children() {
    // Descent requires all children share the parent head, so a function
    // call with List children should only report the outer dimension.
    assert_eq!(interpret("Dimensions[f[{1, 2}, {3, 4}]]").unwrap(), "{2}");
  }

  #[test]
  fn different_function_head_children() {
    assert_eq!(interpret("Dimensions[f[g[1, 2], g[3, 4]]]").unwrap(), "{2}");
  }

  #[test]
  fn max_level_one_on_flat_list() {
    assert_eq!(interpret("Dimensions[{1, 2, 3}, 1]").unwrap(), "{3}");
  }

  #[test]
  fn max_level_one_on_matrix() {
    assert_eq!(
      interpret("Dimensions[{{1, 2, 3}, {4, 5, 6}}, 1]").unwrap(),
      "{2}"
    );
  }

  #[test]
  fn max_level_two_on_three_deep() {
    assert_eq!(
      interpret("Dimensions[{{{1, 2, 3}, {4, 5, 6}}}, 2]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn max_level_three_on_three_deep() {
    assert_eq!(
      interpret("Dimensions[{{{1, 2, 3}, {4, 5, 6}}}, 3]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn max_level_exceeds_actual_depth() {
    assert_eq!(
      interpret("Dimensions[{{{1, 2, 3}, {4, 5, 6}}}, 10]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn max_level_zero_returns_empty() {
    assert_eq!(
      interpret("Dimensions[{{{1, 2, 3}, {4, 5, 6}}}, 0]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn max_level_on_atom_returns_empty() {
    assert_eq!(interpret("Dimensions[5, 1]").unwrap(), "{}");
  }

  #[test]
  fn ragged_matrix_only_top_dim() {
    assert_eq!(
      interpret("Dimensions[{{1, 2}, {3, 4, 5}}, 2]").unwrap(),
      "{2}"
    );
  }
}

mod transpose_extended {
  use super::*;

  #[test]
  fn one_d_list() {
    assert_eq!(interpret("Transpose[{a, b, c}]").unwrap(), "{a, b, c}");
  }

  #[test]
  fn matrix_with_permutation_swap() {
    // Transpose[m, {2, 1}] is the same as the default matrix transpose.
    assert_eq!(
      interpret("Transpose[{{1, 2, 3}, {4, 5, 6}}, {2, 1}]").unwrap(),
      "{{1, 4}, {2, 5}, {3, 6}}"
    );
  }

  #[test]
  fn matrix_with_permutation_identity() {
    assert_eq!(
      interpret("Transpose[{{1, 2, 3}, {4, 5, 6}}, {1, 2}]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6}}"
    );
  }

  #[test]
  fn rank_three_tensor_permutations() {
    // Transpose[a, {2, 1, 3}] swaps the first two levels.
    assert_eq!(
      interpret("Transpose[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {2, 1, 3}]")
        .unwrap(),
      "{{{1, 2}, {5, 6}}, {{3, 4}, {7, 8}}}"
    );
    // Transpose[a, {3, 1, 2}] cycles: original axis 1 -> result axis 3.
    assert_eq!(
      interpret("Transpose[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {3, 1, 2}]")
        .unwrap(),
      "{{{1, 5}, {2, 6}}, {{3, 7}, {4, 8}}}"
    );
  }

  #[test]
  fn empty_list() {
    // Transpose[{}] returns {} — wolframscript also keeps the empty
    // tensor as-is rather than emitting an error.
    assert_eq!(interpret("Transpose[{}]").unwrap(), "{}");
  }

  #[test]
  fn ragged_list_stays_unevaluated() {
    // A ragged list (rows of differing length) cannot be transposed:
    // wolframscript emits Transpose::nmtx and keeps the expression.
    assert_eq!(
      interpret("Transpose[{{1, 2}, {3}}]").unwrap(),
      "Transpose[{{1, 2}, {3}}]"
    );
  }

  #[test]
  fn double_transpose_is_identity() {
    // The classic involution: Transpose[Transpose[m]] == m for any
    // rectangular matrix or higher-rank tensor.
    assert_eq!(
      interpret(
        "matrix = {{1, 2}, {3, 4}, {5, 6}}; Transpose[Transpose[matrix]] == matrix"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "tensor = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}; Transpose[Transpose[tensor]] == tensor"
      )
      .unwrap(),
      "True"
    );
  }
}

mod product_extended {
  use super::*;

  #[test]
  fn symbolic_body() {
    assert_eq!(
      interpret("Product[f[i], {i, 1, 7}]").unwrap(),
      "f[1]*f[2]*f[3]*f[4]*f[5]*f[6]*f[7]"
    );
  }

  #[test]
  fn with_step() {
    // Product[k, {k, 1, 6, 2}] = 1 * 3 * 5 = 15
    assert_eq!(interpret("Product[k, {k, 1, 6, 2}]").unwrap(), "15");
  }

  // Multi-index Product nests like multi-index Sum: the rightmost iterator is
  // innermost, so Product[expr, {i,...}, {j,...}] = Product[Product[expr,
  // {j,...}], {i,...}].
  #[test]
  fn multi_index_independent_bounds() {
    // inner Product[i, {j,1,2}] = i^2; (4!)^2 = 576
    assert_eq!(
      interpret("Product[i, {i, 1, 4}, {j, 1, 2}]").unwrap(),
      "576"
    );
    assert_eq!(
      interpret("Product[i + j, {i, 1, 2}, {j, 1, 2}]").unwrap(),
      "72"
    );
    assert_eq!(
      interpret("Product[i*j, {i, 1, 3}, {j, 1, 2}]").unwrap(),
      "288"
    );
  }

  // The inner iterator's bound may depend on the outer variable.
  #[test]
  fn multi_index_dependent_bound() {
    // Product[i, {j,1,i}] = i^i; 1^1 * 2^2 * 3^3 = 108
    assert_eq!(
      interpret("Product[i, {i, 1, 3}, {j, 1, i}]").unwrap(),
      "108"
    );
  }

  // Product[k + a, {k, 1, n}] = Pochhammer[1 + a, n]. wolframscript prints the
  // Gamma ratio Gamma[1+a+n]/Gamma[1+a] for a numeric shift and keeps
  // Pochhammer for a symbolic shift.
  #[test]
  fn linear_shift_integer() {
    assert_eq!(
      interpret("Product[k + 1, {k, 1, n}]").unwrap(),
      "Gamma[2 + n]"
    );
    assert_eq!(
      interpret("Product[k + 2, {k, 1, n}]").unwrap(),
      "Gamma[3 + n]/2"
    );
    assert_eq!(
      interpret("Product[k + 3, {k, 1, n}]").unwrap(),
      "Gamma[4 + n]/6"
    );
    // Order of the summands does not matter.
    assert_eq!(
      interpret("Product[1 + k, {k, 1, n}]").unwrap(),
      "Gamma[2 + n]"
    );
  }

  #[test]
  fn linear_shift_rational() {
    assert_eq!(
      interpret("Product[k + 1/2, {k, 1, n}]").unwrap(),
      "(2*Gamma[3/2 + n])/Sqrt[Pi]"
    );
    assert_eq!(
      interpret("Product[k + 5/2, {k, 1, n}]").unwrap(),
      "(8*Gamma[7/2 + n])/(15*Sqrt[Pi])"
    );
  }

  #[test]
  fn linear_shift_symbolic() {
    assert_eq!(
      interpret("Product[k + a, {k, 1, n}]").unwrap(),
      "Pochhammer[1 + a, n]"
    );
  }

  // The plain Product[k] factorial form is unchanged.
  #[test]
  fn bare_var_factorial() {
    assert_eq!(interpret("Product[k, {k, 1, n}]").unwrap(), "n!");
  }

  // Rational telescoping over a body whose numerator/denominator factor into
  // several monic linear factors: Product[1 - 1/k^2, {k, 2, n}] = (1+n)/(2n).
  // Previously only linear/linear ratios telescoped; the quadratic
  // (k-1)(k+1)/k^2 was left unevaluated.
  #[test]
  fn quadratic_rational_telescopes() {
    assert_eq!(
      interpret("Product[1 - 1/k^2, {k, 2, n}]").unwrap(),
      "(1 + n)/(2*n)"
    );
    assert_eq!(
      interpret("Product[(k - 1)*(k + 1)/k^2, {k, 2, n}]").unwrap(),
      "(1 + n)/(2*n)"
    );
    assert_eq!(
      interpret("Product[(k - 2)*(k + 2)/k^2, {k, 3, n}]").unwrap(),
      "((1 + n)*(2 + n))/(6*(-1 + n)*n)"
    );
    // The finite numeric case is consistent with the closed form at n = 10.
    assert_eq!(
      interpret("Product[1 - 1/k^2, {k, 2, 10}]").unwrap(),
      "11/20"
    );
  }

  // The single linear/linear ratio still telescopes as before.
  #[test]
  fn linear_ratio_telescopes() {
    assert_eq!(
      interpret("Product[(k - 1)/k, {k, 2, n}]").unwrap(),
      "n^(-1)"
    );
    assert_eq!(interpret("Product[(k + 1)/k, {k, 1, n}]").unwrap(), "1 + n");
    assert_eq!(
      interpret("Product[(k + 2)/(k + 1), {k, 1, n}]").unwrap(),
      "(2 + n)/2"
    );
  }

  // Product[c^i, {i, 1, n}] = c^(n*(1+n)/2) for a finite upper limit.
  #[test]
  fn power_body_finite() {
    assert_eq!(interpret("Product[r^n, {n, 1, 10}]").unwrap(), "r^55");
    assert_eq!(
      interpret("Product[c^i, {i, 1, m}]").unwrap(),
      "c^((m*(1 + m))/2)"
    );
    assert_eq!(interpret("Product[2^n, {n, 1, 5}]").unwrap(), "32768");
  }

  // An infinite upper limit must not plug Infinity into the finite closed
  // form (which produced garbage like r^((Infinity*(1 + Infinity))/2));
  // wolframscript leaves these products unevaluated.
  #[test]
  fn power_body_infinite_unevaluated() {
    assert_eq!(
      interpret("Product[r^n, {n, 1, Infinity}]").unwrap(),
      "Product[r^n, {n, 1, Infinity}]"
    );
    assert_eq!(
      interpret("Product[2^n, {n, 1, Infinity}]").unwrap(),
      "Product[2^n, {n, 1, Infinity}]"
    );
    assert_eq!(
      interpret("Product[n^2, {n, 1, Infinity}]").unwrap(),
      "Product[n^2, {n, 1, Infinity}]"
    );
  }
}

mod real_sign {
  use super::*;

  #[test]
  fn negative_real() {
    assert_eq!(interpret("RealSign[-3.]").unwrap(), "-1");
  }

  #[test]
  fn positive_integer() {
    assert_eq!(interpret("RealSign[5]").unwrap(), "1");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("RealSign[0]").unwrap(), "0");
  }

  #[test]
  fn negative_integer() {
    assert_eq!(interpret("RealSign[-7]").unwrap(), "-1");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("RealSign[3/4]").unwrap(), "1");
  }

  #[test]
  fn complex_stays_symbolic() {
    assert_eq!(
      interpret("RealSign[2. + 3. I]").unwrap(),
      "RealSign[2. + 3.*I]"
    );
  }

  #[test]
  fn symbolic_stays() {
    assert_eq!(interpret("RealSign[x]").unwrap(), "RealSign[x]");
  }

  #[test]
  fn listable() {
    assert_eq!(interpret("RealSign[{-3, 0, 5}]").unwrap(), "{-1, 0, 1}");
  }

  #[test]
  fn symbolic_numeric_decided_by_sign() {
    // Real-valued constants and expressions are decided by their value.
    assert_eq!(interpret("RealSign[-Pi]").unwrap(), "-1");
    assert_eq!(interpret("RealSign[Pi - 3]").unwrap(), "1");
    assert_eq!(interpret("RealSign[Sqrt[2] - 3]").unwrap(), "-1");
  }
}

mod real_abs {
  use super::*;

  #[test]
  fn integers_and_reals() {
    assert_eq!(interpret("RealAbs[-5]").unwrap(), "5");
    assert_eq!(interpret("RealAbs[3.5]").unwrap(), "3.5");
    assert_eq!(interpret("RealAbs[-3.5]").unwrap(), "3.5");
    assert_eq!(interpret("RealAbs[0]").unwrap(), "0");
  }

  #[test]
  fn exact_symbolic_kept_exact() {
    // Negative values are negated exactly; non-negative are returned as-is.
    assert_eq!(interpret("RealAbs[-Pi]").unwrap(), "Pi");
    assert_eq!(interpret("RealAbs[Pi]").unwrap(), "Pi");
    assert_eq!(interpret("RealAbs[-2/3]").unwrap(), "2/3");
    assert_eq!(interpret("RealAbs[Sqrt[2] - 3]").unwrap(), "3 - Sqrt[2]");
    assert_eq!(interpret("RealAbs[1/2 - Pi]").unwrap(), "-1/2 + Pi");
  }

  #[test]
  fn complex_and_symbolic_stay_unevaluated() {
    assert_eq!(interpret("RealAbs[I]").unwrap(), "RealAbs[I]");
    assert_eq!(interpret("RealAbs[x]").unwrap(), "RealAbs[x]");
  }

  #[test]
  fn listable() {
    assert_eq!(
      interpret("RealAbs[{-Pi, 2/3, -5}]").unwrap(),
      "{Pi, 2/3, 5}"
    );
  }
}

mod reverse_sort {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("ReverseSort[{c, b, d, a}]").unwrap(),
      "{d, c, b, a}"
    );
  }

  #[test]
  fn with_less() {
    assert_eq!(
      interpret("ReverseSort[{1, 2, 0, 3}, Less]").unwrap(),
      "{3, 2, 1, 0}"
    );
  }

  #[test]
  fn with_greater() {
    assert_eq!(
      interpret("ReverseSort[{1, 2, 0, 3}, Greater]").unwrap(),
      "{0, 1, 2, 3}"
    );
  }

  // On an association, sorts the pairs descending by value.
  #[test]
  fn association() {
    assert_eq!(
      interpret("ReverseSort[<|a -> 1, b -> 3, c -> 2|>]").unwrap(),
      "<|b -> 3, c -> 2, a -> 1|>"
    );
  }

  #[test]
  fn association_keeps_keys() {
    assert_eq!(
      interpret("ReverseSort[<|x -> 10, y -> 5, z -> 20|>]").unwrap(),
      "<|z -> 20, x -> 10, y -> 5|>"
    );
  }
}

mod sort_with_comparator {
  use super::*;

  #[test]
  fn greater() {
    assert_eq!(
      interpret("Sort[{1, 2, 0, 3}, Greater]").unwrap(),
      "{3, 2, 1, 0}"
    );
  }

  #[test]
  fn less() {
    assert_eq!(interpret("Sort[{3, 1, 2}, Less]").unwrap(), "{1, 2, 3}");
  }

  // With Less/Greater on data they can't compare (non-numeric symbols, where
  // `c < a` stays symbolic), the elements are incomparable, so the original
  // order is kept — not the canonical sort.
  #[test]
  fn symbolic_data_keeps_order() {
    assert_eq!(interpret("Sort[{c, a, b}, Less]").unwrap(), "{c, a, b}");
    assert_eq!(interpret("Sort[{c, a, b}, Greater]").unwrap(), "{c, a, b}");
    // The pure-function comparator behaves the same way.
    assert_eq!(
      interpret("Sort[{c, a, b}, #1 < #2 &]").unwrap(),
      "{c, a, b}"
    );
  }

  // Sort[assoc, p] orders entries by value; a symbolic value with Less keeps
  // the original order.
  #[test]
  fn association_symbolic_values_keep_order() {
    assert_eq!(
      interpret("Sort[<|a -> x, b -> y|>, Less]").unwrap(),
      "<|a -> x, b -> y|>"
    );
  }
}

mod entity {
  use super::*;

  #[test]
  fn entity_strips_string_quotes_in_output() {
    // Entity strips string quotes in output (matching wolframscript)
    assert_eq!(
      interpret("Entity[\"Country\", \"France\"]").unwrap(),
      "Entity[Country, France]"
    );
  }

  #[test]
  fn entity_single_arg() {
    assert_eq!(interpret("Entity[\"Country\"]").unwrap(), "Entity[Country]");
  }

  #[test]
  fn entity_head() {
    assert_eq!(
      interpret("Head[Entity[\"Country\", \"France\"]]").unwrap(),
      "Entity"
    );
  }

  #[test]
  fn entity_attributes() {
    assert_eq!(
      interpret("Attributes[Entity]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  #[test]
  fn entity_stays_symbolic() {
    // Entity expressions are inert - they evaluate to themselves
    assert_eq!(
      interpret("Entity[\"City\", \"Paris\"]").unwrap(),
      "Entity[City, Paris]"
    );
  }

  #[test]
  fn entity_mixed_args() {
    // Entity with mixed arg types preserves strings but evaluates others
    assert_eq!(
      interpret("Entity[\"Planet\", \"Mars\"]").unwrap(),
      "Entity[Planet, Mars]"
    );
  }
}

mod reals {
  use super::*;

  #[test]
  fn reals_evaluates_to_itself() {
    assert_eq!(interpret("Reals").unwrap(), "Reals");
  }

  #[test]
  fn reals_head() {
    assert_eq!(interpret("Head[Reals]").unwrap(), "Symbol");
  }

  #[test]
  fn reals_attributes() {
    assert_eq!(
      interpret("Attributes[Reals]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  #[test]
  fn element_integer_in_reals() {
    assert_eq!(interpret("Element[5, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_rational_in_reals() {
    assert_eq!(interpret("Element[3/7, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_real_in_reals() {
    assert_eq!(interpret("Element[2.5, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_pi_in_reals() {
    assert_eq!(interpret("Element[Pi, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_complex_not_in_reals() {
    assert_eq!(interpret("Element[2 + 3 I, Reals]").unwrap(), "False");
  }

  #[test]
  fn element_sum_drops_real_constant() {
    // wolframscript: Element[x + 5, Reals] → Element[x, Reals].
    assert_eq!(
      interpret("Element[x + 5, Reals]").unwrap(),
      "Element[x, Reals]"
    );
  }

  #[test]
  fn element_sum_drops_multiple_real_constants() {
    // Integer constants combine first, then drop out of Element[..., Reals].
    assert_eq!(
      interpret("Element[x + 2 + 3, Reals]").unwrap(),
      "Element[x, Reals]"
    );
  }

  #[test]
  fn element_sum_pure_constants() {
    // No symbolic terms: the sum is fully real → True.
    assert_eq!(interpret("Element[5 + 3, Reals]").unwrap(), "True");
  }

  #[test]
  fn element_sum_drops_integer_for_integers_domain() {
    assert_eq!(
      interpret("Element[x + 1, Integers]").unwrap(),
      "Element[x, Integers]"
    );
  }

  #[test]
  fn element_sum_unknown_term_stays_symbolic() {
    // x and 5*I both have unknown / non-real status — no simplification.
    let result = interpret("Element[x + 5*I, Reals]").unwrap();
    assert!(
      result.contains("Element"),
      "expected unevaluated, got {}",
      result
    );
  }
}

mod algebraics_and_rationals {
  use super::*;

  #[test]
  fn transcendental_constants_not_algebraic() {
    assert_eq!(interpret("Element[Pi, Algebraics]").unwrap(), "False");
    assert_eq!(interpret("Element[E, Algebraics]").unwrap(), "False");
    assert_eq!(interpret("Element[Degree, Algebraics]").unwrap(), "False");
  }

  #[test]
  fn algebraic_constants_and_imaginary_unit() {
    assert_eq!(
      interpret("Element[GoldenRatio, Algebraics]").unwrap(),
      "True"
    );
    assert_eq!(interpret("Element[I, Algebraics]").unwrap(), "True");
  }

  #[test]
  fn radicals_are_algebraic() {
    assert_eq!(interpret("Element[Sqrt[2], Algebraics]").unwrap(), "True");
    assert_eq!(interpret("Element[2^(1/3), Algebraics]").unwrap(), "True");
    assert_eq!(interpret("Element[Sqrt[2/3], Algebraics]").unwrap(), "True");
  }

  #[test]
  fn radicals_are_not_rational() {
    assert_eq!(interpret("Element[Sqrt[2], Rationals]").unwrap(), "False");
    assert_eq!(interpret("Element[2^(1/3), Rationals]").unwrap(), "False");
  }

  #[test]
  fn irrational_constants_not_rational() {
    assert_eq!(
      interpret("Element[GoldenRatio, Rationals]").unwrap(),
      "False"
    );
    assert_eq!(interpret("Element[I, Rationals]").unwrap(), "False");
  }

  #[test]
  fn open_constants_stay_unevaluated() {
    // The irrationality/transcendence of these is an open problem.
    assert_eq!(
      interpret("Element[Catalan, Algebraics]").unwrap(),
      "Element[Catalan, Algebraics]"
    );
    assert_eq!(
      interpret("Element[EulerGamma, Rationals]").unwrap(),
      "Element[EulerGamma, Rationals]"
    );
  }

  #[test]
  fn rationals_and_integers_are_algebraic() {
    assert_eq!(interpret("Element[2, Algebraics]").unwrap(), "True");
    assert_eq!(interpret("Element[1/2, Algebraics]").unwrap(), "True");
  }
}

mod plus_minus {
  use super::*;

  #[test]
  fn unary() {
    assert_eq!(interpret("PlusMinus[3]").unwrap(), "\u{00B1}3");
  }

  #[test]
  fn binary() {
    assert_eq!(interpret("PlusMinus[a, b]").unwrap(), "a \u{00B1} b");
  }
}

mod circle_times {
  use super::*;

  #[test]
  fn binary() {
    assert_eq!(interpret("CircleTimes[a, b]").unwrap(), "a \u{2297} b");
  }

  #[test]
  fn ternary() {
    assert_eq!(
      interpret("CircleTimes[a, b, c]").unwrap(),
      "a \u{2297} b \u{2297} c"
    );
  }

  // Tilde displays as the ∼ (U+223C) infix operator, like its siblings.
  #[test]
  fn tilde_operator() {
    assert_eq!(interpret("Tilde[a, b]").unwrap(), "a \u{223C} b");
    assert_eq!(
      interpret("Tilde[a, b, c]").unwrap(),
      "a \u{223C} b \u{223C} c"
    );
    // InputForm renders the same infix operator (not the function call).
    assert_eq!(
      interpret("ToString[Tilde[a, b], InputForm]").unwrap(),
      "a \u{223C} b"
    );
  }
}

mod circle_dot {
  use super::*;

  #[test]
  fn binary() {
    assert_eq!(interpret("CircleDot[a, b]").unwrap(), "a \u{2299} b");
  }

  #[test]
  fn ternary() {
    assert_eq!(
      interpret("CircleDot[a, b, c]").unwrap(),
      "a \u{2299} b \u{2299} c"
    );
  }

  #[test]
  fn numbers() {
    assert_eq!(interpret("CircleDot[1, 2]").unwrap(), "1 \u{2299} 2");
  }

  // Single and zero argument forms stay in function notation,
  // matching wolframscript.
  #[test]
  fn single_arg_unevaluated() {
    assert_eq!(interpret("CircleDot[a]").unwrap(), "CircleDot[a]");
  }

  #[test]
  fn zero_args_unevaluated() {
    assert_eq!(interpret("CircleDot[]").unwrap(), "CircleDot[]");
  }

  // A nested CircleDot argument is parenthesized:
  // CircleDot[a, CircleDot[b, c]] -> a ⊙ (b ⊙ c)
  #[test]
  fn nested_parenthesized() {
    assert_eq!(
      interpret("CircleDot[a, CircleDot[b, c]]").unwrap(),
      "a \u{2299} (b \u{2299} c)"
    );
  }
}

mod ring_operators {
  use super::*;

  // \[CirclePlus], \[CircleTimes] and \[CenterDot] are flat infix operators.
  #[test]
  fn circle_plus_operator() {
    assert_eq!(interpret(r#"a \[CirclePlus] b"#).unwrap(), "a \u{2295} b");
    // Flat: a ⊕ b ⊕ c collapses to CirclePlus[a, b, c].
    assert_eq!(
      interpret(r#"Length[a \[CirclePlus] b \[CirclePlus] c]"#).unwrap(),
      "3"
    );
    // Binds tighter than Plus, looser than Times.
    assert_eq!(interpret(r#"Head[a + b \[CirclePlus] c]"#).unwrap(), "Plus");
  }

  #[test]
  fn circle_times_operator() {
    assert_eq!(interpret(r#"a \[CircleTimes] b"#).unwrap(), "a \u{2297} b");
    // Binds tighter than Times.
    assert_eq!(
      interpret(r#"Head[a \[CircleTimes] b * c]"#).unwrap(),
      "Times"
    );
  }

  #[test]
  fn center_dot_operator() {
    assert_eq!(interpret(r#"a \[CenterDot] b"#).unwrap(), "a \u{00B7} b");
    assert_eq!(
      interpret(r#"a \[CenterDot] b \[CenterDot] c"#).unwrap(),
      "a \u{00B7} b \u{00B7} c"
    );
    // CircleTimes binds tighter than CenterDot.
    assert_eq!(
      interpret(r#"a \[CenterDot] b \[CircleTimes] c"#).unwrap(),
      "a \u{00B7} b \u{2297} c"
    );
  }

  // \[CircleMinus] is a binary (non-flat), left-associative operator at the
  // same precedence as \[CirclePlus].
  #[test]
  fn circle_minus_operator() {
    assert_eq!(interpret(r#"a \[CircleMinus] b"#).unwrap(), "a \u{2296} b");
    assert_eq!(
      interpret(r#"Head[a \[CircleMinus] b]"#).unwrap(),
      "CircleMinus"
    );
    // Binary, not flat: a ⊖ b ⊖ c is CircleMinus[CircleMinus[a, b], c].
    assert_eq!(
      interpret(r#"Length[a \[CircleMinus] b \[CircleMinus] c]"#).unwrap(),
      "2"
    );
    assert_eq!(
      interpret(r#"a \[CircleMinus] b \[CircleMinus] c"#).unwrap(),
      "a \u{2296} b \u{2296} c"
    );
    // Binds tighter than Plus, looser than Times.
    assert_eq!(
      interpret(r#"Head[a + b \[CircleMinus] c]"#).unwrap(),
      "Plus"
    );
    assert_eq!(
      interpret(r#"Head[a \[CircleMinus] b * c]"#).unwrap(),
      "CircleMinus"
    );
  }

  // \[Wedge] and \[Vee] are flat infix operators with Wedge tighter than Vee.
  #[test]
  fn wedge_vee_operators() {
    assert_eq!(interpret(r#"a \[Wedge] b"#).unwrap(), "a \u{22C0} b");
    assert_eq!(interpret(r#"a \[Vee] b"#).unwrap(), "a \u{22C1} b");
    // Flat: a ⋀ b ⋀ c ⋀ d collapses to Wedge[a, b, c, d].
    assert_eq!(
      interpret(r#"Length[a \[Wedge] b \[Wedge] c \[Wedge] d]"#).unwrap(),
      "4"
    );
    // Wedge binds tighter than Vee.
    assert_eq!(interpret(r#"Head[a \[Wedge] b \[Vee] c]"#).unwrap(), "Vee");
    // Both bind tighter than CircleTimes; Dot binds tighter than Wedge.
    assert_eq!(
      interpret(r#"Head[a \[Wedge] b \[CircleTimes] c]"#).unwrap(),
      "CircleTimes"
    );
    assert_eq!(interpret(r#"Head[a \[Wedge] b . c]"#).unwrap(), "Wedge");
  }

  // \[Star] is a flat infix operator between CirclePlus and Times.
  #[test]
  fn star_operator() {
    assert_eq!(interpret(r#"a \[Star] b"#).unwrap(), "a \u{22C6} b");
    assert_eq!(
      interpret("Star[a, b, c]").unwrap(),
      "a \u{22C6} b \u{22C6} c"
    );
    // Flat.
    assert_eq!(interpret(r#"Length[a \[Star] b \[Star] c]"#).unwrap(), "3");
    // Binds looser than Times, tighter than Plus and CirclePlus.
    assert_eq!(interpret(r#"Head[a \[Star] b * c]"#).unwrap(), "Star");
    assert_eq!(interpret(r#"Head[a \[Star] b + c]"#).unwrap(), "Plus");
    assert_eq!(
      interpret(r#"Head[a \[Star] b \[CirclePlus] c]"#).unwrap(),
      "CirclePlus"
    );
  }

  // \[SmallCircle] is a flat infix operator that binds tighter than Dot but
  // looser than Power and Map.
  #[test]
  fn small_circle_operator() {
    assert_eq!(interpret(r#"a \[SmallCircle] b"#).unwrap(), "a \u{2218} b");
    assert_eq!(
      interpret("SmallCircle[a, b, c]").unwrap(),
      "a \u{2218} b \u{2218} c"
    );
    // Flat.
    assert_eq!(
      interpret(r#"Length[a \[SmallCircle] b \[SmallCircle] c]"#).unwrap(),
      "3"
    );
    // Binds tighter than Dot, Times and Wedge.
    assert_eq!(interpret(r#"Head[a \[SmallCircle] b . c]"#).unwrap(), "Dot");
    assert_eq!(
      interpret(r#"Head[a \[SmallCircle] b * c]"#).unwrap(),
      "Times"
    );
    // But looser than Power and Map.
    assert_eq!(
      interpret(r#"Head[a \[SmallCircle] b ^ c]"#).unwrap(),
      "SmallCircle"
    );
    assert_eq!(
      interpret(r#"Head[a \[SmallCircle] b /@ c]"#).unwrap(),
      "SmallCircle"
    );
  }

  // \[Diamond] and \[CircleDot] are flat operators bracketing Dot:
  // Wedge < Diamond < Dot < CircleDot < SmallCircle.
  #[test]
  fn diamond_and_circle_dot_operators() {
    assert_eq!(interpret(r#"a \[Diamond] b"#).unwrap(), "a \u{22C4} b");
    assert_eq!(interpret(r#"a \[CircleDot] b"#).unwrap(), "a \u{2299} b");
    assert_eq!(
      interpret("Diamond[a, b, c]").unwrap(),
      "a \u{22C4} b \u{22C4} c"
    );
    // Both flat.
    assert_eq!(
      interpret(r#"Length[a \[Diamond] b \[Diamond] c]"#).unwrap(),
      "3"
    );
    // Diamond is tighter than Wedge but looser than Dot.
    assert_eq!(
      interpret(r#"Head[a \[Diamond] b \[Wedge] c]"#).unwrap(),
      "Wedge"
    );
    assert_eq!(interpret(r#"Head[a \[Diamond] b . c]"#).unwrap(), "Diamond");
    // CircleDot is tighter than Dot but looser than SmallCircle and Power.
    assert_eq!(interpret(r#"Head[a \[CircleDot] b . c]"#).unwrap(), "Dot");
    assert_eq!(
      interpret(r#"Head[a \[CircleDot] b \[SmallCircle] c]"#).unwrap(),
      "CircleDot"
    );
    // CircleDot binds tighter than Diamond.
    assert_eq!(
      interpret(r#"Head[a \[Diamond] b \[CircleDot] c]"#).unwrap(),
      "Diamond"
    );
  }

  // \[Backslash] is a flat operator between Diamond and Dot.
  #[test]
  fn backslash_operator() {
    assert_eq!(interpret(r#"a \[Backslash] b"#).unwrap(), "a \u{2216} b");
    assert_eq!(
      interpret("Backslash[a, b, c]").unwrap(),
      "a \u{2216} b \u{2216} c"
    );
    assert_eq!(
      interpret(r#"Length[a \[Backslash] b \[Backslash] c]"#).unwrap(),
      "3"
    );
    // Tighter than Diamond, Times and Plus; looser than Dot.
    assert_eq!(
      interpret(r#"Head[a \[Backslash] b \[Diamond] c]"#).unwrap(),
      "Diamond"
    );
    assert_eq!(interpret(r#"Head[a \[Backslash] b * c]"#).unwrap(), "Times");
    assert_eq!(
      interpret(r#"Head[a \[Backslash] b . c]"#).unwrap(),
      "Backslash"
    );
  }
}

mod wedge {
  use super::*;

  #[test]
  fn binary() {
    assert_eq!(interpret("Wedge[a, b]").unwrap(), "a \u{22C0} b");
  }
}

mod del {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Del[f]").unwrap(), "\u{2207}f");
  }
}

mod cycles {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("Cycles[{{1, 2, 3}}]").unwrap(),
      "Cycles[{{1, 2, 3}}]"
    );
  }

  #[test]
  fn permutation_list_single_cycle() {
    assert_eq!(
      interpret("PermutationList[Cycles[{{1, 3, 2}}]]").unwrap(),
      "{3, 1, 2}"
    );
  }

  #[test]
  fn permutation_list_transposition() {
    assert_eq!(
      interpret("PermutationList[Cycles[{{1, 2}}]]").unwrap(),
      "{2, 1}"
    );
  }

  #[test]
  fn permutation_list_identity() {
    assert_eq!(interpret("PermutationList[Cycles[{}]]").unwrap(), "{}");
  }

  #[test]
  fn permutation_list_two_cycles() {
    assert_eq!(
      interpret("PermutationList[Cycles[{{1, 2}, {3, 4}}]]").unwrap(),
      "{2, 1, 4, 3}"
    );
  }

  #[test]
  fn permutation_list_with_length() {
    assert_eq!(
      interpret("PermutationList[Cycles[{{1, 2}}], 4]").unwrap(),
      "{2, 1, 3, 4}"
    );
  }

  // PermutationList also accepts a permutation list directly: it validates the
  // list and optionally re-lengths it (padding with fixed points or trimming
  // trailing ones). Verified against wolframscript.
  #[test]
  fn permutation_list_from_permutation_list() {
    assert_eq!(
      interpret("PermutationList[{2, 3, 1}]").unwrap(),
      "{2, 3, 1}"
    );
    // Fixed points are kept when the list is already the right length.
    assert_eq!(
      interpret("PermutationList[{2, 1, 3, 4}]").unwrap(),
      "{2, 1, 3, 4}"
    );
  }

  #[test]
  fn permutation_list_from_list_relengthed() {
    // Padding with trailing fixed points.
    assert_eq!(
      interpret("PermutationList[{2, 3, 1}, 5]").unwrap(),
      "{2, 3, 1, 4, 5}"
    );
    // Trimming trailing fixed points down to the support.
    assert_eq!(
      interpret("PermutationList[{2, 3, 1, 4, 5}, 3]").unwrap(),
      "{2, 3, 1}"
    );
  }

  #[test]
  fn permutation_list_invalid_stays_unevaluated() {
    // Not a permutation of {1, 2, 3}.
    assert_eq!(
      interpret("PermutationList[{2, 3, 3}]").unwrap(),
      "PermutationList[{2, 3, 3}]"
    );
    // Requested length is below the support maximum.
    assert_eq!(
      interpret("PermutationList[{2, 3, 1}, 2]").unwrap(),
      "PermutationList[{2, 3, 1}, 2]"
    );
  }

  #[test]
  fn permutation_cycles_single_cycle() {
    assert_eq!(
      interpret("PermutationCycles[{3, 1, 2}]").unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
  }

  #[test]
  fn permutation_cycles_transposition() {
    assert_eq!(
      interpret("PermutationCycles[{2, 1, 3}]").unwrap(),
      "Cycles[{{1, 2}}]"
    );
  }

  #[test]
  fn permutation_cycles_identity() {
    assert_eq!(
      interpret("PermutationCycles[{1, 2, 3}]").unwrap(),
      "Cycles[{}]"
    );
  }

  #[test]
  fn permutation_cycles_single_element() {
    assert_eq!(interpret("PermutationCycles[{1}]").unwrap(), "Cycles[{}]");
  }

  #[test]
  fn permutation_cycles_full_cycle() {
    assert_eq!(
      interpret("PermutationCycles[{2, 3, 4, 1}]").unwrap(),
      "Cycles[{{1, 2, 3, 4}}]"
    );
  }

  #[test]
  fn permutation_cycles_two_transpositions() {
    assert_eq!(
      interpret("PermutationCycles[{2, 1, 4, 3}]").unwrap(),
      "Cycles[{{1, 2}, {3, 4}}]"
    );
  }

  #[test]
  fn permutation_cycles_mixed() {
    assert_eq!(
      interpret("PermutationCycles[{3, 1, 2, 5, 4}]").unwrap(),
      "Cycles[{{1, 3, 2}, {4, 5}}]"
    );
  }

  #[test]
  fn permutation_cycles_roundtrip() {
    assert_eq!(
      interpret("PermutationList[PermutationCycles[{3, 1, 2, 5, 4}]]").unwrap(),
      "{3, 1, 2, 5, 4}"
    );
  }
}

mod circle_plus {
  use super::*;

  #[test]
  fn binary() {
    assert_eq!(interpret("CirclePlus[a, b]").unwrap(), "a \u{2295} b");
  }

  #[test]
  fn ternary() {
    assert_eq!(
      interpret("CirclePlus[a, b, c]").unwrap(),
      "a \u{2295} b \u{2295} c"
    );
  }
}

mod file_exists_q {
  use super::*;

  #[test]
  fn existing_path() {
    assert_eq!(interpret("FileExistsQ[\"/tmp\"]").unwrap(), "True");
  }

  #[test]
  fn nonexistent_path() {
    assert_eq!(
      interpret("FileExistsQ[\"/nonexistent_path_xyz\"]").unwrap(),
      "False"
    );
  }
}

mod unit_vector {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("UnitVector[3, 2]").unwrap(), "{0, 1, 0}");
  }

  #[test]
  fn first_element() {
    assert_eq!(interpret("UnitVector[5, 1]").unwrap(), "{1, 0, 0, 0, 0}");
  }

  #[test]
  fn shorthand() {
    assert_eq!(interpret("UnitVector[2]").unwrap(), "{0, 1}");
  }

  #[test]
  fn shorthand_first() {
    assert_eq!(interpret("UnitVector[1]").unwrap(), "{1, 0}");
  }

  // A too-large positive direction emits ::nokun and stays unevaluated.
  #[test]
  fn out_of_range_direction() {
    clear_state();
    let r = interpret_with_stdout("UnitVector[5, 6]").unwrap();
    assert_eq!(r.result, "UnitVector[5, 6]");
    assert!(r.warnings[0].contains(
      "UnitVector::nokun: There is no unit vector in direction 6 in 5 dimensions."
    ));
    // The one-argument form is two-dimensional.
    clear_state();
    let r = interpret_with_stdout("UnitVector[3]").unwrap();
    assert_eq!(r.result, "UnitVector[3]");
    assert!(r.warnings[0].contains(
      "UnitVector::nokun: There is no unit vector in direction 3 in 2 dimensions."
    ));
  }

  // A non-positive direction emits ::intpm (positive integer expected).
  #[test]
  fn non_positive_direction() {
    clear_state();
    let r = interpret_with_stdout("UnitVector[5, 0]").unwrap();
    assert_eq!(r.result, "UnitVector[5, 0]");
    assert!(r.warnings[0].contains(
      "UnitVector::intpm: Positive machine-sized integer expected at position 2 in UnitVector[5, 0]."
    ));
    clear_state();
    let r = interpret_with_stdout("UnitVector[0]").unwrap();
    assert_eq!(r.result, "UnitVector[0]");
    assert!(r.warnings[0].contains(
      "UnitVector::intpm: Positive machine-sized integer expected at position 1 in UnitVector[0]."
    ));
  }
}

mod permute {
  use super::*;

  #[test]
  fn list_form() {
    assert_eq!(
      interpret("Permute[{a, b, c, d}, {3, 1, 4, 2}]").unwrap(),
      "{b, d, a, c}"
    );
  }

  #[test]
  fn cycles_form() {
    assert_eq!(
      interpret("Permute[{a, b, c}, Cycles[{{1, 3, 2}}]]").unwrap(),
      "{b, c, a}"
    );
  }

  #[test]
  fn identity() {
    assert_eq!(
      interpret("Permute[{a, b, c}, {1, 2, 3}]").unwrap(),
      "{a, b, c}"
    );
  }

  // Permute operates on any head, not only Lists; the result keeps the head.
  #[test]
  fn non_list_head_cycles() {
    assert_eq!(
      interpret("Permute[f[a, b, c], Cycles[{{1, 2, 3}}]]").unwrap(),
      "f[c, a, b]"
    );
    assert_eq!(
      interpret("Permute[g[x, y, z], Cycles[{{1, 3}}]]").unwrap(),
      "g[z, y, x]"
    );
  }

  #[test]
  fn non_list_head_list_form() {
    assert_eq!(
      interpret("Permute[f[a, b, c, d], {2, 3, 4, 1}]").unwrap(),
      "f[d, a, b, c]"
    );
  }
}

mod delete_file {
  use super::*;

  #[test]
  fn delete_existing() {
    assert_eq!(
      interpret(
        "WriteString[\"/tmp/test_delete_woxi.txt\", \"hello\"]; DeleteFile[\"/tmp/test_delete_woxi.txt\"]; FileExistsQ[\"/tmp/test_delete_woxi.txt\"]"
      )
      .unwrap(),
      "False"
    );
  }
}

mod delete_missing {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("DeleteMissing[{1, Missing[], 3, Missing[\"x\"], 5}]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn no_missing() {
    assert_eq!(interpret("DeleteMissing[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn all_missing() {
    assert_eq!(
      interpret("DeleteMissing[{Missing[], Missing[]}]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("DeleteMissing[{}]").unwrap(), "{}");
  }

  // On an association, pairs whose value is Missing[...] are dropped.
  #[test]
  fn association() {
    assert_eq!(
      interpret("DeleteMissing[<|a -> 1, b -> Missing[], c -> 3|>]").unwrap(),
      "<|a -> 1, c -> 3|>"
    );
  }

  #[test]
  fn association_with_missing_arg() {
    assert_eq!(
      interpret("DeleteMissing[<|a -> Missing[\"NotAvailable\"], b -> 2|>]")
        .unwrap(),
      "<|b -> 2|>"
    );
  }

  #[test]
  fn association_no_missing() {
    assert_eq!(
      interpret("DeleteMissing[<|a -> 1, b -> 2|>]").unwrap(),
      "<|a -> 1, b -> 2|>"
    );
  }
}

mod angle_bracket {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("AngleBracket[a, b, c]").unwrap(),
      "\u{2329} a, b, c \u{232A}"
    );
  }

  #[test]
  fn single() {
    assert_eq!(interpret("AngleBracket[x]").unwrap(), "\u{2329} x \u{232A}");
  }
}

mod before {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Before[3]").unwrap(), "Before[3]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Before[3]]").unwrap(), "Before");
  }

  #[test]
  fn string_arg() {
    assert_eq!(interpret("Before[\"cat\"]").unwrap(), "Before[cat]");
  }
}

mod complexity_function {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("ComplexityFunction[3]").unwrap(),
      "ComplexityFunction[3]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[ComplexityFunction]").unwrap(), "Symbol");
  }

  #[test]
  fn multiple_args() {
    assert_eq!(
      interpret("ComplexityFunction[x, y]").unwrap(),
      "ComplexityFunction[x, y]"
    );
  }
}

mod compilation_options {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("CompilationOptions[1, 2, 3]").unwrap(),
      "CompilationOptions[1, 2, 3]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[CompilationOptions]").unwrap(), "Symbol");
  }

  #[test]
  fn no_args() {
    assert_eq!(
      interpret("CompilationOptions[]").unwrap(),
      "CompilationOptions[]"
    );
  }
}

mod tagging_rules {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("TaggingRules[1, 2]").unwrap(),
      "TaggingRules[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TaggingRules]").unwrap(), "Symbol");
  }
}

mod rationals {
  use super::*;

  #[test]
  fn unevaluated_with_args() {
    assert_eq!(interpret("Rationals[1, 2]").unwrap(), "Rationals[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Rationals]").unwrap(), "Symbol");
  }
}

mod file_function {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("File[1, 2]").unwrap(), "File[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[File]").unwrap(), "Symbol");
  }
}

mod bode_plot {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("BodePlot[x]").unwrap(), "BodePlot[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[BodePlot]").unwrap(), "Symbol");
  }
}

mod session_time {
  use super::*;

  #[test]
  fn returns_real() {
    assert_eq!(interpret("Head[SessionTime[]]").unwrap(), "Real");
  }

  #[test]
  fn non_negative() {
    let result = interpret("SessionTime[] >= 0").unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn monotonically_increasing() {
    let result =
      interpret("t1 = SessionTime[]; t2 = SessionTime[]; t2 >= t1").unwrap();
    assert_eq!(result, "True");
  }
}

mod unix_time {
  use super::*;

  #[test]
  fn returns_integer() {
    assert_eq!(interpret("Head[UnixTime[]]").unwrap(), "Integer");
  }

  #[test]
  fn reasonable_value() {
    // Unix time should be after 2025-01-01
    assert_eq!(interpret("UnixTime[] > 1735689600").unwrap(), "True");
  }
}

mod positive_reals {
  use super::*;

  #[test]
  fn symbol_passthrough() {
    assert_eq!(interpret("PositiveReals").unwrap(), "PositiveReals");
  }

  #[test]
  fn element_positive_integer() {
    assert_eq!(interpret("Element[3, PositiveReals]").unwrap(), "True");
  }

  #[test]
  fn element_negative_integer() {
    assert_eq!(interpret("Element[-3, PositiveReals]").unwrap(), "False");
  }

  #[test]
  fn element_zero() {
    assert_eq!(interpret("Element[0, PositiveReals]").unwrap(), "False");
  }

  #[test]
  fn element_positive_real() {
    assert_eq!(interpret("Element[2.5, PositiveReals]").unwrap(), "True");
  }

  #[test]
  fn element_positive_rational() {
    assert_eq!(interpret("Element[1/3, PositiveReals]").unwrap(), "True");
  }
}

mod positive_integers {
  use super::*;

  #[test]
  fn symbol_passthrough() {
    assert_eq!(interpret("PositiveIntegers").unwrap(), "PositiveIntegers");
  }

  #[test]
  fn element_positive() {
    assert_eq!(interpret("Element[3, PositiveIntegers]").unwrap(), "True");
  }

  #[test]
  fn element_negative() {
    assert_eq!(interpret("Element[-3, PositiveIntegers]").unwrap(), "False");
  }

  #[test]
  fn element_zero() {
    assert_eq!(interpret("Element[0, PositiveIntegers]").unwrap(), "False");
  }

  #[test]
  fn element_real_not_integer() {
    assert_eq!(
      interpret("Element[3.5, PositiveIntegers]").unwrap(),
      "False"
    );
  }

  #[test]
  fn element_non_negative_integers() {
    assert_eq!(
      interpret("Element[0, NonNegativeIntegers]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Element[5, NonNegativeIntegers]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Element[-1, NonNegativeIntegers]").unwrap(),
      "False"
    );
  }

  #[test]
  fn element_negative_reals() {
    assert_eq!(interpret("Element[-2.5, NegativeReals]").unwrap(), "True");
    assert_eq!(interpret("Element[1, NegativeReals]").unwrap(), "False");
  }
}

mod geo_grid_position {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("GeoGridPosition[1, 2]").unwrap(),
      "GeoGridPosition[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[GeoGridPosition]").unwrap(), "Symbol");
  }
}

mod opener_view {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("OpenerView[1, 2]").unwrap(), "OpenerView[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[OpenerView]").unwrap(), "Symbol");
  }
}

mod max_step_size {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("MaxStepSize[1, 2]").unwrap(), "MaxStepSize[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[MaxStepSize]").unwrap(), "Symbol");
  }
}

mod radio_button_bar {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("RadioButtonBar[1, 2]").unwrap(),
      "RadioButtonBar[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[RadioButtonBar]").unwrap(), "Symbol");
  }
}

mod step_monitor {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("StepMonitor[1, 2]").unwrap(), "StepMonitor[1, 2]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[StepMonitor]").unwrap(), "Symbol");
  }
}

mod formula_lookup {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("FormulaLookup[x]").unwrap(), "FormulaLookup[x]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[FormulaLookup]").unwrap(), "Symbol");
  }
}

mod tensor_contract {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("TensorContract[1, 2]").unwrap(),
      "TensorContract[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TensorContract]").unwrap(), "Symbol");
  }

  #[test]
  fn matrix_trace() {
    // Contracting the only two indices of a matrix gives its trace.
    assert_eq!(
      interpret("TensorContract[{{a, b}, {c, d}}, {{1, 2}}]").unwrap(),
      "a + d"
    );
  }

  #[test]
  fn rank3_contract_first_third() {
    // T[i,j,k] contracted on indices 1 and 3 → Sum[T[k, j, k], k].
    // For T = {{{1,2},{3,4}}, {{5,6},{7,8}}}: result[j] = T[1,j,1] + T[2,j,2]
    // result[1] = 1 + 6 = 7, result[2] = 3 + 8 = 11.
    assert_eq!(
      interpret(
        "TensorContract[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{1, 3}}]"
      )
      .unwrap(),
      "{7, 11}"
    );
  }

  #[test]
  fn rank3_contract_first_second() {
    // T[i,j,k] contracted on indices 1 and 2 → Sum[T[k, k, j], k].
    // For T = {{{1,2},{3,4}}, {{5,6},{7,8}}}: result[k] = T[1,1,k] + T[2,2,k]
    // result[1] = 1 + 7 = 8, result[2] = 2 + 8 = 10.
    assert_eq!(
      interpret(
        "TensorContract[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{1, 2}}]"
      )
      .unwrap(),
      "{8, 10}"
    );
  }

  #[test]
  fn audit_case_rank3_contract_second_third() {
    // Audit case: a 2x2x2 symbolic tensor contracted on slots 2 and 3.
    // result[i] = T[i,1,1] + T[i,2,2].
    let result = interpret(
      "TensorContract[{{{Subscript[a, 1, 1, 1], Subscript[a, 1, 1, 2]}, {Subscript[a, 1, 2, 1], Subscript[a, 1, 2, 2]}}, {{Subscript[a, 2, 1, 1], Subscript[a, 2, 1, 2]}, {Subscript[a, 2, 2, 1], Subscript[a, 2, 2, 2]}}}, {{2, 3}}]"
    )
    .unwrap();
    assert_eq!(
      result,
      "{Subscript[a, 1, 1, 1] + Subscript[a, 1, 2, 2], Subscript[a, 2, 1, 1] + Subscript[a, 2, 2, 2]}"
    );
  }

  #[test]
  fn rank4_contract_two_pairs() {
    // Array[a, {2,2,2,2}] contracted on {{1,2}, {3,4}} → scalar.
    // result = Sum[a[i,i,j,j], i, j].
    assert_eq!(
      interpret("TensorContract[Array[a, {2, 2, 2, 2}], {{1, 2}, {3, 4}}]")
        .unwrap(),
      "a[1, 1, 1, 1] + a[1, 1, 2, 2] + a[2, 2, 1, 1] + a[2, 2, 2, 2]"
    );
  }

  // A flat two-integer list {i, j} is the shorthand for the single-pair
  // list-of-pairs {{i, j}} and must give the identical contraction.
  #[test]
  fn single_pair_shorthand_matrix_trace() {
    assert_eq!(
      interpret("TensorContract[{{1, 2}, {3, 4}}, {1, 2}]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("TensorContract[{{a, b}, {c, d}}, {1, 2}]").unwrap(),
      "a + d"
    );
  }

  #[test]
  fn single_pair_shorthand_rank3() {
    // T contracted on slots 1 and 2 with the {1,2} shorthand.
    assert_eq!(
      interpret("TensorContract[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {1, 2}]")
        .unwrap(),
      "{8, 10}"
    );
  }
}

mod clear_system_cache {
  use super::*;

  #[test]
  fn returns_null() {
    assert_eq!(interpret("ClearSystemCache[]").unwrap(), "\0");
  }

  #[test]
  fn with_arg_returns_null() {
    assert_eq!(interpret("ClearSystemCache[\"Numeric\"]").unwrap(), "\0");
  }
}

mod locator_auto_create {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("LocatorAutoCreate[1, 2]").unwrap(),
      "LocatorAutoCreate[1, 2]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[LocatorAutoCreate]").unwrap(), "Symbol");
  }
}

mod transformed_field {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("TransformedField[x, y, z]").unwrap(),
      "TransformedField[x, y, z]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[TransformedField]").unwrap(), "Symbol");
  }
}

mod geo_projection {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("GeoProjection[x, y]").unwrap(),
      "GeoProjection[x, y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[GeoProjection]").unwrap(), "Symbol");
  }
}

mod sound {
  use super::*;

  #[test]
  fn renders_as_minus_sound_minus() {
    // Sound[primitives] formats as -Sound- in the REPL, like Graphics
    // formats as -Graphics-.
    assert_eq!(interpret("Sound[SoundNote[0]]").unwrap(), "-Sound-");
  }

  #[test]
  fn renders_with_list_of_plays() {
    assert_eq!(
      interpret(
        "Sound[{Play[Sin[1000*t], {t, 0, 0.2}], Play[Sin[500*t], {t, 0, 0.5}]}]"
      )
      .unwrap(),
      "-Sound-"
    );
  }

  #[test]
  fn head_is_sound() {
    assert_eq!(interpret("Head[Sound[SoundNote[0]]]").unwrap(), "Sound");
  }

  #[test]
  fn empty_stays_symbolic() {
    // Sound[] with no args has nothing to render; stay unevaluated.
    assert_eq!(interpret("Sound[]").unwrap(), "Sound[]");
  }
}

mod play {
  use super::*;

  // Play[f, {t, tmin, tmax}] builds a sound object. Like Sound, it renders
  // as -Sound- in the REPL regardless of the amplitude function it wraps.
  #[test]
  fn renders_as_minus_sound_minus() {
    // Play a "middle A" sine wave for 1 second.
    assert_eq!(
      interpret("Play[Sin[440*2*Pi*t], {t, 0, 1}]").unwrap(),
      "-Sound-"
    );
  }

  #[test]
  fn head_is_sound() {
    // Play evaluates to a Sound object, so its Head is Sound (not Play).
    assert_eq!(interpret("Head[Play[Sin[t], {t, 0, 1}]]").unwrap(), "Sound");
  }

  #[test]
  fn symbolic_amplitude_renders_as_sound() {
    // The amplitude function may be any expression of the time variable.
    assert_eq!(interpret("Play[t^2, {t, 0, 2}]").unwrap(), "-Sound-");
  }

  #[test]
  fn incomplete_args_stay_symbolic() {
    // Without a proper {t, tmin, tmax} iterator there is nothing to build.
    assert_eq!(interpret("Play[Sin[t]]").unwrap(), "Play[Sin[t]]");
  }
}

mod sound_volume {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("SoundVolume[x, y]").unwrap(), "SoundVolume[x, y]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[SoundVolume]").unwrap(), "Symbol");
  }
}

mod gradient_filter {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("GradientFilter[x, y]").unwrap(),
      "GradientFilter[x, y]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[GradientFilter]").unwrap(), "Symbol");
  }
}

mod take_drop {
  use super::*;

  #[test]
  fn positive_n() {
    assert_eq!(
      interpret("TakeDrop[{a, b, c, d, e}, 3]").unwrap(),
      "{{a, b, c}, {d, e}}"
    );
  }

  #[test]
  fn negative_n() {
    assert_eq!(
      interpret("TakeDrop[{a, b, c, d, e}, -2]").unwrap(),
      "{{d, e}, {a, b, c}}"
    );
  }

  #[test]
  fn range_spec() {
    assert_eq!(
      interpret("TakeDrop[{a, b, c, d, e}, {2, 4}]").unwrap(),
      "{{b, c, d}, {a, e}}"
    );
  }

  #[test]
  fn zero() {
    assert_eq!(
      interpret("TakeDrop[{a, b, c, d, e}, 0]").unwrap(),
      "{{}, {a, b, c, d, e}}"
    );
  }
}

mod array_rules {
  use super::*;

  #[test]
  fn matrix() {
    assert_eq!(
      interpret("ArrayRules[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {_, _} -> 0}"
    );
  }

  #[test]
  fn vector() {
    assert_eq!(
      interpret("ArrayRules[{5, 0, 3, 0, 1}]").unwrap(),
      "{{1} -> 5, {3} -> 3, {5} -> 1, {_} -> 0}"
    );
  }

  #[test]
  fn all_zeros() {
    assert_eq!(interpret("ArrayRules[{0, 0, 0}]").unwrap(), "{{_} -> 0}");
  }

  #[test]
  fn custom_default() {
    assert_eq!(
      interpret("ArrayRules[{1, x, 1}, 1]").unwrap(),
      "{{2} -> x, {_} -> 1}"
    );
  }

  #[test]
  fn sparse_array_1d() {
    assert_eq!(
      interpret("ArrayRules[SparseArray[{1 -> 5, 3 -> 7}, 5]]").unwrap(),
      "{{1} -> 5, {3} -> 7, {_} -> 0}"
    );
  }

  #[test]
  fn sparse_array_1d_list_dim() {
    assert_eq!(
      interpret("ArrayRules[SparseArray[{1 -> 5, 3 -> 7}, {5}]]").unwrap(),
      "{{1} -> 5, {3} -> 7, {_} -> 0}"
    );
  }

  #[test]
  fn sparse_array_2d() {
    assert_eq!(
      interpret(
        "ArrayRules[SparseArray[{{1, 2} -> 5, {3, 1} -> 7}, {4, 3}, 0]]"
      )
      .unwrap(),
      "{{1, 2} -> 5, {3, 1} -> 7, {_, _} -> 0}"
    );
  }

  // Inferred dimensions: 2D rules with no explicit dims
  #[test]
  fn sparse_array_inferred_dims_2d() {
    // Wolfram's CSR internal form: {1, {row_ptr, inner_positions}, values}
    assert_eq!(
      interpret(
        "SparseArray[{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {1, 3} -> 4}]"
      )
      .unwrap(),
      "SparseArray[Automatic, {3, 3}, 0, {1, {{0, 2, 3, 4}, {{1}, {3}, {2}, {3}}}, {1, 4, 2, 3}}]"
    );
  }

  #[test]
  fn sparse_array_inferred_normal() {
    assert_eq!(
      interpret(
        "Normal[SparseArray[{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {1, 3} -> 4}]]"
      )
      .unwrap(),
      "{{1, 0, 4}, {0, 2, 0}, {0, 0, 3}}"
    );
  }

  #[test]
  fn sparse_array_inferred_dimensions() {
    assert_eq!(
      interpret(
        "Dimensions[SparseArray[{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {1, 3} -> 4}]]"
      )
      .unwrap(),
      "{3, 3}"
    );
  }

  #[test]
  fn sparse_array_inferred_array_rules() {
    assert_eq!(
      interpret(
        "ArrayRules[SparseArray[{{1, 1} -> 1, {2, 2} -> 2, {3, 3} -> 3, {1, 3} -> 4}]]"
      )
      .unwrap(),
      "{{1, 1} -> 1, {1, 3} -> 4, {2, 2} -> 2, {3, 3} -> 3, {_, _} -> 0}"
    );
  }

  // Inferred dimensions for 1D (bare-integer positions).
  #[test]
  fn sparse_array_inferred_1d_scalar_positions() {
    assert_eq!(
      interpret("SparseArray[{1 -> 5, 3 -> 7, 5 -> 2}]").unwrap(),
      "SparseArray[Automatic, {5}, 0, {1, {{0, 3}, {{1}, {3}, {5}}}, {5, 7, 2}}]"
    );
  }

  // Dense list input → sparse form (default 0, non-zero entries become rules).
  #[test]
  fn sparse_array_from_dense_2d() {
    assert_eq!(
      interpret("SparseArray[{{0, a}, {b, 0}}]").unwrap(),
      "SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, {a, b}}]"
    );
  }

  #[test]
  fn sparse_array_from_dense_1d_normal() {
    assert_eq!(
      interpret("Normal[SparseArray[{1, 0, 3, 0, 5}]]").unwrap(),
      "{1, 0, 3, 0, 5}"
    );
  }

  // Explicit default fill value.
  #[test]
  fn sparse_array_custom_default() {
    assert_eq!(
      interpret("SparseArray[{{1, 2} -> a}, {3, 3}, x]").unwrap(),
      "SparseArray[Automatic, {3, 3}, x, {1, {{0, 1, 1, 1}, {{2}}}, {a}}]"
    );
  }

  #[test]
  fn sparse_array_custom_default_normal() {
    assert_eq!(
      interpret("Normal[SparseArray[{{1, 2} -> a}, {3, 3}, x]]").unwrap(),
      "{{x, a, x}, {x, x, x}, {x, x, x}}"
    );
  }

  // Rules matching the default are dropped from the canonical form.
  #[test]
  fn sparse_array_drops_default_rules() {
    assert_eq!(
      interpret("SparseArray[{{1, 1} -> 0, {2, 2} -> 3}, {2, 2}]").unwrap(),
      "SparseArray[Automatic, {2, 2}, 0, {1, {{0, 0, 1}, {{2}}}, {3}}]"
    );
  }

  // Earlier rules win over later rules at the same position (matches
  // wolframscript: Normal[SparseArray[{1 -> 5, 1 -> 9}, 3]] == {5, 0, 0}).
  #[test]
  fn sparse_array_earlier_rule_wins() {
    assert_eq!(
      interpret("Normal[SparseArray[{1 -> 5, 1 -> 9}, 3]]").unwrap(),
      "{5, 0, 0}"
    );
  }

  // Normal acts at all levels: SparseArrays nested inside a list densify too
  // (e.g. Normal[CoefficientArrays[...]]). Regression: a list of SparseArrays
  // was returned unchanged.
  #[test]
  fn normal_threads_into_list_of_sparse_arrays() {
    assert_eq!(
      interpret("Normal[{SparseArray[{1 -> 3}, 2], SparseArray[{2 -> 5}, 3]}]")
        .unwrap(),
      "{{3, 0}, {0, 5, 0}}"
    );
    assert_eq!(
      interpret("Normal[CoefficientArrays[1 + 2 x + 3 x^2, x]]").unwrap(),
      "{1, {2}, {{3}}}"
    );
  }

  // Normal densifies SparseArrays inside an arithmetic expression and then
  // evaluates the structural operation over the dense pieces. Regression: a
  // sum of SparseArrays was densified to `{5, 0, 0} + {0, 3, 0}` but the list
  // addition was left unevaluated.
  #[test]
  fn normal_evaluates_sum_of_sparse_arrays() {
    assert_eq!(
      interpret("Normal[SparseArray[{1 -> 5}, 3] + SparseArray[{2 -> 3}, 3]]")
        .unwrap(),
      "{5, 3, 0}"
    );
    assert_eq!(
      interpret("Normal[2 SparseArray[{1 -> 5}, 3]]").unwrap(),
      "{10, 0, 0}"
    );
  }

  // Canonical form is idempotent under re-normalization.
  #[test]
  fn sparse_array_canonical_idempotent() {
    assert_eq!(
      interpret(
        "SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, {a, b}}]"
      )
      .unwrap(),
      "SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, {a, b}}]"
    );
  }
}

mod value_dimensions {
  use super::*;

  #[test]
  fn symbol_returns_itself() {
    assert_eq!(interpret("ValueDimensions").unwrap(), "ValueDimensions");
  }

  #[test]
  fn as_option_rule() {
    assert_eq!(
      interpret("ValueDimensions -> 3").unwrap(),
      "ValueDimensions -> 3"
    );
  }

  #[test]
  fn applied_to_arg() {
    assert_eq!(
      interpret("ValueDimensions[0]").unwrap(),
      "ValueDimensions[0]"
    );
  }
}

mod stream_position {
  use super::*;

  #[test]
  fn initial_position() {
    clear_state();
    assert_eq!(
      interpret("s = StringToStream[\"hello world\"]; StreamPosition[s]")
        .unwrap(),
      "0"
    );
  }

  #[test]
  fn after_read_word() {
    clear_state();
    assert_eq!(
      interpret(
        "s = StringToStream[\"hello world\"]; Read[s, Word]; StreamPosition[s]"
      )
      .unwrap(),
      "5"
    );
  }

  #[test]
  fn after_two_reads() {
    clear_state();
    assert_eq!(
      interpret("s = StringToStream[\"hello world\"]; Read[s, Word]; Read[s, Word]; StreamPosition[s]").unwrap(),
      "11"
    );
  }

  #[test]
  fn set_stream_position() {
    clear_state();
    assert_eq!(
      interpret("s = StringToStream[\"hello world\"]; Read[s, Word]; SetStreamPosition[s, 0]; StreamPosition[s]").unwrap(),
      "0"
    );
  }

  #[test]
  fn set_and_read_again() {
    clear_state();
    assert_eq!(
      interpret("s = StringToStream[\"hello world\"]; Read[s, Word]; SetStreamPosition[s, 0]; Read[s, Word]").unwrap(),
      "hello"
    );
  }
}

mod date_interval {
  use super::*;

  #[test]
  fn from_date_lists() {
    assert_eq!(
      interpret("DateInterval[{{2020, 1, 1}, {2020, 12, 31}}]").unwrap(),
      "DateInterval[{{{2020, 1, 1, 0, 0, 0.}, {2020, 12, 31, 0, 0, 0.}}}, Day, Gregorian, None]"
    );
  }

  #[test]
  fn from_date_strings() {
    assert_eq!(
      interpret("DateInterval[{\"Jan 1, 2020\", \"Dec 31, 2020\"}]").unwrap(),
      "DateInterval[{{{2020, 1, 1, 0, 0, 0.}, {2020, 12, 31, 0, 0, 0.}}}, Day, Gregorian, None]"
    );
  }

  #[test]
  fn from_date_objects() {
    assert_eq!(
      interpret(
        "DateInterval[{DateObject[{2020, 1, 1}], DateObject[{2020, 12, 31}]}]"
      )
      .unwrap(),
      "DateInterval[{{{2020, 1, 1, 0, 0, 0.}, {2020, 12, 31, 0, 0, 0.}}}, Day, Gregorian, None]"
    );
  }

  #[test]
  fn input_form_has_quoted_strings() {
    assert_eq!(
      interpret(
        "ToString[DateInterval[{{2020, 1, 1}, {2020, 12, 31}}], InputForm]"
      )
      .unwrap(),
      "DateInterval[{{{2020, 1, 1, 0, 0, 0.}, {2020, 12, 31, 0, 0, 0.}}}, \"Day\", \"Gregorian\", None]"
    );
  }

  #[test]
  fn invalid_arg() {
    assert_eq!(interpret("DateInterval[0]").unwrap(), "DateInterval[0]");
  }

  // Regression: re-evaluating the canonical form must be a fixed point and
  // not emit DateInterval::argx.
  #[test]
  fn canonical_form_is_fixed_point() {
    assert_eq!(
      interpret(
        "DateInterval[{{{2020, 1, 1, 0, 0, 0.}, {2020, 12, 31, 0, 0, 0.}}}, \
         \"Day\", \"Gregorian\", None]"
      )
      .unwrap(),
      "DateInterval[{{{2020, 1, 1, 0, 0, 0.}, {2020, 12, 31, 0, 0, 0.}}}, Day, Gregorian, None]"
    );
  }
}

mod xml_template {
  use super::*;

  #[test]
  fn returns_unevaluated() {
    assert_eq!(interpret("XMLTemplate[0]").unwrap(), "XMLTemplate[0]");
  }

  #[test]
  fn symbol_returns_itself() {
    assert_eq!(interpret("XMLTemplate").unwrap(), "XMLTemplate");
  }
}

mod byte_array {
  use super::*;

  #[test]
  fn from_list() {
    assert_eq!(interpret("ByteArray[{1, 2, 3}]").unwrap(), "ByteArray[<3>]");
  }

  #[test]
  fn from_base64() {
    assert_eq!(
      interpret("ByteArray[\"SGVsbG8=\"]").unwrap(),
      "ByteArray[<5>]"
    );
  }

  #[test]
  fn normal_extracts_list() {
    assert_eq!(
      interpret("Normal[ByteArray[{72, 101, 108, 108, 111}]]").unwrap(),
      "{72, 101, 108, 108, 111}"
    );
  }

  #[test]
  fn normal_base64() {
    assert_eq!(
      interpret("Normal[ByteArray[\"SGVsbG8=\"]]").unwrap(),
      "{72, 101, 108, 108, 111}"
    );
  }

  #[test]
  fn length() {
    assert_eq!(interpret("Length[ByteArray[{1, 2, 3}]]").unwrap(), "3");
  }

  #[test]
  fn input_form_from_list() {
    assert_eq!(
      interpret("ToString[ByteArray[{1, 2, 3}], InputForm]").unwrap(),
      "ByteArray[\"AQID\"]"
    );
  }

  #[test]
  fn input_form_from_base64() {
    assert_eq!(
      interpret("ToString[ByteArray[\"SGVsbG8=\"], InputForm]").unwrap(),
      "ByteArray[\"SGVsbG8=\"]"
    );
  }

  #[test]
  fn invalid_arg() {
    assert_eq!(interpret("ByteArray[0]").unwrap(), "ByteArray[0]");
  }
}

mod symbolic_wrappers {
  use super::*;

  #[test]
  fn transformation_function() {
    assert_eq!(
      interpret("TransformationFunction[{{1, 0}, {0, 1}}]").unwrap(),
      "TransformationFunction[{{1, 0}, {0, 1}}]"
    );
  }

  #[test]
  fn conic_hull_region() {
    assert_eq!(
      interpret("ConicHullRegion[{{1, 0}, {0, 1}}]").unwrap(),
      "ConicHullRegion[{{1, 0}, {0, 1}}]"
    );
  }

  #[test]
  fn hyperplane() {
    assert_eq!(
      interpret("Hyperplane[{1, 0, 0}, 0]").unwrap(),
      "Hyperplane[{1, 0, 0}, 0]"
    );
  }

  #[test]
  fn abelian_group() {
    assert_eq!(
      interpret("AbelianGroup[{2, 3}]").unwrap(),
      "AbelianGroup[{2, 3}]"
    );
  }

  #[test]
  fn full_graphics() {
    assert_eq!(interpret("FullGraphics[0]").unwrap(), "FullGraphics[0]");
  }

  #[test]
  fn delimited_sequence() {
    assert_eq!(
      interpret("DelimitedSequence[0]").unwrap(),
      "DelimitedSequence[0]"
    );
  }
}

mod primitive_root {
  use super::*;

  #[test]
  fn prime_7() {
    assert_eq!(interpret("PrimitiveRoot[7]").unwrap(), "3");
  }

  #[test]
  fn prime_13() {
    assert_eq!(interpret("PrimitiveRoot[13]").unwrap(), "2");
  }

  #[test]
  fn prime_23() {
    assert_eq!(interpret("PrimitiveRoot[23]").unwrap(), "5");
  }

  #[test]
  fn small_n_2() {
    assert_eq!(interpret("PrimitiveRoot[2]").unwrap(), "1");
  }

  #[test]
  fn n_4() {
    assert_eq!(interpret("PrimitiveRoot[4]").unwrap(), "3");
  }

  #[test]
  fn no_primitive_root() {
    // n=8 has no primitive root
    assert_eq!(interpret("PrimitiveRoot[8]").unwrap(), "PrimitiveRoot[8]");
    assert_eq!(interpret("PrimitiveRoot[12]").unwrap(), "PrimitiveRoot[12]");
    assert_eq!(interpret("PrimitiveRoot[15]").unwrap(), "PrimitiveRoot[15]");
  }

  #[test]
  fn n_too_small() {
    assert_eq!(interpret("PrimitiveRoot[1]").unwrap(), "PrimitiveRoot[1]");
  }

  // n = 2 p^k: wolframscript lifts the primitive root of p^k. When that
  // root g is even it uses g + p^k so the result is odd, e.g.
  // PrimitiveRoot[10] = 7 (from 2 mod 5), not the smaller valid root 3.
  #[test]
  fn twice_odd_prime_power_even_root_lifted() {
    assert_eq!(interpret("PrimitiveRoot[10]").unwrap(), "7");
    assert_eq!(interpret("PrimitiveRoot[18]").unwrap(), "11");
    assert_eq!(interpret("PrimitiveRoot[22]").unwrap(), "13");
    assert_eq!(interpret("PrimitiveRoot[50]").unwrap(), "27");
    assert_eq!(interpret("PrimitiveRoot[250]").unwrap(), "127");
  }

  // When the root of p^k is already odd, no lift is needed.
  #[test]
  fn twice_odd_prime_power_odd_root() {
    assert_eq!(interpret("PrimitiveRoot[6]").unwrap(), "5");
    assert_eq!(interpret("PrimitiveRoot[14]").unwrap(), "3");
    assert_eq!(interpret("PrimitiveRoot[98]").unwrap(), "3");
  }

  // Odd prime powers still use the smallest root directly.
  #[test]
  fn odd_prime_power() {
    assert_eq!(interpret("PrimitiveRoot[9]").unwrap(), "2");
    assert_eq!(interpret("PrimitiveRoot[25]").unwrap(), "2");
    assert_eq!(interpret("PrimitiveRoot[49]").unwrap(), "3");
  }

  // PrimitiveRoot[n, k] — smallest primitive root modulo n that is >= k.
  #[test]
  fn with_lower_bound() {
    assert_eq!(interpret("PrimitiveRoot[7, 2]").unwrap(), "3");
    assert_eq!(interpret("PrimitiveRoot[7, 4]").unwrap(), "5");
    assert_eq!(interpret("PrimitiveRoot[11, 3]").unwrap(), "6");
    assert_eq!(interpret("PrimitiveRoot[23, 6]").unwrap(), "7");
    assert_eq!(interpret("PrimitiveRoot[14, 5]").unwrap(), "5");
  }

  // For n = 2 p^k the bounded form uses the list's smallest root (3),
  // not the special unbounded value PrimitiveRoot[10] = 7.
  #[test]
  fn with_lower_bound_uses_full_list() {
    assert_eq!(interpret("PrimitiveRoot[10, 1]").unwrap(), "3");
    assert_eq!(interpret("PrimitiveRoot[10, 3]").unwrap(), "3");
  }

  // No primitive root reaches k, or n has none at all: stay unevaluated.
  #[test]
  fn with_lower_bound_unevaluated() {
    assert_eq!(
      interpret("PrimitiveRoot[7, 6]").unwrap(),
      "PrimitiveRoot[7, 6]"
    );
    assert_eq!(
      interpret("PrimitiveRoot[8, 2]").unwrap(),
      "PrimitiveRoot[8, 2]"
    );
  }
}

mod morphological_operations {
  use super::*;

  #[test]
  fn opening_1d_binary() {
    assert_eq!(
      interpret("Opening[{0, 1, 1, 0, 1, 1, 1, 0, 0, 1}, 1]").unwrap(),
      "{0, 0, 0, 0, 1, 1, 1, 0, 0, 0}"
    );
  }

  #[test]
  fn opening_1d_large_radius() {
    assert_eq!(
      interpret("Opening[{0, 1, 1, 0, 1, 1, 1, 0, 0, 1}, 2]").unwrap(),
      "{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}"
    );
  }

  #[test]
  fn opening_1d_grayscale() {
    assert_eq!(
      interpret("Opening[{0.5, 0.3, 0.8, 0.1, 0.9}, 1]").unwrap(),
      "{0.3, 0.3, 0.3, 0.1, 0.1}"
    );
  }

  #[test]
  fn opening_2d() {
    assert_eq!(
      interpret("Opening[{{1, 1, 0}, {1, 1, 0}, {0, 0, 0}}, 1]").unwrap(),
      "{{1, 1, 0}, {1, 1, 0}, {0, 0, 0}}"
    );
  }

  #[test]
  fn erosion_1d() {
    assert_eq!(
      interpret("Erosion[{0, 1, 1, 0, 1, 1, 1, 0, 0, 1}, 1]").unwrap(),
      "{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}"
    );
  }

  #[test]
  fn dilation_1d() {
    assert_eq!(
      interpret("Dilation[{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, 1]").unwrap(),
      "{0, 0, 0, 0, 1, 1, 1, 0, 0, 0}"
    );
  }

  #[test]
  fn closing_1d() {
    assert_eq!(
      interpret("Closing[{0, 1, 0, 1, 0, 1, 0}, 1]").unwrap(),
      "{1, 1, 1, 1, 1, 1, 1}"
    );
  }

  // The second argument may be a 0/1 structuring-element matrix instead of a
  // scalar radius.
  #[test]
  fn dilation_with_structuring_element() {
    // Dilating a single pixel by a cross stamps the cross shape.
    assert_eq!(
      interpret(
        "Dilation[{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}]"
      )
      .unwrap(),
      "{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}"
    );
    // An even-sized kernel is anchored above-left and reflected for dilation.
    assert_eq!(
      interpret(
        "Dilation[{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}, {{1, 1}, {1, 1}}]"
      )
      .unwrap(),
      "{{1, 1, 0}, {1, 1, 0}, {0, 0, 0}}"
    );
    // It composes with the structuring-element generators.
    assert_eq!(
      interpret(
        "Dilation[{{0, 1, 0}, {0, 1, 0}, {0, 1, 0}}, DiamondMatrix[1]]"
      )
      .unwrap(),
      "{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}"
    );
  }

  #[test]
  fn erosion_with_structuring_element() {
    // Erosion truncates the element at the border, so a full image survives.
    assert_eq!(
      interpret(
        "Erosion[{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}]"
      )
      .unwrap(),
      "{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}"
    );
    // A cross eroded by a cross leaves only the center.
    assert_eq!(
      interpret(
        "Erosion[{{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}}, CrossMatrix[1]]"
      )
      .unwrap(),
      "{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}"
    );
  }
}

mod base_encode_decode {
  use super::*;

  #[test]
  fn base_encode_byte_array() {
    assert_eq!(
      interpret("BaseEncode[ByteArray[{1, 2, 3}]]").unwrap(),
      "AQID"
    );
    assert_eq!(
      interpret(r#"BaseEncode[StringToByteArray["hello"]]"#).unwrap(),
      "aGVsbG8="
    );
  }

  #[test]
  fn base_decode_string() {
    assert_eq!(
      interpret(r#"BaseDecode["AQID"]"#).unwrap(),
      "ByteArray[<3>]"
    );
    // Round-trips through ByteArrayToString.
    assert_eq!(
      interpret(r#"ByteArrayToString[BaseDecode["aGVsbG8="]]"#).unwrap(),
      "hello"
    );
    // Encode/decode round trip.
    assert_eq!(
      interpret(r#"BaseEncode[BaseDecode["aGVsbG8="]]"#).unwrap(),
      "aGVsbG8="
    );
  }

  #[test]
  fn base_encode_non_byte_array_warns() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout("BaseEncode[{1, 2, 3}]").unwrap();
    assert_eq!(r.result, "BaseEncode[{1, 2, 3}]");
    assert!(
      r.warnings[0]
        .contains("BaseEncode::barray: {1, 2, 3} is not a ByteArray object.")
    );
  }

  #[test]
  fn base_decode_non_string_warns() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout("BaseDecode[123]").unwrap();
    assert_eq!(r.result, "BaseDecode[123]");
    assert!(
      r.warnings[0]
        .contains("BaseDecode::strx: String expected instead of 123.")
    );
  }
}

mod string_to_byte_array {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("StringToByteArray[\"Hello\"]").unwrap(),
      "ByteArray[<5>]"
    );
  }

  #[test]
  fn normal_form() {
    assert_eq!(
      interpret("Normal[StringToByteArray[\"Hello\"]]").unwrap(),
      "{72, 101, 108, 108, 111}"
    );
  }

  #[test]
  fn empty_string() {
    assert_eq!(
      interpret("StringToByteArray[\"\"]").unwrap(),
      "ByteArray[<0>]"
    );
  }

  #[test]
  fn roundtrip() {
    assert_eq!(
      interpret("ByteArrayToString[StringToByteArray[\"Hello World\"]]")
        .unwrap(),
      "Hello World"
    );
  }
}

mod byte_array_to_string {
  use super::*;

  #[test]
  fn from_list() {
    assert_eq!(
      interpret("ByteArrayToString[ByteArray[{72, 101, 108, 108, 111}]]")
        .unwrap(),
      "Hello"
    );
  }

  #[test]
  fn from_base64() {
    assert_eq!(
      interpret("ByteArrayToString[ByteArray[\"SGVsbG8=\"]]").unwrap(),
      "Hello"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("ByteArrayToString[x]").unwrap(),
      "ByteArrayToString[x]"
    );
  }

  // Hyperfactorial
  #[test]
  fn hyperfactorial_0() {
    assert_eq!(interpret("Hyperfactorial[0]").unwrap(), "1");
  }
  #[test]
  fn hyperfactorial_1() {
    assert_eq!(interpret("Hyperfactorial[1]").unwrap(), "1");
  }
  #[test]
  fn hyperfactorial_2() {
    assert_eq!(interpret("Hyperfactorial[2]").unwrap(), "4");
  }
  #[test]
  fn hyperfactorial_3() {
    assert_eq!(interpret("Hyperfactorial[3]").unwrap(), "108");
  }
  #[test]
  fn hyperfactorial_4() {
    assert_eq!(interpret("Hyperfactorial[4]").unwrap(), "27648");
  }
  #[test]
  fn hyperfactorial_5() {
    assert_eq!(interpret("Hyperfactorial[5]").unwrap(), "86400000");
  }
  #[test]
  fn hyperfactorial_10() {
    assert_eq!(
      interpret("Hyperfactorial[10]").unwrap(),
      "215779412229418562091680268288000000000000000"
    );
  }
  #[test]
  fn hyperfactorial_neg1() {
    assert_eq!(interpret("Hyperfactorial[-1]").unwrap(), "1");
  }
  #[test]
  fn hyperfactorial_neg2() {
    assert_eq!(interpret("Hyperfactorial[-2]").unwrap(), "-1");
  }
  #[test]
  fn hyperfactorial_neg3() {
    assert_eq!(interpret("Hyperfactorial[-3]").unwrap(), "-4");
  }
  #[test]
  fn hyperfactorial_neg4() {
    assert_eq!(interpret("Hyperfactorial[-4]").unwrap(), "108");
  }
  #[test]
  fn hyperfactorial_neg5() {
    assert_eq!(interpret("Hyperfactorial[-5]").unwrap(), "27648");
  }
  #[test]
  fn hyperfactorial_neg6() {
    assert_eq!(interpret("Hyperfactorial[-6]").unwrap(), "-86400000");
  }
  #[test]
  fn hyperfactorial_symbolic() {
    assert_eq!(interpret("Hyperfactorial[x]").unwrap(), "Hyperfactorial[x]");
  }
  #[test]
  fn hyperfactorial_listable() {
    assert_eq!(
      interpret("Hyperfactorial[{1, 2, 3}]").unwrap(),
      "{1, 4, 108}"
    );
  }

  #[test]
  fn hyperfactorial_series_order_0() {
    // Hyperfactorial[x] = 1 + O(x) at x = 0.
    assert_eq!(
      interpret("Series[Hyperfactorial[x], {x, 0, 0}]").unwrap(),
      "SeriesData[x, 0, {1}, 0, 1, 1]"
    );
  }

  #[test]
  fn hyperfactorial_series_order_1() {
    // Hyperfactorial[x] = 1 + (1 - Log[2*Pi])/2 * x + O(x^2).
    assert_eq!(
      interpret("Series[Hyperfactorial[x], {x, 0, 1}]").unwrap(),
      "SeriesData[x, 0, {1, (1 - Log[2*Pi])/2}, 0, 2, 1]"
    );
  }

  #[test]
  fn hyperfactorial_series_order_2() {
    // wolframscript:
    //   SeriesData[x, 0,
    //     {1, (1 - Log[2*Pi])/2,
    //      (5 - 4*EulerGamma - 2*Log[2*Pi] + Log[2*Pi]^2)/8}, 0, 3, 1]
    assert_eq!(
      interpret("Series[Hyperfactorial[x], {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, (1 - Log[2*Pi])/2, \
       (5 - 4*EulerGamma - 2*Log[2*Pi] + Log[2*Pi]^2)/8}, 0, 3, 1]"
    );
  }

  // DeBruijnSequence
  #[test]
  fn debruijn_binary_2() {
    assert_eq!(
      interpret("DeBruijnSequence[{0, 1}, 2]").unwrap(),
      "{0, 0, 1, 1}"
    );
  }
  #[test]
  fn debruijn_binary_3() {
    assert_eq!(
      interpret("DeBruijnSequence[{0, 1}, 3]").unwrap(),
      "{0, 0, 0, 1, 0, 1, 1, 1}"
    );
  }
  #[test]
  fn debruijn_binary_4() {
    assert_eq!(
      interpret("DeBruijnSequence[{0, 1}, 4]").unwrap(),
      "{0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1}"
    );
  }
  #[test]
  fn debruijn_symbolic_2() {
    assert_eq!(
      interpret("DeBruijnSequence[{a, b}, 2]").unwrap(),
      "{a, a, b, b}"
    );
  }
  #[test]
  fn debruijn_ternary_2() {
    assert_eq!(
      interpret("DeBruijnSequence[{0, 1, 2}, 2]").unwrap(),
      "{0, 0, 1, 0, 2, 1, 1, 2, 2}"
    );
  }
  #[test]
  fn debruijn_binary_1() {
    assert_eq!(interpret("DeBruijnSequence[{0, 1}, 1]").unwrap(), "{0, 1}");
  }
  #[test]
  fn debruijn_symbolic_3_1() {
    assert_eq!(
      interpret("DeBruijnSequence[{a, b, c}, 1]").unwrap(),
      "{a, b, c}"
    );
  }
  #[test]
  fn debruijn_single_element() {
    assert_eq!(interpret("DeBruijnSequence[{1}, 3]").unwrap(), "{1}");
  }
  #[test]
  fn debruijn_ternary_3() {
    assert_eq!(
      interpret("DeBruijnSequence[{0, 1, 2}, 3]").unwrap(),
      "{0, 0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 1, 2, 0, 2, 1, 0, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2}"
    );
  }
  #[test]
  fn debruijn_symbolic_3_2() {
    assert_eq!(
      interpret("DeBruijnSequence[{a, b, c}, 2]").unwrap(),
      "{a, a, b, a, c, b, b, c, c}"
    );
  }

  // BellY
  #[test]
  fn bell_y_0_0() {
    assert_eq!(interpret("BellY[0, 0, {}]").unwrap(), "1");
  }
  #[test]
  fn bell_y_1_1() {
    assert_eq!(interpret("BellY[1, 1, {x}]").unwrap(), "x");
  }
  #[test]
  fn bell_y_2_1() {
    assert_eq!(interpret("BellY[2, 1, {x1, x2}]").unwrap(), "x2");
  }
  #[test]
  fn bell_y_2_2() {
    assert_eq!(interpret("BellY[2, 2, {x1, x2}]").unwrap(), "x1^2");
  }
  #[test]
  fn bell_y_3_1() {
    assert_eq!(interpret("BellY[3, 1, {x1, x2, x3}]").unwrap(), "x3");
  }
  #[test]
  fn bell_y_3_2() {
    assert_eq!(interpret("BellY[3, 2, {x1, x2, x3}]").unwrap(), "3*x1*x2");
  }
  #[test]
  fn bell_y_3_3() {
    assert_eq!(interpret("BellY[3, 3, {x1, x2, x3}]").unwrap(), "x1^3");
  }
  #[test]
  fn bell_y_4_1() {
    assert_eq!(interpret("BellY[4, 1, {x1, x2, x3, x4}]").unwrap(), "x4");
  }
  #[test]
  fn bell_y_4_2() {
    assert_eq!(
      interpret("BellY[4, 2, {x1, x2, x3}]").unwrap(),
      "3*x2^2 + 4*x1*x3"
    );
  }
  #[test]
  fn bell_y_4_3() {
    assert_eq!(interpret("BellY[4, 3, {x1, x2}]").unwrap(), "6*x1^2*x2");
  }
  #[test]
  fn bell_y_4_4() {
    assert_eq!(interpret("BellY[4, 4, {x1}]").unwrap(), "x1^4");
  }
  #[test]
  fn bell_y_5_2() {
    assert_eq!(
      interpret("BellY[5, 2, {x1, x2, x3, x4}]").unwrap(),
      "10*x2*x3 + 5*x1*x4"
    );
  }
  #[test]
  fn bell_y_5_3() {
    assert_eq!(
      interpret("BellY[5, 3, {x1, x2, x3}]").unwrap(),
      "15*x1*x2^2 + 10*x1^2*x3"
    );
  }
  #[test]
  fn bell_y_6_3() {
    assert_eq!(
      interpret("BellY[6, 3, {x1, x2, x3, x4}]").unwrap(),
      "15*x2^3 + 60*x1*x2*x3 + 15*x1^2*x4"
    );
  }

  // FiniteGroupCount
  #[test]
  fn finite_group_count_1() {
    assert_eq!(interpret("FiniteGroupCount[1]").unwrap(), "1");
  }
  #[test]
  fn finite_group_count_4() {
    assert_eq!(interpret("FiniteGroupCount[4]").unwrap(), "2");
  }
  #[test]
  fn finite_group_count_8() {
    assert_eq!(interpret("FiniteGroupCount[8]").unwrap(), "5");
  }
  #[test]
  fn finite_group_count_16() {
    assert_eq!(interpret("FiniteGroupCount[16]").unwrap(), "14");
  }
  #[test]
  fn finite_group_count_32() {
    assert_eq!(interpret("FiniteGroupCount[32]").unwrap(), "51");
  }
  #[test]
  fn finite_group_count_64() {
    assert_eq!(interpret("FiniteGroupCount[64]").unwrap(), "267");
  }
  #[test]
  fn finite_group_count_128() {
    assert_eq!(interpret("FiniteGroupCount[128]").unwrap(), "2328");
  }
  #[test]
  fn finite_group_count_100() {
    assert_eq!(interpret("FiniteGroupCount[100]").unwrap(), "16");
  }
  #[test]
  fn finite_group_count_200() {
    assert_eq!(interpret("FiniteGroupCount[200]").unwrap(), "52");
  }
  #[test]
  fn finite_group_count_256() {
    assert_eq!(interpret("FiniteGroupCount[256]").unwrap(), "56092");
  }
  #[test]
  fn finite_group_count_512() {
    assert_eq!(interpret("FiniteGroupCount[512]").unwrap(), "10494213");
  }
  #[test]
  fn finite_group_count_1024() {
    assert_eq!(interpret("FiniteGroupCount[1024]").unwrap(), "49487365422");
  }
  #[test]
  fn finite_group_count_zero() {
    assert_eq!(interpret("FiniteGroupCount[0]").unwrap(), "0");
  }
  #[test]
  fn finite_group_count_negative() {
    assert_eq!(
      interpret("FiniteGroupCount[-1]").unwrap(),
      "FiniteGroupCount[-1]"
    );
  }
  #[test]
  fn finite_group_count_listable() {
    assert_eq!(
      interpret("FiniteGroupCount[{1, 2, 3}]").unwrap(),
      "{1, 1, 1}"
    );
  }
  #[test]
  fn finite_group_count_table() {
    assert_eq!(
      interpret("Table[FiniteGroupCount[n], {n, 1, 20}]").unwrap(),
      "{1, 1, 1, 2, 1, 2, 1, 5, 2, 2, 1, 5, 1, 2, 1, 14, 1, 5, 1, 5}"
    );
  }

  // FiniteAbelianGroupCount
  #[test]
  fn finite_abelian_group_count_1() {
    assert_eq!(interpret("FiniteAbelianGroupCount[1]").unwrap(), "1");
  }
  #[test]
  fn finite_abelian_group_count_4() {
    assert_eq!(interpret("FiniteAbelianGroupCount[4]").unwrap(), "2");
  }
  #[test]
  fn finite_abelian_group_count_8() {
    assert_eq!(interpret("FiniteAbelianGroupCount[8]").unwrap(), "3");
  }
  #[test]
  fn finite_abelian_group_count_16() {
    assert_eq!(interpret("FiniteAbelianGroupCount[16]").unwrap(), "5");
  }
  #[test]
  fn finite_abelian_group_count_1000() {
    assert_eq!(interpret("FiniteAbelianGroupCount[1000]").unwrap(), "9");
  }
  #[test]
  fn finite_abelian_group_count_10000() {
    assert_eq!(interpret("FiniteAbelianGroupCount[10000]").unwrap(), "25");
  }
  #[test]
  fn finite_abelian_group_count_zero() {
    assert_eq!(interpret("FiniteAbelianGroupCount[0]").unwrap(), "0");
  }
  #[test]
  fn finite_abelian_group_count_negative() {
    assert_eq!(
      interpret("FiniteAbelianGroupCount[-1]").unwrap(),
      "FiniteAbelianGroupCount[-1]"
    );
  }
  #[test]
  fn finite_abelian_group_count_listable() {
    assert_eq!(
      interpret("FiniteAbelianGroupCount[{1, 2, 3}]").unwrap(),
      "{1, 1, 1}"
    );
  }
  #[test]
  fn finite_abelian_group_count_table() {
    assert_eq!(
      interpret("Table[FiniteAbelianGroupCount[n], {n, 1, 30}]").unwrap(),
      "{1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 2, 1, 1, 1, 5, 1, 2, 1, 2, 1, 1, 1, 3, 2, 1, 3, 2, 1, 1}"
    );
  }
}

mod boolean_function {
  use super::*;

  // BooleanFunction[n, k][b1, …, bk] returns bit `v` of `n`, where `v` is the
  // arguments read as binary (first arg most significant, True = 1).
  #[test]
  fn integer_index_application() {
    assert_eq!(
      interpret("BooleanFunction[7, 2][True, False]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("BooleanFunction[7, 2][True, True]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("BooleanFunction[1, 2][False, False]").unwrap(),
      "True"
    );
    assert_eq!(interpret("BooleanFunction[2, 1][True]").unwrap(), "True");
    assert_eq!(interpret("BooleanFunction[2, 1][False]").unwrap(), "False");
    assert_eq!(
      interpret("BooleanFunction[1, 3][False, False, False]").unwrap(),
      "True"
    );
  }

  // Integer 1/0 are accepted as True/False.
  #[test]
  fn accepts_one_and_zero() {
    assert_eq!(interpret("BooleanFunction[7, 2][1, 0]").unwrap(), "True");
    assert_eq!(interpret("BooleanFunction[7, 2][1, 1]").unwrap(), "False");
  }

  // `n` beyond 2^(2^k) wraps via its bits; negative `n` is two's-complement
  // (BooleanFunction[-1, k] is the constant-True function).
  #[test]
  fn index_overflow_and_negative() {
    assert_eq!(
      interpret("BooleanFunction[16, 2][True, True]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("BooleanFunction[-1, 2][True, True]").unwrap(),
      "True"
    );
  }

  // Symbolic args, a wrong argument count, or the bare object stay unevaluated
  // (no spurious "not implemented" warning for the bare form).
  #[test]
  fn unevaluated_forms() {
    assert_eq!(
      interpret("BooleanFunction[7, 2][a, b]").unwrap(),
      "BooleanFunction[7, 2][a, b]"
    );
    assert_eq!(
      interpret("BooleanFunction[7, 2][True]").unwrap(),
      "BooleanFunction[7, 2][True]"
    );
    assert_eq!(
      interpret("BooleanFunction[7, 2]").unwrap(),
      "BooleanFunction[7, 2]"
    );
  }
}

mod boolean_satisfiability {
  use super::*;

  #[test]
  fn satisfiable_q() {
    assert_eq!(interpret("SatisfiableQ[a && b]").unwrap(), "True");
    assert_eq!(interpret("SatisfiableQ[a && ! a]").unwrap(), "False");
    assert_eq!(
      interpret("SatisfiableQ[(a || b) && (! a || ! b)]").unwrap(),
      "True"
    );
    assert_eq!(interpret("SatisfiableQ[True]").unwrap(), "True");
    assert_eq!(interpret("SatisfiableQ[False]").unwrap(), "False");
    assert_eq!(interpret("SatisfiableQ[Nand[a, a]]").unwrap(), "True");
    assert_eq!(interpret("SatisfiableQ[a && ! b, {a, b}]").unwrap(), "True");
    // Opaque subexpressions act as variables
    assert_eq!(interpret("SatisfiableQ[a && f[b]]").unwrap(), "True");
  }

  #[test]
  fn satisfiable_q_boolv_message() {
    // A variable list that leaves the expression non-Boolean emits boolv
    // (showing the first failing assignment, all-True first)
    assert_eq!(
      interpret("SatisfiableQ[x && y, {x}]").unwrap(),
      "SatisfiableQ[x && y, {x}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "SatisfiableQ::boolv: x && y is not Boolean valued at {True}."
      )),
      "expected boolv message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("SatisfiableQ[x && y && z, {x, y}]").unwrap(),
      "SatisfiableQ[x && y && z, {x, y}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "SatisfiableQ::boolv: x && y && z is not Boolean valued at {True, True}."
      )),
      "expected boolv message, got {:?}",
      msgs
    );
  }

  #[test]
  fn satisfiability_count() {
    assert_eq!(interpret("SatisfiabilityCount[a && b]").unwrap(), "1");
    assert_eq!(interpret("SatisfiabilityCount[a || b]").unwrap(), "3");
    assert_eq!(interpret("SatisfiabilityCount[Xor[a, b, c]]").unwrap(), "4");
    assert_eq!(interpret("SatisfiabilityCount[True]").unwrap(), "1");
    assert_eq!(interpret("SatisfiabilityCount[a && ! a]").unwrap(), "0");
    assert_eq!(
      interpret("SatisfiabilityCount[Implies[a, b]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("SatisfiabilityCount[Equivalent[a, b]]").unwrap(),
      "2"
    );
    // Extra variables multiply the count
    assert_eq!(
      interpret("SatisfiabilityCount[a || b, {a, b, c}]").unwrap(),
      "6"
    );
    assert_eq!(interpret("SatisfiabilityCount[a && f[b]]").unwrap(), "1");
    // Incomplete variable list: silently unevaluated (no boolv here)
    assert_eq!(
      interpret("SatisfiabilityCount[x && y, {x}]").unwrap(),
      "SatisfiabilityCount[x && y, {x}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.is_empty(), "expected no messages, got {:?}", msgs);
  }

  #[test]
  fn boolean_variables() {
    // Canonically sorted, deduplicated
    assert_eq!(
      interpret("BooleanVariables[a && b || ! c]").unwrap(),
      "{a, b, c}"
    );
    assert_eq!(interpret("BooleanVariables[c && a]").unwrap(), "{a, c}");
    assert_eq!(interpret("BooleanVariables[True]").unwrap(), "{}");
    assert_eq!(interpret("BooleanVariables[x]").unwrap(), "{x}");
    // Lists are traversed
    assert_eq!(
      interpret("BooleanVariables[{a && q, z || b}]").unwrap(),
      "{a, b, q, z}"
    );
    // Opaque subexpressions count as variables
    assert_eq!(
      interpret("BooleanVariables[a && f[b]]").unwrap(),
      "{a, f[b]}"
    );
    assert_eq!(
      interpret("BooleanVariables[Nand[p, Nor[q, r]]]").unwrap(),
      "{p, q, r}"
    );
  }

  #[test]
  fn majority() {
    assert_eq!(interpret("Majority[True, False]").unwrap(), "False");
    assert_eq!(interpret("Majority[True, True, False]").unwrap(), "True");
    assert_eq!(interpret("Majority[]").unwrap(), "False");
    // Decided regardless of the unknown
    assert_eq!(interpret("Majority[True, True, x]").unwrap(), "True");
    // True/False pairs cancel
    assert_eq!(interpret("Majority[True, False, x]").unwrap(), "x");
    // Two arguments need both
    assert_eq!(interpret("Majority[a, b]").unwrap(), "a && b");
    // Orderless: symbolic forms sort canonically
    assert_eq!(
      interpret("Majority[x, x, False]").unwrap(),
      "Majority[False, x, x]"
    );
    assert_eq!(interpret("Majority[c, a, b]").unwrap(), "Majority[a, b, c]");
    // Satisfiability integration: majority of 3 has 4 satisfying rows
    assert_eq!(
      interpret("SatisfiabilityCount[Majority[a, b, c]]").unwrap(),
      "4"
    );
  }
}

mod named_logical_operators {
  use super::*;

  #[test]
  fn parse_and_evaluate() {
    // Regression: these previously parsed as products like a*Implies*b
    assert_eq!(interpret(r"a \[Implies] b").unwrap(), "Implies[a, b]");
    assert_eq!(
      interpret(r"{True \[And] False, True \[Or] False, True \[Implies] False, \[Not] True}")
        .unwrap(),
      "{False, True, False, False}"
    );
    assert_eq!(interpret(r"TautologyQ[a \[Or] \[Not] a]").unwrap(), "True");
    assert_eq!(
      interpret(r"SatisfiabilityCount[a \[Xor] b \[Xor] c]").unwrap(),
      "4"
    );
  }

  #[test]
  fn precedences() {
    // Not > And/Nand > Xor > Or/Nor > Equivalent > Implies
    assert_eq!(
      interpret(r"FullForm[a \[And] b \[Or] c]").unwrap(),
      "FullForm[(a && b) || c]"
    );
    assert_eq!(
      interpret(r"FullForm[a \[Xor] b \[And] c]").unwrap(),
      "FullForm[Xor[a, b && c]]"
    );
    assert_eq!(
      interpret(r"FullForm[a \[Or] b \[Xor] c]").unwrap(),
      "FullForm[a || Xor[b, c]]"
    );
    assert_eq!(
      interpret(r"FullForm[a \[Implies] b \[Or] c]").unwrap(),
      "FullForm[Implies[a, b || c]]"
    );
    assert_eq!(
      interpret(r"FullForm[a \[Equivalent] b \[Implies] c]").unwrap(),
      "FullForm[Implies[Equivalent[a, b], c]]"
    );
    assert_eq!(
      interpret(r"FullForm[a \[Nand] b \[And] c]").unwrap(),
      "FullForm[Nand[a, b] && c]"
    );
    assert_eq!(
      interpret(r"FullForm[\[Not] a \[And] b]").unwrap(),
      "FullForm[ !a && b]"
    );
  }

  #[test]
  fn implies_is_right_associative() {
    assert_eq!(
      interpret(r"FullForm[a \[Implies] b \[Implies] c]").unwrap(),
      "FullForm[Implies[a, Implies[b, c]]]"
    );
  }

  #[test]
  fn flat_chains_collapse() {
    assert_eq!(
      interpret(r"FullForm[a \[Xor] b \[Xor] c]").unwrap(),
      "FullForm[Xor[a, b, c]]"
    );
    assert_eq!(
      interpret(r"FullForm[a \[Nand] b \[Nand] c]").unwrap(),
      "FullForm[Nand[a, b, c]]"
    );
    assert_eq!(
      interpret(r"a \[Equivalent] b \[Equivalent] c").unwrap(),
      "a \u{29e6} b \u{29e6} c"
    );
  }

  #[test]
  fn equivalent_keeps_infix_in_full_form() {
    assert_eq!(
      interpret(r"FullForm[a \[Equivalent] b]").unwrap(),
      "FullForm[Equivalent[a, b]]"
    );
    assert_eq!(
      interpret("InputForm[Equivalent[a, b]]").unwrap(),
      "InputForm[a \u{29e6} b]"
    );
  }
}

mod implies_simplification {
  use super::*;

  #[test]
  fn symbolic_antecedent_true_consequent() {
    // Implies[a, True] -> True (anything implies a truth).
    assert_eq!(interpret("Implies[a, True]").unwrap(), "True");
    assert_eq!(interpret("Implies[x > 0, True]").unwrap(), "True");
  }

  #[test]
  fn symbolic_antecedent_false_consequent() {
    // Implies[a, False] -> Not[a].
    assert_eq!(interpret("Implies[a, False]").unwrap(), " !a");
  }

  #[test]
  fn identical_antecedent_and_consequent() {
    // Implies[a, a] -> True (p -> p is a tautology).
    assert_eq!(interpret("Implies[a, a]").unwrap(), "True");
    assert_eq!(interpret("Implies[a && b, a && b]").unwrap(), "True");
  }

  #[test]
  fn distinct_symbolic_stays_unevaluated() {
    assert_eq!(interpret("Implies[a, b]").unwrap(), "Implies[a, b]");
  }

  #[test]
  fn literal_antecedent_rules_unchanged() {
    assert_eq!(interpret("Implies[True, b]").unwrap(), "b");
    assert_eq!(interpret("Implies[False, b]").unwrap(), "True");
    assert_eq!(interpret("Implies[True, False]").unwrap(), "False");
  }
}

mod rule_chains {
  use super::*;

  #[test]
  fn rules_chain_right_associatively() {
    // Regression: a -> b -> c inside lists and call arguments previously
    // failed to parse
    assert_eq!(interpret("{a -> b -> c}").unwrap(), "{a -> b -> c}");
    assert_eq!(interpret("f[a -> b -> c]").unwrap(), "f[a -> b -> c]");
    assert_eq!(
      interpret("FullForm[{a -> b -> c}]").unwrap(),
      "FullForm[{a -> b -> c}]"
    );
  }

  #[test]
  fn mixed_arrows_classify_by_outer_operator() {
    assert_eq!(
      interpret("FullForm[a -> b :> c]").unwrap(),
      "FullForm[a -> b :> c]"
    );
    assert_eq!(
      interpret("FullForm[a :> b -> c]").unwrap(),
      "FullForm[a :> b -> c]"
    );
    // The AST distinguishes the arrows: ReplaceAll exercises the
    // delayed/immediate split through a chained parse
    assert_eq!(interpret("x /. x -> (a :> b)").unwrap(), "a :> b");
  }
}

mod fold_while {
  use super::*;

  #[test]
  fn fold_while_basic() {
    // Folds until the test fails; the failing value is returned
    assert_eq!(
      interpret("FoldWhile[Plus, {1, 2, 3, 4, 5, 6}, # < 6 &]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("FoldWhile[Times, 1, {2, 3, 4, 5}, # < 10 &]").unwrap(),
      "24"
    );
    // Never-failing test folds the whole list
    assert_eq!(
      interpret("FoldWhile[Plus, {1, 2, 3}, # < 100 &]").unwrap(),
      "6"
    );
    // Test failing on the initial value returns it unchanged
    assert_eq!(interpret("FoldWhile[Plus, {5}, # < 3 &]").unwrap(), "5");
    assert_eq!(interpret("FoldWhile[Plus, 10, {}, # < 3 &]").unwrap(), "10");
  }

  #[test]
  fn fold_while_list_three_arg_form() {
    // Regression: the 3-argument form previously warned
    // FoldWhileList::argrx and stayed unevaluated
    assert_eq!(
      interpret("FoldWhileList[Plus, {1, 2, 3, 4, 5, 6}, # < 6 &]").unwrap(),
      "{1, 3, 6}"
    );
    assert_eq!(
      interpret("FoldWhileList[Times, 1, {2, 3, 4, 5}, # < 10 &]").unwrap(),
      "{1, 2, 6, 24}"
    );
    assert_eq!(
      interpret("FoldWhileList[Plus, {5}, # < 3 &]").unwrap(),
      "{5}"
    );
  }
}

mod numerator_denominator {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("NumeratorDenominator[3/7]").unwrap(), "{3, 7}");
    assert_eq!(interpret("NumeratorDenominator[5]").unwrap(), "{5, 1}");
    assert_eq!(interpret("NumeratorDenominator[2.5]").unwrap(), "{2.5, 1}");
    assert_eq!(interpret("NumeratorDenominator[x/y]").unwrap(), "{x, y}");
    assert_eq!(
      interpret("NumeratorDenominator[(a + b)/c^2]").unwrap(),
      "{a + b, c^2}"
    );
  }
}

mod bit_get {
  use super::*;

  #[test]
  fn positive_integers() {
    // 11 = 1011b
    assert_eq!(interpret("BitGet[11, 0]").unwrap(), "1");
    assert_eq!(interpret("BitGet[11, 1]").unwrap(), "1");
    assert_eq!(interpret("BitGet[11, 2]").unwrap(), "0");
    assert_eq!(interpret("BitGet[11, 3]").unwrap(), "1");
    assert_eq!(interpret("BitGet[11, 4]").unwrap(), "0");
    assert_eq!(interpret("BitGet[2^100 + 8, 100]").unwrap(), "1");
  }

  #[test]
  fn negative_uses_twos_complement() {
    assert_eq!(interpret("BitGet[-1, 10]").unwrap(), "1");
    // -6 = ...11111010
    assert_eq!(interpret("BitGet[-6, 0]").unwrap(), "0");
    assert_eq!(interpret("BitGet[-6, 1]").unwrap(), "1");
    assert_eq!(interpret("BitGet[-6, 2]").unwrap(), "0");
    assert_eq!(interpret("BitGet[-6, 100]").unwrap(), "1");
  }

  #[test]
  fn invalid_arguments_stay_unevaluated() {
    assert_eq!(interpret("BitGet[10, -1]").unwrap(), "BitGet[10, -1]");
    assert_eq!(interpret("BitGet[2.5, 1]").unwrap(), "BitGet[2.5, 1]");
    assert_eq!(interpret("BitGet[x, 2]").unwrap(), "BitGet[x, 2]");
  }

  // BitGet is Listable and threads over either or both arguments.
  #[test]
  fn threads_over_position_list() {
    // 10 = 1010b
    assert_eq!(
      interpret("BitGet[10, {0, 1, 2, 3}]").unwrap(),
      "{0, 1, 0, 1}"
    );
  }

  #[test]
  fn threads_over_number_list() {
    assert_eq!(interpret("BitGet[{10, 20}, 0]").unwrap(), "{0, 0}");
  }

  #[test]
  fn threads_over_both_lists_elementwise() {
    assert_eq!(interpret("BitGet[{4, 8}, {2, 3}]").unwrap(), "{1, 1}");
  }

  #[test]
  fn threads_with_negative_two_complement() {
    assert_eq!(interpret("BitGet[-5, {0, 1, 2}]").unwrap(), "{1, 1, 0}");
  }

  #[test]
  fn is_listable() {
    assert_eq!(
      interpret("Attributes[BitGet]").unwrap(),
      "{Listable, Protected}"
    );
  }
}

mod golden_angle {
  use super::*;

  #[test]
  fn symbolic_and_machine() {
    assert_eq!(interpret("GoldenAngle").unwrap(), "GoldenAngle");
    assert_eq!(interpret("N[GoldenAngle]").unwrap(), "2.3999632297286535");
    // Note: N[GoldenAngle, p] for arbitrary p shares Woxi's known
    // last-digits display drift for composite constants (Pi products).
  }
}

mod string_extract {
  use super::*;

  #[test]
  fn whitespace_fields() {
    assert_eq!(
      interpret(r#"StringExtract["aa bb cc dd", 2]"#).unwrap(),
      "bb"
    );
    assert_eq!(
      interpret(r#"StringExtract["aa bb cc dd", {1, 3}]"#).unwrap(),
      "{aa, cc}"
    );
    assert_eq!(interpret(r#"StringExtract["aa bb cc", -1]"#).unwrap(), "cc");
    assert_eq!(
      interpret(r#"StringExtract["one  two   three", {2, -1}]"#).unwrap(),
      "{two, three}"
    );
  }

  #[test]
  fn delimiter_rules_drill_down() {
    assert_eq!(
      interpret(r#"StringExtract["a-b-c,d-e", "," -> 2]"#).unwrap(),
      "d-e"
    );
    assert_eq!(
      interpret(r#"StringExtract["a-b-c,d-e", "," -> 1, "-" -> 3]"#).unwrap(),
      "c"
    );
    assert_eq!(
      interpret(r#"StringExtract["a-b-c,d-e", "," -> {1, 2}, "-" -> 1]"#)
        .unwrap(),
      "{a, d}"
    );
  }

  #[test]
  fn out_of_range_gives_missing() {
    assert_eq!(
      interpret(r#"StringExtract["aa bb cc", 5]"#).unwrap(),
      "Missing[PartAbsent, 5]"
    );
  }

  #[test]
  fn span_of_blocks() {
    // A span `i ;; j` selects a contiguous range of blocks (a list).
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc aa d", 2 ;; 4]"#).unwrap(),
      "{bbb, cccc, aa}"
    );
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc aa d", 2 ;; -1]"#).unwrap(),
      "{bbb, cccc, aa, d}"
    );
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc", -2 ;; -1]"#).unwrap(),
      "{bbb, cccc}"
    );
  }

  #[test]
  fn span_endpoints_clamp() {
    // Both endpoints clamp into [1, len]; a reversed range yields {}.
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc", 2 ;; 5]"#).unwrap(),
      "{bbb, cccc}"
    );
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc", 5 ;; 7]"#).unwrap(),
      "{cccc}"
    );
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc", 0 ;; 2]"#).unwrap(),
      "{a, bbb}"
    );
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc", 3 ;; 2]"#).unwrap(),
      "{}"
    );
  }

  #[test]
  fn open_spans() {
    // `;; j` is `1 ;; j`; `i ;;` runs to the last block.
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc", ;; 2]"#).unwrap(),
      "{a, bbb}"
    );
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc", 2 ;;]"#).unwrap(),
      "{bbb, cccc}"
    );
  }

  #[test]
  fn stepped_span_rejected() {
    // A three-part (stepped) span is not a valid extraction spec; it stays
    // unevaluated and emits ::patt, matching Wolfram.
    assert_eq!(
      interpret(r#"StringExtract["a bbb cccc", 1 ;; 3 ;; 2]"#).unwrap(),
      "StringExtract[a bbb cccc, Span[1, 3, 2]]"
    );
  }

  #[test]
  fn list_of_strings_maps() {
    assert_eq!(
      interpret(r#"StringExtract[{"a b", "c d"}, 2]"#).unwrap(),
      "{b, d}"
    );
  }

  #[test]
  fn non_string_emits_strse() {
    assert_eq!(
      interpret("StringExtract[42, 1]").unwrap(),
      "StringExtract[42, 1]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "StringExtract::strse: A string or list of strings is expected at position 1 in StringExtract[42, 1]."
      )),
      "expected strse message, got {:?}",
      msgs
    );
  }
}

mod boolean_minterms {
  use super::*;

  #[test]
  fn boolean_rows() {
    assert_eq!(
      interpret("BooleanMinterms[{{True, False}, {False, True}}, {a, b}]")
        .unwrap(),
      "(a &&  !b) || ( !a && b)"
    );
    // Rows are ordered by descending minterm index regardless of input order
    assert_eq!(
      interpret("BooleanMinterms[{{False, True}, {True, False}}, {a, b}]")
        .unwrap(),
      "(a &&  !b) || ( !a && b)"
    );
    assert_eq!(
      interpret("BooleanMinterms[{{True, True}}, {a, b}]").unwrap(),
      "a && b"
    );
    assert_eq!(interpret("BooleanMinterms[{{False}}, {a}]").unwrap(), " !a");
    // Short rows cover a prefix of the variables
    assert_eq!(interpret("BooleanMinterms[{{True}}, {a, b}]").unwrap(), "a");
  }

  #[test]
  fn integer_indices() {
    assert_eq!(
      interpret("BooleanMinterms[{1, 2, 7}, {a, b, c}]").unwrap(),
      "(a && b && c) || ( !a && b &&  !c) || ( !a &&  !b && c)"
    );
    // Same set in any order: sorted by descending index
    assert_eq!(
      interpret("BooleanMinterms[{2, 7, 1}, {a, b, c}]").unwrap(),
      "(a && b && c) || ( !a && b &&  !c) || ( !a &&  !b && c)"
    );
    // Indices wrap mod 2^n and duplicates collapse
    assert_eq!(
      interpret("BooleanMinterms[{9}, {a, b}]").unwrap(),
      " !a && b"
    );
    assert_eq!(
      interpret("BooleanMinterms[{3, 3}, {a, b}]").unwrap(),
      "a && b"
    );
    assert_eq!(
      interpret("BooleanMinterms[{0}, {a, b}]").unwrap(),
      " !a &&  !b"
    );
    assert_eq!(
      interpret("BooleanMinterms[{5}, {p, q, r}]").unwrap(),
      "p &&  !q && r"
    );
  }

  #[test]
  fn complete_and_empty_specifications() {
    assert_eq!(
      interpret("BooleanMinterms[{0, 1, 2, 3}, {a, b}]").unwrap(),
      "True"
    );
    // Partial rows covering everything also give True
    assert_eq!(
      interpret("BooleanMinterms[{{True}, {False}}, {a, b}]").unwrap(),
      "True"
    );
    assert_eq!(interpret("BooleanMinterms[{}, {a, b}]").unwrap(), "False");
  }

  #[test]
  fn invalid_specifications_emit_bspec() {
    assert_eq!(
      interpret("BooleanMinterms[a || b, {a, b}]").unwrap(),
      "BooleanMinterms[a || b, {a, b}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "BooleanMinterms::bspec: BooleanMinterms[a || b, {a, b}] is not a valid BooleanMinterms specification."
      )),
      "expected bspec message, got {:?}",
      msgs
    );
    // Mixed row lengths are invalid
    assert_eq!(
      interpret(
        "BooleanMinterms[{{True}, {False, True}, {False, False}}, {a, b}]"
      )
      .unwrap(),
      "BooleanMinterms[{{True}, {False, True}, {False, False}}, {a, b}]"
    );
    assert_eq!(
      interpret("BooleanMinterms[{1, 2}, x]").unwrap(),
      "BooleanMinterms[{1, 2}, x]"
    );
  }

  #[test]
  fn integrates_with_satisfiability() {
    assert_eq!(
      interpret("SatisfiabilityCount[BooleanMinterms[{1, 2, 7}, {a, b, c}]]")
        .unwrap(),
      "3"
    );
  }
}

mod cross_dot_shape_messages {
  use super::*;

  #[test]
  fn cross_wrong_lengths_emit_nonn1() {
    // Regression: wrong-length vectors silently returned unevaluated
    assert_eq!(
      interpret("Cross[{1, 2}, {3, 4}]").unwrap(),
      "Cross[{1, 2}, {3, 4}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Cross::nonn1: The arguments are expected to be vectors of equal length, and the number of arguments is expected to be one less than their length."
      )),
      "expected nonn1 message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("Cross[{1, 2, 3}, {4, 5}]").unwrap(),
      "Cross[{1, 2, 3}, {4, 5}]"
    );
    assert_eq!(interpret("Cross[{1, 2, 3}]").unwrap(), "Cross[{1, 2, 3}]");
    // Symbolic arguments stay silent
    assert_eq!(interpret("Cross[x, y]").unwrap(), "Cross[x, y]");
    assert_eq!(interpret("Cross[x]").unwrap(), "Cross[x]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.is_empty(),
      "symbolic Cross must stay silent, got {:?}",
      msgs
    );
    // Valid forms still work
    assert_eq!(
      interpret("Cross[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "{-3, 6, -3}"
    );
    assert_eq!(interpret("Cross[{1, 2}]").unwrap(), "{-2, 1}");
  }

  // Generalized cross product: k = n - 1 vectors of length n.
  #[test]
  fn cross_generalized_n_dimensional() {
    assert_eq!(
      interpret("Cross[{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}]").unwrap(),
      "{0, 0, 0, 1}"
    );
    assert_eq!(
      interpret(
        "Cross[{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}]"
      )
      .unwrap(),
      "{0, 0, 0, 0, 1}"
    );
    // Symbolic 4D cross matches the determinant expansion.
    assert_eq!(
      interpret("Cross[{a, b, c, d}, {e, f, g, h}, {i, j, k, l}]").unwrap(),
      "{d*g*j - c*h*j - d*f*k + b*h*k + c*f*l - b*g*l, -(d*g*i) + c*h*i + d*e*k - a*h*k - c*e*l + a*g*l, d*f*i - b*h*i - d*e*j + a*h*j + b*e*l - a*f*l, -(c*f*i) + b*g*i + c*e*j - a*g*j - b*e*k + a*f*k}"
    );
  }

  // Wrong vector count / length for the N-dim form emits nonn1.
  #[test]
  fn cross_generalized_wrong_count_emits_nonn1() {
    assert_eq!(
      interpret("Cross[{1, 0, 0}, {0, 1, 0}, {0, 0, 1}]").unwrap(),
      "Cross[{1, 0, 0}, {0, 1, 0}, {0, 0, 1}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains("Cross::nonn1")),
      "expected nonn1, got {:?}",
      msgs
    );
  }

  #[test]
  fn dot_incompatible_shapes_emit_dotsh() {
    // Regression: shape mismatches raised hard errors
    assert_eq!(
      interpret("{1, 2} . {3, 4, 5}").unwrap(),
      "{1, 2} . {3, 4, 5}"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Dot::dotsh: Tensors {1, 2} and {3, 4, 5} have incompatible shapes."
      )),
      "expected dotsh message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("{{1, 2}} . {1, 2, 3}").unwrap(),
      "{{1, 2}} . {1, 2, 3}"
    );
    assert_eq!(
      interpret("{1, 2} . {{1, 2}, {3, 4}, {5, 6}}").unwrap(),
      "{1, 2} . {{1, 2}, {3, 4}, {5, 6}}"
    );
    assert_eq!(
      interpret("{{1, 2}} . {{1, 2}, {3, 4}, {5, 6}}").unwrap(),
      "{{1, 2}} . {{1, 2}, {3, 4}, {5, 6}}"
    );
    // Valid forms still work
    assert_eq!(interpret("{1, 2} . {3, 4}").unwrap(), "11");
    assert_eq!(interpret("{{1, 2}, {3, 4}} . {5, 6}").unwrap(), "{17, 39}");
  }
}

mod map_thread_mptc_message {
  use super::*;

  #[test]
  fn mptc_names_the_first_adjacent_mismatched_pair() {
    // Regression: the message lacked the positions and dimensions
    assert_eq!(
      interpret("MapThread[f, {{a, b}, {c, d, e}}]").unwrap(),
      "MapThread[f, {{a, b}, {c, d, e}}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "MapThread::mptc: Incompatible dimensions of objects at positions {2, 1} and {2, 2} of MapThread[f, {{a, b}, {c, d, e}}]; dimensions are 2 and 3."
      )),
      "expected detailed mptc message, got {:?}",
      msgs
    );
    // Adjacent comparison: the first two lists match, the third differs
    assert_eq!(
      interpret("MapThread[f, {{a, b}, {c, d}, {e}}]").unwrap(),
      "MapThread[f, {{a, b}, {c, d}, {e}}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "MapThread::mptc: Incompatible dimensions of objects at positions {2, 2} and {2, 3} of MapThread[f, {{a, b}, {c, d}, {e}}]; dimensions are 2 and 1."
      )),
      "expected adjacent-pair mptc message, got {:?}",
      msgs
    );
  }

  #[test]
  fn mptc_reports_level_n_mismatches() {
    assert_eq!(
      interpret("MapThread[f, {{{a, b}}, {{c, d}, {e, f}}}, 2]").unwrap(),
      "MapThread[f, {{{a, b}}, {{c, d}, {e, f}}}, 2]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "MapThread::mptc: Incompatible dimensions of objects at positions {2, 1} and {2, 2} of MapThread[f, {{{a, b}}, {{c, d}, {e, f}}}, 2]; dimensions are 1 and 2."
      )),
      "expected level-2 mptc message, got {:?}",
      msgs
    );
    // Valid forms still work
    assert_eq!(
      interpret("MapThread[f, {{a, b}, {c, d}}]").unwrap(),
      "{f[a, c], f[b, d]}"
    );
  }
}
