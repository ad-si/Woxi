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
  }

  #[test]
  fn n_too_small() {
    assert_eq!(interpret("PrimitiveRoot[1]").unwrap(), "PrimitiveRoot[1]");
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
    assert_eq!(
      interpret("FiniteGroupCount[0]").unwrap(),
      "FiniteGroupCount[0]"
    );
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
    assert_eq!(
      interpret("FiniteAbelianGroupCount[0]").unwrap(),
      "FiniteAbelianGroupCount[0]"
    );
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
