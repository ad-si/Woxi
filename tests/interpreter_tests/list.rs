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

  #[test]
  fn rational_list_plus_scalar() {
    // Rational numbers in lists must be fully evaluated when threaded
    assert_eq!(interpret("{1/2, 1/3} + 1").unwrap(), "{3/2, 4/3}");
  }

  #[test]
  fn rational_list_plus_rational_list() {
    assert_eq!(interpret("{1/2} + {1/3}").unwrap(), "{5/6}");
  }

  #[test]
  fn rational_list_times_scalar() {
    assert_eq!(interpret("{1/2, 1/3} * 2").unwrap(), "{1, 2/3}");
  }

  #[test]
  fn rational_list_minus_scalar() {
    assert_eq!(interpret("{1/2, 1/3} - 1").unwrap(), "{-1/2, -2/3}");
  }

  #[test]
  fn unequal_length_lists_plus_unevaluated() {
    // Lists of unequal length emit Thread::tdlen warning and return
    // the expression unevaluated (matches wolframscript).
    assert_eq!(
      interpret("{1, 2} + {4, 5, 6}").unwrap(),
      "{1, 2} + {4, 5, 6}"
    );
  }

  #[test]
  fn unequal_length_lists_times_unevaluated() {
    assert_eq!(interpret("{1, 2} * {4, 5, 6}").unwrap(), "{1, 2}*{4, 5, 6}");
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
  fn table_pi_bounds_half_pi_step() {
    // Table[i, {i, Pi, 2 Pi, Pi/2}] — three elements at Pi, 3 Pi/2, 2 Pi.
    assert_eq!(
      interpret("Table[i, {i, Pi, 2 Pi, Pi / 2}]").unwrap(),
      "{Pi, (3*Pi)/2, 2*Pi}"
    );
  }

  #[test]
  fn table_symbolic_pi_step() {
    assert_eq!(
      interpret("Table[θ, {θ, 0, 2 Pi - Pi/4, Pi/4}]").unwrap(),
      "{0, Pi/4, Pi/2, (3*Pi)/4, Pi, (5*Pi)/4, (3*Pi)/2, (7*Pi)/4}"
    );
  }

  #[test]
  fn table_fully_symbolic_bounds_and_step() {
    // Regression: Table[x, {x, a, a + 5 n, n}] used to error because the
    // iterator bounds weren't numeric. (max - min) / step = 5, so we
    // iterate 6 times producing a, a+n, a+2n, …, a+5n.
    assert_eq!(
      interpret("Table[x, {x, a, a + 5 n, n}]").unwrap(),
      "{a, a + n, a + 2*n, a + 3*n, a + 4*n, a + 5*n}"
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

  #[test]
  fn union_same_test_two_lists_parity() {
    assert_eq!(
      interpret(
        "Union[{1, 2, 3}, {2, 3, 4}, SameTest -> (Mod[#1 - #2, 2] == 0 &)]"
      )
      .unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn union_same_test_single_list_parity() {
    assert_eq!(
      interpret("Union[{3, 1, 2, 4}, SameTest -> (Mod[#1 - #2, 2] == 0 &)]")
        .unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn union_same_test_picks_smallest_representative() {
    assert_eq!(
      interpret("Union[{3, 5, 2, 4}, SameTest -> (Mod[#1 - #2, 2] == 0 &)]")
        .unwrap(),
      "{2, 3}"
    );
  }

  // The candidate goes on the *left* of the SameTest. With `Greater`,
  // every element after the smallest gets absorbed (Greater[k, 1] is
  // True for k > 1), collapsing the result to the singleton smallest.
  #[test]
  fn union_same_test_greater_collapses_to_smallest() {
    assert_eq!(
      interpret("Union[{1, 2, 3, 4}, SameTest -> Greater]").unwrap(),
      "{1}"
    );
  }

  // `Less` is the dual: `Less[k, 1]` is False for every k ≥ 1, so no
  // element is absorbed and the sorted concatenation survives — the
  // mathics doctest case from test_cases.rs case 2518.
  #[test]
  fn union_same_test_less_keeps_all_elements_sorted() {
    assert_eq!(
      interpret("Union[{1, 2, 3}, {2, 3, 4}, SameTest -> Less]").unwrap(),
      "{1, 2, 2, 3, 3, 4}"
    );
  }

  // SameTest applied to a structural test — key off the last element of
  // each pair. The first occurrence wins, so `{a, 1}` keeps its slot
  // while `{c, 1}` (same last-component) is absorbed.
  #[test]
  fn union_same_test_structural_keeps_first_representative() {
    assert_eq!(
      interpret(
        "Union[{{a, 1}, {b, 2}}, {{c, 1}, {d, 3}}, \
         SameTest -> (SameQ[Last[#1], Last[#2]] &)]"
      )
      .unwrap(),
      "{{a, 1}, {b, 2}, {d, 3}}"
    );
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
  fn inner_default_combiner() {
    assert_eq!(
      interpret("Inner[f, {a, b, c}, {x, y, z}]").unwrap(),
      "f[a, x] + f[b, y] + f[c, z]"
    );
  }

  #[test]
  fn inner_times_plus() {
    assert_eq!(
      interpret("Inner[Times, {a, b, c}, {x, y, z}, Plus]").unwrap(),
      "a*x + b*y + c*z"
    );
  }

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

  #[test]
  fn non_list_head() {
    // Outer should thread through non-List heads, preserving them.
    assert_eq!(
      interpret("Outer[g, f[a, b], f[c, d]]").unwrap(),
      "f[f[g[a, c], g[a, d]], f[g[b, c], g[b, d]]]"
    );
  }

  #[test]
  fn non_list_head_asymmetric() {
    assert_eq!(
      interpret("Outer[g, f[a, b, c], f[d, e]]").unwrap(),
      "f[f[g[a, d], g[a, e]], f[g[b, d], g[b, e]], f[g[c, d], g[c, e]]]"
    );
  }

  #[test]
  fn non_list_head_single_element() {
    assert_eq!(
      interpret("Outer[g, f[a], f[c, d]]").unwrap(),
      "f[f[g[a, c], g[a, d]]]"
    );
  }

  #[test]
  fn mixed_heads_return_unevaluated() {
    // Different heads should return an error or unevaluated form.
    // Mathematica gives Outer::heads error and returns unevaluated.
    assert_eq!(
      interpret("Outer[g, f[a, b], {c, d}]").unwrap(),
      "Outer[g, f[a, b], {c, d}]"
    );
  }

  #[test]
  fn level_spec_single() {
    // Outer[f, nested, flat, 1] — descend 1 level into the nested list,
    // treating sublists as atomic elements.
    assert_eq!(
      interpret("Outer[f, {{a, b}, {c, d}}, {x, y}, 1]").unwrap(),
      "{{f[{a, b}, x], f[{a, b}, y]}, {f[{c, d}, x], f[{c, d}, y]}}"
    );
  }

  // When the LAST argument to `Outer` is a SparseArray, wolframscript
  // wraps the leaves as SparseArray[…] with the function applied to the
  // accumulated outer values plus each sparse value/default. The earlier
  // arguments densify normally.
  #[test]
  fn sparse_array_pair() {
    assert_eq!(
      interpret(
        "Outer[f, SparseArray[{{1, 2} -> a, {2, 1} -> b}], \
         SparseArray[{{1, 2} -> c, {2, 1} -> d}]]"
      )
      .unwrap(),
      "{{SparseArray[Automatic, {2, 2}, f[0, 0], \
       {1, {{0, 1, 2}, {{2}, {1}}}, {f[0, c], f[0, d]}}], \
       SparseArray[Automatic, {2, 2}, f[a, 0], \
       {1, {{0, 1, 2}, {{2}, {1}}}, {f[a, c], f[a, d]}}]}, \
       {SparseArray[Automatic, {2, 2}, f[b, 0], \
       {1, {{0, 1, 2}, {{2}, {1}}}, {f[b, c], f[b, d]}}], \
       SparseArray[Automatic, {2, 2}, f[0, 0], \
       {1, {{0, 1, 2}, {{2}, {1}}}, {f[0, c], f[0, d]}}]}}"
    );
  }

  #[test]
  fn sparse_array_with_list() {
    // wolframscript collapses Outer[Times, …SparseArray…] into a single
    // SparseArray (default 0) since Times[…, 0] = 0 always defaults.
    assert_eq!(
      interpret(
        "Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], {c, d}]"
      )
      .unwrap(),
      "SparseArray[Automatic, {2, 2, 2}, 0, {1, {{0, 2, 4}, \
       {{2, 1}, {2, 2}, {1, 1}, {1, 2}}}, {a*c, a*d, b*c, b*d}}]"
    );
  }

  #[test]
  fn level_spec_per_list() {
    // Per-list level specs: descend 1 level in first, 1 level in second.
    assert_eq!(
      interpret("Outer[f, {a, b}, {x, y}, 1, 1]").unwrap(),
      "{{f[a, x], f[a, y]}, {f[b, x], f[b, y]}}"
    );
  }

  #[test]
  fn level_spec_asymmetric() {
    // Descend 1 level in first list, 2 levels in second.
    assert_eq!(
      interpret("Outer[f, {a, b}, {{x, y}, {z, w}}, 1, 2]").unwrap(),
      "{{{f[a, x], f[a, y]}, {f[a, z], f[a, w]}}, {{f[b, x], f[b, y]}, {f[b, z], f[b, w]}}}"
    );
  }

  #[test]
  fn level_spec_shallow_second() {
    // Descend 1 level in each, treating inner lists as atomic.
    assert_eq!(
      interpret("Outer[f, {a, b}, {{x, y}, {z, w}}, 1, 1]").unwrap(),
      "{{f[a, {x, y}], f[a, {z, w}]}, {f[b, {x, y}], f[b, {z, w}]}}"
    );
  }

  #[test]
  fn level_spec_both_nested() {
    assert_eq!(
      interpret("Outer[f, {{a, b}, {c, d}}, {{x, y}, {z, w}}, 1, 1]").unwrap(),
      "{{f[{a, b}, {x, y}], f[{a, b}, {z, w}]}, {f[{c, d}, {x, y}], f[{c, d}, {z, w}]}}"
    );
  }

  #[test]
  fn level_spec_times() {
    assert_eq!(
      interpret("Outer[Times, {{a, b}, {c, d}}, {x, y}, 1]").unwrap(),
      "{{{a*x, b*x}, {a*y, b*y}}, {{c*x, d*x}, {c*y, d*y}}}"
    );
  }

  #[test]
  fn level_spec_ragged() {
    assert_eq!(
      interpret("Outer[f, {{1, 2}, {3}}, {{a, b}, {c}}, 1]").unwrap(),
      "{{f[{1, 2}, {a, b}], f[{1, 2}, {c}]}, {f[{3}, {a, b}], f[{3}, {c}]}}"
    );
  }
}

mod map_indexed {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("MapIndexed[f, {a, b, c}]").unwrap(),
      "{f[a, {1}], f[b, {2}], f[c, {3}]}"
    );
  }

  #[test]
  fn level_spec_2() {
    assert_eq!(
      interpret("MapIndexed[f, {{a, b}, {c, d}}, {2}]").unwrap(),
      "{{f[a, {1, 1}], f[b, {1, 2}]}, {f[c, {2, 1}], f[d, {2, 2}]}}"
    );
  }

  #[test]
  fn level_spec_1() {
    assert_eq!(
      interpret("MapIndexed[f, {{a, b}, {c, d}}, {1}]").unwrap(),
      "{f[{a, b}, {1}], f[{c, d}, {2}]}"
    );
  }

  #[test]
  fn heads_true_basic() {
    assert_eq!(
      interpret("MapIndexed[f, {a, b, c}, Heads->True]").unwrap(),
      "f[List, {0}][f[a, {1}], f[b, {2}], f[c, {3}]]"
    );
  }

  #[test]
  fn heads_true_nested_level_one() {
    assert_eq!(
      interpret("MapIndexed[f, {a, {b, c}}, Heads->True]").unwrap(),
      "f[List, {0}][f[a, {1}], f[{b, c}, {2}]]"
    );
  }

  #[test]
  fn heads_true_with_level_spec() {
    assert_eq!(
      interpret("MapIndexed[f, {a, b}, {1}, Heads -> True]").unwrap(),
      "f[List, {0}][f[a, {1}], f[b, {2}]]"
    );
  }

  // `{-1}` selects every atomic leaf, threading the position index. The
  // mathics doctest in test_cases.rs case 3356 walks a deep nested
  // listified expression — pin the simpler shape here so a regression
  // surfaces directly rather than blowing up in the chained statement.
  #[test]
  fn level_spec_neg_one_visits_atoms_only() {
    assert_eq!(
      interpret("MapIndexed[#2 &, {a, {b, {c}}}, {-1}]").unwrap(),
      "{{1}, {{2, 1}, {{2, 2, 1}}}}"
    );
  }

  // With Heads -> True at level {-1}, the head of every compound node is
  // also rewritten — the head's position is the parent's position with
  // `0` appended.
  #[test]
  fn level_spec_neg_one_with_heads_rewrites_head() {
    assert_eq!(
      interpret("MapIndexed[#2 &, {a, {b}}, {-1}, Heads -> True]").unwrap(),
      "{0}[{1}, {2, 0}[{2, 1}]]"
    );
  }

  // Round-trip identity: extracting each leaf at its `MapIndexed`-found
  // position reconstructs the original expression. This matches case
  // 3358's chained-statement doctest end-to-end.
  #[test]
  fn level_spec_neg_one_round_trip_via_extract() {
    assert_eq!(
      interpret(
        "expr = a + b * f[g] * c ^ e; \
         listified = Apply[List, expr, {0, Infinity}]; \
         MapIndexed[Extract[expr, #2] &, listified, {-1}, Heads -> True]"
      )
      .unwrap(),
      "a + b*c^e*f[g]"
    );
  }
}

mod tensor_symmetry {
  use super::*;

  #[test]
  fn antisymmetric_3x3() {
    // M[i,j] = -M[j,i]; diagonal zero.
    assert_eq!(
      interpret("TensorSymmetry[{{0, 5, 6}, {-5, 0, 3}, {-6, -3, 0}}]")
        .unwrap(),
      "Antisymmetric[{1, 2}]"
    );
  }

  #[test]
  fn antisymmetric_2x2() {
    assert_eq!(
      interpret("TensorSymmetry[{{0, 5}, {-5, 0}}]").unwrap(),
      "Antisymmetric[{1, 2}]"
    );
  }

  #[test]
  fn symmetric_3x3() {
    assert_eq!(
      interpret("TensorSymmetry[{{1, 2, 3}, {2, 4, 5}, {3, 5, 6}}]").unwrap(),
      "Symmetric[{1, 2}]"
    );
  }

  #[test]
  fn symmetric_2x2() {
    assert_eq!(
      interpret("TensorSymmetry[{{1, 2}, {2, 4}}]").unwrap(),
      "Symmetric[{1, 2}]"
    );
  }

  #[test]
  fn identity_matrix_is_symmetric() {
    assert_eq!(
      interpret("TensorSymmetry[IdentityMatrix[3]]").unwrap(),
      "Symmetric[{1, 2}]"
    );
  }

  #[test]
  fn all_zero_matrix() {
    assert_eq!(
      interpret("TensorSymmetry[{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}]").unwrap(),
      "ZeroSymmetric[{}]"
    );
  }

  #[test]
  fn generic_matrix() {
    // Neither symmetric nor antisymmetric.
    assert_eq!(
      interpret("TensorSymmetry[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn non_matrix_stays_symbolic() {
    // 1-D vector — no rank-2 symmetry classification.
    let result = interpret("TensorSymmetry[{1, 2, 3}]").unwrap();
    assert!(
      result.contains("TensorSymmetry"),
      "expected unevaluated, got {}",
      result
    );
  }
}

mod tensor_product {
  use super::*;

  #[test]
  fn two_vectors() {
    assert_eq!(
      interpret("TensorProduct[{a, b}, {c, d}]").unwrap(),
      "{{a*c, a*d}, {b*c, b*d}}"
    );
  }

  #[test]
  fn three_vectors() {
    assert_eq!(
      interpret("TensorProduct[{a, b}, {c, d}, {e, f}]").unwrap(),
      "{{{a*c*e, a*c*f}, {a*d*e, a*d*f}}, {{b*c*e, b*c*f}, {b*d*e, b*d*f}}}"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("TensorProduct[{1, 2}, {3, 4}]").unwrap(),
      "{{3, 4}, {6, 8}}"
    );
  }

  #[test]
  fn operator_short_form() {
    // The \[TensorProduct] operator (U+F3DA) parses as TensorProduct.
    assert_eq!(
      interpret("{a, b} \\[TensorProduct] {c, d}").unwrap(),
      "{{a*c, a*d}, {b*c, b*d}}"
    );
  }

  #[test]
  fn operator_audit_case() {
    // Audit case: three-fold TensorProduct via the operator.
    assert_eq!(
      interpret(
        "{2, 3} \\[TensorProduct] {{a, b}, {c, d}} \\[TensorProduct] {x, y}"
      )
      .unwrap(),
      "{{{{2*a*x, 2*a*y}, {2*b*x, 2*b*y}}, {{2*c*x, 2*c*y}, {2*d*x, 2*d*y}}}, {{{3*a*x, 3*a*y}, {3*b*x, 3*b*y}}, {{3*c*x, 3*c*y}, {3*d*x, 3*d*y}}}}"
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

  #[test]
  fn all_with_greater() {
    assert_eq!(
      interpret("Ordering[{3, 1, 2, 4, 5}, All, Greater]").unwrap(),
      "{5, 4, 1, 3, 2}"
    );
  }

  #[test]
  fn limit_with_greater() {
    assert_eq!(
      interpret("Ordering[{3, 1, 2, 4, 5}, 2, Greater]").unwrap(),
      "{5, 4}"
    );
  }

  #[test]
  fn all_default_ordering() {
    assert_eq!(
      interpret("Ordering[{3, 1, 2, 4, 5}, All]").unwrap(),
      "{2, 3, 1, 4, 5}"
    );
  }

  #[test]
  fn limit_with_pure_function() {
    assert_eq!(
      interpret("Ordering[{3, 1, 2, 4, 5}, 2, (#1 > #2)&]").unwrap(),
      "{5, 4}"
    );
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
  fn delete_out_of_range_returns_unevaluated() {
    // Matches wolframscript: out-of-range position emits Delete::partw and
    // returns the expression unevaluated rather than silently no-oping.
    assert_eq!(
      interpret("Delete[{a, b, c, d}, 5]").unwrap(),
      "Delete[{a, b, c, d}, 5]"
    );
    assert_eq!(
      interpret("Delete[{a, b, c, d}, -5]").unwrap(),
      "Delete[{a, b, c, d}, -5]"
    );
  }

  #[test]
  fn delete_multiple() {
    assert_eq!(
      interpret("Delete[{a, b, c, d, e}, {{1}, {3}}]").unwrap(),
      "{b, d, e}"
    );
  }
}

mod insert {
  use super::*;

  #[test]
  fn insert_positive() {
    assert_eq!(
      interpret("Insert[{a, b, c, d}, x, 2]").unwrap(),
      "{a, x, b, c, d}"
    );
  }

  #[test]
  fn insert_negative() {
    assert_eq!(
      interpret("Insert[{a, b, c}, x, -2]").unwrap(),
      "{a, b, x, c}"
    );
    assert_eq!(
      interpret("Insert[{a, b, c}, x, -1]").unwrap(),
      "{a, b, c, x}"
    );
  }

  #[test]
  fn insert_single_position_in_list() {
    // Insert[list, x, {n}] — position as a length-1 list
    assert_eq!(
      interpret("Insert[{a, b, c}, x, {2}]").unwrap(),
      "{a, x, b, c}"
    );
  }

  #[test]
  fn insert_multiple_positions() {
    // Regression: Insert[list, x, {{p1}, {p2}, ...}] used to error with
    // "position must be an integer". Positions refer to the original list.
    assert_eq!(
      interpret("Insert[{a, b, c, d, e}, x, {{2}, {4}}]").unwrap(),
      "{a, x, b, c, x, d, e}"
    );
    assert_eq!(
      interpret("Insert[{a, b, c, d, e}, x, {{1}, {3}, {5}}]").unwrap(),
      "{x, a, b, x, c, d, x, e}"
    );
  }

  #[test]
  fn insert_into_arbitrary_head() {
    // Insert threads through any non-list head, preserving it.
    assert_eq!(
      interpret("Insert[f[a, b, c, d], x, 2]").unwrap(),
      "f[a, x, b, c, d]"
    );
  }

  #[test]
  fn insert_into_arbitrary_head_negative_position() {
    assert_eq!(
      interpret("Insert[g[a, b, c], x, -1]").unwrap(),
      "g[a, b, c, x]"
    );
  }

  #[test]
  fn insert_into_arbitrary_head_multiple_positions() {
    assert_eq!(
      interpret("Insert[g[a, b, c, d], x, {{1}, {3}}]").unwrap(),
      "g[x, a, b, x, c, d]"
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

  #[test]
  fn tensor_dimensions_2d() {
    assert_eq!(
      interpret("TensorDimensions[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn tensor_dimensions_scalar() {
    assert_eq!(interpret("TensorDimensions[5]").unwrap(), "{}");
  }

  #[test]
  fn tensor_dimensions_3d() {
    assert_eq!(
      interpret("TensorDimensions[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}]")
        .unwrap(),
      "{2, 2, 2}"
    );
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

  #[test]
  fn nothing_filtered_in_map() {
    assert_eq!(
      interpret("Map[If[OddQ[#], #, Nothing] &, {1, 2, 3, 4, 5}]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn nothing_filtered_in_table() {
    assert_eq!(
      interpret("Table[If[OddQ[k], k, Nothing], {k, 1, 5}]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn nothing_filtered_in_array() {
    assert_eq!(
      interpret("Array[If[OddQ[#], #, Nothing] &, 5]").unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn nothing_filtered_in_map_indexed() {
    assert_eq!(
      interpret(
        "MapIndexed[If[OddQ[First[#2]], #1, Nothing] &, {a, b, c, d, e}]"
      )
      .unwrap(),
      "{a, c, e}"
    );
  }

  #[test]
  fn nothing_filtered_in_map_thread() {
    assert_eq!(
      interpret(
        "MapThread[If[#1 > #2, Nothing, #2] &, {{1, 5, 3}, {4, 2, 6}}]"
      )
      .unwrap(),
      "{4, 6}"
    );
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

  #[test]
  fn take_n() {
    assert_eq!(
      interpret("MinimalBy[{-3, 1, 2, -5, 4}, Abs, 3]").unwrap(),
      "{1, 2, -3}"
    );
  }

  #[test]
  fn take_one() {
    assert_eq!(
      interpret("MinimalBy[{5, 3, 1, 4, 2}, Identity, 1]").unwrap(),
      "{1}"
    );
  }

  #[test]
  fn take_n_exceeds_length() {
    // When n > length, return all elements sorted by criterion.
    assert_eq!(
      interpret("MinimalBy[{5, 3, 1}, Identity, 10]").unwrap(),
      "{1, 3, 5}"
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

  #[test]
  fn take_n() {
    assert_eq!(
      interpret("MaximalBy[{-3, 1, 2, -5, 4}, Abs, 3]").unwrap(),
      "{-5, 4, -3}"
    );
  }

  #[test]
  fn take_one() {
    assert_eq!(
      interpret("MaximalBy[{5, 3, 7, 1, 4}, Identity, 1]").unwrap(),
      "{7}"
    );
  }

  #[test]
  fn take_n_string_length() {
    assert_eq!(
      interpret(r#"MaximalBy[{"abc", "a", "ab"}, StringLength, 2]"#).unwrap(),
      "{abc, ab}"
    );
  }

  #[test]
  fn take_n_exceeds_length() {
    // When n > length, return all elements sorted by criterion.
    assert_eq!(
      interpret("MaximalBy[{-3, 1, 2, -5, 4}, Abs, 10]").unwrap(),
      "{-5, 4, -3, 2, 1}"
    );
  }
}

mod map_all {
  use super::*;

  #[test]
  fn basic_expression() {
    assert_eq!(
      interpret("MapAll[f, a + b c]").unwrap(),
      "f[f[a] + f[f[b]*f[c]]]"
    );
  }

  #[test]
  fn list() {
    assert_eq!(
      interpret("MapAll[f, {1, 2, 3}]").unwrap(),
      "f[{f[1], f[2], f[3]}]"
    );
  }

  #[test]
  fn atom() {
    assert_eq!(interpret("MapAll[f, x]").unwrap(), "f[x]");
  }

  #[test]
  fn nested() {
    assert_eq!(interpret("MapAll[f, g[h[x]]]").unwrap(), "f[g[f[h[f[x]]]]]");
  }

  #[test]
  fn integer() {
    assert_eq!(interpret("MapAll[f, 42]").unwrap(), "f[42]");
  }

  #[test]
  fn pure_function() {
    // Applies #+1 to each atom first, then to the list itself
    assert_eq!(
      interpret("MapAll[# + 1 &, {1, 2, 3}]").unwrap(),
      "{3, 4, 5}"
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

  #[test]
  fn span_basic() {
    // MapAt[f, list, start;;end] applies f to elements at positions start..end.
    assert_eq!(
      interpret("MapAt[f, {a, b, c, d, e}, 1;;3]").unwrap(),
      "{f[a], f[b], f[c], d, e}"
    );
  }

  #[test]
  fn span_with_negative_end() {
    assert_eq!(
      interpret("MapAt[f, {a, b, c, d, e}, 2;;-1]").unwrap(),
      "{a, f[b], f[c], f[d], f[e]}"
    );
  }

  #[test]
  fn span_with_step() {
    assert_eq!(
      interpret("MapAt[f, {a, b, c, d, e}, 1;;-1;;2]").unwrap(),
      "{f[a], b, f[c], d, f[e]}"
    );
  }

  #[test]
  fn span_all() {
    // ;; is 1;;All, i.e. all elements.
    assert_eq!(
      interpret("MapAt[f, {a, b, c, d, e}, ;;]").unwrap(),
      "{f[a], f[b], f[c], f[d], f[e]}"
    );
  }

  #[test]
  fn span_from_position_to_end() {
    assert_eq!(
      interpret("MapAt[f, {a, b, c, d, e}, 3;;]").unwrap(),
      "{a, b, f[c], f[d], f[e]}"
    );
  }

  #[test]
  fn top_level_span_bare() {
    // Regression: bare `;;` used to fail parsing at the statement level.
    // wolframscript's REPL preserves the `FullForm` wrapper for Span,
    // printing `FullForm[Span[1, All]]`. The bare head form is reachable
    // via `ToString[FullForm[...]]`.
    assert_eq!(
      interpret(";; // FullForm").unwrap(),
      "FullForm[Span[1, All]]"
    );
    assert_eq!(
      interpret("ToString[FullForm[1 ;; All]]").unwrap(),
      "Span[1, All]"
    );
  }

  #[test]
  fn top_level_span_with_step() {
    assert_eq!(
      interpret("1;;4;;2 // FullForm").unwrap(),
      "FullForm[Span[1, 4, 2]]"
    );
  }

  #[test]
  fn top_level_span_negative_end() {
    assert_eq!(
      interpret("2;;-2 // FullForm").unwrap(),
      "FullForm[Span[2, -2]]"
    );
  }

  #[test]
  fn top_level_span_from_start() {
    assert_eq!(
      interpret(";;3 // FullForm").unwrap(),
      "FullForm[Span[1, 3]]"
    );
  }

  #[test]
  fn map_at_association_positive_index() {
    assert_eq!(
      interpret(r#"MapAt[f, <|"a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4|>, 2]"#)
        .unwrap(),
      "<|a -> 1, b -> f[2], c -> 3, d -> 4|>"
    );
  }

  #[test]
  fn map_at_association_negative_index() {
    assert_eq!(
      interpret(r#"MapAt[f, <|"a" -> 1, "b" -> 2, "c" -> 3, "d" -> 4|>, -2]"#)
        .unwrap(),
      "<|a -> 1, b -> 2, c -> f[3], d -> 4|>"
    );
  }

  #[test]
  fn map_at_association_index_out_of_range_stays_symbolic() {
    assert_eq!(
      interpret(r#"MapAt[f, <|"a" -> 1, "b" -> 2|>, 3]"#).unwrap(),
      "MapAt[f, <|a -> 1, b -> 2|>, 3]"
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

  #[test]
  fn sort_by_non_list_head() {
    assert_eq!(
      interpret("SortBy[f[{3, 1}, {1}, {2, 2, 2}], Length]").unwrap(),
      "f[{1}, {3, 1}, {2, 2, 2}]"
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
  fn sort_function_call_and_byte_array() {
    // ByteArray sorts before an unrelated FunctionCall in canonical order.
    assert_eq!(
      interpret("Sort[{F[2], ByteArray[{2}]}]").unwrap(),
      "{ByteArray[<1>], F[2]}"
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

  #[test]
  fn sort_non_list_head() {
    // Sort should work on any expression head, not just List
    assert_eq!(interpret("Sort[f[c, a, b]]").unwrap(), "f[a, b, c]");
    assert_eq!(interpret("Sort[f[3, 1, 2]]").unwrap(), "f[1, 2, 3]");
    assert_eq!(interpret("Sort[g[c, a, b, d]]").unwrap(), "g[a, b, c, d]");
  }

  #[test]
  fn sort_non_list_head_with_comparator() {
    assert_eq!(
      interpret("Sort[f[1, 3, 2], Greater]").unwrap(),
      "f[3, 2, 1]"
    );
    assert_eq!(interpret("Sort[f[3, 1, 2], Less]").unwrap(), "f[1, 2, 3]");
  }

  #[test]
  fn sort_with_pure_function_greater() {
    assert_eq!(
      interpret("Sort[{5, 2, 8, 1, 9}, #1 > #2 &]").unwrap(),
      "{9, 8, 5, 2, 1}"
    );
  }

  #[test]
  fn sort_with_pure_function_less() {
    assert_eq!(
      interpret("Sort[{5, 2, 8, 1, 9}, #1 < #2 &]").unwrap(),
      "{1, 2, 5, 8, 9}"
    );
  }

  #[test]
  fn sort_with_pure_function_on_nested_list() {
    // Sort pairs by their second element ascending.
    assert_eq!(
      interpret("Sort[{{1, 2}, {3, 1}, {2, 5}}, #1[[2]] < #2[[2]] &]").unwrap(),
      "{{3, 1}, {1, 2}, {2, 5}}"
    );
  }

  #[test]
  fn sort_with_named_function() {
    assert_eq!(
      interpret("Sort[{5, 2, 8, 1, 9}, Function[{a, b}, a > b]]").unwrap(),
      "{9, 8, 5, 2, 1}"
    );
  }

  #[test]
  fn sort_with_pure_function_preserves_head() {
    // Custom comparator on a non-List head should also work.
    assert_eq!(
      interpret("Sort[f[3, 1, 2], #1 > #2 &]").unwrap(),
      "f[3, 2, 1]"
    );
  }

  #[test]
  fn sort_nested_lists_element_wise() {
    // Nested lists should be compared element-wise, not as strings
    assert_eq!(
      interpret("Sort[{{10, 1}, {2, 5}}]").unwrap(),
      "{{2, 5}, {10, 1}}"
    );
    assert_eq!(
      interpret("Sort[{{2, 1}, {1, 3}, {1, 2}}]").unwrap(),
      "{{1, 2}, {1, 3}, {2, 1}}"
    );
  }

  #[test]
  fn sort_nested_lists_by_length() {
    // Shorter lists come before longer lists when prefixes match
    assert_eq!(
      interpret("Sort[{{1, 2, 3}, {1}, {1, 2}}]").unwrap(),
      "{{1}, {1, 2}, {1, 2, 3}}"
    );
  }

  #[test]
  fn sort_lists_after_atoms() {
    // Numbers sort before lists
    assert_eq!(
      interpret("Sort[{3, {1, 2}, 1, {0}}]").unwrap(),
      "{1, 3, {0}, {1, 2}}"
    );
  }

  #[test]
  fn sort_function_calls_element_wise() {
    // Function calls compared by name, then args
    assert_eq!(
      interpret("Sort[{f[2], f[1], g[1]}]").unwrap(),
      "{f[1], f[2], g[1]}"
    );
  }

  #[test]
  fn sort_function_calls_before_lists() {
    // Function calls sort before lists
    assert_eq!(
      interpret("Sort[{f[2], {1}, g[1]}]").unwrap(),
      "{f[2], g[1], {1}}"
    );
  }

  #[test]
  fn sort_with_infinity() {
    // Wolfram canonical ordering: finite numbers first, then -Infinity, then Infinity
    assert_eq!(
      interpret("Sort[{5, Infinity, -Infinity, 0}]").unwrap(),
      "{0, 5, -Infinity, Infinity}"
    );
  }

  #[test]
  fn sort_with_negative_infinity() {
    assert_eq!(
      interpret("Sort[{3, -Infinity, 1, Infinity, -1, 0}]").unwrap(),
      "{-1, 0, 1, 3, -Infinity, Infinity}"
    );
  }

  #[test]
  fn ordered_q_with_infinity() {
    // In Wolfram canonical ordering, -Infinity sorts after finite numbers
    assert_eq!(
      interpret("OrderedQ[{-Infinity, 0, Infinity}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("OrderedQ[{Infinity, 0, -Infinity}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("OrderedQ[{0, 5, -Infinity, Infinity}]").unwrap(),
      "True"
    );
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

mod partition {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5, 6}, 2]").unwrap(),
      "{{1, 2}, {3, 4}, {5, 6}}"
    );
  }

  #[test]
  fn with_offset() {
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5}, 3, 1]").unwrap(),
      "{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}"
    );
  }

  #[test]
  fn overhang_short_sublists() {
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5, 6, 7}, 3, 3, {1, 1}, {}]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6}, {7}}"
    );
  }

  #[test]
  fn overhang_with_padding() {
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5}, 2, 2, {1, 1}, x]").unwrap(),
      "{{1, 2}, {3, 4}, {5, x}}"
    );
  }

  #[test]
  fn overhang_no_remainder() {
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4}, 2, 2, {1, 1}, {}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  #[test]
  fn drops_incomplete_without_overhang() {
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5}, 3]").unwrap(),
      "{{1, 2, 3}}"
    );
  }

  #[test]
  fn upto_audit_case() {
    // Audit case: UpTo[n] partitions into chunks of up to n with the last
    // chunk possibly shorter.
    assert_eq!(
      interpret("Partition[{a, b, c, d, e, f}, UpTo[4]]").unwrap(),
      "{{a, b, c, d}, {e, f}}"
    );
  }

  #[test]
  fn upto_smaller_chunks() {
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4, 5}, UpTo[2]]").unwrap(),
      "{{1, 2}, {3, 4}, {5}}"
    );
  }

  #[test]
  fn upto_exact_multiple() {
    assert_eq!(
      interpret("Partition[{1, 2, 3, 4}, UpTo[2]]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  #[test]
  fn upto_chunk_larger_than_list() {
    assert_eq!(
      interpret("Partition[{1, 2, 3}, UpTo[10]]").unwrap(),
      "{{1, 2, 3}}"
    );
  }

  #[test]
  fn cyclic_11_basic() {
    // {1, 1} alignment: cyclic wrapping
    assert_eq!(
      interpret("Partition[{a, b, c, d, e}, 2, 1, {1, 1}]").unwrap(),
      "{{a, b}, {b, c}, {c, d}, {d, e}, {e, a}}"
    );
  }

  #[test]
  fn cyclic_11_short() {
    assert_eq!(
      interpret("Partition[{a, b, c}, 2, 1, {1, 1}]").unwrap(),
      "{{a, b}, {b, c}, {c, a}}"
    );
  }

  #[test]
  fn cyclic_11_with_stride() {
    assert_eq!(
      interpret("Partition[{a, b, c, d}, 3, 2, {1, 1}]").unwrap(),
      "{{a, b, c}, {c, d, a}}"
    );
  }

  #[test]
  fn cyclic_11_large_window() {
    assert_eq!(
      interpret("Partition[{a, b, c, d, e, f}, 4, 2, {1, 1}]").unwrap(),
      "{{a, b, c, d}, {c, d, e, f}, {e, f, a, b}}"
    );
  }

  #[test]
  fn cyclic_neg11_basic() {
    // {-1, -1}: wraps backwards
    assert_eq!(
      interpret("Partition[{a, b, c, d}, 2, 1, {-1, -1}]").unwrap(),
      "{{d, a}, {a, b}, {b, c}, {c, d}}"
    );
  }

  #[test]
  fn cyclic_neg11_with_stride() {
    assert_eq!(
      interpret("Partition[{a, b, c, d, e}, 3, 2, {-1, -1}]").unwrap(),
      "{{d, e, a}, {a, b, c}, {c, d, e}}"
    );
  }

  #[test]
  fn align_1_neg1_default() {
    // {1, -1} is the default, same as no overhang
    assert_eq!(
      interpret("Partition[{a, b, c, d}, 3, 1, {1, -1}]").unwrap(),
      "{{a, b, c}, {b, c, d}}"
    );
  }
}

mod symmetric_difference {
  use super::*;

  #[test]
  fn basic_two_lists() {
    assert_eq!(
      interpret("SymmetricDifference[{1, 2, 3}, {2, 3, 4}]").unwrap(),
      "{1, 4}"
    );
  }

  #[test]
  fn three_lists() {
    assert_eq!(
      interpret("SymmetricDifference[{1, 2, 3}, {2, 3, 4}, {3, 4, 5}]")
        .unwrap(),
      "{1, 3, 5}"
    );
  }

  #[test]
  fn with_duplicates() {
    assert_eq!(
      interpret("SymmetricDifference[{1, 2, 2, 3}, {2, 3, 4}]").unwrap(),
      "{1, 4}"
    );
  }

  #[test]
  fn identical_lists() {
    assert_eq!(
      interpret("SymmetricDifference[{1, 2, 3}, {1, 2, 3}]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn disjoint_lists() {
    assert_eq!(
      interpret("SymmetricDifference[{1, 2}, {3, 4}]").unwrap(),
      "{1, 2, 3, 4}"
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

  #[test]
  fn clip_list_default() {
    assert_eq!(interpret("Clip[{-2, 0.5, 3}]").unwrap(), "{-1, 0.5, 1}");
  }

  #[test]
  fn clip_list_with_range() {
    assert_eq!(
      interpret("Clip[{-2, 0.5, 3}, {0, 1}]").unwrap(),
      "{0, 0.5, 1}"
    );
  }

  #[test]
  fn clip_list_with_replacements() {
    assert_eq!(
      interpret("Clip[{-2, 0.5, 3}, {0, 1}, {a, b}]").unwrap(),
      "{a, 0.5, b}"
    );
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
  fn part_row_and_column_index() {
    // M = {{a, b, c}, {d, e, f}}; M[[1, 2]] extracts row 1, column 2 → b
    assert_eq!(
      interpret("M = {{a, b, c}, {d, e, f}}; M[[1, 2]]").unwrap(),
      "b"
    );
  }

  #[test]
  fn part_single_index_on_named_list() {
    // A = {a, b, c, d}; A[[3]] extracts the third element.
    assert_eq!(interpret("A = {a, b, c, d}; A[[3]]").unwrap(), "c");
  }

  #[test]
  fn part_full_span_then_column() {
    // B[[;;, 2]] with a 3x3 matrix extracts column 2.
    assert_eq!(
      interpret("B = {{a, b, c}, {d, e, f}, {g, h, i}}; B[[;;, 2]]").unwrap(),
      "{b, e, h}"
    );
  }

  #[test]
  fn part_span_then_span() {
    // B[[1;;2, 1;;2]] extracts the top-left 2x2 sub-matrix.
    assert_eq!(
      interpret("B = {{a, b, c}, {d, e, f}, {g, h, i}}; B[[1;;2, 1;;2]]")
        .unwrap(),
      "{{a, b}, {d, e}}"
    );
  }

  #[test]
  fn part_list_rows_and_negative_span_cols() {
    // Rows 1 and 3, columns -2 through -1 (last two) of a 3×3 matrix.
    assert_eq!(
      interpret("B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; B[[{1, 3}, -2;;-1]]")
        .unwrap(),
      "{{2, 3}, {8, 9}}"
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
  fn part_matrix_assignment_updates_cell() {
    // A[[1, 2]] = 5 updates the matrix in place.
    assert_eq!(
      interpret("A = {{1, 2}, {3, 4}}; A[[1, 2]] = 5; A").unwrap(),
      "{{1, 5}, {3, 4}}"
    );
  }

  #[test]
  fn part_assignment_on_undefined_symbol_returns_rhs() {
    // Matches Mathematica: Part assignment on a symbol with no immediate
    // value emits a 'Set::noval' warning and returns the RHS unchanged.
    clear_state();
    assert_eq!(interpret("freshPartUndefA[[1, 2]] = 5").unwrap(), "5");
  }

  #[test]
  fn part_assignment_on_undefined_symbol_list_rhs() {
    clear_state();
    assert_eq!(
      interpret("freshPartUndefB[[;;, 2]] = {6, 7}").unwrap(),
      "{6, 7}"
    );
  }

  #[test]
  fn part_list_of_indices_assignment_distributes() {
    // a[[{i, j}]] = {x, y} assigns element-wise when lengths match.
    assert_eq!(
      interpret("a = {1, 2, 3, 4, 5}; a[[{1, 3}]] = {99, 88}; a").unwrap(),
      "{99, 2, 88, 4, 5}"
    );
  }

  #[test]
  fn part_list_of_indices_assignment_broadcasts_scalar() {
    // a[[{i, j}]] = scalar broadcasts the scalar to every selected position.
    assert_eq!(
      interpret("a = {1, 2, 3, 4, 5}; a[[{1, 3}]] = 0; a").unwrap(),
      "{0, 2, 0, 4, 5}"
    );
  }

  #[test]
  fn part_list_of_indices_assignment_broadcasts_mismatched_list() {
    // RHS list with different length is broadcast as a whole to each position.
    assert_eq!(
      interpret("a = {1, 2, 3, 4, 5}; a[[{1, 3, 5}]] = {99, 88}; a").unwrap(),
      "{{99, 88}, 2, {99, 88}, 4, {99, 88}}"
    );
  }

  #[test]
  fn part_list_of_indices_assignment_with_inner_column() {
    // Nested case: a[[{i, j}, k]] = {x, y} sets column k of selected rows.
    assert_eq!(
      interpret(
        "a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; a[[{1, 3}, 2]] = {99, 88}; a"
      )
      .unwrap(),
      "{{1, 99, 3}, {4, 5, 6}, {7, 88, 9}}"
    );
  }

  #[test]
  fn part_list_of_indices_assignment_swap_in_module() {
    // Regression: Module-local list with `a[[{i, j}]] = a[[{j, i}]]` swap.
    assert_eq!(
      interpret(
        "swap[lst_, i_, j_] := Module[{a = lst}, a[[{i, j}]] = a[[{j, i}]]; a]; swap[{1, 2, 3, 4, 5}, 1, 3]"
      )
      .unwrap(),
      "{3, 2, 1, 4, 5}"
    );
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

  #[test]
  fn part_list_of_rows_then_column() {
    // Regression: Part[matrix, {i1, i2}, j] should extract column j
    // from rows i1 and i2, not apply `j` to the result of selecting rows.
    assert_eq!(
      interpret("{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}[[{1, 3}, 2]]").unwrap(),
      "{2, 8}"
    );
    assert_eq!(
      interpret("{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}[[{1, 3}, 1]]").unwrap(),
      "{1, 7}"
    );
  }

  #[test]
  fn part_list_of_rows_then_list_of_columns() {
    // Regression: Part[matrix, {i1, i2}, {j1, j2}] should give a sub-matrix
    assert_eq!(
      interpret("{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}[[{1, 3}, {1, 2}]]").unwrap(),
      "{{1, 2}, {7, 8}}"
    );
  }

  #[test]
  fn part_list_of_rows_only() {
    // Single List index still works
    assert_eq!(
      interpret("{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}[[{1, 3}]]").unwrap(),
      "{{1, 2, 3}, {7, 8, 9}}"
    );
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

mod part_paren_extended {
  use super::*;

  #[test]
  fn paren_part_basic() {
    // (expr)[[index]] should work for parenthesized expressions
    assert_eq!(interpret("({a, b, c})[[2]]").unwrap(), "b");
  }

  #[test]
  fn paren_part_with_set() {
    // (var = expr)[[index]] should evaluate the Set and then extract Part
    assert_eq!(interpret("(x = {10, 20, 30})[[2]]").unwrap(), "20");
  }

  #[test]
  fn paren_part_function_result() {
    // (FunctionCall)[[index]] where function returns a list
    assert_eq!(interpret("(MonomialFactor[u, x])[[1]]").unwrap(), "u");
  }

  #[test]
  fn paren_part_unsameq() {
    // (expr)[[1]] =!= 0 should work
    assert_eq!(
      interpret("(MonomialFactor[u, x])[[1]] =!= 0").unwrap(),
      "True"
    );
  }

  #[test]
  fn paren_part_nested_index() {
    // (expr)[[1]][[2]] should apply Part twice
    assert_eq!(interpret("({{a, b}, {c, d}})[[1]][[2]]").unwrap(), "b");
  }
}

mod part_binary_op {
  use super::*;

  #[test]
  fn part_of_power() {
    // Power[base, exp][[1]] = base, [[2]] = exp
    assert_eq!(interpret("(a^b)[[1]]").unwrap(), "a");
    assert_eq!(interpret("(a^b)[[2]]").unwrap(), "b");
    assert_eq!(interpret("(a^b)[[0]]").unwrap(), "Power");
  }

  #[test]
  fn part_of_reciprocal() {
    // 1/(2*x - 3) = Power[-3 + 2*x, -1]
    assert_eq!(interpret("(1/(2*x - 3))[[1]]").unwrap(), "-3 + 2*x");
    assert_eq!(interpret("(1/(2*x - 3))[[2]]").unwrap(), "-1");
  }

  #[test]
  fn part_of_plus() {
    assert_eq!(interpret("(a + b)[[1]]").unwrap(), "a");
    assert_eq!(interpret("(a + b)[[2]]").unwrap(), "b");
    assert_eq!(interpret("(a + b)[[0]]").unwrap(), "Plus");
  }

  #[test]
  fn part_of_plus_three_terms_head() {
    // Part 0 of a Plus expression returns the head symbol Plus.
    assert_eq!(interpret("(a + b + c)[[0]]").unwrap(), "Plus");
  }

  #[test]
  fn part_of_plus_three_terms_second() {
    // Part 2 of `a + b + c` (sorted canonically by Plus) is `b`.
    assert_eq!(interpret("(a + b + c)[[2]]").unwrap(), "b");
  }

  #[test]
  fn part_of_times() {
    assert_eq!(interpret("(a*b)[[1]]").unwrap(), "a");
    assert_eq!(interpret("(a*b)[[2]]").unwrap(), "b");
    assert_eq!(interpret("(a*b)[[0]]").unwrap(), "Times");
  }

  #[test]
  fn part_negative_index() {
    // Negative index: [[−1]] is last part
    assert_eq!(interpret("(a^b)[[-1]]").unwrap(), "b");
    assert_eq!(interpret("(a^b)[[-2]]").unwrap(), "a");
  }

  #[test]
  fn part_unicode_double_struck_brackets() {
    // 〚 (U+301A) and 〛 (U+301B) are the Wolfram double-struck
    // part-bracket glyphs and should behave identically to [[ ]].
    assert_eq!(
      interpret("myList = {10,20,30,40,50}; myList〚3〛").unwrap(),
      "30"
    );
    assert_eq!(interpret("{{1,2,3},{4,5,6}}〚2, 3〛").unwrap(), "6");
    assert_eq!(interpret("{1,2,3,4,5}〚2;;4〛").unwrap(), "{2, 3, 4}");
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

  #[test]
  fn take_first_n_rows_and_cols() {
    // With A = {{a,b,c},{d,e,f}}, Take[A, 2, 2] returns the first 2 rows
    // and first 2 columns.
    assert_eq!(
      interpret("Take[{{a, b, c}, {d, e, f}}, 2, 2]").unwrap(),
      "{{a, b}, {d, e}}"
    );
  }

  #[test]
  fn take_all_rows_single_column() {
    // `{2}` as the second argument selects exactly column 2 (wrapped in a
    // 1-element sublist), matching wolframscript.
    assert_eq!(
      interpret("Take[{{a, b, c}, {d, e, f}}, All, {2}]").unwrap(),
      "{{b}, {e}}"
    );
  }

  #[test]
  fn take_up_to_within_length() {
    assert_eq!(
      interpret("Take[{1, 2, 3, 4, 5}, UpTo[3]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn take_up_to_exceeds_length() {
    assert_eq!(interpret("Take[{1, 2, 3}, UpTo[10]]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn take_up_to_exact_length() {
    assert_eq!(interpret("Take[{1, 2, 3}, UpTo[3]]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn take_non_list_head() {
    assert_eq!(
      interpret("Take[f[a, b, c, d, e], 3]").unwrap(),
      "f[a, b, c]"
    );
  }

  #[test]
  fn take_non_list_head_negative() {
    assert_eq!(interpret("Take[f[a, b, c, d, e], -2]").unwrap(), "f[d, e]");
  }

  #[test]
  fn take_non_list_head_range() {
    assert_eq!(
      interpret("Take[f[a, b, c, d, e], {2, 4}]").unwrap(),
      "f[b, c, d]"
    );
  }

  #[test]
  fn drop_non_list_head() {
    assert_eq!(
      interpret("Drop[f[a, b, c, d, e], 2]").unwrap(),
      "f[c, d, e]"
    );
  }

  #[test]
  fn drop_non_list_head_negative() {
    assert_eq!(
      interpret("Drop[f[a, b, c, d, e], -2]").unwrap(),
      "f[a, b, c]"
    );
  }

  #[test]
  fn drop_up_to_more_than_length() {
    // Drop[list, UpTo[n]] drops at most n; capped at list length.
    assert_eq!(interpret("Drop[{a, b, c, d}, UpTo[6]]").unwrap(), "{}");
  }

  #[test]
  fn drop_up_to_partial() {
    assert_eq!(interpret("Drop[{a, b, c, d}, UpTo[2]]").unwrap(), "{c, d}");
  }

  #[test]
  fn drop_up_to_zero() {
    assert_eq!(
      interpret("Drop[{a, b, c, d}, UpTo[0]]").unwrap(),
      "{a, b, c, d}"
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

  #[test]
  fn unitize_within_tolerance_is_zero() {
    assert_eq!(interpret("Unitize[0.0001, 0.001]").unwrap(), "0");
  }

  #[test]
  fn unitize_outside_tolerance_is_one() {
    assert_eq!(interpret("Unitize[0.01, 0.001]").unwrap(), "1");
  }

  #[test]
  fn unitize_negative_within_tolerance() {
    assert_eq!(interpret("Unitize[-0.005, 0.01]").unwrap(), "0");
  }

  #[test]
  fn unitize_list_with_tolerance() {
    assert_eq!(
      interpret("Unitize[{-0.001, 0.005, 0.02}, 0.01]").unwrap(),
      "{0, 0, 1}"
    );
  }

  #[test]
  fn unitize_rational_tolerance() {
    // 1/100 < 1/10, so should be 0
    assert_eq!(interpret("Unitize[1/100, 1/10]").unwrap(), "0");
  }

  #[test]
  fn unitize_nonzero_constants() {
    // Numeric mathematical constants are all positive, hence non-zero.
    assert_eq!(interpret("Unitize[Pi]").unwrap(), "1");
    assert_eq!(interpret("Unitize[E]").unwrap(), "1");
    assert_eq!(interpret("Unitize[EulerGamma]").unwrap(), "1");
    assert_eq!(interpret("Unitize[GoldenRatio]").unwrap(), "1");
  }

  #[test]
  fn unitize_nonzero_algebraic() {
    // Algebraic / closed-form expressions that evaluate to a non-zero
    // real should also collapse to 1.
    assert_eq!(interpret("Unitize[Sqrt[2]]").unwrap(), "1");
    assert_eq!(interpret("Unitize[2 Pi]").unwrap(), "1");
    assert_eq!(interpret("Unitize[3/4]").unwrap(), "1");
    assert_eq!(interpret("Unitize[Log[2]]").unwrap(), "1");
  }

  #[test]
  fn unitize_symbolic_unchanged() {
    // Free symbols remain unevaluated.
    assert_eq!(interpret("Unitize[x]").unwrap(), "Unitize[x]");
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

mod heaviside_theta {
  use super::*;

  #[test]
  fn positive() {
    assert_eq!(interpret("HeavisideTheta[1]").unwrap(), "1");
    assert_eq!(interpret("HeavisideTheta[5]").unwrap(), "1");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("HeavisideTheta[-1]").unwrap(), "0");
    assert_eq!(interpret("HeavisideTheta[-5]").unwrap(), "0");
  }

  #[test]
  fn at_zero_stays_symbolic() {
    assert_eq!(interpret("HeavisideTheta[0]").unwrap(), "HeavisideTheta[0]");
  }

  #[test]
  fn multi_arg_all_positive() {
    assert_eq!(interpret("HeavisideTheta[1, 2]").unwrap(), "1");
  }

  #[test]
  fn multi_arg_with_negative() {
    assert_eq!(interpret("HeavisideTheta[1, -1]").unwrap(), "0");
  }

  #[test]
  fn multi_arg_with_zero() {
    assert_eq!(
      interpret("HeavisideTheta[1, 0]").unwrap(),
      "HeavisideTheta[0, 1]"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("HeavisideTheta[x]").unwrap(), "HeavisideTheta[x]");
  }

  #[test]
  fn constant_positive() {
    assert_eq!(interpret("HeavisideTheta[Pi]").unwrap(), "1");
  }
}

mod dirac_delta {
  use super::*;

  #[test]
  fn nonzero_is_zero() {
    assert_eq!(interpret("DiracDelta[1]").unwrap(), "0");
    assert_eq!(interpret("DiracDelta[-1]").unwrap(), "0");
    assert_eq!(interpret("DiracDelta[5]").unwrap(), "0");
  }

  #[test]
  fn at_zero_stays_symbolic() {
    assert_eq!(interpret("DiracDelta[0]").unwrap(), "DiracDelta[0]");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("DiracDelta[x]").unwrap(), "DiracDelta[x]");
  }

  #[test]
  fn constant_is_zero() {
    assert_eq!(interpret("DiracDelta[Pi]").unwrap(), "0");
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

  #[test]
  fn with_max_iterations_cap() {
    // NestWhile[f, x, test, m, max] — max caps the number of iterations.
    assert_eq!(
      interpret("NestWhile[#/2 &, 1024, # > 1 &, 1, 5]").unwrap(),
      "32"
    );
  }

  #[test]
  fn list_form_with_max_iterations_cap() {
    assert_eq!(
      interpret("NestWhileList[#/2 &, 1024, # > 1 &, 1, 5]").unwrap(),
      "{1024, 512, 256, 128, 64, 32}"
    );
  }

  #[test]
  fn four_arg_form_with_default_m() {
    // NestWhile[f, x, test, 1] behaves like the 3-arg form.
    assert_eq!(interpret("NestWhile[#/2 &, 16, # > 1 &, 1]").unwrap(), "1");
  }

  #[test]
  fn six_arg_extra_iterations_after_test_fails() {
    // NestWhile[f, x, test, m, max, n] with n > 0 applies f an additional
    // n times after the test stops being True.
    // 1 -> 2 -> 3 -> 4 -> 5 (test fails), then +2 more: 6, 7
    assert_eq!(
      interpret("NestWhile[# + 1 &, 1, # < 5 &, 1, Infinity, 2]").unwrap(),
      "7"
    );
  }

  #[test]
  fn six_arg_negative_n_returns_earlier_value() {
    // NestWhile[..., n] with n < 0 returns the result n iterations *before*
    // the test stopped being True.
    // 1024 -> 512 -> ... -> 2 -> 1 (test fails); n = -1 -> 2.
    assert_eq!(
      interpret(
        "NestWhile[#/2 &, 1024, IntegerQ[#] && # > 1 &, 1, Infinity, -1]"
      )
      .unwrap(),
      "2"
    );
  }

  #[test]
  fn six_arg_zero_n_is_identity() {
    // n = 0 is the same as the 5-arg form.
    assert_eq!(
      interpret("NestWhile[# + 1 &, 1, # < 5 &, 1, Infinity, 0]").unwrap(),
      "5"
    );
  }

  #[test]
  fn list_form_six_arg_extra_iterations() {
    // NestWhileList with n > 0 extends the list by n extra iterations
    // beyond the value at which the test failed.
    assert_eq!(
      interpret("NestWhileList[# + 1 &, 1, # < 5 &, 1, Infinity, 2]").unwrap(),
      "{1, 2, 3, 4, 5, 6, 7}"
    );
  }

  #[test]
  fn list_form_six_arg_negative_n_truncates() {
    // NestWhileList with n < 0 drops the trailing |n| values.
    assert_eq!(
      interpret(
        "NestWhileList[#/2 &, 1024, IntegerQ[#] && # > 1 &, 1, Infinity, -1]"
      )
      .unwrap(),
      "{1024, 512, 256, 128, 64, 32, 16, 8, 4, 2}"
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

  // Sow[val, {tag1, tag2, ...}] sows val once per tag.
  #[test]
  fn sow_with_tag_list_emits_one_entry_per_tag() {
    assert_eq!(
      interpret("Reap[Sow[1, {a, b, a}]]").unwrap(),
      "{1, {{1, 1}, {1}}}"
    );
  }

  // Reap[expr, patt, f] applies f[tag, {vals}] to each unique matched tag.
  #[test]
  fn reap_three_arg_applies_wrapper_function() {
    assert_eq!(
      interpret("Reap[Sow[Null, {a, a, b, d, c, a}], _, # &][[2]]").unwrap(),
      "{a, b, d, c}"
    );
  }

  #[test]
  fn reap_three_arg_with_pattern_list() {
    assert_eq!(
      interpret(
        "Reap[Sow[2, {x, x, x}]; Sow[3, x]; Sow[4, y]; Sow[4, 1], \
         {_Symbol, _Integer, x}, f]"
      )
      .unwrap(),
      "{4, {{f[x, {2, 2, 2, 3}], f[y, {4}]}, {f[1, {4}]}, \
       {f[x, {2, 2, 2, 3}]}}}"
    );
  }

  // Sows whose tag does not match the inner Reap's pattern must bubble up
  // to the enclosing Reap scope. Here `Sow[b, 1]` has Integer tag `1`
  // which does not match `_Symbol`, so it propagates and is collected by
  // the outer `Reap[..., _, f]` as `f[1, {b}]`.
  #[test]
  fn nested_reap_propagates_unmatched_sows() {
    assert_eq!(
      interpret(
        "Reap[Reap[Sow[a, x]; Sow[b, 1], _Symbol, Print[\"Inner: \", #1]&];, _, f]"
      )
      .unwrap(),
      "{Null, {f[1, {b}]}}"
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
  fn list_form_single_dim() {
    // RandomReal[max, {n}] should work like RandomReal[max, n]
    assert_eq!(interpret("Length[RandomReal[1, {10}]]").unwrap(), "10");
    assert_eq!(interpret("Length[RandomReal[{0, 1}, {50}]]").unwrap(), "50");
  }

  #[test]
  fn list_form_multi_dim() {
    // RandomReal[max, {n, m}] should produce an n x m matrix
    assert_eq!(
      interpret("Dimensions[RandomReal[1, {3, 4}]]").unwrap(),
      "{3, 4}"
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

mod legacy_random {
  use super::*;

  #[test]
  fn no_args_is_real() {
    let s = interpret("Random[]").unwrap();
    let v: f64 = s.parse().unwrap();
    assert!((0.0..1.0).contains(&v));
  }

  #[test]
  fn integer_with_range() {
    // Random[Integer, {1, 100}] → integer in [1, 100]
    let s = interpret("Random[Integer, {1, 100}]").unwrap();
    let v: i128 = s.parse().unwrap();
    assert!((1..=100).contains(&v));
  }

  #[test]
  fn real_with_max() {
    let s = interpret("Random[Real, 10]").unwrap();
    let v: f64 = s.parse().unwrap();
    assert!((0.0..10.0).contains(&v));
  }

  #[test]
  fn complex_head() {
    assert_eq!(interpret("Head[Random[Complex]]").unwrap(), "Complex");
  }
}

mod random_complex {
  use super::*;

  #[test]
  fn no_args_is_complex() {
    // RandomComplex[] should be a complex number a + b*I with a, b in [0, 1)
    let s = interpret("RandomComplex[]").unwrap();
    assert!(s.contains("I"));
  }

  #[test]
  fn head_is_complex() {
    assert_eq!(interpret("Head[RandomComplex[]]").unwrap(), "Complex");
  }

  #[test]
  fn list_length() {
    assert_eq!(interpret("Length[RandomComplex[1+I, 5]]").unwrap(), "5");
  }

  #[test]
  fn matrix_dimensions() {
    assert_eq!(
      interpret("Dimensions[RandomComplex[{1+I, 2+2I}, {2, 2}]]").unwrap(),
      "{2, 2}"
    );
  }

  #[test]
  fn real_range_is_respected() {
    // With an all-real range RandomComplex[{0, 3}, 5] imaginary parts should be 0
    assert_eq!(
      interpret("AllTrue[RandomComplex[{0, 3}, 20], Im[#] == 0 &]").unwrap(),
      "True"
    );
  }
}

mod random_date {
  use super::*;

  #[test]
  fn no_args_is_date_object() {
    // RandomDate[] returns a DateObject with structure
    //   DateObject[{y, m, d, h, mi, s}, "Instant", "Gregorian", offset].
    assert_eq!(interpret("Head[RandomDate[]]").unwrap(), "DateObject");
  }

  #[test]
  fn no_args_in_current_year() {
    // The chosen year is the system's current year — the first entry of the
    // DateObject's date list equals `Now`'s year.
    assert_eq!(
      interpret("RandomDate[][[1, 1]] == Now[[1, 1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn n_arg_returns_list_of_n() {
    assert_eq!(interpret("Length[RandomDate[5]]").unwrap(), "5");
  }

  #[test]
  fn n_arg_all_date_objects() {
    assert_eq!(
      interpret("AllTrue[RandomDate[10], Head[#] == DateObject &]").unwrap(),
      "True"
    );
  }

  #[test]
  fn n_zero_returns_empty_list() {
    assert_eq!(interpret("RandomDate[0]").unwrap(), "{}");
  }
}

mod random_color {
  use super::*;

  #[test]
  fn no_args_is_rgb_color() {
    // RandomColor[] should return RGBColor[r, g, b] with r, g, b in [0, 1).
    assert_eq!(interpret("Head[RandomColor[]]").unwrap(), "RGBColor");
  }

  #[test]
  fn no_args_three_channels_in_unit_range() {
    // Each of the three RGB channels must lie in [0, 1).
    assert_eq!(
      interpret("AllTrue[Apply[List, RandomColor[]], (0 <= # < 1) &]").unwrap(),
      "True"
    );
  }

  #[test]
  fn n_arg_returns_list_of_n() {
    assert_eq!(interpret("Length[RandomColor[3]]").unwrap(), "3");
  }

  #[test]
  fn n_arg_all_rgb_colors() {
    // Each element must be an RGBColor with three channels in [0, 1).
    assert_eq!(
      interpret(
        "AllTrue[RandomColor[10], (Head[#] == RGBColor && \
         AllTrue[Apply[List, #], (0 <= # < 1) &]) &]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn n_zero_returns_empty_list() {
    assert_eq!(interpret("RandomColor[0]").unwrap(), "{}");
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

  #[test]
  fn list_form_single_dim() {
    // RandomInteger[max, {n}] should work like RandomInteger[max, n]
    assert_eq!(interpret("Length[RandomInteger[10, {5}]]").unwrap(), "5");
  }

  #[test]
  fn list_form_multi_dim() {
    // RandomInteger[max, {n, m}] should produce an n x m matrix
    assert_eq!(
      interpret("Dimensions[RandomInteger[10, {3, 4}]]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn zero_length() {
    // wolframscript: `RandomInteger[150, 0]` -> `{}`. Required for
    // RosettaCode `sorting_algorithms_selection_sort.wls`, which calls
    // `RandomInteger[150, RandomInteger[1000]]` — the inner draw can
    // legitimately return 0.
    assert_eq!(interpret("RandomInteger[150, 0]").unwrap(), "{}");
  }

  #[test]
  fn zero_outer_dim() {
    assert_eq!(interpret("RandomInteger[150, {0, 3}]").unwrap(), "{}");
  }

  #[test]
  fn zero_inner_dim() {
    // wolframscript: `RandomInteger[150, {3, 0}]` -> `{{}, {}, {}}`
    assert_eq!(
      interpret("RandomInteger[150, {3, 0}]").unwrap(),
      "{{}, {}, {}}"
    );
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

  #[test]
  fn gamma_distribution_inert() {
    assert_eq!(
      interpret("GammaDistribution[2, 3]").unwrap(),
      "GammaDistribution[2, 3]"
    );
  }

  #[test]
  fn gamma_distribution_pdf() {
    let result = interpret("PDF[GammaDistribution[2, 3], x]").unwrap();
    // Should contain x and be a Piecewise
    assert!(
      result.contains("Piecewise") && result.contains("x > 0"),
      "Expected Piecewise PDF: {}",
      result
    );
  }

  #[test]
  fn gamma_distribution_pdf_at_value() {
    assert_eq!(
      interpret("PDF[GammaDistribution[1, 1], 1]").unwrap(),
      "E^(-1)"
    );
  }

  #[test]
  fn gamma_distribution_expectation() {
    // E[x] for Gamma[2,3] = alpha*beta = 6
    assert_eq!(
      interpret("Expectation[x, Distributed[x, GammaDistribution[2, 3]]]")
        .unwrap(),
      "6"
    );
  }

  #[test]
  fn gamma_distribution_expectation_symbolic() {
    assert_eq!(
      interpret("Expectation[x, Distributed[x, GammaDistribution[a, b]]]")
        .unwrap(),
      "a*b"
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

  #[test]
  fn poisson_single_is_non_negative_integer() {
    let result = interpret("RandomVariate[PoissonDistribution[3]]").unwrap();
    let val: i64 = result.parse().unwrap();
    assert!(val >= 0, "Poisson sample must be non-negative, got {}", val);
  }

  #[test]
  fn poisson_list_length() {
    assert_eq!(
      interpret("Length[RandomVariate[PoissonDistribution[3], 10]]").unwrap(),
      "10"
    );
  }

  #[test]
  fn poisson_list_all_non_negative_integers() {
    // All elements must be non-negative integers.
    assert_eq!(
      interpret(
        "AllTrue[RandomVariate[PoissonDistribution[3], 50], (IntegerQ[#] && # >= 0) &]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn poisson_mean_near_lambda() {
    // Mean of many Poisson(5) samples should be near 5.
    let result: f64 =
      interpret("Mean[N[RandomVariate[PoissonDistribution[5], 2000]]]")
        .unwrap()
        .parse()
        .unwrap();
    assert!(result > 4.0 && result < 6.0, "got {}", result);
  }

  #[test]
  fn binormal_single() {
    // BinormalDistribution[rho] is 2-D standard normal with correlation rho.
    let result = interpret("RandomVariate[BinormalDistribution[1/2]]").unwrap();
    assert!(result.starts_with('{'));
    assert!(result.ends_with('}'));
    let inside = &result[1..result.len() - 1];
    let vals: Vec<f64> = inside
      .split(',')
      .map(|s| s.trim().parse().unwrap())
      .collect();
    assert_eq!(vals.len(), 2);
    assert!(vals[0].is_finite() && vals[1].is_finite());
  }

  #[test]
  fn binormal_list() {
    // n=5 → 5 lists of length 2.
    let result =
      interpret("RandomVariate[BinormalDistribution[1/2], 5]").unwrap();
    assert!(result.starts_with("{{"));
    assert_eq!(
      interpret("Length[RandomVariate[BinormalDistribution[1/2], 5]]").unwrap(),
      "5"
    );
    // Each pair has length 2.
    assert_eq!(
      interpret(
        "AllTrue[RandomVariate[BinormalDistribution[1/2], 5], Length[#] == 2 &]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn multivariate_poisson_single() {
    // MultivariatePoissonDistribution[1, {2, 3}] → 2-D non-negative integers.
    let result =
      interpret("RandomVariate[MultivariatePoissonDistribution[1, {2, 3}]]")
        .unwrap();
    assert!(result.starts_with('{'));
    assert!(result.ends_with('}'));
  }

  #[test]
  fn multivariate_poisson_list() {
    assert_eq!(
      interpret(
        "Length[RandomVariate[MultivariatePoissonDistribution[1, {2, 3}], 5]]"
      )
      .unwrap(),
      "5"
    );
    // Each element has length matching the marginals list.
    assert_eq!(
      interpret(
        "AllTrue[\
           RandomVariate[MultivariatePoissonDistribution[1, {2, 3}], 20],\
           (Length[#] == 2 && AllTrue[#, (IntegerQ[#] && # >= 0) &]) &\
         ]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod seed_random {
  use super::*;

  #[test]
  fn deterministic_integer() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[42]");
    let a = interpret("RandomInteger[100]").unwrap();
    let _ = interpret("SeedRandom[42]");
    let b = interpret("RandomInteger[100]").unwrap();
    assert_eq!(a, b);
  }

  #[test]
  fn deterministic_real() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[42]");
    let a = interpret("RandomReal[]").unwrap();
    let _ = interpret("SeedRandom[42]");
    let b = interpret("RandomReal[]").unwrap();
    assert_eq!(a, b);
  }

  #[test]
  fn deterministic_sequence() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[123]");
    let a1 = interpret("RandomInteger[1000]").unwrap();
    let a2 = interpret("RandomInteger[1000]").unwrap();
    let a3 = interpret("RandomReal[]").unwrap();

    let _ = interpret("SeedRandom[123]");
    let b1 = interpret("RandomInteger[1000]").unwrap();
    let b2 = interpret("RandomInteger[1000]").unwrap();
    let b3 = interpret("RandomReal[]").unwrap();

    assert_eq!(a1, b1);
    assert_eq!(a2, b2);
    assert_eq!(a3, b3);
  }

  #[test]
  fn different_seeds_differ() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[1]");
    let a = interpret("RandomInteger[{1, 1000000}]").unwrap();
    let _ = interpret("SeedRandom[2]");
    let b = interpret("RandomInteger[{1, 1000000}]").unwrap();
    assert_ne!(a, b);
  }

  #[test]
  fn returns_null() {
    woxi::clear_state();
    assert_eq!(interpret("SeedRandom[42]").unwrap(), "\0");
  }

  #[test]
  fn string_seed_returns_null() {
    woxi::clear_state();
    assert_eq!(interpret("SeedRandom[\"CKM\"]").unwrap(), "\0");
  }

  #[test]
  fn string_seed_is_reproducible() {
    // The same string seed must produce the same sequence on every call.
    woxi::clear_state();
    let _ = interpret("SeedRandom[\"CKM\"]");
    let a = interpret("RandomInteger[1000000]").unwrap();
    let _ = interpret("SeedRandom[\"CKM\"]");
    let b = interpret("RandomInteger[1000000]").unwrap();
    assert_eq!(a, b);
  }

  #[test]
  fn distinct_string_seeds_differ() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[\"CKM\"]");
    let a = interpret("RandomInteger[1000000]").unwrap();
    let _ = interpret("SeedRandom[\"Other\"]");
    let b = interpret("RandomInteger[1000000]").unwrap();
    assert_ne!(a, b);
  }

  #[test]
  fn unseed_resets() {
    woxi::clear_state();
    // After SeedRandom[], results should no longer be deterministic
    // (in practice we just check it doesn't error)
    assert_eq!(interpret("SeedRandom[]").unwrap(), "\0");
    // Should still produce valid results
    let result: f64 = interpret("RandomReal[]").unwrap().parse().unwrap();
    assert!(result >= 0.0 && result < 1.0);
  }

  #[test]
  fn deterministic_choice() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[42]");
    let a = interpret("RandomChoice[{a, b, c, d, e}]").unwrap();
    let _ = interpret("SeedRandom[42]");
    let b = interpret("RandomChoice[{a, b, c, d, e}]").unwrap();
    assert_eq!(a, b);
  }

  #[test]
  fn deterministic_sample() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[42]");
    let a = interpret("RandomSample[{1, 2, 3, 4, 5}]").unwrap();
    let _ = interpret("SeedRandom[42]");
    let b = interpret("RandomSample[{1, 2, 3, 4, 5}]").unwrap();
    assert_eq!(a, b);
  }

  #[test]
  fn deterministic_integer_list() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[42]");
    let a = interpret("RandomInteger[100, 10]").unwrap();
    let _ = interpret("SeedRandom[42]");
    let b = interpret("RandomInteger[100, 10]").unwrap();
    assert_eq!(a, b);
  }

  #[test]
  fn deterministic_variate() {
    woxi::clear_state();
    let _ = interpret("SeedRandom[42]");
    let a = interpret("RandomVariate[UniformDistribution[{0, 1}]]").unwrap();
    let _ = interpret("SeedRandom[42]");
    let b = interpret("RandomVariate[UniformDistribution[{0, 1}]]").unwrap();
    assert_eq!(a, b);
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

  #[test]
  fn fold_empty_list_unevaluated() {
    // Fold[f, {}] is unevaluated in Wolfram Language
    assert_eq!(interpret("Fold[Plus, {}]").unwrap(), "Fold[Plus, {}]");
  }

  #[test]
  fn fold_non_list_head() {
    // Fold should thread through any non-list head.
    assert_eq!(
      interpret("Fold[f, x, g[a, b, c]]").unwrap(),
      "f[f[f[x, a], b], c]"
    );
  }

  #[test]
  fn fold_two_arg_non_list_head() {
    assert_eq!(
      interpret("Fold[f, g[a, b, c, d]]").unwrap(),
      "f[f[f[a, b], c], d]"
    );
  }

  #[test]
  fn fold_list_non_list_head() {
    // FoldList wraps the result in the original head.
    assert_eq!(
      interpret("FoldList[f, x, g[a, b, c]]").unwrap(),
      "g[x, f[x, a], f[f[x, a], b], f[f[f[x, a], b], c]]"
    );
  }

  #[test]
  fn fold_list_two_arg_non_list_head() {
    assert_eq!(
      interpret("FoldList[f, g[a, b, c, d]]").unwrap(),
      "g[a, f[a, b], f[f[a, b], c], f[f[f[a, b], c], d]]"
    );
  }

  #[test]
  fn fold_list_numeric_non_list() {
    assert_eq!(
      interpret("FoldList[Plus, 0, g[1, 2, 3]]").unwrap(),
      "g[0, 1, 3, 6]"
    );
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

  // Regression: the old matcher only recursed into `Expr::List`, so it
  // couldn't find `x^2` inside `1 + x^2` (a `Plus` expression). Recurse
  // into FunctionCall args and BinaryOp operands as well.
  #[test]
  fn inside_plus_expression() {
    assert_eq!(
      interpret("FirstPosition[{1 + x^2, 5, x^4, a + (1 + x^2)^2}, x^2]")
        .unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn inside_function_call() {
    // Looks inside `f[..., y, ...]` to locate `y` at position {1, 2}.
    assert_eq!(
      interpret("FirstPosition[{f[x, y, z], g[a]}, y]").unwrap(),
      "{1, 2}"
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

  #[test]
  fn ranked_min_negative_index() {
    // RankedMin[list, -n] gives the n-th largest element.
    assert_eq!(
      interpret("RankedMin[{3, 1, 4, 1, 5, 9, 2}, -2]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("RankedMin[{3, 1, 4, 1, 5, 9, 2}, -1]").unwrap(),
      "9"
    );
  }

  #[test]
  fn ranked_max_negative_index() {
    // RankedMax[list, -n] gives the n-th smallest element.
    assert_eq!(
      interpret("RankedMax[{3, 1, 4, 1, 5, 9, 2}, -1]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("RankedMax[{3, 1, 4, 1, 5, 9, 2}, -3]").unwrap(),
      "2"
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

  #[test]
  fn range_too_many_args() {
    clear_state();
    let result = interpret_with_stdout("Range[2, 12, 3, 1]").unwrap();
    assert_eq!(result.result, "Range[2, 12, 3, 1]");
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains(
      "Range::argb: Range called with 4 arguments; between 1 and 3 arguments are expected."
    ));
  }

  #[test]
  fn range_no_args() {
    clear_state();
    let result = interpret_with_stdout("Range[]").unwrap();
    assert_eq!(result.result, "Range[]");
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains(
      "Range::argb: Range called with 0 arguments; between 1 and 3 arguments are expected."
    ));
  }

  #[test]
  fn range_symbolic_endpoints_shift() {
    assert_eq!(
      interpret("Range[a, a + 5]").unwrap(),
      "{a, 1 + a, 2 + a, 3 + a, 4 + a, 5 + a}"
    );
  }

  #[test]
  fn range_symbolic_endpoints_with_coefficient() {
    assert_eq!(
      interpret("Range[2 a, 2 a + 3]").unwrap(),
      "{2*a, 1 + 2*a, 2 + 2*a, 3 + 2*a}"
    );
  }

  #[test]
  fn range_symbolic_endpoints_with_step() {
    assert_eq!(
      interpret("Range[a, a + 5, 2]").unwrap(),
      "{a, 2 + a, 4 + a}"
    );
  }

  #[test]
  fn range_symbolic_empty() {
    assert_eq!(interpret("Range[a, a - 1]").unwrap(), "{}");
  }

  #[test]
  fn range_fully_symbolic_unevaluated() {
    assert_eq!(interpret("Range[a, b]").unwrap(), "Range[a, b]");
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

  // With SameTest -> Equal, numeric equality replaces structural identity,
  // so 1.0 is "contained" in a set holding 1.
  #[test]
  fn contains_only_with_same_test_equal() {
    assert_eq!(
      interpret("ContainsOnly[{a, 1.0}, {1, a, b}, {SameTest -> Equal}]")
        .unwrap(),
      "True"
    );
  }

  // Without SameTest, 1.0 and 1 aren't structurally equal, so the call is
  // False.
  #[test]
  fn contains_only_without_same_test() {
    assert_eq!(
      interpret("ContainsOnly[{a, 1.0}, {1, a, b}]").unwrap(),
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
    assert_eq!(interpret("Rest[{}]").unwrap(), "Rest[{}]");
  }

  #[test]
  fn rest_single_element_evaluates() {
    // Rest[Times[a, b]] should return b (not Times[b])
    assert_eq!(interpret("Rest[Times[a, b]]").unwrap(), "b");
  }

  #[test]
  fn rest_binary_op_power() {
    // Issue #79: Rest should work on BinaryOp expressions (Power)
    assert_eq!(interpret("Rest[a^b]").unwrap(), "b");
  }

  #[test]
  fn first_binary_op_power() {
    // Issue #79: First should work on BinaryOp expressions
    assert_eq!(interpret("First[a^b]").unwrap(), "a");
  }

  #[test]
  fn last_binary_op_power() {
    assert_eq!(interpret("Last[a^b]").unwrap(), "b");
  }

  #[test]
  fn first_rule() {
    // First on a Rule returns the LHS (head is Rule, first arg is LHS).
    assert_eq!(interpret("First[x -> a]").unwrap(), "x");
  }

  #[test]
  fn last_rule() {
    assert_eq!(interpret("Last[x -> a]").unwrap(), "a");
  }

  #[test]
  fn first_rule_delayed() {
    assert_eq!(interpret("First[x :> a]").unwrap(), "x");
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

  #[test]
  fn range_symbolic_pi() {
    assert_eq!(
      interpret("Range[0, 2 Pi, Pi/4]").unwrap(),
      "{0, Pi/4, Pi/2, (3*Pi)/4, Pi, (5*Pi)/4, (3*Pi)/2, (7*Pi)/4, 2*Pi}"
    );
  }

  #[test]
  fn range_symbolic_pi_sixth() {
    assert_eq!(
      interpret("Range[0, Pi, Pi/6]").unwrap(),
      "{0, Pi/6, Pi/3, Pi/2, (2*Pi)/3, (5*Pi)/6, Pi}"
    );
  }

  #[test]
  fn range_symbolic_sqrt() {
    assert_eq!(
      interpret("Range[Sqrt[2], 5 Sqrt[2], Sqrt[2]]").unwrap(),
      "{Sqrt[2], 2*Sqrt[2], 3*Sqrt[2], 4*Sqrt[2], 5*Sqrt[2]}"
    );
  }
}

mod delete_duplicates_with_test {
  use super::*;

  #[test]
  fn delete_duplicates_parity_test() {
    assert_eq!(
      interpret("DeleteDuplicates[{1, 2, 3, 4, 5}, Mod[#1 - #2, 2] == 0 &]")
        .unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn delete_duplicates_test_keeps_first_seen() {
    assert_eq!(
      interpret("DeleteDuplicates[{3, 1, 2, 3, 4, 5}, Mod[#1 - #2, 2] == 0 &]")
        .unwrap(),
      "{3, 2}"
    );
  }

  #[test]
  fn delete_duplicates_test_equal_is_passthrough() {
    assert_eq!(
      interpret("DeleteDuplicates[{1, 2, 3, 4, 5, 6, 7}, #1 == #2 &]").unwrap(),
      "{1, 2, 3, 4, 5, 6, 7}"
    );
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

  #[test]
  fn delete_deep_position_into_atom_errors() {
    // {1, 2} descends into element `a`, which has no parts. wolframscript
    // emits Delete::partw and returns the expression unevaluated.
    // Regression for mathics list/eol.py:346.
    assert_eq!(
      interpret("Delete[{a, b, c, d}, {1, 2}]").unwrap(),
      "Delete[{a, b, c, d}, {1, 2}]"
    );
  }

  // Regression: `{3, 0}` deletes the head of the 3rd element. Removing a
  // head leaves a Sequence that the outer FunctionCall flattens, so
  // `Delete[f[a, b, u + v, c], {3, 0}]` → `f[a, b, u, v, c]`, matching
  // wolframscript.
  #[test]
  fn delete_nested_head_splices_into_parent() {
    assert_eq!(
      interpret("Delete[f[a, b, u + v, c], {3, 0}]").unwrap(),
      "f[a, b, u, v, c]"
    );
  }

  #[test]
  fn delete_head_at_top_level_returns_sequence() {
    // Top-level head deletion of f[a,b,c] produces a Sequence that
    // renders as its spliced args — same as wolframscript's display.
    assert_eq!(interpret("Delete[f[a, b, c], 0]").unwrap(), "abc");
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

  #[test]
  fn extract_with_head_multi() {
    // Extract[expr, positions, h] wraps each result with h
    assert_eq!(
      interpret("Extract[{a, {b, c}, d}, {{2, 1}, {3}}, f]").unwrap(),
      "{f[b], f[d]}"
    );
  }

  #[test]
  fn extract_with_head_single_position() {
    assert_eq!(interpret("Extract[{a, b, c}, {2}, f]").unwrap(), "f[b]");
  }

  #[test]
  fn extract_with_head_nested() {
    assert_eq!(
      interpret("Extract[{{a, b}, {c, d}}, {1, 2}, g]").unwrap(),
      "g[b]"
    );
  }

  #[test]
  fn extract_with_head_multiple_flat() {
    assert_eq!(
      interpret("Extract[{a, b, c, d}, {{1}, {3}}, f]").unwrap(),
      "{f[a], f[c]}"
    );
  }

  #[test]
  fn extract_with_head_preserves_order() {
    assert_eq!(
      interpret("Extract[{10, 20, 30, 40}, {{1}, {4}, {2}}, f]").unwrap(),
      "{f[10], f[40], f[20]}"
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

mod part_multi_index {
  use super::*;

  #[test]
  fn part_two_indices() {
    assert_eq!(interpret("Part[{a, {b, c}, d}, 2, 1]").unwrap(), "b");
  }

  #[test]
  fn part_two_indices_bracket_syntax() {
    assert_eq!(interpret("lst = {a, {b, c}, d}; lst[[2, 1]]").unwrap(), "b");
  }

  #[test]
  fn part_three_indices() {
    assert_eq!(
      interpret("Part[{{{a, b}, {c, d}}, {{e, f}, {g, h}}}, 1, 2, 1]").unwrap(),
      "c"
    );
  }

  #[test]
  fn part_multi_index_nested_matrix() {
    assert_eq!(
      interpret("Part[{{1, 2}, {3, 4}, {5, 6}}, 2, 2]").unwrap(),
      "4"
    );
  }

  #[test]
  fn part_multi_index_negative() {
    assert_eq!(interpret("Part[{{a, b}, {c, d}}, -1, -1]").unwrap(), "d");
  }

  #[test]
  fn part_multi_index_head() {
    assert_eq!(interpret("Part[{f[a, b], g[c, d]}, 2, 0]").unwrap(), "g");
  }

  #[test]
  fn part_rule_first() {
    // Part 1 of a Rule gives the pattern (left-hand side)
    assert_eq!(interpret("Part[a -> b, 1]").unwrap(), "a");
  }

  #[test]
  fn part_rule_second() {
    // Part 2 of a Rule gives the replacement (right-hand side)
    assert_eq!(interpret("Part[a -> b, 2]").unwrap(), "b");
  }

  #[test]
  fn part_list_head() {
    // Part 0 of a List gives the head "List"
    assert_eq!(interpret("Part[{a, b, c}, 0]").unwrap(), "List");
    assert_eq!(interpret("{a, b, c}[[0]]").unwrap(), "List");
  }

  #[test]
  fn part_rule_head() {
    // Part 0 of a Rule gives the head "Rule"
    assert_eq!(interpret("Part[a -> b, 0]").unwrap(), "Rule");
  }

  #[test]
  fn part_rule_negative_index() {
    // Part -1 of a Rule gives the last element (replacement)
    assert_eq!(interpret("Part[a -> b, -1]").unwrap(), "b");
  }

  #[test]
  fn part_rule_delayed_second() {
    // Part 2 of a RuleDelayed gives the replacement
    assert_eq!(interpret("Part[a :> b, 2]").unwrap(), "b");
  }

  #[test]
  fn part_rule_delayed_head() {
    assert_eq!(interpret("Part[a :> b, 0]").unwrap(), "RuleDelayed");
  }

  #[test]
  fn part_atom_head() {
    // Part[atom, 0] returns the Head of the atom (fixes #88)
    assert_eq!(interpret("Part[False, 0]").unwrap(), "Symbol");
    assert_eq!(interpret("Part[True, 0]").unwrap(), "Symbol");
    assert_eq!(interpret("Part[x, 0]").unwrap(), "Symbol");
    assert_eq!(interpret("Part[42, 0]").unwrap(), "Integer");
    assert_eq!(interpret("Part[3.14, 0]").unwrap(), "Real");
    assert_eq!(interpret("(True && False)[[0]]").unwrap(), "Symbol");
    assert_eq!(interpret("False[[0]]").unwrap(), "Symbol");
  }

  #[test]
  fn part_string_is_atom() {
    // Strings are atoms in Wolfram Language — Part[string, n] for n ≠ 0
    // returns unevaluated (matching wolframscript behavior)
    assert_eq!(interpret(r#"Part["Alice", 0]"#).unwrap(), "String");
    assert_eq!(interpret(r#"Part["Alice", 2]"#).unwrap(), r#""Alice"[[2]]"#);
    // Multi-index Part that hits a string at intermediate depth:
    // Part spec is deeper than the object, so the entire Part stays unevaluated
    assert_eq!(
      interpret(r#"x = {{"Alice", 95}, {"Bob", 82}}; x[[1, 1, 2]]"#).unwrap(),
      r#"{{"Alice", 95}, {"Bob", 82}}[[1,1,2]]"#
    );
  }

  #[test]
  fn part_multi_index_on_nested_list() {
    // Multi-index Part on a 3-level deep structure
    assert_eq!(
      interpret(
        "x = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}; {x[[1, 1, 2]], x[[1, 2, 2]]}"
      )
      .unwrap(),
      "{2, 4}"
    );
  }

  #[test]
  fn part_all_from_list_of_rules() {
    // Extracting second element from each rule in a list
    assert_eq!(
      interpret("x = {a -> 1, b -> 2, c -> 3}; x[[All, 2]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn part_all_first_from_list_of_rules() {
    // Extracting first element (pattern) from each rule in a list
    assert_eq!(
      interpret("x = {a -> 1, b -> 2, c -> 3}; x[[All, 1]]").unwrap(),
      "{a, b, c}"
    );
  }
}

mod part_span {
  use super::*;

  #[test]
  fn part_span_basic_bracket_syntax() {
    assert_eq!(interpret("{a, b, c, d, e}[[2;;4]]").unwrap(), "{b, c, d}");
  }

  #[test]
  fn part_span_implicit_end() {
    assert_eq!(
      interpret("Part[{a, b, c, d, e}, 2;;]").unwrap(),
      "{b, c, d, e}"
    );
  }

  #[test]
  fn part_span_implicit_start_and_end() {
    assert_eq!(
      interpret("Part[{a, b, c, d, e}, ;;]").unwrap(),
      "{a, b, c, d, e}"
    );
  }

  #[test]
  fn part_span_negative_end() {
    assert_eq!(
      interpret("Part[{a, b, c, d, e}, ;;-2]").unwrap(),
      "{a, b, c, d}"
    );
  }

  #[test]
  fn part_span_negative_step() {
    assert_eq!(
      interpret("Part[{a, b, c, d, e}, 5;;2;;-1]").unwrap(),
      "{e, d, c, b}"
    );
  }

  #[test]
  fn part_span_on_function_call_preserves_head() {
    assert_eq!(interpret("Part[f[a, b, c, d], 2;;3]").unwrap(), "f[b, c]");
  }

  #[test]
  fn part_span_nested_with_all() {
    assert_eq!(
      interpret("Part[{{a, b, c}, {d, e, f}}, All, 2;;3]").unwrap(),
      "{{b, c}, {e, f}}"
    );
  }

  #[test]
  fn part_span_invalid_range_stays_unevaluated() {
    assert_eq!(
      interpret("Part[{a, b, c, d, e}, 5;;2]").unwrap(),
      "{a, b, c, d, e}[[5 ;; 2]]"
    );
  }

  #[test]
  fn part_span_invalid_range_with_step_stays_unevaluated() {
    assert_eq!(
      interpret("Part[{a, b, c, d, e}, 5;;2;;1]").unwrap(),
      "{a, b, c, d, e}[[5 ;; 2 ;; 1]]"
    );
  }

  #[test]
  fn part_span_implicit_start_end_with_step() {
    // Regression: ";; ;; 2" (implicit start and end with step) used to fail
    // because whitespace skipping caused `!(";"|"=")` lookahead to falsely
    // reject the first `;;`.
    assert_eq!(
      interpret("Range[10][[;; ;; 2]]").unwrap(),
      "{1, 3, 5, 7, 9}"
    );
  }

  #[test]
  fn part_span_implicit_start_end_with_step_no_spaces() {
    // Regression: ";;;;2" used to fail for the same reason.
    assert_eq!(interpret("Range[10][[;;;;2]]").unwrap(), "{1, 3, 5, 7, 9}");
  }

  #[test]
  fn part_span_start_implicit_end_with_step() {
    // Regression: "1 ;; ;; 2" (explicit start, implicit end, step) used to fail.
    assert_eq!(
      interpret("Range[10][[1 ;; ;; 2]]").unwrap(),
      "{1, 3, 5, 7, 9}"
    );
  }

  // ── Adjacent-reverse empty span  —  wolframscript semantics ────────
  #[test]
  fn part_adjacent_reverse_span_returns_empty_list() {
    // `{a, b, c, d, e}[[-1;;-2]]` normalizes to `5;;4` — end == start - 1 —
    // which wolframscript treats as an empty result.
    assert_eq!(interpret("{a, b, c, d, e}[[-1;;-2]]").unwrap(), "{}");
    assert_eq!(interpret("{a, b, c, d, e}[[4;;3]]").unwrap(), "{}");
  }

  #[test]
  fn part_adjacent_reverse_span_folds_empty_plus_to_zero() {
    // `Plus[a, b, c, d][[-1;;-2]]` returns `Plus[]`, which folds to `0`.
    assert_eq!(interpret("(a+b+c+d)[[-1;;-2]]").unwrap(), "0");
  }

  #[test]
  fn part_adjacent_reverse_span_folds_empty_times_to_one() {
    assert_eq!(interpret("(a*b*c)[[-1;;-2]]").unwrap(), "1");
  }
}

mod join_level_spec {
  use super::*;

  #[test]
  fn level_1_is_default() {
    assert_eq!(
      interpret("Join[{a, b}, {c, d}, 1]").unwrap(),
      "{a, b, c, d}"
    );
  }

  #[test]
  fn level_2_basic() {
    assert_eq!(
      interpret("Join[{{a, b}, {c, d}}, {{e, f}, {g, h}}, 2]").unwrap(),
      "{{a, b, e, f}, {c, d, g, h}}"
    );
  }

  #[test]
  fn level_2_ragged() {
    assert_eq!(
      interpret("Join[{{1, 2}, {3}}, {{4, 5}, {6, 7}}, 2]").unwrap(),
      "{{1, 2, 4, 5}, {3, 6, 7}}"
    );
  }

  #[test]
  fn level_2_unequal_outer_lengths() {
    // When second list has fewer rows, only available rows are joined
    assert_eq!(
      interpret("Join[{{1, 2}, {3, 4}}, {{5, 6}}, 2]").unwrap(),
      "{{1, 2, 5, 6}, {3, 4}}"
    );
  }

  #[test]
  fn level_2_asymmetric_inner() {
    assert_eq!(
      interpret("Join[{{a, b}, {c, d}}, {{e}, {f}}, 2]").unwrap(),
      "{{a, b, e}, {c, d, f}}"
    );
  }

  #[test]
  fn level_3() {
    assert_eq!(
      interpret("Join[{{{a}}, {{b}}}, {{{c}}, {{d}}}, 3]").unwrap(),
      "{{{a, c}}, {{b, d}}}"
    );
  }

  #[test]
  fn level_1_numeric() {
    assert_eq!(
      interpret("Join[{1, 2, 3}, {4, 5, 6}, 1]").unwrap(),
      "{1, 2, 3, 4, 5, 6}"
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
  fn join_mismatched_heads_stays_unevaluated() {
    // Different heads (Plus vs Times) can't be joined — wolframscript emits
    // Join::heads and returns the original call unchanged.
    assert_eq!(interpret("Join[a + b, c * d]").unwrap(), "Join[a + b, c*d]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Join::heads: Heads Plus and Times at positions 1 and 2 are expected to be the same."
      )),
      "expected Join::heads message, got {:?}",
      msgs
    );
  }

  #[test]
  fn join_heads_position_reported() {
    // Mismatched head at the third argument should be reported as
    // "positions 1 and 3", matching wolframscript.
    assert_eq!(
      interpret("Join[a + b, c + d, e * f]").unwrap(),
      "Join[a + b, c + d, e*f]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Join::heads: Heads Plus and Times at positions 1 and 3 are expected to be the same."
      )),
      "expected position 3 in Join::heads message, got {:?}",
      msgs
    );
  }

  #[test]
  fn join_list_vs_plus_emits_heads() {
    // Joining a Plus with a List must report the symbolic heads.
    assert_eq!(
      interpret("Join[a + b, {1, 2}]").unwrap(),
      "Join[a + b, {1, 2}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Join::heads: Heads Plus and List at positions 1 and 2 are expected to be the same."
      )),
      "expected Plus/List heads message, got {:?}",
      msgs
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
  fn riffle_every_n_simple() {
    assert_eq!(
      interpret("Riffle[{a, b, c, d, e, f, g}, x, 3]").unwrap(),
      "{a, b, x, c, d, x, e, f, x, g}"
    );
  }

  #[test]
  fn riffle_every_n_no_trailing_after_exhaustion() {
    // Length 6 with n=3: insert only where list still has elements
    assert_eq!(
      interpret("Riffle[{a, b, c, d, e, f}, x, 3]").unwrap(),
      "{a, b, x, c, d, x, e, f}"
    );
  }

  #[test]
  fn riffle_every_n_short_list() {
    assert_eq!(
      interpret("Riffle[{a, b, c, d}, x, 3]").unwrap(),
      "{a, b, x, c, d}"
    );
  }

  #[test]
  fn riffle_every_n_with_list_separator_cycles() {
    assert_eq!(
      interpret("Riffle[{a, b, c, d, e, f, g, h}, {x, y}, 3]").unwrap(),
      "{a, b, x, c, d, y, e, f, x, g, h}"
    );
  }

  #[test]
  fn riffle_triple_spec_negative_end() {
    // {a, -1, s} - b=-1 allows one trailing insert at end of output
    assert_eq!(
      interpret("Riffle[{a, b, c, d, e, f, g, h}, x, {3, -1, 3}]").unwrap(),
      "{a, b, x, c, d, x, e, f, x, g, h, x}"
    );
  }

  #[test]
  fn riffle_triple_spec_negative_end_short() {
    assert_eq!(
      interpret("Riffle[{a, b, c, d}, x, {3, -1, 3}]").unwrap(),
      "{a, b, x, c, d, x}"
    );
  }

  #[test]
  fn riffle_triple_spec_positive_end() {
    assert_eq!(
      interpret("Riffle[{a, b, c, d, e, f, g}, x, {3, 8, 3}]").unwrap(),
      "{a, b, x, c, d, x, e, f, g}"
    );
  }

  #[test]
  fn riffle_triple_spec_every_other() {
    assert_eq!(
      interpret("Riffle[{a, b, c, d, e, f}, x, {2, -1, 2}]").unwrap(),
      "{a, x, b, x, c, x, d, x, e, x, f, x}"
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
  fn thread_comparison_geq() {
    // Thread[{a, b, c} >= 0] should produce {a >= 0, b >= 0, c >= 0}
    assert_eq!(
      interpret("Thread[{a, b, c} >= 0]").unwrap(),
      "{a >= 0, b >= 0, c >= 0}"
    );
  }

  #[test]
  fn thread_comparison_leq() {
    assert_eq!(
      interpret("Thread[{a, b} <= 5]").unwrap(),
      "{a <= 5, b <= 5}"
    );
  }

  #[test]
  fn thread_comparison_array_vars() {
    // Thread with Array-style variables
    assert_eq!(
      interpret("vars = Array[n, 3]; Thread[vars >= 0]").unwrap(),
      "{n[1] >= 0, n[2] >= 0, n[3] >= 0}"
    );
  }

  #[test]
  fn thread_rule() {
    // Thread[{a,b} -> {c,d}] should produce {a -> c, b -> d}
    assert_eq!(
      interpret("Thread[{a, b} -> {c, d}]").unwrap(),
      "{a -> c, b -> d}"
    );
  }

  #[test]
  fn thread_rule_delayed() {
    assert_eq!(
      interpret("Thread[{a, b} :> {c, d}]").unwrap(),
      "{a :> c, b :> d}"
    );
  }

  #[test]
  fn thread_rule_lhs_only() {
    // Thread[{a,b} -> c] should produce {a -> c, b -> c}
    assert_eq!(
      interpret("Thread[{a, b} -> c]").unwrap(),
      "{a -> c, b -> c}"
    );
  }

  #[test]
  fn thread_rule_rhs_only() {
    // Thread[a -> {c,d}] should produce {a -> c, a -> d}
    assert_eq!(
      interpret("Thread[a -> {c, d}]").unwrap(),
      "{a -> c, a -> d}"
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
  fn levi_civita_tensor_sparse_2() {
    assert_eq!(
      interpret("LeviCivitaTensor[2]").unwrap(),
      "SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, {1, -1}}]"
    );
  }

  #[test]
  fn levi_civita_tensor_sparse_3() {
    assert_eq!(
      interpret("LeviCivitaTensor[3]").unwrap(),
      "SparseArray[Automatic, {3, 3, 3}, 0, {1, {{0, 2, 4, 6}, {{2, 3}, {3, 2}, {1, 3}, {3, 1}, {1, 2}, {2, 1}}}, {1, -1, -1, 1, 1, -1}}]"
    );
  }

  #[test]
  fn levi_civita_tensor_normalize_matches_list() {
    assert_eq!(
      interpret("Normal[LeviCivitaTensor[3]]").unwrap(),
      interpret("LeviCivitaTensor[3, List]").unwrap()
    );
  }

  // Normal unwraps NumericArray / ByteArray to their underlying list.
  #[test]
  fn normal_of_numeric_array_unwraps_payload() {
    assert_eq!(
      interpret("Normal[NumericArray[{{1, 2}, {3, 4}}]]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  #[test]
  fn normal_of_byte_array_unwraps_payload() {
    assert_eq!(interpret("Normal[ByteArray[{4, 2}]]").unwrap(), "{4, 2}");
  }

  // `Order` on ByteArrays compares the decoded byte payload, not the
  // wrapping `ByteArray["<base64>"]` string. wolframscript:
  //   Order[ByteArray[{1, 99}], ByteArray[{2, 0}]] = 1
  // because the first byte 1 < 2, even though "AWM=" > "AgA=" lexically.
  #[test]
  fn order_byte_array_first_byte_smaller() {
    assert_eq!(
      interpret("Order[ByteArray[{1, 99}], ByteArray[{2, 0}]]").unwrap(),
      "1"
    );
  }

  #[test]
  fn order_byte_array_first_byte_larger() {
    assert_eq!(
      interpret("Order[ByteArray[{2, 0}], ByteArray[{1, 99}]]").unwrap(),
      "-1"
    );
  }

  #[test]
  fn order_byte_array_equal() {
    assert_eq!(
      interpret("Order[ByteArray[{1, 99}], ByteArray[{1, 99}]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn order_byte_array_prefix_sorts_first() {
    // [1, 2] is a prefix of [1, 2, 3] so it sorts first → Order = 1.
    assert_eq!(
      interpret("Order[ByteArray[{1, 2}], ByteArray[{1, 2, 3}]]").unwrap(),
      "1"
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
  fn apply_integer_level_is_one_to_n() {
    // Apply[f, expr, n] is equivalent to Apply[f, expr, {1, n}].
    assert_eq!(
      interpret("Apply[Plus, {{1, 2}, {3, 4}}, 1]").unwrap(),
      "{3, 7}"
    );
    // n == 0 → empty level range → no replacement.
    assert_eq!(
      interpret("Apply[f, {{a, b}, {c, d}}, 0]").unwrap(),
      "{{a, b}, {c, d}}"
    );
    assert_eq!(
      interpret("Apply[f, {{a, b}, {c, d}}, 2]").unwrap(),
      "{f[a, b], f[c, d]}"
    );
  }

  #[test]
  fn apply_infinity_level() {
    assert_eq!(
      interpret("Apply[Plus, {{1, 2, 3}, {4, 5, 6}}, Infinity]").unwrap(),
      "{6, 15}"
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
  fn pad_right_cyclic_alignment() {
    // Cyclic padding aligns with position 0, not after the list end
    assert_eq!(
      interpret("PadRight[{1, 2, 3}, 7, {a, b}]").unwrap(),
      "{1, 2, 3, b, a, b, a}"
    );
    assert_eq!(
      interpret("PadRight[{1, 2}, 6, {a, b, c}]").unwrap(),
      "{1, 2, c, a, b, c}"
    );
  }

  #[test]
  fn pad_left_cyclic_alignment() {
    assert_eq!(
      interpret("PadLeft[{1, 2, 3}, 7, {a, b}]").unwrap(),
      "{b, a, b, a, 1, 2, 3}"
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
  fn flatten_non_list_head() {
    // Flatten[f[f[a, b], f[c, d]]] flattens subexpressions with the same head
    assert_eq!(
      interpret("Flatten[f[f[a, b], f[c, d]]]").unwrap(),
      "f[a, b, c, d]"
    );
    // Nested same head
    assert_eq!(
      interpret("Flatten[f[f[a, f[b, c]], d]]").unwrap(),
      "f[a, b, c, d]"
    );
    // Only flattens matching heads
    assert_eq!(
      interpret("Flatten[f[g[a, b], f[c, d]]]").unwrap(),
      "f[g[a, b], c, d]"
    );
  }

  #[test]
  fn flatten_non_list_head_with_level() {
    // Flatten[f[f[a, b], f[c, d]], 1] flattens one level
    assert_eq!(
      interpret("Flatten[f[f[a, b], f[c, d]], 1]").unwrap(),
      "f[a, b, c, d]"
    );
    // Flatten[f[f[a, f[b, c]], d], 1] flattens only one level
    assert_eq!(
      interpret("Flatten[f[f[a, f[b, c]], d], 1]").unwrap(),
      "f[a, f[b, c], d]"
    );
  }

  #[test]
  fn permutations_up_to_length() {
    assert_eq!(
      interpret("Permutations[{1, 2, 3}, 2]").unwrap(),
      "{{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}"
    );
  }

  #[test]
  fn permutations_length_range() {
    // Permutations[list, {kmin, kmax}] gives permutations of lengths
    // kmin through kmax inclusive.
    assert_eq!(
      interpret("Permutations[{1, 2, 3}, {1, 2}]").unwrap(),
      "{{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}"
    );
    assert_eq!(
      interpret("Permutations[{1, 2, 3}, {2, 3}]").unwrap(),
      "{{1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}, \
       {1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}}"
    );
    assert_eq!(
      interpret("Permutations[{1, 2, 3}, {0, 1}]").unwrap(),
      "{{}, {1}, {2}, {3}}"
    );
  }

  #[test]
  fn permutations_with_duplicates() {
    // Permutations of a multiset should return only distinct permutations.
    // Wolfram: Permutations[{1, 1, 2}] -> {{1, 1, 2}, {1, 2, 1}, {2, 1, 1}}
    assert_eq!(
      interpret("Permutations[{1, 1, 2}]").unwrap(),
      "{{1, 1, 2}, {1, 2, 1}, {2, 1, 1}}"
    );
  }

  #[test]
  fn permutations_with_two_pairs_of_duplicates() {
    // Wolfram: Permutations[{a, a, b, b}] ->
    //   {{a, a, b, b}, {a, b, a, b}, {a, b, b, a},
    //    {b, a, a, b}, {b, a, b, a}, {b, b, a, a}}
    assert_eq!(
      interpret("Permutations[{a, a, b, b}]").unwrap(),
      "{{a, a, b, b}, {a, b, a, b}, {a, b, b, a}, \
       {b, a, a, b}, {b, a, b, a}, {b, b, a, a}}"
    );
  }

  #[test]
  fn permutations_k_with_duplicates() {
    // Permutations[{1, 1, 2}, {2}] should also dedupe.
    // Wolfram: -> {{1, 1}, {1, 2}, {2, 1}}
    assert_eq!(
      interpret("Permutations[{1, 1, 2}, {2}]").unwrap(),
      "{{1, 1}, {1, 2}, {2, 1}}"
    );
  }

  #[test]
  fn permutations_up_to_length_with_duplicates() {
    // Permutations[{1, 1, 2}, 2] -> distinct perms of length 0, 1, 2.
    // Wolfram: -> {{}, {1}, {2}, {1, 1}, {1, 2}, {2, 1}}
    assert_eq!(
      interpret("Permutations[{1, 1, 2}, 2]").unwrap(),
      "{{}, {1}, {2}, {1, 1}, {1, 2}, {2, 1}}"
    );
  }

  #[test]
  fn flatten_dim_spec_transpose() {
    assert_eq!(
      interpret("Flatten[{{a, b}, {c, d}}, {{2}, {1}}]").unwrap(),
      "{{a, c}, {b, d}}"
    );
  }

  #[test]
  fn flatten_dim_spec_merge() {
    assert_eq!(
      interpret("Flatten[{{a, b}, {c, d}}, {{1, 2}}]").unwrap(),
      "{a, b, c, d}"
    );
  }

  #[test]
  fn flatten_flat_level_list_merges_levels() {
    // A flat {1, 2} is equivalent to {{1, 2}} — merges levels 1 and 2.
    assert_eq!(
      interpret("Flatten[{{1, 2}, {3, 4}}, {1, 2}]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn flatten_dim_spec_ragged_transpose() {
    assert_eq!(
      interpret("Flatten[{{1, 2, 3}, {4}, {6, 7}, {8, 9, 10}}, {{2}, {1}}]")
        .unwrap(),
      "{{1, 4, 6, 8}, {2, 7, 9}, {3, 10}}"
    );
  }

  #[test]
  fn pad_left_ragged_array() {
    assert_eq!(
      interpret("PadLeft[{{}, {1, 2}, {1, 2, 3}}]").unwrap(),
      "{{0, 0, 0}, {0, 1, 2}, {1, 2, 3}}"
    );
  }

  #[test]
  fn pad_right_ragged_array() {
    assert_eq!(
      interpret("PadRight[{{}, {1, 2}, {1, 2, 3}}]").unwrap(),
      "{{0, 0, 0}, {1, 2, 0}, {1, 2, 3}}"
    );
  }

  #[test]
  fn pad_left_multidim() {
    assert_eq!(
      interpret("PadLeft[{{1, 2}, {3, 4}}, {3, 3}]").unwrap(),
      "{{0, 0, 0}, {0, 1, 2}, {0, 3, 4}}"
    );
    assert_eq!(
      interpret("PadLeft[{{1, 2}, {3, 4}}, {3, 3}, x]").unwrap(),
      "{{x, x, x}, {x, 1, 2}, {x, 3, 4}}"
    );
    assert_eq!(
      interpret("PadLeft[{1, 2, 3}, {5}]").unwrap(),
      "{0, 0, 1, 2, 3}"
    );
    assert_eq!(
      interpret("PadLeft[{{{1}}}, {2, 2, 2}, 0]").unwrap(),
      "{{{0, 0}, {0, 0}}, {{0, 0}, {0, 1}}}"
    );
  }

  // 4-arg PadLeft with per-dimension margin `m`. Source is positioned so
  // its last element ends at index `n - 1 - m` in each dimension; entries
  // that fall outside `[0, n)` are dropped and gaps are filled with the
  // pad shape. Regression for the mathics list/rearrange.py PadLeft
  // doctest and verified against wolframscript.
  #[test]
  fn pad_left_multidim_scalar_margin() {
    assert_eq!(
      interpret("PadLeft[{{1, 2, 3}}, {5, 2}, x, 1]").unwrap(),
      "{{x, x}, {x, x}, {x, x}, {3, x}, {x, x}}"
    );
  }

  #[test]
  fn pad_right_multidim() {
    assert_eq!(
      interpret("PadRight[{{1, 2}, {3, 4}}, {3, 3}]").unwrap(),
      "{{1, 2, 0}, {3, 4, 0}, {0, 0, 0}}"
    );
    assert_eq!(
      interpret("PadRight[{{1, 2}, {3, 4}}, {4, 4}, 9]").unwrap(),
      "{{1, 2, 9, 9}, {3, 4, 9, 9}, {9, 9, 9, 9}, {9, 9, 9, 9}}"
    );
  }

  // 4-arg PadRight with a per-dimension margin `m`: the source is placed
  // at offset `m` in each dimension, with gaps filled by the pad shape.
  // Verified against wolframscript. Regression for the mathics
  // list/rearrange.py PadRight doctest.
  #[test]
  fn pad_right_multidim_scalar_margin() {
    assert_eq!(
      interpret("PadRight[{{1, 2, 3}}, {5, 2}, x, 1]").unwrap(),
      "{{x, x}, {x, 1}, {x, x}, {x, x}, {x, x}}"
    );
  }

  #[test]
  fn pad_right_multidim_list_margin() {
    assert_eq!(
      interpret("PadRight[{{1, 2, 3}}, {5, 2}, x, {1, 0}]").unwrap(),
      "{{x, x}, {1, 2}, {x, x}, {x, x}, {x, x}}"
    );
  }

  #[test]
  fn string_position_with_limit() {
    assert_eq!(
      interpret("StringPosition[\"123ABCxyABCzzzABCABC\", \"ABC\", 2]")
        .unwrap(),
      "{{4, 6}, {9, 11}}"
    );
  }

  #[test]
  fn array_multi_dim() {
    assert_eq!(
      interpret("Array[f, {2, 3}]").unwrap(),
      "{{f[1, 1], f[1, 2], f[1, 3]}, {f[2, 1], f[2, 2], f[2, 3]}}"
    );
  }

  #[test]
  fn array_multi_dim_offset() {
    assert_eq!(
      interpret("Array[f, {2, 3}, 3]").unwrap(),
      "{{f[3, 3], f[3, 4], f[3, 5]}, {f[4, 3], f[4, 4], f[4, 5]}}"
    );
  }

  #[test]
  fn array_multi_dim_per_dim_offset() {
    assert_eq!(
      interpret("Array[f, {2, 3}, {4, 6}]").unwrap(),
      "{{f[4, 6], f[4, 7], f[4, 8]}, {f[5, 6], f[5, 7], f[5, 8]}}"
    );
  }

  #[test]
  fn array_multi_dim_with_head() {
    assert_eq!(
      interpret("Array[f, {2, 3}, 1, Plus]").unwrap(),
      "f[1, 1] + f[1, 2] + f[1, 3] + f[2, 1] + f[2, 2] + f[2, 3]"
    );
  }

  #[test]
  fn array_multi_dim_with_list_head() {
    assert_eq!(
      interpret("Array[Plus, {2, 3}, 1, List]").unwrap(),
      "{{2, 3, 4}, {3, 4, 5}}"
    );
  }

  #[test]
  fn array_multi_dim_with_custom_head() {
    assert_eq!(
      interpret("Array[f, {2, 3}, 1, g]").unwrap(),
      "g[g[f[1, 1], f[1, 2], f[1, 3]], g[f[2, 1], f[2, 2], f[2, 3]]]"
    );
  }

  #[test]
  fn array_multi_dim_plus_head_sums_to_total() {
    assert_eq!(interpret("Array[Plus, {2, 3}, 1, Plus]").unwrap(), "21");
  }

  #[test]
  fn array_plus_multi_dim() {
    assert_eq!(
      interpret("Array[Plus, {3, 2}]").unwrap(),
      "{{2, 3}, {3, 4}, {4, 5}}"
    );
  }

  #[test]
  fn array_with_range() {
    // Regression: Array[f, n, {a, b}] should spread n indices evenly
    // from a to b. Previously this was misinterpreted as per-dim origins.
    assert_eq!(
      interpret("Array[f, 5, {2, 10}]").unwrap(),
      "{f[2], f[4], f[6], f[8], f[10]}"
    );
  }

  #[test]
  fn array_with_range_single_value() {
    // Edge case: Array[f, 1, {a, b}] uses the midpoint (a + b)/2,
    // matching wolframscript: Array[f, 1, {5, 10}] → {f[15/2]}.
    assert_eq!(interpret("Array[f, 1, {5, 10}]").unwrap(), "{f[15/2]}");
  }

  #[test]
  fn array_with_range_rational_step() {
    // Array[f, n, {a, b}] with non-integer step must produce exact
    // rationals, not Reals. Previously gave {f[1.], f[1.5], f[2.]}.
    assert_eq!(
      interpret("Array[f, 3, {1, 2}]").unwrap(),
      "{f[1], f[3/2], f[2]}"
    );
  }

  #[test]
  fn array_with_range_fifths() {
    assert_eq!(
      interpret("Array[f, 5, {0, 1}]").unwrap(),
      "{f[0], f[1/4], f[1/2], f[3/4], f[1]}"
    );
  }

  #[test]
  fn array_with_range_real_endpoints_stays_real() {
    // When endpoints are Real, the indices should also be Real.
    assert_eq!(
      interpret("Array[f, 3, {1.5, 2.5}]").unwrap(),
      "{f[1.5], f[2.], f[2.5]}"
    );
  }

  #[test]
  fn to_character_code_list() {
    assert_eq!(
      interpret("ToCharacterCode[{\"ab\", \"c\"}]").unwrap(),
      "{{97, 98}, {99}}"
    );
  }

  #[test]
  fn from_character_code_nested() {
    assert_eq!(
      interpret("FromCharacterCode[{{97, 98, 99}, {100, 101, 102}}]").unwrap(),
      "{abc, def}"
    );
  }

  #[test]
  fn accumulate_symbolic() {
    assert_eq!(
      interpret("Accumulate[{a, b, c}]").unwrap(),
      "{a, a + b, a + b + c}"
    );
  }

  #[test]
  fn accumulate_preserves_arbitrary_head() {
    // Accumulate threads through any head (not just List), preserving it.
    assert_eq!(
      interpret("Accumulate[g[1, 2, 3, 4]]").unwrap(),
      "g[1, 3, 6, 10]"
    );
  }

  #[test]
  fn accumulate_preserves_arbitrary_head_symbolic() {
    assert_eq!(
      interpret("Accumulate[f[a, b, c, d]]").unwrap(),
      "f[a, a + b, a + b + c, a + b + c + d]"
    );
  }

  #[test]
  fn accumulate_through_hold_head() {
    // Even held expressions thread through.
    assert_eq!(
      interpret("Accumulate[Hold[1, 2, 3]]").unwrap(),
      "Hold[1, 3, 6]"
    );
  }

  #[test]
  fn accumulate_through_times_head() {
    // Times head: cumulative sums recombined under Times.
    assert_eq!(
      interpret("Accumulate[Times[a, b, c]]").unwrap(),
      "a*(a + b)*(a + b + c)"
    );
  }

  #[test]
  fn differences_higher_order() {
    assert_eq!(
      interpret("Differences[{1, 4, 9, 16, 25}, 2]").unwrap(),
      "{2, 2, 2}"
    );
  }

  #[test]
  fn differences_with_level_list_single() {
    // `{n}` spec equals the scalar n form.
    assert_eq!(
      interpret("Differences[{1, 4, 9, 16, 25, 36}, {2}]").unwrap(),
      "{2, 2, 2, 2}"
    );
  }

  #[test]
  fn differences_with_level_list_matrix_row_only() {
    // {0, 1}: no differences on rows, 1st difference along columns.
    assert_eq!(
      interpret("Differences[{{1, 2, 3, 4}, {5, 7, 9, 11}}, {0, 1}]").unwrap(),
      "{{1, 1, 1}, {2, 2, 2}}"
    );
  }

  #[test]
  fn differences_with_level_list_matrix_both() {
    // {1, 1}: 1st difference along rows then along columns.
    assert_eq!(
      interpret("Differences[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1}]")
        .unwrap(),
      "{{0, 0}, {0, 0}}"
    );
  }

  #[test]
  fn clip_three_args() {
    assert_eq!(interpret("Clip[0.5, {0, 1}, {-1, 1}]").unwrap(), "0.5");
    assert_eq!(interpret("Clip[-0.5, {0, 1}, {-1, 1}]").unwrap(), "-1");
    assert_eq!(interpret("Clip[1.5, {0, 1}, {-1, 1}]").unwrap(), "1");
  }

  #[test]
  fn delete_non_list_head() {
    assert_eq!(interpret("Delete[f[a, b, c, d], 3]").unwrap(), "f[a, b, d]");
  }

  #[test]
  fn delete_position_zero() {
    // Delete[{a, b, c}, 0] removes the head List, returning Sequence[a, b, c]
    // which at top level displays as concatenated elements (matches Wolfram)
    assert_eq!(interpret("Delete[{a, b, c}, 0]").unwrap(), "abc");
  }

  #[test]
  fn map_level_spec() {
    assert_eq!(
      interpret("Map[f, {{a, b}, {c, d, e}}, {2}]").unwrap(),
      "{{f[a], f[b]}, {f[c], f[d], f[e]}}"
    );
  }

  #[test]
  fn map_heads_true() {
    assert_eq!(
      interpret("Map[f, a + b + c, Heads->True]").unwrap(),
      "f[Plus][f[a], f[b], f[c]]"
    );
  }

  #[test]
  fn map_level_range() {
    assert_eq!(
      interpret("Map[f, {{a, b}, {c, d, e}}, 2]").unwrap(),
      "{f[{f[a], f[b]}], f[{f[c], f[d], f[e]}]}"
    );
  }

  #[test]
  fn map_negative_level_spec() {
    // {-1} maps f to atoms (leaves)
    assert_eq!(
      interpret("Map[f, {1, {2, {3}}}, {-1}]").unwrap(),
      "{f[1], {f[2], {f[3]}}}"
    );
    // {-1} on a flat list
    assert_eq!(
      interpret("Map[f, {1, 2, 3}, {-1}]").unwrap(),
      "{f[1], f[2], f[3]}"
    );
    // {-2, -1} maps to atoms and expressions containing only atoms
    assert_eq!(
      interpret("Map[f, {1, {2, {3}}}, {-2, -1}]").unwrap(),
      "{f[1], {f[2], f[{f[3]}]}}"
    );
  }

  #[test]
  fn map_mixed_level_range() {
    // {0, -1} means all levels
    assert_eq!(
      interpret("Map[f, {1, {2, {3}}}, {0, -1}]").unwrap(),
      "f[{f[1], f[{f[2], f[{f[3]}]}]}]"
    );
  }

  #[test]
  fn map_infinity_level() {
    // Infinity means levels 1 through Infinity (all subexpressions)
    assert_eq!(
      interpret("Map[f, {1, {2, {3}}}, Infinity]").unwrap(),
      "{f[1], f[{f[2], f[{f[3]}]}]}"
    );
    // {0, Infinity} includes level 0 (the whole expression)
    assert_eq!(
      interpret("Map[f, {1, {2, {3}}}, {0, Infinity}]").unwrap(),
      "f[{f[1], f[{f[2], f[{f[3]}]}]}]"
    );
  }

  #[test]
  fn map_operator_form() {
    assert_eq!(
      interpret("Map[f][{1, 2, 3}]").unwrap(),
      "{f[1], f[2], f[3]}"
    );
  }

  #[test]
  fn sort_by_operator_form() {
    assert_eq!(
      interpret("SortBy[Last][{{3, 1}, {1, 2}}]").unwrap(),
      "{{3, 1}, {1, 2}}"
    );
  }

  #[test]
  fn maximal_by_operator_form() {
    assert_eq!(
      interpret("MaximalBy[Last][{{3, 1}, {1, 2}, {2, 5}}]").unwrap(),
      "{{2, 5}}"
    );
  }

  #[test]
  fn minimal_by_operator_form() {
    assert_eq!(
      interpret("MinimalBy[Last][{{3, 1}, {1, 2}, {2, 5}}]").unwrap(),
      "{{3, 1}}"
    );
  }

  #[test]
  fn group_by_operator_form() {
    assert_eq!(
      interpret("GroupBy[Sign][{1, -1, 2, -2, 3}]").unwrap(),
      "<|1 -> {1, 2, 3}, -1 -> {-1, -2}|>"
    );
  }

  #[test]
  fn group_by_with_reducer() {
    assert_eq!(
      interpret("GroupBy[{1, 2, 3, 4, 5, 6}, EvenQ, Total]").unwrap(),
      "<|False -> 9, True -> 12|>"
    );
  }

  #[test]
  fn group_by_with_length_reducer() {
    assert_eq!(
      interpret("GroupBy[{1, 2, 3, 4, 5, 6}, EvenQ, Length]").unwrap(),
      "<|False -> 3, True -> 3|>"
    );
  }

  #[test]
  fn counts_by_operator_form() {
    assert_eq!(
      interpret("CountsBy[Sign][{1, -1, 2, -2, 3}]").unwrap(),
      "<|1 -> 3, -1 -> 2|>"
    );
  }

  #[test]
  fn cases_operator_form() {
    assert_eq!(
      interpret("Cases[_Integer][{1, \"a\", 2, \"b\"}]").unwrap(),
      "{1, 2}"
    );
  }

  #[test]
  fn cases_repeated_pattern_variable() {
    // Regression: Cases used to ignore bindings so all pairs matched.
    assert_eq!(
      interpret("Cases[{{1, 2}, {2, 1}, {1, 1}, {3, 3}}, {a_, a_}]").unwrap(),
      "{{1, 1}, {3, 3}}"
    );
  }

  #[test]
  fn member_q_operator_form() {
    assert_eq!(interpret("MemberQ[2][{1, 2, 3}]").unwrap(), "True");
  }

  #[test]
  fn free_q_operator_form() {
    assert_eq!(interpret("FreeQ[_Integer][{1, 2, x}]").unwrap(), "False");
  }

  #[test]
  fn replace_part_deep() {
    assert_eq!(
      interpret("ReplacePart[{{a, b}, {c, d}}, {2, 1} -> t]").unwrap(),
      "{{a, b}, {t, d}}"
    );
  }

  #[test]
  fn replace_part_head() {
    assert_eq!(
      interpret("ReplacePart[{a, b, c}, 0 -> Times]").unwrap(),
      "a*b*c"
    );
  }

  #[test]
  fn replace_part_multiple_rules() {
    assert_eq!(
      interpret("ReplacePart[{{a, b}, {c, d}}, {{2, 1} -> t, {1, 1} -> t}]")
        .unwrap(),
      "{{t, b}, {t, d}}"
    );
  }

  #[test]
  fn replace_part_multiple_positions() {
    assert_eq!(
      interpret("ReplacePart[{a, b, c}, {{1}, {2}} -> t]").unwrap(),
      "{t, t, c}"
    );
  }

  #[test]
  fn replace_part_rule_delayed() {
    assert_eq!(
      interpret("n = 0; ReplacePart[{a, b, c, d}, {{1}, {3}} :> n++]").unwrap(),
      "{0, b, 1, d}"
    );
  }

  #[test]
  fn replace_part_out_of_range_returns_original() {
    // Out-of-range positions are silently ignored, matching wolframscript.
    assert_eq!(
      interpret("ReplacePart[{a, b, c}, 4 -> t]").unwrap(),
      "{a, b, c}"
    );
    assert_eq!(
      interpret("ReplacePart[{a, b, c}, -10 -> t]").unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn map_on_power_expression() {
    // Map[f, x^2] applies f to each part of Power[x, 2]
    assert_eq!(interpret("Map[f, x^2]").unwrap(), "f[x]^f[2]");
  }

  #[test]
  fn map_on_function_call() {
    // Map[f, g[a, b, c]] applies f to each argument of g
    assert_eq!(
      interpret("Map[f, g[a, b, c]]").unwrap(),
      "g[f[a], f[b], f[c]]"
    );
  }

  #[test]
  fn map_on_plus_expression() {
    // Map[f, x + y] applies f to each summand
    assert_eq!(interpret("Map[f, x + y]").unwrap(), "f[x] + f[y]");
  }

  #[test]
  fn map_on_atom_unevaluated() {
    // Map on an atom should return the atom unchanged
    assert_eq!(interpret("Map[f, x]").unwrap(), "x");
  }

  #[test]
  fn map_recursive_user_function() {
    // Regression test for issue #68: Map with user-defined recursive function
    assert_eq!(
      interpret(
        "TrigSimplify[expr_] := expr /; AtomQ[expr]\n\
         TrigSimplify[expr_] := expr /; Head[expr] === If\n\
         TrigSimplify[expr_] := Map[TrigSimplify, expr]\n\
         TrigSimplify[x^2]"
      )
      .unwrap(),
      "x^2"
    );
  }

  #[test]
  fn map_at_operator_form() {
    assert_eq!(
      interpret("MapAt[f, -1][{a, b, c}]").unwrap(),
      "{a, b, f[c]}"
    );
  }

  #[test]
  fn normalize_complex() {
    assert_eq!(interpret("Normalize[1 + I]").unwrap(), "(1 + I)/Sqrt[2]");
  }

  #[test]
  fn normalize_with_norm_function() {
    assert_eq!(interpret("Normalize[{3, 4}, Norm]").unwrap(), "{3/5, 4/5}");
  }

  #[test]
  fn normalize_with_custom_norm() {
    assert_eq!(
      interpret("Normalize[{1, 2, 3}, (Max[Abs[#]] &)]").unwrap(),
      "{1/3, 2/3, 1}"
    );
  }

  #[test]
  fn keys_list_of_rules() {
    assert_eq!(interpret("Keys[{a -> x, b -> y}]").unwrap(), "{a, b}");
  }

  #[test]
  fn keys_list_of_rules_order() {
    assert_eq!(
      interpret("Keys[{c -> z, b -> y, a -> x}]").unwrap(),
      "{c, b, a}"
    );
  }

  #[test]
  fn values_list_of_rules() {
    assert_eq!(interpret("Values[{a -> x, b -> y}]").unwrap(), "{x, y}");
  }

  #[test]
  fn values_list_of_rules_order() {
    assert_eq!(
      interpret("Values[{c -> z, b -> y, a -> x}]").unwrap(),
      "{z, y, x}"
    );
  }

  #[test]
  fn first_empty_list() {
    assert_eq!(interpret("First[{}]").unwrap(), "First[{}]");
  }

  #[test]
  fn last_empty_list() {
    assert_eq!(interpret("Last[{}]").unwrap(), "Last[{}]");
  }

  #[test]
  fn rest_empty_list() {
    assert_eq!(interpret("Rest[{}]").unwrap(), "Rest[{}]");
  }

  #[test]
  fn pad_left_cyclic_with_offset() {
    assert_eq!(
      interpret("PadLeft[{1, 2, 3}, 10, {a, b, c}, 2]").unwrap(),
      "{b, c, a, b, c, 1, 2, 3, a, b}"
    );
  }

  #[test]
  fn pad_right_cyclic_with_offset() {
    assert_eq!(
      interpret("PadRight[{1, 2, 3}, 10, {a, b, c}, 2]").unwrap(),
      "{b, c, 1, 2, 3, a, b, c, a, b}"
    );
  }

  #[test]
  fn pad_left_cyclic_no_offset() {
    assert_eq!(
      interpret("PadLeft[{1, 2, 3}, 9, {a, b, c}]").unwrap(),
      "{a, b, c, a, b, c, 1, 2, 3}"
    );
  }

  #[test]
  fn pad_right_cyclic_no_offset() {
    assert_eq!(
      interpret("PadRight[{1, 2, 3}, 9, {a, b, c}]").unwrap(),
      "{1, 2, 3, a, b, c, a, b, c}"
    );
  }

  #[test]
  fn pad_left_negative_offset_drops_tail() {
    assert_eq!(
      interpret("PadLeft[{a, b, c}, 7, 0, -1]").unwrap(),
      "{0, 0, 0, 0, 0, a, b}"
    );
  }

  #[test]
  fn pad_left_negative_offset_drops_more_tail() {
    assert_eq!(
      interpret("PadLeft[{a, b, c}, 7, x, -2]").unwrap(),
      "{x, x, x, x, x, x, a}"
    );
  }

  #[test]
  fn pad_left_positive_offset_drops_head() {
    assert_eq!(
      interpret("PadLeft[{a, b, c}, 7, x, 5]").unwrap(),
      "{b, c, x, x, x, x, x}"
    );
  }

  #[test]
  fn pad_right_negative_offset_drops_head() {
    assert_eq!(
      interpret("PadRight[{a, b, c}, 7, x, -1]").unwrap(),
      "{b, c, x, x, x, x, x}"
    );
  }

  #[test]
  fn pad_right_positive_offset_drops_tail() {
    assert_eq!(
      interpret("PadRight[{a, b, c}, 7, x, 5]").unwrap(),
      "{x, x, x, x, x, a, b}"
    );
  }

  #[test]
  fn pad_right_cyclic_offset_non_divisible_cycle_len() {
    // cycle_len=2 does not divide len=3, so wrong-anchor cycling would
    // produce the wrong result here.
    assert_eq!(
      interpret("PadRight[{a, b, c}, 6, {x, y}, 1]").unwrap(),
      "{y, a, b, c, y, x}"
    );
  }

  #[test]
  fn pad_right_cyclic_offset_len3_cycle2() {
    assert_eq!(
      interpret("PadRight[{a, b, c}, 7, {x, y}, 2]").unwrap(),
      "{x, y, a, b, c, y, x}"
    );
  }
}

mod angle_path {
  use super::*;

  #[test]
  fn empty() {
    assert_eq!(interpret("AnglePath[{}]").unwrap(), "{{0, 0}}");
  }

  #[test]
  fn numeric_angles() {
    let result = interpret("AnglePath[{0.5, -0.3, 0.1}]").unwrap();
    assert!(result.starts_with("{{0., 0.}, {0.877582"));
  }

  #[test]
  fn symbolic_integer_angles() {
    assert_eq!(
      interpret("AnglePath[{1, 1}]").unwrap(),
      "{{0, 0}, {Cos[1], Sin[1]}, {Cos[1] + Cos[2], Sin[1] + Sin[2]}}"
    );
  }

  #[test]
  fn symbolic_variable_angles() {
    // Pure symbolic variables must yield Cos/Sin expressions, matching
    // wolframscript byte-for-byte.
    assert_eq!(
      interpret("AnglePath[{a, b}]").unwrap(),
      "{{0, 0}, {Cos[a], Sin[a]}, {Cos[a] + Cos[a + b], Sin[a] + Sin[a + b]}}"
    );
  }

  #[test]
  fn two_argument_form() {
    // AnglePath[{{x0, y0}, θ0}, steps] starts at {x0,y0} facing θ0.
    assert_eq!(
      interpret(
        "AnglePath[{{1, 1}, 90 Degree}, {{1, 90 Degree}, {2, 90 Degree}, {1, 90 Degree}, {2, 90 Degree}}]"
      )
      .unwrap(),
      "{{1, 1}, {0, 1}, {0, -1}, {1, -1}, {1, 1}}"
    );
  }

  #[test]
  fn step_angle_pairs() {
    let result =
      interpret("AnglePath[{{1, 0.5}, {2, -0.3}, {0.5, 0.1}}]").unwrap();
    assert!(result.starts_with("{{0., 0.}, {0.877582"));
  }

  #[test]
  fn length_is_n_plus_1() {
    assert_eq!(
      interpret("Length[AnglePath[{1.0, 2.0, 3.0}]]").unwrap(),
      "4"
    );
  }

  #[test]
  fn with_random_real() {
    // The primary use case: AnglePath with RandomReal
    assert_eq!(
      interpret("Length[AnglePath[RandomReal[{-1, 1}, {100}]]]").unwrap(),
      "101"
    );
  }
}

mod array_flatten {
  use super::*;

  #[test]
  fn two_by_two_blocks() {
    assert_eq!(
      interpret(
        "ArrayFlatten[{{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}}]"
      )
      .unwrap(),
      "{{1, 2, 5, 6}, {3, 4, 7, 8}, {9, 10, 13, 14}, {11, 12, 15, 16}}"
    );
  }

  #[test]
  fn identity_with_zeros() {
    assert_eq!(
      interpret(
        "ArrayFlatten[{{IdentityMatrix[2], {{0},{0}}}, {{{0, 0}}, {{1}}}}]"
      )
      .unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn scalar_blocks() {
    assert_eq!(
      interpret("ArrayFlatten[{{a, b}, {c, d}}]").unwrap(),
      "{{a, b}, {c, d}}"
    );
  }

  #[test]
  fn scalar_zero_expanded_to_block() {
    assert_eq!(
      interpret("ArrayFlatten[{{{{1, 0}, {0, 1}}, 0}, {0, {{2, 0}, {0, 2}}}}]")
        .unwrap(),
      "{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 2, 0}, {0, 0, 0, 2}}"
    );
  }

  #[test]
  fn block_diagonal_with_scalar_zeros() {
    assert_eq!(
      interpret(
        "ArrayFlatten[{{IdentityMatrix[2], 0}, {0, IdentityMatrix[3]}}]"
      )
      .unwrap(),
      "{{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}}"
    );
  }
}

mod pdf {
  use super::*;

  #[test]
  fn normal_standard() {
    assert_eq!(
      interpret("PDF[NormalDistribution[0, 1], x]").unwrap(),
      "1/(E^(x^2/2)*Sqrt[2*Pi])"
    );
  }

  #[test]
  fn normal_symbolic() {
    assert_eq!(
      interpret("PDF[NormalDistribution[mu, sigma], x]").unwrap(),
      "1/(E^((-mu + x)^2/(2*sigma^2))*Sqrt[2*Pi]*sigma)"
    );
  }

  #[test]
  fn normal_numeric_point() {
    assert_eq!(
      interpret("PDF[NormalDistribution[0, 1], 0]").unwrap(),
      "1/Sqrt[2*Pi]"
    );
  }

  #[test]
  fn normal_at_one() {
    assert_eq!(
      interpret("PDF[NormalDistribution[0, 1], 1]").unwrap(),
      "1/Sqrt[2*E*Pi]"
    );
  }

  #[test]
  fn normal_default_args() {
    assert_eq!(
      interpret("PDF[NormalDistribution[], x]").unwrap(),
      "1/(E^(x^2/2)*Sqrt[2*Pi])"
    );
  }

  #[test]
  fn uniform_standard() {
    assert_eq!(
      interpret("PDF[UniformDistribution[{0, 1}], x]").unwrap(),
      "Piecewise[{{1, 0 <= x <= 1}}, 0]"
    );
  }

  #[test]
  fn uniform_symbolic() {
    assert_eq!(
      interpret("PDF[UniformDistribution[{a, b}], x]").unwrap(),
      "Piecewise[{{(-a + b)^(-1), a <= x <= b}}, 0]"
    );
  }

  #[test]
  fn uniform_default() {
    assert_eq!(
      interpret("PDF[UniformDistribution[], x]").unwrap(),
      "Piecewise[{{1, 0 <= x <= 1}}, 0]"
    );
  }

  #[test]
  fn exponential_symbolic() {
    assert_eq!(
      interpret("PDF[ExponentialDistribution[lambda], x]").unwrap(),
      "Piecewise[{{lambda/E^(lambda*x), x >= 0}}, 0]"
    );
  }

  #[test]
  fn exponential_numeric() {
    assert_eq!(
      interpret("PDF[ExponentialDistribution[2], x]").unwrap(),
      "Piecewise[{{2/E^(2*x), x >= 0}}, 0]"
    );
  }

  #[test]
  fn poisson_symbolic() {
    assert_eq!(
      interpret("PDF[PoissonDistribution[mu], k]").unwrap(),
      "Piecewise[{{mu^k/(E^mu*k!), k >= 0}}, 0]"
    );
  }

  #[test]
  fn bernoulli_symbolic() {
    assert_eq!(
      interpret("PDF[BernoulliDistribution[p], k]").unwrap(),
      "Piecewise[{{1 - p, k == 0}, {p, k == 1}}, 0]"
    );
  }

  #[test]
  fn unknown_distribution_unevaluated() {
    assert_eq!(
      interpret("PDF[SomeDistribution[1, 2], x]").unwrap(),
      "PDF[SomeDistribution[1, 2], x]"
    );
  }

  #[test]
  fn distribution_symbols_are_inert() {
    assert_eq!(
      interpret("ExponentialDistribution[3]").unwrap(),
      "ExponentialDistribution[3]"
    );
    assert_eq!(
      interpret("PoissonDistribution[5]").unwrap(),
      "PoissonDistribution[5]"
    );
    assert_eq!(
      interpret("BernoulliDistribution[0.5]").unwrap(),
      "BernoulliDistribution[0.5]"
    );
  }

  #[test]
  fn uniform_distribution_default() {
    assert_eq!(
      interpret("UniformDistribution[]").unwrap(),
      "UniformDistribution[{0, 1}]"
    );
  }
}

mod nearest {
  use super::*;

  #[test]
  fn basic_tie() {
    assert_eq!(interpret("Nearest[{1, 3, 5, 7, 9}, 4]").unwrap(), "{3, 5}");
  }

  #[test]
  fn basic_tie_2() {
    assert_eq!(interpret("Nearest[{1, 3, 5, 7, 9}, 6]").unwrap(), "{5, 7}");
  }

  #[test]
  fn with_count() {
    assert_eq!(
      interpret("Nearest[{1, 3, 5, 7, 9}, 6, 3]").unwrap(),
      "{5, 7, 3}"
    );
  }

  #[test]
  fn single_nearest() {
    assert_eq!(
      interpret("Nearest[{1.0, 2.5, 4.3, 7.1}, 3.0]").unwrap(),
      "{2.5}"
    );
  }

  #[test]
  fn exact_match() {
    assert_eq!(interpret("Nearest[{1, 3, 5, 7, 9}, 5]").unwrap(), "{5}");
  }

  #[test]
  fn with_all_and_radius() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 5, 8, 13, 21}, 10, {All, 5}]").unwrap(),
      "{8, 13, 5}"
    );
  }

  #[test]
  fn with_count_and_radius() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 5, 8, 13, 21}, 10, {3, 5}]").unwrap(),
      "{8, 13, 5}"
    );
  }

  #[test]
  fn with_radius_too_small() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 5, 8, 13, 21}, 10, {1, 1}]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn with_all_and_large_radius() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 5, 8, 13, 21}, 10, {All, 10}]").unwrap(),
      "{8, 13, 5, 3, 2, 1}"
    );
  }

  #[test]
  fn with_zero_radius() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 5, 8, 13, 21}, 10, {All, 0}]").unwrap(),
      "{}"
    );
  }
}

mod array_pad {
  use super::*;

  #[test]
  fn basic_pad() {
    assert_eq!(
      interpret("ArrayPad[{1, 2, 3}, 2]").unwrap(),
      "{0, 0, 1, 2, 3, 0, 0}"
    );
  }

  #[test]
  fn pad_with_value() {
    assert_eq!(
      interpret("ArrayPad[{1, 2, 3}, 2, x]").unwrap(),
      "{x, x, 1, 2, 3, x, x}"
    );
  }

  #[test]
  fn asymmetric_pad() {
    assert_eq!(
      interpret("ArrayPad[{1, 2, 3}, {1, 2}]").unwrap(),
      "{0, 1, 2, 3, 0, 0}"
    );
  }

  #[test]
  fn negative_pad_trims() {
    assert_eq!(
      interpret("ArrayPad[{1, 2, 3, 4, 5}, -1]").unwrap(),
      "{2, 3, 4}"
    );
  }

  #[test]
  fn two_dimensional() {
    assert_eq!(
      interpret("ArrayPad[{{1, 2}, {3, 4}}, 1]").unwrap(),
      "{{0, 0, 0, 0}, {0, 1, 2, 0}, {0, 3, 4, 0}, {0, 0, 0, 0}}"
    );
  }

  #[test]
  fn pad_zero() {
    assert_eq!(interpret("ArrayPad[{1, 2, 3}, 0]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn asymmetric_with_value() {
    assert_eq!(
      interpret("ArrayPad[{1, 2}, {1, 3}, x]").unwrap(),
      "{x, 1, 2, x, x, x}"
    );
  }

  #[test]
  fn per_dimension_short_spec() {
    // {{m1}, {m2}} — equal padding per dim. 1 row each side, 5 cols each side.
    assert_eq!(
      interpret("ArrayPad[{{1, 2}, {3, 4}}, {{1}, {5}}]").unwrap(),
      "{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0}, \
       {0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0}, \
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}"
    );
  }

  #[test]
  fn per_dimension_asymmetric_spec() {
    // {{m1, n1}, {m2, n2}} — asymmetric padding per dim.
    assert_eq!(
      interpret("ArrayPad[{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}]").unwrap(),
      "{{0, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {0, 0, 0, 1, 2, 0, 0, 0, 0}, \
       {0, 0, 0, 3, 4, 0, 0, 0, 0}, \
       {0, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {0, 0, 0, 0, 0, 0, 0, 0, 0}}"
    );
  }

  #[test]
  fn per_dimension_1d_array() {
    // {{1, 2}} — asymmetric padding on a 1-D array's only dim.
    assert_eq!(
      interpret("ArrayPad[{1, 2, 3}, {{1, 2}}]").unwrap(),
      "{0, 1, 2, 3, 0, 0}"
    );
  }

  #[test]
  fn per_dimension_with_pad_value() {
    assert_eq!(
      interpret("ArrayPad[{{1, 2}}, {{1}, {2}}, x]").unwrap(),
      "{{x, x, x, x, x, x}, {x, x, 1, 2, x, x}, {x, x, x, x, x, x}}"
    );
  }
}

mod array_reshape {
  use super::*;

  #[test]
  fn basic_2d() {
    assert_eq!(
      interpret("ArrayReshape[{1, 2, 3, 4, 5, 6}, {2, 3}]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6}}"
    );
  }

  #[test]
  fn pad_with_zeros() {
    assert_eq!(
      interpret("ArrayReshape[{1, 2, 3}, {2, 3}]").unwrap(),
      "{{1, 2, 3}, {0, 0, 0}}"
    );
  }

  #[test]
  fn truncate() {
    assert_eq!(
      interpret("ArrayReshape[{1, 2, 3, 4, 5, 6, 7}, {2, 3}]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6}}"
    );
  }

  #[test]
  fn three_dimensional() {
    assert_eq!(
      interpret("ArrayReshape[Range[12], {2, 2, 3}]").unwrap(),
      "{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}"
    );
  }

  #[test]
  fn flat_to_1d() {
    assert_eq!(
      interpret("ArrayReshape[{1, 2, 3, 4}, {4}]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn pad_with_scalar() {
    assert_eq!(
      interpret("ArrayReshape[Range[5], {2, 3}, x]").unwrap(),
      "{{1, 2, 3}, {4, 5, x}}"
    );
  }

  #[test]
  fn pad_with_explicit_zero() {
    assert_eq!(
      interpret("ArrayReshape[Range[5], {2, 3}, 0]").unwrap(),
      "{{1, 2, 3}, {4, 5, 0}}"
    );
  }

  #[test]
  fn pad_cycles_through_list() {
    // Padding `{a, b}` cycles — only one trailing slot here, so `a`.
    assert_eq!(
      interpret("ArrayReshape[Range[5], {2, 3}, {a, b}]").unwrap(),
      "{{1, 2, 3}, {4, 5, a}}"
    );
  }

  #[test]
  fn pad_cycles_through_multi_element_list() {
    assert_eq!(
      interpret("ArrayReshape[Range[2], {2, 3}, {a, b, c, d}]").unwrap(),
      "{{1, 2, a}, {b, c, d}}"
    );
  }

  #[test]
  fn pad_fills_entire_array_when_input_empty() {
    assert_eq!(
      interpret("ArrayReshape[{}, {2, 2}, {a, b, c, d}]").unwrap(),
      "{{a, b}, {c, d}}"
    );
  }
}

mod position_index {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("PositionIndex[{a, b, c, a, b, a}]").unwrap(),
      "<|a -> {1, 4, 6}, b -> {2, 5}, c -> {3}|>"
    );
  }

  #[test]
  fn all_unique() {
    assert_eq!(
      interpret("PositionIndex[{x, y, z}]").unwrap(),
      "<|x -> {1}, y -> {2}, z -> {3}|>"
    );
  }

  #[test]
  fn all_same() {
    assert_eq!(
      interpret("PositionIndex[{1, 1, 1}]").unwrap(),
      "<|1 -> {1, 2, 3}|>"
    );
  }

  #[test]
  fn empty() {
    assert_eq!(interpret("PositionIndex[{}]").unwrap(), "<||>");
  }
}

mod list_convolve {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("ListConvolve[{1, 1, 1}, {a, b, c, d, e}]").unwrap(),
      "{a + b + c, b + c + d, c + d + e}"
    );
  }

  #[test]
  fn difference() {
    assert_eq!(
      interpret("ListConvolve[{1, -1}, {1, 2, 4, 8, 16}]").unwrap(),
      "{1, 2, 4, 8}"
    );
  }

  #[test]
  fn weighted_average() {
    assert_eq!(
      interpret("ListConvolve[{1, 2, 1}, {1, 0, 0, 1, 0}]").unwrap(),
      "{1, 1, 2}"
    );
  }

  #[test]
  fn single_element_kernel() {
    assert_eq!(
      interpret("ListConvolve[{3}, {1, 2, 3}]").unwrap(),
      "{3, 6, 9}"
    );
  }

  #[test]
  fn kernel_equals_list() {
    assert_eq!(interpret("ListConvolve[{1, 2}, {3, 4}]").unwrap(), "{10}");
  }
}

mod list_correlate {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("ListCorrelate[{1, -1}, {1, 2, 4, 8, 16}]").unwrap(),
      "{-1, -2, -4, -8}"
    );
  }

  #[test]
  fn symmetric_kernel() {
    assert_eq!(
      interpret("ListCorrelate[{1, 1, 1}, {a, b, c, d, e}]").unwrap(),
      "{a + b + c, b + c + d, c + d + e}"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("ListCorrelate[{1, 2, 1}, {1, 0, 0, 1, 0}]").unwrap(),
      "{1, 1, 2}"
    );
  }
}

mod probability {
  use super::*;

  #[test]
  fn uniform_greater_than() {
    // P(x > 1/2) for Uniform[0,1] = 1/2
    assert_eq!(
      interpret("Probability[x > 1/2, Distributed[x, UniformDistribution[]]]")
        .unwrap(),
      "1/2"
    );
  }

  #[test]
  fn uniform_less_than() {
    // P(x < 1/4) for Uniform[0,1] = 1/4
    assert_eq!(
      interpret("Probability[x < 1/4, Distributed[x, UniformDistribution[]]]")
        .unwrap(),
      "1/4"
    );
  }

  #[test]
  fn uniform_range() {
    // P(1/4 < x < 3/4) for Uniform[0,1] = 1/2
    assert_eq!(
      interpret(
        "Probability[1/4 < x < 3/4, Distributed[x, UniformDistribution[]]]"
      )
      .unwrap(),
      "1/2"
    );
  }

  #[test]
  fn bernoulli_equals_one() {
    // P(x == 1) for Bernoulli[1/3] = 1/3
    assert_eq!(
      interpret(
        "Probability[x == 1, Distributed[x, BernoulliDistribution[1/3]]]"
      )
      .unwrap(),
      "1/3"
    );
  }

  #[test]
  fn bernoulli_equals_zero() {
    // P(x == 0) for Bernoulli[1/4] = 3/4
    assert_eq!(
      interpret(
        "Probability[x == 0, Distributed[x, BernoulliDistribution[1/4]]]"
      )
      .unwrap(),
      "3/4"
    );
  }

  #[test]
  fn poisson_equals() {
    // P(x == 3) for Poisson[5] = 125/(6*E^5)
    assert_eq!(
      interpret("Probability[x == 3, Distributed[x, PoissonDistribution[5]]]")
        .unwrap(),
      "125/(6*E^5)"
    );
  }

  #[test]
  fn exponential_less_than() {
    // P(x < 2) for Exp[1] = 1 - E^(-2)
    assert_eq!(
      interpret(
        "Probability[x < 2, Distributed[x, ExponentialDistribution[1]]]"
      )
      .unwrap(),
      "(-1 + E^2)/E^2"
    );
  }

  #[test]
  fn normal_less_than() {
    // P(x < 0) for N[0,1] = 1/2 (by symmetry, CDF(0) = 1/2)
    assert_eq!(
      interpret("Probability[x < 0, Distributed[x, NormalDistribution[0, 1]]]")
        .unwrap(),
      "1/2"
    );
  }

  #[test]
  fn unevaluated_wrong_args() {
    // Wrong number of args returns unevaluated
    assert_eq!(
      interpret("Probability[x > 0]").unwrap(),
      "Probability[x > 0]"
    );
  }

  #[test]
  fn distributed_infix_operator_named_char() {
    // x \[Distributed] dist should parse as Distributed[x, dist]
    assert_eq!(
      interpret(
        "Probability[x == 3, x \\[Distributed] DiscreteUniformDistribution[{1, 6}]]"
      )
      .unwrap(),
      "1/6"
    );
  }

  #[test]
  fn distributed_infix_operator_unicode() {
    // The Unicode private-use char \uF3D2 should also work
    assert_eq!(
      interpret(
        "Probability[x == 3, x \u{F3D2} DiscreteUniformDistribution[{1, 6}]]"
      )
      .unwrap(),
      "1/6"
    );
  }

  #[test]
  fn distributed_infix_matches_function_form() {
    // Both forms must give identical results
    let named = interpret(
      "Probability[x == 3, x \\[Distributed] DiscreteUniformDistribution[{1, 6}]]",
    )
    .unwrap();
    let function_form = interpret(
      "Probability[x == 3, Distributed[x, DiscreteUniformDistribution[{1, 6}]]]",
    )
    .unwrap();
    assert_eq!(named, function_form);
  }

  #[test]
  fn joint_two_dice_sum_twelve() {
    // Classic two-dice probability. Only (6, 6) sums to 12 → 1/36.
    assert_eq!(
      interpret(
        "Probability[x + y == 12, \
         x \\[Distributed] DiscreteUniformDistribution[{1, 6}] \
         && y \\[Distributed] DiscreteUniformDistribution[{1, 6}]]"
      )
      .unwrap(),
      "1/36"
    );
  }

  #[test]
  fn joint_two_dice_sum_seven() {
    // Six outcomes sum to 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)
    // → 6/36 = 1/6.
    assert_eq!(
      interpret(
        "Probability[x + y == 7, \
         x \\[Distributed] DiscreteUniformDistribution[{1, 6}] \
         && y \\[Distributed] DiscreteUniformDistribution[{1, 6}]]"
      )
      .unwrap(),
      "1/6"
    );
  }

  #[test]
  fn joint_two_dice_inequality() {
    // P(x + y <= 4): (1,1),(1,2),(1,3),(2,1),(2,2),(3,1) → 6/36 = 1/6.
    assert_eq!(
      interpret(
        "Probability[x + y <= 4, \
         x \\[Distributed] DiscreteUniformDistribution[{1, 6}] \
         && y \\[Distributed] DiscreteUniformDistribution[{1, 6}]]"
      )
      .unwrap(),
      "1/6"
    );
  }
}

mod expectation {
  use super::*;

  #[test]
  fn normal_mean() {
    assert_eq!(
      interpret("Expectation[x, Distributed[x, NormalDistribution[3, 1]]]")
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn uniform_mean() {
    assert_eq!(
      interpret("Expectation[x, Distributed[x, UniformDistribution[{0, 1}]]]")
        .unwrap(),
      "1/2"
    );
  }

  #[test]
  fn exponential_mean() {
    assert_eq!(
      interpret("Expectation[x, Distributed[x, ExponentialDistribution[3]]]")
        .unwrap(),
      "1/3"
    );
  }

  #[test]
  fn poisson_mean() {
    assert_eq!(
      interpret("Expectation[x, Distributed[x, PoissonDistribution[5]]]")
        .unwrap(),
      "5"
    );
  }

  #[test]
  fn bernoulli_mean() {
    assert_eq!(
      interpret("Expectation[x, Distributed[x, BernoulliDistribution[1/3]]]")
        .unwrap(),
      "1/3"
    );
  }

  #[test]
  fn normal_x_squared() {
    // E[x^2] for N[0,1] = Var + Mean^2 = 1 + 0 = 1
    assert_eq!(
      interpret("Expectation[x^2, Distributed[x, NormalDistribution[0, 1]]]")
        .unwrap(),
      "1"
    );
  }

  #[test]
  fn linear_expression() {
    // E[2x + 1] for Uniform[0,1] = 2*(1/2) + 1 = 2
    assert_eq!(
      interpret(
        "Expectation[2 x + 1, Distributed[x, UniformDistribution[{0, 1}]]]"
      )
      .unwrap(),
      "2"
    );
  }

  #[test]
  fn unevaluated_wrong_args() {
    assert_eq!(interpret("Expectation[x]").unwrap(), "Expectation[x]");
  }
}

mod groupings {
  use super::*;

  #[test]
  fn single_element() {
    assert_eq!(interpret("Groupings[1, 2]").unwrap(), "{1}");
  }

  #[test]
  fn two_elements_binary() {
    assert_eq!(interpret("Groupings[2, 2]").unwrap(), "{{1, 2}}");
  }

  #[test]
  fn three_elements_binary() {
    assert_eq!(
      interpret("Groupings[{a, b, c}, 2]").unwrap(),
      "{{{a, b}, c}, {a, {b, c}}}"
    );
  }

  #[test]
  fn four_elements_binary() {
    assert_eq!(
      interpret("Groupings[{a, b, c, d}, 2]").unwrap(),
      "{{{{a, b}, c}, d}, {a, {{b, c}, d}}, {{a, {b, c}}, d}, {a, {b, {c, d}}}, {{a, b}, {c, d}}}"
    );
  }

  #[test]
  fn integer_form() {
    assert_eq!(
      interpret("Groupings[3, 2]").unwrap(),
      "{{{1, 2}, 3}, {1, {2, 3}}}"
    );
  }

  #[test]
  fn ternary_exact() {
    assert_eq!(interpret("Groupings[{a, b, c}, 3]").unwrap(), "{{a, b, c}}");
  }

  #[test]
  fn ternary_five_elements() {
    assert_eq!(
      interpret("Groupings[{a, b, c, d, e}, 3]").unwrap(),
      "{{{a, b, c}, d, e}, {a, {b, c, d}, e}, {a, b, {c, d, e}}}"
    );
  }

  #[test]
  fn ternary_impossible() {
    // 4 elements can't form a ternary tree
    assert_eq!(interpret("Groupings[{a, b, c, d}, 3]").unwrap(), "{}");
  }
}

mod peak_detect {
  use super::*;

  #[test]
  fn basic_peaks() {
    assert_eq!(
      interpret("PeakDetect[{1, 3, 2, 5, 1, 4, 2}]").unwrap(),
      "{0, 1, 0, 1, 0, 1, 0}"
    );
  }

  #[test]
  fn single_element() {
    assert_eq!(interpret("PeakDetect[{1}]").unwrap(), "{0}");
  }

  #[test]
  fn all_equal() {
    assert_eq!(interpret("PeakDetect[{2, 2, 2}]").unwrap(), "{0, 0, 0}");
  }

  #[test]
  fn plateau_peak() {
    assert_eq!(
      interpret("PeakDetect[{1, 3, 3, 2}]").unwrap(),
      "{0, 1, 1, 0}"
    );
  }

  #[test]
  fn monotonic_increasing() {
    assert_eq!(
      interpret("PeakDetect[{1, 2, 3, 4, 5}]").unwrap(),
      "{0, 0, 0, 0, 1}"
    );
  }

  #[test]
  fn monotonic_decreasing() {
    assert_eq!(interpret("PeakDetect[{3, 2, 1}]").unwrap(), "{1, 0, 0}");
  }

  #[test]
  fn endpoints_as_peaks() {
    assert_eq!(
      interpret("PeakDetect[{5, 3, 1, 3, 5}]").unwrap(),
      "{1, 0, 0, 0, 1}"
    );
  }

  #[test]
  fn sharpness_1() {
    assert_eq!(
      interpret("PeakDetect[{1, 3, 2, 5, 1, 4, 2}, 1]").unwrap(),
      "{0, 0, 0, 1, 0, 1, 0}"
    );
  }

  #[test]
  fn sharpness_2() {
    assert_eq!(
      interpret("PeakDetect[{1, 3, 2, 5, 1, 4, 2}, 2]").unwrap(),
      "{0, 0, 0, 1, 0, 0, 0}"
    );
  }

  #[test]
  fn docs_example() {
    assert_eq!(
      interpret(
        "PeakDetect[{2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1}]"
      )
      .unwrap(),
      "{1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0}"
    );
  }
}

mod through {
  use super::*;

  #[test]
  fn through_list_of_functions() {
    assert_eq!(
      interpret("Through[{Sin, Cos, Tan}[Pi/4]]").unwrap(),
      "{1/Sqrt[2], 1/Sqrt[2], 1}"
    );
  }

  #[test]
  fn through_min_max() {
    assert_eq!(interpret("Through[{Min, Max}[3, 1, 2]]").unwrap(), "{1, 3}");
  }

  #[test]
  fn through_single_function() {
    assert_eq!(interpret("Through[{f}[x, y]]").unwrap(), "{f[x, y]}");
  }
}

mod replace_level_spec {
  use woxi::interpret;

  #[test]
  fn replace_with_infinity_level() {
    assert_eq!(
      interpret("Replace[{1, {2, {3}}}, x_Integer :> x^2, Infinity]").unwrap(),
      "{1, {4, {9}}}"
    );
  }

  #[test]
  fn replace_with_zero_to_infinity() {
    assert_eq!(
      interpret("Replace[{1, {2, {3}}}, x_Integer :> x^2, {0, Infinity}]")
        .unwrap(),
      "{1, {4, {9}}}"
    );
  }

  #[test]
  fn replace_with_exact_level() {
    assert_eq!(
      interpret("Replace[{1, {2, {3}}}, x_Integer :> x^2, {2}]").unwrap(),
      "{1, {4, {3}}}"
    );
  }

  #[test]
  fn replace_with_level_range() {
    assert_eq!(
      interpret("Replace[{1, {2, {3}}}, x_Integer :> x^2, {1, 3}]").unwrap(),
      "{1, {4, {9}}}"
    );
  }
}

mod cases_count_level_spec {
  use woxi::interpret;

  #[test]
  fn cases_infinity_level() {
    assert_eq!(
      interpret("Cases[{1, {2, 3}, {4, {5}}}, _Integer, Infinity]").unwrap(),
      "{1, 2, 3, 4, 5}"
    );
  }

  // Cases[expr, pattern, Heads -> True] includes the head symbol in the
  // candidate pool.
  #[test]
  fn cases_heads_option_default_level() {
    assert_eq!(
      interpret(r"Cases[{b, 6, \[Pi]}, _Symbol, Heads -> True]").unwrap(),
      "{List, b, Pi}"
    );
  }

  // Without the option, Heads defaults to False and the head is excluded.
  #[test]
  fn cases_heads_default_excludes_head() {
    assert_eq!(
      interpret(r"Cases[{b, 6, \[Pi]}, _Symbol]").unwrap(),
      "{b, Pi}"
    );
  }

  #[test]
  fn cases_exact_level() {
    assert_eq!(
      interpret("Cases[{1, {2, 3}, {4, {5}}}, _Integer, {2}]").unwrap(),
      "{2, 3, 4}"
    );
  }

  #[test]
  fn count_infinity_level() {
    assert_eq!(
      interpret("Count[{1, {2, 3}, {4, {5}}}, _Integer, Infinity]").unwrap(),
      "5"
    );
  }

  #[test]
  fn delete_cases_infinity_level() {
    assert_eq!(
      interpret("DeleteCases[{1, {2, 3}, {4, {5}}}, _Integer, Infinity]")
        .unwrap(),
      "{{}, {{}}}"
    );
  }

  #[test]
  fn delete_cases_exact_level() {
    assert_eq!(
      interpret("DeleteCases[{1, {2, 3}, {4, {5}}}, _Integer, {2}]").unwrap(),
      "{1, {}, {{5}}}"
    );
  }

  #[test]
  fn cases_with_max_count() {
    // Cases[expr, pat, levelspec, n] returns at most n matches in scan order.
    assert_eq!(
      interpret("Cases[{1, {2, 3}, {4, {5, 6}}}, _Integer, Infinity, 3]")
        .unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn cases_with_max_count_flat() {
    assert_eq!(interpret("Cases[{a, b, c, a, b}, a, 1, 1]").unwrap(), "{a}");
    assert_eq!(
      interpret("Cases[{a, b, c, a, b}, a, Infinity, 5]").unwrap(),
      "{a, a}"
    );
  }

  #[test]
  fn cases_with_max_count_zero() {
    // n = 0 should always yield {}.
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4, 5}, _Integer, 1, 0]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn cases_with_infinite_max_count() {
    // n = Infinity behaves the same as the 3-arg form.
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4, 5}, _Integer, Infinity, Infinity]")
        .unwrap(),
      "{1, 2, 3, 4, 5}"
    );
  }

  #[test]
  fn cases_with_rule_and_levelspec() {
    // The 3-arg form should still apply Rule/RuleDelayed replacements
    // to matched elements (a regression check for the pattern :> rhs path).
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4}, x_Integer :> x^2, 1]").unwrap(),
      "{1, 4, 9, 16}"
    );
  }

  #[test]
  fn cases_with_rule_and_max_count() {
    // 4-arg form combined with a Rule pattern.
    assert_eq!(
      interpret("Cases[{1, 2, 3, 4, 5}, x_Integer :> x^2, 1, 3]").unwrap(),
      "{1, 4, 9}"
    );
  }
}

mod tally_with_test {
  use super::*;

  #[test]
  fn tally_with_integer_tolerance() {
    // Groups elements equivalent (within 2) to the first representative
    // they encounter. 10 absorbs 11; 20 absorbs 21 and 22.
    assert_eq!(
      interpret(
        "Tally[{10, 11, 20, 21, 22}, Function[{a, b}, Abs[a - b] < 3]]"
      )
      .unwrap(),
      "{{10, 2}, {20, 3}}"
    );
  }

  #[test]
  fn tally_with_mod_equivalence() {
    assert_eq!(
      interpret(
        "Tally[{1, 2, 3, 4, 5, 6}, Function[{a, b}, Mod[a, 3] == Mod[b, 3]]]"
      )
      .unwrap(),
      "{{1, 2}, {2, 2}, {3, 2}}"
    );
  }

  #[test]
  fn tally_with_sort_equivalence() {
    assert_eq!(
      interpret(
        "Tally[{{1, 2}, {2, 1}, {3, 4}, {4, 3}}, \
         Function[{a, b}, Sort[a] == Sort[b]]]"
      )
      .unwrap(),
      "{{{1, 2}, 2}, {{3, 4}, 2}}"
    );
  }

  #[test]
  fn tally_with_test_empty_list() {
    assert_eq!(interpret("Tally[{}, Equal]").unwrap(), "{}");
  }

  #[test]
  fn tally_with_named_test() {
    assert_eq!(
      interpret("Tally[{1, 2, 3, 4, 5, 6}, EvenQ[#1 - #2] &]").unwrap(),
      "{{1, 3}, {2, 3}}"
    );
  }

  #[test]
  fn tally_with_bare_symbol_test() {
    // Everything passes Equal check only if literally equal, so this
    // reduces to the plain Tally behavior.
    assert_eq!(
      interpret("Tally[{a, b, a, c, b, a}, Equal]").unwrap(),
      "{{a, 3}, {b, 2}, {c, 1}}"
    );
  }
}

mod take_largest {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("TakeLargest[{3, 1, 4, 1, 5, 9, 2, 6}, 3]").unwrap(),
      "{9, 6, 5}"
    );
  }

  #[test]
  fn reals() {
    assert_eq!(
      interpret("TakeLargest[{3.5, 1.2, 4.7}, 2]").unwrap(),
      "{4.7, 3.5}"
    );
  }

  #[test]
  fn rationals() {
    assert_eq!(
      interpret("TakeLargest[{1/3, 1/2, 1/4, 1/5}, 2]").unwrap(),
      "{1/2, 1/3}"
    );
  }

  #[test]
  fn negative_numbers() {
    assert_eq!(
      interpret("TakeLargest[{-3, -1, -4, -1, -5}, 3]").unwrap(),
      "{-1, -1, -3}"
    );
  }

  #[test]
  fn n_exceeds_length() {
    // When n > length, return unevaluated.
    assert_eq!(
      interpret("TakeLargest[{5, 2, 8}, 5]").unwrap(),
      "TakeLargest[{5, 2, 8}, 5]"
    );
  }

  #[test]
  fn take_zero() {
    assert_eq!(interpret("TakeLargest[{3, 1, 4}, 0]").unwrap(), "{}");
  }

  #[test]
  fn empty_list_zero() {
    assert_eq!(interpret("TakeLargest[{}, 0]").unwrap(), "{}");
  }

  #[test]
  fn missing_values_excluded() {
    // Default behaviour (mirrors Wolfram's ExcludedForms -> {_Missing}):
    // Missing[...] entries are skipped, so the result is drawn only from
    // the numeric portion of the list.
    assert_eq!(
      interpret("TakeLargest[{-8, 150, Missing[abc]}, 2]").unwrap(),
      "{150, -8}"
    );
  }

  #[test]
  fn excluded_forms_empty_includes_missing() {
    // ExcludedForms -> {} keeps every element; canonical descending
    // order places Missing[...] above numbers.
    assert_eq!(
      interpret("TakeLargest[{-8, 150, Missing[abc]}, 2, ExcludedForms -> {}]")
        .unwrap(),
      "{Missing[abc], 150}"
    );
  }

  #[test]
  fn excluded_forms_multiple_missing() {
    assert_eq!(
      interpret(
        "TakeLargest[{-8, 150, Missing[abc], Missing[foo]}, 3, \
         ExcludedForms -> {}]"
      )
      .unwrap(),
      "{Missing[foo], Missing[abc], 150}"
    );
  }

  #[test]
  fn excluded_forms_explicit_missing_pattern() {
    assert_eq!(
      interpret(
        "TakeLargest[{-8, 150, Missing[abc]}, 2, \
         ExcludedForms -> {_Missing}]"
      )
      .unwrap(),
      "{150, -8}"
    );
  }
}

mod take_smallest {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("TakeSmallest[{3, 1, 4, 1, 5, 9, 2, 6}, 3]").unwrap(),
      "{1, 1, 2}"
    );
  }

  #[test]
  fn reals() {
    assert_eq!(
      interpret("TakeSmallest[{3.5, 1.2, 4.7}, 2]").unwrap(),
      "{1.2, 3.5}"
    );
  }

  #[test]
  fn rationals() {
    assert_eq!(
      interpret("TakeSmallest[{1/3, 1/2, 1/4, 1/5}, 2]").unwrap(),
      "{1/5, 1/4}"
    );
  }

  #[test]
  fn negative_numbers() {
    assert_eq!(
      interpret("TakeSmallest[{-3, -1, -4, -1, -5}, 3]").unwrap(),
      "{-5, -4, -3}"
    );
  }

  #[test]
  fn n_exceeds_length() {
    assert_eq!(
      interpret("TakeSmallest[{5, 2, 8}, 5]").unwrap(),
      "TakeSmallest[{5, 2, 8}, 5]"
    );
  }

  #[test]
  fn take_zero() {
    assert_eq!(interpret("TakeSmallest[{3, 1, 4}, 0]").unwrap(), "{}");
  }
}

mod ratios_tests {
  use super::*;

  #[test]
  fn ratios_basic() {
    assert_eq!(interpret("Ratios[{1, 2, 4, 8}]").unwrap(), "{2, 2, 2}");
  }

  #[test]
  fn ratios_symbolic() {
    assert_eq!(
      interpret("Ratios[{a, b, c, d}]").unwrap(),
      "{b/a, c/b, d/c}"
    );
  }

  #[test]
  fn ratios_empty() {
    assert_eq!(interpret("Ratios[{}]").unwrap(), "{}");
  }

  #[test]
  fn ratios_single_element() {
    assert_eq!(interpret("Ratios[{5}]").unwrap(), "{}");
  }

  #[test]
  fn ratios_iterated_second_arg() {
    // Ratios[list, n] applies Ratios n times.
    assert_eq!(interpret("Ratios[{1, 2, 4, 8}, 2]").unwrap(), "{1, 1}");
  }

  #[test]
  fn ratios_iterated_three() {
    assert_eq!(interpret("Ratios[{1, 2, 4, 8, 16}, 3]").unwrap(), "{1, 1}");
  }

  #[test]
  fn ratios_iterated_symbolic() {
    assert_eq!(
      interpret("Ratios[{a, b, c, d, e}, 2]").unwrap(),
      "{(a*c)/b^2, (b*d)/c^2, (c*e)/d^2}"
    );
  }

  #[test]
  fn ratios_iterated_zero() {
    // Ratios[list, 0] returns the original list.
    assert_eq!(interpret("Ratios[{1, 2, 3}, 0]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn ratios_non_list_head() {
    // Ratios only works on List; other heads return unevaluated.
    assert_eq!(
      interpret("Ratios[f[a, b, c]]").unwrap(),
      "Ratios[f[a, b, c]]"
    );
  }
}

mod splice {
  use super::*;

  #[test]
  fn splice_in_list() {
    assert_eq!(interpret("{1, Splice[{2, 3}], 4}").unwrap(), "{1, 2, 3, 4}");
  }

  #[test]
  fn splice_not_in_other_heads() {
    // Splice without head arg only works inside List, not other heads.
    assert_eq!(
      interpret("f[1, Splice[{2, 3}], 4]").unwrap(),
      "f[1, Splice[{2, 3}], 4]"
    );
  }

  #[test]
  fn splice_unevaluated_at_top_level() {
    assert_eq!(interpret("Splice[{a, b, c}]").unwrap(), "Splice[{a, b, c}]");
  }

  #[test]
  fn splice_with_matching_head() {
    // Splice[list, head] splices into functions with that head.
    assert_eq!(interpret("f[Splice[{1, 2}, f]]").unwrap(), "f[1, 2]");
  }

  #[test]
  fn splice_with_non_matching_head() {
    // Splice[list, head] does NOT splice when the enclosing head differs.
    assert_eq!(
      interpret("f[Splice[{1, 2}, List]]").unwrap(),
      "f[Splice[{1, 2}, List]]"
    );
  }

  #[test]
  fn splice_with_head_not_in_list() {
    // Splice[list, f] does NOT splice inside List.
    assert_eq!(
      interpret("{Splice[{1, 2}, f]}").unwrap(),
      "{Splice[{1, 2}, f]}"
    );
  }

  #[test]
  fn splice_empty_list() {
    assert_eq!(interpret("{1, Splice[{}], 2}").unwrap(), "{1, 2}");
  }

  #[test]
  fn splice_multiple() {
    assert_eq!(
      interpret("{Splice[{a}], Splice[{b, c}]}").unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn splice_with_explicit_list_head() {
    // Splice[list, List] should also work inside List.
    assert_eq!(
      interpret("{1, Splice[{2, 3}, List], 4}").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn splice_two_arg_unevaluated_at_top_level() {
    assert_eq!(
      interpret("Splice[{1, 2}, List]").unwrap(),
      "Splice[{1, 2}, List]"
    );
  }
}

mod matrix_q_with_test {
  use super::*;

  #[test]
  fn number_q() {
    assert_eq!(
      interpret("MatrixQ[{{1, 3}, {4.0, 3/2}}, NumberQ]").unwrap(),
      "True"
    );
  }

  #[test]
  fn positive() {
    assert_eq!(
      interpret("MatrixQ[{{1, 2}, {3, 4 + 5}}, Positive]").unwrap(),
      "True"
    );
  }

  #[test]
  fn positive_with_complex() {
    assert_eq!(
      interpret("MatrixQ[{{1, 2 I}, {3, 4 + 5}}, Positive]").unwrap(),
      "False"
    );
  }

  #[test]
  fn one_arg_still_works() {
    assert_eq!(interpret("MatrixQ[{{1, 2}, {3, 4}}]").unwrap(), "True");
    assert_eq!(interpret("MatrixQ[{{1, 2}, {3}}]").unwrap(), "False");
  }
}

mod table_infinity_step {
  use super::*;

  #[test]
  fn infinity_step_gives_start_only() {
    assert_eq!(interpret("Table[i, {i, 1, 9, Infinity}]").unwrap(), "{1}");
  }
}

// `OptionValue["m"]` (string key) and `OptionValue[m]` (symbol key)
// should both resolve against the same `Options[f]` rule list — they
// share the same underlying name "m". Wolfram folds them together;
// previously Woxi only matched symbol keys.
mod option_value_string_key {
  use super::*;

  #[test]
  fn string_key_resolves_in_options_pattern() {
    clear_state();
    assert_eq!(
      interpret(
        r#"f[x_, OptionsPattern[f]] := x ^ OptionValue["m"]; Options[f] = {"m" -> 7}; f[x]"#
      )
      .unwrap(),
      "x^7"
    );
  }

  #[test]
  fn symbol_lookup_against_string_default() {
    clear_state();
    assert_eq!(
      interpret(
        r#"f[x_, OptionsPattern[f]] := x ^ OptionValue[m]; Options[f] = {"m" -> 7}; f[x]"#
      )
      .unwrap(),
      "x^7"
    );
  }

  #[test]
  fn string_lookup_against_symbol_default() {
    clear_state();
    assert_eq!(
      interpret(
        r#"f[x_, OptionsPattern[f]] := x ^ OptionValue["m"]; Options[f] = {m -> 7}; f[x]"#
      )
      .unwrap(),
      "x^7"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn list_literal() {
    assert_case(r#"{{1, 3}, {5, 7}}[[All, 1]]"#, r#"{1, 5}"#);
  }
  #[test]
  fn take_1() {
    assert_case(
      r#"{{1, 3}, {5, 7}}[[All, 1]]; Take[{{1, 3}, {5, 7}}, All, {1}]"#,
      r#"{{1}, {5}}"#,
    );
  }
  #[test]
  fn nest_list_1() {
    assert_case(
      r#"NestList[#^2 + 1 &, 1, 7]"#,
      r#"{1, 2, 5, 26, 677, 458330, 210066388901, 44127887745906175987802}"#,
    );
  }
  #[test]
  fn table_1() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]"#,
      r#"{0, 1, 9, 36, 100}"#,
    );
  }
  #[test]
  fn sum_1() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]"#,
      r#"HarmonicNumber[n, 2]"#,
    );
  }
  #[test]
  fn sum_2() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]; Sum[k, {k, n, 2 n}]"#,
      r#"(3*n*(1 + n))/2"#,
    );
  }
  #[test]
  fn sum_3() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]; Sum[k, {k, n, 2 n}]; Sum[k, {k, I, I + 1}]"#,
      r#"1 + 2*I"#,
    );
  }
  #[test]
  fn sum_4() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]; Sum[k, {k, n, 2 n}]; Sum[k, {k, I, I + 1}]; Sum[k, {k, Range[5]}]"#,
      r#"15"#,
    );
  }
  #[test]
  fn sum_5() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]; Sum[k, {k, n, 2 n}]; Sum[k, {k, I, I + 1}]; Sum[k, {k, Range[5]}]; Sum[f[i], {i, 1, 7}]"#,
      r#"f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7]"#,
    );
  }
  #[test]
  fn sum_6() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]; Sum[k, {k, n, 2 n}]; Sum[k, {k, I, I + 1}]; Sum[k, {k, Range[5]}]; Sum[f[i], {i, 1, 7}]; Sum[x ^ 2, {x, 1, y}] - y * (y + 1) * (2 * y + 1) / 6"#,
      r#"0"#,
    );
  }
  #[test]
  fn sum_7() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]; Sum[k, {k, n, 2 n}]; Sum[k, {k, I, I + 1}]; Sum[k, {k, Range[5]}]; Sum[f[i], {i, 1, 7}]; Sum[x ^ 2, {x, 1, y}] - y * (y + 1) * (2 * y + 1) / 6; Sum[i, {i, 1, 2.5}]"#,
      r#"3"#,
    );
  }
  #[test]
  fn sum_8() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]; Sum[k, {k, n, 2 n}]; Sum[k, {k, I, I + 1}]; Sum[k, {k, Range[5]}]; Sum[f[i], {i, 1, 7}]; Sum[x ^ 2, {x, 1, y}] - y * (y + 1) * (2 * y + 1) / 6; Sum[i, {i, 1, 2.5}]; Sum[i, {i, 1.1, 2.5}]"#,
      r#"3.2"#,
    );
  }
  #[test]
  fn sum_9() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]; Table[ Sum[i * j, {i, 0, n}, {j, 0, n}], {n, 0, 4} ]; Sum[1 / k ^ 2, {k, 1, n}]; Sum[k, {k, n, 2 n}]; Sum[k, {k, I, I + 1}]; Sum[k, {k, Range[5]}]; Sum[f[i], {i, 1, 7}]; Sum[x ^ 2, {x, 1, y}] - y * (y + 1) * (2 * y + 1) / 6; Sum[i, {i, 1, 2.5}]; Sum[i, {i, 1.1, 2.5}]; Sum[k, {k, I, I + 1.5}]"#,
      r#"1 + 2*I"#,
    );
  }
  // Running-product fast path for Sum[Factorial[var], {var, min, max(, step)}].
  // Iterates large factorial sums in O(n) bignum multiplications instead of
  // O(n^2) — covers the left-factorials RosettaCode script.
  #[test]
  fn sum_factorial_small() {
    assert_case(r#"Sum[k!, {k, 0, 5}]"#, r#"154"#);
  }
  #[test]
  fn sum_factorial_nonzero_min() {
    assert_case(r#"Sum[k!, {k, 3, 6}]"#, r#"870"#);
  }
  #[test]
  fn sum_factorial_with_step() {
    assert_case(r#"Sum[k!, {k, 0, 10, 2}]"#, r#"3669867"#);
  }
  #[test]
  fn sum_factorial_large() {
    assert_case(r#"Length[IntegerDigits[Sum[k!, {k, 0, 1000}]]]"#, r#"2568"#);
  }
  #[test]
  fn reverse_sort_1() {
    assert_case(r#"ReverseSort[{c, b, d, a}]"#, r#"{d, c, b, a}"#);
  }
  #[test]
  fn reverse_sort_2() {
    assert_case(
      r#"ReverseSort[{c, b, d, a}]; ReverseSort[{1, 2, 0, 3}, Less]"#,
      r#"{3, 2, 1, 0}"#,
    );
  }
  #[test]
  fn reverse_sort_3() {
    assert_case(
      r#"ReverseSort[{c, b, d, a}]; ReverseSort[{1, 2, 0, 3}, Less]; ReverseSort[{1, 2, 0, 3}, Greater]"#,
      r#"{0, 1, 2, 3}"#,
    );
  }
  #[test]
  fn sort_1() {
    assert_case(r#"Sort[{4, 1.0, a, 3+I}]"#, r#"{1., 3 + I, 4, a}"#);
  }
  #[test]
  fn list_log_plot_1() {
    assert_case(
      r#"ListLogPlot[Table[Fibonacci[n], {n, 10}]]"#,
      r#"Graphics[{{}, Annotation[{{Annotation[{Directive[PointSize[0.012833333333333334], RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Point[{{1., 0.}, {2., 0.}, {3., 0.6931471805599453}, {4., 1.0986122886681098}, {5., 1.6094379124341003}, {6., 2.0794415416798357}, {7., 2.5649493574615367}, {8., 3.044522437723423}, {9., 3.5263605246161616}, {10., 4.007333185232471}}]}, "Charting`Private`Tag#1"]}}, <|"HighlightElements" -> <|"Label" -> {"XYLabel"}, "Ball" -> {"IndicatedBall"}|>, "LayoutOptions" -> <|"PanelPlotLayout" -> <||>, "PlotRange" -> {{0., 10.}, {-0.31359656347995973, 4.007333185232471}}, "Frame" -> {{False, False}, {False, False}}, "AxesOrigin" -> {0., -0.31359656347995973}, "ImageSize" -> {360, 360/GoldenRatio}, "Axes" -> {True, True}, "LabelStyle" -> {}, "AspectRatio" -> GoldenRatio^(-1), "DefaultStyle" -> {Directive[PointSize[0.012833333333333334], RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]]}, "HighlightLabelingFunctions" -> <|"CoordinatesToolOptions" -> ({(Identity[#1] & )[#1[[1]]], (Exp[#1] & )[#1[[2]]]} & ), "ScalingFunctions" -> {{Identity, Identity}, {Log, Exp}}|>, "Primitives" -> {}, "GCFlag" -> False|>, "Meta" -> <|"DefaultHighlight" -> {"Dynamic", None}, "Index" -> {}, "Function" -> ListLogPlot, "GroupHighlight" -> False|>|>, "DynamicHighlight"], {{}, {}}}, {DisplayFunction -> Identity, GridLines -> {None, None}, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, DisplayFunction -> Identity, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, AspectRatio -> GoldenRatio^(-1), Axes -> {True, True}, AxesLabel -> {None, None}, AxesOrigin -> {0., -0.31359656347995973}, DisplayFunction :> Identity, Frame -> {{False, False}, {False, False}}, FrameLabel -> {{None, None}, {None, None}}, FrameTicks -> {{Charting`ScaledTicks[{Log, Exp}, {Log, Exp}, "Nice", WorkingPrecision -> 15.954589770191003, RotateLabel -> 0], Charting`ScaledFrameTicks[{Log, Exp}]}, {Automatic, Automatic}}, GridLines -> {None, None}, GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]], Method -> {"AxisPadding" -> Scaled[0.02], "DefaultBoundaryStyle" -> Automatic, "DefaultGraphicsInteraction" -> {"Version" -> 1.2, "TrackMousePosition" -> {True, False}, "Effects" -> {"Highlight" -> {"ratio" -> 2}, "HighlightPoint" -> {"ratio" -> 2}, "Droplines" -> {"freeformCursorMode" -> True, "placement" -> {"x" -> "All", "y" -> "None"}}}}, "DefaultMeshStyle" -> AbsolutePointSize[6], "DefaultPlotStyle" -> {Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Directive[RGBColor[0.95, 0.627, 0.1425], AbsoluteThickness[2]], Directive[RGBColor[0.455, 0.7, 0.21], AbsoluteThickness[2]], Directive[RGBColor[0.922526, 0.385626, 0.209179], AbsoluteThickness[2]], Directive[RGBColor[0.578, 0.51, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.772079, 0.431554, 0.102387], AbsoluteThickness[2]], Directive[RGBColor[0.4, 0.64, 1.], AbsoluteThickness[2]], Directive[RGBColor[1., 0.75, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.8, 0.4, 0.76], AbsoluteThickness[2]], Directive[RGBColor[0.637, 0.65, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.915, 0.3325, 0.2125], AbsoluteThickness[2]], Directive[RGBColor[0.40082222609352647, 0.5220066643438841, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.9728288904374106, 0.621644452187053, 0.07336199581899142], AbsoluteThickness[2]], Directive[RGBColor[0.736782672705901, 0.358, 0.5030266573755369], AbsoluteThickness[2]], Directive[RGBColor[0.28026441037696703, 0.715, 0.4292089322474965], AbsoluteThickness[2]]}, "DomainPadding" -> Scaled[0.02], "PointSizeFunction" -> "SmallPointSize", "RangePadding" -> Scaled[0.05], "OptimizePlotMarkers" -> True, "IncludeHighlighting" -> Automatic, "HighlightStyle" -> Automatic, "OptimizePlotMarkers" -> True, "IncludeHighlighting" -> "CurrentPoint", "HighlightStyle" -> Automatic, "OptimizePlotMarkers" -> True, "CoordinatesToolOptions" -> {"DisplayFunction" -> ({(Identity[#1] & )[#1[[1]]], (Exp[#1] & )[#1[[2]]]} & ), "CopiedValueFunction" -> ({(Identity[#1] & )[#1[[1]]], (Exp[#1] & )[#1[[2]]]} & )}}, PlotInteractivity :> <|"SystemLimits" -> 3000, "UserLimits" -> 10000, "UserInteractivity" -> True|>, PlotRange -> {{0., 10.}, {-0.31359656347995973, 4.007333185232471}}, PlotRangeClipping -> True, PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.02], Scaled[0.05]}}, Ticks -> {Automatic, Charting`ScaledTicks[{Log, Exp}, {Log, Exp}, "Nice", WorkingPrecision -> 15.954589770191003, RotateLabel -> 0]}, PlotInteractivity :> $PlotInteractivity}]"#,
    );
  }
  #[test]
  fn list_log_plot_2() {
    assert_case(
      r#"ListLogPlot[Table[Fibonacci[n], {n, 10}]]; ListLogPlot[Table[n!, {n, 10}], Joined -> True]"#,
      r#"Graphics[{{}, Annotation[{{{}, {}, Annotation[{Hue[0.67, 0.6, 0.6], Directive[PointSize[0.012833333333333334], RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Line[{{1., 0.}, {2., 0.6931471805599453}, {3., 1.791759469228055}, {4., 3.1780538303479458}, {5., 4.787491742782046}, {6., 6.579251212010101}, {7., 8.525161361065415}, {8., 10.60460290274525}, {9., 12.801827480081469}, {10., 15.104412573075516}}]}, "Charting`Private`Tag#1"]}}, <|"HighlightElements" -> <|"Label" -> {"XYLabel"}, "Ball" -> {"IndicatedBall"}|>, "LayoutOptions" -> <|"PanelPlotLayout" -> <||>, "PlotRange" -> {{0., 10.}, {-1.1820060018356562, 15.104412573075516}}, "Frame" -> {{False, False}, {False, False}}, "AxesOrigin" -> {0., -1.1820060018356562}, "ImageSize" -> {360, 360/GoldenRatio}, "Axes" -> {True, True}, "LabelStyle" -> {}, "AspectRatio" -> GoldenRatio^(-1), "DefaultStyle" -> {Directive[PointSize[0.012833333333333334], RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]]}, "HighlightLabelingFunctions" -> <|"CoordinatesToolOptions" -> ({(Identity[#1] & )[#1[[1]]], (Exp[#1] & )[#1[[2]]]} & ), "ScalingFunctions" -> {{Identity, Identity}, {Log, Exp}}|>, "Primitives" -> {}, "GCFlag" -> False|>, "Meta" -> <|"DefaultHighlight" -> {"Dynamic", None}, "Index" -> {}, "Function" -> ListLogPlot, "GroupHighlight" -> False|>|>, "DynamicHighlight"], {{}, {}}}, {DisplayFunction -> Identity, GridLines -> {None, None}, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, DisplayFunction -> Identity, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, AspectRatio -> GoldenRatio^(-1), Axes -> {True, True}, AxesLabel -> {None, None}, AxesOrigin -> {0., -1.1820060018356562}, DisplayFunction :> Identity, Frame -> {{False, False}, {False, False}}, FrameLabel -> {{None, None}, {None, None}}, FrameTicks -> {{Charting`ScaledTicks[{Log, Exp}, {Log, Exp}, "Nice", WorkingPrecision -> 15.954589770191003, RotateLabel -> 0], Charting`ScaledFrameTicks[{Log, Exp}]}, {Automatic, Automatic}}, GridLines -> {None, None}, GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]], Method -> {"AxisPadding" -> Scaled[0.02], "DefaultBoundaryStyle" -> Automatic, "DefaultGraphicsInteraction" -> {"Version" -> 1.2, "TrackMousePosition" -> {True, False}, "Effects" -> {"Highlight" -> {"ratio" -> 2}, "HighlightPoint" -> {"ratio" -> 2}, "Droplines" -> {"freeformCursorMode" -> True, "placement" -> {"x" -> "All", "y" -> "None"}}}}, "DefaultMeshStyle" -> AbsolutePointSize[6], "DefaultPlotStyle" -> {Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Directive[RGBColor[0.95, 0.627, 0.1425], AbsoluteThickness[2]], Directive[RGBColor[0.455, 0.7, 0.21], AbsoluteThickness[2]], Directive[RGBColor[0.922526, 0.385626, 0.209179], AbsoluteThickness[2]], Directive[RGBColor[0.578, 0.51, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.772079, 0.431554, 0.102387], AbsoluteThickness[2]], Directive[RGBColor[0.4, 0.64, 1.], AbsoluteThickness[2]], Directive[RGBColor[1., 0.75, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.8, 0.4, 0.76], AbsoluteThickness[2]], Directive[RGBColor[0.637, 0.65, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.915, 0.3325, 0.2125], AbsoluteThickness[2]], Directive[RGBColor[0.40082222609352647, 0.5220066643438841, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.9728288904374106, 0.621644452187053, 0.07336199581899142], AbsoluteThickness[2]], Directive[RGBColor[0.736782672705901, 0.358, 0.5030266573755369], AbsoluteThickness[2]], Directive[RGBColor[0.28026441037696703, 0.715, 0.4292089322474965], AbsoluteThickness[2]]}, "DomainPadding" -> Scaled[0.02], "PointSizeFunction" -> "SmallPointSize", "RangePadding" -> Scaled[0.05], "OptimizePlotMarkers" -> True, "IncludeHighlighting" -> Automatic, "HighlightStyle" -> Automatic, "OptimizePlotMarkers" -> True, "IncludeHighlighting" -> "CurrentSet", "HighlightStyle" -> Automatic, "OptimizePlotMarkers" -> True, "CoordinatesToolOptions" -> {"DisplayFunction" -> ({(Identity[#1] & )[#1[[1]]], (Exp[#1] & )[#1[[2]]]} & ), "CopiedValueFunction" -> ({(Identity[#1] & )[#1[[1]]], (Exp[#1] & )[#1[[2]]]} & )}}, PlotInteractivity :> <|"SystemLimits" -> 3000, "UserLimits" -> 10000, "UserInteractivity" -> True|>, PlotRange -> {{0., 10.}, {-1.1820060018356562, 15.104412573075516}}, PlotRangeClipping -> True, PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.02], Scaled[0.05]}}, Ticks -> {Automatic, Charting`ScaledTicks[{Log, Exp}, {Log, Exp}, "Nice", WorkingPrecision -> 15.954589770191003, RotateLabel -> 0]}, PlotInteractivity :> $PlotInteractivity}]"#,
    );
  }
  #[test]
  fn number_line_plot_1() {
    assert_case(
      r#"NumberLinePlot[Prime[Range[10]]]"#,
      r#"Graphics[{{RGBColor[0.24720000000000017, 0.24, 0.6], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{2, 1}]}}, {RGBColor[0.6, 0.24, 0.4428931686004542], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{3, 1}]}}, {RGBColor[0.6, 0.5470136627990908, 0.24], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{5, 1}]}}, {RGBColor[0.24, 0.6, 0.33692049419863584], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{7, 1}]}}, {RGBColor[0.24, 0.35317267440181815, 0.6], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{11, 1}]}}, {RGBColor[0.6, 0.24, 0.5632658430022722], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{13, 1}]}}, {RGBColor[0.6, 0.4266409883972719, 0.24], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{17, 1}]}}, {RGBColor[0.2634521802031821, 0.6, 0.24], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{19, 1}]}}, {RGBColor[0.24, 0.47354534880363613, 0.6], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{23, 1}]}}, {RGBColor[0.5163614825959097, 0.24, 0.6], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{29, 1}]}}}, AxesLabel -> {None}, Ticks -> {Automatic, Automatic}, FrameTicks -> {{Automatic, Automatic}, {Automatic, Automatic}}, PlotRange -> {{2., 29.}, {0, 1}}, PlotRangePadding -> {{Scaled[0.1], Scaled[0.1]}, {0, 1}}, AspectRatio -> 1/(10*GoldenRatio), AxesOrigin -> {Automatic, Automatic}, Axes -> {True, False}, ImagePadding -> All, {}]"#,
    );
  }
  #[test]
  fn number_line_plot_2() {
    assert_case(
      r#"NumberLinePlot[Prime[Range[10]]]; NumberLinePlot[Table[x^2, {x, 10}]]"#,
      r#"Graphics[{{RGBColor[0.24720000000000017, 0.24, 0.6], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{1, 1}]}}, {RGBColor[0.6, 0.24, 0.4428931686004542], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{4, 1}]}}, {RGBColor[0.6, 0.5470136627990908, 0.24], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{9, 1}]}}, {RGBColor[0.24, 0.6, 0.33692049419863584], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{16, 1}]}}, {RGBColor[0.24, 0.35317267440181815, 0.6], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{25, 1}]}}, {RGBColor[0.6, 0.24, 0.5632658430022722], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{36, 1}]}}, {RGBColor[0.6, 0.4266409883972719, 0.24], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{49, 1}]}}, {RGBColor[0.2634521802031821, 0.6, 0.24], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{64, 1}]}}, {RGBColor[0.24, 0.47354534880363613, 0.6], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{81, 1}]}}, {RGBColor[0.5163614825959097, 0.24, 0.6], PointSize[Medium], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[1.6]], {Point[{100, 1}]}}}, AxesLabel -> {None}, Ticks -> {Automatic, Automatic}, FrameTicks -> {{Automatic, Automatic}, {Automatic, Automatic}}, PlotRange -> {{1., 100.}, {0, 1}}, PlotRangePadding -> {{Scaled[0.1], Scaled[0.1]}, {0, 1}}, AspectRatio -> 1/(10*GoldenRatio), AxesOrigin -> {Automatic, Automatic}, Axes -> {True, False}, ImagePadding -> All, {}]"#,
    );
  }
  #[test]
  fn precision() {
    assert_case(
      r#"AnglePath[{90 Degree, 90 Degree, 90 Degree, 90 Degree}]; AnglePath[{{1, 1}, 90 Degree}, {{1, 90 Degree}, {2, 90 Degree}, {1, 90 Degree}, {2, 90 Degree}}]; AnglePath[{a, b}]; Precision[Part[AnglePath[{N[1/3, 100], N[2/3, 100]}], 2, 1]]"#,
      r#"100.9377270205895"#,
    );
  }
  #[test]
  fn from_continued_fraction() {
    assert_case(
      r#"FromContinuedFraction[{3, 7, 15, 1, 292, 1, 1, 1, 2, 1}]; FromContinuedFraction[Range[5]]"#,
      r#"225 / 157"#,
    );
  }
  #[test]
  fn table_2() {
    assert_case(
      r#"Table[JacobiSymbol[n, m], {n, 0, 10}, {m, 1, n, 2}]"#,
      r#"{{}, {1}, {1}, {1, 0}, {1, 1}, {1, -1, 0}, {1, 0, 1}, {1, 1, -1, 0}, {1, -1, -1, 1}, {1, 0, 1, 1, 0}, {1, 1, 0, -1, 1}}"#,
    );
  }
  #[test]
  fn table_3() {
    assert_case(
      r#"Table[KroneckerSymbol[n, m], {n, 5}, {m, 5}]"#,
      r#"{{1, 1, 1, 1, 1}, {1, 0, -1, 0, -1}, {1, -1, 0, 1, -1}, {1, 0, 1, 0, 1}, {1, -1, -1, 1, 0}}"#,
    );
  }
  #[test]
  fn table_4() {
    assert_case(
      r#"Table[PartitionsP[k], {k, -2, 12}]"#,
      r#"{0, 0, 1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77}"#,
    );
  }
  #[test]
  fn table_5() {
    assert_case(
      r#"Table[SquaresR[2, n], {n, 10}]"#,
      r#"{4, 4, 0, 4, 8, 0, 0, 4, 4, 8}"#,
    );
  }
  #[test]
  fn table_6() {
    assert_case(
      r#"Table[SquaresR[2, n], {n, 10}]; Table[Sum[SquaresR[2, k], {k, 0, n^2}], {n, 5}]"#,
      r#"{5, 13, 29, 49, 81}"#,
    );
  }
  #[test]
  fn table_7() {
    assert_case(
      r#"Table[SquaresR[2, n], {n, 10}]; Table[Sum[SquaresR[2, k], {k, 0, n^2}], {n, 5}]; Table[SquaresR[4, n], {n, 10}]"#,
      r#"{8, 24, 32, 24, 48, 96, 64, 24, 104, 144}"#,
    );
  }
  #[test]
  fn table_8() {
    assert_case(
      r#"Table[SquaresR[2, n], {n, 10}]; Table[Sum[SquaresR[2, k], {k, 0, n^2}], {n, 5}]; Table[SquaresR[4, n], {n, 10}]; Table[SquaresR[6, n], {n, 10}]"#,
      r#"{12, 60, 160, 252, 312, 544, 960, 1020, 876, 1560}"#,
    );
  }
  #[test]
  fn table_9() {
    assert_case(
      r#"Table[SquaresR[2, n], {n, 10}]; Table[Sum[SquaresR[2, k], {k, 0, n^2}], {n, 5}]; Table[SquaresR[4, n], {n, 10}]; Table[SquaresR[6, n], {n, 10}]; Table[SquaresR[8, n], {n, 10}]"#,
      r#"{16, 112, 448, 1136, 2016, 3136, 5504, 9328, 12112, 14112}"#,
    );
  }
  #[test]
  fn member_q_1() {
    assert_case(r#"$PrintForms; MemberQ[$PrintForms, MyForm]"#, r#"False"#);
  }
  #[test]
  fn table_form() {
    assert_case(
      r#"TableForm[Array[a, {3,2}],TableDepth->1]"#,
      r#"TableForm[{{a[1, 1], a[1, 2]}, {a[2, 1], a[2, 2]}, {a[3, 1], a[3, 2]}}, TableDepth -> 1]"#,
    );
  }
  #[test]
  fn array_1() {
    assert_case(
      r#"Array[a,{4,3}]//MatrixForm"#,
      r#"MatrixForm[{{a[1, 1], a[1, 2], a[1, 3]}, {a[2, 1], a[2, 2], a[2, 3]}, {a[3, 1], a[3, 2], a[3, 3]}, {a[4, 1], a[4, 2], a[4, 3]}}]"#,
    );
  }
  #[test]
  fn nearest_1() {
    assert_case(r#"Nearest[{5, 2.5, 10, 11, 15, 8.5, 14}, 12]"#, r#"{11}"#);
  }
  #[test]
  fn nearest_2() {
    assert_case(
      r#"Nearest[{5, 2.5, 10, 11, 15, 8.5, 14}, 12]; Nearest[{5, 2.5, 10, 11, 15, 8.5, 14}, 12, {All, 5}]"#,
      r#"{11, 10, 14, 15, 8.5}"#,
    );
  }
  #[test]
  fn nearest_3() {
    assert_case(
      r#"Nearest[{5, 2.5, 10, 11, 15, 8.5, 14}, 12]; Nearest[{5, 2.5, 10, 11, 15, 8.5, 14}, 12, {All, 5}]; Nearest[{Blue -> "blue", White -> "white", Red -> "red", Green -> "green"}, {Orange, Gray}]; Nearest[{{0, 1}, {1, 2}, {2, 3}} -> {a, b, c}, {1.1, 2}]"#,
      r#"{b}"#,
    );
  }
  #[test]
  fn list_q_1() {
    assert_case(r#"ListQ[{1, 2, 3}]"#, r#"True"#);
  }
  #[test]
  fn list_q_2() {
    assert_case(r#"ListQ[{1, 2, 3}]; ListQ[{{1, 2}, {3, 4}}]"#, r#"True"#);
  }
  #[test]
  fn list_q_3() {
    assert_case(
      r#"ListQ[{1, 2, 3}]; ListQ[{{1, 2}, {3, 4}}]; ListQ[x]"#,
      r#"False"#,
    );
  }
  #[test]
  fn ordered_q_1() {
    assert_case(r#"OrderedQ[{a, b}]"#, r#"True"#);
  }
  #[test]
  fn ordered_q_2() {
    assert_case(r#"OrderedQ[{a, b}]; OrderedQ[{b, a}]"#, r#"False"#);
  }
  #[test]
  fn any_true_1() {
    assert_case(r#"AnyTrue[{1, 3, 5}, EvenQ]"#, r#"False"#);
  }
  #[test]
  fn any_true_2() {
    assert_case(
      r#"AnyTrue[{1, 3, 5}, EvenQ]; AnyTrue[{1, 4, 5}, EvenQ]"#,
      r#"True"#,
    );
  }
  #[test]
  fn all_true_1() {
    assert_case(r#"AllTrue[{2, 4, 6}, EvenQ]"#, r#"True"#);
  }
  #[test]
  fn all_true_2() {
    assert_case(
      r#"AllTrue[{2, 4, 6}, EvenQ]; AllTrue[{2, 4, 7}, EvenQ]"#,
      r#"False"#,
    );
  }
  #[test]
  fn none_true_1() {
    assert_case(r#"NoneTrue[{1, 3, 5}, EvenQ]"#, r#"True"#);
  }
  #[test]
  fn none_true_2() {
    assert_case(
      r#"NoneTrue[{1, 3, 5}, EvenQ]; NoneTrue[{1, 4, 5}, EvenQ]"#,
      r#"False"#,
    );
  }
  #[test]
  fn array_q_1() {
    assert_case(r#"ArrayQ[a]"#, r#"False"#);
  }
  #[test]
  fn array_q_2() {
    assert_case(r#"ArrayQ[a]; ArrayQ[{a}]"#, r#"True"#);
  }
  #[test]
  fn array_q_3() {
    assert_case(
      r#"ArrayQ[a]; ArrayQ[{a}]; ArrayQ[{{{a}},{{b,c}}}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn array_q_4() {
    assert_case(
      r#"ArrayQ[a]; ArrayQ[{a}]; ArrayQ[{{{a}},{{b,c}}}]; ArrayQ[{{a, b}, {c, d}}, 2, SymbolQ]"#,
      r#"False"#,
    );
  }
  #[test]
  fn level_q_1() {
    assert_case(r#"LevelQ[2]"#, r#"LevelQ[2]"#);
  }
  #[test]
  fn level_q_2() {
    assert_case(r#"LevelQ[2]; LevelQ[{2, 4}]"#, r#"LevelQ[{2, 4}]"#);
  }
  #[test]
  fn level_q_3() {
    assert_case(
      r#"LevelQ[2]; LevelQ[{2, 4}]; LevelQ[Infinity]"#,
      r#"LevelQ[Infinity]"#,
    );
  }
  #[test]
  fn level_q_4() {
    assert_case(
      r#"LevelQ[2]; LevelQ[{2, 4}]; LevelQ[Infinity]; LevelQ[a + b]"#,
      r#"LevelQ[a + b]"#,
    );
  }
  #[test]
  fn member_q_2() {
    assert_case(r#"MemberQ[{a, b, c}, b]"#, r#"True"#);
  }
  #[test]
  fn member_q_3() {
    assert_case(
      r#"MemberQ[{a, b, c}, b]; MemberQ[{a, b, c}, d]"#,
      r#"False"#,
    );
  }
  #[test]
  fn member_q_4() {
    assert_case(
      r#"MemberQ[{a, b, c}, b]; MemberQ[{a, b, c}, d]; MemberQ[{"a", b, f[x]}, _?NumericQ]"#,
      r#"False"#,
    );
  }
  #[test]
  fn member_q_5() {
    assert_case(
      r#"MemberQ[{a, b, c}, b]; MemberQ[{a, b, c}, d]; MemberQ[{"a", b, f[x]}, _?NumericQ]; MemberQ[_List][{{}}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn subset_q_1() {
    assert_case(r#"SubsetQ[{1, 2, 3}, {3, 1}]"#, r#"True"#);
  }
  #[test]
  fn subset_q_2() {
    assert_case(r#"SubsetQ[{1, 2, 3}, {3, 1}]; SubsetQ[{}, {}]"#, r#"True"#);
  }
  #[test]
  fn subset_q_3() {
    assert_case(
      r#"SubsetQ[{1, 2, 3}, {3, 1}]; SubsetQ[{}, {}]; SubsetQ[{1, 2, 3}, {}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn subset_q_4() {
    assert_case(
      r#"SubsetQ[{1, 2, 3}, {3, 1}]; SubsetQ[{}, {}]; SubsetQ[{1, 2, 3}, {}]; SubsetQ[{1, 2, 3}, {1, 2, 3}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn select_1() {
    assert_case(
      r#"PrimeQ[2]; PrimeQ[-3]; PrimeQ[137]; PrimeQ[2 ^ 127 - 1]; Select[Range[100], PrimeQ]"#,
      r#"{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}"#,
    );
  }
  #[test]
  fn prime_q() {
    assert_case(
      r#"PrimeQ[2]; PrimeQ[-3]; PrimeQ[137]; PrimeQ[2 ^ 127 - 1]; Select[Range[100], PrimeQ]; PrimeQ[Range[20]]"#,
      r#"{False, True, True, False, True, False, True, False, False, False, True, False, True, False, False, False, True, False, True, False}"#,
    );
  }
  #[test]
  fn table_10() {
    assert_case(
      r#"Table[PauliMatrix[i], {i, 1, 3}]"#,
      r#"{{{0, 1}, {1, 0}}, {{0, -I}, {I, 0}}, {{1, 0}, {0, -1}}}"#,
    );
  }
  #[test]
  fn pauli_matrix() {
    assert_case(
      r#"Table[PauliMatrix[i], {i, 1, 3}]; PauliMatrix[1] . PauliMatrix[2] == I PauliMatrix[3]"#,
      r#"True"#,
    );
  }
  #[test]
  fn character_encodings() {
    assert_case(
      r#"$CharacterEncodings[[;;9]]"#,
      r#"{"AdobeStandard", "ASCII", "CP936", "CP949", "CP950", "EUC-JP", "EUC", "IBM-850", "ISO8859-10"}"#,
    );
  }
  #[test]
  fn accumulate() {
    assert_case(r#"Accumulate[{1, 2, 3}]"#, r#"{1, 3, 6}"#);
  }
  #[test]
  fn depth_1() {
    assert_case(r#"Depth[x]"#, r#"1"#);
  }
  #[test]
  fn depth_2() {
    assert_case(r#"Depth[x]; Depth[x + y]"#, r#"2"#);
  }
  #[test]
  fn depth_3() {
    assert_case(r#"Depth[x]; Depth[x + y]; Depth[{{{{x}}}}]"#, r#"5"#);
  }
  #[test]
  fn depth_4() {
    assert_case(
      r#"Depth[x]; Depth[x + y]; Depth[{{{{x}}}}]; Depth[1 + 2 I]"#,
      r#"1"#,
    );
  }
  #[test]
  fn depth_5() {
    assert_case(
      r#"Depth[x]; Depth[x + y]; Depth[{{{{x}}}}]; Depth[1 + 2 I]; Depth[f[a, b][c]]"#,
      r#"2"#,
    );
  }
  #[test]
  fn contains_only_1() {
    assert_case(r#"ContainsOnly[{b, a, a}, {a, b, c}]"#, r#"True"#);
  }
  #[test]
  fn contains_only_2() {
    assert_case(
      r#"ContainsOnly[{b, a, a}, {a, b, c}]; ContainsOnly[{b, a, d}, {a, b, c}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn contains_only_3() {
    assert_case(
      r#"ContainsOnly[{b, a, a}, {a, b, c}]; ContainsOnly[{b, a, d}, {a, b, c}]; ContainsOnly[{}, {a, b, c}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn contains_only_4() {
    assert_case(
      r#"ContainsOnly[{b, a, a}, {a, b, c}]; ContainsOnly[{b, a, d}, {a, b, c}]; ContainsOnly[{}, {a, b, c}]; ContainsOnly[{a, 1.0}, {1, a, b}, {SameTest -> Equal}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn catenate() {
    assert_case(r#"Catenate[{{1, 2, 3}, {4, 5}}]"#, r#"{1, 2, 3, 4, 5}"#);
  }
  #[test]
  fn complement_1() {
    assert_case(r#"Complement[{a, b, c}, {a, c}]"#, r#"{b}"#);
  }
  #[test]
  fn complement_2() {
    assert_case(
      r#"Complement[{a, b, c}, {a, c}]; Complement[{a, b, c}, {a, c}, {b}]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn complement_3() {
    assert_case(
      r#"Complement[{a, b, c}, {a, c}]; Complement[{a, b, c}, {a, c}, {b}]; Complement[f[z, y, x, w], f[x], f[x, z]]"#,
      r#"f[w, y]"#,
    );
  }
  #[test]
  fn complement_4() {
    assert_case(
      r#"Complement[{a, b, c}, {a, c}]; Complement[{a, b, c}, {a, c}, {b}]; Complement[f[z, y, x, w], f[x], f[x, z]]; Complement[{c, b, a}]"#,
      r#"{a, b, c}"#,
    );
  }
  #[test]
  fn delete_duplicates_1() {
    assert_case(
      r#"DeleteDuplicates[{1, 7, 8, 4, 3, 4, 1, 9, 9, 2, 1}]"#,
      r#"{1, 7, 8, 4, 3, 9, 2}"#,
    );
  }
  #[test]
  fn delete_duplicates_2() {
    assert_case(
      r#"DeleteDuplicates[{1, 7, 8, 4, 3, 4, 1, 9, 9, 2, 1}]; DeleteDuplicates[{3,2,1,2,3,4}, Less]"#,
      r#"{3, 2, 1}"#,
    );
  }
  #[test]
  fn gather_1() {
    assert_case(
      r#"Gather[{1, 7, 3, 7, 2, 3, 9}]"#,
      r#"{{1}, {7, 7}, {3, 3}, {2}, {9}}"#,
    );
  }
  #[test]
  fn gather_2() {
    assert_case(
      r#"Gather[{1, 7, 3, 7, 2, 3, 9}]; Gather[{1/3, 2/6, 1/9}]"#,
      r#"{{1 / 3, 1 / 3}, {1 / 9}}"#,
    );
  }
  #[test]
  fn flatten_1() {
    assert_case(
      r#"Flatten[{{a, b}, {c, {d}, e}, {f, {g, h}}}]"#,
      r#"{a, b, c, d, e, f, g, h}"#,
    );
  }
  #[test]
  fn flatten_2() {
    assert_case(
      r#"Flatten[{{a, b}, {c, {d}, e}, {f, {g, h}}}]; Flatten[{{a, b}, {c, {e}, e}, {f, {g, h}}}, 1]"#,
      r#"{a, b, c, {e}, e, f, {g, h}}"#,
    );
  }
  #[test]
  fn flatten_3() {
    assert_case(
      r#"Flatten[{{a, b}, {c, {d}, e}, {f, {g, h}}}]; Flatten[{{a, b}, {c, {e}, e}, {f, {g, h}}}, 1]; Flatten[f[a, f[b, f[c, d]], e], Infinity, f]"#,
      r#"f[a, b, c, d, e]"#,
    );
  }
  #[test]
  fn flatten_4() {
    assert_case(
      r#"Flatten[{{a, b}, {c, {d}, e}, {f, {g, h}}}]; Flatten[{{a, b}, {c, {e}, e}, {f, {g, h}}}, 1]; Flatten[f[a, f[b, f[c, d]], e], Infinity, f]; Flatten[{{a, b}, {c, d}}, {{2}, {1}}]"#,
      r#"{{a, c}, {b, d}}"#,
    );
  }
  #[test]
  fn flatten_5() {
    assert_case(
      r#"Flatten[{{a, b}, {c, {d}, e}, {f, {g, h}}}]; Flatten[{{a, b}, {c, {e}, e}, {f, {g, h}}}, 1]; Flatten[f[a, f[b, f[c, d]], e], Infinity, f]; Flatten[{{a, b}, {c, d}}, {{2}, {1}}]; Flatten[{{a, b}, {c, d}}, {{1, 2}}]"#,
      r#"{a, b, c, d}"#,
    );
  }
  #[test]
  fn flatten_6() {
    assert_case(
      r#"Flatten[{{a, b}, {c, {d}, e}, {f, {g, h}}}]; Flatten[{{a, b}, {c, {e}, e}, {f, {g, h}}}, 1]; Flatten[f[a, f[b, f[c, d]], e], Infinity, f]; Flatten[{{a, b}, {c, d}}, {{2}, {1}}]; Flatten[{{a, b}, {c, d}}, {{1, 2}}]; Flatten[{{1, 2, 3}, {4}, {6, 7}, {8, 9, 10}}, {{2}, {1}}]"#,
      r#"{{1, 4, 6, 8}, {2, 7, 9}, {3, 10}}"#,
    );
  }
  #[test]
  fn gather_by_1() {
    assert_case(
      r#"GatherBy[{{1, 3}, {2, 2}, {1, 1}}, Total]"#,
      r#"{{{1, 3}, {2, 2}}, {{1, 1}}}"#,
    );
  }
  #[test]
  fn gather_by_2() {
    assert_case(
      r#"GatherBy[{{1, 3}, {2, 2}, {1, 1}}, Total]; GatherBy[{"xy", "abc", "ab"}, StringLength]"#,
      r#"{{"xy", "ab"}, {"abc"}}"#,
    );
  }
  #[test]
  fn gather_by_3() {
    assert_case(
      r#"GatherBy[{{1, 3}, {2, 2}, {1, 1}}, Total]; GatherBy[{"xy", "abc", "ab"}, StringLength]; GatherBy[{{2, 0}, {1, 5}, {1, 0}}, Last]"#,
      r#"{{{2, 0}, {1, 0}}, {{1, 5}}}"#,
    );
  }
  #[test]
  fn gather_by_4() {
    assert_case(
      r#"GatherBy[{{1, 3}, {2, 2}, {1, 1}}, Total]; GatherBy[{"xy", "abc", "ab"}, StringLength]; GatherBy[{{2, 0}, {1, 5}, {1, 0}}, Last]; GatherBy[{{1, 2}, {2, 1}, {3, 5}, {5, 1}, {2, 2, 2}}, {Total, Length}]"#,
      r#"{{{{1, 2}, {2, 1}}}, {{{3, 5}}}, {{{5, 1}}, {{2, 2, 2}}}}"#,
    );
  }
  #[test]
  fn join_1() {
    assert_case(r#"Join[{a, b}, {c, d, e}]"#, r#"{a, b, c, d, e}"#);
  }
  #[test]
  fn join_2() {
    assert_case(
      r#"Join[{a, b}, {c, d, e}]; Join[{{a, b}, {c, d}}, {{1, 2}, {3, 4}}]"#,
      r#"{{a, b}, {c, d}, {1, 2}, {3, 4}}"#,
    );
  }
  #[test]
  fn join_3() {
    assert_case(
      r#"Join[{a, b}, {c, d, e}]; Join[{{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Join[a + b, c + d, e + f]"#,
      r#"a + b + c + d + e + f"#,
    );
  }
  #[test]
  fn pad_left_1() {
    assert_case(r#"PadLeft[{1, 2, 3}, 5]"#, r#"{0, 0, 1, 2, 3}"#);
  }
  #[test]
  fn pad_left_2() {
    assert_case(
      r#"PadLeft[{1, 2, 3}, 5]; PadLeft[x[a, b, c], 5]"#,
      r#"x[0, 0, a, b, c]"#,
    );
  }
  #[test]
  fn pad_left_3() {
    assert_case(
      r#"PadLeft[{1, 2, 3}, 5]; PadLeft[x[a, b, c], 5]; PadLeft[{1, 2, 3}, 2]"#,
      r#"{2, 3}"#,
    );
  }
  #[test]
  fn pad_left_4() {
    assert_case(
      r#"PadLeft[{1, 2, 3}, 5]; PadLeft[x[a, b, c], 5]; PadLeft[{1, 2, 3}, 2]; PadLeft[{{}, {1, 2}, {1, 2, 3}}]"#,
      r#"{{0, 0, 0}, {0, 1, 2}, {1, 2, 3}}"#,
    );
  }
  #[test]
  fn pad_left_5() {
    assert_case(
      r#"PadLeft[{1, 2, 3}, 5]; PadLeft[x[a, b, c], 5]; PadLeft[{1, 2, 3}, 2]; PadLeft[{{}, {1, 2}, {1, 2, 3}}]; PadLeft[{1, 2, 3}, 10, {a, b, c}, 2]"#,
      r#"{b, c, a, b, c, 1, 2, 3, a, b}"#,
    );
  }
  #[test]
  fn pad_left_6() {
    assert_case(
      r#"PadLeft[{1, 2, 3}, 5]; PadLeft[x[a, b, c], 5]; PadLeft[{1, 2, 3}, 2]; PadLeft[{{}, {1, 2}, {1, 2, 3}}]; PadLeft[{1, 2, 3}, 10, {a, b, c}, 2]; PadLeft[{{1, 2, 3}}, {5, 2}, x, 1]"#,
      r#"{{x, x}, {x, x}, {x, x}, {3, x}, {x, x}}"#,
    );
  }
  #[test]
  fn pad_right_1() {
    assert_case(r#"PadRight[{1, 2, 3}, 5]"#, r#"{1, 2, 3, 0, 0}"#);
  }
  #[test]
  fn pad_right_2() {
    assert_case(
      r#"PadRight[{1, 2, 3}, 5]; PadRight[x[a, b, c], 5]"#,
      r#"x[a, b, c, 0, 0]"#,
    );
  }
  #[test]
  fn pad_right_3() {
    assert_case(
      r#"PadRight[{1, 2, 3}, 5]; PadRight[x[a, b, c], 5]; PadRight[{1, 2, 3}, 2]"#,
      r#"{1, 2}"#,
    );
  }
  #[test]
  fn pad_right_4() {
    assert_case(
      r#"PadRight[{1, 2, 3}, 5]; PadRight[x[a, b, c], 5]; PadRight[{1, 2, 3}, 2]; PadRight[{{}, {1, 2}, {1, 2, 3}}]"#,
      r#"{{0, 0, 0}, {1, 2, 0}, {1, 2, 3}}"#,
    );
  }
  #[test]
  fn pad_right_5() {
    assert_case(
      r#"PadRight[{1, 2, 3}, 5]; PadRight[x[a, b, c], 5]; PadRight[{1, 2, 3}, 2]; PadRight[{{}, {1, 2}, {1, 2, 3}}]; PadRight[{1, 2, 3}, 10, {a, b, c}, 2]"#,
      r#"{b, c, 1, 2, 3, a, b, c, a, b}"#,
    );
  }
  #[test]
  fn pad_right_6() {
    assert_case(
      r#"PadRight[{1, 2, 3}, 5]; PadRight[x[a, b, c], 5]; PadRight[{1, 2, 3}, 2]; PadRight[{{}, {1, 2}, {1, 2, 3}}]; PadRight[{1, 2, 3}, 10, {a, b, c}, 2]; PadRight[{{1, 2, 3}}, {5, 2}, x, 1]"#,
      r#"{{x, x}, {x, 1}, {x, x}, {x, x}, {x, x}}"#,
    );
  }
  #[test]
  fn partition_1() {
    assert_case(
      r#"Partition[{a, b, c, d, e, f}, 2]"#,
      r#"{{a, b}, {c, d}, {e, f}}"#,
    );
  }
  #[test]
  fn partition_2() {
    assert_case(
      r#"Partition[{a, b, c, d, e, f}, 2]; Partition[{a, b, c, d, e, f}, 3, 1]"#,
      r#"{{a, b, c}, {b, c, d}, {c, d, e}, {d, e, f}}"#,
    );
  }
  #[test]
  fn reverse_1() {
    assert_case(r#"Reverse[{1, 2, 3}]"#, r#"{3, 2, 1}"#);
  }
  #[test]
  fn reverse_2() {
    assert_case(
      r#"Reverse[{1, 2, 3}]; Reverse[x[a, b, c]]"#,
      r#"x[c, b, a]"#,
    );
  }
  #[test]
  fn reverse_3() {
    assert_case(
      r#"Reverse[{1, 2, 3}]; Reverse[x[a, b, c]]; Reverse[{{1, 2}, {3, 4}}, 1]"#,
      r#"{{3, 4}, {1, 2}}"#,
    );
  }
  #[test]
  fn reverse_4() {
    assert_case(
      r#"Reverse[{1, 2, 3}]; Reverse[x[a, b, c]]; Reverse[{{1, 2}, {3, 4}}, 1]; Reverse[{{1, 2}, {3, 4}}, 2]"#,
      r#"{{2, 1}, {4, 3}}"#,
    );
  }
  #[test]
  fn reverse_5() {
    assert_case(
      r#"Reverse[{1, 2, 3}]; Reverse[x[a, b, c]]; Reverse[{{1, 2}, {3, 4}}, 1]; Reverse[{{1, 2}, {3, 4}}, 2]; Reverse[{{1, 2}, {3, 4}}, {1, 2}]"#,
      r#"{{4, 3}, {2, 1}}"#,
    );
  }
  #[test]
  fn riffle_1() {
    assert_case(r#"Riffle[{a, b, c}, x]"#, r#"{a, x, b, x, c}"#);
  }
  #[test]
  fn riffle_2() {
    assert_case(
      r#"Riffle[{a, b, c}, x]; Riffle[{a, b, c}, {x, y, z}]"#,
      r#"{a, x, b, y, c, z}"#,
    );
  }
  #[test]
  fn riffle_3() {
    assert_case(
      r#"Riffle[{a, b, c}, x]; Riffle[{a, b, c}, {x, y, z}]; Riffle[{a, b, c, d, e, f}, {x, y, z}]"#,
      r#"{a, x, b, y, c, z, d, x, e, y, f}"#,
    );
  }
  #[test]
  fn rotate_left_1() {
    assert_case(r#"RotateLeft[{1, 2, 3}]"#, r#"{2, 3, 1}"#);
  }
  #[test]
  fn rotate_left_2() {
    assert_case(
      r#"RotateLeft[{1, 2, 3}]; RotateLeft[Range[10], 3]"#,
      r#"{4, 5, 6, 7, 8, 9, 10, 1, 2, 3}"#,
    );
  }
  #[test]
  fn rotate_left_3() {
    assert_case(
      r#"RotateLeft[{1, 2, 3}]; RotateLeft[Range[10], 3]; RotateLeft[x[a, b, c], 2]"#,
      r#"x[c, a, b]"#,
    );
  }
  #[test]
  fn rotate_left_4() {
    assert_case(
      r#"RotateLeft[{1, 2, 3}]; RotateLeft[Range[10], 3]; RotateLeft[x[a, b, c], 2]; RotateLeft[{{a, b, c}, {d, e, f}, {g, h, i}}, {1, 2}]"#,
      r#"{{f, d, e}, {i, g, h}, {c, a, b}}"#,
    );
  }
  #[test]
  fn rotate_right_1() {
    assert_case(r#"RotateRight[{1, 2, 3}]"#, r#"{3, 1, 2}"#);
  }
  #[test]
  fn rotate_right_2() {
    assert_case(
      r#"RotateRight[{1, 2, 3}]; RotateRight[Range[10], 3]"#,
      r#"{8, 9, 10, 1, 2, 3, 4, 5, 6, 7}"#,
    );
  }
  #[test]
  fn rotate_right_3() {
    assert_case(
      r#"RotateRight[{1, 2, 3}]; RotateRight[Range[10], 3]; RotateRight[x[a, b, c], 2]"#,
      r#"x[b, c, a]"#,
    );
  }
  #[test]
  fn rotate_right_4() {
    assert_case(
      r#"RotateRight[{1, 2, 3}]; RotateRight[Range[10], 3]; RotateRight[x[a, b, c], 2]; RotateRight[{{a, b, c}, {d, e, f}, {g, h, i}}, {1, 2}]"#,
      r#"{{h, i, g}, {b, c, a}, {e, f, d}}"#,
    );
  }
  #[test]
  fn split_1() {
    assert_case(
      r#"Split[{x, x, x, y, x, y, y, z}]"#,
      r#"{{x, x, x}, {y}, {x}, {y, y}, {z}}"#,
    );
  }
  #[test]
  fn split_2() {
    assert_case(
      r#"Split[{x, x, x, y, x, y, y, z}]; Split[{1, 5, 6, 3, 6, 1, 6, 3, 4, 5, 4}, Less]"#,
      r#"{{1, 5, 6}, {3, 6}, {1, 6}, {3, 4, 5}, {4}}"#,
    );
  }
  #[test]
  fn split_3() {
    assert_case(
      r#"Split[{x, x, x, y, x, y, y, z}]; Split[{1, 5, 6, 3, 6, 1, 6, 3, 4, 5, 4}, Less]; Split[{1, 5, 6, 3, 6, 1, 6, 3, 4, 5, 4}, Greater]"#,
      r#"{{1}, {5}, {6, 3}, {6, 1}, {6, 3}, {4}, {5, 4}}"#,
    );
  }
  #[test]
  fn split_4() {
    assert_case(
      r#"Split[{x, x, x, y, x, y, y, z}]; Split[{1, 5, 6, 3, 6, 1, 6, 3, 4, 5, 4}, Less]; Split[{1, 5, 6, 3, 6, 1, 6, 3, 4, 5, 4}, Greater]; Split[{x -> a, x -> y, 2 -> a, z -> c, z -> a}, First[#1] === First[#2] &]"#,
      r#"{{x -> a, x -> y}, {2 -> a}, {z -> c, z -> a}}"#,
    );
  }
  #[test]
  fn split_by_1() {
    assert_case(
      r#"SplitBy[Range[1, 3, 1/3], Round]"#,
      r#"{{1, 4 / 3}, {5 / 3, 2, 7 / 3}, {8 / 3, 3}}"#,
    );
  }
  #[test]
  fn split_by_2() {
    assert_case(
      r#"SplitBy[Range[1, 3, 1/3], Round]; SplitBy[{1, 2, 1, 1.2}, {Round, Identity}]"#,
      r#"{{{1}}, {{2}}, {{1}, {1.2}}}"#,
    );
  }
  #[test]
  fn union_1() {
    assert_case(r#"Union[{a, b, c}, {c, d, e}]"#, r#"{a, b, c, d, e}"#);
  }
  #[test]
  fn union_2() {
    assert_case(
      r#"Union[{a, b, c}, {c, d, e}]; Union[{a -> b}, {c -> d}]"#,
      r#"{a -> b, c -> d}"#,
    );
  }
  #[test]
  fn union_3() {
    assert_case(
      r#"Union[{a, b, c}, {c, d, e}]; Union[{a -> b}, {c -> d}]; Union[{c, b, a}]"#,
      r#"{a, b, c}"#,
    );
  }
  #[test]
  fn union_4() {
    assert_case(
      r#"Union[{a, b, c}, {c, d, e}]; Union[{a -> b}, {c -> d}]; Union[{c, b, a}]; Union[{5, 1, 3, 7, 1, 8, 3}]"#,
      r#"{1, 3, 5, 7, 8}"#,
    );
  }
  #[test]
  fn union_5() {
    assert_case(
      r#"Union[{a, b, c}, {c, d, e}]; Union[{a -> b}, {c -> d}]; Union[{c, b, a}]; Union[{5, 1, 3, 7, 1, 8, 3}]; Union[{{a, 1}, {b, 2}}, {{c, 1}, {d, 3}}, SameTest->(SameQ[Last[#1],Last[#2]]&)]"#,
      r#"{{a, 1}, {b, 2}, {d, 3}}"#,
    );
  }
  #[test]
  fn union_6() {
    assert_case(
      r#"Union[{a, b, c}, {c, d, e}]; Union[{a -> b}, {c -> d}]; Union[{c, b, a}]; Union[{5, 1, 3, 7, 1, 8, 3}]; Union[{{a, 1}, {b, 2}}, {{c, 1}, {d, 3}}, SameTest->(SameQ[Last[#1],Last[#2]]&)]; Union[{1, 2, 3}, {2, 3, 4}, SameTest->Less]"#,
      r#"{1, 2, 2, 3, 3, 4}"#,
    );
  }
  #[test]
  fn intersection_1() {
    assert_case(
      r#"Intersection[{1000, 100, 10, 1}, {1, 5, 10, 15}]"#,
      r#"{1, 10}"#,
    );
  }
  #[test]
  fn intersection_2() {
    assert_case(
      r#"Intersection[{1000, 100, 10, 1}, {1, 5, 10, 15}]; Intersection[{{a, b}, {x, y}}, {{x, x}, {x, y}, {x, z}}]"#,
      r#"{{x, y}}"#,
    );
  }
  #[test]
  fn intersection_3() {
    assert_case(
      r#"Intersection[{1000, 100, 10, 1}, {1, 5, 10, 15}]; Intersection[{{a, b}, {x, y}}, {{x, x}, {x, y}, {x, z}}]; Intersection[{c, b, a}]"#,
      r#"{a, b, c}"#,
    );
  }
  #[test]
  fn intersection_4() {
    assert_case(
      r#"Intersection[{1000, 100, 10, 1}, {1, 5, 10, 15}]; Intersection[{{a, b}, {x, y}}, {{x, x}, {x, y}, {x, z}}]; Intersection[{c, b, a}]; Intersection[{1, 2, 3}, {2, 3, 4}, SameTest->Less]"#,
      r#"{3}"#,
    );
  }
  #[test]
  fn array_2() {
    assert_case(r#"Array[f, 4]"#, r#"{f[1], f[2], f[3], f[4]}"#);
  }
  #[test]
  fn array_3() {
    assert_case(
      r#"Array[f, 4]; Array[f, {2, 3}]"#,
      r#"{{f[1, 1], f[1, 2], f[1, 3]}, {f[2, 1], f[2, 2], f[2, 3]}}"#,
    );
  }
  #[test]
  fn array_4() {
    assert_case(
      r#"Array[f, 4]; Array[f, {2, 3}]; Array[f, {2, 3}, 3]"#,
      r#"{{f[3, 3], f[3, 4], f[3, 5]}, {f[4, 3], f[4, 4], f[4, 5]}}"#,
    );
  }
  #[test]
  fn array_5() {
    assert_case(
      r#"Array[f, 4]; Array[f, {2, 3}]; Array[f, {2, 3}, 3]; Array[f, {2, 3}, {4, 6}]"#,
      r#"{{f[4, 6], f[4, 7], f[4, 8]}, {f[5, 6], f[5, 7], f[5, 8]}}"#,
    );
  }
  #[test]
  fn array_6() {
    assert_case(
      r#"Array[f, 4]; Array[f, {2, 3}]; Array[f, {2, 3}, 3]; Array[f, {2, 3}, {4, 6}]; Array[f, {2, 3}, 1, Plus]"#,
      r#"f[1, 1] + f[1, 2] + f[1, 3] + f[2, 1] + f[2, 2] + f[2, 3]"#,
    );
  }
  #[test]
  fn constant_array_1() {
    assert_case(r#"ConstantArray[a, 3]"#, r#"{a, a, a}"#);
  }
  #[test]
  fn constant_array_2() {
    assert_case(
      r#"ConstantArray[a, 3]; ConstantArray[a, {2, 3}]"#,
      r#"{{a, a, a}, {a, a, a}}"#,
    );
  }
  #[test]
  fn range_1() {
    assert_case(r#"Range[5]"#, r#"{1, 2, 3, 4, 5}"#);
  }
  #[test]
  fn range_2() {
    assert_case(r#"Range[5]; Range[-3, 2]"#, r#"{-3, -2, -1, 0, 1, 2}"#);
  }
  #[test]
  fn range_3() {
    assert_case(r#"Range[5]; Range[-3, 2]; Range[5, 1, -2]"#, r#"{5, 3, 1}"#);
  }
  #[test]
  fn range_4() {
    assert_case(
      r#"Range[5]; Range[-3, 2]; Range[5, 1, -2]; Range[1.0, 2.3]"#,
      r#"{1., 2.}"#,
    );
  }
  #[test]
  fn range_5() {
    assert_case(
      r#"Range[5]; Range[-3, 2]; Range[5, 1, -2]; Range[1.0, 2.3]; Range[0, 2, 1/3]"#,
      r#"{0, 1 / 3, 2 / 3, 1, 4 / 3, 5 / 3, 2}"#,
    );
  }
  #[test]
  fn range_6() {
    assert_case(
      r#"Range[5]; Range[-3, 2]; Range[5, 1, -2]; Range[1.0, 2.3]; Range[0, 2, 1/3]; Range[1.0, 2.3, .5]"#,
      r#"{1., 1.5, 2.}"#,
    );
  }
  #[test]
  fn permutations_1() {
    assert_case(
      r#"Permutations[{y, 1, x}]"#,
      r#"{{y, 1, x}, {y, x, 1}, {1, y, x}, {1, x, y}, {x, y, 1}, {x, 1, y}}"#,
    );
  }
  #[test]
  fn permutations_2() {
    assert_case(
      r#"Permutations[{y, 1, x}]; Permutations[{a, b, b}]"#,
      r#"{{a, b, b}, {b, a, b}, {b, b, a}}"#,
    );
  }
  #[test]
  fn permutations_3() {
    assert_case(
      r#"Permutations[{y, 1, x}]; Permutations[{a, b, b}]; Permutations[{1, 2, 3}, 2]"#,
      r#"{{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}"#,
    );
  }
  #[test]
  fn permutations_4() {
    assert_case(
      r#"Permutations[{y, 1, x}]; Permutations[{a, b, b}]; Permutations[{1, 2, 3}, 2]; Permutations[{1, 2, 3}, {2}]"#,
      r#"{{1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}"#,
    );
  }
  #[test]
  fn table_11() {
    assert_case(r#"Table[x, 3]"#, r#"{x, x, x}"#);
  }
  #[test]
  fn table_12() {
    assert_case(
      r#"Table[x, 3]; n = 0; Table[n = n + 1, {5}]"#,
      r#"{1, 2, 3, 4, 5}"#,
    );
  }
  #[test]
  fn tuples_1() {
    assert_case(
      r#"Tuples[{a, b, c}, 2]"#,
      r#"{{a, a}, {a, b}, {a, c}, {b, a}, {b, b}, {b, c}, {c, a}, {c, b}, {c, c}}"#,
    );
  }
  #[test]
  fn tuples_2() {
    assert_case(r#"Tuples[{a, b, c}, 2]; Tuples[{}, 2]"#, r#"{}"#);
  }
  #[test]
  fn tuples_3() {
    assert_case(
      r#"Tuples[{a, b, c}, 2]; Tuples[{}, 2]; Tuples[{a, b, c}, 0]"#,
      r#"{{}}"#,
    );
  }
  #[test]
  fn tuples_4() {
    assert_case(
      r#"Tuples[{a, b, c}, 2]; Tuples[{}, 2]; Tuples[{a, b, c}, 0]; Tuples[{{a, b}, {1, 2, 3}}]"#,
      r#"{{a, 1}, {a, 2}, {a, 3}, {b, 1}, {b, 2}, {b, 3}}"#,
    );
  }
  #[test]
  fn tuples_5() {
    assert_case(
      r#"Tuples[{a, b, c}, 2]; Tuples[{}, 2]; Tuples[{a, b, c}, 0]; Tuples[{{a, b}, {1, 2, 3}}]; Tuples[f[a, b, c], 2]"#,
      r#"{f[a, a], f[a, b], f[a, c], f[b, a], f[b, b], f[b, c], f[c, a], f[c, b], f[c, c]}"#,
    );
  }
  #[test]
  fn tuples_6() {
    assert_case(
      r#"Tuples[{a, b, c}, 2]; Tuples[{}, 2]; Tuples[{a, b, c}, 0]; Tuples[{{a, b}, {1, 2, 3}}]; Tuples[f[a, b, c], 2]; Tuples[{f[a, b], g[c, d]}]"#,
      r#"{{a, c}, {a, d}, {b, c}, {b, d}}"#,
    );
  }
  #[test]
  fn append_1() {
    assert_case(r#"Append[{1, 2, 3}, 4]"#, r#"{1, 2, 3, 4}"#);
  }
  #[test]
  fn append_2() {
    assert_case(
      r#"Append[{1, 2, 3}, 4]; Append[f[a, b], c]"#,
      r#"f[a, b, c]"#,
    );
  }
  #[test]
  fn append_3() {
    assert_case(
      r#"Append[{1, 2, 3}, 4]; Append[f[a, b], c]; Append[{a, b}, {c, d}]"#,
      r#"{a, b, {c, d}}"#,
    );
  }
  #[test]
  fn delete_1() {
    assert_case(r#"Delete[{a, b, c, d}, 3]"#, r#"{a, b, d}"#);
  }
  #[test]
  fn delete_2() {
    assert_case(
      r#"Delete[{a, b, c, d}, 3]; Delete[{a, b, c, d}, -2]"#,
      r#"{a, b, d}"#,
    );
  }
  #[test]
  fn delete_3() {
    assert_case(
      r#"Delete[{a, b, c, d}, 3]; Delete[{a, b, c, d}, -2]; Delete[{a, b, c, d}, {{1}, {3}}]"#,
      r#"{b, d}"#,
    );
  }
  #[test]
  fn delete_4() {
    assert_case(
      r#"Delete[{a, b, c, d}, 3]; Delete[{a, b, c, d}, -2]; Delete[{a, b, c, d}, {{1}, {3}}]; Delete[{{a, b}, {c, d}}, {2, 1}]"#,
      r#"{{a, b}, {d}}"#,
    );
  }
  #[test]
  fn delete_5() {
    assert_case(
      r#"Delete[{a, b, c, d}, 3]; Delete[{a, b, c, d}, -2]; Delete[{a, b, c, d}, {{1}, {3}}]; Delete[{{a, b}, {c, d}}, {2, 1}]; Delete[{a, b, c}, 0]; Delete[f[a, b, c, d], 3]"#,
      r#"f[a, b, d]"#,
    );
  }
  #[test]
  fn delete_6() {
    assert_case(
      r#"Delete[{a, b, c, d}, 3]; Delete[{a, b, c, d}, -2]; Delete[{a, b, c, d}, {{1}, {3}}]; Delete[{{a, b}, {c, d}}, {2, 1}]; Delete[{a, b, c}, 0]; Delete[f[a, b, c, d], 3]; Delete[f[a, b, u + v, c], {3, 0}]"#,
      r#"f[a, b, u, v, c]"#,
    );
  }
  #[test]
  fn delete_7() {
    assert_case(
      r#"Delete[{a, b, c, d}, 3]; Delete[{a, b, c, d}, -2]; Delete[{a, b, c, d}, {{1}, {3}}]; Delete[{{a, b}, {c, d}}, {2, 1}]; Delete[{a, b, c}, 0]; Delete[f[a, b, c, d], 3]; Delete[f[a, b, u + v, c], {3, 0}]; Delete[{a, b, c}, 0]; Delete[{a, b, c, d}]"#,
      r#"Delete[{a, b, c, d}]"#,
    );
  }
  #[test]
  fn drop_1() {
    assert_case(r#"Drop[{a, b, c, d}, 3]"#, r#"{d}"#);
  }
  #[test]
  fn drop_2() {
    assert_case(
      r#"Drop[{a, b, c, d}, 3]; Drop[{a, b, c, d}, -2]"#,
      r#"{a, b}"#,
    );
  }
  #[test]
  fn drop_3() {
    assert_case(
      r#"Drop[{a, b, c, d}, 3]; Drop[{a, b, c, d}, -2]; Drop[{a, b, c, d, e}, {2, -2}]"#,
      r#"{a, e}"#,
    );
  }
  #[test]
  fn set_1() {
    assert_case(
      r#"Drop[{a, b, c, d}, 3]; Drop[{a, b, c, d}, -2]; Drop[{a, b, c, d, e}, {2, -2}]; A = Table[i*10 + j, {i, 4}, {j, 4}]"#,
      r#"{{11, 12, 13, 14}, {21, 22, 23, 24}, {31, 32, 33, 34}, {41, 42, 43, 44}}"#,
    );
  }
  #[test]
  fn drop_4() {
    assert_case(
      r#"Drop[{a, b, c, d}, 3]; Drop[{a, b, c, d}, -2]; Drop[{a, b, c, d, e}, {2, -2}]; A = Table[i*10 + j, {i, 4}, {j, 4}]; Drop[A, {2, 3}, {2, 3}]"#,
      r#"{{11, 14}, {41, 44}}"#,
    );
  }
  #[test]
  fn drop_5() {
    assert_case(
      r#"Drop[{a, b, c, d}, 3]; Drop[{a, b, c, d}, -2]; Drop[{a, b, c, d, e}, {2, -2}]; A = Table[i*10 + j, {i, 4}, {j, 4}]; Drop[A, {2, 3}, {2, 3}]; Drop[{a, b, c, d}, 0]"#,
      r#"{a, b, c, d}"#,
    );
  }
  #[test]
  fn drop_6() {
    assert_case(
      r#"Drop[{a, b, c, d}, 3]; Drop[{a, b, c, d}, -2]; Drop[{a, b, c, d, e}, {2, -2}]; A = Table[i*10 + j, {i, 4}, {j, 4}]; Drop[A, {2, 3}, {2, 3}]; Drop[{a, b, c, d}, 0]; Drop[{}, 0]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn first_1() {
    assert_case(r#"First[{a, b, c}]"#, r#"a"#);
  }
  #[test]
  fn first_2() {
    assert_case(r#"First[{a, b, c}]; First[a + b + c]"#, r#"a"#);
  }
  #[test]
  fn insert_1() {
    assert_case(r#"Insert[{a,b,c,d,e}, x, 3]"#, r#"{a, b, x, c, d, e}"#);
  }
  #[test]
  fn insert_2() {
    assert_case(
      r#"Insert[{a,b,c,d,e}, x, 3]; Insert[{a,b,c,d,e}, x, -2]"#,
      r#"{a, b, c, d, x, e}"#,
    );
  }
  #[test]
  fn last_1() {
    assert_case(r#"Last[{a, b, c}]"#, r#"c"#);
  }
  #[test]
  fn last_2() {
    assert_case(r#"Last[{a, b, c}]; Last[a + b + c]"#, r#"c"#);
  }
  #[test]
  fn length_1() {
    assert_case(r#"Length[{1, 2, 3}]"#, r#"3"#);
  }
  #[test]
  fn length_2() {
    assert_case(r#"Length[{1, 2, 3}]; Length[Exp[x]]"#, r#"2"#);
  }
  #[test]
  fn most_1() {
    assert_case(r#"Most[{a, b, c}]"#, r#"{a, b}"#);
  }
  #[test]
  fn most_2() {
    assert_case(r#"Most[{a, b, c}]; Most[a + b + c]"#, r#"a + b"#);
  }
  #[test]
  fn pick_1() {
    assert_case(r#"Pick[{a, b, c}, {False, True, False}]"#, r#"{b}"#);
  }
  #[test]
  fn pick_2() {
    assert_case(
      r#"Pick[{a, b, c}, {False, True, False}]; Pick[f[g[1, 2], h[3, 4]], {{True, False}, {False, True}}]"#,
      r#"f[g[1], h[4]]"#,
    );
  }
  #[test]
  fn pick_3() {
    assert_case(
      r#"Pick[{a, b, c}, {False, True, False}]; Pick[f[g[1, 2], h[3, 4]], {{True, False}, {False, True}}]; Pick[{a, b, c, d, e}, {1, 2, 3.5, 4, 5.5}, _Integer]"#,
      r#"{a, b, d}"#,
    );
  }
  #[test]
  fn prepend_1() {
    assert_case(r#"Prepend[{2, 3, 4}, 1]"#, r#"{1, 2, 3, 4}"#);
  }
  #[test]
  fn prepend_2() {
    assert_case(
      r#"Prepend[{2, 3, 4}, 1]; Prepend[f[b, c], a]"#,
      r#"f[a, b, c]"#,
    );
  }
  #[test]
  fn prepend_3() {
    assert_case(
      r#"Prepend[{2, 3, 4}, 1]; Prepend[f[b, c], a]; Prepend[{c, d}, {a, b}]"#,
      r#"{{a, b}, c, d}"#,
    );
  }
  #[test]
  fn replace_part_1() {
    assert_case(r#"ReplacePart[{a, b, c}, 1 -> t]"#, r#"{t, b, c}"#);
  }
  #[test]
  fn replace_part_2() {
    assert_case(
      r#"ReplacePart[{a, b, c}, 1 -> t]; ReplacePart[{{a, b}, {c, d}}, {2, 1} -> t]"#,
      r#"{{a, b}, {t, d}}"#,
    );
  }
  #[test]
  fn replace_part_3() {
    assert_case(
      r#"ReplacePart[{a, b, c}, 1 -> t]; ReplacePart[{{a, b}, {c, d}}, {2, 1} -> t]; ReplacePart[{{a, b}, {c, d}}, {{2, 1} -> t, {1, 1} -> t}]"#,
      r#"{{t, b}, {t, d}}"#,
    );
  }
  #[test]
  fn replace_part_4() {
    assert_case(
      r#"ReplacePart[{a, b, c}, 1 -> t]; ReplacePart[{{a, b}, {c, d}}, {2, 1} -> t]; ReplacePart[{{a, b}, {c, d}}, {{2, 1} -> t, {1, 1} -> t}]; ReplacePart[{a, b, c}, {{1}, {2}} -> t]"#,
      r#"{t, t, c}"#,
    );
  }
  #[test]
  fn replace_part_5() {
    assert_case(
      r#"ReplacePart[{a, b, c}, 1 -> t]; ReplacePart[{{a, b}, {c, d}}, {2, 1} -> t]; ReplacePart[{{a, b}, {c, d}}, {{2, 1} -> t, {1, 1} -> t}]; ReplacePart[{a, b, c}, {{1}, {2}} -> t]; n = 1; ReplacePart[{a, b, c, d}, {{1}, {3}} :> n++]"#,
      r#"{1, b, 2, d}"#,
    );
  }
  #[test]
  fn replace_part_6() {
    assert_case(
      r#"ReplacePart[{a, b, c}, 1 -> t]; ReplacePart[{{a, b}, {c, d}}, {2, 1} -> t]; ReplacePart[{{a, b}, {c, d}}, {{2, 1} -> t, {1, 1} -> t}]; ReplacePart[{a, b, c}, {{1}, {2}} -> t]; n = 1; ReplacePart[{a, b, c, d}, {{1}, {3}} :> n++]; ReplacePart[{a, b, c}, 4 -> t]"#,
      r#"{a, b, c}"#,
    );
  }
  #[test]
  fn replace_part_7() {
    assert_case(
      r#"ReplacePart[{a, b, c}, 1 -> t]; ReplacePart[{{a, b}, {c, d}}, {2, 1} -> t]; ReplacePart[{{a, b}, {c, d}}, {{2, 1} -> t, {1, 1} -> t}]; ReplacePart[{a, b, c}, {{1}, {2}} -> t]; n = 1; ReplacePart[{a, b, c, d}, {{1}, {3}} :> n++]; ReplacePart[{a, b, c}, 4 -> t]; ReplacePart[{a, b, c}, 0 -> Times]"#,
      r#"a*b*c"#,
    );
  }
  #[test]
  fn replace_part_8() {
    assert_case(
      r#"ReplacePart[{a, b, c}, 1 -> t]; ReplacePart[{{a, b}, {c, d}}, {2, 1} -> t]; ReplacePart[{{a, b}, {c, d}}, {{2, 1} -> t, {1, 1} -> t}]; ReplacePart[{a, b, c}, {{1}, {2}} -> t]; n = 1; ReplacePart[{a, b, c, d}, {{1}, {3}} :> n++]; ReplacePart[{a, b, c}, 4 -> t]; ReplacePart[{a, b, c}, 0 -> Times]; ReplacePart[{a, b, c}, -1 -> t]"#,
      r#"{a, b, t}"#,
    );
  }
  #[test]
  fn rest_1() {
    assert_case(r#"Rest[{a, b, c}]"#, r#"{b, c}"#);
  }
  #[test]
  fn rest_2() {
    assert_case(r#"Rest[{a, b, c}]; Rest[a + b + c]"#, r#"b + c"#);
  }
  #[test]
  fn select_2() {
    assert_case(r#"Select[Range[10], EvenQ]"#, r#"{2, 4, 6, 8, 10}"#);
  }
  #[test]
  fn select_3() {
    assert_case(
      r#"Select[Range[10], EvenQ]; Select[{-3, 0, 10, 3, a}, #>0&]"#,
      r#"{10, 3}"#,
    );
  }
  #[test]
  fn select_4() {
    assert_case(
      r#"Select[Range[10], EvenQ]; Select[{-3, 0, 10, 3, a}, #>0&]; Select[{-3, 0, 10, 3, a}, #>0&, 1]"#,
      r#"{10}"#,
    );
  }
  #[test]
  fn select_5() {
    assert_case(
      r#"Select[Range[10], EvenQ]; Select[{-3, 0, 10, 3, a}, #>0&]; Select[{-3, 0, 10, 3, a}, #>0&, 1]; Select[f[a, 2, 3], NumberQ]"#,
      r#"f[2, 3]"#,
    );
  }
  #[test]
  fn take_2() {
    assert_case(r#"Take[{a, b, c, d}, 3]"#, r#"{a, b, c}"#);
  }
  #[test]
  fn take_3() {
    assert_case(
      r#"Take[{a, b, c, d}, 3]; Take[{a, b, c, d}, -2]"#,
      r#"{c, d}"#,
    );
  }
  #[test]
  fn take_4() {
    assert_case(
      r#"Take[{a, b, c, d}, 3]; Take[{a, b, c, d}, -2]; Take[{a, b, c, d, e}, {2, -2}]"#,
      r#"{b, c, d}"#,
    );
  }
  #[test]
  fn take_5() {
    assert_case(
      r#"Take[{a, b, c, d}, 3]; Take[{a, b, c, d}, -2]; Take[{a, b, c, d, e}, {2, -2}]; A = {{a, b, c}, {d, e, f}}; Take[A, 2, 2]"#,
      r#"{{a, b}, {d, e}}"#,
    );
  }
  #[test]
  fn take_6() {
    assert_case(
      r#"Take[{a, b, c, d}, 3]; Take[{a, b, c, d}, -2]; Take[{a, b, c, d, e}, {2, -2}]; A = {{a, b, c}, {d, e, f}}; Take[A, 2, 2]; Take[A, All, {2}]"#,
      r#"{{b}, {e}}"#,
    );
  }
  #[test]
  fn take_7() {
    assert_case(
      r#"Take[{a, b, c, d}, 3]; Take[{a, b, c, d}, -2]; Take[{a, b, c, d, e}, {2, -2}]; A = {{a, b, c}, {d, e, f}}; Take[A, 2, 2]; Take[A, All, {2}]; Take[{a, b, c, d}, 0]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn table_13() {
    assert_case(
      r#"Pochhammer[1, 3]; Pochhammer[1, 3] == Pochhammer[2, 2]; Table[Pochhammer[0, n], {n, 0, -4, -1}]"#,
      r#"{1, -1, 1 / 2, -1 / 6, 1 / 24}"#,
    );
  }
  #[test]
  fn pochhammer_1() {
    assert_case(
      r#"Pochhammer[1, 3]; Pochhammer[1, 3] == Pochhammer[2, 2]; Table[Pochhammer[0, n], {n, 0, -4, -1}]; Pochhammer[1, 3.001]"#,
      r#"6.007542293946958"#,
    );
  }
  #[test]
  fn pochhammer_2() {
    assert_case(
      r#"Pochhammer[1, 3]; Pochhammer[1, 3] == Pochhammer[2, 2]; Table[Pochhammer[0, n], {n, 0, -4, -1}]; Pochhammer[1, 3.001]; Pochhammer[1, 3.001] == Pochhammer[2, 2.001]"#,
      r#"True"#,
    );
  }
  #[test]
  fn pochhammer_3() {
    assert_case(
      r#"Pochhammer[1, 3]; Pochhammer[1, 3] == Pochhammer[2, 2]; Table[Pochhammer[0, n], {n, 0, -4, -1}]; Pochhammer[1, 3.001]; Pochhammer[1, 3.001] == Pochhammer[2, 2.001]; Pochhammer[1.001, 3] == 1.001 2.001 3.001"#,
      r#"True"#,
    );
  }
  #[test]
  fn table_14() {
    assert_case(
      r#"Table[CatalanNumber[n], {n, 1, 5}]"#,
      r#"{1, 2, 5, 14, 42}"#,
    );
  }
  #[test]
  fn table_15() {
    assert_case(r#"Table[EulerE[k], {k, 1, 9, 2}]"#, r#"{0, 0, 0, 0, 0}"#);
  }
  #[test]
  fn table_16() {
    assert_case(
      r#"Table[EulerE[k], {k, 1, 9, 2}]; Table[EulerE[k], {k, 0, 8, 2}]"#,
      r#"{1, -1, 5, -61, 1385}"#,
    );
  }
  #[test]
  fn euler_e() {
    assert_case(
      r#"Table[EulerE[k], {k, 1, 9, 2}]; Table[EulerE[k], {k, 0, 8, 2}]; EulerE[5, z]"#,
      r#"-1/2 + (5*z^2)/2 - (5*z^4)/2 + z^5"#,
    );
  }
  #[test]
  fn table_17() {
    assert_case(r#"Table[LucasL[n], {n, 1, 5}]"#, r#"{1, 3, 4, 7, 11}"#);
  }
  #[test]
  fn table_18() {
    assert_case(
      r#"Table[PolygonalNumber[n], {n, 10}]"#,
      r#"{1, 3, 6, 10, 15, 21, 28, 36, 45, 55}"#,
    );
  }
  #[test]
  fn table_19() {
    assert_case(
      r#"Table[PolygonalNumber[n], {n, 10}]; Table[PolygonalNumber[n-1] + PolygonalNumber[n], {n, 10}]"#,
      r#"{1, 4, 9, 16, 25, 36, 49, 64, 81, 100}"#,
    );
  }
  #[test]
  fn table_20() {
    assert_case(
      r#"Table[PolygonalNumber[n], {n, 10}]; Table[PolygonalNumber[n-1] + PolygonalNumber[n], {n, 10}]; Table[PolygonalNumber[r, 10], {r, 3, 8}]"#,
      r#"{55, 100, 145, 190, 235, 280}"#,
    );
  }
  #[test]
  fn subsets_1() {
    assert_case(
      r#"Subsets[{a, b, c}]"#,
      r#"{{}, {a}, {b}, {c}, {a, b}, {a, c}, {b, c}, {a, b, c}}"#,
    );
  }
  #[test]
  fn subsets_2() {
    assert_case(
      r#"Subsets[{a, b, c}]; Subsets[{a, b, c, d}, 2]"#,
      r#"{{}, {a}, {b}, {c}, {d}, {a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}}"#,
    );
  }
  #[test]
  fn subsets_3() {
    assert_case(
      r#"Subsets[{a, b, c}]; Subsets[{a, b, c, d}, 2]; Subsets[{a, b, c, d}, {2}]"#,
      r#"{{a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}}"#,
    );
  }
  #[test]
  fn subsets_4() {
    assert_case(
      r#"Subsets[{a, b, c}]; Subsets[{a, b, c, d}, 2]; Subsets[{a, b, c, d}, {2}]; Subsets[{a, b, c, d, e}, {3}, 5]"#,
      r#"{{a, b, c}, {a, b, d}, {a, b, e}, {a, c, d}, {a, c, e}}"#,
    );
  }
  #[test]
  fn subsets_5() {
    assert_case(
      r#"Subsets[{a, b, c}]; Subsets[{a, b, c, d}, 2]; Subsets[{a, b, c, d}, {2}]; Subsets[{a, b, c, d, e}, {3}, 5]; Subsets[{a, b, c, d}, {0, 4, 2}]"#,
      r#"{{}, {a, b}, {a, c}, {a, d}, {b, c}, {b, d}, {c, d}, {a, b, c, d}}"#,
    );
  }
  #[test]
  fn subsets_6() {
    assert_case(
      r#"Subsets[{a, b, c}]; Subsets[{a, b, c, d}, 2]; Subsets[{a, b, c, d}, {2}]; Subsets[{a, b, c, d, e}, {3}, 5]; Subsets[{a, b, c, d}, {0, 4, 2}]; Subsets[Range[5], All, {25}]"#,
      r#"{{2, 4, 5}}"#,
    );
  }
  #[test]
  fn subsets_7() {
    assert_case(
      r#"Subsets[{a, b, c}]; Subsets[{a, b, c, d}, 2]; Subsets[{a, b, c, d}, {2}]; Subsets[{a, b, c, d, e}, {3}, 5]; Subsets[{a, b, c, d}, {0, 4, 2}]; Subsets[Range[5], All, {25}]; Subsets[{a, b, c, d}, All, {15, 1, -2}]"#,
      r#"{{b, c, d}, {a, b, d}, {c, d}, {b, c}, {a, c}, {d}, {b}, {}}"#,
    );
  }
  #[test]
  fn table_21() {
    assert_case(
      r#"BernoulliB[42]; Table[BernoulliB[k], {k, 0, 5}]"#,
      r#"{1, -1/2, 1/6, 0, -1/30, 0}"#,
    );
  }
  #[test]
  fn table_22() {
    assert_case(
      r#"BernoulliB[42]; Table[BernoulliB[k], {k, 0, 5}]; Table[BernoulliB[k, z], {k, 0, 3}]"#,
      r#"{1, -1/2 + z, 1/6 - z + z^2, z/2 - (3*z^2)/2 + z^3}"#,
    );
  }
  #[test]
  fn table_23() {
    assert_case(
      r#"Table[HarmonicNumber[n], {n, 8}]"#,
      r#"{1, 3 / 2, 11 / 6, 25 / 12, 137 / 60, 49 / 20, 363 / 140, 761 / 280}"#,
    );
  }
  #[test]
  fn harmonic_number() {
    assert_case(
      r#"Table[HarmonicNumber[n], {n, 8}]; HarmonicNumber[3.8]"#,
      r#"2.0380634056306492"#,
    );
  }
  #[test]
  fn table_24() {
    assert_case(
      r#"Table[CompositeQ[n], {n, 0, 10}]"#,
      r#"{False, False, False, False, True, False, True, False, True, True, True}"#,
    );
  }
  #[test]
  fn fixed_point_1() {
    assert_case(r#"FixedPoint[Cos, 1.0]"#, r#"0.7390851332151607"#);
  }
  #[test]
  fn fixed_point_2() {
    assert_case(r#"FixedPoint[Cos, 1.0]; FixedPoint[#+1 &, 1, 20]"#, r#"21"#);
  }
  #[test]
  fn fixed_point_list() {
    assert_case(
      r#"FixedPointList[Cos, 1.0, 4]"#,
      r#"{1., 0.5403023058681398, 0.8575532158463933, 0.6542897904977792, 0.7934803587425655}"#,
    );
  }
  #[test]
  fn newton() {
    assert_case(
      r#"FixedPointList[Cos, 1.0, 4]; newton[n_] := FixedPointList[.5(# + n/#) &, 1.]; newton[9]"#,
      r#"{1., 5., 3.4, 3.023529411764706, 3.00009155413138, 3.000000001396984, 3., 3.}"#,
    );
  }
  #[test]
  fn set_2() {
    assert_case(
      r#"FixedPointList[Cos, 1.0, 4]; newton[n_] := FixedPointList[.5(# + n/#) &, 1.]; newton[9]; collatz[1] := 1; collatz[x_ ? EvenQ] := x / 2; collatz[x_] := 3 x + 1; list = FixedPointList[collatz, 14]"#,
      r#"{14, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 1}"#,
    );
  }
  #[test]
  fn fold_1() {
    assert_case(r#"Fold[Plus, 5, {1, 1, 1}]"#, r#"8"#);
  }
  #[test]
  fn fold_2() {
    assert_case(
      r#"Fold[Plus, 5, {1, 1, 1}]; Fold[f, 5, {1, 2, 3}]"#,
      r#"f[f[f[5, 1], 2], 3]"#,
    );
  }
  #[test]
  fn fold_list_1() {
    assert_case(
      r#"FoldList[f, x, {1, 2, 3}]"#,
      r#"{x, f[x, 1], f[f[x, 1], 2], f[f[f[x, 1], 2], 3]}"#,
    );
  }
  #[test]
  fn fold_list_2() {
    assert_case(
      r#"FoldList[f, x, {1, 2, 3}]; FoldList[Times, {1, 2, 3}]"#,
      r#"{1, 2, 6}"#,
    );
  }
  #[test]
  fn nest_1() {
    assert_case(r#"Nest[f, x, 3]"#, r#"f[f[f[x]]]"#);
  }
  #[test]
  fn nest_2() {
    assert_case(
      r#"Nest[f, x, 3]; Nest[(1+#) ^ 2 &, x, 2]"#,
      r#"(1 + (1 + x) ^ 2) ^ 2"#,
    );
  }
  #[test]
  fn nest_list_2() {
    assert_case(r#"NestList[f, x, 3]"#, r#"{x, f[x], f[f[x]], f[f[f[x]]]}"#);
  }
  #[test]
  fn nest_list_3() {
    assert_case(
      r#"NestList[f, x, 3]; NestList[2 # &, 1, 8]"#,
      r#"{1, 2, 4, 8, 16, 32, 64, 128, 256}"#,
    );
  }
  #[test]
  fn nest_while_1() {
    assert_case(r#"NestWhile[#/2&, 10000, IntegerQ]"#, r#"625 / 2"#);
  }
  #[test]
  fn scan() {
    assert_case(r#"Scan[Print, {1, 2, 3}]"#, r#"Null"#);
  }
  #[test]
  fn member_q_6() {
    assert_case(r#"MemberQ[$Packages, "System`"]"#, r#"True"#);
  }
  #[test]
  fn any_true_3() {
    assert_case(r#"AnyTrue[{}, EvenQ]"#, r#"False"#);
  }
  #[test]
  fn all_true_3() {
    assert_case(r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]"#, r#"True"#);
  }
  #[test]
  fn equivalent_1() {
    assert_case(
      r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]; Equivalent[]"#,
      r#"True"#,
    );
  }
  #[test]
  fn equivalent_2() {
    assert_case(
      r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]; Equivalent[]; Equivalent[a]"#,
      r#"True"#,
    );
  }
  #[test]
  fn none_true_3() {
    assert_case(
      r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]; Equivalent[]; Equivalent[a]; NoneTrue[{}, EvenQ]"#,
      r#"True"#,
    );
  }
  #[test]
  fn xor_1() {
    assert_case(
      r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]; Equivalent[]; Equivalent[a]; NoneTrue[{}, EvenQ]; Xor[]"#,
      r#"False"#,
    );
  }
  #[test]
  fn xor_2() {
    assert_case(
      r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]; Equivalent[]; Equivalent[a]; NoneTrue[{}, EvenQ]; Xor[]; Xor[a]"#,
      r#"a"#,
    );
  }
  #[test]
  fn xor_3() {
    assert_case(
      r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]; Equivalent[]; Equivalent[a]; NoneTrue[{}, EvenQ]; Xor[]; Xor[a]; Xor[False]"#,
      r#"False"#,
    );
  }
  #[test]
  fn xor_4() {
    assert_case(
      r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]; Equivalent[]; Equivalent[a]; NoneTrue[{}, EvenQ]; Xor[]; Xor[a]; Xor[False]; Xor[True]"#,
      r#"True"#,
    );
  }
  #[test]
  fn xor_5() {
    assert_case(
      r#"AnyTrue[{}, EvenQ]; AllTrue[{}, EvenQ]; Equivalent[]; Equivalent[a]; NoneTrue[{}, EvenQ]; Xor[]; Xor[a]; Xor[False]; Xor[True]; Xor[a, b]"#,
      r#"Xor[a, b]"#,
    );
  }
  #[test]
  fn subset_q_5() {
    assert_case(r#"SubsetQ[{1, 2, 3}, {0, 1}]"#, r#"False"#);
  }
  #[test]
  fn subset_q_6() {
    assert_case(
      r#"SubsetQ[{1, 2, 3}, {0, 1}]; SubsetQ[{1, 2, 3}, {1, 2, 3, 4}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn nest_while_2() {
    assert_case(r#"NestWhile[#/2&, 10000, IntegerQ]"#, r#"625/2"#);
  }
  #[test]
  fn sort_2() {
    assert_case(r#"Sort[{x_, y_}, PatternsOrderedQ]"#, r#"{x_, y_}"#);
  }
  #[test]
  fn length_3() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]"#,
      r#"96"#,
    );
  }
  #[test]
  fn length_4() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]"#,
      r#"672"#,
    );
  }
  #[test]
  fn fractional_part_1() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]"#,
      r#"FractionalPart[b]"#,
    );
  }
  #[test]
  fn fractional_part_2() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]"#,
      r#"{-0.3999999999999999, -0.5, 0.}"#,
    );
  }
  #[test]
  fn fractional_part_3() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]"#,
      r#"7 / 16"#,
    );
  }
  #[test]
  fn fractional_part_4() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]"#,
      r#"2 / 5 - I / 5"#,
    );
  }
  #[test]
  fn fractional_part_5() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]; FractionalPart[Pi^20]"#,
      r#"-8769956796 + Pi ^ 20"#,
    );
  }
  #[test]
  fn mantissa_exponent_1() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]; FractionalPart[Pi^20]; MantissaExponent[E, Pi]"#,
      r#"{E / Pi, 1}"#,
    );
  }
  #[test]
  fn mantissa_exponent_2() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]; FractionalPart[Pi^20]; MantissaExponent[E, Pi]; MantissaExponent[Pi, Pi]"#,
      r#"{Pi^(-1), 2}"#,
    );
  }
  #[test]
  fn mantissa_exponent_3() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]; FractionalPart[Pi^20]; MantissaExponent[E, Pi]; MantissaExponent[Pi, Pi]; MantissaExponent[5/2 + 3, Pi]"#,
      r#"{11/(2*Pi^2), 2}"#,
    );
  }
  #[test]
  fn mantissa_exponent_4() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]; FractionalPart[Pi^20]; MantissaExponent[E, Pi]; MantissaExponent[Pi, Pi]; MantissaExponent[5/2 + 3, Pi]; MantissaExponent[b]"#,
      r#"MantissaExponent[b]"#,
    );
  }
  #[test]
  fn mantissa_exponent_5() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]; FractionalPart[Pi^20]; MantissaExponent[E, Pi]; MantissaExponent[Pi, Pi]; MantissaExponent[5/2 + 3, Pi]; MantissaExponent[b]; MantissaExponent[17, E]"#,
      r#"{17 / E ^ 3, 3}"#,
    );
  }
  #[test]
  fn mantissa_exponent_6() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]; FractionalPart[Pi^20]; MantissaExponent[E, Pi]; MantissaExponent[Pi, Pi]; MantissaExponent[5/2 + 3, Pi]; MantissaExponent[b]; MantissaExponent[17, E]; MantissaExponent[17., E]"#,
      r#"{0.8463801622536871, 3}"#,
    );
  }
  #[test]
  fn mantissa_exponent_7() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]; Length[Divisors[1000*369]]; Length[Divisors[305*176*369*100]]; FractionalPart[b]; FractionalPart[{-2.4, -2.5, -3.0}]; FractionalPart[14/32]; FractionalPart[4/(1 + 3 I)]; FractionalPart[Pi^20]; MantissaExponent[E, Pi]; MantissaExponent[Pi, Pi]; MantissaExponent[5/2 + 3, Pi]; MantissaExponent[b]; MantissaExponent[17, E]; MantissaExponent[17., E]; MantissaExponent[Exp[Pi], 2]"#,
      r#"{E ^ Pi / 32, 5}"#,
    );
  }
  #[test]
  fn table_25() {
    assert_case(r#"Table[x, {x,0,1/3}]"#, r#"{0}"#);
  }
  #[test]
  fn table_26() {
    assert_case(
      r#"Table[x, {x,0,1/3}]; Table[x, {x, -0.2, 3.9}]"#,
      r#"{-0.2, 0.8, 1.8, 2.8, 3.8}"#,
    );
  }
  #[test]
  fn table_27() {
    assert_case(r#"Table[x, {x,0,1/3}]"#, r#"{0}"#);
  }
  #[test]
  fn table_28() {
    assert_case(
      r#"Table[x, {x,0,1/3}]; Table[x, {x, -0.2, 3.9}]"#,
      r#"{-0.2, 0.8, 1.8, 2.8, 3.8}"#,
    );
  }
  #[test]
  fn delete_8() {
    // `Delete[{}, 0]` removes the `List` head of an empty list, leaving
    // `Sequence[]`. At top level wolframscript prints nothing for an
    // empty sequence — the mathics expectation `InputForm` was a bug.
    assert_case(r#"Delete[{}, 0]"#, "");
  }
  #[test]
  fn subsets_8() {
    assert_case(r#"Binomial[-10, -3.5]; Subsets[{}]"#, r#"{{}}"#);
  }
  #[test]
  fn fixed_point_3() {
    assert_case(r#"FixedPoint[f, x, 0]"#, r#"x"#);
  }
}
