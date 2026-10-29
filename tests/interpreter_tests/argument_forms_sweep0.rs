// Argument forms and options found missing by the reference-page parameter
// sweep (second batch).

use super::*;

mod unique_attributes {
  use super::*;

  #[test]
  fn new_symbol_gets_the_attributes() {
    clear_state();
    assert_eq!(
      interpret("s = Unique[x, {Temporary}]; Attributes[Evaluate[s]]").unwrap(),
      "{Temporary}"
    );
    assert_eq!(
      interpret("Attributes /@ Unique[{x, y}, Listable]").unwrap(),
      "{{Listable}, {Listable}}"
    );
  }

  #[test]
  fn evaluate_inside_attributes() {
    assert_eq!(
      interpret("s = Plus; Attributes[Evaluate[s]] === Attributes[Plus]")
        .unwrap(),
      "True"
    );
  }
}

mod triangular_matrix_q_band {
  use super::*;

  const M: &str = "{{1, 2, 3}, {4, 5, 6}, {0, 7, 8}}";

  #[test]
  fn upper() {
    assert_eq!(
      interpret(&format!("UpperTriangularMatrixQ[{M}, -1]")).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(&format!("UpperTriangularMatrixQ[{M}, 0]")).unwrap(),
      "False"
    );
    assert_eq!(
      interpret("UpperTriangularMatrixQ[{{0, 1, 2}, {0, 0, 3}}, 1]").unwrap(),
      "True"
    );
  }

  #[test]
  fn lower() {
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{{1, 2, 0}, {4, 5, 6}, {0, 7, 8}}, 1]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{{1, 2, 3}, {4, 5, 6}}, 1]").unwrap(),
      "False"
    );
  }

  #[test]
  fn tolerance() {
    assert_eq!(
      interpret(
        "UpperTriangularMatrixQ[{{1., 2.}, {1.*^-14, 1.}}, Tolerance -> 10^-12]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "DiagonalMatrixQ[{{1., 1.*^-14}, {0, 1.}}, Tolerance -> 10^-12]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "DiagonalMatrixQ[{{1., 1.*^-10}, {0, 1.}}, Tolerance -> 10^-12]"
      )
      .unwrap(),
      "False"
    );
  }
}

mod operator_and_extra_forms {
  use super::*;

  #[test]
  fn count_distinct_by_operator() {
    assert_eq!(
      interpret("CountDistinctBy[First] @ {{a, x}, {a, y}, {b, z}}").unwrap(),
      "2"
    );
  }

  #[test]
  fn upsample_offset() {
    assert_eq!(
      interpret("Upsample[{1, 2, 3}, 3, 2]").unwrap(),
      "{0, 1, 0, 0, 2, 0, 0, 3, 0}"
    );
    assert_eq!(interpret("Upsample[{a, b}, 2, 1]").unwrap(), "{a, 0, b, 0}");
  }

  #[test]
  fn position_largest_with_ordering_function() {
    assert_eq!(
      interpret("PositionLargest[{3, 1, 4, 1, 5}, 2, Order[#2, #1] &]")
        .unwrap(),
      "{{4, 2}, {1}}"
    );
    assert_eq!(
      interpret("PositionLargest[{3, 1, 4, 1, 5}, 2, Order]").unwrap(),
      interpret("PositionLargest[{3, 1, 4, 1, 5}, 2]").unwrap()
    );
    assert_eq!(
      interpret("PositionSmallest[{3, 1, 4, 1, 5}, 2, Order]").unwrap(),
      "{{2, 4}, {1}}"
    );
  }

  #[test]
  fn permutation_matrix_size() {
    assert_eq!(
      interpret("PermutationMatrix[Cycles[{{1, 3}}], 4]").unwrap(),
      "{{0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}}"
    );
    assert_eq!(
      interpret("PermutationMatrix[{2, 1}, 3]").unwrap(),
      "{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn disk_matrix_in_width() {
    assert_eq!(
      interpret("DiskMatrix[8, {23, 23}] == ArrayPad[DiskMatrix[8], 3]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("DiskMatrix[1, 5]").unwrap(),
      "{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}"
    );
    assert_eq!(
      interpret("DiskMatrix[2, 3]").unwrap(),
      "{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}"
    );
  }

  #[test]
  fn distribution_covariance_entry() {
    assert_eq!(
      interpret("Correlation[BinormalDistribution[ρ], 2, 1]").unwrap(),
      "ρ"
    );
    assert_eq!(
      interpret("Covariance[BinormalDistribution[ρ], 2, 1]").unwrap(),
      "ρ"
    );
    assert_eq!(
      interpret("Covariance[BinormalDistribution[{1, 2}, {3, 4}, ρ], 1, 2]")
        .unwrap(),
      "12*ρ"
    );
  }
}

mod option_forms {
  use super::*;

  #[test]
  fn symmetric_difference_same_test() {
    assert_eq!(
      interpret(
        "SymmetricDifference[{1, 2, 3}, {2., 3., 4.}, SameTest -> Equal]"
      )
      .unwrap(),
      "{1, 4.}"
    );
  }

  #[test]
  fn counts_ignore_case() {
    assert_eq!(
      interpret("LetterCounts[\"Alabama\", IgnoreCase -> False]").unwrap(),
      interpret("LetterCounts[\"Alabama\"]").unwrap()
    );
    assert_eq!(
      interpret("LetterCounts[\"Alabama\", IgnoreCase -> True][\"a\"]")
        .unwrap(),
      "4"
    );
    assert_eq!(
      interpret("CharacterCounts[\"AaB\", IgnoreCase -> True][\"a\"]").unwrap(),
      "2"
    );
  }

  #[test]
  fn matrix_q_same_test() {
    let t = "SameTest -> (Simplify[#1 - #2, x > 0] == 0 &)";
    assert_eq!(
      interpret(&format!(
        "SymmetricMatrixQ[{{{{1, Sqrt[x^2]}}, {{x, 1}}}}, {t}]"
      ))
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("SymmetricMatrixQ[{{1, Sqrt[x^2]}, {x, 1}}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret(&format!(
        "AntihermitianMatrixQ[{{{{0, Sqrt[x^2]}}, {{-x, 0}}}}, {t}]"
      ))
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "OrthogonalMatrixQ[{{Cos[x], Sin[x]}, {-Sin[x], Cos[x]}}, SameTest -> (Simplify[#1 - #2] == 0 &)]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn replace_part_heads() {
    assert_eq!(
      interpret("ReplacePart[f[x, y], _ -> g, Heads -> True]").unwrap(),
      "g[g, g]"
    );
    assert_eq!(
      interpret("ReplacePart[f[x, y], _ -> g, Heads -> False]").unwrap(),
      "f[g, g]"
    );
  }

  #[test]
  fn target_structure() {
    assert_eq!(
      interpret(
        "UpperTriangularize[{{1, 2}, {3, 4}}, TargetStructure -> \"Dense\"]"
      )
      .unwrap(),
      "{{1, 2}, {0, 4}}"
    );
    assert_eq!(
      interpret(
        "Normal[CompanionMatrix[{1, 2, 3}, TargetStructure -> \"Sparse\"]] == CompanionMatrix[{1, 2, 3}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "Head[LowerTriangularize[{{1, 2}, {3, 4}}, TargetStructure -> \"Sparse\"]]"
      )
      .unwrap(),
      "SparseArray"
    );
  }

  #[test]
  fn variables_modulus() {
    assert_eq!(
      interpret("Variables[x + 2 y + 3 z, Modulus -> 2]").unwrap(),
      "{x, z}"
    );
  }

  #[test]
  fn numerator_denominator_trig() {
    assert_eq!(
      interpret(
        "NumeratorDenominator[{Sin[x], Cos[x], Tan[x], Csc[x], Sec[x], Cot[x]}, Trig -> True]"
      )
      .unwrap(),
      "{{Sin[x], Cos[x], Sin[x], 1, 1, Cos[x]}, {1, 1, Cos[x], Sin[x], Cos[x], Sin[x]}}"
    );
  }

  #[test]
  fn dilation_erosion_padding() {
    assert_eq!(
      interpret("Dilation[{1, 2, 3, 4, 5}, 1, Padding -> 100]").unwrap(),
      "{100, 3, 4, 5, 100}"
    );
    assert_eq!(
      interpret("Erosion[{1, 2, 3, 4, 5}, 1, Padding -> 0]").unwrap(),
      "{0, 1, 2, 3, 0}"
    );
  }

  #[test]
  fn prime_nu_gaussian() {
    assert_eq!(
      interpret("PrimeNu[30, GaussianIntegers -> True]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("PrimeNu[30, GaussianIntegers -> False]").unwrap(),
      "3"
    );
  }
}

mod recurrence_table_multi {
  use super::*;

  #[test]
  fn stirling_numbers() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{s1[n, k] == s1[n - 1, k - 1] - (n - 1) s1[n - 1, k], s1[0, k] == KroneckerDelta[k]}, s1, {n, 0, 4}, {k, 0, 4}]"
      )
      .unwrap(),
      "{{1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, -1, 1, 0, 0}, {0, 2, -3, 1, 0}, {0, -6, 11, -6, 1}}"
    );
  }

  #[test]
  fn pascal_triangle() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{b[n, k] == b[n - 1, k] + b[n - 1, k - 1], b[0, k] == KroneckerDelta[k]}, b, {n, 0, 4}, {k, 0, 3}]"
      )
      .unwrap(),
      "{{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}, {1, 4, 6, 4}}"
    );
  }

  #[test]
  fn dependent_variables_option() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{x[n + 2] == y[n], y[n + 1] == -x[n], x[0] == 1, x[1] == 0, y[0] == 0}, {y, x}, {n, 0, 3}, DependentVariables -> {x, y}]"
      )
      .unwrap(),
      "{{0, 1}, {-1, 0}, {0, 0}, {0, -1}}"
    );
  }
}

mod series_coefficient_multi {
  use super::*;

  #[test]
  fn integer_orders() {
    assert_eq!(
      interpret("SeriesCoefficient[Exp[x + y], {x, 0, 2}, {y, 0, 3}]").unwrap(),
      "1/12"
    );
    assert_eq!(
      interpret("SeriesCoefficient[Exp[3 x + y^2], {x, 0, 1}, {y, 0, 2}]")
        .unwrap(),
      "3"
    );
  }
}
