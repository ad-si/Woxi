// Long-tail argument forms found missing by the reference-page parameter
// sweep (round 5, third batch). Expected values are the ones the Wolfram
// reference pages show, or follow from the definitions.

use super::*;

fn check(code: &str, expected: &str) {
  assert_eq!(interpret(code).unwrap(), expected, "for {code}");
}

mod folds {
  use super::*;

  #[test]
  fn fold_while_operator_form() {
    check("FoldWhile[Subtract, Positive][12, {1, 2, 3, 4, 5, 6}]", "-3");
  }

  #[test]
  fn fold_pair() {
    check(
      "Catch[FoldPair[{If[#1 > 1000, Throw[#1]], #1 #2}&, 1, Range[10]]]",
      "5040",
    );
    check("FoldPair[{#1, #1 + #2}&, 0, {1, 2, 3}]", "3");
  }

  #[test]
  fn reverse_applied_negative() {
    check("ReverseApplied[f, -2][a, b, c]", "f[a, c, b]");
  }
}

mod lists {
  use super::*;

  #[test]
  fn lexicographic_sort_with_order() {
    check(
      "LexicographicSort[{{Infinity, 0.}, {0, 0., 1.}, \
       {-Infinity, 2., 2., 3.}}, NumericalOrder]",
      "{{-Infinity, 2., 2., 3.}, {0, 0., 1.}, {Infinity, 0.}}",
    );
    check("NumericalOrder[Infinity, 3]", "-1");
  }

  #[test]
  fn center_array_forms() {
    check("CenterArray[{1, 2, 3}, 5]", "{0, 1, 2, 3, 0}");
    check(
      "CenterArray[{1, 2, 3}, {5, 5}, 0, SparseArray]",
      "SparseArray[Automatic, {5, 5}, 0, \
       {1, {{0, 0, 0, 3, 3, 3}, {{2}, {3}, {4}}}, {1, 2, 3}}]",
    );
  }

  #[test]
  fn general_heads() {
    check("DeleteAdjacentDuplicates[f[a, a, b, b, a]]", "f[a, b, a]");
    check("SymmetricDifference[f[a, b, c], f[b, d]]", "f[a, c, d]");
  }

  #[test]
  fn position_extremes() {
    check("PositionLargest[<|a -> 1, b -> 3, c -> 3|>]", "{2, 3}");
    check("PositionSmallest[{3, -Infinity, 2}]", "{2}");
  }

  #[test]
  fn sequence_split_alternatives() {
    check(
      "SequenceSplit[{1, 2, 3, 4, 5, 6}, {2, 3} | {5}]",
      "{{1}, {4}, {6}}",
    );
  }

  #[test]
  fn permutation_group_support() {
    check(
      "PermutationSupport[PermutationGroup[{Cycles[{{1, 2}}], \
       Cycles[{{4, 5}}]}]]",
      "{1, 2, 4, 5}",
    );
    check(
      "PermutationMax[PermutationGroup[{Cycles[{{1, 2}}], \
       Cycles[{{4, 5}}]}]]",
      "5",
    );
    check("PermutationMax[PermutationGroup[{}]]", "0");
    check("PermutationMin[PermutationGroup[{}]]", "Infinity");
  }
}

mod strings {
  use super::*;

  #[test]
  fn string_extract_forms() {
    check("StringExtract[\"a b c\\nd e f\", 2]", "b");
    check("StringExtract[\"a,b;c\", \",\" | \";\" -> All]", "{a, b, c}");
  }
}

mod math {
  use super::*;

  #[test]
  fn liouville_lambda_gaussian() {
    check("LiouvilleLambda[1 + I]", "-1");
    // 3 + 4 I = -I (2 + I)^2
    check("LiouvilleLambda[3 + 4 I]", "1");
  }

  #[test]
  fn dirac_comb_even() {
    check("DiracComb[-x]", "DiracComb[x]");
  }

  #[test]
  fn bit_flip_negative_position() {
    check("BitFlip[5, -1]", "1");
    // A position left of the highest bit leaves n unchanged.
    check("BitFlip[5, -5]", "5");
  }

  #[test]
  fn number_digit_listable() {
    check("NumberDigit[{123, 456}, 1]", "{2, 5}");
  }

  #[test]
  fn number_decompose_symbolic() {
    check("NumberDecompose[Pi, {1, 1/10}]", "{3, 10*(-3 + Pi)}");
  }

  #[test]
  fn discrete_shift_step() {
    check("DiscreteShift[f[x], {x, 2, h}]", "f[2*h + x]");
  }

  #[test]
  fn boolean_convert_xnor() {
    check("BooleanConvert[Xnor[a, b]]", "(a && b) || ( !a &&  !b)");
  }

  #[test]
  fn threaded_broadcast() {
    check(
      "Threaded[{-1, 1}] + Threaded[{{a, b}, {c, d}}]",
      "Threaded[{{-1 + a, 1 + b}, {-1 + c, 1 + d}}]",
    );
  }
}

mod linear_algebra {
  use super::*;

  #[test]
  fn triangular_matrices() {
    check(
      "LowerTriangularMatrix[{{2, 0, 0}, {1, -5, 0}, {-3, 2, 4}}]",
      "LowerTriangularMatrix[StructuredArray`StructuredData[{3, 3}, \
       {{2, 0, 0}, {1, -5, 0}, {-3, 2, 4}}]]",
    );
    check(
      "UpperTriangularMatrix[{{1, 2}, {0, 3}}, TargetStructure -> \"Dense\"]",
      "{{1, 2}, {0, 3}}",
    );
  }

  #[test]
  fn fourier_dst_complex() {
    check(
      "FourierDST[{1, 2I, 3, 4I}]",
      "{1.5771610149494748 + 1.6892463972414666*I, \
       -0.7071067811865475 - 0.7071067811865478*I, \
       -0.11208538229199216 + 1.4650756326574839*I, 2. - 3.*I}",
    );
  }

  #[test]
  fn linear_programming_empty_constraints() {
    check("LinearProgramming[{2, 3}, {{}}, {}, {-1, -2}]", "{-1, -2}");
  }

  #[test]
  fn design_matrix_one_dimensional() {
    check(
      "DesignMatrix[{1, 2, 3}, {1, x}, x]",
      "{{1, 1}, {1, 2}, {1, 3}}",
    );
    check(
      "DesignMatrix[{1.5, 2., 3.}, {1, x}, x]",
      "{{1., 1.}, {1., 2.}, {1., 3.}}",
    );
  }
}

mod geometry_and_graphs {
  use super::*;

  #[test]
  fn angle_bisector_exterior() {
    check(
      "AngleBisector[{{1, 0}, {0, 0}, {0, 1}}, \"Exterior\"]",
      "InfiniteLine[{0, 0}, {-1, 1}]",
    );
  }

  #[test]
  fn graph_measures_on_rule_lists() {
    check("EdgeConnectivity[{1 <-> 2, 2 <-> 3, 3 <-> 1}]", "2");
    check("GlobalClusteringCoefficient[{1 <-> 2, 2 <-> 3, 3 <-> 1}]", "1");
  }
}
