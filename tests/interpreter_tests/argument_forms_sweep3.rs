// Argument forms and options found missing by the reference-page parameter
// sweep (third batch).

use super::*;

mod subset_replace_count {
  use super::*;

  #[test]
  fn limits_the_number_of_replacements() {
    assert_eq!(
      interpret("SubsetReplace[{a, b, a, b, a, b, a, b}, {a, a} -> x, 1]")
        .unwrap(),
      "{x, b, b, a, b, a, b}"
    );
    assert_eq!(
      interpret("SubsetReplace[{a, b, a, b, a, b, a, b}, {a, a} -> x, 0]")
        .unwrap(),
      "{a, b, a, b, a, b, a, b}"
    );
    assert_eq!(
      interpret(
        "SubsetReplace[{a, b, a, b, a, b, a, b}, {a, a} -> x, Infinity]"
      )
      .unwrap(),
      "{x, b, b, x, b, b}"
    );
  }
}

mod take_list_levels {
  use super::*;

  #[test]
  fn splits_successive_levels() {
    assert_eq!(
      interpret("TakeList[Partition[Range[25], 5], {2, 3}, {3, 2}]").unwrap(),
      "{{{{1, 2, 3}, {4, 5}}, {{6, 7, 8}, {9, 10}}}, \
       {{{11, 12, 13}, {14, 15}}, {{16, 17, 18}, {19, 20}}, \
       {{21, 22, 23}, {24, 25}}}}"
    );
    assert_eq!(
      interpret("TakeList[{{1, 2}, {3, 4}}, {1, 1}, {1, 1}]").unwrap(),
      "{{{{1}, {2}}}, {{{3}, {4}}}}"
    );
  }
}

mod digit_sum_powers {
  use super::*;

  #[test]
  fn sums_kth_powers_of_digits() {
    assert_eq!(interpret("DigitSum[6345354, 10, 4]").unwrap(), "3220");
    assert_eq!(interpret("DigitSum[153, 10, 3]").unwrap(), "153");
    assert_eq!(interpret("DigitSum[10, 2, 2]").unwrap(), "2");
    assert_eq!(interpret("DigitSum[1234, 10, 1]").unwrap(), "10");
  }
}

mod elliptic_theta_prime_constant {
  use super::*;

  #[test]
  fn two_argument_form_is_z_zero() {
    assert_eq!(
      interpret("EllipticThetaPrime[1, 0.3] == EllipticThetaPrime[1, 0, 0.3]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("EllipticThetaPrime[1, q]").unwrap(),
      "EllipticThetaPrime[1, q]"
    );
  }
}

mod heads_option {
  use super::*;

  #[test]
  fn scan_with_heads() {
    let r =
      interpret_with_stdout("Scan[Print, {a, b}, Heads -> True]").unwrap();
    assert_eq!(r.stdout, "List\na\nb\n");
    let r = interpret_with_stdout("Scan[Print, f[g[x]], {2}, Heads -> True]")
      .unwrap();
    assert_eq!(r.stdout, "g\nx\n");
  }

  #[test]
  fn map_all_with_heads() {
    assert_eq!(
      interpret("MapAll[f, {a, b}, Heads -> True]").unwrap(),
      "f[f[List][f[a], f[b]]]"
    );
    assert_eq!(
      interpret("MapAll[f, {a, b}, Heads -> False]").unwrap(),
      "f[{f[a], f[b]}]"
    );
  }

  #[test]
  fn map_level_zero_with_heads() {
    assert_eq!(
      interpret("Map[f, {a, b}, Heads -> False]").unwrap(),
      "{f[a], f[b]}"
    );
    assert_eq!(
      interpret("Map[f, {a, b}, {0}, Heads -> True]").unwrap(),
      "f[{a, b}]"
    );
    assert_eq!(
      interpret("Map[f, {a, b}, {0, Infinity}, Heads -> True]").unwrap(),
      "f[f[List][f[a], f[b]]]"
    );
  }

  #[test]
  fn depth_with_heads() {
    assert_eq!(
      interpret("Depth[{f[][x], g[y][x], h[g[y]][x]}, Heads -> True]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("Depth[{f[][x], g[y][x], h[g[y]][x]}]").unwrap(),
      "3"
    );
    assert_eq!(interpret("Depth[f[x], Heads -> True]").unwrap(), "2");
  }

  #[test]
  fn comap_apply_with_heads() {
    assert_eq!(
      interpret("ComapApply[f[g], {x}, Heads -> True]").unwrap(),
      "f[x][g[x]]"
    );
    assert_eq!(
      interpret("ComapApply[f[g, h], {x, y}, Heads -> False]").unwrap(),
      "f[g[x, y], h[x, y]]"
    );
  }
}

mod linear_algebra_options {
  use super::*;

  #[test]
  fn rotation_matrix_target_structure() {
    assert_eq!(
      interpret("RotationMatrix[Pi/2, TargetStructure -> \"Dense\"]").unwrap(),
      "{{0, -1}, {1, 0}}"
    );
    assert_eq!(
      interpret(
        "Normal[RotationMatrix[Pi/2, TargetStructure -> \"Sparse\"]] == \
         RotationMatrix[Pi/2]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Head[RotationMatrix[Pi/2, TargetStructure -> \"Sparse\"]]")
        .unwrap(),
      "SparseArray"
    );
  }

  #[test]
  fn inverse_method_and_zero_test() {
    assert_eq!(
      interpret("Inverse[{{1, 2}, {3, 4}}, Method -> \"CofactorExpansion\"]")
        .unwrap(),
      "{{-2, 1}, {3/2, -1/2}}"
    );
    assert_eq!(
      interpret(
        "Inverse[{{1, 2}, {3, 4}}, ZeroTest -> (N[#, $MachinePrecision] == 0 &)]"
      )
      .unwrap(),
      "{{-2, 1}, {3/2, -1/2}}"
    );
  }

  #[test]
  fn hermitian_same_test() {
    assert_eq!(
      interpret(
        "HermitianMatrixQ[{{1, Sqrt[x^2]}, {x, 1}}, \
         SameTest -> (Simplify[#1 - #2, x > 0] == 0 &)]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HermitianMatrixQ[{{1, 2}, {3, 1}}, SameTest -> Equal]")
        .unwrap(),
      "False"
    );
  }

  #[test]
  fn tolerance_on_exact_matrices() {
    assert_eq!(
      interpret("HermitianMatrixQ[{{1, I}, {-I, 1}}, Tolerance -> 10^-12]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "NegativeSemidefiniteMatrixQ[{{-1, 0}, {0, -2}}, Tolerance -> 10^-12]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{1, 0}, {0, -2}}, Tolerance -> 10^-12]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn savitzky_golay_exact() {
    assert_eq!(
      interpret("SavitzkyGolayMatrix[{2}, 2, WorkingPrecision -> Infinity]")
        .unwrap(),
      "{-3/35, 12/35, 17/35, 12/35, -3/35}"
    );
    assert_eq!(
      interpret("SavitzkyGolayMatrix[1, 1, WorkingPrecision -> Infinity]")
        .unwrap(),
      "{{1/9, 1/9, 1/9}, {1/9, 1/9, 1/9}, {1/9, 1/9, 1/9}}"
    );
  }
}

mod random_variate_precision {
  use super::*;

  #[test]
  fn machine_precision_is_the_default() {
    assert_eq!(
      interpret(
        "SeedRandom[1234]; a = RandomVariate[ExponentialDistribution[2], 5]; \
         SeedRandom[1234]; b = RandomVariate[ExponentialDistribution[2], 5, \
         WorkingPrecision -> MachinePrecision]; a == b"
      )
      .unwrap(),
      "True"
    );
  }
}

mod dt_constants {
  use super::*;

  #[test]
  fn declared_constants_have_zero_derivative() {
    assert_eq!(
      interpret("Dt[a x^2 + b, x, Constants -> {a}]").unwrap(),
      "2*a*x + Dt[b, x, Constants -> {a}]"
    );
    assert_eq!(
      interpret("Dt[a x + b, Constants -> {a, b}]").unwrap(),
      "a*Dt[x, Constants -> {a, b}]"
    );
  }
}

mod legendre_q_type {
  use super::*;

  #[test]
  fn type_one_is_the_default() {
    assert_eq!(
      interpret("LegendreQ[2, 0.5, 1, 0.3] == LegendreQ[2, 0.5, 0.3]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("LegendreQ[n, m, 3, z]").unwrap(),
      "LegendreQ[n, m, 3, z]"
    );
  }
}

mod rational_parts_options {
  use super::*;

  #[test]
  fn expand_numerator_and_denominator_modulus() {
    assert_eq!(
      interpret("ExpandNumerator[(x + y)^3/z, Modulus -> 3]").unwrap(),
      "(x^3 + y^3)/z"
    );
    assert_eq!(
      interpret("ExpandDenominator[z/(x + y)^3, Modulus -> 3]").unwrap(),
      "z/(x^3 + y^3)"
    );
  }

  #[test]
  fn expand_numerator_and_denominator_trig() {
    assert_eq!(
      interpret("ExpandNumerator[Sin[x + y]/(a - b), Trig -> True]").unwrap(),
      "(Cos[y]*Sin[x] + Cos[x]*Sin[y])/(a - b)"
    );
    assert_eq!(
      interpret("ExpandDenominator[(a - b)/(Sin[x + y] + 1), Trig -> True]")
        .unwrap(),
      "(a - b)/(1 + Cos[y]*Sin[x] + Cos[x]*Sin[y])"
    );
  }

  #[test]
  fn cancel_modulus() {
    assert_eq!(
      interpret("Cancel[(x^2 + 1)/(x + 1), Modulus -> 2]").unwrap(),
      "1 + x"
    );
  }
}

mod same_test_repeats {
  use super::*;

  #[test]
  fn find_repeat_same_test() {
    assert_eq!(
      interpret(
        "FindRepeat[{1, 2, -3, -1, 2, 3, -1, -2}, SameTest -> (Abs[#1] == Abs[#2] &)]"
      )
      .unwrap(),
      "{1, 2, -3}"
    );
    assert_eq!(
      interpret("FindRepeat[{1, 2, 3, 1, 2, 3}]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn find_transient_repeat_same_test() {
    assert_eq!(
      interpret(
        "FindTransientRepeat[{1, 2, -3, 4, 3, -4, -3, 4}, 3, \
         SameTest -> (Abs[#1] == Abs[#2] &)]"
      )
      .unwrap(),
      "{{1, 2}, {-3, 4}}"
    );
  }
}

mod alignment_options {
  use super::*;

  #[test]
  fn gap_penalty_and_ignore_case() {
    assert_eq!(
      interpret(
        r#"SmithWatermanSimilarity["ABCDEF", "ABCxDEF", GapPenalty -> 1]"#
      )
      .unwrap(),
      "5."
    );
    assert_eq!(
      interpret(
        r#"SmithWatermanSimilarity["ABCDEF", "ABCxDEF", GapPenalty -> 3]"#
      )
      .unwrap(),
      "3."
    );
    assert_eq!(
      interpret(
        r#"SmithWatermanSimilarity["ABCDEF", "abcdef", IgnoreCase -> True]"#
      )
      .unwrap(),
      "6."
    );
    assert_eq!(
      interpret(
        r#"NeedlemanWunschSimilarity["ABCDEF", "ABCxDEF", GapPenalty -> 1]"#
      )
      .unwrap(),
      "5."
    );
    assert_eq!(
      interpret(
        r#"NeedlemanWunschSimilarity["ABCDEF", "abcdef", IgnoreCase -> True]"#
      )
      .unwrap(),
      "6."
    );
  }
}

mod composite_q_gaussian {
  use super::*;

  #[test]
  fn gaussian_integers_option() {
    assert_eq!(
      interpret("CompositeQ[5, GaussianIntegers -> True]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("CompositeQ[3, GaussianIntegers -> True]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("CompositeQ[5, GaussianIntegers -> False]").unwrap(),
      "False"
    );
  }
}

mod alphabetic_order_forms {
  use super::*;

  #[test]
  fn operator_and_language_forms() {
    assert_eq!(interpret(r#"AlphabeticOrder[]["A", "c"]"#).unwrap(), "1");
    assert_eq!(
      interpret(r#"AlphabeticOrder["Spanish"]["ñ", "n"]"#).unwrap(),
      "-1"
    );
    assert_eq!(
      interpret(r#"AlphabeticOrder["ñ", "o", "Spanish"]"#).unwrap(),
      "1"
    );
    assert_eq!(
      interpret(r#"AlphabeticOrder["ñ", "n", "Spanish"]"#).unwrap(),
      "-1"
    );
  }
}

mod remove_diacritics_language {
  use super::*;

  #[test]
  fn keeps_letters_of_the_alphabet() {
    assert_eq!(
      interpret(r#"RemoveDiacritics["Ñandú", Language -> "Spanish"]"#).unwrap(),
      "Ñandu"
    );
    assert_eq!(
      interpret(r#"RemoveDiacritics["Ñandú", Language -> "English"]"#).unwrap(),
      "Nandu"
    );
  }
}

mod crossing_detect_corners {
  use super::*;

  #[test]
  fn corner_neighbors_option() {
    assert_eq!(
      interpret(
        "CrossingDetect[{1, -.2, -1, 1}, CornerNeighbors -> None] // Normal"
      )
      .unwrap(),
      "{1, 0, 0, 1}"
    );
    // Only a diagonal neighbor is negative: no crossing without corners.
    assert_eq!(
      interpret(
        "Normal[CrossingDetect[{{1, 0}, {0, -1}}, CornerNeighbors -> False]]"
      )
      .unwrap(),
      "{{0, 0}, {0, 0}}"
    );
    assert_eq!(
      interpret("Normal[CrossingDetect[{{1, 0}, {0, -1}}]]").unwrap(),
      "{{1, 0}, {0, 0}}"
    );
  }
}

mod name_q_aliases {
  use super::*;

  #[test]
  fn resolve_context_aliases() {
    assert_eq!(
      interpret(r#"NameQ["abc`newSymbol", ResolveContextAliases -> False]"#)
        .unwrap(),
      "False"
    );
    assert_eq!(
      interpret(r#"NameQ["Plus", ResolveContextAliases -> True]"#).unwrap(),
      "True"
    );
  }
}

mod time_value_present {
  use super::*;

  #[test]
  fn two_argument_form_is_time_zero() {
    assert_eq!(
      interpret("TimeValue[Annuity[100, 12], .06] == TimeValue[Annuity[100, 12], .06, 0]")
        .unwrap(),
      "True"
    );
  }
}
