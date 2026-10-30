// Argument forms and options found missing by the reference-page parameter
// sweep (second batch).

use super::*;

mod det_modulus {
  use super::*;

  #[test]
  fn small_matrices() {
    assert_eq!(
      interpret("Det[{{1, 2}, {3, 4}}, Modulus -> 47]").unwrap(),
      "45"
    );
    assert_eq!(
      interpret("Det[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}, Modulus -> 7]")
        .unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Det[{{2, 4}, {1, 2}}, Modulus -> 47]").unwrap(),
      "0"
    );
  }

  #[test]
  fn composite_modulus() {
    // Det = -2, and -2 mod 4 = 2; mod 6 = 4.
    assert_eq!(
      interpret("Det[{{1, 2}, {3, 4}}, Modulus -> 4]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Det[{{1, 2}, {3, 4}}, Modulus -> 6]").unwrap(),
      "4"
    );
  }

  #[test]
  fn large_entries_do_not_overflow() {
    // Regression: the integer Bareiss determinant overflowed (and panicked)
    // before reducing modulo p.
    let m = "Table[10^17 + i^3 + 7 j^2 + i j, {i, 12}, {j, 12}]";
    assert_eq!(
      interpret(&format!("Det[{m}, Modulus -> 47] == Mod[Det[{m}], 47]"))
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(&format!(
        "Det[{m}, Modulus -> 1000000007] == Mod[Det[{m}], 1000000007]"
      ))
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn inverse_with_large_entries() {
    let m = "{{10^18 + 1, 3}, {5, 10^18 + 7}}";
    assert_eq!(
      interpret(&format!("Mod[Inverse[{m}, Modulus -> 101] . {m}, 101]"))
        .unwrap(),
      "{{1, 0}, {0, 1}}"
    );
  }
}

mod target_structure {
  use super::*;

  #[test]
  fn dense_is_the_plain_matrix() {
    assert_eq!(
      interpret("ToeplitzMatrix[3, TargetStructure -> \"Dense\"]").unwrap(),
      "{{1, 2, 3}, {2, 1, 2}, {3, 2, 1}}"
    );
    assert_eq!(
      interpret("HankelMatrix[3, TargetStructure -> \"Dense\"]").unwrap(),
      "{{1, 2, 3}, {2, 3, 0}, {3, 0, 0}}"
    );
    assert_eq!(
      interpret("VandermondeMatrix[{1, 1, 4}, TargetStructure -> \"Dense\"]")
        .unwrap(),
      "{{1, 1, 1}, {1, 1, 1}, {1, 4, 16}}"
    );
    assert_eq!(
      interpret("DiagonalMatrix[{1, 2}, TargetStructure -> \"Dense\"]")
        .unwrap(),
      "{{1, 0}, {0, 2}}"
    );
  }

  #[test]
  fn sparse_wraps_in_sparse_array() {
    assert_eq!(
      interpret(
        "Normal[ToeplitzMatrix[{1, 2}, TargetStructure -> \"Sparse\"]] == \
         ToeplitzMatrix[{1, 2}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Head[HankelMatrix[2, TargetStructure -> \"Sparse\"]]")
        .unwrap(),
      "SparseArray"
    );
  }
}

mod vandermonde_columns {
  use super::*;

  #[test]
  fn given_number_of_columns() {
    assert_eq!(
      interpret("VandermondeMatrix[{0, 1/2, 1}, 4]").unwrap(),
      "{{1, 0, 0, 0}, {1, 1/2, 1/4, 1/8}, {1, 1, 1, 1}}"
    );
    assert_eq!(
      interpret("VandermondeMatrix[{a, b, c}, 2]").unwrap(),
      "{{1, a}, {1, b}, {1, c}}"
    );
    assert_eq!(
      interpret(
        "Dimensions[VandermondeMatrix[{0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}, 11]]"
      )
      .unwrap(),
      "{7, 11}"
    );
  }
}

mod cross_matrix_width {
  use super::*;

  #[test]
  fn cross_in_a_larger_matrix() {
    assert_eq!(
      interpret("CrossMatrix[1, 5]").unwrap(),
      "{{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}, \
       {0, 0, 0, 0, 0}}"
    );
    assert_eq!(
      interpret("CrossMatrix[{1, 2}, {3, 7}]").unwrap(),
      "{{0, 0, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 0, 0}}"
    );
    assert_eq!(
      interpret("Total[CrossMatrix[3, {11, 11}], 2]").unwrap(),
      "13"
    );
    assert_eq!(
      interpret("CrossMatrix[3, {11, 11}][[6]]").unwrap(),
      "{0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0}"
    );
  }
}

mod triangular_offset {
  use super::*;

  #[test]
  fn lower_with_offset() {
    let m = "{{1, 2, 0}, {1, 1, 1}, {1, 1, 1}}";
    assert_eq!(
      interpret(&format!("LowerTriangularMatrixQ[{m}, 1]")).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(&format!("LowerTriangularMatrixQ[{m}]")).unwrap(),
      "False"
    );
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{{1, 0}, {0, 1}}, -1]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{{0, 0}, {1, 0}}, -1]").unwrap(),
      "True"
    );
  }

  #[test]
  fn upper_with_offset() {
    assert_eq!(
      interpret(
        "UpperTriangularMatrixQ[{{1, 1, 1}, {2, 1, 1}, {0, 3, 1}}, -1]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "UpperTriangularMatrixQ[{{1, 1, 1}, {2, 1, 1}, {4, 3, 1}}, -1]"
      )
      .unwrap(),
      "False"
    );
    assert_eq!(
      interpret("UpperTriangularMatrixQ[{{0, 1}, {0, 0}}, 1]").unwrap(),
      "True"
    );
  }
}

mod matrix_q_same_test {
  use super::*;

  #[test]
  fn antisymmetric_with_same_test() {
    let m = "{{0, Sqrt[x^2]}, {-x, 0}}";
    assert_eq!(
      interpret(&format!("AntisymmetricMatrixQ[{m}]")).unwrap(),
      "False"
    );
    assert_eq!(
      interpret(&format!(
        "AntisymmetricMatrixQ[{m}, SameTest -> (Simplify[#1 - #2, x > 0] == 0 &)]"
      ))
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn symmetric_with_same_test() {
    assert_eq!(
      interpret(
        "SymmetricMatrixQ[{{1, Sqrt[x^2]}, {x, 1}}, \
         SameTest -> (Simplify[#1 - #2, x > 0] == 0 &)]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("SymmetricMatrixQ[{{1, 2}, {3, 1}}, SameTest -> Equal]")
        .unwrap(),
      "False"
    );
  }
}

mod operator_forms {
  use super::*;

  #[test]
  fn key_member_q() {
    assert_eq!(
      interpret("KeyMemberQ[a] /@ {<|a -> 1|>, <|b -> 2|>}").unwrap(),
      "{True, False}"
    );
  }

  #[test]
  fn reverse_sort_by() {
    assert_eq!(
      interpret("ReverseSortBy[Last][{{1, 3}, {2, 1}, {2, 2}, {3, 2}}]")
        .unwrap(),
      interpret("ReverseSortBy[{{1, 3}, {2, 1}, {2, 2}, {3, 2}}, Last]")
        .unwrap()
    );
  }
}

mod string_template_options {
  use super::*;

  #[test]
  fn combiner_function() {
    assert_eq!(
      interpret(
        "t = StringTemplate[\"Best friend of `1` is `2`.\", \
         CombinerFunction -> Composition[ToUpperCase, StringJoin]]; \
         t[\"John\", \"Jane\"]"
      )
      .unwrap(),
      "BEST FRIEND OF JOHN IS JANE."
    );
  }

  #[test]
  fn insertion_function() {
    assert_eq!(
      interpret(
        "t = StringTemplate[\"Best friend of `1` is `2`.\", \
         InsertionFunction -> ToUpperCase]; t[\"John\", \"Jane\"]"
      )
      .unwrap(),
      "Best friend of JOHN is JANE."
    );
    assert_eq!(
      interpret(
        "StringTemplate[\"`a` + `b`\", InsertionFunction -> (#^2 &)][<|\"a\" -> 2, \"b\" -> 3|>]"
      )
      .unwrap(),
      "4 + 9"
    );
  }

  #[test]
  fn template_slot_insertion_function() {
    assert_eq!(
      interpret(
        "TemplateApply[TemplateSlot[\"data\", InsertionFunction -> Total], \
         <|\"data\" -> Range[10]|>]"
      )
      .unwrap(),
      "55"
    );
  }
}

mod value_q_method {
  use super::*;

  #[test]
  fn trial_evaluation() {
    assert_eq!(
      interpret("ValueQ[Sin[2], Method -> \"TrialEvaluation\"]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("ValueQ[Sin[2.], Method -> \"TrialEvaluation\"]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("ValueQ[1 + 1, Method -> \"TrialEvaluation\"]").unwrap(),
      "True"
    );
  }
}

mod date_options {
  use super::*;

  #[test]
  fn absolute_time_time_zone() {
    assert_eq!(
      interpret(
        "Abs[AbsoluteTime[TimeZone -> 2] - AbsoluteTime[TimeZone -> 0] - 7200] < 1"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn from_unix_time_granularity() {
    assert_eq!(
      interpret(
        "FromUnixTime[960000000, TimeZone -> 0, DateGranularity -> \"Day\"]"
      )
      .unwrap(),
      "DateObject[{2000, 6, 3}, Day, Gregorian, 0.]"
    );
    assert_eq!(
      interpret(
        "FromUnixTime[960000000, TimeZone -> 5, DateGranularity -> \"Month\"]"
      )
      .unwrap(),
      "DateObject[{2000, 6}, Month, Gregorian, 5.]"
    );
  }
}

mod to_radicals_options {
  use super::*;

  #[test]
  fn cubics_false_keeps_cubic_roots() {
    assert_eq!(
      interpret(
        "ToRadicals[Root[#^3 - 5 #^2 - 7 # + 9 &, 1], Cubics -> False]"
      )
      .unwrap(),
      "Root[9 - 7*#1 - 5*#1^2 + #1^3 & , 1, 0]"
    );
    assert_eq!(
      interpret(
        "ToRadicals[{Root[#^3 - 3 # + 1 &, 1], Root[#^2 - 2 &, 1]}, Cubics -> False]"
      )
      .unwrap(),
      "{Root[1 - 3*#1 + #1^3 & , 1, 0], -Sqrt[2]}"
    );
  }

  #[test]
  fn quartics_false_keeps_quartic_roots() {
    assert_eq!(
      interpret(
        "ToRadicals[Root[#^4 + 3 #^3 - 5 #^2 - 7 # + 9 &, 1], Quartics -> False]"
      )
      .unwrap(),
      "Root[9 - 7*#1 - 5*#1^2 + 3*#1^3 + #1^4 & , 1, 0]"
    );
    assert_eq!(
      interpret("ToRadicals[Root[#^3 - 2 &, 1], Quartics -> False]").unwrap(),
      "2^(1/3)"
    );
  }
}

mod matrix_exp_parameter {
  use super::*;

  #[test]
  fn parameter_scales_the_matrix() {
    assert_eq!(
      interpret(
        "With[{m = {{1, 2}, {3, 4}}}, MatrixExp[m, Parameter -> t] == MatrixExp[m t]]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("MatrixExp[{{0, 1}, {0, 0}}, Parameter -> t]").unwrap(),
      "{{1, t}, {0, 1}}"
    );
  }
}

mod svd_truncated {
  use super::*;

  const M: &str = "N[{{1, 2}, {3, 4}, {5, 6}}]";

  #[test]
  fn largest_singular_values() {
    assert_eq!(
      interpret(&format!("Dimensions /@ SingularValueDecomposition[{M}, 1]"))
        .unwrap(),
      "{{3, 1}, {1, 1}, {2, 1}}"
    );
    assert_eq!(
      interpret(&format!(
        "{{u, s, v}} = SingularValueDecomposition[{M}, 1]; \
         s[[1, 1]] == First[SingularValueList[{M}]]"
      ))
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn smallest_singular_values() {
    assert_eq!(
      interpret(&format!(
        "{{u, s, v}} = SingularValueDecomposition[{M}, -1]; \
         s[[1, 1]] == Last[SingularValueList[{M}]]"
      ))
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(&format!(
        "{{u, s, v}} = SingularValueDecomposition[{M}]; \
         {{u2, s2, v2}} = SingularValueDecomposition[{M}, -1]; \
         u2 == u[[All, {{2}}]] && v2 == v[[All, {{2}}]]"
      ))
      .unwrap(),
      "True"
    );
  }
}

mod euler_angles_sequence {
  use super::*;

  #[test]
  fn round_trip_through_euler_matrix() {
    for (angles, seq) in [
      ("{0.1, 0.2, 0.3}", "{3, 2, 3}"),
      ("{0.1, 0.2, 0.3}", "{1, 2, 3}"),
      ("{-0.4, 1.1, 2.5}", "{1, 3, 2}"),
      ("{-0.4, 1.1, 2.5}", "{2, 1, 2}"),
      ("{0.7, -0.5, -2.5}", "{3, 1, 2}"),
    ] {
      assert_eq!(
        interpret(&format!(
          "Max[Abs[EulerAngles[EulerMatrix[{angles}, {seq}], {seq}] - {angles}]] < 10^-12"
        ))
        .unwrap(),
        "True",
        "{angles} {seq}"
      );
    }
  }

  #[test]
  fn default_sequence_round_trip() {
    // Regression: the default {3, 2, 3} extraction had the signs of the
    // first and last angle wrong.
    assert_eq!(
      interpret(
        "Max[Abs[EulerAngles[EulerMatrix[{0.1, 0.2, 0.3}]] - {0.1, 0.2, 0.3}]] < 10^-12"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("EulerAngles[{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]").unwrap(),
      "{0, 0, 0}"
    );
  }
}

mod algebraic_number_extension {
  use super::*;

  #[test]
  fn trace_over_a_larger_field() {
    assert_eq!(
      interpret("AlgebraicNumberTrace[5, Extension -> Sqrt[2]]").unwrap(),
      "10"
    );
    assert_eq!(
      interpret(
        "AlgebraicNumberTrace[1 + Sqrt[2], Extension -> Sqrt[2] + Sqrt[3]]"
      )
      .unwrap(),
      "4"
    );
  }

  #[test]
  fn norm_over_a_larger_field() {
    assert_eq!(
      interpret("AlgebraicNumberNorm[3, Extension -> Sqrt[2]]").unwrap(),
      "9"
    );
    assert_eq!(
      interpret(
        "AlgebraicNumberNorm[1 + Sqrt[2], Extension -> Sqrt[2] + Sqrt[3]]"
      )
      .unwrap(),
      "1"
    );
  }
}

mod minimal_polynomial_reducible {
  use super::*;

  #[test]
  fn unexpanded_sum_of_radicals() {
    // Regression: the resultant composition returned a reducible degree-8
    // polynomial here.
    assert_eq!(
      interpret(
        "Exponent[MinimalPolynomial[Sqrt[2] + Sqrt[3] + 3 (1 + Sqrt[2]), x], x]"
      )
      .unwrap(),
      "4"
    );
  }
}

mod gaussian_prime_counts {
  use super::*;

  #[test]
  fn prime_omega_and_nu() {
    // 28 = -(1 + I)^4 * 7
    assert_eq!(
      interpret("PrimeOmega[28, GaussianIntegers -> True]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("PrimeNu[28, GaussianIntegers -> True]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("PrimeOmega[5, GaussianIntegers -> True]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("PrimeOmega[28, GaussianIntegers -> False]").unwrap(),
      "3"
    );
  }
}

mod absolute_correlation_distribution {
  use super::*;

  #[test]
  fn binormal() {
    assert_eq!(
      interpret("AbsoluteCorrelation[BinormalDistribution[r], 2, 1]").unwrap(),
      "r"
    );
    assert_eq!(
      interpret("AbsoluteCorrelation[BinormalDistribution[r]]").unwrap(),
      "{{1, r}, {r, 1}}"
    );
  }

  #[test]
  fn nonzero_mean() {
    assert_eq!(
      interpret(
        "AbsoluteCorrelation[MultinormalDistribution[{1, 2}, {{1, 0}, {0, 2}}]]"
      )
      .unwrap(),
      "{{2, 2}, {2, 6}}"
    );
  }
}

mod binary_read_list_count {
  use super::*;

  #[test]
  fn reads_at_most_n_objects() {
    let path = std::env::temp_dir().join("woxi_sweep1_binary_read_list.bin");
    std::fs::write(&path, b"abcdef").unwrap();
    let p = path.to_string_lossy().replace('\\', "/");
    assert_eq!(
      interpret(&format!("BinaryReadList[\"{p}\", \"Character8\", 3]"))
        .unwrap(),
      "{a, b, c}"
    );
    assert_eq!(
      interpret(&format!("BinaryReadList[\"{p}\", \"Byte\", 2]")).unwrap(),
      "{97, 98}"
    );
    assert_eq!(
      interpret(&format!("BinaryReadList[\"{p}\", \"Character8\"]")).unwrap(),
      "{a, b, c, d, e, f}"
    );
    let _ = std::fs::remove_file(&path);
  }
}

mod random_image_type {
  use super::*;

  #[test]
  fn pixel_type_argument() {
    assert_eq!(
      interpret("ImageType[RandomImage[1, 10, \"Byte\"]]").unwrap(),
      "Byte"
    );
    assert_eq!(
      interpret("ImageDimensions[RandomImage[1, {4, 3}, \"Bit\"]]").unwrap(),
      "{4, 3}"
    );
  }
}

mod names_ignore_case {
  use super::*;

  #[test]
  fn ignore_case() {
    assert_eq!(
      interpret("Names[\"plot3d\", IgnoreCase -> True]").unwrap(),
      "{Plot3D}"
    );
    assert_eq!(
      interpret("Names[\"plot3d\", IgnoreCase -> False]").unwrap(),
      "{}"
    );
  }
}

mod position_smallest_order {
  use super::*;

  #[test]
  fn alphabetic_order() {
    assert_eq!(
      interpret(
        "PositionSmallest[{\"abc\", \"def\", \"aga\", \"del\", \"two\", \"one\", \"One\"}, 3, AlphabeticOrder]"
      )
      .unwrap(),
      "{{1}, {3}, {2}}"
    );
  }

  #[test]
  fn order_groups_equal_values() {
    assert_eq!(
      interpret("PositionSmallest[{3, 1, 1, 5}, 2, Order]").unwrap(),
      interpret("PositionSmallest[{3, 1, 1, 5}, 2]").unwrap()
    );
  }
}

mod multiple_harmonic_number_weights {
  use super::*;

  #[test]
  fn with_unit_weights_matches_two_argument_form() {
    assert_eq!(
      interpret("MultipleHarmonicNumber[3, {2, 1}, {1, 1}]").unwrap(),
      "5/12"
    );
  }

  #[test]
  fn with_weights() {
    // Sum over 3 >= i > j >= 1 of x^i/i^2 * y^j/j
    assert_eq!(
      interpret("MultipleHarmonicNumber[3, {2, 1}, {x, y}] // Expand").unwrap(),
      interpret("Expand[x^2 y/4 + x^3 y/9 + x^3 y^2/18]").unwrap()
    );
    assert_eq!(
      interpret(
        "MultipleHarmonicNumber[10, {2, 3}, {1/4, 1/6}] == \
         Sum[(1/4)^i/i^2 (1/6)^j/j^3, {i, 10}, {j, i - 1}]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod arc_length_charts {
  use super::*;

  #[test]
  fn cylindrical_helix() {
    assert_eq!(
      interpret("ArcLength[{1, t, t}, {t, 0, 2 Pi}, \"Cylindrical\"]").unwrap(),
      "2*Sqrt[2]*Pi"
    );
  }

  #[test]
  fn polar_spiral() {
    assert_eq!(
      interpret("ArcLength[{Exp[-t/10], t}, {t, 0, Infinity}, \"Polar\"]")
        .unwrap(),
      "Sqrt[101]"
    );
    assert_eq!(
      interpret("ArcLength[{1, t}, {t, 0, Pi}, \"Polar\"]").unwrap(),
      "Pi"
    );
  }

  #[test]
  fn spherical_equator() {
    assert_eq!(
      interpret("ArcLength[{1, Pi/2, t}, {t, 0, 2 Pi}, \"Spherical\"]")
        .unwrap(),
      "2*Pi"
    );
  }
}

mod angle_bisector_interior {
  use super::*;

  #[test]
  fn interior_is_the_default() {
    assert_eq!(
      interpret("AngleBisector[{{1, 0}, {0, 0}, {1, Sqrt[3]}}, \"Interior\"]")
        .unwrap(),
      interpret("AngleBisector[{{1, 0}, {0, 0}, {1, Sqrt[3]}}]").unwrap()
    );
  }
}

mod polynomial_modulus {
  use super::*;

  #[test]
  fn apart() {
    assert_eq!(
      interpret("Apart[(x^2 + 1)/(x + 1), Modulus -> 2]").unwrap(),
      "1 + x"
    );
    assert_eq!(
      interpret("Apart[1/(x^2 + x), Modulus -> 3]").unwrap(),
      "x^(-1) + 2/(1 + x)"
    );
  }

  #[test]
  fn apart_square_free() {
    assert_eq!(
      interpret("ApartSquareFree[(1 - x^2)/(x^4 + 2 x^2 + 1), Modulus -> 2]")
        .unwrap(),
      "(1 + x)^(-2)"
    );
  }

  #[test]
  fn factor_square_free() {
    assert_eq!(
      interpret("FactorSquareFree[1 + x^4, Modulus -> 2]").unwrap(),
      "(1 + x)^4"
    );
    assert_eq!(
      interpret("FactorSquareFree[x^5 + x^3, Modulus -> 3]").unwrap(),
      "x^3*(1 + x^2)"
    );
    assert_eq!(
      interpret("FactorSquareFree[3 x^2 + 6 x + 3, Modulus -> 5]").unwrap(),
      "3*(1 + x)^2"
    );
  }
}

mod bin_lists_multi {
  use super::*;

  #[test]
  fn widths_per_dimension() {
    assert_eq!(
      interpret("BinLists[{{0.5, 0.5}, {1.5, 0.2}, {1.2, 1.7}}, 1, 1]")
        .unwrap(),
      "{{{{0.5, 0.5}}, {}}, {{{1.5, 0.2}}, {{1.2, 1.7}}}}"
    );
  }

  #[test]
  fn explicit_ranges() {
    assert_eq!(
      interpret(
        "BinLists[{{0.5, 0.5}, {1.5, 0.2}, {1.2, 1.7}}, {0, 2, 1}, {0, 3, 1}]"
      )
      .unwrap(),
      "{{{{0.5, 0.5}}, {}, {}}, {{{1.5, 0.2}}, {{1.2, 1.7}}, {}}}"
    );
  }
}
