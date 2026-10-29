// Argument forms and options found missing by the second reference-page
// parameter sweep (every documented example run through Woxi).

use super::*;

mod map_indexed_heads {
  use super::*;

  #[test]
  fn compound_head_with_infinite_levels() {
    assert_eq!(
      interpret("MapIndexed[f, p[x][a, b, c], Infinity, Heads -> True]")
        .unwrap(),
      "f[f[p, {0, 0}][f[x, {0, 1}]], {0}][f[a, {1}], f[b, {2}], f[c, {3}]]"
    );
  }

  #[test]
  fn list_heads_at_every_level() {
    assert_eq!(
      interpret("MapIndexed[f, {a, {b}}, Infinity, Heads -> True]").unwrap(),
      "f[List, {0}][f[a, {1}], f[f[List, {2, 0}][f[b, {2, 1}]], {2}]]"
    );
  }

  #[test]
  fn heads_false_is_the_plain_form() {
    assert_eq!(
      interpret("MapIndexed[f, {a, {b}}, 2, Heads -> False]").unwrap(),
      "{f[a, {1}], f[{f[b, {2, 1}]}, {2}]}"
    );
  }
}

mod delete_cases_heads {
  use super::*;

  #[test]
  fn deleting_a_head_splices_its_arguments() {
    assert_eq!(
      interpret("DeleteCases[{1, f[2, 3], 4}, f, {2}, Heads -> True]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  #[test]
  fn heads_false_leaves_heads_alone() {
    assert_eq!(
      interpret("DeleteCases[{1, f[2, 3], 4}, f, {2}, Heads -> False]")
        .unwrap(),
      "{1, f[2, 3], 4}"
    );
  }

  #[test]
  fn deletion_count_includes_heads() {
    assert_eq!(
      interpret(
        "DeleteCases[{1, f[2, 3], g[4]}, f | g, {2}, 1, Heads -> True]"
      )
      .unwrap(),
      "{1, 2, 3, g[4]}"
    );
  }
}

mod operator_forms {
  use super::*;

  #[test]
  fn all_same_by() {
    assert_eq!(
      interpret("AllSameBy[Length][{{1, 2, 3}, {4, 5, 6}, {7, 8}}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AllSameBy[Length][{{1, 2}, {4, 5}}]").unwrap(),
      "True"
    );
    assert_eq!(interpret("AllSameBy[Length]").unwrap(), "AllSameBy[Length]");
  }

  #[test]
  fn map_apply() {
    assert_eq!(
      interpret("MapApply[f][{{a, b}, {c}, {d, e}}]").unwrap(),
      "{f[a, b], f[c], f[d, e]}"
    );
    assert_eq!(interpret("MapApply[f]").unwrap(), "MapApply[f]");
  }

  #[test]
  fn date_select_with_named_slots() {
    assert_eq!(
      interpret(
        "fun = DateSelect[(OddQ[#Month] && #Day == 4) &]; \
         fun[{DateObject[{2020, 1, 4}], DateObject[{2020, 2, 4}], \
         DateObject[{2020, 3, 5}]}]"
      )
      .unwrap(),
      "{DateObject[{2020, 1, 4}, Day]}"
    );
  }
}

mod counts_with_keys {
  use super::*;

  #[test]
  fn keys_in_the_given_order_with_zero_counts() {
    assert_eq!(
      interpret("Counts[{a, b, d, d, a}, {a, b, c, d}]").unwrap(),
      "<|a -> 2, b -> 1, c -> 0, d -> 2|>"
    );
  }

  #[test]
  fn unlisted_elements_are_not_counted() {
    assert_eq!(
      interpret("Counts[{1, 1, 0, 2, 1}, {1, 0}]").unwrap(),
      "<|1 -> 3, 0 -> 1|>"
    );
  }
}

mod subsequences_limit {
  use super::*;

  #[test]
  fn first_s_subsequences() {
    assert_eq!(
      interpret("Subsequences[{a, b, c, d, e}, {3}, 2]").unwrap(),
      "{{a, b, c}, {b, c, d}}"
    );
    assert_eq!(
      interpret("Subsequences[{a, b, c}, {1, 2}, 10]").unwrap(),
      interpret("Subsequences[{a, b, c}, {1, 2}]").unwrap()
    );
  }
}

mod nsolve_values_precision {
  use super::*;

  #[test]
  fn domain_and_precision() {
    assert_eq!(
      interpret("NSolveValues[x^5 - x + 2, x, Reals, 30]").unwrap(),
      interpret("x /. NSolve[x^5 - x + 2, x, Reals, 30]").unwrap()
    );
  }
}

mod list_fourier_sequence_transform_start {
  use super::*;

  #[test]
  fn start_index() {
    assert_eq!(
      interpret("ListFourierSequenceTransform[{1, 0, -1}, w, -1]").unwrap(),
      "-E^(-I*w) + E^(I*w)"
    );
    assert_eq!(
      interpret(
        "ListFourierSequenceTransform[{1, 2, 3}, w, 2] == \
         Sum[{1, 2, 3}[[k - 1]] E^(-I k w), {k, 2, 4}]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod rank_correlation_matrices {
  use super::*;

  #[test]
  fn spearman_rho_of_the_columns() {
    assert_eq!(
      interpret("SpearmanRho[{{1, 2}, {2, 1}, {3, 5}}]").unwrap(),
      "{{1, 1/2}, {1/2, 1}}"
    );
  }

  #[test]
  fn kendall_tau_of_the_columns() {
    assert_eq!(
      interpret("KendallTau[{{1, 2, 4}, {2, 1, 3}, {3, 5, 1}}]").unwrap(),
      "{{1, 1/3, -1}, {1/3, 1, -1/3}, {-1, -1/3, 1}}"
    );
  }

  #[test]
  fn sparse_arrays_are_expanded() {
    assert_eq!(
      interpret(
        "SpearmanRho[SparseArray[{{2, 2} -> 1, {5, 3} -> 2, {10, 1} -> 3}]]"
      )
      .unwrap(),
      "{{1, -1/9, -1/9}, {-1/9, 1, -1/9}, {-1/9, -1/9, 1}}"
    );
  }
}

mod boolean_terms {
  use super::*;

  #[test]
  fn native_form_is_the_plain_result() {
    assert_eq!(
      interpret("BooleanMinterms[{1, 3, 5}, {a, b, c}, \"DNF\"]").unwrap(),
      interpret("BooleanMinterms[{1, 3, 5}, {a, b, c}]").unwrap()
    );
    assert_eq!(
      interpret("BooleanMaxterms[{1, 3, 5}, {a, b, c}, \"CNF\"]").unwrap(),
      interpret("BooleanMaxterms[{1, 3, 5}, {a, b, c}]").unwrap()
    );
  }

  #[test]
  fn exponent_vector_form_is_inert() {
    let r =
      interpret_with_stdout("BooleanMinterms[{{0, 1, 1}, {1, 0, 1}}]").unwrap();
    assert_eq!(r.result, "BooleanMinterms[{{0, 1, 1}, {1, 0, 1}}]");
    assert!(r.warnings.is_empty());
    assert_eq!(
      interpret("BooleanMaxterms[{{0, 1, 1}}]").unwrap(),
      "BooleanMaxterms[{{0, 1, 1}}]"
    );
  }
}

mod string_options {
  use super::*;

  #[test]
  fn palindrome_ignore_case_and_diacritics() {
    assert_eq!(
      interpret("PalindromeQ[\"Anna\", IgnoreCase -> True]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("PalindromeQ[\"Anna\", IgnoreCase -> False]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PalindromeQ[\"erigiré\", IgnoreDiacritics -> True]").unwrap(),
      "True"
    );
  }

  #[test]
  fn word_counts_ignore_case() {
    let s =
      "\"Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo\"";
    assert_eq!(
      interpret(&format!("WordCounts[{s}, IgnoreCase -> False]")).unwrap(),
      "<|buffalo -> 5, Buffalo -> 3|>"
    );
    assert_eq!(
      interpret(&format!("WordCounts[{s}, IgnoreCase -> True]")).unwrap(),
      "<|buffalo -> 8|>"
    );
  }
}

mod matrix_options {
  use super::*;

  #[test]
  fn singular_value_list_tolerance() {
    assert_eq!(
      interpret("SingularValueList[{{1, 0}, {0, 0.01}}, Tolerance -> 0.1]")
        .unwrap(),
      "{1.}"
    );
    assert_eq!(
      interpret(
        "SingularValueList[{{3, 0, 0}, {0, 2, 0}, {0, 0, 1/10}}, 3, \
         Tolerance -> 1/5]"
      )
      .unwrap(),
      "{3, 2}"
    );
  }

  #[test]
  fn target_structure() {
    assert_eq!(
      interpret(
        "LowerTriangularize[{{1, 2}, {3, 4}}, TargetStructure -> \"Dense\"]"
      )
      .unwrap(),
      "{{1, 0}, {3, 4}}"
    );
    assert_eq!(
      interpret(
        "Normal[UpperTriangularize[{{1, 2}, {3, 4}}, \
         TargetStructure -> \"Sparse\"]]"
      )
      .unwrap(),
      "{{1, 2}, {0, 4}}"
    );
    assert_eq!(
      interpret(
        "HadamardMatrix[4, TargetStructure -> \"Dense\"] == HadamardMatrix[4]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn unitary_and_normal_tolerance() {
    assert_eq!(
      interpret(
        "UnitaryMatrixQ[{{0.6, 0.8}, {-0.8, 0.6000000000001}}, \
         Tolerance -> 10^-12]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "UnitaryMatrixQ[{{0.6, 0.8}, {-0.8, 0.6000000001}}, \
         Tolerance -> 10^-12]"
      )
      .unwrap(),
      "False"
    );
    assert_eq!(
      interpret(
        "NormalMatrixQ[{{1., 2.}, {2., 1.0000000000001}}, Tolerance -> 10^-12]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("NormalMatrixQ[{{1, 2}, {3, 4}}, Tolerance -> 10^-13]")
        .unwrap(),
      "False"
    );
  }

  #[test]
  fn same_test() {
    assert_eq!(
      interpret(
        "NormalMatrixQ[{{x, 1}, {1, x}}, \
         SameTest -> (FullSimplify[#1 - #2, x > 0] == 0 &)]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "UnitaryMatrixQ[{{Cos[x], Sin[x]}, {-Sin[x], Cos[x]}}, \
         SameTest -> (Simplify[#1 - #2, x > 0] == 0 &)]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod image_forms {
  use super::*;

  #[test]
  fn constant_image_type() {
    assert_eq!(
      interpret("ImageType[ConstantImage[Red, {50, 50}, \"Byte\"]]").unwrap(),
      "Byte"
    );
    assert_eq!(
      interpret(
        "ConstantImage[RGBColor[0, 0.5, 1], {50, 50}, ImageSize -> 100]"
      )
      .unwrap(),
      "-Image-"
    );
  }

  #[test]
  fn image_data_without_interleaving() {
    assert_eq!(
      interpret(
        "ImageData[ConstantImage[RGBColor[1, 0.5, 0], {3, 2}], Automatic, \
         Interleaving -> False]"
      )
      .unwrap(),
      "{{{1., 1., 1.}, {1., 1., 1.}}, {{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}}, \
       {{0., 0., 0.}, {0., 0., 0.}}}"
    );
    assert_eq!(
      interpret(
        "ImageData[ConstantImage[0.5, {3, 2}], Automatic, Interleaving -> False]"
      )
      .unwrap(),
      "{{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}}"
    );
  }
}

mod numeric_array_clipping {
  use super::*;

  #[test]
  fn clip_and_coerce() {
    assert_eq!(
      interpret(
        "Normal[NumericArray[{1, 2, 3, $MaxMachineNumber}, \"Real32\", \
         \"ClipAndCoerce\"]]"
      )
      .unwrap(),
      "{1., 2., 3., 3.4028234663852886*^38}"
    );
    assert_eq!(
      interpret(
        "NumericArray[{0, 1000, 2000}, \"UnsignedInteger8\", \
         \"ClipAndCoerce\"] // Normal"
      )
      .unwrap(),
      "{0, 255, 255}"
    );
  }

  #[test]
  fn clip_and_round() {
    assert_eq!(
      interpret(
        "NumericArray[{-500, 3.6, 300}, \"Integer8\", \"ClipAndRound\"] \
         // Normal"
      )
      .unwrap(),
      "{-128, 4, 127}"
    );
  }
}

mod from_julian_date_forms {
  use super::*;

  #[test]
  fn modified_julian_date() {
    assert_eq!(
      interpret("FromJulianDate[\"Modified\", 0, TimeZone -> 0]").unwrap(),
      "DateObject[{1858, 11, 17, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn time_zone_shifts_the_clock() {
    assert_eq!(
      interpret("FromJulianDate[2400001, TimeZone -> 1]").unwrap(),
      "DateObject[{1858, 11, 17, 13, 0, 0}, Instant, Gregorian, 1.]"
    );
  }
}

mod coordinate_arrays_center_and_padding {
  use super::*;

  #[test]
  fn center_offset() {
    assert_eq!(
      interpret("CoordinateBoundsArray[{{0, 2}, {0, 1}}, 1, Center]").unwrap(),
      "{{{1/2, 1/2}}, {{3/2, 1/2}}}"
    );
  }

  #[test]
  fn padding() {
    assert_eq!(
      interpret("CoordinateBoundsArray[{{0, 1}}, 1, 0, 1]").unwrap(),
      "{{-1}, {0}, {1}, {2}}"
    );
    assert_eq!(
      interpret(
        "CoordinateBoundsArray[{{0, 2}, {0, 1}}, 1, Center, {{-1, 3}, {2, 1}}] \
         == CoordinateBoundingBoxArray[{{0, 0}, {2, 1}}, 1, Center, \
         {{-1, 3}, {2, 1}}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "Dimensions[CoordinateBoundsArray[{{0, 2}, {0, 1}}, 1, Center, \
         {{-1, 3}, {2, 1}}]]"
      )
      .unwrap(),
      "{4, 4, 2}"
    );
  }
}

mod random_working_precision {
  use super::*;

  #[test]
  fn random_complex_precision() {
    assert_eq!(
      interpret("Precision[RandomComplex[WorkingPrecision -> 50]]").unwrap(),
      "50."
    );
    assert_eq!(
      interpret(
        "Precision /@ RandomComplex[{-1 - I, 1 + I}, 3, WorkingPrecision -> 20]"
      )
      .unwrap(),
      "{20., 20., 20.}"
    );
    assert_eq!(
      interpret(
        "And @@ (Abs[Re[#]] <= 1 && Abs[Im[#]] <= 1 & /@ \
         RandomComplex[{-1 - I, 1 + I}, 20, WorkingPrecision -> 20])"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn random_real_precision() {
    assert_eq!(
      interpret("Precision[RandomReal[1, WorkingPrecision -> 30]]").unwrap(),
      "30."
    );
    assert_eq!(
      interpret(
        "Dimensions[RandomReal[{2, 3}, {2, 3}, WorkingPrecision -> 25]]"
      )
      .unwrap(),
      "{2, 3}"
    );
  }
}

mod number_theory_options {
  use super::*;

  #[test]
  fn liouville_lambda_gaussian() {
    // 2 = -I (1 + I)^2, 3 is a Gaussian prime, 5 = (2 + I)(2 - I).
    assert_eq!(
      interpret("LiouvilleLambda[#, GaussianIntegers -> True] & /@ {2, 3, 5}")
        .unwrap(),
      "{1, -1, 1}"
    );
  }

  #[test]
  fn factor_square_free_list_modulus() {
    assert_eq!(
      interpret("FactorSquareFreeList[1 + x^4, Modulus -> 2]").unwrap(),
      "{{1, 1}, {1 + x, 4}}"
    );
    assert_eq!(
      interpret("FactorSquareFreeList[x^2 (x + 1)^2 (x + 2), Modulus -> 3]")
        .unwrap(),
      "{{1, 1}, {2 + x, 1}, {x + x^2, 2}}"
    );
  }
}

mod calculus_options {
  use super::*;

  #[test]
  fn discrete_limit_method() {
    assert_eq!(
      interpret(
        "DiscreteLimit[n/(2 n + 1), n -> Infinity, Method -> \"Limit\"]"
      )
      .unwrap(),
      "1/2"
    );
  }

  #[test]
  fn difference_quotient_assumptions() {
    assert_eq!(
      interpret(
        "DifferenceQuotient[Abs[x], {x, h}, Assumptions -> x > 0 && h > 0]"
      )
      .unwrap(),
      "1"
    );
  }

  #[test]
  fn possible_zero_q_options() {
    assert_eq!(
      interpret("PossibleZeroQ[Sqrt[x^2] - x, Assumptions -> x > 0]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("PossibleZeroQ[Sqrt[2]^2 - 2, Method -> \"ExactAlgebraics\"]")
        .unwrap(),
      "True"
    );
  }
}

mod parametric_measures {
  use super::*;

  #[test]
  fn ellipsoid_volume() {
    assert_eq!(
      interpret(
        "Volume[{3 r Cos[p], 2 r Sin[t] Sin[p], r Cos[t] Sin[p]}, {r, 0, 1}, \
         {t, 0, 2 Pi}, {p, 0, Pi}]"
      )
      .unwrap(),
      "8*Pi"
    );
  }

  #[test]
  fn region_measure_of_a_parametrization() {
    assert_eq!(
      interpret(
        "RegionMeasure[{r Cos[t], r Sin[t]}, {{r, 0, 1}, {t, 0, 2 Pi}}]"
      )
      .unwrap(),
      "Pi"
    );
  }

  #[test]
  fn region_measure_in_a_chart() {
    assert_eq!(
      interpret(
        "RegionMeasure[{r, t, z}, {{r, 0, R}, {t, 0, 2 Pi}, {z, 0, Z}}, \
         \"Cylindrical\"]"
      )
      .unwrap(),
      "Pi*R^2*Z"
    );
    assert_eq!(
      interpret(
        "RegionMeasure[{Exp[-t/10], t}, {{t, 0, Infinity}}, \"Polar\"]"
      )
      .unwrap(),
      "Sqrt[101]"
    );
  }
}

mod abs_under_assumptions {
  use super::*;

  #[test]
  fn products_with_even_powers_and_sums() {
    assert_eq!(
      interpret("Simplify[Abs[r^2 p], 0 <= r <= 1 && 0 <= p <= 1]").unwrap(),
      "p*r^2"
    );
    assert_eq!(
      interpret("Refine[Abs[h + x], x > 0 && h > 0]").unwrap(),
      "h + x"
    );
  }

  #[test]
  fn root_of_an_exponential() {
    assert_eq!(
      interpret("Simplify[Sqrt[E^(-t/5)], t > 0]").unwrap(),
      "E^(-1/10*t)"
    );
  }

  #[test]
  fn refined_result_is_simplified_again() {
    assert_eq!(
      interpret(
        "Simplify[Conjugate[Cos[x]] Cos[x] + Conjugate[Sin[x]] Sin[x] - 1, x > 0]"
      )
      .unwrap(),
      "0"
    );
  }
}
