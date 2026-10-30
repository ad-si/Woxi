// Options found missing by the reference-page parameter sweep.

use super::*;

mod roots_options {
  use super::*;

  #[test]
  fn cubics_and_quartics_false_keep_root_objects() {
    assert_eq!(
      interpret("Roots[x^3 + 2 x^2 + 3 x + 4 == 0, x, Cubics -> False]")
        .unwrap(),
      "x == Root[4 + 3*#1 + 2*#1^2 + #1^3 & , 1, 0] || \
       x == Root[4 + 3*#1 + 2*#1^2 + #1^3 & , 2, 0] || \
       x == Root[4 + 3*#1 + 2*#1^2 + #1^3 & , 3, 0]"
    );
    assert_eq!(
      interpret("Roots[x^4 - 7x^2 + 5 == 0, x, Quartics -> False]").unwrap(),
      interpret("Roots[x^4 - 7x^2 + 5 == 0, x]").unwrap()
    );
  }

  #[test]
  fn multiplicity_caps_repeated_roots() {
    assert_eq!(
      interpret("Roots[(x - a)^3 (x - b) == 0, x, Multiplicity -> 2]").unwrap(),
      "x == a || x == a || x == b"
    );
    assert_eq!(
      interpret("Roots[(x - a)^3 (x - b) == 0, x, Multiplicity -> 1]").unwrap(),
      "x == a || x == b"
    );
    assert_eq!(
      interpret("Roots[(x - 2)^2 == 0, x, Multiplicity -> 1]").unwrap(),
      "x == 2"
    );
  }

  #[test]
  fn modulus_gives_the_residue_roots() {
    assert_eq!(
      interpret("Roots[x^4 - 2x^3 + 5x - 10 == 0, x, Modulus -> 12]").unwrap(),
      "x == 2 || x == 7 || x == 10 || x == 11"
    );
  }

  #[test]
  fn unknown_option_stays_unevaluated() {
    assert_eq!(
      interpret("Roots[x^2 == 1, x, Foo -> 1]").unwrap(),
      "Roots[x^2 == 1, x, Foo -> 1]"
    );
  }
}

mod heads_option {
  use super::*;

  #[test]
  fn leaf_count_without_heads() {
    assert_eq!(
      interpret("LeafCount[{f[][x], g[y][x], h[g[y]][x]}, Heads -> False]")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("LeafCount[{f[][x], g[y][x], h[g[y]][x]}, Heads -> True]")
        .unwrap(),
      "10"
    );
    assert_eq!(
      interpret("LeafCount[f[a, b], Heads -> False]").unwrap(),
      "2"
    );
  }

  #[test]
  fn inactivate_leaves_compound_heads() {
    assert_eq!(
      interpret("Inactivate[f[x][y] + g[z], Heads -> False]").unwrap(),
      "Inactive[Plus][f[x][y], Inactive[g][z]]"
    );
    assert_eq!(
      interpret("Inactivate[f[x][y] + g[z], Heads -> True]").unwrap(),
      "Inactive[Plus][Inactive[f][x][y], Inactive[g][z]]"
    );
  }
}

mod matrix_predicate_options {
  use super::*;

  #[test]
  fn normalized_false_only_needs_orthogonal_columns() {
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{a, b}, {b, -a}}, Normalized -> False]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "m = MapAt[2#&, RotationMatrix[Pi/4, {1, 0, 1}], {All, 2}]; \
         {OrthogonalMatrixQ[m], OrthogonalMatrixQ[m, Normalized -> False], \
         OrthogonalMatrixQ[Transpose[m], Normalized -> False]}"
      )
      .unwrap(),
      "{False, True, False}"
    );
    assert_eq!(
      interpret(
        "UnitaryMatrixQ[{{a, b}, {Conjugate[b], -Conjugate[a]}}, \
         Normalized -> False]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn definiteness_tolerance_is_relative_to_the_largest_eigenvalue() {
    let m = "{{2., 0, 0}, {0, 1., 0}, {0, 0, -2.6*^-12}}";
    assert_eq!(
      interpret(&format!("PositiveSemidefiniteMatrixQ[{m}]")).unwrap(),
      "False"
    );
    assert_eq!(
      interpret(&format!(
        "PositiveSemidefiniteMatrixQ[{m}, Tolerance -> 10^-11]"
      ))
      .unwrap(),
      "True"
    );
    // 2.6*^-12 exceeds 10^-12 times the largest eigenvalue 2.
    assert_eq!(
      interpret(&format!(
        "PositiveSemidefiniteMatrixQ[{m}, Tolerance -> 10^-12]"
      ))
      .unwrap(),
      "False"
    );
    assert_eq!(
      interpret(&format!("IndefiniteMatrixQ[{m}, Tolerance -> 10^-11]"))
        .unwrap(),
      "False"
    );
    assert_eq!(
      interpret(
        "NegativeSemidefiniteMatrixQ[{{-2., 0, 0}, {0, -1., 0}, \
         {0, 0, 5.9*^-13}}, Tolerance -> 10^-12]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn matrix_rank_and_row_reduce_tolerance() {
    let m = "N[{{1, 1, 1}, {0, 10^-10, 0}, {0, 0, 10^-20}}]";
    assert_eq!(
      interpret(&format!("MatrixRank[{m}, Tolerance -> 0]")).unwrap(),
      "3"
    );
    assert_eq!(
      interpret(&format!("MatrixRank[{m}, Tolerance -> 10^-8]")).unwrap(),
      "1"
    );
    let m = "N[{{1, 0, 1}, {1, 10^-20, 0}, {0, 0, 1}}]";
    assert_eq!(
      interpret(&format!("RowReduce[{m}, Tolerance -> 0]")).unwrap(),
      "{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}"
    );
    assert_eq!(
      interpret(&format!("RowReduce[{m}, Tolerance -> 10^-10]")).unwrap(),
      "{{1., 0., 0.}, {0., 0., 1.}, {0., 0., 0.}}"
    );
  }

  #[test]
  fn singular_value_list_tolerance_zero_keeps_all_values() {
    assert_eq!(
      interpret(
        "Length[SingularValueList[N[HilbertMatrix[16]], Tolerance -> 0]]"
      )
      .unwrap(),
      "16"
    );
    assert_eq!(
      interpret(
        "Length[SingularValueList[N[HilbertMatrix[10]], Tolerance -> 0.1]]"
      )
      .unwrap(),
      "2"
    );
  }

  #[test]
  fn large_exact_matrix_does_not_overflow() {
    // The integer characteristic polynomial of HilbertMatrix[16] overflows
    // i128; this used to panic.
    assert!(interpret("SingularValueList[HilbertMatrix[16]]").is_ok());
  }
}

mod polynomial_modulus {
  use super::*;

  #[test]
  fn polynomial_mod_with_modulus() {
    assert_eq!(
      interpret("PolynomialMod[x^2, 2x - 1, Modulus -> 3]").unwrap(),
      "1"
    );
    assert_eq!(interpret("PolynomialMod[x^2, {2x - 1, 3}]").unwrap(), "1");
    assert_eq!(
      interpret("PolynomialMod[3x^2 + 9x + 5, x^2 + 1, Modulus -> 7]").unwrap(),
      "2 + 2*x"
    );
  }

  #[test]
  fn decompose_with_modulus() {
    assert_eq!(
      interpret("Decompose[x^3 + 2x^2 + 4x + 7, x, Modulus -> 2]").unwrap(),
      "{1 + x^3}"
    );
    assert_eq!(
      interpret("Decompose[7x^4 + 2x^2 + 3, x, Modulus -> 5]").unwrap(),
      "{3 + 2*x + 2*x^2, x^2}"
    );
    assert_eq!(
      interpret("Decompose[x^4 + 2x^2 + 3, x, Modulus -> 0]").unwrap(),
      "{3 + 2*x + x^2, x^2}"
    );
  }
}

mod trig_option {
  use super::*;

  #[test]
  fn factor_functions_factor_trigonometrically() {
    assert_eq!(
      interpret("Factor[Sin[x] + Sin[y], Trig -> True]").unwrap(),
      "2*Cos[x/2 - y/2]*Sin[x/2 + y/2]"
    );
    assert_eq!(
      interpret("FactorList[Sin[x] + Sin[y], Trig -> True]").unwrap(),
      "{{2, 1}, {Cos[x/2 - y/2], 1}, {Sin[x/2 + y/2], 1}}"
    );
    assert_eq!(
      interpret("FactorSquareFree[1 - Cos[2x], Trig -> True]").unwrap(),
      "2*Sin[x]^2"
    );
    assert_eq!(
      interpret("FactorSquareFreeList[1 - Cos[2x], Trig -> True]").unwrap(),
      "{{2, 1}, {Sin[x], 2}}"
    );
    assert_eq!(
      interpret("Factor[x^2 - 1, Trig -> False]").unwrap(),
      "(-1 + x)*(1 + x)"
    );
  }

  #[test]
  fn cancel_and_together_treat_sin_and_cos_as_variables() {
    assert_eq!(
      interpret("Cancel[Sin[2x]/Sin[x], Trig -> True]").unwrap(),
      "2*Cos[x]"
    );
    assert_eq!(
      interpret("Together[1/Sin[x] + 1/Cos[x], Trig -> True]").unwrap(),
      "Csc[x]*Sec[x]*(Cos[x] + Sin[x])"
    );
  }
}

mod subset_position_overlaps {
  use super::*;

  #[test]
  fn overlaps_false_drops_sublists_sharing_a_position() {
    assert_eq!(
      interpret(
        "SubsetPosition[{a, b, c, 1, 2}, {_Symbol, _Integer}, Overlaps -> False]"
      )
      .unwrap(),
      "{{1, 4}, {2, 5}}"
    );
    assert_eq!(
      interpret(
        "SubsetPosition[{a, b, c, 1, 2}, {_Symbol, _Integer}, Overlaps -> True]"
      )
      .unwrap(),
      "{{1, 4}, {1, 5}, {2, 4}, {2, 5}, {3, 4}, {3, 5}}"
    );
  }
}

mod resolve_context_aliases {
  use super::*;

  #[test]
  fn names_and_contexts_can_take_an_alias_literally() {
    clear_state();
    assert_eq!(
      interpret(
        "$ContextAliases[\"abc`\"] = \"NewContext`\"; abc`newSymbol; \
         {Names[\"abc`newSymbol\"], \
         Names[\"abc`newSymbol\", ResolveContextAliases -> False], \
         Contexts[\"abc`\"], \
         Contexts[\"abc`\", ResolveContextAliases -> False]}"
      )
      .unwrap(),
      "{{NewContext`newSymbol}, {}, {NewContext`}, {}}"
    );
    clear_state();
  }
}

mod time_zone_option {
  use super::*;

  #[test]
  fn absolute_time_reads_the_date_in_the_zone() {
    // New York is on daylight saving time (UTC-4) in August.
    assert_eq!(
      interpret(
        "AbsoluteTime[{2022, 8, 4, 9, 14, 2}, TimeZone -> \"America/New_York\"] \
         - AbsoluteTime[{2022, 8, 4, 9, 14, 2}, TimeZone -> -4]"
      )
      .unwrap(),
      "0"
    );
    assert_eq!(
      interpret(
        "AbsoluteTime[{2022, 8, 4}, TimeZone -> -4] - \
         AbsoluteTime[{2022, 8, 4}, TimeZone -> 0]"
      )
      .unwrap(),
      "14400"
    );
    assert_eq!(
      interpret(
        "AbsoluteTime[{2022, 1, 4}, TimeZone -> -5] - \
         AbsoluteTime[{2022, 1, 4}, TimeZone -> \"America/New_York\"]"
      )
      .unwrap(),
      "0"
    );
  }

  #[test]
  fn date_list_in_a_named_zone() {
    assert_eq!(
      interpret(
        "DateList[3847240648, TimeZone -> \"Asia/Tokyo\"] == \
         DateList[3847240648, TimeZone -> 9]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod similarity_rules {
  use super::*;

  #[test]
  fn rules_score_the_listed_pairs() {
    assert_eq!(
      interpret(
        "SmithWatermanSimilarity[\"ABCDEF\", \"ABCOEF\", \
         SimilarityRules -> {{\"D\", \"O\"} -> 0.5}]"
      )
      .unwrap(),
      "5.5"
    );
    assert_eq!(
      interpret("SmithWatermanSimilarity[\"ABCDEF\", \"ABCOEF\"]").unwrap(),
      "4."
    );
    assert_eq!(
      interpret(
        "NeedlemanWunschSimilarity[\"ABCDEF\", \"ABCOEF\", \
         SimilarityRules -> {{\"D\", \"O\"} -> 0.5}]"
      )
      .unwrap(),
      "5.5"
    );
  }
}

mod find_repeat_level {
  use super::*;

  #[test]
  fn a_one_element_level_list_compares_whole_rows() {
    assert_eq!(
      interpret(
        "FindRepeat[{{1, 2}, {5, -2}, {2, 1}, {0, 3}, {-1, 4}}, {1}, \
         SameTest -> (Total[#1] == Total[#2] &)]"
      )
      .unwrap(),
      "{{1, 2}}"
    );
    assert_eq!(
      interpret("FindRepeat[{1, 2, 1, 2, 1}, {2}]").unwrap(),
      "{1, 2}"
    );
  }
}

mod moon_phase_options {
  use super::*;

  #[test]
  fn time_direction_finds_the_previous_event() {
    assert_eq!(
      interpret(
        "DateValue[FullMoon[DateObject[{2024, 1, 1}], TimeDirection -> -1], \
         {\"Year\", \"Month\", \"Day\"}]"
      )
      .unwrap(),
      "{2023, 12, 27}"
    );
    assert_eq!(
      interpret(
        "DateValue[FullMoon[DateObject[{2024, 1, 1}]], \
         {\"Year\", \"Month\", \"Day\"}]"
      )
      .unwrap(),
      "{2024, 1, 25}"
    );
    assert_eq!(
      interpret(
        "DateValue[MoonPhaseDate[DateObject[{2024, 1, 1}], \"FirstQuarter\", \
         TimeDirection -> -1], {\"Year\", \"Month\", \"Day\"}]"
      )
      .unwrap(),
      "{2023, 12, 19}"
    );
  }

  #[test]
  fn time_zone_and_day_granularity() {
    assert_eq!(
      interpret(
        "FullMoon[DateObject[{2024, 1, 1}], TimeZone -> 2, \
         DateGranularity -> \"Day\"]"
      )
      .unwrap(),
      "DateObject[{2024, 1, 25}, Day, Gregorian, 2.]"
    );
  }
}

mod scaling_sides {
  use super::*;

  #[test]
  fn assumptions_decide_the_sign() {
    assert_eq!(
      interpret("DivideSides[a x^2 > b x + c, a, Assumptions -> a > 0]")
        .unwrap(),
      "x^2 > (c + b*x)/a"
    );
    assert_eq!(
      interpret("DivideSides[a x^2 > b x + c, a, Assumptions -> a < 0]")
        .unwrap(),
      "x^2 < (c + b*x)/a"
    );
    assert_eq!(
      interpret("DivideSides[a == b, c, Assumptions -> c > 0]").unwrap(),
      "a/c == b/c"
    );
  }

  #[test]
  fn a_negative_number_reverses_an_inequality() {
    assert_eq!(
      interpret("DivideSides[x > 1, -2]").unwrap(),
      "-1/2*x < -1/2"
    );
    assert_eq!(
      interpret("MultiplySides[x >= 1, -2]").unwrap(),
      "-2*x <= -2"
    );
  }
}
