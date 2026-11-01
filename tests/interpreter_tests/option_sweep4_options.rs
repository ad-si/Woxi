// Options that made a call fail or stay unevaluated, found by the
// reference-page parameter sweep (round 4). Expected values are the ones
// the Wolfram reference pages show, or follow from the definitions.

use super::*;

/// `interpret(code)` must evaluate to `True`.
fn assert_true(code: &str) {
  assert_eq!(interpret(code).unwrap(), "True", "{code}");
}

mod extension_automatic {
  use super::*;

  #[test]
  fn together_cancels_over_the_extension() {
    assert_eq!(
      interpret("Together[(x^2 - 2)/(Sqrt[2] + x), Extension -> Automatic]")
        .unwrap(),
      "-Sqrt[2] + x"
    );
  }

  #[test]
  fn cancel_over_the_extension() {
    assert_eq!(
      interpret("Cancel[(x^2 - 2)/(x - Sqrt[2]), Extension -> Automatic]")
        .unwrap(),
      "Sqrt[2] + x"
    );
    assert_eq!(
      interpret("Cancel[(x^2 - 2)/(x - Sqrt[2]), Extension -> Sqrt[2]]")
        .unwrap(),
      "Sqrt[2] + x"
    );
    assert_eq!(
      interpret("Cancel[(x^2 + 1)/(x - I), Extension -> Automatic]").unwrap(),
      "I + x"
    );
  }

  #[test]
  fn without_algebraic_numbers_it_is_the_plain_operation() {
    assert_eq!(
      interpret("Cancel[(x^2 + 1)/(x - 1), Extension -> Automatic]").unwrap(),
      "(1 + x^2)/(-1 + x)"
    );
    assert_eq!(
      interpret("Factor[x^2 - 2, Extension -> Automatic]").unwrap(),
      "-2 + x^2"
    );
    assert_eq!(
      interpret("FactorSquareFree[3 x^2 + 6 x + 3, Extension -> Sqrt[2]]")
        .unwrap(),
      "3*(1 + x)^2"
    );
  }

  #[test]
  fn factor_square_free() {
    assert_eq!(
      interpret(
        "FactorSquareFree[x^2 + 2 Sqrt[2] x + 2, Extension -> Automatic]"
      )
      .unwrap(),
      "(Sqrt[2] + x)^2"
    );
    assert_eq!(
      interpret(
        "FactorSquareFreeList[x^2 + 2 Sqrt[2] x + 2, Extension -> Automatic]"
      )
      .unwrap(),
      "{{1, 1}, {Sqrt[2] + x, 2}}"
    );
  }

  #[test]
  fn factor_with_algebraic_coefficients() {
    assert_eq!(
      interpret("Factor[x^2 + 2 Sqrt[2] x + 2, Extension -> Automatic]")
        .unwrap(),
      "(Sqrt[2] + x)^2"
    );
    // The explicit-extension form of a rational polynomial is unchanged.
    assert_eq!(
      interpret("Factor[x^2 - 2, Extension -> Sqrt[2]]").unwrap(),
      "-((Sqrt[2] - x)*(Sqrt[2] + x))"
    );
  }

  #[test]
  fn another_field_than_the_coefficients_stays_unevaluated() {
    assert_eq!(
      interpret(
        "FactorSquareFree[x^2 + 2 Sqrt[2] x + 2, Extension -> Sqrt[3]]"
      )
      .unwrap(),
      "FactorSquareFree[2 + 2*Sqrt[2]*x + x^2, Extension -> Sqrt[3]]"
    );
  }
}

mod unfactored_sums_are_canonical {
  use super::*;

  // Regression: the rebuilt sum put the variable before the algebraic
  // coefficient (1 + x*Sqrt[2]).
  #[test]
  fn factor_and_factor_square_free() {
    assert_eq!(interpret("Factor[Sqrt[2] x + 1]").unwrap(), "1 + Sqrt[2]*x");
    assert_eq!(
      interpret("Factor[x^2 + 2 Sqrt[2] x + 2]").unwrap(),
      "2 + 2*Sqrt[2]*x + x^2"
    );
    assert_eq!(
      interpret("FactorSquareFree[x^2 + 2 Sqrt[2] x + 2]").unwrap(),
      "2 + 2*Sqrt[2]*x + x^2"
    );
  }
}

mod roots_of_unity {
  use super::*;

  #[test]
  fn minimal_polynomial_is_cyclotomic() {
    assert_eq!(
      interpret("MinimalPolynomial[E^(I Pi/4), x]").unwrap(),
      "1 + x^4"
    );
    assert_eq!(
      interpret("MinimalPolynomial[E^(2 Pi I/5), x]").unwrap(),
      "1 + x + x^2 + x^3 + x^4"
    );
    assert_eq!(
      interpret("MinimalPolynomial[E^(2 Pi I/3), x]").unwrap(),
      "1 + x + x^2"
    );
  }

  #[test]
  fn algebraic_number_norm_in_a_cyclotomic_field() {
    assert_eq!(
      interpret("AlgebraicNumberNorm[Sqrt[2], Extension -> E^(Pi I/4)]")
        .unwrap(),
      "4"
    );
    assert_eq!(interpret("AlgebraicNumberNorm[E^(2 Pi I/5)]").unwrap(), "1");
  }
}

mod roots_equated_to {
  use super::*;

  #[test]
  fn roots_are_equated_to_the_expression() {
    assert_eq!(
      interpret("Roots[x^2 == 4, x, EquatedTo -> z^3]").unwrap(),
      interpret("Roots[x^2 == 4, x] /. Equal[x, r_] :> Equal[z^3, r]").unwrap()
    );
    assert_eq!(
      interpret("Roots[x^3 - 2 x + 1 == 0, x, EquatedTo -> z^3]").unwrap(),
      "z^3 == (-1 - Sqrt[5])/2 || z^3 == (-1 + Sqrt[5])/2 || z^3 == 1"
    );
  }

  #[test]
  fn null_is_the_variable() {
    assert_eq!(
      interpret("Roots[x^2 == 4, x, EquatedTo -> Null]").unwrap(),
      interpret("Roots[x^2 == 4, x]").unwrap()
    );
  }
}

mod positional_and_option_assumptions {
  use super::*;

  #[test]
  fn full_simplify_uses_both() {
    assert_eq!(
      interpret(
        "FullSimplify[Sqrt[x^2] + Sqrt[y^2], x > 0, Assumptions -> y > 0]"
      )
      .unwrap(),
      "x + y"
    );
  }

  #[test]
  fn refine_uses_both() {
    assert_eq!(
      interpret("Refine[Sqrt[x^2] + Sqrt[y^2], x > 0, Assumptions -> y > 0]")
        .unwrap(),
      "x + y"
    );
  }
}

mod dictionary_case {
  use super::*;

  #[test]
  fn proper_nouns_keep_their_capital() {
    assert_eq!(interpret(r#"DictionaryWordQ["paris"]"#).unwrap(), "False");
    assert_eq!(interpret(r#"DictionaryWordQ["Paris"]"#).unwrap(), "True");
    assert_eq!(interpret(r#"DictionaryWordQ["PARIS"]"#).unwrap(), "True");
    assert_eq!(interpret(r#"DictionaryWordQ["hELLo"]"#).unwrap(), "False");
  }

  #[test]
  fn ignore_case() {
    assert_eq!(
      interpret(r#"DictionaryWordQ["paris", IgnoreCase -> True]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"DictionaryWordQ["hELLo", IgnoreCase -> True]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"DictionaryWordQ["paris", IgnoreCase -> False]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn lookup_ignore_case() {
    assert_true(
      r#"MemberQ[DictionaryLookup["pari*", IgnoreCase -> True], "Paris"]"#,
    );
    assert_true(r#"FreeQ[DictionaryLookup["pari*"], "Paris"]"#);
    assert_eq!(
      interpret(
        r#"DictionaryLookup[{"English", "pari*"}, 1, IgnoreCase -> True]"#
      )
      .unwrap(),
      "{pariah}"
    );
  }

  #[test]
  fn languages_without_a_dictionary_stay_unevaluated() {
    assert_eq!(
      interpret(r#"DictionaryWordQ["pomme", Language -> "French"]"#).unwrap(),
      "DictionaryWordQ[pomme, Language -> French]"
    );
  }
}

mod rank_decomposition_modulus {
  use super::*;

  #[test]
  fn singular_modulo_p() {
    // det = -5: rank 1 over GF(5).
    assert_eq!(
      interpret("RankDecomposition[{{1, 2}, {3, 1}}, Modulus -> 5]").unwrap(),
      "{{{1}, {3}}, {{1, 2}}}"
    );
    assert_eq!(
      interpret("RankDecomposition[{{6, 2}, {3, 11}}, Modulus -> 5]").unwrap(),
      "{{{1}, {3}}, {{1, 2}}}"
    );
  }

  #[test]
  fn full_rank_modulo_p() {
    assert_eq!(
      interpret("RankDecomposition[{{1, 2}, {3, 4}}, Modulus -> 5]").unwrap(),
      "{{{1, 2}, {3, 4}}, {{1, 0}, {0, 1}}}"
    );
  }

  #[test]
  fn product_reproduces_the_matrix_modulo_p() {
    assert_true(
      "m = {{2, 4, 1}, {1, 2, 3}, {3, 6, 4}}; \
       {b, c} = RankDecomposition[m, Modulus -> 7]; Mod[b . c - m, 7] == 0 m",
    );
  }
}

mod random_variate_working_precision {
  use super::*;

  #[test]
  fn samples_carry_the_precision() {
    assert_eq!(
      interpret(
        "Precision /@ RandomVariate[ExponentialDistribution[2], 5, \
         WorkingPrecision -> 25]"
      )
      .unwrap(),
      "{25., 25., 25., 25., 25.}"
    );
    assert_eq!(
      interpret(
        "Precision[RandomVariate[ExponentialDistribution[2], \
         WorkingPrecision -> 40]]"
      )
      .unwrap(),
      "40."
    );
  }

  #[test]
  fn array_shapes() {
    assert_eq!(
      interpret(
        "Dimensions[RandomVariate[UniformDistribution[{0, 1}], {2, 3}, \
         WorkingPrecision -> 30]]"
      )
      .unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn samples_lie_in_the_support() {
    assert_true(
      "AllTrue[RandomVariate[UniformDistribution[{2, 3}], 20, \
       WorkingPrecision -> 30], 2 <= # <= 3 &]",
    );
  }

  #[test]
  fn discrete_samples_stay_integers() {
    assert_true(
      "AllTrue[RandomVariate[PoissonDistribution[3], 5, \
       WorkingPrecision -> 20], IntegerQ]",
    );
  }
}

mod quantile_big_rational_level {
  use super::*;

  // Regression: a level whose numerator and denominator exceed machine
  // integers stayed unevaluated.
  #[test]
  fn exponential() {
    assert_true(
      "Abs[N[Quantile[ExponentialDistribution[2], \
       123456789012345678901234567890123456789012345/10^45]] - \
       N[-Log[1 - 1234567890123456789/10^19]/2]] < 10^-12",
    );
  }
}

mod generalized_linear_model_fit_options {
  use super::*;

  const DATA: &str = "{{1, 2}, {2, 3}, {3, 7}, {4, 9}}";

  #[test]
  fn weights_give_the_weighted_least_squares_fit() {
    // Normal equations 6 a + 16 b = 33, 16 a + 50 b = 107.
    assert_true(&format!(
      "Max[Abs[GeneralizedLinearModelFit[{DATA}, x, x, \
       Weights -> {{1, 2, 1, 2}}][\"BestFitParameters\"] - {{-31/22, 57/22}}]] \
       < 10^-10"
    ));
    assert_true(&format!(
      "Max[Abs[GeneralizedLinearModelFit[{DATA}, x, x, \
       Weights -> {{1, 2, 1, 2}}][\"ParameterErrors\"] - \
       LinearModelFit[{DATA}, x, x, Weights -> {{1, 2, 1, 2}}][\
       \"ParameterErrors\"]]] < 10^-10"
    ));
  }

  #[test]
  fn logit_weights_are_repeated_observations() {
    assert_true(
      "Max[Abs[LogitModelFit[{{1, 0}, {2, 0}, {3, 1}, {4, 0}, {5, 1}, \
       {6, 1}}, x, x, Weights -> {1, 2, 1, 2, 1, 2}][\"BestFitParameters\"] - \
       LogitModelFit[{{1, 0}, {2, 0}, {2, 0}, {3, 1}, {4, 0}, {4, 0}, \
       {5, 1}, {6, 1}, {6, 1}}, x, x][\"BestFitParameters\"]]] < 10^-8",
    );
  }

  #[test]
  fn confidence_level() {
    assert_true(&format!(
      "fm = GeneralizedLinearModelFit[{DATA}, x, x, ConfidenceLevel -> 0.9]; \
       z = Quantile[NormalDistribution[], 0.95]; \
       Max[Abs[fm[\"ParameterConfidenceIntervals\"] - \
       Transpose[{{fm[\"BestFitParameters\"] - z fm[\"ParameterErrors\"], \
       fm[\"BestFitParameters\"] + z fm[\"ParameterErrors\"]}}]]] < 10^-10"
    ));
  }

  #[test]
  fn linear_offset_function() {
    // y - 2 x = {0, -1, 1, 1} fitted by a + b x.
    assert_true(&format!(
      "Max[Abs[GeneralizedLinearModelFit[{DATA}, x, x, \
       LinearOffsetFunction -> (2 # &)][\"BestFitParameters\"] - \
       {{-1, 1/2}}]] < 10^-10"
    ));
  }

  #[test]
  fn other_working_precisions_stay_unevaluated() {
    assert_eq!(
      interpret(&format!(
        "Head[GeneralizedLinearModelFit[{DATA}, x, x, WorkingPrecision -> 20]]"
      ))
      .unwrap(),
      "GeneralizedLinearModelFit"
    );
  }
}
