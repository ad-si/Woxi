// Long-tail argument forms found missing by the reference-page parameter
// sweep (round 5). Expected values are the ones the Wolfram reference pages
// show, or follow from the definitions.

use super::*;

mod dms_string {
  use super::*;

  #[test]
  fn dms_list_with_hemisphere_letter() {
    // The letter alone does not print; a direction spec picks the side.
    assert_eq!(
      interpret("DMSString[{27, 12, 3.45, \"S\"}]").unwrap(),
      "27°12'3.450\""
    );
    assert_eq!(
      interpret("DMSString[{27, 12, 3.45, \"S\"}, \"NS\"]").unwrap(),
      "27°12'3.450\"S"
    );
    assert_eq!(
      interpret("DMSString[{27, 12, 3.45, \"S\"}, \"EW\"]").unwrap(),
      "27°12'3.450\"W"
    );
    assert_eq!(
      interpret("DMSString[{27, 12, 3.45, \"N\"}, \"NS\"]").unwrap(),
      "27°12'3.450\"N"
    );
  }

  #[test]
  fn direction_spec_with_precision() {
    assert_eq!(
      interpret("DMSString[-189.34313, {\"EW\", 0}]").unwrap(),
      "189°20'35.\"W"
    );
    assert_eq!(
      interpret("DMSString[189.34313, {\"NS\", 10}]").unwrap(),
      "189°20'35.2680000000\"N"
    );
  }

  #[test]
  fn hemisphere_words_in_strings() {
    assert_eq!(
      interpret("DMSString[\"23d11m49.92s North\"]").unwrap(),
      "23°11'49.920\""
    );
    assert_eq!(
      interpret("DMSString[\"23d11m49.92s South\", \"EW\"]").unwrap(),
      "23°11'49.920\"W"
    );
  }

  #[test]
  fn angle_quantities() {
    assert_eq!(
      interpret("DMSString[Quantity[3.342387, \"Radians\"]]").unwrap(),
      "191°30'16.807\""
    );
    assert_eq!(
      interpret("DMSString[Quantity[-1.342387, \"Radians\"], \"EW\"]").unwrap(),
      "76°54'47.194\"W"
    );
  }
}

mod categorical_distribution {
  use super::*;

  #[test]
  fn uniform_classes() {
    assert_eq!(
      interpret("CategoricalDistribution[{\"A\", \"B\", \"C\"}]").unwrap(),
      "CategoricalDistribution[<|Probabilities -> 1/3, Index -> <|A -> 1, B -> 2, C -> 3|>|>]"
    );
    assert_eq!(
      interpret("CategoricalDistribution[{True, False, x, y}]").unwrap(),
      "CategoricalDistribution[<|Probabilities -> 1/4, Index -> <|True -> 1, False -> 2, x -> 3, y -> 4|>|>]"
    );
  }

  #[test]
  fn weights_and_repeated_classes() {
    assert_eq!(
      interpret("CategoricalDistribution[{\"A\", \"B\", \"C\"}, {1, 2, 3}]")
        .unwrap(),
      "CategoricalDistribution[<|Probabilities -> {1/6, 1/3, 1/2}, Index -> <|A -> 1, B -> 2, C -> 3|>|>]"
    );
    assert_eq!(
      interpret(
        "CategoricalDistribution[{\"A\", \"A\", \"B\", \"A\", \"B\", \"C\", \"A\"}]"
      )
      .unwrap(),
      "CategoricalDistribution[<|Probabilities -> {4/7, 2/7, 1/7}, Index -> <|A -> 1, B -> 2, C -> 3|>|>]"
    );
    assert_eq!(
      interpret("CategoricalDistribution[{\"A\", \"B\", \"C\"}, {a, 2, 3}]")
        .unwrap(),
      "CategoricalDistribution[<|Probabilities -> {a/(5 + a), 2/(5 + a), 3/(5 + a)}, Index -> <|A -> 1, B -> 2, C -> 3|>|>]"
    );
  }

  #[test]
  fn rules_give_sparse_probabilities() {
    assert_eq!(
      interpret(
        "InputForm[CategoricalDistribution[{\"A\" -> 1, \"B\" -> 2, \"C\" -> 3}]]"
      )
      .unwrap(),
      "InputForm[CategoricalDistribution[<|Probabilities -> SparseArray[Automatic, {3}, 0, {1, {{0, 3}, {{1}, {2}, {3}}}, {1/6, 1/3, 1/2}}], Index -> <|A -> 1, B -> 2, C -> 3|>|>]]"
    );
    assert_eq!(
      interpret(
        "InputForm[CategoricalDistribution[{\"A\", \"B\", \"C\"}, {\"A\" -> 1}]]"
      )
      .unwrap(),
      "InputForm[CategoricalDistribution[<|Probabilities -> SparseArray[Automatic, {3}, 0, {1, {{0, 1}, {{1}}}, {1}}], Index -> <|A -> 1, B -> 2, C -> 3|>|>]]"
    );
    assert_eq!(
      interpret(
        "InputForm[CategoricalDistribution[{\"A\", \"B\", \"C\"}, {\"A\" -> 1, _ -> 0.1}]]"
      )
      .unwrap(),
      "InputForm[CategoricalDistribution[<|Probabilities -> SparseArray[Automatic, {3}, 0.08333333333333334, {1, {{0, 1}, {{1}}}, {0.8333333333333334}}], Index -> <|A -> 1, B -> 2, C -> 3|>|>]]"
    );
  }

  #[test]
  fn repeated_rule_keys_stay_unevaluated() {
    assert_eq!(
      interpret("CategoricalDistribution[{a -> 1, a -> 2}]").unwrap(),
      "CategoricalDistribution[{a -> 1, a -> 2}]"
    );
    assert_eq!(
      interpret("CategoricalDistribution[{a, b}, {a -> 1, a -> 2, a -> 3}]")
        .unwrap(),
      "CategoricalDistribution[{a, b}, {a -> 1, a -> 2, a -> 3}]"
    );
  }

  #[test]
  fn multivariate_definitions() {
    assert_eq!(
      interpret(
        "CategoricalDistribution[{{\"A\", \"B\", \"C\"}, {\"1st\", \"2nd\"}}]"
      )
      .unwrap(),
      "CategoricalDistribution[<|Probabilities -> 1/6, Index -> {<|A -> 1, B -> 2, C -> 3|>, <|1st -> 1, 2nd -> 2|>}|>]"
    );
    assert_eq!(
      interpret(
        "CategoricalDistribution[{{\"A\", \"B\", \"C\"}, {\"D\", \"E\"}}, {{1, 2}, {3, 4}, {5, 6}}]"
      )
      .unwrap(),
      "CategoricalDistribution[<|Probabilities -> {{1/21, 2/21}, {1/7, 4/21}, {5/21, 2/7}}, Index -> {<|A -> 1, B -> 2, C -> 3|>, <|D -> 1, E -> 2|>}|>]"
    );
    let sparse = "InputForm[CategoricalDistribution[<|Probabilities -> SparseArray[Automatic, {3, 2}, 0, {1, {{0, 1, 1, 2}, {{1}, {2}}}, {1/2, 1/2}}], Index -> {<|A -> 1, B -> 2, C -> 3|>, <|D -> 1, E -> 2|>}|>]]";
    assert_eq!(
      interpret(
        "InputForm[CategoricalDistribution[{{\"A\", \"B\", \"C\"}, {\"D\", \"E\"}}, SparseArray[{{1, 1} -> 1, {3, 2} -> 1, _ -> 0}]]]"
      )
      .unwrap(),
      sparse
    );
    assert_eq!(
      interpret(
        "InputForm[CategoricalDistribution[{{\"A\", \"B\", \"C\"}, {\"D\", \"E\"}}, {{\"A\", \"D\"} -> 1, {\"C\", \"E\"} -> 1}]]"
      )
      .unwrap(),
      sparse
    );
  }

  #[test]
  fn pdf() {
    assert_eq!(
      interpret("PDF[CategoricalDistribution[{\"A\", \"B\", \"C\"}], \"A\"]")
        .unwrap(),
      "1/3"
    );
    assert_eq!(
      interpret(
        "PDF[CategoricalDistribution[{\"A\", \"B\", \"C\"}, {1, 2, 3}], {\"A\", \"B\", \"C\"}]"
      )
      .unwrap(),
      "{1/6, 1/3, 1/2}"
    );
    assert_eq!(
      interpret(
        "PDF[CategoricalDistribution[{{\"A\", \"B\", \"C\"}, {\"D\", \"E\"}}, {{1, 2}, {3, 4}, {5, 6}}], {\"A\", \"E\"}]"
      )
      .unwrap(),
      "2/21"
    );
    // A symbolic point stays unevaluated until it names a class.
    assert_eq!(
      interpret(
        "pdf = PDF[CategoricalDistribution[{\"A\", \"B\", \"C\"}], x]; {Head[pdf], pdf /. x -> \"A\"}"
      )
      .unwrap(),
      "{PDF, 1/3}"
    );
  }

  #[test]
  fn random_variate() {
    assert_eq!(
      interpret(
        "Union[RandomVariate[CategoricalDistribution[{\"A\", \"B\", \"C\"}, {1, 2, 3}], 200]]"
      )
      .unwrap(),
      "{A, B, C}"
    );
    assert_eq!(
      interpret(
        "Dimensions[RandomVariate[CategoricalDistribution[{{\"A\", \"B\", \"C\"}, {\"1st\", \"2nd\"}}], 6]]"
      )
      .unwrap(),
      "{6, 2}"
    );
    // Symbolic weights cannot be sampled.
    assert_eq!(
      interpret(
        "Head[RandomVariate[CategoricalDistribution[{\"A\", \"B\", \"C\"}, {a, 2, 3}]]]"
      )
      .unwrap(),
      "RandomVariate"
    );
  }

  #[test]
  fn marginal_distribution() {
    assert_eq!(
      interpret(
        "MarginalDistribution[CategoricalDistribution[{{\"A\", \"B\", \"C\"}, {\"D\", \"E\"}}, {{1, 2}, {3, 4}, {5, 6}}], 1]"
      )
      .unwrap(),
      "CategoricalDistribution[<|Probabilities -> {1/7, 1/3, 11/21}, Index -> <|A -> 1, B -> 2, C -> 3|>|>]"
    );
  }

  #[test]
  fn information_properties() {
    let cd = "cd = CategoricalDistribution[{\"A\", \"B\", \"C\"}, {1, 2, 3}];";
    assert_eq!(
      interpret(&format!("{cd} Information[cd, \"Categories\"]")).unwrap(),
      "{A, B, C}"
    );
    assert_eq!(
      interpret(&format!("{cd} Information[cd, \"Probabilities\"]")).unwrap(),
      "<|A -> 1/6, B -> 1/3, C -> 1/2|>"
    );
    assert_eq!(
      interpret(&format!("{cd} Information[cd, \"Entropy\"]")).unwrap(),
      "Log[2]/2 + Log[3]/3 + Log[6]/6"
    );
    assert_eq!(
      interpret(&format!("{cd} Information[cd, \"Properties\"]")).unwrap(),
      "{Properties, Entropy, NEntropy, Categories, DomainSize, DomainElements, Probabilities, ProbabilityTable, ProbabilityArray, ProbabilityPlot, TopProbabilities, Dimension}"
    );
    let cd2 = "cd = CategoricalDistribution[{{\"A\", \"B\", \"C\"}, {\"D\", \"E\"}}, {{1, 2}, {3, 4}, {5, 6}}];";
    assert_eq!(
      interpret(&format!("{cd2} Information[cd, \"Categories\"]")).unwrap(),
      "{{A, B, C}, {D, E}}"
    );
    assert_eq!(
      interpret(&format!("{cd2} Information[cd, \"DomainElements\"]")).unwrap(),
      "{{A, D}, {A, E}, {B, D}, {B, E}, {C, D}, {C, E}}"
    );
    assert_eq!(
      interpret(&format!("{cd2} Information[cd, \"Probabilities\"]")).unwrap(),
      "<|{A, D} -> 1/21, {A, E} -> 2/21, {B, D} -> 1/7, {B, E} -> 4/21, {C, D} -> 5/21, {C, E} -> 2/7|>"
    );
    assert_eq!(
      interpret(&format!("{cd2} Information[cd, \"TopProbabilities\" -> 2]"))
        .unwrap(),
      "{{C, E} -> 2/7, {C, D} -> 5/21}"
    );
    assert_eq!(
      interpret(&format!("{cd2} Information[cd, \"Entropy\"]")).unwrap(),
      "(2*Log[7/2])/7 + (5*Log[21/5])/21 + (4*Log[21/4])/21 + Log[7]/7 + (2*Log[21/2])/21 + Log[21]/21"
    );
  }
}

mod extended_distributions {
  use super::*;

  #[test]
  fn scaled_inverse_chi_square() {
    assert_eq!(
      interpret("PDF[InverseChiSquareDistribution[ν, ξ], x]").unwrap(),
      "Piecewise[{{((ν*ξ)/x)^(ν/2)/(2^(ν/2)*E^((ν*ξ)/(2*x))*x*Gamma[ν/2]), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[InverseChiSquareDistribution[ν, ξ], x]").unwrap(),
      "Piecewise[{{GammaRegularized[ν/2, (ν*ξ)/(2*x)], x > 0}}, 0]"
    );
    assert_eq!(
      interpret(
        "{Mean[InverseChiSquareDistribution[ν]], Mean[InverseChiSquareDistribution[ν, ξ]]}"
      )
      .unwrap(),
      "{Piecewise[{{(-2 + ν)^(-1), ν > 2}}, Indeterminate], Piecewise[{{(ν*ξ)/(-2 + ν), ν > 2}}, Indeterminate]}"
    );
    assert_eq!(
      interpret("Variance[InverseChiSquareDistribution[ν, ξ]]").unwrap(),
      "Piecewise[{{(2*ν^2*ξ^2)/((-4 + ν)*(-2 + ν)^2), ν > 4}}, Indeterminate]"
    );
    assert_eq!(
      interpret("Variance[InverseChiSquareDistribution[ν]]").unwrap(),
      "Piecewise[{{2/((-4 + ν)*(-2 + ν)^2), ν > 4}}, Indeterminate]"
    );
    assert_eq!(
      interpret("Median[InverseChiSquareDistribution[ν, ξ]]").unwrap(),
      "(ν*ξ)/(2*InverseGammaRegularized[ν/2, 1/2])"
    );
    assert_eq!(
      interpret("Quantile[InverseChiSquareDistribution[ν, ξ], q]").unwrap(),
      "ConditionalExpression[Piecewise[{{(ν*ξ)/(2*InverseGammaRegularized[ν/2, q]), 0 < q < 1}, {0, q <= 0}}, Infinity], 0 <= q <= 1]"
    );
    assert_eq!(
      interpret("Mean[InverseChiSquareDistribution[53.2, 1.04]]").unwrap(),
      "1.080625"
    );
  }

  #[test]
  fn generalized_gamma() {
    assert_eq!(
      interpret("PDF[GammaDistribution[α, β, γ, μ], x]").unwrap(),
      "Piecewise[{{(γ*((x - μ)/β)^(-1 + α*γ))/(E^((x - μ)/β)^γ*β*Gamma[α]), x > μ}}, 0]"
    );
    assert_eq!(
      interpret("CDF[GammaDistribution[α, β, γ, μ], x]").unwrap(),
      "Piecewise[{{GammaRegularized[α, 0, ((x - μ)/β)^γ], x > μ}}, 0]"
    );
    assert_eq!(
      interpret("Mean[GammaDistribution[α, β, γ, μ]]").unwrap(),
      "μ + (β*Gamma[α + γ^(-1)])/Gamma[α]"
    );
    assert_eq!(
      interpret("Variance[GammaDistribution[α, β, γ, μ]]").unwrap(),
      "(β^2*(-Gamma[α + γ^(-1)]^2 + Gamma[α]*Gamma[α + 2/γ]))/Gamma[α]^2"
    );
    assert_eq!(
      interpret("Median[GammaDistribution[α, β, γ, μ]]").unwrap(),
      "μ + β*InverseGammaRegularized[α, 1/2]^γ^(-1)"
    );
    assert_eq!(
      interpret("Quantile[GammaDistribution[α, β, γ, μ], q]").unwrap(),
      "ConditionalExpression[Piecewise[{{μ + β*InverseGammaRegularized[α, 0, q]^γ^(-1), 0 < q < 1}, {μ, q <= 0}}, Infinity], 0 <= q <= 1]"
    );
    // Special cases: γ = 1, μ = 0 is the ordinary gamma distribution and
    // α = 1 a Weibull distribution.
    assert_eq!(
      interpret("PDF[GammaDistribution[α, β, 1, 0], x]").unwrap(),
      "Piecewise[{{(x/β)^(-1 + α)/(E^(x/β)*β*Gamma[α]), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[GammaDistribution[1, β, γ, μ], x]").unwrap(),
      "Piecewise[{{(γ*((x - μ)/β)^(-1 + γ))/(E^((x - μ)/β)^γ*β), x > μ}}, 0]"
    );
  }

  #[test]
  fn generalized_inverse_gamma() {
    assert_eq!(
      interpret("PDF[InverseGammaDistribution[α, β, γ, μ], x]").unwrap(),
      "Piecewise[{{(γ*(β/(x - μ))^(1 + α*γ))/(E^(β/(x - μ))^γ*β*Gamma[α]), x > μ}}, 0]"
    );
    assert_eq!(
      interpret("CDF[InverseGammaDistribution[α, β, γ, μ], x]").unwrap(),
      "Piecewise[{{GammaRegularized[α, (β/(x - μ))^γ], x > μ}}, 0]"
    );
    assert_eq!(
      interpret("Mean[InverseGammaDistribution[α, β, γ, μ]]").unwrap(),
      "Piecewise[{{μ + (β*Gamma[α - γ^(-1)])/Gamma[α], α*γ > 1}}, Indeterminate]"
    );
    assert_eq!(
      interpret("Variance[InverseGammaDistribution[α, β, γ, μ]]").unwrap(),
      "Piecewise[{{(β^2*(Gamma[α]*Gamma[α - 2/γ] - Gamma[α - γ^(-1)]^2))/Gamma[α]^2, α*γ > 2}}, Indeterminate]"
    );
    assert_eq!(
      interpret("CDF[InverseGammaDistribution[α, β, 1, 0], x]").unwrap(),
      "Piecewise[{{GammaRegularized[α, β/x], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("Quantile[InverseGammaDistribution[α, β, γ, μ], q]").unwrap(),
      "ConditionalExpression[Piecewise[{{μ + β/InverseGammaRegularized[α, q]^γ^(-1), 0 < q < 1}, {μ, q <= 0}}, Infinity], 0 <= q <= 1]"
    );
  }

  #[test]
  fn four_parameter_gompertz_makeham() {
    assert_eq!(
      interpret("PDF[GompertzMakehamDistribution[λ, ξ, θ, α], t]").unwrap(),
      "Piecewise[{{(λ*(E^(t*λ) + θ + 2*t*α*λ)*ξ)/E^((-1 + E^(t*λ) + t*θ*λ + t^2*α*λ^2)*ξ), t >= 0}}, 0]"
    );
    // The CDF is checked against its closed form (the printed term order of
    // the exponent differs; see conformance_gaps.md).
    assert_eq!(
      interpret(
        "CDF[GompertzMakehamDistribution[2, 1, 3, 2], 1/2] == 1 - E^((1 - E)*1 - 3 - 2) // Simplify"
      )
      .unwrap(),
      "True"
    );
    // Moments have no closed form and stay unevaluated.
    assert_eq!(
      interpret("Mean[GompertzMakehamDistribution[E, 1, 3, 2]]").unwrap(),
      "Mean[GompertzMakehamDistribution[E, 1, 3, 2]]"
    );
  }

  #[test]
  fn generalized_inverse_gaussian() {
    assert_eq!(
      interpret("PDF[InverseGaussianDistribution[μ, λ, θ], x]").unwrap(),
      "Piecewise[{{x^(-1 + θ)/(2*E^((λ*(1 + x^2/μ^2))/(2*x))*μ^θ*BesselK[θ, λ/μ]), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("Mean[InverseGaussianDistribution[μ, λ, θ]]").unwrap(),
      "(μ*BesselK[1 + θ, λ/μ])/BesselK[θ, λ/μ]"
    );
    assert_eq!(
      interpret("Variance[InverseGaussianDistribution[μ, λ, θ]]").unwrap(),
      "(μ^2*(-BesselK[1 + θ, λ/μ]^2 + BesselK[θ, λ/μ]*BesselK[2 + θ, λ/μ]))/BesselK[θ, λ/μ]^2"
    );
  }

  #[test]
  fn random_variates_match_the_means() {
    for (dist, tol) in [
      ("GammaDistribution[2, 3, 2, 1]", "0.05"),
      ("InverseGammaDistribution[3, 2, 2, 1]", "0.05"),
      ("InverseChiSquareDistribution[13, 2]", "0.05"),
      ("InverseGaussianDistribution[1, 5, 2]", "0.05"),
      ("InverseGaussianDistribution[3, 0.5, -2]", "0.05"),
    ] {
      assert_eq!(
        interpret(&format!(
          "Abs[Mean[RandomVariate[{dist}, 40000]] - N[Mean[{dist}]]] < {tol}"
        ))
        .unwrap(),
        "True",
        "{dist}"
      );
    }
    // GompertzMakehamDistribution[E, 1, 3, 2] has mean 0.0748474.
    assert_eq!(
      interpret(
        "Abs[Mean[RandomVariate[GompertzMakehamDistribution[E, 1, 3, 2], 40000]] - 0.0748474] < 0.003"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "Min[RandomVariate[InverseGammaDistribution[0.5, 3, 2, 1], 1000]] > 1"
      )
      .unwrap(),
      "True"
    );
  }
}

mod lu_decomposition_rectangular {
  use super::*;

  #[test]
  fn wide_identity() {
    assert_eq!(
      interpret("Normal /@ Most[LUDecomposition[IdentityMatrix[{3, 4}]]]")
        .unwrap(),
      "{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}"
    );
    assert_eq!(
      interpret("Last[LUDecomposition[IdentityMatrix[{3, 4}]]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn wide_and_tall_factor_the_matrix() {
    // l is square with as many rows as m; u has m's shape; l.u == p.m.
    for m in [
      "{{1, 2, 3}, {4, 5, 7}}",
      "{{1, 2}, {3, 4}, {5, 7}}",
      "{{0, 1, 2}, {3, 4, 5}}",
    ] {
      assert_eq!(
        interpret(&format!(
          "m = {m}; {{l, u, p, c}} = Normal /@ LUDecomposition[m]; {{Dimensions[l] == {{Length[m], Length[m]}}, Dimensions[u] == Dimensions[m], l.u == p.m}}"
        ))
        .unwrap(),
        "{True, True, True}",
        "{m}"
      );
    }
  }
}

mod find_divisions_base_and_unit {
  use super::*;

  #[test]
  fn base() {
    assert_eq!(
      interpret("FindDivisions[{0, 1}, 5, 2]").unwrap(),
      "{0, 1/4, 1/2, 3/4, 1}"
    );
    assert_eq!(
      interpret("N[FindDivisions[{0, 1}, 5, 3]]").unwrap(),
      "{0., 0.3333333333333333, 0.6666666666666666, 1.}"
    );
    assert_eq!(
      interpret("FindDivisions[{0, 1}, 5, 10]").unwrap(),
      "{0, 1/5, 2/5, 3/5, 4/5, 1}"
    );
  }

  #[test]
  fn symbolic_unit() {
    assert_eq!(
      interpret("FindDivisions[{0, 10, Pi}, 5]").unwrap(),
      "{0, Pi, 2*Pi, 3*Pi, 4*Pi}"
    );
  }
}

mod insert_into_associations {
  use super::*;

  #[test]
  fn list_of_rules() {
    assert_eq!(
      interpret("Insert[<|a -> 1, b -> 2|>, {c -> 3, d -> 4}, Key[a]]")
        .unwrap(),
      "<|c -> 3, d -> 4, a -> 1, b -> 2|>"
    );
  }

  #[test]
  fn existing_key_is_replaced() {
    assert_eq!(
      interpret("Insert[<|a -> 1, b -> 2|>, a -> 3, Key[a]]").unwrap(),
      "<|a -> 3, b -> 2|>"
    );
  }

  #[test]
  fn integer_path_into_a_value() {
    assert_eq!(
      interpret("Insert[<|1 -> {a, b}, 2 -> {c, d}|>, x, {2, 1}]").unwrap(),
      "<|1 -> {a, b}, 2 -> {x, c, d}|>"
    );
  }

  #[test]
  fn several_positions_through_keys() {
    assert_eq!(
      interpret(
        "expr = {1, two, <|a -> {3}|>}; Insert[expr, \"an integer follows\", Position[expr, _Integer]]"
      )
      .unwrap(),
      "{an integer follows, 1, two, <|a -> {an integer follows, 3}|>}"
    );
  }

  #[test]
  fn non_rule_is_rejected() {
    let r =
      interpret_with_stdout("Insert[<|a -> 1, b -> 2|>, foo, Key[a]]").unwrap();
    assert_eq!(r.result, "<|a -> 1, b -> 2|>");
    assert!(r.warnings.iter().any(|w| w.contains("Insert::invdt")));
  }
}

mod peaks_at_scale_zero {
  use super::*;

  #[test]
  fn peak_detect_sharpness_and_threshold() {
    assert_eq!(
      interpret(
        "PeakDetect[{2, 1, 3, 5, 6, 6, 4, 3, 2, 3, 7, 2, 3, 4, 3, 2}, 0, 0]"
      )
      .unwrap(),
      "{1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0}"
    );
    assert_eq!(
      interpret(
        "PeakDetect[{2, 1, 3, 5, 6, 6, 4, 3, 2, 3, 7, 2, 3, 4, 3, 2}, 0, 3]"
      )
      .unwrap(),
      "{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}"
    );
    assert_eq!(
      interpret(
        "PeakDetect[{2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 1, 4, 2, 2, 1}, 0, 0, 5.1]"
      )
      .unwrap(),
      "{0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}"
    );
  }

  #[test]
  fn constant_padding() {
    assert_eq!(
      interpret(
        "PeakDetect[{2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 4, 2, 2, 1}, 0, Padding -> 4]"
      )
      .unwrap(),
      "{0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0}"
    );
    assert_eq!(
      interpret(
        "FindPeaks[{2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 4, 2, 2, 1}, 0, Padding -> 4]"
      )
      .unwrap(),
      "{{11/2, 6}, {11, 7}, {13, 4}}"
    );
  }

  #[test]
  fn unsupported_options_stay_unevaluated() {
    assert_eq!(
      interpret(
        "FindPeaks[{2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3}, 0, InterpolationOrder -> 3]"
      )
      .unwrap(),
      "FindPeaks[{2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3}, 0, InterpolationOrder -> 3]"
    );
  }
}

mod regions {
  use super::*;

  #[test]
  fn centroid_of_point_sets_and_spheres() {
    assert_eq!(
      interpret("RegionCentroid[Point[Tuples[Range[5], 2]]]").unwrap(),
      "{3, 3}"
    );
    assert_eq!(
      interpret("RegionCentroid[Point[Tuples[Range[5], 4]]]").unwrap(),
      "{3, 3, 3, 3}"
    );
    assert_eq!(
      interpret("RegionCentroid[Sphere[{1, 2, 3}]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn bounds_of_tetrahedra_and_flat_regions() {
    assert_eq!(
      interpret(
        "RegionBounds[Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]"
      )
      .unwrap(),
      "{{0, 1}, {0, 1}, {0, 1}}"
    );
    assert_eq!(
      interpret(
        "RegionBounds[InfinitePlane[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]]"
      )
      .unwrap(),
      "{{-Infinity, Infinity}, {-Infinity, Infinity}, {0, 0}}"
    );
    assert_eq!(
      interpret("RegionBounds[InfiniteLine[{{1, 0}, {1, 1}}]]").unwrap(),
      "{{1, 1}, {-Infinity, Infinity}}"
    );
  }

  #[test]
  fn bounds_types() {
    assert_eq!(
      interpret("RegionBounds[InfiniteLine[{0, 0}, {1, 1}], \"Finite\"]")
        .unwrap(),
      "{{0, 1}, {0, 1}}"
    );
    assert_eq!(
      interpret("RegionBounds[InfiniteLine[{0, 0}, {1, 1}], \"Minimal\"]")
        .unwrap(),
      "{{-Infinity, Infinity}, {-Infinity, Infinity}}"
    );
    assert_eq!(
      interpret("RegionBounds[InfiniteLine[{0, 0}, {1, 1}], \"Sufficient\"]")
        .unwrap(),
      "{{-Infinity, Infinity}, {-Infinity, Infinity}}"
    );
  }
}

mod list_interpolation_coordinates {
  use super::*;

  #[test]
  fn explicit_coordinate_list() {
    // The same as Interpolation of the {x, f} pairs.
    assert_eq!(
      interpret(
        "f = ListInterpolation[{0, .3, .6, -.2, 3}, {{0, .1, .5, 1, 2}}]; g = Interpolation[{{0, 0}, {.1, .3}, {.5, .6}, {1, -.2}, {2, 3}}]; {f[0.5], f[0.3] == g[0.3], f[1.5] == g[1.5]}"
      )
      .unwrap(),
      "{0.6, True, True}"
    );
  }
}
