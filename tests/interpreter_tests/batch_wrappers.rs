use super::*;

mod batch_unevaluated_wrappers {
  use super::*;

  #[test]
  fn tilde_tilde() {
    assert_eq!(interpret("TildeTilde[x]").unwrap(), "TildeTilde[x]");
  }
  #[test]
  fn notebook_close() {
    assert_eq!(interpret("NotebookClose[x]").unwrap(), "NotebookClose[x]");
  }
  #[test]
  fn failure() {
    assert_eq!(interpret("Failure[x]").unwrap(), "Failure[x]");
  }
  #[test]
  fn time_value() {
    assert_eq!(interpret("TimeValue[x]").unwrap(), "TimeValue[x]");
  }
  #[test]
  fn time_value_future_value() {
    // TimeValue[s, i, t] = s * (1 + i)^t. Future value of $1000 at 5%
    // for 3 periods is 1157.625.
    let result = interpret("TimeValue[1000, 0.05, 3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1157.6250000000002).abs() < 1e-9, "got {}", val);
  }
  #[test]
  fn time_value_present_value() {
    // Negative t for present value: 1000 / 1.05^3.
    let result = interpret("TimeValue[1000, 0.05, -3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 863.837598531476).abs() < 1e-9, "got {}", val);
  }
  #[test]
  fn time_value_exact_rational() {
    // Exact rational inputs preserve rational arithmetic.
    assert_eq!(interpret("TimeValue[100, 1/10, 2]").unwrap(), "121");
  }
  #[test]
  fn time_value_list_of_rates_exact_match() {
    // Five rates over five periods: 1000 * Prod(1 + r_i).
    let result =
      interpret("TimeValue[1000, {0.04, 0.05, 0.06, 0.07, 0.08}, 5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1337.6301120000003).abs() < 1e-9, "got {}", val);
  }
  #[test]
  fn time_value_list_of_rates_extends_with_last() {
    // Fewer rates than periods: the last rate is repeated.
    // 1000 * 1.04 * 1.05 * 1.05 * 1.05 * 1.05 = 1264.1265.
    let result = interpret("TimeValue[1000, {0.04, 0.05}, 5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1264.1265).abs() < 1e-9, "got {}", val);
  }
  #[test]
  fn time_value_list_of_rates_truncates() {
    // More rates than periods: only the first t are used.
    let result = interpret("TimeValue[1000, {0.04, 0.05, 0.06}, 2]").unwrap();
    let val: f64 = result.parse().unwrap();
    // 1000 * 1.04 * 1.05 = 1092.
    assert!((val - 1092.0).abs() < 1e-9, "got {}", val);
  }
  #[test]
  fn time_value_list_of_rates_zero_t() {
    // t = 0 → original principal unchanged.
    assert_eq!(
      interpret("TimeValue[1000, {0.04, 0.05}, 0]").unwrap(),
      "1000"
    );
  }

  #[test]
  fn time_value_annuity_pv() {
    // Audit case: PV of an annuity that pays 100 for 12 periods at 6%.
    // PMT * (1 - (1+r)^-n) / r = 100 * (1 - 1.06^-12)/0.06 ≈ 838.384...
    let result = interpret("TimeValue[Annuity[100, 12], 0.06, 0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 838.384394038332).abs() < 1e-9, "got {}", val);
  }

  #[test]
  fn time_value_annuity_fv() {
    // FV at end of term (t=n): PMT * ((1+r)^n - 1) / r.
    let result = interpret("TimeValue[Annuity[100, 12], 0.06, 12]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1686.994119725919).abs() < 1e-9, "got {}", val);
  }

  #[test]
  fn time_value_cashflow_at_future() {
    // Audit case: Cashflow {0,100,200,450,300,580} valued at t=7 at 6%.
    let result =
      interpret("TimeValue[Cashflow[{0, 100, 200, 450, 300, 580}], 0.06, 7]")
        .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1986.6044587456004).abs() < 1e-7, "got {}", val);
  }

  #[test]
  fn time_value_date_list() {
    // Audit case: 1000 from 2010 to 2013 at 7.5% → 1000 * 1.075^3.
    let result =
      interpret("TimeValue[1000, 0.075, {{2013, 1, 1}, {2010, 1, 1}}]")
        .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1242.2968749999998).abs() < 1e-9, "got {}", val);
  }

  #[test]
  fn time_value_term_pairs_audit_case() {
    // Audit case: period-rate pairs with t = -4.
    // Result = 1000 / ((1+0.04)*(1+0.05)*(1+0.06)*(1+0.07)) = 807.398...
    let result = interpret(
      "TimeValue[1000, {{-4, 0.04}, {-3, 0.05}, {-2, 0.06}, {-1, 0.07}}, -4]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 807.3980918276454).abs() < 1e-7, "got {}", val);
  }

  #[test]
  fn time_value_term_pairs_partial() {
    // Same pairs, t = -2: only the two rates for k >= -2 are applied.
    // 1000 / (1.06 * 1.07).
    let result = interpret(
      "TimeValue[1000, {{-4, 0.04}, {-3, 0.05}, {-2, 0.06}, {-1, 0.07}}, -2]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 881.6787162757889).abs() < 1e-7, "got {}", val);
  }

  #[test]
  fn time_value_yield_curve_at_exact_node() {
    // Audit case: rule list as yield curve, {0, 10} discounts by (1+r_10)^10.
    let result = interpret(
      "TimeValue[1000, {1 -> 0.02, 2 -> 0.025, 5 -> 0.04, 10 -> 0.055}, {0, 10}]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 585.4305794276072).abs() < 1e-7, "got {}", val);
  }

  #[test]
  fn time_value_yield_curve_interpolated() {
    // T=3 isn't a node: linearly interpolate between t=2 and t=5.
    // r(3) = 0.025 + 1/3*(0.04 - 0.025) = 0.030; result = 1000/1.030^3.
    let result = interpret(
      "TimeValue[1000, {1 -> 0.02, 2 -> 0.025, 5 -> 0.04, 10 -> 0.055}, {0, 3}]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 915.1416593531594).abs() < 1e-7, "got {}", val);
  }

  #[test]
  fn time_value_via_effective_interest() {
    // TimeValue[1000, EffectiveInterest[0.05, 1/4], 10]
    // = 1000 * (1 + (1 + 0.05*0.25)^4 - 1)^10
    // = 1000 * 1.0125^40 ≈ 1643.6194634870124.
    let result =
      interpret("TimeValue[1000, EffectiveInterest[0.05, 1/4], 10]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1643.6194634870124).abs() < 1e-9, "got {}", val);
  }
  #[test]
  fn effective_interest_quarterly() {
    // Compounded 4 times per year (period = 1/4).
    let result = interpret("EffectiveInterest[0.05, 1/4]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.050945336914062445).abs() < 1e-12, "got {}", val);
  }
  #[test]
  fn effective_interest_monthly() {
    // 12 compounding periods per year → period = 1/12.
    let result = interpret("EffectiveInterest[0.05, 1/12]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.0511618978817332).abs() < 1e-12, "got {}", val);
  }
  #[test]
  fn effective_interest_annual() {
    // Annual compounding leaves the rate unchanged.
    let result = interpret("EffectiveInterest[0.05, 1]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.05).abs() < 1e-12, "got {}", val);
  }
  #[test]
  fn effective_interest_continuous() {
    // Period 0 → continuous compounding: e^r - 1.
    let result = interpret("EffectiveInterest[0.05, 0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.05127109637602412).abs() < 1e-12, "got {}", val);
  }
  #[test]
  fn effective_interest_symbolic() {
    // Symbolic form: -1 + (1 + p*r)^(1/p).
    assert_eq!(
      interpret("EffectiveInterest[r, p]").unwrap(),
      "-1 + (1 + p*r)^p^(-1)"
    );
  }
  #[test]
  fn effective_interest_zero_symbolic() {
    // Period 0 with symbolic rate: e^r - 1.
    assert_eq!(interpret("EffectiveInterest[r, 0]").unwrap(), "-1 + E^r");
  }
  #[test]
  fn effective_interest_listable_rate() {
    // First argument threads over a list of rates.
    let result =
      interpret("EffectiveInterest[{0.04, 0.05, 0.06}, 1/2]").unwrap();
    assert!(result.starts_with("{"), "got {}", result);
    let stripped = result.trim_start_matches('{').trim_end_matches('}');
    let vals: Vec<f64> = stripped
      .split(',')
      .map(|s| s.trim().parse().unwrap())
      .collect();
    assert!((vals[0] - 0.0404).abs() < 1e-6);
    assert!((vals[1] - 0.050625).abs() < 1e-6);
    assert!((vals[2] - 0.0609).abs() < 1e-6);
  }
  #[test]
  fn line_indent() {
    assert_eq!(interpret("LineIndent[x]").unwrap(), "LineIndent[x]");
  }
  #[test]
  fn layered_graph_plot() {
    assert_eq!(
      interpret("LayeredGraphPlot[x]").unwrap(),
      "LayeredGraphPlot[x]"
    );
  }
  #[test]
  fn word_character() {
    assert_eq!(interpret("WordCharacter[x]").unwrap(), "WordCharacter[x]");
  }
  #[test]
  fn reflection_transform() {
    assert_eq!(
      interpret("ReflectionTransform[x]").unwrap(),
      "ReflectionTransform[x]"
    );
  }
  #[test]
  fn bspline_basis() {
    assert_eq!(
      interpret("BSplineBasis[x, y]").unwrap(),
      "BSplineBasis[x, y]"
    );
  }

  #[test]
  fn bspline_basis_cubic_at_half() {
    // BSplineBasis[d, x] evaluates the zeroth uniform B-spline of
    // degree d at x ∈ [0, 1]. For d = 3 the peak is at x = 1/2 with
    // value 2/3 ≈ 0.6666666666666666 (matches wolframscript).
    let result = interpret("BSplineBasis[3, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.0 / 3.0).abs() < 1e-12, "got {}", val);
  }

  #[test]
  fn bspline_basis_quadratic_at_half() {
    // BSplineBasis[2, 0.5] = 3/4 = 0.75.
    let result = interpret("BSplineBasis[2, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.75).abs() < 1e-12, "got {}", val);
  }

  #[test]
  fn bspline_basis_linear() {
    // d = 1: triangle peaked at 1/2 with value 1.
    let result = interpret("BSplineBasis[1, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.0).abs() < 1e-12, "got {}", val);
    let result = interpret("BSplineBasis[1, 0.25]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.5).abs() < 1e-12, "got {}", val);
  }

  #[test]
  fn bspline_basis_zero_degree() {
    // d = 0: indicator of [0, 1).
    let result = interpret("BSplineBasis[0, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.0).abs() < 1e-12, "got {}", val);
  }

  #[test]
  fn bspline_basis_outside_support() {
    // x outside [0, 1] gives 0.
    assert_eq!(interpret("BSplineBasis[3, -0.5]").unwrap(), "0");
    assert_eq!(interpret("BSplineBasis[3, 1.5]").unwrap(), "0");
  }
  #[test]
  fn parameter_mixture_distribution() {
    assert_eq!(
      interpret("ParameterMixtureDistribution[x, y]").unwrap(),
      "ParameterMixtureDistribution[x, y]"
    );
  }
  #[test]
  fn binary_read_list() {
    assert_eq!(
      interpret("BinaryReadList[x, y]").unwrap(),
      "BinaryReadList[x, y]"
    );
  }
  #[test]
  fn total_layer() {
    assert_eq!(interpret("TotalLayer[x]").unwrap(), "$Failed");
  }
  #[test]
  fn find_distribution_parameters() {
    assert_eq!(
      interpret("FindDistributionParameters[x, y]").unwrap(),
      "FindDistributionParameters[x, y]"
    );
  }
  #[test]
  fn find_path() {
    assert_eq!(interpret("FindPath[x, y, z]").unwrap(), "FindPath[x, y, z]");
  }
  #[test]
  fn find_peaks() {
    assert_eq!(interpret("FindPeaks[x]").unwrap(), "FindPeaks[x]");
  }
  #[test]
  fn nprobability() {
    assert_eq!(
      interpret("NProbability[x, y]").unwrap(),
      "NProbability[x, y]"
    );
  }
  #[test]
  fn net_encoder() {
    assert_eq!(interpret("NetEncoder[x]").unwrap(), "$Failed");
  }
  #[test]
  fn permutation_product() {
    assert_eq!(interpret("PermutationProduct[x]").unwrap(), "x");
  }
  #[test]
  fn syntax_information() {
    assert_eq!(interpret("SyntaxInformation[x]").unwrap(), "{}");
  }
}

mod batch_unevaluated_wrappers_2 {
  use super::*;

  #[test]
  fn dedekind_eta() {
    assert_eq!(interpret("DedekindEta[x]").unwrap(), "DedekindEta[x]");
  }
  #[test]
  fn pixel_value_positions() {
    assert_eq!(
      interpret("PixelValuePositions[x, y]").unwrap(),
      "PixelValuePositions[x, y]"
    );
  }
  #[test]
  fn weights() {
    assert_eq!(interpret("Weights[x]").unwrap(), "Weights[x]");
  }
  #[test]
  fn whitespace_character() {
    assert_eq!(
      interpret("WhitespaceCharacter[x]").unwrap(),
      "WhitespaceCharacter[x]"
    );
  }
  #[test]
  fn bar_chart_3d() {
    assert_eq!(interpret("BarChart3D[x]").unwrap(), "BarChart3D[x]");
  }
  #[test]
  fn vertical_slider() {
    assert_eq!(interpret("VerticalSlider[x]").unwrap(), "VerticalSlider[x]");
  }
  #[test]
  fn cycle_graph() {
    assert_eq!(interpret("CycleGraph[x]").unwrap(), "CycleGraph[x]");
  }
  #[test]
  fn over_dot() {
    assert_eq!(interpret("OverDot[x]").unwrap(), "OverDot[x]");
  }
  #[test]
  fn max_plot_points() {
    assert_eq!(interpret("MaxPlotPoints[x]").unwrap(), "MaxPlotPoints[x]");
  }
  #[test]
  fn launch_kernels() {
    assert_eq!(interpret("LaunchKernels[x]").unwrap(), "{}");
  }
  #[test]
  fn permutation_cycles() {
    assert_eq!(
      interpret("PermutationCycles[x]").unwrap(),
      "PermutationCycles[x]"
    );
  }
  #[test]
  fn animation_repetitions() {
    assert_eq!(
      interpret("AnimationRepetitions[x]").unwrap(),
      "AnimationRepetitions[x]"
    );
  }
  #[test]
  fn arma_process() {
    assert_eq!(interpret("ARMAProcess[x]").unwrap(), "ARMAProcess[x]");
  }
  #[test]
  fn file_name_take() {
    assert_eq!(interpret("FileNameTake[x]").unwrap(), "FileNameTake[x]");
  }
  #[test]
  fn undo_tracked_variables() {
    assert_eq!(
      interpret("UndoTrackedVariables[x]").unwrap(),
      "UndoTrackedVariables[x]"
    );
  }
  #[test]
  fn vector_color_function() {
    assert_eq!(
      interpret("VectorColorFunction[x]").unwrap(),
      "VectorColorFunction[x]"
    );
  }
  #[test]
  fn notebook_get() {
    assert_eq!(interpret("NotebookGet[x]").unwrap(), "NotebookGet[x]");
  }
  #[test]
  fn visible() {
    assert_eq!(interpret("Visible[x]").unwrap(), "Visible[x]");
  }
  #[test]
  fn truncated_distribution() {
    assert_eq!(
      interpret("TruncatedDistribution[x, y]").unwrap(),
      "TruncatedDistribution[x, y]"
    );
  }

  // ─── DiscreteUniformDistribution ───────────────────────────────────
  #[test]
  fn discrete_uniform_distribution_pdf() {
    assert_eq!(
      interpret("PDF[DiscreteUniformDistribution[{1, 10}], 5]").unwrap(),
      "1/10"
    );
  }
  #[test]
  fn discrete_uniform_distribution_pdf_outside() {
    assert_eq!(
      interpret("PDF[DiscreteUniformDistribution[{1, 10}], 0]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("PDF[DiscreteUniformDistribution[{1, 10}], 11]").unwrap(),
      "0"
    );
  }
  #[test]
  fn discrete_uniform_distribution_cdf() {
    assert_eq!(
      interpret("CDF[DiscreteUniformDistribution[{1, 10}], 5]").unwrap(),
      "1/2"
    );
  }
  #[test]
  fn discrete_uniform_distribution_cdf_edges() {
    assert_eq!(
      interpret("CDF[DiscreteUniformDistribution[{1, 10}], 0]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("CDF[DiscreteUniformDistribution[{1, 10}], 10]").unwrap(),
      "1"
    );
  }
  #[test]
  fn discrete_uniform_distribution_mean() {
    assert_eq!(
      interpret("Mean[DiscreteUniformDistribution[{1, 10}]]").unwrap(),
      "11/2"
    );
  }
  #[test]
  fn discrete_uniform_distribution_variance() {
    assert_eq!(
      interpret("Variance[DiscreteUniformDistribution[{1, 10}]]").unwrap(),
      "33/4"
    );
  }

  // ─── PositiveDefiniteMatrixQ ───────────────────────────────────────
  #[test]
  fn positive_definite_matrix_q_true() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{2, -1}, {-1, 2}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn positive_definite_matrix_q_false() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{1, 2}, {2, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn positive_definite_matrix_q_identity() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[IdentityMatrix[3]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn positive_definite_matrix_q_zero() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{0}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn positive_definite_matrix_q_scalar() {
    assert_eq!(interpret("PositiveDefiniteMatrixQ[{{5}}]").unwrap(), "True");
  }
  #[test]
  fn positive_definite_matrix_q_diagonal() {
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]")
        .unwrap(),
      "True"
    );
  }
  // Definiteness predicates answer False (no error, not unevaluated) for a
  // scalar, vector, or non-square matrix, matching wolframscript.
  #[test]
  fn positive_definite_matrix_q_non_matrix() {
    assert_eq!(interpret("PositiveDefiniteMatrixQ[5]").unwrap(), "False");
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{1, 2, 3}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PositiveDefiniteMatrixQ[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn positive_semidefinite_matrix_q_non_matrix() {
    assert_eq!(
      interpret("PositiveSemidefiniteMatrixQ[5]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("PositiveSemidefiniteMatrixQ[{1, 2}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn negative_definite_matrix_q_non_matrix() {
    assert_eq!(interpret("NegativeDefiniteMatrixQ[5]").unwrap(), "False");
    assert_eq!(
      interpret("NegativeDefiniteMatrixQ[{{-2, 0}, {0, -3}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn negative_semidefinite_matrix_q_non_matrix() {
    assert_eq!(
      interpret("NegativeSemidefiniteMatrixQ[5]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("NegativeSemidefiniteMatrixQ[{1, 2, 3}]").unwrap(),
      "False"
    );
  }

  // ─── IndefiniteMatrixQ ─────────────────────────────────────────────
  #[test]
  fn indefinite_matrix_q_true_diag() {
    // Eigenvalues -1 and 1: both signs present.
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{-1, 0}, {0, 1}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn indefinite_matrix_q_positive_definite() {
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{2, 0}, {0, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn indefinite_matrix_q_negative_definite() {
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{-2, 0}, {0, -1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn indefinite_matrix_q_semidefinite_zero_eigenvalue() {
    // Eigenvalues 1 and 0: positive + zero is not indefinite.
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{1, 0}, {0, 0}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn indefinite_matrix_q_offdiagonal() {
    // Eigenvalues 1 and -1.
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{0, 1}, {1, 0}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn indefinite_matrix_q_nonsymmetric_real_eigenvalues() {
    // [[1,2],[3,4]] -> Hermitian part eigenvalues straddle zero.
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn indefinite_matrix_q_uses_hermitian_part_not_eigenvalues() {
    // Nilpotent matrix: eigenvalues are both 0, but the Hermitian part
    // {{0,1},{1,0}} has eigenvalues -1 and 1, so WL reports True.
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{0, 2}, {0, 0}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn indefinite_matrix_q_complex_eigenvalues_false() {
    // Rotation matrix has purely imaginary eigenvalues; Hermitian part is
    // all zeros, so it is not indefinite.
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{0, -1}, {1, 0}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn indefinite_matrix_q_three_by_three() {
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{1, 2, 0}, {0, 3, 0}, {0, 0, -1}}]")
        .unwrap(),
      "True"
    );
  }
  #[test]
  fn indefinite_matrix_q_real_valued() {
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{1.5, 0}, {0, -2.5}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn indefinite_matrix_q_scalar_matrix() {
    assert_eq!(interpret("IndefiniteMatrixQ[{{-3}}]").unwrap(), "False");
  }
  #[test]
  fn indefinite_matrix_q_nonsquare() {
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn indefinite_matrix_q_vector() {
    assert_eq!(interpret("IndefiniteMatrixQ[{1, 2, 3}]").unwrap(), "False");
  }
  #[test]
  fn indefinite_matrix_q_symbolic_unknown_sign() {
    // The sign of `a` is unknown, so it is not *explicitly* indefinite.
    assert_eq!(
      interpret("IndefiniteMatrixQ[{{a, 0}, {0, 1}}]").unwrap(),
      "False"
    );
  }

  // ─── MovingMap ─────────────────────────────────────────────────────
  #[test]
  fn moving_map_total() {
    assert_eq!(
      interpret("MovingMap[Total, {1, 2, 3, 4, 5}, 2]").unwrap(),
      "{6, 9, 12}"
    );
  }
  #[test]
  fn moving_map_mean() {
    assert_eq!(
      interpret("MovingMap[Mean, {1, 2, 3, 4, 5}, 3]").unwrap(),
      "{5/2, 7/2}"
    );
  }
  // Regression: MovingMap must apply pure functions to each window, not just
  // named symbols.
  #[test]
  fn moving_map_pure_function() {
    assert_eq!(
      interpret("MovingMap[(#[[1]]*#[[2]]) &, {1, 2, 3, 4}, 1]").unwrap(),
      "{2, 6, 12}"
    );
    assert_eq!(
      interpret("MovingMap[Total[#]^2 &, {1, 2, 3, 4}, 1]").unwrap(),
      "{9, 25, 49}"
    );
  }

  // ─── Unevaluated batch ────────────────────────────────────────────
  #[test]
  fn notebook_find() {
    assert_eq!(interpret("NotebookFind[x]").unwrap(), "NotebookFind[x]");
  }
  #[test]
  fn classifier_measurements() {
    assert_eq!(
      interpret("ClassifierMeasurements[x, y]").unwrap(),
      "ClassifierMeasurements[x, y]"
    );
  }
  #[test]
  fn estimated_process() {
    assert_eq!(
      interpret("EstimatedProcess[x]").unwrap(),
      "EstimatedProcess[x]"
    );
  }
  #[test]
  fn highlight_mesh() {
    assert_eq!(interpret("HighlightMesh[x]").unwrap(), "HighlightMesh[x]");
  }
  #[test]
  fn animator() {
    assert_eq!(interpret("Animator[x]").unwrap(), "Animator[x]");
  }
  #[test]
  fn auto_scroll() {
    assert_eq!(interpret("AutoScroll[x]").unwrap(), "AutoScroll[x]");
  }
  #[test]
  fn confidence_level() {
    assert_eq!(
      interpret("ConfidenceLevel[x]").unwrap(),
      "ConfidenceLevel[x]"
    );
  }
  #[test]
  fn coefficient_rules() {
    // Single variable, single term
    assert_eq!(interpret("CoefficientRules[x, x]").unwrap(), "{{1} -> 1}");
    // Single variable polynomial
    assert_eq!(
      interpret("CoefficientRules[x^2 + 3*x + 5, x]").unwrap(),
      "{{2} -> 1, {1} -> 3, {0} -> 5}"
    );
    // Symbolic coefficients
    assert_eq!(
      interpret("CoefficientRules[a*x^2 + b*x + c, x]").unwrap(),
      "{{2} -> a, {1} -> b, {0} -> c}"
    );
    // Multivariate
    assert_eq!(
      interpret("CoefficientRules[(x + y)^3, {x, y}]").unwrap(),
      "{{3, 0} -> 1, {2, 1} -> 3, {1, 2} -> 3, {0, 3} -> 1}"
    );
    // Constant polynomial
    assert_eq!(interpret("CoefficientRules[5, x]").unwrap(), "{{0} -> 5}");
    // Zero polynomial
    assert_eq!(interpret("CoefficientRules[0, x]").unwrap(), "{}");
    // Variable list form, single variable
    assert_eq!(
      interpret("CoefficientRules[x^2 + 3*x + 5, {x}]").unwrap(),
      "{{2} -> 1, {1} -> 3, {0} -> 5}"
    );
    // Mixed multivariate
    assert_eq!(
      interpret("CoefficientRules[x^2 + y^2 + 1, {x, y}]").unwrap(),
      "{{2, 0} -> 1, {0, 2} -> 1, {0, 0} -> 1}"
    );
    // Multivariate with symbolic coefficients
    assert_eq!(
      interpret("CoefficientRules[a*x^2*y + b*x + c*y^3, {x, y}]").unwrap(),
      "{{2, 1} -> a, {1, 0} -> b, {0, 3} -> c}"
    );
    // Non-variable second arg returns unevaluated
    assert_eq!(
      interpret("CoefficientRules[x, 5]").unwrap(),
      "CoefficientRules[x, 5]"
    );
  }
  #[test]
  fn coefficient_rules_one_arg() {
    // One-argument form auto-detects variables via Variables[poly].
    assert_eq!(
      interpret("CoefficientRules[(x + y)^3]").unwrap(),
      "{{3, 0} -> 1, {2, 1} -> 3, {1, 2} -> 3, {0, 3} -> 1}"
    );
    assert_eq!(
      interpret("CoefficientRules[x^2 + 2*x + 1]").unwrap(),
      "{{2} -> 1, {1} -> 2, {0} -> 1}"
    );
    // Variable order follows Variables[]: Variables[a*x^2 + b] == {b, a, x}.
    assert_eq!(
      interpret("CoefficientRules[a*x^2 + b]").unwrap(),
      "{{1, 0, 0} -> 1, {0, 1, 2} -> 1}"
    );
    // Variables[3*x^2*y + 2*y] == {y, x}.
    assert_eq!(
      interpret("CoefficientRules[3*x^2*y + 2*y]").unwrap(),
      "{{1, 2} -> 3, {1, 0} -> 2}"
    );
    // Constant polynomial: empty exponent vector.
    assert_eq!(interpret("CoefficientRules[5]").unwrap(), "{{} -> 5}");
    // Zero polynomial.
    assert_eq!(interpret("CoefficientRules[0]").unwrap(), "{}");
  }
  #[test]
  fn thinning() {
    assert_eq!(interpret("Thinning[x]").unwrap(), "Thinning[x]");
  }
  #[test]
  fn erosion() {
    assert_eq!(interpret("Erosion[x]").unwrap(), "Erosion[x]");
  }
  #[test]
  fn tolerance() {
    assert_eq!(interpret("Tolerance[x]").unwrap(), "Tolerance[x]");
  }
  #[test]
  fn net_initialize() {
    assert_eq!(interpret("NetInitialize[x]").unwrap(), "NetInitialize[x]");
  }
  #[test]
  fn boundary_mesh_region() {
    assert_eq!(
      interpret("BoundaryMeshRegion[x]").unwrap(),
      "BoundaryMeshRegion[x]"
    );
  }
  #[test]
  fn geometric_brownian_motion_process() {
    assert_eq!(
      interpret("GeometricBrownianMotionProcess[x]").unwrap(),
      "GeometricBrownianMotionProcess[x]"
    );
  }
  #[test]
  fn boolean_convert() {
    // Simple pass-through
    assert_eq!(interpret("BooleanConvert[x]").unwrap(), "x");
    assert_eq!(interpret("BooleanConvert[a || b]").unwrap(), "a || b");
    assert_eq!(interpret("BooleanConvert[a && b]").unwrap(), "a && b");
    // Default (DNF): eliminate compound connectives
    assert_eq!(
      interpret("BooleanConvert[Implies[a, b]]").unwrap(),
      " !a || b"
    );
    assert_eq!(
      interpret("BooleanConvert[Nand[a, b]]").unwrap(),
      " !a ||  !b"
    );
    assert_eq!(
      interpret("BooleanConvert[Nor[a, b]]").unwrap(),
      " !a &&  !b"
    );
    // CNF form
    assert_eq!(
      interpret("BooleanConvert[Implies[a, b], \"CNF\"]").unwrap(),
      " !a || b"
    );
    assert_eq!(
      interpret("BooleanConvert[a && b, \"CNF\"]").unwrap(),
      "a && b"
    );
    // CNF with Or subterms must display with parens
    assert_eq!(
      interpret(r#"BooleanConvert[a || (b && c), "CNF"]"#).unwrap(),
      "(a || b) && (a || c)"
    );
    assert_eq!(
      interpret(r#"BooleanConvert[(a && b) || (c && d), "CNF"]"#).unwrap(),
      "(a || c) && (a || d) && (b || c) && (b || d)"
    );
  }
  #[test]
  fn boolean_counting_function_exactly_k() {
    // {k} → exactly k vars true. Natural DNF of size-k minterms, in lex
    // order over which variables are true.
    assert_eq!(
      interpret("BooleanCountingFunction[{2}, {a, b, c}]").unwrap(),
      "(a && b &&  !c) || (a &&  !b && c) || ( !a && b && c)"
    );
    assert_eq!(
      interpret("BooleanCountingFunction[{0}, {a, b, c}]").unwrap(),
      " !a &&  !b &&  !c"
    );
    assert_eq!(
      interpret("BooleanCountingFunction[{3}, {a, b, c}]").unwrap(),
      "a && b && c"
    );
    assert_eq!(
      interpret("BooleanCountingFunction[{1}, {a, b}]").unwrap(),
      "(a &&  !b) || ( !a && b)"
    );
    assert_eq!(
      interpret("BooleanCountingFunction[{2}, {a, b, c, d}]").unwrap(),
      "(a && b &&  !c &&  !d) || (a &&  !b && c &&  !d) || (a &&  !b &&  !c && d) || ( !a && b && c &&  !d) || ( !a && b &&  !c && d) || ( !a &&  !b && c && d)"
    );
  }

  #[test]
  fn boolean_counting_function_at_most_k() {
    // Plain integer k_max → at most k_max true. Equivalent to "at least
    // (n - k_max) false"; emit as Or of (n - k_max)-subsets of negated
    // variables (in lex order over the subset's variables).
    assert_eq!(
      interpret("BooleanCountingFunction[2, {a, b, c}]").unwrap(),
      " !a ||  !b ||  !c"
    );
    assert_eq!(
      interpret("BooleanCountingFunction[1, {a, b, c}]").unwrap(),
      "( !a &&  !b) || ( !a &&  !c) || ( !b &&  !c)"
    );
    assert_eq!(
      interpret("BooleanCountingFunction[0, {a, b, c}]").unwrap(),
      " !a &&  !b &&  !c"
    );
  }

  #[test]
  fn boolean_counting_function_between() {
    // {k_min, k_max} with k_min == 0 collapses to the "at most k_max"
    // pattern, so its output coincides with wolframscript's.
    assert_eq!(
      interpret("BooleanCountingFunction[{0, 2}, {a, b, c}]").unwrap(),
      " !a ||  !b ||  !c"
    );
  }

  #[test]
  fn boolean_counting_function_audit_case_evaluates() {
    // Audit regression: previously returned unevaluated. Now produces a
    // minimized DNF that is logically equivalent to "between 2 and 3 of
    // {a, b, c, d}". The exact term selection differs from
    // wolframscript (whose BooleanCountingFunction emits a non-minimal
    // 7-term cover); Woxi runs the input through its full Quine-McCluskey
    // minimizer instead.
    let result =
      interpret("BooleanCountingFunction[{2, 3}, {a, b, c, d}]").unwrap();
    // Output is an Or of And terms — not unevaluated.
    assert!(
      result.contains("||"),
      "expected Or-form DNF, got `{}`",
      result
    );
    assert!(
      !result.starts_with("BooleanCountingFunction"),
      "still unevaluated: {}",
      result
    );
  }

  #[test]
  fn select_components() {
    assert_eq!(
      interpret("SelectComponents[x]").unwrap(),
      "SelectComponents[x]"
    );
  }
  #[test]
  fn mesh_cell_style() {
    assert_eq!(interpret("MeshCellStyle[x]").unwrap(), "MeshCellStyle[x]");
  }
  #[test]
  fn notebook_put() {
    assert_eq!(interpret("NotebookPut[x]").unwrap(), "NotebookPut[x]");
  }
  #[test]
  fn text_sentences() {
    assert_eq!(interpret("TextSentences[x]").unwrap(), "TextSentences[x]");
  }
  #[test]
  fn polynomial_reduce() {
    assert_eq!(
      interpret("PolynomialReduce[x, y]").unwrap(),
      "PolynomialReduce[x, y]"
    );
  }
  #[test]
  fn cumulant_unevaluated() {
    assert_eq!(interpret("Cumulant[x]").unwrap(), "Cumulant[x]");
  }
  #[test]
  fn three_j_symbol() {
    assert_eq!(
      interpret("ThreeJSymbol[x, y]").unwrap(),
      "ThreeJSymbol[x, y]"
    );
  }
  #[test]
  fn copy_file() {
    assert_eq!(interpret("CopyFile[x]").unwrap(), "CopyFile[x]");
  }
  #[test]
  fn create_directory() {
    assert_eq!(interpret("CreateDirectory[x]").unwrap(), "$Failed");
  }

  // ─── DiscreteDelta ─────────────────────────────────────────────────
  #[test]
  fn discrete_delta_zero() {
    assert_eq!(interpret("DiscreteDelta[0]").unwrap(), "1");
  }
  #[test]
  fn discrete_delta_nonzero() {
    assert_eq!(interpret("DiscreteDelta[1]").unwrap(), "0");
  }
  #[test]
  fn discrete_delta_multiple_zeros() {
    assert_eq!(interpret("DiscreteDelta[0, 0]").unwrap(), "1");
  }
  #[test]
  fn discrete_delta_mixed() {
    assert_eq!(interpret("DiscreteDelta[0, 1]").unwrap(), "0");
  }
  #[test]
  fn discrete_delta_no_args() {
    assert_eq!(interpret("DiscreteDelta[]").unwrap(), "1");
  }
  #[test]
  fn discrete_delta_symbolic() {
    assert_eq!(interpret("DiscreteDelta[x]").unwrap(), "DiscreteDelta[x]");
  }

  // ─── Unevaluated batch 5 ──────────────────────────────────────────
  #[test]
  fn magnify() {
    assert_eq!(interpret("Magnify[x]").unwrap(), "Magnify[x]");
  }
  #[test]
  fn script_baseline_shifts() {
    assert_eq!(
      interpret("ScriptBaselineShifts[x]").unwrap(),
      "ScriptBaselineShifts[x]"
    );
  }
  #[test]
  fn line_spacing() {
    assert_eq!(interpret("LineSpacing[x]").unwrap(), "LineSpacing[x]");
  }
  #[test]
  fn function_range() {
    assert_eq!(
      interpret("FunctionRange[x, y]").unwrap(),
      "FunctionRange[x, y]"
    );
  }
  #[test]
  fn sector_origin() {
    assert_eq!(interpret("SectorOrigin[x]").unwrap(), "SectorOrigin[x]");
  }
  #[test]
  fn max_training_rounds() {
    assert_eq!(
      interpret("MaxTrainingRounds[x]").unwrap(),
      "MaxTrainingRounds[x]"
    );
  }
  #[test]
  fn polar_axes() {
    assert_eq!(interpret("PolarAxes[x]").unwrap(), "PolarAxes[x]");
  }
  #[test]
  fn polynomial_gcd() {
    // Coprime polynomials in different variables
    assert_eq!(interpret("PolynomialGCD[x, y]").unwrap(), "1");
    // Basic GCD: (x-1) is common factor
    assert_eq!(
      interpret("PolynomialGCD[x^2 - 1, x - 1]").unwrap(),
      "-1 + x"
    );
    // GCD of x^2-1 = (x-1)(x+1) and x^2-2x+1 = (x-1)^2
    assert_eq!(
      interpret("PolynomialGCD[x^2 - 1, x^2 - 2x + 1]").unwrap(),
      "-1 + x"
    );
    // GCD with numeric content: 6x^2+3x=3x(2x+1), 4x^2+2x=2x(2x+1)
    assert_eq!(
      interpret("PolynomialGCD[6*x^2 + 3*x, 4*x^2 + 2*x]").unwrap(),
      "x + 2*x^2"
    );
    // Integer GCD
    assert_eq!(interpret("PolynomialGCD[12, 8]").unwrap(), "4");
    // x^3-1 = (x-1)(x^2+x+1), x^2-1 = (x-1)(x+1)
    assert_eq!(
      interpret("PolynomialGCD[x^3 - 1, x^2 - 1]").unwrap(),
      "-1 + x"
    );
    // (x+1)^2 and (x+1)(x+2)
    assert_eq!(
      interpret("PolynomialGCD[x^2 + 2*x + 1, x^2 + 3*x + 2]").unwrap(),
      "1 + x"
    );
    // 2(x+1) and (x-1)(x+1)
    assert_eq!(
      interpret("PolynomialGCD[2*x + 2, x^2 - 1]").unwrap(),
      "1 + x"
    );
    // Power GCD
    assert_eq!(interpret("PolynomialGCD[x^2, x^3]").unwrap(), "x^2");
    // Zero case
    assert_eq!(interpret("PolynomialGCD[0, x^2 + 1]").unwrap(), "1 + x^2");
    // GCD with itself
    assert_eq!(interpret("PolynomialGCD[x + 1, x + 1]").unwrap(), "1 + x");
    // Coprime polynomials
    assert_eq!(interpret("PolynomialGCD[x + 1, x + 2]").unwrap(), "1");
  }
  #[test]
  fn system_dialog_input() {
    assert_eq!(
      interpret("SystemDialogInput[x]").unwrap(),
      "SystemDialogInput[x]"
    );
  }
  #[test]
  fn ar_process() {
    assert_eq!(interpret("ARProcess[x]").unwrap(), "ARProcess[x]");
  }
  #[test]
  fn discrete_wavelet_transform() {
    assert_eq!(
      interpret("DiscreteWaveletTransform[x]").unwrap(),
      "DiscreteWaveletTransform[x]"
    );
  }
  #[test]
  fn relation_graph() {
    assert_eq!(interpret("RelationGraph[x]").unwrap(), "RelationGraph[x]");
  }
  #[test]
  fn image_partition() {
    assert_eq!(interpret("ImagePartition[x]").unwrap(), "ImagePartition[x]");
  }
  #[test]
  fn petersen_graph() {
    assert_eq!(interpret("PetersenGraph[x]").unwrap(), "PetersenGraph[x]");
  }
  #[test]
  fn petersen_graph_classic() {
    // The classic Petersen graph has 10 vertices and 15 edges.
    assert_eq!(interpret("Head[PetersenGraph[]]").unwrap(), "Graph");
    assert_eq!(
      interpret("Length[VertexList[PetersenGraph[]]]").unwrap(),
      "10"
    );
    assert_eq!(
      interpret("Length[EdgeList[PetersenGraph[]]]").unwrap(),
      "15"
    );
  }
  #[test]
  fn petersen_graph_with_options() {
    // Audit case: PetersenGraph[5, 2, VertexSize -> {1 -> Medium}].
    assert_eq!(
      interpret("Head[PetersenGraph[5, 2, VertexSize -> {1 -> Medium}]]")
        .unwrap(),
      "Graph"
    );
  }
  #[test]
  fn graph_plot_forwards_to_graph() {
    assert_eq!(
      interpret(
        "GraphPlot[{1 -> 4, 1 -> 5, 2 -> 3, 2 -> 5, 3 -> 5, 4 -> 5, 5 -> 5}]"
      )
      .unwrap(),
      "-Graphics-"
    );
  }
  #[test]
  fn layered_graph_plot_forwards() {
    assert_eq!(
      interpret("LayeredGraphPlot[PetersenGraph[]]").unwrap(),
      "-Graphics-"
    );
    assert_eq!(
      interpret(
        "LayeredGraphPlot[{1 -> 2, 1 -> 3, 2 -> 3, 1 -> 4, 2 -> 4, 1 -> 5}, Left]"
      )
      .unwrap(),
      "-Graphics-"
    );
  }
  #[test]
  fn r_solve_value() {
    assert_eq!(
      interpret("RSolveValue[x, y, z]").unwrap(),
      "RSolveValue[x, y, z]"
    );
  }
  #[test]
  fn feature_extraction() {
    assert_eq!(
      interpret("FeatureExtraction[x]").unwrap(),
      "FeatureExtraction[x]"
    );
  }
  #[test]
  fn graph_distance() {
    assert_eq!(
      interpret("GraphDistance[x, y]").unwrap(),
      "GraphDistance[x, y]"
    );
  }
  #[test]
  fn cell_style() {
    assert_eq!(interpret("CellStyle[x]").unwrap(), "CellStyle[x]");
  }
  #[test]
  fn directory_q() {
    assert_eq!(interpret("DirectoryQ[x]").unwrap(), "False");
  }

  #[test]
  fn directory_q_true() {
    assert_eq!(interpret(r#"DirectoryQ["/tmp"]"#).unwrap(), "True");
  }

  #[test]
  fn directory_q_nonexistent() {
    assert_eq!(
      interpret(r#"DirectoryQ["/nonexistent_dir_xyz"]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn directory_q_file() {
    let file = "/tmp/woxi_test_directoryq_file.txt";
    std::fs::write(file, "test").unwrap();
    assert_eq!(
      interpret(&format!(r#"DirectoryQ["{}"]"#, file)).unwrap(),
      "False"
    );
    std::fs::remove_file(file).unwrap();
  }

  #[test]
  fn file_type_directory() {
    assert_eq!(interpret(r#"FileType["/tmp"]"#).unwrap(), "Directory");
  }

  #[test]
  fn file_type_file() {
    let file = "/tmp/woxi_test_filetype.txt";
    std::fs::write(file, "test").unwrap();
    assert_eq!(
      interpret(&format!(r#"FileType["{}"]"#, file)).unwrap(),
      "File"
    );
    std::fs::remove_file(file).unwrap();
  }

  #[test]
  fn file_type_nonexistent() {
    assert_eq!(
      interpret(r#"FileType["/nonexistent_xyz"]"#).unwrap(),
      "None"
    );
  }

  #[test]
  fn file_type_non_string() {
    assert_eq!(interpret("FileType[x]").unwrap(), "FileType[x]");
  }

  #[test]
  fn file_print_basic() {
    let file = "/tmp/woxi_test_fileprint.txt";
    std::fs::write(file, "Hello World\nLine 2\nLine 3").unwrap();
    let result =
      interpret_with_stdout(&format!(r#"FilePrint["{}"]"#, file)).unwrap();
    assert_eq!(result.stdout, "Hello World\nLine 2\nLine 3\n");
    assert_eq!(result.result, "\0");
    std::fs::remove_file(file).unwrap();
  }

  #[test]
  fn file_print_nonexistent() {
    let result = interpret(r#"FilePrint["/nonexistent_xyz"]"#).unwrap();
    assert_eq!(result, "FilePrint[/nonexistent_xyz]");
  }

  #[test]
  fn file_print_non_string() {
    assert_eq!(interpret("FilePrint[x]").unwrap(), "FilePrint[x]");
  }

  #[test]
  fn image_identify() {
    assert_eq!(interpret("ImageIdentify[x]").unwrap(), "ImageIdentify[x]");
  }
  #[test]
  fn asymptotic() {
    assert_eq!(interpret("Asymptotic[x, y]").unwrap(), "Asymptotic[x, y]");
  }
  #[test]
  fn coordinate_transform() {
    assert_eq!(
      interpret("CoordinateTransform[x, y]").unwrap(),
      "CoordinateTransform[x, y]"
    );
  }
  #[test]
  fn window_margins() {
    assert_eq!(interpret("WindowMargins[x]").unwrap(), "WindowMargins[x]");
  }
  #[test]
  fn affine_transform() {
    assert_eq!(
      interpret("AffineTransform[x]").unwrap(),
      "AffineTransform[x]"
    );
  }
  #[test]
  fn radio_button() {
    assert_eq!(interpret("RadioButton[x]").unwrap(), "RadioButton[x]");
  }
  #[test]
  fn legend_markers() {
    assert_eq!(interpret("LegendMarkers[x]").unwrap(), "LegendMarkers[x]");
  }
  #[test]
  fn powers_representations() {
    assert_eq!(
      interpret("PowersRepresentations[x, y, z]").unwrap(),
      "PowersRepresentations[x, y, z]"
    );
  }
  #[test]
  fn show_string_characters() {
    assert_eq!(
      interpret("ShowStringCharacters[x]").unwrap(),
      "ShowStringCharacters[x]"
    );
  }

  // ─── FlattenAt ─────────────────────────────────────────────────────
  #[test]
  fn flatten_at_single() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, d}, 2]").unwrap(),
      "{a, b, c, d}"
    );
  }
  #[test]
  fn flatten_at_negative() {
    assert_eq!(
      interpret("FlattenAt[{a, {b, c}, d}, -2]").unwrap(),
      "{a, b, c, d}"
    );
  }
  #[test]
  fn flatten_at_first() {
    assert_eq!(
      interpret("FlattenAt[{{a, b}, {c, d}, {e, f}}, 2]").unwrap(),
      "{{a, b}, c, d, {e, f}}"
    );
  }

  // ─── InversePermutation ────────────────────────────────────────────
  #[test]
  fn inverse_permutation_basic() {
    assert_eq!(
      interpret("InversePermutation[{3, 1, 2}]").unwrap(),
      "{2, 3, 1}"
    );
  }
  #[test]
  fn inverse_permutation_4() {
    assert_eq!(
      interpret("InversePermutation[{2, 4, 1, 3}]").unwrap(),
      "{3, 1, 4, 2}"
    );
  }

  // ─── Unevaluated batch 6 ──────────────────────────────────────────
  #[test]
  fn nd_eigensystem() {
    assert_eq!(
      interpret("NDEigensystem[x, y]").unwrap(),
      "NDEigensystem[x, y]"
    );
  }
  #[test]
  fn texture_coordinate_function() {
    assert_eq!(
      interpret("TextureCoordinateFunction[x]").unwrap(),
      "TextureCoordinateFunction[x]"
    );
  }
  #[test]
  fn find_distribution() {
    assert_eq!(
      interpret("FindDistribution[x]").unwrap(),
      "FindDistribution[x]"
    );
  }
  #[test]
  fn text_cases() {
    assert_eq!(interpret("TextCases[x, y]").unwrap(), "TextCases[x, y]");
  }
  #[test]
  fn multicolumn() {
    assert_eq!(interpret("Multicolumn[x]").unwrap(), "Multicolumn[x]");
  }
  #[test]
  fn record() {
    assert_eq!(interpret("Record[x]").unwrap(), "Record[x]");
  }
  #[test]
  fn whittaker_m() {
    assert_eq!(
      interpret("WhittakerM[x, y, z]").unwrap(),
      "WhittakerM[x, y, z]"
    );
  }
  #[test]
  fn interpretation_box() {
    assert_eq!(
      interpret("InterpretationBox[x]").unwrap(),
      "InterpretationBox[x]"
    );
  }
  #[test]
  fn include_pods() {
    assert_eq!(interpret("IncludePods[x]").unwrap(), "IncludePods[x]");
  }
  #[test]
  fn rule_plot() {
    // RulePlot now returns a Graphics placeholder.
    assert_eq!(interpret("RulePlot[x]").unwrap(), "-Graphics-");
  }
  #[test]
  fn mathieu_group_m11() {
    assert_eq!(
      interpret("MathieuGroupM11[x]").unwrap(),
      "MathieuGroupM11[x]"
    );
  }
  #[test]
  fn trig() {
    assert_eq!(interpret("Trig[x]").unwrap(), "Trig[x]");
  }
  #[test]
  fn overlaps() {
    assert_eq!(interpret("Overlaps[x]").unwrap(), "Overlaps[x]");
  }
  #[test]
  fn ito_process() {
    assert_eq!(interpret("ItoProcess[x]").unwrap(), "ItoProcess[x]");
  }
  #[test]
  fn rotation_action() {
    assert_eq!(interpret("RotationAction[x]").unwrap(), "RotationAction[x]");
  }
  #[test]
  fn ket() {
    assert_eq!(interpret("Ket[x]").unwrap(), "Ket[x]");
  }
  #[test]
  fn discrete_markov_process() {
    assert_eq!(
      interpret("DiscreteMarkovProcess[x, y]").unwrap(),
      "DiscreteMarkovProcess[x, y]"
    );
  }
  #[test]
  fn boundary_discretize_graphics() {
    assert_eq!(
      interpret("BoundaryDiscretizeGraphics[x]").unwrap(),
      "BoundaryDiscretizeGraphics[x]"
    );
  }
  #[test]
  fn trading_chart() {
    assert_eq!(interpret("TradingChart[x]").unwrap(), "TradingChart[x]");
  }
  #[test]
  fn find_max_value() {
    assert_eq!(
      interpret("FindMaxValue[x, y]").unwrap(),
      "FindMaxValue[x, y]"
    );
  }
  #[test]
  fn form_page() {
    assert_eq!(interpret("FormPage[x]").unwrap(), "FormPage[x]");
  }
  #[test]
  fn nearest_neighbor_graph() {
    assert_eq!(
      interpret("NearestNeighborGraph[x]").unwrap(),
      "NearestNeighborGraph[x]"
    );
  }
  #[test]
  fn file_print() {
    assert_eq!(interpret("FilePrint[x]").unwrap(), "FilePrint[x]");
  }
  #[test]
  fn riemann_siegel_z() {
    assert_eq!(interpret("RiemannSiegelZ[x]").unwrap(), "RiemannSiegelZ[x]");
  }
  #[test]
  fn chart_base_style() {
    assert_eq!(interpret("ChartBaseStyle[x]").unwrap(), "ChartBaseStyle[x]");
  }

  // ─── Unevaluated batch 7 ──────────────────────────────────────────
  #[test]
  fn moon_phase() {
    assert_eq!(interpret("MoonPhase[x]").unwrap(), "MoonPhase[x]");
  }
  #[test]
  fn hazard_function() {
    assert_eq!(
      interpret("HazardFunction[x, y]").unwrap(),
      "HazardFunction[x, y]"
    );
  }
  #[test]
  fn content_size() {
    assert_eq!(interpret("ContentSize[x]").unwrap(), "ContentSize[x]");
  }
  #[test]
  fn horner_form() {
    assert_eq!(interpret("HornerForm[x]").unwrap(), "x");
  }
  #[test]
  fn word_boundary() {
    assert_eq!(interpret("WordBoundary[x]").unwrap(), "WordBoundary[x]");
  }
  #[test]
  fn n_expectation() {
    assert_eq!(
      interpret("NExpectation[x, y]").unwrap(),
      "NExpectation[x, y]"
    );
  }
  #[test]
  fn mouseover() {
    assert_eq!(interpret("Mouseover[x, y]").unwrap(), "Mouseover[x, y]");
  }
  #[test]
  fn rectangle_chart() {
    assert_eq!(interpret("RectangleChart[x]").unwrap(), "RectangleChart[x]");
  }
  #[test]
  fn affine_state_space_model() {
    assert_eq!(
      interpret("AffineStateSpaceModel[x]").unwrap(),
      "AffineStateSpaceModel[x]"
    );
  }
  #[test]
  fn log_likelihood() {
    assert_eq!(
      interpret("LogLikelihood[x, y]").unwrap(),
      "LogLikelihood[x, y]"
    );
  }
  #[test]
  fn span_from_above() {
    assert_eq!(interpret("SpanFromAbove[x]").unwrap(), "SpanFromAbove[x]");
  }
  #[test]
  fn min_value() {
    assert_eq!(interpret("MinValue[x, y]").unwrap(), "MinValue[x, y]");
  }
  #[test]
  fn sub_plus() {
    assert_eq!(interpret("SubPlus[x]").unwrap(), "SubPlus[x]");
  }
  #[test]
  fn extension() {
    assert_eq!(interpret("Extension[x]").unwrap(), "Extension[x]");
  }
  #[test]
  fn weighted_adjacency_graph() {
    assert_eq!(
      interpret("WeightedAdjacencyGraph[x]").unwrap(),
      "WeightedAdjacencyGraph[x]"
    );
  }
  #[test]
  fn cell_frame() {
    assert_eq!(interpret("CellFrame[x]").unwrap(), "CellFrame[x]");
  }
  #[test]
  fn compiled() {
    assert_eq!(interpret("Compiled[x]").unwrap(), "Compiled[x]");
  }
  #[test]
  fn audio_generator() {
    assert_eq!(interpret("AudioGenerator[x]").unwrap(), "AudioGenerator[x]");
  }
  #[test]
  fn underlined() {
    assert_eq!(interpret("Underlined[x]").unwrap(), "Underlined[x]");
  }
  #[test]
  fn fourier_coefficient() {
    assert_eq!(
      interpret("FourierCoefficient[x, y, z]").unwrap(),
      "FourierCoefficient[x, y, z]"
    );
  }
  #[test]
  fn overscript() {
    assert_eq!(interpret("Overscript[x, y]").unwrap(), "Overscript[x, y]");
  }
  #[test]
  fn primes() {
    assert_eq!(interpret("Primes[x]").unwrap(), "Primes[x]");
  }
  #[test]
  fn community_graph_plot() {
    assert_eq!(
      interpret("CommunityGraphPlot[x]").unwrap(),
      "CommunityGraphPlot[x]"
    );
  }
  #[test]
  fn random_prime() {
    assert_eq!(interpret("RandomPrime[x]").unwrap(), "RandomPrime[x]");
  }
  #[test]
  fn super_dagger() {
    assert_eq!(interpret("SuperDagger[x]").unwrap(), "SuperDagger[x]");
  }
  #[test]
  fn re_im_plot() {
    assert_eq!(interpret("ReImPlot[x, y]").unwrap(), "ReImPlot[x, y]");
  }
  #[test]
  fn exponent_function() {
    assert_eq!(
      interpret("ExponentFunction[x]").unwrap(),
      "ExponentFunction[x]"
    );
  }
  #[test]
  fn product_distribution() {
    assert_eq!(
      interpret("ProductDistribution[x]").unwrap(),
      "ProductDistribution[x]"
    );
  }
  #[test]
  fn toggler_bar() {
    assert_eq!(interpret("TogglerBar[x]").unwrap(), "TogglerBar[x]");
  }

  // ─── TakeList ──────────────────────────────────────────────────────
  #[test]
  fn take_list_basic() {
    assert_eq!(
      interpret("TakeList[{a, b, c, d, e, f}, {2, 3, 1}]").unwrap(),
      "{{a, b}, {c, d, e}, {f}}"
    );
  }
  #[test]
  fn take_list_equal_parts() {
    assert_eq!(
      interpret("TakeList[{1, 2, 3, 4}, {2, 2}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  // ─── MultiplicativeOrder ───────────────────────────────────────────
  #[test]
  fn multiplicative_order_basic() {
    assert_eq!(interpret("MultiplicativeOrder[2, 7]").unwrap(), "3");
  }
  #[test]
  fn multiplicative_order_3_10() {
    assert_eq!(interpret("MultiplicativeOrder[3, 10]").unwrap(), "4");
  }
  #[test]
  fn multiplicative_order_10_7() {
    assert_eq!(interpret("MultiplicativeOrder[10, 7]").unwrap(), "6");
  }
  // Generalized form: smallest m with k^m congruent to some residue mod n.
  #[test]
  fn multiplicative_order_generalized() {
    assert_eq!(
      interpret("MultiplicativeOrder[2, 7, {1, -1}]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("MultiplicativeOrder[7, 11, {1, 3, 9}]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("MultiplicativeOrder[3, 10, {1, -1}]").unwrap(),
      "2"
    );
    assert_eq!(interpret("MultiplicativeOrder[2, 7, {2}]").unwrap(), "1");
  }
  // No power of k reaches any residue: stays unevaluated.
  #[test]
  fn multiplicative_order_generalized_unevaluated() {
    assert_eq!(
      interpret("MultiplicativeOrder[2, 7, {3}]").unwrap(),
      "MultiplicativeOrder[2, 7, {3}]"
    );
    assert_eq!(
      interpret("MultiplicativeOrder[6, 9, {1}]").unwrap(),
      "MultiplicativeOrder[6, 9, {1}]"
    );
  }

  // ─── Unevaluated batch 8 ──────────────────────────────────────────
  #[test]
  fn region_dimension() {
    assert_eq!(
      interpret("RegionDimension[x]").unwrap(),
      "RegionDimension[x]"
    );
  }
  #[test]
  fn feature_extractor() {
    assert_eq!(
      interpret("FeatureExtractor[x]").unwrap(),
      "FeatureExtractor[x]"
    );
  }
  #[test]
  fn arg_max() {
    assert_eq!(interpret("ArgMax[x, y]").unwrap(), "ArgMax[x, y]");
  }
  #[test]
  fn vertex_normals() {
    assert_eq!(interpret("VertexNormals[x]").unwrap(), "VertexNormals[x]");
  }
  #[test]
  fn correlation_function() {
    assert_eq!(
      interpret("CorrelationFunction[x, y]").unwrap(),
      "CorrelationFunction[x, y]"
    );
  }
  #[test]
  fn bell_y() {
    assert_eq!(interpret("BellY[x, y, z]").unwrap(), "BellY[x, y, z]");
  }
  #[test]
  fn barnes_g() {
    assert_eq!(interpret("BarnesG[x]").unwrap(), "BarnesG[x]");
  }
  #[test]
  fn url() {
    assert_eq!(interpret("URL[x]").unwrap(), "URL[x]");
  }
  #[test]
  fn find_geometric_transform() {
    assert_eq!(
      interpret("FindGeometricTransform[x, y]").unwrap(),
      "FindGeometricTransform[x, y]"
    );
  }
  #[test]
  fn deployed() {
    assert_eq!(interpret("Deployed[x]").unwrap(), "Deployed[x]");
  }
  #[test]
  fn dirichlet_distribution() {
    assert_eq!(
      interpret("DirichletDistribution[x]").unwrap(),
      "DirichletDistribution[x]"
    );
  }
  #[test]
  fn riemann_siegel_theta() {
    assert_eq!(
      interpret("RiemannSiegelTheta[x]").unwrap(),
      "RiemannSiegelTheta[x]"
    );
  }
  #[test]
  fn random_instance() {
    assert_eq!(interpret("RandomInstance[x]").unwrap(), "RandomInstance[x]");
  }
  #[test]
  fn notebook_delete() {
    assert_eq!(interpret("NotebookDelete[x]").unwrap(), "NotebookDelete[x]");
  }
  #[test]
  fn find_formula() {
    assert_eq!(interpret("FindFormula[x, y]").unwrap(), "FindFormula[x, y]");
  }
  #[test]
  fn graph_3d() {
    assert_eq!(interpret("Graph3D[x]").unwrap(), "Graph3D[x]");
  }
  #[test]
  fn whittaker_w() {
    assert_eq!(
      interpret("WhittakerW[x, y, z]").unwrap(),
      "WhittakerW[x, y, z]"
    );
  }
  #[test]
  fn max_detect() {
    assert_eq!(interpret("MaxDetect[x]").unwrap(), "MaxDetect[x]");
  }
  #[test]
  fn geometric_scene() {
    assert_eq!(interpret("GeometricScene[x]").unwrap(), "GeometricScene[x]");
  }
  #[test]
  fn clustering_components() {
    assert_eq!(
      interpret("ClusteringComponents[x]").unwrap(),
      "ClusteringComponents[x]"
    );
  }
  #[test]
  fn bernoulli_graph_distribution() {
    assert_eq!(
      interpret("BernoulliGraphDistribution[x, y]").unwrap(),
      "BernoulliGraphDistribution[x, y]"
    );
  }
  #[test]
  fn mandelbrot_set_plot() {
    assert_eq!(
      interpret("MandelbrotSetPlot[x]").unwrap(),
      "MandelbrotSetPlot[x]"
    );
  }
  #[test]
  fn language() {
    assert_eq!(interpret("Language[x]").unwrap(), "Language[x]");
  }
  #[test]
  fn sequence_cases() {
    assert_eq!(
      interpret("SequenceCases[x, y]").unwrap(),
      "SequenceCases[x, y]"
    );
  }
  #[test]
  fn time_constraint() {
    assert_eq!(interpret("TimeConstraint[x]").unwrap(), "TimeConstraint[x]");
  }
  #[test]
  fn double_right_tee() {
    assert_eq!(interpret("DoubleRightTee[x]").unwrap(), "DoubleRightTee[x]");
  }
  #[test]
  fn matrices() {
    assert_eq!(interpret("Matrices[x]").unwrap(), "Matrices[x]");
  }
  #[test]
  fn joined_curve() {
    assert_eq!(interpret("JoinedCurve[x]").unwrap(), "JoinedCurve[x]");
  }
  #[test]
  fn run_process() {
    assert_eq!(interpret("RunProcess[x]").unwrap(), "RunProcess[x]");
  }
  #[test]
  fn starting_step_size() {
    assert_eq!(
      interpret("StartingStepSize[x]").unwrap(),
      "StartingStepSize[x]"
    );
  }
  #[test]
  fn default_button() {
    assert_eq!(interpret("DefaultButton[x]").unwrap(), "DefaultButton[x]");
  }
  #[test]
  fn trigger() {
    assert_eq!(interpret("Trigger[x]").unwrap(), "Trigger[x]");
  }
  #[test]
  fn geo_marker() {
    assert_eq!(interpret("GeoMarker[x]").unwrap(), "GeoMarker[x]");
  }
  #[test]
  fn content_selectable() {
    assert_eq!(
      interpret("ContentSelectable[x]").unwrap(),
      "ContentSelectable[x]"
    );
  }

  // ─── LaplaceDistribution ──────────────────────────────────────────
  #[test]
  fn laplace_distribution_pdf() {
    assert_eq!(
      interpret("PDF[LaplaceDistribution[0, 1], 0]").unwrap(),
      "1/2"
    );
  }
  #[test]
  fn laplace_distribution_cdf() {
    assert_eq!(
      interpret("CDF[LaplaceDistribution[0, 1], 0]").unwrap(),
      "1/2"
    );
  }
  #[test]
  fn laplace_distribution_mean() {
    assert_eq!(interpret("Mean[LaplaceDistribution[2, 3]]").unwrap(), "2");
  }
  #[test]
  fn laplace_distribution_variance() {
    assert_eq!(
      interpret("Variance[LaplaceDistribution[2, 3]]").unwrap(),
      "18"
    );
  }

  // ─── RayleighDistribution ─────────────────────────────────────────
  #[test]
  fn rayleigh_distribution_pdf() {
    assert_eq!(
      interpret("N[PDF[RayleighDistribution[1], 1]]").unwrap(),
      interpret("N[1/Sqrt[E]]").unwrap()
    );
  }
  #[test]
  fn rayleigh_distribution_cdf() {
    assert_eq!(
      interpret("N[CDF[RayleighDistribution[1], 1]]").unwrap(),
      interpret("N[1 - 1/Sqrt[E]]").unwrap()
    );
  }
  #[test]
  fn rayleigh_distribution_mean() {
    assert_eq!(
      interpret("Mean[RayleighDistribution[s]]").unwrap(),
      "Sqrt[Pi/2]*s"
    );
  }
  #[test]
  fn rayleigh_distribution_variance() {
    assert_eq!(
      interpret("Variance[RayleighDistribution[s]]").unwrap(),
      "s^2*(2 - Pi/2)"
    );
  }

  // ─── Unevaluated batch 9 ──────────────────────────────────────────
  #[test]
  fn export_form() {
    assert_eq!(interpret("ExportForm[x, y]").unwrap(), "ExportForm[x, y]");
  }
  #[test]
  fn parallel_submit() {
    assert_eq!(interpret("ParallelSubmit[x]").unwrap(), "ParallelSubmit[x]");
  }
  #[test]
  fn application() {
    assert_eq!(interpret("Application[x]").unwrap(), "Application[x]");
  }
  #[test]
  fn find_file() {
    assert_eq!(interpret("FindFile[x]").unwrap(), "FindFile[x]");
  }
  #[test]
  fn distance_transform() {
    assert_eq!(
      interpret("DistanceTransform[x]").unwrap(),
      "DistanceTransform[x]"
    );
  }
  #[test]
  fn timeline_plot() {
    assert_eq!(interpret("TimelinePlot[x]").unwrap(), "TimelinePlot[x]");
  }
  #[test]
  fn pass_events_down() {
    assert_eq!(interpret("PassEventsDown[x]").unwrap(), "PassEventsDown[x]");
  }
  #[test]
  fn circle_dot() {
    assert_eq!(interpret("CircleDot[x]").unwrap(), "CircleDot[x]");
  }
  #[test]
  fn vector_scaling() {
    assert_eq!(interpret("VectorScaling[x]").unwrap(), "VectorScaling[x]");
  }
  #[test]
  fn find_generating_function() {
    assert_eq!(
      interpret("FindGeneratingFunction[x, y]").unwrap(),
      "FindGeneratingFunction[x, y]"
    );
  }
  #[test]
  fn associate_to() {
    assert_eq!(interpret("AssociateTo[x, y]").unwrap(), "AssociateTo[x, y]");
  }
  #[test]
  fn histogram_distribution() {
    assert_eq!(
      interpret("HistogramDistribution[x]").unwrap(),
      "HistogramDistribution[x]"
    );
  }
  #[test]
  fn gaussian_matrix() {
    assert_eq!(interpret("GaussianMatrix[x]").unwrap(), "GaussianMatrix[x]");
  }
  #[test]
  fn text_recognize() {
    assert_eq!(interpret("TextRecognize[x]").unwrap(), "TextRecognize[x]");
  }
  #[test]
  fn text_recognize_image_returns_empty_string() {
    // Woxi has no OCR backend. For ordinary images wolframscript falls
    // back to an empty string when no text is recognised; Woxi mirrors
    // that fallback so the call doesn't stay unevaluated.
    assert_eq!(interpret("TextRecognize[Image[{{0.5, 0.6}}]]").unwrap(), "");
  }
  #[test]
  fn text_recognize_image_level_returns_empty_list() {
    // Two-arg form with a level (Line / Word / Character) returns an
    // empty list when no text is found.
    assert_eq!(
      interpret("TextRecognize[Image[{{0.5, 0.6}}], \"Line\"]").unwrap(),
      "{}"
    );
    assert_eq!(
      interpret("TextRecognize[Image[{{0.5, 0.6}}], \"Word\"]").unwrap(),
      "{}"
    );
  }
  #[test]
  fn number_signs() {
    assert_eq!(interpret("NumberSigns[x]").unwrap(), "NumberSigns[x]");
  }
  #[test]
  fn weierstrass_zeta() {
    assert_eq!(
      interpret("WeierstrassZeta[x, y]").unwrap(),
      "WeierstrassZeta[x, y]"
    );
  }
  #[test]
  fn list_surface_plot_3d() {
    assert_eq!(
      interpret("ListSurfacePlot3D[x]").unwrap(),
      "ListSurfacePlot3D[x]"
    );
  }
  #[test]
  fn f_ratio_distribution() {
    assert_eq!(
      interpret("FRatioDistribution[x, y]").unwrap(),
      "FRatioDistribution[x, y]"
    );
  }
  #[test]
  fn date_value() {
    assert_eq!(interpret("DateValue[x]").unwrap(), "DateValue[x]");
  }
  #[test]
  fn density_plot_3d() {
    assert_eq!(
      interpret("DensityPlot3D[x, y]").unwrap(),
      "DensityPlot3D[x, y]"
    );
  }
  #[test]
  fn geo_region_value_plot() {
    assert_eq!(
      interpret("GeoRegionValuePlot[x]").unwrap(),
      "GeoRegionValuePlot[x]"
    );
  }
  #[test]
  fn max_extra_conditions() {
    assert_eq!(
      interpret("MaxExtraConditions[x]").unwrap(),
      "MaxExtraConditions[x]"
    );
  }
  #[test]
  fn time_series_model_fit() {
    assert_eq!(
      interpret("TimeSeriesModelFit[x]").unwrap(),
      "TimeSeriesModelFit[x]"
    );
  }
  #[test]
  fn pane_selector() {
    assert_eq!(interpret("PaneSelector[x]").unwrap(), "PaneSelector[x]");
  }
  #[test]
  fn url_execute() {
    assert_eq!(interpret("URLExecute[x]").unwrap(), "URLExecute[x]");
  }
  #[test]
  fn sequence_position() {
    assert_eq!(
      interpret("SequencePosition[x, y]").unwrap(),
      "SequencePosition[x, y]"
    );
  }
  #[test]
  fn file_base_name() {
    assert_eq!(interpret("FileBaseName[x]").unwrap(), "FileBaseName[x]");
  }
  #[test]
  fn coordinates_tool_options() {
    assert_eq!(
      interpret("CoordinatesToolOptions[x]").unwrap(),
      "CoordinatesToolOptions[x]"
    );
  }
  #[test]
  fn color_combine() {
    assert_eq!(interpret("ColorCombine[x]").unwrap(), "ColorCombine[x]");
  }
  #[test]
  fn highlighted() {
    assert_eq!(interpret("Highlighted[x]").unwrap(), "Highlighted[x]");
  }
  #[test]
  fn text_grid() {
    assert_eq!(interpret("TextGrid[x]").unwrap(), "TextGrid[x]");
  }
  #[test]
  fn numeric_function() {
    assert_eq!(
      interpret("NumericFunction[x]").unwrap(),
      "NumericFunction[x]"
    );
  }
  #[test]
  fn scrollbars() {
    assert_eq!(interpret("Scrollbars[x]").unwrap(), "Scrollbars[x]");
  }
  #[test]
  fn color_setter() {
    assert_eq!(interpret("ColorSetter[x]").unwrap(), "ColorSetter[x]");
  }
  #[test]
  fn distance_matrix() {
    assert_eq!(interpret("DistanceMatrix[x]").unwrap(), "DistanceMatrix[x]");
  }
  #[test]
  fn inverse_wavelet_transform() {
    assert_eq!(
      interpret("InverseWaveletTransform[x]").unwrap(),
      "InverseWaveletTransform[x]"
    );
  }
  #[test]
  fn tree_graph() {
    assert_eq!(interpret("TreeGraph[x]").unwrap(), "TreeGraph[x]");
  }
  #[test]
  fn tree_graph_edges_renders() {
    let result = interpret("TreeGraph[{1  2, 1  3}]").unwrap();
    assert!(
      result.contains("-Graphics-"),
      "TreeGraph should render as Graphics, got: {}",
      result
    );
  }

  // ─── DuplicateFreeQ ───────────────────────────────────────────────
  #[test]
  fn duplicate_free_q_true() {
    assert_eq!(interpret("DuplicateFreeQ[{1, 2, 3}]").unwrap(), "True");
  }
  #[test]
  fn duplicate_free_q_false() {
    assert_eq!(interpret("DuplicateFreeQ[{1, 2, 1}]").unwrap(), "False");
  }
  #[test]
  fn duplicate_free_q_empty() {
    assert_eq!(interpret("DuplicateFreeQ[{}]").unwrap(), "True");
  }
  // DuplicateFreeQ[list, test]: True iff no pair i<j has test[e_i, e_j] True.
  #[test]
  fn duplicate_free_q_with_test() {
    assert_eq!(
      interpret("DuplicateFreeQ[{1, 2, 3}, Greater]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("DuplicateFreeQ[{3, 2, 1}, Greater]").unwrap(),
      "False"
    );
    // Custom equivalence: 1 and -1 collide under Abs.
    assert_eq!(
      interpret("DuplicateFreeQ[{1, -1, 2}, Abs[#1] == Abs[#2] &]").unwrap(),
      "False"
    );
    // Parity equivalence: 1 and 3 collide.
    assert_eq!(
      interpret("DuplicateFreeQ[{1, 3, 5}, Mod[#1, 2] == Mod[#2, 2] &]")
        .unwrap(),
      "False"
    );
    assert_eq!(interpret("DuplicateFreeQ[{5}, Greater]").unwrap(), "True");
  }

  // ─── Unevaluated batch 10 ─────────────────────────────────────────
  #[test]
  fn pade_approximant() {
    assert_eq!(
      interpret("PadeApproximant[x, y]").unwrap(),
      "PadeApproximant[x, y]"
    );
  }
  #[test]
  fn filling_transform() {
    assert_eq!(
      interpret("FillingTransform[x]").unwrap(),
      "FillingTransform[x]"
    );
  }
  #[test]
  fn sampling_period() {
    assert_eq!(interpret("SamplingPeriod[x]").unwrap(), "SamplingPeriod[x]");
  }
  #[test]
  fn find_cycle() {
    assert_eq!(interpret("FindCycle[x]").unwrap(), "FindCycle[x]");
  }
  #[test]
  fn time_series_forecast() {
    assert_eq!(
      interpret("TimeSeriesForecast[x]").unwrap(),
      "TimeSeriesForecast[x]"
    );
  }
  #[test]
  fn cube() {
    assert_eq!(interpret("Cube[x]").unwrap(), "Cube[x]");
  }
  #[test]
  fn characteristic_function() {
    assert_eq!(
      interpret("CharacteristicFunction[x]").unwrap(),
      "CharacteristicFunction[x]"
    );
  }
  #[test]
  fn permutation_replace() {
    assert_eq!(
      interpret("PermutationReplace[x, y]").unwrap(),
      "PermutationReplace[x, y]"
    );
  }
  #[test]
  fn discrete_variables() {
    assert_eq!(
      interpret("DiscreteVariables[x]").unwrap(),
      "DiscreteVariables[x]"
    );
  }
  #[test]
  fn strip_on_input() {
    assert_eq!(interpret("StripOnInput[x]").unwrap(), "StripOnInput[x]");
  }
  #[test]
  fn standardize() {
    assert_eq!(interpret("Standardize[x]").unwrap(), "Standardize[x]");
  }
  #[test]
  fn sub_minus() {
    assert_eq!(interpret("SubMinus[x]").unwrap(), "SubMinus[x]");
  }
  #[test]
  fn corner_neighbors() {
    assert_eq!(
      interpret("CornerNeighbors[x]").unwrap(),
      "CornerNeighbors[x]"
    );
  }
  #[test]
  fn triangular_distribution() {
    assert_eq!(
      interpret("TriangularDistribution[x]").unwrap(),
      "TriangularDistribution[x]"
    );
  }
  #[test]
  fn real_exponent() {
    assert_eq!(interpret("RealExponent[x]").unwrap(), "RealExponent[x]");
  }
  #[test]
  fn color_quantize() {
    assert_eq!(interpret("ColorQuantize[x]").unwrap(), "ColorQuantize[x]");
  }
  #[test]
  fn binary_write() {
    assert_eq!(interpret("BinaryWrite[x]").unwrap(), "BinaryWrite[x]");
  }
  #[test]
  fn checkbox_bar() {
    assert_eq!(interpret("CheckboxBar[x]").unwrap(), "CheckboxBar[x]");
  }
  #[test]
  fn tooltip_delay() {
    assert_eq!(interpret("TooltipDelay[x]").unwrap(), "TooltipDelay[x]");
  }
  #[test]
  fn random_permutation() {
    assert_eq!(
      interpret("RandomPermutation[x]").unwrap(),
      "RandomPermutation[x]"
    );
  }
  #[test]
  fn watershed_components() {
    assert_eq!(
      interpret("WatershedComponents[x]").unwrap(),
      "WatershedComponents[x]"
    );
  }
  #[test]
  fn factorial_moment() {
    assert_eq!(
      interpret("FactorialMoment[x]").unwrap(),
      "FactorialMoment[x]"
    );
  }
  #[test]
  fn view_center() {
    assert_eq!(interpret("ViewCenter[x]").unwrap(), "ViewCenter[x]");
  }
  #[test]
  fn quantile_plot() {
    assert_eq!(interpret("QuantilePlot[x]").unwrap(), "QuantilePlot[x]");
  }
  #[test]
  fn fourier_sin_series() {
    assert_eq!(
      interpret("FourierSinSeries[x, y, z]").unwrap(),
      "FourierSinSeries[x, y, z]"
    );
  }
  #[test]
  fn mathieu_characteristic_a() {
    assert_eq!(
      interpret("MathieuCharacteristicA[x, y]").unwrap(),
      "MathieuCharacteristicA[x, y]"
    );
  }
  #[test]
  fn file_type() {
    assert_eq!(interpret("FileType[x]").unwrap(), "FileType[x]");
  }
  #[test]
  fn stieltjes_gamma() {
    assert_eq!(interpret("StieltjesGamma[x]").unwrap(), "StieltjesGamma[x]");
  }
  #[test]
  fn polar_ticks() {
    assert_eq!(interpret("PolarTicks[x]").unwrap(), "PolarTicks[x]");
  }
  #[test]
  fn beckmann_distribution() {
    assert_eq!(
      interpret("BeckmannDistribution[x]").unwrap(),
      "BeckmannDistribution[x]"
    );
  }
  #[test]
  fn first_case() {
    assert_eq!(interpret("FirstCase[x, y]").unwrap(), "Missing[NotFound]");
  }
  #[test]
  fn first_case_with_pattern() {
    assert_eq!(
      interpret(r#"FirstCase[{1, "hello", 2}, _String]"#).unwrap(),
      "hello"
    );
  }
  #[test]
  fn first_case_integer() {
    assert_eq!(interpret("FirstCase[{a, 1, b, 2}, _Integer]").unwrap(), "1");
  }
  #[test]
  fn first_case_no_match() {
    assert_eq!(
      interpret("FirstCase[{1, 2, 3}, _String]").unwrap(),
      "Missing[NotFound]"
    );
  }
  #[test]
  fn first_case_with_default() {
    assert_eq!(
      interpret(r#"FirstCase[{1, 2, 3}, _String, "none"]"#).unwrap(),
      "none"
    );
  }
  #[test]
  fn first_case_with_condition() {
    assert_eq!(
      interpret("FirstCase[{1, 2, 3, 4}, x_ /; x > 2]").unwrap(),
      "3"
    );
  }
  #[test]
  fn first_case_with_rule_delayed() {
    assert_eq!(
      interpret("FirstCase[{1, 2, 3, 4}, x_ /; x > 2 :> x^2]").unwrap(),
      "9"
    );
  }
  #[test]
  fn first_case_with_levelspec() {
    // 4-arg form FirstCase[expr, patt, default, levelspec].
    assert_eq!(
      interpret("FirstCase[{1, 2, 3, 4}, _?EvenQ, x, Infinity]").unwrap(),
      "2"
    );
    // Level {2}: search one level deeper, inside the sublist.
    assert_eq!(
      interpret("FirstCase[{1, {2, 3}, 4}, _?EvenQ, x, {2}]").unwrap(),
      "2"
    );
    // Level 2 means levels 1..2; the first integer in traversal order.
    assert_eq!(
      interpret("FirstCase[{{1, 2}, {3, 4}}, _Integer, x, 2]").unwrap(),
      "1"
    );
    // No match at the requested level returns the default.
    assert_eq!(
      interpret("FirstCase[{1, 3, 5}, _?EvenQ, missing, {1}]").unwrap(),
      "missing"
    );
  }
  // On an association, FirstCase tests the values and returns the first match.
  #[test]
  fn first_case_association() {
    assert_eq!(
      interpret("FirstCase[<|a -> 1, b -> 2, c -> 4|>, _?EvenQ]").unwrap(),
      "2"
    );
  }
  #[test]
  fn first_case_association_no_match_default() {
    assert_eq!(
      interpret(r#"FirstCase[<|a -> 1, b -> 3|>, _?EvenQ, "none"]"#).unwrap(),
      "none"
    );
  }
  #[test]
  fn first_case_association_condition() {
    assert_eq!(
      interpret("FirstCase[<|a -> 1, b -> 2|>, x_ /; x > 1]").unwrap(),
      "2"
    );
  }
  // The rule (pattern -> rhs) form applies on associations too.
  #[test]
  fn first_case_association_rule() {
    assert_eq!(
      interpret(r#"FirstCase[<|a -> 1, b -> 2|>, _?EvenQ -> "found"]"#)
        .unwrap(),
      "found"
    );
  }
  #[test]
  fn weierstrass_sigma() {
    assert_eq!(
      interpret("WeierstrassSigma[x, y]").unwrap(),
      "WeierstrassSigma[x, y]"
    );
  }
  #[test]
  fn mathieu_c() {
    assert_eq!(interpret("MathieuC[x, y, z]").unwrap(), "MathieuC[x, y, z]");
  }
  #[test]
  fn string_replace_part() {
    // Basic single range replacement
    assert_eq!(
      interpret("StringReplacePart[\"abcdefghijk\", \"XY\", {2, 5}]").unwrap(),
      "aXYfghijk"
    );
    // Replace with empty string (deletion)
    assert_eq!(
      interpret("StringReplacePart[\"abcdef\", \"\", {3, 4}]").unwrap(),
      "abef"
    );
    // Multiple ranges with same replacement
    assert_eq!(
      interpret(
        "StringReplacePart[\"abcdefghijk\", \"XY\", {{1, 1}, {3, 5}, {-3, -1}}]"
      )
      .unwrap(),
      "XYbXYfghXY"
    );
    // Different replacements for each range
    assert_eq!(
      interpret(
        "StringReplacePart[\"abcdef\", {\"X\", \"Y\"}, {{1, 2}, {5, 6}}]"
      )
      .unwrap(),
      "XcdY"
    );
    // Negative indices
    assert_eq!(
      interpret("StringReplacePart[\"abcdef\", \"XY\", {-3, -1}]").unwrap(),
      "abcXY"
    );
    // Non-string first arg returns unevaluated
    assert_eq!(
      interpret("StringReplacePart[x, y, z]").unwrap(),
      "StringReplacePart[x, y, z]"
    );
  }
  #[test]
  fn meta_information() {
    assert_eq!(
      interpret("MetaInformation[x]").unwrap(),
      "MetaInformation[x]"
    );
  }
  #[test]
  fn notebook_save() {
    assert_eq!(interpret("NotebookSave[x]").unwrap(), "NotebookSave[x]");
  }
  #[test]
  fn list_contour_plot_3d() {
    assert_eq!(
      interpret("ListContourPlot3D[x]").unwrap(),
      "ListContourPlot3D[x]"
    );
  }

  // ─── Haversine ─────────────────────────────────────────────────────
  #[test]
  fn haversine_zero() {
    assert_eq!(interpret("Haversine[0]").unwrap(), "0");
  }
  #[test]
  fn haversine_pi() {
    assert_eq!(interpret("Haversine[Pi]").unwrap(), "1");
  }
  #[test]
  fn haversine_half_pi() {
    assert_eq!(interpret("Haversine[Pi/2]").unwrap(), "1/2");
  }
  #[test]
  fn haversine_symbolic_stays_held() {
    // wolframscript keeps Haversine[x] held; only FunctionExpand rewrites it.
    assert_eq!(interpret("Haversine[x]").unwrap(), "Haversine[x]");
  }
  #[test]
  fn haversine_exact_rational_angles() {
    // Nice-angle arguments where (1 - Cos[x])/2 reduces to a rational are
    // evaluated eagerly, matching wolframscript.
    assert_eq!(interpret("Haversine[Pi/3]").unwrap(), "1/4");
    assert_eq!(interpret("Haversine[2 Pi/3]").unwrap(), "3/4");
    assert_eq!(interpret("Haversine[2 Pi]").unwrap(), "0");
    assert_eq!(interpret("Haversine[3 Pi/2]").unwrap(), "1/2");
    assert_eq!(interpret("Haversine[-Pi/3]").unwrap(), "1/4");
    // An exact integer argument has no rational Haversine, so it stays held
    // (Cos[1] does not simplify).
    assert_eq!(interpret("Haversine[1]").unwrap(), "Haversine[1]");
  }
  #[test]
  fn haversine_real_matches_wolframscript() {
    // Haversine[1.5] = Sin[1.5/2]^2 ≈ 0.4646313991661485. The exact last ULP
    // is platform-dependent (system libm differs across OSes; Linux CI gives
    // ...854), so compare numerically rather than by exact string.
    let val: f64 = interpret("Haversine[1.5]").unwrap().parse().unwrap();
    assert!((val - 0.4646313991661485).abs() < 1e-12);
  }
  #[test]
  fn inverse_haversine_zero() {
    assert_eq!(interpret("InverseHaversine[0]").unwrap(), "0");
  }
  #[test]
  fn inverse_haversine_one() {
    assert_eq!(interpret("InverseHaversine[1]").unwrap(), "Pi");
  }

  // ─── Unevaluated batch 11 ─────────────────────────────────────────
  #[test]
  fn resampling_method() {
    assert_eq!(
      interpret("ResamplingMethod[x]").unwrap(),
      "ResamplingMethod[x]"
    );
  }
  #[test]
  fn angular_gauge() {
    assert_eq!(interpret("AngularGauge[x]").unwrap(), "AngularGauge[x]");
  }
  #[test]
  fn angular_gauge_with_range_renders_graphics() {
    assert_eq!(
      interpret("AngularGauge[5.5, {0, 10}]").unwrap(),
      "-Graphics-"
    );
    assert_eq!(
      interpret("Head[AngularGauge[5.5, {0, 10}]]").unwrap(),
      "Graphics"
    );
  }
  #[test]
  fn angular_gauge_accepts_options() {
    // Audit case: ScaleDivisions option on AngularGauge.
    assert_eq!(
      interpret("AngularGauge[5.5, {0, 10}, ScaleDivisions -> Automatic]")
        .unwrap(),
      "-Graphics-"
    );
    assert_eq!(
      interpret("AngularGauge[5.5, {0, 10}, ScaleDivisions -> {10, 2}]")
        .unwrap(),
      "-Graphics-"
    );
  }
  #[test]
  fn color_replace() {
    assert_eq!(interpret("ColorReplace[x]").unwrap(), "ColorReplace[x]");
  }
  #[test]
  fn graph_plot_3d() {
    assert_eq!(interpret("GraphPlot3D[x]").unwrap(), "GraphPlot3D[x]");
  }
  #[test]
  fn button_function() {
    assert_eq!(interpret("ButtonFunction[x]").unwrap(), "ButtonFunction[x]");
  }
  #[test]
  fn sunday() {
    assert_eq!(interpret("Sunday[x]").unwrap(), "Sunday[x]");
  }
  #[test]
  fn frobenius_solve() {
    assert_eq!(
      interpret("FrobeniusSolve[x, y]").unwrap(),
      "FrobeniusSolve[x, y]"
    );
  }
  #[test]
  fn image_value() {
    assert_eq!(interpret("ImageValue[x]").unwrap(), "ImageValue[x]");
  }
  #[test]
  fn generated_parameters() {
    assert_eq!(
      interpret("GeneratedParameters[x]").unwrap(),
      "GeneratedParameters[x]"
    );
  }
  #[test]
  fn plot_region() {
    assert_eq!(interpret("PlotRegion[x]").unwrap(), "PlotRegion[x]");
  }
  #[test]
  fn matrix_log() {
    assert_eq!(interpret("MatrixLog[x]").unwrap(), "MatrixLog[x]");
  }
  #[test]
  fn density_histogram() {
    assert_eq!(
      interpret("DensityHistogram[x]").unwrap(),
      "DensityHistogram[x]"
    );
  }
  #[test]
  fn distribution_chart() {
    assert_eq!(
      interpret("DistributionChart[x]").unwrap(),
      "DistributionChart[x]"
    );
  }
  #[test]
  fn inverse_z_transform() {
    // x is constant with respect to y: x*DiscreteDelta[z], verified
    // against wolframscript (this previously asserted the
    // unimplemented-stub behavior InverseZTransform[x, y, z])
    assert_eq!(
      interpret("InverseZTransform[x, y, z]").unwrap(),
      "x*DiscreteDelta[z]"
    );
  }
  #[test]
  fn incidence_matrix() {
    assert_eq!(
      interpret("IncidenceMatrix[x]").unwrap(),
      "IncidenceMatrix[x]"
    );
  }
  #[test]
  fn notebooks() {
    assert_eq!(interpret("Notebooks[x]").unwrap(), "Notebooks[x]");
  }
  #[test]
  fn z_transform() {
    // x is constant with respect to y: Z{x} = (x*z)/(-1 + z), verified
    // against wolframscript (this previously asserted the
    // unimplemented-stub behavior ZTransform[x, y, z])
    assert_eq!(interpret("ZTransform[x, y, z]").unwrap(), "(x*z)/(-1 + z)");
  }
  #[test]
  fn least_squares() {
    assert_eq!(
      interpret("LeastSquares[x, y]").unwrap(),
      "LeastSquares[x, y]"
    );
  }
  #[test]
  fn feature_types() {
    assert_eq!(interpret("FeatureTypes[x]").unwrap(), "FeatureTypes[x]");
  }
  #[test]
  fn covariance_function() {
    assert_eq!(
      interpret("CovarianceFunction[x, y]").unwrap(),
      "CovarianceFunction[x, y]"
    );
  }
  #[test]
  fn xyz_color() {
    assert_eq!(interpret("XYZColor[x]").unwrap(), "XYZColor[x]");
  }
  #[test]
  fn graph_highlight_style() {
    assert_eq!(
      interpret("GraphHighlightStyle[x]").unwrap(),
      "GraphHighlightStyle[x]"
    );
  }
  #[test]
  fn image_trim() {
    assert_eq!(interpret("ImageTrim[x]").unwrap(), "ImageTrim[x]");
  }
  #[test]
  fn b_spline_surface() {
    assert_eq!(interpret("BSplineSurface[x]").unwrap(), "BSplineSurface[x]");
  }
  #[test]
  fn singular_value_list() {
    assert_eq!(
      interpret("SingularValueList[x]").unwrap(),
      "SingularValueList[x]"
    );
  }
  #[test]
  fn morphological_binarize() {
    assert_eq!(
      interpret("MorphologicalBinarize[x]").unwrap(),
      "MorphologicalBinarize[x]"
    );
  }
  #[test]
  fn vertex_weight() {
    assert_eq!(interpret("VertexWeight[x]").unwrap(), "VertexWeight[x]");
  }
  #[test]
  fn single_letter_italics() {
    assert_eq!(
      interpret("SingleLetterItalics[x]").unwrap(),
      "SingleLetterItalics[x]"
    );
  }
  #[test]
  fn polar_grid_lines() {
    assert_eq!(interpret("PolarGridLines[x]").unwrap(), "PolarGridLines[x]");
  }
  #[test]
  fn root_approximant() {
    assert_eq!(
      interpret("RootApproximant[x]").unwrap(),
      "RootApproximant[x]"
    );
  }
  #[test]
  fn interpretation() {
    assert_eq!(interpret("Interpretation[x]").unwrap(), "Interpretation[x]");
  }
  #[test]
  fn symmetric_group() {
    assert_eq!(interpret("SymmetricGroup[x]").unwrap(), "SymmetricGroup[x]");
  }
  #[test]
  fn databin() {
    assert_eq!(interpret("Databin[x]").unwrap(), "Databin[x]");
  }
  #[test]
  fn inverse_erf() {
    // InverseErf[0] = 0
    assert_eq!(interpret("InverseErf[0]").unwrap(), "0");
    // InverseErf[1] = Infinity
    assert_eq!(interpret("InverseErf[1]").unwrap(), "Infinity");
    // InverseErf[-1] = -Infinity
    assert_eq!(interpret("InverseErf[-1]").unwrap(), "-Infinity");
    // Symbolic — returns unevaluated (Wolfram does not simplify symbolically)
    assert_eq!(interpret("InverseErf[x]").unwrap(), "InverseErf[x]");
    assert_eq!(interpret("InverseErf[-x]").unwrap(), "InverseErf[-x]");
    // Out of range returns unevaluated
    assert_eq!(interpret("InverseErf[2]").unwrap(), "InverseErf[2]");
    assert_eq!(interpret("InverseErf[-2]").unwrap(), "InverseErf[-2]");
    // Numeric evaluation
    let result = interpret("InverseErf[0.5]").unwrap();
    assert!(result.starts_with("0.476936"), "InverseErf[0.5] = {result}");
    let result = interpret("InverseErf[-0.5]").unwrap();
    assert!(
      result.starts_with("-0.476936"),
      "InverseErf[-0.5] = {result}"
    );
    let result = interpret("InverseErf[0.9]").unwrap();
    assert!(result.starts_with("1.16308"), "InverseErf[0.9] = {result}");
    // Real-valued boundaries: 1.0 / -1.0 also yield ±Infinity.
    assert_eq!(interpret("InverseErf[1.0]").unwrap(), "Infinity");
    assert_eq!(interpret("InverseErf[-1.0]").unwrap(), "-Infinity");
    // InverseErfc analogues: 0 / 2 (and their Real forms) → ±Infinity.
    assert_eq!(interpret("InverseErfc[0]").unwrap(), "Infinity");
    assert_eq!(interpret("InverseErfc[0.0]").unwrap(), "Infinity");
    assert_eq!(interpret("InverseErfc[2]").unwrap(), "-Infinity");
    assert_eq!(interpret("InverseErfc[2.0]").unwrap(), "-Infinity");
  }
  #[test]
  fn smooth_density_histogram() {
    assert_eq!(
      interpret("SmoothDensityHistogram[x]").unwrap(),
      "SmoothDensityHistogram[x]"
    );
  }
  #[test]
  fn net_extract() {
    assert_eq!(interpret("NetExtract[x]").unwrap(), "NetExtract[x]");
  }
  #[test]
  fn hankel_h1() {
    assert_eq!(interpret("HankelH1[x, y]").unwrap(), "HankelH1[x, y]");
  }
  #[test]
  fn friday() {
    assert_eq!(interpret("Friday[x]").unwrap(), "Friday[x]");
  }
  #[test]
  fn cloud_import() {
    assert_eq!(interpret("CloudImport[x]").unwrap(), "CloudImport[x]");
  }
  #[test]
  fn temporary() {
    assert_eq!(interpret("Temporary[x]").unwrap(), "Temporary[x]");
  }
  #[test]
  fn service_connect() {
    assert_eq!(interpret("ServiceConnect[x]").unwrap(), "ServiceConnect[x]");
  }
  #[test]
  fn nonlinear_state_space_model() {
    assert_eq!(
      interpret("NonlinearStateSpaceModel[x]").unwrap(),
      "NonlinearStateSpaceModel[x]"
    );
  }
  #[test]
  fn closing() {
    assert_eq!(interpret("Closing[x]").unwrap(), "Closing[x]");
  }
  #[test]
  fn default_duration() {
    assert_eq!(
      interpret("DefaultDuration[x]").unwrap(),
      "DefaultDuration[x]"
    );
  }
  #[test]
  fn from_polar_coordinates_symbolic() {
    assert_eq!(
      interpret("FromPolarCoordinates[{r, theta}]").unwrap(),
      "{r*Cos[theta], r*Sin[theta]}"
    );
  }
  #[test]
  fn from_polar_coordinates_numeric() {
    assert_eq!(
      interpret("FromPolarCoordinates[{2, Pi/4}]").unwrap(),
      "{Sqrt[2], Sqrt[2]}"
    );
  }

  #[test]
  fn from_polar_coordinates_3d() {
    // Hyperspherical {r, t, p}: x1 = r Cos[t], x2 = r Sin[t] Cos[p],
    // x3 = r Sin[t] Sin[p]. (Woxi's Times canonicalisation orders the
    // two Sin factors by argument, so Sin[p] precedes Sin[t]; the
    // expression is identical to wolframscript's r*Sin[t]*Sin[p].)
    assert_eq!(
      interpret("FromPolarCoordinates[{r, t, p}]").unwrap(),
      "{r*Cos[t], r*Cos[p]*Sin[t], r*Sin[p]*Sin[t]}"
    );
  }

  #[test]
  fn to_polar_coordinates_3d() {
    // ToPolarCoordinates[{x, y, z}] -> {Sqrt[x^2+y^2+z^2],
    // ArcCos[x/Sqrt[x^2+y^2+z^2]], ArcTan[y, z]}.
    assert_eq!(
      interpret("ToPolarCoordinates[{x, y, z}]").unwrap(),
      "{Sqrt[x^2 + y^2 + z^2], ArcCos[x/Sqrt[x^2 + y^2 + z^2]], ArcTan[y, z]}"
    );
  }

  #[test]
  fn to_polar_coordinates_4d() {
    // 4-D hyperspherical: r, ArcCos[x/r], ArcCos[y/Sqrt[y²+z²+w²]],
    // ArcTan[z, w]. Inner radical drops the leading coordinates.
    assert_eq!(
      interpret("ToPolarCoordinates[{x, y, z, w}]").unwrap(),
      "{Sqrt[w^2 + x^2 + y^2 + z^2], ArcCos[x/Sqrt[w^2 + x^2 + y^2 + z^2]], ArcCos[y/Sqrt[w^2 + y^2 + z^2]], ArcTan[z, w]}"
    );
  }

  #[test]
  fn from_polar_coordinates_4d() {
    // 4-dim hyperspherical: extra Sin[p] Cos[q] / Sin[p] Sin[q] factors.
    assert_eq!(
      interpret("FromPolarCoordinates[{r, t, p, q}]").unwrap(),
      "{r*Cos[t], r*Cos[p]*Sin[t], r*Cos[q]*Sin[p]*Sin[t], r*Sin[p]*Sin[q]*Sin[t]}"
    );
  }
  #[test]
  fn to_polar_coordinates_symbolic() {
    assert_eq!(
      interpret("ToPolarCoordinates[{x, y}]").unwrap(),
      "{Sqrt[x^2 + y^2], ArcTan[x, y]}"
    );
  }
  #[test]
  fn to_polar_coordinates_numeric() {
    assert_eq!(
      interpret("ToPolarCoordinates[{1, 1}]").unwrap(),
      "{Sqrt[2], Pi/4}"
    );
  }
  #[test]
  fn from_spherical_coordinates() {
    assert_eq!(
      interpret("FromSphericalCoordinates[{r, theta, phi}]").unwrap(),
      "{r*Cos[phi]*Sin[theta], r*Sin[phi]*Sin[theta], r*Cos[theta]}"
    );
  }
  #[test]
  fn to_spherical_coordinates() {
    assert_eq!(
      interpret("ToSphericalCoordinates[{x, y, z}]").unwrap(),
      "{Sqrt[x^2 + y^2 + z^2], ArcTan[z, Sqrt[x^2 + y^2]], ArcTan[x, y]}"
    );
  }
  #[test]
  fn qr_decomposition_identity() {
    assert_eq!(
      interpret("QRDecomposition[{{1, 0}, {0, 1}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}}"
    );
  }
  #[test]
  fn qr_decomposition_3x3() {
    assert_eq!(
      interpret(
        "QRDecomposition[{{12, -51, 4}, {6, 167, -68}, {-4, 24, -41}}]"
      )
      .unwrap(),
      "{{{6/7, 3/7, -2/7}, {-69/175, 158/175, 6/35}, {-58/175, 6/175, -33/35}}, {{14, 21, -14}, {0, 175, -70}, {0, 0, 35}}}"
    );
  }
  #[test]
  fn continued_fraction_k_basic() {
    assert_eq!(
      interpret("ContinuedFractionK[k, {k, 1, 3}]").unwrap(),
      "7/10"
    );
  }
  #[test]
  fn continued_fraction_k_constant() {
    assert_eq!(
      interpret("ContinuedFractionK[1, {i, 1, 10}]").unwrap(),
      "55/89"
    );
  }
  #[test]
  fn continued_fraction_k_five() {
    assert_eq!(
      interpret("ContinuedFractionK[k, {k, 1, 5}]").unwrap(),
      "157/225"
    );
  }
  #[test]
  fn continued_fraction_k_three_arg() {
    // Three-argument form: K[f, g, {n, nmin, nmax}] uses f as the numerators
    // and g as the denominators.
    assert_eq!(
      interpret("ContinuedFractionK[1, n, {n, 1, 5}]").unwrap(),
      "157/225"
    );
    assert_eq!(
      interpret("ContinuedFractionK[2 n - 1, n^2, {n, 1, 4}]").unwrap(),
      "228/379"
    );
    assert_eq!(
      interpret("ContinuedFractionK[1, n, {n, 1, 1}]").unwrap(),
      "1"
    );
    // Symbolic numerator/denominator.
    assert_eq!(
      interpret("ContinuedFractionK[a, b, {n, 1, 2}]").unwrap(),
      "a/(a/b + b)"
    );
  }
  #[test]
  fn continued_fraction_k_infinity_constant_one() {
    // K_n=1^∞ 1/1 = 1/(1+1/(1+1/(1+...))) is the fixed point of x = 1/(1+x),
    // which equals -1 + GoldenRatio.
    assert_eq!(
      interpret("ContinuedFractionK[1, {n, 1, Infinity}]").unwrap(),
      "-1 + GoldenRatio"
    );
  }
  #[test]
  fn counts_by_odd_q() {
    assert_eq!(
      interpret("CountsBy[{1, 2, 3, 4, 5}, OddQ]").unwrap(),
      "<|True -> 3, False -> 2|>"
    );
  }
  #[test]
  fn counts_by_string_length() {
    assert_eq!(
      interpret(
        "CountsBy[{\"a\", \"bb\", \"c\", \"dd\", \"eee\"}, StringLength]"
      )
      .unwrap(),
      "<|1 -> 2, 2 -> 2, 3 -> 1|>"
    );
  }
  #[test]
  fn counts_by_lambda() {
    assert_eq!(
      interpret(
        r#"CountsBy[{"apple", "ant", "banana", "berry"}, StringTake[#, 1] &]"#
      )
      .unwrap(),
      "<|a -> 2, b -> 2|>"
    );
  }
  #[test]
  fn find_linear_recurrence_fibonacci() {
    assert_eq!(
      interpret("FindLinearRecurrence[{1, 1, 2, 3, 5, 8, 13}]").unwrap(),
      "{1, 1}"
    );
  }
  #[test]
  fn find_linear_recurrence_powers_of_2() {
    assert_eq!(
      interpret("FindLinearRecurrence[{1, 2, 4, 8, 16, 32}]").unwrap(),
      "{2}"
    );
  }
  #[test]
  fn sss_triangle_345() {
    assert_eq!(
      interpret("SSSTriangle[3, 4, 5]").unwrap(),
      "Triangle[{{0, 0}, {5, 0}, {16/5, 12/5}}]"
    );
  }
  #[test]
  fn sss_triangle_equilateral() {
    assert_eq!(
      interpret("SSSTriangle[1, 1, 1]").unwrap(),
      "Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}]"
    );
  }
  #[test]
  fn fold_pair_list_add_mul() {
    assert_eq!(
      interpret("FoldPairList[{#1 + #2, #1*#2}&, 1, {1, 2, 3}]").unwrap(),
      "{2, 3, 5}"
    );
  }
  #[test]
  fn fold_pair_list_sub() {
    assert_eq!(
      interpret("FoldPairList[{#1 + #2, #1 - #2}&, 0, {1, 2, 3}]").unwrap(),
      "{1, 1, 0}"
    );
  }
  // FoldPairList[f, x, list, g] emits g applied to the whole {emit, newState}
  // pair instead of just the first element.
  #[test]
  fn fold_pair_list_four_arg_identity() {
    assert_eq!(
      interpret(
        "FoldPairList[QuotientRemainder, 498, {100, 50, 20, 5, 1}, Identity]"
      )
      .unwrap(),
      "{{4, 98}, {1, 48}, {2, 8}, {1, 3}, {3, 0}}"
    );
  }
  #[test]
  fn fold_pair_list_four_arg_pure_function() {
    assert_eq!(
      interpret("FoldPairList[{#1 + #2, #1*#2}&, 2, {3, 4}, # + 100 &]")
        .unwrap(),
      "{{105, 106}, {110, 124}}"
    );
  }
  #[test]
  fn fold_pair_list_four_arg_symbolic() {
    assert_eq!(
      interpret("FoldPairList[{#1 + #2, #1*#2}&, 2, {3, 4}, f]").unwrap(),
      "{f[{5, 6}], f[{10, 24}]}"
    );
  }
  // FoldPair folds f[state, e] -> {emit, newState} and returns the last emit.
  #[test]
  fn fold_pair_add_mul() {
    assert_eq!(
      interpret("FoldPair[{#1 + #2, #1*#2}&, 1, {2, 3}]").unwrap(),
      "5"
    );
  }
  #[test]
  fn fold_pair_list_head_function() {
    // f = List makes each step emit the state and carry the element forward.
    assert_eq!(interpret("FoldPair[List, 1, {2, 3, 4}]").unwrap(), "3");
  }
  #[test]
  fn fold_pair_constant_state() {
    assert_eq!(
      interpret("FoldPair[{#1 - #2, #2}&, 100, {10, 20, 30}]").unwrap(),
      "-10"
    );
  }
  // The 4-argument form returns g applied to the final {emit, state} pair.
  #[test]
  fn fold_pair_post_function() {
    assert_eq!(
      interpret("FoldPair[{2 #1, #1 + #2}&, 1, {10, 20}, g]").unwrap(),
      "g[{22, 31}]"
    );
  }
  // FoldPair on an empty list stays unevaluated.
  #[test]
  fn fold_pair_empty_list() {
    assert_eq!(
      interpret("FoldPair[{#1, #2}&, x, {}]").unwrap(),
      "FoldPair[{#1, #2} & , x, {}]"
    );
  }
  #[test]
  fn join_across_basic() {
    assert_eq!(
      interpret(
        "JoinAcross[{<|\"a\" -> 1, \"b\" -> 2|>}, {<|\"a\" -> 1, \"c\" -> 3|>}, \"a\"]"
      )
      .unwrap(),
      "{<|a -> 1, b -> 2, c -> 3|>}"
    );
  }
  #[test]
  fn join_across_multi() {
    assert_eq!(
      interpret(
        "JoinAcross[{<|\"a\" -> 1, \"b\" -> 2|>, <|\"a\" -> 2, \"b\" -> 3|>}, {<|\"a\" -> 1, \"c\" -> 10|>, <|\"a\" -> 2, \"c\" -> 20|>}, \"a\"]"
      )
      .unwrap(),
      "{<|a -> 1, b -> 2, c -> 10|>, <|a -> 2, b -> 3, c -> 20|>}"
    );
  }
  #[test]
  fn exponential_moving_average_real() {
    assert_eq!(
      interpret("ExponentialMovingAverage[{1, 2, 3, 4, 5}, 0.5]").unwrap(),
      "{1, 1.5, 2.25, 3.125, 4.0625}"
    );
  }
  #[test]
  fn exponential_moving_average_rational() {
    assert_eq!(
      interpret("ExponentialMovingAverage[{1, 2, 3, 4, 5}, 1/3]").unwrap(),
      "{1, 4/3, 17/9, 70/27, 275/81}"
    );
  }
  #[test]
  fn letter_counts_basic() {
    assert_eq!(
      interpret("LetterCounts[\"hello world\"]").unwrap(),
      "<|l -> 3, o -> 2, d -> 1, r -> 1, w -> 1, e -> 1, h -> 1|>"
    );
  }

  #[test]
  fn letter_counts_ties_use_reverse_first_occurrence() {
    // Repeated letters with equal counts are ordered by reverse first
    // occurrence (matching CharacterCounts and wolframscript): in
    // "mississippi" s first appears after i, so s -> 4 precedes i -> 4.
    assert_eq!(
      interpret("LetterCounts[\"mississippi\"]").unwrap(),
      "<|s -> 4, i -> 4, p -> 2, m -> 1|>"
    );
    assert_eq!(
      interpret("LetterCounts[\"tooth\"]").unwrap(),
      "<|o -> 2, t -> 2, h -> 1|>"
    );
    assert_eq!(
      interpret("LetterCounts[\"xxyyzz\"]").unwrap(),
      "<|z -> 2, y -> 2, x -> 2|>"
    );
  }
  #[test]
  fn text_words_basic() {
    assert_eq!(
      interpret("TextWords[\"the cat sat on the mat\"]").unwrap(),
      "{the, cat, sat, on, the, mat}"
    );
  }
  #[test]
  fn text_words_with_punctuation() {
    assert_eq!(
      interpret("TextWords[\"Hello, World! This is a test.\"]").unwrap(),
      "{Hello, World, This, is, a, test}"
    );
  }
  #[test]
  fn text_words_keeps_internal_punctuation() {
    // Internal hyphens/apostrophes are part of the word.
    assert_eq!(interpret("TextWords[\"YT-1300\"]").unwrap(), "{YT-1300}");
    assert_eq!(
      interpret("TextWords[\"don't stop\"]").unwrap(),
      "{don't, stop}"
    );
  }
  #[test]
  fn text_words_count_limit() {
    // TextWords[s, n] returns the first n words.
    assert_eq!(
      interpret("TextWords[\"first second third fourth\", 2]").unwrap(),
      "{first, second}"
    );
  }
  #[test]
  fn text_words_count_exceeds_length() {
    assert_eq!(interpret("TextWords[\"a b c\", 5]").unwrap(), "{a, b, c}");
  }
  #[test]
  fn word_counts_basic() {
    assert_eq!(
      interpret("WordCounts[\"the cat sat on the mat\"]").unwrap(),
      "<|the -> 2, mat -> 1, on -> 1, sat -> 1, cat -> 1|>"
    );
  }
  #[test]
  fn word_counts_strips_punctuation() {
    // Trailing punctuation must not make "fish," and "fish" distinct words.
    assert_eq!(
      interpret("WordCounts[\"one fish, two fish, red fish\"]").unwrap(),
      "<|fish -> 3, red -> 1, two -> 1, one -> 1|>"
    );
  }
  #[test]
  fn word_counts_keeps_internal_punctuation() {
    // Internal hyphens/apostrophes are part of the word.
    assert_eq!(
      interpret("WordCounts[\"YT-1300 and don't stop\"]").unwrap(),
      "<|stop -> 1, don't -> 1, and -> 1, YT-1300 -> 1|>"
    );
  }
  #[test]
  fn word_counts_bigrams() {
    assert_eq!(
      interpret("WordCounts[\"the fox jumped over the hare.\", 2]").unwrap(),
      "<|{the, hare} -> 1, {over, the} -> 1, {jumped, over} -> 1, \
       {fox, jumped} -> 1, {the, fox} -> 1|>"
    );
  }
  #[test]
  fn word_counts_bigrams_with_repeats() {
    assert_eq!(
      interpret("WordCounts[\"a b a b a\", 2]").unwrap(),
      "<|{b, a} -> 2, {a, b} -> 2|>"
    );
  }
  #[test]
  fn word_counts_ngram_too_long() {
    assert_eq!(interpret("WordCounts[\"hi\", 5]").unwrap(), "<||>");
  }
  // WordFrequency[text, word] gives the fraction of words equal to `word`.
  #[test]
  fn word_frequency_basic() {
    assert_eq!(
      interpret("WordFrequency[\"a b a c\", \"a\"]").unwrap(),
      "0.5"
    );
  }
  #[test]
  fn word_frequency_word_list() {
    assert_eq!(
      interpret("WordFrequency[\"a b a c\", {\"a\", \"c\"}]").unwrap(),
      "<|a -> 0.5, c -> 0.25|>"
    );
  }
  #[test]
  fn word_frequency_case_sensitive() {
    assert_eq!(
      interpret("WordFrequency[\"The cat\", \"the\"]").unwrap(),
      "0."
    );
  }
  #[test]
  fn word_frequency_ignore_case() {
    let v: f64 =
      interpret("WordFrequency[\"The cat the\", \"the\", IgnoreCase -> True]")
        .unwrap()
        .parse()
        .unwrap();
    assert!((v - 2.0 / 3.0).abs() < 1e-12, "got {}", v);
  }
  #[test]
  fn word_frequency_absent_word() {
    assert_eq!(interpret("WordFrequency[\"a b c\", \"z\"]").unwrap(), "0.");
  }
  #[test]
  fn text_sentences_basic() {
    assert_eq!(
      interpret(
        "TextSentences[\"This is a sentence.  This is another sentence.\"]"
      )
      .unwrap(),
      "{This is a sentence., This is another sentence.}"
    );
  }
  #[test]
  fn text_sentences_mixed_terminators() {
    assert_eq!(
      interpret("TextSentences[\"Hello world! How are you? I am fine.\"]")
        .unwrap(),
      "{Hello world!, How are you?, I am fine.}"
    );
  }
  #[test]
  fn text_sentences_abbreviations() {
    assert_eq!(
      interpret(
        "TextSentences[\"Dr. Smith went to Washington. He arrived at 3 p.m. on Tuesday.\"]"
      )
      .unwrap(),
      "{Dr. Smith went to Washington., He arrived at 3 p.m. on Tuesday.}"
    );
  }
  #[test]
  fn text_sentences_abbreviation_before_capital() {
    // "p.m." does not end a sentence even before a capitalized word
    assert_eq!(
      interpret(
        "TextSentences[\"He arrived at 3 p.m. He left at 4 a.m. the next day.\"]"
      )
      .unwrap(),
      "{He arrived at 3 p.m. He left at 4 a.m. the next day.}"
    );
  }
  #[test]
  fn text_sentences_titles_and_initials() {
    assert_eq!(
      interpret("TextSentences[\"Mr. Jones met Mrs. Smith. They talked.\"]")
        .unwrap(),
      "{Mr. Jones met Mrs. Smith., They talked.}"
    );
    assert_eq!(
      interpret(
        "TextSentences[\"J. R. R. Tolkien wrote books. They are long.\"]"
      )
      .unwrap(),
      "{J. R. R. Tolkien wrote books., They are long.}"
    );
  }
  #[test]
  fn text_sentences_internal_periods() {
    assert_eq!(
      interpret("TextSentences[\"The U.S.A. is big. It has 50 states.\"]")
        .unwrap(),
      "{The U.S.A. is big., It has 50 states.}"
    );
  }
  #[test]
  fn text_sentences_ellipsis_and_punctuation_runs() {
    assert_eq!(
      interpret("TextSentences[\"What?! Really?? Yes... I think so.\"]")
        .unwrap(),
      "{What?!, Really??, Yes... I think so.}"
    );
  }
  #[test]
  fn text_sentences_decimal_not_boundary() {
    assert_eq!(
      interpret("TextSentences[\"I paid $5.50 for it. Then I left.\"]")
        .unwrap(),
      "{I paid $5.50 for it., Then I left.}"
    );
  }
  #[test]
  fn text_sentences_number_abbreviation() {
    assert_eq!(
      interpret("TextSentences[\"No. 5 is missing. Check again.\"]").unwrap(),
      "{No. 5 is missing., Check again.}"
    );
  }
  #[test]
  fn text_sentences_lowercase_after_period_splits() {
    assert_eq!(
      interpret("TextSentences[\"This is fine. okay then.\"]").unwrap(),
      "{This is fine., okay then.}"
    );
  }
  #[test]
  fn text_sentences_closing_quote() {
    assert_eq!(
      interpret("TextSentences[\"He said \\\"Stop!\\\" Then he ran.\"]")
        .unwrap(),
      "{He said \"Stop!\", Then he ran.}"
    );
  }
  #[test]
  fn text_sentences_first_n() {
    assert_eq!(
      interpret(
        "TextSentences[\"This is a sentence.  This is another sentence.\", 1]"
      )
      .unwrap(),
      "{This is a sentence.}"
    );
    // n larger than the number of sentences returns all of them
    assert_eq!(
      interpret("TextSentences[\"Hello world! How are you?\", 5]").unwrap(),
      "{Hello world!, How are you?}"
    );
  }
  #[test]
  fn text_sentences_no_terminator_and_empty() {
    assert_eq!(
      interpret("TextSentences[\"One sentence only\"]").unwrap(),
      "{One sentence only}"
    );
    assert_eq!(interpret("TextSentences[\"\"]").unwrap(), "{}");
  }
  #[test]
  fn text_sentences_invalid_args_stay_unevaluated() {
    // Non-positive n: TextSentences::arg2 message, unevaluated
    assert_eq!(
      interpret("TextSentences[\"A b c. D e f.\", 0]").unwrap(),
      "TextSentences[A b c. D e f., 0]"
    );
    // Non-string first argument: TextSentences::arg1 message, unevaluated
    assert_eq!(interpret("TextSentences[42]").unwrap(), "TextSentences[42]");
  }
  #[test]
  fn orthogonal_matrix_q_identity() {
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{1, 0}, {0, 1}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn orthogonal_matrix_q_rotation() {
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{0, -1}, {1, 0}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn orthogonal_matrix_q_non_orthogonal() {
    assert_eq!(
      interpret("OrthogonalMatrixQ[{{1, 1}, {0, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn circle_through_basic() {
    assert_eq!(
      interpret("CircleThrough[{{0, 0}, {1, 0}, {0, 1}}]").unwrap(),
      "Circle[{1/2, 1/2}, 1/Sqrt[2]]"
    );
  }
  #[test]
  fn circle_through_unit() {
    assert_eq!(
      interpret("CircleThrough[{{1, 0}, {-1, 0}, {0, 1}}]").unwrap(),
      "Circle[{0, 0}, 1]"
    );
  }
  #[test]
  fn numerical_sort_basic() {
    assert_eq!(
      interpret("NumericalSort[{\"b3\", \"a1\", \"c2\", \"a10\"}]").unwrap(),
      "{a1, a10, b3, c2}"
    );
  }
  #[test]
  fn numerical_sort_numbers() {
    assert_eq!(
      interpret("NumericalSort[{\"file10\", \"file2\", \"file1\"}]").unwrap(),
      "{file1, file10, file2}"
    );
  }
  #[test]
  fn from_coefficient_rules_basic() {
    assert_eq!(
      interpret("FromCoefficientRules[{{0} -> 1, {1} -> 3, {2} -> 5}, x]")
        .unwrap(),
      "1 + 3*x + 5*x^2"
    );
  }
  // Multivariate form: the variable spec is a list and each exponent vector
  // selects powers of the corresponding variable.
  #[test]
  fn from_coefficient_rules_multivariate() {
    assert_eq!(
      interpret("FromCoefficientRules[{{2, 0} -> 1, {0, 2} -> 1}, {x, y}]")
        .unwrap(),
      "x^2 + y^2"
    );
    assert_eq!(
      interpret("FromCoefficientRules[{{1, 1} -> 4}, {x, y}]").unwrap(),
      "4*x*y"
    );
    assert_eq!(
      interpret("FromCoefficientRules[{{2, 0, 1} -> 3}, {x, y, z}]").unwrap(),
      "3*x^2*z"
    );
    // A constant term (all-zero exponent vector) drops every variable factor.
    assert_eq!(
      interpret("FromCoefficientRules[{{0, 0} -> 7}, {x, y}]").unwrap(),
      "7"
    );
    // An empty rule list reconstructs to 0.
    assert_eq!(interpret("FromCoefficientRules[{}, {x, y}]").unwrap(), "0");
    // Round-trips with CoefficientRules.
    assert_eq!(
      interpret(
        "FromCoefficientRules[CoefficientRules[(x + y)^3, {x, y}], {x, y}]"
      )
      .unwrap(),
      "x^3 + 3*x^2*y + 3*x*y^2 + y^3"
    );
  }
  // PolynomialExtendedGCD skipped - requires polynomial GCD infrastructure
  #[test]
  fn count_distinct_basic() {
    assert_eq!(interpret("CountDistinct[{1, 2, 3, 2, 1, 4}]").unwrap(), "4");
  }
  #[test]
  fn count_distinct_strings() {
    assert_eq!(
      interpret("CountDistinct[{\"a\", \"b\", \"a\", \"c\"}]").unwrap(),
      "3"
    );
  }
  #[test]
  fn count_distinct_by() {
    // Count distinct values of f applied to each element.
    assert_eq!(interpret("CountDistinctBy[{1, -1, 2}, Abs]").unwrap(), "2");
    assert_eq!(
      interpret("CountDistinctBy[{1, 2, 3, 4, 5}, EvenQ]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret(
        "CountDistinctBy[{\"a\", \"bb\", \"cc\", \"d\"}, StringLength]"
      )
      .unwrap(),
      "2"
    );
    assert_eq!(interpret("CountDistinctBy[{}, Abs]").unwrap(), "0");
    assert_eq!(interpret("CountDistinctBy[{1, 2, 3}, #^2 &]").unwrap(), "3");
  }
  #[test]
  fn diagonalizable_matrix_q_true() {
    assert_eq!(
      interpret("DiagonalizableMatrixQ[{{1, 0}, {0, 2}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn diagonalizable_matrix_q_false() {
    assert_eq!(
      interpret("DiagonalizableMatrixQ[{{1, 1}, {0, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn positive_semidefinite_matrix_q_true() {
    assert_eq!(
      interpret("PositiveSemidefiniteMatrixQ[{{1, 0}, {0, 1}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn positive_semidefinite_matrix_q_false() {
    assert_eq!(
      interpret("PositiveSemidefiniteMatrixQ[{{1, 2}, {2, 1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn symmetric_polynomial_1() {
    assert_eq!(
      interpret("SymmetricPolynomial[1, {a, b, c}]").unwrap(),
      "a + b + c"
    );
  }
  #[test]
  fn symmetric_polynomial_2() {
    assert_eq!(
      interpret("SymmetricPolynomial[2, {a, b, c}]").unwrap(),
      "a*b + a*c + b*c"
    );
  }
  #[test]
  fn symmetric_polynomial_3() {
    assert_eq!(
      interpret("SymmetricPolynomial[3, {a, b, c}]").unwrap(),
      "a*b*c"
    );
  }
  #[test]
  fn adjugate_2x2() {
    assert_eq!(
      interpret("Adjugate[{{1, 2}, {3, 4}}]").unwrap(),
      "{{4, -2}, {-3, 1}}"
    );
  }
  #[test]
  fn adjugate_3x3() {
    assert_eq!(
      interpret("Adjugate[{{1, 2, 3}, {0, 4, 5}, {1, 0, 6}}]").unwrap(),
      "{{24, -12, -2}, {5, 3, -5}, {-4, 2, 4}}"
    );
  }
  #[test]
  fn coordinate_bounds_basic() {
    assert_eq!(
      interpret("CoordinateBounds[{{1, 5}, {3, 2}, {-1, 7}}]").unwrap(),
      "{{-1, 3}, {2, 7}}"
    );
  }
  #[test]
  fn coordinate_bounds_doc_example() {
    assert_eq!(
      interpret("CoordinateBounds[{{0, 1}, {1, 2}, {2, 1}, {3, 2}, {4, 0}}]")
        .unwrap(),
      "{{0, 4}, {0, 2}}"
    );
  }
  #[test]
  fn coordinate_bounds_scalar_pad() {
    assert_eq!(
      interpret(
        "CoordinateBounds[{{0, 1}, {1, 2}, {2, 1}, {3, 2}, {4, 0}}, 1]"
      )
      .unwrap(),
      "{{-1, 5}, {-1, 3}}"
    );
  }
  #[test]
  fn coordinate_bounds_scalar_pad_zero() {
    assert_eq!(
      interpret("CoordinateBounds[{{1, 5}, {3, 2}, {-1, 7}}, 0]").unwrap(),
      "{{-1, 3}, {2, 7}}"
    );
  }
  #[test]
  fn coordinate_bounds_scaled_rational() {
    assert_eq!(
      interpret("CoordinateBounds[{{1, 5}, {3, 2}, {-1, 7}}, Scaled[1/2]]")
        .unwrap(),
      "{{-3, 5}, {-1/2, 19/2}}"
    );
  }
  #[test]
  fn coordinate_bounds_list_pad() {
    assert_eq!(
      interpret(
        "CoordinateBounds[{{0, 1}, {1, 2}, {2, 1}, {3, 2}, {4, 0}}, {1, 2}]"
      )
      .unwrap(),
      "{{-1, 5}, {-2, 4}}"
    );
  }
  #[test]
  fn coordinate_bounds_pair_pad() {
    assert_eq!(
      interpret(
        "CoordinateBounds[{{0, 1}, {1, 2}, {2, 1}, {3, 2}, {4, 0}}, {{1, 2}, {3, 4}}]"
      )
      .unwrap(),
      "{{-1, 6}, {-3, 6}}"
    );
  }
  #[test]
  fn coordinate_bounds_pair_pad_simple() {
    assert_eq!(
      interpret(
        "CoordinateBounds[{{1, 5}, {3, 2}, {-1, 7}}, {{0, 1}, {2, 3}}]"
      )
      .unwrap(),
      "{{-1, 4}, {0, 10}}"
    );
  }
  #[test]
  fn coordinate_bounds_3d_basic() {
    assert_eq!(
      interpret("CoordinateBounds[{{1,2,3}, {4,5,6}, {0,3,9}}]").unwrap(),
      "{{0, 4}, {2, 5}, {3, 9}}"
    );
  }
  #[test]
  fn coordinate_bounds_3d_list_pad() {
    assert_eq!(
      interpret("CoordinateBounds[{{1,2,3}, {4,5,6}, {0,3,9}}, {1, 2, 3}]")
        .unwrap(),
      "{{-1, 5}, {0, 7}, {0, 12}}"
    );
  }
  #[test]
  fn coordinate_bounds_mixed_scaled_in_list() {
    assert_eq!(
      interpret(
        "CoordinateBounds[{{0, 1}, {1, 2}, {2, 1}, {3, 2}, {4, 0}}, {Scaled[1/4], 1}]"
      )
      .unwrap(),
      "{{-1, 5}, {-1, 3}}"
    );
  }
  #[test]
  fn glaisher_symbolic() {
    assert_eq!(interpret("Glaisher").unwrap(), "Glaisher");
  }
  #[test]
  fn glaisher_numeric() {
    assert_eq!(interpret("N[Glaisher]").unwrap(), "1.2824271291006226");
  }
  #[test]
  fn nminvalue_basic() {
    assert_eq!(interpret("NMinValue[x^2 + 3*x + 2, x]").unwrap(), "-0.25");
  }
  #[test]
  fn nmaxvalue_basic() {
    assert_eq!(interpret("NMaxValue[-x^2 + 3*x + 2, x]").unwrap(), "4.25");
  }
  #[test]
  fn find_arg_min_basic() {
    assert_eq!(interpret("FindArgMin[x^2 + 3*x + 2, x]").unwrap(), "{-1.5}");
  }
  #[test]
  fn find_arg_max_basic() {
    assert_eq!(interpret("FindArgMax[-x^2 + 3*x + 2, x]").unwrap(), "{1.5}");
  }

  #[test]
  fn nmaxvalue_linear_on_unit_disk() {
    // Maximize x - 2y on x^2 + y^2 <= 1 → Sqrt[5]
    let result =
      interpret("NMaxValue[{x - 2*y, x^2 + y^2 <= 1}, {x, y}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 5.0_f64.sqrt()).abs() < 1e-9, "got {}", val);
  }

  #[test]
  fn nminvalue_linear_on_unit_disk() {
    let result =
      interpret("NMinValue[{x - 2*y, x^2 + y^2 <= 1}, {x, y}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val + 5.0_f64.sqrt()).abs() < 1e-9, "got {}", val);
  }

  #[test]
  fn nargmax_linear_on_unit_disk() {
    let result =
      interpret("NArgMax[{x - 2*y, x^2 + y^2 <= 1}, {x, y}]").unwrap();
    assert!(result.starts_with("{"), "got {}", result);
    let inner = &result[1..result.len() - 1];
    let parts: Vec<f64> =
      inner.split(", ").map(|s| s.parse().unwrap()).collect();
    assert!((parts[0] - 1.0 / 5.0_f64.sqrt()).abs() < 1e-9);
    assert!((parts[1] + 2.0 / 5.0_f64.sqrt()).abs() < 1e-9);
  }

  #[test]
  fn nargmin_linear_on_unit_disk() {
    let result =
      interpret("NArgMin[{x - 2*y, x^2 + y^2 <= 1}, {x, y}]").unwrap();
    let inner = &result[1..result.len() - 1];
    let parts: Vec<f64> =
      inner.split(", ").map(|s| s.parse().unwrap()).collect();
    assert!((parts[0] + 1.0 / 5.0_f64.sqrt()).abs() < 1e-9);
    assert!((parts[1] - 2.0 / 5.0_f64.sqrt()).abs() < 1e-9);
  }

  #[test]
  fn findargmax_linear_on_unit_disk() {
    let result =
      interpret("FindArgMax[{x - 2*y, x^2 + y^2 <= 1}, {x, y}]").unwrap();
    let inner = &result[1..result.len() - 1];
    let parts: Vec<f64> =
      inner.split(", ").map(|s| s.parse().unwrap()).collect();
    assert!((parts[0] - 1.0 / 5.0_f64.sqrt()).abs() < 1e-9);
    assert!((parts[1] + 2.0 / 5.0_f64.sqrt()).abs() < 1e-9);
  }

  #[test]
  fn findargmin_linear_on_unit_disk() {
    // Audit case: FindArgMin[{-x + 2y, x^2+y^2<=1}, {x, y}] → (0.447, -0.894)
    let result =
      interpret("FindArgMin[{-x + 2*y, x^2 + y^2 <= 1}, {x, y}]").unwrap();
    let inner = &result[1..result.len() - 1];
    let parts: Vec<f64> =
      inner.split(", ").map(|s| s.parse().unwrap()).collect();
    assert!((parts[0] - 1.0 / 5.0_f64.sqrt()).abs() < 1e-9);
    assert!((parts[1] + 2.0 / 5.0_f64.sqrt()).abs() < 1e-9);
  }

  // Symbolic constrained Minimize/Maximize: linear objective on a quadratic
  // disk constraint has a closed-form solution from Lagrange multipliers.
  #[test]
  fn minimize_linear_on_unit_disk_symbolic() {
    assert_eq!(
      interpret("Minimize[{x - 2*y, x^2 + y^2 <= 1}, {x, y}]").unwrap(),
      "{-Sqrt[5], {x -> -(1/Sqrt[5]), y -> 2/Sqrt[5]}}"
    );
  }

  #[test]
  fn maximize_linear_on_unit_disk_symbolic() {
    assert_eq!(
      interpret("Maximize[{x - 2*y, x^2 + y^2 <= 1}, {x, y}]").unwrap(),
      "{Sqrt[5], {x -> 1/Sqrt[5], y -> -2/Sqrt[5]}}"
    );
  }

  // When a^2 + b^2 is a perfect square, Sqrt simplifies and the result is
  // rational rather than involving Sqrt.
  #[test]
  fn minimize_linear_on_unit_disk_rational() {
    assert_eq!(
      interpret("Minimize[{3*x + 4*y, x^2 + y^2 <= 1}, {x, y}]").unwrap(),
      "{-5, {x -> -3/5, y -> -4/5}}"
    );
  }

  #[test]
  fn maximize_linear_on_unit_disk_rational() {
    assert_eq!(
      interpret("Maximize[{3*x + 4*y, x^2 + y^2 <= 1}, {x, y}]").unwrap(),
      "{5, {x -> 3/5, y -> 4/5}}"
    );
  }

  // Larger radius: constraint x^2+y^2 <= 4 gives R = 2 and value 2*Sqrt[5].
  #[test]
  fn minimize_linear_on_radius_two_disk() {
    assert_eq!(
      interpret("Minimize[{x - 2*y, x^2 + y^2 <= 4}, {x, y}]").unwrap(),
      "{-2*Sqrt[5], {x -> -2/Sqrt[5], y -> 4/Sqrt[5]}}"
    );
  }

  #[test]
  fn string_replace_list_basic() {
    assert_eq!(
      interpret("StringReplaceList[\"abcabc\", \"a\" -> \"X\"]").unwrap(),
      "{Xbcabc, abcXbc}"
    );
  }
  #[test]
  fn string_replace_list_overlap() {
    assert_eq!(
      interpret("StringReplaceList[\"aaa\", \"aa\" -> \"X\"]").unwrap(),
      "{Xa, aX}"
    );
  }
  #[test]
  fn chessboard_distance_basic() {
    assert_eq!(
      interpret("ChessboardDistance[{1, 2}, {3, 5}]").unwrap(),
      "3"
    );
  }
  #[test]
  fn chessboard_distance_3d() {
    assert_eq!(
      interpret("ChessboardDistance[{1, 2, 3}, {4, 6, 5}]").unwrap(),
      "4"
    );
  }
  #[test]
  fn negative_definite_matrix_q_true() {
    assert_eq!(
      interpret("NegativeDefiniteMatrixQ[{{-2, 0}, {0, -3}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn negative_definite_matrix_q_false() {
    assert_eq!(
      interpret("NegativeDefiniteMatrixQ[{{-1, 0}, {0, 0}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn negative_semidefinite_matrix_q_true() {
    assert_eq!(
      interpret("NegativeSemidefiniteMatrixQ[{{-1, 0}, {0, 0}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn negative_semidefinite_matrix_q_false() {
    assert_eq!(
      interpret("NegativeSemidefiniteMatrixQ[{{1, 0}, {0, -1}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn sequence_count_basic() {
    assert_eq!(
      interpret("SequenceCount[{1, 2, 3, 1, 2, 3, 1}, {1, 2}]").unwrap(),
      "2"
    );
  }
  #[test]
  fn sequence_count_no_match() {
    assert_eq!(interpret("SequenceCount[{1, 2, 3}, {4, 5}]").unwrap(), "0");
  }
  #[test]
  fn sequence_count_default_non_overlapping() {
    assert_eq!(interpret("SequenceCount[{1, 1, 1}, {1, 1}]").unwrap(), "1");
    assert_eq!(
      interpret("SequenceCount[{1, 1, 1}, {1, 1}, Overlaps -> False]").unwrap(),
      "1"
    );
  }
  #[test]
  fn sequence_count_overlaps_true() {
    assert_eq!(
      interpret("SequenceCount[{1, 1, 1}, {1, 1}, Overlaps -> True]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("SequenceCount[{1, 2, 1, 2, 1}, {1, 2}, Overlaps -> True]")
        .unwrap(),
      "2"
    );
  }
  #[test]
  fn sequence_count_overlaps_all() {
    assert_eq!(
      interpret("SequenceCount[{1, 1, 1, 1}, {1, 1}, Overlaps -> All]")
        .unwrap(),
      "3"
    );
  }
  // Pattern element matching.
  #[test]
  fn sequence_count_blank_sequence_runs() {
    // {__Symbol} counts maximal runs of consecutive symbols.
    assert_eq!(
      interpret(
        "SequenceCount[{1, 2, a, b, 3, c, d, 4, 5, 6, e, f, g, 7}, {__Symbol}]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("SequenceCount[{a, b, 1, c}, {__Symbol}]").unwrap(),
      "2"
    );
  }
  #[test]
  fn sequence_count_single_blank_pattern() {
    // {_Symbol} matches one symbol at a time.
    assert_eq!(
      interpret("SequenceCount[{1, a, 2, b, 3}, {_Symbol}]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("SequenceCount[{a, b, c}, {_Symbol}]").unwrap(),
      "3"
    );
  }
  #[test]
  fn sequence_count_multi_blank_window() {
    // Two single-element patterns form a fixed window of length 2.
    assert_eq!(
      interpret("SequenceCount[{1, 2, 3}, {_Integer, _Integer}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("SequenceCount[{1, a, 2, b}, {_, _}]").unwrap(),
      "2"
    );
  }
  #[test]
  fn sequence_count_mixed_literal_pattern() {
    assert_eq!(
      interpret("SequenceCount[{1, a, 1, b, 2}, {1, _}]").unwrap(),
      "2"
    );
  }
  #[test]
  fn sequence_position_basic() {
    assert_eq!(
      interpret("SequencePosition[{1, 2, 3, 1, 2}, {1, 2}]").unwrap(),
      "{{1, 2}, {4, 5}}"
    );
  }
  #[test]
  fn sequence_position_no_match() {
    assert_eq!(
      interpret("SequencePosition[{1, 2, 3}, {4, 5}]").unwrap(),
      "{}"
    );
  }
  #[test]
  fn sequence_position_overlapping() {
    assert_eq!(
      interpret("SequencePosition[{1, 1, 1}, {1, 1}]").unwrap(),
      "{{1, 2}, {2, 3}}"
    );
  }
  #[test]
  fn sequence_position_overlaps_false() {
    // The default is Overlaps -> True; Overlaps -> False skips past matches.
    assert_eq!(
      interpret("SequencePosition[{1, 1, 1}, {1, 1}, Overlaps -> False]")
        .unwrap(),
      "{{1, 2}}"
    );
    assert_eq!(
      interpret("SequencePosition[{1, 2, 1, 2, 1}, {1, 2}, Overlaps -> False]")
        .unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }
  #[test]
  fn sequence_position_overlaps_true_explicit() {
    assert_eq!(
      interpret("SequencePosition[{1, 1, 1}, {1, 1}, Overlaps -> True]")
        .unwrap(),
      "{{1, 2}, {2, 3}}"
    );
  }
  #[test]
  fn sequence_position_symbolic() {
    assert_eq!(
      interpret("SequencePosition[{a, b, c, a, b, c}, {a, b}]").unwrap(),
      "{{1, 2}, {4, 5}}"
    );
  }
  // SequencePosition matches blank patterns, not just literals.
  #[test]
  fn sequence_position_blank_pattern() {
    assert_eq!(
      interpret("SequencePosition[{a, b, c}, {x_}]").unwrap(),
      "{{1, 1}, {2, 2}, {3, 3}}"
    );
    assert_eq!(
      interpret("SequencePosition[{1, 2, 3, 4, 5}, {_, _, _}]").unwrap(),
      "{{1, 3}, {2, 4}, {3, 5}}"
    );
  }
  // A Condition pattern is honored.
  #[test]
  fn sequence_position_condition() {
    assert_eq!(
      interpret("SequencePosition[Range[6], {x_, y_} /; y == x + 1]").unwrap(),
      "{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}"
    );
  }
  // A sequence (BlankSequence) pattern reports each matching span.
  #[test]
  fn sequence_position_sequence_pattern() {
    assert_eq!(
      interpret("SequencePosition[{a, 1, 2, 3, b}, {__Integer}]").unwrap(),
      "{{2, 4}, {3, 4}, {4, 4}}"
    );
  }
  // A count limit keeps only the first n matches (overlapping by default).
  #[test]
  fn sequence_position_count_limit() {
    assert_eq!(
      interpret("SequencePosition[{1, 2, 3, 4}, {_, _}, 2]").unwrap(),
      "{{1, 2}, {2, 3}}"
    );
    assert_eq!(
      interpret("SequencePosition[{1, 1, 1, 1}, {1, 1}, 1]").unwrap(),
      "{{1, 2}}"
    );
  }
  #[test]
  fn sequence_cases_literal() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 3, 4, 1, 2}, {1, 2}]").unwrap(),
      "{{1, 2}, {1, 2}}"
    );
  }
  #[test]
  fn sequence_split_single_element_separator() {
    assert_eq!(
      interpret("SequenceSplit[{1, 0, 2, 3, 0, 4}, {0}]").unwrap(),
      "{{1}, {2, 3}, {4}}"
    );
  }
  #[test]
  fn sequence_split_multi_element_separator() {
    assert_eq!(
      interpret("SequenceSplit[{1, 2, 1, 2, 3, 1, 2}, {1, 2}]").unwrap(),
      "{{3}}"
    );
  }
  #[test]
  fn sequence_split_no_match_returns_whole_list() {
    assert_eq!(
      interpret("SequenceSplit[{1, 2, 3}, {5}]").unwrap(),
      "{{1, 2, 3}}"
    );
  }
  #[test]
  fn sequence_split_drops_leading_and_trailing_empty() {
    assert_eq!(interpret("SequenceSplit[{0, 1, 0}, {0}]").unwrap(), "{{1}}");
  }
  #[test]
  fn sequence_split_pattern_separator() {
    assert_eq!(
      interpret("SequenceSplit[{1, 2, 3, 4}, {x_ /; EvenQ[x]}]").unwrap(),
      "{{1}, {3}}"
    );
  }
  #[test]
  fn sequence_split_conditioned_pair_separator() {
    assert_eq!(
      interpret("SequenceSplit[{1, 2, 3, 4, 5, 6}, {a_, b_} /; a + b == 7]")
        .unwrap(),
      "{{1, 2}, {5, 6}}"
    );
  }
  #[test]
  fn sequence_split_empty_list() {
    // No separator match in an empty list yields a single empty segment.
    assert_eq!(interpret("SequenceSplit[{}, {0}]").unwrap(), "{{}}");
  }
  #[test]
  fn sequence_split_all_separators() {
    assert_eq!(interpret("SequenceSplit[{1, 1, 1}, {1}]").unwrap(), "{}");
  }
  #[test]
  fn sequence_split_strings() {
    assert_eq!(
      interpret("SequenceSplit[{\"a\", \"x\", \"b\", \"x\", \"c\"}, {\"x\"}]")
        .unwrap(),
      "{{a}, {b}, {c}}"
    );
  }
  #[test]
  fn sequence_cases_pattern() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 3, 4, 1, 2}, {_, _}]").unwrap(),
      "{{1, 2}, {3, 4}, {1, 2}}"
    );
  }
  #[test]
  fn sequence_cases_blank_sequence() {
    assert_eq!(
      interpret("SequenceCases[{a, 1, 2, b, 3}, {__Integer}]").unwrap(),
      "{{1, 2}, {3}}"
    );
  }
  #[test]
  fn sequence_cases_blank_sequence_all() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 3}, {__Integer}]").unwrap(),
      "{{1, 2, 3}}"
    );
  }
  #[test]
  fn sequence_cases_blank_sequence_none() {
    assert_eq!(
      interpret("SequenceCases[{a, b, c}, {__Integer}]").unwrap(),
      "{}"
    );
  }
  #[test]
  fn sequence_cases_repeated_pattern() {
    assert_eq!(
      interpret("SequenceCases[{a, 1, 2, 3, b, 4, 5}, {Repeated[_Integer]}]")
        .unwrap(),
      "{{1, 2, 3}, {4, 5}}"
    );
  }
  #[test]
  fn sequence_cases_condition() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 3, 4, 5}, {x_, y_} /; x + y > 6]")
        .unwrap(),
      "{{3, 4}}"
    );
  }
  #[test]
  fn sequence_cases_condition_equality() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 3, 4, 5}, {x_, y_} /; x + y == 5]")
        .unwrap(),
      "{{2, 3}}"
    );
  }
  #[test]
  fn sequence_cases_rule_delayed() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 3, 1, 2}, {x_, y_} :> x + y]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn sequence_cases_named_binding_with_repeated_null() {
    // Regression: previously returned `{}` because the `l : {...}` Pattern
    // wrapper wasn't unwrapped when computing match lengths.
    assert_eq!(
      interpret("SequenceCases[{1/2, 1/3, 1/16}, l : {_, 1 ...} :> Length[l]]")
        .unwrap(),
      "{1, 1, 1}"
    );
  }
  #[test]
  fn sequence_cases_three_element_condition() {
    assert_eq!(
      interpret("SequenceCases[Range[10], {x_, y_, z_} /; x + y == z]")
        .unwrap(),
      "{{1, 2, 3}}"
    );
  }
  // Overlaps -> True reports overlapping subsequence matches.
  #[test]
  fn sequence_cases_overlaps_condition() {
    assert_eq!(
      interpret(
        "SequenceCases[Range[6], {x_, y_} /; y == x + 1, Overlaps -> True]"
      )
      .unwrap(),
      "{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}"
    );
  }
  #[test]
  fn sequence_cases_overlaps_fixed_length() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 3, 4, 5}, {_, _, _}, Overlaps -> True]")
        .unwrap(),
      "{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}"
    );
  }
  #[test]
  fn sequence_cases_overlaps_literal() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 1, 2, 1}, {1, 2}, Overlaps -> True]")
        .unwrap(),
      "{{1, 2}, {1, 2}}"
    );
  }
  // A count limit keeps only the first n matches.
  #[test]
  fn sequence_cases_count_limit() {
    assert_eq!(
      interpret("SequenceCases[{1, 2, 3, 4}, {a_, b_}, 2]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }
  // The count limit and Overlaps combine.
  #[test]
  fn sequence_cases_count_with_overlaps() {
    assert_eq!(
      interpret(
        "SequenceCases[{1, 2, 3, 4, 5, 6}, {_, _}, 2, Overlaps -> True]"
      )
      .unwrap(),
      "{{1, 2}, {2, 3}}"
    );
  }
  #[test]
  fn chebyshev_distance_basic() {
    assert_eq!(interpret("ChebyshevDistance[{1, 2}, {3, 5}]").unwrap(), "3");
  }
  #[test]
  fn hermitian_matrix_q_real_symmetric() {
    assert_eq!(
      interpret("HermitianMatrixQ[{{1, 0}, {0, 2}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn hermitian_matrix_q_false() {
    assert_eq!(
      interpret("HermitianMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn normal_matrix_q_diagonal() {
    assert_eq!(
      interpret("NormalMatrixQ[{{1, 0}, {0, 2}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn normal_matrix_q_false() {
    assert_eq!(
      interpret("NormalMatrixQ[{{1, 1}, {0, 1}}]").unwrap(),
      "False"
    );
  }
  // Matrix predicates return False (not unevaluated) for non-square or
  // non-matrix arguments, matching wolframscript.
  #[test]
  fn normal_matrix_q_non_matrix() {
    assert_eq!(interpret("NormalMatrixQ[5]").unwrap(), "False");
    assert_eq!(interpret("NormalMatrixQ[{1, 2, 3}]").unwrap(), "False");
    assert_eq!(
      interpret("NormalMatrixQ[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn hermitian_matrix_q_non_matrix() {
    assert_eq!(interpret("HermitianMatrixQ[5]").unwrap(), "False");
    assert_eq!(
      interpret("HermitianMatrixQ[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn antihermitian_matrix_q_non_matrix() {
    assert_eq!(interpret("AntihermitianMatrixQ[{1, 2}]").unwrap(), "False");
    assert_eq!(
      interpret("AntihermitianMatrixQ[{{0, 1}, {-1, 0}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn parallel_sum_basic() {
    assert_eq!(interpret("ParallelSum[i^2, {i, 1, 5}]").unwrap(), "55");
  }
  #[test]
  fn parallel_product_basic() {
    assert_eq!(interpret("ParallelProduct[i, {i, 1, 5}]").unwrap(), "120");
  }
  #[test]
  fn mangoldt_lambda_prime() {
    assert_eq!(interpret("MangoldtLambda[7]").unwrap(), "Log[7]");
  }
  #[test]
  fn mangoldt_lambda_prime_power() {
    assert_eq!(interpret("MangoldtLambda[8]").unwrap(), "Log[2]");
  }
  #[test]
  fn mangoldt_lambda_composite() {
    assert_eq!(interpret("MangoldtLambda[6]").unwrap(), "0");
  }
  #[test]
  fn mangoldt_lambda_one() {
    assert_eq!(interpret("MangoldtLambda[1]").unwrap(), "0");
  }
  #[test]
  fn liouville_lambda_basic() {
    assert_eq!(interpret("LiouvilleLambda[6]").unwrap(), "1");
  }
  #[test]
  fn liouville_lambda_prime() {
    assert_eq!(interpret("LiouvilleLambda[7]").unwrap(), "-1");
  }
  #[test]
  fn liouville_lambda_prime_power() {
    assert_eq!(interpret("LiouvilleLambda[8]").unwrap(), "-1");
  }
  // LiouvilleLambda is Listable.
  #[test]
  fn liouville_lambda_threads_over_list() {
    assert_eq!(
      interpret("LiouvilleLambda[{1, 2, 3, 4}]").unwrap(),
      "{1, -1, -1, 1}"
    );
  }
  #[test]
  fn bray_curtis_distance() {
    assert_eq!(
      interpret("BrayCurtisDistance[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "3/7"
    );
  }
  #[test]
  fn canberra_distance() {
    assert_eq!(
      interpret("CanberraDistance[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "143/105"
    );
  }
  #[test]
  fn cosine_distance_orthogonal() {
    assert_eq!(interpret("CosineDistance[{1, 0}, {0, 1}]").unwrap(), "1");
  }

  #[test]
  fn cosine_distance_zero_vector_real() {
    // When one vector is all-zero we bypass 0/0 and return 0. (matches wolframscript)
    assert_eq!(
      interpret("CosineDistance[{0.0, 0.0}, {x, y}]").unwrap(),
      "0."
    );
  }

  #[test]
  fn cosine_distance_zero_vector_integer() {
    // Integer-only zero vector returns exact 0 (matches wolframscript).
    assert_eq!(interpret("CosineDistance[{0, 0}, {1, 2}]").unwrap(), "0");
  }

  #[test]
  fn cosine_distance_symbolic_uses_conjugate() {
    // Regression: CosineDistance uses the Hermitian inner product, so the
    // second vector must be conjugated. Before this fix woxi returned
    // `1 - x/Sqrt[...]` instead of `1 - Conjugate[x]/Sqrt[...]`.
    assert_eq!(
      interpret("CosineDistance[{1, 0}, {x, y}]").unwrap(),
      "1 - Conjugate[x]/Sqrt[Abs[x]^2 + Abs[y]^2]"
    );
  }
  #[test]
  fn cosine_distance_scalar_complex() {
    // CosineDistance for numeric scalars uses (u/|u|)·(Conj(v)/|v|) so that
    // common integer factors cancel — matches wolframscript's canonical form
    // (regression for mathics distance/numeric.py:180).
    assert_eq!(
      interpret("CosineDistance[1 + 2 I, 5]").unwrap(),
      "1 - (1 + 2*I)/Sqrt[5]"
    );
  }
  #[test]
  fn cosine_distance_scalar_zero() {
    assert_eq!(interpret("CosineDistance[0, 5]").unwrap(), "0");
  }
  #[test]
  fn cosine_distance_scalar_symbolic_unevaluated() {
    // Pure-symbolic scalar pairs stay unevaluated, matching wolframscript.
    assert_eq!(
      interpret("CosineDistance[a, b]").unwrap(),
      "CosineDistance[a, b]"
    );
  }
  #[test]
  fn nearest_vector_euclidean() {
    assert_eq!(
      interpret("Nearest[{{0, 1}, {1, 2}, {2, 3}}, {1.1, 2}]").unwrap(),
      "{{1, 2}}"
    );
  }
  #[test]
  fn nearest_rule_returns_labels() {
    // Rule form: distances measured on the points list, result is drawn from
    // the corresponding labels (regression for mathics distance/clusters.py:405).
    assert_eq!(
      interpret("Nearest[{{0, 1}, {1, 2}, {2, 3}} -> {a, b, c}, {1.1, 2}]")
        .unwrap(),
      "{b}"
    );
  }

  // Nearest[points -> "Index", target] returns the 1-based positions
  // of the nearest points instead of the points themselves.
  #[test]
  fn nearest_index_view() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 10} -> \"Index\", 4]").unwrap(),
      "{3}"
    );
  }

  // Nearest[points -> "Distance", target] returns the distance(s) to
  // the nearest point(s).
  #[test]
  fn nearest_distance_view() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 10} -> \"Distance\", 4]").unwrap(),
      "{1.}"
    );
  }

  // Nearest[points -> "Element", target] is equivalent to the
  // points-only form (returns the points themselves).
  #[test]
  fn nearest_element_view() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 10} -> \"Element\", 4]").unwrap(),
      "{3}"
    );
  }

  // Index view honours the `n` argument and ties.
  #[test]
  fn nearest_index_with_count() {
    assert_eq!(
      interpret("Nearest[{1, 2, 3, 10} -> \"Index\", 4, 2]").unwrap(),
      "{3, 2}"
    );
  }

  // Nearest[points -> Automatic, target] labels each point by its 1-based
  // position, so the result is the indices of the nearest points.
  #[test]
  fn nearest_automatic_returns_indices() {
    assert_eq!(
      interpret("Nearest[{1, 5, 10, 15} -> Automatic, 7]").unwrap(),
      "{2}"
    );
    assert_eq!(
      interpret("Nearest[{1, 5, 10, 15} -> Automatic, 7, 2]").unwrap(),
      "{2, 3}"
    );
    assert_eq!(
      interpret("Nearest[{1, 5, 10, 15} -> Automatic, 7, All]").unwrap(),
      "{2, 3, 1, 4}"
    );
  }
  // Strings are compared via EditDistance (Wolfram's default string metric).
  #[test]
  fn nearest_strings_edit_distance() {
    assert_eq!(
      interpret(r#"Nearest[{"aaaa", "abaa", "bbbb", "aaaba"}, "aaba", 3]"#)
        .unwrap(),
      "{aaaa, aaaba, abaa}"
    );
    assert_eq!(
      interpret(r#"Nearest[{"cat", "car", "dog"}, "cot"]"#).unwrap(),
      "{cat}"
    );
    assert_eq!(
      interpret(r#"Nearest[{"cat", "car", "dog", "cab"}, "cat", 2]"#).unwrap(),
      "{cat, car}"
    );
    // The labelled form draws the result from the labels.
    assert_eq!(
      interpret(r#"Nearest[{"cat", "car", "dog"} -> {1, 2, 3}, "cot"]"#)
        .unwrap(),
      "{1}"
    );
  }
  #[test]
  fn key_sort_by_basic() {
    assert_eq!(
      interpret(
        "KeySortBy[<|\"ba\" -> 2, \"a\" -> 1, \"ccc\" -> 3|>, StringLength]"
      )
      .unwrap(),
      "<|a -> 1, ba -> 2, ccc -> 3|>"
    );
  }
  // Regression: KeySortBy must apply pure functions (not just named symbols).
  #[test]
  fn key_sort_by_pure_function() {
    assert_eq!(
      interpret("KeySortBy[<|3 -> a, 1 -> b, 2 -> c|>, -# &]").unwrap(),
      "<|3 -> a, 2 -> c, 1 -> b|>"
    );
    assert_eq!(
      interpret("KeySortBy[<|1 -> a, 2 -> b, 3 -> c|>, Function[k, -k]]")
        .unwrap(),
      "<|3 -> c, 2 -> b, 1 -> a|>"
    );
  }
  // A named function that reorders the keys by magnitude.
  #[test]
  fn key_sort_by_abs() {
    assert_eq!(
      interpret("KeySortBy[<|-3 -> a, 1 -> b, -2 -> c|>, Abs]").unwrap(),
      "<|1 -> b, -2 -> c, -3 -> a|>"
    );
  }
  // A list of functions sorts the keys lexicographically by each criterion.
  #[test]
  fn key_sort_by_multiple_criteria() {
    assert_eq!(
      interpret("KeySortBy[<|1 -> a, 2 -> b, 3 -> c|>, {Mod[#, 2] &, # &}]")
        .unwrap(),
      "<|2 -> b, 1 -> a, 3 -> c|>"
    );
  }
  // Keys are compared numerically, not as strings (10 sorts after 2).
  #[test]
  fn key_sort_by_numeric_order() {
    assert_eq!(
      interpret("KeySortBy[<|10 -> a, 2 -> b, 1 -> c|>, # &]").unwrap(),
      "<|1 -> c, 2 -> b, 10 -> a|>"
    );
  }
  // Equal sort keys are broken by the canonical order of the key itself
  // (like SortBy), not by the association's original order.
  #[test]
  fn key_sort_by_tie_breaks_on_key() {
    assert_eq!(
      interpret("KeySortBy[<|3 -> a, 1 -> b, 2 -> c|>, Mod[#, 2] &]").unwrap(),
      "<|2 -> c, 1 -> b, 3 -> a|>"
    );
  }
  #[test]
  fn max_filter_basic() {
    assert_eq!(
      interpret("MaxFilter[{1, 5, 2, 8, 3}, 1]").unwrap(),
      "{5, 5, 8, 8, 8}"
    );
  }
  #[test]
  fn min_filter_basic() {
    assert_eq!(
      interpret("MinFilter[{1, 5, 2, 8, 3}, 1]").unwrap(),
      "{1, 1, 2, 2, 3}"
    );
  }
  #[test]
  fn median_filter_basic() {
    // MedianFilter[list, r] replaces each element with the median of
    // its range-r neighbourhood; boundary windows are clipped.
    assert_eq!(
      interpret("MedianFilter[{1, 2, 3, 2, 1}, 1]").unwrap(),
      "{3/2, 2, 2, 2, 3/2}"
    );
  }
  #[test]
  fn median_filter_radius_two() {
    // Window of 5 (clipped at the boundaries).
    assert_eq!(
      interpret("MedianFilter[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "{2, 5/2, 3, 7/2, 4}"
    );
  }
  #[test]
  fn median_filter_2d_uses_square_neighbourhood() {
    // 2D MedianFilter uses a (2r+1)×(2r+1) window per cell, clipped at
    // the boundaries — not a per-row 1D window.
    assert_eq!(
      interpret("MedianFilter[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]").unwrap(),
      "{{3, 7/2, 4}, {9/2, 5, 11/2}, {6, 13/2, 7}}"
    );
  }
  #[test]
  fn median_filter_2d_unsorted_input() {
    assert_eq!(
      interpret("MedianFilter[{{1, 5, 2}, {8, 3, 9}, {4, 7, 6}}, 1]").unwrap(),
      "{{4, 4, 4}, {9/2, 5, 11/2}, {11/2, 13/2, 13/2}}"
    );
  }
  #[test]
  fn gradient_filter_basic_radius_1() {
    // 1-D GradientFilter[list, 1]: |central difference| with edge replication.
    // wolframscript: {0.5, 1., 1., 1., 1.5, 0., 1.5, 1., 1., 1., 0.5}
    assert_eq!(
      interpret("GradientFilter[{1, 2, 3, 4, 5, 1, 5, 4, 3, 2, 1}, 1]")
        .unwrap(),
      "{0.5, 1., 1., 1., 1.5, 0., 1.5, 1., 1., 1., 0.5}"
    );
  }
  #[test]
  fn gradient_filter_constant_list() {
    // Constant input → all zeros.
    assert_eq!(
      interpret("GradientFilter[{5, 5, 5, 5}, 1]").unwrap(),
      "{0., 0., 0., 0.}"
    );
  }
  #[test]
  fn gradient_filter_unevaluated_for_non_list() {
    // Non-list, non-image input stays symbolic.
    assert_eq!(
      interpret("GradientFilter[x, 1]").unwrap(),
      "GradientFilter[x, 1]"
    );
  }
  #[test]
  fn upsample_basic() {
    assert_eq!(
      interpret("Upsample[{a, b, c}, 2]").unwrap(),
      "{a, 0, b, 0, c, 0}"
    );
  }
  #[test]
  fn downsample_basic() {
    assert_eq!(
      interpret("Downsample[{a, b, c, d, e, f}, 2]").unwrap(),
      "{a, c, e}"
    );
  }
  #[test]
  fn euler_angles_identity() {
    assert_eq!(
      interpret("EulerAngles[{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]").unwrap(),
      "{0, 0, 0}"
    );
  }
  #[test]
  fn trimmed_mean_basic() {
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 100}, 0.25]").unwrap(),
      "5/2"
    );
  }
  #[test]
  fn trimmed_mean_larger() {
    assert_eq!(
      interpret("TrimmedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]").unwrap(),
      "11/2"
    );
  }
  #[test]
  fn winsorized_mean_basic() {
    assert_eq!(
      interpret("WinsorizedMean[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]")
        .unwrap(),
      "11/2"
    );
  }
  // WinsorizedMean[list, {flow, fhigh}] winsorizes asymmetrically.
  #[test]
  fn winsorized_mean_two_fractions() {
    assert_eq!(
      interpret("WinsorizedMean[Range[10], {1/10, 1/10}]").unwrap(),
      "11/2"
    );
    assert_eq!(
      interpret("WinsorizedMean[Range[10], {2/10, 1/10}]").unwrap(),
      "57/10"
    );
    assert_eq!(
      interpret("WinsorizedMean[Range[20], {1/10, 1/10}]").unwrap(),
      "21/2"
    );
  }
  #[test]
  fn trimmed_variance_basic() {
    assert_eq!(
      interpret("TrimmedVariance[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]")
        .unwrap(),
      "7/2"
    );
  }
  #[test]
  fn trimmed_variance_asymmetric() {
    // Asymmetric trim {f1, f2}: drop the lower fraction f1 (0.2 here
    // drops one element, -10) and keep the upper end intact.
    // Remaining {1, 1, 1, 1, 20} -> sample variance 361/5.
    assert_eq!(
      interpret("TrimmedVariance[{-10, 1, 1, 1, 1, 20}, {0.2, 0}]").unwrap(),
      "361/5"
    );
  }
  #[test]
  fn trimmed_variance_default() {
    // Default form (no fraction) trims 5% from each end. For a 10-element
    // list floor(0.05 * 10) = 0, so the variance equals the raw sample
    // variance.
    assert_eq!(
      interpret("TrimmedVariance[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]").unwrap(),
      "55/6"
    );
  }
  #[test]
  fn winsorized_variance_basic() {
    assert_eq!(
      interpret("WinsorizedVariance[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]")
        .unwrap(),
      "85/18"
    );
  }

  // The winsorized count is Floor[f*n], not a rounded value. With n=5 and
  // f=0.1, Floor[0.5] = 0, so nothing is winsorized and the result is the
  // plain mean/variance (regression: previously rounded 0.5 up to 1).
  #[test]
  fn winsorized_count_uses_floor() {
    assert_eq!(
      interpret("WinsorizedMean[{1, 2, 3, 4, 100}, 0.1]").unwrap(),
      "22"
    );
    // f=0.14 -> Floor[0.7] = 0 as well.
    assert_eq!(
      interpret("WinsorizedMean[{1, 2, 3, 4, 100}, 0.14]").unwrap(),
      "22"
    );
    // f=0.2 -> Floor[1.0] = 1: winsorize one value at each end.
    assert_eq!(
      interpret("WinsorizedMean[{1, 2, 3, 4, 100}, 0.2]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("WinsorizedVariance[{1, 2, 3, 4, 100}, 0.1]").unwrap(),
      "3805/2"
    );
  }

  // EqualTo operator form
  #[test]
  fn equal_to_true() {
    assert_eq!(interpret("EqualTo[5][5]").unwrap(), "True");
  }
  #[test]
  fn equal_to_false() {
    assert_eq!(interpret("EqualTo[5][3]").unwrap(), "False");
  }
  #[test]
  fn equal_to_symbolic() {
    assert_eq!(interpret("EqualTo[5][x]").unwrap(), "x == 5");
  }

  // GreaterThan, LessThan, etc.
  #[test]
  fn greater_than_false() {
    assert_eq!(interpret("GreaterThan[5][3]").unwrap(), "False");
  }
  #[test]
  fn greater_than_true() {
    assert_eq!(interpret("GreaterThan[5][7]").unwrap(), "True");
  }
  #[test]
  fn less_than_true() {
    assert_eq!(interpret("LessThan[5][3]").unwrap(), "True");
  }
  #[test]
  fn less_than_false() {
    assert_eq!(interpret("LessThan[5][7]").unwrap(), "False");
  }
  #[test]
  fn greater_equal_than_true() {
    assert_eq!(interpret("GreaterEqualThan[3][3]").unwrap(), "True");
  }
  #[test]
  fn greater_equal_than_false() {
    assert_eq!(interpret("GreaterEqualThan[5][3]").unwrap(), "False");
  }
  #[test]
  fn less_equal_than_true() {
    assert_eq!(interpret("LessEqualThan[3][3]").unwrap(), "True");
  }
  #[test]
  fn less_equal_than_false() {
    assert_eq!(interpret("LessEqualThan[3][5]").unwrap(), "False");
  }
  #[test]
  fn unequal_to_true() {
    assert_eq!(interpret("UnequalTo[5][3]").unwrap(), "True");
  }
  #[test]
  fn unequal_to_false() {
    assert_eq!(interpret("UnequalTo[5][5]").unwrap(), "False");
  }

  // FileNameDrop
  #[test]
  fn file_name_drop_default() {
    assert_eq!(interpret("FileNameDrop[\"a/b/c/d.txt\"]").unwrap(), "a/b/c");
  }
  #[test]
  fn file_name_drop_positive() {
    assert_eq!(
      interpret("FileNameDrop[\"a/b/c/d.txt\", 1]").unwrap(),
      "b/c/d.txt"
    );
  }
  #[test]
  fn file_name_drop_positive_2() {
    assert_eq!(
      interpret("FileNameDrop[\"a/b/c/d.txt\", 2]").unwrap(),
      "c/d.txt"
    );
  }
  #[test]
  fn file_name_drop_negative() {
    assert_eq!(
      interpret("FileNameDrop[\"a/b/c/d.txt\", -1]").unwrap(),
      "a/b/c"
    );
  }
  #[test]
  fn file_name_drop_negative_2() {
    assert_eq!(
      interpret("FileNameDrop[\"a/b/c/d.txt\", -2]").unwrap(),
      "a/b"
    );
  }

  // FromDMS
  #[test]
  fn from_dms_three() {
    assert_eq!(interpret("FromDMS[{40, 26, 46}]").unwrap(), "72803/1800");
  }
  #[test]
  fn from_dms_two() {
    assert_eq!(interpret("FromDMS[{40, 26}]").unwrap(), "1213/30");
  }
  #[test]
  fn from_dms_one() {
    assert_eq!(interpret("FromDMS[46]").unwrap(), "46");
  }

  // NArgMin / NArgMax
  #[test]
  fn nargmin_basic() {
    assert_eq!(interpret("NArgMin[x^2 + 3x + 1, x]").unwrap(), "-1.5");
  }
  #[test]
  fn nargmax_basic() {
    assert_eq!(interpret("NArgMax[-x^2 + 3x + 1, x]").unwrap(), "1.5");
  }

  // AddSides / SubtractSides / MultiplySides / DivideSides / ApplySides
  #[test]
  fn add_sides_basic() {
    assert_eq!(interpret("AddSides[x == 2, 3]").unwrap(), "3 + x == 5");
  }
  #[test]
  fn subtract_sides_basic() {
    assert_eq!(interpret("SubtractSides[x + 3 == 5, 3]").unwrap(), "x == 2");
  }
  #[test]
  fn multiply_sides_basic() {
    assert_eq!(interpret("MultiplySides[x == 2, 3]").unwrap(), "3*x == 6");
  }
  #[test]
  fn divide_sides_basic() {
    assert_eq!(interpret("DivideSides[2x == 6, 2]").unwrap(), "x == 3");
  }
  #[test]
  fn apply_sides_basic() {
    assert_eq!(interpret("ApplySides[f, x == y]").unwrap(), "f[x] == f[y]");
  }

  // When the second argument is itself an equation, the corresponding sides
  // are paired: SubtractSides[a == b, c == d] -> a - c == b - d.
  #[test]
  fn add_sides_equation_second_arg() {
    assert_eq!(
      interpret("AddSides[a == b, c == d]").unwrap(),
      "a + c == b + d"
    );
  }
  #[test]
  fn subtract_sides_equation_second_arg() {
    assert_eq!(
      interpret("SubtractSides[a == b, c == d]").unwrap(),
      "a - c == b - d"
    );
  }
  #[test]
  fn multiply_sides_equation_second_arg() {
    assert_eq!(
      interpret("MultiplySides[a == b, c == d]").unwrap(),
      "a*c == b*d"
    );
  }
  #[test]
  fn divide_sides_equation_second_arg() {
    // Dividing by c is reversible only when c != 0, so the result is guarded.
    assert_eq!(
      interpret("DivideSides[a == b, c == d]").unwrap(),
      "Piecewise[{{a/c == b/d, c != 0}}, a == b]"
    );
  }
  #[test]
  fn add_sides_inequality_with_equation() {
    assert_eq!(
      interpret("AddSides[a < b, c == d]").unwrap(),
      "a + c < b + d"
    );
  }
  #[test]
  fn subtract_sides_inequality_with_equation() {
    assert_eq!(
      interpret("SubtractSides[a <= b, c == d]").unwrap(),
      "a - c <= b - d"
    );
  }
  #[test]
  fn subtract_sides_equation_simplifies() {
    assert_eq!(
      interpret("SubtractSides[2*x == 6, x == 1]").unwrap(),
      "x == 5"
    );
  }

  // One-argument forms: subtract / divide by the right-hand side.
  #[test]
  fn subtract_sides_one_arg() {
    assert_eq!(interpret("SubtractSides[a == b]").unwrap(), "a - b == 0");
  }
  #[test]
  fn subtract_sides_one_arg_compound() {
    assert_eq!(
      interpret("SubtractSides[a + b == c]").unwrap(),
      "a + b - c == 0"
    );
  }
  #[test]
  fn subtract_sides_one_arg_inequality() {
    // Works for any relation: subtracting the rhs preserves the direction.
    assert_eq!(interpret("SubtractSides[a <= b]").unwrap(), "a - b <= 0");
  }
  #[test]
  fn divide_sides_one_arg_symbolic() {
    // Dividing by a possibly-zero rhs is guarded.
    assert_eq!(
      interpret("DivideSides[a == b]").unwrap(),
      "Piecewise[{{a/b == 1, b != 0}}, a == b]"
    );
  }
  #[test]
  fn divide_sides_one_arg_numeric() {
    // A nonzero numeric rhs needs no guard.
    assert_eq!(interpret("DivideSides[a*b == 3]").unwrap(), "(a*b)/3 == 1");
    assert_eq!(interpret("DivideSides[2*x == 6]").unwrap(), "x/3 == 1");
  }

  // Multiplying/dividing an equation by a symbolic scalar is guarded by c != 0.
  #[test]
  fn multiply_sides_symbolic_scalar_guarded() {
    assert_eq!(
      interpret("MultiplySides[a == b, c]").unwrap(),
      "Piecewise[{{a*c == b*c, c != 0}}, a == b]"
    );
  }
  #[test]
  fn divide_sides_symbolic_scalar_guarded() {
    assert_eq!(
      interpret("DivideSides[a == b, c]").unwrap(),
      "Piecewise[{{a/c == b/c, c != 0}}, a == b]"
    );
  }
  #[test]
  fn multiply_sides_compound_scalar_guarded() {
    assert_eq!(
      interpret("MultiplySides[a == b, x + 1]").unwrap(),
      "Piecewise[{{a*(1 + x) == b*(1 + x), 1 + x != 0}}, a == b]"
    );
  }
  #[test]
  fn multiply_sides_negative_numeric_no_guard() {
    // A nonzero numeric scalar needs no guard.
    assert_eq!(
      interpret("MultiplySides[x == 2, -3]").unwrap(),
      "-3*x == -6"
    );
  }
  #[test]
  fn divide_sides_negative_numeric_no_guard() {
    assert_eq!(
      interpret("DivideSides[a == b, -2]").unwrap(),
      "-1/2*a == -1/2*b"
    );
  }

  // DayCount
  #[test]
  fn day_count_basic() {
    assert_eq!(
      interpret("DayCount[{2020, 1, 1}, {2020, 12, 31}]").unwrap(),
      "365"
    );
  }
  #[test]
  fn day_count_same_month() {
    assert_eq!(
      interpret("DayCount[{2023, 1, 1}, {2023, 1, 31}]").unwrap(),
      "30"
    );
  }
  #[test]
  fn day_count_weekday() {
    // 3-arg form counts a specific weekday in (start, end].
    assert_eq!(
      interpret("DayCount[{2024, 1, 1}, {2024, 12, 31}, Sunday]").unwrap(),
      "52"
    );
    assert_eq!(
      interpret("DayCount[{2024, 1, 1}, {2024, 1, 15}, Saturday]").unwrap(),
      "2"
    );
    // The earlier endpoint is excluded: Jan 1 2024 is a Monday, not counted.
    assert_eq!(
      interpret("DayCount[{2024, 1, 1}, {2024, 1, 7}, Monday]").unwrap(),
      "0"
    );
    // Symmetric for reversed dates.
    assert_eq!(
      interpret("DayCount[{2024, 1, 8}, {2024, 1, 1}, Monday]").unwrap(),
      "1"
    );
  }

  // ArrayResample
  #[test]
  fn array_resample_downsample() {
    assert_eq!(
      interpret("ArrayResample[{1, 2, 3, 4, 5}, 3]").unwrap(),
      "{1, 3, 5}"
    );
  }
  #[test]
  fn array_resample_upsample() {
    assert_eq!(
      interpret("ArrayResample[{1, 2, 3}, 5]").unwrap(),
      "{1, 3/2, 2, 5/2, 3}"
    );
  }
  #[test]
  fn array_resample_exact() {
    assert_eq!(
      interpret("ArrayResample[{10, 20, 30}, 5]").unwrap(),
      "{10, 15, 20, 25, 30}"
    );
  }

  // IntersectingQ
  #[test]
  fn intersecting_q_true() {
    assert_eq!(
      interpret("IntersectingQ[{1, 2, 3}, {3, 4, 5}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn intersecting_q_false() {
    assert_eq!(
      interpret("IntersectingQ[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn intersecting_q_empty() {
    assert_eq!(interpret("IntersectingQ[{}, {1}]").unwrap(), "False");
  }
  // IntersectingQ/DisjointQ accept a SameTest -> f option (applied as
  // f[a_elem, b_elem]).
  #[test]
  fn intersecting_disjoint_same_test() {
    assert_eq!(
      interpret("IntersectingQ[{1, 2}, {2, 3}, SameTest -> Equal]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("DisjointQ[{1, 2}, {3, 4}, SameTest -> Equal]").unwrap(),
      "True"
    );
    // Cross-type equality via the test.
    assert_eq!(
      interpret("IntersectingQ[{1.0}, {1}, SameTest -> Equal]").unwrap(),
      "True"
    );
    // The asymmetric Greater test confirms f[a, b] ordering.
    assert_eq!(
      interpret("IntersectingQ[{5}, {3}, SameTest -> Greater]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("IntersectingQ[{3}, {5}, SameTest -> Greater]").unwrap(),
      "False"
    );
  }

  // AlternatingFactorial
  #[test]
  fn alternating_factorial_0() {
    assert_eq!(interpret("AlternatingFactorial[0]").unwrap(), "0");
  }
  #[test]
  fn alternating_factorial_1() {
    assert_eq!(interpret("AlternatingFactorial[1]").unwrap(), "1");
  }
  #[test]
  fn alternating_factorial_3() {
    assert_eq!(interpret("AlternatingFactorial[3]").unwrap(), "5");
  }
  #[test]
  fn alternating_factorial_5() {
    assert_eq!(interpret("AlternatingFactorial[5]").unwrap(), "101");
  }
  #[test]
  fn alternating_factorial_10() {
    assert_eq!(interpret("AlternatingFactorial[10]").unwrap(), "3301819");
  }

  // AlphabeticOrder
  #[test]
  fn alphabetic_order_less() {
    assert_eq!(
      interpret("AlphabeticOrder[\"apple\", \"banana\"]").unwrap(),
      "1"
    );
  }
  #[test]
  fn alphabetic_order_greater() {
    assert_eq!(
      interpret("AlphabeticOrder[\"banana\", \"apple\"]").unwrap(),
      "-1"
    );
  }
  #[test]
  fn alphabetic_order_equal() {
    assert_eq!(
      interpret("AlphabeticOrder[\"apple\", \"apple\"]").unwrap(),
      "0"
    );
  }

  // BinaryDistance
  #[test]
  fn binary_distance_same() {
    assert_eq!(
      interpret("BinaryDistance[{1, 0, 1}, {1, 0, 1}]").unwrap(),
      "0"
    );
  }
  #[test]
  fn binary_distance_different() {
    assert_eq!(
      interpret("BinaryDistance[{1, 0, 1, 1}, {1, 1, 0, 1}]").unwrap(),
      "1"
    );
  }

  // SquaresR
  #[test]
  fn squares_r_2_5() {
    assert_eq!(interpret("SquaresR[2, 5]").unwrap(), "8");
  }
  #[test]
  fn squares_r_2_25() {
    assert_eq!(interpret("SquaresR[2, 25]").unwrap(), "12");
  }
  #[test]
  fn squares_r_2_0() {
    assert_eq!(interpret("SquaresR[2, 0]").unwrap(), "1");
  }
  #[test]
  fn squares_r_1_4() {
    assert_eq!(interpret("SquaresR[1, 4]").unwrap(), "2");
  }
  #[test]
  fn squares_r_1_3() {
    assert_eq!(interpret("SquaresR[1, 3]").unwrap(), "0");
  }
  #[test]
  fn squares_r_4_5() {
    assert_eq!(interpret("SquaresR[4, 5]").unwrap(), "48");
  }

  // HankelMatrix
  #[test]
  fn hankel_matrix_basic() {
    assert_eq!(
      interpret("HankelMatrix[{1, 2, 3}]").unwrap(),
      "{{1, 2, 3}, {2, 3, 0}, {3, 0, 0}}"
    );
  }
  #[test]
  fn hankel_matrix_two_args() {
    assert_eq!(
      interpret("HankelMatrix[{1, 2, 3}, {3, 4, 5}]").unwrap(),
      "{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}"
    );
  }
  #[test]
  fn hankel_matrix_integer_n() {
    // HankelMatrix[n] gives the n×n integer Hankel matrix whose first
    // row/column are 1..n and lower-right triangle zero.
    assert_eq!(
      interpret("HankelMatrix[4]").unwrap(),
      "{{1, 2, 3, 4}, {2, 3, 4, 0}, {3, 4, 0, 0}, {4, 0, 0, 0}}"
    );
  }
  #[test]
  fn hankel_matrix_integer_one() {
    assert_eq!(interpret("HankelMatrix[1]").unwrap(), "{{1}}");
  }
  #[test]
  fn hankel_matrix_integer_three() {
    assert_eq!(
      interpret("HankelMatrix[3]").unwrap(),
      "{{1, 2, 3}, {2, 3, 0}, {3, 0, 0}}"
    );
  }

  // HadamardMatrix
  #[test]
  fn hadamard_matrix_2() {
    assert_eq!(
      interpret("HadamardMatrix[2]").unwrap(),
      "{{1/Sqrt[2], 1/Sqrt[2]}, {1/Sqrt[2], -(1/Sqrt[2])}}"
    );
  }
  #[test]
  fn hadamard_matrix_4() {
    assert_eq!(
      interpret("HadamardMatrix[4]").unwrap(),
      "{{1/2, 1/2, 1/2, 1/2}, {1/2, 1/2, -1/2, -1/2}, {1/2, -1/2, -1/2, 1/2}, {1/2, -1/2, 1/2, -1/2}}"
    );
  }

  // PrimitiveRootList
  #[test]
  fn primitive_root_list_7() {
    assert_eq!(interpret("PrimitiveRootList[7]").unwrap(), "{3, 5}");
  }
  #[test]
  fn primitive_root_list_13() {
    assert_eq!(interpret("PrimitiveRootList[13]").unwrap(), "{2, 6, 7, 11}");
  }

  // DMSList (inverse of FromDMS)
  #[test]
  fn dms_list_basic() {
    assert_eq!(interpret("DMSList[72803/1800]").unwrap(), "{40, 26, 46}");
  }
  #[test]
  fn dms_list_from_triple_with_negative_minutes() {
    // wolframscript: DMSList[{11, -30, 5.}] = {10, 30, 4.999999999997584}.
    assert_eq!(
      interpret("DMSList[{11, -30, 5.}]").unwrap(),
      "{10, 30, 4.999999999997584}"
    );
  }
  #[test]
  fn dms_list_from_integer_triple() {
    // All-integer input → all-integer output, normalised via rationals.
    assert_eq!(interpret("DMSList[{10, 30, 5}]").unwrap(), "{10, 30, 5}");
  }
  #[test]
  fn dms_list_from_rational_triple() {
    assert_eq!(interpret("DMSList[{1, 2, 1/2}]").unwrap(), "{1, 2, 1/2}");
  }
  #[test]
  fn dms_list_from_negative_triple() {
    // wolframscript: DMSList[{-10, -30, -5.}] = {-10, -30, -4.999999999997584}.
    assert_eq!(
      interpret("DMSList[{-10, -30, -5.}]").unwrap(),
      "{-10, -30, -4.999999999997584}"
    );
  }

  // WordCount
  #[test]
  fn word_count_basic() {
    assert_eq!(
      interpret("WordCount[\"hello world foo bar\"]").unwrap(),
      "4"
    );
  }
  #[test]
  fn word_count_empty() {
    assert_eq!(interpret("WordCount[\"\"]").unwrap(), "0");
  }

  // CenterArray
  #[test]
  fn center_array_basic() {
    assert_eq!(
      interpret("CenterArray[{a, b, c}, 7]").unwrap(),
      "{0, 0, a, b, c, 0, 0}"
    );
  }
  #[test]
  fn center_array_smaller() {
    assert_eq!(interpret("CenterArray[{a, b, c}, 2]").unwrap(), "{b, c}");
  }

  // ScalingMatrix
  #[test]
  fn scaling_matrix_2d() {
    assert_eq!(
      interpret("ScalingMatrix[{2, 3}]").unwrap(),
      "{{2, 0}, {0, 3}}"
    );
  }
  #[test]
  fn scaling_matrix_3d() {
    assert_eq!(
      interpret("ScalingMatrix[{1, 2, 3}]").unwrap(),
      "{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}"
    );
  }

  // ReverseSortBy
  #[test]
  fn reverse_sort_by_basic() {
    assert_eq!(
      interpret("ReverseSortBy[{3, 1, 2, 5, 4}, Identity]").unwrap(),
      "{5, 4, 3, 2, 1}"
    );
  }

  // ReverseSortBy[list, f] == Reverse[SortBy[list, f]], so equal keys are
  // reversed too (regression: ties used to keep their original order).
  #[test]
  fn reverse_sort_by_reverses_ties() {
    assert_eq!(
      interpret("ReverseSortBy[{1, 3, 2, 4}, EvenQ]").unwrap(),
      "{4, 2, 3, 1}"
    );
  }

  // A list of functions gives a lexicographic multi-criteria reverse sort.
  #[test]
  fn reverse_sort_by_multiple_criteria() {
    assert_eq!(
      interpret("ReverseSortBy[{1, 2, 3, 4}, {Mod[#, 2] &, # &}]").unwrap(),
      "{3, 1, 4, 2}"
    );
  }

  // CorrelationDistance
  #[test]
  fn correlation_distance_perfect() {
    assert_eq!(
      interpret("CorrelationDistance[{1, 2, 3}, {2, 4, 6}]").unwrap(),
      "0"
    );
  }

  // PowerModList
  #[test]
  fn power_mod_list_basic() {
    assert_eq!(
      // PowerModList[a, n, m] finds all x in {0,...,m-1} such that x^n ≡ a (mod m)
      interpret("PowerModList[2, 5, 7]").unwrap(),
      "{4}"
    );
  }
  #[test]
  fn power_mod_list_3() {
    assert_eq!(interpret("PowerModList[3, 4, 5]").unwrap(), "{1}");
  }

  // Antisymmetric
  #[test]
  fn antisymmetric_basic() {
    assert_eq!(
      interpret("Antisymmetric[{1, 2}]").unwrap(),
      "Antisymmetric[{1, 2}]"
    );
  }
  #[test]
  fn antisymmetric_three_indices() {
    assert_eq!(
      interpret("Antisymmetric[{1, 2, 3}]").unwrap(),
      "Antisymmetric[{1, 2, 3}]"
    );
  }
  #[test]
  fn antisymmetric_no_args() {
    assert_eq!(interpret("Antisymmetric[]").unwrap(), "Antisymmetric[]");
  }
  #[test]
  fn antisymmetric_single_arg() {
    assert_eq!(interpret("Antisymmetric[1]").unwrap(), "Antisymmetric[1]");
  }
  #[test]
  fn antisymmetric_two_args() {
    assert_eq!(
      interpret("Antisymmetric[{1, 2}, x]").unwrap(),
      "Antisymmetric[{1, 2}, x]"
    );
  }

  // AntisymmetricMatrixQ
  #[test]
  fn antisymmetric_matrix_q_true() {
    assert_eq!(
      interpret("AntisymmetricMatrixQ[{{0, 1, -2}, {-1, 0, 3}, {2, -3, 0}}]")
        .unwrap(),
      "True"
    );
  }
  #[test]
  fn antisymmetric_matrix_q_false() {
    assert_eq!(
      interpret("AntisymmetricMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }

  // ShearingMatrix
  #[test]
  fn shearing_matrix_2d() {
    assert_eq!(
      interpret("ShearingMatrix[2, {1, 0}, {0, 1}]").unwrap(),
      "{{1, Tan[2]}, {0, 1}}"
    );
  }

  // DiagonalMatrixQ
  #[test]
  fn diagonal_matrix_q_true() {
    assert_eq!(
      interpret("DiagonalMatrixQ[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn diagonal_matrix_q_false() {
    assert_eq!(
      interpret("DiagonalMatrixQ[{{1, 2}, {0, 3}}]").unwrap(),
      "False"
    );
  }
  // DiagonalMatrixQ[m, k] — nonzeros allowed only on the k-th diagonal.
  #[test]
  fn diagonal_matrix_q_band_offset() {
    // Superdiagonal (k = 1).
    assert_eq!(
      interpret("DiagonalMatrixQ[{{0, 1, 0}, {0, 0, 1}, {0, 0, 0}}, 1]")
        .unwrap(),
      "True"
    );
    // A main-diagonal matrix is not 1-banded.
    assert_eq!(
      interpret("DiagonalMatrixQ[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, 1]")
        .unwrap(),
      "False"
    );
    // Subdiagonal (k = -1).
    assert_eq!(
      interpret("DiagonalMatrixQ[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, -1]")
        .unwrap(),
      "True"
    );
    // Symbolic nonzeros on the k-th diagonal are allowed.
    assert_eq!(
      interpret("DiagonalMatrixQ[{{0, a, 0}, {0, 0, b}, {0, 0, 0}}, 1]")
        .unwrap(),
      "True"
    );
    // A band beyond the matrix means every entry must be zero.
    assert_eq!(
      interpret("DiagonalMatrixQ[{{1, 2}, {3, 4}}, 5]").unwrap(),
      "False"
    );
  }
  // Regression: rectangular (non-square) matrices are accepted.
  #[test]
  fn diagonal_matrix_q_rectangular() {
    assert_eq!(
      interpret("DiagonalMatrixQ[{{1, 0, 0}, {0, 2, 0}}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("DiagonalMatrixQ[{{0, 1, 0}, {0, 0, 2}}, 1]").unwrap(),
      "True"
    );
  }

  // UpperTriangularMatrixQ
  #[test]
  fn upper_triangular_q_true() {
    assert_eq!(
      interpret("UpperTriangularMatrixQ[{{1, 2, 3}, {0, 4, 5}, {0, 0, 6}}]")
        .unwrap(),
      "True"
    );
  }
  #[test]
  fn upper_triangular_q_false() {
    assert_eq!(
      interpret("UpperTriangularMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }

  // LowerTriangularMatrixQ
  #[test]
  fn lower_triangular_q_true() {
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{{1, 0, 0}, {2, 3, 0}, {4, 5, 6}}]")
        .unwrap(),
      "True"
    );
  }
  #[test]
  fn lower_triangular_q_false() {
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }

  // Non-matrix arguments (scalar, symbol, vector) answer False rather than
  // staying unevaluated, matching wolframscript.
  #[test]
  fn diagonal_matrix_q_non_matrix() {
    assert_eq!(interpret("DiagonalMatrixQ[5]").unwrap(), "False");
    assert_eq!(interpret("DiagonalMatrixQ[x]").unwrap(), "False");
    assert_eq!(interpret("DiagonalMatrixQ[{1, 2, 3}]").unwrap(), "False");
  }
  #[test]
  fn upper_triangular_q_non_matrix() {
    assert_eq!(interpret("UpperTriangularMatrixQ[5]").unwrap(), "False");
    assert_eq!(interpret("UpperTriangularMatrixQ[x]").unwrap(), "False");
    assert_eq!(
      interpret("UpperTriangularMatrixQ[{1, 2}]").unwrap(),
      "False"
    );
  }
  #[test]
  fn lower_triangular_q_non_matrix() {
    assert_eq!(interpret("LowerTriangularMatrixQ[5]").unwrap(), "False");
    assert_eq!(
      interpret("LowerTriangularMatrixQ[{1, 2}]").unwrap(),
      "False"
    );
  }

  // KroneckerSymbol
  #[test]
  fn kronecker_symbol_basic() {
    assert_eq!(interpret("KroneckerSymbol[2, 7]").unwrap(), "1");
  }
  #[test]
  fn kronecker_symbol_neg() {
    assert_eq!(interpret("KroneckerSymbol[3, 7]").unwrap(), "-1");
  }
  #[test]
  fn kronecker_symbol_zero() {
    assert_eq!(interpret("KroneckerSymbol[7, 7]").unwrap(), "0");
  }
  // KroneckerSymbol is Listable (threads over either argument).
  #[test]
  fn kronecker_symbol_threads() {
    assert_eq!(interpret("KroneckerSymbol[{1, 2}, 5]").unwrap(), "{1, -1}");
    assert_eq!(
      interpret("KroneckerSymbol[2, {3, 5, 7}]").unwrap(),
      "{-1, -1, 1}"
    );
  }
  // NextPrime is Listable.
  #[test]
  fn next_prime_threads() {
    assert_eq!(interpret("NextPrime[{10, 20}]").unwrap(), "{11, 23}");
    assert_eq!(interpret("NextPrime[{10, 20}, 2]").unwrap(), "{13, 29}");
    assert_eq!(interpret("NextPrime[10, {1, 2}]").unwrap(), "{11, 13}");
  }

  // NormalizedSquaredEuclideanDistance
  #[test]
  fn normalized_sqeuclidean_same() {
    assert_eq!(
      interpret("NormalizedSquaredEuclideanDistance[{1, 2, 3}, {1, 2, 3}]")
        .unwrap(),
      "0"
    );
  }

  // CrossMatrix
  #[test]
  fn cross_matrix_basic() {
    assert_eq!(
      interpret("CrossMatrix[{1, 0, 0}]").unwrap(),
      "{{0, 0, 0}, {0, 0, -1}, {0, 1, 0}}"
    );
  }
  #[test]
  fn cross_matrix_unit_y() {
    assert_eq!(
      interpret("CrossMatrix[{0, 1, 0}]").unwrap(),
      "{{0, 0, 1}, {0, 0, 0}, {-1, 0, 0}}"
    );
  }
  #[test]
  fn cross_matrix_general_numeric() {
    assert_eq!(
      interpret("CrossMatrix[{2, 3, 4}]").unwrap(),
      "{{0, -4, 3}, {4, 0, -2}, {-3, 2, 0}}"
    );
  }
  #[test]
  fn cross_matrix_symbolic_unevaluated() {
    // Symbolic arguments return unevaluated (matches Wolfram behavior)
    assert_eq!(
      interpret("CrossMatrix[{a, b, c}]").unwrap(),
      "CrossMatrix[{a, b, c}]"
    );
  }

  // FourierMatrix
  #[test]
  fn fourier_matrix_1() {
    assert_eq!(interpret("FourierMatrix[1]").unwrap(), "{{1}}");
  }
  #[test]
  fn fourier_matrix_2() {
    assert_eq!(
      interpret("FourierMatrix[2]").unwrap(),
      "{{1/Sqrt[2], 1/Sqrt[2]}, {1/Sqrt[2], -(1/Sqrt[2])}}"
    );
  }
  #[test]
  fn fourier_matrix_4_dimensions() {
    // For n = 4 a 4x4 matrix should be materialised.
    let result = interpret("Dimensions[FourierMatrix[4]]").unwrap();
    assert_eq!(result, "{4, 4}");
  }
  #[test]
  fn fourier_matrix_large_does_not_materialize() {
    // wolframscript: FourierMatrix[800] is too large to materialise and
    // is returned as a `FourierMatrix[StructuredArray\`StructuredData[...]]`
    // placeholder. Woxi must not time out by trying to enumerate 640k
    // symbolic entries; it leaves the call unevaluated instead.
    let result = interpret("Head[FourierMatrix[800]]").unwrap();
    assert_eq!(result, "FourierMatrix");
  }
  #[test]
  fn fourier_matrix_boundary_materializes() {
    // n = 100 is well below the materialisation threshold and must
    // produce a concrete 100×100 list.
    assert_eq!(
      interpret("Dimensions[FourierMatrix[100]]").unwrap(),
      "{100, 100}"
    );
  }

  // Symmetrize: wolframscript returns SymmetrizedArray with the upper
  // triangle of (M + M^T)/2.
  #[test]
  fn symmetrize_symmetric() {
    assert_eq!(
      interpret("Symmetrize[{{1, 2}, {2, 3}}]").unwrap(),
      "SymmetrizedArray[StructuredArray`StructuredData[{2, 2}, {{{1, 1} -> 1, {1, 2} -> 2, {2, 2} -> 3}, Symmetric[{1, 2}]}]]"
    );
  }
  #[test]
  fn symmetrize_asymmetric() {
    assert_eq!(
      interpret("Symmetrize[{{1, 2}, {4, 3}}]").unwrap(),
      "SymmetrizedArray[StructuredArray`StructuredData[{2, 2}, {{{1, 1} -> 1, {1, 2} -> 3, {2, 2} -> 3}, Symmetric[{1, 2}]}]]"
    );
  }

  // DisjointQ
  #[test]
  fn disjoint_q_true() {
    assert_eq!(
      interpret("DisjointQ[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "True"
    );
  }
  #[test]
  fn disjoint_q_false() {
    assert_eq!(
      interpret("DisjointQ[{1, 2, 3}, {3, 4, 5}]").unwrap(),
      "False"
    );
  }

  // CoordinateBoundsArray
  #[test]
  fn coordinate_bounds_array_basic() {
    assert_eq!(
      interpret("CoordinateBoundsArray[{{0, 1}, {0, 2}}]").unwrap(),
      "{{{0, 0}, {0, 1}, {0, 2}}, {{1, 0}, {1, 1}, {1, 2}}}"
    );
  }
  #[test]
  fn coordinate_bounds_array_with_step() {
    // CoordinateBoundsArray with step size 1
    assert_eq!(
      interpret("CoordinateBoundsArray[{{0, 3}}, 1]").unwrap(),
      "{{0}, {1}, {2}, {3}}"
    );
  }

  // FindPermutation
  #[test]
  fn find_permutation_basic() {
    assert_eq!(
      interpret("FindPermutation[{a, b, c}, {b, c, a}]").unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
  }
  #[test]
  fn find_permutation_identity() {
    assert_eq!(
      interpret("FindPermutation[{a, b, c}, {a, b, c}]").unwrap(),
      "Cycles[{}]"
    );
  }

  #[test]
  fn find_permutation_custom_head() {
    // FindPermutation accepts any matching head, not just List.
    assert_eq!(
      interpret("FindPermutation[head[a, c, d, e, b], head[c, a, b, d, e]]")
        .unwrap(),
      "Cycles[{{1, 2}, {3, 4, 5}}]"
    );
  }

  #[test]
  fn find_permutation_mixed_heads_unevaluated() {
    // Different heads should remain unevaluated.
    assert_eq!(
      interpret("FindPermutation[head[a, b], List[a, b]]").unwrap(),
      "FindPermutation[head[a, b], {a, b}]"
    );
  }

  // One-argument form: FindPermutation[list] = FindPermutation[Sort[list], list].
  #[test]
  fn find_permutation_single_argument() {
    assert_eq!(
      interpret("FindPermutation[{c, a, b}]").unwrap(),
      "Cycles[{{1, 2, 3}}]"
    );
    assert_eq!(
      interpret("FindPermutation[{b, c, a}]").unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
    // An already-sorted list gives the identity permutation.
    assert_eq!(
      interpret("FindPermutation[{1, 2, 3}]").unwrap(),
      "Cycles[{}]"
    );
  }

  // KeyMemberQ
  #[test]
  fn key_member_q_true() {
    assert_eq!(
      interpret("KeyMemberQ[<|\"a\" -> 1, \"b\" -> 2|>, \"a\"]").unwrap(),
      "True"
    );
  }
  #[test]
  fn key_member_q_false() {
    assert_eq!(
      interpret("KeyMemberQ[<|\"a\" -> 1, \"b\" -> 2|>, \"c\"]").unwrap(),
      "False"
    );
  }

  // PermutationOrder
  #[test]
  fn permutation_order_identity() {
    assert_eq!(interpret("PermutationOrder[{1, 2, 3}]").unwrap(), "1");
  }
  #[test]
  fn permutation_order_swap() {
    assert_eq!(interpret("PermutationOrder[{2, 1, 3}]").unwrap(), "2");
  }
  #[test]
  fn permutation_order_cycle3() {
    assert_eq!(interpret("PermutationOrder[{2, 3, 1}]").unwrap(), "3");
  }

  // PermutationPower
  #[test]
  fn permutation_power_identity() {
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, 3]").unwrap(),
      "{1, 2, 3}"
    );
  }
  #[test]
  fn permutation_power_square() {
    assert_eq!(
      interpret("PermutationPower[{2, 3, 1}, 2]").unwrap(),
      "{3, 1, 2}"
    );
  }

  // PermutationProduct (multi-arg cycle composition)
  #[test]
  fn permutation_product_two_swaps() {
    // Apply Cycles[{{1,2}}] first then Cycles[{{2,3}}].
    assert_eq!(
      interpret("PermutationProduct[Cycles[{{1, 2}}], Cycles[{{2, 3}}]]")
        .unwrap(),
      "Cycles[{{1, 3, 2}}]"
    );
  }

  #[test]
  fn permutation_product_three_swaps() {
    assert_eq!(
      interpret(
        "PermutationProduct[Cycles[{{1, 2}}], Cycles[{{2, 3}}], \
         Cycles[{{3, 4}}]]"
      )
      .unwrap(),
      "Cycles[{{1, 4, 3, 2}}]"
    );
  }

  #[test]
  fn permutation_product_identity_via_self_inverse() {
    // Two copies of the same swap cancel to the identity.
    assert_eq!(
      interpret("PermutationProduct[Cycles[{{1, 2}}], Cycles[{{1, 2}}]]")
        .unwrap(),
      "Cycles[{}]"
    );
  }

  #[test]
  fn permutation_product_with_identity_cycle() {
    // Cycles[{}] is the identity.
    assert_eq!(
      interpret("PermutationProduct[Cycles[{}], Cycles[{{1, 2}}]]").unwrap(),
      "Cycles[{{1, 2}}]"
    );
  }

  #[test]
  fn permutation_product_empty_is_identity() {
    assert_eq!(interpret("PermutationProduct[]").unwrap(), "Cycles[{}]");
  }

  #[test]
  fn permutation_product_disjoint_then_merge() {
    // Cycles[{{1,2,3},{4,5}}] · Cycles[{{1,4}}].
    assert_eq!(
      interpret(
        "PermutationProduct[Cycles[{{1, 2, 3}, {4, 5}}], Cycles[{{1, 4}}]]"
      )
      .unwrap(),
      "Cycles[{{1, 2, 3, 4, 5}}]"
    );
  }

  // PermutationProduct with permutation-list (image-list) arguments composes
  // left to right and returns a permutation list: result[i] = p2[p1[i]].
  #[test]
  fn permutation_product_lists_basic() {
    assert_eq!(
      interpret("PermutationProduct[{2, 1, 3}, {1, 3, 2}]").unwrap(),
      "{3, 1, 2}"
    );
  }

  #[test]
  fn permutation_product_lists_three_cycle_cubed_is_identity() {
    assert_eq!(
      interpret("PermutationProduct[{2, 3, 1}, {2, 3, 1}, {2, 3, 1}]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn permutation_product_lists_length_four() {
    assert_eq!(
      interpret("PermutationProduct[{2, 3, 1, 4}, {1, 2, 4, 3}]").unwrap(),
      "{2, 4, 1, 3}"
    );
  }

  #[test]
  fn permutation_product_lists_single_arg() {
    assert_eq!(
      interpret("PermutationProduct[{2, 1, 3}]").unwrap(),
      "{2, 1, 3}"
    );
  }

  #[test]
  fn permutation_product_lists_different_lengths() {
    // The shorter permutation fixes points beyond its length.
    assert_eq!(
      interpret("PermutationProduct[{2, 1}, {2, 3, 1}]").unwrap(),
      "{3, 2, 1}"
    );
  }

  #[test]
  fn permutation_product_invalid_list_unevaluated() {
    // {2, 1, 4} is not a permutation of {1, 2, 3}.
    assert_eq!(
      interpret("PermutationProduct[{2, 1, 4}, {1, 3, 2}]").unwrap(),
      "PermutationProduct[{2, 1, 4}, {1, 3, 2}]"
    );
  }

  // PermutationLength
  #[test]
  fn permutation_length_identity() {
    assert_eq!(interpret("PermutationLength[{1, 2, 3}]").unwrap(), "0");
  }
  #[test]
  fn permutation_length_swap() {
    assert_eq!(interpret("PermutationLength[{2, 1, 3}]").unwrap(), "2");
  }

  // PermutationListQ
  #[test]
  fn permutation_list_q_true() {
    assert_eq!(interpret("PermutationListQ[{2, 3, 1}]").unwrap(), "True");
  }
  #[test]
  fn permutation_list_q_false() {
    assert_eq!(interpret("PermutationListQ[{1, 1, 2}]").unwrap(), "False");
  }

  // FoldWhileList
  #[test]
  fn fold_while_list_basic() {
    assert_eq!(
      interpret("FoldWhileList[Plus, 0, {1, 2, 3, 4, 5}, Function[# < 10]]")
        .unwrap(),
      "{0, 1, 3, 6, 10}"
    );
  }

  // PermutationCyclesQ
  #[test]
  fn permutation_cycles_q_true() {
    assert_eq!(
      interpret("PermutationCyclesQ[Cycles[{{1, 3}, {2, 4}}]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn permutation_cycles_q_false() {
    assert_eq!(interpret("PermutationCyclesQ[{1, 2, 3}]").unwrap(), "False");
  }

  // PermutationSupport
  #[test]
  fn permutation_support_basic() {
    assert_eq!(
      interpret("PermutationSupport[{2, 1, 3}]").unwrap(),
      "{1, 2}"
    );
  }
  #[test]
  fn permutation_support_identity() {
    assert_eq!(interpret("PermutationSupport[{1, 2, 3}]").unwrap(), "{}");
  }

  // PermutationMax
  #[test]
  fn permutation_max_basic() {
    assert_eq!(interpret("PermutationMax[{2, 1, 3, 4}]").unwrap(), "2");
  }

  // PermutationMin
  #[test]
  fn permutation_min_basic() {
    assert_eq!(interpret("PermutationMin[{2, 1, 3, 4}]").unwrap(), "1");
  }

  // Splice
  #[test]
  fn splice_basic() {
    assert_eq!(interpret("{1, Splice[{2, 3}], 4}").unwrap(), "{1, 2, 3, 4}");
  }

  // SubsetMap
  #[test]
  fn subset_map_basic() {
    assert_eq!(
      interpret("SubsetMap[Reverse, {a, b, c, d, e}, {2, 4}]").unwrap(),
      "{a, d, c, b, e}"
    );
  }
  // SubsetMap with a Span position specification.
  #[test]
  fn subset_map_span() {
    assert_eq!(
      interpret("SubsetMap[Accumulate, {x1, x2, x3, x4, x5, x6}, 2 ;; 5]")
        .unwrap(),
      "{x1, x2, x2 + x3, x2 + x3 + x4, x2 + x3 + x4 + x5, x6}"
    );
  }
  #[test]
  fn subset_map_span_numeric() {
    assert_eq!(
      interpret("SubsetMap[Reverse, {10, 20, 30, 40, 50}, 2 ;; 4]").unwrap(),
      "{10, 40, 30, 20, 50}"
    );
  }
  #[test]
  fn subset_map_span_negative_end() {
    assert_eq!(
      interpret("SubsetMap[Reverse, {a, b, c, d, e, f}, 2 ;; -1]").unwrap(),
      "{a, f, e, d, c, b}"
    );
  }
  #[test]
  fn subset_map_span_with_step() {
    assert_eq!(
      interpret("SubsetMap[Reverse, {a, b, c, d, e, f}, 1 ;; 6 ;; 2]").unwrap(),
      "{e, b, c, d, a, f}"
    );
  }
  #[test]
  fn subset_map_span_all_end() {
    assert_eq!(
      interpret("SubsetMap[Accumulate, {1, 2, 3, 4, 5}, 2 ;; All]").unwrap(),
      "{1, 2, 5, 9, 14}"
    );
  }

  // Assert — returns unevaluated (Wolfram default behavior without AssertTools package)
  #[test]
  fn assert_unevaluated() {
    assert_eq!(
      interpret("Assert[1 + 1 == 2]").unwrap(),
      "Assert[1 + 1 == 2]"
    );
  }
  #[test]
  fn assert_false_unevaluated() {
    assert_eq!(
      interpret("Assert[1 + 1 == 3]").unwrap(),
      "Assert[1 + 1 == 3]"
    );
  }

  // StarGraph
  #[test]
  fn star_graph_basic() {
    assert_eq!(
      interpret("VertexList[StarGraph[4]]").unwrap(),
      "{1, 2, 3, 4}"
    );
    assert_eq!(
      interpret("EdgeList[StarGraph[4]]").unwrap(),
      "{1  2, 1  3, 1  4}"
    );
  }

  // CirculantGraph
  #[test]
  fn circulant_graph_basic() {
    assert_eq!(
      interpret("VertexList[CirculantGraph[4, {1}]]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }
  // Edges wrap around modularly: CirculantGraph[6, {1}] is the 6-cycle (6
  // edges, including 1-6), not a 5-edge path.
  #[test]
  fn circulant_graph_wraparound() {
    assert_eq!(interpret("EdgeCount[CirculantGraph[6, {1}]]").unwrap(), "6");
    assert_eq!(
      interpret("EdgeCount[CirculantGraph[6, {1, 2}]]").unwrap(),
      "12"
    );
    assert_eq!(
      interpret("EdgeCount[CirculantGraph[5, {1, 2}]]").unwrap(),
      "10"
    );
  }
  // The jump may also be given as a bare integer.
  #[test]
  fn circulant_graph_integer_jump() {
    assert_eq!(interpret("EdgeCount[CirculantGraph[6, 2]]").unwrap(), "6");
  }

  // KaryTree
  #[test]
  fn kary_tree_binary() {
    assert_eq!(
      interpret("VertexList[KaryTree[7]]").unwrap(),
      "{1, 2, 3, 4, 5, 6, 7}"
    );
    let edges = interpret("EdgeList[KaryTree[7]]").unwrap();
    assert!(edges.contains("1  2"));
    assert!(edges.contains("1  3"));
  }

  // HypercubeGraph
  #[test]
  fn hypercube_graph_2() {
    assert_eq!(
      interpret("VertexList[HypercubeGraph[2]]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }

  // EdgeQ
  #[test]
  fn edge_q_true() {
    assert_eq!(interpret("EdgeQ[CompleteGraph[3], 1  2]").unwrap(), "True");
  }
  #[test]
  fn edge_q_false() {
    assert_eq!(interpret("EdgeQ[StarGraph[3], 2  3]").unwrap(), "False");
  }

  // Booleans domain
  #[test]
  fn booleans_element() {
    assert_eq!(interpret("Element[True, Booleans]").unwrap(), "True");
  }

  // UndirectedGraphQ
  #[test]
  fn undirected_graph_q_true() {
    assert_eq!(
      interpret("UndirectedGraphQ[CompleteGraph[3]]").unwrap(),
      "True"
    );
  }

  // DeBruijnGraph
  #[test]
  fn debruijn_graph_vertex_count() {
    // DeBruijnGraph[2, 3] has 2^3 = 8 vertices
    assert_eq!(interpret("VertexCount[DeBruijnGraph[2, 3]]").unwrap(), "8");
  }
  #[test]
  fn debruijn_graph_edge_count() {
    // DeBruijnGraph[2, 3] has 2^3 * 2 = 16 edges (each vertex has m=2 outgoing edges)
    assert_eq!(interpret("EdgeCount[DeBruijnGraph[2, 3]]").unwrap(), "16");
  }
  #[test]
  fn debruijn_graph_is_directed() {
    assert_eq!(
      interpret("DirectedGraphQ[DeBruijnGraph[2, 2]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn debruijn_graph_2_2_vertex_count() {
    // DeBruijnGraph[2, 2] has 4 vertices
    assert_eq!(interpret("VertexCount[DeBruijnGraph[2, 2]]").unwrap(), "4");
  }
  #[test]
  fn debruijn_graph_2_2_edge_count() {
    // DeBruijnGraph[2, 2] has 8 edges
    assert_eq!(interpret("EdgeCount[DeBruijnGraph[2, 2]]").unwrap(), "8");
  }

  // TuranGraph
  #[test]
  fn turan_graph_4_2() {
    // T(4,2) = complete bipartite K_{2,2} = 4 edges
    assert_eq!(interpret("EdgeCount[TuranGraph[4, 2]]").unwrap(), "4");
  }
  #[test]
  fn turan_graph_n_1() {
    // T(n,1) = no edges (all in same partition)
    assert_eq!(interpret("EdgeCount[TuranGraph[5, 1]]").unwrap(), "0");
  }
  #[test]
  fn turan_graph_n_n() {
    // T(n,n) = complete graph K_n
    assert_eq!(interpret("EdgeCount[TuranGraph[4, 4]]").unwrap(), "6");
  }
  #[test]
  fn turan_graph_vertex_count() {
    assert_eq!(interpret("VertexCount[TuranGraph[6, 3]]").unwrap(), "6");
  }
  #[test]
  fn turan_graph_5_2_edges() {
    // T(5,2): partitions of size 3 and 2 → 3*2 = 6 edges
    assert_eq!(interpret("EdgeCount[TuranGraph[5, 2]]").unwrap(), "6");
  }

  // GraphIntersection
  #[test]
  fn graph_intersection_basic() {
    // K3 ∩ path{1,2,3} = union vertices, intersect edges
    assert_eq!(
      interpret(
        "EdgeList[GraphIntersection[CompleteGraph[3], PathGraph[{1, 2, 3}]]]"
      )
      .unwrap(),
      "{1  2, 2  3}"
    );
    assert_eq!(
      interpret(
        "VertexList[GraphIntersection[CompleteGraph[3], PathGraph[{1, 2, 3}]]]"
      )
      .unwrap(),
      "{1, 2, 3}"
    );
  }
  #[test]
  fn graph_intersection_disjoint() {
    // No common edges
    assert_eq!(
      interpret("EdgeList[GraphIntersection[Graph[{1, 2}, {1  2}], Graph[{3, 4}, {3  4}]]]").unwrap(),
      "{}"
    );
    assert_eq!(
      interpret("VertexList[GraphIntersection[Graph[{1, 2}, {1  2}], Graph[{3, 4}, {3  4}]]]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }
  #[test]
  fn graph_intersection_same() {
    // Intersection with itself
    assert_eq!(
      interpret(
        "EdgeList[GraphIntersection[CompleteGraph[3], CompleteGraph[3]]]"
      )
      .unwrap(),
      "{1  2, 1  3, 2  3}"
    );
  }

  // VertexAdd
  #[test]
  fn vertex_add_single() {
    assert_eq!(
      interpret("VertexList[VertexAdd[CompleteGraph[3], 4]]").unwrap(),
      "{1, 2, 3, 4}"
    );
    assert_eq!(
      interpret("EdgeList[VertexAdd[CompleteGraph[3], 4]]").unwrap(),
      "{1  2, 1  3, 2  3}"
    );
  }
  #[test]
  fn vertex_add_multiple() {
    assert_eq!(
      interpret("VertexList[VertexAdd[CompleteGraph[2], {3, 4}]]").unwrap(),
      "{1, 2, 3, 4}"
    );
    assert_eq!(
      interpret("EdgeList[VertexAdd[CompleteGraph[2], {3, 4}]]").unwrap(),
      "{1  2}"
    );
  }
  #[test]
  fn vertex_add_existing_ignored() {
    // Adding an existing vertex does nothing
    assert_eq!(
      interpret("VertexList[VertexAdd[CompleteGraph[3], 1]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("EdgeList[VertexAdd[CompleteGraph[3], 1]]").unwrap(),
      "{1  2, 1  3, 2  3}"
    );
  }
  #[test]
  fn vertex_add_to_directed() {
    assert_eq!(
      interpret("VertexList[VertexAdd[Graph[{1 -> 2}], 3]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("EdgeList[VertexAdd[Graph[{1 -> 2}], 3]]").unwrap(),
      "{1  2}"
    );
  }

  // IndexGraph
  #[test]
  fn index_graph_basic() {
    assert_eq!(
      interpret(r#"VertexList[IndexGraph[Graph[{"a", "b", "c"}, {"a"  "b", "b"  "c"}]]]"#).unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret(r#"EdgeList[IndexGraph[Graph[{"a", "b", "c"}, {"a"  "b", "b"  "c"}]]]"#).unwrap(),
      "{1  2, 2  3}"
    );
  }
  #[test]
  fn index_graph_with_start() {
    assert_eq!(
      interpret(r#"VertexList[IndexGraph[Graph[{"a", "b", "c"}, {"a"  "b", "b"  "c"}], 5]]"#).unwrap(),
      "{5, 6, 7}"
    );
    assert_eq!(
      interpret(r#"EdgeList[IndexGraph[Graph[{"a", "b", "c"}, {"a"  "b", "b"  "c"}], 5]]"#).unwrap(),
      "{5  6, 6  7}"
    );
  }
  #[test]
  fn index_graph_directed() {
    assert_eq!(
      interpret("VertexList[IndexGraph[Graph[{1 -> 2, 3 -> 1}]]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("EdgeList[IndexGraph[Graph[{1 -> 2, 3 -> 1}]]]").unwrap(),
      "{1  2, 3  1}"
    );
  }
  #[test]
  fn index_graph_complete() {
    assert_eq!(
      interpret("VertexList[IndexGraph[CompleteGraph[3], 10]]").unwrap(),
      "{10, 11, 12}"
    );
    assert_eq!(
      interpret("EdgeList[IndexGraph[CompleteGraph[3], 10]]").unwrap(),
      "{10  11, 10  12, 11  12}"
    );
  }

  // DirectedGraphQ
  #[test]
  fn directed_graph_q_true() {
    assert_eq!(
      interpret("DirectedGraphQ[Graph[{1 -> 2, 2 -> 3}]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn directed_graph_q_false_undirected() {
    assert_eq!(
      interpret("DirectedGraphQ[CompleteGraph[3]]").unwrap(),
      "False"
    );
  }
  #[test]
  fn directed_graph_q_non_graph() {
    assert_eq!(interpret("DirectedGraphQ[5]").unwrap(), "False");
  }
  #[test]
  fn directed_graph_q_adjacency_directed() {
    assert_eq!(
      interpret("DirectedGraphQ[AdjacencyGraph[{{0, 1}, {0, 0}}]]").unwrap(),
      "True"
    );
  }

  // TreeGraphQ
  #[test]
  fn tree_graph_q_star() {
    assert_eq!(interpret("TreeGraphQ[StarGraph[4]]").unwrap(), "True");
  }
  #[test]
  fn tree_graph_q_cycle() {
    assert_eq!(interpret("TreeGraphQ[CompleteGraph[3]]").unwrap(), "False");
  }

  // AcyclicGraphQ
  #[test]
  fn acyclic_graph_q_tree() {
    assert_eq!(interpret("AcyclicGraphQ[StarGraph[4]]").unwrap(), "True");
  }

  // ConnectedGraphComponents
  #[test]
  fn connected_graph_components_single() {
    assert_eq!(
      interpret("Length[ConnectedGraphComponents[CompleteGraph[3]]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("VertexList[ConnectedGraphComponents[CompleteGraph[3]][[1]]]")
        .unwrap(),
      "{1, 2, 3}"
    );
  }
  #[test]
  fn connected_graph_components_disconnected() {
    assert_eq!(
      interpret(
        "Length[ConnectedGraphComponents[Graph[{1, 2, 3, 4}, {1  2, 3  4}]]]"
      )
      .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("VertexList[ConnectedGraphComponents[Graph[{1, 2, 3, 4}, {1  2, 3  4}]][[1]]]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("VertexList[ConnectedGraphComponents[Graph[{1, 2, 3, 4}, {1  2, 3  4}]][[2]]]").unwrap(),
      "{3, 4}"
    );
  }
  #[test]
  fn connected_graph_components_directed() {
    assert_eq!(
      interpret(
        "Length[ConnectedGraphComponents[Graph[{1, 2, 3}, {1  2, 2  1}]]]"
      )
      .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("VertexList[ConnectedGraphComponents[Graph[{1, 2, 3}, {1  2, 2  1}]][[1]]]").unwrap(),
      "{1, 2}"
    );
  }

  // EulerianGraphQ
  #[test]
  fn eulerian_graph_q_cycle() {
    // A cycle (all vertices have degree 2) is Eulerian
    assert_eq!(
      interpret("EulerianGraphQ[Graph[{1  2, 2  3, 3  4, 4  1}]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn eulerian_graph_q_complete_odd() {
    // CompleteGraph[3] = K3 is a triangle, all degrees 2, Eulerian
    assert_eq!(
      interpret("EulerianGraphQ[CompleteGraph[3]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn eulerian_graph_q_path_false() {
    // A path graph has vertices of degree 1 at endpoints, not Eulerian
    assert_eq!(
      interpret("EulerianGraphQ[PathGraph[{1, 2, 3}]]").unwrap(),
      "False"
    );
  }
  #[test]
  fn eulerian_graph_q_star_false() {
    // Star graph: center has degree n-1, leaves have degree 1
    assert_eq!(interpret("EulerianGraphQ[StarGraph[4]]").unwrap(), "False");
  }
  #[test]
  fn eulerian_graph_q_non_graph() {
    assert_eq!(interpret("EulerianGraphQ[42]").unwrap(), "False");
  }
  #[test]
  fn eulerian_graph_q_directed() {
    // Directed cycle: each vertex has in-degree 1, out-degree 1
    assert_eq!(
      interpret("EulerianGraphQ[Graph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn eulerian_graph_q_complete_even_false() {
    // CompleteGraph[4] = K4: all vertices have degree 3 (odd), not Eulerian
    assert_eq!(
      interpret("EulerianGraphQ[CompleteGraph[4]]").unwrap(),
      "False"
    );
  }

  // GraphDiameter
  #[test]
  fn graph_diameter_path() {
    assert_eq!(interpret("GraphDiameter[KaryTree[3]]").unwrap(), "2");
  }

  // GraphRadius
  #[test]
  fn graph_radius_star() {
    assert_eq!(interpret("GraphRadius[StarGraph[4]]").unwrap(), "1");
  }

  // GraphCenter
  #[test]
  fn graph_center_star() {
    assert_eq!(interpret("GraphCenter[StarGraph[4]]").unwrap(), "{1}");
  }

  // GraphPeriphery
  #[test]
  fn graph_periphery_star() {
    assert_eq!(
      interpret("GraphPeriphery[StarGraph[4]]").unwrap(),
      "{2, 3, 4}"
    );
  }

  // DegreeCentrality
  #[test]
  fn degree_centrality_complete3() {
    assert_eq!(
      interpret("DegreeCentrality[CompleteGraph[3]]").unwrap(),
      "{2, 2, 2}"
    );
  }
  #[test]
  fn degree_centrality_star4() {
    assert_eq!(
      interpret("DegreeCentrality[StarGraph[4]]").unwrap(),
      "{3, 1, 1, 1}"
    );
  }

  // GraphComplement
  #[test]
  fn graph_complement_star3() {
    // StarGraph[3] = Graph[{1,2,3}, {1<->2, 1<->3}]
    // Complement should have only edge 2<->3
    assert_eq!(
      interpret("EdgeList[GraphComplement[StarGraph[3]]]").unwrap(),
      "{2  3}"
    );
  }

  // GraphPower[g, k] connects vertices within graph distance k.
  #[test]
  fn graph_power_cycle() {
    // The square of a 5-cycle is the complete graph K5 (10 edges).
    assert_eq!(
      interpret("EdgeCount[GraphPower[CycleGraph[5], 2]]").unwrap(),
      "10"
    );
  }
  #[test]
  fn graph_power_path_edge_count() {
    assert_eq!(
      interpret("EdgeCount[GraphPower[PathGraph[{1, 2, 3, 4}], 2]]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("EdgeCount[GraphPower[PathGraph[{1, 2, 3, 4}], 3]]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("VertexCount[GraphPower[CycleGraph[5], 2]]").unwrap(),
      "5"
    );
  }

  // GraphUnion combines the vertex sets (sorted) and edge sets (deduplicated).
  #[test]
  fn graph_union_paths() {
    assert_eq!(
      interpret(
        "VertexList[GraphUnion[PathGraph[{1, 2, 3}], PathGraph[{3, 4, 5}]]]"
      )
      .unwrap(),
      "{1, 2, 3, 4, 5}"
    );
    assert_eq!(
      interpret(
        "EdgeCount[GraphUnion[PathGraph[{1, 2, 3}], PathGraph[{3, 4, 5}]]]"
      )
      .unwrap(),
      "4"
    );
  }
  #[test]
  fn graph_union_dedup_and_sort() {
    // Shared edges are counted once.
    assert_eq!(
      interpret("EdgeCount[GraphUnion[CycleGraph[3], PathGraph[{1, 2, 3}]]]")
        .unwrap(),
      "3"
    );
    // Vertices come back in canonical (sorted) order.
    assert_eq!(
      interpret("VertexList[GraphUnion[Graph[{3, 1}, {}], Graph[{2, 5}, {}]]]")
        .unwrap(),
      "{1, 2, 3, 5}"
    );
    // Variadic: three paths chained share endpoints.
    assert_eq!(
      interpret(
        "EdgeCount[GraphUnion[PathGraph[{1, 2}], PathGraph[{2, 3}], PathGraph[{3, 4}]]]"
      )
      .unwrap(),
      "3"
    );
  }

  // GraphDifference[g1, g2] removes g2's edges from g1, keeping g1's vertices.
  #[test]
  fn graph_difference() {
    assert_eq!(
      interpret(
        "EdgeCount[GraphDifference[CompleteGraph[3], PathGraph[{1, 2, 3}]]]"
      )
      .unwrap(),
      "1"
    );
    assert_eq!(
      interpret(
        "VertexCount[GraphDifference[CompleteGraph[4], CycleGraph[4]]]"
      )
      .unwrap(),
      "4"
    );
    assert_eq!(
      interpret("EdgeCount[GraphDifference[CompleteGraph[4], CycleGraph[4]]]")
        .unwrap(),
      "2"
    );
    // Removing edges absent from g1 changes nothing beyond the shared ones.
    assert_eq!(
      interpret("EdgeCount[GraphDifference[CycleGraph[4], CompleteGraph[4]]]")
        .unwrap(),
      "0"
    );
  }

  // Tree canonicalizes scalar children into leaf Tree[child, None] nodes.
  #[test]
  fn tree_canonicalizes_children() {
    assert_eq!(
      interpret("Tree[1, {2, 3}]").unwrap(),
      "Tree[1, {Tree[2, None], Tree[3, None]}]"
    );
    // A list child is wrapped as leaf data, not recursed into.
    assert_eq!(
      interpret("Tree[1, {2, {3, 4}}]").unwrap(),
      "Tree[1, {Tree[2, None], Tree[{3, 4}, None]}]"
    );
    // A leaf stays as Tree[data, None].
    assert_eq!(interpret("Tree[1, None]").unwrap(), "Tree[1, None]");
  }
  #[test]
  fn tree_data_and_children() {
    assert_eq!(interpret("TreeData[Tree[1, {2, 3}]]").unwrap(), "1");
    assert_eq!(
      interpret("TreeChildren[Tree[1, {2, 3}]]").unwrap(),
      "{Tree[2, None], Tree[3, None]}"
    );
    // Children of a leaf is None.
    assert_eq!(interpret("TreeChildren[Tree[x, None]]").unwrap(), "None");
  }

  // VertexOutComponent
  #[test]
  fn vertex_out_component_star() {
    let result = interpret("VertexOutComponent[StarGraph[4], 1]").unwrap();
    assert_eq!(result, "{1, 2, 3, 4}");
  }

  // ClosenessCentrality
  #[test]
  fn closeness_centrality_complete3() {
    assert_eq!(
      interpret("ClosenessCentrality[CompleteGraph[3]]").unwrap(),
      "{1., 1., 1.}"
    );
  }

  // ButterflyGraph
  #[test]
  fn butterfly_graph_basic() {
    // ButterflyGraph[n] has (n+1)*2^n vertices; ButterflyGraph[2] has 3*4 = 12
    assert_eq!(interpret("VertexCount[ButterflyGraph[2]]").unwrap(), "12");
  }

  // HalfNormalDistribution
  #[test]
  fn half_normal_distribution_basic() {
    assert_eq!(
      interpret("HalfNormalDistribution[1]").unwrap(),
      "HalfNormalDistribution[1]"
    );
  }
  #[test]
  fn half_normal_distribution_symbolic() {
    assert_eq!(
      interpret("HalfNormalDistribution[t]").unwrap(),
      "HalfNormalDistribution[t]"
    );
  }
  #[test]
  fn half_normal_distribution_pdf() {
    assert_eq!(
      interpret("PDF[HalfNormalDistribution[t], x]").unwrap(),
      "Piecewise[{{(2*t)/(E^((t^2*x^2)/Pi)*Pi), x > 0}}, 0]"
    );
  }
  #[test]
  fn half_normal_distribution_cdf() {
    assert_eq!(
      interpret("CDF[HalfNormalDistribution[t], x]").unwrap(),
      "Piecewise[{{Erf[(t*x)/Sqrt[Pi]], x > 0}}, 0]"
    );
  }

  // PrincipalComponents
  #[test]
  fn principal_components_2x2() {
    assert_eq!(
      interpret("PrincipalComponents[{{1, 2}, {3, 4}}]").unwrap(),
      "{{Sqrt[2], 0}, {-Sqrt[2], 0}}"
    );
  }
  #[test]
  fn principal_components_3x2() {
    assert_eq!(
      interpret("PrincipalComponents[{{1, 2}, {3, 4}, {5, 6}}]").unwrap(),
      "{{2*Sqrt[2], 0}, {0, 0}, {-2*Sqrt[2], 0}}"
    );
  }
  #[test]
  fn principal_components_single_row() {
    assert_eq!(
      interpret("PrincipalComponents[{{1, 2}}]").unwrap(),
      "{{0, 0}}"
    );
  }
  #[test]
  fn principal_components_numerical() {
    let result =
      interpret("PrincipalComponents[N[{{1, 2}, {3, 4}, {5, 6}}]]").unwrap();
    assert!(result.contains("2.828427124746"));
  }

  // UnderscriptBox
  #[test]
  fn underscript_box_basic() {
    assert_eq!(
      interpret("UnderscriptBox[x, y]").unwrap(),
      "UnderscriptBox[x, y]"
    );
  }
  #[test]
  fn underscript_box_no_args() {
    assert_eq!(interpret("UnderscriptBox[]").unwrap(), "UnderscriptBox[]");
  }
  #[test]
  fn underscript_box_single_arg() {
    assert_eq!(interpret("UnderscriptBox[x]").unwrap(), "UnderscriptBox[x]");
  }
  #[test]
  fn underscript_box_three_args() {
    assert_eq!(
      interpret("UnderscriptBox[x, y, z]").unwrap(),
      "UnderscriptBox[x, y, z]"
    );
  }

  // MathieuSPrime
  #[test]
  fn mathieu_s_prime_symbolic() {
    assert_eq!(
      interpret("MathieuSPrime[a, q, z]").unwrap(),
      "MathieuSPrime[a, q, z]"
    );
  }
  #[test]
  fn mathieu_s_prime_no_args() {
    assert_eq!(interpret("MathieuSPrime[]").unwrap(), "MathieuSPrime[]");
  }
  #[test]
  fn mathieu_s_prime_in_expr() {
    assert_eq!(
      interpret("2 * MathieuSPrime[a, q, z]").unwrap(),
      "2*MathieuSPrime[a, q, z]"
    );
  }

  // SmithDecomposition
  #[test]
  fn smith_decomposition_2x2() {
    // SmithDecomposition returns {U, S, V} where U.M.V = S
    // S is the Smith normal form
    let result = interpret("SmithDecomposition[{{1, 2}, {3, 4}}]").unwrap();
    // Verify the Smith normal form (diagonal part)
    assert!(result.contains("{{1, 0}, {0, 2}}"));
    // Verify U.M.V = S
    let verify = interpret(
      "Module[{d = SmithDecomposition[{{1, 2}, {3, 4}}]}, d[[1]].{{1, 2}, {3, 4}}.d[[3]] == d[[2]]]",
    )
    .unwrap();
    assert_eq!(verify, "True");
  }
  #[test]
  fn smith_decomposition_identity() {
    assert_eq!(
      interpret("SmithDecomposition[{{1, 0}, {0, 1}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}}"
    );
  }
  #[test]
  fn smith_decomposition_3x3_verify() {
    let verify = interpret(
      "Module[{m = {{2, 4, 4}, {-6, 6, 12}, {10, -4, -16}}, d = SmithDecomposition[{{2, 4, 4}, {-6, 6, 12}, {10, -4, -16}}]}, d[[1]].m.d[[3]] == d[[2]]]",
    )
    .unwrap();
    assert_eq!(verify, "True");
  }

  #[test]
  fn smith_decomposition_destructured_module_pattern() {
    // The pattern shown in the SmithDecomposition usage docs: bind the
    // result, then check that u . m . v == r.
    assert_eq!(
      interpret(
        "Module[{m, u, r, v}, m = {{1, 2}, {3, 4}}; \
         {u, r, v} = SmithDecomposition[m]; u . m . v == r]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn smith_decomposition_non_square() {
    // Single-row matrix: Smith form is {{gcd(2,4,4), 0, 0}} == {{2, 0, 0}}.
    let verify = interpret(
      "Module[{m = {{2, 4, 4}}, d = SmithDecomposition[{{2, 4, 4}}]}, \
       {d[[1]].m.d[[3]] == d[[2]], d[[2]][[1, 1]]}]",
    )
    .unwrap();
    assert_eq!(verify, "{True, 2}");
  }

  #[test]
  fn smith_decomposition_singular_3x3() {
    // {{5,0,0},{0,0,0},{0,0,7}} is rank 2 with elementary divisors gcd=1
    // and lcm=35; the trailing diagonal entry must be 0.
    let verify = interpret(
      "Module[{m = {{5, 0, 0}, {0, 0, 0}, {0, 0, 7}}, \
         d = SmithDecomposition[{{5, 0, 0}, {0, 0, 0}, {0, 0, 7}}]}, \
         {d[[1]].m.d[[3]] == d[[2]], d[[2]][[1, 1]], d[[2]][[2, 2]], d[[2]][[3, 3]]}]"
    )
    .unwrap();
    assert_eq!(verify, "{True, 1, 35, 0}");
  }

  #[test]
  fn smith_decomposition_non_rational_emits_latm() {
    // wolframscript: `SmithDecomposition::latm: Matrix contains an entry
    // that is not rational.` Woxi must surface the same message and leave
    // the call unevaluated.
    let result =
      interpret("SmithDecomposition[{{Sqrt[2], 1}, {0, 1}}]").unwrap();
    assert_eq!(result, "SmithDecomposition[{{Sqrt[2], 1}, {0, 1}}]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains("SmithDecomposition::latm")),
      "expected SmithDecomposition::latm warning, got: {msgs:?}"
    );
  }

  // ChromaticPolynomial
  #[test]
  fn chromatic_polynomial_complete_4() {
    assert_eq!(
      interpret("ChromaticPolynomial[CompleteGraph[4], k]").unwrap(),
      "-6*k + 11*k^2 - 6*k^3 + k^4"
    );
  }
  #[test]
  fn chromatic_polynomial_path_3() {
    assert_eq!(
      interpret("ChromaticPolynomial[PathGraph[{1,2,3}], k]").unwrap(),
      "k - 2*k^2 + k^3"
    );
  }
  #[test]
  fn chromatic_polynomial_star_4() {
    assert_eq!(
      interpret("ChromaticPolynomial[StarGraph[4], k]").unwrap(),
      "-k + 3*k^2 - 3*k^3 + k^4"
    );
  }
  #[test]
  fn chromatic_polynomial_single_vertex() {
    assert_eq!(
      interpret("ChromaticPolynomial[CompleteGraph[1], k]").unwrap(),
      "k"
    );
  }
  #[test]
  fn chromatic_polynomial_cycle_3() {
    // C_3 (triangle): k(k-1)(k-2) = k^3 - 3k^2 + 2k.
    assert_eq!(
      interpret("ChromaticPolynomial[CycleGraph[3], k]").unwrap(),
      "2*k - 3*k^2 + k^3"
    );
  }
  #[test]
  fn chromatic_polynomial_cycle_4() {
    // C_4: (k-1)^4 + (k-1).
    assert_eq!(
      interpret("ChromaticPolynomial[CycleGraph[4], k]").unwrap(),
      "-3*k + 6*k^2 - 4*k^3 + k^4"
    );
  }
  #[test]
  fn chromatic_polynomial_cycle_5() {
    assert_eq!(
      interpret("ChromaticPolynomial[CycleGraph[5], k]").unwrap(),
      "4*k - 10*k^2 + 10*k^3 - 5*k^4 + k^5"
    );
  }

  // ChiDistribution
  #[test]
  fn chi_distribution_basic() {
    assert_eq!(
      interpret("ChiDistribution[3]").unwrap(),
      "ChiDistribution[3]"
    );
  }
  #[test]
  fn chi_distribution_symbolic() {
    assert_eq!(
      interpret("ChiDistribution[n]").unwrap(),
      "ChiDistribution[n]"
    );
  }
  #[test]
  fn chi_distribution_pdf() {
    assert_eq!(
      interpret("PDF[ChiDistribution[n], x]").unwrap(),
      "Piecewise[{{(2^(1 - n/2)*x^(-1 + n))/(E^(x^2/2)*Gamma[n/2]), x > 0}}, 0]"
    );
  }
  #[test]
  fn chi_distribution_cdf() {
    assert_eq!(
      interpret("CDF[ChiDistribution[n], x]").unwrap(),
      "Piecewise[{{GammaRegularized[n/2, 0, x^2/2], x > 0}}, 0]"
    );
  }
  // Generalized 3-argument GammaRegularized: stays symbolic except the
  // elementary a == 1 case. Regression: it previously emitted ::argrx.
  #[test]
  fn gamma_regularized_three_arg() {
    assert_eq!(
      interpret("GammaRegularized[1, 0, 1]").unwrap(),
      "1 - E^(-1)"
    );
    assert_eq!(
      interpret("GammaRegularized[2, 1, 3]").unwrap(),
      "GammaRegularized[2, 1, 3]"
    );
  }

  // ResetDirectory
  #[test]
  fn reset_directory_restores_previous() {
    let result = interpret(
      "old = Directory[]; SetDirectory[\"/tmp\"]; ResetDirectory[]; Directory[] == old",
    )
    .unwrap();
    assert_eq!(result, "True");
  }
  #[test]
  fn reset_directory_returns_restored_dir() {
    let result = interpret(
      "old = Directory[]; SetDirectory[\"/tmp\"]; restored = ResetDirectory[]; restored == old",
    )
    .unwrap();
    assert_eq!(result, "True");
  }

  // MatrixPropertyDistribution
  #[test]
  fn matrix_property_distribution_basic() {
    assert_eq!(
      interpret("MatrixPropertyDistribution[x, y]").unwrap(),
      "MatrixPropertyDistribution[x, y]"
    );
  }
  #[test]
  fn matrix_property_distribution_no_args() {
    assert_eq!(
      interpret("MatrixPropertyDistribution[]").unwrap(),
      "MatrixPropertyDistribution[]"
    );
  }
  #[test]
  fn matrix_property_distribution_single_arg() {
    assert_eq!(
      interpret("MatrixPropertyDistribution[x]").unwrap(),
      "MatrixPropertyDistribution[x]"
    );
  }

  #[test]
  fn rename_file_basic() {
    // Clean up stale files from prior failed runs
    let _ = interpret(r#"Quiet[DeleteFile["/tmp/test_rename_src_woxi.txt"]]"#);
    let _ = interpret(r#"Quiet[DeleteFile["/tmp/test_rename_dst_woxi.txt"]]"#);
    // Create a temp file, rename it, check old doesn't exist and new does, clean up
    let result = interpret(
      r#"Block[{src = CreateFile["/tmp/test_rename_src_woxi.txt"]},
        Close[src];
        RenameFile["/tmp/test_rename_src_woxi.txt", "/tmp/test_rename_dst_woxi.txt"];
        result = {FileExistsQ["/tmp/test_rename_src_woxi.txt"], FileExistsQ["/tmp/test_rename_dst_woxi.txt"]};
        DeleteFile["/tmp/test_rename_dst_woxi.txt"];
        result]"#,
    )
    .unwrap();
    assert_eq!(result, "{False, True}");
  }

  #[test]
  fn rename_file_returns_dest_path() {
    // Clean up stale files from prior failed runs
    let _ = interpret(r#"Quiet[DeleteFile["/tmp/test_rename_ret_woxi.txt"]]"#);
    let _ = interpret(r#"Quiet[DeleteFile["/tmp/test_rename_ret2_woxi.txt"]]"#);
    let result = interpret(
      r#"Block[{src = CreateFile["/tmp/test_rename_ret_woxi.txt"]},
        Close[src];
        result = RenameFile["/tmp/test_rename_ret_woxi.txt", "/tmp/test_rename_ret2_woxi.txt"];
        DeleteFile["/tmp/test_rename_ret2_woxi.txt"];
        result]"#,
    )
    .unwrap();
    assert!(result.contains("test_rename_ret2_woxi.txt"));
  }

  #[test]
  fn rename_file_not_found() {
    let result =
      interpret(r#"RenameFile["nonexistent_woxi_file.txt", "dest.txt"]"#)
        .unwrap();
    assert_eq!(result, "$Failed");
  }

  #[test]
  fn substitution_system_list_mode() {
    assert_eq!(
      interpret("SubstitutionSystem[{0 -> {0, 1}, 1 -> {1, 0}}, {0}, 3]")
        .unwrap(),
      "{{0}, {0, 1}, {0, 1, 1, 0}, {0, 1, 1, 0, 1, 0, 0, 1}}"
    );
  }

  #[test]
  fn substitution_system_list_mode_step0() {
    assert_eq!(
      interpret("SubstitutionSystem[{0 -> {0, 1}, 1 -> {1, 0}}, {0}, 0]")
        .unwrap(),
      "{{0}}"
    );
  }

  #[test]
  fn substitution_system_string_mode() {
    assert_eq!(
      interpret(r#"SubstitutionSystem[{"A" -> "AB", "B" -> "A"}, "A", 4]"#)
        .unwrap(),
      "{A, AB, ABA, ABAAB, ABAABABA}"
    );
  }

  #[test]
  fn substitution_system_string_multi_char_init() {
    assert_eq!(
      interpret(r#"SubstitutionSystem[{"A" -> "AB", "B" -> "A"}, "AB", 2]"#)
        .unwrap(),
      "{AB, ABA, ABAAB}"
    );
  }

  #[test]
  fn substitution_system_symbols() {
    assert_eq!(
      interpret("SubstitutionSystem[{a -> {a, b}, b -> {a}}, {a}, 3]").unwrap(),
      "{{a}, {a, b}, {a, b, a}, {a, b, a, a, b}}"
    );
  }

  #[test]
  fn inverse_gamma_distribution_head() {
    assert_eq!(
      interpret("InverseGammaDistribution[2, 3]").unwrap(),
      "InverseGammaDistribution[2, 3]"
    );
  }

  #[test]
  fn inverse_gamma_distribution_pdf() {
    assert_eq!(
      interpret("PDF[InverseGammaDistribution[a, b], x]").unwrap(),
      "Piecewise[{{(b/x)^a/(E^(b/x)*x*Gamma[a]), x > 0}}, 0]"
    );
  }

  #[test]
  fn inverse_gamma_distribution_cdf() {
    assert_eq!(
      interpret("CDF[InverseGammaDistribution[a, b], x]").unwrap(),
      "Piecewise[{{GammaRegularized[a, b/x], x > 0}}, 0]"
    );
  }

  #[test]
  fn end_of_line_inert() {
    assert_eq!(interpret("EndOfLine[]").unwrap(), "EndOfLine[]");
    assert_eq!(interpret("EndOfLine[x]").unwrap(), "EndOfLine[x]");
  }

  #[test]
  fn row_lines_inert() {
    assert_eq!(interpret("RowLines[True]").unwrap(), "RowLines[True]");
  }

  #[test]
  fn delete_contents_inert() {
    assert_eq!(
      interpret("DeleteContents[True]").unwrap(),
      "DeleteContents[True]"
    );
  }

  #[test]
  fn column_spacings_inert() {
    assert_eq!(interpret("ColumnSpacings[1]").unwrap(), "ColumnSpacings[1]");
  }

  #[test]
  fn criterion_function_inert() {
    assert_eq!(
      interpret("CriterionFunction[x]").unwrap(),
      "CriterionFunction[x]"
    );
  }

  #[test]
  fn interval_markers_inert() {
    assert_eq!(
      interpret("IntervalMarkers[x]").unwrap(),
      "IntervalMarkers[x]"
    );
  }

  #[test]
  fn logistic_distribution_head() {
    assert_eq!(
      interpret("LogisticDistribution[m, s]").unwrap(),
      "LogisticDistribution[m, s]"
    );
  }

  #[test]
  fn logistic_distribution_default() {
    assert_eq!(
      interpret("LogisticDistribution[]").unwrap(),
      "LogisticDistribution[0, 1]"
    );
  }

  #[test]
  fn logistic_distribution_pdf() {
    assert_eq!(
      interpret("PDF[LogisticDistribution[m, s], x]").unwrap(),
      "E^((m - x)/s)/((1 + E^((m - x)/s))^2*s)"
    );
  }

  #[test]
  fn logistic_distribution_cdf() {
    assert_eq!(
      interpret("CDF[LogisticDistribution[m, s], x]").unwrap(),
      "(1 + E^((m - x)/s))^(-1)"
    );
  }

  #[test]
  fn inverse_chi_square_distribution_head() {
    assert_eq!(
      interpret("InverseChiSquareDistribution[5]").unwrap(),
      "InverseChiSquareDistribution[5]"
    );
  }

  #[test]
  fn inverse_chi_square_distribution_pdf() {
    assert_eq!(
      interpret("PDF[InverseChiSquareDistribution[n], x]").unwrap(),
      "Piecewise[{{(x^(-1))^(1 + n/2)/(2^(n/2)*E^(1/(2*x))*Gamma[n/2]), x > 0}}, 0]"
    );
  }

  #[test]
  fn inverse_chi_square_distribution_cdf() {
    assert_eq!(
      interpret("CDF[InverseChiSquareDistribution[n], x]").unwrap(),
      "Piecewise[{{GammaRegularized[n/2, 1/(2*x)], x > 0}}, 0]"
    );
  }

  #[test]
  fn frechet_distribution_head() {
    assert_eq!(
      interpret("FrechetDistribution[2, 3]").unwrap(),
      "FrechetDistribution[2, 3]"
    );
  }

  #[test]
  fn frechet_distribution_pdf() {
    let result = interpret("PDF[FrechetDistribution[a, b], x]").unwrap();
    assert!(result.starts_with("Piecewise["));
    assert!(result.contains("x > 0"));
  }

  #[test]
  fn frechet_distribution_cdf() {
    let result = interpret("CDF[FrechetDistribution[a, b], x]").unwrap();
    assert!(result.starts_with("Piecewise["));
    assert!(result.contains("x > 0"));
  }

  #[test]
  fn frechet_distribution_3arg_head() {
    assert_eq!(
      interpret("FrechetDistribution[2, 3, 5]").unwrap(),
      "FrechetDistribution[2, 3, 5]"
    );
  }

  #[test]
  fn frechet_distribution_3arg_mean() {
    assert_eq!(
      interpret("Mean[FrechetDistribution[a, b, c]]").unwrap(),
      "Piecewise[{{c + b*Gamma[1 - a^(-1)], 1 < a}}, Infinity]"
    );
  }

  #[test]
  fn frechet_distribution_3arg_variance() {
    assert_eq!(
      interpret("Variance[FrechetDistribution[a, b, c]]").unwrap(),
      "Piecewise[{{b^2*(Gamma[1 - 2/a] - Gamma[1 - a^(-1)]^2), a > 2}}, Infinity]"
    );
  }

  #[test]
  fn frechet_distribution_3arg_median() {
    assert_eq!(
      interpret("Median[FrechetDistribution[a, b, c]]").unwrap(),
      "c + b/Log[2]^a^(-1)"
    );
  }

  #[test]
  fn frechet_distribution_3arg_pdf() {
    let result = interpret("PDF[FrechetDistribution[a, b, c], x]").unwrap();
    assert!(result.starts_with("Piecewise["));
    assert!(result.contains("x > c"));
  }

  #[test]
  fn frechet_distribution_3arg_cdf() {
    let result = interpret("CDF[FrechetDistribution[a, b, c], x]").unwrap();
    assert!(result.starts_with("Piecewise["));
    assert!(result.contains("x > c"));
  }

  #[test]
  fn extreme_value_distribution_head() {
    assert_eq!(
      interpret("ExtremeValueDistribution[1, 2]").unwrap(),
      "ExtremeValueDistribution[1, 2]"
    );
  }

  #[test]
  fn extreme_value_distribution_pdf() {
    assert_eq!(
      interpret("PDF[ExtremeValueDistribution[a, b], x]").unwrap(),
      "E^(-E^((a - x)/b) + (a - x)/b)/b"
    );
  }

  #[test]
  fn extreme_value_distribution_cdf() {
    assert_eq!(
      interpret("CDF[ExtremeValueDistribution[a, b], x]").unwrap(),
      "E^(-E^((a - x)/b))"
    );
  }

  #[test]
  fn knight_tour_graph_vertex_count() {
    assert_eq!(
      interpret("Length[VertexList[KnightTourGraph[3, 3]]]").unwrap(),
      "9"
    );
  }

  #[test]
  fn knight_tour_graph_edge_count() {
    assert_eq!(
      interpret("Length[EdgeList[KnightTourGraph[3, 3]]]").unwrap(),
      "8"
    );
  }

  #[test]
  fn knight_tour_graph_5x5() {
    assert_eq!(
      interpret("Length[VertexList[KnightTourGraph[5, 5]]]").unwrap(),
      "25"
    );
  }

  #[test]
  fn knight_tour_graph_edges_2x3() {
    assert_eq!(
      interpret("EdgeList[KnightTourGraph[2, 3]]").unwrap(),
      "{1  6, 3  4}"
    );
  }

  #[test]
  fn file_extension_basic() {
    assert_eq!(interpret(r#"FileExtension["hello.txt"]"#).unwrap(), "txt");
  }

  #[test]
  fn file_extension_multi() {
    assert_eq!(interpret(r#"FileExtension["file.tar.gz"]"#).unwrap(), "gz");
  }

  #[test]
  fn file_extension_none() {
    assert_eq!(interpret(r#"FileExtension["noext"]"#).unwrap(), "");
  }

  #[test]
  fn file_base_name_basic() {
    assert_eq!(interpret(r#"FileBaseName["file.txt"]"#).unwrap(), "file");
  }

  #[test]
  fn file_base_name_with_path() {
    assert_eq!(
      interpret(r#"FileBaseName["/path/to/file.txt"]"#).unwrap(),
      "file"
    );
  }

  #[test]
  fn file_base_name_multi_extension() {
    assert_eq!(
      interpret(r#"FileBaseName["file.tar.gz"]"#).unwrap(),
      "file.tar"
    );
  }

  #[test]
  fn file_base_name_no_extension() {
    assert_eq!(interpret(r#"FileBaseName["noext"]"#).unwrap(), "noext");
  }

  #[test]
  fn file_base_name_hidden_file() {
    assert_eq!(interpret(r#"FileBaseName[".hidden"]"#).unwrap(), ".hidden");
  }

  #[test]
  fn file_base_name_empty() {
    assert_eq!(interpret(r#"FileBaseName[""]"#).unwrap(), "");
  }

  #[test]
  fn file_base_name_non_string() {
    assert_eq!(
      interpret(r#"FileBaseName[123]"#).unwrap(),
      "FileBaseName[123]"
    );
  }

  #[test]
  fn file_base_name_path_with_multi_ext() {
    assert_eq!(interpret(r#"FileBaseName["a/b/c.d.e"]"#).unwrap(), "c.d");
  }

  #[test]
  fn create_directory_basic() {
    let dir = "/tmp/woxi_test_create_dir_basic";
    // Clean up if left over from previous test
    let _ = std::fs::remove_dir_all(dir);
    assert_eq!(
      interpret(&format!(r#"CreateDirectory["{}"]"#, dir)).unwrap(),
      dir
    );
    assert!(std::path::Path::new(dir).is_dir());
    std::fs::remove_dir_all(dir).unwrap();
  }

  #[test]
  fn create_directory_nested() {
    let dir = "/tmp/woxi_test_create_dir_nested/sub1/sub2";
    let base = "/tmp/woxi_test_create_dir_nested";
    let _ = std::fs::remove_dir_all(base);
    assert_eq!(
      interpret(&format!(r#"CreateDirectory["{}"]"#, dir)).unwrap(),
      dir
    );
    assert!(std::path::Path::new(dir).is_dir());
    std::fs::remove_dir_all(base).unwrap();
  }

  #[test]
  fn create_directory_already_exists() {
    let dir = "/tmp/woxi_test_create_dir_exists";
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let result = interpret(&format!(r#"CreateDirectory["{}"]"#, dir)).unwrap();
    assert_eq!(result, "$Failed");
    std::fs::remove_dir_all(dir).unwrap();
  }

  #[test]
  fn create_directory_no_args() {
    let result = interpret("CreateDirectory[]").unwrap();
    // Should return a string path (temp directory)
    assert!(result.starts_with('/'), "Expected a path, got: {}", result);
  }

  #[test]
  fn create_directory_non_string() {
    assert_eq!(interpret("CreateDirectory[123]").unwrap(), "$Failed");
  }

  #[test]
  fn copy_file_basic() {
    let src = "/tmp/woxi_test_copy_src.txt";
    let dst = "/tmp/woxi_test_copy_dst.txt";
    let _ = std::fs::remove_file(dst);
    std::fs::write(src, "hello").unwrap();
    assert_eq!(
      interpret(&format!(r#"CopyFile["{}", "{}"]"#, src, dst)).unwrap(),
      dst
    );
    assert_eq!(std::fs::read_to_string(dst).unwrap(), "hello");
    std::fs::remove_file(src).unwrap();
    std::fs::remove_file(dst).unwrap();
  }

  #[test]
  fn copy_file_source_not_found() {
    let result =
      interpret(r#"CopyFile["/tmp/woxi_nonexistent_file", "/tmp/woxi_out"]"#)
        .unwrap();
    assert_eq!(result, "$Failed");
  }

  #[test]
  fn copy_file_dest_exists() {
    let src = "/tmp/woxi_test_copy_exists_src.txt";
    let dst = "/tmp/woxi_test_copy_exists_dst.txt";
    std::fs::write(src, "hello").unwrap();
    std::fs::write(dst, "world").unwrap();
    let result =
      interpret(&format!(r#"CopyFile["{}", "{}"]"#, src, dst)).unwrap();
    assert_eq!(result, "$Failed");
    std::fs::remove_file(src).unwrap();
    std::fs::remove_file(dst).unwrap();
  }

  #[test]
  fn copy_file_non_string_args() {
    assert_eq!(interpret("CopyFile[x, y]").unwrap(), "CopyFile[x, y]");
  }

  #[test]
  fn betweenness_centrality_line() {
    assert_eq!(
      interpret("BetweennessCentrality[Graph[{1,2,3},{1  2,2  3}]]").unwrap(),
      "{0., 1., 0.}"
    );
  }

  #[test]
  fn betweenness_centrality_star() {
    assert_eq!(
      interpret("BetweennessCentrality[StarGraph[5]]").unwrap(),
      "{6., 0., 0., 0., 0.}"
    );
  }

  #[test]
  fn local_clustering_coefficient_triangle() {
    assert_eq!(
      interpret(
        "LocalClusteringCoefficient[Graph[{1,2,3},{1  2,2  3,1  3}]]"
      )
      .unwrap(),
      "{1, 1, 1}"
    );
  }

  #[test]
  fn local_clustering_coefficient_square() {
    assert_eq!(
      interpret(
        "LocalClusteringCoefficient[Graph[{1,2,3,4},{1  2,2  3,3  4,1  4}]]"
      )
      .unwrap(),
      "{0, 0, 0, 0}"
    );
  }

  #[test]
  fn batch_inert_symbols() {
    // Test a batch of inert symbolic heads
    assert_eq!(interpret("AnyOrder[x, y]").unwrap(), "AnyOrder[x, y]");
    assert_eq!(
      interpret("IntervalMarkersStyle[x]").unwrap(),
      "IntervalMarkersStyle[x]"
    );
    assert_eq!(interpret("HatchFilling[x]").unwrap(), "HatchFilling[x]");
    assert_eq!(
      interpret("IncludeConstantBasis[x]").unwrap(),
      "IncludeConstantBasis[x]"
    );
    assert_eq!(interpret("HeaderLines[x]").unwrap(), "HeaderLines[x]");
    assert_eq!(interpret("SelfLoopStyle[x]").unwrap(), "SelfLoopStyle[x]");
    assert_eq!(interpret("ScaleDivisions[x]").unwrap(), "ScaleDivisions[x]");
    assert_eq!(
      interpret("ColumnAlignments[x]").unwrap(),
      "ColumnAlignments[x]"
    );
    assert_eq!(
      interpret("ExtentElementFunction[x]").unwrap(),
      "ExtentElementFunction[x]"
    );
    assert_eq!(interpret("Subset[x, y]").unwrap(), "x \u{2282} y");
    assert_eq!(interpret("TargetUnits[x]").unwrap(), "TargetUnits[x]");
    assert_eq!(interpret("RowSpacings[x]").unwrap(), "RowSpacings[x]");
    assert_eq!(interpret("PassEventsUp[x]").unwrap(), "PassEventsUp[x]");
    assert_eq!(
      interpret("NormalsFunction[x]").unwrap(),
      "NormalsFunction[x]"
    );
    assert_eq!(interpret("StartOfLine[x]").unwrap(), "StartOfLine[x]");
    assert_eq!(interpret("LeftArrow[x, y]").unwrap(), "x \u{2190} y");
    assert_eq!(interpret("DotEqual[x, y]").unwrap(), "x \u{2250} y");
    assert_eq!(interpret("NumberMarks[x]").unwrap(), "NumberMarks[x]");
  }

  #[test]
  fn weakly_connected_components_directed() {
    assert_eq!(
      interpret("WeaklyConnectedComponents[Graph[{1,2,3,4,5},{1  2,3  4}]]")
        .unwrap(),
      "{{2, 1}, {4, 3}, {5}}"
    );
  }

  #[test]
  fn weakly_connected_components_undirected() {
    assert_eq!(
      interpret("WeaklyConnectedComponents[Graph[{1,2,3,4,5},{1  2,3  4}]]")
        .unwrap(),
      "{{2, 1}, {4, 3}, {5}}"
    );
  }

  #[test]
  fn gompertz_makeham_distribution_head() {
    assert_eq!(
      interpret("GompertzMakehamDistribution[l, x0]").unwrap(),
      "GompertzMakehamDistribution[l, x0]"
    );
  }

  #[test]
  fn gompertz_makeham_distribution_pdf() {
    let result =
      interpret("PDF[GompertzMakehamDistribution[l, x0], x]").unwrap();
    assert!(result.starts_with("Piecewise["));
    assert!(result.contains("x >= 0"));
  }

  #[test]
  fn gompertz_makeham_distribution_cdf() {
    let result =
      interpret("CDF[GompertzMakehamDistribution[l, x0], x]").unwrap();
    assert!(result.starts_with("Piecewise["));
    assert!(result.contains("x >= 0"));
  }

  #[test]
  fn mixed_radix_inert() {
    assert_eq!(
      interpret("MixedRadix[{24, 60, 60}]").unwrap(),
      "MixedRadix[{24, 60, 60}]"
    );
  }

  #[test]
  fn xml_object_inert() {
    assert_eq!(
      interpret(r#"XMLObject["Document"][x]"#).unwrap(),
      "XMLObject[Document][x]"
    );
  }

  #[test]
  fn underoverscript_box_inert() {
    assert_eq!(
      interpret("UnderoverscriptBox[x, y, z]").unwrap(),
      "UnderoverscriptBox[x, y, z]"
    );
  }

  #[test]
  fn inverse_gaussian_distribution_head() {
    assert_eq!(
      interpret("InverseGaussianDistribution[m, l]").unwrap(),
      "InverseGaussianDistribution[m, l]"
    );
  }

  #[test]
  fn inverse_gaussian_distribution_pdf() {
    let result =
      interpret("PDF[InverseGaussianDistribution[m, l], x]").unwrap();
    assert!(result.starts_with("Piecewise["));
    assert!(result.contains("x > 0"));
  }

  #[test]
  fn inverse_gaussian_distribution_cdf() {
    let result =
      interpret("CDF[InverseGaussianDistribution[m, l], x]").unwrap();
    assert!(result.starts_with("Piecewise["));
    assert!(result.contains("x > 0"));
  }
}

mod option_symbols_batch {
  use super::*;

  #[test]
  fn template_slot() {
    assert_eq!(interpret("TemplateSlot[1]").unwrap(), "TemplateSlot[1]");
  }

  #[test]
  fn trace_depth() {
    assert_eq!(interpret("TraceDepth[0]").unwrap(), "TraceDepth[0]");
  }

  #[test]
  fn mean_around() {
    // Non-list arg stays unevaluated.
    assert_eq!(interpret("MeanAround[0]").unwrap(), "MeanAround[0]");
  }

  #[test]
  fn mean_around_list() {
    // MeanAround[{x1, ..., xn}] = Around[N[Mean], N[StdDev/Sqrt[n]]].
    // For {1, 2, 3, 4, 3, 2, 1}: mean = 16/7, stderr = Sqrt[26/147].
    assert_eq!(
      interpret("MeanAround[{1, 2, 3, 4, 3, 2, 1}]").unwrap(),
      "Around[2.2857142857142856, 0.42056004125370694]"
    );
  }

  #[test]
  fn mean_around_real_list() {
    // {1., 2., 3.}: mean = 2, stderr = 1/Sqrt[3] ≈ 0.5773502691896258.
    let result = interpret("MeanAround[{1., 2., 3.}]").unwrap();
    // Match Wolfram's output Around[2., 0.5773502691896258].
    assert!(
      result.starts_with("Around[2.,") && result.contains("0.577350269189"),
      "got {}",
      result
    );
  }

  #[test]
  fn sector_spacing() {
    assert_eq!(interpret("SectorSpacing[0]").unwrap(), "SectorSpacing[0]");
  }

  #[test]
  fn polar_axes_origin() {
    assert_eq!(
      interpret("PolarAxesOrigin[0]").unwrap(),
      "PolarAxesOrigin[0]"
    );
  }

  #[test]
  fn dihedral_group() {
    assert_eq!(interpret("DihedralGroup[3]").unwrap(), "DihedralGroup[3]");
    assert_eq!(interpret("DihedralGroup[10]").unwrap(), "DihedralGroup[10]");
  }

  #[test]
  fn alternating_group() {
    assert_eq!(
      interpret("AlternatingGroup[4]").unwrap(),
      "AlternatingGroup[4]"
    );
  }

  #[test]
  fn watts_strogatz_graph_distribution() {
    assert_eq!(
      interpret("WattsStrogatzGraphDistribution[10, 0.5]").unwrap(),
      "WattsStrogatzGraphDistribution[10, 0.5]"
    );
  }

  #[test]
  fn barabasi_albert_graph_distribution() {
    assert_eq!(
      interpret("BarabasiAlbertGraphDistribution[10, 2]").unwrap(),
      "BarabasiAlbertGraphDistribution[10, 2]"
    );
  }

  #[test]
  fn spatial_graph_distribution() {
    assert_eq!(
      interpret("SpatialGraphDistribution[10, 0.3]").unwrap(),
      "SpatialGraphDistribution[10, 0.3]"
    );
  }

  #[test]
  fn uniform_graph_distribution() {
    assert_eq!(
      interpret("UniformGraphDistribution[10, 20]").unwrap(),
      "UniformGraphDistribution[10, 20]"
    );
  }

  #[test]
  fn bounded_region_q_bounded() {
    assert_eq!(interpret("BoundedRegionQ[Disk[]]").unwrap(), "True");
    assert_eq!(interpret("BoundedRegionQ[Ball[]]").unwrap(), "True");
    assert_eq!(interpret("BoundedRegionQ[Rectangle[]]").unwrap(), "True");
    assert_eq!(interpret("BoundedRegionQ[Circle[]]").unwrap(), "True");
    assert_eq!(interpret("BoundedRegionQ[Sphere[]]").unwrap(), "True");
    assert_eq!(interpret("BoundedRegionQ[Point[{0, 0}]]").unwrap(), "True");
    assert_eq!(
      interpret("BoundedRegionQ[Polygon[{{0,0},{1,0},{0,1}}]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("BoundedRegionQ[Line[{{0,0},{1,1}}]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("BoundedRegionQ[Interval[{0, 1}]]").unwrap(),
      "True"
    );
    assert_eq!(interpret("BoundedRegionQ[Cuboid[]]").unwrap(), "True");
  }

  #[test]
  fn bounded_region_q_unbounded() {
    assert_eq!(
      interpret("BoundedRegionQ[HalfPlane[{0,0},{1,0},{0,1}]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("BoundedRegionQ[InfiniteLine[{0,0},{1,1}]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn tutte_polynomial_empty_graph() {
    assert_eq!(
      interpret("TuttePolynomial[Graph[{1, 2, 3}, {}]][x, y]").unwrap(),
      "1"
    );
  }

  #[test]
  fn tutte_polynomial_single_vertex() {
    assert_eq!(
      interpret("TuttePolynomial[Graph[{1}, {}]][x, y]").unwrap(),
      "1"
    );
  }

  #[test]
  fn tutte_polynomial_single_edge() {
    assert_eq!(
      interpret("TuttePolynomial[Graph[{1, 2}, {1  2}]][x, y]").unwrap(),
      "x"
    );
  }

  #[test]
  fn tutte_polynomial_path() {
    assert_eq!(
      interpret("TuttePolynomial[PathGraph[{1, 2, 3}]][x, y]").unwrap(),
      "x^2"
    );
  }

  #[test]
  fn tutte_polynomial_k3() {
    assert_eq!(
      interpret("TuttePolynomial[CompleteGraph[3]][x, y]").unwrap(),
      "x + x^2 + y"
    );
  }

  #[test]
  fn tutte_polynomial_k4() {
    assert_eq!(
      interpret("TuttePolynomial[CompleteGraph[4]][x, y]").unwrap(),
      "2*x + 3*x^2 + x^3 + 2*y + 4*x*y + 3*y^2 + y^3"
    );
  }
  #[test]
  fn tutte_polynomial_two_arg_form() {
    // wolframscript: TuttePolynomial[CycleGraph[5], {x, y}] = x + x^2 + x^3 + x^4 + y.
    assert_eq!(
      interpret("TuttePolynomial[CycleGraph[5], {x, y}]").unwrap(),
      "x + x^2 + x^3 + x^4 + y"
    );
  }
  #[test]
  fn tutte_polynomial_two_arg_form_triangle() {
    assert_eq!(
      interpret("TuttePolynomial[CycleGraph[3], {x, y}]").unwrap(),
      "x + x^2 + y"
    );
  }
  #[test]
  fn tutte_polynomial_two_arg_form_k4() {
    assert_eq!(
      interpret("TuttePolynomial[CompleteGraph[4], {x, y}]").unwrap(),
      "2*x + 3*x^2 + x^3 + 2*y + 4*x*y + 3*y^2 + y^3"
    );
  }

  #[test]
  fn function_continuous_polynomial() {
    assert_eq!(interpret("FunctionContinuous[x^2, x]").unwrap(), "True");
    assert_eq!(
      interpret("FunctionContinuous[x^2 + 3*x + 1, x]").unwrap(),
      "True"
    );
  }

  #[test]
  fn function_continuous_trig() {
    assert_eq!(interpret("FunctionContinuous[Sin[x], x]").unwrap(), "True");
    assert_eq!(interpret("FunctionContinuous[Cos[x], x]").unwrap(), "True");
    assert_eq!(interpret("FunctionContinuous[Tan[x], x]").unwrap(), "False");
  }

  #[test]
  fn function_continuous_exp() {
    assert_eq!(interpret("FunctionContinuous[Exp[x], x]").unwrap(), "True");
  }

  #[test]
  fn function_continuous_discontinuous() {
    assert_eq!(interpret("FunctionContinuous[1/x, x]").unwrap(), "False");
    assert_eq!(
      interpret("FunctionContinuous[Floor[x], x]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("FunctionContinuous[Sign[x], x]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("FunctionContinuous[Sqrt[x], x]").unwrap(),
      "False"
    );
    assert_eq!(interpret("FunctionContinuous[Log[x], x]").unwrap(), "False");
  }

  #[test]
  fn function_continuous_abs() {
    assert_eq!(interpret("FunctionContinuous[Abs[x], x]").unwrap(), "True");
  }

  #[test]
  fn function_continuous_constant() {
    assert_eq!(interpret("FunctionContinuous[5, x]").unwrap(), "True");
  }

  #[test]
  fn function_continuous_composite() {
    assert_eq!(
      interpret("FunctionContinuous[x^2 + Sin[x], x]").unwrap(),
      "True"
    );
  }

  #[test]
  fn function_continuous_restricted_domain() {
    assert_eq!(
      interpret("FunctionContinuous[{Log[x], x > 0}, x]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("FunctionContinuous[{Sqrt[x], x > 0}, x]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("FunctionContinuous[{1/x, x > 0}, x]").unwrap(),
      "True"
    );
  }

  #[test]
  fn function_continuous_multivariate() {
    assert_eq!(
      interpret("FunctionContinuous[x + y, {x, y}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn function_continuous_fractional_power() {
    assert_eq!(
      interpret("FunctionContinuous[x^(1/3), x]").unwrap(),
      "False"
    );
  }

  #[test]
  fn graph_property_distribution_edge_count_bernoulli() {
    assert_eq!(
      interpret(
        "GraphPropertyDistribution[EdgeCount[g], Distributed[g, BernoulliGraphDistribution[n, p]]]"
      )
      .unwrap(),
      "BinomialDistribution[((-1 + n)*n)/2, p]"
    );
  }

  #[test]
  fn graph_property_distribution_vertex_count_bernoulli() {
    assert_eq!(
      interpret(
        "GraphPropertyDistribution[VertexCount[g], Distributed[g, BernoulliGraphDistribution[n, p]]]"
      )
      .unwrap(),
      "DiscreteUniformDistribution[{n, n}]"
    );
  }

  #[test]
  fn graph_property_distribution_vertex_degree_bernoulli() {
    assert_eq!(
      interpret(
        "GraphPropertyDistribution[VertexDegree[g, 1], Distributed[g, BernoulliGraphDistribution[n, p]]]"
      )
      .unwrap(),
      "BinomialDistribution[-1 + n, p]"
    );
  }

  #[test]
  fn graph_property_distribution_edge_count_uniform() {
    assert_eq!(
      interpret(
        "GraphPropertyDistribution[EdgeCount[g], Distributed[g, UniformGraphDistribution[n, m]]]"
      )
      .unwrap(),
      "DiscreteUniformDistribution[{m, m}]"
    );
  }

  #[test]
  fn graph_property_distribution_vertex_count_uniform() {
    assert_eq!(
      interpret(
        "GraphPropertyDistribution[VertexCount[g], Distributed[g, UniformGraphDistribution[n, m]]]"
      )
      .unwrap(),
      "DiscreteUniformDistribution[{n, n}]"
    );
  }

  #[test]
  fn graph_property_distribution_vertex_degree_uniform() {
    assert_eq!(
      interpret(
        "GraphPropertyDistribution[VertexDegree[g, 1], Distributed[g, UniformGraphDistribution[n, m]]]"
      )
      .unwrap(),
      "HypergeometricDistribution[m, -1 + n, ((-1 + n)*n)/2]"
    );
  }

  #[test]
  fn graph_property_distribution_unevaluated() {
    // Unknown property stays unevaluated
    assert_eq!(
      interpret(
        "GraphPropertyDistribution[SomeProperty[g], Distributed[g, BernoulliGraphDistribution[n, p]]]"
      )
      .unwrap(),
      "GraphPropertyDistribution[SomeProperty[\\[FormalG]], Distributed[\\[FormalG], BernoulliGraphDistribution[n, p]]]"
    );
  }

  #[test]
  fn graph_property_distribution_concrete_values() {
    assert_eq!(
      interpret(
        "GraphPropertyDistribution[VertexCount[g], Distributed[g, BernoulliGraphDistribution[5, 0.3]]]"
      )
      .unwrap(),
      "DiscreteUniformDistribution[{5, 5}]"
    );
  }
}
