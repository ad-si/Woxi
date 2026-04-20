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
    assert_eq!(interpret("RulePlot[x]").unwrap(), "RulePlot[x]");
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
    let result =
      interpret("TreeGraph[{DirectedEdge[1, 2], DirectedEdge[1, 3]}]").unwrap();
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
    assert_eq!(
      interpret("InverseZTransform[x, y, z]").unwrap(),
      "InverseZTransform[x, y, z]"
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
    assert_eq!(
      interpret("ZTransform[x, y, z]").unwrap(),
      "ZTransform[x, y, z]"
    );
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
  fn word_counts_basic() {
    assert_eq!(
      interpret("WordCounts[\"the cat sat on the mat\"]").unwrap(),
      "<|the -> 2, mat -> 1, on -> 1, sat -> 1, cat -> 1|>"
    );
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
  fn sequence_position_symbolic() {
    assert_eq!(
      interpret("SequencePosition[{a, b, c, a, b, c}, {a, b}]").unwrap(),
      "{{1, 2}, {4, 5}}"
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
  fn sequence_cases_three_element_condition() {
    assert_eq!(
      interpret("SequenceCases[Range[10], {x_, y_, z_} /; x + y == z]")
        .unwrap(),
      "{{1, 2, 3}}"
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
  fn key_sort_by_basic() {
    assert_eq!(
      interpret(
        "KeySortBy[<|\"ba\" -> 2, \"a\" -> 1, \"ccc\" -> 3|>, StringLength]"
      )
      .unwrap(),
      "<|a -> 1, ba -> 2, ccc -> 3|>"
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
  #[test]
  fn trimmed_variance_basic() {
    assert_eq!(
      interpret("TrimmedVariance[{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}, 0.2]")
        .unwrap(),
      "7/2"
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

  // Symmetrize
  #[test]
  fn symmetrize_symmetric() {
    // Already symmetric: (M + M^T)/2 = M
    assert_eq!(
      interpret("Symmetrize[{{1, 2}, {2, 3}}]").unwrap(),
      "{{1, 2}, {2, 3}}"
    );
  }
  #[test]
  fn symmetrize_asymmetric() {
    // (M + M^T)/2: {{1, (2+4)/2}, {(4+2)/2, 3}} = {{1, 3}, {3, 3}}
    assert_eq!(
      interpret("Symmetrize[{{1, 2}, {4, 3}}]").unwrap(),
      "{{1, 3}, {3, 3}}"
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
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[1, 4]}"
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

  // KaryTree
  #[test]
  fn kary_tree_binary() {
    assert_eq!(
      interpret("VertexList[KaryTree[7]]").unwrap(),
      "{1, 2, 3, 4, 5, 6, 7}"
    );
    let edges = interpret("EdgeList[KaryTree[7]]").unwrap();
    assert!(edges.contains("UndirectedEdge[1, 2]"));
    assert!(edges.contains("UndirectedEdge[1, 3]"));
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
    assert_eq!(
      interpret("EdgeQ[CompleteGraph[3], UndirectedEdge[1, 2]]").unwrap(),
      "True"
    );
  }
  #[test]
  fn edge_q_false() {
    assert_eq!(
      interpret("EdgeQ[StarGraph[3], UndirectedEdge[2, 3]]").unwrap(),
      "False"
    );
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
      "{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}"
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
      interpret("EdgeList[GraphIntersection[Graph[{1, 2}, {UndirectedEdge[1, 2]}], Graph[{3, 4}, {UndirectedEdge[3, 4]}]]]").unwrap(),
      "{}"
    );
    assert_eq!(
      interpret("VertexList[GraphIntersection[Graph[{1, 2}, {UndirectedEdge[1, 2]}], Graph[{3, 4}, {UndirectedEdge[3, 4]}]]]").unwrap(),
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
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[2, 3]}"
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
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[2, 3]}"
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
      "{UndirectedEdge[1, 2]}"
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
      "{UndirectedEdge[1, 2], UndirectedEdge[1, 3], UndirectedEdge[2, 3]}"
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
      "{DirectedEdge[1, 2]}"
    );
  }

  // IndexGraph
  #[test]
  fn index_graph_basic() {
    assert_eq!(
      interpret(r#"VertexList[IndexGraph[Graph[{"a", "b", "c"}, {UndirectedEdge["a", "b"], UndirectedEdge["b", "c"]}]]]"#).unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret(r#"EdgeList[IndexGraph[Graph[{"a", "b", "c"}, {UndirectedEdge["a", "b"], UndirectedEdge["b", "c"]}]]]"#).unwrap(),
      "{UndirectedEdge[1, 2], UndirectedEdge[2, 3]}"
    );
  }
  #[test]
  fn index_graph_with_start() {
    assert_eq!(
      interpret(r#"VertexList[IndexGraph[Graph[{"a", "b", "c"}, {UndirectedEdge["a", "b"], UndirectedEdge["b", "c"]}], 5]]"#).unwrap(),
      "{5, 6, 7}"
    );
    assert_eq!(
      interpret(r#"EdgeList[IndexGraph[Graph[{"a", "b", "c"}, {UndirectedEdge["a", "b"], UndirectedEdge["b", "c"]}], 5]]"#).unwrap(),
      "{UndirectedEdge[5, 6], UndirectedEdge[6, 7]}"
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
      "{DirectedEdge[1, 2], DirectedEdge[3, 1]}"
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
      "{UndirectedEdge[10, 11], UndirectedEdge[10, 12], UndirectedEdge[11, 12]}"
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
      interpret("Length[ConnectedGraphComponents[Graph[{1, 2, 3, 4}, {UndirectedEdge[1, 2], UndirectedEdge[3, 4]}]]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("VertexList[ConnectedGraphComponents[Graph[{1, 2, 3, 4}, {UndirectedEdge[1, 2], UndirectedEdge[3, 4]}]][[1]]]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("VertexList[ConnectedGraphComponents[Graph[{1, 2, 3, 4}, {UndirectedEdge[1, 2], UndirectedEdge[3, 4]}]][[2]]]").unwrap(),
      "{3, 4}"
    );
  }
  #[test]
  fn connected_graph_components_directed() {
    assert_eq!(
      interpret("Length[ConnectedGraphComponents[Graph[{1, 2, 3}, {DirectedEdge[1, 2], DirectedEdge[2, 1]}]]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("VertexList[ConnectedGraphComponents[Graph[{1, 2, 3}, {DirectedEdge[1, 2], DirectedEdge[2, 1]}]][[1]]]").unwrap(),
      "{1, 2}"
    );
  }

  // EulerianGraphQ
  #[test]
  fn eulerian_graph_q_cycle() {
    // A cycle (all vertices have degree 2) is Eulerian
    assert_eq!(
      interpret("EulerianGraphQ[Graph[{UndirectedEdge[1, 2], UndirectedEdge[2, 3], UndirectedEdge[3, 4], UndirectedEdge[4, 1]}]]").unwrap(),
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
      "{UndirectedEdge[2, 3]}"
    );
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
      "{UndirectedEdge[1, 6], UndirectedEdge[3, 4]}"
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
      interpret("BetweennessCentrality[Graph[{1,2,3},{UndirectedEdge[1,2],UndirectedEdge[2,3]}]]").unwrap(),
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
      interpret("LocalClusteringCoefficient[Graph[{1,2,3},{UndirectedEdge[1,2],UndirectedEdge[2,3],UndirectedEdge[1,3]}]]")
        .unwrap(),
      "{1, 1, 1}"
    );
  }

  #[test]
  fn local_clustering_coefficient_square() {
    assert_eq!(
      interpret(
        "LocalClusteringCoefficient[Graph[{1,2,3,4},{UndirectedEdge[1,2],UndirectedEdge[2,3],UndirectedEdge[3,4],UndirectedEdge[1,4]}]]"
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
      interpret("WeaklyConnectedComponents[Graph[{1,2,3,4,5},{DirectedEdge[1,2],DirectedEdge[3,4]}]]")
        .unwrap(),
      "{{2, 1}, {4, 3}, {5}}"
    );
  }

  #[test]
  fn weakly_connected_components_undirected() {
    assert_eq!(
      interpret("WeaklyConnectedComponents[Graph[{1,2,3,4,5},{UndirectedEdge[1,2],UndirectedEdge[3,4]}]]")
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
    assert_eq!(interpret("MeanAround[0]").unwrap(), "MeanAround[0]");
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
      interpret("TuttePolynomial[Graph[{1, 2}, {UndirectedEdge[1, 2]}]][x, y]")
        .unwrap(),
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
