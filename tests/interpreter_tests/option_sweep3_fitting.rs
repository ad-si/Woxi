// Model-fitting data forms, weighted least squares and matrix-decomposition
// argument forms found missing by the reference-page parameter sweep.

use super::*;

/// `interpret(code)` must evaluate to `True`.
fn assert_true(code: &str) {
  assert_eq!(interpret(code).unwrap(), "True", "{code}");
}

mod linear_model_fit_weights {
  use super::*;

  // y = x^2 weighted by 1/x: the weighted normal equations give
  // a = -415800/30791, b = 291137/30791 exactly.
  const FIT: &str =
    "lm = LinearModelFit[Range[10]^2, x, x, Weights -> 1/Range[10]];";

  #[test]
  fn best_fit_is_the_weighted_least_squares_fit() {
    assert_true(&format!(
      "{FIT} Max[Abs[lm[\"BestFitParameters\"] - \
       {{-415800/30791, 291137/30791}}]] < 10^-10"
    ));
    assert_true(&format!(
      "{FIT} Abs[(Normal[lm] /. x -> 2) - (-415800 + 2 291137)/30791] < 10^-10"
    ));
  }

  #[test]
  fn statistics_are_weighted() {
    // R² around the weighted mean, the weighted residual sum of squares
    // over the error degrees of freedom.
    assert_true(&format!(
      "{FIT} Abs[lm[\"RSquared\"] - 700502089/748375255] < 10^-12"
    ));
    assert_true(&format!(
      "{FIT} Abs[lm[\"EstimatedVariance\"] - 1962015/123164] < 10^-10"
    ));
    assert_true(&format!(
      "{FIT} Abs[lm[\"ANOVATableEntries\"][[2, 3]] - 1962015/123164] < 10^-10"
    ));
    // The weighted hat matrix still has trace = number of parameters.
    assert_true(&format!(
      "{FIT} Abs[Total[lm[\"HatDiagonal\"]] - 2] < 10^-12"
    ));
    // Parameter errors: Sqrt[Diagonal[σ² Inverse[Xᵀ W X]]].
    assert_true(&format!(
      "{FIT} x0 = Transpose[{{ConstantArray[1, 10], Range[10]}}]; \
       c = 1962015/123164 Inverse[Transpose[x0] . DiagonalMatrix[1/Range[10]] . x0]; \
       Max[Abs[lm[\"ParameterErrors\"] - Sqrt[Diagonal[c]]]] < 10^-10"
    ));
    assert_true(&format!(
      "{FIT} Max[Abs[Flatten[lm[\"CovarianceMatrix\"] - c]]] < 10^-10"
    ));
  }

  #[test]
  fn residuals_are_on_the_original_scale() {
    assert_true(&format!(
      "{FIT} Max[Abs[lm[\"FitResiduals\"] - (Range[10]^2 - \
       (-415800 + 291137 Range[10])/30791)]] < 10^-10"
    ));
  }

  #[test]
  fn unit_weights_change_nothing() {
    assert_true(
      "Normal[LinearModelFit[{1, 4, 3, 6}, x, x, Weights -> {1, 1, 1, 1}]] \
       === Normal[LinearModelFit[{1, 4, 3, 6}, x, x]]",
    );
    assert_true(
      "Normal[LinearModelFit[{1, 4, 3, 6}, x, x, Weights -> Automatic]] \
       === Normal[LinearModelFit[{1, 4, 3, 6}, x, x]]",
    );
  }

  #[test]
  fn unusable_weights_stay_unevaluated() {
    assert_eq!(
      interpret("Head[LinearModelFit[{1, 4, 3, 6}, x, x, Weights -> {1, 2}]]")
        .unwrap(),
      "LinearModelFit"
    );
    assert_eq!(
      interpret("Head[LinearModelFit[{1, 4, 3, 6}, x, x, Weights -> f]]")
        .unwrap(),
      "LinearModelFit"
    );
  }
}

mod fit_data_as_rules {
  use super::*;

  #[test]
  fn list_of_rules() {
    assert_true(
      "Normal[LinearModelFit[{{1, 1} -> 2, {2, 2} -> 3, {3, 1} -> 6, \
       {3, 2} -> 5}, {x, y}, {x, y}]] === Normal[LinearModelFit[{{1, 1, 2}, \
       {2, 2, 3}, {3, 1, 6}, {3, 2, 5}}, {x, y}, {x, y}]]",
    );
    assert_true(
      "Normal[LinearModelFit[{1 -> 1, 2 -> 4, 3 -> 3, 4 -> 6}, x, x]] === \
       Normal[LinearModelFit[{{1, 1}, {2, 4}, {3, 3}, {4, 6}}, x, x]]",
    );
  }

  #[test]
  fn rule_of_lists() {
    assert_true(
      "Normal[LinearModelFit[{1, 2, 3, 4, 5} -> {4, 4, 2, -2, -8}, {1, x^2}, \
       {x}]] === Normal[LinearModelFit[{{1, 4}, {2, 4}, {3, 2}, {4, -2}, \
       {5, -8}}, {1, x^2}, {x}]]",
    );
    // 5.705882352941177 - 0.5187165775401069 x^2: 97/17 - 97/187 x^2.
    assert_true(
      "Max[Abs[LinearModelFit[{1, 2, 3, 4, 5} -> {4, 4, 2, -2, -8}, \
       {1, x^2}, {x}][\"BestFitParameters\"] - {97/17, -97/187}]] < 10^-12",
    );
    assert_true(
      "Normal[LinearModelFit[{{1, 1}, {2, 2}, {3, 1}, {3, 2}} -> {2, 3, 6, 5}, \
       {x, y}, {x, y}]] === Normal[LinearModelFit[{{1, 1, 2}, {2, 2, 3}, \
       {3, 1, 6}, {3, 2, 5}}, {x, y}, {x, y}]]",
    );
  }

  #[test]
  fn response_column() {
    for k in ["1", "-3"] {
      assert_true(&format!(
        "Normal[LinearModelFit[{{{{2, 1, 1}}, {{3, 2, 2}}, {{6, 3, 1}}, \
         {{5, 3, 2}}}} -> {k}, {{x, y}}, {{x, y}}]] === \
         Normal[LinearModelFit[{{{{1, 1, 2}}, {{2, 2, 3}}, {{3, 1, 6}}, \
         {{3, 2, 5}}}}, {{x, y}}, {{x, y}}]]"
      ));
    }
    assert_true(
      "Normal[LinearModelFit[{{1, 1, 2}, {2, 2, 3}, {3, 1, 6}, {3, 2, 5}} -> 3, \
       {x, y}, {x, y}]] === Normal[LinearModelFit[{{1, 1, 2}, {2, 2, 3}, \
       {3, 1, 6}, {3, 2, 5}}, {x, y}, {x, y}]]",
    );
  }

  #[test]
  fn malformed_rule_data_stays_as_written() {
    assert_eq!(
      interpret("Head[LinearModelFit[{1, 2, 3} -> {1, 2}, x, x]]").unwrap(),
      "LinearModelFit"
    );
    assert_eq!(
      interpret("LinearModelFit[{{1, 2}, {3, 4}} -> 5, x, x]").unwrap(),
      "LinearModelFit[{{1, 2}, {3, 4}} -> 5, x, x]"
    );
  }

  #[test]
  fn nonlinear_model_fit() {
    let rows = "Normal[NonlinearModelFit[{{0, 0}, {1, 1}, {2, 2}, {3, 3}, \
                {4, 3}, {5, 4}}, Log[a + b x^2], {a, b}, {x}]]";
    assert_true(&format!(
      "Normal[NonlinearModelFit[{{0, 1, 2, 3, 4, 5}} -> {{0, 1, 2, 3, 3, 4}}, \
       Log[a + b x^2], {{a, b}}, {{x}}]] === {rows}"
    ));
    assert_true(&format!(
      "Normal[NonlinearModelFit[{{0 -> 0, 1 -> 1, 2 -> 2, 3 -> 3, 4 -> 3, \
       5 -> 4}}, Log[a + b x^2], {{a, b}}, x]] === {rows}"
    ));
  }
}

mod qr_decomposition_full {
  use super::*;

  #[test]
  fn machine_tall_matrix_gets_a_square_q() {
    let m = "{{1., 2}, {3, 4}, {5, 6}}";
    assert_true(&format!(
      "{{q, r}} = QRDecomposition[{m}, Full]; Dimensions /@ {{q, r}} == \
       {{{{3, 3}}, {{3, 2}}}}"
    ));
    assert_true(&format!(
      "{{q, r}} = QRDecomposition[{m}, Full]; \
       Max[Abs[Flatten[Transpose[q] . r - {m}]]] < 10^-12 && \
       Max[Abs[Flatten[q . Transpose[q] - IdentityMatrix[3]]]] < 10^-12"
    ));
    // The leading rows are the economy decomposition's, R gains zero rows.
    assert_true(&format!(
      "{{q, r}} = QRDecomposition[{m}, Full]; {{q0, r0}} = \
       QRDecomposition[{m}]; Take[q, 2] == q0 && Take[r, 2] == r0 && \
       r[[3]] == {{0., 0.}}"
    ));
  }

  #[test]
  fn square_or_wide_matrix_is_already_full() {
    assert_true(
      "QRDecomposition[{{1, 2}, {3, 4}}, Full] === \
       QRDecomposition[{{1, 2}, {3, 4}}]",
    );
    assert_true(
      "QRDecomposition[{{1., 2, 3}, {4, 5, 6}}, Full] === \
       QRDecomposition[{{1., 2, 3}, {4, 5, 6}}]",
    );
  }

  #[test]
  fn exact_tall_matrix_is_not_completed() {
    assert_eq!(
      interpret("QRDecomposition[{{1, 2}, {3, 4}, {5, 6}}, Full]").unwrap(),
      "QRDecomposition[{{1, 2}, {3, 4}, {5, 6}}, Full]"
    );
  }
}

mod decomposition_target_structure {
  use super::*;

  #[test]
  fn lu_dense_and_sparse() {
    assert_eq!(
      interpret(
        "LUDecomposition[{{1, 2}, {3, 4}}, TargetStructure -> \"Dense\"]"
      )
      .unwrap(),
      "{{{1, 0}, {3, 1}}, {{1, 2}, {0, -2}}, {{1, 0}, {0, 1}}, 0}"
    );
    assert_true(
      "lu = LUDecomposition[{{1, 2}, {3, 4}}, TargetStructure -> \"Sparse\"]; \
       Head /@ Most[lu] === {SparseArray, SparseArray, SparseArray} && \
       Normal /@ Most[lu] === Most[LUDecomposition[{{1, 2}, {3, 4}}, \
       TargetStructure -> \"Dense\"]] && Last[lu] === 0",
    );
    assert_true(
      "LUDecomposition[{{1, 2}, {3, 4}}, TargetStructure -> Automatic] === \
       LUDecomposition[{{1, 2}, {3, 4}}]",
    );
  }

  #[test]
  fn other_decompositions() {
    for (call, m) in [
      ("QRDecomposition", "{{1, 2}, {3, 4}}"),
      ("SchurDecomposition", "{{1., 2}, {3, 4}}"),
      (
        "HessenbergDecomposition",
        "{{1., 2, 3}, {3, 4, 5}, {1, 1, 1}}",
      ),
      ("SingularValueDecomposition", "{{1., 2}, {3, 4}}"),
    ] {
      assert_true(&format!(
        "{call}[{m}, TargetStructure -> \"Dense\"] === {call}[{m}]"
      ));
      assert_true(&format!(
        "Normal /@ {call}[{m}, TargetStructure -> \"Sparse\"] === {call}[{m}]"
      ));
    }
  }

  #[test]
  fn structured_targets_stay_unevaluated() {
    assert_eq!(
      interpret(
        "LUDecomposition[{{1, 2}, {3, 4}}, TargetStructure -> \"Structured\"]"
      )
      .unwrap(),
      "LUDecomposition[{{1, 2}, {3, 4}}, TargetStructure -> Structured]"
    );
  }
}
