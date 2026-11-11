use super::*;

mod classify_tests {
  use super::*;

  /// Two well-separated point clouds, the training set every method below
  /// must separate.
  const TRAIN: &str = "<|\"a\" -> {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}, \
     \"b\" -> {{5, 5}, {6, 5}, {5, 6}, {6, 6}, {5.5, 5.5}}|>";

  const METHODS: &[&str] = &[
    "LogisticRegression",
    "NaiveBayes",
    "NearestNeighbors",
    "NeuralNetwork",
    "RandomForest",
    "SupportVectorMachine",
    "DecisionTree",
  ];

  #[test]
  fn every_method_separates_two_clusters() {
    for m in METHODS {
      clear_state();
      let code = format!(
        "c = Classify[{TRAIN}, Method -> \"{m}\"]; \
         {{c[{{0.2, 0.3}}], c[{{5.8, 5.1}}], c[{{{{0, 0}}, {{6, 6}}}}]}}"
      );
      assert_eq!(interpret(&code).unwrap(), "{a, b, {a, b}}", "method {m}");
    }
  }

  #[test]
  fn probabilities_are_a_distribution_over_the_classes() {
    for m in METHODS {
      clear_state();
      let code = format!(
        "c = Classify[{TRAIN}, Method -> \"{m}\"]; \
         p = c[{{2.5, 2.7}}, \"Probabilities\"]; \
         {{Keys[p], Abs[Total[p] - 1] < 10^-9, AllTrue[Values[p], 0 <= # <= 1 &], \
           c[{{0, 0}}, {{\"Probability\", \"a\"}}] > 0.5, \
           c[{{6, 6}}, {{\"Probability\", \"a\"}}] < 0.5}}"
      );
      assert_eq!(
        interpret(&code).unwrap(),
        "{{a, b}, True, True, True, True}",
        "method {m}"
      );
    }
  }

  #[test]
  fn classes_are_sorted_and_method_is_reported() {
    clear_state();
    assert_eq!(
      interpret(
        "c = Classify[{3 -> \"z\", 1 -> \"x\", 2 -> \"y\", 3.2 -> \"z\"}, \
         Method -> \"NearestNeighbors\"]; {c[\"Classes\"], c[\"Method\"]}"
      )
      .unwrap(),
      "{{x, y, z}, NearestNeighbors}"
    );
  }

  #[test]
  fn scalar_examples_and_parallel_lists() {
    clear_state();
    assert_eq!(
      interpret(
        "c = Classify[{1, 2, 3, 10, 11, 12} -> {0, 0, 0, 1, 1, 1}]; \
         {c[1.5], c[11.5], c[{0, 20}], c[\"Method\"]}"
      )
      .unwrap(),
      "{0, 1, {0, 1}, LogisticRegression}"
    );
  }

  #[test]
  fn decision_and_top_probabilities() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "c = Classify[{TRAIN}, Method -> \"LogisticRegression\"]; \
         {{c[{{0, 0}}, \"Decision\"], First[First[c[{{6, 6}}, \"TopProbabilities\"]]], \
           Length[c[{{3, 3}}, {{\"TopProbabilities\", 1}}]]}}"
      ))
      .unwrap(),
      "{a, b, 1}"
    );
  }

  #[test]
  fn more_than_two_classes() {
    for m in METHODS {
      clear_state();
      let code = format!(
        "c = Classify[<|1 -> {{{{0, 0}}, {{0, 1}}, {{1, 0}}}}, \
           2 -> {{{{10, 0}}, {{11, 0}}, {{10, 1}}}}, \
           3 -> {{{{0, 10}}, {{1, 10}}, {{0, 11}}}}|>, Method -> \"{m}\"]; \
         c[{{{{0.5, 0.5}}, {{10.5, 0.5}}, {{0.5, 10.5}}}}]"
      );
      assert_eq!(interpret(&code).unwrap(), "{1, 2, 3}", "method {m}");
    }
  }

  #[test]
  fn training_is_deterministic() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "c1 = Classify[{TRAIN}, Method -> \"RandomForest\"]; \
         c2 = Classify[{TRAIN}, Method -> \"RandomForest\"]; \
         c1[{{2.9, 3.1}}, \"Probabilities\"] === c2[{{2.9, 3.1}}, \"Probabilities\"]"
      ))
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn classifier_measurements_on_a_test_set() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "c = Classify[{TRAIN}, Method -> \"SupportVectorMachine\"]; \
         cm = ClassifierMeasurements[c, <|\"a\" -> {{{{0.1, 0.2}}, {{6, 6}}}}, \
           \"b\" -> {{{{5, 5.2}}, {{5.9, 5.3}}}}|>]; \
         {{cm[\"Accuracy\"], cm[\"Error\"], cm[\"ConfusionMatrix\"], \
           cm[\"Recall\"], cm[\"AccuracyBaseline\"]}}"
      ))
      .unwrap(),
      "{0.75, 0.25, {{1, 1}, {0, 2}}, <|a -> 0.5, b -> 1.|>, 0.5}"
    );
  }

  #[test]
  fn classifier_measurements_with_a_property_argument() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "c = Classify[{TRAIN}]; \
         {{ClassifierMeasurements[c, {{{{0, 0}} -> \"a\", {{6, 6}} -> \"b\"}}, \"Accuracy\"], \
           \"classification accuracy: \" <> ToString[ClassifierMeasurements[c, \
             {{{{0, 0}}, {{6, 6}}}} -> {{\"a\", \"b\"}}, \"Accuracy\"]]}}"
      ))
      .unwrap(),
      "{1., classification accuracy: 1.}"
    );
  }

  #[test]
  fn classifier_inside_contour_plot() {
    clear_state();
    assert_eq!(
      interpret(&format!(
        "c = Classify[{TRAIN}, Method -> \"NeuralNetwork\"]; \
         Head[ContourPlot[c[{{x, y}}, {{\"Probability\", \"b\"}}], {{x, 0, 6}}, {{y, 0, 6}}, \
           Contours -> {{0.5}}, PlotPoints -> 10]]"
      ))
      .unwrap(),
      "Graphics"
    );
  }

  #[test]
  fn unknown_method_stays_unevaluated() {
    clear_state();
    assert_eq!(
      interpret(
        "Head[Classify[{1 -> \"a\", 2 -> \"b\"}, Method -> \"Bogus\"]]"
      )
      .unwrap(),
      "Classify"
    );
    assert!(
      woxi::get_captured_warnings()
        .iter()
        .any(|w| w.contains("Classify::bdmtd"))
    );
  }
}
