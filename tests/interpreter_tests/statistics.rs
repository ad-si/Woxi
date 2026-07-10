use super::*;

// Statistics functions operate on an association's values, like wolframscript.
mod association_statistics {
  use super::*;

  #[test]
  fn geometric_mean() {
    assert_eq!(
      interpret("GeometricMean[<|a -> 1, b -> 2, c -> 4|>]").unwrap(),
      "2"
    );
  }

  #[test]
  fn harmonic_mean() {
    assert_eq!(
      interpret("HarmonicMean[<|a -> 1, b -> 2, c -> 4|>]").unwrap(),
      "12/7"
    );
  }

  #[test]
  fn root_mean_square() {
    assert_eq!(
      interpret("RootMeanSquare[<|a -> 3, b -> 4|>]").unwrap(),
      "5/Sqrt[2]"
    );
  }

  #[test]
  fn kurtosis() {
    assert_eq!(
      interpret("Kurtosis[<|a -> 1, b -> 2, c -> 3, d -> 4|>]").unwrap(),
      "41/25"
    );
  }

  #[test]
  fn skewness_numeric() {
    // The symbolic form is Woxi's canonical one; the value matches Wolfram.
    let v: f64 = interpret("N[Skewness[<|a -> 1, b -> 2, c -> 4|>]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 0.381_801_774_160_606).abs() < 1e-12, "got {}", v);
  }

  #[test]
  fn quartiles() {
    assert_eq!(
      interpret(
        "Quartiles[<|a -> 1, b -> 2, c -> 3, d -> 4, e -> 5, f -> 6, g -> 7, h -> 8|>]"
      )
      .unwrap(),
      "{5/2, 9/2, 13/2}"
    );
  }

  #[test]
  fn interquartile_range() {
    assert_eq!(
      interpret(
        "InterquartileRange[<|a -> 1, b -> 2, c -> 3, d -> 4, e -> 5|>]"
      )
      .unwrap(),
      "5/2"
    );
  }

  #[test]
  fn trimmed_mean() {
    assert_eq!(
      interpret("TrimmedMean[<|a -> 1, b -> 2, c -> 3, d -> 100|>, 1/4]")
        .unwrap(),
      "5/2"
    );
  }

  #[test]
  fn central_moment() {
    assert_eq!(
      interpret("CentralMoment[<|a -> 1, b -> 2, c -> 3|>, 2]").unwrap(),
      "2/3"
    );
  }
}

mod variance {
  use super::*;

  #[test]
  fn variance_integers() {
    assert_eq!(interpret("Variance[{1, 2, 3}]").unwrap(), "1");
    assert_eq!(interpret("Variance[{1, 2, 3, 4, 5}]").unwrap(), "5/2");
  }

  #[test]
  fn variance_large_integers_no_overflow() {
    // Regression: the exact integer path squared values in i128, so large
    // entries (whose squares exceed i128) panicked with "multiply with
    // overflow". It now computes in BigInt. Variance of {a, 2a, 3a} is a^2.
    assert_eq!(
      interpret("Variance[{10^20, 2*10^20, 3*10^20}]").unwrap(),
      "10000000000000000000000000000000000000000"
    );
    // StandardDeviation = Sqrt[Variance]; a BigInteger variance must take the
    // square root rather than echoing unevaluated.
    assert_eq!(
      interpret("StandardDeviation[{10^20, 2*10^20, 3*10^20}]").unwrap(),
      "100000000000000000000"
    );
  }

  #[test]
  fn variance_reals() {
    let result = interpret("Variance[{1.0, 2.0, 3.0}]").unwrap();
    // Should be 1 or close to it
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
  }

  #[test]
  fn variance_four_integers() {
    assert_eq!(interpret("Variance[{2, 4, 6, 8}]").unwrap(), "20/3");
  }
}

mod standard_deviation {
  use super::*;

  #[test]
  fn stddev_integers() {
    assert_eq!(interpret("StandardDeviation[{1, 2, 3}]").unwrap(), "1");
    assert_eq!(
      interpret("StandardDeviation[{1, 2, 3, 4, 5}]").unwrap(),
      "Sqrt[5/2]"
    );
  }

  #[test]
  fn stddev_reals() {
    let result = interpret("StandardDeviation[{1.0, 2.0, 3.0}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
  }

  #[test]
  fn stddev_perfect_square_variance() {
    // Variance of {1,3,5,7} = 20/3, not perfect square
    // Variance of {0, 2} = 2, StdDev = Sqrt[2]
    assert_eq!(interpret("StandardDeviation[{0, 2}]").unwrap(), "Sqrt[2]");
  }
}

mod geometric_mean {
  use super::*;

  #[test]
  fn geometric_mean_perfect_result() {
    assert_eq!(interpret("GeometricMean[{2, 8}]").unwrap(), "4");
  }

  #[test]
  fn geometric_mean_large_perfect_power() {
    // Regression: the product (10^20)^3 = 10^60 exceeds i128, and the cube
    // root was left unevaluated / mis-extracted (the i128->u64 perfect-root
    // factorization truncated). It must equal 10^20.
    assert_eq!(
      interpret("GeometricMean[{10^20, 10^20, 10^20}]").unwrap(),
      "100000000000000000000"
    );
  }

  #[test]
  fn geometric_mean_real_element_keeps_real() {
    // A Real element makes the result inexact even when it is a whole number
    // (regression: Woxi returned Integer 6 instead of Real 6.).
    assert_eq!(interpret("GeometricMean[{4.0, 9}]").unwrap(), "6.");
    assert_eq!(interpret("GeometricMean[{4, 9.0}]").unwrap(), "6.");
    assert_eq!(interpret("GeometricMean[{2.0, 2, 2}]").unwrap(), "2.");
    assert_eq!(interpret("GeometricMean[{1.0, 1, 1}]").unwrap(), "1.");
    // All-exact input still yields an exact result.
    assert_eq!(interpret("GeometricMean[{2, 8}]").unwrap(), "4");
  }

  // An empty list stays unevaluated rather than raising an error (matching
  // wolframscript). The same holds for HarmonicMean and RootMeanSquare.
  #[test]
  fn empty_list_unevaluated() {
    assert_eq!(interpret("GeometricMean[{}]").unwrap(), "GeometricMean[{}]");
    assert_eq!(interpret("HarmonicMean[{}]").unwrap(), "HarmonicMean[{}]");
    assert_eq!(
      interpret("RootMeanSquare[{}]").unwrap(),
      "RootMeanSquare[{}]"
    );
  }

  #[test]
  fn geometric_mean_exact_irrational() {
    // (1*2*4*8)^(1/4) = 64^(1/4) = 2*Sqrt[2]
    assert_eq!(
      interpret("GeometricMean[{1, 2, 4, 8}]").unwrap(),
      "2*Sqrt[2]"
    );
  }

  #[test]
  fn geometric_mean_symbolic() {
    assert_eq!(
      interpret("GeometricMean[{a, b, c}]").unwrap(),
      "(a*b*c)^(1/3)"
    );
  }

  #[test]
  fn geometric_mean_reals() {
    let result = interpret("GeometricMean[{1.0, 2.0, 3.0}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.8171205928321397).abs() < 1e-10);
  }

  #[test]
  fn geometric_mean_matrix_columnwise() {
    // Audit case: GeometricMean of a list of lists computes column-wise.
    // Col 1: (5*2*4*12)^(1/4) = 480^(1/4) = 2*30^(1/4)
    // Col 2: (10*1*3*15)^(1/4) = 450^(1/4) = 2^(1/4)*Sqrt[15]
    assert_eq!(
      interpret("GeometricMean[{{5, 10}, {2, 1}, {4, 3}, {12, 15}}]").unwrap(),
      "{2*30^(1/4), 2^(1/4)*Sqrt[15]}"
    );
  }

  #[test]
  fn geometric_mean_matrix_simple() {
    // Col 1: (1*4)^(1/2) = 2; Col 2: (2*8)^(1/2) = 4.
    assert_eq!(
      interpret("GeometricMean[{{1, 2}, {4, 8}}]").unwrap(),
      "{2, 4}"
    );
  }

  #[test]
  fn geometric_mean_matrix_3xn() {
    // Col 1: (1*2*4)^(1/3) = 2; Col 2: (2*4*8)^(1/3) = 4; Col 3: (3*9*27)^(1/3) = 9.
    assert_eq!(
      interpret("GeometricMean[{{1, 2, 3}, {2, 4, 9}, {4, 8, 27}}]").unwrap(),
      "{2, 4, 9}"
    );
  }
}

mod harmonic_mean {
  use super::*;

  #[test]
  fn harmonic_mean_integers() {
    assert_eq!(interpret("HarmonicMean[{1, 2, 3}]").unwrap(), "18/11");
  }

  #[test]
  fn harmonic_mean_reals() {
    let result = interpret("HarmonicMean[{1.0, 2.0, 3.0}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.6363636363636365).abs() < 1e-10);
  }

  #[test]
  fn harmonic_mean_equal_elements() {
    assert_eq!(interpret("HarmonicMean[{5, 5, 5}]").unwrap(), "5");
  }
}

mod contraharmonic_mean {
  use super::*;

  #[test]
  fn default_is_total_of_squares_over_total() {
    assert_eq!(interpret("ContraharmonicMean[{1, 2, 3, 4}]").unwrap(), "3");
    assert_eq!(interpret("ContraharmonicMean[{2, 4, 6}]").unwrap(), "14/3");
    assert_eq!(interpret("ContraharmonicMean[{5}]").unwrap(), "5");
  }

  #[test]
  fn reals_and_symbols() {
    let result = interpret("ContraharmonicMean[{1.5, 2.5, 3.5}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.7666666666666666).abs() < 1e-12);
    assert_eq!(
      interpret("ContraharmonicMean[{a, b}]").unwrap(),
      "(a^2 + b^2)/(a + b)"
    );
  }

  #[test]
  fn columnwise_on_a_matrix() {
    assert_eq!(
      interpret("ContraharmonicMean[{{1, 2}, {3, 4}}]").unwrap(),
      "{5/2, 10/3}"
    );
  }

  #[test]
  fn lehmer_mean_with_exponent() {
    // ContraharmonicMean[list, p] = Total[list^p] / Total[list^(p-1)].
    assert_eq!(
      interpret("ContraharmonicMean[{1, 2, 3, 4}, 3]").unwrap(),
      "10/3"
    );
    // p = 1 is the arithmetic mean; p = 0 is the harmonic mean.
    assert_eq!(
      interpret("ContraharmonicMean[{1, 2, 3, 4}, 1]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("ContraharmonicMean[{1, 2, 3, 4}, 0]").unwrap(),
      "48/25"
    );
    assert_eq!(
      interpret("ContraharmonicMean[{a, b}, 3]").unwrap(),
      "(a^3 + b^3)/(a^2 + b^2)"
    );
  }

  #[test]
  fn invalid_inputs_stay_unevaluated() {
    assert_eq!(
      interpret("ContraharmonicMean[{}]").unwrap(),
      "ContraharmonicMean[{}]"
    );
    assert_eq!(
      interpret("ContraharmonicMean[5]").unwrap(),
      "ContraharmonicMean[5]"
    );
    // The exponent must be a scalar.
    assert_eq!(
      interpret("ContraharmonicMean[{1, 2}, {3, 4}]").unwrap(),
      "ContraharmonicMean[{1, 2}, {3, 4}]"
    );
  }
}

mod absolute_correlation {
  use super::*;

  // AbsoluteCorrelation[v1, v2] = Mean[v1 * Conjugate[v2]].
  #[test]
  fn two_vectors() {
    assert_eq!(
      interpret("AbsoluteCorrelation[{1, 2, 3}, {2, 4, 6}]").unwrap(),
      "28/3"
    );
    assert_eq!(
      interpret("AbsoluteCorrelation[{1, 2, 3, 4}, {4, 3, 2, 1}]").unwrap(),
      "5"
    );
  }

  #[test]
  fn single_argument_is_self_correlation() {
    assert_eq!(interpret("AbsoluteCorrelation[{1, 2, 3}]").unwrap(), "14/3");
  }

  #[test]
  fn symbolic_keeps_conjugate() {
    assert_eq!(
      interpret("AbsoluteCorrelation[{a, b}, {c, d}]").unwrap(),
      "(a*Conjugate[c] + b*Conjugate[d])/2"
    );
  }

  #[test]
  fn complex_uses_conjugate_of_second() {
    assert_eq!(
      interpret("AbsoluteCorrelation[{3 + 4 I, 1}, {1, 2 I}]").unwrap(),
      "3/2 + I"
    );
  }

  #[test]
  fn mismatched_lengths_stay_unevaluated() {
    assert_eq!(
      interpret("AbsoluteCorrelation[{1, 2}, {3, 4, 5}]").unwrap(),
      "AbsoluteCorrelation[{1, 2}, {3, 4, 5}]"
    );
  }

  // Matrix form: AbsoluteCorrelation[m] is the p×p matrix
  // (Transpose[m] . Conjugate[m]) / n where rows are observations.
  #[test]
  fn single_matrix_gives_correlation_matrix() {
    assert_eq!(
      interpret("AbsoluteCorrelation[{{1, 2}, {3, 4}, {5, 6}}]").unwrap(),
      "{{35/3, 44/3}, {44/3, 56/3}}"
    );
    // Two observations of three variables → 3×3 matrix.
    assert_eq!(
      interpret("AbsoluteCorrelation[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "{{17/2, 11, 27/2}, {11, 29/2, 18}, {27/2, 18, 45/2}}"
    );
  }

  // Two-matrix form: (Transpose[m1] . Conjugate[m2]) / n, a p×q matrix.
  #[test]
  fn two_matrices_give_cross_correlation_matrix() {
    assert_eq!(
      interpret(
        "AbsoluteCorrelation[{{1, 2}, {3, 4}, {5, 6}}, {{1, 1}, {2, 2}, {3, 3}}]"
      )
      .unwrap(),
      "{{22/3, 22/3}, {28/3, 28/3}}"
    );
  }

  // Float entries are divided per-entry, matching wolframscript to the last
  // ULP (matrix/n would double-round via Times[_, 1/n]).
  #[test]
  fn float_matrix_matches_per_entry_division() {
    assert_eq!(
      interpret("AbsoluteCorrelation[{{1., 2.}, {3., 4.}, {5., 6.}}]").unwrap(),
      "{{11.666666666666666, 14.666666666666666}, {14.666666666666666, 18.666666666666668}}"
    );
  }

  // A matrix paired with a vector is an invalid argument pair.
  #[test]
  fn matrix_vector_mix_stays_unevaluated() {
    assert_eq!(
      interpret("AbsoluteCorrelation[{{1, 2}, {3, 4}}, {1, 2}]").unwrap(),
      "AbsoluteCorrelation[{{1, 2}, {3, 4}}, {1, 2}]"
    );
  }
}

mod quartile_deviation {
  use super::*;

  #[test]
  fn half_the_interquartile_range() {
    // QuartileDeviation = (Q3 - Q1) / 2.
    assert_eq!(
      interpret("QuartileDeviation[{1, 2, 3, 4, 5, 6, 7, 8}]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("QuartileDeviation[{1, 3, 5, 7, 9}]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("QuartileDeviation[{2, 4, 4, 4, 5, 5, 7, 9}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn reals_stay_real() {
    assert_eq!(
      interpret("QuartileDeviation[{1.0, 2.0, 3.0, 4.0, 5.0}]").unwrap(),
      "1.25"
    );
  }

  #[test]
  fn single_element_is_zero() {
    assert_eq!(interpret("QuartileDeviation[{5}]").unwrap(), "0");
  }
}

mod spearman_rho {
  use super::*;

  #[test]
  fn rank_correlation_no_ties() {
    assert_eq!(
      interpret("SpearmanRho[{1, 2, 3, 4, 5}, {2, 3, 1, 5, 4}]").unwrap(),
      "3/5"
    );
    // Identical and reversed rankings give +1 and -1.
    assert_eq!(
      interpret("SpearmanRho[{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("SpearmanRho[{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}]").unwrap(),
      "-1"
    );
  }

  #[test]
  fn handles_ties_with_average_ranks() {
    assert_eq!(
      interpret("SpearmanRho[{1, 1, 2, 3}, {1, 2, 2, 4}]").unwrap(),
      "5/6"
    );
    // Ties can make the exact result irrational.
    assert_eq!(
      interpret("SpearmanRho[{1, 2, 2, 2, 3}, {5, 4, 4, 3, 1}]").unwrap(),
      "-4/Sqrt[19]"
    );
  }

  #[test]
  fn mismatched_or_non_numeric_stays_unevaluated() {
    assert_eq!(
      interpret("SpearmanRho[{1, 2, 3}, {1, 2}]").unwrap(),
      "SpearmanRho[{1, 2, 3}, {1, 2}]"
    );
    assert_eq!(
      interpret("SpearmanRho[{a, b, c}, {1, 2, 3}]").unwrap(),
      "SpearmanRho[{a, b, c}, {1, 2, 3}]"
    );
  }
}

mod kendall_tau {
  use super::*;

  #[test]
  fn rank_correlation_no_ties() {
    assert_eq!(
      interpret("KendallTau[{1, 2, 3, 4, 5}, {2, 1, 4, 3, 5}]").unwrap(),
      "3/5"
    );
    // Identical and reversed orderings give +1 and -1.
    assert_eq!(
      interpret("KendallTau[{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("KendallTau[{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}]").unwrap(),
      "-1"
    );
    // A single discordant pair out of three.
    assert_eq!(
      interpret("KendallTau[{1, 2, 3}, {1, 3, 2}]").unwrap(),
      "1/3"
    );
    // Monotone shift is perfectly concordant.
    assert_eq!(
      interpret("KendallTau[{2, 4, 6, 8}, {1, 3, 5, 7}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn ties_use_tau_b_correction() {
    // Tie corrections make the denominator a perfect square here.
    assert_eq!(
      interpret("KendallTau[{1, 1, 2, 3}, {1, 2, 2, 3}]").unwrap(),
      "4/5"
    );
    // Ties can make the exact result irrational.
    assert_eq!(
      interpret("KendallTau[{1, 1, 1, 2}, {1, 2, 3, 4}]").unwrap(),
      "1/Sqrt[2]"
    );
  }

  #[test]
  fn machine_real_inputs_give_real_result() {
    assert_eq!(
      interpret("KendallTau[{1.5, 2.3, 3.1, 4.8}, {2.1, 1.9, 5.0, 4.2}]")
        .unwrap(),
      "0.3333333333333333"
    );
  }

  #[test]
  fn zero_variance_is_indeterminate() {
    // Constant data in either vector cannot be ranked.
    assert_eq!(
      interpret("KendallTau[{5, 5, 5}, {1, 2, 3}]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn mismatched_or_non_numeric_stays_unevaluated() {
    assert_eq!(
      interpret("KendallTau[{1, 2, 3}, {1, 2}]").unwrap(),
      "KendallTau[{1, 2, 3}, {1, 2}]"
    );
    assert_eq!(
      interpret("KendallTau[{a, b, c}, {1, 2, 3}]").unwrap(),
      "KendallTau[{a, b, c}, {1, 2, 3}]"
    );
  }
}

mod quartile_skewness {
  use super::*;

  #[test]
  fn bowley_skewness() {
    // (Q1 - 2 Q2 + Q3) / (Q3 - Q1).
    assert_eq!(
      interpret("QuartileSkewness[{1, 2, 3, 4, 5, 6, 7, 8}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("QuartileSkewness[{1, 1, 1, 2, 10}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("QuartileSkewness[{2, 4, 4, 4, 5, 5, 7, 9}]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn reals() {
    assert_eq!(
      interpret("QuartileSkewness[{1.0, 2.0, 3.0, 4.0, 8.0}]").unwrap(),
      "0.23076923076923078"
    );
  }

  #[test]
  fn coincident_quartiles_are_indeterminate() {
    // Q3 == Q1 gives 0/0 -> Indeterminate.
    assert_eq!(
      interpret("QuartileSkewness[{5, 5, 5, 5}]").unwrap(),
      "Indeterminate"
    );
  }
}

mod quartiles_edge_cases {
  use super::*;

  // Regression: Quartiles of a single-element list used to panic with an
  // index underflow; it now returns the element repeated, matching
  // wolframscript.
  #[test]
  fn single_element() {
    assert_eq!(interpret("Quartiles[{5}]").unwrap(), "{5, 5, 5}");
    assert_eq!(interpret("InterquartileRange[{5}]").unwrap(), "0");
  }

  // Regression: machine-real inputs must produce machine-real quartiles, not
  // a mix of exact rationals and reals.
  #[test]
  fn real_inputs_stay_real() {
    assert_eq!(
      interpret("Quartiles[{1.0, 2.0, 3.0, 4.0, 5.0}]").unwrap(),
      "{1.75, 3., 4.25}"
    );
    assert_eq!(
      interpret("InterquartileRange[{1.0, 2.0, 3.0, 4.0, 5.0}]").unwrap(),
      "2.5"
    );
    // An interpolated quartile that lands on an integral value still prints
    // as a machine real (5., not 5).
    assert_eq!(
      interpret("Quartiles[{1.0, 2.0, 3.0, 4.0, 8.0}]").unwrap(),
      "{1.75, 3., 5.}"
    );
  }

  // Integer inputs still give exact rational quartiles.
  #[test]
  fn integer_inputs_stay_exact() {
    assert_eq!(
      interpret("Quartiles[{1, 2, 3, 4, 5, 6, 7, 8}]").unwrap(),
      "{5/2, 9/2, 13/2}"
    );
  }

  // Matrix input is reduced columnwise (Map[f, Transpose[matrix]]), matching
  // wolframscript. Regression: previously returned unevaluated.
  #[test]
  fn matrix_columnwise() {
    assert_eq!(
      interpret("Quartiles[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}]").unwrap(),
      "{{3/2, 5/2, 7/2}, {15, 25, 35}}"
    );
    assert_eq!(
      interpret("Quartiles[{{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}}]")
        .unwrap(),
      "{{7/4, 3, 17/4}, {35/2, 30, 85/2}}"
    );
    assert_eq!(
      interpret("InterquartileRange[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}]")
        .unwrap(),
      "{2, 20}"
    );
    assert_eq!(
      interpret("QuartileDeviation[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}]")
        .unwrap(),
      "{1, 10}"
    );
    assert_eq!(
      interpret("QuartileSkewness[{{1, 10}, {2, 20}, {3, 30}, {4, 40}}]")
        .unwrap(),
      "{0, 0}"
    );
  }
}

mod root_mean_square {
  use super::*;

  #[test]
  fn rms_integers() {
    assert_eq!(
      interpret("RootMeanSquare[{1, 2, 3}]").unwrap(),
      "Sqrt[14/3]"
    );
  }

  #[test]
  fn rms_perfect_result() {
    // RMS of {3, 4} = Sqrt[(9+16)/2] = Sqrt[25/2]
    // RMS of {1, 1} = 1
    assert_eq!(interpret("RootMeanSquare[{1, 1}]").unwrap(), "1");
  }

  #[test]
  fn rms_reals() {
    let result = interpret("RootMeanSquare[{1.0, 2.0, 3.0}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.160246899469287).abs() < 1e-10);
  }

  // The radical is reduced to Wolfram's form rather than left as Sqrt[25/2].
  #[test]
  fn rms_reduces_rational_radical() {
    assert_eq!(interpret("RootMeanSquare[{3, 4}]").unwrap(), "5/Sqrt[2]");
  }

  #[test]
  fn rms_extracts_square_factor() {
    assert_eq!(
      interpret("RootMeanSquare[{2, 4, 6}]").unwrap(),
      "2*Sqrt[14/3]"
    );
  }

  // Integer (denominator 1) non-perfect-square result is reduced too.
  #[test]
  fn rms_integer_radical_reduced() {
    assert_eq!(interpret("RootMeanSquare[{6, 8}]").unwrap(), "5*Sqrt[2]");
  }
}

mod rescale {
  use super::*;

  #[test]
  fn rescale_basic() {
    assert_eq!(interpret("Rescale[5, {0, 10}]").unwrap(), "1/2");
    assert_eq!(interpret("Rescale[0, {0, 10}]").unwrap(), "0");
    assert_eq!(interpret("Rescale[10, {0, 10}]").unwrap(), "1");
  }

  #[test]
  fn rescale_with_target_range() {
    assert_eq!(interpret("Rescale[5, {0, 10}, {-1, 1}]").unwrap(), "0");
  }

  #[test]
  fn rescale_list() {
    assert_eq!(
      interpret("Rescale[{1, 2, 3, 4, 5}]").unwrap(),
      "{0, 1/4, 1/2, 3/4, 1}"
    );
  }

  #[test]
  fn rescale_boundary() {
    assert_eq!(interpret("Rescale[3, {1, 5}]").unwrap(), "1/2");
  }

  // Rescale[list] uses the GLOBAL min/max across all (possibly nested)
  // elements and preserves the list structure.
  #[test]
  fn rescale_nested_list() {
    assert_eq!(
      interpret("Rescale[{{1, 2}, {3, 4}}]").unwrap(),
      "{{0, 1/3}, {2/3, 1}}"
    );
    assert_eq!(
      interpret("Rescale[{{0, 5}, {10, 15}}]").unwrap(),
      "{{0, 1/3}, {2/3, 1}}"
    );
  }

  #[test]
  fn rescale_rank_three_list() {
    assert_eq!(
      interpret("Rescale[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}]").unwrap(),
      "{{{0, 1/7}, {2/7, 3/7}}, {{4/7, 5/7}, {6/7, 1}}}"
    );
  }

  // Degenerate (zero-range) nested data maps every element to 0.
  #[test]
  fn rescale_nested_all_equal() {
    assert_eq!(
      interpret("Rescale[{{1, 1}, {1, 1}}]").unwrap(),
      "{{0, 0}, {0, 0}}"
    );
  }

  #[test]
  fn rescale_list_with_range() {
    // Rescale should thread over list first argument
    assert_eq!(
      interpret("Rescale[{0, 5, 10}, {0, 10}]").unwrap(),
      "{0, 1/2, 1}"
    );
    assert_eq!(
      interpret("Rescale[{2, 4, 6}, {0, 10}, {0, 100}]").unwrap(),
      "{20, 40, 60}"
    );
  }

  #[test]
  fn rescale_preserves_exact_rational() {
    // A rational input must stay exact, not be floatified to 0.333…
    assert_eq!(interpret("Rescale[1/3, {0, 1}]").unwrap(), "1/3");
    assert_eq!(interpret("Rescale[1/3, {1, 4}]").unwrap(), "-2/9");
  }

  // An inexact list gives an inexact result even at whole-number values:
  // the min and max rescale to the machine reals 0. and 1., not the exact
  // Integers 0 and 1 (which num_to_expr would have produced).
  #[test]
  fn rescale_inexact_list_stays_real() {
    assert_eq!(interpret("Rescale[{1., 2., 3.}]").unwrap(), "{0., 0.5, 1.}");
    assert_eq!(interpret("Rescale[{1., 3.}]").unwrap(), "{0., 1.}");
    // A single inexact entry makes the whole result inexact.
    assert_eq!(interpret("Rescale[{1, 2, 3.}]").unwrap(), "{0., 0.5, 1.}");
    // An exact list is unchanged.
    assert_eq!(interpret("Rescale[{1, 2, 3}]").unwrap(), "{0, 1/2, 1}");
  }

  // Degenerate (zero-range) inexact data maps to the machine real 0.,
  // while exact data maps to the Integer 0.
  #[test]
  fn rescale_inexact_degenerate_stays_real() {
    assert_eq!(interpret("Rescale[{2., 2.}]").unwrap(), "{0., 0.}");
    assert_eq!(interpret("Rescale[{2, 2}]").unwrap(), "{0, 0}");
  }

  #[test]
  fn rescale_symbolic_constant_kept_exact() {
    // Symbolic real numerics rescale exactly (xmin = 0).
    assert_eq!(interpret("Rescale[Pi, {0, 10}]").unwrap(), "Pi/10");
    assert_eq!(interpret("Rescale[Sqrt[2], {0, 2}]").unwrap(), "1/Sqrt[2]");
  }

  #[test]
  fn rescale_symbolic_variable() {
    assert_eq!(interpret("Rescale[x, {0, 10}]").unwrap(), "x/10");
    assert_eq!(interpret("Rescale[x, {0, 10}, {5, 15}]").unwrap(), "5 + x");
  }

  #[test]
  fn rescale_real_input_stays_real() {
    assert_eq!(interpret("Rescale[2.0, {0, 10}]").unwrap(), "0.2");
  }
}

mod bin_counts {
  use super::*;

  #[test]
  fn bin_counts_explicit_bins() {
    assert_eq!(
      interpret("BinCounts[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 10, 2}]")
        .unwrap(),
      "{1, 2, 2, 2, 2}"
    );
  }

  #[test]
  fn bin_counts_partial_trailing_bin_dropped() {
    // (max - min) not a multiple of dx: only whole bins are counted; the
    // leftover values 9 and 10 fall in [9, 12), which exceeds max, so they
    // are dropped rather than added to the last bin.
    assert_eq!(
      interpret("BinCounts[Range[10], {0, 10, 3}]").unwrap(),
      "{2, 3, 3}"
    );
    assert_eq!(
      interpret("BinCounts[Range[10], {0, 10, 4}]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn bin_counts_with_dx() {
    // Bins aligned to dx multiples: [0,2),[2,4),[4,6) = {1,2,2}
    assert_eq!(
      interpret("BinCounts[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "{1, 2, 2}"
    );
    assert_eq!(interpret("BinCounts[{3, 4, 5}, 2]").unwrap(), "{1, 2}");
  }

  #[test]
  fn bin_counts_values_at_boundaries() {
    // 5 goes in [5,10), 10 is excluded from [0,10)
    assert_eq!(
      interpret("BinCounts[{0, 5, 10}, {0, 10, 5}]").unwrap(),
      "{1, 1}"
    );
  }

  #[test]
  fn bin_counts_large_dataset() {
    assert_eq!(
      interpret("BinCounts[{50, 98, 94, 45, 31, 44, 28, 69, 74, 63, 15, 83, 19, 8, 32, 84, 98, 38, 89, 90, 97, 69, 54, 46, 18, 77, 82, 72, 72, 62, 47, 83, 90, 95, 78, 25, 54, 40, 69, 76, 70, 43, 56, 99, 11, 10, 82, 10, 62, 87, 94, 19, 74, 81, 16, 59, 23, 28, 85, 7, 78, 80, 7, 43, 66, 71, 36, 2, 70, 9, 87, 31, 68, 62, 34, 58, 66, 91, 34, 86, 22, 93, 92, 57, 88, 95, 5, 22, 47, 36, 38, 30, 91, 43, 67, 7, 87, 41, 36, 35}, {0, 100, 10}]").unwrap(),
      "{7, 8, 6, 12, 10, 7, 11, 11, 14, 14}"
    );
  }

  #[test]
  fn bin_counts_empty_bins() {
    assert_eq!(
      interpret("BinCounts[{1, 9}, {0, 10, 2}]").unwrap(),
      "{1, 0, 0, 0, 1}"
    );
  }

  #[test]
  fn bin_counts_reals() {
    assert_eq!(
      interpret("BinCounts[{0.0, 0.5, 1.0, 1.5, 2.0}, {0, 3, 1}]").unwrap(),
      "{2, 2, 1}"
    );
  }

  #[test]
  fn bin_counts_out_of_range() {
    // Values outside [min, max) are not counted
    assert_eq!(
      interpret("BinCounts[{-5, 0, 5, 15}, {0, 10, 5}]").unwrap(),
      "{1, 1}"
    );
  }

  #[test]
  fn bin_counts_symbolic_returns_unevaluated() {
    assert_eq!(
      interpret("BinCounts[x, {0, 10, 1}]").unwrap(),
      "BinCounts[x, {0, 10, 1}]"
    );
  }

  #[test]
  fn bin_counts_explicit_edges() {
    // {{e1, e2, ..., en}} — bins [e_i, e_{i+1}).
    assert_eq!(
      interpret(
        "BinCounts[{1, 3, 2, 1, 4, 5, 6, 2}, {{-Infinity, 2, 5, 7, Infinity}}]"
      )
      .unwrap(),
      "{2, 4, 2, 0}"
    );
  }

  #[test]
  fn bin_counts_explicit_edges_finite() {
    assert_eq!(
      interpret("BinCounts[{1, 2, 3, 4, 5}, {{1, 2, 5}}]").unwrap(),
      "{1, 3}"
    );
  }

  #[test]
  fn bin_counts_explicit_edges_real_values() {
    assert_eq!(
      interpret("BinCounts[{2.5, 5.0, 5.5}, {{2, 5, 6}}]").unwrap(),
      "{1, 2}"
    );
  }
}

mod bin_lists {
  use super::*;

  #[test]
  fn bin_lists_explicit_bins() {
    assert_eq!(
      interpret("BinLists[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {0, 10, 2}]")
        .unwrap(),
      "{{1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}}"
    );
  }

  #[test]
  fn bin_lists_partial_trailing_bin_dropped() {
    // (max - min) is not a multiple of dx: only whole bins [0,3),[3,6),[6,9)
    // are produced; the leftover values 9 and 10 fall outside and are dropped.
    assert_eq!(
      interpret("BinLists[Range[10], {0, 10, 3}]").unwrap(),
      "{{1, 2}, {3, 4, 5}, {6, 7, 8}}"
    );
    // dx = 4 → bins [0,4),[4,8); the [8,12) range exceeds max, so 9 and 10
    // are dropped (and only two bins exist).
    assert_eq!(
      interpret("BinLists[Range[10], {0, 10, 4}]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6, 7}}"
    );
  }

  #[test]
  fn bin_lists_with_dx() {
    assert_eq!(
      interpret("BinLists[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "{{1}, {2, 3}, {4, 5}}"
    );
  }

  #[test]
  fn bin_lists_out_of_range() {
    assert_eq!(
      interpret("BinLists[{-5, 0, 5, 15}, {0, 10, 5}]").unwrap(),
      "{{0}, {5}}"
    );
  }

  #[test]
  fn bin_lists_empty_bins() {
    assert_eq!(
      interpret("BinLists[{1, 9}, {0, 10, 2}]").unwrap(),
      "{{1}, {}, {}, {}, {9}}"
    );
  }

  #[test]
  fn bin_lists_symbolic_returns_unevaluated() {
    assert_eq!(
      interpret("BinLists[x, {0, 10, 1}]").unwrap(),
      "BinLists[x, {0, 10, 1}]"
    );
  }

  // The `vectmat` message text matches Wolfram verbatim, including the
  // hyphenation in "unit-compatible".
  #[test]
  fn bin_lists_symbolic_emits_vectmat_message() {
    clear_state();
    let result = interpret_with_stdout("BinLists[x, {0, 10, 1}]").unwrap();
    assert_eq!(result.result, "BinLists[x, {0, 10, 1}]");
    assert!(result.warnings.iter().any(|w| w.contains(
      "BinLists::vectmat: The first argument is expected to be a \
       unit-compatible vector or a matrix with unit-compatible columns."
    )));
  }

  #[test]
  fn bin_lists_explicit_edges() {
    assert_eq!(
      interpret(
        "BinLists[{1, 3, 2, 1, 4, 5, 6, 2}, {{-Infinity, 2, 5, 7, Infinity}}]"
      )
      .unwrap(),
      "{{1, 1}, {3, 2, 4, 2}, {5, 6}, {}}"
    );
  }

  #[test]
  fn bin_lists_explicit_edges_finite() {
    assert_eq!(
      interpret("BinLists[{1, 2, 3, 4, 5}, {{1, 2, 5}}]").unwrap(),
      "{{1}, {2, 3, 4}}"
    );
  }
}

mod histogram_list {
  use super::*;

  #[test]
  fn auto_binning_basic() {
    assert_eq!(
      interpret("HistogramList[{50, 98, 94, 45, 31, 44, 28, 69}]").unwrap(),
      "{{0, 50, 100}, {4, 4}}"
    );
  }

  #[test]
  fn auto_binning_range_10() {
    assert_eq!(
      interpret("HistogramList[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]").unwrap(),
      "{{0, 5, 10, 15}, {4, 5, 1}}"
    );
  }

  #[test]
  fn auto_binning_range_20() {
    assert_eq!(
      interpret("HistogramList[Range[20]]").unwrap(),
      "{{0, 10, 20, 30}, {9, 10, 1}}"
    );
  }

  #[test]
  fn auto_binning_range_100() {
    assert_eq!(
      interpret("HistogramList[Range[100]]").unwrap(),
      "{{0, 20, 40, 60, 80, 100, 120}, {19, 20, 20, 20, 20, 1}}"
    );
  }

  #[test]
  fn auto_binning_floats() {
    assert_eq!(
      interpret("HistogramList[{1.5, 2.3, 3.7, 4.1}]").unwrap(),
      "{{0, 2, 4, 6}, {1, 2, 1}}"
    );
  }

  #[test]
  fn auto_binning_negative_values() {
    assert_eq!(
      interpret("HistogramList[{-5, -3, 0, 1, 7, 12}]").unwrap(),
      "{{-10, 0, 10, 20}, {2, 3, 1}}"
    );
  }

  #[test]
  fn auto_binning_all_same() {
    assert_eq!(
      interpret("HistogramList[{1, 1, 1, 1}]").unwrap(),
      "{{1, 2}, {4}}"
    );
  }

  #[test]
  fn auto_binning_single_element() {
    assert_eq!(interpret("HistogramList[{5}]").unwrap(), "{{5, 10}, {1}}");
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("HistogramList[{}]").unwrap(), "{{}, {}}");
  }

  // When the spread of data are all integer multiples of the bin width,
  // wolframscript centers the bins on the values (edges offset by dx/2) so no
  // value lands on a boundary, and reports the edges as reals.
  #[test]
  fn auto_binning_centers_on_commensurate_data() {
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}]").unwrap(),
      "{{0.5, 1.5, 2.5, 3.5}, {1, 2, 3}}"
    );
    assert_eq!(
      interpret("HistogramList[{10, 20, 20, 30}]").unwrap(),
      "{{5., 15., 25., 35.}, {1, 2, 1}}"
    );
    assert_eq!(
      interpret("HistogramList[{2, 4, 4, 6}]").unwrap(),
      "{{1., 3., 5., 7.}, {1, 2, 1}}"
    );
  }

  // The same centering applies to an explicit `{dx}` bin-width spec.
  #[test]
  fn explicit_width_centers_on_commensurate_data() {
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}, {1}]").unwrap(),
      "{{0.5, 1.5, 2.5, 3.5}, {1, 2, 3}}"
    );
    assert_eq!(
      interpret("HistogramList[{2, 4, 4, 6}, {2}]").unwrap(),
      "{{1., 3., 5., 7.}, {1, 2, 1}}"
    );
  }

  // Non-commensurate data keeps edges at multiples of the bin width.
  #[test]
  fn no_centering_for_non_commensurate_data() {
    assert_eq!(
      interpret("HistogramList[{1, 2, 3, 4, 5}]").unwrap(),
      "{{0, 2, 4, 6}, {1, 2, 2}}"
    );
  }

  #[test]
  fn explicit_bin_width() {
    assert_eq!(
      interpret("HistogramList[{50, 98, 94, 45, 31, 44, 28, 69}, {20}]")
        .unwrap(),
      "{{20, 40, 60, 80, 100}, {2, 3, 1, 2}}"
    );
  }

  #[test]
  fn explicit_bin_spec() {
    assert_eq!(
      interpret(
        "HistogramList[{50, 98, 94, 45, 31, 44, 28, 69}, {0, 100, 25}]"
      )
      .unwrap(),
      "{{0, 25, 50, 75, 100}, {0, 4, 2, 2}}"
    );
  }

  #[test]
  fn explicit_bin_spec_partial_trailing() {
    // (max - min) is not a multiple of dx: only whole bins are produced and
    // the max boundary value is excluded.
    assert_eq!(
      interpret("HistogramList[Range[10], {0, 10, 3}]").unwrap(),
      "{{0, 3, 6, 9}, {2, 3, 3}}"
    );
    assert_eq!(
      interpret("HistogramList[Range[10], {0, 10, 4}]").unwrap(),
      "{{0, 4, 8}, {3, 4}}"
    );
    // Even division: the value at max is excluded (last bin counts {8, 9}).
    assert_eq!(
      interpret("HistogramList[Range[10], {0, 10, 2}]").unwrap(),
      "{{0, 2, 4, 6, 8, 10}, {1, 2, 2, 2, 2}}"
    );
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    assert_eq!(interpret("HistogramList[x]").unwrap(), "HistogramList[x]");
  }

  // HistogramList[data, n]: wolframscript's userBinningN — the width is
  // (max-min)/(n-1) floored by the smallest data gap and snapped to the
  // linearly nearest of {1,2,5,10}*10^k (ties to the smaller nice number).
  #[test]
  fn bin_count_spec_nice_widths() {
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}, 2]").unwrap(),
      "{{0, 2, 4}, {1, 5}}"
    );
    assert_eq!(
      interpret("HistogramList[{1.5, 2.3, 4.7, 8.1, 9.9}, 3]").unwrap(),
      "{{0, 5, 10}, {3, 2}}"
    );
    assert_eq!(
      interpret("HistogramList[{1.5, 2.3, 4.7, 8.1, 9.9}, 5]").unwrap(),
      "{{0, 2, 4, 6, 8, 10}, {1, 1, 1, 0, 2}}"
    );
    // Tie 1.5 between 1 and 2 goes to the smaller nice width.
    assert_eq!(
      interpret("HistogramList[{1, 2.5, 4}, 3]").unwrap(),
      "{{1, 2, 3, 4, 5}, {1, 1, 0, 1}}"
    );
    // Rational bin widths give exact rational edges.
    assert_eq!(
      interpret("HistogramList[{0.1, 0.15, 0.2, 0.9}, 4]").unwrap(),
      "{{0, 1/5, 2/5, 3/5, 4/5, 1}, {2, 1, 0, 0, 1}}"
    );
    // A maximum lying exactly on an edge gets pushed into an extra bin.
    assert_eq!(
      interpret("HistogramList[Range[100], 10]").unwrap(),
      "{{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110}, \
       {9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1}}"
    );
    assert_eq!(
      interpret("HistogramList[Range[10], 3]").unwrap(),
      "{{0, 5, 10, 15}, {4, 5, 1}}"
    );
  }

  // n = 1 keeps only the two end edges of the aligned cover (which is why
  // the single bin can have a non-nice width like 4).
  #[test]
  fn bin_count_spec_single_bin() {
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}, 1]").unwrap(),
      "{{0, 4}, {6}}"
    );
    assert_eq!(
      interpret("HistogramList[{1.5, 2.3, 4.7, 8.1, 9.9}, 1]").unwrap(),
      "{{0, 10}, {5}}"
    );
  }

  // The smallest gap between sorted values floors the width, so a huge n
  // cannot make bins finer than the data resolution.
  #[test]
  fn bin_count_spec_granularity_floor() {
    assert_eq!(
      interpret("HistogramList[{1.5, 2.3, 4.7, 8.1, 9.9}, 100]").unwrap(),
      "{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1, 1, 0, 1, 0, 0, 0, 1, 1}}"
    );
    assert_eq!(
      interpret("HistogramList[{0.3, 9.7}, 4]").unwrap(),
      "{{0, 10}, {2}}"
    );
    assert_eq!(
      interpret("HistogramList[{1.0, 1.3, 5.0}, 50]").unwrap(),
      "{{1, 6/5, 7/5, 8/5, 9/5, 2, 11/5, 12/5, 13/5, 14/5, 3, 16/5, 17/5, \
       18/5, 19/5, 4, 21/5, 22/5, 23/5, 24/5, 5, 26/5}, \
       {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}}"
    );
  }

  // Centering: when the smallest data gap equals the chosen width (with a
  // 1/2/5 mantissa) and the FIRST value is a multiple of it, edges shift by
  // half a bin and become reals. wolframscript's result really does depend
  // on the data order, and a value on the shifted last edge is dropped.
  #[test]
  fn bin_count_spec_centering() {
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}, 3]").unwrap(),
      "{{0.5, 1.5, 2.5, 3.5}, {1, 2, 3}}"
    );
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}, 4]").unwrap(),
      "{{0.5, 1.5, 2.5, 3.5}, {1, 2, 3}}"
    );
    assert_eq!(
      interpret("HistogramList[{2., 3., 4.5}, 3]").unwrap(),
      "{{1.5, 2.5, 3.5, 4.5}, {1, 1, 0}}"
    );
    assert_eq!(
      interpret("HistogramList[{4.5, 3., 2.}, 3]").unwrap(),
      "{{2, 3, 4, 5}, {1, 1, 1}}"
    );
    // Negative on-multiple minima gain an extra bin below and lose the
    // maximum off the shifted top edge — faithful to wolframscript.
    assert_eq!(
      interpret("HistogramList[{-3, -2, -1}, 3]").unwrap(),
      "{{-4.5, -3.5, -2.5, -1.5}, {0, 1, 1}}"
    );
  }

  // All-identical data: exactly n machine-real bins over mean ± 1/2; a
  // single data point gets one exact half-integer bin instead.
  #[test]
  fn bin_count_spec_degenerate_data() {
    assert_eq!(
      interpret("HistogramList[{5, 5, 5}, 3]").unwrap(),
      "{{4.5, 4.833333333333333, 5.166666666666667, 5.5}, {0, 3, 0}}"
    );
    assert_eq!(
      interpret("HistogramList[{5, 5, 5}, 1]").unwrap(),
      "{{4.5, 5.5}, {3}}"
    );
    assert_eq!(
      interpret("HistogramList[{0, 0, 0}, 2]").unwrap(),
      "{{-0.5, 0., 0.5}, {0, 3}}"
    );
    assert_eq!(
      interpret("HistogramList[{7}, 3]").unwrap(),
      "{{13/2, 15/2}, {1}}"
    );
    assert_eq!(
      interpret("HistogramList[{7.5}, 3]").unwrap(),
      "{{7, 8}, {1}}"
    );
  }

  // Exact rational data is machine-numericized like wolframscript does
  // (previously it was silently dropped as non-numeric).
  #[test]
  fn rational_data() {
    assert_eq!(
      interpret("HistogramList[{1/2, 3/2, 5/2, 7/2}]").unwrap(),
      "{{0, 2, 4}, {2, 2}}"
    );
    assert_eq!(
      interpret("HistogramList[{1/2, 3/2, 5/2, 7/2}, 3]").unwrap(),
      "{{0, 1, 2, 3, 4}, {1, 1, 1, 1}}"
    );
    assert_eq!(
      interpret("BinCounts[{1/2, 3/2, 5/2}, {0, 3, 1}]").unwrap(),
      "{1, 1, 1}"
    );
    assert_eq!(
      interpret("BinLists[{1/2, 3/2, 5/2}, {0, 3, 1}]").unwrap(),
      "{{1/2}, {3/2}, {5/2}}"
    );
  }

  // Invalid bare bin specs emit ::hbins; a positive Real falls back to
  // automatic binning while zero/negative specs stay unevaluated.
  #[test]
  fn bin_count_spec_invalid() {
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}, 2.]").unwrap(),
      "{{0.5, 1.5, 2.5, 3.5}, {1, 2, 3}}"
    );
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}, 0]").unwrap(),
      "HistogramList[{1, 2, 2, 3, 3, 3}, 0]"
    );
    assert_eq!(
      interpret("HistogramList[{1, 2, 2, 3, 3, 3}, -2]").unwrap(),
      "HistogramList[{1, 2, 2, 3, 3, 3}, -2]"
    );
  }
}

mod total {
  use super::*;

  #[test]
  fn total_integers() {
    assert_eq!(interpret("Total[{1, 2, 3}]").unwrap(), "6");
    assert_eq!(interpret("Total[{1, 2, 3, 4, 5}]").unwrap(), "15");
  }

  #[test]
  fn total_empty_list() {
    assert_eq!(interpret("Total[{}]").unwrap(), "0");
  }

  #[test]
  fn total_rationals() {
    assert_eq!(interpret("Total[{1/2, 1/3, 1/6}]").unwrap(), "1");
  }

  #[test]
  fn total_reals() {
    assert_eq!(interpret("Total[{1, 2.5, 3}]").unwrap(), "6.5");
  }

  #[test]
  fn total_symbolic() {
    assert_eq!(interpret("Total[{x, y, z}]").unwrap(), "x + y + z");
  }

  #[test]
  fn total_nested_level_1() {
    // Total[{{1,2},{3,4}}] sums top-level elements = {4,6}
    assert_eq!(interpret("Total[{{1, 2}, {3, 4}}]").unwrap(), "{4, 6}");
  }

  #[test]
  fn total_level_2() {
    // Total[list, 2] sums across levels 1 and 2 = scalar
    assert_eq!(interpret("Total[{{1, 2}, {3, 4}}, 2]").unwrap(), "10");
  }

  // Total of a SparseArray sums its dense form.
  #[test]
  fn total_sparse_array() {
    assert_eq!(
      interpret("Total[SparseArray[{1 -> 2, 3 -> 4}, 5]]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("Total[SparseArray[{{1, 1} -> 1, {2, 2} -> 2}, {2, 2}]]")
        .unwrap(),
      "{1, 2}"
    );
    // With a level spec, the whole matrix is summed.
    assert_eq!(
      interpret("Total[SparseArray[{{1, 1} -> 1, {2, 2} -> 2}, {2, 2}], 2]")
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn total_level_spec_exact_2() {
    // Total[list, {2}] sums at exactly level 2 = row sums
    assert_eq!(interpret("Total[{{1, 2}, {3, 4}}, {2}]").unwrap(), "{3, 7}");
  }

  #[test]
  fn total_3d_level_1() {
    assert_eq!(
      interpret("Total[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}]").unwrap(),
      "{{6, 8}, {10, 12}}"
    );
  }

  #[test]
  fn total_3d_level_2() {
    assert_eq!(
      interpret("Total[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, 2]").unwrap(),
      "{16, 20}"
    );
  }

  #[test]
  fn total_3d_level_3() {
    assert_eq!(
      interpret("Total[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, 3]").unwrap(),
      "36"
    );
  }

  #[test]
  fn total_level_range() {
    // Total[list, {n1, n2}] collapses levels n1..=n2 together.
    assert_eq!(
      interpret("Total[{{1, 2, 3}, {4, 5, 6}}, {1, 2}]").unwrap(),
      "21"
    );
    assert_eq!(
      interpret("Total[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {1, 2}]").unwrap(),
      "{16, 20}"
    );
    assert_eq!(
      interpret("Total[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {2, 3}]").unwrap(),
      "{10, 26}"
    );
    assert_eq!(
      interpret("Total[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {1, 3}]").unwrap(),
      "36"
    );
  }

  // Negative levels count from the deepest level: -1 is the last level.
  #[test]
  fn total_negative_through_level() {
    // Total[list, -k] sums levels 1 through (depth - k + 1).
    assert_eq!(interpret("Total[{1, 2, 3}, -1]").unwrap(), "6");
    assert_eq!(interpret("Total[{{1, 2}, {3, 4}}, -1]").unwrap(), "10");
    assert_eq!(
      interpret("Total[{{{1}, {2}}, {{3}, {4}}}, -1]").unwrap(),
      "10"
    );
  }

  #[test]
  fn total_negative_exact_level() {
    // Total[list, {-1}] sums the innermost lists; {-2} the level above.
    assert_eq!(interpret("Total[{1, 2, 3}, {-1}]").unwrap(), "6");
    assert_eq!(
      interpret("Total[{{1, 2}, {3, 4}}, {-1}]").unwrap(),
      "{3, 7}"
    );
    assert_eq!(
      interpret("Total[{{1, 2}, {3, 4}}, {-2}]").unwrap(),
      "{4, 6}"
    );
  }

  // Summing rows of unequal length emits Total::tllen and stays unevaluated
  // (rather than leaking an internal error).
  #[test]
  fn total_unequal_rows_emit_tllen() {
    clear_state();
    assert_eq!(
      interpret("Total[{{1, 2}, {3, 4, 5}}]").unwrap(),
      "Total[{{1, 2}, {3, 4, 5}}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Total::tllen: Lists of unequal length in {{1, 2}, {3, 4, 5}} cannot be added."
      )),
      "expected Total::tllen, got {msgs:?}"
    );
  }

  #[test]
  fn total_equal_rows_emit_nothing() {
    clear_state();
    assert_eq!(interpret("Total[{{1, 2}, {3, 4}}]").unwrap(), "{4, 6}");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().all(|m| !m.contains("Total::tllen")),
      "unexpected tllen message: {msgs:?}"
    );
  }

  #[test]
  fn total_3d_exact_level_2() {
    assert_eq!(
      interpret("Total[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {2}]").unwrap(),
      "{{4, 6}, {12, 14}}"
    );
  }

  #[test]
  fn total_3d_exact_level_3() {
    assert_eq!(
      interpret("Total[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {3}]").unwrap(),
      "{{3, 7}, {11, 15}}"
    );
  }

  #[test]
  fn total_infinity() {
    assert_eq!(
      interpret("Total[{{1, 2}, {3, 4}}, Infinity]").unwrap(),
      "10"
    );
  }

  #[test]
  fn total_level_0() {
    // Total[list, 0] returns the list unchanged
    assert_eq!(interpret("Total[{1, 2, 3}, 0]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn total_exact_level_0_is_identity() {
    // Total[list, {0}] addresses level 0 — the whole expression — so it is
    // left unchanged, unlike {1} which sums the outermost list.
    assert_eq!(interpret("Total[{1, 2, 3}, {0}]").unwrap(), "{1, 2, 3}");
    assert_eq!(
      interpret("Total[{{1, 2}, {3, 4}}, {0}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
    assert_eq!(interpret("Total[5, {0}]").unwrap(), "5");
  }

  // An empty top-level list totals to 0 at every positive level, but a pure
  // level-0 spec leaves it untouched as {}.
  #[test]
  fn total_empty_list_exact_level() {
    assert_eq!(interpret("Total[{}, {1}]").unwrap(), "0");
    assert_eq!(interpret("Total[{}, {2}]").unwrap(), "0");
    assert_eq!(interpret("Total[{}, {3}]").unwrap(), "0");
    assert_eq!(interpret("Total[{}, {1, 2}]").unwrap(), "0");
    assert_eq!(interpret("Total[{}, Infinity]").unwrap(), "0");
  }

  #[test]
  fn total_empty_list_level_0_untouched() {
    assert_eq!(interpret("Total[{}, 0]").unwrap(), "{}");
    assert_eq!(interpret("Total[{}, {0}]").unwrap(), "{}");
  }

  // Nested empty lists keep their structure: only the addressed level is
  // collapsed, and there is nothing to collapse below an empty list.
  #[test]
  fn total_nested_empty_lists() {
    assert_eq!(interpret("Total[{{}}, {1}]").unwrap(), "{}");
    assert_eq!(interpret("Total[{{}}, {2}]").unwrap(), "{0}");
    assert_eq!(interpret("Total[{{}}, {3}]").unwrap(), "{{}}");
    assert_eq!(interpret("Total[{{{}}}, {3}]").unwrap(), "{{0}}");
    assert_eq!(interpret("Total[{{}, {1, 2}}, {2}]").unwrap(), "{0, 3}");
  }

  #[test]
  fn total_non_list() {
    assert_eq!(interpret("Total[5]").unwrap(), "5");
  }

  // A numeric scalar is its own total and is returned as-is.
  #[test]
  fn total_numeric_scalar_returns_itself() {
    assert_eq!(interpret("Total[Pi]").unwrap(), "Pi");
    assert_eq!(interpret("Total[Sin[1]]").unwrap(), "Sin[1]");
    assert_eq!(interpret("Total[2 Pi]").unwrap(), "2*Pi");
    assert_eq!(interpret("Total[1 + I]").unwrap(), "1 + I");
    // The numeric short-circuit applies even with a level spec.
    assert_eq!(interpret("Total[Pi, 2]").unwrap(), "Pi");
  }

  // A non-numeric, non-list argument stays unevaluated (no message).
  #[test]
  fn total_non_numeric_stays_unevaluated() {
    assert_eq!(interpret("Total[x]").unwrap(), "Total[x]");
    assert_eq!(interpret("Total[x + y]").unwrap(), "Total[x + y]");
    assert_eq!(interpret("Total[Sin[x]]").unwrap(), "Total[Sin[x]]");
    assert_eq!(interpret("Total[f[1, 2, 3]]").unwrap(), "Total[f[1, 2, 3]]");
    assert_eq!(interpret("Total[Infinity]").unwrap(), "Total[Infinity]");
    assert_eq!(interpret("Total[x, 2]").unwrap(), "Total[x, 2]");
  }

  // A string can never be a list: Total emits ::normal and stays unevaluated.
  #[test]
  fn total_string_emits_normal() {
    clear_state();
    assert_eq!(interpret(r#"Total["str"]"#).unwrap(), "Total[str]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Total::normal: Nonatomic expression expected at position 1 in Total[str]."
      )),
      "expected Total::normal, got {msgs:?}"
    );
  }

  // A wrong argument count leaves Total unevaluated (with a message) rather
  // than raising an evaluation error.
  #[test]
  fn wrong_argument_count_stays_unevaluated() {
    use woxi::interpret_with_stdout;
    let r = interpret_with_stdout("Total[]").unwrap();
    assert_eq!(r.result, "Total[]");
    assert!(r.warnings.iter().any(|m| m.contains(
      "Total::argt: Total called with 0 arguments; 1 or 2 arguments are expected."
    )));
    // Three or more arguments (an unexpected option) also stay unevaluated.
    assert_eq!(
      interpret("Total[{1, 2, 3}, 2, 3]").unwrap(),
      "Total[{1, 2, 3}, 2, 3]"
    );
  }
}

mod normalize {
  use super::*;

  #[test]
  fn normalize_integers_perfect() {
    assert_eq!(interpret("Normalize[{3, 4}]").unwrap(), "{3/5, 4/5}");
  }

  #[test]
  fn normalize_integers_irrational() {
    assert_eq!(
      interpret("Normalize[{1, 1, 1}]").unwrap(),
      "{1/Sqrt[3], 1/Sqrt[3], 1/Sqrt[3]}"
    );
  }

  #[test]
  fn normalize_zero_vector() {
    assert_eq!(interpret("Normalize[{0, 0, 0}]").unwrap(), "{0, 0, 0}");
  }

  #[test]
  fn normalize_three_elements() {
    assert_eq!(
      interpret("Normalize[{1, 2, 2}]").unwrap(),
      "{1/3, 2/3, 2/3}"
    );
  }

  #[test]
  fn normalize_symbolic() {
    assert_eq!(
      interpret("Normalize[{a, b}]").unwrap(),
      "{a/Sqrt[Abs[a]^2 + Abs[b]^2], b/Sqrt[Abs[a]^2 + Abs[b]^2]}"
    );
  }

  #[test]
  fn normalize_symbolic_evaluates_numerically() {
    assert_eq!(
      interpret("(Normalize[{a, b}] /. a -> 3) /. b -> 4").unwrap(),
      "{3/5, 4/5}"
    );
  }
}

mod rectt_messages {
  use super::*;

  // Mean/Median/Variance/StandardDeviation applied to a numeric scalar emit
  // <F>::rectt and stay unevaluated; non-numeric atoms/expressions stay
  // unevaluated with no message.
  fn assert_rectt(input: &str, result: &str, fragment: &str) {
    clear_state();
    assert_eq!(interpret(input).unwrap(), result);
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(fragment)),
      "expected {fragment:?}, got {msgs:?}"
    );
  }

  #[test]
  fn mean_numeric_scalar() {
    assert_rectt(
      "Mean[5]",
      "Mean[5]",
      "Mean::rectt: Rectangular array expected at position 1 in Mean[5].",
    );
    assert_rectt(
      "Mean[Pi]",
      "Mean[Pi]",
      "Mean::rectt: Rectangular array expected at position 1 in Mean[Pi].",
    );
  }

  #[test]
  fn median_numeric_scalar() {
    assert_rectt(
      "Median[5]",
      "Median[5]",
      "Median::rectt: Rectangular array expected at position 1 in Median[5].",
    );
  }

  #[test]
  fn variance_numeric_scalar() {
    assert_rectt(
      "Variance[5]",
      "Variance[5]",
      "Variance::rectt: Rectangular array expected at position 1 in Variance[5].",
    );
  }

  #[test]
  fn standard_deviation_numeric_scalar() {
    // Must report StandardDeviation::rectt, not Variance::rectt from delegation.
    assert_rectt(
      "StandardDeviation[5]",
      "StandardDeviation[5]",
      "StandardDeviation::rectt: Rectangular array expected at position 1 in StandardDeviation[5].",
    );
  }

  #[test]
  fn non_numeric_arguments_emit_nothing() {
    for input in [
      "Mean[x]",
      r#"Mean["str"]"#,
      "Mean[True]",
      "Mean[Infinity]",
      "Mean[f[1, 2, 3]]",
      "Mean[a + b]",
      "Median[x]",
      "StandardDeviation[x]",
    ] {
      clear_state();
      let _ = interpret(input).unwrap();
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().all(|m| !m.contains("::rectt")),
        "unexpected rectt for {input}: {msgs:?}"
      );
    }
  }

  // A ragged (unequal-length rows) or mixed scalar/list array is not
  // rectangular: emit ::rectt and stay unevaluated rather than returning a
  // bogus column-wise result.
  #[test]
  fn mean_ragged_emits_rectt() {
    assert_rectt(
      "Mean[{{1, 2}, {3}}]",
      "Mean[{{1, 2}, {3}}]",
      "Mean::rectt: Rectangular array expected at position 1 in Mean[{{1, 2}, {3}}].",
    );
    assert_rectt(
      "Mean[{1, {2, 3}}]",
      "Mean[{1, {2, 3}}]",
      "Mean::rectt: Rectangular array expected at position 1 in Mean[{1, {2, 3}}].",
    );
  }

  #[test]
  fn variance_std_ragged_emit_rectt() {
    assert_rectt(
      "Variance[{{1, 2}, {3}}]",
      "Variance[{{1, 2}, {3}}]",
      "Variance::rectt: Rectangular array expected at position 1 in Variance[{{1, 2}, {3}}].",
    );
    assert_rectt(
      "StandardDeviation[{{1, 2}, {3}}]",
      "StandardDeviation[{{1, 2}, {3}}]",
      "StandardDeviation::rectt: Rectangular array expected at position 1 in StandardDeviation[{{1, 2}, {3}}].",
    );
  }

  #[test]
  fn rectangular_arrays_unaffected() {
    clear_state();
    assert_eq!(interpret("Mean[{{1, 2}, {3, 4}}]").unwrap(), "{2, 3}");
    assert_eq!(
      interpret("Mean[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "{5/2, 7/2, 9/2}"
    );
    assert_eq!(interpret("Mean[{1, 2, 3}]").unwrap(), "2");
    assert_eq!(interpret("Variance[{1, 2, 3, 4}]").unwrap(), "5/3");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().all(|m| !m.contains("::rectt")),
      "unexpected rectt: {msgs:?}"
    );
  }

  // Median is stricter than Mean: it requires a rectangular array of real
  // numbers, so a ragged/mixed array OR symbolic/complex entries emit the
  // (differently-tagged) Median::rectn message. A numeric scalar still emits
  // Median::rectt.
  fn assert_rectn(input: &str, call: &str) {
    clear_state();
    assert_eq!(interpret(input).unwrap(), call);
    let expected = format!(
      "Median::rectn: A rectangular array of real numbers is expected at position 1 in {call}."
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(&expected)),
      "expected {expected:?}, got {msgs:?}"
    );
  }

  #[test]
  fn median_ragged_and_mixed_emit_rectn() {
    assert_rectn("Median[{{1, 2}, {3}}]", "Median[{{1, 2}, {3}}]");
    assert_rectn("Median[{1, {2, 3}}]", "Median[{1, {2, 3}}]");
  }

  #[test]
  fn median_symbolic_and_complex_emit_rectn() {
    assert_rectn("Median[{a, b, c}]", "Median[{a, b, c}]");
    assert_rectn("Median[{1, 2, x}]", "Median[{1, 2, x}]");
    assert_rectn("Median[{1, 2, 3 + I}]", "Median[{1, 2, 3 + I}]");
    assert_rectn("Median[{{a, b}, {c, d}}]", "Median[{{a, b}, {c, d}}]");
  }

  #[test]
  fn median_real_arrays_and_quantities_unaffected() {
    clear_state();
    assert_eq!(interpret("Median[{3, 1, 2}]").unwrap(), "2");
    assert_eq!(interpret("Median[{1, 2, 3, 4}]").unwrap(), "5/2");
    assert_eq!(interpret("Median[{1.5, 2.5, 3.5}]").unwrap(), "2.5");
    assert_eq!(interpret("Median[{{1, 2}, {3, 4}}]").unwrap(), "{2, 3}");
    // A list of compatible quantities is valid (sorted by magnitude).
    assert_eq!(
      interpret(r#"Median[{Quantity[2, "Meters"], Quantity[4, "Meters"]}]"#)
        .unwrap(),
      "Quantity[3, Meters]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().all(|m| !m.contains("::rectn")),
      "unexpected rectn: {msgs:?}"
    );
  }
}

mod mean {
  use super::*;

  #[test]
  fn mean_symbolic() {
    assert_eq!(interpret("Mean[{a, b}]").unwrap(), "(a + b)/2");
  }

  // Numeric constants mixed with inexact entries evaluate numerically
  // (found by the differential fuzzer: these stayed unevaluated).
  #[test]
  fn mean_numeric_constants_with_reals() {
    assert_eq!(interpret("Mean[{Pi, 16.0}]").unwrap(), "9.570796326794897");
    assert_eq!(
      interpret("Mean[{-5.5, Pi}]").unwrap(),
      "-1.1792036732051034"
    );
    // ... while all-exact input stays exact
    assert_eq!(interpret("Mean[{Pi, 1}]").unwrap(), "(1 + Pi)/2");
    // Symbolic entries with an inexact one form the symbolic quotient
    assert_eq!(interpret("Mean[{x, 1.0}]").unwrap(), "(1. + x)/2");
    // The summation stays in list order (Plus would canonical-sort the
    // reals and drift the last float digit)
    assert_eq!(interpret("Mean[{23.1, 24.4, 21.8, 25.5}]").unwrap(), "23.7");
  }

  #[test]
  fn mean_integers() {
    assert_eq!(interpret("Mean[{1, 2, 3}]").unwrap(), "2");
  }

  #[test]
  fn mean_rationals() {
    assert_eq!(interpret("Mean[{1/2, 1/3, 1/6}]").unwrap(), "1/3");
  }

  #[test]
  fn mean_rationals_noninteger_sum() {
    // Regression: a rational sum used to be left as the unevaluated
    // quotient (3/8)/2 instead of folding to 3/16
    assert_eq!(interpret("Mean[{1/4, 1/8}]").unwrap(), "3/16");
    assert_eq!(interpret("Mean[{1/2, 3/2}]").unwrap(), "1");
  }

  #[test]
  fn mean_empty_list() {
    // Mean of empty list returns unevaluated, matching Wolfram Language
    assert_eq!(interpret("Mean[{}]").unwrap(), "Mean[{}]");
  }

  // Mean of quantities collapses the sum/count division into a single
  // Quantity (verified against wolframscript).
  #[test]
  fn mean_quantities() {
    assert_eq!(
      interpret(r#"Mean[{Quantity[2, "Meters"], Quantity[4, "Meters"]}]"#)
        .unwrap(),
      "Quantity[3, Meters]"
    );
    assert_eq!(
      interpret(r#"Mean[{Quantity[2, "Meters"], Quantity[5, "Meters"]}]"#)
        .unwrap(),
      "Quantity[7/2, Meters]"
    );
    assert_eq!(
      interpret(
        r#"Mean[{Quantity[1, "Meters"], Quantity[2, "Meters"], Quantity[3, "Meters"]}]"#
      )
      .unwrap(),
      "Quantity[2, Meters]"
    );
  }
}

mod median_edge_cases {
  use super::*;

  #[test]
  fn median_empty_list() {
    // Median of empty list returns unevaluated
    assert_eq!(interpret("Median[{}]").unwrap(), "Median[{}]");
  }

  // Median of quantities sharing a unit: middle element (odd count) or the
  // mean of the two middle ones (even count). Verified against wolframscript.
  #[test]
  fn median_quantities_odd() {
    assert_eq!(
      interpret(
        r#"Median[{Quantity[6, "Meters"], Quantity[2, "Meters"], Quantity[4, "Meters"]}]"#
      )
      .unwrap(),
      "Quantity[4, Meters]"
    );
  }

  #[test]
  fn median_quantities_even() {
    assert_eq!(
      interpret(r#"Median[{Quantity[5, "Meters"], Quantity[1, "Meters"]}]"#)
        .unwrap(),
      "Quantity[3, Meters]"
    );
  }
}

mod variance_extended {
  use super::*;

  #[test]
  fn variance_complex() {
    assert_eq!(interpret("Variance[{1 + 2I, 3 - 10I}]").unwrap(), "74");
  }

  #[test]
  fn variance_symbolic_equal() {
    assert_eq!(interpret("Variance[{a, a}]").unwrap(), "0");
  }

  #[test]
  fn variance_matrix() {
    assert_eq!(
      interpret("Variance[{{1, 3, 5}, {4, 10, 100}}]").unwrap(),
      "{9/2, 49/2, 9025/2}"
    );
  }

  #[test]
  fn variance_empty_list() {
    // Variance of empty or single-element list returns unevaluated
    assert_eq!(interpret("Variance[{}]").unwrap(), "Variance[{}]");
    assert_eq!(interpret("Variance[{1}]").unwrap(), "Variance[{1}]");
  }

  #[test]
  fn variance_single_element_emits_shlen() {
    clear_state();
    assert_eq!(interpret("Variance[{5}]").unwrap(), "Variance[{5}]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Variance::shlen: The argument {5} should have at least two elements."
      )),
      "expected Variance::shlen, got {msgs:?}"
    );
  }

  #[test]
  fn variance_empty_list_emits_nothing() {
    // An empty list returns unevaluated silently — no shlen message,
    // unlike a single-element list (matching wolframscript).
    clear_state();
    assert_eq!(interpret("Variance[{}]").unwrap(), "Variance[{}]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().all(|m| !m.contains("::shlen")),
      "unexpected shlen for empty list: {msgs:?}"
    );
  }
}

mod stddev_extended {
  use super::*;

  #[test]
  fn stddev_symbolic_equal() {
    assert_eq!(interpret("StandardDeviation[{a, a}]").unwrap(), "0");
  }

  #[test]
  fn stddev_matrix() {
    assert_eq!(
      interpret("StandardDeviation[{{1, 10}, {-1, 20}}]").unwrap(),
      "{Sqrt[2], 5*Sqrt[2]}"
    );
  }

  #[test]
  fn stddev_empty_list() {
    assert_eq!(
      interpret("StandardDeviation[{}]").unwrap(),
      "StandardDeviation[{}]"
    );
    assert_eq!(
      interpret("StandardDeviation[{1}]").unwrap(),
      "StandardDeviation[{1}]"
    );
  }

  #[test]
  fn stddev_single_element_emits_shlen() {
    clear_state();
    assert_eq!(
      interpret("StandardDeviation[{5}]").unwrap(),
      "StandardDeviation[{5}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "StandardDeviation::shlen: The argument {5} should have at least two elements."
      )),
      "expected StandardDeviation::shlen, got {msgs:?}"
    );
  }

  #[test]
  fn stddev_empty_list_emits_nothing() {
    // An empty list returns unevaluated silently — no shlen message,
    // unlike a single-element list (matching wolframscript).
    clear_state();
    assert_eq!(
      interpret("StandardDeviation[{}]").unwrap(),
      "StandardDeviation[{}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().all(|m| !m.contains("::shlen")),
      "unexpected shlen for empty list: {msgs:?}"
    );
  }
}

mod covariance {
  use super::*;

  #[test]
  fn covariance_reals() {
    assert_eq!(
      interpret("Covariance[{0.2, 0.3, 0.1}, {0.3, 0.3, -0.2}]").unwrap(),
      "0.025"
    );
  }

  #[test]
  fn covariance_integers() {
    assert_eq!(interpret("Covariance[{1, 2, 3}, {4, 5, 6}]").unwrap(), "1");
  }

  #[test]
  fn covariance_exact_rational() {
    // Should return exact rational, not float
    assert_eq!(
      interpret("Covariance[{1, 2, 3, 4}, {2, 4, 6, 8}]").unwrap(),
      "10/3"
    );
  }

  // Single-argument matrix form: covariance matrix of the columns.

  #[test]
  fn covariance_matrix_2x2() {
    assert_eq!(
      interpret("Covariance[{{1, 2}, {3, 4}, {5, 6}}]").unwrap(),
      "{{4, 4}, {4, 4}}"
    );
  }

  // A single flat numeric vector is one variable, so its covariance is its
  // variance (covariance of the variable with itself).
  #[test]
  fn covariance_single_vector() {
    assert_eq!(interpret("Covariance[{1, 2, 3}]").unwrap(), "1");
    assert_eq!(interpret("Covariance[{2, 4, 6}]").unwrap(), "4");
    assert_eq!(interpret("Covariance[{1, 2, 3, 4, 5}]").unwrap(), "5/2");
    // A length-1 list has too few observations and stays unevaluated.
    assert_eq!(interpret("Covariance[{5}]").unwrap(), "Covariance[{5}]");
  }

  #[test]
  fn covariance_matrix_3x3() {
    assert_eq!(
      interpret("Covariance[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {2, 1, 0}}]")
        .unwrap(),
      "{{7, 8, 9}, {8, 10, 12}, {9, 12, 15}}"
    );
  }

  #[test]
  fn covariance_matrix_rational_entries() {
    assert_eq!(
      interpret("Covariance[{{1, 2}, {3, 5}, {6, 4}, {8, 9}}]").unwrap(),
      "{{29/3, 23/3}, {23/3, 26/3}}"
    );
  }

  #[test]
  fn covariance_matrix_off_diagonal_rational() {
    assert_eq!(
      interpret("Covariance[{{1, 4}, {2, 8}, {3, 7}}]").unwrap(),
      "{{1, 3/2}, {3/2, 13/3}}"
    );
  }

  #[test]
  fn covariance_symbolic_vectors() {
    // Symbolic vectors now produce wolframscript's closed form. For two
    // elements it factors; for three or more it is the expanded sum (the
    // mean of the second vector drops out because the first vector's
    // deviations sum to zero).
    assert_eq!(
      interpret("Covariance[{a, b}, {x, y}]").unwrap(),
      "((a - b)*(Conjugate[x] - Conjugate[y]))/2"
    );
    assert_eq!(
      interpret("Covariance[{a, b, c}, {x, y, z}]").unwrap(),
      "((2*a - b - c)*Conjugate[x] + (-a + 2*b - c)*Conjugate[y] \
       + (-a - b + 2*c)*Conjugate[z])/6"
    );
    // One-argument vector form is the variance (covariance with itself).
    assert_eq!(
      interpret("Covariance[{a, b}]").unwrap(),
      "((a - b)*(Conjugate[a] - Conjugate[b]))/2"
    );
  }

  #[test]
  fn covariance_symbolic_matrix_unevaluated() {
    // The symbolic covariance *matrix* is still left unevaluated: its
    // lower-triangle entries multiply a plain difference by a Conjugate
    // difference, and Woxi's Times ordering of those factors diverges from
    // wolframscript's.
    assert_eq!(
      interpret("Covariance[{{a, b}, {c, d}}]").unwrap(),
      "Covariance[{{a, b}, {c, d}}]"
    );
  }

  // Two-matrix cross-covariance: the (i, j) entry is the covariance of column i
  // of the first matrix with column j of the second. Result is p×q.
  #[test]
  fn covariance_two_matrices() {
    assert_eq!(
      interpret(
        "Covariance[{{1, 2}, {3, 4}, {5, 6}}, {{1, 1}, {2, 2}, {3, 3}}]"
      )
      .unwrap(),
      "{{2, 2}, {2, 2}}"
    );
    // Asymmetric: 3 variables on the left, 2 on the right → 3×2.
    assert_eq!(
      interpret(
        "Covariance[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}, {{1, 0}, {0, 1}, {1, 1}}]"
      )
      .unwrap(),
      "{{0, 3/2}, {0, 3/2}, {1/6, 5/3}}"
    );
  }

  #[test]
  fn covariance_two_matrices_real() {
    assert_eq!(
      interpret(
        "Covariance[{{1., 2.}, {3., 4.}, {5., 7.}}, {{2., 1.}, {4., 3.}, {6., 8.}}]"
      )
      .unwrap(),
      "{{4., 7.}, {5., 9.}}"
    );
  }
}

mod correlation {
  use super::*;

  #[test]
  fn correlation_reals() {
    assert_eq!(
      interpret("Correlation[{10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5}, {8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68}]").unwrap(),
      "0.81642051634484"
    );
  }

  #[test]
  fn correlation_perfect() {
    assert_eq!(interpret("Correlation[{1, 2, 3}, {2, 4, 6}]").unwrap(), "1");
  }

  // Regression: all-integer inputs must return an exact symbolic result,
  // not a floating-point approximation.
  #[test]
  fn correlation_integers_symbolic() {
    assert_eq!(
      interpret(
        "Correlation[{1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5}, \
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3}]"
      )
      .unwrap(),
      "(23*Sqrt[3/7])/28"
    );
  }

  // Constant list -> zero variance. Wolfram emits Correlation::zerosd
  // and leaves the expression unevaluated.
  #[test]
  fn correlation_constant_list() {
    assert_eq!(
      interpret("Correlation[{1, 1, 1}, {2, 4, 6}]").unwrap(),
      "Correlation[{1, 1, 1}, {2, 4, 6}]"
    );
  }

  #[test]
  fn correlation_large_integer_lists() {
    // Regression: Correlation of large-integer lists overflowed/failed to
    // cancel. Two independent bugs were involved:
    //   1. Sqrt of a Rational whose numerator exceeded u64 truncated via
    //      `as u64` (gave garbage radicands).
    //   2. The large-integer (>2^53) branch of times_ast folded only
    //      Integer/BigInteger factors, leaving a Rational coefficient
    //      uncancelled (e.g. `(15*10^15 Sqrt[3/7])/10^16`).
    // Correlation is scale-invariant, so all of these equal the {1,2,4}/{1,2,3}
    // value `(3 Sqrt[3/7])/2`, matching wolframscript.
    for k in ["15", "20", "25", "40"] {
      let expr =
        format!("Correlation[{{10^{k}, 2*10^{k}, 3*10^{k}}}, {{1, 2, 4}}]");
      assert_eq!(
        interpret(&expr).unwrap(),
        "(3*Sqrt[3/7])/2",
        "failed at k={k}"
      );
    }
    assert_eq!(
      interpret("Correlation[{10^20, 2*10^20, 4*10^20}, {1, 2, 3}]").unwrap(),
      "(3*Sqrt[3/7])/2"
    );
  }

  #[test]
  fn correlation_symbolic_two_vectors_audit_case() {
    // Audit case: two symbolic 2-vectors.
    assert_eq!(
      interpret("Correlation[{a, b}, {x, y}]").unwrap(),
      "((a - b)*(Conjugate[x] - Conjugate[y]))/(Sqrt[(a - b)*(Conjugate[a] - Conjugate[b])]*Sqrt[(x - y)*(Conjugate[x] - Conjugate[y])])"
    );
  }

  #[test]
  fn correlation_symbolic_matrix_audit_case() {
    // Audit case: symbolic 2x2 against 2x1 matrix → 2x1 correlation matrix.
    assert_eq!(
      interpret("Correlation[{{a, b}, {c, d}}, {{x}, {y}}]").unwrap(),
      "{{((a - c)*(Conjugate[x] - Conjugate[y]))/(Sqrt[(a - c)*(Conjugate[a] - Conjugate[c])]*Sqrt[(x - y)*(Conjugate[x] - Conjugate[y])])}, {((b - d)*(Conjugate[x] - Conjugate[y]))/(Sqrt[(b - d)*(Conjugate[b] - Conjugate[d])]*Sqrt[(x - y)*(Conjugate[x] - Conjugate[y])])}}"
    );
  }

  // Single-argument matrix form: the p×p correlation matrix of the columns.
  #[test]
  fn correlation_single_matrix() {
    assert_eq!(
      interpret("Correlation[{{1, 2}, {3, 5}, {5, 4}}]").unwrap(),
      "{{1, Sqrt[3/7]}, {Sqrt[3/7], 1}}"
    );
    assert_eq!(
      interpret("Correlation[{{1, 5}, {2, 4}, {3, 3}}]").unwrap(),
      "{{1, -1}, {-1, 1}}"
    );
  }

  // A single flat vector is one variable, perfectly correlated with itself.
  #[test]
  fn correlation_single_vector() {
    assert_eq!(interpret("Correlation[{1, 2, 3}]").unwrap(), "1");
    assert_eq!(interpret("Correlation[{a, b, c}]").unwrap(), "1");
    // A length-1 list has too few observations and stays unevaluated.
    assert_eq!(interpret("Correlation[{5}]").unwrap(), "Correlation[{5}]");
  }
}

mod central_moment {
  use super::*;

  #[test]
  fn central_moment_fourth() {
    assert_eq!(
      interpret("CentralMoment[{1.1, 1.2, 1.4, 2.1, 2.4}, 4]").unwrap(),
      "0.10084511999999998"
    );
  }

  #[test]
  fn central_moment_second() {
    // Second central moment = variance * (n-1)/n
    assert_eq!(interpret("CentralMoment[{1, 2, 3, 4, 5}, 2]").unwrap(), "2");
  }
}

mod kurtosis {
  use super::*;

  #[test]
  fn kurtosis_reals() {
    assert_eq!(
      interpret("Kurtosis[{1.1, 1.2, 1.4, 2.1, 2.4}]").unwrap(),
      "1.4209750290831373"
    );
  }

  #[test]
  fn kurtosis_integers_exact() {
    // Kurtosis should return exact rational for integer input
    assert_eq!(interpret("Kurtosis[{1, 2, 3, 4, 5}]").unwrap(), "17/10");
  }
}

mod skewness {
  use super::*;

  #[test]
  fn skewness_reals() {
    assert_eq!(
      interpret("Skewness[{1.1, 1.2, 1.4, 2.1, 2.4}]").unwrap(),
      "0.4070412816074878"
    );
  }

  #[test]
  fn skewness_symmetric() {
    // Symmetric distribution has zero skewness
    assert_eq!(interpret("Skewness[{1, 2, 3, 4, 5}]").unwrap(), "0");
  }
}

mod central_moment_exact {
  use super::*;

  #[test]
  fn fourth_moment_integers() {
    // CentralMoment should return exact rational for integer input
    assert_eq!(
      interpret("CentralMoment[{1, 2, 3, 4, 5}, 4]").unwrap(),
      "34/5"
    );
  }
}

mod cdf {
  use super::*;

  #[test]
  fn normal_at_zero() {
    assert_eq!(interpret("CDF[NormalDistribution[], 0]").unwrap(), "1/2");
  }

  #[test]
  fn normal_symbolic() {
    assert_eq!(
      interpret("CDF[NormalDistribution[mu, sigma], x]").unwrap(),
      "Erfc[(mu - x)/(Sqrt[2]*sigma)]/2"
    );
  }

  #[test]
  fn normal_numeric() {
    let result = interpret("N[CDF[NormalDistribution[], 1]]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - 0.8413447460685429).abs() < 1e-10,
      "Expected ~0.8413, got {}",
      val
    );
  }

  #[test]
  fn student_t_symbolic() {
    // CDF[StudentTDistribution[v], x] = Piecewise[
    //   {{BetaRegularized[v/(v + x^2), v/2, 1/2]/2, x <= 0}},
    //   (1 + BetaRegularized[x^2/(v + x^2), 1/2, v/2])/2]
    assert_eq!(
      interpret("CDF[StudentTDistribution[v], x]").unwrap(),
      "Piecewise[{{BetaRegularized[v/(v + x^2), v/2, 1/2]/2, x <= 0}}, (1 + BetaRegularized[x^2/(v + x^2), 1/2, v/2])/2]"
    );
  }

  #[test]
  fn student_t_at_zero() {
    // x = 0 -> first branch (x <= 0 is True): BetaRegularized[1, v/2, 1/2]/2
    // = 1/2 by identity.
    assert_eq!(interpret("CDF[StudentTDistribution[v], 0]").unwrap(), "1/2");
  }

  #[test]
  fn hypergeometric_pdf_concrete() {
    // PDF[HypergeometricDistribution[20, 50, 100], k] =
    //   Binomial[50, k] * Binomial[50, 20 - k] / Binomial[100, 20] for
    //   0 <= k <= 20, else 0. Binomial[100, 20] = 535983370403809682970.
    // (Times canonicalisation puts Binomial[50, k] before
    //  Binomial[50, 20 - k]; mathematically identical to wolframscript's
    //  reverse order.)
    assert_eq!(
      interpret("PDF[HypergeometricDistribution[20, 50, 100], k]").unwrap(),
      "Piecewise[{{(Binomial[50, 20 - k]*Binomial[50, k])/535983370403809682970, 0 <= k <= 20}}, 0]"
    );
  }

  #[test]
  fn hypergeometric_pdf_at_zero() {
    // PDF at k = 0: Binomial[50, 0] * Binomial[50, 20] / Binomial[100, 20]
    // = 47129212243960 / 535983370403809682970, reduced to 148/1683150111.
    assert_eq!(
      interpret("PDF[HypergeometricDistribution[20, 50, 100], 0]").unwrap(),
      "148/1683150111"
    );
  }

  #[test]
  fn hypergeometric_pdf_outside_support() {
    // k outside [0, n] -> 0 (Piecewise default branch).
    assert_eq!(
      interpret("PDF[HypergeometricDistribution[20, 50, 100], -1]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("PDF[HypergeometricDistribution[20, 50, 100], 25]").unwrap(),
      "0"
    );
  }

  // Mean = n*ns/nt; Variance = n*ns*(1 - ns/nt)*(nt - n) / ((nt - 1)*nt).
  // (The symbolic-parameter variance is value-correct but reorders the two
  // central factors versus wolframscript, so only the numeric forms are
  // asserted here.)
  #[test]
  fn hypergeometric_mean_variance() {
    assert_eq!(
      interpret("Mean[HypergeometricDistribution[5, 10, 20]]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("Variance[HypergeometricDistribution[5, 10, 20]]").unwrap(),
      "75/76"
    );
    assert_eq!(
      interpret("Variance[HypergeometricDistribution[3, 7, 15]]").unwrap(),
      "16/25"
    );
    assert_eq!(
      interpret("Variance[HypergeometricDistribution[4, 8, 12]]").unwrap(),
      "64/99"
    );
    assert_eq!(
      interpret("StandardDeviation[HypergeometricDistribution[5, 10, 20]]")
        .unwrap(),
      "(5*Sqrt[3/19])/2"
    );
  }

  #[test]
  fn binormal_pdf_one_arg() {
    // PDF[BinormalDistribution[rho], {x, y}]
    //   = E^(-(x^2 - 2 rho x y + y^2) / (2 (1 - rho^2)))
    //   / (2 Pi Sqrt[1 - rho^2]).
    // For rho = 1/3 this collapses to (3 E^(9 (-x^2 + (2 x y)/3 - y^2)/16))
    // / (4 Sqrt[2] Pi).
    assert_eq!(
      interpret("PDF[BinormalDistribution[1/3], {x, y}]").unwrap(),
      "(3*E^((9*(-x^2 + (2*x*y)/3 - y^2))/16))/(4*Sqrt[2]*Pi)"
    );
  }

  #[test]
  fn binormal_pdf_symbolic() {
    // PDF[BinormalDistribution[rho], {x, y}] with symbolic rho.
    assert_eq!(
      interpret("PDF[BinormalDistribution[r], {x, y}]").unwrap(),
      "E^((-x^2 + 2*r*x*y - y^2)/(2*(1 - r^2)))/(2*Pi*Sqrt[1 - r^2])"
    );
  }

  #[test]
  fn exponential() {
    assert_eq!(
      interpret("CDF[ExponentialDistribution[1], x]").unwrap(),
      "Piecewise[{{1 - E^(-x), x >= 0}}, 0]"
    );
  }

  #[test]
  fn uniform_default() {
    assert_eq!(
      interpret("CDF[UniformDistribution[{0, 1}], x]").unwrap(),
      "Piecewise[{{x, 0 <= x <= 1}, {1, x > 1}}, 0]"
    );
  }

  #[test]
  fn bernoulli() {
    assert_eq!(
      interpret("CDF[BernoulliDistribution[p], k]").unwrap(),
      "Piecewise[{{0, k < 0}, {1 - p, Inequality[0, LessEqual, k, Less, 1]}}, 1]"
    );
  }

  #[test]
  fn normal_at_one_exact() {
    // CDF[NormalDistribution[], 1] = Erfc[-1/Sqrt[2]]/2
    let result = interpret("CDF[NormalDistribution[], 1]").unwrap();
    assert!(
      result.contains("Erfc"),
      "Expected Erfc expression, got: {}",
      result
    );
  }
}

mod moment {
  use super::*;

  #[test]
  fn first_moment_is_mean() {
    assert_eq!(interpret("Moment[{1,2,3,4,5}, 1]").unwrap(), "3");
  }

  #[test]
  fn second_moment() {
    // Sum[x^2]/5 = (1+4+9+16+25)/5 = 55/5 = 11
    assert_eq!(interpret("Moment[{1,2,3,4,5}, 2]").unwrap(), "11");
  }

  #[test]
  fn third_moment() {
    // Sum[x^3]/5 = (1+8+27+64+125)/5 = 225/5 = 45
    assert_eq!(interpret("Moment[{1,2,3,4,5}, 3]").unwrap(), "45");
  }

  #[test]
  fn zeroth_moment() {
    assert_eq!(interpret("Moment[{1,2,3}, 0]").unwrap(), "1");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("Moment[{a,b,c}, 1]").unwrap(), "(a + b + c)/3");
  }
}

// Raw and central moments of the chi-square distribution. Chi-square[nu] has
// raw moments E[x^n] = Product_{i=0}^{n-1} (nu + 2 i), which wolframscript
// prints in the factored form nu*(2 + nu)*(4 + nu)*.... The expected strings
// were verified against wolframscript.
mod chi_square_moments {
  use super::*;

  #[test]
  fn raw_moments_symbolic() {
    assert_eq!(
      interpret("Moment[ChiSquareDistribution[k], 3]").unwrap(),
      "k*(2 + k)*(4 + k)"
    );
    assert_eq!(
      interpret("Moment[ChiSquareDistribution[k], 4]").unwrap(),
      "k*(2 + k)*(4 + k)*(6 + k)"
    );
  }

  #[test]
  fn raw_moment_numeric() {
    // 6*8*10 = 480.
    assert_eq!(
      interpret("Moment[ChiSquareDistribution[6], 3]").unwrap(),
      "480"
    );
  }

  #[test]
  fn central_moment_third() {
    assert_eq!(
      interpret("CentralMoment[ChiSquareDistribution[k], 3]").unwrap(),
      "8*k"
    );
    assert_eq!(
      interpret("CentralMoment[ChiSquareDistribution[6], 3]").unwrap(),
      "48"
    );
  }

  #[test]
  fn kurtosis_reduces() {
    // Kurtosis = CentralMoment[4]/Variance^2 collapses to 3 + 12/nu.
    assert_eq!(
      interpret("Kurtosis[ChiSquareDistribution[k]]").unwrap(),
      "3 + 12/k"
    );
  }
}

mod factorial_moment {
  use super::*;

  #[test]
  fn second_order() {
    // Sum[x(x-1)]/5 = (0+2+6+12+20)/5 = 8
    assert_eq!(interpret("FactorialMoment[{1,2,3,4,5}, 2]").unwrap(), "8");
  }

  #[test]
  fn first_order_is_mean() {
    assert_eq!(interpret("FactorialMoment[{1,2,3,4,5}, 1]").unwrap(), "3");
  }

  #[test]
  fn third_order() {
    // Sum[x(x-1)(x-2)]/5 = (0+0+6+24+60)/5 = 18
    assert_eq!(interpret("FactorialMoment[{1,2,3,4,5}, 3]").unwrap(), "18");
  }

  #[test]
  fn zeroth_order() {
    assert_eq!(interpret("FactorialMoment[{1,2,3}, 0]").unwrap(), "1");
  }

  #[test]
  fn real_data() {
    assert_eq!(
      interpret("FactorialMoment[{2.5, 3.1, 4.7}, 2]").unwrap(),
      "9.216666666666667"
    );
  }

  #[test]
  fn negative_order() {
    // Mean[{1/((3+1)), 1/((7+1))}] = (1/4 + 1/8)/2 = 3/16
    assert_eq!(interpret("FactorialMoment[{3, 7}, -1]").unwrap(), "3/16");
  }

  #[test]
  fn multivariate() {
    // Mean of x*y(y-1): (1*2*1 + 3*4*3 + 5*6*5)/3 = 188/3
    assert_eq!(
      interpret("FactorialMoment[{{1, 2}, {3, 4}, {5, 6}}, {1, 2}]").unwrap(),
      "188/3"
    );
  }

  #[test]
  fn symbolic_order() {
    assert_eq!(
      interpret("FactorialMoment[{1, 2, 3}, r]").unwrap(),
      "(FactorialPower[1, r] + FactorialPower[2, r] + FactorialPower[3, r])/3"
    );
  }

  // Distribution factorial moments E[X(X-1)...(X-r+1)].
  #[test]
  fn poisson() {
    // The defining property of the Poisson distribution: E[X^(r)] = lambda^r.
    assert_eq!(
      interpret("FactorialMoment[PoissonDistribution[m], 2]").unwrap(),
      "m^2"
    );
    assert_eq!(
      interpret("FactorialMoment[PoissonDistribution[m], 1]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("FactorialMoment[PoissonDistribution[3], 2]").unwrap(),
      "9"
    );
    assert_eq!(
      interpret("FactorialMoment[PoissonDistribution[m], 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn bernoulli() {
    // X in {0,1}, so the falling factorial is 0 for r >= 2.
    assert_eq!(
      interpret("FactorialMoment[BernoulliDistribution[p], 1]").unwrap(),
      "p"
    );
    assert_eq!(
      interpret("FactorialMoment[BernoulliDistribution[p], 2]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("FactorialMoment[BernoulliDistribution[p], 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn geometric() {
    // r! (1/p - 1)^r.
    assert_eq!(
      interpret("FactorialMoment[GeometricDistribution[p], 1]").unwrap(),
      "-1 + p^(-1)"
    );
    assert_eq!(
      interpret("FactorialMoment[GeometricDistribution[p], 2]").unwrap(),
      "2*(-1 + p^(-1))^2"
    );
  }

  #[test]
  fn binomial() {
    // Falling factorial n(n-1)...(n-r+1) times p^r.
    assert_eq!(
      interpret("FactorialMoment[BinomialDistribution[n, p], 1]").unwrap(),
      "n*p"
    );
    assert_eq!(
      interpret("FactorialMoment[BinomialDistribution[n, p], 2]").unwrap(),
      "-((1 - n)*n*p^2)"
    );
    assert_eq!(
      interpret("FactorialMoment[BinomialDistribution[n, p], 3]").unwrap(),
      "(1 - n)*(2 - n)*n*p^3"
    );
  }

  // Continuous (and other) distributions resolve through the defining
  // expectation E[X(X-1)...(X-r+1)]. The expected strings match wolframscript.
  #[test]
  fn continuous_normal() {
    // First factorial moment is the mean.
    assert_eq!(
      interpret("FactorialMoment[NormalDistribution[0, 1], 1]").unwrap(),
      "0"
    );
    // r=2: E[x(x-1)] = E[x^2] - E[x] = (mu^2 + sigma^2) - mu.
    assert_eq!(
      interpret("FactorialMoment[NormalDistribution[mu, sigma], 2]").unwrap(),
      "-mu + mu^2 + sigma^2"
    );
    // r=3 over a standard normal: E[x(x-1)(x-2)] = -3.
    assert_eq!(
      interpret("FactorialMoment[NormalDistribution[0, 1], 3]").unwrap(),
      "-3"
    );
    // r=0 is always 1.
    assert_eq!(
      interpret("FactorialMoment[NormalDistribution[0, 1], 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn continuous_exponential_uniform_gamma() {
    // ExponentialDistribution[1]: E[x(x-1)] = 2 - 1 = 1.
    assert_eq!(
      interpret("FactorialMoment[ExponentialDistribution[1], 2]").unwrap(),
      "1"
    );
    // UniformDistribution[{0, 1}]: E[x(x-1)] = 1/3 - 1/2 = -1/6.
    assert_eq!(
      interpret("FactorialMoment[UniformDistribution[{0, 1}], 2]").unwrap(),
      "-1/6"
    );
    // GammaDistribution[a, b]: symbolic parameters resolve exactly.
    assert_eq!(
      interpret("FactorialMoment[GammaDistribution[a, b], 2]").unwrap(),
      "-(a*b) + a*(1 + a)*b^2"
    );
  }
}

mod beta_distribution {
  use super::*;

  #[test]
  fn inert_form() {
    assert_eq!(
      interpret("BetaDistribution[2, 3]").unwrap(),
      "BetaDistribution[2, 3]"
    );
  }

  #[test]
  fn pdf_symbolic() {
    let result = interpret("PDF[BetaDistribution[a, b], x]").unwrap();
    assert!(
      result.contains("Piecewise") && result.contains("Beta[a, b]"),
      "Expected Piecewise with Beta, got: {}",
      result
    );
  }

  #[test]
  fn pdf_numeric() {
    assert_eq!(
      interpret("PDF[BetaDistribution[2, 3], 0.5]").unwrap(),
      "1.5"
    );
  }

  #[test]
  fn cdf_symbolic() {
    let result = interpret("CDF[BetaDistribution[2, 3], x]").unwrap();
    assert!(
      result.contains("BetaRegularized"),
      "Expected BetaRegularized, got: {}",
      result
    );
  }

  #[test]
  fn mean_numeric() {
    assert_eq!(interpret("Mean[BetaDistribution[2, 3]]").unwrap(), "2/5");
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[BetaDistribution[a, b]]").unwrap(),
      "a/(a + b)"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[BetaDistribution[2, 3]]").unwrap(),
      "1/25"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[BetaDistribution[a, b]]").unwrap(),
      "(a*b)/((a + b)^2*(1 + a + b))"
    );
  }
}

mod student_t_distribution {
  use super::*;

  #[test]
  fn inert_form() {
    assert_eq!(
      interpret("StudentTDistribution[5]").unwrap(),
      "StudentTDistribution[5]"
    );
  }

  #[test]
  fn pdf_symbolic() {
    let result = interpret("PDF[StudentTDistribution[nu], x]").unwrap();
    assert!(
      result.contains("Beta[nu/2, 1/2]"),
      "Expected Beta in denominator, got: {}",
      result
    );
  }

  #[test]
  fn pdf_at_zero() {
    // PDF at x=0 simplifies: (1+0)^(-...) / denom
    let result = interpret("PDF[StudentTDistribution[5], 0]").unwrap();
    assert!(!result.is_empty(), "Expected non-empty result");
  }

  #[test]
  fn mean() {
    assert_eq!(interpret("Mean[StudentTDistribution[5]]").unwrap(), "0");
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[StudentTDistribution[nu]]").unwrap(),
      "Piecewise[{{0, nu > 1}}, Indeterminate]"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[StudentTDistribution[5]]").unwrap(),
      "5/3"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[StudentTDistribution[nu]]").unwrap(),
      "Piecewise[{{nu/(-2 + nu), nu > 2}}, Indeterminate]"
    );
  }

  // PDF / StandardDeviation of the 1-argument form now match wolframscript's
  // canonical (nu/(nu+x^2))^((1+nu)/2) and threaded-Sqrt Piecewise forms.
  #[test]
  fn pdf_and_sd_canonical_form() {
    assert_eq!(
      interpret("PDF[StudentTDistribution[v], x]").unwrap(),
      "(v/(v + x^2))^((1 + v)/2)/(Sqrt[v]*Beta[v/2, 1/2])"
    );
    assert_eq!(
      interpret("StandardDeviation[StudentTDistribution[v]]").unwrap(),
      "Piecewise[{{Sqrt[v/(-2 + v)], v > 2}}, Indeterminate]"
    );
  }

  // The 3-argument StudentTDistribution[m, s, v] is the location-scale form.
  #[test]
  fn three_argument_location_scale() {
    assert_eq!(
      interpret("Mean[StudentTDistribution[m, s, v]]").unwrap(),
      "Piecewise[{{m, v > 1}}, Indeterminate]"
    );
    assert_eq!(
      interpret("Variance[StudentTDistribution[m, s, v]]").unwrap(),
      "Piecewise[{{(s^2*v)/(-2 + v), v > 2}}, Indeterminate]"
    );
    assert_eq!(
      interpret("StandardDeviation[StudentTDistribution[m, s, v]]").unwrap(),
      "Piecewise[{{s*Sqrt[v/(-2 + v)], v > 2}}, Indeterminate]"
    );
    assert_eq!(
      interpret("PDF[StudentTDistribution[m, s, v], x]").unwrap(),
      "(v/(v + (-m + x)^2/s^2))^((1 + v)/2)/(s*Sqrt[v]*Beta[v/2, 1/2])"
    );
    assert_eq!(
      interpret("CDF[StudentTDistribution[m, s, v], x]").unwrap(),
      "Piecewise[{{BetaRegularized[(s^2*v)/(s^2*v + (-m + x)^2), v/2, 1/2]/2, x <= m}}, (1 + BetaRegularized[(-m + x)^2/(s^2*v + (-m + x)^2), 1/2, v/2])/2]"
    );
    // Numeric forms.
    assert_eq!(
      interpret("Mean[StudentTDistribution[3, 2, 5]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Variance[StudentTDistribution[3, 2, 5]]").unwrap(),
      "20/3"
    );
  }

  // StandardDeviation of a distribution with a Piecewise variance threads the
  // square root into each branch (Pareto here; also StudentT/FRatio).
  #[test]
  fn standard_deviation_threads_piecewise() {
    assert_eq!(
      interpret("StandardDeviation[ParetoDistribution[k, a]]").unwrap(),
      "Piecewise[{{(Sqrt[a/(-2 + a)]*k)/(-1 + a), a > 2}}, Indeterminate]"
    );
  }
}

mod lognormal_distribution {
  use super::*;

  #[test]
  fn inert_form() {
    assert_eq!(
      interpret("LogNormalDistribution[0, 1]").unwrap(),
      "LogNormalDistribution[0, 1]"
    );
  }

  #[test]
  fn pdf_symbolic() {
    let result = interpret("PDF[LogNormalDistribution[mu, sigma], x]").unwrap();
    assert!(
      result.contains("Piecewise") && result.contains("Log[x]"),
      "Expected Piecewise with Log[x], got: {}",
      result
    );
  }

  #[test]
  fn cdf_symbolic() {
    let result = interpret("CDF[LogNormalDistribution[mu, sigma], x]").unwrap();
    assert!(
      result.contains("Erfc"),
      "Expected Erfc expression, got: {}",
      result
    );
  }

  #[test]
  fn mean_numeric() {
    assert_eq!(
      interpret("Mean[LogNormalDistribution[0, 1]]").unwrap(),
      "Sqrt[E]"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[LogNormalDistribution[mu, sigma]]").unwrap(),
      "E^(mu + sigma^2/2)"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[LogNormalDistribution[0, 1]]").unwrap(),
      "(-1 + E)*E"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[LogNormalDistribution[mu, sigma]]").unwrap(),
      "E^(2*mu + sigma^2)*(-1 + E^sigma^2)"
    );
  }
}

mod chi_square_distribution {
  use super::*;

  #[test]
  fn inert_form() {
    assert_eq!(
      interpret("ChiSquareDistribution[5]").unwrap(),
      "ChiSquareDistribution[5]"
    );
  }

  #[test]
  fn pdf_symbolic() {
    // x^(-1 + k/2) is wolframscript's canonical exponent order (this
    // previously asserted the unevaluated x^(k/2 - 1) form that predated
    // Piecewise evaluating its piece values)
    assert_eq!(
      interpret("PDF[ChiSquareDistribution[k], x]").unwrap(),
      "Piecewise[{{x^(-1 + k/2)/(2^(k/2)*E^(x/2)*Gamma[k/2]), x > 0}}, 0]"
    );
  }

  #[test]
  fn cdf_symbolic() {
    assert_eq!(
      interpret("CDF[ChiSquareDistribution[k], x]").unwrap(),
      "Piecewise[{{GammaRegularized[k/2, 0, x/2], x > 0}}, 0]"
    );
  }

  #[test]
  fn mean() {
    assert_eq!(interpret("Mean[ChiSquareDistribution[k]]").unwrap(), "k");
  }

  #[test]
  fn mean_numeric() {
    assert_eq!(interpret("Mean[ChiSquareDistribution[5]]").unwrap(), "5");
  }

  #[test]
  fn variance() {
    assert_eq!(
      interpret("Variance[ChiSquareDistribution[k]]").unwrap(),
      "2*k"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[ChiSquareDistribution[5]]").unwrap(),
      "10"
    );
  }
}

mod pareto_distribution {
  use super::*;

  #[test]
  fn inert_form() {
    assert_eq!(
      interpret("ParetoDistribution[1, 2]").unwrap(),
      "ParetoDistribution[1, 2]"
    );
  }

  #[test]
  fn pdf_symbolic() {
    // a*k^a*x^(-1 - a) is wolframscript's canonical form (this previously
    // asserted the unevaluated quotient spelling that predated Piecewise
    // evaluating its piece values)
    assert_eq!(
      interpret("PDF[ParetoDistribution[k, a], x]").unwrap(),
      "Piecewise[{{a*k^a*x^(-1 - a), x >= k}}, 0]"
    );
  }

  #[test]
  fn cdf_symbolic() {
    assert_eq!(
      interpret("CDF[ParetoDistribution[k, a], x]").unwrap(),
      "Piecewise[{{1 - (k/x)^a, x >= k}}, 0]"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[ParetoDistribution[k, a]]").unwrap(),
      "Piecewise[{{(a*k)/(-1 + a), a > 1}}, Indeterminate]"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[ParetoDistribution[1, 3]]").unwrap(),
      "3/4"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[ParetoDistribution[k, a]]").unwrap(),
      "Piecewise[{{(a*k^2)/((-2 + a)*(-1 + a)^2), a > 2}}, Indeterminate]"
    );
  }
}

mod weibull_distribution {
  use super::*;

  #[test]
  fn constructor() {
    assert_eq!(
      interpret("WeibullDistribution[2, 3]").unwrap(),
      "WeibullDistribution[2, 3]"
    );
  }

  #[test]
  fn pdf_numeric() {
    assert_eq!(
      interpret("PDF[WeibullDistribution[2, 3], 1]").unwrap(),
      "2/(9*E^(1/9))"
    );
  }

  #[test]
  fn pdf_symbolic() {
    // (x/b)^(-1 + a) is wolframscript's canonical exponent order (this
    // previously asserted the unevaluated a - 1 form that predated
    // Piecewise evaluating its piece values)
    assert_eq!(
      interpret("PDF[WeibullDistribution[a, b], x]").unwrap(),
      "Piecewise[{{(a*(x/b)^(-1 + a))/(b*E^(x/b)^a), x > 0}}, 0]"
    );
  }

  #[test]
  fn pdf_numeric_parameters_fold() {
    // Regression: (x/5)^(2 - 1) used to be left unfolded; the piece now
    // evaluates to wolframscript's (2*x)/(25*E^(x^2/25))
    assert_eq!(
      interpret("PDF[WeibullDistribution[2, 5], x]").unwrap(),
      "Piecewise[{{(2*x)/(25*E^(x^2/25)), x > 0}}, 0]"
    );
  }

  #[test]
  fn survival_function_normalizes_sqrt() {
    // Piece evaluation also normalizes Sqrt[x/2] to Sqrt[x]/Sqrt[2],
    // matching wolframscript
    assert_eq!(
      interpret("SurvivalFunction[WeibullDistribution[1/2, 2], x]").unwrap(),
      "Piecewise[{{E^(-(Sqrt[x]/Sqrt[2])), x > 0}}, 1]"
    );
  }

  #[test]
  fn cdf_symbolic() {
    assert_eq!(
      interpret("CDF[WeibullDistribution[a, b], x]").unwrap(),
      "Piecewise[{{1 - E^(-(x/b)^a), x > 0}}, 0]"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[WeibullDistribution[a, b]]").unwrap(),
      "b*Gamma[1 + a^(-1)]"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[WeibullDistribution[2, 3]]").unwrap(),
      "9*(1 - Pi/4)"
    );
  }

  #[test]
  fn pdf_at_zero() {
    assert_eq!(interpret("PDF[WeibullDistribution[2, 3], 0]").unwrap(), "0");
  }

  // The 3-argument WeibullDistribution[a, b, m] is the location-shifted form:
  // Mean = m + b Gamma[1 + 1/a], and PDF/CDF use (x - m) with support x > m.
  #[test]
  fn three_argument_location_shifted() {
    assert_eq!(
      interpret("Mean[WeibullDistribution[a, b, m]]").unwrap(),
      "m + b*Gamma[1 + a^(-1)]"
    );
    assert_eq!(
      interpret("Mean[WeibullDistribution[2, 3, 1]]").unwrap(),
      "1 + (3*Sqrt[Pi])/2"
    );
    assert_eq!(
      interpret("PDF[WeibullDistribution[a, b, m], x]").unwrap(),
      "Piecewise[{{(a*((-m + x)/b)^(-1 + a))/(b*E^((-m + x)/b)^a), x > m}}, 0]"
    );
    assert_eq!(
      interpret("CDF[WeibullDistribution[a, b, m], x]").unwrap(),
      "Piecewise[{{1 - E^(-((-m + x)/b)^a), x > m}}, 0]"
    );
    // Variance is location-independent; numeric form matches.
    assert_eq!(
      interpret("Variance[WeibullDistribution[2, 3, 1]]").unwrap(),
      "9*(1 - Pi/4)"
    );
    assert_eq!(
      interpret("CDF[WeibullDistribution[2, 3, 1], 4]").unwrap(),
      "1 - E^(-1)"
    );
  }
}

mod multinomial_distribution {
  use super::*;

  #[test]
  fn displays_unevaluated() {
    assert_eq!(
      interpret("MultinomialDistribution[3, {1/3, 1/3, 1/3}]").unwrap(),
      "MultinomialDistribution[3, {1/3, 1/3, 1/3}]"
    );
  }

  #[test]
  fn pdf_numeric() {
    // PDF[MultinomialDistribution[3, {1/3, 1/3, 1/3}], {1, 1, 1}] = 2/9
    assert_eq!(
      interpret("PDF[MultinomialDistribution[3, {1/3, 1/3, 1/3}], {1, 1, 1}]")
        .unwrap(),
      "2/9"
    );
  }

  #[test]
  fn pdf_two_categories() {
    // PDF[MultinomialDistribution[10, {1/2, 1/2}], {5, 5}] = 63/256
    assert_eq!(
      interpret("PDF[MultinomialDistribution[10, {1/2, 1/2}], {5, 5}]")
        .unwrap(),
      "63/256"
    );
  }

  #[test]
  fn pdf_three_categories() {
    // PDF[MultinomialDistribution[5, {1/2, 1/3, 1/6}], {2, 2, 1}] = 5/36
    assert_eq!(
      interpret("PDF[MultinomialDistribution[5, {1/2, 1/3, 1/6}], {2, 2, 1}]")
        .unwrap(),
      "5/36"
    );
  }

  #[test]
  fn pdf_sum_not_equal_n() {
    // When sum of xi != n, PDF is 0
    assert_eq!(
      interpret("PDF[MultinomialDistribution[5, {1/2, 1/3, 1/6}], {2, 2, 2}]")
        .unwrap(),
      "0"
    );
  }

  #[test]
  fn pdf_negative_value() {
    assert_eq!(
      interpret("PDF[MultinomialDistribution[3, {1/3, 1/3, 1/3}], {-1, 2, 2}]")
        .unwrap(),
      "0"
    );
  }

  #[test]
  fn mean_exact() {
    assert_eq!(
      interpret("Mean[MultinomialDistribution[5, {1/2, 1/3, 1/6}]]").unwrap(),
      "{5/2, 5/3, 5/6}"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[MultinomialDistribution[n, {p1, p2, p3}]]").unwrap(),
      "{n*p1, n*p2, n*p3}"
    );
  }

  #[test]
  fn variance_exact() {
    assert_eq!(
      interpret("Variance[MultinomialDistribution[5, {1/2, 1/3, 1/6}]]")
        .unwrap(),
      "{5/4, 10/9, 25/36}"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[MultinomialDistribution[n, {p1, p2, p3}]]").unwrap(),
      "{n*(1 - p1)*p1, n*(1 - p2)*p2, n*(1 - p3)*p3}"
    );
  }

  #[test]
  fn standard_deviation_exact() {
    assert_eq!(
      interpret(
        "StandardDeviation[MultinomialDistribution[5, {1/2, 1/3, 1/6}]]"
      )
      .unwrap(),
      "{Sqrt[5]/2, Sqrt[10]/3, 5/6}"
    );
  }

  #[test]
  fn standard_deviation_symbolic() {
    assert_eq!(
      interpret("StandardDeviation[MultinomialDistribution[n, {p1, p2, p3}]]")
        .unwrap(),
      "{Sqrt[n*(1 - p1)*p1], Sqrt[n*(1 - p2)*p2], Sqrt[n*(1 - p3)*p3]}"
    );
  }
}

mod dirichlet_distribution {
  use super::*;

  #[test]
  fn displays_unevaluated() {
    assert_eq!(
      interpret("DirichletDistribution[{2, 3, 2}]").unwrap(),
      "DirichletDistribution[{2, 3, 2}]"
    );
  }

  #[test]
  fn mean_exact() {
    // The mean is a k-vector {αi/α0} — the last component is determined by
    // the others and dropped.
    assert_eq!(
      interpret("Mean[DirichletDistribution[{2, 3, 2}]]").unwrap(),
      "{2/7, 3/7}"
    );
    assert_eq!(
      interpret("Mean[DirichletDistribution[{2, 3, 4, 5}]]").unwrap(),
      "{1/7, 3/14, 2/7}"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[DirichletDistribution[{a, b, c}]]").unwrap(),
      "{a/(a + b + c), b/(a + b + c)}"
    );
  }

  #[test]
  fn variance_exact() {
    assert_eq!(
      interpret("Variance[DirichletDistribution[{2, 3, 2}]]").unwrap(),
      "{5/196, 3/98}"
    );
  }

  #[test]
  fn variance_symbolic() {
    // Variance_i = αi (α0 - αi) / (α0² (1 + α0)), the numerator spelled
    // αi·(sum of the other parameters) as Wolfram displays it.
    assert_eq!(
      interpret("Variance[DirichletDistribution[{a, b, c}]]").unwrap(),
      "{(a*(b + c))/((a + b + c)^2*(1 + a + b + c)), \
       (b*(a + c))/((a + b + c)^2*(1 + a + b + c))}"
    );
  }

  #[test]
  fn standard_deviation_exact() {
    assert_eq!(
      interpret("StandardDeviation[DirichletDistribution[{2, 3, 2}]]").unwrap(),
      "{Sqrt[5]/14, Sqrt[3/2]/7}"
    );
  }

  #[test]
  fn covariance_exact() {
    assert_eq!(
      interpret("Covariance[DirichletDistribution[{2, 3, 2}]]").unwrap(),
      "{{5/196, -3/196}, {-3/196, 3/98}}"
    );
  }

  #[test]
  fn covariance_symbolic() {
    // Diagonal (-αi² + αi α0) and off-diagonal -αi αj over α0² (1 + α0).
    assert_eq!(
      interpret("Covariance[DirichletDistribution[{a, b, c}]]").unwrap(),
      "{{(-a^2 + a*(a + b + c))/((a + b + c)^2*(1 + a + b + c)), \
       -((a*b)/((a + b + c)^2*(1 + a + b + c)))}, \
       {-((a*b)/((a + b + c)^2*(1 + a + b + c))), \
       (-b^2 + b*(a + b + c))/((a + b + c)^2*(1 + a + b + c))}}"
    );
  }

  #[test]
  fn pdf_numeric_params() {
    // Gamma[7]/(Gamma[2] Gamma[3] Gamma[2]) = 360 on the open simplex.
    assert_eq!(
      interpret("PDF[DirichletDistribution[{2, 3, 2}], {x, y}]").unwrap(),
      "Piecewise[{{360*x*(1 - x - y)*y^2, x > 0 && y > 0 && 1 - x - y > 0}}, 0]"
    );
  }

  #[test]
  fn pdf_symbolic_params() {
    assert_eq!(
      interpret("PDF[DirichletDistribution[{a, b, c}], {x, y}]").unwrap(),
      "Piecewise[{{(x^(-1 + a)*(1 - x - y)^(-1 + c)*y^(-1 + b)*\
       Gamma[a + b + c])/(Gamma[a]*Gamma[b]*Gamma[c]), \
       x > 0 && y > 0 && 1 - x - y > 0}}, 0]"
    );
  }

  #[test]
  fn pdf_unit_alphas_is_constant() {
    // All αi = 1 is the uniform density on the simplex: Gamma[3] = 2.
    assert_eq!(
      interpret("PDF[DirichletDistribution[{1, 1, 1}], {x, y}]").unwrap(),
      "Piecewise[{{2, x > 0 && y > 0 && 1 - x - y > 0}}, 0]"
    );
  }

  #[test]
  fn pdf_higher_dimension() {
    assert_eq!(
      interpret("PDF[DirichletDistribution[{2, 3, 4, 5}], {x, y, z}]").unwrap(),
      "Piecewise[{{21621600*x*y^2*(1 - x - y - z)^4*z^3, \
       x > 0 && y > 0 && z > 0 && 1 - x - y - z > 0}}, 0]"
    );
  }

  #[test]
  fn pdf_at_numeric_point() {
    // Inside the simplex the piecewise value applies; outside it is 0.
    assert_eq!(
      interpret("PDF[DirichletDistribution[{2, 3, 2}], {1/4, 1/2}]").unwrap(),
      "45/8"
    );
    assert_eq!(
      interpret("PDF[DirichletDistribution[{2, 3, 2}], {3/4, 1/2}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn pdf_wrong_point_shape_stays_unevaluated() {
    // A scalar or wrong-length point echoes back unevaluated, as in Wolfram.
    assert_eq!(
      interpret("PDF[DirichletDistribution[{2, 3, 2}], x]").unwrap(),
      "PDF[DirichletDistribution[{2, 3, 2}], x]"
    );
    assert_eq!(
      interpret("PDF[DirichletDistribution[{2, 3, 2}], {x}]").unwrap(),
      "PDF[DirichletDistribution[{2, 3, 2}], {x}]"
    );
  }
}

mod negative_binomial_distribution {
  use super::*;

  #[test]
  fn displays_unevaluated() {
    assert_eq!(
      interpret("NegativeBinomialDistribution[3, 1/2]").unwrap(),
      "NegativeBinomialDistribution[3, 1/2]"
    );
  }

  #[test]
  fn pdf_numeric() {
    assert_eq!(
      interpret("PDF[NegativeBinomialDistribution[3, 1/2], 5]").unwrap(),
      "21/256"
    );
  }

  #[test]
  fn pdf_symbolic() {
    assert_eq!(
      interpret("PDF[NegativeBinomialDistribution[n, p], k]").unwrap(),
      "Piecewise[{{(1 - p)^k*p^n*Binomial[-1 + k + n, -1 + n], k >= 0}}, 0]"
    );
  }

  #[test]
  fn mean_numeric() {
    assert_eq!(
      interpret("Mean[NegativeBinomialDistribution[3, 1/2]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[NegativeBinomialDistribution[n, p]]").unwrap(),
      "(n*(1 - p))/p"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[NegativeBinomialDistribution[3, 1/2]]").unwrap(),
      "6"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[NegativeBinomialDistribution[n, p]]").unwrap(),
      "(n*(1 - p))/p^2"
    );
  }

  #[test]
  fn standard_deviation_symbolic() {
    // Distribution parameters are assumed positive, so p^(-2) is extracted:
    // Sqrt[n*(1-p)*p^(-2)] → Sqrt[n*(1-p)] / p
    assert_eq!(
      interpret("StandardDeviation[NegativeBinomialDistribution[n, p]]")
        .unwrap(),
      "Sqrt[n*(1 - p)]/p"
    );
  }
}

mod pascal_distribution {
  use super::*;

  #[test]
  fn displays_unevaluated() {
    assert_eq!(
      interpret("PascalDistribution[3, 1/2]").unwrap(),
      "PascalDistribution[3, 1/2]"
    );
  }

  #[test]
  fn pdf_numeric() {
    assert_eq!(
      interpret("PDF[PascalDistribution[3, 1/2], 5]").unwrap(),
      "3/16"
    );
  }

  #[test]
  fn pdf_below_support() {
    assert_eq!(
      interpret("PDF[PascalDistribution[3, 1/2], 2]").unwrap(),
      "0"
    );
  }

  #[test]
  fn mean_numeric() {
    assert_eq!(interpret("Mean[PascalDistribution[3, 1/2]]").unwrap(), "6");
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[PascalDistribution[3, 1/2]]").unwrap(),
      "6"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(interpret("Mean[PascalDistribution[n, p]]").unwrap(), "n/p");
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[PascalDistribution[n, p]]").unwrap(),
      "(n*(1 - p))/p^2"
    );
  }
}

mod dagum_distribution {
  use super::*;

  #[test]
  fn displays_unevaluated() {
    assert_eq!(
      interpret("DagumDistribution[2, 3, 1]").unwrap(),
      "DagumDistribution[2, 3, 1]"
    );
  }

  #[test]
  fn pdf_numeric() {
    assert_eq!(
      interpret("PDF[DagumDistribution[2, 3, 1], 1/2]").unwrap(),
      "32/243"
    );
  }

  #[test]
  fn pdf_zero() {
    assert_eq!(
      interpret("PDF[DagumDistribution[2, 3, 1], 0]").unwrap(),
      "0"
    );
  }

  #[test]
  fn pdf_negative() {
    assert_eq!(
      interpret("PDF[DagumDistribution[2, 3, 1], -1]").unwrap(),
      "0"
    );
  }

  #[test]
  fn mean() {
    assert_eq!(
      interpret("Mean[DagumDistribution[2, 3, 1]]").unwrap(),
      "Gamma[2/3]*Gamma[7/3]"
    );
  }
}

mod hyperbolic_distribution {
  use super::*;

  #[test]
  fn displays_unevaluated() {
    assert_eq!(
      interpret("HyperbolicDistribution[3, 1, 2, 0]").unwrap(),
      "HyperbolicDistribution[3, 1, 2, 0]"
    );
  }

  #[test]
  fn pdf_symbolic() {
    assert_eq!(
      interpret("PDF[HyperbolicDistribution[a, b, d, m], x]").unwrap(),
      "(Sqrt[a^2 - b^2]*E^(b*(-m + x) - a*Sqrt[d^2 + (-m + x)^2]))/(2*a*d*BesselK[1, Sqrt[a^2 - b^2]*d])"
    );
  }

  #[test]
  fn pdf_numeric() {
    assert_eq!(
      interpret("PDF[HyperbolicDistribution[3, 1, 2, 0], 0]").unwrap(),
      "1/(3*Sqrt[2]*E^6*BesselK[1, 4*Sqrt[2]])"
    );
  }

  #[test]
  fn pdf_at_half() {
    assert_eq!(
      interpret("PDF[HyperbolicDistribution[3, 1, 2, 0], 1/2]").unwrap(),
      "E^(1/2 - (3*Sqrt[17])/2)/(3*Sqrt[2]*BesselK[1, 4*Sqrt[2]])"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[HyperbolicDistribution[a, b, d, m]]").unwrap(),
      "m + (b*d*BesselK[2, Sqrt[a^2 - b^2]*d])/(Sqrt[a^2 - b^2]*BesselK[1, Sqrt[a^2 - b^2]*d])"
    );
  }

  #[test]
  fn mean_numeric() {
    assert_eq!(
      interpret("Mean[HyperbolicDistribution[3, 1, 2, 0]]").unwrap(),
      "BesselK[2, 4*Sqrt[2]]/(Sqrt[2]*BesselK[1, 4*Sqrt[2]])"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[HyperbolicDistribution[a, b, d, m]]").unwrap(),
      "(d*BesselK[2, Sqrt[a^2 - b^2]*d])/(Sqrt[a^2 - b^2]*BesselK[1, Sqrt[a^2 - b^2]*d]) - (b^2*d^2*BesselK[2, Sqrt[a^2 - b^2]*d]^2)/((a^2 - b^2)*BesselK[1, Sqrt[a^2 - b^2]*d]^2) + (b^2*d^2*BesselK[3, Sqrt[a^2 - b^2]*d])/((a^2 - b^2)*BesselK[1, Sqrt[a^2 - b^2]*d])"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[HyperbolicDistribution[3, 1, 2, 0]]").unwrap(),
      "BesselK[2, 4*Sqrt[2]]/(Sqrt[2]*BesselK[1, 4*Sqrt[2]]) - BesselK[2, 4*Sqrt[2]]^2/(2*BesselK[1, 4*Sqrt[2]]^2) + BesselK[3, 4*Sqrt[2]]/(2*BesselK[1, 4*Sqrt[2]])"
    );
  }
}

mod stable_distribution {
  use super::*;

  #[test]
  fn canonical_form_2_params() {
    clear_state();
    // StableDistribution[alpha, beta] normalizes to 5-param form
    assert_eq!(
      interpret("StableDistribution[1, 0]").unwrap(),
      "StableDistribution[1, 1, 0, 0, 1]"
    );
  }

  #[test]
  fn canonical_form_4_params() {
    clear_state();
    // StableDistribution[alpha, beta, mu, sigma] normalizes to 5-param form
    assert_eq!(
      interpret("StableDistribution[1, 0, 0, 1]").unwrap(),
      "StableDistribution[1, 1, 0, 0, 1]"
    );
  }

  #[test]
  fn pdf_cauchy_standard() {
    clear_state();
    // StableDistribution[1, 0] is standard Cauchy: 1/(Pi*(1 + x^2))
    assert_eq!(
      interpret("PDF[StableDistribution[1, 0], x]").unwrap(),
      "1/(Pi*(1 + x^2))"
    );
  }

  #[test]
  fn pdf_cauchy_with_location_scale() {
    clear_state();
    // Expanded Cauchy form: sigma/(Pi*(sigma^2 + (x-mu)^2))
    assert_eq!(
      interpret("PDF[StableDistribution[1, 0, 2, 3], x]").unwrap(),
      "3/(Pi*(9 + (-2 + x)^2))"
    );
  }

  #[test]
  fn pdf_normal_special_case() {
    clear_state();
    // StableDistribution[2, 0] corresponds to Normal(0, Sqrt[2])
    let result = interpret("PDF[StableDistribution[2, 0], x]").unwrap();
    assert!(
      result.contains("E^"),
      "Expected exponential form, got: {}",
      result
    );
    assert!(
      result.contains("Pi"),
      "Expected Pi in denominator, got: {}",
      result
    );
  }

  #[test]
  fn pdf_general_unevaluated() {
    clear_state();
    // General case returns unevaluated (with canonical 5-param form)
    assert_eq!(
      interpret("PDF[StableDistribution[3/2, 1/2], x]").unwrap(),
      "PDF[StableDistribution[1, 3/2, 1/2, 0, 1], x]"
    );
  }

  #[test]
  fn cdf_cauchy_standard() {
    clear_state();
    let result = interpret("CDF[StableDistribution[1, 0], x]").unwrap();
    assert!(
      result.contains("ArcTan"),
      "Expected ArcTan, got: {}",
      result
    );
  }
}

mod location_test {
  use super::*;

  #[test]
  fn one_sample_default_mu0() {
    // LocationTest[data] tests if mean is 0
    let result =
      interpret("LocationTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 0.0033200264871519245).abs() < 1e-10,
      "Expected ~0.00332, got {}",
      val
    );
  }

  #[test]
  fn one_sample_test_statistic() {
    let result = interpret(
      "LocationTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, 0, \"TestStatistic\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 5.252257314388901).abs() < 1e-10,
      "Expected ~5.2523, got {}",
      val
    );
  }

  #[test]
  fn one_sample_with_mu0() {
    let result =
      interpret("LocationTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, 1, \"PValue\"]")
        .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 0.2461912778840749).abs() < 1e-8,
      "Expected ~0.2462, got {}",
      val
    );
  }

  #[test]
  fn one_sample_with_mu0_test_statistic() {
    let result = interpret(
      "LocationTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, 1, \"TestStatistic\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 1.313064328597225).abs() < 1e-10,
      "Expected ~1.3131, got {}",
      val
    );
  }

  #[test]
  fn one_sample_integers() {
    let result = interpret("LocationTest[{1, 2, 3}, 0, \"PValue\"]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 0.07417990022744857).abs() < 1e-8,
      "Expected ~0.0742, got {}",
      val
    );
  }

  #[test]
  fn one_sample_integers_test_statistic() {
    let result =
      interpret("LocationTest[{1, 2, 3}, 0, \"TestStatistic\"]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 3.464101615137754).abs() < 1e-8,
      "Expected ~3.4641, got {}",
      val
    );
  }

  #[test]
  fn two_sample_test() {
    let result = interpret(
      "LocationTest[{{1.2, 0.5, 1.9}, {2.1, 0.8, 1.5}}, 0, \"PValue\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 0.6541475396789262).abs() < 1e-3,
      "Expected ~0.654, got {}",
      val
    );
  }

  #[test]
  fn two_sample_test_statistic() {
    let result = interpret(
      "LocationTest[{{1.2, 0.5, 1.9}, {2.1, 0.8, 1.5}}, 0, \"TestStatistic\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - (-0.4832976746641418)).abs() < 1e-10,
      "Expected ~-0.4833, got {}",
      val
    );
  }

  #[test]
  fn mean_equals_mu0() {
    // When mu0 equals the sample mean, p-value should be 1 and t-stat should be 0
    let result = interpret(
      "LocationTest[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5.5, \"TestStatistic\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(val.abs() < 1e-10, "Expected 0, got {}", val);
  }

  #[test]
  fn test_data_table() {
    // TestDataTable returns a Grid expression (Woxi keeps Grid symbolic in
    // CLI mode to match wolframscript). The grid contains the column
    // headers and a row with the T statistic and P-value.
    let result = interpret(
      "LocationTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, 0, \"TestDataTable\"]",
    )
    .unwrap();
    assert!(
      result.starts_with("Grid["),
      "Expected Grid[...], got: {}",
      result
    );
    assert!(result.contains("Statistic"), "Expected 'Statistic' header");
    assert!(
      result.contains("P\u{2010}Value"),
      "Expected 'P-Value' header"
    );
  }

  #[test]
  fn automatic_mu0() {
    // Automatic should be treated as 0
    let result = interpret(
      "LocationTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, Automatic, \"PValue\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 0.0033200264871519245).abs() < 1e-10,
      "Expected ~0.00332, got {}",
      val
    );
  }
}

mod discrete_asymptotic {
  use super::*;

  #[test]
  fn factorial_stirling() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Factorial[n], n -> Infinity]").unwrap(),
      "(n^(1/2 + n)*Sqrt[2*Pi])/E^n"
    );
  }

  #[test]
  fn polynomial_leading_term() {
    assert_eq!(
      interpret("DiscreteAsymptotic[n^3 + 5*n^2 + 1, n -> Infinity]").unwrap(),
      "n^3"
    );
  }

  #[test]
  fn polynomial_with_coefficient() {
    assert_eq!(
      interpret("DiscreteAsymptotic[3*n^2/2 + n, n -> Infinity]").unwrap(),
      "(3*n^2)/2"
    );
  }

  #[test]
  fn exponential_identity() {
    assert_eq!(
      interpret("DiscreteAsymptotic[2^n, n -> Infinity]").unwrap(),
      "2^n"
    );
  }

  #[test]
  fn exponential_dominant() {
    assert_eq!(
      interpret("DiscreteAsymptotic[3^n + 2^n, n -> Infinity]").unwrap(),
      "3^n"
    );
  }

  #[test]
  fn harmonic_number() {
    assert_eq!(
      interpret("DiscreteAsymptotic[HarmonicNumber[n], n -> Infinity]")
        .unwrap(),
      "Log[n]"
    );
  }

  #[test]
  fn inverse_n() {
    assert_eq!(
      interpret("DiscreteAsymptotic[1/n, n -> Infinity]").unwrap(),
      "n^(-1)"
    );
  }

  #[test]
  fn sqrt_n() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Sqrt[n], n -> Infinity]").unwrap(),
      "Sqrt[n]"
    );
  }

  #[test]
  fn n_log_n() {
    assert_eq!(
      interpret("DiscreteAsymptotic[n*Log[n], n -> Infinity]").unwrap(),
      "n*Log[n]"
    );
  }

  #[test]
  fn gamma_stirling() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Gamma[n], n -> Infinity]").unwrap(),
      "(n^(-1/2 + n)*Sqrt[2*Pi])/E^n"
    );
  }

  #[test]
  fn polynomial_over_exponential() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Log[n] + n^2, n -> Infinity]").unwrap(),
      "n^2"
    );
  }

  #[test]
  fn binomial_central() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Binomial[n, n/2], n -> Infinity]").unwrap(),
      "2^(1/2 + n)/(Sqrt[n]*Sqrt[Pi])"
    );
  }

  #[test]
  fn constant_expression() {
    assert_eq!(
      interpret("DiscreteAsymptotic[42, n -> Infinity]").unwrap(),
      "42"
    );
  }

  #[test]
  fn identity_n() {
    assert_eq!(
      interpret("DiscreteAsymptotic[n, n -> Infinity]").unwrap(),
      "n"
    );
  }
}

mod likelihood {
  use super::*;

  #[test]
  fn standard_normal_numeric() {
    let result =
      interpret("Likelihood[NormalDistribution[], {0.5, 1.0, -0.3}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 0.032490238142406966).abs() < 1e-12,
      "Expected ~0.0325, got {}",
      val
    );
  }

  #[test]
  fn exponential_numeric() {
    let result =
      interpret("Likelihood[ExponentialDistribution[2], {1, 2, 3}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 0.00004915369882662568).abs() < 1e-15,
      "Expected ~4.9e-5, got {}",
      val
    );
  }

  #[test]
  fn poisson_numeric() {
    let result =
      interpret("Likelihood[PoissonDistribution[2], {1, 2, 3}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 0.013220011608887245).abs() < 1e-12,
      "Expected ~0.01322, got {}",
      val
    );
  }

  #[test]
  fn exponential_symbolic() {
    let result =
      interpret("Likelihood[ExponentialDistribution[lambda], {1, 2, 3}]")
        .unwrap();
    // Result should contain lambda^3 and exponential terms
    assert!(
      result.contains("lambda"),
      "Expected symbolic result with lambda, got: {}",
      result
    );
  }

  #[test]
  fn empty_data() {
    assert_eq!(
      interpret("Likelihood[NormalDistribution[], {}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn single_data_point() {
    let result = interpret("Likelihood[NormalDistribution[], {0.0}]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    assert!(
      (val - expected).abs() < 1e-12,
      "Expected ~{}, got {}",
      expected,
      val
    );
  }
}

mod group_generators {
  use super::*;

  #[test]
  fn symmetric_group_3() {
    assert_eq!(
      interpret("GroupGenerators[SymmetricGroup[3]]").unwrap(),
      "{Cycles[{{1, 2}}], Cycles[{{1, 2, 3}}]}"
    );
  }

  #[test]
  fn symmetric_group_5() {
    assert_eq!(
      interpret("GroupGenerators[SymmetricGroup[5]]").unwrap(),
      "{Cycles[{{1, 2}}], Cycles[{{1, 2, 3, 4, 5}}]}"
    );
  }

  #[test]
  fn cyclic_group_5() {
    assert_eq!(
      interpret("GroupGenerators[CyclicGroup[5]]").unwrap(),
      "{Cycles[{{1, 2, 3, 4, 5}}]}"
    );
  }

  #[test]
  fn dihedral_group_3() {
    assert_eq!(
      interpret("GroupGenerators[DihedralGroup[3]]").unwrap(),
      "{Cycles[{{2, 3}}], Cycles[{{1, 2, 3}}]}"
    );
  }

  #[test]
  fn dihedral_group_4() {
    assert_eq!(
      interpret("GroupGenerators[DihedralGroup[4]]").unwrap(),
      "{Cycles[{{1, 4}, {2, 3}}], Cycles[{{1, 2, 3, 4}}]}"
    );
  }

  #[test]
  fn dihedral_group_5() {
    assert_eq!(
      interpret("GroupGenerators[DihedralGroup[5]]").unwrap(),
      "{Cycles[{{2, 5}, {3, 4}}], Cycles[{{1, 2, 3, 4, 5}}]}"
    );
  }

  #[test]
  fn alternating_group_3() {
    assert_eq!(
      interpret("GroupGenerators[AlternatingGroup[3]]").unwrap(),
      "{Cycles[{{1, 2, 3}}]}"
    );
  }

  #[test]
  fn alternating_group_4() {
    assert_eq!(
      interpret("GroupGenerators[AlternatingGroup[4]]").unwrap(),
      "{Cycles[{{1, 2, 3}}], Cycles[{{2, 3, 4}}]}"
    );
  }

  #[test]
  fn alternating_group_5() {
    assert_eq!(
      interpret("GroupGenerators[AlternatingGroup[5]]").unwrap(),
      "{Cycles[{{1, 2, 3}}], Cycles[{{1, 2, 3, 4, 5}}]}"
    );
  }

  #[test]
  fn abelian_group_2_2_3() {
    assert_eq!(
      interpret("GroupGenerators[AbelianGroup[{2, 2, 3}]]").unwrap(),
      "{Cycles[{{1, 2}}], Cycles[{{3, 4}}], Cycles[{{5, 6, 7}}]}"
    );
  }

  #[test]
  fn abelian_group_5() {
    assert_eq!(
      interpret("GroupGenerators[AbelianGroup[{5}]]").unwrap(),
      "{Cycles[{{1, 2, 3, 4, 5}}]}"
    );
  }

  #[test]
  fn abelian_group_skips_trivial_factors() {
    // ni = 1 consumes a slot but emits no generator.
    assert_eq!(
      interpret("GroupGenerators[AbelianGroup[{1, 2, 3}]]").unwrap(),
      "{Cycles[{{2, 3}}], Cycles[{{4, 5, 6}}]}"
    );
  }

  #[test]
  fn abelian_group_empty() {
    assert_eq!(
      interpret("GroupGenerators[AbelianGroup[{}]]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn permutation_group_round_trips() {
    // GroupGenerators[PermutationGroup[{gens}]] returns {gens} verbatim.
    assert_eq!(
      interpret(
        "GroupGenerators[PermutationGroup[{Cycles[{{1, 2}}], Cycles[{{1, 3}, {2, 4}}]}]]"
      )
      .unwrap(),
      "{Cycles[{{1, 2}}], Cycles[{{1, 3}, {2, 4}}]}"
    );
  }

  #[test]
  fn permutation_group_single_generator() {
    assert_eq!(
      interpret("GroupGenerators[PermutationGroup[{Cycles[{{1, 2}}]}]]")
        .unwrap(),
      "{Cycles[{{1, 2}}]}"
    );
  }

  #[test]
  fn permutation_group_stays_symbolic() {
    // PermutationGroup itself preserves its argument when no operation is applied.
    assert_eq!(
      interpret("PermutationGroup[{Cycles[{{1, 2}}]}]").unwrap(),
      "PermutationGroup[{Cycles[{{1, 2}}]}]"
    );
  }
}

mod mathieu_groups {
  use super::*;

  #[test]
  fn formal_heads_echo() {
    for g in ["M11", "M12", "M22", "M23", "M24"] {
      let input = format!("MathieuGroup{g}[]");
      assert_eq!(interpret(&input).unwrap(), input);
    }
  }

  #[test]
  fn group_orders() {
    for (g, order) in [
      ("M11", "7920"),
      ("M12", "95040"),
      ("M22", "443520"),
      ("M23", "10200960"),
      ("M24", "244823040"),
    ] {
      assert_eq!(
        interpret(&format!("GroupOrder[MathieuGroup{g}[]]")).unwrap(),
        order
      );
    }
  }

  #[test]
  fn group_generators() {
    assert_eq!(
      interpret("GroupGenerators[MathieuGroupM11[]]").unwrap(),
      "{Cycles[{{2, 10}, {4, 11}, {5, 7}, {8, 9}}], \
       Cycles[{{1, 4, 3, 8}, {2, 5, 6, 9}}]}"
    );
    assert_eq!(
      interpret("GroupGenerators[MathieuGroupM12[]]").unwrap(),
      "{Cycles[{{1, 4}, {3, 10}, {5, 11}, {6, 12}}], \
       Cycles[{{1, 8, 9}, {2, 3, 4}, {5, 12, 11}, {6, 10, 7}}]}"
    );
    assert_eq!(
      interpret("GroupGenerators[MathieuGroupM22[]]").unwrap(),
      "{Cycles[{{1, 13}, {2, 8}, {3, 16}, {4, 12}, {6, 22}, {7, 17}, \
       {9, 10}, {11, 14}}], Cycles[{{1, 22, 3, 21}, {2, 18, 4, 13}, \
       {5, 12}, {6, 11, 7, 15}, {8, 14, 20, 10}, {17, 19}}]}"
    );
    assert_eq!(
      interpret("GroupGenerators[MathieuGroupM23[]]").unwrap(),
      "{Cycles[{{1, 2}, {3, 4}, {7, 8}, {9, 10}, {13, 14}, {15, 16}, \
       {19, 20}, {21, 22}}], Cycles[{{1, 16, 11, 3}, {2, 9, 21, 12}, \
       {4, 5, 8, 23}, {6, 22, 14, 18}, {13, 20}, {15, 17}}]}"
    );
    assert_eq!(
      interpret("GroupGenerators[MathieuGroupM24[]]").unwrap(),
      "{Cycles[{{1, 4}, {2, 7}, {3, 17}, {5, 13}, {6, 9}, {8, 15}, \
       {10, 19}, {11, 18}, {12, 21}, {14, 16}, {20, 24}, {22, 23}}], \
       Cycles[{{1, 4, 6}, {2, 21, 14}, {3, 9, 15}, {5, 18, 10}, \
       {13, 17, 16}, {19, 24, 23}}]}"
    );
  }
}

mod group_order {
  use super::*;

  #[test]
  fn abelian_group_product_of_factors() {
    assert_eq!(
      interpret("GroupOrder[AbelianGroup[{2, 2, 3}]]").unwrap(),
      "12"
    );
    assert_eq!(
      interpret("GroupOrder[AbelianGroup[{2, 3, 5, 7}]]").unwrap(),
      "210"
    );
    assert_eq!(interpret("GroupOrder[AbelianGroup[{4}]]").unwrap(), "4");
  }

  #[test]
  fn abelian_group_empty_is_one() {
    assert_eq!(interpret("GroupOrder[AbelianGroup[{}]]").unwrap(), "1");
  }

  #[test]
  fn cyclic_group_n() {
    assert_eq!(interpret("GroupOrder[CyclicGroup[5]]").unwrap(), "5");
    assert_eq!(interpret("GroupOrder[CyclicGroup[1]]").unwrap(), "1");
  }

  #[test]
  fn symmetric_group_n_factorial() {
    assert_eq!(interpret("GroupOrder[SymmetricGroup[4]]").unwrap(), "24");
    assert_eq!(interpret("GroupOrder[SymmetricGroup[5]]").unwrap(), "120");
  }

  #[test]
  fn alternating_group_half_factorial() {
    assert_eq!(interpret("GroupOrder[AlternatingGroup[4]]").unwrap(), "12");
    assert_eq!(interpret("GroupOrder[AlternatingGroup[5]]").unwrap(), "60");
  }

  #[test]
  fn dihedral_group_2n() {
    assert_eq!(interpret("GroupOrder[DihedralGroup[4]]").unwrap(), "8");
    assert_eq!(interpret("GroupOrder[DihedralGroup[5]]").unwrap(), "10");
  }
}

mod group_elements {
  use super::*;

  #[test]
  fn abelian_group_2_2_3() {
    // 12 elements in lex order (rightmost generator varies fastest).
    assert_eq!(
      interpret("GroupElements[AbelianGroup[{2, 2, 3}]]").unwrap(),
      "{Cycles[{}], Cycles[{{5, 6, 7}}], Cycles[{{5, 7, 6}}], \
       Cycles[{{3, 4}}], Cycles[{{3, 4}, {5, 6, 7}}], \
       Cycles[{{3, 4}, {5, 7, 6}}], Cycles[{{1, 2}}], \
       Cycles[{{1, 2}, {5, 6, 7}}], Cycles[{{1, 2}, {5, 7, 6}}], \
       Cycles[{{1, 2}, {3, 4}}], Cycles[{{1, 2}, {3, 4}, {5, 6, 7}}], \
       Cycles[{{1, 2}, {3, 4}, {5, 7, 6}}]}"
    );
  }

  #[test]
  fn abelian_group_3_2() {
    assert_eq!(
      interpret("GroupElements[AbelianGroup[{3, 2}]]").unwrap(),
      "{Cycles[{}], Cycles[{{4, 5}}], Cycles[{{1, 2, 3}}], \
       Cycles[{{1, 2, 3}, {4, 5}}], Cycles[{{1, 3, 2}}], \
       Cycles[{{1, 3, 2}, {4, 5}}]}"
    );
  }

  #[test]
  fn abelian_group_single_2() {
    assert_eq!(
      interpret("GroupElements[AbelianGroup[{2}]]").unwrap(),
      "{Cycles[{}], Cycles[{{1, 2}}]}"
    );
  }

  #[test]
  fn abelian_group_skips_trivial_factors() {
    // {1, 2, 3} → ni=1 contributes a slot but no element factor.
    assert_eq!(
      interpret("GroupElements[AbelianGroup[{1, 2, 3}]]").unwrap(),
      "{Cycles[{}], Cycles[{{4, 5, 6}}], Cycles[{{4, 6, 5}}], \
       Cycles[{{2, 3}}], Cycles[{{2, 3}, {4, 5, 6}}], \
       Cycles[{{2, 3}, {4, 6, 5}}]}"
    );
  }

  #[test]
  fn abelian_group_empty() {
    assert_eq!(
      interpret("GroupElements[AbelianGroup[{}]]").unwrap(),
      "{Cycles[{}]}"
    );
  }

  // GroupElements[group, {positions}] selects the elements at the given 1-based
  // positions of the full element list, in the given order. Verified against
  // wolframscript.
  #[test]
  fn positional_selection() {
    // GroupElements[SymmetricGroup[3]] =
    //   {Cycles[{}], Cycles[{{2,3}}], Cycles[{{1,2}}], Cycles[{{1,2,3}}],
    //    Cycles[{{1,3,2}}], Cycles[{{1,3}}]}
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[3], {1, 3}]").unwrap(),
      "{Cycles[{}], Cycles[{{1, 2}}]}"
    );
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[3], {2}]").unwrap(),
      "{Cycles[{{2, 3}}]}"
    );
    assert_eq!(
      interpret("GroupElements[CyclicGroup[4], {1, 2, 4}]").unwrap(),
      "{Cycles[{}], Cycles[{{1, 2, 3, 4}}], Cycles[{{1, 4, 3, 2}}]}"
    );
  }

  #[test]
  fn positional_order_preserved_and_negative_indices() {
    // The given order is preserved; negative positions count from the end.
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[3], {3, 1}]").unwrap(),
      "{Cycles[{{1, 2}}], Cycles[{}]}"
    );
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[3], {-1}]").unwrap(),
      "{Cycles[{{1, 3}}]}"
    );
    // Duplicates are allowed.
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[3], {1, 1}]").unwrap(),
      "{Cycles[{}], Cycles[{}]}"
    );
  }

  #[test]
  fn positional_out_of_range_stays_unevaluated() {
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[3], {1, 99}]").unwrap(),
      "GroupElements[SymmetricGroup[3], {1, 99}]"
    );
    // A bare integer (not a list of positions) is not the positional form.
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[3], 3]").unwrap(),
      "GroupElements[SymmetricGroup[3], 3]"
    );
  }
}

// GroupOrbits[group, {points}] gives the orbits of the seed points under the
// group action: each orbit is the sorted set of images of a seed under every
// group element, deduplicated, with the orbits sorted lexicographically.
// Verified against wolframscript.
mod group_orbits {
  use super::*;

  #[test]
  fn transitive_group_single_orbit() {
    assert_eq!(
      interpret("GroupOrbits[SymmetricGroup[3], {1}]").unwrap(),
      "{{1, 2, 3}}"
    );
    assert_eq!(
      interpret("GroupOrbits[CyclicGroup[4], {1, 3}]").unwrap(),
      "{{1, 2, 3, 4}}"
    );
    assert_eq!(
      interpret("GroupOrbits[DihedralGroup[4], {1}]").unwrap(),
      "{{1, 2, 3, 4}}"
    );
  }

  #[test]
  fn seeds_in_same_orbit_are_deduplicated() {
    assert_eq!(
      interpret("GroupOrbits[SymmetricGroup[3], {1, 2}]").unwrap(),
      "{{1, 2, 3}}"
    );
    assert_eq!(
      interpret("GroupOrbits[SymmetricGroup[3], {1, 1}]").unwrap(),
      "{{1, 2, 3}}"
    );
  }

  #[test]
  fn points_outside_the_domain_are_fixed() {
    // The orbits are sorted regardless of the seed order; points the group
    // never moves form singleton orbits.
    assert_eq!(
      interpret("GroupOrbits[CyclicGroup[4], {6, 1, 5}]").unwrap(),
      "{{1, 2, 3, 4}, {5}, {6}}"
    );
  }

  #[test]
  fn abelian_group_has_multiple_orbits() {
    assert_eq!(
      interpret("GroupOrbits[AbelianGroup[{2, 2}], {1, 3}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }
}

// GroupElementQ[group, perm] tests whether a permutation (in cycle notation)
// is an element of the group. Verified against wolframscript.
mod group_element_q {
  use super::*;

  #[test]
  fn member_permutations() {
    assert_eq!(
      interpret("GroupElementQ[SymmetricGroup[3], Cycles[{{1, 2}}]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("GroupElementQ[SymmetricGroup[3], Cycles[{{1, 2, 3}}]]")
        .unwrap(),
      "True"
    );
    // The identity is in every group.
    assert_eq!(
      interpret("GroupElementQ[SymmetricGroup[3], Cycles[{}]]").unwrap(),
      "True"
    );
    // The generator of C4 is a member; its non-canonical cycle form matches
    // by permutation action.
    assert_eq!(
      interpret("GroupElementQ[CyclicGroup[4], Cycles[{{1, 2, 3, 4}}]]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("GroupElementQ[SymmetricGroup[3], Cycles[{{2, 1}}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn non_member_permutations() {
    // A permutation moving a point outside the domain is not in S3.
    assert_eq!(
      interpret("GroupElementQ[SymmetricGroup[3], Cycles[{{1, 4}}]]").unwrap(),
      "False"
    );
    // An odd permutation is not in the cyclic group C4.
    assert_eq!(
      interpret("GroupElementQ[CyclicGroup[4], Cycles[{{1, 2}}]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn invalid_permutation_stays_unevaluated() {
    assert_eq!(
      interpret("GroupElementQ[SymmetricGroup[3], foo]").unwrap(),
      "GroupElementQ[SymmetricGroup[3], foo]"
    );
  }
}

// GroupElementPosition[group, perm] gives the 1-based position of a permutation
// in GroupElements[group]. Verified against wolframscript.
mod group_element_position {
  use super::*;

  #[test]
  fn position_in_element_list() {
    // GroupElements[SymmetricGroup[3]] =
    //   {Cycles[{}], Cycles[{{2,3}}], Cycles[{{1,2}}], Cycles[{{1,2,3}}], …}
    assert_eq!(
      interpret("GroupElementPosition[SymmetricGroup[3], Cycles[{}]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("GroupElementPosition[SymmetricGroup[3], Cycles[{{1, 2}}]]")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("GroupElementPosition[SymmetricGroup[3], Cycles[{{1, 2, 3}}]]")
        .unwrap(),
      "4"
    );
    assert_eq!(
      interpret("GroupElementPosition[CyclicGroup[4], Cycles[{{1, 2, 3, 4}}]]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn non_canonical_cycle_matches_by_action() {
    assert_eq!(
      interpret("GroupElementPosition[SymmetricGroup[3], Cycles[{{2, 1}}]]")
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn permutation_not_in_group_stays_unevaluated() {
    assert_eq!(
      interpret("GroupElementPosition[SymmetricGroup[3], Cycles[{{1, 4}}]]")
        .unwrap(),
      "GroupElementPosition[SymmetricGroup[3], Cycles[{{1, 4}}]]"
    );
  }
}

mod longitude_latitude {
  use super::*;

  #[test]
  fn longitude_geoposition() {
    assert_eq!(
      interpret("Longitude[GeoPosition[{40.7128, -74.006}]]").unwrap(),
      "Quantity[-74.006, AngularDegrees]"
    );
  }

  #[test]
  fn latitude_geoposition() {
    assert_eq!(
      interpret("Latitude[GeoPosition[{40.7128, -74.006}]]").unwrap(),
      "Quantity[40.7128, AngularDegrees]"
    );
  }

  #[test]
  fn longitude_list() {
    assert_eq!(
      interpret("Longitude[{1, 2}]").unwrap(),
      "Quantity[2, AngularDegrees]"
    );
  }

  #[test]
  fn latitude_list() {
    assert_eq!(
      interpret("Latitude[{1, 2}]").unwrap(),
      "Quantity[1, AngularDegrees]"
    );
  }

  #[test]
  fn longitude_integers() {
    assert_eq!(
      interpret("Longitude[GeoPosition[{40, -74}]]").unwrap(),
      "Quantity[-74, AngularDegrees]"
    );
  }

  #[test]
  fn latitude_longitude_geoposition() {
    assert_eq!(
      interpret("LatitudeLongitude[GeoPosition[{40.7128, -74.006}]]").unwrap(),
      "{Quantity[40.7128, AngularDegrees], Quantity[-74.006, AngularDegrees]}"
    );
  }

  #[test]
  fn latitude_longitude_list() {
    assert_eq!(
      interpret("LatitudeLongitude[{51.5, -0.12}]").unwrap(),
      "{Quantity[51.5, AngularDegrees], Quantity[-0.12, AngularDegrees]}"
    );
  }
}

mod pearson_chi_square_test {
  use super::*;

  #[test]
  fn default_returns_pvalue() {
    let result =
      interpret("PearsonChiSquareTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}]")
        .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      val > 0.0 && val < 1.0,
      "Expected p-value between 0 and 1, got {}",
      val
    );
  }

  #[test]
  fn test_statistic_automatic() {
    let result = interpret(
      "PearsonChiSquareTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, Automatic, \"TestStatistic\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      val >= 0.0,
      "Expected non-negative test statistic, got {}",
      val
    );
  }

  #[test]
  fn normal_distribution_test_statistic() {
    let result = interpret(
      "PearsonChiSquareTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, NormalDistribution[], \"TestStatistic\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    // Data doesn't fit N(0,1) well, so statistic should be large
    assert!(
      val > 5.0,
      "Expected large chi-square statistic for non-fitting distribution, got {}",
      val
    );
  }

  #[test]
  fn normal_distribution_pvalue() {
    let result = interpret(
      "PearsonChiSquareTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, NormalDistribution[], \"PValue\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    // p-value should be small since data doesn't fit N(0,1)
    assert!(val < 0.1, "Expected small p-value, got {}", val);
  }

  #[test]
  fn well_fitting_data_high_pvalue() {
    // Data from N(0,1) should have high p-value
    let result = interpret(
      "PearsonChiSquareTest[{-0.5, 0.3, -1.2, 0.8, -0.1, 0.6, -0.3, 1.1, -0.7, 0.2}]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      val > 0.05,
      "Expected high p-value for well-fitting data, got {}",
      val
    );
  }
}

mod multivariate_poisson_distribution {
  use super::*;

  #[test]
  fn symbolic_form() {
    assert_eq!(
      interpret("MultivariatePoissonDistribution[1, {2, 3}]").unwrap(),
      "MultivariatePoissonDistribution[1, {2, 3}]"
    );
  }

  #[test]
  fn symbolic_parameters() {
    assert_eq!(
      interpret("MultivariatePoissonDistribution[a, {b, c}]").unwrap(),
      "MultivariatePoissonDistribution[a, {b, c}]"
    );
  }

  #[test]
  fn mean_numeric() {
    assert_eq!(
      interpret("Mean[MultivariatePoissonDistribution[1, {2, 3}]]").unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[MultivariatePoissonDistribution[a, {b, c}]]").unwrap(),
      "{a + b, a + c}"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[MultivariatePoissonDistribution[1, {2, 3}]]")
        .unwrap(),
      "{3, 4}"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[MultivariatePoissonDistribution[a, {b, c}]]")
        .unwrap(),
      "{a + b, a + c}"
    );
  }

  // ─── PDF for the bivariate Poisson distribution ───────────────────
  //
  // PDF[MultivariatePoissonDistribution[μ_0, {μ_1, μ_2}], {x, y}] has
  // the closed form (for non-negative integer x, y):
  //
  //   (-μ_0)^x · μ_2^(y-x) · HypergeometricU[-x, 1-x+y, -μ_1·μ_2/μ_0]
  //   ───────────────────────────────────────────────────────────────
  //              E^(μ_0+μ_1+μ_2) · x! · y!
  //
  // Wolfram wraps it in `Piecewise[{{<formula>, x >= 0 && y >= 0}}, 0]`.
  #[test]
  fn pdf_symbolic_audit_case() {
    // Audit case.
    assert_eq!(
      interpret("PDF[MultivariatePoissonDistribution[1, {2, 3}], {x, y}]")
        .unwrap(),
      "Piecewise[{{((-1)^x*3^(-x + y)*HypergeometricU[-x, 1 - x + y, -6])/(E^6*x!*y!), x >= 0 && y >= 0}}, 0]"
    );
  }

  #[test]
  fn pdf_symbolic_different_params() {
    assert_eq!(
      interpret("PDF[MultivariatePoissonDistribution[1, {4, 3}], {x, y}]")
        .unwrap(),
      "Piecewise[{{((-1)^x*3^(-x + y)*HypergeometricU[-x, 1 - x + y, -12])/(E^8*x!*y!), x >= 0 && y >= 0}}, 0]"
    );
  }

  // Concrete (x, y) evaluates to the correct numeric closed form
  // (Wolfram prints `E^(-6)` rather than `1/E^6`).
  #[test]
  fn pdf_concrete_at_origin() {
    assert_eq!(
      interpret("PDF[MultivariatePoissonDistribution[1, {2, 3}], {0, 0}]")
        .unwrap(),
      "E^(-6)"
    );
  }

  #[test]
  fn pdf_concrete_at_2_3() {
    assert_eq!(
      interpret("PDF[MultivariatePoissonDistribution[1, {2, 3}], {2, 3}]")
        .unwrap(),
      "39/(2*E^6)"
    );
  }
}

mod binormal_distribution {
  use super::*;

  // BinormalDistribution[{m1, m2}, {s1, s2}, rho]: the mean is the location
  // vector, variances are the squared sigmas, and Covariance is the 2x2 matrix.
  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[BinormalDistribution[{m1, m2}, {s1, s2}, r]]").unwrap(),
      "{m1, m2}"
    );
  }

  #[test]
  fn mean_numeric_and_standard() {
    assert_eq!(
      interpret("Mean[BinormalDistribution[{1, 2}, {1, 1}, 0]]").unwrap(),
      "{1, 2}"
    );
    // One-argument standard form has zero means.
    assert_eq!(
      interpret("Mean[BinormalDistribution[r]]").unwrap(),
      "{0, 0}"
    );
  }

  #[test]
  fn variance_and_standard_deviation() {
    assert_eq!(
      interpret("Variance[BinormalDistribution[{m1, m2}, {s1, s2}, r]]")
        .unwrap(),
      "{s1^2, s2^2}"
    );
    assert_eq!(
      interpret(
        "StandardDeviation[BinormalDistribution[{m1, m2}, {s1, s2}, r]]"
      )
      .unwrap(),
      "{s1, s2}"
    );
    assert_eq!(
      interpret("Variance[BinormalDistribution[r]]").unwrap(),
      "{1, 1}"
    );
  }

  #[test]
  fn covariance_matrix() {
    assert_eq!(
      interpret("Covariance[BinormalDistribution[{m1, m2}, {s1, s2}, r]]")
        .unwrap(),
      "{{s1^2, r*s1*s2}, {r*s1*s2, s2^2}}"
    );
    assert_eq!(
      interpret("Covariance[BinormalDistribution[r]]").unwrap(),
      "{{1, r}, {r, 1}}"
    );
  }
}

mod arcsin_distribution {
  use super::*;

  #[test]
  fn default_form() {
    assert_eq!(
      interpret("ArcSinDistribution[]").unwrap(),
      "ArcSinDistribution[{0, 1}]"
    );
  }

  #[test]
  fn symbolic_form() {
    assert_eq!(
      interpret("ArcSinDistribution[{a, b}]").unwrap(),
      "ArcSinDistribution[{a, b}]"
    );
  }

  #[test]
  fn mean_default() {
    assert_eq!(interpret("Mean[ArcSinDistribution[]]").unwrap(), "1/2");
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[ArcSinDistribution[{a, b}]]").unwrap(),
      "(a + b)/2"
    );
  }

  #[test]
  fn variance_default() {
    assert_eq!(interpret("Variance[ArcSinDistribution[]]").unwrap(), "1/8");
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[ArcSinDistribution[{a, b}]]").unwrap(),
      "(-a + b)^2/8"
    );
  }
}

mod noncentral_f_ratio_distribution {
  use super::*;

  #[test]
  fn displays_unevaluated() {
    assert_eq!(
      interpret("NoncentralFRatioDistribution[3, 5, 2]").unwrap(),
      "NoncentralFRatioDistribution[3, 5, 2]"
    );
  }

  #[test]
  fn pdf_symbolic() {
    assert_eq!(
      interpret("PDF[NoncentralFRatioDistribution[n, m, l], x]").unwrap(),
      "Piecewise[{{(m^(m/2)*n^(n/2)*x^((-2 + n)/2)*(m + n*x)^((-m - n)/2)*Hypergeometric1F1[(m + n)/2, n/2, (l*n*x)/(2*(m + n*x))])/(E^(l/2)*Beta[n/2, m/2]), x > 0}}, 0]"
    );
  }

  #[test]
  fn pdf_zero() {
    assert_eq!(
      interpret("PDF[NoncentralFRatioDistribution[3, 5, 2], 0]").unwrap(),
      "0"
    );
  }

  #[test]
  fn pdf_negative() {
    assert_eq!(
      interpret("PDF[NoncentralFRatioDistribution[3, 5, 2], -1]").unwrap(),
      "0"
    );
  }

  #[test]
  fn mean_numeric() {
    assert_eq!(
      interpret("Mean[NoncentralFRatioDistribution[3, 5, 2]]").unwrap(),
      "25/9"
    );
  }

  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[NoncentralFRatioDistribution[n, m, l]]").unwrap(),
      "Piecewise[{{(m*(l + n))/((-2 + m)*n), m > 2}}, Indeterminate]"
    );
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[NoncentralFRatioDistribution[3, 5, 2]]").unwrap(),
      "2300/81"
    );
  }

  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[NoncentralFRatioDistribution[n, m, l]]").unwrap(),
      "Piecewise[{{(2*m^2*((l + n)^2 + (-2 + m)*(2*l + n)))/((-4 + m)*(-2 + m)^2*n^2), m > 4}}, Indeterminate]"
    );
  }

  #[test]
  fn mean_indeterminate_when_m_leq_2() {
    assert_eq!(
      interpret("Mean[NoncentralFRatioDistribution[3, 2, 1]]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Mean[NoncentralFRatioDistribution[3, 1, 1]]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn variance_indeterminate_when_m_leq_4() {
    assert_eq!(
      interpret("Variance[NoncentralFRatioDistribution[3, 4, 1]]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Variance[NoncentralFRatioDistribution[3, 3, 1]]").unwrap(),
      "Indeterminate"
    );
  }
}

mod fratio_distribution {
  use super::*;

  // FRatioDistribution[n, m]: Mean = m/(m-2) (m > 2),
  // Variance = 2 m^2 (m + n - 2) / ((m - 4)(m - 2)^2 n) (m > 4).
  #[test]
  fn mean_numeric() {
    assert_eq!(interpret("Mean[FRatioDistribution[5, 10]]").unwrap(), "5/4");
    assert_eq!(interpret("Mean[FRatioDistribution[3, 6]]").unwrap(), "3/2");
  }

  #[test]
  fn variance_numeric() {
    assert_eq!(
      interpret("Variance[FRatioDistribution[5, 10]]").unwrap(),
      "65/48"
    );
    assert_eq!(
      interpret("StandardDeviation[FRatioDistribution[5, 10]]").unwrap(),
      "Sqrt[65/3]/4"
    );
  }

  #[test]
  fn symbolic_piecewise() {
    assert_eq!(
      interpret("Mean[FRatioDistribution[n, m]]").unwrap(),
      "Piecewise[{{m/(-2 + m), m > 2}}, Indeterminate]"
    );
    assert_eq!(
      interpret("Variance[FRatioDistribution[n, m]]").unwrap(),
      "Piecewise[{{(2*m^2*(-2 + m + n))/((-4 + m)*(-2 + m)^2*n), m > 4}}, Indeterminate]"
    );
  }

  // Out of the support of the mean/variance: the Piecewise default branch.
  #[test]
  fn indeterminate_low_dof() {
    assert_eq!(
      interpret("Mean[FRatioDistribution[5, 2]]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Variance[FRatioDistribution[5, 4]]").unwrap(),
      "Indeterminate"
    );
  }
}

mod johnson_distribution {
  use super::*;

  #[test]
  fn displays_unevaluated() {
    assert_eq!(
      interpret(r#"JohnsonDistribution["SU", 0, 1, 0, 1]"#).unwrap(),
      "JohnsonDistribution[SU, 0, 1, 0, 1]"
    );
    assert_eq!(
      interpret(r#"JohnsonDistribution["SB", 1, 2, 3, 4]"#).unwrap(),
      "JohnsonDistribution[SB, 1, 2, 3, 4]"
    );
  }

  // --- PDF tests ---

  #[test]
  fn pdf_sn_symbolic() {
    assert_eq!(
      interpret(
        r#"PDF[JohnsonDistribution["SN", gamma, delta, mu, sigma], x]"#
      )
      .unwrap(),
      "delta/(E^((gamma + (delta*(-mu + x))/sigma)^2/2)*Sqrt[2*Pi]*sigma)"
    );
  }

  #[test]
  fn pdf_sn_numeric() {
    assert_eq!(
      interpret(r#"PDF[JohnsonDistribution["SN", 0, 1, 0, 1], 0]"#).unwrap(),
      "1/Sqrt[2*Pi]"
    );
  }

  #[test]
  fn pdf_su_symbolic() {
    assert_eq!(
      interpret(
        r#"PDF[JohnsonDistribution["SU", gamma, delta, mu, sigma], x]"#
      )
      .unwrap(),
      "delta/(E^((gamma + delta*ArcSinh[(-mu + x)/sigma])^2/2)*Sqrt[2*Pi]*Sqrt[sigma^2 + (-mu + x)^2])"
    );
  }

  #[test]
  fn pdf_su_numeric() {
    assert_eq!(
      interpret(r#"PDF[JohnsonDistribution["SU", 0, 1, 0, 1], 0]"#).unwrap(),
      "1/Sqrt[2*Pi]"
    );
  }

  #[test]
  fn pdf_sl_symbolic() {
    assert_eq!(
      interpret(
        r#"PDF[JohnsonDistribution["SL", gamma, delta, mu, sigma], x]"#
      )
      .unwrap(),
      "Piecewise[{{delta/(E^((gamma + delta*Log[(-mu + x)/sigma])^2/2)*Sqrt[2*Pi]*(-mu + x)), x > mu}}, 0]"
    );
  }

  #[test]
  fn pdf_sl_numeric() {
    assert_eq!(
      interpret(r#"PDF[JohnsonDistribution["SL", 0, 1, 0, 1], 1]"#).unwrap(),
      "1/Sqrt[2*Pi]"
    );
  }

  #[test]
  fn pdf_sb_symbolic() {
    assert_eq!(
      interpret(
        r#"PDF[JohnsonDistribution["SB", gamma, delta, mu, sigma], x]"#
      )
      .unwrap(),
      "Piecewise[{{(delta*sigma)/(E^((gamma + delta*Log[(-mu + x)/(mu + sigma - x)])^2/2)*Sqrt[2*Pi]*(mu + sigma - x)*(-mu + x)), mu < x < mu + sigma}}, 0]"
    );
  }

  #[test]
  fn pdf_sb_numeric() {
    assert_eq!(
      interpret(r#"PDF[JohnsonDistribution["SB", 0, 1, 0, 1], 1/2]"#).unwrap(),
      "4/Sqrt[2*Pi]"
    );
  }

  // --- CDF tests ---

  #[test]
  fn cdf_sn_symbolic() {
    assert_eq!(
      interpret(
        r#"CDF[JohnsonDistribution["SN", gamma, delta, mu, sigma], x]"#
      )
      .unwrap(),
      "Erfc[(-gamma - (delta*(-mu + x))/sigma)/Sqrt[2]]/2"
    );
  }

  #[test]
  fn cdf_sn_numeric() {
    assert_eq!(
      interpret(r#"CDF[JohnsonDistribution["SN", 0, 1, 0, 1], 0]"#).unwrap(),
      "1/2"
    );
  }

  #[test]
  fn cdf_su_symbolic() {
    assert_eq!(
      interpret(
        r#"CDF[JohnsonDistribution["SU", gamma, delta, mu, sigma], x]"#
      )
      .unwrap(),
      "(1 + Erf[(gamma + delta*ArcSinh[(-mu + x)/sigma])/Sqrt[2]])/2"
    );
  }

  #[test]
  fn cdf_su_numeric() {
    assert_eq!(
      interpret(r#"CDF[JohnsonDistribution["SU", 0, 1, 0, 1], 0]"#).unwrap(),
      "1/2"
    );
  }

  #[test]
  fn cdf_sl_symbolic() {
    assert_eq!(
      interpret(
        r#"CDF[JohnsonDistribution["SL", gamma, delta, mu, sigma], x]"#
      )
      .unwrap(),
      "Piecewise[{{Erfc[(-gamma - delta*Log[(-mu + x)/sigma])/Sqrt[2]]/2, Inequality[mu, Less, x, LessEqual, mu + sigma]}, {(1 + Erf[(gamma + delta*Log[(-mu + x)/sigma])/Sqrt[2]])/2, x > mu + sigma}}, 0]"
    );
  }

  #[test]
  fn cdf_sl_numeric() {
    assert_eq!(
      interpret(r#"CDF[JohnsonDistribution["SL", 0, 1, 0, 1], 1]"#).unwrap(),
      "1/2"
    );
  }

  #[test]
  fn cdf_sb_symbolic() {
    assert_eq!(
      interpret(
        r#"CDF[JohnsonDistribution["SB", gamma, delta, mu, sigma], x]"#
      )
      .unwrap(),
      "Piecewise[{{Erfc[(-gamma - delta*Log[(-mu + x)/(mu + sigma - x)])/Sqrt[2]]/2, mu < x < mu + sigma/2}, {(1 + Erf[(gamma + delta*Log[(-mu + x)/(mu + sigma - x)])/Sqrt[2]])/2, Inequality[mu + sigma/2, LessEqual, x, Less, mu + sigma]}, {1, x >= mu + sigma}}, 0]"
    );
  }

  #[test]
  fn cdf_sb_numeric() {
    assert_eq!(
      interpret(r#"CDF[JohnsonDistribution["SB", 0, 1, 0, 1], 1/2]"#).unwrap(),
      "1/2"
    );
  }

  // --- Mean tests ---

  #[test]
  fn mean_sn_symbolic() {
    assert_eq!(
      interpret(r#"Mean[JohnsonDistribution["SN", gamma, delta, mu, sigma]]"#)
        .unwrap(),
      "(delta*mu - gamma*sigma)/delta"
    );
  }

  #[test]
  fn mean_sn_numeric() {
    assert_eq!(
      interpret(r#"Mean[JohnsonDistribution["SN", 1, 2, 3, 4]]"#).unwrap(),
      "1"
    );
  }

  #[test]
  fn mean_su_symbolic() {
    assert_eq!(
      interpret(r#"Mean[JohnsonDistribution["SU", gamma, delta, mu, sigma]]"#)
        .unwrap(),
      "mu - E^(1/(2*delta^2))*sigma*Sinh[gamma/delta]"
    );
  }

  #[test]
  fn mean_su_numeric() {
    assert_eq!(
      interpret(r#"Mean[JohnsonDistribution["SU", 0, 1, 0, 1]]"#).unwrap(),
      "0"
    );
  }

  #[test]
  fn mean_sl_symbolic() {
    assert_eq!(
      interpret(r#"Mean[JohnsonDistribution["SL", gamma, delta, mu, sigma]]"#)
        .unwrap(),
      "mu + E^((1 - 2*delta*gamma)/(2*delta^2))*sigma"
    );
  }

  #[test]
  fn mean_sl_numeric() {
    assert_eq!(
      interpret(r#"Mean[JohnsonDistribution["SL", 0, 1, 0, 1]]"#).unwrap(),
      "Sqrt[E]"
    );
  }

  #[test]
  fn mean_sb_unevaluated() {
    assert_eq!(
      interpret(r#"Mean[JohnsonDistribution["SB", 0, 1, 0, 1]]"#).unwrap(),
      "Mean[JohnsonDistribution[SB, 0, 1, 0, 1]]"
    );
  }

  // --- Variance tests ---

  #[test]
  fn variance_sn_symbolic() {
    assert_eq!(
      interpret(
        r#"Variance[JohnsonDistribution["SN", gamma, delta, mu, sigma]]"#
      )
      .unwrap(),
      "sigma^2/delta^2"
    );
  }

  #[test]
  fn variance_sn_numeric() {
    assert_eq!(
      interpret(r#"Variance[JohnsonDistribution["SN", 1, 2, 3, 4]]"#).unwrap(),
      "4"
    );
  }

  #[test]
  fn variance_su_symbolic() {
    assert_eq!(
      interpret(
        r#"Variance[JohnsonDistribution["SU", gamma, delta, mu, sigma]]"#
      )
      .unwrap(),
      "((-1 + E^delta^(-2))*sigma^2*(1 + E^delta^(-2)*Cosh[(2*gamma)/delta]))/2"
    );
  }

  #[test]
  fn variance_su_numeric() {
    assert_eq!(
      interpret(r#"Variance[JohnsonDistribution["SU", 1, 2, 3, 4]]"#).unwrap(),
      "8*(-1 + E^(1/4))*(1 + E^(1/4)*Cosh[1])"
    );
  }

  #[test]
  fn variance_sl_symbolic() {
    assert_eq!(
      interpret(
        r#"Variance[JohnsonDistribution["SL", gamma, delta, mu, sigma]]"#
      )
      .unwrap(),
      "E^((1 - 2*delta*gamma)/delta^2)*(-1 + E^delta^(-2))*sigma^2"
    );
  }

  #[test]
  fn variance_sl_numeric() {
    assert_eq!(
      interpret(r#"Variance[JohnsonDistribution["SL", 0, 1, 0, 1]]"#).unwrap(),
      "(-1 + E)*E"
    );
  }

  #[test]
  fn variance_sb_unevaluated() {
    assert_eq!(
      interpret(r#"Variance[JohnsonDistribution["SB", 0, 1, 0, 1]]"#).unwrap(),
      "Variance[JohnsonDistribution[SB, 0, 1, 0, 1]]"
    );
  }
}

mod standardize {
  use super::*;

  #[test]
  fn standardize_middle_element_is_zero() {
    // Middle element of odd-length symmetric range is always 0
    assert_eq!(interpret("Standardize[{1, 2, 3, 4, 5}][[3]]").unwrap(), "0");
  }

  #[test]
  fn standardize_preserves_length() {
    assert_eq!(
      interpret("Length[Standardize[{1, 2, 3, 4, 5}]]").unwrap(),
      "5"
    );
  }

  #[test]
  fn standardize_two_elements() {
    // Standardize[{0, 2}]: mean=1, sd=Sqrt[2]
    // (0-1)/Sqrt[2] and (2-1)/Sqrt[2]
    assert_eq!(
      interpret("Standardize[{0, 2}][[1]] + Standardize[{0, 2}][[2]] // Chop")
        .unwrap(),
      "0"
    );
  }

  #[test]
  fn standardize_exact_symbolic() {
    // Integer input must stay exact/symbolic (matches Wolfram Language).
    assert_eq!(
      interpret("Standardize[{1, 2, 3, 4, 5}]").unwrap(),
      "{-2*Sqrt[2/5], -Sqrt[2/5], 0, Sqrt[2/5], 2*Sqrt[2/5]}"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn quantile_1() {
    assert_case(r#"Quantile[Range[11], 1/3]"#, r#"4"#);
  }
  #[test]
  fn quantile_2() {
    assert_case(
      r#"Quantile[Range[11], 1/3]; Quantile[Range[16], 1/4]"#,
      r#"4"#,
    );
  }
  #[test]
  fn quantile_3() {
    assert_case(
      r#"Quantile[Range[11], 1/3]; Quantile[Range[16], 1/4]; Quantile[{1, 2, 3, 4, 5, 6, 7}, {1/4, 3/4}]"#,
      r#"{2, 6}"#,
    );
  }
  #[test]
  fn quartiles() {
    assert_case(r#"Quartiles[Range[25]]"#, r#"{27 / 4, 13, 77 / 4}"#);
  }
  #[test]
  fn mean_1() {
    assert_case(r#"Mean[{26, 64, 36}]"#, r#"42"#);
  }
  #[test]
  fn mean_2() {
    assert_case(
      r#"Mean[{26, 64, 36}]; Mean[{1, 1, 2, 3, 5, 8}]"#,
      r#"10 / 3"#,
    );
  }
  #[test]
  fn mean_3() {
    assert_case(
      r#"Mean[{26, 64, 36}]; Mean[{1, 1, 2, 3, 5, 8}]; Mean[{a, b}]"#,
      r#"(a + b) / 2"#,
    );
  }
  #[test]
  fn median_1() {
    assert_case(r#"Median[{26, 64, 36}]"#, r#"36"#);
  }
  #[test]
  fn median_2() {
    assert_case(
      r#"Median[{26, 64, 36}]; Median[{-11, 38, 501, 1183}]"#,
      r#"539 / 2"#,
    );
  }
  #[test]
  fn median_3() {
    assert_case(
      r#"Median[{26, 64, 36}]; Median[{-11, 38, 501, 1183}]; Median[{{100, 1, 10, 50}, {-1, 1, -2, 2}}]"#,
      r#"{99 / 2, 1, 4, 26}"#,
    );
  }
  #[test]
  fn correlation() {
    assert_case(
      r#"Correlation[{10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5}, {8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68}]"#,
      r#"0.81642051634484"#,
    );
  }
  #[test]
  fn covariance() {
    assert_case(
      r#"Covariance[{0.2, 0.3, 0.1}, {0.3, 0.3, -0.2}]"#,
      r#"0.025"#,
    );
  }
  #[test]
  fn standard_deviation_1() {
    assert_case(r#"StandardDeviation[{1, 2, 3}]"#, r#"1"#);
  }
  #[test]
  fn standard_deviation_2() {
    assert_case(
      r#"StandardDeviation[{1, 2, 3}]; StandardDeviation[{7, -5, 101, 100}]"#,
      r#"Sqrt[13297] / 2"#,
    );
  }
  #[test]
  fn standard_deviation_3() {
    assert_case(
      r#"StandardDeviation[{1, 2, 3}]; StandardDeviation[{7, -5, 101, 100}]; StandardDeviation[{a, a}]"#,
      r#"0"#,
    );
  }
  #[test]
  fn standard_deviation_4() {
    assert_case(
      r#"StandardDeviation[{1, 2, 3}]; StandardDeviation[{7, -5, 101, 100}]; StandardDeviation[{a, a}]; StandardDeviation[{{1, 10}, {-1, 20}}]"#,
      r#"{Sqrt[2], 5*Sqrt[2]}"#,
    );
  }
  #[test]
  fn variance_1() {
    assert_case(r#"Variance[{1, 2, 3}]"#, r#"1"#);
  }
  #[test]
  fn variance_2() {
    assert_case(
      r#"Variance[{1, 2, 3}]; Variance[{7, -5, 101, 3}]"#,
      r#"7475 / 3"#,
    );
  }
  #[test]
  fn variance_3() {
    assert_case(
      r#"Variance[{1, 2, 3}]; Variance[{7, -5, 101, 3}]; Variance[{1 + 2I, 3 - 10I}]"#,
      r#"74"#,
    );
  }
  #[test]
  fn variance_4() {
    assert_case(
      r#"Variance[{1, 2, 3}]; Variance[{7, -5, 101, 3}]; Variance[{1 + 2I, 3 - 10I}]; Variance[{a, a}]"#,
      r#"0"#,
    );
  }
  #[test]
  fn variance_5() {
    assert_case(
      r#"Variance[{1, 2, 3}]; Variance[{7, -5, 101, 3}]; Variance[{1 + 2I, 3 - 10I}]; Variance[{a, a}]; Variance[{{1, 3, 5}, {4, 10, 100}}]"#,
      r#"{9 / 2, 49 / 2, 9025 / 2}"#,
    );
  }
  #[test]
  fn kurtosis() {
    assert_case(
      r#"Kurtosis[{1.1, 1.2, 1.4, 2.1, 2.4}]"#,
      r#"1.4209750290831373"#,
    );
  }
  #[test]
  fn skewness() {
    assert_case(
      r#"Skewness[{1.1, 1.2, 1.4, 2.1, 2.4}]"#,
      r#"0.4070412816074878"#,
    );
  }
  #[test]
  fn central_moment() {
    assert_case(
      r#"CentralMoment[{1.1, 1.2, 1.4, 2.1, 2.4}, 4]"#,
      r#"0.10084511999999998"#,
    );
  }
  #[test]
  fn design_matrix_1() {
    assert_case(
      r#"DesignMatrix[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]"#,
      r#"{{1, 2}, {1, 3}, {1, 5}, {1, 7}}"#,
    );
  }
  #[test]
  fn design_matrix_2() {
    assert_case(
      r#"DesignMatrix[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; DesignMatrix[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, f[x], x]"#,
      r#"{{1, f[2]}, {1, f[3]}, {1, f[5]}, {1, f[7]}}"#,
    );
  }
  #[test]
  fn m_1() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]"#,
      r#"{1, x}"#,
    );
  }
  #[test]
  fn m_2() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]"#,
      r#"0.18644067796610153 + 0.7796610169491526*x"#,
    );
  }
  #[test]
  fn m_3() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]"#,
      r#"{0.18644067796610153, 0.7796610169491526}"#,
    );
  }
  #[test]
  fn m_4() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]; m["DesignMatrix"]"#,
      r#"{{1., 2.}, {1., 3.}, {1., 5.}, {1., 7.}}"#,
    );
  }
  #[test]
  fn m_5() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]; m["DesignMatrix"]; m["Function"]"#,
      r#"0.18644067796610153 + 0.7796610169491526*#1 & "#,
    );
  }
  #[test]
  fn m_6() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]; m["DesignMatrix"]; m["Function"]; m["Response"]"#,
      r#"{1, 4, 3, 6}"#,
    );
  }
  #[test]
  fn m_7() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]; m["DesignMatrix"]; m["Function"]; m["Response"]; m["FitResiduals"]"#,
      r#"{-0.7457627118644066, 1.4745762711864407, -1.0847457627118642, 0.35593220338983045}"#,
    );
  }
  #[test]
  fn m_8() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]; m["DesignMatrix"]; m["Function"]; m["Response"]; m["FitResiduals"]; m = LinearModelFit[{{2, 2, 1}, {3, 2, 4}, {5, 6, 3}, {7, 9, 6}}, {Sin[x], Cos[y]}, {x, y}]; m["BasisFunctions"]"#,
      r#"{1, Sin[x], Cos[y]}"#,
    );
  }
  #[test]
  fn m_9() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]; m["DesignMatrix"]; m["Function"]; m["Response"]; m["FitResiduals"]; m = LinearModelFit[{{2, 2, 1}, {3, 2, 4}, {5, 6, 3}, {7, 9, 6}}, {Sin[x], Cos[y]}, {x, y}]; m["BasisFunctions"]; m["Function"]"#,
      r#"3.330769555225489 - 5.6522127921995375*Cos[#2] - 5.010415400981359*Sin[#1] & "#,
    );
  }
  #[test]
  fn m_10() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]; m["DesignMatrix"]; m["Function"]; m["Response"]; m["FitResiduals"]; m = LinearModelFit[{{2, 2, 1}, {3, 2, 4}, {5, 6, 3}, {7, 9, 6}}, {Sin[x], Cos[y]}, {x, y}]; m["BasisFunctions"]; m["Function"]; m = LinearModelFit[{{{1, 4}, {1, 5}, {1, 7}}, {1, 2, 3}}]; m["BasisFunctions"]"#,
      r#"{#1, #2}"#,
    );
  }
  #[test]
  fn m_11() {
    assert_case(
      r#"m = LinearModelFit[{{2, 1}, {3, 4}, {5, 3}, {7, 6}}, x, x]; m["BasisFunctions"]; m["BestFit"]; m["BestFitParameters"]; m["DesignMatrix"]; m["Function"]; m["Response"]; m["FitResiduals"]; m = LinearModelFit[{{2, 2, 1}, {3, 2, 4}, {5, 6, 3}, {7, 9, 6}}, {Sin[x], Cos[y]}, {x, y}]; m["BasisFunctions"]; m["Function"]; m = LinearModelFit[{{{1, 4}, {1, 5}, {1, 7}}, {1, 2, 3}}]; m["BasisFunctions"]; m["FitResiduals"]"#,
      r#"{-0.1428571428571428, 0.2142857142857144, -0.07142857142857162}"#,
    );
  }
  #[test]
  fn clustering_components() {
    assert_case(
      r#"ClusteringComponents[{1, 2, 3, 1, 2, 10, 100}]"#,
      r#"{1, 1, 1, 1, 1, 1, 2}"#,
    );
  }

  // ─── ClusteringComponents[list, n] (2-arg form) ───────────────────
  //
  // Audit case: `ClusteringComponents[{1, 2, 3, 7, 8}, 2]` should
  // split {1, 2, 3} from {7, 8}. wolframscript labels them
  // `{2, 2, 2, 1, 1}` (small-values cluster gets label 2). Woxi uses
  // the largest-gap split with cluster 1 always covering the smaller
  // values, so labels come out `{1, 1, 1, 2, 2}` — same partition,
  // mirrored numbering.
  #[test]
  fn clustering_components_two_clusters_audit_case() {
    assert_case(
      r#"ClusteringComponents[{1, 2, 3, 7, 8}, 2]"#,
      r#"{1, 1, 1, 2, 2}"#,
    );
  }

  #[test]
  fn clustering_components_two_clusters_other_gap() {
    assert_case(
      r#"ClusteringComponents[{10, 11, 12, 50, 51}, 2]"#,
      r#"{1, 1, 1, 2, 2}"#,
    );
  }

  // n = 1 ⇒ everything in the single cluster.
  #[test]
  fn clustering_components_one_cluster() {
    assert_case(
      r#"ClusteringComponents[{1, 2, 3, 7, 8}, 1]"#,
      r#"{1, 1, 1, 1, 1}"#,
    );
  }

  // n = 3 ⇒ three clusters via the three largest gaps.
  #[test]
  fn clustering_components_three_clusters() {
    assert_case(
      r#"ClusteringComponents[{1, 2, 10, 11, 50, 51}, 3]"#,
      r#"{1, 1, 2, 2, 3, 3}"#,
    );
  }
  #[test]
  fn with_1() {
    // FindClusters auto-selects the cluster count via a heuristic
    // (Wolfram uses a gap statistic; Mathics groups more coarsely).
    // Both are documented as valid: the mathics docstring lists
    // \`{{1, 2, 20, 10, 11, 19}, {40, 42}}\` for this input, and Woxi
    // matches mathics. Verify the documented contract: the result is a
    // List of nonempty Lists whose flattened concatenation is a
    // permutation of the input.
    assert_case(
      r#"With[{c = FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]}, Head[c] === List && Length[c] >= 1 && AllTrue[c, Head[#] === List && Length[#] >= 1 &] && Sort[Flatten[c]] === Sort[{1, 2, 20, 10, 11, 40, 19, 42}]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn find_clusters_1() {
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]"#,
      r#"{{25, 17, 20}, {100}}"#,
    );
  }
  #[test]
  fn with_2() {
    // Same FindClusters heuristic-disagreement family as case 1698:
    // wolframscript and mathics group differently for this input
    // (Mathics docs show `{{3, 6, 1, 5, -10, 2}, {100}, {20, 25, 17}}`;
    // Woxi follows mathics). Verify the documented contract: a list of
    // nonempty subsets whose flattened union is a permutation of the
    // input.
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; With[{c = FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]}, Head[c] === List && Length[c] >= 1 && AllTrue[c, Head[#] === List && Length[#] >= 1 &] && Sort[Flatten[c]] === Sort[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn with_3() {
    // Same FindClusters heuristic-disagreement family as cases
    // 1698/1700. The last expression's expected
    // `{{1, 2}, {20, 21}, {10, 11}}` is Wolfram's 3-cluster grouping;
    // Woxi follows mathics and groups as `{{1, 2, 10, 11}, {20, 21}}`
    // (2 clusters). Verify the partition-of-input contract.
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]; With[{c = FindClusters[{1, 2, 10, 11, 20, 21}]}, Head[c] === List && Length[c] >= 1 && AllTrue[c, Head[#] === List && Length[#] >= 1 &] && Sort[Flatten[c]] === Sort[{1, 2, 10, 11, 20, 21}]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn find_clusters_2() {
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]; FindClusters[{1, 2, 10, 11, 20, 21}]; FindClusters[{1, 2, 10, 11, 20, 21}, 2]"#,
      r#"{{1, 2, 10, 11}, {20, 21}}"#,
    );
  }
  #[test]
  fn find_clusters_3() {
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]; FindClusters[{1, 2, 10, 11, 20, 21}]; FindClusters[{1, 2, 10, 11, 20, 21}, 2]; FindClusters[{1 -> a, 2 -> b, 10 -> c}]"#,
      r#"{{c}, {a, b}}"#,
    );
  }
  #[test]
  fn find_clusters_4() {
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]; FindClusters[{1, 2, 10, 11, 20, 21}]; FindClusters[{1, 2, 10, 11, 20, 21}, 2]; FindClusters[{1 -> a, 2 -> b, 10 -> c}]; FindClusters[{1, 2, 5} -> {a, b, c}]"#,
      r#"{{c}, {a, b}}"#,
    );
  }
  #[test]
  fn with_4() {
    // Same FindClusters heuristic-disagreement family as cases
    // 1698/1700/1701. The last expression also exercises the
    // `Method -> "Agglomerate"` option — Woxi now accepts it (along
    // with other recognised options like CriterionFunction,
    // PerformanceGoal, WorkingPrecision) and silently ignores them
    // since the default algorithm is used. Verify the partition-of-
    // input contract.
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]; FindClusters[{1, 2, 10, 11, 20, 21}]; FindClusters[{1, 2, 10, 11, 20, 21}, 2]; FindClusters[{1 -> a, 2 -> b, 10 -> c}]; FindClusters[{1, 2, 5} -> {a, b, c}]; With[{c = FindClusters[{1, 2, 3, 1, 2, 10, 100}, Method -> "Agglomerate"]}, Head[c] === List && Length[c] >= 1 && AllTrue[c, Head[#] === List && Length[#] >= 1 &] && Sort[Flatten[c]] === Sort[{1, 2, 3, 1, 2, 10, 100}]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn with_5() {
    // Same FindClusters heuristic-disagreement family as cases
    // 1698/1700/1701/1705. Wolfram returns 3 clusters
    // (`{{17, 18}, {10}, {1, 2, 3}}`); Woxi (mathics-derived) returns
    // 2 (`{{1, 2, 3}, {10, 17, 18}}`). Verify the partition-of-input
    // contract.
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]; FindClusters[{1, 2, 10, 11, 20, 21}]; FindClusters[{1, 2, 10, 11, 20, 21}, 2]; FindClusters[{1 -> a, 2 -> b, 10 -> c}]; FindClusters[{1, 2, 5} -> {a, b, c}]; FindClusters[{1, 2, 3, 1, 2, 10, 100}, Method -> "Agglomerate"]; With[{c = FindClusters[{1, 2, 3, 10, 17, 18}, Method -> "Agglomerate"]}, Head[c] === List && Length[c] >= 1 && AllTrue[c, Head[#] === List && Length[#] >= 1 &] && Sort[Flatten[c]] === Sort[{1, 2, 3, 10, 17, 18}]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn find_clusters_5() {
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]; FindClusters[{1, 2, 10, 11, 20, 21}]; FindClusters[{1, 2, 10, 11, 20, 21}, 2]; FindClusters[{1 -> a, 2 -> b, 10 -> c}]; FindClusters[{1, 2, 5} -> {a, b, c}]; FindClusters[{1, 2, 3, 1, 2, 10, 100}, Method -> "Agglomerate"]; FindClusters[{1, 2, 3, 10, 17, 18}, Method -> "Agglomerate"]; FindClusters[{{1}, {5, 6}, {7}, {2, 4}}, DistanceFunction -> (Abs[Length[#1] - Length[#2]]&)]"#,
      r#"{{{5, 6}, {2, 4}}, {{1}, {7}}}"#,
    );
  }
  #[test]
  fn with_6() {
    // Same FindClusters family. Last expression clusters strings via
    // EditDistance (Levenshtein) into k=3 groups. Woxi now handles the
    // string-input case via single-linkage agglomerative clustering;
    // its grouping is `{{meep, deep, weep, keep}, {heap, leap},
    // {sheep}}` versus wolframscript's
    // `{{meep, heap, sheep, leap, keep}, {deep}, {weep}}` — both are
    // valid 3-clusterings of the same input. Verify the partition-of-
    // input contract with exactly k = 3 clusters.
    assert_case(
      r#"FindClusters[{1, 2, 20, 10, 11, 40, 19, 42}]; FindClusters[{25, 100, 17, 20}]; FindClusters[{3, 6, 1, 100, 20, 5, 25, 17, -10, 2}]; FindClusters[{1, 2, 10, 11, 20, 21}]; FindClusters[{1, 2, 10, 11, 20, 21}, 2]; FindClusters[{1 -> a, 2 -> b, 10 -> c}]; FindClusters[{1, 2, 5} -> {a, b, c}]; FindClusters[{1, 2, 3, 1, 2, 10, 100}, Method -> "Agglomerate"]; FindClusters[{1, 2, 3, 10, 17, 18}, Method -> "Agglomerate"]; FindClusters[{{1}, {5, 6}, {7}, {2, 4}}, DistanceFunction -> (Abs[Length[#1] - Length[#2]]&)]; With[{c = FindClusters[{"meep", "heap", "deep", "weep", "sheep", "leap", "keep"}, 3]}, Head[c] === List && Length[c] === 3 && AllTrue[c, Head[#] === List && Length[#] >= 1 &] && Sort[Flatten[c]] === Sort[{"meep", "heap", "deep", "weep", "sheep", "leap", "keep"}]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn total_1() {
    assert_case(r#"Total[{1, 2, 3}]"#, r#"6"#);
  }
  #[test]
  fn total_2() {
    assert_case(
      r#"Total[{1, 2, 3}]; Total[{{1, 2, 3}, {4, 5, 6}, {7, 8 ,9}}]"#,
      r#"{12, 15, 18}"#,
    );
  }
  #[test]
  fn total_3() {
    assert_case(
      r#"Total[{1, 2, 3}]; Total[{{1, 2, 3}, {4, 5, 6}, {7, 8 ,9}}]; Total[{{1, 2, 3}, {4, 5, 6}, {7, 8 ,9}}, 2]"#,
      r#"45"#,
    );
  }
  #[test]
  fn total_4() {
    assert_case(
      r#"Total[{1, 2, 3}]; Total[{{1, 2, 3}, {4, 5, 6}, {7, 8 ,9}}]; Total[{{1, 2, 3}, {4, 5, 6}, {7, 8 ,9}}, 2]; Total[{{1, 2, 3}, {4, 5, 6}, {7, 8 ,9}}, {2}]"#,
      r#"{6, 15, 24}"#,
    );
  }
  #[test]
  fn tally_1() {
    assert_case(r#"Tally[{a, b, c, b, a}]"#, r#"{{a, 2}, {b, 2}, {c, 1}}"#);
  }
  #[test]
  fn tally_2() {
    assert_case(
      r#"Tally[{a, b, c, b, a}]; Tally[{b, b, a, a, a, d, d, d, d, c}]"#,
      r#"{{b, 2}, {a, 3}, {d, 4}, {c, 1}}"#,
    );
  }
  #[test]
  fn tally_association_values() {
    // On an association, Tally counts the values.
    assert_case(
      r#"Tally[<|a -> 1, b -> 1, c -> 2|>]"#,
      r#"{{1, 2}, {2, 1}}"#,
    );
  }
  #[test]
  fn dice_dissimilarity() {
    assert_case(
      r#"DiceDissimilarity[{1, 0, 1, 1, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 1}]"#,
      r#"1 / 2"#,
    );
  }
  #[test]
  fn dice_dissimilarity_boolean() {
    // Boolean inputs collapse to the same dissimilarity as 1/0 inputs.
    assert_case(
      r#"DiceDissimilarity[{True, False, True}, {True, True, False}]"#,
      r#"1 / 2"#,
    );
  }
  #[test]
  fn dice_dissimilarity_identical() {
    // Identical Boolean vectors -> dissimilarity 0.
    assert_case(
      r#"DiceDissimilarity[{True, True, True}, {True, True, True}]"#,
      r#"0"#,
    );
  }
  #[test]
  fn jaccard_dissimilarity() {
    assert_case(
      r#"JaccardDissimilarity[{1, 0, 1, 1, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 1}]"#,
      r#"2 / 3"#,
    );
  }
  #[test]
  fn matching_dissimilarity() {
    assert_case(
      r#"MatchingDissimilarity[{1, 0, 1, 1, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 1}]"#,
      r#"4 / 7"#,
    );
  }
  #[test]
  fn rogers_tanimoto_dissimilarity() {
    assert_case(
      r#"RogersTanimotoDissimilarity[{1, 0, 1, 1, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 1}]"#,
      r#"8 / 11"#,
    );
  }
  #[test]
  fn russell_rao_dissimilarity() {
    assert_case(
      r#"RussellRaoDissimilarity[{1, 0, 1, 1, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 1}]"#,
      r#"5 / 7"#,
    );
  }
  #[test]
  fn sokal_sneath_dissimilarity() {
    assert_case(
      r#"SokalSneathDissimilarity[{1, 0, 1, 1, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 1}]"#,
      r#"4 / 5"#,
    );
  }
  #[test]
  fn yule_dissimilarity() {
    assert_case(
      r#"YuleDissimilarity[{1, 0, 1, 1, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 1}]"#,
      r#"6 / 5"#,
    );
  }
  #[test]
  fn nest_while_1() {
    assert_case(
      r#"NestWhile[#/2&, 10000, IntegerQ]; NestWhile[Total[IntegerDigits[#]^3] &, 5, UnsameQ, All]"#,
      r#"371"#,
    );
  }
  #[test]
  fn nest_while_2() {
    assert_case(
      r#"NestWhile[#/2&, 10000, IntegerQ]; NestWhile[Total[IntegerDigits[#]^3] &, 5, UnsameQ, All]"#,
      r#"371"#,
    );
  }
  #[test]
  fn nest_while_3() {
    assert_case(
      r#"NestWhile[#/2&, 10000, IntegerQ]; NestWhile[Total[IntegerDigits[#]^3] &, 5, UnsameQ, All]; NestWhile[Total[IntegerDigits[#]^3] &, 6, UnsameQ, All]"#,
      r#"153"#,
    );
  }
}

mod arma_covariance_function {
  use super::*;

  #[test]
  fn ar1_zero_mean() {
    // CovarianceFunction[ARMAProcess[{a}, {}, σ²], s, t]
    //   = (a^Abs[s-t] σ²) / (1 - a²)
    assert_eq!(
      interpret("CovarianceFunction[ARMAProcess[{a}, {}, s2], s, t]").unwrap(),
      "(a^Abs[s - t]*s2)/(1 - a^2)"
    );
  }

  #[test]
  fn arma11_zero_mean() {
    // ARMA(1,1): closed-form Piecewise, matching wolframscript.
    assert_eq!(
      interpret("CovarianceFunction[ARMAProcess[{a}, {b}, s2], s, t]").unwrap(),
      "Piecewise[{{-((a^(-1 + Abs[s - t])*(a + b + a^2*b + a*b^2)*s2)/((-1 + a)*(1 + a))), Abs[s - t] > 0}}, -((b*s2)/a) - ((a + b + a^2*b + a*b^2)*s2)/((-1 + a)*a*(1 + a))]"
    );
  }

  #[test]
  fn arma11_with_constant() {
    // The audit case: a non-zero constant c does not affect covariance.
    assert_eq!(
      interpret("CovarianceFunction[ARMAProcess[c, {a}, {b}, s2], s, t]")
        .unwrap(),
      "Piecewise[{{-((a^(-1 + Abs[s - t])*(a + b + a^2*b + a*b^2)*s2)/((-1 + a)*(1 + a))), Abs[s - t] > 0}}, -((b*s2)/a) - ((a + b + a^2*b + a*b^2)*s2)/((-1 + a)*a*(1 + a))]"
    );
  }
}

mod data_covariance_function {
  use super::*;

  // Sample autocovariance of a numeric series at lag h:
  //   gamma(h) = (1/n) Sum_{t=1}^{n-|h|} (x_t - xbar)(x_{t+|h|} - xbar).
  #[test]
  fn integer_lags() {
    assert_eq!(
      interpret("CovarianceFunction[{2, 3, 4, 3}, 0]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("CovarianceFunction[{2, 3, 4, 3}, 1]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("CovarianceFunction[{2, 3, 4, 3}, 2]").unwrap(),
      "-1/4"
    );
    assert_eq!(
      interpret("CovarianceFunction[{2, 3, 4, 3}, 3]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("CovarianceFunction[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "-1/5"
    );
  }

  // The autocovariance is symmetric, so a negative lag equals its magnitude.
  #[test]
  fn negative_lag_symmetric() {
    assert_eq!(
      interpret("CovarianceFunction[{1, 2, 3, 4, 5}, -2]").unwrap(),
      "-1/5"
    );
  }

  // Machine-real data yields a machine-real result.
  #[test]
  fn real_data() {
    assert_eq!(
      interpret("CovarianceFunction[{1., 2., 3., 4., 5.}, 1]").unwrap(),
      "0.8"
    );
  }

  // A lag whose magnitude is not less than the length stays unevaluated.
  #[test]
  fn lag_out_of_range_unevaluated() {
    assert_eq!(
      interpret("CovarianceFunction[{2, 3, 4, 3}, 4]").unwrap(),
      "CovarianceFunction[{2, 3, 4, 3}, 4]"
    );
    assert_eq!(
      interpret("CovarianceFunction[{1, 2, 3, 4, 5}, -5]").unwrap(),
      "CovarianceFunction[{1, 2, 3, 4, 5}, -5]"
    );
  }
}

mod characteristic_function {
  use super::*;

  #[test]
  fn normal() {
    assert_eq!(
      interpret("CharacteristicFunction[NormalDistribution[], t]").unwrap(),
      "E^(-1/2*t^2)"
    );
    assert_eq!(
      interpret("CharacteristicFunction[NormalDistribution[m, s], t]").unwrap(),
      "E^(I*m*t - (s^2*t^2)/2)"
    );
    // Numeric parameters fold to the standard normal form
    assert_eq!(
      interpret("CharacteristicFunction[NormalDistribution[0, 1], t]").unwrap(),
      "E^(-1/2*t^2)"
    );
  }

  #[test]
  fn exponential_and_gamma() {
    assert_eq!(
      interpret("CharacteristicFunction[ExponentialDistribution[a], t]")
        .unwrap(),
      "a/(a - I*t)"
    );
    assert_eq!(
      interpret("CharacteristicFunction[GammaDistribution[a, b], t]").unwrap(),
      "(1 - I*b*t)^(-a)"
    );
  }

  #[test]
  fn discrete_distributions() {
    assert_eq!(
      interpret("CharacteristicFunction[PoissonDistribution[m], t]").unwrap(),
      "E^((-1 + E^(I*t))*m)"
    );
    assert_eq!(
      interpret("CharacteristicFunction[BernoulliDistribution[p], t]").unwrap(),
      "1 - p + E^(I*t)*p"
    );
    assert_eq!(
      interpret("CharacteristicFunction[BinomialDistribution[n, p], t]")
        .unwrap(),
      "(1 - p + E^(I*t)*p)^n"
    );
    assert_eq!(
      interpret("CharacteristicFunction[GeometricDistribution[p], t]").unwrap(),
      "p/(1 - E^(I*t)*(1 - p))"
    );
  }

  #[test]
  fn uniform() {
    assert_eq!(
      interpret("CharacteristicFunction[UniformDistribution[], t]").unwrap(),
      "(-I*(-1 + E^(I*t)))/t"
    );
    assert_eq!(
      interpret("CharacteristicFunction[UniformDistribution[{a, b}], t]")
        .unwrap(),
      "(-I*(-E^(I*a*t) + E^(I*b*t)))/((-a + b)*t)"
    );
  }

  #[test]
  fn chisquare_and_beta() {
    // ChiSquare: (1 - 2 I t)^(-k/2), also folding for a numeric k.
    assert_eq!(
      interpret("CharacteristicFunction[ChiSquareDistribution[k], t]").unwrap(),
      "(1 - (2*I)*t)^(-1/2*k)"
    );
    assert_eq!(
      interpret("CharacteristicFunction[ChiSquareDistribution[4], t]").unwrap(),
      "(1 - (2*I)*t)^(-2)"
    );
    // Beta: the confluent hypergeometric function evaluated at I t.
    assert_eq!(
      interpret("CharacteristicFunction[BetaDistribution[a, b], t]").unwrap(),
      "Hypergeometric1F1[a, a + b, I*t]"
    );
  }

  #[test]
  fn numeric_argument() {
    assert_eq!(
      interpret("CharacteristicFunction[NormalDistribution[], 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("CharacteristicFunction[x, t]").unwrap(),
      "CharacteristicFunction[x, t]"
    );
  }
}

mod moment_generating_function {
  use super::*;

  #[test]
  fn normal() {
    assert_eq!(
      interpret("MomentGeneratingFunction[NormalDistribution[], t]").unwrap(),
      "E^(t^2/2)"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[NormalDistribution[m, s], t]")
        .unwrap(),
      "E^(m*t + (s^2*t^2)/2)"
    );
  }

  #[test]
  fn exponential_and_gamma() {
    assert_eq!(
      interpret("MomentGeneratingFunction[ExponentialDistribution[a], t]")
        .unwrap(),
      "a/(a - t)"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[GammaDistribution[a, b], t]")
        .unwrap(),
      "(1 - b*t)^(-a)"
    );
  }

  #[test]
  fn poisson_bernoulli_binomial() {
    assert_eq!(
      interpret("MomentGeneratingFunction[PoissonDistribution[m], t]").unwrap(),
      "E^((-1 + E^t)*m)"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[BernoulliDistribution[p], t]")
        .unwrap(),
      "1 - p + E^t*p"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[BinomialDistribution[n, p], t]")
        .unwrap(),
      "(1 + (-1 + E^t)*p)^n"
    );
  }

  #[test]
  fn geometric_and_uniform() {
    assert_eq!(
      interpret("MomentGeneratingFunction[GeometricDistribution[p], t]")
        .unwrap(),
      "p/(1 - E^t*(1 - p))"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[UniformDistribution[], t]").unwrap(),
      "(-1 + E^t)/t"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[UniformDistribution[{a, b}], t]")
        .unwrap(),
      "(-E^(a*t) + E^(b*t))/((-a + b)*t)"
    );
  }

  #[test]
  fn chisquare_and_beta() {
    // ChiSquare: (1 - 2 t)^(-k/2), also folding for a numeric k.
    assert_eq!(
      interpret("MomentGeneratingFunction[ChiSquareDistribution[k], t]")
        .unwrap(),
      "(1 - 2*t)^(-1/2*k)"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[ChiSquareDistribution[6], t]")
        .unwrap(),
      "(1 - 2*t)^(-3)"
    );
    // Beta: the confluent hypergeometric function.
    assert_eq!(
      interpret("MomentGeneratingFunction[BetaDistribution[a, b], t]").unwrap(),
      "Hypergeometric1F1[a, a + b, t]"
    );
  }

  #[test]
  fn no_mgf_is_indeterminate() {
    // Student-t and Cauchy have no moment-generating function.
    assert_eq!(
      interpret("MomentGeneratingFunction[StudentTDistribution[n], t]")
        .unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[CauchyDistribution[a, b], t]")
        .unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn numeric_argument_folds() {
    // The MGF at t = 0 is always 1.
    assert_eq!(
      interpret("MomentGeneratingFunction[NormalDistribution[0, 1], 0]")
        .unwrap(),
      "1"
    );
    assert_eq!(
      interpret("MomentGeneratingFunction[ExponentialDistribution[2], t]")
        .unwrap(),
      "2/(2 - t)"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("MomentGeneratingFunction[x, t]").unwrap(),
      "MomentGeneratingFunction[x, t]"
    );
  }
}

mod cumulant_generating_function {
  use super::*;

  // CGF = Log[MGF]. Where the MGF is E^X the CGF is X (Normal, Poisson); where
  // it is base^exp the CGF is exp*Log[base] (Binomial, Gamma); otherwise Log.
  #[test]
  fn normal_and_poisson() {
    assert_eq!(
      interpret("CumulantGeneratingFunction[NormalDistribution[], t]").unwrap(),
      "t^2/2"
    );
    assert_eq!(
      interpret("CumulantGeneratingFunction[NormalDistribution[m, s], t]")
        .unwrap(),
      "m*t + (s^2*t^2)/2"
    );
    assert_eq!(
      interpret("CumulantGeneratingFunction[PoissonDistribution[lambda], t]")
        .unwrap(),
      "(-1 + E^t)*lambda"
    );
  }

  #[test]
  fn log_forms() {
    assert_eq!(
      interpret("CumulantGeneratingFunction[ExponentialDistribution[a], t]")
        .unwrap(),
      "Log[a/(a - t)]"
    );
    assert_eq!(
      interpret("CumulantGeneratingFunction[BernoulliDistribution[p], t]")
        .unwrap(),
      "Log[1 - p + E^t*p]"
    );
    assert_eq!(
      interpret("CumulantGeneratingFunction[UniformDistribution[], t]")
        .unwrap(),
      "Log[(-1 + E^t)/t]"
    );
  }

  #[test]
  fn power_forms() {
    assert_eq!(
      interpret("CumulantGeneratingFunction[BinomialDistribution[n, p], t]")
        .unwrap(),
      "n*Log[1 + (-1 + E^t)*p]"
    );
    assert_eq!(
      interpret("CumulantGeneratingFunction[GammaDistribution[a, b], t]")
        .unwrap(),
      "-(a*Log[1 - b*t])"
    );
    // ChiSquare derives its CGF from the (1 - 2 t)^(-k/2) MGF.
    assert_eq!(
      interpret("CumulantGeneratingFunction[ChiSquareDistribution[k], t]")
        .unwrap(),
      "-1/2*(k*Log[1 - 2*t])"
    );
  }

  // Distributions with no MGF have no CGF either: Log[Indeterminate] folds to
  // Indeterminate rather than being left as a nested Log.
  #[test]
  fn no_cgf_is_indeterminate() {
    assert_eq!(
      interpret("CumulantGeneratingFunction[StudentTDistribution[n], t]")
        .unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("CumulantGeneratingFunction[CauchyDistribution[a, b], t]")
        .unwrap(),
      "Indeterminate"
    );
  }

  // Geometric and two-parameter Uniform use Wolfram's own canonical forms.
  #[test]
  fn special_forms() {
    assert_eq!(
      interpret("CumulantGeneratingFunction[GeometricDistribution[p], t]")
        .unwrap(),
      "-t - Log[1 - (1 - E^(-t))/p]"
    );
    assert_eq!(
      interpret("CumulantGeneratingFunction[UniformDistribution[{a, b}], t]")
        .unwrap(),
      "a*t + Log[(-1 + E^((-a + b)*t))/((-a + b)*t)]"
    );
  }

  #[test]
  fn numeric_argument_folds() {
    // CGF at t = 0 is always 0.
    assert_eq!(
      interpret("CumulantGeneratingFunction[NormalDistribution[0, 1], 0]")
        .unwrap(),
      "0"
    );
    assert_eq!(
      interpret("CumulantGeneratingFunction[ExponentialDistribution[2], t]")
        .unwrap(),
      "Log[2/(2 - t)]"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("CumulantGeneratingFunction[x, t]").unwrap(),
      "CumulantGeneratingFunction[x, t]"
    );
  }
}

mod log_likelihood {
  use super::*;

  #[test]
  fn exponential() {
    assert_eq!(
      interpret("LogLikelihood[ExponentialDistribution[a], {2, 3, 5}]")
        .unwrap(),
      "-10*a + 3*Log[a]"
    );
    assert_eq!(
      interpret("LogLikelihood[ExponentialDistribution[2], {1, 4}]").unwrap(),
      "-10 + 2*Log[2]"
    );
  }

  #[test]
  fn poisson() {
    // -n*m + Sum[x]*Log[m] - Log[x_i!] per observation
    assert_eq!(
      interpret("LogLikelihood[PoissonDistribution[m], {1, 2, 3}]").unwrap(),
      "-3*m - Log[2] - Log[6] + 6*Log[m]"
    );
    // Numeric mean combines the Log[2] terms
    assert_eq!(
      interpret("LogLikelihood[PoissonDistribution[2], {1, 2, 3}]").unwrap(),
      "-6 + 5*Log[2] - Log[6]"
    );
  }

  #[test]
  fn bernoulli() {
    assert_eq!(
      interpret("LogLikelihood[BernoulliDistribution[p], {1, 0, 1, 1}]")
        .unwrap(),
      "Log[1 - p] + 3*Log[p]"
    );
    assert_eq!(
      interpret("LogLikelihood[BernoulliDistribution[p], {1, 1}]").unwrap(),
      "2*Log[p]"
    );
    assert_eq!(
      interpret("LogLikelihood[BernoulliDistribution[1/3], {1, 0, 1, 1}]")
        .unwrap(),
      "-Log[3/2] - 3*Log[3]"
    );
  }

  #[test]
  fn normal() {
    // Grouped wolframscript form with the sum of squares expanded in m
    assert_eq!(
      interpret("LogLikelihood[NormalDistribution[m, s], {1, 2}]").unwrap(),
      "-1/2*(5 - 6*m + 2*m^2)/s^2 - 2*((Log[2] + Log[Pi])/2 + Log[s])"
    );
    assert_eq!(
      interpret("LogLikelihood[NormalDistribution[], {1, 2}]").unwrap(),
      "-5/2 - Log[2] - Log[Pi]"
    );
    // Numeric parameters fold like the standard normal
    assert_eq!(
      interpret("LogLikelihood[NormalDistribution[0, 1], {1, 2}]").unwrap(),
      "-5/2 - Log[2] - Log[Pi]"
    );
    // Real data folds to a machine number
    assert_eq!(
      interpret("LogLikelihood[NormalDistribution[0, 1], {1.5, 2.5}]").unwrap(),
      "-6.087877066409345"
    );
  }

  #[test]
  fn symbolic_data_stays_unevaluated() {
    // wolframscript wraps symbolic observations in domain Piecewise
    // conditions; Woxi leaves them unevaluated
    assert_eq!(
      interpret("LogLikelihood[ExponentialDistribution[a], {x1, x2}]").unwrap(),
      "LogLikelihood[ExponentialDistribution[a], {x1, x2}]"
    );
  }
}

mod correlation_function {
  use super::*;

  #[test]
  fn basic_lags() {
    assert_eq!(
      interpret("CorrelationFunction[{1, 2, 3, 4, 5}, 1]").unwrap(),
      "2/5"
    );
    assert_eq!(
      interpret("CorrelationFunction[{1, 2, 3, 4, 5}, 2]").unwrap(),
      "-1/10"
    );
    assert_eq!(
      interpret("CorrelationFunction[{2, 4, 3, 5, 7, 6}, 1]").unwrap(),
      "5/14"
    );
  }

  #[test]
  fn edge_lags() {
    assert_eq!(
      interpret("CorrelationFunction[{1, 2, 3, 4, 5}, 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("CorrelationFunction[{1, 2, 3, 4, 5}, 4]").unwrap(),
      "-2/5"
    );
    // Autocorrelation is symmetric in the lag sign
    assert_eq!(
      interpret("CorrelationFunction[{1, 2, 3, 4, 5}, -1]").unwrap(),
      "2/5"
    );
  }

  #[test]
  fn out_of_range_lag_stays_unevaluated() {
    // |k| >= n: CorrelationFunction::bdlag message, unevaluated
    assert_eq!(
      interpret("CorrelationFunction[{1, 2, 3, 4, 5}, 7]").unwrap(),
      "CorrelationFunction[{1, 2, 3, 4, 5}, 7]"
    );
  }

  #[test]
  fn constant_data_is_indeterminate() {
    // 0/0 without the division messages a literal 0/0 would emit
    assert_eq!(
      interpret("CorrelationFunction[{3, 3, 3}, 1]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Check[CorrelationFunction[{3, 3, 3}, 1], \"msg\"]").unwrap(),
      "Indeterminate"
    );
  }
}

mod ztest {
  use super::*;

  #[test]
  fn p_values_via_round_projections() {
    // Default: variance estimated from the data, mu0 = 0
    assert_eq!(
      interpret("Round[10^15 ZTest[{1.1, 1.9, 3.2, 0.5, 2.3, 1.8}]]").unwrap(),
      "2600383126"
    );
    // Second argument is the KNOWN VARIANCE, not the mean
    assert_eq!(
      interpret("Round[10^10 ZTest[{1.1, 1.9, 3.2, 0.5, 2.3, 1.8}, 1, 1]]")
        .unwrap(),
      "500435212"
    );
    // Automatic falls back to the sample variance
    assert_eq!(
      interpret(
        "Round[10^10 ZTest[{1.1, 1.9, 3.2, 0.5, 2.3, 1.8}, Automatic, 1]]"
      )
      .unwrap(),
      "367138564"
    );
    // Exact data numericizes
    assert_eq!(
      interpret("Round[10^10 ZTest[{1, 2, 3}]]").unwrap(),
      "5320055"
    );
  }

  #[test]
  fn test_statistic_property() {
    assert_eq!(
      interpret(
        "Round[10^10 ZTest[{1.1, 1.9, 3.2, 0.5, 2.3, 1.8}, 1, 1, \"TestStatistic\"]]"
      )
      .unwrap(),
      "19595917942"
    );
  }

  #[test]
  fn invalid_data_messages() {
    // ZTest::rctndm1 message, unevaluated
    assert_eq!(interpret("ZTest[x]").unwrap(), "ZTest[x]");
    assert_eq!(interpret("ZTest[{1, x}]").unwrap(), "ZTest[{1, x}]");
  }
}

mod fisher_ratio_test {
  use super::*;

  #[test]
  fn p_values_and_statistic() {
    // T = Total[(x - mean)^2]/sigma0^2, chi-square(n-1) two-sided
    assert_eq!(
      interpret("Round[10^10 FisherRatioTest[{1.2, 2.1, 0.5, 1.8}]]").unwrap(),
      "6354593393"
    );
    assert_eq!(
      interpret("Round[10^10 FisherRatioTest[{1.2, 2.1, 0.5, 1.8}, 2]]")
        .unwrap(),
      "2772298392"
    );
    assert_eq!(
      interpret(
        "Round[10^10 FisherRatioTest[{1.2, 2.1, 0.5, 1.8}, 1, \"TestStatistic\"]]"
      )
      .unwrap(),
      "15000000000"
    );
    // Upper-tail side of the two-sided minimum
    assert_eq!(
      interpret("Round[10^10 FisherRatioTest[{1, 2, 3, 4}]]").unwrap(),
      "3435942886"
    );
    assert_eq!(
      interpret("Round[10^10 FisherRatioTest[{0.5, 0.9, 1.3, 2.0, 0.7}, 3]]")
        .unwrap(),
      "471662125"
    );
  }

  #[test]
  fn invalid_arguments() {
    // A list second argument is not a variance: FisherRatioTest::sigmnt
    assert_eq!(
      interpret("FisherRatioTest[{1, 2, 3, 4}, {2, 4, 6, 8}]").unwrap(),
      "FisherRatioTest[{1, 2, 3, 4}, {2, 4, 6, 8}]"
    );
    // Non-vector data: FisherRatioTest::vctnln1
    assert_eq!(
      interpret("FisherRatioTest[x]").unwrap(),
      "FisherRatioTest[x]"
    );
  }
}

// Parametric 3-arg Quantile[list, q, {{a,b},{c,d}}] (Hyndman-Fan).
// All expected values verified against wolframscript.
mod quantile_parametric {
  use super::*;

  const D10: &str = "{1,2,3,4,5,6,7,8,9,10}";

  #[test]
  fn type1_inverse_cdf_default() {
    // {{0,0},{1,0}} is the 2-arg default; discontinuous (step) at integer x.
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 1/4, {{{{0,0}},{{1,0}}}}]")).unwrap(),
      "3"
    );
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 1/2, {{{{0,0}},{{1,0}}}}]")).unwrap(),
      "5"
    );
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 3/4, {{{{0,0}},{{1,0}}}}]")).unwrap(),
      "8"
    );
  }

  #[test]
  fn type7_linear_interpolation() {
    // {{0,1},{0,1}} interpolates linearly between order statistics.
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 1/4, {{{{0,1}},{{0,1}}}}]")).unwrap(),
      "11/4"
    );
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 1/2, {{{{0,1}},{{0,1}}}}]")).unwrap(),
      "11/2"
    );
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 3/4, {{{{0,1}},{{0,1}}}}]")).unwrap(),
      "33/4"
    );
  }

  #[test]
  fn other_parameter_sets() {
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 1/2, {{{{1/2,0}},{{0,1}}}}]"))
        .unwrap(),
      "11/2"
    );
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 1/4, {{{{1,-1}},{{0,1}}}}]"))
        .unwrap(),
      "13/4"
    );
    assert_eq!(
      interpret("Quantile[{10,20,30,40,50}, 1/5, {{1/2,0},{1/2,0}}]").unwrap(),
      "15"
    );
    // c=d=0 with non-integer x rounds down to the lower order statistic.
    assert_eq!(
      interpret("Quantile[{10,20,30,40,50}, 1/2, {{0,0},{0,0}}]").unwrap(),
      "20"
    );
  }

  #[test]
  fn frac_zero_returns_lower_value() {
    // When x is an integer, the result is s[[x]] regardless of c (discontinuity).
    assert_eq!(
      interpret(&format!("Quantile[{D10}, 3/10, {{{{0,0}},{{1,0}}}}]"))
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn list_of_quantiles() {
    assert_eq!(
      interpret(&format!(
        "Quantile[{D10}, {{1/4, 1/2, 3/4}}, {{{{0,1}},{{0,1}}}}]"
      ))
      .unwrap(),
      "{11/4, 11/2, 33/4}"
    );
  }

  #[test]
  fn two_arg_form_unchanged() {
    assert_eq!(interpret("Quantile[{1,2,3,4,5}, 1/2]").unwrap(), "3");
    assert_eq!(interpret("Quantile[{1,2,3,4}, 0.25]").unwrap(), "1");
  }

  // With d = 0 the result is an exact list element; an inexact probability
  // must not leak a real into it (the d*frac term drops out entirely).
  // wolframscript: Quantile[{1,2,3,4,5}, 0.5, {{0,0},{1,0}}] = 3, not 3.
  #[test]
  fn real_probability_stays_exact_when_d_zero() {
    assert_eq!(
      interpret("Quantile[{1,2,3,4,5}, 0.5, {{0,0},{1,0}}]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Quantile[{1,2,3,4,5}, 0.3, {{0,0},{1,0}}]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Quantile[{10,20,30,40}, 0.25, {{0,0},{1,0}}]").unwrap(),
      "10"
    );
    assert_eq!(
      interpret("Quantile[{1,2,3,4,5}, 0.6, {{1/2,0},{0,0}}]").unwrap(),
      "3"
    );
  }

  // A real list still yields a real result (the elements carry the type).
  #[test]
  fn real_list_stays_real_when_d_zero() {
    assert_eq!(
      interpret("Quantile[{1.,2.,3.,4.,5.}, 0.5, {{0,0},{1,0}}]").unwrap(),
      "3."
    );
  }

  // Genuine interpolation (d != 0) with a real probability is still real.
  #[test]
  fn real_probability_interpolates_when_d_nonzero() {
    assert_eq!(
      interpret("Quantile[{1,2,3,4,5}, 0.7, {{1,-1},{0,1}}]").unwrap(),
      "3.8"
    );
  }
}

// WeightedData[data, weights] and the weighted statistics over it.
mod weighted_data {
  use super::*;

  // The constructor canonicalizes to the internal Automatic form.
  #[test]
  fn canonical_form() {
    assert_eq!(
      interpret("WeightedData[{1, 2, 3}, {1, 1, 2}]").unwrap(),
      "WeightedData[Automatic, {{1, 2, 3}, {1, 1, 2}}]"
    );
  }

  // Weighted mean = Σ(wᵢ xᵢ) / Σwᵢ, exactly.
  #[test]
  fn weighted_mean() {
    assert_eq!(
      interpret("Mean[WeightedData[{1, 2, 3}, {1, 1, 2}]]").unwrap(),
      "9/4"
    );
  }

  // Weighted (population) variance = Σwᵢ(xᵢ−μ)²/Σwᵢ.
  #[test]
  fn weighted_variance() {
    assert_eq!(
      interpret("Variance[WeightedData[{1, 2, 3}, {1, 1, 2}]]").unwrap(),
      "11/16"
    );
    assert_eq!(
      interpret("Variance[WeightedData[{2, 4, 6}, {1, 2, 1}]]").unwrap(),
      "2"
    );
    // Unit weights use the same (biased) normalization, not n−1.
    assert_eq!(
      interpret("Variance[WeightedData[{1, 2, 3, 4}, {1, 1, 1, 1}]]").unwrap(),
      "5/4"
    );
  }

  #[test]
  fn weighted_standard_deviation() {
    assert_eq!(
      interpret("StandardDeviation[WeightedData[{1, 2, 3}, {1, 1, 2}]]")
        .unwrap(),
      "Sqrt[11]/4"
    );
  }

  // Weighted median: the first value, in value order, whose cumulative weight
  // reaches half the total (no averaging of the two middle elements).
  #[test]
  fn weighted_median() {
    assert_eq!(
      interpret("Median[WeightedData[{1, 2, 3}, {1, 1, 2}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Median[WeightedData[{1, 2, 3, 4}, {1, 1, 1, 1}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Median[WeightedData[{10, 20, 30}, {1, 1, 1}]]").unwrap(),
      "20"
    );
    assert_eq!(
      interpret("Median[WeightedData[{1, 2, 3}, {5, 1, 1}]]").unwrap(),
      "1"
    );
  }

  // Plain-list statistics are unaffected by the WeightedData handling.
  #[test]
  fn plain_list_unaffected() {
    assert_eq!(interpret("Mean[{1, 2, 3}]").unwrap(), "2");
    assert_eq!(interpret("Median[{1, 2, 3, 4}]").unwrap(), "5/2");
  }
}

mod dms_string_tests {
  use woxi::interpret;

  #[test]
  fn scalar_angles() {
    assert_eq!(interpret("DMSString[30.264]").unwrap(), "30°15'50.400\"");
    assert_eq!(interpret("DMSString[123.456]").unwrap(), "123°27'21.600\"");
    // A bare scalar drops the sign entirely.
    assert_eq!(interpret("DMSString[-30.264]").unwrap(), "30°15'50.400\"");
    assert_eq!(interpret("DMSString[-30]").unwrap(), "30°0'0\"");
    // Exact input with integral seconds prints them without decimals…
    assert_eq!(interpret("DMSString[30]").unwrap(), "30°0'0\"");
    assert_eq!(interpret("DMSString[61/2]").unwrap(), "30°30'0\"");
    assert_eq!(interpret("DMSString[1/3]").unwrap(), "0°20'0\"");
    assert_eq!(interpret("DMSString[30 + 1/3600]").unwrap(), "30°0'1\"");
    // …but exact fractional seconds get the decimal treatment.
    assert_eq!(interpret("DMSString[30 + 1/7200]").unwrap(), "30°0'0.500\"");
    // Reals always show decimals, even for whole seconds.
    assert_eq!(interpret("DMSString[30.5]").unwrap(), "30°30'0.000\"");
    // Exact-but-irrational input takes the machine-real path.
    assert_eq!(interpret("DMSString[Pi]").unwrap(), "3°8'29.734\"");
  }

  // Seconds rounding carries into minutes and degrees.
  #[test]
  fn rounding_carry() {
    assert_eq!(interpret("DMSString[59.9999999]").unwrap(), "60°0'0.000\"");
    assert_eq!(
      interpret("DMSString[29.999999999999]").unwrap(),
      "30°0'0.000\""
    );
  }

  // The second argument sets the number of decimals on the seconds
  // (default 3, trailing dot at 0, half-even rounding). It is ignored for
  // exact integral seconds.
  #[test]
  fn precision_argument() {
    assert_eq!(interpret("DMSString[30.264, 1]").unwrap(), "30°15'50.4\"");
    assert_eq!(interpret("DMSString[30.264, 2]").unwrap(), "30°15'50.40\"");
    assert_eq!(
      interpret("DMSString[30.264, 6]").unwrap(),
      "30°15'50.400000\""
    );
    assert_eq!(interpret("DMSString[30.264, 0]").unwrap(), "30°15'50.\"");
    assert_eq!(interpret("DMSString[30, 2]").unwrap(), "30°0'0\"");
    assert_eq!(interpret("DMSString[30 + 1/7200, 0]").unwrap(), "30°0'0.\"");
    assert_eq!(interpret("DMSString[30 + 3/7200, 0]").unwrap(), "30°0'2.\"");
  }

  // {lat, lon} pairs get hemisphere suffixes (N/E for zero), two spaces
  // between the parts; GeoPosition unwraps (altitude dropped).
  #[test]
  fn lat_lon_pairs() {
    assert_eq!(
      interpret("DMSString[{30.264, -87.155}]").unwrap(),
      "30°15'50.400\"N  87°9'18.000\"W"
    );
    assert_eq!(
      interpret("DMSString[{-30.264, 87.155}]").unwrap(),
      "30°15'50.400\"S  87°9'18.000\"E"
    );
    assert_eq!(
      interpret("DMSString[{0, 0}]").unwrap(),
      "0°0'0\"N  0°0'0\"E"
    );
    assert_eq!(
      interpret("DMSString[{30, 87}]").unwrap(),
      "30°0'0\"N  87°0'0\"E"
    );
    assert_eq!(
      interpret("DMSString[{30.264, -87.155}, 1]").unwrap(),
      "30°15'50.4\"N  87°9'18.0\"W"
    );
    assert_eq!(
      interpret("DMSString[GeoPosition[{30.264, -87.155}]]").unwrap(),
      "30°15'50.400\"N  87°9'18.000\"W"
    );
    assert_eq!(
      interpret("DMSString[GeoPosition[{30.264, -87.155, 100}]]").unwrap(),
      "30°15'50.400\"N  87°9'18.000\"W"
    );
  }

  // A 3-element list is a {d, m, s} value, not a coordinate pair.
  #[test]
  fn dms_triple_input() {
    assert_eq!(
      interpret("DMSString[{30.1, 87.2, 100}]").unwrap(),
      "31°34'52.000\""
    );
    assert_eq!(
      interpret("DMSString[{30, 15, 50.5}]").unwrap(),
      "30°15'50.500\""
    );
  }

  // DMS strings parse and re-format; sign and hemisphere suffix drop.
  #[test]
  fn string_input() {
    assert_eq!(
      interpret(r#"DMSString["30°15'50\""]"#).unwrap(),
      "30°15'50\""
    );
    assert_eq!(interpret(r#"DMSString["30°"]"#).unwrap(), "30°0'0\"");
    assert_eq!(interpret(r#"DMSString["30°15'"]"#).unwrap(), "30°15'0\"");
    assert_eq!(
      interpret(r#"DMSString["30°15'50.5\""]"#).unwrap(),
      "30°15'50.500\""
    );
    assert_eq!(
      interpret(r#"DMSString["30°15'50\"N"]"#).unwrap(),
      "30°15'50\""
    );
    assert_eq!(
      interpret(r#"DMSString["-30°15'50\""]"#).unwrap(),
      "30°15'50\""
    );
  }

  // Invalid inputs echo the call (each with its own message tag).
  #[test]
  fn invalid_inputs() {
    assert_eq!(interpret("DMSString[x]").unwrap(), "DMSString[x]");
    assert_eq!(
      interpret("DMSString[{30.264}]").unwrap(),
      "DMSString[{30.264}]"
    );
    assert_eq!(
      interpret("DMSString[{{30, 15, 50}, {87, 9, 18}}]").unwrap(),
      "DMSString[{{30, 15, 50}, {87, 9, 18}}]"
    );
    assert_eq!(
      interpret(r#"DMSString["30.264"]"#).unwrap(),
      "DMSString[30.264]"
    );
    assert_eq!(
      interpret(r#"DMSString[45.5074, "Latitude"]"#).unwrap(),
      "DMSString[45.5074, Latitude]"
    );
    assert_eq!(
      interpret("DMSString[30.264, -1]").unwrap(),
      "DMSString[30.264, -1]"
    );
  }
}

mod hoeffding_d_tests {
  use woxi::interpret;

  // Exact input gives an exact rational result; D depends only on ranks.
  #[test]
  fn exact_vectors() {
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, 5}, {2, 3, 1, 5, 4}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}]").unwrap(),
      "1"
    );
    // D measures dependence, not direction: a reversal is still 1.
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, 5, 6}, {1, 3, 2, 4, 6, 5}]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 5, 4, 7, 6}]")
        .unwrap(),
      "4/7"
    );
    // Rank invariance: scaling the values changes nothing.
    assert_eq!(
      interpret("HoeffdingD[{2, 4, 6, 8, 10}, {2, 3, 1, 5, 4}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("HoeffdingD[{1/2, 1, 3/2, 2, 5/2}, {2, 3, 1, 5, 4}]").unwrap(),
      "0"
    );
  }

  // Ties use mid-ranks and the phi = 1/2 convention.
  #[test]
  fn ties() {
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 2, 3, 4}, {2, 2, 3, 4, 5}]").unwrap(),
      "9/64"
    );
  }

  #[test]
  fn real_input_gives_real() {
    assert_eq!(
      interpret("HoeffdingD[{1.5, 2.5, 3.5, 4.5, 5.5}, {2, 3, 1, 5, 4}]")
        .unwrap(),
      "0."
    );
  }

  // An n×k matrix gives the k×k column-pairwise matrix; two matrices give
  // the cross matrix.
  #[test]
  fn matrix_forms() {
    assert_eq!(
      interpret("HoeffdingD[{{1, 2}, {2, 3}, {3, 1}, {4, 5}, {5, 4}}]")
        .unwrap(),
      "{{1, 0}, {0, 1}}"
    );
    assert_eq!(
      interpret(
        "HoeffdingD[{{1, 2, 3}, {2, 3, 1}, {3, 1, 2}, {4, 5, 4}, {5, 4, 5}}]"
      )
      .unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
    assert_eq!(
      interpret(
        "HoeffdingD[{{1, 2}, {2, 3}, {3, 1}, {4, 5}, {5, 4}}, {{2, 1}, {3, 2}, {1, 3}, {5, 4}, {4, 5}}]"
      )
      .unwrap(),
      "{{0, 1}, {1, 0}}"
    );
  }

  // Short data emits ::dtlnth (with the offending position); mismatched
  // lengths and non-numeric data emit ::rctneqln; extra arguments stay
  // silently unevaluated.
  #[test]
  fn error_forms() {
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4}, {1, 2, 3, 4}]").unwrap(),
      "HoeffdingD[{1, 2, 3, 4}, {1, 2, 3, 4}]"
    );
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, 5}, {1, 2, 3, 4}]").unwrap(),
      "HoeffdingD[{1, 2, 3, 4, 5}, {1, 2, 3, 4}]"
    );
    assert_eq!(
      interpret("HoeffdingD[{{1, 2}, {2, 3}, {3, 1}}]").unwrap(),
      "HoeffdingD[{{1, 2}, {2, 3}, {3, 1}}]"
    );
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}]").unwrap(),
      "HoeffdingD[{1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5}]"
    );
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, x}, {1, 2, 3, 4, 5}]").unwrap(),
      "HoeffdingD[{1, 2, 3, 4, x}, {1, 2, 3, 4, 5}]"
    );
    assert_eq!(
      interpret("HoeffdingD[{1, 2, 3, 4, 5}, {2, 3, 1, 5, 4}, 3]").unwrap(),
      "HoeffdingD[{1, 2, 3, 4, 5}, {2, 3, 1, 5, 4}, 3]"
    );
  }
}

mod goodman_kruskal_gamma_tests {
  use woxi::interpret;

  // γ = (C - D)/(C + D) over untied pairs; exact input stays exact.
  #[test]
  fn exact_vectors() {
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 2, 3, 4, 5}, {2, 3, 1, 5, 4}]")
        .unwrap(),
      "2/5"
    );
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 2, 3}, {1, 3, 2}]").unwrap(),
      "1/3"
    );
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 2, 3, 4, 5, 6}, {2, 1, 4, 3, 6, 5}]")
        .unwrap(),
      "3/5"
    );
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 2}, {2, 1}]").unwrap(),
      "-1"
    );
    // Tied pairs are ignored entirely.
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 2, 2, 3}, {1, 2, 3, 3}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn real_input_gives_real() {
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1., 2., 3.}, {1, 3, 2}]").unwrap(),
      "0.3333333333333333"
    );
  }

  // No untied pairs at all → ::zrvr and Indeterminate (even when neither
  // vector is constant, like {1, 1} vs {1, 2}).
  #[test]
  fn zero_variance() {
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 1, 1}, {1, 2, 3}]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 1}, {1, 2}]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn matrix_forms_and_errors() {
    assert_eq!(
      interpret(
        "GoodmanKruskalGamma[{{1, 2}, {2, 3}, {3, 1}, {4, 5}, {5, 4}}]"
      )
      .unwrap(),
      "{{1, 2/5}, {2/5, 1}}"
    );
    assert_eq!(
      interpret(
        "GoodmanKruskalGamma[{{1, 2}, {2, 3}, {3, 1}}, {{2, 1}, {3, 2}, {1, 3}}]"
      )
      .unwrap(),
      "{{-1/3, 1}, {1, -1/3}}"
    );
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 2, 3}, {1, 2}]").unwrap(),
      "GoodmanKruskalGamma[{1, 2, 3}, {1, 2}]"
    );
    // Extra arguments stay silently unevaluated.
    assert_eq!(
      interpret("GoodmanKruskalGamma[{1, 2, 3}, {1, 3, 2}, 5]").unwrap(),
      "GoodmanKruskalGamma[{1, 2, 3}, {1, 3, 2}, 5]"
    );
  }
}

mod blomqvist_beta_tests {
  use woxi::interpret;

  // β correlates the sign vectors around the medians; points on a median
  // line shrink the denominator, giving values like 3/4 and 1/Sqrt[2].
  #[test]
  fn exact_vectors() {
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, 3, 4, 5}, {2, 3, 1, 5, 4}]").unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, 3, 4}, {1, 3, 2, 4}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, 3}, {3, 2, 1}]").unwrap(),
      "-1"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, 3, 4}, {1, 1, 2, 2}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, 2, 3}, {1, 2, 3, 4}]").unwrap(),
      "1/Sqrt[2]"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, 3, 4, 5}, {1, 2, 4, 3, 5}]").unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, 3, 4, 5}, {2, 1, 3, 5, 4}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 1, 2, 2}, {1, 2, 1, 2}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn real_input_gives_real() {
    assert_eq!(
      interpret("BlomqvistBeta[{1., 2., 3., 4.}, {1, 3, 2, 4}]").unwrap(),
      "0."
    );
  }

  // Constant data divides 0/Sqrt[0], which surfaces as Indeterminate (with
  // wolframscript's Power::infy / Infinity::indet messages).
  #[test]
  fn constant_data_is_indeterminate() {
    assert_eq!(
      interpret("BlomqvistBeta[{1, 1, 1}, {1, 2, 3}]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn matrix_forms_and_errors() {
    assert_eq!(
      interpret("BlomqvistBeta[{{1, 2}, {2, 3}, {3, 1}, {4, 5}, {5, 4}}]")
        .unwrap(),
      "{{1, 3/4}, {3/4, 1}}"
    );
    assert_eq!(
      interpret(
        "BlomqvistBeta[{{1, 2}, {2, 3}, {3, 1}}, {{2, 1}, {3, 2}, {1, 3}}]"
      )
      .unwrap(),
      "{{-1/2, 1}, {1, -1/2}}"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, x}, {1, 2, 3}]").unwrap(),
      "BlomqvistBeta[{1, 2, x}, {1, 2, 3}]"
    );
    assert_eq!(
      interpret("BlomqvistBeta[{1, 2, 3, 4}, {1, 3, 2, 4}, 5]").unwrap(),
      "BlomqvistBeta[{1, 2, 3, 4}, {1, 3, 2, 4}, 5]"
    );
  }
}

mod erlang_tests {
  use woxi::interpret;

  // ErlangB[c, a] = (a^c/c!) / Σ_{k<=c} a^k/k!, exact over the rationals.
  #[test]
  fn erlang_b() {
    assert_eq!(interpret("ErlangB[3, 2]").unwrap(), "4/19");
    assert_eq!(interpret("ErlangB[1, 1]").unwrap(), "1/2");
    assert_eq!(interpret("ErlangB[5, 3/2]").unwrap(), "81/5711");
    assert_eq!(
      interpret("ErlangB[20, 15]").unwrap(),
      "81091461181640625/1778586136832190169"
    );
    // Machine reals convert exactly and round back.
    assert_eq!(interpret("ErlangB[3, 2.]").unwrap(), "0.21052631578947367");
  }

  // ErlangC clamps to 1 for a >= c (unstable queue) and gives 0 for a = 0.
  #[test]
  fn erlang_c() {
    assert_eq!(interpret("ErlangC[3, 2]").unwrap(), "4/9");
    assert_eq!(interpret("ErlangC[1, 1/2]").unwrap(), "1/2");
    assert_eq!(interpret("ErlangC[3, 2.]").unwrap(), "0.4444444444444444");
    assert_eq!(interpret("ErlangC[2, 3]").unwrap(), "1");
    assert_eq!(interpret("ErlangC[2, 2]").unwrap(), "1");
    assert_eq!(interpret("ErlangC[2, 5/2]").unwrap(), "1");
    assert_eq!(interpret("ErlangC[3, 0]").unwrap(), "0");
  }

  // Non-positive-integer server counts emit ::intp; non-positive traffic
  // emits ::posprm for ErlangB (which, unlike ErlangC, rejects a = 0).
  #[test]
  fn error_forms() {
    assert_eq!(interpret("ErlangB[0, 2]").unwrap(), "ErlangB[0, 2]");
    assert_eq!(interpret("ErlangB[3, -1]").unwrap(), "ErlangB[3, -1]");
    assert_eq!(interpret("ErlangB[3, 0]").unwrap(), "ErlangB[3, 0]");
    assert_eq!(interpret("ErlangB[x, 0]").unwrap(), "ErlangB[x, 0]");
    assert_eq!(interpret("ErlangB[0, a]").unwrap(), "ErlangB[0, a]");
    assert_eq!(interpret("ErlangB[3]").unwrap(), "ErlangB[3]");
  }

  // Symbolic arguments produce wolframscript's Piecewise Gamma closed form
  // (a^c/(E^a*Gamma[1 + c, a]) over the parameter domain).
  #[test]
  fn symbolic_closed_form() {
    assert_eq!(
      interpret("ErlangB[x, 2]").unwrap(),
      "Piecewise[{{2^x/(E^2*Gamma[1 + x, 2]), Element[x, Integers] && x > 0}}, Indeterminate]"
    );
    assert_eq!(
      interpret("ErlangB[x, a]").unwrap(),
      "Piecewise[{{a^x/(E^a*Gamma[1 + x, a]), Element[x, Integers] && Element[a, Reals] && a > 0 && x > 0}}, Indeterminate]"
    );
    assert_eq!(
      interpret("ErlangB[3, a]").unwrap(),
      "Piecewise[{{a^3/(E^a*Gamma[4, a]), Element[a, Reals] && a > 0}}, Indeterminate]"
    );
    assert_eq!(
      interpret("ErlangB[x, 2.5]").unwrap(),
      "Piecewise[{{(0.0820849986238988*2.5^x)/Gamma[1 + x, 2.5], Element[x, Integers] && x > 0}}, Indeterminate]"
    );
  }
}

mod central_feature_tests {
  use woxi::interpret;

  // The element minimizing total distance to the others; earliest wins
  // ties. All values verified against wolframscript.
  #[test]
  fn numeric_data() {
    assert_eq!(interpret("CentralFeature[{1, 2, 3, 4}]").unwrap(), "2");
    assert_eq!(interpret("CentralFeature[{1, 2, 3, 4, 5}]").unwrap(), "3");
    assert_eq!(interpret("CentralFeature[{3, 1, 2}]").unwrap(), "2");
    assert_eq!(interpret("CentralFeature[{1., 5., 2.5}]").unwrap(), "2.5");
    assert_eq!(interpret("CentralFeature[{1/2, 3/2, 5}]").unwrap(), "3/2");
    assert_eq!(interpret("CentralFeature[{2, 2, 8}]").unwrap(), "2");
    assert_eq!(interpret("CentralFeature[{7}]").unwrap(), "7");
    // Numeric symbolic constants participate numerically and the original
    // expression is returned.
    assert_eq!(interpret("CentralFeature[{Pi, E, 3}]").unwrap(), "3");
    assert_eq!(
      interpret("CentralFeature[{Sqrt[2], 2, 0}]").unwrap(),
      "Sqrt[2]"
    );
    // Complex numbers act as 2D points.
    assert_eq!(
      interpret("CentralFeature[{1 + 2 I, 3 I, 2}]").unwrap(),
      "1 + 2*I"
    );
  }

  #[test]
  fn vector_data() {
    assert_eq!(
      interpret(
        "CentralFeature[{{1., 3., 5.}, {7., 1., 2.}, {9., 3., 1.}, {4., 5., 6.}}]"
      )
      .unwrap(),
      "{7., 1., 2.}"
    );
    assert_eq!(
      interpret("CentralFeature[{{1, 2, 3}, {2, 3, 4}, {10, 10, 10}}]")
        .unwrap(),
      "{2, 3, 4}"
    );
    // Symmetric tie picks the earlier element.
    assert_eq!(
      interpret("CentralFeature[{{0, 0}, {1, 0}, {0, 1}, {5, 5}}]").unwrap(),
      "{1, 0}"
    );
  }

  // Strings use EditDistance; non-metric data falls back to the discrete
  // equality metric (so frequency wins, then position).
  #[test]
  fn strings_and_fallback() {
    assert_eq!(
      interpret(r#"CentralFeature[{"ab", "abc", "abcde"}]"#).unwrap(),
      "abc"
    );
    assert_eq!(interpret("CentralFeature[{b, a, a}]").unwrap(), "a");
    assert_eq!(interpret("CentralFeature[{b, b, a}]").unwrap(), "b");
    assert_eq!(interpret("CentralFeature[{b, a, c}]").unwrap(), "b");
    assert_eq!(interpret("CentralFeature[{0, 10, 11, a}]").unwrap(), "0");
    assert_eq!(interpret("CentralFeature[{1, a, 2}]").unwrap(), "1");
    assert_eq!(
      interpret(r#"CentralFeature[{"ab", "abc", 5, 6}]"#).unwrap(),
      "ab"
    );
    assert_eq!(
      interpret("CentralFeature[{{1, 2}, {3}, {4, 5}}]").unwrap(),
      "{1, 2}"
    );
    assert_eq!(
      interpret("CentralFeature[{True, False, True}]").unwrap(),
      "True"
    );
  }

  // Rule forms return the value belonging to the central key.
  #[test]
  fn rule_forms() {
    assert_eq!(interpret("CentralFeature[{1, 2} -> {x, y}]").unwrap(), "x");
    assert_eq!(
      interpret("CentralFeature[{1, 2, 3} -> {x, y, z}]").unwrap(),
      "y"
    );
    assert_eq!(
      interpret("CentralFeature[{1 -> x, 2 -> y, 3 -> z}]").unwrap(),
      "y"
    );
    assert_eq!(interpret("CentralFeature[{x -> 1, 2 -> y}]").unwrap(), "1");
  }

  // Invalid data emits `near1`; extra non-options emit `nonopt`; zero
  // arguments emit `argx` — all unevaluated, matching wolframscript.
  #[test]
  fn message_cases() {
    let r = woxi::interpret_with_stdout("CentralFeature[{}]").unwrap();
    assert_eq!(r.result, "CentralFeature[{}]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "CentralFeature::near1: {} is neither a list of real points nor a valid list of rules."
    )));

    let r = woxi::interpret_with_stdout("CentralFeature[7]").unwrap();
    assert_eq!(r.result, "CentralFeature[7]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "CentralFeature::near1: 7 is neither a list of real points nor a valid list of rules."
    )));

    let r =
      woxi::interpret_with_stdout("CentralFeature[{1, 2} -> {x}]").unwrap();
    assert_eq!(r.result, "CentralFeature[{1, 2} -> {x}]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "CentralFeature::near1: {1, 2} -> {x} is neither a list of real points nor a valid list of rules."
    )));

    let r = woxi::interpret_with_stdout("CentralFeature[{1, 2}, 3]").unwrap();
    assert_eq!(r.result, "CentralFeature[{1, 2}, 3]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "CentralFeature::nonopt: Options expected (instead of 3) beyond position 1 in CentralFeature[{1, 2}, 3]. An option must be a rule or a list of rules."
    )));

    let r = woxi::interpret_with_stdout("CentralFeature[]").unwrap();
    assert_eq!(r.result, "CentralFeature[]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "CentralFeature::argx: CentralFeature called with 0 arguments; 1 argument is expected."
    )));
  }
}

mod power_distribution_tests {
  use woxi::interpret;

  // PowerDistribution[k, a] has support (0, 1/k]. All values verified
  // against wolframscript.
  #[test]
  fn pdf_and_cdf() {
    assert_eq!(
      interpret("PDF[PowerDistribution[k, a], x]").unwrap(),
      "Piecewise[{{a*k^a*x^(-1 + a), Inequality[0, Less, x, LessEqual, k^(-1)]}}, 0]"
    );
    assert_eq!(
      interpret("PDF[PowerDistribution[2, 3], x]").unwrap(),
      "Piecewise[{{24*x^2, Inequality[0, Less, x, LessEqual, 1/2]}}, 0]"
    );
    assert_eq!(
      interpret("CDF[PowerDistribution[k, a], x]").unwrap(),
      "Piecewise[{{(k*x)^a, Inequality[0, Less, x, LessEqual, k^(-1)]}, {1, x > k^(-1)}}, 0]"
    );
    assert_eq!(
      interpret("CDF[PowerDistribution[2, 3], x]").unwrap(),
      "Piecewise[{{8*x^3, Inequality[0, Less, x, LessEqual, 1/2]}, {1, x > 1/2}}, 0]"
    );
    // Exact arguments give exact values.
    assert_eq!(interpret("PDF[PowerDistribution[2, 3], 1/2]").unwrap(), "6");
    assert_eq!(interpret("PDF[PowerDistribution[2, 3], 1]").unwrap(), "0");
    assert_eq!(interpret("PDF[PowerDistribution[2, 3], 5]").unwrap(), "0");
    assert_eq!(interpret("PDF[PowerDistribution[2, 3], 0]").unwrap(), "0");
    assert_eq!(interpret("CDF[PowerDistribution[2, 3], 1/2]").unwrap(), "1");
    assert_eq!(interpret("CDF[PowerDistribution[2, 3], 3]").unwrap(), "1");
    assert_eq!(interpret("CDF[PowerDistribution[2, 3], 0]").unwrap(), "0");
    // Machine-real arguments numericize the result (unlike most other
    // distributions in wolframscript).
    assert_eq!(
      interpret("PDF[PowerDistribution[2, 3], 1.5]").unwrap(),
      "0."
    );
    assert_eq!(
      interpret("CDF[PowerDistribution[2, 3], 1.5]").unwrap(),
      "1."
    );
    assert_eq!(
      interpret("PDF[PowerDistribution[2, 3], -1.]").unwrap(),
      "0."
    );
    assert_eq!(
      interpret("CDF[PowerDistribution[2, 3], -1.]").unwrap(),
      "0."
    );
    assert_eq!(
      interpret("CDF[PowerDistribution[2, 3], 0.3]").unwrap(),
      "0.21599999999999997"
    );
  }

  #[test]
  fn moments_and_median() {
    assert_eq!(
      interpret("Mean[PowerDistribution[k, a]]").unwrap(),
      "a/(k + a*k)"
    );
    assert_eq!(interpret("Mean[PowerDistribution[2, 3]]").unwrap(), "3/8");
    assert_eq!(
      interpret("Mean[PowerDistribution[2., 3.]]").unwrap(),
      "0.375"
    );
    assert_eq!(
      interpret("Variance[PowerDistribution[k, a]]").unwrap(),
      "a/((1 + a)^2*(2 + a)*k^2)"
    );
    assert_eq!(
      interpret("Variance[PowerDistribution[2, 3]]").unwrap(),
      "3/320"
    );
    assert_eq!(
      interpret("StandardDeviation[PowerDistribution[2, 3]]").unwrap(),
      "Sqrt[3/5]/8"
    );
    assert_eq!(
      interpret("Median[PowerDistribution[k, a]]").unwrap(),
      "1/(2^a^(-1)*k)"
    );
    assert_eq!(
      interpret("Median[PowerDistribution[2, 3]]").unwrap(),
      "1/(2*2^(1/3))"
    );
  }
}

mod pert_distribution_tests {
  use woxi::interpret;

  // PERTDistribution[{min, max}, mode] — a Beta distribution rescaled to
  // the interval, shape exponent 4 (or g in the modified 3-argument
  // form). All values verified against wolframscript.
  #[test]
  fn pdf_and_cdf() {
    assert_eq!(
      interpret("PDF[PERTDistribution[{a, b}, m], x]").unwrap(),
      "Piecewise[{{((b - x)^((4*(b - m))/(-a + b))*(-a + x)^((4*(-a + m))/(-a + b)))/((-a + b)^5*Beta[1 + (4*(-a + m))/(-a + b), 1 + (4*(b - m))/(-a + b)]), a < x < b}}, 0]"
    );
    assert_eq!(
      interpret("CDF[PERTDistribution[{a, b}, m], x]").unwrap(),
      "Piecewise[{{BetaRegularized[(-a + x)/(-a + b), 1 + (4*(-a + m))/(-a + b), 1 + (4*(b - m))/(-a + b)], a < x < b}, {1, x >= b}}, 0]"
    );
    assert_eq!(
      interpret("PDF[PERTDistribution[{0, 4}, 3], x]").unwrap(),
      "Piecewise[{{(5*(4 - x)*x^3)/256, 0 < x < 4}}, 0]"
    );
    assert_eq!(
      interpret("CDF[PERTDistribution[{0, 4}, 3], x]").unwrap(),
      "Piecewise[{{BetaRegularized[x/4, 4, 2], 0 < x < 4}, {1, x >= 4}}, 0]"
    );
    // The modified (3-argument) form keeps (b-a)^(-1-g) as a factor.
    assert_eq!(
      interpret("PDF[PERTDistribution[{a, b}, m, g], x]").unwrap(),
      "Piecewise[{{((-a + b)^(-1 - g)*(b - x)^((g*(b - m))/(-a + b))*(-a + x)^((g*(-a + m))/(-a + b)))/Beta[1 + (g*(-a + m))/(-a + b), 1 + (g*(b - m))/(-a + b)], a < x < b}}, 0]"
    );
    assert_eq!(
      interpret("CDF[PERTDistribution[{0, 4}, 3, 2], x]").unwrap(),
      "Piecewise[{{BetaRegularized[x/4, 5/2, 3/2], 0 < x < 4}, {1, x >= 4}}, 0]"
    );
    // Exact arguments give exact values; machine reals numericize.
    assert_eq!(
      interpret("PDF[PERTDistribution[{0, 4}, 3], 2]").unwrap(),
      "5/16"
    );
    assert_eq!(
      interpret("CDF[PERTDistribution[{0, 4}, 3], 2]").unwrap(),
      "3/16"
    );
    assert_eq!(
      interpret("PDF[PERTDistribution[{0, 4}, 3], 5]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("CDF[PERTDistribution[{0, 4}, 3], 5]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("PDF[PERTDistribution[{0, 4}, 3], 5.]").unwrap(),
      "0."
    );
    assert_eq!(
      interpret("CDF[PERTDistribution[{0, 4}, 3], 5.]").unwrap(),
      "1."
    );
  }

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[PERTDistribution[{a, b}, m]]").unwrap(),
      "(a + b + 4*m)/6"
    );
    assert_eq!(
      interpret("Mean[PERTDistribution[{0, 4}, 3]]").unwrap(),
      "8/3"
    );
    assert_eq!(
      interpret("Mean[PERTDistribution[{0., 4.}, 3.]]").unwrap(),
      "2.6666666666666665"
    );
    assert_eq!(
      interpret("Variance[PERTDistribution[{a, b}, m]]").unwrap(),
      "((-a + 5*b - 4*m)*(-5*a + b + 4*m))/252"
    );
    assert_eq!(
      interpret("Variance[PERTDistribution[{0, 4}, 3]]").unwrap(),
      "32/63"
    );
    assert_eq!(
      interpret("StandardDeviation[PERTDistribution[{0, 4}, 3]]").unwrap(),
      "(4*Sqrt[2/7])/3"
    );
    // Modified form.
    assert_eq!(
      interpret("Mean[PERTDistribution[{a, b}, m, g]]").unwrap(),
      "(a + b + g*m)/(2 + g)"
    );
    assert_eq!(
      interpret("Mean[PERTDistribution[{0, 4}, 3, 2]]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("Variance[PERTDistribution[{0, 4}, 3, 2]]").unwrap(),
      "3/4"
    );
  }

  // One argument emits `argtu` via the arg-count table.
  #[test]
  fn argument_count() {
    let r = woxi::interpret_with_stdout("PERTDistribution[{0, 4}]").unwrap();
    assert_eq!(r.result, "PERTDistribution[{0, 4}]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "PERTDistribution::argtu: PERTDistribution called with 1 argument; 2 or 3 arguments are expected."
    )));
  }
}

mod factorial_central_mgf_tests {
  use woxi::interpret;

  // FactorialMomentGeneratingFunction[dist, t] = E[t^X], i.e. the MGF at
  // Log[t]. All values verified against wolframscript.
  #[test]
  fn factorial_mgf() {
    assert_eq!(
      interpret("FactorialMomentGeneratingFunction[PoissonDistribution[m], t]")
        .unwrap(),
      "E^(m*(-1 + t))"
    );
    assert_eq!(
      interpret(
        "FactorialMomentGeneratingFunction[BernoulliDistribution[p], t]"
      )
      .unwrap(),
      "1 - p + p*t"
    );
    assert_eq!(
      interpret(
        "FactorialMomentGeneratingFunction[BinomialDistribution[n, p], t]"
      )
      .unwrap(),
      "(1 + p*(-1 + t))^n"
    );
    assert_eq!(
      interpret(
        "FactorialMomentGeneratingFunction[GeometricDistribution[p], t]"
      )
      .unwrap(),
      "p/(1 - (1 - p)*t)"
    );
    assert_eq!(
      interpret(
        "FactorialMomentGeneratingFunction[ExponentialDistribution[a], t]"
      )
      .unwrap(),
      "a/(a - Log[t])"
    );
    // Symbolic Normal keeps the Log form unfolded; numeric m folds to
    // t^m — both engines agree on that split.
    assert_eq!(
      interpret(
        "FactorialMomentGeneratingFunction[NormalDistribution[m, s], t]"
      )
      .unwrap(),
      "E^(m*Log[t] + (s^2*Log[t]^2)/2)"
    );
    assert_eq!(
      interpret(
        "FactorialMomentGeneratingFunction[NormalDistribution[2, 1], t]"
      )
      .unwrap(),
      "E^(Log[t]^2/2)*t^2"
    );
    assert_eq!(
      interpret("FactorialMomentGeneratingFunction[NormalDistribution[], t]")
        .unwrap(),
      "E^(Log[t]^2/2)"
    );
    assert_eq!(
      interpret("FactorialMomentGeneratingFunction[PoissonDistribution[3], t]")
        .unwrap(),
      "E^(3*(-1 + t))"
    );
  }

  // CentralMomentGeneratingFunction[dist, t] = E^(-t Mean) MGF(t).
  #[test]
  fn central_mgf() {
    assert_eq!(
      interpret("CentralMomentGeneratingFunction[NormalDistribution[m, s], t]")
        .unwrap(),
      "E^((s^2*t^2)/2)"
    );
    assert_eq!(
      interpret("CentralMomentGeneratingFunction[NormalDistribution[0, 1], t]")
        .unwrap(),
      "E^(t^2/2)"
    );
    assert_eq!(
      interpret("CentralMomentGeneratingFunction[PoissonDistribution[m], t]")
        .unwrap(),
      "E^((-1 + E^t)*m - m*t)"
    );
    assert_eq!(
      interpret("CentralMomentGeneratingFunction[PoissonDistribution[4], t]")
        .unwrap(),
      "E^(4*(-1 + E^t) - 4*t)"
    );
    assert_eq!(
      interpret(
        "CentralMomentGeneratingFunction[ExponentialDistribution[a], t]"
      )
      .unwrap(),
      "a/(E^(t/a)*(a - t))"
    );
    assert_eq!(
      interpret(
        "CentralMomentGeneratingFunction[UniformDistribution[{a, b}], t]"
      )
      .unwrap(),
      "(-E^(a*t) + E^(b*t))/((-a + b)*E^(((a + b)*t)/2)*t)"
    );
    assert_eq!(
      interpret(
        "CentralMomentGeneratingFunction[UniformDistribution[{0, 2}], t]"
      )
      .unwrap(),
      "(-1 + E^(2*t))/(2*E^t*t)"
    );
    assert_eq!(
      interpret("CentralMomentGeneratingFunction[BernoulliDistribution[p], t]")
        .unwrap(),
      "(1 - p + E^t*p)/E^(p*t)"
    );
  }
}
