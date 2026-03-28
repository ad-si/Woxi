use super::*;

mod variance {
  use super::*;

  #[test]
  fn variance_integers() {
    assert_eq!(interpret("Variance[{1, 2, 3}]").unwrap(), "1");
    assert_eq!(interpret("Variance[{1, 2, 3, 4, 5}]").unwrap(), "5/2");
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
  fn geometric_mean_reals() {
    let result = interpret("GeometricMean[{1.0, 2.0, 3.0}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.8171205928321397).abs() < 1e-10);
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
  fn symbolic_returns_unevaluated() {
    assert_eq!(interpret("HistogramList[x]").unwrap(), "HistogramList[x]");
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
  fn total_non_list() {
    assert_eq!(interpret("Total[5]").unwrap(), "5");
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

mod mean {
  use super::*;

  #[test]
  fn mean_symbolic() {
    assert_eq!(interpret("Mean[{a, b}]").unwrap(), "(a + b)/2");
  }

  #[test]
  fn mean_integers() {
    assert_eq!(interpret("Mean[{1, 2, 3}]").unwrap(), "2");
  }

  #[test]
  fn mean_rationals() {
    assert_eq!(interpret("Mean[{1/2, 1/3, 1/6}]").unwrap(), "1/3");
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
      "E*(-1 + E)"
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
    assert_eq!(
      interpret("PDF[ChiSquareDistribution[k], x]").unwrap(),
      "Piecewise[{{x^(k/2 - 1)/(2^(k/2)*E^(x/2)*Gamma[k/2]), x > 0}}, 0]"
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
    assert_eq!(
      interpret("PDF[ParetoDistribution[k, a], x]").unwrap(),
      "Piecewise[{{(a*k^a)/x^(1 + a), x >= k}}, 0]"
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
      "2/3/(3*E^(1/9))"
    );
  }

  #[test]
  fn pdf_symbolic() {
    assert_eq!(
      interpret("PDF[WeibullDistribution[a, b], x]").unwrap(),
      "Piecewise[{{(a*(x/b)^(a - 1))/(b*E^(x/b)^a), x > 0}}, 0]"
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
      "(Sqrt[a^2 - b^2]*E^(-(a*Sqrt[d^2 + (-m + x)^2]) + b*(-m + x)))/(2*a*d*BesselK[1, Sqrt[a^2 - b^2]*d])"
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
      "(d*BesselK[2, Sqrt[a^2 - b^2]*d])/(Sqrt[a^2 - b^2]*BesselK[1, Sqrt[a^2 - b^2]*d]) + (b^2*d^2*BesselK[3, Sqrt[a^2 - b^2]*d])/((a^2 - b^2)*BesselK[1, Sqrt[a^2 - b^2]*d]) - (b^2*d^2*BesselK[2, Sqrt[a^2 - b^2]*d]^2)/((a^2 - b^2)*BesselK[1, Sqrt[a^2 - b^2]*d]^2)"
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
    // TestDataTable returns a Grid which renders as -Graphics-
    let result = interpret(
      "LocationTest[{1.2, 0.5, 1.9, 2.1, 0.8, 1.5}, 0, \"TestDataTable\"]",
    )
    .unwrap();
    assert!(
      result == "-Graphics-",
      "Expected -Graphics-, got: {}",
      result
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
      "delta/(E^(((delta*(-mu + x))/sigma + gamma)^2/2)*Sqrt[2*Pi]*sigma)"
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
      "delta/(E^((delta*ArcSinh[(-mu + x)/sigma] + gamma)^2/2)*Sqrt[2*Pi]*Sqrt[(-mu + x)^2 + sigma^2])"
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
      "1/(Sqrt[2*Pi]/4)"
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
      "Erfc[(-((delta*(-mu + x))/sigma) - gamma)/Sqrt[2]]/2"
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
      "(1 + Erf[(delta*ArcSinh[(-mu + x)/sigma] + gamma)/Sqrt[2]])/2"
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
      "E^((1 - 2*delta*gamma)/(2*delta^2))*sigma + mu"
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
      "(sigma^2*(-1 + E^(1/delta^2))*(1 + E^(1/delta^2)*Cosh[(2*gamma)/delta]))/2"
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
      "E^((1 - 2*delta*gamma)/delta^2)*sigma^2*(-1 + E^(1/delta^2))"
    );
  }

  #[test]
  fn variance_sl_numeric() {
    assert_eq!(
      interpret(r#"Variance[JohnsonDistribution["SL", 0, 1, 0, 1]]"#).unwrap(),
      "E*(-1 + E)"
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
