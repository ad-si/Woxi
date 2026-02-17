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
      "{a/Sqrt[a^2 + b^2], b/Sqrt[a^2 + b^2]}"
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
