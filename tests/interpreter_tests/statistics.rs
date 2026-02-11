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
}
