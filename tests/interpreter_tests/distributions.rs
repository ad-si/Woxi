use super::*;

mod cauchy_distribution {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("CauchyDistribution[0, 1]").unwrap(),
      "CauchyDistribution[0, 1]"
    );
  }

  #[test]
  fn pdf_at_zero() {
    assert_eq!(
      interpret("PDF[CauchyDistribution[0, 1], 0]").unwrap(),
      "Pi^(-1)"
    );
  }

  #[test]
  fn cdf_at_zero() {
    assert_eq!(
      interpret("CDF[CauchyDistribution[0, 1], 0]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn cdf_at_one_numeric() {
    let result = interpret("N[CDF[CauchyDistribution[0, 1], 1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.75).abs() < 1e-10);
  }

  #[test]
  fn mean_indeterminate() {
    assert_eq!(
      interpret("Mean[CauchyDistribution[0, 1]]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn variance_indeterminate() {
    assert_eq!(
      interpret("Variance[CauchyDistribution[0, 1]]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn pdf_with_params_numeric() {
    let result = interpret("N[PDF[CauchyDistribution[2, 3], 0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 3.0 / (13.0 * std::f64::consts::PI);
    assert!((val - expected).abs() < 1e-10);
  }
}

mod hypoexponential_distribution {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("HypoexponentialDistribution[{1, 2, 3}]").unwrap(),
      "HypoexponentialDistribution[{1, 2, 3}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(
      interpret("Head[HypoexponentialDistribution]").unwrap(),
      "Symbol"
    );
  }
}

mod geometric_distribution {
  use super::*;

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("GeometricDistribution[1/3]").unwrap(),
      "GeometricDistribution[1/3]"
    );
  }

  #[test]
  fn mean() {
    assert_eq!(interpret("Mean[GeometricDistribution[1/3]]").unwrap(), "2");
  }

  #[test]
  fn variance() {
    assert_eq!(
      interpret("Variance[GeometricDistribution[1/3]]").unwrap(),
      "6"
    );
  }

  #[test]
  fn pdf_at_5() {
    assert_eq!(
      interpret("PDF[GeometricDistribution[1/3], 5]").unwrap(),
      "32/729"
    );
  }

  #[test]
  fn cdf_at_5() {
    assert_eq!(
      interpret("CDF[GeometricDistribution[1/3], 5]").unwrap(),
      "665/729"
    );
  }

  #[test]
  fn pdf_at_0() {
    assert_eq!(
      interpret("PDF[GeometricDistribution[1/3], 0]").unwrap(),
      "1/3"
    );
  }

  #[test]
  fn cdf_at_0() {
    assert_eq!(
      interpret("CDF[GeometricDistribution[1/3], 0]").unwrap(),
      "1/3"
    );
  }

  #[test]
  fn standard_deviation() {
    assert_eq!(
      interpret("StandardDeviation[GeometricDistribution[1/3]]").unwrap(),
      "Sqrt[6]"
    );
  }
}

mod censored_distribution {
  use super::*;

  #[test]
  fn with_normal() {
    assert_eq!(
      interpret("CensoredDistribution[{1, 5}, NormalDistribution[]]").unwrap(),
      "CensoredDistribution[{1, 5}, NormalDistribution[0, 1]]"
    );
  }

  #[test]
  fn with_explicit_normal() {
    assert_eq!(
      interpret("CensoredDistribution[{-2, 2}, NormalDistribution[0, 1]]")
        .unwrap(),
      "CensoredDistribution[{-2, 2}, NormalDistribution[0, 1]]"
    );
  }

  #[test]
  fn with_exponential() {
    assert_eq!(
      interpret("CensoredDistribution[{0, 10}, ExponentialDistribution[1]]")
        .unwrap(),
      "CensoredDistribution[{0, 10}, ExponentialDistribution[1]]"
    );
  }

  #[test]
  fn with_infinity_bounds() {
    assert_eq!(
      interpret("CensoredDistribution[{-Infinity, 5}, NormalDistribution[]]")
        .unwrap(),
      "CensoredDistribution[{-Infinity, 5}, NormalDistribution[0, 1]]"
    );
  }
}

mod distribution_parameter_q {
  use super::*;

  #[test]
  fn normal_valid() {
    assert_eq!(
      interpret("DistributionParameterQ[NormalDistribution[0, 1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn normal_invalid() {
    assert_eq!(
      interpret("DistributionParameterQ[NormalDistribution[0, -1]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn exponential_valid() {
    assert_eq!(
      interpret("DistributionParameterQ[ExponentialDistribution[1]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn exponential_invalid() {
    assert_eq!(
      interpret("DistributionParameterQ[ExponentialDistribution[-1]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn poisson_valid() {
    assert_eq!(
      interpret("DistributionParameterQ[PoissonDistribution[5]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn bernoulli_valid() {
    assert_eq!(
      interpret("DistributionParameterQ[BernoulliDistribution[0.5]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn bernoulli_invalid() {
    assert_eq!(
      interpret("DistributionParameterQ[BernoulliDistribution[2]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn uniform_valid() {
    assert_eq!(
      interpret("DistributionParameterQ[UniformDistribution[{0, 1}]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn non_distribution() {
    assert_eq!(
      interpret("DistributionParameterQ[42]").unwrap(),
      "DistributionParameterQ[42]"
    );
  }

  #[test]
  fn binomial_valid() {
    assert_eq!(
      interpret("DistributionParameterQ[BinomialDistribution[10, 0.6]]")
        .unwrap(),
      "True"
    );
  }

  #[test]
  fn normal_defaults() {
    assert_eq!(
      interpret("DistributionParameterQ[NormalDistribution[]]").unwrap(),
      "True"
    );
  }
}

mod moving_average {
  use super::*;

  #[test]
  fn basic_integers() {
    assert_eq!(
      interpret("MovingAverage[{1, 2, 3, 4, 5}, 3]").unwrap(),
      "{2, 3, 4}"
    );
  }

  #[test]
  fn window_two() {
    assert_eq!(
      interpret("MovingAverage[{1, 2, 3, 4, 5, 6}, 2]").unwrap(),
      "{3/2, 5/2, 7/2, 9/2, 11/2}"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("MovingAverage[{a, b, c, d}, 2]").unwrap(),
      "{(a + b)/2, (b + c)/2, (c + d)/2}"
    );
  }

  #[test]
  fn window_equals_length() {
    assert_eq!(interpret("MovingAverage[{1, 2, 3}, 3]").unwrap(), "{2}");
  }
}

mod moving_median {
  use super::*;

  #[test]
  fn basic_integers() {
    assert_eq!(
      interpret("MovingMedian[{1, 2, 3, 4, 5}, 3]").unwrap(),
      "{2, 3, 4}"
    );
  }

  #[test]
  fn window_two() {
    assert_eq!(
      interpret("MovingMedian[{1, 2, 3, 4, 5, 6}, 2]").unwrap(),
      "{3/2, 5/2, 7/2, 9/2, 11/2}"
    );
  }

  #[test]
  fn window_equals_length() {
    assert_eq!(interpret("MovingMedian[{1, 2, 3}, 3]").unwrap(), "{2}");
  }
}

mod mean_deviation {
  use super::*;

  #[test]
  fn basic_list() {
    assert_eq!(interpret("MeanDeviation[{1, 2, 3, 4, 5}]").unwrap(), "6/5");
  }

  #[test]
  fn even_list() {
    assert_eq!(interpret("MeanDeviation[{2, 4, 6, 8}]").unwrap(), "2");
  }

  #[test]
  fn identical_values() {
    assert_eq!(interpret("MeanDeviation[{5, 5, 5}]").unwrap(), "0");
  }
}

mod median_deviation {
  use super::*;

  #[test]
  fn basic_list() {
    assert_eq!(interpret("MedianDeviation[{1, 2, 3, 4, 5}]").unwrap(), "1");
  }

  #[test]
  fn even_list() {
    assert_eq!(interpret("MedianDeviation[{2, 4, 6, 8}]").unwrap(), "2");
  }

  #[test]
  fn identical_values() {
    assert_eq!(interpret("MedianDeviation[{5, 5, 5}]").unwrap(), "0");
  }

  #[test]
  fn odd_data() {
    assert_eq!(
      interpret("MedianDeviation[{1, 1, 2, 2, 4, 6, 9}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    assert_eq!(
      interpret("MedianDeviation[x]").unwrap(),
      "MedianDeviation[x]"
    );
  }
}

mod quartiles {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("Quartiles[Range[25]]").unwrap(),
      "{27/4, 13, 77/4}"
    );
  }

  #[test]
  fn small_list() {
    assert_eq!(
      interpret("Quartiles[{1, 2, 3, 4, 5}]").unwrap(),
      "{7/4, 3, 17/4}"
    );
  }
}
