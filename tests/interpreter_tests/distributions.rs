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

  // Quantile[CauchyDistribution[a, b], q] = a + b Tan[Pi (q - 1/2)].
  #[test]
  fn quantile_standard() {
    assert_eq!(
      interpret("Quantile[CauchyDistribution[0, 1], 1/2]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Quantile[CauchyDistribution[0, 1], 1/4]").unwrap(),
      "-1"
    );
    assert_eq!(
      interpret("Quantile[CauchyDistribution[0, 1], 3/4]").unwrap(),
      "1"
    );
    // A non-special probability keeps the exact Tan value.
    assert_eq!(
      interpret("Quantile[CauchyDistribution[0, 1], 1/3]").unwrap(),
      "-(1/Sqrt[3])"
    );
  }

  #[test]
  fn quantile_with_params() {
    // a + b Tan[-Pi/4] = 2 - 3 = -1.
    assert_eq!(
      interpret("Quantile[CauchyDistribution[2, 3], 1/4]").unwrap(),
      "-1"
    );
    assert_eq!(
      interpret("Quantile[CauchyDistribution[5, 2], 1/2]").unwrap(),
      "5"
    );
  }

  // The boundaries map to the infinite tails.
  #[test]
  fn quantile_boundaries() {
    assert_eq!(
      interpret("Quantile[CauchyDistribution[0, 1], 0]").unwrap(),
      "-Infinity"
    );
    assert_eq!(
      interpret("Quantile[CauchyDistribution[0, 1], 1]").unwrap(),
      "Infinity"
    );
  }
}

mod distribution_list_threading {
  use super::*;

  // CDF of a univariate distribution at a list of points threads, rather than
  // leaking the list into a Piecewise condition (the discrete-distribution
  // bug).
  #[test]
  fn cdf_threads_over_value_list() {
    assert_eq!(
      interpret("CDF[PoissonDistribution[3], {1, 2, 3}]").unwrap(),
      "{4/E^3, 17/(2*E^3), 13/E^3}"
    );
    // Scalar form unchanged.
    assert_eq!(
      interpret("CDF[PoissonDistribution[3], 2]").unwrap(),
      "17/(2*E^3)"
    );
  }

  // Quantile of a distribution at a list of probabilities threads.
  #[test]
  fn quantile_threads_over_probability_list() {
    assert_eq!(
      interpret("Quantile[ExponentialDistribution[1], {1/2, 3/4}]").unwrap(),
      "{Log[2], Log[4]}"
    );
    // Scalar form unchanged.
    assert_eq!(
      interpret("Quantile[ExponentialDistribution[1], 1/2]").unwrap(),
      "Log[2]"
    );
  }

  // PDF of a univariate (discrete) distribution at a list of points threads,
  // rather than leaking the list into a Piecewise condition.
  #[test]
  fn pdf_threads_over_value_list() {
    assert_eq!(
      interpret("PDF[PoissonDistribution[3], {1, 2, 3}]").unwrap(),
      "{3/E^3, 9/(2*E^3), 9/(2*E^3)}"
    );
    // Scalar form unchanged.
    assert_eq!(
      interpret("PDF[PoissonDistribution[3], 2]").unwrap(),
      "9/(2*E^3)"
    );
  }

  // InverseCDF threads over a list of probabilities.
  #[test]
  fn inverse_cdf_threads_over_probability_list() {
    assert_eq!(
      interpret("InverseCDF[NormalDistribution[], {1/4, 1/2}]").unwrap(),
      "{-(Sqrt[2]*InverseErfc[1/2]), 0}"
    );
  }

  // Operator (curried) forms: f[dist][x] == f[dist, x].
  #[test]
  fn operator_forms() {
    assert_eq!(
      interpret("CDF[PoissonDistribution[3]][2]").unwrap(),
      "17/(2*E^3)"
    );
    assert_eq!(
      interpret("PDF[NormalDistribution[]][0]").unwrap(),
      "1/Sqrt[2*Pi]"
    );
    assert_eq!(
      interpret("Quantile[NormalDistribution[]][1/2]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("SurvivalFunction[ExponentialDistribution[1]][2]").unwrap(),
      "E^(-2)"
    );
    // The operator form composes with value-list threading.
    assert_eq!(
      interpret("CDF[PoissonDistribution[3]][{1, 2, 3}]").unwrap(),
      "{4/E^3, 17/(2*E^3), 13/E^3}"
    );
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

  #[test]
  fn median_single_rate_collapses_to_exponential() {
    // HypoexponentialDistribution[{lambda}] = ExponentialDistribution[lambda].
    assert_eq!(
      interpret("Median[HypoexponentialDistribution[{2}]]").unwrap(),
      "Log[2]/2"
    );
  }

  #[test]
  fn median_two_rates_log2() {
    // For rates {2, 3}, F(Log[2]) = 1/2 exactly, so Median = Log[2].
    assert_eq!(
      interpret("Median[HypoexponentialDistribution[{2, 3}]]").unwrap(),
      "Log[2]"
    );
  }

  #[test]
  fn median_three_rates_log2() {
    // For rates {3, 4, 5}, the CDF at Log[2] is exactly 1/2.
    assert_eq!(
      interpret("Median[HypoexponentialDistribution[{3, 4, 5}]]").unwrap(),
      "Log[2]"
    );
  }

  #[test]
  fn median_rates_without_log2_root_stays_symbolic() {
    // For {3, 4} the CDF at Log[2] is 5/16 ≠ 1/2, so we don't have a
    // closed form here; Woxi leaves it unevaluated.
    let result =
      interpret("Median[HypoexponentialDistribution[{3, 4}]]").unwrap();
    assert!(
      result.contains("Median"),
      "expected unevaluated, got {}",
      result
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

  // wolframscript returns the mean in the expanded Apart form, not (1-p)/p.
  #[test]
  fn mean_symbolic() {
    assert_eq!(
      interpret("Mean[GeometricDistribution[p]]").unwrap(),
      "-1 + p^(-1)"
    );
  }

  #[test]
  fn variance() {
    assert_eq!(
      interpret("Variance[GeometricDistribution[1/3]]").unwrap(),
      "6"
    );
  }

  // The variance keeps the combined-fraction form (unlike the mean).
  #[test]
  fn variance_symbolic() {
    assert_eq!(
      interpret("Variance[GeometricDistribution[p]]").unwrap(),
      "(1 - p)/p^2"
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

mod standard_deviation_symbolic_distributions {
  use super::*;

  // Variance is sigma^2; the standard deviation must simplify Sqrt[sigma^2]
  // to sigma (distribution parameters are positive, so no Abs).
  #[test]
  fn normal_symbolic_sigma() {
    assert_eq!(
      interpret("StandardDeviation[NormalDistribution[mu, sigma]]").unwrap(),
      "sigma"
    );
  }

  // Variance is a bare symbol (lambda); the standard deviation is Sqrt[lambda].
  #[test]
  fn poisson_symbolic_lambda() {
    assert_eq!(
      interpret("StandardDeviation[PoissonDistribution[lambda]]").unwrap(),
      "Sqrt[lambda]"
    );
  }

  // Variance is (1-p)/p^2; the p^2 in the denominator comes out as 1/p.
  #[test]
  fn geometric_symbolic_p() {
    assert_eq!(
      interpret("StandardDeviation[GeometricDistribution[p]]").unwrap(),
      "Sqrt[1 - p]/p"
    );
  }

  // Variance is n*p*(1-p); nothing is a perfect square, so it stays under Sqrt.
  #[test]
  fn binomial_symbolic() {
    assert_eq!(
      interpret("StandardDeviation[BinomialDistribution[n, p]]").unwrap(),
      "Sqrt[n*(1 - p)*p]"
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

  #[test]
  fn weighted_symbolic() {
    // MovingAverage[list, {w1, w2}] = weighted average over a sliding
    // window of width 2; denominator is Sum[wi].
    assert_eq!(
      interpret("MovingAverage[{a, b, c, d, e}, {1, 2}]").unwrap(),
      "{(a + 2*b)/3, (b + 2*c)/3, (c + 2*d)/3, (d + 2*e)/3}"
    );
  }

  #[test]
  fn weighted_numeric() {
    // Concrete weights and values.
    assert_eq!(
      interpret("MovingAverage[{1, 2, 3, 4}, {1, 2}]").unwrap(),
      "{5/3, 8/3, 11/3}"
    );
  }

  #[test]
  fn weighted_three_taps() {
    // Three-tap weights {1, 1, 1} matches the unweighted window-3 case.
    assert_eq!(
      interpret("MovingAverage[{1, 2, 3, 4, 5}, {1, 1, 1}]").unwrap(),
      "{2, 3, 4}"
    );
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

  #[test]
  fn exponential_distribution_symbolic() {
    // Quartiles[ExponentialDistribution[λ]] uses the closed-form quantile
    // -Log[1-p]/λ. For p ∈ {1/4, 1/2, 3/4} this collapses to
    // {Log[4/3]/λ, Log[2]/λ, Log[4]/λ}.
    assert_eq!(
      interpret("Quartiles[ExponentialDistribution[a]]").unwrap(),
      "{Log[4/3]/a, Log[2]/a, Log[4]/a}"
    );
  }

  #[test]
  fn iqr_exponential_distribution_symbolic() {
    // InterquartileRange[ExponentialDistribution[λ]] = Log[3]/λ. Naive
    // subtraction Log[4]/λ - Log[4/3]/λ doesn't collapse symbolically in
    // Woxi, so this routes through a distribution-aware closed form.
    assert_eq!(
      interpret("InterquartileRange[ExponentialDistribution[a]]").unwrap(),
      "Log[3]/a"
    );
  }

  #[test]
  fn iqr_exponential_distribution_numeric() {
    // Numerical λ continues to use the Q3 - Q1 path.
    let result =
      interpret("InterquartileRange[ExponentialDistribution[2.]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.5493061443340549).abs() < 1e-12, "got {}", val);
  }

  #[test]
  fn quantile_exponential_distribution_symbolic() {
    // Direct Quantile calls should produce the same closed form.
    assert_eq!(
      interpret("Quantile[ExponentialDistribution[a], 1/4]").unwrap(),
      "Log[4/3]/a"
    );
    assert_eq!(
      interpret("Quantile[ExponentialDistribution[a], 1/2]").unwrap(),
      "Log[2]/a"
    );
    assert_eq!(
      interpret("Quantile[ExponentialDistribution[a], 3/4]").unwrap(),
      "Log[4]/a"
    );
  }
}

mod probability_distribution {
  use super::*;

  #[test]
  fn expectation_univariate() {
    // E[x, x ~ PD[2/y^3, {y, 1, ∞}]] = ∫ x * 2/x^3 dx from 1 to ∞ = 2.
    assert_eq!(
      interpret(
        "Expectation[x, x \\[Distributed] ProbabilityDistribution[2/y^3, {y, 1, Infinity}]]"
      )
      .unwrap(),
      "2"
    );
  }

  #[test]
  fn expectation_multivariate() {
    // E[x, {x,y} ~ PD[(10-x*y^2)/64, {x,2,10}, {y,0,1}]] = 52/9.
    // From the actuarial example notebook (cell #2).
    assert_eq!(
      interpret(
        "Expectation[x, {x, y} \\[Distributed] ProbabilityDistribution[(10 - x*y^2)/64, {x, 2, 10}, {y, 0, 1}]]"
      )
      .unwrap(),
      "52/9"
    );
  }

  #[test]
  fn pdf_with_renamed_variable() {
    // PD's iterator is `y`; PDF[..., 2] substitutes y → 2 and gates by domain.
    let result =
      interpret("PDF[ProbabilityDistribution[3/(1+y)^4, {y, 0, Infinity}], 2]")
        .unwrap();
    // (3/(1+2)^4) on the support → 3/81 → 1/27.
    assert!(result.contains("1/27"), "got {result}");
  }

  #[test]
  fn cdf_at_endpoint() {
    // CDF[..., 2] = ∫₀² 3/(1+y)^4 dy = 1 - 1/27 = 26/27.
    assert_eq!(
      interpret("CDF[ProbabilityDistribution[3/(1+y)^4, {y, 0, Infinity}], 2]")
        .unwrap(),
      "26/27"
    );
  }

  #[test]
  fn survival_function() {
    // S(t) = 1 - CDF.
    assert_eq!(
      interpret(
        "SurvivalFunction[ProbabilityDistribution[3/(1+y)^4, {y, 0, Infinity}], 2]"
      )
      .unwrap(),
      "1/27"
    );
  }

  #[test]
  fn probability_greater_than() {
    // P[x > 2] for the same PD = SurvivalFunction at 2.
    assert_eq!(
      interpret(
        "Probability[x > 2, x \\[Distributed] ProbabilityDistribution[3/(1+y)^4, {y, 0, Infinity}]]"
      )
      .unwrap(),
      "1/27"
    );
  }

  #[test]
  fn probability_less_than() {
    assert_eq!(
      interpret(
        "Probability[x < 2, x \\[Distributed] ProbabilityDistribution[3/(1+y)^4, {y, 0, Infinity}]]"
      )
      .unwrap(),
      "26/27"
    );
  }

  #[test]
  fn probability_bounded() {
    // P[1 < x < 3] = CDF[3] - CDF[1] = (1 - 1/64) - (1 - 1/8) = 1/8 - 1/64 = 7/64.
    assert_eq!(
      interpret(
        "Probability[1 < x < 3, x \\[Distributed] ProbabilityDistribution[3/(1+y)^4, {y, 0, Infinity}]]"
      )
      .unwrap(),
      "7/64"
    );
  }

  #[test]
  fn n_expectation_routes_through_n() {
    // `NExpectation` should wrap `Expectation`'s result with `N`. For a
    // closed-form result like 2 (from the 2/y^3 distribution) the numeric
    // output is `2.`.
    let result = interpret(
      "NExpectation[x, x \\[Distributed] ProbabilityDistribution[2/y^3, {y, 1, Infinity}]]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.0).abs() < 1e-12, "got {val}");
  }

  #[test]
  fn n_probability_numeric() {
    let result = interpret(
      "NProbability[x > 10, x \\[Distributed] ProbabilityDistribution[3/(1+y)^4, {y, 0, Infinity}]]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    // P[x > 10] = (1/11)^3 = 1/1331 ≈ 0.000751…
    assert!((val - 1.0 / 1331.0).abs() < 1e-12, "got {val}");
  }

  #[test]
  fn quantile_numeric() {
    // Quantile is the inverse CDF — for a continuous PD we bisect.
    let result = interpret(
      "Quantile[ProbabilityDistribution[3/(1+y)^4, {y, 0, Infinity}], 0.95]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    // CDF(t) = 1 - 1/(1+t)^3 = 0.95 → 1+t = 20^(1/3) → t ≈ 1.71441...
    assert!((val - 1.7144176165949045).abs() < 1e-9, "got {val}");
  }
}

mod censored_distribution_expectation {
  use super::*;

  #[test]
  fn pareto_censored_at_10() {
    // From actuarial example notebook (cell #8):
    // lossDist = PD[2/y^3, {y, 1, ∞}]; censoredLoss = CensoredDistribution[{1, 10}, lossDist];
    // E[x ~ censoredLoss] = ∫₁¹⁰ x · 2/x³ dx + 10 · P(Y > 10) = 9/5 + 10·1/100 = 19/10.
    assert_eq!(
      interpret(
        "Expectation[x, x \\[Distributed] CensoredDistribution[{1, 10}, ProbabilityDistribution[2/y^3, {y, 1, Infinity}]]]"
      )
      .unwrap(),
      "19/10"
    );
  }

  #[test]
  fn n_expectation_matches_floating_point() {
    let result = interpret(
      "NExpectation[x, x \\[Distributed] CensoredDistribution[{1, 10}, ProbabilityDistribution[2/y^3, {y, 1, Infinity}]]]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.9).abs() < 1e-9, "got {val}");
  }
}

mod conditioned_operator {
  use super::*;

  #[test]
  fn parses_as_conditioned_function_call() {
    // `\[Conditioned]` is the typeset name for the conditional-probability operator.
    assert_eq!(
      interpret("a \\[Conditioned] b").unwrap(),
      "Conditioned[a, b]"
    );
  }

  #[test]
  fn binds_looser_than_comparison() {
    // m > 40000 \[Conditioned] m > 10000 → Conditioned[m > 40000, m > 10000]
    // (the comparisons bind tighter than the conditional operator).
    assert_eq!(
      interpret("m > 40000 \\[Conditioned] m > 10000").unwrap(),
      "Conditioned[m > 40000, m > 10000]"
    );
  }

  #[test]
  fn probability_conditional() {
    // P[a > 1/2 | a > 1/4] over PD with uniform-like pdf 1 on [0,1].
    // = P[a > 1/2 ∧ a > 1/4] / P[a > 1/4] = (1/2) / (3/4) = 2/3.
    let result = interpret(
      "Probability[a > 1/2 \\[Conditioned] a > 1/4, a \\[Distributed] ProbabilityDistribution[1, {y, 0, 1}]]",
    )
    .unwrap();
    assert_eq!(result, "2/3");
  }
}

mod solve_values {
  use super::*;

  #[test]
  fn single_variable_quadratic() {
    // SolveValues drops the rule wrapping that Solve produces.
    assert_eq!(interpret("SolveValues[m^2 == 4, m]").unwrap(), "{-2, 2}");
  }

  #[test]
  fn with_domain_restriction() {
    assert_eq!(
      interpret("SolveValues[m^2 == 4 && m > 0, m]").unwrap(),
      "{2}"
    );
  }

  #[test]
  fn first_of_solve_values() {
    // Common pattern from the actuarial example notebook: pluck the
    // positive root with `First @ SolveValues[…]`.
    assert_eq!(
      interpret("First @ SolveValues[m^2 == 4 && m > 0, m]").unwrap(),
      "2"
    );
  }
}

mod integrate_linear_power {
  use super::*;

  #[test]
  fn negative_integer_exponent_indefinite() {
    // ∫ 1/(1+x)^4 dx = -1/(3*(1+x)^3) — Wolfram keeps this factored.
    // We accept any equivalent factored shape.
    let result = interpret("Integrate[1/(1+x)^4, x]").unwrap();
    assert!(
      result.contains("(1 + x)^3") || result.contains("(1 + x)^(-3)"),
      "expected a factored (1+x)^3 form, got: {result}"
    );
  }

  #[test]
  fn negative_integer_exponent_definite() {
    // The case that motivated the change: PD with `1/(1+y)^4` integrand
    // for SurvivalFunction / NProbability needs to be evaluable.
    assert_eq!(
      interpret("Integrate[3/(1+x)^4, {x, 10, Infinity}]").unwrap(),
      "1/1331"
    );
  }

  #[test]
  fn small_positive_integer_still_expands() {
    // Regression: my new linear-power case must not steal the
    // `(2x-1)^2` path away from the Expand-then-integrate fallback,
    // because Wolfram emits the expanded form for small `n`.
    assert_eq!(
      interpret("Integrate[(2*x - 1)^2, x]").unwrap(),
      "x - 2*x^2 + (4*x^3)/3"
    );
  }
}

mod find_distribution_parameters {
  use super::*;

  #[test]
  fn laplace_mle() {
    // μ = median = (1.5 + 2.8)/2 = 2.15
    // σ = mean(|x_i - μ|) = (1.65 + 0.65 + 0.65 + 2.15)/4 = 1.275
    assert_eq!(
      interpret(
        "FindDistributionParameters[{1.5, 2.8, 4.3, 0.5}, \
         LaplaceDistribution[mu, sigma]]"
      )
      .unwrap(),
      "{mu -> 2.15, sigma -> 1.275}"
    );
  }

  #[test]
  fn normal_mle() {
    // μ = mean = 2.275
    // σ = sqrt(variance) with denominator n
    assert_eq!(
      interpret(
        "FindDistributionParameters[{1.5, 2.8, 4.3, 0.5}, \
         NormalDistribution[mu, sigma]]"
      )
      .unwrap(),
      "{mu -> 2.275, sigma -> 1.4254385290148432}"
    );
  }

  #[test]
  fn laplace_odd_count() {
    // Median of {1, 2, 3, 4, 5} is 3.
    // |1-3|+|2-3|+|3-3|+|4-3|+|5-3| = 6. b = 6/5 = 1.2.
    // wolframscript coerces integer-data results to Real, so check 3.
    assert_eq!(
      interpret(
        "FindDistributionParameters[{1, 2, 3, 4, 5}, \
         LaplaceDistribution[m, b]]"
      )
      .unwrap(),
      "{m -> 3., b -> 1.2}"
    );
  }

  #[test]
  fn unknown_distribution_stays_symbolic() {
    assert_eq!(
      interpret("FindDistributionParameters[{1, 2, 3}, FooDistribution[a, b]]")
        .unwrap(),
      "FindDistributionParameters[{1, 2, 3}, FooDistribution[a, b]]"
    );
  }
}

mod parameter_mixture_distribution {
  use super::*;

  #[test]
  fn binomial_beta_reduces_to_beta_binomial() {
    assert_eq!(
      interpret(
        "ParameterMixtureDistribution[BinomialDistribution[n, p], \
         Distributed[p, BetaDistribution[a, b]]]"
      )
      .unwrap(),
      "BetaBinomialDistribution[a, b, n]"
    );
  }

  #[test]
  fn binomial_beta_with_symbolic_n() {
    assert_eq!(
      interpret(
        "ParameterMixtureDistribution[BinomialDistribution[5, p], \
         Distributed[p, BetaDistribution[alpha, beta]]]"
      )
      .unwrap(),
      "BetaBinomialDistribution[alpha, beta, 5]"
    );
  }

  #[test]
  fn unknown_mixture_stays_symbolic() {
    assert_eq!(
      interpret(
        "ParameterMixtureDistribution[NormalDistribution[mu, sigma], \
         Distributed[mu, UniformDistribution[{0, 1}]]]"
      )
      .unwrap(),
      "ParameterMixtureDistribution[NormalDistribution[mu, sigma], \
       Distributed[mu, UniformDistribution[{0, 1}]]]"
    );
  }
}

mod survival_function_symbolic {
  use super::*;

  #[test]
  fn exponential_piecewise() {
    // Regression: used to print the unsimplified 1 - Piecewise[...] form
    assert_eq!(
      interpret("SurvivalFunction[ExponentialDistribution[a], x]").unwrap(),
      "Piecewise[{{E^(-(a*x)), x >= 0}}, 1]"
    );
  }

  #[test]
  fn uniform_piecewise() {
    assert_eq!(
      interpret("SurvivalFunction[UniformDistribution[{0, 1}], x]").unwrap(),
      "Piecewise[{{1 - x, 0 <= x <= 1}, {0, x > 1}}, 1]"
    );
  }

  #[test]
  fn geometric_piecewise() {
    assert_eq!(
      interpret("SurvivalFunction[GeometricDistribution[1/3], x]").unwrap(),
      "Piecewise[{{(2/3)^(1 + Floor[x]), x >= 0}}, 1]"
    );
  }

  #[test]
  fn normal_erfc_reflection() {
    // 1 - Erfc[z]/2 folds to Erfc[-z]/2 with the argument normalized
    assert_eq!(
      interpret("SurvivalFunction[NormalDistribution[], x]").unwrap(),
      "Erfc[x/Sqrt[2]]/2"
    );
    assert_eq!(
      interpret("SurvivalFunction[NormalDistribution[m, s], x]").unwrap(),
      "Erfc[(-m + x)/(Sqrt[2]*s)]/2"
    );
  }

  #[test]
  fn numeric_argument_still_folds() {
    assert_eq!(
      interpret("SurvivalFunction[ExponentialDistribution[2], 3]").unwrap(),
      "E^(-6)"
    );
  }
}

mod hazard_function {
  use super::*;

  #[test]
  fn exponential_constant_hazard() {
    assert_eq!(
      interpret("HazardFunction[ExponentialDistribution[a], x]").unwrap(),
      "Piecewise[{{a, x >= 0}}, 0]"
    );
  }

  #[test]
  fn pareto() {
    assert_eq!(
      interpret("HazardFunction[ParetoDistribution[k, a], x]").unwrap(),
      "Piecewise[{{a/x, x >= k}}, 0]"
    );
  }

  #[test]
  fn standard_normal() {
    assert_eq!(
      interpret("HazardFunction[NormalDistribution[], x]").unwrap(),
      "Sqrt[2/Pi]/(E^(x^2/2)*Erfc[x/Sqrt[2]])"
    );
  }

  #[test]
  fn numeric_argument() {
    assert_eq!(
      interpret("HazardFunction[ExponentialDistribution[2], 3]").unwrap(),
      "2"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("HazardFunction[PoissonDistribution[2], x]").unwrap(),
      "HazardFunction[PoissonDistribution[2], x]"
    );
  }
}

mod inverse_cdf {
  use super::*;

  #[test]
  fn exponential() {
    assert_eq!(
      interpret("InverseCDF[ExponentialDistribution[2], 1/2]").unwrap(),
      "Log[2]/2"
    );
    assert_eq!(
      interpret("InverseCDF[ExponentialDistribution[a], 1/4]").unwrap(),
      "Log[4/3]/a"
    );
  }

  #[test]
  fn exponential_edges() {
    assert_eq!(
      interpret("InverseCDF[ExponentialDistribution[a], 1]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("InverseCDF[ExponentialDistribution[a], 0]").unwrap(),
      "0"
    );
  }

  #[test]
  fn uniform() {
    assert_eq!(
      interpret("InverseCDF[UniformDistribution[{0, 10}], 1/4]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("InverseCDF[UniformDistribution[{a, b}], 1/4]").unwrap(),
      "(3*a)/4 + b/4"
    );
  }

  #[test]
  fn normal_median() {
    assert_eq!(
      interpret("InverseCDF[NormalDistribution[0, 1], 1/2]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("InverseCDF[NormalDistribution[m, s], 1/2]").unwrap(),
      "m"
    );
  }

  #[test]
  fn normal_symbolic_quartile() {
    assert_eq!(
      interpret("InverseCDF[NormalDistribution[0, 1], 1/4]").unwrap(),
      "-(Sqrt[2]*InverseErfc[1/2])"
    );
    assert_eq!(
      interpret("InverseCDF[NormalDistribution[m, s], 1/4]").unwrap(),
      "m - Sqrt[2]*s*InverseErfc[1/2]"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("InverseCDF[x, 1/2]").unwrap(),
      "InverseCDF[x, 1/2]"
    );
  }
}

mod quantile_distribution_extended {
  use super::*;

  #[test]
  fn quantile_matches_inverse_cdf() {
    // Quantile gained the same closed forms
    assert_eq!(
      interpret("Quantile[NormalDistribution[0, 1], 1/2]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Quantile[UniformDistribution[{a, b}], 1/4]").unwrap(),
      "(3*a)/4 + b/4"
    );
    assert_eq!(
      interpret("Quantile[NormalDistribution[m, s], 1/4]").unwrap(),
      "m - Sqrt[2]*s*InverseErfc[1/2]"
    );
    // Upper quantile (q > 1/2): InverseErfc[2q] reflects so the result uses
    // +InverseErfc[2-2q] rather than the unreduced InverseErfc[2q].
    assert_eq!(
      interpret("Quantile[NormalDistribution[0, 1], 3/4]").unwrap(),
      "Sqrt[2]*InverseErfc[1/2]"
    );
    assert_eq!(
      interpret("Quantile[NormalDistribution[m, s], 3/4]").unwrap(),
      "m + Sqrt[2]*s*InverseErfc[1/2]"
    );
    assert_eq!(
      interpret("Quartiles[NormalDistribution[0, 1]]").unwrap(),
      "{-(Sqrt[2]*InverseErfc[1/2]), 0, Sqrt[2]*InverseErfc[1/2]}"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[NormalDistribution[0, 1], 3/4]")
        .unwrap(),
      "-(Sqrt[2]*InverseErfc[1/2])"
    );
  }

  #[test]
  fn exponential_at_one_is_infinity() {
    // Regression: used to produce the unreduced Infinity/a
    assert_eq!(
      interpret("Quantile[ExponentialDistribution[a], 1]").unwrap(),
      "Infinity"
    );
  }
}

mod gamma_family_quantile {
  use super::*;

  // Quantile[GammaDistribution[a, b], q] = b InverseGammaRegularized[a, 0, q].
  #[test]
  fn gamma_closed_form() {
    assert_eq!(
      interpret("Quantile[GammaDistribution[2, 3], 1/2]").unwrap(),
      "3*InverseGammaRegularized[2, 0, 1/2]"
    );
    // Scale b == 1 drops the leading factor.
    assert_eq!(
      interpret("Quantile[GammaDistribution[2, 1], 1/2]").unwrap(),
      "InverseGammaRegularized[2, 0, 1/2]"
    );
    // InverseCDF shares the same closed form.
    assert_eq!(
      interpret("InverseCDF[GammaDistribution[2, 3], 1/2]").unwrap(),
      "3*InverseGammaRegularized[2, 0, 1/2]"
    );
  }

  #[test]
  fn gamma_edges() {
    assert_eq!(
      interpret("Quantile[GammaDistribution[2, 3], 0]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Quantile[GammaDistribution[2, 3], 1]").unwrap(),
      "Infinity"
    );
  }

  // ChiSquareDistribution[v] = GammaDistribution[v/2, 2], quantile is
  // 2 InverseGammaRegularized[v/2, 0, q].
  #[test]
  fn chi_square_closed_form() {
    assert_eq!(
      interpret("Quantile[ChiSquareDistribution[4], 1/2]").unwrap(),
      "2*InverseGammaRegularized[2, 0, 1/2]"
    );
    assert_eq!(
      interpret("InverseCDF[ChiSquareDistribution[3], 1/2]").unwrap(),
      "2*InverseGammaRegularized[3/2, 0, 1/2]"
    );
  }

  // Median[dist] = Quantile[dist, 1/2].
  #[test]
  fn gamma_family_median() {
    assert_eq!(
      interpret("Median[GammaDistribution[2, 3]]").unwrap(),
      "3*InverseGammaRegularized[2, 0, 1/2]"
    );
    assert_eq!(
      interpret("Median[ChiSquareDistribution[3]]").unwrap(),
      "2*InverseGammaRegularized[3/2, 0, 1/2]"
    );
  }

  // Beta edges are exact; the interior exact case stays unevaluated (a Root in
  // wolframscript), but the numeric path evaluates.
  #[test]
  fn beta_edges() {
    assert_eq!(
      interpret("Quantile[BetaDistribution[2, 3], 0]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Quantile[BetaDistribution[2, 3], 1]").unwrap(),
      "1"
    );
  }

  #[test]
  fn numeric_quantiles() {
    let g: f64 = interpret("N[Quantile[GammaDistribution[2, 3], 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((g - 5.035040970049984).abs() < 1e-8, "got {g}");

    let c: f64 = interpret("N[InverseCDF[ChiSquareDistribution[3], 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((c - 2.3659738843753386).abs() < 1e-8, "got {c}");

    let b: f64 = interpret("N[Quantile[BetaDistribution[2, 3], 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((b - 0.38572756813238956).abs() < 1e-8, "got {b}");

    let m: f64 = interpret("N[Median[GammaDistribution[2, 3]]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((m - 5.035040970049984).abs() < 1e-8, "got {m}");
  }
}

mod student_t_quantile {
  use super::*;

  // Quantile[StudentTDistribution[nu], q] =
  //   Sqrt[nu (1/InverseBetaRegularized[2(1-q), nu/2, 1/2] - 1)] for q > 1/2.
  #[test]
  fn closed_form_upper() {
    assert_eq!(
      interpret("Quantile[StudentTDistribution[5], 3/4]").unwrap(),
      "Sqrt[5*(-1 + InverseBetaRegularized[1/2, 5/2, 1/2]^(-1))]"
    );
    assert_eq!(
      interpret("Quantile[StudentTDistribution[3], 9/10]").unwrap(),
      "Sqrt[3*(-1 + InverseBetaRegularized[1/5, 3/2, 1/2]^(-1))]"
    );
    // InverseCDF shares the closed form.
    assert_eq!(
      interpret("InverseCDF[StudentTDistribution[5], 3/4]").unwrap(),
      "Sqrt[5*(-1 + InverseBetaRegularized[1/2, 5/2, 1/2]^(-1))]"
    );
  }

  // Below the median the result is negated (the distribution is symmetric).
  #[test]
  fn closed_form_lower() {
    assert_eq!(
      interpret("Quantile[StudentTDistribution[5], 1/4]").unwrap(),
      "-Sqrt[5*(-1 + InverseBetaRegularized[1/2, 5/2, 1/2]^(-1))]"
    );
  }

  #[test]
  fn median_and_edges() {
    assert_eq!(
      interpret("Quantile[StudentTDistribution[5], 1/2]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Quantile[StudentTDistribution[5], 0]").unwrap(),
      "-Infinity"
    );
    assert_eq!(
      interpret("Quantile[StudentTDistribution[5], 1]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn numeric_values() {
    let u: f64 = interpret("N[Quantile[StudentTDistribution[5], 3/4]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((u - 0.7266868438004223).abs() < 1e-8, "got {u}");

    let l: f64 = interpret("N[Quantile[StudentTDistribution[5], 1/4]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((l + 0.7266868438004223).abs() < 1e-8, "got {l}");

    let h: f64 = interpret("N[Quantile[StudentTDistribution[3], 9/10]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((h - 1.6377443536962104).abs() < 1e-8, "got {h}");
  }
}

mod f_ratio_quantile {
  use super::*;

  // Quantile[FRatioDistribution[n, m], q] =
  //   (m/n) (1/InverseBetaRegularized[1, -q, m/2, n/2] - 1).
  #[test]
  fn closed_form() {
    assert_eq!(
      interpret("Quantile[FRatioDistribution[3, 5], 1/2]").unwrap(),
      "(5*(-1 + InverseBetaRegularized[1, -1/2, 5/2, 3/2]^(-1)))/3"
    );
    assert_eq!(
      interpret("Quantile[FRatioDistribution[3, 5], 1/4]").unwrap(),
      "(5*(-1 + InverseBetaRegularized[1, -1/4, 5/2, 3/2]^(-1)))/3"
    );
    // InverseCDF shares the closed form.
    assert_eq!(
      interpret("InverseCDF[FRatioDistribution[3, 5], 1/2]").unwrap(),
      "(5*(-1 + InverseBetaRegularized[1, -1/2, 5/2, 3/2]^(-1)))/3"
    );
  }

  #[test]
  fn edges() {
    assert_eq!(
      interpret("Quantile[FRatioDistribution[3, 5], 0]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Quantile[FRatioDistribution[3, 5], 1]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn numeric_values() {
    let h: f64 = interpret("N[Quantile[FRatioDistribution[3, 5], 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((h - 0.9071462198190188).abs() < 1e-8, "got {h}");

    let l: f64 = interpret("N[Quantile[FRatioDistribution[3, 5], 1/4]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((l - 0.41502457605411).abs() < 1e-8, "got {l}");
  }
}

mod f_ratio_pdf_cdf {
  use super::*;

  // PDF[FRatioDistribution[n, m], x] =
  //   n^(n/2) m^(m/2) x^(n/2-1) / ((m + n x)^((n+m)/2) Beta[n/2, m/2]) for x > 0.
  #[test]
  fn pdf_symbolic() {
    assert_eq!(
      interpret("PDF[FRatioDistribution[3, 5], x]").unwrap(),
      "Piecewise[{{(1200*Sqrt[15]*Sqrt[x])/(Pi*(5 + 3*x)^4), x > 0}}, 0]"
    );
    // Even degrees of freedom give a rational density (no Sqrt, no Pi).
    assert_eq!(
      interpret("PDF[FRatioDistribution[2, 4], x]").unwrap(),
      "Piecewise[{{64/(4 + 2*x)^3, x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[FRatioDistribution[4, 6], x]").unwrap(),
      "Piecewise[{{(41472*x)/(6 + 4*x)^5, x > 0}}, 0]"
    );
    // Half-integer total degrees of freedom keep a half-integer exponent.
    assert_eq!(
      interpret("PDF[FRatioDistribution[3, 4], x]").unwrap(),
      "Piecewise[{{(180*Sqrt[3]*Sqrt[x])/(4 + 3*x)^(7/2), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[FRatioDistribution[1, 1], x]").unwrap(),
      "Piecewise[{{1/(Pi*Sqrt[x]*(1 + x)), x > 0}}, 0]"
    );
  }

  #[test]
  fn pdf_numeric() {
    assert_eq!(
      interpret("PDF[FRatioDistribution[3, 5], 1]").unwrap(),
      "(75*Sqrt[15])/(256*Pi)"
    );
    assert_eq!(
      interpret("PDF[FRatioDistribution[3, 5], 2]").unwrap(),
      "(1200*Sqrt[30])/(14641*Pi)"
    );
    // Outside the support the density is zero.
    assert_eq!(interpret("PDF[FRatioDistribution[3, 5], 0]").unwrap(), "0");
    assert_eq!(interpret("PDF[FRatioDistribution[3, 5], -1]").unwrap(), "0");
  }

  // CDF[FRatioDistribution[n, m], x] =
  //   BetaRegularized[n x / (n x + m), n/2, m/2] for x > 0.
  #[test]
  fn cdf() {
    assert_eq!(
      interpret("CDF[FRatioDistribution[3, 5], x]").unwrap(),
      "Piecewise[{{BetaRegularized[(3*x)/(5 + 3*x), 3/2, 5/2], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[FRatioDistribution[3, 5], 1]").unwrap(),
      "BetaRegularized[3/8, 3/2, 5/2]"
    );
    assert_eq!(interpret("CDF[FRatioDistribution[3, 5], 0]").unwrap(), "0");
  }
}

mod waring_yule_distribution {
  use super::*;

  // PDF[WaringYuleDistribution[a, b], k] =
  //   a Pochhammer[b, k] / Pochhammer[a + b, 1 + k] for k >= 0.
  #[test]
  fn pdf_symbolic() {
    assert_eq!(
      interpret("PDF[WaringYuleDistribution[a, b], k]").unwrap(),
      "Piecewise[{{(a*Pochhammer[b, k])/Pochhammer[a + b, 1 + k], k >= 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[WaringYuleDistribution[3, 2], k]").unwrap(),
      "Piecewise[{{(3*Pochhammer[2, k])/Pochhammer[5, 1 + k], k >= 0}}, 0]"
    );
  }

  #[test]
  fn pdf_numeric() {
    assert_eq!(
      interpret("PDF[WaringYuleDistribution[3, 2], 0]").unwrap(),
      "3/5"
    );
    assert_eq!(
      interpret("PDF[WaringYuleDistribution[3, 2], 1]").unwrap(),
      "1/5"
    );
    assert_eq!(
      interpret("PDF[WaringYuleDistribution[3, 2], 2]").unwrap(),
      "3/35"
    );
    // Different parameters.
    assert_eq!(
      interpret("PDF[WaringYuleDistribution[4, 3], 1]").unwrap(),
      "3/14"
    );
  }

  // CDF[WaringYuleDistribution[a, b], k] =
  //   1 - Pochhammer[b, 1 + Floor[k]] / Pochhammer[a + b, 1 + Floor[k]] for k >= 0.
  #[test]
  fn cdf() {
    assert_eq!(
      interpret("CDF[WaringYuleDistribution[3, 2], k]").unwrap(),
      "Piecewise[{{1 - Pochhammer[2, 1 + Floor[k]]/Pochhammer[5, 1 + Floor[k]], k >= 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[WaringYuleDistribution[3, 2], 0]").unwrap(),
      "3/5"
    );
    assert_eq!(
      interpret("CDF[WaringYuleDistribution[3, 2], 2]").unwrap(),
      "31/35"
    );
  }
}

mod polynomial_expectation {
  use super::*;

  #[test]
  fn exponential_moments() {
    // Regression: the numerical fallback truncated the domain at 10/lambda
    // and returned 0.7422479289 instead of 3/4
    assert_eq!(
      interpret(
        "Expectation[x^3, x \\[Distributed] ExponentialDistribution[2]]"
      )
      .unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret(
        "Expectation[x^4, x \\[Distributed] ExponentialDistribution[a]]"
      )
      .unwrap(),
      "24/a^4"
    );
    assert_eq!(
      interpret(
        "Expectation[x^5, x \\[Distributed] ExponentialDistribution[2]]"
      )
      .unwrap(),
      "15/4"
    );
  }

  #[test]
  fn normal_moments() {
    assert_eq!(
      interpret("Expectation[x^3, x \\[Distributed] NormalDistribution[m, s]]")
        .unwrap(),
      "m*(m^2 + 3*s^2)"
    );
    assert_eq!(
      interpret("Expectation[x^4, x \\[Distributed] NormalDistribution[m, s]]")
        .unwrap(),
      "m^4 + 6*m^2*s^2 + 3*s^4"
    );
    assert_eq!(
      interpret("Expectation[x^4, x \\[Distributed] NormalDistribution[0, 1]]")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Expectation[x^6, x \\[Distributed] NormalDistribution[0, 1]]")
        .unwrap(),
      "15"
    );
  }

  #[test]
  fn uniform_moments() {
    assert_eq!(
      interpret(
        "Expectation[x^3, x \\[Distributed] UniformDistribution[{0, 1}]]"
      )
      .unwrap(),
      "1/4"
    );
    assert_eq!(
      interpret(
        "Expectation[x^3, x \\[Distributed] UniformDistribution[{a, b}]]"
      )
      .unwrap(),
      "(a^3 + a^2*b + a*b^2 + b^3)/4"
    );
  }

  #[test]
  fn gamma_and_poisson_moments() {
    assert_eq!(
      interpret("Expectation[x^3, x \\[Distributed] GammaDistribution[a, b]]")
        .unwrap(),
      "a*(1 + a)*(2 + a)*b^3"
    );
    assert_eq!(
      interpret("Expectation[x^3, x \\[Distributed] PoissonDistribution[m]]")
        .unwrap(),
      "m + 3*m^2 + m^3"
    );
    assert_eq!(
      interpret("Expectation[x^4, x \\[Distributed] PoissonDistribution[m]]")
        .unwrap(),
      "m + 7*m^2 + 6*m^3 + m^4"
    );
  }

  #[test]
  fn bernoulli_moments() {
    // The support is {0, 1}, so x^k = x and every raw moment equals p.
    // Previously E[x^k] for k >= 3 fell through to numerical integration and
    // returned the call unevaluated for a symbolic parameter.
    assert_eq!(
      interpret("Expectation[x^3, x \\[Distributed] BernoulliDistribution[p]]")
        .unwrap(),
      "p"
    );
    assert_eq!(
      interpret("Moment[BernoulliDistribution[p], 3]").unwrap(),
      "p"
    );
    assert_eq!(
      interpret("Moment[BernoulliDistribution[p], 5]").unwrap(),
      "p"
    );
    assert_eq!(
      interpret("Moment[BernoulliDistribution[1/3], 4]").unwrap(),
      "1/3"
    );
  }

  #[test]
  fn polynomial_combination() {
    assert_eq!(
      interpret(
        "Expectation[2 x^3 + x - 5, x \\[Distributed] ExponentialDistribution[2]]"
      )
      .unwrap(),
      "-3"
    );
    assert_eq!(
      interpret(
        "N[Expectation[x^3 + x, x \\[Distributed] ExponentialDistribution[2]]]"
      )
      .unwrap(),
      "1.25"
    );
  }
}

mod transformed_distribution {
  use super::*;

  #[test]
  fn normal_linear() {
    assert_eq!(
      interpret(
        "TransformedDistribution[2 x, x \\[Distributed] NormalDistribution[0, 1]]"
      )
      .unwrap(),
      "NormalDistribution[0, 2]"
    );
    assert_eq!(
      interpret(
        "TransformedDistribution[x + 3, x \\[Distributed] NormalDistribution[m, s]]"
      )
      .unwrap(),
      "NormalDistribution[3 + m, s]"
    );
    assert_eq!(
      interpret(
        "TransformedDistribution[2 x + 1, x \\[Distributed] NormalDistribution[m, s]]"
      )
      .unwrap(),
      "NormalDistribution[1 + 2*m, 2*s]"
    );
    // Negative scale: the deviation stays positive
    assert_eq!(
      interpret(
        "TransformedDistribution[-x, x \\[Distributed] NormalDistribution[m, s]]"
      )
      .unwrap(),
      "NormalDistribution[-m, s]"
    );
  }

  #[test]
  fn uniform_linear() {
    assert_eq!(
      interpret(
        "TransformedDistribution[x + 2, x \\[Distributed] UniformDistribution[{0, 1}]]"
      )
      .unwrap(),
      "UniformDistribution[{2, 3}]"
    );
    assert_eq!(
      interpret(
        "TransformedDistribution[2 x, x \\[Distributed] UniformDistribution[{0, 1}]]"
      )
      .unwrap(),
      "UniformDistribution[{0, 2}]"
    );
    // Negative scale sorts the bounds
    assert_eq!(
      interpret(
        "TransformedDistribution[-2 x, x \\[Distributed] UniformDistribution[{0, 1}]]"
      )
      .unwrap(),
      "UniformDistribution[{-2, 0}]"
    );
  }

  #[test]
  fn exponential_and_gamma_scaling() {
    assert_eq!(
      interpret(
        "TransformedDistribution[3 x, x \\[Distributed] ExponentialDistribution[a]]"
      )
      .unwrap(),
      "ExponentialDistribution[a/3]"
    );
    assert_eq!(
      interpret(
        "TransformedDistribution[2 x, x \\[Distributed] GammaDistribution[a, b]]"
      )
      .unwrap(),
      "GammaDistribution[a, 2*b]"
    );
  }

  #[test]
  fn composes_with_mean() {
    assert_eq!(
      interpret(
        "Mean[TransformedDistribution[2 x + 1, x \\[Distributed] NormalDistribution[m, s]]]"
      )
      .unwrap(),
      "1 + 2*m"
    );
  }
}

mod multinormal_distribution {
  use super::*;

  #[test]
  fn pdf_identity_covariance() {
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}], {x, y}]"
      )
      .unwrap(),
      "E^((-x^2 - y^2)/2)/(2*Pi)"
    );
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0, 0}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}], {x, y, z}]"
      )
      .unwrap(),
      "E^((-x^2 - y^2 - z^2)/2)/(2*Sqrt[2]*Pi^(3/2))"
    );
  }

  #[test]
  fn pdf_diagonal_covariance() {
    // wolframscript's term styles are position-dependent: the first
    // scaled term prints -1/q*v^2, later ones -v^2/q
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0}, {{2, 0}, {0, 3}}], {x, y}]"
      )
      .unwrap(),
      "E^((-1/2*x^2 - y^2/3)/2)/(2*Sqrt[6]*Pi)"
    );
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0}, {{3, 0}, {0, 2}}], {x, y}]"
      )
      .unwrap(),
      "E^((-1/3*x^2 - y^2/2)/2)/(2*Sqrt[6]*Pi)"
    );
  }

  #[test]
  fn pdf_shifted_mean() {
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{1, 2}, {{1, 0}, {0, 1}}], {x, y}]"
      )
      .unwrap(),
      "E^((-(-1 + x)^2 - (-2 + y)^2)/2)/(2*Pi)"
    );
  }

  // At a fully numeric point the PDF must reduce to a closed value rather than
  // leaving pieces like 0^2 or 1^2 unevaluated in the exponent.
  #[test]
  fn pdf_numeric_point_reduces() {
    // Standard bivariate normal at the origin.
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}], {0, 0}]"
      )
      .unwrap(),
      "1/(2*Pi)"
    );
    // Off the mean.
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}], {1, 1}]"
      )
      .unwrap(),
      "1/(2*E*Pi)"
    );
    // Scaled diagonal variances, evaluated at the mean.
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{1, 2}, {{2, 0}, {0, 3}}], {1, 2}]"
      )
      .unwrap(),
      "1/(2*Sqrt[6]*Pi)"
    );
    // Rational point.
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}], {1/2, 1/2}]"
      )
      .unwrap(),
      "1/(2*E^(1/4)*Pi)"
    );
    // Trivariate at the origin.
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0, 0}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}], {0, 0, 0}]"
      )
      .unwrap(),
      "1/(2*Sqrt[2]*Pi^(3/2))"
    );
  }

  #[test]
  fn statistics() {
    assert_eq!(
      interpret("Mean[MultinormalDistribution[{a, b}, {{1, 0}, {0, 1}}]]")
        .unwrap(),
      "{a, b}"
    );
    assert_eq!(
      interpret(
        "Covariance[MultinormalDistribution[{0, 0}, {{2, 1}, {1, 3}}]]"
      )
      .unwrap(),
      "{{2, 1}, {1, 3}}"
    );
    assert_eq!(
      interpret("Variance[MultinormalDistribution[{0, 0}, {{2, 1}, {1, 3}}]]")
        .unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn non_diagonal_pdf_stays_unevaluated() {
    assert_eq!(
      interpret(
        "PDF[MultinormalDistribution[{0, 0}, {{2, 1}, {1, 3}}], {x, y}]"
      )
      .unwrap(),
      "PDF[MultinormalDistribution[{0, 0}, {{2, 1}, {1, 3}}], {x, y}]"
    );
  }
}

mod empirical_distribution {
  use super::*;

  #[test]
  fn constructor_normalizes() {
    // Values sort and tally into weights regardless of input order
    assert_eq!(
      interpret("EmpiricalDistribution[{1, 2, 2, 3}]").unwrap(),
      "DataDistribution[Empirical, {{1/4, 1/2, 1/4}, {1, 2, 3}, False}, 1, 4]"
    );
    assert_eq!(
      interpret("EmpiricalDistribution[{3, 1, 2, 2}]").unwrap(),
      "DataDistribution[Empirical, {{1/4, 1/2, 1/4}, {1, 2, 3}, False}, 1, 4]"
    );
  }

  #[test]
  fn exact_statistics() {
    assert_eq!(
      interpret("Mean[EmpiricalDistribution[{1, 2, 2, 3}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Mean[EmpiricalDistribution[{1, 1, 2}]]").unwrap(),
      "4/3"
    );
    assert_eq!(
      interpret("Variance[EmpiricalDistribution[{1, 2, 2, 3}]]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn pdf_point_masses() {
    assert_eq!(
      interpret("PDF[EmpiricalDistribution[{1, 2, 2, 3}], 2]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("PDF[EmpiricalDistribution[{1, 2, 2, 3}], 4]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cdf_steps() {
    assert_eq!(
      interpret("CDF[EmpiricalDistribution[{1, 2, 2, 3}], 2]").unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("CDF[EmpiricalDistribution[{1, 2, 2, 3}], 3/2]").unwrap(),
      "1/4"
    );
    assert_eq!(
      interpret("CDF[EmpiricalDistribution[{1, 2, 2, 3}], 0]").unwrap(),
      "0"
    );
  }

  #[test]
  fn symbolic_data_stays_unevaluated() {
    assert_eq!(
      interpret("EmpiricalDistribution[{x, y}]").unwrap(),
      "EmpiricalDistribution[{x, y}]"
    );
  }
}

mod empirical_quantiles {
  use super::*;

  #[test]
  fn quantile_is_inclusive_step() {
    // Smallest support value whose cumulative weight reaches q
    assert_eq!(
      interpret("Quantile[EmpiricalDistribution[{1, 2, 2, 3}], 1/2]").unwrap(),
      "2"
    );
    // CDF(1) = 1/4 reaches q = 1/4 inclusively
    assert_eq!(
      interpret("Quantile[EmpiricalDistribution[{1, 2, 2, 3}], 1/4]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Quantile[EmpiricalDistribution[{1, 2, 2, 3}], 9/10]").unwrap(),
      "3"
    );
  }

  #[test]
  fn quantile_boundaries() {
    assert_eq!(
      interpret("Quantile[EmpiricalDistribution[{1, 2, 2, 3}], 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Quantile[EmpiricalDistribution[{1, 2, 2, 3}], 1]").unwrap(),
      "3"
    );
  }

  #[test]
  fn median_inverse_cdf_survival() {
    assert_eq!(
      interpret("Median[EmpiricalDistribution[{1, 2, 2, 3}]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("InverseCDF[EmpiricalDistribution[{1, 2, 2, 3}], 1/2]")
        .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("SurvivalFunction[EmpiricalDistribution[{1, 2, 2, 3}], 2]")
        .unwrap(),
      "1/4"
    );
  }
}

mod product_distribution {
  use super::*;

  #[test]
  fn pdf_normal_pair() {
    assert_eq!(
      interpret(
        "PDF[ProductDistribution[NormalDistribution[], NormalDistribution[]], {x, y}]"
      )
      .unwrap(),
      "E^(-1/2*x^2 - y^2/2)/(2*Pi)"
    );
  }

  #[test]
  fn pdf_normal_exponential_coefficients() {
    // lambda = 2 folds 2/Sqrt[2*Pi] into the postfix Sqrt[2/Pi]
    assert_eq!(
      interpret(
        "PDF[ProductDistribution[NormalDistribution[], ExponentialDistribution[2]], {x, y}]"
      )
      .unwrap(),
      "Piecewise[{{E^(-1/2*x^2 - 2*y)*Sqrt[2/Pi], y >= 0}}, 0]"
    );
    assert_eq!(
      interpret(
        "PDF[ProductDistribution[NormalDistribution[], ExponentialDistribution[1]], {x, y}]"
      )
      .unwrap(),
      "Piecewise[{{E^(-1/2*x^2 - y)/Sqrt[2*Pi], y >= 0}}, 0]"
    );
    assert_eq!(
      interpret(
        "PDF[ProductDistribution[NormalDistribution[], ExponentialDistribution[3]], {x, y}]"
      )
      .unwrap(),
      "Piecewise[{{(3*E^(-1/2*x^2 - 3*y))/Sqrt[2*Pi], y >= 0}}, 0]"
    );
  }

  #[test]
  fn pdf_exponential_pair() {
    assert_eq!(
      interpret(
        "PDF[ProductDistribution[ExponentialDistribution[1], ExponentialDistribution[1]], {x, y}]"
      )
      .unwrap(),
      "Piecewise[{{E^(-x - y), x >= 0 && y >= 0}}, 0]"
    );
    assert_eq!(
      interpret(
        "PDF[ProductDistribution[ExponentialDistribution[2], ExponentialDistribution[3]], {x, y}]"
      )
      .unwrap(),
      "Piecewise[{{6*E^(-2*x - 3*y), x >= 0 && y >= 0}}, 0]"
    );
  }

  #[test]
  fn component_statistics() {
    assert_eq!(
      interpret(
        "Mean[ProductDistribution[NormalDistribution[m, s], ExponentialDistribution[2]]]"
      )
      .unwrap(),
      "{m, 1/2}"
    );
    assert_eq!(
      interpret(
        "Variance[ProductDistribution[NormalDistribution[m, s], ExponentialDistribution[2]]]"
      )
      .unwrap(),
      "{s^2, 1/4}"
    );
    assert_eq!(
      interpret(
        "Mean[ProductDistribution[ExponentialDistribution[1], ExponentialDistribution[2]]]"
      )
      .unwrap(),
      "{1, 1/2}"
    );
  }
}

mod uniform_sum_distribution {
  use super::*;

  #[test]
  fn mean_and_variance() {
    assert_eq!(interpret("Mean[UniformSumDistribution[3]]").unwrap(), "3/2");
    assert_eq!(
      interpret("Variance[UniformSumDistribution[3]]").unwrap(),
      "1/4"
    );
    // Symbolic n and the {min, max} range variant
    assert_eq!(interpret("Mean[UniformSumDistribution[n]]").unwrap(), "n/2");
    assert_eq!(
      interpret("Variance[UniformSumDistribution[n]]").unwrap(),
      "n/12"
    );
    assert_eq!(
      interpret("Mean[UniformSumDistribution[3, {-1, 1}]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Variance[UniformSumDistribution[3, {-1, 1}]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Mean[UniformSumDistribution[n, {a, b}]]").unwrap(),
      "((a + b)*n)/2"
    );
    assert_eq!(
      interpret("Variance[UniformSumDistribution[n, {a, b}]]").unwrap(),
      "((-a + b)^2*n)/12"
    );
  }

  #[test]
  fn pdf_piecewise_forms() {
    // n = 1 has no leading zero piece
    assert_eq!(
      interpret("PDF[UniformSumDistribution[1], x]").unwrap(),
      "Piecewise[{{1, 0 <= x <= 1}}, 0]"
    );
    assert_eq!(
      interpret("PDF[UniformSumDistribution[2], x]").unwrap(),
      "Piecewise[{{0, x < 0}, {x, Inequality[0, LessEqual, x, Less, 1]}, {2 - x, 1 <= x <= 2}}, 0]"
    );
    // Ascending inclusion-exclusion up to the midpoint, x -> n-x
    // reflections past it
    assert_eq!(
      interpret("PDF[UniformSumDistribution[3], x]").unwrap(),
      "Piecewise[{{0, x < 0}, {x^2/2, Inequality[0, LessEqual, x, Less, 1]}, {(-3*(-1 + x)^2 + x^2)/2, Inequality[1, LessEqual, x, Less, 2]}, {(3 - x)^2/2, 2 <= x <= 3}}, 0]"
    );
    assert_eq!(
      interpret("PDF[UniformSumDistribution[5], x]").unwrap(),
      "Piecewise[{{0, x < 0}, {x^4/24, Inequality[0, LessEqual, x, Less, 1]}, {(-5*(-1 + x)^4 + x^4)/24, Inequality[1, LessEqual, x, Less, 2]}, {(10*(-2 + x)^4 - 5*(-1 + x)^4 + x^4)/24, Inequality[2, LessEqual, x, Less, 3]}, {(-5*(4 - x)^4 + (5 - x)^4)/24, Inequality[3, LessEqual, x, Less, 4]}, {(5 - x)^4/24, 4 <= x <= 5}}, 0]"
    );
  }

  #[test]
  fn cdf_piecewise_forms() {
    assert_eq!(
      interpret("CDF[UniformSumDistribution[1], x]").unwrap(),
      "Piecewise[{{0, x < 0}, {x, 0 <= x <= 1}}, 1]"
    );
    assert_eq!(
      interpret("CDF[UniformSumDistribution[2], x]").unwrap(),
      "Piecewise[{{0, x < 0}, {x^2/2, Inequality[0, LessEqual, x, Less, 1]}, {1 - (2 - x)^2/2, 1 <= x <= 2}}, 1]"
    );
    // Middle pieces print expanded; past the midpoint they expand in
    // powers of (n - x)
    assert_eq!(
      interpret("CDF[UniformSumDistribution[3], x]").unwrap(),
      "Piecewise[{{0, x < 0}, {x^3/6, Inequality[0, LessEqual, x, Less, 1]}, {1/2 - (3*x)/2 + (3*x^2)/2 - x^3/3, Inequality[1, LessEqual, x, Less, 2]}, {1 - (3 - x)^3/6, 2 <= x <= 3}}, 1]"
    );
    assert_eq!(
      interpret("CDF[UniformSumDistribution[4], x]").unwrap(),
      "Piecewise[{{0, x < 0}, {x^4/24, Inequality[0, LessEqual, x, Less, 1]}, {-1/6 + (2*x)/3 - x^2 + (2*x^3)/3 - x^4/8, Inequality[1, LessEqual, x, Less, 2]}, {7/6 - (2*(4 - x))/3 + (4 - x)^2 - (2*(4 - x)^3)/3 + (4 - x)^4/8, Inequality[2, LessEqual, x, Less, 3]}, {1 - (4 - x)^4/24, 3 <= x <= 4}}, 1]"
    );
  }

  #[test]
  fn point_evaluation() {
    assert_eq!(
      interpret("PDF[UniformSumDistribution[3], 3/2]").unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("PDF[UniformSumDistribution[3], 1.5]").unwrap(),
      "0.75"
    );
    assert_eq!(
      interpret("CDF[UniformSumDistribution[3], 1/2]").unwrap(),
      "1/48"
    );
  }
}

mod beta_binomial_distribution {
  use super::*;

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[BetaBinomialDistribution[2, 3, 10]]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Variance[BetaBinomialDistribution[2, 3, 10]]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("Mean[BetaBinomialDistribution[1/2, 3/2, 6]]").unwrap(),
      "3/2"
    );
    // Fully symbolic closed forms
    assert_eq!(
      interpret("Mean[BetaBinomialDistribution[a, b, n]]").unwrap(),
      "(a*n)/(a + b)"
    );
    assert_eq!(
      interpret("Variance[BetaBinomialDistribution[a, b, n]]").unwrap(),
      "(a*b*n*(a + b + n))/((a + b)^2*(1 + a + b))"
    );
  }

  #[test]
  fn pdf_values() {
    assert_eq!(
      interpret("PDF[BetaBinomialDistribution[2, 3, 10], 4]").unwrap(),
      "20/143"
    );
    assert_eq!(
      interpret("PDF[BetaBinomialDistribution[1/2, 3/2, 4], 2]").unwrap(),
      "9/64"
    );
    // Out of range or non-integer points vanish
    assert_eq!(
      interpret("PDF[BetaBinomialDistribution[2, 3, 10], -1]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("PDF[BetaBinomialDistribution[2, 3, 10], 11]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("PDF[BetaBinomialDistribution[2, 3, 10], 1/2]").unwrap(),
      "0"
    );
  }

  #[test]
  fn pdf_symbolic_forms() {
    // Numeric n evaluates the Pochhammer denominator and prints 10 - k
    assert_eq!(
      interpret("PDF[BetaBinomialDistribution[2, 3, 10], k]").unwrap(),
      "Piecewise[{{(Binomial[10, k]*Pochhammer[2, k]*Pochhammer[3, 10 - k])/3632428800, 0 <= k <= 10}}, 0]"
    );
    // Symbolic parameters keep everything unevaluated and print -k + n
    assert_eq!(
      interpret("PDF[BetaBinomialDistribution[a, b, n], k]").unwrap(),
      "Piecewise[{{(Binomial[n, k]*Pochhammer[a, k]*Pochhammer[b, -k + n])/Pochhammer[a + b, n], 0 <= k <= n}}, 0]"
    );
  }

  #[test]
  fn cdf_values() {
    assert_eq!(
      interpret("CDF[BetaBinomialDistribution[2, 3, 4], 2]").unwrap(),
      "53/70"
    );
    assert_eq!(
      interpret("CDF[BetaBinomialDistribution[2, 3, 10], -1]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("CDF[BetaBinomialDistribution[2, 3, 10], 15]").unwrap(),
      "1"
    );
  }
}

mod beta_prime_distribution {
  use super::*;

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[BetaPrimeDistribution[3, 5]]").unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("Variance[BetaPrimeDistribution[3, 5]]").unwrap(),
      "7/16"
    );
    // Existence conditions: mean needs q > 1, variance q > 2 — and the
    // dead branch must not evaluate (no Power::infy message)
    assert_eq!(
      interpret("Mean[BetaPrimeDistribution[3, 1]]").unwrap(),
      "Infinity"
    );
    assert_eq!(interpret("Mean[BetaPrimeDistribution[3, 2]]").unwrap(), "3");
    assert_eq!(
      interpret("Variance[BetaPrimeDistribution[3, 2]]").unwrap(),
      "Indeterminate"
    );
    // Symbolic parameters keep the conditions as Piecewise
    assert_eq!(
      interpret("Mean[BetaPrimeDistribution[p, q]]").unwrap(),
      "Piecewise[{{p/(-1 + q), q > 1}}, Infinity]"
    );
    assert_eq!(
      interpret("Variance[BetaPrimeDistribution[p, q]]").unwrap(),
      "Piecewise[{{(p*(-1 + p + q))/((-2 + q)*(-1 + q)^2), q > 2}}, Indeterminate]"
    );
  }

  #[test]
  fn pdf_forms() {
    assert_eq!(
      interpret("PDF[BetaPrimeDistribution[3, 5], x]").unwrap(),
      "Piecewise[{{(105*x^2)/(1 + x)^8, x > 0}}, 0]"
    );
    // Half-integer parameters hoist the Beta rational into the numerator
    assert_eq!(
      interpret("PDF[BetaPrimeDistribution[1/2, 3/2], x]").unwrap(),
      "Piecewise[{{2/(Pi*Sqrt[x]*(1 + x)^2), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[BetaPrimeDistribution[p, q], x]").unwrap(),
      "Piecewise[{{(x^(-1 + p)*(1 + x)^(-p - q))/Beta[p, q], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[BetaPrimeDistribution[3, 5], 2]").unwrap(),
      "140/2187"
    );
    assert_eq!(
      interpret("PDF[BetaPrimeDistribution[3, 5], -1]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[BetaPrimeDistribution[3, 5], x]").unwrap(),
      "Piecewise[{{BetaRegularized[x/(1 + x), 3, 5], x > 0}}, 0]"
    );
    // Exact through the new BetaRegularized integer path
    assert_eq!(
      interpret("CDF[BetaPrimeDistribution[3, 5], 2]").unwrap(),
      "232/243"
    );
  }
}

mod beta_regularized_exact {
  use super::*;

  #[test]
  fn integer_parameters_evaluate_exactly() {
    // Regression: stayed unevaluated before; I_z(a,b) is polynomial in
    // z for integer a, b
    assert_eq!(interpret("BetaRegularized[2/3, 3, 5]").unwrap(), "232/243");
    assert_eq!(interpret("BetaRegularized[1/2, 2, 2]").unwrap(), "1/2");
  }
}

mod noncentral_chi_square_distribution {
  use super::*;

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[NoncentralChiSquareDistribution[3, 2]]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("Variance[NoncentralChiSquareDistribution[3, 2]]").unwrap(),
      "14"
    );
    assert_eq!(
      interpret("Mean[NoncentralChiSquareDistribution[v, l]]").unwrap(),
      "l + v"
    );
    assert_eq!(
      interpret("Variance[NoncentralChiSquareDistribution[v, l]]").unwrap(),
      "4*l + 2*v"
    );
  }

  #[test]
  fn pdf_odd_degrees_collapse_to_hyperbolics() {
    // v = 3: Sinh with the Sqrt[2 l Pi] denominator canonicalizing
    // per lambda (extracted square, merged radical, dropped unit)
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[3, 2], x]").unwrap(),
      "Piecewise[{{(E^((-2 - x)/2)*Sinh[Sqrt[2]*Sqrt[x]])/(2*Sqrt[Pi]), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[3, 3], x]").unwrap(),
      "Piecewise[{{(E^((-3 - x)/2)*Sinh[Sqrt[3]*Sqrt[x]])/Sqrt[6*Pi], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[3, 1/2], x]").unwrap(),
      "Piecewise[{{(E^((-1/2 - x)/2)*Sinh[Sqrt[x]/Sqrt[2]])/Sqrt[Pi], x > 0}}, 0]"
    );
    // v = 1: Cosh form
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[1, 2], x]").unwrap(),
      "Piecewise[{{(E^((-2 - x)/2)*Cosh[Sqrt[2]*Sqrt[x]])/(Sqrt[2*Pi]*Sqrt[x]), x > 0}}, 0]"
    );
  }

  #[test]
  fn pdf_even_degrees_keep_hypergeometric() {
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[2, 2], x]").unwrap(),
      "Piecewise[{{(E^((-2 - x)/2)*Hypergeometric0F1Regularized[1, x/2])/2, x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[4, 3], x]").unwrap(),
      "Piecewise[{{(E^((-3 - x)/2)*x*Hypergeometric0F1Regularized[2, (3*x)/4])/4, x > 0}}, 0]"
    );
    // Fully symbolic skeleton
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[v, l], x]").unwrap(),
      "Piecewise[{{(E^((-l - x)/2)*x^(-1 + v/2)*Hypergeometric0F1Regularized[v/2, (l*x)/4])/2^(v/2), x > 0}}, 0]"
    );
    // Points evaluate for even v
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[2, 2], 3]").unwrap(),
      "Hypergeometric0F1Regularized[1, 3/2]/(2*E^(5/2))"
    );
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[3, 2], -1]").unwrap(),
      "0"
    );
  }

  #[test]
  fn zero_noncentrality_is_chi_square() {
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[1, 0], x]").unwrap(),
      "Piecewise[{{1/(E^(x/2)*Sqrt[2*Pi]*Sqrt[x]), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[2, 0], x]").unwrap(),
      "Piecewise[{{1/(2*E^(x/2)), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[3, 0], x]").unwrap(),
      "Piecewise[{{Sqrt[x]/(E^(x/2)*Sqrt[2*Pi]), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[4, 0], x]").unwrap(),
      "Piecewise[{{x/(4*E^(x/2)), x > 0}}, 0]"
    );
    // (v-2)!! coefficient appears from v = 5
    assert_eq!(
      interpret("PDF[NoncentralChiSquareDistribution[5, 0], x]").unwrap(),
      "Piecewise[{{x^(3/2)/(3*E^(x/2)*Sqrt[2*Pi]), x > 0}}, 0]"
    );
  }
}

mod noncentral_chi_square_cdf {
  use super::*;

  #[test]
  fn marcum_q_forms() {
    assert_eq!(
      interpret("CDF[NoncentralChiSquareDistribution[3, 2], x]").unwrap(),
      "Piecewise[{{MarcumQ[3/2, Sqrt[2], 0, Sqrt[x]], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[NoncentralChiSquareDistribution[v, l], x]").unwrap(),
      "Piecewise[{{MarcumQ[v/2, Sqrt[l], 0, Sqrt[x]], x > 0}}, 0]"
    );
    // Exact points stay symbolic, just like wolframscript
    assert_eq!(
      interpret("CDF[NoncentralChiSquareDistribution[3, 2], 2]").unwrap(),
      "MarcumQ[3/2, Sqrt[2], 0, Sqrt[2]]"
    );
    assert_eq!(
      interpret("CDF[NoncentralChiSquareDistribution[3, 2], -1]").unwrap(),
      "0"
    );
    // Real points evaluate through the MarcumQ series
    assert_eq!(
      interpret("Round[10^10 CDF[NoncentralChiSquareDistribution[3, 2], 2.5]]")
        .unwrap(),
      "2899007987"
    );
  }
}

mod exponential_power_distribution {
  use super::*;

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[ExponentialPowerDistribution[2, 0, 1]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Variance[ExponentialPowerDistribution[2, 0, 1]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Variance[ExponentialPowerDistribution[1, 0, 1]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Variance[ExponentialPowerDistribution[3, 2, 5]]").unwrap(),
      "(25*3^(2/3))/Gamma[1/3]"
    );
    assert_eq!(
      interpret("Mean[ExponentialPowerDistribution[k, m, s]]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("Variance[ExponentialPowerDistribution[k, m, s]]").unwrap(),
      "(k^(2/k)*s^2*Gamma[3/k])/Gamma[k^(-1)]"
    );
  }

  #[test]
  fn pdf_forms() {
    // k = 2, m = 0: both branches canonicalize identically and the
    // Piecewise collapses to the plain normal density
    assert_eq!(
      interpret("PDF[ExponentialPowerDistribution[2, 0, 1], x]").unwrap(),
      "1/(E^(x^2/2)*Sqrt[2*Pi])"
    );
    // Nonzero location keeps the mirrored branches
    assert_eq!(
      interpret("PDF[ExponentialPowerDistribution[2, 1, 2], x]").unwrap(),
      "Piecewise[{{1/(2*E^((-1 + x)^2/8)*Sqrt[2*Pi]), x >= 1}}, 1/(2*E^((1 - x)^2/8)*Sqrt[2*Pi])]"
    );
    // k = 1 is Laplace
    assert_eq!(
      interpret("PDF[ExponentialPowerDistribution[1, 0, 1], x]").unwrap(),
      "Piecewise[{{1/(2*E^x), x >= 0}}, E^x/2]"
    );
    // Odd k flips the lower branch's exponent into the numerator
    assert_eq!(
      interpret("PDF[ExponentialPowerDistribution[3, 0, 1], x]").unwrap(),
      "Piecewise[{{1/(2*3^(1/3)*E^(x^3/3)*Gamma[4/3]), x >= 0}}, E^(x^3/3)/(2*3^(1/3)*Gamma[4/3])]"
    );
    assert_eq!(
      interpret("PDF[ExponentialPowerDistribution[k, m, s], x]").unwrap(),
      "Piecewise[{{1/(2*E^(((-m + x)/s)^k/k)*k^k^(-1)*s*Gamma[1 + k^(-1)]), x >= m}}, 1/(2*E^(((m - x)/s)^k/k)*k^k^(-1)*s*Gamma[1 + k^(-1)])]"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[ExponentialPowerDistribution[2, 0, 1], x]").unwrap(),
      "Piecewise[{{GammaRegularized[1/2, x^2/2]/2, x < 0}}, 1 - GammaRegularized[1/2, x^2/2]/2]"
    );
    // k = 1 evaluates through GammaRegularized[1, z] = E^(-z)
    assert_eq!(
      interpret("CDF[ExponentialPowerDistribution[1, 0, 1], x]").unwrap(),
      "Piecewise[{{E^x/2, x < 0}}, 1 - 1/(2*E^x)]"
    );
    assert_eq!(
      interpret("CDF[ExponentialPowerDistribution[k, m, s], x]").unwrap(),
      "Piecewise[{{GammaRegularized[k^(-1), ((m - x)/s)^k/k]/2, x < m}}, 1 - GammaRegularized[k^(-1), ((-m + x)/s)^k/k]/2]"
    );
  }
}

mod gamma_regularized_order_one {
  use super::*;

  #[test]
  fn evaluates_symbolically() {
    // Wolfram auto-evaluates order one even for symbolic z
    assert_eq!(interpret("GammaRegularized[1, x]").unwrap(), "E^(-x)");
    // ...but not higher integer orders
    assert_eq!(
      interpret("GammaRegularized[4, x]").unwrap(),
      "GammaRegularized[4, x]"
    );
  }
}

mod rice_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    assert_eq!(
      interpret("PDF[RiceDistribution[2, 1], x]").unwrap(),
      "Piecewise[{{E^((-4 - x^2)/2)*x*BesselI[0, 2*x], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[RiceDistribution[a, b], x]").unwrap(),
      "Piecewise[{{(E^((-a^2 - x^2)/(2*b^2))*x*BesselI[0, (a*x)/b^2])/b^2, x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[RiceDistribution[2, 1], 1]").unwrap(),
      "BesselI[0, 2]/E^(5/2)"
    );
    assert_eq!(interpret("PDF[RiceDistribution[2, 1], -1]").unwrap(), "0");
  }

  #[test]
  fn moments_factor_bessel_sums() {
    assert_eq!(
      interpret("Mean[RiceDistribution[2, 1]]").unwrap(),
      "(Sqrt[Pi/2]*(3*BesselI[0, 1] + 2*BesselI[1, 1]))/E"
    );
    // Rational k pulls the common denominator out of the sum
    assert_eq!(
      interpret("Mean[RiceDistribution[1, 1]]").unwrap(),
      "(Sqrt[Pi/2]*(3*BesselI[0, 1/4] + BesselI[1, 1/4]))/(2*E^(1/4))"
    );
    assert_eq!(
      interpret("Mean[RiceDistribution[1, 2]]").unwrap(),
      "(Sqrt[Pi/2]*(9*BesselI[0, 1/16] + BesselI[1, 1/16]))/(4*E^(1/16))"
    );
    // b/q can cancel completely
    assert_eq!(
      interpret("Mean[RiceDistribution[2, 2]]").unwrap(),
      "(Sqrt[Pi/2]*(3*BesselI[0, 1/4] + BesselI[1, 1/4]))/E^(1/4)"
    );
    // Rayleigh special case
    assert_eq!(
      interpret("Mean[RiceDistribution[0, 1]]").unwrap(),
      "Sqrt[Pi/2]"
    );
    assert_eq!(
      interpret("Variance[RiceDistribution[2, 1]]").unwrap(),
      "6 - (Pi*(3*BesselI[0, 1] + 2*BesselI[1, 1])^2)/(2*E^2)"
    );
    assert_eq!(
      interpret("Variance[RiceDistribution[1, 1]]").unwrap(),
      "3 - (Pi*(3*BesselI[0, 1/4] + BesselI[1, 1/4])^2)/(8*Sqrt[E])"
    );
    assert_eq!(
      interpret("Variance[RiceDistribution[0, 1]]").unwrap(),
      "2 - Pi/2"
    );
    // Symbolic parameters keep the LaguerreL forms
    assert_eq!(
      interpret("Mean[RiceDistribution[a, b]]").unwrap(),
      "b*Sqrt[Pi/2]*LaguerreL[1/2, -1/2*a^2/b^2]"
    );
    assert_eq!(
      interpret("Variance[RiceDistribution[a, b]]").unwrap(),
      "a^2 + 2*b^2 - (b^2*Pi*LaguerreL[1/2, -1/2*a^2/b^2]^2)/2"
    );
  }

  #[test]
  fn cdf_uses_marcum_q() {
    assert_eq!(
      interpret("CDF[RiceDistribution[2, 1], x]").unwrap(),
      "Piecewise[{{MarcumQ[1, 2, 0, x], x > 0}}, 0]"
    );
    // Real points evaluate through the MarcumQ series
    assert_eq!(
      interpret("Round[10^10 CDF[RiceDistribution[2, 1], 2.5]]").unwrap(),
      "6058960755"
    );
  }
}

mod min_stable_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    // g = 0 is the Gumbel-min branch with full real support
    assert_eq!(
      interpret("PDF[MinStableDistribution[0, 1, 0], x]").unwrap(),
      "E^(-E^x + x)"
    );
    assert_eq!(
      interpret("PDF[MinStableDistribution[2, 3, 0], x]").unwrap(),
      "E^(-E^((-2 + x)/3) - (2 - x)/3)/3"
    );
    assert_eq!(
      interpret("PDF[MinStableDistribution[0, 1, 1], x]").unwrap(),
      "Piecewise[{{1/(E^(1 - x)^(-1)*(1 - x)^2), 1 - x > 0}}, 0]"
    );
    // Negative shape folds the sign into the difference
    assert_eq!(
      interpret("PDF[MinStableDistribution[0, 1, -1], x]").unwrap(),
      "Piecewise[{{E^(-1 - x), 1 + x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[MinStableDistribution[a, b, g], x]").unwrap(),
      "Piecewise[{{E^(-E^((-a + x)/b) - (a - x)/b)/b, g == 0}, {(1 + (g*(a - x))/b)^(-1 - g^(-1))/(b*E^(1 + (g*(a - x))/b)^(-g^(-1))), g != 0 && 1 + (g*(a - x))/b > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[MinStableDistribution[0, 1, 1], 1/2]").unwrap(),
      "4/E^2"
    );
    assert_eq!(
      interpret("PDF[MinStableDistribution[0, 1, 1], 2]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[MinStableDistribution[0, 1, 0], x]").unwrap(),
      "1 - E^(-E^x)"
    );
    // Default is 1 above the support for g > 0, 0 below it for g < 0
    assert_eq!(
      interpret("CDF[MinStableDistribution[0, 1, 1], x]").unwrap(),
      "Piecewise[{{1 - E^(-(1 - x)^(-1)), 1 - x > 0}}, 1]"
    );
    assert_eq!(
      interpret("CDF[MinStableDistribution[0, 1, -1], x]").unwrap(),
      "Piecewise[{{1 - E^(-1 - x), 1 + x > 0}}, 0]"
    );
    // The symbolic form keeps three pieces
    assert_eq!(
      interpret("CDF[MinStableDistribution[a, b, g], x]").unwrap(),
      "Piecewise[{{1 - E^(-E^((-a + x)/b)), g == 0}, {1 - E^(-(1 + (g*(a - x))/b)^(-g^(-1))), g != 0 && 1 + (g*(a - x))/b > 0}, {1, g > 0 && 1 + (g*(a - x))/b <= 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[MinStableDistribution[0, 1, 1], 1/2]").unwrap(),
      "1 - E^(-2)"
    );
  }

  #[test]
  fn moments_with_existence_conditions() {
    assert_eq!(
      interpret("Mean[MinStableDistribution[0, 1, 0]]").unwrap(),
      "-EulerGamma"
    );
    assert_eq!(
      interpret("Mean[MinStableDistribution[2, 3, 0]]").unwrap(),
      "2 - 3*EulerGamma"
    );
    assert_eq!(
      interpret("Mean[MinStableDistribution[2, 3, 1/2]]").unwrap(),
      "2*(4 - 3*Sqrt[Pi])"
    );
    // The mean needs g < 1, the variance 2 g < 1
    assert_eq!(
      interpret("Mean[MinStableDistribution[0, 1, 1]]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Variance[MinStableDistribution[0, 1, 1/2]]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Variance[MinStableDistribution[2, 3, 0]]").unwrap(),
      "(3*Pi^2)/2"
    );
    assert_eq!(
      interpret("Variance[MinStableDistribution[0, 1, 1/4]]").unwrap(),
      "16*(Sqrt[Pi] - Gamma[3/4]^2)"
    );
    assert_eq!(
      interpret("Mean[MinStableDistribution[a, b, g]]").unwrap(),
      "Piecewise[{{a - b*EulerGamma, g == 0}, {(b + a*g - b*Gamma[1 - g])/g, g != 0 && g < 1}}, Indeterminate]"
    );
    assert_eq!(
      interpret("Variance[MinStableDistribution[a, b, g]]").unwrap(),
      "Piecewise[{{(b^2*Pi^2)/6, g == 0}, {(b^2*(Gamma[1 - 2*g] - Gamma[1 - g]^2))/g^2, g != 0 && 2*g < 1}}, Indeterminate]"
    );
  }
}

mod max_stable_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    assert_eq!(
      interpret("PDF[MaxStableDistribution[0, 1, 0], x]").unwrap(),
      "E^(-E^(-x) - x)"
    );
    // The second exponent term reuses the (a - x)/b argument
    assert_eq!(
      interpret("PDF[MaxStableDistribution[2, 3, 0], x]").unwrap(),
      "E^(-E^((2 - x)/3) + (2 - x)/3)/3"
    );
    assert_eq!(
      interpret("PDF[MaxStableDistribution[0, 1, 1], x]").unwrap(),
      "Piecewise[{{1/(E^(1 + x)^(-1)*(1 + x)^2), 1 + x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[MaxStableDistribution[a, b, g], x]").unwrap(),
      "Piecewise[{{E^(-E^((a - x)/b) - (-a + x)/b)/b, g == 0}, {(1 + (g*(-a + x))/b)^(-1 - g^(-1))/(b*E^(1 + (g*(-a + x))/b)^(-g^(-1))), g != 0 && 1 + (g*(-a + x))/b > 0}}, 0]"
    );
    // Negative shape folds the sign into (a - x)
    assert_eq!(
      interpret("PDF[MaxStableDistribution[1, 2, -1/2], x]").unwrap(),
      "Piecewise[{{(1 + (1 - x)/4)/(2*E^(1 + (1 - x)/4)^2), 1 + (1 - x)/4 > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[MaxStableDistribution[0, 1, 1], -1/2]").unwrap(),
      "4/E^2"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[MaxStableDistribution[0, 1, 0], x]").unwrap(),
      "E^(-E^(-x))"
    );
    // Defaults mirror MinStable: 0 below the support for g > 0,
    // 1 above it for g < 0
    assert_eq!(
      interpret("CDF[MaxStableDistribution[0, 1, 1], x]").unwrap(),
      "Piecewise[{{E^(-(1 + x)^(-1)), 1 + x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[MaxStableDistribution[0, 1, -1], x]").unwrap(),
      "Piecewise[{{E^(-1 + x), 1 - x > 0}}, 1]"
    );
    assert_eq!(
      interpret("CDF[MaxStableDistribution[a, b, g], x]").unwrap(),
      "Piecewise[{{E^(-E^((a - x)/b)), g == 0}, {E^(-(1 + (g*(-a + x))/b)^(-g^(-1))), g != 0 && 1 + (g*(-a + x))/b > 0}, {0, g > 0 && 1 + (g*(-a + x))/b <= 0}}, 1]"
    );
    assert_eq!(
      interpret("CDF[MaxStableDistribution[0, 1, 1], -2]").unwrap(),
      "0"
    );
  }

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[MaxStableDistribution[0, 1, 0]]").unwrap(),
      "EulerGamma"
    );
    assert_eq!(
      interpret("Mean[MaxStableDistribution[2, 3, 1/2]]").unwrap(),
      "2*(-2 + 3*Sqrt[Pi])"
    );
    assert_eq!(
      interpret("Mean[MaxStableDistribution[0, 1, 1]]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Mean[MaxStableDistribution[a, b, g]]").unwrap(),
      "Piecewise[{{a + b*EulerGamma, g == 0}, {(-b + a*g + b*Gamma[1 - g])/g, g != 0 && g < 1}}, Indeterminate]"
    );
    // Variance is identical to MinStableDistribution's
    assert_eq!(
      interpret("Variance[MaxStableDistribution[0, 1, 1/4]]").unwrap(),
      "16*(Sqrt[Pi] - Gamma[3/4]^2)"
    );
    assert_eq!(
      interpret("Variance[MaxStableDistribution[a, b, g]]").unwrap(),
      "Piecewise[{{(b^2*Pi^2)/6, g == 0}, {(b^2*(Gamma[1 - 2*g] - Gamma[1 - g]^2))/g^2, g != 0 && 2*g < 1}}, Indeterminate]"
    );
  }
}

mod triangular_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    assert_eq!(
      interpret("PDF[TriangularDistribution[{0, 2}], x]").unwrap(),
      "Piecewise[{{x, 0 <= x <= 1}, {2 - x, Inequality[1, Less, x, LessEqual, 2]}}, 0]"
    );
    assert_eq!(
      interpret("PDF[TriangularDistribution[{0, 4}, 1], x]").unwrap(),
      "Piecewise[{{x/2, 0 <= x <= 1}, {(4 - x)/6, Inequality[1, Less, x, LessEqual, 4]}}, 0]"
    );
    // No arguments defaults to {0, 1} with mode 1/2; the falling piece
    // keeps its factored 4*(1 - x) form
    assert_eq!(
      interpret("PDF[TriangularDistribution[], x]").unwrap(),
      "Piecewise[{{4*x, 0 <= x <= 1/2}, {4*(1 - x), Inequality[1/2, Less, x, LessEqual, 1]}}, 0]"
    );
    assert_eq!(
      interpret("PDF[TriangularDistribution[{1, 5}], x]").unwrap(),
      "Piecewise[{{(-1 + x)/4, 1 <= x <= 3}, {(5 - x)/4, Inequality[3, Less, x, LessEqual, 5]}}, 0]"
    );
    assert_eq!(
      interpret("PDF[TriangularDistribution[{a, b}, c], x]").unwrap(),
      "Piecewise[{{(2*(-a + x))/((-a + b)*(-a + c)), a <= x <= c}, {(2*(b - x))/((-a + b)*(b - c)), Inequality[c, Less, x, LessEqual, b]}}, 0]"
    );
    assert_eq!(
      interpret("PDF[TriangularDistribution[{0, 2}], 1/2]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("PDF[TriangularDistribution[{0, 2}], 3]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[TriangularDistribution[{0, 2}], x]").unwrap(),
      "Piecewise[{{x^2/2, 0 <= x <= 1}, {1 - (2 - x)^2/2, Inequality[1, Less, x, LessEqual, 2]}, {1, x > 2}}, 0]"
    );
    assert_eq!(
      interpret("CDF[TriangularDistribution[], x]").unwrap(),
      "Piecewise[{{2*x^2, 0 <= x <= 1/2}, {1 - 2*(1 - x)^2, Inequality[1/2, Less, x, LessEqual, 1]}, {1, x > 1}}, 0]"
    );
    assert_eq!(
      interpret("CDF[TriangularDistribution[{a, b}, c], x]").unwrap(),
      "Piecewise[{{(-a + x)^2/((-a + b)*(-a + c)), a <= x <= c}, {1 - (b - x)^2/((-a + b)*(b - c)), Inequality[c, Less, x, LessEqual, b]}, {1, x > b}}, 0]"
    );
    assert_eq!(
      interpret("CDF[TriangularDistribution[{0, 4}, 1], 2]").unwrap(),
      "2/3"
    );
    assert_eq!(
      interpret("CDF[TriangularDistribution[{0, 2}], -1]").unwrap(),
      "0"
    );
  }

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[TriangularDistribution[{0, 4}, 1]]").unwrap(),
      "5/3"
    );
    assert_eq!(
      interpret("Mean[TriangularDistribution[{0, 2}]]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Mean[TriangularDistribution[{a, b}, c]]").unwrap(),
      "(a + b + c)/3"
    );
    assert_eq!(
      interpret("Variance[TriangularDistribution[{0, 4}, 1]]").unwrap(),
      "13/18"
    );
    assert_eq!(
      interpret("Variance[TriangularDistribution[{a, b}, c]]").unwrap(),
      "(a^2 - a*b + b^2 - a*c - b*c + c^2)/18"
    );
  }

  // The symmetric 2-argument form TriangularDistribution[{a, b}] (mode at the
  // midpoint) must collapse to the simple symmetric moments, not the
  // unsimplified 3-parameter formula with c = (a+b)/2 substituted.
  #[test]
  fn symmetric_two_arg_moments() {
    assert_eq!(
      interpret("Mean[TriangularDistribution[{a, b}]]").unwrap(),
      "(a + b)/2"
    );
    assert_eq!(
      interpret("Variance[TriangularDistribution[{a, b}]]").unwrap(),
      "(-a + b)^2/24"
    );
    assert_eq!(
      interpret("StandardDeviation[TriangularDistribution[{a, b}]]").unwrap(),
      "(-a + b)/(2*Sqrt[6])"
    );
    // Numeric symmetric form.
    assert_eq!(
      interpret("Variance[TriangularDistribution[{0, 6}]]").unwrap(),
      "3/2"
    );
  }
}

mod maxwell_distribution {
  use super::*;

  #[test]
  fn pdf_radical_canonicalization() {
    assert_eq!(
      interpret("PDF[MaxwellDistribution[1], x]").unwrap(),
      "Piecewise[{{(Sqrt[2/Pi]*x^2)/E^(x^2/2), x > 0}}, 0]"
    );
    // Powers of two in the scale's cube merge into Sqrt[2*Pi]...
    assert_eq!(
      interpret("PDF[MaxwellDistribution[2], x]").unwrap(),
      "Piecewise[{{x^2/(4*E^(x^2/8)*Sqrt[2*Pi]), x > 0}}, 0]"
    );
    // ...while odd factors and numerators keep Sqrt[2/Pi]
    assert_eq!(
      interpret("PDF[MaxwellDistribution[3], x]").unwrap(),
      "Piecewise[{{(Sqrt[2/Pi]*x^2)/(27*E^(x^2/18)), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[MaxwellDistribution[1/2], x]").unwrap(),
      "Piecewise[{{(8*Sqrt[2/Pi]*x^2)/E^(2*x^2), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[MaxwellDistribution[s], x]").unwrap(),
      "Piecewise[{{(Sqrt[2/Pi]*x^2)/(E^(x^2/(2*s^2))*s^3), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[MaxwellDistribution[1], 2]").unwrap(),
      "(4*Sqrt[2/Pi])/E^2"
    );
    assert_eq!(interpret("PDF[MaxwellDistribution[1], -1]").unwrap(), "0");
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[MaxwellDistribution[1], x]").unwrap(),
      "Piecewise[{{-((Sqrt[2/Pi]*x)/E^(x^2/2)) + Erf[x/Sqrt[2]], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[MaxwellDistribution[2], x]").unwrap(),
      "Piecewise[{{-(x/(E^(x^2/8)*Sqrt[2*Pi])) + Erf[x/(2*Sqrt[2])], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[MaxwellDistribution[s], x]").unwrap(),
      "Piecewise[{{-((Sqrt[2/Pi]*x)/(E^(x^2/(2*s^2))*s)) + Erf[x/(Sqrt[2]*s)], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[MaxwellDistribution[1], 2]").unwrap(),
      "(-2*Sqrt[2/Pi])/E^2 + Erf[Sqrt[2]]"
    );
    assert_eq!(
      interpret("Round[10^10 CDF[MaxwellDistribution[1.], 2.]]").unwrap(),
      "7385358701"
    );
    assert_eq!(interpret("CDF[MaxwellDistribution[1], -1]").unwrap(), "0");
  }

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[MaxwellDistribution[s]]").unwrap(),
      "2*Sqrt[2/Pi]*s"
    );
    assert_eq!(
      interpret("Mean[MaxwellDistribution[2]]").unwrap(),
      "4*Sqrt[2/Pi]"
    );
    // The Pi-sum factor prints first for symbolic s
    assert_eq!(
      interpret("Variance[MaxwellDistribution[s]]").unwrap(),
      "((-8 + 3*Pi)*s^2)/Pi"
    );
    assert_eq!(
      interpret("Variance[MaxwellDistribution[2]]").unwrap(),
      "(4*(-8 + 3*Pi))/Pi"
    );
  }
}

mod wigner_semicircle_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    assert_eq!(
      interpret("PDF[WignerSemicircleDistribution[1], x]").unwrap(),
      "Piecewise[{{(2*Sqrt[1 - x^2])/Pi, -1 < x < 1}}, 0]"
    );
    // The 2/(Pi r) coefficient merges for r = 2
    assert_eq!(
      interpret("PDF[WignerSemicircleDistribution[2], x]").unwrap(),
      "Piecewise[{{Sqrt[1 - x^2/4]/Pi, -2 < x < 2}}, 0]"
    );
    assert_eq!(
      interpret("PDF[WignerSemicircleDistribution[a, r], x]").unwrap(),
      "Piecewise[{{(2*Sqrt[1 - (-a + x)^2/r^2])/(Pi*r), a - r < x < a + r}}, 0]"
    );
    assert_eq!(
      interpret("PDF[WignerSemicircleDistribution[2], 1]").unwrap(),
      "Sqrt[3]/(2*Pi)"
    );
    assert_eq!(
      interpret("PDF[WignerSemicircleDistribution[1], 3]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[WignerSemicircleDistribution[1], x]").unwrap(),
      "Piecewise[{{1/2 + (x*Sqrt[1 - x^2])/Pi + ArcSin[x]/Pi, -1 < x < 1}, {1, x >= 1}}, 0]"
    );
    // Sqrt factor order flips when the shifted variable leads with a
    // number
    assert_eq!(
      interpret("CDF[WignerSemicircleDistribution[3, 2], x]").unwrap(),
      "Piecewise[{{1/2 + (Sqrt[1 - (-3 + x)^2/4]*(-3 + x))/(2*Pi) + ArcSin[(-3 + x)/2]/Pi, 1 < x < 5}, {1, x >= 5}}, 0]"
    );
    assert_eq!(
      interpret("CDF[WignerSemicircleDistribution[a, r], x]").unwrap(),
      "Piecewise[{{1/2 + ((-a + x)*Sqrt[1 - (-a + x)^2/r^2])/(Pi*r) + ArcSin[(-a + x)/r]/Pi, a - r < x < a + r}, {1, x >= a + r}}, 0]"
    );
    assert_eq!(
      interpret("CDF[WignerSemicircleDistribution[1], 1/2]").unwrap(),
      "2/3 + Sqrt[3]/(4*Pi)"
    );
    assert_eq!(
      interpret("CDF[WignerSemicircleDistribution[1], -2]").unwrap(),
      "0"
    );
  }

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[WignerSemicircleDistribution[r]]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Mean[WignerSemicircleDistribution[a, r]]").unwrap(),
      "a"
    );
    assert_eq!(
      interpret("Variance[WignerSemicircleDistribution[r]]").unwrap(),
      "r^2/4"
    );
    assert_eq!(
      interpret("Variance[WignerSemicircleDistribution[a, r]]").unwrap(),
      "r^2/4"
    );
  }
}

mod sech_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    assert_eq!(
      interpret("PDF[SechDistribution[], x]").unwrap(),
      "Sech[(Pi*x)/2]/2"
    );
    assert_eq!(
      interpret("PDF[SechDistribution[0, 1], x]").unwrap(),
      "Sech[(Pi*x)/2]/2"
    );
    assert_eq!(
      interpret("PDF[SechDistribution[m, s], x]").unwrap(),
      "Sech[(Pi*(-m + x))/(2*s)]/(2*s)"
    );
    assert_eq!(
      interpret("PDF[SechDistribution[1, 2], x]").unwrap(),
      "Sech[(Pi*(-1 + x))/4]/4"
    );
    // Points: Sech[0] collapses, others stay closed-form
    assert_eq!(interpret("PDF[SechDistribution[0, 1], 0]").unwrap(), "1/2");
    assert_eq!(
      interpret("PDF[SechDistribution[0, 1], 1]").unwrap(),
      "Sech[Pi/2]/2"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[SechDistribution[m, s], x]").unwrap(),
      "(2*ArcTan[E^((Pi*(-m + x))/(2*s))])/Pi"
    );
    assert_eq!(
      interpret("CDF[SechDistribution[0, 1], x]").unwrap(),
      "(2*ArcTan[E^((Pi*x)/2)])/Pi"
    );
    // ArcTan[1] collapses to Pi/4 at the median
    assert_eq!(interpret("CDF[SechDistribution[0, 1], 0]").unwrap(), "1/2");
    assert_eq!(
      interpret("CDF[SechDistribution[1, 2], 3]").unwrap(),
      "(2*ArcTan[E^(Pi/2)])/Pi"
    );
  }

  #[test]
  fn moments() {
    assert_eq!(interpret("Mean[SechDistribution[m, s]]").unwrap(), "m");
    assert_eq!(
      interpret("Variance[SechDistribution[m, s]]").unwrap(),
      "s^2"
    );
    assert_eq!(interpret("Variance[SechDistribution[0, 2]]").unwrap(), "4");
  }
}

mod moyal_distribution {
  use super::*;

  #[test]
  fn pdf_sign_folding() {
    // Symbolic and zero locations use the reciprocal exponent form
    assert_eq!(
      interpret("PDF[MoyalDistribution[m, s], x]").unwrap(),
      "E^(-1/2*1/E^((-m + x)/s) - (-m + x)/(2*s))/(Sqrt[2*Pi]*s)"
    );
    assert_eq!(
      interpret("PDF[MoyalDistribution[0, 1], x]").unwrap(),
      "E^(-1/2*1/E^x - x/2)/Sqrt[2*Pi]"
    );
    assert_eq!(
      interpret("PDF[MoyalDistribution[], x]").unwrap(),
      "E^(-1/2*1/E^x - x/2)/Sqrt[2*Pi]"
    );
    // Numeric nonzero locations fold the sign into (m - x)/s
    assert_eq!(
      interpret("PDF[MoyalDistribution[1, 2], x]").unwrap(),
      "E^(-1/2*E^((1 - x)/2) + (1 - x)/4)/(2*Sqrt[2*Pi])"
    );
    assert_eq!(
      interpret("PDF[MoyalDistribution[-1, 2], x]").unwrap(),
      "E^(-1/2*E^((-1 - x)/2) + (-1 - x)/4)/(2*Sqrt[2*Pi])"
    );
    // The mode collapses to 1/Sqrt[2 E Pi]
    assert_eq!(
      interpret("PDF[MoyalDistribution[0, 1], 0]").unwrap(),
      "1/Sqrt[2*E*Pi]"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[MoyalDistribution[m, s], x]").unwrap(),
      "Erfc[1/(Sqrt[2]*E^((-m + x)/(2*s)))]"
    );
    assert_eq!(
      interpret("CDF[MoyalDistribution[0, 1], x]").unwrap(),
      "Erfc[1/(Sqrt[2]*E^(x/2))]"
    );
    assert_eq!(
      interpret("CDF[MoyalDistribution[0, 1], 0]").unwrap(),
      "Erfc[1/Sqrt[2]]"
    );
  }

  #[test]
  fn moments() {
    // s prints before the EulerGamma sum
    assert_eq!(
      interpret("Mean[MoyalDistribution[m, s]]").unwrap(),
      "m + s*(EulerGamma + Log[2])"
    );
    assert_eq!(
      interpret("Mean[MoyalDistribution[2, 3]]").unwrap(),
      "2 + 3*(EulerGamma + Log[2])"
    );
    assert_eq!(
      interpret("Mean[MoyalDistribution[]]").unwrap(),
      "EulerGamma + Log[2]"
    );
    assert_eq!(
      interpret("Variance[MoyalDistribution[m, s]]").unwrap(),
      "(Pi^2*s^2)/2"
    );
    assert_eq!(
      interpret("Variance[MoyalDistribution[2, 3]]").unwrap(),
      "(9*Pi^2)/2"
    );
  }
}

mod borel_tanner_distribution {
  use super::*;

  #[test]
  fn pdf_prime_power_merging() {
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[a, n], x]").unwrap(),
      "Piecewise[{{(a^(-n + x)*n*x^(-1 - n + x))/(E^(a*x)*(-n + x)!), x >= n}}, 0]"
    );
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[a, 2], x]").unwrap(),
      "Piecewise[{{(2*a^(-2 + x)*x^(-3 + x))/(E^(a*x)*(-2 + x)!), x >= 2}}, 0]"
    );
    // n = 2 merges into the q = 2 base: 2*2^(2-x) -> 2^(3-x)
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[1/2, 2], x]").unwrap(),
      "Piecewise[{{(2^(3 - x)*x^(-3 + x))/(E^(x/2)*(-2 + x)!), x >= 2}}, 0]"
    );
    // ...but stays separate when no base matches
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[1/3, 2], x]").unwrap(),
      "Piecewise[{{(2*3^(2 - x)*x^(-3 + x))/(E^(x/3)*(-2 + x)!), x >= 2}}, 0]"
    );
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[1/2, 3], x]").unwrap(),
      "Piecewise[{{(3*2^(3 - x)*x^(-4 + x))/(E^(x/2)*(-3 + x)!), x >= 3}}, 0]"
    );
    // Rational a splits into p- and q-power factors, n merging into p
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[2/3, 2], x]").unwrap(),
      "Piecewise[{{(2^(-1 + x)*3^(2 - x)*x^(-3 + x))/(E^((2*x)/3)*(-2 + x)!), x >= 2}}, 0]"
    );
  }

  #[test]
  fn pdf_points() {
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[1/2, 2], 5]").unwrap(),
      "25/(24*E^(5/2))"
    );
    // Below the support and non-integers vanish
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[1/2, 2], 1]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("PDF[BorelTannerDistribution[1/2, 2], 7/2]").unwrap(),
      "0"
    );
  }

  #[test]
  fn moments_and_unevaluated_cdf() {
    assert_eq!(
      interpret("Mean[BorelTannerDistribution[a, n]]").unwrap(),
      "n/(1 - a)"
    );
    assert_eq!(
      interpret("Mean[BorelTannerDistribution[1/2, 2]]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Variance[BorelTannerDistribution[a, n]]").unwrap(),
      "(a*n)/(1 - a)^3"
    );
    assert_eq!(
      interpret("Variance[BorelTannerDistribution[a, 2]]").unwrap(),
      "(2*a)/(1 - a)^3"
    );
    assert_eq!(
      interpret("Variance[BorelTannerDistribution[1/2, 2]]").unwrap(),
      "8"
    );
    // wolframscript leaves the CDF symbolic too
    assert_eq!(
      interpret("CDF[BorelTannerDistribution[1/2, 2], x]").unwrap(),
      "CDF[BorelTannerDistribution[1/2, 2], x]"
    );
  }
}

mod benktander_gibrat_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    // Symbolic keeps x^(-2 - a) in the numerator with the f1*f2 order
    assert_eq!(
      interpret("PDF[BenktanderGibratDistribution[a, b], x]").unwrap(),
      "Piecewise[{{(x^(-2 - a)*((-2*b)/a + (1 + a + 2*b*Log[x])*(1 + (2*b*Log[x])/a)))/E^(b*Log[x]^2), x >= 1}}, 0]"
    );
    // Numeric parameters hoist the x power and swap the factor order
    assert_eq!(
      interpret("PDF[BenktanderGibratDistribution[1, 1/2], x]").unwrap(),
      "Piecewise[{{(-1 + (1 + Log[x])*(2 + Log[x]))/(E^(Log[x]^2/2)*x^3), x >= 1}}, 0]"
    );
    assert_eq!(
      interpret("PDF[BenktanderGibratDistribution[2, 1/2], x]").unwrap(),
      "Piecewise[{{(-1/2 + (1 + Log[x]/2)*(3 + Log[x]))/(E^(Log[x]^2/2)*x^4), x >= 1}}, 0]"
    );
    assert_eq!(
      interpret("PDF[BenktanderGibratDistribution[1, 1/2], 2]").unwrap(),
      "(-1 + (1 + Log[2])*(2 + Log[2]))/(8*E^(Log[2]^2/2))"
    );
    assert_eq!(
      interpret("PDF[BenktanderGibratDistribution[1, 1/2], 1/2]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[BenktanderGibratDistribution[a, b], x]").unwrap(),
      "Piecewise[{{1 - (x^(-1 - a)*(1 + (2*b*Log[x])/a))/E^(b*Log[x]^2), x >= 1}}, 0]"
    );
    assert_eq!(
      interpret("CDF[BenktanderGibratDistribution[1, 1/2], x]").unwrap(),
      "Piecewise[{{1 - (1 + Log[x])/(E^(Log[x]^2/2)*x^2), x >= 1}}, 0]"
    );
    assert_eq!(
      interpret("CDF[BenktanderGibratDistribution[1, 1/2], 2]").unwrap(),
      "1 - (1 + Log[2])/(4*E^(Log[2]^2/2))"
    );
  }

  #[test]
  fn moments_and_validation() {
    assert_eq!(
      interpret("Mean[BenktanderGibratDistribution[a, b]]").unwrap(),
      "1 + a^(-1)"
    );
    assert_eq!(
      interpret("Mean[BenktanderGibratDistribution[1, 1/2]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Variance[BenktanderGibratDistribution[a, b]]").unwrap(),
      "(-1 + (a*E^((-1 + a)^2/(4*b))*Sqrt[Pi]*Erfc[(-1 + a)/(2*Sqrt[b])])/Sqrt[b])/a^2"
    );
    // Numeric b merges Sqrt[Pi/b]
    assert_eq!(
      interpret("Variance[BenktanderGibratDistribution[1, 1/2]]").unwrap(),
      "-1 + Sqrt[2*Pi]"
    );
    assert_eq!(
      interpret("Variance[BenktanderGibratDistribution[2, 1/2]]").unwrap(),
      "(-1 + 2*Sqrt[2*E*Pi]*Erfc[1/Sqrt[2]])/4"
    );
    // b > a (a + 1)/2 emits BenktanderGibratDistribution::lsseq
    assert_eq!(
      interpret("PDF[BenktanderGibratDistribution[1, 2], x]").unwrap(),
      "PDF[BenktanderGibratDistribution[1, 2], x]"
    );
  }

  // Invalid parameters must leave Mean/Variance unevaluated (after the
  // ::lsseq message), like PDF — not surface as an evaluation error.
  #[test]
  fn mean_variance_invalid_params_unevaluated() {
    assert_eq!(
      interpret("Mean[BenktanderGibratDistribution[1, 2]]").unwrap(),
      "Mean[BenktanderGibratDistribution[1, 2]]"
    );
    assert_eq!(
      interpret("Variance[BenktanderGibratDistribution[1, 2]]").unwrap(),
      "Variance[BenktanderGibratDistribution[1, 2]]"
    );
  }
}

mod gumbel_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    assert_eq!(
      interpret("PDF[GumbelDistribution[a, b], x]").unwrap(),
      "E^(-E^((-a + x)/b) + (-a + x)/b)/b"
    );
    assert_eq!(
      interpret("PDF[GumbelDistribution[0, 1], x]").unwrap(),
      "E^(-E^x + x)"
    );
    assert_eq!(
      interpret("PDF[GumbelDistribution[], x]").unwrap(),
      "E^(-E^x + x)"
    );
    assert_eq!(
      interpret("PDF[GumbelDistribution[2, 3], x]").unwrap(),
      "E^(-E^((-2 + x)/3) + (-2 + x)/3)/3"
    );
    assert_eq!(
      interpret("PDF[GumbelDistribution[0, 1], 1]").unwrap(),
      "E^(1 - E)"
    );
  }

  #[test]
  fn cdf_forms() {
    assert_eq!(
      interpret("CDF[GumbelDistribution[a, b], x]").unwrap(),
      "1 - E^(-E^((-a + x)/b))"
    );
    assert_eq!(
      interpret("CDF[GumbelDistribution[0, 1], x]").unwrap(),
      "1 - E^(-E^x)"
    );
    assert_eq!(
      interpret("CDF[GumbelDistribution[2, 3], x]").unwrap(),
      "1 - E^(-E^((-2 + x)/3))"
    );
    assert_eq!(
      interpret("CDF[GumbelDistribution[0, 1], 0]").unwrap(),
      "1 - E^(-1)"
    );
  }

  #[test]
  fn moments() {
    assert_eq!(
      interpret("Mean[GumbelDistribution[a, b]]").unwrap(),
      "a - b*EulerGamma"
    );
    assert_eq!(
      interpret("Mean[GumbelDistribution[2, 3]]").unwrap(),
      "2 - 3*EulerGamma"
    );
    assert_eq!(
      interpret("Variance[GumbelDistribution[a, b]]").unwrap(),
      "(b^2*Pi^2)/6"
    );
    assert_eq!(
      interpret("Variance[GumbelDistribution[2, 3]]").unwrap(),
      "(3*Pi^2)/2"
    );
  }
}

mod zipf_distribution {
  use super::*;

  #[test]
  fn pdf_forms() {
    assert_eq!(
      interpret("PDF[ZipfDistribution[r], x]").unwrap(),
      "Piecewise[{{x^(-1 - r)/Zeta[1 + r], x >= 1}}, 0]"
    );
    // Zeta[2] collapses with the rational hoisted
    assert_eq!(
      interpret("PDF[ZipfDistribution[1], x]").unwrap(),
      "Piecewise[{{6/(Pi^2*x^2), x >= 1}}, 0]"
    );
    assert_eq!(
      interpret("PDF[ZipfDistribution[1], 3]").unwrap(),
      "2/(3*Pi^2)"
    );
    // Non-integer points vanish
    assert_eq!(interpret("PDF[ZipfDistribution[1], 1/2]").unwrap(), "0");
    assert_eq!(interpret("PDF[ZipfDistribution[1], 5/2]").unwrap(), "0");
    // The bounded form normalizes by HarmonicNumber
    assert_eq!(
      interpret("PDF[ZipfDistribution[n, r], x]").unwrap(),
      "Piecewise[{{x^(-1 - r)/HarmonicNumber[n, 1 + r], 1 <= x <= n}}, 0]"
    );
    assert_eq!(
      interpret("PDF[ZipfDistribution[5, 1], x]").unwrap(),
      "Piecewise[{{3600/(5269*x^2), 1 <= x <= 5}}, 0]"
    );
  }

  #[test]
  fn moments_with_existence_thresholds() {
    // Mean needs r > 1, variance r > 2
    assert_eq!(
      interpret("Mean[ZipfDistribution[r]]").unwrap(),
      "Piecewise[{{Zeta[r]/Zeta[1 + r], r > 1}}, Infinity]"
    );
    assert_eq!(
      interpret("Mean[ZipfDistribution[2]]").unwrap(),
      "Pi^2/(6*Zeta[3])"
    );
    assert_eq!(interpret("Mean[ZipfDistribution[1]]").unwrap(), "Infinity");
    assert_eq!(
      interpret("Variance[ZipfDistribution[r]]").unwrap(),
      "Piecewise[{{-(Zeta[r]^2/Zeta[1 + r]^2) + Zeta[-1 + r]/Zeta[1 + r], r > 2}}, Infinity]"
    );
    assert_eq!(
      interpret("Variance[ZipfDistribution[2]]").unwrap(),
      "Infinity"
    );
    // Bounded form gives exact rationals
    assert_eq!(
      interpret("Mean[ZipfDistribution[5, 1]]").unwrap(),
      "8220/5269"
    );
    assert_eq!(
      interpret("Variance[ZipfDistribution[5, 1]]").unwrap(),
      "27273600/27762361"
    );
  }
}

mod distribution_moments {
  use super::*;

  // Moment[dist, n] is the raw moment E[x^n].
  #[test]
  fn raw_moments() {
    assert_eq!(
      interpret("Moment[NormalDistribution[0, 1], 4]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Moment[ExponentialDistribution[2], 3]").unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("Moment[UniformDistribution[{0, 1}], 3]").unwrap(),
      "1/4"
    );
    assert_eq!(
      interpret("Moment[PoissonDistribution[m], 3]").unwrap(),
      "m + 3*m^2 + m^3"
    );
  }

  // CentralMoment[dist, n] = E[(x - mean)^n].
  #[test]
  fn central_moments_numeric() {
    assert_eq!(
      interpret("CentralMoment[NormalDistribution[0, 1], 4]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("CentralMoment[ExponentialDistribution[2], 3]").unwrap(),
      "1/4"
    );
    assert_eq!(
      interpret("CentralMoment[ExponentialDistribution[1], 4]").unwrap(),
      "9"
    );
    assert_eq!(
      interpret("CentralMoment[UniformDistribution[{0, 1}], 4]").unwrap(),
      "1/80"
    );
    assert_eq!(
      interpret("CentralMoment[GammaDistribution[2, 3], 2]").unwrap(),
      "18"
    );
  }

  #[test]
  fn central_moments_symbolic_parameters() {
    assert_eq!(
      interpret("CentralMoment[PoissonDistribution[m], 2]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("CentralMoment[PoissonDistribution[m], 3]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("CentralMoment[NormalDistribution[mu, sigma], 4]").unwrap(),
      "3*sigma^4"
    );
    assert_eq!(
      interpret("CentralMoment[ExponentialDistribution[a], 3]").unwrap(),
      "2/a^3"
    );
  }

  // Skewness = m3 / m2^(3/2).
  #[test]
  fn skewness() {
    assert_eq!(
      interpret("Skewness[ExponentialDistribution[2]]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Skewness[PoissonDistribution[m]]").unwrap(),
      "1/Sqrt[m]"
    );
    assert_eq!(
      interpret("Skewness[NormalDistribution[mu, sigma]]").unwrap(),
      "0"
    );
  }

  // Kurtosis = m4 / m2^2.
  #[test]
  fn kurtosis() {
    assert_eq!(
      interpret("Kurtosis[NormalDistribution[0, 1]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Kurtosis[NormalDistribution[mu, sigma]]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Kurtosis[ExponentialDistribution[1]]").unwrap(),
      "9"
    );
    assert_eq!(
      interpret("Kurtosis[PoissonDistribution[m]]").unwrap(),
      "3 + m^(-1)"
    );
  }
}

mod distribution_cumulants {
  use super::*;

  // First cumulant is the mean, second is the variance.
  #[test]
  fn low_order() {
    assert_eq!(
      interpret("Cumulant[PoissonDistribution[m], 1]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("Cumulant[PoissonDistribution[m], 2]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("Cumulant[NormalDistribution[mu, sigma], 1]").unwrap(),
      "mu"
    );
    assert_eq!(
      interpret("Cumulant[NormalDistribution[mu, sigma], 2]").unwrap(),
      "sigma^2"
    );
    assert_eq!(
      interpret("Cumulant[UniformDistribution[{0, 1}], 2]").unwrap(),
      "1/12"
    );
  }

  // All Poisson cumulants equal the rate; Normal cumulants vanish past order 2.
  #[test]
  fn higher_order_symbolic() {
    assert_eq!(
      interpret("Cumulant[PoissonDistribution[m], 3]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("Cumulant[PoissonDistribution[m], 4]").unwrap(),
      "m"
    );
    assert_eq!(
      interpret("Cumulant[NormalDistribution[0, 1], 3]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Cumulant[NormalDistribution[mu, sigma], 4]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Cumulant[ExponentialDistribution[a], 2]").unwrap(),
      "a^(-2)"
    );
  }

  // Exponential cumulants: kappa_n = (n-1)! / lambda^n.
  #[test]
  fn exponential_numeric() {
    assert_eq!(
      interpret("Cumulant[ExponentialDistribution[1], 3]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Cumulant[ExponentialDistribution[1], 4]").unwrap(),
      "6"
    );
  }
}

mod binomial_distribution_pdf {
  use super::*;

  #[test]
  fn exact_values() {
    assert_eq!(
      interpret("PDF[BinomialDistribution[10, 1/2], 5]").unwrap(),
      "63/256"
    );
    assert_eq!(
      interpret("PDF[BinomialDistribution[10, 1/3], 4]").unwrap(),
      "4480/19683"
    );
    assert_eq!(
      interpret("PDF[BinomialDistribution[5, 1/4], 2]").unwrap(),
      "135/512"
    );
  }

  #[test]
  fn boundaries() {
    assert_eq!(
      interpret("PDF[BinomialDistribution[10, 1/2], 0]").unwrap(),
      "1/1024"
    );
    assert_eq!(
      interpret("PDF[BinomialDistribution[10, 1/2], 10]").unwrap(),
      "1/1024"
    );
  }

  // Outside the support 0 <= k <= n the density is zero.
  #[test]
  fn out_of_support_is_zero() {
    assert_eq!(
      interpret("PDF[BinomialDistribution[10, 1/2], 11]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("PDF[BinomialDistribution[10, 1/2], -1]").unwrap(),
      "0"
    );
  }

  #[test]
  fn symbolic_piecewise() {
    assert_eq!(
      interpret("PDF[BinomialDistribution[n, p], k]").unwrap(),
      "Piecewise[{{(1 - p)^(-k + n)*p^k*Binomial[n, k], 0 <= k <= n}}, 0]"
    );
    assert_eq!(
      interpret("PDF[BinomialDistribution[10, 1/2], x]").unwrap(),
      "Piecewise[{{Binomial[10, x]/1024, 0 <= x <= 10}}, 0]"
    );
  }
}

mod binomial_distribution_cdf {
  use super::*;

  #[test]
  fn exact_values() {
    assert_eq!(
      interpret("CDF[BinomialDistribution[10, 1/2], 5]").unwrap(),
      "319/512"
    );
    assert_eq!(
      interpret("CDF[BinomialDistribution[10, 1/3], 4]").unwrap(),
      "15488/19683"
    );
    assert_eq!(
      interpret("CDF[BinomialDistribution[5, 1/4], 2]").unwrap(),
      "459/512"
    );
    assert_eq!(
      interpret("CDF[BinomialDistribution[10, 1/2], 0]").unwrap(),
      "1/1024"
    );
  }

  // At and beyond the top of the support the CDF saturates at 1; below 0 it is 0.
  #[test]
  fn saturation() {
    assert_eq!(
      interpret("CDF[BinomialDistribution[10, 1/2], 10]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("CDF[BinomialDistribution[10, 1/2], 11]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("CDF[BinomialDistribution[10, 1/2], -1]").unwrap(),
      "0"
    );
  }

  // CDF and PDF are consistent: F(k) - F(k-1) = f(k).
  #[test]
  fn consistent_with_pdf() {
    assert_eq!(
      interpret(
        "CDF[BinomialDistribution[10, 1/3], 4] \
         - CDF[BinomialDistribution[10, 1/3], 3] \
         == PDF[BinomialDistribution[10, 1/3], 4]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod discrete_quantile {
  use super::*;

  // Quantile[dist, q] is the smallest integer k with CDF[k] >= q.
  #[test]
  fn binomial() {
    assert_eq!(
      interpret("Quantile[BinomialDistribution[10, 1/2], 1/2]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("Quantile[BinomialDistribution[10, 1/2], 3/4]").unwrap(),
      "6"
    );
    // The least positive probability still lands on the support minimum.
    assert_eq!(
      interpret("Quantile[BinomialDistribution[10, 1/2], 1/1024]").unwrap(),
      "0"
    );
  }

  #[test]
  fn poisson_and_geometric() {
    assert_eq!(
      interpret("Quantile[PoissonDistribution[3], 1/2]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Quantile[PoissonDistribution[3], 99/100]").unwrap(),
      "8"
    );
    assert_eq!(
      interpret("Quantile[GeometricDistribution[1/3], 1/2]").unwrap(),
      "1"
    );
  }

  #[test]
  fn bernoulli_and_discrete_uniform() {
    assert_eq!(
      interpret("Quantile[BernoulliDistribution[1/4], 3/4]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Quantile[DiscreteUniformDistribution[{1, 6}], 1/2]").unwrap(),
      "3"
    );
  }

  // q = 0 gives the support minimum, q = 1 the maximum (Infinity if unbounded).
  #[test]
  fn boundary_probabilities() {
    assert_eq!(
      interpret("Quantile[BinomialDistribution[10, 1/2], 0]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Quantile[BinomialDistribution[10, 1/2], 1]").unwrap(),
      "10"
    );
    assert_eq!(
      interpret("Quantile[DiscreteUniformDistribution[{1, 6}], 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Quantile[DiscreteUniformDistribution[{1, 6}], 1]").unwrap(),
      "6"
    );
    assert_eq!(
      interpret("Quantile[PoissonDistribution[3], 1]").unwrap(),
      "Infinity"
    );
  }

  // InverseCDF coincides with Quantile for discrete distributions.
  #[test]
  fn inverse_cdf_matches_quantile() {
    assert_eq!(
      interpret("InverseCDF[BinomialDistribution[10, 1/2], 1/2]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("InverseCDF[PoissonDistribution[3], 0.9]").unwrap(),
      "5"
    );
  }
}

// Closed-form Quantile (inverse CDF) of location-scale distributions with
// elementary quantile functions, for exact probabilities.
mod distribution_quantile_closed_forms {
  use super::*;

  #[test]
  fn weibull() {
    assert_eq!(
      interpret("Quantile[WeibullDistribution[2, 1], 1/2]").unwrap(),
      "Sqrt[Log[2]]"
    );
    // Lists of probabilities thread.
    assert_eq!(
      interpret("Quantile[WeibullDistribution[2, 1], {1/4, 3/4}]").unwrap(),
      "{Sqrt[Log[4/3]], Sqrt[Log[4]]}"
    );
    assert_eq!(
      interpret("Quantile[WeibullDistribution[2, 3], 1]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn pareto() {
    assert_eq!(
      interpret("Quantile[ParetoDistribution[1, 2], 1/2]").unwrap(),
      "Sqrt[2]"
    );
    // The lower bound is the minimum k.
    assert_eq!(
      interpret("Quantile[ParetoDistribution[1, 2], 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn rayleigh() {
    assert_eq!(
      interpret("Quantile[RayleighDistribution[1], 1/2]").unwrap(),
      "Sqrt[Log[4]]"
    );
  }

  #[test]
  fn laplace() {
    // Piecewise about q = 1/2.
    assert_eq!(
      interpret("Quantile[LaplaceDistribution[0, 1], 1/4]").unwrap(),
      "-Log[2]"
    );
    assert_eq!(
      interpret("Quantile[LaplaceDistribution[0, 1], 3/4]").unwrap(),
      "Log[2]"
    );
    assert_eq!(
      interpret("Quantile[LaplaceDistribution[0, 1], 0]").unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn logistic() {
    assert_eq!(
      interpret("Quantile[LogisticDistribution[0, 1], 1/4]").unwrap(),
      "-Log[3]"
    );
  }

  #[test]
  fn gumbel() {
    assert_eq!(
      interpret("Quantile[GumbelDistribution[0, 1], 1/2]").unwrap(),
      "Log[Log[2]]"
    );
  }
}

mod inverse_survival_function {
  use super::*;

  // InverseSurvivalFunction[dist, q] is the x with SurvivalFunction == q,
  // i.e. InverseCDF[dist, 1 - q]. The gamma/normal family uses the native
  // survival parametrization.
  #[test]
  fn normal_native_form() {
    assert_eq!(
      interpret("InverseSurvivalFunction[NormalDistribution[0, 1], 1/4]")
        .unwrap(),
      "Sqrt[2]*InverseErfc[1/2]"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[NormalDistribution[m, s], 1/4]")
        .unwrap(),
      "m + Sqrt[2]*s*InverseErfc[1/2]"
    );
    // The median survival point is the mean.
    assert_eq!(
      interpret("InverseSurvivalFunction[NormalDistribution[0, 1], 1/2]")
        .unwrap(),
      "0"
    );
  }

  #[test]
  fn normal_boundaries() {
    assert_eq!(
      interpret("InverseSurvivalFunction[NormalDistribution[0, 1], 0]")
        .unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[NormalDistribution[0, 1], 1]")
        .unwrap(),
      "-Infinity"
    );
  }

  // GammaDistribution uses the 2-arg upper InverseGammaRegularized directly.
  #[test]
  fn gamma_family_native_form() {
    assert_eq!(
      interpret("InverseSurvivalFunction[GammaDistribution[2, 3], 1/4]")
        .unwrap(),
      "3*InverseGammaRegularized[2, 1/4]"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[GammaDistribution[2, 3], 0]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[GammaDistribution[2, 3], 1]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[ChiSquareDistribution[3], 1/4]")
        .unwrap(),
      "2*InverseGammaRegularized[3/2, 1/4]"
    );
  }

  // Distributions whose survival inverse equals InverseCDF[dist, 1 - q] in form.
  #[test]
  fn elementary_delegation() {
    assert_eq!(
      interpret("InverseSurvivalFunction[ExponentialDistribution[2], 1/4]")
        .unwrap(),
      "Log[4]/2"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[WeibullDistribution[2, 3], 1/4]")
        .unwrap(),
      "3*Sqrt[Log[4]]"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[CauchyDistribution[0, 1], 1/4]")
        .unwrap(),
      "1"
    );
  }

  #[test]
  fn numeric_values() {
    let g: f64 =
      interpret("N[InverseSurvivalFunction[GammaDistribution[2, 3], 1/4]]")
        .unwrap()
        .parse()
        .unwrap();
    assert!((g - 8.077903586669088).abs() < 1e-8, "got {g}");

    let n: f64 =
      interpret("N[InverseSurvivalFunction[NormalDistribution[0, 1], 1/4]]")
        .unwrap()
        .parse()
        .unwrap();
    assert!((n - 0.6744897501960817).abs() < 1e-8, "got {n}");
  }

  // An unsupported distribution or non-distribution stays unevaluated.
  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("Head[InverseSurvivalFunction[BetaDistribution[2, 3], 1/4]]")
        .unwrap(),
      "InverseSurvivalFunction"
    );
    assert_eq!(
      interpret("InverseSurvivalFunction[x, 1/4]").unwrap(),
      "InverseSurvivalFunction[x, 1/4]"
    );
  }
}

// ErlangDistribution[k, λ] is identical to GammaDistribution[k, 1/λ]
// (Erlang parameterises by rate, Gamma by scale). Every property reuses the
// Gamma machinery; the bare head stays unevaluated like wolframscript.
mod erlang_distribution {
  use super::*;

  #[test]
  fn head_stays_unevaluated() {
    assert_eq!(
      interpret("ErlangDistribution[3, 2]").unwrap(),
      "ErlangDistribution[3, 2]"
    );
  }

  #[test]
  fn mean() {
    assert_eq!(interpret("Mean[ErlangDistribution[3, 2]]").unwrap(), "3/2");
    assert_eq!(interpret("Mean[ErlangDistribution[k, l]]").unwrap(), "k/l");
  }

  #[test]
  fn variance() {
    assert_eq!(
      interpret("Variance[ErlangDistribution[3, 2]]").unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("Variance[ErlangDistribution[k, l]]").unwrap(),
      "k/l^2"
    );
  }

  #[test]
  fn standard_deviation() {
    assert_eq!(
      interpret("StandardDeviation[ErlangDistribution[3, 2]]").unwrap(),
      "Sqrt[3]/2"
    );
    assert_eq!(
      interpret("StandardDeviation[ErlangDistribution[k, lambda]]").unwrap(),
      "Sqrt[k]/lambda"
    );
  }

  #[test]
  fn pdf() {
    assert_eq!(
      interpret("PDF[ErlangDistribution[2, 1], x]").unwrap(),
      "Piecewise[{{x/E^x, x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[ErlangDistribution[3, 2], 1]").unwrap(),
      "4/E^2"
    );
  }

  #[test]
  fn cdf() {
    assert_eq!(
      interpret("CDF[ErlangDistribution[2, 1], 1]").unwrap(),
      "GammaRegularized[2, 0, 1]"
    );
    assert_eq!(
      interpret("CDF[ErlangDistribution[3, 2], x]").unwrap(),
      "Piecewise[{{GammaRegularized[3, 0, 2*x], x > 0}}, 0]"
    );
  }

  #[test]
  fn quantile() {
    assert_eq!(
      interpret("Quantile[ErlangDistribution[2, 1], 1/2]").unwrap(),
      "InverseGammaRegularized[2, 0, 1/2]"
    );
    assert_eq!(
      interpret("Quantile[ErlangDistribution[3, 2], 1/3]").unwrap(),
      "InverseGammaRegularized[3, 0, 1/3]/2"
    );
  }
}

// SkellamDistribution[a, b] is the difference of two Poisson variables:
// Mean = a-b, Variance = a+b. The bare head stays unevaluated.
mod skellam_distribution {
  use super::*;

  #[test]
  fn head_stays_unevaluated() {
    assert_eq!(
      interpret("SkellamDistribution[1, 2]").unwrap(),
      "SkellamDistribution[1, 2]"
    );
  }

  #[test]
  fn mean_variance_sd() {
    assert_eq!(
      interpret("Mean[SkellamDistribution[a, b]]").unwrap(),
      "a - b"
    );
    assert_eq!(
      interpret("Variance[SkellamDistribution[a, b]]").unwrap(),
      "a + b"
    );
    assert_eq!(
      interpret("StandardDeviation[SkellamDistribution[a, b]]").unwrap(),
      "Sqrt[a + b]"
    );
    assert_eq!(interpret("Mean[SkellamDistribution[1, 2]]").unwrap(), "-1");
    assert_eq!(
      interpret("Variance[SkellamDistribution[1, 2]]").unwrap(),
      "3"
    );
  }

  #[test]
  fn skewness_kurtosis() {
    assert_eq!(
      interpret("Skewness[SkellamDistribution[a, b]]").unwrap(),
      "(a - b)/(a + b)^(3/2)"
    );
    assert_eq!(
      interpret("Kurtosis[SkellamDistribution[a, b]]").unwrap(),
      "3 + (a + b)^(-1)"
    );
    assert_eq!(
      interpret("Kurtosis[SkellamDistribution[1, 2]]").unwrap(),
      "10/3"
    );
  }

  #[test]
  fn central_moments() {
    assert_eq!(
      interpret("CentralMoment[SkellamDistribution[a, b], 2]").unwrap(),
      "a + b"
    );
    assert_eq!(
      interpret("CentralMoment[SkellamDistribution[a, b], 3]").unwrap(),
      "a - b"
    );
    assert_eq!(
      interpret("CentralMoment[SkellamDistribution[a, b], 4]").unwrap(),
      "a + b + 3*(a + b)^2"
    );
  }

  #[test]
  fn pdf() {
    assert_eq!(
      interpret("PDF[SkellamDistribution[a, b], k]").unwrap(),
      "(a/b)^(k/2)*E^(-a - b)*BesselI[k, 2*Sqrt[a*b]]"
    );
    assert_eq!(
      interpret("PDF[SkellamDistribution[1, 2], 0]").unwrap(),
      "BesselI[0, 2*Sqrt[2]]/E^3"
    );
    // Negative k uses the integer-order symmetry BesselI[-1, z] = BesselI[1, z].
    assert_eq!(
      interpret("PDF[SkellamDistribution[2, 3], -1]").unwrap(),
      "(Sqrt[3/2]*BesselI[1, 2*Sqrt[6]])/E^5"
    );
  }

  #[test]
  fn cdf() {
    assert_eq!(
      interpret("CDF[SkellamDistribution[a, b], k]").unwrap(),
      "1 - MarcumQ[-Floor[k], Sqrt[2]*Sqrt[a], Sqrt[2]*Sqrt[b]]"
    );
    assert_eq!(
      interpret("CDF[SkellamDistribution[1, 2], 0]").unwrap(),
      "1 - MarcumQ[0, Sqrt[2], 2]"
    );
  }
}

// PolyaAeppliDistribution[t, p] (geometric-Poisson): Mean = t/(1-p),
// Variance = (1+p) t/(1-p)^2. Moment properties match wolframscript; PDF/CDF
// (Hypergeometric1F1 / factor-order forms) are left unevaluated.
mod polya_aeppli_distribution {
  use super::*;

  #[test]
  fn head_stays_unevaluated() {
    assert_eq!(
      interpret("PolyaAeppliDistribution[2, 1/2]").unwrap(),
      "PolyaAeppliDistribution[2, 1/2]"
    );
  }

  #[test]
  fn mean_variance_sd() {
    assert_eq!(
      interpret("Mean[PolyaAeppliDistribution[t, p]]").unwrap(),
      "t/(1 - p)"
    );
    assert_eq!(
      interpret("Variance[PolyaAeppliDistribution[t, p]]").unwrap(),
      "((1 + p)*t)/(1 - p)^2"
    );
    assert_eq!(
      interpret("StandardDeviation[PolyaAeppliDistribution[t, p]]").unwrap(),
      "Sqrt[(1 + p)*t]/(1 - p)"
    );
    assert_eq!(
      interpret("Mean[PolyaAeppliDistribution[2, 1/2]]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Variance[PolyaAeppliDistribution[2, 1/2]]").unwrap(),
      "12"
    );
  }

  #[test]
  fn skewness_kurtosis() {
    assert_eq!(
      interpret("Skewness[PolyaAeppliDistribution[t, p]]").unwrap(),
      "(1 + 4*p + p^2)/((1 + p)*Sqrt[(1 + p)*t])"
    );
    assert_eq!(
      interpret("Kurtosis[PolyaAeppliDistribution[t, p]]").unwrap(),
      "3 + (1 + 10*p + p^2)/((1 + p)*t)"
    );
    assert_eq!(
      interpret("Skewness[PolyaAeppliDistribution[2, 1/2]]").unwrap(),
      "13/(6*Sqrt[3])"
    );
    assert_eq!(
      interpret("Kurtosis[PolyaAeppliDistribution[2, 1/2]]").unwrap(),
      "61/12"
    );
  }
}

// ChiDistribution[k]: Mean = Sqrt[2] Gamma[(1+k)/2]/Gamma[k/2],
// Variance = k - 2 Gamma[(1+k)/2]^2/Gamma[k/2]^2. The half-integer Gammas are
// re-combined via Simplify so the Sqrt forms match wolframscript.
mod chi_distribution {
  use super::*;

  #[test]
  fn mean_symbolic_and_numeric() {
    assert_eq!(
      interpret("Mean[ChiDistribution[k]]").unwrap(),
      "(Sqrt[2]*Gamma[(1 + k)/2])/Gamma[k/2]"
    );
    assert_eq!(interpret("Mean[ChiDistribution[1]]").unwrap(), "Sqrt[2/Pi]");
    assert_eq!(interpret("Mean[ChiDistribution[2]]").unwrap(), "Sqrt[Pi/2]");
    assert_eq!(
      interpret("Mean[ChiDistribution[3]]").unwrap(),
      "2*Sqrt[2/Pi]"
    );
    assert_eq!(
      interpret("Mean[ChiDistribution[4]]").unwrap(),
      "(3*Sqrt[Pi/2])/2"
    );
  }

  #[test]
  fn variance_and_standard_deviation() {
    assert_eq!(
      interpret("Variance[ChiDistribution[k]]").unwrap(),
      "k - (2*Gamma[(1 + k)/2]^2)/Gamma[k/2]^2"
    );
    assert_eq!(
      interpret("Variance[ChiDistribution[3]]").unwrap(),
      "3 - 8/Pi"
    );
    assert_eq!(
      interpret("StandardDeviation[ChiDistribution[3]]").unwrap(),
      "Sqrt[3 - 8/Pi]"
    );
  }

  #[test]
  fn pdf_and_cdf_symbolic() {
    assert_eq!(
      interpret("PDF[ChiDistribution[k], x]").unwrap(),
      "Piecewise[{{(2^(1 - k/2)*x^(-1 + k))/(E^(x^2/2)*Gamma[k/2]), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[ChiDistribution[k], x]").unwrap(),
      "Piecewise[{{GammaRegularized[k/2, 0, x^2/2], x > 0}}, 0]"
    );
    assert_eq!(
      interpret("PDF[ChiDistribution[2], x]").unwrap(),
      "Piecewise[{{x/E^(x^2/2), x > 0}}, 0]"
    );
    assert_eq!(
      interpret("CDF[ChiDistribution[2], 1]").unwrap(),
      "1 - 1/Sqrt[E]"
    );
  }
}

// BetaNegativeBinomialDistribution[a, b, n]: Mean and Variance are Piecewise
// (finite only for a > 1 / a > 2, else Infinity).
mod beta_negative_binomial_distribution {
  use super::*;

  #[test]
  fn mean() {
    assert_eq!(
      interpret("Mean[BetaNegativeBinomialDistribution[a, b, n]]").unwrap(),
      "Piecewise[{{(b*n)/(-1 + a), a > 1}}, Infinity]"
    );
    assert_eq!(
      interpret("Mean[BetaNegativeBinomialDistribution[3, 2, 5]]").unwrap(),
      "5"
    );
  }

  #[test]
  fn variance() {
    assert_eq!(
      interpret("Variance[BetaNegativeBinomialDistribution[a, b, n]]").unwrap(),
      "Piecewise[{{(b*(-1 + a + b)*n*(-1 + a + n))/((-2 + a)*(-1 + a)^2), a > 2}}, Infinity]"
    );
    assert_eq!(
      interpret("Variance[BetaNegativeBinomialDistribution[5, 2, 3]]").unwrap(),
      "21/4"
    );
  }

  #[test]
  fn head_stays_unevaluated() {
    assert_eq!(
      interpret("BetaNegativeBinomialDistribution[a, b, n]").unwrap(),
      "BetaNegativeBinomialDistribution[a, b, n]"
    );
  }
}

// KumaraswamyDistribution[a, b] — a continuous distribution on (0, 1).
mod kumaraswamy_distribution {
  use super::*;

  #[test]
  fn mean_variance() {
    assert_eq!(
      interpret("Mean[KumaraswamyDistribution[a, b]]").unwrap(),
      "b*Beta[b, 1 + a^(-1)]"
    );
    assert_eq!(
      interpret("Variance[KumaraswamyDistribution[a, b]]").unwrap(),
      "-(b^2*Beta[b, 1 + a^(-1)]^2) + b*Beta[b, 1 + 2/a]"
    );
    assert_eq!(
      interpret("Mean[KumaraswamyDistribution[2, 3]]").unwrap(),
      "16/35"
    );
    assert_eq!(
      interpret("Variance[KumaraswamyDistribution[2, 3]]").unwrap(),
      "201/4900"
    );
  }

  #[test]
  fn pdf_and_cdf() {
    assert_eq!(
      interpret("PDF[KumaraswamyDistribution[a, b], x]").unwrap(),
      "Piecewise[{{a*b*x^(-1 + a)*(1 - x^a)^(-1 + b), 0 < x < 1}}, 0]"
    );
    assert_eq!(
      interpret("CDF[KumaraswamyDistribution[a, b], x]").unwrap(),
      "Piecewise[{{1 - (1 - x^a)^b, 0 < x < 1}, {1, x >= 1}}, 0]"
    );
    assert_eq!(
      interpret("CDF[KumaraswamyDistribution[2, 3], 1/2]").unwrap(),
      "37/64"
    );
  }

  #[test]
  fn head_stays_unevaluated() {
    assert_eq!(
      interpret("KumaraswamyDistribution[a, b]").unwrap(),
      "KumaraswamyDistribution[a, b]"
    );
  }
}

// ExpGammaDistribution[k, t, m] — the distribution of m + t Log[GammaVariate].
mod exp_gamma_distribution {
  use super::*;

  #[test]
  fn mean_variance_sd() {
    assert_eq!(
      interpret("Mean[ExpGammaDistribution[k, t, m]]").unwrap(),
      "m + t*PolyGamma[0, k]"
    );
    assert_eq!(
      interpret("Variance[ExpGammaDistribution[k, t, m]]").unwrap(),
      "t^2*PolyGamma[1, k]"
    );
    assert_eq!(
      interpret("StandardDeviation[ExpGammaDistribution[k, t, m]]").unwrap(),
      "Sqrt[t^2*PolyGamma[1, k]]"
    );
    assert_eq!(
      interpret("Mean[ExpGammaDistribution[2, 3, 1]]").unwrap(),
      "1 + 3*(1 - EulerGamma)"
    );
    assert_eq!(
      interpret("Variance[ExpGammaDistribution[2, 1, 0]]").unwrap(),
      "-1 + Pi^2/6"
    );
  }

  #[test]
  fn cdf() {
    assert_eq!(
      interpret("CDF[ExpGammaDistribution[k, t, m], x]").unwrap(),
      "GammaRegularized[k, 0, E^((-m + x)/t)]"
    );
  }

  #[test]
  fn head_stays_unevaluated() {
    assert_eq!(
      interpret("ExpGammaDistribution[k, t, m]").unwrap(),
      "ExpGammaDistribution[k, t, m]"
    );
  }
}
