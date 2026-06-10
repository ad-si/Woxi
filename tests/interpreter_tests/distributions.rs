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
