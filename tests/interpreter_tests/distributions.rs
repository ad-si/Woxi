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
