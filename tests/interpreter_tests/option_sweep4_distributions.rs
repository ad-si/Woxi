// Probability-distribution argument forms found missing by the
// reference-page parameter sweep (round 4). Expected values come from the
// examples on the Wolfram reference pages.

use super::*;

/// `interpret(code)` must evaluate to `True`.
fn assert_true(code: &str) {
  assert_eq!(interpret(code).unwrap(), "True", "{code}");
}

/// `|code - expected| < 10^-6`, for six-digit reference values.
fn assert_close(code: &str, expected: f64) {
  assert_true(&format!("Abs[({code}) - ({expected})] < 10^-6"));
}

mod probability_compound_events {
  use super::*;

  // These used to recurse until $RecursionLimit.

  #[test]
  fn quadratic_event_under_normal() {
    assert_true(
      "Probability[(x - 1)^2 <= 3 x, x \\[Distributed] NormalDistribution[]] \
       === (1/2) (-Erfc[(-5 + Sqrt[21])/(2 Sqrt[2])] + \
       Erfc[-(5 + Sqrt[21])/(2 Sqrt[2])])",
    );
    assert_true(
      "Probability[(x - 3)^2 < 5, x \\[Distributed] NormalDistribution[0, 1]] \
       === (1/2) (-Erfc[(-3 + Sqrt[5])/Sqrt[2]] + Erfc[-(3 + Sqrt[5])/Sqrt[2]])",
    );
    assert_true(
      "Probability[y^2 < 2, y \\[Distributed] NormalDistribution[0, 1]] === \
       (1/2) (Erfc[-1] - Erfc[1])",
    );
  }

  #[test]
  fn conjunction_reduces_to_one_tail() {
    assert_true(
      "Probability[x^2 > 1 && x > 1/2, \
       x \\[Distributed] LaplaceDistribution[0, 1/2]] === 1/(2 E^2)",
    );
  }

  #[test]
  fn conditioned_compound_event() {
    assert_true(
      "Probability[x^2 > 1 \\[Conditioned] x > 1/2, \
       x \\[Distributed] LaplaceDistribution[0, 1/2]] === 1/E",
    );
  }

  #[test]
  fn two_intervals_under_uniform() {
    assert_true(
      "Probability[3 < x^2 < 7, x \\[Distributed] UniformDistribution[{-2, 11}]] \
       === (1/13) (2 - 2 Sqrt[3] + Sqrt[7])",
    );
  }

  #[test]
  fn symbolic_rate_disjunction_and_conjunction() {
    // Wolfram writes the first as
    // -E^(-Sqrt[3] λ) (-1 - E^(Sqrt[3] λ) + E^(Sqrt[3] λ) Cosh[λ]
    //   - E^(Sqrt[3] λ) Sinh[λ]); the value must agree.
    assert_true(
      "p = Probability[x^2 > 3 || Abs[x] < 1, \
       x \\[Distributed] ExponentialDistribution[λ]]; \
       w = -E^(-Sqrt[3] λ) (-1 - E^(Sqrt[3] λ) + E^(Sqrt[3] λ) Cosh[λ] - \
       E^(Sqrt[3] λ) Sinh[λ]); \
       Max[Table[Abs[N[(p - w) /. λ -> l]], {l, {1/3, 1, 2, 7}}]] < 10^-12",
    );
    // P[1/Sqrt[2] < x < 2] = E^(-λ/Sqrt[2]) - E^(-2 λ).
    assert_true(
      "p = Probability[x < 2 && 1/2 < x^2 < 5, \
       x \\[Distributed] ExponentialDistribution[λ]]; \
       Max[Table[Abs[N[(p - (E^(-λ/Sqrt[2]) - E^(-2 λ))) /. λ -> l]], \
       {l, {1/3, 1, 2, 7}}]] < 10^-12",
    );
  }

  #[test]
  fn discrete_strict_bounds_exclude_the_end_points() {
    assert_true(
      "Probability[1 < x < 7, x \\[Distributed] PoissonDistribution[1]] === \
       517/(720 E)",
    );
    // P[x >= 2] = 1 - P[x <= 1].
    assert_true(
      "Probability[x >= 2, x \\[Distributed] PoissonDistribution[1]] == 1 - 2/E",
    );
    // x^2 < 5 keeps 0, 1, 2.
    assert_true(
      "Probability[x^2 < 5, x \\[Distributed] PoissonDistribution[3]] == \
       Total[Table[PDF[PoissonDistribution[3], k], {k, 0, 2}]]",
    );
  }

  #[test]
  fn opaque_event_stays_unevaluated() {
    assert_eq!(
      interpret(
        "f[x_?NumericQ] := x^3; \
         Probability[f[x] < 1, x \\[Distributed] NormalDistribution[]]"
      )
      .unwrap(),
      "Probability[f[x] < 1, Distributed[x, NormalDistribution[0, 1]]]"
    );
  }

  #[test]
  fn empirical_distribution_weighs_the_satisfying_values() {
    assert_eq!(
      interpret(
        "Probability[x^2 > 1, \
         x \\[Distributed] EmpiricalDistribution[{-2, 1/2, 3/2, 0, 3}]]"
      )
      .unwrap(),
      "3/5"
    );
  }
}

mod n_probability_compound_events {
  use super::*;

  #[test]
  fn reference_values() {
    assert_close(
      "NProbability[(x - 1)^2 <= 3 x, x \\[Distributed] NormalDistribution[]]",
      0.417336,
    );
    assert_close(
      "NProbability[x^2 > 3 || Abs[x] < 1, \
       x \\[Distributed] ExponentialDistribution[2]]",
      0.895966,
    );
    assert_close(
      "NProbability[x^2 > 1 \\[Conditioned] x > 1/2, \
       x \\[Distributed] LaplaceDistribution[0, 1/2]]",
      0.367879,
    );
    assert_close(
      "NProbability[1 < x < 7, x \\[Distributed] PoissonDistribution[1]]",
      0.264158,
    );
    assert_close(
      "NProbability[x < 2 \\[Conditioned] 1/2 < x^2 < 5, \
       x \\[Distributed] ExponentialDistribution[2]]",
      0.970251,
    );
    assert_close(
      "NProbability[x^2 > 12 \\[Conditioned] x > 2, \
       x \\[Distributed] PoissonDistribution[2]]",
      0.4419,
    );
    assert_close(
      "NProbability[(x^2 < 5 && (2 x + 3) < 5) || (x^3 + x > 11), \
       x \\[Distributed] ExponentialDistribution[2]]",
      0.88045,
    );
    assert_close(
      "NProbability[3 < x < 5, x \\[Distributed] NormalDistribution[]]",
      0.00134961,
    );
  }

  #[test]
  fn numeric_only_event_is_integrated() {
    assert_close(
      "f[x_?NumericQ] := x^3; \
       NProbability[f[x] < 1, x \\[Distributed] NormalDistribution[]]",
      0.841345,
    );
  }

  #[test]
  fn symbolic_event_stays_unevaluated() {
    assert_eq!(
      interpret(
        "NProbability[x^2 > a, x \\[Distributed] NormalDistribution[]]"
      )
      .unwrap(),
      "NProbability[x^2 > a, Distributed[x, NormalDistribution[0, 1]]]"
    );
  }
}

mod reduce_compound_univariate {
  use super::*;

  #[test]
  fn conjunction_with_a_chain() {
    assert_eq!(
      interpret("Reduce[x < 2 && 1/2 < x^2 < 5, x, Reals]").unwrap(),
      "Inequality[-Sqrt[5], Less, x, Less, -(1/Sqrt[2])] || \
       Inequality[1/Sqrt[2], Less, x, Less, 2]"
    );
    assert_eq!(
      interpret("Reduce[x < 2 && x^2 > 1/2, x]").unwrap(),
      "x < -(1/Sqrt[2]) || Inequality[1/Sqrt[2], Less, x, Less, 2]"
    );
  }

  #[test]
  fn disjunction_is_ordered_and_merged() {
    assert_eq!(
      interpret("Reduce[x^2 > 3 || Abs[x] < 1, x, Reals]").unwrap(),
      "x < -Sqrt[3] || Inequality[-1, Less, x, Less, 1] || x > Sqrt[3]"
    );
    assert_eq!(
      interpret("Reduce[x^2 > 1 || x < 1/2, x, Reals]").unwrap(),
      "x < 1/2 || x > 1"
    );
  }

  #[test]
  fn isolated_points_and_one_tail() {
    assert_eq!(
      interpret("Reduce[x^2 >= 1 && x <= 1, x, Reals]").unwrap(),
      "x <= -1 || x == 1"
    );
    assert_eq!(
      interpret("Reduce[x^2 > 1 && x > 1/2, x, Reals]").unwrap(),
      "x > 1"
    );
  }
}

mod operator_forms {
  use super::*;

  #[test]
  fn pdf_cdf_survival_of_a_distribution_are_pure_functions() {
    assert_true(
      "PDF[GammaDistribution[1, 2]] === Function[\\[FormalX], \
       Evaluate[Piecewise[{{1/(E^(\\[FormalX]/2)*2), \\[FormalX] > 0}}, 0]], \
       Listable]",
    );
    assert_true(
      "CDF[GammaDistribution[1, 2]] === Function[\\[FormalX], \
       Evaluate[Piecewise[{{1 - E^(-\\[FormalX]/2), \\[FormalX] > 0}}, 0]], \
       Listable]",
    );
    assert_true(
      "SurvivalFunction[GammaDistribution[1, 2]] === Function[\\[FormalX], \
       Evaluate[Piecewise[{{E^(-\\[FormalX]/2), \\[FormalX] > 0}}, 1]], \
       Listable]",
    );
  }

  #[test]
  fn the_function_is_listable() {
    assert_eq!(
      interpret("PDF[GammaDistribution[1, 2]][{1, 2}]").unwrap(),
      "{1/(2*Sqrt[E]), 1/(2*E)}"
    );
  }

  #[test]
  fn unknown_distribution_stays_unevaluated() {
    assert_eq!(interpret("PDF[foo]").unwrap(), "PDF[foo]");
  }
}

mod derived_distributions {
  use super::*;

  #[test]
  fn product_of_densities_on_the_joint_support() {
    assert_eq!(
      interpret(
        "PDF[ProductDistribution[{ExponentialDistribution[4], 3}, \
         NormalDistribution[]], {w, x, y, z}]"
      )
      .unwrap(),
      "Piecewise[{{32*E^(-4*w - 4*x - 4*y - z^2/2)*Sqrt[2/Pi], \
       w >= 0 && x >= 0 && y >= 0}}, 0]"
    );
  }

  #[test]
  fn censored_cdf_at_a_symbol() {
    assert_true(
      "CDF[CensoredDistribution[{-1, Infinity}, NormalDistribution[]], x] === \
       Piecewise[{{(1/2)*Erfc[-(x/Sqrt[2])], x >= -1}}, 0]",
    );
    assert_eq!(
      interpret(
        "CDF[CensoredDistribution[{2, 2}, ParetoDistribution[1, 1/3]], x]"
      )
      .unwrap(),
      "Piecewise[{{1, x >= 2}}, 0]"
    );
  }
}

mod multivariate_moments {
  use super::*;

  #[test]
  fn multivariate_hypergeometric_mean_and_variance() {
    assert_true(
      "Mean[MultivariateHypergeometricDistribution[n, {m1, m2}]] === \
       {(n m1)/(m1 + m2), (n m2)/(m1 + m2)}",
    );
    assert_true(
      "Variance[MultivariateHypergeometricDistribution[n, {m1, m2}]] === \
       Table[(n m1 m2 (-n + m1 + m2))/((-1 + m1 + m2) (m1 + m2)^2), 2]",
    );
    assert_true(
      "StandardDeviation[MultivariateHypergeometricDistribution[n, {m1, m2}]] \
       === Sqrt[Variance[MultivariateHypergeometricDistribution[n, {m1, m2}]]]",
    );
    assert_eq!(
      interpret("Mean[MultivariateHypergeometricDistribution[10, {9, 6, 3}]]")
        .unwrap(),
      "{5, 10/3, 5/3}"
    );
  }

  #[test]
  fn multivariate_poisson_covariance_and_correlation() {
    assert_eq!(
      interpret("Covariance[MultivariatePoissonDistribution[μ, {m1, m2, m3}]]")
        .unwrap(),
      "{{m1 + μ, μ, μ}, {μ, m2 + μ, μ}, {μ, μ, m3 + μ}}"
    );
    assert_true(
      "Correlation[MultivariatePoissonDistribution[μ, {m1, m2}]] === \
       {{1, μ/(Sqrt[μ + m1] Sqrt[μ + m2])}, {μ/(Sqrt[μ + m1] Sqrt[μ + m2]), 1}}",
    );
    assert_eq!(
      interpret("Covariance[MultivariatePoissonDistribution[μ, {2, 3}], 2, 1]")
        .unwrap(),
      "μ"
    );
    assert_true(
      "Correlation[MultivariatePoissonDistribution[μ, {2, 3}], 2, 1] === \
       μ/(Sqrt[2 + μ] Sqrt[3 + μ])",
    );
  }

  #[test]
  fn multinomial_covariance() {
    assert_true(
      "Covariance[MultinomialDistribution[n, {p1, p2}]] === \
       {{n (p1 - p1^2), -n p1 p2}, {-n p1 p2, n (p2 - p2^2)}}",
    );
  }
}

mod random_variates {
  use super::*;

  #[test]
  fn multivariate_t_draws_vectors() {
    assert_eq!(
      interpret(
        "Dimensions[RandomVariate[MultivariateTDistribution[\
         {{1, 1/2}, {1/2, 1}}, 5], 2]]"
      )
      .unwrap(),
      "{2, 2}"
    );
    // The covariance is Σ ν/(ν - 2) = 5/3 Σ, around the location.
    assert_true(
      "SeedRandom[7]; s = RandomVariate[MultivariateTDistribution[{1, 2}, \
       {{1, 1/2}, {1/2, 1}}, 5], 20000]; \
       Max[Abs[Mean[s] - {1, 2}]] < 0.05 && \
       Max[Abs[Flatten[Covariance[s] - 5/3 {{1, 1/2}, {1/2, 1}}]]] < 0.2",
    );
  }

  #[test]
  fn multivariate_t_needs_a_positive_definite_matrix() {
    assert_eq!(
      interpret(
        "RandomVariate[MultivariateTDistribution[{{1, 2}, {2, 1}}, 5]]"
      )
      .unwrap(),
      "RandomVariate[MultivariateTDistribution[{{1, 2}, {2, 1}}, 5]]"
    );
  }

  #[test]
  fn quantity_parameters_make_a_quantity_distribution() {
    assert_eq!(
      interpret(
        "NormalDistribution[Quantity[180, \"Centimeters\"], \
         Quantity[5, \"Centimeters\"]]"
      )
      .unwrap(),
      "QuantityDistribution[NormalDistribution[180, 5], Centimeters]"
    );
    assert_eq!(
      interpret("WeibullDistribution[1.65, Quantity[256.5, \"Days\"]]")
        .unwrap(),
      "QuantityDistribution[WeibullDistribution[1.65, 256.5], Days]"
    );
    assert_eq!(
      interpret("ParetoDistribution[Quantity[500., \"USDollars\"], 4]")
        .unwrap(),
      "QuantityDistribution[ParetoDistribution[500., 4], USDollars]"
    );
    assert_eq!(
      interpret("RayleighDistribution[Quantity[25, \"Watts\"]]").unwrap(),
      "QuantityDistribution[RayleighDistribution[25], Watts]"
    );
    assert_eq!(
      interpret(
        "UniformDistribution[{Quantity[1, \"Meters\"], Quantity[3, \"Meters\"]}]"
      )
      .unwrap(),
      "QuantityDistribution[UniformDistribution[{1, 3}], Meters]"
    );
    // Mixed units are left alone.
    assert_eq!(
      interpret(
        "NormalDistribution[Quantity[180, \"Centimeters\"], \
         Quantity[5, \"Meters\"]]"
      )
      .unwrap(),
      "NormalDistribution[Quantity[180, Centimeters], Quantity[5, Meters]]"
    );
  }

  #[test]
  fn variates_of_a_quantity_distribution_carry_the_unit() {
    assert_true(
      "q = RandomVariate[NormalDistribution[Quantity[180, \"Centimeters\"], \
       Quantity[5, \"Centimeters\"]]]; \
       Head[q] === Quantity && QuantityUnit[q] === \"Centimeters\" && \
       Head[QuantityMagnitude[q]] === Real",
    );
    assert_true(
      "q = RandomVariate[UniformDistribution[{Quantity[1, \"Meters\"], \
       Quantity[3, \"Meters\"]}]]; \
       QuantityUnit[q] === \"Meters\" && 1 <= QuantityMagnitude[q] <= 3",
    );
    assert_close(
      "QuantityMagnitude[Mean[WeibullDistribution[1.65, \
       Quantity[256.5, \"Days\"]]]]",
      229.3654452456232,
    );
  }
}

mod estimated_distribution {
  use super::*;

  const BDATA: &str = "bdata = {23, 16, 24, 17, 16, 17, 18, 17, 19, 23, 19, \
                       13, 19, 18, 22, 21, 19, 22, 19, 23};";

  #[test]
  fn binomial_estimates() {
    assert_true(&format!(
      "{BDATA} d = EstimatedDistribution[bdata, BinomialDistribution[n, p]]; \
       d[[1]] === 33 && Abs[d[[2]] - 0.583333] < 10^-6"
    ));
    assert_true(&format!(
      "{BDATA} d = EstimatedDistribution[bdata, BinomialDistribution[50, p]]; \
       d[[1]] === 50 && Abs[d[[2]] - 0.385] < 10^-12"
    ));
    assert_eq!(
      interpret(&format!(
        "{BDATA} EstimatedDistribution[bdata, BinomialDistribution[n, 2/5]]"
      ))
      .unwrap(),
      "BinomialDistribution[48, 2/5]"
    );
  }

  #[test]
  fn uniform_bounds_are_the_sample_extremes() {
    assert_eq!(
      interpret(
        "EstimatedDistribution[{1, 2, 3, 4, 5}, UniformDistribution[{min, max}]]"
      )
      .unwrap(),
      "UniformDistribution[{1., 5.}]"
    );
  }

  #[test]
  fn log_series_and_beta() {
    assert_close(
      "EstimatedDistribution[{16, 1, 2, 5, 3, 5, 1, 6, 4, 14, 2, 9, 1, 1, 2, \
       2, 8, 1, 3, 1}, LogSeriesDistribution[θ]][[1]]",
      0.914521,
    );
    assert_true(
      "d = EstimatedDistribution[{0.8, 0.98, 0.66, 0.09, 0.03, 0.16, 0.88, \
       0.32, 0.56, 0.3, 0.82, 0.34, 0.93, 0.33, 0.25, 0.74, 0.3, 0.74, 0.16, \
       0.02}, BetaDistribution[α, β]]; \
       Abs[d[[1]] - 0.832619] < 10^-6 && Abs[d[[2]] - 0.921159] < 10^-6",
    );
  }

  #[test]
  fn quantity_data_give_a_quantity_distribution() {
    assert_true(
      "d = EstimatedDistribution[Quantity[{2.35, 1.55, 2.7, 0.25, 1.5, 3.35, \
       1.85, 0.15, 2.3, 0.65, 0.85, 5.15}, \"Minutes\"], \
       ExponentialDistribution[λ]]; \
       Head[d] === QuantityDistribution && d[[2]] === \"Minutes\" && \
       Abs[d[[1, 1]] - 0.529801] < 10^-6",
    );
  }
}
