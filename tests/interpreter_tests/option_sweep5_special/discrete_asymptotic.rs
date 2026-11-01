// DiscreteAsymptotic examples from the reference page (round 5 of the
// reference-page sweep), plus regression tests for the leading-term rules.

use super::*;

mod discrete_asymptotic {
  use super::*;

  #[test]
  fn leading_polynomial() {
    assert_eq!(
      interpret("DiscreteAsymptotic[n^4 + 5 n^3 + 3 n - 11, n -> Infinity]")
        .unwrap(),
      "n^4"
    );
  }

  #[test]
  fn rational_constant() {
    assert_eq!(
      interpret("DiscreteAsymptotic[(3 n^2 + 5)/(4 n^2 + 11), n -> Infinity]")
        .unwrap(),
      "3/4"
    );
  }

  #[test]
  fn rational_growing() {
    assert_eq!(
      interpret("DiscreteAsymptotic[(3 n^3 + 5)/(4 n^2 + 11), n -> Infinity]")
        .unwrap(),
      "(3*n)/4"
    );
  }

  #[test]
  fn rational_decaying() {
    assert_eq!(
      interpret("DiscreteAsymptotic[(3 n + 5)/(4 n^2 + 11), n -> Infinity]")
        .unwrap(),
      "3/(4*n)"
    );
  }

  #[test]
  fn exponential_dominates_decaying() {
    assert_eq!(
      interpret("DiscreteAsymptotic[2^n + 3^(-n), n -> Infinity]").unwrap(),
      "2^n"
    );
  }

  #[test]
  fn polynomial_exponential() {
    assert_eq!(
      interpret("DiscreteAsymptotic[(n^2 + 3 n) 2^n, n -> Infinity]").unwrap(),
      "2^n*n^2"
    );
  }

  #[test]
  fn rational_exponential_constant() {
    assert_eq!(
      interpret("DiscreteAsymptotic[(n^2/(3 n^2 + 1)) 2^n, n -> Infinity]")
        .unwrap(),
      "2^n/3"
    );
  }

  #[test]
  fn rational_exponential_decaying() {
    assert_eq!(
      interpret("DiscreteAsymptotic[(n^2/(3 n^3 + 1)) 2^n, n -> Infinity]")
        .unwrap(),
      "2^n/(3*n)"
    );
  }

  #[test]
  fn rational_exponential_growing() {
    assert_eq!(
      interpret("DiscreteAsymptotic[(5 n^4/(3 n^3 + 1)) 2^n, n -> Infinity]")
        .unwrap(),
      "(5*2^n*n)/3"
    );
  }

  #[test]
  fn sinh() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Sinh[n], n -> Infinity]").unwrap(),
      "E^n/2"
    );
  }

  #[test]
  fn cosh_beats_polynomial() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Cosh[2 n] + n^5, n -> Infinity]").unwrap(),
      "E^(2*n)/2"
    );
  }

  #[test]
  fn tanh() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Tanh[n], n -> Infinity]").unwrap(),
      "1"
    );
  }

  #[test]
  fn arcsinh() {
    assert_eq!(
      interpret("DiscreteAsymptotic[ArcSinh[n], n -> Infinity]").unwrap(),
      "Log[n]"
    );
  }

  #[test]
  fn arccosh_scaled() {
    assert_eq!(
      interpret("DiscreteAsymptotic[ArcCosh[3 n], n -> Infinity]").unwrap(),
      "Log[n]"
    );
  }

  #[test]
  fn log_rational_decaying() {
    assert_eq!(
      interpret(
        "DiscreteAsymptotic[(5 n^2/(3 n^3 + 1)) Log[n], n -> Infinity]"
      )
      .unwrap(),
      "(5*Log[n])/(3*n)"
    );
  }

  #[test]
  fn log_rational_growing() {
    assert_eq!(
      interpret(
        "DiscreteAsymptotic[(5 n^4/(3 n^3 + 1)) Log[n], n -> Infinity]"
      )
      .unwrap(),
      "(5*n*Log[n])/3"
    );
  }

  #[test]
  fn log_factorial() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Log[n!], n -> Infinity]").unwrap(),
      "n*Log[n]"
    );
  }

  #[test]
  fn q_sequence_constant() {
    assert_eq!(
      interpret("DiscreteAsymptotic[2^n/(1 + 2^n), n -> Infinity]").unwrap(),
      "1"
    );
  }

  #[test]
  fn q_sequence_growing() {
    assert_eq!(
      interpret("DiscreteAsymptotic[4^n/(1 + 2^n), n -> Infinity]").unwrap(),
      "2^n"
    );
  }

  #[test]
  fn q_sequence_decaying() {
    assert_eq!(
      interpret("DiscreteAsymptotic[2^n/(1 + 4^n), n -> Infinity]").unwrap(),
      "2^(-n)"
    );
  }

  #[test]
  fn fibonacci() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Fibonacci[n], n -> Infinity]").unwrap(),
      "GoldenRatio^n/Sqrt[5]"
    );
  }

  #[test]
  fn pochhammer() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Pochhammer[a, n], n -> Infinity]").unwrap(),
      "(n^(-1/2 + a + n)*Sqrt[2*Pi])/(E^n*Gamma[a])"
    );
  }

  #[test]
  fn factorial_power() {
    assert_eq!(
      interpret("DiscreteAsymptotic[FactorialPower[a, n], n -> Infinity]")
        .unwrap(),
      "(n^(-1/2 - a + n)*Sqrt[2/Pi]*Gamma[1 + a]*Sin[(1 + a - n)*Pi])/E^n"
    );
  }

  #[test]
  fn factorial_power_fixed_order() {
    assert_eq!(
      interpret("DiscreteAsymptotic[FactorialPower[n, 3], n -> Infinity]")
        .unwrap(),
      "n^3"
    );
  }

  #[test]
  fn binomial_fixed_k() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Binomial[n, k], n -> Infinity]").unwrap(),
      "n^k/k!"
    );
  }

  #[test]
  fn binomial_shifted_fixed_k() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Binomial[n + 2, 3], n -> Infinity]")
        .unwrap(),
      "n^3/6"
    );
  }

  #[test]
  fn central_binomial() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Binomial[2 n, n], n -> Infinity]").unwrap(),
      "4^n/(Sqrt[n]*Sqrt[Pi])"
    );
  }

  #[test]
  fn polygamma() {
    assert_eq!(
      interpret("DiscreteAsymptotic[PolyGamma[n], n -> Infinity]").unwrap(),
      "Log[n]"
    );
  }

  #[test]
  fn zeta() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Zeta[n], n -> Infinity]").unwrap(),
      "1"
    );
  }

  #[test]
  fn prime() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Prime[n], n -> Infinity]").unwrap(),
      "n*Log[n]"
    );
  }

  #[test]
  fn bell() {
    assert_eq!(
      interpret("DiscreteAsymptotic[BellB[n], n -> Infinity]").unwrap(),
      "(E^(-1 - n + n/ProductLog[n])*(n/ProductLog[n])^(1/2 + n))/Sqrt[n]"
    );
  }

  #[test]
  fn bernoulli_even() {
    assert_eq!(
      interpret("DiscreteAsymptotic[BernoulliB[2 n], n -> Infinity]").unwrap(),
      "(4*(-1)^(-1 + n)*n^(1/2 + 2*n)*Pi^(1/2 - 2*n))/E^(2*n)"
    );
  }

  #[test]
  fn euler_even() {
    assert_eq!(
      interpret("DiscreteAsymptotic[EulerE[2 n], n -> Infinity]").unwrap(),
      "((-1)^n*2^(3 + 4*n)*n^(1/2 + 2*n)*Pi^(-1/2 - 2*n))/E^(2*n)"
    );
  }

  #[test]
  fn factorial_term_goal_one() {
    assert_eq!(
      interpret("DiscreteAsymptotic[n!, n -> Infinity, SeriesTermGoal -> 1]")
        .unwrap(),
      "(n^(1/2 + n)*Sqrt[2*Pi])/E^n"
    );
  }

  #[test]
  fn factorial_term_goal_three() {
    assert_eq!(
      interpret("DiscreteAsymptotic[n!, n -> Infinity, SeriesTermGoal -> 3]")
        .unwrap(),
      "(n^(1/2 + n)*Sqrt[2*Pi]*(1 + 1/(288*n^2) + 1/(12*n)))/E^n"
    );
  }

  #[test]
  fn factorial_list_spec() {
    assert_eq!(
      interpret("DiscreteAsymptotic[n!, {n, Infinity, 3}]").unwrap(),
      "(n^(1/2 + n)*Sqrt[2*Pi]*(1 + 1/(288*n^2) + 1/(12*n)))/E^n"
    );
  }

  #[test]
  fn gamma_list_spec() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Gamma[n], {n, Infinity, 3}]").unwrap(),
      "(n^(-1/2 + n)*Sqrt[2*Pi]*(1 + 1/(288*n^2) + 1/(12*n)))/E^n"
    );
  }

  #[test]
  fn generating_function_exponential_pole() {
    assert_eq!(
      interpret("DiscreteAsymptotic[SeriesCoefficient[1/(2 - E^z), {z, 0, n}], n -> Infinity]").unwrap(),
      "Log[2]^(-1 - n)/2"
    );
  }

  #[test]
  fn generating_function_rational() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Inactive[SeriesCoefficient][-(z/(-1 + z + z^2)), {z, 0, n}], n -> Infinity]").unwrap(),
      "(2/(-1 + Sqrt[5]))^n/Sqrt[5]"
    );
  }

  #[test]
  fn inactive_sum() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Inactive[Sum][Binomial[n, k]^2, {k, 0, n}], n -> Infinity]").unwrap(),
      "4^n/(Sqrt[n]*Sqrt[Pi])"
    );
  }

  #[test]
  fn cancelling_terms_use_series() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Sqrt[n^2 + 1] - n, n -> Infinity]")
        .unwrap(),
      "1/(2*n)"
    );
  }

  #[test]
  fn oscillating_unevaluated() {
    assert_eq!(
      interpret("DiscreteAsymptotic[Sin[n], n -> Infinity]").unwrap(),
      "DiscreteAsymptotic[Sin[n], n -> Infinity]"
    );
  }

  #[test]
  fn unknown_function_unevaluated() {
    assert_eq!(
      interpret("DiscreteAsymptotic[f[n], n -> Infinity]").unwrap(),
      "DiscreteAsymptotic[f[n], n -> Infinity]"
    );
  }

  #[test]
  fn invalid_term_goal_unevaluated() {
    assert_eq!(
      interpret("DiscreteAsymptotic[n!, n -> Infinity, SeriesTermGoal -> 0]")
        .unwrap(),
      "DiscreteAsymptotic[n!, n -> Infinity, SeriesTermGoal -> 0]"
    );
  }
}
