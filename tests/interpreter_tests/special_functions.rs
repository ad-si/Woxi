use super::*;

mod hypergeometric_0f1_regularized {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("Hypergeometric0F1Regularized[a, z]").unwrap(),
      "Hypergeometric0F1Regularized[a, z]"
    );
  }

  #[test]
  fn zero_at_a_zero_z_zero() {
    // 1/Gamma(0) = 0
    assert_eq!(
      interpret("Hypergeometric0F1Regularized[0, 0]").unwrap(),
      "0"
    );
  }

  #[test]
  fn one_at_positive_integer_z_zero() {
    assert_eq!(
      interpret("Hypergeometric0F1Regularized[1, 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Hypergeometric0F1Regularized[2, 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn numeric_a1() {
    // Hypergeometric0F1Regularized[1, 1.0] ≈ 2.279585302336067
    let result = interpret("Hypergeometric0F1Regularized[1, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.279585302336067).abs() < 1e-10);
  }

  #[test]
  fn numeric_a2() {
    // Hypergeometric0F1Regularized[2, 1.0] ≈ 1.5906368546373288
    let result = interpret("Hypergeometric0F1Regularized[2, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.5906368546373288).abs() < 1e-10);
  }

  #[test]
  fn numeric_a3() {
    // Hypergeometric0F1Regularized[3, 2.0] ≈ 0.9287588901146092
    let result = interpret("Hypergeometric0F1Regularized[3, 2.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.9287588901146092).abs() < 1e-10);
  }

  #[test]
  fn numeric_a_half() {
    // Hypergeometric0F1Regularized[0.5, 1.0] ≈ 2.122591620177637
    let result = interpret("Hypergeometric0F1Regularized[0.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.122591620177637).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_z() {
    // Hypergeometric0F1Regularized[3, -2.0] ≈ 0.2397640410755054
    let result = interpret("Hypergeometric0F1Regularized[3, -2.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.2397640410755054).abs() < 1e-10);
  }

  #[test]
  fn numeric_a_zero_z_nonzero() {
    // Hypergeometric0F1Regularized[0, 1.0] ≈ 1.5906368546373288
    let result = interpret("Hypergeometric0F1Regularized[0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.5906368546373288).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_z() {
    // Hypergeometric0F1Regularized[1, 5.0] ≈ 17.05777785336906
    let result = interpret("Hypergeometric0F1Regularized[1, 5.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 17.05777785336906).abs() < 1e-8);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Hypergeometric0F1Regularized]").unwrap(),
      "{Listable, NumericFunction, Protected}"
    );
  }
}

mod struve_h {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("StruveH[n, z]").unwrap(), "StruveH[n, z]");
  }

  #[test]
  fn zero_arg_integer_order() {
    assert_eq!(interpret("StruveH[0, 0]").unwrap(), "0");
    assert_eq!(interpret("StruveH[1, 0]").unwrap(), "0");
    assert_eq!(interpret("StruveH[2, 0]").unwrap(), "0");
  }

  #[test]
  fn numeric_order_0() {
    // StruveH[0, 1.0] ≈ 0.5686566270482879
    let result = interpret("StruveH[0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.5686566270482879).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_1() {
    // StruveH[1, 1.0] ≈ 0.1984573362019444
    let result = interpret("StruveH[1, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.1984573362019444).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_2() {
    // StruveH[2, 1.0] ≈ 0.040464636144794626
    let result = interpret("StruveH[2, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.040464636144794626).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_z() {
    // StruveH[0, 10.0] ≈ 0.11874368368750424
    let result = interpret("StruveH[0, 10.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.11874368368750424).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_z_order_1() {
    // StruveH[1, 10.0] ≈ 0.8918324920945468
    let result = interpret("StruveH[1, 10.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8918324920945468).abs() < 1e-10);
  }

  #[test]
  fn numeric_fractional_order() {
    // StruveH[0.5, 3.0] ≈ 0.9167076867564138
    let result = interpret("StruveH[0.5, 3.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.9167076867564138).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_fractional_order() {
    // StruveH[-0.5, 3.0] ≈ 0.06500818287737578
    let result = interpret("StruveH[-0.5, 3.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.06500818287737578).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_integer_order() {
    // StruveH[-1, 1.0] ≈ 0.43816243616563694
    let result = interpret("N[StruveH[-1, 1.0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.43816243616563694).abs() < 1e-10);
  }

  #[test]
  fn numeric_high_order() {
    // StruveH[5, 10.0] ≈ 7.644815648083951
    let result = interpret("StruveH[5, 10.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 7.644815648083951).abs() < 1e-8);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[StruveH]").unwrap(),
      "{Listable, NumericFunction, Protected, ReadProtected}"
    );
  }
}

mod struve_l {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("StruveL[n, z]").unwrap(), "StruveL[n, z]");
  }

  #[test]
  fn zero_arg_integer_order() {
    assert_eq!(interpret("StruveL[0, 0]").unwrap(), "0");
    assert_eq!(interpret("StruveL[1, 0]").unwrap(), "0");
    assert_eq!(interpret("StruveL[2, 0]").unwrap(), "0");
  }

  #[test]
  fn numeric_order_0() {
    // StruveL[0, 1.0] ≈ 0.7102431859378909
    let result = interpret("StruveL[0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.7102431859378909).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_1() {
    // StruveL[1, 1.0] ≈ 0.22676438105580865
    let result = interpret("StruveL[1, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.22676438105580865).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_2() {
    // StruveL[2, 1.0] ≈ 0.044507833037079836
    let result = interpret("StruveL[2, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.044507833037079836).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_z() {
    // StruveL[0, 2.5] ≈ 3.0112116937373057
    let result = interpret("StruveL[0, 2.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 3.0112116937373057).abs() < 1e-10);
  }

  #[test]
  fn numeric_half_order() {
    // StruveL[1/2, 1.0] ≈ 0.4333156537901021
    let result = interpret("StruveL[0.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.4333156537901021).abs() < 1e-10);
  }

  #[test]
  fn numeric_neg_half_order() {
    // StruveL[-1/2, 1.0] ≈ 0.9376748882454876
    let result = interpret("StruveL[-0.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.9376748882454876).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_z() {
    // StruveL[0, -1.0] ≈ -0.7102431859378909
    let result = interpret("StruveL[0, -1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-0.7102431859378909)).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_integer_order() {
    // StruveL[-1, 1.0] ≈ 0.86338415342339
    let result = interpret("StruveL[-1.0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.86338415342339).abs() < 1e-10);
  }

  #[test]
  fn numeric_small_z() {
    // StruveL[0, 0.5] ≈ 0.32724069939418077
    let result = interpret("StruveL[0, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.32724069939418077).abs() < 1e-10);
  }

  #[test]
  fn numeric_n_pi() {
    // N[StruveL[0, Pi]] ≈ 5.256595137877723
    let result = interpret("N[StruveL[0, Pi]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 5.256595137877723).abs() < 1e-8);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[StruveL]").unwrap(),
      "{Listable, NumericFunction, Protected, ReadProtected}"
    );
  }
}

mod generating_function {
  use super::*;

  #[test]
  fn constant_sequence() {
    assert_eq!(
      interpret("GeneratingFunction[1, n, x]").unwrap(),
      "(1 - x)^(-1)"
    );
  }

  #[test]
  fn constant_a() {
    assert_eq!(
      interpret("GeneratingFunction[a, n, x]").unwrap(),
      "a/(1 - x)"
    );
  }

  #[test]
  fn identity_sequence() {
    assert_eq!(
      interpret("GeneratingFunction[n, n, x]").unwrap(),
      "x/(-1 + x)^2"
    );
  }

  #[test]
  fn n_squared() {
    assert_eq!(
      interpret("GeneratingFunction[n^2, n, x]").unwrap(),
      "(-x - x^2)/(-1 + x)^3"
    );
  }

  #[test]
  fn n_cubed() {
    assert_eq!(
      interpret("GeneratingFunction[n^3, n, x]").unwrap(),
      "(x + 4*x^2 + x^3)/(-1 + x)^4"
    );
  }

  #[test]
  fn n_to_4() {
    assert_eq!(
      interpret("GeneratingFunction[n^4, n, x]").unwrap(),
      "(-x - 11*x^2 - 11*x^3 - x^4)/(-1 + x)^5"
    );
  }

  #[test]
  fn exponential_2n() {
    assert_eq!(
      interpret("GeneratingFunction[2^n, n, x]").unwrap(),
      "(1 - 2*x)^(-1)"
    );
  }

  #[test]
  fn alternating() {
    assert_eq!(
      interpret("GeneratingFunction[(-1)^n, n, x]").unwrap(),
      "(1 + x)^(-1)"
    );
  }

  #[test]
  fn exponential_a_n() {
    assert_eq!(
      interpret("GeneratingFunction[a^n, n, x]").unwrap(),
      "(1 - a*x)^(-1)"
    );
  }

  #[test]
  fn reciprocal_factorial() {
    assert_eq!(
      interpret("GeneratingFunction[1/Factorial[n], n, x]").unwrap(),
      "E^x"
    );
  }

  #[test]
  fn reciprocal_n_plus_1() {
    assert_eq!(
      interpret("GeneratingFunction[1/(n+1), n, x]").unwrap(),
      "-(Log[1 - x]/x)"
    );
  }

  #[test]
  fn reciprocal_factorial_squared() {
    // Regression: 1/n!^2 used to cause a stack overflow via infinite recursion
    // in gf_divide → gf_inner → extract_num_den → gf_divide.
    assert_eq!(
      interpret("GeneratingFunction[1/n!^2, n, x]").unwrap(),
      "BesselI[0, 2*Sqrt[x]]"
    );
    assert_eq!(
      interpret("GeneratingFunction[1/Factorial[n]^2, n, x]").unwrap(),
      "BesselI[0, 2*Sqrt[x]]"
    );
  }

  #[test]
  fn binomial_n_2() {
    // Binomial[n,2] evaluates to ((-1+n)*n)/2 which expands into GF terms
    assert_eq!(
      interpret("Simplify[GeneratingFunction[Binomial[n, 2], n, x]]").unwrap(),
      "-(x^2/(-1 + x)^3)"
    );
  }

  #[test]
  fn binomial_2n_n() {
    assert_eq!(
      interpret("GeneratingFunction[Binomial[2*n, n], n, x]").unwrap(),
      "1/Sqrt[1 - 4*x]"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("GeneratingFunction[f[n], n, x]").unwrap(),
      "GeneratingFunction[f[n], n, x]"
    );
  }

  #[test]
  fn shifted_sequence() {
    assert_eq!(
      interpret("GeneratingFunction[f[n + 1], n, x]").unwrap(),
      "(GeneratingFunction[f[n], n, x] - f[0])/x"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[GeneratingFunction]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }

  // The Fibonacci and Lucas sequences share the denominator 1 - x - x^2.
  #[test]
  fn fibonacci_sequence() {
    assert_eq!(
      interpret("GeneratingFunction[Fibonacci[n], n, x]").unwrap(),
      "-(x/(-1 + x + x^2))"
    );
    // The variable name is respected.
    assert_eq!(
      interpret("GeneratingFunction[Fibonacci[n], n, t]").unwrap(),
      "-(t/(-1 + t + t^2))"
    );
  }

  #[test]
  fn lucas_sequence() {
    assert_eq!(
      interpret("GeneratingFunction[LucasL[n], n, x]").unwrap(),
      "(-2 + x)/(-1 + x + x^2)"
    );
  }

  // A scalar multiple threads through and the closed form round-trips to the
  // sequence via a series expansion.
  #[test]
  fn fibonacci_scaled_and_series() {
    assert_eq!(
      interpret("GeneratingFunction[3*Fibonacci[n], n, x]").unwrap(),
      "(-3*x)/(-1 + x + x^2)"
    );
    assert_eq!(
      interpret("Series[GeneratingFunction[Fibonacci[n], n, x], {x, 0, 6}]")
        .unwrap(),
      "SeriesData[x, 0, {1, 1, 2, 3, 5, 8}, 1, 7, 1]"
    );
  }

  #[test]
  fn catalan_and_harmonic() {
    assert_eq!(
      interpret("GeneratingFunction[CatalanNumber[n], n, x]").unwrap(),
      "2/(1 + Sqrt[1 - 4*x])"
    );
    assert_eq!(
      interpret("GeneratingFunction[HarmonicNumber[n], n, x]").unwrap(),
      "Log[1 - x]/(-1 + x)"
    );
    // The Catalan closed form round-trips to 1, 1, 2, 5, 14, 42.
    assert_eq!(
      interpret(
        "Series[GeneratingFunction[CatalanNumber[n], n, x], {x, 0, 5}]"
      )
      .unwrap(),
      "SeriesData[x, 0, {1, 1, 2, 5, 14, 42}, 0, 6, 1]"
    );
  }

  // Sum[x^m/(m+k)] for a positive integer k, generalizing 1/(n+1).
  #[test]
  fn reciprocal_shifted_integer() {
    assert_eq!(
      interpret("GeneratingFunction[1/(n + 1), n, x]").unwrap(),
      "-(Log[1 - x]/x)"
    );
    assert_eq!(
      interpret("GeneratingFunction[1/(n + 2), n, x]").unwrap(),
      "(-1 - Log[1 - x]/x)/x"
    );
    assert_eq!(
      interpret("GeneratingFunction[1/(n + 3), n, x]").unwrap(),
      "(-1 - x/2 - Log[1 - x]/x)/x^2"
    );
  }

  // Regression: `1/(a n + b)` forms with `a >= 2` used to rewrite endlessly and
  // overflow the stack. They now stay unevaluated instead of crashing.
  #[test]
  fn linear_denominator_does_not_overflow() {
    assert_eq!(
      interpret("GeneratingFunction[1/(2 n + 1), n, x]").unwrap(),
      "GeneratingFunction[(1 + 2*n)^(-1), n, x]"
    );
    assert_eq!(
      interpret("GeneratingFunction[1/(2 n), n, x]").unwrap(),
      "GeneratingFunction[1/(2*n), n, x]"
    );
  }
}

mod airy_bi {
  use super::*;

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("AiryBi[x]").unwrap(), "AiryBi[x]");
  }

  #[test]
  fn numeric_zero() {
    let result = interpret("N[AiryBi[0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.6149266274460007).abs() < 1e-10);
  }

  #[test]
  fn numeric_positive() {
    let result = interpret("N[AiryBi[1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.2074235949528715).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result = interpret("N[AiryBi[-1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.10399738949694468).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_positive() {
    let result = interpret("N[AiryBi[10]]").unwrap();
    let val: f64 = result.replace("*^", "e").parse().unwrap();
    assert!((val - 4.556411535482249e8).abs() / 4.556411535482249e8 < 1e-6);
  }

  #[test]
  fn numeric_large_negative() {
    let result = interpret("N[AiryBi[-10]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-0.3146798296438386)).abs() < 1e-6);
  }

  #[test]
  fn direct_real_input() {
    let result = interpret("AiryBi[0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8542770431031554).abs() < 1e-10);
  }

  #[test]
  fn attributes() {
    // Wolfram versions disagree on whether AiryBi has ReadProtected, so
    // only assert that the essential attributes are present.
    let result = interpret("Attributes[AiryBi]").unwrap();
    for attr in ["Listable", "NumericFunction", "Protected"] {
      assert!(result.contains(attr), "missing {attr} in {result}");
    }
  }
}

mod parabolic_cylinder_d {
  use super::*;

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(
      interpret("ParabolicCylinderD[n, x]").unwrap(),
      "ParabolicCylinderD[n, x]"
    );
  }

  #[test]
  fn numeric_d0_0() {
    let result = interpret("N[ParabolicCylinderD[0, 0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
  }

  #[test]
  fn numeric_d1_0() {
    let result = interpret("N[ParabolicCylinderD[1, 0]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(val.abs() < 1e-10);
  }

  #[test]
  fn numeric_d2_1_5() {
    let result = interpret("ParabolicCylinderD[2, 1.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.7122285309136538).abs() < 1e-8);
  }

  #[test]
  fn numeric_negative_order() {
    let result = interpret("ParabolicCylinderD[-1, 2.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.15501307659733082).abs() < 1e-8);
  }

  #[test]
  fn numeric_half_integer_order() {
    let result = interpret("ParabolicCylinderD[0.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8422032440698396).abs() < 1e-10);
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ParabolicCylinderD]").unwrap(),
      "{Listable, NumericFunction, Protected, ReadProtected}"
    );
  }
}

mod elliptic_theta_prime {
  use super::*;

  #[test]
  fn theta1() {
    assert_eq!(
      interpret("EllipticThetaPrime[1, 0.5, 0.1]").unwrap(),
      "0.9846106693769313"
    );
  }

  #[test]
  fn theta2() {
    assert_eq!(
      interpret("EllipticThetaPrime[2, 0.5, 0.1]").unwrap(),
      "-0.5728609100292524"
    );
  }

  #[test]
  fn theta3() {
    assert_eq!(
      interpret("EllipticThetaPrime[3, 0.5, 0.1]").unwrap(),
      "-0.33731583355805805"
    );
  }

  #[test]
  fn theta4() {
    assert_eq!(
      interpret("EllipticThetaPrime[4, 0.5, 0.1]").unwrap(),
      "0.33586095767513946"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("EllipticThetaPrime[1, z, q]").unwrap(),
      "EllipticThetaPrime[1, z, q]"
    );
  }
}

mod spherical_bessel_j {
  use super::*;

  // Non-negative integer orders with a real argument use the elementary
  // Sin/Cos closed form (j_0 = Sin[z]/z, j_1 = Sin[z]/z^2 - Cos[z]/z, then
  // upward recurrence), matching wolframscript to machine precision.

  #[test]
  fn order0_matches_sinc() {
    assert_eq!(
      interpret("SphericalBesselJ[0, 1.]").unwrap(),
      "0.8414709848078965"
    );
  }

  #[test]
  fn order0_threads_over_list() {
    assert_eq!(
      interpret("SphericalBesselJ[0, {1., 2.}]").unwrap(),
      "{0.8414709848078965, 0.45464871341284085}"
    );
  }

  #[test]
  fn order1_closed_form() {
    assert_eq!(
      interpret("SphericalBesselJ[1, 1.]").unwrap(),
      "0.30116867893975674"
    );
  }

  #[test]
  fn order2_recurrence() {
    assert_eq!(
      interpret("SphericalBesselJ[2, 3.]").unwrap(),
      "0.2986374970757335"
    );
  }

  #[test]
  fn exact_special_values() {
    // z = 0 stays exact; non-numeric arguments stay symbolic.
    assert_eq!(interpret("SphericalBesselJ[0, 0]").unwrap(), "1");
    assert_eq!(interpret("SphericalBesselJ[2, 0]").unwrap(), "0");
    assert_eq!(
      interpret("SphericalBesselJ[1, x]").unwrap(),
      "SphericalBesselJ[1, x]"
    );
  }
}

// CoulombF[L, eta, z] / CoulombG[L, eta, z] reduce to spherical Bessel functions
// at eta == 0 (and to Sin/Cos at L == 0); nonzero eta stays unevaluated.
mod coulomb_wave {
  use super::*;

  #[test]
  fn coulomb_f_eta_zero() {
    assert_eq!(interpret("CoulombF[0, 0, z]").unwrap(), "Sin[z]");
    assert_eq!(
      interpret("CoulombF[1, 0, z]").unwrap(),
      "z*SphericalBesselJ[1, z]"
    );
    assert_eq!(
      interpret("CoulombF[2, 0, z]").unwrap(),
      "z*SphericalBesselJ[2, z]"
    );
    // Symbolic order and symbolic argument.
    assert_eq!(
      interpret("CoulombF[L, 0, z]").unwrap(),
      "z*SphericalBesselJ[L, z]"
    );
    assert_eq!(interpret("CoulombF[0, 0, x + y]").unwrap(), "Sin[x + y]");
  }

  #[test]
  fn coulomb_g_eta_zero() {
    assert_eq!(interpret("CoulombG[0, 0, z]").unwrap(), "Cos[z]");
    assert_eq!(
      interpret("CoulombG[1, 0, z]").unwrap(),
      "-(z*SphericalBesselY[1, z])"
    );
    assert_eq!(
      interpret("CoulombG[L, 0, z]").unwrap(),
      "-(z*SphericalBesselY[L, z])"
    );
  }

  // The Hankel-type Coulomb waves reduce to spherical Hankel functions, and to
  // E^(+-I z) at order 0.
  #[test]
  fn coulomb_h1_h2_eta_zero() {
    assert_eq!(interpret("CoulombH1[0, 0, z]").unwrap(), "E^(I*z)");
    assert_eq!(interpret("CoulombH2[0, 0, z]").unwrap(), "E^(-I*z)");
    assert_eq!(
      interpret("CoulombH1[1, 0, z]").unwrap(),
      "I*z*SphericalHankelH1[1, z]"
    );
    assert_eq!(
      interpret("CoulombH2[1, 0, z]").unwrap(),
      "-I*z*SphericalHankelH2[1, z]"
    );
    assert_eq!(
      interpret("CoulombH1[L, 0, z]").unwrap(),
      "I*z*SphericalHankelH1[L, z]"
    );
    assert_eq!(
      interpret("CoulombH1[0, 0, x + y]").unwrap(),
      "E^(I*(x + y))"
    );
  }

  // Nonzero eta (the genuine Coulomb regime) stays unevaluated, like
  // wolframscript — not the generic "not implemented" message.
  #[test]
  fn nonzero_eta_unevaluated() {
    assert_eq!(interpret("CoulombF[1, 1, z]").unwrap(), "CoulombF[1, 1, z]");
    assert_eq!(interpret("CoulombG[2, 3, z]").unwrap(), "CoulombG[2, 3, z]");
    assert_eq!(
      interpret("CoulombH1[1, 1, z]").unwrap(),
      "CoulombH1[1, 1, z]"
    );
    assert_eq!(
      interpret("CoulombH2[1, 2, z]").unwrap(),
      "CoulombH2[1, 2, z]"
    );
  }
}

mod bessel_j_zero {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("BesselJZero[0, 1]").unwrap(), "BesselJZero[0, 1]");
  }

  #[test]
  fn numeric_j0_1() {
    assert_eq!(
      interpret("N[BesselJZero[0, 1]]").unwrap(),
      "2.404825557695773"
    );
  }

  #[test]
  fn numeric_j0_2() {
    assert_eq!(
      interpret("N[BesselJZero[0, 2]]").unwrap(),
      "5.520078110286309"
    );
  }

  #[test]
  fn numeric_j0_3() {
    assert_eq!(
      interpret("N[BesselJZero[0, 3]]").unwrap(),
      "8.653727912911014"
    );
  }

  #[test]
  fn numeric_j1_1() {
    assert_eq!(
      interpret("N[BesselJZero[1, 1]]").unwrap(),
      "3.8317059702075125"
    );
  }

  #[test]
  fn is_actual_zero() {
    // BesselJ at the zero should be approximately 0
    assert_eq!(
      interpret("Chop[BesselJ[0, N[BesselJZero[0, 1]]]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn bessel_j_at_matching_zero_is_zero() {
    // BesselJ[n, BesselJZero[n, k]] = 0 exactly, regardless of n and k.
    assert_eq!(interpret("BesselJ[0, BesselJZero[0, 1]]").unwrap(), "0");
    assert_eq!(interpret("BesselJ[3, BesselJZero[3, 5]]").unwrap(), "0");
  }

  #[test]
  fn bessel_j_at_symbolic_matching_zero_stays_symbolic() {
    // Wolfram only simplifies BesselJ[n, BesselJZero[n, k]] when BOTH n and k
    // are concrete positive integers. For symbolic args, it stays unevaluated.
    assert_eq!(
      interpret("BesselJ[n, BesselJZero[n, k]]").unwrap(),
      "BesselJ[n, BesselJZero[n, k]]"
    );
  }

  #[test]
  fn bessel_j_at_three_arg_zero_is_zero() {
    // BesselJZero[n, k, x0] is a different zero of J_n; identity still holds.
    assert_eq!(interpret("BesselJ[2, BesselJZero[2, 1, 4]]").unwrap(), "0");
  }

  #[test]
  fn bessel_j_at_mismatched_zero_stays_symbolic() {
    // J_1 at a zero of J_0 is not generally zero.
    let result = interpret("BesselJ[1, BesselJZero[0, 1]]").unwrap();
    assert!(
      result.contains("BesselJ"),
      "expected unevaluated, got {}",
      result
    );
  }
}

mod bessel_y_zero {
  use super::*;

  // BesselYZero stays symbolic when both arguments are exact (matching
  // BesselJZero's pattern) so users can still reason about it as a
  // symbol; numeric evaluation kicks in when either argument is a Real.
  #[test]
  fn symbolic() {
    assert_eq!(interpret("BesselYZero[0, 1]").unwrap(), "BesselYZero[0, 1]");
  }

  // The first positive zero of Y_0 is `0.89357696627916752…`. f64
  // resolves the value to about 16 digits; the implementation uses a
  // bisection scan plus Newton polish identical to `BesselJZero`'s.
  #[test]
  fn numeric_y0_1() {
    let result = interpret("N[BesselYZero[0, 1]]").unwrap();
    let val: f64 = result.parse().expect("should parse to a Real");
    assert!(
      (val - 0.8935769662791675).abs() < 1e-13,
      "expected ~0.89357696627916752, got {val}"
    );
  }

  #[test]
  fn numeric_y0_2() {
    let result = interpret("N[BesselYZero[0, 2]]").unwrap();
    let val: f64 = result.parse().expect("should parse to a Real");
    assert!(
      (val - 3.957678419314858).abs() < 1e-13,
      "expected ~3.95767841931486, got {val}"
    );
  }

  // Round-trip: the computed zero should drive `BesselY[0, x]` to
  // approximately 0 at machine precision.
  #[test]
  fn is_actual_zero() {
    assert_eq!(
      interpret("Chop[BesselY[0, N[BesselYZero[0, 1]]]]").unwrap(),
      "0"
    );
  }

  #[test]
  fn bessel_y_at_matching_zero_is_zero() {
    // BesselY[n, BesselYZero[n, k]] = 0 by definition.
    assert_eq!(interpret("BesselY[0, BesselYZero[0, 1]]").unwrap(), "0");
    assert_eq!(interpret("BesselY[2, BesselYZero[2, 4]]").unwrap(), "0");
  }

  #[test]
  fn bessel_y_at_mismatched_zero_stays_symbolic() {
    let result = interpret("BesselY[1, BesselYZero[0, 1]]").unwrap();
    assert!(
      result.contains("BesselY"),
      "expected unevaluated, got {}",
      result
    );
  }
}

mod jacobi_zeta {
  use super::*;

  #[test]
  fn zero_phi() {
    assert_eq!(interpret("JacobiZeta[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn zero_m() {
    assert_eq!(interpret("JacobiZeta[0.5, 0]").unwrap(), "0.");
  }

  #[test]
  fn numeric() {
    let result = interpret("JacobiZeta[0.5, 0.3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.06715126391766499).abs() < 1e-10);
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("JacobiZeta[x, m]").unwrap(), "JacobiZeta[x, m]");
  }
}

mod elliptic_exp {
  use super::*;

  fn parse_pair(s: &str) -> (f64, f64) {
    let inner = s
      .trim()
      .trim_start_matches('{')
      .trim_end_matches('}')
      .trim();
    let mut parts = inner.split(',');
    let x: f64 = parts.next().unwrap().trim().parse().unwrap();
    let y: f64 = parts.next().unwrap().trim().parse().unwrap();
    (x, y)
  }

  fn assert_close(got: f64, expected: f64, msg: &str) {
    let tol = 1e-9 * expected.abs().max(1.0);
    assert!(
      (got - expected).abs() < tol,
      "{msg}: got {got}, expected {expected}"
    );
  }

  #[test]
  fn negative_u_audit_case() {
    // Audit case: EllipticExp[-0.4, {4, 1}] → {5.043827135411493, 15.333640384130709}
    let result = interpret("EllipticExp[-0.4, {4, 1}]").unwrap();
    let (x, y) = parse_pair(&result);
    assert_close(x, 5.043827135411493, "x");
    assert_close(y, 15.333640384130709, "y");
  }

  #[test]
  fn small_negative_u_large_x() {
    let result = interpret("EllipticExp[-0.1, {4, 1}]").unwrap();
    let (x, y) = parse_pair(&result);
    assert_close(x, 98.67528490530962, "x");
    assert_close(y, 999.9142994129613, "y");
  }

  #[test]
  fn larger_negative_u_small_x() {
    let result = interpret("EllipticExp[-1.0, {4, 1}]").unwrap();
    let (x, y) = parse_pair(&result);
    assert_close(x, 0.21791718623082618, "x");
    assert_close(y, 0.6466971594261198, "y");
  }

  #[test]
  fn positive_u_flips_y_sign() {
    // EllipticExp[u, {a, b}] == {x(|u|), -sign(u) * y(|u|)}
    let result = interpret("EllipticExp[0.4, {4, 1}]").unwrap();
    let (x, y) = parse_pair(&result);
    assert_close(x, 5.043827135411493, "x");
    assert_close(y, -15.333640384130709, "y");
  }

  #[test]
  fn different_curve_parameters() {
    let result = interpret("EllipticExp[-0.3, {2, 1}]").unwrap();
    let (x, y) = parse_pair(&result);
    assert_close(x, 10.450531251495653, "x");
    assert_close(y, 37.01645463778015, "y");
  }

  #[test]
  fn b_greater_than_a() {
    let result = interpret("EllipticExp[-0.5, {1, 2}]").unwrap();
    let (x, y) = parse_pair(&result);
    assert_close(x, 3.58917180413188, "x");
    assert_close(y, 8.142282396294094, "y");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(
      interpret("EllipticExp[u, {a, b}]").unwrap(),
      "EllipticExp[u, {a, b}]"
    );
  }

  // Exact (integer/rational) arguments stay symbolic — only an inexact
  // argument or N[...] triggers numeric evaluation.
  #[test]
  fn exact_args_stay_symbolic() {
    assert_eq!(
      interpret("EllipticExp[1, {1, 1}]").unwrap(),
      "EllipticExp[1, {1, 1}]"
    );
    // An inexact invariant still numericizes.
    let result = interpret("EllipticExp[1, {1.0, 1}]").unwrap();
    let (x, _) = parse_pair(&result);
    assert_close(x, 0.5749706105873643, "x");
  }

  // The pole at u = 0 is structural and reported even for exact arguments.
  #[test]
  fn pole_at_zero_exact() {
    assert_eq!(
      interpret("EllipticExp[0, {1, 1}]").unwrap(),
      "{ComplexInfinity, ComplexInfinity}"
    );
  }
}

mod elliptic_e_incomplete {
  use super::*;

  #[test]
  fn two_arg_numeric() {
    let result = interpret("EllipticE[0.5, 0.3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.49399114472896827).abs() < 1e-10);
  }

  #[test]
  fn two_arg_zero_phi() {
    assert_eq!(interpret("EllipticE[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("EllipticE[x, m]").unwrap(), "EllipticE[x, m]");
  }
}

mod q_binomial {
  use super::*;

  #[test]
  fn numeric_q() {
    assert_eq!(interpret("QBinomial[4, 2, 2]").unwrap(), "35");
  }

  #[test]
  fn rational_q() {
    assert_eq!(interpret("QBinomial[5, 2, 1/2]").unwrap(), "155/64");
  }

  #[test]
  fn k_zero() {
    assert_eq!(interpret("QBinomial[4, 0, q]").unwrap(), "1");
  }

  #[test]
  fn k_equals_n() {
    assert_eq!(interpret("QBinomial[4, 4, q]").unwrap(), "1");
  }

  #[test]
  fn symbolic_q() {
    assert_eq!(
      interpret("QBinomial[3, 1, q]").unwrap(),
      "QBinomial[3, 1, q]"
    );
  }

  #[test]
  fn q_one_half_small() {
    assert_eq!(interpret("QBinomial[3, 1, 2]").unwrap(), "7");
  }
}

mod piecewise_expand {
  use super::*;

  #[test]
  fn min_two_args() {
    assert_eq!(
      interpret("PiecewiseExpand[Min[x, y]]").unwrap(),
      "Piecewise[{{x, x - y <= 0}}, y]"
    );
  }

  #[test]
  fn max_two_args() {
    assert_eq!(
      interpret("PiecewiseExpand[Max[x, y]]").unwrap(),
      "Piecewise[{{x, x - y >= 0}}, y]"
    );
  }

  #[test]
  fn unit_step() {
    assert_eq!(
      interpret("PiecewiseExpand[UnitStep[x]]").unwrap(),
      "Piecewise[{{1, x >= 0}}, 0]"
    );
  }

  #[test]
  fn clip_default() {
    assert_eq!(
      interpret("PiecewiseExpand[Clip[x]]").unwrap(),
      "Piecewise[{{-1, x < -1}, {1, x > 1}}, x]"
    );
  }

  #[test]
  fn clip_custom_bounds() {
    assert_eq!(
      interpret("PiecewiseExpand[Clip[x, {0, 10}]]").unwrap(),
      "Piecewise[{{0, x < 0}, {10, x > 10}}, x]"
    );
  }

  #[test]
  fn unsupported_returns_unchanged() {
    // Non-expandable functions pass through unchanged
    assert_eq!(interpret("PiecewiseExpand[Sin[x]]").unwrap(), "Sin[x]");
  }
}

mod heaviside_pi {
  use super::*;

  #[test]
  fn zero_is_one() {
    assert_eq!(interpret("HeavisidePi[0]").unwrap(), "1");
  }

  #[test]
  fn inside_is_one() {
    assert_eq!(interpret("HeavisidePi[1/4]").unwrap(), "1");
  }

  #[test]
  fn outside_is_zero() {
    assert_eq!(interpret("HeavisidePi[-1]").unwrap(), "0");
  }

  #[test]
  fn at_boundary_unevaluated() {
    assert_eq!(interpret("HeavisidePi[1/2]").unwrap(), "HeavisidePi[1/2]");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("HeavisidePi[x]").unwrap(), "HeavisidePi[x]");
  }
}

mod heaviside_lambda {
  use super::*;

  #[test]
  fn zero_is_one() {
    assert_eq!(interpret("HeavisideLambda[0]").unwrap(), "1");
  }

  #[test]
  fn inside_is_value() {
    assert_eq!(interpret("HeavisideLambda[1/3]").unwrap(), "2/3");
  }

  #[test]
  fn at_boundary_is_zero() {
    assert_eq!(interpret("HeavisideLambda[1]").unwrap(), "0");
  }

  #[test]
  fn outside_real_is_zero() {
    assert_eq!(interpret("HeavisideLambda[1.5]").unwrap(), "0.");
  }

  #[test]
  fn negative_inside() {
    assert_eq!(interpret("HeavisideLambda[-0.5]").unwrap(), "0.5");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("HeavisideLambda[x]").unwrap(),
      "HeavisideLambda[x]"
    );
  }
}

mod prime_omega {
  use super::*;

  #[test]
  fn one_has_zero() {
    assert_eq!(interpret("PrimeOmega[1]").unwrap(), "0");
  }

  #[test]
  fn prime_has_one() {
    assert_eq!(interpret("PrimeOmega[7]").unwrap(), "1");
  }

  #[test]
  fn composite_with_multiplicity() {
    // 12 = 2^2 * 3, so PrimeOmega = 2 + 1 = 3
    assert_eq!(interpret("PrimeOmega[12]").unwrap(), "3");
  }

  #[test]
  fn larger_number() {
    // 100 = 2^2 * 5^2, so PrimeOmega = 2 + 2 = 4
    assert_eq!(interpret("PrimeOmega[100]").unwrap(), "4");
  }

  // PrimeOmega counts factors of |n|, so it is defined on negatives.
  #[test]
  fn negative_uses_absolute_value() {
    assert_eq!(interpret("PrimeOmega[-12]").unwrap(), "3");
    assert_eq!(interpret("PrimeOmega[-1]").unwrap(), "0");
  }

  // 0 has no prime factorization; wolframscript leaves it unevaluated.
  #[test]
  fn zero_unevaluated() {
    assert_eq!(interpret("PrimeOmega[0]").unwrap(), "PrimeOmega[0]");
  }

  // PrimeOmega is Listable.
  #[test]
  fn threads_over_list() {
    assert_eq!(interpret("PrimeOmega[{12, 30}]").unwrap(), "{3, 3}");
    assert_eq!(
      interpret("PrimeOmega[{2, 4, 8, 16}]").unwrap(),
      "{1, 2, 3, 4}"
    );
  }
}

mod prime_nu {
  use super::*;

  #[test]
  fn one_has_zero() {
    assert_eq!(interpret("PrimeNu[1]").unwrap(), "0");
  }

  #[test]
  fn prime_has_one() {
    assert_eq!(interpret("PrimeNu[7]").unwrap(), "1");
  }

  #[test]
  fn composite_distinct_factors() {
    // 12 = 2^2 * 3, so PrimeNu = 2 (distinct primes: 2, 3)
    assert_eq!(interpret("PrimeNu[12]").unwrap(), "2");
  }

  #[test]
  fn three_distinct_factors() {
    // 60 = 2^2 * 3 * 5, so PrimeNu = 3
    assert_eq!(interpret("PrimeNu[60]").unwrap(), "3");
  }

  // PrimeNu counts distinct primes of |n|, so it is defined on negatives.
  #[test]
  fn negative_uses_absolute_value() {
    assert_eq!(interpret("PrimeNu[-12]").unwrap(), "2");
    assert_eq!(interpret("PrimeNu[-1]").unwrap(), "0");
  }

  // 0 has no prime factorization; wolframscript leaves it unevaluated.
  #[test]
  fn zero_unevaluated() {
    assert_eq!(interpret("PrimeNu[0]").unwrap(), "PrimeNu[0]");
  }

  // PrimeNu is Listable.
  #[test]
  fn threads_over_list() {
    assert_eq!(interpret("PrimeNu[{12, 30}]").unwrap(), "{2, 3}");
  }
}

mod polynomial_quotient_remainder {
  use super::*;

  #[test]
  fn basic_division() {
    assert_eq!(
      interpret("PolynomialQuotientRemainder[x^3 + 2x + 1, x + 1, x]").unwrap(),
      "{3 - x + x^2, -2}"
    );
  }

  #[test]
  fn exact_division() {
    assert_eq!(
      interpret("PolynomialQuotientRemainder[x^2 - 1, x - 1, x]").unwrap(),
      "{1 + x, 0}"
    );
  }

  #[test]
  fn linear_by_linear() {
    assert_eq!(
      interpret("PolynomialQuotientRemainder[2x + 3, x + 1, x]").unwrap(),
      "{2, 1}"
    );
  }
}

mod square_wave {
  use super::*;

  #[test]
  fn positive_first_half() {
    assert_eq!(interpret("SquareWave[0.1]").unwrap(), "1");
  }

  #[test]
  fn positive_at_zero() {
    assert_eq!(interpret("SquareWave[0]").unwrap(), "1");
  }

  #[test]
  fn negative_second_half() {
    assert_eq!(interpret("SquareWave[0.5]").unwrap(), "-1");
  }

  #[test]
  fn negative_at_0_7() {
    assert_eq!(interpret("SquareWave[0.7]").unwrap(), "-1");
  }

  #[test]
  fn periodic_wrapping() {
    assert_eq!(interpret("SquareWave[1.3]").unwrap(), "1");
  }

  #[test]
  fn negative_argument() {
    assert_eq!(interpret("SquareWave[-0.3]").unwrap(), "-1");
  }

  #[test]
  fn exact_rational_input() {
    assert_eq!(interpret("SquareWave[1/4]").unwrap(), "1");
    assert_eq!(interpret("SquareWave[1/2]").unwrap(), "-1");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("SquareWave[x]").unwrap(), "SquareWave[x]");
  }

  #[test]
  fn multi_level_two_values() {
    assert_eq!(interpret("SquareWave[{1/3, 2/3}, 0.2]").unwrap(), "2/3");
    assert_eq!(interpret("SquareWave[{1/3, 2/3}, 0.5]").unwrap(), "1/3");
    assert_eq!(interpret("SquareWave[{1/3, 2/3}, 0.8]").unwrap(), "1/3");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[SquareWave]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod triangle_wave {
  use super::*;

  #[test]
  fn integer_zero() {
    assert_eq!(interpret("TriangleWave[0]").unwrap(), "0");
  }

  #[test]
  fn integer_one() {
    assert_eq!(interpret("TriangleWave[1]").unwrap(), "0");
  }

  #[test]
  fn integer_negative() {
    assert_eq!(interpret("TriangleWave[-1]").unwrap(), "0");
  }

  #[test]
  fn integer_large() {
    assert_eq!(interpret("TriangleWave[100]").unwrap(), "0");
  }

  #[test]
  fn float_quarter() {
    assert_eq!(interpret("TriangleWave[0.25]").unwrap(), "1.");
  }

  #[test]
  fn float_half() {
    assert_eq!(interpret("TriangleWave[0.5]").unwrap(), "0.");
  }

  #[test]
  fn float_three_quarter() {
    assert_eq!(interpret("TriangleWave[0.75]").unwrap(), "-1.");
  }

  #[test]
  fn float_negative() {
    assert_eq!(interpret("TriangleWave[-0.25]").unwrap(), "-1.");
  }

  #[test]
  fn rational_quarter() {
    assert_eq!(interpret("TriangleWave[1/4]").unwrap(), "1");
  }

  #[test]
  fn rational_third() {
    assert_eq!(interpret("TriangleWave[1/3]").unwrap(), "2/3");
  }

  #[test]
  fn rational_three_eighths() {
    assert_eq!(interpret("TriangleWave[3/8]").unwrap(), "1/2");
  }

  #[test]
  fn periodic_wrapping() {
    assert_eq!(interpret("TriangleWave[2]").unwrap(), "0");
  }

  #[test]
  fn two_arg_scaling() {
    assert_eq!(interpret("TriangleWave[{2, 5}, 0.25]").unwrap(), "5.");
  }

  #[test]
  fn two_arg_unit() {
    assert_eq!(interpret("TriangleWave[{-1, 1}, 0.25]").unwrap(), "1.");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("TriangleWave[x]").unwrap(), "TriangleWave[x]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[TriangleWave]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod sawtooth_wave {
  use super::*;

  #[test]
  fn integer_zero() {
    assert_eq!(interpret("SawtoothWave[0]").unwrap(), "0");
  }

  #[test]
  fn integer_one() {
    assert_eq!(interpret("SawtoothWave[1]").unwrap(), "0");
  }

  #[test]
  fn integer_negative() {
    assert_eq!(interpret("SawtoothWave[-3]").unwrap(), "0");
  }

  #[test]
  fn rational_quarter() {
    assert_eq!(interpret("SawtoothWave[1/4]").unwrap(), "1/4");
  }

  #[test]
  fn rational_third() {
    assert_eq!(interpret("SawtoothWave[1/3]").unwrap(), "1/3");
  }

  #[test]
  fn rational_three_quarters() {
    assert_eq!(interpret("SawtoothWave[3/4]").unwrap(), "3/4");
  }

  #[test]
  fn float_quarter() {
    assert_eq!(interpret("SawtoothWave[0.25]").unwrap(), "0.25");
  }

  #[test]
  fn float_half() {
    assert_eq!(interpret("SawtoothWave[0.5]").unwrap(), "0.5");
  }

  #[test]
  fn float_periodic() {
    assert_eq!(interpret("SawtoothWave[1.5]").unwrap(), "0.5");
  }

  #[test]
  fn float_negative() {
    assert_eq!(interpret("SawtoothWave[-0.3]").unwrap(), "0.7");
  }

  #[test]
  fn two_arg_scaled() {
    assert_eq!(interpret("SawtoothWave[{0, 10}, 0.3]").unwrap(), "3.");
  }

  #[test]
  fn two_arg_negative_range() {
    assert_eq!(interpret("SawtoothWave[{-1, 1}, 0.25]").unwrap(), "-0.5");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("SawtoothWave[x]").unwrap(), "SawtoothWave[x]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[SawtoothWave]").unwrap(),
      "{Protected, ReadProtected}"
    );
  }
}

mod power_range {
  use super::*;

  #[test]
  fn default_factor_10() {
    assert_eq!(
      interpret("PowerRange[1, 1000, 10]").unwrap(),
      "{1, 10, 100, 1000}"
    );
  }

  #[test]
  fn factor_2() {
    assert_eq!(
      interpret("PowerRange[2, 32, 2]").unwrap(),
      "{2, 4, 8, 16, 32}"
    );
  }

  #[test]
  fn two_arg_default_factor() {
    assert_eq!(interpret("PowerRange[1, 100]").unwrap(), "{1, 10, 100}");
  }

  #[test]
  fn fractional_factor() {
    assert_eq!(
      interpret("PowerRange[1, 1/27, 1/3]").unwrap(),
      "{1, 1/3, 1/9, 1/27}"
    );
  }
}

mod monomial_list {
  use super::*;

  #[test]
  fn basic_two_variables() {
    assert_eq!(
      interpret("MonomialList[x^2 + 3*x*y + y^3, {x, y}]").unwrap(),
      "{x^2, 3*x*y, y^3}"
    );
  }

  #[test]
  fn three_terms_two_variables() {
    assert_eq!(
      interpret("MonomialList[x^3 + 2*x^2*y + y^2, {x, y}]").unwrap(),
      "{x^3, 2*x^2*y, y^2}"
    );
  }

  #[test]
  fn three_variables() {
    assert_eq!(
      interpret("MonomialList[a + b + c, {a, b, c}]").unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn constant_polynomial() {
    assert_eq!(interpret("MonomialList[5, {x}]").unwrap(), "{5}");
  }

  #[test]
  fn expansion_needed() {
    assert_eq!(
      interpret("MonomialList[(x + y)^3, {x, y}]").unwrap(),
      "{x^3, 3*x^2*y, 3*x*y^2, y^3}"
    );
  }

  #[test]
  fn single_variable() {
    assert_eq!(interpret("MonomialList[x, {x}]").unwrap(), "{x}");
  }

  #[test]
  fn single_variable_polynomial() {
    assert_eq!(
      interpret("MonomialList[x^3 + 2*x + 1, {x}]").unwrap(),
      "{x^3, 2*x, 1}"
    );
  }

  // One-argument form auto-detects the variables (= Variables[poly]).
  // Verified against wolframscript.
  #[test]
  fn one_argument_auto_variables() {
    assert_eq!(
      interpret("MonomialList[x^2 + 2 x y + y^2]").unwrap(),
      "{x^2, 2*x*y, y^2}"
    );
    assert_eq!(
      interpret("MonomialList[1 + x + x^2]").unwrap(),
      "{x^2, x, 1}"
    );
    assert_eq!(
      interpret("MonomialList[3 x y + 2 x^2]").unwrap(),
      "{2*x^2, 3*x*y}"
    );
    assert_eq!(interpret("MonomialList[a x + b y]").unwrap(), "{a*x, b*y}");
    assert_eq!(interpret("MonomialList[5]").unwrap(), "{5}");
    assert_eq!(
      interpret("MonomialList[(x + y)^2]").unwrap(),
      "{x^2, 2*x*y, y^2}"
    );
  }
}

mod discriminant {
  use super::*;

  #[test]
  fn quadratic_monic() {
    assert_eq!(
      interpret("Discriminant[x^2 + b*x + c, x]").unwrap(),
      "b^2 - 4*c"
    );
  }

  #[test]
  fn quadratic_general() {
    assert_eq!(
      interpret("Discriminant[a*x^2 + b*x + c, x]").unwrap(),
      "b^2 - 4*a*c"
    );
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("Discriminant[x^3 + p*x + q, x]").unwrap(),
      "-4*p^3 - 27*q^2"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(interpret("Discriminant[x^2 - 4, x]").unwrap(), "16");
  }

  #[test]
  fn degree_zero_constant() {
    // Regression (diff-fuzz): the resultant formula does not apply when the
    // polynomial has degree 0 (p' = 0). The root-product form leaves
    // a_0^(2n-2) = a_0^(-2); the zero polynomial is special-cased to 0.
    assert_eq!(interpret("Discriminant[3, x]").unwrap(), "1/9");
    assert_eq!(interpret("Discriminant[1/2, x]").unwrap(), "4");
    assert_eq!(interpret("Discriminant[-4, x]").unwrap(), "1/16");
    assert_eq!(interpret("Discriminant[2.0, x]").unwrap(), "0.25");
    assert_eq!(interpret("Discriminant[a, x]").unwrap(), "a^(-2)");
    assert_eq!(interpret("Discriminant[0, x]").unwrap(), "0");
    // Reduces to degree 0 after the zero-coefficient term cancels.
    assert_eq!(interpret("Discriminant[3 + 0 x, x]").unwrap(), "1/9");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[Discriminant]").unwrap(),
      "{Listable, Protected}"
    );
  }
}

mod factorial_power {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("FactorialPower[10, 3]").unwrap(), "720");
  }

  #[test]
  fn zero_power() {
    assert_eq!(interpret("FactorialPower[5, 0]").unwrap(), "1");
  }

  #[test]
  fn one_power() {
    assert_eq!(interpret("FactorialPower[5, 1]").unwrap(), "5");
  }

  #[test]
  fn with_step() {
    assert_eq!(interpret("FactorialPower[10, 3, 2]").unwrap(), "480");
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("FactorialPower[n, 3]").unwrap(),
      "FactorialPower[n, 3]"
    );
  }

  #[test]
  fn negative_order() {
    // FactorialPower[n, -k] = 1/((n+1)(n+2)...(n+k))
    assert_eq!(interpret("FactorialPower[3, -1]").unwrap(), "1/4");
    assert_eq!(interpret("FactorialPower[3, -2]").unwrap(), "1/20");
  }

  #[test]
  fn negative_order_with_step() {
    // FactorialPower[n, -k, h] = 1/((n+h)(n+2h)...(n+kh))
    assert_eq!(interpret("FactorialPower[3, -2, 2]").unwrap(), "1/35");
  }

  #[test]
  fn negative_order_pole() {
    // Zero in the denominator product
    assert_eq!(
      interpret("FactorialPower[-1, -1]").unwrap(),
      "ComplexInfinity"
    );
  }

  #[test]
  fn rational_base() {
    assert_eq!(interpret("FactorialPower[5/2, 2]").unwrap(), "15/4");
  }

  #[test]
  fn real_base() {
    assert_eq!(interpret("FactorialPower[2.5, 2]").unwrap(), "3.75");
    assert_eq!(
      interpret("FactorialPower[2.5, -2]").unwrap(),
      "0.06349206349206349"
    );
  }

  #[test]
  fn real_step() {
    assert_eq!(interpret("FactorialPower[2.5, 2, 0.5]").unwrap(), "5.");
  }

  #[test]
  fn series_at_zero_order_5() {
    // FactorialPower[x, 5] = x*(x-1)*(x-2)*(x-3)*(x-4)
    //                     = 24 x - 50 x^2 + 35 x^3 - 10 x^4 + x^5.
    // Series coefficients from x^1 to x^5 are the signed Stirling numbers
    // of the first kind: {24, -50, 35, -10, 1}.
    assert_eq!(
      interpret("Series[FactorialPower[x, 5], {x, 0, 5}]").unwrap(),
      "SeriesData[x, 0, {24, -50, 35, -10, 1}, 1, 6, 1]"
    );
  }

  #[test]
  fn series_at_zero_lower_order() {
    // Truncated to order 3, only the first three coefficients survive.
    assert_eq!(
      interpret("Series[FactorialPower[x, 5], {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {24, -50, 35}, 1, 4, 1]"
    );
  }

  #[test]
  fn series_at_zero_order_3_for_n_3() {
    // FactorialPower[x, 3] = x*(x-1)*(x-2) = 2 x - 3 x^2 + x^3.
    assert_eq!(
      interpret("Series[FactorialPower[x, 3], {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {2, -3, 1}, 1, 4, 1]"
    );
  }
}

mod recurrence_filter {
  use super::*;

  #[test]
  fn iir_filter() {
    assert_eq!(
      interpret("RecurrenceFilter[{{1, -1/2}, {1}}, {1, 2, 3, 4, 5}]").unwrap(),
      "{1, 5/2, 17/4, 49/8, 129/16}"
    );
  }

  #[test]
  fn cumulative_sum() {
    assert_eq!(
      interpret("RecurrenceFilter[{{1, -1}, {1}}, {1, 2, 3, 4, 5}]").unwrap(),
      "{1, 3, 6, 10, 15}"
    );
  }

  #[test]
  fn fir_filter() {
    assert_eq!(
      interpret("RecurrenceFilter[{{1}, {1, 1}}, {1, 0, 0, 0, 0}]").unwrap(),
      "{1, 1, 0, 0, 0}"
    );
  }
}

mod function_interpolation {
  use super::*;

  #[test]
  fn returns_interpolating_function() {
    assert_eq!(
      interpret("Head[FunctionInterpolation[Sin[x], {x, 0, 6.28}]]").unwrap(),
      "InterpolatingFunction"
    );
  }

  #[test]
  fn evaluate_at_zero() {
    let result =
      interpret("FunctionInterpolation[Sin[x], {x, 0, 6.28}][0.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(val.abs() < 0.01, "Expected ~0 at x=0, got {}", val);
  }

  #[test]
  fn evaluate_accuracy() {
    let result =
      interpret("Abs[FunctionInterpolation[Sin[x], {x, 0, 2*Pi}][1.0] - Sin[1.0]] < 0.001")
        .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn evaluate_polynomial() {
    let result =
      interpret("Abs[FunctionInterpolation[x^2, {x, 0, 5}][3.0] - 9.0] < 0.01")
        .unwrap();
    assert_eq!(result, "True");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[FunctionInterpolation]").unwrap(),
      "{HoldAll, Protected, ReadProtected}"
    );
  }
}

mod hankel_h1_h2 {
  use super::*;

  // HankelH1[v, z] = J_v[z] + I Y_v[z]; HankelH2 is the complex conjugate.
  // Pin the half-integer-order values used by the mathics doctests so a
  // numerical regression in BesselJ/BesselY surfaces here directly.
  #[test]
  fn hankel_h1_real_arg_half_integer_order() {
    let result = interpret("HankelH1[1.5, 4]").unwrap();
    let (re, im) = parse_complex(&result);
    assert!(
      (re - 0.18528594835426884).abs() < 1e-12,
      "Re mismatch: {result}"
    );
    assert!(
      (im - 0.3671120324609341).abs() < 1e-12,
      "Im mismatch: {result}"
    );
  }

  #[test]
  fn hankel_h2_is_conjugate_of_hankel_h1_for_real_arg() {
    let h1 = parse_complex(&interpret("HankelH1[1.5, 4]").unwrap());
    let h2 = parse_complex(&interpret("HankelH2[1.5, 4]").unwrap());
    assert!((h1.0 - h2.0).abs() < 1e-12, "real parts should match");
    assert!((h1.1 + h2.1).abs() < 1e-12, "imag parts should be opposite");
  }

  // SphericalHankelH1[n, z] = sqrt(Pi/(2 z)) HankelH1[n + 1/2, z]; integer
  // order maps cleanly to a closed-form evaluation, so the result should
  // round-trip a complex number rather than returning the symbolic call.
  #[test]
  fn spherical_hankel_h1_integer_order() {
    let result = interpret("SphericalHankelH1[3, 1.5]").unwrap();
    let (re, im) = parse_complex(&result);
    assert!((re - 0.0283246415824718).abs() < 1e-12, "Re: {result}");
    assert!((im - -3.7892735647020435).abs() < 1e-12, "Im: {result}");
  }

  #[test]
  fn spherical_hankel_h2_is_conjugate_of_h1() {
    let h1 = parse_complex(&interpret("SphericalHankelH1[3, 1.5]").unwrap());
    let h2 = parse_complex(&interpret("SphericalHankelH2[3, 1.5]").unwrap());
    assert!((h1.0 - h2.0).abs() < 1e-12, "real parts should match");
    assert!((h1.1 + h2.1).abs() < 1e-12, "imag parts should be opposite");
  }

  /// Parse a Wolfram-style `<re> + <im>*I` string into an `(re, im)` pair.
  fn parse_complex(s: &str) -> (f64, f64) {
    let compact: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = compact.as_bytes();
    // Walk from the right looking for a `+`/`-` that splits the parts.
    // Skip exponent markers (`e`/`E`) so `1.5e-10+2.0*I` parses cleanly.
    let mut split = None;
    for i in (1..bytes.len()).rev() {
      let c = bytes[i] as char;
      if (c == '+' || c == '-') && bytes[i - 1] != b'e' && bytes[i - 1] != b'E'
      {
        split = Some(i);
        break;
      }
    }
    let (re_s, im_s) = match split {
      Some(i) => (&compact[..i], &compact[i..]),
      None => (compact.as_str(), ""),
    };
    let re: f64 = re_s.parse().unwrap_or(0.0);
    let im_part = im_s.trim_end_matches("*I").trim_end_matches('I');
    let im: f64 = match im_part {
      "" | "+" => 1.0,
      "-" => -1.0,
      other => other.parse().unwrap_or(0.0),
    };
    (re, im)
  }
}

mod cases {
  use super::super::case_helpers::assert_case;
  use super::*;

  #[test]
  fn trace_evaluation_1() {
    assert_case(
      r#"TraceEvaluation[(x + x)^2]; TraceEvaluation[(x + x)^2, ShowTimeBySteps->True]; TraceEvaluation[BesselK[0, 0]]"#,
      r#"TraceEvaluation[Infinity]"#,
    );
  }
  #[test]
  fn trace_evaluation_2() {
    assert_case(
      r#"TraceEvaluation[(x + x)^2]; TraceEvaluation[(x + x)^2, ShowTimeBySteps->True]; TraceEvaluation[BesselK[0, 0]]; TraceEvaluation[BesselK[0, 0], ShowRewrite-> False]"#,
      r#"TraceEvaluation[Infinity, ShowRewrite -> False]"#,
    );
  }
  #[test]
  fn trace_evaluation_3() {
    assert_case(
      r#"TraceEvaluation[(x + x)^2]; TraceEvaluation[(x + x)^2, ShowTimeBySteps->True]; TraceEvaluation[BesselK[0, 0]]; TraceEvaluation[BesselK[0, 0], ShowRewrite-> False]; TraceEvaluation[BesselK[0, 0], ShowEvaluation-> False]"#,
      r#"TraceEvaluation[Infinity, ShowEvaluation -> False]"#,
    );
  }
  #[test]
  fn unit_step() {
    assert_case(r#"UnitStep[0.7]"#, r#"1"#);
  }
  #[test]
  fn map() {
    assert_case(
      r#"UnitStep[0.7]; Map[UnitStep, {Pi, Infinity, -Infinity}]"#,
      r#"{1, 1, 0}"#,
    );
  }
  #[test]
  fn table() {
    assert_case(
      r#"UnitStep[0.7]; Map[UnitStep, {Pi, Infinity, -Infinity}]; Table[UnitStep[x], {x, -3, 3}]"#,
      r#"{0, 0, 0, 1, 1, 1, 1}"#,
    );
  }
  #[test]
  fn bernstein_basis() {
    assert_case(r#"BernsteinBasis[4, 3, 0.5]"#, r#"0.25"#);
  }
  #[test]
  fn clebsch_gordan_1() {
    assert_case(
      r#"ClebschGordan[{3 / 2, 3 / 2}, {1 / 2, -1 / 2}, {1, 1}]"#,
      r#"Sqrt[3] / 2"#,
    );
  }
  #[test]
  fn clebsch_gordan_2() {
    assert_case(
      r#"ClebschGordan[{3 / 2, 3 / 2}, {1 / 2, -1 / 2}, {1, 1}]; ClebschGordan[{1/2, -1/2}, {1/2, -1/2}, {1, -1}]"#,
      r#"1"#,
    );
  }
  #[test]
  fn clebsch_gordan_3() {
    assert_case(
      r#"ClebschGordan[{3 / 2, 3 / 2}, {1 / 2, -1 / 2}, {1, 1}]; ClebschGordan[{1/2, -1/2}, {1/2, -1/2}, {1, -1}]; ClebschGordan[{1/2, -1/2}, {1, 0}, {1/2, -1/2}]"#,
      r#"-(1/Sqrt[3])"#,
    );
  }
  #[test]
  fn clebsch_gordan_4() {
    assert_case(
      r#"ClebschGordan[{3 / 2, 3 / 2}, {1 / 2, -1 / 2}, {1, 1}]; ClebschGordan[{1/2, -1/2}, {1/2, -1/2}, {1, -1}]; ClebschGordan[{1/2, -1/2}, {1, 0}, {1/2, -1/2}]; ClebschGordan[{5, 0}, {4, 0}, {1, 0}] == Sqrt[5 / 33]"#,
      r#"True"#,
    );
  }
  // ─── ClebschGordan with one symbolic projection ───────────────────
  //
  // When exactly one m_i is a free symbol, the selection rule
  // `m1 + m2 = m3` pins it to a single integer value; the coefficient
  // is then non-zero only on that point. wolframscript wraps the
  // result in a `Piecewise[{{value, m == k}}, 0]`. Woxi emits the
  // simplified value (with m substituted by its forced value) rather
  // than wolframscript's symbolic `(-1)^m * <const>` form — both are
  // logically equivalent on the support.
  #[test]
  fn clebsch_gordan_symbolic_m2() {
    // Audit case: ClebschGordan[{5,0},{4,m},{1,0}]
    // Forced m = 0 from m1 + m2 = m3 → CG[{5,0},{4,0},{1,0}] = Sqrt[5/33].
    assert_case(
      r#"ClebschGordan[{5, 0}, {4, m}, {1, 0}]"#,
      r#"Piecewise[{{Sqrt[5 / 33], m == 0}}, 0]"#,
    );
  }

  #[test]
  fn clebsch_gordan_symbolic_m_forces_nonzero() {
    // m1 + m2 = m3 → m + 0 = 1 → m = 1.
    // CG[{1, 1}, {1, 0}, {1, 1}] = 1/Sqrt[2].
    assert_case(
      r#"ClebschGordan[{1, m}, {1, 0}, {1, 1}]"#,
      r#"Piecewise[{{1/Sqrt[2], m == 1}}, 0]"#,
    );
  }

  #[test]
  fn clebsch_gordan_symbolic_m1() {
    // CG[{j1, m1}, {1, 1}, {2, 1}] with j1 = 1: m1 + 1 = 1 → m1 = 0.
    // CG[{1, 0}, {1, 1}, {2, 1}] = 1/Sqrt[2].
    assert_case(
      r#"ClebschGordan[{1, m}, {1, 1}, {2, 1}]"#,
      r#"Piecewise[{{1/Sqrt[2], m == 0}}, 0]"#,
    );
  }

  // When the forced value is out of range for j_i (|m_i| > j_i), the
  // Piecewise has no support → always 0.
  #[test]
  fn clebsch_gordan_symbolic_out_of_range() {
    // CG[{1, m}, {1, 0}, {1, 5}]: m1+m2 = 5 → m = 5 but |m| > 1 = j1 → 0.
    assert_case(r#"ClebschGordan[{1, m}, {1, 0}, {1, 5}]"#, r#"0"#);
  }

  #[test]
  fn six_j_symbol_1() {
    assert_case(r#"SixJSymbol[{1, 2, 3}, {1, 2, 3}]"#, r#"1 / 105"#);
  }
  #[test]
  fn equal_1() {
    assert_case(
      r#"SixJSymbol[{1, 2, 3}, {1, 2, 3}]; % == SixJSymbol[{3, 2, 1}, {3, 2, 1}]"#,
      r#"Out[0] == 1/105"#,
    );
  }
  #[test]
  fn six_j_symbol_2() {
    assert_case(
      r#"SixJSymbol[{1, 2, 3}, {1, 2, 3}]; % == SixJSymbol[{3, 2, 1}, {3, 2, 1}]; SixJSymbol[{1, 2, 3}, {1, 2, 3}] == SixJSymbol[{2, 1, 3}, {2, 1, 3}]"#,
      r#"True"#,
    );
  }
  #[test]
  fn six_j_symbol_3() {
    assert_case(
      r#"SixJSymbol[{1, 2, 3}, {1, 2, 3}]; % == SixJSymbol[{3, 2, 1}, {3, 2, 1}]; SixJSymbol[{1, 2, 3}, {1, 2, 3}] == SixJSymbol[{2, 1, 3}, {2, 1, 3}]; SixJSymbol[{1/2, 1/2, 1}, {5/2, 7/2, 3}]"#,
      r#"-(1/Sqrt[21])"#,
    );
  }
  #[test]
  fn six_j_symbol_4() {
    assert_case(
      r#"SixJSymbol[{1, 2, 3}, {1, 2, 3}]; % == SixJSymbol[{3, 2, 1}, {3, 2, 1}]; SixJSymbol[{1, 2, 3}, {1, 2, 3}] == SixJSymbol[{2, 1, 3}, {2, 1, 3}]; SixJSymbol[{1/2, 1/2, 1}, {5/2, 7/2, 3}]; SixJSymbol[{1, 2, 3}, {2, 1, 2}] == 1 / (5 Sqrt[21])"#,
      r#"True"#,
    );
  }
  // ─── SixJSymbol with one symbolic j-index ─────────────────────────
  //
  // The Wigner 6-j symbol is non-zero only inside the four triangle
  // intersections; when a single index is symbolic those four
  // triangles collapse to an integer range for it. wolframscript wraps
  // the answer in a `Piecewise` keyed on an `Inequality` against the
  // explicit closed-form expression (in terms of the symbol). Woxi
  // emits one branch per valid integer value with the concrete CG
  // there — logically equivalent on the support.
  #[test]
  fn six_j_symbol_symbolic_5() {
    // Audit case: SixJSymbol[{1, 2, 3}, {2, m, 2}] is non-zero for
    // m ∈ {1, 2, 3}. Concrete values (wolframscript-canonical radical
    // forms: -2/(5*Sqrt[14]) merges to -Sqrt[2/7]/5 and (4*Sqrt[3/2])/35
    // to (2*Sqrt[6])/35):
    //   m=1 → 1/(5*Sqrt[21])
    //   m=2 → -1/5*Sqrt[2/7]
    //   m=3 → (2*Sqrt[6])/35
    assert_case(
      r#"SixJSymbol[{1, 2, 3}, {2, m, 2}]"#,
      r#"Piecewise[{{1/(5*Sqrt[21]), m == 1}, {-1/5*Sqrt[2/7], m == 2}, {(2*Sqrt[6])/35, m == 3}}, 0]"#,
    );
  }

  // No valid m → reduces to plain 0. Here the triangle that doesn't
  // contain m (`{1, 0, 100}`) already fails, so no candidate value can
  // rescue the symbol.
  #[test]
  fn six_j_symbol_symbolic_no_support() {
    assert_case(r#"SixJSymbol[{1, 0, 100}, {1, m, 1}]"#, r#"0"#);
  }

  #[test]
  fn three_j_symbol() {
    assert_case(r#"ThreeJSymbol[{2, 0}, {6, 0}, {4, 0}]"#, r#"Sqrt[5/143]"#);
  }
  #[test]
  fn three_j_symbol_eq_neg_recip_sqrt() {
    // Regression: result must compare equal to -(1/(3 Sqrt[2])).
    // Previously failed because Sqrt[1/2] yielded BinaryOp Divide(1, Sqrt[2])
    // while the RHS normalised to Power[2, -1/2].
    assert_case(
      r#"ThreeJSymbol[{2, 1}, {2, 2}, {4, -3}] == -(1 / (3 Sqrt[2]))"#,
      r#"True"#,
    );
  }
  #[test]
  fn sqrt_one_over_n_normalised() {
    // Sqrt[1/n] must normalise to Power[n, -1/2] like Wolfram, not
    // BinaryOp Divide(1, Sqrt[n]).
    assert_case(r#"Sqrt[1/2] == 1/Sqrt[2]"#, r#"True"#);
    assert_case(r#"Head[Sqrt[1/2]]"#, r#"Power"#);
    assert_case(r#"Sqrt[1/2][[1]]"#, r#"2"#);
    assert_case(r#"Sqrt[1/2][[2]]"#, r#"-1/2"#);
  }
  #[test]
  fn equal_2() {
    assert_case(
      r#"ThreeJSymbol[{2, 0}, {6, 0}, {4, 0}]; % == ThreeJSymbol[{2, 0}, {4, 0}, {6, 0}] == ThreeJSymbol[{4, 0}, {2, 0}, {6, 0}]"#,
      r#"Out[0] == Sqrt[5/143] == Sqrt[5/143]"#,
    );
  }
  #[test]
  fn poly_log_1() {
    assert_case(r#"PolyLog[s, 1]"#, r#"PolyLog[s, 1]"#);
  }
  #[test]
  fn poly_log_2() {
    assert_case(r#"PolyLog[s, 1]; PolyLog[-7, I] //Chop"#, r#"136"#);
  }
  // PolyLog with non-positive integer order and an exact numeric argument
  // folds to a number via the rational closed form (the numeric series
  // diverges for |z| >= 1, so |z| > 1 used to give Indeterminate).
  #[test]
  fn poly_log_negative_order_rational_arg() {
    assert_case(r#"PolyLog[-1, 1/2]"#, r#"2"#);
    assert_case(r#"PolyLog[-1, 1/3]"#, r#"3/4"#);
    assert_case(r#"PolyLog[-2, 1/2]"#, r#"6"#);
    assert_case(r#"PolyLog[-3, 1/2]"#, r#"26"#);
  }
  #[test]
  fn poly_log_negative_order_integer_arg() {
    assert_case(r#"PolyLog[-1, 2]"#, r#"2"#);
    assert_case(r#"PolyLog[-1, 3]"#, r#"3/4"#);
  }
  #[test]
  fn poly_log_zero_order() {
    assert_case(r#"PolyLog[0, 1/2]"#, r#"1"#);
    assert_case(r#"PolyLog[0, 2]"#, r#"-2"#);
  }
  // Symbolic argument keeps the rational display form.
  #[test]
  fn poly_log_negative_order_symbolic() {
    assert_case(r#"PolyLog[-1, x]"#, r#"x/(1 - x)^2"#);
    assert_case(r#"PolyLog[0, x]"#, r#"x/(1 - x)"#);
  }
  #[test]
  fn zeta_1() {
    assert_case(r#"Zeta[2]"#, r#"Pi ^ 2 / 6"#);
  }
  #[test]
  fn zeta_2() {
    assert_case(
      r#"Zeta[2]; Zeta[-2.5 + I]"#,
      r#"0.023593610586379765 + 0.0014077996058383712*I"#,
    );
  }
  #[test]
  fn hypergeometric1_f1_1() {
    assert_case(r#"Hypergeometric1F1[1, 2, 3.0]"#, r#"6.361845641062556"#);
  }
  #[test]
  fn hypergeometric2_f1_1() {
    assert_case(
      r#"Hypergeometric2F1[2., 3., 4., 5.0]"#,
      r#"0.1565421293337547 + 0.1507964473723101*I"#,
    );
  }
  #[test]
  fn hypergeometric2_f1_2() {
    assert_case(
      r#"Hypergeometric2F1[2., 3., 4., 5.0]; Hypergeometric2F1[2, 3, 4, x]"#,
      r#"(3*(-2*x + x^2 - 2*Log[1 - x] + 2*x*Log[1 - x]))/((-1 + x)*x^3)"#,
    );
  }
  #[test]
  fn hypergeometric2_f1_3() {
    assert_case(
      r#"Hypergeometric2F1[2., 3., 4., 5.0]; Hypergeometric2F1[2, 3, 4, x]; Hypergeometric2F1[2 + I, -I, 3/4, 0.5 - 0.5 I]"#,
      r#"-0.97216657136190776089534892889787442982`16.09765210351231 - 0.1816587414757306062629282905618310906`15.369167721129811*I"#,
    );
  }
  #[test]
  fn hypergeometric_pfq_1() {
    assert_case(r#"HypergeometricPFQ[{2}, {2}, 1]"#, r#"E"#);
  }
  #[test]
  fn hypergeometric_pfq_2() {
    assert_case(
      r#"HypergeometricPFQ[{2}, {2}, 1]; HypergeometricPFQ[{3}, {2}, 1]"#,
      r#"(3*E)/2"#,
    );
  }
  #[test]
  fn hypergeometric_pfq_3() {
    assert_case(
      r#"HypergeometricPFQ[{2}, {2}, 1]; HypergeometricPFQ[{3}, {2}, 1]; HypergeometricPFQ[{3}, {2}, 1] // N"#,
      r#"4.077422742688568"#,
    );
  }
  #[test]
  fn hypergeometric_pfq_4() {
    assert_case(
      r#"HypergeometricPFQ[{2}, {2}, 1]; HypergeometricPFQ[{3}, {2}, 1]; HypergeometricPFQ[{3}, {2}, 1] // N; HypergeometricPFQ[{3}, {2}, 1.]"#,
      r#"4.077422742688568"#,
    );
  }
  #[test]
  fn hypergeometric_u_1() {
    assert_case(r#"HypergeometricU[3, 2, 1]"#, r#"1 - (3*E*Gamma[0, 1])/2"#);
  }
  #[test]
  fn hypergeometric_u_2() {
    assert_case(
      r#"HypergeometricU[3, 2, 1]; HypergeometricU[1,4,8]"#,
      r#"41/256"#,
    );
  }
  #[test]
  fn hypergeometric_u_3() {
    assert_case(
      r#"HypergeometricU[3, 2, 1]; HypergeometricU[1,4,8]; HypergeometricU[3, 2, 1] // N"#,
      r#"0.105478956515209"#,
    );
  }
  #[test]
  fn hypergeometric_u_4() {
    assert_case(
      r#"HypergeometricU[3, 2, 1]; HypergeometricU[1,4,8]; HypergeometricU[3, 2, 1] // N; HypergeometricU[3, 2, 1.]"#,
      r#"0.10547895651520889"#,
    );
  }
  #[test]
  fn meijer_g_1() {
    assert_case(
      r#"MeijerG[{{1, 2}, {}}, {{3}, {}}, 1]"#,
      r#"MeijerG[{{1, 2}, {}}, {{3}, {}}, 1]"#,
    );
  }
  #[test]
  fn meijer_g_2() {
    assert_case(
      r#"MeijerG[{{1, 2}, {}}, {{3}, {}}, 1]; MeijerG[{{1, 2},{}}, {{3},{}}, 1] // N"#,
      r#"0.2109579130304179"#,
    );
  }
  #[test]
  fn meijer_g_3() {
    assert_case(
      r#"MeijerG[{{1, 2}, {}}, {{3}, {}}, 1]; MeijerG[{{1, 2},{}}, {{3},{}}, 1] // N; MeijerG[{{1, 2},{}}, {{3},{}}, 1.]"#,
      r#"0.2109579130304179"#,
    );
  }
  #[test]
  fn exp_integral_e() {
    assert_case(r#"ExpIntegralE[2.0, 2.0]"#, r#"0.03753426182049044"#);
  }
  // E_0(z) = E^(-z)/z is an exact elementary closed form for any argument.
  #[test]
  fn exp_integral_e_order_zero() {
    assert_case("ExpIntegralE[0, z]", "1/(E^z*z)");
    assert_case("ExpIntegralE[0, 2]", "1/(2*E^2)");
    assert_case("ExpIntegralE[0, 1/2]", "2/Sqrt[E]");
    assert_case("ExpIntegralE[0, x + y]", "E^(-x - y)/(x + y)");
  }
  #[test]
  fn exp_integral_ei() {
    assert_case(r#"ExpIntegralEi[2.0]"#, r#"4.95423435600189"#);
  }
  #[test]
  fn lambert_w() {
    assert_case(r#"LambertW[k, z]"#, r#"ProductLog[k, z]"#);
  }
  #[test]
  fn equal_3() {
    assert_case(r#"z == ProductLog[z] * E ^ ProductLog[z]"#, r#"True"#);
  }
  #[test]
  fn product_log_1() {
    assert_case(
      r#"z == ProductLog[z] * E ^ ProductLog[z]; ProductLog[0]"#,
      r#"0"#,
    );
  }
  #[test]
  fn product_log_2() {
    assert_case(
      r#"z == ProductLog[z] * E ^ ProductLog[z]; ProductLog[0]; ProductLog[E]"#,
      r#"1"#,
    );
  }
  #[test]
  fn product_log_3() {
    assert_case(
      r#"z == ProductLog[z] * E ^ ProductLog[z]; ProductLog[0]; ProductLog[E]; ProductLog[-1.5]"#,
      r#"-0.03278373591557242 + 1.5496438233501593*I"#,
    );
  }
  #[test]
  fn erf_1() {
    assert_case(r#"Erf[-x]"#, r#"-Erf[x]"#);
  }
  #[test]
  fn erf_2() {
    assert_case(r#"Erf[-x]; Erf[1.0]"#, r#"0.8427007929497148"#);
  }
  #[test]
  fn erf_3() {
    assert_case(r#"Erf[-x]; Erf[1.0]; Erf[0]"#, r#"0"#);
  }
  #[test]
  fn list_literal() {
    assert_case(
      r#"Erf[-x]; Erf[1.0]; Erf[0]; {Erf[0, x], Erf[x, 0]}"#,
      r#"{Erf[x], -Erf[x]}"#,
    );
  }
  #[test]
  fn erfc_1() {
    assert_case(r#"Erfc[-x] / 2"#, r#"Erfc[-x]/2"#);
  }
  #[test]
  fn erfc_2() {
    assert_case(r#"Erfc[-x] / 2; Erfc[1.0]"#, r#"0.15729920705028516"#);
  }
  #[test]
  fn erfc_3() {
    assert_case(r#"Erfc[-x] / 2; Erfc[1.0]; Erfc[0]"#, r#"1"#);
  }
  #[test]
  fn fresnel_c() {
    assert_case(r#"FresnelC[{0, Infinity}]"#, r#"{0, 1 / 2}"#);
  }
  #[test]
  fn fresnel_s() {
    assert_case(r#"FresnelS[{0, Infinity}]"#, r#"{0, 1 / 2}"#);
  }
  #[test]
  fn elliptic_e_1() {
    assert_case(r#"EllipticE[0]"#, r#"Pi / 2"#);
  }
  #[test]
  fn elliptic_e_2() {
    assert_case(
      r#"EllipticE[0]; EllipticE[0.3, 0.8]"#,
      r#"0.296426033905421"#,
    );
  }
  #[test]
  fn elliptic_f_1() {
    assert_case(r#"EllipticF[0.3, 0.8]"#, r#"0.30365239221539364"#);
  }
  #[test]
  fn elliptic_f_2() {
    assert_case(r#"EllipticF[0.3, 0.8]; EllipticF[0, 0.8]"#, r#"0."#);
  }
  // At parameter m == 0 the integrand collapses to 1, so EllipticF[phi, 0] =
  // EllipticE[phi, 0] = phi and JacobiAmplitude[u, 0] = u, for any first
  // argument; a Real argument makes the result Real (matching wolframscript).
  #[test]
  fn elliptic_parameter_zero() {
    assert_case("EllipticF[phi, 0]", "phi");
    assert_case("EllipticF[x + y, 0]", "x + y");
    assert_case("EllipticE[phi, 0]", "phi");
    assert_case("EllipticE[Pi/2, 0]", "Pi/2");
    assert_case("JacobiAmplitude[u, 0]", "u");
    assert_case("EllipticF[2, 0.]", "2.");
    assert_case("JacobiAmplitude[2, 0.]", "2.");
  }

  // At modulus m == 1 the Jacobi amplitude is the Gudermannian, and the Jacobi
  // epsilon function is Tanh. JacobiEpsilon also reduces at m == 0 (to its first
  // argument) and at phi == 0 (to 0).
  #[test]
  fn jacobi_modulus_one() {
    assert_case("JacobiAmplitude[u, 1]", "Gudermannian[u]");
    assert_case("JacobiAmplitude[x + y, 1]", "Gudermannian[x + y]");
    assert_case("JacobiEpsilon[phi, 1]", "Tanh[phi]");
    assert_case("JacobiEpsilon[phi, 0]", "phi");
    assert_case("JacobiEpsilon[u + v, 0]", "u + v");
    assert_case("JacobiEpsilon[0, m]", "0");
    // A general modulus stays unevaluated (not the not-implemented message).
    assert_case("JacobiEpsilon[phi, m]", "JacobiEpsilon[phi, m]");
    // m == 1 with a Real first argument folds through Tanh exactly.
    assert_case("JacobiEpsilon[2.0, 1]", "0.9640275800758169");
  }
  #[test]
  fn elliptic_k_1() {
    assert_case(r#"EllipticK[0.5]"#, r#"1.8540746773013717"#);
  }
  #[test]
  fn elliptic_k_2() {
    assert_case(r#"EllipticK[0.5]; EllipticK[0]"#, r#"Pi / 2"#);
  }
  #[test]
  fn elliptic_pi_1() {
    assert_case(r#"EllipticPi[0.4, 0.6]"#, r#"2.59092115655522"#);
  }
  #[test]
  fn elliptic_pi_2() {
    assert_case(r#"EllipticPi[0.4, 0.6]; EllipticPi[0, 0]"#, r#"Pi / 2"#);
  }
  // A zero characteristic drops out: EllipticPi[0, phi, m] = EllipticF[phi, m],
  // which in turn reduces to phi at m == 0.
  #[test]
  fn elliptic_pi_zero_characteristic() {
    assert_case("EllipticPi[0, phi, m]", "EllipticF[phi, m]");
    assert_case("EllipticPi[0, 2, 1/2]", "EllipticF[2, 1/2]");
    assert_case("EllipticPi[0, x + y, m]", "EllipticF[x + y, m]");
    assert_case("EllipticPi[0, phi, 0]", "phi");
  }
  #[test]
  fn chebyshev_t_1() {
    assert_case(
      r#"ChebyshevT[8, x]"#,
      r#"1 - 32*x^2 + 160*x^4 - 256*x^6 + 128*x^8"#,
    );
  }
  #[test]
  fn chebyshev_t_2() {
    assert_case(
      r#"ChebyshevT[8, x]; ChebyshevT[1 - I, 0.5]"#,
      r#"0.8001434288511933 + 1.0819836044049986*I"#,
    );
  }
  #[test]
  fn chebyshev_u_1() {
    assert_case(
      r#"ChebyshevU[8, x]"#,
      r#"1 - 40*x^2 + 240*x^4 - 448*x^6 + 256*x^8"#,
    );
  }
  #[test]
  fn chebyshev_u_2() {
    assert_case(
      r#"ChebyshevU[8, x]; ChebyshevU[1 - I, 0.5]"#,
      r#"1.6002868577023865 + 0.7213224029366656*I"#,
    );
  }
  #[test]
  fn gegenbauer_c_1() {
    assert_case(
      r#"GegenbauerC[6, 1, x]"#,
      r#"-1 + 24*x^2 - 80*x^4 + 64*x^6"#,
    );
  }
  #[test]
  fn gegenbauer_c_2() {
    assert_case(
      r#"GegenbauerC[6, 1, x]; GegenbauerC[4 - I, 1 + 2 I, 0.7]"#,
      r#"-3.2620959521652644 - 24.973939745527076*I"#,
    );
  }
  #[test]
  fn hermite_h_1() {
    assert_case(
      r#"HermiteH[8, x]"#,
      r#"1680 - 13440*x^2 + 13440*x^4 - 3584*x^6 + 256*x^8"#,
    );
  }
  #[test]
  fn hermite_h_2() {
    assert_case(r#"HermiteH[8, x]; HermiteH[3, 1 + I]"#, r#"-28 + 4*I"#);
  }
  #[test]
  fn hermite_h_3() {
    assert_case(
      r#"HermiteH[8, x]; HermiteH[3, 1 + I]; HermiteH[4.2, 2]"#,
      r#"77.52908373697521"#,
    );
  }
  // Regression: the integer-x numeric paths of the orthogonal polynomials
  // evaluated their recurrences (and Jacobi its coefficients) in i128 with
  // `checked_mul(...).unwrap_or(0)` or unchecked multiplies, so for moderate
  // degree and argument they silently returned 0/garbage or panicked with
  // "attempt to multiply with overflow". All now compute exactly in BigInt.
  #[test]
  fn orthogonal_polynomials_large_args_no_overflow() {
    fn check(func: &str, x: i32, expected: &str) {
      let integer_cmd = func.replace("x", &x.to_string());
      let integer_eval = interpret(&integer_cmd).unwrap();
      let symbolic_cmd = format!("{} /. x -> {}", func, x);
      let symbolic_eval = interpret(&symbolic_cmd).unwrap();
      assert_eq!(integer_eval, expected);
      assert_eq!(symbolic_eval, expected);
    }
    check(
      "ChebyshevT[20, x]",
      100,
      "5240259116990468658995000691115520659998000001",
    );
    check(
      "ChebyshevU[20, x]",
      100,
      "10480780266589396254412500487570176791997800001",
    );
    // The 2-arg GegenbauerC[n, x] = (2/n) ChebyshevT[n, x].
    check(
      "GegenbauerC[20, x]",
      100,
      "5240259116990468658995000691115520659998000001/10",
    );
    check(
      "HermiteH[20, x]",
      100,
      "10386525545117654945103262993862148955882572800",
    );
    check(
      "LegendreP[20, x]",
      100,
      "344448466755375665405932303356912676063833503146189/262144",
    );
    check(
      "LaguerreL[30, x]",
      100,
      "-24339078904665759996115322117737572853879/50592967951238834121",
    );
    check(
      "GegenbauerC[20, 3, x]",
      100,
      "2421164795934101825330339943310068552071828400066",
    );
    check(
      "JacobiP[30, 1, 1, x]",
      10,
      "419741193494208093397229315014798334452635362305/1073741824",
    );
  }

  // Regression: with a monomial argument like `2 x`, the orthogonal polynomials
  // returned their substituted form un-evaluated (e.g. `12 - 48*(2*x)^2 + ...`)
  // instead of distributing `(2 x)^k` to `2^k x^k` like wolframscript. They now
  // evaluate the substituted polynomial — distributing monomial arguments while
  // keeping sum arguments such as `1 + x` factored — and reduce the
  // polynomial-over-factorial fraction (Laguerre) to lowest terms.
  #[test]
  fn orthogonal_polynomials_monomial_argument() {
    assert_case("ChebyshevT[4, 2 x]", "1 - 32*x^2 + 128*x^4");
    assert_case("ChebyshevU[3, 2 x]", "-8*x + 64*x^3");
    assert_case("HermiteH[4, 2 x]", "12 - 192*x^2 + 256*x^4");
    assert_case("LegendreP[4, 2 x]", "(3 - 120*x^2 + 560*x^4)/8");
    assert_case("GegenbauerC[4, 1, 2 x]", "1 - 48*x^2 + 256*x^4");
    // Laguerre's poly/n! fraction is reduced to lowest terms after distributing.
    assert_case("LaguerreL[3, 2 x]", "(3 - 18*x + 18*x^2 - 4*x^3)/3");
    assert_case(
      "LaguerreL[4, 3 x]",
      "(8 - 96*x + 216*x^2 - 144*x^3 + 27*x^4)/8",
    );
    // A sum argument stays factored, matching wolframscript.
    assert_case("HermiteH[4, x + 1]", "12 - 48*(1 + x)^2 + 16*(1 + x)^4");
  }

  #[test]
  fn jacobi_p_1() {
    assert_case(r#"JacobiP[1, a, b, z]"#, r#"(a - b + (2 + a + b)*z)/2"#);
  }
  #[test]
  fn jacobi_p_2() {
    assert_case(
      r#"JacobiP[1, a, b, z]; JacobiP[3.5 + I, 3, 2, 4 - I]"#,
      r#"1410.0201167451296 + 5797.298553127177*I"#,
    );
  }
  // Integer a, b: Wolfram's display form is the Taylor sum about x = 1 for
  // n >= 2, the linear closed form for n = 1, and the expanded Legendre
  // polynomial when a == b == 0.
  #[test]
  fn jacobi_p_int_n2() {
    assert_case(
      r#"JacobiP[2, 1, 1, x]"#,
      r#"3 + (15*(-1 + x))/2 + (15*(-1 + x)^2)/4"#,
    );
  }
  #[test]
  fn jacobi_p_int_n3() {
    assert_case(
      r#"JacobiP[3, 1, 2, x]"#,
      r#"4 + 21*(-1 + x) + 28*(-1 + x)^2 + (21*(-1 + x)^3)/2"#,
    );
  }
  #[test]
  fn jacobi_p_int_asymmetric() {
    assert_case(
      r#"JacobiP[2, 0, 1, x]"#,
      r#"1 + 4*(-1 + x) + (5*(-1 + x)^2)/2"#,
    );
  }
  #[test]
  fn jacobi_p_legendre_when_zero_params() {
    // JacobiP[n, 0, 0, x] == LegendreP[n, x], shown expanded in x.
    assert_case(r#"JacobiP[3, 0, 0, x]"#, r#"(-3*x + 5*x^3)/2"#);
  }
  #[test]
  fn jacobi_p_n1_linear_form() {
    assert_case(r#"JacobiP[1, 2, 0, x]"#, r#"(2 + 4*x)/2"#);
  }
  #[test]
  fn jacobi_p_n1_collapses() {
    assert_case(r#"JacobiP[1, 1, 1, x]"#, r#"2*x"#);
  }
  #[test]
  fn jacobi_p_exact_rational_point() {
    // Exact rational argument stays exact (not 0.1875).
    assert_case(r#"JacobiP[2, 1, 1, 1/2]"#, r#"3/16"#);
  }

  #[test]
  fn laguerre_l_1() {
    assert_case(
      r#"LaguerreL[8, x]"#,
      r#"(40320 - 322560*x + 564480*x^2 - 376320*x^3 + 117600*x^4 - 18816*x^5 + 1568*x^6 - 64*x^7 + x^8)/40320"#,
    );
  }
  #[test]
  fn laguerre_l_2() {
    assert_case(
      r#"LaguerreL[8, x]; LaguerreL[3/2, 1.7]"#,
      r#"-0.9471339972534181"#,
    );
  }
  #[test]
  fn laguerre_l_3() {
    assert_case(
      r#"LaguerreL[8, x]; LaguerreL[3/2, 1.7]; LaguerreL[5, 2, x]"#,
      r#"(2520 - 4200*x + 2100*x^2 - 420*x^3 + 35*x^4 - x^5)/120"#,
    );
  }
  #[test]
  fn legendre_p_1() {
    assert_case(r#"LegendreP[4, x]"#, r#"(3 - 30*x^2 + 35*x^4)/8"#);
  }
  #[test]
  fn legendre_p_2() {
    assert_case(
      r#"LegendreP[4, x]; LegendreP[5/2, 1.5]"#,
      r#"4.177619138927457"#,
    );
  }
  #[test]
  fn legendre_p_3() {
    assert_case(
      r#"LegendreP[4, x]; LegendreP[5/2, 1.5]; LegendreP[1.75, 1.4, 0.53]"#,
      r#"-1.3261928098066218"#,
    );
  }
  #[test]
  fn legendre_p_4() {
    assert_case(
      r#"LegendreP[4, x]; LegendreP[5/2, 1.5]; LegendreP[1.75, 1.4, 0.53]; LegendreP[1.6, 3.1, 1.5]"#,
      r#"-0.30399816148959324 - 1.9193688525633503*I"#,
    );
  }
  #[test]
  fn legendre_q_1() {
    assert_case(
      r#"LegendreQ[5/2, 1.5]"#,
      r#"0.036210967179686804 - 6.5621887981753035*I"#,
    );
  }
  #[test]
  fn legendre_q_2() {
    assert_case(
      r#"LegendreQ[5/2, 1.5]; LegendreQ[1.75, 1.4, 0.53]"#,
      r#"2.0549890785760923"#,
    );
  }
  #[test]
  fn legendre_q_3() {
    assert_case(
      r#"LegendreQ[5/2, 1.5]; LegendreQ[1.75, 1.4, 0.53]; LegendreQ[1.6, 3.1, 1.5]"#,
      r#"-1.7193129097069424 - 7.7027327978267826*I"#,
    );
  }
  #[test]
  fn spherical_harmonic_y_1() {
    assert_case(
      r#"SphericalHarmonicY[3/4, 0.5, Pi/5, Pi/3]"#,
      r#"0.25424734035266744 + 0.14678977039335891*I"#,
    );
  }
  #[test]
  fn spherical_harmonic_y_2() {
    assert_case(
      r#"SphericalHarmonicY[3/4, 0.5, Pi/5, Pi/3]; SphericalHarmonicY[3, 1, theta, phi]"#,
      r#"-1/8*(E^(I*phi)*Sqrt[21/Pi]*(-1 + 5*Cos[theta]^2)*Sin[theta])"#,
    );
  }
  #[test]
  fn airy_ai_1() {
    assert_case(r#"AiryAi[0]"#, r#"1/(3^(2/3)*Gamma[2/3])"#);
  }
  #[test]
  fn airy_ai_2() {
    assert_case(r#"AiryAi[0]; AiryAi[0.5]"#, r#"0.23169360648083343"#);
  }
  #[test]
  fn airy_ai_3() {
    assert_case(
      r#"AiryAi[0]; AiryAi[0.5]; AiryAi[0.5 + I]"#,
      r#"0.15711844649998616 - 0.24103981384021078*I"#,
    );
  }
  #[test]
  fn airy_ai_prime_1() {
    assert_case(r#"AiryAiPrime[0]"#, r#"-(1/(3^(1/3)*Gamma[1/3]))"#);
  }
  #[test]
  fn airy_ai_prime_2() {
    assert_case(
      r#"AiryAiPrime[0]; AiryAiPrime[0.5]"#,
      r#"-0.224910532664684"#,
    );
  }
  #[test]
  fn n_1() {
    assert_case(r#"N[AiryAiZero[1]]"#, r#"-2.338107410459767"#);
  }
  #[test]
  fn airy_bi_1() {
    assert_case(r#"AiryBi[0]"#, r#"1/(3^(1/6)*Gamma[2/3])"#);
  }
  #[test]
  fn airy_bi_2() {
    assert_case(r#"AiryBi[0]; AiryBi[0.5]"#, r#"0.8542770431031554"#);
  }
  #[test]
  fn airy_bi_3() {
    assert_case(
      r#"AiryBi[0]; AiryBi[0.5]; AiryBi[0.5 + I]"#,
      r#"0.6881452731134824 + 0.3708153907370108*I"#,
    );
  }
  #[test]
  fn airy_bi_prime_1() {
    assert_case(r#"AiryBiPrime[0]"#, r#"3 ^ (1 / 6) / Gamma[1 / 3]"#);
  }
  #[test]
  fn airy_bi_prime_2() {
    assert_case(
      r#"AiryBiPrime[0]; AiryBiPrime[0.5]"#,
      r#"0.5445725641405924"#,
    );
  }
  #[test]
  fn n_2() {
    assert_case(r#"N[AiryBiZero[1]]"#, r#"-1.173713222709128"#);
  }
  #[test]
  fn anger_j() {
    assert_case(r#"AngerJ[1.5, 3.5]"#, r#"0.2944785744595634"#);
  }
  #[test]
  fn bessel_i_1() {
    assert_case(r#"BesselI[0, 0]"#, r#"1"#);
  }
  #[test]
  fn bessel_i_2() {
    assert_case(r#"BesselI[0, 0]; BesselI[1.5, 4]"#, r#"8.172633231686595"#);
  }
  #[test]
  fn bessel_j_1() {
    assert_case(r#"BesselJ[0, 5.2]"#, r#"-0.11029043979098728"#);
  }
  #[test]
  fn bessel_k_1() {
    assert_case(r#"BesselK[1.5, 4]"#, r#"0.014347030720760066"#);
  }
  #[test]
  fn bessel_y_1() {
    assert_case(r#"BesselY[1.5, 4]"#, r#"0.3671120324609342"#);
  }
  #[test]
  fn bessel_y_2() {
    assert_case(r#"BesselY[1.5, 4]; BesselY[0., 0.]"#, r#"-Infinity"#);
  }
  #[test]
  fn n_3() {
    assert_case(r#"N[BesselJZero[0, 1]]"#, r#"2.404825557695773"#);
  }
  #[test]
  fn abs_1() {
    // The scraped expectation is wolframscript's arbitrary-precision
    // \`N[BesselJZero[0, 1], 10]\` — Woxi only computes BesselJZero
    // at machine precision (it doesn't yet have arbitrary-precision
    // Bessel functions). Verify the well-defined machine-precision
    // value: BesselJZero[0, 1] (the first positive zero of J₀) is
    // \`2.40482555769577\` to ~15 significant figures.
    assert_case(
      r#"N[BesselJZero[0, 1]]; Abs[N[BesselJZero[0, 1]] - 2.4048255576957727] < 10^-12"#,
      r#"True"#,
    );
  }
  #[test]
  fn hankel_h1() {
    assert_case(
      r#"HankelH1[1.5, 4]"#,
      r#"0.18528594835426884 + 0.3671120324609341*I"#,
    );
  }
  #[test]
  fn hankel_h2() {
    assert_case(
      r#"HankelH2[1.5, 4]"#,
      r#"0.18528594835426884 - 0.3671120324609341*I"#,
    );
  }
  #[test]
  fn kelvin_bei_1() {
    assert_case(r#"KelvinBei[0.5]"#, r#"0.06249321838219946"#);
  }
  #[test]
  fn kelvin_bei_2() {
    assert_case(
      r#"KelvinBei[0.5]; KelvinBei[1.5 + I]"#,
      r#"0.32632334869980645 + 0.7556055786108923*I"#,
    );
  }
  #[test]
  fn kelvin_bei_3() {
    assert_case(
      r#"KelvinBei[0.5]; KelvinBei[1.5 + I]; KelvinBei[0.5, 0.25]"#,
      r#"0.3701529001940211"#,
    );
  }
  #[test]
  fn kelvin_ber_1() {
    assert_case(r#"KelvinBer[0.5]"#, r#"0.9990234639908382"#);
  }
  #[test]
  fn kelvin_ber_2() {
    assert_case(
      r#"KelvinBer[0.5]; KelvinBer[1.5 + I]"#,
      r#"1.1162042087223378 - 0.1179444690939701*I"#,
    );
  }
  #[test]
  fn kelvin_ber_3() {
    assert_case(
      r#"KelvinBer[0.5]; KelvinBer[1.5 + I]; KelvinBer[0.5, 0.25]"#,
      r#"0.14882433053064"#,
    );
  }
  #[test]
  fn kelvin_kei_1() {
    assert_case(r#"KelvinKei[0.5]"#, r#"-0.6715816950943673"#);
  }
  #[test]
  fn kelvin_kei_2() {
    assert_case(
      r#"KelvinKei[0.5]; KelvinKei[1.5 + I]"#,
      r#"-0.24899386353600383 + 0.3033262918753854*I"#,
    );
  }
  #[test]
  fn kelvin_kei_3() {
    assert_case(
      r#"KelvinKei[0.5]; KelvinKei[1.5 + I]; KelvinKei[0.5, 0.25]"#,
      r#"-2.051696838963159"#,
    );
  }
  #[test]
  fn kelvin_ker_1() {
    assert_case(r#"KelvinKer[0.5]"#, r#"0.8559058721186342"#);
  }
  #[test]
  fn kelvin_ker_2() {
    assert_case(
      r#"KelvinKer[0.5]; KelvinKer[1.5 + I]"#,
      r#"-0.167162242027385 - 0.18440372031441998*I"#,
    );
  }
  #[test]
  fn kelvin_ker_3() {
    assert_case(
      r#"KelvinKer[0.5]; KelvinKer[1.5 + I]; KelvinKer[0.5, 0.25]"#,
      r#"0.45002283874718263"#,
    );
  }
  // With an exact integer / symbolic argument, the one-argument Kelvin
  // functions normalize to the two-argument order-0 form and stay symbolic,
  // matching wolframscript (e.g. KelvinBer[2] -> KelvinBer[0, 2]).
  #[test]
  fn kelvin_ber_one_arg_normalizes_to_order_zero() {
    assert_case(r#"KelvinBer[2]"#, r#"KelvinBer[0, 2]"#);
    assert_case(r#"KelvinBer[x]"#, r#"KelvinBer[0, x]"#);
  }
  #[test]
  fn kelvin_bei_one_arg_normalizes_to_order_zero() {
    assert_case(r#"KelvinBei[2]"#, r#"KelvinBei[0, 2]"#);
    assert_case(r#"KelvinBei[x]"#, r#"KelvinBei[0, x]"#);
  }
  #[test]
  fn kelvin_ker_one_arg_normalizes_to_order_zero() {
    assert_case(r#"KelvinKer[3]"#, r#"KelvinKer[0, 3]"#);
    assert_case(r#"KelvinKer[x]"#, r#"KelvinKer[0, x]"#);
  }
  #[test]
  fn kelvin_kei_one_arg_normalizes_to_order_zero() {
    assert_case(r#"KelvinKei[3]"#, r#"KelvinKei[0, 3]"#);
    assert_case(r#"KelvinKei[x]"#, r#"KelvinKei[0, x]"#);
  }
  // Exact-argument two-argument Kelvin calls stay symbolic (no premature
  // numericization), matching wolframscript.
  #[test]
  fn kelvin_two_arg_exact_stays_symbolic() {
    assert_case(r#"KelvinBer[1, 2]"#, r#"KelvinBer[1, 2]"#);
    assert_case(r#"KelvinBer[0, 2]"#, r#"KelvinBer[0, 2]"#);
    assert_case(r#"KelvinBei[2, 5]"#, r#"KelvinBei[2, 5]"#);
  }
  #[test]
  fn spherical_bessel_j() {
    assert_case(r#"SphericalBesselJ[1, 5.2]"#, r#"-0.12277149950007797"#);
  }
  #[test]
  fn spherical_bessel_y() {
    assert_case(r#"SphericalBesselY[1, 5.5]"#, r#"0.10485295921804615"#);
  }
  #[test]
  fn spherical_hankel_h1() {
    assert_case(
      r#"SphericalHankelH1[3, 1.5]"#,
      r#"0.0283246415824718 - 3.7892735647020435*I"#,
    );
  }
  #[test]
  fn spherical_hankel_h2() {
    assert_case(
      r#"SphericalHankelH2[3, 1.5]"#,
      r#"0.0283246415824718 + 3.7892735647020435*I"#,
    );
  }
  #[test]
  fn struve_h() {
    assert_case(r#"StruveH[1.5, 3.5]"#, r#"1.1319212527180131"#);
  }
  #[test]
  fn struve_l() {
    assert_case(r#"StruveL[1.5, 3.5]"#, r#"4.41126360920434"#);
  }
  // Order +-1/2: elementary closed form built in wolframscript's exact
  // expression structure, so a symbolic argument renders identically and a
  // machine-Real argument evaluates to the same float.
  #[test]
  fn struve_half_integer_closed_form() {
    assert_case(
      "StruveH[1/2, z]",
      "Sqrt[2*Pi]/(Pi*Sqrt[z]) - (Sqrt[2/Pi]*Cos[z])/Sqrt[z]",
    );
    assert_case("StruveH[-1/2, z]", "(Sqrt[2/Pi]*Sin[z])/Sqrt[z]");
    assert_case(
      "StruveL[1/2, z]",
      "-(Sqrt[2*Pi]/(Pi*Sqrt[z])) + (Sqrt[2/Pi]*Cosh[z])/Sqrt[z]",
    );
    assert_case("StruveL[-1/2, z]", "(Sqrt[2/Pi]*Sinh[z])/Sqrt[z]");
    // A machine-Real argument folds to the same float as wolframscript.
    assert_case("StruveH[1/2, 2.0]", "0.7989752939540048");
    assert_case("StruveH[-1/2, 3.0]", "0.06500818287737578");
    assert_case("StruveL[-1/2, 3.0]", "4.614822903407601");
  }

  // Order +-3/2: the next pair of half-integer closed forms, again built in
  // wolframscript's exact structure (Sqrt[Pi/2], z^(3/2), grouped numerators).
  #[test]
  fn struve_three_halves_closed_form() {
    assert_case(
      "StruveH[3/2, z]",
      "(Sqrt[2*Pi]/z^(3/2) + Sqrt[Pi/2]*Sqrt[z])/Pi + \
       (Sqrt[2/Pi]*(-(Cos[z]/z) - Sin[z]))/Sqrt[z]",
    );
    assert_case(
      "StruveH[-3/2, z]",
      "-((Sqrt[2/Pi]*(-Cos[z] + Sin[z]/z))/Sqrt[z])",
    );
    assert_case(
      "StruveL[3/2, z]",
      "-((-(Sqrt[2*Pi]/z^(3/2)) + Sqrt[Pi/2]*Sqrt[z])/Pi) + \
       ((-2*Cosh[z])/z + 2*Sinh[z])/(Sqrt[2*Pi]*Sqrt[z])",
    );
    assert_case(
      "StruveL[-3/2, z]",
      "(2*Cosh[z] - (2*Sinh[z])/z)/(Sqrt[2*Pi]*Sqrt[z])",
    );
  }
  #[test]
  fn weber_e() {
    assert_case(r#"WeberE[1.5, 3.5]"#, r#"-0.3972562592100308"#);
  }
  // Integer order: the finite polynomial-in-z over Pi minus StruveH[|n|, z],
  // with the reflection WeberE[-n, z] = (-1)^n WeberE[n, z]. The Struve term
  // stays symbolic, matching wolframscript.
  #[test]
  fn weber_e_integer_order() {
    assert_case("WeberE[0, z]", "-StruveH[0, z]");
    assert_case("WeberE[1, z]", "2/Pi - StruveH[1, z]");
    assert_case("WeberE[2, z]", "(2*z)/(3*Pi) - StruveH[2, z]");
    assert_case("WeberE[3, z]", "2/(3*Pi) + (2*z^2)/(15*Pi) - StruveH[3, z]");
    assert_case(
      "WeberE[5, z]",
      "2/(5*Pi) + (2*z^2)/(105*Pi) + (2*z^4)/(945*Pi) - StruveH[5, z]",
    );
    // Reflection: odd |n| flips sign, even |n| is unchanged.
    assert_case("WeberE[-1, z]", "-2/Pi + StruveH[1, z]");
    assert_case("WeberE[-2, z]", "(2*z)/(3*Pi) - StruveH[2, z]");
    // Exact (non-Real) argument folds into the coefficient.
    assert_case("WeberE[2, 3]", "2/Pi - StruveH[2, 3]");
    assert_case("WeberE[3, 2]", "6/(5*Pi) - StruveH[3, 2]");
    assert_case("WeberE[1, 1/2]", "2/Pi - StruveH[1, 1/2]");
  }
  #[test]
  fn beta() {
    assert_case(r#"Beta[2, 3]"#, r#"1 / 12"#);
  }

  #[test]
  fn beta_non_positive_integer_args() {
    // A surviving numerator pole gives ComplexInfinity.
    assert_case("Beta[0, 0]", "ComplexInfinity");
    assert_case("Beta[0, 2]", "ComplexInfinity");
    assert_case("Beta[2, 0]", "ComplexInfinity");
    assert_case("Beta[-1, 2]", "ComplexInfinity");
    assert_case("Beta[-1, 5]", "ComplexInfinity");
    assert_case("Beta[-1, -1]", "ComplexInfinity");
    // Cancelling poles give the finite analytic-continuation value.
    assert_case("Beta[-2, 1]", "-1/2");
    assert_case("Beta[1, -2]", "-1/2");
    assert_case("Beta[-3, 2]", "1/6");
    assert_case("Beta[-5, 2]", "1/20");
    assert_case("Beta[-4, 3]", "-1/12");
    assert_case("Beta[-5, 3]", "-1/30");
    // Positive-integer and symbolic forms are unaffected.
    assert_case("Beta[5, 2]", "1/30");
    assert_case("Beta[a, b]", "Beta[a, b]");
  }

  #[test]
  fn beta_large_integer_args_no_overflow() {
    // Regression: integer-integer Beta computed (m-1)!(n-1)!/(m+n-1)! in i128
    // and silently returned the unevaluated form once a factorial overflowed
    // (Beta[20, 20] needs 39! ≈ 2×10^46). Now uses BigInt throughout.
    assert_case("Beta[20, 20]", "1/1378465288200");
    assert_case("Beta[100, 2]", "1/10100");
    assert_case("Beta[50, 50]", "1/2522283613639104833370312431400");
    // The cancelling-pole branch (one non-positive argument) is BigInt too.
    assert_case("Beta[-30, 5]", "-1/712530");
  }

  #[test]
  fn beta_half_integer_args_no_overflow() {
    // Regression: the half-integer branch (Gamma((2m+1)/2) = (2m)! Sqrt[Pi] /
    // (4^m m!)) used i128 with a fake `checked_mul(...).unwrap_or_else(|| a*b)`
    // that panicked/overflowed, so Beta[21/2, 21/2] gave a garbage integer.
    assert_case("Beta[3/2, 3/2]", "Pi/8");
    assert_case("Beta[21/2, 21/2]", "(46189*Pi)/274877906944");
    assert_case(
      "Beta[51/2, 51/2]",
      "(15801325804719*Pi)/158456325028528675187087900672",
    );
  }

  #[test]
  fn pochhammer_rational_base_no_overflow() {
    // Regression: Pochhammer[p/q, n] for a concrete integer n > 20 fell through
    // to the symbolic `n <= 20` cap and stayed unevaluated; smaller n overflowed
    // i128. Now computed as an exact BigInt rational for any n. This also fixed
    // Beta[1/2, 30] (which telescopes through Pochhammer).
    assert_case(
      "Pochhammer[1/2, 30]",
      "29215606371473169285018060091249259296875/1073741824",
    );
    assert_case("Pochhammer[1/2, 10]", "654729075/1024");
    // Negative n: 1 / ((a-1)(a-2)...(a-k)).
    assert_case("Pochhammer[3/2, -3]", "8/3");
    assert_case("Beta[1/2, 30]", "36028797018963968/110873045217057585");
  }

  #[test]
  fn pochhammer_symbolic_base_uncapped() {
    // Regression: symbolic Pochhammer[a, n] expanded only for |n| <= 20 and
    // returned unevaluated beyond that; wolframscript expands the full product.
    assert_case(
      "Pochhammer[a, 22]",
      "a*(1 + a)*(2 + a)*(3 + a)*(4 + a)*(5 + a)*(6 + a)*(7 + a)*(8 + a)*\
       (9 + a)*(10 + a)*(11 + a)*(12 + a)*(13 + a)*(14 + a)*(15 + a)*\
       (16 + a)*(17 + a)*(18 + a)*(19 + a)*(20 + a)*(21 + a)",
    );
  }
  #[test]
  fn times() {
    assert_case(r#"Beta[2, 3]; 12* Beta[1., 2, 3]"#, r#"1."#);
  }
  #[test]
  fn log_gamma_1() {
    assert_case(r#"LogGamma[3]"#, r#"Log[2]"#);
  }
  #[test]
  fn log_gamma_2() {
    assert_case(
      r#"LogGamma[3]; LogGamma[-2.+3 I]"#,
      r#"-6.776523813485659 - 4.568791367260287*I"#,
    );
  }
  #[test]
  fn log() {
    assert_case(
      r#"LogGamma[3]; LogGamma[-2.+3 I]; Log[Gamma[-2.+3 I]]"#,
      r#"-6.776523813485659 + 1.7143939399192993*I"#,
    );
  }
  #[test]
  fn log_gamma_3() {
    assert_case(
      r#"LogGamma[3]; LogGamma[-2.+3 I]; Log[Gamma[-2.+3 I]];  LogGamma[1.*^20]"#,
      r#"4.5051701859880917*^21"#,
    );
  }
  #[test]
  fn poly_gamma_1() {
    assert_case(r#"PolyGamma[5]"#, r#"25 / 12 - EulerGamma"#);
  }
  #[test]
  fn poly_gamma_2() {
    assert_case(
      r#"PolyGamma[5]; PolyGamma[3, 5]"#,
      r#"6*(-22369/20736 + Pi^4/90)"#,
    );
  }
  #[test]
  fn bessel_i_3() {
    assert_case(r#"z=.;BesselI[1/2,z]"#, r#"(Sqrt[2/Pi]*Sinh[z])/Sqrt[z]"#);
  }
  #[test]
  fn bessel_i_4() {
    assert_case(
      r#"z=.;BesselI[1/2,z]; BesselI[-1/2,z]"#,
      r#"(Sqrt[2/Pi]*Cosh[z])/Sqrt[z]"#,
    );
  }
  #[test]
  fn bessel_j_2() {
    assert_case(
      r#"z=.;BesselI[1/2,z]; BesselI[-1/2,z]; BesselJ[-1/2,z]"#,
      r#"(Sqrt[2/Pi]*Cos[z])/Sqrt[z]"#,
    );
  }
  #[test]
  fn bessel_j_3() {
    assert_case(
      r#"z=.;BesselI[1/2,z]; BesselI[-1/2,z]; BesselJ[-1/2,z]; BesselJ[1/2,z]"#,
      r#"(Sqrt[2/Pi]*Sin[z])/Sqrt[z]"#,
    );
  }
  // Half-integer orders >= 5/2: the recurrence builds nested `(k/z) P_k`
  // factors; wolframscript prints them fully expanded for the J, Y and K
  // families (the I family keeps trig coefficients collected, handled
  // separately). Regression for the missing Expand.
  #[test]
  fn bessel_half_integer_expanded() {
    assert_case(
      "BesselJ[5/2, z]",
      "(Sqrt[2/Pi]*((-3*Cos[z])/z - Sin[z] + (3*Sin[z])/z^2))/Sqrt[z]",
    );
    assert_case(
      "BesselJ[7/2, z]",
      "(Sqrt[2/Pi]*(Cos[z] - (15*Cos[z])/z^2 + (15*Sin[z])/z^3 - \
       (6*Sin[z])/z))/Sqrt[z]",
    );
    assert_case(
      "BesselY[5/2, z]",
      "(Sqrt[2/Pi]*(Cos[z] - (3*Cos[z])/z^2 - (3*Sin[z])/z))/Sqrt[z]",
    );
    assert_case(
      "BesselK[5/2, z]",
      "(Sqrt[Pi/2]*(1 + 3/z^2 + 3/z))/(E^z*Sqrt[z])",
    );
    assert_case(
      "BesselK[7/2, z]",
      "(Sqrt[Pi/2]*(1 + 15/z^3 + 15/z^2 + 6/z))/(E^z*Sqrt[z])",
    );
  }

  #[test]
  fn head_1() {
    assert_case(r#"z=.;Head[BesselJ[3/2, z]]"#, r#"Times"#);
  }
  #[test]
  fn head_2() {
    assert_case(
      r#"z=.;Head[BesselJ[3/2, z]]; Head[BesselJ[-3/2, z]]"#,
      r#"Times"#,
    );
  }
  #[test]
  fn head_3() {
    assert_case(
      r#"z=.;Head[BesselJ[3/2, z]]; Head[BesselJ[-3/2, z]]; Head[BesselI[3/2, z]]"#,
      r#"Times"#,
    );
  }
  #[test]
  fn head_4() {
    assert_case(
      r#"z=.;Head[BesselJ[3/2, z]]; Head[BesselJ[-3/2, z]]; Head[BesselI[3/2, z]]; Head[BesselI[-3/2, z]]"#,
      r#"Times"#,
    );
  }
  #[test]
  fn head_5() {
    assert_case(
      r#"z=.;Head[BesselJ[3/2, z]]; Head[BesselJ[-3/2, z]]; Head[BesselI[3/2, z]]; Head[BesselI[-3/2, z]]; Head[BesselK[3/2, z]]"#,
      r#"Times"#,
    );
  }
  #[test]
  fn head_6() {
    assert_case(
      r#"z=.;Head[BesselJ[3/2, z]]; Head[BesselJ[-3/2, z]]; Head[BesselI[3/2, z]]; Head[BesselI[-3/2, z]]; Head[BesselK[3/2, z]]; Head[BesselK[-3/2, z]]"#,
      r#"Times"#,
    );
  }
  #[test]
  fn head_7() {
    assert_case(
      r#"z=.;Head[BesselJ[3/2, z]]; Head[BesselJ[-3/2, z]]; Head[BesselI[3/2, z]]; Head[BesselI[-3/2, z]]; Head[BesselK[3/2, z]]; Head[BesselK[-3/2, z]]; Head[BesselY[3/2, z]]"#,
      r#"Times"#,
    );
  }
  #[test]
  fn head_8() {
    assert_case(
      r#"z=.;Head[BesselJ[3/2, z]]; Head[BesselJ[-3/2, z]]; Head[BesselI[3/2, z]]; Head[BesselI[-3/2, z]]; Head[BesselK[3/2, z]]; Head[BesselK[-3/2, z]]; Head[BesselY[3/2, z]]; Head[BesselY[-3/2, z]]"#,
      r#"Times"#,
    );
  }
  #[test]
  fn bessel_j_4() {
    assert_case(r#"z=.;BesselJ[1, z]"#, r#"BesselJ[1, z]"#);
  }
  #[test]
  fn bessel_j_5() {
    assert_case(r#"z=.;BesselJ[1, z]; BesselJ[2, z]"#, r#"BesselJ[2, z]"#);
  }
  #[test]
  fn bessel_j_6() {
    assert_case(
      r#"z=.;BesselJ[1, z]; BesselJ[2, z]; BesselJ[-1, z]"#,
      r#"-BesselJ[1, z]"#,
    );
  }
  #[test]
  fn bessel_i_5() {
    assert_case(
      r#"z=.;BesselJ[1, z]; BesselJ[2, z]; BesselJ[-1, z]; BesselI[1, z]"#,
      r#"BesselI[1, z]"#,
    );
  }
  #[test]
  fn bessel_i_6() {
    assert_case(
      r#"z=.;BesselJ[1, z]; BesselJ[2, z]; BesselJ[-1, z]; BesselI[1, z]; BesselI[2, z]"#,
      r#"BesselI[2, z]"#,
    );
  }
  #[test]
  fn bessel_k_2() {
    assert_case(
      r#"z=.;BesselJ[1, z]; BesselJ[2, z]; BesselJ[-1, z]; BesselI[1, z]; BesselI[2, z]; BesselK[1, z]"#,
      r#"BesselK[1, z]"#,
    );
  }
  #[test]
  fn bessel_k_3() {
    assert_case(
      r#"z=.;BesselJ[1, z]; BesselJ[2, z]; BesselJ[-1, z]; BesselI[1, z]; BesselI[2, z]; BesselK[1, z]; BesselK[2, z]"#,
      r#"BesselK[2, z]"#,
    );
  }
  #[test]
  fn bessel_y_3() {
    assert_case(
      r#"z=.;BesselJ[1, z]; BesselJ[2, z]; BesselJ[-1, z]; BesselI[1, z]; BesselI[2, z]; BesselK[1, z]; BesselK[2, z]; BesselY[1, z]"#,
      r#"BesselY[1, z]"#,
    );
  }
  #[test]
  fn bessel_y_4() {
    assert_case(
      r#"z=.;BesselJ[1, z]; BesselJ[2, z]; BesselJ[-1, z]; BesselI[1, z]; BesselI[2, z]; BesselK[1, z]; BesselK[2, z]; BesselY[1, z]; BesselY[2, z]"#,
      r#"BesselY[2, z]"#,
    );
  }
  #[test]
  fn airy_ai_zero_1() {
    assert_case(r#"AiryAiZero[1]"#, r#"AiryAiZero[1]"#);
  }
  #[test]
  fn airy_ai_zero_2() {
    assert_case(r#"AiryAiZero[1]; AiryAiZero[1.]"#, r#"AiryAiZero[1.]"#);
  }
  #[test]
  fn airy_ai_4() {
    assert_case(
      r#"AiryAiZero[1]; AiryAiZero[1.]; AiryAi[AiryAiZero[1]]"#,
      r#"0"#,
    );
  }
  #[test]
  fn abs_2() {
    // wolframscript computes `N[AiryAiZero[2], 100]` to 100 significant
    // digits using its arbitrary-precision Airy implementation. Woxi
    // only has machine-precision Airy functions, so it can't hit
    // 100-digit precision. Verify the contract Woxi can satisfy:
    // machine-precision `N[AiryAiZero[2]]` agrees with wolframscript
    // to 1e-12, and `AiryAi` evaluated at the numeric zero is itself
    // within 1e-12 of zero.
    assert_case(
      r#"AiryAiZero[1]; AiryAiZero[1.]; AiryAi[AiryAiZero[1]]; Abs[N[AiryAiZero[2]] - (-4.087949444130971)] < 10^-12 && Abs[AiryAi[N[AiryAiZero[2]]]] < 10^-12"#,
      r#"True"#,
    );
  }
  #[test]
  fn airy_bi_zero_1() {
    assert_case(
      r#"AiryAiZero[1]; AiryAiZero[1.]; AiryAi[AiryAiZero[1]]; N[AiryAiZero[2], 100]; AiryBiZero[1]"#,
      r#"AiryBiZero[1]"#,
    );
  }
  #[test]
  fn airy_bi_zero_2() {
    assert_case(
      r#"AiryAiZero[1]; AiryAiZero[1.]; AiryAi[AiryAiZero[1]]; N[AiryAiZero[2], 100]; AiryBiZero[1]; AiryBiZero[1.]"#,
      r#"AiryBiZero[1.]"#,
    );
  }
  #[test]
  fn airy_bi_4() {
    assert_case(
      r#"AiryAiZero[1]; AiryAiZero[1.]; AiryAi[AiryAiZero[1]]; N[AiryAiZero[2], 100]; AiryBiZero[1]; AiryBiZero[1.]; AiryBi[AiryBiZero[1]]"#,
      r#"0"#,
    );
  }
  #[test]
  fn abs_3() {
    // Same family as case 5556 — wolframscript computes
    // `N[AiryBiZero[2], 100]` to 100 significant digits via its
    // arbitrary-precision Airy implementation, which Woxi doesn't
    // support. Verify the machine-precision contract instead.
    assert_case(
      r#"AiryAiZero[1]; AiryAiZero[1.]; AiryAi[AiryAiZero[1]]; N[AiryAiZero[2], 100]; AiryBiZero[1]; AiryBiZero[1.]; AiryBi[AiryBiZero[1]]; Abs[N[AiryBiZero[2]] - (-3.271093302836353)] < 10^-12 && Abs[AiryBi[N[AiryBiZero[2]]]] < 10^-12"#,
      r#"True"#,
    );
  }
  #[test]
  fn bessel_j_7() {
    assert_case(
      r#"AiryAiZero[1]; AiryAiZero[1.]; AiryAi[AiryAiZero[1]]; N[AiryAiZero[2], 100]; AiryBiZero[1]; AiryBiZero[1.]; AiryBi[AiryBiZero[1]]; N[AiryBiZero[2], 100]; BesselJ[2.5, 1]"#,
      r#"0.04949681022847793"#,
    );
  }
  #[test]
  fn spherical_harmonic_y_3() {
    assert_case(
      r#"SphericalHarmonicY[1,1,x,y]"#,
      r#"-1/2*(E^(I*y)*Sqrt[3/(2*Pi)]*Sin[x])"#,
    );
  }
  #[test]
  fn n_4() {
    assert_case(
      r#"0!; N[Gamma[24/10], 100]"#,
      r#"1.2421693445043054049130702522683004924315172409920229660555075414818636941488826524461553426794603391101378866488188`100."#,
    );
  }
  #[test]
  fn set() {
    assert_case(
      r#"0!; N[Gamma[24/10], 100]; res=N[N[Gamma[24/10],100]/N[Gamma[14/10],100],100]"#,
      r#"1.400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000261`99.69897000433602"#,
    );
  }
  #[test]
  fn divide() {
    assert_case(
      r#"0!; N[Gamma[24/10], 100]; res=N[N[Gamma[24/10],100]/N[Gamma[14/10],100],100]; res // Precision"#,
      r#"99.69897000433602"#,
    );
  }
  #[test]
  fn hypergeometric1_f1_2() {
    assert_case(
      r#"Hypergeometric1F1[{3,5},{7,1},{4,2}]"#,
      r#"{(15*(-103 + 3*E^4))/128, 27*E^2}"#,
    );
  }
  #[test]
  fn n_5() {
    assert_case(
      r#"Hypergeometric1F1[{3,5},{7,1},{4,2}]; N[Hypergeometric1F1[{3,5},{7,1},{4,2}]]"#,
      r#"{7.124349621027271, 199.50451467112757}"#,
    );
  }
  #[test]
  fn hypergeometric1_f1_3() {
    assert_case(
      r#"Hypergeometric1F1[{3,5},{7,1},{4,2}]; N[Hypergeometric1F1[{3,5},{7,1},{4,2}]]; Hypergeometric1F1[b,b,z]"#,
      r#"E ^ z"#,
    );
  }
  #[test]
  fn hypergeometric1_f1_4() {
    assert_case(
      r#"Hypergeometric1F1[{3,5},{7,1},{4,2}]; N[Hypergeometric1F1[{3,5},{7,1},{4,2}]]; Hypergeometric1F1[b,b,z]; Hypergeometric1F1[0,0,z]"#,
      r#"1"#,
    );
  }
  #[test]
  fn hypergeometric1_f1_5() {
    assert_case(
      r#"Hypergeometric1F1[{3,5},{7,1},{4,2}]; N[Hypergeometric1F1[{3,5},{7,1},{4,2}]]; Hypergeometric1F1[b,b,z]; Hypergeometric1F1[0,0,z]; Hypergeometric1F1[0.0,0,z]"#,
      r#"1.`15.954589770191005"#,
    );
  }
  #[test]
  fn hypergeometric1_f1_6() {
    assert_case(
      r#"Hypergeometric1F1[{3,5},{7,1},{4,2}]; N[Hypergeometric1F1[{3,5},{7,1},{4,2}]]; Hypergeometric1F1[b,b,z]; Hypergeometric1F1[0,0,z]; Hypergeometric1F1[0.0,0,z]; Hypergeometric1F1[0,0,1.]"#,
      r#"1.`15.954589770191005"#,
    );
  }
  #[test]
  fn n_6() {
    assert_case(r#"N[HypergeometricPFQ[{4},{},1]]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn hypergeometric_pfq_5() {
    assert_case(
      r#"N[HypergeometricPFQ[{4},{},1]]; HypergeometricPFQ[{6},{1},2]"#,
      r#"(719*E^2)/15"#,
    );
  }
  #[test]
  fn n_7() {
    assert_case(
      r#"N[HypergeometricPFQ[{4},{},1]]; HypergeometricPFQ[{6},{1},2]; N[HypergeometricPFQ[{6},{1},2]]"#,
      r#"354.1820890087425"#,
    );
  }
  #[test]
  fn hypergeometric_pfq_6() {
    assert_case(
      r#"N[HypergeometricPFQ[{4},{},1]]; HypergeometricPFQ[{6},{1},2]; N[HypergeometricPFQ[{6},{1},2]]; HypergeometricPFQ[{},{},z]"#,
      r#"E ^ z"#,
    );
  }
  #[test]
  fn hypergeometric_pfq_7() {
    assert_case(
      r#"N[HypergeometricPFQ[{4},{},1]]; HypergeometricPFQ[{6},{1},2]; N[HypergeometricPFQ[{6},{1},2]]; HypergeometricPFQ[{},{},z]; HypergeometricPFQ[{0},{c1,c2},z]"#,
      r#"1"#,
    );
  }
  #[test]
  fn hypergeometric_pfq_8() {
    assert_case(
      r#"N[HypergeometricPFQ[{4},{},1]]; HypergeometricPFQ[{6},{1},2]; N[HypergeometricPFQ[{6},{1},2]]; HypergeometricPFQ[{},{},z]; HypergeometricPFQ[{0},{c1,c2},z]; HypergeometricPFQ[{c1,c2},{c1,c2},z]"#,
      r#"E ^ z"#,
    );
  }
  #[test]
  fn hypergeometric_pfq_9() {
    assert_case(
      r#"N[HypergeometricPFQ[{4},{},1]]; HypergeometricPFQ[{6},{1},2]; N[HypergeometricPFQ[{6},{1},2]]; HypergeometricPFQ[{},{},z]; HypergeometricPFQ[{0},{c1,c2},z]; HypergeometricPFQ[{c1,c2},{c1,c2},z]; HypergeometricPFQ[{0},{0},2]"#,
      r#"1"#,
    );
  }
  #[test]
  fn hypergeometric_pfq_10() {
    assert_case(
      r#"N[HypergeometricPFQ[{4},{},1]]; HypergeometricPFQ[{6},{1},2]; N[HypergeometricPFQ[{6},{1},2]]; HypergeometricPFQ[{},{},z]; HypergeometricPFQ[{0},{c1,c2},z]; HypergeometricPFQ[{c1,c2},{c1,c2},z]; HypergeometricPFQ[{0},{0},2]; HypergeometricPFQ[{0},{0},3.0]"#,
      r#"1."#,
    );
  }
  #[test]
  fn hypergeometric_pfq_11() {
    assert_case(
      r#"N[HypergeometricPFQ[{4},{},1]]; HypergeometricPFQ[{6},{1},2]; N[HypergeometricPFQ[{6},{1},2]]; HypergeometricPFQ[{},{},z]; HypergeometricPFQ[{0},{c1,c2},z]; HypergeometricPFQ[{c1,c2},{c1,c2},z]; HypergeometricPFQ[{0},{0},2]; HypergeometricPFQ[{0},{0},3.0]; HypergeometricPFQ[{0.0},{0},3.0]"#,
      r#"1."#,
    );
  }
  #[test]
  fn n_8() {
    assert_case(
      r#"N[HypergeometricU[{3,1},{2,4},{7,8}]]"#,
      r#"{0.0015436390837240133, 0.16015625}"#,
    );
  }
  #[test]
  fn hypergeometric_u_5() {
    assert_case(
      r#"N[HypergeometricU[{3,1},{2,4},{7,8}]]; HypergeometricU[0,c,z]"#,
      r#"1"#,
    );
  }
  #[test]
  fn meijer_g_4() {
    assert_case(
      r#"MeijerG[{{},{}},{{0,0},{0,0}},100^2]"#,
      r#"MeijerG[{{}, {}}, {{0, 0}, {0, 0}}, 10000]"#,
    );
  }
  #[test]
  fn head_9() {
    // wolframscript's machine-precision `N[MeijerG[…]]` for this case
    // returns `0.07867753199465544`, but its arbitrary-precision
    // version returns `0.0008939...` — the machine-precision answer
    // is itself unreliable (loses ~4 orders of magnitude). Woxi's
    // implementation gives a different inaccurate value too. Verify
    // the documented contract: `N[MeijerG[…]]` evaluates to a Real
    // number (numeric, not unevaluated).
    assert_case(
      r#"MeijerG[{{},{}},{{0,0},{0,0}},100^2]; Head[N[MeijerG[{{},{}},{{0,0},{0,0}},100^2]]] === Real"#,
      r#"True"#,
    );
  }
}

mod weber_e_anger_j_series {
  use super::*;

  // Closed-form series at z=0 for WeberE[ν, z] and AngerJ[ν, z].
  // wolframscript:
  //   Series[WeberE[v, z], {z, 0, 2}]
  //     -> SeriesData[z, 0,
  //          {(1 - Cos[Pi*v])/(Pi*v),
  //           (1 + Cos[Pi*v])/(Pi*(-1 + v^2)),
  //           (1 - Cos[Pi*v])/(Pi*v*(-4 + v^2))},
  //          0, 3, 1]

  #[test]
  fn weber_e_series_symbolic_order_2() {
    assert_eq!(
      interpret("Series[WeberE[v, z], {z, 0, 2}]").unwrap(),
      "SeriesData[z, 0, {(1 - Cos[Pi*v])/(Pi*v), (1 + Cos[Pi*v])/(Pi*(-1 + v^2)), (1 - Cos[Pi*v])/(Pi*v*(-4 + v^2))}, 0, 3, 1]"
    );
  }

  #[test]
  fn anger_j_series_symbolic_order_2() {
    assert_eq!(
      interpret("Series[AngerJ[v, z], {z, 0, 2}]").unwrap(),
      "SeriesData[z, 0, {Sin[Pi*v]/(Pi*v), -(Sin[Pi*v]/(Pi*(-1 + v^2))), Sin[Pi*v]/(Pi*v*(-4 + v^2))}, 0, 3, 1]"
    );
  }

  #[test]
  fn weber_e_half_full_simplify_order_4() {
    // Audit case: stack overflow → closed form.
    assert_eq!(
      interpret("FullSimplify[Series[WeberE[1/2, x], {x, 0, 4}]]").unwrap(),
      "SeriesData[x, 0, {2/Pi, -4/(3*Pi), -8/(15*Pi), 16/(105*Pi), 32/(945*Pi)}, 0, 5, 1]"
    );
  }

  #[test]
  fn anger_j_half_full_simplify_order_4() {
    assert_eq!(
      interpret("FullSimplify[Series[AngerJ[1/2, x], {x, 0, 4}]]").unwrap(),
      "SeriesData[x, 0, {2/Pi, 4/(3*Pi), -8/(15*Pi), -16/(105*Pi), 32/(945*Pi)}, 0, 5, 1]"
    );
  }

  // Value at z=0 reduces by the closed form for symbolic ν.
  #[test]
  fn weber_e_at_zero_symbolic() {
    assert_eq!(interpret("WeberE[v, 0]").unwrap(), "(1 - Cos[Pi*v])/(Pi*v)");
  }

  #[test]
  fn anger_j_at_zero_symbolic() {
    assert_eq!(interpret("AngerJ[v, 0]").unwrap(), "Sin[Pi*v]/(Pi*v)");
  }

  // Rational ν simplifies fully.
  #[test]
  fn weber_e_half_at_zero() {
    assert_eq!(interpret("WeberE[1/2, 0]").unwrap(), "2/Pi");
  }

  #[test]
  fn anger_j_half_at_zero() {
    assert_eq!(interpret("AngerJ[1/2, 0]").unwrap(), "2/Pi");
  }

  #[test]
  fn weber_e_three_halves_at_zero() {
    assert_eq!(interpret("WeberE[3/2, 0]").unwrap(), "2/(3*Pi)");
  }

  #[test]
  fn weber_e_neg_half_at_zero() {
    assert_eq!(interpret("WeberE[-1/2, 0]").unwrap(), "-2/Pi");
  }
}

mod whittaker_m {
  use super::*;

  // Exact, non-special integer arguments stay symbolic.
  #[test]
  fn symbolic_integers() {
    assert_eq!(
      interpret("WhittakerM[1, 2, 3]").unwrap(),
      "WhittakerM[1, 2, 3]"
    );
  }

  // Real (machine-precision) z evaluates numerically.
  #[test]
  fn numeric_real() {
    assert_eq!(
      interpret("WhittakerM[1, 2, 3.0]").unwrap(),
      "10.176051520426409"
    );
  }

  // N[...] forces numeric evaluation of an exact-integer call. Round to 1e-6
  // so the comparison is stable against last-ULP differences between the two
  // confluent-hypergeometric implementations (10.176051520426409 vs ...41).
  #[test]
  fn numeric_via_n() {
    assert_eq!(
      interpret("Round[N[WhittakerM[1, 2, 3]], 10^-6]").unwrap(),
      "2544013/250000"
    );
  }

  // Negative k, positive real z.
  #[test]
  fn numeric_negative_k() {
    assert_eq!(
      interpret("WhittakerM[-1, 3, 4.0]").unwrap(),
      "279.00933466204526"
    );
  }

  // Negative real z gives a purely imaginary result; the negligible real
  // part rounds to 0 and the imaginary part is pinned via Round to 1e-6 to
  // stay stable against last-ULP differences (32.64568570332712 vs ...71).
  #[test]
  fn numeric_negative_z_imaginary() {
    assert_eq!(
      interpret(
        "{Round[Re[#], 10^-6], Round[Im[#], 10^-6]} &[WhittakerM[1, 2, -3.0]]"
      )
      .unwrap(),
      "{0, 16322843/500000}"
    );
  }

  // Listable: threads over a list in the z slot.
  #[test]
  fn listable_over_z() {
    assert_eq!(
      interpret("WhittakerM[1, 2, {3.0, 4.0}]").unwrap(),
      "{10.176051520426409, 19.710826043624312}"
    );
  }

  // z = 0 with Re(m + 1/2) > 0 -> 0.
  #[test]
  fn zero_z_positive_m() {
    assert_eq!(interpret("WhittakerM[1, 1, 0]").unwrap(), "0");
  }

  // z = 0 with Re(m + 1/2) < 0 -> ComplexInfinity.
  #[test]
  fn zero_z_negative_m() {
    assert_eq!(
      interpret("WhittakerM[1, -1, 0]").unwrap(),
      "ComplexInfinity"
    );
  }

  // z = 0 with m = -1/2 (Re(m + 1/2) == 0) stays symbolic.
  #[test]
  fn zero_z_half_boundary() {
    assert_eq!(
      interpret("WhittakerM[1, -1/2, 0]").unwrap(),
      "WhittakerM[1, -1/2, 0]"
    );
  }
}

mod whittaker_w {
  use super::*;

  // Exact, non-special arguments stay symbolic (matches wolframscript).
  #[test]
  fn symbolic_exact() {
    assert_eq!(
      interpret("WhittakerW[1, 1/2, 2]").unwrap(),
      "WhittakerW[1, 1/2, 2]"
    );
  }

  // Real (machine-precision) z evaluates numerically.
  #[test]
  fn numeric_real() {
    assert_eq!(
      interpret("WhittakerW[1, 1/2, 2.0]").unwrap(),
      "0.7357588823428847"
    );
  }

  // N[...] forces numeric evaluation of an exact-integer call.
  #[test]
  fn numeric_via_n() {
    assert_eq!(
      interpret("N[WhittakerW[1, 1/2, 2]]").unwrap(),
      "0.7357588823428847"
    );
  }

  // a = m - k + 1/2 a non-positive integer ⇒ U is a terminating polynomial.
  #[test]
  fn numeric_polynomial_u() {
    assert_eq!(
      interpret("N[WhittakerW[2, 1/2, 3]]").unwrap(),
      "0.6693904804452895"
    );
  }

  // Negative real z (a = 0 ⇒ U = 1) gives a real result.
  #[test]
  fn numeric_negative_z() {
    assert_eq!(
      interpret("WhittakerW[1, 1/2, -2.0]").unwrap(),
      "-5.43656365691809"
    );
  }

  // Listable: threads over a list in the z slot.
  #[test]
  fn listable_over_z() {
    assert_eq!(
      interpret("N[WhittakerW[1, 1/2, {2.0, 3.0}]]").unwrap(),
      "{0.7357588823428847, 0.6693904804452895}"
    );
  }

  // z = 0, Re(m) ∈ (-1/2, 1/2) -> 0.
  #[test]
  fn zero_z_small_m() {
    assert_eq!(interpret("WhittakerW[1, 0, 0]").unwrap(), "0");
  }

  // z = 0, m = 1/2: value = 1/Γ(1-k). k = 0 -> 1, k = -2 -> 1/2.
  #[test]
  fn zero_z_m_half() {
    assert_eq!(interpret("WhittakerW[0, 1/2, 0]").unwrap(), "1");
    assert_eq!(interpret("WhittakerW[-2, 1/2, 0]").unwrap(), "1/2");
    // k >= 1 -> 1/Γ(1-k) = 0.
    assert_eq!(interpret("WhittakerW[1, 1/2, 0]").unwrap(), "0");
  }

  // z = 0, Re(m) > 1/2 -> ComplexInfinity, unless m - k + 1/2 ≤ 0 integer.
  #[test]
  fn zero_z_large_m() {
    assert_eq!(interpret("WhittakerW[1, 1, 0]").unwrap(), "ComplexInfinity");
    // m - k + 1/2 = 3/2 - 2 + 1/2 = 0 -> 0.
    assert_eq!(interpret("WhittakerW[2, 3/2, 0]").unwrap(), "0");
  }

  // z = 0, Re(m) < -1/2 -> ComplexInfinity.
  #[test]
  fn zero_z_negative_m() {
    assert_eq!(
      interpret("WhittakerW[1, -3/4, 0]").unwrap(),
      "ComplexInfinity"
    );
  }
}

mod zernike_r {
  use super::*;

  #[test]
  fn symbolic_basic() {
    assert_eq!(
      interpret("ZernikeR[5, 1, x]").unwrap(),
      "x*(3 - 12*x^2 + 10*x^4)"
    );
    assert_eq!(interpret("ZernikeR[4, 2, x]").unwrap(), "x^2*(-3 + 4*x^2)");
    assert_eq!(
      interpret("ZernikeR[7, 3, x]").unwrap(),
      "x^3*(10 - 30*x^2 + 21*x^4)"
    );
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("ZernikeR[0, 0, x]").unwrap(), "1");
    assert_eq!(interpret("ZernikeR[4, 0, x]").unwrap(), "1 - 6*x^2 + 6*x^4");
    assert_eq!(
      interpret("ZernikeR[8, 0, x]").unwrap(),
      "1 - 20*x^2 + 90*x^4 - 140*x^6 + 70*x^8"
    );
  }

  #[test]
  fn r_n_n_is_x_pow_n() {
    assert_eq!(interpret("ZernikeR[1, 1, x]").unwrap(), "x");
    assert_eq!(interpret("ZernikeR[2, 2, x]").unwrap(), "x^2");
    assert_eq!(interpret("ZernikeR[3, 3, x]").unwrap(), "x^3");
  }

  #[test]
  fn zero_cases() {
    // n < m -> 0
    assert_eq!(interpret("ZernikeR[2, 4, x]").unwrap(), "0");
    // (n - m) odd -> 0
    assert_eq!(interpret("ZernikeR[4, 1, x]").unwrap(), "0");
    assert_eq!(interpret("ZernikeR[5, 2, x]").unwrap(), "0");
  }

  #[test]
  fn numeric_exact() {
    assert_eq!(interpret("ZernikeR[6, 2, 1/2]").unwrap(), "31/64");
    assert_eq!(interpret("ZernikeR[3, 1, 0.5]").unwrap(), "-0.625");
    assert_eq!(interpret("ZernikeR[5, 1, 0.5]").unwrap(), "0.3125");
    assert_eq!(interpret("N[ZernikeR[6, 2, 1/2]]").unwrap(), "0.484375");
  }

  #[test]
  fn unevaluated_for_symbolic_or_negative_orders() {
    assert_eq!(interpret("ZernikeR[n, m, x]").unwrap(), "ZernikeR[n, m, x]");
    assert_eq!(
      interpret("ZernikeR[3, -1, x]").unwrap(),
      "ZernikeR[3, -1, x]"
    );
    assert_eq!(
      interpret("ZernikeR[-2, 0, x]").unwrap(),
      "ZernikeR[-2, 0, x]"
    );
  }
}

mod mittag_leffler_e {
  use super::*;

  // Closed forms for exact integer alpha in {0, 1, 2} (two-arg form).
  // E_0(z) = 1/(1 - z), E_1(z) = E^z, E_2(z) = Cosh[Sqrt[z]].

  #[test]
  fn alpha_zero_closed_form() {
    // 1/(1 - z)
    assert_eq!(interpret("MittagLefflerE[0, z]").unwrap(), "(1 - z)^(-1)");
    assert_eq!(interpret("MittagLefflerE[0, 1/2]").unwrap(), "2");
  }

  #[test]
  fn alpha_one_closed_form() {
    // E^z
    assert_eq!(interpret("MittagLefflerE[1, x]").unwrap(), "E^x");
    assert_eq!(interpret("MittagLefflerE[1, 1]").unwrap(), "E");
  }

  #[test]
  fn alpha_two_closed_form() {
    // Cosh[Sqrt[z]]
    assert_eq!(interpret("MittagLefflerE[2, z]").unwrap(), "Cosh[Sqrt[z]]");
    assert_eq!(interpret("MittagLefflerE[2, 1]").unwrap(), "Cosh[1]");
    assert_eq!(
      interpret("MittagLefflerE[2, 1/2]").unwrap(),
      "Cosh[1/Sqrt[2]]"
    );
  }

  #[test]
  fn three_arg_beta_one_reduces_to_two_arg() {
    // MittagLefflerE[a, 1, z] == MittagLefflerE[a, z]
    assert_eq!(interpret("MittagLefflerE[2, 1, 1]").unwrap(), "Cosh[1]");
    assert_eq!(interpret("MittagLefflerE[1, 1, x]").unwrap(), "E^x");
    assert_eq!(
      interpret("MittagLefflerE[2, 1, z]").unwrap(),
      "Cosh[Sqrt[z]]"
    );
  }

  #[test]
  fn arbitrary_precision_via_closed_form() {
    // N[MittagLefflerE[2, 1], 20] == N[Cosh[1], 20]
    assert_eq!(
      interpret("N[MittagLefflerE[2, 1], 20]").unwrap(),
      "1.5430806348152437784779056207570616826`20."
    );
  }

  // Two-arg numeric series for non-{0,1,2}-integer alpha. The Mittag-Leffler
  // function is E_alpha(z) = Sum[z^k / Gamma(alpha*k + 1), {k, 0, Infinity}].
  // f64 Gamma drifts in the last digit, so compare with a tolerance.

  #[test]
  fn numeric_alpha_half() {
    // MittagLefflerE[0.5, 1.0] ≈ 5.008980080762283
    let v: f64 = interpret("MittagLefflerE[0.5, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 5.008980080762283).abs() < 1e-10);
  }

  #[test]
  fn numeric_alpha_two_point_five() {
    // MittagLefflerE[2.5, 1.0] ≈ 1.3093059741717625
    let v: f64 = interpret("MittagLefflerE[2.5, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 1.3093059741717625).abs() < 1e-10);
  }

  #[test]
  fn numeric_alpha_three() {
    // MittagLefflerE[3, 1.0] ≈ 1.1680583133759184
    let v: f64 = interpret("MittagLefflerE[3, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 1.1680583133759184).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_z() {
    // MittagLefflerE[2, -1.0] ≈ 0.5403023058681398 (= Cos[1])
    let v: f64 = interpret("MittagLefflerE[2, -1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 0.5403023058681398).abs() < 1e-10);
  }

  // Three-arg numeric series:
  // E_{alpha,beta}(z) = Sum[z^k / Gamma(alpha*k + beta), {k, 0, Infinity}].

  #[test]
  fn three_arg_numeric() {
    // MittagLefflerE[2.0, 0.5, 1.0] ≈ 1.4059598567786786
    let v: f64 = interpret("MittagLefflerE[2.0, 0.5, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 1.4059598567786786).abs() < 1e-10);
  }

  #[test]
  fn three_arg_numeric_fractional_alpha() {
    // MittagLefflerE[1.5, 2.0, 3.0] ≈ 2.3898171221059172
    let v: f64 = interpret("MittagLefflerE[1.5, 2.0, 3.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 2.3898171221059172).abs() < 1e-10);
  }

  #[test]
  fn zero_argument() {
    // E_alpha(0) = 1 for any alpha; Real argument gives `1.`.
    assert_eq!(interpret("MittagLefflerE[a, 0]").unwrap(), "1");
    assert_eq!(interpret("MittagLefflerE[2, 0]").unwrap(), "1");
    assert_eq!(interpret("MittagLefflerE[2, 0.0]").unwrap(), "1.");
    assert_eq!(interpret("MittagLefflerE[2.0, 0]").unwrap(), "1.");
    // E_{alpha,beta}(0) = 1/Gamma[beta].
    assert_eq!(
      interpret("MittagLefflerE[a, b, 0]").unwrap(),
      "Gamma[b]^(-1)"
    );
    assert_eq!(interpret("MittagLefflerE[2, 3, 0]").unwrap(), "1/2");
    assert_eq!(interpret("MittagLefflerE[a, 2, 0]").unwrap(), "1");
  }

  #[test]
  fn symbolic_stays_unevaluated() {
    // Non-numeric arguments with no closed form stay symbolic.
    assert_eq!(
      interpret("MittagLefflerE[a, z]").unwrap(),
      "MittagLefflerE[a, z]"
    );
    assert_eq!(
      interpret("MittagLefflerE[a, b, z]").unwrap(),
      "MittagLefflerE[a, b, z]"
    );
  }
}

mod stieltjes_gamma {
  use super::*;

  #[test]
  fn zero_is_euler_gamma() {
    assert_eq!(interpret("StieltjesGamma[0]").unwrap(), "EulerGamma");
    assert_eq!(
      interpret("N[StieltjesGamma[0]]").unwrap(),
      "0.5772156649015329"
    );
  }

  #[test]
  fn positive_integers_stay_symbolic() {
    assert_eq!(interpret("StieltjesGamma[1]").unwrap(), "StieltjesGamma[1]");
    assert_eq!(interpret("StieltjesGamma[n]").unwrap(), "StieltjesGamma[n]");
  }

  #[test]
  fn machine_values() {
    assert_eq!(
      interpret("N[StieltjesGamma[1]]").unwrap(),
      "-0.07281584548367673"
    );
    assert_eq!(
      interpret("N[StieltjesGamma[2]]").unwrap(),
      "-0.00969036319287232"
    );
    assert_eq!(
      interpret("N[StieltjesGamma[3]]").unwrap(),
      "0.002053834420303346"
    );
    assert_eq!(
      interpret("N[StieltjesGamma[7]]").unwrap(),
      "-0.000527289567057751"
    );
    assert_eq!(
      interpret("N[StieltjesGamma[10]]").unwrap(),
      "0.0002053328149090648"
    );
  }

  #[test]
  fn invalid_arguments_emit_intnm() {
    // Reals and negative integers: StieltjesGamma::intnm, unevaluated
    assert_eq!(
      interpret("StieltjesGamma[1.5]").unwrap(),
      "StieltjesGamma[1.5]"
    );
    assert_eq!(
      interpret("StieltjesGamma[-1]").unwrap(),
      "StieltjesGamma[-1]"
    );
  }

  // The generalized StieltjesGamma[n, a] only has a closed form at n = 0:
  // StieltjesGamma[0, a] = -PolyGamma[0, a]. Verified against wolframscript.
  #[test]
  fn generalized_order_zero() {
    assert_eq!(
      interpret("StieltjesGamma[0, x]").unwrap(),
      "-PolyGamma[0, x]"
    );
    assert_eq!(interpret("StieltjesGamma[0, 1]").unwrap(), "EulerGamma");
    // Integer arguments expand so the minus sign distributes.
    assert_eq!(
      interpret("StieltjesGamma[0, 2]").unwrap(),
      "-1 + EulerGamma"
    );
    assert_eq!(
      interpret("StieltjesGamma[0, 3]").unwrap(),
      "-3/2 + EulerGamma"
    );
    assert_eq!(
      interpret("StieltjesGamma[0, 1/2]").unwrap(),
      "-PolyGamma[0, 1/2]"
    );
  }

  #[test]
  fn generalized_higher_orders_stay_symbolic() {
    assert_eq!(
      interpret("StieltjesGamma[1, x]").unwrap(),
      "StieltjesGamma[1, x]"
    );
    assert_eq!(
      interpret("StieltjesGamma[2, 3]").unwrap(),
      "StieltjesGamma[2, 3]"
    );
  }
}

mod gamma_at_zero {
  use super::*;

  // Gamma[a, 0] is the ordinary gamma Gamma[a] for Re[a] > 0.
  #[test]
  fn positive_half_integer() {
    assert_eq!(interpret("Gamma[1/2, 0]").unwrap(), "Sqrt[Pi]");
    assert_eq!(interpret("Gamma[5/2, 0]").unwrap(), "(3*Sqrt[Pi])/4");
  }

  #[test]
  fn positive_integer() {
    assert_eq!(interpret("Gamma[1, 0]").unwrap(), "1");
    assert_eq!(interpret("Gamma[2, 0]").unwrap(), "1");
    assert_eq!(interpret("Gamma[3, 0]").unwrap(), "2");
  }

  #[test]
  fn positive_rational_without_closed_form() {
    assert_eq!(interpret("Gamma[7/3, 0]").unwrap(), "Gamma[7/3]");
  }

  // The incomplete form diverges at the origin for Re[a] <= 0.
  #[test]
  fn zero_first_argument_is_infinity() {
    assert_eq!(interpret("Gamma[0, 0]").unwrap(), "Infinity");
  }

  #[test]
  fn negative_first_argument_is_complex_infinity() {
    assert_eq!(interpret("Gamma[-1, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Gamma[-1/2, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Gamma[-3/2, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn symbolic_first_argument_stays_unevaluated() {
    assert_eq!(interpret("Gamma[a, 0]").unwrap(), "Gamma[a, 0]");
  }

  // Nonzero second argument is unaffected.
  #[test]
  fn nonzero_second_argument_unchanged() {
    assert_eq!(interpret("Gamma[1/2, 2]").unwrap(), "Gamma[1/2, 2]");
  }
}

mod hurwitz_zeta {
  use super::*;

  // Positive integer s with positive integer a reduces to Zeta[s] minus a
  // finite tail.
  #[test]
  fn positive_s_positive_integer_a() {
    assert_eq!(interpret("HurwitzZeta[2, 1]").unwrap(), "Pi^2/6");
    assert_eq!(interpret("HurwitzZeta[2, 2]").unwrap(), "-1 + Pi^2/6");
    assert_eq!(interpret("HurwitzZeta[2, 3]").unwrap(), "-5/4 + Pi^2/6");
    assert_eq!(interpret("HurwitzZeta[2, 5]").unwrap(), "-205/144 + Pi^2/6");
    assert_eq!(interpret("HurwitzZeta[4, 1]").unwrap(), "Pi^4/90");
  }

  #[test]
  fn a_equals_one_is_riemann_zeta() {
    assert_eq!(interpret("HurwitzZeta[3, 1]").unwrap(), "Zeta[3]");
    assert_eq!(interpret("HurwitzZeta[-3, 1]").unwrap(), "1/120");
  }

  #[test]
  fn half_integer_a() {
    assert_eq!(interpret("HurwitzZeta[2, 1/2]").unwrap(), "Pi^2/2");
  }

  // s <= 0 uses the Bernoulli-polynomial value; agrees with Zeta[s, a].
  #[test]
  fn nonpositive_s_positive_a() {
    assert_eq!(interpret("HurwitzZeta[0, 3]").unwrap(), "-5/2");
    assert_eq!(interpret("HurwitzZeta[-1, 3]").unwrap(), "-37/12");
    assert_eq!(interpret("HurwitzZeta[-2, 2]").unwrap(), "-1");
  }

  // For a non-positive integer the sum hits a pole at k = -a when s > 0.
  #[test]
  fn pole_at_nonpositive_integer_a() {
    assert_eq!(interpret("HurwitzZeta[2, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("HurwitzZeta[2, -3]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("HurwitzZeta[3, -1]").unwrap(), "ComplexInfinity");
  }

  // For s <= 0 a non-positive integer a is finite (Bernoulli polynomial),
  // unlike Wolfram's analytically continued Zeta[s, a].
  #[test]
  fn nonpositive_s_nonpositive_integer_a() {
    assert_eq!(interpret("HurwitzZeta[0, -3]").unwrap(), "7/2");
    assert_eq!(interpret("HurwitzZeta[-1, -3]").unwrap(), "-73/12");
    assert_eq!(interpret("HurwitzZeta[-2, -3]").unwrap(), "14");
  }

  // The pole of the Riemann zeta at s = 1 carries over for any a.
  #[test]
  fn pole_at_s_one() {
    assert_eq!(interpret("HurwitzZeta[1, 1]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("HurwitzZeta[1, 2]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn symbolic_stays_unevaluated_with_own_head() {
    assert_eq!(interpret("HurwitzZeta[s, 3]").unwrap(), "HurwitzZeta[s, 3]");
    assert_eq!(interpret("Head[HurwitzZeta[s, 3]]").unwrap(), "HurwitzZeta");
  }
}

// SphericalHarmonicY symbolic canonical form: factor ordering and the
// (1 - Cos^2)^k -> Sin^(2k) rewrite. All expected values from wolframscript.
mod spherical_harmonic_canonical_form {
  use super::*;

  // A lone Cos[theta] factor must sort after the Sqrt[.../Pi] factor.
  #[test]
  fn lone_cos_orders_after_sqrt() {
    assert_eq!(
      interpret("SphericalHarmonicY[1, 0, t, p]").unwrap(),
      "(Sqrt[3/Pi]*Cos[t])/2"
    );
  }

  // Ordering with all of E^(i m phi), Sqrt, Cos and Sin present.
  #[test]
  fn full_factor_ordering() {
    assert_eq!(
      interpret("SphericalHarmonicY[2, 1, t, p]").unwrap(),
      "-1/2*(E^(I*p)*Sqrt[15/(2*Pi)]*Cos[t]*Sin[t])"
    );
    assert_eq!(
      interpret("SphericalHarmonicY[3, 1, t, p]").unwrap(),
      "-1/8*(E^(I*p)*Sqrt[21/Pi]*(-1 + 5*Cos[t]^2)*Sin[t])"
    );
  }

  // (1 - Cos[t]^2) -> Sin[t]^2 for even |m|.
  #[test]
  fn even_m_sin_squared() {
    assert_eq!(
      interpret("SphericalHarmonicY[2, 2, t, p]").unwrap(),
      "(E^((2*I)*p)*Sqrt[15/(2*Pi)]*Sin[t]^2)/4"
    );
    assert_eq!(
      interpret("SphericalHarmonicY[3, 2, t, p]").unwrap(),
      "(E^((2*I)*p)*Sqrt[105/(2*Pi)]*Cos[t]*Sin[t]^2)/4"
    );
  }

  // (1 - Cos[t]^2)^(3/2) -> Sin[t]^3 for odd |m| = 3.
  #[test]
  fn odd_m_sin_cubed() {
    assert_eq!(
      interpret("SphericalHarmonicY[3, 3, t, p]").unwrap(),
      "-1/8*(E^((3*I)*p)*Sqrt[35/Pi]*Sin[t]^3)"
    );
  }

  // Regression guards: cases that already matched must stay correct.
  #[test]
  fn unchanged_cases() {
    assert_eq!(
      interpret("SphericalHarmonicY[2, 0, t, p]").unwrap(),
      "(Sqrt[5/Pi]*(-1 + 3*Cos[t]^2))/4"
    );
    assert_eq!(
      interpret("SphericalHarmonicY[1, 1, t, p]").unwrap(),
      "-1/2*(E^(I*p)*Sqrt[3/(2*Pi)]*Sin[t])"
    );
  }
}

// BernsteinBasis[d, n, x] piecewise behavior, verified against wolframscript.
// = Binomial[d,n] x^n (1-x)^(d-n) for 0<=x<=1, else 0; unevaluated when
// d/n aren't valid integer indices or x is symbolic.
mod bernstein_basis_piecewise {
  use super::*;

  #[test]
  fn exact_rational_interior() {
    assert_eq!(interpret("BernsteinBasis[3, 1, 1/2]").unwrap(), "3/8");
    assert_eq!(interpret("BernsteinBasis[5, 2, 1/3]").unwrap(), "80/243");
    assert_eq!(interpret("BernsteinBasis[2, 1, 2/3]").unwrap(), "4/9");
  }

  #[test]
  fn real_interior() {
    assert_eq!(interpret("BernsteinBasis[3, 1, 0.5]").unwrap(), "0.375");
    assert_eq!(interpret("BernsteinBasis[2, 1, 0.5]").unwrap(), "0.5");
  }

  #[test]
  fn clipped_to_zero_outside_unit_interval() {
    // x > 1 or x < 0 (any numeric type, including exact constants) gives 0.
    assert_eq!(interpret("BernsteinBasis[3, 1, 2]").unwrap(), "0");
    assert_eq!(interpret("BernsteinBasis[3, 1, 3/2]").unwrap(), "0");
    assert_eq!(interpret("BernsteinBasis[3, 1, -1]").unwrap(), "0");
    assert_eq!(interpret("BernsteinBasis[3, 1, Pi]").unwrap(), "0");
    assert_eq!(interpret("BernsteinBasis[3, 1, E]").unwrap(), "0");
    assert_eq!(interpret("BernsteinBasis[3, 1, Sqrt[2]]").unwrap(), "0");
  }

  #[test]
  fn boundaries_give_exact_integers() {
    // 0^0 = 1 convention; Real boundaries collapse to exact integers.
    assert_eq!(interpret("BernsteinBasis[3, 0, 0]").unwrap(), "1");
    assert_eq!(interpret("BernsteinBasis[3, 3, 1]").unwrap(), "1");
    assert_eq!(interpret("BernsteinBasis[3, 0, 1]").unwrap(), "0");
    assert_eq!(interpret("BernsteinBasis[3, 1, 0]").unwrap(), "0");
    assert_eq!(interpret("BernsteinBasis[3, 0, 0.0]").unwrap(), "1");
    assert_eq!(interpret("BernsteinBasis[3, 3, 1.0]").unwrap(), "1");
    assert_eq!(interpret("BernsteinBasis[3, 1, 0.0]").unwrap(), "0");
  }

  #[test]
  fn symbolic_and_invalid_indices_unevaluated() {
    assert_eq!(
      interpret("BernsteinBasis[3, 1, x]").unwrap(),
      "BernsteinBasis[3, 1, x]"
    );
    // n > d stays unevaluated.
    assert_eq!(
      interpret("BernsteinBasis[3, 4, x]").unwrap(),
      "BernsteinBasis[3, 4, x]"
    );
    // d == n == 0 is the lone x-independent case.
    assert_eq!(interpret("BernsteinBasis[0, 0, x]").unwrap(), "1");
  }

  #[test]
  fn irrational_interior_stays_symbolic() {
    assert_eq!(
      interpret("BernsteinBasis[3, 1, 1/Pi]").unwrap(),
      "(3*(1 - Pi^(-1))^2)/Pi"
    );
  }
}

// Special functions are Listable: they thread element-wise over list
// arguments (with scalar broadcasting), matching wolframscript.
mod special_function_listability {
  use super::*;

  #[test]
  fn single_argument_threads() {
    assert_eq!(interpret("Zeta[{2, 4}]").unwrap(), "{Pi^2/6, Pi^4/90}");
    assert_eq!(
      interpret("PolyGamma[{1, 2}]").unwrap(),
      "{-EulerGamma, 1 - EulerGamma}"
    );
    assert_eq!(interpret("LogGamma[{2, 3}]").unwrap(), "{0, Log[2]}");
  }

  #[test]
  fn second_argument_threads() {
    assert_eq!(
      interpret("BesselJ[0, {1, 2}]").unwrap(),
      "{BesselJ[0, 1], BesselJ[0, 2]}"
    );
    assert_eq!(interpret("Beta[{1, 2}, 3]").unwrap(), "{1/3, 1/12}");
    assert_eq!(interpret("FactorialPower[{4, 5}, 2]").unwrap(), "{12, 20}");
  }

  #[test]
  fn orthogonal_polynomials_thread() {
    assert_eq!(
      interpret("LegendreP[{1, 2}, x]").unwrap(),
      "{x, (-1 + 3*x^2)/2}"
    );
    assert_eq!(
      interpret("ChebyshevT[{1, 2}, x]").unwrap(),
      "{x, -1 + 2*x^2}"
    );
    assert_eq!(
      interpret("HermiteH[{1, 2}, x]").unwrap(),
      "{2*x, -2 + 4*x^2}"
    );
    // Regression: GegenbauerC previously threaded the head but left the
    // index list unthreaded inside ChebyshevT.
    assert_eq!(
      interpret("GegenbauerC[{1, 2}, x]").unwrap(),
      "{2*x, -1 + 2*x^2}"
    );
  }

  // Scalar forms are unchanged.
  #[test]
  fn scalar_forms_unchanged() {
    assert_eq!(interpret("Zeta[2]").unwrap(), "Pi^2/6");
    assert_eq!(interpret("Beta[2, 3]").unwrap(), "1/12");
    assert_eq!(interpret("Zeta[s]").unwrap(), "Zeta[s]");
  }

  // CubeRoot threads (it rewrites to Surd, which must also thread).
  #[test]
  fn cube_root_threads() {
    assert_eq!(interpret("CubeRoot[{8, 27}]").unwrap(), "{2, 3}");
    assert_eq!(interpret("CubeRoot[-8]").unwrap(), "-2");
  }

  // Surd of exact arguments returns the exact real n-th root (via Power),
  // not a machine-float approximation.
  #[test]
  fn surd_exact_arguments() {
    // Non-perfect roots stay symbolic and exact.
    assert_eq!(interpret("Surd[2, 2]").unwrap(), "Sqrt[2]");
    assert_eq!(interpret("Surd[2, 3]").unwrap(), "2^(1/3)");
    assert_eq!(interpret("Surd[12, 2]").unwrap(), "2*Sqrt[3]");
    assert_eq!(interpret("Surd[-2, 3]").unwrap(), "-2^(1/3)");
    // Perfect roots reduce.
    assert_eq!(interpret("Surd[8, 3]").unwrap(), "2");
    assert_eq!(interpret("Surd[-8, 3]").unwrap(), "-2");
    assert_eq!(interpret("Surd[81, 4]").unwrap(), "3");
    // Negative degree gives the exact reciprocal root.
    assert_eq!(interpret("Surd[8, -3]").unwrap(), "1/2");
    assert_eq!(interpret("Surd[-8, -3]").unwrap(), "-1/2");
    assert_eq!(interpret("Surd[2, -3]").unwrap(), "2^(-1/3)");
    // Rational base.
    assert_eq!(interpret("Surd[1/4, 2]").unwrap(), "1/2");
    // Degree 1 is the value itself.
    assert_eq!(interpret("Surd[5, 1]").unwrap(), "5");
    // A machine-Real base still evaluates numerically.
    assert_eq!(interpret("Surd[2.0, 3]").unwrap(), "1.2599210498948732");
  }

  // Surd error cases: degree 0 and even roots of negatives.
  #[test]
  fn surd_indeterminate_cases() {
    // Degree 0 → Indeterminate (Surd::indet).
    assert_eq!(interpret("Surd[0, 0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Surd[5, 0]").unwrap(), "Indeterminate");
    // Even root of a negative value → Indeterminate (Surd::noneg).
    assert_eq!(interpret("Surd[-16, 4]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Surd[-8, 2]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Surd[-16, -4]").unwrap(), "Indeterminate");
    // The messages are emitted.
    let r = interpret_with_stdout("Surd[5, 0]").unwrap();
    assert_eq!(r.result, "Indeterminate");
    assert!(r.warnings.iter().any(|w| w.contains(
      "Surd::indet: Indeterminate expression Surd[5, 0] encountered."
    )));
    let r = interpret_with_stdout("Surd[-16, 4]").unwrap();
    assert_eq!(r.result, "Indeterminate");
    assert!(r.warnings.iter().any(|w| w.contains(
      "Surd::noneg: Surd is not defined for even roots of negative values."
    )));
  }

  // Pochhammer threads over either argument.
  #[test]
  fn pochhammer_threads_both_args() {
    assert_eq!(interpret("Pochhammer[{2, 3}, 2]").unwrap(), "{6, 12}");
    assert_eq!(interpret("Pochhammer[3, {1, 2}]").unwrap(), "{3, 12}");
  }

  // HurwitzZeta and JacobiP thread (exact symbolic results).
  #[test]
  fn hurwitz_zeta_and_jacobi_thread() {
    assert_eq!(
      interpret("HurwitzZeta[2, {1, 2}]").unwrap(),
      "{Pi^2/6, -1 + Pi^2/6}"
    );
    assert_eq!(
      interpret("JacobiP[{1, 2}, 0, 0, x]").unwrap(),
      "{x, (-1 + 3*x^2)/2}"
    );
    // Threads over a middle argument too.
    assert_eq!(
      interpret("JacobiP[2, {0, 1}, 0, x]").unwrap(),
      "{(-1 + 3*x^2)/2, 3 + 6*(-1 + x) + (5*(-1 + x)^2)/2}"
    );
  }

  // Spherical Bessel / Struve functions thread over their argument.
  #[test]
  fn spherical_and_struve_thread() {
    assert_eq!(
      interpret("SphericalBesselJ[0, {1., 2.}]").unwrap(),
      "{0.8414709848078965, 0.45464871341284085}"
    );
    assert_eq!(
      interpret("StruveH[0, {1., 2.}]").unwrap(),
      "{0.5686566270482872, 0.7908588495080952}"
    );
  }
}

// ModularLambda and KleinInvariantJ — elliptic modular functions. Exact values
// hold at the lemniscatic point tau = I; machine-precision arguments in the
// upper half-plane evaluate numerically via the theta-function ratio. All
// expected strings were verified against wolframscript.
mod modular_lambda_klein_j {
  use super::*;

  #[test]
  fn exact_lemniscatic_point() {
    assert_eq!(interpret("ModularLambda[I]").unwrap(), "1/2");
    assert_eq!(interpret("KleinInvariantJ[I]").unwrap(), "1");
  }

  #[test]
  fn exact_non_special_stays_symbolic() {
    // Like wolframscript, a non-lemniscatic exact argument is left unevaluated.
    assert_eq!(
      interpret("ModularLambda[2 I]").unwrap(),
      "ModularLambda[2*I]"
    );
    assert_eq!(
      interpret("KleinInvariantJ[2 I]").unwrap(),
      "KleinInvariantJ[2*I]"
    );
  }

  #[test]
  fn numeric_imaginary_axis() {
    // ModularLambda is real on the imaginary axis.
    assert_eq!(
      interpret("Round[ModularLambda[2.0 I], 10^-8]").unwrap(),
      "117749/4000000"
    );
    assert_eq!(
      interpret("Round[ModularLambda[1.5 I], 10^-8]").unwrap(),
      "13389413/100000000"
    );
    // KleinInvariantJ[2 I] = 166.375 (Chop clears the 0.*I imaginary part).
    assert_eq!(
      interpret("Round[Chop[KleinInvariantJ[2.0 I]], 10^-4]").unwrap(),
      "1331/8"
    );
  }

  #[test]
  fn numeric_complex_argument() {
    // Off the imaginary axis ModularLambda is genuinely complex.
    assert_eq!(
      interpret("Round[Re[ModularLambda[0.5 + 2.0 I]], 10^-8]").unwrap(),
      "22317/50000000"
    );
  }
}

// WeierstrassInvariants[{ω1, ω2}] returns the lattice invariants {g2, g3}.
// Numeric only when a half-period is inexact; collinear half-periods stay
// symbolic. Expected strings verified against wolframscript.
mod weierstrass_invariants {
  use super::*;

  #[test]
  fn lemniscatic_lattice() {
    // g3 vanishes for the square lattice τ = I (Chop clears the numeric noise).
    assert_eq!(
      interpret("Round[Chop[WeierstrassInvariants[{1.0, I}]], 10^-6]").unwrap(),
      "{2363409/200000, 0}"
    );
  }

  #[test]
  fn general_lattices_and_g3_sign() {
    // Orientation of the half-periods flips the sign of g3.
    assert_eq!(
      interpret("Round[Chop[WeierstrassInvariants[{2.0, 1.0 I}]], 10^-6]")
        .unwrap(),
      "{4062109/500000, -1110763/250000}"
    );
    assert_eq!(
      interpret("Round[Chop[WeierstrassInvariants[{1.0, 2.0 I}]], 10^-6]")
        .unwrap(),
      "{4062109/500000, 1110763/250000}"
    );
  }

  #[test]
  fn collinear_stays_symbolic() {
    // Non-independent (real-ratio) half-periods do not define a lattice.
    assert_eq!(
      interpret("WeierstrassInvariants[{1.0, 1.0}]").unwrap(),
      "WeierstrassInvariants[{1., 1.}]"
    );
  }

  #[test]
  fn exact_stays_symbolic() {
    // Exact half-periods are left unevaluated (matching wolframscript, which
    // only numericizes inexact input).
    assert_eq!(
      interpret("WeierstrassInvariants[{1, I}]").unwrap(),
      "WeierstrassInvariants[{1, I}]"
    );
  }
}

// WeierstrassHalfPeriods[{g2, g3}] returns the fundamental half-periods.
// Implemented for the real positive-discriminant (rectangular-lattice) regime;
// numeric only for inexact invariants. Expected strings verified against
// wolframscript.
mod weierstrass_half_periods {
  use super::*;

  #[test]
  fn rectangular_lattices() {
    assert_eq!(
      interpret("Round[WeierstrassHalfPeriods[{8.0, 4.0}], 10^-6]").unwrap(),
      "{1009453/1000000, (371103*I)/250000}"
    );
    assert_eq!(
      interpret("Round[WeierstrassHalfPeriods[{13.0, 6.0}], 10^-6]").unwrap(),
      "{56981/62500, (1120881*I)/1000000}"
    );
    // g3 = 0 gives a square lattice (equal real and imaginary half-periods).
    assert_eq!(
      interpret("Round[WeierstrassHalfPeriods[{4.0, 0.0}], 10^-6]").unwrap(),
      "{1311029/1000000, (1311029*I)/1000000}"
    );
  }

  #[test]
  fn exact_stays_symbolic() {
    // Exact invariants are left unevaluated (matching wolframscript, which only
    // numericizes inexact input).
    assert_eq!(
      interpret("WeierstrassHalfPeriods[{8, 4}]").unwrap(),
      "WeierstrassHalfPeriods[{8, 4}]"
    );
  }
}

// Owen's T function T(h, a): numeric for inexact arguments (odd in a, even in
// h) with exact values at h = 0 and a = 0. Verified against wolframscript.
mod owen_t {
  use super::*;

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("Round[OwenT[1.0, 0.5], 10^-10]").unwrap(),
      "430646911/10000000000"
    );
    assert_eq!(
      interpret("Round[OwenT[0.5, 1.0], 10^-10]").unwrap(),
      "106671063/1000000000"
    );
    assert_eq!(
      interpret("Round[OwenT[1.0, 3.0], 10^-10]").unwrap(),
      "792995047/10000000000"
    );
  }

  #[test]
  fn symmetries() {
    // Odd in the second argument, even in the first.
    assert_eq!(
      interpret("OwenT[1.0, -0.5] == -OwenT[1.0, 0.5]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("OwenT[-1.0, 0.5] == OwenT[1.0, 0.5]").unwrap(),
      "True"
    );
  }

  #[test]
  fn exact_special_cases() {
    assert_eq!(interpret("OwenT[2.0, 0.0]").unwrap(), "0.");
    assert_eq!(interpret("OwenT[h, 0]").unwrap(), "0");
    assert_eq!(interpret("OwenT[0, a]").unwrap(), "ArcTan[a]/(2*Pi)");
    // Exact, non-special arguments stay symbolic.
    assert_eq!(interpret("OwenT[1, 1/2]").unwrap(), "OwenT[1, 1/2]");
  }
}
