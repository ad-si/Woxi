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
  fn bessel_j_at_symbolic_matching_zero_is_zero() {
    // Symbolic order matches → identity still holds.
    assert_eq!(interpret("BesselJ[n, BesselJZero[n, k]]").unwrap(), "0");
  }

  #[test]
  fn bessel_j_at_three_arg_zero_is_zero() {
    // BesselJZero[n, k, x0] is a different zero of J_n; identity still holds.
    assert_eq!(
      interpret("BesselJ[2, BesselJZero[2, 1, 4]]").unwrap(),
      "0"
    );
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
  #[test]
  fn weber_e() {
    assert_case(r#"WeberE[1.5, 3.5]"#, r#"-0.3972562592100308"#);
  }
  #[test]
  fn beta() {
    assert_case(r#"Beta[2, 3]"#, r#"1 / 12"#);
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
