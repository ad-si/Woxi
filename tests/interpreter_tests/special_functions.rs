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
