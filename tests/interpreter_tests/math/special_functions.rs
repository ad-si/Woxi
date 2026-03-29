use super::*;

mod gudermannian {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("Gudermannian[0]").unwrap(), "0");
  }

  #[test]
  fn zero_float() {
    assert_eq!(interpret("Gudermannian[0.]").unwrap(), "0.");
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("Gudermannian[4.2]").unwrap(),
      "1.5408074208608435"
    );
  }

  #[test]
  fn infinity() {
    assert_eq!(interpret("Gudermannian[Infinity]").unwrap(), "Pi/2");
  }

  #[test]
  fn negative_infinity() {
    assert_eq!(interpret("Gudermannian[-Infinity]").unwrap(), "-1/2*Pi");
  }

  #[test]
  fn complex_infinity() {
    // Gudermannian[ComplexInfinity] returns unevaluated (matching wolframscript)
    assert_eq!(
      interpret("Gudermannian[ComplexInfinity]").unwrap(),
      "Gudermannian[ComplexInfinity]"
    );
  }

  #[test]
  fn undefined() {
    assert_eq!(interpret("Gudermannian[Undefined]").unwrap(), "Undefined");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("Gudermannian[z]").unwrap(), "Gudermannian[z]");
  }
}

mod bessel_j {
  use super::*;

  #[test]
  fn zero_order_at_origin() {
    assert_eq!(interpret("BesselJ[0, 0]").unwrap(), "1");
  }

  #[test]
  fn nonzero_order_at_origin() {
    assert_eq!(interpret("BesselJ[1, 0]").unwrap(), "0");
    assert_eq!(interpret("BesselJ[2, 0]").unwrap(), "0");
  }

  #[test]
  fn zero_order_real_origin() {
    assert_eq!(interpret("BesselJ[0, 0.]").unwrap(), "1.");
  }

  #[test]
  fn numeric_zero_order() {
    let result: f64 = interpret("BesselJ[0, 5.2]").unwrap().parse().unwrap();
    assert!((result - (-0.11029043979098728)).abs() < 1e-10);
  }

  #[test]
  fn numeric_first_order() {
    let result: f64 = interpret("BesselJ[1, 3.0]").unwrap().parse().unwrap();
    assert!((result - 0.3390589585259365).abs() < 1e-10);
  }

  #[test]
  fn numeric_second_order() {
    let result: f64 = interpret("BesselJ[2, 1.5]").unwrap().parse().unwrap();
    assert!((result - 0.23208767214421472).abs() < 1e-10);
  }

  #[test]
  fn negative_order() {
    let result: f64 = interpret("BesselJ[-1, 3.0]").unwrap().parse().unwrap();
    assert!((result - (-0.3390589585259365)).abs() < 1e-10);
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    assert_eq!(interpret("BesselJ[0, x]").unwrap(), "BesselJ[0, x]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[BesselJ[0, 5]]").unwrap().parse().unwrap();
    assert!((result - (-0.17759677131433846)).abs() < 1e-10);
  }
}

mod bessel_i {
  use super::*;

  #[test]
  fn at_zero_order_zero() {
    assert_eq!(interpret("BesselI[0, 0]").unwrap(), "1");
  }

  #[test]
  fn at_zero_order_nonzero() {
    assert_eq!(interpret("BesselI[1, 0]").unwrap(), "0");
  }

  #[test]
  fn numeric_order_zero() {
    let result: f64 = interpret("BesselI[0, 1.5]").unwrap().parse().unwrap();
    assert!((result - 1.6467231897728904).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_one() {
    let result: f64 = interpret("BesselI[1, 2.0]").unwrap().parse().unwrap();
    assert!((result - 1.5906368546373288).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("BesselI[n, z]").unwrap(), "BesselI[n, z]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[BesselI[0, 1]]").unwrap().parse().unwrap();
    assert!((result - 1.2660658777520082).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_order_two() {
    let result: f64 = interpret("N[BesselI[2, 1]]").unwrap().parse().unwrap();
    assert!((result - 0.13574766976703828).abs() < 1e-10);
  }
}

mod bessel_k {
  use super::*;

  #[test]
  fn at_zero_order_zero() {
    assert_eq!(interpret("BesselK[0, 0]").unwrap(), "Infinity");
  }

  #[test]
  fn at_zero_order_nonzero() {
    assert_eq!(interpret("BesselK[1, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn numeric_order_zero() {
    let result: f64 = interpret("BesselK[0, 1.5]").unwrap().parse().unwrap();
    assert!((result - 0.2138055626475258).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_one() {
    let result: f64 = interpret("BesselK[1, 2.0]").unwrap().parse().unwrap();
    assert!((result - 0.13986588181652243).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("BesselK[n, z]").unwrap(), "BesselK[n, z]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[BesselK[0, 1]]").unwrap().parse().unwrap();
    assert!((result - 0.42102443824070834).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_order_two() {
    let result: f64 = interpret("N[BesselK[2, 1]]").unwrap().parse().unwrap();
    assert!((result - 1.6248388986351778).abs() < 1e-10);
  }
}

mod bessel_y {
  use super::*;

  #[test]
  fn at_zero_order_zero() {
    assert_eq!(interpret("BesselY[0, 0]").unwrap(), "-Infinity");
  }

  #[test]
  fn at_zero_order_one() {
    assert_eq!(interpret("BesselY[1, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn at_zero_order_two() {
    assert_eq!(interpret("BesselY[2, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn numeric_order_zero() {
    let result: f64 = interpret("BesselY[0, 2.5]").unwrap().parse().unwrap();
    assert!((result - 0.49807035961523194).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_one() {
    let result: f64 = interpret("BesselY[1, 2.5]").unwrap().parse().unwrap();
    assert!((result - 0.14591813796678565).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_two() {
    let result: f64 = interpret("BesselY[2, 3.0]").unwrap().parse().unwrap();
    assert!((result - (-0.16040039348492371)).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_three() {
    let result: f64 = interpret("BesselY[3, 5.0]").unwrap().parse().unwrap();
    assert!((result - 0.14626716269319232).abs() < 1e-10);
  }

  #[test]
  fn negative_order() {
    // BesselY[-1, z] = -BesselY[1, z]
    let result: f64 = interpret("BesselY[-1, 2.5]").unwrap().parse().unwrap();
    assert!((result - (-0.14591813796678565)).abs() < 1e-10);
  }

  #[test]
  fn numeric_small_arg() {
    let result: f64 = interpret("BesselY[0, 0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.4445187335067067)).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_arg() {
    let result: f64 = interpret("BesselY[0, 10.0]").unwrap().parse().unwrap();
    assert!((result - 0.055671167283599395).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[BesselY[0, 5/2]]").unwrap().parse().unwrap();
    assert!((result - 0.49807035961523194).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("BesselY[0, x]").unwrap(), "BesselY[0, x]");
  }
}

mod hypergeometric2f1 {
  use super::*;

  #[test]
  fn z_zero() {
    assert_eq!(interpret("Hypergeometric2F1[1, 2, 3, 0]").unwrap(), "1");
  }

  #[test]
  fn numeric_basic() {
    let result: f64 = interpret("Hypergeometric2F1[1, 2, 3, 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.5451774444795618).abs() < 1e-10);
  }

  #[test]
  fn numeric_log_related() {
    let result: f64 = interpret("Hypergeometric2F1[1, 1, 2, 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.38629436111989).abs() < 1e-10);
  }

  #[test]
  fn symbolic_simplification_1_2_3_x() {
    // 2F1(1, 2, 3, x) = -(2/x^2)*(x + Log[1-x])
    assert_eq!(
      interpret("Hypergeometric2F1[1, 2, 3, x]").unwrap(),
      "(-2*(x + Log[1 - x]))/x^2"
    );
  }

  #[test]
  fn symbolic_1_b_c() {
    // 2F1(1, b, c, z) with positive integer b < c, c > b+1
    assert_eq!(
      interpret("Hypergeometric2F1[1, 2, 4, x]").unwrap(),
      "(-3*(-2*x + x^2 - 2*Log[1 - x] + 2*x*Log[1 - x]))/x^3"
    );
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[Hypergeometric2F1[1, 2, 3, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.5451774444795618).abs() < 1e-10);
  }
}

mod hypergeometric_1f1 {
  use super::*;

  #[test]
  fn at_z_zero() {
    assert_eq!(interpret("Hypergeometric1F1[2, 3, 0]").unwrap(), "1");
  }

  #[test]
  fn exp_case() {
    // 1F1[1, 1, z] = e^z
    let result: f64 = interpret("Hypergeometric1F1[1, 1, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - std::f64::consts::E).abs() < 1e-10);
  }

  #[test]
  fn numeric_basic() {
    let result: f64 = interpret("Hypergeometric1F1[1, 2, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.7182818284590455).abs() < 1e-10);
  }

  #[test]
  fn numeric_higher() {
    let result: f64 = interpret("Hypergeometric1F1[2, 3, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 2.0).abs() < 1e-8);
  }

  #[test]
  fn numeric_negative_z() {
    let result: f64 = interpret("Hypergeometric1F1[0.5, 1.5, -1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.7468241328124272).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[Hypergeometric1F1[1, 2, 1]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.7182818284590455).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("Hypergeometric1F1[a, b, z]").unwrap(),
      "Hypergeometric1F1[a, b, z]"
    );
  }
}

mod hypergeometric_0f1 {
  use super::*;

  #[test]
  fn basic() {
    let result = interpret("Hypergeometric0F1[1.5, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.813430203923509).abs() < 1e-10);
  }

  #[test]
  fn zero_z() {
    assert_eq!(interpret("Hypergeometric0F1[a, 0]").unwrap(), "1");
  }

  #[test]
  fn unit_a() {
    let result = interpret("Hypergeometric0F1[1, 2.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 5.57162224874372).abs() < 1e-6);
  }

  #[test]
  fn small_z() {
    // For small z, 0F1(a; z) ≈ 1 + z/a
    let result = interpret("Hypergeometric0F1[2.0, 0.001]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.0005).abs() < 1e-4);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("Hypergeometric0F1[a, z]").unwrap(),
      "Hypergeometric0F1[a, z]"
    );
  }
}

mod hypergeometric_u {
  use super::*;

  #[test]
  fn numeric_non_integer_b() {
    let result: f64 = interpret("HypergeometricU[0.5, 0.5, 1.]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.7578721561413122).abs() < 1e-10);
  }

  #[test]
  fn numeric_integer_b() {
    let result: f64 = interpret("HypergeometricU[3, 2, 1.]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.10547895651520889).abs() < 1e-6);
  }

  #[test]
  fn numeric_b_equals_1() {
    let result: f64 = interpret("HypergeometricU[1, 1, 1.]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.596347362323194).abs() < 1e-6);
  }

  #[test]
  fn numeric_b_equals_3() {
    let result: f64 = interpret("HypergeometricU[2, 3, 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 4.0).abs() < 1e-6);
  }

  #[test]
  fn numeric_u_1_2_1() {
    let result: f64 = interpret("HypergeometricU[1, 2, 1.]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.0).abs() < 1e-6);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("HypergeometricU[a, b, c]").unwrap(),
      "HypergeometricU[a, b, c]"
    );
  }

  #[test]
  fn symbolic_with_variable() {
    // HypergeometricU[a, a+1, z] = z^(-a) (matching wolframscript)
    assert_eq!(interpret("HypergeometricU[1, 2, x]").unwrap(), "x^(-1)");
  }
}

mod elliptic_k {
  use super::*;

  #[test]
  fn zero_arg() {
    assert_eq!(interpret("EllipticK[0]").unwrap(), "Pi/2");
  }

  #[test]
  fn one_arg() {
    assert_eq!(interpret("EllipticK[1]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn real_one_arg() {
    assert_eq!(interpret("EllipticK[1.0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn real_zero_arg() {
    let result: f64 = interpret("EllipticK[0.0]").unwrap().parse().unwrap();
    assert!((result - 1.5707963267948966).abs() < 1e-10);
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("EllipticK[0.5]").unwrap().parse().unwrap();
    assert!((result - 1.8540746773013717).abs() < 1e-10);
  }

  #[test]
  fn numeric_near_one() {
    let result: f64 = interpret("EllipticK[0.9]").unwrap().parse().unwrap();
    assert!((result - 2.578092113348173).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result: f64 = interpret("EllipticK[-0.5]").unwrap().parse().unwrap();
    assert!((result - 1.415737208425956).abs() < 1e-10);
  }

  #[test]
  fn symbolic_exact() {
    assert_eq!(interpret("EllipticK[1/2]").unwrap(), "EllipticK[1/2]");
  }

  #[test]
  fn symbolic_variable() {
    assert_eq!(interpret("EllipticK[x]").unwrap(), "EllipticK[x]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[EllipticK[1/2]]").unwrap().parse().unwrap();
    assert!((result - 1.8540746773013717).abs() < 1e-10);
  }
}

mod elliptic_e {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("EllipticE[0]").unwrap(), "Pi/2");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("EllipticE[1]").unwrap(), "1");
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("EllipticE[0.5]").unwrap().parse().unwrap();
    assert!((result - 1.3506438810476753).abs() < 1e-10);
  }

  #[test]
  fn numeric_quarter() {
    let result: f64 = interpret("EllipticE[0.25]").unwrap().parse().unwrap();
    assert!((result - 1.4674622093394272).abs() < 1e-10);
  }

  #[test]
  fn numeric_point_nine() {
    let result: f64 = interpret("EllipticE[0.9]").unwrap().parse().unwrap();
    assert!((result - 1.1047747327040733).abs() < 1e-10);
  }

  #[test]
  fn numeric_one_real() {
    let result: f64 = interpret("EllipticE[1.0]").unwrap().parse().unwrap();
    assert!((result - 1.0).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[EllipticE[1/2]]").unwrap().parse().unwrap();
    assert!((result - 1.3506438810476753).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("EllipticE[x]").unwrap(), "EllipticE[x]");
  }
}

mod elliptic_f {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("EllipticF[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn numeric_one() {
    let result: f64 =
      interpret("EllipticF[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 1.0832167728451683).abs() < 1e-8);
  }

  #[test]
  fn numeric_half() {
    let result: f64 =
      interpret("EllipticF[0.5, 0.5]").unwrap().parse().unwrap();
    assert!((result - 0.5104671356280047).abs() < 1e-8);
  }

  #[test]
  fn numeric_high_m() {
    let result: f64 =
      interpret("EllipticF[1.0, 0.9]").unwrap().parse().unwrap();
    assert!((result - 1.1885008994681587).abs() < 1e-8);
  }

  #[test]
  fn at_pi_half_equals_k() {
    // EllipticF[Pi/2, m] = EllipticK[m]
    let result: f64 = interpret("EllipticF[N[Pi/2], 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    let k: f64 = interpret("EllipticK[0.5]").unwrap().parse().unwrap();
    assert!((result - k).abs() < 1e-8);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[EllipticF[1, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 1.0832167728451683).abs() < 1e-8);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("EllipticF[phi, m]").unwrap(), "EllipticF[phi, m]");
  }
}

mod elliptic_pi {
  use super::*;

  #[test]
  fn complete_basic() {
    let result = interpret("EllipticPi[0.5, 0.3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.461255352272422).abs() < 1e-6);
  }

  #[test]
  fn complete_zero_n_equals_elliptic_k() {
    // EllipticPi[0, m] = EllipticK[m]
    let pi_result = interpret("EllipticPi[0, 0.3]").unwrap();
    let k_result = interpret("EllipticK[0.3]").unwrap();
    assert_eq!(pi_result, k_result);
  }

  #[test]
  fn incomplete() {
    let result = interpret("EllipticPi[0.3, Pi/4, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8777133738264999).abs() < 1e-6);
  }

  #[test]
  fn incomplete_zero_phi() {
    assert_eq!(interpret("EllipticPi[0.3, 0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("EllipticPi[n, m]").unwrap(), "EllipticPi[n, m]");
  }
}

mod elliptic_nome_q {
  use super::*;

  #[test]
  fn zero_arg() {
    assert_eq!(interpret("EllipticNomeQ[0]").unwrap(), "0");
  }

  #[test]
  fn one_arg() {
    assert_eq!(interpret("EllipticNomeQ[1]").unwrap(), "1");
  }

  #[test]
  fn half_symbolic() {
    assert_eq!(interpret("EllipticNomeQ[1/2]").unwrap(), "E^(-Pi)");
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("EllipticNomeQ[0.5]").unwrap().parse().unwrap();
    assert!((result - 0.04321391826377226).abs() < 1e-10);
  }

  #[test]
  fn numeric_small() {
    let result: f64 = interpret("EllipticNomeQ[0.3]").unwrap().parse().unwrap();
    assert!((result - 0.02227743615715351).abs() < 1e-10);
  }

  #[test]
  fn numeric_near_one() {
    let result: f64 =
      interpret("EllipticNomeQ[0.99]").unwrap().parse().unwrap();
    assert!((result - 0.26219626791770934).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("EllipticNomeQ[m]").unwrap(), "EllipticNomeQ[m]");
  }
}

mod elliptic_theta {
  use super::*;

  #[test]
  fn theta1_at_q_zero() {
    assert_eq!(interpret("EllipticTheta[1, z, 0]").unwrap(), "0");
  }

  #[test]
  fn theta2_at_q_zero() {
    assert_eq!(interpret("EllipticTheta[2, z, 0]").unwrap(), "0");
  }

  #[test]
  fn theta3_at_q_zero() {
    assert_eq!(interpret("EllipticTheta[3, z, 0]").unwrap(), "1");
  }

  #[test]
  fn theta4_at_q_zero() {
    assert_eq!(interpret("EllipticTheta[4, z, 0]").unwrap(), "1");
  }

  #[test]
  fn theta1_numeric() {
    let result: f64 = interpret("EllipticTheta[1, 0.5, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.5279836054564474).abs() < 1e-10);
  }

  #[test]
  fn theta2_numeric() {
    let result: f64 = interpret("EllipticTheta[2, 0.5, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.9877965496358908).abs() < 1e-10);
  }

  #[test]
  fn theta3_numeric() {
    let result: f64 = interpret("EllipticTheta[3, 0.5, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.1079772298263333).abs() < 1e-10);
  }

  #[test]
  fn theta4_numeric() {
    let result: f64 = interpret("EllipticTheta[4, 0.5, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.8918563114390474).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("EllipticTheta[3, z, q]").unwrap(),
      "EllipticTheta[3, z, q]"
    );
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[EllipticTheta[3, 0, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 2.128936827211877).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_theta1() {
    let result: f64 = interpret("N[EllipticTheta[1, 1, 1/3]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.2529300675643478).abs() < 1e-10);
  }
}

mod polylog {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("PolyLog[2, 0]").unwrap(), "0");
    assert_eq!(interpret("PolyLog[3, 0]").unwrap(), "0");
  }

  #[test]
  fn at_one_gives_zeta() {
    assert_eq!(interpret("PolyLog[2, 1]").unwrap(), "Pi^2/6");
    assert_eq!(interpret("PolyLog[3, 1]").unwrap(), "Zeta[3]");
    assert_eq!(interpret("PolyLog[4, 1]").unwrap(), "Pi^4/90");
  }

  #[test]
  fn at_neg_one() {
    assert_eq!(interpret("PolyLog[2, -1]").unwrap(), "-1/12*Pi^2");
    assert_eq!(interpret("PolyLog[3, -1]").unwrap(), "(-3*Zeta[3])/4");
    assert_eq!(interpret("PolyLog[4, -1]").unwrap(), "(-7*Pi^4)/720");
  }

  #[test]
  fn s_one_symbolic() {
    assert_eq!(interpret("PolyLog[1, x]").unwrap(), "-Log[1 - x]");
  }

  #[test]
  fn s_one_rational() {
    assert_eq!(interpret("PolyLog[1, 1/2]").unwrap(), "Log[2]");
  }

  #[test]
  fn s_zero() {
    assert_eq!(interpret("PolyLog[0, x]").unwrap(), "x/(1 - x)");
    assert_eq!(interpret("PolyLog[0, 0]").unwrap(), "0");
    assert_eq!(interpret("PolyLog[0, 1]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn negative_s() {
    assert_eq!(interpret("PolyLog[-1, x]").unwrap(), "x/(1 - x)^2");
    assert_eq!(interpret("PolyLog[-2, x]").unwrap(), "(x + x^2)/(1 - x)^3");
    assert_eq!(
      interpret("PolyLog[-3, x]").unwrap(),
      "(x + 4*x^2 + x^3)/(1 - x)^4"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("PolyLog[2, x]").unwrap(), "PolyLog[2, x]");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("PolyLog[2, 0.5]").unwrap().parse().unwrap();
    assert!((result - 0.5822405264650125).abs() < 1e-8);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[PolyLog[3, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 0.5372131936080402).abs() < 1e-8);
  }
}

mod legendre_p {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("LegendreP[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("LegendreP[1, x]").unwrap(), "x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("LegendreP[2, x]").unwrap(), "(-1 + 3*x^2)/2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(interpret("LegendreP[3, x]").unwrap(), "(-3*x + 5*x^3)/2");
  }

  #[test]
  fn degree_four() {
    assert_eq!(
      interpret("LegendreP[4, x]").unwrap(),
      "(3 - 30*x^2 + 35*x^4)/8"
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("LegendreP[2, 0]").unwrap(), "-1/2");
    assert_eq!(interpret("LegendreP[3, 0]").unwrap(), "0");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("LegendreP[2, 1]").unwrap(), "1");
    assert_eq!(interpret("LegendreP[3, 1]").unwrap(), "1");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("LegendreP[2, 0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.125)).abs() < 1e-10);
  }

  #[test]
  fn at_rational() {
    assert_eq!(interpret("LegendreP[3, 1/2]").unwrap(), "-7/16");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[LegendreP[3, 1/2]]").unwrap().parse().unwrap();
    assert!((result - (-0.4375)).abs() < 1e-10);
  }

  #[test]
  fn high_degree() {
    assert_eq!(
      interpret("LegendreP[10, x]").unwrap(),
      "(-63 + 3465*x^2 - 30030*x^4 + 90090*x^6 - 109395*x^8 + 46189*x^10)/256"
    );
  }
}

mod legendre_q {
  use super::*;

  #[test]
  fn numeric_order_zero() {
    let result: f64 = interpret("LegendreQ[0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 0.5493061443340543).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_one() {
    let result: f64 = interpret("LegendreQ[1, 0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.7253469278329724)).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_two() {
    let result: f64 = interpret("LegendreQ[2, 0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.8186632680417569)).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_three() {
    let result: f64 = interpret("LegendreQ[3, 0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.19865477147948235)).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[LegendreQ[1, 1/2]]").unwrap().parse().unwrap();
    assert!((result - (-0.7253469278329724)).abs() < 1e-8);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("LegendreQ[n, x]").unwrap(), "LegendreQ[n, x]");
  }
}

mod polygamma {
  use super::*;

  #[test]
  fn digamma_one() {
    assert_eq!(interpret("PolyGamma[1]").unwrap(), "-EulerGamma");
  }

  #[test]
  fn digamma_two() {
    assert_eq!(interpret("PolyGamma[2]").unwrap(), "1 - EulerGamma");
  }

  #[test]
  fn digamma_three() {
    assert_eq!(interpret("PolyGamma[3]").unwrap(), "3/2 - EulerGamma");
  }

  #[test]
  fn digamma_five() {
    assert_eq!(interpret("PolyGamma[0, 5]").unwrap(), "25/12 - EulerGamma");
  }

  #[test]
  fn trigamma_one() {
    assert_eq!(interpret("PolyGamma[1, 1]").unwrap(), "Pi^2/6");
  }

  #[test]
  fn trigamma_two() {
    assert_eq!(interpret("PolyGamma[1, 2]").unwrap(), "-1 + Pi^2/6");
  }

  #[test]
  fn trigamma_three() {
    assert_eq!(interpret("PolyGamma[1, 3]").unwrap(), "-5/4 + Pi^2/6");
  }

  #[test]
  fn tetragamma_unevaluated() {
    assert_eq!(interpret("PolyGamma[2, 1]").unwrap(), "PolyGamma[2, 1]");
  }

  #[test]
  fn polygamma_3_1() {
    assert_eq!(interpret("PolyGamma[3, 1]").unwrap(), "Pi^4/15");
  }

  #[test]
  fn polygamma_5_1() {
    assert_eq!(interpret("PolyGamma[5, 1]").unwrap(), "(8*Pi^6)/63");
  }

  #[test]
  fn polygamma_3_3_factored() {
    assert_eq!(
      interpret("PolyGamma[3, 3]").unwrap(),
      "6*(-17/16 + Pi^4/90)"
    );
  }

  #[test]
  fn pole_at_zero() {
    assert_eq!(interpret("PolyGamma[0, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[0, -1]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[1, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("PolyGamma[x]").unwrap(), "PolyGamma[0, x]");
    assert_eq!(interpret("PolyGamma[1/2]").unwrap(), "PolyGamma[0, 1/2]");
  }

  #[test]
  fn numeric_digamma() {
    let result: f64 = interpret("PolyGamma[1.0]").unwrap().parse().unwrap();
    assert!((result - (-0.5772156649015329)).abs() < 1e-10);
  }

  #[test]
  fn numeric_trigamma() {
    let result: f64 = interpret("PolyGamma[1, 1.0]").unwrap().parse().unwrap();
    assert!((result - 1.6449340668482264).abs() < 1e-8);
  }

  #[test]
  fn n_evaluates_polygamma() {
    let result: f64 = interpret("N[PolyGamma[2, 1]]").unwrap().parse().unwrap();
    assert!((result - (-2.4041138063191885)).abs() < 1e-8);
  }
}

mod zeta {
  use super::*;

  #[test]
  fn pole_at_one() {
    assert_eq!(interpret("Zeta[1]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("Zeta[0]").unwrap(), "-1/2");
  }

  #[test]
  fn positive_even_2() {
    assert_eq!(interpret("Zeta[2]").unwrap(), "Pi^2/6");
  }

  #[test]
  fn positive_even_4() {
    assert_eq!(interpret("Zeta[4]").unwrap(), "Pi^4/90");
  }

  #[test]
  fn positive_even_6() {
    assert_eq!(interpret("Zeta[6]").unwrap(), "Pi^6/945");
  }

  #[test]
  fn positive_even_12() {
    assert_eq!(interpret("Zeta[12]").unwrap(), "(691*Pi^12)/638512875");
  }

  #[test]
  fn positive_even_20() {
    assert_eq!(
      interpret("Zeta[20]").unwrap(),
      "(174611*Pi^20)/1531329465290625"
    );
  }

  #[test]
  fn positive_odd_unevaluated() {
    assert_eq!(interpret("Zeta[3]").unwrap(), "Zeta[3]");
    assert_eq!(interpret("Zeta[5]").unwrap(), "Zeta[5]");
  }

  #[test]
  fn negative_even_trivial_zeros() {
    assert_eq!(interpret("Zeta[-2]").unwrap(), "0");
    assert_eq!(interpret("Zeta[-4]").unwrap(), "0");
    assert_eq!(interpret("Zeta[-6]").unwrap(), "0");
  }

  #[test]
  fn negative_odd() {
    assert_eq!(interpret("Zeta[-1]").unwrap(), "-1/12");
    assert_eq!(interpret("Zeta[-3]").unwrap(), "1/120");
    assert_eq!(interpret("Zeta[-5]").unwrap(), "-1/252");
    assert_eq!(interpret("Zeta[-7]").unwrap(), "1/240");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("Zeta[x]").unwrap(), "Zeta[x]");
    assert_eq!(interpret("Zeta[1/2]").unwrap(), "Zeta[1/2]");
  }

  #[test]
  fn numeric_real_positive() {
    let result: f64 = interpret("Zeta[2.0]").unwrap().parse().unwrap();
    assert!((result - 1.6449340668482264).abs() < 1e-10);
  }

  #[test]
  fn numeric_real_half() {
    let result: f64 = interpret("Zeta[0.5]").unwrap().parse().unwrap();
    assert!((result - (-1.460354508809588)).abs() < 1e-6);
  }

  #[test]
  fn n_evaluates_zeta() {
    let result: f64 = interpret("N[Zeta[2]]").unwrap().parse().unwrap();
    assert!((result - 1.6449340668482264).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_zeta_3() {
    let result: f64 = interpret("N[Zeta[3]]").unwrap().parse().unwrap();
    assert!((result - 1.2020569031595942).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_real() {
    let result: f64 = interpret("Zeta[-0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.20788622497735454)).abs() < 1e-6);
  }

  #[test]
  fn complex_on_critical_line() {
    // Zeta[1/2 + 13 I] ≈ 0.443... - 0.655...*I
    let result = interpret("N[Zeta[1/2 + 13 I]]").unwrap();
    assert!(
      result.contains("0.44300478250536")
        && result.contains("0.65548309832117"),
      "Expected Zeta[1/2+13I] ≈ 0.443...-0.655...*I, got: {}",
      result
    );
  }

  #[test]
  fn complex_positive_real_part() {
    // Zeta[2 + 3 I] ≈ 0.798... - 0.114...*I
    let result = interpret("N[Zeta[2 + 3 I]]").unwrap();
    assert!(
      result.contains("0.798021985146275")
        && result.contains("0.113744308052938"),
      "Expected Zeta[2+3I] ≈ 0.798...-0.114...*I, got: {}",
      result
    );
  }

  #[test]
  fn complex_with_precision() {
    // N[Zeta[1/2 + 13 I], 40] should evaluate (at machine precision)
    let result = interpret("N[Zeta[1/2 + 13 I], 40]").unwrap();
    assert!(
      result.contains("0.443") && result.contains("I"),
      "Expected numeric complex result, got: {}",
      result
    );
  }
}

mod jacobi_dn {
  use super::*;

  #[test]
  fn at_zero_u() {
    assert_eq!(interpret("JacobiDN[0, m]").unwrap(), "1");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiDN[u, 0]").unwrap(), "1");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiDN[u, 1]").unwrap(), "Sech[u]");
  }

  #[test]
  fn even_symmetry() {
    assert_eq!(interpret("JacobiDN[-u, m]").unwrap(), "JacobiDN[u, m]");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("JacobiDN[4., 2/3]").unwrap().parse().unwrap();
    assert!((result - 0.9988832842546814).abs() < 1e-10);
  }

  #[test]
  fn numeric_real_2() {
    let result: f64 = interpret("JacobiDN[0.5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 0.9656789647459513).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiDN[u, m]").unwrap(), "JacobiDN[u, m]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[JacobiDN[1, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 0.8231610016315964).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_2() {
    let result: f64 =
      interpret("N[JacobiDN[2, 1/3]]").unwrap().parse().unwrap();
    assert!((result - 0.8259983048005796).abs() < 1e-10);
  }
}

mod jacobi_sn {
  use super::*;

  #[test]
  fn at_zero_u() {
    assert_eq!(interpret("JacobiSN[0, m]").unwrap(), "0");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiSN[u, 0]").unwrap(), "Sin[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiSN[u, 1]").unwrap(), "Tanh[u]");
  }

  #[test]
  fn odd_symmetry() {
    assert_eq!(interpret("JacobiSN[-u, m]").unwrap(), "-JacobiSN[u, m]");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("JacobiSN[0.5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 0.47421562271182044).abs() < 1e-10);
  }

  #[test]
  fn numeric_real_2() {
    let result: f64 = interpret("JacobiSN[4., 2/3]").unwrap().parse().unwrap();
    assert!((result - 0.05786429516439241).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiSN[u, m]").unwrap(), "JacobiSN[u, m]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[JacobiSN[1, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 0.8030018248956442).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_2() {
    let result: f64 =
      interpret("N[JacobiSN[2, 1/3]]").unwrap().parse().unwrap();
    assert!((result - 0.9763095827654809).abs() < 1e-10);
  }
}

mod jacobi_cn {
  use super::*;

  #[test]
  fn at_zero_u() {
    assert_eq!(interpret("JacobiCN[0, m]").unwrap(), "1");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiCN[u, 0]").unwrap(), "Cos[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiCN[u, 1]").unwrap(), "Sech[u]");
  }

  #[test]
  fn even_symmetry() {
    assert_eq!(interpret("JacobiCN[-u, m]").unwrap(), "JacobiCN[u, m]");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("JacobiCN[0.5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 0.8804087364264626).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiCN[u, m]").unwrap(), "JacobiCN[u, m]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[JacobiCN[1, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 0.5959765676721407).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_2() {
    let result: f64 =
      interpret("N[JacobiCN[2, 1/3]]").unwrap().parse().unwrap();
    assert!((result - (-0.21637836906745786)).abs() < 1e-10);
  }
}

mod jacobi_sc {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiSC[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiSC[u, 0]").unwrap(), "Tan[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiSC[u, 1]").unwrap(), "Sinh[u]");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiSC[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 1.3473714713854195).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiSC[u, m]").unwrap(), "JacobiSC[u, m]");
  }
}

mod jacobi_dc {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiDC[0, 0.5]").unwrap(), "1.");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiDC[u, 0]").unwrap(), "Sec[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiDC[u, 1]").unwrap(), "1");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiDC[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 1.3811969233066135).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiDC[u, m]").unwrap(), "JacobiDC[u, m]");
  }
}

mod jacobi_cd {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiCD[0, 0.5]").unwrap(), "1.");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiCD[u, 0]").unwrap(), "Cos[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiCD[u, 1]").unwrap(), "1");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiCD[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 0.7240097216593704).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiCD[u, m]").unwrap(), "JacobiCD[u, m]");
  }
}

mod jacobi_sd {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiSD[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiSD[u, 0]").unwrap(), "Sin[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiSD[u, 1]").unwrap(), "Sinh[u]");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiSD[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 0.9755100439695339).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiSD[u, m]").unwrap(), "JacobiSD[u, m]");
  }
}

mod jacobi_cs {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiCS[0, 0.5]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiCS[u, 0]").unwrap(), "Cot[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiCS[u, 1]").unwrap(), "Csch[u]");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiCS[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 0.7421858197515206).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiCS[u, m]").unwrap(), "JacobiCS[u, m]");
  }
}

mod jacobi_ds {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiDS[0, 0.5]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiDS[u, 0]").unwrap(), "Csc[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiDS[u, 1]").unwrap(), "Csch[u]");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiDS[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 1.025104770762597).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiDS[u, m]").unwrap(), "JacobiDS[u, m]");
  }
}

mod jacobi_ns {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiNS[0, 0.5]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiNS[u, 0]").unwrap(), "Csc[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiNS[u, 1]").unwrap(), "Coth[u]");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiNS[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 1.245327182326089).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiNS[u, m]").unwrap(), "JacobiNS[u, m]");
  }
}

mod jacobi_nd {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiND[0, 0.5]").unwrap(), "1.");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiND[u, 0]").unwrap(), "1");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiND[u, 1]").unwrap(), "Cosh[u]");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiND[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 1.2148291743873787).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiND[u, m]").unwrap(), "JacobiND[u, m]");
  }
}

mod jacobi_nc {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("JacobiNC[0, 0.5]").unwrap(), "1.");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiNC[u, 0]").unwrap(), "Sec[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiNC[u, 1]").unwrap(), "Cosh[u]");
  }

  #[test]
  fn numeric() {
    let result: f64 = interpret("JacobiNC[1.0, 0.5]").unwrap().parse().unwrap();
    assert!((result - 1.6779183180069608).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiNC[u, m]").unwrap(), "JacobiNC[u, m]");
  }
}

mod jacobi_amplitude {
  use super::*;

  #[test]
  fn basic() {
    let result = interpret("JacobiAmplitude[1.0, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.9323150798838539).abs() < 1e-10);
  }

  #[test]
  fn zero_u() {
    assert_eq!(interpret("JacobiAmplitude[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn zero_m() {
    // am(u, 0) = u
    let result = interpret("JacobiAmplitude[0.5, 0.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.5).abs() < 1e-10);
  }

  #[test]
  fn inverse_of_elliptic_f() {
    let am = interpret("JacobiAmplitude[1.5, 0.3]").unwrap();
    let check = interpret(&format!("EllipticF[{}, 0.3]", am)).unwrap();
    let val: f64 = check.parse().unwrap();
    assert!((val - 1.5).abs() < 1e-6);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("JacobiAmplitude[u, m]").unwrap(),
      "JacobiAmplitude[u, m]"
    );
  }
}

mod jacobi_p {
  use super::*;

  #[test]
  fn degree_zero() {
    // JacobiP[0, a, b, x] = 1 for any a, b, x
    assert_eq!(interpret("JacobiP[0, 1, 2, 0.5]").unwrap(), "1.");
    assert_eq!(interpret("JacobiP[0, 3, 5, -0.7]").unwrap(), "1.");
  }

  #[test]
  fn degree_one() {
    // P_1^{(1,2)}(0.5) = (1 - 2)/2 + (1 + 2 + 2)*0.5/2 = -0.5 + 1.25 = 0.75
    assert_eq!(interpret("JacobiP[1, 1, 2, 0.5]").unwrap(), "0.75");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("JacobiP[2, 1, 2, 0.5]").unwrap(), "-0.1875");
  }

  #[test]
  fn matches_legendre_p() {
    // JacobiP[n, 0, 0, x] = LegendreP[n, x]
    let j2 = interpret("JacobiP[2, 0, 0, 0.5]").unwrap();
    let l2 = interpret("LegendreP[2, 0.5]").unwrap();
    assert_eq!(j2, l2);

    let j3 = interpret("JacobiP[3, 0, 0, 0.5]").unwrap();
    let l3 = interpret("LegendreP[3, 0.5]").unwrap();
    assert_eq!(j3, l3);
  }

  #[test]
  fn higher_degree() {
    let result = interpret("JacobiP[5, 2, 3, -0.3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-0.31473093749999986)).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("JacobiP[x, 1, 2, 0.5]").unwrap(),
      "JacobiP[x, 1, 2, 0.5]"
    );
  }

  #[test]
  fn at_zero() {
    // JacobiP[n, a, b, 0] should work
    let result = interpret("JacobiP[3, 1, 1, 0]").unwrap();
    let val: f64 = result.parse().unwrap();
    // For a=b (Gegenbauer case), odd n at x=0 should be 0
    assert!(val.abs() < 1e-12);
  }
}

mod exp_integral_ei {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("ExpIntegralEi[0]").unwrap(), "-Infinity");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("ExpIntegralEi[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn at_neg_infinity() {
    assert_eq!(interpret("ExpIntegralEi[-Infinity]").unwrap(), "0");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("ExpIntegralEi[x]").unwrap(), "ExpIntegralEi[x]");
  }

  #[test]
  fn numeric_positive() {
    let result: f64 = interpret("ExpIntegralEi[0.5]").unwrap().parse().unwrap();
    assert!((result - 0.4542199048631736).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result: f64 =
      interpret("ExpIntegralEi[-0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.5597735947761607)).abs() < 1e-10);
  }

  #[test]
  fn numeric_larger() {
    let result: f64 = interpret("ExpIntegralEi[2.0]").unwrap().parse().unwrap();
    assert!((result - 4.95423435600189).abs() < 1e-8);
  }

  #[test]
  fn n_evaluates_positive() {
    let result: f64 =
      interpret("N[ExpIntegralEi[1]]").unwrap().parse().unwrap();
    assert!((result - 1.8951178163559368).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_negative() {
    let result: f64 =
      interpret("N[ExpIntegralEi[-1]]").unwrap().parse().unwrap();
    assert!((result - (-0.21938393439552026)).abs() < 1e-10);
  }

  #[test]
  fn n_complex_evaluates() {
    // N[ExpIntegralEi[1+I], 20] should produce a complex result
    let result = interpret("N[ExpIntegralEi[1+I],20]").unwrap();
    // Check the real part starts correctly (first 20 digits should match)
    assert!(result.contains("1.7646259855638540684267381613"));
    // Check the imaginary part starts correctly
    assert!(result.contains("2.387769851510522419262792089103"));
    // Check it has I
    assert!(result.contains("*I"));
  }

  #[test]
  fn series_at_zero_positive_assumption() {
    assert_eq!(
      interpret("Series[ExpIntegralEi[x], {x, 0, 5}, Assumptions -> x > 0]")
        .unwrap(),
      "SeriesData[x, 0, {EulerGamma + Log[x], 1, 1/4, 1/18, 1/96, 1/600}, 0, 6, 1]"
    );
  }

  #[test]
  fn series_at_zero_negative_assumption() {
    assert_eq!(
      interpret("Series[ExpIntegralEi[x], {x, 0, 5}, Assumptions -> x < 0]")
        .unwrap(),
      "SeriesData[x, 0, {EulerGamma + Log[-x], 1, 1/4, 1/18, 1/96, 1/600}, 0, 6, 1]"
    );
  }

  #[test]
  fn series_at_infinity() {
    assert_eq!(
      interpret(r#"Series[ExpIntegralEi[x], {x, \[Infinity], 6}] // Normal"#)
        .unwrap(),
      "E^x*(120/x^6 + 24/x^5 + 6/x^4 + 2/x^3 + x^(-2) + x^(-1)) + (Log[-x^(-1)] - Log[-x] + 2*Log[x])/2"
    );
  }
}

mod exp_integral_e {
  use super::*;

  #[test]
  fn order_one_at_zero() {
    assert_eq!(interpret("ExpIntegralE[1, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn order_two_at_zero() {
    assert_eq!(interpret("ExpIntegralE[2, 0]").unwrap(), "1");
  }

  #[test]
  fn order_three_at_zero() {
    assert_eq!(interpret("ExpIntegralE[3, 0]").unwrap(), "1/2");
  }

  #[test]
  fn numeric_order_one() {
    let result: f64 =
      interpret("ExpIntegralE[1, 1.0]").unwrap().parse().unwrap();
    assert!((result - 0.21938393439552026).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_two() {
    let result: f64 =
      interpret("ExpIntegralE[2, 1.0]").unwrap().parse().unwrap();
    assert!((result - 0.14849550677592208).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_zero() {
    let result: f64 =
      interpret("ExpIntegralE[0, 1.0]").unwrap().parse().unwrap();
    assert!((result - 0.36787944117144233).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_three() {
    let result: f64 =
      interpret("ExpIntegralE[3, 1.0]").unwrap().parse().unwrap();
    assert!((result - 0.10969196719776013).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[ExpIntegralE[1, 1]]").unwrap().parse().unwrap();
    assert!((result - 0.21938393439552026).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("ExpIntegralE[n, z]").unwrap(),
      "ExpIntegralE[n, z]"
    );
  }
}

mod cos_integral {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("CosIntegral[0]").unwrap(), "-Infinity");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("CosIntegral[Infinity]").unwrap(), "0");
  }

  #[test]
  fn at_neg_infinity() {
    assert_eq!(interpret("CosIntegral[-Infinity]").unwrap(), "I*Pi");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("CosIntegral[x]").unwrap(), "CosIntegral[x]");
  }

  #[test]
  fn numeric_one() {
    let result: f64 = interpret("CosIntegral[1.0]").unwrap().parse().unwrap();
    assert!((result - 0.3374039229009681).abs() < 1e-10);
  }

  #[test]
  fn numeric_two() {
    let result: f64 = interpret("CosIntegral[2.0]").unwrap().parse().unwrap();
    assert!((result - 0.422980828774865).abs() < 1e-10);
  }

  #[test]
  fn numeric_small() {
    let result: f64 = interpret("CosIntegral[0.1]").unwrap().parse().unwrap();
    assert!((result - (-1.7278683866572964)).abs() < 1e-10);
  }
}

mod sin_integral {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("SinIntegral[0]").unwrap(), "0");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("SinIntegral[Infinity]").unwrap(), "Pi/2");
  }

  #[test]
  fn at_neg_infinity() {
    assert_eq!(interpret("SinIntegral[-Infinity]").unwrap(), "-1/2*Pi");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("SinIntegral[x]").unwrap(), "SinIntegral[x]");
  }

  #[test]
  fn numeric_one() {
    let result: f64 = interpret("SinIntegral[1.0]").unwrap().parse().unwrap();
    assert!((result - 0.946083070367183).abs() < 1e-10);
  }

  #[test]
  fn numeric_two() {
    let result: f64 = interpret("SinIntegral[2.0]").unwrap().parse().unwrap();
    assert!((result - 1.6054129768026948).abs() < 1e-10);
  }
}

mod chebyshev_t {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("ChebyshevT[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("ChebyshevT[1, x]").unwrap(), "x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("ChebyshevT[2, x]").unwrap(), "-1 + 2*x^2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(interpret("ChebyshevT[3, x]").unwrap(), "-3*x + 4*x^3");
  }

  #[test]
  fn degree_four() {
    assert_eq!(interpret("ChebyshevT[4, x]").unwrap(), "1 - 8*x^2 + 8*x^4");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("ChebyshevT[5, 1]").unwrap(), "1");
  }

  #[test]
  fn at_rational() {
    assert_eq!(interpret("ChebyshevT[3, 1/2]").unwrap(), "-1");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("ChebyshevT[5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 0.9988800000000002).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("ChebyshevT[n, x]").unwrap(), "ChebyshevT[n, x]");
  }

  #[test]
  fn at_rational_exact() {
    assert_eq!(interpret("ChebyshevT[4, 1/3]").unwrap(), "17/81");
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("ChebyshevT[4, 0]").unwrap(), "1");
  }
}

mod chebyshev_u {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("ChebyshevU[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("ChebyshevU[1, x]").unwrap(), "2*x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("ChebyshevU[2, x]").unwrap(), "-1 + 4*x^2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(interpret("ChebyshevU[3, x]").unwrap(), "-4*x + 8*x^3");
  }

  #[test]
  fn degree_four() {
    assert_eq!(
      interpret("ChebyshevU[4, x]").unwrap(),
      "1 - 12*x^2 + 16*x^4"
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("ChebyshevU[3, 0]").unwrap(), "0");
    assert_eq!(interpret("ChebyshevU[4, 0]").unwrap(), "1");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("ChebyshevU[3, 1]").unwrap(), "4");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("ChebyshevU[5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 1.01376).abs() < 1e-4);
  }

  #[test]
  fn rational_arg() {
    let result = interpret("N[ChebyshevU[3, 1/2]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-1.0)).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[ChebyshevU[2, 3/4]]").unwrap().parse().unwrap();
    // U_2(3/4) = -1 + 4*(9/16) = -1 + 9/4 = 5/4 = 1.25
    assert!((result - 1.25).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("ChebyshevU[n, x]").unwrap(), "ChebyshevU[n, x]");
  }
}

mod laguerre_l {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("LaguerreL[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("LaguerreL[1, x]").unwrap(), "1 - x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("LaguerreL[2, x]").unwrap(), "(2 - 4*x + x^2)/2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(
      interpret("LaguerreL[3, x]").unwrap(),
      "(6 - 18*x + 9*x^2 - x^3)/6"
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("LaguerreL[4, 0]").unwrap(), "1");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("LaguerreL[3, 1]").unwrap(), "-2/3");
  }

  #[test]
  fn at_rational() {
    assert_eq!(interpret("LaguerreL[3, 1/2]").unwrap(), "-7/48");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("LaguerreL[5, 0.3]").unwrap().parse().unwrap();
    assert!((result - (-0.09333274999999995)).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("LaguerreL[n, x]").unwrap(), "LaguerreL[n, x]");
  }
}

mod beta_fn {
  use super::*;

  #[test]
  fn both_one() {
    assert_eq!(interpret("Beta[1, 1]").unwrap(), "1");
  }

  #[test]
  fn positive_integers() {
    assert_eq!(interpret("Beta[2, 3]").unwrap(), "1/12");
  }

  #[test]
  fn positive_integers_2() {
    assert_eq!(interpret("Beta[3, 4]").unwrap(), "1/60");
  }

  #[test]
  fn half_half() {
    assert_eq!(interpret("Beta[1/2, 1/2]").unwrap(), "Pi");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("Beta[1.5, 2.5]").unwrap().parse().unwrap();
    assert!((result - 0.19634954084936207).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("Beta[a, b]").unwrap(), "Beta[a, b]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[Beta[2, 3]]").unwrap().parse().unwrap();
    assert!((result - 0.08333333333333333).abs() < 1e-10);
  }
}

mod log_integral {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("LogIntegral[0]").unwrap(), "0");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("LogIntegral[1]").unwrap(), "-Infinity");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("LogIntegral[x]").unwrap(), "LogIntegral[x]");
  }

  #[test]
  fn numeric_two() {
    let result: f64 = interpret("LogIntegral[2.0]").unwrap().parse().unwrap();
    assert!((result - 1.0451637801174924).abs() < 1e-10);
  }

  #[test]
  fn numeric_ten() {
    let result: f64 = interpret("LogIntegral[10.0]").unwrap().parse().unwrap();
    assert!((result - 6.165599504787296).abs() < 1e-10);
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("LogIntegral[0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.37867104306108795)).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[LogIntegral[2]]").unwrap().parse().unwrap();
    assert!((result - 1.0451637801174924).abs() < 1e-10);
  }
}

mod hermite_h {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("HermiteH[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("HermiteH[1, x]").unwrap(), "2*x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("HermiteH[2, x]").unwrap(), "-2 + 4*x^2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(interpret("HermiteH[3, x]").unwrap(), "-12*x + 8*x^3");
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("HermiteH[4, 0]").unwrap(), "12");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("HermiteH[3, 1]").unwrap(), "-4");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("HermiteH[5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 31.757759999999998).abs() < 1e-8);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("HermiteH[n, x]").unwrap(), "HermiteH[n, x]");
  }
}

mod airy_ai {
  use super::*;

  #[test]
  fn at_zero() {
    let result: f64 = interpret("AiryAi[0.0]").unwrap().parse().unwrap();
    assert!((result - 0.3550280538878173).abs() < 1e-10);
  }

  #[test]
  fn numeric_positive() {
    let result: f64 = interpret("AiryAi[1.0]").unwrap().parse().unwrap();
    assert!((result - 0.13529241631288147).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result: f64 = interpret("AiryAi[-1.0]").unwrap().parse().unwrap();
    assert!((result - 0.5355608832923522).abs() < 1e-10);
  }

  #[test]
  fn numeric_larger_positive() {
    let result: f64 = interpret("AiryAi[2.0]").unwrap().parse().unwrap();
    assert!((result - 0.03492413042327438).abs() < 1e-10);
  }

  #[test]
  fn numeric_larger_negative() {
    let result: f64 = interpret("AiryAi[-2.0]").unwrap().parse().unwrap();
    assert!((result - 0.22740742820168558).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[AiryAi[1]]").unwrap().parse().unwrap();
    assert!((result - 0.13529241631288147).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("AiryAi[x]").unwrap(), "AiryAi[x]");
  }
}

mod gegenbauer_c {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("GegenbauerC[0, 2, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("GegenbauerC[1, 2, x]").unwrap(), "4*x");
  }

  #[test]
  fn degree_two_lambda_two() {
    assert_eq!(interpret("GegenbauerC[2, 2, x]").unwrap(), "-2 + 12*x^2");
  }

  #[test]
  fn degree_three_lambda_two() {
    assert_eq!(interpret("GegenbauerC[3, 2, x]").unwrap(), "-12*x + 32*x^3");
  }

  #[test]
  fn reduces_to_chebyshev_u() {
    // GegenbauerC[n, 1, x] = ChebyshevU[n, x]
    assert_eq!(interpret("GegenbauerC[2, 1, x]").unwrap(), "-1 + 4*x^2");
    assert_eq!(interpret("GegenbauerC[3, 1, x]").unwrap(), "-4*x + 8*x^3");
  }

  #[test]
  fn rational_lambda() {
    assert_eq!(
      interpret("GegenbauerC[2, 3/2, x]").unwrap(),
      "-3/2 + (15*x^2)/2"
    );
  }

  #[test]
  fn at_integer() {
    assert_eq!(interpret("GegenbauerC[2, 2, 0]").unwrap(), "-2");
    assert_eq!(interpret("GegenbauerC[3, 2, 0]").unwrap(), "0");
    assert_eq!(interpret("GegenbauerC[3, 2, 1]").unwrap(), "20");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("GegenbauerC[2, 2, 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.0).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[GegenbauerC[3, 2, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - (-2.0)).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("GegenbauerC[n, m, x]").unwrap(),
      "GegenbauerC[n, m, x]"
    );
  }
}

mod fourier {
  use super::*;

  #[test]
  fn basic_integer_list() {
    assert_eq!(
      interpret("Fourier[{1, 2, 3, 4}]").unwrap(),
      "{5. + 0.*I, -1. - 1.*I, -1. + 0.*I, -1. + 1.*I}"
    );
  }

  #[test]
  fn single_element() {
    assert_eq!(interpret("Fourier[{5}]").unwrap(), "{5.}");
  }

  #[test]
  fn two_equal_elements() {
    // All imaginary parts are exactly 0 → Real output
    let result = interpret("Fourier[{2, 2}]").unwrap();
    assert!(result.contains("2.828427124746190"), "got: {}", result);
    assert!(result.contains("0."), "got: {}", result);
    // Should not contain I (all real)
    assert!(!result.contains("I"), "got: {}", result);
  }

  #[test]
  fn three_elements() {
    let result = interpret("Fourier[{1, 2, 3}]").unwrap();
    // Should be complex since some elements have nonzero imaginary parts
    assert!(result.contains("I"), "got: {}", result);
  }

  #[test]
  fn complex_input() {
    let result = interpret("Fourier[{1 + 2*I, 3 - I}]").unwrap();
    assert!(result.contains("I"), "got: {}", result);
  }

  #[test]
  fn fourier_parameters_signal_processing() {
    // FourierParameters -> {1, -1} (signal processing convention)
    assert_eq!(
      interpret("Fourier[{1, 2, 3, 4}, FourierParameters -> {1, -1}]").unwrap(),
      "{10. + 0.*I, -2. + 2.*I, -2. + 0.*I, -2. - 2.*I}"
    );
  }

  #[test]
  fn fourier_parameters_data_analysis() {
    // FourierParameters -> {-1, 1} (data analysis convention)
    let result =
      interpret("Fourier[{1, 2, 3, 4}, FourierParameters -> {-1, 1}]").unwrap();
    assert!(result.contains("I"), "got: {}", result);
  }

  #[test]
  fn impulse_all_real() {
    // Fourier of impulse: all results are equal and real
    assert_eq!(
      interpret("Fourier[{1, 0, 0, 0}]").unwrap(),
      "{0.5, 0.5, 0.5, 0.5}"
    );
  }

  #[test]
  fn empty_list_returns_unevaluated() {
    assert_eq!(interpret("Fourier[{}]").unwrap(), "Fourier[{}]");
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    assert_eq!(
      interpret("Fourier[{a, b, c}]").unwrap(),
      "Fourier[{a, b, c}]"
    );
  }

  #[test]
  fn non_list_returns_unevaluated() {
    assert_eq!(interpret("Fourier[\"hello\"]").unwrap(), "Fourier[hello]");
  }

  #[test]
  fn roundtrip() {
    // InverseFourier[Fourier[x]] should return x
    assert_eq!(
      interpret("InverseFourier[Fourier[{1, 2, 3, 4}]]").unwrap(),
      "{1., 2., 3., 4.}"
    );
  }
}

mod inverse_fourier {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret(
        "InverseFourier[{5. + 0.*I, -1. - 1.*I, -1. + 0.*I, -1. + 1.*I}]"
      )
      .unwrap(),
      "{1., 2., 3., 4.}"
    );
  }

  #[test]
  fn single_element() {
    assert_eq!(interpret("InverseFourier[{5.}]").unwrap(), "{5.}");
  }

  #[test]
  fn fourier_parameters() {
    let result =
      interpret("InverseFourier[{1, 2, 3, 4}, FourierParameters -> {1, -1}]")
        .unwrap();
    assert!(result.contains("I"), "got: {}", result);
  }

  #[test]
  fn empty_list_returns_unevaluated() {
    assert_eq!(
      interpret("InverseFourier[{}]").unwrap(),
      "InverseFourier[{}]"
    );
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    assert_eq!(
      interpret("InverseFourier[{a, b}]").unwrap(),
      "InverseFourier[{a, b}]"
    );
  }
}

mod spherical_harmonic_y {
  use super::*;

  #[test]
  fn y00() {
    // Y_0^0 = 1/sqrt(4*Pi)
    let result = interpret("SphericalHarmonicY[0, 0, 0, 0]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
    assert!((val - expected).abs() < 1e-12);
  }

  #[test]
  fn y10() {
    // Y_1^0(0, 0) = sqrt(3/(4*Pi))
    let result = interpret("SphericalHarmonicY[1, 0, 0, 0]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = (3.0 / (4.0 * std::f64::consts::PI)).sqrt();
    assert!((val - expected).abs() < 1e-12);
  }

  #[test]
  fn y20_at_pi_over_3() {
    // Y_2^0(Pi/3, 0) = sqrt(5/(4*Pi)) * P_2(cos(Pi/3))
    let result = interpret("SphericalHarmonicY[2, 0, Pi/3, 0]").unwrap();
    let val: f64 = result.parse().unwrap();
    let cos_theta = (std::f64::consts::PI / 3.0).cos();
    let p2 = (3.0 * cos_theta * cos_theta - 1.0) / 2.0;
    let expected = (5.0 / (4.0 * std::f64::consts::PI)).sqrt() * p2;
    assert!((val - expected).abs() < 1e-12);
  }

  #[test]
  fn m_greater_than_l() {
    assert_eq!(interpret("SphericalHarmonicY[2, 3, Pi/4, 0]").unwrap(), "0");
  }

  #[test]
  fn complex_result() {
    // Y_1^1(Pi/4, Pi/4) should have both real and imaginary parts
    let result = interpret("SphericalHarmonicY[1, 1, Pi/4, Pi/4]").unwrap();
    assert!(result.contains("I"));
  }

  #[test]
  fn negative_m() {
    // Y_1^{-1}(Pi/4, Pi/4) has imaginary part
    let result = interpret("SphericalHarmonicY[1, -1, Pi/4, Pi/4]").unwrap();
    assert!(result.contains("I"));
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("SphericalHarmonicY[x, 0, 0, 0]").unwrap(),
      "SphericalHarmonicY[x, 0, 0, 0]"
    );
  }
}

mod nsum {
  use super::*;

  #[test]
  fn finite_sum() {
    assert_eq!(
      interpret("NSum[1/i^2, {i, 1, 10}]").unwrap(),
      "1.5497677311665408"
    );
  }

  #[test]
  fn finite_sum_simple() {
    assert_eq!(interpret("NSum[i, {i, 1, 5}]").unwrap(), "15.");
  }

  #[test]
  fn infinite_sum_reciprocal_squares() {
    // Pi^2/6 ≈ 1.6449340668482264
    let result = interpret("NSum[1/i^2, {i, 1, Infinity}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.6449340668482264).abs() < 1e-10);
  }

  #[test]
  fn infinite_sum_factorial() {
    // e ≈ 2.718281828459045
    let result = interpret("NSum[1/Factorial[k], {k, 0, Infinity}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - std::f64::consts::E).abs() < 1e-10);
  }

  #[test]
  fn infinite_sum_alternating() {
    // ln(2) ≈ 0.6931471805599453
    let result = interpret("NSum[(-1)^(k+1)/k, {k, 1, Infinity}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.0_f64.ln()).abs() < 1e-10);
  }

  #[test]
  fn infinite_sum_geometric() {
    // Σ 1/2^k from k=0 = 2
    let result = interpret("NSum[1/2^k, {k, 0, Infinity}]").unwrap();
    assert_eq!(result, "2.");
  }

  #[test]
  fn infinite_sum_reciprocal_cubes() {
    // ζ(3) ≈ 1.2020569031595943
    let result = interpret("NSum[1/i^3, {i, 1, Infinity}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.2020569031595943).abs() < 1e-10);
  }

  #[test]
  fn scoped_variable() {
    // Iteration variable should be scoped locally
    let result =
      interpret("n = 5; NSum[1/Factorial[n], {n, 0, Infinity}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - std::f64::consts::E).abs() < 1e-10);
  }

  #[test]
  fn unevaluated_symbolic() {
    assert_eq!(
      interpret("NSum[f[x], {x, 1, Infinity}]").unwrap(),
      "NSum[f[x], {x, 1, Infinity}]"
    );
  }
}

mod nproduct {
  use super::*;

  #[test]
  fn basic_product() {
    assert_eq!(interpret("NProduct[i, {i, 1, 5}]").unwrap(), "120.");
  }

  #[test]
  fn product_of_squares() {
    assert_eq!(interpret("NProduct[i^2, {i, 1, 4}]").unwrap(), "576.");
  }

  #[test]
  fn product_of_reciprocals() {
    assert_eq!(
      interpret("NProduct[1/i, {i, 1, 5}]").unwrap(),
      "0.008333333333333333"
    );
  }

  #[test]
  fn product_shifted() {
    assert_eq!(interpret("NProduct[i + 1, {i, 0, 3}]").unwrap(), "24.");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[NProduct]").unwrap(),
      "{HoldAll, Protected}"
    );
  }
}

mod lerch_phi {
  use super::*;

  #[test]
  fn lerch_phi_zero_z() {
    // LerchPhi[0, s, a] = a^(-s)
    // With numeric args, evaluates to numeric
    let result = interpret("LerchPhi[0, 2.0, 3.0]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - 1.0 / 9.0).abs() < 1e-10,
      "LerchPhi[0, 2, 3] should be 1/9, got {}",
      val
    );
  }

  #[test]
  fn lerch_phi_numeric() {
    // LerchPhi[0.5, 2, 1] = Σ 0.5^k / (k+1)^2
    let result = interpret("LerchPhi[0.5, 2, 1]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - 1.16459).abs() < 0.001,
      "LerchPhi[0.5, 2, 1] should be near 1.16459, got {}",
      val
    );
  }

  #[test]
  fn lerch_phi_relates_to_zeta() {
    // LerchPhi[1, 2, 1] = Σ 1/(k+1)^2 = π²/6
    let result = interpret("LerchPhi[1, 2, 1]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    assert!(
      (val - expected).abs() < 1e-6,
      "LerchPhi[1, 2, 1] should be pi^2/6 = {}, got {}",
      expected,
      val
    );
  }

  #[test]
  fn lerch_phi_small_z() {
    let result = interpret("LerchPhi[0.1, 1, 1]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      val > 1.0 && val < 2.0,
      "Should be between 1 and 2, got {}",
      val
    );
  }

  #[test]
  fn lerch_phi_symbolic() {
    assert_eq!(interpret("LerchPhi[z, s, a]").unwrap(), "LerchPhi[z, s, a]");
  }
}

mod weierstrass_p {
  use super::*;

  #[test]
  fn weierstrass_p_at_zero() {
    // Pole at origin
    assert_eq!(
      interpret("WeierstrassP[0, {1, 1}]").unwrap(),
      "ComplexInfinity"
    );
  }

  #[test]
  fn weierstrass_p_numeric() {
    // Numeric evaluation should return a real number
    let result = interpret("WeierstrassP[0.5, {4, 0}]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(val.is_finite(), "Result should be finite: {}", val);
  }

  #[test]
  fn weierstrass_p_even_function() {
    // ℘ is even: ℘(-u) == ℘(u)
    let pos = interpret("WeierstrassP[0.3, {2, 3}]").unwrap();
    let neg = interpret("WeierstrassP[-0.3, {2, 3}]").unwrap();
    let p: f64 = pos.parse().unwrap();
    let n: f64 = neg.parse().unwrap();
    assert!(
      (p - n).abs() < 1e-10,
      "WeierstrassP should be even: {} vs {}",
      p,
      n
    );
  }

  #[test]
  fn weierstrass_p_degenerate() {
    // g2 = 0, g3 = 0: ℘(u) = 1/u²
    let result = interpret("WeierstrassP[0.5, {0, 0}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 4.0).abs() < 1e-10,
      "WeierstrassP[0.5, {{0,0}}] should be 1/0.25 = 4.0, got {}",
      val
    );
  }

  #[test]
  fn weierstrass_p_symbolic() {
    // Symbolic inputs should return unevaluated
    assert_eq!(
      interpret("WeierstrassP[u, {g2, g3}]").unwrap(),
      "WeierstrassP[u, {g2, g3}]"
    );
  }

  #[test]
  fn weierstrass_p_differential_equation() {
    // Verify ℘ satisfies: (℘')² = 4℘³ - g₂℘ - g₃ approximately
    // Use numerical differentiation: ℘'(u) ≈ (℘(u+h) - ℘(u-h)) / (2h)
    let g2 = 4.0;
    let g3 = 1.0;
    let u = 0.4;
    let h = 1e-6;
    let p_u = interpret(&format!("WeierstrassP[{}, {{{}, {}}}]", u, g2, g3))
      .unwrap()
      .parse::<f64>()
      .unwrap();
    let p_plus =
      interpret(&format!("WeierstrassP[{}, {{{}, {}}}]", u + h, g2, g3))
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let p_minus =
      interpret(&format!("WeierstrassP[{}, {{{}, {}}}]", u - h, g2, g3))
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let pp = (p_plus - p_minus) / (2.0 * h); // numerical derivative
    let lhs = pp * pp;
    let rhs = 4.0 * p_u * p_u * p_u - g2 * p_u - g3;
    assert!(
      (lhs - rhs).abs() / rhs.abs().max(1.0) < 1e-4,
      "Should satisfy (℘')² = 4℘³ - g₂℘ - g₃: lhs={}, rhs={}",
      lhs,
      rhs
    );
  }
}

mod weierstrass_p_prime {
  use super::*;

  #[test]
  fn at_zero() {
    // Pole of order 3 at origin
    assert_eq!(
      interpret("WeierstrassPPrime[0, {1, 2}]").unwrap(),
      "ComplexInfinity"
    );
  }

  #[test]
  fn numeric_basic() {
    // WeierstrassPPrime[2., {1, 2}] ≈ 8.3966
    let result = interpret("WeierstrassPPrime[2., {1, 2}]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - 8.3966).abs() < 0.01,
      "WeierstrassPPrime[2., {{1, 2}}] should be ≈ 8.3966, got {}",
      val
    );
  }

  #[test]
  fn numeric_negative() {
    // WeierstrassPPrime[0.5, {1, 2}] ≈ -15.914
    let result = interpret("WeierstrassPPrime[0.5, {1, 2}]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - (-15.914)).abs() < 0.01,
      "WeierstrassPPrime[0.5, {{1, 2}}] should be ≈ -15.914, got {}",
      val
    );
  }

  #[test]
  fn numeric_three_real_roots() {
    // Δ > 0 case: WeierstrassPPrime[1.0, {4, 0}] ≈ -1.5157
    let result = interpret("WeierstrassPPrime[1.0, {4, 0}]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - (-1.5157)).abs() < 0.01,
      "WeierstrassPPrime[1.0, {{4, 0}}] should be ≈ -1.5157, got {}",
      val
    );
  }

  #[test]
  fn odd_function() {
    // ℘' is odd: ℘'(-u) == -℘'(u)
    let pos = interpret("WeierstrassPPrime[0.3, {2, 3}]").unwrap();
    let neg = interpret("WeierstrassPPrime[-0.3, {2, 3}]").unwrap();
    let p: f64 = pos.parse().unwrap();
    let n: f64 = neg.parse().unwrap();
    assert!(
      (p + n).abs() < 1e-4,
      "WeierstrassPPrime should be odd: {} vs {}",
      p,
      n
    );
  }

  #[test]
  fn symbolic() {
    // Symbolic inputs should return unevaluated
    assert_eq!(
      interpret("WeierstrassPPrime[z, {g2, g3}]").unwrap(),
      "WeierstrassPPrime[z, {g2, g3}]"
    );
  }

  #[test]
  fn differential_equation() {
    // Verify (℘')² = 4℘³ - g₂℘ - g₃
    let g2 = 4.0;
    let g3 = 1.0;
    let u = 0.4;
    let p_u = interpret(&format!("WeierstrassP[{}, {{{}, {}}}]", u, g2, g3))
      .unwrap()
      .parse::<f64>()
      .unwrap();
    let pp_u =
      interpret(&format!("WeierstrassPPrime[{}, {{{}, {}}}]", u, g2, g3))
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let lhs = pp_u * pp_u;
    let rhs = 4.0 * p_u * p_u * p_u - g2 * p_u - g3;
    assert!(
      (lhs - rhs).abs() / rhs.abs().max(1.0) < 1e-4,
      "Should satisfy (℘')² = 4℘³ - g₂℘ - g₃: lhs={}, rhs={}",
      lhs,
      rhs
    );
  }
}

mod inverse_weierstrass_p {
  use super::*;

  #[test]
  fn roundtrip_form1() {
    // WeierstrassP[0.5, {4, 0}] ≈ 4.050208734712057
    // InverseWeierstrassP[4.050208734712057, {4.0, 0.0}] should return {u, p'}
    // where u ≈ 0.5
    let result =
      interpret("InverseWeierstrassP[4.050208734712057, {4.0, 0.0}]").unwrap();
    // result is a list {u, pp}
    assert!(result.starts_with('{'), "expected list, got {}", result);
    let inner = &result[1..result.len() - 1];
    let parts: Vec<&str> = inner.split(", ").collect();
    assert_eq!(parts.len(), 2, "expected 2 elements, got {:?}", parts);
    let u: f64 = parts[0].parse().unwrap();
    assert!(
      (u.abs() - 0.5).abs() < 1e-4,
      "expected |u| ≈ 0.5, got {}",
      u
    );
  }

  #[test]
  fn roundtrip_form2() {
    // InverseWeierstrassP[{4.050208734712057, -15.797491968339017}, {4.0, 0.0}]
    // should return u ≈ 0.5
    let result = interpret(
      "InverseWeierstrassP[{4.050208734712057, -15.797491968339017}, {4.0, 0.0}]",
    )
    .unwrap();
    let u: f64 = result.parse().unwrap();
    assert!((u - 0.5).abs() < 1e-4, "expected u ≈ 0.5, got {}", u);
  }

  #[test]
  fn consistency_check() {
    // Compute WeierstrassP at a point, then invert, and verify roundtrip
    let p_str = interpret("WeierstrassP[1.0, {2.0, 3.0}]").unwrap();
    let pp_str = interpret("WeierstrassPPrime[1.0, {2.0, 3.0}]").unwrap();
    let p: f64 = p_str.parse().unwrap();
    let pp: f64 = pp_str.parse().unwrap();

    let inv_result = interpret(&format!(
      "InverseWeierstrassP[{{{}, {}}}, {{2.0, 3.0}}]",
      p, pp
    ))
    .unwrap();
    let u: f64 = inv_result.parse().unwrap();
    assert!((u - 1.0).abs() < 1e-4, "expected u ≈ 1.0, got {}", u);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("InverseWeierstrassP[p, {g2, g3}]").unwrap(),
      "InverseWeierstrassP[p, {g2, g3}]"
    );
  }

  #[test]
  fn wrong_arg_count() {
    assert_eq!(
      interpret("InverseWeierstrassP[1]").unwrap(),
      "InverseWeierstrassP[1]"
    );
  }
}

mod inverse_jacobi {
  use super::*;

  fn assert_f64_approx(code: &str, expected: f64, tol: f64) {
    let result = interpret(code).unwrap().parse::<f64>().unwrap();
    assert!(
      (result - expected).abs() < tol,
      "{} = {} (expected {})",
      code,
      result,
      expected
    );
  }

  #[test]
  fn inverse_jacobi_sn_numeric() {
    assert_f64_approx("InverseJacobiSN[0.5, 0.3]", 0.5306368995398673, 1e-8);
  }

  #[test]
  fn inverse_jacobi_cn_numeric() {
    assert_f64_approx("InverseJacobiCN[0.5, 0.3]", 1.0991352230920428, 1e-8);
  }

  #[test]
  fn inverse_jacobi_sn_zero() {
    assert_eq!(interpret("InverseJacobiSN[0, 1/2]").unwrap(), "0");
    assert_eq!(interpret("InverseJacobiSN[0, 0.5]").unwrap(), "0.");
  }

  #[test]
  fn inverse_jacobi_cn_one() {
    assert_eq!(interpret("InverseJacobiCN[1, 0.5]").unwrap(), "0");
  }

  #[test]
  fn inverse_jacobi_sn_symbolic() {
    assert_eq!(
      interpret("InverseJacobiSN[x, m]").unwrap(),
      "InverseJacobiSN[x, m]"
    );
  }

  #[test]
  fn inverse_jacobi_cn_symbolic() {
    assert_eq!(
      interpret("InverseJacobiCN[x, m]").unwrap(),
      "InverseJacobiCN[x, m]"
    );
  }

  #[test]
  fn inverse_jacobi_sn_roundtrip() {
    assert_f64_approx("JacobiSN[InverseJacobiSN[0.5, 0.3], 0.3]", 0.5, 1e-8);
  }

  #[test]
  fn inverse_jacobi_cn_roundtrip() {
    assert_f64_approx("JacobiCN[InverseJacobiCN[0.5, 0.3], 0.3]", 0.5, 1e-8);
  }

  #[test]
  fn inverse_jacobi_sn_negative() {
    assert_f64_approx("InverseJacobiSN[-0.5, 0.3]", -0.5306368995398673, 1e-8);
  }

  #[test]
  fn inverse_jacobi_cn_m_zero() {
    // When m=0, InverseJacobiCN[x, 0] = ArcCos[x]
    assert_f64_approx(
      "InverseJacobiCN[0.5, 0]",
      std::f64::consts::FRAC_PI_3,
      1e-8,
    );
  }

  #[test]
  fn inverse_jacobi_sn_m_zero() {
    // When m=0, InverseJacobiSN[x, 0] = ArcSin[x]
    assert_f64_approx(
      "InverseJacobiSN[0.5, 0]",
      std::f64::consts::FRAC_PI_6,
      1e-8,
    );
  }

  #[test]
  fn inverse_jacobi_sn_attributes() {
    assert_eq!(
      interpret("Attributes[InverseJacobiSN]").unwrap(),
      "{Listable, NumericFunction, Protected, ReadProtected}"
    );
  }

  #[test]
  fn inverse_jacobi_cn_attributes() {
    assert_eq!(
      interpret("Attributes[InverseJacobiCN]").unwrap(),
      "{Listable, NumericFunction, Protected, ReadProtected}"
    );
  }
}

mod hypergeometric_pfq {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("HypergeometricPFQ[{a, b}, {c}, x]").unwrap(),
      "HypergeometricPFQ[{a, b}, {c}, x]"
    );
  }

  #[test]
  fn zero_arg() {
    assert_eq!(interpret("HypergeometricPFQ[{1, 2}, {3}, 0]").unwrap(), "1");
  }

  #[test]
  fn empty_lists_is_exp() {
    assert_eq!(interpret("HypergeometricPFQ[{}, {}, x]").unwrap(), "E^x");
  }

  #[test]
  fn numeric_2f1() {
    assert_eq!(
      interpret("HypergeometricPFQ[{1, 2}, {3}, 0.5]").unwrap(),
      "1.5451774444795623"
    );
  }

  #[test]
  fn numeric_1f1() {
    assert_eq!(
      interpret("HypergeometricPFQ[{1}, {2}, 1.0]").unwrap(),
      "1.7182818284590453"
    );
  }

  #[test]
  fn divergent_at_one() {
    assert_eq!(
      interpret("HypergeometricPFQ[{1, 2}, {3}, 1]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn rational_params() {
    assert_eq!(
      interpret("N[HypergeometricPFQ[{1/2}, {3/2}, -1]]").unwrap(),
      "0.746824132812427"
    );
  }

  #[test]
  fn n_wrapper() {
    assert_eq!(
      interpret("N[HypergeometricPFQ[{1, 2}, {3}, 1/2]]").unwrap(),
      "1.5451774444795623"
    );
  }
}

mod riemann_r {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("RiemannR[x]").unwrap(), "RiemannR[x]");
  }

  #[test]
  fn integer_symbolic() {
    assert_eq!(interpret("RiemannR[100]").unwrap(), "RiemannR[100]");
  }

  #[test]
  fn one() {
    assert_eq!(interpret("RiemannR[1]").unwrap(), "1");
  }

  #[test]
  fn zero_symbolic() {
    assert_eq!(interpret("RiemannR[0]").unwrap(), "RiemannR[0]");
  }

  #[test]
  fn numeric_10() {
    assert_eq!(interpret("RiemannR[10.]").unwrap(), "4.564583141005091");
  }

  #[test]
  fn numeric_100() {
    assert_eq!(interpret("RiemannR[100.]").unwrap(), "25.66163326692419");
  }

  #[test]
  fn numeric_1000() {
    assert_eq!(interpret("RiemannR[1000.]").unwrap(), "168.3594462811673");
  }

  #[test]
  fn n_wrapper() {
    assert_eq!(
      interpret("N[RiemannR[1000000]]").unwrap(),
      "78527.39942912768"
    );
  }
}

mod meijer_g {
  use super::*;

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("MeijerG[{{a}, {b}}, {{c}, {d}}, z]").unwrap(),
      "MeijerG[{{a}, {b}}, {{c}, {d}}, z]"
    );
  }

  #[test]
  fn bad_args_unevaluated() {
    assert_eq!(interpret("MeijerG[1, 2, 3]").unwrap(), "MeijerG[1, 2, 3]");
  }

  #[test]
  fn z_zero_simple() {
    assert_eq!(interpret("MeijerG[{{}, {}}, {{0}, {}}, 0]").unwrap(), "1");
  }

  #[test]
  fn numeric_simple_pole_exp() {
    // G^{1,0}_{0,1}(z | ; 0) = e^{-z}
    let result = interpret("MeijerG[{{}, {}}, {{0}, {}}, 3.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = (-3.0_f64).exp();
    assert!(
      (val - expected).abs() < 1e-8,
      "got {} expected {}",
      val,
      expected
    );
  }

  #[test]
  fn numeric_double_pole() {
    // G^{2,0}_{0,2}(z | ; 1/2, 3/2)
    let result = interpret("MeijerG[{{}, {}}, {{0.5, 1.5}, {}}, 3.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 0.13914993366209005;
    assert!(
      (val - expected).abs() < 1e-4,
      "got {} expected {}",
      val,
      expected
    );
  }

  #[test]
  fn numeric_with_upper_params() {
    // MeijerG[{{1, 2}, {}}, {{3}, {}}, 1.0]
    let result = interpret("MeijerG[{{1, 2}, {}}, {{3}, {}}, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 0.2109579130304179;
    assert!(
      (val - expected).abs() < 1e-4,
      "got {} expected {}",
      val,
      expected
    );
  }

  #[test]
  fn n_wrapper_integer_args() {
    let result = interpret("N[MeijerG[{{1, 2}, {}}, {{3}, {}}, 1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 0.2109579130304179;
    assert!(
      (val - expected).abs() < 1e-4,
      "got {} expected {}",
      val,
      expected
    );
  }

  #[test]
  fn n_wrapper_rational_args() {
    let result = interpret("N[MeijerG[{{}, {1/2}}, {{0}, {}}, 1/2]]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 0.7978845608028653;
    assert!(
      (val - expected).abs() < 1e-6,
      "got {} expected {}",
      val,
      expected
    );
  }

  #[test]
  fn numeric_coinciding_poles_double() {
    // MeijerG[{{}, {}}, {{0, 1}, {}}, 2.0]
    let result = interpret("N[MeijerG[{{}, {}}, {{0, 1}, {}}, 2]]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 0.1396674740152931;
    assert!(
      (val - expected).abs() < 1e-4,
      "got {} expected {}",
      val,
      expected
    );
  }

  #[test]
  fn numeric_with_lower_rest() {
    // MeijerG[{{}, {}}, {{0}, {1}}, 2.0] - has lower rest parameters
    let result = interpret("N[MeijerG[{{}, {}}, {{0}, {1}}, 2]]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = -0.5659599737610849;
    assert!(
      (val - expected).abs() < 1e-4,
      "got {} expected {}",
      val,
      expected
    );
  }

  #[test]
  fn numeric_non_coinciding() {
    // MeijerG[{{1/2}, {}}, {{0, 1/2}, {}}, 1.0]
    let result =
      interpret("N[MeijerG[{{1/2}, {}}, {{0, 1/2}, {}}, 1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    let expected = 1.3432934216467352;
    assert!(
      (val - expected).abs() < 1e-4,
      "got {} expected {}",
      val,
      expected
    );
  }

  #[test]
  fn hdiv_positive_integer_diff() {
    // MeijerG[{{1}, {}}, {{0}, {}}, 1.0] - a_i - b_j = 1 (positive integer) → does not exist
    assert_eq!(
      interpret("MeijerG[{{1}, {}}, {{0}, {}}, 1.0]").unwrap(),
      "MeijerG[{{1}, {}}, {{0}, {}}, 1.]"
    );
  }

  #[test]
  fn empty_m_and_n_invalid() {
    // MeijerG[{{}, {}}, {{}, {}}, x] - m=0, n=0 → does not exist
    assert_eq!(
      interpret("MeijerG[{{}, {}}, {{}, {}}, x]").unwrap(),
      "MeijerG[{{}, {}}, {{}, {}}, x]"
    );
  }
}

mod hypergeometric_pfq_regularized {
  use super::*;

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("HypergeometricPFQRegularized[{a, b}, {c}, z]").unwrap(),
      "HypergeometricPFQRegularized[{a, b}, {c}, z]"
    );
  }

  #[test]
  fn z_zero_exact() {
    assert_eq!(
      interpret("HypergeometricPFQRegularized[{1, 2}, {3}, 0]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn empty_b_list() {
    assert_eq!(
      interpret("HypergeometricPFQRegularized[{}, {}, x]").unwrap(),
      "E^x"
    );
  }

  #[test]
  fn numeric_real() {
    let result =
      interpret("HypergeometricPFQRegularized[{1, 2}, {3}, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.7725887222397811).abs() < 1e-10, "got {}", val);
  }

  #[test]
  fn n_wrapper_rational() {
    let result =
      interpret("N[HypergeometricPFQRegularized[{1/2}, {3/2}, -1]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8427007929497148).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn infinity_case() {
    assert_eq!(
      interpret("HypergeometricPFQRegularized[{1, 2}, {3}, 1]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn multiple_b_params() {
    let result =
      interpret("N[HypergeometricPFQRegularized[{1, 2}, {3, 4}, 1/2]]")
        .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.09083551648531873).abs() < 1e-8, "got {}", val);
  }
}

mod hypergeometric_2f1_regularized {
  use super::*;

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("Hypergeometric2F1Regularized[a, b, c, z]").unwrap(),
      "Hypergeometric2F1Regularized[a, b, c, z]"
    );
  }

  #[test]
  fn z_zero_exact() {
    // Hypergeometric2F1Regularized[1, 2, 3, 0] = 1/Gamma[3] = 1/2
    assert_eq!(
      interpret("Hypergeometric2F1Regularized[1, 2, 3, 0]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn z_zero_c_one() {
    // Hypergeometric2F1Regularized[1, 1, 1, 0] = 1/Gamma[1] = 1
    assert_eq!(
      interpret("Hypergeometric2F1Regularized[1, 1, 1, 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn numeric_real() {
    // Hypergeometric2F1Regularized[1.0, 2.0, 3.0, 0.5] ≈ 0.7725887222397809
    let result =
      interpret("Hypergeometric2F1Regularized[1.0, 2.0, 3.0, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.7725887222397809).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn n_wrapper() {
    let result =
      interpret("N[Hypergeometric2F1Regularized[1, 2, 3, 1/4]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.6029131592284935).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn wrong_arg_count() {
    assert_eq!(
      interpret("Hypergeometric2F1Regularized[1, 2, 3]").unwrap(),
      "Hypergeometric2F1Regularized[1, 2, 3]"
    );
  }
}

mod q_pochhammer {
  use super::*;

  #[test]
  fn n_zero() {
    assert_eq!(interpret("QPochhammer[a, q, 0]").unwrap(), "1");
  }

  #[test]
  fn n_one() {
    // QPochhammer[a, q, 1] = 1 - a
    assert_eq!(interpret("QPochhammer[a, q, 1]").unwrap(), "1 - a");
  }

  #[test]
  fn zero_a() {
    // QPochhammer[0, q, n] = 1 for any n
    assert_eq!(interpret("QPochhammer[0, q, 3]").unwrap(), "1");
  }

  #[test]
  fn rational_example() {
    // QPochhammer[1/2, 1/3, 3] = 85/216
    assert_eq!(interpret("QPochhammer[1/2, 1/3, 3]").unwrap(), "85/216");
  }

  #[test]
  fn integer_example() {
    // QPochhammer[2, 3, 4] = 4505
    assert_eq!(interpret("QPochhammer[2, 3, 4]").unwrap(), "4505");
  }

  #[test]
  fn one_arg_n() {
    // QPochhammer[2, 3, 1] = 1 - 2 = -1
    assert_eq!(interpret("QPochhammer[2, 3, 1]").unwrap(), "-1");
  }

  #[test]
  fn numeric_float() {
    // QPochhammer[0.5, 0.25, 4] ≈ 0.4205169677734375
    let result = interpret("QPochhammer[0.5, 0.25, 4]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.4205169677734375).abs() < 1e-10, "got {}", val);
  }

  #[test]
  fn four_factor_rational() {
    // QPochhammer[1/2, 1/4, 4] = 27559/65536
    assert_eq!(
      interpret("QPochhammer[1/2, 1/4, 4]").unwrap(),
      "27559/65536"
    );
  }

  #[test]
  fn symbolic_n_unevaluated() {
    assert_eq!(
      interpret("QPochhammer[a, q, n]").unwrap(),
      "QPochhammer[a, q, n]"
    );
  }

  #[test]
  fn a_one_q_one() {
    // QPochhammer[1, 1, 3] = (1-1)(1-1)(1-1) = 0
    assert_eq!(interpret("QPochhammer[1, 1, 3]").unwrap(), "0");
  }
}

mod spherical_bessel_j {
  use super::*;

  #[test]
  fn z_zero_n_zero() {
    assert_eq!(interpret("SphericalBesselJ[0, 0]").unwrap(), "1");
  }

  #[test]
  fn z_zero_n_one() {
    assert_eq!(interpret("SphericalBesselJ[1, 0]").unwrap(), "0");
  }

  #[test]
  fn z_zero_n_two() {
    assert_eq!(interpret("SphericalBesselJ[2, 0]").unwrap(), "0");
  }

  #[test]
  fn numeric_n0() {
    // SphericalBesselJ[0, 1.0] = sin(1)/1 ≈ 0.8414709848078965
    let result = interpret("SphericalBesselJ[0, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.8414709848078965).abs() < 1e-10, "got {}", val);
  }

  #[test]
  fn numeric_n1() {
    // SphericalBesselJ[1, 1.0] ≈ 0.30116867893975674
    let result = interpret("SphericalBesselJ[1, 1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.30116867893975674).abs() < 1e-10, "got {}", val);
  }

  #[test]
  fn n_wrapper() {
    let result = interpret("N[SphericalBesselJ[2, 3]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.2986374970757335).abs() < 1e-10, "got {}", val);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("SphericalBesselJ[0, x]").unwrap(),
      "SphericalBesselJ[0, x]"
    );
  }

  #[test]
  fn wrong_arg_count() {
    assert_eq!(
      interpret("SphericalBesselJ[1]").unwrap(),
      "SphericalBesselJ[1]"
    );
  }
}

mod log_gamma {
  use super::*;

  #[test]
  fn at_one() {
    assert_eq!(interpret("LogGamma[1]").unwrap(), "0");
  }

  #[test]
  fn at_two() {
    assert_eq!(interpret("LogGamma[2]").unwrap(), "0");
  }

  #[test]
  fn at_five() {
    // LogGamma[5] = Log[24]
    assert_eq!(interpret("LogGamma[5]").unwrap(), "Log[24]");
  }

  #[test]
  fn at_half_numeric() {
    // LogGamma[1/2] = Log[Sqrt[Pi]] ≈ 0.5723649429247001
    let result = interpret("N[LogGamma[1/2]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.5723649429247001).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("LogGamma[0]").unwrap(), "Infinity");
  }

  #[test]
  fn at_neg_one() {
    assert_eq!(interpret("LogGamma[-1]").unwrap(), "Infinity");
  }

  #[test]
  fn numeric_real() {
    let result = interpret("LogGamma[1.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-0.12078223763524526)).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn n_wrapper() {
    let result = interpret("N[LogGamma[3/2]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-0.12078223763524526)).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("LogGamma[x]").unwrap(), "LogGamma[x]");
  }
}

#[cfg(test)]
mod anger_j_tests {
  use super::*;

  // AngerJ[n, z] = BesselJ[n, z] for integer n
  #[test]
  fn integer_order_equals_bessel_j() {
    // AngerJ[0, 0] = BesselJ[0, 0] = 1
    assert_eq!(interpret("AngerJ[0, 0]").unwrap(), "1");

    // AngerJ[1, 0] = BesselJ[1, 0] = 0
    assert_eq!(interpret("AngerJ[1, 0]").unwrap(), "0");

    // AngerJ[2, 0] = BesselJ[2, 0] = 0
    assert_eq!(interpret("AngerJ[2, 0]").unwrap(), "0");
  }

  #[test]
  fn integer_order_numeric() {
    // AngerJ[0, 1.0] = BesselJ[0, 1.0] ≈ 0.7651976865579666
    let result: f64 = interpret("AngerJ[0, 1.0]").unwrap().parse().unwrap();
    assert!((result - 0.7651976865579666).abs() < 1e-8, "got {}", result);

    // AngerJ[1, 3.0] = BesselJ[1, 3.0] ≈ 0.33905895852593644
    let result: f64 = interpret("AngerJ[1, 3.0]").unwrap().parse().unwrap();
    assert!(
      (result - 0.33905895852593644).abs() < 1e-8,
      "got {}",
      result
    );
  }

  #[test]
  fn non_integer_order_at_zero() {
    // AngerJ[0.5, 0] = Sin[0.5 * Pi] / (0.5 * Pi) = 1 / (Pi/2) = 2/Pi ≈ 0.6366197723675814
    let result: f64 = interpret("AngerJ[0.5, 0]").unwrap().parse().unwrap();
    assert!(
      (result - 2.0 / std::f64::consts::PI).abs() < 1e-8,
      "got {}",
      result
    );
  }

  #[test]
  fn non_integer_order_numeric() {
    // AngerJ[0.5, 1.0] ≈ 0.8551653096792619
    let result: f64 = interpret("AngerJ[0.5, 1.0]").unwrap().parse().unwrap();
    assert!((result - 0.8551653096792676).abs() < 1e-6, "got {}", result);

    // AngerJ[1.5, 2.0] ≈ 0.4036548767715761
    let result: f64 = interpret("AngerJ[1.5, 2.0]").unwrap().parse().unwrap();
    assert!(
      (result - 0.40365487677157613).abs() < 1e-6,
      "got {}",
      result
    );
  }

  #[test]
  fn negative_order() {
    // AngerJ[-1, 3.0] = BesselJ[-1, 3.0] = -BesselJ[1, 3.0]
    let result: f64 = interpret("AngerJ[-1, 3.0]").unwrap().parse().unwrap();
    assert!(
      (result - (-0.33905895852593644)).abs() < 1e-8,
      "got {}",
      result
    );
  }

  #[test]
  fn n_wrapper() {
    // N[AngerJ[0, 5]] should force numeric evaluation
    let result: f64 = interpret("N[AngerJ[0, 5]]").unwrap().parse().unwrap();
    assert!(
      (result - (-0.17759677131433826)).abs() < 1e-8,
      "got {}",
      result
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    // Integer order reduces to BesselJ
    assert_eq!(interpret("AngerJ[0, x]").unwrap(), "BesselJ[0, x]");
    // Non-integer symbolic stays as AngerJ
    assert_eq!(interpret("AngerJ[n, z]").unwrap(), "AngerJ[n, z]");
  }

  #[test]
  fn wrong_arg_count() {
    // AngerJ with wrong number of args should return unevaluated
    assert_eq!(interpret("AngerJ[1]").unwrap(), "AngerJ[1]");
  }
}

#[cfg(test)]
mod weber_e_tests {
  use super::*;

  #[test]
  fn at_zero() {
    // WeberE[0, 0] = 0
    assert_eq!(interpret("WeberE[0, 0]").unwrap(), "0");
  }

  #[test]
  fn integer_order_numeric() {
    // WeberE[0, 1.0] ≈ -0.5686566270483014
    let result: f64 = interpret("WeberE[0, 1.0]").unwrap().parse().unwrap();
    assert!(
      (result - (-0.5686566270483014)).abs() < 1e-6,
      "got {}",
      result
    );

    // WeberE[1, 3.0] ≈ -0.38348979681886397
    let result: f64 = interpret("WeberE[1, 3.0]").unwrap().parse().unwrap();
    assert!(
      (result - (-0.38348979681886397)).abs() < 1e-6,
      "got {}",
      result
    );
  }

  #[test]
  fn non_integer_order() {
    // WeberE[0.5, 1.0] ≈ 0.09950754264004202
    let result: f64 = interpret("WeberE[0.5, 1.0]").unwrap().parse().unwrap();
    assert!(
      (result - 0.09950754264004202).abs() < 1e-6,
      "got {}",
      result
    );

    // WeberE[1.5, 2.0] ≈ 0.10882159808007938
    let result: f64 = interpret("WeberE[1.5, 2.0]").unwrap().parse().unwrap();
    assert!(
      (result - 0.10882159808007938).abs() < 1e-6,
      "got {}",
      result
    );
  }

  #[test]
  fn non_integer_at_zero() {
    // WeberE[0.5, 0] = (1 - Cos[0.5*Pi]) / (0.5*Pi) = 1 / (Pi/2) = 2/Pi ≈ 0.6366
    let result: f64 = interpret("WeberE[0.5, 0]").unwrap().parse().unwrap();
    assert!(
      (result - 2.0 / std::f64::consts::PI).abs() < 1e-6,
      "got {}",
      result
    );
  }

  #[test]
  fn n_wrapper() {
    // N[WeberE[0, 5]] should force numeric evaluation
    let result: f64 = interpret("N[WeberE[0, 5]]").unwrap().parse().unwrap();
    assert!(result.is_finite(), "got {}", result);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("WeberE[n, z]").unwrap(), "WeberE[n, z]");
  }

  #[test]
  fn wrong_arg_count() {
    assert_eq!(interpret("WeberE[1]").unwrap(), "WeberE[1]");
  }
}

#[cfg(test)]
mod wigner_d_tests {
  use super::*;

  #[test]
  fn j1_identity() {
    // WignerD[{1, 0, 0}, 0] = d^1_{0,0}(0) = 1
    let result: f64 = interpret("WignerD[{1, 0, 0}, 0.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.0).abs() < 1e-10, "got {}", result);
  }

  #[test]
  fn j1_at_pi() {
    // d^1_{0,0}(Pi) = cos(Pi) = -1
    let result: f64 = interpret("WignerD[{1, 0, 0}, 3.141592653589793]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - (-1.0)).abs() < 1e-10, "got {}", result);
  }

  #[test]
  fn j_half() {
    // d^{1/2}_{1/2, 1/2}(theta) = cos(theta/2)
    let theta: f64 = 1.0;
    let expected = (theta / 2.0).cos();
    let result: f64 =
      interpret(&format!("WignerD[{{1/2, 1/2, 1/2}}, {}]", theta))
        .unwrap()
        .parse()
        .unwrap();
    assert!(
      (result - expected).abs() < 1e-10,
      "expected {}, got {}",
      expected,
      result
    );
  }

  #[test]
  fn j_half_off_diag() {
    // d^{1/2}_{1/2, -1/2}(theta) = -sin(theta/2)
    let theta: f64 = 1.0;
    let expected = -(theta / 2.0).sin();
    let result: f64 =
      interpret(&format!("WignerD[{{1/2, 1/2, -1/2}}, {}]", theta))
        .unwrap()
        .parse()
        .unwrap();
    assert!(
      (result - expected).abs() < 1e-10,
      "expected {}, got {}",
      expected,
      result
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("WignerD[{j, m1, m2}, theta]").unwrap(),
      "WignerD[{j, m1, m2}, theta]"
    );
  }

  #[test]
  fn wrong_arg_count() {
    assert_eq!(
      interpret("WignerD[{1, 0, 0}]").unwrap(),
      "WignerD[{1, 0, 0}]"
    );
  }
}

mod cantor_staircase {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("CantorStaircase[0]").unwrap(), "0");
  }

  #[test]
  fn one() {
    assert_eq!(interpret("CantorStaircase[1]").unwrap(), "1");
  }

  #[test]
  fn one_third() {
    assert_eq!(interpret("CantorStaircase[1/3]").unwrap(), "1/2");
  }

  #[test]
  fn two_thirds() {
    assert_eq!(interpret("CantorStaircase[2/3]").unwrap(), "1/2");
  }

  #[test]
  fn one_half() {
    // 1/2 is in the middle third [1/3, 2/3], so value is 1/2
    assert_eq!(interpret("CantorStaircase[1/2]").unwrap(), "1/2");
  }

  #[test]
  fn one_fourth() {
    assert_eq!(interpret("CantorStaircase[1/4]").unwrap(), "1/3");
  }

  #[test]
  fn one_ninth() {
    assert_eq!(interpret("CantorStaircase[1/9]").unwrap(), "1/4");
  }

  #[test]
  fn two_ninths() {
    assert_eq!(interpret("CantorStaircase[2/9]").unwrap(), "1/4");
  }

  #[test]
  fn seven_ninths() {
    assert_eq!(interpret("CantorStaircase[7/9]").unwrap(), "3/4");
  }

  #[test]
  fn eight_ninths() {
    assert_eq!(interpret("CantorStaircase[8/9]").unwrap(), "3/4");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("CantorStaircase[-1]").unwrap(), "0");
  }

  #[test]
  fn greater_than_one() {
    assert_eq!(interpret("CantorStaircase[2]").unwrap(), "1");
  }

  #[test]
  fn numeric_float() {
    assert_eq!(interpret("CantorStaircase[0.5]").unwrap(), "0.5");
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("CantorStaircase[x]").unwrap(),
      "CantorStaircase[x]"
    );
  }
}

mod congruent {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("Congruent[a, b]").unwrap(), "a \u{2261} b");
  }

  #[test]
  fn three_args() {
    assert_eq!(
      interpret("Congruent[a, b, c]").unwrap(),
      "a \u{2261} b \u{2261} c"
    );
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("Congruent[a]").unwrap(), "Congruent[a]");
  }
}

mod bond {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("Bond[{\"Single\", 1, 2}]").unwrap(),
      "Bond[{Single, 1, 2}]"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Bond[{\"Single\", 1, 2}]]").unwrap(), "Bond");
  }
}

mod planar_graph {
  use super::*;

  #[test]
  fn from_rules() {
    assert_eq!(
      interpret("PlanarGraph[{1 -> 2, 2 -> 3, 3 -> 1}]").unwrap(),
      "Graph[{1, 2, 3}, {DirectedEdge[1, 2], DirectedEdge[2, 3], DirectedEdge[3, 1]}]"
    );
  }

  #[test]
  fn vertex_list() {
    assert_eq!(
      interpret("VertexList[PlanarGraph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[PlanarGraph[{1 -> 2}]]").unwrap(), "Graph");
  }
}

mod colon {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("Colon[a, b]").unwrap(), "a \u{2236} b");
  }

  #[test]
  fn three_args() {
    assert_eq!(
      interpret("Colon[a, b, c]").unwrap(),
      "a \u{2236} b \u{2236} c"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(interpret("Colon[1, 2]").unwrap(), "1 \u{2236} 2");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("Colon[a]").unwrap(), "Colon[a]");
  }

  #[test]
  fn head() {
    assert_eq!(interpret("Head[Colon[a, b]]").unwrap(), "Colon");
  }
}

mod bandpass_filter {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("BandpassFilter[{1, 2, 3, 4, 5}, {0.1, 0.3}]").unwrap(),
      "{0.18547982385040732, 0.288059447503783, 0.4244918102099795, 0.5609241729161761, 0.6635037965695517}"
    );
  }

  #[test]
  fn with_order() {
    assert_eq!(
      interpret("BandpassFilter[{1, 2, 3, 4, 5}, {0.1, 0.3}, 3]").unwrap(),
      "{0.07991129198211216, 0.14898970746732165, 0.22348456120098245, 0.2979794149346433, 0.3670578304198528}"
    );
  }

  #[test]
  fn order_1() {
    assert_eq!(
      interpret("BandpassFilter[{1, 2, 3, 4, 5}, {0.1, 0.3}, 1]").unwrap(),
      "{0.06366197723675814, 0.12732395447351627, 0.1909859317102744, 0.25464790894703254, 0.3183098861837907}"
    );
  }

  #[test]
  fn even_order() {
    assert_eq!(
      interpret("BandpassFilter[{1, 2, 3, 4, 5}, {0.1, 0.3}, 4]").unwrap(),
      "{0.11353554513578962, 0.167668935248101, 0.270666950561557, 0.3789337307861797, 0.4819317460996357}"
    );
  }

  #[test]
  fn eight_elements() {
    assert_eq!(
      interpret("BandpassFilter[{1, 2, 3, 4, 5, 6, 7, 8}, {0.1, 0.3}]")
        .unwrap(),
      "{0.3161491615585342, 0.43414813964807336, 0.6126020827245306, 0.8301409615621332, 1.061990802805853, 1.2938406440495724, 1.5113795228871751, 1.6898334659636323}"
    );
  }

  #[test]
  fn single_element() {
    assert_eq!(
      interpret("BandpassFilter[{1}, {0.1, 0.3}]").unwrap(),
      "{0.06366197723675814}"
    );
  }

  #[test]
  fn constant_input() {
    assert_eq!(
      interpret("BandpassFilter[{1, 1, 1}, {0.1, 0.3}]").unwrap(),
      "{0.07449485373366081, 0.07449485373366081, 0.07449485373366081}"
    );
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    assert_eq!(
      interpret("BandpassFilter[{a, b, c}, {0.1, 0.3}]").unwrap(),
      "BandpassFilter[{a, b, c}, {0.1, 0.3}]"
    );
  }
}

mod lowpass_filter {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("LowpassFilter[{1, 2, 3, 4, 5}, 0.3]").unwrap(),
      "{1.3128492422860063, 2.0366239766680576, 3., 3.963376023331943, 4.687150757713994}"
    );
  }

  #[test]
  fn constant_input() {
    assert_eq!(
      interpret("LowpassFilter[{1, 1, 1, 1, 1}, 0.3]").unwrap(),
      "{1., 1., 1., 1., 1.}"
    );
  }
}

mod highpass_filter {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("HighpassFilter[{1, 2, 3, 4, 5}, 0.3]").unwrap(),
      "{0.7198793058278876, 1.565448565047395, 2.359894452882456, 3.154340340717517, 3.999909599937024}"
    );
  }
}

mod bandstop_filter {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(
      interpret("BandstopFilter[{1, 2, 3, 4, 5}, {0.1, 0.3}]").unwrap(),
      "{0.8145201761495926, 1.711940552496217, 2.5755081897900203, 3.439075827083824, 4.336496203430448}"
    );
  }
}
