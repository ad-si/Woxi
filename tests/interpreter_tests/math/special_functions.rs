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

  #[test]
  fn imaginary_multiple_of_pi() {
    // Gudermannian[2*Pi*I] stays unevaluated (matches wolframscript).
    assert_eq!(
      interpret("Gudermannian[2 Pi I]").unwrap(),
      "Gudermannian[(2*I)*Pi]"
    );
  }

  // Regression (mathics test_hyperbolic.py:20): at the simple poles
  // `(2n+1) π i / 2` Gudermannian returns DirectedInfinity[±I]; sign
  // alternates with k mod 4 (k = 2n+1).
  #[test]
  fn pole_3_pi_i_over_2() {
    assert_eq!(
      interpret("Gudermannian[6/4 Pi I]").unwrap(),
      "DirectedInfinity[-I]"
    );
  }

  #[test]
  fn pole_pi_i_over_2() {
    assert_eq!(
      interpret("Gudermannian[Pi I / 2]").unwrap(),
      "DirectedInfinity[I]"
    );
  }

  #[test]
  fn pole_neg_pi_i_over_2() {
    assert_eq!(
      interpret("Gudermannian[-Pi I / 2]").unwrap(),
      "DirectedInfinity[-I]"
    );
  }

  #[test]
  fn pole_5_pi_i_over_2() {
    assert_eq!(
      interpret("Gudermannian[5 Pi I / 2]").unwrap(),
      "DirectedInfinity[I]"
    );
  }

  #[test]
  fn pole_7_pi_i_over_2() {
    assert_eq!(
      interpret("Gudermannian[7 Pi I / 2]").unwrap(),
      "DirectedInfinity[-I]"
    );
  }

  #[test]
  fn pole_neg_3_pi_i_over_2() {
    assert_eq!(
      interpret("Gudermannian[-3 Pi I / 2]").unwrap(),
      "DirectedInfinity[I]"
    );
  }

  // Not a pole: coefficient is `1/4`, not an odd-half-integer.
  #[test]
  fn imaginary_pi_over_4_stays_symbolic() {
    assert_eq!(
      interpret("Gudermannian[Pi I / 4]").unwrap(),
      "Gudermannian[I/4*Pi]"
    );
  }
}

mod inverse_gudermannian {
  use super::*;

  #[test]
  fn odd_symmetry_on_real_argument() {
    // Regression: f64 atanh is not bit-exactly odd, so the raw formula
    // 2 * atanh(tan(x/2)) yielded different last-bit values for +x and -x,
    // making `InverseGudermannian[-.5] == -InverseGudermannian[.5]` False.
    assert_eq!(
      interpret("InverseGudermannian[-.5] == -InverseGudermannian[.5]")
        .unwrap(),
      "True"
    );
    // And the individual values should be exact negatives of one another.
    assert_eq!(
      interpret("InverseGudermannian[0.5]").unwrap(),
      "0.5222381032784403"
    );
    assert_eq!(
      interpret("InverseGudermannian[-0.5]").unwrap(),
      "-0.5222381032784403"
    );
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

  // Closed-form rules for half-integer orders:
  // BesselJ[1/2, z]  = Sqrt[2/(Pi z)] Sin[z]
  // BesselJ[-1/2, z] = Sqrt[2/(Pi z)] Cos[z]
  #[test]
  fn half_order_sin_closed_form() {
    assert_eq!(
      interpret("BesselJ[1/2, x]").unwrap(),
      "(Sqrt[2/Pi]*Sin[x])/Sqrt[x]"
    );
  }

  #[test]
  fn negative_half_order_cos_closed_form() {
    assert_eq!(
      interpret("BesselJ[-1/2, x]").unwrap(),
      "(Sqrt[2/Pi]*Cos[x])/Sqrt[x]"
    );
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

  // At an inexact origin the pole behaves like the exact case: K_0(0.) =
  // Infinity, and every other order is ComplexInfinity (rather than the
  // Indeterminate/0 the numeric series produced). Per wolframscript.
  #[test]
  fn at_real_zero() {
    assert_eq!(interpret("BesselK[0.0, 0.0]").unwrap(), "Infinity");
    assert_eq!(interpret("BesselK[1, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("BesselK[2, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("BesselK[1.5, 0.0]").unwrap(), "ComplexInfinity");
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

  // At an inexact origin the pole behaves like the exact case: Y_0(0.) =
  // -Infinity, and every other order is ComplexInfinity (rather than the
  // Indeterminate/0 the numeric series produced). Per wolframscript.
  #[test]
  fn at_real_zero() {
    assert_eq!(interpret("BesselY[0.0, 0.0]").unwrap(), "-Infinity");
    assert_eq!(interpret("BesselY[1, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("BesselY[2, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("BesselY[0.5, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("BesselY[-1.0, 0.0]").unwrap(), "ComplexInfinity");
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
  fn euler_transform_sign_canonical() {
    // After the Euler transformation collapses to `(1-z)^(-1) · (-3·X)`,
    // wolframscript's canonical surface form negates both factors so the
    // numeric coefficient is positive: `(z-1)^(-1) · 3·X`.
    assert_eq!(
      interpret("Hypergeometric2F1[2, 3, 4, x]").unwrap(),
      "(3*(-2*x + x^2 - 2*Log[1 - x] + 2*x*Log[1 - x]))/((-1 + x)*x^3)"
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

  #[test]
  fn complex_args() {
    // wolframscript: Hypergeometric2F1[2 + I, -I, 3/4, 0.5 - 0.5 I]
    //   ≈ -0.97216657 - 0.18165874 I
    let s =
      interpret("Hypergeometric2F1[2 + I, -I, 3/4, 0.5 - 0.5 I]").unwrap();
    assert!(
      s.starts_with("-0.972166571361907") && s.contains(" - 0.181658741475730"),
      "got: {}",
      s
    );
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
    assert!((result - 1.71828182845904535).abs() < 1e-10);
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
  fn complex_a_real_z() {
    // wolframscript: Hypergeometric1F1[2 + I, 2, 0.5] ≈ 1.61833 + 0.37926 I
    let s = interpret("Hypergeometric1F1[2 + I, 2, 0.5]").unwrap();
    assert!(
      s.starts_with("1.6183312284015")
        && s.contains(" + 0.3792577715004")
        && s.ends_with("*I"),
      "got: {}",
      s
    );
  }

  // 1F1[a, a, z] = E^z — Pochhammer factors cancel, leaving the exponential
  // series. With a matrix argument, this applies element-wise.
  #[test]
  fn same_upper_lower_reduces_to_exp_matrix() {
    assert_eq!(
      interpret("Hypergeometric1F1[1, 1, {{1, 0}, {0, 1}}]").unwrap(),
      "{{E, 1}, {1, E}}"
    );
  }

  #[test]
  fn same_upper_lower_symbolic_z() {
    assert_eq!(interpret("Hypergeometric1F1[a, a, x]").unwrap(), "E^x");
  }

  // A non-positive integer first argument terminates the series, giving a
  // degree-n polynomial in z — valid for any (compatible) b, including
  // symbolic. Verified against wolframscript.
  #[test]
  fn terminating_polynomial_negative_a() {
    assert_eq!(interpret("Hypergeometric1F1[-1, 2, x]").unwrap(), "1 - x/2");
    assert_eq!(
      interpret("Hypergeometric1F1[-2, 3, x]").unwrap(),
      "1 - (2*x)/3 + x^2/12"
    );
    assert_eq!(
      interpret("Hypergeometric1F1[-3, 1, x]").unwrap(),
      "1 - 3*x + (3*x^2)/2 - x^3/6"
    );
    // Symbolic lower parameter.
    assert_eq!(interpret("Hypergeometric1F1[-1, b, x]").unwrap(), "1 - x/b");
    // Numeric argument gives an exact value.
    assert_eq!(interpret("Hypergeometric1F1[-3, 2, 2]").unwrap(), "-1/3");
    // a == b a non-positive integer is a truncated exponential, not E^z.
    assert_eq!(
      interpret("Hypergeometric1F1[-2, -2, x]").unwrap(),
      "1 + x + x^2/2"
    );
  }

  // Closed-form expansion for `1F1[positive integer a, 1, z]` via the
  // Laguerre/Kummer identity — matches wolframscript.
  #[test]
  fn closed_form_a_eq_2_b_eq_1() {
    assert_eq!(
      interpret("Hypergeometric1F1[2, 1, x]").unwrap(),
      "E^x*(1 + x)"
    );
  }

  #[test]
  fn closed_form_a_eq_3_b_eq_1() {
    assert_eq!(
      interpret("Hypergeometric1F1[3, 1, x]").unwrap(),
      "E^x*(1 + 2*x + x^2/2)"
    );
  }

  #[test]
  fn closed_form_a_eq_4_b_eq_1() {
    assert_eq!(
      interpret("Hypergeometric1F1[4, 1, x]").unwrap(),
      "E^x*(1 + 3*x + (3*x^2)/2 + x^3/6)"
    );
  }

  // PFQ with the same multisets of upper/lower parameters also collapses
  // to E^z — the Pochhammer ratio is identically 1.
  #[test]
  fn pfq_matching_params_collapses_to_exp() {
    assert_eq!(interpret("HypergeometricPFQ[{2}, {2}, 1]").unwrap(), "E");
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

  #[test]
  fn a_zero_gives_one() {
    // HypergeometricU[0, b, z] = 1 for any b and z
    assert_eq!(interpret("HypergeometricU[0, b, z]").unwrap(), "1");
  }
}

mod scorer_functions {
  use super::*;

  fn num(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  // Exact (non-zero) arguments stay symbolic; the value at 0 has a closed form.
  #[test]
  fn exact_forms() {
    assert_eq!(interpret("ScorerGi[1]").unwrap(), "ScorerGi[1]");
    assert_eq!(
      interpret("ScorerGi[0]").unwrap(),
      "1/(3*3^(1/6)*Gamma[2/3])"
    );
    assert_eq!(
      interpret("ScorerHi[0]").unwrap(),
      "2/(3*3^(1/6)*Gamma[2/3])"
    );
  }

  #[test]
  fn hi_values() {
    assert!((num("N[ScorerHi[0]]") - 0.40995108496400046).abs() < 1e-9);
    assert!((num("N[ScorerHi[1]]") - 0.9722051551424336).abs() < 1e-9);
    assert!((num("N[ScorerHi[2]]") - 3.129141434324205).abs() < 1e-9);
    assert!((num("N[ScorerHi[-1]]") - 0.22066960679295983).abs() < 1e-9);
  }

  #[test]
  fn gi_values() {
    assert!((num("N[ScorerGi[0]]") - 0.20497554248200023).abs() < 1e-9);
    assert!((num("N[ScorerGi[1]]") - 0.2352184398104379).abs() < 1e-9);
    assert!((num("N[ScorerGi[-1]]") - (-0.11667221729601539)).abs() < 1e-9);
  }

  // Gi(x) + Hi(x) = Bi(x) (DLMF 9.12.3).
  #[test]
  fn gi_plus_hi_is_airy_bi() {
    let sum = num("N[ScorerGi[2] + ScorerHi[2]]");
    let bi = num("N[AiryBi[2]]");
    assert!((sum - bi).abs() < 1e-9, "Gi+Hi {sum} vs Bi {bi}");
  }
}

mod carlson_integrals {
  use super::*;

  fn num(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  // Exact (integer/rational) arguments stay symbolic, matching wolframscript.
  #[test]
  fn exact_stays_symbolic() {
    assert_eq!(interpret("CarlsonRC[1, 2]").unwrap(), "CarlsonRC[1, 2]");
    assert_eq!(
      interpret("CarlsonRF[1, 2, 3]").unwrap(),
      "CarlsonRF[1, 2, 3]"
    );
    assert_eq!(
      interpret("CarlsonRJ[1, 2, 3, 4]").unwrap(),
      "CarlsonRJ[1, 2, 3, 4]"
    );
  }

  // R_C(x, y) — degenerate integral. R_C(1, 2) = Pi/4, R_C(2, 2) = 1/Sqrt[2].
  #[test]
  fn rc_values() {
    assert!(
      (num("N[CarlsonRC[1, 2]]") - std::f64::consts::FRAC_PI_4).abs() < 1e-9
    );
    assert!((num("N[CarlsonRC[1, 4]]") - 0.6045997880780727).abs() < 1e-9);
    assert!((num("N[CarlsonRC[2, 2]]") - 0.7071067811865475).abs() < 1e-9);
  }

  // R_F(x, y, z) — first kind. R_F(1, 1, 1) = 1.
  #[test]
  fn rf_values() {
    assert!((num("N[CarlsonRF[1, 2, 3]]") - 0.7269459354689082).abs() < 1e-9);
    assert!((num("N[CarlsonRF[1, 1, 1]]") - 1.0).abs() < 1e-9);
    // A direct machine-real argument numericizes without an explicit N.
    assert!((num("CarlsonRF[1.0, 2, 3]") - 0.7269459354689082).abs() < 1e-9);
  }

  #[test]
  fn rd_value() {
    assert!((num("N[CarlsonRD[1, 2, 3]]") - 0.29046028102899024).abs() < 1e-9);
  }

  #[test]
  fn rj_value() {
    assert!(
      (num("N[CarlsonRJ[1, 2, 3, 4]]") - 0.23984809974956603).abs() < 1e-9
    );
  }

  #[test]
  fn rg_value() {
    assert!((num("N[CarlsonRG[1, 2, 3]]") - 1.4018470999908947).abs() < 1e-9);
  }

  // R_D(x, y, z) = R_J(x, y, z, z): consistency between the two kernels.
  #[test]
  fn rd_rj_consistency() {
    let rd = num("N[CarlsonRD[1, 2, 3]]");
    let rj = num("N[CarlsonRJ[1, 2, 3, 3]]");
    assert!((rd - rj).abs() < 1e-9, "RD {rd} vs RJ {rj}");
  }

  // R_E(x, y) is the complete integral (4/Pi) R_G(0, x, y); R_E(1, 1) = 1.
  #[test]
  fn re_values() {
    assert_eq!(interpret("CarlsonRE[1, 2]").unwrap(), "CarlsonRE[1, 2]");
    assert!((num("N[CarlsonRE[1, 1]]") - 1.0).abs() < 1e-9);
    assert!((num("N[CarlsonRE[1, 2]]") - 1.2160067234249798).abs() < 1e-9);
    assert!((num("N[CarlsonRE[2, 3]]") - 1.5771482616833925).abs() < 1e-9);
  }

  // R_E(x, y) == (4/Pi) R_G(0, x, y): consistency with the R_G kernel.
  #[test]
  fn re_rg_consistency() {
    let re = num("N[CarlsonRE[2, 5]]");
    let rg = num("N[(4/Pi) CarlsonRG[0, 2, 5]]");
    assert!((re - rg).abs() < 1e-9, "RE {re} vs (4/Pi)RG {rg}");
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
    assert!((result - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
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

  #[test]
  fn numeric_negative_audit_case() {
    // EllipticNomeQ[-2.] = -0.06823782774533839 (audit)
    let result: f64 = interpret("EllipticNomeQ[-2.]").unwrap().parse().unwrap();
    assert!((result - (-0.06823782774533839)).abs() < 1e-12);
  }

  #[test]
  fn numeric_negative_one() {
    // EllipticNomeQ[-1.] = -EllipticNomeQ[1/2] ≈ -0.04321391826377226
    let result: f64 = interpret("EllipticNomeQ[-1.]").unwrap().parse().unwrap();
    assert!((result - (-0.04321391826377226)).abs() < 1e-12);
  }

  #[test]
  fn numeric_negative_small() {
    // EllipticNomeQ[-0.5] (negative, finite). Should match wolframscript.
    // wolframscript: -0.02531991336628519
    let result: f64 =
      interpret("EllipticNomeQ[-0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.02531991336628519)).abs() < 1e-12);
  }
}

mod inverse_elliptic_nome_q {
  use super::*;

  // InverseEllipticNomeQ[q] inverts EllipticNomeQ: it returns the parameter m
  // with EllipticNomeQ[m] == q. Exact rationals stay symbolic.
  #[test]
  fn exact_stays_symbolic() {
    assert_eq!(
      interpret("InverseEllipticNomeQ[1/2]").unwrap(),
      "InverseEllipticNomeQ[1/2]"
    );
    assert_eq!(
      interpret("InverseEllipticNomeQ[q]").unwrap(),
      "InverseEllipticNomeQ[q]"
    );
  }

  // Elementary boundaries: q = 0 -> m = 0, q = 1 -> m = 1.
  #[test]
  fn boundaries() {
    assert_eq!(interpret("InverseEllipticNomeQ[0]").unwrap(), "0");
    assert_eq!(interpret("InverseEllipticNomeQ[1]").unwrap(), "1");
  }

  #[test]
  fn numeric_values() {
    let m: f64 = interpret("N[InverseEllipticNomeQ[1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((m - 0.9999895221373106).abs() < 1e-9, "got {m}");

    let m: f64 = interpret("N[InverseEllipticNomeQ[1/4]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((m - 0.987135606911965).abs() < 1e-9, "got {m}");

    let m: f64 = interpret("N[InverseEllipticNomeQ[1/3]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((m - 0.9979950089217557).abs() < 1e-9, "got {m}");

    // A direct machine-real argument numericizes without an explicit N.
    let m: f64 = interpret("InverseEllipticNomeQ[0.04321391826377226]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((m - 0.5).abs() < 1e-9, "got {m}");
  }

  // Round-trip: InverseEllipticNomeQ inverts EllipticNomeQ.
  #[test]
  fn round_trip() {
    let m: f64 = interpret("N[InverseEllipticNomeQ[1/4]]")
      .unwrap()
      .parse()
      .unwrap();
    let q: f64 = interpret(&format!("N[EllipticNomeQ[{m}]]"))
      .unwrap()
      .parse()
      .unwrap();
    assert!((q - 0.25).abs() < 1e-9, "got {q}");
  }
}

mod dedekind_eta {
  use super::*;

  fn parse_complex(s: &str) -> (f64, f64) {
    // Parses Wolfram-style complex strings produced by Woxi, e.g.
    // "0.572 + 0.153*I", "0.572 - 0.153*I", "0.572", "0.153*I".
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix("*I") {
      if let Some(idx) = stripped.rfind(" + ") {
        let re = stripped[..idx].parse::<f64>().unwrap();
        let im = stripped[idx + 3..].parse::<f64>().unwrap();
        return (re, im);
      } else if let Some(idx) = stripped.rfind(" - ") {
        let re = stripped[..idx].parse::<f64>().unwrap();
        let im = stripped[idx + 3..].parse::<f64>().unwrap();
        return (re, -im);
      } else {
        return (0.0, stripped.parse::<f64>().unwrap());
      }
    }
    if s == "I" {
      return (0.0, 1.0);
    }
    (s.parse::<f64>().unwrap(), 0.0)
  }

  #[test]
  fn numeric_1_plus_2i() {
    // DedekindEta[1 + 2.*I] = 0.5721978275379304 + 0.15331994579963126*I
    let result = interpret("DedekindEta[1 + 2.*I]").unwrap();
    let (re, im) = parse_complex(&result);
    assert!((re - 0.5721978275379304).abs() < 1e-10);
    assert!((im - 0.15331994579963126).abs() < 1e-10);
  }

  #[test]
  fn numeric_pure_imag_unit() {
    // DedekindEta[1.0*I] = 0.7682254223260567
    let result: f64 = interpret("DedekindEta[1.0*I]").unwrap().parse().unwrap();
    assert!((result - 0.7682254223260567).abs() < 1e-10);
  }

  #[test]
  fn numeric_pure_imag_2() {
    // DedekindEta[2.0*I] = 0.592382781332416
    let result: f64 = interpret("DedekindEta[2.0*I]").unwrap().parse().unwrap();
    assert!((result - 0.592382781332416).abs() < 1e-10);
  }

  #[test]
  fn numeric_small_real_part() {
    // DedekindEta[0.1 + 1.0*I] = 0.7682606134774355 + 0.01926994165837382*I
    let result = interpret("DedekindEta[0.1 + 1.0*I]").unwrap();
    let (re, im) = parse_complex(&result);
    assert!((re - 0.7682606134774355).abs() < 1e-10);
    assert!((im - 0.01926994165837382).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("DedekindEta[tau]").unwrap(), "DedekindEta[tau]");
  }

  #[test]
  fn exact_imag_unit() {
    // DedekindEta[I] = Gamma[1/4]/(2*Pi^(3/4))
    assert_eq!(
      interpret("DedekindEta[I]").unwrap(),
      "Gamma[1/4]/(2*Pi^(3/4))"
    );
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

  // Exact (integer/rational) arguments stay symbolic — they are NOT
  // numericized automatically (only N[...] or an inexact argument does that).
  #[test]
  fn exact_args_stay_symbolic() {
    assert_eq!(
      interpret("EllipticTheta[3, 0, 1/2]").unwrap(),
      "EllipticTheta[3, 0, 1/2]"
    );
    assert_eq!(
      interpret("EllipticTheta[2, 0, 1/3]").unwrap(),
      "EllipticTheta[2, 0, 1/3]"
    );
    assert_eq!(
      interpret("EllipticTheta[1, 5, 1/2]").unwrap(),
      "EllipticTheta[1, 5, 1/2]"
    );
  }

  // An inexact (machine) argument triggers numeric evaluation.
  #[test]
  fn inexact_arg_numericizes() {
    let result: f64 = interpret("EllipticTheta[3, 0, 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 2.128936827211877).abs() < 1e-10);
  }

  // theta1 is odd in z, so theta1(0, q) is exactly 0 for any q.
  #[test]
  fn theta1_at_z_zero_exact() {
    assert_eq!(interpret("EllipticTheta[1, 0, 1/2]").unwrap(), "0");
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
  fn symbolic_s_at_one() {
    // Wolfram keeps PolyLog[s, 1] unevaluated for symbolic s.
    assert_eq!(interpret("PolyLog[s, 1]").unwrap(), "PolyLog[s, 1]");
  }

  #[test]
  fn at_one_half_s2() {
    // PolyLog[2, 1/2] = Pi^2/12 - Log[2]^2/2
    assert_eq!(
      interpret("PolyLog[2, 1/2]").unwrap(),
      "Pi^2/12 - Log[2]^2/2"
    );
  }

  #[test]
  fn at_one_half_s3() {
    // Audit case: PolyLog[3, 1/2] = (-2*Pi^2*Log[2] + 4*Log[2]^3 + 21*Zeta[3])/24
    assert_eq!(
      interpret("PolyLog[3, 1/2]").unwrap(),
      "(-2*Pi^2*Log[2] + 4*Log[2]^3 + 21*Zeta[3])/24"
    );
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

  // Half-integer / real degree uses the hypergeometric closed form
  //   P_ν(x) = Hypergeometric2F1[-ν, ν + 1, 1, (1 - x) / 2]
  // and matches wolframscript bit-for-bit.
  #[test]
  fn half_integer_degree_rational() {
    assert_eq!(
      interpret("LegendreP[5/2, 1.5]").unwrap(),
      "4.177619138927457"
    );
  }

  #[test]
  fn half_integer_degree_inside_unit_interval() {
    assert_eq!(
      interpret("LegendreP[3/2, 0.7]").unwrap(),
      "0.47591713809060504"
    );
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

  #[test]
  fn associated_legendre() {
    assert_eq!(
      interpret("LegendreP[2, 1, x]").unwrap(),
      "-3*Sqrt[1 - x^2]*x"
    );
  }

  #[test]
  fn associated_legendre_m0() {
    assert_eq!(interpret("LegendreP[3, 0, x]").unwrap(), "(-3*x + 5*x^3)/2");
  }

  #[test]
  fn associated_legendre_m_gt_n() {
    assert_eq!(interpret("LegendreP[2, 3, x]").unwrap(), "0");
  }

  #[test]
  fn associated_complex_branch_z_gt_one() {
    // wolframscript: LegendreP[1.6, 3.1, 1.5]
    //   ≈ -0.30399816148959324 - 1.9193688525633503*I
    let s = interpret("LegendreP[1.6, 3.1, 1.5]").unwrap();
    assert!(
      s.starts_with("-0.30399816148959") && s.contains(" - 1.9193688525633"),
      "got: {}",
      s
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

  #[test]
  fn complex_type_2_z_gt_one() {
    // wolframscript: LegendreQ[5/2, 1.5]
    //   ≈ 0.036210967179686804 - 6.5621887981753035*I
    let s = interpret("LegendreQ[5/2, 1.5]").unwrap();
    assert!(
      s.starts_with("0.0362109671796") && s.contains(" - 6.56218879817530"),
      "got: {}",
      s
    );
  }

  #[test]
  fn associated_real_z_lt_one() {
    // 3-arg form, |z| < 1: result is real (Ferrers Q on the cut).
    // wolframscript: LegendreQ[1.75, 1.4, 0.53] ≈ 2.0549890785760923
    let s = interpret("LegendreQ[1.75, 1.4, 0.53]").unwrap();
    let val: f64 = s.parse().unwrap();
    assert!((val - 2.0549890785760923).abs() < 1e-10, "got: {}", s);
  }

  #[test]
  fn associated_complex_branch_z_gt_one() {
    // 3-arg form, |z| > 1: complex result with the same Ferrers identity
    // (the (1+z)^(μ/2) prefactor needs the principal-branch log).
    // wolframscript: LegendreQ[1.6, 3.1, 1.5]
    //   ≈ -1.7193129097069424 - 7.7027327978267826*I
    let s = interpret("LegendreQ[1.6, 3.1, 1.5]").unwrap();
    assert!(
      s.starts_with("-1.7193129097069") && s.contains(" - 7.70273279782677"),
      "got: {}",
      s
    );
  }

  #[test]
  fn fractional_order_real_z_lt_one() {
    // wolframscript: LegendreQ[1/3, 0.5] ≈ -0.03995329475988949
    let result: f64 =
      interpret("LegendreQ[1/3, 0.5]").unwrap().parse().unwrap();
    assert!(
      (result - (-0.03995329475988949)).abs() < 1e-10,
      "got {}",
      result
    );
  }

  #[test]
  fn fractional_order_real_z_half() {
    // wolframscript: LegendreQ[1/2, 0.5] ≈ -0.26559640763727543
    let result: f64 =
      interpret("LegendreQ[1/2, 0.5]").unwrap().parse().unwrap();
    assert!(
      (result - (-0.26559640763727543)).abs() < 1e-10,
      "got {}",
      result
    );
  }

  #[test]
  fn fractional_real_order_real_z_lt_one() {
    // wolframscript: LegendreQ[2.5, 0.3] ≈ -0.10341858691052352
    let result: f64 =
      interpret("LegendreQ[2.5, 0.3]").unwrap().parse().unwrap();
    assert!(
      (result - (-0.10341858691052352)).abs() < 1e-8,
      "got {}",
      result
    );
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

  // Odd n at a half-integer has a closed pi-power form (via the
  // polygamma-Hurwitz relation PolyGamma[n, z] = (-1)^(n+1) n! Zeta[n+1, z]).
  #[test]
  fn polygamma_odd_n_half_integer() {
    assert_eq!(interpret("PolyGamma[1, 1/2]").unwrap(), "Pi^2/2");
    assert_eq!(interpret("PolyGamma[1, 3/2]").unwrap(), "-4 + Pi^2/2");
    assert_eq!(interpret("PolyGamma[1, 5/2]").unwrap(), "-40/9 + Pi^2/2");
    assert_eq!(
      interpret("PolyGamma[1, 7/2]").unwrap(),
      "-1036/225 + Pi^2/2"
    );
    assert_eq!(interpret("PolyGamma[3, 1/2]").unwrap(), "Pi^4");
    // n! is kept un-distributed, matching wolframscript.
    assert_eq!(interpret("PolyGamma[3, 3/2]").unwrap(), "6*(-16 + Pi^4/6)");
  }

  // Even n at a half-integer involves an odd Zeta value (no closed form), so
  // it stays unevaluated like wolframscript.
  #[test]
  fn polygamma_even_n_half_integer_unevaluated() {
    assert_eq!(interpret("PolyGamma[2, 1/2]").unwrap(), "PolyGamma[2, 1/2]");
    assert_eq!(interpret("PolyGamma[2, 3/2]").unwrap(), "PolyGamma[2, 3/2]");
  }

  #[test]
  fn pole_at_zero() {
    assert_eq!(interpret("PolyGamma[0, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[0, -1]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[1, 0]").unwrap(), "ComplexInfinity");
  }

  // The poles at the non-positive integers are ComplexInfinity for an inexact
  // (Real) argument too, not the +/-Infinity the numeric series diverges to.
  // A negative non-integer argument stays finite. Per wolframscript.
  #[test]
  fn pole_at_real_nonpositive_integers() {
    assert_eq!(interpret("PolyGamma[0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[-1.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[-2.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[1, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[2, -3.0]").unwrap(), "ComplexInfinity");
    // A negative half-integer is finite (equals PolyGamma[2.5] here).
    assert_eq!(interpret("PolyGamma[-1.5]").unwrap(), "0.7031566406452437");
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

  // An inexact zero argument evaluates numerically to -0.5. Previously the
  // reflection formula computed sin(0)*Zeta[1] = 0*Infinity = NaN, so
  // Zeta[0.] came back as Indeterminate. Per wolframscript.
  #[test]
  fn zero_real() {
    assert_eq!(interpret("Zeta[0.0]").unwrap(), "-0.5");
  }

  // The pole at s = 1 is ComplexInfinity for an inexact argument too, not a
  // real Infinity. Per wolframscript.
  #[test]
  fn pole_at_one_real() {
    assert_eq!(interpret("Zeta[1.0]").unwrap(), "ComplexInfinity");
  }

  // As s -> +Infinity the series collapses to its first term, so
  // Zeta[Infinity] = 1; an undirected ComplexInfinity is Indeterminate.
  // Zeta[-Infinity] stays unevaluated. Per wolframscript.
  #[test]
  fn infinite_limits() {
    assert_eq!(interpret("Zeta[Infinity]").unwrap(), "1");
    assert_eq!(interpret("Zeta[ComplexInfinity]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Zeta[-Infinity]").unwrap(), "Zeta[-Infinity]");
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

  #[test]
  fn zeta_of_zeta_zero_is_zero() {
    // ZetaZero[k] denotes the k-th non-trivial zero of zeta, so
    // Zeta[ZetaZero[k]] = 0 — but only when k is a concrete positive integer.
    // For symbolic k, the simplification doesn't apply (matches wolframscript).
    assert_eq!(interpret("Zeta[ZetaZero[1]]").unwrap(), "0");
    assert_eq!(interpret("Zeta[ZetaZero[10]]").unwrap(), "0");
    assert_eq!(interpret("Zeta[ZetaZero[k]]").unwrap(), "Zeta[ZetaZero[k]]");
    assert_eq!(interpret("Zeta[ZetaZero[5, 100]]").unwrap(), "0");
  }

  #[test]
  fn zeta_zero_stays_symbolic() {
    // ZetaZero by itself remains symbolic.
    assert_eq!(interpret("ZetaZero[10]").unwrap(), "ZetaZero[10]");
    assert_eq!(interpret("ZetaZero[1]").unwrap(), "ZetaZero[1]");
  }
}

mod hurwitz_zeta {
  use super::*;

  #[test]
  fn zeta_s_1_reduces_to_riemann() {
    assert_eq!(interpret("Zeta[4, 1]").unwrap(), "Pi^4/90");
    assert_eq!(interpret("Zeta[2, 1]").unwrap(), "Pi^2/6");
    assert_eq!(interpret("Zeta[3, 1]").unwrap(), "Zeta[3]");
  }

  #[test]
  fn pole_at_s_1() {
    assert_eq!(interpret("Zeta[1, 1/2]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Zeta[1, 3]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn half_even() {
    assert_eq!(interpret("Zeta[4, 1/2]").unwrap(), "Pi^4/6");
    assert_eq!(interpret("Zeta[2, 1/2]").unwrap(), "Pi^2/2");
    assert_eq!(interpret("Zeta[6, 1/2]").unwrap(), "Pi^6/15");
  }

  #[test]
  fn half_odd() {
    assert_eq!(interpret("Zeta[3, 1/2]").unwrap(), "7*Zeta[3]");
  }

  // A half-integer second argument > 1 reduces to the 1/2 case via the
  // recurrence Zeta[s, a] = Zeta[s, a-1] - (a-1)^(-s).
  #[test]
  fn half_integer_above_one_reduces() {
    assert_eq!(interpret("Zeta[2, 3/2]").unwrap(), "-4 + Pi^2/2");
    assert_eq!(interpret("Zeta[2, 5/2]").unwrap(), "-40/9 + Pi^2/2");
    assert_eq!(interpret("Zeta[4, 3/2]").unwrap(), "-16 + Pi^4/6");
    // Odd s keeps the symbolic Zeta value.
    assert_eq!(interpret("Zeta[3, 5/2]").unwrap(), "-224/27 + 7*Zeta[3]");
  }

  // A non-special fractional part (1/3) has no closed form, so it stays
  // unevaluated even when reduced from above 1.
  #[test]
  fn non_special_fraction_unevaluated() {
    assert_eq!(interpret("Zeta[2, 1/3]").unwrap(), "Zeta[2, 1/3]");
  }

  #[test]
  fn s_zero() {
    assert_eq!(interpret("Zeta[0, 1/2]").unwrap(), "0");
    assert_eq!(interpret("Zeta[0, 3]").unwrap(), "-5/2");
  }

  #[test]
  fn negative_integer_s() {
    assert_eq!(interpret("Zeta[-1, 1/2]").unwrap(), "1/24");
    assert_eq!(interpret("Zeta[-1, 2]").unwrap(), "-13/12");
    assert_eq!(interpret("Zeta[-2, 1/2]").unwrap(), "0");
    assert_eq!(interpret("Zeta[-3, 1/2]").unwrap(), "-7/960");
    assert_eq!(interpret("Zeta[-3, 2]").unwrap(), "-119/120");
  }

  #[test]
  fn positive_integer_a() {
    assert_eq!(interpret("Zeta[4, 2]").unwrap(), "-1 + Pi^4/90");
    assert_eq!(interpret("Zeta[4, 3]").unwrap(), "-17/16 + Pi^4/90");
    assert_eq!(interpret("Zeta[2, 2]").unwrap(), "-1 + Pi^2/6");
    assert_eq!(interpret("Zeta[2, 3]").unwrap(), "-5/4 + Pi^2/6");
  }

  #[test]
  fn numeric_float() {
    let result: f64 = interpret("Zeta[2, 0.5]").unwrap().parse().unwrap();
    assert!((result - 4.934802200544678).abs() < 1e-10);

    let result: f64 = interpret("Zeta[4, 0.5]").unwrap().parse().unwrap();
    assert!((result - 16.234848505667077).abs() < 1e-8);

    let result: f64 = interpret("Zeta[2.5, 3]").unwrap().parse().unwrap();
    assert!((result - 0.1647105619542803).abs() < 1e-10);

    let result: f64 = interpret("Zeta[0.5, 2.0]").unwrap().parse().unwrap();
    assert!((result - (-2.4603545088095875)).abs() < 1e-8);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("Zeta[0, a]").unwrap(), "Zeta[0, a]");
    assert_eq!(interpret("Zeta[x, y]").unwrap(), "Zeta[x, y]");
    assert_eq!(interpret("Zeta[s, 1/2]").unwrap(), "(-1 + 2^s)*Zeta[s]");
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

  // Exact (integer/rational) arguments stay symbolic — only N[...] or an
  // inexact (machine) argument triggers numeric evaluation. Covers the whole
  // Jacobi family.
  #[test]
  fn exact_args_stay_symbolic() {
    assert_eq!(interpret("JacobiSN[1, 1/2]").unwrap(), "JacobiSN[1, 1/2]");
    assert_eq!(interpret("JacobiCN[1, 1/2]").unwrap(), "JacobiCN[1, 1/2]");
    assert_eq!(interpret("JacobiDN[1, 1/2]").unwrap(), "JacobiDN[1, 1/2]");
    assert_eq!(interpret("JacobiSC[1, 1/2]").unwrap(), "JacobiSC[1, 1/2]");
    assert_eq!(interpret("JacobiCD[1, 1/2]").unwrap(), "JacobiCD[1, 1/2]");
    assert_eq!(interpret("JacobiNS[1, 1/2]").unwrap(), "JacobiNS[1, 1/2]");
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
  fn complex_n_args() {
    // wolframscript: JacobiP[3.5 + I, 3, 2, 4 - I]
    //   ≈ 1410.0201167451296 + 5797.298553127177*I
    let s = interpret("JacobiP[3.5 + I, 3, 2, 4 - I]").unwrap();
    assert!(
      s.starts_with("1410.020116745") && s.contains(" + 5797.298553127"),
      "got: {}",
      s
    );
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

  // E_n(0) = 1/(n-1), but an inexact zero argument gives an inexact result:
  // E_2(0.) is 1., not the exact 1, and E_3(0.) is 0.5, not 1/2. Per
  // wolframscript. (The exact-argument results above are unchanged.)
  #[test]
  fn inexact_zero_stays_real() {
    assert_eq!(interpret("ExpIntegralE[2, 0.0]").unwrap(), "1.");
    assert_eq!(interpret("Head[ExpIntegralE[2, 0.0]]").unwrap(), "Real");
    assert_eq!(interpret("ExpIntegralE[3, 0.0]").unwrap(), "0.5");
    // Order 1 still diverges to ComplexInfinity for an inexact zero.
    assert_eq!(
      interpret("ExpIntegralE[1, 0.0]").unwrap(),
      "ComplexInfinity"
    );
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

mod fresnel_s {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("FresnelS[0]").unwrap(), "0");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("FresnelS[Infinity]").unwrap(), "1/2");
  }

  #[test]
  fn at_neg_infinity() {
    assert_eq!(interpret("FresnelS[-Infinity]").unwrap(), "-1/2");
  }

  #[test]
  fn at_complex_infinity() {
    assert_eq!(
      interpret("FresnelS[ComplexInfinity]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("FresnelS[x]").unwrap(), "FresnelS[x]");
  }

  #[test]
  fn odd_function() {
    assert_eq!(interpret("FresnelS[-x]").unwrap(), "-FresnelS[x]");
    assert_eq!(interpret("FresnelS[-2]").unwrap(), "-FresnelS[2]");
  }

  #[test]
  fn pure_imaginary() {
    // FresnelS[I*z] = -I*FresnelS[z]
    assert_eq!(interpret("FresnelS[I]").unwrap(), "-I*FresnelS[1]");
  }

  #[test]
  fn numeric_one() {
    let result: f64 = interpret("FresnelS[1.0]").unwrap().parse().unwrap();
    assert!((result - 0.43825914739035476).abs() < 1e-10);
  }

  #[test]
  fn numeric_two() {
    let result: f64 = interpret("FresnelS[2.0]").unwrap().parse().unwrap();
    assert!((result - 0.34341567836369946).abs() < 1e-10);
  }

  // Definite integrals `∫₀ᶻ Cos[π x²/2] dx = FresnelC[z]` and the Sin form.
  // Regression for mathics specialfns/erf.py:114.
  #[test]
  fn definite_integral_to_fresnel_c() {
    assert_eq!(
      interpret("Integrate[Cos[x^2 Pi/2], {x, 0, z}]").unwrap(),
      "FresnelC[z]"
    );
  }

  #[test]
  fn definite_integral_to_fresnel_s() {
    assert_eq!(
      interpret("Integrate[Sin[x^2 Pi/2], {x, 0, z}]").unwrap(),
      "FresnelS[z]"
    );
  }

  // Bessel integral representation:
  //   ∫_0^π Cos[n · Sin[w]] dw = π · BesselJ[0, n]
  // Regression for mathics specialfns/bessel.py:380.
  #[test]
  fn definite_integral_cos_n_sin_gives_bessel_j() {
    assert_eq!(
      interpret("Integrate[Cos[3 Sin[w]], {w, 0, Pi}]").unwrap(),
      "Pi*BesselJ[0, 3]"
    );
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("FresnelS[0.5]").unwrap().parse().unwrap();
    assert!((result - 0.06473243285999927).abs() < 1e-10);
  }

  #[test]
  fn numeric_three_point_five() {
    let result: f64 = interpret("FresnelS[3.5]").unwrap().parse().unwrap();
    assert!((result - 0.41524801197243752).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result: f64 = interpret("FresnelS[-2.0]").unwrap().parse().unwrap();
    assert!((result - (-0.34341567836369946)).abs() < 1e-10);
  }

  #[test]
  fn numeric_zero() {
    assert_eq!(interpret("FresnelS[0.0]").unwrap(), "0.");
  }

  #[test]
  fn n_fresnel_s() {
    let result = interpret("N[FresnelS[1]]").unwrap();
    assert!(result.starts_with("0.4382591"));
  }

  #[test]
  fn listable() {
    assert_eq!(interpret("FresnelS[{0, Infinity}]").unwrap(), "{0, 1/2}");
  }
}

mod fresnel_c {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("FresnelC[0]").unwrap(), "0");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("FresnelC[Infinity]").unwrap(), "1/2");
  }

  #[test]
  fn at_neg_infinity() {
    assert_eq!(interpret("FresnelC[-Infinity]").unwrap(), "-1/2");
  }

  #[test]
  fn at_complex_infinity() {
    assert_eq!(
      interpret("FresnelC[ComplexInfinity]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("FresnelC[x]").unwrap(), "FresnelC[x]");
  }

  #[test]
  fn odd_function() {
    assert_eq!(interpret("FresnelC[-x]").unwrap(), "-FresnelC[x]");
    assert_eq!(interpret("FresnelC[-2]").unwrap(), "-FresnelC[2]");
  }

  #[test]
  fn pure_imaginary() {
    // FresnelC[I*z] = I*FresnelC[z]
    assert_eq!(interpret("FresnelC[I]").unwrap(), "I*FresnelC[1]");
  }

  #[test]
  fn numeric_one() {
    let result: f64 = interpret("FresnelC[1.0]").unwrap().parse().unwrap();
    assert!((result - 0.7798934003768226).abs() < 1e-10);
  }

  #[test]
  fn numeric_two() {
    let result: f64 = interpret("FresnelC[2.0]").unwrap().parse().unwrap();
    assert!((result - 0.48825340607534046).abs() < 1e-10);
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("FresnelC[0.5]").unwrap().parse().unwrap();
    assert!((result - 0.49234422587144633).abs() < 1e-10);
  }

  #[test]
  fn numeric_three_point_five() {
    let result: f64 = interpret("FresnelC[3.5]").unwrap().parse().unwrap();
    assert!((result - 0.5325724350715935).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result: f64 = interpret("FresnelC[-2.0]").unwrap().parse().unwrap();
    assert!((result - (-0.48825340607534046)).abs() < 1e-10);
  }

  #[test]
  fn numeric_zero() {
    assert_eq!(interpret("FresnelC[0.0]").unwrap(), "0.");
  }

  #[test]
  fn n_fresnel_c() {
    let result = interpret("N[FresnelC[1]]").unwrap();
    assert!(result.starts_with("0.7798934"));
  }

  #[test]
  fn listable() {
    assert_eq!(interpret("FresnelC[{0, Infinity}]").unwrap(), "{0, 1/2}");
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
  fn non_integer_order() {
    // Non-integer order uses T_n(x) = Cos[n ArcCos[x]]. A half-integer order
    // rewrites for any x; other orders rewrite only when x is numeric.
    // Verified against wolframscript.
    assert_eq!(interpret("ChebyshevT[1/2, 0]").unwrap(), "1/Sqrt[2]");
    assert_eq!(interpret("ChebyshevT[1/2, 1/2]").unwrap(), "Sqrt[3]/2");
    assert_eq!(interpret("ChebyshevT[1/2, -1]").unwrap(), "0");
    assert_eq!(interpret("ChebyshevT[3/2, 0]").unwrap(), "-(1/Sqrt[2])");
    assert_eq!(interpret("ChebyshevT[1/3, 1/2]").unwrap(), "Cos[Pi/9]");
    assert_eq!(
      interpret("ChebyshevT[1/2, 1/3]").unwrap(),
      "Cos[ArcCos[1/3]/2]"
    );
    // Half-integer order rewrites even for symbolic x.
    assert_eq!(interpret("ChebyshevT[1/2, x]").unwrap(), "Cos[ArcCos[x]/2]");
    // A non-half-integer order with symbolic x stays unevaluated.
    assert_eq!(
      interpret("ChebyshevT[1/3, x]").unwrap(),
      "ChebyshevT[1/3, x]"
    );
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

  // T is even in its order: T_{-n}(x) = T_n(x).
  #[test]
  fn negative_index() {
    assert_eq!(interpret("ChebyshevT[-1, x]").unwrap(), "x");
    assert_eq!(interpret("ChebyshevT[-2, x]").unwrap(), "-1 + 2*x^2");
    assert_eq!(interpret("ChebyshevT[-3, x]").unwrap(), "-3*x + 4*x^3");
    assert_eq!(
      interpret("ChebyshevT[-5, x]").unwrap(),
      "5*x - 20*x^3 + 16*x^5"
    );
  }

  #[test]
  fn negative_index_numeric() {
    assert_eq!(interpret("ChebyshevT[-3, 2]").unwrap(), "26");
    assert_eq!(interpret("ChebyshevT[-4, 1/2]").unwrap(), "-1/2");
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
  fn non_integer_order() {
    // Non-integer order uses U_n(x) = Sin[(n+1) ArcCos[x]]/(Sqrt[1-x] Sqrt[1+x]),
    // with U_n(1) = n + 1 (removable singularity). Verified against
    // wolframscript.
    assert_eq!(interpret("ChebyshevU[1/2, 0]").unwrap(), "1/Sqrt[2]");
    assert_eq!(interpret("ChebyshevU[1/2, 1/2]").unwrap(), "2/Sqrt[3]");
    assert_eq!(interpret("ChebyshevU[3/2, 0]").unwrap(), "-(1/Sqrt[2])");
    assert_eq!(interpret("ChebyshevU[1/2, 1]").unwrap(), "3/2");
    assert_eq!(interpret("ChebyshevU[1/2, -1]").unwrap(), "ComplexInfinity");
    assert_eq!(
      interpret("ChebyshevU[3/2, x]").unwrap(),
      "Sin[(5*ArcCos[x])/2]/(Sqrt[1 - x]*Sqrt[1 + x])"
    );
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

  // Reflection: U_{-1}(x) = 0 and U_{-n}(x) = -U_{n-2}(x) for n >= 2.
  #[test]
  fn negative_index() {
    assert_eq!(interpret("ChebyshevU[-1, x]").unwrap(), "0");
    assert_eq!(interpret("ChebyshevU[-2, x]").unwrap(), "-1");
    assert_eq!(interpret("ChebyshevU[-3, x]").unwrap(), "-2*x");
    assert_eq!(interpret("ChebyshevU[-4, x]").unwrap(), "1 - 4*x^2");
    assert_eq!(interpret("ChebyshevU[-5, x]").unwrap(), "4*x - 8*x^3");
  }

  #[test]
  fn negative_index_table() {
    assert_eq!(
      interpret("Table[ChebyshevU[-n, x], {n, 1, 6}]").unwrap(),
      "{0, -1, -2*x, 1 - 4*x^2, 4*x - 8*x^3, -1 + 12*x^2 - 16*x^4}"
    );
  }

  #[test]
  fn negative_index_numeric() {
    assert_eq!(interpret("ChebyshevU[-4, 3]").unwrap(), "-35");
    assert_eq!(interpret("ChebyshevU[-5, 1/2]").unwrap(), "1");
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

  // Non-integer (rational) n routes through the
  // `LaguerreL[n, x] = Hypergeometric1F1[-n, 1, x]` identity.
  // Regression for mathics specialfns/orthogonal.py:144.
  #[test]
  fn rational_n_numeric() {
    assert_eq!(
      interpret("LaguerreL[3/2, 1.7]").unwrap(),
      "-0.9471339972534181"
    );
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

  #[test]
  fn generalized_laguerre() {
    // Wolfram puts every term over a common denominator, matching
    // wolframscript: `(2520 - 4200*x + 2100*x^2 - 420*x^3 + 35*x^4 -
    // x^5)/120`. Per-term rendering would otherwise give the
    // mathematically-equal `21 - 35*x + (35*x^2)/2 - …`.
    assert_eq!(
      interpret("LaguerreL[5, 2, x]").unwrap(),
      "(2520 - 4200*x + 2100*x^2 - 420*x^3 + 35*x^4 - x^5)/120"
    );
  }

  #[test]
  fn generalized_laguerre_l0() {
    assert_eq!(interpret("LaguerreL[0, 5, x]").unwrap(), "1");
  }

  #[test]
  fn generalized_laguerre_l1() {
    assert_eq!(interpret("LaguerreL[1, 3, x]").unwrap(), "4 - x");
  }

  // L_n^(a)(x) coefficients all share `n!` in the denominator; emit the
  // result as `(numerator)/n!` instead of distributing the rational
  // factor over each term, matching wolframscript's display.
  #[test]
  fn generalized_laguerre_uses_common_denominator() {
    assert_eq!(
      interpret("LaguerreL[2, 1, x]").unwrap(),
      "(6 - 6*x + x^2)/2"
    );
    assert_eq!(
      interpret("LaguerreL[3, 0, x]").unwrap(),
      "(6 - 18*x + 9*x^2 - x^3)/6"
    );
    assert_eq!(
      interpret("LaguerreL[4, 1, x]").unwrap(),
      "(120 - 240*x + 120*x^2 - 20*x^3 + x^4)/24"
    );
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

  // Beta[a, b] = Gamma[a] Gamma[b] / Gamma[a+b] has poles when a or b is a
  // non-positive integer. With inexact (Real) arguments the naive gamma_fn
  // product returned Infinity/Indeterminate or a large garbage value; it must
  // instead resolve by pole order, matching the exact-integer case and
  // wolframscript.
  #[test]
  fn real_pole_is_complex_infinity() {
    // A surviving numerator pole -> ComplexInfinity.
    assert_eq!(interpret("Beta[0.0, 2.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Beta[2.0, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Beta[0.0, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Beta[-1.0, 2.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Beta[-2.0, 5.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Beta[-1.0, -1.0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn real_cancelling_poles_are_finite() {
    // Numerator and denominator poles cancel to a finite limit.
    assert_eq!(interpret("Beta[-3.0, 2.0]").unwrap(), "0.16666666666666666");
    assert_eq!(interpret("Beta[2.0, -2.0]").unwrap(), "0.5");
    // A denominator-only pole (a+b a non-positive integer) -> 0.
    assert_eq!(interpret("Beta[1.5, -3.5]").unwrap(), "0.");
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

  // ── Incomplete Beta: Beta[z, a, b] ───────────────────────────────────
  #[test]
  fn incomplete_real_z_integer_b() {
    // 12 * Beta[1., 2, 3] == 1.  (the Mathics `Beta` doctest).
    assert_eq!(interpret("12 * Beta[1., 2, 3]").unwrap(), "1.");
  }

  #[test]
  fn incomplete_rational_z_integer_b() {
    // Closed-form expansion for integer b with exact z preserves rationals:
    // Beta[1/2, 2, 3] = integral_0^{1/2} t (1-t)^2 dt = 11/192.
    assert_eq!(interpret("Beta[1/2, 2, 3]").unwrap(), "11/192");
  }

  #[test]
  fn incomplete_symbolic_z_stays_held() {
    assert_eq!(interpret("Beta[z, 2, 3]").unwrap(), "Beta[z, 2, 3]");
  }

  #[test]
  fn incomplete_z_one_reduces_to_complete() {
    // Beta[1, a, b] = Beta[a, b].
    assert_eq!(interpret("Beta[1, 2, 3]").unwrap(), "1/12");
  }

  // ── Complete Beta with one integer + one rational argument ───────────
  // Beta[a, n] = (n-1)! / Pochhammer[a, n] is an exact rational for any
  // rational a; wolframscript evaluates these (regression for a gap where
  // half-integer/integer and general rational/integer pairs stayed symbolic).
  #[test]
  fn complete_half_integer_and_integer() {
    assert_eq!(interpret("Beta[1/2, 3]").unwrap(), "16/15");
    assert_eq!(interpret("Beta[3/2, 2]").unwrap(), "4/15");
    assert_eq!(interpret("Beta[5/2, 2]").unwrap(), "4/35");
    // Symmetric argument order.
    assert_eq!(interpret("Beta[3, 1/2]").unwrap(), "16/15");
  }

  #[test]
  fn complete_general_rational_and_integer() {
    assert_eq!(interpret("Beta[1/3, 2]").unwrap(), "9/4");
    assert_eq!(interpret("Beta[7/3, 2]").unwrap(), "9/70");
    // Negative rational still telescopes through the rising factorial.
    assert_eq!(interpret("Beta[-1/2, 2]").unwrap(), "-4");
  }

  // ── Incomplete Beta: exact non-integer a stays symbolic ──────────────
  // wolframscript only auto-evaluates Beta[z, a, b] to a closed form when both
  // a and b are integers. For an exact non-integer a it stays symbolic, even
  // though the rational closed form exists (e.g. Beta[2, 1/2, 3] = 14 Sqrt[2]/15).
  #[test]
  fn incomplete_noninteger_a_stays_symbolic() {
    assert_eq!(interpret("Beta[2, 1/2, 3]").unwrap(), "Beta[2, 1/2, 3]");
    assert_eq!(interpret("Beta[2, 1/2, 2]").unwrap(), "Beta[2, 1/2, 2]");
    assert_eq!(interpret("Beta[1/2, 1/2, 3]").unwrap(), "Beta[1/2, 1/2, 3]");
  }

  #[test]
  fn incomplete_integer_a_evaluates() {
    // Both a and b integers ⇒ exact rational closed form.
    assert_eq!(interpret("Beta[2, 2, 3]").unwrap(), "2/3");
    assert_eq!(interpret("Beta[-1, 2, 3]").unwrap(), "17/12");
  }

  #[test]
  fn incomplete_inexact_evaluates_numerically() {
    // Any inexact argument ⇒ a machine-number result, even when a is a
    // non-integer (previously these stayed symbolic).
    let r1: f64 = interpret("Beta[2., 0.5, 3.]").unwrap().parse().unwrap();
    assert!((r1 - 1.3199326582148885).abs() < 1e-12);
    let r2: f64 = interpret("N[Beta[2, 1/2, 3]]").unwrap().parse().unwrap();
    assert!((r2 - 1.3199326582148885).abs() < 1e-12);
    // a real, b a whole-valued real.
    let r3: f64 = interpret("Beta[0.5, 2, 3.]").unwrap().parse().unwrap();
    assert!((r3 - 0.057291666666666685).abs() < 1e-12);
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

  #[test]
  fn complex_argument_is_expanded() {
    // Regression: `HermiteH[3, 1 + I]` used to stay as
    // `-12 (1 + I) + 8 (1 + I)^3`; numeric complex arguments are now
    // expanded to `-28 + 4 I` matching wolframscript.
    assert_eq!(interpret("HermiteH[3, 1 + I]").unwrap(), "-28 + 4*I");
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

  #[test]
  fn attributes() {
    // Wolfram versions disagree on whether AiryAi has ReadProtected, so
    // only assert that the essential attributes are present.
    let result = interpret("Attributes[AiryAi]").unwrap();
    for attr in ["Listable", "NumericFunction", "Protected"] {
      assert!(result.contains(attr), "missing {attr} in {result}");
    }
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
  fn complex_n_args() {
    // wolframscript: GegenbauerC[4 - I, 1 + 2 I, 0.7]
    //   ≈ -3.2620959521652644 - 24.973939745527076*I
    let s = interpret("GegenbauerC[4 - I, 1 + 2 I, 0.7]").unwrap();
    assert!(
      s.starts_with("-3.262095952165") && s.contains(" - 24.973939745526"),
      "got: {}",
      s
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("GegenbauerC[n, m, x]").unwrap(),
      "GegenbauerC[n, m, x]"
    );
  }

  // Two-argument (renormalized) form: GegenbauerC[n, x] = (2/n) ChebyshevT[n, x]

  #[test]
  fn two_arg_zero() {
    // (2/n) blows up at n = 0
    assert_eq!(interpret("GegenbauerC[0, x]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn two_arg_n1() {
    assert_eq!(interpret("GegenbauerC[1, x]").unwrap(), "2*x");
  }

  #[test]
  fn two_arg_n2() {
    assert_eq!(interpret("GegenbauerC[2, x]").unwrap(), "-1 + 2*x^2");
  }

  #[test]
  fn two_arg_n3() {
    assert_eq!(
      interpret("GegenbauerC[3, x]").unwrap(),
      "(2*(-3*x + 4*x^3))/3"
    );
  }

  #[test]
  fn two_arg_n4() {
    assert_eq!(
      interpret("GegenbauerC[4, x]").unwrap(),
      "(1 - 8*x^2 + 8*x^4)/2"
    );
  }

  #[test]
  fn two_arg_n5() {
    assert_eq!(
      interpret("GegenbauerC[5, x]").unwrap(),
      "(2*(5*x - 20*x^3 + 16*x^5))/5"
    );
  }

  #[test]
  fn two_arg_integer_arg() {
    // ChebyshevT[2, 3] = 17, times (2/2) = 17
    assert_eq!(interpret("GegenbauerC[2, 3]").unwrap(), "17");
  }

  #[test]
  fn two_arg_values_at_one() {
    // GegenbauerC[n, 1] = (2/n) ChebyshevT[n, 1] = 2/n
    assert_eq!(
      interpret("Table[GegenbauerC[n, 1], {n, 1, 5}]").unwrap(),
      "{2, 1, 2/3, 1/2, 2/5}"
    );
  }

  #[test]
  fn two_arg_real() {
    let result: f64 =
      interpret("GegenbauerC[3, 0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.6666666666666666)).abs() < 1e-12);
  }

  #[test]
  fn two_arg_symbolic_n() {
    // General form: (2/n) ChebyshevT[n, x]
    assert_eq!(
      interpret("GegenbauerC[n, x]").unwrap(),
      "(2*ChebyshevT[n, x])/n"
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

  // For power-of-2 inputs, the FFT path should produce identical
  // results to the naive DFT at small sizes (matches existing
  // basic_integer_list output).
  #[test]
  fn fft_8_matches_dft() {
    assert_eq!(
      interpret("Fourier[{1, 2, 3, 4, 5, 6, 7, 8}]").unwrap(),
      interpret("Fourier[{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}]").unwrap()
    );
  }

  // Inverse round-trip: InverseFourier[Fourier[x]] == x.
  #[test]
  fn fft_inverse_round_trip_pow2() {
    let result =
      interpret("Chop[Re[InverseFourier[Fourier[{1.0, 2.0, 3.0, 4.0}]]]]")
        .unwrap();
    assert_eq!(result, "{1., 2., 3., 4.}");
  }

  // Large power-of-2 sizes must complete in a reasonable time (the
  // old O(n²) DFT would TIMEOUT). The FFT path makes this feasible.
  // Note: this test isn't a timing assertion, just an existence check.
  #[test]
  fn fft_8192_completes() {
    let result =
      interpret("Length[Fourier[N[Table[k, {k, 0, 8191}]]]]").unwrap();
    assert_eq!(result, "8192");
  }

  #[test]
  fn single_element() {
    assert_eq!(interpret("Fourier[{5}]").unwrap(), "{5.}");
  }

  #[test]
  fn two_equal_elements() {
    // All imaginary parts are exactly 0 → Real output
    let result = interpret("Fourier[{2, 2}]").unwrap();
    assert!(result.contains("2.82842712474619"), "got: {}", result);
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

mod fourier_dct {
  use super::*;

  // Machine-precision DCT results differ from wolframscript only in the
  // last 1-2 ulps (no algorithm reproduces Wolfram's proprietary FFT-DCT
  // rounding bit-for-bit). The values agree with wolframscript to ~15
  // significant digits, so the tests round to 10 digits, which matches
  // wolframscript exactly.

  #[test]
  fn dct2_default_integer_list() {
    // wolframscript: FourierDCT[{1, 2, 3, 4}]
    //   = {5., -1.577161014949475, 0., -0.11208538229199139}
    assert_eq!(
      interpret("Round[10^10 FourierDCT[{1, 2, 3, 4}]]").unwrap(),
      "{50000000000, -15771610149, 0, -1120853823}"
    );
  }

  #[test]
  fn dct1() {
    // wolframscript: FourierDCT[{1, 2, 3, 4}, 1]
    assert_eq!(
      interpret("Round[10^10 FourierDCT[{1, 2, 3, 4}, 1]]").unwrap(),
      "{61237243570, -16329931619, 0, -4082482905}"
    );
  }

  #[test]
  fn dct2_explicit() {
    // FourierDCT[{1, 2, 3, 4}, 2] equals the default form.
    assert_eq!(
      interpret("FourierDCT[{1, 2, 3, 4}, 2]").unwrap(),
      interpret("FourierDCT[{1, 2, 3, 4}]").unwrap()
    );
  }

  #[test]
  fn dct3() {
    // wolframscript: FourierDCT[{1, 2, 3, 4}, 3]
    assert_eq!(
      interpret("Round[10^10 FourierDCT[{1, 2, 3, 4}, 3]]").unwrap(),
      "{59998131380, -45514716089, 13088309218, -7571724509}"
    );
  }

  #[test]
  fn dct4() {
    // wolframscript: FourierDCT[{1, 2, 3, 4}, 4]
    assert_eq!(
      interpret("Round[10^10 FourierDCT[{1, 2, 3, 4}, 4]]").unwrap(),
      "{35997367212, -33399112628, 17714079076, -16580115558}"
    );
  }

  #[test]
  fn odd_length() {
    // wolframscript: FourierDCT[{1, 2, 3}]
    //   = {3.4641016151377553, -1.0000000000000002, -1.6653345369377348*^-16}
    assert_eq!(
      interpret("Round[10^10 FourierDCT[{1, 2, 3}]]").unwrap(),
      "{34641016151, -10000000000, 0}"
    );
  }

  #[test]
  fn dct3_odd_length() {
    // wolframscript: FourierDCT[{1, 2, 3}, 3]
    assert_eq!(
      interpret("Round[10^10 FourierDCT[{1, 2, 3}, 3]]").unwrap(),
      "{43094010768, -28867513459, 3094010768}"
    );
  }

  #[test]
  fn single_element() {
    // wolframscript: FourierDCT[{1}] = {1.}
    assert_eq!(interpret("FourierDCT[{1}]").unwrap(), "{1.}");
  }

  #[test]
  fn dct1_two_elements() {
    // wolframscript: FourierDCT[{1, 2}, 1]
    //   = {2.1213203435596424, -0.7071067811865475}
    assert_eq!(
      interpret("Round[10^10 FourierDCT[{1, 2}, 1]]").unwrap(),
      "{21213203436, -7071067812}"
    );
  }

  #[test]
  fn real_valued_input() {
    // Real machine-precision input gives the same result as integer input.
    assert_eq!(
      interpret("FourierDCT[{1., 2., 3., 4.}]").unwrap(),
      interpret("FourierDCT[{1, 2, 3, 4}]").unwrap()
    );
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    // wolframscript leaves non-numeric input unevaluated (with a message).
    assert_eq!(
      interpret("FourierDCT[{a, b, c}]").unwrap(),
      "FourierDCT[{a, b, c}]"
    );
  }

  #[test]
  fn empty_list_returns_unevaluated() {
    assert_eq!(interpret("FourierDCT[{}]").unwrap(), "FourierDCT[{}]");
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

  #[test]
  fn infinite_product_with_acceleration() {
    // Product_{i=1}^∞ (1 + 1/i^2) = Sinh[Pi]/Pi ≈ 3.6760779100585657.
    // Woxi's finite-term + Wynn-epsilon estimate gets to within 1e-4.
    let result: f64 = interpret("NProduct[1 + 1/i^2, {i, 1, Infinity}]")
      .unwrap()
      .parse()
      .unwrap();
    assert!(
      (result - 3.6760779100585657).abs() < 1e-3,
      "expected ≈ 3.6760779, got {}",
      result
    );
  }

  #[test]
  fn infinite_product_geometric() {
    // Product_{i=1}^∞ (1 - 1/(i+1)^2) = 1/2 (telescoping).
    let result: f64 = interpret("NProduct[1 - 1/(i+1)^2, {i, 1, Infinity}]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.5).abs() < 1e-3, "got {}", result);
  }
}

mod hurwitz_lerch_phi {
  use super::*;

  // HurwitzLerchPhi coincides with LerchPhi except at non-positive integer a.
  #[test]
  fn closed_form_matches_lerch_phi() {
    assert_eq!(
      interpret("HurwitzLerchPhi[1/2, 2, 1]").unwrap(),
      "2*(Pi^2/12 - Log[2]^2/2)"
    );
    // z = 0 gives a^(-s).
    assert_eq!(interpret("HurwitzLerchPhi[0, 2, 3]").unwrap(), "1/9");
  }

  // At a non-positive integer a the singular k = -a term diverges.
  #[test]
  fn non_positive_integer_a_is_complex_infinity() {
    assert_eq!(
      interpret("HurwitzLerchPhi[1/2, 2, 0]").unwrap(),
      "ComplexInfinity"
    );
    assert_eq!(
      interpret("HurwitzLerchPhi[1/2, 2, -1]").unwrap(),
      "ComplexInfinity"
    );
    assert_eq!(
      interpret("HurwitzLerchPhi[1/2, 2, -2]").unwrap(),
      "ComplexInfinity"
    );
  }

  #[test]
  fn numeric_values() {
    let v: f64 = interpret("N[HurwitzLerchPhi[1/2, 2, 1]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 1.1644810529300247).abs() < 1e-9, "got {v}");

    let v: f64 = interpret("N[HurwitzLerchPhi[1/3, 3, 2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 0.13945075039356025).abs() < 1e-9, "got {v}");

    let v: f64 = interpret("N[HurwitzLerchPhi[1/2, 2, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 4.27714550058095).abs() < 1e-9, "got {v}");
  }

  // Fully symbolic arguments stay unevaluated under the HurwitzLerchPhi head.
  #[test]
  fn symbolic_stays_unevaluated() {
    assert_eq!(
      interpret("HurwitzLerchPhi[z, s, a]").unwrap(),
      "HurwitzLerchPhi[z, s, a]"
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
    // LerchPhi[1, 2, 1] = Σ 1/(k+1)^2 = π²/6, returned exactly (matching
    // wolframscript) rather than as a machine float.
    assert_eq!(interpret("LerchPhi[1, 2, 1]").unwrap(), "Pi^2/6");
  }

  #[test]
  fn lerch_phi_z1_exact_via_hurwitz_zeta() {
    // LerchPhi[1, s, a] == HurwitzZeta[s, a]; exact in the regime where both
    // engines produce the same closed form (even s, or a in {1, 2}).
    assert_eq!(interpret("LerchPhi[1, 4, 1]").unwrap(), "Pi^4/90");
    assert_eq!(interpret("LerchPhi[1, 2, 2]").unwrap(), "-1 + Pi^2/6");
    assert_eq!(interpret("LerchPhi[1, 2, 3]").unwrap(), "-5/4 + Pi^2/6");
    assert_eq!(interpret("LerchPhi[1, 3, 1]").unwrap(), "Zeta[3]");
    assert_eq!(interpret("LerchPhi[1, 3, 2]").unwrap(), "-1 + Zeta[3]");
    assert_eq!(interpret("LerchPhi[1, 5, 2]").unwrap(), "-1 + Zeta[5]");
    assert_eq!(interpret("LerchPhi[1, 2, 1/2]").unwrap(), "Pi^2/2");
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

  // LerchPhi[z, s, 1] = PolyLog[s, z]/z: delegating to PolyLog yields
  // wolframscript's exact closed forms instead of floatifying.
  #[test]
  fn lerch_phi_a1_reduces_to_polylog() {
    // PolyLog[2, 1/2] has a closed form, so LerchPhi[1/2, 2, 1] is exact.
    assert_eq!(
      interpret("LerchPhi[1/2, 2, 1]").unwrap(),
      "2*(Pi^2/12 - Log[2]^2/2)"
    );
    // No PolyLog closed form: stays symbolic as a scaled PolyLog.
    assert_eq!(
      interpret("LerchPhi[1/3, 2, 1]").unwrap(),
      "3*PolyLog[2, 1/3]"
    );
    // Fully symbolic: PolyLog[s, z]/z.
    assert_eq!(interpret("LerchPhi[z, s, 1]").unwrap(), "PolyLog[s, z]/z");
  }

  #[test]
  fn lerch_phi_hurwitz_zeta_quarter() {
    // Hurwitz-zeta identity: ζ(2, 1/4) = LerchPhi[1, 2, 1/4] = π² + 8·Catalan.
    // Tighter Euler-Maclaurin tail brings this within machine ε.
    let result = interpret("LerchPhi[1, 2, 1/4]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let pi = std::f64::consts::PI;
    let catalan = 0.915_965_594_177_219_015;
    let expected = pi * pi + 8.0 * catalan;
    assert!(
      (val - expected).abs() < 1e-12,
      "LerchPhi[1, 2, 1/4] should be π²+8C ≈ {}, got {}",
      expected,
      val
    );
  }

  // Exact arguments with no closed form stay symbolic — they are NOT
  // numericized automatically (only N[...] or an inexact argument does that).
  #[test]
  fn exact_args_stay_symbolic() {
    assert_eq!(
      interpret("LerchPhi[1/2, 2, 1/3]").unwrap(),
      "LerchPhi[1/2, 2, 1/3]"
    );
    // N[...] forces the numeric value.
    let v: f64 = interpret("N[LerchPhi[1/2, 2, 1/3]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 9.34347465937594).abs() < 1e-9);
  }

  // The z = 0 special case is exact for exact arguments: a^(-s).
  #[test]
  fn zero_z_exact() {
    assert_eq!(interpret("LerchPhi[0, 2, 3]").unwrap(), "1/9");
    assert_eq!(interpret("LerchPhi[0, 2, 1/3]").unwrap(), "9");
  }

  fn parse_complex(s: &str) -> (f64, f64) {
    // Parse forms like "12.34 + 5.6*I", "12.34 - 5.6*I", "-12.34", etc.
    let s = s.trim();
    // Split on " + " or " - " (last occurrence so we don't break a leading minus).
    let bytes = s.as_bytes();
    let mut sign_pos: Option<(usize, f64)> = None;
    for i in (1..s.len()).rev() {
      if bytes[i] == b'+' && bytes[i - 1] == b' ' {
        sign_pos = Some((i, 1.0));
        break;
      }
      if bytes[i] == b'-' && bytes[i - 1] == b' ' {
        sign_pos = Some((i, -1.0));
        break;
      }
    }
    if let Some((pos, sign)) = sign_pos {
      let real_part: f64 = s[..pos].trim().parse().unwrap();
      let imag_chunk = s[pos + 1..]
        .trim()
        .trim_end_matches("*I")
        .trim_end_matches('I')
        .trim();
      let imag_mag: f64 = if imag_chunk.is_empty() {
        1.0
      } else {
        imag_chunk.parse().unwrap()
      };
      (real_part, sign * imag_mag)
    } else if s.ends_with("*I") || s.ends_with('I') {
      let mag: f64 = s
        .trim_end_matches("*I")
        .trim_end_matches('I')
        .trim()
        .parse()
        .unwrap();
      (0.0, mag)
    } else {
      (s.parse().unwrap(), 0.0)
    }
  }

  fn assert_close(got: f64, expected: f64, msg: &str) {
    let tol = (expected.abs() * 1e-4).max(1e-6);
    assert!(
      (got - expected).abs() < tol,
      "{msg}: got {got}, expected {expected}"
    );
  }

  #[test]
  fn lerch_phi_z_gt_one_negative_a() {
    // Audit case: `LerchPhi[2, 3, -1.5]`. wolframscript reports
    // `51.981861922538684 - 2.1345964981239467*I`.
    let result = interpret("LerchPhi[2, 3, -1.5]").unwrap();
    let (re, im) = parse_complex(&result);
    assert_close(re, 51.981861922538684, "Re");
    assert_close(im, -2.1345964981239467, "Im");
  }

  #[test]
  fn lerch_phi_z_gt_one_half_integer_a() {
    // `LerchPhi[2, 3, 0.5]` ≈ 8.92139 - 0.53365·I.
    let result = interpret("LerchPhi[2, 3, 0.5]").unwrap();
    let (re, im) = parse_complex(&result);
    assert_close(re, 8.921391406560604, "Re");
    assert_close(im, -0.5336491245309853, "Im");
  }

  #[test]
  fn lerch_phi_z_gt_one_imaginary_matches_residue() {
    // The imaginary part is closed form:
    //   Im LerchPhi(z, s, a) = -π · (ln z)^(s−1) · z^(−a) / (s−1)!
    // For (z, s, a) = (3, 2, 0.5):
    //   Im = -π · ln(3) · 3^(-0.5) / 1 = -π·ln(3)/√3
    let result = interpret("LerchPhi[3, 2, 0.5]").unwrap();
    let (_re, im) = parse_complex(&result);
    let expected_im = -std::f64::consts::PI * 3.0_f64.ln() / 3.0_f64.sqrt();
    assert_close(im, expected_im, "Im");
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

  // Exact (integer/rational) arguments stay symbolic — only an inexact
  // argument or N[...] triggers numeric evaluation.
  #[test]
  fn exact_args_stay_symbolic() {
    assert_eq!(
      interpret("WeierstrassP[1, {1, 1}]").unwrap(),
      "WeierstrassP[1, {1, 1}]"
    );
    // An inexact invariant still numericizes.
    let v: f64 = interpret("WeierstrassP[1, {1.0, 1}]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v - 1.0871464472148646).abs() < 1e-12);
  }

  // The Laurent-series recurrence had a wrong denominator factor (k-1 vs k-2),
  // so c[3] (and higher) were off and the value was only ~8-digit accurate.
  #[test]
  fn accurate_value() {
    let v: f64 = interpret("WeierstrassP[1.0, {1, 1}]")
      .unwrap()
      .parse()
      .unwrap();
    assert!(
      (v - 1.0871464472148646).abs() < 1e-12,
      "WeierstrassP[1.0, {{1,1}}] = {} (expected 1.0871464472148646)",
      v
    );
    let v2: f64 = interpret("WeierstrassP[0.5, {1, 2}]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((v2 - 4.016981503916559).abs() < 1e-11);
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
    // Routes through `Hypergeometric1F1[1, 2, 1.0]`; both pdq numeric and
    // 1F1 numeric paths produce a one-ulp drift around `E - 1 ≈ 1.71828…`.
    assert_eq!(
      interpret("HypergeometricPFQ[{1}, {2}, 1.0]").unwrap(),
      "1.7182818284590455"
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
  fn one_f_one_b_plus_one_closed_form() {
    // 1F1[b+1, b, z] = ((b + z) / b) * E^z. Regression for mathics
    // specialfns/hypergeom.py:159 — z=1 should yield the symbolic 3E/2,
    // not Infinity, since 1F1 has infinite convergence radius.
    assert_eq!(
      interpret("HypergeometricPFQ[{3}, {2}, 1]").unwrap(),
      "(3*E)/2"
    );
    assert_eq!(
      interpret("HypergeometricPFQ[{2}, {1}, z]").unwrap(),
      "E^z*(1 + z)"
    );
    assert_eq!(
      interpret("HypergeometricPFQ[{4}, {3}, z]").unwrap(),
      "(E^z*(3 + z))/3"
    );
  }

  #[test]
  fn one_f_one_half_one_bessel_identity() {
    // 1F1[1/2, 1, z] = E^(z/2) * BesselI[0, z/2]. Regression for mathics
    // specialfns/hypergeom.py:65.
    assert_eq!(
      interpret("Hypergeometric1F1[1/2, 1, x]").unwrap(),
      "E^(x/2)*BesselI[0, x/2]"
    );
  }

  #[test]
  fn one_f_one_one_half_erf_identity() {
    // 1F1[1, 1/2, z] = 1 + E^z · √π · √z · Erf[√z]. Regression for mathics
    // specialfns/hypergeom.py:83.
    assert_eq!(
      interpret("Hypergeometric1F1[1, 1/2, x]").unwrap(),
      "1 + E^x*Sqrt[Pi]*Sqrt[x]*Erf[Sqrt[x]]"
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

  // If any upper parameter is exactly 0, the Pochhammer (0)_n is 0 for all
  // n >= 1, so the series collapses to 1 regardless of b or z.
  #[test]
  fn zero_in_a_list_is_one() {
    assert_eq!(interpret("HypergeometricPFQ[{0}, {b}, z]").unwrap(), "1");
    assert_eq!(
      interpret("HypergeometricPFQ[{0, 2}, {b, c}, z]").unwrap(),
      "1"
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

  #[test]
  fn non_positive_integer_b_finite() {
    // When a b argument is a non-positive integer, Γ(b) is a pole but the
    // regularized form is still finite (the series simply skips the early
    // n terms where 1/Γ(b + n) = 0). Audit case.
    let result =
      interpret("HypergeometricPFQRegularized[{1/3, 1/3, 1/3}, {-2, -3}, 0.5]")
        .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 15.062110268295829).abs() < 1e-9, "got {}", val);
  }

  #[test]
  fn non_positive_integer_b_simple() {
    // HypergeometricPFQRegularized[{1, 2}, {-2}, z] = 24*z^3/(1-z)^5
    // at z = 0.5: 24 * 0.125 / 0.5^5 = 3 / 0.03125 = 96
    let result =
      interpret("HypergeometricPFQRegularized[{1, 2}, {-2}, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 96.0).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn non_positive_integer_b_one_a() {
    // HypergeometricPFQRegularized[{1}, {-2}, z] = z^3 * E^z
    // at z = 0.5: 0.125 * E^0.5 ≈ 0.20609015883751602
    let result =
      interpret("HypergeometricPFQRegularized[{1}, {-2}, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.20609015883751602).abs() < 1e-10, "got {}", val);
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

  // For non-positive integer c = -m the regularized form has a finite
  // value through the DLMF identity
  //   2F1Reg(a, b; -m; z)
  //     = (a)_{m+1} (b)_{m+1} / (m+1)! · z^{m+1} · 2F1(a+m+1, b+m+1; m+2; z).
  // (See https://dlmf.nist.gov/15.4.E1.)
  #[test]
  fn non_positive_c_audit_case() {
    // Hypergeometric2F1Regularized[1, 2, -3, 4.5] = 26.768438320767707
    let result =
      interpret("Hypergeometric2F1Regularized[1, 2, -3, 4.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 26.768438320767707).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn non_positive_c_m_zero() {
    // 2F1Reg(1, 2; 0; 1.5) = -24.
    let result =
      interpret("Hypergeometric2F1Regularized[1, 2, 0, 1.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val + 24.0).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn non_positive_c_m_one() {
    // 2F1Reg(1, 2; -1; 0.5) = 24.
    let result =
      interpret("Hypergeometric2F1Regularized[1, 2, -1, 0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 24.0).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn non_positive_c_general() {
    // 2F1Reg(2, 3; -2; 0.25) = 31.604938271604937 (no a=c shortcut; the
    // inner 2F1[5, 6; 4; 0.25] is computed directly because |z| < 1).
    let result =
      interpret("Hypergeometric2F1Regularized[2, 3, -2, 0.25]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 31.604938271604937).abs() < 1e-8, "got {}", val);
  }

  #[test]
  fn non_positive_c_z_zero() {
    // For z = 0 the formula collapses to 0 because of the z^{m+1} factor.
    assert_eq!(
      interpret("Hypergeometric2F1Regularized[1, 2, -3, 0]").unwrap(),
      "0"
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

  // One- and two-argument (infinite product) forms.

  #[test]
  fn one_arg_rewrites_to_two_arg() {
    // QPochhammer[q] = QPochhammer[q, q] (Euler function)
    assert_eq!(interpret("QPochhammer[q]").unwrap(), "QPochhammer[q, q]");
  }

  #[test]
  fn two_arg_symbolic_unevaluated() {
    assert_eq!(interpret("QPochhammer[a, q]").unwrap(), "QPochhammer[a, q]");
  }

  #[test]
  fn two_arg_zero_a() {
    // Every factor is (1 - 0) = 1
    assert_eq!(interpret("QPochhammer[0, q]").unwrap(), "1");
  }

  #[test]
  fn two_arg_exact_stays_symbolic() {
    // Like wolframscript, exact arguments keep the infinite product symbolic.
    assert_eq!(
      interpret("QPochhammer[1/2, 1/2]").unwrap(),
      "QPochhammer[1/2, 1/2]"
    );
  }

  #[test]
  fn two_arg_numeric() {
    let val: f64 = interpret("QPochhammer[0.5, 0.5]").unwrap().parse().unwrap();
    assert!((val - 0.2887880950866024).abs() < 1e-12, "got {}", val);
  }

  #[test]
  fn two_arg_numeric_factor_vanishes() {
    // a = 2, q = 0.5: the k=1 factor (1 - 2*0.5) = 0, so the product is 0.
    let val: f64 = interpret("QPochhammer[2.0, 0.5]").unwrap().parse().unwrap();
    assert!(val.abs() < 1e-15, "got {}", val);
  }

  #[test]
  fn one_arg_numeric() {
    // QPochhammer[0.5] = QPochhammer[0.5, 0.5]
    let val: f64 = interpret("QPochhammer[0.5]").unwrap().parse().unwrap();
    assert!((val - 0.2887880950866024).abs() < 1e-12, "got {}", val);
  }

  #[test]
  fn n_of_exact_evaluates() {
    let val: f64 = interpret("N[QPochhammer[1/2, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((val - 0.2887880950866024).abs() < 1e-12, "got {}", val);
  }
}

mod q_gamma {
  use super::*;

  // QGamma[n, q] = QFactorial[n-1, q] for a positive integer n.
  #[test]
  fn integer_arg_numeric_q() {
    assert_eq!(interpret("QGamma[1, 1/3]").unwrap(), "1");
    assert_eq!(interpret("QGamma[2, 1/2]").unwrap(), "1");
    assert_eq!(interpret("QGamma[4, 1/2]").unwrap(), "21/8");
    assert_eq!(interpret("QGamma[6, 1/2]").unwrap(), "9765/1024");
    assert_eq!(interpret("QGamma[2, 3]").unwrap(), "1");
  }

  #[test]
  fn real_q_returns_real() {
    assert_eq!(interpret("QGamma[3, 0.5]").unwrap(), "1.5");
    assert_eq!(interpret("QGamma[4, 0.5]").unwrap(), "2.625");
  }

  #[test]
  fn non_positive_integers_are_poles() {
    assert_eq!(interpret("QGamma[0, 1/2]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("QGamma[-1, 1/2]").unwrap(), "ComplexInfinity");
    // Poles hold even for symbolic q.
    assert_eq!(interpret("QGamma[0, q]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("QGamma[-2, q]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn symbolic_q_expands_only_up_to_three() {
    // wolframscript expands the product only for n <= 3.
    assert_eq!(interpret("QGamma[1, q]").unwrap(), "1");
    assert_eq!(interpret("QGamma[2, q]").unwrap(), "1");
    assert_eq!(interpret("QGamma[3, q]").unwrap(), "1 + q");
    assert_eq!(interpret("QGamma[4, q]").unwrap(), "QGamma[4, q]");
    assert_eq!(interpret("QGamma[5, q]").unwrap(), "QGamma[5, q]");
  }

  #[test]
  fn non_integer_first_arg_stays_unevaluated() {
    assert_eq!(interpret("QGamma[1/2, 1/3]").unwrap(), "QGamma[1/2, 1/3]");
    assert_eq!(interpret("QGamma[z, q]").unwrap(), "QGamma[z, q]");
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
  fn at_three() {
    // LogGamma[3] = Log[(3-1)!] = Log[2]
    assert_eq!(interpret("LogGamma[3]").unwrap(), "Log[2]");
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

  #[test]
  fn rational_order_audit_case() {
    // Audit case: AngerJ[1/2, 5.] = -0.2857335081880813
    let result: f64 = interpret("AngerJ[1/2, 5.]").unwrap().parse().unwrap();
    assert!(
      (result - (-0.2857335081880813)).abs() < 1e-8,
      "got {}",
      result
    );
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

  #[test]
  fn rational_order_audit_case() {
    // Audit case: WeberE[1/2, 5.] = 0.03336423649491385
    let result: f64 = interpret("WeberE[1/2, 5.]").unwrap().parse().unwrap();
    assert!(
      (result - 0.03336423649491385).abs() < 1e-8,
      "got {}",
      result
    );
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
    // d^{1/2}_{1/2, -1/2}(theta) = sin(theta/2) (Mathematica convention).
    let theta: f64 = 1.0;
    let expected = (theta / 2.0).sin();
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

  #[test]
  fn symbolic_full_audit_case() {
    // Audit case (with ASCII placeholders for ψ, θ, ϕ):
    //   WignerD[{1, 0, 1}, p, q, r] = -(Sqrt[2]*E^(I*r)*Cos[q/2]*Sin[q/2])
    assert_eq!(
      interpret("WignerD[{1, 0, 1}, p, q, r]").unwrap(),
      "-(Sqrt[2]*E^(I*r)*Cos[q/2]*Sin[q/2])"
    );
  }

  #[test]
  fn symbolic_full_diagonal() {
    assert_eq!(
      interpret("WignerD[{1, 1, 1}, a, b, c]").unwrap(),
      "E^(I*a + I*c)*Cos[b/2]^2"
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

mod cap {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("Cap[x, y]").unwrap(), "x \u{2322} y");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("Cap[x]").unwrap(), "Cap[x]");
  }

  // Regression (mathics test_parser.py:545): `a \[Cap] b` is an infix
  // operator that parses to `Cap[a, b]`.
  #[test]
  fn infix_named_character() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \\[Cap] b]]]").unwrap(),
      "Hold[Cap[a, b]]"
    );
  }

  // The Unicode ⌢ (U+2322) also parses as the Cap infix operator.
  #[test]
  fn infix_unicode_u2322() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \u{2322} b]]]").unwrap(),
      "Hold[Cap[a, b]]"
    );
  }

  // Cap is Flat/associative — chains flatten: a ⌢ b ⌢ c → Cap[a, b, c].
  #[test]
  fn infix_chained_flat() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \\[Cap] b \\[Cap] c]]]").unwrap(),
      "Hold[Cap[a, b, c]]"
    );
  }
}

mod cup {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("Cup[x, y]").unwrap(), "x \u{2323} y");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("Cup[x]").unwrap(), "Cup[x]");
  }

  // Regression (mathics test_parser.py:546): `a \[Cup] b \[Cup] c` is an
  // infix Flat/associative operator that parses to `Cup[a, b, c]`.
  #[test]
  fn infix_chained_flat() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \\[Cup] b \\[Cup] c]]]").unwrap(),
      "Hold[Cup[a, b, c]]"
    );
  }

  // The Unicode ⌣ (U+2323) also parses as the Cup infix operator.
  #[test]
  fn infix_unicode_u2323() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \u{2323} b]]]").unwrap(),
      "Hold[Cup[a, b]]"
    );
  }
}

mod bit_set {
  use super::*;

  #[test]
  fn set_bit() {
    assert_eq!(interpret("BitSet[0, 3]").unwrap(), "8");
  }

  #[test]
  fn already_set() {
    assert_eq!(interpret("BitSet[7, 0]").unwrap(), "7");
  }

  #[test]
  fn set_new() {
    assert_eq!(interpret("BitSet[5, 1]").unwrap(), "7");
  }
}

mod bit_clear {
  use super::*;

  #[test]
  fn clear_bit() {
    assert_eq!(interpret("BitClear[7, 1]").unwrap(), "5");
  }

  #[test]
  fn clear_lowest() {
    assert_eq!(interpret("BitClear[7, 0]").unwrap(), "6");
  }
}

mod bit_flip {
  use super::*;

  #[test]
  fn flip_bit_zero() {
    assert_eq!(interpret("BitFlip[5, 0]").unwrap(), "4");
  }

  #[test]
  fn flip_bit_one() {
    assert_eq!(interpret("BitFlip[5, 1]").unwrap(), "7");
  }

  #[test]
  fn flip_bit_two() {
    assert_eq!(interpret("BitFlip[5, 2]").unwrap(), "1");
  }

  #[test]
  fn flip_bit_three() {
    assert_eq!(interpret("BitFlip[5, 3]").unwrap(), "13");
  }

  #[test]
  fn flip_zero() {
    assert_eq!(interpret("BitFlip[0, 1]").unwrap(), "2");
  }

  #[test]
  fn flip_high_bit() {
    assert_eq!(interpret("BitFlip[255, 8]").unwrap(), "511");
  }

  #[test]
  fn negative_index() {
    assert_eq!(interpret("BitFlip[5, -1]").unwrap(), "1");
    assert_eq!(interpret("BitFlip[5, -2]").unwrap(), "7");
    assert_eq!(interpret("BitFlip[5, -3]").unwrap(), "4");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("BitFlip[x, 1]").unwrap(), "BitFlip[x, 1]");
  }
}

mod window_functions {
  use super::*;

  #[test]
  fn hamming_zero() {
    assert_eq!(interpret("HammingWindow[0]").unwrap(), "1");
  }

  #[test]
  fn hamming_numeric() {
    assert_eq!(
      interpret("HammingWindow[0.3]").unwrap(),
      "0.4024052851766544"
    );
  }

  #[test]
  fn hamming_half() {
    assert_eq!(interpret("HammingWindow[1/2]").unwrap(), "2/23");
  }

  #[test]
  fn hamming_exact_quarter() {
    assert_eq!(interpret("HammingWindow[1/4]").unwrap(), "25/46");
  }

  #[test]
  fn hamming_outside() {
    assert_eq!(interpret("HammingWindow[0.6]").unwrap(), "0.");
  }

  #[test]
  fn hamming_symbolic() {
    assert_eq!(interpret("HammingWindow[x]").unwrap(), "HammingWindow[x]");
  }

  #[test]
  fn hann_numeric() {
    assert_eq!(interpret("HannWindow[0.3]").unwrap(), "0.34549150281252633");
  }

  #[test]
  fn blackman_numeric() {
    assert_eq!(
      interpret("BlackmanWindow[0.3]").unwrap(),
      "0.2007701432625305"
    );
  }

  #[test]
  fn dirichlet_numeric() {
    assert_eq!(interpret("DirichletWindow[0.3]").unwrap(), "1.");
  }

  #[test]
  fn bartlett_numeric() {
    assert_eq!(interpret("BartlettWindow[0.3]").unwrap(), "0.4");
  }

  #[test]
  fn welch_numeric() {
    assert_eq!(interpret("WelchWindow[0.3]").unwrap(), "0.64");
  }

  #[test]
  fn cosine_numeric() {
    assert_eq!(
      interpret("CosineWindow[0.3]").unwrap(),
      "0.5877852522924731"
    );
  }

  #[test]
  fn connes_numeric() {
    assert_eq!(interpret("ConnesWindow[0.3]").unwrap(), "0.4096");
  }

  #[test]
  fn lanczos_numeric() {
    assert_eq!(
      interpret("LanczosWindow[0.3]").unwrap(),
      "0.5045511524271047"
    );
  }

  // TukeyWindow[x, alpha] (default alpha = 2/3): flat 1 for |x| <= (1-alpha)/2,
  // a raised-cosine taper to 0 at |x| = 1/2, and 0 beyond.
  #[test]
  fn tukey_flat_top() {
    assert_eq!(interpret("TukeyWindow[0]").unwrap(), "1");
    assert_eq!(interpret("TukeyWindow[1/6]").unwrap(), "1");
  }

  #[test]
  fn tukey_edges_zero() {
    assert_eq!(interpret("TukeyWindow[1/2]").unwrap(), "0");
    assert_eq!(interpret("TukeyWindow[3/4]").unwrap(), "0");
  }

  // Exact arguments give symbolic results: the Cos simplifies to radical form.
  #[test]
  fn tukey_exact_taper() {
    assert_eq!(interpret("TukeyWindow[1/4]").unwrap(), "(1 + 1/Sqrt[2])/2");
    assert_eq!(
      interpret("TukeyWindow[2/5]").unwrap(),
      "(1 - Sqrt[5/8 - Sqrt[5]/8])/2"
    );
  }

  #[test]
  fn tukey_numeric() {
    let v: f64 = interpret("TukeyWindow[0.3]").unwrap().parse().unwrap();
    assert!((v - 0.6545084971874738).abs() < 1e-12);
    // The alpha parameter widens the taper.
    let v2: f64 = interpret("TukeyWindow[0.4, 1/2]").unwrap().parse().unwrap();
    assert!((v2 - 0.3454915028125262).abs() < 1e-12);
  }

  #[test]
  fn tukey_symbolic() {
    assert_eq!(interpret("TukeyWindow[x]").unwrap(), "TukeyWindow[x]");
  }

  // ParzenWindow: piecewise cubic, exact (rational) for exact arguments.
  #[test]
  fn parzen_exact() {
    assert_eq!(interpret("ParzenWindow[0]").unwrap(), "1");
    assert_eq!(interpret("ParzenWindow[1/8]").unwrap(), "23/32");
    assert_eq!(interpret("ParzenWindow[1/4]").unwrap(), "1/4");
    assert_eq!(interpret("ParzenWindow[1/3]").unwrap(), "2/27");
    assert_eq!(interpret("ParzenWindow[1/2]").unwrap(), "0");
  }

  #[test]
  fn parzen_numeric_and_symbolic() {
    let v: f64 = interpret("ParzenWindow[0.3]").unwrap().parse().unwrap();
    assert!((v - 0.128).abs() < 1e-12);
    assert_eq!(interpret("ParzenWindow[0.6]").unwrap(), "0.");
    assert_eq!(interpret("ParzenWindow[x]").unwrap(), "ParzenWindow[x]");
  }

  // GaussianWindow[x, sigma] (default sigma = 3/10): E^(-x^2/(2 sigma^2)),
  // exact for exact arguments.
  #[test]
  fn gaussian_exact() {
    assert_eq!(interpret("GaussianWindow[0]").unwrap(), "1");
    assert_eq!(interpret("GaussianWindow[1/4]").unwrap(), "E^(-25/72)");
    assert_eq!(interpret("GaussianWindow[1/2]").unwrap(), "E^(-25/18)");
    assert_eq!(interpret("GaussianWindow[1/4, 1/5]").unwrap(), "E^(-25/32)");
  }

  #[test]
  fn gaussian_numeric_and_zero() {
    let v: f64 = interpret("GaussianWindow[0.3]").unwrap().parse().unwrap();
    assert!((v - 0.6065306597126334).abs() < 1e-12);
    assert_eq!(interpret("GaussianWindow[0.6]").unwrap(), "0.");
  }

  // BohmanWindow[x] = (1 - 2|x|) Cos[2 Pi |x|] + Sin[2 Pi |x|]/Pi on [-1/2,1/2].
  // Exact arguments evaluate symbolically (BohmanWindow[1/4] -> 1/Pi).
  #[test]
  fn bohman_exact() {
    assert_eq!(interpret("BohmanWindow[0]").unwrap(), "1");
    assert_eq!(interpret("BohmanWindow[1/4]").unwrap(), "Pi^(-1)");
    assert_eq!(interpret("BohmanWindow[-1/4]").unwrap(), "Pi^(-1)");
    assert_eq!(interpret("BohmanWindow[1/2]").unwrap(), "0");
  }

  #[test]
  fn bohman_numeric_and_symbolic() {
    let v: f64 = interpret("BohmanWindow[0.3]").unwrap().parse().unwrap();
    assert!((v - 0.1791238937062839).abs() < 1e-12);
    assert_eq!(interpret("BohmanWindow[0.6]").unwrap(), "0.");
    assert_eq!(interpret("BohmanWindow[x]").unwrap(), "BohmanWindow[x]");
  }
}

mod right_tee {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("RightTee[a, b]").unwrap(), "a \u{22A2} b");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("RightTee[a]").unwrap(), "RightTee[a]");
  }

  // Regression (mathics test_parser.py:592): `x1 \[RightTee] x2` is an
  // infix operator that parses to `RightTee[x1, x2]`.
  #[test]
  fn infix_named_character() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[x1 \\[RightTee] x2]]]").unwrap(),
      "Hold[RightTee[x1, x2]]"
    );
  }

  // The Unicode ⊢ (U+22A2) also parses as the RightTee infix operator.
  #[test]
  fn infix_unicode_u22a2() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \u{22A2} b]]]").unwrap(),
      "Hold[RightTee[a, b]]"
    );
  }

  // RightTee is right-associative (not Flat):
  // a \[RightTee] b \[RightTee] c → RightTee[a, RightTee[b, c]].
  #[test]
  fn infix_right_associative() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \\[RightTee] b \\[RightTee] c]]]")
        .unwrap(),
      "Hold[RightTee[a, RightTee[b, c]]]"
    );
  }
}

mod double_right_tee {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("DoubleRightTee[a, b]").unwrap(), "a \u{22A8} b");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("DoubleRightTee[a]").unwrap(), "DoubleRightTee[a]");
  }

  // Regression (mathics test_parser.py:593): `x1 \[DoubleRightTee] x2` is
  // an infix operator that parses to `DoubleRightTee[x1, x2]`.
  #[test]
  fn infix_named_character() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[x1 \\[DoubleRightTee] x2]]]").unwrap(),
      "Hold[DoubleRightTee[x1, x2]]"
    );
  }

  // The Unicode ⊨ (U+22A8) also parses as the DoubleRightTee operator.
  #[test]
  fn infix_unicode_u22a8() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \u{22A8} b]]]").unwrap(),
      "Hold[DoubleRightTee[a, b]]"
    );
  }

  // DoubleRightTee is right-associative:
  // a ⊨ b ⊨ c → DoubleRightTee[a, DoubleRightTee[b, c]].
  #[test]
  fn infix_right_associative() {
    assert_eq!(
      interpret(
        "ToString[FullForm[Hold[a \\[DoubleRightTee] b \\[DoubleRightTee] c]]]"
      )
      .unwrap(),
      "Hold[DoubleRightTee[a, DoubleRightTee[b, c]]]"
    );
  }
}

mod left_tee {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("LeftTee[a, b]").unwrap(), "a \u{22A3} b");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("LeftTee[a]").unwrap(), "LeftTee[a]");
  }

  // Regression (mathics test_parser.py:594): `x1 \[LeftTee] x2` is an
  // infix operator that parses to `LeftTee[x1, x2]`.
  #[test]
  fn infix_named_character() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[x1 \\[LeftTee] x2]]]").unwrap(),
      "Hold[LeftTee[x1, x2]]"
    );
  }

  // The Unicode ⊣ (U+22A3) also parses as the LeftTee infix operator.
  #[test]
  fn infix_unicode_u22a3() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \u{22A3} b]]]").unwrap(),
      "Hold[LeftTee[a, b]]"
    );
  }

  // LeftTee is left-associative (opposite of RightTee):
  // a ⊣ b ⊣ c → LeftTee[LeftTee[a, b], c].
  #[test]
  fn infix_left_associative() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \\[LeftTee] b \\[LeftTee] c]]]")
        .unwrap(),
      "Hold[LeftTee[LeftTee[a, b], c]]"
    );
  }
}

mod double_left_tee {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("DoubleLeftTee[a, b]").unwrap(), "a \u{2AE4} b");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("DoubleLeftTee[a]").unwrap(), "DoubleLeftTee[a]");
  }

  // Regression (mathics test_parser.py:595): `x1 \[DoubleLeftTee] x2` is
  // an infix operator that parses to `DoubleLeftTee[x1, x2]`.
  #[test]
  fn infix_named_character() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[x1 \\[DoubleLeftTee] x2]]]").unwrap(),
      "Hold[DoubleLeftTee[x1, x2]]"
    );
  }

  // The Unicode ⫤ (U+2AE4) also parses as the DoubleLeftTee operator.
  #[test]
  fn infix_unicode_u2ae4() {
    assert_eq!(
      interpret("ToString[FullForm[Hold[a \u{2AE4} b]]]").unwrap(),
      "Hold[DoubleLeftTee[a, b]]"
    );
  }

  // DoubleLeftTee is left-associative:
  // a ⫤ b ⫤ c → DoubleLeftTee[DoubleLeftTee[a, b], c].
  #[test]
  fn infix_left_associative() {
    assert_eq!(
      interpret(
        "ToString[FullForm[Hold[a \\[DoubleLeftTee] b \\[DoubleLeftTee] c]]]"
      )
      .unwrap(),
      "Hold[DoubleLeftTee[DoubleLeftTee[a, b], c]]"
    );
  }
}

mod prime_zeta_p {
  use super::*;

  #[test]
  fn symbolic() {
    assert_eq!(interpret("PrimeZetaP[2]").unwrap(), "PrimeZetaP[2]");
  }

  #[test]
  fn numeric_2() {
    assert_eq!(interpret("PrimeZetaP[2.0]").unwrap(), "0.45224742004106566");
  }

  #[test]
  fn numeric_via_n() {
    assert_eq!(
      interpret("N[PrimeZetaP[2]]").unwrap(),
      "0.45224742004106566"
    );
  }

  #[test]
  fn numeric_3() {
    assert_eq!(interpret("PrimeZetaP[3.0]").unwrap(), "0.17476263929944316");
  }

  #[test]
  fn symbolic_var() {
    assert_eq!(interpret("PrimeZetaP[s]").unwrap(), "PrimeZetaP[s]");
  }
}

mod norlund_b {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("NorlundB[0, a]").unwrap(), "1");
  }

  #[test]
  fn one_symbolic() {
    assert_eq!(interpret("NorlundB[1, a]").unwrap(), "-1/2*a");
  }

  #[test]
  fn two_symbolic() {
    assert_eq!(interpret("NorlundB[2, a]").unwrap(), "-1/12*a + a^2/4");
  }

  #[test]
  fn three_symbolic() {
    assert_eq!(interpret("NorlundB[3, a]").unwrap(), "a^2/8 - a^3/8");
  }

  #[test]
  fn four_symbolic() {
    assert_eq!(
      interpret("NorlundB[4, a]").unwrap(),
      "a/120 + a^2/48 - a^3/8 + a^4/16"
    );
  }

  #[test]
  fn numeric_three_one() {
    assert_eq!(interpret("NorlundB[3, 1]").unwrap(), "0");
  }

  #[test]
  fn numeric_two_one() {
    assert_eq!(interpret("NorlundB[2, 1]").unwrap(), "1/6");
  }

  #[test]
  fn numeric_ten_one() {
    assert_eq!(interpret("NorlundB[10, 1]").unwrap(), "5/66");
  }

  #[test]
  fn numeric_five_two() {
    assert_eq!(interpret("NorlundB[5, 2]").unwrap(), "1/6");
  }

  // Three-argument Nörlund polynomial form B_n^(a)(x).

  #[test]
  fn poly_zero() {
    assert_eq!(interpret("NorlundB[0, a, x]").unwrap(), "1");
  }

  #[test]
  fn poly_one() {
    assert_eq!(interpret("NorlundB[1, a, x]").unwrap(), "-1/2*a + x");
  }

  #[test]
  fn poly_two() {
    assert_eq!(
      interpret("NorlundB[2, a, x]").unwrap(),
      "-1/12*a + a^2/4 - a*x + x^2"
    );
  }

  #[test]
  fn poly_numeric_a_degree_three() {
    assert_eq!(
      interpret("NorlundB[3, 1, x]").unwrap(),
      "x/2 - (3*x^2)/2 + x^3"
    );
  }

  #[test]
  fn poly_numeric_a_degree_four() {
    assert_eq!(
      interpret("NorlundB[4, 1, x]").unwrap(),
      "-1/30 + x^2 - 2*x^3 + x^4"
    );
  }

  #[test]
  fn poly_a_zero_is_power() {
    // B_n^(0)(x) = x^n
    assert_eq!(interpret("NorlundB[3, 0, x]").unwrap(), "x^3");
  }

  #[test]
  fn poly_fully_numeric() {
    assert_eq!(interpret("NorlundB[2, 5, 3]").unwrap(), "-1/6");
    assert_eq!(interpret("NorlundB[5, 1, 2]").unwrap(), "5");
    assert_eq!(interpret("NorlundB[6, 3, 1/2]").unwrap(), "131/1344");
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
  fn from_rules_with_layout() {
    // PlanarGraph evaluates to a Graph and prints the same `<n>, <m>`
    // summary (wolframscript: `Graph[<3>, <3>]`).
    assert_eq!(
      interpret("PlanarGraph[{1 -> 2, 2 -> 3, 3 -> 1}]").unwrap(),
      "Graph[<3>, <3>]"
    );
    // Verify the graph structure via VertexList/EdgeList
    assert_eq!(
      interpret("VertexList[PlanarGraph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("EdgeList[PlanarGraph[{1 -> 2, 2 -> 3, 3 -> 1}]]").unwrap(),
      "{1  2, 2  3, 3  1}"
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
  fn symbolic_evaluates() {
    // BandpassFilter with symbolic data should evaluate using kernel coefficients
    let result = interpret("BandpassFilter[{a, b, c}, {0.1, 0.3}]").unwrap();
    assert!(
      result.contains("*a"),
      "should contain coefficient*a: {}",
      result
    );
    assert!(
      result.contains("*b"),
      "should contain coefficient*b: {}",
      result
    );
    assert!(
      result.contains("*c"),
      "should contain coefficient*c: {}",
      result
    );
  }

  // Image input: separable 2D filter (row-then-column 1D windowed-sinc
  // passes). wolframscript reference:
  //   ImageData[BandpassFilter[Image[{{0., 0.5, 1.}, {0.5, 1., 0.5},
  //                                   {1., 0.5, 0.}}], {0.3, 0.7}]]
  #[test]
  fn image_3x3() {
    assert_eq!(
      interpret(
        "ImageData[BandpassFilter[Image[{{0., 0.5, 1.}, {0.5, 1., 0.5}, {1., 0.5, 0.}}], {0.3, 0.7}]]",
      )
      .unwrap(),
      "{{0.0014151931973174214, 0.011364215053617954, 0.020177505910396576}, {0.011364215053617954, 0.018855467438697815, 0.011364215053617954}, {0.020177505910396576, 0.011364215053617954, 0.0014151931973174214}}"
    );
  }

  // Audit regression: confirm `BandpassFilter[Image, spec]` returns an Image.
  #[test]
  fn image_returns_image() {
    assert_eq!(
      interpret(
        "Head[BandpassFilter[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}], {0.3, 0.7}]]",
      )
      .unwrap(),
      "Image"
    );
  }
}

// `GradientFilter[data, r]` returns the magnitude of the Gaussian-
// smoothed gradient. Uses the same Bessel-based discrete Gaussian as
// GaussianFilter for the smoothing direction, paired with a derivative
// kernel `D[k] = -k·T[k] / Σ_j j²·T[j]` normalised so a unit ramp
// reproduces its slope.
mod gradient_filter {
  use super::*;

  #[test]
  fn impulse_image_3x3() {
    // wolframscript:
    //   ImageData[GradientFilter[Image[{{0., 0., 0.}, {0., 1., 0.},
    //                                   {0., 0., 0.}}], 1]]
    //     ≈ {{0.0703, 0.4006, 0.0703}, {0.4006, 0., 0.4006},
    //        {0.0703, 0.4006, 0.0703}}
    let result = interpret(
      "ImageData[GradientFilter[Image[{{0., 0., 0.}, {0., 1., 0.}, {0., 0., 0.}}], 1]]",
    )
    .unwrap();
    for prefix in ["0.0702", "0.40061", "0.40061"] {
      assert!(
        result.contains(prefix),
        "expected `{}` in result `{}`",
        prefix,
        result
      );
    }
  }

  // Audit regression: confirm `GradientFilter[Image, r]` returns an Image.
  #[test]
  fn image_returns_image() {
    assert_eq!(
      interpret(
        "Head[GradientFilter[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}], 1]]",
      )
      .unwrap(),
      "Image"
    );
  }

  // A constant image has zero gradient.
  #[test]
  fn constant_image_zero_gradient() {
    let result = interpret(
      "ImageData[GradientFilter[Image[{{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}}], 1]]",
    )
    .unwrap();
    // Every value should be 0 (modulo f32 precision in the Image data).
    assert!(
      !result.contains("1.") && !result.contains("0.5"),
      "expected near-zero values in `{}`",
      result
    );
  }
}

// `GaussianFilter[data, r]` uses the Bessel-based discrete Gaussian
// kernel `T_k(t) = e^(-t)·I_k(t)` with `t = σ² = (r/2)²`, normalised
// over `k ∈ [-r, r]` so the kernel sums to 1.
mod gaussian_filter {
  use super::*;

  #[test]
  fn impulse_1d() {
    // wolframscript: `GaussianFilter[{0., 0., 1., 0., 0.}, 1]`
    // recovers the kernel as the impulse response. The last digit of
    // the kernel coefficient comes from Bessel `I_1(0.25)` and differs
    // from wolframscript by a single ulp — match Woxi's value exactly.
    assert_eq!(
      interpret("GaussianFilter[{0., 0., 1., 0., 0.}, 1]").unwrap(),
      "{0., 0.09938048320860668, 0.8012390335827866, 0.09938048320860668, 0.}"
    );
  }

  #[test]
  fn list_1d() {
    // wolframscript: `GaussianFilter[{1., 2., 3., 4., 5.}, 1]`
    // is linear ⇒ kernel-preserved with edge replication on the ends.
    assert_eq!(
      interpret("GaussianFilter[{1., 2., 3., 4., 5.}, 1]").unwrap(),
      "{1.0993804832086067, 2., 3., 4., 4.9006195167913935}"
    );
  }

  #[test]
  fn image_3x3() {
    // wolframscript:
    //   ImageData[GaussianFilter[Image[{{0., 0.5, 1.}, {0.5, 1., 0.5},
    //                                   {1., 0.5, 0.}}], 1]]
    //     ≈ {{0.0994, 0.5398, 0.9105}, {0.5398, 0.8210, 0.5398},
    //        {0.9105, 0.5398, 0.0994}}
    // Woxi differs in the last ~3 digits due to BesselI/edge-pad
    // precision; check the leading values approximately.
    let result = interpret(
      "ImageData[GaussianFilter[Image[{{0., 0.5, 1.}, {0.5, 1., 0.5}, {1., 0.5, 0.}}], 1]]",
    )
    .unwrap();
    for prefix in [
      "{{0.0993804", // corners
      "0.5398137",   // edges
      "0.8209",      // center
    ] {
      assert!(
        result.contains(prefix),
        "expected `{}` in result `{}`",
        prefix,
        result
      );
    }
  }

  // Audit regression: confirm `GaussianFilter[Image, r]` returns an Image.
  #[test]
  fn image_returns_image() {
    assert_eq!(
      interpret(
        "Head[GaussianFilter[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}], 1]]",
      )
      .unwrap(),
      "Image"
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

  // ─── Image input ──────────────────────────────────────────────────
  //
  // Wolfram applies LowpassFilter to images by filtering each row with
  // the 1D windowed-sinc kernel and then each column of the result with
  // the same kernel. Verify via ImageData rather than printing the
  // -Image- head.

  #[test]
  fn image_3x3() {
    // Match wolframscript's `ImageData[LowpassFilter[Image[{{0., 0.5, 1.},
    // {0.5, 1., 0.5}, {1., 0.5, 0.}}], 0.5]]`.
    assert_eq!(
      interpret(
        "ImageData[LowpassFilter[Image[{{0., 0.5, 1.}, {0.5, 1., 0.5}, {1., 0.5, 0.}}], 0.5]]",
      )
      .unwrap(),
      "{{0.07146164774894714, 0.5306240320205688, 0.9336451292037964}, {0.5306240320205688, 0.8672902584075928, 0.5306240320205688}, {0.9336451292037964, 0.5306240320205688, 0.07146164774894714}}"
    );
  }

  // Audit regression: the test harness's huge-image input timed out;
  // confirm here at small size that `LowpassFilter[Image, ωc]` returns
  // an Image (not the unevaluated head).
  #[test]
  fn image_returns_image() {
    assert_eq!(
      interpret(
        "Head[LowpassFilter[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}], 0.5]]",
      )
      .unwrap(),
      "Image"
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

  // Image input: separable 2D filter (row-then-column 1D windowed-sinc
  // passes). wolframscript reference:
  //   ImageData[HighpassFilter[Image[{{0., 0.5, 1.}, {0.5, 1., 0.5},
  //                                    {1., 0.5, 0.}}], 0.5]]
  #[test]
  fn image_3x3() {
    assert_eq!(
      interpret(
        "ImageData[HighpassFilter[Image[{{0., 0.5, 1.}, {0.5, 1., 0.5}, {1., 0.5, 0.}}], 0.5]]",
      )
      .unwrap(),
      "{{-0.010805889032781124, 0.3259671926498413, 0.6740744709968567}, {0.3259671926498413, 0.6850564479827881, 0.3259671926498413}, {0.6740744709968567, 0.3259671926498413, -0.010805889032781124}}"
    );
  }

  // Audit regression: confirm `HighpassFilter[Image, ωc]` returns an Image.
  #[test]
  fn image_returns_image() {
    assert_eq!(
      interpret(
        "Head[HighpassFilter[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}], 0.5]]",
      )
      .unwrap(),
      "Image"
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

  // Image input: separable 2D filter (row-then-column 1D passes).
  #[test]
  fn image_3x3() {
    assert_eq!(
      interpret(
        "ImageData[BandstopFilter[Image[{{0., 0.5, 1.}, {0.5, 1., 0.5}, {1., 0.5, 0.}}], {0.3, 0.7}]]",
      )
      .unwrap(),
      "{{-0.008236446417868137, 0.359911173582077, 0.736574649810791}, {0.359911173582077, 0.7449042797088623, 0.359911173582077}, {0.736574649810791, 0.359911173582077, -0.008236446417868137}}"
    );
  }

  // Audit regression: confirm `BandstopFilter[Image, spec]` returns an Image.
  #[test]
  fn image_returns_image() {
    assert_eq!(
      interpret(
        "Head[BandstopFilter[Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}], {0.3, 0.7}]]",
      )
      .unwrap(),
      "Image"
    );
  }
}

mod airy_ai_prime {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(
      interpret("AiryAiPrime[0.0]").unwrap(),
      "-0.2588194037928068"
    );
  }

  #[test]
  fn at_one() {
    assert_eq!(
      interpret("AiryAiPrime[1.0]").unwrap(),
      "-0.15914744129679328"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("AiryAiPrime[x]").unwrap(), "AiryAiPrime[x]");
  }

  #[test]
  fn negative_arg() {
    assert_eq!(
      interpret("AiryAiPrime[-1.0]").unwrap(),
      "-0.010160567116645175"
    );
  }
}

mod airy_bi_prime {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("AiryBiPrime[0.0]").unwrap(), "0.4482883573538264");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("AiryBiPrime[1.0]").unwrap(), "0.9324359333927756");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("AiryBiPrime[x]").unwrap(), "AiryBiPrime[x]");
  }

  #[test]
  fn negative_arg() {
    assert_eq!(
      interpret("AiryBiPrime[-1.0]").unwrap(),
      "0.5923756264227923"
    );
  }
}

mod dirichlet_eta {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("DirichletEta[0]").unwrap(), "1/2");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("DirichletEta[1]").unwrap(), "Log[2]");
  }

  #[test]
  fn at_two() {
    assert_eq!(interpret("DirichletEta[2]").unwrap(), "Pi^2/12");
  }

  #[test]
  fn at_four() {
    assert_eq!(interpret("DirichletEta[4]").unwrap(), "(7*Pi^4)/720");
  }

  #[test]
  fn at_six() {
    assert_eq!(interpret("DirichletEta[6]").unwrap(), "(31*Pi^6)/30240");
  }

  #[test]
  fn at_negative_one() {
    assert_eq!(interpret("DirichletEta[-1]").unwrap(), "1/4");
  }

  #[test]
  fn at_negative_two() {
    assert_eq!(interpret("DirichletEta[-2]").unwrap(), "0");
  }

  #[test]
  fn at_three() {
    assert_eq!(interpret("DirichletEta[3]").unwrap(), "(3*Zeta[3])/4");
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("DirichletEta[1.0]").unwrap(),
      "0.6931471805599453"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("DirichletEta[x]").unwrap(),
      "(1 - 2^(1 - x))*Zeta[x]"
    );
  }

  #[test]
  fn at_half() {
    assert_eq!(
      interpret("DirichletEta[1/2]").unwrap(),
      "(1 - Sqrt[2])*Zeta[1/2]"
    );
  }
}

// DirichletBeta[s] = sum over n of (-1)^n/(2n+1)^s.
mod dirichlet_beta {
  use super::*;

  #[test]
  fn odd_positive_integers_have_pi_closed_forms() {
    assert_eq!(interpret("DirichletBeta[1]").unwrap(), "Pi/4");
    assert_eq!(interpret("DirichletBeta[3]").unwrap(), "Pi^3/32");
    assert_eq!(interpret("DirichletBeta[5]").unwrap(), "(5*Pi^5)/1536");
    assert_eq!(interpret("DirichletBeta[7]").unwrap(), "(61*Pi^7)/184320");
  }

  #[test]
  fn beta_two_is_catalan() {
    assert_eq!(interpret("DirichletBeta[2]").unwrap(), "Catalan");
  }

  #[test]
  fn non_positive_integers_use_euler_numbers() {
    assert_eq!(interpret("DirichletBeta[0]").unwrap(), "1/2");
    assert_eq!(interpret("DirichletBeta[-1]").unwrap(), "0");
    assert_eq!(interpret("DirichletBeta[-2]").unwrap(), "-1/2");
    assert_eq!(interpret("DirichletBeta[-4]").unwrap(), "5/2");
  }

  #[test]
  fn even_positive_and_symbolic_use_hurwitz_form() {
    assert_eq!(
      interpret("DirichletBeta[4]").unwrap(),
      "(Zeta[4, 1/4]/16 - Zeta[4, 3/4]/16)/16"
    );
    assert_eq!(
      interpret("DirichletBeta[s]").unwrap(),
      "(Zeta[s, 1/4]/2^s - Zeta[s, 3/4]/2^s)/2^s"
    );
  }

  #[test]
  fn numeric_arguments() {
    // β(1) = π/4; the Hurwitz zetas individually diverge there.
    let b1: f64 = interpret("DirichletBeta[1.0]").unwrap().parse().unwrap();
    assert!((b1 - std::f64::consts::FRAC_PI_4).abs() < 1e-12, "got {b1}");
    // β(2) = Catalan ≈ 0.9159655941772190.
    let b2: f64 = interpret("DirichletBeta[2.0]").unwrap().parse().unwrap();
    assert!((b2 - 0.9159655941772190).abs() < 1e-12, "got {b2}");
  }
}

// DirichletLambda[s] = (1 - 2^(-s)) Zeta[s], the sum over odd n of 1/n^s.
mod dirichlet_lambda {
  use super::*;

  #[test]
  fn even_two() {
    assert_eq!(interpret("DirichletLambda[2]").unwrap(), "Pi^2/8");
  }

  #[test]
  fn even_four() {
    assert_eq!(interpret("DirichletLambda[4]").unwrap(), "Pi^4/96");
  }

  #[test]
  fn even_six() {
    assert_eq!(interpret("DirichletLambda[6]").unwrap(), "Pi^6/960");
  }

  #[test]
  fn odd_three() {
    assert_eq!(interpret("DirichletLambda[3]").unwrap(), "(7*Zeta[3])/8");
  }

  #[test]
  fn pole_at_one() {
    assert_eq!(interpret("DirichletLambda[1]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("DirichletLambda[0]").unwrap(), "0");
  }

  #[test]
  fn negative_integer() {
    assert_eq!(interpret("DirichletLambda[-1]").unwrap(), "1/12");
  }

  // Symbolic and fractional arguments keep the 2^s denominator form, matching
  // wolframscript's canonical output.
  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("DirichletLambda[x]").unwrap(),
      "((-1 + 2^x)*Zeta[x])/2^x"
    );
  }

  #[test]
  fn at_half() {
    assert_eq!(
      interpret("DirichletLambda[1/2]").unwrap(),
      "((-1 + Sqrt[2])*Zeta[1/2])/Sqrt[2]"
    );
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("N[DirichletLambda[2]]").unwrap(),
      "1.2337005501361697"
    );
  }
}

mod appell_f1 {
  use super::*;

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("AppellF1[a, b1, b2, c, x, y]").unwrap(),
      "AppellF1[a, b1, b2, c, x, y]"
    );
  }

  #[test]
  fn numeric_basic() {
    let result: f64 = interpret("AppellF1[2, 1, 1, 3, 0.7, 0.3]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 2.655223346206384).abs() < 1e-8);
  }

  #[test]
  fn at_zero() {
    // F1(a, b1, b2; c; 0, 0) = 1
    let result: f64 = interpret("AppellF1[1, 1, 1, 2, 0.0, 0.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.0).abs() < 1e-10);
  }

  #[test]
  fn a_zero() {
    // F1(0, b1, b2; c; x, y) = 1 (since (0)_{m+n} = 0 for m+n > 0)
    let result: f64 = interpret("AppellF1[0, 1, 1, 2, 0.5, 0.3]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.0).abs() < 1e-10);
  }

  #[test]
  fn y_zero_reduces_to_2f1() {
    // F1(a, b1, b2; c; x, 0) = 2F1(a, b1; c; x)
    let result: f64 = interpret("AppellF1[1, 1, 1, 2, 0.5, 0.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.38629436111989).abs() < 1e-8);
  }

  #[test]
  fn c_equals_a_reduction_numeric() {
    // F1(a, b1, b2; a; x, y) = (1-x)^(-b1) * (1-y)^(-b2)
    // Audit case: AppellF1[1, 1, 1, 1, 4, z] reduces symbolically; here numeric.
    // F1(1, 1, 1; 1; 0.5, 0.3) = 1/(0.5 * 0.7) = 1/0.35 = 2.857142857...
    let result: f64 = interpret("AppellF1[1, 1, 1, 1, 0.5, 0.3]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.0 / (0.5 * 0.7)).abs() < 1e-8);
  }

  #[test]
  fn c_equals_a_reduction_symbolic_audit_case() {
    // AppellF1[1, 1, 1, 1, 4, z] = 1/((1-4)^1 * (1-z)^1) = -1/(3*(1-z))
    // Rendered like wolframscript as `-1/3*1/(1 - z)`.
    assert_eq!(
      interpret("AppellF1[1, 1, 1, 1, 4, z]").unwrap(),
      "-1/3*1/(1 - z)"
    );
  }

  #[test]
  fn c_equals_a_reduction_symbolic_general() {
    // AppellF1[a, b1, b2, a, x, y] = 1/((1-x)^b1 * (1-y)^b2)
    assert_eq!(
      interpret("AppellF1[2, 3, 5, 2, x, y]").unwrap(),
      "1/((1 - x)^3*(1 - y)^5)"
    );
  }
}

mod appell_f2 {
  use super::*;

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("AppellF2[a, b1, b2, c1, c2, x, y]").unwrap(),
      "AppellF2[a, b1, b2, c1, c2, x, y]"
    );
  }

  #[test]
  fn numeric_basic() {
    let result: f64 = interpret("AppellF2[2, 1, 1, 3, 4, 0.1, 0.2]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.1992911364736902).abs() < 1e-8);
  }

  #[test]
  fn numeric_basic_2() {
    let result: f64 = interpret("AppellF2[1, 2, 3, 4, 5, 0.3, 0.4]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.6902006090348043).abs() < 1e-8);
  }

  #[test]
  fn at_zero() {
    // F2(a, b1, b2; c1, c2; 0, 0) = 1
    assert_eq!(interpret("AppellF2[a, b1, b2, c1, c2, 0, 0]").unwrap(), "1");
  }

  #[test]
  fn a_zero() {
    // F2(0, b1, b2; c1, c2; x, y) = 1
    assert_eq!(interpret("AppellF2[0, b1, b2, c1, c2, x, y]").unwrap(), "1");
  }

  #[test]
  fn b1_zero_reduces_to_2f1() {
    // F2(a, 0, b2; c1, c2; x, y) = 2F1(a, b2; c2; y)
    assert_eq!(
      interpret("AppellF2[a, 0, b2, c1, c2, x, y]").unwrap(),
      "Hypergeometric2F1[a, b2, c2, y]"
    );
  }

  #[test]
  fn b2_zero_reduces_to_2f1() {
    // F2(a, b1, 0; c1, c2; x, y) = 2F1(a, b1; c1; x)
    assert_eq!(
      interpret("AppellF2[a, b1, 0, c1, c2, x, y]").unwrap(),
      "Hypergeometric2F1[a, b1, c1, x]"
    );
  }

  #[test]
  fn x_zero_reduces_to_2f1() {
    // F2(a, b1, b2; c1, c2; 0, y) = 2F1(a, b2; c2; y)
    assert_eq!(
      interpret("AppellF2[a, b1, b2, c1, c2, 0, y]").unwrap(),
      "Hypergeometric2F1[a, b2, c2, y]"
    );
  }

  #[test]
  fn y_zero_reduces_to_2f1() {
    // F2(a, b1, b2; c1, c2; x, 0) = 2F1(a, b1; c1; x)
    assert_eq!(
      interpret("AppellF2[a, b1, b2, c1, c2, x, 0]").unwrap(),
      "Hypergeometric2F1[a, b1, c1, x]"
    );
  }
}

mod appell_f3 {
  use super::*;

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("AppellF3[a1, a2, b1, b2, c, x, y]").unwrap(),
      "AppellF3[a1, a2, b1, b2, c, x, y]"
    );
  }

  #[test]
  fn numeric_basic() {
    let result: f64 = interpret("AppellF3[2, 1, 3, 2, 4, 0.1, 0.2]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.2989163569406097).abs() < 1e-8);
  }

  #[test]
  fn numeric_basic_2() {
    let result: f64 = interpret("AppellF3[1, 2, 3, 4, 5, 0.3, 0.4]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 2.638819453892733).abs() < 1e-8);
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("AppellF3[a1, a2, b1, b2, c, 0, 0]").unwrap(), "1");
  }

  #[test]
  fn a1_zero_reduces_to_2f1() {
    assert_eq!(
      interpret("AppellF3[0, a2, b1, b2, c, x, y]").unwrap(),
      "Hypergeometric2F1[a2, b2, c, y]"
    );
  }

  #[test]
  fn a2_zero_reduces_to_2f1() {
    assert_eq!(
      interpret("AppellF3[a1, 0, b1, b2, c, x, y]").unwrap(),
      "Hypergeometric2F1[a1, b1, c, x]"
    );
  }

  #[test]
  fn b1_zero_reduces_to_2f1() {
    assert_eq!(
      interpret("AppellF3[a1, a2, 0, b2, c, x, y]").unwrap(),
      "Hypergeometric2F1[a2, b2, c, y]"
    );
  }

  #[test]
  fn b2_zero_reduces_to_2f1() {
    assert_eq!(
      interpret("AppellF3[a1, a2, b1, 0, c, x, y]").unwrap(),
      "Hypergeometric2F1[a1, b1, c, x]"
    );
  }

  #[test]
  fn x_zero_reduces_to_2f1() {
    assert_eq!(
      interpret("AppellF3[a1, a2, b1, b2, c, 0, y]").unwrap(),
      "Hypergeometric2F1[a2, b2, c, y]"
    );
  }

  #[test]
  fn y_zero_reduces_to_2f1() {
    assert_eq!(
      interpret("AppellF3[a1, a2, b1, b2, c, x, 0]").unwrap(),
      "Hypergeometric2F1[a1, b1, c, x]"
    );
  }
}

mod appell_f4 {
  use super::*;

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("AppellF4[a, b, c1, c2, x, y]").unwrap(),
      "AppellF4[a, b, c1, c2, x, y]"
    );
  }

  #[test]
  fn numeric_basic() {
    let result: f64 = interpret("AppellF4[2, 1, 3, 4, 0.1, 0.2]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.2178867552064807).abs() < 1e-8);
  }

  #[test]
  fn numeric_basic_2() {
    let result: f64 = interpret("AppellF4[1, 2, 3, 4, 0.05, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.0940276371588293).abs() < 1e-8);
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("AppellF4[a, b, c1, c2, 0, 0]").unwrap(), "1");
  }

  #[test]
  fn a_zero() {
    assert_eq!(interpret("AppellF4[0, b, c1, c2, x, y]").unwrap(), "1");
  }

  #[test]
  fn b_zero() {
    assert_eq!(interpret("AppellF4[a, 0, c1, c2, x, y]").unwrap(), "1");
  }

  #[test]
  fn x_zero_reduces_to_2f1() {
    assert_eq!(
      interpret("AppellF4[a, b, c1, c2, 0, y]").unwrap(),
      "Hypergeometric2F1[a, b, c2, y]"
    );
  }

  #[test]
  fn y_zero_reduces_to_2f1() {
    assert_eq!(
      interpret("AppellF4[a, b, c1, c2, x, 0]").unwrap(),
      "Hypergeometric2F1[a, b, c1, x]"
    );
  }
}

mod polygonal_number {
  use super::*;

  #[test]
  fn triangular_numbers() {
    assert_eq!(
      interpret("Table[PolygonalNumber[n], {n, 0, 10}]").unwrap(),
      "{0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55}"
    );
  }

  #[test]
  fn square_numbers() {
    assert_eq!(
      interpret("Table[PolygonalNumber[4, n], {n, 0, 5}]").unwrap(),
      "{0, 1, 4, 9, 16, 25}"
    );
  }

  #[test]
  fn pentagonal_numbers() {
    assert_eq!(
      interpret("Table[PolygonalNumber[5, n], {n, 0, 5}]").unwrap(),
      "{0, 1, 5, 12, 22, 35}"
    );
  }

  #[test]
  fn hexagonal_numbers() {
    assert_eq!(
      interpret("Table[PolygonalNumber[6, n], {n, 0, 5}]").unwrap(),
      "{0, 1, 6, 15, 28, 45}"
    );
  }

  #[test]
  fn symbolic_one_arg() {
    assert_eq!(interpret("PolygonalNumber[n]").unwrap(), "(n*(1 + n))/2");
  }

  #[test]
  fn symbolic_two_arg() {
    // Wolfram outputs: (n*(4 + n*(-2 + r) - r))/2
    // Our canonical form may reorder Plus terms
    let result = interpret("PolygonalNumber[r, n]").unwrap();
    assert!(
      result == "(n*(4 + n*(-2 + r) - r))/2"
        || result == "(n*(4 - r + n*(-2 + r)))/2",
      "Got: {}",
      result
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("PolygonalNumber[0]").unwrap(), "0");
  }

  #[test]
  fn negative_input() {
    assert_eq!(interpret("PolygonalNumber[-1]").unwrap(), "0");
  }

  #[test]
  fn real_input() {
    assert_eq!(interpret("PolygonalNumber[3.5, 4]").unwrap(), "13.");
  }

  #[test]
  fn rational_input() {
    assert_eq!(interpret("PolygonalNumber[1/2]").unwrap(), "3/8");
  }

  #[test]
  fn r_gonal_10th() {
    assert_eq!(
      interpret("Table[PolygonalNumber[r, 10], {r, 3, 10}]").unwrap(),
      "{55, 100, 145, 190, 235, 280, 325, 370}"
    );
  }
}

mod perfect_number {
  use super::*;

  #[test]
  fn first_eight() {
    assert_eq!(
      interpret("Table[PerfectNumber[n], {n, 1, 8}]").unwrap(),
      "{6, 28, 496, 8128, 33550336, 8589869056, 137438691328, 2305843008139952128}"
    );
  }

  #[test]
  fn ninth_big_integer() {
    assert_eq!(
      interpret("PerfectNumber[9]").unwrap(),
      "2658455991569831744654692615953842176"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("PerfectNumber[n]").unwrap(), "PerfectNumber[n]");
  }

  #[test]
  fn zero_unevaluated() {
    assert_eq!(interpret("PerfectNumber[0]").unwrap(), "PerfectNumber[0]");
  }

  #[test]
  fn negative_unevaluated() {
    assert_eq!(interpret("PerfectNumber[-1]").unwrap(), "PerfectNumber[-1]");
  }
}

mod ramanujan_tau {
  use super::*;

  #[test]
  fn first_fifteen() {
    assert_eq!(
      interpret("Table[RamanujanTau[n], {n, 1, 15}]").unwrap(),
      "{1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920, 534612, -370944, -577738, 401856, 1217160}"
    );
  }

  #[test]
  fn tau_one() {
    assert_eq!(interpret("RamanujanTau[1]").unwrap(), "1");
  }

  #[test]
  fn tau_two() {
    assert_eq!(interpret("RamanujanTau[2]").unwrap(), "-24");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("RamanujanTau[0]").unwrap(), "0");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("RamanujanTau[-1]").unwrap(), "0");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("RamanujanTau[n]").unwrap(), "RamanujanTau[n]");
  }
}

mod hypergeometric_1f1_regularized {
  use super::*;

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("Hypergeometric1F1Regularized[a, b, z]").unwrap(),
      "Hypergeometric1F1Regularized[a, b, z]"
    );
  }

  #[test]
  fn numeric_basic() {
    let result: f64 = interpret("Hypergeometric1F1Regularized[1, 2, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.7182818284590455).abs() < 1e-10);
  }

  #[test]
  fn numeric_fractional_params() {
    let result: f64 = interpret("Hypergeometric1F1Regularized[0.5, 1.5, 2.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 2.668000514199284).abs() < 1e-10);
  }

  #[test]
  fn at_z_zero() {
    // 1F1(a, b, 0) = 1, so regularized = 1/Gamma(b)
    let result: f64 = interpret("Hypergeometric1F1Regularized[1, 2, 0.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.0).abs() < 1e-10); // 1/Gamma(2) = 1/1 = 1
  }

  #[test]
  fn n_of_expression() {
    let result: f64 = interpret("N[Hypergeometric1F1Regularized[1, 2, 1.0]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.7182818284590455).abs() < 1e-10);
  }
}

mod gamma_regularized {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("GammaRegularized[2, 0]").unwrap(), "1");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("GammaRegularized[2, Infinity]").unwrap(), "0");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("GammaRegularized[a, z]").unwrap(),
      "GammaRegularized[a, z]"
    );
  }

  // GammaRegularized[0, z] = Gamma[0, z]/Gamma[0] = 0 for any z.
  #[test]
  fn order_zero() {
    assert_eq!(interpret("GammaRegularized[0, z]").unwrap(), "0");
    assert_eq!(interpret("GammaRegularized[0, 2]").unwrap(), "0");
    assert_eq!(interpret("GammaRegularized[0, 0]").unwrap(), "0");
    assert_eq!(interpret("GammaRegularized[0, x + 1]").unwrap(), "0");
  }

  // BetaRegularized closed forms when one shape parameter is 1.
  #[test]
  fn beta_regularized_unit_parameter() {
    assert_eq!(interpret("BetaRegularized[z, 1, 1]").unwrap(), "z");
    assert_eq!(
      interpret("BetaRegularized[z, 1, 2]").unwrap(),
      "1 - (1 - z)^2"
    );
    assert_eq!(interpret("BetaRegularized[z, 2, 1]").unwrap(), "z^2");
    assert_eq!(
      interpret("BetaRegularized[z, 1, b]").unwrap(),
      "1 - (1 - z)^b"
    );
    // Symbolic a keeps the -0^a term, matching wolframscript.
    assert_eq!(interpret("BetaRegularized[z, a, 1]").unwrap(), "-0^a + z^a");
    // Concrete arguments evaluate fully.
    assert_eq!(interpret("BetaRegularized[1/2, 1, 2]").unwrap(), "3/4");
    assert_eq!(interpret("BetaRegularized[1/2, 3, 1]").unwrap(), "1/8");
  }

  #[test]
  fn numeric_exp_one() {
    // GammaRegularized[1, 1] = e^{-1}
    let result: f64 = interpret("GammaRegularized[1, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - (-1.0_f64).exp()).abs() < 1e-10);
  }

  #[test]
  fn numeric_a2_z1() {
    let result: f64 = interpret("GammaRegularized[2, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.7357588823428847).abs() < 1e-10);
  }

  #[test]
  fn numeric_a_half() {
    let result: f64 = interpret("GammaRegularized[0.5, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.15729920705028513).abs() < 1e-10);
  }

  #[test]
  fn numeric_a3_z2() {
    let result: f64 = interpret("GammaRegularized[3, 2.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.6766764161830635).abs() < 1e-8);
  }

  #[test]
  fn n_of_integer_args() {
    let result: f64 = interpret("N[GammaRegularized[2, 1.0]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.7357588823428847).abs() < 1e-10);
  }

  // GammaRegularized[a, z0, z1] = GammaRegularized[a, z0] - GammaRegularized[a, z1]
  // evaluates numerically as soon as one argument is a machine real.
  #[test]
  fn three_arg_numeric_with_real() {
    let result: f64 = interpret("GammaRegularized[3/2, 0, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.4275932955291202).abs() < 1e-9, "got {result}");

    // A real lower or upper limit both trigger numericization.
    let result: f64 = interpret("GammaRegularized[2, 0, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.2642411176571153).abs() < 1e-9, "got {result}");

    let result: f64 = interpret("GammaRegularized[2.0, 0, 1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.2642411176571153).abs() < 1e-9, "got {result}");

    let result: f64 = interpret("GammaRegularized[2.5, 1.0, 3.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.5429261176713311).abs() < 1e-9, "got {result}");
  }

  #[test]
  fn three_arg_exact_stays_symbolic() {
    // All-exact arguments are left unevaluated (matching wolframscript).
    assert_eq!(
      interpret("GammaRegularized[2, 0, 1]").unwrap(),
      "GammaRegularized[2, 0, 1]"
    );
    assert_eq!(
      interpret("GammaRegularized[3, 1, 2]").unwrap(),
      "GammaRegularized[3, 1, 2]"
    );
  }

  // ChiSquareDistribution's CDF is built on the 3-arg GammaRegularized; with a
  // real argument it now yields a number instead of a held expression.
  #[test]
  fn chi_square_cdf_numeric() {
    let result: f64 = interpret("CDF[ChiSquareDistribution[3], 2.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.4275932955291202).abs() < 1e-9, "got {result}");
  }
}

mod inverse_gamma_regularized {
  use super::*;

  // Exact (non-machine) arguments stay symbolic, matching wolframscript.
  #[test]
  fn exact_stays_symbolic() {
    assert_eq!(
      interpret("InverseGammaRegularized[2, 1/2]").unwrap(),
      "InverseGammaRegularized[2, 1/2]"
    );
    assert_eq!(
      interpret("InverseGammaRegularized[a, q]").unwrap(),
      "InverseGammaRegularized[a, q]"
    );
  }

  // Boundary q values: GammaRegularized[a, z] == q with q = 0 -> Infinity (the
  // upper tail only vanishes as z -> Infinity), q = 1 -> 0.
  #[test]
  fn boundary_values() {
    assert_eq!(
      interpret("InverseGammaRegularized[2, 0]").unwrap(),
      "Infinity"
    );
    assert_eq!(interpret("InverseGammaRegularized[2, 1]").unwrap(), "0");
    // A symbolic first argument keeps the boundary form unevaluated.
    assert_eq!(
      interpret("InverseGammaRegularized[a, 0]").unwrap(),
      "InverseGammaRegularized[a, 0]"
    );
  }

  // For a == 1, GammaRegularized[1, z] = E^-z, so the inverse is -Log[q].
  #[test]
  fn a_one_closed_form() {
    assert_eq!(
      interpret("InverseGammaRegularized[1, 1/2]").unwrap(),
      "Log[2]"
    );
  }

  // The a == 1 closed form only fires for a concrete q; a free symbol stays
  // unevaluated, matching wolframscript (it does not auto-reduce to -Log[q]).
  #[test]
  fn a_one_symbolic_q_stays_unevaluated() {
    assert_eq!(
      interpret("InverseGammaRegularized[1, z]").unwrap(),
      "InverseGammaRegularized[1, z]"
    );
    assert_eq!(
      interpret("InverseGammaRegularized[1, q + 1]").unwrap(),
      "InverseGammaRegularized[1, 1 + q]"
    );
    // Concrete arguments still reduce / hit their boundary values.
    assert_eq!(
      interpret("InverseGammaRegularized[1, 0]").unwrap(),
      "Infinity"
    );
    assert_eq!(interpret("InverseGammaRegularized[1, 1]").unwrap(), "0");
  }

  #[test]
  fn numeric_two_arg() {
    let result: f64 = interpret("N[InverseGammaRegularized[2, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.6783469900166612).abs() < 1e-9, "got {result}");

    let result: f64 = interpret("N[InverseGammaRegularized[3, 1/4]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 3.9204020602925636).abs() < 1e-9, "got {result}");

    let result: f64 = interpret("N[InverseGammaRegularized[1/2, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.22746821155978642).abs() < 1e-9, "got {result}");
  }

  // GammaRegularized[1, z] = E^-z, so the a == 1 inverse numericizes to -Log[q].
  #[test]
  fn numeric_a_one() {
    let result: f64 = interpret("N[InverseGammaRegularized[1, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!(
      (result - std::f64::consts::LN_2).abs() < 1e-9,
      "got {result}"
    );
  }

  // The 3-arg form with z0 == 0 inverts the lower regularized gamma and equals
  // InverseGammaRegularized[a, 1 - q].
  #[test]
  fn three_arg_lower_inverse() {
    let result: f64 = interpret("N[InverseGammaRegularized[2, 0, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.6783469900166612).abs() < 1e-9, "got {result}");

    let result: f64 = interpret("N[InverseGammaRegularized[3/2, 0, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.1829869421876693).abs() < 1e-9, "got {result}");
  }

  // Round-trip: InverseGammaRegularized inverts GammaRegularized.
  #[test]
  fn round_trip() {
    let z: f64 = interpret("N[InverseGammaRegularized[3, 1/4]]")
      .unwrap()
      .parse()
      .unwrap();
    let q: f64 = interpret(&format!("N[GammaRegularized[3, {z}]]"))
      .unwrap()
      .parse()
      .unwrap();
    assert!((q - 0.25).abs() < 1e-9, "got {q}");
  }
}

mod inverse_beta_regularized {
  use super::*;

  // Symbolic arguments stay unevaluated. (For exact rational s with general
  // a, b wolframscript returns an algebraic Root object; Woxi keeps the form
  // symbolic, but both numericize identically under N.)
  #[test]
  fn symbolic_stays_unevaluated() {
    assert_eq!(
      interpret("InverseBetaRegularized[s, 2, 3]").unwrap(),
      "InverseBetaRegularized[s, 2, 3]"
    );
    assert_eq!(
      interpret("InverseBetaRegularized[s, a, b]").unwrap(),
      "InverseBetaRegularized[s, a, b]"
    );
  }

  // Boundary s values (numeric a, b): I_z(a,b) reaches 0 at z = 0 and 1 at z = 1.
  #[test]
  fn boundary_values() {
    assert_eq!(interpret("InverseBetaRegularized[0, 2, 3]").unwrap(), "0");
    assert_eq!(interpret("InverseBetaRegularized[1, 2, 3]").unwrap(), "1");
  }

  // I_z(1, 1) = z, so the inverse is the identity on a numeric s (returned
  // exactly), while a symbolic s stays unevaluated.
  #[test]
  fn a_b_one_identity() {
    assert_eq!(
      interpret("InverseBetaRegularized[1/2, 1, 1]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("InverseBetaRegularized[1/3, 1, 1]").unwrap(),
      "1/3"
    );
    assert_eq!(
      interpret("InverseBetaRegularized[0.7, 1, 1]").unwrap(),
      "0.7"
    );
    assert_eq!(
      interpret("InverseBetaRegularized[s, 1, 1]").unwrap(),
      "InverseBetaRegularized[s, 1, 1]"
    );
  }

  #[test]
  fn numeric_values() {
    let result: f64 = interpret("N[InverseBetaRegularized[1/2, 2, 3]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.38572756813238956).abs() < 1e-9, "got {result}");

    let result: f64 = interpret("N[InverseBetaRegularized[1/4, 2, 3]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.2430220837560763).abs() < 1e-9, "got {result}");

    let result: f64 = interpret("N[InverseBetaRegularized[3/10, 5, 2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.6396423096199797).abs() < 1e-9, "got {result}");

    // Direct machine-real input numericizes without an explicit N.
    let result: f64 = interpret("InverseBetaRegularized[0.8, 1, 2]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.552786404500042).abs() < 1e-9, "got {result}");
  }

  // Round-trip: InverseBetaRegularized inverts BetaRegularized.
  #[test]
  fn round_trip() {
    let z: f64 = interpret("N[InverseBetaRegularized[3/10, 5, 2]]")
      .unwrap()
      .parse()
      .unwrap();
    let s: f64 = interpret(&format!("N[BetaRegularized[{z}, 5, 2]]"))
      .unwrap()
      .parse()
      .unwrap();
    assert!((s - 0.3).abs() < 1e-9, "got {s}");
  }

  // Generalized 4-arg InverseBetaRegularized[z0, s, a, b] solves
  // I_z(a,b) - I_z0(a,b) == s. Exact input stays symbolic.
  #[test]
  fn four_arg_symbolic() {
    assert_eq!(
      interpret("InverseBetaRegularized[1, -1/2, 5/2, 3/2]").unwrap(),
      "InverseBetaRegularized[1, -1/2, 5/2, 3/2]"
    );
  }

  // For z0 == 1 the 4-arg form reduces to the 3-arg InverseBetaRegularized[1+s].
  #[test]
  fn four_arg_numeric_reduces_to_three_arg() {
    let four: f64 = interpret("N[InverseBetaRegularized[1, -1/2, 5/2, 3/2]]")
      .unwrap()
      .parse()
      .unwrap();
    let three: f64 = interpret("N[InverseBetaRegularized[1/2, 5/2, 3/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((four - 0.6475477201228692).abs() < 1e-9, "got {four}");
    assert!((four - three).abs() < 1e-9, "{four} vs {three}");
  }
}

mod beta_regularized {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("BetaRegularized[0, 2, 3]").unwrap(), "0");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("BetaRegularized[1, 2, 3]").unwrap(), "1");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("BetaRegularized[z, 2, 3]").unwrap(),
      "BetaRegularized[z, 2, 3]"
    );
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("BetaRegularized[0.5, 2.0, 3.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.6875).abs() < 1e-10);
  }

  #[test]
  fn numeric_uniform() {
    // BetaRegularized[x, 1, 1] = x
    let result: f64 = interpret("BetaRegularized[0.3, 1.0, 1.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.3).abs() < 1e-10);
  }

  #[test]
  fn numeric_half_half() {
    let result: f64 = interpret("BetaRegularized[0.3, 0.5, 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.3690101195655453).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_params() {
    let result: f64 = interpret("BetaRegularized[0.5, 10.0, 10.0]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.5).abs() < 1e-10);
  }

  #[test]
  fn n_of_integer_args() {
    let result: f64 = interpret("N[BetaRegularized[0.5, 2.0, 3.0]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.6875).abs() < 1e-10);
  }

  // BetaRegularized[z0, z1, a, b] = BetaRegularized[z1, a, b] -
  // BetaRegularized[z0, a, b], evaluated when all arguments are numbers.
  #[test]
  fn four_arg_exact() {
    assert_eq!(
      interpret("BetaRegularized[1/4, 3/4, 2, 3]").unwrap(),
      "11/16"
    );
    assert_eq!(interpret("BetaRegularized[0, 1, 2, 3]").unwrap(), "1");
  }

  #[test]
  fn four_arg_numeric() {
    let result: f64 = interpret("BetaRegularized[0.3, 0.7, 2, 3]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.5680000000000003).abs() < 1e-9, "got {result}");
  }

  #[test]
  fn four_arg_symbolic_stays_unevaluated() {
    assert_eq!(
      interpret("BetaRegularized[z0, z1, a, b]").unwrap(),
      "BetaRegularized[z0, z1, a, b]"
    );
  }
}

// Beta[z0, z1, a, b] is the generalized incomplete Beta = Beta[z1, a, b] -
// Beta[z0, a, b].
mod generalized_incomplete_beta {
  use super::*;

  #[test]
  fn four_arg_exact() {
    assert_eq!(interpret("Beta[1/4, 3/4, 2, 3]").unwrap(), "11/192");
    assert_eq!(interpret("Beta[0, 1, 2, 3]").unwrap(), "1/12");
  }

  #[test]
  fn four_arg_numeric() {
    let result: f64 =
      interpret("Beta[0.2, 0.8, 2, 3]").unwrap().parse().unwrap();
    assert!((result - 0.066).abs() < 1e-9, "got {result}");
  }

  #[test]
  fn four_arg_symbolic_stays_unevaluated() {
    assert_eq!(
      interpret("Beta[z0, z1, a, b]").unwrap(),
      "Beta[z0, z1, a, b]"
    );
  }
}

mod sinh_integral {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("SinhIntegral[0]").unwrap(), "0");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("SinhIntegral[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn at_neg_infinity() {
    assert_eq!(interpret("SinhIntegral[-Infinity]").unwrap(), "-Infinity");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("SinhIntegral[x]").unwrap(), "SinhIntegral[x]");
  }

  #[test]
  fn integer_unevaluated() {
    assert_eq!(interpret("SinhIntegral[1]").unwrap(), "SinhIntegral[1]");
  }

  #[test]
  fn numeric_one() {
    let result: f64 = interpret("SinhIntegral[1.0]").unwrap().parse().unwrap();
    assert!((result - 1.0572508753757286).abs() < 1e-10);
  }

  #[test]
  fn numeric_two() {
    let result: f64 = interpret("SinhIntegral[2.0]").unwrap().parse().unwrap();
    assert!((result - 2.501567433354976).abs() < 1e-10);
  }

  #[test]
  fn numeric_five() {
    let result: f64 = interpret("SinhIntegral[5.0]").unwrap().parse().unwrap();
    assert!((result - 20.093211825697225).abs() < 1e-8);
  }

  #[test]
  fn numeric_negative() {
    let result: f64 = interpret("SinhIntegral[-1.0]").unwrap().parse().unwrap();
    assert!((result - (-1.0572508753757286)).abs() < 1e-10);
  }

  #[test]
  fn n_of_integer() {
    let result: f64 = interpret("N[SinhIntegral[1]]").unwrap().parse().unwrap();
    assert!((result - 1.0572508753757286).abs() < 1e-10);
  }
}

mod cosh_integral {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("CoshIntegral[0]").unwrap(), "-Infinity");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("CoshIntegral[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("CoshIntegral[x]").unwrap(), "CoshIntegral[x]");
  }

  #[test]
  fn integer_unevaluated() {
    assert_eq!(interpret("CoshIntegral[1]").unwrap(), "CoshIntegral[1]");
  }

  #[test]
  fn numeric_one() {
    let result: f64 = interpret("CoshIntegral[1.0]").unwrap().parse().unwrap();
    assert!((result - 0.8378669409802083).abs() < 1e-10);
  }

  #[test]
  fn numeric_two() {
    let result: f64 = interpret("CoshIntegral[2.0]").unwrap().parse().unwrap();
    assert!((result - 2.452666922646914).abs() < 1e-10);
  }

  #[test]
  fn numeric_five() {
    let result: f64 = interpret("CoshIntegral[5.0]").unwrap().parse().unwrap();
    assert!((result - 20.092063530105946).abs() < 1e-8);
  }

  #[test]
  fn numeric_small() {
    let result: f64 = interpret("CoshIntegral[0.1]").unwrap().parse().unwrap();
    assert!((result - (-1.7228683861943335)).abs() < 1e-10);
  }

  #[test]
  fn n_of_integer() {
    let result: f64 = interpret("N[CoshIntegral[1]]").unwrap().parse().unwrap();
    assert!((result - 0.8378669409802083).abs() < 1e-10);
  }
}

mod barnes_g {
  use super::*;

  #[test]
  fn integer_values() {
    assert_eq!(
      interpret("Table[BarnesG[n], {n, 0, 10}]").unwrap(),
      "{0, 1, 1, 1, 2, 12, 288, 34560, 24883200, 125411328000, 5056584744960000}"
    );
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("BarnesG[0]").unwrap(), "0");
  }

  #[test]
  fn negative_integers() {
    assert_eq!(interpret("BarnesG[-1]").unwrap(), "0");
    assert_eq!(interpret("BarnesG[-2]").unwrap(), "0");
    assert_eq!(interpret("BarnesG[-10]").unwrap(), "0");
  }

  #[test]
  fn float_value() {
    let result: f64 = interpret("BarnesG[3.5]").unwrap().parse().unwrap();
    assert!((result - 1.2596482574951955).abs() < 1e-10);
  }

  #[test]
  fn float_half() {
    let result: f64 = interpret("BarnesG[0.5]").unwrap().parse().unwrap();
    assert!((result - 0.6032442812094462).abs() < 1e-10);
  }

  #[test]
  fn float_integer_point() {
    let result: f64 = interpret("BarnesG[5.0]").unwrap().parse().unwrap();
    assert!((result - 12.0).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("BarnesG[x]").unwrap(), "BarnesG[x]");
  }

  #[test]
  fn series_at_zero_order_2() {
    // wolframscript:
    //   SeriesData[x, 0, {1, EulerGamma + (-1 + Log[2*Pi])/2}, 1, 3, 1]
    // Woxi's Plus canonicalisation reorders the EulerGamma summand.
    assert_eq!(
      interpret("Series[BarnesG[x], {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, (-1 + Log[2*Pi])/2 + EulerGamma}, 1, 3, 1]"
    );
  }

  #[test]
  fn series_at_zero_order_1() {
    // Only the leading linear term: BarnesG[x] = x + O(x^2).
    assert_eq!(
      interpret("Series[BarnesG[x], {x, 0, 1}]").unwrap(),
      "SeriesData[x, 0, {1}, 1, 2, 1]"
    );
  }
}

mod powers_representations {
  use super::*;

  #[test]
  fn taxicab_number() {
    assert_eq!(
      interpret("PowersRepresentations[1729, 2, 3]").unwrap(),
      "{{1, 12}, {9, 10}}"
    );
  }

  #[test]
  fn sum_of_two_squares() {
    assert_eq!(
      interpret("PowersRepresentations[100, 2, 2]").unwrap(),
      "{{0, 10}, {6, 8}}"
    );
  }

  #[test]
  fn no_representation() {
    assert_eq!(interpret("PowersRepresentations[3, 2, 2]").unwrap(), "{}");
  }

  #[test]
  fn zero() {
    assert_eq!(
      interpret("PowersRepresentations[0, 2, 2]").unwrap(),
      "{{0, 0}}"
    );
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("PowersRepresentations[-5, 2, 2]").unwrap(), "{}");
  }

  #[test]
  fn single_term() {
    assert_eq!(
      interpret("PowersRepresentations[1, 1, 2]").unwrap(),
      "{{1}}"
    );
  }

  #[test]
  fn power_one_partitions() {
    assert_eq!(
      interpret("PowersRepresentations[10, 2, 1]").unwrap(),
      "{{0, 10}, {1, 9}, {2, 8}, {3, 7}, {4, 6}, {5, 5}}"
    );
  }

  #[test]
  fn sum_of_three_squares() {
    assert_eq!(
      interpret("PowersRepresentations[30, 3, 2]").unwrap(),
      "{{1, 2, 5}}"
    );
  }

  #[test]
  fn nine_as_sum_of_two_squares() {
    assert_eq!(
      interpret("PowersRepresentations[9, 2, 2]").unwrap(),
      "{{0, 3}}"
    );
  }

  #[test]
  fn fifty_as_sum_of_two_squares() {
    assert_eq!(
      interpret("PowersRepresentations[50, 2, 2]").unwrap(),
      "{{1, 7}, {5, 5}}"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("PowersRepresentations[x, 2, 2]").unwrap(),
      "PowersRepresentations[x, 2, 2]"
    );
  }

  #[test]
  fn ten_as_sum_of_three_squares() {
    assert_eq!(
      interpret("PowersRepresentations[10, 3, 2]").unwrap(),
      "{{0, 1, 3}}"
    );
  }
}

mod gamma_incomplete {
  use super::*;

  #[test]
  fn gamma_1_x() {
    assert_eq!(interpret("Gamma[1, x]").unwrap(), "E^(-x)");
  }

  #[test]
  fn gamma_0_x() {
    // Wolfram keeps Gamma[0, x] unevaluated (no ExpIntegralE rewrite).
    assert_eq!(interpret("Gamma[0, x]").unwrap(), "Gamma[0, x]");
  }

  #[test]
  fn gamma_2_x() {
    // Wolfram keeps Gamma[n, x] unevaluated for symbolic x and n >= 2.
    assert_eq!(interpret("Gamma[2, x]").unwrap(), "Gamma[2, x]");
  }

  #[test]
  fn gamma_1_arg_still_works() {
    assert_eq!(interpret("Gamma[1/2]").unwrap(), "Sqrt[Pi]");
    assert_eq!(interpret("Gamma[5]").unwrap(), "24");
  }

  // Gamma[a, 0] = Gamma[a] for Re[a] > 0, but diverges for a <= 0:
  // Gamma[0, 0] = Infinity, Gamma[a, 0] = ComplexInfinity for a < 0. With an
  // inexact zero second argument these previously stayed unevaluated (a = 0)
  // or returned a large garbage value (a < 0). Per wolframscript.
  #[test]
  fn gamma_at_real_zero() {
    assert_eq!(interpret("Gamma[0, 0.0]").unwrap(), "Infinity");
    assert_eq!(interpret("Gamma[-1, 0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Gamma[-2, 0.0]").unwrap(), "ComplexInfinity");
    // A non-integer negative order also diverges.
    assert_eq!(interpret("Gamma[-0.5, 0.0]").unwrap(), "ComplexInfinity");
    // Positive order stays finite (Gamma[a, 0.] = Gamma[a] as a real).
    assert_eq!(interpret("Gamma[1, 0.0]").unwrap(), "1.");
  }

  // Limits at infinity: Gamma[Infinity] = Infinity, Gamma[-Infinity] is
  // Indeterminate, Gamma[ComplexInfinity] = ComplexInfinity. Per wolframscript.
  #[test]
  fn gamma_infinite_limits() {
    assert_eq!(interpret("Gamma[Infinity]").unwrap(), "Infinity");
    assert_eq!(interpret("Gamma[-Infinity]").unwrap(), "Indeterminate");
    assert_eq!(
      interpret("Gamma[ComplexInfinity]").unwrap(),
      "ComplexInfinity"
    );
  }

  // An integer-valued real gives the exact factorial Gamma[n] = (n-1)!
  // rounded to a machine real, not the float-Lanczos approximation that
  // drifted (Gamma[5.0] used to give 23.999999999999996). Per wolframscript.
  #[test]
  fn gamma_integer_valued_real_is_exact() {
    assert_eq!(interpret("Gamma[2.0]").unwrap(), "1.");
    assert_eq!(interpret("Gamma[5.0]").unwrap(), "24.");
    assert_eq!(interpret("Gamma[1.0]").unwrap(), "1.");
  }

  #[test]
  fn gamma_1_numeric() {
    // Gamma[1, x] = E^(-x); with a machine-Real argument it evaluates numerically.
    assert_eq!(interpret("Gamma[1, 1.]").unwrap(), "0.36787944117144233");
  }

  // Real-argument upper incomplete gamma with z >= a+1 uses the continued
  // fraction branch. Regression: it previously diverged (the Lentz `c` seed
  // was 1e-30 instead of 1e30) and returned absurd magnitudes like 1e28.
  #[test]
  fn gamma_incomplete_real_cf_branch() {
    assert_eq!(interpret("Gamma[3., 5.]").unwrap(), "0.24930403896616227");
    assert_eq!(interpret("Gamma[4., 6.]").unwrap(), "0.9072232966598872");
  }

  // Three-argument generalized incomplete gamma stays symbolic in Wolfram
  // (it does not expand to Gamma[a, z0] - Gamma[a, z1]).
  #[test]
  fn gamma_three_arg_symbolic() {
    assert_eq!(interpret("Gamma[3, 2, 5]").unwrap(), "Gamma[3, 2, 5]");
    assert_eq!(interpret("Gamma[a, z0, z1]").unwrap(), "Gamma[a, z0, z1]");
    // Gamma[a, z0, Infinity] = Gamma[a, z0].
    assert_eq!(interpret("Gamma[3, 2, Infinity]").unwrap(), "10/E^2");
    assert_eq!(interpret("Gamma[3, 0, Infinity]").unwrap(), "2");
    // Gamma[a, z, z] = 0.
    assert_eq!(interpret("Gamma[2, 1, 1]").unwrap(), "0");
    // Inexact argument forces numeric evaluation.
    assert_eq!(
      interpret("Gamma[3, 2.0, 5.0]").unwrap(),
      "1.1040487933999648"
    );
  }

  // Derivatives of the incomplete gamma: d/dz Gamma[a, z] = -z^(a-1) E^(-z),
  // and the three-argument form differentiates as the difference.
  #[test]
  fn gamma_incomplete_derivatives() {
    assert_eq!(interpret("D[Gamma[a, x], x]").unwrap(), "-(x^(-1 + a)/E^x)");
    assert_eq!(interpret("D[Gamma[3, x], x]").unwrap(), "-(x^2/E^x)");
    assert_eq!(
      interpret("D[Gamma[a, x, b], x]").unwrap(),
      "-(x^(-1 + a)/E^x)"
    );
    assert_eq!(interpret("D[Gamma[a, 0, x], x]").unwrap(), "x^(-1 + a)/E^x");
  }
}

mod lambert_w {
  use super::*;

  #[test]
  fn alias_to_product_log() {
    assert_eq!(interpret("LambertW[k, z]").unwrap(), "ProductLog[k, z]");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("LambertW[0]").unwrap(), "0");
    assert_eq!(interpret("LambertW[E]").unwrap(), "1");
  }

  #[test]
  fn product_log_branch_0() {
    assert_eq!(interpret("ProductLog[0, E]").unwrap(), "1");
  }
}

mod subfactorial {
  use super::*;

  #[test]
  fn listable() {
    assert_eq!(
      interpret("Subfactorial[{0, 1, 2, 3}]").unwrap(),
      "{1, 0, 1, 2}"
    );
  }

  #[test]
  fn float_arg() {
    assert_eq!(interpret("Subfactorial[6.0]").unwrap(), "265.");
  }

  #[test]
  fn basic_values() {
    assert_eq!(interpret("Subfactorial[0]").unwrap(), "1");
    assert_eq!(interpret("Subfactorial[1]").unwrap(), "0");
    assert_eq!(interpret("Subfactorial[4]").unwrap(), "9");
  }
}

mod cases {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn polar_plot_1() {
    assert_case(
      r#"PolarPlot[Cos[5t], {t, 0, Pi}]"#,
      r#"Graphics[{{{{}, {}}, {}, {{{}, {}, Annotation[{Hue[0.67, 0.6, 0.6], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Line[{{1., 0.}, {0.9999879312413658, 0.0009635073215078271}, {0.9999517252998661, 0.0019269466617112766}, {0.9998913831786994, 0.0028902500421094527}, {0.9998069065498294, 0.0038533494898083235}, {0.9996982977539343, 0.004816177040323897}, {0.9995655598003358, 0.0057786647403850954}, {0.9994086963669071, 0.006740744650736219}, {0.9992277117999624, 0.007702348848938909}, {0.9990226111141239, 0.008663409432173494}, {0.9987933999921701, 0.009623858520039625}, {0.9985400847848632, 0.01058362825735611}, {0.9982626725107563, 0.011542650816959819}, {0.99796117085598, 0.01250085840250358}, {0.9976355881740098, 0.013458183251252961}, {0.9969122164775692, 0.015369913872266626}, {0.9965144475043882, 0.016324184312279107}, {0.9960926375859837, 0.017277301356577728}, {0.995176942322979, 0.01917980509733686}, {0.9946830823465402, 0.02012905684214597}, {0.9941652321604298, 0.02107688529350874}, {0.9930576192060382, 0.02296800303899587}, {0.992467887120469, 0.02391115785119044}, {0.9918542261897216, 0.024852620411632415}, {0.9905551864487123, 0.02673020056203551}, {0.9876707530796953, 0.030462107694530803}, {0.986890103222007, 0.03138985073683306}, {0.986085679010598, 0.032315367139179055}, {0.9844055972958322, 0.0341594544632042}, {0.9807611658414247, 0.03781799371719275}, {0.9797909816868382, 0.038726069544171224}, {0.9787972197613491, 0.03963139022399152}, {0.9767390733057385, 0.04143350393857968}, {0.9723411013390684, 0.0450017920380595}, {0.9711831018109552, 0.045885997160659275}, {0.9700017628043716, 0.04676692590942658}, {0.9675691977877631, 0.0485186961341135}, {0.9624254757486513, 0.05198008831408238}, {0.9510318486129903, 0.058720011546318764}, {0.9493752066431971, 0.05961325359605632}, {0.947691842828106, 0.06050144571711726}, {0.9442451693823171, 0.062262364314351566}, {0.9370336254956099, 0.06572014210632711}, {0.9351647889600131, 0.0665707862108788}, {0.9332696923488436, 0.06741575537211568}, {0.9294009659958807, 0.06908836145125985}, {0.9213511577167245, 0.07236210864012521}, {0.9192740115685997, 0.07316521529370505}, {0.9171711214185347, 0.07396203967071849}, {0.9128883830211716, 0.07553654359511969}, {0.9040170374181183, 0.07860690368866428}, {0.8850672661075282, 0.08441675857353154}, {0.8825868673361422, 0.0851105797033173}, {0.8800819147958538, 0.08579696454275454}, {0.874998674010113, 0.08714714897615577}, {0.8645411726252029, 0.08975526298476083}, {0.8618666611731369, 0.09038767739408697}, {0.8591682674002166, 0.09101211249793482}, {0.8537001832566653, 0.09223678056773131}, {0.8424813235669453, 0.09458748085824817}, {0.8396182431653036, 0.09515423081361016}, {0.8367320007785521, 0.09571248345048466}, {0.8308904043805331, 0.09680324555025831}, {0.818933426378416, 0.09888006222430595}, {0.8158877061947707, 0.09937709590563797}, {0.8128195916105092, 0.09986514092904299}, {0.8066165766854941, 0.10081402760093003}, {0.7939462253530064, 0.1026013507193782}, {0.7909383557256536, 0.10299706353568576}, {0.7879116642883494, 0.1033845390542258}, {0.7818021573608692, 0.10413459663861822}, {0.7787195138468895, 0.10449708881640217}, {0.7756183924700288, 0.10485116392433723}, {0.7693610656883013, 0.10553388712131957}, {0.7662050363246271, 0.10586244821740975}, {0.7630308811750114, 0.10618241826168438}, {0.7566285510891932, 0.1067964152628908}, {0.7436103423049661, 0.10791946831714509}, {0.7403118336868543, 0.10817812982435561}, {0.7369959271252842, 0.10842787018786149}, {0.7303122932295163, 0.10890042967483148}, {0.7269447536009399, 0.10912317086249743}, {0.7235601914343467, 0.10933683503847132}, {0.7167403800132294, 0.10973678078468917}, {0.7133053221693086, 0.10992298755611221}, {0.7098536246026635, 0.11009996772169287}, {0.7063853841528092, 0.11026768487635684}, {0.7029006981075752, 0.11042610301585452}, {0.6993996642001016, 0.11057518653845491}, {0.6958823806058209, 0.11071490024662389}, {0.6887994592518226, 0.11096607946048151}, {0.6852340200270799, 0.11107747660698167}, {0.6816527281793493, 0.11117936722392403}, {0.6780556840497864, 0.1112717181594058}, {0.674442988403453, 0.11135449667547268}, {0.6708147424262065, 0.11142767044968975}, {0.6671710477215796, 0.11149120757669777}, {0.6635120063076431, 0.11154507656975274}, {0.6598377206138605, 0.11158924636225076}, {0.656148293477928, 0.11162368630923658}, {0.6524438281426019, 0.11164836618889643}, {0.6487244282525156, 0.1116632562040352}, {0.6449901978509827, 0.11166832698353736}, {0.6412412413767896, 0.11166354958381235}, {0.6374776636609755, 0.1116488954902235}, {0.6336995699236011, 0.11162433661850114}, {0.6299070657705047, 0.1115898453161395}, {0.6261002571900487, 0.11154539436377733}, {0.6222792505498523, 0.11149095697656244}, {0.618444152593515, 0.11142650680549979}, {0.6145950704373279, 0.11135201793878324}, {0.6107321115669732, 0.11126746490311099}, {0.6068553838342148, 0.11117282266498454}, {0.6029649954535756, 0.11106806663199091}, {0.5990610549990063, 0.11095317265406873}, {0.5951436714005413, 0.11082811702475734}, {0.5912129539409462, 0.11069287648242937}, {0.5833119563128869, 0.11039174984365946}, {0.5793418964432802, 0.11022581945898846}, {0.575358943303479, 0.11004961558719054}, {0.5673548015287071, 0.10966630375585634}, {0.5634128313185823, 0.10946331805279999}, {0.5594588952305343, 0.10925038012280679}, {0.5515155489781681, 0.10879457556386571}, {0.547526351325687, 0.1085516739803842}, {0.5435256128126781, 0.10829875026278}, {0.5354899413864332, 0.10776277117625367}, {0.5314552232654555, 0.10747968424658783}, {0.5274093938621233, 0.10718651206307059}, {0.5192848338112734, 0.10656985351223232}, {0.5029071518478914, 0.10521489476659023}, {0.49878658465388076, 0.10485073790832626}, {0.49465577820240597, 0.10447639014025605}, {0.4863638883581716, 0.10369707726252439}, {0.46966210017003446, 0.1020156942406715}, {0.4358114768404064, 0.09815996347697427}, {0.43154042950868643, 0.097631645322931}, {0.4272609326437441, 0.09709300860540471}, {0.41867704509362086, 0.09598476300193537}, {0.40141288865382246, 0.09364429266668375}, {0.36652475019534025, 0.08846708629110818}, {0.36213224861375876, 0.08777341286584282}, {0.3577331368417933, 0.08706940669416657}, {0.3489155480325598, 0.08563040823482838}, {0.33120617394608265, 0.08262857688049798}, {0.2955168592310755, 0.07613083451555538}, {0.2906523279040651, 0.07519921328244016}, {0.28578223662097624, 0.07425555297977704}, {0.2760259771596316, 0.07233216774484413}, {0.25645337745383034, 0.06834154170743886}, {0.21710174512162084, 0.05978861390721111}, {0.21216639364276285, 0.05866627494295601}, {0.20722790419302664, 0.05753217941071495}, {0.19734211894992545, 0.05522881098808628}, {0.1775395773987713, 0.05048204571270641}, {0.13784465042773805, 0.04043436650843072}, {0.13287732432974958, 0.03912701929893889}, {0.1279092909757794, 0.037808356245578156}, {0.11797170933943948, 0.03513721417385355}, {0.09809473090500698, 0.02966062101951971}, {0.05836740605326911, 0.018178342884206937}, {-0.020711022339762446, -0.006829439196533276}, {-0.025298291685988027, -0.008369247922030023}, {-0.029882002478339564, -0.009917746255936367}, {-0.039038269078886384, -0.013040643995955399}, {-0.05730283647365413, -0.019388845686297264}, {-0.09361349130753989, -0.03248493570439868}, {-0.1651506394789174, -0.06019081618517847}, {-0.16956551793919167, -0.06198587786187123}, {-0.1739731005563567, -0.0637880859207122}, {-0.18276592689359822, -0.06741372174081918}, {-0.20025907987708133, -0.07474831502378314}, {-0.23485047503472448, -0.089738019811121}, {-0.30225844332282775, -0.1208925365645011}, {-0.3067303452436255, -0.12305455516596706}, {-0.31118960260538286, -0.12522269296036678}, {-0.3200696620204635, -0.12957698995220457}, {-0.33767233722420276, -0.13835529226911597}, {-0.3722193808952308, -0.15617136032213572}, {-0.4384594813314086, -0.19268105302767433}, {-0.44246471252034086, -0.19499523025091633}, {-0.4464533976798307, -0.19731264320610117}, {-0.4543806823758643, -0.20195679181425272}, {-0.4700318126786507, -0.2112796707826866}, {-0.5004960958259076, -0.2300418307095908}, {-0.5578880934771647, -0.26785264976359974}, {-0.5612493276901601, -0.27017810287355093}, {-0.5645914828195691, -0.2725035816629776}, {-0.5712182163613616, -0.2771542217357365}, {-0.5842390391576605, -0.2864514606003343}, {-0.6093319974492978, -0.30500747213872903}, {-0.655587901641662, -0.3417854598655952}, {-0.6582999384534592, -0.34406211994992614}, {-0.6609905291937008, -0.3463355849802373}, {-0.6663071259648942, -0.3508725213538255}, {-0.6766802977451082, -0.3599035575254463}, {-0.6963737331189638, -0.3777713678802428}, {-0.7314567959423646, -0.41254662744186454}, {-0.7333220891809369, -0.414527803071639}, {-0.7351673366051079, -0.4165033609451852}, {-0.7387975746035361, -0.420437294259587}, {-0.7458162211390293, -0.42823413105156377}, {-0.7588802878985309, -0.44352535307542346}, {-0.7604215862453266, -0.44540681178130526}, {-0.7619424313298304, -0.4472813457694063}, {-0.7649226810951509, -0.4510093163329861}, {-0.7706368887947668, -0.45837861169847144}, {-0.7810764070857159, -0.4727526789842048}, {-0.7822883576681542, -0.47451371167961504}, {-0.783479603218786, -0.4762665418155143}, {-0.7857999376266036, -0.47974728008281325}, {-0.7901917226251646, -0.48660687765072075}, {-0.7912377716393588, -0.4883000978156965}, {-0.7922630482737512, -0.4899844911895052}, {-0.7942512623022783, -0.49332648882593855}, {-0.7979782127532593, -0.49990117711853077}, {-0.7988579578205413, -0.5015216311503401}, {-0.7997169021117465, -0.5031326459361529}, {-0.8013723857020135, -0.5063260553573006}, {-0.8044337481774609, -0.5125962886168058}, {-0.8051470980517373, -0.514139122270813}, {-0.8058396575533385, -0.5156719174666236}, {-0.8071624221125019, -0.5187070971230742}, {-0.809558683865837, -0.524653759136393}, {-0.8101510671280211, -0.5262369480831007}, {-0.8107190627975706, -0.5278076390148975}, {-0.8117819375491887, -0.5309111603357968}, {-0.8122768417895323, -0.5324438083610521}, {-0.8127474087516501, -0.5339635936518888}, {-0.8136155902117779, -0.5369642152854819}, {-0.8140132364470117, -0.5384448721817265}, {-0.8143866088760877, -0.5399123074586207}, {-0.8147357254141498, -0.5413664323369933}, {-0.8150606047927084, -0.5428071584225462}, {-0.8153612665583467, -0.5442343977098557}, {-0.815637731071397, -0.5456480625863621}, {-0.8158900195045876, -0.547048065836346}, {-0.8161181538416608, -0.548434320644891}, {-0.8163221568759602, -0.5498067406018327}, {-0.8165020522089906, -0.5511652397056958}, {-0.8166578642489467, -0.5525097323676159}, {-0.8167896182092147, -0.553840133415248}, {-0.8168973401068422, -0.5551563580966613}, {-0.816981056760982, -0.5564583220842194}, {-0.8170407957913045, -0.5577459414784467}, {-0.8170765856163827, -0.559019132811879}, {-0.8170884554520476, -0.5602778130529016}, {-0.817076435309715, -0.5615218996095707}, {-0.8170405559946843, -0.5627513103334209}, {-0.8169808491044075, -0.5639659635232563}, {-0.8168973470267312, -0.5651657779289291}, {-0.8167900829381082, -0.5663506727550989}, {-0.8166590908017823, -0.5675205676649793}, {-0.8165044053659453, -0.5686753827840678}, {-0.8163260621618629, -0.5698150387038599}, {-0.8161240975019763, -0.5709394564855466}, {-0.815898548477973, -0.5720485576636963}, {-0.8156494529588306, -0.573142264249921}, {-0.8153768495888326, -0.5742204987365234}, {-0.8150807777855573, -0.5752831841001304}, {-0.814761277737837, -0.576330243805308}, {-0.8144183904036915, -0.5773616018081594}, {-0.8140521575082336, -0.578377182559907}, {-0.8136626215415467, -0.5793769110104546}, {-0.8132498257565348, -0.5803607126119358}, {-0.812813814166747, -0.5813285133222411}, {-0.8123546315441725, -0.5822802396085307}, {-0.8118723234170102, -0.583215818450726}, {-0.8108385165291937, -0.5850382443071649}, {-0.810287112585531, -0.5859249478762482}, {-0.8097127727666144, -0.5867952171177755}, {-0.8084954833446559, -0.5884861715334775}, {-0.8078526345156695, -0.5893067175020176}, {-0.807187051354678, -0.5901105507384419}, {-0.805787891686235, -0.5916678065574302}, {-0.8050544218321251, -0.5924210942812219}, {-0.8042984309476237, -0.5931573995619066}, {-0.8027191073847684, -0.5945787991749025}, {-0.8018958871579314, -0.5952637631001655}, {-0.8010503707984611, -0.59593148377462}, {-0.7992926824599669, -0.5972149407967516}, {-0.7984419157145998, -0.5977901884494505}, {-0.7975719198241876, -0.5983501924175072}, {-0.7957744389350216, -0.5994242694756381}, {-0.7948470544285221, -0.5999382437655361}, {-0.7939006417574693, -0.6004367767747453}, {-0.7919509387096291, -0.601387326289239}, {-0.7909477530353526, -0.6018392475967779}, {-0.789925748597401, -0.602275537232329}, {-0.7878254985481689, -0.6031010361231025}, {-0.7867473617833394, -0.6034901538500851}, {-0.7856506239434659, -0.6038634568526776}, {-0.7834015683506352, -0.6045624407494249}, {-0.782249363519732, -0.6048880338500652}, {-0.7810787834534966, -0.6051976366432443}, {-0.7798898858690692, -0.6054912064152032}, {-0.7786827289820263, -0.6057687009288909}, {-0.77745737150413, -0.6060300784258918}, {-0.7762138726410628, -0.6062752976283362}, {-0.7749522920901467, -0.6065043177407946}, {-0.7736726900380478, -0.6067170984521597}, {-0.7723751271584666, -0.6069135999375099}, {-0.7710596646098127, -0.6070937828599587}, {-0.7697263640328649, -0.6072576083724882}, {-0.768375287548419, -0.607405038119767}, {-0.7670064977549182, -0.607536034239952}, {-0.7656200577260704, -0.607650559366475}, {-0.7642160310084517, -0.6077485766298131}, {-0.7627944816190956, -0.6078300496592424}, {-0.7613554740430672, -0.6078949425845769}, {-0.7598990732310247, -0.6079432200378911}, {-0.7584253445967659, -0.6079748471552257}, {-0.756934354014762, -0.6079897895782767}, {-0.7554261678176759, -0.607988013456071}, {-0.7539008527938694, -0.6079694854466217}, {-0.7523584761848944, -0.6079341727185693}, {-0.7507991056829721, -0.6078820429528076}, {-0.7492228094284583, -0.6078130643440892}, {-0.7476296560072958, -0.6077272056026184}, {-0.7460197144484524, -0.6076244359556248}, {-0.7443930542213482, -0.6075047251489215}, {-0.7427497452332673, -0.6073680434484459}, {-0.7410898578267575, -0.6072143616417841}, {-0.7394134627770187, -0.6070436510396778}, {-0.7377206312892759, -0.6068558834775147}, {-0.7360114349961416, -0.6066510313168021}, {-0.7342859459549641, -0.6064290674466218}, {-0.7307863799655648, -0.6059336987806829}, {-0.7290124492316905, -0.6056602424138308}, {-0.7272225181730808, -0.6053695711981163}, {-0.7235949520535666, -0.6047364869488482}, {-0.7217574664973146, -0.6043940266208794}, {-0.7199042796201476, -0.6040342568578697}, {-0.7161511053352766, -0.6032627003676724}, {-0.7084597007759594, -0.6015109292938365}, {-0.7065374701869603, -0.6010390407147377}, {-0.7046007414810271, -0.6005503605576931}, {-0.7006840872126807, -0.5995225583639551}, {-0.6926800929651091, -0.5972647449797623}, {-0.6906439766939535, -0.5966580787519845}, {-0.6885939661557227, -0.5960344999361012}, {-0.6844525705441722, -0.5947365537898404}, {-0.6760064571154731, -0.5919371811532093}, {-0.6738613603271112, -0.5911948843319111}, {-0.6717029941137719, -0.5904355868958919}, {-0.6673467717678697, -0.588865956055118}, {-0.6584786127467287, -0.5855223409555702}, {-0.6401376456119715, -0.5780164625817864}, {-0.6377899446745803, -0.5770014106696596}, {-0.6354302814244275, -0.5759692827027262}, {-0.6306754044061225, -0.5738537982854665}, {-0.6210258175854236, -0.5694179431612776}, {-0.6011864745774383, -0.5597273582350661}, {-0.5595034826929736, -0.5370832716867154}, {-0.556586965410131, -0.5353910443736164}, {-0.5536596749960301, -0.5336790215010532}, {-0.5477732394583372, -0.530195677028992}, {-0.5358762148389302, -0.5229924507357823}, {-0.511611849104566, -0.5076457371914783}, {-0.4614146572627587, -0.4732483928344073}, {-0.4582119583262145, -0.4709372226343432}, {-0.45500229225038646, -0.4686073240763374}, {-0.44856254329682665, -0.46389152481339824}, {-0.43560477476134857, -0.45423723903940344}, {-0.4094033794008395, -0.4340490990245185}, {-0.35607539014779765, -0.3902536419949132}, {-0.3529353492535911, -0.38756287035824877}, {-0.3497926252135177, -0.38485736815803534}, {-0.3434995222743988, -0.379402394234565}, {-0.330885456049103, -0.36831815243164173}, {-0.3055679019116703, -0.34546559678132605}, {-0.25475015216594393, -0.2971357944025533}, {-0.25157282371098666, -0.29400364917325544}, {-0.24839592875692618, -0.2908587964514942}, {-0.24204382120310783, -0.28453125399748486}, {-0.2293489999141118, -0.27172684534848024}, {-0.20401811132795244, -0.24553731277208915}, {-0.15375834838040126, -0.19097401981291556}, {-0.1503798515548009, -0.18717837727228823}, {-0.14700535898113556, -0.18337075135378939}, {-0.14026884074026205, -0.1757199840803879}, {-0.12684883189885088, -0.1602794612342332}, {-0.1002459083310474, -0.1288674986332499}, {-0.04818322900078782, -0.064127062953561}, {-0.04498741377724458, -0.06000437868848389}, {-0.04179905227289829, -0.05587343820970122}, {-0.03544509362506279, -0.04758728598129177}, {-0.022831030300398403, -0.03092140155858744}, {0.0019997059911864004, 0.0027563048523342203}, {0.049902464448069526, 0.07126331284772118}, {0.05275899305555136, 0.07550753304524147}, {0.055605462720526064, 0.07975569917956818}, {0.06126790554169911, 0.08826335573731851}, {0.07246859667557719, 0.10532035701008197}, {0.09435597923435629, 0.1395720366590049}, {0.13594430692504095, 0.20839061159033045}, {0.1384419134832109, 0.21269632279121128}, {0.14092716950938716, 0.217001766449364}, {0.1458603815885052, 0.225611313200894}, {0.15557590038877372, 0.2428212725913436}, {0.17439017512712562, 0.27717439810399347}, {0.20945471157889, 0.34537049871485637}, {0.21138985525047677, 0.34931562022231605}, {0.21331278888800648, 0.3532567444197571}, {0.2171218881995972, 0.36112656001813737}, {0.2245920769043378, 0.376813375532362}, {0.23893331635213985, 0.40795110039218635}, {0.26516900374890495, 0.4690865776690434}, {0.26669867851461715, 0.472849357545431}, {0.26821527785484844, 0.4766046418287666}, {0.2712091750065408, 0.4840922926561492}, {0.27703926414163993, 0.498972914190668}, {0.28806504529141125, 0.5283314857423915}, {0.3075570642570502, 0.5852485651933511}, {0.30875420333969494, 0.5890119041299557}, {0.30993553655865164, 0.5927624543994836}, {0.31225077050102285, 0.6002246692019869}, {0.31669144208533695, 0.6149899363093765}, {0.32481363788784806, 0.6438551558111365}, {0.32575779489492107, 0.647398466123675}, {0.32668616484629354, 0.6509269494479047}, {0.328495571797316, 0.6579389386851651}, {0.3319252886662085, 0.6717795551794852}, {0.3380307684752967, 0.6987000756409343}, {0.33872353030296126, 0.701991482298914}, {0.33940069462334954, 0.7052661254037598}, {0.3407083010378396, 0.7117646525043662}, {0.3431371531662049, 0.7245554179204822}, {0.343705658591795, 0.727709460545967}, {0.3442587238332956, 0.7308458152914742}, {0.3453186245742099, 0.7370650083872005}, {0.3472541781899974, 0.749286188433582}, {0.34769982878060496, 0.7522955839533677}, {0.3481302372063183, 0.7552863995446112}, {0.3489454384673816, 0.7612118549686077}, {0.34933028830396695, 0.7641462781743363}, {0.3497000099743836, 0.7670616882056599}, {0.3503941895943816, 0.7728350415606164}, {0.35071870946280004, 0.7756927726920819}, {0.3510282249987816, 0.7785310662728413}, {0.3516023736002739, 0.7841489226447884}, {0.35186707343915985, 0.7869282778072318}, {0.3521169024882272, 0.7896877801693675}, {0.35257208836255316, 0.7951468176560859}, {0.35276428854259634, 0.7976669372856434}, {0.35294365597278105, 0.8001694135785917}, {0.35311022090322125, 0.8026541651929648}, {0.35326401403030705, 0.8051211112459513}, {0.3534050664951197, 0.8075701713165867}, {0.35353340988183024, 0.8100012654484289}, {0.353649076216083, 0.8124143141522212}, {0.3537520979633637, 0.8148092384085397}, {0.3538425080273532, 0.8171859596704291}, {0.35392033974826637, 0.8195443998660245}, {0.3539856269011752, 0.8218844814011578}, {0.35403840369431705, 0.8242061271619506}, {0.3540787047673891, 0.8265092605173932}, {0.35410656518982747, 0.8287938053219099}, {0.35412202045907193, 0.8310596859179101}, {0.35412510649881457, 0.8333068271383218}, {0.3541158596572366, 0.8355351543091153}, {0.3540943167052286, 0.8377445932518102}, {0.3540605148345975, 0.8399350702859673}, {0.35401449165625787, 0.8421065122316653}, {0.35395628519841044, 0.8442588464119646}, {0.3538859339047063, 0.8463920006553557}, {0.3538034766323959, 0.8485059032981914}, {0.35370895265046487, 0.8506004831871044}, {0.3536024016377555, 0.8526756696814098}, {0.353483863681075, 0.854731392655494}, {0.35335337927328914, 0.8567675825011865}, {0.3532109893114022, 0.8587841701301144}, {0.35305673509462354, 0.8607810869760466}, {0.3528906583224212, 0.8627582649972186}, {0.3525232058991296, 0.8666531350344006}, {0.35232191563055354, 0.8685706936099252}, {0.35210897356759197, 0.8704682464842618}, {0.3516483091311307, 0.8742030741270781}, {0.35140067526263474, 0.8760402197418512}, {0.35114156660566775, 0.8778571013524369}, {0.3505891061542321, 0.8814298202298533}, {0.3502958459217658, 0.883185532700366}, {0.34999129402063744, 0.8849207315783393}, {0.34934850246234644, 0.8883293450349318}, {0.3490103573570492, 0.8900026392431442}, {0.34866110968252473, 0.8916551791221542}, {0.3479294997613017, 0.8948977613144197}, {0.34754723498759293, 0.8964876877495763}, {0.34715406258624265, 0.8980566281033494}, {0.3463351937820477, 0.9011313250620131}, {0.3459095977030167, 0.9026369703453486}, {0.3454732946394831, 0.9041214069080686}, {0.3445687720435613, 0.907026437565806}, {0.34410065561399106, 0.9084469249578505}, {0.34362203840097366, 0.9098459902270726}, {0.3426335115667949, 0.9125796474110861}, {0.34053277654104236, 0.9177876524120168}, {0.33993566443329676, 0.919138628907879}, {0.3393267315211036, 0.9204639803529739}, {0.3380736839356833, 0.9230375693521548}, {0.33742971067600075, 0.9242856896102449}, {0.3367741994317204, 0.9255079502305029}, {0.33542885052260163, 0.9278746671411882}, {0.33473915767534246, 0.9290190128210258}, {0.3340382164712738, 0.9301372776465421}, {0.3326028831723212, 0.9322953527930804}, {0.3296004948534554, 0.9362962490533182}, {0.3288228940918695, 0.937230505982797}, {0.3280346437055231, 0.9381382803291084}, {0.3264265008111706, 0.9398741967067115}, {0.32560676261738897, 0.9407022486182373}, {0.3247766834195649, 0.9415036377110195}, {0.3230858146834081, 0.943026256760358}, {0.3222251823799494, 0.9437474035592505}, {0.3213545235343556, 0.9444417212277766}, {0.3195834445575689, 0.9457497125003343}, {0.31868318445645616, 0.9463633099680875}, {0.31777321786542384, 0.9469499260363708}, {0.3159244889703972, 0.9480420714207531}, {0.3149858893645791, 0.9485475316772048}, {0.314037908657151, 0.9490258724180898}, {0.3130806289988608, 0.9494770613379644}, {0.31211413285384065, 0.9499010670239022}, {0.3111385029955428, 0.9502978589569824}, {0.31015382250266665, 0.9506674075137436}, {0.30916017475507385, 0.9510096839676057}, {0.3081576434296918, 0.9513246604902609}, {0.30714631249640684, 0.95161231015303}, {0.3061262662139475, 0.9518726069281883}, {0.3050975891257562, 0.9521055256902574}, {0.3040603660558521, 0.9523110422172654}, {0.3030146821046822, 0.9524891331919738}, {0.3019606226449635, 0.9526397762030723}, {0.3008982733175166, 0.9527629497463408}, {0.29982772002708724, 0.9528586332257774}, {0.2987490489381606, 0.9529268069546954}, {0.2976623464707656, 0.9529674521567862}, {0.29656769929627025, 0.9529805509671492}, {0.29546519433316887, 0.9529660864332892}, {0.2943549187428596, 0.9529240425160809}, {0.2932369599254136, 0.9528544040906997}, {0.2921114055153382, 0.9527571569475196}, {0.29097834337732803, 0.952632287792978}, {0.289837861602012, 0.9524797842504077}, {0.2886900485016905, 0.9522996348608346}, {0.28753499260606524, 0.9520918290837429}, {0.28637278265796356, 0.9518563572978077}, {0.2852035076090524, 0.9515932108015916}, {0.2840272566155479, 0.951302381814212}, {0.28165418441657675, 0.9506376498489524}, {0.28047938955997065, 0.9502707914938326}, {0.27929821453763115, 0.9498772284639635}, {0.2769170652681493, 0.9490099771455198}, {0.27571726206862984, 0.9485362853860059}, {0.2745114207907097, 0.9480358820106226}, {0.2720819678196665, 0.9469549429218946}, {0.2708585284083355, 0.9463744106130318}, {0.26962939547423836, 0.9457671734977797}, {0.2671543951513122, 0.9444726011208121}, {0.2621398867838664, 0.9415632258950098}, {0.26087332829106596, 0.940769211225644}, {0.25960177177315163, 0.9399485466790766}, {0.2570440145965068, 0.9382273117790964}, {0.2518723787304798, 0.9344655737375973}, {0.25056829031434524, 0.9334587090070675}, {0.249259906179311, 0.9324253044288389}, {0.24663060348133692, 0.9302789470890608}, {0.24132433318243357, 0.9256685873061581}, {0.23998835127578777, 0.9244499454021239}, {0.23864878070813447, 0.9232049286957612}, {0.23595922808240352, 0.9206358696918139}, {0.23054096021931234, 0.9151823889314494}, {0.2195676585466987, 0.903020610074518}, {0.21818490564567436, 0.9013833883019917}, {0.21679998420559143, 0.8997202859083414}, {0.21402399063550243, 0.8963165925009567}, {0.20844988352625804, 0.8892003710683802}, {0.19723301551264663, 0.8737422529806739}, {0.1959206097864771, 0.8718340689135001}, {0.19460754260806404, 0.8699039757574627}, {0.1919797091238057, 0.8659782296194292}, {0.18671923830049114, 0.8578657049721203}, {0.17619495576819572, 0.8406067013514136}, {0.17488048272529616, 0.8383533234303298}, {0.17356647973606984, 0.8360787778578279}, {0.17094016344047, 0.8314663905172509}, {0.1656962398213631, 0.8219899121054909}, {0.15525872696108844, 0.8020424910524767}, {0.15396009626814136, 0.7994569509504588}, {0.1526630392389992, 0.7968511394796487}, {0.1500739172067393, 0.7915789471135182}, {0.14491752968913058, 0.7807940298233337}, {0.13470714940434692, 0.7582765070224478}, {0.11481327927006393, 0.7095737879904227}, {0.1134955851079394, 0.7061022585023131}, {0.11218219212717788, 0.7026095307046831}, {0.10956862322273547, 0.6955608822351728}, {0.10439652985832897, 0.6812136680194409}, {0.0942896456386599, 0.6515429927312351}, {0.09304999737479787, 0.6477447245875939}, {0.09181586474280501, 0.6439269473575169}, {0.08936443739991798, 0.6362333117294254}, {0.08453095016299896, 0.6206168751119694}, {0.07515724174459117, 0.5884931548041483}, {0.07401436755019512, 0.5843964661838533}, {0.07287812731458934, 0.580282127109586}, {0.07062581390430925, 0.5720009839375851}, {0.0662036874209647, 0.5552322750279637}, {0.05770372175071292, 0.5208971534184924}, {0.05667471367158213, 0.5165329561549226}, {0.05565334888357978, 0.5121531206255403}, {0.05363378547679867, 0.503347057188838}, {0.04968895071241433, 0.48555304707770647}, {0.04218906096038329, 0.44926741215428345}, {0.04134860939751159, 0.44497441023167317}, {0.04051559247891673, 0.4406696408647724}, {0.038872029372009, 0.4320252505249175}, {0.03567593172826567, 0.41460023772988375}, {0.02965668268339985, 0.3792309574256659}, {0.028939979750172238, 0.3747633757675418}, {0.028231332528138215, 0.3702858678882091}, {0.026838346287087513, 0.3613015446945352}, {0.02415057084219395, 0.34321898332159473}, {0.019175169445080336, 0.30662495151583263}, {0.01859136813643846, 0.3020128355160502}, {0.018016138057632823, 0.29739271010323964}, {0.016891505525011088, 0.2881289191977666}, {0.014746324152761328, 0.26951060713719044}, {0.014231868897784284, 0.26483784420708634}, {0.013726201403868198, 0.26015805408796633}, {0.012741329624957883, 0.25077788752364527}, {0.01087811357117491, 0.23193868709166596}, {0.01043463815492398, 0.2272131831585396}, {0.01000013880365987, 0.2224816476604177}, {0.00915815396336101, 0.2130009833495564}, {0.007582814196188791, 0.19397280078221307}, {0.007211724898478139, 0.18920256235709626}, {0.0068497712737447865, 0.18442729949586126}, {0.006153342238855693, 0.17486220702993327}, {0.004870869858825478, 0.15567730904515834}, {0.0045790955100677285, 0.15096486208688534}, {0.00429622672113516, 0.1462485608564812}, {0.003757259162968122, 0.13680487684374218}, {0.003501186000426444, 0.13207773495776054}, {0.0032540696101390726, 0.12734722058545267}, {0.0027867536705200335, 0.11787655726212057}, {0.0025665763153072617, 0.11313664998080207}, {0.0023554001197577776, 0.10839385354564528}, {0.0019600908880123605, 0.09890007748852533}, {0.0017759766201004606, 0.09414934019883488}, {0.0016009010477189248, 0.0893961984129014}, {0.0012778988023742432, 0.0798831867994328}, {0.0011299874606471485, 0.07512355985562265}, {0.0009911454763500117, 0.07036201417605395}, {0.0008613794386632595, 0.06559867134707915}, {0.0007406955057333858, 0.06083365300657606}, {0.0006290994044429699, 0.05606708084045825}, {0.0005265964301968786, 0.0512990765791885}, {0.0004331914467245696, 0.046529761994290826}, {0.00034888888589845876, 0.04175925889485674}, {0.0002736927475684863, 0.03698768912405319}, {0.00020760659941278245, 0.0322151745556328}, {0.00015063357680441828, 0.0274418370904354}, {0.0001027763826943845, 0.022667798652893777}, {0.00006403728751067613, 0.017893181187538996}, {0.00003441812907353131, 0.013118106655505772}, {0.00001392031252680627, 0.008342697031033887}, {2.54481028551191*^-6, 0.0035670742979674003}, {2.9216199952903676*^-7, -0.0012086395537348952}, {7.162474533437195*^-6, -0.005984322531501631}, {0.00002315542196253739, -0.010759852643650218}, {0.00004827024558502363, -0.015535107902880175}, {0.00008250575395029973, -0.02030996632976988}, {0.00012586032290346158, -0.025084305956275744}, {0.00017833189564594456, -0.029858004829231025}, {0.00023991798281232284, -0.034630941013841544}, {0.00031061566256319743, -0.03940299259717727}, {0.0003904215806943059, -0.0441740376916712}, {0.00047933195076181714, -0.048943954438618396}, {0.0005773425542236099, -0.05371262101166169}, {0.0006844487405968072, -0.058479915620287884}, {0.0008006454276314296, -0.0632457165133212}, {0.0010602878170040762, -0.07277235036552399}, {0.0012037211977948448, -0.07753294005042304}, {0.0013562204366126091, -0.08229154947816568}, {0.001688387106273763, -0.09180234161375334}, {0.001868038770408244, -0.09655428150150558}, {0.002056724759740663, -0.1013037554988765}, {0.0024611634541222226, -0.11079482093556159}, {0.002676896956720002, -0.11553617012031236}, {0.00290162638032996, -0.12027456891249406}, {0.0033780298745508435, -0.12974203171609322}, {0.004438289714165078, -0.14863623810301524}, {0.004750462234536797, -0.15375044856767828}, {0.005073106152034661, -0.15886003931793435}, {0.005749735404328359, -0.16906474844147368}, {0.007227817334272963, -0.18941199384957697}, {0.0076232229089918755, -0.19448496221297018}, {0.008028939118414366, -0.19955208886923786}, {0.008871211988154682, -0.20966820959674115}, {0.010678446952252719, -0.22982368007112325}, {0.011155669098761672, -0.23484567279345808}, {0.011643003959876655, -0.23986061442230544}, {0.012647901950897683, -0.24986874401522002}, {0.014777809986632407, -0.2697937997300578}, {0.019511191135178494, -0.309245808449756}, {0.020146620274843836, -0.31413710996484967}, {0.02079165714706667, -0.31901898982046584}, {0.0221104083494963, -0.32875390218695777}, {0.0248615698010312, -0.34810427930013954}, {0.03080966803090477, -0.3862950649997685}, {0.031594161214955487, -0.39101832863585356}, {0.0323876165343804, -0.39572988124764386}, {0.034001233616409214, -0.40511729405212815}, {0.03733400484036544, -0.4237454571600437}, {0.044410956784523004, -0.4603843422042415}, {0.04527131040762121, -0.46460219220905935}, {0.046138793478170806, -0.4688079454684813}, {0.04789497573059194, -0.47718272960347674}, {0.05149097856101316, -0.4937824034899513}, {0.05900742561045034, -0.5263584460535935}, {0.0599764916282972, -0.5303699787320563}, {0.06095195646483377, -0.5343677292559138}, {0.06292188768418323, -0.5423214760855732}, {0.06693637303636417, -0.5580591214913999}, {0.07525249084224193, -0.5888327195116982}, {0.0763179142494079, -0.5926117255393744}, {0.07738891854063877, -0.5963753671081394}, {0.07954745410751346, -0.6038561765217553}, {0.08392910639041878, -0.6186292634503368}, {0.09293827023994042, -0.6474006023040043}, {0.11183647853663459, -0.7016847154141514}, {0.11315441307800733, -0.7051981137395782}, {0.11447670667828136, -0.7086901560430383}, {0.1171340518336042, -0.715609768903494}, {0.12249752147342859, -0.7291883231108722}, {0.13340129902406045, -0.755281096525148}, {0.13477923686506205, -0.7584410825259219}, {0.13616022137029216, -0.7615781846267177}, {0.1389309922641555, -0.7677833819456622}, {0.1445053602656622, -0.7799153067751564}, {0.1557661427546523, -0.8030464065129775}, {0.15718255124661898, -0.8058300146606835}, {0.15860062555977397, -0.8085894105476337}, {0.16144141860985045, -0.814035261817237}, {0.16713910013715444, -0.8246331226378413}, {0.17857883290932489, -0.8446375413199627}, {0.18001120462180065, -0.8470250833025764}, {0.18144380905379073, -0.8493872951649312}, {0.18430935225481626, -0.8540354787473184}, {0.190039221976456, -0.863025185141674}, {0.201473564651916, -0.8797650821817895}, {0.20287267593035996, -0.8817044835090017}, {0.20427059778160273, -0.8836186061319613}, {0.20706252281629078, -0.8873708317161084}, {0.21262824228369506, -0.8945699655915944}, {0.21401538269968462, -0.8963058928902101}, {0.2154006315355514, -0.898016195871857}, {0.21816510258261104, -0.9013597718388838}, {0.2236674735253331, -0.9077377511791759}, {0.2250370169135166, -0.9092676255156575}, {0.22640396439260113, -0.9107715831602122}, {0.2291297192517071, -0.9137016181034994}, {0.23454620228356413, -0.9192493018317527}, {0.24521934664132733, -0.9290889151330326}, {0.24653679241054036, -0.9302006611255941}, {0.24785023519437407, -0.9312860672468151}, {0.2504647615076182, -0.9333777837296188}, {0.25564192890157356, -0.9372443482472951}, {0.2569249021721605, -0.9381448839144889}, {0.25820317377816404, -0.9390189495319027}, {0.26074526426066796, -0.9406876216993992}, {0.26576920580735036, -0.9437068350365082}, {0.267012136637784, -0.9443953096157035}, {0.2682496730739629, -0.9450572384553808}, {0.2707082186041657, -0.9463014372548145}, {0.27192905594376215, -0.946883698512632}, {0.2731441553881168, -0.94743939662673}, {0.27555679859835647, -0.9484710953825355}, {0.27675417173242106, -0.9489470941303351}, {0.2779454657154387, -0.9493965259460379}, {0.2791305956136737, -0.9498193920080826}, {0.2803094766532964, -0.9502156943444563}, {0.2814820242242841, -0.9505854358323085}, {0.28264815388431835, -0.9509286201975388}, {0.2849608225641156, -0.9515353367047806}, {0.28603027859461094, -0.9517819823489907}, {0.28709386380889124, -0.9520055439959078}, {0.28815151039929077, -0.952206028037268}, {0.2892031507022856, -0.9523834415048921}, {0.2902487172012177, -0.9525377920702515}, {0.2912881425290143, -0.952669088044014}, {0.2923213594709031, -0.9527773383755699}, {0.29334830096712355, -0.9528625526525404}, {0.2943689001156334, -0.9529247411002639}, {0.29538309017481085, -0.9529639145812662}, {0.29639080456615247, -0.9529800845947094}, {0.2973919768769654, -0.9529732632758214}, {0.29838654086305677, -0.9529434633953086}, {0.29937443045141615, -0.9528906983587454}, {0.30035557974289323, -0.9528149822059486}, {0.3013299230148724, -0.9527163296103296}, {0.30229739472393985, -0.95259475587823}, {0.3032579295085471, -0.952450276948235}, {0.30421146219166784, -0.9522829093904712}, {0.30515792778345047, -0.9520926704058832}, {0.3060972614838642, -0.9518795778254903}, {0.3070293986853403, -0.9516436501096277}, {0.30795427497540667, -0.9513849063471651}, {0.30887182613931746, -0.9511033662547078}, {0.3097819881626769, -0.9507990501757799}, {0.3106846972340559, -0.9504719790799864}, {0.31157988974760337, -0.9501221745621584}, {0.3124675023056515, -0.9497496588414787}, {0.31334747172131455, -0.9493544547605882}, {0.31421973502108097, -0.9489365857846744}, {0.3159408924612583, -0.9480329501156565}, {0.31678966174475753, -0.9475472334571899}, {0.3176304752036753, -0.9470389519710207}, {0.31928798740461567, -0.9459548013866026}, {0.32010456309958024, -0.9453789872645317}, {0.3209129368809302, -0.9447807182650122}, {0.32250483519136125, -0.943516932341986}, {0.32328823856456046, -0.9428514753025501}, {0.3240631977174059, -0.9421636831515556}, {0.32558754374370374, -0.9407212199900418}, {0.3285322834302517, -0.9375696975262747}, {0.329246468084107, -0.9367264746197743}, {0.32995173589212473, -0.9358611852877011}, {0.33133528961477116, -0.9340645531252442}, {0.3320134605253882, -0.9331332846772543}, {0.3326824845869757, -0.9321800985657948}, {0.33399286517239823, -0.9302081286580989}, {0.334634108898864, -0.9291894239941973}, {0.33526598018632403, -0.92814895992867}, {0.33650138297184845, -0.926002918336571}, {0.3388572883221847, -0.9214515672841791}, {0.3394692532582646, -0.9201583543046151}, {0.3400696762279984, -0.9188399612998277}, {0.3412356248664026, -0.9161278694654069}, {0.34180101589769246, -0.9147342897437808}, {0.34235459569467097, -0.9133157682080724}, {0.34342605706411927, -0.9104041465808828}, {0.34394380747318043, -0.908911171890473}, {0.3444494843277758, -0.9073935061830275}, {0.34542435994698883, -0.9042843610783035}, {0.34722642227113326, -0.8977726748252013}, {0.34764580457929817, -0.8960840252595034}, {0.3480526103845723, -0.8943712233695152}, {0.34882824987606165, -0.8908734464172595}, {0.34919696345160633, -0.8890886151323407}, {0.3495528603091588, -0.8872799190720955}, {0.35022596897203195, -0.883591228377142}, {0.35054306455368955, -0.8817113834662138}, {0.35084711097611915, -0.8798079732215781}, {0.3511380513666247, -0.8778810743397377}, {0.35141582935357885, -0.8759307642456038}, {0.35168038906940136, -0.8739571210894388}, {0.3519316751535188, -0.8719602237437742}, {0.3523942075369872, -0.8678969855667292}, {0.3526053456765739, -0.8658308060636482}, {0.35280299387070546, -0.8637416950213239}, {0.3529870993375286, -0.8616297348765043}, {0.353157609819531, -0.8594950087691822}, {0.3533144735863622, -0.8573376005393445}, {0.35345763943763, -0.8551575947236886}, {0.3535870567056785, -0.8529550765523203}, {0.35370267525834626, -0.8507301319454292}, {0.3538044455017002, -0.848482847509934}, {0.3538923183827517, -0.8462133105361095}, {0.35396624539215116, -0.8439216089941916}, {0.35402617856685903, -0.8416078315309518}, {0.35407207049279793, -0.8392720674662553}, {0.3541038743074835, -0.836914406789596}, {0.3541215437026304, -0.8345349401566019}, {0.3541250329267413, -0.8321337588855271}, {0.3541142967876698, -0.8297109549537129}, {0.3540892906551638, -0.8272666209940309}, {0.3540499704633875, -0.8248008502913065}, {0.35399629271341787, -0.8223137367787102}, {0.3539282144757233, -0.8198053750341404}, {0.35384569339261596, -0.8172758602765716}, {0.3537486876806837, -0.8147252883623888}, {0.3536371561332004, -0.8121537557816998}, {0.3535110581225102, -0.8095613596546211}, {0.353370353602392, -0.8069481977275464}, {0.35321500311040094, -0.8043143683693965}, {0.35304496777018407, -0.8016599705678396}, {0.3528602092937759, -0.7989851039254985}, {0.35266068998387023, -0.7962898686561369}, {0.35221722104109615, -0.7908386961240501}, {0.35198989221471944, -0.7882668778764572}, {0.3517495766673533, -0.7856776590767433}, {0.351229872732205, -0.7804473559002664}, {0.35095042915285696, -0.777806440677218}, {0.35065788847236273, -0.7751484632038619}, {0.35003341050425835, -0.7697816646547071}, {0.34970142172662005, -0.7670730162389384}, {0.34935623287062734, -0.7643476508872014}, {0.3486261570860648, -0.7588471194114439}, {0.34700655474792114, -0.7476493271993883}, {0.3465683035949435, -0.7448094141157108}, {0.34611666897832166, -0.7419534950270243}, {0.3451731667095636, -0.7361940020857574}, {0.3431246784810352, -0.7244869621600468}, {0.3383777625139335, -0.70034151191227}, {0.33772315331579544, -0.6972566069914403}, {0.3370548864464448, -0.6941571938178517}, {0.3356773283586423, -0.6879152300762964}, {0.3327577833786117, -0.6752614840482796}, {0.32625874849895403, -0.649296849769851}, {0.3253843421001163, -0.6459916140109329}, {0.32449613096033414, -0.6426734583940944}, {0.32267827542853306, -0.6359987958291377}, {0.3188767268173204, -0.6224989396483548}, {0.3106098747939013, -0.5949203369742881}, {0.3095360034856283, -0.591489812268369}, {0.3084488499363478, -0.588048470424241}, {0.30623470922074214, -0.5811337363675548}, {0.3016472119695314, -0.5671788866314635}, {0.2918367077900936, -0.5387903990160915}, {0.26969097842385875, -0.4802827888515886}, {0.2681960317157131, -0.47655682977461444}, {0.2666881430606366, -0.4728233570532241}, {0.26363361484236947, -0.4653342944061195}, {0.25737010634728325, -0.45027067323513625}, {0.24423009878765575, -0.4198253247878567}, {0.21554311778372306, -0.3578544545239761}, {0.2134836113213651, -0.3536078708260588}, {0.21140985151201405, -0.3493564955144243}, {0.20721974606224694, -0.3408399235838226}, {0.19867044765952158, -0.32375537057519455}, {0.18090608960738236, -0.2894116436716607}, {0.14280760658558764, -0.22027353886590417}, {0.1403166955619593, -0.2159422261829284}, {0.13781323589897204, -0.2116105401057374}, {0.1327689253901623, -0.20294659520473374}, {0.12253255643639108, -0.18562023915391918}, {0.10148386056170007, -0.15100413752890965}, {0.05721072215921225, -0.08215983075118463}, {0.054544447516570824, -0.07816998800588418}, {0.05186926536463003, -0.07418355130206385}, {0.046492444362713374, -0.06622132003376098}, {0.03563485875754308, -0.05034238240811204}, {0.013519309202947341, -0.018790073002706256}, {-0.032185013604196205, 0.04330776214079015}, {-0.03510149460372335, 0.04713714427881056}, {-0.038024570412273695, 0.050959844354761495}, {-0.0438901933328966, 0.058584806898436405}, {-0.055697113953441284, 0.0737502586897249}, {-0.07959569866168784, 0.10372175004809812}, {-0.12838404513450455, 0.16205938055628877}, {-0.13173088265243615, 0.1659274877938192}, {-0.1350823404287218, 0.16978424323229585}, {-0.14179866952551212, 0.1774632565870516}, {-0.15528183196428255, 0.1926802284636727}, {-0.18242469566036573, 0.22252589644689352}, {-0.23720864253358181, 0.27967934766116964}, {-0.24064633313662373, 0.2831320836966883}, {-0.2440849165417696, 0.286570208828176}, {-0.2509642783537697, 0.2934022594695508}, {-0.26472838655229797, 0.30688702979893473}, {-0.29225081599035063, 0.33311953282128604}, {-0.34705171902565873, 0.3824876304020765}, {-0.3502317392131393, 0.3852361391428722}, {-0.3534091028935087, 0.3879696358380743}, {-0.35975545268508463, 0.39339136480409004}, {-0.37241178315485657, 0.40405219754751176}, {-0.3975560685708278, 0.4246312837726453}, {-0.44698777580077453, 0.46272999651102265}, {-0.45003208008849377, 0.46497248724520457}, {-0.45307045697824744, 0.46719841112113125}, {-0.45912902126617405, 0.47160040200016184}, {-0.4711705444489794, 0.48020389461165675}, {-0.49492845397444096, 0.49660091837045656}, {-0.540963317507906, 0.5260989447399594}, {-0.5437129740097416, 0.5277616615324316}, {-0.5464538486519152, 0.5294075273771943}, {-0.5519088825330033, 0.5326486296825966}, {-0.5627095089688494, 0.538927811639429}, {-0.583852502924329, 0.5506705041930057}, {-0.5864506446127093, 0.5520615999188049}, {-0.5890385452643409, 0.5534356020775658}, {-0.5941832653041795, 0.5561322842074552}, {-0.6043458850419923, 0.5613200999382869}, {-0.6241440656630934, 0.5708719178518658}, {-0.6265677215591204, 0.5719885614681435}, {-0.62897972954326, 0.5730880089018457}, {-0.6337684583785935, 0.5752353086181353}, {-0.6432023781766346, 0.5793235115529021}, {-0.6455304596927899, 0.5803025660136434}, {-0.647846213294034, 0.5812644251520902}, {-0.6524404018613134, 0.5831365681454509}, {-0.6614771759559412, 0.5866746578691647}, {-0.6637043037468408, 0.5875162508485149}, {-0.6659184412325487, 0.5883406838224831}, {-0.6703074196391717, 0.589938097573484}, {-0.6789259276506546, 0.592927340805213}, {-0.6810468683908016, 0.5936318745200349}, {-0.6831541755047031, 0.5943193176725543}, {-0.6873275731646583, 0.5956429770596411}, {-0.6955073106301248, 0.5980857309620013}, {-0.6976865525945177, 0.598701220367418}, {-0.6998490131954985, 0.5992967243843508}, {-0.7041232012842497, 0.6004278551304351}, {-0.7062347354587645, 0.6009635230915101}, {-0.708329101651331, 0.6014792881248842}, {-0.7124659488222675, 0.6024511997285654}, {-0.7145082404043319, 0.6029073932362764}, {-0.716532985219686, 0.603343777687817}, {-0.7205294612888256, 0.6041572210819875}, {-0.7225010071887625, 0.604534332615042}, {-0.7244546356219963, 0.6048917402692832}, {-0.7283077750819565, 0.6055475568309305}, {-0.7302071049206655, 0.605846023925947}, {-0.732088154924211, 0.606124903513673}, {-0.7339508358307804, 0.6063842264223563}, {-0.7357950589181939, 0.6066240241706047}, {-0.7376207360076266, 0.6068443289654435}, {-0.7394277794673215, 0.6070451737003556}, {-0.7412161022162745, 0.6072265919532889}, {-0.7429856177279022, 0.6073886179846402}, {-0.7447362400336988, 0.6075312867352176}, {-0.7464678837268668, 0.6076546338241736}, {-0.7481804639659279, 0.6077586955469129}, {-0.7498738964783236, 0.607843508872979}, {-0.7515480975639888, 0.607909111443913}, {-0.7532029840989055, 0.6079555415710874}, {-0.7548384735386453, 0.6079828382335171}, {-0.7564544839218855, 0.6079910410756448}, {-0.7580509338739037, 0.6079801904051015}, {-0.7596277426100596, 0.6079503271904433}, {-0.7611848299392542, 0.607901493058864}, {-0.7627221162673643, 0.607833730293882}, {-0.764239522600662, 0.6077470818330063}, {-0.7657369705492137, 0.607641591265375}, {-0.7672143823302547, 0.6075173028293712}, {-0.7686716807715458, 0.6073742614102171}, {-0.7701087893147107, 0.6072125125375412}, {-0.7715256320185486, 0.6070321023829233}, {-0.772922133562327, 0.6068330777574178}, {-0.774298219249057, 0.606615486109051}, {-0.7756538150087409, 0.6063793755202949}, {-0.7769888474016029, 0.606124794705522}, {-0.7795969314980964, 0.6055604203994591}, {-0.7808698395020478, 0.6052507274731538}, {-0.7821218967461242, 0.6049227654455459}, {-0.7833530329893401, 0.6045765861514797}, {-0.7845631786398507, 0.6042122420419311}, {-0.7857522647580256, 0.6038297861813015}, {-0.7869202230595023, 0.603429272244687}, {-0.7891924863694174, 0.6025742878808381}, {-0.7902966581126333, 0.6021199278324044}, {-0.791379435514652, 0.6016477304599761}, {-0.7934805481161171, 0.6006500510844754}, {-0.7944987554117299, 0.6001246842338464}, {-0.7954953125642397, 0.5995817103583245}, {-0.7974232281111283, 0.5984431782945873}, {-0.7982930380639102, 0.5978879925965912}, {-0.7991437795463272, 0.597317675291073}, {-0.8007878639198938, 0.5961318456262673}, {-0.8015811115416083, 0.5955164342316346}, {-0.8023551001579048, 0.5948860931544627}, {-0.8038451156872674, 0.5935808285880743}, {-0.804561051596346, 0.5929060094750865}, {-0.8052575464957175, 0.5922164694275865}, {-0.8065920371907769, 0.5907934398906586}, {-0.8072299463068794, 0.5900600581181408}, {-0.8078482410576561, 0.5893121708403971}, {-0.8090258201160972, 0.5877730997128093}, {-0.8095850221254183, 0.5869820268480113}, {-0.8101244451761455, 0.5861766704434441}, {-0.8111437958928451, 0.5845233333932355}, {-0.8116236456964103, 0.5836754669266202}, {-0.812083560819849, 0.5828135452735559}, {-0.8129434374619328, 0.581047769070493}, {-0.8133433256066205, 0.5801440318177794}, {-0.8137231323264112, 0.5792264739683588}, {-0.8140828223496189, 0.5782951551260701}, {-0.8144223609734434, 0.5773501352724445}, {-0.8147417140654822, 0.5763914747642508}, {-0.8150408480652219, 0.5754192343310296}, {-0.8153197299855155, 0.5744334750726183}, {-0.815578327414039, 0.5734342584566606}, {-0.8158166085147324, 0.5724216463161073}, {-0.8160345420292218, 0.5713957008467058}, {-0.8162320972782231, 0.570356484604481}, {-0.8164092441629297, 0.5693040605032015}, {-0.8165659531663799, 0.568238491811836}, {-0.816702195354808, 0.5671598421520009}, {-0.8168179423789763, 0.5660681754953971}, {-0.8169131664754891, 0.564963556161235}, {-0.8169878404680893, 0.5638460488136482}, {-0.817041937768936, 0.5627157184590996}, {-0.8170754323798636, 0.5615726304437781}, {-0.817088298893623, 0.5604168504509807}, {-0.8170805124951049, 0.5592484444984865}, {-0.817052048962544, 0.5580674789359255}, {-0.8170028846687043, 0.5568740204421304}, {-0.8169329965820467, 0.5556681360224821}, {-0.8168423622678777, 0.5544498930062469}, {-0.8167309598894802, 0.553219359043903}, {-0.8165987682092243, 0.5519766021044543}, {-0.8164457665896597, 0.550721690472742}, {-0.8162719349945912, 0.5494546927467411}, {-0.8160772539901321, 0.5481756778348477}, {-0.8158617047457426, 0.5468847149531633}, {-0.8156252690352456, 0.5455818736227633}, {-0.8153679292378269, 0.5442672236669596}, {-0.8150896683390129, 0.5429408352085544}, {-0.8144703182167654, 0.5402531247560703}, {-0.8141291980046472, 0.5388919444802146}, {-0.8137670947155913, 0.5375193091326521}, {-0.8129798836435548, 0.5347399598203032}, {-0.8125182274345186, 0.5332152016827568}, {-0.8120318677142505, 0.5316773368507435}, {-0.810984980358774, 0.5285626576832633}, {-0.8104244261551294, 0.5269860295681682}, {-0.8098391153056368, 0.5253966671908584}, {-0.8085941798938588, 0.5221801160922425}, {-0.8079345355769704, 0.5205531164800326}, {-0.8072500951070285, 0.5189137608152106}, {-0.8058067956053466, 0.5155983633592581}, {-0.802622359659358, 0.5088235169377693}, {-0.8017641774374373, 0.5071003594621472}, {-0.8008811611785048, 0.5053656185833604}, {-0.7990406239595625, 0.5018617789926956}, {-0.7950615609447795, 0.4947194319273492}, {-0.7859124492689284, 0.4799184141249423}, {-0.7846573031118715, 0.4780218468900421}, {-0.783377412952394, 0.4761152986109177}, {-0.7807434535014024, 0.4722726685855323}, {-0.7751792368787771, 0.4644721551215269}, {-0.7628696935078761, 0.4484333378900669}, {-0.7335704851605077, 0.41479282256025507}, {-0.7315716138337616, 0.412668126957455}, {-0.729549707539723, 0.4105370556252015}, {-0.725436944864926, 0.40625619291132026}, {-0.7169367474057733, 0.3976224414874623}, {-0.6988473554471688, 0.3800897203091571}, {-0.6584006731345752, 0.344146969947351}, {-0.6556876935816681, 0.341868969676075}, {-0.6529533008460925, 0.33958786872001007}, {-0.6474205302547968, 0.335016773581571}, {-0.6361008717610885, 0.3258418688797204}, {-0.6124602985890435, 0.3073840087365765}, {-0.5613057107949403, 0.2702172244710096}, {-0.5581691203328316, 0.26804657037009577}, {-0.5550159748235522, 0.2658760172499802}, {-0.5486603000425831, 0.26153553346036057}, {-0.5357534140557719, 0.2528592842732571}, {-0.5091739148819929, 0.23554336128233755}, {-0.45309006340385743, 0.20119717516954552}, {-0.44946104374631574, 0.19906859872537847}, {-0.4458179637203894, 0.19694259778258733}, {-0.43848996635649773, 0.19269861862466298}, {-0.423669054127711, 0.18424481208601098}, {-0.39338738515794475, 0.16749009591029915}, {-0.33042901207114866, 0.13471921304255396}, {-0.3260541925224027, 0.13253933527363537}, {-0.32166633067702677, 0.13036513539533026}, {-0.31285198604744097, 0.12603410394691486}, {-0.2950723830745079, 0.11744383335925951}, {-0.2589384970053112, 0.10056865480832233}, {-0.18461007181631592, 0.06817929162011853}, {-0.17988295409642527, 0.06622048573074026}, {-0.17514706968180188, 0.0642698173588077}, {-0.16564956333265782, 0.060393168676323675}, {-0.1465555588643968, 0.05274052356174348}, {-0.10800354188280027, 0.03785260731064001}, {-0.029703425007111185, 0.00985723284509578}, {-0.025095056571181898, 0.008300818653908085}, {-0.020483096802489935, 0.006753189099036077}, {-0.011248891328625379, 0.0036844516647245398}, {0.00725723815989307, -0.0023457867465191133}, {0.04439284787125561, -0.013968492089231753}, {0.049043957458506905, -0.015379553566204237}, {0.05369668958161232, -0.016781210844936188}, {0.06300652492919752, -0.01955617366846293}, {0.08164019900817387, -0.024991744007564643}, {0.11893564035620895, -0.03539826300957181}, {0.12359776828626508, -0.0366549266332101}, {0.12825952303405322, -0.03790168293761176}, {0.13758141182778097, -0.04036536409001945}, {0.1562151963968718, -0.045172660420807635}, {0.1934146900195483, -0.05430146486414066}, {0.2673165143089894, -0.0705781217285296}, {0.27181433782818004, -0.0714883173826489}, {0.27630795032138633, -0.07238837274208527}, {0.28528207218591645, -0.07415801696095982}, {0.3031746174000357, -0.07757514506672952}, {0.3387106989250155, -0.08391878013977787}, {0.3431270896201174, -0.08466559907651504}, {0.3475373994936292, -0.08540214701378981}, {0.35633931309542805, -0.08684441282722946}, {0.37386507728136975, -0.08960552971545696}, {0.4085785650473555, -0.09463372220896563}, {0.4128838258695706, -0.09521592908440604}, {0.4171811678193897, -0.095787848281229}, {0.42575164124656634, -0.09690083497777258}, {0.4427925795512824, -0.09900350563195774}, {0.4470313215131469, -0.09950351203390655}, {0.4512612417860233, -0.0999932648790545}, {0.45969416960148574, -0.10094203529903038}, {0.4764492524706679, -0.10271683731722868}, {0.48061429767336084, -0.10313501445499608}, {0.4847696314382237, -0.10354300017482952}, {0.49305072402249145, -0.10432843668566093}, {0.5094915382438435, -0.10577746944988588}, {0.513575819188803, -0.10611441166462052}, {0.5176495136841369, -0.10644125231823598}, {0.5257647105526391, -0.10706468202333348}, {0.5418633210787434, -0.10819093044500594}, {0.5461988644193353, -0.10846878302550018}, {0.5505209250272132, -0.10873488457438525}, {0.5591240566407782, -0.10923191999850636}, {0.5634048579358972, -0.10946289803706069}, {0.5676716370865038, -0.10968221336785385}, {0.576162593996135, -0.11008595059440172}, {0.5803865053140924, -0.11027042127271848}, {0.5845958616146409, -0.1104433268054137}, {0.5887905307359446, -0.11060469301544887}, {0.5929703809479118, -0.11075454629568658}, {0.5971352809567536, -0.110892913607313}, {0.6012850999095368, -0.11101982247823668}, {0.6054197073987212, -0.11113530100146304}, {0.6095389734666746, -0.1112393778334453}, {0.6136427686101938, -0.11133208219241217}, {0.6177309637849827, -0.11141344385666963}, {0.6218034304101309, -0.11148349316288163}, {0.6258600403725788, -0.11154226100432528}, {0.6299006660315638, -0.11158977882912453}, {0.6339251802230413, -0.11162607863845729}, {0.637933456264106, -0.1116511929847422}, {0.6419253679573808, -0.11166515496980076}, {0.6459007895954071, -0.11166799824299614}, {0.6498595959650034, -0.11165975699934917}, {0.6538016623516091, -0.11164046597763125}, {0.6577268645436272, -0.11161016045843475}, {0.6616350788367259, -0.11156887626221891}, {0.6655261820381406, -0.11151664974733493}, {0.6694000514709577, -0.1114535178080276}, {0.6732565649783644, -0.11137951787241374}, {0.6770956009279061, -0.11129468790043856}, {0.6809170382157076, -0.11119906638180978}, {0.6847207562706781, -0.11109269233390949}, {0.6885066350587102, -0.11097560529968284}, {0.6922745550868454, -0.1108478453455053}, {0.6960243974074356, -0.11070945305902859}, {0.7034693758866987, -0.11040093643307622}, {0.7071642769137247, -0.11023089585557899}, {0.710840629978096, -0.11005039046527647}, {0.7181372281508583, -0.1096581583978976}, {0.7217572426538628, -0.10944651956405392}, {0.7253582479914578, -0.10922459159923559}, {0.7325027763319835, -0.10875004948945703}, {0.7360460733841524, -0.10849752719482865}, {0.7395699093772543, -0.10823489946508467}, {0.7465587528306467, -0.10767951682213671}, {0.750023539120633, -0.10738685768950615}, {0.7534684220186401, -0.10708428467793758}, {0.760298042014026, -0.10644959389160956}, {0.7636825628447678, -0.10611757574816044}, {0.7670467477577931, -0.10577584298326226}, {0.7737136841796604, -0.10506343805189099}, {0.7867988794338228, -0.10352432790896188}, {0.7898031685265657, -0.10314364022175257}, {0.79278902974359, -0.10275485711998658}, {0.7987051392393423, -0.10195318254803777}, {0.810312629468694, -0.10025466393456836}, {0.8131672150773607, -0.0998104745335612}, {0.8160027273402741, -0.09935855417315097}, {0.8216162188296463, -0.09843170906010083}, {0.8326109079210817, -0.09648733863683363}, {0.835310741414803, -0.09598263664233699}, {0.8379908892161403, -0.09547058894087851}, {0.8432918316628127, -0.09442465498689911}, {0.8536542616607797, -0.09224683988388298}, {0.8734054240090705, -0.08755892639522991}, {0.8757816934619322, -0.08694280844006581}, {0.8781371565954565, -0.08632017333596817}, {0.8827854033601218, -0.08505556874474667}, {0.8918293872280488, -0.0824505564809809}, {0.8940374104952832, -0.08178383386041135}, {0.8962241213433302, -0.08111103511440357}, {0.9005333637830116, -0.07974743467670516}, {0.9088934706501267, -0.07694981397553532}, {0.9109293294894174, -0.07623607251485331}, {0.9129434072895543, -0.07551671219071232}, {0.9169059967903566, -0.07406136813627044}, {0.9245673843080742, -0.07108583154036938}, {0.9388232879351648, -0.06488871151635217}, {0.9406983195825123, -0.06399924032084234}, {0.942545179157959, -0.06310397030499564}, {0.9461541272893745, -0.06129637731486708}, {0.9530312104799415, -0.05761540035643333}, {0.9546791297777617, -0.056681957316610465}, {0.9562983928306398, -0.055743409911031265}, {0.9594507268177732, -0.05385135444153531}, {0.9654091182593241, -0.05000988223591117}, {0.9668262726285926, -0.04903808375294194}, {0.9682143498734019, -0.0480618922799079}, {0.9709030815687181, -0.046096690562890494}, {0.9759295703253095, -0.04211754564910643}, {0.9771128172172342, -0.04111313522469106}, {0.9782666304216785, -0.040105057736047896}, {0.9804857967754007, -0.038078268327043745}, {0.9845692335269307, -0.03398472919791393}, {0.9855159480619144, -0.03295355831401011}, {0.9864329375297514, -0.031919458428614454}, {0.9881776150860492, -0.029842843758375752}, {0.9890052426641501, -0.028800515387724178}, {0.989803024156684, -0.02775563083431077}, {0.991308939234596, -0.02565856749464528}, {0.992017020584304, -0.02460657617944271}, {0.9926951513790904, -0.023552403613876397}, {0.9939614682476058, -0.02143789094756139}, {0.9945496103921214, -0.02037773921951434}, {0.9951077141249578, -0.019315782976765398}, {0.9961337299500572, -0.017186834752193687}, {0.9966016064462798, -0.01612003188886401}, {0.9970393733400948, -0.015051802738043376}, {0.9974470154431848, -0.013982242052953125}, {0.9978245186121918, -0.012911444655594559}, {0.9981718697492555, -0.011839505431848713}, {0.9984890568025117, -0.010766519326573264}, {0.998776068766549, -0.009692581338698488}, {0.9990328956828287, -0.00861778651631859}, {0.9992595286400627, -0.007542229951780787}, {0.9994559597745529, -0.006466006776774146}, {0.999622182270489, -0.0053892121574166466}, {0.9997581903602085, -0.004311941289338913}, {0.9998639793244157, -0.0032342893927665927}, {0.9999395454923615, -0.0021563517076034055}, {0.9999848862419821, -0.0010782234885132922}, {1., -1.2246467991473532*^-16}}]}, "Charting`Private`Tag$24499#1"]}}, {}, {{{}, {}, {}, {}}, {}}}, {}}, {DisplayFunction -> Identity, PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.02], Scaled[0.02]}}, PlotRangeClipping -> True, ImagePadding -> All, PlotInteractivity :> $PlotInteractivity, PlotRange -> {{Automatic, Automatic}, {Automatic, Automatic}}, PlotRangePadding -> Automatic, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, Axes -> True, AxesOrigin -> {0, 0}, CoordinatesToolOptions -> {"DisplayFunction" -> ({Sqrt[#1[[1]]^2 + #1[[2]]^2], Mod[ArcTan[#1[[1]], #1[[2]]], 2*Pi]} & ), "CopiedValueFunction" -> ({Sqrt[#1[[1]]^2 + #1[[2]]^2], Mod[ArcTan[#1[[1]], #1[[2]]], 2*Pi]} & )}, DisplayFunction :> Identity, FrameTicks -> {{Automatic, Automatic}, {Automatic, Automatic}}, GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]], Method -> {"AxisPadding" -> Scaled[0.02], "DefaultBoundaryStyle" -> Automatic, "DefaultGraphicsInteraction" -> {"Version" -> 1.2, "TrackMousePosition" -> {True, False}, "Effects" -> {"Highlight" -> {"ratio" -> 2}, "HighlightPoint" -> {"ratio" -> 2}, "Droplines" -> {"freeformCursorMode" -> True, "placement" -> {"x" -> "All", "y" -> "None"}}}}, "DefaultMeshStyle" -> AbsolutePointSize[6], "DefaultPlotStyle" -> {Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Directive[RGBColor[0.95, 0.627, 0.1425], AbsoluteThickness[2]], Directive[RGBColor[0.455, 0.7, 0.21], AbsoluteThickness[2]], Directive[RGBColor[0.922526, 0.385626, 0.209179], AbsoluteThickness[2]], Directive[RGBColor[0.578, 0.51, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.772079, 0.431554, 0.102387], AbsoluteThickness[2]], Directive[RGBColor[0.4, 0.64, 1.], AbsoluteThickness[2]], Directive[RGBColor[1., 0.75, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.8, 0.4, 0.76], AbsoluteThickness[2]], Directive[RGBColor[0.637, 0.65, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.915, 0.3325, 0.2125], AbsoluteThickness[2]], Directive[RGBColor[0.40082222609352647, 0.5220066643438841, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.9728288904374106, 0.621644452187053, 0.07336199581899142], AbsoluteThickness[2]], Directive[RGBColor[0.736782672705901, 0.358, 0.5030266573755369], AbsoluteThickness[2]], Directive[RGBColor[0.28026441037696703, 0.715, 0.4292089322474965], AbsoluteThickness[2]]}, "DomainPadding" -> Scaled[0.02], "RangePadding" -> Scaled[0.05]}, PlotRange -> {{-0.8170884554520476, 1.}, {-0.9529800845947094, 0.9529805509671492}}, PlotRangeClipping -> True, PlotRangePadding -> {Scaled[0.02], Scaled[0.02]}, Ticks -> {Automatic, Automatic}}]"#,
    );
  }
  #[test]
  fn polar_plot_2() {
    assert_case(
      r#"PolarPlot[Cos[5t], {t, 0, Pi}]; PolarPlot[Abs[Cos[5t]], {t, 0, Pi}]"#,
      r#"Graphics[{{{{}, {}}, {}, {{{}, {}, Annotation[{Hue[0.67, 0.6, 0.6], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Line[{{1., 0.}, {0.9999879312413658, 0.0009635073215078271}, {0.9999517252998661, 0.0019269466617112766}, {0.9998913831786994, 0.0028902500421094527}, {0.9998069065498294, 0.0038533494898083235}, {0.9996982977539343, 0.004816177040323897}, {0.9995655598003358, 0.0057786647403850954}, {0.9994086963669071, 0.006740744650736219}, {0.9992277117999624, 0.007702348848938909}, {0.9990226111141239, 0.008663409432173494}, {0.9987933999921701, 0.009623858520039625}, {0.9985400847848632, 0.01058362825735611}, {0.9982626725107563, 0.011542650816959819}, {0.99796117085598, 0.01250085840250358}, {0.9976355881740098, 0.013458183251252961}, {0.9969122164775692, 0.015369913872266626}, {0.9965144475043882, 0.016324184312279107}, {0.9960926375859837, 0.017277301356577728}, {0.995176942322979, 0.01917980509733686}, {0.9946830823465402, 0.02012905684214597}, {0.9941652321604298, 0.02107688529350874}, {0.9930576192060382, 0.02296800303899587}, {0.992467887120469, 0.02391115785119044}, {0.9918542261897216, 0.024852620411632415}, {0.9905551864487123, 0.02673020056203551}, {0.9876707530796953, 0.030462107694530803}, {0.986890103222007, 0.03138985073683306}, {0.986085679010598, 0.032315367139179055}, {0.9844055972958322, 0.0341594544632042}, {0.9807611658414247, 0.03781799371719275}, {0.9797909816868382, 0.038726069544171224}, {0.9787972197613491, 0.03963139022399152}, {0.9767390733057385, 0.04143350393857968}, {0.9723411013390684, 0.0450017920380595}, {0.9711831018109552, 0.045885997160659275}, {0.9700017628043716, 0.04676692590942658}, {0.9675691977877631, 0.0485186961341135}, {0.9624254757486513, 0.05198008831408238}, {0.9510318486129903, 0.058720011546318764}, {0.9493752066431971, 0.05961325359605632}, {0.947691842828106, 0.06050144571711726}, {0.9442451693823171, 0.062262364314351566}, {0.9370336254956099, 0.06572014210632711}, {0.9351647889600131, 0.0665707862108788}, {0.9332696923488436, 0.06741575537211568}, {0.9294009659958807, 0.06908836145125985}, {0.9213511577167245, 0.07236210864012521}, {0.9192740115685997, 0.07316521529370505}, {0.9171711214185347, 0.07396203967071849}, {0.9128883830211716, 0.07553654359511969}, {0.9040170374181183, 0.07860690368866428}, {0.8850672661075282, 0.08441675857353154}, {0.8825868673361422, 0.0851105797033173}, {0.8800819147958538, 0.08579696454275454}, {0.874998674010113, 0.08714714897615577}, {0.8645411726252029, 0.08975526298476083}, {0.8618666611731369, 0.09038767739408697}, {0.8591682674002166, 0.09101211249793482}, {0.8537001832566653, 0.09223678056773131}, {0.8424813235669453, 0.09458748085824817}, {0.8396182431653036, 0.09515423081361016}, {0.8367320007785521, 0.09571248345048466}, {0.8308904043805331, 0.09680324555025831}, {0.818933426378416, 0.09888006222430595}, {0.8158877061947707, 0.09937709590563797}, {0.8128195916105092, 0.09986514092904299}, {0.8066165766854941, 0.10081402760093003}, {0.7939462253530064, 0.1026013507193782}, {0.7909383557256536, 0.10299706353568576}, {0.7879116642883494, 0.1033845390542258}, {0.7818021573608692, 0.10413459663861822}, {0.7787195138468895, 0.10449708881640217}, {0.7756183924700288, 0.10485116392433723}, {0.7693610656883013, 0.10553388712131957}, {0.7662050363246271, 0.10586244821740975}, {0.7630308811750114, 0.10618241826168438}, {0.7566285510891932, 0.1067964152628908}, {0.7436103423049661, 0.10791946831714509}, {0.7403118336868543, 0.10817812982435561}, {0.7369959271252842, 0.10842787018786149}, {0.7303122932295163, 0.10890042967483148}, {0.7269447536009399, 0.10912317086249743}, {0.7235601914343467, 0.10933683503847132}, {0.7167403800132294, 0.10973678078468917}, {0.7133053221693086, 0.10992298755611221}, {0.7098536246026635, 0.11009996772169287}, {0.7063853841528092, 0.11026768487635684}, {0.7029006981075752, 0.11042610301585452}, {0.6993996642001016, 0.11057518653845491}, {0.6958823806058209, 0.11071490024662389}, {0.6887994592518226, 0.11096607946048151}, {0.6852340200270799, 0.11107747660698167}, {0.6816527281793493, 0.11117936722392403}, {0.6780556840497864, 0.1112717181594058}, {0.674442988403453, 0.11135449667547268}, {0.6708147424262065, 0.11142767044968975}, {0.6671710477215796, 0.11149120757669777}, {0.6635120063076431, 0.11154507656975274}, {0.6598377206138605, 0.11158924636225076}, {0.656148293477928, 0.11162368630923658}, {0.6524438281426019, 0.11164836618889643}, {0.6487244282525156, 0.1116632562040352}, {0.6449901978509827, 0.11166832698353736}, {0.6412412413767896, 0.11166354958381235}, {0.6374776636609755, 0.1116488954902235}, {0.6336995699236011, 0.11162433661850114}, {0.6299070657705047, 0.1115898453161395}, {0.6261002571900487, 0.11154539436377733}, {0.6222792505498523, 0.11149095697656244}, {0.618444152593515, 0.11142650680549979}, {0.6145950704373279, 0.11135201793878324}, {0.6107321115669732, 0.11126746490311099}, {0.6068553838342148, 0.11117282266498454}, {0.6029649954535756, 0.11106806663199091}, {0.5990610549990063, 0.11095317265406873}, {0.5951436714005413, 0.11082811702475734}, {0.5912129539409462, 0.11069287648242937}, {0.5833119563128869, 0.11039174984365946}, {0.5793418964432802, 0.11022581945898846}, {0.575358943303479, 0.11004961558719054}, {0.5673548015287071, 0.10966630375585634}, {0.5634128313185823, 0.10946331805279999}, {0.5594588952305343, 0.10925038012280679}, {0.5515155489781681, 0.10879457556386571}, {0.547526351325687, 0.1085516739803842}, {0.5435256128126781, 0.10829875026278}, {0.5354899413864332, 0.10776277117625367}, {0.5314552232654555, 0.10747968424658783}, {0.5274093938621233, 0.10718651206307059}, {0.5192848338112734, 0.10656985351223232}, {0.5029071518478914, 0.10521489476659023}, {0.49878658465388076, 0.10485073790832626}, {0.49465577820240597, 0.10447639014025605}, {0.4863638883581716, 0.10369707726252439}, {0.46966210017003446, 0.1020156942406715}, {0.4358114768404064, 0.09815996347697427}, {0.43154042950868643, 0.097631645322931}, {0.4272609326437441, 0.09709300860540471}, {0.41867704509362086, 0.09598476300193537}, {0.40141288865382246, 0.09364429266668375}, {0.36652475019534025, 0.08846708629110818}, {0.36213224861375876, 0.08777341286584282}, {0.3577331368417933, 0.08706940669416657}, {0.3489155480325598, 0.08563040823482838}, {0.33120617394608265, 0.08262857688049798}, {0.2955168592310755, 0.07613083451555538}, {0.2906523279040651, 0.07519921328244016}, {0.28578223662097624, 0.07425555297977704}, {0.2760259771596316, 0.07233216774484413}, {0.25645337745383034, 0.06834154170743886}, {0.21710174512162084, 0.05978861390721111}, {0.13784465042773805, 0.04043436650843072}, {0.13287732432974958, 0.03912701929893889}, {0.1279092909757794, 0.037808356245578156}, {0.11797170933943948, 0.03513721417385355}, {0.09809473090500698, 0.02966062101951971}, {0.05836740605326911, 0.018178342884206937}, {0.05340693092104925, 0.016694208477823137}, {0.04844816671410838, 0.015199354199107942}, {0.03853637189091668, 0.012177655830674799}, {0.0335836413359283, 0.010650898110658894}, {0.028633221818564064, 0.009113593254898002}, {0.018739914386252072, 0.006007521298337176}, {0.013797325319112125, 0.004438845230895413}, {0.00885764497508879, 0.0028598040914029716}, {0.003921022358358174, 0.0012704448419552084}, {0.0010123937038791161, 0.00032918497999473944}, {0.005942454566433059, 0.0019390372632358812}, {0.010869011771151274, 0.003559063324424544}, {0.020711022339762446, 0.006829439196533276}, {0.025298291685988027, 0.008369247922030023}, {0.029882002478339564, 0.009917746255936367}, {0.039038269078886384, 0.013040643995955399}, {0.05730283647365413, 0.019388845686297264}, {0.09361349130753989, 0.03248493570439868}, {0.1651506394789174, 0.06019081618517847}, {0.16956551793919167, 0.06198587786187123}, {0.1739731005563567, 0.0637880859207122}, {0.18276592689359822, 0.06741372174081918}, {0.20025907987708133, 0.07474831502378314}, {0.23485047503472448, 0.089738019811121}, {0.30225844332282775, 0.1208925365645011}, {0.3067303452436255, 0.12305455516596706}, {0.31118960260538286, 0.12522269296036678}, {0.3200696620204635, 0.12957698995220457}, {0.33767233722420276, 0.13835529226911597}, {0.3722193808952308, 0.15617136032213572}, {0.4384594813314086, 0.19268105302767433}, {0.44246471252034086, 0.19499523025091633}, {0.4464533976798307, 0.19731264320610117}, {0.4543806823758643, 0.20195679181425272}, {0.4700318126786507, 0.2112796707826866}, {0.5004960958259076, 0.2300418307095908}, {0.5578880934771647, 0.26785264976359974}, {0.5612493276901601, 0.27017810287355093}, {0.5645914828195691, 0.2725035816629776}, {0.5712182163613616, 0.2771542217357365}, {0.5842390391576605, 0.2864514606003343}, {0.6093319974492978, 0.30500747213872903}, {0.655587901641662, 0.3417854598655952}, {0.6582999384534592, 0.34406211994992614}, {0.6609905291937008, 0.3463355849802373}, {0.6663071259648942, 0.3508725213538255}, {0.6766802977451082, 0.3599035575254463}, {0.6963737331189638, 0.3777713678802428}, {0.7314567959423646, 0.41254662744186454}, {0.7333220891809369, 0.414527803071639}, {0.7351673366051079, 0.4165033609451852}, {0.7387975746035361, 0.420437294259587}, {0.7458162211390293, 0.42823413105156377}, {0.7588802878985309, 0.44352535307542346}, {0.7604215862453266, 0.44540681178130526}, {0.7619424313298304, 0.4472813457694063}, {0.7649226810951509, 0.4510093163329861}, {0.7706368887947668, 0.45837861169847144}, {0.7810764070857159, 0.4727526789842048}, {0.7822883576681542, 0.47451371167961504}, {0.783479603218786, 0.4762665418155143}, {0.7857999376266036, 0.47974728008281325}, {0.7901917226251646, 0.48660687765072075}, {0.7912377716393588, 0.4883000978156965}, {0.7922630482737512, 0.4899844911895052}, {0.7942512623022783, 0.49332648882593855}, {0.7979782127532593, 0.49990117711853077}, {0.7988579578205413, 0.5015216311503401}, {0.7997169021117465, 0.5031326459361529}, {0.8013723857020135, 0.5063260553573006}, {0.8044337481774609, 0.5125962886168058}, {0.8051470980517373, 0.514139122270813}, {0.8058396575533385, 0.5156719174666236}, {0.8071624221125019, 0.5187070971230742}, {0.809558683865837, 0.524653759136393}, {0.8101510671280211, 0.5262369480831007}, {0.8107190627975706, 0.5278076390148975}, {0.8117819375491887, 0.5309111603357968}, {0.8122768417895323, 0.5324438083610521}, {0.8127474087516501, 0.5339635936518888}, {0.8136155902117779, 0.5369642152854819}, {0.8140132364470117, 0.5384448721817265}, {0.8143866088760877, 0.5399123074586207}, {0.8147357254141498, 0.5413664323369933}, {0.8150606047927084, 0.5428071584225462}, {0.8153612665583467, 0.5442343977098557}, {0.815637731071397, 0.5456480625863621}, {0.8158900195045876, 0.547048065836346}, {0.8161181538416608, 0.548434320644891}, {0.8163221568759602, 0.5498067406018327}, {0.8165020522089906, 0.5511652397056958}, {0.8166578642489467, 0.5525097323676159}, {0.8167896182092147, 0.553840133415248}, {0.8168973401068422, 0.5551563580966613}, {0.816981056760982, 0.5564583220842194}, {0.8170407957913045, 0.5577459414784467}, {0.8170765856163827, 0.559019132811879}, {0.8170884554520476, 0.5602778130529016}, {0.817076435309715, 0.5615218996095707}, {0.8170405559946843, 0.5627513103334209}, {0.8169808491044075, 0.5639659635232563}, {0.8168973470267312, 0.5651657779289291}, {0.8167900829381082, 0.5663506727550989}, {0.8166590908017823, 0.5675205676649793}, {0.8165044053659453, 0.5686753827840678}, {0.8163260621618629, 0.5698150387038599}, {0.8161240975019763, 0.5709394564855466}, {0.815898548477973, 0.5720485576636963}, {0.8156494529588306, 0.573142264249921}, {0.8153768495888326, 0.5742204987365234}, {0.8150807777855573, 0.5752831841001304}, {0.814761277737837, 0.576330243805308}, {0.8144183904036915, 0.5773616018081594}, {0.8140521575082336, 0.578377182559907}, {0.8136626215415467, 0.5793769110104546}, {0.8132498257565348, 0.5803607126119358}, {0.812813814166747, 0.5813285133222411}, {0.8123546315441725, 0.5822802396085307}, {0.8118723234170102, 0.583215818450726}, {0.8108385165291937, 0.5850382443071649}, {0.810287112585531, 0.5859249478762482}, {0.8097127727666144, 0.5867952171177755}, {0.8084954833446559, 0.5884861715334775}, {0.8078526345156695, 0.5893067175020176}, {0.807187051354678, 0.5901105507384419}, {0.805787891686235, 0.5916678065574302}, {0.8050544218321251, 0.5924210942812219}, {0.8042984309476237, 0.5931573995619066}, {0.8027191073847684, 0.5945787991749025}, {0.8018958871579314, 0.5952637631001655}, {0.8010503707984611, 0.59593148377462}, {0.7992926824599669, 0.5972149407967516}, {0.7984419157145998, 0.5977901884494505}, {0.7975719198241876, 0.5983501924175072}, {0.7957744389350216, 0.5994242694756381}, {0.7948470544285221, 0.5999382437655361}, {0.7939006417574693, 0.6004367767747453}, {0.7919509387096291, 0.601387326289239}, {0.7909477530353526, 0.6018392475967779}, {0.789925748597401, 0.602275537232329}, {0.7878254985481689, 0.6031010361231025}, {0.7867473617833394, 0.6034901538500851}, {0.7856506239434659, 0.6038634568526776}, {0.7834015683506352, 0.6045624407494249}, {0.782249363519732, 0.6048880338500652}, {0.7810787834534966, 0.6051976366432443}, {0.7798898858690692, 0.6054912064152032}, {0.7786827289820263, 0.6057687009288909}, {0.77745737150413, 0.6060300784258918}, {0.7762138726410628, 0.6062752976283362}, {0.7749522920901467, 0.6065043177407946}, {0.7736726900380478, 0.6067170984521597}, {0.7723751271584666, 0.6069135999375099}, {0.7710596646098127, 0.6070937828599587}, {0.7697263640328649, 0.6072576083724882}, {0.768375287548419, 0.607405038119767}, {0.7670064977549182, 0.607536034239952}, {0.7656200577260704, 0.607650559366475}, {0.7642160310084517, 0.6077485766298131}, {0.7627944816190956, 0.6078300496592424}, {0.7613554740430672, 0.6078949425845769}, {0.7598990732310247, 0.6079432200378911}, {0.7584253445967659, 0.6079748471552257}, {0.756934354014762, 0.6079897895782767}, {0.7554261678176759, 0.607988013456071}, {0.7539008527938694, 0.6079694854466217}, {0.7523584761848944, 0.6079341727185693}, {0.7507991056829721, 0.6078820429528076}, {0.7492228094284583, 0.6078130643440892}, {0.7476296560072958, 0.6077272056026184}, {0.7460197144484524, 0.6076244359556248}, {0.7443930542213482, 0.6075047251489215}, {0.7427497452332673, 0.6073680434484459}, {0.7410898578267575, 0.6072143616417841}, {0.7394134627770187, 0.6070436510396778}, {0.7377206312892759, 0.6068558834775147}, {0.7360114349961416, 0.6066510313168021}, {0.7342859459549641, 0.6064290674466218}, {0.7307863799655648, 0.6059336987806829}, {0.7290124492316905, 0.6056602424138308}, {0.7272225181730808, 0.6053695711981163}, {0.7235949520535666, 0.6047364869488482}, {0.7217574664973146, 0.6043940266208794}, {0.7199042796201476, 0.6040342568578697}, {0.7161511053352766, 0.6032627003676724}, {0.7084597007759594, 0.6015109292938365}, {0.7065374701869603, 0.6010390407147377}, {0.7046007414810271, 0.6005503605576931}, {0.7006840872126807, 0.5995225583639551}, {0.6926800929651091, 0.5972647449797623}, {0.6906439766939535, 0.5966580787519845}, {0.6885939661557227, 0.5960344999361012}, {0.6844525705441722, 0.5947365537898404}, {0.6760064571154731, 0.5919371811532093}, {0.6738613603271112, 0.5911948843319111}, {0.6717029941137719, 0.5904355868958919}, {0.6673467717678697, 0.588865956055118}, {0.6584786127467287, 0.5855223409555702}, {0.6401376456119715, 0.5780164625817864}, {0.6377899446745803, 0.5770014106696596}, {0.6354302814244275, 0.5759692827027262}, {0.6306754044061225, 0.5738537982854665}, {0.6210258175854236, 0.5694179431612776}, {0.6011864745774383, 0.5597273582350661}, {0.5595034826929736, 0.5370832716867154}, {0.556586965410131, 0.5353910443736164}, {0.5536596749960301, 0.5336790215010532}, {0.5477732394583372, 0.530195677028992}, {0.5358762148389302, 0.5229924507357823}, {0.511611849104566, 0.5076457371914783}, {0.4614146572627587, 0.4732483928344073}, {0.4582119583262145, 0.4709372226343432}, {0.45500229225038646, 0.4686073240763374}, {0.44856254329682665, 0.46389152481339824}, {0.43560477476134857, 0.45423723903940344}, {0.4094033794008395, 0.4340490990245185}, {0.35607539014779765, 0.3902536419949132}, {0.3529353492535911, 0.38756287035824877}, {0.3497926252135177, 0.38485736815803534}, {0.3434995222743988, 0.379402394234565}, {0.330885456049103, 0.36831815243164173}, {0.3055679019116703, 0.34546559678132605}, {0.25475015216594393, 0.2971357944025533}, {0.15375834838040126, 0.19097401981291556}, {0.1503798515548009, 0.18717837727228823}, {0.14700535898113556, 0.18337075135378939}, {0.14026884074026205, 0.1757199840803879}, {0.12684883189885088, 0.1602794612342332}, {0.1002459083310474, 0.1288674986332499}, {0.09694491860619406, 0.12489344754363976}, {0.09364970849505981, 0.1209092135534338}, {0.08707705776314564, 0.1129106649047223}, {0.07400582620249543, 0.09679654836295445}, {0.04818322900078782, 0.064127062953561}, {0.04498741377724458, 0.06000437868848389}, {0.04179905227289829, 0.05587343820970122}, {0.03544509362506279, 0.04758728598129177}, {0.022831030300398403, 0.03092140155858744}, {0.01969764452159877, 0.026736169671051014}, {0.016572506741230487, 0.022543687104720177}, {0.010347363276734845, 0.01413748037178633}, {0.007247550423502579, 0.009924012393674556}, {0.004156371223811987, 0.005703806104969877}, {0.0010739208585428966, 0.0014769905505688157}, {0.0019997059911864004, 0.0027563048523342203}, {0.005064415147282172, 0.006995950318697797}, {0.008120112937913648, 0.011241815700097904}, {0.014204102286049227, 0.019751683824302303}, {0.02625987997935271, 0.03684029323913921}, {0.049902464448069526, 0.07126331284772118}, {0.05275899305555136, 0.07550753304524147}, {0.055605462720526064, 0.07975569917956818}, {0.06126790554169911, 0.08826335573731851}, {0.07246859667557719, 0.10532035701008197}, {0.09435597923435629, 0.1395720366590049}, {0.13594430692504095, 0.20839061159033045}, {0.1384419134832109, 0.21269632279121128}, {0.14092716950938716, 0.217001766449364}, {0.1458603815885052, 0.225611313200894}, {0.15557590038877372, 0.2428212725913436}, {0.17439017512712562, 0.27717439810399347}, {0.20945471157889, 0.34537049871485637}, {0.21138985525047677, 0.34931562022231605}, {0.21331278888800648, 0.3532567444197571}, {0.2171218881995972, 0.36112656001813737}, {0.2245920769043378, 0.376813375532362}, {0.23893331635213985, 0.40795110039218635}, {0.26516900374890495, 0.4690865776690434}, {0.26669867851461715, 0.472849357545431}, {0.26821527785484844, 0.4766046418287666}, {0.2712091750065408, 0.4840922926561492}, {0.27703926414163993, 0.498972914190668}, {0.28806504529141125, 0.5283314857423915}, {0.3075570642570502, 0.5852485651933511}, {0.30875420333969494, 0.5890119041299557}, {0.30993553655865164, 0.5927624543994836}, {0.31225077050102285, 0.6002246692019869}, {0.31669144208533695, 0.6149899363093765}, {0.32481363788784806, 0.6438551558111365}, {0.32575779489492107, 0.647398466123675}, {0.32668616484629354, 0.6509269494479047}, {0.328495571797316, 0.6579389386851651}, {0.3319252886662085, 0.6717795551794852}, {0.3380307684752967, 0.6987000756409343}, {0.33872353030296126, 0.701991482298914}, {0.33940069462334954, 0.7052661254037598}, {0.3407083010378396, 0.7117646525043662}, {0.3431371531662049, 0.7245554179204822}, {0.343705658591795, 0.727709460545967}, {0.3442587238332956, 0.7308458152914742}, {0.3453186245742099, 0.7370650083872005}, {0.3472541781899974, 0.749286188433582}, {0.34769982878060496, 0.7522955839533677}, {0.3481302372063183, 0.7552863995446112}, {0.3489454384673816, 0.7612118549686077}, {0.34933028830396695, 0.7641462781743363}, {0.3497000099743836, 0.7670616882056599}, {0.3503941895943816, 0.7728350415606164}, {0.35071870946280004, 0.7756927726920819}, {0.3510282249987816, 0.7785310662728413}, {0.3516023736002739, 0.7841489226447884}, {0.35186707343915985, 0.7869282778072318}, {0.3521169024882272, 0.7896877801693675}, {0.35257208836255316, 0.7951468176560859}, {0.35276428854259634, 0.7976669372856434}, {0.35294365597278105, 0.8001694135785917}, {0.35311022090322125, 0.8026541651929648}, {0.35326401403030705, 0.8051211112459513}, {0.3534050664951197, 0.8075701713165867}, {0.35353340988183024, 0.8100012654484289}, {0.353649076216083, 0.8124143141522212}, {0.3537520979633637, 0.8148092384085397}, {0.3538425080273532, 0.8171859596704291}, {0.35392033974826637, 0.8195443998660245}, {0.3539856269011752, 0.8218844814011578}, {0.35403840369431705, 0.8242061271619506}, {0.3540787047673891, 0.8265092605173932}, {0.35410656518982747, 0.8287938053219099}, {0.35412202045907193, 0.8310596859179101}, {0.35412510649881457, 0.8333068271383218}, {0.3541158596572366, 0.8355351543091153}, {0.3540943167052286, 0.8377445932518102}, {0.3540605148345975, 0.8399350702859673}, {0.35401449165625787, 0.8421065122316653}, {0.35395628519841044, 0.8442588464119646}, {0.3538859339047063, 0.8463920006553557}, {0.3538034766323959, 0.8485059032981914}, {0.35370895265046487, 0.8506004831871044}, {0.3536024016377555, 0.8526756696814098}, {0.353483863681075, 0.854731392655494}, {0.35335337927328914, 0.8567675825011865}, {0.3532109893114022, 0.8587841701301144}, {0.35305673509462354, 0.8607810869760466}, {0.3528906583224212, 0.8627582649972186}, {0.3525232058991296, 0.8666531350344006}, {0.35232191563055354, 0.8685706936099252}, {0.35210897356759197, 0.8704682464842618}, {0.3516483091311307, 0.8742030741270781}, {0.35140067526263474, 0.8760402197418512}, {0.35114156660566775, 0.8778571013524369}, {0.3505891061542321, 0.8814298202298533}, {0.3502958459217658, 0.883185532700366}, {0.34999129402063744, 0.8849207315783393}, {0.34934850246234644, 0.8883293450349318}, {0.3490103573570492, 0.8900026392431442}, {0.34866110968252473, 0.8916551791221542}, {0.3479294997613017, 0.8948977613144197}, {0.34754723498759293, 0.8964876877495763}, {0.34715406258624265, 0.8980566281033494}, {0.3463351937820477, 0.9011313250620131}, {0.3459095977030167, 0.9026369703453486}, {0.3454732946394831, 0.9041214069080686}, {0.3445687720435613, 0.907026437565806}, {0.34410065561399106, 0.9084469249578505}, {0.34362203840097366, 0.9098459902270726}, {0.3426335115667949, 0.9125796474110861}, {0.34053277654104236, 0.9177876524120168}, {0.33993566443329676, 0.919138628907879}, {0.3393267315211036, 0.9204639803529739}, {0.3380736839356833, 0.9230375693521548}, {0.33742971067600075, 0.9242856896102449}, {0.3367741994317204, 0.9255079502305029}, {0.33542885052260163, 0.9278746671411882}, {0.33473915767534246, 0.9290190128210258}, {0.3340382164712738, 0.9301372776465421}, {0.3326028831723212, 0.9322953527930804}, {0.3296004948534554, 0.9362962490533182}, {0.3288228940918695, 0.937230505982797}, {0.3280346437055231, 0.9381382803291084}, {0.3264265008111706, 0.9398741967067115}, {0.32560676261738897, 0.9407022486182373}, {0.3247766834195649, 0.9415036377110195}, {0.3230858146834081, 0.943026256760358}, {0.3222251823799494, 0.9437474035592505}, {0.3213545235343556, 0.9444417212277766}, {0.3195834445575689, 0.9457497125003343}, {0.31868318445645616, 0.9463633099680875}, {0.31777321786542384, 0.9469499260363708}, {0.3159244889703972, 0.9480420714207531}, {0.3149858893645791, 0.9485475316772048}, {0.314037908657151, 0.9490258724180898}, {0.3130806289988608, 0.9494770613379644}, {0.31211413285384065, 0.9499010670239022}, {0.3111385029955428, 0.9502978589569824}, {0.31015382250266665, 0.9506674075137436}, {0.30916017475507385, 0.9510096839676057}, {0.3081576434296918, 0.9513246604902609}, {0.30714631249640684, 0.95161231015303}, {0.3061262662139475, 0.9518726069281883}, {0.3050975891257562, 0.9521055256902574}, {0.3040603660558521, 0.9523110422172654}, {0.3030146821046822, 0.9524891331919738}, {0.3019606226449635, 0.9526397762030723}, {0.3008982733175166, 0.9527629497463408}, {0.29982772002708724, 0.9528586332257774}, {0.2987490489381606, 0.9529268069546954}, {0.2976623464707656, 0.9529674521567862}, {0.29656769929627025, 0.9529805509671492}, {0.29546519433316887, 0.9529660864332892}, {0.2943549187428596, 0.9529240425160809}, {0.2932369599254136, 0.9528544040906997}, {0.2921114055153382, 0.9527571569475196}, {0.29097834337732803, 0.952632287792978}, {0.289837861602012, 0.9524797842504077}, {0.2886900485016905, 0.9522996348608346}, {0.28753499260606524, 0.9520918290837429}, {0.28637278265796356, 0.9518563572978077}, {0.2852035076090524, 0.9515932108015916}, {0.2840272566155479, 0.951302381814212}, {0.28165418441657675, 0.9506376498489524}, {0.28047938955997065, 0.9502707914938326}, {0.27929821453763115, 0.9498772284639635}, {0.2769170652681493, 0.9490099771455198}, {0.27571726206862984, 0.9485362853860059}, {0.2745114207907097, 0.9480358820106226}, {0.2720819678196665, 0.9469549429218946}, {0.2708585284083355, 0.9463744106130318}, {0.26962939547423836, 0.9457671734977797}, {0.2671543951513122, 0.9444726011208121}, {0.2621398867838664, 0.9415632258950098}, {0.26087332829106596, 0.940769211225644}, {0.25960177177315163, 0.9399485466790766}, {0.2570440145965068, 0.9382273117790964}, {0.2518723787304798, 0.9344655737375973}, {0.25056829031434524, 0.9334587090070675}, {0.249259906179311, 0.9324253044288389}, {0.24663060348133692, 0.9302789470890608}, {0.24132433318243357, 0.9256685873061581}, {0.23998835127578777, 0.9244499454021239}, {0.23864878070813447, 0.9232049286957612}, {0.23595922808240352, 0.9206358696918139}, {0.23054096021931234, 0.9151823889314494}, {0.2195676585466987, 0.903020610074518}, {0.21818490564567436, 0.9013833883019917}, {0.21679998420559143, 0.8997202859083414}, {0.21402399063550243, 0.8963165925009567}, {0.20844988352625804, 0.8892003710683802}, {0.19723301551264663, 0.8737422529806739}, {0.1959206097864771, 0.8718340689135001}, {0.19460754260806404, 0.8699039757574627}, {0.1919797091238057, 0.8659782296194292}, {0.18671923830049114, 0.8578657049721203}, {0.17619495576819572, 0.8406067013514136}, {0.17488048272529616, 0.8383533234303298}, {0.17356647973606984, 0.8360787778578279}, {0.17094016344047, 0.8314663905172509}, {0.1656962398213631, 0.8219899121054909}, {0.15525872696108844, 0.8020424910524767}, {0.15396009626814136, 0.7994569509504588}, {0.1526630392389992, 0.7968511394796487}, {0.1500739172067393, 0.7915789471135182}, {0.14491752968913058, 0.7807940298233337}, {0.13470714940434692, 0.7582765070224478}, {0.11481327927006393, 0.7095737879904227}, {0.1134955851079394, 0.7061022585023131}, {0.11218219212717788, 0.7026095307046831}, {0.10956862322273547, 0.6955608822351728}, {0.10439652985832897, 0.6812136680194409}, {0.0942896456386599, 0.6515429927312351}, {0.09304999737479787, 0.6477447245875939}, {0.09181586474280501, 0.6439269473575169}, {0.08936443739991798, 0.6362333117294254}, {0.08453095016299896, 0.6206168751119694}, {0.07515724174459117, 0.5884931548041483}, {0.07401436755019512, 0.5843964661838533}, {0.07287812731458934, 0.580282127109586}, {0.07062581390430925, 0.5720009839375851}, {0.0662036874209647, 0.5552322750279637}, {0.05770372175071292, 0.5208971534184924}, {0.05667471367158213, 0.5165329561549226}, {0.05565334888357978, 0.5121531206255403}, {0.05363378547679867, 0.503347057188838}, {0.04968895071241433, 0.48555304707770647}, {0.04218906096038329, 0.44926741215428345}, {0.04134860939751159, 0.44497441023167317}, {0.04051559247891673, 0.4406696408647724}, {0.038872029372009, 0.4320252505249175}, {0.03567593172826567, 0.41460023772988375}, {0.02965668268339985, 0.3792309574256659}, {0.028939979750172238, 0.3747633757675418}, {0.028231332528138215, 0.3702858678882091}, {0.026838346287087513, 0.3613015446945352}, {0.02415057084219395, 0.34321898332159473}, {0.019175169445080336, 0.30662495151583263}, {0.01859136813643846, 0.3020128355160502}, {0.018016138057632823, 0.29739271010323964}, {0.016891505525011088, 0.2881289191977666}, {0.014746324152761328, 0.26951060713719044}, {0.014231868897784284, 0.26483784420708634}, {0.013726201403868198, 0.26015805408796633}, {0.012741329624957883, 0.25077788752364527}, {0.01087811357117491, 0.23193868709166596}, {0.01043463815492398, 0.2272131831585396}, {0.01000013880365987, 0.2224816476604177}, {0.00915815396336101, 0.2130009833495564}, {0.007582814196188791, 0.19397280078221307}, {0.007211724898478139, 0.18920256235709626}, {0.0068497712737447865, 0.18442729949586126}, {0.006153342238855693, 0.17486220702993327}, {0.004870869858825478, 0.15567730904515834}, {0.0045790955100677285, 0.15096486208688534}, {0.00429622672113516, 0.1462485608564812}, {0.003757259162968122, 0.13680487684374218}, {0.003501186000426444, 0.13207773495776054}, {0.0032540696101390726, 0.12734722058545267}, {0.0027867536705200335, 0.11787655726212057}, {0.0025665763153072617, 0.11313664998080207}, {0.0023554001197577776, 0.10839385354564528}, {0.0019600908880123605, 0.09890007748852533}, {0.0017759766201004606, 0.09414934019883488}, {0.0016009010477189248, 0.0893961984129014}, {0.0012778988023742432, 0.0798831867994328}, {0.0011299874606471485, 0.07512355985562265}, {0.0009911454763500117, 0.07036201417605395}, {0.0008613794386632595, 0.06559867134707915}, {0.0007406955057333858, 0.06083365300657606}, {0.0006290994044429699, 0.05606708084045825}, {0.0005265964301968786, 0.0512990765791885}, {0.0004331914467245696, 0.046529761994290826}, {0.00034888888589845876, 0.04175925889485674}, {0.0002736927475684863, 0.03698768912405319}, {0.00020760659941278245, 0.0322151745556328}, {0.00015063357680441828, 0.0274418370904354}, {0.0001027763826943845, 0.022667798652893777}, {0.00006403728751067613, 0.017893181187538996}, {0.00003441812907353131, 0.013118106655505772}, {0.00001392031252680627, 0.008342697031033887}, {2.54481028551191*^-6, 0.0035670742979674003}, {-2.9216199952903676*^-7, 0.0012086395537348952}, {-7.162474533437195*^-6, 0.005984322531501631}, {-0.00002315542196253739, 0.010759852643650218}, {-0.00004827024558502363, 0.015535107902880175}, {-0.00008250575395029973, 0.02030996632976988}, {-0.00012586032290346158, 0.025084305956275744}, {-0.00017833189564594456, 0.029858004829231025}, {-0.00023991798281232284, 0.034630941013841544}, {-0.00031061566256319743, 0.03940299259717727}, {-0.0003904215806943059, 0.0441740376916712}, {-0.00047933195076181714, 0.048943954438618396}, {-0.0005773425542236099, 0.05371262101166169}, {-0.0006844487405968072, 0.058479915620287884}, {-0.0008006454276314296, 0.0632457165133212}, {-0.0010602878170040762, 0.07277235036552399}, {-0.0012037211977948448, 0.07753294005042304}, {-0.0013562204366126091, 0.08229154947816568}, {-0.001688387106273763, 0.09180234161375334}, {-0.001868038770408244, 0.09655428150150558}, {-0.002056724759740663, 0.1013037554988765}, {-0.0024611634541222226, 0.11079482093556159}, {-0.002676896956720002, 0.11553617012031236}, {-0.00290162638032996, 0.12027456891249406}, {-0.0033780298745508435, 0.12974203171609322}, {-0.004438289714165078, 0.14863623810301524}, {-0.004750462234536797, 0.15375044856767828}, {-0.005073106152034661, 0.15886003931793435}, {-0.005749735404328359, 0.16906474844147368}, {-0.007227817334272963, 0.18941199384957697}, {-0.0076232229089918755, 0.19448496221297018}, {-0.008028939118414366, 0.19955208886923786}, {-0.008871211988154682, 0.20966820959674115}, {-0.010678446952252719, 0.22982368007112325}, {-0.011155669098761672, 0.23484567279345808}, {-0.011643003959876655, 0.23986061442230544}, {-0.012647901950897683, 0.24986874401522002}, {-0.014777809986632407, 0.2697937997300578}, {-0.019511191135178494, 0.309245808449756}, {-0.020146620274843836, 0.31413710996484967}, {-0.02079165714706667, 0.31901898982046584}, {-0.0221104083494963, 0.32875390218695777}, {-0.0248615698010312, 0.34810427930013954}, {-0.03080966803090477, 0.3862950649997685}, {-0.031594161214955487, 0.39101832863585356}, {-0.0323876165343804, 0.39572988124764386}, {-0.034001233616409214, 0.40511729405212815}, {-0.03733400484036544, 0.4237454571600437}, {-0.044410956784523004, 0.4603843422042415}, {-0.04527131040762121, 0.46460219220905935}, {-0.046138793478170806, 0.4688079454684813}, {-0.04789497573059194, 0.47718272960347674}, {-0.05149097856101316, 0.4937824034899513}, {-0.05900742561045034, 0.5263584460535935}, {-0.0599764916282972, 0.5303699787320563}, {-0.06095195646483377, 0.5343677292559138}, {-0.06292188768418323, 0.5423214760855732}, {-0.06693637303636417, 0.5580591214913999}, {-0.07525249084224193, 0.5888327195116982}, {-0.0763179142494079, 0.5926117255393744}, {-0.07738891854063877, 0.5963753671081394}, {-0.07954745410751346, 0.6038561765217553}, {-0.08392910639041878, 0.6186292634503368}, {-0.09293827023994042, 0.6474006023040043}, {-0.11183647853663459, 0.7016847154141514}, {-0.11315441307800733, 0.7051981137395782}, {-0.11447670667828136, 0.7086901560430383}, {-0.1171340518336042, 0.715609768903494}, {-0.12249752147342859, 0.7291883231108722}, {-0.13340129902406045, 0.755281096525148}, {-0.13477923686506205, 0.7584410825259219}, {-0.13616022137029216, 0.7615781846267177}, {-0.1389309922641555, 0.7677833819456622}, {-0.1445053602656622, 0.7799153067751564}, {-0.1557661427546523, 0.8030464065129775}, {-0.15718255124661898, 0.8058300146606835}, {-0.15860062555977397, 0.8085894105476337}, {-0.16144141860985045, 0.814035261817237}, {-0.16713910013715444, 0.8246331226378413}, {-0.17857883290932489, 0.8446375413199627}, {-0.18001120462180065, 0.8470250833025764}, {-0.18144380905379073, 0.8493872951649312}, {-0.18430935225481626, 0.8540354787473184}, {-0.190039221976456, 0.863025185141674}, {-0.201473564651916, 0.8797650821817895}, {-0.20287267593035996, 0.8817044835090017}, {-0.20427059778160273, 0.8836186061319613}, {-0.20706252281629078, 0.8873708317161084}, {-0.21262824228369506, 0.8945699655915944}, {-0.21401538269968462, 0.8963058928902101}, {-0.2154006315355514, 0.898016195871857}, {-0.21816510258261104, 0.9013597718388838}, {-0.2236674735253331, 0.9077377511791759}, {-0.2250370169135166, 0.9092676255156575}, {-0.22640396439260113, 0.9107715831602122}, {-0.2291297192517071, 0.9137016181034994}, {-0.23454620228356413, 0.9192493018317527}, {-0.24521934664132733, 0.9290889151330326}, {-0.24653679241054036, 0.9302006611255941}, {-0.24785023519437407, 0.9312860672468151}, {-0.2504647615076182, 0.9333777837296188}, {-0.25564192890157356, 0.9372443482472951}, {-0.2569249021721605, 0.9381448839144889}, {-0.25820317377816404, 0.9390189495319027}, {-0.26074526426066796, 0.9406876216993992}, {-0.26576920580735036, 0.9437068350365082}, {-0.267012136637784, 0.9443953096157035}, {-0.2682496730739629, 0.9450572384553808}, {-0.2707082186041657, 0.9463014372548145}, {-0.27192905594376215, 0.946883698512632}, {-0.2731441553881168, 0.94743939662673}, {-0.27555679859835647, 0.9484710953825355}, {-0.27675417173242106, 0.9489470941303351}, {-0.2779454657154387, 0.9493965259460379}, {-0.2791305956136737, 0.9498193920080826}, {-0.2803094766532964, 0.9502156943444563}, {-0.2814820242242841, 0.9505854358323085}, {-0.28264815388431835, 0.9509286201975388}, {-0.2849608225641156, 0.9515353367047806}, {-0.28603027859461094, 0.9517819823489907}, {-0.28709386380889124, 0.9520055439959078}, {-0.28815151039929077, 0.952206028037268}, {-0.2892031507022856, 0.9523834415048921}, {-0.2902487172012177, 0.9525377920702515}, {-0.2912881425290143, 0.952669088044014}, {-0.2923213594709031, 0.9527773383755699}, {-0.29334830096712355, 0.9528625526525404}, {-0.2943689001156334, 0.9529247411002639}, {-0.29538309017481085, 0.9529639145812662}, {-0.29639080456615247, 0.9529800845947094}, {-0.2973919768769654, 0.9529732632758214}, {-0.29838654086305677, 0.9529434633953086}, {-0.29937443045141615, 0.9528906983587454}, {-0.30035557974289323, 0.9528149822059486}, {-0.3013299230148724, 0.9527163296103296}, {-0.30229739472393985, 0.95259475587823}, {-0.3032579295085471, 0.952450276948235}, {-0.30421146219166784, 0.9522829093904712}, {-0.30515792778345047, 0.9520926704058832}, {-0.3060972614838642, 0.9518795778254903}, {-0.3070293986853403, 0.9516436501096277}, {-0.30795427497540667, 0.9513849063471651}, {-0.30887182613931746, 0.9511033662547078}, {-0.3097819881626769, 0.9507990501757799}, {-0.3106846972340559, 0.9504719790799864}, {-0.31157988974760337, 0.9501221745621584}, {-0.3124675023056515, 0.9497496588414787}, {-0.31334747172131455, 0.9493544547605882}, {-0.31421973502108097, 0.9489365857846744}, {-0.3159408924612583, 0.9480329501156565}, {-0.31678966174475753, 0.9475472334571899}, {-0.3176304752036753, 0.9470389519710207}, {-0.31928798740461567, 0.9459548013866026}, {-0.32010456309958024, 0.9453789872645317}, {-0.3209129368809302, 0.9447807182650122}, {-0.32250483519136125, 0.943516932341986}, {-0.32328823856456046, 0.9428514753025501}, {-0.3240631977174059, 0.9421636831515556}, {-0.32558754374370374, 0.9407212199900418}, {-0.3285322834302517, 0.9375696975262747}, {-0.329246468084107, 0.9367264746197743}, {-0.32995173589212473, 0.9358611852877011}, {-0.33133528961477116, 0.9340645531252442}, {-0.3320134605253882, 0.9331332846772543}, {-0.3326824845869757, 0.9321800985657948}, {-0.33399286517239823, 0.9302081286580989}, {-0.334634108898864, 0.9291894239941973}, {-0.33526598018632403, 0.92814895992867}, {-0.33650138297184845, 0.926002918336571}, {-0.3388572883221847, 0.9214515672841791}, {-0.3394692532582646, 0.9201583543046151}, {-0.3400696762279984, 0.9188399612998277}, {-0.3412356248664026, 0.9161278694654069}, {-0.34180101589769246, 0.9147342897437808}, {-0.34235459569467097, 0.9133157682080724}, {-0.34342605706411927, 0.9104041465808828}, {-0.34394380747318043, 0.908911171890473}, {-0.3444494843277758, 0.9073935061830275}, {-0.34542435994698883, 0.9042843610783035}, {-0.34722642227113326, 0.8977726748252013}, {-0.34764580457929817, 0.8960840252595034}, {-0.3480526103845723, 0.8943712233695152}, {-0.34882824987606165, 0.8908734464172595}, {-0.34919696345160633, 0.8890886151323407}, {-0.3495528603091588, 0.8872799190720955}, {-0.35022596897203195, 0.883591228377142}, {-0.35054306455368955, 0.8817113834662138}, {-0.35084711097611915, 0.8798079732215781}, {-0.3511380513666247, 0.8778810743397377}, {-0.35141582935357885, 0.8759307642456038}, {-0.35168038906940136, 0.8739571210894388}, {-0.3519316751535188, 0.8719602237437742}, {-0.3523942075369872, 0.8678969855667292}, {-0.3526053456765739, 0.8658308060636482}, {-0.35280299387070546, 0.8637416950213239}, {-0.3529870993375286, 0.8616297348765043}, {-0.353157609819531, 0.8594950087691822}, {-0.3533144735863622, 0.8573376005393445}, {-0.35345763943763, 0.8551575947236886}, {-0.3535870567056785, 0.8529550765523203}, {-0.35370267525834626, 0.8507301319454292}, {-0.3538044455017002, 0.848482847509934}, {-0.3538923183827517, 0.8462133105361095}, {-0.35396624539215116, 0.8439216089941916}, {-0.35402617856685903, 0.8416078315309518}, {-0.35407207049279793, 0.8392720674662553}, {-0.3541038743074835, 0.836914406789596}, {-0.3541215437026304, 0.8345349401566019}, {-0.3541250329267413, 0.8321337588855271}, {-0.3541142967876698, 0.8297109549537129}, {-0.3540892906551638, 0.8272666209940309}, {-0.3540499704633875, 0.8248008502913065}, {-0.35399629271341787, 0.8223137367787102}, {-0.3539282144757233, 0.8198053750341404}, {-0.35384569339261596, 0.8172758602765716}, {-0.3537486876806837, 0.8147252883623888}, {-0.3536371561332004, 0.8121537557816998}, {-0.3535110581225102, 0.8095613596546211}, {-0.353370353602392, 0.8069481977275464}, {-0.35321500311040094, 0.8043143683693965}, {-0.35304496777018407, 0.8016599705678396}, {-0.3528602092937759, 0.7989851039254985}, {-0.35266068998387023, 0.7962898686561369}, {-0.35221722104109615, 0.7908386961240501}, {-0.35198989221471944, 0.7882668778764572}, {-0.3517495766673533, 0.7856776590767433}, {-0.351229872732205, 0.7804473559002664}, {-0.35095042915285696, 0.777806440677218}, {-0.35065788847236273, 0.7751484632038619}, {-0.35003341050425835, 0.7697816646547071}, {-0.34970142172662005, 0.7670730162389384}, {-0.34935623287062734, 0.7643476508872014}, {-0.3486261570860648, 0.7588471194114439}, {-0.34700655474792114, 0.7476493271993883}, {-0.3465683035949435, 0.7448094141157108}, {-0.34611666897832166, 0.7419534950270243}, {-0.3451731667095636, 0.7361940020857574}, {-0.3431246784810352, 0.7244869621600468}, {-0.3383777625139335, 0.70034151191227}, {-0.33772315331579544, 0.6972566069914403}, {-0.3370548864464448, 0.6941571938178517}, {-0.3356773283586423, 0.6879152300762964}, {-0.3327577833786117, 0.6752614840482796}, {-0.32625874849895403, 0.649296849769851}, {-0.3253843421001163, 0.6459916140109329}, {-0.32449613096033414, 0.6426734583940944}, {-0.32267827542853306, 0.6359987958291377}, {-0.3188767268173204, 0.6224989396483548}, {-0.3106098747939013, 0.5949203369742881}, {-0.3095360034856283, 0.591489812268369}, {-0.3084488499363478, 0.588048470424241}, {-0.30623470922074214, 0.5811337363675548}, {-0.3016472119695314, 0.5671788866314635}, {-0.2918367077900936, 0.5387903990160915}, {-0.26969097842385875, 0.4802827888515886}, {-0.2681960317157131, 0.47655682977461444}, {-0.2666881430606366, 0.4728233570532241}, {-0.26363361484236947, 0.4653342944061195}, {-0.25737010634728325, 0.45027067323513625}, {-0.24423009878765575, 0.4198253247878567}, {-0.21554311778372306, 0.3578544545239761}, {-0.2134836113213651, 0.3536078708260588}, {-0.21140985151201405, 0.3493564955144243}, {-0.20721974606224694, 0.3408399235838226}, {-0.19867044765952158, 0.32375537057519455}, {-0.18090608960738236, 0.2894116436716607}, {-0.14280760658558764, 0.22027353886590417}, {-0.1403166955619593, 0.2159422261829284}, {-0.13781323589897204, 0.2116105401057374}, {-0.1327689253901623, 0.20294659520473374}, {-0.12253255643639108, 0.18562023915391918}, {-0.10148386056170007, 0.15100413752890965}, {-0.05721072215921225, 0.08215983075118463}, {-0.054544447516570824, 0.07816998800588418}, {-0.05186926536463003, 0.07418355130206385}, {-0.046492444362713374, 0.06622132003376098}, {-0.03563485875754308, 0.05034238240811204}, {-0.03289920439337705, 0.04638274575149631}, {-0.030155184513558354, 0.042427357616896706}, {-0.024642326656419033, 0.034529743940078864}, {-0.013519309202947341, 0.018790073002706256}, {-0.0107186986336417, 0.01486733214707173}, {-0.007910289370780449, 0.010949667675090806}, {-0.005094153961375905, 0.007037181963613012}, {-0.002270365321993008, 0.0031299771336903023}, {-0.0005610032633736669, 0.0007718449524119984}, {-0.003399878145436881, 0.004668182694194335}, {-0.009099850388959581, 0.012444000069434774}, {-0.011960798644807957, 0.016323277837394807}, {-0.014828954990500114, 0.02019666753592497}, {-0.02058658982639067, 0.027925382014789238}, {-0.032185013604196205, 0.04330776214079015}, {-0.03510149460372335, 0.04713714427881056}, {-0.038024570412273695, 0.050959844354761495}, {-0.0438901933328966, 0.058584806898436405}, {-0.055697113953441284, 0.0737502586897249}, {-0.07959569866168784, 0.10372175004809812}, {-0.12838404513450455, 0.16205938055628877}, {-0.13173088265243615, 0.1659274877938192}, {-0.1350823404287218, 0.16978424323229585}, {-0.14179866952551212, 0.1774632565870516}, {-0.15528183196428255, 0.1926802284636727}, {-0.18242469566036573, 0.22252589644689352}, {-0.23720864253358181, 0.27967934766116964}, {-0.24064633313662373, 0.2831320836966883}, {-0.2440849165417696, 0.286570208828176}, {-0.2509642783537697, 0.2934022594695508}, {-0.26472838655229797, 0.30688702979893473}, {-0.29225081599035063, 0.33311953282128604}, {-0.34705171902565873, 0.3824876304020765}, {-0.3502317392131393, 0.3852361391428722}, {-0.3534091028935087, 0.3879696358380743}, {-0.35975545268508463, 0.39339136480409004}, {-0.37241178315485657, 0.40405219754751176}, {-0.3975560685708278, 0.4246312837726453}, {-0.44698777580077453, 0.46272999651102265}, {-0.45003208008849377, 0.46497248724520457}, {-0.45307045697824744, 0.46719841112113125}, {-0.45912902126617405, 0.47160040200016184}, {-0.4711705444489794, 0.48020389461165675}, {-0.49492845397444096, 0.49660091837045656}, {-0.540963317507906, 0.5260989447399594}, {-0.5437129740097416, 0.5277616615324316}, {-0.5464538486519152, 0.5294075273771943}, {-0.5519088825330033, 0.5326486296825966}, {-0.5627095089688494, 0.538927811639429}, {-0.583852502924329, 0.5506705041930057}, {-0.5864506446127093, 0.5520615999188049}, {-0.5890385452643409, 0.5534356020775658}, {-0.5941832653041795, 0.5561322842074552}, {-0.6043458850419923, 0.5613200999382869}, {-0.6241440656630934, 0.5708719178518658}, {-0.6265677215591204, 0.5719885614681435}, {-0.62897972954326, 0.5730880089018457}, {-0.6337684583785935, 0.5752353086181353}, {-0.6432023781766346, 0.5793235115529021}, {-0.6455304596927899, 0.5803025660136434}, {-0.647846213294034, 0.5812644251520902}, {-0.6524404018613134, 0.5831365681454509}, {-0.6614771759559412, 0.5866746578691647}, {-0.6637043037468408, 0.5875162508485149}, {-0.6659184412325487, 0.5883406838224831}, {-0.6703074196391717, 0.589938097573484}, {-0.6789259276506546, 0.592927340805213}, {-0.6810468683908016, 0.5936318745200349}, {-0.6831541755047031, 0.5943193176725543}, {-0.6873275731646583, 0.5956429770596411}, {-0.6955073106301248, 0.5980857309620013}, {-0.6976865525945177, 0.598701220367418}, {-0.6998490131954985, 0.5992967243843508}, {-0.7041232012842497, 0.6004278551304351}, {-0.7062347354587645, 0.6009635230915101}, {-0.708329101651331, 0.6014792881248842}, {-0.7124659488222675, 0.6024511997285654}, {-0.7145082404043319, 0.6029073932362764}, {-0.716532985219686, 0.603343777687817}, {-0.7205294612888256, 0.6041572210819875}, {-0.7225010071887625, 0.604534332615042}, {-0.7244546356219963, 0.6048917402692832}, {-0.7283077750819565, 0.6055475568309305}, {-0.7302071049206655, 0.605846023925947}, {-0.732088154924211, 0.606124903513673}, {-0.7339508358307804, 0.6063842264223563}, {-0.7357950589181939, 0.6066240241706047}, {-0.7376207360076266, 0.6068443289654435}, {-0.7394277794673215, 0.6070451737003556}, {-0.7412161022162745, 0.6072265919532889}, {-0.7429856177279022, 0.6073886179846402}, {-0.7447362400336988, 0.6075312867352176}, {-0.7464678837268668, 0.6076546338241736}, {-0.7481804639659279, 0.6077586955469129}, {-0.7498738964783236, 0.607843508872979}, {-0.7515480975639888, 0.607909111443913}, {-0.7532029840989055, 0.6079555415710874}, {-0.7548384735386453, 0.6079828382335171}, {-0.7564544839218855, 0.6079910410756448}, {-0.7580509338739037, 0.6079801904051015}, {-0.7596277426100596, 0.6079503271904433}, {-0.7611848299392542, 0.607901493058864}, {-0.7627221162673643, 0.607833730293882}, {-0.764239522600662, 0.6077470818330063}, {-0.7657369705492137, 0.607641591265375}, {-0.7672143823302547, 0.6075173028293712}, {-0.7686716807715458, 0.6073742614102171}, {-0.7701087893147107, 0.6072125125375412}, {-0.7715256320185486, 0.6070321023829233}, {-0.772922133562327, 0.6068330777574178}, {-0.774298219249057, 0.606615486109051}, {-0.7756538150087409, 0.6063793755202949}, {-0.7769888474016029, 0.606124794705522}, {-0.7795969314980964, 0.6055604203994591}, {-0.7808698395020478, 0.6052507274731538}, {-0.7821218967461242, 0.6049227654455459}, {-0.7833530329893401, 0.6045765861514797}, {-0.7845631786398507, 0.6042122420419311}, {-0.7857522647580256, 0.6038297861813015}, {-0.7869202230595023, 0.603429272244687}, {-0.7891924863694174, 0.6025742878808381}, {-0.7902966581126333, 0.6021199278324044}, {-0.791379435514652, 0.6016477304599761}, {-0.7934805481161171, 0.6006500510844754}, {-0.7944987554117299, 0.6001246842338464}, {-0.7954953125642397, 0.5995817103583245}, {-0.7974232281111283, 0.5984431782945873}, {-0.7982930380639102, 0.5978879925965912}, {-0.7991437795463272, 0.597317675291073}, {-0.8007878639198938, 0.5961318456262673}, {-0.8015811115416083, 0.5955164342316346}, {-0.8023551001579048, 0.5948860931544627}, {-0.8038451156872674, 0.5935808285880743}, {-0.804561051596346, 0.5929060094750865}, {-0.8052575464957175, 0.5922164694275865}, {-0.8065920371907769, 0.5907934398906586}, {-0.8072299463068794, 0.5900600581181408}, {-0.8078482410576561, 0.5893121708403971}, {-0.8090258201160972, 0.5877730997128093}, {-0.8095850221254183, 0.5869820268480113}, {-0.8101244451761455, 0.5861766704434441}, {-0.8111437958928451, 0.5845233333932355}, {-0.8116236456964103, 0.5836754669266202}, {-0.812083560819849, 0.5828135452735559}, {-0.8129434374619328, 0.581047769070493}, {-0.8133433256066205, 0.5801440318177794}, {-0.8137231323264112, 0.5792264739683588}, {-0.8140828223496189, 0.5782951551260701}, {-0.8144223609734434, 0.5773501352724445}, {-0.8147417140654822, 0.5763914747642508}, {-0.8150408480652219, 0.5754192343310296}, {-0.8153197299855155, 0.5744334750726183}, {-0.815578327414039, 0.5734342584566606}, {-0.8158166085147324, 0.5724216463161073}, {-0.8160345420292218, 0.5713957008467058}, {-0.8162320972782231, 0.570356484604481}, {-0.8164092441629297, 0.5693040605032015}, {-0.8165659531663799, 0.568238491811836}, {-0.816702195354808, 0.5671598421520009}, {-0.8168179423789763, 0.5660681754953971}, {-0.8169131664754891, 0.564963556161235}, {-0.8169878404680893, 0.5638460488136482}, {-0.817041937768936, 0.5627157184590996}, {-0.8170754323798636, 0.5615726304437781}, {-0.817088298893623, 0.5604168504509807}, {-0.8170805124951049, 0.5592484444984865}, {-0.817052048962544, 0.5580674789359255}, {-0.8170028846687043, 0.5568740204421304}, {-0.8169329965820467, 0.5556681360224821}, {-0.8168423622678777, 0.5544498930062469}, {-0.8167309598894802, 0.553219359043903}, {-0.8165987682092243, 0.5519766021044543}, {-0.8164457665896597, 0.550721690472742}, {-0.8162719349945912, 0.5494546927467411}, {-0.8160772539901321, 0.5481756778348477}, {-0.8158617047457426, 0.5468847149531633}, {-0.8156252690352456, 0.5455818736227633}, {-0.8153679292378269, 0.5442672236669596}, {-0.8150896683390129, 0.5429408352085544}, {-0.8144703182167654, 0.5402531247560703}, {-0.8141291980046472, 0.5388919444802146}, {-0.8137670947155913, 0.5375193091326521}, {-0.8129798836435548, 0.5347399598203032}, {-0.8125182274345186, 0.5332152016827568}, {-0.8120318677142505, 0.5316773368507435}, {-0.810984980358774, 0.5285626576832633}, {-0.8104244261551294, 0.5269860295681682}, {-0.8098391153056368, 0.5253966671908584}, {-0.8085941798938588, 0.5221801160922425}, {-0.8079345355769704, 0.5205531164800326}, {-0.8072500951070285, 0.5189137608152106}, {-0.8058067956053466, 0.5155983633592581}, {-0.802622359659358, 0.5088235169377693}, {-0.8017641774374373, 0.5071003594621472}, {-0.8008811611785048, 0.5053656185833604}, {-0.7990406239595625, 0.5018617789926956}, {-0.7950615609447795, 0.4947194319273492}, {-0.7859124492689284, 0.4799184141249423}, {-0.7846573031118715, 0.4780218468900421}, {-0.783377412952394, 0.4761152986109177}, {-0.7807434535014024, 0.4722726685855323}, {-0.7751792368787771, 0.4644721551215269}, {-0.7628696935078761, 0.4484333378900669}, {-0.7335704851605077, 0.41479282256025507}, {-0.7315716138337616, 0.412668126957455}, {-0.729549707539723, 0.4105370556252015}, {-0.725436944864926, 0.40625619291132026}, {-0.7169367474057733, 0.3976224414874623}, {-0.6988473554471688, 0.3800897203091571}, {-0.6584006731345752, 0.344146969947351}, {-0.6556876935816681, 0.341868969676075}, {-0.6529533008460925, 0.33958786872001007}, {-0.6474205302547968, 0.335016773581571}, {-0.6361008717610885, 0.3258418688797204}, {-0.6124602985890435, 0.3073840087365765}, {-0.5613057107949403, 0.2702172244710096}, {-0.5581691203328316, 0.26804657037009577}, {-0.5550159748235522, 0.2658760172499802}, {-0.5486603000425831, 0.26153553346036057}, {-0.5357534140557719, 0.2528592842732571}, {-0.5091739148819929, 0.23554336128233755}, {-0.45309006340385743, 0.20119717516954552}, {-0.44946104374631574, 0.19906859872537847}, {-0.4458179637203894, 0.19694259778258733}, {-0.43848996635649773, 0.19269861862466298}, {-0.423669054127711, 0.18424481208601098}, {-0.39338738515794475, 0.16749009591029915}, {-0.33042901207114866, 0.13471921304255396}, {-0.3260541925224027, 0.13253933527363537}, {-0.32166633067702677, 0.13036513539533026}, {-0.31285198604744097, 0.12603410394691486}, {-0.2950723830745079, 0.11744383335925951}, {-0.2589384970053112, 0.10056865480832233}, {-0.18461007181631592, 0.06817929162011853}, {-0.17988295409642527, 0.06622048573074026}, {-0.17514706968180188, 0.0642698173588077}, {-0.16564956333265782, 0.060393168676323675}, {-0.1465555588643968, 0.05274052356174348}, {-0.10800354188280027, 0.03785260731064001}, {-0.10315322049566211, 0.036031993493539666}, {-0.0982964227162179, 0.03422055898261325}, {-0.08856398173379781, 0.030625469703493877}, {-0.0690278203589528, 0.023548072223188065}, {-0.029703425007111185, 0.00985723284509578}, {-0.025095056571181898, 0.008300818653908085}, {-0.020483096802489935, 0.006753189099036077}, {-0.011248891328625379, 0.0036844516647245398}, {-0.006626890051435621, 0.0021634265717375627}, {-0.002001786290190362, 0.0006513516854824901}, {-0.0026262973604016656, 0.0008517327031811535}, {-0.00725723815989307, 0.0023457867465191133}, {-0.011890913225765033, 0.003830771041112702}, {-0.016527199537081486, 0.005306646629025827}, {-0.025807113142230748, 0.008230918087376924}, {-0.04439284787125561, 0.013968492089231753}, {-0.049043957458506905, 0.015379553566204237}, {-0.05369668958161232, 0.016781210844936188}, {-0.06300652492919752, 0.01955617366846293}, {-0.08164019900817387, 0.024991744007564643}, {-0.11893564035620895, 0.03539826300957181}, {-0.12359776828626508, 0.0366549266332101}, {-0.12825952303405322, 0.03790168293761176}, {-0.13758141182778097, 0.04036536409001945}, {-0.1562151963968718, 0.045172660420807635}, {-0.1934146900195483, 0.05430146486414066}, {-0.2673165143089894, 0.0705781217285296}, {-0.27181433782818004, 0.0714883173826489}, {-0.27630795032138633, 0.07238837274208527}, {-0.28528207218591645, 0.07415801696095982}, {-0.3031746174000357, 0.07757514506672952}, {-0.3387106989250155, 0.08391878013977787}, {-0.3431270896201174, 0.08466559907651504}, {-0.3475373994936292, 0.08540214701378981}, {-0.35633931309542805, 0.08684441282722946}, {-0.37386507728136975, 0.08960552971545696}, {-0.4085785650473555, 0.09463372220896563}, {-0.4128838258695706, 0.09521592908440604}, {-0.4171811678193897, 0.095787848281229}, {-0.42575164124656634, 0.09690083497777258}, {-0.4427925795512824, 0.09900350563195774}, {-0.4470313215131469, 0.09950351203390655}, {-0.4512612417860233, 0.0999932648790545}, {-0.45969416960148574, 0.10094203529903038}, {-0.4764492524706679, 0.10271683731722868}, {-0.48061429767336084, 0.10313501445499608}, {-0.4847696314382237, 0.10354300017482952}, {-0.49305072402249145, 0.10432843668566093}, {-0.5094915382438435, 0.10577746944988588}, {-0.513575819188803, 0.10611441166462052}, {-0.5176495136841369, 0.10644125231823598}, {-0.5257647105526391, 0.10706468202333348}, {-0.5418633210787434, 0.10819093044500594}, {-0.5461988644193353, 0.10846878302550018}, {-0.5505209250272132, 0.10873488457438525}, {-0.5591240566407782, 0.10923191999850636}, {-0.5634048579358972, 0.10946289803706069}, {-0.5676716370865038, 0.10968221336785385}, {-0.576162593996135, 0.11008595059440172}, {-0.5803865053140924, 0.11027042127271848}, {-0.5845958616146409, 0.1104433268054137}, {-0.5887905307359446, 0.11060469301544887}, {-0.5929703809479118, 0.11075454629568658}, {-0.5971352809567536, 0.110892913607313}, {-0.6012850999095368, 0.11101982247823668}, {-0.6054197073987212, 0.11113530100146304}, {-0.6095389734666746, 0.1112393778334453}, {-0.6136427686101938, 0.11133208219241217}, {-0.6177309637849827, 0.11141344385666963}, {-0.6218034304101309, 0.11148349316288163}, {-0.6258600403725788, 0.11154226100432528}, {-0.6299006660315638, 0.11158977882912453}, {-0.6339251802230413, 0.11162607863845729}, {-0.637933456264106, 0.1116511929847422}, {-0.6419253679573808, 0.11166515496980076}, {-0.6459007895954071, 0.11166799824299614}, {-0.6498595959650034, 0.11165975699934917}, {-0.6538016623516091, 0.11164046597763125}, {-0.6577268645436272, 0.11161016045843475}, {-0.6616350788367259, 0.11156887626221891}, {-0.6655261820381406, 0.11151664974733493}, {-0.6694000514709577, 0.1114535178080276}, {-0.6732565649783644, 0.11137951787241374}, {-0.6770956009279061, 0.11129468790043856}, {-0.6809170382157076, 0.11119906638180978}, {-0.6847207562706781, 0.11109269233390949}, {-0.6885066350587102, 0.11097560529968284}, {-0.6922745550868454, 0.1108478453455053}, {-0.6960243974074356, 0.11070945305902859}, {-0.7034693758866987, 0.11040093643307622}, {-0.7071642769137247, 0.11023089585557899}, {-0.710840629978096, 0.11005039046527647}, {-0.7181372281508583, 0.1096581583978976}, {-0.7217572426538628, 0.10944651956405392}, {-0.7253582479914578, 0.10922459159923559}, {-0.7325027763319835, 0.10875004948945703}, {-0.7360460733841524, 0.10849752719482865}, {-0.7395699093772543, 0.10823489946508467}, {-0.7465587528306467, 0.10767951682213671}, {-0.750023539120633, 0.10738685768950615}, {-0.7534684220186401, 0.10708428467793758}, {-0.760298042014026, 0.10644959389160956}, {-0.7636825628447678, 0.10611757574816044}, {-0.7670467477577931, 0.10577584298326226}, {-0.7737136841796604, 0.10506343805189099}, {-0.7867988794338228, 0.10352432790896188}, {-0.7898031685265657, 0.10314364022175257}, {-0.79278902974359, 0.10275485711998658}, {-0.7987051392393423, 0.10195318254803777}, {-0.810312629468694, 0.10025466393456836}, {-0.8131672150773607, 0.0998104745335612}, {-0.8160027273402741, 0.09935855417315097}, {-0.8216162188296463, 0.09843170906010083}, {-0.8326109079210817, 0.09648733863683363}, {-0.835310741414803, 0.09598263664233699}, {-0.8379908892161403, 0.09547058894087851}, {-0.8432918316628127, 0.09442465498689911}, {-0.8536542616607797, 0.09224683988388298}, {-0.8734054240090705, 0.08755892639522991}, {-0.8757816934619322, 0.08694280844006581}, {-0.8781371565954565, 0.08632017333596817}, {-0.8827854033601218, 0.08505556874474667}, {-0.8918293872280488, 0.0824505564809809}, {-0.8940374104952832, 0.08178383386041135}, {-0.8962241213433302, 0.08111103511440357}, {-0.9005333637830116, 0.07974743467670516}, {-0.9088934706501267, 0.07694981397553532}, {-0.9109293294894174, 0.07623607251485331}, {-0.9129434072895543, 0.07551671219071232}, {-0.9169059967903566, 0.07406136813627044}, {-0.9245673843080742, 0.07108583154036938}, {-0.9388232879351648, 0.06488871151635217}, {-0.9406983195825123, 0.06399924032084234}, {-0.942545179157959, 0.06310397030499564}, {-0.9461541272893745, 0.06129637731486708}, {-0.9530312104799415, 0.05761540035643333}, {-0.9546791297777617, 0.056681957316610465}, {-0.9562983928306398, 0.055743409911031265}, {-0.9594507268177732, 0.05385135444153531}, {-0.9654091182593241, 0.05000988223591117}, {-0.9668262726285926, 0.04903808375294194}, {-0.9682143498734019, 0.0480618922799079}, {-0.9709030815687181, 0.046096690562890494}, {-0.9759295703253095, 0.04211754564910643}, {-0.9771128172172342, 0.04111313522469106}, {-0.9782666304216785, 0.040105057736047896}, {-0.9804857967754007, 0.038078268327043745}, {-0.9845692335269307, 0.03398472919791393}, {-0.9855159480619144, 0.03295355831401011}, {-0.9864329375297514, 0.031919458428614454}, {-0.9881776150860492, 0.029842843758375752}, {-0.9890052426641501, 0.028800515387724178}, {-0.989803024156684, 0.02775563083431077}, {-0.991308939234596, 0.02565856749464528}, {-0.992017020584304, 0.02460657617944271}, {-0.9926951513790904, 0.023552403613876397}, {-0.9939614682476058, 0.02143789094756139}, {-0.9945496103921214, 0.02037773921951434}, {-0.9951077141249578, 0.019315782976765398}, {-0.9961337299500572, 0.017186834752193687}, {-0.9966016064462798, 0.01612003188886401}, {-0.9970393733400948, 0.015051802738043376}, {-0.9974470154431848, 0.013982242052953125}, {-0.9978245186121918, 0.012911444655594559}, {-0.9981718697492555, 0.011839505431848713}, {-0.9984890568025117, 0.010766519326573264}, {-0.998776068766549, 0.009692581338698488}, {-0.9990328956828287, 0.00861778651631859}, {-0.9992595286400627, 0.007542229951780787}, {-0.9994559597745529, 0.006466006776774146}, {-0.999622182270489, 0.0053892121574166466}, {-0.9997581903602085, 0.004311941289338913}, {-0.9998639793244157, 0.0032342893927665927}, {-0.9999395454923615, 0.0021563517076034055}, {-0.9999848862419821, 0.0010782234885132922}, {-1., 1.2246467991473532*^-16}}]}, "Charting`Private`Tag$24584#1"]}}, {}, {{{}, {}, {}, {}}, {}}}, {}}, {DisplayFunction -> Identity, PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.02], Scaled[0.02]}}, PlotRangeClipping -> True, ImagePadding -> All, PlotInteractivity :> $PlotInteractivity, PlotRange -> {{Automatic, Automatic}, {Automatic, Automatic}}, PlotRangePadding -> Automatic, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, Axes -> True, AxesOrigin -> {0, 0}, CoordinatesToolOptions -> {"DisplayFunction" -> ({Sqrt[#1[[1]]^2 + #1[[2]]^2], Mod[ArcTan[#1[[1]], #1[[2]]], 2*Pi]} & ), "CopiedValueFunction" -> ({Sqrt[#1[[1]]^2 + #1[[2]]^2], Mod[ArcTan[#1[[1]], #1[[2]]], 2*Pi]} & )}, DisplayFunction :> Identity, FrameTicks -> {{Automatic, Automatic}, {Automatic, Automatic}}, GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]], Method -> {"AxisPadding" -> Scaled[0.02], "DefaultBoundaryStyle" -> Automatic, "DefaultGraphicsInteraction" -> {"Version" -> 1.2, "TrackMousePosition" -> {True, False}, "Effects" -> {"Highlight" -> {"ratio" -> 2}, "HighlightPoint" -> {"ratio" -> 2}, "Droplines" -> {"freeformCursorMode" -> True, "placement" -> {"x" -> "All", "y" -> "None"}}}}, "DefaultMeshStyle" -> AbsolutePointSize[6], "DefaultPlotStyle" -> {Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Directive[RGBColor[0.95, 0.627, 0.1425], AbsoluteThickness[2]], Directive[RGBColor[0.455, 0.7, 0.21], AbsoluteThickness[2]], Directive[RGBColor[0.922526, 0.385626, 0.209179], AbsoluteThickness[2]], Directive[RGBColor[0.578, 0.51, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.772079, 0.431554, 0.102387], AbsoluteThickness[2]], Directive[RGBColor[0.4, 0.64, 1.], AbsoluteThickness[2]], Directive[RGBColor[1., 0.75, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.8, 0.4, 0.76], AbsoluteThickness[2]], Directive[RGBColor[0.637, 0.65, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.915, 0.3325, 0.2125], AbsoluteThickness[2]], Directive[RGBColor[0.40082222609352647, 0.5220066643438841, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.9728288904374106, 0.621644452187053, 0.07336199581899142], AbsoluteThickness[2]], Directive[RGBColor[0.736782672705901, 0.358, 0.5030266573755369], AbsoluteThickness[2]], Directive[RGBColor[0.28026441037696703, 0.715, 0.4292089322474965], AbsoluteThickness[2]]}, "DomainPadding" -> Scaled[0.02], "RangePadding" -> Scaled[0.05]}, PlotRange -> {{-1., 1.}, {0., 0.9529805509671492}}, PlotRangeClipping -> True, PlotRangePadding -> {Scaled[0.02], Scaled[0.02]}, Ticks -> {Automatic, Automatic}}]"#,
    );
  }
  #[test]
  fn polar_plot_3() {
    assert_case(
      r#"PolarPlot[Cos[5t], {t, 0, Pi}]; PolarPlot[Abs[Cos[5t]], {t, 0, Pi}]; PolarPlot[{1, 1 + Sin[20 t] / 5}, {t, 0, 2 Pi}]"#,
      r#"Graphics[{{{{}, {}}, {}, {{{}, {}, Annotation[{Hue[0.67, 0.6, 0.6], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Line[{{1., 0.}, {0.9999981432641898, 0.0019270361109708098}, {0.9999925730636541, 0.00385406506594771}, {0.9999832894190775, 0.005781079708963364}, {0.9999702923649348, 0.007708072884103584}, {0.9999535819494902, 0.0096350374355339}, {0.9999331582347972, 0.011561966207526137}, {0.9999090212966988, 0.013488852044484986}, {0.9998811712248267, 0.015415687790974575}, {0.9998496081226015, 0.017342466291745042}, {0.9998143321072317, 0.019269180391759105}, {0.999775343309714, 0.021195822936218637}, {0.999732641874832, 0.02312238677059122}, {0.9996862279611565, 0.025048864740636737}, {0.999636101741044, 0.026975249692433912}, {0.9995247131398624, 0.03082771192735184}, {0.9994634511724317, 0.03275377490446341}, {0.9993984777258393, 0.03467971625136141}, {0.9992573973740597, 0.038531205447280795}, {0.9991812909927702, 0.04045673899390635}, {0.999101474180113, 0.04238212230557979}, {0.9989307104600623, 0.046232409625229655}, {0.9988397641867953, 0.04815729933527338}, {0.9987451087504106, 0.0500820102145527}, {0.9985446718080617, 0.05393086689207186}, {0.9980993043546484, 0.06162611984188932}, {0.9979786953896135, 0.06354937882029202}, {0.9978543804590155, 0.06547240180987997}, {0.9975946345614501, 0.06931771125914929}, {0.9970306924135589, 0.07700518414586999}, {0.9968804494203873, 0.07892635531561416}, {0.9967265045399579, 0.08084723339457779}, {0.9964075114177497, 0.08468808174877324}, {0.9957251286004899, 0.09236594758642257}, {0.9955452872855635, 0.09428457436667424}, {0.9953617490414658, 0.09620285102383476}, {0.9949835845057389, 0.10003832547634522}, {0.9941829231925392, 0.10770475956213632}, {0.9924044427064662, 0.12301797468853186}, {0.9921452767887315, 0.12509096588407673}, {0.9918817806931483, 0.127163411125171}, {0.9913418025874019, 0.13130662756601683}, {0.99020993610812, 0.13958611117425973}, {0.9877388450548672, 0.1561152586061882}, {0.9874105448646385, 0.15817843054638553}, {0.9870779351611068, 0.16024091212323371}, {0.9863997930395828, 0.16436376818346624}, {0.984991859769404, 0.17260080007928943}, {0.9819697475369292, 0.1890381308685088}, {0.9815726798505238, 0.19108917858178473}, {0.9811713281299405, 0.1931393922940856}, {0.9803557796116636, 0.1972372819271428}, {0.9786733524908273, 0.2054226597149519}, {0.9751035953768852, 0.22174981010831057}, {0.9671480594262245, 0.25421375090283377}, {0.96665032840068, 0.25609986841632215}, {0.9661489191010385, 0.2579850114248789}, {0.9651350733252556, 0.26175234523773333}, {0.9630633273712093, 0.26927500343087013}, {0.9587440640060909, 0.28427068039649245}, {0.949406207621952, 0.31405071712846444}, {0.9487917872815314, 0.315902112033328}, {0.948173756621896, 0.3177523048752307}, {0.9469268737655705, 0.32144905621352093}, {0.9443898886171415, 0.3288278246707586}, {0.9391435857768649, 0.3435248539686357}, {0.9279661947381223, 0.37266438174750394}, {0.9272518068527987, 0.3744383616674714}, {0.9265340276786219, 0.37621097213349675}, {0.9250883059767947, 0.37975205877649215}, {0.9221562834443444, 0.38681751369879036}, {0.9161305265212557, 0.40088010473941793}, {0.903437167683602, 0.4287205197419773}, {0.9026156217568477, 0.4304474873436936}, {0.9017907746445101, 0.43217288064622295}, {0.9001311889421819, 0.4356189191179993}, {0.8967725368554739, 0.44249182720339336}, {0.8898980018754775, 0.4561595622784122}, {0.8755257046001983, 0.483171543640896}, {0.8745214755019604, 0.48498679248601617}, {0.8735134828092075, 0.48679995413981747}, {0.8714862240083427, 0.4904199846699567}, {0.8673867352274144, 0.49763465670261275}, {0.8590088663298325, 0.5119607090067908}, {0.8415457250036454, 0.5401858871981836}, {0.8404232920867025, 0.5419305214860565}, {0.8392972423203386, 0.5436728235193247}, {0.837034311639184, 0.5471504008394013}, {0.8324652623513429, 0.5540772391808834}, {0.8231555184021175, 0.5678159847381379}, {0.8038585055812217, 0.5948205637036476}, {0.8027053852481759, 0.5963757745696731}, {0.8015492560804802, 0.5979287500002227}, {0.7992279885867881, 0.6010279712788056}, {0.7945495390155037, 0.6071993330449651}, {0.7850499673340297, 0.6194324408592425}, {0.7654881692405278, 0.6434499691139827}, {0.7642409709280108, 0.6449308012143716}, {0.7629909079595985, 0.6464092158772086}, {0.7604822068085879, 0.6493587707327437}, {0.7554306355236773, 0.6552286279706443}, {0.7451918809556914, 0.6668501035148144}, {0.7241805768819538, 0.6896103915015496}, {0.7227320693714611, 0.6911283208652685}, {0.7212803801894602, 0.6926432076861418}, {0.718367482387902, 0.6956638270369296}, {0.7125037888545526, 0.701668262691072}, {0.7006262602688457, 0.7135284461194885}, {0.6762824225186148, 0.7366424403958503}, {0.6747353396377139, 0.7380597682057861}, {0.6731852863805193, 0.7394738468673373}, {0.6700762960453908, 0.7422922318589155}, {0.6638229719780838, 0.7478897391154555}, {0.6511764961948867, 0.7589263276519999}, {0.6253367726430245, 0.7803549966396104}, {0.6237279956775166, 0.7816414698620512}, {0.6221165721239248, 0.7829246264416374}, {0.6188858126139098, 0.785480967907702}, {0.612392836138119, 0.7905536125062682}, {0.599282599913912, 0.8005375478017394}, {0.5725775132198905, 0.8198505908730725}, {0.5708874913650509, 0.8210283017064143}, {0.569195047133723, 0.8222025287716153}, {0.5658029202772095, 0.8245405116825866}, {0.5589899147101889, 0.8291744540519177}, {0.5452505184320892, 0.8382731488897506}, {0.5173312969104025, 0.8557852120929649}, {0.5156859905372025, 0.8567776602851315}, {0.5140387802583, 0.8577669452657635}, {0.5107386723163052, 0.8597360109944086}, {0.5041158776171086, 0.8636360239906232}, {0.4907813445364621, 0.8712827737623317}, {0.4637675672833218, 0.8859569083968529}, {0.46206438628074464, 0.8868463806832606}, {0.4603594993427026, 0.8877325787448246}, {0.4569446328441761, 0.8894951391181972}, {0.45009470599512547, 0.8929808260176484}, {0.4363154951800072, 0.8997937478477083}, {0.40845107247678125, 0.9127802152723115}, {0.4065483672435676, 0.9136292601989}, {0.4046438971170178, 0.9144743389105855}, {0.40082969526200807, 0.9161525830319782}, {0.39318047711606735, 0.919461316431954}, {0.3778006451329392, 0.925886965313334}, {0.3467303922651678, 0.9379648368034075}, {0.3447753477094177, 0.9386852292498535}, {0.3428188064271705, 0.9394015467093133}, {0.33890126766431544, 0.9408219442463701}, {0.3310486035728087, 0.9436137038388608}, {0.3152748402443372, 0.9490004083818445}, {0.28346894359945013, 0.9589814169287181}, {0.2816030904676497, 0.9595309788845114}, {0.2797361719113241, 0.960076910525609}, {0.2759991667824956, 0.9611578746154912}, {0.2685126824846626, 0.963276149058457}, {0.2534914051936521, 0.9673376388278023}, {0.25160935506492815, 0.9678288755993029}, {0.24972635299076465, 0.9683164506616274}, {0.24595752150649225, 0.969280608293792}, {0.2384087488954436, 0.9711649028100785}, {0.22326836562103494, 0.9747570142927476}, {0.22137194122189063, 0.9751894501273338}, {0.21947467927833209, 0.9756181964045532}, {0.2156776714737173, 0.9764646138123364}, {0.2080739213791149, 0.9781131035017975}, {0.1928290952683961, 0.981232357812344}, {0.19092013324651744, 0.9816055739048817}, {0.18901044889239427, 0.9819750761651213}, {0.1851889420907205, 0.9827029336108241}, {0.1775375785867782, 0.9841140219453961}, {0.16220307393755762, 0.9867573981507345}, {0.16032119296226943, 0.9870648991262706}, {0.15843872905385625, 0.9873688111014034}, {0.1546720798185521, 0.9879658636433768}, {0.14713208745622053, 0.9891168529758125}, {0.14524573478310884, 0.9893956117384566}, {0.1433588539917332, 0.9896707730261498}, {0.13958353549912825, 0.9902102991877855}, {0.13202686309380596, 0.9912461386666832}, {0.13013647768026987, 0.9914960903488085}, {0.12824561908622525, 0.9917424369185736}, {0.12446250985920221, 0.9922243111512377}, {0.11689091586110689, 0.9931447597350306}, {0.1017277679181786, 0.9948122743685791}, {0.09983063754732988, 0.9950044441141425}, {0.09793314418900288, 0.995192995990856}, {0.09413709610860219, 0.9955592434085677}, {0.0865409477547274, 0.9962482945339046}, {0.08464110673655259, 0.9964115028693775}, {0.08274095796053281, 0.9965710882198888}, {0.0789397647720659, 0.9968793876581714}, {0.07133398936902466, 0.9974524860667299}, {0.06943187979968378, 0.9975866950132616}, {0.06752951777374677, 0.9977172767017946}, {0.06372406402118042, 0.997967556418857}, {0.06182098613130594, 0.9980872535373613}, {0.05991768345829476, 0.9982033215778194}, {0.056110431445538865, 0.9984245687496855}, {0.05420649594908755, 0.9985297470766314}, {0.05230236335603579, 0.9986312947166102}, {0.048493534574777716, 0.9988234964719467}, {0.046588852235598, 0.9989141498884524}, {0.04468400049782087, 0.9990011712202898}, {0.040873816531479146, 0.999164316377517}, {0.03880652880980655, 0.9992467429627848}, {0.03673907497784812, 0.9993248922996825}, {0.03467146388528346, 0.9993987640536942}, {0.032603704382465126, 0.9994683579086138}, {0.030535805320381366, 0.9995336735665467}, {0.02846777555061782, 0.9995947107479108}, {0.024331359297151894, 0.9997039486541768}, {0.022262990519266224, 0.9997521489114884}, {0.02019452644525817, 0.9997960697570539}, {0.018125975929131663, 0.9998357110028709}, {0.016057347825260422, 0.9998710724792566}, {0.013988650988350724, 0.9999021540348466}, {0.011919894273403056, 0.9999289555365976}, {0.009851086535673768, 0.9999514768697862}, {0.0077822366306378316, 0.9999697179380107}, {0.005713353413950717, 0.9999836786631906}, {0.0036444457414104812, 0.9999933589855674}, {0.0015755224689198667, 0.9999987588637047}, {-0.0004934075475518332, 0.9999998782744886}, {-0.0025623354520060123, 0.9999967172131273}, {-0.004631252388453105, 0.9999892756931519}, {-0.00670014950095094, 0.9999775537464153}, {-0.008769017933641977, 0.9999615514230932}, {-0.01083784883079144, 0.999941268791683}, {-0.012906633336825223, 0.9999167059390041}, {-0.014975362596367796, 0.9998878629701969}, {-0.017044027754280332, 0.9998547400087231}, {-0.019112619955697945, 0.9998173371963646}, {-0.021181130346068035, 0.999775654693223}, {-0.023249550071188643, 0.9997296926777194}, {-0.02531787027724567, 0.9996794513465926}, {-0.02738608211085101, 0.9996249309148996}, {-0.029454176719080447, 0.9995661316160133}, {-0.03152214524951153, 0.9995030537016226}, {-0.033589978850261715, 0.9994356974417309}, {-0.03565766867002558, 0.9993640631246546}, {-0.03772520585811314, 0.9992881510570227}, {-0.041859786939805556, 0.9991234949881591}, {-0.04392681313544911, 0.9990347516917334}, {-0.04599365130356973, 0.9989417320543604}, {-0.05012672816990795, 0.9987428653677483}, {-0.05839023636094281, 0.9982938346487538}, {-0.06045551066900542, 0.9981708927982972}, {-0.06252052619890709, 0.9980436783046179}, {-0.06664974556840327, 0.9977764335840305}, {-0.07490469012166165, 0.9971906976089266}, {-0.07696764677244343, 0.9970335908836333}, {-0.07903027396531702, 0.996872216383307}, {-0.0831545046627104, 0.9965366668388571}, {-0.0913986241848828, 0.9958143860665554}, {-0.09332086003482103, 0.9956360866714109}, {-0.09524274809831948, 0.9954540767582802}, {-0.09908545221749475, 0.99507892810513}, {-0.106766371960605, 0.9942841353548642}, {-0.10868562504618864, 0.9940761715826002}, {-0.11060447308421326, 0.9938645031058125}, {-0.11444092541461309, 0.9934300552078374}, {-0.12210865495952976, 0.992516738591332}, {-0.12402446752097412, 0.9922791600432506}, {-0.1259398178703476, 0.9920378834877143}, {-0.12976910338226325, 0.9915442399647952}, {-0.1374218139038816, 0.9905126173165928}, {-0.15270219646224348, 0.9882722495322868}, {-0.15460975808742805, 0.9879756184765629}, {-0.15651674351585163, 0.9876753054517426}, {-0.16032895735691888, 0.9870636379853344}, {-0.16794615812067265, 0.9857961695870532}, {-0.1831500630519511, 0.9830849680490829}, {-0.2134232080586018, 0.976959842706943}, {-0.2154671055114055, 0.976511098985842}, {-0.21751005945794577, 0.9760580792322765}, {-0.2215931010529431, 0.9751392195813581}, {-0.22974746883130207, 0.9732502764271942}, {-0.24600734462843146, 0.9692679641816644}, {-0.24803507488386628, 0.9687510524522566}, {-0.25006171902155633, 0.9682298986707569}, {-0.2541117134507089, 0.9671748740983426}, {-0.26219827874383206, 0.9650140219829823}, {-0.2783157337917263, 0.960489641965999}, {-0.28032502398529735, 0.9599051416299647}, {-0.2823330866670619, 0.9593164379770887}, {-0.28634549432826134, 0.9581264310506749}, {-0.294355192978285, 0.9556960920537012}, {-0.31031216136742384, 0.9506347156018856}, {-0.3419607628493819, 0.939714231387111}, {-0.3438905088339884, 0.9390097539077539}, {-0.3458188035290893, 0.9383013136118424}, {-0.3496710065057145, 0.9368725565461294}, {-0.3573576393360665, 0.9339676212846748}, {-0.3726579980509388, 0.9279687583580961}, {-0.402952659617261, 0.915220822592764}, {-0.40483195976420844, 0.9143911003249486}, {-0.40670955143604653, 0.9135575191364197}, {-0.41045957766632857, 0.9118787940850358}, {-0.4179387802643575, 0.9084751928099859}, {-0.43281202094084537, 0.901484195385089}, {-0.4622038256587628, 0.8867737160890619}, {-0.4639019211086217, 0.8858865658715737}, {-0.46559831377694527, 0.8849961639510452}, {-0.4689859658683882, 0.883205618085899}, {-0.47574056311885377, 0.8795856505215145}, {-0.4891655436269747, 0.8721909601274977}, {-0.5156676460493473, 0.8567887013832086}, {-0.5173081959688094, 0.8557991764330555}, {-0.5189468470756032, 0.8548065102175407}, {-0.522218428799087, 0.8528117685765205}, {-0.5287385423137027, 0.8487847512013755}, {-0.5416852317887715, 0.8405814116800019}, {-0.5671938549556563, 0.8235843192415346}, {-0.5689040075432809, 0.8224039337218632}, {-0.5706117036421728, 0.8212199971180543}, {-0.5740196968895359, 0.8188414911219664}, {-0.5808058817905724, 0.8140420920796883}, {-0.594257407930225, 0.8042749114078159}, {-0.6206641346510916, 0.7840765472565874}, {-0.6222920761387304, 0.7827851378091875}, {-0.6239173306115948, 0.7814903483489107}, {-0.6271597504536807, 0.7788906517675488}, {-0.633612037961167, 0.7736509454209286}, {-0.6463848480502397, 0.7630115518201993}, {-0.67139111930272, 0.7411032080091415}, {-0.6728272406197728, 0.7397996379290696}, {-0.674260830936091, 0.7384932849155584}, {-0.6771203970048929, 0.7358722497553065}, {-0.6828089204911824, 0.7305970011556756}, {-0.6940623377765104, 0.7199149055827399}, {-0.7160652239785724, 0.6980333767152663}, {-0.7174177264922167, 0.6966432413472043}, {-0.7187675302673393, 0.6952504853888198}, {-0.721459021301751, 0.6924571326676617}, {-0.7268093957760018, 0.6868392113236717}, {-0.7373785907069512, 0.6754796917502632}, {-0.757981914717511, 0.6522755682694056}, {-0.7593510703835291, 0.6506811445765035}, {-0.7607168721673688, 0.6490838469719417}, {-0.7634383899735999, 0.6458806582601136}, {-0.7688409145684109, 0.6394401051588879}, {-0.7794825884714144, 0.6264238934379049}, {-0.800101988515113, 0.5998639912298136}, {-0.8013609024109107, 0.5981811632667571}, {-0.8026162768765897, 0.5964956932726015}, {-0.8051163853544266, 0.5931168569799905}, {-0.8100738862320412, 0.5863278083503441}, {-0.8198168028010416, 0.5726258899535349}, {-0.838604741995495, 0.5447403846812436}, {-0.8397269450097521, 0.5430088929516612}, {-0.840845572967814, 0.5412750894142724}, {-0.8430720846807915, 0.5378005764518892}, {-0.8474819989068509, 0.5308241342750424}, {-0.8561283443339204, 0.5167632514295692}, {-0.872718940893068, 0.48822295133113275}, {-0.8737244562413927, 0.4864211904981964}, {-0.8747262517924085, 0.4846173587741198}, {-0.8767186664581779, 0.481003513379886}, {-0.8806586716232009, 0.47375131038854057}, {-0.8883584512968195, 0.4591505875086261}, {-0.9030298316595368, 0.4295778429259925}, {-0.9038550546576457, 0.4278388016178806}, {-0.9046769286398826, 0.4260981750567688}, {-0.9063106173881411, 0.4226121919792737}, {-0.9095376735357649, 0.4156214869555572}, {-0.9158298350354837, 0.4015665738814409}, {-0.9277612001156141, 0.37317443047459115}, {-0.9284778061586285, 0.37138788815854007}, {-0.9291909719521322, 0.3695999697546687}, {-0.9306069722335117, 0.36602003118733245}, {-0.933397574389019, 0.35884393282413973}, {-0.9388126103428196, 0.3444283418380393}, {-0.9489735837718971, 0.3153555728112668}, {-0.9496296627245983, 0.31337438260579925}, {-0.9502816055138119, 0.31139182748121674}, {-0.9515730712615169, 0.30742265702111804}, {-0.9541062450624436, 0.2994683174107812}, {-0.958972950099611, 0.2834975854875117}, {-0.9595625198053647, 0.2814955960308777}, {-0.9601479100845336, 0.2794923805049118}, {-0.9613061421825354, 0.2754823061507416}, {-0.9635723422937693, 0.26744782886854596}, {-0.967903139568502, 0.2513235214090332}, {-0.9684255427050285, 0.24930296475668132}, {-0.9689437278115933, 0.24728132225174818}, {-0.9699674349252864, 0.2432348149103257}, {-0.9719641347179835, 0.23512915774510307}, {-0.9757541957434864, 0.218869252040989}, {-0.9825173653353512, 0.18617096125464916}, {-0.9828782639955619, 0.18425612110611297}, {-0.9832354308039806, 0.18234058136276382}, {-0.9839385634551665, 0.17850743218640322}, {-0.9852999878170545, 0.17083305888414055}, {-0.9878431880799993, 0.15545364506483353}, {-0.9881442223247691, 0.15352848558550103}, {-0.988441504723637, 0.15160274317990532}, {-0.9890248094829577, 0.14774953883921024}, {-0.990146348141178, 0.14003645690565467}, {-0.9922089083459688, 0.1245852406944774}, {-0.9924497855947262, 0.12265163298131038}, {-0.992686894649958, 0.1207175595769864}, {-0.9931498045930757, 0.1168480450702349}, {-0.9940303675041287, 0.10910375098779576}, {-0.9942410747987538, 0.10716662345627442}, {-0.994448007098578, 0.10522908902801839}, {-0.9948505435853872, 0.10135282890901343}, {-0.9956102830115791, 0.09359574969838869}, {-0.9957907691514137, 0.09165557305933865}, {-0.995967474412474, 0.08971504841685106}, {-0.9963095396289281, 0.08583298459446356}, {-0.9969482709579569, 0.07806500518119648}, {-0.9970984920859093, 0.07612225088636024}, {-0.9972449273698454, 0.07417920756594643}, {-0.9975264381960681, 0.07029228335931248}, {-0.9980440062199017, 0.06251529131763409}, {-0.9981616073917627, 0.060608625863754546}, {-0.9982755660691032, 0.05870173923668468}, {-0.9984925542900828, 0.054887330298221425}, {-0.9985955830418879, 0.05297982190638129}, {-0.9986949677155073, 0.05107212018040612}, {-0.9988828033907885, 0.047256164573095234}, {-0.9989712537069997, 0.045347924616956886}, {-0.9990560585741266, 0.043439519177027874}, {-0.9992147307365504, 0.039622239703011854}, {-0.9992885974528206, 0.037713379598953274}, {-0.9993588175619544, 0.035804381871109604}, {-0.999488316947132, 0.03198600140982574}, {-0.9995475957506061, 0.030076632610431704}, {-0.9996032270018066, 0.028167154055294484}, {-0.9996552104977238, 0.02625757271248887}, {-0.9997035460486592, 0.02434789555046519}, {-0.9997482334782267, 0.022438129538023426}, {-0.9997892726233527, 0.020528281644287806}, {-0.9998266633342772, 0.018618358838681355}, {-0.9998604054745537, 0.016708368090900018}, {-0.9998904989210504, 0.014798316370888565}, {-0.9999169435639501, 0.012888210648814269}, {-0.999939739306751, 0.01097805789504058}, {-0.9999588860662667, 0.009067865080103012}, {-0.9999743837726269, 0.007157639174683281}, {-0.9999862323692774, 0.005247387149583851}, {-0.99999443181298, 0.0033371159757025017}, {-0.9999989820738134, 0.0014268326240064497}, {-0.9999998831351729, -0.0004834559344917636}, {-0.9999971349937701, -0.002393742728760595}, {-0.9999907376596338, -0.0043040207877758285}, {-0.9999806911561089, -0.0062142831405446785}, {-0.9999669955198572, -0.008124522816131676}, {-0.9999496508008571, -0.010034732843684108}, {-0.9999286570624026, -0.01194490625245745}, {-0.9999040143811044, -0.013855036071841251}, {-0.9998757228468887, -0.01576511533138324}, {-0.9998437825629969, -0.017675137060815648}, {-0.9998081936459856, -0.019585094290081536}, {-0.999768956225726, -0.02149498004935889}, {-0.9997260704454035, -0.023404787369086508}, {-0.999679536461517, -0.025314509279989426}, {-0.9995755245756119, -0.02913366899980558}, {-0.9995180470531541, -0.031043092871828985}, {-0.9994569220862517, -0.032952403461298456}, {-0.999323730724653, -0.03677065692316656}, {-0.9992516648159989, -0.0386795858619825}, {-0.9991759524349834, -0.04058837365113037}, {-0.9990135893743329, -0.04440549791874492}, {-0.9989269392871939, -0.04631382046774955}, {-0.9988366439126827, -0.04822197400821242}, {-0.9986451186328701, -0.05203774621119637}, {-0.9982183400141599, -0.05966695617655674}, {-0.9980925438805436, -0.06173551530573861}, {-0.9979624611774583, -0.06380380929560425}, {-0.9976894383159852, -0.06793956632724359}, {-0.9970919795742298, -0.0762075079552116}, {-0.9969319075583752, -0.07827369731920913}, {-0.9967675539577079, -0.08033955051634153}, {-0.9964260048437585, -0.08447021292210712}, {-0.9956915598875682, -0.09272711344941838}, {-0.995497255986636, -0.09479036513843714}, {-0.9952986766623775, -0.09685320972544109}, {-0.994888695173643, -0.10097764215749036}, {-0.9940174658714866, -0.10922123210451287}, {-0.9920701576650242, -0.12568533037110632}, {-0.9918075596064438, -0.12774100636643834}, {-0.9915407019709032, -0.1297961337446098}, {-0.9909942125715898, -0.1339047073467349}, {-0.989850170502473, -0.142114882951171}, {-0.9873581145662658, -0.1585053740418562}, {-0.9870275110394756, -0.1605512144183837}, {-0.986692668464894, -0.16259636526573015}, {-0.9860102719428318, -0.16668456324214007}, {-0.9845946748192611, -0.17485229857669427}, {-0.981560610816476, -0.19115116346385758}, {-0.9811891009812859, -0.19304908214113867}, {-0.9808139213861553, -0.19494627879267873}, {-0.9800525585426527, -0.19873847763832792}, {-0.9784858583967625, -0.20631389899265334}, {-0.9751768930078618, -0.22142725067961533}, {-0.9747468430727753, -0.22331276703238118}, {-0.9743131474725328, -0.22519744816974832}, {-0.9734348257784957, -0.22896427660573967}, {-0.971634506692502, -0.23648760095281585}, {-0.9678595473656629, -0.25149134492688613}, {-0.9673713690576109, -0.25336265377439443}, {-0.966879572669523, -0.255233015016476}, {-0.9658851330242653, -0.25897086670646435}, {-0.9638529186116171, -0.2664348912658912}, {-0.9596155794676454, -0.28131466304296177}, {-0.9504528820335238, -0.3108686523825912}, {-0.9497992046848729, -0.3128601457200691}, {-0.9491413545455842, -0.3148502645566199}, {-0.9478131474740497, -0.31882634375994623}, {-0.9451067882370389, -0.32676162386098656}, {-0.9394949613532749, -0.34256272066850546}, {-0.9274804940436945, -0.3738715463477588}, {-0.9266948113041101, -0.37581475051152485}, {-0.9259050572793387, -0.377756303593394}, {-0.924313349270737, -0.3816344223991234}, {-0.9210812316927475, -0.3893704722027228}, {-0.9144229914544563, -0.40475991982838844}, {-0.9003371379222682, -0.4351931043547664}, {-0.8994397071396347, -0.43704486408213084}, {-0.8985384677730155, -0.43889477318842796}, {-0.8967245785687481, -0.4425890082126997}, {-0.8930512661438983, -0.4499549266746394}, {-0.8855233816467264, -0.4645948133125748}, {-0.8697497996135031, -0.49349294429836704}, {-0.8687324662935634, -0.49528163907770967}, {-0.867711454416259, -0.49706823663841226}, {-0.8656584122985875, -0.5006351098521643}, {-0.8615083766444536, -0.5077433573878031}, {-0.8530334889322179, -0.5218561744006941}, {-0.8353925685434854, -0.5496537604913825}, {-0.8343360678532378, -0.5512561345501721}, {-0.8332764936388429, -0.5528564778936351}, {-0.8311481402619422, -0.5560510488607273}, {-0.8268547237184403, -0.5624155633199546}, {-0.8181219409419928, -0.5750447719519817}, {-0.8000798423155425, -0.5998935288201871}, {-0.7989269804274569, -0.6014280338869031}, {-0.7977711754550907, -0.6029603234152334}, {-0.7954507532993428, -0.6060182332863493}, {-0.7907747800419476, -0.6121072187530625}, {-0.7812832859669485, -0.6241765992639321}, {-0.7617499966360892, -0.6478710848810264}, {-0.7603998595462573, -0.6494551975325412}, {-0.7590464281889996, -0.6510364965603095}, {-0.7563297061402441, -0.6541906263542853}, {-0.7508569897783979, -0.660464821849675}, {-0.7397558004549559, -0.67287543846781}, {-0.7169412429326064, -0.6971335985177084}, {-0.7154886672888723, -0.6986243389556317}, {-0.7140329919457096, -0.7001120527551702}, {-0.7111123674000902, -0.7030783746714439}, {-0.7052342002935654, -0.708974416136644}, {-0.6933316245546836, -0.7206186636442073}, {-0.668953127499395, -0.7433045897939674}, {-0.6675076464264518, -0.7446029424882895}, {-0.6660596454218619, -0.7458984842057956}, {-0.6631561054928344, -0.7484811151576081}, {-0.6573190276114416, -0.7536124308555088}, {-0.6455261169042014, -0.7637381962391192}, {-0.6214752975641329, -0.7834337588574879}, {-0.6199519363447553, -0.7846397878150129}, {-0.618426234722969, -0.7858428546512154}, {-0.6153678333199036, -0.7882400838038923}, {-0.609223199702275, -0.7929987975681438}, {-0.5968239111591668, -0.802372244702342}, {-0.5715957797315255, -0.8205353524334642}, {-0.5700318649622106, -0.8216225854537497}, {-0.5684658821714164, -0.8227068377053113}, {-0.5653277352578193, -0.8248663841788347}, {-0.5590268756474315, -0.8291495355506575}, {-0.546328175864445, -0.8375712054844221}, {-0.520553212266078, -0.8538292295298092}, {-0.5189259750659755, -0.8548191811148288}, {-0.5172968552517095, -0.8558060314970278}, {-0.5140329914286638, -0.8577704143434298}, {-0.5074829329204249, -0.8616618088290112}, {-0.49429482542585124, -0.8692943261963851}, {-0.467577400774867, -0.8839521334804388}, {-0.4657494526722557, -0.8849166329861217}, {-0.463919515049335, -0.8858773524345173}, {-0.46025370251847103, -0.8877874347601675}, {-0.452898547720154, -0.8915620592381528}, {-0.43809588139767625, -0.8989282500302197}, {-0.408135328684896, -0.9129214388321001}, {-0.4062476338727381, -0.9137630217801559}, {-0.40435820371142855, -0.9146007014491443}, {-0.4005741696326941, -0.9162643366534996}, {-0.392985632901418, -0.9195446113881979}, {-0.3777285028246765, -0.9259163991170198}, {-0.3469086519798865, -0.937898921622953}, {-0.3450994058283436, -0.9385661405020556}, {-0.34328887640648476, -0.9392298692735306}, {-0.33966399468675795, -0.9405468466341451}, {-0.3323991285522974, -0.9431388123376502}, {-0.31781049987437504, -0.9481542523079247}, {-0.2884098321333906, -0.9575070593623786}, {-0.2865628833161549, -0.9580614353503285}, {-0.2847148688997454, -0.9586122487363703}, {-0.28101567076109313, -0.9597031795230716}, {-0.27360478992134835, -0.9618422006400503}, {-0.2587346344958829, -0.9659484400899884}, {-0.22881315401874486, -0.9734703593576921}, {-0.22677766719392967, -0.9739465538016341}, {-0.22474118935801873, -0.9744184921305338}, {-0.22066529625457573, -0.9753495922123925}, {-0.21250200977099007, -0.9771606294992088}, {-0.1961314347630779, -0.9805776156418097}, {-0.1940811620143555, -0.9809854751988724}, {-0.19203004113721267, -0.981389047881033}, {-0.1879252908546578, -0.9821833255849909}, {-0.1797060073786087, -0.9837203621517852}, {-0.163230321341833, -0.9865879900924407}, {-0.16116755612042719, -0.9869270585277156}, {-0.15910408660199762, -0.9872618141235606}, {-0.15497507074628333, -0.9879183809643307}, {-0.14670898443291414, -0.9891796974699102}, {-0.14464083748696765, -0.9894842232855805}, {-0.14257205846522336, -0.9897844250870939}, {-0.13843264035905883, -0.9903718514190614}, {-0.13014661719926426, -0.9914947594574508}, {-0.12807366693057703, -0.991764657486319}, {-0.12600015698409409, -0.992030221535606}, {-0.12185149430483287, -0.992548343072356}, {-0.11354785166330976, -0.9935325285981567}, {-0.09691733002704285, -0.9952924349860343}, {-0.09497433169515279, -0.9954797217015819}, {-0.09303097148077906, -0.9956632153219995}, {-0.08914319502521573, -0.9960188204952235}, {-0.0813636254253578, -0.996684483905334}, {-0.07941793944226978, -0.9968414070927952}, {-0.07747195085151243, -0.9969945319966709}, {-0.07357909550745885, -0.9972893846343218}, {-0.06579007985391357, -0.9978334858045282}, {-0.06384218068590843, -0.9979600071973164}, {-0.06189403825884086, -0.9980827260442959}, {-0.057997053320588604, -0.9983167542449284}, {-0.05604822565815902, -0.9984280627068592}, {-0.054099184434120616, -0.9985355668395408}, {-0.050200491007873195, -0.9987391604931533}, {-0.04825085366092914, -0.9988352492383273}, {-0.04630103246284964, -0.9989275321027418}, {-0.04240086823171466, -0.9991006787972856}, {-0.040450540059528396, -0.999181541967671}, {-0.03850005775788849, -0.9992585979378107}, {-0.03459866049464092, -0.9994012871174304}, {-0.03264776039860072, -0.999466919783219}, {-0.030696735904185157, -0.9995287441613816}, {-0.028745594445418708, -0.9995867600163477}, {-0.026794343456771536, -0.9996409671270584}, {-0.02484299037313291, -0.9996913652869671}, {-0.022891542629781117, -0.9997379543040411}, {-0.020940007662353363, -0.9997807340007611}, {-0.01898839290681919, -0.9998197042141229}, {-0.017036705799453943, -0.9998548647956377}, {-0.015084953776808652, -0.9998862156113322}, {-0.013133144275679909, -0.9999137565417501}, {-0.011181284733083328, -0.9999374874819514}, {-0.00922938258622697, -0.9999574083415138}, {-0.007277445272481225, -0.9999735190445325}, {-0.005325480229348714, -0.9999858195296205}, {-0.00337349489443771, -0.9999943097499091}, {-0.0014214967054355814, -0.9999989896730478}, {0.0005305068999213248, -0.9999998592812047}, {0.0024825084838778, -0.9999969185710661}, {0.0044345006086863375, -0.9999901675538373}, {0.006386475836633697, -0.9999796062552416}, {0.00833842673007102, -0.9999652347155211}, {0.010290345851443951, -0.9999470529894359}, {0.012242225763319186, -0.9999250611462641}, {0.014194059028411057, -0.9998992592698016}, {0.016145838209611626, -0.9998696474583619}, {0.018097555870020817, -0.9998362258247755}, {0.02004920457297296, -0.9997989944963893}, {0.022000776882063356, -0.9997579536150666}, {0.023952265361178404, -0.9997131033371863}, {0.027854961086660548, -0.9996119752898425}, {0.0297679057219342, -0.999556837698052}, {0.03168074133530848, -0.9994980393319665}, {0.035506057474532116, -0.999369461151688}, {0.03741852399055296, -0.9992996818083995}, {0.03933085346507139, -0.9992262426326227}, {0.04315507327528709, -0.9990683858728614}, {0.04506694960517192, -0.9989839688670108}, {0.046978660881980785, -0.9988958931849384}, {0.05080156027121469, -0.9987087670957986}, {0.058445070392789976, -0.9982906258934728}, {0.0603554302946532, -0.9981769492599732}, {0.06226556915094957, -0.9980596169058782}, {0.06608515574492621, -0.9978139867681596}, {0.07372136863322164, -0.9972788776500005}, {0.07562976433986439, -0.9971359680333954}, {0.07753788306026002, -0.9969894065086817}, {0.08135326159019347, -0.9966853298953675}, {0.08898038740149865, -0.9960333782850248}, {0.09088637160610766, -0.9958612691817452}, {0.0927920229487118, -0.9956855128388108}, {0.09660229913213973, -0.9953230610220908}, {0.10421855015265444, -0.9945544197297997}, {0.10612167611364579, -0.9943531514803131}, {0.10802441341489372, -0.9941482415144978}, {0.11182869416528818, -0.9937274994490625}, {0.1194322852301427, -0.9928423486358274}, {0.12133210687587313, -0.9926119684151817}, {0.12323148415521823, -0.9923779528551117}, {0.1270288777913126, -0.9918990191582406}, {0.13461802670299144, -0.9908975662936067}, {0.14977221520161957, -0.9887205285385753}, {0.1518243081010024, -0.9884074966680756}, {0.15387574677817237, -0.9880902056763137}, {0.15797662610952845, -0.9874428518162711}, {0.16617014534799293, -0.9860970960280874}, {0.18252225000375322, -0.9832017230729243}, {0.1845628162408335, -0.9828207195929746}, {0.18660258718303957, -0.9824354810655996}, {0.19067970802819978, -0.9816523055269012}, {0.19882401967523147, -0.9800352081431478}, {0.2150709587498645, -0.9765984244828649}, {0.217097747535204, -0.976149869648683}, {0.21912360083023558, -0.9756971085122634}, {0.22317246603523833, -0.9747789751547529}, {0.2312585867359702, -0.9728923198694097}, {0.24738243949835276, -0.9689179163519703}, {0.2794210519753168, -0.9601686704496285}, {0.2812807459472695, -0.9596255217319658}, {0.2831393841397613, -0.9590787710865863}, {0.28685346528496863, -0.957974472235041}, {0.29426865153133325, -0.9557227426016035}, {0.3090455626113735, -0.9510472334380767}, {0.33837313848704775, -0.9410120186004136}, {0.34019560797180615, -0.9403546928243051}, {0.3420168005418087, -0.9396938374529997}, {0.3456553275991788, -0.9383615478600476}, {0.352916758286105, -0.935654723560367}, {0.36737560130836544, -0.9300726678939216}, {0.3960250851539529, -0.9182396919806964}, {0.3979520370247532, -0.9174062220346282}, {0.3998772347963314, -0.9165687083310446}, {0.40372233410592323, -0.9148815644355641}, {0.41139111067672435, -0.9114589151772948}, {0.4266410853694596, -0.9044210215794243}, {0.4567757085299878, -0.8895818973523054}, {0.45864236008264025, -0.8886209459256662}, {0.46050699002431406, -0.8876560776217027}, {0.4642301522078926, -0.8857146074108956}, {0.47165185621120587, -0.8817848527461356}, {0.48639495388725645, -0.8737390622108031}, {0.5154654021912465, -0.8569103915485076}, {0.5172305829346528, -0.8558460866750979}, {0.5189935661681087, -0.8547781456472252}, {0.5225129101536348, -0.8526313732925733}, {0.5295248990648398, -0.8482943954019567}, {0.5434404155544523, -0.8394477439019085}, {0.570824412774256, -0.8210721586930869}, {0.5725156056487928, -0.8198938231799261}, {0.5742043661289922, -0.8187120042593747}, {0.5775745612172214, -0.8163379362952172}, {0.5842854476630115, -0.8115482213949053}, {0.5975876125094739, -0.8018036202058625}, {0.6237009085643952, -0.781663083851315}, {0.6252027149645676, -0.7804624047325621}, {0.6267022099572866, -0.7792588402030823}, {0.6296942435540789, -0.7768430727211618}, {0.6356503305099396, -0.7719771093255322}, {0.647449353272906, -0.7621084797760066}, {0.6705850705816354, -0.7418326382096052}, {0.6720102054782721, -0.740541885198299}, {0.673432855917691, -0.7392483943645346}, {0.6762706823956224, -0.736653218368154}, {0.6819162910596531, -0.7314302235944632}, {0.6930861655148038, -0.7208547476239482}, {0.7149313436849485, -0.6991946608898226}, {0.7163875149180309, -0.6977026074693774}, {0.7178405722595412, -0.6962075213742627}, {0.7207373200176317, -0.6932082771676936}, {0.726493171843649, -0.6871736834778776}, {0.7378529043001282, -0.6749615482498732}, {0.7599534990134456, -0.6499774452526956}, {0.7613069612512419, -0.6483916337757607}, {0.7626571143479915, -0.6468030039620986}, {0.7653474696581459, -0.6436173169577349}, {0.770688213331967, -0.6372124275555056}, {0.7812085377818643, -0.6242701502528546}, {0.8015943713631442, -0.5978682662584842}, {0.8027565557510561, -0.5963068943075382}, {0.803915698851669, -0.594743263215173}, {0.8062248436366258, -0.5916092473103326}, {0.8108064453079671, -0.5853143670260099}, {0.8198219289391232, -0.5726185508962622}, {0.8372544297825208, -0.5468135146551756}, {0.8383171726240044, -0.5451828299604593}, {0.8393767394548391, -0.5435500798106492}, {0.8414863290396989, -0.5402784078966062}, {0.8456672202636742, -0.5337105513023986}, {0.8538749679938094, -0.5204781830524418}, {0.8696674251083102, -0.49363809588146834}, {0.8706077800920344, -0.4919777365320709}, {0.871544965140016, -0.49031558586188984}, {0.8734098117909169, -0.4869859347736392}, {0.8771013164953485, -0.4803054034696326}, {0.884330822726824, -0.46686078864614317}, {0.8981699891648729, -0.4396483487556527}, {0.8990072714267694, -0.43793370037232243}, {0.899841280348653, -0.43621745744582124}, {0.9014994660376017, -0.4327802129648709}, {0.9047764245214415, -0.4258868648246812}, {0.9111720257739014, -0.41202613927648424}, {0.9233248141433473, -0.3840199051965317}, {0.9241178515933844, -0.38210757172088}, {0.924906928341106, -0.38019360055979406}, {0.9264731862188865, -0.3763607780008758}, {0.9295580258810365, -0.3686758420620344}, {0.9355362661713228, -0.35323065365030226}, {0.9467213436337172, -0.3220538736118061}, {0.9473860459475743, -0.32009323632938674}, {0.9480466878333165, -0.31813122715049225}, {0.9493557790120163, -0.3142031267452438}, {0.951925112815113, -0.30633083356386814}, {0.9568677754092104, -0.29052376905070765}, {0.965964429872143, -0.25867493156814725}, {0.9664623367994261, -0.2568083946217348}, {0.9669566370134772, -0.25494089929939945}, {0.9679344099366913, -0.25120306140751775}, {0.9698465949617334, -0.24371619199620553}, {0.9734971266473279, -0.2286992444443058}, {0.9739371123566737, -0.22681821173429595}, {0.9743734634578862, -0.22493633256786957}, {0.9752352553358558, -0.22117006296062822}, {0.9769151530538428, -0.21362767548655043}, {0.9800998578366155, -0.1985050847425478}, {0.9804815020170066, -0.1966113531881481}, {0.9808594871664432, -0.19471688790544578}, {0.9816044747437449, -0.19092578443745253}, {0.9830504803763755, -0.18333508401770046}, {0.9857663159609072, -0.1681212964405782}, {0.9860892540340166, -0.16621667509199156}, {0.9864085121487617, -0.16431143344413468}, {0.987035983751175, -0.16049911769336989}, {0.98824671595336, -0.15286735559758693}, {0.9904910879471646, -0.13757690466514436}, {0.9907854662971457, -0.1354406134597285}, {0.9910752370996552, -0.13330369240145704}, {0.9916409506934939, -0.12902800047936752}, {0.9927170294556908, -0.12046949584301006}, {0.9929745108951917, -0.11832844422392577}, {0.993227374607291, -0.11618684233039418}, {0.993719244167096, -0.11190202755971618}, {0.9946475202569744, -0.10332623310007784}, {0.9948680281548192, -0.101181058283365}, {0.9950839095196523, -0.09903541293438661}, {0.9955017886560645, -0.09474275055420407}, {0.9962819858017171, -0.08615221858424324}, {0.9964654545307307, -0.08400355901308147}, {0.9966442892980629, -0.0818545087918749}, {0.9969880536426243, -0.07755527637686274}, {0.99761993964314, -0.06895256359568983}, {0.99776631459968, -0.06680105875038529}, {0.9979080495450204, -0.0646492432535213}, {0.9981775967871878, -0.0603447203337167}, {0.9983054078305124, -0.05819203292852401}, {0.998428576355639, -0.05603907490717458}, {0.9986609835817595, -0.05173238706567387}, {0.998770221201967, -0.049578677273338256}, {0.9988748141424091, -0.04742473692038406}, {0.9990700640600031, -0.043116204600375005}, {0.9991607201291649, -0.04096163266971336}, {0.9992467297025857, -0.038806870251127286}, {0.999404807783898, -0.034496814033037816}, {0.9994768755566635, -0.032341540277014386}, {0.999544295363439, -0.030186116119934093}, {0.9996070668906956, -0.028030551585377816}, {0.9996651898465204, -0.025874856697580124}, {0.9997186639606187, -0.023719041481381774}, {0.9997674889843146, -0.021563115962182224}, {0.999811664690552, -0.019407090165895643}, {0.9998511908738965, -0.017250974118902543}, {0.9998860673505354, -0.015094777848001359}, {0.9999162939582794, -0.012938511380363591}, {0.9999418705565626, -0.010782184743488969}, {0.9999627970264435, -0.008625807965157026}, {0.9999790732706059, -0.0064693910733786995}, {0.9999906992133586, -0.004312944096351472}, {0.9999976748006364, -0.0021564770624145075}, {1., -2.4492935982947064*^-16}}]}, "Charting`Private`Tag$24676#1"], Annotation[{Hue[0.9060679774997897, 0.6, 0.6], Directive[RGBColor[0.95, 0.627, 0.1425], AbsoluteThickness[2]], Line[{{1., 0.}, {1.0077043700451922, 0.0019418863158300389}, {1.0153934955103079, 0.003913421663970403}, {1.0306799928526125, 0.007944792526092332}, {1.0605667538596588, 0.016351308965004672}, {1.0678299441588937, 0.018521590308630525}, {1.0749877201357205, 0.02071797896171729}, {1.0889441764671324, 0.025185722027278187}, {1.1151311393675687, 0.03439328820360314}, {1.1212604738528027, 0.03674522877926693}, {1.1272039799639595, 0.039114642511271745}, {1.1384979297904605, 0.043900298110716676}, {1.1438313192400063, 0.04631360249907175}, {1.1489447974723634, 0.04873851174005461}, {1.1584814151215972, 0.05361671912404251}, {1.1628901080333411, 0.05606669761709347}, {1.1670500184328483, 0.05852164924960269}, {1.170954827056901, 0.06097982461514559}, {1.1745985938634227, 0.06343944563097072}, {1.1779757669347737, 0.06589870964031061}, {1.1810811907901226, 0.06835579358427549}, {1.1839101140943196, 0.07080885823484705}, {1.1864581967516354, 0.07325605248035323}, {1.1887215163736735, 0.07569551765468355}, {1.1906965741117166, 0.07812539190140054}, {1.1923802998447515, 0.0805438145638157}, {1.1937700567154084, 0.08294893059202768}, {1.194863645007038, 0.08533889495786774}, {1.1956593053561775, 0.08771187706866111}, {1.196155721295662, 0.09006606517069497}, {1.1963520211246772, 0.09239967073328251}, {1.19624777910307, 0.09471093280432862}, {1.1958430159682765, 0.09699812232833749}, {1.1951381987742669, 0.09925954641785267}, {1.1941342400529356, 0.10149355256938977}, {1.1928324962994181, 0.10369853281500725}, {1.191234765783837, 0.10587292780076404}, {1.1893432856930213, 0.10801523078343257}, {1.1871607286067625, 0.11012399153697275}, {1.1846901983141953, 0.11219782016042552}, {1.1819352249768909, 0.11423539077905392}, {1.1788997596462605, 0.11623544513074555}, {1.175588168143849, 0.11819679602989118}, {1.172005224314065, 0.12011833070117049}, {1.1681561026598708, 0.12199901397590818}, {1.1640463703728705, 0.12383789134390823}, {1.1596819787701893, 0.12563409185393565}, {1.1550692541514154, 0.12738683085628758}, {1.1502148880897731, 0.12909541258118298}, {1.1451259271725598, 0.1307592325469996}, {1.1398097622067054, 0.13237777979269824}, {1.1342741169061397, 0.13395063892909806}, {1.128527036078433, 0.13547749200400017}, {1.1164322783013865, 0.13839240519619758}, {1.109561561861226, 0.13989496369957613}, {1.102484569415945, 0.1413431532756103}, {1.09521338989551, 0.1427372332841781}, {1.0877604414006552, 0.14407760752876073}, {1.080138448978102, 0.14536482486650001}, {1.0723604218595446, 0.14659957947171562}, {1.0563895813958344, 0.1489152029029407}, {1.0482239959101631, 0.1499981841367811}, {1.039956782840845, 0.1510329255241765}, {1.023173904248739, 0.15296347806947075}, {1.0146867753419822, 0.15386253051500923}, {1.0061550412542362, 0.15471982096145007}, {0.9890156945330821, 0.15631706871772796}, {0.9804371158282046, 0.15706132068134865}, {0.9718719482060709, 0.15777239253363187}, {0.9548396450860502, 0.15910490166841398}, {0.921566609857483, 0.16148674997679488}, {0.9134948765441013, 0.1620400658962173}, {0.905549007935966, 0.16258265152562332}, {0.8900875320047691, 0.16364849335569054}, {0.8825975774373941, 0.16417833969575507}, {0.8752847479284007, 0.16471062212956278}, {0.8612382924601609, 0.1657962248301669}, {0.8545276820750212, 0.1663564972773882}, {0.8480401875743994, 0.1669330949379215}, {0.8417863286310375, 0.16752954678981244}, {0.8357761958839682, 0.16814938882785968}, {0.8300194333594866, 0.16879615606189383}, {0.8245252217033662, 0.16947337444502233}, {0.8193022622537638, 0.17018455274934666}, {0.814358761982733, 0.1709331744068677}, {0.8097024193326717, 0.17172268933347118}, {0.805340410972397, 0.17255650575401882}, {0.8012793794958668, 0.1734379820466702}, {0.7975254220848462, 0.17437041862461913}, {0.7940840801550674, 0.1753570498734478}, {0.7909603300036371, 0.1764010361622842}, {0.7881585744736317, 0.1775054559468891}, {0.785682635649973, 0.17867329798270218}, {0.7835357485988013, 0.17990745366574068}, {0.7817205561606804, 0.18121070951906712}, {0.7802391048060502, 0.18258573984233034}, {0.779092841559433, 0.18403509954162964}, {0.7782826119969511, 0.18556121715666124}, {0.7778086593197924, 0.18716638810177932}, {0.7776706245043012, 0.18885276813723573}, {0.777867547527446, 0.1906223670864631}, {0.7783978696644628, 0.1924770428148266}, {0.7792594368535548, 0.19441849548479775}, {0.780449504120602, 0.19644826210199662}, {0.7819647410549346, 0.19856771136600956}, {0.7838012383253313, 0.20077803883931625}, {0.7859545152235446, 0.20308026244705885}, {0.7884195282208084, 0.20547521831975038}, {0.791190680520975, 0.20796355699036098}, {0.7940491612748781, 0.2103717132699256}, {0.7971637867124539, 0.21286191451092945}, {0.8041378397763156, 0.21808860881063505}, {0.8208787123264935, 0.22951981639816393}, {0.8637554872241956, 0.25610626367107914}, {0.8698195051853322, 0.2597512477804841}, {0.8760069714809298, 0.2634612801784589}, {0.8887101033256238, 0.27106704284972255}, {0.9151256965937391, 0.2869476758923198}, {0.96948595264452, 0.3206928248727243}, {0.9762030700235488, 0.3250287531234206}, {0.9828652430154992, 0.3293781273409235}, {0.9959804709779763, 0.33810106278832824}, {1.0210715933682681, 0.35552768504546095}, {1.0270457501258563, 0.35985569590680816}, {1.0328801975439887, 0.3641636929498238}, {1.0440911823959769, 0.37270243017770777}, {1.0644241765686022, 0.38935064387831764}, {1.069021845672962, 0.3933984349524767}, {1.0734092144172165, 0.3973921354100465}, {1.0815235147769986, 0.4052008714680999}, {1.0852365416837586, 0.4090078627396532}, {1.088711479419225, 0.41274468967043276}, {1.0949232926710877, 0.41999259193512156}, {1.0976492290670643, 0.42349626378257876}, {1.1001152139272272, 0.42691497856244287}, {1.1023167705541348, 0.4302452921756629}, {1.1042498314851064, 0.4334838759607569}, {1.1059107442076952, 0.4366275233441709}, {1.1072962762196314, 0.43967315632632825}, {1.108403619426221, 0.4426178317909498}, {1.109230393869232, 0.4454587476254707}, {1.1097667015248194, 0.4481406479675202}, {1.110030033545567, 0.45071790732159345}, {1.110019360227054, 0.4531883120406192}, {1.1097340548903836, 0.45554979920974065}, {1.1091738942152194, 0.4578004611890056}, {1.1083390579673529, 0.4599385499296179}, {1.1072301281220838, 0.46196248105513726}, {1.1058480873856036, 0.4638708376993658}, {1.1041943171174926, 0.46566237409302735}, {1.1022705946583264, 0.4673360188917193}, {1.1000790900673068, 0.46889087823801834}, {1.0976223622756955, 0.47032623855102607}, {1.0949033546627247, 0.4716415690370607}, {1.0919253900615167, 0.4728365239156398}, {1.0886921652033936, 0.4739109443553391}, {1.085207744609812, 0.474864860114576}, {1.0814765539419628, 0.47569849088282584}, {1.0775033728188972, 0.4764122473182633}, {1.073293327115829, 0.47700673177830544}, {1.0688518807550254, 0.4774827387400213}, {1.064184827002458, 0.4778412549078847}, {1.0592982792841148, 0.47808345900684485}, {1.0541986615365755, 0.47821072125920955}, {1.0488926981071456, 0.4782246025443534}, {1.0433874032194952, 0.47812685324078}, {1.0376900700213916, 0.47791941175060404}, {1.0318082592317208, 0.4776044027070354}, {1.0257497874045662, 0.47718413486598404}, {1.019522714828686, 0.47666109868343454}, {1.01313533308123, 0.47603796358076195}, {1.0065961522550515, 0.4753175749006986}, {0.9999138878794309, 0.4745029505571799}, {0.9930974475544482, 0.47359727738282376}, {0.9861559173196675, 0.4726039071783198}, {0.9719347399971225, 0.47036828197048064}, {0.9646740312070661, 0.46913351577940704}, {0.9573260803209902, 0.46782602027854003}, {0.9424076083610509, 0.46500940590386003}, {0.9348568811286064, 0.4635089017027259}, {0.9272584696209715, 0.4619528855408388}, {0.9119588076097007, 0.4586928769703292}, {0.8812305861121766, 0.4517166659326065}, {0.8735804062088594, 0.4499069036537414}, {0.8659629641955255, 0.4480812230666914}, {0.8508657354889266, 0.444403332591291}, {0.8214922981755227, 0.4370923909761374}, {0.7680699438521148, 0.42387052538299463}, {0.761553229249061, 0.4223375506574874}, {0.7552404210348208, 0.42088761027697774}, {0.7432611077226945, 0.41826260818971867}, {0.7376108853543223, 0.41710011714275513}, {0.7321971080081611, 0.41604577673402665}, {0.7270268487961611, 0.4151054991556311}, {0.7221067321198098, 0.41428502552194635}, {0.7174429230385241, 0.4135899141156141}, {0.7130411174749149, 0.41302552890506805}, {0.7089065332725684, 0.41259702835682416}, {0.7050439021204269, 0.41230935456530926}, {0.7014574623562699, 0.4121672227225106}, {0.6981509526601951, 0.41217511094919834}, {0.6951276066473835, 0.4123372505088891}, {0.6923901483678016, 0.41265761642510085}, {0.6899407887188543, 0.4131399185217835}, {0.6877812227753484, 0.4137875929061045}, {0.6859126280394796, 0.41460379391202423}, {0.6843356636119003, 0.4155913865223136}, {0.6830504702832688, 0.416752939285843}, {0.6820566715440353, 0.41809071774612006}, {0.6813533755085792, 0.41960667839615684}, {0.680939177748182, 0.4213024631738322}, {0.6808121650256984, 0.423179394510952}, {0.6809699199231944, 0.42523847094823336}, {0.6814095263522351, 0.42748036332742306}, {0.6821275759349433, 0.4299054115707278}, {0.6831201752424224, 0.4325136220566688}, {0.6843829538756193, 0.43530466560039244}, {0.6859110733722357, 0.43827787604536833}, {0.6876992369218431, 0.44143224947227966}, {0.6897416998699547, 0.4447664440297797}, {0.6920322809904284, 0.44827878039063523}, {0.6973309628214613, 0.45582948096631765}, {0.7105882702412146, 0.47295761730343694}, {0.7457678697091037, 0.5144336736597995}, {0.7507840904579249, 0.5201958020778985}, {0.7559019457674834, 0.5260660934394914}, {0.7663973944634049, 0.5381039497751292}, {0.7881047994989112, 0.5631662852275465}, {0.8317674613503958, 0.6154719851761332}, {0.8366738818703674, 0.6216129149406244}, {0.841508767898968, 0.6277372000374448}, {0.8509285392657445, 0.6399073367819019}, {0.8685290498589887, 0.6637349012347294}, {0.8726233364011736, 0.6695438698671037}, {0.8765797321890528, 0.6752797747523422}, {0.884049532954602, 0.6865051821479444}, {0.8970258958693926, 0.7077854446376982}, {0.8998242404064929, 0.7128244960422071}, {0.902432821673512, 0.7177389545325107}, {0.9048464279686208, 0.7225229994392326}, {0.9070601569659698, 0.7271709820359656}, {0.9090694223690067, 0.7316774351887803}, {0.9108699600543321, 0.7360370827589674}, {0.9124578336978878, 0.7402448487429667}, {0.9138294398761138, 0.7442958661337874}, {0.9149815126355533, 0.748185485488634}, {0.9159111275252396, 0.7519092831878473}, {0.9166157050870567, 0.7554630693707273}, {0.9170930138001331, 0.7588428955342567}, {0.9173411724761971, 0.7620450617812506}, {0.9173586521036932, 0.7650661237049556}, {0.9171442771393353, 0.7679028988976708}, {0.916697226246642, 0.7705524730715153}, {0.9160170324818748, 0.7730122057800473}, {0.915103582928668, 0.7752797357300447}, {0.9139571177834995, 0.7773529856733712}, {0.9125782288950202, 0.779230166869491}, {0.9109678577610993, 0.7809097831098543}, {0.9091272929882982, 0.7823906342960419}, {0.9070581672193112, 0.7836718195642514}, {0.9047624535347387, 0.7847527399493979}, {0.9022424613363689, 0.7856331005828331}, {0.8995008317199358, 0.7863129124183941}, {0.8965405323461166, 0.78679249348225}, {0.8933648518192797, 0.7870724696427454}, {0.8899773935842629, 0.7871537748972092}, {0.8863820693521773, 0.7870376511734501}, {0.8825830920669525, 0.7867256476444391}, {0.8785849684250283, 0.7862196195554448}, {0.8743924909612643, 0.7855217265636765}, {0.8700107297147905, 0.7846344305912574}, {0.8654450234891434, 0.7835604931931479}, {0.8607009707216264, 0.7823029724424063}, {0.8557844199774193, 0.7808652193359655}, {0.8507014600844929, 0.7792508737248781}, {0.8400618079066493, 0.7755083809549006}, {0.834518401112308, 0.7733889145833872}, {0.8288351341779702, 0.771110205900112}, {0.8170777175068635, 0.7660953435878328}, {0.7922093361099414, 0.7543916639937892}, {0.7852116414827238, 0.7508757758234954}, {0.7781224279568875, 0.7472284416373801}, {0.7637112554165748, 0.7395745321436891}, {0.7342550977699308, 0.723088784766395}, {0.6751996698193874, 0.6876336194446264}, {0.6680093206180551, 0.6831722573714086}, {0.6608953613696905, 0.6787399223825356}, {0.6469352339144817, 0.6700051192351711}, {0.6203442992631959, 0.6533514194899652}, {0.614024418701484, 0.6494186851894818}, {0.6078520429865566, 0.6455985430644927}, {0.59598012775045, 0.6383353635656038}, {0.5743213800330619, 0.6255811017288745}, {0.5693773585066827, 0.6228138598264347}, {0.5646325665095843, 0.620231939810119}, {0.5557595142930087, 0.6156552211681865}, {0.551639679560054, 0.6136751712873675}, {0.5477359134136233, 0.6119099114835082}, {0.5440514070292903, 0.6103659221170349}, {0.5405889349440403, 0.6090493016824194}, {0.5373508518242103, 0.6079657543316844}, {0.5343390900217818, 0.6071205780785932}, {0.5315551579204102, 0.606518653707551}, {0.5290001390710874, 0.6061644344100102}, {0.5266746921158614, 0.6060619361698897}, {0.5245790514965598, 0.6062147289181967}, {0.5227130289440047, 0.6066259284756698}, {0.5210760157417611, 0.607298189300856}, {0.5196669857570291, 0.608233698059593}, {0.5184844992298796, 0.6094341680303927}, {0.5175267073106431, 0.6109008343587096}, {0.5167913573338979, 0.6126344501715469}, {0.516275798816163, 0.6146352835622864}, {0.5159769901630931, 0.6169031154540447}, {0.5158915060706952, 0.6194372383482549}, {0.5160155456038444, 0.6222364559635463}, {0.5163449409341659, 0.6252990837683621}, {0.5168751667181843, 0.6286229504091088}, {0.5176013500955097, 0.6322054000339684}, {0.5185182812857491, 0.636043295510855}, {0.5196204247617829, 0.6401330225363246}, {0.5209019309760576, 0.6444704946305968}, {0.523978137360513, 0.6538700033439551}, {0.5257257986740551, 0.6588273876878378}, {0.5276211299032619, 0.664003491535248}, {0.5318272575726402, 0.6749875025790294}, {0.5416971546655224, 0.6992907448195721}, {0.5655009421823998, 0.7554111158897377}, {0.5686955307832945, 0.762949372662865}, {0.5719104134865852, 0.7705699501265603}, {0.5783668254867389, 0.786014913675793}, {0.5911485736819811, 0.8174099674979243}, {0.5942679167556972, 0.825298085228938}, {0.5973404493444406, 0.8331790597765908}, {0.6033132713510516, 0.8488735308697986}, {0.6143556112722127, 0.8796709603128966}, {0.616885363280222, 0.8871806613777057}, {0.6193089368605186, 0.8945920674147247}, {0.6238115920072175, 0.909076130280762}, {0.6258782845997356, 0.9161272558958966}, {0.6278140403969399, 0.923037062477641}, {0.6312707628564009, 0.9363918317234232}, {0.6327814203643729, 0.9428169398813088}, {0.6341405382145111, 0.9490610558667074}, {0.6353436758430202, 0.9551149213372632}, {0.6363866902554499, 0.9609695763365576}, {0.6372657417891969, 0.9666163756910171}, {0.6379772992960444, 0.9720470048965321}, {0.6385181447381298, 0.9772534954654853}, {0.6388853771918422, 0.9822282397057923}, {0.6390764162552688, 0.9869640049044779}, {0.6390890048559269, 0.9914539468893206}, {0.6389212114566485, 0.9956916229431341}, {0.638571431658595, 0.9996710040463366}, {0.6380383892015071, 1.0033864864246076}, {0.6373211363624035, 1.006832902379599}, {0.6364190537550499, 1.0100055303818944}, {0.6353318495336189, 1.0129001044066708}, {0.6340595580050423, 1.0155128224938113}, {0.6326025376556418, 1.017840354515557}, {0.6309614685986656, 1.0198798491361474}, {0.6291373494504211, 1.0216289399492984}, {0.6271314936436927, 1.0230857507807942}, {0.6249455251881598, 1.024248900144916}, {0.6225813738884894, 1.02511750484491}, {0.620041270031745, 1.025691182709187}, {0.617515492455654, 1.025960542897098}, {0.6148410884748815, 1.025973880842156}, {0.6120205471768212, 1.0257320713168534}, {0.6090565660017299, 1.0252363701959744}, {0.6059520453160973, 1.0244884133916645}, {0.6027100826923381, 1.023490215207334}, {0.5993339669054669, 1.0222441661117125}, {0.5958271716578194, 1.0207530299352627}, {0.5921933490432771, 1.019019940492062}, {0.5884363227628244, 1.0170483976311502}, {0.5805687696940423, 1.0124057535814734}, {0.5764666840476391, 1.0097434388440862}, {0.5722582619088877, 1.0068602317911253}, {0.5635408230823603, 1.0004524763003246}, {0.5590413216461508, 0.9969394146277765}, {0.5544544977426344, 0.9932284181490895}, {0.5450390878468819, 0.9852390192188079}, {0.5253873728383069, 0.9671914436492898}, {0.48401660944766883, 0.9246396021846287}, {0.47875933950633587, 0.9188892285706891}, {0.4735014065131701, 0.9130747279971405}, {0.4630053217339841, 0.9012929651997441}, {0.4422415366016522, 0.8774002613088597}, {0.4027534313105384, 0.8305802187197672}, {0.39809898283798406, 0.8250156422290028}, {0.39352107901380773, 0.8195480261560874}, {0.38460992957900164, 0.8089418693120528}, {0.3678623696557927, 0.7893041448115407}, {0.36391742697253926, 0.784774375963023}, {0.3600747436998687, 0.7804133225676602}, {0.3527056530817602, 0.7722291548579635}, {0.33928848219112856, 0.7582201019473124}, {0.3359651567506391, 0.7550088066908855}, {0.332778667107048, 0.7520626253712133}, {0.32973010565674443, 0.7493887001402334}, {0.32682029409788066, 0.7469936987311425}, {0.32404978402503054, 0.7448838017775719}, {0.32141885804272363, 0.7430646909799455}, {0.31892753139235047, 0.741541538141673}, {0.3165755540859523, 0.7403189950962851}, {0.3143624135394453, 0.7394011845450563}, {0.3122873376968893, 0.7387916918230468}, {0.31034929863648586, 0.7384935576098488}, {0.3085470166480933, 0.738509271599649}, {0.3068789647711739, 0.7388407671435115}, {0.3053433737812374, 0.7394894168750544}, {0.30393823761202926, 0.7404560293289352}, {0.30266131919991834, 0.7417408465597942}, {0.30151015673618226, 0.7433435427675048}, {0.30048207031215896, 0.7452632239327811}, {0.2995741689415431, 0.7474984284653802}, {0.29878335794344213, 0.7500471288653113}, {0.29810634666918967, 0.7529067343956445}, {0.29753965655532405, 0.7560740947636848}, {0.2970796294845939, 0.7595455048054596}, {0.2967224364363448, 0.7633167101666479}, {0.2964640864071701, 0.7673829139712817}, {0.29630043558228253, 0.7717387844677546}, {0.2962271967376764, 0.7763784636398993}, {0.2962399488528034, 0.7812955767691387}, {0.2963341469131846, 0.7864832429319876}, {0.29650513188211935, 0.7919340864154681}, {0.2967481408204368, 0.7976402490313355}, {0.297058317133061, 0.8035934033083532}, {0.29743072092103207, 0.8097847665402612}, {0.297860339417538, 0.8162051156654967}, {0.2988708681620243, 0.8296937724685989}, {0.2994414832078796, 0.836741577289247}, {0.3000487436746442, 0.8439773974425617}, {0.3013523146706187, 0.8589680510288455}, {0.30203816830470576, 0.8666995501682189}, {0.3027397743350828, 0.8745724364607196}, {0.30416949224621265, 0.8906925457202746}, {0.3069920048370023, 0.9240684658962702}, {0.30766084557720513, 0.9325715021528228}, {0.3083052983693338, 0.9411109842818551}, {0.30892068736952, 0.9496731413443655}, {0.3095024422441153, 0.9582441367036129}, {0.3100461063227779, 0.9668100924402661}, {0.3105473444527597, 0.9753571138724321}, {0.31100195054044555, 0.9838713141368253}, {0.31140585476691857, 0.9923388387871713}, {0.3117551304650774, 1.0007458903658677}, {0.3120460006465956, 1.0090787529048997}, {0.31227484416780743, 1.0173238163121094}, {0.31243820152440904, 1.0254676005990497}, {0.3125327802656909, 1.0334967799069001}, {0.3125554600198555, 1.0413982062872318}, {0.31250329712282854, 1.049158933194801}, {0.3123735288438344, 1.0567662386500147}, {0.3121800633259086, 1.0637185879384}, {0.311914764816955, 1.0705164144725376}, {0.3115758287968238, 1.0771500007549961}, {0.31116159763701107, 1.0836098649618031}, {0.31067056216760824, 1.0898867758256918}, {0.3101013629629985, 1.0959717671467253}, {0.3094527913458571, 1.1018561519072994}, {0.3087237901095197, 1.1075315359691296}, {0.30791345395929814, 1.112989831330452}, {0.3070210296738299, 1.1182232689223284}, {0.30604591598804926, 1.123224410923667}, {0.3049876631998687, 1.1279861625752896}, {0.30384597250314765, 1.1325017834741267}, {0.30262069505001554, 1.1367648983294605}, {0.30131183074608797, 1.1407695071639148}, {0.2999195267825901, 1.1445099949427704}, {0.298444075909853, 1.1479811406160543}, {0.2968859144571041, 1.1511781255587636}, {0.2952456201039097, 1.154096541395499}, {0.29352390940905, 1.1567323971967498}, {0.29172163510302546, 1.1590821260350355}, {0.2898397831507937, 1.161142590890097}, {0.28787946959171906, 1.1629110898933483}, {0.28584193716409373, 1.1643853609028232}, {0.2837285517219456, 1.1655635854008763}, {0.2815407984521866, 1.1664443917079785}, {0.27928027790048404, 1.1670268575069844}, {0.2769487018145465, 1.1673105116733395}, {0.27454788881380143, 1.1672953354077686}, {0.2720797598947203, 1.1669817626690817}, {0.26954633378130394, 1.1663706799058224}, {0.26694972213046775, 1.165463425086576}, {0.2642921246022949, 1.1642617860298654}, {0.2615758238053184, 1.1627679980356382}, {0.25880318012717224, 1.160984740821463}, {0.25597662646110964, 1.1589151347676319}, {0.25309866283902743, 1.1565627364764461}, {0.25017185098175476, 1.1539315336520555}, {0.2441822046990902, 1.1478507853128357}, {0.241124752171891, 1.1444113152776894}, {0.2380292039019357, 1.1407131768057777}, {0.2317349931493463, 1.1325654539887198}, {0.22854198111570928, 1.128129106255843}, {0.22532216279833045, 1.1234605434991862}, {0.2188135661550847, 1.1134572360090862}, {0.21553052424234487, 1.1081385726341992}, {0.21223213764713916, 1.1026198326706487}, {0.20560071339182928, 1.0910177569006856}, {0.19227768144806096, 1.0658203402706485}, {0.1659856242655355, 1.0097684264216693}, {0.16284294823850523, 1.0025908322319743}, {0.15972462900739845, 0.9953823663470576}, {0.15356851068538324, 0.980916830404667}, {0.14162582753463068, 0.9521002199661683}, {0.1195224515362599, 0.8973641105531327}, {0.11694603681153837, 0.8909995133363569}, {0.11441323564608119, 0.8847745595043964}, {0.10947986770026742, 0.872781582475817}, {0.10014622940645392, 0.8508762396936198}, {0.09792359897562704, 0.8458830364032529}, {0.09574479698859513, 0.8410989913401623}, {0.09151696248695611, 0.8321877442982718}, {0.0894668458173857, 0.8280744130006684}, {0.08745839340419925, 0.8241979604887942}, {0.08356336402296702, 0.817179634615006}, {0.08167502506701432, 0.8140488221993415}, {0.07982483178368785, 0.8111769938067099}, {0.07801171223718503, 0.8085687329537641}, {0.07623452346161648, 0.8062282313393379}, {0.07449205463540218, 0.8041592824981413}, {0.072783030377179, 0.8023652760443912}, {0.07110611415617182, 0.8008491925141755}, {0.06945991180978288, 0.7996135988144586}, {0.06784297516096977, 0.7986606442857229}, {0.06625380572781289, 0.797992057384326}, {0.06469085851751777, 0.7976091429897177}, {0.06315254589695804, 0.7975127803407376}, {0.061637241531738045, 0.7977034216042563}, {0.06014328438564667, 0.798181091078488}, {0.05866898277227658, 0.7989453850323405}, {0.057212618450504606, 0.7999954721812209}, {0.05577245075546743, 0.8013300947987589}, {0.054346720756616916, 0.8029475704629523}, {0.0529336554344088, 0.8048457944342896}, {0.05153147186716226, 0.8070222426624568}, {0.050138381419627884, 0.8094739754172849}, {0.048752593924817504, 0.8121976415386678}, {0.04737232185068093, 0.8151894832992385}, {0.04599578444326219, 0.8184453418726786}, {0.044621211838030035, 0.8219606633996219}, {0.04324684913115831, 0.8257305056422196}, {0.04049183268309299, 0.8340120853991054}, {0.03910777985671436, 0.8385120644750552}, {0.03771714649203124, 0.8432430646504891}, {0.034909692265678575, 0.8533707338222954}, {0.03336847343333284, 0.8592198121949024}, {0.03181201408552314, 0.8653058785236031}, {0.02864608342396328, 0.8781472689238325}, {0.022055453952817484, 0.9061936958274643}, {0.0203451426373805, 0.9136283849211423}, {0.018607139536890745, 0.9212072899472551}, {0.015043589066771619, 0.9367455757835406}, {0.007540958310109574, 0.9689669322898153}, {0.005583067034314601, 0.9771802139116303}, {0.0035913647155365277, 0.9854285452584871}, {-0.0004943813358184401, 1.0019734762725407}, {-0.009075030155226142, 1.034857187190443}, {-0.011304023220352821, 1.0429522959663584}, {-0.01356560076452587, 1.0509689457006715}, {-0.018183702804500778, 1.0667115603249433}, {-0.027773917788125758, 1.096656811656572}, {-0.030238627301740262, 1.1037462607872173}, {-0.03272753782143437, 1.1106519353601958}, {-0.037771893950020105, 1.1238643329285485}, {-0.040324089916547415, 1.1301480955955003}, {-0.04289399604850066, 1.1362022029506114}, {-0.048079584184088954, 1.147579710729591}, {-0.050691418413333376, 1.1528832845512662}, {-0.053313277340960676, 1.1579175842545097}, {-0.05594308960909178, 1.1626738164836785}, {-0.05857873454062648, 1.1671436640830615}, {-0.06121804774574868, 1.1713193004930136}, {-0.06385882686086915, 1.1751934032848792}, {-0.06649883740659109, 1.1787591668110549}, {-0.06913581875098729, 1.1820103139481435}, {-0.07176749016421279, 1.1849411069127243}, {-0.07439155695023632, 1.1875463571309355}, {-0.07700571664126898, 1.1898214341447109}, {-0.0796076652402946, 1.1917622735392455}, {-0.08219510349696162, 1.1933653838779723}, {-0.08476574320199087, 1.194627852633113}, {-0.08731731348517134, 1.1955473511016201}, {-0.08984756710197242, 1.1961221382981297}, {-0.0923542866937932, 1.1963510638183694}, {-0.09483529100688688, 1.1962335696682536}, {-0.09728844105505512, 1.1957696910557574}, {-0.09971164621129419, 1.1949600561444755}, {-0.10210287021369577, 1.1938058847696094}, {-0.10446013707105845, 1.1923089861189669}, {-0.1067815368538511, 1.1904717553833686}, {-0.10906523135638894, 1.1882971693826985}, {-0.11116055278582945, 1.1859669716568992}, {-0.1132201464117554, 1.1833494787488175}, {-0.11524271267481558, 1.180448451518662}, {-0.11722702187219465, 1.177268071739638}, {-0.11917191758786952, 1.1738129356051654}, {-0.12107631998762036, 1.1700880466069739}, {-0.12293922897149137, 1.1660988077951977}, {-0.12475972717665074, 1.1618510134325446}, {-0.1265369828238718, 1.1573508400555386}, {-0.12827025240113904, 1.1526048369567352}, {-0.12995888317818427, 1.147619916102715}, {-0.13160231554606452, 1.142403341503494}, {-0.13320008517621693, 1.13696271804986}, {-0.13475182499376034, 1.13130597983592}, {-0.13771624366070756, 1.1193774679974662}, {-0.1391286896926892, 1.1131230966408536}, {-0.14049464285038651, 1.1066873884012156}, {-0.14308774336633134, 1.0933097636231217}, {-0.1443154900539249, 1.0863873570068934}, {-0.14549794342491562, 1.079322603533153}, {-0.14772933300684538, 1.0648074285600138}, {-0.14877971499694762, 1.0573781484532645}, {-0.1497876943890985, 1.0498487720293412}, {-0.15168048870439366, 1.0345336646412961}, {-0.15256758240740015, 1.0267701916129104}, {-0.15341682848243385, 1.0189511042604382}, {-0.15500743623533667, 1.0031915142779606}, {-0.1557518683112313, 0.9952738451130237}, {-0.1564645892580733, 0.9873461938734878}, {-0.157802050922555, 0.971506763549987}, {-0.16017988045412676, 0.9402103290932835}, {-0.16072479024869712, 0.9325316029244}, {-0.1612544064362825, 0.924933184213616}, {-0.16227733293672097, 0.9100205602267597}, {-0.16424809898587964, 0.881625888918502}, {-0.16474031388344604, 0.8748841794368193}, {-0.1652376946166883, 0.868304566909565}, {-0.16574297257482212, 0.8618962735172192}, {-0.16625890145093328, 0.8556682506764706}, {-0.1667882520659944, 0.8496291656571086}, {-0.16733380711660173, 0.8437873886415368}, {-0.16848468871804761, 0.8327276795253131}, {-0.16909559189423123, 0.8275248924701127}, {-0.1697338418733826, 0.8225496810289437}, {-0.17040219995460984, 0.8178087526565638}, {-0.17110340674346536, 0.813308450413136}, {-0.17184017664175602, 0.8090547436260531}, {-0.17261519234137568, 0.8050532191287193}, {-0.1734310993326009, 0.8013090730892757}, {-0.17429050043731484, 0.7978271034413437}, {-0.17527415083637832, 0.7943539838751953}, {-0.17631508921575573, 0.7911991185532791}, {-0.1774164162713898, 0.7883669791653584}, {-0.17858114302928318, 0.7858614533487491}, {-0.17981218311206174, 0.7836858382918783}, {-0.18111234515115576, 0.7818428354014114}, {-0.18248432536232567, 0.7803345460416126}, {-0.18393070030200398, 0.7791624683526585}, {-0.18545391982163165, 0.7783274951526765}, {-0.18705630023683983, 0.7778299129263035}, {-0.1887400177279591, 0.7776694019006024}, {-0.19050710198793191, 0.7778450372071978}, {-0.1923594301332675, 0.7783552911275361}, {-0.19429872089319578, 0.7791980364162153}, {-0.19632652909166903, 0.7803705506953784}, {-0.19844424043631825, 0.7818695219112463}, {-0.20065306662788548, 0.7836910548419403}, {-0.20295404080305168, 0.7858306786438582}, {-0.20534801332293637, 0.7882833554219983}, {-0.20783564791887627, 0.7910434898077928}, {-0.2104174182063983, 0.7941049395261998}, {-0.21309360457756935, 0.797461026932024}, {-0.2187293650993339, 0.8050278032012357}, {-0.22168851142860552, 0.8092225768727532}, {-0.22474121477233663, 0.8136801873337008}, {-0.23112421513423637, 0.8233468749138633}, {-0.24496765446190738, 0.8454027787859206}, {-0.2486437537323586, 0.8514203057889825}, {-0.25240198248316936, 0.8576159940462188}, {-0.26015532123517604, 0.870492794163375}, {-0.2765323471434693, 0.8978298660793372}, {-0.31200268310657187, 0.9558135930446261}, {-0.31661294881794577, 0.9631004703775788}, {-0.3212474660484378, 0.9703492473726094}, {-0.33056994288295527, 0.9846769935929208}, {-0.349288709629935, 1.012271887932338}, {-0.35395382460602454, 1.018872203047566}, {-0.35860263580446994, 1.0253274553927827}, {-0.3678301561400789, 1.037753446716973}, {-0.3858561430554561, 1.0603395134460483}, {-0.3901627465648367, 1.0653583487315215}, {-0.3944130516428336, 1.0701508439838097}, {-0.4027254317472323, 1.0790211307410964}, {-0.4067780097876943, 1.0830821180830987}, {-0.41075531486613887, 1.086883184760071}, {-0.41846607094498905, 1.093676803985397}, {-0.42219077068981004, 1.0966561457065402}, {-0.4258227135509275, 1.0993491664394535}, {-0.42935783179455883, 1.1017504725290586}, {-0.43279219699941984, 1.103855171244151}, {-0.43612202873092815, 1.1056588784300467}, {-0.4393437029958701, 1.1071577252698093}, {-0.44245376045961715, 1.1083483641436178}, {-0.4454489144083337, 1.1092279735774262}, {-0.44832605843901197, 1.1097942622736552}, {-0.4510822738606045, 1.110045472218295}, {-0.45371483678999003, 1.1099803808603979}, {-0.4562212249270124, 1.1095983023615845}, {-0.45859912399336034, 1.1088990879148055}, {-0.4608464338206261, 1.1078831251332355}, {-0.4629612740734776, 1.1065513365117972}, {-0.4649419895945079, 1.1049051769654357}, {-0.4667871553579801, 1.1029466304498605}, {-0.46849558102037353, 1.1006782056720876}, {-0.47006631505634733, 1.0981029308996744}, {-0.471498648469479, 1.0952243478791155}, {-0.4727921180678922, 1.092046504875405}, {-0.47394650929567717, 1.088573948846292}, {-0.47496185861181256, 1.0848117167662472}, {-0.4758384554091226, 1.0807653261166175}, {-0.47657684346665274, 1.0764407645598906}, {-0.47717782192970165, 1.0718444788173769}, {-0.47764244581263293, 1.066983362770988}, {-0.4779720260204761, 1.0618647448111127}, {-0.4781681288862301, 1.0564963744538778}, {-0.47823257522169865, 1.0508864082523124}, {-0.4781674388806046, 1.0450433950271427}, {-0.4779750448336659, 1.0389762604440897}, {-0.47765796675624345, 1.032694290965636}, {-0.47721902413011374, 1.0262071172062803}, {-0.476661278861856, 1.0195246967212952}, {-0.47598803142128276, 1.0126572962599454}, {-0.47520281650428237, 1.0056154735150105}, {-0.47430939822537166, 0.9984100584012767}, {-0.4733117648461855, 0.9910521338964362}, {-0.4722141230470537, 0.9835530164785563}, {-0.47102089174972006, 0.9759242361948988}, {-0.46973669550016584, 0.9681775163974805}, {-0.46691489174679307, 0.9523779945614469}, {-0.46040427469938666, 0.9198960999804681}, {-0.45862850044748954, 0.9116639190511302}, {-0.45680522944629925, 0.9034120314954122}, {-0.45304075646218556, 0.8868991740413422}, {-0.445217998260477, 0.854185095124367}, {-0.4433724833130449, 0.8466826903960522}, {-0.4415301164446039, 0.83924801220292}, {-0.43787671270416373, 0.824619926468828}, {-0.43084913085731774, 0.7965869266167359}, {-0.4291824353989509, 0.7898852093273758}, {-0.42756272557742153, 0.783323031913286}, {-0.4244857823555036, 0.7706493511490393}, {-0.419129214527817, 0.7473149259982793}, {-0.41798603728584416, 0.7419413120112757}, {-0.41693112774078184, 0.7367644673195178}, {-0.4159692450133251, 0.7317902425773266}, {-0.41510503115009356, 0.727024169025564}, {-0.4143430030784079, 0.7224714508609091}, {-0.4136875447160328, 0.7181369581145962}, {-0.4131428992494185, 0.7140252200504333}, {-0.41271316159374466, 0.7101404190910765}, {-0.41240227104786037, 0.706486385280755}, {-0.41221400415693554, 0.7030665912917734}, {-0.41215196779537616, 0.6998841479812838}, {-0.41221959248225354, 0.6969418005039718}, {-0.41242012594117133, 0.6942419249854236}, {-0.41275662691616033, 0.6917865257601069}, {-0.4132319592548244, 0.6895772331769856}, {-0.4138487862695835, 0.6876153019749498}, {-0.41460956538745225, 0.6859016102293543}, {-0.4155165430983851, 0.6844366588700775}, {-0.4165717502117688, 0.6832205717706479}, {-0.4177769974301949, 0.68225309640711}, {-0.419133871249177, 0.6815336050844306}, {-0.4206437301909811, 0.6810610967273837}, {-0.4223077013802413, 0.6808341992320014}, {-0.4241266774685156, 0.6808511723728194}, {-0.4261013139144039, 0.6811099112603113}, {-0.4282320266253052, 0.6816079503420753}, {-0.43051898996633664, 0.6823424679405179}, {-0.4329621351413759, 0.6833102913189715}, {-0.4355611489506004, 0.6845079022673966}, {-0.4383154729283154, 0.6859314431980343}, {-0.4412243028642675, 0.6875767237406218}, {-0.44428658871104193, 0.6894392278260398}, {-0.44750103487952514, 0.6915141212465348}, {-0.45086610092380014, 0.6937962596799553}, {-0.45804071341281616, 0.698960195011853}, {-0.4741027504493732, 0.7115163047667215}, {-0.5123631172401407, 0.7439682666337811}, {-0.5180938288317287, 0.7489530697950305}, {-0.5239354599083553, 0.7540439043391991}, {-0.5359244635233406, 0.7644984818781256}, {-0.5609240380767164, 0.7861762281849687}, {-0.6132477490496487, 0.829976660713097}, {-0.6198428583051724, 0.8352654914231881}, {-0.6264211586746906, 0.8404749008503752}, {-0.639492105549424, 0.8506115338722285}, {-0.6650453663598478, 0.8694600768139439}, {-0.6712602522361565, 0.8738165015027456}, {-0.6773886220417731, 0.8780115228791259}, {-0.6893523239891348, 0.8858815243975503}, {-0.6951712321779596, 0.8895394092975836}, {-0.7008708074563709, 0.8930017287245742}, {-0.7118808560599247, 0.8993093889520554}, {-0.7171762509139078, 0.9021405412849341}, {-0.7223221850896185, 0.9047477740239934}, {-0.727311657591744, 0.907124995610204}, {-0.7321379078496346, 0.9092665334728749}, {-0.7367944291105117, 0.9111671427839376}, {-0.7412749814302054, 0.9128220144206054}, {-0.7455736042357266, 0.9142267821244051}, {-0.749684628434676, 0.9153775288460846}, {-0.7536026880472214, 0.9162707922673994}, {-0.7573227313371738, 0.9169035694922978}, {-0.760840031419521, 0.9172733209015532}, {-0.7641501963226633, 0.9173779731664268}, {-0.7672491784845157, 0.9172159214184726}, {-0.7701332836626067, 0.9167860305741498}, {-0.7727991792393145, 0.9160876358144272}, {-0.7752439019044279, 0.9151205422211081}, {-0.7774648646982981, 0.9138850235731263}, {-0.779459863399966, 0.9123818203075714}, {-0.7812270822458038, 0.9106121366517124}, {-0.7827650989653949, 0.90857763693377}, {-0.7840728891225739, 0.9062804410816484}, {-0.7851498297508072, 0.903723119320303}, {-0.7859957022733369, 0.9009086860798196}, {-0.7866106946998133, 0.8978405931276926}, {-0.7869954030924339, 0.8945227219401571}, {-0.7871508322959362, 0.8909593753287592}, {-0.7870783959271286, 0.8871552683396576}, {-0.7867799156209913, 0.8831155184444186}, {-0.7862576195317493, 0.8788456350423014}, {-0.7855141400886712, 0.8743515082951909}, {-0.7845525110077483, 0.8696393973175406}, {-0.7833761635617507, 0.8647159177447216}, {-0.7820878211526551, 0.8599358824778025}, {-0.7806191611866177, 0.8549836831132406}, {-0.7771559943702168, 0.8445876575534635}, {-0.7751697492166167, 0.8391568284774603}, {-0.7730196991858093, 0.8335798078768296}, {-0.7682477057868664, 0.8220154323538318}, {-0.7656361719405654, 0.8160426495960427}, {-0.7628816379699779, 0.8099527958080629}, {-0.7569670846800132, 0.7974527485735005}, {-0.7436931460545243, 0.7713943717787094}, {-0.7401129654229314, 0.7647069575445389}, {-0.7364423414570922, 0.757967065609716}, {-0.7288595820365682, 0.7443635613798941}, {-0.712944698199476, 0.7168987668939686}, {-0.6799382097412264, 0.6628161075416279}, {-0.6758652693361492, 0.6562940314375468}, {-0.6718322410546735, 0.6498508516646401}, {-0.6639194877053662, 0.6372306273877139}, {-0.6489387252733945, 0.6132509635325823}, {-0.6454167674488628, 0.6075573235615781}, {-0.6419998676044706, 0.6019965847729659}, {-0.6355117014624654, 0.5912967542553154}, {-0.6241219719415311, 0.5717303465746767}, {-0.6216454407885866, 0.5672481750003747}, {-0.6193302860753269, 0.5629381126948606}, {-0.6152083201423749, 0.554848717263478}, {-0.6134130013905265, 0.5510758535660065}, {-0.6118020244645401, 0.5474880302543537}, {-0.6103804776439169, 0.544087734215077}, {-0.6091531697600858, 0.5408771479336528}, {-0.6081246217872541, 0.5378581473276024}, {-0.6072990588578253, 0.5350323000715089}, {-0.6066804027162762, 0.5324008644147685}, {-0.6062722646247288, 0.5299647884921191}, {-0.6060779387327724, 0.5277247101262066}, {-0.6061003959233935, 0.5256809571206597}, {-0.6063422781461493, 0.5238335480413694}, {-0.6068058932479741, 0.5221821934828845}, {-0.6075608341901142, 0.5206134479944939}, {-0.6085804751439041, 0.5192730310726003}, {-0.6098663884966461, 0.5181593671289476}, {-0.6114196640339414, 0.5172704703690308}, {-0.6132409052962832, 0.5166039503397669}, {-0.6153302268173049, 0.5161570182036753}, {-0.6176872522515707, 0.5159264937247867}, {-0.6203111133981641, 0.5159088129502182}, {-0.6232004501246842, 0.5161000365701076}, {-0.6263534111945958, 0.5164958589373833}, {-0.6297676559992075, 0.5170916177276772}, {-0.6334403571938694, 0.5178823042185567}, {-0.6373682042363147, 0.518862574166173}, {-0.6415474078233537, 0.5200267592563649}, {-0.6506423664772756, 0.5228826537916877}, {-0.655548201519759, 0.5245615168740008}, {-0.6606855681093142, 0.5263986289007331}, {-0.6716301198727517, 0.5305189610026393}, {-0.6960013876344058, 0.5403307184704628}, {-0.752774272353995, 0.564380773940367}, {-0.7604339105679908, 0.5676309386227133}, {-0.7681829149294015, 0.5709051929324838}, {-0.7839034228027949, 0.5774895937609973}, {-0.8159062054958657, 0.5905492146068324}, {-0.8239541118549242, 0.5937394570067057}, {-0.8319966221131333, 0.5968824040098211}, {-0.8480165608892238, 0.6029925418179709}, {-0.8794544992965169, 0.6142816463540146}, {-0.887118010724562, 0.6168645585207575}, {-0.8946793029091927, 0.6193370072580024}, {-0.9094486512169961, 0.6239229556174348}, {-0.9166338535319605, 0.6260233149603649}, {-0.9236711714416257, 0.6279869519248354}, {-0.9372588170137998, 0.6314808174201525}, {-0.9437881191860038, 0.6330001644954101}, {-0.9501275255301869, 0.6343610432195416}, {-0.956267254526934, 0.6355587890840507}, {-0.9621978510810104, 0.6365890611223587}, {-0.9679102045455735, 0.6374478481624817}, {-0.9733955661643735, 0.6381314744240765}, {-0.9786455658984509, 0.6386366044525149}, {-0.9836522286048905, 0.6389602473839449}, {-0.9883236852499057, 0.6390988801713691}, {-0.9927465317794587, 0.639057854414373}, {-0.9969144156507133, 0.6388353266525051}, {-1.0008214043437829, 0.6384297831249719}, {-1.0044619967921162, 0.6378400405974011}, {-1.0078311340822501, 0.6370652465815694}, {-1.0109242094022692, 0.6361048789507842}, {-1.013737077219591, 0.6349587449547078}, {-1.0162660616700088, 0.6336269796385177}, {-1.0185079641413062, 0.6321100436723746}, {-1.0204600700361433, 0.6304087205982442}, {-1.0221201547003373, 0.6285241135021615}, {-1.0234864885041193, 0.6264576411210593}, {-1.024557841065422, 0.6242110333942912}, {-1.0253334846057582, 0.6217863264709673}, {-1.025813196430771, 0.6191858571851767}, {-1.0259972605290693, 0.6164122570121014}, {-1.0258864682845212, 0.613468445518944}, {-1.0254821182987468, 0.610357623325453}, {-1.0247860153221096, 0.6070832645896744}, {-1.0238004682931081, 0.6036491090353782}, {-1.0225282874876334, 0.6000591535383705}, {-1.0209727807811557, 0.5963176432896549}, {-1.0191377490284819, 0.5924290625540984}, {-1.0170274805672939, 0.5883981250439279}, {-1.014646744853269, 0.5842297639270088}, {-1.0090953108874088, 0.5755015384805409}, {-1.0059364878926926, 0.5709525431408178}, {-1.0025309295216889, 0.5662878399699377}, {-0.9950082316439912, 0.556634940173165}, {-0.990906455842589, 0.551658929174044}, {-0.9865886471300509, 0.5465915575175456}, {-0.977340006043012, 0.5362084721802066}, {-0.9566561587126196, 0.514634242965003}, {-0.9089218395751595, 0.46977883311757473}, {-0.8095126955136985, 0.3850910627403168}, {-0.8043695433612464, 0.38074744364841556}, {-0.7993602540370531, 0.37649456361198497}, {-0.7897789724994088, 0.36827354368747456}, {-0.7726033652244984, 0.3530481131509136}, {-0.7687701305652799, 0.349509240943952}, {-0.7651361239345732, 0.3460814151297045}, {-0.7584934138673511, 0.33956546236601415}, {-0.7554977501101712, 0.33648012625197254}, {-0.7527273729459186, 0.3335114165145741}, {-0.7478849510244302, 0.3279272916772073}, {-0.7458232754509566, 0.3253130914932841}, {-0.7440076079856559, 0.3228179484512863}, {-0.7424422290996647, 0.3204419798345238}, {-0.7411310443237368, 0.31818510881764783}, {-0.7400775777265751, 0.3160470660459191}, {-0.7392849659637437, 0.3140273915230947}, {-0.7387559529071304, 0.312125436802926}, {-0.73849288486406, 0.31034036747877414}, {-0.7384977063942587, 0.30867116596538025}, {-0.7387719567319676, 0.30711663456635807}, {-0.7393167668195809, 0.30567539882053557}, {-0.7401328569582525, 0.30434591111983555}, {-0.7412205350799738, 0.3031264545909594}, {-0.742579695644679, 0.30201514723273926}, {-0.7442098191649741, 0.30100994630063094}, {-0.7461099723601271, 0.3001086529294463}, {-0.7482788089399912, 0.299308916985068}, {-0.7507145710185652, 0.29860824213555154}, {-0.7534150911559284, 0.2980039911316943}, {-0.7563777950263162, 0.29749339128685337}, {-0.7595997047091385, 0.29707354014550497}, {-0.7630774425987856, 0.29674141132977794}, {-0.7668072359281057, 0.2964938605529465}, {-0.7707849218994843, 0.29632763178863936}, {-0.7750059534165173, 0.29623936358432296}, {-0.7794654054083486, 0.29622559550742855}, {-0.7841579817378017, 0.29628277471232833}, {-0.7890780226835371, 0.29640726261622713}, {-0.7942195129855738, 0.29659534167190893}, {-0.7995760904426271, 0.2968432222251803}, {-0.8051410550488652, 0.2971470494447769}, {-0.810907378656827, 0.2975029103124341}, {-0.8168677151524232, 0.2979068406607912}, {-0.8230144111271432, 0.2983548322467829}, {-0.8358347987954514, 0.2993667883709628}, {-0.8633437323940664, 0.3017398417377585}, {-0.8705572923646305, 0.3023818192598693}, {-0.8778863536034673, 0.30303505613645376}, {-0.8928513424548955, 0.30435898654447113}, {-0.9237222547485219, 0.3069642461562497}, {-0.9322376239580427, 0.3076350721938981}, {-0.9407900994022943, 0.30828161529099285}, {-0.9493658446439368, 0.30889917244022036}, {-0.9579509561922417, 0.30948314653137604}, {-0.9665314881163236, 0.3100290545829516}, {-0.9750934767658754, 0.31053253567245565}, {-0.9836229655551688, 0.31098935855134036}, {-0.9921060297659031, 0.31139542893112576}, {-1.0005288013244225, 0.3117467964280745}, {-1.0088774935087854, 0.3120396611545552}, {-1.017138425541237, 0.3122703799460162}, {-1.0252980470218775, 0.31243547221333623}, {-1.033342962159396, 0.31253162541111895}, {-1.041259953755212, 0.3125557001133835}, {-1.0490360068976785, 0.3125047346889414}, {-1.0566583323234704, 0.31237594956963954}, {-1.0641143894038743, 0.31216675110553066}, {-1.0713919087142998, 0.3118747350019196}, {-1.0784789141460334, 0.3114976893341462}, {-1.0853637445200472, 0.31103359713685497}, {-1.0920350746635374, 0.31048063856541397}, {-1.098481935910762, 0.3098371926280522}, {-1.10469373599076, 0.3091018384881822}, {-1.1106602782656396, 0.3082733563372859}, {-1.116371780284162, 0.30735072783962497}, {-1.1218188916166363, 0.3063331361509367}, {-1.1269927109383588, 0.3052199655141473}, {-1.1318848023301409, 0.30401080043601014}, {-1.1364872107658741, 0.3027054244494305}, {-1.1407924767584994, 0.30130381846707427}, {-1.1447936501372398, 0.29980615873269956}, {-1.1484843029304903, 0.2982128143774468}, {-1.1518585413303541, 0.296524344589112}, {-1.15491101671643, 0.2947414954032097}, {-1.1576369357181373, 0.2928651961253676}, {-1.1600320692965926, 0.2908965553953188}, {-1.1620927608287568, 0.2888368569034656}, {-1.1638159331784033, 0.2866875547716525}, {-1.1651990947402353, 0.2844502686104267}, {-1.1662403444453244, 0.2821267782656944}, {-1.1669383757179086, 0.27971901826825296}, {-1.1672924793754629, 0.2772290720002335}, {-1.1673025454658503, 0.2746591655930217}, {-1.1669690640372725, 0.2720116615717021}, {-1.1662931248386585, 0.269289052261524}, {-1.1652764159500462, 0.2664939529723192}, {-1.163921221344454, 0.2636290949771758}, {-1.1622304173846425, 0.26069731830201176}, {-1.1602074682601085, 0.25770156434302643}, {-1.1578564203715584, 0.2546448683292584}, {-1.1551818956719988, 0.25153035164771614}, {-1.1521890839754905, 0.2483612140487609}, {-1.148883734246457, 0.24514072574956144}, {-1.1452721448843068, 0.24187221945356202}, {-1.1371581228424035, 0.23520474778952866}, {-1.1326709329761377, 0.23181268762003457}, {-1.1279079629301103, 0.22838640359065282}, {-1.1175906171488357, 0.22144527280430287}, {-1.0940672988665974, 0.2073078480783401}, {-1.0880792897456122, 0.2039777220928819}, {-1.081922669719917, 0.20064208673295145}, {-1.0691432510178127, 0.19396537900540273}, {-1.0420101549653331, 0.1806655682148624}, {-0.9839198588851611, 0.15483624360730838}, {-0.9765243363995604, 0.15172309782077123}, {-0.9691380973982217, 0.1486420727818424}, {-0.9544405116995704, 0.14258302127607175}, {-0.9256714390937368, 0.13091776668448885}, {-0.8731604355631912, 0.10963709569074329}, {-0.86727561175688, 0.10718201723737823}, {-0.8615746876407597, 0.10477341269018361}, {-0.8507609466175466, 0.10009542666636688}, {-0.8456655737070842, 0.09782568166652737}, {-0.8407889612372172, 0.09560168653294207}, {-0.8317232726030654, 0.09128909114988634}, {-0.8275489346822497, 0.08919931727089903}, {-0.8236228106045315, 0.08715295062582354}, {-0.8199513292985572, 0.08514916886823028}, {-0.8165405248579513, 0.08318706025424295}, {-0.8133960270573082, 0.0812656265919813}, {-0.8105230524924713, 0.07938378635769776}, {-0.8079263963591441, 0.07754037797142506}, {-0.805610424882918, 0.07573416322466532}, {-0.8035790684127373, 0.07396383085236231}, {-0.8018358151888242, 0.07222800024113883}, {-0.8003837057950136, 0.07052522526553685}, {-0.799225328304384, 0.06885399824376373}, {-0.7983628141259849, 0.06721275400423354}, {-0.7977978345593718, 0.06559987405400229}, {-0.7975315980625408, 0.06401369084001086}, {-0.7975648482377485, 0.06245249209389092}, {-0.7978978625385735, 0.06091452525095058}, {-0.798530451700451, 0.05939800193383048}, {-0.7994619598957754, 0.057901102491214856}, {-0.8006912656135303, 0.056421980581902384}, {-0.8022167832622719, 0.054958767794469794}, {-0.8040364654941585, 0.05350957829271443}, {-0.8061478062465874, 0.05207251347703882}, {-0.8085478444968703, 0.05064566665192705}, {-0.8111777133245849, 0.049254916410153476}, {-0.8140781449222342, 0.0478703522411977}, {-0.8172451032063484, 0.046490180761934834}, {-0.820674151918031, 0.045112618066104594}, {-0.8243604611250164, 0.04373589384806414}, {-0.8282988143125488, 0.04235825548077559}, {-0.8369089002398092, 0.039593338265678346}, {-0.8415683388825835, 0.03820267845550881}, {-0.8464552514351519, 0.036804350278093076}, {-0.8568830729899414, 0.033978308636857385}, {-0.880137737064739, 0.02816649921890251}, {-0.8864024937202658, 0.026672068705817568}, {-0.8928298748606277, 0.025158458827507512}, {-0.906134870915762, 0.022069019639765656}, {-0.9342700840270616, 0.015612307852946088}, {-0.9942924300377717, 0.0014186903211025716}, {-1.0019336765898221, -0.00048439083852275255}, {-1.0095684303981032, -0.00241665401318051}, {-1.0247735587686022, -0.0063683560147238276}, {-1.0546140928045986, -0.01461311894694597}, {-1.0618912762575778, -0.016742919201923643}, {-1.069073635415941, -0.01889897538357762}, {-1.08311162387806, -0.023286843026591397}, {-1.109594206116906, -0.032340277978331666}, {-1.1158254718272747, -0.034655376011279174}, {-1.1218821779021877, -0.03698880196367558}, {-1.1334362802225777, -0.041705400685434074}, {-1.1389165412388444, -0.044085811110048896}, {-1.1441879975341247, -0.046479030903293875}, {-1.1540734877102643, -0.05129780856003389}, {-1.1586728184007062, -0.05372020993904097}, {-1.1630339597204595, -0.056149114790797726}, {-1.1671504090320104, -0.05858284949465178}, {-1.171016021069014, -0.0610197089803595}, {-1.17462501699622, -0.06345796058293723}, {-1.1779719929210757, -0.06589584797125377}, {-1.181051927844352, -0.06833159514252606}, {-1.1838601910379802, -0.07076341047473768}, {-1.1865946002981456, -0.0733950269016141}, {-1.1889994340417374, -0.07601758191653107}, {-1.1910703908212443, -0.07862875982081106}, {-1.1928037434745082, -0.0812262472999183}, {-1.1941963454886944, -0.08380773976522925}, {-1.195245636343116, -0.08637094770511532}, {-1.1959496458217687, -0.08891360303020998}, {-1.1963069972882192, -0.0914334653976927}, {-1.196316909917379, -0.0939283284994481}, {-1.1959791998805, -0.09639602629899605}, {-1.1952942804816196, -0.09883443920216271}, {-1.194263161245515, -0.10124150014657188}, {-1.1928874459591108, -0.10361520059518335}, {-1.191169329670109, -0.10595359641928183}, {-1.1891115946484807, -0.1082548136565238}, {-1.1867176053182833, -0.11051705412989159}, {-1.183991302169096, -0.11273860091368139}, {-1.1809371946581588, -0.11491782363295512}, {-1.177560353116105, -0.11705318358321813}, {-1.1738663996709133, -0.11914323865744868}, {-1.1698614982064783, -0.12118664806800579}, {-1.1655523433738484, -0.12318217685136106}, {-1.1609461486749038, -0.12512870014405025}, {-1.156050633639865, -0.12702520721871632}, {-1.150874010121601, -0.1288708052696215}, {-1.1454249677312955, -0.13066472293753556}, {-1.1397126584415074, -0.1324063135644526}, {-1.133746680384178, -0.13409505816916759}, {-1.1275370608724764, -0.1357305681353344}, {-1.121094238676806, -0.1373125876042528}, {-1.1075526872906154, -0.14031580763719537}, {-1.1004767236102042, -0.14173717753531728}, {-1.0932130481200164, -0.14310539821823887}, {-1.0857738671931494, -0.14442090271076463}, {-1.0781716785074973, -0.14568426459872474}, {-1.0704192490517106, -0.14689619819286606}, {-1.062529592669748, -0.14805755835951456}, {-1.046391751136883, -0.15023267729337372}, {-1.0381706201942353, -0.15124884235635544}, {-1.029866323248816, -0.1522192438978899}, {-1.0130639279386575, -0.15402906242459782}, {-1.0045939151159868, -0.1548719611794614}, {-0.996096858181343, -0.15567605462085188}, {-0.9790782945958021, -0.1571761744920109}, {-0.9705851194655666, -0.15787667302454395}, {-0.9621215137596233, -0.158547302600985}, {-0.9453390968264871, -0.15980912060861183}, {-0.9127395005922537, -0.16209167463717578}, {-0.9048692556384622, -0.162629101841181}, {-0.8971369995271017, -0.1631593926265122}, {-0.882136726583088, -0.1642113917864504}, {-0.8748931280116579, -0.16473964990398035}, {-0.8678363125552454, -0.1652738573500623}, {-0.8543282578757567, -0.16637366931132386}, {-0.8483211677043645, -0.16690729913575553}, {-0.842513905881198, -0.1674578094798065}, {-0.8369145056191325, -0.1680280053082486}, {-0.8315306736395379, -0.16862069156212803}, {-0.8263697786857115, -0.1692386676777931}, {-0.8214388405619032, -0.1698847220707284}, {-0.8167445197144227, -0.17056162659449298}, {-0.8122931073704162, -0.17127213098515134}, {-0.8080905162490818, -0.1720189573016776}, {-0.8041422718591852, -0.17280479437286675}, {-0.8004535043958045, -0.17363229226133264}, {-0.7970289412483048, -0.17450405675519395}, {-0.7938729001306069, -0.17542264389805298}, {-0.7909892828438182, -0.176390554567854}, {-0.7883815696803418, -0.17741022911517793}, {-0.786052814477541, -0.17848404207146493}, {-0.7840056403280633, -0.1796142969375918}, {-0.7822422359528917, -0.18080322106312902}, {-0.7807643527421663, -0.18205296062649234}, {-0.7795733024677702, -0.18336557572606998}, {-0.7786699556706637, -0.1847430355922547}, {-0.7780547407248677, -0.18618721393013635}, {-0.777727643578983, -0.1876998844024258}, {-0.7776882081750617, -0.18928271626196827}, {-0.777935537543615, -0.1909372701429836}, {-0.7784682955724894, -0.19266499401992013}, {-0.7792847094463072, -0.19446721934255431}, {-0.780382572752136, -0.1963451573556908}, {-0.7817592492460181, -0.19829989561151595}, {-0.7834116772739862, -0.20033239468235622}, {-0.7853363748401764, -0.20244348508126336}, {-0.7875294453136533, -0.20463386439750467}, {-0.7899865837646002, -0.20690409465369147}, {-0.7927030839195371, -0.20925459989089656}, {-0.7988933835025938, -0.2141974287190203}, {-0.8141365893428711, -0.22504926785884088}, {-0.8185048683495959, -0.22796158407464912}, {-0.8230814396388244, -0.2309523900755505}, {-0.8328275202495588, -0.2371657781182152}, {-0.8544517546397045, -0.2504855200207181}, {-0.9578552883284298, -0.31328978878270647}, {-0.9651448807577165, -0.31791495143967985}, {-0.9723923126134792, -0.3225630991768925}, {-0.986705027868415, -0.33190883376457464}, {-1.014219815159548, -0.3506567912518308}, {-1.0207896287152556, -0.35532534848095204}, {-1.0272102055020897, -0.35997598569664374}, {-1.039554421657778, -0.3692021839787023}, {-1.0619283921169733, -0.3872049283103415}, {-1.066975790362958, -0.39158102048226656}, {-1.0717843980556863, -0.395896821034737}, {-1.0806479429681792, -0.40432710471000993}, {-1.0846853691394456, -0.40843156072873876}, {-1.0884490126579938, -0.41245569235740404}, {-1.0951252525664745, -0.4202440130347828}, {-1.0980242752967153, -0.4239990172067113}, {-1.1006223907998813, -0.4276553476396268}, {-1.1029141376676515, -0.4312087573085435}, {-1.1048946017426355, -0.4346551550332921}, {-1.106559424064093, -0.4379906149447816}, {-1.1079048078014537, -0.44121138569272017}, {-1.108927524164611, -0.4443138993743925}, {-1.1096249172818482, -0.44729478016453494}, {-1.1099949080381355, -0.4501508526268412}, {-1.1100359968684514, -0.45287914968816056}, {-1.1097472655026779, -0.4554769202570254}, {-1.1091283776605443, -0.45794163646875646}, {-1.1081795786970012, -0.4602710005400465}, {-1.106901694200318, -0.4624629512166148}, {-1.1052961275470998, -0.4645156697982483}, {-1.1033648564203102, -0.46642758572631166}, {-1.1011104282982702, -0.4681973817196006}, {-1.098535954924458, -0.4698239984452464}, {-1.0956451057697747, -0.47130663871223305}, {-1.0924421005007705, -0.4726447711759871}, {-1.0889317004690897, -0.4738381335434071}, {-1.0851191992391775, -0.47488673526865066}, {-1.0810104121729986, -0.4757908597309619}, {-1.0766116650922013, -0.4765510658868108}, {-1.0719297820398423, -0.477168189389628}, {-1.0669720721653355, -0.47764334317145046}, {-1.0617463157578868, -0.477977917481829}, {-1.0562607494551983, -0.4781735793804258}, {-1.0505240506556248, -0.478232271680784}, {-1.044545321163435, -0.47815621134384967}, {-1.038334070098138, -0.47794788732090776}, {-1.0319001961001397, -0.4776100578467002}, {-1.0252539688662161, -0.47714574718458536}, {-1.0184060100494503, -0.4765582418267083}, {-1.0113672735594108, -0.47585108615326}, {-1.0041490252993206, -0.4750280775559829}, {-0.9967628223779869, -0.47409326103219984}, {-0.9892204918351276, -0.47305092325671594}, {-0.973715974960378, -0.4706619998823377}, {-0.9659244919132601, -0.46935034658892866}, {-0.9580300784999736, -0.4679536926816983}, {-0.9419811822240022, -0.4649259395353851}, {-0.909189990177837, -0.4580862609715699}, {-0.9009207598619717, -0.4562468084066721}, {-0.8926480979282837, -0.4543677035029081}, {-0.876142621397055, -0.45051581477659713}, {-0.8436381702032162, -0.4426195019944614}, {-0.8356886388894809, -0.44064411720348307}, {-0.8278335233424967, -0.4386817213176753}, {-0.8124526202192411, -0.43482314011589773}, {-0.7833176762407615, -0.42756140977879004}, {-0.7764342596822816, -0.42588027735588313}, {-0.7697308288278752, -0.42426619060103293}, {-0.7569009936814196, -0.42126522549158274}, {-0.750792227148599, -0.41989114546155965}, {-0.7448986928998332, -0.4186096830779266}, {-0.7337883452985065, -0.4163488984695419}, {-0.7285860120817226, -0.41538136109096074}, {-0.7236278482129629, -0.4145299876623667}, {-0.7189200110006084, -0.41380025960284295}, {-0.7144682064775127, -0.4131974736847735}, {-0.710277680551427, -0.41272673119781333}, {-0.7063532109767301, -0.4123929274083754}, {-0.7026991001597472, -0.4122007413358332}, {-0.6993191688084903, -0.4121546258661752}, {-0.696216750436097, -0.41225879822329853}, {-0.6933946867257517, -0.4125172308175733}, {-0.6908553237633177, -0.41293364249069087}, {-0.6886005091423864, -0.4135114901751679}, {-0.6866315899448884, -0.41425396098619827}, {-0.684949411598855, -0.4151639647628089}, {-0.6835543176133854, -0.4162441270745381}, {-0.6824461501893202, -0.41749678270905377}, {-0.6816242517025776, -0.4189239696553097}, {-0.6810874670555743, -0.4205274235959887}, {-0.6808341468906468, -0.42230857292209606}, {-0.6808621516578609, -0.42426853428165556}, {-0.6811688565281243, -0.4264081086735242}, {-0.6817511571410342, -0.4287277780963875}, {-0.6826054761754464, -0.43122770276200106}, {-0.6837277707293326, -0.4339077188807446}, {-0.6851135404940825, -0.43676733702653303}, {-0.6867578367070464, -0.4398057410870781}, {-0.6908000301781668, -0.4464140069097936}, {-0.6931858787492048, -0.4499806018549932}, {-0.6958061794483027, -0.4537194511429759}, {-0.7017216345390777, -0.4617038141900663}, {-0.7047744352334797, -0.465653165272161}, {-0.7080052919313249, -0.4697424144510406}, {-0.7149735664273583, -0.4783284498408829}, {-0.7307300744487161, -0.4970328579701644}, {-0.7677095879849654, -0.5396107387608439}, {-0.7726754557586294, -0.5453198528739416}, {-0.777687455344967, -0.551096233474392}, {-0.7878128332767262, -0.562826580822119}, {-0.8082088023948208, -0.5868214253498651}, {-0.8475190985741928, -0.6354631074252973}, {-0.8521193926625217, -0.6414710023833733}, {-0.8566214682088749, -0.6474397338572824}, {-0.8652985188912784, -0.6592321114899378}, {-0.8811277379514317, -0.6820458399229905}, {-0.88472470669639, -0.6875496574170797}, {-0.8881638323090131, -0.6929605943390634}, {-0.8945431234256831, -0.70347858766897}, {-0.897471235269515, -0.7085733306475491}, {-0.900217413402099, -0.7135505863461926}, {-0.9051429485174126, -0.7231295710544965}, {-0.9073125346252527, -0.7277201719091507}, {-0.9092806583603764, -0.732171047812268}, {-0.9110432020253871, -0.7364770804905131}, {-0.9125963651020882, -0.7406333459388145}, {-0.9139366690815325, -0.7446351228756709}, {-0.9150609617889162, -0.7484779009180988}, {-0.9159664211980725, -0.7521573884622682}, {-0.9166505587311233, -0.7556695202562782}, {-0.9171112220396344, -0.7590104646519574}, {-0.917346597264441, -0.7621766305230268}, {-0.9173552107721454, -0.7651646738374687}, {-0.9171359303670629, -0.7679715038724229}, {-0.9166879659782378, -0.7705942890604573}, {-0.9160108698219613, -0.7730304624566214}, {-0.9151045360410145, -0.7752777268162182}, {-0.9139691998226869, -0.7773340592738434}, {-0.9124798287668874, -0.7793462347427061}, {-0.910723005963421, -0.7811299719227006}, {-0.9087003767193546, -0.7826838178508082}, {-0.906414045262002, -0.7840067198105243}, {-0.9038665694384239, -0.7850980290388341}, {-0.9010609546170152, -0.7859575037113149}, {-0.8980006468047433, -0.7865853111969147}, {-0.8946895249949878, -0.7869820295752749}, {-0.8911318927622939, -0.7871486484108094}, {-0.8873324691216591, -0.7870865687790969}, {-0.8832963786712819, -0.7867976025425251}, {-0.8790291410389329, -0.7862839708734978}, {-0.8745366596533151, -0.7855483020248945}, {-0.8698252098629669, -0.7845936283488874}, {-0.8649014264263455, -0.7834233825665841}, {-0.8597722903978485, -0.7820413932923962}, {-0.8544451154354965, -0.7804518798183876}, {-0.8489275335570489, -0.7786594461652876}, {-0.8373531798191882, -0.7744861172863665}, {-0.8313131284357894, -0.7721162901079657}, {-0.8251160791937221, -0.7695656619616982}, {-0.8122871813943401, -0.7639479905511581}, {-0.8056739699799181, -0.7608947658481623}, {-0.7989410001115479, -0.7576883551053302}, {-0.7851550554398898, -0.7508469945394357}, {-0.7565855235856768, -0.735682587440806}, {-0.7492952966579989, -0.7316341337645668}, {-0.7419667690766663, -0.7275012270215695}, {-0.7272367167130555, -0.7190205545958944}, {-0.6977770482751783, -0.7014777150463148}, {-0.6411358840504762, -0.6663686865221914}, {-0.6344578595908273, -0.662180929116588}, {-0.6278992593659437, -0.6580722404145721}, {-0.6151735782883485, -0.6501323761922964}, {-0.591505281581904, -0.6356368115649803}, {-0.5860013420694283, -0.6323563399227787}, {-0.5806756426013123, -0.6292312795704431}, {-0.5705821730611398, -0.6234815433462606}, {-0.5658251027123359, -0.6208732638247771}, {-0.5612676557244584, -0.6184531568801986}, {-0.5527688834050503, -0.6142069320553882}, {-0.5490902140532541, -0.6125086225819386}, {-0.5455986468177607, -0.6109981387452833}, {-0.5422965145087552, -0.6096804431711642}, {-0.5391858421048961, -0.6085602123438901}, {-0.5362683448397443, -0.6076418283558206}, {-0.5335454267859173, -0.60692937109377}, {-0.5310181799373848, -0.6064266108760529}, {-0.5286873837895031, -0.6061370015531733}, {-0.526553505415612, -0.6060636740845032}, {-0.5246167000381837, -0.6062094306025337}, {-0.5228768120917674, -0.6065767389755631}, {-0.5213333767741627, -0.6071677278789115}, {-0.5199856220814892, -0.6079841823839796}, {-0.5188324713220698, -0.6090275400736558}, {-0.5178725461032653, -0.6102988876917803}, {-0.5171041697846863, -0.6117989583335371}, {-0.5165253713904627, -0.6135281291827935}, {-0.5161338899725431, -0.6154864198015767}, {-0.5159271794162956, -0.6176734909759938}, {-0.5159024136790119, -0.6200886441220264}, {-0.5160564924512344, -0.622730821253769}, {-0.5163860472301997, -0.6255986055157847}, {-0.516887447794047, -0.6286902222803267}, {-0.517556809064862, -0.6320035408093508}, {-0.5183899983480065, -0.6355360764802697}, {-0.5193826429346622, -0.6392849935735386}, {-0.5205301380539499, -0.6432471086192744}, {-0.5218276551604891, -0.6474188942991923}, {-0.5232701505427545, -0.6517964838992318}, {-0.5248523742371565, -0.6563756763074348}, {-0.5284140309474312, -0.6661204268629242}, {-0.53038201696893, -0.6712759632772141}, {-0.5324668570278582, -0.6766130727319087}, {-0.5369624003239039, -0.6878085992038206}, {-0.5470763995424919, -0.7121050662993744}, {-0.5702665280315131, -0.7666684018851719}, {-0.5733050627889998, -0.7738888421301382}, {-0.576350742785472, -0.7811687801168058}, {-0.5824349337302258, -0.7958702155839873}, {-0.5943768529689366, -0.8255755340015065}, {-0.5972746869694273, -0.8330091254064781}, {-0.6001239189766745, -0.8404263420215043}, {-0.6056503160860017, -0.8551729977644219}, {-0.615838124476559, -0.8840459821916121}, {-0.6181247364483013, -0.8909418495880737}, {-0.6203177085995417, -0.8977489000133958}, {-0.6244025402950395, -0.9110620858719393}, {-0.6262847414645246, -0.9175512964380017}, {-0.6280539990643833, -0.9239178661489048}, {-0.6312364317561957, -0.9362508620129322}, {-0.632641479694067, -0.942201611477673}, {-0.6339173401561805, -0.947998390932927}, {-0.6350604711149952, -0.9536338499007075}, {-0.6360675473140248, -0.9591008539360768}, {-0.636935464244826, -0.9643924957810565}, {-0.6376613417625989, -0.9695021062041848}, {-0.6382425273364006, -0.9744232645086575}, {-0.6386765989305682, -0.9791498086924504}, {-0.6389613675145225, -0.9836758452442885}, {-0.6390948791987541, -0.9879957585598892}, {-0.6390754169953663, -0.9921042199634091}, {-0.6389015022021531, -0.9959961963195999}, {-0.6385718954098075, -0.9996669582227885}, {-0.6380855971324305, -1.0031120877493842}, {-0.6374418480621153, -1.006327485761257}, {-0.6366401289489791, -1.0093093787479979}, {-0.635680160108583, -1.012054325196718}, {-0.634561900559283, -1.014559221478769}, {-0.6332855467926174, -1.016821307243454}, {-0.6318515311803988, -1.0188381703095237}, {-0.6302605200227576, -1.0206077510460205}, {-0.6285134112419102, -1.0221283462347504}, {-0.626611331726987, -1.0233986124074703}, {-0.6245556343357701, -1.0244175686516308}, {-0.6223478945597162, -1.0251845988793244}, {-0.6199899068591477, -1.0256994535548853}, {-0.6174836806759819, -1.0259622508773938}, {-0.6148314361318548, -1.0259734774151674}, {-0.612035599419954, -1.025733988190131}, {-0.6090987978993315, -1.0252450062107965}, {-0.6060238549008974, -1.0245081214534162}, {-0.6028137842547089, -1.023525289291691}, {-0.5994717845485833, -1.022298828376287}, {-0.5960012331284281, -1.0208314179662012}, {-0.5924056798510557, -1.0191260947148861}, {-0.5886888406006023, -1.0171862489148746}, {-0.5848545905799777, -1.0150156202054368}, {-0.5809069573890993, -1.0126182927486689}, {-0.5726883709546808, -1.0071615682414916}, {-0.5684261698575278, -1.0041120114016033}, {-0.5640680747429383, -1.0008554229768902}, {-0.55508302615079, -0.9937443213549317}, {-0.5361748032470607, -0.9773091996573192}, {-0.5312818924588397, -0.9727799078334505}, {-0.5263330605415845, -0.9680973518711282}, {-0.5162888925733895, -0.9583042608811316}, {-0.49576512829221775, -0.9372408549533666}, {-0.490132685844723, -0.9312443924209329}, {-0.48448567075133403, -0.925149448071851}, {-0.4731753999881903, -0.9127122112184919}, {-0.45067930548114216, -0.8871933276744909}, {-0.4076246408073839, -0.8364043593862512}, {-0.40254063188947475, -0.8303257613682408}, {-0.39754082702198945, -0.8243486146315202}, {-0.38781321377544253, -0.8127468195713274}, {-0.3695698169196841, -0.791278967049084}, {-0.36528462814258134, -0.7863386983056184}, {-0.3611167220278341, -0.7815903238074644}, {-0.35314484819045033, -0.7727091886114901}, {-0.33872736193085556, -0.7576689614744494}, {-0.3354526527330527, -0.754525575209986}, {-0.3323127761935904, -0.7516441991716665}, {-0.3293087578817384, -0.7490317342335587}, {-0.3264413613207722, -0.7466946200278167}, {-0.32371108863213927, -0.744638822882632}, {-0.32111818167328343, -0.7428698245704716}, {-0.3186626236638075, -0.7413926118878155}, {-0.3163441412937212, -0.7402116670861592}, {-0.31416220730662014, -0.7393309591725586}, {-0.31211604354977035, -0.7387539360964895}, {-0.3102046244822068, -0.7384835178382579}, {-0.30842668113110155, -0.7385220904125979}, {-0.3067807054858646, -0.7388715007995194}, {-0.30526495531862824, -0.7395330528128256}, {-0.30387745941900374, -0.7405075039150772}, {-0.3026160232302693, -0.7417950629861192}, {-0.3014782348734277, -0.7433953890506017}, {-0.3004614715448924, -0.7453075909682413}, {-0.29956290627292065, -0.747530228088862}, {-0.29877951501729083, -0.7500613118725518}, {-0.2981080840961354, -0.7528983084735495}, {-0.29754521792329774, -0.7560381422847987}, {-0.29708734703906414, -0.759477200438356}, {-0.296730736416651, -0.7632113382551682}, {-0.2964714940263806, -0.7672358856360267}, {-0.2963055796390832, -0.7715456543838525}, {-0.2962288138498964, -0.7761349464457721}, {-0.2962368873033079, -0.7809975630618013}, {-0.29632537010000654, -0.7861268148053497}, {-0.2964897213658528, -0.7915155324991457}, {-0.29672529896308747, -0.7971560789886065}, {-0.2970273693237222, -0.8030403617531383}, {-0.29736492266054665, -0.8087427653266589}, {-0.29775220103061023, -0.8146426525010564}, {-0.29865983599916274, -0.8270042493739613}, {-0.299172051609032, -0.8334499002931974}, {-0.29971772461542495, -0.8400609408186699}, {-0.3008929084209644, -0.8537440562041699}, {-0.30151413101738744, -0.8607979431414332}, {-0.302152248773545, -0.8679808707824316}, {-0.3034626533538992, -0.8826950364545388}, {-0.30610584927957457, -0.9132346563926691}, {-0.3067479187490426, -0.9210405172718356}, {-0.3073746656592819, -0.9288934832911568}, {-0.30856701039988876, -0.944697405481391}, {-0.3091251945814537, -0.9526265312847304}, {-0.3096532403660594, -0.9605591344450919}, {-0.31014764345346696, -0.9684842063496841}, {-0.3106049902276828, -0.9763907309546416}, {-0.31102196282774724, -0.984267701483818}, {-0.31139534401205426, -0.9921041371330113}, {-0.3117220218090249, -0.9998890997540358}, {-0.3119989939473602, -1.0076117104930211}, {-0.3122233720595214, -1.0152611663574316}, {-0.3123923856525377, -1.0228267566863336}, {-0.3125033858406574, -1.0302978794985413}, {-0.31255384883483533, -1.0376640576935043}, {-0.3125413791844937, -1.0449149550799024}, {-0.3124637127674534, -1.0520403922072497}, {-0.3123187195244218, -1.059030361976064}, {-0.31210440593485644, -1.0658750450023338}, {-0.31181891723155386, -1.0725648247126562}, {-0.3114605393517391, -1.079090302146505}, {-0.31102770062294427, -1.0854423104426987}, {-0.31051897318244287, -1.0916119289875983}, {-0.3099330741294629, -1.0975904972028894}, {-0.3092688664099246, -1.1033696279516243}, {-0.30852535943387965, -1.1089412205414841}, {-0.3077017094263475, -1.1142974733049773}, {-0.30679721951268485, -1.119430895736911}, {-0.3058113395401103, -1.1243343201700664}, {-0.3047436656374709, -1.1290009129707812}, {-0.30359393951577535, -1.1334241852368006}, {-0.30236204751250073, -1.13759800298053}, {-0.30104801938309284, -1.1415165967816272}, {-0.29965202684353043, -1.1451745708936238}, {-0.2981743818682631, -1.1485669117901165}, {-0.2966155347482131, -1.1516889961369252}, {-0.2949760719139788, -1.1545365981774685}, {-0.2932567135297471, -1.1571058965195042}, {-0.29145831086381047, -1.1593934803123014}, {-0.2895818434419752, -1.1613963548042185}, {-0.28762841599047023, -1.163111946271632}, {-0.2855992551753432, -1.1645381063110953}, {-0.2834957061456562, -1.1656731154875892}, {-0.28131922888809013, -1.1665156863327184}, {-0.2790713944008932, -1.1670649656876861}, {-0.2767538806953825, -1.1673205363868913}, {-0.27436846863347364, -1.1672824182790065}, {-0.271708031304926, -1.166909881400979}, {-0.2689725303389216, -1.1661939148140172}, {-0.266164607646327, -1.1651362832238725}, {-0.2632870210479089, -1.1637393508294742}, {-0.26034263698322657, -1.1620060776670589}, {-0.25733442304769016, -1.1599400148847285}, {-0.254265440375239, -1.1575452989549793}, {-0.25113883588436614, -1.1548266448347122}, {-0.2479578344053669, -1.1517893380840893}, {-0.2447257307068559, -1.1484392259575231}, {-0.23812169701685984, -1.140826722539405}, {-0.234756633446638, -1.1365787399726168}, {-0.2313541841389136, -1.1320467447343239}, {-0.22445123974422657, -1.1221651529858414}, {-0.22095784471247706, -1.116833978344355}, {-0.21744124776154686, -1.1112556027537894}, {-0.2103526679878102, -1.0993990327237644}, {-0.19605243127954475, -1.0732015668943569}, {-0.19247185967396455, -1.066204508860049}, {-0.18889579859045239, -1.059053641747778}, {-0.18177015314119246, -1.0443432816205291}, {-0.1677110874276959, -1.013670397150062}, {-0.1642533259200852, -1.0058231055050275}, {-0.1608237877735732, -0.9979327864075548}, {-0.15405935496009648, -0.9820809746475803}, {-0.14097218025388294, -0.950499515447022}, {-0.11695973833401582, -0.8910332832403959}, {-0.11418511401271456, -0.8842158048792421}, {-0.11146338841866484, -0.8775786677792479}, {-0.10617985472348486, -0.8648941030613069}, {-0.10361829765254445, -0.8588701413669411}, {-0.10111014400225347, -0.8530734106708354}, {-0.09625299884673487, -0.8422042683195268}, {-0.09390314145918359, -0.8371520917561284}, {-0.0916049609659535, -0.832367579963436}, {-0.08716045487388795, -0.8236368937527133}, {-0.08501222998014654, -0.8197071423366037}, {-0.0829118909845829, -0.8160778720376799}, {-0.07884978694467491, -0.8097478172830145}, {-0.07701399265331073, -0.8072272434590686}, {-0.0752151504390717, -0.8049895355824237}, {-0.0734519214367295, -0.803038516533997}, {-0.07172289805211238, -0.8013775622486586}, {-0.07002660786852669, -0.800009596321645}, {-0.06836151767140024, -0.7989370853180915}, {-0.06672603758227222, -0.7981620347932515}, {-0.06511852529306159, -0.7976859860298698}, {-0.0635372903913644, -0.7975100134980415}, {-0.061980598767372476, -0.7976347230417471}, {-0.06044667709287525, -0.7980602507951301}, {-0.058933717362685704, -0.7987862628304236}, {-0.057439881488726245, -0.7998119555382828}, {-0.055963305936929435, -0.8011360567401343}, {-0.054502106397053336, -0.8027568275310049}, {-0.053054382475468155, -0.8046720648501223}, {-0.05161822240094453, -0.8068791047754651}, {-0.05019170773347091, -0.809374826537282}, {-0.048772918066148856, -0.8121556572444641}, {-0.04735993571025367, -0.8152175773165483}, {-0.04595085035359667, -0.8185561266130055}, {-0.04454376368240085, -0.8221664112503675}, {-0.0417280805318069, -0.8301804879317085}, {-0.04031578831003935, -0.8345723942601507}, {-0.0388981121240124, -0.8392122827643744}, {-0.03603956252116636, -0.8492078789915481}, {-0.030182735711908187, -0.8718448774593038}, {-0.028680218287605645, -0.878005997368571}, {-0.027159447011750337, -0.8843496601236766}, {-0.024057789280443193, -0.8975458488856329}, {-0.01758049442741813, -0.9256878570301118}, {-0.01589799067838981, -0.9330256393095218}, {-0.014188443223595948, -0.940462198967304}, {-0.010685347724453093, -0.9555860539752328}, {-0.0033280074665649737, -0.9865106168850122}, {-0.001413415179946547, -0.9943137726105327}, {0.0005316326291358069, -1.0021218468674884}, {0.004513056967504637, -1.0177048085808913}, {0.012835756815358425, -1.0484037107788153}, {0.014989189602725633, -1.0559121637313464}, {0.017170605586725618, -1.063330818243031}, {0.02161444316404293, -1.077853161872599}, {0.030800831777800425, -1.1053284259916525}, {0.03310713910088618, -1.1116827489992838}, {0.035432674003532165, -1.1178680391342015}, {0.040136323170824766, -1.1296950017221175}, {0.04251173589558057, -1.1353190779064877}, {0.04490097913800036, -1.1407389548369273}, {0.049714961205933265, -1.1509341144876841}, {0.052136584075825554, -1.1556941869715462}, {0.05456581254923774, -1.1602196622986118}, {0.05700098660504441, -1.164503773257384}, {0.059440412058430925, -1.1685401063592642}, {0.06188236439808015, -1.1723226113333394}, {0.06432509270297719, -1.1758456100756438}, {0.06676682363098736, -1.1791038050395457}, {0.0692057654712054, -1.1820922870547084}, {0.07164011225196659, -1.1848065425629974}, {0.0740680478962496, -1.1872424602605554}, {0.07648775041611246, -1.1893963371361773}, {0.07889739613771092, -1.1912648838970217}, {0.08129516394834962, -1.1928452297736392}, {0.08367923955699205, -1.1941349266972276}, {0.08604781975957941, -1.1951319528429716}, {0.08839911670048092, -1.1958347155342888}, {0.09073136212141525, -1.196242053503775}, {0.09304281158915008, -1.1963532385076023}, {0.09533174869332958, -1.1961679762911164}, {0.09759648920581529, -1.195686406904345}, {0.09983538519295891, -1.1949091043671285}, {0.10204682907233062, -1.1938370756845542}, {0.10422925760548196, -1.1924717592143559}, {0.10638115581842575, -1.1908150223889407}, {0.10850106084166035, -1.1888691587956455}, {0.1105875656616651, -1.186636884619825}, {0.11263932277596246, -1.1841213344563193}, {0.11465504774400567, -1.181326056495794}, {0.11663352262631192, -1.178255007093428}, {0.1185735993044854, -1.1749125447282933}, {0.12047420267495572, -1.1713034233627329}, {0.12233433370948235, -1.1674327852119617}, {0.12415307237573216, -1.163306152934937}, {0.12592958041145771, -1.1589294212584769}, {0.12766310394608513, -1.1543088480474506}, {0.12935297596378392, -1.1494510448346558}, {0.13099861860238, -1.1443629668248485}, {0.1325995452827722, -1.1390519023881889}, {0.13566577241793862, -1.1277915670571559}, {0.1371305728231615, -1.1218584373479357}, {0.13854966016736198, -1.1157345792612199}, {0.13992302996118575, -1.1094287726860856}, {0.14125077795617555, -1.1029500578619669}, {0.14253310096515498, -1.096307721785791}, {0.14377029748222805, -1.0895112842558545}, {0.14611101572313093, -1.0754952619248284}, {0.14721564557375766, -1.0682957504628987}, {0.14827736499262645, -1.0609822541158405}, {0.15027540983083235, -1.0460553024502854}, {0.15121365580775825, -1.0384631857296434}, {0.15211283061170705, -1.0307997293770264}, {0.15379889385134088, -1.0153026274776074}, {0.15465379511347516, -1.0068280395299771}, {0.15546923872052942, -0.9983225770150989}, {0.15699001519387315, -0.9812759781452745}, {0.15965559315571914, -0.9474380397621649}, {0.16025838438000986, -0.9391139741165104}, {0.16084119289080395, -0.9308718660308256}, {0.1619585162816328, -0.9146879000186806}, {0.16408595517753677, -0.8838900126384776}, {0.16461417133074138, -0.8765916213121054}, {0.16514755883957635, -0.8694778773684406}, {0.16568950662620247, -0.8625603174642892}, {0.16624343466305722, -0.8558501195710143}, {0.16681278654925472, -0.8493580836481437}, {0.167401021967551, -0.8430946129978676}, {0.1680116090377391, -0.8370696963330267}, {0.1686480165826431, -0.831292890589766}, {0.16931370632316212, -0.8257733045146985}, {0.17001212501905558, -0.8205195830551273}, {0.17074669657237057, -0.8155398925793078}, {0.17152081411058503, -0.8108419069522514}, {0.1723378320666792, -0.8064327944911195}, {0.17320105827344642, -0.8023192058225708}, {0.17411374608942495, -0.7985072626628222}, {0.17507908657385451, -0.795002547539595}, {0.1761002007280568, -0.7918100944733416}, {0.17718013182059322, -0.7889343806334558}, {0.17832183781347286, -0.786379318983446}, {0.17952818390655684, -0.7841482519272579}, {0.18080193521715443, -0.7822439459671264}, {0.18214574961162108, -0.7806685873815579}, {0.183562170705519, -0.7794237789302081}, {0.185053621048652, -0.7785105375905789}, {0.1866223955109865, -0.7779292933296258}, {0.18827065488511746, -0.7776798889115285}, {0.19000041972058132, -0.777761580741014}, {0.1918135644049182, -0.7781730407397965}, {0.1937118115059244, -0.7789123592518492}, {0.19569672638908728, -0.7799770489714032}, {0.19776971212369224, -0.7813640498857516}, {0.19993200469053632, -0.783069735223129}, {0.2021846685036462, -0.7850899183941741}, {0.20452859225779602, -0.7874198609137063}, {0.20949287322651533, -0.7929873648496376}, {0.21211409664220165, -0.7962127745252336}, {0.21482830654643623, -0.7997236625099006}, {0.22053533244672763, -0.8075720047881966}, {0.22352748712665893, -0.8118933272046757}, {0.2266113028652465, -0.8164678935326791}, {0.2330504367788268, -0.8263395508241114}, {0.24697931818801971, -0.8486898245379701}, {0.2504220622757345, -0.8543471447192361}, {0.25393427868720475, -0.8601522415541707}, {0.2611593253572876, -0.8721664443895313}, {0.27634949032860995, -0.8975250725451196}, {0.30908269575374214, -0.9511615058192109}, {0.31333349182911535, -0.9579245151950988}, {0.3176079505066783, -0.9646633776544241}, {0.3262127716291283, -0.9780243943085797}, {0.34353661458607077, -1.003974134418375}, {0.34786790629666164, -1.0102383041994967}, {0.3521911049381261, -1.0163923347517996}, {0.36079651614785646, -1.0283296671198412}, {0.3777255912377508, -1.0504507617742722}, {0.3818745923052074, -1.0555620252873177}, {0.3859819047030734, -1.0604883053793477}, {0.39405490232533125, -1.0697534178226744}, {0.3980124031101301, -1.0740767631037507}, {0.40191186099584064, -1.0781841736899154}, {0.40952086010976935, -1.0857238092603665}, {0.413222676264366, -1.0891432122952307}, {0.4168510136568392, -1.0923210553874247}, {0.4204022106584219, -1.0952518373488884}, {0.4238726985630865, -1.0979304356970296}, {0.4272590084028622, -1.100352113956036}, {0.43055777764091263, -1.1025125283539976}, {0.43376575673002477, -1.104407733906339}, {0.4368798155242687, -1.106034189876965}, {0.43989694953180075, -1.1073887646095526}, {0.4428142859970091, -1.1084687397223834}, {0.4456290898004126, -1.1092718136611355}, {0.44833876916496257, -1.109796104605016}, {0.4509408811577182, -1.110040152722678}, {0.45343313697611903, -1.1100029217753458}, {0.4558134070083958, -1.109683800065603}, {0.4580797256580281, -1.1090826007313246}, {0.4602302959224736, -1.108199561385234}, {0.46226349371677894, -1.107035343101608}, {0.46417787193307786, -1.1055910287526467}, {0.4659721642273864, -1.103868120698035}, {0.4676452885255249, -1.1018685378322386}, {0.4691963502404341, -1.0995946119950477}, {0.47062464519361896, -1.09704908375189}, {0.4719296622339193, -1.0942350975513773}, {0.4732043215728909, -1.090886711248678}, {0.47433370425054083, -1.0872322622822803}, {0.475317944690143, -1.0832773101692788}, {0.4761574538714392, -1.0790279345835327}, {0.47685292075224206, -1.0744907235070766}, {0.4774053131496061, -1.069672760482899}, {0.47781587807528264, -1.0645816109937594}, {0.47808614152126067, -1.059225307993319}, {0.4782179076922742, -1.0536123366173231}, {0.47821325768323, -1.0477516181040571}, {0.47807454760064877, -1.0416524929547137}, {0.4778044061282667, -1.0353247033655284}, {0.47740573153812477, -1.0287783749650254}, {0.4768816881495305, -1.0220239978907224}, {0.4762357022394276, -1.0150724072409052}, {0.4754714574088293, -1.0079347629381732}, {0.47459288941105293, -1.0006225290423851}, {0.4736041804486334, -0.9931474525516637}, {0.47131426281308314, -0.9777570440083262}, {0.4700225921916186, -0.9698664234803097}, {0.4686398417259388, -0.9618623380674585}, {0.46562254655132174, -0.9455652342155672}, {0.46399921933283317, -0.9372982910936123}, {0.46230722853393463, -0.9289699862675237}, {0.45874166157695684, -0.9121824438081102}, {0.4510635079829365, -0.8784572466631168}, {0.4490688955907675, -0.8700723298078247}, {0.4470580257777618, -0.8617323563541086}, {0.4430157376630329, -0.8452391734894087}, {0.43504954510734356, -0.8133543714878693}, {0.4331247931705361, -0.8056841688169434}, {0.4312412256982284, -0.7981579196488859}, {0.4276263958063833, -0.7835821346013284}, {0.42120408513309804, -0.7566329777937167}, {0.4198070378441637, -0.7504118841743209}, {0.4185068081953344, -0.7444153675303636}, {0.4173098711626046, -0.7386517420138734}, {0.41622255025808086, -0.7331288786026511}, {0.41525100450756486, -0.7278541919307459}, {0.414401215663921, -0.72283462797567}, {0.4136789756823155, -0.7180766526228729}, {0.41308987448309303, -0.7135862411264506}, {0.41263928802760624, -0.7093688684833862}, {0.4123323667318304, -0.7054295007368487}, {0.4121740242420767, -0.7017725872224035}, {0.41216892659651616, -0.6984020537691727}, {0.41232148179560835, -0.695321296866247}, {0.41263582980385227, -0.6925331788028457}, {0.41311583300453375, -0.690040023788902}, {0.41376506712838557, -0.6878436150609614}, {0.4145702742146764, -0.6859771222833505}, {0.41554458264083305, -0.684398518474944}, {0.41669052379904237, -0.6831079623231359}, {0.41801033403930776, -0.6821051082883813}, {0.4195059487913303, -0.6813891084260213}, {0.4211789972057627, -0.6809586150828454}, {0.4230307973275218, -0.6808117844616592}, {0.42506235181289, -0.6809462810455998}, {0.42727434420119464, -0.6813592828724369}, {0.4296671357508658, -0.6820474876476018}, {0.4322407628486713, -0.6830071196832619}, {0.4349949349998828, -0.6842339376492718}, {0.4379290334061058, -0.6857232431204757}, {0.4410421101364314, -0.6874698899034315}, {0.44779976039888786, -0.6917124450582867}, {0.45144079333742054, -0.694195916679916}, {0.4552537259663413, -0.6969118799118788}, {0.4633846288257041, -0.7030120278354789}, {0.4815741591678666, -0.7176601608013707}, {0.5246711242447263, -0.7546854040349391}, {0.5305810707972938, -0.759839798864371}, {0.5365859420642459, -0.7650749071213695}, {0.5488522632714202, -0.7757421361939842}, {0.5742033384926228, -0.7975445904678224}, {0.5806663759549618, -0.8030255813237142}, {0.5871634429125252, -0.8084957027011755}, {0.600226591838527, -0.8193576027353664}, {0.6263929290374333, -0.8404526928938065}, {0.632893054529594, -0.8455311945818871}, {0.6393594144450864, -0.850510163758921}, {0.6521565020283739, -0.8601281847804562}, {0.6769700686110014, -0.8777283539406883}, {0.6829620297529954, -0.8817331068491996}, {0.6888530692670007, -0.8855622252929988}, {0.7003006932838771, -0.8926610758337142}, {0.7058418148431596, -0.8959154251892063}, {0.711251117966445, -0.8989633993563373}, {0.7216453356869362, -0.9044134949848773}, {0.7262873582715536, -0.9066498986582319}, {0.7307908120486081, -0.9086854834401813}, {0.7351504914778493, -0.9105160101962434}, {0.7393613839616712, -0.9121375576285191}, {0.7434186784621678, -0.9135465273074476}, {0.7473177738392721, -0.9147396481947615}, {0.7510542868957347, -0.9157139806520711}, {0.7546240601150859, -0.9164669199303301}, {0.7580231690791324, -0.9169961991362179}, {0.7612479295520805, -0.9172998916723506}, {0.7642949042187756, -0.9173764131490025}, {0.7671609090650844, -0.9172245227658871}, {0.7698430193889886, -0.9168433241633477}, {0.7723385754314757, -0.9162322657431522}, {0.7746451876168684, -0.9153911404598917}, {0.7767607413928587, -0.9143200850848203}, {0.7786834016610663, -0.9130195789447706}, {0.7804116167895768, -0.9114904421396014}, {0.7819441221995446, -0.9097338332424307}, {0.7832799435185965, -0.9077512464876896}, {0.7844183992944256, -0.9055445084528307}, {0.7853591032626254, -0.9031157742402907}, {0.7861019661635322, -0.9004675231670517}, {0.7866471971035035, -0.8976025539699068}, {0.7869953044567839, -0.8945239795352758}, {0.787147096304816, -0.8912352211630877}, {0.7871036804105852, -0.887740002374997}, {0.7868664637262887, -0.884042342277856}, {0.7864371514333844, -0.8801465484940173}, {0.7858177455147779, -0.8760572096706871}, {0.7850105428596758, -0.8717791875812165}, {0.784018132902337, -0.867317608831713}, {0.782843394796737, -0.8626778561870458}, {0.7814894941298562, -0.8578655595307912}, {0.7799598791770704, -0.8528865864742287}, {0.7782582767038567, -0.8477470326299932}, {0.7763886873187207, -0.8424532115664386}, {0.7743553803830406, -0.8370116444593036}, {0.7698160014789097, -0.8257123307808732}, {0.7673197601151116, -0.8198685675662712}, {0.7646794492399355, -0.8139050024822776}, {0.7589889352703032, -0.8016481852404863}, {0.7461369372801988, -0.7760310050576752}, {0.742655011621902, -0.769442876892904}, {0.7390791392380844, -0.7627973470312888}, {0.7316741618869224, -0.7493668151723663}, {0.7160596530978031, -0.7221785810178126}, {0.6833710964484987, -0.6683291007783433}, {0.6789652245262486, -0.661256369867358}, {0.6745980441464261, -0.6542681625820659}, {0.6660215894860841, -0.6405824504728612}, {0.6497891406767219, -0.6146210516330143}, {0.64598173900053, -0.6084730022647954}, {0.6422942774540994, -0.6024774023575492}, {0.6353170838013275, -0.5909721733562058}, {0.6232067757118694, -0.5700873544887418}, {0.6206132244466788, -0.5653427693889236}, {0.6182092605019717, -0.5607989470196674}, {0.6160023367288594, -0.5564602787336127}, {0.6139995761956726, -0.5523307646662745}, {0.6122077581827149, -0.5484140082155103}, {0.6106333047517232, -0.5447132112661857}, {0.6092822679164607, -0.5412311701658726}, {0.6081603174398011, -0.5379702724559654}, {0.6072727292816513, -0.534932494361263}, {0.6066243747208903, -0.5321193990395788}, {0.6062197101733651, -0.5295321355915682}, {0.6060627677267681, -0.5271714388295475}, {0.6061571464119621, -0.5250376298026584}, {0.6065060042290433, -0.5231306170743859}, {0.6071120509450876, -0.5214498987470081}, {0.6079775416791762, -0.5199945652262252}, {0.6091042712888896, -0.5187633027178667}, {0.610493569571038, -0.5177543974472355}, {0.6121462972879453, -0.5169657405903598}, {0.6140628430291102, -0.5163948339051363}, {0.616243120916595, -0.516038796049099}, {0.6186865691609361, -0.5158943695693267}, {0.6213921494728565, -0.515957928548812}, {0.6243583473345103, -0.5162254868924497}, {0.6275831731324096, -0.5166927072346996}, {0.631064164152622, -0.5173549104498704}, {0.6347983874372514, -0.5182070857449441}, {0.6387824434996424, -0.5192439013138617}, {0.6430124708941436, -0.5204597155312066}, {0.6474841516347231, -0.5218485886623385}, {0.6571329569017825, -0.5251203358657173}, {0.6622992232488889, -0.5269899520616651}, {0.6676854432240262, -0.5290061380616976}, {0.6790913761520576, -0.533449048018059}, {0.7042305029642877, -0.5437614163955956}, {0.7618669608436087, -0.5682376215099901}, {0.7690552351838157, -0.5712727421009433}, {0.7763093271143993, -0.5743198486257979}, {0.790978144804295, -0.580421192152573}, {0.8206947589332615, -0.5924526576304285}, {0.8281470486644177, -0.5953838997962076}, {0.835589249226109, -0.5982707559454881}, {0.8504045220649429, -0.6038846160172002}, {0.8794873429716603, -0.614292872740944}, {0.8865879237799702, -0.6166883225027409}, {0.8936015253520563, -0.618989440902083}, {0.9073307226124251, -0.6232866907413619}, {0.9140280977381061, -0.6252722924786084}, {0.9206020815174929, -0.6271425183372352}, {0.93334512465407, -0.6305179934918005}, {0.9394972646183996, -0.6320143545058685}, {0.9454922017088222, -0.6333775750469635}, {0.9513219928732937, -0.6346037734092321}, {0.9569789252448023, -0.6356893022664817}, {0.9624555287141265, -0.6366307532736758}, {0.9677445881527599, -0.6374249612589351}, {0.9728391552660096, -0.6380690080011746}, {0.977732560056688, -0.6385602255891003}, {0.9824184218804481, -0.6388961993580601}, {0.9868906600743992, -0.6390747704019324}, {0.9911435041412766, -0.6390940376579517}, {0.9951715034721267, -0.638952359563083}, {0.9989695365911184, -0.638648355281264}, {1.0025328199068593, -0.6381809055015688}, {1.00585691595529, -0.637549152808008}, {1.0089377411200788, -0.6367525016224452}, {1.0117715728171373, -0.6357906177227485}, {1.0143550561308, -0.6346634273390346}, {1.016685209889968, -0.6333711158315171}, {1.0187594321734224, -0.6319141259541652}, {1.0205755052343881, -0.630293155709037}, {1.022131599835318, -0.628509155796805}, {1.0234262789847839, -0.626563326669647}, {1.0244585010692988, -0.6244571151932895}, {1.025227622373822, -0.6221922109256178}, {1.0257333989856627, -0.6197705420198767}, {1.0259759880774626, -0.6171942707610492}, {1.0259559485659047, -0.6144657887445931}, {1.025674241143782, -0.6115877117072557}, {1.0251322276840413, -0.6085628740202152}, {1.0243316700154104, -0.605394322855323}, {1.0232747280702088, -0.6020853120357135}, {1.0219639574059334, -0.5986392955825078}, {1.0204023061032101, -0.5950599209698032}, {1.0185931110436854, -0.5913510221005658}, {1.0165400935724211, -0.5875166120164397}, {1.0142473545503485, -0.5835608753548832}, {1.0117193688032784, -0.5794881605673915}, {1.0059773887928498, -0.5710099612408361}, {1.0028392076887453, -0.5667013031426944}, {0.9994954925840626, -0.562298260691509}, {0.9922152471025111, -0.5532281216489751}, {0.9754665173757552, -0.5341707170972695}, {0.9708648275697287, -0.5292452246405904}, {0.9661125617482518, -0.5242658008706126}, {0.956188665939923, -0.5141666019915552}, {0.9349017721813332, -0.49355848224469545}, {0.9293320931362296, -0.48835284337571977}, {0.923681673131451, -0.4831366359728885}, {0.9121766020050744, -0.4726941087996666}, {0.8886213715743791, -0.4519175462073423}, {0.8415547660172847, -0.4119355664632788}, {0.8358700352769104, -0.40717763828340614}, {0.8302622632766509, -0.4024875290889358}, {0.8193159874820786, -0.3933266306928991}, {0.7987619817349799, -0.37598485871489257}, {0.7939550066802813, -0.37187316527457276}, {0.7892979780661542, -0.3678570268764864}, {0.7804667793523324, -0.3601221519156637}, {0.7649536852942954, -0.3459071446025016}, {0.7615655591192392, -0.34262623003927006}, {0.7583869061556516, -0.33945762410535857}, {0.7526824502340356, -0.33346217333452627}, {0.7501680690097701, -0.33063727042477886}, {0.7478859864117933, -0.32792855966029405}, {0.7458410800703491, -0.3253365361085729}, {0.7440378797627842, -0.32286150525610624}, {0.742480560107711, -0.32050358391534184}, {0.7411729337791492, -0.3182627014326252}, {0.7401184452516242, -0.31613860119335385}, {0.7393201650864325, -0.31413084242009676}, {0.738780784768491, -0.3122388022589943}, {0.7385026121023525, -0.31046167814929543}, {0.7384875671751533, -0.308798490470428}, {0.7387371788933982, -0.30724808546058724}, {0.7393086386080818, -0.3056919938985259}, {0.7401938786967424, -0.30426518304796096}, {0.7413932689714272, -0.3029655117631253}, {0.7429066259791992, -0.30179061833040377}, {0.7447332124099152, -0.3007379268125932}, {0.7468717374780749, -0.2998046537598967}, {0.7493203582795662, -0.2989878152721433}, {0.7520766821224115, -0.29828423439612795}, {0.7551377698288885, -0.2976905488413872}, {0.758500140004654, -0.29720321899721613}, {0.7621597742687748, -0.29681853623321647}, {0.7661121234368885, -0.2965326314652273}, {0.7703521146479466, -0.296341483968064}, {0.7748741594233616, -0.29624093041611166}, {0.7796721626456724, -0.2962266741324819}, {0.7847395324421927, -0.2962942945271417}, {0.790069190957513, -0.2964392567041632}, {0.7956535859970475, -0.2966569212180191}, {0.8014847035223304, -0.2969425539586846}, {0.807554080977187, -0.2972913361451506}, {0.8138528214223807, -0.29769837440687663}, {0.8203716084548899, -0.2981587109326345}, {0.8340300541555004, -0.29921918652841756}, {0.8411491274420806, -0.2998091796450141}, {0.848447111460525, -0.30043219956010675}, {0.8635348394506244, -0.3017568090718516}, {0.8953460049577289, -0.30457710820455874}, {0.9035675280893729, -0.305288277725252}, {0.9118709635323137, -0.3059919225016485}, {0.9286710010420665, -0.3073571981082969}, {0.9371409792813741, -0.30800930145500005}, {0.945639668370317, -0.3086348424168473}, {0.9626689425775097, -0.30978821301785725}, {0.9711723044916861, -0.3103073060272317}, {0.979649978335725, -0.31078237858906627}, {0.9880883545083681, -0.31120937150713407}, {0.9964738670735194, -0.31158436183259675}, {1.0047930175288624, -0.3119035693561355}, {1.0130323984853946, -0.3121633627578007}, {1.021178717215935, -0.3123602654045187}, {1.0292188190308498, -0.3124909607859729}, {1.037139710439564, -0.3125522975803672}, {1.044928582056624, -0.3125412943423747}, {1.0525728312116336, -0.31245514380641115}, {1.0600600842227936, -0.3122912167991707}, {1.0673782182942988, -0.3120470657562219}, {1.0745153829985146, -0.3117204278382755}, {1.0814600213045635, -0.3113092276436083}, {1.0882008901156484, -0.3108115795139533}, {1.0947270802783515, -0.3102257894320218}, {1.1010280360279636, -0.3095503565096812}, {1.1070935738349146, -0.3087839740666458}, {1.1129139006184197, -0.30792553030038594}, {1.1184796312944407, -0.3069741085487996}, {1.1237818056263418, -0.3059289871480202}, {1.1288119043477318, -0.30478963888855337}, {1.1335618645282477, -0.3035557300737436}, {1.1377348157609593, -0.3023189217165164}, {1.1416514395905384, -0.3009996865987789}, {1.145306324466619, -0.2995982022873795}, {1.1486944440975717, -0.29811478754845594}, {1.1518111654728214, -0.2965499003792613}, {1.1546522562915942, -0.29490413580285624}, {1.1572138917861252, -0.29317822343126876}, {1.1594926609282934, -0.29137302480310406}, {1.1614855720095676, -0.289489530501971}, {1.163190057585136, -0.28752885706246184}, {1.1646039787740459, -0.28549224367075876}, {1.1657256289081732, -0.28338104866727676}, {1.1665537365238392, -0.281196745859083}, {1.167087467690902, -0.2789409206501353}, {1.1673264276751614, -0.27661526599766567}, {1.1672706619309579, -0.2742215782033069}, {1.1669206564218564, -0.2717617525478236}, {1.1662773372683561, -0.26923777877855043}, {1.1653420697225993, -0.266651736458851}, {1.1641166564710888, -0.2640057901891158}, {1.1626033352674627, -0.2613021847090093}, {1.1608047758984197, -0.25854323989084965}, {1.1587240764868987, -0.2557313456341293}, {1.1563647591376578, -0.2528689566713348}, {1.1537307649314263, -0.24995858729533454}, {1.150826448274779, -0.2470028060186727}, {1.1442262935214524, -0.24096552047454037}, {1.1405411711665467, -0.2378893755198494}, {1.1366071421790918, -0.2347785262994606}, {1.1280179881729555, -0.22846376779162667}, {1.1233765812636511, -0.2252654326754357}, {1.1185136836344187, -0.22204353060408613}, {1.1081546142458196, -0.21554026539874835}, {1.085139205547692, -0.20237423346135605}, {1.0789592528004999, -0.19906563599792446}, {1.0726279164333266, -0.19575563695633122}, {1.0595510938544737, -0.1891420232951476}, {1.0320320468587214, -0.17601186293019225}, {1.0249304931227836, -0.17276381227189358}, {1.017762277584602, -0.16953420076536513}, {1.0032708436679165, -0.16313902215015547}, {0.973986662144048, -0.1506615332241669}, {0.9167733325088424, -0.1273376801678438}, {0.9091841928757391, -0.12428569959871338}, {0.9017380024123283, -0.121287467187711}, {0.8873325204294299, -0.11545584194688506}, {0.8608573651873199, -0.10446788933773128}, {0.8548017485637859, -0.10186299841297763}, {0.8489959787713984, -0.09931478375095426}, {0.838179424303596, -0.09438679746714897}, {0.8331901026524294, -0.09200585123539909}, {0.8284935127578693, -0.08967923871451755}, {0.8200154804920652, -0.0851850620017683}, {0.8162510635661699, -0.08301517799272867}, {0.8128133965369557, -0.08089499749171551}, {0.8097094697400831, -0.07882303894243223}, {0.8069456390943431, -0.07679770168844126}, {0.8045276138068886, -0.07481727172403412}, {0.8024604453036742, -0.07287992770578205}, {0.8007485174067244, -0.0709837472082803}, {0.7993955377774582, -0.06912671320705671}, {0.7984045306429045, -0.0673067207711258}, {0.7977778308191593, -0.06552158394720928}, {0.7975170790439627, -0.06376904281722864}, {0.7976232186277558, -0.06204677071030624}, {0.7980964934300635, -0.06035238155019749}, {0.7989364471654954, -0.05868343731879975}, {0.8001419240411104, -0.057037455616150375}, {0.801711070724338, -0.055411917297145036}, {0.8036413396380958, -0.05380427416508013}, {0.805929493577183, -0.05221195670203673}, {0.8085716116375101, -0.050632381816081685}, {0.8115630964471905, -0.04906296058527154}, {0.8148986826860274, -0.04750110597850933}, {0.8185724468774368, -0.04594424053341027}, {0.8225778184344132, -0.04438980397148095}, {0.8269075919387519, -0.042835260731118395}, {0.8315539406303295, -0.041278107399186234}, {0.8365084310809531, -0.0397158800222202}, {0.847305166316673, -0.03656658749377795}, {0.8721618605519477, -0.030104723607327785}, {0.8789898485786206, -0.028442764696340956}, {0.8860387986421859, -0.026758263927518018}, {0.9007469704414847, -0.023314504713854402}, {0.9322150336597108, -0.01608401086651682}, {0.9404245394122512, -0.014197117020393822}, {0.9487407983904695, -0.012276321219246482}, {0.9656313129423232, -0.008329660158709546}, {0.9741740132015394, -0.006302444554475256}, {0.9827604157531901, -0.004238630155845371}, {0.9913744539313147, -0.00213788124116669}, {0.999999999999999, -2.449293598294704*^-16}}]}, "Charting`Private`Tag$24676#2"]}}, {}, {{{}, {}, {}, {}}, {}}}, {}}, {DisplayFunction -> Identity, PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.02], Scaled[0.02]}}, PlotRangeClipping -> True, ImagePadding -> All, PlotInteractivity :> $PlotInteractivity, PlotRange -> {{Automatic, Automatic}, {Automatic, Automatic}}, PlotRangePadding -> Automatic, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, Axes -> True, AxesOrigin -> {0, 0}, CoordinatesToolOptions -> {"DisplayFunction" -> ({Sqrt[#1[[1]]^2 + #1[[2]]^2], Mod[ArcTan[#1[[1]], #1[[2]]], 2*Pi]} & ), "CopiedValueFunction" -> ({Sqrt[#1[[1]]^2 + #1[[2]]^2], Mod[ArcTan[#1[[1]], #1[[2]]], 2*Pi]} & )}, DisplayFunction :> Identity, FrameTicks -> {{Automatic, Automatic}, {Automatic, Automatic}}, GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]], Method -> {"AxisPadding" -> Scaled[0.02], "DefaultBoundaryStyle" -> Automatic, "DefaultGraphicsInteraction" -> {"Version" -> 1.2, "TrackMousePosition" -> {True, False}, "Effects" -> {"Highlight" -> {"ratio" -> 2}, "HighlightPoint" -> {"ratio" -> 2}, "Droplines" -> {"freeformCursorMode" -> True, "placement" -> {"x" -> "All", "y" -> "None"}}}}, "DefaultMeshStyle" -> AbsolutePointSize[6], "DefaultPlotStyle" -> {Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Directive[RGBColor[0.95, 0.627, 0.1425], AbsoluteThickness[2]], Directive[RGBColor[0.455, 0.7, 0.21], AbsoluteThickness[2]], Directive[RGBColor[0.922526, 0.385626, 0.209179], AbsoluteThickness[2]], Directive[RGBColor[0.578, 0.51, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.772079, 0.431554, 0.102387], AbsoluteThickness[2]], Directive[RGBColor[0.4, 0.64, 1.], AbsoluteThickness[2]], Directive[RGBColor[1., 0.75, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.8, 0.4, 0.76], AbsoluteThickness[2]], Directive[RGBColor[0.637, 0.65, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.915, 0.3325, 0.2125], AbsoluteThickness[2]], Directive[RGBColor[0.40082222609352647, 0.5220066643438841, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.9728288904374106, 0.621644452187053, 0.07336199581899142], AbsoluteThickness[2]], Directive[RGBColor[0.736782672705901, 0.358, 0.5030266573755369], AbsoluteThickness[2]], Directive[RGBColor[0.28026441037696703, 0.715, 0.4292089322474965], AbsoluteThickness[2]]}, "DomainPadding" -> Scaled[0.02], "RangePadding" -> Scaled[0.05]}, PlotRange -> {{-1.196316909917379, 1.1963520211246772}, {-1.1963532385076023, 1.1963510638183694}}, PlotRangeClipping -> True, PlotRangePadding -> {Scaled[0.02], Scaled[0.02]}, Ticks -> {Automatic, Automatic}}]"#,
    );
  }
  #[test]
  fn polar_plot_4() {
    assert_case(
      r#"PolarPlot[Cos[5t], {t, 0, Pi}]; PolarPlot[Abs[Cos[5t]], {t, 0, Pi}]; PolarPlot[{1, 1 + Sin[20 t] / 5}, {t, 0, 2 Pi}]; PolarPlot[Sqrt[t], {t, 0, 16 Pi}]"#,
      r#"Graphics[{{{{}, {}}, {}, {{{}, {}, Annotation[{Hue[0.67, 0.6, 0.6], Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Line[{{0., 0.}, {0.1241476332720891, 0.001914048598561522}, {0.1755086754747637, 0.005413103665331791}, {0.2148256065915677, 0.009942536892629528}, {0.2478527848551439, 0.015303292323708322}, {0.2768111530228325, 0.02137937575477584}, {0.30283435836927947, 0.02809167075236853}, {0.3265918730667346, 0.035381314990448796}, {0.3485168240163577, 0.04320197692630951}, {0.3689076012859915, 0.05151570222082658}, {0.3879794397373392, 0.060290452246505324}, {0.40589313329359145, 0.06949854002205637}, {0.4227721676518645, 0.07911558258730761}, {0.43871351270464654, 0.08911976968458847}, {0.45379472460146036, 0.09949133596081429}, {0.46807879052208057, 0.11021216947264909}, {0.4816175348174481, 0.12126551456182659}, {0.4944540759852566, 0.13263574192594269}, {0.5066246386228872, 0.14430816769679733}, {0.5181599156913748, 0.15626890901130683}, {0.5290861101887774, 0.1685047672538805}, {0.5394257437245097, 0.1810031326202226}, {0.5491982926186487, 0.1937519053453003}, {0.5671077553210235, 0.21995444108524112}, {0.5752724820237712, 0.2333860153344227}, {0.5829263530503604, 0.2470235334456513}, {0.5900795437616473, 0.26085664572656214}, {0.596741113602624, 0.27487524324876167}, {0.6029191632566866, 0.2890694328733522}, {0.6086209673613693, 0.30342951563925735}, {0.6138530872562771, 0.31794596800085323}, {0.6186214672995571, 0.33260942549378203}, {0.6229315175729215, 0.3474106684812035}, {0.6267881852410486, 0.362340609691477}, {0.6301960163988232, 0.3773902833056403}, {0.6331592098998331, 0.39255083539151325}, {0.6356816643901513, 0.4078135155126934}, {0.6377670195564838, 0.42316966936654743}, {0.6394186924251632, 0.4386107323266802}, {0.6406399094089775, 0.45412822378311635}, {0.6414337346854555, 0.46971374218828044}, {0.641803095397577, 0.4853589607293116}, {0.6417508040917368, 0.5010556235577609}, {0.64127957874495, 0.5167955425166182}, {0.6403920606811278, 0.5325705943121753}, {0.6390908306328126, 0.5483727180847031}, {0.6373784231683893, 0.5641939133374447}, {0.6352573396742399, 0.5800262381881949}, {0.6327300600555311, 0.5958618079118378}, {0.6297990532975026, 0.6116927937457726}, {0.626466787010583, 0.6275114219332384}, {0.6227357360668473, 0.6433099729822462}, {0.6186083904218167, 0.6590807811201631}, {0.61408726220399, 0.6748162339260534}, {0.6038738554108265, 0.7061508895277239}, {0.5981867669004902, 0.7217351331089327}, {0.5921162860452823, 0.7372541032016509}, {0.5788360334425016, 0.7680668930076846}, {0.5716318404587692, 0.7833461834647217}, {0.5640554194796403, 0.7985311430118246}, {0.5477977181995143, 0.828589620619114}, {0.5383763430665581, 0.8446934631885671}, {0.5285320942644136, 0.8606526552126482}, {0.507591505484668, 0.8921020677488107}, {0.4965036868356178, 0.9075749879461354}, {0.48501004127370445, 0.9228686600569781}, {0.4608234814869119, 0.9528846157283428}, {0.4481399464886556, 0.9675903029730653}, {0.4350693451283491, 0.9820835529835769}, {0.40778692414847917, 1.0104005803316654}, {0.34880337420540775, 1.0641371390208783}, {0.33316868868509936, 1.0769230722040282}, {0.3171898252105401, 1.0894348660719977}, {0.2842231410168776, 1.1136071959495824}, {0.2672473853178435, 1.1252535935805144}, {0.24995158012732574, 1.1365975789530047}, {0.21442515063256437, 1.1583513162231773}, {0.19620745839637388, 1.1687478703100571}, {0.17769557843799266, 1.178815620253286}, {0.13981626945851733, 1.1979396746389372}, {0.12046260184425371, 1.2069837815402267}, {0.10084226629375977, 1.2156746928096616}, {0.06083019983472732, 1.231973975507186}, {0.04045301135345727, 1.2395712071019738}, {0.019838236202259343, 1.2467929667964461}, {-0.0010066243875137772, 1.2536340289099273}, {-0.02207398062440714, 1.2600893084134477}, {-0.04335615616298395, 1.2661538625246742}, {-0.06484539016540883, 1.2718228922702133}, {-0.08653383941642621, 1.2770917440141176}, {-0.10841358048975216, 1.2819559109514855}, {-0.13047661196396115, 1.2864110345660755}, {-0.15271485668601448, 1.290452906050902}, {-0.1751201640806449, 1.2940774676908295}, {-0.19768431250385832, 1.2972808142062042}, {-0.22039901163886377, 1.300059194056617}, {-0.24325590493279658, 1.3024090107039141}, {-0.26624657207263386, 1.3043268238336156}, {-0.28936253149873986, 1.3058093505339323}, {-0.3125952429545227, 1.3068534664316063}, {-0.33593611007070595, 1.307456206783829}, {-0.3593764829827557, 1.3076147675255259}, {-0.3829076609800284, 1.3073265062713244}, {-0.4065208951852282, 1.3065889432715545}, {-0.4302073912627929, 1.3053997623216547}, {-0.45395831215484034, 1.3037568116243887}, {-0.47776478084333074, 1.3016581046043085}, {-0.5016178831371222, 1.2991018206739158}, {-0.5255086704826081, 1.2960863059510142}, {-0.5494281627966418, 1.29261007392676}, {-0.5733673513204751, 1.2886718060839517}, {-0.5973172014934365, 1.2842703524651164}, {-0.6212686558451064, 1.2794047321899873}, {-0.645212636904743, 1.2740741339219788}, {-0.6691400501267292, 1.2682779162832976}, {-0.6914588354385194, 1.2624450277608792}, {-0.7137478105983089, 1.256205525262063}, {-0.7582066043692982, 1.242506331227694}, {-0.7803615513497948, 1.2350467913539642}, {-0.8024569491614496, 1.2271809410229344}, {-0.8464393621482922, 1.2102320449776311}, {-0.8683115224695692, 1.201150192956726}, {-0.8900944281953426, 1.1916644177777833}, {-0.9333628683473161, 1.1714849170645938}, {-1.0185038479678115, 1.1263281160606573}, {-1.0394554964923186, 1.1140479674477892}, {-1.060259128455673, 1.1013749012473977}, {-1.101393402351485, 1.07485796505642}, {-1.1217096511179627, 1.0610183881470117}, {-1.1418491012480054, 1.046794478866335}, {-1.1815692040106034, 1.017203640554328}, {-1.201135754058242, 1.001842013570415}, {-1.2204973045049492, 0.9861066570795943}, {-1.258577678872045, 0.9535267247974258}, {-1.3319761649827282, 0.8840206123221197}, {-1.3497110626808029, 0.8657589063810921}, {-1.3671868378677048, 0.8471505631651838}, {-1.4013350369742514, 0.8089097722709654}, {-1.4179946286716651, 0.7892855175372555}, {-1.4343694373422207, 0.7693310093600567}, {-1.4662397858503657, 0.7284488687280573}, {-1.481723044877834, 0.7075303328114829}, {-1.4968969630558502, 0.6862997342074347}, {-1.526293043947412, 0.6429217420449217}, {-1.5811165452289409, 0.5526402548080013}, {-1.593962915223747, 0.5293661259761748}, {-1.6064546178623806, 0.5058214317295159}, {-1.6303530114138927, 0.45794300675873045}, {-1.6417494303408704, 0.4336208445746793}, {-1.652770641236758, 0.40905125100495804}, {-1.6736679548046862, 0.3591939236124334}, {-1.6835345643942747, 0.3339184897525602}, {-1.6930069828846717, 0.3084202211614018}, {-1.7107513890782642, 0.2567807253311795}, {-1.719014712132388, 0.2306524771509618}, {-1.7268665176707787, 0.20432734887852105}, {-1.7413194397298584, 0.1511132799866699}, {-1.7477875260252853, 0.12476678359888495}, {-1.7538451775138977, 0.09825706885615987}, {-1.7594889883152385, 0.07159071249916836}, {-1.7647156595664573, 0.04477435481841002}, {-1.7695220005290129, 0.017814697921751024}, {-1.7739049296677916, -0.009281496016893672}, {-1.7778614757023161, -0.0365074065272877}, {-1.781388778629724, -0.06385615663056687}, {-1.784484090719209, -0.09132081464444992}, {-1.7871447774776212, -0.1188943960055018}, {-1.7893683185859295, -0.14656986510799144}, {-1.7911523088062598, -0.17434013715888014}, {-1.7924944588592313, -0.20219808004846196}, {-1.7933925962713149, -0.23013651623618187}, {-1.7938446661919512, -0.258148224651154}, {-1.7938487321801733, -0.28622594260688655}, {-1.7934029769604871, -0.31436236772972753}, {-1.792505703147762, -0.3425501599005415}, {-1.7911553339409112, -0.37078194320911045}, {-1.7893504137851275, -0.39905030792076135}, {-1.787089609002467, -0.42734781245471715}, {-1.7843717083905672, -0.4556669853736533}, {-1.7811956237893052, -0.4840003273839525}, {-1.7775603906152055, -0.5123403133461434}, {-1.7734651683634113, -0.5406793942949976}, {-1.76890924107705, -0.5690099994687647}, {-1.7638920177838284, -0.5973245383470233}, {-1.7584130328996967, -0.6256154026966099}, {-1.7524719465994407, -0.6538749686251033}, {-1.746068545154056, -0.6820955986413284}, {-1.7318745741836583, -0.7383894453863551}, {-1.7240842102505272, -0.7664473377710781}, {-1.7158319427963233, -0.7944356497169002}, {-1.6979435073072329, -0.8501728336956283}, {-1.6883085629060615, -0.8779063557286009}, {-1.6782141624206874, -0.9055396015144749}, {-1.6566508439401182, -0.9604746065526042}, {-1.645184170328954, -0.9877610571938329}, {-1.6332625292982004, -1.0149166185701433}, {-1.608060235901223, -1.0688045909670454}, {-1.5522679570752016, -1.1746755521400607}, {-1.5372090031319448, -1.2007024260397872}, {-1.5217101607708603, -1.2265379707388682}, {-1.4894027453249734, -1.2776053432008856}, {-1.4725994514258334, -1.3028223893789659}, {-1.4553668265573796, -1.327818547322203}, {-1.4196255043902448, -1.3771190454861948}, {-1.4011230741643412, -1.4014089128623712}, {-1.382203845922368, -1.4254489503917716}, {-1.343128866354083, -1.4727510902801961}, {-1.260136615850415, -1.564047352974717}, {-1.2365428376301182, -1.5880007339876663}, {-1.2124953264807288, -1.6115941521324957}, {-1.1630616969903058, -1.6576671354317216}, {-1.1376872664538182, -1.6801299456086052}, {-1.1118824756085541, -1.702199288656321}, {-1.0590068805819672, -1.745125124373222}, {-1.0319489865490925, -1.765965647877801}, {-1.0044865491130364, -1.7863807714092474}, {-0.9483754831067767, -1.8259040626009941}, {-0.8316086312202998, -1.899513942355504}, {-0.8015133975653934, -1.9167416397970245}, {-0.7710721408880898, -1.9334854799662757}, {-0.7091833772539647, -1.9654947214612237}, {-0.677752097474336, -1.9807470218379652}, {-0.646007244523544, -1.9954892677384872}, {-0.5816106226007379, -2.023418907761188}, {-0.5489760500528181, -2.0365943132007067}, {-0.5160622923463795, -2.0492356911777643}, {-0.44943285822058165, -2.072893998375556}, {-0.4157352701311178, -2.083900121216054}, {-0.3817946683970512, -2.0943506078745315}, {-0.31322173244087154, -2.1135647609710673}, {-0.27860829615001015, -2.1223188679587923}, {-0.24378963664451256, -2.1304982235448575}, {-0.2087754343159486, -2.138098452591648}, {-0.17357545711983643, -2.1451153449453035}, {-0.1381995577441675, -2.151544857110593}, {-0.10265767074687066, -2.157383113877113}, {-0.06695980966306832, -2.1626264098962453}, {-0.03111606408296394, -2.1672712112083197}, {0.0048634032987915745, -2.1713141567194514}, {0.040968359661370524, -2.174752059627525}, {0.07718850506281097, -2.1775819087968262}, {0.11351347547095637, -2.1798008700808253}, {0.1499328458193815, -2.181406287592633}, {0.18643613308741053, -2.182395684922682}, {0.22301279940332494, -2.182766766303165}, {0.25965225516982493, -2.1825174177188287}, {0.29634386221082826, -2.1816457079636846}, {0.33307693693868035, -2.180149889643255}, {0.3698407535408271, -2.1780284001219683}, {0.4066245471849941, -2.1752798624153313}, {0.4434175172419252, -2.1719030860265445}, {0.48020883052473085, -2.1678970677272127}, {0.5169876245438579, -2.163260992281845}, {0.553743010776719, -2.15799423311584}, {0.5904640779510081, -2.1520963529266783}, {0.6271398953407021, -2.1455671042380504}, {0.6637595160737684, -2.138406429896682}, {0.700311980450595, -2.1306144635116127}, {0.7367863192721268, -2.122191529835726}, {0.7731715571767146, -2.1131381450893265}, {0.8456308180495531, -2.093143046137693}, {0.8792809678682402, -2.0829529122387154}, {0.9128159166089641, -2.0722170543825054}, {0.9795045946774551, -2.0491130503592916}, {1.0126405439710973, -2.036747710415368}, {1.045625736889913, -2.0238422588134353}, {1.1111084685103623, -1.996418234606375}, {1.1435883665751612, -1.981903632149543}, {1.1758822317568893, -1.966856857872822}, {1.2398768620383844, -1.9351763266853481}, {1.3652504874502867, -1.865542121479859}, {1.3959987153794144, -1.8468439160327441}, {1.4264918026268116, -1.827636332423922}, {1.486678775489708, -1.7877071267756675}, {1.5163558939870136, -1.7669929027634068}, {1.5457443421922965, -1.7457840952578378}, {1.6036222831106959, -1.7018990504269629}, {1.632095449177343, -1.6792313159708772}, {1.6602472961985588, -1.656086002071562}, {1.7155550696096558, -1.6083811302011843}, {1.821967030998598, -1.5074513083088468}, {1.847649307831607, -1.4810988850399562}, {1.8729481880682728, -1.4543092930454335}, {1.922366182515631, -1.399441253288415}, {1.9464707200351297, -1.3713744419962086}, {1.9701627117391964, -1.3428937323616719}, {2.0162808793425357, -1.284715232571068}, {2.0386931980282332, -1.2550300476662597}, {2.0606652604718714, -1.2249561720916675}, {2.1032619658155527, -1.163668839693411}, {2.1828848436732593, -1.0367275807552117}, {2.2015935847362558, -1.004123781271245}, {2.219811471125507, -0.971187147853198}, {2.2547514501837123, -0.9043453253313699}, {2.271462216166362, -0.8704553596320557}, {2.287659477511195, -0.8362630032269122}, {2.3184921373308582, -0.7670026276399003}, {2.3331171661375545, -0.7319505951736832}, {2.347207954149322, -0.696628141628166}, {2.373767443616166, -0.6252049243382269}, {2.3862267832695685, -0.5891208497626619}, {2.398133161459886, -0.5527997284314112}, {2.4202697504440738, -0.4794806158662171}, {2.431321668546929, -0.43939454778746884}, {2.4417042710405745, -0.39907164271407064}, {2.4514127268749877, -0.35852317138050777}, {2.460442396302389, -0.3177604995028051}, {2.4687888326915397, -0.27679508451049145}, {2.476447784285979, -0.23563847224467502}, {2.483415195905552, -0.19430229362321108}, {2.489687210590645, -0.15279826127393242}, {2.495260171188527, -0.11113816613691777}, {2.500130621881232, -0.06933387403681665}, {2.504295309654421, -0.027397322226227885}, {2.5077511857067023, 0.014659484098861962}, {2.510495406798867, 0.05682447531053248}, {2.512525336542567, 0.09908552088624685}, {2.5138385466279227, 0.14143043297099572}, {2.514432817989619, 0.18384696996319172}, {2.5143061419110273, 0.22632284012326193}, {2.5134567210659347, 0.26884570520387585}, {2.5118829704974583, 0.31140318410071816}, {2.5095835185337694, 0.35398285652271166}, {2.506557207640239, 0.39657226668060275}, {2.502803095207662, 0.43915892699281944}, {2.498320454276216, 0.48173032180747327}, {2.493108774194852, 0.5242739111393985}, {2.487167761215809, 0.5667771344211175}, {2.480497339023978, 0.6092274142665858}, {2.473097649200869, 0.6516121602465943}, {2.4649690516229286, 0.6939187726747068}, {2.4561121247939988, 0.7361346464025668}, {2.4465276661117215, 0.7782471746234407}, {2.4251804383812794, 0.8621117818951582}, {2.4134203600667767, 0.903838673364846}, {2.4009381314340814, 0.9454118518115306}, {2.3738150164677596, 1.0280468595556715}, {2.359178574302185, 1.0690836408199005}, {2.3438288696878806, 1.1099166206512847}, {2.311000964841304, 1.1909214134497375}, {2.2935289547441973, 1.2310684403905663}, {2.275356061466733, 1.2709621014756427}, {2.2369223888642664, 1.3499402554812543}, {2.151818875894972, 1.5043195096699227}, {2.1288536552686828, 1.5420989186643446}, {2.105222638904796, 1.5795283789170766}, {2.0559847775979483, 1.6532904919515572}, {2.0303892304519384, 1.6895998721655388}, {2.004150480575316, 1.7255127658269767}, {1.9497682208752483, 1.7961035426771261}, {1.921637639818578, 1.8307588948438716}, {1.8928897104652553, 1.8649727055636343}, {1.8335698506546692, 1.932031792849957}, {1.707841309092985, 2.060374845368875}, {1.6755820724526178, 2.0906387396809767}, {1.6427865247408053, 2.1203955377277417}, {1.575618731629205, 2.1783500124262183}, {1.5412630205631743, 2.2065290948566316}, {1.506404062944317, 2.234163898431675}, {1.4352112642458434, 2.2877649605466144}, {1.398895246379191, 2.3137137157148295}, {1.3621116238337483, 2.3390831908575698}, {1.28717889992751, 2.3880508978933594}, {1.1321211254828645, 2.4786760764989366}, {1.0923327757603833, 2.499764313466088}, {1.0521552425155456, 2.5202105154991}, {0.9706744445106891, 2.55914852663371}, {0.92939240852873, 2.577626619678073}, {0.8877636409410203, 2.5954352502486366}, {0.803509711853202, 2.6290186239698037}, {0.7609067412974495, 2.6447810673431578}, {0.7180014150690456, 2.6598494528814562}, {0.6748051056664232, 2.6742180902663266}, {0.6313292909086895, 2.6878814766072}, {0.5875855508300457, 2.700834298362145}, {0.5435855645396197, 2.713071433207012}, {0.45486404605842373, 2.7353791198069057}, {0.4101663387322564, 2.7454403990889036}, {0.3652600284150731, 2.7547674498815504}, {0.3201572413388325, 2.763356132135199}, {0.2748701832923066, 2.771202507113829}, {0.22941113626360993, 2.7783028388859226}, {0.18379245505534397, 2.7846535957591354}, {0.13802656387330187, 2.7902514516582735}, {0.0921259528896843, 2.7950932874460843}, {0.04610317478178846, 2.7991761921864167}, {-0.000029158752861700717, 2.802497464349289}, {-0.0462583805039661, 2.8050546129574365}, {-0.09257177127749568, 2.806845358673935}, {-0.13895656344534335, 2.807867634830485}, {-0.1853999445145797, 2.808119588395995}, {-0.23188906071530604, 2.807599580885083}, {-0.27841102060609585, 2.806306189206162}, {-0.32495289869599087, 2.8042382064487725}, {-0.3715017390820505, 2.801394642609853}, {-0.4180445591014096, 2.797774725258648}, {-0.4645683529967961, 2.7933779001399826}, {-0.5110600955944897, 2.788203831715641}, {-0.5575067459936608, 2.7822524036436036}, {-0.6038952512660425, 2.775523719194918}, {-0.6502125501648798, 2.7680181016080025}, {-0.6964455768420951, 2.759736094380183}, {-0.742581264572609, 2.7506784614963027}, {-0.7886065494847486, 2.7408461875942436}, {-0.8345083742956731, 2.730240478067225}, {-0.8802736920507462, 2.71886275910277}, {-0.9258894698657781, 2.706714677658224}, {-1.0166203669561305, 2.680115118415852}, {-1.058685311185006, 2.6666629083561197}, {-1.1005757242232583, 2.6525479009099238}, {-1.1837910721699596, 2.6223385101321015}, {-1.2250951356135553, 2.606249049768102}, {-1.2661829302055876, 2.5895066372626094}, {-1.347668338343654, 2.5540746190934063}, {-1.38804536174004, 2.5353912558563843}, {-1.4281649412143855, 2.5160674243066588}, {-1.5075910717151737, 2.475512637082608}, {-1.6629091872216284, 2.3868835825663592}, {-1.7009443259627641, 2.3631843428086547}, {-1.7386420254469295, 2.338877376418155}, {-1.8129862611535927, 2.288459648833501}, {-1.849613548728293, 2.2623589689263603}, {-1.885864904732761, 2.235670722952754}, {-1.9572021444100494, 2.1805533837050306}, {-1.9922693880729485, 2.1521355940247284}, {-2.026923425004503, 2.1231528431412396}, {-2.094955525658158, 2.0635167032483714}, {-2.2256664338552845, 1.937739742607478}, {-2.2571780386321745, 1.9049790873490406}, {-2.2882061548502173, 1.8717057062145586}, {-2.348778669786981, 1.8036495477293295}, {-2.3783067111871126, 1.7688814981911123}, {-2.4073185537604904, 1.7336301751789438}, {-2.463762156513274, 1.6617086124238205}, {-2.491178463830332, 1.6250541450266702}, {-2.518047670764401, 1.5879479453459326}, {-2.5701151988959277, 1.5124132658352685}, {-2.6673666427002156, 1.3562919160270985}, {-2.690205240483295, 1.3162598408355684}, {-2.712440887231286, 1.2758449256262259}, {-2.7550779241250236, 1.193903155984247}, {-2.775466960565797, 1.1523948546951401}, {-2.795228342024674, 1.110540815419293}, {-2.8328450010136303, 1.025833738917862}, {-2.8506890739504365, 0.9830000507526427}, {-2.8678830859154267, 0.9398593182441112}, {-2.9003001574326808, 0.8526964303115168}, {-2.915513212631754, 0.8086943482488849}, {-2.93005620106865, 0.7644253638839761}, {-2.957113673777134, 0.6751277446752277}, {-2.9706429793097526, 0.6263125520324562}, {-2.9833634065068013, 0.5772332978594045}, {-2.995269908899478, 0.5279034384890798}, {-3.0063576647561336, 0.4783365250272066}, {-3.016622078835885, 0.4285461995849415}, {-3.026058784078877, 0.3785461914795527}, {-3.0346636432326055, 0.3283503134041299}, {-3.0424327504137363, 0.27797245756740413}, {-3.0493624326048874, 0.22742659180476818}, {-3.0554492510858418, 0.17672675566159401}, {-3.0606900027986903, 0.12588705644995704}, {-3.065081721646414, 0.0749216652798838}, {-3.0686216797244437, 0.023844813066249127}, {-3.071307388484741, -0.027329213487542064}, {-3.073136599831974, -0.0785860759279464}, {-3.0741073071513836, -0.1299113881105959}, {-3.0742177462679368, -0.18129072027937626}, {-3.073466396336409, -0.23270960316410427}, {-3.071851980662033, -0.28415353209563227}, {-3.069373467451399, -0.33560797113720126}, {-3.0660300704932717, -0.38705835723085735}, {-3.061821249769055, -0.43849010435774194}, {-3.056746711992621, -0.48988860771105874}, {-3.050806411079255, -0.5412392478805165}, {-3.0440005485434942, -0.5925273950470424}, {-3.0363295738256344, -0.643738413186555}, {-3.027794184546734, -0.6948576642815817}, {-3.0183953266919343, -0.7458705125395038}, {-3.008134194721955, -0.7967623286162072}, {-2.9970122316126346, -0.8475184938439129}, {-2.972192826187678, -0.9485654758493234}, {-2.958499511745879, -0.9988271467575959}, {-2.9439536214864286, -1.0488948835432663}, {-2.912315095231374, -1.1483905835756931}, {-2.895228567720515, -1.1977896556151422}, {-2.877301680357455, -1.2469370195569482}, {-2.838941748948308, -1.3444193470711698}, {-2.8185167779353737, -1.3927258090821482}, {-2.797267591556131, -1.440723568246963}, {-2.7523153919172705, -1.5357366657273406}, {-2.6527408829748818, -1.7214431210995012}, {-2.62586567988377, -1.766889474381878}, {-2.5982101154335986, -1.8119165410187146}, {-2.540584306727321, -1.9006592225175718}, {-2.5106278409818836, -1.944348301077424}, {-2.479918567626235, -1.9875650283310071}, {-2.416271655046977, -2.07252958412899}, {-2.3833496033623196, -2.1142517892417145}, {-2.349705915330092, -2.155450403908118}, {-2.2802872232033518, -2.236227013545805}, {-2.1331717266334076, -2.3909564925885607}, {-2.0972972385169957, -2.4256957116872813}, {-2.0608566030395594, -2.4598955835008254}, {-1.986309560994093, -2.5266404974157175}, {-1.9482198638945323, -2.5591674554108406}, {-1.9095974345987339, -2.591118902773607}, {-1.8307894026521945, -2.6532605128279916}, {-1.7906216661801697, -2.6834336328725414}, {-1.7499569257861802, -2.7129971616181043}, {-1.6671736811422226, -2.7702628779992473}, {-1.4960602709673687, -2.8771290329884707}, {-1.4521836660345984, -2.9022045473336266}, {-1.407887921155603, -2.9266091408924693}, {-1.3180802881440112, -2.973377788199218}, {-1.2725893257754288, -2.995728350986159}, {-1.2267210700678197, -3.0173810148638855}, {-1.13389574154363, -3.0585674547372084}, {-1.0869604626152127, -3.078089051377377}, {-1.0396914729598754, -3.096888393885388}, {-0.944197047858916, -3.1322978263889714}, {-0.8959941918225187, -3.148897104421866}, {-0.8475027784897424, -3.164752507558499}, {-0.7497004196122989, -3.1942120045071927}, {-0.7004127528178061, -3.2078067043112855}, {-0.6508830806365136, -3.2206387440234443}, {-0.6011232387344785, -3.2327038812187383}, {-0.5511451361808146, -3.2439980576885508}, {-0.5009607525723048, -3.254517400696236}, {-0.4505821351367989, -3.264258224187771}, {-0.4000213958160016, -3.273217029957055}, {-0.3492907083284101, -3.2813905087655204}, {-0.2984023052131735, -3.288775541415681}, {-0.24736847485549254, -3.2953691997783414}, {-0.1962015584943403, -3.301168747773129}, {-0.14491394721329, -3.306171642302047}, {-0.09351807891508421, -3.3103755341357752}, {-0.042026435280741486, -3.3137782687524253}, {0.009548461285998249, -3.316377887128491}, {0.0611940507282419, -3.3181726264817457}, {0.11289773841870947, -3.3191609209658353}, {0.16464689825491888, -3.3193414023163395}, {0.21642887576604777, -3.3187129004480864}, {0.26823099123072613, -3.317274444003514}, {0.32004054280497074, -3.3150252608518804}, {0.37184480965942435, -3.311964778539147}, {0.42363105512522836, -3.308092624688362}, {0.4753865298476932, -3.3034086273503855}, {0.5270984749469267, -3.297912815304818}, {0.5787541251847425, -3.2916054183109957}, {0.6303407121370006, -3.284486867308928}, {0.6818454673705404, -3.2765577945700857}, {0.7332556256240129, -3.267819033797926}, {0.7845584279917656, -3.258271620178081}, {0.8867909803440217, -3.2367559824970384}, {0.9366953513538447, -3.2250339628711853}, {0.9864478636253262, -3.2125406713596942}, {1.0854495500708958, -3.185248129547107}, {1.1346749047259286, -3.170453270904383}, {1.1837007673655366, -3.154895923103007}, {1.281106709998342, -3.121504574658228}, {1.3294632268664415, -3.1036764409802013}, {1.3775731311952777, -3.0850975511262435}, {1.4730064345296447, -3.0457012485356887}, {1.6604031003348338, -2.9580609248929672}, {1.7064625123574269, -2.9343318310596}, {1.7521833398902307, -2.909884165071277}, {1.842564395293828, -2.858852598448934}, {1.887202374782343, -2.832278876973321}, {1.9314572785055717, -2.80500693965485}, {2.018774189752904, -2.748390681968779}, {2.061814564777845, -2.719057921465795}, {2.1044286044054217, -2.6890500625028975}, {2.1883353588242933, -2.627034024664931}, {2.350572544603273, -2.495184754976621}, {2.389910616299777, -2.460634106257715}, {2.4287402102388698, -2.425462566155284}, {2.5048348473652973, -2.3532869514399386}, {2.5420806111525014, -2.316298332397228}, {2.5787793432542183, -2.2787197297911916}, {2.6504984350370036, -2.201825146041796}, {2.6855004593855107, -2.162525819588389}, {2.719918786050874, -2.1226698153279537}, {2.7869690605088984, -2.0413226651179617}, {2.9136844766729193, -1.8723398135050113}, {2.9437764253948413, -1.8288380666425708}, {2.9732175982476448, -1.7848529743518133}, {3.0301167394237716, -1.695471908286051}, {3.0575596379551317, -1.6500958204121414}, {3.084321624976618, -1.6042761544591293}, {3.135774389258095, -1.5113471690776017}, {3.1604513168113275, -1.4642586758295295}, {3.1844196371090807, -1.4167682521644873}, {3.2302045025383013, -1.3206244723878617}, {3.2520084767156243, -1.2719928075774645}, {3.2730787051065193, -1.2230025900250365}, {3.312994603956914, -1.1239909781325534}, {3.333395268833137, -1.0697272512179432}, {3.3529051602158684, -1.015095484366706}, {3.371517677002755, -0.9601103347783837}, {3.3892264598543873, -0.9047865773869809}, {3.4060253932828854, -0.8491391008635484}, {3.4219086076742395, -0.7931829035818764}, {3.450905641923957, -0.6804048642976481}, {3.4640089691843756, -0.623613530753734}, {3.476175595782579, -0.5665744850600037}, {3.487400909445923, -0.5093032123770268}, {3.4976805544834373, -0.45181528265064946}, {3.5070104333273107, -0.3941263463511136}, {3.5153867080034864, -0.33625213018438616}, {3.5228058015308443, -0.2782084327769141}, {3.5292643992485075, -0.22001112033502362}, {3.534759450070811, -0.1616761222801098}, {3.539288167669509, -0.10321942686085497}, {3.542848031582786, -0.044657076743684035}, {3.545436788250693, 0.013994835417326029}, {3.5470524519766276, 0.07272017142982098}, {3.547693305814496, 0.13150275203064873}, {3.5473579023812434, 0.19032636137016865}, {3.54604506459441, 0.24917475151210997}, {3.5437538863344575, 0.3080316469477812}, {3.5404837330315573, 0.3668807491233385}, {3.5362342421766195, 0.42570574097885444}, {3.5310053237563244, 0.4844902914979231}, {3.5247971606119397, 0.5432180602664953}, {3.517610208721757, 0.6018727020397244}, {3.5094451974069636, 0.660437871315506}, {3.5003031294608213, 0.7188972269133986}, {3.4901852812010254, 0.7772344365576914}, {3.4790932024451404, 0.8354331814632978}, {3.4670287164090436, 0.8934771609231903}, {3.453993919528312, 0.9513500968960856}, {3.439991181202526, 1.0090357385930557}, {3.4250231434624676, 1.0665178670618218}, {3.392203098482262, 1.180806895167806}, {3.374357734385431, 1.2375815572834594}, {3.3555603559561127, 1.2940882402591618}, {3.3151258151100653, 1.4062337633000659}, {3.2934974538707382, 1.4618408032024708}, {3.270934678835287, 1.5171162726896765}, {3.223026424595739, 1.6266096690471255}, {3.1976918755149684, 1.6807963778673467}, {3.1714447704582227, 1.734589089100372}, {3.1162376362725346, 1.8409310485143944}, {3.08729062739924, 1.8934498035556642}, {3.057457100035035, 1.9455135834951074}, {2.995159349474417, 2.0482163910064477}, {2.9627101882600635, 2.0988257905357335}, {2.9294046289040185, 2.1489209674392575}, {2.860257181390435, 2.2475107455720167}, {2.712060692238789, 2.4378901285325076}, {2.6756405138643404, 2.4809218485879305}, {2.638536827594157, 2.5234089420291643}, {2.562311725694776, 2.6067059101001098}, {2.5232071374097127, 2.647494423820127}, {2.483452692196032, 2.6876955945769714}, {2.402029737788466, 2.7662946018753054}, {2.360379392966271, 2.804672121281729}, {2.318115515894705, 2.8424216686925967}, {2.2317852576505914, 2.915997745801402}, {2.0521900365670525, 3.055192293097292}, {2.005905615696137, 3.0882795300871115}, {1.9590878025620548, 3.1206644609664695}, {1.863894841800072, 3.183293187392272}, {1.8155414588241063, 3.2135202831984744}, {1.7666982078887021, 3.243011677828221}, {1.6675870884640096, 3.299755809870768}, {1.6173420311489357, 3.3269932007320056}, {1.5666527229460232, 3.353464201415302}, {1.4639883181190312, 3.404078276273853}, {1.4120369947864786, 3.428207422690921}, {1.3596889616152072, 3.451542327185155}, {1.2538515333822913, 3.4958035761501574}, {1.2003867852127088, 3.5167174719795518}, {1.146574615218266, 3.536812232028559}, {1.0379583984579535, 3.574521545334529}, {0.9831797798875804, 3.5921251838515538}, {0.9281045898984501, 3.6088878601724352}, {0.8171163160522257, 3.63987066325607}, {0.76122934596414, 3.654081458189479}, {0.7050980258184738, 3.6674326299576703}, {0.6487356021379345, 3.679920018269456}, {0.5921553916651127, 3.6915396673357956}, {0.5353707781995677, 3.7022878270458563}, {0.4783952094153837, 3.7121609540940637}, {0.4212421936599216, 3.7211557130578283}, {0.3639252967345363, 3.729268977425632}, {0.30645813865803156, 3.736497830575175}, {0.24885439041362858, 3.742839566701303}, {0.19112777068022763, 3.7482916916934226}, {0.1332920425487487, 3.752851923962164}, {0.07536101022434166, 3.7565181952150186}, {0.017348515715257577, 3.7592886511807273}, {-0.04073156449081222, 3.761161652282185}, {-0.09886532276216933, 3.7621357742576573}, {-0.15703882366940758, 3.762209808730084}, {-0.215238107341897, 3.7613827637243196}, {-0.27344919283358066, 3.759653864132094}, {-0.3316580814972091, 3.7570225521245564}, {-0.3898507603663221, 3.7534884875122403}, {-0.4480132055440564, 3.749051548052315}, {-0.5061313855979981, 3.7437118297029985}, {-0.5641912649602473, 3.7374696468250117}, {-0.6270439525778473, 3.7296846599434197}, {-0.6897937724574885, 3.720840895449115}, {-0.7524228600245455, 3.710939624160731}, {-0.8149133646116732, 3.6999824187848485}, {-0.8772474545675393, 3.6879711538026516}, {-0.939407322365522, 3.6749080052700265}, {-1.063133312645611, 3.6456362678450027}, {-1.1246639866475132, 3.6294335359264793}, {-1.1859495517252319, 3.6121906333994027}, {-1.3077149683069111, 3.574599326678964}, {-1.3681597682163757, 3.5542591731537745}, {-1.428289366136511, 3.532895348659651}, {-1.5475335859610022, 3.4871164494026146}, {-1.6066137150791908, 3.462711991860867}, {-1.6653096666423388, 3.4373050954099473}, {-1.7814810052404786, 3.3835084315075923}, {-1.8389226183457432, 3.3551316097193604}, {-1.8959125158794734, 3.3257782372881013}, {-2.008470787211587, 3.264170878152251}, {-2.06400626336635, 3.2319321166030144}, {-2.119024238295618, 3.198747250994869}, {-2.2274432727070663, 3.1295727245864695}, {-2.4373705392386684, 2.980254414106105}, {-2.488321995068248, 2.9406947825119643}, {-2.5386308597318683, 2.9002631173285884}, {-2.6372612332796277, 2.8168257405334662}, {-2.685553380827994, 2.773841693482311}, {-2.7331442232765264, 2.730028936336323}, {-2.8261652551929455, 2.6399633637942137}, {-2.871567548657882, 2.593734192629696}, {-2.9162127533712985, 2.546723593616324}, {-3.0031782746737323, 2.450408008701234}, {-3.167446055270657, 2.2489613585550496}, {-3.2064265558531315, 2.196841160377608}, {-3.2445482464690074, 2.1440451616116425}, {-3.3181685673498946, 2.0364826566211023}, {-3.353644471622126, 1.9817451013467235}, {-3.3882161209140196, 1.9263896395635374}, {-3.454603869761501, 1.8138850299399425}, {-3.4863992000932917, 1.7567663626920398}, {-3.517248743437349, 1.6990907418298016}, {-3.5760717414590495, 1.5821315512603187}, {-3.604026486685712, 1.5228798591842947}, {-3.6309980318996384, 1.4631349600321488}, {-3.6819570454044244, 1.3422310561509276}, {-3.705498726677238, 1.2822268121591363}, {-3.728061065536848, 1.221810796229015}, {-3.7702192493941773, 1.099807610223405}, {-3.7898015185575806, 1.0382528289744581}, {-3.8083772979013055, 0.9763510448050238}, {-3.842485173594334, 0.8515725068506084}, {-3.8580058409647067, 0.788729036948316}, {-3.8724971642446895, 0.7256051229747149}, {-3.8859541149011982, 0.6622176510633458}, {-3.898371941844316, 0.5985835968897654}, {-3.909746173011738, 0.53472002110607}, {-3.9200726168779823, 0.4706440647475731}, {-3.929347363887863, 0.40637294461284607}, {-3.937566787813782, 0.3419239486183493}, {-3.9447275470363783, 0.2773144311289367}, {-3.950826585748101, 0.21256180826552915}, {-3.955861135079318, 0.14768355319120124}, {-3.9598287141465693, 0.08269719137693643}, {-3.962727131022602, 0.017620295848363693}, {-3.964554483627842, -0.047529517585201055}, {-3.965309160542991, -0.1127345951181367}, {-3.964989841742443, -0.1779772497510448}, {-3.963595499248249, -0.24323976610708384}, {-3.9611253977043708, -0.30850440526076617}, {-3.95757909487099, -0.37375340957792424}, {-3.9529564420386705, -0.4389690075655499}, {-3.947257584362168, -0.5041334187301498}, {-3.940482961113738, -0.569228858443256}, {-3.9326333058557865, -0.6342375428127828}, {-3.923709646532738, -0.6991416935589233}, {-3.9137133054820312, -0.7639235428932127}, {-3.902645899364151, -0.8285653383993817}, {-3.890509339011654, -0.8930493479147195}, {-3.877305829197142, -0.9573578644105701}, {-3.8630378683201814, -1.0214732108705904}, {-3.8477082480131735, -1.0853777451654802}, {-3.813876658870949, -1.2124840123905414}, {-3.7953817347836933, -1.275650679293011}, {-3.7758392394075684, -1.3385364116779102}, {-3.7336288201644727, -1.4633955577112252}, {-3.7109702609498942, -1.5253343785427105}, {-3.687282857752552, -1.5869230888326247}, {-3.63684340287643, -1.7089818207793477}, {-3.610103003783607, -1.7694178765344766}, {-3.582357063243753, -1.829435899418696}, {-3.5238749613910003, -1.9481509523109541}, {-3.4931526984303938, -2.0068147948637556}, {-3.46145268746297, -2.064994238930968}, {-3.395150246988744, -2.1798347993802576}, {-3.3605639093856277, -2.236463655434352}, {-3.325032003702579, -2.2925436014228184}, {-3.251166617463746, -2.402993685679312}, {-3.0924901298832808, -2.6166210845684534}, {-3.0534485328030794, -2.6649540588211424}, {-3.013650868412223, -2.712704092947418}, {-2.931822371454544, -2.8064081285080045}, {-2.8898095172720364, -2.8523388525685296}, {-2.8470765485961143, -2.897640087535528}, {-2.759488211979116, -2.9863090441796016}, {-2.71465225726464, -3.0296546001152405}, {-2.669135010201514, -3.072326341009183}, {-2.5760973593898524, -3.1556056775567924}, {-2.382303779956817, -3.3136175692392347}, {-2.332310622393167, -3.351281371402637}, {-2.281721862558806, -3.3881900752511465}, {-2.1788033619516782, -3.459704684362379}, {-2.126496903842673, -3.4942922724736842}, {-2.0736414035083115, -3.5280881326362477}, {-1.966331410830308, -3.5932699973306286}, {-1.9119013286250361, -3.6246391234172606}, {-1.856971018901187, -3.6551827690058367}, {-1.7456599849772345, -3.7137619209727344}, {-1.5175950816976418, -3.8206765702828585}, {-1.4595210716255402, -3.8452294473351207}, {-1.401050506211059, -3.868898514395114}, {-1.282973683749543, -3.913559851515093}, {-1.2233946679382715, -3.934539953478141}, {-1.1634735729494474, -3.9546119125110977}, {-1.0426606772775562, -3.9920093680410202}, {-0.9817968646601327, -4.009324378680596}, {-0.9206469424898833, -4.025710277540338}, {-0.8592251122659944, -4.0411623590660435}, {-0.7975456525388133, -4.055676135056649}, {-0.7356229155789104, -4.069247335920692}, {-0.6734713240257898, -4.08187191188164}, {-0.6111053675169793, -4.0935460341317915}, {-0.5485395992982283, -4.104266095934422}, {-0.48578863281566814, -4.114028713673859}, {-0.4228671382907955, -4.122830727853197}, {-0.3597898392790236, -4.130669204039365}, {-0.2965715092125559, -4.137541433755302}, {-0.2332269679284514, -4.143444935318949}, {-0.1697710781827616, -4.148377454628843}, {-0.10621874215150202, -4.152336965896052}, {-0.042584897919226046, -4.15532167232227}, {0.021115484043910476, -4.157330006723833}, {0.08486740441570127, -4.158360632101478}, {0.1486558385652804, -4.158412442155656}, {0.2124657401055715, -4.157484561747256}, {0.2762820444538942, -4.155576347303538}, {0.3400896723998231, -4.152687387169191}, {0.40387353367951073, -4.148817501902342}, {0.4676185305556855, -4.143966744515424}, {0.5366951709147746, -4.137597065132399}, {0.6056890591116826, -4.130074533668194}, {0.6745809318525272, -4.121400146366496}, {0.743351536213701, -4.111575221712985}, {0.8119816350382867, -4.100601400363447}, {0.8804520123328281, -4.088480644981438}, {0.9487434786630116, -4.07521523998547}, {1.0168368765468148, -4.06080779120566}, {1.084713085843468, -4.045261225449931}, {1.1523530291367858, -4.028578789979744}, {1.2197376771113315, -4.010764051895486}, {1.2868480539198852, -3.9918208974315594}, {1.3536652425407678, -3.971753531161318}, {1.4201703901233738, -3.9505664751119975}, {1.5521695036058853, -3.9048529631154087}, {1.6176261325757266, -3.880337129269837}, {1.6826960572320835, -3.854722847451453}, {1.8116020802075778, -3.800223621695113}, {1.87540156683431, -3.771351792809048}, {1.9387411361817024, -3.7414077429481196}, {2.0639684899268222, -3.6783325821515604}, {2.125820554513516, -3.6452170295392667}, {2.1871412724042503, -3.6110603687958203}, {2.308118644028329, -3.5396581300803653}, {2.54293490911801, -3.3847552704990065}, {2.6000530888907933, -3.34356603617125}, {2.656503711106027, -3.301412068535984}, {2.7673372302405195, -3.2142535195178956}, {2.821688044991214, -3.169271417923804}, {2.8753071471364398, -3.1233695368060226}, {2.98028810329473, -3.028854353612101}, {3.0316193920666286, -2.980265666981267}, {3.082157846646643, -2.9308064253039503}, {3.1807973738679096, -2.82932832680421}, {3.367926840296773, -2.6165118676010093}, {3.412513970988623, -2.5613351328559943}, {3.456196248897524, -2.5053982645895485}, {3.540794639573544, -2.391303769010162}, {3.581685563139764, -2.3331765111576517}, {3.6216212629966122, -2.2743498506032624}, {3.698579395624323, -2.1546613863910937}, {3.735578681044091, -2.0938316225468365}, {3.7715764549805626, -2.0323665270297906}, {3.840524110491589, -1.9075965592601023}, {3.9659397800483465, -1.6511712744669742}, {3.992758613509824, -1.5901068317846927}, {4.018636224873413, -1.5286072549974414}, {4.067539799207927, -1.404360908404968}, {4.090552333904442, -1.3416435250974221}, {4.112596793589649, -1.2785497737098341}, {4.1537571462443115, -1.1512931042191092}, {4.172861448564981, -1.0871604014719958}, {4.190974497774758, -1.0227117541617556}, {4.208091085839161, -0.9579625003257713}, {4.224206242321813, -0.8929280635803002}, {4.239315235786392, -0.827623949435745}, {4.253413575141387, -0.7620657415890254}, {4.2664970109272655, -0.6962690981939387}, {4.2785615365456975, -0.630249748110404}, {4.289603389430507, -0.5640234871334967}, {4.299619052159997, -0.4976061742031704}, {4.308605253510342, -0.4310137275955868}, {4.316558969449753, -0.3642621210969677}, {4.323477424073109, -0.29736738016088937}, {4.3293580904767905, -0.23034557804995007}, {4.334198691573464, -0.16321283196273936}, {4.33799720084655, -0.09598529914704622}, {4.340751843044162, -0.02867917300024567}, {4.342461094812301, 0.038689320842192655}, {4.343123685267091, 0.10610392842912643}, {4.342738596505878, 0.17354837142614246}, {4.341305064057021, 0.2410063510569846}, {4.338822577268212, 0.3084615520530507}, {4.335290879633203, 0.3758976466100002}, {4.330709969056775, 0.44329829835050855}, {4.325080098057878, 0.5106471662922064}, {4.318401773910825, 0.5779279088198366}, {4.310675758724451, 0.6451241876606603}, {4.301903069459194, 0.7122196718621445}, {4.292084977882029, 0.7791980417709599}, {4.281223010459215, 0.8460429930123153}, {4.269318948186852, 0.9127382404686609}, {4.256374826359234, 0.9792675222567813}, {4.242392934274996, 1.0456146037023104}, {4.227375814881101, 1.111763281310691}, {4.211326264354688, 1.17769738673361}, {4.194247331622849, 1.2434007907299287}, {4.176142317820401, 1.3088574071201464}, {4.157014775685733, 1.3740511967334168}, {4.115707571333671, 1.5035863976112487}, {4.093536266308849, 1.5678960009769474}, {4.070359145697046, 1.6318791695944013}, {4.021006902536713, 1.7588032920621486}, {3.994842118079117, 1.821712970719623}, {3.9676921920879176, 1.8842336719637391}, {3.910460277007765, 2.0080464673315115}, {3.880390572860721, 2.0693079424586083}, {3.849360294453655, 2.13011920976332}, {3.7844452143239735, 2.250330925752979}, {3.643400527500767, 2.4846955778180737}, {3.6026688676180956, 2.5466977539164564}, {3.560881198306027, 2.608031904012031}, {3.4741819901878035, 2.7286243095049048}, {3.4292932952914414, 2.7878471119345463}, {3.3833942727757367, 2.8463309928550427}, {3.288614230663579, 2.961013199968594}, {3.2397584371815435, 3.0171776378775697}, {3.1899427615971025, 3.0725353877242703}, {3.087485369097913, 3.180765384190168}, {2.8716571744821735, 3.386837907406555}, {2.8155110605197544, 3.4361019059652604}, {2.7585193663360235, 3.4844343367213115}, {2.6420613709151186, 3.5782466734790557}, {2.582626755679879, 3.6236983160962004}, {2.5224099235922512, 3.6681618726388794}, {2.3996956112572554, 3.7540711397632474}, {2.3372317072194124, 3.795490745232439}, {2.2740527295438695, 3.8358700623088313}, {2.145619135579591, 3.9134587361144098}, {1.8809481333367541, 4.055628986862252}, {1.813260818824862, 4.088397053073632}, {1.7450028621246239, 4.120034365527751}, {1.6068508296157378, 4.1798773158895965}, {1.5369950841624338, 4.208064041465141}, {1.4666453465712836, 4.235082194536304}, {1.32454231714329, 4.2855785163460425}, {1.2528286090104337, 4.309040374306239}, {1.1807000650944859, 4.331301043396587}, {1.1081768249186024, 4.3523532023770795}, {1.0352791569708217, 4.372189868013133}, {0.9620274530254143, 4.390804397342992}, {0.8884422224245874, 4.408190489849616}, {0.8145440863220091, 4.42434218953636}, {0.7403537718896332, 4.439253886905834}, {0.6658921064897504, 4.452920320841214}, {0.5911800118136763, 4.465336580389452}, {0.5162384979888072, 4.4764981064457805}, {0.4410886576557874, 4.486400693338929}, {0.3657516600172219, 4.495040490316574}, {0.29024874485992197, 4.5024140029304585}, {0.214601216552231, 4.508518094320744}, {0.13883043801798783, 4.513349986399148}, {0.06295782468913977, 4.516907260930435}, {-0.01299516156151954, 4.519187860511872}, {-0.08900701850568338, 4.520190089450313}, {-0.16505620966112763, 4.519912614536548}, {-0.2411211704080741, 4.518354465716647}, {-0.3171803141182753, 4.515515036660001}, {-0.3932120382952197, 4.511394085223842}, {-0.4691947307238519, 4.505991733814004}, {-0.543725380315454, 4.499441598117181}, {-0.618167477497423, 4.491657571248978}, {-0.6925005762524499, 4.482640795670688}, {-0.7667042442343099, 4.472392752936791}, {-0.8407580683964112, 4.460915263553218}, {-0.9146416606192764, 4.448210486742132}, {-0.9883346633354846, 4.434280920113195}, {-1.061816755150595, 4.419129399241345}, {-1.1350676564583584, 4.4027590971511446}, {-1.208067135048737, 4.3851735237077545}, {-1.2807950117071678, 4.366376524914646}, {-1.3532311658035139, 4.346372282118162}, {-1.4253555408691536, 4.3251653111190915}, {-1.4971481501606618, 4.302760461191412}, {-1.6396585063493967, 4.254378182476532}, {-1.7103366782402436, 4.22841210947676}, {-1.7806039453530178, 4.201270866514578}, {-1.9198276470252003, 4.143489191104909}, {-1.988745284750138, 4.112862731358408}, {-2.0571744348558334, 4.081089043713178}, {-2.192490949191464, 4.014131468064832}, {-2.2593404679429305, 3.978964116279203}, {-2.3256258187166026, 3.942682604974458}, {-2.456429830004032, 3.8668136274677685}, {-2.7104602158645603, 3.7021247536555}, {-2.772284569099151, 3.6583148203613374}, {-2.8334004481562154, 3.6134717019586784}, {-2.953437860219645, 3.5207320564541447}, {-3.0123254004433635, 3.4728593223586928}, {-3.070436490553886, 3.4240009830682596}, {-3.18426350180855, 3.3233781747917073}, {-3.239947030058702, 3.271639736825101}, {-3.294789331625205, 3.2189677489678696}, {-3.401887840096603, 3.1108781395522467}, {-3.6053163053731305, 2.8841160080796153}, {-3.653844346722491, 2.825306900728107}, {-3.701412371074839, 2.7656808991926796}, {-3.793613601837104, 2.6440411842322433}, {-3.8382200698961464, 2.5820595311753487}, {-3.88181305199377, 2.519325096208341}, {-3.9659080029361005, 2.3916644393217856}, {-4.006385376927323, 2.3267720286137266}, {-4.045800082419705, 2.261194449448421}, {-4.121395379260816, 2.128053650636156}, {-4.259342940424828, 1.8543292755443217}, {-4.28892146443265, 1.7891593716847942}, {-4.317502048605792, 1.7235178805903053}, {-4.371639574218861, 1.590881433575552}, {-4.3971821924580246, 1.5239174216383864}, {-4.421698228213578, 1.4565437031984103}, {-4.467624534797403, 1.3206302567079384}, {-4.489022403281722, 1.2521223425696335}, {-4.509368887896799, 1.1832683422904244}, {-4.528658401194541, 1.1140844049530225}, {-4.546885603986752, 1.0445867696838493}, {-4.564045406793577, 0.974791761833291}, {-4.580132971233131, 0.904715789132612}, {-4.609073294895808, 0.763786968796366}, {-4.621917644520777, 0.6929673136327752}, {-4.633672938943899, 0.6219330707286783}, {-4.644335614033418, 0.5507010013236902}, {-4.653902363838124, 0.47928792554274985}, {-4.662370141555577, 0.407710718415969}, {-4.669736160438902, 0.33598630588262796}, {-4.675997894641939, 0.2641316607802733}, {-4.681153080002469, 0.19216379881984696}, {-4.6851997147633, 0.12009977454789068}, {-4.688136060231007, 0.04795667729663371}, {-4.689960641372115, -0.024248372876988313}, {-4.690672247346539, -0.09649822926233351}, {-4.690269931978125, -0.1687757225752527}, {-4.688753014162117, -0.2410636650464312}, {-4.686121078209441, -0.31334485451691446}, {-4.68237397412764, -0.3856020785398402}, {-4.677511817838396, -0.45781811848739684}, {-4.671534991331504, -0.5299757536620266}, {-4.664444142755254, -0.6020577654108904}, {-4.656240186443127, -0.6740469412426079}, {-4.646924302876778, -0.7459260789452894}, {-4.6364979385852685, -0.8176779907048707}, {-4.624962805980522, -0.8892855072227633}, {-4.612320883129007, -0.96073148183185}, {-4.598574413459661, -1.0319987946097517}, {-4.583725905408087, -1.103070356488531}, {-4.567778131997041, -1.1739291133597543}, {-4.550734130353315, -1.2445580501738513}, {-4.532597201161045, -1.3149401950329254}, {-4.513370908051561, -1.3850586232759494}, {-4.471665795237922, -1.5244368919042226}, {-4.449195411154694, -1.5936631557924958}, {-4.425652532733425, -1.6625585581722455}, {-4.3753690188446415, -1.7992903398057258}, {-4.348638890211456, -1.8670936825974418}, {-4.320857278745427, -1.934500098950256}, {-4.262163429864943, -2.068056970054246}, {-4.231263735891495, -2.1341750562380497}, {-4.199337643309965, -2.199831486703041}, {-4.132434101353276, -2.329695707281297}, {-3.986630253838224, -2.583189862557234}, {-3.944406114930033, -2.6503673371904743}, {-3.901049223584191, -2.7168543485847034}, {-3.8109826923225656, -2.8476807167381026}, {-3.7642966103957685, -2.9119824088297626}, {-3.7165248858574236, -2.975518319197581}, {-3.6177750933760175, -3.1002196403222526}, {-3.566823089230079, -3.16134899518129}, {-3.5148375631916298, -3.221640466948875}, {-3.4078213927552157, -3.3396400552153946}, {-3.1820130675051463, -3.5648308628002505}, {-3.123195251310718, -3.6187806840210346}, {-3.0634622855167297, -3.6717593898262244}, {-2.941315350778644, -3.7747416001957124}, {-2.8789342593507823, -3.824714845059357}, {-2.8157037644973553, -3.8736564639863666}, {-2.6867631036778765, -3.9683873380148906}, {-2.621087816648636, -4.01414856126734}, {-2.5546328746205047, -4.058822103162889}, {-2.419456361788916, -4.144853296804161}, {-2.1405555755982744, -4.303299160554821}, {-2.0691601467275, -4.340003928495709}, {-1.997135340308865, -4.375523308058114}, {-1.851276565415467, -4.4429630669889315}, {-1.7774825391952993, -4.474862850842123}, {-1.703139008979836, -4.505536061534644}, {-1.5528852127718307, -4.5631652557261475}, {-1.477016236325319, -4.590103337338927}, {-1.4006803236089096, -4.6157790475260265}, {-1.3238984951796111, -4.640184299539564}, {-1.2466919115153157, -4.663311356933126}, {-1.1690818671723977, -4.685152835980355}, {-1.0910897849016954, -4.705701707995962}, {-0.934045802969095, -4.742895304634343}, {-0.8550373362726267, -4.759527766601785}, {-0.7757336855451118, -4.774843100175233}, {-0.6961568249018166, -4.788836083228359}, {-0.616328820563167, -4.801501860515875}, {-0.5362718247246829, -4.812835945293328}, {-0.45600806939835103, -4.822834220834417}, {-0.3755598602274572, -4.831492941845278}, {-0.2949495702764523, -4.838808735775307}, {-0.2141996337974265, -4.844778604024089}, {-0.13333253997524164, -4.849399923044004}, {-0.052370826652914426, -4.852670445338113}, {0.028662925961151543, -4.854588300353031}, {0.10974610160201406, -4.855151995266391}, {0.19085605427151067, -4.854360415668664}, {0.2719701145744315, -4.852212826139036}, {0.3530655960654597, -4.848708870715134}, {0.4287447198287153, -4.844212982884456}, {0.5043693854037076, -4.838535048095181}, {0.5799211172100753, -4.831675673784656}, {0.6553814454167433, -4.823635756364974}, {0.730731910464733, -4.81441648119047}, {0.8059540675904253, -4.804019322454371}, {0.8810294913480108, -4.79244604301459}, {0.9559397801300886, -4.779698694148681}, {1.0306665606853678, -4.765779615237932}, {1.1051914926322075, -4.750691433380666}, {1.179496272966975, -4.7344370629348}, {1.2535626405661904, -4.717019704989693}, {1.3273723806812, -4.698442846767393}, {1.4009073294243395, -4.678710260953414}, {1.5470804783981518, -4.635794420101717}, {1.6196826453928823, -4.612620130744632}, {1.6919379634389549, -4.588308043327321}, {1.8353367595618844, -4.536291504311676}, {1.9064447893165724, -4.50859826649345}, {1.9771350822524403, -4.479789655789357}, {2.1171925278355688, -4.4188517913785255}, {2.186524957417497, -4.386735961387807}, {2.2553702126684008, -4.353531603009369}, {2.391530911018202, -4.283887133986954}, {2.657261551859039, -4.1318789722242295}, {2.722224352868359, -4.091276793251483}, {2.786566552201135, -4.049652640258446}, {2.913324946142152, -3.963376625465391}, {2.9757094078781012, -3.918744501395313}, {3.0374098101415488, -3.873129874256317}, {3.1586966706528576, -3.7789953136923446}, {3.218252652327364, -3.7304970894159224}, {3.277063629043099, -3.6810597753644667}, {3.3923914841441483, -3.5794139153513607}, {3.6134672848800156, -3.36537246990717}, {3.6666585284245765, -3.3096950533680896}, {3.7189913828667374, -3.253176545281776}, {3.8210289088569347, -3.13766943427592}, {3.8707075870153904, -3.0787079444438885}, {3.919475895908208, -3.0189595828704996}, {4.014231753393092, -2.897158703309668}, {4.0601950225198165, -2.835134905338418}, {4.105199369662279, -2.7723816688793126}, {4.192285211425355, -2.6447464054572705}, {4.354455902562926, -2.3813874856378}, {4.391700825093325, -2.3152837276419893}, {4.427941605591713, -2.2485974131824467}, {4.497374525235717, -2.1135383114212085}, {4.53054913941908, -2.045196475130244}, {4.56268456604902, -1.9763339772938064}, {4.623805417401047, -1.8371103739471146}, {4.652775226418519, -1.7667812736638413}, {4.680674620437174, -1.6959955149204178}, {4.7332336314923005, -1.553119346260472}, {4.757879605483339, -1.481061877922659}, {4.781427881893325, -1.408613626675157}, {4.82520683408987, -1.2626118058365081}, {4.845425895610174, -1.1890919924999095}, {4.864524034055985, -1.1152489009456494}, {4.882496088701442, -1.0410996327324564}, {4.899337159909062, -0.966661371673915}, {4.915042610438711, -0.8919513798497376}, {4.929608066695599, -0.8169869935962539}, {4.943029419916957, -0.741785619477035}, {4.9553028272971025, -0.666364730234574}, {4.9664247130505546, -0.5907418607240524}, {4.976391769412975, -0.5149346038299834}, {4.985200957579605, -0.43896060636677164}, {4.992849508580952, -0.3628375649642332}, {4.999334924095514, -0.28658322193888025}, {5.004654977199271, -0.2102153611520223}, {5.008807713051762, -0.1337518038556336}, {5.0117914495185305, -0.05721040452694329}, {5.013604777729771, 0.01939095330718974}, {5.014246562574977, 0.09603436125529848}, {5.0137159431334855, 0.1727018902546562}, {5.012012333040726, 0.2493755947718793}, {5.009135420790088, 0.3260375170099592}, {5.005085169970283, 0.40266969112064627}, {4.9998618194381095, 0.47925414742121464}, {4.993465883426541, 0.5557729166146339}, {4.985898151588085, 0.6322080340120663}, {4.977159688973348, 0.7085415437568569}, {4.967251835944781, 0.7847555030489314}, {4.9561762080255924, 0.8608319863685185}, {4.943934695683836, 0.9367530896983626}, {4.930529464051651, 1.0125009347433376}, {4.900237874626999, 1.163405490700536}, {4.883357216985681, 1.2385266115536988}, {4.86532423934165, 1.3134033024091127}, {4.825815723568018, 1.4623526988561062}, {4.804348063519292, 1.5363901883109712}, {4.7817438380997155, 1.6101128238315923}, {4.7331444146757375, 1.756543769303485}, {4.707159248209552, 1.8292173703915156}, {4.680057577408362, 1.901506708055769}, {4.622527710136923, 2.0448640254643546}, {4.494337495391084, 2.326218184330689}, {4.456596930166456, 2.4011837525994166}, {4.417605211963986, 2.47553688138001}, {4.335908432100348, 2.622322410486592}, {4.293224298221988, 2.694713515270773}, {4.2493308615874845, 2.766409601110137}, {4.15796170501344, 2.9076360515461683}, {4.1105096395264304, 2.977126550903924}, {4.061895574075261, 3.045842312098928}, {3.961232389782267, 3.18087206521408}, {3.746540463028733, 3.440793497511585}, {3.6901647255924677, 3.503553721544764}, {3.6327371455725603, 3.5653897768774128}, {3.514787388150623, 3.6862190849482945}, {3.454296411408611, 3.7451778163922795}, {3.3928159851472772, 3.8031433464075337}, {3.266952331393252, 3.91602862388084}, {3.202602570612353, 3.9709159574945123}, {3.1373302848315117, 4.024745271291585}, {3.0040880240688748, 4.129168082653694}, {2.727316288416745, 4.324654433201209}, {2.656087976123986, 4.370659654432873}, {2.5840834234990244, 4.415491032753073}, {2.437823246133564, 4.501580179980261}, {2.3636069933296615, 4.542812710798341}, {2.288693233455744, 4.582820929904339}, {2.1368542300601456, 4.6591175653941725}, {2.0599699989014812, 4.695383385884887}, {1.9824702743070626, 4.730379709540706}, {1.8257084208781336, 4.796522432256028}, {1.7464887674246856, 4.8276489841938135}, {1.6667385597732005, 4.857466350971548}, {1.5057332319448185, 4.913137725316457}, {1.4245218658391052, 4.938974729190724}, {1.3428674417707642, 4.963468545664199}, {1.2607921889450724, 4.986611582466441}, {1.1783184665686612, 5.008396615441333}, {1.0954687577497602, 5.02881679077123}, {1.0122656633603493, 5.047865627099995}, {0.9287318958619678, 5.065537017554226}, {0.844890273096945, 5.0818252316620685}, {0.7607637120466771, 5.096724917169022}, {0.6763752225585907, 5.110231101750189}, {0.591747901043582, 5.12233919461843}, {0.5069049241457352, 5.133044988027869}, {0.4218695423860019, 5.142344658672324}, {0.3366650737814062, 5.150234768978211}, {0.25131489744185553, 5.156712268291456}, {0.16584244714615629, 5.161774493958055}, {0.085990301522915, 5.165219919370971}, {0.006071238339583996, 5.167429321351089}, {-0.07389559901248058, 5.168401454532324}, {-0.15389104464192152, 5.168135370209992}, {-0.23389591475727498, 5.166630416738525}, {-0.3138910122737958, 5.163886239857815}, {-0.3938571314260865, 5.159902782948073}, {-0.4737750623854037, 5.154680287213114}, {-0.5536255958806192, 5.148219291791957}, {-0.6333895278215618, 5.140520633798714}, {-0.7130476639237125, 5.131585448290673}, {-0.7925808243332344, 5.121415168164576}, {-0.8719698482510528, 5.110011523981065}, {-0.9511955985549632, 5.097376543717289}, {-1.0302389664186316, 5.0835125524477185}, {-1.109080875926352, 5.068422171953179}, {-1.1877022886825397, 5.052108320258157}, {-1.2660842084146748, 5.034574211096478}, {-1.3442076855686826, 5.0158233533054295}, {-1.422053821895718, 4.995859550148415}, {-1.4996037750290878, 4.974686898566296}, {-1.5768387630502856, 4.952309788357549}, {-1.7302890456341034, 4.903961210126071}, {-1.8064671195202031, 4.877999977616462}, {-1.8822557959791737, 4.850854755371258}, {-2.032591397585206, 4.793035985365679}, {-2.10710176655988, 4.762374974272738}, {-2.181149634660646, 4.7305550440841575}, {-2.327785834473639, 4.6634666150897655}, {-2.4003384168425383, 4.628212910998594}, {-2.4723570083741855, 4.591829873995613}, {-2.614721991349811, 4.515708436543463}, {-2.6850335798096743, 4.475987036943234}, {-2.754741579978543, 4.435170302344438}, {-2.8922786686443445, 4.350287794331338}, {-2.9600740367067684, 4.306241167362682}, {-3.0271983838815655, 4.261137493976316}, {-3.1593682167166426, 4.1678001723318285}, {-3.4149408007406254, 3.9689080547942512}, {-3.476915329760613, 3.9167056454781823}, {-3.538091516896099, 3.8635350710695215}, {-3.65798852919624, 3.7543385242846226}, {-3.716679657406074, 3.698337689091422}, {-3.7745130571769723, 3.6414189573848903}, {-3.887549429726553, 3.5248806051803885}, {-3.94272428901838, 3.4652879438511195}, {-3.996985199887205, 3.404831298107051}, {-4.102711256675368, 3.2813823624165663}, {-4.302615115237312, 3.024747766152728}, {-4.354046650089058, 2.9530732952078678}, {-4.404283567845925, 2.8805240167946966}, {-4.501114349395079, 2.732880979774382}, {-4.547679447240202, 2.6578278199309726}, {-4.592992404625242, 2.581981038972188}, {-4.6798081329149195, 2.427990451438109}, {-4.721284897989684, 2.3498891345110064}, {-4.761457518712218, 2.2710791640284853}, {-4.8378422557978285, 2.1114206256491395}, {-4.8740312506835375, 2.0306162505670518}, {-4.908869865231372, 1.9491915953125687}, {-4.97445380684009, 1.7845719462218754}, {-5.005179008239704, 1.7014226524916451}, {-5.034513584116023, 1.617744466098844}, {-5.062448432836021, 1.5335606575631129}, {-5.088974842395086, 1.4488946539780938}, {-5.114084493109422, 1.3637700324839315}, {-5.137769460199043, 1.2782105136935014}, {-5.180835633628886, 1.1058823442870993}, {-5.200202986627504, 1.0191617924854888}, {-5.218117953705948, 0.9321025275744022}, {-5.2345746194644835, 0.8447288874332974}, {-5.2495674765649465, 0.7570653131036155}, {-5.263091427527278, 0.6691363419431887}, {-5.275141786411148, 0.5809666007494064}, {-5.28571428038214, 0.49258079885306916}, {-5.294805051161967, 0.4040037211848712}, {-5.302410656362265, 0.31526022131645826}, {-5.308528070701494, 0.2263752144780177}, {-5.313154687104565, 0.13737367055436453}, {-5.316288317684779, 0.04828060706149475}, {-5.31792719460776, -0.0408789178944153}, {-5.318069970837059, -0.1300798126735806}, {-5.31671572076115, -0.21929695917275974}, {-5.313863940701568, -0.3085052199069938}, {-5.309514549301996, -0.39767944510093234}, {-5.303667887798087, -0.48679447978774354}, {-5.2963247201679255, -0.5758251709136039}, {-5.287486233162963, -0.6647463744457559}, {-5.277154036219395, -0.7535329624821255}, {-5.265330161249913, -0.842159830360485}, {-5.252017062315833, -0.9306019037651302}, {-5.237217615179605, -1.018834145829168}, {-5.220935116737806, -1.1068315642301247}, {-5.203173284334644, -1.19456921827721}, {-5.183936254956159, -1.2820222259880645}, {-5.163228584305263, -1.369165771152806}, {-5.141055245757778, -1.4559751103837246}, {-5.117421629199768, -1.5424255801483455}, {-5.065797196342353, -1.7141516984916234}, {-5.037819230245087, -1.7993784823076153}, {-5.008406683389644, -1.8841486810503811}, {-4.945308057905956, -2.0522228069950277}, {-4.911638100187609, -2.1354787868842617}, {-4.8765657994458005, -2.218182300765621}, {-4.802250834200774, -2.381837551090854}, {-4.763755801929836, -2.461271211284231}, {-4.723946225477011, -2.540082576662895}, {-4.640424051701775, -2.695751528043524}, {-4.596732658843736, -2.772566079506295}, {-4.551769125364662, -2.848672278980987}, {-4.45807194035104, -2.998675497850478}, {-4.409362311131973, -3.07253092701623}, {-4.359428579774135, -3.1455948352734464}, {-4.255940617608434, -3.289267108694224}, {-4.034864005935756, -3.5662346519464716}, {-3.976740462929077, -3.633201694262313}, {-3.91750498866151, -3.6992210052993983}, {-3.795760395842873, -3.8283428063175413}, {-3.7332831151842116, -3.891409118656272}, {-3.669757570722538, -3.953455354566879}, {-3.5396286331310636, -4.0744181487721685}, {-3.4730594317100745, -4.13330066857992}, {-3.405510340916134, -4.191095044559953}, {-3.26754394616045, -4.303354407625324}, {-2.980653430576732, -4.514117434305632}, {-2.906758270563842, -4.563854682389417}, {-2.8320329211568205, -4.612381690676876}, {-2.6801712104678224, -4.705749871453343}, {-2.603075200626838, -4.750564302431182}, {-2.5252296935422596, -4.79411501823356}, {-2.36737329637691, -4.877375496299012}, {-2.2874044794127424, -4.917061208222573}, {-2.2067703002135124, -4.9554351112176285}, {-2.0435921619678905, -5.028203219643444}, {-1.9610918181788057, -5.06257617668215}, {-1.8780133309993112, -5.095594834185168}, {-1.710211062416108, -5.157530722222765}, {-1.625532251753538, -5.18642960510787}, {-1.5403652271175932, -5.2139374985576366}, {-1.4547328494203822, -5.240046157253193}, {-1.3686581187760587, -5.264747711788599}, {-1.2821641683163232, -5.2880346710199255}, {-1.1952742579662252, -5.309899924312567}, {-1.020400193652843, -5.349338785855776}, {-0.9324631369678208, -5.366900094170939}, {-0.8442243022502598, -5.383015100448276}, {-0.7557074887606996, -5.397678626700862}, {-0.6669365844706387, -5.410885886761533}, {-0.5779355596085959, -5.422632487800423}, {-0.48872846018015276, -5.432914431736141}, {-0.39933940146390673, -5.441728116540145}, {-0.30979256148501766, -5.449070337433887}, {-0.22614942292607743, -5.454589809422471}, {-0.1424098057282702, -5.458824743007386}, {-0.058593446511370724, -5.461773476880482}, {0.025279889843409724, -5.463434653127157}, {0.1091904148594148, -5.463807217702318}, {0.1931183211385015, -5.462890420834534}, {0.27704378704184085, -5.460683817358277}, {0.3609469813764537, -5.457187266974097}, {0.44480806808637224, -5.452400934436702}, {0.5286072109473166, -5.446325289670772}, {0.6123245782637733, -5.438961107814522}, {0.695940347567345, -5.43030946919093}, {0.7794347103153512, -5.420371759206597}, {0.8627878765883927, -5.409149668178248}, {0.945980079785862, -5.396645191086871}, {1.0289915813183788, -5.382860627259505}, {1.1118026752958634, -5.367798579978725}, {1.194393693210231, -5.35146195601989}, {1.2767450086115697, -5.333853965116219}, {1.3588370417766946, -5.314978119351771}, {1.4406502643689623, -5.2948382324824745}, {1.5221652040882387, -5.273438419185291}, {1.684222653711813, -5.226876971613577}, {1.7647265408880735, -5.201725063537822}, {1.8448549089485848, -5.175332679429663}, {2.0039086802290735, -5.118849200096717}, {2.082796093421586, -5.088770199404564}, {2.1612320165250316, -5.057474909177434}, {2.316674450638207, -4.99126285925278}, {2.3936437495836826, -4.95636052134699}, {2.470087143226794, -4.920270734364695}, {2.6213230254317117, -4.844560800021552}, {2.696079222493109, -4.804957351725074}, {2.7702369399659283, -4.764199848809457}, {2.9166857805313744, -4.679259141656707}, {2.9889416720792448, -4.635094855030678}, {3.060528629429549, -4.589814344802824}, {3.2016268932332874, -4.495945462527874}, {3.475047712621187, -4.295276716216746}, {3.541478007250548, -4.242481205040289}, {3.6071058940105596, -4.188657887372239}, {3.735890997623786, -4.077976880281306}, {3.799016959107636, -4.021144327643543}, {3.861278009569549, -3.9633342363764688}, {3.983145006362262, -3.8448343460927585}, {4.042721272254472, -3.784171584713924}, {4.101373273475166, -3.7225853538113363}, {4.215847420967504, -3.5966990591502994}, {4.433089092641593, -3.334479420245533}, {4.489209571459412, -3.261085807454903}, {4.544114411818369, -3.186747361354356}, {4.650213865074106, -3.035316814983638}, {4.701377661391803, -2.9582658021117325}, {4.751264195031821, -2.8803521195704525}, {4.847147630265991, -2.7220217228317045}, {4.8931164927066355, -2.641648105007736}, {4.937752022346236, -2.5604979985213943}, {5.0229709767565565, -2.395957058616287}, {5.063529270689981, -2.3126111427703235}, {5.102703977537577, -2.228578561134791}, {5.176856505506133, -2.0585455137989483}, {5.2118122222995735, -1.9725915905730436}, {5.2453401499134795, -1.8860440737544009}, {5.308072713410767, -1.7112633435675593}, {5.337258376418601, -1.6230780936144815}, {5.364978310197565, -1.5343951637238573}, {5.415987457236779, -1.3556339018551011}, {5.439260920267926, -1.265604743503622}, {5.461037158427013, -1.1751762389275608}, {5.48130932342282, -1.0843732991670065}, {5.500070981631059, -0.9932209526263421}, {5.517316116104353, -0.9017443381685032}, {5.533039128467148, -0.809968698174835}, {5.547234840694982, -0.7179193715724732}, {5.559898496777547, -0.6256217868311706}, {5.571025764265025, -0.5331014549315118}, {5.580612735697203, -0.44038396230645777}, {5.5886559299149035, -0.3474949637581798}, {5.595152293253315, -0.2544601753521422}, {5.600099200616819, -0.1613053672904064}, {5.603494456434958, -0.06805635676613135}, {5.60533629549921, 0.025260999198744385}, {5.605623383680288, 0.11862081093065169}, {5.604354818525679, 0.21199716329437576}, {5.601530129737219, 0.30536412289387854}, {5.5971492795285025, 0.39869574528195606}, {5.591212662861948, 0.491966082176724}, {5.583721107565419, 0.5851491886829223}, {5.574675874328275, 0.6782191305160281}, {5.5640786565768225, 0.771149991227162}, {5.55193158022912, 0.8639158794267752}, {5.538237203329142, 0.9564909360050998}, {5.522998515560356, 1.0488493413473485}, {5.5062189376387725, 1.1409653225416492}, {5.48790232058558, 1.2328131605776975}, {5.468052944879516, 1.3243671975341191}, {5.4466755194891245, 1.4156018437525277}, {5.399357491333202, 1.597010989591878}, {5.373428438567233, 1.68713471555115}, {5.345994433343709, 1.7768375176719964}, {5.286639316557791, 1.9548798959602596}, {5.2547331291499715, 2.043169529223095}, {5.221351833874828, 2.1309383668590005}, {5.1501983405610465, 2.304815171504538}, {5.115000317895831, 2.3851690586019907}, {5.0785480954077515, 2.4649853348627815}, {5.00191420957816, 2.6229268451524548}, {4.961749888395975, 2.701013292872765}, {4.920366047797705, 2.7784845666701043}, {4.83397780613736, 2.931505524105119}, {4.788993146220913, 3.007017539324975}, {4.742828444677357, 3.081839053063898}, {4.647001615479359, 3.229336948493557}, {4.441673157544915, 3.515246189928171}, {4.387561013057246, 3.5847270941078793}, {4.332361580250753, 3.653374726026577}, {4.218752462319412, 3.788102288616227}, {4.160369246784096, 3.8541487758038415}, {4.100951675942978, 3.9192951120665964}, {3.9790692585936642, 4.046822728480523}, {3.916632938747455, 4.109172246550594}, {3.8532193102421735, 4.170558097443498}, {3.7235198865329933, 4.290377732567503}, {3.453063945541407, 4.517794361699726}, {3.3832389959159714, 4.572018610131506}, {3.31256230191798, 4.625163244584856}, {3.168720689958587, 4.728160400062404}, {3.0955897983487777, 4.777986934716376}, {3.021675207015707, 4.826681889200853}, {2.8715651861332874, 4.920628012211782}, {2.795405367853308, 4.965855339256712}, {2.718533063934411, 5.00990340921004}, {2.5627242453701378, 5.0944171568545595}, {2.2433721481460647, 5.248818743667057}, {2.1620335265964323, 5.28431373000247}, {2.0801332624228523, 5.318547729037806}, {1.9147261758341563, 5.383197520397119}, {1.831258885068972, 5.413596451173807}, {1.7473090055232987, 5.442700675932344}, {1.5780419669508223, 5.4969946785186385}, {1.4927653504333007, 5.522170073615762}, {1.4070872204345375, 5.546022001015962}, {1.3210281253673641, 5.568544059439881}, {1.2346087156573906, 5.589730166781679}, {1.1478497387834832, 5.609574561733941}, {1.0607720342920253, 5.628071805335582}, {0.9733965287864967, 5.645216782442219}, {0.8857442308939483, 5.661004703118562}, {0.7995698132338405, 5.675160161226309}, {0.7131695924191834, 5.688003298589611}, {0.6265635615275323, 5.699530519805663}, {0.539771770783912, 5.709738534189433}, {0.4528143229146584, 5.718624356680447}, {0.36571136848689484, 5.726185308678683}, {0.2784831012347252, 5.732419018809376}, {0.19114975337323192, 5.737323423616516}, {0.10373159090136283, 5.740896768184836}, {0.016248908894801947, 5.743137606690125}, {-0.07127797321008325, 5.7440448028777}, {-0.15882871634012102, 5.743617530468898}, {-0.24638296651594135, 5.741855273495434}, {-0.3339203595882078, 5.7387578265615655}, {-0.4214205259783723, 5.734325295033922}, {-0.5088630954228477, 5.728558095158943}, {-0.5962277017194927, 5.721456954107869}, {-0.6834939874753025, 5.713022909949228}, {-0.7706416088541967, 5.7032573115488026}, {-0.8576502403237958, 5.692161818397069}, {-0.9444995794000796, 5.679738400364103}, {-1.0311693513888183, 5.6659893373819985}, {-1.1176393141226666, 5.650917219054812}, {-1.2038892626928146, 5.634524944196123}, {-1.289899034174089, 5.616815720294262}, {-1.375648512342398, 5.597793062905306}, {-1.4611176323834134, 5.577460794973958}, {-1.5462863855913913, 5.555823046082425}, {-1.6311348240570258, 5.532884251627442}, {-1.715643065343241, 5.508649151925613}, {-1.883559781951769, 5.45631051677873}, {-1.9669288616523675, 5.4282179775141595}, {-2.049878962179737, 5.398851123075589}, {-2.214444377175376, 5.336319762733791}, {-2.296021004965722, 5.303168647612075}, {-2.3771012893278267, 5.268769996028305}, {-2.5376965978633463, 5.196260105979122}, {-2.6171737898731373, 5.158164607291151}, {-2.696078983045218, 5.118853048291626}, {-2.8520990488414513, 5.036616396892417}, {-3.1564651008783002, 4.857947342534262}, {-3.230852661690167, 4.810379719404752}, {-3.3045231734556815, 4.761672627661208}, {-3.4496433717985493, 4.6608835562965085}, {-3.521058631219318, 4.608824002278393}, {-3.5916879954504726, 4.555669825866561}, {-3.7305220840609112, 4.44612533565274}, {-3.7986937834805374, 4.389759527566207}, {-3.8660135456485265, 4.332348103036817}, {-3.998033282302877, 4.214440175829802}, {-4.251156894848586, 3.9666600282204243}, {-4.317214461924123, 3.896783340141194}, {-4.38212302773316, 3.8258062243412096}, {-4.5084201182757795, 3.680626618637135}, {-4.5697729070813535, 3.6064628422524563}, {-4.629905232871528, 3.5312760548588797}, {-4.746440575674813, 3.3779141011688614}, {-4.802810464864773, 3.2997799681223507}, {-4.857893644803042, 3.2207048795825335}, {-4.964137382875443, 3.059816890243583}, {-5.160516721055714, 2.72769941052518}, {-5.206172898550307, 2.6426454622260165}, {-5.250426701076577, 2.5568271610325963}, {-5.3346763686447645, 2.382990241039985}, {-5.374647785250142, 2.295018522304968}, {-5.413167937089257, 2.2063762381412606}, {-5.48580983938742, 2.0271759661515727}, {-5.519910278061766, 1.936666440035267}, {-5.552516834627337, 1.8455832587936098}, {-5.583619858768398, 1.7539510890771197}, {-5.613210107623152, 1.6617947598705909}, {-5.641278748514183, 1.569139255759906}, {-5.667817361567605, 1.4760097101529317}, {-5.716272903613458, 1.288429731209394}, {-5.738175078874612, 1.1940302471799056}, {-5.758517723282801, 1.099258606418181}, {-5.777294516320881, 1.0041405832769006}, {-5.794499563611499, 0.9087020593954617}, {-5.810127398737155, 0.8129690166517127}, {-5.824172984943654, 0.7169675300826249}, {-5.836631716726397, 0.6207237607762012}, {-5.847499421299023, 0.5242639487363967}, {-5.856772359943983, 0.42761440572283577}, {-5.8644472292445515, 0.33080150806764974}, {-5.870521162197937, 0.23385168947119525}, {-5.874991729209125, 0.13679143377841094}, {-5.877856938965093, 0.0396472677381649}, {-5.879115239189134, -0.057554246252593554}, {-5.878765517275001, -0.15478651741805816}, {-5.876807100800659, -0.25202293388379915}, {-5.8732397579214375, -0.3492368699662198}, {-5.8680636976424285, -0.44640169346940467}, {-5.86127956996998, -0.5434907729871924}, {-5.852888465942211, -0.6404774852082995}, {-5.842891917538472, -0.7373352222226601}, {-5.831291897467715, -0.8340373988271408}, {-5.818090818835762, -0.9305574598284527}, {-5.803291534691554, -1.0268688873410938}, {-5.7868973374524115, -1.1229452080784728}, {-5.768911958208399, -1.2187600006353825}, {-5.74933956590594, -1.3142869027596433}, {-5.728184766410874, -1.409499618610756}, {-5.705452601451153, -1.504371926003725}, {-5.68114854743935, -1.5988776836362646}, {-5.6278488434293275, -1.786685432052974}, {-5.600856064937335, -1.873703717012375}, {-5.572517321885208, -1.9603142141689465}, {-5.511826907466153, -2.1322286337304366}, {-5.479488523254237, -2.217491188164037}, {-5.445830745901581, -2.3022632288131715}, {-5.374587093680076, -2.4702541860270455}, {-5.33701704870648, -2.5534325984143673}, {-5.298159266879104, -2.636039498433972}, {-5.216615585113586, -2.7994591218761085}, {-5.03847858275873, -3.118571864773298}, {-4.99086728362235, -3.1966275203148515}, {-4.942046347307255, -3.2739560915084684}, {-4.840820281751152, -3.426357142976992}, {-4.788438236176637, -3.501392633147062}, {-4.7348927154970095, -3.5756270677407733}, {-4.624360545765544, -3.721620773396785}, {-4.5673992422800245, -3.793344519223573}, {-4.509325149012885, -3.864196167649329}, {-4.389892287993742, -4.0032142966363}, {-4.138278568224757, -4.270039445194925}, {-4.07280372428162, -4.334314598286431}, {-4.006330270762865, -4.397585453030154}, {-3.8704494166496124, -4.52105242744146}, {-3.801073556348305, -4.581218222202418}, {-3.730762159734638, -4.640319076248329}, {-3.5873983960785427, -4.7552680106487655}, {-3.5143794103397803, -4.811087746773942}, {-3.4404916431203265, -4.86578586071047}, {-3.290178915279767, -4.971763387119659}, {-2.9799003071237538, -5.169681808238063}, {-2.9004251580424065, -5.216164930589395}, {-2.8202247734031984, -5.26142582858402}, {-2.6577236861407654, -5.3482359721424855}, {-2.575461114976391, -5.389763454227772}, {-2.4925495622182794, -5.430025190444549}, {-2.3248576009688957, -5.506711151638371}, {-2.2401166287626926, -5.543115992255106}, {-2.1548055382523206, -5.578216323168432}, {-1.982553497981389, -5.644468049938075}, {-1.8956531408849993, -5.675602518700107}, {-1.8082638412397165, -5.705398627876875}, {-1.6321010131325318, -5.760945372838599}, {-1.5358917518803539, -5.788782722622483}, {-1.439207727713185, -5.8150127753123035}, {-1.3420758023427148, -5.839627434918154}, {-1.2445229755142149, -5.862619054537642}, {-1.146576377495814, -5.883980438708744}, {-1.0482632615276835, -5.903704845637008}, {-0.9496109962328516, -5.921785989296487}, {-0.8506470579925154, -5.938218041403597}, {-0.7513990232876042, -5.952995633263383}, {-0.6518945610083454, -5.9661138574876516}, {-0.5521614247347424, -5.9775682695843235}, {-0.4522274449897382, -5.987354889417568}, {-0.35212052146684764, -5.99547020253827}, {-0.25186861523519444, -6.001911161384335}, {-0.15149974092374996, -6.006675186350463}, {-0.051041958886577275, -6.009760166727065}, {0.049476632647960185, -6.011164461507945}, {0.15002790544221062, -6.010886900066485}, {0.25058370912664746, -6.008926782700099}, {0.3511158790867084, -6.0052838810427085}, {0.45159624435723356, -5.999958438345104}, {0.5519966355226711, -5.9929511696230096}, {0.652288892620066, -5.984263261672787}, {0.7524448730429976, -5.973896372954708}, {0.8524364594446279, -5.961852633343731}, {0.9522355676368744, -5.948134643747848}, {1.051814154483869, -5.932745475594029}, {1.1511442257878661, -5.915688670181795}, {1.250197844164615, -5.8969682379046375}, {1.3489471369063648, -5.876588657339399}, {1.5454216251119783, -5.830872300182037}, {1.6430914690943657, -5.805546811620005}, {1.7403463000830908, -5.778584748088147}, {1.933501306690191, -5.719778560987578}, {2.0293469605082133, -5.68794941792821}, {2.1246685731315265, -5.654513657133477}, {2.3136320572910503, -5.582857252562785}, {2.4072204830136466, -5.544655221247326}, {2.500177991039601, -5.504883792306142}, {2.6840951222036433, -5.4206748734167975}, {3.04321198115527, -5.233912289616109}, {3.1310260983160236, -5.183471439743198}, {3.21800392397105, -5.131556773300319}, {3.3893519575216486, -5.023361885729275}, {3.4736733760587524, -4.967110647120918}, {3.5570609382704017, -4.909443549226353}, {3.7209396263189927, -4.789924209583528}, {3.801383969285916, -4.7281041794412}, {3.8808009035456337, -4.664932704547907}, {4.036461987310151, -4.534604125012171}, {4.3344753705551655, -4.258506489009942}, {4.401375616417713, -4.191188730057911}, {4.467230034372886, -4.122819683254341}, {4.595735540905551, -3.982992998290811}, {4.658354342432272, -3.911568641809855}, {4.719862750060172, -3.8391595529118048}, {4.839486693833365, -3.6914564954869533}, {4.897572060541062, -3.616197790849157}, {4.954486702079775, -3.5400248731500685}, {5.064746524177171, -3.385009497059826}, {5.270603405342223, -3.06482471349209}, {5.318936114452958, -2.982774786013304}, {5.365991095896864, -2.8999624834359423}, {5.456220075979131, -2.7321305596193883}, {5.499370953139268, -2.647151311308762}, {5.541197865590396, -2.5614904240390506}, {5.620837041500009, -2.3882064392996334}, {5.658628734453875, -2.300625115549138}, {5.695055326988879, -2.212445690201201}, {5.763775676628581, -2.0343778225301454}, {5.796051498965021, -1.9445323935370267}, {5.8269263560087845, -1.8541748792094586}, {5.884441016330789, -1.672011133966689}, {5.911065595156164, -1.580248989690405}, {5.936258763789846, -1.488062922680495}, {5.960013773365475, -1.3954752691086079}, {5.9823242236740475, -1.3025084725691247}, {6.003184064873389, -1.209185078635027}, {6.022587599112621, -1.1155277293864034}, {6.040529482071152, -1.0215591579130794}, {6.0570047244117395, -0.9273021827928912}, {6.072008693147306, -0.8327797025467681}, {6.085537112921118, -0.7380146900718002}, {6.09758606719998, -0.6430301870538194}, {6.10815199938008, -0.5478492983610238}, {6.117231713805204, -0.45249518641984027}, {6.124822376697074, -0.35699106557422194}, {6.130921516997495, -0.2613601964299266}, {6.135527027122065, -0.16562588018532923}, {6.1386371636252655, -0.06981145294997916}, {6.140250547776711, 0.026059719947887243}, {6.140366166048386, 0.12196424966331514}, {6.138983370512703, 0.2178787295445133}, {6.136101879151261, 0.31377974084942944}, {6.131721776074199, 0.40964385846779633}, {6.125843511650017, 0.5054476566470746}, {6.118467902545854, 0.6011677147207185}, {6.109596131678121, 0.6967806228375355}, {6.099229748073515, 0.7922629876909142}, {6.087370666640351, 0.8875914382463397}, {6.074021167850287, 0.9827426314656212}, {6.059183897330487, 1.0776932580266054}, {6.042861865366226, 1.1724200480371407}, {6.025058446314053, 1.266899776741724}, {6.005777377925651, 1.361109270219249}, {5.985443740907843, 1.4531873582646737}, {5.9636979604242155, 1.5449617554289183}, {5.915988441107874, 1.7275128896218614}, {5.890034768403602, 1.8182465072814586}, {5.862689082390215, 1.9085902055129396}, {5.803845431594121, 2.0880224820980913}, {5.772360165900619, 2.1770686047572414}, {5.7395082823532695, 2.265639906650422}, {5.669733631046521, 2.441274235520139}, {5.514121525614441, 2.7859294322077206}, {5.471921770921074, 2.8705959396122966}, {5.42842161711712, 2.954623085526647}, {5.337559176567649, 3.120679524971383}, {5.29021718736113, 3.2026693064690392}, {5.241615389594897, 3.2839407118163306}, {5.140676281272207, 3.444251107650815}, {5.088361668860612, 3.523251873393584}, {5.034832639079669, 3.60145782261465}, {4.924179931324511, 3.755410761259122}, {4.688852075810826, 4.052969757283634}, {4.627176225771883, 4.125099724013417}, {4.564390472495542, 4.196290872557438}, {4.435546698783507, 4.335788600203353}, {4.369518053809589, 4.4040616702411794}, {4.302438249824552, 4.471328912014295}, {4.165186721136385, 4.602781391861134}, {4.095046394428734, 4.666934958383614}, {4.023917697516992, 4.730019361168539}, {3.8787606381162774, 4.852920000973144}, {3.577318904721658, 5.0852380217663145}, {3.4997409661227215, 5.140425991029535}, {3.421311434793556, 5.19443011123196}, {3.261970082207949, 5.2988345065050275}, {3.181095008102998, 5.349209325599732}, {3.0994418270921966, 5.398349389882539}, {2.933876759840848, 5.4928774574957755}, {2.850003141310022, 5.5382422856566444}, {2.7654279424432375, 5.5823260143402}, {2.5942512669070776, 5.666607065043586}, {2.507689436953771, 5.706783583451731}, {2.420505310908905, 5.745637400283656}, {2.24435119080548, 5.8193386787703165}, {2.1478647404486524, 5.857055457598836}, {2.050744364059712, 5.893171962591694}, {1.8547082111939752, 5.9605619684914295}, {1.755846064223801, 5.991815482672609}, {1.6564572355541043, 6.021428755467801}, {1.4562085712386768, 6.075699525893487}, {1.3554036210775726, 6.100340630375468}, {1.2541817451330604, 6.123308712090514}, {1.152570718629298, 6.144596717661427}, {1.0505984360941574, 6.16419805497539}, {0.9482929036967007, 6.182106595201429}, {0.8456822315505484, 6.198316674680541}, {0.7427946259848657, 6.212823096687994}, {0.6396583817858735, 6.225621133067086}, {0.5363018744105935, 6.236706525733976}, {0.43275355217482653, 6.2460754880530995}, {0.32904192841770175, 6.2537247060827275}, {0.22519557364512696, 6.259651339690224}, {0.12124310765417784, 6.26385302353671}, {0.01721319164015265, 6.266327867930772}, {-0.08686547971072998, 6.267074459550925}, {-0.19096418614120644, 6.266091862036571}, {-0.2950541897536305, 6.263379616447258}, {-0.3991067429484299, 6.258937741590014}, {-0.5030930963685635, 6.252766734214681}, {-0.6069845068481836, 6.244867569077053}, {-0.7107522453625014, 6.23524169886981}, {-0.8143676049770572, 6.223891054021213}, {-0.9178019087945963, 6.210818042361487}, {-1.0210265178965452, 6.196025548657033}, {-1.1240128392773354, 6.179516934012478}, {-1.226732333769508, 6.161296035140685}, {-1.3291565239572198, 6.141367163500852}, {-1.4312570020757847, 6.119735104304942}, {-1.5330054378951872, 6.096405115392644}, {-1.6343735865858178, 6.071382925975025}, {-1.8358565173117192, 6.0162872108711545}, {-1.9359153071801858, 5.9862274873263726}, {-2.0354818411552493, 5.954503164133197}, {-2.2330274709748736, 5.886093430511772}, {-2.3309515694956127, 5.84942552651599}, {-2.4282734327943545, 5.8111280312797}, {-2.6210021108771993, 5.729684292926976}, {-2.7163551682037292, 5.686559189242503}, {-2.8109984903945953, 5.6418467677112085}, {-2.9980503974154753, 5.547707146145}, {-3.0904067045941104, 5.49830463104581}, {-3.181948735950986, 5.447364161510462}, {-3.362487721788383, 5.340923486500094}, {-3.451434112208559, 5.2854514081256765}, {-3.5394651128527537, 5.228497622091977}, {-3.712682435838626, 5.110205778776023}, {-4.047063159850806, 4.8565337797644785}, {-4.122645182492923, 4.79415648257077}, {-4.197265523523411, 4.730602594959925}, {-4.3435481069051365, 4.600024656671466}, {-4.415174398739613, 4.533031119901046}, {-4.485767116764715, 4.464922013397526}, {-4.623782561342972, 4.325421164415653}, {-4.691171272677557, 4.254062131757301}, {-4.757458388148116, 4.181652941335301}, {-4.886662618059223, 4.033752385383418}, {-5.131149515293267, 3.726114722629873}, {-5.189282115871627, 3.6468447324612976}, {-5.24618974385434, 3.5666675845326203}, {-5.3562737382972605, 3.403667777573203}, {-5.409422668304304, 3.320883647704474}, {-5.4612917596418615, 3.237269409288593}, {-5.561138862225673, 3.0676299776661433}, {-5.609091860013293, 2.981644971665413}, {-5.655714998472888, 2.8949102220045173}, {-5.744925119021489, 2.7192739705366495}, {-5.906892674117226, 2.3599218020163444}, {-5.943896546050095, 2.2685248359491132}, {-5.979485835197241, 2.1765475722878582}, {-6.046384599867881, 1.9909398963358842}, {-6.077676892891632, 1.8973537077321443}, {-6.107520242526666, 1.8032756583654146}, {-6.162829533784222, 1.6137338575576716}, {-6.188281057451078, 1.5183153446486564}, {-6.212254805554539, 1.4224954372370902}, {-6.234744448345129, 1.3262970286545162}, {-6.255744010833793, 1.2297431120350806}, {-6.27524787436526, 1.1328567748163474}, {-6.29325077810605, 1.0356611932157194}, {-6.30974782044693, 0.9381796266829849}, {-6.324734460319319, 0.8404354123307147}, {-6.338206518425263, 0.74245195934425}, {-6.3501601783807695, 0.6442527433718162}, {-6.360591987772127, 0.5458613008965087}, {-6.369498859124867, 0.4473012235919061}, {-6.37687807078523, 0.3485961526618557}, {-6.382727267713787, 0.24976977316619547}, {-6.387044462191024, 0.15084580833418118}, {-6.389828034434694, 0.05184801386617185}, {-6.391076733128766, -0.04719982777460518}, {-6.390789675863761, -0.1462739130787008}, {-6.388966349488403, -0.24535042321430844}, {-6.385606610372401, -0.3444055297571507}, {-6.380710684580323, -0.44341540042482797}, {-6.374279167956443, -0.5423562048140654}, {-6.366313026120545, -0.6412041201396601}, {-6.3568135943746125, -0.7399353369739248}, {-6.344790297445629, -0.8467675595442996}, {-6.330971079117379, -0.9534045603857345}, {-6.315359107034243, -1.0598161404285318}, {-6.297958056788917, -1.1659721519697328}, {-6.278772111089885, -1.2718425072191273}, {-6.257805958785049, -1.3773971868317756}, {-6.210554313583592, -1.587439835074829}, {-6.184280718283841, -1.691868183797142}, {-6.156250708616601, -1.7958616339986044}, {-6.094950742989733, -2.00242575897231}, {-6.061696676650667, -2.104937700236609}, {-6.026717971094908, -2.20689729267245}, {-5.951623837172927, -2.409043492379227}, {-5.911528225026096, -2.5091725197817127}, {-5.869747600892739, -2.6086340550276508}, {-5.781176254024507, -2.805441379711158}, {-5.734409190797251, -2.902731005367304}, {-5.68600442829365, -2.999240827338455}, {-5.58433429451318, -3.189810977147271}, {-5.531096325184639, -3.283816813749273}, {-5.476275453101627, -3.3769338797818076}, {-5.361944816103497, -3.5603953066448777}, {-5.11497159532062, -3.915496890216366}, {-5.049510822945881, -4.00166048148908}, {-4.982598215579057, -4.086729379858127}, {-4.844491155058921, -4.253485527717176}, {-4.773334557620012, -4.335124765090659}, {-4.700801826243243, -4.415573299152463}, {-4.551688074683019, -4.5728057824977375}, {-4.475148083630621, -4.649544335218423}, {-4.397314005612827, -4.725001404608682}, {-4.237849803342826, -4.87198414054031}, {-3.9043609992082287, -5.149635810127123}, {-3.818089432644706, -5.215531872717961}, {-3.730705438774188, -5.279982235011685}, {-3.5526974195141072, -5.404471130663248}, {-3.4621228055824482, -5.464473321170924}, {-3.370534573538936, -5.522957136283948}, {-3.1844193883820444, -5.635301560999553}, {-3.0899442183189643, -5.689129203499677}, {-2.9945589818310263, -5.741372546064137}, {-2.801164871340386, -5.841045219461054}, {-2.703209921441222, -5.888445113505787}, {-2.604452738281581, -5.93420184271939}, {-2.4046421872934576, -6.020731949773346}, {-2.3054916703959867, -6.0607520277468065}, {-2.2056749993521456, -6.099138833106239}, {-2.004151026570659, -6.170968709960048}, {-1.9024980949895827, -6.204390935405387}, {-1.8002877348990045, -6.236138202166548}, {-1.5943053092886283, -6.294571123917936}, {-1.4905889140352717, -6.32123955144969}, {-1.3864264157512671, -6.3461985704304205}, {-1.2818459958414627, -6.369440718676995}, {-1.1768759605270875, -6.39095899873392}, {-1.0715447331801051, -6.41074687996086}, {-0.9658808466224775, -6.428798300493729}, {-0.859912935392607, -6.445107669078694}, {-0.7536697279812318, -6.459669866778527}, {-0.6471800390387094, -6.472480248550858}, {-0.5404727615556111, -6.483534644697819}, {-0.43357685901893656, -6.492829362186642}, {-0.3265213575462617, -6.500361185840765}, {-0.21933533799977759, -6.506127379401118}, {-0.11204792808217402, -6.510125686457221}, {-0.004688294416706112, -6.512354331247831}, {0.1027143653862199, -6.512812019330785}, {0.21013083067396476, -6.511497938121903}, {0.3175318657061014, -6.5084117573027}, {0.4248882276160362, -6.503553629096745}, {0.5321706743774496, -6.496924188414551}, {0.6393499727735742, -6.488524552866943}, {0.7463969063673163, -6.478356322646765}, {0.853282283469867, -6.466421580278985}, {0.9599769451054412, -6.452722890239178}, {1.0664517729701537, -6.437263298440476}, {1.1726776973830493, -6.4200463315890195}, {1.2786257052269192, -6.401075996408064}, {1.3842668478765627, -6.3803567787309285}, {1.489572249112501, -6.357893642462971}, {1.594513113018165, -6.333692028412776}, {1.803186493935645, -6.280097506790249}, {1.9068618914257225, -6.250717853007178}, {2.010058528184772, -6.219626225772358}, {2.2149025399985427, -6.152338731059128}, {2.316493751052252, -6.116159869467721}, {2.417493888779027, -6.078303041921888}, {2.617610215559531, -5.997594582785486}, {2.7166714425442873, -5.954763640774759}, {2.8150316871659276, -5.910296106681101}, {3.009541237809751, -5.816497610646735}, {3.1056370252382908, -5.767190937395345}, {3.200924807847067, -5.71629624316169}, {3.3889715794400206, -5.609796203710224}, {3.4816787289188103, -5.554218645874271}, {3.5734742088623372, -5.497108634577646}, {3.7542290551920856, -5.378351504991336}, {4.103700693762514, -5.123133828310684}, {4.182763501352485, -5.060309405385681}, {4.260864283216488, -4.996264557267199}, {4.414104623877801, -4.8645728173274465}, {4.4892071786225545, -4.796956265460136}, {4.563273708472396, -4.728179961318222}, {4.708227284414906, -4.587211885531752}, {4.77907923878487, -4.515052696658282}, {4.848824993309385, -4.441798913521084}, {4.984930506852264, -4.292075675611815}, {5.243140574317128, -3.980257705240538}, {5.304684185965028, -3.8998310667338956}, {5.364993792249022, -3.81845262517853}, {5.481852387416478, -3.6529163148922152}, {5.538372805999241, -3.568797002143807}, {5.593602085597468, -3.4838029893286513}, {5.700133375413264, -3.3112703638304004}, {5.751409223701828, -3.22377202064791}, {5.801341615566782, -3.135479506997428}, {5.897127137533929, -2.9565946902635907}, {6.072056800197512, -2.5902153525839067}, {6.112258483373718, -2.4969516955130096}, {6.151027326295562, -2.4030639916942698}, {6.224228077016317, -2.2135046694597733}, {6.258641626110747, -2.1178775347185104}, {6.291585622133109, -2.021715310134571}, {6.353032019529628, -1.8278760773427734}, {6.3815188234903975, -1.7302446302305063}, {6.408504883494597, -1.632169204752598}, {6.4339832798640595, -1.533672875439746}, {6.457947447790732, -1.4347788246681525}, {6.48039117901744, -1.33551033720034}, {6.5013086234346265, -1.2358907946997724}, {6.520694290592606, -1.1359436702209915}, {6.53854305112911, -1.0356925226757427}, {6.554850138111654, -0.9351609912768502}, {6.56961114829436, -0.8343727899612949}, {6.582822043288941, -0.7333517017936286}, {6.594479150649572, -0.6321215733508547}, {6.604579164871314, -0.5307063090902921}, {6.613119148301822, -0.42912986570193695}, {6.620096531966097, -0.32741624644647277}, {6.6255091163041, -0.2255894954800808}, {6.629355071820968, -0.1236736921675769}, {6.631632939649655, -0.021692945385412058}, {6.632341632025858, 0.0803286121843035}, {6.631480432675079, 0.1823668297675905}, {6.62904899711168, 0.2843975442187103}, {6.625047352849853, 0.3863965857447666}, {6.619475899526422, 0.4883397836342917}, {6.612335408935385, 0.590202971988033}, {6.602821284631357, 0.7005348720600345}, {6.591465732792677, 0.8107135329583519}, {6.578271219650573, 0.9207082139134659}, {6.563240724747027, 1.030488213819902}, {6.546377740297269, 1.140022879809388}, {6.527686270409025, 1.249281615814002}, {6.507170830158706, 1.3582338911166405}, {6.484836444524739, 1.466849248886458}, {6.460688647178296, 1.5750973146968974}, {6.434733479131666, 1.6829478050239197}, {6.4069774872446486, 1.7903705357221054}, {6.377427722589335, 1.8973354304759806}, {6.346091738673551, 2.003812529224882}, {6.312977589523607, 2.109771996558134}, {6.241449501733653, 2.3200193687326442}, {6.203054154521958, 2.4242483011033835}, {6.16291782012091, 2.5278416736654488}, {6.07746476770938, 2.7330055639593014}, {6.032170551010817, 2.8345184378035073}, {5.9851803438365865, 2.935280480266101}, {5.886162229604305, 3.1344389104840493}, {5.8341606392110466, 3.232779242077342}, {5.780515684688182, 3.3302566457479483}, {5.668353435615157, 3.522513032924365}, {5.424969336257617, 3.8954784173747132}, {5.360247318216535, 3.9861600573068934}, {5.294009150761828, 4.075766104907805}, {5.157056309541496, 4.251650300904494}, {5.08637866205995, 4.337878638511354}, {5.014258907725043, 4.422931776317696}, {4.86577166506419, 4.5894162837791175}, {4.789444473662605, 4.670800388735324}, {4.711755758321195, 4.750914777912062}, {4.552378646328304, 4.9072436300032996}, {4.218240952859196, 5.2036862231278755}, {4.131634071651747, 5.2742948209656015}, {4.04384502619433, 5.343461638298754}, {3.864816805875691, 5.4773911457522875}, {3.7736266349521634, 5.542115448262802}, {3.6813522942370405, 5.605321206046815}, {3.4936525833399887, 5.727104852790979}, {3.39827870701806, 5.785647683078937}, {3.3019236344928715, 5.842601861381216}, {3.1063760521751265, 5.951678910341893}, {2.704689227278587, 6.15007527389496}, {2.6090500022405263, 6.192511537932919}, {2.512744851451928, 6.233460174254591}, {2.3182295166054057, 6.310853118449276}, {2.2200661090505953, 6.34727760670975}, {2.121330317497847, 6.3821748330918675}, {1.9222367932762283, 6.447351888058156}, {1.8219270137457695, 6.477614837681228}, {1.7211407449390472, 6.5063167718585495}, {1.6199022850658444, 6.533450187870745}, {1.5182360504753438, 6.559007961306302}, {1.416166569764649, 6.582983347921827}, {1.31371847785801, 6.605369985410514}, {1.2109165100583759, 6.626161895078352}, {1.107785496072293, 6.645353483427709}, {1.0043503540098202, 6.662939543647824}, {0.9006360843611025, 6.678915257011788}, {0.7966677639508762, 6.69327619417967}, {0.692470539872174, 6.706018316407473}, {0.5880696234008876, 6.717137976661574}, {0.4834902838929006, 6.726631920638275}, {0.37875784266484114, 6.734497287688303}, {0.273897666860127, 6.740731611645907}, {0.16893516330198144, 6.745332821562374}, {0.06389577233447905, 6.748299242343723}, {-0.04119503834664484, 6.749629595292421}, {-0.14631177987168598, 6.749322998552898}, {-0.25142894838209473, 6.7473789674608025}, {-0.35652103122133916, 6.743797414795794}, {-0.4615625131298119, 6.738578650937858}, {-0.5665278824427078, 6.731723383927015}, {-0.6713916372891257, 6.723232719426395}, {-0.7761282917906961, 6.713108160588662}, {-0.8807123822584193, 6.701351607825795}, {-0.9851184733863978, 6.687965358482214}, {-1.0893211644407639, 6.672952106411323}, {-1.193295095442057, 6.656314941455554}, {-1.2970149533399749, 6.638057348829975}, {-1.400455478178807, 6.618183208409595}, {-1.503591469251851, 6.5966967939205325}, {-1.6063977912437475, 6.573602772035149}, {-1.7088493803589895, 6.548906201371364}, {-1.810921250434882, 6.522612531396417}, {-1.912588499037884, 6.494727601235239}, {-2.013826313541653, 6.465257638383693}, {-2.214914875109567, 6.401589458063916}, {-2.3147165003728976, 6.367405624535849}, {-2.413990459939566, 6.331665522963344}, {-2.610858415132648, 6.255549481322819}, {-2.7084042477629398, 6.215190968806948}, {-2.8053261004915444, 6.17331103937209}, {-2.9972030770939693, 6.085025897675052}, {-3.092111185314848, 6.038641092933165}, {-3.186301293852364, 5.99077568160398}, {-3.372435271876928, 5.890647866006633}, {-3.7350825465162836, 5.673138090913001}, {-3.8218780255942715, 5.616391282529373}, {-3.9078111707368985, 5.558316272756519}, {-4.07700934160164, 5.438234388924998}, {-4.160234311463359, 5.376254659821295}, {-4.242516845153695, 5.313001013104371}, {-4.404176807173898, 5.182729599745669}, {-4.483515886787183, 5.115741392895955}, {-4.5618358418938225, 5.047538381311102}, {-4.7153441975467105, 4.90755026098825}, {-5.009328884662458, 4.613697606398207}, {-5.0800075695447955, 4.537440889366986}, {-5.1495251455479485, 4.460101205088848}, {-5.285010853011191, 4.30224392312744}, {-5.350946602870289, 4.221762451500284}, {-5.41565648838571, 4.140270256469531}, {-5.541336966388701, 3.9743286386510652}, {-5.602277424027098, 3.8899172768116803}, {-5.661931754846026, 3.804571304733507}, {-5.777324990240618, 3.6311541516085644}, {-5.992067354233606, 3.273981421826586}, {-6.042334349324335, 3.1826539856300333}, {-6.091209078364799, 3.090554642463302}, {-6.18473464064584, 2.9041253357023047}, {-6.229362737331, 2.8098383726752707}, {-6.272553100374832, 2.7148654938906596}, {-6.354578785176372, 2.5229498639873786}, {-6.393394025555661, 2.4260514504705917}, {-6.430731374443284, 2.3285557861528208}, {-6.500935977831816, 2.131863029810861}, {-6.533785884612157, 2.032711448513807}, {-6.565123208733918, 1.933053627335042}, {-6.62322925265483, 1.732311705512229}, {-6.649983426756248, 1.6312741202134928}, {-6.67519593042846, 1.529823314940689}, {-6.698860381310108, 1.4279828183113115}, {-6.720970756208331, 1.325776257520343}, {-6.74152139262122, 1.2232273528565403}, {-6.760506990176594, 1.1203599121947811}, {-6.777922611986678, 1.017197825466173}, {-6.793763685918221, 0.9137650591076475}, {-6.808026005777936, 0.8100856504914615}, {-6.820705732412822, 0.7061837023363369}, {-6.831799394725068, 0.602083377101968}, {-6.841303890601386, 0.4978088913673395}, {-6.849216487756444, 0.3933845101945896}, {-6.855534824490167, 0.2888345414801733}, {-6.860256910358759, 0.18418333029376635}, {-6.86338112675923, 0.0794552532066647}, {-6.864962059729965, -0.034227703027965406}, {-6.864659569153705, -0.14794122340454433}, {-6.862473076657067, -0.2616540977748769}, {-6.858402521161848, -0.3753351052270834}, {-6.852448359082685, -0.488953022661215}, {-6.844611564382507, -0.6024766333687024}, {-6.834893628485708, -0.7158747356130349}, {-6.8232965600490605, -0.829116151209069}, {-6.809822884590303, -0.9421697340993324}, {-6.794475643974538, -1.055004378924043}, {-6.777258395758513, -1.1675890295829212}, {-6.758175212392829, -1.2798926877868708}, {-6.737230680282392, -1.3918844215962554}, {-6.714429898705201, -1.5035333739441428}, {-6.66328254115149, -1.725679931364729}, {-6.634948716388171, -1.8361162731149998}, {-6.604784141435225, -1.9460873236610008}, {-6.538993814341903, -2.1645122544826685}, {-6.503384855399897, -2.272905808680176}, {-6.465978728399644, -2.3807134361845566}, {-6.385814037641393, -2.594451856471327}, {-6.343076240842102, -2.700323527009186}, {-6.298582804537588, -2.805491042724579}, {-6.204375914177885, -3.013597323720075}, {-5.995437239696586, -3.4200920417294216}, {-5.939006074703304, -3.519526403146921}, {-5.8809254458089955, -3.6180294564053592}, {-5.759877743788973, -3.812132421719381}, {-5.6969427756290525, -3.9076783675344986}, {-5.632422544990187, -4.0021850877170735}, {-5.498695390323527, -4.187975905330938}, {-5.429524100268363, -4.279208245992736}, {-5.358838806187469, -4.369297861903672}, {-5.213002152616853, -4.545948711884884}, {-4.904019281217801, -4.884453293747201}, {-4.82328766055502, -4.9658581556122865}, {-4.7412044793334625, -5.045928373653476}, {-4.573072084435668, -5.201975466588356}, {-4.487068121276697, -5.277908554363543}, {-4.39980308641365, -5.35241943636295}, {-4.221584243916358, -5.497091182709921}, {-4.130678521045782, -5.567211325614517}, {-4.038607883022172, -5.635827831316372}, {-3.8510716896479633, -5.768472917140207}, {-3.4631360607501342, -6.014895637864234}, {-3.370309826922408, -6.068669296026081}, {-3.2766484803439133, -6.121010091035028}, {-3.0869091400094155, -6.221341547306786}, {-2.9908759902883917, -6.269307293628936}, {-2.8940974052841404, -6.315790353097025}, {-2.6983956958452278, -6.40426241910935}, {-2.59951889944514, -6.446229316423144}, {-2.499989312859771, -6.486669313938077}, {-2.299066263466501, -6.562928352314666}, {-2.1977204372055934, -6.598728177886073}, {-2.0958170825549, -6.6329626779082025}, {-1.8904346520108046, -6.696701340597132}, {-1.7870043404545843, -6.72618925917108}, {-1.6831140176000363, -6.754079368134747}, {-1.5787883718721594, -6.780364490826415}, {-1.4740522036639812, -6.805037832136214}, {-1.3689304194384462, -6.828092980253061}, {-1.2634480258032874, -6.849523908320351}, {-1.1576301235598614, -6.8693249760000725}, {-1.051501901727555, -6.887490930944899}, {-0.9450886315453847, -6.90401691017786}, {-0.8384156604517718, -6.918898441379283}, {-0.7315084060441815, -6.932131444080632}, {-0.6243923500203034, -6.9437122307649055}, {-0.5170930321017793, -6.953637507873349}, {-0.4096360439421277, -6.961904376718199}, {-0.3020470230205103, -6.968510334301177}, {-0.19435164652235798, -6.973453274037562}, {-0.08657562520856256, -6.976731486385595}, {0.021255302725103058, -6.9783436593810535}, {0.1291153777970661, -6.978288879076826}, {0.23697882539626605, -6.976566629887373}, {0.34481986194402736, -6.973176794837948}, {0.45261270105977097, -6.968119655718478}, {0.5603315597295354, -6.961395893142061}, {0.6679506644756346, -6.953006586508022}, {0.7754442575257743, -6.94295321386951}, {0.8827866029805995, -6.93123765170564}, {0.989951992977944, -6.917862174598176}, {1.0969147538520598, -6.9028294548128715}, {1.2036492522867965, -6.8861425617854595}, {1.3101299014610535, -6.867804961512427}, {1.4163311671848384, -6.847820515846715}, {1.522227574024902, -6.826193481698411}, {1.627793711418228, -6.802928510140626}, {1.7330042397717211, -6.7780306454207935}, {1.942257502327795, -6.723358372763432}, {2.0462499668719825, -6.693596008973534}, {2.1497862951404723, -6.662224837680691}, {2.3553910747497024, -6.594684425823521}, {2.457410066006613, -6.5585303234042085}, {2.5588740127223732, -6.520797686391637}, {2.760039186064759, -6.440631278054193}, {3.1546432044370545, -6.261756271080441}, {3.2627674987644877, -6.207492684215817}, {3.369958199026309, -6.151361698215505}, {3.581410062407557, -6.033562349896166}, {3.685607549037663, -5.971927786698853}, {3.7887441078038933, -5.908493412920184}, {3.9917103792991337, -5.77629883510858}, {4.091478856496522, -5.7075767696452155}, {4.1900639532453825, -5.6371311597421725}, {4.383565233875921, -5.491151349627696}, {4.755080901797553, -5.179440147448991}, {4.844568757163332, -5.097528814105304}, {4.932645309268137, -5.014071054177689}, {5.104458008838913, -4.842614002698241}, {5.188141964955721, -4.754664733129435}, {5.270310250966596, -4.665269066348315}, {5.4300002465196355, -4.48224347681025}, {5.5074733065488495, -4.388667094931425}, {5.583333412264674, -4.293751381931831}, {5.730122607925555, -4.100013595127985}, {6.0033591037380605, -3.6977159670723445}, {6.067307433089617, -3.5942254024786076}, {6.129471347966497, -3.4896287131692736}, {6.248369920915065, -3.277240386737417}, {6.305067905111807, -3.1695112813900166}, {6.359908138385678, -3.060801096460831}, {6.463947989077911, -2.8405659532982335}, {6.513115312037093, -2.7291059529670267}, {6.560360305165531, -2.6167947696927185}, {6.649024922439581, -2.3897517529654264}, {6.690416791647683, -2.275086997003272}, {6.729830829819715, -2.1597051932551756}, {6.802676308048132, -1.9269271463724784}, {6.836084671350128, -1.809599783241861}, {6.867469057392981, -1.6916931119437042}, {6.896819419633581, -1.573242054511502}, {6.9241263142015725, -1.454281707225399}, {6.9493809029217, -1.3348473302092971}, {6.972574956156528, -1.214974336975534}, {6.993700855468629, -1.09469828392044}, {7.012751596101444, -0.9740548597734409}, {7.029720789277976, -0.8530798750030848}, {7.044602664316517, -0.7318092511833401}, {7.057392070562791, -0.6102790103231255}, {7.068084479137863, -0.4885252641620431}, {7.076675984501212, -0.3665842034356933}, {7.083163305828412, -0.24449208711396733}, {7.087543788202994, -0.12228523161531736}, {7.0898154036220635, -1.3892031584946177*^-14}}]}, "Charting`Private`Tag$24799#1"]}}, {}, {{{}, {}, {}, {}}, {}}}, {}}, {DisplayFunction -> Identity, PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.02], Scaled[0.02]}}, PlotRangeClipping -> True, ImagePadding -> All, PlotInteractivity :> $PlotInteractivity, PlotRange -> {{Automatic, Automatic}, {Automatic, Automatic}}, PlotRangePadding -> Automatic, DisplayFunction -> Identity, PlotInteractivity :> $PlotInteractivity, DefaultBaseStyle -> {"PlotGraphics", "Graphics"}, Axes -> True, AxesOrigin -> {0, 0}, CoordinatesToolOptions -> {"DisplayFunction" -> ({Sqrt[#1[[1]]^2 + #1[[2]]^2], Mod[ArcTan[#1[[1]], #1[[2]]], 2*Pi]} & ), "CopiedValueFunction" -> ({Sqrt[#1[[1]]^2 + #1[[2]]^2], Mod[ArcTan[#1[[1]], #1[[2]]], 2*Pi]} & )}, DisplayFunction :> Identity, FrameTicks -> {{Automatic, Automatic}, {Automatic, Automatic}}, GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]], Method -> {"AxisPadding" -> Scaled[0.02], "DefaultBoundaryStyle" -> Automatic, "DefaultGraphicsInteraction" -> {"Version" -> 1.2, "TrackMousePosition" -> {True, False}, "Effects" -> {"Highlight" -> {"ratio" -> 2}, "HighlightPoint" -> {"ratio" -> 2}, "Droplines" -> {"freeformCursorMode" -> True, "placement" -> {"x" -> "All", "y" -> "None"}}}}, "DefaultMeshStyle" -> AbsolutePointSize[6], "DefaultPlotStyle" -> {Directive[RGBColor[0.24, 0.6, 0.8], AbsoluteThickness[2]], Directive[RGBColor[0.95, 0.627, 0.1425], AbsoluteThickness[2]], Directive[RGBColor[0.455, 0.7, 0.21], AbsoluteThickness[2]], Directive[RGBColor[0.922526, 0.385626, 0.209179], AbsoluteThickness[2]], Directive[RGBColor[0.578, 0.51, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.772079, 0.431554, 0.102387], AbsoluteThickness[2]], Directive[RGBColor[0.4, 0.64, 1.], AbsoluteThickness[2]], Directive[RGBColor[1., 0.75, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.8, 0.4, 0.76], AbsoluteThickness[2]], Directive[RGBColor[0.637, 0.65, 0.], AbsoluteThickness[2]], Directive[RGBColor[0.915, 0.3325, 0.2125], AbsoluteThickness[2]], Directive[RGBColor[0.40082222609352647, 0.5220066643438841, 0.85], AbsoluteThickness[2]], Directive[RGBColor[0.9728288904374106, 0.621644452187053, 0.07336199581899142], AbsoluteThickness[2]], Directive[RGBColor[0.736782672705901, 0.358, 0.5030266573755369], AbsoluteThickness[2]], Directive[RGBColor[0.28026441037696703, 0.715, 0.4292089322474965], AbsoluteThickness[2]]}, "DomainPadding" -> Scaled[0.02], "RangePadding" -> Scaled[0.05]}, PlotRange -> {{-6.864962059729965, 7.0898154036220635}, {-6.9783436593810535, 6.749629595292421}}, PlotRangeClipping -> True, PlotRangePadding -> {Scaled[0.02], Scaled[0.02]}, Ticks -> {Automatic, Automatic}}]"#,
    );
  }
  #[test]
  fn arc_cos_1() {
    assert_case(r#"ArcCos[1]"#, r#"0"#);
  }
  #[test]
  fn arc_cos_2() {
    assert_case(r#"ArcCos[1]; ArcCos[0]"#, r#"Pi / 2"#);
  }
  #[test]
  fn arc_cot_1() {
    assert_case(r#"ArcCot[0]"#, r#"Pi / 2"#);
  }
  #[test]
  fn arc_cot_2() {
    assert_case(r#"ArcCot[0]; ArcCot[1]"#, r#"Pi / 4"#);
  }
  #[test]
  fn arc_csc_1() {
    assert_case(r#"ArcCsc[1]"#, r#"Pi / 2"#);
  }
  #[test]
  fn arc_csc_2() {
    assert_case(r#"ArcCsc[1]; ArcCsc[-1]"#, r#"-1/2*Pi"#);
  }
  #[test]
  fn arc_sec_1() {
    assert_case(r#"ArcSec[1]"#, r#"0"#);
  }
  #[test]
  fn arc_sec_2() {
    assert_case(r#"ArcSec[1]; ArcSec[-1]"#, r#"Pi"#);
  }
  #[test]
  fn arc_csc_zero_is_complex_infinity() {
    // ArcCsc[0] = ArcSin[1/0] = ComplexInfinity (and must not emit a spurious
    // Power::infy message on the way).
    assert_case(r#"ArcCsc[0]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn arc_sec_zero_is_complex_infinity() {
    assert_case(r#"ArcSec[0]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn csch_zero_is_complex_infinity() {
    // Csch[0] = 1/Sinh[0] = ComplexInfinity.
    assert_case(r#"Csch[0]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn arc_sin_1() {
    assert_case(r#"ArcSin[0]"#, r#"0"#);
  }
  #[test]
  fn arc_sin_2() {
    assert_case(r#"ArcSin[0]; ArcSin[1]"#, r#"Pi / 2"#);
  }
  #[test]
  fn cos_1() {
    assert_case(r#"Cos[3 Pi]"#, r#"-1"#);
  }
  #[test]
  fn cot_1() {
    assert_case(r#"Cot[0]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn cot_2() {
    assert_case(r#"Cot[0]; Cot[1.]"#, r#"0.6420926159343308"#);
  }
  #[test]
  fn csc_1() {
    assert_case(r#"Csc[0]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn csc_2() {
    assert_case(r#"Csc[0]; Csc[1] (* Csc[1] in Mathematica *)"#, r#"Csc[1]"#);
  }
  #[test]
  fn csc_3() {
    assert_case(
      r#"Csc[0]; Csc[1] (* Csc[1] in Mathematica *); Csc[1.]"#,
      r#"1.1883951057781212"#,
    );
  }
  #[test]
  fn haversine_1() {
    assert_case(r#"Haversine[1.5]"#, r#"0.4646313991661485"#);
  }
  #[test]
  fn haversine_2() {
    assert_case(
      r#"Haversine[1.5]; Haversine[0.5 + 2I]"#,
      r#"-1.150818666457047 + 0.8694047522371582*I"#,
    );
  }
  #[test]
  fn inverse_haversine_1() {
    assert_case(r#"InverseHaversine[0.5]"#, r#"1.5707963267948968"#);
  }
  #[test]
  fn inverse_haversine_2() {
    assert_case(
      r#"InverseHaversine[0.5]; InverseHaversine[1 + 2.5 I]"#,
      r#"1.764589463349829 + 2.3309746530493123*I"#,
    );
  }
  // Sinc[0] = 1, but an inexact zero gives the machine real 1., not the exact
  // integer 1. Per wolframscript.
  #[test]
  fn sinc_exact_zero() {
    assert_case(r#"Sinc[0]"#, r#"1"#);
  }
  #[test]
  fn sinc_inexact_zero() {
    assert_case(r#"Sinc[0.0]"#, r#"1."#);
    assert_case(r#"Head[Sinc[0.0]]"#, r#"Real"#);
  }
  #[test]
  fn sec_1() {
    assert_case(r#"Sec[0]"#, r#"1"#);
  }
  #[test]
  fn sec_2() {
    assert_case(r#"Sec[0]; Sec[1] (* Sec[1] in Mathematica *)"#, r#"Sec[1]"#);
  }
  #[test]
  fn sec_3() {
    assert_case(
      r#"Sec[0]; Sec[1] (* Sec[1] in Mathematica *); Sec[1.]"#,
      r#"1.8508157176809255"#,
    );
  }
  #[test]
  fn sin_1() {
    assert_case(r#"Sin[0]"#, r#"0"#);
  }
  #[test]
  fn sin_2() {
    assert_case(r#"Sin[0]; Sin[0.5]"#, r#"0.479425538604203"#);
  }
  #[test]
  fn sin_3() {
    assert_case(r#"Sin[0]; Sin[0.5]; Sin[3 Pi]"#, r#"0"#);
  }
  #[test]
  fn sin_4() {
    assert_case(
      r#"Sin[0]; Sin[0.5]; Sin[3 Pi]; Sin[1.0 + I]"#,
      r#"1.2984575814159773 + 0.6349639147847361*I"#,
    );
  }
  #[test]
  fn tan_1() {
    assert_case(r#"Tan[0]"#, r#"0"#);
  }
  #[test]
  fn tan_2() {
    assert_case(r#"Tan[0]; Tan[Pi / 2]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn coefficient_arrays() {
    assert_case(
      r#"CoefficientArrays[1 + x^3, x]; CoefficientArrays[1 + x y+ x^3, {x, y}]; CoefficientArrays[{1 + x^2, x y}, {x, y}]; CoefficientArrays[(x+y+Sin[z])^3, {x,y}]"#,
      r#"{Sin[z]^3, SparseArray[Automatic, {2}, 0, {1, {{0, 2}, {{2}, {1}}}, {3*Sin[z]^2, 3*Sin[z]^2}}], SparseArray[Automatic, {2, 2}, 0, {1, {{0, 2, 3}, {{1}, {2}, {2}}}, {3*Sin[z], 6*Sin[z], 3*Sin[z]}}], SparseArray[Automatic, {2, 2, 2}, 0, {1, {{0, 3, 4}, {{1, 2}, {2, 2}, {1, 1}, {2, 2}}}, {3, 3, 1, 1}}]}"#,
    );
  }
  #[test]
  fn expand_all_1() {
    assert_case(
      r#"ExpandAll[(a + b) ^ 2 / (c + d)^2]; ExpandAll[(a + Sin[x (1 + y)])^2]"#,
      r#"a^2 + 2*a*Sin[x + x*y] + Sin[x + x*y]^2"#,
    );
  }
  #[test]
  fn expand_all_2() {
    assert_case(
      r#"ExpandAll[(a + b) ^ 2 / (c + d)^2]; ExpandAll[(a + Sin[x (1 + y)])^2]; ExpandAll[Sin[(x+y)^2]]"#,
      r#"Sin[x^2 + 2*x*y + y^2]"#,
    );
  }
  #[test]
  fn expand_all_3() {
    assert_case(
      r#"ExpandAll[(a + b) ^ 2 / (c + d)^2]; ExpandAll[(a + Sin[x (1 + y)])^2]; ExpandAll[Sin[(x+y)^2]]; ExpandAll[Sin[(x+y)^2], Trig->True]"#,
      r#"Cos[x*y]^2*Cos[y^2]*Sin[x^2] + 2*Cos[x^2]*Cos[x*y]*Cos[y^2]*Sin[x*y] - Cos[y^2]*Sin[x^2]*Sin[x*y]^2 + Cos[x^2]*Cos[x*y]^2*Sin[y^2] - 2*Cos[x*y]*Sin[x^2]*Sin[x*y]*Sin[y^2] - Cos[x^2]*Sin[x*y]^2*Sin[y^2]"#,
    );
  }
  #[test]
  fn expand_all_4() {
    assert_case(
      r#"ExpandAll[(a + b) ^ 2 / (c + d)^2]; ExpandAll[(a + Sin[x (1 + y)])^2]; ExpandAll[Sin[(x+y)^2]]; ExpandAll[Sin[(x+y)^2], Trig->True]; ExpandAll[((1 + x)(1 + y))[x]]"#,
      r#"(1 + x + y + x*y)[x]"#,
    );
  }
  #[test]
  fn expand_all_5() {
    assert_case(
      r#"ExpandAll[(a + b) ^ 2 / (c + d)^2]; ExpandAll[(a + Sin[x (1 + y)])^2]; ExpandAll[Sin[(x+y)^2]]; ExpandAll[Sin[(x+y)^2], Trig->True]; ExpandAll[((1 + x)(1 + y))[x]]; ExpandAll[(1 + a) ^ 6 / (x + y)^3, Modulus -> 3]"#,
      r#"(1 + 2*a^3 + a^6)/(x^3 + y^3)"#,
    );
  }
  #[test]
  fn variables() {
    assert_case(
      r#"Variables[a x^2 + b x + c]; Variables[{a + b x, c y^2 + x/2}]; Variables[x + Sin[y]]"#,
      r#"{x, Sin[y]}"#,
    );
  }
  #[test]
  fn cos_2() {
    assert_case(r#"Cos[60 Degree]"#, r#"1 / 2"#);
  }
  #[test]
  fn equal_1() {
    assert_case(r#"Cos[60 Degree]; Degree == Pi / 180"#, r#"True"#);
  }
  #[test]
  fn equal_2() {
    assert_case(
      r#"MachinePrecision -Log[10., $MinMachineNumber]==Accuracy[0`]"#,
      r#"True"#,
    );
  }
  #[test]
  fn arc_cosh_1() {
    assert_case(r#"ArcCosh[0]"#, r#"I/2*Pi"#);
  }
  #[test]
  fn arc_cosh_2() {
    assert_case(r#"ArcCosh[0]; ArcCosh[0.]"#, r#"0. + 1.5707963267948966*I"#);
  }
  #[test]
  fn arc_cosh_3() {
    assert_case(
      r#"ArcCosh[0]; ArcCosh[0.]; ArcCosh[0.00000000000000000000000000000000000000]"#,
      r#"1.5707963267948966192313216916397514420985846996875529104875`38.*I"#,
    );
  }
  #[test]
  fn arc_coth_1() {
    assert_case(r#"ArcCoth[0]"#, r#"I/2*Pi"#);
  }
  #[test]
  fn arc_coth_2() {
    assert_case(r#"ArcCoth[0]; ArcCoth[1]"#, r#"Infinity"#);
  }
  #[test]
  fn arc_coth_3() {
    assert_case(
      r#"ArcCoth[0]; ArcCoth[1]; ArcCoth[0.0]"#,
      r#"0. + 1.5707963267948966*I"#,
    );
  }
  #[test]
  fn arc_coth_4() {
    assert_case(
      // Round to avoid a last-ULP libm divergence between platforms in the
      // real part (macOS ...549 vs Linux ...548).
      r#"ArcCoth[0]; ArcCoth[1]; ArcCoth[0.0]; Round[ArcCoth[0.5], 0.0001]"#,
      r#"0.5493 - 1.5708*I"#,
    );
  }
  #[test]
  fn arc_csch_1() {
    assert_case(r#"ArcCsch[0]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn arc_csch_2() {
    assert_case(r#"ArcCsch[0]; ArcCsch[1.0]"#, r#"0.881373587019543"#);
  }
  #[test]
  fn arc_sech_1() {
    assert_case(r#"ArcSech[0]"#, r#"Infinity"#);
  }
  #[test]
  fn arc_sech_2() {
    assert_case(r#"ArcSech[0]; ArcSech[1]"#, r#"0"#);
  }
  #[test]
  fn arc_sech_3() {
    assert_case(
      r#"ArcSech[0]; ArcSech[1]; ArcSech[0.5]"#,
      r#"1.3169578969248166"#,
    );
  }
  #[test]
  fn arc_sinh_1() {
    assert_case(r#"ArcSinh[0]"#, r#"0"#);
  }
  #[test]
  fn arc_sinh_2() {
    assert_case(r#"ArcSinh[0]; ArcSinh[0.]"#, r#"0."#);
  }
  #[test]
  fn arc_sinh_3() {
    assert_case(
      r#"ArcSinh[0]; ArcSinh[0.]; ArcSinh[1.0]"#,
      r#"0.881373587019543"#,
    );
  }
  // ArcSinh is odd: ArcSinh[-x] -> -ArcSinh[x] for negative integers,
  // rationals, and negated symbolic arguments.
  #[test]
  fn arc_sinh_odd_reflection() {
    assert_case(r#"ArcSinh[-1]"#, r#"-ArcSinh[1]"#);
    assert_case(r#"ArcSinh[-2]"#, r#"-ArcSinh[2]"#);
    assert_case(r#"ArcSinh[-1/2]"#, r#"-ArcSinh[1/2]"#);
    assert_case(r#"ArcSinh[-x]"#, r#"-ArcSinh[x]"#);
  }
  // ArcTanh is odd too (reals stay on the numeric path).
  #[test]
  fn arc_tanh_odd_reflection() {
    assert_case(r#"ArcTanh[-1/2]"#, r#"-ArcTanh[1/2]"#);
    assert_case(r#"ArcTanh[-2]"#, r#"-ArcTanh[2]"#);
    assert_case(r#"ArcTanh[-x]"#, r#"-ArcTanh[x]"#);
  }
  #[test]
  fn arc_tanh_1() {
    assert_case(r#"ArcTanh[0]"#, r#"0"#);
  }
  #[test]
  fn arc_tanh_2() {
    assert_case(r#"ArcTanh[0]; ArcTanh[1]"#, r#"Infinity"#);
  }
  #[test]
  fn arc_tanh_3() {
    assert_case(r#"ArcTanh[0]; ArcTanh[1]; ArcTanh[0]"#, r#"0"#);
  }
  #[test]
  fn arc_tanh_4() {
    assert_case(
      r#"ArcTanh[0]; ArcTanh[1]; ArcTanh[0]; ArcTanh[.5 + 2 I]"#,
      r#"0.09641562020299617 + 1.1265564408348223*I"#,
    );
  }
  #[test]
  fn arc_tanh_5() {
    assert_case(
      r#"ArcTanh[0]; ArcTanh[1]; ArcTanh[0]; ArcTanh[.5 + 2 I]; ArcTanh[2 + I]"#,
      r#"ArcTanh[2 + I]"#,
    );
  }
  #[test]
  fn cosh() {
    assert_case(r#"Cosh[0]"#, r#"1"#);
  }
  #[test]
  fn coth() {
    assert_case(r#"Coth[0]"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn csch() {
    assert_case(r#"Csch[1.8]"#, r#"0.3398846914154937"#);
  }
  #[test]
  fn gudermannian_1() {
    assert_case(r#"Gudermannian[4.2]"#, r#"1.5408074208608435"#);
  }
  #[test]
  fn gudermannian_2() {
    assert_case(
      r#"Gudermannian[4.2]; Gudermannian[-4.2] ==  -Gudermannian[4.2]"#,
      r#"True"#,
    );
  }
  #[test]
  fn inverse_gudermannian_1() {
    assert_case(r#"InverseGudermannian[.5]"#, r#"0.5222381032784403"#);
  }
  #[test]
  fn inverse_gudermannian_2() {
    assert_case(
      r#"InverseGudermannian[.5]; InverseGudermannian[-.5] ==  -InverseGudermannian[.5]"#,
      r#"True"#,
    );
  }
  #[test]
  fn sech() {
    assert_case(r#"Sech[0]"#, r#"1"#);
  }
  #[test]
  fn sinh() {
    assert_case(r#"Sinh[0]"#, r#"0"#);
  }
  #[test]
  fn tanh() {
    assert_case(r#"Tanh[0]"#, r#"0"#);
  }
  #[test]
  fn exp_1() {
    assert_case(r#"Exp[1]"#, r#"E"#);
  }
  #[test]
  fn exp_2() {
    assert_case(r#"Exp[1]; Exp[10.0]"#, r#"22026.465794806718"#);
  }
  #[test]
  fn exp_3() {
    assert_case(
      r#"Exp[1]; Exp[10.0]; Exp[x] // FullForm"#,
      r#"FullForm[E^x]"#,
    );
  }
  #[test]
  fn log_1() {
    assert_case(
      r#"Log[{0, 1, E, E * E, E ^ 3, E ^ x}]"#,
      r#"{-Infinity, 0, 1, 2, 3, Log[E ^ x]}"#,
    );
  }
  #[test]
  fn log_2() {
    assert_case(
      r#"Log[{0, 1, E, E * E, E ^ 3, E ^ x}]; Log[0.]"#,
      r#"Indeterminate"#,
    );
  }
  #[test]
  fn log2_1() {
    assert_case(r#"Log2[4 ^ 8]"#, r#"16"#);
  }
  #[test]
  fn log2_2() {
    assert_case(r#"Log2[4 ^ 8]; Log2[5.6]"#, r#"2.4854268271702415"#);
  }
  #[test]
  fn log2_3() {
    assert_case(r#"Log2[4 ^ 8]; Log2[5.6]; Log2[E ^ 2]"#, r#"2 / Log[2]"#);
  }
  #[test]
  fn log10_1() {
    assert_case(r#"Log10[1000]"#, r#"3"#);
  }
  #[test]
  fn log10_2() {
    assert_case(
      r#"Log10[1000]; Log10[{2., 5.}]"#,
      r#"{0.3010299956639812, 0.6989700043360189}"#,
    );
  }
  #[test]
  fn log10_3() {
    assert_case(
      r#"Log10[1000]; Log10[{2., 5.}]; Log10[E ^ 3]"#,
      r#"3 / Log[10]"#,
    );
  }
  #[test]
  fn logistic_sigmoid_1() {
    assert_case(r#"LogisticSigmoid[0.5]"#, r#"0.6224593312018546"#);
  }
  #[test]
  fn logistic_sigmoid_2() {
    assert_case(
      r#"LogisticSigmoid[0.5]; LogisticSigmoid[0.5 + 2.3 I]"#,
      r#"1.0647505893884985 + 0.8081774171575826*I"#,
    );
  }
  #[test]
  fn logistic_sigmoid_3() {
    assert_case(
      r#"LogisticSigmoid[0.5]; LogisticSigmoid[0.5 + 2.3 I]; LogisticSigmoid[{-0.2, 0.1, 0.3}]"#,
      r#"{0.4501660026875221, 0.52497918747894, 0.574442516811659}"#,
    );
  }
  #[test]
  fn curl_1() {
    assert_case(
      r#"Curl[{y, -x}, {x, y}]; v[x_, y_] := {Cos[x] Sin[y], Cos[y] Sin[x]}; Curl[v[x, y], {x, y}]"#,
      r#"0"#,
    );
  }
  #[test]
  fn curl_2() {
    assert_case(
      r#"Curl[{y, -x}, {x, y}]; v[x_, y_] := {Cos[x] Sin[y], Cos[y] Sin[x]}; Curl[v[x, y], {x, y}]; Curl[{y, -x, 2 z}, {x, y, z}]"#,
      r#"{0, 0, -2}"#,
    );
  }
  #[test]
  fn sin_5() {
    assert_case(r#"Sin[1]"#, r#"Sin[1]"#);
  }
  #[test]
  fn symbol_literal() {
    assert_case(r#"Sin[1]; a"#, r#"a"#);
  }
  #[test]
  fn sin_6() {
    assert_case(r#"Sin[1]; a; Sin[a]"#, r#"Sin[a]"#);
  }
  #[test]
  fn sin_7() {
    assert_case(
      r#"Sin[1]; a; Sin[a]; NumericQ[a]=True; Sin[a]"#,
      r#"Sin[a]"#,
    );
  }
  #[test]
  fn cos_3() {
    assert_case(
      r#"ArcTan[ComplexInfinity]; ArcTan[-1, 1]; ArcTan[1, -1]; ArcTan[-1, -1]; ArcTan[1, 0]; ArcTan[-1, 0]; ArcTan[0, 1]; ArcTan[0, -1]; Cos[1.5 Pi]"#,
      r#"-1.8369701987210297*^-16"#,
    );
  }
  #[test]
  fn tan_3() {
    assert_case(
      r#"ArcTan[ComplexInfinity]; ArcTan[-1, 1]; ArcTan[1, -1]; ArcTan[-1, -1]; ArcTan[1, 0]; ArcTan[-1, 0]; ArcTan[0, 1]; ArcTan[0, -1]; Cos[1.5 Pi]; N[Sin[1], 40]; Tan[0.5 Pi]"#,
      r#"1.633123935319537*^16"#,
    );
  }
  #[test]
  fn arc_cosh_4() {
    assert_case(r#"ArcCosh[1.4]"#, r#"0.867014726490565"#);
  }
  #[test]
  fn arc_coth_5() {
    assert_case(
      r#"ArcCosh[1.4]; ArcCoth[0.000000000000000000000000000000000000000]"#,
      r#"1.5707963267948966192313216916397514420985846996875529104875`39.*I"#,
    );
  }
  #[test]
  fn power_1() {
    assert_case(
      r#"E^(3+I Pi); E^(I Pi/2); E^1; log2=Log[2.]; E^log2"#,
      r#"2."#,
    );
  }
  #[test]
  fn divide() {
    assert_case(
      r#"E^(3+I Pi); E^(I Pi/2); E^1; log2=Log[2.]; E^log2; log2=Log[2.]; Chop[E^(log2+I Pi)]; log2=.; E^(I Pi/4)"#,
      r#"E^(I/4*Pi)"#,
    );
  }
  #[test]
  fn power_2() {
    assert_case(
      r#"E^(3+I Pi); E^(I Pi/2); E^1; log2=Log[2.]; E^log2; log2=Log[2.]; Chop[E^(log2+I Pi)]; log2=.; E^(I Pi/4); E^(.25 I Pi)"#,
      r#"0.7071067811865476 + 0.7071067811865475*I"#,
    );
  }
  #[test]
  fn exp_4() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]"#,
      r#"E^a"#,
    );
  }
  #[test]
  fn exp_5() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]"#,
      r#"9.974182454814718"#,
    );
  }
  #[test]
  fn log_3() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]"#,
      r#"-Log[2]"#,
    );
  }
  #[test]
  fn exp_6() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]; Exp[I]"#,
      r#"E^I"#,
    );
  }
  #[test]
  fn log_4() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]; Exp[I]; Log[3]"#,
      r#"Log[3]"#,
    );
  }
  #[test]
  fn log_5() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]; Exp[I]; Log[3]; Log[I]"#,
      r#"I/2*Pi"#,
    );
  }
  #[test]
  fn sin_8() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]; Exp[I]; Log[3]; Log[I]; Abs[a]; Abs[0]; Abs[1+3 I]; Sin[Pi]"#,
      r#"0"#,
    );
  }
  #[test]
  fn exp_7() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]"#,
      r#"E^a"#,
    );
  }
  #[test]
  fn exp_8() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]"#,
      r#"9.974182454814718"#,
    );
  }
  #[test]
  fn log_6() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]"#,
      r#"-Log[2]"#,
    );
  }
  #[test]
  fn exp_9() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]; Exp[I]"#,
      r#"E^I"#,
    );
  }
  #[test]
  fn log_7() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]; Exp[I]; Log[3]"#,
      r#"Log[3]"#,
    );
  }
  #[test]
  fn log_8() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]; Exp[I]; Log[3]; Log[I]"#,
      r#"I/2*Pi"#,
    );
  }
  #[test]
  fn sin_9() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2); Exp[a]; Exp[2.3]; Log[1/2]; Exp[I]; Log[3]; Log[I]; Abs[a]; Abs[0]; Abs[1+3 I]; Sin[Pi]"#,
      r#"0"#,
    );
  }
  #[test]
  fn legendre_q_0_symbolic() {
    assert_case(r#"LegendreQ[0, x]"#, r#"-1/2*Log[1 - x] + Log[1 + x]/2"#);
  }
  #[test]
  fn legendre_q_1_symbolic() {
    assert_case(
      r#"LegendreQ[1, x]"#,
      r#"-1 + x*(-1/2*Log[1 - x] + Log[1 + x]/2)"#,
    );
  }
  #[test]
  fn legendre_q_2_symbolic_at_half() {
    // Verify Q_2 closed form by substituting x=1/2. wolframscript prints
    // this as -3/4 + (-1/2*Log[3/2] - Log[2]/2)/8 — mathematically equal.
    assert_case(
      r#"LegendreQ[2, x] /. x -> 1/2"#,
      r#"-3/4 - (Log[2]/2 + Log[3/2]/2)/8"#,
    );
  }
  #[test]
  fn hypergeometric_u_3_2_1_float() {
    // HypergeometricU[3, 2, 1.] should match wolframscript at full
    // machine precision (~16 sig figs). Previously the float fast-path
    // used Richardson extrapolation (only ~12 sig figs); the symbolic
    // recurrence U[a,2,z] is exact and stays accurate at z=1.0.
    assert_case(r#"HypergeometricU[3, 2, 1.]"#, r#"0.10547895651520889"#);
  }
  #[test]
  fn hypergeometric_u_a_aplus1_evaluates() {
    // U[a, a+1, z] = z^(-a) must actually evaluate, not return
    // unevaluated z^(-a). wolframscript: 1/25 (int) and 0.04 (real).
    assert_case(r#"HypergeometricU[2, 3, 5]"#, r#"1/25"#);
    assert_case(r#"HypergeometricU[2, 3, 5.]"#, r#"0.04"#);
  }
  #[test]
  fn gamma_overflow_huge_real() {
    // Gamma[1.*^20] overflows f64. Wolfram returns Overflow[] with
    // General::ovfl message. mathics doctest at gamma.py:381 / 383.
    assert_case(r#"Quiet[Gamma[1.*^20]]"#, r#"Overflow[]"#);
  }
  #[test]
  fn log_of_overflow() {
    // Per wolframscript: Log[Overflow[]] = Overflow[].
    assert_case(r#"Log[Overflow[]]"#, r#"Overflow[]"#);
  }
  #[test]
  fn log_gamma_overflow_huge_real() {
    // Log[Gamma[1.*^20]] should propagate Overflow[].
    assert_case(r#"Quiet[Log[Gamma[1.*^20]]]"#, r#"Overflow[]"#);
  }
}

mod mathieu_s {
  use super::*;

  fn parse_real(s: &str) -> f64 {
    s.trim().parse().unwrap()
  }

  /// Woxi's normalisation pins `y(0) = 0`, `y'(0) = √a` so the q → 0
  /// limit reproduces sin(√a · z). Wolfram's normalisation introduces
  /// an additional (a, q)-dependent factor for q ≠ 0; tests compare
  /// shape ratios where Wolfram and Woxi must agree.

  #[test]
  fn mathieu_s_q_zero_matches_sine() {
    let val = parse_real(&interpret("MathieuS[2, 0, 1.5]").unwrap());
    let expected = (2.0_f64.sqrt() * 1.5).sin();
    assert!(
      (val - expected).abs() < 1e-9,
      "MathieuS(2, 0, 1.5): got {val}, expected {expected}"
    );
  }

  #[test]
  fn mathieu_s_prime_q_zero_at_origin() {
    let val = parse_real(&interpret("MathieuSPrime[3, 0, 0]").unwrap());
    assert!(
      (val - 3.0_f64.sqrt()).abs() < 1e-9,
      "MathieuSPrime(3, 0, 0): got {val}, expected √3"
    );
  }

  #[test]
  fn mathieu_s_prime_q_zero_matches_derivative() {
    let val = parse_real(&interpret("MathieuSPrime[2, 0, 1.5]").unwrap());
    let sa = 2.0_f64.sqrt();
    let expected = sa * (sa * 1.5).cos();
    assert!(
      (val - expected).abs() < 1e-9,
      "MathieuSPrime(2, 0, 1.5): got {val}, expected {expected}"
    );
  }

  #[test]
  fn mathieu_s_prime_audit_case_is_numeric() {
    // Audit case `MathieuSPrime[2, 1, 3.2]`. Previously unevaluated.
    // Now returns a finite Real (with Woxi's q → 0 normalisation —
    // not the same scalar Wolfram returns, but the call is no longer
    // symbolic). The normalisation-invariant shape ratio
    // `MathieuSPrime(a, q, z) / MathieuSPrime(a, q, 0)` matches.
    let val = parse_real(&interpret("MathieuSPrime[2, 1, 3.2]").unwrap());
    assert!(val.is_finite(), "expected a finite real, got {val}");
    let val0 = parse_real(&interpret("MathieuSPrime[2, 1, 0]").unwrap());
    let ratio = val / val0;
    let wolfram_ratio = -0.41459068965368717 / 0.5336161514292099;
    assert!(
      (ratio - wolfram_ratio).abs() < 1e-3,
      "shape ratio mismatch: got {ratio}, expected {wolfram_ratio}"
    );
  }

  #[test]
  fn mathieu_s_zero_at_origin() {
    let val = parse_real(&interpret("MathieuS[2, 1, 0]").unwrap());
    assert!(
      val.abs() < 1e-12,
      "MathieuS(2, 1, 0) should be 0, got {val}"
    );
  }

  #[test]
  fn mathieu_s_symbolic_passthrough() {
    assert_eq!(interpret("MathieuS[a, q, z]").unwrap(), "MathieuS[a, q, z]");
    assert_eq!(
      interpret("MathieuSPrime[a, q, z]").unwrap(),
      "MathieuSPrime[a, q, z]"
    );
  }
}

mod riemann_siegel_z {
  use super::*;

  fn val(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  #[test]
  fn numeric_small() {
    // wolframscript: RiemannSiegelZ[2.0] = -0.5396331256461437
    assert!((val("RiemannSiegelZ[2.0]") - (-0.5396331256461437)).abs() < 1e-10);
  }

  #[test]
  fn at_zero() {
    // wolframscript: RiemannSiegelZ[0.0] = -1.460354508809588
    assert!((val("RiemannSiegelZ[0.0]") - (-1.460354508809588)).abs() < 1e-10);
  }

  #[test]
  fn even_function() {
    // Z is even: Z[-t] = Z[t]
    let a = val("RiemannSiegelZ[3.5]");
    let b = val("RiemannSiegelZ[-3.5]");
    assert!((a - b).abs() < 1e-12);
    // wolframscript: RiemannSiegelZ[3.5] = -0.5688748138132947
    assert!((a - (-0.5688748138132947)).abs() < 1e-10);
  }

  #[test]
  fn larger_argument() {
    // wolframscript: RiemannSiegelZ[50.0] = -0.3407350059550225
    assert!((val("RiemannSiegelZ[50.0]") - (-0.3407350059550225)).abs() < 1e-9);
    // wolframscript: RiemannSiegelZ[100.0] = 2.6926970566644393
    assert!((val("RiemannSiegelZ[100.0]") - 2.6926970566644393).abs() < 1e-8);
  }

  #[test]
  fn near_first_zero() {
    // First nontrivial zero near t = 14.134725; |Z| is tiny there.
    // Output uses Wolfram scientific notation ("*^-7"); normalize to "e".
    let s = interpret("RiemannSiegelZ[14.134725]").unwrap();
    let v: f64 = s.replace("*^", "e").parse().unwrap();
    assert!(v.abs() < 1e-6);
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("RiemannSiegelZ[t]").unwrap(), "RiemannSiegelZ[t]");
    // Exact integer argument stays unevaluated (matches wolframscript).
    assert_eq!(interpret("RiemannSiegelZ[2]").unwrap(), "RiemannSiegelZ[2]");
  }
}

mod riemann_siegel_theta {
  use super::*;

  fn val(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  #[test]
  fn numeric_small() {
    // wolframscript: RiemannSiegelTheta[1.0] = -1.7675479528122908
    assert!(
      (val("RiemannSiegelTheta[1.0]") - (-1.7675479528122908)).abs() < 1e-10
    );
    // wolframscript: RiemannSiegelTheta[10.0] = -3.067074396289898
    assert!(
      (val("RiemannSiegelTheta[10.0]") - (-3.067074396289898)).abs() < 1e-10
    );
  }

  #[test]
  fn at_zero() {
    // wolframscript: RiemannSiegelTheta[0.0] = 0.
    assert_eq!(interpret("RiemannSiegelTheta[0.0]").unwrap(), "0.");
  }

  #[test]
  fn negative_argument() {
    // theta is odd: theta(-t) = -theta(t).
    let a = val("RiemannSiegelTheta[5.0]");
    let b = val("RiemannSiegelTheta[-5.0]");
    assert!((a + b).abs() < 1e-10);
    // wolframscript: RiemannSiegelTheta[-5.0] = 3.4596203753634622
    assert!((b - 3.4596203753634622).abs() < 1e-10);
  }

  #[test]
  fn larger_argument() {
    // wolframscript: RiemannSiegelTheta[100.0] = 87.97216523178722
    assert!(
      (val("RiemannSiegelTheta[100.0]") - 87.97216523178722).abs() < 1e-8
    );
  }

  #[test]
  fn n_of_exact() {
    // N forces a Real argument, so the numeric branch fires.
    // wolframscript: N[RiemannSiegelTheta[1]] = -1.7675479528122908
    assert!(
      (val("N[RiemannSiegelTheta[1]]") - (-1.7675479528122908)).abs() < 1e-10
    );
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(
      interpret("RiemannSiegelTheta[t]").unwrap(),
      "RiemannSiegelTheta[t]"
    );
    // Exact integer argument stays unevaluated (matches wolframscript).
    assert_eq!(
      interpret("RiemannSiegelTheta[2]").unwrap(),
      "RiemannSiegelTheta[2]"
    );
  }
}

mod ramanujan_tau_theta {
  use super::*;

  fn val(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  // RamanujanTauTheta[t] = Im(logGamma(6 + i t)) - t log(2 Pi).
  #[test]
  fn numeric_values() {
    assert!(
      (val("RamanujanTauTheta[1.0]") - (-0.12634683710523698)).abs() < 1e-9
    );
    assert!(
      (val("RamanujanTauTheta[2.0]") - (-0.22140434565662526)).abs() < 1e-9
    );
    assert!((val("RamanujanTauTheta[9.22]") - 1.4037204336632279).abs() < 1e-8);
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("RamanujanTauTheta[0.0]").unwrap(), "0.");
  }

  // theta is an odd function: theta(-t) = -theta(t).
  #[test]
  fn odd_function() {
    let a = val("RamanujanTauTheta[2.0]");
    let b = val("RamanujanTauTheta[-2.0]");
    assert!((a + b).abs() < 1e-9);
  }

  // N forces a Real argument, so the numeric branch fires.
  #[test]
  fn n_of_exact() {
    assert!(
      (val("N[RamanujanTauTheta[5]]") - (-0.09796566631115056)).abs() < 1e-9
    );
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(
      interpret("RamanujanTauTheta[t]").unwrap(),
      "RamanujanTauTheta[t]"
    );
    assert_eq!(
      interpret("RamanujanTauTheta[2]").unwrap(),
      "RamanujanTauTheta[2]"
    );
  }
}

mod ramanujan_tau_l {
  use super::*;

  fn val(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  // L(s, Delta) via the smoothed approximate functional equation.
  #[test]
  fn real_argument() {
    // s = 2 (reflected to L(10) via the functional equation).
    assert!((val("RamanujanTauL[2.0]") - 0.14637454209126602).abs() < 1e-9);
    // s = 10, deep in the convergent Dirichlet region.
    assert!((val("RamanujanTauL[10.0]") - 0.9798090882512205).abs() < 1e-9);
    // s = 3.5, between the strip and the convergent region.
    assert!((val("RamanujanTauL[3.5]") - 0.4090683518030643).abs() < 1e-9);
  }

  // N forces the numeric branch on an exact integer argument.
  #[test]
  fn n_of_exact() {
    assert!((val("N[RamanujanTauL[5]]") - 0.6667091884340036).abs() < 1e-9);
  }

  // On the critical line s = 6 + i t the L-function is complex.
  #[test]
  fn critical_line_complex() {
    let re = val("Re[N[RamanujanTauL[6 + 2 I]]]");
    let im = val("Im[N[RamanujanTauL[6 + 2 I]]]");
    assert!((re - 0.8790164686371028).abs() < 1e-8, "re {re}");
    assert!((im - 0.19786173432815862).abs() < 1e-8, "im {im}");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("RamanujanTauL[s]").unwrap(), "RamanujanTauL[s]");
    assert_eq!(interpret("RamanujanTauL[2]").unwrap(), "RamanujanTauL[2]");
  }
}

mod ramanujan_tau_z {
  use super::*;

  fn val(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  // Z(t) = e^{i theta(t)} L(6 + i t) is real on the critical line.
  #[test]
  fn numeric_values() {
    assert!((val("RamanujanTauZ[1.0]") - 0.8194349386751951).abs() < 1e-8);
    assert!((val("RamanujanTauZ[2.0]") - 0.9010101098470454).abs() < 1e-8);
    assert!((val("RamanujanTauZ[10.0]") - (-0.843434165151841)).abs() < 1e-6);
  }

  // Z(t) and L(6+it) have the same modulus (rotation by e^{i theta}).
  #[test]
  fn modulus_matches_l() {
    let z = val("RamanujanTauZ[3.0]").abs();
    let l = val("Abs[N[RamanujanTauL[6 + 3 I]]]");
    assert!((z - l).abs() < 1e-7, "Z {z} vs |L| {l}");
  }

  #[test]
  fn symbolic_passthrough() {
    assert_eq!(interpret("RamanujanTauZ[t]").unwrap(), "RamanujanTauZ[t]");
    assert_eq!(interpret("RamanujanTauZ[2]").unwrap(), "RamanujanTauZ[2]");
  }
}

mod real_exponent {
  use super::*;

  fn val(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  #[test]
  fn integer_base_two() {
    // RealExponent[x, b] = Log[b, Abs[x]] as a machine real.
    assert_eq!(interpret("RealExponent[8, 2]").unwrap(), "3.");
    assert_eq!(interpret("RealExponent[1/8, 2]").unwrap(), "-3.");
    assert_eq!(interpret("RealExponent[81, 3]").unwrap(), "4.");
    // wolframscript: RealExponent[12, 2] = 3.584962500721156
    assert!((val("RealExponent[12, 2]") - 3.584962500721156).abs() < 1e-12);
  }

  #[test]
  fn default_base_ten() {
    // RealExponent[x] defaults to base 10.
    assert_eq!(interpret("RealExponent[100]").unwrap(), "2.");
    assert_eq!(interpret("RealExponent[1000]").unwrap(), "3.");
    // wolframscript: RealExponent[8] = 0.9030899869919436
    assert!((val("RealExponent[8]") - 0.9030899869919436).abs() < 1e-12);
  }

  #[test]
  fn negative_uses_magnitude() {
    // RealExponent uses Abs, so negative inputs give the same result.
    assert!((val("RealExponent[-8]") - val("RealExponent[8]")).abs() < 1e-15);
  }

  #[test]
  fn complex_uses_modulus() {
    // RealExponent[3 + 4 I] = Log[10, Abs[3+4I]] = Log[10, 5].
    // wolframscript: 0.6989700043360187
    assert!((val("RealExponent[3 + 4 I]") - 0.6989700043360187).abs() < 1e-12);
    // RealExponent[3 + 4 I, 5] = Log[5, 5] = 1.
    assert_eq!(interpret("RealExponent[3 + 4 I, 5]").unwrap(), "1.");
  }

  #[test]
  fn real_base() {
    // wolframscript: RealExponent[100, 2.5] = 5.025883189464119
    assert_eq!(
      interpret("RealExponent[100, 2.5]").unwrap(),
      "5.025883189464119"
    );
  }

  #[test]
  fn zero_is_negative_infinity() {
    assert_eq!(interpret("RealExponent[0]").unwrap(), "-Infinity");
    assert_eq!(interpret("RealExponent[0, 2]").unwrap(), "-Infinity");
  }

  #[test]
  fn symbolic_passthrough() {
    // Symbolic argument stays unevaluated.
    assert_eq!(interpret("RealExponent[x]").unwrap(), "RealExponent[x]");
  }

  #[test]
  fn invalid_base_passthrough() {
    // Base must be a real number greater than 1; otherwise unevaluated
    // (matches wolframscript, which additionally prints a message).
    assert_eq!(
      interpret("RealExponent[8, 1/2]").unwrap(),
      "RealExponent[8, 1/2]"
    );
    assert_eq!(
      interpret("RealExponent[8, 1]").unwrap(),
      "RealExponent[8, 1]"
    );
  }
}

mod log_barnes_g {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn exact_integers() {
    assert_case(r#"LogBarnesG[5]"#, r#"Log[12]"#);
    assert_case(r#"LogBarnesG[10]"#, r#"Log[5056584744960000]"#);
    // G(1) = G(2) = G(3) = 1, so the log collapses to 0
    assert_case(r#"LogBarnesG[1]"#, r#"0"#);
    assert_case(r#"LogBarnesG[3]"#, r#"0"#);
  }

  #[test]
  fn poles_at_non_positive_integers() {
    assert_case(r#"LogBarnesG[0]"#, r#"-Infinity"#);
    assert_case(r#"LogBarnesG[-1]"#, r#"-Infinity"#);
    assert_case(r#"LogBarnesG[0.]"#, r#"-Infinity"#);
    assert_case(r#"LogBarnesG[-1.]"#, r#"-Infinity"#);
  }

  #[test]
  fn unevaluated_forms() {
    // Exact non-integers and symbols stay symbolic (wolframscript
    // keeps LogBarnesG[1/2] unevaluated too)
    assert_case(r#"LogBarnesG[1/2]"#, r#"LogBarnesG[1/2]"#);
    assert_case(r#"LogBarnesG[x]"#, r#"LogBarnesG[x]"#);
  }

  #[test]
  fn machine_real_evaluates() {
    // The real path matches wolframscript to ~12 digits (see the
    // implementation note); assert through a Round projection that is
    // robust to the final-digit drift
    assert_case(r#"Round[LogBarnesG[5.] * 10^10]"#, r#"24849066498"#);
    assert_case(r#"Round[LogBarnesG[15.5] * 10^10]"#, r#"1363686988426"#);
  }
}

mod fourier_dst {
  use super::*;

  #[test]
  fn type_1() {
    // wolframscript: FourierDST[{1, 2, 3, 4}, 1]
    assert_eq!(
      interpret("Round[10^10 FourierDST[{1, 2, 3, 4}, 1]]").unwrap(),
      "{48662449473, -21762508995, 11487646027, -5137431484}"
    );
  }

  #[test]
  fn type_2_is_default() {
    // wolframscript: FourierDST[{1, 2, 3, 4}]
    assert_eq!(
      interpret("Round[10^10 FourierDST[{1, 2, 3, 4}]]").unwrap(),
      "{32664074122, -14142135624, 13529902504, -10000000000}"
    );
    assert_eq!(
      interpret("FourierDST[{1, 2, 3, 4}, 2]").unwrap(),
      interpret("FourierDST[{1, 2, 3, 4}]").unwrap()
    );
  }

  #[test]
  fn type_3() {
    assert_eq!(
      interpret("Round[10^10 FourierDST[{1, 2, 3, 4}, 3]]").unwrap(),
      "{65685355923, -8099572022, 3616156730, -2598915325}"
    );
  }

  #[test]
  fn type_4() {
    assert_eq!(
      interpret("Round[10^10 FourierDST[{1, 2, 3, 4}, 4]]").unwrap(),
      "{54615377423, -1580148114, 3546673293, 1443879993}"
    );
  }

  #[test]
  fn short_and_exact_input() {
    // Exact input numericizes; a singleton is its own transform
    assert_eq!(interpret("FourierDST[{5.}]").unwrap(), "{5.}");
    assert_eq!(
      interpret("Round[10^10 FourierDST[{1, 2, 3}]]").unwrap(),
      "{23094010768, -10000000000, 11547005384}"
    );
  }

  #[test]
  fn invalid_input() {
    // Empty or non-numeric lists emit FourierDST::fftl
    assert_eq!(interpret("FourierDST[{}]").unwrap(), "FourierDST[{}]");
    assert_eq!(
      interpret("FourierDST[{1, x}]").unwrap(),
      "FourierDST[{1, x}]"
    );
  }
}

mod marcum_q {
  use super::*;

  #[test]
  fn symbolic_rules() {
    // Exact arguments stay symbolic
    assert_eq!(interpret("MarcumQ[1, 2, 3]").unwrap(), "MarcumQ[1, 2, 3]");
    assert_eq!(
      interpret("MarcumQ[3/2, Sqrt[2], 0, Sqrt[x]]").unwrap(),
      "MarcumQ[3/2, Sqrt[2], 0, Sqrt[x]]"
    );
    // a = 0 reduces to GammaRegularized
    assert_eq!(
      interpret("MarcumQ[m, 0, b]").unwrap(),
      "GammaRegularized[m, b^2/2]"
    );
    assert_eq!(interpret("MarcumQ[1, 0, 3]").unwrap(), "E^(-9/2)");
    // b = 0 needs a numeric order: 1 when positive, pole at negative
    // integers, the closed form at m = 0, otherwise untouched
    assert_eq!(interpret("MarcumQ[1, a, 0]").unwrap(), "1");
    assert_eq!(interpret("MarcumQ[1/2, a, 0]").unwrap(), "1");
    assert_eq!(interpret("MarcumQ[-1, a, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("MarcumQ[0, a, 0]").unwrap(), "1 - E^(-1/2*a^2)");
    assert_eq!(interpret("MarcumQ[m, a, 0]").unwrap(), "MarcumQ[m, a, 0]");
  }

  #[test]
  fn numeric_evaluation() {
    // wolframscript agrees to ~15 digits; assert via Round projections
    assert_eq!(
      interpret("Round[10^10 MarcumQ[1., 2., 3.]]").unwrap(),
      "2143620882"
    );
    assert_eq!(
      interpret("Round[10^10 MarcumQ[2., 1., 3.]]").unwrap(),
      "1236287686"
    );
    assert_eq!(
      interpret("Round[10^10 MarcumQ[1, 2., 0.5]]").unwrap(),
      "9820693673"
    );
  }
}

mod gamma_regularized_exact_and_cf {
  use super::*;

  #[test]
  fn integer_first_argument_evaluates_exactly() {
    // Regression: stayed unevaluated before
    assert_eq!(interpret("GammaRegularized[1, 9/2]").unwrap(), "E^(-9/2)");
    assert_eq!(interpret("GammaRegularized[2, 3]").unwrap(), "4/E^3");
    assert_eq!(
      interpret("GammaRegularized[3, 1/2]").unwrap(),
      "13/(8*Sqrt[E])"
    );
    // Symbolic z stays put
    assert_eq!(
      interpret("GammaRegularized[4, x]").unwrap(),
      "GammaRegularized[4, x]"
    );
  }

  #[test]
  fn continued_fraction_branch_regression() {
    // Regression: the z >= a + 1 branch returned 0.0643 instead of
    // e^(-4.5)*5.5 = 0.0611 (broken Lentz iteration)
    assert_eq!(
      interpret("Round[10^10 GammaRegularized[2., 4.5]]").unwrap(),
      "610994810"
    );
  }
}
