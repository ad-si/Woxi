use super::*;

mod integrate_with_sum {
  use super::*;

  #[test]
  fn integrate_polynomial() {
    assert_eq!(interpret("Integrate[x^2, x]").unwrap(), "x^3/3");
  }

  #[test]
  fn integrate_sin() {
    assert_eq!(interpret("Integrate[Sin[x], x]").unwrap(), "-Cos[x]");
  }

  #[test]
  fn integrate_sum_of_terms() {
    // The ordering may differ from Mathematica but the result is correct
    let result = interpret("Integrate[x^2 + Sin[x], x]").unwrap();
    // Accept either ordering
    assert!(
      result == "x^3/3 - Cos[x]" || result == "-Cos[x] + x^3/3",
      "Got: {}",
      result
    );
  }

  #[test]
  fn integrate_cos() {
    assert_eq!(interpret("Integrate[Cos[x], x]").unwrap(), "Sin[x]");
  }

  #[test]
  fn integrate_sin_linear_arg() {
    // ∫ sin(2x) dx = -1/2*cos(2x)
    assert_eq!(
      interpret("Integrate[Sin[2*x], x]").unwrap(),
      "-1/2*Cos[2*x]"
    );
  }

  #[test]
  fn integrate_cos_linear_arg() {
    // ∫ cos(3x) dx = sin(3x)/3
    assert_eq!(interpret("Integrate[Cos[3*x], x]").unwrap(), "Sin[3*x]/3");
  }

  #[test]
  fn integrate_sin_squared() {
    // ∫ sin²(x) dx = x/2 - sin(2x)/4
    assert_eq!(
      interpret("Integrate[Sin[x]^2, x]").unwrap(),
      "x/2 - Sin[2*x]/4"
    );
  }

  #[test]
  fn integrate_cos_squared() {
    // ∫ cos²(x) dx = x/2 + sin(2x)/4
    assert_eq!(
      interpret("Integrate[Cos[x]^2, x]").unwrap(),
      "x/2 + Sin[2*x]/4"
    );
  }

  #[test]
  fn integrate_sin_squared_definite() {
    // ∫_0^Pi sin²(x) dx = Pi/2
    assert_eq!(
      interpret("Integrate[Sin[x]^2, {x, 0, Pi}]").unwrap(),
      "Pi/2"
    );
  }
}

mod definite_integrals {
  use super::*;

  #[test]
  fn gaussian_integral_full() {
    // ∫_{-∞}^{∞} E^(-x^2) dx = Sqrt[Pi]
    assert_eq!(
      interpret("Integrate[E^(-x^2), {x, -Infinity, Infinity}]").unwrap(),
      "Sqrt[Pi]"
    );
  }

  #[test]
  fn gaussian_integral_with_coefficient() {
    // ∫_{-∞}^{∞} E^(-2x^2) dx = Sqrt[Pi/2]
    assert_eq!(
      interpret("Integrate[E^(-2*x^2), {x, -Infinity, Infinity}]").unwrap(),
      "Sqrt[Pi/2]"
    );
  }

  #[test]
  fn half_gaussian_integral() {
    // ∫_0^{∞} E^(-x^2) dx = Sqrt[Pi]/2
    assert_eq!(
      interpret("Integrate[E^(-x^2), {x, 0, Infinity}]").unwrap(),
      "Sqrt[Pi]/2"
    );
  }

  #[test]
  fn definite_integral_polynomial() {
    // ∫_0^1 x^2 dx = 1/3
    assert_eq!(interpret("Integrate[x^2, {x, 0, 1}]").unwrap(), "1/3");
  }

  #[test]
  fn definite_integral_constant() {
    // ∫_0^3 5 dx = 15
    assert_eq!(interpret("Integrate[5, {x, 0, 3}]").unwrap(), "15");
  }

  #[test]
  fn definite_integral_reciprocal_square() {
    // ∫_1^2 1/x^2 dx = 1/2
    assert_eq!(interpret("Integrate[1/x^2, {x, 1, 2}]").unwrap(), "1/2");
  }

  #[test]
  fn definite_integral_reciprocal_square_plus_one() {
    // ∫_1^2 (1/x^2 + 1) dx = 3/2
    assert_eq!(interpret("Integrate[1/x^2 + 1, {x, 1, 2}]").unwrap(), "3/2");
  }

  #[test]
  fn definite_integral_user_defined_function() {
    // f[x_] := 1/x^2 + 1; ∫_1^2 f[x] dx = 3/2
    assert_eq!(
      interpret("f[x_] := 1/x^2 + 1; Integrate[f[x], {x, 1, 2}]").unwrap(),
      "3/2"
    );
  }

  #[test]
  fn definite_integral_reciprocal_cube() {
    // ∫_1^2 1/x^3 dx = 3/8
    assert_eq!(interpret("Integrate[1/x^3, {x, 1, 2}]").unwrap(), "3/8");
  }
}

mod integrate_reciprocal_powers {
  use super::*;

  #[test]
  fn integrate_one_over_x_squared() {
    // ∫ 1/x^2 dx = -x^(-1)
    assert_eq!(interpret("Integrate[1/x^2, x]").unwrap(), "-x^(-1)");
  }

  #[test]
  fn integrate_one_over_x_cubed() {
    // ∫ 1/x^3 dx = -x^(-2)/2
    assert_eq!(interpret("Integrate[1/x^3, x]").unwrap(), "-1/2*1/x^2");
  }

  #[test]
  fn integrate_one_over_x_fourth() {
    // ∫ 1/x^4 dx = -x^(-3)/3
    assert_eq!(interpret("Integrate[1/x^4, x]").unwrap(), "-1/3*1/x^3");
  }

  #[test]
  fn integrate_const_over_x_squared() {
    // ∫ 3/x^2 dx = -3*x^(-1) = -3/x
    assert_eq!(interpret("Integrate[3/x^2, x]").unwrap(), "-3/x");
  }

  #[test]
  fn integrate_reciprocal_plus_polynomial() {
    // ∫ (1/x^2 + 1) dx = -x^(-1) + x
    let result = interpret("Integrate[1/x^2 + 1, x]").unwrap();
    assert!(
      result == "-x^(-1) + x" || result == "x - x^(-1)",
      "Got: {}",
      result
    );
  }
}

mod differentiate_plus_times {
  use super::*;

  #[test]
  fn d_x_plus_one() {
    assert_eq!(interpret("D[x + 1, x]").unwrap(), "1");
  }

  #[test]
  fn d_x_squared_plus_x() {
    assert_eq!(interpret("D[x^2 + x, x]").unwrap(), "1 + 2*x");
  }

  #[test]
  fn d_log_one_plus_t() {
    assert_eq!(interpret("D[Log[1 + t], t]").unwrap(), "(1 + t)^(-1)");
  }

  #[test]
  fn d_times_three_factors() {
    // D[4*(3 + 2*x)*x, x] should work with 3-factor Times
    assert_eq!(
      interpret("D[4*(3 + 2*x)*x, x]").unwrap(),
      "8*x + 4*(3 + 2*x)"
    );
  }
}

mod derivative_prime_notation {
  use super::*;

  #[test]
  fn derivative_simple_polynomial() {
    assert_eq!(interpret("f[x_] := x^2; f'[x]").unwrap(), "2*x");
  }

  #[test]
  fn derivative_second_order() {
    assert_eq!(interpret("f[x_] := x^3; f''[x]").unwrap(), "6*x");
  }

  #[test]
  fn derivative_third_order() {
    assert_eq!(interpret("f[x_] := x^3; f'''[x]").unwrap(), "6");
  }

  #[test]
  fn derivative_fourth_order_vanishes() {
    assert_eq!(interpret("f[x_] := x^3; f''''[x]").unwrap(), "0");
  }

  #[test]
  fn derivative_sin() {
    assert_eq!(interpret("g[x_] := Sin[x]; g'[x]").unwrap(), "Cos[x]");
  }

  #[test]
  fn derivative_sin_at_zero() {
    assert_eq!(interpret("g[x_] := Sin[x]; g'[0]").unwrap(), "1");
  }

  #[test]
  fn derivative_cos_second() {
    assert_eq!(interpret("h[x_] := Cos[x]; h''[x]").unwrap(), "-Cos[x]");
  }

  #[test]
  fn standalone_derivative_symbolic() {
    // f' without brackets returns Derivative[1][f]
    assert_eq!(interpret("f'").unwrap(), "Derivative[1][f]");
  }

  #[test]
  fn standalone_derivative_double() {
    assert_eq!(interpret("f''").unwrap(), "Derivative[2][f]");
  }

  #[test]
  fn derivative_in_list() {
    // {f'[x], f''[x]} with f defined
    assert_eq!(
      interpret("f[x_] := x^2; {f'[x], f''[x]}").unwrap(),
      "{2*x, 2}"
    );
  }

  #[test]
  fn derivative_undefined_function() {
    // Derivative of an undefined function stays symbolic
    assert_eq!(interpret("h'[x]").unwrap(), "Derivative[1][h][x]");
  }
}

mod series {
  use super::*;

  #[test]
  fn series_exp() {
    assert_eq!(
      interpret("Series[Exp[x], {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1, 1, 1/2, 1/6}, 0, 4, 1]"
    );
  }

  #[test]
  fn series_exp_order4() {
    assert_eq!(
      interpret("Series[Exp[x], {x, 0, 4}]").unwrap(),
      "SeriesData[x, 0, {1, 1, 1/2, 1/6, 1/24}, 0, 5, 1]"
    );
  }

  #[test]
  fn series_sin_strips_leading_zero() {
    assert_eq!(
      interpret("Series[Sin[x], {x, 0, 5}]").unwrap(),
      "SeriesData[x, 0, {1, 0, -1/6, 0, 1/120}, 1, 6, 1]"
    );
  }

  #[test]
  fn series_cos_no_leading_zero() {
    assert_eq!(
      interpret("Series[Cos[x], {x, 0, 6}]").unwrap(),
      "SeriesData[x, 0, {1, 0, -1/2, 0, 1/24, 0, -1/720}, 0, 7, 1]"
    );
  }

  #[test]
  fn series_log_around_1() {
    assert_eq!(
      interpret("Series[Log[x], {x, 1, 3}]").unwrap(),
      "SeriesData[x, 1, {1, -1/2, 1/3}, 1, 4, 1]"
    );
  }

  #[test]
  fn series_geometric() {
    assert_eq!(
      interpret("Series[1/(1 - x), {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1, 1, 1, 1}, 0, 4, 1]"
    );
  }

  #[test]
  fn series_zero_returns_zero() {
    assert_eq!(interpret("Series[0, {x, 0, 3}]").unwrap(), "0");
  }

  #[test]
  fn series_tan_order5() {
    assert_eq!(
      interpret("Series[Tan[x], {x, 0, 5}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1/3, 0, 2/15}, 1, 6, 1]"
    );
  }

  #[test]
  fn series_tan_order15() {
    assert_eq!(
      interpret("Series[Tan[x], {x, 0, 15}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1/3, 0, 2/15, 0, 17/315, 0, 62/2835, 0, 1382/155925, 0, 21844/6081075, 0, 929569/638512875}, 1, 16, 1]"
    );
  }

  #[test]
  fn series_sec_order6() {
    assert_eq!(
      interpret("Series[Sec[x], {x, 0, 6}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1/2, 0, 5/24, 0, 61/720}, 0, 7, 1]"
    );
  }

  #[test]
  fn series_cot_order6() {
    assert_eq!(
      interpret("Series[Cot[x], {x, 0, 6}]").unwrap(),
      "SeriesData[x, 0, {1, 0, -1/3, 0, -1/45, 0, -2/945}, -1, 7, 1]"
    );
  }

  #[test]
  fn series_csc_order6() {
    assert_eq!(
      interpret("Series[Csc[x], {x, 0, 6}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1/6, 0, 7/360, 0, 31/15120}, -1, 7, 1]"
    );
  }

  #[test]
  fn series_exp_neg_x_sin_2x() {
    assert_eq!(
      interpret("Series[Exp[-x] Sin[2x], {x, 0, 6}]").unwrap(),
      "SeriesData[x, 0, {2, -2, -1/3, 1, -19/60, -11/180}, 1, 7, 1]"
    );
  }

  #[test]
  fn series_log_1_plus_x() {
    assert_eq!(
      interpret("Series[Log[1 + x], {x, 0, 5}]").unwrap(),
      "SeriesData[x, 0, {1, -1/2, 1/3, -1/4, 1/5}, 1, 6, 1]"
    );
  }

  #[test]
  fn series_sin_around_pi() {
    assert_eq!(
      interpret("Series[Sin[x], {x, Pi, 5}]").unwrap(),
      "SeriesData[x, Pi, {-1, 0, 1/6, 0, -1/120}, 1, 6, 1]"
    );
  }

  #[test]
  fn normal_series_sin() {
    assert_eq!(
      interpret("Normal[Series[Sin[x], {x, 0, 7}]]").unwrap(),
      "x - x^3/6 + x^5/120 - x^7/5040"
    );
  }

  #[test]
  fn normal_series_exp() {
    assert_eq!(
      interpret("Normal[Series[Exp[x], {x, 0, 5}]]").unwrap(),
      "1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120"
    );
  }

  #[test]
  fn normal_series_cos() {
    assert_eq!(
      interpret("Normal[Series[Cos[x], {x, 0, 6}]]").unwrap(),
      "1 - x^2/2 + x^4/24 - x^6/720"
    );
  }

  #[test]
  fn normal_series_log() {
    assert_eq!(
      interpret("Normal[Series[Log[1 + x], {x, 0, 5}]]").unwrap(),
      "x - x^2/2 + x^3/3 - x^4/4 + x^5/5"
    );
  }

  #[test]
  fn normal_series_exp_neg_x_sin_2x() {
    assert_eq!(
      interpret("Normal[Series[Exp[-x] Sin[2x], {x, 0, 6}]]").unwrap(),
      "2*x - 2*x^2 - x^3/3 + x^4 - (19*x^5)/60 - (11*x^6)/180"
    );
  }

  #[test]
  fn normal_series_around_pi() {
    assert_eq!(
      interpret("Normal[Series[Sin[x], {x, Pi, 5}]]").unwrap(),
      "Pi - x + (-Pi + x)^3/6 - (-Pi + x)^5/120"
    );
  }

  #[test]
  fn normal_series_geometric() {
    assert_eq!(
      interpret("Normal[Series[1/(1 - x), {x, 0, 5}]]").unwrap(),
      "1 + x + x^2 + x^3 + x^4 + x^5"
    );
  }
}

mod limit {
  use super::*;

  #[test]
  fn limit_sin_x_over_x() {
    assert_eq!(interpret("Limit[Sin[x]/x, x -> 0]").unwrap(), "1");
  }

  #[test]
  fn limit_direct_substitution() {
    assert_eq!(interpret("Limit[x^2, x -> 3]").unwrap(), "9");
  }

  #[test]
  fn limit_compound_interest() {
    assert_eq!(interpret("Limit[(1 + 1/n)^n, n -> Infinity]").unwrap(), "E");
  }

  #[test]
  fn limit_compound_interest_general() {
    assert_eq!(
      interpret("Limit[(1 + 2/n)^n, n -> Infinity]").unwrap(),
      "E^2"
    );
  }

  #[test]
  fn limit_one_over_n() {
    assert_eq!(interpret("Limit[1/n, n -> Infinity]").unwrap(), "0");
  }

  #[test]
  fn limit_n_to_infinity() {
    assert_eq!(interpret("Limit[n, n -> Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn limit_one_over_x_at_zero_from_above() {
    assert_eq!(
      interpret(r#"Limit[1/x, x -> 0, Direction -> "FromAbove"]"#).unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn limit_one_over_x_at_zero_from_below() {
    assert_eq!(
      interpret(r#"Limit[1/x, x -> 0, Direction -> "FromBelow"]"#).unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn limit_one_over_x_at_zero_no_direction() {
    // Without direction, 1/x at 0 is indeterminate (different from left and right)
    assert_eq!(interpret("Limit[1/x, x -> 0]").unwrap(), "Indeterminate");
  }

  #[test]
  fn limit_one_over_x_squared_at_zero() {
    // 1/x^2 -> Infinity from both sides, so no direction needed
    assert_eq!(interpret("Limit[1/x^2, x -> 0]").unwrap(), "Infinity");
  }

  #[test]
  fn limit_one_over_x_squared_from_above() {
    assert_eq!(
      interpret(r#"Limit[1/x^2, x -> 0, Direction -> "FromAbove"]"#).unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn limit_sqrt_x_at_zero_from_above() {
    assert_eq!(
      interpret(r#"Limit[Sqrt[x], x -> 0, Direction -> "FromAbove"]"#).unwrap(),
      "0"
    );
  }

  #[test]
  fn limit_log_x_at_zero_from_above() {
    assert_eq!(
      interpret(r#"Limit[Log[x], x -> 0, Direction -> "FromAbove"]"#).unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn limit_exp_neg_one_over_x_from_above() {
    assert_eq!(
      interpret(r#"Limit[Exp[-1/x], x -> 0, Direction -> "FromAbove"]"#)
        .unwrap(),
      "0"
    );
  }

  #[test]
  fn limit_exp_neg_one_over_x_from_below() {
    assert_eq!(
      interpret(r#"Limit[Exp[-1/x], x -> 0, Direction -> "FromBelow"]"#)
        .unwrap(),
      "Infinity"
    );
  }
}

mod nintegrate {
  use super::*;

  fn assert_approx(code: &str, expected: f64, tol: f64) {
    let result = interpret(code).unwrap();
    let val: f64 = result.parse().unwrap_or_else(|_| {
      panic!("NIntegrate result should be a number, got: {}", result)
    });
    assert!(
      (val - expected).abs() < tol,
      "NIntegrate mismatch for {}: got {}, expected {} (diff {})",
      code,
      val,
      expected,
      (val - expected).abs()
    );
  }

  #[test]
  fn nintegrate_polynomial() {
    // ∫₀¹ x² dx = 1/3
    assert_approx("NIntegrate[x^2, {x, 0, 1}]", 1.0 / 3.0, 1e-10);
  }

  #[test]
  fn nintegrate_sin() {
    // ∫₀^π sin(x) dx = 2
    assert_approx("NIntegrate[Sin[x], {x, 0, Pi}]", 2.0, 1e-10);
  }

  #[test]
  fn nintegrate_exp_neg_x_squared() {
    // ∫₀¹ e^(-x²) dx ≈ 0.7468241328124271
    assert_approx(
      "NIntegrate[Exp[-x^2], {x, 0, 1}]",
      0.7468241328124271,
      1e-10,
    );
  }

  #[test]
  fn nintegrate_one_over_x() {
    // ∫₁^e 1/x dx = 1
    assert_approx("NIntegrate[1/x, {x, 1, E}]", 1.0, 1e-10);
  }

  #[test]
  fn nintegrate_oscillatory() {
    // ∫₀¹⁰ sin(x²) dx ≈ 0.5836708999296233
    assert_approx(
      "NIntegrate[Sin[x^2], {x, 0, 10}]",
      0.5836708999296233,
      1e-10,
    );
  }

  #[test]
  fn nintegrate_constant() {
    // ∫₀⁵ 3 dx = 15
    assert_approx("NIntegrate[3, {x, 0, 5}]", 15.0, 1e-10);
  }

  #[test]
  fn nintegrate_cos() {
    // ∫₀^(π/2) cos(x) dx = 1
    assert_approx("NIntegrate[Cos[x], {x, 0, Pi/2}]", 1.0, 1e-10);
  }

  #[test]
  fn nintegrate_error_no_range() {
    // NIntegrate requires {var, lo, hi}
    let result = interpret("NIntegrate[x^2, x]");
    assert!(result.is_err());
  }
}

mod trig_sec_csc_cot {
  use super::*;

  #[test]
  fn sec_zero() {
    assert_eq!(interpret("Sec[0]").unwrap(), "1");
  }

  #[test]
  fn sec_pi_third() {
    assert_eq!(interpret("Sec[Pi/3]").unwrap(), "2");
  }

  #[test]
  fn sec_pi_fourth() {
    assert_eq!(interpret("Sec[Pi/4]").unwrap(), "Sqrt[2]");
  }

  #[test]
  fn sec_pi_sixth() {
    assert_eq!(interpret("Sec[Pi/6]").unwrap(), "2/Sqrt[3]");
  }

  #[test]
  fn sec_pi_half() {
    assert_eq!(interpret("Sec[Pi/2]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn sec_pi() {
    assert_eq!(interpret("Sec[Pi]").unwrap(), "-1");
  }

  #[test]
  fn csc_pi_half() {
    assert_eq!(interpret("Csc[Pi/2]").unwrap(), "1");
  }

  #[test]
  fn csc_pi_sixth() {
    assert_eq!(interpret("Csc[Pi/6]").unwrap(), "2");
  }

  #[test]
  fn csc_pi_fourth() {
    assert_eq!(interpret("Csc[Pi/4]").unwrap(), "Sqrt[2]");
  }

  #[test]
  fn csc_pi_third() {
    assert_eq!(interpret("Csc[Pi/3]").unwrap(), "2/Sqrt[3]");
  }

  #[test]
  fn cot_pi_fourth() {
    assert_eq!(interpret("Cot[Pi/4]").unwrap(), "1");
  }

  #[test]
  fn cot_pi_third() {
    assert_eq!(interpret("Cot[Pi/3]").unwrap(), "1/Sqrt[3]");
  }

  #[test]
  fn cot_pi_sixth() {
    assert_eq!(interpret("Cot[Pi/6]").unwrap(), "Sqrt[3]");
  }

  #[test]
  fn cot_pi_half() {
    assert_eq!(interpret("Cot[Pi/2]").unwrap(), "0");
  }

  #[test]
  fn d_sec() {
    assert_eq!(interpret("D[Sec[x], x]").unwrap(), "Sec[x]*Tan[x]");
  }

  #[test]
  fn d_csc() {
    assert_eq!(interpret("D[Csc[x], x]").unwrap(), "-(Cot[x]*Csc[x])");
  }

  #[test]
  fn d_cot() {
    assert_eq!(interpret("D[Cot[x], x]").unwrap(), "-Csc[x]^2");
  }

  #[test]
  fn sec_negative_angle() {
    // Sec[-Pi/3] = Sec[Pi/3] = 2
    assert_eq!(interpret("Sec[-Pi/3]").unwrap(), "2");
  }

  #[test]
  fn csc_negative_angle() {
    // Csc[-Pi/6] = -Csc[Pi/6] = -2
    assert_eq!(interpret("Csc[-Pi/6]").unwrap(), "-2");
  }

  #[test]
  fn cot_zero() {
    assert_eq!(interpret("Cot[0]").unwrap(), "ComplexInfinity");
  }
}

mod erf {
  use super::*;

  #[test]
  fn erf_zero() {
    assert_eq!(interpret("Erf[0]").unwrap(), "0");
  }

  #[test]
  fn erf_symbolic() {
    assert_eq!(interpret("Erf[x]").unwrap(), "Erf[x]");
  }

  #[test]
  fn erf_negative_arg() {
    // Erf[-x] = -Erf[x] (odd function)
    assert_eq!(interpret("Erf[-x]").unwrap(), "-Erf[x]");
  }

  #[test]
  fn erfc_zero() {
    assert_eq!(interpret("Erfc[0]").unwrap(), "1");
  }

  #[test]
  fn erfc_symbolic() {
    assert_eq!(interpret("Erfc[x]").unwrap(), "Erfc[x]");
  }
}

mod integrate_gaussian {
  use super::*;

  #[test]
  fn integrate_exp_neg_x_squared() {
    // ∫ Exp[-x^2] dx = (Sqrt[Pi]*Erf[x])/2
    assert_eq!(
      interpret("Integrate[Exp[-x^2], x]").unwrap(),
      "(Sqrt[Pi]*Erf[x])/2"
    );
  }

  #[test]
  fn integrate_exp_neg_3_x_squared() {
    // ∫ Exp[-3*x^2] dx = (Sqrt[Pi/3]*Erf[Sqrt[3]*x])/2
    assert_eq!(
      interpret("Integrate[Exp[-3*x^2], x]").unwrap(),
      "(Sqrt[Pi/3]*Erf[Sqrt[3]*x])/2"
    );
  }

  #[test]
  fn integrate_exp_neg_a_x_squared() {
    // ∫ Exp[-a*x^2] dx = (Sqrt[Pi/a]*Erf[Sqrt[a]*x])/2
    assert_eq!(
      interpret("Integrate[Exp[-a*x^2], x]").unwrap(),
      "(Sqrt[Pi]*Erf[Sqrt[a]*x])/(2*Sqrt[a])"
    );
  }
}

mod big_o {
  use super::*;

  #[test]
  fn o_basic() {
    assert_eq!(interpret("O[x]").unwrap(), "SeriesData[x, 0, {}, 1, 1, 1]");
  }

  #[test]
  fn o_with_center() {
    assert_eq!(
      interpret("O[x, 1]").unwrap(),
      "SeriesData[x, 1, {}, 1, 1, 1]"
    );
  }

  // D threading over lists
  #[test]
  fn d_list_simple() {
    assert_eq!(interpret("D[{x^2, x^3}, x]").unwrap(), "{2*x, 3*x^2}");
  }

  #[test]
  fn d_list_trig() {
    assert_eq!(
      interpret("D[{Cos[x] + Sin[x], Sin[x]}, x]").unwrap(),
      "{Cos[x] - Sin[x], Cos[x]}"
    );
  }

  #[test]
  fn d_list_single_element() {
    assert_eq!(interpret("D[{x^2}, x]").unwrap(), "{2*x}");
  }

  #[test]
  fn d_list_higher_order() {
    assert_eq!(interpret("D[{x^3, x^4}, {x, 2}]").unwrap(), "{6*x, 12*x^2}");
  }

  #[test]
  fn d_list_nested() {
    // D should thread over the outer list
    assert_eq!(interpret("D[{x, x^2, x^3}, x]").unwrap(), "{1, 2*x, 3*x^2}");
  }
}

mod find_minimum {
  use super::*;

  #[test]
  fn quadratic_minimum() {
    clear_state();
    let result = interpret("FindMinimum[x^2 - 4 x + 5, {x, 0}]").unwrap();
    assert_eq!(result, "{1., {x -> 2.}}");
  }

  #[test]
  fn sin_minimum() {
    clear_state();
    let result = interpret("FindMinimum[Sin[x], {x, 5}]").unwrap();
    // Minimum of Sin near x=5 is at x = 3*Pi/2 ≈ 4.7124
    assert!(result.starts_with("{-1., {x -> 4.71238"));
  }

  #[test]
  fn x_cos_x_minimum() {
    clear_state();
    let result = interpret("FindMinimum[x Cos[x], {x, 2}]").unwrap();
    // Should find local minimum near x ≈ 3.4256
    assert!(result.starts_with("{-3.28837"));
  }

  #[test]
  fn quartic_minimum() {
    clear_state();
    let result = interpret("FindMinimum[x^4 - 3 x^2 + 2, {x, 2}]").unwrap();
    // Minimum near x ≈ 1.2247
    assert!(result.starts_with("{-0.25"));
    assert!(result.contains("x -> 1.224"));
  }

  #[test]
  fn multivariable_minimum() {
    clear_state();
    let result =
      interpret("FindMinimum[(x - 3)^2 + (y - 2)^2, {{x, 0}, {y, 0}}]")
        .unwrap();
    assert_eq!(result, "{0., {x -> 3., y -> 2.}}");
  }

  #[test]
  fn find_maximum_sin() {
    clear_state();
    let result = interpret("FindMaximum[Sin[x], {x, 1}]").unwrap();
    // Maximum of Sin near x=1 is at x = Pi/2 ≈ 1.5708
    assert!(result.starts_with("{1., {x -> 1.5707"));
  }

  #[test]
  fn find_maximum_negative_quadratic() {
    clear_state();
    let result = interpret("FindMaximum[-(x - 5)^2 + 10, {x, 0}]").unwrap();
    assert!(result.starts_with("{10., {x -> 5."));
  }
}

mod dt {
  use super::*;

  #[test]
  fn constant() {
    assert_eq!(interpret("Dt[5, x]").unwrap(), "0");
  }

  #[test]
  fn same_variable() {
    assert_eq!(interpret("Dt[x, x]").unwrap(), "1");
  }

  #[test]
  fn other_variable() {
    assert_eq!(interpret("Dt[y, x]").unwrap(), "Dt[y, x]");
  }

  #[test]
  fn polynomial() {
    assert_eq!(interpret("Dt[x^2, x]").unwrap(), "2*x");
  }

  #[test]
  fn product_with_dependent_var() {
    assert_eq!(interpret("Dt[x*y, x]").unwrap(), "y + x*Dt[y, x]");
  }

  #[test]
  fn sum_with_dependent_var() {
    assert_eq!(interpret("Dt[x^2 + y^2, x]").unwrap(), "2*x + 2*y*Dt[y, x]");
  }

  #[test]
  fn sin_of_same_var() {
    assert_eq!(interpret("Dt[Sin[x], x]").unwrap(), "Cos[x]");
  }

  #[test]
  fn log_of_same_var() {
    assert_eq!(interpret("Dt[Log[x], x]").unwrap(), "x^(-1)");
  }

  #[test]
  fn cubic_polynomial() {
    assert_eq!(interpret("Dt[x^3 + 2*x, x]").unwrap(), "2 + 3*x^2");
  }
}

mod minimize {
  use super::*;

  // --- Unconstrained single-variable ---

  #[test]
  fn quadratic_exact() {
    // x^2 - 4x + 5 has minimum 1 at x=2
    assert_eq!(
      interpret("Minimize[x^2 - 4*x + 5, x]").unwrap(),
      "{1, {x -> 2}}"
    );
  }

  #[test]
  fn quadratic_list_var() {
    // Same, but var given as {x}
    assert_eq!(
      interpret("Minimize[x^2 - 4*x + 5, {x}]").unwrap(),
      "{1, {x -> 2}}"
    );
  }

  #[test]
  fn cubic_unbounded() {
    // x^3 has no lower bound
    assert_eq!(
      interpret("Minimize[x^3, x]").unwrap(),
      "{-Infinity, {x -> -Infinity}}"
    );
  }

  #[test]
  fn quartic_sqrt_minimum() {
    // x^4 - 4x^2 has minimum -4 at x = ±Sqrt[2]
    assert_eq!(
      interpret("Minimize[x^4 - 4*x^2, x]").unwrap(),
      "{-4, {x -> -Sqrt[2]}}"
    );
  }

  #[test]
  fn quartic_rational_minimum() {
    // x^4 - 3x^2 + 1 has minimum -5/4 at x = ±Sqrt[3/2]
    assert_eq!(
      interpret("Minimize[x^4 - 3*x^2 + 1, x]").unwrap(),
      "{-5/4, {x -> -Sqrt[3/2]}}"
    );
  }

  #[test]
  fn exponential_minus_x() {
    // E^x - x has minimum 1 at x=0
    assert_eq!(interpret("Minimize[E^x - x, x]").unwrap(), "{1, {x -> 0}}");
  }

  // --- Unconstrained multi-variable ---

  #[test]
  fn two_var_paraboloid() {
    // (x-3)^2 + (y-2)^2 has minimum 0 at (3,2)
    assert_eq!(
      interpret("Minimize[(x - 3)^2 + (y - 2)^2, {x, y}]").unwrap(),
      "{0, {x -> 3, y -> 2}}"
    );
  }

  #[test]
  fn three_var_origin() {
    assert_eq!(
      interpret("Minimize[x^2 + y^2 + z^2, {x, y, z}]").unwrap(),
      "{0, {x -> 0, y -> 0, z -> 0}}"
    );
  }

  // --- Constrained ---

  #[test]
  fn constrained_1d_bound() {
    // x >= 1: minimum of x is 1 at x=1
    assert_eq!(
      interpret("Minimize[{x, x >= 1}, x]").unwrap(),
      "{1, {x -> 1}}"
    );
  }

  #[test]
  fn constrained_2d_quadratic() {
    // x^2 + y^2 subject to x + y >= 1: minimum 1/2 at (1/2, 1/2)
    assert_eq!(
      interpret("Minimize[{x^2 + y^2, x + y >= 1}, {x, y}]").unwrap(),
      "{1/2, {x -> 1/2, y -> 1/2}}"
    );
  }

  #[test]
  fn constrained_2d_lp() {
    // 2x + 3y subject to x + y >= 1, x >= 0, y >= 0: minimum 2 at (1,0)
    assert_eq!(
      interpret("Minimize[{2*x + 3*y, x + y >= 1, x >= 0, y >= 0}, {x, y}]")
        .unwrap(),
      "{2, {x -> 1, y -> 0}}"
    );
  }

  #[test]
  fn ilp_integers_domain_simple() {
    // Minimize[{x + y, {2*x + 3*y == 6, x >= 0, y >= 0}}, {x, y}, Integers]
    // Solutions: (0,2)=2, (3,0)=3 → minimum is 2 at (0,2)
    assert_eq!(
      interpret(
        "Minimize[{x + y, {2*x + 3*y == 6, x >= 0, y >= 0}}, {x, y}, Integers]"
      )
      .unwrap(),
      "{2, {x -> 0, y -> 2}}"
    );
  }

  #[test]
  fn ilp_funccall_vars() {
    // Minimize with Array-style variables n[1], n[2]
    // 3*n[1] + 5*n[2] == 10, n[i] >= 0, minimize n[1]+n[2]
    // Solutions: (0,2)=2 coins → minimum 2
    assert_eq!(
      interpret(
        "vars = Array[n, 2]; Minimize[{Total[vars], {vars . {3, 5} == 10, vars[[1]] >= 0, vars[[2]] >= 0}}, vars, Integers]"
      )
      .unwrap(),
      "{2, {n[1] -> 0, n[2] -> 2}}"
    );
  }

  // --- Maximize ---

  #[test]
  fn maximize_parabola() {
    // -(x-5)^2 + 10 has maximum 10 at x=5
    assert_eq!(
      interpret("Maximize[-(x - 5)^2 + 10, x]").unwrap(),
      "{10, {x -> 5}}"
    );
  }

  #[test]
  fn maximize_unbounded() {
    // x^2 - 4x + 5 has no upper bound
    assert_eq!(
      interpret("Maximize[x^2 - 4*x + 5, x]").unwrap(),
      "{Infinity, {x -> -Infinity}}"
    );
  }
}

mod integrate_rational {
  use super::*;

  #[test]
  fn integrate_1_over_x() {
    assert_eq!(interpret("Integrate[1/x, x]").unwrap(), "Log[x]");
  }

  #[test]
  fn integrate_x4_over_x2_minus_1() {
    // Polynomial long division + partial fractions with linear factors
    assert_eq!(
      interpret("Integrate[x^4/(x^2-1), x]").unwrap(),
      "x + x^3/3 + Log[1 - x]/2 - Log[1 + x]/2"
    );
  }

  #[test]
  fn integrate_1_over_x2_minus_1() {
    // Partial fractions with linear factors only
    assert_eq!(
      interpret("Integrate[1/(x^2-1), x]").unwrap(),
      "Log[1 - x]/2 - Log[1 + x]/2"
    );
  }

  #[test]
  fn integrate_1_over_x2_plus_1() {
    // Irreducible quadratic denominator
    assert_eq!(interpret("Integrate[1/(x^2+1), x]").unwrap(), "ArcTan[x]");
  }

  #[test]
  fn integrate_x_over_1_minus_x3() {
    // Mixed linear + irreducible quadratic factors
    assert_eq!(
      interpret("Integrate[x/(1-x^3), x]").unwrap(),
      "-(ArcTan[(1 + 2*x)/Sqrt[3]]/Sqrt[3]) - Log[1 - x]/3 + Log[1 + x + x^2]/6"
    );
  }

  #[test]
  fn integrate_quadratic_only() {
    // Pure irreducible quadratic: (2x+3)/(x^2+x+1)
    assert_eq!(
      interpret("Integrate[(2*x+3)/(x^2+x+1), x]").unwrap(),
      "(4*ArcTan[(1 + 2*x)/Sqrt[3]])/Sqrt[3] + Log[1 + x + x^2]"
    );
  }

  #[test]
  fn integrate_x_plus_1_over_x2_plus_1() {
    // Quadratic with both Log and ArcTan parts
    assert_eq!(
      interpret("Integrate[(x+1)/(x^2+1), x]").unwrap(),
      "ArcTan[x] + Log[1 + x^2]/2"
    );
  }
}

mod dsolve {
  use super::*;

  #[test]
  fn first_order_constant_rhs() {
    // y'[x] == 0 → y[x] -> C[1]
    assert_eq!(
      interpret("DSolve[y'[x] == 0, y[x], x]").unwrap(),
      "{{y[x] -> C[1]}}"
    );
  }

  #[test]
  fn second_order_zero_rhs() {
    // y''[x] == 0 → y[x] -> C[1] + x*C[2]
    let result = interpret("DSolve[y''[x] == 0, y[x], x]").unwrap();
    assert!(
      result == "{{y[x] -> C[1] + x*C[2]}}"
        || result == "{{y[x] -> x*C[2] + C[1]}}",
      "Got: {}",
      result
    );
  }

  #[test]
  fn harmonic_oscillator() {
    // y''[x] + y[x] == 0 → y[x] -> C[1]*Cos[x] + C[2]*Sin[x]
    let result = interpret("DSolve[y''[x] + y[x] == 0, y[x], x]").unwrap();
    assert!(
      result == "{{y[x] -> C[1]*Cos[x] + C[2]*Sin[x]}}"
        || result == "{{y[x] -> C[2]*Sin[x] + C[1]*Cos[x]}}",
      "Got: {}",
      result
    );
  }

  #[test]
  fn harmonic_oscillator_with_ic() {
    // y''[x] + y[x] == 0, y[0]==1, y'[0]==0 → y[x] -> Cos[x]
    assert_eq!(
      interpret("DSolve[{y''[x] + y[x] == 0, y[0] == 1, y'[0] == 0}, y[x], x]")
        .unwrap(),
      "{{y[x] -> Cos[x]}}"
    );
  }

  #[test]
  fn exponential_growth() {
    // y'[x] == y[x] → y[x] -> C[1]*E^x
    assert_eq!(
      interpret("DSolve[y'[x] == y[x], y[x], x]").unwrap(),
      "{{y[x] -> E^x*C[1]}}"
    );
  }

  #[test]
  fn exponential_growth_with_coeff() {
    // y'[x] == 2*y[x] → y[x] -> C[1]*E^(2*x)
    assert_eq!(
      interpret("DSolve[y'[x] == 2*y[x], y[x], x]").unwrap(),
      "{{y[x] -> E^(2*x)*C[1]}}"
    );
  }

  #[test]
  fn direct_integration() {
    // y'[x] == 2*x → y[x] -> x^2 + C[1]
    let result = interpret("DSolve[y'[x] == 2*x, y[x], x]").unwrap();
    assert!(
      result == "{{y[x] -> x^2 + C[1]}}" || result == "{{y[x] -> C[1] + x^2}}",
      "Got: {}",
      result
    );
  }

  #[test]
  fn damped_oscillator_underdamped() {
    // y'' + 2y' + 10y == 0, y(0)==1, y'(0)==0
    // Roots: -1 ± 3i → E^(-x)(Cos[3x] + Sin[3x]/3)
    let result = interpret(
      "DSolve[{y''[x] + 2*y'[x] + 10*y[x] == 0, y[0] == 1, y'[0] == 0}, y[x], x]",
    )
    .unwrap();
    assert!(
      result.contains("Cos[3*x]") && result.contains("Sin[3*x]"),
      "Got: {}",
      result
    );
  }

  #[test]
  fn real_distinct_roots() {
    // y'' - 3y' + 2y == 0 → roots r=1, r=2
    let result =
      interpret("DSolve[y''[x] - 3*y'[x] + 2*y[x] == 0, y[x], x]").unwrap();
    assert!(
      result.contains("E^x") && result.contains("E^(2*x)"),
      "Got: {}",
      result
    );
  }

  #[test]
  fn repeated_root() {
    // y'' - 2y' + y == 0 → double root r=1
    let result =
      interpret("DSolve[y''[x] - 2*y'[x] + y[x] == 0, y[x], x]").unwrap();
    // Should contain x*E^x for the repeated root part
    assert!(result.contains("E^x"), "Got: {}", result);
  }

  #[test]
  fn function_form() {
    // y'' + y == 0, returning Function form
    let result = interpret("DSolve[y''[x] + y[x] == 0, y, x]").unwrap();
    assert!(result.contains("Function["), "Got: {}", result);
  }

  #[test]
  fn spring_damper_system() {
    // Full spring-damper test matching the test script
    let result = interpret(
      "m = 1; k = 10; c = 2; sol = DSolve[{m*x''[t] + c*x'[t] + k*x[t] == 0, x[0] == 1, x'[0] == 0}, x[t], t][[1]]; x[t] /. sol",
    )
    .unwrap();
    assert!(
      result.contains("Cos") && result.contains("Sin"),
      "Got: {}",
      result
    );
  }
}

mod ndsolve {
  use super::*;

  #[test]
  fn exponential_growth() {
    // NDSolve y'=y, y(0)=1, check y(0.5) ≈ E^0.5
    let result = interpret(
      "sol = NDSolve[{y'[x] == y[x], y[0] == 1}, y, {x, 0, 1}]; y[0.5] /. sol[[1]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let expected = std::f64::consts::E.powf(0.5);
    assert!(
      (val - expected).abs() < 0.001,
      "Expected {}, got {}",
      expected,
      val
    );
  }

  #[test]
  fn linear_growth() {
    // NDSolve y'=1, y(0)=0, check y(0.5) ≈ 0.5
    let result = interpret(
      "sol = NDSolve[{y'[x] == 1, y[0] == 0}, y, {x, 0, 1}]; y[0.5] /. sol[[1]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 0.5).abs() < 0.001, "Expected 0.5, got {}", val);
  }

  #[test]
  fn quadratic_growth() {
    // NDSolve y'=x, y(0)=0, check y(1) ≈ 0.5
    let result = interpret(
      "sol = NDSolve[{y'[x] == x, y[0] == 0}, y, {x, 0, 1}]; y[1] /. sol[[1]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 0.5).abs() < 0.001, "Expected 0.5, got {}", val);
  }

  #[test]
  fn interpolating_function_display() {
    // InterpolatingFunction should display with <> for data
    let result =
      interpret("NDSolve[{y'[x] == y[x], y[0] == 1}, y, {x, 0, 1}]").unwrap();
    assert!(
      result.contains("InterpolatingFunction") && result.contains("<>"),
      "Got: {}",
      result
    );
  }

  #[test]
  fn second_order_harmonic() {
    // NDSolve y'' + y = 0, y(0)=1, y'(0)=0, check y(Pi) ≈ -1
    let result = interpret(
      "sol = NDSolve[{y''[x] + y[x] == 0, y[0] == 1, y'[0] == 0}, y, {x, 0, 4}]; y[N[Pi]] /. sol[[1]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - (-1.0)).abs() < 0.01, "Expected -1.0, got {}", val);
  }
}
