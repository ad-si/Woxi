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
    assert_eq!(interpret("D[Log[1 + t], t]").unwrap(), "1/(1 + t)");
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
    assert_eq!(interpret("D[Csc[x], x]").unwrap(), "-(Csc[x]*Cot[x])");
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
