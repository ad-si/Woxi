use super::*;

mod financial_derivative {
  use super::*;

  const PARAMS: &str = r#"{"StrikePrice" -> 100, "Expiration" -> 1}, {"InterestRate" -> 0.05, "Volatility" -> 0.2, "CurrentPrice" -> 100, "Dividend" -> 0}"#;

  fn eval_f64(code: &str) -> f64 {
    interpret(code).unwrap().parse().unwrap()
  }

  #[test]
  fn european_call_value() {
    let v = eval_f64(&format!(
      r#"FinancialDerivative[{{"European", "Call"}}, {PARAMS}]"#
    ));
    assert!((v - 10.450583572185565).abs() < 1e-9);
  }

  #[test]
  fn european_put_value_and_parity() {
    let call = eval_f64(&format!(
      r#"FinancialDerivative[{{"European", "Call"}}, {PARAMS}]"#
    ));
    let put = eval_f64(&format!(
      r#"FinancialDerivative[{{"European", "Put"}}, {PARAMS}, "Value"]"#
    ));
    assert!((put - 5.573526022256971).abs() < 1e-9);
    // put-call parity: C - P = S - K Exp[-r T]
    assert!((call - put - (100.0 - 100.0 * (-0.05f64).exp())).abs() < 1e-9);
  }

  #[test]
  fn greeks_list() {
    let r = interpret(&format!(
      r#"FinancialDerivative[{{"European", "Call"}}, {PARAMS}, {{"Delta", "Gamma", "Vega", "Theta", "Rho"}}]"#
    ))
    .unwrap();
    let nums: Vec<f64> = r
      .trim_matches(|c| c == '{' || c == '}')
      .split(", ")
      .map(|s| s.parse().unwrap())
      .collect();
    let expected = [
      0.6368306511756191,
      0.018762017345846895,
      37.52403469169379,
      -6.414027546438197,
      53.23248154537634,
    ];
    assert_eq!(nums.len(), 5);
    for (a, b) in nums.iter().zip(expected) {
      assert!((a - b).abs() < 1e-8, "{a} vs {b}");
    }
  }

  #[test]
  fn single_greek_is_scalar() {
    let v = eval_f64(&format!(
      r#"FinancialDerivative[{{"European", "Put"}}, {PARAMS}, "Delta"]"#
    ));
    assert!((v - -0.3631693488243809).abs() < 1e-9);
  }

  #[test]
  fn continuous_dividend() {
    let with_q = eval_f64(
      r#"FinancialDerivative[{"European", "Call"}, {"StrikePrice" -> 100, "Expiration" -> 1}, {"InterestRate" -> 0.05, "Volatility" -> 0.2, "CurrentPrice" -> 100, "Dividend" -> 0.03}]"#,
    );
    let without = eval_f64(&format!(
      r#"FinancialDerivative[{{"European", "Call"}}, {PARAMS}]"#
    ));
    assert!(with_q < without);
  }

  #[test]
  fn symbolic_spot_stays_symbolic() {
    let r = interpret(
      r#"FinancialDerivative[{"European", "Call"}, {"StrikePrice" -> 100, "Expiration" -> 1}, {"InterestRate" -> 0.05, "Volatility" -> 0.2, "CurrentPrice" -> s, "Dividend" -> 0}, "Delta"]"#,
    )
    .unwrap();
    assert!(r.contains('s') && r.contains("Erf"), "{r}");
  }

  #[test]
  fn unknown_contract_unevaluated() {
    let r = interpret(&format!(
      r#"FinancialDerivative[{{"Asian", "Call"}}, {PARAMS}]"#
    ))
    .unwrap();
    assert!(r.starts_with("FinancialDerivative["), "{r}");
  }

  /// Regression: a body that evaluates to a one-element list (here
  /// `FinancialDerivative[..., {"Delta"}]`) is a single surface for Plot3D.
  #[test]
  fn plot3d_of_one_element_list_result() {
    let r = interpret(
      r#"Head[Plot3D[FinancialDerivative[{"European", "Call"}, {"StrikePrice" -> 100, "Expiration" -> t/365}, {"InterestRate" -> 0.04, "Volatility" -> 0.55, "CurrentPrice" -> p, "Dividend" -> 0}, {"Delta"}], {p, 50, 150}, {t, 1, 90}]]"#,
    )
    .unwrap();
    assert_eq!(r, "Graphics3D");
  }
}
