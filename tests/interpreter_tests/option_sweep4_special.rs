// Argument forms of special functions found missing by the reference-page
// parameter sweep (round 4). Reference values are from mpmath at 60 digits.

use super::*;

/// `expr` evaluates to a machine number within `tol` of `expected`
/// (relative to `max(1, |expected|)`).
fn assert_close(expr: &str, expected: f64, tol: f64) {
  let out = interpret(expr).unwrap();
  let v: f64 = out
    .replace("*^", "e")
    .trim_end_matches('.')
    .parse()
    .unwrap_or_else(|_| panic!("{expr} gave {out}"));
  assert!(
    (v - expected).abs() <= tol * expected.abs().max(1.0),
    "{expr} gave {out}, expected {expected}"
  );
}

/// `expr` evaluates to an arbitrary-precision number tagged with
/// `precision` whose digits start with `digits`.
fn assert_digits(expr: &str, digits: &str, precision: &str) {
  let out = interpret(expr).unwrap();
  assert!(
    out.starts_with(digits) && out.ends_with(&format!("`{precision}")),
    "{expr} gave {out}, expected {digits}…`{precision}"
  );
}

/// `expr` evaluates to `re + im I` (machine numbers) within `tol`.
fn assert_complex(expr: &str, re: f64, im: f64, tol: f64) {
  assert!(
    interpret(&format!("Head[{expr}]")).unwrap() == "Complex",
    "{expr} gave {}",
    interpret(expr).unwrap()
  );
  assert_close(&format!("Re[{expr}]"), re, tol);
  assert_close(&format!("Im[{expr}]"), im, tol);
}

mod arbitrary_precision_n {
  use super::*;

  #[test]
  fn legendre_p_non_integer_degree() {
    assert_digits(
      "N[LegendreP[3/2, 2], 50]",
      "3.243939666040804915450228792970455767207515411017",
      "50.",
    );
    // Inside [-1, 1] the series is the direct 2F1 one.
    assert_digits(
      "N[LegendreP[3/2, 1/3], 30]",
      "-0.04915384690018100791464546634",
      "30.",
    );
    // An arbitrary-precision degree evaluates at its own precision.
    assert_digits(
      "LegendreP[1.5`30, 2]",
      "3.24393966604080491545022879297",
      "30.",
    );
  }

  #[test]
  fn fibonacci_non_integer_index() {
    assert_digits(
      "N[Fibonacci[15/17], 50]",
      "0.95651991392431122508582263427692298648606969012060",
      "50.",
    );
    assert_digits(
      "N[Fibonacci[15/17, 3], 30]",
      "0.886029649604731503100914014642",
      "30.",
    );
  }

  #[test]
  fn laguerre_l_non_integer_degree() {
    assert_digits(
      "N[LaguerreL[2/5, 1/3, 8/7], 50]",
      "0.68821856392835285644553764124515707164455717484011",
      "50.",
    );
  }

  #[test]
  fn hermite_h_non_integer_degree() {
    assert_digits(
      "N[HermiteH[2/3, 8/7], 50]",
      "1.79597886323858663943529058124860952227981960257342",
      "50.",
    );
  }

  #[test]
  fn q_pochhammer_non_integer_count() {
    // (a; q)_n = (a; q)_∞ / (a q^n; q)_∞.
    assert_digits(
      "N[QPochhammer[2/7, 1/9, 1/3], 25]",
      "0.8124796227545460712197297",
      "25.",
    );
  }

  #[test]
  fn harmonic_number_non_integer_index() {
    assert_digits(
      "N[HarmonicNumber[5/2, 2], 30]",
      "1.31457631074799157149961411115",
      "30.",
    );
    assert_digits("N[HarmonicNumber[E], 20]", "1.7500213805365417335", "20.");
  }
}

mod machine_numerics {
  use super::*;

  #[test]
  fn hermite_h_inexact_integer_degree() {
    assert_eq!(interpret("HermiteH[3., 5]").unwrap(), "940.");
  }

  #[test]
  fn hermite_h_non_integer_degree() {
    assert_close("HermiteH[2.5, 1.2]", 3.005732159668029, 1e-14);
    assert_close("HermiteH[2.5, 6.]", 485.8290704832117, 1e-14);
    assert_close("HermiteH[2.5, -6.]", -17194021656160.201, 1e-14);
    assert_close("HermiteH[2/3, 8/7.]", 1.7959788632385865, 1e-14);
  }

  #[test]
  fn laguerre_l_real() {
    assert_close("LaguerreL[0.4, 1/3, 8/7.]", 0.6882185639283529, 1e-14);
    assert_close("LaguerreL[0.4, 1.2]", 0.410842562865181, 1e-14);
  }

  #[test]
  fn laguerre_l_complex_argument() {
    assert_complex(
      "LaguerreL[2.5, 1.3, 8 + I]",
      8.50390868075879,
      0.576867401490914,
      1e-12,
    );
    assert_complex(
      "LaguerreL[2.5, 8 + I]",
      3.29073569415783,
      -5.88415063864657,
      1e-12,
    );
    assert_eq!(interpret("LaguerreL[2.5, x]").unwrap(), "LaguerreL[2.5, x]");
  }

  #[test]
  fn fibonacci_polynomial_non_integer_index() {
    assert_close("Fibonacci[5.8, 3]", 283.48273083294987, 1e-14);
    // The one-argument form keeps its machine formula.
    assert_eq!(interpret("Fibonacci[-0.5]").unwrap(), "0.35157758425414287");
    assert_eq!(interpret("Fibonacci[2.0]").unwrap(), "1.");
  }

  #[test]
  fn q_pochhammer_non_integer_count() {
    assert_close(
      "QPochhammer[1/3, 1/9, 1.1111111111111]",
      0.6604933117939918,
      1e-14,
    );
  }

  #[test]
  fn legendre_p_non_integer_degree() {
    assert_close("LegendreP[1.5, 0.3]", -0.08978728099047947, 1e-14);
    assert_close("LegendreP[1.5, 2.]", 3.243939666040805, 1e-14);
  }

  #[test]
  fn hurwitz_zeta_complex_a() {
    assert_complex(
      "HurwitzZeta[2.3, 8 + I]",
      0.0544700827321307,
      -0.00944851533647025,
      1e-12,
    );
    // The imaginary part of a used to be dropped.
    assert_complex(
      "HurwitzZeta[2.3, 1.5 + I]",
      0.277843688096386,
      -0.398528066713875,
      1e-12,
    );
    assert_complex(
      "Zeta[2.3, 8 + I]",
      0.0544700827321307,
      -0.00944851533647025,
      1e-12,
    );
    assert_complex(
      "HurwitzZeta[2. + I, 0.5]",
      3.50749884820183,
      2.03139880893399,
      1e-12,
    );
  }

  #[test]
  fn harmonic_number_inexact() {
    assert_close("HarmonicNumber[E, 1.]", 1.7500213805365417, 1e-15);
    assert_close("HarmonicNumber[2.5, 2]", 1.3145763107479915, 1e-15);
    assert_close("HarmonicNumber[1/2, 0.5]", 0.5587576969851386, 1e-15);
    assert_close("HarmonicNumber[2.5, -1.5]", 6.10155774864957, 1e-15);
    assert_eq!(
      interpret("HarmonicNumber[x, 2.]").unwrap(),
      "HarmonicNumber[x, 2.]"
    );
  }
}

mod special_values {
  use super::*;

  #[test]
  fn chebyshev_at_zero() {
    assert_eq!(interpret("ChebyshevT[n, 0]").unwrap(), "Cos[(n*Pi)/2]");
    assert_eq!(interpret("ChebyshevU[n, 0]").unwrap(), "Cos[(n*Pi)/2]");
    assert_eq!(interpret("ChebyshevT[n, x]").unwrap(), "ChebyshevT[n, x]");
  }

  #[test]
  fn lerch_phi_zero_shift_is_polylog() {
    assert_eq!(interpret("LerchPhi[z, s, 0]").unwrap(), "PolyLog[s, z]");
    assert_eq!(
      interpret("LerchPhi[1/2, 2, 0]").unwrap(),
      interpret("PolyLog[2, 1/2]").unwrap()
    );
  }

  #[test]
  fn elliptic_theta1_at_multiples_of_pi() {
    assert_eq!(interpret("EllipticTheta[1, Pi, 1/2]").unwrap(), "0");
    assert_eq!(interpret("EllipticTheta[1, -3 Pi, q]").unwrap(), "0");
    assert_eq!(
      interpret("EllipticTheta[1, Pi/2, 1/2]").unwrap(),
      "EllipticTheta[1, Pi/2, 1/2]"
    );
  }

  #[test]
  fn elliptic_f_infinite_parameter() {
    assert_eq!(interpret("EllipticF[\\[Phi], Infinity]").unwrap(), "0");
  }

  #[test]
  fn zeta_negative_integer_shift() {
    assert_eq!(interpret("Zeta[s, -1]").unwrap(), "1 + Zeta[s]");
    assert_eq!(
      interpret("Zeta[s, -3]").unwrap(),
      "1 + 2^(-s) + 3^(-s) + Zeta[s]"
    );
    // HurwitzZeta has a pole there for s > 0, so a symbolic s stays.
    assert_eq!(
      interpret("HurwitzZeta[s, -1]").unwrap(),
      "HurwitzZeta[s, -1]"
    );
    assert_eq!(interpret("Zeta[s, -1/2]").unwrap(), "Zeta[s, -1/2]");
  }
}

mod series_at_infinity {
  use super::*;

  #[test]
  fn asymptotic_expansions_stay_unevaluated() {
    // A Taylor series in 1/x does not exist for these; Woxi used to return
    // one with coefficients like BesselJ[0, Infinity].
    for f in [
      "BesselJ[0, x]",
      "BesselY[0, x]",
      "BesselI[0, x]",
      "BesselK[0, x]",
      "BesselK[n, x]",
      "LegendreP[1/2, x]",
      "ChebyshevT[1/3, x]",
      "LaguerreL[1/2, x]",
    ] {
      let input = format!("Series[{f}, {{x, Infinity, 3}}]");
      assert_eq!(interpret(&input).unwrap(), input);
    }
  }

  #[test]
  fn rational_functions_still_expand() {
    assert_eq!(
      interpret("Series[1/(1 + x), {x, Infinity, 3}]").unwrap(),
      "SeriesData[x, Infinity, {1, -1, 1}, 1, 4, 1]"
    );
  }
}

mod hypergeometric_family_n {
  use super::*;

  #[test]
  fn arbitrary_precision_values() {
    assert_digits(
      "N[Hypergeometric1F1[10, 1/3, -1], 50]",
      "1.0856469662771144181060999200053894821341819655655",
      "50.",
    );
    assert_digits(
      "N[Hypergeometric0F1[5/2, 3], 30]",
      "2.84569534228708148673198638072",
      "30.",
    );
    assert_digits(
      "N[Hypergeometric0F1Regularized[5/2, 3], 30]",
      "2.14068222675831771301727361237",
      "30.",
    );
    // At a pole of Gamma[b] the regularized limit z^(n+1) 0F1(n+2; z)/(n+1)!.
    assert_digits(
      "N[Hypergeometric0F1Regularized[0, -48], 50]",
      "-0.75356407978144308380327864211537660841009027837077",
      "50.",
    );
    assert_digits(
      "N[Hypergeometric0F1Regularized[-2, 3/2], 20]",
      "0.80786154386559628042",
      "20.",
    );
  }

  #[test]
  fn fresnel_and_logistic_sigmoid() {
    assert_digits(
      "N[FresnelS[2], 50]",
      "0.34341567836369824219530081595806845688654181220252",
      "50.",
    );
    assert_digits(
      "N[FresnelC[2], 50]",
      "0.48825340607534075450022350335726103768836715450921",
      "50.",
    );
    assert_digits(
      "N[FresnelS[20], 30]",
      "0.484084535925953892714754244855",
      "30.",
    );
    assert_digits(
      "N[LogisticSigmoid[1/9], 50]",
      "0.52774923505451317879089698533394369096217714446845",
      "50.",
    );
  }
}

mod more_special_values {
  use super::*;

  #[test]
  fn harmonic_number_complex_order() {
    // Reference page: 1.02598 + 0.0251513 I.
    assert_complex(
      "N[HarmonicNumber[27, 5 - I]]",
      1.0259792059718833,
      0.02515125431798788,
      1e-14,
    );
  }

  #[test]
  fn hurwitz_lerch_phi_alternating() {
    assert_eq!(
      interpret("HurwitzLerchPhi[-1, s, a]").unwrap(),
      "HurwitzZeta[s, a/2]/2^s - HurwitzZeta[s, (1 + a)/2]/2^s"
    );
    assert_eq!(interpret("HurwitzLerchPhi[-1, 2, 1]").unwrap(), "Pi^2/12");
  }

  #[test]
  fn gegenbauer_legendre_and_chebyshev_u() {
    assert_eq!(
      interpret("GegenbauerC[n, 1/2, x]").unwrap(),
      "LegendreP[n, x]"
    );
    assert_eq!(
      interpret("GegenbauerC[n, 1, x]").unwrap(),
      "ChebyshevU[n, x]"
    );
    assert_eq!(
      interpret("GegenbauerC[3, 1/2, x]").unwrap(),
      "(-3*x + 5*x^3)/2"
    );
  }
}
