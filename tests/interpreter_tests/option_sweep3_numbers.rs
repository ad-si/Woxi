// Number theory and algebra options and argument forms found missing by the
// reference-page parameter sweep (round 3). Expected values are the ones the
// Wolfram reference pages show.

use super::*;

/// Parse a machine real or complex result `a + b*I` / `a - b*I` / `b*I`.
fn parse_complex(s: &str) -> (f64, f64) {
  let s = s.trim();
  if let Some(body) = s.strip_suffix("*I") {
    // Split at the last sign that is not the leading one or an exponent sign.
    let bytes = body.as_bytes();
    for i in (1..bytes.len()).rev() {
      if (bytes[i] == b'+' || bytes[i] == b'-')
        && bytes[i - 1] == b' '
        && i >= 2
      {
        let re: f64 = body[..i - 1].trim().parse().unwrap();
        let im: f64 = body[i + 1..].trim().parse().unwrap();
        return (re, if bytes[i] == b'-' { -im } else { im });
      }
    }
    return (0.0, body.trim().parse().unwrap());
  }
  (s.parse().unwrap(), 0.0)
}

fn assert_close(expr: &str, re: f64, im: f64, tol: f64) {
  let out = interpret(expr).unwrap();
  let (r, i) = parse_complex(&out);
  assert!(
    (r - re).abs() <= tol && (i - im).abs() <= tol,
    "{expr} gave {out}, expected {re} + {im} I"
  );
}

mod find_integer_null_vector_none {
  use super::*;

  #[test]
  fn no_relation_found() {
    let r = interpret_with_stdout("FindIntegerNullVector[{E, Pi}]").unwrap();
    assert_eq!(r.result, "FindIntegerNullVector[{E, Pi}]");
    assert_eq!(
      r.warnings,
      vec![
        "FindIntegerNullVector::rnfu: FindIntegerNullVector has not found an \
         integer null vector for {E, Pi}."
      ]
    );
  }

  #[test]
  fn no_relation_within_the_norm_bound() {
    let r =
      interpret_with_stdout("FindIntegerNullVector[{E, Pi}, 1000000]").unwrap();
    assert_eq!(r.result, "FindIntegerNullVector[{E, Pi}, 1000000]");
    assert_eq!(
      r.warnings,
      vec![
        "FindIntegerNullVector::norel: There is no integer null vector for \
         {E, Pi} with norm less than or equal to 1000000."
      ]
    );
  }

  #[test]
  fn relation_above_the_norm_bound_does_not_count() {
    assert_eq!(
      interpret("FindIntegerNullVector[{1, Sqrt[2], 1 + Sqrt[2]}]").unwrap(),
      "{1, 1, -1}"
    );
    let r = interpret_with_stdout(
      "FindIntegerNullVector[{1, Sqrt[2], 1 + Sqrt[2]}, 1]",
    )
    .unwrap();
    assert_eq!(
      r.result,
      "FindIntegerNullVector[{1, Sqrt[2], 1 + Sqrt[2]}, 1]"
    );
    assert!(r.warnings[0].starts_with("FindIntegerNullVector::norel:"));
  }
}

mod lerch_phi_options {
  use super::*;

  #[test]
  fn options() {
    assert_eq!(
      interpret("Options[LerchPhi]").unwrap(),
      "{DoublyInfinite -> False, IncludeSingularTerm -> False}"
    );
  }

  #[test]
  fn doubly_infinite() {
    assert_close("LerchPhi[1, 2, 0.5]", 4.934802200544679, 0.0, 1e-12);
    // Sum over all integers k of 1/(k + 1/2)^2 = Pi^2.
    assert_close(
      "LerchPhi[1, 2, 0.5, DoublyInfinite -> True]",
      9.869604401089358,
      0.0,
      1e-12,
    );
    assert_close(
      "LerchPhi[2, 3, 0.5, DoublyInfinite -> True]",
      13.0054027696275,
      -0.5336491245309853,
      1e-9,
    );
    assert_eq!(
      interpret("LerchPhi[z, s, a, DoublyInfinite -> True]").unwrap(),
      "LerchPhi[z, s, a, DoublyInfinite -> True]"
    );
  }

  #[test]
  fn nonpositive_integer_a_omits_the_singular_term() {
    assert_eq!(
      interpret("LerchPhi[2, 3, -4]").unwrap(),
      "14843/1728 + 16*PolyLog[3, 2]"
    );
    assert_eq!(
      interpret("LerchPhi[z, 2, -2]").unwrap(),
      "1/4 + z + z^2*PolyLog[2, z]"
    );
    assert_eq!(interpret("LerchPhi[2, -3, -4]").unwrap(), "574");
  }

  #[test]
  fn include_singular_term() {
    assert_eq!(
      interpret("LerchPhi[2, 3, -4, IncludeSingularTerm -> True]").unwrap(),
      "ComplexInfinity"
    );
    assert_eq!(
      interpret(
        "LerchPhi[2, -3, -4] == LerchPhi[2, -3, -4, IncludeSingularTerm -> True]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "LerchPhi[2, 3, -3.7] == LerchPhi[2, 3, -3.7, IncludeSingularTerm -> True]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("LerchPhi[z, 0.3, -2, IncludeSingularTerm -> True]").unwrap(),
      "ComplexInfinity"
    );
    assert_eq!(
      interpret("LerchPhi[z, 0.3 I, -2, IncludeSingularTerm -> True]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("LerchPhi[z, 0, 0, IncludeSingularTerm -> True]").unwrap(),
      interpret("1/(1 - z)").unwrap()
    );
  }
}

mod polylog_above_one {
  use super::*;

  #[test]
  fn integer_order_on_the_branch_cut() {
    // PolyLog[2, 2] = Pi^2/4 - I Pi Log[2]
    assert_close(
      "PolyLog[2, 2.]",
      2.4674011002723395,
      -2.177586090303602,
      1e-13,
    );
    assert_close(
      "PolyLog[3, 2.]",
      2.76207190622892,
      -0.754693829460248,
      1e-13,
    );
    assert_close("PolyLog[4, 3.]", 3.7485098910701, -0.694275724012699, 1e-12);
    assert_close(
      "N[PolyLog[3, 2]]",
      2.76207190622892,
      -0.754693829460248,
      1e-13,
    );
  }
}

mod harmonic_number_decorated {
  use super::*;

  #[test]
  fn numeric() {
    assert_close(
      "HarmonicNumber[Pi, 3/2, 0.5]",
      0.614118093103937,
      0.0,
      1e-12,
    );
    assert_close(
      "HarmonicNumber[27, 3/2, 0.5 + I]",
      -0.04305102893936945,
      1.0512031362976717,
      1e-12,
    );
  }

  #[test]
  fn threads_over_lists() {
    let out = interpret("HarmonicNumber[{2, 3, 5, 6}, 1/2, 0.3]").unwrap();
    let values: Vec<f64> = out
      .trim_matches(|c| c == '{' || c == '}')
      .split(", ")
      .map(|v| v.parse().unwrap())
      .collect();
    for (v, e) in values.iter().zip([0.36364, 0.379228, 0.384365, 0.384662]) {
      assert!((v - e).abs() < 1e-5, "{out}");
    }
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("HarmonicNumber[2, 3/2, s]").unwrap(),
      "s + s^2/(2*Sqrt[2])"
    );
    assert_eq!(
      interpret("HarmonicNumber[4, r, s]").unwrap(),
      "s + s^2/2^r + s^3/3^r + s^4/4^r"
    );
    assert_eq!(interpret("HarmonicNumber[3, 2, 2]").unwrap(), "35/9");
    assert_eq!(
      interpret("HarmonicNumber[Infinity, r, s]").unwrap(),
      "PolyLog[r, s]"
    );
    assert_eq!(
      interpret("HarmonicNumber[n, r, s]").unwrap(),
      "HarmonicNumber[n, r, s]"
    );
  }

  #[test]
  fn decoration_one_is_the_two_argument_form() {
    assert_eq!(
      interpret("HarmonicNumber[n, r, 1] === HarmonicNumber[n, r]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HarmonicNumber[n, 1, 1] === HarmonicNumber[n]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("HarmonicNumber[n, 1]").unwrap(),
      "HarmonicNumber[n]"
    );
  }
}

mod meijer_g_scaled {
  use super::*;

  #[test]
  fn numeric_list() {
    let out =
      interpret("MeijerG[{{1/3}, {}}, {{1/2}, {1}}, {0.1, 0.3, 0.5}, 1/2]")
        .unwrap();
    let values: Vec<f64> = out
      .trim_matches(|c| c == '{' || c == '}')
      .split(", ")
      .map(|v| v.parse().unwrap())
      .collect();
    for (v, e) in values.iter().zip([
      0.051_128_452_708_100_5,
      0.12611212330424154,
      0.13390257706925418,
    ]) {
      assert!((v - e).abs() < 1e-11, "{out}");
    }
  }

  #[test]
  fn r_is_a_power_of_the_argument() {
    assert_close(
      "MeijerG[{{1/3}, {}}, {{1/2}, {1}}, 0.3, 2]",
      0.0493924694685434,
      0.0,
      1e-11,
    );
    assert_eq!(
      interpret("MeijerG[{{1/3}, {}}, {{1/2}, {1}}, 0.3, 1]").unwrap(),
      interpret("MeijerG[{{1/3}, {}}, {{1/2}, {1}}, 0.3]").unwrap()
    );
    assert_eq!(
      interpret("MeijerG[{{}, {}}, {{a}, {b}}, z, 1/3]").unwrap(),
      "MeijerG[{{}, {}}, {{a}, {b}}, z, 1/3]"
    );
  }
}

mod factor_quadratic_extension {
  use super::*;

  #[test]
  fn sqrt_extension() {
    for (input, expected) in [
      (
        "Factor[x^2 - 2, Extension -> Sqrt[2]]",
        "-((Sqrt[2] - x)*(Sqrt[2] + x))",
      ),
      (
        "Factor[x^4 - 9, Extension -> Sqrt[3]]",
        "-((Sqrt[3] - x)*(Sqrt[3] + x)*(3 + x^2))",
      ),
      (
        "Factor[x^4 - 2, Extension -> Sqrt[2]]",
        "-((Sqrt[2] - x^2)*(Sqrt[2] + x^2))",
      ),
      (
        "Factor[1 + x^4, Extension -> Sqrt[2]]",
        "-((-1 + Sqrt[2]*x - x^2)*(1 + Sqrt[2]*x + x^2))",
      ),
      (
        "Factor[(x^2 - 2)^2, Extension -> Sqrt[2]]",
        "(Sqrt[2] - x)^2*(Sqrt[2] + x)^2",
      ),
    ] {
      assert_eq!(interpret(input).unwrap(), expected, "{input}");
    }
  }

  #[test]
  fn gaussian_extension_unchanged() {
    assert_eq!(
      interpret("Factor[x^4 + 1, Extension -> I]").unwrap(),
      "(-I + x^2)*(I + x^2)"
    );
    assert_eq!(
      interpret("Factor[1 + x^2, GaussianIntegers -> True]").unwrap(),
      "(-I + x)*(I + x)"
    );
  }

  #[test]
  fn expands_back() {
    assert_eq!(
      interpret("Expand[Factor[x^6 - 8, Extension -> Sqrt[2]]]").unwrap(),
      "-8 + x^6"
    );
  }
}
