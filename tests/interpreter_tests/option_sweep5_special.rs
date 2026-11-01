// Argument forms of special functions found missing by the reference-page
// parameter sweep (round 5). Expected values are the reference pages'
// outputs (cross-checked with mpmath).

use super::*;

mod discrete_asymptotic;
mod wavelets;

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
/// `precision` whose digits start with `digits` — as the reference page
/// shows them, so its last (rounded) digit is not compared.
fn assert_digits(expr: &str, digits: &str, precision: &str) {
  let out = interpret(expr).unwrap();
  let shown = &digits[..digits.len() - 1];
  assert!(
    out.starts_with(shown) && out.ends_with(&format!("`{precision}")),
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

mod inverse_jacobi {
  use super::*;

  #[test]
  fn machine_values_leaving_the_real_domain() {
    // These used to hang (R_F of a NaN never converged).
    assert_complex(
      "InverseJacobiDN[0.3, 0.7]",
      2.075363135292469,
      1.1247480244947659,
      1e-12,
    );
    assert_complex("InverseJacobiND[0.5, 0.3]", 0.0, 1.5344909127565234, 1e-12);
    assert_close("InverseJacobiDS[2., 0.2]", 0.514024688619913, 1e-13);
    assert_close("InverseJacobiND[3., 2]", 0.9772817890897876, 1e-13);
    assert_close("InverseJacobiSD[3., 2]", 1.081549419743707, 1e-13);
    assert_close("InverseJacobiSN[0.5, 2.]", 0.5840828416771519, 1e-13);
    assert_close("InverseJacobiCD[0.3, 0.5]", 1.547019746805618, 1e-13);
    assert_close("InverseJacobiSC[3.5, 0.3]", 1.3830436305000104, 1e-13);
  }

  #[test]
  fn complex_arguments() {
    assert_complex(
      "InverseJacobiCD[0.2 + I, 0.1 - 0.3 I]",
      1.4921507780315901,
      -1.018348957851805,
      1e-12,
    );
    assert_complex(
      "InverseJacobiDN[0.2 + I, 0.1 - 0.3 I]",
      1.8348313856302043,
      1.6912817010277799,
      1e-12,
    );
    assert_complex(
      "InverseJacobiCS[1.2 + 0.3 I, 1.5 - I]",
      0.6902764916291189,
      -0.22329611418895046,
      1e-12,
    );
    assert_complex(
      "InverseJacobiCN[0.2 - 0.3 I, 0.1 + 0.5 I]",
      1.297391559,
      0.4355993206,
      1e-8,
    );
    assert_complex(
      "InverseJacobiSN[0.2 + I, 0.1 - 0.3 I]",
      0.09689212763,
      0.89110195,
      1e-8,
    );
    assert_complex(
      "InverseJacobiSD[0.2 + I, 0.1 - 0.3 I]",
      0.2105037976,
      0.867139519,
      1e-8,
    );
    assert_complex(
      "InverseJacobiDC[0.2 + I, 0.1 - 0.3 I]",
      1.428652564,
      0.7035989084,
      1e-8,
    );
    assert_complex(
      "InverseJacobiND[0.2 + I, 0.1 - 0.3 I]",
      0.2384133214,
      1.9027861,
      1e-8,
    );
    assert_complex(
      "InverseJacobiSC[1.2 + 0.3 I, 1.5 - I]",
      1.107667754,
      -0.1155805117,
      1e-8,
    );
  }

  #[test]
  fn arbitrary_precision() {
    assert_digits(
      "N[InverseJacobiCD[1/2, 1/2], 50]",
      "1.31845194449596859872958298180484612304528136436878",
      "50.",
    );
    assert_digits(
      "N[InverseJacobiNC[5/3, 3/5], 50]",
      "1.01066697226036685206615313051924539139207955916979",
      "50.",
    );
    assert_digits(
      "N[InverseJacobiDC[3, 1/2], 50]",
      "1.51095865141876726055020445739787811629821677724029",
      "50.",
    );
    assert_digits(
      "N[InverseJacobiSC[3, 1/2], 50]",
      "1.40641199399314547624248043323548138694975622331844",
      "50.",
    );
    assert_digits(
      "InverseJacobiCD[1/2, 0.5`32]",
      "1.318451944495968598729582981805",
      "32.",
    );
    let dn = interpret("N[InverseJacobiDN[1/2, 1/2], 50]").unwrap();
    assert!(
      dn.starts_with("1.85407467730137191843385034719526004621759882352")
        && dn.contains("+ 1.02805680105212673297761093939137023961728586009"),
      "{dn}"
    );
    let nd = interpret("N[InverseJacobiND[1/3, 1/2], 50]").unwrap();
    assert!(
      nd.starts_with("1.51095865141876726055020445739787811629821677724029")
        && nd.ends_with("*I"),
      "{nd}"
    );
  }

  #[test]
  fn special_values() {
    assert_eq!(interpret("InverseJacobiCD[v, Infinity]").unwrap(), "0");
    assert_eq!(interpret("InverseJacobiNC[v, Infinity]").unwrap(), "0");
    assert_eq!(interpret("InverseJacobiSN[v, Infinity]").unwrap(), "0");
    assert_eq!(interpret("InverseJacobiDC[v, 0]").unwrap(), "ArcSec[v]");
    assert_eq!(interpret("InverseJacobiDN[v, 1]").unwrap(), "ArcSech[v]");
    assert_eq!(interpret("InverseJacobiCD[v, 0]").unwrap(), "ArcCos[v]");
    assert_eq!(interpret("InverseJacobiDN[1, m]").unwrap(), "0");
  }

  #[test]
  fn plots_are_fast() {
    for f in ["CN", "SN", "SD", "DS", "NS", "DC"] {
      let out = interpret(&format!(
        "Head[Plot[InverseJacobi{f}[x, 1/3], {{x, -2, 2}}]]"
      ))
      .unwrap();
      assert_eq!(out, "Graphics", "{f}");
    }
  }
}

mod elliptic_f {
  use super::*;

  #[test]
  fn complex_amplitude_and_parameter() {
    assert_complex(
      "EllipticF[3 + 2.5 I, 2.3 - 1.5 I]",
      2.13626296176181,
      -0.573329373615825,
      1e-12,
    );
    assert_complex(
      "EllipticF[2., 3]",
      1.00107738045611,
      -1.49027804474453,
      1e-12,
    );
  }

  #[test]
  fn arbitrary_precision() {
    assert_digits(
      "N[EllipticK[1/3], 30]",
      "1.733916885257935025084922325668",
      "30.",
    );
    assert_digits(
      "N[EllipticF[1/2, 1/3], 30]",
      "0.506847756265431109203677128745",
      "30.",
    );
  }
}

mod inverse_weierstrass_p {
  use super::*;

  #[test]
  fn representative_matches_wolfram() {
    assert_close("Chop[InverseWeierstrassP[2., {1, 2}]]", 1.90163, 1e-5);
    assert_close("InverseWeierstrassP[7., {1, 2}]", -0.378197, 1e-6);
    assert_close("InverseWeierstrassP[11., {3, 2}]", -0.301707, 1e-6);
    assert_close(
      "InverseWeierstrassP[3.2222222222222222, {1, 2}]",
      -0.5590535109500647,
      1e-12,
    );
    assert_complex(
      "InverseWeierstrassP[3. + I, {I, I + 2}]",
      -0.556669,
      0.0895241,
      1e-6,
    );
    assert_complex(
      "N[InverseWeierstrassP[0, {1, 2}]]",
      1.30836,
      0.812185,
      1e-5,
    );
    assert_complex(
      "N[InverseWeierstrassP[0, {1/2, 1/2}]]",
      1.63117,
      1.02908,
      1e-4,
    );
    assert_complex(
      "InverseWeierstrassP[0.5, {1, 2}]",
      1.3083647626370971,
      0.4690885106578608,
      1e-12,
    );
  }

  #[test]
  fn pole() {
    assert_eq!(
      interpret("InverseWeierstrassP[0, {0, 0}]").unwrap(),
      "ComplexInfinity"
    );
  }
}

mod gudermannian {
  use super::*;

  #[test]
  fn numerics() {
    assert_digits(
      "N[Gudermannian[4/3], 50]",
      "1.05532739575939671672512012198767886795343848118984",
      "50.",
    );
    assert_close("Re[N[Gudermannian[I Pi + 5]]]", 1.58427, 1e-5);
    assert_complex(
      "Gudermannian[2. + 1. I]",
      1.4229114624592265,
      0.2275106584319431,
      1e-12,
    );
    let g = interpret("Gudermannian[1.11111111111111111111111]").unwrap();
    assert!(g.starts_with("0.93475707376596190564"), "{g}");
  }

  #[test]
  fn around() {
    assert_eq!(
      interpret("Gudermannian[Around[3, 0.01]]").unwrap(),
      "Around[1.4713043411171927, 0.000993279274194332]"
    );
  }
}

mod owen_t {
  use super::*;

  #[test]
  fn numerics() {
    assert_close("OwenT[4., 1]", 1.5835119382780334e-05, 1e-15);
    assert_close("OwenT[4., -7]", -1.5835620916559962e-05, 1e-15);
    assert_close("OwenT[0, .5]", 0.0737918088252166, 1e-14);
    assert_digits(
      "N[OwenT[1/8, -1], 50]",
      "-0.12376305449537457063916405905206417131142512192856",
      "50.",
    );
    assert_digits(
      "OwenT[.44444444444444444444444, -7]",
      "-0.164167299506686528999",
      "23.",
    );
    assert_digits(
      "OwenT[1/7, -1`100]",
      "-0.12338696537969482583605158962966645768335732151369",
      "100.",
    );
    assert_complex(
      "OwenT[2 + I, -7.5]",
      0.012637129894601971,
      0.012452243168905785,
      1e-12,
    );
    assert_complex(
      "OwenT[0.3 + 0.2 I, 0.5]",
      0.07167811284146922,
      -0.00464379214279158,
      1e-12,
    );
    assert_close("Table[OwenT[x, 2], {x, -2, 2}][[1]] // N", 0.0113749, 1e-6);
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("OwenT[x, 1]").unwrap(),
      "(Erfc[-(x/Sqrt[2])]*Erfc[x/Sqrt[2]])/8"
    );
    assert_eq!(interpret("OwenT[-x, a]").unwrap(), "OwenT[x, a]");
    assert_eq!(interpret("OwenT[-x, a] == OwenT[x, a]").unwrap(), "True");
    assert_eq!(interpret("OwenT[0, 1]").unwrap(), "1/8");
  }
}

mod mittag_leffler {
  use super::*;

  #[test]
  fn negative_alpha() {
    assert_close("MittagLefflerE[-7., 5]", -3.968254014136951e-05, 1e-15);
    assert_close("MittagLefflerE[-.2, .51, .87]", -51.632978650108164, 1e-12);
  }

  #[test]
  fn cancelling_series() {
    // E_(1/2)(-x) = e^(x^2) Erfc[x]
    assert_close("MittagLefflerE[1/2, -3.] - Exp[9.] Erfc[3.]", 0.0, 1e-14);
    assert_close("MittagLefflerE[1/4, -4.]", 0.17291766990277474, 1e-14);
    assert_close("MittagLefflerE[0.7, -4.]", 0.09976025489051461, 1e-14);
    assert_eq!(
      interpret("Head[Plot[MittagLefflerE[1/4, z], {z, -4, 1}]]").unwrap(),
      "Graphics"
    );
  }

  #[test]
  fn precision_and_complex() {
    assert_digits(
      "N[MittagLefflerE[2/5, 1/3, 8/7], 50]",
      "12.8104636847114408031859174465428728984483824623",
      "50.",
    );
    assert_digits(
      "MittagLefflerE[1/5, 1/3, 1/7`100]",
      "0.47877101006625880852722205907262445782688970135012",
      "100.",
    );
    assert_complex("MittagLefflerE[2.8, 8 + I]", 2.89216, 0.261472, 1e-5);
  }
}

mod coulomb {
  use super::*;

  #[test]
  fn general_order_and_complex_eta() {
    assert_close("N[CoulombF[1/2, 7, 1]]", 9.264698838537824e-8, 1e-12);
    assert_close("N[CoulombF[1/2, -7, 1]]", -0.37457574802248095, 1e-12);
    assert_complex(
      "CoulombF[2/3, 0.7 + 0.1 I, 3.2]",
      1.132683472485845,
      -0.01993514125618819,
      1e-12,
    );
    assert_complex(
      "CoulombG[2/3, 0.7 + 0.1 I, 3.2]",
      0.3691002882039472,
      0.15764665951060694,
      1e-10,
    );
    assert_complex("CoulombH1[2/3, 0.7 + 0.1 I, 3.2]", 0.389035, 1.29033, 1e-5);
    assert_complex(
      "CoulombH2[2/3, 0.7 + 0.1 I, 3.2]",
      0.349165,
      -0.975037,
      1e-5,
    );
    assert_close(
      "CoulombF[-2/5, -3, 1.23456789101112131415]",
      -0.13746135373248888,
      1e-9,
    );
    assert_close(
      "CoulombG[-2/5, -3, 1.23456789101112131415]",
      0.6271565711206498,
      1e-9,
    );
    assert_close("CoulombF[3, 0.12, 7.]", 0.158992, 1e-6);
  }
}

mod degree_trig {
  use super::*;

  #[test]
  fn parity_stays_in_degree_form() {
    assert_eq!(interpret("SinDegrees[-x]").unwrap(), "-SinDegrees[x]");
    assert_eq!(interpret("CosDegrees[-x]").unwrap(), "CosDegrees[x]");
    assert_eq!(interpret("TanDegrees[-x]").unwrap(), "-TanDegrees[x]");
    assert_eq!(interpret("CotDegrees[-x]").unwrap(), "-CotDegrees[x]");
    assert_eq!(interpret("SecDegrees[-x]").unwrap(), "SecDegrees[x]");
    assert_eq!(interpret("CscDegrees[-x]").unwrap(), "-CscDegrees[x]");
    assert_eq!(
      interpret("CotDegrees[-x + 180 k]").unwrap(),
      "CotDegrees[180*k - x]"
    );
    assert_eq!(interpret("SinDegrees[180/24]").unwrap(), "SinDegrees[15/2]");
  }

  #[test]
  fn complex_and_around() {
    assert_complex("SinDegrees[2.5 + I]", 0.043626, 0.0174376, 1e-6);
    assert_complex("CosDegrees[22.5 + I]", 0.92402, -0.00667942, 1e-5);
    assert_complex("TanDegrees[2.5 + I]", 0.0436476, 0.0174848, 1e-6);
    assert_complex("CotDegrees[2.5 + I]", 19.7426, -7.90869, 1e-5);
    assert_eq!(
      interpret("SinDegrees[Around[2, 0.01]]").unwrap(),
      "Around[0.03489949670250097, 0.00017442660445712328]"
    );
  }

  #[test]
  fn arbitrary_precision() {
    assert_digits(
      "N[CosDegrees[12/10], 50]",
      "0.99978068347484548461275180157263782579660382229773",
      "50.",
    );
    assert_digits(
      "N[TanDegrees[122/10], 50]",
      "0.21620765294566872792905409607419258011180552551477",
      "50.",
    );
    assert_digits(
      "N[CotDegrees[122/10], 50]",
      "4.6251831809639600767643314583492252133287770117044",
      "50.",
    );
    assert_digits(
      "N[SecDegrees[112/10], 50]",
      "1.0194145925498796285977645402557475536058569058870",
      "50.",
    );
  }

  #[test]
  fn derivative() {
    assert_eq!(
      interpret("D[SinDegrees[x], x]").unwrap(),
      "(Pi*CosDegrees[x])/180"
    );
  }

  #[test]
  fn interval() {
    assert_eq!(
      interpret("CotDegrees[Interval[{-45, 45}]]").unwrap(),
      "Interval[{-Infinity, -1}, {1, Infinity}]"
    );
  }
}

mod around_numeric_value {
  use super::*;

  #[test]
  fn numeric_constant_value_and_coth() {
    assert_eq!(
      interpret("Around[Pi, 0.1]").unwrap(),
      "Around[3.141592653589793, 0.1]"
    );
    assert_close(
      "Coth[Around[Sqrt[3], 0.1]][\"Value\"]",
      1.0646250629022294,
      1e-15,
    );
    assert_close(
      "Coth[Around[Sqrt[3], 0.1]][\"Uncertainty\"]",
      0.013342652455957591,
      1e-15,
    );
  }
}
