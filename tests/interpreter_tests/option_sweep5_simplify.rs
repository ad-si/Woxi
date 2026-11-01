// Symbolic simplification and transformation forms found missing by the
// reference-page parameter sweep (round 5): Simplify/FullSimplify/Refine
// identities, PiecewiseExpand of step-like functions, FunctionExpand closed
// forms and ComplexExpand. Expected values are the ones the Wolfram
// reference pages show.

use super::*;

fn ev(code: &str) -> String {
  interpret(code).unwrap()
}

mod piecewise_expand_rounding {
  use super::*;

  #[test]
  fn floor_and_ceiling_with_a_step() {
    assert_eq!(
      ev("PiecewiseExpand[Floor[x, 2], 0 < x < 4]"),
      "Piecewise[{{2, x >= 2}}, 0]"
    );
    assert_eq!(
      ev("PiecewiseExpand[Ceiling[x, 2], 0 < x < 4]"),
      "Piecewise[{{2, x <= 2}}, 4]"
    );
  }

  #[test]
  fn round_breaks_ties_to_even() {
    assert_eq!(
      ev("PiecewiseExpand[Round[x], 0 < x < 4]"),
      "Piecewise[{{1, 1/2 < x < 3/2}, {2, 3/2 <= x <= 5/2}, \
       {3, 5/2 < x < 7/2}, {4, x >= 7/2}}, 0]"
    );
  }

  #[test]
  fn integer_and_fractional_part() {
    assert_eq!(
      ev("PiecewiseExpand[IntegerPart[x], 0 < x < 4]"),
      "Piecewise[{{1, Inequality[1, LessEqual, x, Less, 2]}, \
       {2, Inequality[2, LessEqual, x, Less, 3]}, {3, x >= 3}}, 0]"
    );
    assert_eq!(
      ev("PiecewiseExpand[FractionalPart[x], 0 < x < 4]"),
      "Piecewise[{{-3 + x, x >= 3}, {-2 + x, Inequality[2, LessEqual, x, \
       Less, 3]}, {-1 + x, Inequality[1, LessEqual, x, Less, 2]}}, x]"
    );
    assert_eq!(
      ev("PiecewiseExpand[IntegerPart[x], -2 < x < 2]"),
      "Piecewise[{{-1, x <= -1}, {1, x >= 1}}, 0]"
    );
  }

  #[test]
  fn quotient_of_a_linear_numerator() {
    assert_eq!(
      ev("PiecewiseExpand[Quotient[2 x + 1, 3], 0 < x < 5]"),
      "Piecewise[{{1, Inequality[1, LessEqual, x, Less, 5/2]}, \
       {2, Inequality[5/2, LessEqual, x, Less, 4]}, {3, x >= 4}}, 0]"
    );
  }

  #[test]
  fn decreasing_argument_and_attained_bounds() {
    assert_eq!(
      ev("PiecewiseExpand[Floor[-x], 0 <= x < 2]"),
      "Piecewise[{{-2, x > 1}, {-1, Inequality[0, Less, x, LessEqual, 1]}}, \
       0]"
    );
    assert_eq!(ev("PiecewiseExpand[Floor[x], 0 < x < 1]"), "0");
    // Unbounded: nothing to expand.
    assert_eq!(ev("PiecewiseExpand[Floor[x], x > 0]"), "Floor[x]");
  }

  #[test]
  fn kronecker_delta() {
    assert_eq!(
      ev("PiecewiseExpand[KroneckerDelta[x, y], -2 < x < 2]"),
      "Piecewise[{{1, x - y == 0}}, 0]"
    );
  }

  #[test]
  fn waves() {
    assert_eq!(
      ev("PiecewiseExpand[SawtoothWave[x], -1 <= x <= 1]"),
      "Piecewise[{{-1 + x, x >= 1}, {x, Inequality[0, LessEqual, x, Less, \
       1]}}, 1 + x]"
    );
    assert_eq!(
      ev("PiecewiseExpand[SawtoothWave[x], -1 < x < 1]"),
      "Piecewise[{{x, x >= 0}}, 1 + x]"
    );
    assert_eq!(
      ev("PiecewiseExpand[SawtoothWave[x], 0 < x < 2]"),
      "Piecewise[{{-1 + x, x >= 1}}, x]"
    );
    assert_eq!(
      ev("PiecewiseExpand[SquareWave[x], -1 <= x <= 1]"),
      "Piecewise[{{-1, Inequality[-1/2, LessEqual, x, Less, 0] || \
       Inequality[1/2, LessEqual, x, Less, 1]}}, 1]"
    );
    assert_eq!(
      ev("PiecewiseExpand[SquareWave[x], 0 < x < 2]"),
      "Piecewise[{{-1, Inequality[1/2, LessEqual, x, Less, 1] || \
       x >= 3/2}}, 1]"
    );
    assert_eq!(
      ev("PiecewiseExpand[TriangleWave[x], 0 < x < 1]"),
      "Piecewise[{{4*(-1 + x), x >= 3/4}, {4*x, x < 1/4}}, -2*(-1 + 2*x)]"
    );
  }

  #[test]
  fn unit_box_and_triangle() {
    assert_eq!(
      ev("PiecewiseExpand[UnitBox[x]]"),
      "Piecewise[{{1, -1/2 <= x <= 1/2}}, 0]"
    );
    assert_eq!(
      ev("PiecewiseExpand[UnitBox[x], 0 < x < 2]"),
      "Piecewise[{{1, x <= 1/2}}, 0]"
    );
    assert_eq!(
      ev("PiecewiseExpand[UnitTriangle[x]]"),
      "Piecewise[{{1 - x, 0 <= x <= 1}, {1 + x, Inequality[-1, LessEqual, \
       x, Less, 0]}}, 0]"
    );
    assert_eq!(
      ev("PiecewiseExpand[UnitTriangle[x], 0 < x < 2]"),
      "Piecewise[{{1 - x, x <= 1}}, 0]"
    );
    assert_eq!(
      ev("PiecewiseExpand[UnitTriangle[a, a, a, a], 0 < a < 1]"),
      "(-1 + a)^4"
    );
    assert_eq!(ev("Simplify[UnitBox[a, a, a, a], 0 < a < 1/2]"), "1");
    assert_eq!(ev("Simplify[UnitBox[b, a], a < -1/2 && b > 0]"), "0");
    assert_eq!(ev("Simplify[UnitTriangle[b, a], a < -1 && b > 0]"), "0");
  }
}

mod refine_rounding_on_intervals {
  use super::*;

  #[test]
  fn constant_on_the_interval() {
    assert_eq!(
      ev("FullSimplify[IntegerPart[x] + IntegerPart[x + 1/2], 1/3 < x < 1/2]"),
      "0"
    );
    assert_eq!(
      ev(
        "FullSimplify[FractionalPart[x] + FractionalPart[x + 1/2], \
         0 < x < 1/3]"
      ),
      "1/2 + 2*x"
    );
    assert_eq!(
      ev("Refine[y + FractionalPart[x + 1/2], 0 < x < 1/3]"),
      "1/2 + x + y"
    );
  }
}

mod trig_at_multiples_of_pi {
  use super::*;

  #[test]
  fn half_integer_multiples() {
    for (code, expected) in [
      (
        "Assuming[m ∈ Integers, FullSimplify[Sin[Pi (1/2 + m)]]]",
        "(-1)^m",
      ),
      ("Assuming[m ∈ Integers, Refine[Cos[Pi (1/2 + m)]]]", "0"),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[Tan[Pi (1/2 + m)]]]]",
        "ComplexInfinity",
      ),
      ("Assuming[m ∈ Integers, Refine[Cot[Pi (1/2 + m)]]]", "0"),
      (
        "Assuming[m ∈ Integers, Refine[Sec[Pi (1/2 + m)]]]",
        "ComplexInfinity",
      ),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[Csc[Pi (1/2 + m)]]]]",
        "(-1)^m",
      ),
    ] {
      assert_eq!(ev(code), expected, "{code}");
    }
  }

  #[test]
  fn integer_multiples() {
    for (code, expected) in [
      ("Assuming[m ∈ Integers, Refine[Sin[Pi m]]]", "0"),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[Cot[Pi m]]]]",
        "ComplexInfinity",
      ),
      (
        "Assuming[m ∈ Integers, Refine[Csc[Pi m]]]",
        "ComplexInfinity",
      ),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[Sec[Pi m]]]]",
        "(-1)^m",
      ),
      ("Refine[Cos[Pi (2 k + 1)], k ∈ Integers]", "-1"),
    ] {
      assert_eq!(ev(code), expected, "{code}");
    }
  }

  #[test]
  fn degrees_variants() {
    for (code, expected) in [
      ("Assuming[m ∈ Integers, Refine[SinDegrees[180 m]]]", "0"),
      (
        "Assuming[m ∈ Integers, FullSimplify[SinDegrees[180 (1/2 + m)]]]",
        "(-1)^m",
      ),
      (
        "Assuming[m ∈ Integers, Refine[CosDegrees[180 (1/2 + m)]]]",
        "0",
      ),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[CosDegrees[180 m]]]]",
        "(-1)^m",
      ),
      ("Assuming[m ∈ Integers, Refine[TanDegrees[180 m]]]", "0"),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[TanDegrees[180 (1/2 + \
         m)]]]]",
        "ComplexInfinity",
      ),
      (
        "Assuming[m ∈ Integers, Refine[CotDegrees[180 (1/2 + m)]]]",
        "0",
      ),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[CotDegrees[180 m]]]]",
        "ComplexInfinity",
      ),
      (
        "Assuming[m ∈ Integers, Refine[SecDegrees[180 (1/2 + m)]]]",
        "ComplexInfinity",
      ),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[SecDegrees[180 m]]]]",
        "(-1)^m",
      ),
      (
        "Assuming[m ∈ Integers, Refine[CscDegrees[180 m]]]",
        "ComplexInfinity",
      ),
      (
        "Assuming[m ∈ Integers, FullSimplify[Refine[CscDegrees[180 (1/2 + \
         m)]]]]",
        "(-1)^m",
      ),
    ] {
      assert_eq!(ev(code), expected, "{code}");
    }
  }

  #[test]
  fn sinc_zeros_and_gamma_poles() {
    assert_eq!(
      ev("Assuming[m ∈ Integers && m != 0, FullSimplify[Sinc[Pi m]]]"),
      "0"
    );
    assert_eq!(
      ev("Assuming[m ∈ Integers, Refine[Sinc[Pi m]]]"),
      "Sinc[m*Pi]"
    );
    assert_eq!(
      ev("Assuming[{n ∈ Integers && n <= 0}, Refine[Gamma[n]]]"),
      "ComplexInfinity"
    );
    assert_eq!(ev("Refine[Gamma[n], n ∈ Integers]"), "Gamma[n]");
  }
}

mod identities_under_full_simplify {
  use super::*;

  #[test]
  fn mirror_symmetry() {
    for f in [
      "Sin",
      "Cos",
      "Tan",
      "Cot",
      "Sec",
      "Csc",
      "Sinh",
      "Cosh",
      "Tanh",
      "Coth",
      "Sech",
      "Csch",
      "SinDegrees",
      "TanDegrees",
      "CotDegrees",
      "CscDegrees",
      "ArcTan",
      "ArcCot",
      "ArcSinh",
      "ArcCsch",
      "ArcTanDegrees",
      "Gamma",
      "Zeta",
      "Erf",
      "Erfc",
      "Erfi",
      "LogisticSigmoid",
      "Haversine",
    ] {
      let code =
        format!("FullSimplify[{f}[Conjugate[z]] == Conjugate[{f}[z]]]");
      assert_eq!(ev(&code), "True", "{code}");
    }
    assert_eq!(
      ev(
        "FullSimplify[Beta[Conjugate[a], Conjugate[b]] == Conjugate[Beta[a, \
          b]]]"
      ),
      "True"
    );
    assert_eq!(
      ev(
        "FullSimplify[PascalBinomial[Conjugate[z], 2] == \
          Conjugate[PascalBinomial[z, 2]]]"
      ),
      "True"
    );
    // Log has its branch cut on the real axis: no mirror rule.
    assert_eq!(
      ev("FullSimplify[Log[Conjugate[z]] == Conjugate[Log[z]]]"),
      "Log[Conjugate[z]] == Conjugate[Log[z]]"
    );
  }

  #[test]
  fn arg_of_the_conjugate() {
    assert_eq!(
      ev("FullSimplify[Arg[Conjugate[z]] == -Arg[z], !(z < 0)]"),
      "True"
    );
    assert_eq!(
      ev("FullSimplify[Arg[Conjugate[z]] == -Arg[z]]"),
      "Arg[Conjugate[z]] == -Arg[z]"
    );
  }

  #[test]
  fn bessel_parity() {
    assert_eq!(
      ev(
        "FullSimplify[BesselJ[n, z] == (-1)^n BesselJ[n, -z], n ∈ \
          Integers]"
      ),
      "True"
    );
    assert_eq!(
      ev(
        "FullSimplify[BesselI[n, z] == (-1)^n BesselI[n, -z], n ∈ \
          Integers]"
      ),
      "True"
    );
    assert_eq!(
      ev(
        "FullSimplify[BesselY[-n, z] == (-1)^n BesselY[n, z], n ∈ \
          Integers]"
      ),
      "True"
    );
    // Without the integer assumption it is not an identity.
    assert_eq!(
      ev("FullSimplify[BesselJ[n, z] == (-1)^n BesselJ[n, -z]]"),
      "BesselJ[n, z] == (-1)^n*BesselJ[n, -z]"
    );
  }

  #[test]
  fn hankel_functions() {
    for code in [
      "FullSimplify[HankelH1[x, n] == BesselJ[x, n] + I BesselY[x, n]]",
      "FullSimplify[HankelH2[x, n] == BesselJ[x, n] - I BesselY[x, n]]",
      "FullSimplify[SphericalHankelH1[x, n] == SphericalBesselJ[x, n] + I \
       SphericalBesselY[x, n]]",
      "FullSimplify[SphericalHankelH2[x, n] == SphericalBesselJ[x, n] - I \
       SphericalBesselY[x, n]]",
    ] {
      assert_eq!(ev(code), "True", "{code}");
    }
  }

  #[test]
  fn assorted_identities() {
    for code in [
      "FullSimplify[Mod[a, -b] + Mod[-a, b] == 0]",
      "FullSimplify[Abs[x^n] == Abs[x]^n, n ∈ Reals]",
      "FullSimplify[Clip[-x] == -Clip[x]]",
      "Simplify[Ramp[x] == (x + RealAbs[x])/2]",
      "FullSimplify[Haversine[x] == Sin[x/2]^2]",
      "FullSimplify[Sinc[x] == Sqrt[Pi x/2] BesselJ[1/2, x]/x]",
      "FullSimplify[Sinc[x] == -I Sqrt[Pi I x/2] BesselI[1/2, I x]/x]",
    ] {
      assert_eq!(ev(code), "True", "{code}");
    }
    assert_eq!(
      ev("FullSimplify[Abs[x^n] == Abs[x]^n]"),
      "Abs[x^n] == Abs[x]^n"
    );
    assert_eq!(ev("FullSimplify[Sin[x] == Cos[x]]"), "Sin[x] == Cos[x]");
  }
}

mod inverse_trig_on_intervals {
  use super::*;

  #[test]
  fn of_their_own_function() {
    for f in ["Sin", "Cos", "Sec", "Csc"] {
      let code = format!("Refine[Arc{f}[{f}[z]], 0 < z < Pi/2]");
      assert_eq!(ev(&code), "z", "{code}");
    }
    assert_eq!(ev("Refine[ArcSin[Sin[z]], 0 < z < 2]"), "ArcSin[Sin[z]]");
  }

  #[test]
  fn of_the_complementary_root() {
    for (code, expected) in [
      (
        "FullSimplify[ArcCos[Sqrt[1 - x^2]], 0 <= x <= 1]",
        "ArcSin[x]",
      ),
      (
        "FullSimplify[ArcSin[Sqrt[1 - x^2]], 0 <= x <= 1]",
        "ArcCos[x]",
      ),
      (
        "FullSimplify[ArcSec[Sqrt[1/(1 - x^2)]], 0 <= x <= 1]",
        "ArcSin[x]",
      ),
      (
        "FullSimplify[ArcCsc[Sqrt[1/(1 - x^2)]], 0 <= x <= 1]",
        "ArcCos[x]",
      ),
      (
        "FullSimplify[ArcCosDegrees[Sqrt[1 - x^2]], 0 <= x <= 1]",
        "ArcSinDegrees[x]",
      ),
      (
        "FullSimplify[ArcSinDegrees[Sqrt[1 - x^2]], 0 <= x <= 1]",
        "ArcCosDegrees[x]",
      ),
      (
        "FullSimplify[ArcSecDegrees[Sqrt[1/(1 - x^2)]], 0 <= x <= 1]",
        "ArcSinDegrees[x]",
      ),
      (
        "FullSimplify[ArcCscDegrees[Sqrt[1/(1 - x^2)]], 0 <= x <= 1]",
        "ArcCosDegrees[x]",
      ),
      // x may be negative: no rewrite.
      (
        "FullSimplify[ArcCos[Sqrt[1 - x^2]], -1 <= x <= 1]",
        "ArcCos[Sqrt[1 - x^2]]",
      ),
    ] {
      assert_eq!(ev(code), expected, "{code}");
    }
  }
}

mod step_functions_under_refine {
  use super::*;

  #[test]
  fn several_arguments() {
    for (code, expected) in [
      ("Assuming[a < 0 && b > 0, Refine[UnitStep[b, b, a]]]", "0"),
      ("Assuming[a > 0, Refine[HeavisideTheta[a, a, a, a]]]", "1"),
      ("Assuming[a < 0, Refine[HeavisideTheta[a, a, a, a]]]", "0"),
      (
        "Assuming[a < 0 && b > 0, Refine[HeavisideTheta[b, b, a]]]",
        "0",
      ),
      (
        "Assuming[x < 0 && y > 0, Refine[KroneckerDelta[x, x, y]]]",
        "0",
      ),
      (
        "Assuming[x < 0 && y > 0, Refine[DiscreteDelta[x, x, y]]]",
        "0",
      ),
      (
        "Assuming[x > 0 && y > 0, Refine[KroneckerDelta[x, y]]]",
        "KroneckerDelta[x, y]",
      ),
    ] {
      assert_eq!(ev(code), expected, "{code}");
    }
  }
}

mod number_theory_under_assumptions {
  use super::*;

  #[test]
  fn gcd_and_lcm() {
    assert_eq!(
      ev("FullSimplify[GCD[p, q], {p, q} ∈ Primes && p != q]"),
      "1"
    );
    assert_eq!(
      ev("FullSimplify[LCM[p, q], {p, q} ∈ Primes && p != q]"),
      "p*q"
    );
    assert_eq!(ev("FullSimplify[GCD[p, q], {p, q} ∈ Primes]"), "GCD[p, q]");
    assert_eq!(
      ev("FullSimplify[LCM[m, n] GCD[m, n], {m, n} ∈ Integers]"),
      "Abs[m*n]"
    );
  }

  #[test]
  fn quotient() {
    assert_eq!(
      ev("Simplify[Quotient[4 n + 5, 2], Element[n, Integers]]"),
      "2*(1 + n)"
    );
    assert_eq!(
      ev("Refine[Quotient[2 n - 3, 7], (n + 2)/7 ∈ Integers]"),
      "(-3 + 2*n)/7"
    );
    assert_eq!(
      ev("Refine[Quotient[2 n - 3, 7], n ∈ Integers]"),
      "Quotient[-3 + 2*n, 7]"
    );
    assert_eq!(ev("FunctionExpand[Quotient[m, n]]"), "Floor[m/n]");
  }
}

mod function_expand_closed_forms {
  use super::*;

  #[test]
  fn special_functions() {
    for (code, expected) in [
      ("FunctionExpand[Erfc[x]]", "1 - Erf[x]"),
      (
        "FunctionExpand[HeavisidePi[x]]",
        "-HeavisideTheta[-1/2 + x] + HeavisideTheta[1/2 + x]",
      ),
      (
        "FunctionExpand[HeavisideLambda[x]]",
        "-HeavisideTheta[-1 + x] + x*HeavisideTheta[-1 + x] - \
         2*x*HeavisideTheta[x] + HeavisideTheta[1 + x] + x*HeavisideTheta[1 + \
         x]",
      ),
      (
        "FunctionExpand[DiracComb[x, y]]",
        "DiracComb[x]*DiracComb[y]",
      ),
      (
        "FunctionExpand[WhittakerM[k, m, x]]",
        "(x^(1/2 + m)*Hypergeometric1F1[1/2 - k + m, 1 + 2*m, x])/E^(x/2)",
      ),
      (
        "FunctionExpand[WhittakerW[k, m, x]]",
        "(x^(1/2 + m)*HypergeometricU[1/2 - k + m, 1 + 2*m, x])/E^(x/2)",
      ),
    ] {
      assert_eq!(ev(code), expected, "{code}");
    }
  }

  #[test]
  fn windows() {
    let c = "-1/2 <= x <= 1/2";
    for (code, value) in [
      ("HammingWindow[x]", "25/46 + (21*Cos[2*Pi*x])/46"),
      ("HannWindow[x]", "1/2 + Cos[2*Pi*x]/2"),
      ("HannWindow[x, α]", "α + Cos[2*Pi*x] - α*Cos[2*Pi*x]"),
      (
        "BlackmanWindow[x]",
        "(21 + 25*Cos[2*Pi*x] + 4*Cos[4*Pi*x])/50",
      ),
      ("DirichletWindow[x]", "1"),
      ("CosineWindow[x]", "Cos[Pi*x]"),
      ("CosineWindow[x, α]", "Cos[Pi*x]^α"),
      ("GaussianWindow[x]", "E^((-50*x^2)/9)"),
      ("CauchyWindow[x]", "(1 + 36*x^2)^(-1)"),
      ("CauchyWindow[x, α]", "(1 + 4*x^2*α^2)^(-1)"),
      ("ConnesWindow[x]", "(1 - 4*x^2)^2"),
      ("ConnesWindow[x, α]", "(-4*x^2 + α^2)^2/α^4"),
      ("WelchWindow[x]", "1 - 4*x^2"),
      ("WelchWindow[x, α]", "(-4*x^2 + α^2)/α^2"),
      (
        "KaiserWindow[x]",
        "BesselI[0, 3*Sqrt[1 - 2*x]*Sqrt[1 + 2*x]]/BesselI[0, 3]",
      ),
      (
        "KaiserWindow[x, α]",
        "BesselI[0, Sqrt[1 - 2*x]*Sqrt[1 + 2*x]*α]/BesselI[0, α]",
      ),
      (
        "BlackmanHarrisWindow[x]",
        "(35875 + 48829*Cos[2*Pi*x] + 14128*Cos[4*Pi*x] + \
         1168*Cos[6*Pi*x])/100000",
      ),
      (
        "BlackmanNuttallWindow[x]",
        "(3635819 + 4891775*Cos[2*Pi*x] + 1365995*Cos[4*Pi*x] + \
         106411*Cos[6*Pi*x])/10000000",
      ),
      (
        "NuttallWindow[x]",
        "(88942 + 121849*Cos[2*Pi*x] + 36058*Cos[4*Pi*x] + \
         3151*Cos[6*Pi*x])/250000",
      ),
      (
        "ExactBlackmanWindow[x]",
        "(3969 + 4620*Cos[2*Pi*x] + 715*Cos[4*Pi*x])/9304",
      ),
      (
        "FlatTopWindow[x]",
        "(215578947 + 416631580*Cos[2*Pi*x] + 277263158*Cos[4*Pi*x] + \
         83578947*Cos[6*Pi*x] + 6947368*Cos[8*Pi*x])/1000000000",
      ),
      (
        "KaiserBesselWindow[x]",
        "(402 + 498*Cos[2*Pi*x] + 99*Cos[4*Pi*x] + Cos[6*Pi*x])/1000",
      ),
      ("LanczosWindow[x]", "Sin[2*Pi*x]/(2*Pi*x)"),
    ] {
      assert_eq!(
        ev(&format!("FunctionExpand[{code}]")),
        format!("Piecewise[{{{{{value}, {c}}}}}, 0]"),
        "{code}"
      );
    }
  }

  #[test]
  fn windows_with_two_pieces() {
    assert_eq!(
      ev("FunctionExpand[BartlettWindow[x]]"),
      "Piecewise[{{1 - 2*x, 0 <= x <= 1/2}, {1 + 2*x, Inequality[-1/2, \
       LessEqual, x, Less, 0]}}, 0]"
    );
    assert_eq!(
      ev("FunctionExpand[PoissonWindow[x, α]]"),
      "Piecewise[{{E^(-2*x*α), 0 <= x <= 1/2}, {E^(2*x*α), \
       Inequality[-1/2, LessEqual, x, Less, 0]}}, 0]"
    );
    assert_eq!(
      ev("FunctionExpand[ParzenWindow[x]]"),
      "Piecewise[{{-2*(-1 + 2*x)^3, Inequality[1/4, Less, x, LessEqual, \
       1/2]}, {2*(1 + 2*x)^3, Inequality[-1/2, LessEqual, x, Less, -1/4]}, \
       {1 - 24*x^2 - 48*x^3, Inequality[-1/4, LessEqual, x, Less, 0]}, \
       {1 - 24*x^2 + 48*x^3, 0 <= x <= 1/4}}, 0]"
    );
  }

  #[test]
  fn parameterized_windows_evaluate() {
    assert_eq!(ev("HannWindow[1/4, 1/3]"), "1/3");
    assert_eq!(ev("HannWindow[1, 1/3]"), "0");
    assert_eq!(ev("CosineWindow[1/3, 2]"), "1/4");
    assert_eq!(ev("ConnesWindow[1/4, 1]"), "9/16");
    assert_eq!(ev("WelchWindow[0.2, 2]"), "0.96");
    assert_eq!(ev("HannWindow[x, a]"), "HannWindow[x, a]");
  }
}

mod max_min_and_sign {
  use super::*;

  #[test]
  fn dominated_arguments_drop() {
    assert_eq!(ev("Simplify[Max[1 - x, x, 1 + x]]"), "Max[1 - x, 1 + x]");
    assert_eq!(ev("Simplify[Min[1 - x, x, 1 + x]]"), "Min[1 - x, x]");
    assert_eq!(ev("Simplify[Max[x, y]]"), "Max[x, y]");
  }

  #[test]
  fn abs_times_sign() {
    assert_eq!(ev("FullSimplify[Abs[z] Sign[z]]"), "z");
    assert_eq!(ev("FullSimplify[2 Sign[z] Abs[z]]"), "2*z");
  }
}

mod trig_factor_sums {
  use super::*;

  #[test]
  fn quotient_functions() {
    for (code, expected) in [
      ("TrigFactor[Tan[x] + Tan[y]]", "Sec[x]*Sec[y]*Sin[x + y]"),
      ("TrigFactor[Cot[x] + Cot[y]]", "Csc[x]*Csc[y]*Sin[x + y]"),
      (
        "TrigFactor[Tanh[x] + Tanh[y]]",
        "Sech[x]*Sech[y]*Sinh[x + y]",
      ),
      (
        "TrigFactor[Coth[x] + Coth[y]]",
        "Csch[x]*Csch[y]*Sinh[x + y]",
      ),
      (
        "TrigFactor[Csc[x] + Csc[y]]",
        "(Cos[x/2 - y/2]*Csc[x/2]*Csc[y/2]*Sec[x/2]*Sec[y/2]*Sin[x/2 + \
         y/2])/2",
      ),
      (
        "TrigFactor[Csch[x] + Csch[y]]",
        "(Cosh[x/2 - y/2]*Csch[x/2]*Csch[y/2]*Sech[x/2]*Sech[y/2]*Sinh[x/2 + \
         y/2])/2",
      ),
      (
        "TrigFactor[Sech[x] + Sech[y]]",
        "(2*Cosh[x/2 - y/2]*Cosh[x/2 + y/2])/((Cosh[x/2] - \
         I*Sinh[x/2])*(Cosh[x/2] + I*Sinh[x/2])*(Cosh[y/2] - \
         I*Sinh[y/2])*(Cosh[y/2] + I*Sinh[y/2]))",
      ),
    ] {
      assert_eq!(ev(code), expected, "{code}");
    }
  }

  #[test]
  fn expand_with_trig() {
    assert_eq!(
      ev("Expand[Cosh[2 x], Trig -> True]"),
      "Cosh[x]^2 + Sinh[x]^2"
    );
    assert_eq!(
      ev("Expand[Coth[2 x], Trig -> True]"),
      "Coth[x]/2 + Tanh[x]/2"
    );
  }
}

mod trig_to_exp_inverse_functions {
  use super::*;

  #[test]
  fn arc_sin_cos_csch() {
    assert_eq!(
      ev("TrigToExp[ArcCos[z]]"),
      "Pi/2 + I*Log[I*z + Sqrt[1 - z^2]]"
    );
    assert_eq!(ev("TrigToExp[ArcSin[z]]"), "-I*Log[I*z + Sqrt[1 - z^2]]");
    assert_eq!(ev("TrigToExp[Csch[z]]"), "2/(-E^(-z) + E^z)");
    // The reciprocal form keeps the square root first.
    assert_eq!(
      ev("TrigToExp[ArcSec[z]]"),
      "Pi/2 + I*Log[Sqrt[1 - z^(-2)] + I/z]"
    );
  }

  #[test]
  fn degrees() {
    assert_eq!(
      ev("TrigToExp[ArcTanDegrees[z]]"),
      "((90*I)*Log[1 - I*z])/Pi - ((90*I)*Log[1 + I*z])/Pi"
    );
  }

  #[test]
  fn imaginary_coefficient_orders_like_a_real_one() {
    assert_eq!(ev("I z + Sqrt[1 - z^2]"), "I*z + Sqrt[1 - z^2]");
    assert_eq!(ev("2 z + Sqrt[1 - z^2]"), "2*z + Sqrt[1 - z^2]");
  }
}

mod complex_expand_forms {
  use super::*;

  #[test]
  fn fractional_power() {
    assert_eq!(
      ev("ComplexExpand[Sqrt[x + I y]]"),
      "(x^2 + y^2)^(1/4)*Cos[Arg[x + I*y]/2] + I*(x^2 + y^2)^(1/4)*Sin[Arg[x \
       + I*y]/2]"
    );
  }

  #[test]
  fn logarithmic_inverse_functions() {
    assert_eq!(
      ev("ComplexExpand[ArcTan[x + I y]]"),
      "-1/2*Arg[1 - I*(x + I*y)] + Arg[1 + I*(x + I*y)]/2 + I*(-1/4*Log[x^2 \
       + (1 - y)^2] + Log[x^2 + (1 + y)^2]/4)"
    );
    assert_eq!(
      ev("ComplexExpand[ArcTanh[x + I y]]"),
      "I*(-1/2*Arg[1 - x - I*y] + Arg[1 + x + I*y]/2) - Log[(1 - x)^2 + \
       y^2]/4 + Log[(1 + x)^2 + y^2]/4"
    );
  }

  #[test]
  fn target_functions() {
    assert_eq!(
      ev("ComplexExpand[Arg[x + I y], TargetFunctions -> {Re, Im}]"),
      "ArcTan[x, y]"
    );
    assert_eq!(
      ev("ComplexExpand[Sign[x + I y], TargetFunctions -> {Re, Im}]"),
      "(x + I*y)/Sqrt[x^2 + y^2]"
    );
    assert_eq!(
      ev("ComplexExpand[Re[Sin[z]], z, TargetFunctions -> Conjugate]"),
      "Cos[(-z + Conjugate[z])/2]*Sin[(z + Conjugate[z])/2]"
    );
    assert_eq!(
      ev("ComplexExpand[Re[Tan[z]], z, TargetFunctions -> Conjugate]"),
      "Sin[z + Conjugate[z]]/(Cos[z - Conjugate[z]] + Cos[z + Conjugate[z]])"
    );
  }
}

mod function_expand_with_assumptions {
  use super::*;

  #[test]
  fn two_argument_form() {
    assert_eq!(
      ev("FunctionExpand[(n - 1)!, {Element[n, Integers] && n > 0}]"),
      "Gamma[n]"
    );
    assert_eq!(ev("FunctionExpand[Sqrt[x^2], x > 0]"), "x");
  }
}
