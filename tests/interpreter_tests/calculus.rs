use super::*;

mod integrate_symbolic_bounds {
  use super::*;

  #[test]
  fn unknown_integrand_with_symbolic_bounds_stays_unevaluated() {
    assert_eq!(
      interpret("Integrate[F[x], {x, a, g[b]}]").unwrap(),
      "Integrate[F[x], {x, a, g[b]}]"
    );
  }
}

// Integrating a signed-Infinity constant over a non-empty range returns
// that same signed Infinity. The antiderivative-then-substitute path ran
// into `(-Infinity) * 0 == Indeterminate` at the lower bound; these cases
// now short-circuit before substitution.
mod integrate_infinite_constant {
  use super::*;

  #[test]
  fn neg_infinity_over_0_to_infinity_is_neg_infinity() {
    assert_eq!(
      interpret("Integrate[-Infinity, {x, 0, Infinity}]").unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn pos_infinity_over_finite_range_is_infinity() {
    assert_eq!(
      interpret("Integrate[Infinity, {x, 0, 1}]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn neg_infinity_over_finite_range_is_neg_infinity() {
    assert_eq!(
      interpret("Integrate[-Infinity, {x, 0, 1}]").unwrap(),
      "-Infinity"
    );
  }

  // Empty range (lo == hi) leaves the integral as Indeterminate because
  // the general short-circuit doesn't fire — the value of the integrand
  // never factors into a definite answer.
  #[test]
  fn neg_infinity_over_empty_range_is_indeterminate() {
    assert_eq!(
      interpret("Integrate[-Infinity, {x, 0, 0}]").unwrap(),
      "Indeterminate"
    );
  }
}

mod integrate_with_sum {
  use super::*;

  #[test]
  fn integrate_constant_wrt_other_var() {
    assert_eq!(interpret("Integrate[x*x, y]").unwrap(), "x^2*y");
    assert_eq!(interpret("Integrate[Sin[x], y]").unwrap(), "y*Sin[x]");
    assert_eq!(interpret("Integrate[Log[x], y]").unwrap(), "y*Log[x]");
    assert_eq!(interpret("Integrate[x^2 + y, y]").unwrap(), "x^2*y + y^2/2");
  }

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
      "Got: {result}"
    );
  }

  #[test]
  fn integrate_cos() {
    assert_eq!(interpret("Integrate[Cos[x], x]").unwrap(), "Sin[x]");
  }

  #[test]
  fn integrate_arccos_indefinite() {
    // ∫ ArcCos[x] dx = x ArcCos[x] - Sqrt[1 - x^2]
    assert_eq!(
      interpret("Integrate[ArcCos[x], x]").unwrap(),
      "-Sqrt[1 - x^2] + x*ArcCos[x]"
    );
  }

  #[test]
  fn integrate_arccos_definite_pi() {
    // ∫_{-1}^{1} ArcCos[x] dx = Pi
    assert_eq!(interpret("Integrate[ArcCos[x], {x, -1, 1}]").unwrap(), "Pi");
  }

  // u-substitution power law: ∫ c x (a + b x^2)^p dx =
  //   c (a + b x^2)^(p+1) / (2 b (p+1)), for non-integer p.
  // (Integer powers are handled by a separate path; Wolfram expands those.)
  #[test]
  fn integrate_x_times_sqrt_quadratic() {
    assert_eq!(
      interpret("Integrate[x Sqrt[1 - x^2], x]").unwrap(),
      "-1/3*(1 - x^2)^(3/2)"
    );
    assert_eq!(
      interpret("Integrate[x Sqrt[a - x^2], x]").unwrap(),
      "-1/3*(a - x^2)^(3/2)"
    );
    assert_eq!(
      interpret("Integrate[x Sqrt[x^2 - 1], x]").unwrap(),
      "(-1 + x^2)^(3/2)/3"
    );
    assert_eq!(
      interpret("Integrate[x Sqrt[3 x^2 + 2], x]").unwrap(),
      "(2 + 3*x^2)^(3/2)/9"
    );
    // Leading numeric coefficient on the x factor.
    assert_eq!(
      interpret("Integrate[2 x Sqrt[1 - x^2], x]").unwrap(),
      "(-2*(1 - x^2)^(3/2))/3"
    );
  }

  #[test]
  fn integrate_x_times_negative_half_power() {
    // ∫ x / Sqrt[1 - x^2] dx = -Sqrt[1 - x^2]
    assert_eq!(
      interpret("Integrate[x / Sqrt[1 - x^2], x]").unwrap(),
      "-Sqrt[1 - x^2]"
    );
    // ∫ x (1 - x^2)^(3/2) dx = -(1 - x^2)^(5/2) / 5
    assert_eq!(
      interpret("Integrate[x (1 - x^2)^(3/2), x]").unwrap(),
      "-1/5*(1 - x^2)^(5/2)"
    );
  }

  #[test]
  fn integrate_x_sqrt_quadratic_definite() {
    // ∫_0^1 x Sqrt[1 - x^2] dx = 1/3
    assert_eq!(
      interpret("Integrate[x Sqrt[1 - x^2], {x, 0, 1}]").unwrap(),
      "1/3"
    );
    // ∫_0^1 x / Sqrt[1 - x^2] dx = 1
    assert_eq!(
      interpret("Integrate[x / Sqrt[1 - x^2], {x, 0, 1}]").unwrap(),
      "1"
    );
  }

  // The same u = h(x) power law applies when h is trigonometric:
  // ∫ Sin[x] Cos[x]^(1/2) dx = -2/3 Cos[x]^(3/2)  (u = Cos[x]).
  #[test]
  fn integrate_sin_times_sqrt_cos() {
    assert_eq!(
      interpret("Integrate[Sin[x] Cos[x]^(1/2), x]").unwrap(),
      "(-2*Cos[x]^(3/2))/3"
    );
  }

  #[test]
  fn integrate_arcsin_indefinite() {
    // ∫ ArcSin[x] dx = x ArcSin[x] + Sqrt[1 - x^2]
    assert_eq!(
      interpret("Integrate[ArcSin[x], x]").unwrap(),
      "Sqrt[1 - x^2] + x*ArcSin[x]"
    );
  }

  #[test]
  fn integrate_arctan_indefinite() {
    // ∫ ArcTan[x] dx = x ArcTan[x] - Log[1 + x^2] / 2
    assert_eq!(
      interpret("Integrate[ArcTan[x], x]").unwrap(),
      "x*ArcTan[x] - Log[1 + x^2]/2"
    );
  }

  // Integration by parts of the exponential/trig integral functions.
  #[test]
  fn integrate_exp_trig_integral_functions() {
    assert_eq!(
      interpret("Integrate[SinIntegral[x], x]").unwrap(),
      "Cos[x] + x*SinIntegral[x]"
    );
    assert_eq!(
      interpret("Integrate[CosIntegral[x], x]").unwrap(),
      "x*CosIntegral[x] - Sin[x]"
    );
    assert_eq!(
      interpret("Integrate[SinhIntegral[x], x]").unwrap(),
      "-Cosh[x] + x*SinhIntegral[x]"
    );
    assert_eq!(
      interpret("Integrate[CoshIntegral[x], x]").unwrap(),
      "x*CoshIntegral[x] - Sinh[x]"
    );
    assert_eq!(
      interpret("Integrate[ExpIntegralEi[x], x]").unwrap(),
      "-E^x + x*ExpIntegralEi[x]"
    );
    // ExpIntegralEi round-trips (its correction -E^x cancels cleanly).
    assert_eq!(
      interpret("D[Integrate[ExpIntegralEi[x], x], x]").unwrap(),
      "ExpIntegralEi[x]"
    );
  }

  // Integration by parts of the inverse hyperbolic functions.
  #[test]
  fn integrate_inverse_hyperbolic() {
    assert_eq!(
      interpret("Integrate[ArcSinh[x], x]").unwrap(),
      "-Sqrt[1 + x^2] + x*ArcSinh[x]"
    );
    assert_eq!(
      interpret("Integrate[ArcCosh[x], x]").unwrap(),
      "-(Sqrt[-1 + x]*Sqrt[1 + x]) + x*ArcCosh[x]"
    );
    assert_eq!(
      interpret("Integrate[ArcTanh[x], x]").unwrap(),
      "x*ArcTanh[x] + Log[1 - x^2]/2"
    );
    assert_eq!(
      interpret("Integrate[ArcCoth[x], x]").unwrap(),
      "x*ArcCoth[x] + Log[1 - x^2]/2"
    );
    // Round-trips back to the integrand.
    assert_eq!(
      interpret("D[Integrate[ArcSinh[x], x], x]").unwrap(),
      "ArcSinh[x]"
    );
    assert_eq!(
      interpret("D[Integrate[ArcTanh[x], x], x]").unwrap(),
      "ArcTanh[x]"
    );
  }

  // Integration by parts of the error / Fresnel functions.
  #[test]
  fn integrate_erf_fresnel_family() {
    assert_eq!(
      interpret("Integrate[Erf[x], x]").unwrap(),
      "1/(E^x^2*Sqrt[Pi]) + x*Erf[x]"
    );
    assert_eq!(
      interpret("Integrate[Erfc[x], x]").unwrap(),
      "-(1/(E^x^2*Sqrt[Pi])) + x*Erfc[x]"
    );
    assert_eq!(
      interpret("Integrate[Erfi[x], x]").unwrap(),
      "-(E^x^2/Sqrt[Pi]) + x*Erfi[x]"
    );
    assert_eq!(
      interpret("Integrate[FresnelS[x], x]").unwrap(),
      "Cos[(Pi*x^2)/2]/Pi + x*FresnelS[x]"
    );
    assert_eq!(
      interpret("Integrate[FresnelC[x], x]").unwrap(),
      "x*FresnelC[x] - Sin[(Pi*x^2)/2]/Pi"
    );
    // Round-trips back to the integrand.
    assert_eq!(interpret("D[Integrate[Erf[x], x], x]").unwrap(), "Erf[x]");
  }

  #[test]
  fn integrate_arcsin_linear_reciprocal_integer() {
    // ∫ ArcSin[x/3] dx = Sqrt[9 - x^2] + x*ArcSin[x/3]
    // Coefficient a = 1/q: `q^2` moves into the Sqrt as the constant term.
    assert_eq!(
      interpret("Integrate[ArcSin[x/3], x]").unwrap(),
      "Sqrt[9 - x^2] + x*ArcSin[x/3]"
    );
    assert_eq!(
      interpret("Integrate[ArcSin[x/5], x]").unwrap(),
      "Sqrt[25 - x^2] + x*ArcSin[x/5]"
    );
  }

  #[test]
  fn integrate_arcsin_linear_integer() {
    // ∫ ArcSin[n x] dx = Sqrt[1 - n^2 x^2]/n + x*ArcSin[n x]
    assert_eq!(
      interpret("Integrate[ArcSin[2*x], x]").unwrap(),
      "Sqrt[1 - 4*x^2]/2 + x*ArcSin[2*x]"
    );
    assert_eq!(
      interpret("Integrate[ArcSin[3*x], x]").unwrap(),
      "Sqrt[1 - 9*x^2]/3 + x*ArcSin[3*x]"
    );
  }

  #[test]
  fn integrate_arcsin_linear_rational() {
    // ∫ ArcSin[(p/q) x] dx = Sqrt[q^2 - p^2 x^2]/p + x*ArcSin[(p/q) x]
    assert_eq!(
      interpret("Integrate[ArcSin[2*x/3], x]").unwrap(),
      "Sqrt[9 - 4*x^2]/2 + x*ArcSin[(2*x)/3]"
    );
  }

  #[test]
  fn integrate_arccos_linear_reciprocal_integer() {
    // ∫ ArcCos[x/3] dx = -Sqrt[9 - x^2] + x*ArcCos[x/3]
    assert_eq!(
      interpret("Integrate[ArcCos[x/3], x]").unwrap(),
      "-Sqrt[9 - x^2] + x*ArcCos[x/3]"
    );
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

  // ∫ sinh²(a x) dx = sinh(2 a x)/(4 a) - x/2;
  // ∫ cosh²(a x) dx = sinh(2 a x)/(4 a) + x/2.
  #[test]
  fn integrate_hyperbolic_squared() {
    assert_eq!(
      interpret("Integrate[Sinh[x]^2, x]").unwrap(),
      "-1/2*x + Sinh[2*x]/4"
    );
    assert_eq!(
      interpret("Integrate[Cosh[x]^2, x]").unwrap(),
      "x/2 + Sinh[2*x]/4"
    );
    // Linear argument carries the 1/a factor.
    assert_eq!(
      interpret("Integrate[Cosh[3 x]^2, x]").unwrap(),
      "x/2 + Sinh[6*x]/12"
    );
    // Symbolic coefficient.
    assert_eq!(
      interpret("Integrate[Sinh[a x]^2, x]").unwrap(),
      "-1/2*x + Sinh[2*a*x]/(4*a)"
    );
  }

  // ∫ tan²/cot²/tanh²/coth² — wolframscript keeps the linear term as
  // ArcTan[Tan[u]] / ArcTanh[Tanh[u]] rather than simplifying it back to x.
  #[test]
  fn integrate_tangent_squared_family() {
    assert_eq!(
      interpret("Integrate[Tan[x]^2, x]").unwrap(),
      "-ArcTan[Tan[x]] + Tan[x]"
    );
    assert_eq!(
      interpret("Integrate[Cot[x]^2, x]").unwrap(),
      "-ArcTan[Tan[x]] - Cot[x]"
    );
    assert_eq!(
      interpret("Integrate[Tanh[x]^2, x]").unwrap(),
      "ArcTanh[Tanh[x]] - Tanh[x]"
    );
    assert_eq!(
      interpret("Integrate[Coth[x]^2, x]").unwrap(),
      "ArcTanh[Tanh[x]] - Coth[x]"
    );
    // Linear argument carries the 1/a factor.
    assert_eq!(
      interpret("Integrate[Tan[3 x]^2, x]").unwrap(),
      "-1/3*ArcTan[Tan[3*x]] + Tan[3*x]/3"
    );
  }

  #[test]
  fn integrate_log_power_over_x() {
    // ∫ Log[x]^n/x dx = Log[x]^(n+1)/(n+1) via u = Log[x].
    assert_eq!(interpret("Integrate[Log[x]^2/x, x]").unwrap(), "Log[x]^3/3");
    assert_eq!(interpret("Integrate[Log[x]^3/x, x]").unwrap(), "Log[x]^4/4");
    // Negative powers.
    assert_eq!(
      interpret("Integrate[1/(x Log[x]^2), x]").unwrap(),
      "-Log[x]^(-1)"
    );
    assert_eq!(
      interpret("Integrate[1/(x Log[x]^3), x]").unwrap(),
      "-1/2*1/Log[x]^2"
    );
    // Constant coefficient is carried through.
    assert_eq!(
      interpret("Integrate[5 Log[x]^2/x, x]").unwrap(),
      "(5*Log[x]^3)/3"
    );
    // n = 1 and n = -1 keep their existing closed forms.
    assert_eq!(interpret("Integrate[Log[x]/x, x]").unwrap(), "Log[x]^2/2");
    assert_eq!(
      interpret("Integrate[1/(x Log[x]), x]").unwrap(),
      "Log[Log[x]]"
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

  #[test]
  fn integrate_tan() {
    // ∫ tan(x) dx = -Log[Cos[x]]
    assert_eq!(interpret("Integrate[Tan[x], x]").unwrap(), "-Log[Cos[x]]");
  }

  #[test]
  fn integrate_cot() {
    // ∫ cot(x) dx = Log[Sin[x]]
    assert_eq!(interpret("Integrate[Cot[x], x]").unwrap(), "Log[Sin[x]]");
  }

  // Products of trig functions of one argument reducing to Sin^p/Cos or
  // Cos^p/Sin. All expected strings verified against wolframscript.
  #[test]
  fn integrate_trig_quotients() {
    // Even sin power over cos: ArcTanh[Sin] minus odd sin powers.
    assert_eq!(
      interpret("Integrate[Sin[x]*Tan[x], x]").unwrap(),
      "ArcTanh[Sin[x]] - Sin[x]"
    );
    assert_eq!(
      interpret("Integrate[Sin[x]^2/Cos[x], x]").unwrap(),
      "ArcTanh[Sin[x]] - Sin[x]"
    );
    assert_eq!(
      interpret("Integrate[Sin[x]^3*Tan[x], x]").unwrap(),
      "ArcTanh[Sin[x]] - Sin[x] - Sin[x]^3/3"
    );
    assert_eq!(
      interpret("Integrate[Sin[x]^5*Tan[x], x]").unwrap(),
      "ArcTanh[Sin[x]] - Sin[x] - Sin[x]^3/3 - Sin[x]^5/5"
    );
    // Linear argument coefficient is divided through.
    assert_eq!(
      interpret("Integrate[Sin[2*x]*Tan[2*x], x]").unwrap(),
      "ArcTanh[Sin[2*x]]/2 - Sin[2*x]/2"
    );
    // Odd sin power over cos: -Log[Cos] plus even cos powers.
    assert_eq!(
      interpret("Integrate[Sin[x]^2*Tan[x], x]").unwrap(),
      "Cos[x]^2/2 - Log[Cos[x]]"
    );
    assert_eq!(
      interpret("Integrate[Sin[x]^4*Tan[x], x]").unwrap(),
      "Cos[x]^2 - Cos[x]^4/4 - Log[Cos[x]]"
    );
    assert_eq!(
      interpret("Integrate[Sin[2*x]^2*Tan[2*x], x]").unwrap(),
      "Cos[2*x]^2/4 - Log[Cos[2*x]]/2"
    );
    // Cos powers over sin.
    assert_eq!(
      interpret("Integrate[Cos[x]*Cot[x], x]").unwrap(),
      "Cos[x] + Log[Tan[x/2]]"
    );
    assert_eq!(
      interpret("Integrate[Cos[3*x]*Cot[3*x], x]").unwrap(),
      "Cos[3*x]/3 + Log[Tan[(3*x)/2]]/3"
    );
    assert_eq!(
      interpret("Integrate[Cos[x]^2*Cot[x], x]").unwrap(),
      "Log[Sin[x]] - Sin[x]^2/2"
    );
    assert_eq!(
      interpret("Integrate[Cos[x]^4*Cot[x], x]").unwrap(),
      "Log[Sin[x]] - Sin[x]^2 + Sin[x]^4/4"
    );
    // Sec/Csc spellings of the same quotients. Regression: Sin[x]*Sec[x]
    // used to reach integration by parts and gain a spurious constant
    // (-1 - Log[Cos[x]]); Cos[x]*Csc[x] hit the log-derivative rule with an
    // uncanonical base (Log[1 - Cos[x]^2]/2).
    assert_eq!(
      interpret("Integrate[Sin[x]*Sec[x], x]").unwrap(),
      "-Log[Cos[x]]"
    );
    assert_eq!(
      interpret("Integrate[Cos[x]*Csc[x], x]").unwrap(),
      "Log[Sin[x]]"
    );
  }

  // ∫ g'/g dx = Log[g] for a transcendental g (logarithmic-derivative rule).
  #[test]
  fn integrate_one_over_x_log_x() {
    assert_eq!(
      interpret("Integrate[1/(x Log[x]), x]").unwrap(),
      "Log[Log[x]]"
    );
  }

  #[test]
  fn integrate_cos_over_sin() {
    assert_eq!(
      interpret("Integrate[Cos[x]/Sin[x], x]").unwrap(),
      "Log[Sin[x]]"
    );
  }

  #[test]
  fn integrate_exp_over_one_plus_exp() {
    assert_eq!(
      interpret("Integrate[E^x/(1 + E^x), x]").unwrap(),
      "Log[1 + E^x]"
    );
  }

  #[test]
  fn integrate_exp_times_sin() {
    assert_eq!(
      interpret("Integrate[E^x Sin[x], x]").unwrap(),
      "(E^x*(-Cos[x] + Sin[x]))/2"
    );
  }

  #[test]
  fn integrate_exp_times_cos() {
    assert_eq!(
      interpret("Integrate[E^x Cos[x], x]").unwrap(),
      "(E^x*(Cos[x] + Sin[x]))/2"
    );
  }

  #[test]
  fn integrate_exp_times_trig_scaled() {
    // Scaled exponent and trig argument: a = 2, b = 3, a^2 + b^2 = 13.
    assert_eq!(
      interpret("Integrate[E^(2 x) Sin[3 x], x]").unwrap(),
      "(E^(2*x)*(-3*Cos[3*x] + 2*Sin[3*x]))/13"
    );
    // Negative exponent coefficient.
    assert_eq!(
      interpret("Integrate[E^(-x) Cos[x], x]").unwrap(),
      "(-Cos[x] + Sin[x])/(2*E^x)"
    );
  }

  #[test]
  fn integrate_exp_times_trig_symbolic() {
    assert_eq!(
      interpret("Integrate[E^(a x) Sin[b x], x]").unwrap(),
      "(E^(a*x)*(-(b*Cos[b*x]) + a*Sin[b*x]))/(a^2 + b^2)"
    );
  }

  #[test]
  fn integrate_const_times_exp_trig() {
    assert_eq!(
      interpret("Integrate[3 E^x Sin[x], x]").unwrap(),
      "(3*E^x*(-Cos[x] + Sin[x]))/2"
    );
  }

  #[test]
  fn integrate_reciprocal_trig_squared() {
    // 1/Cos[x]^2 = Sec[x]^2 etc. — integrate via the reciprocal-trig rewrite.
    assert_eq!(interpret("Integrate[1/Cos[x]^2, x]").unwrap(), "Tan[x]");
    assert_eq!(interpret("Integrate[1/Sin[x]^2, x]").unwrap(), "-Cot[x]");
    assert_eq!(interpret("Integrate[1/Cosh[x]^2, x]").unwrap(), "Tanh[x]");
    assert_eq!(interpret("Integrate[1/Sinh[x]^2, x]").unwrap(), "-Coth[x]");
    // A linear argument scales the result.
    assert_eq!(
      interpret("Integrate[1/Cos[2 x]^2, x]").unwrap(),
      "Tan[2*x]/2"
    );
  }

  #[test]
  fn integrate_reciprocal_trig_first_power() {
    assert_eq!(
      interpret("Integrate[1/Cos[x], x]").unwrap(),
      "ArcCoth[Sin[x]]"
    );
    assert_eq!(
      interpret("Integrate[1/Sin[x], x]").unwrap(),
      "-ArcTanh[Cos[x]]"
    );
  }

  #[test]
  fn integrate_deriv_over_power() {
    // ∫ g'(x)/g(x)^n dx = g^(1-n)/(1-n) (u = g(x) substitution, n >= 2).
    assert_eq!(
      interpret("Integrate[x/(x^2 + 1)^2, x]").unwrap(),
      "-1/2*1/(1 + x^2)"
    );
    assert_eq!(
      interpret("Integrate[x/(x^2 + 1)^3, x]").unwrap(),
      "-1/4*1/(1 + x^2)^2"
    );
    assert_eq!(
      interpret("Integrate[x/(x^2 + 4)^2, x]").unwrap(),
      "-1/2*1/(4 + x^2)"
    );
    // Numerator that is exactly g'(x).
    assert_eq!(
      interpret("Integrate[(2 x + 1)/(x^2 + x + 1)^2, x]").unwrap(),
      "-(1 + x + x^2)^(-1)"
    );
    // Higher-degree base.
    assert_eq!(
      interpret("Integrate[x^2/(x^3 + 1)^2, x]").unwrap(),
      "-1/3*1/(1 + x^3)"
    );
  }

  #[test]
  fn integrate_deriv_over_power_definite() {
    assert_eq!(
      interpret("Integrate[x/(x^2 + 1)^2, {x, 0, 1}]").unwrap(),
      "1/4"
    );
  }

  #[test]
  fn integrate_one_over_x_one_plus_log() {
    assert_eq!(
      interpret("Integrate[1/(x (1 + Log[x])), x]").unwrap(),
      "Log[1 + Log[x]]"
    );
  }

  #[test]
  fn integrate_scaled_log_derivative() {
    assert_eq!(
      interpret("Integrate[1/(2 x Log[x]), x]").unwrap(),
      "Log[Log[x]]/2"
    );
  }

  // Not a logarithmic derivative (the ratio isn't constant): must NOT collapse
  // to Log, but integrate to Log[x]^2/2 via the existing path.
  #[test]
  fn integrate_log_over_x_not_log_derivative() {
    assert_eq!(interpret("Integrate[Log[x]/x, x]").unwrap(), "Log[x]^2/2");
  }

  // Elementary hyperbolic / reciprocal-trig antiderivatives (wolframscript's
  // ArcCoth/ArcTanh forms for Sec/Csc).
  #[test]
  fn integrate_tanh_coth() {
    assert_eq!(interpret("Integrate[Tanh[x], x]").unwrap(), "Log[Cosh[x]]");
    assert_eq!(
      interpret("Integrate[Tanh[2 x], x]").unwrap(),
      "Log[Cosh[2*x]]/2"
    );
    assert_eq!(interpret("Integrate[Coth[x], x]").unwrap(), "Log[Sinh[x]]");
    assert_eq!(
      interpret("Integrate[Coth[3 x], x]").unwrap(),
      "Log[Sinh[3*x]]/3"
    );
  }

  #[test]
  fn integrate_sec_csc() {
    assert_eq!(
      interpret("Integrate[Sec[x], x]").unwrap(),
      "ArcCoth[Sin[x]]"
    );
    assert_eq!(
      interpret("Integrate[Sec[2 x], x]").unwrap(),
      "ArcCoth[Sin[2*x]]/2"
    );
    assert_eq!(
      interpret("Integrate[Csc[x], x]").unwrap(),
      "-ArcTanh[Cos[x]]"
    );
    assert_eq!(
      interpret("Integrate[Csc[5 x], x]").unwrap(),
      "-1/5*ArcTanh[Cos[5*x]]"
    );
  }

  #[test]
  fn integrate_sin_cos_product() {
    // ∫ Sin[x]*Cos[x] dx = -1/2*Cos[x]^2
    assert_eq!(
      interpret("Integrate[Sin[x] * Cos[x], x]").unwrap(),
      "-1/2*Cos[x]^2"
    );
  }

  // Products that are exact derivatives of an elementary function.
  #[test]
  fn integrate_derivative_products() {
    assert_eq!(interpret("Integrate[Sec[x] Tan[x], x]").unwrap(), "Sec[x]");
    assert_eq!(interpret("Integrate[Csc[x] Cot[x], x]").unwrap(), "-Csc[x]");
    assert_eq!(
      interpret("Integrate[Sech[x] Tanh[x], x]").unwrap(),
      "-Sech[x]"
    );
    assert_eq!(
      interpret("Integrate[Csch[x] Coth[x], x]").unwrap(),
      "-Csch[x]"
    );
  }

  #[test]
  fn integrate_derivative_product_with_factors() {
    // Linear argument carries a 1/a factor; constant factors pass through.
    assert_eq!(
      interpret("Integrate[Sec[2 x] Tan[2 x], x]").unwrap(),
      "Sec[2*x]/2"
    );
    assert_eq!(
      interpret("Integrate[2 Sec[x] Tan[x], x]").unwrap(),
      "2*Sec[x]"
    );
  }

  #[test]
  fn integrate_hyperbolic_squares() {
    assert_eq!(interpret("Integrate[Sech[x]^2, x]").unwrap(), "Tanh[x]");
    assert_eq!(interpret("Integrate[Csch[x]^2, x]").unwrap(), "-Coth[x]");
    assert_eq!(
      interpret("Integrate[Sech[3 x]^2, x]").unwrap(),
      "Tanh[3*x]/3"
    );
  }

  // ∫ sech = -ArcCot[Sinh], ∫ csch = -ArcTanh[Cosh] (wolframscript's forms).
  #[test]
  fn integrate_sech_csch() {
    assert_eq!(
      interpret("Integrate[Sech[x], x]").unwrap(),
      "-ArcCot[Sinh[x]]"
    );
    assert_eq!(
      interpret("Integrate[Sech[2 x], x]").unwrap(),
      "-1/2*ArcCot[Sinh[2*x]]"
    );
    assert_eq!(
      interpret("Integrate[Csch[x], x]").unwrap(),
      "-ArcTanh[Cosh[x]]"
    );
    assert_eq!(
      interpret("Integrate[Csch[3 x], x]").unwrap(),
      "-1/3*ArcTanh[Cosh[3*x]]"
    );
  }

  #[test]
  fn integrate_four_sin_cos_product() {
    // ∫ 4 Sin[x] Cos[x] dx = -2 Cos[x]^2 (up to an additive constant;
    // matches wolframscript's canonical branch).
    assert_eq!(
      interpret("Integrate[4 Sin[x] Cos[x], x]").unwrap(),
      "-2*Cos[x]^2"
    );
  }

  #[test]
  fn integrate_derivative_of_undefined_function_stays_unevaluated() {
    // Integrate[f'[x], {x, a, b}] — without the Fundamental Theorem rule,
    // both Woxi and wolframscript leave this unevaluated.
    assert_eq!(
      interpret("Integrate[f'[x], {x, a, b}]").unwrap(),
      "Integrate[Derivative[1][f][x], {x, a, b}]"
    );
  }

  #[test]
  fn integrate_sin_cos_squared() {
    // ∫ Sin[x]*Cos[x]^2 dx = -1/3*Cos[x]^3
    assert_eq!(
      interpret("Integrate[Sin[x] * Cos[x]^2, x]").unwrap(),
      "-1/3*Cos[x]^3"
    );
  }

  #[test]
  fn integrate_sin_squared_cos() {
    // ∫ Sin[x]^2*Cos[x] dx = Sin[x]^3/3
    assert_eq!(
      interpret("Integrate[Sin[x]^2 * Cos[x], x]").unwrap(),
      "Sin[x]^3/3"
    );
  }

  #[test]
  fn integrate_sin_cos_product_linear_arg() {
    // ∫ Sin[2x]*Cos[2x] dx = -Cos[4x]/8 via the double-angle identity,
    // matching wolframscript (which uses this form when |a| > 1).
    assert_eq!(
      interpret("Integrate[Sin[2*x] * Cos[2*x], x]").unwrap(),
      "-1/8*Cos[4*x]"
    );
  }

  #[test]
  fn integrate_sin_cubed_cos() {
    // ∫ Sin[x]^3*Cos[x] dx = Sin[x]^4/4
    assert_eq!(
      interpret("Integrate[Sin[x]^3 * Cos[x], x]").unwrap(),
      "Sin[x]^4/4"
    );
  }

  /// `Sin[a x] Sin[b x]` with `a != b` is the same-argument product's
  /// same-argument neighbour: the double-angle identity above only applies
  /// when both factors share one argument, so a genuinely different pair
  /// of frequencies needs the product-to-sum identity
  /// `Sin[A] Sin[B] = (Cos[A-B] - Cos[A+B]) / 2` instead. This is the
  /// orthogonality integral a Fourier series relies on.
  #[test]
  fn integrate_sin_sin_product_different_frequencies() {
    assert_eq!(
      interpret("Integrate[Sin[x] * Sin[2*x], x]").unwrap(),
      "Sin[x]/2 - Sin[3*x]/6"
    );
  }

  #[test]
  fn integrate_cos_cos_product_different_frequencies() {
    // Cos[A] Cos[B] = (Cos[A-B] + Cos[A+B]) / 2
    assert_eq!(
      interpret("Integrate[Cos[x] * Cos[3*x], x]").unwrap(),
      "Sin[2*x]/4 + Sin[4*x]/8"
    );
  }

  #[test]
  fn integrate_sin_cos_product_different_frequencies() {
    // Sin[A] Cos[B] = (Sin[A+B] + Sin[A-B]) / 2
    assert_eq!(
      interpret("Integrate[Sin[2*x] * Cos[3*x], x]").unwrap(),
      "Cos[x]/2 - Cos[5*x]/10"
    );
  }

  /// The orthogonality relations a Fourier series is built on: sines and
  /// cosines of different integer multiples of `Pi` over a symmetric
  /// period integrate to exactly zero, while a matching pair of
  /// frequencies integrates to a nonzero constant. Regression test for a
  /// bug where `Integrate[Sin[m Pi x] Sin[n Pi x], {x, -1, 1}]` (m != n)
  /// was left unevaluated instead of reducing to `0`, because the
  /// same-argument product handler bails out on differing arguments and
  /// nothing else picked up the product-to-sum identity.
  #[test]
  fn orthogonality_of_sines_and_cosines_over_symmetric_period() {
    assert_eq!(
      interpret("Integrate[Sin[1*Pi*x] * Sin[2*Pi*x], {x, -1, 1}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Integrate[Cos[1*Pi*x] * Cos[2*Pi*x], {x, -1, 1}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Integrate[Sin[2*Pi*x] * Sin[3*Pi*x], {x, -1, 1}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Integrate[Sin[2*Pi*x] * Cos[3*Pi*x], {x, -1, 1}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Integrate[Sin[2*Pi*x] * Sin[2*Pi*x], {x, -1, 1}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Integrate[Cos[3*Pi*x] * Cos[3*Pi*x], {x, -1, 1}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn integrate_sin_cos_cubed() {
    // ∫ Sin[x]*Cos[x]^3 dx = -1/4*Cos[x]^4
    assert_eq!(
      interpret("Integrate[Sin[x] * Cos[x]^3, x]").unwrap(),
      "-1/4*Cos[x]^4"
    );
  }

  #[test]
  fn integrate_sec_squared() {
    // ∫ sec²(x) dx = Tan[x]
    assert_eq!(interpret("Integrate[Sec[x]^2, x]").unwrap(), "Tan[x]");
  }

  #[test]
  fn integrate_csc_squared() {
    // ∫ csc²(x) dx = -Cot[x]
    assert_eq!(interpret("Integrate[Csc[x]^2, x]").unwrap(), "-Cot[x]");
  }

  #[test]
  fn integrate_tan_linear_arg() {
    // ∫ tan(2x) dx = -Log[Cos[2x]]/2
    assert_eq!(
      interpret("Integrate[Tan[2*x], x]").unwrap(),
      "-1/2*Log[Cos[2*x]]"
    );
  }

  #[test]
  fn integrate_sec_squared_linear_arg() {
    // ∫ sec²(3x) dx = Tan[3x]/3
    assert_eq!(interpret("Integrate[Sec[3*x]^2, x]").unwrap(), "Tan[3*x]/3");
  }

  #[test]
  fn integrate_inverse_sqrt_one_minus_x2() {
    // ∫ 1/Sqrt[1 - x^2] dx = ArcSin[x]
    assert_eq!(
      interpret("Integrate[1/Sqrt[1 - x^2], x]").unwrap(),
      "ArcSin[x]"
    );
  }

  #[test]
  fn integrate_inverse_sqrt_one_plus_x2() {
    // ∫ 1/Sqrt[1 + x^2] dx = ArcSinh[x]
    assert_eq!(
      interpret("Integrate[1/Sqrt[1 + x^2], x]").unwrap(),
      "ArcSinh[x]"
    );
  }
}

mod definite_integrals {
  use super::*;

  // Definite integral of Abs of a linear argument, via the continuous
  // antiderivative u*Abs[u]/(2 u').
  #[test]
  fn abs_linear() {
    // Interval straddling the sign change at 0.
    assert_eq!(interpret("Integrate[Abs[x], {x, -1, 1}]").unwrap(), "1");
    assert_eq!(interpret("Integrate[Abs[x], {x, -2, 3}]").unwrap(), "13/2");
    // Wholly positive / wholly negative sub-intervals.
    assert_eq!(interpret("Integrate[Abs[x], {x, 0, 2}]").unwrap(), "2");
    assert_eq!(interpret("Integrate[Abs[x], {x, 1, 3}]").unwrap(), "4");
    assert_eq!(interpret("Integrate[Abs[x], {x, -3, -1}]").unwrap(), "4");
    assert_eq!(interpret("Integrate[Abs[x], {x, 1/2, 2}]").unwrap(), "15/8");
    // Shifted and scaled arguments.
    assert_eq!(
      interpret("Integrate[Abs[x - 1], {x, 0, 3}]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("Integrate[Abs[2 x + 1], {x, -2, 2}]").unwrap(),
      "17/2"
    );
  }

  #[test]
  fn log_sin_over_half_period() {
    // Euler: ∫_0^{π/2} Log[Sin[x]] dx = -(π Log[2])/2
    assert_eq!(
      interpret("Integrate[Log[Sin[x]], {x, 0, Pi/2}]").unwrap(),
      "-1/2*(Pi*Log[2])"
    );
  }

  #[test]
  fn log_cos_over_half_period() {
    assert_eq!(
      interpret("Integrate[Log[Cos[x]], {x, 0, Pi/2}]").unwrap(),
      "-1/2*(Pi*Log[2])"
    );
  }

  #[test]
  fn log_tan_and_log_cot_over_half_period() {
    // Tan and Cot swap under x -> Pi/2 - x, so both integrals vanish.
    assert_eq!(
      interpret("Integrate[Log[Tan[x]], {x, 0, Pi/2}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Integrate[Log[Cot[x]], {x, 0, Pi/2}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn log_sin_over_full_period() {
    // ∫_0^π Log[Sin[x]] dx = -π Log[2]
    assert_eq!(
      interpret("Integrate[Log[Sin[x]], {x, 0, Pi}]").unwrap(),
      "-(Pi*Log[2])"
    );
  }

  #[test]
  fn log_sin_other_variable_name() {
    assert_eq!(
      interpret("Integrate[Log[Sin[y]], {y, 0, Pi/2}]").unwrap(),
      "-1/2*(Pi*Log[2])"
    );
  }

  #[test]
  fn log_sin_wrong_bounds_stays_unevaluated() {
    // The table entry must not misfire on other bounds. (wolframscript
    // returns a PolyLog closed form here, which woxi doesn't support yet —
    // falling through unevaluated is the safe behaviour.)
    assert_eq!(
      interpret("Integrate[Log[Sin[x]], {x, 0, 1}]").unwrap(),
      "Integrate[Log[Sin[x]], {x, 0, 1}]"
    );
  }

  #[test]
  fn definite_integral_constant_factor_linearity() {
    // Constant factors are pulled out of known definite integrals.
    assert_eq!(
      interpret("Integrate[3 Log[Sin[x]], {x, 0, Pi/2}]").unwrap(),
      "(-3*Pi*Log[2])/2"
    );
    assert_eq!(
      interpret("Integrate[Log[Sin[x]]/2, {x, 0, Pi/2}]").unwrap(),
      "-1/4*(Pi*Log[2])"
    );
    assert_eq!(
      interpret("Integrate[-Log[Sin[x]], {x, 0, Pi/2}]").unwrap(),
      "(Pi*Log[2])/2"
    );
    // Symbolic constant factor times the Bessel integral representation.
    assert_eq!(
      interpret("Integrate[Pi Cos[Sin[x]], {x, 0, Pi}]").unwrap(),
      "Pi^2*BesselJ[0, 1]"
    );
  }

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
  fn gaussian_moment_full_range() {
    // ∫_{-∞}^{∞} x^(2m) E^(-x^2) dx = (2m-1)!! Sqrt[Pi] / 2^m
    assert_eq!(
      interpret("Integrate[x^2 Exp[-x^2], {x, -Infinity, Infinity}]").unwrap(),
      "Sqrt[Pi]/2"
    );
    assert_eq!(
      interpret("Integrate[x^4 Exp[-x^2], {x, -Infinity, Infinity}]").unwrap(),
      "(3*Sqrt[Pi])/4"
    );
    assert_eq!(
      interpret("Integrate[x^6 Exp[-x^2], {x, -Infinity, Infinity}]").unwrap(),
      "(15*Sqrt[Pi])/8"
    );
    // Odd powers integrate to zero by symmetry.
    assert_eq!(
      interpret("Integrate[x^3 Exp[-x^2], {x, -Infinity, Infinity}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn gaussian_moment_with_coefficient_and_constant() {
    // Coefficient a in the exponent: ∫_{-∞}^{∞} x^2 E^(-2 x^2) dx = Sqrt[Pi/2]/4
    assert_eq!(
      interpret("Integrate[x^2 Exp[-2 x^2], {x, -Infinity, Infinity}]")
        .unwrap(),
      "Sqrt[Pi/2]/4"
    );
    // A constant factor multiplies through.
    assert_eq!(
      interpret("Integrate[3 x^2 Exp[-x^2], {x, -Infinity, Infinity}]")
        .unwrap(),
      "(3*Sqrt[Pi])/2"
    );
  }

  #[test]
  fn gaussian_moment_half_range() {
    // ∫_0^{∞} x^(2m) E^(-x^2) dx is half the full-range even moment.
    assert_eq!(
      interpret("Integrate[x^2 Exp[-x^2], {x, 0, Infinity}]").unwrap(),
      "Sqrt[Pi]/4"
    );
    assert_eq!(
      interpret("Integrate[x^4 Exp[-x^2], {x, 0, Infinity}]").unwrap(),
      "(3*Sqrt[Pi])/8"
    );
    // Odd powers over (0, ∞): ∫_0^∞ x^(2k+1) E^(-a x^2) dx = k!/(2 a^(k+1)).
    assert_eq!(
      interpret("Integrate[x Exp[-x^2], {x, 0, Infinity}]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Integrate[x^3 Exp[-x^2], {x, 0, Infinity}]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Integrate[x^3 Exp[-2 x^2], {x, 0, Infinity}]").unwrap(),
      "1/8"
    );
  }

  #[test]
  fn dirac_delta_sifting() {
    // ∫ g(x) DiracDelta[x - x0] dx = g(x0) when x0 is inside the bounds.
    assert_eq!(
      interpret("Integrate[DiracDelta[x - 2] x^2, {x, 0, 5}]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Integrate[DiracDelta[x] Exp[x], {x, -1, 1}]").unwrap(),
      "1"
    );
    // No weight factor → just 1.
    assert_eq!(
      interpret("Integrate[DiracDelta[x - 3], {x, 0, 5}]").unwrap(),
      "1"
    );
    // Infinite bounds.
    assert_eq!(
      interpret("Integrate[DiracDelta[x] Cos[x], {x, -Infinity, Infinity}]")
        .unwrap(),
      "1"
    );
    // Negative shift inside the interval.
    assert_eq!(
      interpret("Integrate[DiracDelta[x + 1] x^2, {x, -3, 3}]").unwrap(),
      "1"
    );
    // Rational root with a transcendental weight.
    assert_eq!(
      interpret("Integrate[DiracDelta[x - 1/2] Sin[Pi x], {x, 0, 1}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn dirac_delta_scaled_argument_and_outside() {
    // DiracDelta[c x + d] scales by 1/|c|: DiracDelta[2x-4] = DiracDelta[x-2]/2.
    assert_eq!(
      interpret("Integrate[DiracDelta[2 x - 4] x, {x, 0, 5}]").unwrap(),
      "1"
    );
    // Root strictly outside the bounds → integral is 0.
    assert_eq!(
      interpret("Integrate[DiracDelta[x - 2] x^2, {x, 0, 1}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn dirac_delta_symbolic_and_boundary_roots() {
    // Symbolic root over the whole real line → ConditionalExpression.
    assert_eq!(
      interpret("Integrate[f[x] DiracDelta[x - a], {x, -Infinity, Infinity}]")
        .unwrap(),
      "ConditionalExpression[f[a], Element[a, Reals]]"
    );
    // Root landing on a boundary → g(x0) * HeavisideTheta[0].
    assert_eq!(
      interpret("Integrate[DiracDelta[x - 5] x, {x, 0, 5}]").unwrap(),
      "5*HeavisideTheta[0]"
    );
    assert_eq!(
      interpret("Integrate[DiracDelta[x] (x + 1), {x, 0, 5}]").unwrap(),
      "HeavisideTheta[0]"
    );
    assert_eq!(
      interpret("Integrate[DiracDelta[x - 5] (x + 1), {x, 0, 5}]").unwrap(),
      "6*HeavisideTheta[0]"
    );
  }

  #[test]
  fn divergent_integral_returns_unevaluated() {
    // Improper integrals that diverge at an infinite bound stay unevaluated,
    // matching wolframscript's Integrate::idiv behaviour. Regression for
    // mathics calculus.py:1006.
    assert_eq!(
      interpret("Integrate[1, {x, Infinity, 0}]").unwrap(),
      "Integrate[1, {x, Infinity, 0}]"
    );
    assert_eq!(
      interpret("Integrate[x, {x, 0, Infinity}]").unwrap(),
      "Integrate[x, {x, 0, Infinity}]"
    );
    assert_eq!(
      interpret("Integrate[x^2, {x, 0, Infinity}]").unwrap(),
      "Integrate[x^2, {x, 0, Infinity}]"
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

  // An `Assumptions -> cond` option (and other option rules) must be treated as
  // an option, not as an extra integration variable. Previously the option was
  // swallowed by the multivariate path, yielding a nested `Integrate[Integrate[
  // …, Assumptions -> …], …]`. The Assumptions option is honoured via the same
  // mechanism as `Assuming[…]`.
  #[test]
  fn definite_integral_with_assumptions_option() {
    assert_eq!(
      interpret("Integrate[x^2, {x, 0, b}, Assumptions -> b > 0]").unwrap(),
      "b^3/3"
    );
    assert_eq!(
      interpret("Integrate[Cos[x], {x, 0, t}, Assumptions -> t > 0]").unwrap(),
      "Sin[t]"
    );
    // An irrelevant assumption is simply stripped — no nested Integrate.
    assert_eq!(
      interpret("Integrate[x^2, {x, 0, 1}, Assumptions -> x > 0]").unwrap(),
      "1/3"
    );
    // A non-Assumptions option is accepted and ignored.
    assert_eq!(
      interpret("Integrate[x, {x, 0, 1}, GenerateConditions -> False]")
        .unwrap(),
      "1/2"
    );
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

  #[test]
  fn multi_variable_definite_integral_polynomial() {
    // ∫_0^1 ∫_0^1 x*y dy dx = 1/4
    assert_eq!(
      interpret("Integrate[x*y, {x, 0, 1}, {y, 0, 1}]").unwrap(),
      "1/4"
    );
  }

  #[test]
  fn multi_variable_matches_nested_integrate() {
    // Integrate[f, {x, a, b}, {y, c, d}] == Integrate[Integrate[f, {y, c, d}], {x, a, b}]
    let multi =
      interpret("Integrate[x^2 + y^2, {x, 0, 1}, {y, 0, 1}]").unwrap();
    let nested =
      interpret("Integrate[Integrate[x^2 + y^2, {y, 0, 1}], {x, 0, 1}]")
        .unwrap();
    assert_eq!(multi, nested);
  }

  #[test]
  fn multi_variable_dependent_bounds() {
    // ∫_{-1}^{1} ∫_{-2}^{x} (x^3 Sin[y] + y^2 Cos[x^2]) dy dx
    let multi =
      interpret("Integrate[x^3 Sin[y] + y^2 Cos[x^2], {x, -1, 1}, {y, -2, x}]")
        .unwrap();
    let nested = interpret(
      "Integrate[Integrate[x^3 Sin[y] + y^2 Cos[x^2], {y, -2, x}], {x, -1, 1}]",
    )
    .unwrap();
    assert_eq!(multi, nested);
  }

  #[test]
  fn multi_variable_simple_polynomial() {
    // ∫_0^2 ∫_0^3 x + y dy dx = 15
    assert_eq!(
      interpret("Integrate[x + y, {x, 0, 2}, {y, 0, 3}]").unwrap(),
      "15"
    );
  }

  #[test]
  fn definite_integral_poly_times_exp_horner_form() {
    // Definite integral of x^2*E^x should produce Horner form for the
    // polynomial factor and correct factor ordering (polynomial before E^var).
    assert_eq!(
      interpret("Integrate[x^2 E^x, {x, a, b}]").unwrap(),
      "-((2 + (-2 + a)*a)*E^a) + (2 + (-2 + b)*b)*E^b"
    );
  }

  #[test]
  fn definite_integral_poly_times_exp_x4() {
    assert_eq!(
      interpret("Integrate[x^4 E^x, {x, a, b}]").unwrap(),
      "-((24 + a*(-24 + a*(12 + (-4 + a)*a)))*E^a) + (24 + b*(-24 + b*(12 + (-4 + b)*b)))*E^b"
    );
  }

  #[test]
  fn definite_integral_poly_times_exp_linear() {
    // Linear case: no Horner form needed (degree 1)
    assert_eq!(
      interpret("Integrate[x E^x, {x, a, b}]").unwrap(),
      "-((-1 + a)*E^a) + (-1 + b)*E^b"
    );
  }

  #[test]
  fn factor_ordering_poly_times_const_power() {
    // Polynomial before E^a when variable 'a' < 'E' alphabetically
    assert_eq!(
      interpret("E^a*(2 - 2*a + a^2)").unwrap(),
      "(2 - 2*a + a^2)*E^a"
    );
    // E^x before polynomial when variable 'x' > 'E'
    assert_eq!(
      interpret("E^x*(2 - 2*x + x^2)").unwrap(),
      "E^x*(2 - 2*x + x^2)"
    );
  }

  // ∫ Sqrt[a - x^2] / Sqrt[a + x^2] over an interval. Uses the continuous
  // ArcSin/ArcSinh antiderivative so the closed form is exact. Verified
  // against `wolframscript -code 'InputForm[Integrate[…]]'`.

  #[test]
  fn definite_sqrt_unit_semicircle() {
    // ∫_{-1}^{1} Sqrt[1 - x^2] dx = Pi/2 (area of the unit semicircle).
    assert_eq!(
      interpret("Integrate[Sqrt[1 - x^2], {x, -1, 1}]").unwrap(),
      "Pi/2"
    );
  }

  #[test]
  fn definite_sqrt_unit_quarter_circle() {
    // ∫_0^1 Sqrt[1 - x^2] dx = Pi/4.
    assert_eq!(
      interpret("Integrate[Sqrt[1 - x^2], {x, 0, 1}]").unwrap(),
      "Pi/4"
    );
  }

  #[test]
  fn definite_sqrt_radius_two_semicircle() {
    // ∫_{-2}^{2} Sqrt[4 - x^2] dx = 2*Pi (semicircle of radius 2).
    assert_eq!(
      interpret("Integrate[Sqrt[4 - x^2], {x, -2, 2}]").unwrap(),
      "2*Pi"
    );
  }

  #[test]
  fn definite_sqrt_radius_two_quarter_circle() {
    // ∫_0^2 Sqrt[4 - x^2] dx = Pi (quarter circle of radius 2).
    assert_eq!(
      interpret("Integrate[Sqrt[4 - x^2], {x, 0, 2}]").unwrap(),
      "Pi"
    );
  }

  #[test]
  fn definite_sqrt_radius_three_quarter_circle() {
    // ∫_0^3 Sqrt[9 - x^2] dx = (9*Pi)/4.
    assert_eq!(
      interpret("Integrate[Sqrt[9 - x^2], {x, 0, 3}]").unwrap(),
      "(9*Pi)/4"
    );
  }

  #[test]
  fn definite_sqrt_partial_interval() {
    // ∫_{-1/2}^{1/2} Sqrt[1 - x^2] dx = Sqrt[3]/4 + Pi/6.
    assert_eq!(
      interpret("Integrate[Sqrt[1 - x^2], {x, -1/2, 1/2}]").unwrap(),
      "Sqrt[3]/4 + Pi/6"
    );
  }

  #[test]
  fn definite_sqrt_non_monic_half_ellipse() {
    // Non-monic radicand: ∫_{-1/2}^{1/2} Sqrt[1 - 4 x^2] dx = Pi/4
    // (substitution u = 2x maps it to a unit quarter circle).
    assert_eq!(
      interpret("Integrate[Sqrt[1 - 4*x^2], {x, -1/2, 1/2}]").unwrap(),
      "Pi/4"
    );
  }

  #[test]
  fn definite_sqrt_out_of_domain_unevaluated() {
    // The radicand 1 - x^2 is negative for |x| > 1, so over [0, 2] the
    // integrand is not real-valued throughout. Rather than emit an
    // analytic-continuation form that diverges from wolframscript, the
    // Sqrt-quadratic rule bails and the integral stays unevaluated.
    assert_eq!(
      interpret("Integrate[Sqrt[1 - x^2], {x, 0, 2}]").unwrap(),
      "Integrate[Sqrt[1 - x^2], {x, 0, 2}]"
    );
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
      "Got: {result}"
    );
  }
}

mod differentiate_arctan2 {
  use super::*;

  // Two-argument arctangent: d/dx ArcTan[u, v] = u v'/(u^2+v^2) - v u'/(u^2+v^2).
  #[test]
  fn d_arctan2_first_arg() {
    assert_eq!(interpret("D[ArcTan[x, y], x]").unwrap(), "-(y/(x^2 + y^2))");
  }

  #[test]
  fn d_arctan2_second_arg() {
    assert_eq!(interpret("D[ArcTan[x, y], y]").unwrap(), "x/(x^2 + y^2)");
  }

  #[test]
  fn d_arctan2_unrelated_var() {
    assert_eq!(interpret("D[ArcTan[x, y], z]").unwrap(), "0");
  }

  #[test]
  fn d_arctan2_scaled_first() {
    assert_eq!(
      interpret("D[ArcTan[2 x, y], x]").unwrap(),
      "(-2*y)/(4*x^2 + y^2)"
    );
  }

  #[test]
  fn d_arctan2_squared_args() {
    assert_eq!(
      interpret("D[ArcTan[x^2, y^2], y]").unwrap(),
      "(2*x^2*y)/(x^4 + y^4)"
    );
  }

  #[test]
  fn d_arctan2_const_first() {
    assert_eq!(interpret("D[ArcTan[a, t], t]").unwrap(), "a/(a^2 + t^2)");
  }

  #[test]
  fn d_arctan2_literal_const_first() {
    assert_eq!(interpret("D[ArcTan[1, y], y]").unwrap(), "(1 + y^2)^(-1)");
  }

  // Both arguments depend on the variable (unit-circle parametrization): the
  // chain rule keeps the sum of partials, matching Wolfram's unsimplified form.
  #[test]
  fn d_arctan2_unit_circle() {
    assert_eq!(
      interpret("D[ArcTan[Cos[t], Sin[t]], t]").unwrap(),
      "Cos[t]^2/(Cos[t]^2 + Sin[t]^2) + Sin[t]^2/(Cos[t]^2 + Sin[t]^2)"
    );
  }

  #[test]
  fn d_arctan2_equal_args_cancels() {
    assert_eq!(interpret("D[ArcTan[x, x], x]").unwrap(), "0");
  }

  #[test]
  fn d_arctan2_second_derivative() {
    assert_eq!(
      interpret("D[ArcTan[x, y], {x, 2}]").unwrap(),
      "(2*x*y)/(x^2 + y^2)^2"
    );
  }

  #[test]
  fn d_arctan2_product_rule() {
    assert_eq!(
      interpret("D[x ArcTan[x, y], x]").unwrap(),
      "-((x*y)/(x^2 + y^2)) + ArcTan[x, y]"
    );
  }

  // Single-argument ArcTan derivative is unchanged.
  #[test]
  fn d_arctan_single_arg_unchanged() {
    assert_eq!(interpret("D[ArcTan[x], x]").unwrap(), "(1 + x^2)^(-1)");
  }
}

mod differentiate_plus_times {
  use super::*;

  #[test]
  fn d_x_plus_one() {
    assert_eq!(interpret("D[x + 1, x]").unwrap(), "1");
  }

  // The quotient-rule sum orders the squared-denominator term first
  // (shared denominator base, ascending exponent), and D does NOT
  // canonicalize the quotient signs — both wolframscript-verified
  // (found by the differential fuzzer).
  #[test]
  fn d_rational_quotient_term_order_and_sign() {
    assert_eq!(
      interpret("D[(4 - x - 5 x^2)/(3 - 2 x - x^2 - 2 x^3), x]").unwrap(),
      "-(((-2 - 2*x - 6*x^2)*(4 - x - 5*x^2))/(3 - 2*x - x^2 - 2*x^3)^2) + \
       (-1 - 10*x)/(3 - 2*x - x^2 - 2*x^3)"
    );
    assert_eq!(interpret("D[ArcCoth[x^2], x]").unwrap(), "(2*x)/(1 - x^4)");
  }

  // Shared sum-denominator terms whose single-sum numerators contain the
  // base variable compare by NUMERATOR polynomial order, not by exponent:
  // the u'/v term of the quotient rule leads when its numerator's top
  // monomial is smaller. A bare power (numerator-less) term instead
  // orders by ascending exponent against a product term. All
  // wolframscript-verified (differential fuzzer, seed 1783515124284605000).
  #[test]
  fn d_quotient_rule_numerator_term_order() {
    assert_eq!(
      interpret("D[(5 + 2 x + 2 x^2 - 5 x^3)/(3 + x), x]").unwrap(),
      "(2 + 4*x - 15*x^2)/(3 + x) - (5 + 2*x + 2*x^2 - 5*x^3)/(3 + x)^2"
    );
    assert_eq!(
      interpret("D[x/(1 + x), x]").unwrap(),
      "-(x/(1 + x)^2) + (1 + x)^(-1)"
    );
  }

  // A numeric-only term against a sharing product follows the
  // numerator-vs-base rule: the numeric side leads when the sharing
  // numerator sorts above the base ((1+4x) > (5-4x) by top-coefficient
  // order), ascending exponent otherwise. And D's keep-sign Cancel must
  // not re-factor an uncancellable quotient (2/(1-2x)^2 stays, while
  // Cancel itself canonicalizes to 2/(-1+2x)^2). All wolframscript-
  // verified (differential fuzzer, seed 1783515124284605000 re-run).
  #[test]
  fn d_quotient_negative_leading_base() {
    assert_eq!(
      interpret("D[(1 + 4 x)/(5 - 4 x), x]").unwrap(),
      "4/(5 - 4*x) + (4*(1 + 4*x))/(5 - 4*x)^2"
    );
    assert_eq!(interpret("D[1/(1 - 2 x), x]").unwrap(), "2/(1 - 2*x)^2");
    assert_eq!(interpret("Cancel[2/(1-2 x)^2]").unwrap(), "2/(-1 + 2*x)^2");
    assert_eq!(
      interpret("Plus[Power[Plus[5, Times[-4, x]], -1], Times[Plus[1, Times[4, x]], Power[Plus[5, Times[-4, x]], -2]]]").unwrap(),
      "(5 - 4*x)^(-1) + (1 + 4*x)/(5 - 4*x)^2"
    );
    assert_eq!(
      interpret("Plus[Times[4, Power[Plus[3, x], -1]], Times[Plus[2, x], Power[Plus[3, x], -2]]]").unwrap(),
      "(2 + x)/(3 + x)^2 + 4/(3 + x)"
    );
    assert_eq!(
      interpret("Plus[Power[Plus[1, Times[-2, x]], -1], Times[x, Power[Plus[1, Times[-2, x]], -2]]]").unwrap(),
      "(1 - 2*x)^(-1) + x/(1 - 2*x)^2"
    );
  }

  // Direct Plus canonicalization probes for the shared-denominator rules
  // (both input orders give the same canonical form in wolframscript).
  #[test]
  fn shared_denominator_plus_term_order() {
    // Sharing numerators: numerator polynomial order decides.
    assert_eq!(
      interpret("Plus[Times[Plus[2, x], Power[Plus[3, x], -1]], Times[Plus[5, x], Power[Plus[3, x], -2]]]").unwrap(),
      "(2 + x)/(3 + x) + (5 + x)/(3 + x)^2"
    );
    assert_eq!(
      interpret("Plus[Times[Plus[5, x], Power[Plus[3, x], -1]], Times[Plus[2, x], Power[Plus[3, x], -2]]]").unwrap(),
      "(2 + x)/(3 + x)^2 + (5 + x)/(3 + x)"
    );
    // Equal monomials tie-break by ascending coefficient.
    assert_eq!(
      interpret("Plus[Times[Plus[2, Times[4, x]], Power[Plus[3, x], -1]], Times[Plus[5, Times[2, x]], Power[Plus[3, x], -2]]]").unwrap(),
      "(5 + 2*x)/(3 + x)^2 + (2 + 4*x)/(3 + x)"
    );
    // Sharing beats non-sharing regardless of exponent.
    assert_eq!(
      interpret("Plus[Times[Plus[2, x], Power[Plus[3, x], -1]], Times[Plus[5, y], Power[Plus[3, x], -2]]]").unwrap(),
      "(2 + x)/(3 + x) + (5 + y)/(3 + x)^2"
    );
    assert_eq!(
      interpret("Plus[Times[Plus[a, b], Power[Plus[1, x], -1]], Times[x, Power[Plus[1, x], -2]]]").unwrap(),
      "x/(1 + x)^2 + (a + b)/(1 + x)"
    );
    // Neither shares: ascending exponent.
    assert_eq!(
      interpret("Plus[Times[Plus[a, b], Power[Plus[1, x], -1]], Times[Plus[c, d], Power[Plus[1, x], -2]]]").unwrap(),
      "(c + d)/(1 + x)^2 + (a + b)/(1 + x)"
    );
    // Cross-shape (bare power vs product): ascending exponent, both
    // directions.
    assert_eq!(
      interpret(
        "Plus[Power[Plus[3, x], -1], Times[Plus[2, x], Power[Plus[3, x], -2]]]"
      )
      .unwrap(),
      "(2 + x)/(3 + x)^2 + (3 + x)^(-1)"
    );
    assert_eq!(
      interpret(
        "Plus[Power[Plus[3, x], -2], Times[Plus[2, x], Power[Plus[3, x], -1]]]"
      )
      .unwrap(),
      "(3 + x)^(-2) + (2 + x)/(3 + x)"
    );
  }

  #[test]
  fn high_order_derivative_of_x_x_does_not_panic() {
    // Previously D[x^x, {x, 10}] aborted with a "comparison function does
    // not correctly implement a total order" panic during Plus sorting.
    // Now it produces a (long) symbolic result without panicking.
    let result = interpret("D[x^x, {x, 10}]").unwrap();
    // Result is a non-empty symbolic expression containing the variable.
    assert!(
      result.contains('x') && !result.is_empty(),
      "expected non-trivial derivative, got {} chars",
      result.len()
    );
  }

  #[test]
  fn high_order_derivative_of_x_x_order_8() {
    let result = interpret("D[x^x, {x, 8}]").unwrap();
    assert!(result.contains('x'));
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
  fn d_x_to_the_x() {
    // Logarithmic differentiation: d/dx[x^x] = x^x*(1 + Log[x])
    assert_eq!(interpret("D[x^x, x]").unwrap(), "x^x*(1 + Log[x])");
  }

  #[test]
  fn d_general_power_f_to_g() {
    // d/dx[x^(2*x)] where both base and exponent depend on x
    assert_eq!(
      interpret("D[x^(2*x), x]").unwrap(),
      "x^(2*x)*(2 + 2*Log[x])"
    );
  }

  // Regression: D builds the Cos derivative as a UnaryOp minus (-Sin[t]).
  // Raising that to an integer power must distribute the sign correctly:
  // (-Sin[t])^2 = Sin[t]^2, not -Sin[t]^2.
  #[test]
  fn power_of_cos_derivative_signs_correctly() {
    assert_eq!(interpret("D[Cos[t], t]^2").unwrap(), "Sin[t]^2");
    assert_eq!(interpret("D[Cos[t], t]^3").unwrap(), "-Sin[t]^3");
    assert_eq!(interpret("Power[D[Cos[t], t], 4]").unwrap(), "Sin[t]^4");
    // The Pythagorean speed of a unit circle: Cos^2 + Sin^2 (then 1).
    assert_eq!(
      interpret("D[Cos[x], x]^2 + D[Sin[x], x]^2").unwrap(),
      "Cos[x]^2 + Sin[x]^2"
    );
  }

  #[test]
  fn d_times_three_factors() {
    // D[4*(3 + 2*x)*x, x] should work with 3-factor Times
    assert_eq!(
      interpret("D[4*(3 + 2*x)*x, x]").unwrap(),
      "8*x + 4*(3 + 2*x)"
    );
  }

  #[test]
  fn d_gradient_form() {
    // D[expr, {{x, y}}] — gradient form differentiates wrt each variable.
    // Matches wolframscript.
    assert_eq!(
      interpret("D[x^3 * Cos[y], {{x, y}}]").unwrap(),
      "{3*x^2*Cos[y], -(x^3*Sin[y])}"
    );
  }

  // D[expr, var] requires var to be a symbol. A compound term like `2x`
  // (Times with a numeric coefficient) is not a valid variable — Wolfram
  // emits `D::ivar` and returns the call unevaluated.
  #[test]
  fn derivative_wrt_numeric_times_is_unevaluated() {
    assert_eq!(interpret("D[2x, 2x]").unwrap(), "D[2*x, 2*x]");
  }

  // A bare number in the variable slot is not a valid variable: emit D::ivar
  // and stay unevaluated, rather than treating the number as an absent
  // variable and returning 0.
  #[test]
  fn derivative_wrt_number_emits_ivar() {
    for (input, call, bad) in [
      ("D[x^2, 3]", "D[x^2, 3]", "3"),
      ("D[x^2 + y, 2]", "D[x^2 + y, 2]", "2"),
    ] {
      clear_state();
      assert_eq!(interpret(input).unwrap(), call, "for {input}");
      let expected = format!("D::ivar: {bad} is not a valid variable.");
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().any(|m| m.contains(&expected)),
        "expected {expected:?} for {input}, got {msgs:?}"
      );
    }
  }

  // A sum or a product of symbols is likewise not a valid variable — the old
  // behaviour returned 0 (sum) or 1 (self-equal product).
  #[test]
  fn derivative_wrt_compound_emits_ivar() {
    clear_state();
    assert_eq!(interpret("D[x^2, x + 1]").unwrap(), "D[x^2, 1 + x]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs
        .iter()
        .any(|m| m.contains("D::ivar: 1 + x is not a valid variable.")),
      "expected D::ivar for 1 + x, got {msgs:?}"
    );

    clear_state();
    assert_eq!(interpret("D[a b, a b]").unwrap(), "D[a*b, a*b]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs
        .iter()
        .any(|m| m.contains("D::ivar: a b is not a valid variable.")),
      "expected D::ivar for a b, got {msgs:?}"
    );
  }

  // A power or fraction variable also stays unevaluated (the message renders
  // the 2D form in wolframscript, so only the result is asserted here).
  #[test]
  fn derivative_wrt_power_or_fraction_unevaluated() {
    assert_eq!(interpret("D[y, x^2]").unwrap(), "D[y, x^2]");
    assert_eq!(interpret("D[y, x/2]").unwrap(), "D[y, x/2]");
  }

  // A symbol-headed variable (an indexed symbol or a function application)
  // remains valid and is unaffected.
  #[test]
  fn derivative_wrt_symbol_headed_var_unaffected() {
    assert_eq!(interpret("D[f[x[i]], x[k]]").unwrap(), "0");
    assert_eq!(interpret("D[y, Sin[x]]").unwrap(), "0");
  }
}

mod differentiate_piecewise {
  use super::*;

  // D[Piecewise[...]] differentiates each piece value, keeps the condition, and
  // sets the default to Indeterminate (the derivative is undefined at the piece
  // boundaries). Regression: this used to emit a Derivative[1, 0][Piecewise]
  // mess from the generic chain rule.
  #[test]
  fn multi_piece_covering() {
    assert_eq!(
      interpret("D[Piecewise[{{x^2, x < 0}, {x, x > 0}}], x]").unwrap(),
      "Piecewise[{{2*x, x < 0}, {1, x > 0}}, Indeterminate]"
    );
  }

  #[test]
  fn explicit_default_ignored() {
    assert_eq!(
      interpret("D[Piecewise[{{x^2, x < 0}, {Sin[x], x > 0}}, x^3], x]")
        .unwrap(),
      "Piecewise[{{2*x, x < 0}, {Cos[x], x > 0}}, Indeterminate]"
    );
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("D[Piecewise[{{a x^2, x < 0}, {b x, x > 0}}], x]").unwrap(),
      "Piecewise[{{2*a*x, x < 0}, {b, x > 0}}, Indeterminate]"
    );
  }

  // Inclusive boundaries (<=, >=) tighten to strict (<, >) since the boundary
  // point falls to the Indeterminate default.
  #[test]
  fn inclusive_boundary_tightened() {
    assert_eq!(
      interpret("D[Piecewise[{{Cos[x], x < 0}, {x^3, x >= 0}}], x]").unwrap(),
      "Piecewise[{{-Sin[x], x < 0}, {3*x^2, x > 0}}, Indeterminate]"
    );
    assert_eq!(
      interpret("D[Piecewise[{{x^2, x <= 0}, {x, x > 0}}], x]").unwrap(),
      "Piecewise[{{2*x, x < 0}, {1, x > 0}}, Indeterminate]"
    );
  }

  #[test]
  fn second_derivative() {
    assert_eq!(
      interpret("D[Piecewise[{{x^2, x < 0}, {x, x > 0}}], {x, 2}]").unwrap(),
      "Piecewise[{{2, x < 0}, {0, x > 0}}, Indeterminate]"
    );
  }
}

mod differentiate_nonconstants {
  use super::*;

  // A trailing `NonConstants -> {…}` option must not be consumed as an extra
  // differentiation variable. When the option's variables do not occur in the
  // expression the derivative is the plain one.
  #[test]
  fn option_not_treated_as_variable() {
    assert_eq!(interpret("D[x^2, x, NonConstants -> {a}]").unwrap(), "2*x");
    assert_eq!(
      interpret("D[x^3, x, NonConstants -> {a, b}]").unwrap(),
      "3*x^2"
    );
    assert_eq!(
      interpret("D[Sin[x], x, NonConstants -> {a}]").unwrap(),
      "Cos[x]"
    );
    // The {var, n} spec still works alongside the option.
    assert_eq!(
      interpret("D[x^2, {x, 2}, NonConstants -> {a}]").unwrap(),
      "2"
    );
    // Another symbol in the expression is still treated as constant.
    assert_eq!(
      interpret("D[x^2 y, x, NonConstants -> {a}]").unwrap(),
      "2*x*y"
    );
  }

  // NonConstants is D's only option; any other option is unknown, so the call
  // is left unevaluated (wolframscript emits D::optx).
  #[test]
  fn unknown_option_stays_unevaluated() {
    assert_eq!(
      interpret("D[x^2, x, Assumptions -> x > 0]").unwrap(),
      "D[x^2, x, Assumptions -> x > 0]"
    );
  }

  // When a NonConstants variable actually occurs, the derivative carries
  // symbolic `D[a, x, NonConstants -> {…}]` terms via the product rule.
  #[test]
  fn nonconstant_variable_present_carries_derivative() {
    assert_eq!(
      interpret("D[a x^2, x, NonConstants -> {a}]").unwrap(),
      "2*a*x + x^2*D[a, x, NonConstants -> {a}]"
    );
    // The bare NonConstants symbol differentiates to its carried derivative.
    assert_eq!(
      interpret("D[a, x, NonConstants -> {a}]").unwrap(),
      "D[a, x, NonConstants -> {a}]"
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

  #[test]
  fn derivative_indexed_call_undefined() {
    // `h[10]'[t]` differentiates the indexed function `h[10]` and applies
    // the result to `t`: `Derivative[1][h[10]][t]`, not a bare `Derivative`
    // wrapper around the whole call (which would drop the `[t]` argument).
    assert_eq!(interpret("h[10]'[t]").unwrap(), "Derivative[1][h[10]][t]");
  }

  #[test]
  fn derivative_indexed_call_as_implicit_times_factor() {
    // Regression: `h[10]'[t]` used to fail to parse when it wasn't the
    // first factor of an implicit-multiplication chain, because the
    // `FunctionCall` grammar rule allowed only a single trailing prime with
    // no further bracket calls after it — so the second `[t]` was left
    // dangling. Real-world Wolfram Demonstrations write coefficients like
    // `10 h[10]'[t]` on the left of an indexed derivative call routinely.
    assert_eq!(
      interpret("2 h[10]'[t]").unwrap(),
      "2*Derivative[1][h[10]][t]"
    );
    assert_eq!(
      interpret("x h[10]'[t]").unwrap(),
      "x*Derivative[1][h[10]][t]"
    );
  }

  // `Derivative[n][List]` and `Derivative[n1, ..., nk][List]` — Wolfram
  // treats `List` as a varargs identity function, so its derivative is
  // a Function returning a list of zeros and ones at the differentiated
  // positions. The chain `D[List[var]] = {1}`, `D[{1}] = {0}`, etc.
  #[test]
  fn derivative_on_list_first_order() {
    assert_eq!(interpret("Derivative[1][List]").unwrap(), "{1} & ");
  }

  #[test]
  fn derivative_on_list_higher_order_vanishes() {
    // Each list slot is linear in its argument, so any second or
    // higher derivative is the zero list of the same length.
    assert_eq!(interpret("Derivative[2][List]").unwrap(), "{0} & ");
    assert_eq!(interpret("Derivative[3][List]").unwrap(), "{0} & ");
  }

  #[test]
  fn derivative_multi_index_on_list() {
    // `Derivative[0, 0, 1][List]` differentiates the third slot —
    // result has a 1 at position 3, zeros elsewhere.
    assert_eq!(
      interpret("Derivative[0, 0, 1][List]").unwrap(),
      "{0, 0, 1} & "
    );
    assert_eq!(
      interpret("Derivative[1, 0, 0][List]").unwrap(),
      "{1, 0, 0} & "
    );
    // Mixed first-order indices collapse to zeros (each slot is linear
    // in only its own variable).
    assert_eq!(interpret("Derivative[1, 1][List]").unwrap(), "{0, 0} & ");
  }

  #[test]
  fn derivative_prime_on_paren_anonymous_function() {
    // `(#^4&)'` — derivative on a parenthesized anonymous function.
    // Previously rejected by the parser; ParenExtended now accepts an
    // optional DerivativePrime suffix.
    assert_eq!(interpret("(#^4&)'").unwrap(), "4*#1^3 & ");
  }

  #[test]
  fn derivative_double_prime_on_paren_anonymous_function() {
    // Wolframscript prints the chain unsimplified rather than `12*#1^2 & `.
    assert_eq!(interpret("(#^4&)''").unwrap(), "4*(3*#1^2) & ");
  }

  #[test]
  fn derivative_prime_on_paren_then_apply() {
    // `(#^3&)'[x]` — the prime suffix can still be followed by [args].
    assert_eq!(interpret("(#^3&)'[x]").unwrap(), "3*x^2");
  }

  #[test]
  fn derivative_symbolic_via_inputform() {
    // `f'[x] // InputForm` keeps the symbolic Derivative wrapped in
    // unevaluated InputForm (matches wolframscript).
    assert_eq!(
      interpret("f'[x] // InputForm").unwrap(),
      "InputForm[Derivative[1][f][x]]"
    );
  }

  #[test]
  fn derivative_multi_index_inputform() {
    // InputForm[Derivative[1, 0][f][x]] stays wrapped (matches wolframscript).
    assert_eq!(
      interpret("InputForm[Derivative[1, 0][f][x]]").unwrap(),
      "InputForm[Derivative[1, 0][f][x]]"
    );
  }

  #[test]
  fn plus_function_call_inputform_postfix() {
    // `2 + F[x] // InputForm` — InputForm wraps the Plus expression with
    // its canonical 2 first.
    assert_eq!(
      interpret("2+F[x] // InputForm").unwrap(),
      "InputForm[2 + F[x]]"
    );
  }

  #[test]
  fn derivative_multi_index_symbolic() {
    // Derivative[2, 1][h] — mixed partial derivatives of unknown h
    // stays symbolic in curried form.
    assert_eq!(
      interpret("Derivative[2, 1][h]").unwrap(),
      "Derivative[2, 1][h]"
    );
  }

  #[test]
  fn derivative_multi_index_applied_symbolic() {
    // Derivative[2, 0, 1, 0][h[g]] — applied multi-index derivative stays
    // symbolic in curried form.
    assert_eq!(
      interpret("Derivative[2, 0, 1, 0][h[g]]").unwrap(),
      "Derivative[2, 0, 1, 0][h[g]]"
    );
  }

  #[test]
  fn derivative_builtin_sin_prime() {
    assert_eq!(interpret("Sin'[x]").unwrap(), "Cos[x]");
  }

  #[test]
  fn derivative_builtin_cos_prime() {
    assert_eq!(interpret("Cos'[x]").unwrap(), "-Sin[x]");
  }

  #[test]
  fn derivative_builtin_tan_prime() {
    assert_eq!(interpret("Tan'[x]").unwrap(), "Sec[x]^2");
  }

  #[test]
  fn derivative_builtin_exp_prime() {
    assert_eq!(interpret("Exp'[x]").unwrap(), "E^x");
  }

  #[test]
  fn derivative_builtin_log_prime() {
    assert_eq!(interpret("Log'[x]").unwrap(), "x^(-1)");
  }

  #[test]
  fn derivative_builtin_sin_double_prime() {
    assert_eq!(interpret("Sin''[x]").unwrap(), "-Sin[x]");
  }

  #[test]
  fn derivative_builtin_sin_prime_at_zero() {
    assert_eq!(interpret("Sin'[0]").unwrap(), "1");
  }

  #[test]
  fn derivative_builtin_cos_prime_at_pi() {
    assert_eq!(interpret("Cos'[Pi]").unwrap(), "0");
  }

  #[test]
  fn derivative_product_sin_cos() {
    assert_eq!(
      interpret("D[Sin[x]*Cos[x], x]").unwrap(),
      "Cos[x]^2 - Sin[x]^2"
    );
  }

  #[test]
  fn derivative_product_x_squared_cos() {
    assert_eq!(
      interpret("D[x^2*Cos[x], x]").unwrap(),
      "2*x*Cos[x] - x^2*Sin[x]"
    );
  }

  #[test]
  fn derivative_product_exp_sin() {
    assert_eq!(
      interpret("D[Exp[x]*Sin[x], x]").unwrap(),
      "E^x*Cos[x] + E^x*Sin[x]"
    );
  }

  // Derivative[n][f] returning pure functions
  #[test]
  fn derivative_n_sin() {
    assert_eq!(interpret("Derivative[1][Sin]").unwrap(), "Cos[#1] & ");
  }

  #[test]
  fn derivative_n_cos() {
    assert_eq!(interpret("Derivative[1][Cos]").unwrap(), "-Sin[#1] & ");
  }

  #[test]
  fn derivative_n_exp() {
    assert_eq!(interpret("Derivative[1][Exp]").unwrap(), "E^#1 & ");
  }

  #[test]
  fn derivative_n_log() {
    assert_eq!(interpret("Derivative[1][Log]").unwrap(), "#1^(-1) & ");
  }

  #[test]
  fn derivative_n_sin_second() {
    assert_eq!(interpret("Derivative[2][Sin]").unwrap(), "-Sin[#1] & ");
  }

  #[test]
  fn derivative_n_sin_third() {
    assert_eq!(interpret("Derivative[3][Sin]").unwrap(), "-Cos[#1] & ");
  }

  #[test]
  fn derivative_n_sin_fourth() {
    assert_eq!(interpret("Derivative[4][Sin]").unwrap(), "Sin[#1] & ");
  }

  #[test]
  fn derivative_n_pure_function_cubic() {
    assert_eq!(interpret("Derivative[1][#^3&]").unwrap(), "3*#1^2 & ");
  }

  #[test]
  fn derivative_n_pure_function_cubic_second() {
    // Wolframscript prints `3*(2*#1) & ` instead of the folded `6*#1 & `.
    assert_eq!(interpret("Derivative[2][#^3&]").unwrap(), "3*(2*#1) & ");
  }

  #[test]
  fn derivative_n_pure_function_cubic_third() {
    // Chain runs all the way down to the residual `1` factor.
    assert_eq!(interpret("Derivative[3][#^3&]").unwrap(), "3*(2*1) & ");
  }

  #[test]
  fn derivative_n_applied() {
    // Derivative[1][Sin][x] should evaluate like Sin'[x]
    assert_eq!(interpret("Derivative[1][Sin][x]").unwrap(), "Cos[x]");
  }

  #[test]
  fn derivative_n_applied_numeric() {
    assert_eq!(interpret("Derivative[1][Sin][0]").unwrap(), "1");
  }

  #[test]
  fn derivative_n_undefined_function() {
    // Derivative of undefined function stays symbolic
    assert_eq!(interpret("Derivative[1][g]").unwrap(), "Derivative[1][g]");
  }

  // Derivative[0, 0, ..., 0][f][x, ...] is the identity — applying it to an
  // expression returns the expression unchanged.
  #[test]
  fn derivative_all_zero_is_identity() {
    assert_eq!(interpret("Derivative[0,0,0][a+b+c]").unwrap(), "a + b + c");
  }

  #[test]
  fn derivative_zero_on_symbol_returns_symbol() {
    assert_eq!(interpret("Derivative[0, 0][f]").unwrap(), "f");
  }

  // A non-zero component is still preserved — only all-zero vectors collapse.
  #[test]
  fn derivative_with_nonzero_stays_symbolic() {
    assert_eq!(
      interpret("Derivative[1, 0][f]").unwrap(),
      "Derivative[1, 0][f]"
    );
  }

  #[test]
  fn output_form_renders_derivative_as_prime() {
    // OutputForm[f'[x]] should render as `f'[x]` (prime notation), matching
    // wolframscript. Default output keeps Derivative[1][f][x].
    assert_eq!(interpret("ToString[OutputForm[f'[x]]]").unwrap(), "f'[x]");
  }

  #[test]
  fn output_form_renders_higher_derivative_as_primes() {
    assert_eq!(
      interpret("ToString[OutputForm[Derivative[3][g][y]]]").unwrap(),
      "g'''[y]"
    );
  }

  #[test]
  fn output_form_renders_fourth_derivative_as_superscript() {
    // n >= 4 uses f^(n)[args] notation.
    assert_eq!(
      interpret("ToString[OutputForm[Derivative[4][f][x]]]").unwrap(),
      "f^(4)[x]"
    );
  }

  #[test]
  fn output_form_unapplied_derivative() {
    // Derivative[1][f] without arguments renders as f'.
    assert_eq!(
      interpret("ToString[OutputForm[Derivative[1][f]]]").unwrap(),
      "f'"
    );
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

  // A sum containing a term with a pole at x0 (a Laurent series) is expanded
  // by linearity — each summand on its own, then added — so it no longer
  // chokes on the pole. (Series of a single `1/x` already worked.)
  #[test]
  fn series_laurent_sum() {
    assert_eq!(
      interpret("Series[1/x + 1 + x, {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, 1, 1}, -1, 3, 1]"
    );
    assert_eq!(
      interpret("Series[1/x^2 + x, {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 0, 1}, -2, 4, 1]"
    );
    // Mixing a transcendental analytic part with a pole part.
    assert_eq!(
      interpret("Series[Exp[x] + 1/x, {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1, 1, 1, 1/2, 1/6}, -1, 4, 1]"
    );
    // Subtraction (a - b) is handled too.
    assert_eq!(
      interpret("Series[1/x - 3 + 2 x, {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1, -3, 2}, -1, 4, 1]"
    );
    // Nonzero expansion center.
    assert_eq!(
      interpret("Series[1/(x - 1) + x, {x, 2, 2}]").unwrap(),
      "SeriesData[x, 2, {3, 0, 1}, 0, 3, 1]"
    );
  }

  // Gamma[x] has a simple pole at x = 0. The direct coefficient path samples
  // Gamma[0] = ComplexInfinity and used to return a bogus all-ComplexInfinity
  // series; expand via Gamma[x] = x!/x instead (the analytic factorial series
  // one order higher, shifted down one integer power). Orders 0 and 1 match
  // wolframscript's SeriesData exactly. (Order >= 2 is value-correct but the
  // interior coefficients inherit the factorial series' form divergence, e.g.
  // (EulerGamma^2 + Pi^2/6)/2 vs WL's (6*EulerGamma^2 + Pi^2)/12.)
  #[test]
  fn series_gamma_pole_at_zero() {
    assert_eq!(
      interpret("Series[Gamma[x], {x, 0, 0}]").unwrap(),
      "SeriesData[x, 0, {1, -EulerGamma}, -1, 1, 1]"
    );
    assert_eq!(
      interpret("Series[Gamma[x], {x, 0, 1}]").unwrap(),
      "SeriesData[x, 0, {1, -EulerGamma, (6*EulerGamma^2 + Pi^2)/12}, -1, 2, 1]"
    );
    // Works with any expansion variable, not just x.
    assert_eq!(
      interpret("Series[Gamma[y], {y, 0, 0}]").unwrap(),
      "SeriesData[y, 0, {1, -EulerGamma}, -1, 1, 1]"
    );
  }

  // Zeta[x] has a simple pole at x = 1: Laurent series 1/(x-1) + Sum_{n>=0}
  // (-1)^n StieltjesGamma[n]/n! (x-1)^n, with StieltjesGamma[0] = EulerGamma.
  #[test]
  fn series_zeta_pole_at_one() {
    assert_eq!(
      interpret("Series[Zeta[x], {x, 1, 0}]").unwrap(),
      "SeriesData[x, 1, {1, EulerGamma}, -1, 1, 1]"
    );
    assert_eq!(
      interpret("Series[Zeta[x], {x, 1, 1}]").unwrap(),
      "SeriesData[x, 1, {1, EulerGamma, -StieltjesGamma[1]}, -1, 2, 1]"
    );
    assert_eq!(
      interpret("Series[Zeta[x], {x, 1, 2}]").unwrap(),
      "SeriesData[x, 1, {1, EulerGamma, -StieltjesGamma[1], StieltjesGamma[2]/2}, -1, 3, 1]"
    );
    assert_eq!(
      interpret("Series[Zeta[x], {x, 1, 3}]").unwrap(),
      "SeriesData[x, 1, {1, EulerGamma, -StieltjesGamma[1], StieltjesGamma[2]/2, -1/6*StieltjesGamma[3]}, -1, 4, 1]"
    );
    // Any expansion variable.
    assert_eq!(
      interpret("Series[Zeta[s], {s, 1, 0}]").unwrap(),
      "SeriesData[s, 1, {1, EulerGamma}, -1, 1, 1]"
    );
  }

  // Puiseux (fractional-power) expansion: f = x^(p/q) g(x) expands with den = q,
  // the cofactor g's coefficients interleaved with q-1 zeros. nmax follows the
  // rule max(order*q, nmin) + 1.
  #[test]
  fn series_fractional_power() {
    assert_eq!(
      interpret("Series[Sqrt[x], {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1}, 1, 7, 2]"
    );
    assert_eq!(
      interpret("Series[x^(3/2), {x, 0, 4}]").unwrap(),
      "SeriesData[x, 0, {1}, 3, 9, 2]"
    );
    // Cube-root power gives den = 3.
    assert_eq!(
      interpret("Series[x^(1/3), {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1}, 1, 7, 3]"
    );
    // Analytic cofactor (Exp) interleaves with a zero at the integer power.
    assert_eq!(
      interpret("Series[Sqrt[x] Exp[x], {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1}, 1, 5, 2]"
    );
    // A Sqrt cofactor is itself analytic at 0.
    assert_eq!(
      interpret("Series[Sqrt[x] Sqrt[1 + x], {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1/2}, 1, 5, 2]"
    );
    // Negative fractional power (leading exponent below zero).
    assert_eq!(
      interpret("Series[1/Sqrt[x], {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1}, -1, 5, 2]"
    );
    // Order 0 still emits the leading fractional term.
    assert_eq!(
      interpret("Series[Sqrt[x], {x, 0, 0}]").unwrap(),
      "SeriesData[x, 0, {1}, 1, 2, 2]"
    );
    // A constant multiple keeps the coefficient.
    assert_eq!(
      interpret("Series[2 Sqrt[x], {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {2}, 1, 5, 2]"
    );
  }

  // A sum containing a fractional-power term is expanded by linearity, so
  // mixed-denominator summands recombine (SeriesData addition handles the
  // common denominator).
  #[test]
  fn series_fractional_sum() {
    assert_eq!(
      interpret("Series[x^(1/2) + x, {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, 1}, 1, 5, 2]"
    );
    assert_eq!(
      interpret("Series[Sqrt[x] + x^2, {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 0, 1}, 1, 5, 2]"
    );
    // Different fractional denominators combine via their lcm (2 and 3 -> 6).
    assert_eq!(
      interpret("Series[x^(1/3) + x^(1/2), {x, 0, 1}]").unwrap(),
      "SeriesData[x, 0, {1, 1}, 2, 8, 6]"
    );
    // An integer constant/term mixes with the fractional part.
    assert_eq!(
      interpret("Series[1 + Sqrt[x] + x, {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, 1, 1}, 0, 5, 2]"
    );
    // A fractional term plus a pole term.
    assert_eq!(
      interpret("Series[Sqrt[x] + 1/x, {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 0, 1}, -2, 5, 2]"
    );
  }

  // Puiseux expansion about a nonzero center: a fractional power of (x - x0).
  // (Sqrt[x] about x0 = 1 is analytic and uses the ordinary path; the genuine
  // fractional case is a power of the shift (x - 1).)
  #[test]
  fn series_fractional_nonzero_center() {
    assert_eq!(
      interpret("Series[Sqrt[x - 1], {x, 1, 3}]").unwrap(),
      "SeriesData[x, 1, {1}, 1, 7, 2]"
    );
    assert_eq!(
      interpret("Series[(x - 2)^(1/2), {x, 2, 2}]").unwrap(),
      "SeriesData[x, 2, {1}, 1, 5, 2]"
    );
    // Product with an analytic cofactor about the same center.
    assert_eq!(
      interpret("Series[Sqrt[x - 1] Exp[x], {x, 1, 2}]").unwrap(),
      "SeriesData[x, 1, {E, 0, E}, 1, 5, 2]"
    );
    // Negative fractional power of the shift.
    assert_eq!(
      interpret("Series[1/Sqrt[x - 1], {x, 1, 2}]").unwrap(),
      "SeriesData[x, 1, {1}, -1, 5, 2]"
    );
    // A fractional sum about a nonzero center.
    assert_eq!(
      interpret("Series[Sqrt[x - 1] + (x - 1)^2, {x, 1, 2}]").unwrap(),
      "SeriesData[x, 1, {1, 0, 0, 1}, 1, 5, 2]"
    );
  }

  // Integrating a fractional-power SeriesData works via the den != 1 path.
  #[test]
  fn integrate_series_fractional() {
    assert_eq!(
      interpret("Integrate[Series[Sqrt[x], {x, 0, 3}], x]").unwrap(),
      "SeriesData[x, 0, {2/3}, 3, 9, 2]"
    );
  }

  // A purely analytic sum is unaffected by the linearity path.
  #[test]
  fn series_analytic_sum_unchanged() {
    assert_eq!(
      interpret("Series[Exp[x] + Sin[x], {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1, 2, 1/2}, 0, 4, 1]"
    );
  }

  // Integrating a SeriesData (w.r.t. the series variable) integrates the
  // truncated power series term-by-term: nmin/nmax rise by den and each
  // coefficient c_k is scaled by den/(nmin+k+den). Matches wolframscript.
  #[test]
  fn integrate_series_exp() {
    assert_eq!(
      interpret("Integrate[Series[Exp[x], {x, 0, 3}], x]").unwrap(),
      "SeriesData[x, 0, {1, 1/2, 1/6, 1/24}, 1, 5, 1]"
    );
  }

  #[test]
  fn integrate_series_geometric() {
    assert_eq!(
      interpret("Integrate[Series[1/(1 - x), {x, 0, 4}], x]").unwrap(),
      "SeriesData[x, 0, {1, 1/2, 1/3, 1/4, 1/5}, 1, 6, 1]"
    );
  }

  #[test]
  fn integrate_series_sin_shifts_nmin() {
    // Sin starts at x^1, so the integrated series starts at x^2.
    assert_eq!(
      interpret("Integrate[Series[Sin[x], {x, 0, 6}], x]").unwrap(),
      "SeriesData[x, 0, {1/2, 0, -1/24, 0, 1/720}, 2, 8, 1]"
    );
  }

  #[test]
  fn integrate_series_nonzero_center() {
    assert_eq!(
      interpret("Integrate[Series[Exp[x], {x, 1, 3}], x]").unwrap(),
      "SeriesData[x, 1, {E, E/2, E/6, E/24}, 1, 5, 1]"
    );
  }

  #[test]
  fn integrate_series_then_normal() {
    assert_eq!(
      interpret("Normal[Integrate[Series[Exp[x], {x, 0, 4}], x]]").unwrap(),
      "x + x^2/2 + x^3/6 + x^4/24 + x^5/120"
    );
  }

  // Integrating a Laurent SeriesData: negative-power terms integrate fine as
  // long as no term has exponent -1 (which would produce a logarithm). The
  // exponent -1 slot is allowed when its coefficient is zero.
  #[test]
  fn integrate_series_laurent() {
    assert_eq!(
      interpret("Integrate[Series[1/x^2 + x, {x, 0, 3}], x]").unwrap(),
      "SeriesData[x, 0, {-1, 0, 0, 1/2}, -1, 5, 1]"
    );
    assert_eq!(
      interpret("Integrate[Series[1/x^3, {x, 0, 2}], x]").unwrap(),
      "SeriesData[x, 0, {-1/2}, -2, 4, 1]"
    );
    // A genuine 1/x term integrates to a logarithm, which has no SeriesData
    // form, so Woxi leaves it unevaluated.
    assert_eq!(
      interpret("Integrate[Series[1/x + x, {x, 0, 3}], x]").unwrap(),
      "Integrate[SeriesData[x, 0, {1, 0, 1}, -1, 4, 1], x]"
    );
  }

  #[test]
  fn series_at_infinity_rational() {
    // Series[f, {x, Infinity, n}] for rational f: substitute x -> 1/t, expand
    // at t = 0, relabel to base Infinity. Verified against wolframscript.
    assert_eq!(
      interpret("Series[1/(1 + x), {x, Infinity, 3}]").unwrap(),
      "SeriesData[x, Infinity, {1, -1, 1}, 1, 4, 1]"
    );
    assert_eq!(
      interpret("Series[(x + 1)/(x - 1), {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1, 2, 2}, 0, 3, 1]"
    );
    assert_eq!(
      interpret("Series[1/x, {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1}, 1, 3, 1]"
    );
    // A function that grows at infinity gives negative-index terms.
    assert_eq!(
      interpret("Series[x^2 + x, {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1, 1}, -2, 3, 1]"
    );
  }

  #[test]
  fn series_at_infinity_normal() {
    // Normal of an Infinity series yields powers of 1/x in canonical order
    // (most-negative exponent first), matching wolframscript.
    assert_eq!(
      interpret("Normal[Series[1/(1 + x), {x, Infinity, 3}]]").unwrap(),
      "x^(-3) - x^(-2) + x^(-1)"
    );
    assert_eq!(
      interpret("Normal[Series[(x + 1)/(x - 1), {x, Infinity, 2}]]").unwrap(),
      "1 + 2/x^2 + 2/x"
    );
  }

  // Regression: Simplify/Expand on a bare SeriesData head with x0 == 0 used
  // to recurse infinitely through `try_series_data_plus` (a single-arg call
  // re-entered the same single-SeriesData lifting branch).
  #[test]
  fn simplify_series_data_no_stack_overflow() {
    assert_eq!(
      interpret("Simplify[SeriesData[x, 0, {a, b, c, d}, 0, 4, 1]]").unwrap(),
      "SeriesData[x, 0, {a, b, c, d}, 0, 4, 1]"
    );
  }

  #[test]
  fn full_simplify_series_data_no_stack_overflow() {
    assert_eq!(
      interpret("FullSimplify[SeriesData[x, 0, {a, b, c, d}, 0, 4, 1]]")
        .unwrap(),
      "SeriesData[x, 0, {a, b, c, d}, 0, 4, 1]"
    );
  }

  #[test]
  fn expand_series_data_no_stack_overflow() {
    assert_eq!(
      interpret("Expand[SeriesData[x, 0, {a, b, c, d}, 0, 4, 1]]").unwrap(),
      "SeriesData[x, 0, {a, b, c, d}, 0, 4, 1]"
    );
  }

  #[test]
  fn factor_series_data_no_stack_overflow() {
    assert_eq!(
      interpret("Factor[SeriesData[x, 0, {a, b, c, d}, 0, 4, 1]]").unwrap(),
      "SeriesData[x, 0, {a, b, c, d}, 0, 4, 1]"
    );
  }

  #[test]
  fn series_exp_two_vars_nested_series_data() {
    assert_eq!(
      interpret("Series[Exp[x-y], {x, 0, 2}, {y, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {SeriesData[y, 0, {1, -1, 1/2}, 0, 3, 1], SeriesData[y, 0, {1, -1, 1/2}, 0, 3, 1], SeriesData[y, 0, {1/2, -1/2, 1/4}, 0, 3, 1]}, 0, 3, 1]"
    );
  }

  #[test]
  fn series_exp_sin_order5() {
    assert_eq!(
      interpret("Series[Exp[Sin[x]], {x, 0, 5}]").unwrap(),
      "SeriesData[x, 0, {1, 1, 1/2, 0, -1/8, -1/15}, 0, 6, 1]"
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

  // Removable-singularity quotients (Sin[x]/x, etc.) expand via power-series
  // long division. The internal probe at x0 hits a 0/0, which must be evaluated
  // quietly so no spurious Power::infy / Infinity::indet messages leak out.
  // wolframscript emits no such message.
  #[test]
  fn series_removable_singularity_no_warnings() {
    clear_state();
    let result = interpret_with_stdout("Series[Sin[x]/x, {x, 0, 4}]").unwrap();
    assert_eq!(
      result.result,
      "SeriesData[x, 0, {1, 0, -1/6, 0, 1/120}, 0, 5, 1]"
    );
    assert!(
      result.warnings.is_empty(),
      "Expected no warnings but got: {:?}",
      result.warnings
    );
  }

  #[test]
  fn series_one_minus_cos_over_x2() {
    clear_state();
    let result =
      interpret_with_stdout("Series[(1 - Cos[x])/x^2, {x, 0, 4}]").unwrap();
    assert_eq!(
      result.result,
      "SeriesData[x, 0, {1/2, 0, -1/24, 0, 1/720}, 0, 5, 1]"
    );
    assert!(
      result.warnings.is_empty(),
      "Expected no warnings but got: {:?}",
      result.warnings
    );
  }

  #[test]
  fn series_exp_minus_one_over_x() {
    clear_state();
    let result =
      interpret_with_stdout("Series[(Exp[x] - 1)/x, {x, 0, 3}]").unwrap();
    assert_eq!(
      result.result,
      "SeriesData[x, 0, {1, 1/2, 1/6, 1/24}, 0, 4, 1]"
    );
    assert!(
      result.warnings.is_empty(),
      "Expected no warnings but got: {:?}",
      result.warnings
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

  // Quotients with a removable singularity at the expansion point: the direct
  // Taylor evaluation hits 0/0, so Series falls back to power-series long
  // division of numerator by denominator.
  #[test]
  fn series_sinc_removable() {
    assert_eq!(
      interpret("Series[Sin[x]/x, {x, 0, 4}]").unwrap(),
      "SeriesData[x, 0, {1, 0, -1/6, 0, 1/120}, 0, 5, 1]"
    );
  }

  #[test]
  fn series_x_over_exp_minus_one() {
    // Bernoulli generating function: 1 - x/2 + x^2/12 - x^4/720.
    assert_eq!(
      interpret("Series[x/(E^x - 1), {x, 0, 4}]").unwrap(),
      "SeriesData[x, 0, {1, -1/2, 1/12, 0, -1/720}, 0, 5, 1]"
    );
  }

  #[test]
  fn series_one_minus_cos_over_x_squared() {
    assert_eq!(
      interpret("Series[(1 - Cos[x])/x^2, {x, 0, 4}]").unwrap(),
      "SeriesData[x, 0, {1/2, 0, -1/24, 0, 1/720}, 0, 5, 1]"
    );
  }

  #[test]
  fn series_x_minus_sin_over_x_cubed() {
    assert_eq!(
      interpret("Series[(x - Sin[x])/x^3, {x, 0, 4}]").unwrap(),
      "SeriesData[x, 0, {1/6, 0, -1/120, 0, 1/5040}, 0, 5, 1]"
    );
  }

  // Genuine poles: the quotient has a negative leading power (Laurent series).
  #[test]
  fn series_reciprocal_sin_pole() {
    assert_eq!(
      interpret("Series[1/Sin[x], {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1/6, 0, 7/360}, -1, 4, 1]"
    );
  }

  #[test]
  fn series_reciprocal_x_squared_pole() {
    assert_eq!(
      interpret("Series[1/x^2, {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1}, -2, 4, 1]"
    );
  }

  // Removable singularity away from 0: expansion about x0 = 1.
  #[test]
  fn series_sinc_shifted_center() {
    assert_eq!(
      interpret("Normal[Series[Sin[x - 1]/(x - 1), {x, 1, 3}]]").unwrap(),
      "1 - (-1 + x)^2/6"
    );
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

  // A top-order cancellation in series addition/multiplication trims the
  // trailing zero coefficient (keeping nmax). Verified against wolframscript.
  #[test]
  fn series_sum_trims_trailing_zero() {
    // Exp + Sin: the x^3 terms (1/6 and -1/6) cancel.
    assert_eq!(
      interpret("Series[Exp[x], {x, 0, 3}] + Series[Sin[x], {x, 0, 3}]")
        .unwrap(),
      "SeriesData[x, 0, {1, 2, 1/2}, 0, 4, 1]"
    );
  }

  #[test]
  fn series_product_trims_trailing_zero() {
    // Cos * Sin: the x^4 coefficient is zero.
    assert_eq!(
      interpret("Series[Cos[x], {x, 0, 4}] * Series[Sin[x], {x, 0, 4}]")
        .unwrap(),
      "SeriesData[x, 0, {1, 0, -2/3}, 1, 5, 1]"
    );
    // Exp[x] * Exp[-x] = 1: all higher coefficients cancel.
    assert_eq!(
      interpret("Series[Exp[x], {x, 0, 4}] * Series[Exp[-x], {x, 0, 4}]")
        .unwrap(),
      "SeriesData[x, 0, {1}, 0, 5, 1]"
    );
  }

  // A directly-constructed SeriesData is normalized: leading zeros advance
  // nmin, trailing zeros are dropped (nmax kept). Verified against wolframscript.
  #[test]
  fn series_data_normalizes_zeros() {
    assert_eq!(
      interpret("SeriesData[x, 0, {1, 2, 0}, 0, 3, 1]").unwrap(),
      "SeriesData[x, 0, {1, 2}, 0, 3, 1]"
    );
    assert_eq!(
      interpret("SeriesData[x, 0, {0, 1, 0}, 1, 3, 1]").unwrap(),
      "SeriesData[x, 0, {1}, 2, 3, 1]"
    );
    assert_eq!(
      interpret("SeriesData[x, 0, {0, 0, 5}, 0, 4, 1]").unwrap(),
      "SeriesData[x, 0, {5}, 2, 4, 1]"
    );
    // Already-normal SeriesData is unchanged.
    assert_eq!(
      interpret("SeriesData[x, 0, {1, 2, 3}, 0, 3, 1]").unwrap(),
      "SeriesData[x, 0, {1, 2, 3}, 0, 3, 1]"
    );
  }

  // SeriesData^n: O[x]^n scales the order; a real series is squared/cubed.
  #[test]
  fn series_data_power() {
    assert_eq!(
      interpret("O[x]^3").unwrap(),
      "SeriesData[x, 0, {}, 3, 3, 1]"
    );
    assert_eq!(
      interpret("Series[Exp[x], {x, 0, 2}]^2").unwrap(),
      "SeriesData[x, 0, {1, 2, 2}, 0, 3, 1]"
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

  #[test]
  fn series_head_is_series_data() {
    // A Series result has head SeriesData, matching wolframscript.
    assert_eq!(
      interpret("series = Series[Cosh[x], {x, 0, 2}]; Head[series]").unwrap(),
      "SeriesData"
    );
  }

  #[test]
  fn series_full_form_structure() {
    // FullForm of Series[Cosh[x], {x, 0, 2}] reveals the underlying
    // SeriesData with coefficient list and order bounds. Wolfram keeps the
    // FullForm wrapper in place and renders the coefficient List with `{}`
    // and any Rational with `n/d` notation, so the output reads
    // `FullForm[SeriesData[x, 0, {1, 0, 1/2}, 0, 3, 1]]`.
    assert_eq!(
      interpret("series = Series[Cosh[x], {x, 0, 2}]; series // FullForm")
        .unwrap(),
      "FullForm[SeriesData[x, 0, {1, 0, 1/2}, 0, 3, 1]]"
    );
  }

  #[test]
  fn series_cosh_raw_output() {
    // Default output of Series uses the raw SeriesData form —
    // matches wolframscript; mathics renders it prettied as
    // 1 + x^2/2 + O[x]^3 instead.
    assert_eq!(
      interpret("Series[Cosh[x], {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1/2}, 0, 3, 1]"
    );
  }

  // Differentiating a SeriesData with respect to its own expansion variable
  // applies the term-by-term power rule (it previously returned 0 because the
  // coefficients — all constants — were differentiated as a parameter).
  #[test]
  fn derivative_of_seriesdata_power_rule() {
    assert_eq!(
      interpret("D[SeriesData[x, 0, {1, 1, 1}, 0, 3, 1], x]").unwrap(),
      "SeriesData[x, 0, {1, 2}, 0, 2, 1]"
    );
  }

  #[test]
  fn derivative_of_exp_series_is_exp_series() {
    assert_eq!(
      interpret("D[Series[Exp[x], {x, 0, 4}], x]").unwrap(),
      "SeriesData[x, 0, {1, 1, 1/2, 1/6}, 0, 4, 1]"
    );
  }

  #[test]
  fn derivative_of_sin_series() {
    assert_eq!(
      interpret("D[Series[Sin[x], {x, 0, 6}], x]").unwrap(),
      "SeriesData[x, 0, {1, 0, -1/2, 0, 1/24}, 0, 6, 1]"
    );
  }

  #[test]
  fn derivative_of_seriesdata_with_offset_nmin() {
    assert_eq!(
      interpret("D[SeriesData[x, 0, {1, 2, 3}, 1, 4, 1], x]").unwrap(),
      "SeriesData[x, 0, {1, 4, 9}, 0, 3, 1]"
    );
  }

  #[test]
  fn derivative_of_shifted_center_series() {
    assert_eq!(
      interpret("D[Series[Sqrt[x], {x, 1, 3}], x]").unwrap(),
      "SeriesData[x, 1, {1/2, -1/4, 3/16}, 0, 3, 1]"
    );
  }

  #[test]
  fn derivative_of_fractional_power_series() {
    // den = 2 (half-integer powers): exponents shift down by one integer
    // power, so nmin and nmax each drop by den.
    assert_eq!(
      interpret("D[SeriesData[x, 0, {1, 0, 1}, 1, 5, 2], x]").unwrap(),
      "SeriesData[x, 0, {1/2, 0, 3/2}, -1, 3, 2]"
    );
  }

  // wolframscript drops trailing zero coefficients from the SeriesData list
  // while keeping the truncation order (nmax).
  #[test]
  fn trailing_zeros_trimmed_linear() {
    assert_eq!(
      interpret("Series[1 + x, {x, 0, 3}]").unwrap(),
      "SeriesData[x, 0, {1, 1}, 0, 4, 1]"
    );
  }

  #[test]
  fn trailing_zeros_trimmed_monomial() {
    assert_eq!(
      interpret("Series[x^2, {x, 0, 5}]").unwrap(),
      "SeriesData[x, 0, {1}, 2, 6, 1]"
    );
  }

  #[test]
  fn trailing_zeros_trimmed_keeps_internal_zeros() {
    assert_eq!(
      interpret("Series[x^2 + x^4, {x, 0, 7}]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1}, 2, 8, 1]"
    );
  }

  // A constant (or any expression free of the expansion variable) has no
  // SeriesData wrapper — the series is the expression itself.
  #[test]
  fn constant_returns_bare_value() {
    assert_eq!(interpret("Series[3, {x, 0, 3}]").unwrap(), "3");
  }

  #[test]
  fn variable_free_symbol_returns_bare() {
    assert_eq!(interpret("Series[a, {x, 0, 3}]").unwrap(), "a");
  }

  #[test]
  fn variable_free_expression_returns_bare() {
    assert_eq!(interpret("Series[a + b, {x, 0, 3}]").unwrap(), "a + b");
  }

  #[test]
  fn series_in_unrelated_variable_returns_bare() {
    assert_eq!(interpret("Series[Sin[y], {x, 0, 3}]").unwrap(), "Sin[y]");
  }

  // The constant coefficient inside a multivariate expansion likewise stays
  // bare rather than being wrapped in a degenerate SeriesData.
  #[test]
  fn multivariate_constant_coefficient_stays_bare() {
    assert_eq!(
      interpret("Series[x + y, {x, 0, 2}, {y, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {SeriesData[y, 0, {1}, 1, 3, 1], 1}, 0, 3, 1]"
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

  // An unknown function of the limit variable can be discontinuous at the
  // point, so wolframscript keeps the limit unevaluated rather than
  // substituting f[0]. A var-free argument (f[0]) is a constant and still
  // substitutes.
  #[test]
  fn limit_unknown_function_stays_unevaluated() {
    assert_eq!(
      interpret("Limit[f[x], x -> 0]").unwrap(),
      "Limit[f[x], x -> 0]"
    );
    assert_eq!(
      interpret("Limit[f[x] + 1, x -> 0]").unwrap(),
      "Limit[1 + f[x], x -> 0]"
    );
    assert_eq!(
      interpret("Limit[g[x] h[x], x -> 0]").unwrap(),
      "Limit[g[x]*h[x], x -> 0]"
    );
    assert_eq!(
      interpret("Limit[Sin[f[x]], x -> 0]").unwrap(),
      "Limit[Sin[f[x]], x -> 0]"
    );
    // f[0] is a var-free constant, so only the `+ x` term is substituted.
    assert_eq!(interpret("Limit[f[0] + x, x -> 0]").unwrap(), "f[0]");
  }

  #[test]
  fn limit_free_of_variable() {
    // The limit of an expression not involving the limit variable is itself.
    assert_eq!(interpret("Limit[Log[a], x -> 0]").unwrap(), "Log[a]");
    assert_eq!(interpret("Limit[a, x -> 0]").unwrap(), "a");
    assert_eq!(interpret("Limit[a + b, x -> 5]").unwrap(), "a + b");
    assert_eq!(interpret("Limit[Sin[a], x -> 0]").unwrap(), "Sin[a]");
  }

  #[test]
  fn limit_symbolic_substitution_value() {
    // Direct substitution at a point of continuity yields a symbolic value.
    assert_eq!(interpret("Limit[a x, x -> 2]").unwrap(), "2*a");
    assert_eq!(interpret("Limit[a x^2 + b, x -> 3]").unwrap(), "9*a + b");
    assert_eq!(interpret("Limit[a Sin[x], x -> Pi/2]").unwrap(), "a");
  }

  #[test]
  fn limit_symbolic_base_exponential_ratio() {
    // L'Hopital finishing with a constant-w.r.t.-x value: (a^x-1)/x -> Log[a].
    assert_eq!(interpret("Limit[(a^x - 1)/x, x -> 0]").unwrap(), "Log[a]");
  }

  // One-sided limits at jump discontinuities must use the value approached
  // from the given side, not the value AT the point.
  #[test]
  fn one_sided_floor_from_below() {
    assert_eq!(
      interpret("Limit[Floor[x], x -> 2, Direction -> \"FromBelow\"]").unwrap(),
      "1"
    );
  }

  #[test]
  fn one_sided_floor_from_above() {
    assert_eq!(
      interpret("Limit[Floor[x], x -> 2, Direction -> \"FromAbove\"]").unwrap(),
      "2"
    );
  }

  #[test]
  fn one_sided_ceiling_from_above() {
    assert_eq!(
      interpret("Limit[Ceiling[x], x -> 2, Direction -> \"FromAbove\"]")
        .unwrap(),
      "3"
    );
  }

  #[test]
  fn one_sided_sign_from_above() {
    assert_eq!(
      interpret("Limit[Sign[x], x -> 0, Direction -> \"FromAbove\"]").unwrap(),
      "1"
    );
  }

  #[test]
  fn one_sided_sign_from_below() {
    assert_eq!(
      interpret("Limit[Sign[x], x -> 0, Direction -> \"FromBelow\"]").unwrap(),
      "-1"
    );
  }

  // A two-sided (default) limit at a jump discontinuity does NOT exist:
  // direct substitution returns Floor[2] = 2, but the left limit is 1 and the
  // right limit is 2, so Wolfram returns Indeterminate.
  #[test]
  fn two_sided_jump_is_indeterminate() {
    assert_eq!(
      interpret("Limit[Floor[x], x -> 2]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Limit[Ceiling[x], x -> 2]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Limit[Sign[x], x -> 0]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Limit[UnitStep[x], x -> 0]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Limit[FractionalPart[x], x -> 2]").unwrap(),
      "Indeterminate"
    );
    // Round and Mod step at half-integers / multiples of the modulus.
    assert_eq!(
      interpret("Limit[Round[x], x -> 5/2]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Limit[Mod[x, 3], x -> 3]").unwrap(),
      "Indeterminate"
    );
  }

  // ... but where the step function is continuous AT the point, the two-sided
  // limit is the ordinary value (no spurious Indeterminate).
  #[test]
  fn two_sided_continuous_step_point() {
    // Floor is continuous at non-integers.
    assert_eq!(interpret("Limit[Floor[x], x -> 5/2]").unwrap(), "2");
    // Round is continuous at integers (it jumps at half-integers).
    assert_eq!(interpret("Limit[Round[x], x -> 2]").unwrap(), "2");
    // x^2 + 1 approaches 1 from above on both sides, so Floor -> 1.
    assert_eq!(interpret("Limit[Floor[x^2 + 1], x -> 0]").unwrap(), "1");
  }

  #[test]
  fn one_sided_unit_step_from_below() {
    assert_eq!(
      interpret("Limit[UnitStep[x], x -> 0, Direction -> \"FromBelow\"]")
        .unwrap(),
      "0"
    );
  }

  // Abs[x]/x reduces via L'Hopital to Sign[x]; the Direction must propagate.
  #[test]
  fn one_sided_abs_over_x_from_below() {
    assert_eq!(
      interpret("Limit[Abs[x]/x, x -> 0, Direction -> \"FromBelow\"]").unwrap(),
      "-1"
    );
  }

  // A continuous point keeps its exact (symbolic) value — the cross-check must
  // not force a numerical approximation here.
  #[test]
  fn one_sided_continuous_stays_exact() {
    assert_eq!(
      interpret("Limit[Floor[x] + Pi, x -> 5/2, Direction -> \"FromBelow\"]")
        .unwrap(),
      "2 + Pi"
    );
  }

  // Indeterminate power forms f^g, resolved via Exp[Limit[g Log[f]]].
  // 0^0 forms:
  #[test]
  fn limit_x_pow_x() {
    assert_eq!(interpret("Limit[x^x, x -> 0]").unwrap(), "1");
  }

  #[test]
  fn limit_x_pow_sin_x() {
    assert_eq!(interpret("Limit[x^Sin[x], x -> 0]").unwrap(), "1");
  }

  #[test]
  fn limit_sin_x_pow_x() {
    assert_eq!(interpret("Limit[Sin[x]^x, x -> 0]").unwrap(), "1");
  }

  // 1^Infinity forms:
  #[test]
  fn limit_one_plus_x_pow_recip_x() {
    assert_eq!(interpret("Limit[(1 + x)^(1/x), x -> 0]").unwrap(), "E");
  }

  #[test]
  fn limit_cos_pow_recip_x_squared() {
    // (Cos[x])^(1/x^2) -> Exp[-1/2] = 1/Sqrt[E]; the numerical fallback would
    // wrongly give 1.
    assert_eq!(
      interpret("Limit[(Cos[x])^(1/x^2), x -> 0]").unwrap(),
      "1/Sqrt[E]"
    );
  }

  // Infinity^0 form:
  #[test]
  fn limit_recip_x_pow_x() {
    assert_eq!(interpret("Limit[(1/x)^x, x -> 0]").unwrap(), "1");
  }

  // Factorial / Gamma diverge to +Infinity, as do Log[n!] and Sqrt[n!].
  #[test]
  fn limit_factorial_diverges() {
    assert_eq!(interpret("Limit[n!, n -> Infinity]").unwrap(), "Infinity");
    assert_eq!(
      interpret("Limit[Log[n!], n -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Gamma[n], n -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Sqrt[n!], n -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(interpret("Limit[n!!, n -> Infinity]").unwrap(), "Infinity");
  }

  // The reciprocal still decays to 0.
  #[test]
  fn limit_recip_factorial_decays() {
    assert_eq!(interpret("Limit[1/n!, n -> Infinity]").unwrap(), "0");
  }

  // Gamma ratios collapse via FunctionExpand before the limit is taken, so
  // Gamma[x+1]/Gamma[x] = x diverges and Gamma[x]/Gamma[x+1] = 1/x decays.
  #[test]
  fn limit_gamma_ratio() {
    assert_eq!(
      interpret("Limit[Gamma[x + 1]/Gamma[x], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Gamma[x + 2]/Gamma[x], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Gamma[x]/Gamma[x + 1], x -> Infinity]").unwrap(),
      "0"
    );
  }

  // HarmonicNumber[n] at +Infinity is replaced by its asymptotic expansion
  // (Log[n] + EulerGamma + 1/(2n) - 1/(12 n^2) + …) so the limit resolves
  // symbolically. This also prevents a hang: the numeric fallback would
  // otherwise sum HarmonicNumber at a probe of n = 10^6 (astronomically slow).
  #[test]
  fn limit_harmonic_number_asymptotic() {
    // H_n - Log[n] -> EulerGamma.
    assert_eq!(
      interpret("Limit[HarmonicNumber[n] - Log[n], n -> Infinity]").unwrap(),
      "EulerGamma"
    );
    // The 1/(2n) term gives n (H_n - Log[n] - EulerGamma) -> 1/2.
    assert_eq!(
      interpret(
        "Limit[n (HarmonicNumber[n] - Log[n] - EulerGamma), n -> Infinity]"
      )
      .unwrap(),
      "1/2"
    );
    // The -1/(12 n^2) term gives the next order -> -1/12.
    assert_eq!(
      interpret(
        "Limit[n^2 (HarmonicNumber[n] - Log[n] - EulerGamma - 1/(2 n)), \
         n -> Infinity]"
      )
      .unwrap(),
      "-1/12"
    );
    // H_n itself still diverges, and its reciprocal decays.
    assert_eq!(
      interpret("Limit[HarmonicNumber[n], n -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[1/HarmonicNumber[n], n -> Infinity]").unwrap(),
      "0"
    );
    // The exact value at a finite integer is unaffected.
    assert_eq!(interpret("HarmonicNumber[10]").unwrap(), "7381/2520");
  }

  // Regression: an exact non-integer argument must stay symbolic (only a
  // machine-precision real numericizes). wolframscript keeps HarmonicNumber[1/2]
  // unevaluated but HarmonicNumber[0.5] is a float.
  #[test]
  fn harmonic_number_exact_non_integer_stays_symbolic() {
    assert_eq!(
      interpret("HarmonicNumber[1/2]").unwrap(),
      "HarmonicNumber[1/2]"
    );
    assert_eq!(
      interpret("HarmonicNumber[3/2]").unwrap(),
      "HarmonicNumber[3/2]"
    );
    assert_eq!(
      interpret("HarmonicNumber[Sqrt[2]]").unwrap(),
      "HarmonicNumber[Sqrt[2]]"
    );
    // A machine real still numericizes.
    assert_eq!(
      interpret("HarmonicNumber[3.0]").unwrap(),
      "1.8333333333333335"
    );
  }

  // Product 0 * Infinity at a finite point: the L'Hopital rewrite must use the
  // 0/0 orientation (Log[2-x]/Cot[Pi x/2]) — the Infinity/Infinity orientation
  // differentiates Tan into ever-larger expressions that never resolve. This
  // case previously did not terminate; here we only require that it resolves
  // quickly to the correct numeric value (2/Pi ~ 0.6366).
  #[test]
  fn limit_tan_times_log_product_terminates() {
    let out = interpret("Limit[Tan[Pi x/2] Log[2 - x], x -> 1]").unwrap();
    let val: f64 = out.parse().unwrap_or_else(|_| {
      panic!("expected a numeric limit, got {out}");
    });
    assert!(
      (val - std::f64::consts::FRAC_2_PI).abs() < 1e-6,
      "expected ~2/Pi, got {val}"
    );
  }

  // Direct substitution to a real, directed +/-Infinity (e.g. Log[0]) is
  // the limit. ComplexInfinity poles (1/x at 0) stay Indeterminate.
  #[test]
  fn limit_log_at_zero() {
    assert_eq!(interpret("Limit[Log[x], x -> 0]").unwrap(), "-Infinity");
  }

  #[test]
  fn limit_negative_log_at_zero() {
    assert_eq!(interpret("Limit[-Log[x], x -> 0]").unwrap(), "Infinity");
  }

  #[test]
  fn limit_scaled_log_at_zero() {
    assert_eq!(interpret("Limit[3 Log[x], x -> 0]").unwrap(), "-Infinity");
  }

  #[test]
  fn limit_shifted_log_at_zero() {
    assert_eq!(interpret("Limit[Log[x] + 5, x -> 0]").unwrap(), "-Infinity");
    assert_eq!(
      interpret("Limit[2 Log[x] - 1, x -> 0]").unwrap(),
      "-Infinity"
    );
  }

  // Slowly-growing monotonic forms diverging at +Infinity, detected
  // structurally (they never reach the numeric |f| > 1e5 fast-path).
  #[test]
  fn limit_log_at_infinity() {
    assert_eq!(
      interpret("Limit[Log[x], x -> Infinity]").unwrap(),
      "Infinity"
    );
  }

  // Limits of expressions that are rational in E^x (and Sinh/Cosh) at
  // infinity: the standard large-x probe overflows E^x, so a moderate-point
  // probe recovers the finite value.
  #[test]
  fn limit_exponential_ratio_at_infinity() {
    assert_eq!(
      interpret("Limit[(1 + Sinh[x])/Exp[x], x -> Infinity]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Limit[Sinh[x]/Exp[x], x -> Infinity]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Limit[E^x/(E^x + 1), x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[(2 E^x + 3)/(E^x + 1), x -> Infinity]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Limit[Sinh[x]/Cosh[x], x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(interpret("Limit[1/(E^x + 1), x -> Infinity]").unwrap(), "0");
    // Negative infinity: E^x -> 0.
    assert_eq!(
      interpret("Limit[(E^x + 1)/(E^x - 1), x -> -Infinity]").unwrap(),
      "-1"
    );
    // General bases c^x: the larger base dominates (and the giant exact
    // integer 3^1000000 is avoided so this no longer hangs).
    assert_eq!(
      interpret("Limit[(3^x + 2^x)/3^x, x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(interpret("Limit[2^x/3^x, x -> Infinity]").unwrap(), "0");
    assert_eq!(
      interpret("Limit[3^x/2^x, x -> Infinity]").unwrap(),
      "Infinity"
    );
  }

  // Rational-function limits at infinity whose value is a non-integer rational
  // (the ratio of leading coefficients) are now recognised.
  #[test]
  fn limit_rational_function_at_infinity() {
    assert_eq!(
      interpret("Limit[(x^2 + 1)/(2 x^2 - 3), x -> Infinity]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Limit[(3 x + 1)/(2 x - 5), x -> Infinity]").unwrap(),
      "3/2"
    );
    assert_eq!(
      interpret("Limit[(5 x^2 - x)/(3 x^2 + 2), x -> Infinity]").unwrap(),
      "5/3"
    );
    assert_eq!(
      interpret("Limit[(2 x^3)/(3 x^3 + x), x -> Infinity]").unwrap(),
      "2/3"
    );
    // An irrational limit must not be mis-rounded to a rational.
    assert_eq!(
      interpret("Limit[ArcTan[x], x -> Infinity]").unwrap(),
      "Pi/2"
    );
  }

  #[test]
  fn limit_sqrt_at_infinity() {
    assert_eq!(
      interpret("Limit[Sqrt[x], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Sqrt[x] + 1, x -> Infinity]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn limit_fractional_power_at_infinity() {
    assert_eq!(
      interpret("Limit[x^(1/3), x -> Infinity]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn limit_nested_and_scaled_log_at_infinity() {
    assert_eq!(
      interpret("Limit[Log[Log[x]], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Log[2 x], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Log[x]^2, x -> Infinity]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn limit_log_over_power_decays() {
    // A power of x dominates any logarithm: Log[x]/x^p -> 0 for p > 0.
    assert_eq!(
      interpret("Limit[Log[x]/Sqrt[x], x -> Infinity]").unwrap(),
      "0"
    );
    assert_eq!(interpret("Limit[Log[x]^3/x, x -> Infinity]").unwrap(), "0");
    assert_eq!(
      interpret("Limit[Log[x]^2/Sqrt[x], x -> Infinity]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Limit[Log[x]/x^(1/3), x -> Infinity]").unwrap(),
      "0"
    );
  }

  #[test]
  fn limit_conjugate_difference() {
    // Sqrt[A] - Sqrt[B] (a polynomial term counting as Sqrt[p^2]) at infinity.
    assert_eq!(
      interpret("Limit[Sqrt[x^2 + x] - x, x -> Infinity]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Limit[Sqrt[x^2 + 3 x] - x, x -> Infinity]").unwrap(),
      "3/2"
    );
    assert_eq!(
      interpret("Limit[x - Sqrt[x^2 - x], x -> Infinity]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Limit[Sqrt[4 x^2 + x] - 2 x, x -> Infinity]").unwrap(),
      "1/4"
    );
    // Degree-1 radicands: the difference decays to 0.
    assert_eq!(
      interpret("Limit[Sqrt[n + 1] - Sqrt[n], n -> Infinity]").unwrap(),
      "0"
    );
  }

  #[test]
  fn limit_power_over_log_diverges() {
    // Dually, x^p / Log[x] -> Infinity.
    assert_eq!(
      interpret("Limit[x/Log[x], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Sqrt[x]/Log[x], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[-x/Log[x], x -> Infinity]").unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn limit_negated_log_at_infinity() {
    assert_eq!(
      interpret("Limit[-Log[x], x -> Infinity]").unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn limit_mixed_growth_defers_to_numeric() {
    // Indeterminate Infinity - Infinity forms are left to the numeric path,
    // which resolves them by the dominant term.
    assert_eq!(
      interpret("Limit[Log[x] - x, x -> Infinity]").unwrap(),
      "-Infinity"
    );
    assert_eq!(
      interpret("Limit[x - Log[x], x -> Infinity]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn limit_simple_pole_stays_indeterminate() {
    // ComplexInfinity, not a signed Infinity: the two sides disagree.
    assert_eq!(interpret("Limit[1/x, x -> 0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Limit[1/x^3, x -> 0]").unwrap(), "Indeterminate");
  }

  // Functions that stay symbolic at an exact integer argument (ArcCot,
  // ArcCoth) but evaluate under N[...]; the numeric limit heuristic now
  // falls back to N so these resolve.
  #[test]
  fn limit_arccot_at_infinity() {
    assert_eq!(interpret("Limit[ArcCot[x], x -> Infinity]").unwrap(), "0");
    assert_eq!(interpret("Limit[ArcCot[x], x -> -Infinity]").unwrap(), "0");
  }

  #[test]
  fn limit_arccoth_at_infinity() {
    assert_eq!(interpret("Limit[ArcCoth[x], x -> Infinity]").unwrap(), "0");
  }

  // A function with a known finite value at Infinity is resolved by direct
  // substitution. Regression: Limit[HarmonicNumber[n, 2], n -> Infinity] used
  // to fall into the numeric fallback and hang summing ~10^7 exact terms;
  // it now resolves to Zeta[2] = Pi^2/6 immediately.
  #[test]
  fn limit_harmonic_number_at_infinity() {
    assert_eq!(
      interpret("Limit[HarmonicNumber[n, 2], n -> Infinity]").unwrap(),
      "Pi^2/6"
    );
    assert_eq!(
      interpret("Limit[HarmonicNumber[n], n -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("DiscreteLimit[Sum[1/k^2, {k, 1, n}], n -> Infinity]").unwrap(),
      "Pi^2/6"
    );
  }

  // Reciprocals of slowly-diverging forms decay to 0 (and sums of those with
  // constants tend to the constant part) — detected structurally because the
  // numeric path's threshold misses the slow decay.
  #[test]
  fn limit_reciprocal_log_decays_to_zero() {
    assert_eq!(interpret("Limit[1/Log[x], x -> Infinity]").unwrap(), "0");
    assert_eq!(interpret("Limit[1/Log[x]^2, x -> Infinity]").unwrap(), "0");
    assert_eq!(
      interpret("Limit[1/Log[Log[x]], x -> Infinity]").unwrap(),
      "0"
    );
  }

  #[test]
  fn limit_reciprocal_sqrt_decays_to_zero() {
    assert_eq!(interpret("Limit[1/Sqrt[x], x -> Infinity]").unwrap(), "0");
    assert_eq!(interpret("Limit[1/x^(1/3), x -> Infinity]").unwrap(), "0");
  }

  #[test]
  fn limit_scaled_reciprocal_decays_to_zero() {
    assert_eq!(interpret("Limit[2/Log[x], x -> Infinity]").unwrap(), "0");
  }

  #[test]
  fn limit_constant_plus_decaying_term() {
    assert_eq!(
      interpret("Limit[1 + 1/Log[x], x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[5 - 3/Sqrt[x], x -> Infinity]").unwrap(),
      "5"
    );
  }

  #[test]
  fn limit_divergent_plus_decaying_still_diverges() {
    // The divergent term dominates; decay detection must not hijack this.
    assert_eq!(
      interpret("Limit[x + 1/Log[x], x -> Infinity]").unwrap(),
      "Infinity"
    );
  }

  // 0 * Infinity indeterminate products at a finite point, resolved by
  // rewriting as an Infinity/Infinity quotient and applying L'Hopital.
  #[test]
  fn limit_x_log_x() {
    assert_eq!(interpret("Limit[x Log[x], x -> 0]").unwrap(), "0");
  }

  #[test]
  fn limit_x_squared_log_x() {
    assert_eq!(interpret("Limit[x^2 Log[x], x -> 0]").unwrap(), "0");
  }

  #[test]
  fn limit_sqrt_x_log_x() {
    assert_eq!(interpret("Limit[Sqrt[x] Log[x], x -> 0]").unwrap(), "0");
  }

  #[test]
  fn limit_x_log_x_squared() {
    // Iterated logarithm power: requires recursive L'Hopital.
    assert_eq!(interpret("Limit[x Log[x]^2, x -> 0]").unwrap(), "0");
  }

  #[test]
  fn limit_sin_x_log_x() {
    assert_eq!(interpret("Limit[Sin[x] Log[x], x -> 0]").unwrap(), "0");
  }

  #[test]
  fn limit_x_cot_x() {
    // x -> 0, Cot[x] -> Infinity; the product tends to 1.
    assert_eq!(interpret("Limit[x Cot[x], x -> 0]").unwrap(), "1");
  }

  #[test]
  fn limit_product_zero_times_zero_unaffected() {
    // Both factors -> 0 (not a 0*Infinity form): stays 0.
    assert_eq!(interpret("Limit[x Sin[x], x -> 0]").unwrap(), "0");
  }

  #[test]
  fn limit_compound_interest() {
    assert_eq!(interpret("Limit[(1 + 1/n)^n, n -> Infinity]").unwrap(), "E");
  }

  // The generalized e-limit holds with a symbolic rate: (1 + a/n)^n -> E^a.
  #[test]
  fn limit_compound_interest_symbolic_rate() {
    assert_eq!(
      interpret("Limit[(1 + a/n)^n, n -> Infinity]").unwrap(),
      "E^a"
    );
    assert_eq!(
      interpret("Limit[(1 + x/n)^n, n -> Infinity]").unwrap(),
      "E^x"
    );
    // Numeric rates still resolve.
    assert_eq!(
      interpret("Limit[(1 + 2/n)^n, n -> Infinity]").unwrap(),
      "E^2"
    );
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
  fn limit_sin_x_over_x_no_warnings() {
    clear_state();
    // Limit[Sin[x]/x, x -> 0] should not emit Power::infy warnings
    // during its internal trial substitution
    let result = interpret_with_stdout("Limit[Sin[x]/x, x -> 0]").unwrap();
    assert_eq!(result.result, "1");
    assert!(
      result.warnings.is_empty(),
      "Expected no warnings but got: {:?}",
      result.warnings
    );
  }

  #[test]
  fn limit_lhopital_x2_minus_1_over_x_minus_1() {
    assert_eq!(interpret("Limit[(x^2 - 1)/(x - 1), x -> 1]").unwrap(), "2");
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

  #[test]
  fn limit_direction_numeric_from_below() {
    // Direction -> 1 means from below (from the left)
    assert_eq!(
      interpret("Limit[1/x, x -> 0, Direction -> 1]").unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn limit_direction_numeric_from_above() {
    // Direction -> -1 means from above (from the right)
    assert_eq!(
      interpret("Limit[1/x, x -> 0, Direction -> -1]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn limit_piecewise_from_below() {
    assert_eq!(
      interpret(
        "f[x_] := Piecewise[{{x, x < -1}, {x^2, x >= -1}}]; \
         Limit[f[x], x -> -1, Direction -> 1]"
      )
      .unwrap(),
      "-1"
    );
  }

  #[test]
  fn limit_piecewise_from_above() {
    assert_eq!(
      interpret(
        "f[x_] := Piecewise[{{x, x < -1}, {x^2, x >= -1}}]; \
         Limit[f[x], x -> -1, Direction -> -1]"
      )
      .unwrap(),
      "1"
    );
  }

  #[test]
  fn limit_piecewise_two_sided_indeterminate() {
    // Two-sided limit at a discontinuity should be Indeterminate
    assert_eq!(
      interpret(
        "f[x_] := Piecewise[{{x, x < -1}, {x^2, x >= -1}}]; \
         Limit[f[x], x -> -1]"
      )
      .unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn limit_piecewise_continuous_point() {
    // At a point where both branches agree, the limit should exist
    assert_eq!(
      interpret(
        "g[x_] := Piecewise[{{x^2, x < 0}, {x, x >= 0}}]; \
         Limit[g[x], x -> 0]"
      )
      .unwrap(),
      "0"
    );
  }

  #[test]
  fn limit_arctan_at_infinity() {
    assert_eq!(
      interpret("Limit[ArcTan[x], x -> Infinity]").unwrap(),
      "Pi/2"
    );
  }

  #[test]
  fn limit_arctan_at_negative_infinity() {
    assert_eq!(
      interpret("Limit[ArcTan[x], x -> -Infinity]").unwrap(),
      "-1/2*Pi"
    );
  }

  #[test]
  fn limit_negative_infinity_convergence() {
    // Limit[1/x, x -> -Infinity] = 0
    assert_eq!(interpret("Limit[1/x, x -> -Infinity]").unwrap(), "0");
  }

  #[test]
  fn limit_exp_minus_1_over_x() {
    // L'Hôpital for 0/0 form in canonical Times[Power[x,-1],...] form
    assert_eq!(interpret("Limit[(E^x - 1)/x, x -> 0]").unwrap(), "1");
  }

  #[test]
  fn limit_log_1_plus_x_over_x() {
    assert_eq!(interpret("Limit[Log[1 + x]/x, x -> 0]").unwrap(), "1");
  }

  #[test]
  fn limit_1_minus_cos_over_x_squared() {
    // Requires two applications of L'Hôpital's rule
    assert_eq!(interpret("Limit[(1 - Cos[x])/x^2, x -> 0]").unwrap(), "1/2");
  }

  #[test]
  fn limit_exp_minus_1_minus_x_over_x_squared() {
    assert_eq!(
      interpret("Limit[(E^x - 1 - x)/x^2, x -> 0]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn discrete_limit_symbolic_stays_unevaluated() {
    // DiscreteLimit on a symbolic `f[n]` has no obvious closed form, so both
    // Woxi and wolframscript leave it unevaluated (mathics returns
    // f[Infinity] — a mathics-specific simplification we deliberately do
    // NOT apply).
    assert_eq!(
      interpret("DiscreteLimit[f[n], n -> Infinity]").unwrap(),
      "DiscreteLimit[f[n], n -> Infinity]"
    );
  }

  #[test]
  fn discrete_limit_rational_at_infinity() {
    // For rational sequences where Limit succeeds, DiscreteLimit returns
    // the same value (matches wolframscript).
    assert_eq!(
      interpret("DiscreteLimit[n/(n + 1), n -> Infinity]").unwrap(),
      "1"
    );
  }
  #[test]
  fn limit_factors_out_constants_at_infinity() {
    // Limit[a*f(n), n -> Infinity] should pull `a` (free of n) out of
    // the limit. wolframscript reduces this to `a` (since n/(n+2) -> 1).
    assert_eq!(interpret("Limit[a*n/(n + 2), n -> Infinity]").unwrap(), "a");
  }
  #[test]
  fn discrete_limit_multivar_at_infinity() {
    // DiscreteLimit applied to a 2-variable expression with iterated
    // substitutions {m -> Inf, n -> Inf} should reduce to E^(-1).
    // The outer limit pulls n/(n+2) -> 1; the inner pulls
    // E^(-m/(m+1)) -> E^(-1).
    assert_eq!(
      interpret(
        "DiscreteLimit[(n/(n + 2)) E^(-m/(m + 1)), {m -> Infinity, n -> Infinity}]"
      )
      .unwrap(),
      "E^(-1)"
    );
  }

  // ─── Bounded oscillating sums at infinity → Interval ────────────────
  //
  // When the limit at infinity is undefined but the expression is a sum
  // of trig terms whose polynomial arguments have *different* degrees
  // in the limit variable, the accumulation set densely fills
  // [-bound, bound] where `bound` = Σ|coefficients|. wolframscript
  // returns the closed Interval; sums of trig terms whose arguments all
  // have the same polynomial degree stay Indeterminate.

  #[test]
  fn osc_sum_sin_x_plus_sin_x2() {
    // Audit case.
    assert_eq!(
      interpret("Limit[Sin[x] + Sin[x^2], x -> Infinity]").unwrap(),
      "Interval[{-2, 2}]"
    );
  }

  #[test]
  fn osc_sum_cos_x_plus_cos_x2() {
    assert_eq!(
      interpret("Limit[Cos[x] + Cos[x^2], x -> Infinity]").unwrap(),
      "Interval[{-2, 2}]"
    );
  }

  #[test]
  fn osc_sum_mixed_sin_cos() {
    assert_eq!(
      interpret("Limit[Sin[x] + Cos[x^2], x -> Infinity]").unwrap(),
      "Interval[{-2, 2}]"
    );
  }

  #[test]
  fn osc_sum_three_terms_distinct_degrees() {
    assert_eq!(
      interpret("Limit[Sin[x] + Sin[x^2] + Sin[x^3], x -> Infinity]").unwrap(),
      "Interval[{-3, 3}]"
    );
  }

  #[test]
  fn osc_sum_three_terms_same_degree_pair_then_distinct() {
    // Two same-degree terms get "broken" by a third at a different degree;
    // the bound is the literal count of trig summands.
    assert_eq!(
      interpret("Limit[Sin[x] + Cos[x] + Sin[x^2], x -> Infinity]").unwrap(),
      "Interval[{-3, 3}]"
    );
  }

  #[test]
  fn osc_sum_with_coefficients() {
    assert_eq!(
      interpret("Limit[2*Sin[x] + Sin[x^2], x -> Infinity]").unwrap(),
      "Interval[{-3, 3}]"
    );
    assert_eq!(
      interpret("Limit[3*Sin[x] + Sin[x^2], x -> Infinity]").unwrap(),
      "Interval[{-4, 4}]"
    );
  }

  // Counterexample: all trig args at the same polynomial degree → stays
  // Indeterminate (matches wolframscript).
  #[test]
  fn osc_sum_same_degree_stays_indeterminate() {
    assert_eq!(
      interpret("Limit[Sin[x] + Cos[x], x -> Infinity]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Limit[Sin[x] + Sin[2*x], x -> Infinity]").unwrap(),
      "Indeterminate"
    );
  }

  // Limit[-f] peels to -Limit[f] so the exact quotient paths apply
  // instead of the low-accuracy numeric fallback
  // (wolframscript-verified).
  #[test]
  fn minus_wrapper_peels() {
    assert_eq!(
      interpret("Limit[-((Pi - z)/Sin[z]), z -> Pi]").unwrap(),
      "-1"
    );
  }

  // Zeta at 1 and Gamma at nonpositive integers cross their poles via
  // truncated Laurent models; the 0*Infinity strategies previously
  // returned a wrong 0 for the Zeta product (wolframscript-verified).
  #[test]
  fn known_pole_products() {
    assert_eq!(interpret("Limit[(z - 1)*Zeta[z], z -> 1]").unwrap(), "1");
    assert_eq!(
      interpret("Limit[(z + 3)*Gamma[z], z -> -3]").unwrap(),
      "-1/6"
    );
    assert_eq!(
      interpret("Limit[Zeta[z], z -> 1]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Limit[Gamma[z], z -> 0]").unwrap(),
      "Indeterminate"
    );
  }

  // Reciprocal trig heads rewrite to Sin/Cos quotients before a strategy
  // is picked (wolframscript-verified).
  #[test]
  fn reciprocal_trig_limits() {
    assert_eq!(interpret("Limit[x*Cot[x], x -> 0]").unwrap(), "1");
    assert_eq!(interpret("Limit[Cot[x], x -> 0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Limit[(z + Pi)*Csc[z], z -> -Pi]").unwrap(), "-1");
  }
}

mod residue {
  use super::*;

  #[test]
  fn simple_pole_at_zero() {
    assert_eq!(interpret("Residue[1/z, {z, 0}]").unwrap(), "1");
  }

  #[test]
  fn simple_pole_shifted() {
    assert_eq!(interpret("Residue[1/(z - 2), {z, 2}]").unwrap(), "1");
  }

  #[test]
  fn simple_pole_of_rational() {
    // 1/(z^2-1) has simple poles at ±1, residue 1/2 at z = 1.
    assert_eq!(interpret("Residue[1/(z^2 - 1), {z, 1}]").unwrap(), "1/2");
  }

  #[test]
  fn removable_singularity_is_simple_pole() {
    // Sin[z]/z^2 = 1/z - z/6 + ... — a simple pole with residue 1.
    assert_eq!(interpret("Residue[Sin[z]/z^2, {z, 0}]").unwrap(), "1");
  }

  #[test]
  fn double_pole_residue_zero() {
    // 1/z^2 has no 1/z term, so the residue is 0.
    assert_eq!(interpret("Residue[1/z^2, {z, 0}]").unwrap(), "0");
  }

  #[test]
  fn higher_order_pole() {
    // Exp[z]/z^3 = (1 + z + z^2/2 + ...)/z^3, coefficient of 1/z is 1/2.
    assert_eq!(interpret("Residue[Exp[z]/z^3, {z, 0}]").unwrap(), "1/2");
  }

  #[test]
  fn second_order_pole_with_numerator() {
    assert_eq!(
      interpret("Residue[(z + 1)/(z - 1)^2, {z, 1}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn regular_point_residue_zero() {
    // No singularity at z = 0, so the residue is 0.
    assert_eq!(interpret("Residue[z^2, {z, 0}]").unwrap(), "0");
  }

  #[test]
  fn cotangent_residue() {
    assert_eq!(interpret("Residue[Cot[z], {z, 0}]").unwrap(), "1");
  }

  #[test]
  fn complex_pole() {
    // 1/(z^2+1) has poles at ±I; residue at z = I is -I/2.
    assert_eq!(interpret("Residue[1/(z^2 + 1), {z, I}]").unwrap(), "-1/2*I");
  }

  #[test]
  fn symbolic_pole_location() {
    assert_eq!(interpret("Residue[1/(z - a), {z, a}]").unwrap(), "1");
  }

  #[test]
  fn symbolic_double_pole_residue_zero() {
    assert_eq!(interpret("Residue[b/(z - a)^2, {z, a}]").unwrap(), "0");
  }

  #[test]
  fn quartic_simple_pole() {
    assert_eq!(interpret("Residue[1/(z^4 - 1), {z, 1}]").unwrap(), "1/4");
  }

  #[test]
  fn complex_pole_with_numerator() {
    assert_eq!(interpret("Residue[z/(z^2 + 1), {z, I}]").unwrap(), "1/2");
  }

  // Essential singularities: the pole-order limit loop is invalid there
  // (z*Sin[1/z] -> 0 is a bounded-oscillation limit, not the residue);
  // the Laurent coefficient comes from the z -> z0 + 1/w substitution
  // (all wolframscript-verified).
  #[test]
  fn essential_singularity_residues() {
    assert_eq!(interpret("Residue[Exp[1/z], {z, 0}]").unwrap(), "1");
    assert_eq!(interpret("Residue[Exp[2/z], {z, 0}]").unwrap(), "2");
    assert_eq!(interpret("Residue[z*Exp[1/z], {z, 0}]").unwrap(), "1/2");
    assert_eq!(interpret("Residue[Sin[1/z], {z, 0}]").unwrap(), "1");
    assert_eq!(interpret("Residue[Exp[1/z^2], {z, 0}]").unwrap(), "0");
    assert_eq!(interpret("Residue[Cos[1/z]/z, {z, 0}]").unwrap(), "1");
    assert_eq!(interpret("Residue[z^2*Sin[1/z], {z, 0}]").unwrap(), "-1/6");
    assert_eq!(interpret("Residue[Exp[1/z] + 1/z, {z, 0}]").unwrap(), "2");
    assert_eq!(interpret("Residue[Cosh[1/z]*z, {z, 0}]").unwrap(), "1/2");
  }

  // Gamma has simple poles at nonpositive integers with residue
  // (-1)^n/n!, handled via a truncated Laurent model
  // (wolframscript-verified).
  #[test]
  fn gamma_pole_residues() {
    assert_eq!(interpret("Residue[Gamma[z], {z, 0}]").unwrap(), "1");
    assert_eq!(interpret("Residue[Gamma[z], {z, -1}]").unwrap(), "-1");
    assert_eq!(interpret("Residue[Gamma[z], {z, -3}]").unwrap(), "-1/6");
    assert_eq!(interpret("Residue[Gamma[z], {z, -4}]").unwrap(), "1/24");
    assert_eq!(
      interpret("Residue[Gamma[z]/z, {z, 0}]").unwrap(),
      "-EulerGamma"
    );
    assert_eq!(
      interpret("Residue[Gamma[z]/z^2, {z, 0}]").unwrap(),
      "(6*EulerGamma^2 + Pi^2)/12"
    );
    // A Gamma model times another pole at a NONZERO point hits a
    // pathologically slow Simplify blowup (PolyGamma constants over
    // non-monomial pole quotients), so it stays unevaluated for now
    // (wolframscript: -1 + EulerGamma).
    assert_eq!(
      interpret("Residue[Gamma[z]/(z + 1), {z, -1}]").unwrap(),
      "Residue[Gamma[z]/(1 + z), {z, -1}]"
    );
  }

  // Zeta has a simple pole at 1 with residue 1; the wrong value 0 came
  // from Limit mishandling the 0*Infinity product (regression:
  // differential audit 2026-07-10, wolframscript-verified).
  #[test]
  fn zeta_pole_residues() {
    assert_eq!(interpret("Residue[Zeta[z], {z, 1}]").unwrap(), "1");
    assert_eq!(interpret("Residue[Zeta[z]/z, {z, 1}]").unwrap(), "1");
    assert_eq!(interpret("Residue[2*Zeta[z], {z, 1}]").unwrap(), "2");
    assert_eq!(
      interpret("Residue[Zeta[z]/(z - 1), {z, 1}]").unwrap(),
      "EulerGamma"
    );
    assert_eq!(
      interpret("Residue[Zeta[z]/(z - 1)^2, {z, 1}]").unwrap(),
      "-StieltjesGamma[1]"
    );
  }

  // Reciprocal trig at shifted poles: Simplify turns (z-Pi)/Sin[z] into
  // Csc forms that the limit only resolved numerically
  // (wolframscript-verified).
  #[test]
  fn reciprocal_trig_shifted_poles() {
    assert_eq!(interpret("Residue[1/Sin[z], {z, 0}]").unwrap(), "1");
    assert_eq!(interpret("Residue[1/Sin[z], {z, Pi}]").unwrap(), "-1");
    assert_eq!(interpret("Residue[1/Sin[z], {z, -Pi}]").unwrap(), "-1");
    assert_eq!(interpret("Residue[1/Sin[z], {z, 2*Pi}]").unwrap(), "1");
    assert_eq!(interpret("Residue[Csc[z], {z, Pi}]").unwrap(), "-1");
    assert_eq!(interpret("Residue[Tan[z], {z, Pi/2}]").unwrap(), "-1");
    assert_eq!(interpret("Residue[Sec[z], {z, Pi/2}]").unwrap(), "-1");
    assert_eq!(interpret("Residue[Cot[z], {z, Pi}]").unwrap(), "1");
    assert_eq!(interpret("Residue[1/Sin[z]^2, {z, 0}]").unwrap(), "0");
  }

  // Analytic-function residue via the Laurent picture: f[z]/z at 0.
  #[test]
  fn function_over_z_gives_value_at_zero() {
    assert_eq!(interpret("Residue[f[z]/z, {z, 0}]").unwrap(), "f[0]");
  }
}

mod inverse_series {
  use super::*;

  #[test]
  fn arcsin_from_sin() {
    // Reversion of Sin gives the ArcSin series x + x^3/6 + 3 x^5/40.
    assert_eq!(
      interpret("InverseSeries[Series[Sin[x], {x, 0, 5}]]").unwrap(),
      "SeriesData[x, 0, {1, 0, 1/6, 0, 3/40}, 1, 6, 1]"
    );
  }

  #[test]
  fn log1p_from_expm1() {
    // Reversion of Exp[x]-1 gives Log[1+x] = x - x^2/2 + x^3/3 - x^4/4.
    assert_eq!(
      interpret("InverseSeries[Series[Exp[x] - 1, {x, 0, 4}]]").unwrap(),
      "SeriesData[x, 0, {1, -1/2, 1/3, -1/4}, 1, 5, 1]"
    );
  }

  #[test]
  fn arctan_from_tan_trims_trailing_zero() {
    // Reversion of Tan gives ArcTan; the order-6 term is 0 and is dropped
    // while the truncation order (nmax = 7) is preserved.
    assert_eq!(
      interpret("InverseSeries[Series[Tan[x], {x, 0, 6}]]").unwrap(),
      "SeriesData[x, 0, {1, 0, -1/3, 0, 1/5}, 1, 7, 1]"
    );
  }

  #[test]
  fn polynomial_reversion_catalan() {
    assert_eq!(
      interpret("InverseSeries[Series[x + x^2, {x, 0, 5}]]").unwrap(),
      "SeriesData[x, 0, {1, -1, 2, -5, 14}, 1, 6, 1]"
    );
  }

  #[test]
  fn leading_coefficient_not_one() {
    assert_eq!(
      interpret("InverseSeries[Series[2 x + 3 x^2, {x, 0, 4}]]").unwrap(),
      "SeriesData[x, 0, {1/2, -3/8, 9/16, -135/128}, 1, 5, 1]"
    );
  }

  #[test]
  fn two_arg_form_renames_variable() {
    assert_eq!(
      interpret("InverseSeries[Series[Sin[x], {x, 0, 5}], y]").unwrap(),
      "SeriesData[y, 0, {1, 0, 1/6, 0, 3/40}, 1, 6, 1]"
    );
  }

  #[test]
  fn normal_of_inverse_series() {
    assert_eq!(
      interpret("Normal[InverseSeries[Series[Sin[x], {x, 0, 5}]]]").unwrap(),
      "x + x^3/6 + (3*x^5)/40"
    );
  }
}

mod compose_series {
  use super::*;

  #[test]
  fn exp_of_sin() {
    assert_eq!(
      interpret(
        "ComposeSeries[Series[Exp[x], {x, 0, 3}], Series[Sin[x], {x, 0, 3}]]"
      )
      .unwrap(),
      "SeriesData[x, 0, {1, 1, 1/2}, 0, 4, 1]"
    );
  }

  #[test]
  fn exp_of_polynomial() {
    assert_eq!(
      interpret(
        "ComposeSeries[Series[Exp[y], {y, 0, 4}], Series[x + x^2, {x, 0, 4}]]"
      )
      .unwrap(),
      "SeriesData[x, 0, {1, 1, 3/2, 7/6, 25/24}, 0, 5, 1]"
    );
  }

  #[test]
  fn geometric_of_square_truncates_by_outer() {
    // Inner has nmin = 2; result order M = nmin2 + nmax1 - 1 = 2 + 3 - 1 = 4.
    assert_eq!(
      interpret(
        "ComposeSeries[Series[1/(1-y), {y, 0, 2}], Series[x^2, {x, 0, 5}]]"
      )
      .unwrap(),
      "SeriesData[x, 0, {1, 0, 1}, 0, 4, 1]"
    );
  }

  #[test]
  fn truncation_capped_by_inner_accuracy() {
    // Outer is known to high order but the inner Sin is only known to order 2,
    // so the result is capped at the inner truncation (nmax2 = 3).
    assert_eq!(
      interpret(
        "ComposeSeries[Series[1/(1-y), {y, 0, 5}], Series[Sin[x], {x, 0, 2}]]"
      )
      .unwrap(),
      "SeriesData[x, 0, {1, 1, 1}, 0, 3, 1]"
    );
  }

  #[test]
  fn outer_with_no_constant_term() {
    assert_eq!(
      interpret(
        "ComposeSeries[Series[Log[1+y], {y, 0, 4}], Series[x + x^3, {x, 0, 4}]]"
      )
      .unwrap(),
      "SeriesData[x, 0, {1, -1/2, 4/3, -5/4}, 1, 5, 1]"
    );
  }

  #[test]
  fn nary_composition() {
    assert_eq!(
      interpret(
        "ComposeSeries[Series[Exp[x], {x, 0, 3}], \
         Series[Sin[y], {y, 0, 3}], Series[z^1, {z, 0, 3}]]"
      )
      .unwrap(),
      "SeriesData[z, 0, {1, 1, 1/2}, 0, 4, 1]"
    );
  }

  #[test]
  fn symbolic_outer_coefficients() {
    assert_eq!(
      interpret(
        "ComposeSeries[Series[f[y], {y, 0, 3}], Series[x + x^2, {x, 0, 3}]]"
      )
      .unwrap(),
      "SeriesData[x, 0, {f[0], Derivative[1][f][0], \
       Derivative[1][f][0] + Derivative[2][f][0]/2, \
       Derivative[2][f][0] + Derivative[3][f][0]/6}, 0, 4, 1]"
    );
  }
}

mod pade_approximant {
  use super::*;

  #[test]
  fn exp_two_two() {
    assert_eq!(
      interpret("PadeApproximant[Exp[x], {x, 0, {2, 2}}]").unwrap(),
      "(1 + x/2 + x^2/12)/(1 - x/2 + x^2/12)"
    );
  }

  #[test]
  fn exp_one_one() {
    assert_eq!(
      interpret("PadeApproximant[Exp[x], {x, 0, {1, 1}}]").unwrap(),
      "(1 + x/2)/(1 - x/2)"
    );
  }

  #[test]
  fn single_order_spec_is_diagonal() {
    // PadeApproximant[f, {x, x0, n}] is the diagonal [n/n] approximant,
    // equivalent to PadeApproximant[f, {x, x0, {n, n}}].
    assert_eq!(
      interpret("PadeApproximant[Exp[x], {x, 0, 2}]").unwrap(),
      "(1 + x/2 + x^2/12)/(1 - x/2 + x^2/12)"
    );
    assert_eq!(
      interpret("PadeApproximant[Sin[x], {x, 0, 4}]").unwrap(),
      "(x - (31*x^3)/294)/(1 + (3*x^2)/49 + (11*x^4)/5880)"
    );
  }

  #[test]
  fn exp_three_two_unequal_degrees() {
    assert_eq!(
      interpret("PadeApproximant[Exp[x], {x, 0, {3, 2}}]").unwrap(),
      "(1 + (3*x)/5 + (3*x^2)/20 + x^3/60)/(1 - (2*x)/5 + x^2/20)"
    );
  }

  #[test]
  fn cosine_even_function() {
    assert_eq!(
      interpret("PadeApproximant[Cos[x], {x, 0, {2, 2}}]").unwrap(),
      "(1 - (5*x^2)/12)/(1 + x^2/12)"
    );
  }

  #[test]
  fn log_with_zero_constant_term() {
    assert_eq!(
      interpret("PadeApproximant[Log[1 + x], {x, 0, {2, 2}}]").unwrap(),
      "(x + x^2/2)/(1 + x + x^2/6)"
    );
  }

  #[test]
  fn sqrt_rational_coefficients() {
    assert_eq!(
      interpret("PadeApproximant[Sqrt[1 + x], {x, 0, {1, 1}}]").unwrap(),
      "(1 + (3*x)/4)/(1 + x/4)"
    );
  }

  #[test]
  fn arctan_three_two() {
    assert_eq!(
      interpret("PadeApproximant[ArcTan[x], {x, 0, {3, 2}}]").unwrap(),
      "(x + (4*x^3)/15)/(1 + (3*x^2)/5)"
    );
  }

  #[test]
  fn zero_denominator_degree_is_taylor_polynomial() {
    assert_eq!(
      interpret("PadeApproximant[Exp[x], {x, 0, {2, 0}}]").unwrap(),
      "1 + x + x^2/2"
    );
  }

  #[test]
  fn degenerate_rational_input_stays_unevaluated() {
    // The denominator system is singular for an already-rational function;
    // Woxi leaves it unevaluated rather than guessing the minimal-degree form.
    assert_eq!(
      interpret("PadeApproximant[1/(1 - x), {x, 0, {2, 2}}]").unwrap(),
      "PadeApproximant[(1 - x)^(-1), {x, 0, {2, 2}}]"
    );
  }
}

mod nintegrate {
  use super::*;

  fn assert_approx(code: &str, expected: f64, tol: f64) {
    let result = interpret(code).unwrap();
    let val: f64 = result.parse().unwrap_or_else(|_| {
      panic!("NIntegrate result should be a number, got: {result}")
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

  // Iterated (multi-dimensional) integration: additional ranges are inner
  // integration variables, not ignored. Verified against wolframscript.
  #[test]
  fn nintegrate_iterated_constant_bounds() {
    // ∫₀¹∫₀³ 1 dy dx = 3 (previously ignored the y range and gave 1).
    assert_approx("NIntegrate[1, {x, 0, 1}, {y, 0, 3}]", 3.0, 1e-8);
    // ∫₀¹∫₀² x y dy dx = (1/2)(4/2) = 1.
    assert_approx("NIntegrate[x y, {x, 0, 1}, {y, 0, 2}]", 1.0, 1e-8);
    // ∫₀¹∫₀¹ x² dy dx = 1/3.
    assert_approx("NIntegrate[x^2, {x, 0, 1}, {y, 0, 1}]", 1.0 / 3.0, 1e-8);
  }

  #[test]
  fn nintegrate_iterated_dependent_bounds() {
    // ∫₀¹∫₀ˣ x y dy dx = ∫₀¹ x³/2 dx = 1/8.
    assert_approx("NIntegrate[x y, {x, 0, 1}, {y, 0, x}]", 0.125, 1e-8);
    // Area of the triangle {0<=y<=x<=1} = 1/2.
    assert_approx("NIntegrate[1, {x, 0, 1}, {y, 0, x}]", 0.5, 1e-8);
    // Volume of the simplex {0<=z<=y<=x<=1} = 1/6.
    assert_approx(
      "NIntegrate[1, {x, 0, 1}, {y, 0, x}, {z, 0, y}]",
      1.0 / 6.0,
      1e-7,
    );
  }
  #[test]
  fn n_falls_back_to_nintegrate_for_unevaluated_integrate() {
    // Integrate[Abs[Sin[phi]], {phi, 0, 2 Pi}] doesn't reduce
    // symbolically in Woxi. With // N, wolframscript returns 4.; we
    // should match by routing to NIntegrate.
    assert_approx("Integrate[Abs[Sin[phi]], {phi, 0, 2Pi}] // N", 4.0, 1e-6);
  }

  #[test]
  fn nintegrate_sin() {
    // ∫₀^π sin(x) dx = 2
    assert_approx("NIntegrate[Sin[x], {x, 0, Pi}]", 2.0, 1e-10);
  }

  #[test]
  fn nintegrate_multi_segment_with_interior_waypoint() {
    // `{x, -1, 0, 1}` breaks the interval at the x = 0 singularity of
    // 1/Sqrt[|x|]; ∫₋₁¹ |x|^(-1/2) dx = 4.
    assert_approx("NIntegrate[1/Abs[Sqrt[x]], {x, -1, 0, 1}]", 4.0, 1e-3);
  }

  // An integrand that blows up at an endpoint is integrated by tanh-sinh
  // quadrature, whose nodes crowd towards the endpoints and whose weights
  // vanish there, so the endpoint itself is never sampled. Adaptive Simpson
  // needs an endpoint value and used to substitute one from just inside the
  // interval, which made that one sample enormous: these were off by anything
  // from 1e-6 to a factor of 54.
  #[test]
  fn endpoint_singularities_reach_machine_precision() {
    // Was 2.0000177257751255.
    assert_approx("NIntegrate[1/Sqrt[x], {x, 0, 1}]", 2.0, 1e-13);
    // Was 1.9999988610188073 — the singularity at the upper end.
    assert_approx("NIntegrate[1/Sqrt[1 - x], {x, 0, 1}]", 2.0, 1e-7);
    // Was 1.499999999972582.
    assert_approx("NIntegrate[1/x^(1/3), {x, 0, 1}]", 1.5, 1e-13);
    // A logarithmic singularity on top of an algebraic one; was -4.00010178590504.
    assert_approx("NIntegrate[Log[x]/Sqrt[x], {x, 0, 1}]", -4.0, 1e-13);
    // Was 1.5708147178026441.
    assert_approx(
      "NIntegrate[1/(Sqrt[x] (1 + x)), {x, 0, 1}]",
      std::f64::consts::FRAC_PI_2,
      1e-13,
    );
    // Was 1.5731199744528073 — wrong in the third digit.
    assert_approx(
      "NIntegrate[1/Sqrt[1 - x^2], {x, 0, 1}]",
      std::f64::consts::FRAC_PI_2,
      1e-7,
    );
    // Was 9.751074177483138 for an integral of 10 — 2.5% out.
    assert_approx("NIntegrate[x^(-0.9), {x, 0, 1}]", 10.0, 1e-8);
    // Was 121.33668114667366 for an integral of 2.22 — a factor of 54.
    assert_approx(
      "NIntegrate[Sqrt[Tan[x]], {x, 0, Pi/2}]",
      2.221441469079183,
      1e-7,
    );
  }

  // Over an interval wide enough that tanh-sinh's endpoint-crowded nodes
  // never resolve the interesting region near a removable singularity, it
  // gives up and falls back to adaptive Simpson — which, unlike tanh-sinh,
  // samples the literal endpoint. That endpoint evaluation (Sin[0.]^2 /
  // 0.^2) is discarded and replaced by a nearby perturbed value, exactly
  // like the small-interval case above, but it used to also print a
  // spurious `Power::infy` for the discarded sample before being thrown
  // away — a message wolframscript never shows for this integral.
  #[test]
  fn wide_interval_removable_singularity_prints_no_spurious_message() {
    clear_state();
    let result = interpret("NIntegrate[Sin[x]^2/x^2, {x, 0, 15000}]").unwrap();
    let val: f64 = result.parse().unwrap();
    // ∫₀^∞ Sin[x]^2/x^2 dx = Pi/2; truncating at 15000 drops a tail bounded
    // by 1/15000 (Sin^2 <= 1), on the order of 1e-4. The adaptive-Simpson
    // fallback used for this wide, highly oscillatory interval is not itself
    // fully accurate — that inaccuracy is a separate, pre-existing limitation
    // — so this checks against the true value with a tolerance loose enough
    // to not pin the fallback's current error as correct, while still
    // catching a grossly wrong (e.g. zero or divergent) result.
    let true_value = std::f64::consts::FRAC_PI_2 - 1.0 / 30000.0;
    assert!(
      (val - true_value).abs() < 2e-3,
      "got {val}, expected ~{true_value}"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().all(|m| !m.contains("Power::infy")),
      "spurious message for a discarded endpoint sample: {msgs:?}"
    );
  }

  // Smooth integrands keep their accuracy, including the oscillatory ones that
  // fall back to the adaptive rule because tanh-sinh does not settle on them.
  #[test]
  fn smooth_and_oscillatory_integrands_keep_working() {
    assert_approx("NIntegrate[x^2, {x, 0, 1}]", 1.0 / 3.0, 1e-13);
    assert_approx("NIntegrate[Sin[x], {x, 0, Pi}]", 2.0, 1e-13);
    assert_approx("NIntegrate[1/x, {x, 1, E}]", 1.0, 1e-13);
    assert_approx("NIntegrate[Sin[x]/x, {x, 0, 10}]", 1.658347594218876, 1e-8);
    assert_approx("NIntegrate[Sin[x^2], {x, 0, 10}]", 0.583670880639779, 1e-6);
    assert_approx(
      "NIntegrate[Exp[-x^2], {x, 0, Infinity}]",
      0.886226925452758,
      1e-8,
    );
  }

  #[test]
  fn nintegrate_evaluation_monitor_sows_abscissae() {
    // EvaluationMonitor :> Sow[x] fires at every sampled point, so Reap
    // collects a non-empty list of abscissae, all inside [0, 1].
    clear_state();
    let r = interpret(
      "Module[{pts}, pts = Reap[NIntegrate[x^2, {x, 0, 1}, \
       EvaluationMonitor :> Sow[x]]][[2, 1]]; \
       {Length[pts] > 0, Min[pts] >= 0, Max[pts] <= 1}]",
    )
    .unwrap();
    assert_eq!(r, "{True, True, True}");
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

  // The integrand has a removable singularity at the lower bound (Sin[0]/0
  // is 0/0 but the limit is 1); quadrature should use the limit, not abort.
  #[test]
  fn nintegrate_removable_singularity_at_endpoint() {
    // ∫₀¹⁰ sin(x)/x dx ≈ 1.6583475942188...
    assert_approx("NIntegrate[Sin[x]/x, {x, 0, 10}]", 1.658347594218876, 1e-8);
  }

  // ─── Narrow-Gaussian fast path ────────────────────────────────────
  //
  // Adaptive Simpson can't find the peak of `Exp[-α x²]` once α is
  // large enough that the spike is much narrower than the initial
  // sample spacing. NIntegrate detects the Gaussian shape and uses
  // the closed form
  //
  //   ∫_lo^hi Exp[α x²] dx = Sqrt[π/(-α)]·(Erf[hi·Sqrt[-α]] − Erf[lo·Sqrt[-α]])/2
  //
  // with `α < 0`. This restores the audit case
  // `NIntegrate[Exp[(-10^8) x²], {x, -1, 1}, WorkingPrecision -> 20,
  // MaxRecursion -> 20]` from a timeout to its analytic value.
  #[test]
  fn nintegrate_narrow_gaussian_f64() {
    // The Gaussian's width is ~10⁻⁴ vs an interval [-1, 1]; the spike
    // would be missed by ordinary adaptive Simpson without the closed
    // form. Answer ≈ Sqrt[π]/10⁴ ≈ 1.7724538509e-4.
    assert_approx(
      "NIntegrate[Exp[(-10^8)*x^2], {x, -1, 1}]",
      0.000_177_245_385_090_551_6,
      1e-15,
    );
  }

  #[test]
  fn nintegrate_narrow_gaussian_with_working_precision() {
    // Audit case: with WorkingPrecision -> 20 the result is returned
    // as a 20-digit BigFloat. Woxi's closed form is exact:
    //   Sqrt[π/10^8]·Erf[10^4] = Sqrt[π/10^8]·1 to ~10^-43.
    let result = interpret(
      "NIntegrate[Exp[(-10^8)*x^2], {x, -1, 1}, WorkingPrecision -> 20, MaxRecursion -> 20]",
    )
    .unwrap();
    // Result starts with the canonical Sqrt[Pi/10^8] prefix.
    assert!(
      result.starts_with("0.000177245385090551602"),
      "expected Sqrt[Pi/10^8] prefix, got `{result}`"
    );
    // 20-digit backtick precision marker.
    assert!(
      result.contains("`20."),
      "expected `20. precision marker, got `{result}`"
    );
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

  #[test]
  fn nintegrate_semi_infinite() {
    // ∫₀^∞ e^(-x²) dx = √π/2 ≈ 0.8862269254527580
    assert_approx(
      "NIntegrate[Exp[-x^2], {x, 0, Infinity}]",
      0.886226925452758,
      1e-6,
    );
  }

  #[test]
  fn nintegrate_fully_infinite() {
    // ∫_{-∞}^∞ e^(-x²) dx = √π ≈ 1.7724538509055159
    assert_approx(
      "NIntegrate[Exp[-x^2], {x, -Infinity, Infinity}]",
      1.7724538509055159,
      1e-6,
    );
  }

  // A convergent oscillatory integral over a semi-infinite domain.
  // ∫₀^∞ e^(-x) sin(x) dx = 1/2.
  #[test]
  fn nintegrate_oscillatory_semi_infinite() {
    assert_approx("NIntegrate[Exp[-x] Sin[x], {x, 0, Infinity}]", 0.5, 1e-6);
  }

  // A non-converging oscillatory integrand (Sin[x]/x over [0, ∞) transformed
  // onto a finite interval) must not hang: the adaptive quadrature is bounded
  // by a node budget, so N[...] terminates and returns a real number.
  #[test]
  fn nintegrate_non_converging_terminates() {
    assert_eq!(
      interpret("NumberQ[N[Integrate[Sin[x]/x, {x, 0, Infinity}]]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn nintegrate_infinite_rational() {
    // ∫_{-∞}^∞ 1/(1+x²) dx = π
    assert_approx(
      "NIntegrate[1/(1 + x^2), {x, -Infinity, Infinity}]",
      std::f64::consts::PI,
      1e-6,
    );
  }

  #[test]
  fn nintegrate_with_options() {
    // NIntegrate should accept and use options like Tolerance and Method
    assert_approx(
      "NIntegrate[Exp[-x],{x,0,Infinity},Tolerance->1*^-6, Method->\"GaussLegendre\"]",
      1.0,
      1e-5,
    );
  }

  #[test]
  fn nintegrate_neg_infinity_with_options() {
    assert_approx(
      "NIntegrate[Exp[x],{x,-Infinity, 0},Tolerance->1*^-6, Method->\"GaussLegendre\"]",
      1.0,
      1e-5,
    );
  }

  #[test]
  fn nintegrate_gaussian_with_options() {
    // ∫_{-∞}^∞ e^(-x²/2) dx = √(2π) ≈ 2.5066
    assert_approx(
      "NIntegrate[Exp[-x^2/2.],{x,-Infinity, Infinity},Tolerance->1*^-6, Method->\"GaussLegendre\"]",
      2.5066282746310002,
      1e-3,
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
  fn sec_one_point_zero_machine_real() {
    // Sec[1.] = 1/Cos[1.] ≈ 1.8508157176809255. The last ULP is platform-
    // dependent (system libm differs across OSes; Linux CI gives ...257), so
    // compare numerically rather than by exact string.
    let val: f64 = interpret("Sec[1.]").unwrap().parse().unwrap();
    assert!((val - 1.8508157176809255).abs() < 1e-12);
  }

  #[test]
  fn csc_one_point_zero_machine_real() {
    // Csc[1.] = 1/Sin[1.] ≈ 1.1883951057781212 (matches wolframscript).
    assert_eq!(interpret("Csc[1.]").unwrap(), "1.1883951057781212");
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
  fn erf_negative_rational_coefficient() {
    // The odd-function fold also handles negative rational coefficients.
    assert_eq!(interpret("Erf[-1/2 x]").unwrap(), "-Erf[x/2]");
    assert_eq!(interpret("Erf[-2/3 x]").unwrap(), "-Erf[(2*x)/3]");
    assert_eq!(interpret("Erf[-1/2]").unwrap(), "-Erf[1/2]");
    assert_eq!(interpret("Erfi[-1/2 x]").unwrap(), "-Erfi[x/2]");
    assert_eq!(interpret("FresnelS[-1/2 x]").unwrap(), "-FresnelS[x/2]");
    assert_eq!(interpret("FresnelC[-1/2 x]").unwrap(), "-FresnelC[x/2]");
  }

  #[test]
  fn erf_infinity() {
    assert_eq!(interpret("Erf[Infinity]").unwrap(), "1");
    assert_eq!(interpret("Erf[-Infinity]").unwrap(), "-1");
  }

  #[test]
  fn erfc_zero() {
    assert_eq!(interpret("Erfc[0]").unwrap(), "1");
  }

  #[test]
  fn erfc_infinity() {
    assert_eq!(interpret("Erfc[Infinity]").unwrap(), "0");
    assert_eq!(interpret("Erfc[-Infinity]").unwrap(), "2");
  }

  #[test]
  fn erfc_symbolic() {
    assert_eq!(interpret("Erfc[x]").unwrap(), "Erfc[x]");
  }

  #[test]
  fn erf_two_arg() {
    assert_eq!(
      interpret("{Erf[0, x], Erf[x, 0]}").unwrap(),
      "{Erf[x], -Erf[x]}"
    );
  }

  // Numeric Erf[a, b] uses the complementary identity Erfc[a] - Erfc[b] for
  // same-signed arguments, avoiding the catastrophic cancellation of the naive
  // Erf[b] - Erf[a] (both near +/-1 for large |a|, |b|). Compared via an
  // integer projection so the result matches wolframscript to full precision
  // (the naive form lost ~4 digits: Round[10^14 Erf[2.,3.]] was 465564448402).
  #[test]
  fn erf_two_arg_numeric_precision() {
    assert_eq!(
      interpret("Round[10^14 Erf[2., 3.]]").unwrap(),
      "465564448405"
    );
    assert_eq!(
      interpret("Round[10^15 Erf[3., 4.]]").unwrap(),
      "22075079741"
    );
    // Negative same-signed arguments use the mirrored identity.
    assert_eq!(
      interpret("Round[10^14 Erf[-3., -2.]]").unwrap(),
      "465564448405"
    );
    // Straddling zero keeps the direct Erf form (no cancellation there).
    assert_eq!(
      interpret("Round[10^14 Erf[0.1, 0.2]]").unwrap(),
      "11023967319219"
    );
  }

  // A single inexact argument numericizes the whole two-argument form, even
  // when the other argument is an exact integer (Erf[1, 2.] = Erf[2.] -
  // Erf[1.]). Projected to an integer for a full-precision match.
  #[test]
  fn erf_two_arg_mixed_exactness_numeric() {
    assert_eq!(
      interpret("Round[10^13 Erf[1, 2.]]").unwrap(),
      "1526214720692"
    );
    assert_eq!(
      interpret("Round[10^13 Erf[1., 2]]").unwrap(),
      "1526214720692"
    );
    assert_eq!(
      interpret("Round[10^13 Erf[2., 1]]").unwrap(),
      "-1526214720692"
    );
  }

  // Wolfram keeps the generalized two-argument Erf symbolic instead of
  // rewriting it to a difference of one-argument Erfs.
  #[test]
  fn erf_two_arg_symbolic() {
    assert_eq!(interpret("Erf[1, 2]").unwrap(), "Erf[1, 2]");
    assert_eq!(interpret("Erf[a, b]").unwrap(), "Erf[a, b]");
    // Erf[z, z] = 0.
    assert_eq!(interpret("Erf[x, x]").unwrap(), "0");
  }

  // D[Erf[z0, z1], x] = 2/(Sqrt[Pi] E^(z1^2)) z1' - 2/(Sqrt[Pi] E^(z0^2)) z0'.
  #[test]
  fn d_erf_two_arg() {
    assert_eq!(interpret("D[Erf[a, x], x]").unwrap(), "2/(E^x^2*Sqrt[Pi])");
    assert_eq!(interpret("D[Erf[x, b], x]").unwrap(), "-2/(E^x^2*Sqrt[Pi])");
  }

  #[test]
  fn erfc_negative_arg() {
    // Wolfram keeps Erfc[-x] unevaluated (no symbolic 2 - Erfc[x] rewrite).
    assert_eq!(interpret("Erfc[-x] / 2").unwrap(), "Erfc[-x]/2");
  }

  #[test]
  fn inverse_erfc_special_values() {
    assert_eq!(
      interpret("InverseErfc /@ {0, 1, 2}").unwrap(),
      "{Infinity, 0, -Infinity}"
    );
  }

  #[test]
  fn d_erf_x() {
    // D[Erf[x], x] = 2/(E^(x^2)*Sqrt[Pi]) — tests denominator formatting
    assert_eq!(interpret("D[Erf[x],x]").unwrap(), "2/(E^x^2*Sqrt[Pi])");
  }

  // Fresnel integrals: D[FresnelS[z]] = Sin[(Pi z^2)/2], etc.
  #[test]
  fn d_fresnel() {
    assert_eq!(interpret("D[FresnelS[x], x]").unwrap(), "Sin[(Pi*x^2)/2]");
    assert_eq!(interpret("D[FresnelC[x], x]").unwrap(), "Cos[(Pi*x^2)/2]");
    assert_eq!(interpret("D[FresnelS[2 x], x]").unwrap(), "2*Sin[2*Pi*x^2]");
    assert_eq!(
      interpret("D[FresnelS[x^2], x]").unwrap(),
      "2*x*Sin[(Pi*x^4)/2]"
    );
  }

  #[test]
  fn d_log_gamma_and_log_integral() {
    assert_eq!(interpret("D[LogGamma[x], x]").unwrap(), "PolyGamma[0, x]");
    assert_eq!(
      interpret("D[LogGamma[x^2], x]").unwrap(),
      "2*x*PolyGamma[0, x^2]"
    );
    assert_eq!(interpret("D[LogIntegral[x], x]").unwrap(), "Log[x]^(-1)");
    assert_eq!(interpret("D[LogIntegral[2 x], x]").unwrap(), "2/Log[2*x]");
  }

  // The trig/hyperbolic exponential integrals. D[SinIntegral] prints with the
  // named Sinc; the others give f[z]/z. Verified against wolframscript.
  #[test]
  fn d_si_ci_shi_chi() {
    assert_eq!(interpret("D[SinIntegral[x], x]").unwrap(), "Sinc[x]");
    assert_eq!(interpret("D[CosIntegral[x], x]").unwrap(), "Cos[x]/x");
    assert_eq!(interpret("D[SinhIntegral[x], x]").unwrap(), "Sinh[x]/x");
    assert_eq!(interpret("D[CoshIntegral[x], x]").unwrap(), "Cosh[x]/x");
    // Chain rule.
    assert_eq!(
      interpret("D[SinIntegral[x^2], x]").unwrap(),
      "2*x*Sinc[x^2]"
    );
    assert_eq!(interpret("D[CosIntegral[2 x], x]").unwrap(), "Cos[2*x]/x");
    assert_eq!(
      interpret("D[CoshIntegral[x^2], x]").unwrap(),
      "(2*Cosh[x^2])/x"
    );
    // A constant argument differentiates to 0.
    assert_eq!(interpret("D[SinIntegral[a], x]").unwrap(), "0");
  }

  // Factorial[z] = Gamma[1 + z], so D[z!, z] = Gamma[1 + z] PolyGamma[0, 1 + z].
  // The bare product's factor order is the long-standing Times-ordering
  // divergence (Woxi emits PolyGamma·Gamma, wolframscript Gamma·PolyGamma), so
  // these assertions use order-independent forms that match wolframscript
  // exactly: the structural difference (→ 0), and evaluation at integer points.
  #[test]
  fn d_factorial() {
    // Structural correctness, independent of factor ordering.
    assert_eq!(
      interpret("Simplify[D[x!, x] - Gamma[1 + x] PolyGamma[0, 1 + x]]")
        .unwrap(),
      "0"
    );
    // At x = 0: Gamma[1] PolyGamma[0, 1] = -EulerGamma.
    assert_eq!(interpret("D[x!, x] /. x -> 0").unwrap(), "-EulerGamma");
    // At x = 3: Gamma[4] PolyGamma[0, 4] = 6 (11/6 - EulerGamma).
    assert_eq!(
      interpret("D[x!, x] /. x -> 3").unwrap(),
      "6*(11/6 - EulerGamma)"
    );
    // Chain rule through a constant multiple: D[(2 n)!, n] at n = 0
    // = 2 Gamma[1] PolyGamma[0, 1] = -2 EulerGamma.
    assert_eq!(
      interpret("D[(2 n)!, n] /. n -> 0").unwrap(),
      "-2*EulerGamma"
    );
    // A constant factorial differentiates to 0.
    assert_eq!(interpret("D[3!, x]").unwrap(), "0");
  }

  // PolyGamma[n, z]:  D[PolyGamma[n, z], z] = PolyGamma[n+1, z].
  // PolyGamma[z] is the digamma PolyGamma[0, z], so its derivative is
  // PolyGamma[1, z].  Differentiating w.r.t. the order n has no elementary
  // form and must stay an unevaluated Derivative.
  #[test]
  fn d_polygamma() {
    assert_eq!(interpret("D[PolyGamma[x], x]").unwrap(), "PolyGamma[1, x]");
    assert_eq!(
      interpret("D[PolyGamma[2, x], x]").unwrap(),
      "PolyGamma[3, x]"
    );
    assert_eq!(
      interpret("D[PolyGamma[n, x], x]").unwrap(),
      "PolyGamma[1 + n, x]"
    );
    // Chain rule.
    assert_eq!(
      interpret("D[PolyGamma[x^2], x]").unwrap(),
      "2*x*PolyGamma[1, x^2]"
    );
    assert_eq!(
      interpret("D[PolyGamma[3, x^2], x]").unwrap(),
      "2*x*PolyGamma[4, x^2]"
    );
    // Derivative w.r.t. the order stays unevaluated.
    assert_eq!(
      interpret("D[PolyGamma[a, x], a]").unwrap(),
      "Derivative[1, 0][PolyGamma][a, x]"
    );
  }

  // Incomplete Beta: D[Beta[z, a, b], z] = z^(a-1) (1-z)^(b-1).
  #[test]
  fn d_incomplete_beta() {
    assert_eq!(
      interpret("D[Beta[x, a, b], x]").unwrap(),
      "(1 - x)^(-1 + b)*x^(-1 + a)"
    );
  }

  // Hypergeometric derivatives:
  //   D[Hypergeometric1F1[a, b, z], z] = (a/b) Hypergeometric1F1[a+1, b+1, z]
  //   D[Hypergeometric2F1[a, b, c, z], z] =
  //     (a b / c) Hypergeometric2F1[a+1, b+1, c+1, z]
  #[test]
  fn d_hypergeometric() {
    assert_eq!(
      interpret("D[Hypergeometric1F1[a, b, x], x]").unwrap(),
      "(a*Hypergeometric1F1[1 + a, 1 + b, x])/b"
    );
    assert_eq!(
      interpret("D[Hypergeometric2F1[a, b, c, x], x]").unwrap(),
      "(a*b*Hypergeometric2F1[1 + a, 1 + b, 1 + c, x])/c"
    );
    // Chain rule through a linear inner argument.
    assert_eq!(
      interpret("D[Hypergeometric1F1[a, b, x^2], x]").unwrap(),
      "(2*a*x*Hypergeometric1F1[1 + a, 1 + b, x^2])/b"
    );
  }

  #[test]
  fn d_airy() {
    assert_eq!(interpret("D[AiryAi[x], x]").unwrap(), "AiryAiPrime[x]");
    assert_eq!(interpret("D[AiryBi[x], x]").unwrap(), "AiryBiPrime[x]");
    assert_eq!(
      interpret("D[AiryAi[2 x], x]").unwrap(),
      "2*AiryAiPrime[2*x]"
    );
  }

  // AiryAiPrime/AiryBiPrime satisfy the Airy ODE y'' = z y.
  #[test]
  fn d_airy_prime() {
    assert_eq!(interpret("D[AiryAiPrime[x], x]").unwrap(), "x*AiryAi[x]");
    assert_eq!(interpret("D[AiryBiPrime[x], x]").unwrap(), "x*AiryBi[x]");
    assert_eq!(
      interpret("D[AiryAiPrime[2 x], x]").unwrap(),
      "4*x*AiryAi[2*x]"
    );
  }

  // Incomplete elliptic integrals differentiated w.r.t. the amplitude:
  //   D[EllipticF[phi, m], phi] = 1/Sqrt[1 - m Sin[phi]^2]
  //   D[EllipticE[phi, m], phi] = Sqrt[1 - m Sin[phi]^2]
  #[test]
  fn d_elliptic_incomplete() {
    assert_eq!(
      interpret("D[EllipticF[phi, m], phi]").unwrap(),
      "1/Sqrt[1 - m*Sin[phi]^2]"
    );
    assert_eq!(
      interpret("D[EllipticE[phi, m], phi]").unwrap(),
      "Sqrt[1 - m*Sin[phi]^2]"
    );
    // Chain rule through the amplitude.
    assert_eq!(
      interpret("D[EllipticF[2 x, m], x]").unwrap(),
      "2/Sqrt[1 - m*Sin[2*x]^2]"
    );
    assert_eq!(
      interpret("D[EllipticE[2 x, m], x]").unwrap(),
      "2*Sqrt[1 - m*Sin[2*x]^2]"
    );
    // A variable that appears in neither argument gives 0.
    assert_eq!(interpret("D[EllipticF[phi, m], a]").unwrap(), "0");
  }

  // PolyLog[n, z]' = PolyLog[n-1, z]/z; ExpIntegralE[n, z]' = -ExpIntegralE[n-1, z].
  #[test]
  fn d_polylog_and_exp_integral_e() {
    assert_eq!(interpret("D[PolyLog[2, x], x]").unwrap(), "-(Log[1 - x]/x)");
    assert_eq!(interpret("D[PolyLog[3, x], x]").unwrap(), "PolyLog[2, x]/x");
    assert_eq!(
      interpret("D[PolyLog[n, x], x]").unwrap(),
      "PolyLog[-1 + n, x]/x"
    );
    assert_eq!(
      interpret("D[ExpIntegralE[n, x], x]").unwrap(),
      "-ExpIntegralE[-1 + n, x]"
    );
    assert_eq!(
      interpret("D[ExpIntegralE[2, x], x]").unwrap(),
      "-ExpIntegralE[1, x]"
    );
  }

  // Ramp[z] is flat below 0 and the identity above, with a corner at 0, so
  // D[Ramp[z], z] = Piecewise[{{0, z < 0}, {1, z > 0}}, Indeterminate].
  // (A composite argument keeps the chain-rule factor; Woxi does not yet
  // simplify the Piecewise conditions the way wolframscript does, so only the
  // direct form is asserted here.)
  #[test]
  fn d_ramp() {
    assert_eq!(
      interpret("D[Ramp[x], x]").unwrap(),
      "Piecewise[{{0, x < 0}, {1, x > 0}}, Indeterminate]"
    );
    // A constant argument differentiates to 0.
    assert_eq!(interpret("D[Ramp[5], x]").unwrap(), "0");
  }

  // wolframscript keeps the derivative of Sign as the unevaluated Sign'
  // (Derivative[1][Sign]), not the equivalent Abs'' (Derivative[2][Abs]).
  #[test]
  fn d_sign() {
    assert_eq!(
      interpret("D[Sign[x], x]").unwrap(),
      "Derivative[1][Sign][x]"
    );
    // Sign[2 x] simplifies to Sign[x] (positive scaling), so the result matches.
    assert_eq!(
      interpret("D[Sign[2 x], x]").unwrap(),
      "Derivative[1][Sign][x]"
    );
    // Composite argument keeps the chain-rule factor.
    assert_eq!(
      interpret("D[Sign[x^2], x]").unwrap(),
      "2*Sign[x]*Derivative[1][Sign][x]"
    );
    // Product rule.
    assert_eq!(
      interpret("D[x Sign[x], x]").unwrap(),
      "x*Derivative[1][Sign][x] + Sign[x]"
    );
    // Second derivative differentiates Sign' to Sign''.
    assert_eq!(
      interpret("D[Sign[x], {x, 2}]").unwrap(),
      "Derivative[2][Sign][x]"
    );
    // A constant argument differentiates to 0.
    assert_eq!(interpret("D[Sign[7], x]").unwrap(), "0");
  }

  // wolframscript keeps the derivative of Abs as the unevaluated Abs'
  // (Derivative[1][Abs]) because Abs is non-analytic, rather than Sign[x].
  #[test]
  fn d_abs() {
    assert_eq!(interpret("D[Abs[x], x]").unwrap(), "Derivative[1][Abs][x]");
    // Composite argument keeps the chain-rule factor (with positive scaling
    // 3x simplified to x, mirroring D[Abs[3 x], x] = 3 Abs'[x]).
    assert_eq!(
      interpret("D[Abs[3 x], x]").unwrap(),
      "3*Derivative[1][Abs][x]"
    );
    // Product rule.
    assert_eq!(
      interpret("D[x Abs[x], x]").unwrap(),
      "Abs[x] + x*Derivative[1][Abs][x]"
    );
    // Second derivative differentiates Abs' to Abs''.
    assert_eq!(
      interpret("D[Abs[x], {x, 2}]").unwrap(),
      "Derivative[2][Abs][x]"
    );
    // A constant argument differentiates to 0.
    assert_eq!(interpret("D[Abs[7], x]").unwrap(), "0");
    // Limits over the reals still resolve Abs'/Sign internally (regression):
    // Abs[x]/x -> -1 from below, +1 from above.
    assert_eq!(
      interpret("Limit[Abs[x]/x, x -> 0, Direction -> \"FromBelow\"]").unwrap(),
      "-1"
    );
  }

  // D[HeavisideTheta[z], z] = DiracDelta[z], with the chain rule for a
  // composite argument. Regression: this used to emit a garbage
  // Derivative[1][HeavisideTheta] from the generic chain rule.
  #[test]
  fn d_heaviside_theta() {
    assert_eq!(
      interpret("D[HeavisideTheta[x], x]").unwrap(),
      "DiracDelta[x]"
    );
    // A pure-scaling argument folds via the DiracDelta scaling law:
    // 2 DiracDelta[2 x] = 2 (DiracDelta[x]/2) = DiracDelta[x].
    assert_eq!(
      interpret("D[HeavisideTheta[2 x], x]").unwrap(),
      "DiracDelta[x]"
    );
    // Chain rule with a linear shift.
    assert_eq!(
      interpret("D[HeavisideTheta[x - 2], x]").unwrap(),
      "DiracDelta[-2 + x]"
    );
    // Chain rule with a nonlinear argument keeps the derivative factor.
    assert_eq!(
      interpret("D[HeavisideTheta[x^2 - 1], x]").unwrap(),
      "2*x*DiracDelta[-1 + x^2]"
    );
    // Constant multiple carries through.
    assert_eq!(
      interpret("D[3 HeavisideTheta[x], x]").unwrap(),
      "3*DiracDelta[x]"
    );
    // A constant argument differentiates to 0.
    assert_eq!(interpret("D[HeavisideTheta[a], x]").unwrap(), "0");
  }

  // KroneckerDelta is piecewise-constant, so its derivative is 0 for any
  // argument list. Regression: this used to emit Derivative[1][KroneckerDelta].
  #[test]
  fn d_kronecker_delta() {
    assert_eq!(interpret("D[KroneckerDelta[x], x]").unwrap(), "0");
    assert_eq!(interpret("D[KroneckerDelta[x, 2], x]").unwrap(), "0");
    assert_eq!(interpret("D[KroneckerDelta[x^2], x]").unwrap(), "0");
  }

  // Floor/Ceiling are locally constant with jumps at the integers:
  //   D[Floor[u], x]   = D[u, x] Piecewise[{{0, u > Floor[u]}},   Indeterminate]
  //   D[Ceiling[u], x] = D[u, x] Piecewise[{{0, u < Ceiling[u]}}, Indeterminate]
  // Regression: these used to emit Derivative[1][Floor]/[Ceiling].
  #[test]
  fn d_floor_ceiling() {
    assert_eq!(
      interpret("D[Floor[x], x]").unwrap(),
      "Piecewise[{{0, x > Floor[x]}}, Indeterminate]"
    );
    assert_eq!(
      interpret("D[Ceiling[x], x]").unwrap(),
      "Piecewise[{{0, x < Ceiling[x]}}, Indeterminate]"
    );
    // Chain-rule factor from a linear argument.
    assert_eq!(
      interpret("D[Floor[2 x], x]").unwrap(),
      "2*Piecewise[{{0, 2*x > Floor[2*x]}}, Indeterminate]"
    );
    // Chain-rule factor from a nonlinear argument.
    assert_eq!(
      interpret("D[Floor[x^2], x]").unwrap(),
      "2*x*Piecewise[{{0, x^2 > Floor[x^2]}}, Indeterminate]"
    );
    // A constant argument differentiates to 0.
    assert_eq!(interpret("D[Floor[5], x]").unwrap(), "0");
  }

  // The generic chain rule for an unknown function with a list-valued argument
  // must give a single Derivative[…][f][…] (with a structurally-matching list
  // of zero indices for the constant list argument), not a malformed list of
  // repeated terms.
  #[test]
  fn d_unknown_function_with_list_argument() {
    assert_eq!(
      interpret("D[f[x, {1, 2, 3}], x]").unwrap(),
      "Derivative[1, {0, 0, 0}][f][x, {1, 2, 3}]"
    );
    assert_eq!(
      interpret("D[g[{1, 2}, x], x]").unwrap(),
      "Derivative[{0, 0}, 1][g][{1, 2}, x]"
    );
    assert_eq!(
      interpret("D[f[x, {a, b}], x]").unwrap(),
      "Derivative[1, {0, 0}][f][x, {a, b}]"
    );
    // Scalar-only arguments are unaffected.
    assert_eq!(
      interpret("D[f[x, y], x]").unwrap(),
      "Derivative[1, 0][f][x, y]"
    );
  }

  // The same chain rule for an unknown function applied to N arguments must
  // also work when the function's head is itself a compound expression
  // (`Subscript[c, A]`, an `InterpolatingFunction[…]`) rather than a plain
  // symbol — a shape a PDE's dependent-variable list commonly takes
  // (`NDSolve[…, {Subscript[c, A], Subscript[c, B]}, …]`). Previously only
  // a compound head applied to exactly one argument differentiated; two or
  // more left the whole `D[…]` call unevaluated, which then produced
  // corrupted results downstream when a `ReplaceAll` following it
  // substituted through the still-unevaluated call's own arguments (both
  // the differentiation variable's slot and the function's own argument).
  #[test]
  fn d_unknown_compound_head_function_with_multiple_arguments() {
    assert_eq!(
      interpret("D[Subscript[c, A][x, y], x]").unwrap(),
      "Derivative[1, 0][Subscript[c, A]][x, y]"
    );
    assert_eq!(
      interpret("D[Subscript[c, A][x, y], y]").unwrap(),
      "Derivative[0, 1][Subscript[c, A]][x, y]"
    );
    // A single argument still works (the previously-supported case).
    assert_eq!(
      interpret("D[Subscript[c, A][x], x]").unwrap(),
      "Derivative[1][Subscript[c, A]][x]"
    );
    // Differentiating then substituting must differentiate first: the
    // regression this fixes had `x` in `D[…, x] /. x -> 1` get replaced
    // inside the still-unevaluated `D[…]` before it could differentiate,
    // corrupting both the function's argument and the derivative order.
    assert_eq!(
      interpret("D[Subscript[c, A][x, y], x] /. x -> 1").unwrap(),
      "Derivative[1, 0][Subscript[c, A]][1, y]"
    );
  }

  // Inverse error functions: D[InverseErf[z]] = (Sqrt[Pi]/2) E^(InverseErf[z]^2).
  #[test]
  fn d_inverse_erf() {
    assert_eq!(
      interpret("D[InverseErf[x], x]").unwrap(),
      "(E^InverseErf[x]^2*Sqrt[Pi])/2"
    );
    assert_eq!(
      interpret("D[InverseErf[x^2], x]").unwrap(),
      "E^InverseErf[x^2]^2*Sqrt[Pi]*x"
    );
  }

  #[test]
  fn d_inverse_erfc() {
    assert_eq!(
      interpret("D[InverseErfc[x], x]").unwrap(),
      "-1/2*(E^InverseErfc[x]^2*Sqrt[Pi])"
    );
    assert_eq!(
      interpret("D[InverseErfc[2 x], x]").unwrap(),
      "-(E^InverseErfc[2*x]^2*Sqrt[Pi])"
    );
  }

  #[test]
  fn n_erf_1() {
    // N[Erf[1], 20] — small argument, Taylor series path
    let result = interpret("N[Erf[1], 20]").unwrap();
    assert!(
      result.starts_with("0.84270079294971486934"),
      "N[Erf[1], 20] = {result}"
    );
  }

  #[test]
  fn n_erf_5() {
    // N[Erf[5], 20] — large argument, continued fraction path
    let result = interpret("N[Erf[5], 20]").unwrap();
    assert!(
      result.starts_with("0.99999999999846254020"),
      "N[Erf[5], 20] = {result}"
    );
  }

  #[test]
  fn n_erf_10() {
    // N[Erf[10], 20] — very large argument, result is 1 to 20 digits
    assert_eq!(interpret("N[Erf[10], 20]").unwrap(), "1.`20.");
  }

  #[test]
  fn n_erf_neg_10() {
    // Erf is odd: N[Erf[-10], 20] = -1
    assert_eq!(interpret("N[Erf[-10], 20]").unwrap(), "-1.`20.");
  }

  #[test]
  fn n_erfc_5() {
    // N[Erfc[5], 20] — continued fraction path
    let result = interpret("N[Erfc[5], 20]").unwrap();
    assert!(
      result.starts_with("1.53745979442803485018"),
      "N[Erfc[5], 20] = {result}"
    );
  }

  #[test]
  fn n_erfc_10() {
    // N[Erfc[10], 20] — very small result
    let result = interpret("N[Erfc[10], 20]").unwrap();
    assert!(
      result.starts_with("2.08848758376254475700"),
      "N[Erfc[10], 20] = {result}"
    );
  }

  #[test]
  fn n_erf_3() {
    // N[Erf[3], 20] — moderate argument, Taylor series path
    let result = interpret("N[Erf[3], 20]").unwrap();
    assert!(
      result.starts_with("0.99997790950300141455"),
      "N[Erf[3], 20] = {result}"
    );
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

  // A positive x^2 coefficient gives the Erfi branch. Verified against
  // wolframscript.
  #[test]
  fn integrate_exp_pos_x_squared_gives_erfi() {
    assert_eq!(
      interpret("Integrate[Exp[x^2], x]").unwrap(),
      "(Sqrt[Pi]*Erfi[x])/2"
    );
    assert_eq!(
      interpret("Integrate[Exp[2 x^2], x]").unwrap(),
      "(Sqrt[Pi/2]*Erfi[Sqrt[2]*x])/2"
    );
    assert_eq!(
      interpret("Integrate[Exp[a x^2], x]").unwrap(),
      "(Sqrt[Pi]*Erfi[Sqrt[a]*x])/(2*Sqrt[a])"
    );
  }
}

// ∫ 1/Log[x] dx = LogIntegral[x], the reverse of D[LogIntegral[x]] = 1/Log[x].
// Verified against wolframscript.
mod integrate_log_integral {
  use super::*;

  #[test]
  fn one_over_log() {
    assert_eq!(
      interpret("Integrate[1/Log[x], x]").unwrap(),
      "LogIntegral[x]"
    );
    assert_eq!(
      interpret("Integrate[Log[x]^(-1), x]").unwrap(),
      "LogIntegral[x]"
    );
  }
}

// ∫ Sin[a x^2] dx = Sqrt[Pi/2]/Sqrt[a] FresnelS[Sqrt[a] Sqrt[2/Pi] x]
// (and FresnelC for Cos). Verified against wolframscript.
mod integrate_fresnel {
  use super::*;

  #[test]
  fn unit_coefficient() {
    assert_eq!(
      interpret("Integrate[Sin[x^2], x]").unwrap(),
      "Sqrt[Pi/2]*FresnelS[Sqrt[2/Pi]*x]"
    );
    assert_eq!(
      interpret("Integrate[Cos[x^2], x]").unwrap(),
      "Sqrt[Pi/2]*FresnelC[Sqrt[2/Pi]*x]"
    );
  }

  #[test]
  fn integer_coefficient() {
    assert_eq!(
      interpret("Integrate[Cos[3 x^2], x]").unwrap(),
      "Sqrt[Pi/6]*FresnelC[Sqrt[6/Pi]*x]"
    );
    // A perfect-square 2a collapses the radicals into Wolfram's mixed form.
    assert_eq!(
      interpret("Integrate[Sin[2 x^2], x]").unwrap(),
      "(Sqrt[Pi]*FresnelS[(2*x)/Sqrt[Pi]])/2"
    );
  }

  #[test]
  fn canonical_argument() {
    // The Fresnel convention: ∫ Sin[Pi x^2/2] dx = FresnelS[x].
    assert_eq!(
      interpret("Integrate[Sin[Pi x^2/2], x]").unwrap(),
      "FresnelS[x]"
    );
    assert_eq!(
      interpret("Integrate[Cos[Pi x^2/2], x]").unwrap(),
      "FresnelC[x]"
    );
  }

  #[test]
  fn symbolic_coefficient() {
    assert_eq!(
      interpret("Integrate[Sin[a x^2], x]").unwrap(),
      "(Sqrt[Pi/2]*FresnelS[Sqrt[a]*Sqrt[2/Pi]*x])/Sqrt[a]"
    );
  }
}

mod erfi {
  use super::*;

  #[test]
  fn erfi_zero() {
    assert_eq!(interpret("Erfi[0]").unwrap(), "0");
  }

  #[test]
  fn erfi_symbolic() {
    assert_eq!(interpret("Erfi[x]").unwrap(), "Erfi[x]");
  }

  #[test]
  fn erfi_negative_arg() {
    // Erfi[-x] = -Erfi[x] (odd function)
    assert_eq!(interpret("Erfi[-x]").unwrap(), "-Erfi[x]");
  }

  #[test]
  fn erfi_negative_integer() {
    // Erfi[-1] = -Erfi[1]
    assert_eq!(interpret("Erfi[-1]").unwrap(), "-Erfi[1]");
  }

  #[test]
  fn erfi_infinity() {
    assert_eq!(interpret("Erfi[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn erfi_neg_infinity() {
    assert_eq!(interpret("Erfi[-Infinity]").unwrap(), "-Infinity");
  }

  #[test]
  fn erfi_real() {
    // Erfi[1.0] ≈ 1.6504257587975429
    let result = interpret("Erfi[1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val - 1.6504257587975429).abs() < 1e-10,
      "Erfi[1.0] = {result}"
    );
  }

  #[test]
  fn erfi_real_negative() {
    // Erfi[-1.0] = -Erfi[1.0]
    let result = interpret("Erfi[-1.0]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      (val + 1.6504257587975429).abs() < 1e-10,
      "Erfi[-1.0] = {result}"
    );
  }

  #[test]
  fn erfi_listable() {
    assert_eq!(interpret("Erfi[{0, x}]").unwrap(), "{0, Erfi[x]}");
  }

  #[test]
  fn d_erfi_x() {
    // D[Erfi[x], x] = 2*E^(x^2)/Sqrt[Pi]
    assert_eq!(interpret("D[Erfi[x],x]").unwrap(), "(2*E^x^2)/Sqrt[Pi]");
  }

  #[test]
  fn n_erfi_1() {
    // N[Erfi[1], 20] — small argument, Taylor series
    let result = interpret("N[Erfi[1], 20]").unwrap();
    assert!(
      result.starts_with("1.65042575879754287602"),
      "N[Erfi[1], 20] = {result}"
    );
  }

  #[test]
  fn n_erfi_0() {
    let result = interpret("N[Erfi[0], 20]").unwrap();
    assert!(result.starts_with('0'), "N[Erfi[0], 20] = {result}");
  }

  #[test]
  fn n_erfi_half() {
    // N[Erfi[1/2], 20] ≈ 0.61427...
    let result = interpret("N[Erfi[1/2], 20]").unwrap();
    assert!(
      result.starts_with("0.61495209469651"),
      "N[Erfi[1/2], 20] = {result}"
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

mod mixed_partial_derivatives {
  use super::*;

  #[test]
  fn d_two_variables() {
    assert_eq!(interpret("D[x^2 y, x, y]").unwrap(), "2*x");
  }

  #[test]
  fn d_two_variables_higher() {
    assert_eq!(interpret("D[x^2 y^3, x, y]").unwrap(), "6*x*y^2");
  }

  #[test]
  fn d_same_variable_twice() {
    assert_eq!(interpret("D[x^3, x, x]").unwrap(), "6*x");
  }

  #[test]
  fn d_three_variables() {
    assert_eq!(interpret("D[x^2 y^3, x, x, y]").unwrap(), "6*y^2");
  }

  #[test]
  fn d_mixed_with_list_spec() {
    assert_eq!(interpret("D[x^2 y^3, {x, 2}, y]").unwrap(), "6*y^2");
  }

  #[test]
  fn d_mixed_trig() {
    let result = interpret("D[Sin[x] Cos[y], x, y]").unwrap();
    assert!(
      result == "-(Cos[x]*Sin[y])" || result == "-Sin[y]*Cos[x]",
      "Got: {result}"
    );
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
  fn find_minimum_bare_symbol_default_start() {
    // FindMinimum[f, x] uses Wolfram's automatic starting point x = 1
    clear_state();
    let result = interpret("FindMinimum[(x - 3)^2, x]").unwrap();
    assert_eq!(result, "{0., {x -> 3.}}");
  }

  #[test]
  fn find_max_value_univariate() {
    clear_state();
    let result = interpret("FindMaxValue[-2*x^2 - 3*x + 5, x]").unwrap();
    assert_eq!(result, "6.125");
  }

  #[test]
  fn find_max_value_with_start() {
    clear_state();
    let result = interpret("FindMaxValue[Sin[x], {x, 2}]").unwrap();
    assert_eq!(result, "1.");
  }

  #[test]
  fn find_max_value_multivariate_bare_symbols() {
    // {x, y} is a list of variables with automatic starting points,
    // not a {var, start} pair
    clear_state();
    let result = interpret("FindMaxValue[Sin[x]*Sin[2*y], {x, y}]").unwrap();
    assert_eq!(result, "1.");
  }

  #[test]
  fn find_min_value_univariate() {
    clear_state();
    let result = interpret("FindMinValue[x^2 - 3 x + 2, x]").unwrap();
    assert_eq!(result, "-0.25");
  }

  #[test]
  fn find_min_value_with_start() {
    clear_state();
    let result = interpret("FindMinValue[Cos[x], {x, 3}]").unwrap();
    assert_eq!(result, "-1.");
  }

  #[test]
  fn find_min_value_bare_symbol() {
    clear_state();
    let result = interpret("FindMinValue[Sin[x], x]").unwrap();
    assert_eq!(result, "-1.");
  }

  #[test]
  fn find_min_value_multivariate() {
    clear_state();
    let result =
      interpret("FindMinValue[(x - 3)^2 + (y - 2)^2, {{x, 0}, {y, 0}}]")
        .unwrap();
    assert_eq!(result, "0.");
  }

  #[test]
  fn find_maximum_multivariate_newton_convergence() {
    // Regression: plain gradient descent stalled at ~0.9967 within the
    // iteration budget; the damped-Newton step converges to the exact
    // maximum value 1.
    clear_state();
    let result =
      interpret("FindMaximum[Sin[x]*Sin[2*y], {{x, 2}, {y, 1}}]").unwrap();
    assert!(result.starts_with("{1., {x -> 1.5707"));
  }

  #[test]
  fn find_maximum_negative_quadratic() {
    clear_state();
    let result = interpret("FindMaximum[-(x - 5)^2 + 10, {x, 0}]").unwrap();
    assert!(result.starts_with("{10., {x -> 5."));
  }

  #[test]
  fn quadratic_with_tiny_coefficient() {
    // Regression: a quadratic scaled by 10^-30 has gradient ~10^-30
    // and Hessian ~10^-30, so the Newton step (-grad/Hess) is still
    // O(1) and reaches the exact minimum in one iteration. The earlier
    // implementation declared convergence on |grad| < 1e-15 *before*
    // even attempting the step and froze at the starting point.
    clear_state();
    let result = interpret("FindMinimum[10*^-30 *(x-3)^2+2., {x, 1}]").unwrap();
    assert_eq!(result, "{2., {x -> 3.}}");
  }

  #[test]
  fn maximum_with_tiny_coefficient() {
    clear_state();
    let result =
      interpret("FindMaximum[-10*^-30 *(x-3)^2+2., {x, 1}]").unwrap();
    assert_eq!(result, "{2., {x -> 3.}}");
  }

  #[test]
  fn constrained_maximum_reports_that_it_cannot_monitor_steps() {
    // A constrained FindMaximum (the `{f, cons}` form) is solved by a
    // global sampler with a local refinement, not by an iterative method
    // whose steps mean anything to the caller. Wolfram reports `noopmon`
    // for a `StepMonitor :> Sow[...]` on such a call and fires no monitor
    // at all, so Reap comes back with no tag groups.
    clear_state();
    let r = woxi::interpret_with_stdout(
      "p = Reap[FindMaximum[{-((x - 2)^2 + (y - 3)^2), \
       0 < x < 10 && 0 < y < 10}, {{x, 5}, {y, 5}}, \
       StepMonitor :> Sow[{x, y}]]]; Length[p[[2]]]",
    )
    .unwrap();
    assert_eq!(r.result, "0");
    assert!(r.warnings[0].contains(
      "FindMaximum::noopmon: The optimization was solved by an algorithm \
       that does not provide monitoring information."
    ));
  }

  #[test]
  fn constrained_maximum_without_step_monitor_unaffected() {
    // The StepMonitor plumbing must not change the plain constrained
    // FindMaximum result when the option isn't supplied. Rounded because
    // Woxi lands exactly on the optimum where wolframscript's interior
    // point method stops a few nanometres short of it.
    clear_state();
    let result = interpret(
      "sol = FindMaximum[{-((x - 2)^2 + (y - 3)^2), \
       0 < x < 10 && 0 < y < 10}, {{x, 5}, {y, 5}}]; \
       {Round[sol[[1]], 10^-6], Round[x /. sol[[2]], 10^-6], \
       Round[y /. sol[[2]], 10^-6]}",
    )
    .unwrap();
    assert_eq!(result, "{0, 2, 3}");
  }

  #[test]
  fn constrained_maximum_over_black_box_function() {
    // Regression: an objective built on a function Woxi can only evaluate
    // numerically (no symbolic derivative rule, here a distribution's CDF)
    // must not be mistaken for one with a valid symbolic gradient — that
    // silently broke optimization, converging on a fixed grid-search sample
    // instead of refining it. The maximum of CDF[NormalDistribution[],x-c]
    // over a bounded x range is at the upper bound.
    clear_state();
    let result = interpret(
      "s = FindMaximum[{CDF[NormalDistribution[], x - 3], 0 < x < 5}, \
       {x, 1}]; {Round[s[[1]], 10^-4], Round[x /. s[[2]], 10^-4]}",
    )
    .unwrap();
    assert_eq!(result, "{2443/2500, 5}");
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

  // One-argument Dt is the total differential: sum of D[expr, v] * Dt[v] over
  // the free variables v.
  #[test]
  fn total_differential_monomial() {
    assert_eq!(interpret("Dt[x^2]").unwrap(), "2*x*Dt[x]");
    assert_eq!(interpret("Dt[x y]").unwrap(), "y*Dt[x] + x*Dt[y]");
    assert_eq!(
      interpret("Dt[x y z]").unwrap(),
      "y*z*Dt[x] + x*z*Dt[y] + x*y*Dt[z]"
    );
    assert_eq!(interpret("Dt[a x^2]").unwrap(), "x^2*Dt[a] + 2*a*x*Dt[x]");
  }

  #[test]
  fn total_differential_sum_and_quotient() {
    assert_eq!(interpret("Dt[x + y]").unwrap(), "Dt[x] + Dt[y]");
    assert_eq!(interpret("Dt[x^2 + y^2]").unwrap(), "2*x*Dt[x] + 2*y*Dt[y]");
    assert_eq!(interpret("Dt[x/y]").unwrap(), "Dt[x]/y - (x*Dt[y])/y^2");
    assert_eq!(interpret("Dt[Log[x]]").unwrap(), "Dt[x]/x");
    assert_eq!(interpret("Dt[E^x]").unwrap(), "E^x*Dt[x]");
  }

  #[test]
  fn total_differential_constants_and_bare() {
    assert_eq!(interpret("Dt[5]").unwrap(), "0");
    assert_eq!(interpret("Dt[Pi]").unwrap(), "0");
    assert_eq!(interpret("Dt[x]").unwrap(), "Dt[x]");
    assert_eq!(interpret("Dt[c]").unwrap(), "Dt[c]");
  }

  // `Dt[f, {x, n}]` is the n-fold total derivative. Rodrigues's formulas
  // (Demonstrations on the special functions of quantum mechanics are
  // written with them) use exactly this spelling.
  #[test]
  fn higher_order_with_integer_count() {
    assert_eq!(interpret("Dt[x^3, {x, 0}]").unwrap(), "x^3");
    assert_eq!(interpret("Dt[x^3, {x, 1}]").unwrap(), "3*x^2");
    assert_eq!(interpret("Dt[x^2, {x, 2}]").unwrap(), "2");
    assert_eq!(interpret("Dt[x^4, {x, 3}]").unwrap(), "24*x");
    assert_eq!(interpret("Dt[Sin[x], {x, 2}]").unwrap(), "-Sin[x]");
  }

  // A dependent variable's own derivative grows another argument instead of
  // nesting, and the arguments come back in canonical shape: repeats fold
  // into `{x, n}` and the variables are sorted.
  #[test]
  fn higher_order_carries_dependent_variables() {
    assert_eq!(
      interpret("Dt[x y, {x, 2}]").unwrap(),
      "2*Dt[y, x] + x*Dt[y, {x, 2}]"
    );
    assert_eq!(interpret("Dt[Dt[y, x], x]").unwrap(), "Dt[y, {x, 2}]");
    assert_eq!(interpret("Dt[y, x, x, x]").unwrap(), "Dt[y, {x, 3}]");
    assert_eq!(interpret("Dt[Dt[y, x], w]").unwrap(), "Dt[y, w, x]");
    assert_eq!(interpret("Dt[y, x, z, x]").unwrap(), "Dt[y, {x, 2}, z]");
  }

  // A held total derivative does not depend on the symbol it differentiates,
  // so differentiating it by that symbol again gives 0.
  #[test]
  fn held_derivative_of_its_own_symbol_is_zero() {
    assert_eq!(interpret("Dt[Dt[y, x], y]").unwrap(), "0");
    assert_eq!(interpret("Dt[Dt[x, y], x]").unwrap(), "0");
    assert_eq!(interpret("Dt[Dt[z, x], y]").unwrap(), "Dt[z, x, y]");
  }

  // A symbolic order cannot be carried out, so the call is held.
  #[test]
  fn higher_order_with_symbolic_count_is_held() {
    assert_eq!(
      interpret("Dt[E^(-x^2), {x, n}]").unwrap(),
      "Dt[E^(-x^2), {x, n}]"
    );
    assert_eq!(
      interpret("Dt[(x^2 - 1)^n, {x, n}]").unwrap(),
      "Dt[(-1 + x^2)^n, {x, n}]"
    );
  }

  // Dt[f, x1, x2, …] differentiates against each variable in turn.
  #[test]
  fn multiple_variables() {
    assert_eq!(interpret("Dt[x^3, x, x]").unwrap(), "6*x");
    assert_eq!(interpret("Dt[x y, x, y]").unwrap(), "1 + Dt[x, y]*Dt[y, x]");
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

  // The kink of Abs[g(x)] (where g == 0) is the minimizer of a convex objective.
  #[test]
  fn abs_kink_minimum() {
    assert_eq!(
      interpret("Minimize[Abs[x - 3], x]").unwrap(),
      "{0, {x -> 3}}"
    );
    assert_eq!(interpret("Minimize[Abs[x], x]").unwrap(), "{0, {x -> 0}}");
    assert_eq!(
      interpret("Minimize[Abs[x + 2] + 1, x]").unwrap(),
      "{1, {x -> -2}}"
    );
    assert_eq!(
      interpret("Minimize[Abs[2*x - 4], x]").unwrap(),
      "{0, {x -> 2}}"
    );
    assert_eq!(
      interpret("Minimize[3*Abs[x - 1], x]").unwrap(),
      "{0, {x -> 1}}"
    );
  }

  #[test]
  fn abs_kink_maximum() {
    assert_eq!(
      interpret("Maximize[-Abs[x - 5], x]").unwrap(),
      "{0, {x -> 5}}"
    );
    assert_eq!(
      interpret("Maximize[10 - Abs[x], x]").unwrap(),
      "{10, {x -> 0}}"
    );
  }

  // A concave Abs kink is a maximum, not a minimum: the objective runs off to
  // -Infinity, so Minimize reports the unbounded result rather than mistaking
  // the kink for a minimum.
  #[test]
  fn concave_abs_not_reported_as_minimum() {
    assert_eq!(
      interpret("Minimize[-Abs[x], x]").unwrap(),
      "{-Infinity, {x -> -Infinity}}"
    );
    assert_eq!(
      interpret("Minimize[-Abs[x - 5], x]").unwrap(),
      "{-Infinity, {x -> -Infinity}}"
    );
  }

  // Objectives with negative powers (Laurent/rational) must not overflow when
  // extracting polynomial coefficients. x^2 + 1/x^2 has minimum 2 at x = +/-1;
  // the complex roots (+/-I) of the critical equation must be ignored.
  #[test]
  fn rational_objective_ignores_complex_critical_points() {
    assert_eq!(
      interpret("Minimize[x^2 + 1/x^2, x]").unwrap(),
      "{2, {x -> -1}}"
    );
    assert_eq!(
      interpret("Minimize[x^2/(1 + x^2), x]").unwrap(),
      "{0, {x -> 0}}"
    );
  }

  // A non-symbol in the variable slot (a constraint, equation, or literal)
  // emits <func>::ivar and stays unevaluated, rather than raising a hard
  // error.
  #[test]
  fn invalid_variable_emits_ivar() {
    for (input, call, bad) in [
      ("Minimize[x^2, x >= 1]", "Minimize[x^2, x >= 1]", "x >= 1"),
      ("Maximize[x^2, x < 5]", "Maximize[x^2, x < 5]", "x < 5"),
      ("Minimize[x^2, x == 2]", "Minimize[x^2, x == 2]", "x == 2"),
      ("Minimize[x^2, 3]", "Minimize[x^2, 3]", "3"),
    ] {
      clear_state();
      assert_eq!(interpret(input).unwrap(), call, "for {input}");
      let func = if input.starts_with("Maximize") {
        "Maximize"
      } else {
        "Minimize"
      };
      let expected = format!("{func}::ivar: {bad} is not a valid variable.");
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().any(|m| m.contains(&expected)),
        "expected {expected:?} for {input}, got {msgs:?}"
      );
    }
  }

  #[test]
  fn invalid_variable_in_list_emits_ivar() {
    clear_state();
    assert_eq!(
      interpret("Minimize[x^2, {x, y >= 1}]").unwrap(),
      "Minimize[x^2, {x, y >= 1}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs
        .iter()
        .any(|m| m.contains("Minimize::ivar: y >= 1 is not a valid variable.")),
      "expected Minimize::ivar for y >= 1, got {msgs:?}"
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

  // Regression: with 3+ variables, the ILP path previously bailed because
  // `minimize_try_ilp` only recognized `Element[Identifier, Integers]` —
  // not the post-evaluation shapes `Element[{v1, …, vn}, Integers]` or
  // `Element[v1 | v2 | … | vn, Integers]` (BinaryOp Alternatives chain).
  // The fallback dropped into `minimize_lp_2d`, which called
  // `minimize_try_f64` on the third symbol, which re-dispatched `N[c]`
  // back into `n_eval` forever — exhausting WASM's 1 MB stack before the
  // RECURSION_LIMIT guard tripped.
  #[test]
  fn ilp_3var_list_domain() {
    // The objective is fully constrained to 5, but the minimizer is
    // non-unique (any feasible integer point with x+y+z==5 is optimal), so
    // the reported {x->…} differs harmlessly between engines. Assert the
    // unique optimal value via First[…], which both engines agree on.
    assert_eq!(
      interpret(
        "First[Minimize[{x + y + z, x + y + z == 5, x >= 0, y >= 0, z >= 0, Element[{x, y, z}, Integers]}, {x, y, z}]]"
      )
      .unwrap(),
      "5"
    );
  }

  #[test]
  fn ilp_3var_alternatives_domain() {
    // `Element[x | y | z, Integers]` parses to a BinaryOp Alternatives
    // chain — different shape from the FunctionCall Alternatives variant.
    // As above, the minimizer is non-unique; assert the unique optimal value.
    assert_eq!(
      interpret(
        "First[Minimize[{x + y + z, x + y + z == 5, x >= 0, y >= 0, z >= 0, Element[x | y | z, Integers]}, {x, y, z}]]"
      )
      .unwrap(),
      "5"
    );
  }

  #[test]
  fn ilp_euro_coins_8var() {
    // The original report: 8 Array-style variables, dot-product equality,
    // and `Element[vars, Integers]`. Must match wolframscript:
    // 10 × 2€ (85.0 g) + 2 × 1€ (15.0 g) = 100.0 g in 12 coins.
    assert_eq!(
      interpret(
        "coins = {\"a\" -> 8.50, \"b\" -> 7.50, \"c\" -> 7.80, \"d\" -> 5.74, \"e\" -> 4.10, \"f\" -> 3.92, \"g\" -> 3.06, \"h\" -> 2.30}; \
         vars = Array[n, 8]; \
         constraints = Join[{Round[Values[coins]*100] . vars == 10000}, Thread[vars >= 0]]; \
         Minimize[{Total[vars], And @@ constraints, Element[vars, Integers]}, vars]"
      ).unwrap(),
      "{12, {n[1] -> 10, n[2] -> 2, n[3] -> 0, n[4] -> 0, n[5] -> 0, n[6] -> 0, n[7] -> 0, n[8] -> 0}}"
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

  // The maximum of `-2 x^2 - 3 x + 5` is the rational `49/8` at `-3/4`.
  // The negate-back step inside `minimize_single_var` produced
  // `Times[-1, Rational[-49, 8]]` which surfaced as `--49/8`; pin the
  // rational-output branch so a regression in negate_expr is caught
  // directly rather than via the long compound-statement case_499.
  #[test]
  fn maximize_rational_value() {
    assert_eq!(
      interpret("Maximize[-2 x^2 - 3 x + 5, x]").unwrap(),
      "{49/8, {x -> -3/4}}"
    );
  }

  #[test]
  fn constrained_chained_comparison() {
    // Chained comparison 0 <= x <= 30 should be split into two constraints
    assert_eq!(
      interpret("Minimize[{x, 0 <= x <= 30}, {x}]").unwrap(),
      "{0, {x -> 0}}"
    );
  }

  #[test]
  fn ilp_with_element_constraints() {
    // ILP with inline Element[x, Integers] constraints
    // 2x + 3y = 12, x,y >= 0: (0,4)=4, (3,2)=5, (6,0)=6 → min is 4
    assert_eq!(
      interpret(
        "Minimize[{x + y, 2*x + 3*y == 12, x >= 0, y >= 0, Element[x, Integers], Element[y, Integers]}, {x, y}]"
      )
      .unwrap(),
      "{4, {x -> 0, y -> 4}}"
    );
  }

  #[test]
  fn ilp_decimal_coefficients() {
    // ILP with decimal (non-integer) coefficients that need scaling
    assert_eq!(
      interpret(
        "Minimize[{x + y, 8.5*x + 7.5*y == 100, x >= 0, y >= 0, Element[x, Integers], Element[y, Integers]}, {x, y}]"
      )
      .unwrap(),
      "{12., {x -> 10, y -> 2}}"
    );
  }

  #[test]
  fn ilp_with_upper_bounds() {
    // ILP with upper bound constraints
    // 2x + 3y = 12, 0<=x<=30, 0<=y<=30: (0,4)=4, (3,2)=5, (6,0)=6 → min is 4
    assert_eq!(
      interpret(
        "Minimize[{x + y, 2*x + 3*y == 12, 0 <= x <= 30, 0 <= y <= 30, Element[x, Integers], Element[y, Integers]}, {x, y}]"
      )
      .unwrap(),
      "{4, {x -> 0, y -> 4}}"
    );
  }

  #[test]
  fn ilp_coin_change_problem() {
    // Full coin-change style problem similar to the euro coins problem
    assert_eq!(
      interpret(
        "coins = {\"2\\[Euro]\" -> 8.50, \"1\\[Euro]\" -> 7.50, \"50c\" -> 7.80, \"20c\" -> 5.74, \"10c\" -> 4.10, \"5c\" -> 3.92, \"2c\" -> 3.06, \"1c\" -> 2.30}; weights = coins[[All, 2]]; nTypes = Length[weights]; result = Minimize[{Total[Array[n, nTypes]], Total[Array[n, nTypes] * weights] == 100 && And @@ Table[0 <= n[i] <= 30, {i, nTypes}] && And @@ Table[n[i] \\[Element] Integers, {i, nTypes}]}, Array[n, nTypes]]; result"
      )
      .unwrap(),
      "{12., {n[1] -> 10, n[2] -> 2, n[3] -> 0, n[4] -> 0, n[5] -> 0, n[6] -> 0, n[7] -> 0, n[8] -> 0}}"
    );
  }

  // Reals is the optimization family's default domain, so naming it explicitly
  // must change nothing. It used to send the call down a slower path that could
  // not solve the problem at all, leaving it unevaluated.
  #[test]
  fn explicit_reals_domain_matches_the_default() {
    assert_eq!(
      interpret("MaxValue[{x + y, x^2 + y^2 <= 1}, {x, y}, Reals]").unwrap(),
      "Sqrt[2]"
    );
    assert_eq!(
      interpret("MinValue[{x + y, x^2 + y^2 <= 1}, {x, y}, Reals]").unwrap(),
      "-Sqrt[2]"
    );
    assert_eq!(
      interpret("Maximize[-x^2 + 4 x, x, Reals]").unwrap(),
      "{4, {x -> 2}}"
    );
    assert_eq!(
      interpret("Minimize[{x^2 + y^2, x + y == 1}, {x, y}, Reals]").unwrap(),
      "{1/2, {x -> 1/2, y -> 1/2}}"
    );
  }

  // An Integers domain whose real optimum is already an integer is optimal over
  // the integers too, so the answer carries over. Every one of these was
  // unevaluated before.
  #[test]
  fn integers_domain_when_the_real_optimum_is_integral() {
    assert_eq!(
      interpret("Minimize[x^2, x, Integers]").unwrap(),
      "{0, {x -> 0}}"
    );
    assert_eq!(
      interpret("Minimize[x^2 - 4 x, x, Integers]").unwrap(),
      "{-4, {x -> 2}}"
    );
    assert_eq!(interpret("MinValue[x^2 + 1, x, Integers]").unwrap(), "1");
    assert_eq!(
      interpret("MaxValue[x (10 - x), x, Integers]").unwrap(),
      "25"
    );
    assert_eq!(
      interpret("ArgMax[{x (10 - x), 0 <= x <= 10}, x, Integers]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("ArgMin[{x^2, x >= 1}, x, Integers]").unwrap(),
      "1"
    );
    // An unbounded real problem is unbounded over the integers as well.
    assert_eq!(
      interpret("Maximize[x^2, x, Integers]").unwrap(),
      "{Infinity, {x -> -Infinity}}"
    );
    // The bounded-constraint path still handles what it always did.
    assert_eq!(
      interpret("Maximize[{x y, x + y == 10}, {x, y}, Integers]").unwrap(),
      "{25, {x -> 5, y -> 5}}"
    );
  }

  // A real optimum sitting on an excluded boundary is not a solution over the
  // integers — Maximize[{x, x < 3}, x] reports the boundary point {3, {x -> 3}},
  // which x < 3 rules out — so the search steps inward to the nearest feasible
  // integer point.
  #[test]
  fn integers_domain_steps_in_from_an_infeasible_boundary_optimum() {
    assert_eq!(
      interpret("Maximize[{x, x < 3}, x, Integers]").unwrap(),
      "{2, {x -> 2}}"
    );
    assert_eq!(
      interpret("Minimize[{x, x > 3}, x, Integers]").unwrap(),
      "{4, {x -> 4}}"
    );
    assert_eq!(
      interpret("Maximize[{x + y, x < 3 && y < 2}, {x, y}, Integers]").unwrap(),
      "{3, {x -> 2, y -> 1}}"
    );
    // Without the domain the boundary point is what wolframscript reports too.
    assert_eq!(
      interpret("Maximize[{x, x < 3}, x]").unwrap(),
      "{3, {x -> 3}}"
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
  fn integrate_x_pow_neg1() {
    // x^-1 is the same as 1/x, should give Log[x]
    assert_eq!(interpret("Integrate[x^-1, x]").unwrap(), "Log[x]");
  }

  #[test]
  fn integrate_x_pow_neg1_parens() {
    assert_eq!(interpret("Integrate[x^(-1), x]").unwrap(), "Log[x]");
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
  fn integrate_1_over_x2_plus_c_simplifies_arctan() {
    // Regression: ArcTan argument and coefficient must be fully simplified
    // by extracting perfect square factors from the discriminant.
    // e.g. 1/(x^2+50): neg_disc=200=100*2, so sqrt(200)=10*sqrt(2),
    // and the factor of 2 cancels: 2x/(10*sqrt(2)) -> x/(5*sqrt(2))
    assert_eq!(
      interpret("Integrate[1/(x^2+50), x]").unwrap(),
      "ArcTan[x/(5*Sqrt[2])]/(5*Sqrt[2])"
    );
    assert_eq!(
      interpret("Integrate[1/(x^2+3), x]").unwrap(),
      "ArcTan[x/Sqrt[3]]/Sqrt[3]"
    );
    assert_eq!(
      interpret("Integrate[1/(x^2+2*x+5), x]").unwrap(),
      "ArcTan[(1 + x)/2]/2"
    );
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

// Integral of 1/(p x^2 + q) for constant p, q (numeric or symbolic):
//   q > 0:        ArcTan[Sqrt[p/q] x] / Sqrt[p q]
//   q = -a^2:     -ArcTanh[x/a] / a
mod integrate_reciprocal_quadratic {
  use super::*;

  #[test]
  fn symbolic_a_squared_arctan() {
    assert_eq!(
      interpret("Integrate[1/(x^2 + a^2), x]").unwrap(),
      "ArcTan[x/a]/a"
    );
  }

  #[test]
  fn symbolic_a_squared_order_independent() {
    assert_eq!(
      interpret("Integrate[1/(a^2 + x^2), x]").unwrap(),
      "ArcTan[x/a]/a"
    );
  }

  #[test]
  fn symbolic_difference_arctanh() {
    assert_eq!(
      interpret("Integrate[1/(x^2 - a^2), x]").unwrap(),
      "-(ArcTanh[x/a]/a)"
    );
  }

  #[test]
  fn bare_symbol_constant() {
    assert_eq!(
      interpret("Integrate[1/(b + x^2), x]").unwrap(),
      "ArcTan[x/Sqrt[b]]/Sqrt[b]"
    );
  }

  #[test]
  fn leading_coefficient() {
    assert_eq!(
      interpret("Integrate[1/(9 x^2 + 1), x]").unwrap(),
      "ArcTan[3*x]/3"
    );
  }

  #[test]
  fn numeric_constant_unchanged() {
    assert_eq!(
      interpret("Integrate[1/(x^2 + 4), x]").unwrap(),
      "ArcTan[x/2]/2"
    );
  }

  // Numeric x^2 - c stays in partial-fraction Log form (not ArcTanh).
  #[test]
  fn numeric_difference_stays_log() {
    assert_eq!(
      interpret("Integrate[1/(x^2 - 4), x]").unwrap(),
      "Log[2 - x]/4 - Log[2 + x]/4"
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

  // First-order linear inhomogeneous ODEs: the integrating factor is
  // distributed over the particular and homogeneous parts, matching
  // wolframscript, while a grouped particular (e.g. a trig combination) stays
  // grouped. Verified against wolframscript.
  #[test]
  fn first_order_linear_inhomogeneous_distributes() {
    assert_eq!(
      interpret("DSolve[y'[x] + 2 y[x] == Exp[x], y[x], x]").unwrap(),
      "{{y[x] -> E^x/3 + C[1]/E^(2*x)}}"
    );
    assert_eq!(
      interpret("DSolve[y'[x] + y[x] == x, y[x], x]").unwrap(),
      "{{y[x] -> -1 + x + C[1]/E^x}}"
    );
    assert_eq!(
      interpret("DSolve[y'[x] - y[x] == Exp[2 x], y[x], x]").unwrap(),
      "{{y[x] -> E^(2*x) + E^x*C[1]}}"
    );
    // The trig particular solution is kept grouped, not split into terms.
    assert_eq!(
      interpret("DSolve[y'[x] + 3 y[x] == Sin[x], y[x], x]").unwrap(),
      "{{y[x] -> C[1]/E^(3*x) + (-Cos[x] + 3*Sin[x])/10}}"
    );
  }

  // An equation with no derivative of the dependent function is purely
  // algebraic: DSolve reduces to Solve for the dependent function, matching
  // wolframscript.
  #[test]
  fn algebraic_no_derivative_linear() {
    assert_eq!(
      interpret("DSolve[y[x] + 2 == 5, y[x], x]").unwrap(),
      "{{y[x] -> 3}}"
    );
    assert_eq!(
      interpret("DSolve[2 y[x] == x, y[x], x]").unwrap(),
      "{{y[x] -> x/2}}"
    );
  }

  #[test]
  fn algebraic_no_derivative_quadratic() {
    assert_eq!(
      interpret("DSolve[y[x]^2 == 4, y[x], x]").unwrap(),
      "{{y[x] -> -2}, {y[x] -> 2}}"
    );
  }

  #[test]
  fn algebraic_no_derivative_system() {
    assert_eq!(
      interpret(
        "DSolve[{y[x] + z[x] == 1, y[x] - z[x] == 3}, {y[x], z[x]}, x]"
      )
      .unwrap(),
      "{{y[x] -> 2, z[x] -> -1}}"
    );
  }

  #[test]
  fn unsolvable_ode_stays_unevaluated() {
    // A nonlinear ODE Woxi can't classify must return the unevaluated DSolve
    // (like wolframscript for genuinely unsolvable equations), not leak an
    // internal "DSolve: cannot classify…" error.
    assert_eq!(
      interpret("DSolve[y'[x] == y[x]^2, y[x], x]").unwrap(),
      "DSolve[Derivative[1][y][x] == y[x]^2, y[x], x]"
    );
    assert_eq!(
      interpret("DSolve[y'[x] == Sin[y[x]], y[x], x]").unwrap(),
      "DSolve[Derivative[1][y][x] == Sin[y[x]], y[x], x]"
    );
  }

  // Regression: a separable nonlinear ODE whose right-hand side is a PRODUCT
  // of an x-factor and a nonlinear y-factor (e.g. x*y[x]^2) used to be
  // misclassified — the x*y^2 term was treated as a y-free forcing term and
  // "integrated", yielding the bogus circular C[1] + Integrate[x*y[x]^2, x].
  // Without an initial condition to pin the constant it must stay
  // unevaluated, like the bare y[x]^2 case.
  #[test]
  fn separable_nonlinear_product_stays_unevaluated() {
    assert_eq!(
      interpret("DSolve[y'[x] == x y[x]^2, y[x], x]").unwrap(),
      "DSolve[Derivative[1][y][x] == x*y[x]^2, y[x], x]"
    );
    assert_eq!(
      interpret("DSolve[y'[x] == x^2 y[x]^2, y[x], x]").unwrap(),
      "DSolve[Derivative[1][y][x] == x^2*y[x]^2, y[x], x]"
    );
  }

  // An initial condition pins the constant of a separable equation, so the
  // implicit relation ∫dy/h(y) == ∫g(x)dx + C can be solved outright — the
  // nonlinear right-hand sides above are the ones the linear term classifier
  // rejects.
  #[test]
  fn separable_nonlinear_initial_value_problem() {
    assert_eq!(
      interpret("DSolve[{y'[x] == x y[x]^2, y[0] == 2}, y[x], x]").unwrap(),
      "{{y[x] -> -2/(-1 + x^2)}}"
    );
    assert_eq!(
      interpret("DSolve[{y'[x] == y[x]^2, y[0] == 1}, y[x], x]").unwrap(),
      "{{y[x] -> (1 - x)^(-1)}}"
    );
    assert_eq!(
      interpret("DSolve[{y'[x] == y[x]^3, y[0] == 1}, y[x], x]").unwrap(),
      "{{y[x] -> 1/Sqrt[1 - 2*x]}}"
    );
    // The x-factor may be any closed-form function of x, not just a monomial.
    assert_eq!(
      interpret("DSolve[{y'[x] == Cos[x] y[x]^2, y[0] == 1}, y[x], x]")
        .unwrap(),
      "{{y[x] -> (1 - Sin[x])^(-1)}}"
    );
    // A quotient separates too: y' == x/y.
    assert_eq!(
      interpret("DSolve[{y'[x] == x/y[x], y[0] == 1}, y[x], x]").unwrap(),
      "{{y[x] -> Sqrt[1 + x^2]}}"
    );
    // `DSolve[…, y, x]` asks for the Function form.
    assert_eq!(
      interpret("DSolve[{y'[t] == -t y[t]^2, y[0] == 1}, y, t]").unwrap(),
      "{{y -> Function[{t}, 2/(2 + t^2)]}}"
    );
    assert_eq!(
      interpret("DSolve[{y'[t] == (t - t^3) y[t]^2, y[0] == 1}, y, t]")
        .unwrap(),
      "{{y -> Function[{t}, 4/(4 - 2*t^2 + t^4)]}}"
    );
  }

  // Squaring away the square root when solving for y loses the branch, so the
  // root that does not meet the initial condition has to be dropped: with
  // y[0] == -1 the answer is the negative branch, not the positive one.
  #[test]
  fn separable_solution_picks_the_branch_the_condition_selects() {
    assert_eq!(
      interpret("DSolve[{y'[x] == x/y[x], y[0] == -1}, y[x], x]").unwrap(),
      "{{y[x] -> -Sqrt[1 + x^2]}}"
    );
  }

  // Initial conditions pin the constant: the ODE must not be misread as an
  // initial condition (its point is the variable x, not a number).
  #[test]
  fn first_order_initial_condition() {
    assert_eq!(
      interpret("DSolve[{y'[x] == y[x], y[0] == 1}, y[x], x]").unwrap(),
      "{{y[x] -> E^x}}"
    );
    assert_eq!(
      interpret("DSolve[{y'[x] == y[x], y[0] == 2}, y[x], x]").unwrap(),
      "{{y[x] -> 2*E^x}}"
    );
    assert_eq!(
      interpret("DSolve[{y'[x] == -y[x], y[0] == 3}, y[x], x]").unwrap(),
      "{{y[x] -> 3/E^x}}"
    );
  }

  #[test]
  fn first_order_forcing_with_ic() {
    assert_eq!(
      interpret("DSolve[{y'[x] == 2 x, y[0] == 0}, y[x], x]").unwrap(),
      "{{y[x] -> x^2}}"
    );
    assert_eq!(
      interpret("DSolve[{y'[x] == 2 x, y[0] == 1}, y[x], x]").unwrap(),
      "{{y[x] -> 1 + x^2}}"
    );
    assert_eq!(
      interpret("DSolve[{y'[x] == Cos[x], y[0] == 0}, y[x], x]").unwrap(),
      "{{y[x] -> Sin[x]}}"
    );
  }

  #[test]
  fn second_order_initial_conditions() {
    assert_eq!(
      interpret("DSolve[{y''[x] == -y[x], y[0] == 0, y'[0] == 1}, y[x], x]")
        .unwrap(),
      "{{y[x] -> Sin[x]}}"
    );
    assert_eq!(
      interpret("DSolve[{y''[x] == -y[x], y[0] == 1, y'[0] == 0}, y[x], x]")
        .unwrap(),
      "{{y[x] -> Cos[x]}}"
    );
  }

  #[test]
  fn second_order_zero_rhs() {
    // y''[x] == 0 → y[x] -> C[1] + x*C[2]
    let result = interpret("DSolve[y''[x] == 0, y[x], x]").unwrap();
    assert!(
      result == "{{y[x] -> C[1] + x*C[2]}}"
        || result == "{{y[x] -> x*C[2] + C[1]}}",
      "Got: {result}"
    );
  }

  #[test]
  fn harmonic_oscillator() {
    // y''[x] + y[x] == 0 → y[x] -> C[1]*Cos[x] + C[2]*Sin[x]
    let result = interpret("DSolve[y''[x] + y[x] == 0, y[x], x]").unwrap();
    assert!(
      result == "{{y[x] -> C[1]*Cos[x] + C[2]*Sin[x]}}"
        || result == "{{y[x] -> C[2]*Sin[x] + C[1]*Cos[x]}}",
      "Got: {result}"
    );
  }

  #[test]
  fn exponential_pair_general_solution() {
    // y''[x] == y[x] has general solution C[1]*E^x + C[2]*E^(-x).
    assert_eq!(
      interpret("DSolve[y''[x] == y[x], y[x], x]").unwrap(),
      "{{y[x] -> E^x*C[1] + C[2]/E^x}}"
    );
  }

  #[test]
  fn second_order_nonconstant_forcing_variation_of_parameters() {
    // Non-constant forcing on a constant-coefficient second-order ODE is
    // solved by variation of parameters. Regression: the forcing term used to
    // be silently dropped, returning only the homogeneous solution.
    // wolframscript: {{y[x] -> -(ArcTanh[Sin[x]]*Cos[x]) + C[1]*Cos[x] + C[2]*Sin[x]}}
    assert_eq!(
      interpret("DSolve[y''[x] + y[x] == Tan[x], y[x], x]").unwrap(),
      "{{y[x] -> -(ArcTanh[Sin[x]]*Cos[x]) + C[1]*Cos[x] + C[2]*Sin[x]}}"
    );
    // wolframscript: {{y[x] -> x + C[1]*Cos[x] + C[2]*Sin[x]}}
    assert_eq!(
      interpret("DSolve[y''[x] + y[x] == x, y[x], x]").unwrap(),
      "{{y[x] -> x + C[1]*Cos[x] + C[2]*Sin[x]}}"
    );
    // wolframscript: {{y[x] -> C[1]*Cos[x] + Cos[x]*Log[Cos[x]] + x*Sin[x] + C[2]*Sin[x]}}
    assert_eq!(
      interpret("DSolve[y''[x] + y[x] == Sec[x], y[x], x]").unwrap(),
      "{{y[x] -> C[1]*Cos[x] + Cos[x]*Log[Cos[x]] + x*Sin[x] + C[2]*Sin[x]}}"
    );
    // Distinct real roots with exponential forcing. The fundamental pair is
    // ordered by ascending root, so C[1] attaches to E^x (r=1) and C[2] to
    // E^(2*x) (r=2), matching wolframscript.
    assert_eq!(
      interpret("DSolve[y''[x] - 3*y'[x] + 2*y[x] == E^(3*x), y[x], x]")
        .unwrap(),
      "{{y[x] -> E^(3*x)/2 + E^x*C[1] + E^(2*x)*C[2]}}"
    );
    // Roots symmetric about zero (±1) are the exception: the positive root
    // leads, so C[1] attaches to E^x and C[2] to E^(-x).
    assert_eq!(
      interpret("DSolve[y''[x] - y[x] == 0, y[x], x]").unwrap(),
      "{{y[x] -> E^x*C[1] + C[2]/E^x}}"
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
      "Got: {result}"
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
      "Got: {result}"
    );
  }

  #[test]
  fn real_distinct_roots() {
    // y'' - 3y' + 2y == 0 → roots r=1, r=2
    let result =
      interpret("DSolve[y''[x] - 3*y'[x] + 2*y[x] == 0, y[x], x]").unwrap();
    assert!(
      result.contains("E^x") && result.contains("E^(2*x)"),
      "Got: {result}"
    );
  }

  #[test]
  fn repeated_root() {
    // y'' - 2y' + y == 0 → double root r=1
    let result =
      interpret("DSolve[y''[x] - 2*y'[x] + y[x] == 0, y[x], x]").unwrap();
    // Should contain x*E^x for the repeated root part
    assert!(result.contains("E^x"), "Got: {result}");
  }

  #[test]
  fn function_form() {
    // y'' + y == 0, returning Function form
    let result = interpret("DSolve[y''[x] + y[x] == 0, y, x]").unwrap();
    assert!(result.contains("Function["), "Got: {result}");
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
      "Got: {result}"
    );
  }
}

mod ndsolve {
  use super::*;

  /// An algebraic constraint that determines one unknown explicitly is
  /// eliminated — the constraint is solved for it, substituted into the
  /// remaining equations, and the unknown rebuilt from the solution
  /// afterwards. This is the index-1 DAE a Lagrangian model writes when a
  /// rigid link ties two coordinates together ("Dynamics of a Spring-
  /// Pendulum System"). Values checked against wolframscript.
  #[test]
  fn algebraic_constraint_is_eliminated() {
    let code = "sol = NDSolve[{x''[t] == -x[t] - v[t], v[t] == x[t]/2, \
       x[0] == 1, x'[0] == 0}, {x, v}, {t, 0, 4}]; \
       {Length[sol[[1]]], N[Round[x[3] /. sol[[1]], 0.0001]], \
        N[Round[v[3] /. sol[[1]], 0.0001]]}";
    // x'' = -3x/2 with x(0)=1, x'(0)=0 → x(3) = Cos[Sqrt[3/2] 3], v = x/2.
    let expected = (1.5f64.sqrt() * 3.0).cos();
    let result = interpret(code).unwrap();
    let nums: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(", ")
      .map(|s| s.parse().unwrap())
      .collect();
    assert_eq!(nums[0], 2.0, "both functions come back: {result}");
    assert!((nums[1] - expected).abs() < 1e-3, "{result} vs {expected}");
    assert!(
      (nums[2] - expected / 2.0).abs() < 1e-3,
      "the eliminated function is rebuilt from its constraint: {result}"
    );
  }

  /// A constraint already written as `f[x] == rhs` (or the mirror image) is
  /// its own solution — no need to hand it to the general `Solve[]`. A
  /// second constraint in a form `Solve[]` genuinely has to work for
  /// (`z[t] + y[t] == 1`, not a bare call on either side) checks that the
  /// fallback still runs correctly alongside the fast path, in the same
  /// elimination chain. Independently written, not from any Demonstration.
  #[test]
  fn explicit_algebraic_constraints_take_the_fast_path() {
    // y = x/3 substituted into x' = -x + y gives x' = -2x/3, so
    // x(t) = 3 Exp[-2t/3]; y = x/3; z = 1 - y.
    let code = "sol = NDSolve[{x'[t] == -x[t] + y[t], y[t] == x[t]/3, \
       z[t] + y[t] == 1, x[0] == 3}, {x, y, z}, {t, 0, 2}][[1]]; \
       {x[1], y[1], z[1]} /. sol /. t -> 1.0";
    let result = interpret(code).unwrap();
    let nums: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(", ")
      .map(|s| s.parse().unwrap())
      .collect();
    let x1 = 3.0 * (-2.0 / 3.0f64).exp();
    let expected = [x1, x1 / 3.0, 1.0 - x1 / 3.0];
    for (got, want) in nums.iter().zip(expected) {
      assert!((got - want).abs() < 1e-3, "{result}: {got} vs {want}");
    }
  }

  /// The mass/energy-balance systems a chemical-engineering or circuit
  /// Demonstration writes have highest-derivative coefficients that are
  /// plain holdup/capacitance numerals, not expressions of the state —
  /// `2 x1'[t] == …`, not `x1[t] x1'[t] == …`. `NDSolve` can then build the
  /// mass matrix once instead of at every stage of every step; this checks
  /// the *result* stays exact when it does, against the closed form for two
  /// tanks (unequal constant holdups) relaxing to a shared level.
  /// Independently written, not from any Demonstration.
  #[test]
  fn constant_mass_matrix_system_matches_closed_form() {
    // 2 x1' = -(x1 - x2), 3 x2' = (x1 - x2); y = x1 - x2 decays as
    // Exp[-5t/6] (from 1/2 + 1/3), and 2 x1 + 3 x2 is conserved at 2.
    let code = "sol = NDSolve[{2 x1'[t] == -(x1[t] - x2[t]), \
       3 x2'[t] == x1[t] - x2[t], x1[0] == 1, x2[0] == 0}, \
       {x1, x2}, {t, 0, 3}][[1]]; \
       {x1[1], x2[1]} /. sol /. t -> 1.0";
    let result = interpret(code).unwrap();
    let nums: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(", ")
      .map(|s| s.parse().unwrap())
      .collect();
    let y1 = (-5.0 / 6.0f64).exp();
    let expected = [(2.0 + 3.0 * y1) / 5.0, (2.0 - 2.0 * y1) / 5.0];
    for (got, want) in nums.iter().zip(expected) {
      assert!((got - want).abs() < 1e-3, "{result}: {got} vs {want}");
    }
  }

  /// Unknowns need not be bare symbols: a transport equation discretized in
  /// space is written with `Subscript[c, i]` for each cell, so the whole
  /// system is `NDSolve[…, Table[Subscript[c, i], {i, 1, n}], …]`. Each
  /// compound head is keyed by a fresh symbol while integrating and restored
  /// in the solution rules.
  #[test]
  fn subscripted_unknowns_are_solved_and_restored() {
    // Two well-mixed tanks exchanging content: c1' = c2 - c1, c2' = c1 - c2,
    // so c1 + c2 is conserved and both relax to 1/2 as exp(-2t).
    let system = "sol = NDSolve[{Subscript[c, 1]'[t] == Subscript[c, 2][t] \
       - Subscript[c, 1][t], Subscript[c, 2]'[t] == Subscript[c, 1][t] \
       - Subscript[c, 2][t], Subscript[c, 1][0] == 1, \
       Subscript[c, 2][0] == 0}, {Subscript[c, 1], Subscript[c, 2]}, \
       {t, 0, 5}]; ";
    assert_eq!(
      interpret(&format!("{system}sol[[1]][[All, 1]]")).unwrap(),
      "{Subscript[c, 1], Subscript[c, 2]}",
      "the compound heads come back verbatim"
    );
    let value: f64 = interpret(&format!(
      "{system}First[Subscript[c, 1][t] /. sol] /. t -> 1.0"
    ))
    .unwrap()
    .parse()
    .expect("should be a number");
    let expected = f64::midpoint(1.0, (-2.0f64).exp());
    assert!(
      (value - expected).abs() < 1e-6,
      "expected {expected}, got {value}"
    );
  }

  /// A forcing term that is nonzero only on a very narrow interval — an
  /// injected tracer pulse — must not be integrated as though it lasted a
  /// whole grid step, which would inflate the solution by the ratio of the
  /// two widths. The step is bisected until the pulse is resolved.
  #[test]
  fn a_narrow_pulse_is_resolved() {
    // y' = -y + 1000 * Boole[t <= 1/1000] over [0, 5]: the pulse injects
    // exactly 1 unit, so y(t) = Exp[-t] to within the pulse width.
    let code = "s = NDSolve[{y'[t] == -y[t] \
       + 1000 If[0 <= t <= 1/1000, 1, 0], y[0] == 0}, y, {t, 0, 5}]; \
       {(y /. s[[1]])[1.0], (y /. s[[1]])[3.0]}";
    let result = interpret(code).unwrap();
    let nums: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(", ")
      .map(|s| s.parse().unwrap())
      .collect();
    for (value, t) in nums.iter().zip([1.0f64, 3.0]) {
      let expected = (-t).exp();
      assert!(
        (value - expected).abs() < 1e-3 * expected,
        "y({t}) should be about {expected}, got {value} (in {result})"
      );
    }
  }

  /// A smooth problem must not be refined at all: the fixed 1000-step grid
  /// is already far more accurate than the refinement tolerance, so the
  /// interpolating solution keeps exactly the points it always had.
  #[test]
  fn a_smooth_problem_keeps_the_nominal_grid() {
    let result = interpret(
      "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]; \
       Length[(y /. s[[1]])[[2]]]",
    )
    .unwrap();
    assert_eq!(result, "1001");
  }

  /// The solution rules come back in the order the functions were asked
  /// for, even though an eliminated one is solved last.
  #[test]
  fn constraint_solution_keeps_the_requested_order() {
    let result = interpret(
      "sol = NDSolve[{x'[t] == 1, w[t] == 2 x[t], x[0] == 0}, {w, x}, \
       {t, 0, 1}]; sol[[1]][[All, 1]]",
    )
    .unwrap();
    assert_eq!(result, "{w, x}");
  }

  /// Substituting the solution rules into a derivative — `y'[t] /. sol`, how
  /// a Demonstration plots a phase portrait — leaves
  /// `Derivative[1][InterpolatingFunction[…]][t]`, which has to evaluate to
  /// the slope, not stay a symbolic `Derivative[…]` echo.
  #[test]
  fn a_substituted_derivative_evaluates_numerically() {
    // y' = y, y(0) = 1 → y(t) = y'(t) = E^t.
    let result = interpret(
      "sol = NDSolve[{y'[t] == y[t], y[0] == 1}, y, {t, 0, 1}]; \
       {y[0.5], y'[0.5]} /. sol[[1]]",
    )
    .unwrap();
    let nums: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(", ")
      .map(|s| s.parse().expect("both entries are numbers"))
      .collect();
    let expected = std::f64::consts::E.powf(0.5);
    for value in &nums {
      assert!(
        (value - expected).abs() < 1e-3,
        "expected about {expected}, got {result}"
      );
    }
  }

  /// The same for a second derivative, and for the whole solution list
  /// (`/. sol` rather than `/. sol[[1]]`).
  #[test]
  fn a_substituted_second_derivative_evaluates_numerically() {
    // y'' = -y with y(0) = 0, y'(0) = 1 → y = Sin[t], y'' = -Sin[t].
    let result = interpret(
      "sol = NDSolve[{y''[t] == -y[t], y[0] == 0, y'[0] == 1}, y, \
       {t, 0, 3}]; y''[1.0] /. sol",
    )
    .unwrap();
    let value: f64 = result
      .trim_matches(['{', '}'])
      .parse()
      .expect("a single numeric solution");
    let expected = -1.0f64.sin();
    assert!(
      (value - expected).abs() < 1e-3,
      "expected about {expected}, got {result}"
    );
  }

  /// `f'`/`f''` on machine-precision interpolation data takes a fast
  /// numeric path (Newton divided differences expanded to monomial
  /// coefficients) rather than building and simplifying a symbolic
  /// Lagrange polynomial through the general evaluator — the latter turned
  /// every sample in an adaptively-plotted phase portrait into a
  /// multi-millisecond symbolic evaluation. The data here is exactly `x^2`
  /// (`f = x^2`, `f' = 2x`, `f'' = 2`), including a query exactly at a grid
  /// node — the classic singularity for a naive log-derivative Lagrange
  /// formula — to guard the replacement algorithm's correctness there too.
  #[test]
  fn interpolation_derivative_numeric_path_matches_analytic_derivative() {
    let result = interpret(
      "f = Interpolation[{{0., 0.}, {1., 1.}, {2., 4.}, {3., 9.}}]; \
       {f[1.5], f'[1.5], f''[1.5], f[1.0], f'[1.0], f''[1.0]}",
    )
    .unwrap();
    let nums: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(", ")
      .map(|s| s.parse().expect("all entries are numbers"))
      .collect();
    let expected = [2.25, 3.0, 2.0, 1.0, 2.0, 2.0];
    for (got, want) in nums.iter().zip(expected.iter()) {
      assert!(
        (got - want).abs() < 1e-9,
        "expected {expected:?}, got {result}"
      );
    }
  }

  /// A derivative order at or beyond the local interpolating polynomial's
  /// degree is exactly zero, not a numerical artifact from over-differentiating
  /// the expanded monomial coefficients.
  #[test]
  fn interpolation_derivative_beyond_polynomial_degree_is_zero() {
    assert_eq!(
      interpret(
        "f = Interpolation[{{0., 0.}, {1., 1.}, {2., 4.}, {3., 9.}}]; \
         Derivative[3][f][1.5]"
      )
      .unwrap(),
      "0."
    );
  }

  /// The derivative path shares the value path's extrapolation window, so a
  /// query outside the data range still differentiates the boundary piece
  /// instead of panicking or returning nonsense.
  #[test]
  fn interpolation_derivative_extrapolates_past_data_range() {
    let result = interpret(
      "f = Interpolation[{{0., 0.}, {1., 1.}, {2., 4.}, {3., 9.}}]; f'[5.0]",
    )
    .unwrap();
    let value: f64 = result.parse().expect("a number");
    assert!((value - 10.0).abs() < 1e-9, "expected 10., got {result}");
  }

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
      "Expected {expected}, got {val}"
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
    assert!((val - 0.5).abs() < 0.001, "Expected 0.5, got {val}");
  }

  #[test]
  fn quadratic_growth() {
    // NDSolve y'=x, y(0)=0, check y(1) ≈ 0.5
    let result = interpret(
      "sol = NDSolve[{y'[x] == x, y[0] == 0}, y, {x, 0, 1}]; y[1] /. sol[[1]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 0.5).abs() < 0.001, "Expected 0.5, got {val}");
  }

  #[test]
  fn interpolating_function_display() {
    // InterpolatingFunction should display with <> for data
    let result =
      interpret("NDSolve[{y'[x] == y[x], y[0] == 1}, y, {x, 0, 1}]").unwrap();
    assert!(
      result.contains("InterpolatingFunction") && result.contains("<>"),
      "Got: {result}"
    );
  }

  #[test]
  fn dependent_variable_form_keeps_the_argument() {
    // `NDSolve[…, y[x], …]` solves for y[x], so both sides of the rule carry
    // the argument; only the `y` form returns the bare function.
    assert_eq!(
      interpret("NDSolve[{y'[x] == y[x], y[0] == 1}, y[x], {x, 0, 1}]")
        .unwrap(),
      "{{y[x] -> InterpolatingFunction[{{0., 1.}}, <>][x]}}"
    );
    assert_eq!(
      interpret("NDSolve[{y'[x] == y[x], y[0] == 1}, y, {x, 0, 1}]").unwrap(),
      "{{y -> InterpolatingFunction[{{0., 1.}}, <>]}}"
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
    assert!((val - (-1.0)).abs() < 0.01, "Expected -1.0, got {val}");
  }

  #[test]
  fn coupled_first_order_system() {
    // x' = y, y' = -x with x(0)=1, y(0)=0 → x = cos t.
    let result = interpret(
      "s = NDSolve[{x'[t] == y[t], y'[t] == -x[t], x[0] == 1, y[0] == 0}, \
       {x, y}, {t, 0, 4}]; (x /. s[[1]])[N[Pi]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - (-1.0)).abs() < 0.01, "Expected -1.0, got {val}");
  }

  #[test]
  fn implicit_second_order_system() {
    // Equations coupled in the highest derivatives (a mass-matrix
    // system, like the trebuchet demonstration's Lagrangian equations):
    // u'' + v'' == -(u+v), u'' - v'' == -(u-v) decouples to u'' = -u,
    // v'' = -v, so with u(0)=1, u'(0)=0 the solution is u(t) = cos t.
    let result = interpret(
      "s = NDSolve[{u''[t] + v''[t] == -(u[t] + v[t]), \
       u''[t] - v''[t] == -(u[t] - v[t]), \
       u[0] == 1, u'[0] == 0, v[0] == 0, v'[0] == 0}, {u, v}, {t, 0, 3}]; \
       (u /. s[[1]])[1.0]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let expected = 1.0_f64.cos();
    assert!(
      (val - expected).abs() < 0.01,
      "Expected {expected}, got {val}"
    );
  }

  #[test]
  fn ndsolve_second_argument_requests_a_derivative_alongside_its_function() {
    // Regression: `NDSolve[…, {y, y'}, …]` used to bail out unevaluated —
    // `Derivative[1][y]` in the second argument wasn't recognized as a
    // request for y's derivative and was instead treated as an opaque
    // "compound head" to rename away. It should return both `y` and
    // `Derivative[1][y]` as separate InterpolatingFunction rules, sparing
    // the caller from differentiating the interpolant themselves.
    // y'' + y == 0, y(0) = 1, y'(0) = 0 → y = Cos[t], y' = -Sin[t].
    let result = interpret(
      "sol = NDSolve[{y''[t] + y[t] == 0, y[0] == 1, y'[0] == 0}, {y, y'}, \
       {t, 0, 4}]; y'[N[Pi/2]] /. sol[[1,2]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let expected = -(std::f64::consts::FRAC_PI_2.sin());
    assert!(
      (val - expected).abs() < 0.01,
      "Expected {expected}, got {val}"
    );
  }

  #[test]
  fn ndsolve_requested_derivative_keeps_its_own_rule_and_function_intact() {
    // The `y -> …` rule must still come back too, and unaffected by the
    // extra `Derivative[1][y] -> …` rule alongside it.
    let result = interpret(
      "sol = NDSolve[{y''[t] + y[t] == 0, y[0] == 1, y'[0] == 0}, {y, y'}, \
       {t, 0, 4}]; y[N[Pi]] /. sol[[1,1]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - (-1.0)).abs() < 0.01, "Expected -1.0, got {val}");
  }

  #[test]
  fn ndsolve_flattens_a_nested_initial_condition_list() {
    // Regression: an equations argument that groups one function's initial
    // conditions into their own sublist — `{ode, {ic1, ic2}}`, a common
    // Demonstrations idiom — used to be counted as a second, bogus equation
    // (a List isn't itself an equation) rather than flattened, so the
    // equation/function count mismatched and NDSolve bailed out unevaluated.
    // Same system as `second_order_harmonic`: y'' + y = 0, y(0)=1, y'(0)=0,
    // so y(Pi) ≈ -1.
    let result = interpret(
      "sol = NDSolve[{y''[x] + y[x] == 0, {y[0] == 1, y'[0] == 0}}, y, \
       {x, 0, 4}]; y[N[Pi]] /. sol[[1]]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - (-1.0)).abs() < 0.01, "Expected -1.0, got {val}");
  }

  #[test]
  fn interpolating_function_input_form_has_no_placeholder() {
    // Regression: InterpolatingFunction's `<>` data placeholder — meant
    // only to keep display output short — was also applied under
    // InputForm, where it isn't valid syntax at all. Manipulate relies on
    // InputForm to round-trip a variable's value back through the parser
    // between frames (`ManipulateState::reevaluate`), so a body that binds
    // a Demonstration's NDSolve solution to a variable would throw a parse
    // error on the very next frame. InputForm must print the literal data.
    let result = interpret(
      "ToString[NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}], InputForm]",
    )
    .unwrap();
    assert!(
      result.contains("InterpolatingFunction") && !result.contains("<>"),
      "Got: {result}"
    );
    // OutputForm (the default) is unaffected — it should still abbreviate.
    let output =
      interpret("NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]").unwrap();
    assert!(output.contains("<>"), "Got: {output}");
  }

  #[test]
  fn ndsolve_implicit_coefficient_times_indexed_double_derivative() {
    // Regression: a coefficient placed via implicit multiplication directly
    // before an indexed function's double derivative (`mu pos[1]''[t]`, no
    // `*`) used to fail to parse — this is exactly the shape of the
    // equations of motion in a lattice/chain ODE (e.g. coupled oscillators),
    // where each particle's acceleration is written as `mass x[k]''[t]`.
    // With mass canceling out, `mu pos[1]''[t] == -2 mu pos[1][t]` reduces
    // to `pos1'' = -2 pos1`, so with pos1(0) = 1, pos1'(0) = 0 the solution
    // is `pos1(t) = Cos[Sqrt[2] t]`.
    let result = interpret(
      "mu = 2; \
       sol = NDSolve[{mu pos[1]''[t] == -2 mu pos[1][t], \
       pos[1][0] == 1, pos[1]'[0] == 0}, pos[1], {t, 0, 2}]; \
       (pos[1] /. sol[[1]])[1.0]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let expected = (2.0_f64.sqrt()).cos();
    assert!(
      (val - expected).abs() < 0.01,
      "Expected {expected}, got {val}"
    );
  }

  #[test]
  fn ndsolve_rhs_calling_an_interpolating_function_stays_fast() {
    // Regression: a right-hand side that looks a value up in a table built
    // with `Interpolation[…]` (`xy = Interpolation[…]; y'[t] == -xy[y[t]]`,
    // a common way to drive an ODE off precomputed data, e.g. a physical
    // property curve) used to fall out of the residual compiler's
    // closed-form arithmetic subset entirely: every RK4 stage re-resolved
    // the call through the full generic evaluator instead of the compiled
    // numeric tree, turning a solve that should take well under a second
    // into one that took minutes. With the table set to the identity
    // function, `y' = -y` from `y(0) = 1` has the closed form `y = e^-t`.
    let start = std::time::Instant::now();
    let result = interpret(
      "xy = Interpolation[Table[{i, N[i]}, {i, -50, 50}], \
       InterpolationOrder -> 1]; \
       sol = NDSolve[{y'[t] == -xy[y[t]], y[0] == 1}, y, {t, 0, 5}]; \
       y[3.0] /. sol[[1]]",
    )
    .unwrap();
    assert!(
      start.elapsed().as_secs() < 5,
      "an Interpolation-driven residual must use the compiled numeric \
       path, not fall back to the full evaluator on every RK4 stage"
    );
    let val: f64 = result.parse().expect("should be a number");
    let expected = (-3.0_f64).exp();
    assert!(
      (val - expected).abs() < 1e-4,
      "Expected {expected}, got {val}"
    );
  }

  #[test]
  fn blow_up_before_domain_end_returns_a_truncated_interpolating_function() {
    // Regression: a solution that diverges to infinity partway through the
    // requested domain (y' = y^2, y(0) = 1 has the closed form
    // y = 1/(1-t), a vertical asymptote at t = 1) used to discard every
    // point integrated up to the blow-up and return NDSolve unevaluated —
    // a shooting-method boundary-value search routinely guesses initial
    // slopes whose trajectory blows up before reaching the requested
    // endpoint, so failing outright on any such guess broke every solve
    // downstream in the search. NDSolve should hand back an
    // InterpolatingFunction truncated to the domain it actually covered,
    // as wolframscript does (matching its `NDSolve::ndsz` behavior of
    // returning a partial solution rather than none at all).
    let result = interpret(
      "sol = NDSolve[{y[t]^2 == y'[t], y[0] == 1}, y, {t, 0, 2}]; \
       Head[sol[[1, 1, 2]]]",
    )
    .unwrap();
    assert_eq!(result, "InterpolatingFunction");

    let domain_end = interpret(
      "sol = NDSolve[{y[t]^2 == y'[t], y[0] == 1}, y, {t, 0, 2}]; \
       sol[[1, 1, 2, 1, 1, 2]]",
    )
    .unwrap();
    let end: f64 = domain_end.parse().expect("should be a number");
    assert!(
      end > 0.9 && end < 1.5,
      "expected the domain to be truncated near the t = 1 asymptote \
       (well short of the requested t = 2), got {end}"
    );
  }

  #[test]
  fn interior_initial_point_integrates_both_directions() {
    // The initial condition sits inside the domain: y' = y, y(1) = 1 on
    // {t, 0, 2} → y(t) = E^(t-1) on both sides of t = 1.
    let result = interpret(
      "s = NDSolve[{w'[t] == w[t], w[1] == 1}, w, {t, 0, 2}]; \
       f = w /. s[[1]]; {f[0.0], f[2.0]}",
    )
    .unwrap();
    let expected_lo = (-1.0_f64).exp();
    let expected_hi = std::f64::consts::E;
    let vals: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(',')
      .map(|p| p.trim().parse().expect("should be numbers"))
      .collect();
    assert!(
      (vals[0] - expected_lo).abs() < 0.01
        && (vals[1] - expected_hi).abs() < 0.01,
      "Expected {{{expected_lo}, {expected_hi}}}, got {result}"
    );
  }

  #[test]
  fn event_locator_stops_integration() {
    // Method -> {"EventLocator", …} stops at the event crossing and runs
    // the (held) EventAction: y' = -y from y(0)=1 crosses 0.5 at ln 2.
    let result = interpret(
      "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 10}, \
       Method -> {\"EventLocator\", \"Event\" -> y[t] - 0.5, \
       \"EventAction\" :> Throw[stopT = t, \"StopIntegration\"]}]; \
       {stopT, (y /. s[[1]])[stopT]}",
    )
    .unwrap();
    let vals: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(',')
      .map(|p| p.trim().parse().expect("should be numbers"))
      .collect();
    assert!(
      (vals[0] - std::f64::consts::LN_2).abs() < 0.001,
      "Expected event at ln 2 ≈ 0.6931, got {}",
      vals[0]
    );
    assert!(
      (vals[1] - 0.5).abs() < 0.001,
      "Expected y at event ≈ 0.5, got {}",
      vals[1]
    );
  }

  #[test]
  fn symbolic_initial_condition_value() {
    // An exact symbolic IC value (like the trebuchet's
    // `θ[0] == -ArcCos[(143 - L4)/L1]`) must numericise.
    let result = interpret(
      "s = NDSolve[{y'[t] == 0, y[0] == -ArcCos[31/40]}, y, {t, 0, 1}]; \
       (y /. s[[1]])[1.0]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let expected = -(31.0_f64 / 40.0).acos();
    assert!(
      (val - expected).abs() < 0.001,
      "Expected {expected}, got {val}"
    );
  }

  #[test]
  fn nonlinear_pendulum_equation() {
    // Nothing about integrating an ODE numerically needs it to be linear
    // in the dependent variable. The damped driven pendulum,
    // `θ'' == -(g/l) Sin[θ] - γ θ' + a Cos[ω t]`, is the classic
    // counter-example — `Sin[θ[t]]` used to make NDSolve give up with
    // "DSolve: cannot classify term involving θ".
    let result = interpret(
      "s = NDSolve[{th''[t] == -4.905 Sin[th[t]] - 1.11 th'[t] \
         + 5.73 Cos[1.48 t], th[0] == -0.075, th'[0] == -0.075}, \
       th, {t, 0, 25}]; (th /. s[[1]])[10.0]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    // Reference: the same initial-value problem integrated with RK4 at
    // 200x this step size.
    assert!(
      (val - -0.6704006).abs() < 1e-4,
      "Expected -0.6704006, got {val}"
    );
  }

  #[test]
  fn nonlinear_first_order_equation() {
    // y' = y², y(0) = 1 has the closed form 1/(1 - t).
    let result = interpret(
      "s = NDSolve[{y'[t] == y[t]^2, y[0] == 1}, y, {t, 0, 0.5}]; \
       (y /. s[[1]])[0.4]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 1.0 / 0.6).abs() < 1e-6, "Expected 1.6667, got {val}");
  }

  #[test]
  fn domain_ends_exactly_where_asked() {
    // Stepping by adding `h` a thousand times drifts off the end of the
    // domain; the reported domain must still be the one asked for.
    assert_eq!(
      interpret("NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]").unwrap(),
      "{{y -> InterpolatingFunction[{{0., 5.}}, <>]}}"
    );
    assert_eq!(
      interpret("NDSolve[{y'[t] == -y[t], y[1] == 1}, y, {t, 1, 3.5}]")
        .unwrap(),
      "{{y -> InterpolatingFunction[{{1., 3.5}}, <>]}}"
    );
  }

  #[test]
  fn solution_interpolates_between_grid_points() {
    // NDSolve's InterpolatingFunction interpolates to third order, as the
    // Wolfram Language's does, so reading the solution back off the grid
    // keeps the accuracy the integration had. Linear interpolation would
    // be out by ~1e-5 here, and its derivative by ~1e-3.
    let value = interpret(
      "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]; \
       (y /. s[[1]])[2.0023]",
    )
    .unwrap()
    .parse::<f64>()
    .expect("should be a number");
    assert!(
      (value - (-2.0023_f64).exp()).abs() < 1e-9,
      "Expected {}, got {value}",
      (-2.0023_f64).exp()
    );
    let slope = interpret(
      "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]; \
       (y /. s[[1]])'[2.0023]",
    )
    .unwrap()
    .parse::<f64>()
    .expect("should be a number");
    assert!(
      (slope - -(-2.0023_f64).exp()).abs() < 1e-7,
      "Expected {}, got {slope}",
      -(-2.0023_f64).exp()
    );
  }

  #[test]
  fn derivative_of_a_part_extracted_solution() {
    // `solutions[[i]][[2]]'[t]` — the prime differentiates the
    // InterpolatingFunction the rule was carrying. It used to be a parse
    // error (the parser demanded a `;;` after the part index).
    let result = interpret(
      "s = Flatten[NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 5}]]; \
       s[[1]][[2]]'[2.0]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - -(-2.0_f64).exp()).abs() < 1e-6,
      "Expected {}, got {val}",
      -(-2.0_f64).exp()
    );
  }

  /// An orthogonal-collocation scheme (used in Demonstrations of nonlinear
  /// heat conduction) pins one member of an indexed family of functions to
  /// a fixed boundary value via `n = N; Y[n][t_] := c`, after the *other*
  /// equations were already captured with a literal, still-unresolved
  /// `Y[N][t]` call inside them. That leftover call is outside the system's
  /// own dependent variables, so the residual can't be reduced to
  /// closed-form arithmetic at compile time — it used to force the *whole*
  /// right-hand side through per-step symbolic re-evaluation for every one
  /// of the fixed grid's 1000 steps, which was slow enough to look like a
  /// hang on a system of even a handful of equations. Only that one
  /// unresolved call should pay the symbolic-evaluation cost now.
  #[test]
  fn a_pinned_external_function_does_not_blow_up_the_solve() {
    // Y3' = 2 - Y3, Y2' = Y3 - Y2, Y1' = Y2 - Y1, all zero initial
    // conditions: a linear cascade with a closed-form solution via
    // repeated convolution with Exp[-t].
    let result = interpret(
      "eq1 = Y[1]'[t] == Y[2][t] - Y[1][t]; \
       eq2 = Y[2]'[t] == Y[3][t] - Y[2][t]; \
       eq3 = Y[3]'[t] == Y[4][t] - Y[3][t]; \
       n = 4; Y[n][t_] := 2; \
       sol = NDSolve[{eq1, eq2, eq3, Y[1][0] == 0, Y[2][0] == 0, \
         Y[3][0] == 0}, {Y[1], Y[2], Y[3]}, {t, 0, 1}]; \
       {Y[1][1.], Y[2][1.], Y[3][1.]} /. sol[[1]]",
    )
    .unwrap();
    let nums: Vec<f64> = result
      .trim_matches(['{', '}'])
      .split(", ")
      .map(|s| s.parse().expect("all three solutions are numbers"))
      .collect();
    let y3 = 2.0 * (1.0 - (-1.0_f64).exp());
    let y2 = 2.0 - 2.0 * (-1.0_f64).exp() * 2.0;
    let y1 = 2.0 - (-1.0_f64).exp() * 5.0;
    for (value, expected, name) in [
      (nums[0], y1, "Y[1]"),
      (nums[1], y2, "Y[2]"),
      (nums[2], y3, "Y[3]"),
    ] {
      assert!(
        (value - expected).abs() < 1e-3,
        "{name}(1) should be about {expected}, got {value} (in {result})"
      );
    }
  }

  /// `{t, tmax}` (no explicit `tmin`) integrates from the initial
  /// conditions' t-value out to `tmax` — the shorthand used throughout the
  /// Wolfram Demonstrations Project for time-only integration. It must
  /// match the explicit three-argument form exactly, and it must also work
  /// backward when `tmax` is on the other side of the initial point.
  #[test]
  fn two_argument_domain_matches_explicit_bounds() {
    let explicit =
      interpret("s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 2}]; (y /. s[[1]])[2]")
        .unwrap();
    let shorthand = interpret(
      "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, 2}]; (y /. s[[1]])[2]",
    )
    .unwrap();
    assert_eq!(explicit, shorthand);

    // Backward: tmax below the initial point integrates the other way.
    let backward = interpret(
      "s = NDSolve[{y'[t] == -y[t], y[0] == 1}, y, {t, -2}]; (y /. s[[1]])[-2]",
    )
    .unwrap();
    let val: f64 = backward.parse().expect("should be a number");
    let expected = 2.0_f64.exp();
    assert!(
      (val - expected).abs() < 1e-3,
      "y(-2) should be about {expected}, got {val}"
    );
  }

  /// A coupled linear system whose coefficients are astronomically larger
  /// than the state (an oscillator with a huge angular frequency, as in
  /// models working in natural units far from 1) used to come back
  /// unevaluated: the Jacobian used to advance each step was recovered by
  /// perturbing the highest-derivative slot by a fixed `1.0` and
  /// subtracting the unperturbed residual, and when that residual is
  /// already of order `1e16` the `+1.0` rounds away in `f64`, so every
  /// entry of that column reads as exactly zero — a falsely singular
  /// pivot. Scaling the perturbation to the residual's own magnitude fixes
  /// it. `x'' = -omega^2 x` with `omega = 10^9` has the closed form
  /// `x(t) = A Cos[omega t]`.
  #[test]
  fn coupled_system_survives_a_huge_coefficient() {
    let result = interpret(
      "s = NDSolve[{x'[t] == v[t], v'[t] == -(10^9)^2 x[t], \
       x[0] == 1, v[0] == 0}, {x, v}, {t, 10^-9}]; (x /. s[[1]])[10^-9]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let expected = 1.0_f64.cos(); // Cos[omega * t] at t = 1/omega.
    assert!(
      (val - expected).abs() < 1e-3,
      "x(1/omega) should be about {expected}, got {val}"
    );
  }

  /// `NDSolve[eqns, u, {t, t0, t1}, {x, x0, x1}]` — four positional
  /// arguments rather than three — solves a scalar parabolic PDE (a
  /// heat/diffusion-equation shape) by the method of lines. A linear
  /// (harmonic) initial profile with boundary conditions consistent with
  /// it is a steady state of the heat equation (`D[x, {x, 2}] == 0`), so
  /// the solution must stay exactly on that line for every `t`.
  #[test]
  fn pde_heat_equation_preserves_a_harmonic_steady_state() {
    let result = interpret(
      "sol = NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], u[0, x] == x, \
       u[t, 0] == 0, u[t, 1] == 1}, u, {t, 0, 1}, {x, 0, 1}]; \
       (u[t, x] /. sol[[1]]) /. {t -> 0.7, x -> 0.4}",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 0.4).abs() < 1e-6, "Expected 0.4, got {val}");
  }

  /// `Sin[Pi x] Exp[-Pi^2 t]` is the textbook separated solution of the
  /// heat equation `u_t == u_xx` with `u(0, x) = Sin[Pi x]` and Dirichlet
  /// zero at both ends — a check independent of the solver's own
  /// discretization.
  #[test]
  fn pde_heat_equation_matches_sinusoidal_decay() {
    let result = interpret(
      "sol = NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], \
       u[0, x] == Sin[Pi x], u[t, 0] == 0, u[t, 1] == 0}, u, {t, 0, 0.1}, \
       {x, 0, 1}]; (u[t, x] /. sol[[1]]) /. {t -> 0.05, x -> 0.5}",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    let expected = (std::f64::consts::PI / 2.0).sin()
      * (-std::f64::consts::PI.powi(2) * 0.05).exp();
    assert!(
      (val - expected).abs() < 1e-2,
      "Expected about {expected}, got {val}"
    );
  }

  #[test]
  fn pde_heat_equation_returns_named_interpolating_function() {
    let result = interpret(
      "NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], u[0, x] == x, \
       u[t, 0] == 0, u[t, 1] == 1}, u, {t, 0, 1}, {x, 0, 1}]",
    )
    .unwrap();
    assert!(
      result.starts_with("{{u -> InterpolatingFunction[")
        && result.contains("{1, 1}"),
      "Got: {result}"
    );
  }

  /// A PDE missing one of its two Dirichlet boundary conditions doesn't
  /// match the one recognised shape, so the call is left unevaluated —
  /// the same fallback NDSolve gives any equation system it can't classify
  /// — rather than panicking or silently returning a wrong answer.
  #[test]
  fn pde_heat_equation_without_both_boundaries_stays_unevaluated() {
    let result = interpret(
      "NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], u[0, x] == x, \
       u[t, 0] == 0}, u, {t, 0, 1}, {x, 0, 1}]",
    )
    .unwrap();
    assert!(
      result.starts_with("NDSolve["),
      "Expected an unevaluated NDSolve, got: {result}"
    );
  }

  /// The PDE branch's `InterpolatingFunction` must be usable exactly like
  /// any other two-argument one — in particular as the function
  /// `ContourPlot` samples over its `{t, x}` grid, the shape a
  /// `Manipulate`-driven heat-equation plot depends on.
  #[test]
  fn pde_heat_equation_feeds_a_contour_plot() {
    let result = interpret(
      "sol = NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], u[0, x] == x, \
       u[t, 0] == 0, u[t, 1] == 1}, u, {t, 0, 1}, {x, 0, 1}]; \
       ExportString[ContourPlot[Evaluate[u[t, x] /. sol[[1]]], {t, 0, 1}, \
       {x, 0, 1}, PlotPoints -> 10, Contours -> {0.5}], \"SVG\"]",
    )
    .unwrap();
    assert!(result.contains("<svg"), "Got: {result}");
  }

  /// The Neumann analogue of `pde_heat_equation_preserves_a_harmonic_steady_state`:
  /// a linear ramp `u = x` is a steady state of `u_t == u_xx` (its second
  /// derivative is zero) whose own slope is `1` everywhere, so it's also
  /// consistent with a zero-flux Dirichlet start (`u(t, 0) == 0`) and a
  /// unit-flux Neumann end (`∂u/∂x(t, 1) == 1`, the parsed form of
  /// `D[u[t, x], x] /. x -> 1`). The solution must stay exactly on that
  /// line for every `t`.
  #[test]
  fn pde_neumann_boundary_preserves_a_harmonic_steady_state() {
    let result = interpret(
      "sol = NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], u[0, x] == x, \
       u[t, 0] == 0, (D[u[t, x], x] /. x -> 1) == 1}, u, {t, 0, 1}, \
       {x, 0, 1}]; (u[t, x] /. sol[[1]]) /. {t -> 0.7, x -> 0.4}",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 0.4).abs() < 1e-6, "Expected 0.4, got {val}");
  }

  /// A coefficient (here a Nusselt-style parabolic velocity profile,
  /// `Vmax (1 - (x/L)^2)`) multiplying the evolution equation's time
  /// derivative — the shape a convection-diffusion-reaction transport
  /// equation across a falling liquid film takes — must be recognised and
  /// divided out. The velocity profile vanishes exactly at the channel
  /// wall (`x == L`), which is also where the zero-flux Neumann condition
  /// sits, so this doubles as a regression test for that degenerate
  /// coefficient: dividing by it must not blow up into `Indeterminate`.
  /// With no closed form to check against, the assertion instead pins down
  /// the two invariants any positive diffusion process with these boundary
  /// values must satisfy: the concentration stays within the range set by
  /// its Dirichlet source (`(0, 1]`), and it's monotonically decreasing
  /// away from that source.
  #[test]
  fn pde_evolution_coefficient_vanishing_at_a_neumann_wall_stays_bounded() {
    let result = interpret(
      "sol = NDSolve[{5 (1 - (x/0.99)^2) D[c[z, x], z] == \
       D[c[z, x], x, x], c[0, x] == 0, c[z, 0] == 1, \
       (D[c[z, x], x] /. x -> 0.99) == 0}, c, {z, 0, 5}, {x, 0, 0.99}]; \
       {c[z, x] /. sol[[1]] /. {z -> 3, x -> 0.2}, \
       c[z, x] /. sol[[1]] /. {z -> 3, x -> 0.9}}",
    )
    .unwrap();
    let vals: Vec<f64> = result
      .trim_start_matches('{')
      .trim_end_matches('}')
      .split(',')
      .map(|s| s.trim().parse().expect("should be a number"))
      .collect();
    let [near, far] = vals[..] else {
      panic!("expected two values, got: {result}");
    };
    assert!(
      (0.0..=1.0).contains(&near) && (0.0..=1.0).contains(&far),
      "Both points should stay within the Dirichlet source's range, got near={near} far={far}"
    );
    assert!(
      near > far,
      "Concentration should decrease with distance from the source: near={near} far={far}"
    );
  }

  /// A coupled reaction pair — `A` decaying into `B` — where the second
  /// unknown's evolution equation references the first unknown's plain
  /// value (not a derivative) for the reaction term. Since `A`'s own
  /// equation doesn't depend on `B`, solving it standalone through the
  /// single-unknown PDE branch and reading it back out of the two-unknown
  /// coupled solve must agree exactly: coupling other unknowns in must
  /// never perturb an unknown whose own equation doesn't reference them.
  #[test]
  fn pde_coupled_system_matches_a_standalone_solve_of_the_uncoupled_unknown() {
    let coupled = interpret(
      "sol = NDSolve[{D[Subscript[c, 1][x, z], z] == \
       D[Subscript[c, 1][x, z], x, x] - 0.5 Subscript[c, 1][x, z], \
       Subscript[c, 1][x, 0] == 0, Subscript[c, 1][0, z] == 1, \
       Subscript[c, 1][1, z] == 0, D[Subscript[c, 2][x, z], z] == \
       D[Subscript[c, 2][x, z], x, x] + 0.5 Subscript[c, 1][x, z], \
       Subscript[c, 2][x, 0] == 0, Subscript[c, 2][0, z] == 0, \
       Subscript[c, 2][1, z] == 0}, {Subscript[c, 1], Subscript[c, 2]}, \
       {x, 0, 1}, {z, 0, 1}]; \
       N[Subscript[c, 1][x, z] /. sol[[1]] /. {x -> 0.5, z -> 0.5}]",
    )
    .unwrap();
    let alone = interpret(
      "sol = NDSolve[{D[u[x, z], z] == D[u[x, z], x, x] - 0.5 u[x, z], \
       u[x, 0] == 0, u[0, z] == 1, u[1, z] == 0}, u, {x, 0, 1}, {z, 0, 1}]; \
       N[u[x, z] /. sol[[1]] /. {x -> 0.5, z -> 0.5}]",
    )
    .unwrap();
    let coupled_val: f64 = coupled.parse().expect("should be a number");
    let alone_val: f64 = alone.parse().expect("should be a number");
    assert!(
      (coupled_val - alone_val).abs() < 1e-9,
      "Coupling in Subscript[c, 2] shouldn't change Subscript[c, 1]'s own \
       solution: coupled={coupled_val}, standalone={alone_val}"
    );
  }

  /// A chained equality `u[0, x] == u[t, 1] == c` — the shorthand a
  /// `NDSolve` demonstration commonly uses to state an initial condition
  /// and a same-valued Dirichlet boundary condition in one equation —
  /// must expand into both conditions rather than leaving the whole
  /// system unrecognised (three equation items don't fit any single
  /// unknown's required four). With a zero-flux wall at the other end,
  /// a uniform initial profile matching the shared boundary value is a
  /// fixed point of the heat equation, so it must stay exactly at that
  /// value everywhere.
  #[test]
  fn pde_chained_equality_states_initial_and_dirichlet_boundary_together() {
    let result = interpret(
      "sol = NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], \
       u[0, x] == u[t, 1] == 1, \
       (D[u[t, x], x] /. x -> 0) == 0}, u, {t, 0, 1}, {x, 0, 1}]; \
       (u[t, x] /. sol[[1]]) /. {t -> 0.5, x -> 0}",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 1.0).abs() < 1e-6, "Expected 1., got {val}");
  }

  /// A Neumann boundary condition whose flux is a Robin-type formula
  /// referencing the unknown's own plain value at that same boundary
  /// (`u[t, 0]`, not just `t`) — the shape a convective or reaction-rate
  /// boundary condition takes (`flux ∝ u - u_target`). A uniform initial
  /// profile at the Dirichlet value doubles as the Robin condition's own
  /// target, so the flux term vanishes identically and the whole system
  /// is already a steady state: it must stay exactly there.
  #[test]
  fn pde_neumann_flux_references_the_unknowns_own_boundary_value() {
    let result = interpret(
      "sol = NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], \
       u[0, x] == u[t, 1] == 2, \
       (D[u[t, x], x] /. x -> 0) == 3 (u[t, 0] - 2)}, \
       u, {t, 0, 1}, {x, 0, 1}]; \
       (u[t, x] /. sol[[1]]) /. {t -> 0.6, x -> 0.3}",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 2.0).abs() < 1e-6, "Expected 2., got {val}");
  }

  /// The same Robin-type flux away from its fixed point: with the
  /// boundary relaxing toward a target above its uniform start, the
  /// unknown's value at that wall must climb monotonically toward the
  /// target over time, while staying inside the physical range the
  /// target and the far Dirichlet value bound it to, and a point deep in
  /// the domain — not yet reached by diffusion from the wall — must
  /// still read far closer to its own (lower) initial value.
  #[test]
  fn pde_neumann_flux_drives_the_boundary_toward_its_target_over_time() {
    let result = interpret(
      "sol = NDSolve[{D[u[t, x], t] == D[u[t, x], {x, 2}], \
       u[0, x] == u[t, 1] == 0, \
       (D[u[t, x], x] /. x -> 0) == 2 (u[t, 0] - 1)}, \
       u, {t, 0, 0.3}, {x, 0, 1}]; \
       N[{u[t, x] /. sol[[1]] /. {t -> 0.1, x -> 0}, \
          u[t, x] /. sol[[1]] /. {t -> 0.3, x -> 0}, \
          u[t, x] /. sol[[1]] /. {t -> 0.3, x -> 0.9}}]",
    )
    .unwrap();
    let vals: Vec<f64> = result
      .trim_start_matches('{')
      .trim_end_matches('}')
      .split(',')
      .map(|s| s.trim().parse().expect("should be a number"))
      .collect();
    let [early_wall, late_wall, late_far] = vals[..] else {
      panic!("expected three values, got: {result}");
    };
    assert!(
      (0.0..=1.0).contains(&early_wall) && (0.0..=1.0).contains(&late_wall),
      "the wall's value must stay within the range set by its target and \
       its own start: early={early_wall}, late={late_wall}"
    );
    assert!(
      late_wall > early_wall,
      "the wall should keep climbing toward its Robin target over time: \
       early={early_wall}, late={late_wall}"
    );
    assert!(
      late_far < early_wall,
      "a point far from the wall shouldn't yet have caught up to where \
       the wall itself was earlier: late_far={late_far}, early_wall={early_wall}"
    );
  }

  /// Two coupled unknowns whose boundary condition at one end isn't
  /// either one's own — it's a shared flux-conservation law spanning
  /// both, `(D[p, x] + D[q, x]) /. x -> 0 == 0` (no net accumulation at
  /// a reacting interface), alongside `p`'s own Robin condition that
  /// consumes `q`'s boundary value. With equal diffusivities and
  /// boundary/initial values for `p` and `q` that sum to the same
  /// constant at every one of the *other* three conditions (the shared
  /// interface's own flux being conserved, not fixed, contributes
  /// nothing extra), `p + q` itself satisfies the heat equation with a
  /// uniform initial profile, a zero-flux wall, and a Dirichlet value
  /// all equal to that constant — so it must stay exactly there
  /// everywhere, for both ends of the domain.
  #[test]
  fn pde_coupled_flux_conservation_boundary_splits_a_conserved_total() {
    let result = interpret(
      "sol = NDSolve[{D[p[t, x], t] == D[p[t, x], {x, 2}], \
       p[0, x] == p[t, 1] == 1, \
       (D[p[t, x], x] /. x -> 0) == 0.5 (p[t, 0] - q[t, 0]), \
       D[q[t, x], t] == D[q[t, x], {x, 2}], \
       q[0, x] == q[t, 1] == 0, \
       ((D[p[t, x], x] + D[q[t, x], x]) /. x -> 0) == 0}, \
       {p, q}, {t, 0, 0.3}, {x, 0, 1}]; \
       N[{p[t, x] /. sol[[1]] /. {t -> 0.3, x -> 0}, \
          q[t, x] /. sol[[1]] /. {t -> 0.3, x -> 0}, \
          p[t, x] /. sol[[1]] /. {t -> 0.3, x -> 0.6}, \
          q[t, x] /. sol[[1]] /. {t -> 0.3, x -> 0.6}}]",
    )
    .unwrap();
    let vals: Vec<f64> = result
      .trim_start_matches('{')
      .trim_end_matches('}')
      .split(',')
      .map(|s| s.trim().parse().expect("should be a number"))
      .collect();
    let [p_wall, q_wall, p_mid, q_mid] = vals[..] else {
      panic!("expected four values, got: {result}");
    };
    assert!(
      (p_wall + q_wall - 1.0).abs() < 1e-6,
      "p + q must stay conserved at the shared wall: p={p_wall}, q={q_wall}"
    );
    assert!(
      (p_mid + q_mid - 1.0).abs() < 1e-6,
      "p + q must stay conserved away from the wall too: p={p_mid}, q={q_mid}"
    );
  }
}

mod sinh_cosh {
  use super::*;

  #[test]
  fn d_sinh() {
    assert_eq!(interpret("D[Sinh[x], x]").unwrap(), "Cosh[x]");
  }

  #[test]
  fn d_cosh() {
    assert_eq!(interpret("D[Cosh[x], x]").unwrap(), "Sinh[x]");
  }

  #[test]
  fn d_sinh_chain_rule() {
    assert_eq!(interpret("D[Sinh[2*x], x]").unwrap(), "2*Cosh[2*x]");
  }

  #[test]
  fn d_cosh_chain_rule() {
    assert_eq!(interpret("D[Cosh[3*x], x]").unwrap(), "3*Sinh[3*x]");
  }

  #[test]
  fn integrate_sinh() {
    assert_eq!(interpret("Integrate[Sinh[x], x]").unwrap(), "Cosh[x]");
  }

  #[test]
  fn integrate_cosh() {
    assert_eq!(interpret("Integrate[Cosh[x], x]").unwrap(), "Sinh[x]");
  }

  #[test]
  fn integrate_sinh_linear_arg() {
    assert_eq!(interpret("Integrate[Sinh[2*x], x]").unwrap(), "Cosh[2*x]/2");
  }

  #[test]
  fn integrate_cosh_linear_arg() {
    assert_eq!(interpret("Integrate[Cosh[3*x], x]").unwrap(), "Sinh[3*x]/3");
  }
}

mod tanh_sech_csch_coth {
  use super::*;

  #[test]
  fn d_tanh() {
    assert_eq!(interpret("D[Tanh[x], x]").unwrap(), "Sech[x]^2");
  }

  #[test]
  fn d_sech() {
    assert_eq!(interpret("D[Sech[x], x]").unwrap(), "-(Sech[x]*Tanh[x])");
  }

  #[test]
  fn d_csch() {
    assert_eq!(interpret("D[Csch[x], x]").unwrap(), "-(Coth[x]*Csch[x])");
  }

  #[test]
  fn d_coth() {
    assert_eq!(interpret("D[Coth[x], x]").unwrap(), "-Csch[x]^2");
  }

  #[test]
  fn d_tanh_chain_rule() {
    assert_eq!(interpret("D[Tanh[2*x], x]").unwrap(), "2*Sech[2*x]^2");
  }

  // Gudermannian / InverseGudermannian / Haversine derivatives.
  #[test]
  fn d_gudermannian() {
    assert_eq!(interpret("D[Gudermannian[x], x]").unwrap(), "Sech[x]");
    assert_eq!(
      interpret("D[Gudermannian[x^2], x]").unwrap(),
      "2*x*Sech[x^2]"
    );
  }

  #[test]
  fn d_inverse_gudermannian() {
    assert_eq!(interpret("D[InverseGudermannian[x], x]").unwrap(), "Sec[x]");
    assert_eq!(
      interpret("D[InverseGudermannian[3*x], x]").unwrap(),
      "3*Sec[3*x]"
    );
  }

  #[test]
  fn d_haversine() {
    assert_eq!(interpret("D[Haversine[x], x]").unwrap(), "Sin[x]/2");
    assert_eq!(interpret("D[Haversine[3*x], x]").unwrap(), "(3*Sin[3*x])/2");
    assert_eq!(interpret("D[Haversine[x^2], x]").unwrap(), "x*Sin[x^2]");
  }
}

mod inverse_trig_derivatives {
  use super::*;

  #[test]
  fn d_arcsin() {
    assert_eq!(interpret("D[ArcSin[x], x]").unwrap(), "1/Sqrt[1 - x^2]");
  }

  #[test]
  fn d_arccos() {
    assert_eq!(interpret("D[ArcCos[x], x]").unwrap(), "-(1/Sqrt[1 - x^2])");
  }

  #[test]
  fn d_arctan() {
    assert_eq!(interpret("D[ArcTan[x], x]").unwrap(), "(1 + x^2)^(-1)");
  }

  #[test]
  fn d_arccot() {
    assert_eq!(interpret("D[ArcCot[x], x]").unwrap(), "-(1 + x^2)^(-1)");
  }

  #[test]
  fn d_arctan_chain_rule() {
    assert_eq!(interpret("D[ArcTan[2*x], x]").unwrap(), "2/(1 + 4*x^2)");
  }

  #[test]
  fn d_arctan_quotient_no_panic() {
    // Regression test: D[ArcTan[x/y], x] used to panic with integer overflow
    let result = interpret("D[ArcTan[x/y], x]").unwrap();
    assert!(!result.is_empty(), "Should return a result, not panic");
  }

  #[test]
  fn d_arcsin_chain_rule() {
    assert_eq!(interpret("D[ArcSin[3*x], x]").unwrap(), "3/Sqrt[1 - 9*x^2]");
  }
}

mod inverse_hyperbolic_derivatives {
  use super::*;

  #[test]
  fn d_arcsinh() {
    assert_eq!(interpret("D[ArcSinh[x], x]").unwrap(), "1/Sqrt[1 + x^2]");
  }

  #[test]
  fn d_arccosh() {
    assert_eq!(
      interpret("D[ArcCosh[x], x]").unwrap(),
      "1/(Sqrt[-1 + x]*Sqrt[1 + x])"
    );
  }

  #[test]
  fn d_arctanh() {
    assert_eq!(interpret("D[ArcTanh[x], x]").unwrap(), "(1 - x^2)^(-1)");
  }

  // ArcCoth shares ArcTanh's derivative 1/(1 - x^2).
  #[test]
  fn d_arccoth() {
    assert_eq!(interpret("D[ArcCoth[x], x]").unwrap(), "(1 - x^2)^(-1)");
  }

  #[test]
  fn d_arccoth_chain_rule() {
    assert_eq!(interpret("D[ArcCoth[x^2], x]").unwrap(), "(2*x)/(1 - x^4)");
    assert_eq!(interpret("D[ArcCoth[3*x], x]").unwrap(), "3/(1 - 9*x^2)");
    assert_eq!(
      interpret("D[ArcCoth[Sin[x]], x]").unwrap(),
      "Cos[x]/(1 - Sin[x]^2)"
    );
  }

  #[test]
  fn d_arcsinh_chain_rule() {
    assert_eq!(
      interpret("D[ArcSinh[2*x], x]").unwrap(),
      "2/Sqrt[1 + 4*x^2]"
    );
  }
}

mod integrate_log {
  use super::*;

  #[test]
  fn integrate_log_x() {
    assert_eq!(
      interpret("Integrate[Log[x], x]").unwrap(),
      "x*(-1 + Log[x])"
    );
  }

  // ∫ Log[x]^n dx = x Σ (-1)^(n-k) (n!/k!) Log[x]^k.
  #[test]
  fn integrate_log_squared() {
    assert_eq!(
      interpret("Integrate[Log[x]^2, x]").unwrap(),
      "x*(2 - 2*Log[x] + Log[x]^2)"
    );
  }

  #[test]
  fn integrate_log_cubed() {
    assert_eq!(
      interpret("Integrate[Log[x]^3, x]").unwrap(),
      "x*(-6 + 6*Log[x] - 3*Log[x]^2 + Log[x]^3)"
    );
  }

  #[test]
  fn integrate_log_power_other_variable() {
    assert_eq!(
      interpret("Integrate[Log[y]^2, y]").unwrap(),
      "y*(2 - 2*Log[y] + Log[y]^2)"
    );
  }

  // Definite ∫_1^E Log[x]^2 dx = E - 2.
  #[test]
  fn integrate_log_squared_definite() {
    assert_eq!(
      interpret("Integrate[Log[x]^2, {x, 1, E}]").unwrap(),
      "-2 + E"
    );
  }

  // ∫ Log[Log[u]] dx = (u Log[Log[u]])/a - LogIntegral[u]/a for u = a x + b.
  #[test]
  fn integrate_log_log_x() {
    assert_eq!(
      interpret("Integrate[Log[Log[x]], x]").unwrap(),
      "x*Log[Log[x]] - LogIntegral[x]"
    );
  }

  #[test]
  fn integrate_log_log_scaled() {
    assert_eq!(
      interpret("Integrate[Log[Log[2*x]], x]").unwrap(),
      "x*Log[Log[2*x]] - LogIntegral[2*x]/2"
    );
  }

  #[test]
  fn integrate_log_log_symbolic_coefficient() {
    assert_eq!(
      interpret("Integrate[Log[Log[a*x]], x]").unwrap(),
      "x*Log[Log[a*x]] - LogIntegral[a*x]/a"
    );
  }

  #[test]
  fn integrate_log_log_shifted() {
    assert_eq!(
      interpret("Integrate[Log[Log[x + 1]], x]").unwrap(),
      "(1 + x)*Log[Log[1 + x]] - LogIntegral[1 + x]"
    );
  }

  #[test]
  fn integrate_log_log_linear() {
    assert_eq!(
      interpret("Integrate[Log[Log[2*x + 3]], x]").unwrap(),
      "((3 + 2*x)*Log[Log[3 + 2*x]])/2 - LogIntegral[3 + 2*x]/2"
    );
  }

  // Regression: LogIntegral (and friends) sort as transcendental terms in
  // a Plus, after polynomial-like terms and alphabetically among function
  // calls — `Log[x] + LogIntegral[x]`, not `LogIntegral[x] + Log[x]`.
  #[test]
  fn log_integral_plus_ordering() {
    assert_eq!(
      interpret("Log[x] + LogIntegral[x]").unwrap(),
      "Log[x] + LogIntegral[x]"
    );
    assert_eq!(
      interpret("x + LogIntegral[x]").unwrap(),
      "x + LogIntegral[x]"
    );
    assert_eq!(
      interpret("Sin[x] + SinIntegral[x]").unwrap(),
      "Sin[x] + SinIntegral[x]"
    );
    assert_eq!(interpret("Erf[x] + Erfi[x]").unwrap(), "Erf[x] + Erfi[x]");
  }
}

mod integrate_by_parts {
  use super::*;

  #[test]
  fn x_sin_x() {
    assert_eq!(
      interpret("Integrate[x*Sin[x], x]").unwrap(),
      "-(x*Cos[x]) + Sin[x]"
    );
  }

  #[test]
  fn x_exp_x() {
    assert_eq!(interpret("Integrate[x*Exp[x], x]").unwrap(), "E^x*(-1 + x)");
  }

  #[test]
  fn x_squared_exp_x() {
    assert_eq!(
      interpret("Integrate[x^2*Exp[x], x]").unwrap(),
      "E^x*(2 - 2*x + x^2)"
    );
  }

  #[test]
  fn x_cos_x() {
    assert_eq!(
      interpret("Integrate[x*Cos[x], x]").unwrap(),
      "Cos[x] + x*Sin[x]"
    );
  }

  #[test]
  fn x_sinh_x() {
    assert_eq!(
      interpret("Integrate[x*Sinh[x], x]").unwrap(),
      "x*Cosh[x] - Sinh[x]"
    );
  }

  #[test]
  fn x_cosh_x() {
    assert_eq!(
      interpret("Integrate[x*Cosh[x], x]").unwrap(),
      "-Cosh[x] + x*Sinh[x]"
    );
  }

  #[test]
  fn x_log_x() {
    assert_eq!(
      interpret("Integrate[x*Log[x], x]").unwrap(),
      "(x^2*(-1 + 2*Log[x]))/4"
    );
  }

  // General constant-base exponential (lowercase e is a symbol, not Euler's E)
  #[test]
  fn general_exp_basic() {
    assert_eq!(interpret("Integrate[e^x, x]").unwrap(), "e^x/Log[e]");
  }

  #[test]
  fn exp_real_overflow_returns_overflow_object() {
    // Wolfram emits General::ovfl and returns Overflow[] when a real
    // Exp argument blows past the f64 range. Woxi was returning bare
    // Infinity (the f64 saturate value).
    assert_eq!(interpret("Exp[10.*^20]").unwrap(), "Overflow[]");
  }

  #[test]
  fn x_general_exp() {
    assert_eq!(
      interpret("Integrate[x * e^x, x]").unwrap(),
      "(e^x*(-1 + x*Log[e]))/Log[e]^2"
    );
  }

  #[test]
  fn x_squared_general_exp() {
    assert_eq!(
      interpret("Integrate[x^2 * e^x, x]").unwrap(),
      "(e^x*(2 - 2*x*Log[e] + x^2*Log[e]^2))/Log[e]^3"
    );
  }

  #[test]
  fn general_exp_differentiation() {
    assert_eq!(interpret("D[e^x, x]").unwrap(), "e^x*Log[e]");
  }

  #[test]
  fn x4_exp_x_half() {
    // ∫ x^4 * E^(x/2) dx using closed-form poly × E^(cx) integration
    assert_eq!(
      interpret("Integrate[x^4 * E^(x/2), x]").unwrap(),
      "E^(x/2)*(768 - 384*x + 96*x^2 - 16*x^3 + 2*x^4)"
    );
  }

  #[test]
  fn exp_x_half() {
    // ∫ E^(x/2) dx = 2*E^(x/2)
    assert_eq!(interpret("Integrate[E^(x/2), x]").unwrap(), "2*E^(x/2)");
  }

  #[test]
  fn exp_x_third() {
    // ∫ E^(x/3) dx = 3*E^(x/3)
    assert_eq!(interpret("Integrate[E^(x/3), x]").unwrap(), "3*E^(x/3)");
  }

  #[test]
  fn x_exp_x_half() {
    // ∫ x * E^(x/2) dx
    assert_eq!(
      interpret("Integrate[x * E^(x/2), x]").unwrap(),
      "E^(x/2)*(-4 + 2*x)"
    );
  }
}

mod integrate_u_substitution {
  use super::*;

  #[test]
  fn x_exp_neg_x_squared() {
    assert_eq!(
      interpret("Integrate[x Exp[-x^2], x]").unwrap(),
      "-1/2*1/E^x^2"
    );
  }

  #[test]
  fn x_exp_x_squared() {
    assert_eq!(interpret("Integrate[x Exp[x^2], x]").unwrap(), "E^x^2/2");
  }

  #[test]
  fn x_squared_exp_x_cubed() {
    assert_eq!(interpret("Integrate[x^2 Exp[x^3], x]").unwrap(), "E^x^3/3");
  }

  #[test]
  fn cos_x_exp_sin_x() {
    assert_eq!(
      interpret("Integrate[Cos[x] Exp[Sin[x]], x]").unwrap(),
      "E^Sin[x]"
    );
  }

  #[test]
  fn x_sin_x_squared() {
    assert_eq!(
      interpret("Integrate[x Sin[x^2], x]").unwrap(),
      "-1/2*Cos[x^2]"
    );
  }

  #[test]
  fn x_cos_x_squared() {
    assert_eq!(interpret("Integrate[x Cos[x^2], x]").unwrap(), "Sin[x^2]/2");
  }

  #[test]
  fn log_x_over_x() {
    // ∫ Log[x]/x dx = Log[x]^2/2 via u = Log[x]
    assert_eq!(interpret("Integrate[Log[x]/x, x]").unwrap(), "Log[x]^2/2");
  }

  #[test]
  fn sin_cos_product_u_sub() {
    // ∫ Sin[x]*Cos[x] dx = -Cos[x]^2/2 via u = Cos[x]
    assert_eq!(
      interpret("Integrate[Sin[x] Cos[x], x]").unwrap(),
      "-1/2*Cos[x]^2"
    );
  }
}

mod integrate_polynomial_power {
  use super::*;

  #[test]
  fn x_plus_1_squared() {
    // ∫ (x+1)^2 dx — expand then integrate term-by-term
    assert_eq!(
      interpret("Integrate[(x + 1)^2, x]").unwrap(),
      "x + x^2 + x^3/3"
    );
  }

  #[test]
  fn x_plus_1_cubed() {
    // ∫ (x+1)^3 dx — substitution form for n >= 3
    let result = interpret("Integrate[(x + 1)^3, x]").unwrap();
    assert_eq!(result, "(1 + x)^4/4");
  }

  #[test]
  fn two_x_minus_1_squared() {
    // ∫ (2x-1)^2 dx — expand then integrate term-by-term
    assert_eq!(
      interpret("Integrate[(2*x - 1)^2, x]").unwrap(),
      "x - 2*x^2 + (4*x^3)/3"
    );
  }

  #[test]
  fn nested_definite_with_exp_bound_to_log13() {
    // Area under y = e^x over [0, Log[13]] equals 13 - 1 = 12.
    assert_eq!(
      interpret("Integrate[Integrate[1,{y,0,E^x}],{x,0,Log[13]}]").unwrap(),
      "12"
    );
  }
}

mod integrate_exp_integral_ei {
  use super::*;

  #[test]
  fn exp_2x_over_2x() {
    // ∫ E^(2x) / (2*x) dx = ExpIntegralEi[2*x] / 2
    assert_eq!(
      interpret("Integrate[E^(2x) / (2*x), x]").unwrap(),
      "ExpIntegralEi[2*x]/2"
    );
  }

  #[test]
  fn exp_x_over_x() {
    // ∫ E^x / x dx = ExpIntegralEi[x]
    assert_eq!(
      interpret("Integrate[E^x / x, x]").unwrap(),
      "ExpIntegralEi[x]"
    );
  }

  #[test]
  fn exp_3x_over_x() {
    // ∫ E^(3x) / x dx = ExpIntegralEi[3*x]
    assert_eq!(
      interpret("Integrate[E^(3x) / x, x]").unwrap(),
      "ExpIntegralEi[3*x]"
    );
  }

  #[test]
  fn exp_x_over_3x() {
    // ∫ E^x / (3*x) dx = ExpIntegralEi[x] / 3
    assert_eq!(
      interpret("Integrate[E^x / (3*x), x]").unwrap(),
      "ExpIntegralEi[x]/3"
    );
  }
}

// ∫ f(a x)/(c x) dx for the trig/hyperbolic exponential-integral family.
// Verified against wolframscript.
mod integrate_si_ci {
  use super::*;

  #[test]
  fn sin_and_cos() {
    assert_eq!(
      interpret("Integrate[Sin[x]/x, x]").unwrap(),
      "SinIntegral[x]"
    );
    assert_eq!(
      interpret("Integrate[Cos[x]/x, x]").unwrap(),
      "CosIntegral[x]"
    );
    // Linear coefficient carries into the argument.
    assert_eq!(
      interpret("Integrate[Sin[2 x]/x, x]").unwrap(),
      "SinIntegral[2*x]"
    );
    assert_eq!(
      interpret("Integrate[Cos[3 x]/x, x]").unwrap(),
      "CosIntegral[3*x]"
    );
    // A constant denominator factor divides the result.
    assert_eq!(
      interpret("Integrate[Sin[x]/(2 x), x]").unwrap(),
      "SinIntegral[x]/2"
    );
  }

  #[test]
  fn hyperbolic() {
    assert_eq!(
      interpret("Integrate[Sinh[x]/x, x]").unwrap(),
      "SinhIntegral[x]"
    );
    assert_eq!(
      interpret("Integrate[Cosh[x]/x, x]").unwrap(),
      "CoshIntegral[x]"
    );
  }

  #[test]
  fn nonmatching_forms_stay_unevaluated() {
    // Tan is not part of the family.
    assert_eq!(
      interpret("Integrate[Tan[x]/x, x]").unwrap(),
      "Integrate[Tan[x]/x, x]"
    );
  }

  // ∫ f[x^n]/x dx = FIntegral[x^n]/n via the substitution u = x^n.
  #[test]
  fn power_argument_over_x() {
    assert_eq!(
      interpret("Integrate[Sin[x^2]/x, x]").unwrap(),
      "SinIntegral[x^2]/2"
    );
    assert_eq!(
      interpret("Integrate[Cos[x^2]/x, x]").unwrap(),
      "CosIntegral[x^2]/2"
    );
    assert_eq!(
      interpret("Integrate[Sin[x^3]/x, x]").unwrap(),
      "SinIntegral[x^3]/3"
    );
    assert_eq!(
      interpret("Integrate[Sinh[x^2]/x, x]").unwrap(),
      "SinhIntegral[x^2]/2"
    );
  }
}

mod sqrt_differentiation {
  use super::*;

  #[test]
  fn d_sqrt_x() {
    assert_eq!(interpret("D[Sqrt[x], x]").unwrap(), "1/(2*Sqrt[x])");
  }

  #[test]
  fn d_sqrt_chain_rule() {
    // D[Sqrt[1 + x^2], x] = x/Sqrt[1 + x^2]
    assert_eq!(interpret("D[Sqrt[1 + x^2], x]").unwrap(), "x/Sqrt[1 + x^2]");
  }

  #[test]
  fn d_sqrt_constant() {
    assert_eq!(interpret("D[Sqrt[5], x]").unwrap(), "0");
  }
}

mod nmaximize {
  use super::*;

  #[test]
  fn nmaximize_sin() {
    // NMaximize[{Sin[x], 0 < x < 2*Pi}, x] finds the max at Pi/2. The Newton
    // polish on the gradient converges the objective to an exact 1. (it used
    // to stall at 0.99999… on the objective's float plateau).
    // wolframscript: {1., {x -> 1.5707963268033855}} (x is solver noise
    // around Pi/2; woxi converges to Pi/2 at machine precision).
    assert_eq!(
      interpret("NMaximize[{Sin[x], 0 < x < 2*Pi}, x]").unwrap(),
      "{1., {x -> 1.5707963267948966}}"
    );
  }

  #[test]
  fn nminimize_quadratic() {
    // NMinimize[{x^2 - 4*x + 5, -10 < x < 10}, x] should find min at x=2
    let result =
      interpret("NMinimize[{x^2 - 4*x + 5, -10 < x < 10}, x]").unwrap();
    assert!(
      result.starts_with('{'),
      "Expected list result, got: {result}"
    );
    assert!(result.contains("1."), "Min should be ~1, got: {result}");
  }

  #[test]
  fn nminimize_quartic_saddle_point() {
    // x^4 - 3x^2 + 2 has a local max at x=0 (value 2) and
    // global minima at x=±sqrt(3/2) (value -0.25).
    // Must not get stuck at the saddle point x=0.
    let result = interpret("NMinimize[x^4 - 3 x^2 + 2, x]").unwrap();
    assert!(
      result.starts_with("{-0.25"),
      "Min should be ~-0.25, got: {result}"
    );
  }

  #[test]
  fn nminimize_unconstrained_quadratic() {
    let result = interpret("NMinimize[x^2, x]").unwrap();
    assert!(result.starts_with("{0."), "Min should be ~0, got: {result}");
  }

  #[test]
  fn nminimize_takes_trailing_options() {
    // Options follow the two positional arguments; an unhandled one is
    // accepted silently rather than rejected on arity. Rounded because the
    // two solvers stop at slightly different points.
    clear_state();
    assert_eq!(
      interpret(
        "r = NMinimize[{(x - 2)^2, 0 < x < 10}, {x}, MaxIterations -> 50]; \
         {Round[r[[1]], 10^-6], Round[x /. r[[2]], 10^-6]}"
      )
      .unwrap(),
      "{0, 2}"
    );
    // A monitor, though, is reported as unsupported: the constrained solver
    // is not an iterative method whose steps are meaningful, so nothing is
    // sown (see `constrained_maximum_reports_that_it_cannot_monitor_steps`).
    clear_state();
    let r = woxi::interpret_with_stdout(
      "Length[Reap[NMinimize[{(x - 2)^2, 0 < x < 10}, {x}, \
       StepMonitor :> Sow[x]]][[2]]]",
    )
    .unwrap();
    assert_eq!(r.result, "0");
    assert!(r.warnings[0].contains(
      "NMinimize::noopmon: The optimization was solved by an algorithm \
       that does not provide monitoring information."
    ));
  }

  #[test]
  fn nminimize_periodic_prefers_near_origin_optimum() {
    // A periodic objective has infinitely many equally good minima across
    // the default ±10^6 box; wolframscript starts its search near the origin
    // and reports the minimum there with the objective converged to an exact
    // -1. Regression for `{-0.9999999999999998, {x -> 960000.0291023118}}`
    // (a distant sample won the grid scan and the descent stalled on the
    // objective's float plateau).
    // wolframscript: {-1., {x -> -1.5707963267952805}} (x is solver noise
    // around -Pi/2; woxi converges to -Pi/2 at machine precision).
    assert_eq!(
      interpret("NMinimize[Sin[x], x]").unwrap(),
      "{-1., {x -> -1.5707963267948966}}"
    );
  }

  #[test]
  fn nminimize_equality_coupling_constraint() {
    // A constraint coupling two variables (x + y == 1) can't be reduced to
    // per-variable box bounds; the numeric solver must respect it rather than
    // ignore it. Minimum of x^2+y^2 on x+y==1 is at x=y=1/2, value 1/2.
    let result =
      interpret("NMinimize[{x^2 + y^2, x + y == 1}, {x, y}]").unwrap();
    assert_eq!(result, "{0.5, {x -> 0.5, y -> 0.5}}");
  }

  #[test]
  fn nminimize_equality_coupling_constraint_2() {
    // Minimum of x^2+y^2 on x+y==2 is at x=y=1, value 2.
    let result =
      interpret("NMinimize[{x^2 + y^2, x + y == 2}, {x, y}]").unwrap();
    assert_eq!(result, "{2., {x -> 1., y -> 1.}}");
  }

  #[test]
  fn nminimize_equality_coupling_weighted() {
    // Minimum of 2x^2+y^2 on x+y==3 is at x=1, y=2, value 6.
    let result =
      interpret("NMinimize[{2 x^2 + y^2, x + y == 3}, {x, y}]").unwrap();
    assert_eq!(result, "{6., {x -> 1., y -> 2.}}");
  }

  // A multi-variable problem with per-variable box bounds whose ranges are
  // empty (x in [5, 2]) is infeasible. Each conjunct mentions only one
  // variable, so it must go through the box sampler (which detects the empty
  // range) rather than the coupling path — and report NMinimize::nsol with
  // {Infinity, {x -> Indeterminate, y -> Indeterminate}}, matching
  // wolframscript.
  #[test]
  fn nminimize_infeasible_box_multivariable() {
    let result =
      interpret("NMinimize[{x + y, x >= 5 && x <= 2 && y >= 0}, {x, y}]")
        .unwrap();
    assert_eq!(
      result,
      "{Infinity, {x -> Indeterminate, y -> Indeterminate}}"
    );
  }
}

mod findroot_symbolic_start {
  use super::*;

  #[test]
  fn findroot_pi_over_4() {
    // FindRoot should accept Pi/4 as starting point
    let result = interpret("FindRoot[Sin[x] - 0.5, {x, Pi/4}]").unwrap();
    assert!(
      result.contains("x ->"),
      "Expected rule result, got: {result}"
    );
  }

  #[test]
  fn findroot_sin_x_equals_x_at_origin() {
    assert_eq!(
      interpret("FindRoot[Sin[x] == x, {x, 0}]").unwrap(),
      "{x -> 0.}"
    );
  }

  // Multivariate FindRoot: a system of equations with one {var, start} per
  // variable, solved by multidimensional Newton iteration.
  #[test]
  fn findroot_multivariate_linear() {
    assert_eq!(
      interpret("FindRoot[{x + y == 3, x - y == 1}, {{x, 0}, {y, 0}}]")
        .unwrap(),
      "{x -> 2., y -> 1.}"
    );
  }

  #[test]
  fn findroot_multivariate_nonlinear() {
    assert_eq!(
      interpret("FindRoot[{x^2 + y^2 == 1, x == y}, {{x, 0.5}, {y, 0.5}}]")
        .unwrap(),
      "{x -> 0.7071067811865476, y -> 0.7071067811865476}"
    );
  }

  // A per-variable {var, x0, x1} secant/two-point spec, not just {var, x0},
  // is also valid in the documented trailing-argument multivariate form and
  // in the nested-list form — this is what a shooting-method Demonstration's
  // `FindRoot[{eqn1, eqn2}, {a, a0, a1}, {b, b0, b1}]` uses. Regression test
  // for a bug where a two-point spec fell out of multivariate detection
  // entirely (which requires every spec to have exactly 2 elements),
  // silently dropped the second variable, and searched only the first as if
  // it were a single-variable secant problem.
  #[test]
  fn findroot_multivariate_two_point_trailing_args() {
    assert_eq!(
      interpret(
        "FindRoot[{x + y - 1 == 0, x - y - 0.5 == 0}, {x, 0.1, 0.2}, {y, 0.1, 0.2}]"
      )
      .unwrap(),
      "{x -> 0.75, y -> 0.25}"
    );
  }

  #[test]
  fn findroot_multivariate_two_point_nested_list() {
    assert_eq!(
      interpret(
        "FindRoot[{x + y - 1 == 0, x - y - 0.5 == 0}, {{x, 0.1, 0.2}, {y, 0.1, 0.2}}]"
      )
      .unwrap(),
      "{x -> 0.75, y -> 0.25}"
    );
  }

  #[test]
  fn findroot_multivariate_two_point_with_max_iterations() {
    assert_eq!(
      interpret(
        "FindRoot[{x + y - 1 == 0, x - y - 0.5 == 0}, {x, 0.1, 0.2}, {y, 0.1, 0.2}, MaxIterations -> 500]"
      )
      .unwrap(),
      "{x -> 0.75, y -> 0.25}"
    );
  }

  // A multivariate system built from opaque (non-symbolically-differentiable)
  // user functions has no symbolic Jacobian entry, so the solver falls back
  // to Broyden's method (a finite-difference Jacobian seeded once, then
  // cheaply rank-1 updated) rather than recomputing a finite-difference
  // Jacobian every iteration. Regression test for that fallback converging
  // to the correct root rather than stopping early on a spuriously "small"
  // backtracked step.
  #[test]
  fn findroot_multivariate_opaque_function_broyden() {
    assert_eq!(
      interpret(
        "f[a_?NumericQ] := a^2 - 2; g[a_?NumericQ, b_?NumericQ] := a + b - 3; FindRoot[{f[x] == 0, g[x, y] == 0}, {x, 1, 1.2}, {y, 1, 1.2}]"
      )
      .unwrap(),
      "{x -> 1.4142135623730951, y -> 1.585786437626905}"
    );
  }
}

mod laplace_transform {
  use super::*;

  #[test]
  fn constant() {
    assert_eq!(interpret("LaplaceTransform[1, t, s]").unwrap(), "s^(-1)");
  }

  // L[UnitStep[c t + b]] with a positive numeric slope c: the step sits at
  // t0 = -b/c. t0 <= 0 gives 1/s; t0 > 0 gives Exp[-t0 s]/s. HeavisideTheta
  // behaves the same. Verified against wolframscript.
  #[test]
  fn unit_step() {
    assert_eq!(
      interpret("LaplaceTransform[UnitStep[t], t, s]").unwrap(),
      "s^(-1)"
    );
    assert_eq!(
      interpret("LaplaceTransform[UnitStep[t - 1], t, s]").unwrap(),
      "1/(E^s*s)"
    );
    assert_eq!(
      interpret("LaplaceTransform[UnitStep[t - 2], t, s]").unwrap(),
      "1/(E^(2*s)*s)"
    );
    // A slope other than 1 shifts the delay to t0 = 1/2.
    assert_eq!(
      interpret("LaplaceTransform[UnitStep[2 t - 1], t, s]").unwrap(),
      "1/(E^(s/2)*s)"
    );
    // A step located at t0 <= 0 is 1 over all of [0, inf).
    assert_eq!(
      interpret("LaplaceTransform[UnitStep[t + 1], t, s]").unwrap(),
      "s^(-1)"
    );
    // HeavisideTheta transforms identically.
    assert_eq!(
      interpret("LaplaceTransform[HeavisideTheta[t - 1], t, s]").unwrap(),
      "1/(E^s*s)"
    );
    // Linearity: a constant multiple and an added term are handled.
    assert_eq!(
      interpret("LaplaceTransform[3 UnitStep[t - 1], t, s]").unwrap(),
      "3/(E^s*s)"
    );
    assert_eq!(
      interpret("LaplaceTransform[UnitStep[t - 1] + t, t, s]").unwrap(),
      "s^(-2) + 1/(E^s*s)"
    );
  }

  #[test]
  fn variable_t() {
    assert_eq!(interpret("LaplaceTransform[t, t, s]").unwrap(), "s^(-2)");
  }

  #[test]
  fn t_squared() {
    assert_eq!(interpret("LaplaceTransform[t^2, t, s]").unwrap(), "2/s^3");
  }

  #[test]
  fn t_cubed() {
    assert_eq!(interpret("LaplaceTransform[t^3, t, s]").unwrap(), "6/s^4");
  }

  #[test]
  fn sin_t() {
    assert_eq!(
      interpret("LaplaceTransform[Sin[t], t, s]").unwrap(),
      "(1 + s^2)^(-1)"
    );
  }

  #[test]
  fn cos_t() {
    assert_eq!(
      interpret("LaplaceTransform[Cos[t], t, s]").unwrap(),
      "s/(1 + s^2)"
    );
  }

  #[test]
  fn sin_3t() {
    assert_eq!(
      interpret("LaplaceTransform[Sin[3*t], t, s]").unwrap(),
      "3/(9 + s^2)"
    );
  }

  #[test]
  fn exp_neg_at() {
    assert_eq!(
      interpret("LaplaceTransform[Exp[-a*t], t, s]").unwrap(),
      "(a + s)^(-1)"
    );
  }

  #[test]
  fn exp_at() {
    assert_eq!(
      interpret("LaplaceTransform[Exp[a*t], t, s]").unwrap(),
      "(-a + s)^(-1)"
    );
  }

  #[test]
  fn linearity_sum() {
    assert_eq!(
      interpret("LaplaceTransform[3*t^2 + 2*Sin[t], t, s]").unwrap(),
      "6/s^3 + 2/(1 + s^2)"
    );
  }

  // s-shifting theorem: L[E^(c t) g(t), t, s] = (L[g])(s - c).
  #[test]
  fn s_shift_exponential_times_trig() {
    assert_eq!(
      interpret("LaplaceTransform[E^(-a t) Cos[b t], t, s]").unwrap(),
      "(a + s)/(b^2 + (a + s)^2)"
    );
    assert_eq!(
      interpret("LaplaceTransform[E^(-a t) Sin[b t], t, s]").unwrap(),
      "b/(b^2 + (a + s)^2)"
    );
    // Exponential times a power of t.
    assert_eq!(
      interpret("LaplaceTransform[E^(2 t) t, t, s]").unwrap(),
      "(-2 + s)^(-2)"
    );
    assert_eq!(
      interpret("LaplaceTransform[E^(-t) t^2, t, s]").unwrap(),
      "2/(1 + s)^3"
    );
    // The shift threads through linearity and a leading constant.
    assert_eq!(
      interpret("LaplaceTransform[E^(-t) (t + Sin[t]), t, s]").unwrap(),
      "(1 + s)^(-2) + (1 + (1 + s)^2)^(-1)"
    );
    assert_eq!(
      interpret("LaplaceTransform[3 E^(-t) Cos[t], t, s]").unwrap(),
      "(3*(1 + s))/(1 + (1 + s)^2)"
    );
  }

  #[test]
  fn constant_multiple() {
    assert_eq!(interpret("LaplaceTransform[5*t, t, s]").unwrap(), "5/s^2");
  }

  // Hyperbolic functions: L[Cosh[a t]] = s/(s^2 - a^2), L[Sinh[a t]] = a/(s^2 - a^2).
  #[test]
  fn cosh_t() {
    assert_eq!(
      interpret("LaplaceTransform[Cosh[t], t, s]").unwrap(),
      "s/(-1 + s^2)"
    );
  }

  #[test]
  fn sinh_t() {
    assert_eq!(
      interpret("LaplaceTransform[Sinh[t], t, s]").unwrap(),
      "(-1 + s^2)^(-1)"
    );
  }

  #[test]
  fn cosh_at() {
    assert_eq!(
      interpret("LaplaceTransform[Cosh[a*t], t, s]").unwrap(),
      "s/(-a^2 + s^2)"
    );
  }

  #[test]
  fn sinh_at() {
    assert_eq!(
      interpret("LaplaceTransform[Sinh[a*t], t, s]").unwrap(),
      "a/(-a^2 + s^2)"
    );
  }

  #[test]
  fn cosh_3t() {
    assert_eq!(
      interpret("LaplaceTransform[Cosh[3*t], t, s]").unwrap(),
      "s/(-9 + s^2)"
    );
  }

  // L[DiracDelta[t]] = 1.
  #[test]
  fn dirac_delta() {
    assert_eq!(
      interpret("LaplaceTransform[DiracDelta[t], t, s]").unwrap(),
      "1"
    );
  }

  #[test]
  fn bessel_j0() {
    assert_eq!(
      interpret("LaplaceTransform[BesselJ[0, t], t, s]").unwrap(),
      "1/Sqrt[1 + s^2]"
    );
  }

  #[test]
  fn bessel_j1() {
    assert_eq!(
      interpret("LaplaceTransform[BesselJ[1, t], t, s]").unwrap(),
      "1/(Sqrt[1 + s^2]*(s + Sqrt[1 + s^2]))"
    );
  }
}

mod grad {
  use super::*;

  #[test]
  fn basic_2d() {
    assert_eq!(
      interpret("Grad[x^2 + y^3, {x, y}]").unwrap(),
      "{2*x, 3*y^2}"
    );
  }

  #[test]
  fn basic_3d() {
    assert_eq!(
      interpret("Grad[x^2*y + y^2*z, {x, y, z}]").unwrap(),
      "{2*x*y, x^2 + 2*y*z, y^2}"
    );
  }

  #[test]
  fn trig() {
    assert_eq!(
      interpret("Grad[Sin[x]*Cos[y], {x, y}]").unwrap(),
      "{Cos[x]*Cos[y], -(Sin[x]*Sin[y])}"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("Grad[5, {x, y}]").unwrap(), "{0, 0}");
  }

  #[test]
  fn single_variable() {
    assert_eq!(interpret("Grad[x^3, {x}]").unwrap(), "{3*x^2}");
  }

  // Grad[f, vars, "Coordinates"] applies orthogonal-curvilinear scale factors:
  // component i = (1/h_i) ∂f/∂x_i.
  #[test]
  fn polar() {
    assert_eq!(
      interpret(r#"Grad[r^2, {r, t}, "Polar"]"#).unwrap(),
      "{2*r, 0}"
    );
    // The 1/r factor on the angular component shows up here.
    assert_eq!(
      interpret(r#"Grad[r Cos[t], {r, t}, "Polar"]"#).unwrap(),
      "{Cos[t], -Sin[t]}"
    );
  }

  #[test]
  fn cylindrical() {
    assert_eq!(
      interpret(r#"Grad[r^2 z, {r, t, z}, "Cylindrical"]"#).unwrap(),
      "{2*r*z, 0, r^2}"
    );
  }

  #[test]
  fn spherical() {
    // r and theta components; the phi component is 0 here (f independent of p).
    assert_eq!(
      interpret(r#"Grad[r^2, {r, t, p}, "Spherical"]"#).unwrap(),
      "{2*r, 0, 0}"
    );
  }

  #[test]
  fn cartesian_named() {
    assert_eq!(
      interpret(r#"Grad[x^2, {x, y}, "Cartesian"]"#).unwrap(),
      "{2*x, 0}"
    );
  }

  // An unknown coordinate system stays unevaluated.
  #[test]
  fn unknown_coordinates_unevaluated() {
    assert_eq!(
      interpret(r#"Grad[r^2, {r, t}, "Bogus"]"#).unwrap(),
      "Grad[r^2, {r, t}, Bogus]"
    );
  }

  // For a vector field the derivative is the LAST index, so the result is the
  // Jacobian with row i = gradient of f_i: result[[i, j]] = D[f_i, x_j].
  #[test]
  fn vector_field_jacobian() {
    assert_eq!(
      interpret("Grad[{x y, x + y}, {x, y}]").unwrap(),
      "{{y, x}, {1, 1}}"
    );
    assert_eq!(
      interpret("Grad[{x^2 y, y^3}, {x, y}]").unwrap(),
      "{{2*x*y, x^2}, {0, 3*y^2}}"
    );
  }

  // A non-square Jacobian: a 2-component field over 3 variables is 2x3.
  #[test]
  fn vector_field_nonsquare() {
    assert_eq!(
      interpret("Grad[{x y z, x + y}, {x, y, z}]").unwrap(),
      "{{y*z, x*z, x*y}, {1, 1, 0}}"
    );
  }

  // A rank-2 field gains the derivative axis as the innermost dimension.
  #[test]
  fn rank_two_field() {
    assert_eq!(
      interpret("Grad[{{x y, x}, {y, x^2}}, {x, y}]").unwrap(),
      "{{{y, x}, {1, 0}}, {{0, 1}, {2*x, 0}}}"
    );
  }
}

mod recurrence_table {
  use super::*;

  #[test]
  fn geometric() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{a[n+1] == 2*a[n], a[1] == 1}, a, {n, 1, 10}]"
      )
      .unwrap(),
      "{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}"
    );
  }

  #[test]
  fn fibonacci() {
    assert_eq!(
      interpret("RecurrenceTable[{a[n+1] == a[n] + a[n-1], a[1] == 1, a[2] == 1}, a, {n, 1, 8}]")
        .unwrap(),
      "{1, 1, 2, 3, 5, 8, 13, 21}"
    );
  }

  #[test]
  fn affine() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{a[n] == 3*a[n-1] + 1, a[0] == 0}, a, {n, 0, 5}]"
      )
      .unwrap(),
      "{0, 1, 4, 13, 40, 121}"
    );
  }

  /// A coupled system: each function has its own recurrence, and the table
  /// lists one tuple per step. A Demonstration iterates a two-variable
  /// orbit-fractal this way; the whole form used to come back unevaluated.
  #[test]
  fn coupled_system() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{a[i + 1] == a[i] + b[i], b[i + 1] == a[i], \
         a[1] == 1, b[1] == 1}, {a, b}, {i, 1, 8}]"
      )
      .unwrap(),
      "{{1, 1}, {2, 1}, {3, 2}, {5, 3}, {8, 5}, {13, 8}, {21, 13}, {34, 21}}"
    );
    // Both sides of `b[i + 1] == a[i]` look like a step of some function,
    // so the equation belongs to the side that steps furthest forward.
    assert_eq!(
      interpret(
        "RecurrenceTable[{u[k + 1] == 2 u[k] - v[k], v[k + 1] == u[k], \
         u[1] == 3, v[1] == 1}, {u, v}, {k, 1, 5}]"
      )
      .unwrap(),
      "{{3, 1}, {5, 3}, {7, 5}, {9, 7}, {11, 9}}"
    );
    // A single function named in a list still tabulates as tuples.
    assert_eq!(
      interpret(
        "RecurrenceTable[{p[i + 1] == p[i]/2, p[1] == 1.}, {p}, {i, 1, 4}]"
      )
      .unwrap(),
      "{{1.}, {0.5}, {0.25}, {0.125}}"
    );
  }

  /// A window that starts after the initial conditions is reached by
  /// stepping through the values in between: they carry the recurrence.
  #[test]
  fn window_after_the_initial_conditions() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{q[i + 1] == q[i] + 1, q[1] == 0}, q, {i, 3, 6}]"
      )
      .unwrap(),
      "{2, 3, 4, 5}"
    );
    assert_eq!(
      interpret(
        "RecurrenceTable[{a[i + 1] == a[i] + b[i], b[i + 1] == a[i], \
         a[1] == 1, b[1] == 1}, {a, b}, {i, 3, 6}]"
      )
      .unwrap(),
      "{{3, 2}, {5, 3}, {8, 5}, {13, 8}}"
    );
  }

  #[test]
  fn unevaluated_bad_args() {
    assert_eq!(
      interpret("RecurrenceTable[x, y]").unwrap(),
      "RecurrenceTable[x, y]"
    );
  }

  // The two-element range spec {n, nmax} defaults nmin to 1.
  #[test]
  fn two_element_range_fibonacci() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{a[n] == a[n-1] + a[n-2], a[1] == 1, a[2] == 1}, a, {n, 10}]"
      )
      .unwrap(),
      "{1, 1, 2, 3, 5, 8, 13, 21, 34, 55}"
    );
  }

  #[test]
  fn two_element_range_geometric() {
    assert_eq!(
      interpret("RecurrenceTable[{a[n] == 2*a[n-1], a[1] == 1}, a, {n, 6}]")
        .unwrap(),
      "{1, 2, 4, 8, 16, 32}"
    );
  }
}

mod inverse_laplace_transform {
  use super::*;

  #[test]
  fn constant_one_over_s() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/s, s, t]").unwrap(),
      "1"
    );
  }

  #[test]
  fn one_over_s_squared() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/s^2, s, t]").unwrap(),
      "t"
    );
  }

  #[test]
  fn two_over_s_cubed() {
    assert_eq!(
      interpret("InverseLaplaceTransform[2/s^3, s, t]").unwrap(),
      "t^2"
    );
  }

  #[test]
  fn six_over_s_fourth() {
    assert_eq!(
      interpret("InverseLaplaceTransform[6/s^4, s, t]").unwrap(),
      "t^3"
    );
  }

  #[test]
  fn sin_t() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^2 + 1), s, t]").unwrap(),
      "Sin[t]"
    );
  }

  #[test]
  fn cos_t() {
    assert_eq!(
      interpret("InverseLaplaceTransform[s/(s^2 + 1), s, t]").unwrap(),
      "Cos[t]"
    );
  }

  #[test]
  fn sin_at() {
    assert_eq!(
      interpret("InverseLaplaceTransform[a/(s^2 + a^2), s, t]").unwrap(),
      "Sin[a*t]"
    );
  }

  #[test]
  fn cos_at() {
    assert_eq!(
      interpret("InverseLaplaceTransform[s/(s^2 + a^2), s, t]").unwrap(),
      "Cos[a*t]"
    );
  }

  #[test]
  fn exp_at() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s - a), s, t]").unwrap(),
      "E^(a*t)"
    );
  }

  // Negative constant term gives the hyperbolic forms:
  //   1/(s^2 - a^2) -> Sinh[a t]/a,  s/(s^2 - a^2) -> Cosh[a t].
  // (Previously 1/(s^2 - 1) produced a malformed `--1*Sinh[t]`.)
  #[test]
  fn sinh_t() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^2 - 1), s, t]").unwrap(),
      "Sinh[t]"
    );
  }

  #[test]
  fn cosh_t() {
    assert_eq!(
      interpret("InverseLaplaceTransform[s/(s^2 - 1), s, t]").unwrap(),
      "Cosh[t]"
    );
  }

  #[test]
  fn sinh_3t() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^2 - 9), s, t]").unwrap(),
      "Sinh[3*t]/3"
    );
  }

  #[test]
  fn cosh_3t() {
    assert_eq!(
      interpret("InverseLaplaceTransform[s/(s^2 - 9), s, t]").unwrap(),
      "Cosh[3*t]"
    );
  }

  #[test]
  fn sinh_at_symbolic() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^2 - a^2), s, t]").unwrap(),
      "Sinh[a*t]/a"
    );
  }

  #[test]
  fn cosh_at_symbolic() {
    assert_eq!(
      interpret("InverseLaplaceTransform[s/(s^2 - a^2), s, t]").unwrap(),
      "Cosh[a*t]"
    );
  }

  #[test]
  fn sinh_irrational() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^2 - 2), s, t]").unwrap(),
      "Sinh[Sqrt[2]*t]/Sqrt[2]"
    );
  }

  #[test]
  fn exp_neg_at() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s + a), s, t]").unwrap(),
      "E^(-(a*t))"
    );
  }

  #[test]
  fn unevaluated_unknown() {
    assert_eq!(
      interpret("InverseLaplaceTransform[Log[s], s, t]").unwrap(),
      "InverseLaplaceTransform[Log[s], s, t]"
    );
  }

  // L^-1[c] = c DiracDelta[t] for an s-independent constant.
  #[test]
  fn constant_one_dirac() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1, s, t]").unwrap(),
      "DiracDelta[t]"
    );
  }

  #[test]
  fn constant_scaled_dirac() {
    assert_eq!(
      interpret("InverseLaplaceTransform[5, s, t]").unwrap(),
      "5*DiracDelta[t]"
    );
  }

  #[test]
  fn constant_symbol_dirac() {
    assert_eq!(
      interpret("InverseLaplaceTransform[a, s, t]").unwrap(),
      "a*DiracDelta[t]"
    );
  }

  // L^-1[E^(-c s)] = DiracDelta[t - c] (time shift).
  #[test]
  fn exp_shift_dirac() {
    assert_eq!(
      interpret("InverseLaplaceTransform[E^(-s), s, t]").unwrap(),
      "DiracDelta[-1 + t]"
    );
  }

  #[test]
  fn exp_shift_two_dirac() {
    assert_eq!(
      interpret("InverseLaplaceTransform[E^(-2 s), s, t]").unwrap(),
      "DiracDelta[-2 + t]"
    );
  }

  #[test]
  fn exp_shift_symbolic_dirac() {
    assert_eq!(
      interpret("InverseLaplaceTransform[E^(-a s), s, t]").unwrap(),
      "DiracDelta[-a + t]"
    );
  }

  #[test]
  fn exp_shift_scaled_dirac() {
    assert_eq!(
      interpret("InverseLaplaceTransform[3 E^(-2 s), s, t]").unwrap(),
      "3*DiracDelta[-2 + t]"
    );
  }

  // ─── Rational transfer functions ─────────────────────────────────────
  // A proper rational function with exact coefficients is inverted
  // exactly, off its partial-fraction decomposition; float coefficients
  // (a Manipulate slider's value substituted into a control-system
  // transfer function) keep the numeric residue sum.

  // Three distinct real poles: the textbook partial-fraction expansion of
  // 1/((s+1)(s+2)(s+3)) is (1/2)E^-t - E^-2t + (1/2)E^-3t, which regroups
  // into a factored polynomial in E^t — the form wolframscript prints.
  #[test]
  fn partial_fractions_three_real_poles() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/((s + 1)(s + 2)(s + 3)), s, t]")
        .unwrap(),
      "(-1 + E^t)^2/(2*E^(3*t))"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[1/((s + 1)(s + 2)), s, t]").unwrap(),
      "(-1 + E^t)/E^(2*t)"
    );
    assert_eq!(
      interpret(
        "InverseLaplaceTransform[1/((s + 1)(s + 2)(s + 3)(s + 4)), s, t]"
      )
      .unwrap(),
      "(-1 + E^t)^3/(6*E^(4*t))"
    );
    // A numerator of its own (a zero at s = -3/2, not just a gain).
    assert_eq!(
      interpret("InverseLaplaceTransform[(2 s + 3)/((s + 1)(s + 4)), s, t]")
        .unwrap(),
      "(5 + E^(3*t))/(3*E^(4*t))"
    );
  }

  // Float coefficients stay on the numeric residue path.
  #[test]
  fn partial_fractions_machine_number_poles() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/((s + 1.5)(s + 2.5)), s, t]")
        .unwrap(),
      "-1./E^(2.5*t) + 1./E^(1.5*t)"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[s/(s + 1.5), s, t]").unwrap(),
      "-1.5/E^(1.5*t) + 1.*DiracDelta[t]"
    );
  }

  // An improper fraction keeps a polynomial part, and `L^-1[s^k]` is the
  // k-th derivative of DiracDelta. (The `Plus` order of a `Derivative`
  // term differs from wolframscript's — see conformance_gaps.md.)
  #[test]
  fn improper_fraction_gives_dirac_delta_derivatives() {
    assert_eq!(
      interpret("InverseLaplaceTransform[s, s, t]").unwrap(),
      "Derivative[1][DiracDelta][t]"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[s^2, s, t]").unwrap(),
      "Derivative[2][DiracDelta][t]"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[s/(s + 1), s, t]").unwrap(),
      "-E^(-t) + DiracDelta[t]"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[(s + 1)/(s + 1), s, t]").unwrap(),
      "DiracDelta[t]"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[s^2/(s + 1), s, t]").unwrap(),
      "Derivative[1][DiracDelta][t] + E^(-t) - DiracDelta[t]"
    );
  }

  // A real pole at s = 0 plus a complex-conjugate pair (-1 ± 2 I) from
  // s^2 + 2 s + 5: the real pole contributes a constant, the pair an
  // exponentially damped oscillation. (wolframscript prints the same
  // function as a sum of complex exponentials — see
  // tests/cli/comparison/mathematica/conformance_gaps.md.)
  #[test]
  fn partial_fractions_real_and_complex_poles() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^3 + 2 s^2 + 5 s), s, t]")
        .unwrap(),
      "(2*E^t - 2*Cos[2*t] - Sin[2*t])/(10*E^t)"
    );
  }

  // An irreducible quadratic with irrational real roots inverts to the
  // hyperbolic counterpart of the oscillating case.
  #[test]
  fn partial_fractions_irrational_real_poles() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^2 - 2), s, t]").unwrap(),
      "Sinh[Sqrt[2]*t]/Sqrt[2]"
    );
  }

  // A repeated pole contributes the `t^(k-1) E^(r t)/(k-1)!` terms a
  // simple-pole residue sum doesn't produce.
  #[test]
  fn partial_fractions_repeated_pole() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s + 1)^3, s, t]").unwrap(),
      "t^2/(2*E^t)"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[1/((s + 1)^2 (s + 2)), s, t]")
        .unwrap(),
      "(1 + E^t*(-1 + t))/E^(2*t)"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^2 (s + 1)), s, t]").unwrap(),
      "-1 + E^(-t) + t"
    );
  }

  // At t = 0 the three-real-pole case above must reproduce the exact
  // partial-fraction coefficients (1/2 - 1 + 1/2 = 0), a quick numeric
  // sanity check independent of the exact printed form.
  #[test]
  fn partial_fractions_value_at_zero() {
    assert_eq!(
      interpret(
        "Chop[InverseLaplaceTransform[1/((s + 1)(s + 2)(s + 3)), s, t] \
         /. t -> 0]"
      )
      .unwrap(),
      "0"
    );
  }

  // ─── Two-variable InverseLaplaceTransform ───────────────────────────
  // wolframscript:
  //   InverseLaplaceTransform[1/(p*q), {p, q}, {x, y}]      -> 1
  //   InverseLaplaceTransform[1/(p + q), {p, q}, {x, y}]    -> DiracDelta[-x + y]
  //   InverseLaplaceTransform[1/(1 + p*q), {p, q}, {x, y}]  -> BesselJ[0, 2*Sqrt[x]*Sqrt[y]]
  //   InverseLaplaceTransform[1/Sqrt[1 + p*q], {p, q}, {x, y}]
  //     -> Cosh[2*Sqrt[-(x*y)]]/(Pi*Sqrt[x]*Sqrt[y])
  //
  // These follow Efros's theorem: L^-1_{p,q}[F(p*q)] involves f(τ) (the
  // 1-var inverse transform of F) convolved with J_0(2*Sqrt[x*y*τ])
  // over τ. We pattern-match the specific F(p*q) shapes Wolfram lists
  // as canonical examples.

  #[test]
  fn two_var_product_reciprocal() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(p*q), {p, q}, {x, y}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn two_var_sum_reciprocal() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(p + q), {p, q}, {x, y}]").unwrap(),
      "DiracDelta[-x + y]"
    );
  }

  #[test]
  fn two_var_one_plus_pq_reciprocal() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(1 + p*q), {p, q}, {x, y}]")
        .unwrap(),
      "BesselJ[0, 2*Sqrt[x]*Sqrt[y]]"
    );
  }

  #[test]
  fn two_var_sqrt_one_plus_pq_reciprocal() {
    // Audit case: previously returned unevaluated.
    assert_eq!(
      interpret("InverseLaplaceTransform[1/Sqrt[1 + p*q], {p, q}, {x, y}]")
        .unwrap(),
      "Cosh[2*Sqrt[-(x*y)]]/(Pi*Sqrt[x]*Sqrt[y])"
    );
  }

  // Unknown 2-var input stays unevaluated.
  #[test]
  fn two_var_unevaluated_unknown() {
    assert_eq!(
      interpret("InverseLaplaceTransform[Log[1 + p*q], {p, q}, {x, y}]")
        .unwrap(),
      "InverseLaplaceTransform[Log[1 + p*q], {p, q}, {x, y}]"
    );
  }

  // L^-1[1/Sqrt[s^2 + a^2]] = BesselJ[0, a*t] and its hyperbolic counterpart
  // L^-1[1/Sqrt[s^2 - a^2]] = BesselI[0, a*t].
  #[test]
  fn bessel_j_at_symbolic() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/Sqrt[s^2 + a^2], s, t]").unwrap(),
      "BesselJ[0, a*t]"
    );
  }

  #[test]
  fn bessel_j_at_numeric_constant() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/Sqrt[s^2 + 4], s, t]").unwrap(),
      "BesselJ[0, 2*t]"
    );
  }

  #[test]
  fn bessel_j_irrational() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/Sqrt[s^2 + 2], s, t]").unwrap(),
      "BesselJ[0, Sqrt[2]*t]"
    );
  }

  #[test]
  fn bessel_i_at_symbolic() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/Sqrt[s^2 - a^2], s, t]").unwrap(),
      "BesselI[0, a*t]"
    );
  }

  // A third argument that is not a plain symbol names the point the inverse
  // transform is taken at, so the result comes back evaluated there.
  #[test]
  fn third_argument_number() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s + 1), s, 2.]").unwrap(),
      interpret("E^-2.").unwrap()
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[1/Sqrt[s^2 + 1], s, 1.5]").unwrap(),
      interpret("BesselJ[0, 1.5]").unwrap()
    );
  }

  #[test]
  fn third_argument_expression() {
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s + 1), s, 2 u]").unwrap(),
      "E^(-2*u)"
    );
    assert_eq!(
      interpret("InverseLaplaceTransform[1/(s^2 + 1), s, 2 u]").unwrap(),
      "Sin[2*u]"
    );
  }

  // An unrecognised transform stays unevaluated with the original third
  // argument rather than leaking the internal placeholder variable.
  #[test]
  fn third_argument_expression_unevaluated() {
    assert_eq!(
      interpret("InverseLaplaceTransform[Sqrt[s], s, 1.]").unwrap(),
      "InverseLaplaceTransform[Sqrt[s], s, 1.]"
    );
  }
}

mod laplacian {
  use super::*;

  #[test]
  fn laplacian_2d() {
    assert_eq!(interpret("Laplacian[x^2 + y^2, {x, y}]").unwrap(), "4");
  }

  #[test]
  fn laplacian_3d() {
    assert_eq!(
      interpret("Laplacian[x^2*y + z^3, {x, y, z}]").unwrap(),
      "2*y + 6*z"
    );
  }

  #[test]
  fn laplacian_harmonic() {
    // x^2 - y^2 is harmonic (Laplacian = 0)
    assert_eq!(interpret("Laplacian[x^2 - y^2, {x, y}]").unwrap(), "0");
  }

  #[test]
  fn laplacian_single_var() {
    assert_eq!(interpret("Laplacian[x^3, {x}]").unwrap(), "6*x");
  }

  // Laplacian[f, vars, "Coordinates"] uses orthogonal-curvilinear scale
  // factors: Lap f = (1/J) Σ ∂/∂x_i((J/h_i²) ∂f/∂x_i).
  #[test]
  fn laplacian_polar() {
    assert_eq!(
      interpret(r#"Laplacian[r^2, {r, t}, "Polar"]"#).unwrap(),
      "4"
    );
    // Log[r] is harmonic in 2D.
    assert_eq!(
      interpret(r#"Laplacian[Log[r], {r, t}, "Polar"]"#).unwrap(),
      "0"
    );
  }

  #[test]
  fn laplacian_cylindrical() {
    assert_eq!(
      interpret(r#"Laplacian[r^2, {r, t, z}, "Cylindrical"]"#).unwrap(),
      "4"
    );
  }

  #[test]
  fn laplacian_spherical() {
    // 1/r is harmonic in 3D.
    assert_eq!(
      interpret(r#"Laplacian[1/r, {r, t, p}, "Spherical"]"#).unwrap(),
      "0"
    );
    assert_eq!(
      interpret(r#"Laplacian[r^3, {r, t, p}, "Spherical"]"#).unwrap(),
      "12*r"
    );
  }

  #[test]
  fn laplacian_unknown_coordinates_unevaluated() {
    assert_eq!(
      interpret(r#"Laplacian[r^2, {r, t}, "Bogus"]"#).unwrap(),
      "Laplacian[r^2, {r, t}, Bogus]"
    );
  }
}

mod div {
  use super::*;

  #[test]
  fn div_3d() {
    assert_eq!(
      interpret("Div[{x^2, y^2, z^2}, {x, y, z}]").unwrap(),
      "2*x + 2*y + 2*z"
    );
  }

  #[test]
  fn div_2d() {
    assert_eq!(interpret("Div[{x*y, x + y}, {x, y}]").unwrap(), "1 + y");
  }

  #[test]
  fn div_constant_field() {
    assert_eq!(interpret("Div[{1, 2, 3}, {x, y, z}]").unwrap(), "0");
  }

  // Div[F, vars, "Coordinates"] uses orthogonal-curvilinear scale factors:
  // Div F = (1/J) Σ ∂/∂x_i((J/h_i) F_i).
  #[test]
  fn div_polar() {
    assert_eq!(interpret(r#"Div[{r, 0}, {r, t}, "Polar"]"#).unwrap(), "2");
    assert_eq!(interpret(r#"Div[{1/r, 0}, {r, t}, "Polar"]"#).unwrap(), "0");
  }

  #[test]
  fn div_cylindrical() {
    assert_eq!(
      interpret(r#"Div[{r, 0, z}, {r, t, z}, "Cylindrical"]"#).unwrap(),
      "3"
    );
  }

  #[test]
  fn div_spherical() {
    assert_eq!(
      interpret(r#"Div[{r, 0, 0}, {r, t, p}, "Spherical"]"#).unwrap(),
      "3"
    );
  }

  #[test]
  fn div_unknown_coordinates_unevaluated() {
    assert_eq!(
      interpret(r#"Div[{r, 0}, {r, t}, "Bogus"]"#).unwrap(),
      "Div[{r, 0}, {r, t}, Bogus]"
    );
  }

  // Div contracts the LAST index, so for a rank-2 tensor T the result is the
  // vector result[[i]] = Sum_j D[T[[i, j]], x_j] (divergence of each row).
  #[test]
  fn rank_two_tensor() {
    assert_eq!(
      interpret("Div[{{x, y}, {z, w}}, {x, y}]").unwrap(),
      "{2, 0}"
    );
  }

  // A non-square tensor: a 3x2 field over 2 variables yields a length-3 vector.
  #[test]
  fn nonsquare_tensor() {
    assert_eq!(
      interpret("Div[{{x, y}, {z, w}, {a, b}}, {x, y}]").unwrap(),
      "{2, 0, 0}"
    );
  }

  // Rank-3 tensor: the divergence is taken over the innermost index.
  #[test]
  fn rank_three_tensor() {
    assert_eq!(
      interpret("Div[{{{x, y}, {z, x}}, {{y, z}, {x, y}}}, {x, y}]").unwrap(),
      "{{2, 0}, {0, 2}}"
    );
  }
}

mod curl {
  use super::*;

  // The well-defined low-rank forms: scalar in 2D, vector in 2D (scalar
  // result), and vector in 3D (vector result).
  #[test]
  fn scalar_2d() {
    assert_eq!(interpret("Curl[x^2, {x, y}]").unwrap(), "{0, 2*x}");
  }

  #[test]
  fn vector_2d() {
    assert_eq!(interpret("Curl[{-y, x}, {x, y}]").unwrap(), "2");
    assert_eq!(interpret("Curl[{x, y}, {x, y}]").unwrap(), "0");
  }

  #[test]
  fn vector_3d() {
    assert_eq!(
      interpret("Curl[{x^2 y, y^2 z, z^2 x}, {x, y, z}]").unwrap(),
      "{-y^2, -z^2, -x^2}"
    );
  }

  // A rank-2 tensor in dimension 2 has no curl (rank >= dimension): Curl stays
  // unevaluated (wolframscript emits Curl::hrank). Previously Woxi treated the
  // rows as scalars and returned a bogus {0, -1}.
  #[test]
  fn rank_too_high_unevaluated() {
    assert_eq!(
      interpret("Curl[{{x, y}, {z, w}}, {x, y}]").unwrap(),
      "Curl[{{x, y}, {z, w}}, {x, y}]"
    );
  }

  // A vector whose length differs from the variable count has no curl
  // (wolframscript emits Curl::ndimv).
  #[test]
  fn vector_dimension_mismatch_unevaluated() {
    assert_eq!(
      interpret("Curl[{x, y, z}, {x, y}]").unwrap(),
      "Curl[{x, y, z}, {x, y}]"
    );
    assert_eq!(
      interpret("Curl[{a, b}, {x, y, z}]").unwrap(),
      "Curl[{a, b}, {x, y, z}]"
    );
  }

  // A tensor whose dimensions do not match the space dimension has no curl
  // (wolframscript emits Curl::ndimt).
  #[test]
  fn tensor_dimension_mismatch_unevaluated() {
    assert_eq!(
      interpret("Curl[{{x, y}, {z, w}}, {x, y, z}]").unwrap(),
      "Curl[{{x, y}, {z, w}}, {x, y, z}]"
    );
  }
}

mod dsolve_value {
  use super::*;

  #[test]
  fn simple_ode() {
    assert_eq!(
      interpret("DSolveValue[y'[x] == y[x], y[x], x]").unwrap(),
      "E^x*C[1]"
    );
  }

  #[test]
  fn second_order() {
    assert_eq!(
      interpret("DSolveValue[y''[x] + y[x] == 0, y[x], x]").unwrap(),
      "C[1]*Cos[x] + C[2]*Sin[x]"
    );
  }
}

mod ndsolve_value {
  use super::*;

  #[test]
  fn returns_interpolating_function() {
    let result =
      interpret("NDSolveValue[{y'[x] == -y[x], y[0] == 1}, y, {x, 0, 10}]")
        .unwrap();
    assert!(
      result.contains("InterpolatingFunction"),
      "Expected InterpolatingFunction, got: {result}"
    );
  }

  #[test]
  fn can_evaluate() {
    let result = interpret(
      "f = NDSolveValue[{y'[x] == -y[x], y[0] == 1}, y, {x, 0, 10}]; f[0]",
    )
    .unwrap();
    assert_eq!(result, "1.");
  }

  /// An equation the numeric solver can't handle (here one with no initial
  /// condition to start from) stays unevaluated under its own head — it used
  /// to come back as the `NDSolve` that `NDSolveValue` delegates to.
  #[test]
  fn unsolvable_keeps_its_own_head() {
    assert_eq!(
      interpret("NDSolveValue[{y'[x] == Sin[y[x]^2 + x]}, y, {x, 0, 1}]")
        .unwrap(),
      "NDSolveValue[{Derivative[1][y][x] == Sin[x + y[x]^2]}, y, {x, 0, 1}]"
    );
  }
}

mod dsolve_value_unevaluated {
  use super::*;

  /// Same for the symbolic solver: `DSolveValue` must not report itself as
  /// `DSolve` when it leaves an equation unsolved.
  #[test]
  fn unsolvable_keeps_its_own_head() {
    assert_eq!(
      interpret("DSolveValue[y'[x] == Sin[y[x]^2 + x], y, x]").unwrap(),
      "DSolveValue[Derivative[1][y][x] == Sin[x + y[x]^2], y, x]"
    );
  }
}

/// A homogeneous second-order PDE with constant coefficients has one
/// arbitrary function per characteristic slope — the roots of
/// `a λ² + b λ + c`, in the order their reciprocals take canonically.
mod second_order_constant_pde {
  use super::*;

  #[test]
  fn laplace_equation() {
    assert_eq!(
      interpret(
        "DSolveValue[D[u[x, y], x, x] + D[u[x, y], y, y] == 0, u, {x, y}]"
      )
      .unwrap(),
      "Function[{x, y}, C[1][I*x + y] + C[2][-I*x + y]]"
    );
    // The rule forms report the same body.
    assert_eq!(
      interpret(
        "DSolve[D[u[x, y], x, x] + D[u[x, y], y, y] == 0, u[x, y], {x, y}]"
      )
      .unwrap(),
      "{{u[x, y] -> C[1][I*x + y] + C[2][-I*x + y]}}"
    );
  }

  #[test]
  fn real_and_rational_slopes() {
    assert_eq!(
      interpret(
        "DSolveValue[D[u[x, y], x, x] - D[u[x, y], y, y] == 0, u, {x, y}]"
      )
      .unwrap(),
      "Function[{x, y}, C[1][-x + y] + C[2][x + y]]"
    );
    assert_eq!(
      interpret(
        "DSolveValue[D[u[x, y], x, x] - 5*D[D[u[x, y], x], y] + \
         6*D[u[x, y], y, y] == 0, u, {x, y}]"
      )
      .unwrap(),
      "Function[{x, y}, C[1][3*x + y] + C[2][2*x + y]]"
    );
    assert_eq!(
      interpret(
        "DSolveValue[4*D[u[x, y], x, x] - D[u[x, y], y, y] == 0, u, {x, y}]"
      )
      .unwrap(),
      "Function[{x, y}, C[1][-1/2*x + y] + C[2][x/2 + y]]"
    );
  }

  /// A repeated slope: the second solution picks up a factor of `x`.
  #[test]
  fn repeated_root() {
    assert_eq!(
      interpret(
        "DSolveValue[D[u[x, y], x, x] + 4*D[D[u[x, y], x], y] + \
         4*D[u[x, y], y, y] == 0, u, {x, y}]"
      )
      .unwrap(),
      "Function[{x, y}, C[1][-2*x + y] + x*C[2][-2*x + y]]"
    );
  }

  /// The coefficients have to be constants for the characteristic slopes to
  /// be constants; a variable one leaves the equation unsolved, as it does
  /// in wolframscript.
  #[test]
  fn a_variable_coefficient_is_not_handled() {
    assert_eq!(
      interpret(
        "DSolveValue[x*D[u[x, y], x, x] + D[u[x, y], y, y] == 0, u, {x, y}]"
      )
      .unwrap(),
      "DSolveValue[Derivative[0, 2][u][x, y] + x*Derivative[2, 0][u][x, y] \
       == 0, u, {x, y}]"
    );
  }
}

mod wronskian {
  use super::*;

  #[test]
  fn sin_cos() {
    assert_eq!(interpret("Wronskian[{Sin[x], Cos[x]}, x]").unwrap(), "-1");
  }

  #[test]
  fn polynomials() {
    assert_eq!(interpret("Wronskian[{1, x, x^2}, x]").unwrap(), "2");
  }

  #[test]
  fn exponentials() {
    assert_eq!(
      interpret("Wronskian[{E^x, E^(2*x)}, x]").unwrap(),
      "E^(3*x)"
    );
  }
}

// Series[f, {x, Infinity, n}] expands in powers of 1/x. All values verified
// against wolframscript.
mod series_at_infinity {
  use super::*;

  #[test]
  fn rational_functions() {
    assert_eq!(
      interpret("Series[1/(1 + x), {x, Infinity, 3}]").unwrap(),
      "SeriesData[x, Infinity, {1, -1, 1}, 1, 4, 1]"
    );
    assert_eq!(
      interpret("Series[1/x, {x, Infinity, 1}]").unwrap(),
      "SeriesData[x, Infinity, {1}, 1, 2, 1]"
    );
    assert_eq!(
      interpret("Series[x/(x + 1), {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1, -1, 1}, 0, 3, 1]"
    );
    // A nested reciprocal has to be cleared before expanding.
    assert_eq!(
      interpret("Series[(x + 1)/x, {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1, 1}, 0, 3, 1]"
    );
    assert_eq!(
      interpret("Series[1/(x^2 + 1), {x, Infinity, 4}]").unwrap(),
      "SeriesData[x, Infinity, {1, 0, -1}, 2, 5, 1]"
    );
  }

  // A term of positive degree keeps its place, so nmin goes negative.
  #[test]
  fn positive_powers() {
    assert_eq!(
      interpret("Series[x^2, {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1}, -2, 3, 1]"
    );
  }

  // Algebraic (non-rational) expressions previously failed here with an
  // Infinity::indet error; the reciprocal substitution handles them.
  #[test]
  fn algebraic_expressions() {
    assert_eq!(
      interpret("Series[Sqrt[x^2 + 1], {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1, 0, 1/2}, -1, 3, 1]"
    );
    assert_eq!(
      interpret("Series[Sqrt[x^2 - 1], {x, Infinity, 3}]").unwrap(),
      "SeriesData[x, Infinity, {1, 0, -1/2, 0, -1/8}, -1, 4, 1]"
    );
    assert_eq!(
      interpret("Series[(x^3 + 1)^(1/3), {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1, 0, 0, 1/3}, -1, 3, 1]"
    );
    // Half-integer powers give a fractional step, recorded as the denominator.
    assert_eq!(
      interpret("Series[Sqrt[x + 1], {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1, 0, 1/2, 0, -1/8}, -1, 5, 2]"
    );
    assert_eq!(
      interpret("Series[1/Sqrt[x], {x, Infinity, 2}]").unwrap(),
      "SeriesData[x, Infinity, {1}, 1, 5, 2]"
    );
  }

  #[test]
  fn normal_gives_the_polynomial_in_one_over_x() {
    assert_eq!(
      interpret("Normal[Series[1/(1 + x), {x, Infinity, 3}]]").unwrap(),
      "x^(-3) - x^(-2) + x^(-1)"
    );
    assert_eq!(
      interpret("Normal[Series[x/(x + 1), {x, Infinity, 2}]]").unwrap(),
      "1 + x^(-2) - x^(-1)"
    );
    assert_eq!(
      interpret("Normal[Series[(2 x + 1)/(x - 1), {x, Infinity, 2}]]").unwrap(),
      "2 + 3/x^2 + 3/x"
    );
    assert_eq!(
      interpret("Normal[Series[Sqrt[x^2 + 1], {x, Infinity, 2}]]").unwrap(),
      "1/(2*x) + x"
    );
  }

  // The special-cased asymptotic expansions still take precedence.
  #[test]
  fn asymptotic_special_cases_still_apply() {
    assert!(
      interpret("Series[ExpIntegralEi[x], {x, Infinity, 6}]")
        .unwrap()
        .starts_with("E^x*("),
      "the ExpIntegralEi asymptotic expansion should still be used"
    );
  }
}

mod series_coefficient {
  use super::*;

  #[test]
  fn geometric_series() {
    assert_eq!(
      interpret("SeriesCoefficient[1/(1-x), {x, 0, 5}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn exp_coefficient() {
    assert_eq!(
      interpret("SeriesCoefficient[Exp[x], {x, 0, 3}]").unwrap(),
      "1/6"
    );
  }

  #[test]
  fn sin_coefficient() {
    assert_eq!(
      interpret("SeriesCoefficient[Sin[x], {x, 0, 5}]").unwrap(),
      "1/120"
    );
  }

  #[test]
  fn log_coefficient() {
    assert_eq!(
      interpret("SeriesCoefficient[Log[1+x], {x, 0, 4}]").unwrap(),
      "-1/4"
    );
  }

  #[test]
  fn zero_coefficient() {
    // Sin has no even-order terms
    assert_eq!(
      interpret("SeriesCoefficient[Sin[x], {x, 0, 4}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn linear_polynomial_degree_two() {
    // 2x is a polynomial of degree 1; the degree-2 coefficient is 0.
    assert_eq!(interpret("SeriesCoefficient[2x, {x, 0, 2}]").unwrap(), "0");
  }

  #[test]
  fn exp_sin_fourth_coefficient() {
    // Exp[Sin[x]] = 1 + x + x^2/2 - x^4/8 - ... — degree-4 coefficient is -1/8.
    assert_eq!(
      interpret("SeriesCoefficient[Exp[Sin[x]], {x, 0, 4}]").unwrap(),
      "-1/8"
    );
  }

  // SeriesCoefficient[SeriesData[...], q] — query a SeriesData directly
  // with an integer or rational exponent. `q * den` indexes into the
  // stored coefficient list after subtracting nmin. Regression for the
  // mathics calculus.py SeriesCoefficient doctests.
  #[test]
  fn seriesdata_rational_exponent_in_range() {
    // Table[i^2, {i, 10}] = {1, 4, 9, ..., 100}. 14/3 * 3 = 14; idx =
    // 14 - 7 = 7; coeffs[7] = 64.
    assert_eq!(
      interpret(
        "SeriesCoefficient[\
         SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], \
         14/3]"
      )
      .unwrap(),
      "64"
    );
  }

  #[test]
  fn seriesdata_rational_exponent_below_range() {
    // 6/3 * 3 = 6 < nmin (7), so the series has no term there.
    assert_eq!(
      interpret(
        "SeriesCoefficient[\
         SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], \
         6/3]"
      )
      .unwrap(),
      "0"
    );
  }

  #[test]
  fn seriesdata_rational_exponent_beyond_tracked_range() {
    // 17/3 * 3 = 17; idx = 17 - 7 = 10, but coeffs only has 10 entries
    // (indices 0..9). The coefficient is past the tracked range so we
    // don't know it → Indeterminate, not 0.
    assert_eq!(
      interpret(
        "SeriesCoefficient[\
         SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], \
         17/3]"
      )
      .unwrap(),
      "Indeterminate"
    );
  }

  // SeriesCoefficient[series, {x, x0, n}] — the {x, x0, n} spec also works
  // when the first argument is an already-computed SeriesData, using n as
  // the exponent.
  #[test]
  fn seriesdata_with_spec_form() {
    assert_eq!(
      interpret("SeriesCoefficient[Series[Exp[x], {x, 0, 10}], {x, 0, 5}]")
        .unwrap(),
      "1/120"
    );
    assert_eq!(
      interpret("SeriesCoefficient[Series[Sin[x], {x, 0, 10}], {x, 0, 3}]")
        .unwrap(),
      "-1/6"
    );
    // Even-order Sine coefficient is 0.
    assert_eq!(
      interpret("SeriesCoefficient[Series[Sin[x], {x, 0, 10}], {x, 0, 2}]")
        .unwrap(),
      "0"
    );
    assert_eq!(
      interpret("SeriesCoefficient[Series[Exp[x], {x, 0, 10}], {x, 0, 0}]")
        .unwrap(),
      "1"
    );
  }

  // A symbolic index returns the general term as a Piecewise for the
  // recognized standard Maclaurin series.
  #[test]
  fn symbolic_index_exp() {
    assert_eq!(
      interpret("SeriesCoefficient[Exp[x], {x, 0, n}]").unwrap(),
      "Piecewise[{{n!^(-1), n >= 0}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[Exp[2 x], {x, 0, n}]").unwrap(),
      "Piecewise[{{2^n/n!, n >= 0}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[Exp[-x], {x, 0, n}]").unwrap(),
      "Piecewise[{{(-1)^n/n!, n >= 0}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[Exp[a x], {x, 0, n}]").unwrap(),
      "Piecewise[{{a^n/n!, n >= 0}}, 0]"
    );
  }

  #[test]
  fn symbolic_index_geometric() {
    assert_eq!(
      interpret("SeriesCoefficient[1/(1 - x), {x, 0, n}]").unwrap(),
      "Piecewise[{{1, n >= 0}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[1/(1 - 2 x), {x, 0, n}]").unwrap(),
      "Piecewise[{{2^n, n >= 0}}, 0]"
    );
    // Higher powers give the polynomial binomial term over a common denominator.
    assert_eq!(
      interpret("SeriesCoefficient[1/(1 - x)^2, {x, 0, n}]").unwrap(),
      "Piecewise[{{1 + n, n >= 0}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[1/(1 - x)^3, {x, 0, n}]").unwrap(),
      "Piecewise[{{(2 + 3*n + n^2)/2, n >= 0}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[1/(1 - 2 x)^2, {x, 0, n}]").unwrap(),
      "Piecewise[{{2^n*(1 + n), n >= 0}}, 0]"
    );
  }

  #[test]
  fn symbolic_index_hyperbolic() {
    assert_eq!(
      interpret("SeriesCoefficient[Cosh[x], {x, 0, n}]").unwrap(),
      "Piecewise[{{n!^(-1), Mod[n, 2] == 0 && n >= 0}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[Sinh[x], {x, 0, n}]").unwrap(),
      "Piecewise[{{n!^(-1), Mod[n, 2] == 1 && n >= 0}}, 0]"
    );
  }

  // An unrecognized form is left unevaluated (not crashed).
  #[test]
  fn symbolic_index_unrecognized_unevaluated() {
    assert_eq!(
      interpret("SeriesCoefficient[Exp[x^2], {x, 0, n}]").unwrap(),
      "SeriesCoefficient[E^x^2, {x, 0, n}]"
    );
  }

  // (a + b x)^p: the binomial series. A positive integer exponent bounds the
  // index; a non-unit constant term keeps the a^(p-n) factor.
  #[test]
  fn symbolic_index_binomial() {
    assert_eq!(
      interpret("SeriesCoefficient[(1 + x)^5, {x, 0, n}]").unwrap(),
      "Piecewise[{{Binomial[5, n], Inequality[0, LessEqual, n, LessEqual, 5]}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[(1 + 2 x)^3, {x, 0, n}]").unwrap(),
      "Piecewise[{{2^n*Binomial[3, n], Inequality[0, LessEqual, n, LessEqual, 3]}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[(x + 2)^3, {x, 0, n}]").unwrap(),
      "Piecewise[{{2^(3 - n)*Binomial[3, n], Inequality[0, LessEqual, n, LessEqual, 3]}}, 0]"
    );
    // A fractional exponent runs for all n >= 0.
    assert_eq!(
      interpret("SeriesCoefficient[Sqrt[1 + x], {x, 0, n}]").unwrap(),
      "Piecewise[{{Binomial[1/2, n], n >= 0}}, 0]"
    );
  }

  #[test]
  fn symbolic_index_log() {
    assert_eq!(
      interpret("SeriesCoefficient[Log[1 + x], {x, 0, n}]").unwrap(),
      "Piecewise[{{-((-1)^n/n), n >= 1}}, 0]"
    );
    assert_eq!(
      interpret("SeriesCoefficient[Log[1 + 2 x], {x, 0, n}]").unwrap(),
      "Piecewise[{{-((-2)^n/n), n >= 1}}, 0]"
    );
  }
}

mod exp_to_trig {
  use super::*;

  #[test]
  fn exp_ix() {
    assert_eq!(
      interpret("ExpToTrig[Exp[I x]]").unwrap(),
      "Cos[x] + I*Sin[x]"
    );
  }

  #[test]
  fn exp_real() {
    assert_eq!(interpret("ExpToTrig[Exp[x]]").unwrap(), "Cosh[x] + Sinh[x]");
  }

  #[test]
  fn exp_2ix() {
    assert_eq!(
      interpret("ExpToTrig[Exp[2 I x]]").unwrap(),
      "Cos[2*x] + I*Sin[2*x]"
    );
  }

  #[test]
  fn exp_3x() {
    assert_eq!(
      interpret("ExpToTrig[Exp[3 x]]").unwrap(),
      "Cosh[3*x] + Sinh[3*x]"
    );
  }

  #[test]
  fn in_sum() {
    assert_eq!(
      interpret("ExpToTrig[x + Exp[I y]]").unwrap(),
      "x + Cos[y] + I*Sin[y]"
    );
  }
}

mod trig_to_exp {
  use super::*;

  #[test]
  fn cos_to_exp() {
    let result = interpret("TrigToExp[Cos[x]]").unwrap();
    // Should contain exponential terms with I
    assert!(
      result.contains("E^") && result.contains('I'),
      "Expected exponential form, got: {result}"
    );
  }

  #[test]
  fn cosh_to_exp() {
    assert_eq!(
      interpret("TrigToExp[Cosh[x]]").unwrap(),
      "1/(2*E^x) + E^x/2"
    );
  }

  #[test]
  fn sinh_to_exp() {
    assert_eq!(
      interpret("TrigToExp[Sinh[x]]").unwrap(),
      "-1/2*1/E^x + E^x/2"
    );
  }

  #[test]
  fn sec_to_exp() {
    assert_eq!(
      interpret("TrigToExp[Sec[x]]").unwrap(),
      "2/(E^(-I*x) + E^(I*x))"
    );
  }

  #[test]
  fn cot_to_exp() {
    assert_eq!(
      interpret("TrigToExp[Cot[x]]").unwrap(),
      "(-I*(E^(-I*x) + E^(I*x)))/(E^(-I*x) - E^(I*x))"
    );
  }

  #[test]
  fn tan_to_exp() {
    // Tan[x] = I*(E^(-I*x) - E^(I*x))/(E^(-I*x) + E^(I*x)); the E^(-I*x) term
    // comes first so the negated term keeps a positive exponent, matching
    // wolframscript's form exactly.
    assert_eq!(
      interpret("TrigToExp[Tan[x]]").unwrap(),
      "(I*(E^(-I*x) - E^(I*x)))/(E^(-I*x) + E^(I*x))"
    );
  }

  #[test]
  fn tan_to_exp_double_angle() {
    assert_eq!(
      interpret("TrigToExp[Tan[2 y]]").unwrap(),
      "(I*(E^((-2*I)*y) - E^((2*I)*y)))/(E^((-2*I)*y) + E^((2*I)*y))"
    );
  }

  #[test]
  fn csc_to_exp() {
    // Csc[x] = -2*I/(E^(-I*x) - E^(I*x)); the imaginary-exponent denominator
    // renders identically to wolframscript.
    assert_eq!(
      interpret("TrigToExp[Csc[x]]").unwrap(),
      "(-2*I)/(E^(-I*x) - E^(I*x))"
    );
  }

  #[test]
  fn csc_to_exp_double_angle() {
    assert_eq!(
      interpret("TrigToExp[Csc[2 y]]").unwrap(),
      "(-2*I)/(E^((-2*I)*y) - E^((2*I)*y))"
    );
  }

  // Csc composed with its reciprocal collapses to 1 after conversion.
  #[test]
  fn csc_times_sin_to_exp() {
    assert_eq!(interpret("TrigToExp[Sin[x] Csc[x]]").unwrap(), "1");
  }

  #[test]
  fn csc_to_exp_input_form() {
    assert_eq!(
      interpret("ToString[TrigToExp[Csc[x]], InputForm]").unwrap(),
      "(-2*I)/(E^((-I)*x) - E^(I*x))"
    );
  }

  #[test]
  fn sech_to_exp() {
    assert_eq!(interpret("TrigToExp[Sech[x]]").unwrap(), "2/(E^(-x) + E^x)");
  }

  // InputForm parenthesises imaginary coefficients: `(-I)*x`, `(I/2)*x`.
  // OutputForm (the bare echo above) keeps the bare `-I*x` / `I/2*x` form.
  #[test]
  fn imaginary_coefficient_input_form() {
    assert_eq!(
      interpret("ToString[TrigToExp[Sec[x]], InputForm]").unwrap(),
      "2/(E^((-I)*x) + E^(I*x))"
    );
    assert_eq!(
      interpret("ToString[TrigToExp[Cot[x]], InputForm]").unwrap(),
      "((-I)*(E^((-I)*x) + E^(I*x)))/(E^((-I)*x) - E^(I*x))"
    );
    assert_eq!(
      interpret("ToString[TrigToExp[ArcTan[x]], InputForm]").unwrap(),
      "(I/2)*Log[1 - I*x] - (I/2)*Log[1 + I*x]"
    );
    assert_eq!(
      interpret("ToString[2^(-I*x), InputForm]").unwrap(),
      "2^((-I)*x)"
    );
    // OutputForm bare echo keeps the bare form (no extra parens).
    assert_eq!(interpret("E^(I Pi/4)").unwrap(), "E^(I/4*Pi)");
    assert_eq!(
      interpret("CharacteristicFunction[UniformDistribution[{a, b}], t]")
        .unwrap(),
      "(-I*(-E^(I*a*t) + E^(I*b*t)))/((-a + b)*t)"
    );
  }

  #[test]
  fn tanh_to_exp() {
    assert_eq!(
      interpret("TrigToExp[Tanh[x]]").unwrap(),
      "-(1/(E^x*(E^(-x) + E^x))) + E^x/(E^(-x) + E^x)"
    );
  }

  #[test]
  fn tanh_to_exp_double_angle() {
    assert_eq!(
      interpret("TrigToExp[Tanh[2 x]]").unwrap(),
      "-(1/(E^(2*x)*(E^(-2*x) + E^(2*x)))) + E^(2*x)/(E^(-2*x) + E^(2*x))"
    );
  }

  #[test]
  fn tanh_to_exp_sum_argument() {
    assert_eq!(
      interpret("TrigToExp[Tanh[a + b]]").unwrap(),
      "-(E^(-a - b)/(E^(-a - b) + E^(a + b))) + E^(a + b)/(E^(-a - b) + E^(a + b))"
    );
  }

  #[test]
  fn sec_double_angle() {
    assert_eq!(
      interpret("TrigToExp[Sec[2 x]]").unwrap(),
      "2/(E^((-2*I)*x) + E^((2*I)*x))"
    );
  }

  #[test]
  fn symbolic() {
    // TrigToExp should not affect non-trig expressions
    assert_eq!(interpret("TrigToExp[x + 1]").unwrap(), "1 + x");
  }

  // Inverse trigonometric and hyperbolic functions expand to logarithmic
  // forms. (ArcSin/ArcCos/ArcCsch/ArcSech are intentionally not expanded:
  // their Log argument is a Plus whose term order Woxi canonicalizes
  // differently from wolframscript.)
  #[test]
  fn arctan_to_log() {
    assert_eq!(
      interpret("TrigToExp[ArcTan[x]]").unwrap(),
      "I/2*Log[1 - I*x] - I/2*Log[1 + I*x]"
    );
  }

  #[test]
  fn arctan_double_angle() {
    assert_eq!(
      interpret("TrigToExp[ArcTan[2 x]]").unwrap(),
      "I/2*Log[1 - (2*I)*x] - I/2*Log[1 + (2*I)*x]"
    );
  }

  #[test]
  fn arccot_to_log() {
    assert_eq!(
      interpret("TrigToExp[ArcCot[x]]").unwrap(),
      "I/2*Log[1 - I/x] - I/2*Log[1 + I/x]"
    );
  }

  #[test]
  fn arcsec_to_log() {
    assert_eq!(
      interpret("TrigToExp[ArcSec[x]]").unwrap(),
      "Pi/2 + I*Log[Sqrt[1 - x^(-2)] + I/x]"
    );
  }

  #[test]
  fn arccsc_to_log() {
    assert_eq!(
      interpret("TrigToExp[ArcCsc[x]]").unwrap(),
      "-I*Log[Sqrt[1 - x^(-2)] + I/x]"
    );
  }

  #[test]
  fn arcsinh_to_log() {
    assert_eq!(
      interpret("TrigToExp[ArcSinh[x]]").unwrap(),
      "Log[x + Sqrt[1 + x^2]]"
    );
  }

  #[test]
  fn arccosh_to_log() {
    assert_eq!(
      interpret("TrigToExp[ArcCosh[x]]").unwrap(),
      "Log[x + Sqrt[-1 + x]*Sqrt[1 + x]]"
    );
  }

  #[test]
  fn arctanh_to_log() {
    assert_eq!(
      interpret("TrigToExp[ArcTanh[x]]").unwrap(),
      "-1/2*Log[1 - x] + Log[1 + x]/2"
    );
  }

  #[test]
  fn arccoth_to_log() {
    assert_eq!(
      interpret("TrigToExp[ArcCoth[x]]").unwrap(),
      "-1/2*Log[1 - x^(-1)] + Log[1 + x^(-1)]/2"
    );
  }
}

mod interpolation {
  use super::*;

  #[test]
  fn basic_list_of_values() {
    // Interpolation[{y1, y2, ...}] — x values are 1, 2, 3, ...
    let result =
      interpret("f = Interpolation[{1, 2, 3, 5, 8, 5}]; f[1]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 1.0).abs() < 0.001, "Expected 1.0, got {val}");
  }

  #[test]
  fn values_at_data_points() {
    // Interpolation should return exact values at data points
    let result =
      interpret("f = Interpolation[{1, 2, 3, 5, 8, 5}]; f[4]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 5.0).abs() < 0.001, "Expected 5.0, got {val}");
  }

  #[test]
  fn last_data_point() {
    let result =
      interpret("f = Interpolation[{1, 2, 3, 5, 8, 5}]; f[6]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 5.0).abs() < 0.001, "Expected 5.0, got {val}");
  }

  #[test]
  fn explicit_xy_pairs() {
    // Interpolation[{{x1, y1}, {x2, y2}, ...}]
    let result =
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}]; f[2]")
        .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 4.0).abs() < 0.001, "Expected 4.0, got {val}");
  }

  // Exact symbolic data values (e.g. Sin[1], Sin[2]) are numericised via N,
  // matching wolframscript. Previously these raised a hard "cannot convert
  // ... to numeric value" error. Values verified against wolframscript.
  #[test]
  fn symbolic_values_are_numericized() {
    // Table[{x, Sin[x]}, ...] yields exact Sin[k] y-values.
    let result =
      interpret("Interpolation[Table[{x, Sin[x]}, {x, 0, 10}]][5.5]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - (-0.69032762869809)).abs() < 1e-9,
      "expected -0.69032762869809, got {val}"
    );
  }

  #[test]
  fn symbolic_value_list_is_numericized() {
    // A bare list of exact symbolic values (x = 1, 2, 3, ...).
    let result =
      interpret("Interpolation[{Sin[1], Sin[2], Sin[3], Sin[4]}][2.5]")
        .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!(
      (val - 0.5855680265293732).abs() < 1e-9,
      "expected 0.5855680265293732, got {val}"
    );
  }

  // A single data point is a constant interpolation (order reduced to 0):
  // the function returns that value for any input.
  #[test]
  fn single_point_is_constant() {
    assert_eq!(interpret("Interpolation[{5}][1]").unwrap(), "5");
    assert_eq!(interpret("Interpolation[{5}][100]").unwrap(), "5");
    assert_eq!(
      interpret("Head[Interpolation[{7}]]").unwrap(),
      "InterpolatingFunction"
    );
  }

  // An empty data list emits innd and stays unevaluated (not a hard error).
  #[test]
  fn empty_data_emits_innd() {
    let r = woxi::interpret_with_stdout("Interpolation[{}]").unwrap();
    assert_eq!(r.result, "Interpolation[{}]");
    assert!(
      r.warnings.iter().any(|w| w.contains("Interpolation::innd")),
      "expected innd, got {:?}",
      r.warnings
    );
  }

  #[test]
  fn interpolation_between_points() {
    // Test interpolation at a point between data values
    let result =
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}]; f[1.5]")
        .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    // Cubic interpolation of x^2 data should approximate 1.5^2 = 2.25
    assert!((val - 2.25).abs() < 0.5, "Expected ~2.25, got {val}");
  }

  #[test]
  fn returns_interpolating_function() {
    let result = interpret("Interpolation[{1, 2, 3, 4}]").unwrap();
    assert!(
      result.contains("InterpolatingFunction"),
      "Expected InterpolatingFunction, got: {result}"
    );
    assert!(
      result.contains("<>"),
      "Expected <> in display, got: {result}"
    );
  }

  #[test]
  fn interpolation_order_1() {
    // Linear interpolation
    let result = interpret(
      "f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}, InterpolationOrder -> 1]; f[0.5]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    // Linear interpolation between (0,0) and (1,1): 0.5
    assert!((val - 0.5).abs() < 0.001, "Expected 0.5, got {val}");
  }

  #[test]
  fn interpolation_order_1_second_interval() {
    let result = interpret(
      "f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}, InterpolationOrder -> 1]; f[1.5]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    // Linear interpolation between (1,1) and (2,4): 1 + 0.5*3 = 2.5
    assert!((val - 2.5).abs() < 0.001, "Expected 2.5, got {val}");
  }

  #[test]
  fn domain_display() {
    let result =
      interpret("Interpolation[{{0, 1}, {1, 2}, {2, 3}, {3, 4}}]").unwrap();
    assert!(
      result.contains("{{0., 3.}}"),
      "Expected domain {{0., 3.}}, got: {result}"
    );
  }

  // InterpolatingFunction[…]["property"] returns grid metadata. The implicit
  // grid coordinates are reported as integers.
  #[test]
  fn property_domain() {
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]["Domain"]"#).unwrap(),
      "{{1, 4}}"
    );
  }

  #[test]
  fn property_grid_and_values() {
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]["Grid"]"#).unwrap(),
      "{{1}, {2}, {3}, {4}}"
    );
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]["ValuesOnGrid"]"#).unwrap(),
      "{1, 4, 9, 16}"
    );
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]["Coordinates"]"#).unwrap(),
      "{{1, 2, 3, 4}}"
    );
  }

  #[test]
  fn property_orders() {
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]["InterpolationOrder"]"#)
        .unwrap(),
      "{3}"
    );
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]["DerivativeOrder"]"#).unwrap(),
      "0"
    );
  }

  // Explicit {x, y} pairs report their own grid.
  #[test]
  fn property_explicit_pairs() {
    assert_eq!(
      interpret(
        r#"Interpolation[{{0, 0}, {2, 4}, {4, 16}, {6, 36}}]["Domain"]"#
      )
      .unwrap(),
      "{{0, 6}}"
    );
    assert_eq!(
      interpret(
        r#"Interpolation[{{0, 0}, {2, 4}, {4, 16}, {6, 36}}]["ValuesOnGrid"]"#
      )
      .unwrap(),
      "{0, 4, 16, 36}"
    );
  }

  #[test]
  fn symbolic_argument_returns_unevaluated() {
    let result = interpret("f = Interpolation[{1, 2, 3, 4}]; f[x]").unwrap();
    assert!(
      result.contains("InterpolatingFunction"),
      "Expected unevaluated form with symbolic arg, got: {result}"
    );
  }

  #[test]
  fn order_reduced_when_too_few_points() {
    // Default order 3 with only 3 points should reduce to order 2
    let result =
      interpret("f = Interpolation[{{1, 3}, {2, 5}, {3, 11}}]; f[1.5]")
        .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 3.5).abs() < 0.001, "Expected 3.5, got {val}");
  }

  #[test]
  fn quadratic_interpolation() {
    // Quadratic interpolation of x^2 data: 1.5^2 = 2.25
    let result = interpret(
      "f = Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}, {4, 16}}, InterpolationOrder -> 2]; f[1.5]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 2.25).abs() < 0.001, "Expected 2.25, got {val}");
  }

  // Default order-3 interpolation must reproduce a degree-<=3 polynomial
  // EXACTLY. wolframscript uses local (divided-difference) polynomial
  // interpolation, not a natural cubic spline whose zero-curvature boundary
  // conditions would distort the fit. These check the exact values a spline
  // got wrong (e.g. 6.2 instead of 6.25).
  #[test]
  fn cubic_reproduces_quadratic_exactly() {
    // {1,4,9,16} is x^2 at x=1..4; cubic through 4 points → exact 2.5^2.
    assert_eq!(interpret("Interpolation[{1,4,9,16}][2.5]").unwrap(), "6.25");
    assert_eq!(
      interpret("Interpolation[{1,4,9,16}][3.5]").unwrap(),
      "12.25"
    );
    // Five-point x^2 data still exact under the local cubic stencil.
    assert_eq!(
      interpret("Interpolation[{1,4,9,16,25}][2.5]").unwrap(),
      "6.25"
    );
  }

  #[test]
  fn cubic_reproduces_cubic_exactly() {
    // {0,1,8,27,64} is x^3 at x=1..5; default cubic must be exact: 2.5^3.
    assert_eq!(
      interpret("Interpolation[{0,1,8,27,64}][2.5]").unwrap(),
      "3.375"
    );
  }

  #[test]
  fn cubic_local_lagrange_nonpolynomial() {
    // Non-polynomial data: local cubic Lagrange through the nearest 4 points.
    // Verified against wolframscript (2.3125 / 3.9375).
    assert_eq!(
      interpret("Interpolation[{2,3,5,7,11}][1.5]").unwrap(),
      "2.3125"
    );
    assert_eq!(
      interpret("Interpolation[{2,3,5,7,11}][2.5]").unwrap(),
      "3.9375"
    );
  }

  // Regression: the cubic stencil for x in [x_i, x_i+1] must be the centered
  // {i-1, i, i+1, i+2} window (clamped at the ends). The old window trailed
  // behind the interval — for x in [4, 5] it used the points at x = 1..4,
  // extrapolating past the window instead of interpolating (8.625 came out
  // as 7.6875).
  #[test]
  fn cubic_stencil_centered_on_interval() {
    // Window {2, 3, 4, 5}: 3, 5, 7, 11 at x = 3.5 and 4.5.
    assert_eq!(
      interpret("Interpolation[{2,3,5,7,11}][3.5]").unwrap(),
      "5.875"
    );
    assert_eq!(
      interpret("Interpolation[{2,3,5,7,11}][4.5]").unwrap(),
      "8.625"
    );
  }
}

mod list_interpolation {
  use super::*;

  #[test]
  fn basic_evaluation() {
    // ListInterpolation is an alias for Interpolation
    let result =
      interpret("f = ListInterpolation[{1, 4, 9, 16}]; f[1]").unwrap();
    assert_eq!(result, "1");
  }

  #[test]
  fn endpoint() {
    let result =
      interpret("f = ListInterpolation[{1, 4, 9, 16}]; Round[f[4]]").unwrap();
    assert_eq!(result, "16");
  }

  #[test]
  fn returns_interpolating_function() {
    let result = interpret("Head[ListInterpolation[{1, 4, 9, 16}]]").unwrap();
    assert_eq!(result, "InterpolatingFunction");
  }

  // A {{xmin, xmax}} domain spec spaces the values uniformly across the
  // interval instead of over the default 1, 2, 3, … grid.
  #[test]
  fn domain_spec_maps_grid() {
    // {1, 4, 9} over [0, 2] sits at x = 0, 1, 2.
    assert_eq!(
      interpret("ListInterpolation[{1, 4, 9}, {{0, 2}}][1]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("ListInterpolation[{1, 4, 9}, {{0, 2}}][2]").unwrap(),
      "9"
    );
    // A shifted domain: {1, 4, 9} over [10, 20] has its midpoint at 15.
    assert_eq!(
      interpret("ListInterpolation[{1, 4, 9}, {{10, 20}}][15]").unwrap(),
      "4"
    );
    // Linear interpolation across a domain.
    assert_eq!(
      interpret(
        "ListInterpolation[{1, 4, 9, 16}, {{0, 3}}, \
         InterpolationOrder -> 1][1.5]"
      )
      .unwrap(),
      "6.5"
    );
  }

  #[test]
  fn order_reduction_uses_listinterpolation_tag() {
    // The order-reduction warning must be tagged with the actual head
    // (ListInterpolation), not Interpolation.
    woxi::clear_state();
    assert_eq!(interpret("ListInterpolation[{1, 4, 9}][2]").unwrap(), "4");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "ListInterpolation::inhr: Requested order is too high; order has been reduced to {2}."
      )),
      "expected ListInterpolation::inhr, got {msgs:?}"
    );
  }

  #[test]
  fn interpolation_order_reduction_keeps_interpolation_tag() {
    woxi::clear_state();
    assert_eq!(interpret("Interpolation[{1, 4, 9}][2]").unwrap(), "4");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Interpolation::inhr: Requested order is too high; order has been reduced to {2}."
      )),
      "expected Interpolation::inhr, got {msgs:?}"
    );
  }

  #[test]
  fn non_list_first_argument_emits_innd_and_keeps_head() {
    // A non-list first argument keeps the call unevaluated with its own head
    // and emits the Interpolation::innd message (always tagged Interpolation).
    woxi::clear_state();
    assert_eq!(
      interpret("ListInterpolation[foo]").unwrap(),
      "ListInterpolation[foo]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Interpolation::innd: First argument in foo does not contain a list of data and coordinates."
      )),
      "expected Interpolation::innd, got {msgs:?}"
    );
  }

  // A numeric matrix is a 2-D grid (values on the integer grid), evaluated by
  // tensor-product Lagrange interpolation.
  #[test]
  fn grid_bilinear() {
    // Order 1 = bilinear interpolation within a cell.
    assert_eq!(
      interpret(
        "ListInterpolation[{{1, 4, 9}, {16, 25, 36}, {49, 64, 81}}, InterpolationOrder -> 1][1.5, 2.5]"
      )
      .unwrap(),
      "18.5"
    );
  }

  #[test]
  fn grid_simple_cell() {
    assert_eq!(
      interpret("ListInterpolation[{{1, 2, 3}, {4, 5, 6}}][1.5, 2]").unwrap(),
      "3.5"
    );
  }

  #[test]
  fn grid_default_order() {
    // Default order reduces per dimension to {2, 2} for a 3x3 grid.
    assert_eq!(
      interpret(
        "ListInterpolation[{{1, 4, 9}, {16, 25, 36}, {49, 64, 81}}][1.5, 2.5]"
      )
      .unwrap(),
      "16."
    );
  }

  #[test]
  fn grid_exact_point_integer_coords_keep_type() {
    // Integer coordinates at a grid point return the stored entry's type.
    assert_eq!(
      interpret(
        "ListInterpolation[{{1, 4, 9}, {16, 25, 36}, {49, 64, 81}}][1, 3]"
      )
      .unwrap(),
      "9"
    );
    // Real coordinates force a real result.
    assert_eq!(
      interpret(
        "ListInterpolation[{{1, 4, 9}, {16, 25, 36}, {49, 64, 81}}, InterpolationOrder -> 1][1., 3.]"
      )
      .unwrap(),
      "9."
    );
  }

  #[test]
  fn grid_order_reduction_message() {
    woxi::clear_state();
    interpret("ListInterpolation[{{1, 2, 3}, {4, 5, 6}}][1.5, 2.5]").unwrap();
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "ListInterpolation::inhr: Requested order is too high; order has been reduced to {1, 2}."
      )),
      "expected 2-D inhr {{1, 2}}, got {msgs:?}"
    );
  }
}

// `Interpolation` (not `ListInterpolation`, whose 3-wide rows are raw grid
// values, not `{x, y, z}` coordinates) of a flat `{x, y, z}` triple list —
// the shape `Flatten[Table[Table[{x, y, f[x, y]}, {y, ys}], {x, xs}], 1]`
// (or the equivalent built with `Join`) produces. Regression: this used to
// raise "cannot convert {1., 1., 2.} to a numeric value" because the triple
// format wasn't recognised at all; the distinct x/y coordinates now recover
// the grid so it interpolates the same way `ListInterpolation`'s implicit
// integer grid does.
mod interpolation_2d_scattered {
  use super::*;

  // f(x, y) = x + 2y is exactly linear, so both an exact grid point and an
  // interpolated point reproduce it exactly.
  #[test]
  fn exact_grid_point_keeps_integer_type() {
    assert_eq!(
      interpret(
        "Interpolation[Flatten[Table[{x, y, x + 2 y}, {x, 0, 3}, {y, 0, 3}], 1], \
         InterpolationOrder -> 1][1, 1]"
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn bilinear_of_a_linear_function_is_exact() {
    assert_eq!(
      interpret(
        "Interpolation[Flatten[Table[{x, y, x + 2 y}, {x, 0, 3}, {y, 0, 3}], 1], \
         InterpolationOrder -> 1][2.5, 1.5]"
      )
      .unwrap(),
      "5.5"
    );
  }

  // f(x, y) = x^2 doesn't depend on y, so bilinear interpolation reduces to
  // plain 1-D linear interpolation in x between the bracketing grid columns:
  // at x = 2.5 that's (4 + 9)/2 = 6.5, the same at every y.
  #[test]
  fn bilinear_interpolates_within_a_cell() {
    assert_eq!(
      interpret(
        "f = Interpolation[Flatten[Table[{x, y, x^2}, {x, 0, 4}, {y, 0, 4}], 1], \
         InterpolationOrder -> 1]; f[2.5, 1]"
      )
      .unwrap(),
      "6.5"
    );
    assert_eq!(
      interpret(
        "f = Interpolation[Flatten[Table[{x, y, x^2}, {x, 0, 4}, {y, 0, 4}], 1], \
         InterpolationOrder -> 1]; f[2.5, 3]"
      )
      .unwrap(),
      "6.5"
    );
  }

  // Default order 3, clamped per axis to the 3x3 grid's `dim - 1 = 2` (the
  // same reduction `grid_default_order` exercises for `ListInterpolation`),
  // interpolates the quadratic x^2 + y^2 exactly at the midpoint.
  #[test]
  fn default_order_is_reduced_and_still_exact_for_a_quadratic() {
    woxi::clear_state();
    assert_eq!(
      interpret(
        "Interpolation[Flatten[Table[{x, y, x^2 + y^2}, {x, 0, 2}, {y, 0, 2}], 1]][1.5, 1.5]"
      )
      .unwrap(),
      "4.5"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Interpolation::inhr: Requested order is too high; order has been reduced to {2, 2}."
      )),
      "expected 2-D inhr {{2, 2}}, got {msgs:?}"
    );
  }

  #[test]
  fn property_domain() {
    assert_eq!(
      interpret(
        "Interpolation[Flatten[Table[{x, y, x + y}, {x, 0, 3}, {y, 0, 3}], 1]][\"Domain\"]"
      )
      .unwrap(),
      "{{0, 3}, {0, 3}}"
    );
  }

  // A per-axis `InterpolationOrder -> {orderX, orderY}` is honoured
  // separately for each dimension.
  #[test]
  fn per_axis_interpolation_order() {
    assert_eq!(
      interpret(
        "Interpolation[Flatten[Table[{x, y, x + 2 y}, {x, 0, 3}, {y, 0, 3}], 1], \
         InterpolationOrder -> {1, 1}][2, 3]"
      )
      .unwrap(),
      "8"
    );
  }

  // Fewer than four points, or a triple list whose coordinates don't tile a
  // complete rectangular grid, isn't a shape this path supports (and isn't
  // the `Flatten[Table[Table[…]]]` idiom that reaches it in practice); it
  // falls back to the ordinary (non-grid) handling rather than being
  // silently misread as one, and no longer names the wrong caller in its
  // error when that fallback also can't make sense of it.
  #[test]
  fn incomplete_grid_does_not_crash_or_blame_ndsolve() {
    let err = interpret("Interpolation[{{1, 1, 2}, {1, 2, 3}, {2, 1, 4}}]")
      .unwrap_err()
      .to_string();
    assert!(
      !err.contains("NDSolve"),
      "a numeric-conversion failure reached from Interpolation must not \
       blame NDSolve: {err}"
    );
  }
}

mod trig_expand {
  use super::*;

  #[test]
  fn sin_double_angle() {
    assert_eq!(interpret("TrigExpand[Sin[2x]]").unwrap(), "2*Cos[x]*Sin[x]");
  }

  #[test]
  fn cos_double_angle() {
    assert_eq!(
      interpret("TrigExpand[Cos[2x]]").unwrap(),
      "Cos[x]^2 - Sin[x]^2"
    );
  }

  #[test]
  fn sin_triple_angle() {
    assert_eq!(
      interpret("TrigExpand[Sin[3x]]").unwrap(),
      "3*Cos[x]^2*Sin[x] - Sin[x]^3"
    );
  }

  #[test]
  fn cos_triple_angle() {
    assert_eq!(
      interpret("TrigExpand[Cos[3x]]").unwrap(),
      "Cos[x]^3 - 3*Cos[x]*Sin[x]^2"
    );
  }

  #[test]
  fn sin_sum() {
    assert_eq!(
      interpret("TrigExpand[Sin[a + b]]").unwrap(),
      "Cos[b]*Sin[a] + Cos[a]*Sin[b]"
    );
  }

  #[test]
  fn cos_sum() {
    assert_eq!(
      interpret("TrigExpand[Cos[a + b]]").unwrap(),
      "Cos[a]*Cos[b] - Sin[a]*Sin[b]"
    );
  }

  #[test]
  fn tan_double_angle() {
    assert_eq!(
      interpret("TrigExpand[Tan[2x]]").unwrap(),
      "(2*Cos[x]*Sin[x])/(Cos[x]^2 - Sin[x]^2)"
    );
  }

  #[test]
  fn sinh_double_angle() {
    assert_eq!(
      interpret("TrigExpand[Sinh[2x]]").unwrap(),
      "2*Cosh[x]*Sinh[x]"
    );
  }

  #[test]
  fn cosh_double_angle() {
    assert_eq!(
      interpret("TrigExpand[Cosh[2x]]").unwrap(),
      "Cosh[x]^2 + Sinh[x]^2"
    );
  }

  #[test]
  fn sin_no_expand() {
    // Sin[x] alone should not be expanded
    assert_eq!(interpret("TrigExpand[Sin[x]]").unwrap(), "Sin[x]");
  }

  #[test]
  fn non_trig_passthrough() {
    // Non-trig expressions should pass through
    assert_eq!(
      interpret("TrigExpand[x + Sin[2y]]").unwrap(),
      "x + 2*Cos[y]*Sin[y]"
    );
  }

  #[test]
  fn sin_quadruple_angle() {
    assert_eq!(
      interpret("TrigExpand[Sin[4x]]").unwrap(),
      "4*Cos[x]^3*Sin[x] - 4*Cos[x]*Sin[x]^3"
    );
  }

  #[test]
  fn distributes_product_over_sum() {
    // Regression: TrigExpand should distribute Times over Plus.
    assert_eq!(
      interpret("TrigExpand[Sin[x^2] * Cos[2 x]]").unwrap(),
      "Cos[x]^2*Sin[x^2] - Sin[x]^2*Sin[x^2]"
    );
  }

  #[test]
  fn distributes_sum_times_cos_sum() {
    assert_eq!(
      interpret("TrigExpand[(a + b) Cos[x + y]]").unwrap(),
      "a*Cos[x]*Cos[y] + b*Cos[x]*Cos[y] - a*Sin[x]*Sin[y] - b*Sin[x]*Sin[y]"
    );
  }

  #[test]
  fn expands_squared_sum_alongside_sin() {
    assert_eq!(
      interpret("TrigExpand[Sin[2 x] + (a + b)^2]").unwrap(),
      "a^2 + 2*a*b + b^2 + 2*Cos[x]*Sin[x]"
    );
  }
}

mod fourier_transform {
  use super::*;

  #[test]
  fn gaussian() {
    assert_eq!(
      interpret("FourierTransform[Exp[-t^2], t, w]").unwrap(),
      "1/(Sqrt[2]*E^(w^2/4))"
    );
  }

  #[test]
  fn exp_neg_abs_t() {
    assert_eq!(
      interpret("FourierTransform[Exp[-Abs[t]], t, w]").unwrap(),
      "Sqrt[2/Pi]/(1 + w^2)"
    );
  }

  // F[1/(p + q t^2)] = Sqrt[Pi/2]/Sqrt[p q] * Exp[-Sqrt[p/q] Abs[w]].
  // Verified against wolframscript.
  #[test]
  fn lorentzian() {
    assert_eq!(
      interpret("FourierTransform[1/(1 + t^2), t, w]").unwrap(),
      "Sqrt[Pi/2]/E^Abs[w]"
    );
    assert_eq!(
      interpret("FourierTransform[1/(4 + t^2), t, w]").unwrap(),
      "Sqrt[Pi/2]/(2*E^(2*Abs[w]))"
    );
    // A non-unit t^2 coefficient: p = 3, q = 2, so b = Sqrt[3/2].
    assert_eq!(
      interpret("FourierTransform[1/(3 + 2 t^2), t, w]").unwrap(),
      "Sqrt[Pi/3]/(2*E^(Sqrt[3/2]*Abs[w]))"
    );
    // A numerator constant is pulled out by linearity.
    assert_eq!(
      interpret("FourierTransform[2/(4 + t^2), t, w]").unwrap(),
      "Sqrt[Pi/2]/E^(2*Abs[w])"
    );
  }

  #[test]
  fn dirac_delta() {
    assert_eq!(
      interpret("FourierTransform[DiracDelta[t], t, w]").unwrap(),
      "1/Sqrt[2*Pi]"
    );
  }

  #[test]
  fn constant_one() {
    assert_eq!(
      interpret("FourierTransform[1, t, w]").unwrap(),
      "Sqrt[2*Pi]*DiracDelta[w]"
    );
  }

  #[test]
  fn cos_3t() {
    assert_eq!(
      interpret("FourierTransform[Cos[3 t], t, w]").unwrap(),
      "Sqrt[Pi/2]*DiracDelta[-3 + w] + Sqrt[Pi/2]*DiracDelta[3 + w]"
    );
  }

  #[test]
  fn sin_t() {
    assert_eq!(
      interpret("FourierTransform[Sin[t], t, w]").unwrap(),
      "I*Sqrt[Pi/2]*DiracDelta[-1 + w] - I*Sqrt[Pi/2]*DiracDelta[1 + w]"
    );
  }

  #[test]
  fn reciprocal_t() {
    assert_eq!(
      interpret("FourierTransform[1/t, t, w]").unwrap(),
      "(I*Pi*Sign[w])/Sqrt[2*Pi]"
    );
  }

  #[test]
  fn linearity_constant_factor() {
    assert_eq!(
      interpret("FourierTransform[3*Exp[-t^2], t, w]").unwrap(),
      "3/(Sqrt[2]*E^(w^2/4))"
    );
  }

  #[test]
  fn linearity_sum() {
    assert_eq!(
      interpret("FourierTransform[Sin[t] + Cos[t], t, w]").unwrap(),
      "(1 + I)*Sqrt[Pi/2]*DiracDelta[-1 + w] + (1 - I)*Sqrt[Pi/2]*DiracDelta[1 + w]"
    );
  }

  #[test]
  fn unevaluated_for_unknown() {
    let result = interpret("FourierTransform[f[t], t, w]").unwrap();
    assert!(
      result.contains("FourierTransform"),
      "Should return unevaluated: {result}"
    );
  }

  #[test]
  fn unit_box_half_width() {
    // FourierTransform[UnitBox[t/2], t, w] = Sqrt[2/Pi] * Sinc[w]
    assert_eq!(
      interpret("FourierTransform[UnitBox[t/2], t, w]").unwrap(),
      "Sqrt[2/Pi]*Sinc[w]"
    );
  }

  #[test]
  fn radial_inverse_sqrt_2d() {
    // FourierTransform[1/Sqrt[x^2 + y^2], {x, y}, {u, v}] = 1/Sqrt[u^2 + v^2]
    assert_eq!(
      interpret("FourierTransform[1/Sqrt[x^2 + y^2], {x, y}, {u, v}]").unwrap(),
      "1/Sqrt[u^2 + v^2]"
    );
  }
}

mod inverse_fourier_transform {
  use super::*;

  #[test]
  fn gaussian() {
    assert_eq!(
      interpret("InverseFourierTransform[Exp[-w^2], w, t]").unwrap(),
      "1/(Sqrt[2]*E^(t^2/4))"
    );
  }

  #[test]
  fn dirac_delta() {
    assert_eq!(
      interpret("InverseFourierTransform[DiracDelta[w], w, t]").unwrap(),
      "1/Sqrt[2*Pi]"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(
      interpret("InverseFourierTransform[1, w, t]").unwrap(),
      "Sqrt[2*Pi]*DiracDelta[t]"
    );
  }

  // F^-1[1/(p + q w^2)] = Sqrt[Pi/2]/Sqrt[p q] * Exp[-Sqrt[p/q] Abs[t]].
  // Verified against wolframscript.
  #[test]
  fn lorentzian() {
    assert_eq!(
      interpret("InverseFourierTransform[1/(1 + w^2), w, t]").unwrap(),
      "Sqrt[Pi/2]/E^Abs[t]"
    );
    assert_eq!(
      interpret("InverseFourierTransform[1/(9 + w^2), w, t]").unwrap(),
      "Sqrt[Pi/2]/(3*E^(3*Abs[t]))"
    );
  }

  #[test]
  fn unevaluated_for_unknown() {
    let result = interpret("InverseFourierTransform[g[w], w, t]").unwrap();
    assert!(
      result.contains("InverseFourierTransform"),
      "Should return unevaluated: {result}"
    );
  }

  #[test]
  fn sinc_inverts_to_signs_audit_case() {
    // Audit case:
    //   InverseFourierTransform[Sinc[w], w, t]
    //     = (Sqrt[Pi/2]*(Sign[1 - t] + Sign[1 + t]))/2
    assert_eq!(
      interpret("InverseFourierTransform[Sinc[w], w, t]").unwrap(),
      "(Sqrt[Pi/2]*(Sign[1 - t] + Sign[1 + t]))/2"
    );
  }
}

mod trig_reduce {
  use super::*;

  #[test]
  fn sin_squared() {
    assert_eq!(
      interpret("TrigReduce[Sin[x]^2]").unwrap(),
      "(1 - Cos[2*x])/2"
    );
  }

  #[test]
  fn cos_squared() {
    assert_eq!(
      interpret("TrigReduce[Cos[x]^2]").unwrap(),
      "(1 + Cos[2*x])/2"
    );
  }

  #[test]
  fn sum_of_terms_is_combined() {
    // After reducing each term, like terms are collected.
    assert_eq!(interpret("TrigReduce[Sin[x]^2 + Cos[x]^2]").unwrap(), "1");
    assert_eq!(
      interpret("TrigReduce[Sin[x]^2 - Cos[x]^2]").unwrap(),
      "-Cos[2*x]"
    );
    assert_eq!(
      interpret("TrigReduce[Sin[x]^2 + Cos[x]^2 + 1]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("TrigReduce[3 Sin[x]^2 + 3 Cos[x]^2]").unwrap(),
      "3"
    );
  }

  #[test]
  fn sin_cos_product() {
    assert_eq!(
      interpret("TrigReduce[Sin[x] Cos[x]]").unwrap(),
      "Sin[2*x]/2"
    );
  }

  // Mixed powers and products must be fully linearized: reducing one power
  // leaves a residual product (Cos[2x] Sin[x]) that a second pass reduces.
  // Verified against wolframscript.
  #[test]
  fn mixed_power_product_fully_linearizes() {
    assert_eq!(
      interpret("TrigReduce[Cos[x]^2 Sin[x]]").unwrap(),
      "(Sin[x] + Sin[3*x])/4"
    );
    assert_eq!(
      interpret("TrigReduce[Sin[x]^2 Cos[x]]").unwrap(),
      "(Cos[x] - Cos[3*x])/4"
    );
    assert_eq!(
      interpret("TrigReduce[Cos[x]^2 Sin[x]^2]").unwrap(),
      "(1 - Cos[4*x])/8"
    );
    assert_eq!(
      interpret("TrigReduce[Sin[x]^3 Cos[x]^2]").unwrap(),
      "(2*Sin[x] + Sin[3*x] - Sin[5*x])/16"
    );
    // A triple product collapses all the way to a single sine.
    assert_eq!(
      interpret("TrigReduce[Sin[x] Cos[x] Cos[2 x]]").unwrap(),
      "Sin[4*x]/4"
    );
  }

  // Hyperbolic powers and products reduce with the cosh/sinh identities.
  // Verified against wolframscript.
  #[test]
  fn hyperbolic_powers() {
    assert_eq!(
      interpret("TrigReduce[Sinh[x]^2]").unwrap(),
      "(-1 + Cosh[2*x])/2"
    );
    assert_eq!(
      interpret("TrigReduce[Cosh[x]^2]").unwrap(),
      "(1 + Cosh[2*x])/2"
    );
    assert_eq!(
      interpret("TrigReduce[Sinh[x]^3]").unwrap(),
      "(-3*Sinh[x] + Sinh[3*x])/4"
    );
    assert_eq!(
      interpret("TrigReduce[Cosh[x]^3]").unwrap(),
      "(3*Cosh[x] + Cosh[3*x])/4"
    );
    assert_eq!(
      interpret("TrigReduce[Sinh[x]^4]").unwrap(),
      "(3 - 4*Cosh[2*x] + Cosh[4*x])/8"
    );
    assert_eq!(
      interpret("TrigReduce[Sinh[x]^2 + Cosh[x]^2]").unwrap(),
      "Cosh[2*x]"
    );
  }

  #[test]
  fn hyperbolic_products() {
    assert_eq!(
      interpret("TrigReduce[Sinh[x] Cosh[x]]").unwrap(),
      "Sinh[2*x]/2"
    );
    assert_eq!(
      interpret("TrigReduce[Sinh[a] Sinh[b]]").unwrap(),
      "(-Cosh[a - b] + Cosh[a + b])/2"
    );
    assert_eq!(
      interpret("TrigReduce[Cosh[a] Cosh[b]]").unwrap(),
      "(Cosh[a - b] + Cosh[a + b])/2"
    );
    assert_eq!(
      interpret("TrigReduce[Sinh[x] Cosh[y]]").unwrap(),
      "(Sinh[x - y] + Sinh[x + y])/2"
    );
    // Mixed power/product linearizes across the fixed-point iteration.
    assert_eq!(
      interpret("TrigReduce[Cosh[x]^2 Sinh[x]]").unwrap(),
      "(Sinh[x] + Sinh[3*x])/4"
    );
  }

  #[test]
  fn sin_cubed() {
    assert_eq!(
      interpret("TrigReduce[Sin[x]^3]").unwrap(),
      "(3*Sin[x] - Sin[3*x])/4"
    );
  }

  #[test]
  fn cos_cubed() {
    assert_eq!(
      interpret("TrigReduce[Cos[x]^3]").unwrap(),
      "(3*Cos[x] + Cos[3*x])/4"
    );
  }

  #[test]
  fn sin_a_cos_b() {
    assert_eq!(
      interpret("TrigReduce[Sin[a] Cos[b]]").unwrap(),
      "(Sin[a - b] + Sin[a + b])/2"
    );
  }

  #[test]
  fn cos_a_cos_b() {
    assert_eq!(
      interpret("TrigReduce[Cos[a] Cos[b]]").unwrap(),
      "(Cos[a - b] + Cos[a + b])/2"
    );
  }

  #[test]
  fn sin_a_sin_b() {
    assert_eq!(
      interpret("TrigReduce[Sin[a] Sin[b]]").unwrap(),
      "(Cos[a - b] - Cos[a + b])/2"
    );
  }

  #[test]
  fn unevaluated_wrong_args() {
    assert_eq!(interpret("TrigReduce[]").unwrap(), "TrigReduce[]");
  }
}

mod function_domain {
  use super::*;

  #[test]
  fn reciprocal() {
    // FunctionDomain[1/x, x] = x < 0 || x > 0
    assert_eq!(
      interpret("FunctionDomain[1/x, x]").unwrap(),
      "x < 0 || x > 0"
    );
  }

  #[test]
  fn sqrt_x() {
    assert_eq!(interpret("FunctionDomain[Sqrt[x], x]").unwrap(), "x >= 0");
  }

  #[test]
  fn log_x() {
    assert_eq!(interpret("FunctionDomain[Log[x], x]").unwrap(), "x > 0");
  }

  #[test]
  fn polynomial() {
    // No domain restrictions for a polynomial
    assert_eq!(interpret("FunctionDomain[x^2 + 1, x]").unwrap(), "True");
  }

  #[test]
  fn sqrt_x_minus_1() {
    assert_eq!(
      interpret("FunctionDomain[Sqrt[x - 1], x]").unwrap(),
      "x >= 1"
    );
  }

  #[test]
  fn reciprocal_square() {
    // 1/(x^2 - 1) → interval complement of {-1, 1}
    assert_eq!(
      interpret("FunctionDomain[1/(x^2 - 1), x]").unwrap(),
      "x < -1 || Inequality[-1, Less, x, Less, 1] || x > 1"
    );
  }

  #[test]
  fn constant_function() {
    assert_eq!(interpret("FunctionDomain[5, x]").unwrap(), "True");
  }

  #[test]
  fn log_of_sqrt() {
    // Log[Sqrt[x]] → x > 0 && x >= 0 → simplifies to x > 0
    let result = interpret("FunctionDomain[Log[Sqrt[x]], x]").unwrap();
    assert!(
      result.contains('x') && result.contains('0'),
      "Should contain domain constraint: {result}"
    );
  }
}

mod exponential_generating_function {
  use super::*;

  #[test]
  fn egf_constant() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[1, n, x]").unwrap(),
      "E^x"
    );
  }

  #[test]
  fn egf_constant_c() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[5, n, x]").unwrap(),
      "5*E^x"
    );
  }

  #[test]
  fn egf_variable_n() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[n, n, x]").unwrap(),
      "E^x*x"
    );
  }

  #[test]
  fn egf_n_squared() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[n^2, n, x]").unwrap(),
      "E^x*x*(1 + x)"
    );
  }

  #[test]
  fn egf_exponential_2n() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[2^n, n, x]").unwrap(),
      "E^(2*x)"
    );
  }

  #[test]
  fn egf_exponential_neg1() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[(-1)^n, n, x]").unwrap(),
      "E^(-x)"
    );
  }

  #[test]
  fn egf_factorial() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[Factorial[n], n, x]").unwrap(),
      "(1 - x)^(-1)"
    );
  }

  #[test]
  fn egf_n_plus_1() {
    clear_state();
    // n + 1 → E^x*(1 + x)
    assert_eq!(
      interpret("ExponentialGeneratingFunction[n + 1, n, x]").unwrap(),
      "E^x*(1 + x)"
    );
  }

  #[test]
  fn egf_constant_times_n() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[3*n, n, x]").unwrap(),
      "3*E^x*x"
    );
  }

  #[test]
  fn egf_3_to_n() {
    clear_state();
    assert_eq!(
      interpret("ExponentialGeneratingFunction[3^n, n, x]").unwrap(),
      "E^(3*x)"
    );
  }

  #[test]
  fn egf_n_cubed() {
    clear_state();
    // S(3,1)=1, S(3,2)=3, S(3,3)=1 → x*(1 + 3*x + x^2)
    assert_eq!(
      interpret("ExponentialGeneratingFunction[n^3, n, x]").unwrap(),
      "E^x*x*(1 + 3*x + x^2)"
    );
  }

  #[test]
  fn egf_sin_n() {
    clear_state();
    // EGF[Sin[n], n, x] = Sin[x*Sin[1]] * (Cosh[x*Cos[1]] + Sinh[x*Cos[1]])
    // which equals E^(x*Cos[1]) * Sin[x*Sin[1]]
    assert_eq!(
      interpret("ExponentialGeneratingFunction[Sin[n], n, x]").unwrap(),
      "Sin[x*Sin[1]]*(Cosh[x*Cos[1]] + Sinh[x*Cos[1]])"
    );
  }

  #[test]
  fn egf_unevaluated_unknown() {
    clear_state();
    // Unknown pattern returns unevaluated
    assert_eq!(
      interpret("ExponentialGeneratingFunction[Log[n], n, x]").unwrap(),
      "ExponentialGeneratingFunction[Log[n], n, x]"
    );
  }

  #[test]
  fn egf_zero_power() {
    clear_state();
    // n^0 = 1, so EGF[1, n, x] = E^x
    assert_eq!(
      interpret("ExponentialGeneratingFunction[n^0, n, x]").unwrap(),
      "E^x"
    );
  }
}

// Asymptotic[f, x->x0] = leading series term; Asymptotic[f, {x,x0,n}] = series.
mod asymptotic {
  use super::*;

  #[test]
  fn leading_term_at_zero() {
    assert_eq!(interpret("Asymptotic[Sin[x], x -> 0]").unwrap(), "x");
    assert_eq!(interpret("Asymptotic[Cos[x], x -> 0]").unwrap(), "1");
    assert_eq!(interpret("Asymptotic[Exp[x], x -> 0]").unwrap(), "1");
    assert_eq!(interpret("Asymptotic[Log[1 + x], x -> 0]").unwrap(), "x");
    assert_eq!(
      interpret("Asymptotic[Cos[x] - 1, x -> 0]").unwrap(),
      "-1/2*x^2"
    );
  }

  #[test]
  fn leading_term_of_polynomial() {
    assert_eq!(interpret("Asymptotic[x^2 + x^3, x -> 0]").unwrap(), "x^2");
    assert_eq!(interpret("Asymptotic[1 + x^2, x -> 0]").unwrap(), "1");
    assert_eq!(interpret("Asymptotic[2 + 3 x, x -> 0]").unwrap(), "2");
    assert_eq!(interpret("Asymptotic[x + x^2 + x^3, x -> 0]").unwrap(), "x");
  }

  #[test]
  fn removable_singularity_is_quiet() {
    // Sin[x]/x has a removable singularity at 0; leading term is 1 with no
    // Power::infy / Infinity::indet messages leaking out.
    let r = interpret_with_stdout("Asymptotic[Sin[x]/x, x -> 0]").unwrap();
    assert_eq!(r.result, "1");
    assert!(
      r.warnings.is_empty(),
      "expected no messages, got: {:?}",
      r.warnings
    );
    assert_eq!(interpret("Asymptotic[E^x - 1, x -> 0]").unwrap(), "x");
  }

  #[test]
  fn nonzero_expansion_point() {
    assert_eq!(interpret("Asymptotic[Sin[x], x -> 1]").unwrap(), "Sin[1]");
    assert_eq!(interpret("Asymptotic[1/x, x -> 1]").unwrap(), "1");
  }

  #[test]
  fn constant_is_returned() {
    assert_eq!(interpret("Asymptotic[5, x -> 0]").unwrap(), "5");
  }

  #[test]
  fn order_n_list_form_is_series() {
    assert_eq!(
      interpret("Asymptotic[Sin[x], {x, 0, 5}]").unwrap(),
      "x - x^3/6 + x^5/120"
    );
    assert_eq!(
      interpret("Asymptotic[Log[1 + x], {x, 0, 3}]").unwrap(),
      "x - x^2/2 + x^3/3"
    );
  }
}

mod asymptotic_solve {
  use super::*;

  #[test]
  fn integer_third_arg_unevaluated() {
    clear_state();
    // A plain integer 3rd arg is invalid; Wolfram returns unevaluated
    let result = interpret("AsymptoticSolve[x - 1 == 0, x -> 0, 3]").unwrap();
    assert!(
      result.starts_with("AsymptoticSolve["),
      "expected unevaluated for integer 3rd arg, got {result}"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    clear_state();
    // With wrong number of args
    let result = interpret("AsymptoticSolve[x^2 - 1]").unwrap();
    assert!(
      result.starts_with("AsymptoticSolve["),
      "expected unevaluated, got {result}"
    );
  }
}

mod fourier_sin_transform {
  use super::*;

  #[test]
  fn exp_decay() {
    clear_state();
    // FourierSinTransform[E^(-a*t), t, w] = Sqrt[2/Pi] * w / (a^2 + w^2)
    let result = interpret("FourierSinTransform[E^(-a*t), t, w]").unwrap();
    assert!(
      result.contains('w') && !result.contains("FourierSinTransform"),
      "expected evaluated result, got {result}"
    );
  }

  #[test]
  fn linearity() {
    clear_state();
    let result = interpret("FourierSinTransform[3*E^(-t), t, w]").unwrap();
    assert!(
      !result.contains("FourierSinTransform"),
      "expected evaluated result, got {result}"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    clear_state();
    let result = interpret("FourierSinTransform[f[t], t, w]").unwrap();
    assert!(
      result.contains("FourierSinTransform"),
      "expected unevaluated, got {result}"
    );
  }

  #[test]
  fn cos_over_t_symbolic() {
    // FourierSinTransform[Cos[t]/t, t, w] = (Pi - Pi*Sign[1-w])/(2*Sqrt[2*Pi]).
    clear_state();
    assert_eq!(
      interpret("FourierSinTransform[Cos[t]/t, t, w]").unwrap(),
      "(Pi - Pi*Sign[1 - w])/(2*Sqrt[2*Pi])"
    );
  }

  #[test]
  fn cos_over_t_real_w_above_one_audit_case() {
    // Audit case: numeric Real w > 1.
    assert_eq!(
      interpret("FourierSinTransform[Cos[t]/t, t, 1.2]").unwrap(),
      "1.2533141373155003 + 0.*I"
    );
  }

  #[test]
  fn cos_over_t_real_w_below_one() {
    // Numeric Real w < 1 → 0. + 0.*I (matches wolframscript).
    assert_eq!(
      interpret("FourierSinTransform[Cos[t]/t, t, 0.5]").unwrap(),
      "0. + 0.*I"
    );
  }

  #[test]
  fn cos_over_t_integer_w_above_one() {
    // Integer w > 1 → symbolic Pi/Sqrt[2*Pi].
    assert_eq!(
      interpret("FourierSinTransform[Cos[t]/t, t, 2]").unwrap(),
      "Pi/Sqrt[2*Pi]"
    );
  }

  #[test]
  fn cos_over_t_integer_w_equals_one() {
    // w = 1 boundary: Pi/(2*Sqrt[2*Pi]).
    assert_eq!(
      interpret("FourierSinTransform[Cos[t]/t, t, 1]").unwrap(),
      "Pi/(2*Sqrt[2*Pi])"
    );
  }
}

mod fourier_cos_transform {
  use super::*;

  #[test]
  fn exp_decay() {
    clear_state();
    let result = interpret("FourierCosTransform[E^(-a*t), t, w]").unwrap();
    assert!(
      result.contains('a') && !result.contains("FourierCosTransform"),
      "expected evaluated result, got {result}"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    clear_state();
    let result = interpret("FourierCosTransform[g[t], t, w]").unwrap();
    assert!(
      result.contains("FourierCosTransform"),
      "expected unevaluated, got {result}"
    );
  }

  #[test]
  fn inverse_sqrt() {
    // FourierCosTransform[1/Sqrt[t], t, w] = 1/Sqrt[w]
    clear_state();
    assert_eq!(
      interpret("FourierCosTransform[1/Sqrt[t], t, w]").unwrap(),
      "1/Sqrt[w]"
    );
  }

  #[test]
  fn gaussian() {
    // FourierCosTransform[E^(-t^2), t, w] = 1/(Sqrt[2]*E^(w^2/4))
    clear_state();
    assert_eq!(
      interpret("FourierCosTransform[E^(-t^2), t, w]").unwrap(),
      "1/(Sqrt[2]*E^(w^2/4))"
    );
  }

  #[test]
  fn gaussian_scaled() {
    // FourierCosTransform[E^(-a*t^2), t, w] = E^(-w^2/(4 a))/Sqrt[2 a].
    clear_state();
    let result = interpret("FourierCosTransform[E^(-2*t^2), t, w]").unwrap();
    assert_eq!(result, "1/(2*E^(w^2/8))");
  }

  #[test]
  fn radial_inverse_distance_2d() {
    // 2-D radial FCT: 1/Sqrt[x^2 + y^2] → 1/Sqrt[u^2 + v^2].
    clear_state();
    assert_eq!(
      interpret("FourierCosTransform[1/Sqrt[x^2 + y^2], {x, y}, {u, v}]")
        .unwrap(),
      "1/Sqrt[u^2 + v^2]"
    );
    // Symmetric in the order of {t1, t2}.
    assert_eq!(
      interpret("FourierCosTransform[1/Sqrt[y^2 + x^2], {x, y}, {u, v}]")
        .unwrap(),
      "1/Sqrt[u^2 + v^2]"
    );
  }
}

mod fourier_sin_cos_sqrt {
  use super::*;

  #[test]
  fn fst_inverse_sqrt() {
    // FourierSinTransform[1/Sqrt[t], t, w] = 1/Sqrt[w]
    clear_state();
    assert_eq!(
      interpret("FourierSinTransform[1/Sqrt[t], t, w]").unwrap(),
      "1/Sqrt[w]"
    );
  }
}

mod discrete_convolve {
  use super::*;

  #[test]
  fn finite_sum_case() {
    clear_state();
    // DiscreteConvolve with Piecewise-like expressions
    // Use UnitStep to create finite support:
    // f[n] = UnitStep[n] * UnitStep[2-n] (nonzero for n=0,1,2)
    // DiscreteConvolve on simple expressions
    // For now just test that the function runs and returns something
    let result =
      interpret("DiscreteConvolve[KroneckerDelta[n], KroneckerDelta[m], n, m]")
        .unwrap();
    // The Sum may not simplify the infinite sum, but it should return something
    assert!(!result.is_empty(), "expected non-empty result");
  }

  #[test]
  fn symbolic_unevaluated() {
    clear_state();
    assert_eq!(
      interpret("DiscreteConvolve[f, g, n]").unwrap(),
      "DiscreteConvolve[f, g, n]"
    );
  }

  #[test]
  fn divergent_stays_symbolic_no_leak() {
    // A convolution whose defining sum does not reduce to a closed form must
    // stay symbolic, not leak the internal `Sum[…, {k$dc, -Infinity,
    // Infinity}]`. Matches wolframscript, which also leaves these unevaluated.
    clear_state();
    assert_eq!(
      interpret("DiscreteConvolve[1, 1, n, m]").unwrap(),
      "DiscreteConvolve[1, 1, n, m]"
    );
    let r = interpret("DiscreteConvolve[2^n, 3^n, n, m]").unwrap();
    assert_eq!(r, "DiscreteConvolve[2^n, 3^n, n, m]");
    // No internal summation variable or raw Sum leaks through.
    assert!(!r.contains("$k"), "internal sum variable leaked: {r}");
    assert!(!r.contains("Sum["), "raw Sum leaked: {r}");
  }

  // Convolving with KroneckerDelta[n] leaves the other sequence unchanged (at m).
  #[test]
  fn kronecker_delta_identity() {
    clear_state();
    assert_eq!(
      interpret("DiscreteConvolve[a^n, KroneckerDelta[n], n, m]").unwrap(),
      "a^m"
    );
    assert_eq!(
      interpret("DiscreteConvolve[1, KroneckerDelta[n], n, m]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("DiscreteConvolve[n^2, KroneckerDelta[n], n, m]").unwrap(),
      "m^2"
    );
    assert_eq!(
      interpret("DiscreteConvolve[Sin[n], KroneckerDelta[n], n, m]").unwrap(),
      "Sin[m]"
    );
    assert_eq!(
      interpret("DiscreteConvolve[KroneckerDelta[n], KroneckerDelta[n], n, m]")
        .unwrap(),
      "KroneckerDelta[m]"
    );
  }

  // A shifted delta KroneckerDelta[n - c] shifts the sequence by c.
  #[test]
  fn kronecker_delta_shift() {
    clear_state();
    assert_eq!(
      interpret("DiscreteConvolve[a^n, KroneckerDelta[n - 2], n, m]").unwrap(),
      "a^(-2 + m)"
    );
    // The delta may be either operand (convolution is commutative).
    assert_eq!(
      interpret("DiscreteConvolve[KroneckerDelta[n - 2], a^n, n, m]").unwrap(),
      "a^(-2 + m)"
    );
    assert_eq!(
      interpret("DiscreteConvolve[1, KroneckerDelta[n - 2], n, m]").unwrap(),
      "1"
    );
  }

  // An opaque undefined function of n is left symbolic, matching wolframscript.
  #[test]
  fn kronecker_delta_opaque_unevaluated() {
    clear_state();
    assert_eq!(
      interpret("DiscreteConvolve[g[n], KroneckerDelta[n], n, m]").unwrap(),
      "DiscreteConvolve[g[n], KroneckerDelta[n], n, m]"
    );
  }
}

mod list_fourier_sequence_transform {
  use super::*;

  #[test]
  fn single_element() {
    clear_state();
    assert_eq!(
      interpret("ListFourierSequenceTransform[{5}, omega]").unwrap(),
      "5"
    );
  }

  #[test]
  fn two_elements_symbolic() {
    clear_state();
    let result =
      interpret("ListFourierSequenceTransform[{1, 1}, omega]").unwrap();
    assert!(
      result.contains('E'),
      "expected expression with E, got {result}"
    );
  }

  #[test]
  fn numeric_at_zero() {
    clear_state();
    // At omega = 0, E^0 = 1 for all terms, so sum = 1 + 2 + 3 = 6
    assert_eq!(
      interpret("ListFourierSequenceTransform[{1, 2, 3}, 0]").unwrap(),
      "6"
    );
  }

  #[test]
  fn numeric_at_pi() {
    clear_state();
    // {1.0, 1.0} at omega = 1.0: should produce a complex number
    let result =
      interpret("ListFourierSequenceTransform[{1.0, 1.0}, 1.0]").unwrap();
    // Should contain numeric values (possibly complex)
    assert!(
      !result.contains("ListFourierSequenceTransform"),
      "expected evaluated, got {result}"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    clear_state();
    assert_eq!(
      interpret("ListFourierSequenceTransform[x]").unwrap(),
      "ListFourierSequenceTransform[x]"
    );
  }

  #[test]
  fn empty_list() {
    clear_state();
    assert_eq!(
      interpret("ListFourierSequenceTransform[{}, omega]").unwrap(),
      "{}"
    );
  }
}

mod frenet_serret_system {
  use super::*;

  #[test]
  fn two_d_parabola() {
    assert_eq!(
      interpret("FrenetSerretSystem[{t, t^2}, t]").unwrap(),
      "{{2/(1 + 4*t^2)^(3/2)}, {{1/Sqrt[1 + 4*t^2], (2*t)/Sqrt[1 + 4*t^2]}, {(-2*t)/Sqrt[1 + 4*t^2], 1/Sqrt[1 + 4*t^2]}}}"
    );
  }

  #[test]
  fn two_d_straight_line() {
    assert_eq!(
      interpret("FrenetSerretSystem[{3*t, 4*t}, t]").unwrap(),
      "{{0}, {{3/5, 4/5}, {-4/5, 3/5}}}"
    );
  }

  #[test]
  fn two_d_symbolic_coefficients() {
    assert_eq!(
      interpret("FrenetSerretSystem[{a*t, b*t^2}, t]").unwrap(),
      "{{(2*a*b)/(a^2 + 4*b^2*t^2)^(3/2)}, {{a/Sqrt[a^2 + 4*b^2*t^2], (2*b*t)/Sqrt[a^2 + 4*b^2*t^2]}, {(-2*b*t)/Sqrt[a^2 + 4*b^2*t^2], a/Sqrt[a^2 + 4*b^2*t^2]}}}"
    );
  }

  #[test]
  fn three_d_straight_line() {
    assert_eq!(
      interpret("FrenetSerretSystem[{t, 0, 0}, t]").unwrap(),
      "{{0, 0}, {{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}}"
    );
  }

  #[test]
  fn three_d_polynomial_curve() {
    // FrenetSerretSystem[{t, t^2, t^3}, t] - tangent vector
    let result =
      interpret("FrenetSerretSystem[{t, t^2, t^3}, t][[2, 1]]").unwrap();
    assert_eq!(
      result,
      "{1/Sqrt[1 + 4*t^2 + 9*t^4], (2*t)/Sqrt[1 + 4*t^2 + 9*t^4], (3*t^2)/Sqrt[1 + 4*t^2 + 9*t^4]}"
    );
  }

  #[test]
  fn scalar_function_treated_as_2d_curve() {
    // FrenetSerretSystem[f[t], t] treats scalar f[t] as the 2D curve {t, f[t]}
    assert_eq!(
      interpret("FrenetSerretSystem[f[t], t]").unwrap(),
      "{{Derivative[2][f][t]/(1 + Derivative[1][f][t]^2)^(3/2)}, {{1/Sqrt[1 + Derivative[1][f][t]^2], Derivative[1][f][t]/Sqrt[1 + Derivative[1][f][t]^2]}, {-(Derivative[1][f][t]/Sqrt[1 + Derivative[1][f][t]^2]), 1/Sqrt[1 + Derivative[1][f][t]^2]}}}"
    );
  }
}

mod asymptotic_integrate {
  use super::*;

  #[test]
  fn exp_neg_x_squared() {
    assert_eq!(
      interpret("AsymptoticIntegrate[Exp[-x^2], x, {x, 0, 5}]").unwrap(),
      "x - x^3/3 + x^5/10"
    );
  }

  #[test]
  fn reciprocal_1_plus_x() {
    assert_eq!(
      interpret("AsymptoticIntegrate[1/(1+x), x, {x, 0, 4}]").unwrap(),
      "x - x^2/2 + x^3/3 - x^4/4"
    );
  }

  #[test]
  fn sin_x() {
    assert_eq!(
      interpret("AsymptoticIntegrate[Sin[x], x, {x, 0, 6}]").unwrap(),
      "-1 + x^2/2 - x^4/24 + x^6/720"
    );
  }

  #[test]
  fn cos_x() {
    assert_eq!(
      interpret("AsymptoticIntegrate[Cos[x], x, {x, 0, 5}]").unwrap(),
      "x - x^3/6 + x^5/120"
    );
  }

  #[test]
  fn polynomial() {
    assert_eq!(
      interpret("AsymptoticIntegrate[x^2, x, {x, 0, 4}]").unwrap(),
      "x^3/3"
    );
  }

  #[test]
  fn definite_sin_tx_rule_form() {
    assert_eq!(
      interpret("AsymptoticIntegrate[Sin[t*x], {t, 0, 1}, x -> 0]").unwrap(),
      "x/2"
    );
  }

  #[test]
  fn definite_sin_tx_list_form_order_5() {
    assert_eq!(
      interpret("AsymptoticIntegrate[Sin[t*x], {t, 0, 1}, {x, 0, 5}]").unwrap(),
      "x/2 - x^3/24 + x^5/720"
    );
  }

  #[test]
  fn definite_cos_tx_rule_form() {
    assert_eq!(
      interpret("AsymptoticIntegrate[Cos[t*x], {t, 0, 1}, x -> 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn definite_exp_tx_rule_form() {
    assert_eq!(
      interpret("AsymptoticIntegrate[Exp[t*x], {t, 0, 1}, x -> 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn definite_rule_form_expands_until_nonzero() {
    assert_eq!(
      interpret("AsymptoticIntegrate[(t*x)^2, {t, 0, 1}, x -> 0]").unwrap(),
      "x^2/3"
    );
    assert_eq!(
      interpret("AsymptoticIntegrate[(t*x)^5, {t, 0, 1}, x -> 0]").unwrap(),
      "x^5/6"
    );
  }
}

mod max_limit {
  use super::*;

  #[test]
  fn one_over_x_at_zero() {
    assert_eq!(interpret("MaxLimit[1/x, x -> 0]").unwrap(), "Infinity");
  }

  #[test]
  fn sin_x_over_x_at_zero() {
    assert_eq!(interpret("MaxLimit[Sin[x]/x, x -> 0]").unwrap(), "1");
  }

  #[test]
  fn polynomial_at_infinity() {
    assert_eq!(
      interpret("MaxLimit[x^2, x -> Infinity]").unwrap(),
      "Infinity"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("MaxLimit[5, x -> 0]").unwrap(), "5");
  }

  // A bounded trig oscillation has limit superior 1 once its argument grows
  // without bound.
  #[test]
  fn bounded_trig_oscillation() {
    assert_eq!(interpret("MaxLimit[Sin[x], x -> Infinity]").unwrap(), "1");
    assert_eq!(interpret("MaxLimit[Cos[x], x -> Infinity]").unwrap(), "1");
    assert_eq!(interpret("MaxLimit[Sin[x^2], x -> Infinity]").unwrap(), "1");
    assert_eq!(interpret("MaxLimit[Cos[3 x], x -> Infinity]").unwrap(), "1");
    // Argument diverges as x -> 0.
    assert_eq!(interpret("MaxLimit[Sin[1/x], x -> 0]").unwrap(), "1");
    // At a finite point the trig is continuous, so the ordinary value stands.
    assert_eq!(interpret("MaxLimit[Sin[x], x -> 5]").unwrap(), "Sin[5]");
  }
}

mod min_limit {
  use super::*;

  #[test]
  fn one_over_x_at_zero() {
    assert_eq!(interpret("MinLimit[1/x, x -> 0]").unwrap(), "-Infinity");
  }

  #[test]
  fn sin_x_over_x_at_zero() {
    assert_eq!(interpret("MinLimit[Sin[x]/x, x -> 0]").unwrap(), "1");
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("MinLimit[5, x -> 0]").unwrap(), "5");
  }

  // A bounded trig oscillation has limit inferior -1.
  #[test]
  fn bounded_trig_oscillation() {
    assert_eq!(interpret("MinLimit[Sin[x], x -> Infinity]").unwrap(), "-1");
    assert_eq!(interpret("MinLimit[Cos[x], x -> Infinity]").unwrap(), "-1");
    assert_eq!(interpret("MinLimit[Sin[1/x], x -> 0]").unwrap(), "-1");
  }
}

mod arc_curvature {
  use super::*;

  #[test]
  fn parabola_2d() {
    assert_eq!(
      interpret("ArcCurvature[{t, t^2}, t]").unwrap(),
      "2/(1 + 4*t^2)^(3/2)"
    );
  }

  // The result is simplified, so the unit circle has constant curvature 1
  // (1/Sqrt[Cos^2 + Sin^2] reduces to 1).
  #[test]
  fn unit_circle() {
    assert_eq!(interpret("ArcCurvature[{Cos[t], Sin[t]}, t]").unwrap(), "1");
  }

  // A circle of radius 2 has curvature 1/2.
  #[test]
  fn radius_two_circle() {
    assert_eq!(
      interpret("ArcCurvature[{2*Cos[t], 2*Sin[t]}, t]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn scalar_function() {
    // Scalar treated as {t, f(t)}
    assert_eq!(
      interpret("ArcCurvature[t^2, t]").unwrap(),
      "2/(1 + 4*t^2)^(3/2)"
    );
  }

  #[test]
  fn space_curve_3d() {
    // 3D curve {t, t^2, t^3}
    assert_eq!(
      interpret("ArcCurvature[{t, t^2, t^3}, t]").unwrap(),
      "(2*Sqrt[1 + 9*t^2 + 9*t^4])/(1 + 4*t^2 + 9*t^4)^(3/2)"
    );
  }

  #[test]
  fn straight_line() {
    assert_eq!(interpret("ArcCurvature[{t, 2*t}, t]").unwrap(), "0");
  }

  #[test]
  fn straight_line_3d() {
    assert_eq!(interpret("ArcCurvature[{t, 0, 0}, t]").unwrap(), "0");
  }
}

mod difference_delta {
  use super::*;

  #[test]
  fn constant() {
    assert_eq!(interpret("DifferenceDelta[5, x]").unwrap(), "0");
  }

  #[test]
  fn linear() {
    assert_eq!(interpret("DifferenceDelta[x, x]").unwrap(), "1");
  }

  #[test]
  fn linear_with_coefficients() {
    assert_eq!(interpret("DifferenceDelta[a*x + b, x]").unwrap(), "a");
  }

  #[test]
  fn quadratic() {
    assert_eq!(interpret("DifferenceDelta[x^2, x]").unwrap(), "1 + 2*x");
  }

  #[test]
  fn cubic() {
    assert_eq!(
      interpret("DifferenceDelta[x^3, x]").unwrap(),
      "1 + 3*x + 3*x^2"
    );
  }

  #[test]
  fn symbolic_function() {
    assert_eq!(
      interpret("DifferenceDelta[f[x], x]").unwrap(),
      "-f[x] + f[1 + x]"
    );
  }

  #[test]
  fn sin_function() {
    // Wolfram simplifies to: 2*Sin[1/2]*Sin[(1 + Pi)/2 + x]
    assert_eq!(
      interpret("DifferenceDelta[Sin[x], x]").unwrap(),
      "2*Sin[1/2]*Sin[(1 + Pi)/2 + x]"
    );
  }

  #[test]
  fn cos_function() {
    // Δ Cos[x] = Cos[1 + x] - Cos[x] = -2 Sin[1/2] Sin[1/2 + x], which
    // wolframscript writes with the quarter turn kept inside the argument as
    // one fraction — `2 Sin[d] Cos[(2d + Pi)/2 + f]` — the same shape as the
    // sine's difference.
    assert_eq!(
      interpret("DifferenceDelta[Cos[x], x]").unwrap(),
      "2*Cos[(1 + Pi)/2 + x]*Sin[1/2]"
    );
    assert_eq!(
      interpret("DifferenceDelta[Cos[2*x], x]").unwrap(),
      "2*Cos[(2 + Pi)/2 + 2*x]*Sin[1]"
    );
    assert_eq!(
      interpret("DifferenceDelta[Cos[x], {x, 1, h}]").unwrap(),
      "2*Cos[(h + Pi)/2 + x]*Sin[h/2]"
    );
    // The derivative it defines has to come out as the cosine's own.
    assert_eq!(
      interpret("Limit[DifferenceQuotient[Cos[x], {x, h}], h -> 0]").unwrap(),
      "-Sin[x]"
    );
  }

  #[test]
  fn second_order() {
    // Second-order difference of x^2 is 2
    assert_eq!(interpret("DifferenceDelta[x^2, {x, 2}]").unwrap(), "2");
  }

  #[test]
  fn zeroth_order() {
    // Zeroth-order difference returns the expression itself
    assert_eq!(interpret("DifferenceDelta[x^2, {x, 0}]").unwrap(), "x^2");
  }

  #[test]
  fn with_step_h() {
    // DifferenceDelta[x^2, {x, 1, h}] = (x+h)^2 - x^2 = 2*h*x + h^2
    assert_eq!(
      interpret("DifferenceDelta[x^2, {x, 1, h}]").unwrap(),
      "h^2 + 2*h*x"
    );
  }

  #[test]
  fn independent_variable() {
    // DifferenceDelta[y, x] = 0 when y doesn't depend on x
    assert_eq!(interpret("DifferenceDelta[y, x]").unwrap(), "0");
  }

  #[test]
  fn exponential() {
    // DifferenceDelta[2^x, x] = 2^(x+1) - 2^x = 2^x
    assert_eq!(interpret("DifferenceDelta[2^x, x]").unwrap(), "2^x");
  }

  // A rational summand combines over a common denominator (previously it left
  // an unsimplified `1 + 2 + 2 n` and two separate fractions).
  #[test]
  fn rational_summand() {
    assert_eq!(
      interpret("DifferenceDelta[1/(2 n + 1), n]").unwrap(),
      "-2/((1 + 2*n)*(3 + 2*n))"
    );
    assert_eq!(
      interpret("DifferenceDelta[1/n, n]").unwrap(),
      "-(1/(n*(1 + n)))"
    );
  }

  // With a numeric step wolframscript factors the polynomial result.
  #[test]
  fn numeric_step_factors() {
    assert_eq!(
      interpret("DifferenceDelta[x^2 + x, x]").unwrap(),
      "2*(1 + x)"
    );
    assert_eq!(
      interpret("DifferenceDelta[a*x^2, x]").unwrap(),
      "a*(1 + 2*x)"
    );
    assert_eq!(
      interpret("DifferenceDelta[x^2, {x, 1, 2}]").unwrap(),
      "4*(1 + x)"
    );
    // An irreducible polynomial stays expanded.
    assert_eq!(
      interpret("DifferenceDelta[x^3, x]").unwrap(),
      "1 + 3*x + 3*x^2"
    );
  }

  // With a symbolic step the result stays expanded (not factored).
  #[test]
  fn symbolic_step_stays_expanded() {
    assert_eq!(
      interpret("DifferenceDelta[x^2 + x, {x, 1, h}]").unwrap(),
      "h + h^2 + 2*h*x"
    );
  }
}

// DiscreteShift[f, n] substitutes n -> n+1 in f.
mod discrete_shift {
  use super::*;

  #[test]
  fn basic_shift() {
    assert_eq!(interpret("DiscreteShift[f[n], n]").unwrap(), "f[1 + n]");
    assert_eq!(interpret("DiscreteShift[n^2, n]").unwrap(), "(1 + n)^2");
    assert_eq!(interpret("DiscreteShift[2^n, n]").unwrap(), "2^(1 + n)");
    assert_eq!(interpret("DiscreteShift[Sin[n], n]").unwrap(), "Sin[1 + n]");
    assert_eq!(interpret("DiscreteShift[c, n]").unwrap(), "c");
  }

  #[test]
  fn polynomial_sum_is_expanded() {
    // A top-level Plus result is expanded; a single power/product is not.
    assert_eq!(
      interpret("DiscreteShift[n^2 + 3 n + 1, n]").unwrap(),
      "5 + 5*n + n^2"
    );
    assert_eq!(
      interpret("DiscreteShift[a n^2 + b n, n]").unwrap(),
      "a + b + 2*a*n + b*n + a*n^2"
    );
    assert_eq!(interpret("DiscreteShift[2 n^2, n]").unwrap(), "2*(1 + n)^2");
  }

  #[test]
  fn shift_by_k_via_list() {
    assert_eq!(
      interpret("DiscreteShift[f[n], {n, 2}]").unwrap(),
      "f[2 + n]"
    );
    assert_eq!(
      interpret("DiscreteShift[f[n], {n, -1}]").unwrap(),
      "f[-1 + n]"
    );
  }

  #[test]
  fn multiple_variables() {
    assert_eq!(
      interpret("DiscreteShift[f[n, m], n, m]").unwrap(),
      "f[1 + n, 1 + m]"
    );
  }

  #[test]
  fn threads_over_lists() {
    assert_eq!(
      interpret("DiscreteShift[{n, n^2}, n]").unwrap(),
      "{1 + n, (1 + n)^2}"
    );
  }

  #[test]
  fn one_argument_is_identity() {
    assert_eq!(interpret("DiscreteShift[f[n]]").unwrap(), "f[n]");
  }

  #[test]
  fn non_variable_specifier_emits_ivar() {
    let r = interpret_with_stdout("DiscreteShift[f[n], n, 2]").unwrap();
    assert_eq!(r.result, "DiscreteShift[f[n], n, 2]");
    assert!(
      r.warnings
        .iter()
        .any(|w| w.contains("General::ivar: 2 is not a valid variable.")),
      "expected ivar message, got: {:?}",
      r.warnings
    );
  }

  // A rational summand is combined over a common denominator for an integer
  // shift (previously it left the unsimplified `(1 + 2 (1 + n))^-1`).
  #[test]
  fn rational_summand_integer_shift() {
    assert_eq!(
      interpret("DiscreteShift[1/(2 n + 1), n]").unwrap(),
      "(3 + 2*n)^(-1)"
    );
    assert_eq!(
      interpret("DiscreteShift[1/(3 n - 2), n]").unwrap(),
      "(1 + 3*n)^(-1)"
    );
    assert_eq!(
      interpret("DiscreteShift[1/(2 n + 1), {n, 2}]").unwrap(),
      "(5 + 2*n)^(-1)"
    );
    // A mixed sum combines into a single fraction.
    assert_eq!(
      interpret("DiscreteShift[1/(2 n + 1) + n, n]").unwrap(),
      "(4 + 5*n + 2*n^2)/(3 + 2*n)"
    );
  }

  // A symbolic shift is left unfolded (folding would distribute the step).
  #[test]
  fn rational_symbolic_shift_unfolded() {
    assert_eq!(
      interpret("DiscreteShift[1/(2 n + 1), {n, k}]").unwrap(),
      "(1 + 2*(k + n))^(-1)"
    );
  }
}

// DiscreteRatio[f, n] = f(n+1)/f(n), the multiplicative analog of
// DifferenceDelta.
mod discrete_ratio {
  use super::*;

  #[test]
  fn basic_ratio() {
    assert_eq!(
      interpret("DiscreteRatio[f[n], n]").unwrap(),
      "f[1 + n]/f[n]"
    );
    assert_eq!(interpret("DiscreteRatio[2^n, n]").unwrap(), "2");
    assert_eq!(interpret("DiscreteRatio[a^n, n]").unwrap(), "a");
    assert_eq!(interpret("DiscreteRatio[n^2, n]").unwrap(), "(1 + n)^2/n^2");
    assert_eq!(interpret("DiscreteRatio[c, n]").unwrap(), "1");
  }

  #[test]
  fn cancels_common_factors() {
    assert_eq!(interpret("DiscreteRatio[n^2 + n, n]").unwrap(), "(2 + n)/n");
    assert_eq!(interpret("DiscreteRatio[2^n 3^n, n]").unwrap(), "6");
  }

  #[test]
  fn higher_order_applies_operator_repeatedly() {
    assert_eq!(
      interpret("DiscreteRatio[f[n], {n, 3}]").unwrap(),
      "(f[1 + n]^3*f[3 + n])/(f[n]*f[2 + n]^3)"
    );
    // Order 0 is the identity.
    assert_eq!(interpret("DiscreteRatio[f[n], {n, 0}]").unwrap(), "f[n]");
  }

  #[test]
  fn step_in_three_element_spec() {
    assert_eq!(
      interpret("DiscreteRatio[f[n], {n, 1, 2}]").unwrap(),
      "f[2 + n]/f[n]"
    );
    assert_eq!(
      interpret("DiscreteRatio[f[n], {n, 2, 3}]").unwrap(),
      "(f[n]*f[6 + n])/f[3 + n]^2"
    );
  }

  #[test]
  fn multiple_variables_compose() {
    assert_eq!(
      interpret("DiscreteRatio[f[n, m], n, m]").unwrap(),
      "(f[n, m]*f[1 + n, 1 + m])/(f[n, 1 + m]*f[1 + n, m])"
    );
    assert_eq!(interpret("DiscreteRatio[n m, n, m]").unwrap(), "1");
  }

  #[test]
  fn threads_over_lists() {
    assert_eq!(
      interpret("DiscreteRatio[{n, n^2}, n]").unwrap(),
      "{(1 + n)/n, (1 + n)^2/n^2}"
    );
  }

  #[test]
  fn one_argument_is_identity() {
    assert_eq!(interpret("DiscreteRatio[f[n]]").unwrap(), "f[n]");
  }

  #[test]
  fn symbolic_order_stays_unevaluated() {
    assert_eq!(
      interpret("DiscreteRatio[f[n], {n, h}]").unwrap(),
      "DiscreteRatio[f[n], {n, h}]"
    );
  }

  #[test]
  fn bad_specifiers_emit_messages() {
    // A non-variable specifier is an ivar error.
    let r = interpret_with_stdout("DiscreteRatio[f[n], n, 2]").unwrap();
    assert_eq!(r.result, "DiscreteRatio[f[n], n, 2]");
    assert!(
      r.warnings
        .iter()
        .any(|w| w.contains("General::ivar: 2 is not a valid variable.")),
      "expected ivar message, got: {:?}",
      r.warnings
    );
    // A non-integer order is a dvar error.
    let r = interpret_with_stdout("DiscreteRatio[f[n], {n, 2.5}]").unwrap();
    assert_eq!(r.result, "DiscreteRatio[f[n], {n, 2.5}]");
    assert!(
      r.warnings.iter().any(|w| w.contains(
        "DiscreteRatio::dvar: Ratio specifier {n, 2.5} does not have the form"
      )),
      "expected dvar message, got: {:?}",
      r.warnings
    );
  }

  // Factorial/Gamma/Pochhammer ratios collapse the same way wolframscript
  // reduces them: (n+1)!/n! = n+1, Gamma[n+1]/Gamma[n] = n, etc. Cancel alone
  // leaves these untouched, so DiscreteRatio applies FunctionExpand.
  #[test]
  fn factorial_and_gamma_ratios_reduce() {
    assert_eq!(interpret("DiscreteRatio[n!, n]").unwrap(), "1 + n");
    assert_eq!(interpret("DiscreteRatio[Gamma[n], n]").unwrap(), "n");
    assert_eq!(
      interpret("DiscreteRatio[Pochhammer[n, 3], n]").unwrap(),
      "(3 + n)/n"
    );
    assert_eq!(
      interpret("DiscreteRatio[Binomial[n, 2], n]").unwrap(),
      "(1 + n)/(-1 + n)"
    );
  }
}

mod difference_quotient {
  use super::*;

  #[test]
  fn bare_var_unevaluated() {
    // DifferenceQuotient[f, x] returns unevaluated (only {x, h} form evaluates)
    assert_eq!(
      interpret("DifferenceQuotient[x, x]").unwrap(),
      "DifferenceQuotient[x, x]"
    );
  }

  #[test]
  fn bare_var_quadratic_unevaluated() {
    assert_eq!(
      interpret("DifferenceQuotient[x^2, x]").unwrap(),
      "DifferenceQuotient[x^2, x]"
    );
  }

  #[test]
  fn quadratic_step_h() {
    assert_eq!(
      interpret("DifferenceQuotient[x^2, {x, h}]").unwrap(),
      "h + 2*x"
    );
  }

  #[test]
  fn cubic_step_h() {
    assert_eq!(
      interpret("DifferenceQuotient[x^3, {x, h}]").unwrap(),
      "h^2 + 3*h*x + 3*x^2"
    );
  }

  #[test]
  fn symbolic_function() {
    assert_eq!(
      interpret("DifferenceQuotient[f[x], {x, h}]").unwrap(),
      "(-f[x] + f[h + x])/h"
    );
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("DifferenceQuotient[5, {x, h}]").unwrap(), "0");
  }
}

mod differentiate_integrate_leibniz {
  use woxi::interpret;

  // Full Leibniz rule: the integrand depends on `x` through both the
  // bound variable `u` and `x` directly. Result must include the inner
  // partial-derivative integral.
  #[test]
  fn integrand_depends_on_x_and_variable_bounds() {
    assert_eq!(
      interpret("D[Integrate[f[u, x], {u, a[x], b[x]}], x]").unwrap(),
      "Integrate[Derivative[0, 1][f][u, x], {u, a[x], b[x]}] - \
        f[a[x], x]*Derivative[1][a][x] + f[b[x], x]*Derivative[1][b][x]"
    );
  }

  // Integrand independent of `x`: only the boundary terms should appear,
  // so no `Integrate[...]` term is emitted.
  #[test]
  fn integrand_independent_of_x_with_variable_bounds() {
    assert_eq!(
      interpret("D[Integrate[f[u], {u, a[x], b[x]}], x]").unwrap(),
      "-(f[a[x]]*Derivative[1][a][x]) + f[b[x]]*Derivative[1][b][x]"
    );
  }

  // Fundamental theorem for an indefinite integral: differentiating by the
  // integration variable recovers the integrand.
  #[test]
  fn indefinite_integral_fundamental_theorem() {
    assert_eq!(interpret("D[Integrate[f[x], x], x]").unwrap(), "f[x]");
    assert_eq!(
      interpret("D[Integrate[Sin[x^2], x], x]").unwrap(),
      "Sin[x^2]"
    );
    assert_eq!(
      interpret("D[Integrate[h[x] + f[x], x], x]").unwrap(),
      "f[x] + h[x]"
    );
  }

  // Differentiating an indefinite integral by another variable applies
  // differentiation under the integral sign.
  #[test]
  fn indefinite_integral_other_variable() {
    assert_eq!(interpret("D[Integrate[f[x], x], y]").unwrap(), "0");
    assert_eq!(interpret("D[Integrate[x y, x], y]").unwrap(), "x^2/2");
    assert_eq!(
      interpret("D[Integrate[f[x, y], x], y]").unwrap(),
      "Integrate[Derivative[0, 1][f][x, y], x]"
    );
  }
}

mod integrate_piecewise_definite {
  use super::*;

  // A bounded-support condition (0 < x < 1) integrates only over its support,
  // not to 0 (which the discontinuous default-0 antiderivative produced).
  #[test]
  fn bounded_support() {
    assert_eq!(
      interpret("Integrate[Piecewise[{{1, 0 < x < 1}}], {x, -2, 2}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Integrate[Piecewise[{{x, 0 < x < 1}}], {x, 0, 1}]").unwrap(),
      "1/2"
    );
    // Integration bounds narrower than the support.
    assert_eq!(
      interpret("Integrate[Piecewise[{{1, 0 < x < 10}}], {x, 2, 5}]").unwrap(),
      "3"
    );
  }

  #[test]
  fn multiple_bounded_pieces() {
    assert_eq!(
      interpret(
        "Integrate[Piecewise[{{1, 0 < x < 1}, {2, 1 < x < 2}}], {x, 0, 2}]"
      )
      .unwrap(),
      "3"
    );
  }

  #[test]
  fn abs_condition() {
    assert_eq!(
      interpret("Integrate[Piecewise[{{1, Abs[x] < 1}}], {x, -2, 2}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn nonconstant_value_over_support() {
    assert_eq!(
      interpret("Integrate[Piecewise[{{x^2, 0 < x < 2}}], {x, -5, 5}]")
        .unwrap(),
      "8/3"
    );
    assert_eq!(
      interpret("Integrate[Piecewise[{{Sin[x], 0 < x < Pi}}], {x, -1, 4}]")
        .unwrap(),
      "2"
    );
  }

  // One-sided and complementary conditions keep working.
  #[test]
  fn one_sided_and_complementary() {
    assert_eq!(
      interpret("Integrate[Piecewise[{{1, x > 0}}], {x, -1, 3}]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret(
        "Integrate[Piecewise[{{x^2, x < 0}, {x, x >= 0}}], {x, -1, 1}]"
      )
      .unwrap(),
      "5/6"
    );
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn integrate_1() {
    assert_case(
      r#"Piecewise[{{0, x <= 0}}, 1]; Integrate[Piecewise[{{1, x <= 0}, {-1, x > 0}}], x]"#,
      r#"Piecewise[{{x, x <= 0}}, -x]"#,
    );
  }
  #[test]
  fn integrate_2() {
    assert_case(
      r#"Piecewise[{{0, x <= 0}}, 1]; Integrate[Piecewise[{{1, x <= 0}, {-1, x > 0}}], x]; Integrate[Piecewise[{{1, x <= 0}, {-1, x > 0}}], {x, -1, 2}]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn piecewise() {
    assert_case(
      r#"Piecewise[{{0, x <= 0}}, 1]; Integrate[Piecewise[{{1, x <= 0}, {-1, x > 0}}], x]; Integrate[Piecewise[{{1, x <= 0}, {-1, x > 0}}], {x, -1, 2}]; Piecewise[{{1, False}}]"#,
      r#"0"#,
    );
  }
  #[test]
  fn d_1() {
    assert_case(
      r#"RealAbs[-3.]; RealAbs[2. + 3. I]; D[RealAbs[x ^ 2], x]"#,
      r#"(2*x^3)/RealAbs[x^2]"#,
    );
  }
  #[test]
  fn integrate_real_abs() {
    // Indefinite integral of RealAbs[x]: (x*RealAbs[x])/2.
    // Derivative recovers RealAbs[x] (away from 0):
    //   d/dx[(x*RealAbs[x])/2] = (RealAbs[x] + x^2/RealAbs[x]) / 2
    //                         = RealAbs[x]  (since x^2/RealAbs[x] = RealAbs[x]).
    assert_case(r#"Integrate[RealAbs[x], x]"#, r#"(x*RealAbs[x])/2"#);
  }
  #[test]
  fn integrate_real_abs_definite() {
    // Definite integrals over symmetric and one-sided intervals.
    assert_case(r#"Integrate[RealAbs[x], {x, 0, 2}]"#, r#"2"#);
    assert_case(r#"Integrate[RealAbs[x], {x, -1, 1}]"#, r#"1"#);
  }
  #[test]
  fn d_2() {
    assert_case(
      r#"RealSign[-3.]; RealSign[2. + 3. I]; D[RealSign[x^2],x]"#,
      r#"2*x*Piecewise[{{0, x^2 != 0}}, Indeterminate]"#,
    );
  }
  #[test]
  fn integrate_3() {
    assert_case(
      r#"RealSign[-3.]; RealSign[2. + 3. I]; D[RealSign[x^2],x]; Integrate[RealSign[u],{u,0,x}]"#,
      r#"Abs[x]"#,
    );
  }
  #[test]
  fn integrate_4() {
    assert_case(
      r#"ArcCos[1]; ArcCos[0]; Integrate[ArcCos[x], {x, -1, 1}]"#,
      r#"Pi"#,
    );
  }
  #[test]
  fn coefficient_list_1() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]; CoefficientList[(x + y)^4, x]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, x]; CoefficientList[(x + 2)/(y - 3) + x/(y - 2), x]; CoefficientList[(x + y)^3, z]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, {x, y}]; CoefficientList[(x - 2 y + 3 z)^3, {x, y, z}]; CoefficientList[Series[Log[1-x], {x, 0, 9}], x]"#,
      r#"{0, -1, -1 / 2, -1 / 3, -1 / 4, -1 / 5, -1 / 6, -1 / 7, -1 / 8, -1 / 9}"#,
    );
  }
  #[test]
  fn coefficient_list_2() {
    assert_case(
      r#"CoefficientList[(x + 3)^5, x]; CoefficientList[(x + y)^4, x]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, x]; CoefficientList[(x + 2)/(y - 3) + x/(y - 2), x]; CoefficientList[(x + y)^3, z]; CoefficientList[a x^2 + b y^3 + c x + d y + 5, {x, y}]; CoefficientList[(x - 2 y + 3 z)^3, {x, y, z}]; CoefficientList[Series[Log[1-x], {x, 0, 9}], x]; CoefficientList[Series[2x, {x, 0, 9}], x]"#,
      r#"{0, 2}"#,
    );
  }
  #[test]
  fn limit_1() {
    assert_case(
      r#"Precision[1]; 1 / Infinity; Infinity + 100; Sum[1/x^2, {x, 1, Infinity}]; Limit[1/x, x->0]"#,
      r#"Indeterminate"#,
    );
  }
  #[test]
  fn full_form() {
    assert_case(
      r#"Precision[1]; 1 / Infinity; Infinity + 100; Sum[1/x^2, {x, 1, Infinity}]; Limit[1/x, x->0]; FullForm[Infinity]"#,
      r#"FullForm[Infinity]"#,
    );
  }
  #[test]
  fn d_3() {
    assert_case(r#"D[x^3 + x^2, x]"#, r#"2*x + 3*x^2"#);
  }
  #[test]
  fn d_4() {
    assert_case(r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]"#, r#"2 + 6*x"#);
  }
  #[test]
  fn d_5() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]"#,
      r#"-(Cos[Cos[x]]*Sin[x])"#,
    );
  }
  #[test]
  fn d_6() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]"#,
      r#"-Sin[x]"#,
    );
  }
  #[test]
  fn d_7() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]"#,
      r#"-Cos[t]"#,
    );
  }
  #[test]
  fn d_8() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]"#,
      r#"0"#,
    );
  }
  #[test]
  fn d_9() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]"#,
      r#"1"#,
    );
  }
  #[test]
  fn d_10() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]; D[x + y, x]"#,
      r#"1"#,
    );
  }
  #[test]
  fn d_11() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]; D[x + y, x]; D[f[x], x]"#,
      r#"Derivative[1][f][x]"#,
    );
  }
  #[test]
  fn d_12() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]; D[x + y, x]; D[f[x], x]; D[f[x, x], x]"#,
      r#"Derivative[0, 1][f][x, x] + Derivative[1, 0][f][x, x]"#,
    );
  }
  #[test]
  fn d_13() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]; D[x + y, x]; D[f[x], x]; D[f[x, x], x]; D[f[x, x], x] // InputForm"#,
      r#"InputForm[Derivative[0, 1][f][x, x] + Derivative[1, 0][f][x, x]]"#,
    );
  }
  #[test]
  fn d_14() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]; D[x + y, x]; D[f[x], x]; D[f[x, x], x]; D[f[x, x], x] // InputForm; D[f[2x+1, 2y, x+y], x]"#,
      r#"Derivative[0, 0, 1][f][1 + 2*x, 2*y, x + y] + 2*Derivative[1, 0, 0][f][1 + 2*x, 2*y, x + y]"#,
    );
  }
  #[test]
  fn d_15() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]; D[x + y, x]; D[f[x], x]; D[f[x, x], x]; D[f[x, x], x] // InputForm; D[f[2x+1, 2y, x+y], x]; D[f[x^2, x, 2y], {x,2}, y] // Expand"#,
      r#"2*Derivative[0, 2, 1][f][x^2, x, 2*y] + 4*Derivative[1, 0, 1][f][x^2, x, 2*y] + 8*x*Derivative[1, 1, 1][f][x^2, x, 2*y] + 8*x^2*Derivative[2, 0, 1][f][x^2, x, 2*y]"#,
    );
  }
  #[test]
  fn d_16() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]; D[x + y, x]; D[f[x], x]; D[f[x, x], x]; D[f[x, x], x] // InputForm; D[f[2x+1, 2y, x+y], x]; D[f[x^2, x, 2y], {x,2}, y] // Expand; D[x ^ 3 * Cos[y], {{x, y}}]"#,
      r#"{3*x^2*Cos[y], -(x^3*Sin[y])}"#,
    );
  }
  #[test]
  fn d_17() {
    assert_case(
      r#"D[x^3 + x^2, x]; D[x^3 + x^2, {x, 2}]; D[Sin[Cos[x]], x]; D[Sin[x], {x, 2}]; D[Cos[t], {t, 2}]; D[y, x]; D[x, x]; D[x + y, x]; D[f[x], x]; D[f[x, x], x]; D[f[x, x], x] // InputForm; D[f[2x+1, 2y, x+y], x]; D[f[x^2, x, 2y], {x,2}, y] // Expand; D[x ^ 3 * Cos[y], {{x, y}}]; D[Sin[x] * Cos[y], {{x,y}, 2}]"#,
      r#"{{-(Cos[y]*Sin[x]), -(Cos[x]*Sin[y])}, {-(Cos[x]*Sin[y]), -(Cos[y]*Sin[x])}}"#,
    );
  }
  #[test]
  fn derivative_1() {
    assert_case(r#"Derivative[1][Sin]"#, r#"Cos[#1]&"#);
  }
  #[test]
  fn derivative_2() {
    assert_case(r#"Derivative[1][Sin]; Derivative[3][Sin]"#, r#"-Cos[#1]&"#);
  }
  #[test]
  fn derivative_3() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]"#,
      r#"3*(2*#1) & "#,
    );
  }
  #[test]
  fn expr_1() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]"#,
      r#"Cos[x]"#,
    );
  }
  #[test]
  fn anonymous_function() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''"#,
      r#"4*(3*#1^2) & "#,
    );
  }
  #[test]
  fn divide_1() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''; f'[x] // InputForm"#,
      r#"InputForm[Derivative[1][f][x]]"#,
    );
  }
  #[test]
  fn derivative_4() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''; f'[x] // InputForm; Derivative[1][#2 Sin[#1]+Cos[#2]&]"#,
      r#"#2*Cos[#1] & "#,
    );
  }
  #[test]
  fn derivative_5() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''; f'[x] // InputForm; Derivative[1][#2 Sin[#1]+Cos[#2]&]; Derivative[1,2][#2^3 Sin[#1]+Cos[#2]&]"#,
      r#"Cos[#1]*(3*(2*#2)) & "#,
    );
  }
  #[test]
  fn derivative_6() {
    // `Derivative[m1, …, mk][body &]` returns `0 &` whenever any
    // non-zero `mi` corresponds to a slot beyond the maximum slot index
    // referenced by the body. Here the body uses `#1` and `#2`; the
    // third index `1` differentiates with respect to the absent `#3`,
    // collapsing the whole derivative to `0 &`.
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''; f'[x] // InputForm; Derivative[1][#2 Sin[#1]+Cos[#2]&]; Derivative[1,2][#2^3 Sin[#1]+Cos[#2]&]; Derivative[1,2,1][#2^3 Sin[#1]+Cos[#2]&]"#,
      r#"0&"#,
    );
  }
  #[test]
  fn derivative_7() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''; f'[x] // InputForm; Derivative[1][#2 Sin[#1]+Cos[#2]&]; Derivative[1,2][#2^3 Sin[#1]+Cos[#2]&]; Derivative[1,2,1][#2^3 Sin[#1]+Cos[#2]&]; Derivative[0,0,0][a+b+c]"#,
      r#"a + b + c"#,
    );
  }
  #[test]
  fn expr_2() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''; f'[x] // InputForm; Derivative[1][#2 Sin[#1]+Cos[#2]&]; Derivative[1,2][#2^3 Sin[#1]+Cos[#2]&]; Derivative[1,2,1][#2^3 Sin[#1]+Cos[#2]&]; Derivative[0,0,0][a+b+c]; f[x_] := x ^ 2; f'[x]"#,
      r#"2*x"#,
    );
  }
  #[test]
  fn derivative_8() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''; f'[x] // InputForm; Derivative[1][#2 Sin[#1]+Cos[#2]&]; Derivative[1,2][#2^3 Sin[#1]+Cos[#2]&]; Derivative[1,2,1][#2^3 Sin[#1]+Cos[#2]&]; Derivative[0,0,0][a+b+c]; f[x_] := x ^ 2; f'[x]; Derivative[2, 1][h]"#,
      r#"Derivative[2, 1][h]"#,
    );
  }
  #[test]
  fn derivative_9() {
    assert_case(
      r#"Derivative[1][Sin]; Derivative[3][Sin]; Derivative[2][# ^ 3&]; Sin'[x]; (# ^ 4&)''; f'[x] // InputForm; Derivative[1][#2 Sin[#1]+Cos[#2]&]; Derivative[1,2][#2^3 Sin[#1]+Cos[#2]&]; Derivative[1,2,1][#2^3 Sin[#1]+Cos[#2]&]; Derivative[0,0,0][a+b+c]; f[x_] := x ^ 2; f'[x]; Derivative[2, 1][h]; Derivative[2, 0, 1, 0][h[g]]"#,
      r#"Derivative[2, 0, 1, 0][h[g]]"#,
    );
  }
  #[test]
  fn integrate_5() {
    assert_case(
      r#"Integrate[6 x ^ 2 + 3 x ^ 2 - 4 x + 10, x]"#,
      r#"10*x - 2*x^2 + 3*x^3"#,
    );
  }
  #[test]
  fn integrate_6() {
    assert_case(
      r#"Integrate[6 x ^ 2 + 3 x ^ 2 - 4 x + 10, x]; Integrate[Sin[x] ^ 5, x]"#,
      r#"(-5*Cos[x])/8 + (5*Cos[3*x])/48 - Cos[5*x]/80"#,
    );
  }
  #[test]
  fn integrate_7() {
    assert_case(
      r#"Integrate[6 x ^ 2 + 3 x ^ 2 - 4 x + 10, x]; Integrate[Sin[x] ^ 5, x]; Integrate[x ^ 2 + x, {x, 1, 3}]"#,
      r#"38 / 3"#,
    );
  }
  #[test]
  fn integrate_8() {
    assert_case(
      r#"Integrate[6 x ^ 2 + 3 x ^ 2 - 4 x + 10, x]; Integrate[Sin[x] ^ 5, x]; Integrate[x ^ 2 + x, {x, 1, 3}]; Integrate[Sin[x], {x, 0, Pi/2}]"#,
      r#"1"#,
    );
  }
  #[test]
  fn integrate_9() {
    assert_case(
      r#"Integrate[6 x ^ 2 + 3 x ^ 2 - 4 x + 10, x]; Integrate[Sin[x] ^ 5, x]; Integrate[x ^ 2 + x, {x, 1, 3}]; Integrate[Sin[x], {x, 0, Pi/2}]; Integrate[1 / (1 - 4 x + x^2), x]"#,
      r#"(Log[2 + Sqrt[3] - x] - Log[-2 + Sqrt[3] + x])/(2*Sqrt[3])"#,
    );
  }
  #[test]
  fn integrate_10() {
    assert_case(
      r#"Integrate[6 x ^ 2 + 3 x ^ 2 - 4 x + 10, x]; Integrate[Sin[x] ^ 5, x]; Integrate[x ^ 2 + x, {x, 1, 3}]; Integrate[Sin[x], {x, 0, Pi/2}]; Integrate[1 / (1 - 4 x + x^2), x]; Integrate[4 Sin[x] Cos[x], x]"#,
      r#"-2*Cos[x]^2"#,
    );
  }
  #[test]
  fn limit_2() {
    assert_case(r#"Limit[x, x->2]"#, r#"2"#);
  }
  #[test]
  fn limit_3() {
    assert_case(r#"Limit[x, x->2]; Limit[Sin[x] / x, x->0]"#, r#"1"#);
  }
  #[test]
  fn series_1() {
    assert_case(
      r#"Series[1/(1-x),{x,0,2}]"#,
      r#"SeriesData[x, 0, {1, 1, 1}, 0, 3, 1]"#,
    );
  }
  #[test]
  fn o() {
    assert_case(
      r#"Series[1/(1-x),{x,0,2}]; O[x] // FullForm"#,
      r#"FullForm[SeriesData[x, 0, {}, 1, 1, 1]]"#,
    );
  }
  #[test]
  fn integrate_11() {
    assert_case(
      r#"Integrate[1/(x^5 + 11 x + 1), {x, 1, 3}]"#,
      r#"-RootSum[1 + 11*#1 + #1^5 & , Log[1 - #1]/(11 + 5*#1^4) & ] + RootSum[1 + 11*#1 + #1^5 & , Log[3 - #1]/(11 + 5*#1^4) & ]"#,
    );
  }
  #[test]
  fn n() {
    assert_case(
      r#"Integrate[1/(x^5 + 11 x + 1), {x, 1, 3}]; N[%, 50]"#,
      r#"Out[0]"#,
    );
  }
  #[test]
  fn root_sum_1() {
    assert_case(
      r#"Integrate[1/(x^5 + 11 x + 1), {x, 1, 3}]; N[%, 50]; RootSum[#^5 - 11 # + 1 &, (#^2 - 1)/(#^3 - 2 # + c) &]; RootSum[#^5 - 3 # - 7 &, Sin] //N//Chop"#,
      r#"0.2921876302209532"#,
    );
  }
  #[test]
  fn root_sum_2() {
    assert_case(
      r#"Integrate[1/(x^5 + 11 x + 1), {x, 1, 3}]; N[%, 50]; RootSum[#^5 - 11 # + 1 &, (#^2 - 1)/(#^3 - 2 # + c) &]; RootSum[#^5 - 3 # - 7 &, Sin] //N//Chop; RootSum[1+#+#^2+#^3+#^4 &, Log[x + #] &]"#,
      r#"RootSum[1 + #1 + #1^2 + #1^3 + #1^4 & , Log[x + #1] & ]"#,
    );
  }
  #[test]
  fn divide_2() {
    assert_case(
      r#"Integrate[1/(x^5 + 11 x + 1), {x, 1, 3}]; N[%, 50]; RootSum[#^5 - 11 # + 1 &, (#^2 - 1)/(#^3 - 2 # + c) &]; RootSum[#^5 - 3 # - 7 &, Sin] //N//Chop; RootSum[1+#+#^2+#^3+#^4 &, Log[x + #] &]; %//Normal"#,
      r#"Out[0]"#,
    );
  }
  #[test]
  fn set_1() {
    assert_case(
      r#"series = Series[Exp[x^2], {x,0,2}]"#,
      r#"SeriesData[x, 0, {1, 0, 1}, 0, 3, 1]"#,
    );
  }
  #[test]
  fn divide_3() {
    assert_case(
      r#"series = Series[Exp[x^2], {x,0,2}]; series // FullForm"#,
      r#"FullForm[SeriesData[x, 0, {1, 0, 1}, 0, 3, 1]]"#,
    );
  }
  #[test]
  fn series_2() {
    assert_case(
      r#"Series[Exp[Sin[x]], {x, 0, 5}]"#,
      r#"SeriesData[x, 0, {1, 1, 1/2, 0, -1/8, -1/15}, 0, 6, 1]"#,
    );
  }
  #[test]
  fn series_coefficient_1() {
    assert_case(
      r#"Series[Exp[Sin[x]], {x, 0, 5}]; SeriesCoefficient[%, 4]"#,
      r#"SeriesCoefficient[Out[0], 4]"#,
    );
  }
  #[test]
  fn series_coefficient_2() {
    assert_case(
      r#"Series[Exp[Sin[x]], {x, 0, 5}]; SeriesCoefficient[%, 4]; SeriesCoefficient[Exp[Sin[x]], {x, 0, 4}]"#,
      r#"-1 / 8"#,
    );
  }
  #[test]
  fn series_coefficient_3() {
    assert_case(
      r#"Series[Exp[Sin[x]], {x, 0, 5}]; SeriesCoefficient[%, 4]; SeriesCoefficient[Exp[Sin[x]], {x, 0, 4}]; SeriesCoefficient[2x, {x, 0, 2}]"#,
      r#"0"#,
    );
  }
  #[test]
  fn series_coefficient_4() {
    assert_case(
      r#"Series[Exp[Sin[x]], {x, 0, 5}]; SeriesCoefficient[%, 4]; SeriesCoefficient[Exp[Sin[x]], {x, 0, 4}]; SeriesCoefficient[2x, {x, 0, 2}]; SeriesCoefficient[SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], 14/3]"#,
      r#"64"#,
    );
  }
  #[test]
  fn series_coefficient_5() {
    assert_case(
      r#"Series[Exp[Sin[x]], {x, 0, 5}]; SeriesCoefficient[%, 4]; SeriesCoefficient[Exp[Sin[x]], {x, 0, 4}]; SeriesCoefficient[2x, {x, 0, 2}]; SeriesCoefficient[SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], 14/3]; SeriesCoefficient[SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], 6/3]"#,
      r#"0"#,
    );
  }
  #[test]
  fn series_coefficient_6() {
    assert_case(
      r#"Series[Exp[Sin[x]], {x, 0, 5}]; SeriesCoefficient[%, 4]; SeriesCoefficient[Exp[Sin[x]], {x, 0, 4}]; SeriesCoefficient[2x, {x, 0, 2}]; SeriesCoefficient[SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], 14/3]; SeriesCoefficient[SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], 6/3]; SeriesCoefficient[SeriesData[x, c, Table[i^2, {i, 10}], 7, 17, 3], 17/3]"#,
      r#"Indeterminate"#,
    );
  }
  #[test]
  fn set_2() {
    assert_case(
      r#"series = Series[Cosh[x],{x,0,2}]"#,
      r#"SeriesData[x, 0, {1, 0, 1/2}, 0, 3, 1]"#,
    );
  }
  #[test]
  fn head_1() {
    assert_case(
      r#"series = Series[Cosh[x],{x,0,2}]; Head[series]"#,
      r#"SeriesData"#,
    );
  }
  #[test]
  fn divide_4() {
    assert_case(
      r#"series = Series[Cosh[x],{x,0,2}]; Head[series]; series // FullForm"#,
      r#"FullForm[SeriesData[x, 0, {1, 0, 1/2}, 0, 3, 1]]"#,
    );
  }
  #[test]
  fn plus_1() {
    assert_case(
      r#"series = Series[Cosh[x],{x,0,2}]; Head[series]; series // FullForm; series + Series[Sinh[x],{x, 0, 3}]"#,
      r#"SeriesData[x, 0, {1, 1, 1/2}, 0, 3, 1]"#,
    );
  }
  #[test]
  fn series_3() {
    assert_case(
      r#"series = Series[Cosh[x],{x,0,2}]; Head[series]; series // FullForm; series + Series[Sinh[x],{x, 0, 3}]; Series[f[x],{x,0,2}] * g[w]"#,
      r#"SeriesData[x, 0, {f[0]*g[w], g[w]*Derivative[1][f][0], (g[w]*Derivative[2][f][0])/2}, 0, 3, 1]"#,
    );
  }
  #[test]
  fn series_4() {
    assert_case(
      r#"series = Series[Cosh[x],{x,0,2}]; Head[series]; series // FullForm; series + Series[Sinh[x],{x, 0, 3}]; Series[f[x],{x,0,2}] * g[w]; Series[Exp[-a x],{x,0,2}] * Series[Exp[-b x],{x,0,2}]"#,
      r#"SeriesData[x, 0, {1, -a - b, a^2/2 + a*b + b^2/2}, 0, 3, 1]"#,
    );
  }
  #[test]
  fn d_18() {
    assert_case(
      r#"series = Series[Cosh[x],{x,0,2}]; Head[series]; series // FullForm; series + Series[Sinh[x],{x, 0, 3}]; Series[f[x],{x,0,2}] * g[w]; Series[Exp[-a x],{x,0,2}] * Series[Exp[-b x],{x,0,2}]; D[Series[Exp[-a x],{x,0,2}],a]"#,
      r#"SeriesData[x, 0, {-1, a}, 1, 3, 1]"#,
    );
  }
  #[test]
  fn d_solve_1() {
    assert_case(
      r#"DSolve[y''[x] == 0, y[x], x]"#,
      r#"{{y[x] -> C[1] + x*C[2]}}"#,
    );
  }
  #[test]
  fn d_solve_2() {
    assert_case(
      r#"DSolve[y''[x] == 0, y[x], x]; DSolve[y''[x] == y[x], y[x], x]"#,
      r#"{{y[x] -> E^x*C[1] + C[2]/E^x}}"#,
    );
  }
  #[test]
  fn d_solve_3() {
    assert_case(
      r#"DSolve[y''[x] == 0, y[x], x]; DSolve[y''[x] == y[x], y[x], x]; DSolve[y''[x] == y[x], y, x]"#,
      r#"{{y -> Function[{x}, E^x*C[1] + C[2]/E^x]}}"#,
    );
  }
  #[test]
  fn d_solve_4() {
    assert_case(
      r#"DSolve[y''[x] == 0, y[x], x]; DSolve[y''[x] == y[x], y[x], x]; DSolve[y''[x] == y[x], y, x]; DSolve[D[f[x, y], x] / f[x, y] + 3 D[f[x, y], y] / f[x, y] == 2, f, {x, y}]"#,
      r#"{{f -> Function[{x, y}, E^(2*x)*C[1][-3*x + y]]}}"#,
    );
  }
  #[test]
  fn d_solve_5() {
    assert_case(
      r#"DSolve[y''[x] == 0, y[x], x]; DSolve[y''[x] == y[x], y[x], x]; DSolve[y''[x] == y[x], y, x]; DSolve[D[f[x, y], x] / f[x, y] + 3 D[f[x, y], y] / f[x, y] == 2, f, {x, y}]; DSolve[D[f[x, y], x] x + D[f[x, y], y] y == 2, f[x, y], {x, y}]"#,
      r#"{{f[x, y] -> 2*Log[x] + C[1][y/x]}}"#,
    );
  }
  #[test]
  fn d_solve_euler_pde_zero_rhs() {
    // Regression: the `c == 0` case used to leak an unfolded `0*Log[x]` term.
    assert_case(
      r#"DSolve[x D[f[x, y], x] + y D[f[x, y], y] == 0, f, {x, y}]"#,
      r#"{{f -> Function[{x, y}, C[1][y/x]]}}"#,
    );
  }
  #[test]
  fn d_solve_6() {
    assert_case(
      r#"DSolve[y''[x] == 0, y[x], x]; DSolve[y''[x] == y[x], y[x], x]; DSolve[y''[x] == y[x], y, x]; DSolve[D[f[x, y], x] / f[x, y] + 3 D[f[x, y], y] / f[x, y] == 2, f, {x, y}]; DSolve[D[f[x, y], x] x + D[f[x, y], y] y == 2, f[x, y], {x, y}]; DSolve[D[y[x, t], t] + 2 D[y[x, t], x] == 0, y[x, t], {x, t}]"#,
      r#"{{y[x, t] -> C[1][t - x/2]}}"#,
    );
  }
  #[test]
  fn input_form() {
    assert_case(
      r#"InputForm["A string"]; InputForm[f'[x]]; InputForm[Derivative[1, 0][f][x]]"#,
      r#"InputForm[Derivative[1, 0][f][x]]"#,
    );
  }
  #[test]
  fn plus_2() {
    assert_case(
      r#"InputForm["A string"]; InputForm[f'[x]]; InputForm[Derivative[1, 0][f][x]]; 2+F[x] // InputForm"#,
      r#"InputForm[2 + F[x]]"#,
    );
  }
  #[test]
  fn plus_3() {
    assert_case(
      r#"InputForm["A string"]; InputForm[f'[x]]; InputForm[Derivative[1, 0][f][x]]; 2+F[x] // InputForm; 2+F[x] // FullForm"#,
      r#"FullForm[2 + F[x]]"#,
    );
  }
  #[test]
  fn foo_1() {
    // Wolframscript-matched expectation. mathics expected
    // \`InputForm[Bar]\` — the user-defined Format rule rewriting
    // Foo[x] to Bar — but wolframscript -code (and Woxi) keep
    // \`InputForm[Foo[x]]\` because the Format rule only fires inside
    // the front-end's display pipeline, not at top-level OutputForm.
    assert_case(
      r#"InputForm["A string"]; InputForm[f'[x]]; InputForm[Derivative[1, 0][f][x]]; 2+F[x] // InputForm; 2+F[x] // FullForm; Format[Foo[x], InputForm] := Bar; Foo[x] // InputForm"#,
      r#"InputForm[Foo[x]]"#,
    );
  }
  #[test]
  fn foo_2() {
    // mathics applied the user `Format` rule and printed `Baz`, but
    // wolframscript only fires `Format` rules in the front-end display
    // pipeline — at top-level it keeps the bare `Foo[x]` inside the
    // FullForm wrapper. Woxi matches wolframscript.
    assert_case(
      r#"InputForm["A string"]; InputForm[f'[x]]; InputForm[Derivative[1, 0][f][x]]; 2+F[x] // InputForm; 2+F[x] // FullForm; Format[Foo[x], InputForm] := Bar; Foo[x] // InputForm; Format[Foo[x], InputForm] := Baz; Foo[x] // FullForm"#,
      r#"FullForm[Foo[x]]"#,
    );
  }
  #[test]
  fn output_form_1() {
    // mathics rendered the partial-derivative as 2D ASCII art `(1,0)\nf [x]`;
    // wolframscript -code returns the unevaluated wrapper
    // `OutputForm[Derivative[1, 0][f][x]]` verbatim. Woxi matches.
    assert_case(
      r#"OutputForm[f'[x]]; OutputForm[Derivative[1, 0][f][x]]"#,
      r#"OutputForm[Derivative[1, 0][f][x]]"#,
    );
  }
  #[test]
  fn output_form_2() {
    // OutputForm keeps its wrapper to match wolframscript:
    // `OutputForm[{"A string", a + b}]` → `OutputForm[{A string, a + b}]`
    // (strings unquoted inside the OutputForm rendering).
    assert_case(
      r#"OutputForm[f'[x]]; OutputForm[Derivative[1, 0][f][x]]; OutputForm[{"A string", a + b}]"#,
      r#"OutputForm[{A string, a + b}]"#,
    );
  }
  #[test]
  fn list_literal() {
    assert_case(
      r#"OutputForm[f'[x]]; OutputForm[Derivative[1, 0][f][x]]; OutputForm[{"A string", a + b}]; {"A string", a + b}"#,
      r#"{"A string", a + b}"#,
    );
  }
  #[test]
  fn output_form_3() {
    // OutputForm wrapper is now preserved (commit f8941596), so
    // `OutputForm[StringForm[…]]` round-trips through Woxi to the
    // wolframscript-matching string.
    assert_case(
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]; StringForm["`2` bla `1` blub `` bla `3`", a, b, c]; StringForm["`-1` bla", a]; StringForm["`2` bla", a]; StringForm["`` is Global`a", a]; StringForm["`` is Global\\`a", a]; OutputForm[StringForm["Integral of f: ``", Integrate[F[x],x]]]"#,
      r#"OutputForm[StringForm[Integral of f: ``, Integrate[F[x], x]]]"#,
    );
  }
  #[test]
  fn standard_form() {
    // mathics rendered StandardForm contents to box-syntax markup;
    // wolframscript -code returns the unevaluated wrapper
    // `StandardForm[StringForm[Integral of f: \`\`, Integrate[F[x], x]]]`
    // verbatim. Woxi matches.
    assert_case(
      r#"StringForm["`1` bla `2` blub `3` bla `2`", a, b, c]; StringForm["`2` bla `1` blub `` bla `3`", a, b, c]; StringForm["`-1` bla", a]; StringForm["`2` bla", a]; StringForm["`` is Global`a", a]; StringForm["`` is Global\\`a", a]; OutputForm[StringForm["Integral of f: ``", Integrate[F[x],x]]]; StandardForm[StringForm["Integral of f: ``", Integrate[F[x],x]]]"#,
      r#"StandardForm[StringForm[Integral of f: ``, Integrate[F[x], x]]]"#,
    );
  }
  #[test]
  fn my_d_1() {
    assert_case(r#"MyD[Sin[f_],x_?NotListQ] := D[f,x]*Cos[f]"#, r#"Null"#);
  }
  #[test]
  fn my_d_2() {
    assert_case(
      r#"MyD[Sin[f_],x_?NotListQ] := D[f,x]*Cos[f]; MyD[Sin[2 x], x]"#,
      r#"MyD[Sin[2*x], x]"#,
    );
  }
  #[test]
  fn d_19() {
    assert_case(
      r#"MyD[Sin[f_],x_?NotListQ] := D[f,x]*Cos[f]; MyD[Sin[2 x], x]; D[Sin[2 x], x]"#,
      r#"2*Cos[2*x]"#,
    );
  }
  #[test]
  fn my_d_3() {
    assert_case(
      r#"MyD[Sin[f_],x_?NotListQ] := D[f,x]*Cos[f]; MyD[Sin[2 x], x]; D[Sin[2 x], x]; MyD[{Sin[2], Sin[4]}, {1, 2}]"#,
      r#"MyD[{Sin[2], Sin[4]}, {1, 2}]"#,
    );
  }
  #[test]
  fn series_5() {
    assert_case(
      r#"Normal[Pi]; Series[Exp[x], {x, 0, 5}]"#,
      r#"SeriesData[x, 0, {1, 1, 1/2, 1/6, 1/24, 1/120}, 0, 6, 1]"#,
    );
  }
  #[test]
  fn normal() {
    assert_case(
      r#"Normal[Pi]; Series[Exp[x], {x, 0, 5}]; Normal[%]"#,
      r#"Out[0]"#,
    );
  }
  #[test]
  fn integrate_12() {
    assert_case(
      r#"FresnelC[{0, Infinity}]; Integrate[Cos[x^2 Pi/2], {x, 0, z}]"#,
      r#"FresnelC[z]"#,
    );
  }
  #[test]
  fn integrate_13() {
    assert_case(
      r#"FresnelS[{0, Infinity}]; Integrate[Sin[x^2 Pi/2], {x, 0, z}]"#,
      r#"FresnelS[z]"#,
    );
  }
  #[test]
  fn d_20() {
    assert_case(
      r#"BesselJ[0, 5.2]; D[BesselJ[n, z], z]"#,
      r#"(BesselJ[-1 + n, z] - BesselJ[1 + n, z])/2"#,
    );
  }
  #[test]
  fn bessel_j() {
    assert_case(
      r#"BesselJ[0, 5.2]; D[BesselJ[n, z], z]; BesselJ[0., 0.]"#,
      r#"1."#,
    );
  }
  // Derivatives of the other cylinder functions: Y/HankelH like J (subtract),
  // I adds the neighbours, K adds and negates.
  #[test]
  fn d_bessel_y_i_k_hankel() {
    assert_case(r#"D[BesselY[0, x], x]"#, r#"-BesselY[1, x]"#);
    assert_case(
      r#"D[BesselY[n, x], x]"#,
      r#"(BesselY[-1 + n, x] - BesselY[1 + n, x])/2"#,
    );
    assert_case(r#"D[BesselI[0, x], x]"#, r#"BesselI[1, x]"#);
    assert_case(
      r#"D[BesselI[n, x], x]"#,
      r#"(BesselI[-1 + n, x] + BesselI[1 + n, x])/2"#,
    );
    assert_case(r#"D[BesselK[0, x], x]"#, r#"-BesselK[1, x]"#);
    assert_case(
      r#"D[BesselK[n, x], x]"#,
      r#"(-BesselK[-1 + n, x] - BesselK[1 + n, x])/2"#,
    );
    assert_case(
      r#"D[HankelH1[n, x], x]"#,
      r#"(HankelH1[-1 + n, x] - HankelH1[1 + n, x])/2"#,
    );
    // Chain rule.
    assert_case(r#"D[BesselK[0, 2 x], x]"#, r#"-2*BesselK[1, 2*x]"#);
  }
  #[test]
  fn series_6() {
    assert_case(
      r#"Table[LucasL[n], {n, 1, 5}]; Series[LucasL[1/2, x], {x, 0, 5}]"#,
      r#"SeriesData[x, 0, {1, 1/4, 1/32, -1/128, -5/2048, 7/8192}, 0, 6, 1]"#,
    );
  }
  #[test]
  fn expr_3() {
    assert_case(
      r#"System`Convert`B64Dump`B64Encode["Hello world"]; System`Convert`B64Dump`B64Decode[%]; System`Convert`B64Dump`B64Encode[Integrate[f[x],{x,0,2}]]"#,
      r#"System`Convert`B64Dump`B64Encode[Integrate[f[x], {x, 0, 2}]]"#,
    );
  }
  #[test]
  fn expr_4() {
    assert_case(
      r#"System`Convert`B64Dump`B64Encode["Hello world"]; System`Convert`B64Dump`B64Decode[%]; System`Convert`B64Dump`B64Encode[Integrate[f[x],{x,0,2}]]; System`Convert`B64Dump`B64Decode[%]"#,
      r#"System`Convert`B64Dump`B64Decode[Out[0]]"#,
    );
  }
  #[test]
  fn series_7() {
    assert_case(
      r#"Series[F[x,z],{x, g[y], 2}, {z, a, 2}]//FullForm"#,
      r#"FullForm[SeriesData[x, g[y], {SeriesData[z, a, {F[g[y], a], Derivative[0, 1][F][g[y], a], Derivative[0, 2][F][g[y], a]/2}, 0, 3, 1], SeriesData[z, a, {Derivative[1, 0][F][g[y], a], Derivative[1, 1][F][g[y], a], Derivative[1, 2][F][g[y], a]/2}, 0, 3, 1], SeriesData[z, a, {Derivative[2, 0][F][g[y], a]/2, Derivative[2, 1][F][g[y], a]/2, Derivative[2, 2][F][g[y], a]/4}, 0, 3, 1]}, 0, 3, 1]]"#,
    );
  }
  #[test]
  fn d_21() {
    assert_case(
      r#"D[Series[F[x,z],{x, g[y], 2}, {z, a, 2}], y]//FullForm"#,
      r#"FullForm[SeriesData[x, g[y], {}, 2, 2, 1]]"#,
    );
  }
  #[test]
  fn series_8() {
    assert_case(
      r#"D[Series[F[x,z],{x, g[y], 2}, {z, a, 2}], y]//FullForm; Series[Exp[x], {x, 0, 2}] * (x ^ (1 / 3))"#,
      r#"SeriesData[x, 0, {1, 0, 0, 1}, 1, 7, 3]"#,
    );
  }
  #[test]
  fn series_9() {
    assert_case(
      r#"D[Series[F[x,z],{x, g[y], 2}, {z, a, 2}], y]//FullForm; Series[Exp[x], {x, 0, 2}] * (x ^ (1 / 3)); Series[Exp[x],{x, 0, 2}]Series[Exp[-y],{y, 0,2}]"#,
      r#"SeriesData[x, 0, {SeriesData[y, 0, {1, -1, 1/2}, 0, 3, 1], SeriesData[y, 0, {1, -1, 1/2}, 0, 3, 1], SeriesData[y, 0, {1/2, -1/2, 1/4}, 0, 3, 1]}, 0, 3, 1]"#,
    );
  }
  #[test]
  fn series_10() {
    assert_case(
      r#"D[Series[F[x,z],{x, g[y], 2}, {z, a, 2}], y]//FullForm; Series[Exp[x], {x, 0, 2}] * (x ^ (1 / 3)); Series[Exp[x],{x, 0, 2}]Series[Exp[-y],{y, 0,2}]; Series[Exp[x],{x, 0, 2}]Series[Exp[-y],{y, 0,2}]//Normal"#,
      r#"1 - y + y^2/2 + x^2*(1/2 - y/2 + y^2/4) + x*(1 - y + y^2/2)"#,
    );
  }
  #[test]
  fn series_11() {
    assert_case(
      r#"D[Series[F[x,z],{x, g[y], 2}, {z, a, 2}], y]//FullForm; Series[Exp[x], {x, 0, 2}] * (x ^ (1 / 3)); Series[Exp[x],{x, 0, 2}]Series[Exp[-y],{y, 0,2}]; Series[Exp[x],{x, 0, 2}]Series[Exp[-y],{y, 0,2}]//Normal; Series[Exp[x],{x,0,3}]-1-x-x^2"#,
      r#"SeriesData[x, 0, {-1/2, 1/6}, 2, 4, 1]"#,
    );
  }
  #[test]
  fn minus_1() {
    assert_case(
      r#"D[Series[F[x,z],{x, g[y], 2}, {z, a, 2}], y]//FullForm; Series[Exp[x], {x, 0, 2}] * (x ^ (1 / 3)); Series[Exp[x],{x, 0, 2}]Series[Exp[-y],{y, 0,2}]; Series[Exp[x],{x, 0, 2}]Series[Exp[-y],{y, 0,2}]//Normal; Series[Exp[x],{x,0,3}]-1-x-x^2; (Series[Exp[x-y],{x, 0, 2},{y, 0 , 2}]//Normal)-(1-(x-y))// ExpandAll"#,
      r#"2*x + x^2/2 - 2*y - x*y - (x^2*y)/2 + y^2/2 + (x*y^2)/2 + (x^2*y^2)/4"#,
    );
  }
  #[test]
  fn time_constrained_1() {
    assert_case(
      r#"TimeConstrained[Integrate[Sin[x]^1000, x];,.001]"#,
      r#"$Aborted"#,
    );
  }
  #[test]
  fn time_constrained_2() {
    assert_case(
      r#"TimeConstrained[Integrate[Sin[x]^1000, x];,.001]; TimeConstrained[Integrate[Cos[x]^1000,x];,.001, Integrate[Cos[x],x]]"#,
      r#"Sin[x]"#,
    );
  }
  // MemoryConstrained: returns the evaluated result if its ByteCount
  // fits within the budget; otherwise $Aborted (2-arg form) or the
  // supplied fallback (3-arg form). Body is HoldFirst, so it isn't
  // evaluated before the size budget is checked.
  #[test]
  fn memory_constrained_small_result() {
    assert_case(r#"MemoryConstrained[1 + 2, 1000]"#, r#"3"#);
  }
  #[test]
  fn memory_constrained_large_result_aborts() {
    // Range[1000] is a 1000-element list — well over 100 bytes.
    assert_case(r#"MemoryConstrained[Range[1000], 100]"#, r#"$Aborted"#);
  }
  #[test]
  fn memory_constrained_fallback_form() {
    assert_case(
      r#"MemoryConstrained[Range[1000], 100, $Failed]"#,
      r#"$Failed"#,
    );
  }
  // Body that fits returns the result even with the fallback form.
  #[test]
  fn memory_constrained_fallback_unused_when_fits() {
    assert_case(r#"MemoryConstrained[1 + 2, 1000, $Failed]"#, r#"3"#);
  }
  #[test]
  fn d_22() {
    assert_case(r#"D[{y, -x}[2], {x, y}]"#, r#"D[{y, -x}[2], {x, y}]"#);
  }
  #[test]
  fn d_23() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand"#,
      r#"(-2*Sin[x])/3 - (2*x*Cos[x]^2*Sin[x])/3 - (Cos[x]*Sin[x]^2)/3 + (x*Sin[x]^3)/3"#,
    );
  }
  #[test]
  fn d_24() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]"#,
      r#"Derivative[2][f][#1]"#,
    );
  }
  #[test]
  fn d_25() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]"#,
      r#"0"#,
    );
  }
  #[test]
  fn apart_1() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]"#,
      r#"Derivative[2][f][2*x]"#,
    );
  }
  #[test]
  fn apart_2() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]"#,
      r#"Derivative[2][f][2*x]"#,
    );
  }
  #[test]
  fn d_26() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]"#,
      r#"{2*#1}"#,
    );
  }
  #[test]
  fn find_root() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]"#,
      r#"{x -> 2.5}"#,
    );
  }
  #[test]
  fn head_2() {
    // wolframscript dumps the internal `Integrate` package's
    // private-context rules (`Integrate`ImproperDump`f_` etc.) which
    // are wolframscript-installation-specific and not part of
    // Woxi's symbolic-Integrate implementation. Verify the
    // documented contract: `DownValues[Integrate]` returns a List
    // (empty when no user-defined rules exist, matching Woxi).
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; Head[DownValues[Integrate]] === List"#,
      r#"True"#,
    );
  }
  #[test]
  fn definition() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]"#,
      r#"Attributes[Integrate] = {Protected, ReadProtected}

Options[Integrate] := {Assumptions :> $Assumptions, GenerateConditions -> Automatic, GeneratedParameters -> None, PrincipalValue -> False}"#,
    );
  }
  #[test]
  fn integrate_14() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]"#,
      r#"Integrate[Hold[x + x], {x, a, b}]"#,
    );
  }
  #[test]
  fn integrate_15() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]"#,
      r#"Integrate[sin[x], x]"#,
    );
  }
  #[test]
  fn integrate_16() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]"#,
      r#"x^2/2 + 0.2222222222222222*x^(9/2)"#,
    );
  }
  #[test]
  fn integrate_17() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]"#,
      r#"(-"p" + "q")*F[a, "x"]"#,
    );
  }
  #[test]
  fn integrate_18() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]"#,
      r#"(ArcTan*x^2)/2"#,
    );
  }
  #[test]
  fn integrate_19() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]"#,
      r#"Integrate[E[x], x]"#,
    );
  }
  #[test]
  fn integrate_20() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]"#,
      r#"2*Sqrt[Pi]"#,
    );
  }
  #[test]
  fn integrate_21() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]"#,
      r#"x/E^x^(-2) + Sqrt[Pi]*Erf[x^(-1)]"#,
    );
  }
  #[test]
  fn expression_1() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]; True'"#,
      r#"Derivative[1][True]"#,
    );
  }
  #[test]
  fn expression_2() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]; True'; False'"#,
      r#"Derivative[1][False]"#,
    );
  }
  #[test]
  fn expression_3() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]; True'; False'; List'"#,
      r#"{1}&"#,
    );
  }
  #[test]
  fn expression_4() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]; True'; False'; List'; 1'"#,
      r#"0&"#,
    );
  }
  #[test]
  fn minus_2() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]; True'; False'; List'; 1'; -1.4'"#,
      r#"-(0&)"#,
    );
  }
  #[test]
  fn divide_5() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]; True'; False'; List'; 1'; -1.4'; (2/3)'"#,
      r#"0&"#,
    );
  }
  #[test]
  fn expression_5() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]; True'; False'; List'; 1'; -1.4'; (2/3)'; I'"#,
      r#"0&"#,
    );
  }
  #[test]
  fn derivative_10() {
    assert_case(
      r#"D[2/3 Cos[x] - 1/3 x Cos[x] Sin[x] ^ 2,x]//Expand; D[f[#1], {#1,2}]; D[(#1&)[t],{t,4}]; Attributes[f] ={HoldAll}; Apart[f''[x + x]]; Attributes[f] = {}; Apart[f''[x + x]]; D[{#^2}, #]; FindRoot[2.5==x,{x,0}]; DownValues[Integrate]; Definition[Integrate]; Integrate[Hold[x + x], {x, a, b}]; Integrate[sin[x], x]; Integrate[x ^ 3.5 + x, x]; Integrate[F[a, "x"],{x,"p","q"}]; Integrate[ArcTan(x), x]; Integrate[E[x], x]; Integrate[Exp[-(x/2)^2],{x,-Infinity,+Infinity}]; Integrate[Exp[-1/(x^2)], x]; True'; False'; List'; 1'; -1.4'; (2/3)'; I'; Derivative[0,0,1][List]"#,
      r#"{0, 0, 1}&"#,
    );
  }
  #[test]
  fn d_solve_7() {
    assert_case(
      r#"DSolve[f'[x] == f[x], f, x] // FullForm"#,
      r#"FullForm[{{f -> Function[{x}, E^x*C[1]]}}]"#,
    );
  }
  #[test]
  fn d_solve_8() {
    // ReplaceAll on the DSolve result must descend into the
    // `Function[{x}, …]` body so the C[1] → 1 rule fires there.
    assert_case(
      r#"DSolve[f'[x] == f[x], f, x] // FullForm; DSolve[f'[x] == f[x], f, x] /. {C[1] -> 1}"#,
      r#"{{f -> Function[{x}, E^x*1]}}"#,
    );
  }
  #[test]
  fn d_solve_9() {
    assert_case(
      r#"DSolve[f'[x] == f[x], f, x] // FullForm; DSolve[f'[x] == f[x], f, x] /. {C[1] -> 1}; DSolve[f'[x] == f[x], f, x] /. {C -> D}"#,
      r#"{{f -> Function[{x}, E^x*D[1]]}}"#,
    );
  }
  #[test]
  fn d_solve_10() {
    assert_case(
      r#"DSolve[f'[x] == f[x], f, x] // FullForm; DSolve[f'[x] == f[x], f, x] /. {C[1] -> 1}; DSolve[f'[x] == f[x], f, x] /. {C -> D}; DSolve[f'[x] == f[x], f, x] /. {C[1] -> C[0]}"#,
      r#"{{f -> Function[{x}, E^x*C[0]]}}"#,
    );
  }
  #[test]
  fn limit_4() {
    assert_case(r#"Limit[Tan[x], x->Pi/2]"#, r#"Indeterminate"#);
  }
  #[test]
  fn limit_5() {
    assert_case(
      r#"Limit[Tan[x], x->Pi/2]; Limit[Cot[x], x->0]"#,
      r#"Indeterminate"#,
    );
  }
  #[test]
  fn limit_6() {
    assert_case(
      r#"Limit[Tan[x], x->Pi/2]; Limit[Cot[x], x->0]; Limit[Cot[x], x->Infinity]"#,
      r#"Indeterminate"#,
    );
  }
  #[test]
  fn limit_7() {
    assert_case(
      r#"Limit[Tan[x], x->Pi/2]; Limit[Cot[x], x->0]; Limit[Cot[x], x->Infinity]; Limit[Cot[x], x->-Infinity]"#,
      r#"Indeterminate"#,
    );
  }
}

mod ndeigenvalues_diffusion_line {
  use super::*;

  // For 1D Laplacian (DiffusionPDETerm) on Line[{{a}, {b}}] with the
  // default Neumann boundary condition, the eigenvalues are (kπ/L)²
  // for k = 0, 1, 2, … where L = b - a. Wolfram's finite-element
  // solver returns these with small discretisation errors; Woxi
  // returns the analytic values directly.

  fn parse_list(s: &str) -> Vec<f64> {
    s.trim()
      .trim_start_matches('{')
      .trim_end_matches('}')
      .split(',')
      .map(|p| p.trim().parse().unwrap())
      .collect()
  }

  fn assert_close(got: f64, expected: f64, msg: &str) {
    let tol = (expected.abs() * 1e-3).max(1e-9);
    assert!(
      (got - expected).abs() < tol,
      "{msg}: got {got}, expected {expected}"
    );
  }

  #[test]
  fn unit_interval_three_modes() {
    let result = interpret(
      "NDEigenvalues[DiffusionPDETerm[{u[x], {x}}], u, \
       Element[{x}, Line[{{0}, {1}}]], 3]",
    )
    .unwrap();
    let xs = parse_list(&result);
    assert_eq!(xs.len(), 3);
    let pi_sq = std::f64::consts::PI.powi(2);
    assert_close(xs[0], 0.0, "λ_0");
    assert_close(xs[1], pi_sq, "λ_1 = π²");
    assert_close(xs[2], 4.0 * pi_sq, "λ_2 = 4π²");
  }

  #[test]
  fn double_interval_three_modes() {
    let result = interpret(
      "NDEigenvalues[DiffusionPDETerm[{u[x], {x}}], u, \
       Element[{x}, Line[{{0}, {2}}]], 3]",
    )
    .unwrap();
    let xs = parse_list(&result);
    let pi = std::f64::consts::PI;
    assert_close(xs[0], 0.0, "λ_0");
    assert_close(xs[1], (pi / 2.0).powi(2), "λ_1 = (π/2)²");
    assert_close(xs[2], pi.powi(2), "λ_2 = π²");
  }

  #[test]
  fn pi_interval_four_modes() {
    let result = interpret(
      "NDEigenvalues[DiffusionPDETerm[{u[x], {x}}], u, \
       Element[{x}, Line[{{0}, {Pi}}]], 4]",
    )
    .unwrap();
    let xs = parse_list(&result);
    assert_close(xs[0], 0.0, "λ_0");
    assert_close(xs[1], 1.0, "λ_1 = 1");
    assert_close(xs[2], 4.0, "λ_2 = 4");
    assert_close(xs[3], 9.0, "λ_3 = 9");
  }
}

mod z_transform {
  use super::*;

  #[test]
  fn polynomial_monomials() {
    assert_eq!(interpret("ZTransform[1, n, z]").unwrap(), "z/(-1 + z)");
    assert_eq!(interpret("ZTransform[n, n, z]").unwrap(), "z/(-1 + z)^2");
    assert_eq!(
      interpret("ZTransform[n^2, n, z]").unwrap(),
      "(z*(1 + z))/(-1 + z)^3"
    );
    assert_eq!(
      interpret("ZTransform[n^3, n, z]").unwrap(),
      "(z*(1 + 4*z + z^2))/(-1 + z)^4"
    );
    assert_eq!(
      interpret("ZTransform[n^4, n, z]").unwrap(),
      "(z*(1 + 11*z + 11*z^2 + z^3))/(-1 + z)^5"
    );
  }

  #[test]
  fn constant_multiples() {
    assert_eq!(interpret("ZTransform[2, n, z]").unwrap(), "(2*z)/(-1 + z)");
    assert_eq!(
      interpret("ZTransform[2 n, n, z]").unwrap(),
      "(2*z)/(-1 + z)^2"
    );
    assert_eq!(
      interpret("ZTransform[2 n^2, n, z]").unwrap(),
      "(2*z*(1 + z))/(-1 + z)^3"
    );
    // Symbols free of n act as constants
    assert_eq!(interpret("ZTransform[x, n, z]").unwrap(), "(x*z)/(-1 + z)");
  }

  #[test]
  fn geometric_sequences() {
    // Symbolic base keeps the (a - z) denominator with a sign wrapper
    assert_eq!(interpret("ZTransform[a^n, n, z]").unwrap(), "-(z/(a - z))");
    // Numeric bases use the canonical (-a + z) order
    assert_eq!(interpret("ZTransform[2^n, n, z]").unwrap(), "z/(-2 + z)");
    // Rational bases clear denominators
    assert_eq!(
      interpret("ZTransform[(1/3)^n, n, z]").unwrap(),
      "(3*z)/(-1 + 3*z)"
    );
  }

  #[test]
  fn polynomial_times_geometric() {
    assert_eq!(
      interpret("ZTransform[n a^n, n, z]").unwrap(),
      "(a*z)/(a - z)^2"
    );
    assert_eq!(
      interpret("ZTransform[n 2^n, n, z]").unwrap(),
      "(2*z)/(-2 + z)^2"
    );
    // Even denominator powers of a rational base flip to a positive
    // constant term: (1 - 3*z)^2, not (-1 + 3*z)^2
    assert_eq!(
      interpret("ZTransform[n/3^n, n, z]").unwrap(),
      "(3*z)/(1 - 3*z)^2"
    );
    assert_eq!(
      interpret("ZTransform[n^2 a^n, n, z]").unwrap(),
      "-((a*z*(a + z))/(a - z)^3)"
    );
    assert_eq!(
      interpret("ZTransform[n^2 2^n, n, z]").unwrap(),
      "(2*z*(2 + z))/(-2 + z)^3"
    );
    // The documentation example
    assert_eq!(
      interpret("ZTransform[n^2/2^n, n, z]").unwrap(),
      "(2*z*(1 + 2*z))/(-1 + 2*z)^3"
    );
  }

  #[test]
  fn inverse_factorial() {
    assert_eq!(interpret("ZTransform[1/n!, n, z]").unwrap(), "E^z^(-1)");
    assert_eq!(interpret("ZTransform[3^n/n!, n, z]").unwrap(), "E^(3/z)");
  }

  // Z{Sin[a n]} = (z Sin[a])/(1 + z^2 - 2 z Cos[a]),
  // Z{Cos[a n]} = (z (z - Cos[a]))/(1 + z^2 - 2 z Cos[a]).
  #[test]
  fn trig_z_transforms() {
    assert_eq!(
      interpret("ZTransform[Sin[n], n, z]").unwrap(),
      "(z*Sin[1])/(1 + z^2 - 2*z*Cos[1])"
    );
    assert_eq!(
      interpret("ZTransform[Cos[n], n, z]").unwrap(),
      "(z*(z - Cos[1]))/(1 + z^2 - 2*z*Cos[1])"
    );
    assert_eq!(
      interpret("ZTransform[Sin[a n], n, z]").unwrap(),
      "(z*Sin[a])/(1 + z^2 - 2*z*Cos[a])"
    );
    assert_eq!(
      interpret("ZTransform[Cos[a n], n, z]").unwrap(),
      "(z*(z - Cos[a]))/(1 + z^2 - 2*z*Cos[a])"
    );
    // Numeric and fractional coefficients.
    assert_eq!(
      interpret("ZTransform[Sin[2 n], n, z]").unwrap(),
      "(z*Sin[2])/(1 + z^2 - 2*z*Cos[2])"
    );
    assert_eq!(
      interpret("ZTransform[Sin[n/2], n, z]").unwrap(),
      "(z*Sin[1/2])/(1 + z^2 - 2*z*Cos[1/2])"
    );
  }

  // UnitStep[n] = 1 for n >= 0, so its Z-transform is z/(z - 1).
  #[test]
  fn unit_step_z_transform() {
    assert_eq!(
      interpret("ZTransform[UnitStep[n], n, z]").unwrap(),
      "z/(-1 + z)"
    );
  }

  #[test]
  fn unsupported_forms_stay_unevaluated() {
    // Sin[n^2] is not a linear a*n argument, so (like wolframscript) there is
    // no closed form. (Sin[a*n] IS supported — see trig_z_transforms below.)
    assert_eq!(
      interpret("ZTransform[Sin[n^2], n, z]").unwrap(),
      "ZTransform[Sin[n^2], n, z]"
    );
    assert_eq!(
      interpret("ZTransform[n^2 + n + 1, n, z]").unwrap(),
      "ZTransform[1 + n + n^2, n, z]"
    );
  }
}

mod inverse_z_transform {
  use super::*;

  #[test]
  fn geometric() {
    assert_eq!(
      interpret("InverseZTransform[z/(z - a), z, n]").unwrap(),
      "a^n"
    );
    assert_eq!(
      interpret("InverseZTransform[z/(z - 1), z, n]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("InverseZTransform[z/(z - 2), z, n]").unwrap(),
      "2^n"
    );
    assert_eq!(
      interpret("InverseZTransform[(3*z)/(-1 + 3*z), z, n]").unwrap(),
      "3^(-n)"
    );
    // Sign-wrapped spelling of the same transform
    assert_eq!(
      interpret("InverseZTransform[-(z/(a - z)), z, n]").unwrap(),
      "a^n"
    );
  }

  #[test]
  fn polynomial_sequences() {
    assert_eq!(
      interpret("InverseZTransform[z/(z - 1)^2, z, n]").unwrap(),
      "n"
    );
    assert_eq!(
      interpret("InverseZTransform[(z*(1 + z))/(-1 + z)^3, z, n]").unwrap(),
      "n^2"
    );
  }

  #[test]
  fn binomial_sequences() {
    assert_eq!(
      interpret("InverseZTransform[z/(z - 1)^3, z, n]").unwrap(),
      "((-1 + n)*n)/2"
    );
    assert_eq!(
      interpret("InverseZTransform[z/(z - 1)^4, z, n]").unwrap(),
      "((-2 + n)*(-1 + n)*n)/6"
    );
  }

  #[test]
  fn polynomial_times_geometric() {
    assert_eq!(
      interpret("InverseZTransform[(a*z)/(a - z)^2, z, n]").unwrap(),
      "a^n*n"
    );
    assert_eq!(
      interpret("InverseZTransform[(2*z)/(-2 + z)^2, z, n]").unwrap(),
      "2^n*n"
    );
    assert_eq!(
      interpret("InverseZTransform[(3*z)/(1 - 3*z)^2, z, n]").unwrap(),
      "n/3^n"
    );
    assert_eq!(
      interpret("InverseZTransform[(2*z*(2 + z))/(-2 + z)^3, z, n]").unwrap(),
      "2^n*n^2"
    );
    assert_eq!(
      interpret("InverseZTransform[-((a*z*(a + z))/(a - z)^3), z, n]").unwrap(),
      "a^n*n^2"
    );
    assert_eq!(
      interpret("InverseZTransform[(2*z*(1 + 2*z))/(-1 + 2*z)^3, z, n]")
        .unwrap(),
      "n^2/2^n"
    );
  }

  #[test]
  fn exponential_forms() {
    assert_eq!(
      interpret("InverseZTransform[E^(1/z), z, n]").unwrap(),
      "n!^(-1)"
    );
    assert_eq!(
      interpret("InverseZTransform[E^(3/z), z, n]").unwrap(),
      "3^n/n!"
    );
  }

  #[test]
  fn constants() {
    assert_eq!(
      interpret("InverseZTransform[(2*z)/(-1 + z), z, n]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("InverseZTransform[(x*z)/(-1 + z), z, n]").unwrap(),
      "x"
    );
    // z-free input is a DiscreteDelta impulse
    assert_eq!(
      interpret("InverseZTransform[1, z, n]").unwrap(),
      "DiscreteDelta[n]"
    );
    assert_eq!(
      interpret("InverseZTransform[5, z, n]").unwrap(),
      "5*DiscreteDelta[n]"
    );
  }

  #[test]
  fn round_trip_with_z_transform() {
    assert_eq!(
      interpret("InverseZTransform[ZTransform[n^2/2^n, n, z], z, n]").unwrap(),
      "n^2/2^n"
    );
    assert_eq!(
      interpret("ZTransform[InverseZTransform[z/(z - 2), z, n], n, z]")
        .unwrap(),
      "z/(-2 + z)"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("InverseZTransform[Sin[z], z, n]").unwrap(),
      "InverseZTransform[Sin[z], z, n]"
    );
  }
}

mod convolve {
  use super::*;

  #[test]
  fn gaussian_pairs() {
    assert_eq!(
      interpret("Convolve[E^(-x^2), E^(-x^2), x, y]").unwrap(),
      "Sqrt[Pi/2]/E^(y^2/2)"
    );
    assert_eq!(
      interpret("Convolve[E^(-x^2), E^(-2*x^2), x, y]").unwrap(),
      "Sqrt[Pi/3]/E^((2*y^2)/3)"
    );
  }

  #[test]
  fn arity_is_four_or_more() {
    use woxi::interpret_with_stdout;
    // Convolve takes 4 or more arguments. Fewer than 4 emits the `argm`
    // (minimum-count) message — not `argrx` — and stays unevaluated,
    // matching wolframscript.
    let r = interpret_with_stdout("Convolve[a, b, c]").unwrap();
    assert_eq!(r.result, "Convolve[a, b, c]");
    assert!(
      r.warnings.iter().any(|w| w.contains(
        "Convolve::argm: Convolve called with 3 arguments; 4 or more arguments are expected."
      )),
      "expected argm message, got {:?}",
      r.warnings
    );
    // More than four arguments is a valid arity and stays unevaluated with
    // no message when it cannot be computed symbolically.
    let r5 = interpret_with_stdout("Convolve[a, b, c, d, e]").unwrap();
    assert_eq!(r5.result, "Convolve[a, b, c, d, e]");
    assert!(
      !r5.warnings.iter().any(|w| w.contains("Convolve::")),
      "five arguments should not warn, got {:?}",
      r5.warnings
    );
  }

  #[test]
  fn unit_functions() {
    assert_eq!(
      interpret("Convolve[UnitBox[x], UnitBox[x], x, y]").unwrap(),
      "UnitTriangle[y]"
    );
    assert_eq!(
      interpret("Convolve[UnitStep[x], UnitStep[x], x, y]").unwrap(),
      "y*UnitStep[y]"
    );
  }

  #[test]
  fn exponential_step() {
    assert_eq!(
      interpret("Convolve[Exp[-x]*UnitStep[x], Exp[-x]*UnitStep[x], x, y]")
        .unwrap(),
      "(y*UnitStep[y])/E^y"
    );
    assert_eq!(
      interpret("Convolve[Exp[-2 x]*UnitStep[x], Exp[-2 x]*UnitStep[x], x, y]")
        .unwrap(),
      "(y*UnitStep[y])/E^(2*y)"
    );
  }

  #[test]
  fn dirac_delta_identity() {
    // DiracDelta is the convolution identity
    assert_eq!(
      interpret("Convolve[DiracDelta[x], Sin[x], x, y]").unwrap(),
      "Sin[y]"
    );
    assert_eq!(
      interpret("Convolve[DiracDelta[x], x^2 + 1, x, y]").unwrap(),
      "1 + y^2"
    );
    // Commuted argument order works too
    assert_eq!(
      interpret("Convolve[Sin[x], DiracDelta[x], x, y]").unwrap(),
      "Sin[y]"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    // Divergent convolution: wolframscript also leaves it unevaluated
    assert_eq!(
      interpret("Convolve[Sin[x], Cos[x], x, y]").unwrap(),
      "Convolve[Sin[x], Cos[x], x, y]"
    );
  }
}

mod function_range {
  use super::*;

  #[test]
  fn polynomials() {
    assert_eq!(interpret("FunctionRange[x^2, x, y]").unwrap(), "y >= 0");
    assert_eq!(
      interpret("FunctionRange[x^2 + 2 x + 3, x, y]").unwrap(),
      "y >= 2"
    );
    // Negative leading coefficient bounds from above
    assert_eq!(
      interpret("FunctionRange[-2 x^2 + 4, x, y]").unwrap(),
      "y <= 4"
    );
    // Non-constant linear and odd powers cover the reals
    assert_eq!(interpret("FunctionRange[2 x + 1, x, y]").unwrap(), "True");
    assert_eq!(interpret("FunctionRange[x^3, x, y]").unwrap(), "True");
    assert_eq!(interpret("FunctionRange[x^4, x, y]").unwrap(), "y >= 0");
  }

  #[test]
  fn trigonometric_and_hyperbolic() {
    assert_eq!(
      interpret("FunctionRange[Sin[x], x, y]").unwrap(),
      "-1 <= y <= 1"
    );
    assert_eq!(interpret("FunctionRange[Tan[x], x, y]").unwrap(), "True");
    // wolframscript prints the constant first for Cosh
    assert_eq!(interpret("FunctionRange[Cosh[x], x, y]").unwrap(), "1 <= y");
    assert_eq!(
      interpret("FunctionRange[Tanh[x], x, y]").unwrap(),
      "-1 < y < 1"
    );
  }

  #[test]
  fn exponential_and_friends() {
    assert_eq!(interpret("FunctionRange[E^x, x, y]").unwrap(), "y > 0");
    assert_eq!(interpret("FunctionRange[Log[x], x, y]").unwrap(), "True");
    assert_eq!(interpret("FunctionRange[Sqrt[x], x, y]").unwrap(), "y >= 0");
    assert_eq!(interpret("FunctionRange[Abs[x], x, y]").unwrap(), "y >= 0");
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("FunctionRange[Gamma[x], x, y]").unwrap(),
      "FunctionRange[Gamma[x], x, y]"
    );
  }
}

mod fourier_coefficient {
  use super::*;

  #[test]
  fn monomials_symbolic_order() {
    assert_eq!(
      interpret("FourierCoefficient[t, t, n]").unwrap(),
      "Piecewise[{{0, n == 0}}, (I*(-1)^n)/n]"
    );
    assert_eq!(
      interpret("FourierCoefficient[t^2, t, n]").unwrap(),
      "Piecewise[{{Pi^2/3, n == 0}}, (2*(-1)^n)/n^2]"
    );
    assert_eq!(
      interpret("FourierCoefficient[t^3, t, n]").unwrap(),
      "Piecewise[{{0, n == 0}}, (I*(-1)^n*(-6 + n^2*Pi^2))/n^3]"
    );
  }

  #[test]
  fn constants_are_discrete_delta() {
    assert_eq!(
      interpret("FourierCoefficient[1, t, n]").unwrap(),
      "DiscreteDelta[n]"
    );
  }

  #[test]
  fn linear_combination() {
    // The constant only contributes to the n == 0 piece; the coefficient
    // groups with I as (2*I)
    assert_eq!(
      interpret("FourierCoefficient[2 t + 1, t, n]").unwrap(),
      "Piecewise[{{1, n == 0}}, ((2*I)*(-1)^n)/n]"
    );
  }

  #[test]
  fn numeric_orders() {
    assert_eq!(interpret("FourierCoefficient[t, t, 2]").unwrap(), "I/2");
    assert_eq!(
      interpret("FourierCoefficient[t^2, t, 0]").unwrap(),
      "Pi^2/3"
    );
    assert_eq!(interpret("FourierCoefficient[t^2, t, 3]").unwrap(), "-2/9");
    assert_eq!(
      interpret("FourierCoefficient[t^3, t, 1]").unwrap(),
      "-I*(-6 + Pi^2)"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("FourierCoefficient[Sin[t], t, n]").unwrap(),
      "FourierCoefficient[Sin[t], t, n]"
    );
  }
}

mod fourier_sin_cos_coefficient {
  use super::*;

  #[test]
  fn sine_monomials() {
    assert_eq!(
      interpret("FourierSinCoefficient[t, t, n]").unwrap(),
      "(-2*(-1)^n)/n"
    );
    assert_eq!(
      interpret("FourierSinCoefficient[t^2, t, n]").unwrap(),
      "(-2*(2 - 2*(-1)^n + (-1)^n*n^2*Pi^2))/(n^3*Pi)"
    );
    assert_eq!(
      interpret("FourierSinCoefficient[t^3, t, n]").unwrap(),
      "(-2*(-1)^n*(-6 + n^2*Pi^2))/n^3"
    );
    assert_eq!(
      interpret("FourierSinCoefficient[1, t, n]").unwrap(),
      "(-2*(-1 + (-1)^n))/(n*Pi)"
    );
    // Scaled monomial
    assert_eq!(
      interpret("FourierSinCoefficient[2 t, t, n]").unwrap(),
      "(-4*(-1)^n)/n"
    );
  }

  #[test]
  fn cosine_monomials() {
    assert_eq!(
      interpret("FourierCosCoefficient[t, t, n]").unwrap(),
      "(2*(-1 + (-1)^n))/(n^2*Pi)"
    );
    assert_eq!(
      interpret("FourierCosCoefficient[t^2, t, n]").unwrap(),
      "(4*(-1)^n)/n^2"
    );
    assert_eq!(
      interpret("FourierCosCoefficient[t^3, t, n]").unwrap(),
      "(6*(2 - 2*(-1)^n + (-1)^n*n^2*Pi^2))/(n^4*Pi)"
    );
    assert_eq!(
      interpret("FourierCosCoefficient[1, t, n]").unwrap(),
      "2*DiscreteDelta[n]"
    );
  }

  #[test]
  fn cosine_zero_order_means() {
    assert_eq!(interpret("FourierCosCoefficient[t, t, 0]").unwrap(), "Pi");
    assert_eq!(
      interpret("FourierCosCoefficient[t^2, t, 0]").unwrap(),
      "(2*Pi^2)/3"
    );
    assert_eq!(
      interpret("FourierCosCoefficient[t^3, t, 0]").unwrap(),
      "Pi^3/2"
    );
  }

  #[test]
  fn numeric_orders() {
    assert_eq!(interpret("FourierSinCoefficient[t, t, 3]").unwrap(), "2/3");
    assert_eq!(
      interpret("FourierCosCoefficient[t^2, t, 4]").unwrap(),
      "1/4"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("FourierSinCoefficient[Sin[t], t, n]").unwrap(),
      "FourierSinCoefficient[Sin[t], t, n]"
    );
  }
}

mod binomial_theorem_sum {
  use super::*;

  // Sum[Binomial[N, k] r^k, {k, 0, N}] = (1 + r)^N.
  #[test]
  fn row_sum_is_two_pow_n() {
    assert_eq!(interpret("Sum[Binomial[n, k], {k, 0, n}]").unwrap(), "2^n");
  }

  #[test]
  fn binomial_theorem_symbolic_base() {
    assert_eq!(
      interpret("Sum[Binomial[n, k] x^k, {k, 0, n}]").unwrap(),
      "(1 + x)^n"
    );
  }

  #[test]
  fn binomial_theorem_numeric_base() {
    assert_eq!(
      interpret("Sum[Binomial[n, k] 2^k, {k, 0, n}]").unwrap(),
      "3^n"
    );
  }

  #[test]
  fn binomial_theorem_combined_base() {
    // (-1)^k 2^k = (-2)^k -> (1 - 2)^n = (-1)^n
    assert_eq!(
      interpret("Sum[Binomial[n, k] (-2)^k, {k, 0, n}]").unwrap(),
      "(-1)^n"
    );
  }

  #[test]
  fn alternating_row_sum_is_kronecker_delta() {
    assert_eq!(
      interpret("Sum[(-1)^k Binomial[n, k], {k, 0, n}]").unwrap(),
      "KroneckerDelta[n]"
    );
  }

  // Must NOT fire when the upper limit differs from the Binomial's first
  // argument, or for a different identity; those stay unevaluated.
  #[test]
  fn partial_row_sum_stays_unevaluated() {
    assert_eq!(
      interpret("Sum[Binomial[n, k], {k, 1, n}]").unwrap(),
      "Sum[Binomial[n, k], {k, 1, n}]"
    );
  }

  #[test]
  fn k_weighted_sum_stays_unevaluated() {
    assert_eq!(
      interpret("Sum[k Binomial[n, k], {k, 0, n}]").unwrap(),
      "Sum[k*Binomial[n, k], {k, 0, n}]"
    );
  }

  // Concrete bounds are unaffected (computed directly, not via the identity).
  #[test]
  fn concrete_bounds_unaffected() {
    assert_eq!(interpret("Sum[Binomial[5, k], {k, 0, 5}]").unwrap(), "32");
  }
}

mod harmonic_number_sum {
  use super::*;

  // Sum[HarmonicNumber[k], {k, 1, n}] = HyperHarmonicNumber[2, n]; the
  // generalized order carries through as the third argument.
  #[test]
  fn cumulative_harmonic() {
    assert_eq!(
      interpret("Sum[HarmonicNumber[k], {k, 1, n}]").unwrap(),
      "HyperHarmonicNumber[2, n]"
    );
    assert_eq!(
      interpret("Sum[HarmonicNumber[k, 2], {k, 1, n}]").unwrap(),
      "HyperHarmonicNumber[2, n, 2]"
    );
    assert_eq!(
      interpret("Sum[HarmonicNumber[k, r], {k, 1, n}]").unwrap(),
      "HyperHarmonicNumber[2, n, r]"
    );
    // A different expansion variable works, and numeric bounds still fold.
    assert_eq!(
      interpret("Sum[HarmonicNumber[k], {k, 1, m}]").unwrap(),
      "HyperHarmonicNumber[2, m]"
    );
    assert_eq!(
      interpret("Sum[HarmonicNumber[k], {k, 1, 5}]").unwrap(),
      "87/10"
    );
    assert_eq!(
      interpret("Sum[HarmonicNumber[k, 2], {k, 1, 5}]").unwrap(),
      "3899/600"
    );
  }
}

mod symbolic_sum_constant_factor {
  use super::*;

  // Sum pulls a constant factor out of a monomial: Sum[c k^p] = c Sum[k^p].
  #[test]
  fn integer_coefficient_times_var() {
    assert_eq!(interpret("Sum[2 k, {k, 1, n}]").unwrap(), "n*(1 + n)");
  }

  #[test]
  fn integer_coefficient_times_square() {
    assert_eq!(
      interpret("Sum[3 k^2, {k, 1, n}]").unwrap(),
      "(n*(1 + n)*(1 + 2*n))/2"
    );
  }

  #[test]
  fn symbolic_coefficient() {
    assert_eq!(interpret("Sum[a k, {k, 1, n}]").unwrap(), "(a*n*(1 + n))/2");
    assert_eq!(
      interpret("Sum[a k^2, {k, 1, n}]").unwrap(),
      "(a*n*(1 + n)*(1 + 2*n))/6"
    );
  }

  #[test]
  fn divided_by_constant() {
    assert_eq!(interpret("Sum[k/2, {k, 1, n}]").unwrap(), "(n*(1 + n))/4");
    assert_eq!(
      interpret("Sum[k^2/3, {k, 1, n}]").unwrap(),
      "(n*(1 + n)*(1 + 2*n))/18"
    );
  }

  #[test]
  fn divided_by_constant_symbol() {
    // n is constant w.r.t. the summation index k.
    assert_eq!(interpret("Sum[k/n, {k, 1, n}]").unwrap(), "(1 + n)/2");
  }
}

// Closed form of the Mercator series Sum[r^k/k, {k, 1, Infinity}] = -Log[1-r],
// which converges on the real interval [-1, 1). The boundary base r = -1 is the
// (conditionally convergent) alternating harmonic series.
mod infinite_log_series {
  use super::*;

  #[test]
  fn alternating_harmonic() {
    assert_eq!(
      interpret("Sum[(-1)^n/n, {n, 1, Infinity}]").unwrap(),
      "-Log[2]"
    );
  }

  // Alternating p-series Sum[(-1)^(n+c)/n^s, {n, 1, Infinity}] = +/-DirichletEta[s].
  // Covers the (-1)^(n+1) sign convention and s >= 2 that the Mercator log
  // series above does not.
  #[test]
  fn alternating_p_series_dirichlet_eta() {
    // s = 1: +/- Log[2].
    assert_eq!(
      interpret("Sum[(-1)^(n+1)/n, {n, 1, Infinity}]").unwrap(),
      "Log[2]"
    );
    assert_eq!(
      interpret("Sum[(-1)^(n-1)/n, {n, 1, Infinity}]").unwrap(),
      "Log[2]"
    );
    // s = 2: +/- Pi^2/12.
    assert_eq!(
      interpret("Sum[(-1)^(n+1)/n^2, {n, 1, Infinity}]").unwrap(),
      "Pi^2/12"
    );
    assert_eq!(
      interpret("Sum[(-1)^n/n^2, {n, 1, Infinity}]").unwrap(),
      "-1/12*Pi^2"
    );
    // s = 3: (3/4) Zeta[3]; s = 4: 7 Pi^4/720.
    assert_eq!(
      interpret("Sum[(-1)^(n+1)/n^3, {n, 1, Infinity}]").unwrap(),
      "(3*Zeta[3])/4"
    );
    assert_eq!(
      interpret("Sum[(-1)^n/n^4, {n, 1, Infinity}]").unwrap(),
      "(-7*Pi^4)/720"
    );
  }

  // Sums over the odd positive integers: Sum[1/(2n-1)^s] = DirichletLambda[s]
  // and Sum[(-1)^(n+1)/(2n-1)^s] = DirichletBeta[s]. (Previously the shifted
  // Leibniz Sum[(-1)^(n+1)/(2n-1)] hung.)
  #[test]
  fn odd_integer_reciprocal_sums() {
    // Dirichlet beta (alternating): Pi/4, Catalan, Pi^3/32.
    assert_eq!(
      interpret("Sum[(-1)^(n+1)/(2 n - 1), {n, 1, Infinity}]").unwrap(),
      "Pi/4"
    );
    assert_eq!(
      interpret("Sum[(-1)^(n+1)/(2 n - 1)^2, {n, 1, Infinity}]").unwrap(),
      "Catalan"
    );
    assert_eq!(
      interpret("Sum[(-1)^(n+1)/(2 n - 1)^3, {n, 1, Infinity}]").unwrap(),
      "Pi^3/32"
    );
    // Dirichlet lambda (non-alternating, s >= 2): Pi^2/8, Pi^4/96.
    assert_eq!(
      interpret("Sum[1/(2 n - 1)^2, {n, 1, Infinity}]").unwrap(),
      "Pi^2/8"
    );
    assert_eq!(
      interpret("Sum[1/(2 n - 1)^4, {n, 1, Infinity}]").unwrap(),
      "Pi^4/96"
    );
    // The (2n+1), {n, 0, …} spelling of the same odd integers.
    assert_eq!(
      interpret("Sum[1/(2 n + 1)^2, {n, 0, Infinity}]").unwrap(),
      "Pi^2/8"
    );
    // The divergent harmonic-over-odds (s = 1, non-alternating) is left alone.
    assert_eq!(
      interpret("Sum[1/(2 n - 1), {n, 1, Infinity}]").unwrap(),
      "Sum[(2*n - 1)^(-1), {n, 1, Infinity}]"
    );
  }

  // Sum[1/(a n)^s, {n, 1, Infinity}] = Zeta[s]/a^s for a pure multiple a*n of
  // the index (a numeric or symbolic, s >= 2).
  #[test]
  fn scaled_reciprocal_power_sums() {
    assert_eq!(
      interpret("Sum[1/(2 n)^2, {n, 1, Infinity}]").unwrap(),
      "Pi^2/24"
    );
    assert_eq!(
      interpret("Sum[1/(2 n)^4, {n, 1, Infinity}]").unwrap(),
      "Pi^4/1440"
    );
    assert_eq!(
      interpret("Sum[1/(3 n)^2, {n, 1, Infinity}]").unwrap(),
      "Pi^2/54"
    );
    assert_eq!(
      interpret("Sum[1/(2 n)^3, {n, 1, Infinity}]").unwrap(),
      "Zeta[3]/8"
    );
    // A symbolic coefficient is carried through.
    assert_eq!(
      interpret("Sum[1/(a n)^2, {n, 1, Infinity}]").unwrap(),
      "Pi^2/(6*a^2)"
    );
    // The divergent s = 1 case is left unevaluated.
    assert_eq!(
      interpret("Sum[1/(2 n), {n, 1, Infinity}]").unwrap(),
      "Sum[1/(2*n), {n, 1, Infinity}]"
    );
  }

  // Sum[1/n^s, {n, 1, Infinity}] = Zeta[s] for a symbolic exponent, matching the
  // Riemann-zeta definition. More generally Sum[n^e] = Zeta[-e].
  #[test]
  fn symbolic_exponent_zeta_sums() {
    assert_eq!(
      interpret("Sum[1/n^s, {n, 1, Infinity}]").unwrap(),
      "Zeta[s]"
    );
    assert_eq!(
      interpret("Sum[n^(-s), {n, 1, Infinity}]").unwrap(),
      "Zeta[s]"
    );
    assert_eq!(interpret("Sum[n^s, {n, 1, Infinity}]").unwrap(), "Zeta[-s]");
    assert_eq!(
      interpret("Sum[1/n^(s + 1), {n, 1, Infinity}]").unwrap(),
      "Zeta[1 + s]"
    );
    // Numeric exponents keep their closed forms / divergence behavior.
    assert_eq!(interpret("Sum[1/n^2, {n, 1, Infinity}]").unwrap(), "Pi^2/6");
    assert_eq!(
      interpret("Sum[1/n^3, {n, 1, Infinity}]").unwrap(),
      "Zeta[3]"
    );
  }

  // Sum[Fibonacci[k], {k, a, n}] = Fibonacci[n+2] - Fibonacci[a+1].
  #[test]
  fn fibonacci_partial_sum() {
    assert_eq!(
      interpret("Sum[Fibonacci[k], {k, 1, n}]").unwrap(),
      "-1 + Fibonacci[2 + n]"
    );
    assert_eq!(
      interpret("Sum[Fibonacci[k], {k, 0, n}]").unwrap(),
      "-1 + Fibonacci[2 + n]"
    );
    assert_eq!(
      interpret("Sum[Fibonacci[k], {k, 2, n}]").unwrap(),
      "-2 + Fibonacci[2 + n]"
    );
    // Constant-factor linearity composes with the identity.
    assert_eq!(
      interpret("Sum[2 Fibonacci[k], {k, 1, n}]").unwrap(),
      "2*(-1 + Fibonacci[2 + n])"
    );
    // A concrete upper bound folds to the integer value (Fib[12] - 1 = 143).
    assert_eq!(interpret("Sum[Fibonacci[k], {k, 1, 10}]").unwrap(), "143");
  }

  // Sum[Fibonacci[k]^2, {k, a, n}] = Fibonacci[n] Fibonacci[n+1]
  //                                  - Fibonacci[a-1] Fibonacci[a].
  #[test]
  fn fibonacci_square_partial_sum() {
    assert_eq!(
      interpret("Sum[Fibonacci[k]^2, {k, 1, n}]").unwrap(),
      "Fibonacci[n]*Fibonacci[1 + n]"
    );
    assert_eq!(
      interpret("Sum[Fibonacci[k]^2, {k, 2, n}]").unwrap(),
      "-1 + Fibonacci[n]*Fibonacci[1 + n]"
    );
    // Fib[6] Fib[7] = 8*13 = 104.
    assert_eq!(interpret("Sum[Fibonacci[k]^2, {k, 1, 6}]").unwrap(), "104");
  }

  // Vandermonde: Sum[Binomial[N, k]^2, {k, 0, N}] = Binomial[2 N, N].
  #[test]
  fn binomial_square_sum_vandermonde() {
    assert_eq!(
      interpret("Sum[Binomial[n, k]^2, {k, 0, n}]").unwrap(),
      "Binomial[2*n, n]"
    );
    assert_eq!(
      interpret("Sum[Binomial[m, k]^2, {k, 0, m}]").unwrap(),
      "Binomial[2*m, m]"
    );
    // Binomial[10, 5] = 252.
    assert_eq!(
      interpret("Sum[Binomial[5, k]^2, {k, 0, 5}]").unwrap(),
      "252"
    );
  }

  #[test]
  fn negative_fractional_base() {
    assert_eq!(
      interpret("Sum[(-1/2)^n/n, {n, 1, Infinity}]").unwrap(),
      "-Log[3/2]"
    );
  }

  #[test]
  fn positive_fractional_base() {
    assert_eq!(
      interpret("Sum[(1/2)^n/n, {n, 1, Infinity}]").unwrap(),
      "Log[2]"
    );
  }

  #[test]
  fn symbolic_base_unchanged() {
    assert_eq!(
      interpret("Sum[x^n/n, {n, 1, Infinity}]").unwrap(),
      "-Log[1 - x]"
    );
  }

  // Divergent bases (harmonic r = 1, and |r| > 1) stay unevaluated.
  #[test]
  fn divergent_bases_stay_unevaluated() {
    assert_eq!(
      interpret("Sum[2^n/n, {n, 1, Infinity}]").unwrap(),
      "Sum[2^n/n, {n, 1, Infinity}]"
    );
    assert_eq!(
      interpret("Sum[1/n, {n, 1, Infinity}]").unwrap(),
      "Sum[n^(-1), {n, 1, Infinity}]"
    );
  }

  // Infinite products with provable tail behavior: |term| -> Infinity
  // emits Product::div (n, n^2, Sqrt[n], n/2, 2^n all message), and
  // |term| -> 0 gives 0 (Product[1/n], Product[2/n], Product[2^-n]).
  // Convergent products and symbolic constants are unaffected. All
  // verified against wolframscript 15.0.
  #[test]
  fn product_tail_behavior() {
    for call in [
      "Product[n, {n, 1, Infinity}]",
      "Product[n^2, {n, 1, Infinity}]",
      "Product[Sqrt[n], {n, 1, Infinity}]",
      "Product[n/2, {n, 1, Infinity}]",
      "Product[-n, {n, 1, Infinity}]",
      "Product[2^n, {n, 1, Infinity}]",
    ] {
      clear_state();
      interpret(call).unwrap();
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs
          .iter()
          .any(|m| m.contains("Product::div: Product does not converge.")),
        "expected Product::div for {call}, got {msgs:?}"
      );
    }
    assert_eq!(interpret("Product[1/n, {n, 1, Infinity}]").unwrap(), "0");
    assert_eq!(interpret("Product[2/n, {n, 1, Infinity}]").unwrap(), "0");
    assert_eq!(interpret("Product[1/n^2, {n, 1, Infinity}]").unwrap(), "0");
    assert_eq!(interpret("Product[2^(-n), {n, 1, Infinity}]").unwrap(), "0");
    // Convergent and undecidable cases are untouched
    assert_eq!(
      interpret("Product[1 + 1/n^2, {n, 1, Infinity}]").unwrap(),
      "Sinh[Pi]/Pi"
    );
    clear_state();
    assert_eq!(
      interpret("Product[c, {n, 1, Infinity}]").unwrap(),
      "Product[c, {n, 1, Infinity}]"
    );
    assert!(woxi::get_captured_messages_raw().is_empty());
  }

  // Provably divergent infinite sums emit Sum::div before staying
  // unevaluated: p-series with p <= 1, polynomial growth, constants
  // (including symbolic ones — wolframscript treats them as generically
  // nonzero), and rational functions at the divergence boundary.
  // Exponentials, oscillating terms and Log-mixed terms stay silent,
  // matching wolframscript's own behavior (2^n and Log[n]/n are silent
  // there too). All verified against wolframscript 15.0.
  #[test]
  fn divergent_sums_emit_div_message() {
    for call in [
      "Sum[1/n, {n, 1, Infinity}]",
      "Sum[1, {n, 1, Infinity}]",
      "Sum[n, {n, 1, Infinity}]",
      "Sum[1/Sqrt[n], {n, 1, Infinity}]",
      "Sum[(n + 1)/n, {n, 1, Infinity}]",
      "Sum[n/(n^2 + 1), {n, 1, Infinity}]",
      "Sum[c, {n, 1, Infinity}]",
      "Sum[5/n, {n, 1, Infinity}]",
      "Sum[Log[n], {n, 1, Infinity}]",
      "Sum[n^(3/2), {n, 1, Infinity}]",
      "Sum[1/n, {n, 5, Infinity}]",
      "Sum[1/n + 1/(n + 1), {n, 1, Infinity}]",
    ] {
      clear_state();
      interpret(call).unwrap();
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs
          .iter()
          .any(|m| m.contains("Sum::div: Sum does not converge.")),
        "expected Sum::div for {call}, got {msgs:?}"
      );
    }
    // Silent classes: exponential/oscillating/Log-mixed terms, plus the
    // telescoping sum whose combined form is convergent
    for call in [
      "Sum[2^n, {n, 1, Infinity}]",
      "Sum[(-1)^n, {n, 1, Infinity}]",
      "Sum[Log[n]/n, {n, 1, Infinity}]",
      "Sum[1/n - 1/(n + 1), {n, 1, Infinity}]",
    ] {
      clear_state();
      interpret(call).unwrap();
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        !msgs.iter().any(|m| m.contains("Sum::div")),
        "no Sum::div expected for {call}, got {msgs:?}"
      );
    }
    // The zero term sums to 0 exactly
    assert_eq!(interpret("Sum[0, {n, 1, Infinity}]").unwrap(), "0");
    // Convergent sums are unaffected
    assert_eq!(interpret("Sum[1/n^2, {n, 1, Infinity}]").unwrap(), "Pi^2/6");
  }
}

// Taylor series of the circular / hyperbolic functions:
// Sum[(-1)^n x^(2n+1)/(2n+1)!] = Sin[x] and the Cos / Sinh / Cosh variants,
// including sign, coefficient, index-shift, and exponent-offset spellings.
// All verified against wolframscript 15.0.
mod infinite_factorial_trig_series {
  use super::*;

  #[test]
  fn sin_series() {
    assert_eq!(
      interpret("Sum[(-1)^n / Factorial[2n+1] * x^(2n+1), {n, 0, Infinity}]")
        .unwrap(),
      "Sin[x]"
    );
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n+1)/Factorial[2n+1], {n, 0, Infinity}]")
        .unwrap(),
      "Sin[x]"
    );
  }

  #[test]
  fn cos_sinh_cosh_series() {
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n)/Factorial[2n], {n, 0, Infinity}]").unwrap(),
      "Cos[x]"
    );
    assert_eq!(
      interpret("Sum[x^(2n+1)/Factorial[2n+1], {n, 0, Infinity}]").unwrap(),
      "Sinh[x]"
    );
    assert_eq!(
      interpret("Sum[x^(2n)/Factorial[2n], {n, 0, Infinity}]").unwrap(),
      "Cosh[x]"
    );
  }

  // A (-1)^(n+d) factor with odd d flips the overall sign; a constant
  // coefficient (numeric or symbolic) is carried through outside.
  #[test]
  fn sign_and_coefficient() {
    assert_eq!(
      interpret("Sum[(-1)^(n+1) x^(2n+1)/Factorial[2n+1], {n, 0, Infinity}]")
        .unwrap(),
      "-Sin[x]"
    );
    assert_eq!(
      interpret("Sum[2 (-1)^n x^(2n+1)/Factorial[2n+1], {n, 0, Infinity}]")
        .unwrap(),
      "2*Sin[x]"
    );
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n+1)/(Factorial[2n+1] y), {n, 0, Infinity}]")
        .unwrap(),
      "Sin[x]/y"
    );
  }

  // Without a base power the series is evaluated at 1; a numeric base is
  // kept inside the function (Sin[3], not a decimal).
  #[test]
  fn numeric_base() {
    assert_eq!(
      interpret("Sum[(-1)^n/Factorial[2n+1], {n, 0, Infinity}]").unwrap(),
      "Sin[1]"
    );
    assert_eq!(
      interpret("Sum[(-1)^n 3^(2n+1)/Factorial[2n+1], {n, 0, Infinity}]")
        .unwrap(),
      "Sin[3]"
    );
  }

  // Starting at n = 1, or an offset factorial argument, subtracts the head
  // term; the s^j prefactor sign from re-indexing is distributed into the
  // corrected sum while other coefficients stay factored outside.
  #[test]
  fn shifted_series() {
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n+1)/Factorial[2n+1], {n, 1, Infinity}]")
        .unwrap(),
      "-x + Sin[x]"
    );
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n+3)/Factorial[2n+3], {n, 0, Infinity}]")
        .unwrap(),
      "x - Sin[x]"
    );
    assert_eq!(
      interpret("Sum[x^(2n+2)/Factorial[2n+2], {n, 0, Infinity}]").unwrap(),
      "-1 + Cosh[x]"
    );
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n+2)/Factorial[2n+2], {n, 0, Infinity}]")
        .unwrap(),
      "1 - Cos[x]"
    );
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n)/Factorial[2n], {n, 1, Infinity}]").unwrap(),
      "-1 + Cos[x]"
    );
    assert_eq!(
      interpret("Sum[2 (-1)^n x^(2n+1)/Factorial[2n+1], {n, 1, Infinity}]")
        .unwrap(),
      "2*(-x + Sin[x])"
    );
    assert_eq!(
      interpret("Sum[(-1)^(n+1) x^(2n+1)/Factorial[2n+1], {n, 1, Infinity}]")
        .unwrap(),
      "x - Sin[x]"
    );
  }

  // An exponent offset b' /= b in the base power contributes a plain
  // x^(b'-b) factor.
  #[test]
  fn exponent_offset() {
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n)/Factorial[2n+1], {n, 0, Infinity}]")
        .unwrap(),
      "Sin[x]/x"
    );
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n)/Factorial[2n+1], {n, 1, Infinity}]")
        .unwrap(),
      "(-x + Sin[x])/x"
    );
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n+1)/Factorial[2n], {n, 0, Infinity}]")
        .unwrap(),
      "x*Cos[x]"
    );
  }

  // Head corrections longer than one term are combined over a common
  // denominator, the way wolframscript prints them. The plain exponential
  // series is unaffected.
  #[test]
  fn longer_head_corrections() {
    assert_eq!(
      interpret("Sum[(-1)^n x^(2n+1)/Factorial[2n+1], {n, 2, Infinity}]")
        .unwrap(),
      "(-6*x + x^3 + 6*Sin[x])/6"
    );
    assert_eq!(
      interpret("Sum[x^n/Factorial[n], {n, 0, Infinity}]").unwrap(),
      "E^x"
    );
  }
}

// Geometric series from k = 1: Sum[c r^k, {k, 1, Infinity}] = c r/(1 - r),
// for a numeric ratio r with |r| < 1.
mod infinite_geometric_from_one {
  use super::*;

  #[test]
  fn rational_ratio() {
    assert_eq!(interpret("Sum[(1/2)^k, {k, 1, Infinity}]").unwrap(), "1");
    assert_eq!(interpret("Sum[(1/3)^k, {k, 1, Infinity}]").unwrap(), "1/2");
    assert_eq!(interpret("Sum[(2/3)^k, {k, 1, Infinity}]").unwrap(), "2");
  }

  #[test]
  fn negative_ratio() {
    assert_eq!(
      interpret("Sum[(-1/2)^k, {k, 1, Infinity}]").unwrap(),
      "-1/3"
    );
  }

  #[test]
  fn with_coefficient() {
    assert_eq!(interpret("Sum[3 (1/2)^k, {k, 1, Infinity}]").unwrap(), "3");
  }

  // Divergent (|r| >= 1) numeric ratios stay unevaluated.
  #[test]
  fn divergent_unevaluated() {
    assert_eq!(
      interpret("Sum[(3/2)^k, {k, 1, Infinity}]").unwrap(),
      "Sum[(3/2)^k, {k, 1, Infinity}]"
    );
  }

  // A symbolic ratio yields the formal closed form -(c r^m)/(-1 + r),
  // matching wolframscript.
  #[test]
  fn symbolic_geometric_closed_form() {
    assert_eq!(
      interpret("Sum[x^k, {k, 1, Infinity}]").unwrap(),
      "-(x/(-1 + x))"
    );
    assert_eq!(
      interpret("Sum[2 x^k, {k, 1, Infinity}]").unwrap(),
      "(-2*x)/(-1 + x)"
    );
    assert_eq!(
      interpret("Sum[c x^k, {k, 1, Infinity}]").unwrap(),
      "-((c*x)/(-1 + x))"
    );
    // Lower bound > 1 multiplies the numerator by r^m.
    assert_eq!(
      interpret("Sum[x^k, {k, 3, Infinity}]").unwrap(),
      "-(x^3/(-1 + x))"
    );
  }
}

// Geometric series whose term has an integer-multiple exponent:
// Sum[x^(c n), {n, 0, Infinity}] = 1/(1 - x^c) (ratio x^c, symbolic base).
mod infinite_geometric_exponent_multiple {
  use super::*;

  #[test]
  fn square_and_cube() {
    assert_eq!(
      interpret("Sum[x^(2 n), {n, 0, Infinity}]").unwrap(),
      "(1 - x^2)^(-1)"
    );
    assert_eq!(
      interpret("Sum[x^(3 n), {n, 0, Infinity}]").unwrap(),
      "(1 - x^3)^(-1)"
    );
  }

  #[test]
  fn with_coefficient() {
    assert_eq!(
      interpret("Sum[3 x^(2 n), {n, 0, Infinity}]").unwrap(),
      "3/(1 - x^2)"
    );
  }

  // A symbolic exponent coefficient (a^(k n)) gives ratio a^k; wolframscript
  // renders the closed form with the negated `-1 + a^k` denominator.
  #[test]
  fn symbolic_exponent_coefficient() {
    assert_eq!(
      interpret("Sum[a^(k n), {n, 0, Infinity}]").unwrap(),
      "-(-1 + a^k)^(-1)"
    );
  }
}

// Arithmetico-geometric series Sum[k^p r^k, {k, 1, Infinity}] = PolyLog[-p, r],
// folded to a number for an exact numeric ratio r with |r| < 1.
mod infinite_arith_geometric {
  use super::*;

  #[test]
  fn first_order() {
    // k/2^k (division form) and k (1/2)^k (explicit ratio) both = 2.
    assert_eq!(interpret("Sum[k/2^k, {k, 1, Infinity}]").unwrap(), "2");
    assert_eq!(interpret("Sum[k (1/2)^k, {k, 1, Infinity}]").unwrap(), "2");
    assert_eq!(
      interpret("Sum[k (1/3)^k, {k, 1, Infinity}]").unwrap(),
      "3/4"
    );
  }

  #[test]
  fn higher_order() {
    assert_eq!(interpret("Sum[k^2/3^k, {k, 1, Infinity}]").unwrap(), "3/2");
    assert_eq!(interpret("Sum[k^3/2^k, {k, 1, Infinity}]").unwrap(), "26");
  }

  #[test]
  fn negative_ratio() {
    assert_eq!(
      interpret("Sum[k (-1/2)^k, {k, 1, Infinity}]").unwrap(),
      "-2/9"
    );
  }

  // The lower bound 0 just adds the (zero) k=0 term.
  #[test]
  fn from_zero() {
    assert_eq!(interpret("Sum[k/2^k, {k, 0, Infinity}]").unwrap(), "2");
  }

  // A first-order symbolic ratio gives r/(-1 + r)^2; a divergent ratio
  // (|r| >= 1) has no value and stays unevaluated.
  #[test]
  fn symbolic_and_divergent() {
    assert_eq!(
      interpret("Sum[k x^k, {k, 1, Infinity}]").unwrap(),
      "x/(-1 + x)^2"
    );
    assert_eq!(
      interpret("Sum[k 2^k, {k, 1, Infinity}]").unwrap(),
      "Sum[k*2^k, {k, 1, Infinity}]"
    );
  }
}

mod sum_convergence {
  use super::*;

  #[test]
  fn p_series() {
    assert_eq!(interpret("SumConvergence[1/n^2, n]").unwrap(), "True");
    assert_eq!(interpret("SumConvergence[1/n^3, n]").unwrap(), "True");
    // The harmonic series diverges
    assert_eq!(interpret("SumConvergence[1/n, n]").unwrap(), "False");
    assert_eq!(interpret("SumConvergence[n, n]").unwrap(), "False");
    // Symbolic exponent: condition on the real part
    assert_eq!(interpret("SumConvergence[1/n^p, n]").unwrap(), "Re[p] > 1");
  }

  #[test]
  fn rational_functions() {
    // Converges iff deg(denominator) - deg(numerator) >= 2
    assert_eq!(interpret("SumConvergence[1/(n^2 + 1), n]").unwrap(), "True");
    assert_eq!(interpret("SumConvergence[n/(n^3 + 1), n]").unwrap(), "True");
    assert_eq!(
      interpret("SumConvergence[n/(n^2 + 1), n]").unwrap(),
      "False"
    );
  }

  #[test]
  fn geometric() {
    assert_eq!(interpret("SumConvergence[1/2^n, n]").unwrap(), "True");
    assert_eq!(interpret("SumConvergence[(2/3)^n, n]").unwrap(), "True");
    assert_eq!(interpret("SumConvergence[2^n, n]").unwrap(), "False");
    // Geometric decay dominates polynomial growth
    assert_eq!(interpret("SumConvergence[n/2^n, n]").unwrap(), "True");
    assert_eq!(interpret("SumConvergence[(-3)^n/n, n]").unwrap(), "False");
    // Symbolic base: condition on the absolute value
    assert_eq!(interpret("SumConvergence[x^n, n]").unwrap(), "Abs[x] < 1");
  }

  #[test]
  fn alternating() {
    assert_eq!(interpret("SumConvergence[(-1)^n/n, n]").unwrap(), "True");
    assert_eq!(interpret("SumConvergence[(-1)^n, n]").unwrap(), "False");
  }

  #[test]
  fn factorial_decay() {
    assert_eq!(interpret("SumConvergence[1/n!, n]").unwrap(), "True");
    assert_eq!(interpret("SumConvergence[n^2/n!, n]").unwrap(), "True");
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("SumConvergence[Sin[n], n]").unwrap(),
      "SumConvergence[Sin[n], n]"
    );
  }
}

mod mellin_transform_tests {
  use woxi::interpret;

  // Classical Mellin table entries, all verified against wolframscript.
  #[test]
  fn exponential_entries() {
    assert_eq!(
      interpret("MellinTransform[E^(-x), x, s]").unwrap(),
      "Gamma[s]"
    );
    assert_eq!(
      interpret("MellinTransform[E^(-a x), x, s]").unwrap(),
      "Gamma[s]/a^s"
    );
    assert_eq!(
      interpret("MellinTransform[E^(-3 x), x, s]").unwrap(),
      "Gamma[s]/3^s"
    );
    assert_eq!(
      interpret("MellinTransform[E^(-x^2), x, s]").unwrap(),
      "Gamma[s/2]/2"
    );
    assert_eq!(
      interpret("MellinTransform[E^(-x^3), x, s]").unwrap(),
      "Gamma[s/3]/3"
    );
    assert_eq!(
      interpret("MellinTransform[E^(-a x^2), x, s]").unwrap(),
      "Gamma[s/2]/(2*a^(s/2))"
    );
    // The x^p factor shifts s; constants factor out.
    assert_eq!(
      interpret("MellinTransform[x^2 E^(-x), x, s]").unwrap(),
      "Gamma[2 + s]"
    );
    assert_eq!(
      interpret("MellinTransform[x E^(-2 x), x, s]").unwrap(),
      "2^(-1 - s)*Gamma[1 + s]"
    );
    assert_eq!(
      interpret("MellinTransform[Sqrt[x] E^(-x), x, s]").unwrap(),
      "Gamma[1/2 + s]"
    );
    assert_eq!(
      interpret("MellinTransform[x^a E^(-b x), x, s]").unwrap(),
      "b^(-a - s)*Gamma[a + s]"
    );
    assert_eq!(
      interpret("MellinTransform[2 E^(-x), x, s]").unwrap(),
      "2*Gamma[s]"
    );
    // Divergent integrand stays unevaluated.
    assert_eq!(
      interpret("MellinTransform[E^x, x, s]").unwrap(),
      "MellinTransform[E^x, x, s]"
    );
  }

  #[test]
  fn algebraic_and_special_entries() {
    assert_eq!(
      interpret("MellinTransform[1/(1 + x), x, s]").unwrap(),
      "Pi*Csc[Pi*s]"
    );
    assert_eq!(
      interpret("MellinTransform[1/(a + x), x, s]").unwrap(),
      "a^(-1 + s)*Pi*Csc[Pi*s]"
    );
    assert_eq!(
      interpret("MellinTransform[1/(1 + x^2), x, s]").unwrap(),
      "(Pi*Csc[(Pi*s)/2])/2"
    );
    assert_eq!(
      interpret("MellinTransform[(1 + x)^(-a), x, s]").unwrap(),
      "(Gamma[a - s]*Gamma[s])/Gamma[a]"
    );
    assert_eq!(
      interpret("MellinTransform[1/(1 + x)^2, x, s]").unwrap(),
      "Gamma[2 - s]*Gamma[s]"
    );
    assert_eq!(
      interpret("MellinTransform[Sin[x], x, s]").unwrap(),
      "Gamma[s]*Sin[(Pi*s)/2]"
    );
    assert_eq!(
      interpret("MellinTransform[Cos[x], x, s]").unwrap(),
      "Cos[(Pi*s)/2]*Gamma[s]"
    );
    assert_eq!(
      interpret("MellinTransform[Sin[a x], x, s]").unwrap(),
      "(Gamma[s]*Sin[(Pi*s)/2])/a^s"
    );
    assert_eq!(
      interpret("MellinTransform[Cos[a x], x, s]").unwrap(),
      "(Cos[(Pi*s)/2]*Gamma[s])/a^s"
    );
    assert_eq!(
      interpret("MellinTransform[Log[1 + x], x, s]").unwrap(),
      "(Pi*Csc[Pi*s])/s"
    );
    assert_eq!(
      interpret("MellinTransform[UnitStep[1 - x], x, s]").unwrap(),
      "s^(-1)"
    );
    assert_eq!(
      interpret("MellinTransform[HeavisideTheta[1 - x], x, s]").unwrap(),
      "s^(-1)"
    );
    assert_eq!(
      interpret("MellinTransform[DiracDelta[x - a], x, s]").unwrap(),
      "a^(-1 + s)"
    );
    assert_eq!(
      interpret("MellinTransform[Erfc[x], x, s]").unwrap(),
      "Gamma[1/2 + s/2]/(Sqrt[Pi]*s)"
    );
    assert_eq!(
      interpret("MellinTransform[BesselJ[0, x], x, s]").unwrap(),
      "(2^(-1 + s)*Gamma[s/2])/Gamma[1 - s/2]"
    );
  }

  // Powers of x alone give DiracDelta distributions; expressions free of
  // the variable use the constant rule; unknown integrands stay
  // unevaluated; short calls emit argm/argmu.
  #[test]
  fn distributional_and_edge_cases() {
    assert_eq!(
      interpret("MellinTransform[1, x, s]").unwrap(),
      "2*Pi*DiracDelta[I*s]"
    );
    assert_eq!(
      interpret("MellinTransform[x, x, s]").unwrap(),
      "2*Pi*DiracDelta[I*(1 + s)]"
    );
    assert_eq!(
      interpret("MellinTransform[E^(-x), 5, s]").unwrap(),
      "(2*Pi*DiracDelta[I*s])/E^x"
    );
    assert_eq!(
      interpret("MellinTransform[f[x], x, s]").unwrap(),
      "MellinTransform[f[x], x, s]"
    );

    let r = woxi::interpret_with_stdout("MellinTransform[E^(-x), x]").unwrap();
    assert_eq!(r.result, "MellinTransform[E^(-x), x]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "MellinTransform::argm: MellinTransform called with 2 arguments; 3 or more arguments are expected."
    )));

    let r = woxi::interpret_with_stdout("MellinTransform[E^(-x)]").unwrap();
    assert_eq!(r.result, "MellinTransform[E^(-x)]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "MellinTransform::argmu: MellinTransform called with 1 argument; 3 or more arguments are expected."
    )));
  }
}

mod inverse_fourier_sin_cos_tests {
  use woxi::interpret;

  // With default parameters the sine/cosine transforms are involutions,
  // so the inverse is the same operator. All verified against
  // wolframscript.
  #[test]
  fn inverse_transforms() {
    assert_eq!(
      interpret("InverseFourierSinTransform[E^(-w), w, t]").unwrap(),
      "(Sqrt[2/Pi]*t)/(1 + t^2)"
    );
    assert_eq!(
      interpret("InverseFourierCosTransform[E^(-w), w, t]").unwrap(),
      "Sqrt[2/Pi]/(1 + t^2)"
    );
    assert_eq!(
      interpret("InverseFourierSinTransform[1/w, w, t]").unwrap(),
      "Sqrt[Pi/2]*Sign[t]"
    );
    assert_eq!(
      interpret("InverseFourierCosTransform[1/(1 + w^2), w, t]").unwrap(),
      "Sqrt[Pi/2]/E^t"
    );
    assert_eq!(
      interpret("InverseFourierSinTransform[w E^(-w^2), w, t]").unwrap(),
      "t/(2*Sqrt[2]*E^(t^2/4))"
    );
    assert_eq!(
      interpret("InverseFourierCosTransform[E^(-w^2), w, t]").unwrap(),
      "1/(Sqrt[2]*E^(t^2/4))"
    );
    // Unknown integrands echo the Inverse head, not the forward one.
    assert_eq!(
      interpret("InverseFourierSinTransform[f[w], w, t]").unwrap(),
      "InverseFourierSinTransform[f[w], w, t]"
    );
  }

  // Forward-table entries added alongside (and one regression: 1/t was
  // missing the Sign[w] factor).
  #[test]
  fn forward_table_additions() {
    assert_eq!(
      interpret("FourierSinTransform[1/t, t, w]").unwrap(),
      "Sqrt[Pi/2]*Sign[w]"
    );
    assert_eq!(
      interpret("FourierSinTransform[1/t, t, 2]").unwrap(),
      "Sqrt[Pi/2]"
    );
    assert_eq!(
      interpret("FourierSinTransform[1/t, t, -2]").unwrap(),
      "-Sqrt[Pi/2]"
    );
    assert_eq!(
      interpret("FourierCosTransform[1/(1 + t^2), t, w]").unwrap(),
      "Sqrt[Pi/2]/E^w"
    );
    assert_eq!(
      interpret("FourierCosTransform[1/(a^2 + t^2), t, w]").unwrap(),
      "(Sqrt[a^(-2)]*Sqrt[Pi/2])/E^(w/Sqrt[a^(-2)])"
    );
    assert_eq!(
      interpret("FourierSinTransform[t/(1 + t^2), t, w]").unwrap(),
      "Sqrt[Pi/2]/E^w"
    );
    assert_eq!(
      interpret("FourierSinTransform[t/(a^2 + t^2), t, w]").unwrap(),
      "Sqrt[Pi/2]/E^(w/Sqrt[a^(-2)])"
    );
    assert_eq!(
      interpret("FourierSinTransform[t E^(-t^2), t, w]").unwrap(),
      "w/(2*Sqrt[2]*E^(w^2/4))"
    );
    assert_eq!(
      interpret("FourierSinTransform[t E^(-a t^2), t, w]").unwrap(),
      "w/(2*Sqrt[2]*a^(3/2)*E^(w^2/(4*a)))"
    );
    // Radical coefficients still fold through the linearity rule.
    assert_eq!(
      interpret("FourierSinTransform[Sqrt[3] t/(1 + t^2), t, w]").unwrap(),
      "Sqrt[(3*Pi)/2]/E^w"
    );
  }

  #[test]
  fn argument_counts() {
    let r = woxi::interpret_with_stdout("InverseFourierSinTransform[E^(-w)]")
      .unwrap();
    assert_eq!(r.result, "InverseFourierSinTransform[E^(-w)]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "InverseFourierSinTransform::argmu: InverseFourierSinTransform called with 1 argument; 3 or more arguments are expected."
    )));
  }
}

mod inverse_mellin_transform_tests {
  use woxi::interpret;

  // The reverse Mellin table, matching wolframscript's result forms
  // (including deliberately unsimplified ones like Pi/(Pi + Pi*x)).
  #[test]
  fn reverse_table() {
    assert_eq!(
      interpret("InverseMellinTransform[Gamma[s], s, x]").unwrap(),
      "E^(-x)"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Gamma[s]/a^s, s, x]").unwrap(),
      "E^(-(a*x))"
    );
    assert_eq!(
      interpret("InverseMellinTransform[2 Gamma[s], s, x]").unwrap(),
      "2/E^x"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Gamma[s + 2], s, x]").unwrap(),
      "x^2/E^x"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Gamma[s/2]/2, s, x]").unwrap(),
      "E^(-x^2)"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Gamma[s] Sin[Pi s/2], s, x]").unwrap(),
      "Sin[x]"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Gamma[s] Cos[Pi s/2], s, x]").unwrap(),
      "Cos[x]"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Pi Csc[Pi s], s, x]").unwrap(),
      "Pi/(Pi + Pi*x)"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Pi Csc[Pi s]/s, s, x]").unwrap(),
      "Log[1 + x^(-1)]"
    );
    assert_eq!(
      interpret("InverseMellinTransform[1/s, s, x]").unwrap(),
      "HeavisideTheta[1 - x]"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Gamma[s]^2, s, x]").unwrap(),
      "2*BesselK[0, 2*Sqrt[x]]"
    );
    assert_eq!(
      interpret("InverseMellinTransform[Gamma[a - s] Gamma[s]/Gamma[a], s, x]")
        .unwrap(),
      "(1 + x)^(-a)"
    );
  }

  // Unknown transforms stay unevaluated; short calls emit argmu.
  #[test]
  fn edge_cases() {
    assert_eq!(
      interpret("InverseMellinTransform[f[s], s, x]").unwrap(),
      "InverseMellinTransform[f[s], s, x]"
    );
    let r =
      woxi::interpret_with_stdout("InverseMellinTransform[Gamma[s]]").unwrap();
    assert_eq!(r.result, "InverseMellinTransform[Gamma[s]]");
    assert!(r.warnings.iter().any(|w| w.contains(
      "InverseMellinTransform::argmu: InverseMellinTransform called with 1 argument; 3 or more arguments are expected."
    )));
  }
}

mod interpolation_exact_and_derivative {
  use super::*;

  // An exact query point over exact data interpolates exactly rather than
  // dropping to machine precision or staying unevaluated.
  #[test]
  fn exact_points_give_exact_values() {
    assert_eq!(interpret("Interpolation[{1, 4, 9}][5/2]").unwrap(), "25/4");
    assert_eq!(
      interpret("Interpolation[{{0, 0}, {1, 1}, {2, 4}}][3/2]").unwrap(),
      "9/4"
    );
    assert_eq!(
      interpret("Interpolation[{1, 4, 9, 16}][7/2]").unwrap(),
      "49/4"
    );
    assert_eq!(
      interpret("Interpolation[{0, 1, 8, 27}][5/2]").unwrap(),
      "27/8"
    );
  }

  // Rational data stays rational.
  #[test]
  fn rational_data_stays_rational() {
    assert_eq!(
      interpret("Interpolation[{1/2, 3/2, 7/2}][3/2]").unwrap(),
      "7/8"
    );
    assert_eq!(
      interpret("Interpolation[{{0, 1/2}, {1, 3/2}, {2, 7/2}}][1/2]").unwrap(),
      "7/8"
    );
  }

  // Machine data keeps giving machine values, whatever the query point.
  #[test]
  fn machine_data_gives_machine_values() {
    assert_eq!(
      interpret("Interpolation[{1., 4., 9.}][3/2]").unwrap(),
      "2.25"
    );
    assert_eq!(interpret("Interpolation[{1, 4., 9}][3/2]").unwrap(), "2.25");
    assert_eq!(
      interpret("Interpolation[{{0, 0}, {1, 1}, {2, 4}}][1.5]").unwrap(),
      "2.25"
    );
  }

  // The interpolation order picks the stencil, so a lower order gives the
  // straight-line value between the neighbouring points.
  #[test]
  fn interpolation_order_selects_the_stencil() {
    assert_eq!(
      interpret("Interpolation[{1, 4, 9}, InterpolationOrder -> 1][3/2]")
        .unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("Interpolation[{0, 1, 0, 1, 0}, InterpolationOrder -> 2][3/2]")
        .unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("Interpolation[{0, 1, 0, 1, 0}][3/2]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Interpolation[{0, 1, 0, 1, 0}][7/2]").unwrap(),
      "1/2"
    );
    assert_eq!(
      interpret("Interpolation[{0, 1, 0, 1, 0}][9/2]").unwrap(),
      "1"
    );
  }

  // Outside the data range the boundary polynomial is extended rather than
  // the value being clamped to the last data point.
  #[test]
  fn outside_the_range_extrapolates() {
    assert_eq!(
      interpret("Interpolation[{{0, 0}, {1, 1}, {2, 4}}][3]").unwrap(),
      "9"
    );
    assert_eq!(
      interpret("Interpolation[{{0, 0}, {1, 1}, {2, 4}}][5]").unwrap(),
      "25"
    );
    assert_eq!(
      interpret("Interpolation[{{0, 0}, {1, 1}, {2, 4}}][-1]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Interpolation[{{0, 0}, {1, 1}, {2, 4}}][3.]").unwrap(),
      "9."
    );
    assert_eq!(interpret("Interpolation[{1, 4, 9}][1/2]").unwrap(), "1/4");
  }

  #[test]
  fn extrapolation_warns() {
    let r = woxi::interpret_with_stdout("Interpolation[{1, 4, 9}][5]").unwrap();
    assert_eq!(r.result, "25");
    assert!(
      r.warnings.iter().any(|w| w.contains(
        "InterpolatingFunction::dmval: Input value {5} lies outside the range of data in the interpolating function. Extrapolation will be used."
      )),
      "expected dmval, got {:?}",
      r.warnings
    );
  }

  // f' differentiates the local polynomial piece and stays an interpolating
  // function.
  #[test]
  fn derivative_of_an_interpolating_function() {
    assert_eq!(
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}]; f'[1]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}]; f'[3/2]")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}]; f''[1]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}]; f'[1.]").unwrap(),
      "2."
    );
    assert_eq!(
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}]; Head[f']")
        .unwrap(),
      "InterpolatingFunction"
    );
    assert_eq!(interpret("Interpolation[{5}]'[2]").unwrap(), "0");
  }

  #[test]
  fn higher_derivatives_run_out_of_polynomial() {
    assert_eq!(
      interpret("Interpolation[{1, 4, 9, 16}]'[5/2]").unwrap(),
      "5"
    );
    assert_eq!(
      interpret("Interpolation[{1, 4, 9, 16}]''[5/2]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Interpolation[{1, 4, 9, 16}]'''[5/2]").unwrap(),
      "0"
    );
  }

  // The derivative order shows up as the Derivative operator itself, and the
  // grid metadata carries over.
  #[test]
  fn derivative_order_property() {
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]["DerivativeOrder"]"#).unwrap(),
      "0"
    );
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]'["DerivativeOrder"]"#).unwrap(),
      "Derivative[1]"
    );
    assert_eq!(
      interpret(r#"Interpolation[{1, 4, 9, 16}]'["Domain"]"#).unwrap(),
      "{{1, 4}}"
    );
  }

  // D differentiates through a compound head the same way it does through a
  // symbol head.
  #[test]
  fn d_through_a_compound_head() {
    assert_eq!(
      interpret(
        "f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}]; D[f[x], x] /. x -> 1"
      )
      .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("D[g[1][x], x]").unwrap(),
      "Derivative[1][g[1]][x]"
    );
  }

  // A prime may follow a bracketed call and be applied to arguments.
  #[test]
  fn prime_after_a_bracketed_call() {
    assert_eq!(
      interpret("Interpolation[{{0, 0}, {1, 1}, {2, 4}}]'[1]").unwrap(),
      "2"
    );
    assert_eq!(interpret("Interpolation[{1, 4, 9}]''[2]").unwrap(), "2");
    assert_eq!(interpret("g[1]'[2]").unwrap(), "Derivative[1][g[1]][2]");
    assert_eq!(interpret("Sin[x]'").unwrap(), "Derivative[1][Sin[x]]");
  }

  // The 2-D grid form takes its coordinates either as two arguments or as
  // one list, and interpolates exactly over an exact grid.
  #[test]
  fn two_dimensional_grid() {
    assert_eq!(
      interpret("ListInterpolation[{{1, 2}, {3, 4}}][{3/2, 3/2}]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("ListInterpolation[{{1, 2}, {3, 4}}][3/2, 3/2]").unwrap(),
      "5/2"
    );
    assert_eq!(
      interpret("ListInterpolation[{{1, 2}, {3, 4}}][{1.5, 1.5}]").unwrap(),
      "2.5"
    );
    assert_eq!(
      interpret("ListInterpolation[{{1, 2}, {3, 4}}][{2, 1}]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret(
        "ListInterpolation[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}][{3/2, 5/2}]"
      )
      .unwrap(),
      "253/64"
    );
  }
}

/// `∫ Log[a x] dx` and the negative-exponent rendering that shows up in
/// antiderivatives.
mod integrate_log_scaled {
  use super::*;

  #[test]
  fn a_scaled_logarithm_integrates() {
    clear_state();
    // The scale does not factor out the way the plain logarithm's x does.
    assert_eq!(
      interpret("ToString[Integrate[Log[2 x], x], InputForm]").unwrap(),
      "-x + x*Log[2*x]"
    );
    assert_eq!(
      interpret("ToString[Simplify[D[Integrate[Log[2 x], x], x]], InputForm]")
        .unwrap(),
      "Log[2*x]"
    );
  }

  #[test]
  fn a_negative_exponent_keeps_its_power_form_after_a_minus() {
    clear_state();
    for (code, expected) in [
      ("ToString[Integrate[E^(-x), x], InputForm]", "-E^(-x)"),
      ("ToString[-E^(-x), InputForm]", "-E^(-x)"),
      ("ToString[-2^(-x), InputForm]", "-2^(-x)"),
      ("ToString[-a^(-x), InputForm]", "-a^(-x)"),
      ("ToString[-x^(-y), InputForm]", "-x^(-y)"),
      // More than the -1 coefficient, and the reciprocal form is the one
      // wolframscript writes.
      ("ToString[-2 E^(-x), InputForm]", "-2/E^x"),
      ("ToString[a E^(-x), InputForm]", "a/E^x"),
      ("ToString[-E^(-x) y, InputForm]", "-(y/E^x)"),
    ] {
      assert_eq!(interpret(code).unwrap(), expected, "{code}");
    }
  }

  #[test]
  fn a_logarithmic_antiderivative_keeps_its_common_factor() {
    clear_state();
    for (code, expected) in [
      ("Integrate[x*Log[x], x]", "(x^2*(-1 + 2*Log[x]))/4"),
      ("Integrate[x^2*Log[x], x]", "(x^3*(-1 + 3*Log[x]))/9"),
      // A sum with nothing logarithmic in it stays as it was.
      ("Integrate[x^3 + 2 x, x]", "x^2 + x^4/4"),
      ("Integrate[x/(x + 1), x]", "x - Log[1 + x]"),
    ] {
      assert_eq!(
        interpret(&format!("ToString[{code}, InputForm]")).unwrap(),
        expected,
        "{code}"
      );
    }
  }
}

// CaputoD — the Caputo fractional differintegral. For a power the answer is a
// single Gamma ratio, and the operator is linear, so a polynomial goes term by
// term. Values verified against wolframscript.
mod caputo_d {
  use super::*;

  /// The result of `code`, written the way `InputForm` writes it.
  fn form(code: &str) -> String {
    interpret(&format!("ToString[{code}, InputForm]")).unwrap()
  }

  #[test]
  fn a_power_gives_a_gamma_ratio() {
    clear_state();
    for (code, expected) in [
      ("CaputoD[t^2, {t, 1/2}]", "(8*t^(3/2))/(3*Sqrt[Pi])"),
      ("CaputoD[t^3, {t, 1/2}]", "(16*t^(5/2))/(5*Sqrt[Pi])"),
      ("CaputoD[t, {t, 1/2}]", "(2*Sqrt[t])/Sqrt[Pi]"),
      ("CaputoD[t^4, {t, 1/2}]", "(128*t^(7/2))/(35*Sqrt[Pi])"),
      (
        "CaputoD[t^10, {t, 1/2}]",
        "(262144*t^(19/2))/(46189*Sqrt[Pi])",
      ),
      // An order past one differentiates further before integrating back.
      ("CaputoD[t^2, {t, 3/2}]", "(4*Sqrt[t])/Sqrt[Pi]"),
      ("CaputoD[t^5, {t, 5/2}]", "(64*t^(5/2))/Sqrt[Pi]"),
      // A Gamma that does not reduce is left standing.
      ("CaputoD[t^2, {t, 1/3}]", "(2*t^(5/3))/Gamma[8/3]"),
      (
        "CaputoD[t^(2/3), {t, 1/2}]",
        "(t^(1/6)*Gamma[5/3])/Gamma[7/6]",
      ),
      // The power need not be whole.
      ("CaputoD[t^(1/2), {t, 1/2}]", "Sqrt[Pi]/2"),
      ("CaputoD[t^(3/2), {t, 1/2}]", "(3*Sqrt[Pi]*t)/4"),
      // A whole order is the ordinary derivative.
      ("CaputoD[t^2, {t, 1}]", "2*t"),
      ("CaputoD[t^2, {t, 2}]", "2"),
      ("CaputoD[t^3, {t, 3}]", "6"),
      // Order zero asks for nothing.
      ("CaputoD[t^2, {t, 0}]", "t^2"),
      // A negative order integrates instead.
      ("CaputoD[t^2, {t, -1}]", "t^3/3"),
      ("CaputoD[t^2, {t, -2}]", "t^4/12"),
      ("CaputoD[t^2, {t, -1/2}]", "(16*t^(5/2))/(15*Sqrt[Pi])"),
      ("CaputoD[1, {t, -1/2}]", "(2*Sqrt[t])/Sqrt[Pi]"),
    ] {
      assert_eq!(form(code), expected, "{code}");
    }
  }

  // The property Caputo is chosen for: a constant differentiates away, where
  // Riemann–Liouville would leave a `t^(-α)` behind.
  #[test]
  fn a_constant_vanishes_under_a_positive_order() {
    clear_state();
    for code in [
      "CaputoD[1, {t, 1/2}]",
      "CaputoD[2, {t, 1/2}]",
      "CaputoD[c, {t, 1/2}]",
      "CaputoD[1, {t, 1}]",
      "CaputoD[0, {t, 1/2}]",
      // Nothing in the expression depends on the variable asked about.
      "CaputoD[t^2, {s, 1/2}]",
    ] {
      assert_eq!(interpret(code).unwrap(), "0", "{code}");
    }
  }

  #[test]
  fn the_operator_is_linear_over_the_expanded_input() {
    clear_state();
    for (code, expected) in [
      (
        "CaputoD[t^2 + t, {t, 1/2}]",
        "(2*Sqrt[t])/Sqrt[Pi] + (8*t^(3/2))/(3*Sqrt[Pi])",
      ),
      ("CaputoD[3 t^2, {t, 1/2}]", "(8*t^(3/2))/Sqrt[Pi]"),
      ("CaputoD[t^2/3, {t, 1/2}]", "(8*t^(3/2))/(9*Sqrt[Pi])"),
      ("CaputoD[-t^2, {t, 1/2}]", "(-8*t^(3/2))/(3*Sqrt[Pi])"),
      // A symbolic factor rides along as a coefficient.
      ("CaputoD[c t^2, {t, 1/2}]", "(8*c*t^(3/2))/(3*Sqrt[Pi])"),
      ("CaputoD[t^2 u, {t, 1/2}]", "(8*t^(3/2)*u)/(3*Sqrt[Pi])"),
      (
        "CaputoD[a t^2 + b t + c, {t, 1/2}]",
        "(2*b*Sqrt[t])/Sqrt[Pi] + (8*a*t^(3/2))/(3*Sqrt[Pi])",
      ),
      // The input is expanded first.
      (
        "CaputoD[(t + 1)^2, {t, 1/2}]",
        "(4*Sqrt[t])/Sqrt[Pi] + (8*t^(3/2))/(3*Sqrt[Pi])",
      ),
      (
        "CaputoD[5 t^3 + 2 t, {t, 1/2}]",
        "(4*Sqrt[t])/Sqrt[Pi] + (16*t^(5/2))/Sqrt[Pi]",
      ),
    ] {
      assert_eq!(form(code), expected, "{code}");
    }
  }

  // A symbolic order stands for the plain Gamma ratio only for a whole power
  // of at least two; below that wolframscript splits on the order's sign,
  // which is left unevaluated here.
  #[test]
  fn a_symbolic_order_covers_the_whole_powers() {
    clear_state();
    for (code, expected) in [
      (
        "CaputoD[t^2, {t, alpha}]",
        "(2*t^(2 - alpha))/Gamma[3 - alpha]",
      ),
      (
        "CaputoD[t^3, {t, alpha}]",
        "(6*t^(3 - alpha))/Gamma[4 - alpha]",
      ),
      (
        "CaputoD[t^5, {t, alpha}]",
        "(120*t^(5 - alpha))/Gamma[6 - alpha]",
      ),
    ] {
      assert_eq!(form(code), expected, "{code}");
    }
  }

  // A power the integration step cannot reach reports ::sing; one the
  // differentiation step already flattens is left standing, as wolframscript
  // leaves it.
  #[test]
  fn orders_the_integral_cannot_reach_are_reported() {
    clear_state();
    for code in [
      "CaputoD[t^(-1/2), {t, 1/2}]",
      "CaputoD[t^(-1/4), {t, 1/2}]",
      "CaputoD[t^(1/2), {t, 3/2}]",
      "CaputoD[t^(1/2), {t, 2}]",
      "CaputoD[t^(3/2), {t, 5/2}]",
      "CaputoD[t^(1/4), {t, 3/2}]",
    ] {
      let result = interpret_with_stdout(code).unwrap();
      assert!(
        result.warnings.iter().any(|w| w.contains("CaputoD::sing")),
        "expected ::sing for {code}, got {:?}",
        result.warnings
      );
    }
    // A whole power below the order, and a specification that is not one.
    for (code, expected) in [
      ("CaputoD[t^2, {t, 3}]", "CaputoD[t^2, {t, 3}]"),
      ("CaputoD[t, {t, 2}]", "CaputoD[t, {t, 2}]"),
      ("CaputoD[t^2, {t, 5/2}]", "CaputoD[t^2, {t, 5/2}]"),
      ("CaputoD[t, {t, 3/2}]", "CaputoD[t, {t, 3/2}]"),
      ("CaputoD[f[t], {t, 1/2}]", "CaputoD[f[t], {t, 1/2}]"),
      ("CaputoD[t^2, {t}]", "CaputoD[t^2, {t}]"),
      ("CaputoD[t^2, t]", "CaputoD[t^2, t]"),
    ] {
      assert_eq!(form(code), expected, "{code}");
    }
  }

  // A power that stays exact where the integral converges, including below
  // the axis when the order integrates rather than differentiates.
  #[test]
  fn a_negative_order_reaches_powers_a_positive_one_cannot() {
    clear_state();
    for (code, expected) in [
      ("CaputoD[t^(-1/2), {t, -1/2}]", "Sqrt[Pi]"),
      ("CaputoD[t^(-1/2), {t, -1}]", "2*Sqrt[t]"),
      ("CaputoD[t^(-3/2), {t, -1/2}]", "0"),
      ("CaputoD[t, {t, -1/2}]", "(4*t^(3/2))/(3*Sqrt[Pi])"),
    ] {
      assert_eq!(form(code), expected, "{code}");
    }
  }
}

mod bounded_oscillation_extrema {
  use super::*;

  #[test]
  fn abs_of_a_bounded_oscillation_sweeps_zero_to_one() {
    // |Sin[x]| oscillates over [0, 1], so its liminf is 0 rather than the -1
    // of the unwrapped Sin.
    assert_eq!(
      interpret("MaxLimit[Abs[Sin[x]], x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("MinLimit[Abs[Sin[x]], x -> Infinity]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("MaxLimit[Abs[Cos[x]], x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("MinLimit[Abs[Cos[x]], x -> Infinity]").unwrap(),
      "0"
    );
  }

  #[test]
  fn reciprocal_of_a_bounded_oscillation_is_unbounded_above() {
    // 1/|Sin[x]| comes arbitrarily close to 1 and grows without bound.
    assert_eq!(
      interpret("MaxLimit[1/Abs[Sin[x]], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("MinLimit[1/Abs[Sin[x]], x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("MaxLimit[Abs[1/Sin[x]], x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("MaxLimit[1/Abs[Cos[x]], x -> Infinity]").unwrap(),
      "Infinity"
    );
  }
}

mod asymptotic_comparisons {
  use super::*;

  #[test]
  fn little_o_ordering_of_powers() {
    assert_eq!(
      interpret("AsymptoticLess[x, x^2, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[x^2, x, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticLess[x^2, x^3, x -> Infinity]").unwrap(),
      "True"
    );
    // Near 0 the ordering flips: the higher power is the smaller one.
    assert_eq!(
      interpret("AsymptoticLess[x, x^2, x -> 0]").unwrap(),
      "False"
    );
    assert_eq!(interpret("AsymptoticLess[x^2, x, x -> 0]").unwrap(), "True");
    assert_eq!(
      interpret("AsymptoticLess[1/x, 1/x^2, x -> 0]").unwrap(),
      "True"
    );
    // A function is never little-o of itself.
    assert_eq!(
      interpret("AsymptoticLess[x, x, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticGreater[x, x, x -> Infinity]").unwrap(),
      "False"
    );
  }

  #[test]
  fn little_o_across_growth_classes() {
    assert_eq!(
      interpret("AsymptoticLess[Log[x], x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[x, Exp[x], x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[Exp[-x], 1/x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[1, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[1/x, 1, x -> Infinity]").unwrap(),
      "True"
    );
    // Two constants are the same order, so neither is little-o of the other.
    assert_eq!(
      interpret("AsymptoticLess[2, 3, x -> Infinity]").unwrap(),
      "False"
    );
    // A limit point other than infinity is honoured.
    assert_eq!(
      interpret("AsymptoticLess[x, x^2, x -> 1]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticLess[x, x^2, x -> -Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[Sin[x], x, x -> 0]").unwrap(),
      "False"
    );
  }

  #[test]
  fn greater_is_less_with_the_arguments_swapped() {
    assert_eq!(
      interpret("AsymptoticGreater[x^2, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticGreater[x, x^2, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticGreater[x, Log[x], x -> Infinity]").unwrap(),
      "True"
    );
  }

  #[test]
  fn big_o_allows_the_same_order() {
    assert_eq!(
      interpret("AsymptoticLessEqual[x, x^2, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLessEqual[x, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLessEqual[x^2, x, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticLessEqual[2, 3, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticGreaterEqual[x, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticGreaterEqual[x, x^2, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticGreaterEqual[x^2, x, x -> Infinity]").unwrap(),
      "True"
    );
  }

  #[test]
  fn big_o_tolerates_bounded_oscillation() {
    // Sin[x] has no limit, but it stays bounded, so it is O(1) without being
    // o(1) — and x Sin[x] is O(x) without being Theta[x].
    assert_eq!(
      interpret("AsymptoticLess[Sin[x], 1, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticLessEqual[Sin[x], 1, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLessEqual[Cos[x], 1, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEqual[Sin[x], 1, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticLessEqual[x*Sin[x], x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[x*Sin[x], x^2, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEqual[x*Sin[x], x, x -> Infinity]").unwrap(),
      "False"
    );
  }

  #[test]
  fn theta_ignores_constant_factors_and_lower_order_terms() {
    assert_eq!(
      interpret("AsymptoticEqual[x, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEqual[2 x, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEqual[3 x + 1, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEqual[x^2 + x, x^2, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEqual[x, x^2, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticEqual[2, 3, x -> Infinity]").unwrap(),
      "True"
    );
  }

  #[test]
  fn equivalence_needs_the_ratio_to_reach_one() {
    // Unlike Theta, a constant factor breaks asymptotic equivalence.
    assert_eq!(
      interpret("AsymptoticEquivalent[x, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEquivalent[2 x, x, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticEquivalent[2, 3, x -> Infinity]").unwrap(),
      "False"
    );
    // A lower-order additive term does not.
    assert_eq!(
      interpret("AsymptoticEquivalent[x + 1, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEquivalent[x^2 + x, x^2, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEquivalent[Sin[x], x, x -> 0]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEquivalent[Sin[x], x, x -> Infinity]").unwrap(),
      "False"
    );
  }

  #[test]
  fn zero_arguments() {
    // 0/0 satisfies every comparison; a zero divisor satisfies none of the
    // "at most" ones.
    assert_eq!(
      interpret("AsymptoticLess[0, 0, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEqual[0, 0, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticEquivalent[0, 0, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[0, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLessEqual[0, x, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticLess[x, 0, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticLessEqual[x, 0, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticEqual[x, 0, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticEquivalent[x, 0, x -> Infinity]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AsymptoticGreater[x, 0, x -> Infinity]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AsymptoticGreaterEqual[x, 0, x -> Infinity]").unwrap(),
      "True"
    );
  }

  #[test]
  fn options_are_accepted_as_a_fourth_argument() {
    assert_eq!(
      interpret("AsymptoticLess[x, x^2, x -> Infinity, Assumptions -> True]")
        .unwrap(),
      "True"
    );
    // A fourth argument that is not an option leaves the call alone.
    assert_eq!(
      interpret("AsymptoticLess[x, x^2, x -> Infinity, 5]").unwrap(),
      "AsymptoticLess[x, x^2, x -> Infinity, 5]"
    );
  }

  #[test]
  fn bad_limit_specification_and_argument_count() {
    assert_eq!(
      interpret("AsymptoticLess[x, x^2, x]").unwrap(),
      "AsymptoticLess[x, x^2, x]"
    );
    assert_eq!(
      interpret("AsymptoticLess[x, x^2, {x -> Infinity}]").unwrap(),
      "AsymptoticLess[x, x^2, {x -> Infinity}]"
    );
    assert_eq!(
      interpret("AsymptoticLess[x, x^2]").unwrap(),
      "AsymptoticLess[x, x^2]"
    );
  }

  #[test]
  fn undecided_forms_stay_unevaluated() {
    // The multivariate form and parametric answers (which wolframscript
    // reports as a ConditionalExpression) are out of scope, and so is any
    // comparison whose underlying limit does not resolve.
    assert_eq!(
      interpret(
        "AsymptoticLess[x + y, x^2 + y^2, {x, y} -> {Infinity, Infinity}]"
      )
      .unwrap(),
      "AsymptoticLess[x + y, x^2 + y^2, {x, y} -> {Infinity, Infinity}]"
    );
    assert_eq!(
      interpret("AsymptoticLess[x^a, x^2, x -> Infinity]").unwrap(),
      "AsymptoticLess[x^a, x^2, x -> Infinity]"
    );
  }
}

mod limits_of_power_sums_at_infinity {
  use super::*;

  #[test]
  fn fractional_powers_in_a_sum() {
    // The approach to these limits is only O(x^(-1/2)) or O(x^(-1/3)), far too
    // slow for a numeric probe at x = 10^7 to recognize; they are decided from
    // the leading exponent instead.
    assert_eq!(
      interpret("Limit[x/(x + Sqrt[x]), x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[(x + Sqrt[x])/x, x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[x/(x + x^(2/3)), x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[Sqrt[x]/(Sqrt[x] + 1), x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[(Sqrt[x] + 1)/(Sqrt[x] + 2), x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[(x + Sqrt[x])/(x - Sqrt[x]), x -> Infinity]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret(
        "Limit[(x^(1/2) + x^(1/3))/(x^(1/2) - x^(1/3)), x -> Infinity]"
      )
      .unwrap(),
      "1"
    );
  }

  #[test]
  fn leading_coefficient_ratio() {
    assert_eq!(
      interpret("Limit[(2 x + 3)/(x + Sqrt[x]), x -> Infinity]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("Limit[(3 x^(3/2) + x)/(2 x^(3/2) - 1), x -> Infinity]")
        .unwrap(),
      "3/2"
    );
    assert_eq!(
      interpret("Limit[(2 x + 1)/(3 x - 1), x -> Infinity]").unwrap(),
      "2/3"
    );
    assert_eq!(
      interpret("Limit[(x^2 + 1)/(2 x^2 - 3), x -> Infinity]").unwrap(),
      "1/2"
    );
  }

  #[test]
  fn unequal_orders_give_zero_or_infinity() {
    assert_eq!(
      interpret("Limit[Sqrt[x]/(x + 1), x -> Infinity]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Limit[(x + 1)/(x^2 + Sqrt[x]), x -> Infinity]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Limit[1/(Sqrt[x] + 1), x -> Infinity]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Limit[(x^2 + Sqrt[x])/(x + 1), x -> Infinity]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("Limit[Sqrt[x] + 1, x -> Infinity]").unwrap(),
      "Infinity"
    );
    // The sign comes from the leading coefficient, not from the lower terms.
    assert_eq!(
      interpret("Limit[(-x^2 + x)/(x + 1), x -> Infinity]").unwrap(),
      "-Infinity"
    );
    assert_eq!(
      interpret("Limit[(1 - x)/Sqrt[x], x -> Infinity]").unwrap(),
      "-Infinity"
    );
    assert_eq!(
      interpret("Limit[(2 - x^3)/(x^2 + 1), x -> Infinity]").unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn symbolic_leading_coefficients() {
    // A symbolic coefficient survives into the answer instead of being
    // silently dropped.
    assert_eq!(
      interpret("Limit[(a x + 1)/(x + 1), x -> Infinity]").unwrap(),
      "a"
    );
    assert_eq!(interpret("Limit[(a x + 5)/x, x -> Infinity]").unwrap(), "a");
    assert_eq!(
      interpret("Limit[(a x + b)/(c x + d), x -> Infinity]").unwrap(),
      "a/c"
    );
    // With an unknown sign the divergence stays symbolic.
    assert_eq!(
      interpret("Limit[(a x^2 + 1)/(x + 1), x -> Infinity]").unwrap(),
      "a*Infinity"
    );
    assert_eq!(
      interpret("Limit[a Sqrt[x], x -> Infinity]").unwrap(),
      "a*Infinity"
    );
    // A lower-order symbolic term does not change a zero limit.
    assert_eq!(
      interpret("Limit[(a x + 1)/(x^2 + 1), x -> Infinity]").unwrap(),
      "0"
    );
  }

  #[test]
  fn cancelling_leading_terms_defer_to_the_other_paths() {
    // Sqrt[x^2 + x] - x has both leading terms of order 1 with coefficients
    // that cancel, so the leading-order analysis declines and the
    // conjugate-difference path answers.
    assert_eq!(
      interpret("Limit[Sqrt[x^2 + x] - x, x -> Infinity]").unwrap(),
      "1/2"
    );
    // Logarithms and exponentials are outside the analysis entirely.
    assert_eq!(interpret("Limit[Log[x]/x, x -> Infinity]").unwrap(), "0");
    assert_eq!(interpret("Limit[(1 + 1/n)^n, n -> Infinity]").unwrap(), "E");
    assert_eq!(interpret("Limit[x/Exp[x], x -> Infinity]").unwrap(), "0");
  }
}

mod limits_of_power_sums_at_zero {
  use super::*;

  #[test]
  fn lowest_order_terms_decide_the_ratio() {
    // As x -> 0 the smallest exponent dominates, so these are the ratio of
    // the two lowest-order coefficients. L'Hopital cannot settle them: each
    // step just produces another fractional-power quotient of the same shape.
    assert_eq!(
      interpret("Limit[(x + Sqrt[x])/(x - Sqrt[x]), x -> 0]").unwrap(),
      "-1"
    );
    assert_eq!(
      interpret("Limit[(Sqrt[x] + x)/(Sqrt[x] - x), x -> 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[Sqrt[x]/(Sqrt[x] + x), x -> 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[(x + x^(1/3))/(x - x^(1/3)), x -> 0]").unwrap(),
      "-1"
    );
    assert_eq!(
      interpret("Limit[(2 Sqrt[x] + x)/(3 Sqrt[x] - x), x -> 0]").unwrap(),
      "2/3"
    );
    assert_eq!(
      interpret("Limit[(x^2 + Sqrt[x])/(x + Sqrt[x]), x -> 0]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Limit[(x + x^2)/(x - x^3), x -> 0]").unwrap(),
      "1"
    );
  }

  #[test]
  fn a_higher_order_numerator_vanishes() {
    assert_eq!(interpret("Limit[x/Sqrt[x], x -> 0]").unwrap(), "0");
    assert_eq!(
      interpret("Limit[(x^(3/2) + x)/(x^(1/2) + x), x -> 0]").unwrap(),
      "0"
    );
    assert_eq!(interpret("Limit[x^2/x, x -> 0]").unwrap(), "0");
  }

  #[test]
  fn divergence_needs_the_two_sides_to_agree() {
    // 1/x^k only keeps its sign across the origin when k is an even integer.
    assert_eq!(interpret("Limit[1/x^2, x -> 0]").unwrap(), "Infinity");
    assert_eq!(interpret("Limit[1/x^4, x -> 0]").unwrap(), "Infinity");
    assert_eq!(interpret("Limit[5/x^6, x -> 0]").unwrap(), "Infinity");
    assert_eq!(interpret("Limit[-3/x^4, x -> 0]").unwrap(), "-Infinity");
    assert_eq!(interpret("Limit[(x + 1)/x^2, x -> 0]").unwrap(), "Infinity");
    assert_eq!(
      interpret("Limit[(x - 2)/x^2, x -> 0]").unwrap(),
      "-Infinity"
    );
    // An odd integer power flips sign, and a fractional one is not real to
    // the left of 0; both leave the two-sided limit undefined.
    assert_eq!(interpret("Limit[1/x, x -> 0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Limit[1/x^3, x -> 0]").unwrap(), "Indeterminate");
    assert_eq!(
      interpret("Limit[1/Sqrt[x], x -> 0]").unwrap(),
      "Indeterminate"
    );
    assert_eq!(
      interpret("Limit[1/x^(2/3), x -> 0]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn a_direction_reinstates_the_one_sided_answer() {
    // The two-sided Indeterminate above is not allowed to swallow the
    // one-sided limits.
    assert_eq!(
      interpret(r#"Limit[1/x, x -> 0, Direction -> "FromAbove"]"#).unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret(r#"Limit[1/x, x -> 0, Direction -> "FromBelow"]"#).unwrap(),
      "-Infinity"
    );
  }

  #[test]
  fn transcendental_limits_keep_their_own_paths() {
    assert_eq!(interpret("Limit[Sin[x]/x, x -> 0]").unwrap(), "1");
    assert_eq!(interpret("Limit[Tan[x]/x, x -> 0]").unwrap(), "1");
    assert_eq!(interpret("Limit[(1 - Cos[x])/x^2, x -> 0]").unwrap(), "1/2");
    assert_eq!(
      interpret("Limit[(Sin[x] - x)/x^3, x -> 0]").unwrap(),
      "-1/6"
    );
    assert_eq!(interpret("Limit[x Log[x], x -> 0]").unwrap(), "0");
  }
}

mod unevaluated_limit_echoes_the_original {
  use super::*;

  #[test]
  fn an_unresolved_limit_does_not_leak_a_rewritten_expression() {
    // Several strategies recurse on a rewritten form (a L'Hopital derivative
    // ratio, an asymptotic expansion). When none resolves, the echo has to be
    // the caller's own input, not the internal rewrite.
    let out = interpret("Limit[f[x]/g[x], x -> 0]").unwrap();
    assert_eq!(out, "Limit[f[x]/g[x], x -> 0]");
    // Whatever a Gamma quotient at infinity does internally, an unresolved
    // result still echoes what was asked.
    let out = interpret("Limit[h[x], x -> Infinity]").unwrap();
    assert_eq!(out, "Limit[h[x], x -> Infinity]");
  }
}

mod differential_notation {
  use super::*;

  /// `ⅆx` is the differential of `x`, the notation the "Linear
  /// Approximations" chapter states its error estimates in. It has no value
  /// of its own, so an equation between differentials stays symbolic and a
  /// rule can replace a whole differential.
  #[test]
  fn a_differential_is_a_symbolic_factor() {
    assert_eq!(
      interpret(r"\[DifferentialD]area == 2 \[Pi] r \[DifferentialD]r")
        .unwrap(),
      "DifferentialD[area] == 2*Pi*r*DifferentialD[r]"
    );
    assert_eq!(
      interpret(
        r"sol = \[DifferentialD]area == 2 \[Pi] r \[DifferentialD]r; \
          sol /. {r -> 50, \[DifferentialD]r -> 0.1}"
      )
      .unwrap(),
      "DifferentialD[area] == 31.41592653589793"
    );
  }

  /// The bare character a notebook stores it as parses the same way, even
  /// though Unicode files it under *letters* — `ⅆarea` is a differential,
  /// not a symbol named "ⅆarea".
  #[test]
  fn the_differential_character_is_not_part_of_the_name() {
    assert_eq!(interpret("\u{2146}area").unwrap(), "DifferentialD[area]");
    assert_eq!(interpret("\u{F74C}x").unwrap(), "DifferentialD[x]");
  }
}
