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
      "Got: {}",
      result
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

  #[test]
  fn integrate_sin_cos_product() {
    // ∫ Sin[x]*Cos[x] dx = -1/2*Cos[x]^2
    assert_eq!(
      interpret("Integrate[Sin[x] * Cos[x], x]").unwrap(),
      "-1/2*Cos[x]^2"
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
    assert_eq!(interpret("Derivative[2][#^3&]").unwrap(), "6*#1 & ");
  }

  #[test]
  fn derivative_n_pure_function_cubic_third() {
    assert_eq!(interpret("Derivative[3][#^3&]").unwrap(), "6 & ");
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
    // SeriesData with coefficient list and order bounds.
    assert_eq!(
      interpret("series = Series[Cosh[x], {x, 0, 2}]; series // FullForm")
        .unwrap(),
      "SeriesData[x, 0, List[1, 0, Rational[1, 2]], 0, 3, 1]"
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
      "NIntegrate[Exp[-x],{x,0,Infinity},Tolerance->1*^-6, Method->\"Internal\"]",
      1.0,
      1e-5,
    );
  }

  #[test]
  fn nintegrate_neg_infinity_with_options() {
    assert_approx(
      "NIntegrate[Exp[x],{x,-Infinity, 0},Tolerance->1*^-6, Method->\"Internal\"]",
      1.0,
      1e-5,
    );
  }

  #[test]
  fn nintegrate_gaussian_with_options() {
    // ∫_{-∞}^∞ e^(-x²/2) dx = √(2π) ≈ 2.5066
    assert_approx(
      "NIntegrate[Exp[-x^2/2.],{x,-Infinity, Infinity},Tolerance->1*^-6, Method->\"Internal\"]",
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
    // Sec[1.] = 1/Cos[1.] ≈ 1.8508157176809255 (matches wolframscript).
    assert_eq!(interpret("Sec[1.]").unwrap(), "1.8508157176809255");
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
    assert!(result.starts_with("0"), "N[Erfi[0], 20] = {result}");
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
      "Got: {}",
      result
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
  fn exponential_pair_general_solution() {
    // y''[x] == y[x] has general solution C[1]*E^x + C[2]*E^(-x).
    assert_eq!(
      interpret("DSolve[y''[x] == y[x], y[x], x]").unwrap(),
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
    assert_eq!(interpret("Integrate[Log[x], x]").unwrap(), "-x + x*Log[x]");
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
      "-1/4*x^2 + (x^2*Log[x])/2"
    );
  }

  // General constant-base exponential (lowercase e is a symbol, not Euler's E)
  #[test]
  fn general_exp_basic() {
    assert_eq!(interpret("Integrate[e^x, x]").unwrap(), "e^x/Log[e]");
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
    // NMaximize[{Sin[x], 0 < x < 2*Pi}, x] should find max near Pi/2
    let result = interpret("NMaximize[{Sin[x], 0 < x < 2*Pi}, x]").unwrap();
    assert!(
      result.starts_with("{"),
      "Expected list result, got: {}",
      result
    );
    // Check that the max value is close to 1
    assert!(
      result.contains("0.99999"),
      "Max should be ~1, got: {}",
      result
    );
  }

  #[test]
  fn nminimize_quadratic() {
    // NMinimize[{x^2 - 4*x + 5, -10 < x < 10}, x] should find min at x=2
    let result =
      interpret("NMinimize[{x^2 - 4*x + 5, -10 < x < 10}, x]").unwrap();
    assert!(
      result.starts_with("{"),
      "Expected list result, got: {}",
      result
    );
    assert!(result.contains("1."), "Min should be ~1, got: {}", result);
  }

  #[test]
  fn nminimize_quartic_saddle_point() {
    // x^4 - 3x^2 + 2 has a local max at x=0 (value 2) and
    // global minima at x=±sqrt(3/2) (value -0.25).
    // Must not get stuck at the saddle point x=0.
    let result = interpret("NMinimize[x^4 - 3 x^2 + 2, x]").unwrap();
    assert!(
      result.starts_with("{-0.25"),
      "Min should be ~-0.25, got: {}",
      result
    );
  }

  #[test]
  fn nminimize_unconstrained_quadratic() {
    let result = interpret("NMinimize[x^2, x]").unwrap();
    assert!(
      result.starts_with("{0."),
      "Min should be ~0, got: {}",
      result
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
      "Expected rule result, got: {}",
      result
    );
  }

  #[test]
  fn findroot_sin_x_equals_x_at_origin() {
    assert_eq!(
      interpret("FindRoot[Sin[x] == x, {x, 0}]").unwrap(),
      "{x -> 0.}"
    );
  }
}

mod laplace_transform {
  use super::*;

  #[test]
  fn constant() {
    assert_eq!(interpret("LaplaceTransform[1, t, s]").unwrap(), "s^(-1)");
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

  #[test]
  fn constant_multiple() {
    assert_eq!(interpret("LaplaceTransform[5*t, t, s]").unwrap(), "5/s^2");
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

  #[test]
  fn unevaluated_bad_args() {
    assert_eq!(
      interpret("RecurrenceTable[x, y]").unwrap(),
      "RecurrenceTable[x, y]"
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
      "Expected InterpolatingFunction, got: {}",
      result
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
      result.contains("E^") && result.contains("I"),
      "Expected exponential form, got: {}",
      result
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
  fn symbolic() {
    // TrigToExp should not affect non-trig expressions
    assert_eq!(interpret("TrigToExp[x + 1]").unwrap(), "1 + x");
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
    assert!((val - 1.0).abs() < 0.001, "Expected 1.0, got {}", val);
  }

  #[test]
  fn values_at_data_points() {
    // Interpolation should return exact values at data points
    let result =
      interpret("f = Interpolation[{1, 2, 3, 5, 8, 5}]; f[4]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 5.0).abs() < 0.001, "Expected 5.0, got {}", val);
  }

  #[test]
  fn last_data_point() {
    let result =
      interpret("f = Interpolation[{1, 2, 3, 5, 8, 5}]; f[6]").unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 5.0).abs() < 0.001, "Expected 5.0, got {}", val);
  }

  #[test]
  fn explicit_xy_pairs() {
    // Interpolation[{{x1, y1}, {x2, y2}, ...}]
    let result =
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}]; f[2]")
        .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 4.0).abs() < 0.001, "Expected 4.0, got {}", val);
  }

  #[test]
  fn interpolation_between_points() {
    // Test interpolation at a point between data values
    let result =
      interpret("f = Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}]; f[1.5]")
        .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    // Cubic interpolation of x^2 data should approximate 1.5^2 = 2.25
    assert!((val - 2.25).abs() < 0.5, "Expected ~2.25, got {}", val);
  }

  #[test]
  fn returns_interpolating_function() {
    let result = interpret("Interpolation[{1, 2, 3, 4}]").unwrap();
    assert!(
      result.contains("InterpolatingFunction"),
      "Expected InterpolatingFunction, got: {}",
      result
    );
    assert!(
      result.contains("<>"),
      "Expected <> in display, got: {}",
      result
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
    assert!((val - 0.5).abs() < 0.001, "Expected 0.5, got {}", val);
  }

  #[test]
  fn interpolation_order_1_second_interval() {
    let result = interpret(
      "f = Interpolation[{{0, 0}, {1, 1}, {2, 4}}, InterpolationOrder -> 1]; f[1.5]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    // Linear interpolation between (1,1) and (2,4): 1 + 0.5*3 = 2.5
    assert!((val - 2.5).abs() < 0.001, "Expected 2.5, got {}", val);
  }

  #[test]
  fn domain_display() {
    let result =
      interpret("Interpolation[{{0, 1}, {1, 2}, {2, 3}, {3, 4}}]").unwrap();
    assert!(
      result.contains("{{0., 3.}}"),
      "Expected domain {{0., 3.}}, got: {}",
      result
    );
  }

  #[test]
  fn symbolic_argument_returns_unevaluated() {
    let result = interpret("f = Interpolation[{1, 2, 3, 4}]; f[x]").unwrap();
    assert!(
      result.contains("InterpolatingFunction"),
      "Expected unevaluated form with symbolic arg, got: {}",
      result
    );
  }

  #[test]
  fn order_reduced_when_too_few_points() {
    // Default order 3 with only 3 points should reduce to order 2
    let result =
      interpret("f = Interpolation[{{1, 3}, {2, 5}, {3, 11}}]; f[1.5]")
        .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 3.5).abs() < 0.001, "Expected 3.5, got {}", val);
  }

  #[test]
  fn quadratic_interpolation() {
    // Quadratic interpolation of x^2 data: 1.5^2 = 2.25
    let result = interpret(
      "f = Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}, {4, 16}}, InterpolationOrder -> 2]; f[1.5]",
    )
    .unwrap();
    let val: f64 = result.parse().expect("should be a number");
    assert!((val - 2.25).abs() < 0.001, "Expected 2.25, got {}", val);
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
      "Should return unevaluated: {}",
      result
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

  #[test]
  fn unevaluated_for_unknown() {
    let result = interpret("InverseFourierTransform[g[w], w, t]").unwrap();
    assert!(
      result.contains("InverseFourierTransform"),
      "Should return unevaluated: {}",
      result
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
  fn sin_cos_product() {
    assert_eq!(
      interpret("TrigReduce[Sin[x] Cos[x]]").unwrap(),
      "Sin[2*x]/2"
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
      "x < -1 || -1 < x < 1 || x > 1"
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
      result.contains("x") && result.contains("0"),
      "Should contain domain constraint: {}",
      result
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
    // EGF[Sin[n], n, x] = (Cosh[x*Cos[1]] + Sinh[x*Cos[1]]) * Sin[x*Sin[1]]
    // which equals E^(x*Cos[1]) * Sin[x*Sin[1]]
    assert_eq!(
      interpret("ExponentialGeneratingFunction[Sin[n], n, x]").unwrap(),
      "(Cosh[x*Cos[1]] + Sinh[x*Cos[1]])*Sin[x*Sin[1]]"
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

mod asymptotic_solve {
  use super::*;

  #[test]
  fn integer_third_arg_unevaluated() {
    clear_state();
    // A plain integer 3rd arg is invalid; Wolfram returns unevaluated
    let result = interpret("AsymptoticSolve[x - 1 == 0, x -> 0, 3]").unwrap();
    assert!(
      result.starts_with("AsymptoticSolve["),
      "expected unevaluated for integer 3rd arg, got {}",
      result
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    clear_state();
    // With wrong number of args
    let result = interpret("AsymptoticSolve[x^2 - 1]").unwrap();
    assert!(
      result.starts_with("AsymptoticSolve["),
      "expected unevaluated, got {}",
      result
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
      result.contains("w") && !result.contains("FourierSinTransform"),
      "expected evaluated result, got {}",
      result
    );
  }

  #[test]
  fn linearity() {
    clear_state();
    let result = interpret("FourierSinTransform[3*E^(-t), t, w]").unwrap();
    assert!(
      !result.contains("FourierSinTransform"),
      "expected evaluated result, got {}",
      result
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    clear_state();
    let result = interpret("FourierSinTransform[f[t], t, w]").unwrap();
    assert!(
      result.contains("FourierSinTransform"),
      "expected unevaluated, got {}",
      result
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
      result.contains("a") && !result.contains("FourierCosTransform"),
      "expected evaluated result, got {}",
      result
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    clear_state();
    let result = interpret("FourierCosTransform[g[t], t, w]").unwrap();
    assert!(
      result.contains("FourierCosTransform"),
      "expected unevaluated, got {}",
      result
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
      result.contains("E"),
      "expected expression with E, got {}",
      result
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
      "expected evaluated, got {}",
      result
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
}
