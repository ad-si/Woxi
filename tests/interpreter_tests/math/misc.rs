use super::*;

mod minus_wrong_arity {
  use super::*;

  #[test]
  fn minus_single_arg_negates() {
    assert_eq!(interpret("Minus[5]").unwrap(), "-5");
  }

  #[test]
  fn minus_two_args_returns_unevaluated() {
    // Minus[5, 2] should print warning and return 5 − 2 (Unicode minus, matching Wolfram)
    let result = interpret("Minus[5, 2]").unwrap();
    assert_eq!(result, "5 \u{2212} 2");
  }
}

mod therefore {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("Therefore[a, b]").unwrap(), "a \u{2234} b");
  }

  #[test]
  fn three_args() {
    assert_eq!(
      interpret("Therefore[a, b, c]").unwrap(),
      "a \u{2234} b \u{2234} c"
    );
  }

  #[test]
  fn one_arg() {
    assert_eq!(interpret("Therefore[a]").unwrap(), "Therefore[a]");
  }

  #[test]
  fn zero_args() {
    assert_eq!(interpret("Therefore[]").unwrap(), "Therefore[]");
  }

  #[test]
  fn args_evaluated() {
    assert_eq!(interpret("Therefore[1+2, 3]").unwrap(), "3 \u{2234} 3");
  }
}

mod because {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("Because[a, b]").unwrap(), "a \u{2235} b");
  }

  #[test]
  fn three_args() {
    assert_eq!(
      interpret("Because[a, b, c]").unwrap(),
      "a \u{2235} b \u{2235} c"
    );
  }

  #[test]
  fn one_arg() {
    assert_eq!(interpret("Because[a]").unwrap(), "Because[a]");
  }

  #[test]
  fn args_evaluated() {
    assert_eq!(interpret("Because[1+2, 3]").unwrap(), "3 \u{2235} 3");
  }
}

mod implicit_multiply_power_precedence {
  use super::*;

  #[test]
  fn b_y_cubed() {
    assert_eq!(interpret("FullForm[b y^3]").unwrap(), "FullForm[b*y^3]");
    assert_eq!(
      interpret("ToString[FullForm[b y^3]]").unwrap(),
      "Times[b, Power[y, 3]]"
    );
  }

  #[test]
  fn two_x_squared_y_cubed() {
    assert_eq!(
      interpret("FullForm[2 x^2 y^3]").unwrap(),
      "FullForm[2*x^2*y^3]"
    );
    assert_eq!(
      interpret("ToString[FullForm[2 x^2 y^3]]").unwrap(),
      "Times[2, Power[x, 2], Power[y, 3]]"
    );
  }

  #[test]
  fn coefficient_with_implicit_multiply() {
    assert_eq!(
      interpret("Coefficient[a x^2 + b y^3 + c x + d y + 5, y, 3]").unwrap(),
      "b"
    );
  }

  #[test]
  fn function_call_implicit_times() {
    // Regression: Sin[x] Sin[y] was not parsed as implicit multiplication
    assert_eq!(interpret("Sin[x] Cos[y]").unwrap(), "Cos[y]*Sin[x]");
  }

  #[test]
  fn function_call_implicit_times_three_factors() {
    assert_eq!(
      interpret("Sin[x] Cos[y] Tan[z]").unwrap(),
      "Cos[y]*Sin[x]*Tan[z]"
    );
  }

  #[test]
  fn function_call_implicit_times_with_number() {
    assert_eq!(interpret("2 Sin[x]").unwrap(), "2*Sin[x]");
  }

  #[test]
  fn function_call_implicit_times_with_implicit_arg() {
    // Sin[3y] should parse 3y as implicit multiplication inside the argument
    assert_eq!(interpret("Sin[x] Sin[3y]").unwrap(), "Sin[x]*Sin[3*y]");
  }

  #[test]
  fn function_call_implicit_times_evaluates() {
    assert_eq!(interpret("Sin[0] Sin[Pi/2]").unwrap(), "0");
  }

  #[test]
  fn implicit_times_with_part_access() {
    // Regression: 2 x[[1]] was not parsed as implicit multiplication with Part access
    assert_eq!(interpret("x = {10, 20, 30}; 2 x[[2]]").unwrap(), "40");
  }

  #[test]
  fn implicit_times_real_with_part_access() {
    // Regression: 100. masses[[i]] failed to parse
    assert_eq!(
      interpret("masses = {1, 2, 3}; 100. masses[[2]]").unwrap(),
      "200."
    );
  }

  #[test]
  fn implicit_times_part_access_multiple_factors() {
    assert_eq!(interpret("a = {2}; b = {3}; a[[1]] b[[1]]").unwrap(), "6");
  }

  #[test]
  fn implicit_times_function_call_with_part_access() {
    // 2 f[x][[1]] should parse as 2 * Part[f[x], 1]
    assert_eq!(
      interpret("FullForm[Hold[2 f[x][[1]]]]").unwrap(),
      "FullForm[Hold[2*f[x][[1]]]]"
    );
  }
}

mod max_symbolic {
  use super::*;

  #[test]
  fn filters_numeric_keeps_symbolic() {
    assert_eq!(interpret("Max[5, x, -3, y, 40]").unwrap(), "Max[40, x, y]");
  }

  #[test]
  fn all_numeric() {
    assert_eq!(interpret("Max[5, 3, 8, 1]").unwrap(), "8");
  }
}

mod min_symbolic {
  use super::*;

  #[test]
  fn filters_numeric_keeps_symbolic() {
    assert_eq!(interpret("Min[5, x, -3, y, 40]").unwrap(), "Min[-3, x, y]");
  }
}

mod implicit_times_with_patterns {
  use super::*;

  #[test]
  fn pattern_optional_default_implicit_times() {
    // Regression: c_. x_^2 failed to parse as implicit multiplication
    assert_eq!(interpret("Hold[c_. x_^2]").unwrap(), "Hold[(c_.)*(x_)^2]");
  }

  #[test]
  fn pattern_optional_default_implicit_times_with_power() {
    // c_. x_^2 should be Times[c_., Power[x_, 2]]
    assert_eq!(
      interpret("Hold[c_. x_^2] // FullForm").unwrap(),
      "FullForm[Hold[(c_.)*(x_)^2]]"
    );
  }

  #[test]
  fn number_times_pattern_implicit() {
    assert_eq!(interpret("Hold[2 x_]").unwrap(), "Hold[2*(x_)]");
  }

  #[test]
  fn complex_pattern_expression_implicit_times() {
    // The expression from compare_output.sh
    assert_eq!(
      interpret("Int[(a_.+b_.*x_+c_. x_^2)^n_,x_Symbol] := Foo[x]").unwrap(),
      "\0"
    );
  }
}

mod precision_real {
  use super::*;

  #[test]
  fn parse_precision_real() {
    assert_eq!(interpret("0.1`1").unwrap(), "0.1`1.");
  }

  #[test]
  fn parse_bare_backtick_machine_precision() {
    // Bare backtick is machine precision, displayed as normal Real
    assert_eq!(interpret("0.1`").unwrap(), "0.1");
  }

  #[test]
  fn precision_real_addition() {
    assert_eq!(interpret("0.1`1 + 0.2`1").unwrap(), "0.3`1.");
  }

  #[test]
  fn precision_real_plus_integer() {
    assert_eq!(interpret("0.1`1 + 1").unwrap(), "1.1`2.041392685158225");
  }
}

mod max_min_flatten {
  use super::*;

  #[test]
  fn max_flattens_nested_lists() {
    assert_eq!(
      interpret("Max[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]").unwrap(),
      "3.5"
    );
  }

  #[test]
  fn min_flattens_nested_lists() {
    assert_eq!(
      interpret("Min[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]").unwrap(),
      "-Infinity"
    );
  }
}

mod diagonal {
  use super::*;

  #[test]
  fn main_diagonal() {
    assert_eq!(
      interpret("Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "{1, 5, 9}"
    );
  }

  #[test]
  fn superdiagonal() {
    assert_eq!(
      interpret("Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]").unwrap(),
      "{2, 6}"
    );
  }

  #[test]
  fn subdiagonal() {
    assert_eq!(
      interpret("Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, -1]").unwrap(),
      "{4, 8}"
    );
  }

  #[test]
  fn rectangular() {
    assert_eq!(
      interpret("Diagonal[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "{1, 5}"
    );
  }
}

mod pauli_matrix {
  use super::*;

  #[test]
  fn pauli_1() {
    assert_eq!(interpret("PauliMatrix[1]").unwrap(), "{{0, 1}, {1, 0}}");
  }

  #[test]
  fn pauli_2() {
    assert_eq!(interpret("PauliMatrix[2]").unwrap(), "{{0, -I}, {I, 0}}");
  }

  #[test]
  fn pauli_3() {
    assert_eq!(interpret("PauliMatrix[3]").unwrap(), "{{1, 0}, {0, -1}}");
  }

  #[test]
  fn pauli_table() {
    assert_eq!(
      interpret("Table[PauliMatrix[i], {i, 1, 3}]").unwrap(),
      "{{{0, 1}, {1, 0}}, {{0, -I}, {I, 0}}, {{1, 0}, {0, -1}}}"
    );
  }
}

mod wigner_symbols {
  use super::*;

  // ThreeJSymbol returns 0 when m1 + m2 + m3 ≠ 0.
  #[test]
  fn three_j_nonzero_m_sum() {
    assert_eq!(
      interpret("ThreeJSymbol[{2, 0}, {6, 0}, {4, 1}]").unwrap(),
      "0"
    );
  }

  // ThreeJSymbol returns 0 when |m_i| > j_i for any i.
  #[test]
  fn three_j_m_exceeds_j() {
    assert_eq!(
      interpret("ThreeJSymbol[{1, 2}, {3, 4}, {5, 12}]").unwrap(),
      "0"
    );
  }

  // SixJSymbol returns 0 when any triangle inequality fails.
  #[test]
  fn six_j_triangle_fails() {
    assert_eq!(interpret("SixJSymbol[{1, 2, 3}, {4, 5, 12}]").unwrap(), "0");
  }

  // ThreeJSymbol now evaluates to its closed form via the Racah formula.
  #[test]
  fn three_j_valid_evaluates() {
    assert_eq!(
      interpret("ThreeJSymbol[{2, 0}, {6, 0}, {4, 0}]").unwrap(),
      "Sqrt[5/143]"
    );
  }

  // ClebschGordan of the "stretched state" (max |m| aligned with max j)
  // is exactly 1.
  #[test]
  fn clebsch_gordan_stretched_negative() {
    assert_eq!(
      interpret("ClebschGordan[{1/2, -1/2}, {1/2, -1/2}, {1, -1}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn clebsch_gordan_stretched_positive() {
    assert_eq!(
      interpret("ClebschGordan[{1/2, 1/2}, {1/2, 1/2}, {1, 1}]").unwrap(),
      "1"
    );
  }

  // ClebschGordan is zero when m1 + m2 ≠ m.
  #[test]
  fn clebsch_gordan_m_mismatch() {
    assert_eq!(
      interpret("ClebschGordan[{1/2, 1/2}, {1/2, 1/2}, {0, 0}]").unwrap(),
      "0"
    );
  }
}

mod curl {
  use super::*;

  #[test]
  fn curl_2d() {
    assert_eq!(interpret("Curl[{y, -x}, {x, y}]").unwrap(), "-2");
  }

  #[test]
  fn curl_3d() {
    assert_eq!(
      interpret("Curl[{y, -x, 2 z}, {x, y, z}]").unwrap(),
      "{0, 0, -2}"
    );
  }

  // Curl[F, vars, "Coordinates"] uses orthogonal-curvilinear scale factors.
  // 2D gives a scalar; 3D gives a vector.
  #[test]
  fn curl_polar_scalar() {
    assert_eq!(interpret(r#"Curl[{0, r}, {r, t}, "Polar"]"#).unwrap(), "2");
  }

  #[test]
  fn curl_cylindrical() {
    assert_eq!(
      interpret(r#"Curl[{0, 0, r}, {r, t, z}, "Cylindrical"]"#).unwrap(),
      "{0, -1, 0}"
    );
    assert_eq!(
      interpret(r#"Curl[{r, 0, 0}, {r, t, z}, "Cylindrical"]"#).unwrap(),
      "{0, 0, 0}"
    );
  }

  #[test]
  fn curl_spherical() {
    assert_eq!(
      interpret(r#"Curl[{0, 0, r Sin[t]}, {r, t, p}, "Spherical"]"#).unwrap(),
      "{2*Cos[t], -2*Sin[t], 0}"
    );
  }

  #[test]
  fn curl_unknown_coordinates_unevaluated() {
    assert_eq!(
      interpret(r#"Curl[{0, r}, {r, t}, "Bogus"]"#).unwrap(),
      "Curl[{0, r}, {r, t}, Bogus]"
    );
  }
}

mod log {
  use super::*;

  #[test]
  fn log_zero() {
    assert_eq!(interpret("Log[0]").unwrap(), "-Infinity");
  }

  #[test]
  fn log_two_arg_exact() {
    assert_eq!(interpret("Log[2, 8]").unwrap(), "3");
    assert_eq!(interpret("Log[2, 16]").unwrap(), "4");
    assert_eq!(interpret("Log[3, 9]").unwrap(), "2");
  }

  #[test]
  fn log_two_arg_symbolic() {
    assert_eq!(interpret("Log[2, 5]").unwrap(), "Log[5]/Log[2]");
  }

  #[test]
  fn log_infinity() {
    assert_eq!(interpret("Log[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn log_complex_infinity() {
    assert_eq!(interpret("Log[ComplexInfinity]").unwrap(), "Infinity");
  }

  #[test]
  fn log_i() {
    assert_eq!(interpret("Log[I]").unwrap(), "I/2*Pi");
  }

  #[test]
  fn log_neg_i() {
    assert_eq!(interpret("Log[-I]").unwrap(), "(-1/2*I)*Pi");
  }

  #[test]
  fn log_e_to_pi() {
    assert_eq!(interpret("Log[E^Pi]").unwrap(), "Pi");
  }

  #[test]
  fn log_e_to_x() {
    // Log[E^x] stays unevaluated for symbolic x
    assert_eq!(interpret("Log[E^x]").unwrap(), "Log[E^x]");
  }

  // Log[E^z] for a numeric complex exponent z reduces to z with its imaginary
  // part brought into (-Pi, Pi].
  #[test]
  fn log_e_to_imaginary() {
    assert_eq!(interpret("Log[E^(2 I)]").unwrap(), "2*I");
    assert_eq!(interpret("Log[E^(3 I)]").unwrap(), "3*I");
    assert_eq!(interpret("Log[E^(I Pi/2)]").unwrap(), "I/2*Pi");
  }

  #[test]
  fn log_e_to_imaginary_reduced() {
    // Im outside (-Pi, Pi] is reduced by a multiple of 2 Pi.
    assert_eq!(interpret("Log[E^(5 I)]").unwrap(), "5*I - (2*I)*Pi");
    assert_eq!(interpret("Log[E^(-5 I)]").unwrap(), "-5*I + (2*I)*Pi");
  }

  #[test]
  fn log_e_to_complex_in_range() {
    // Re concrete, Im in range: result is the exponent itself.
    assert_eq!(interpret("Log[E^(2 + 3 I)]").unwrap(), "2 + 3*I");
  }

  #[test]
  fn log_e_to_complex_symbolic_real_part() {
    // A symbolic real part keeps it unevaluated (E^a need not be positive real).
    assert_eq!(interpret("Log[E^(a + 3 I)]").unwrap(), "Log[E^(a + 3*I)]");
  }
}

mod linear_recurrence {
  use super::*;

  #[test]
  fn fibonacci_via_recurrence() {
    assert_eq!(
      interpret("LinearRecurrence[{1, 1}, {1, 1}, 10]").unwrap(),
      "{1, 1, 2, 3, 5, 8, 13, 21, 34, 55}"
    );
  }
}

mod zero_divided_by_symbolic {
  use super::*;

  #[test]
  fn zero_over_symbolic() {
    assert_eq!(interpret("0/x").unwrap(), "0");
    assert_eq!(interpret("0/(2*Pi)").unwrap(), "0");
    assert_eq!(interpret("0/Sqrt[2]").unwrap(), "0");
  }

  #[test]
  fn zero_real_over_symbolic() {
    assert_eq!(interpret("0.0/x").unwrap(), "0.");
    assert_eq!(interpret("0.0/Pi").unwrap(), "0.");
  }

  #[test]
  fn zero_over_integer() {
    assert_eq!(interpret("0/5").unwrap(), "0");
  }
}

mod eigensystem {
  use super::*;

  #[test]
  fn two_by_two() {
    assert_eq!(
      interpret("Eigensystem[{{1, 2}, {3, 4}}]").unwrap(),
      "{{(5 + Sqrt[33])/2, (5 - Sqrt[33])/2}, {{(-3 + Sqrt[33])/6, 1}, {(-3 - Sqrt[33])/6, 1}}}"
    );
  }

  #[test]
  fn diagonal() {
    assert_eq!(
      interpret("Eigensystem[{{2, 0}, {0, 3}}]").unwrap(),
      "{{3, 2}, {{0, 1}, {1, 0}}}"
    );
  }

  #[test]
  fn three_by_three_diagonal() {
    assert_eq!(
      interpret("Eigensystem[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "{{3, 2, 1}, {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}}"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("Eigensystem[m]").unwrap(), "Eigensystem[m]");
  }
}

mod unit_box {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("UnitBox[0]").unwrap(), "1");
  }

  #[test]
  fn inside() {
    assert_eq!(interpret("UnitBox[0.3]").unwrap(), "1");
  }

  #[test]
  fn boundary() {
    assert_eq!(interpret("UnitBox[0.5]").unwrap(), "1");
  }

  #[test]
  fn outside() {
    assert_eq!(interpret("UnitBox[1]").unwrap(), "0");
  }

  #[test]
  fn negative_inside() {
    assert_eq!(interpret("UnitBox[-0.3]").unwrap(), "1");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("UnitBox[x]").unwrap(), "UnitBox[x]");
  }
}

mod unit_triangle {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("UnitTriangle[0]").unwrap(), "1");
  }

  #[test]
  fn half() {
    assert_eq!(interpret("UnitTriangle[0.5]").unwrap(), "0.5");
  }

  #[test]
  fn boundary() {
    assert_eq!(interpret("UnitTriangle[1]").unwrap(), "0");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("UnitTriangle[-0.5]").unwrap(), "0.5");
  }

  #[test]
  fn outside() {
    assert_eq!(interpret("UnitTriangle[2]").unwrap(), "0");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("UnitTriangle[x]").unwrap(), "UnitTriangle[x]");
  }
}

mod angle_vector {
  use super::*;

  #[test]
  fn basic_angle() {
    assert_eq!(
      interpret("AngleVector[Pi/4]").unwrap(),
      "{1/Sqrt[2], 1/Sqrt[2]}"
    );
  }

  #[test]
  fn zero_angle() {
    assert_eq!(interpret("AngleVector[0]").unwrap(), "{1, 0}");
  }

  #[test]
  fn pi_half() {
    assert_eq!(interpret("AngleVector[Pi/2]").unwrap(), "{0, 1}");
  }

  #[test]
  fn with_radius() {
    assert_eq!(interpret("AngleVector[{2, Pi/3}]").unwrap(), "{1, Sqrt[3]}");
  }

  #[test]
  fn with_center() {
    assert_eq!(
      interpret("AngleVector[{1, 2}, Pi/6]").unwrap(),
      "{1 + Sqrt[3]/2, 5/2}"
    );
  }

  #[test]
  fn with_center_and_radius() {
    assert_eq!(
      interpret("AngleVector[{1, 0}, {2, Pi/2}]").unwrap(),
      "{1, 2}"
    );
  }
}

mod inverse_trig_identities {
  use super::*;

  #[test]
  fn sin_arcsin() {
    assert_eq!(interpret("Sin[ArcSin[x]]").unwrap(), "x");
    assert_eq!(interpret("Sin[ArcSin[1/2]]").unwrap(), "1/2");
  }

  #[test]
  fn cos_arccos() {
    assert_eq!(interpret("Cos[ArcCos[x]]").unwrap(), "x");
  }

  #[test]
  fn tan_arctan() {
    assert_eq!(interpret("Tan[ArcTan[x]]").unwrap(), "x");
  }

  #[test]
  fn arctan_sqrt3() {
    assert_eq!(interpret("ArcTan[Sqrt[3]]").unwrap(), "Pi/3");
  }

  #[test]
  fn arctan_inv_sqrt3() {
    assert_eq!(interpret("ArcTan[1/Sqrt[3]]").unwrap(), "Pi/6");
  }

  #[test]
  fn arctan_neg_sqrt3() {
    assert_eq!(interpret("ArcTan[-Sqrt[3]]").unwrap(), "-1/3*Pi");
  }

  #[test]
  fn arctan_neg_inv_sqrt3() {
    assert_eq!(interpret("ArcTan[-1/Sqrt[3]]").unwrap(), "-1/6*Pi");
  }

  // Twelfth-angle values: inverse of Tan[Pi/12] = 2 - Sqrt[3] and
  // Tan[5 Pi/12] = 2 + Sqrt[3]. ArcCot picks these up via ArcTan.
  #[test]
  fn arctan_twelfth_angle() {
    assert_eq!(interpret("ArcTan[2 - Sqrt[3]]").unwrap(), "Pi/12");
    assert_eq!(interpret("ArcTan[2 + Sqrt[3]]").unwrap(), "(5*Pi)/12");
    assert_eq!(interpret("ArcTan[-(2 - Sqrt[3])]").unwrap(), "-1/12*Pi");
    assert_eq!(interpret("ArcTan[-(2 + Sqrt[3])]").unwrap(), "(-5*Pi)/12");
    assert_eq!(interpret("ArcCot[2 - Sqrt[3]]").unwrap(), "(5*Pi)/12");
    assert_eq!(interpret("ArcCot[2 + Sqrt[3]]").unwrap(), "Pi/12");
  }

  #[test]
  fn sinh_arcsinh() {
    assert_eq!(interpret("Sinh[ArcSinh[x]]").unwrap(), "x");
  }

  #[test]
  fn cosh_arccosh() {
    assert_eq!(interpret("Cosh[ArcCosh[x]]").unwrap(), "x");
  }

  #[test]
  fn tanh_arctanh() {
    assert_eq!(interpret("Tanh[ArcTanh[x]]").unwrap(), "x");
  }

  // ArcSinh is odd and unbounded: ArcSinh[±Infinity] = ±Infinity; an
  // undirected ComplexInfinity maps to ComplexInfinity. Per wolframscript.
  #[test]
  fn arcsinh_infinite_limits() {
    assert_eq!(interpret("ArcSinh[Infinity]").unwrap(), "Infinity");
    assert_eq!(interpret("ArcSinh[-Infinity]").unwrap(), "-Infinity");
    assert_eq!(
      interpret("ArcSinh[ComplexInfinity]").unwrap(),
      "ComplexInfinity"
    );
    // A finite exact argument stays symbolic.
    assert_eq!(interpret("ArcSinh[2]").unwrap(), "ArcSinh[2]");
  }

  // ArcCosh grows without bound in magnitude, so every infinite argument
  // (Infinity, -Infinity, ComplexInfinity) maps to Infinity. Per wolframscript.
  #[test]
  fn arccosh_infinite_limits() {
    assert_eq!(interpret("ArcCosh[Infinity]").unwrap(), "Infinity");
    assert_eq!(interpret("ArcCosh[-Infinity]").unwrap(), "Infinity");
    assert_eq!(interpret("ArcCosh[ComplexInfinity]").unwrap(), "Infinity");
    // A finite exact argument stays symbolic.
    assert_eq!(interpret("ArcCosh[2]").unwrap(), "ArcCosh[2]");
  }

  // ArcSin/ArcCos/ArcTanh of an inexact real outside their real domain give a
  // numeric complex result (an exact integer/rational argument stays
  // symbolic). All expected values verified against wolframscript.
  #[test]
  fn arcsin_out_of_domain_real_is_complex() {
    // Round to avoid last-ULP libm divergence between platforms.
    assert_eq!(
      interpret("Round[ArcSin[2.0], 0.001]").unwrap(),
      "1.571 - 1.317*I"
    );
    assert_eq!(
      interpret("Round[ArcSin[-1.5], 0.001]").unwrap(),
      "-1.571 + 0.962*I"
    );
    // Exact integer stays symbolic.
    assert_eq!(interpret("ArcSin[3]").unwrap(), "ArcSin[3]");
  }

  #[test]
  fn arccos_out_of_domain_real_is_complex() {
    // Round to avoid last-ULP libm divergence between platforms.
    assert_eq!(
      interpret("Round[ArcCos[2.0], 0.001]").unwrap(),
      "0. + 1.317*I"
    );
    assert_eq!(
      interpret("Round[ArcCos[-1.5], 0.001]").unwrap(),
      "3.142 - 0.962*I"
    );
  }

  #[test]
  fn arctanh_out_of_domain_real_is_complex() {
    // Round to avoid a last-ULP libm divergence between platforms in the real
    // part (macOS ...549 vs Linux ...548).
    assert_eq!(
      interpret("Round[ArcTanh[2.0], 0.0001]").unwrap(),
      "0.5493 - 1.5708*I"
    );
    assert_eq!(
      interpret("Round[ArcTanh[-2.0], 0.0001]").unwrap(),
      "-0.5493 + 1.5708*I"
    );
  }

  // In-domain reals are unchanged (still return a plain real).
  #[test]
  fn arc_in_domain_reals_stay_real() {
    // Round to avoid a last-ULP libm divergence between platforms in the
    // full-precision output (macOS ...988 vs Linux ...989).
    assert_eq!(interpret("Round[ArcSin[0.5], 0.001]").unwrap(), "0.524");
    assert_eq!(interpret("Round[ArcCos[0.5], 0.001]").unwrap(), "1.047");
    assert_eq!(interpret("Round[ArcTanh[0.5], 0.001]").unwrap(), "0.549");
  }

  // ArcSec/ArcCsc/ArcSech follow the reciprocal identities (ArcSec[x] =
  // ArcCos[1/x], ArcCsc[x] = ArcSin[1/x], ArcSech[x] = ArcCosh[1/x]), so an
  // inexact real outside their real domain gives a complex result. Also
  // covers the ArcCosh[x < -1] branch. Verified against wolframscript.
  #[test]
  fn arcsec_out_of_domain_real_is_complex() {
    // Round to avoid last-ULP libm divergence between platforms.
    assert_eq!(
      interpret("Round[ArcSec[0.5], 0.001]").unwrap(),
      "0. + 1.317*I"
    );
  }

  #[test]
  fn arccsc_out_of_domain_real_is_complex() {
    // Round to avoid last-ULP libm divergence between platforms.
    assert_eq!(
      interpret("Round[ArcCsc[0.5], 0.001]").unwrap(),
      "1.571 - 1.317*I"
    );
  }

  #[test]
  fn arcsech_out_of_domain_real_is_complex() {
    // Round to avoid last-ULP libm divergence between platforms.
    assert_eq!(
      interpret("Round[ArcSech[2.0], 0.001]").unwrap(),
      "0. + 1.047*I"
    );
    assert_eq!(
      interpret("Round[ArcSech[-0.5], 0.001]").unwrap(),
      "1.317 + 3.142*I"
    );
  }

  #[test]
  fn arccosh_below_negative_one_is_complex() {
    // Previously returned Indeterminate (x.acos() is NaN for x < -1).
    // Round to avoid last-ULP libm divergence between platforms.
    assert_eq!(
      interpret("Round[ArcCosh[-2.0], 0.001]").unwrap(),
      "1.317 + 3.142*I"
    );
    assert_eq!(
      interpret("Round[ArcCosh[-1.0], 0.001]").unwrap(),
      "0. + 3.142*I"
    );
  }

  // In-domain reals still return a plain real, and exact integer arguments
  // stay symbolic rather than being numericized.
  #[test]
  fn arcsec_arccsc_in_domain_and_exact() {
    // Round to avoid last-ULP libm divergence between platforms.
    assert_eq!(interpret("Round[ArcSec[2.0], 0.001]").unwrap(), "1.047");
    assert_eq!(interpret("Round[ArcCsc[2.0], 0.001]").unwrap(), "0.524");
    assert_eq!(interpret("Round[ArcSech[0.5], 0.001]").unwrap(), "1.317");
    assert_eq!(interpret("ArcSec[3]").unwrap(), "ArcSec[3]");
    assert_eq!(interpret("ArcCsc[3]").unwrap(), "ArcCsc[3]");
    assert_eq!(interpret("ArcSec[2]").unwrap(), "Pi/3");
  }

  // A zero real argument: ArcSec/ArcCsc diverge to ComplexInfinity, ArcSech
  // to Indeterminate. Per wolframscript.
  #[test]
  fn arcsec_family_zero_real() {
    assert_eq!(interpret("ArcSec[0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("ArcCsc[0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("ArcSech[0.0]").unwrap(), "Indeterminate");
  }

  // ArcCoth[±1.] is a singularity: it returns Indeterminate (the (x+1)/(x-1)
  // ratio divides by zero). The exact ArcCoth[±1] instead stays ±Infinity,
  // and ArcCsch[0.] diverges to ComplexInfinity. Per wolframscript.
  #[test]
  fn arccoth_singular_real_is_indeterminate() {
    assert_eq!(interpret("ArcCoth[1.0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("ArcCoth[-1.0]").unwrap(), "Indeterminate");
    // Exact ±1 is a different limit and stays ±Infinity.
    assert_eq!(interpret("ArcCoth[1]").unwrap(), "Infinity");
    assert_eq!(interpret("ArcCoth[-1]").unwrap(), "-Infinity");
  }

  #[test]
  fn arccsch_zero_real_is_complex_infinity() {
    assert_eq!(interpret("ArcCsch[0.0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("ArcCsch[0]").unwrap(), "ComplexInfinity");
  }

  // The non-singular real branches are unchanged.
  #[test]
  fn arccoth_arccsch_real_branches_unchanged() {
    // Round to avoid a last-ULP libm divergence between platforms
    // (macOS ...549 vs Linux ...548).
    assert_eq!(interpret("Round[ArcCoth[2.0], 0.001]").unwrap(), "0.549");
    assert_eq!(
      interpret("Round[ArcCoth[0.5], 0.0001]").unwrap(),
      "0.5493 - 1.5708*I"
    );
    assert_eq!(interpret("Round[ArcCsch[2.0], 0.0001]").unwrap(), "0.4812");
  }
}

mod inverse_function {
  use super::*;

  #[test]
  fn trig_inverses() {
    assert_eq!(interpret("InverseFunction[Sin]").unwrap(), "ArcSin");
    assert_eq!(interpret("InverseFunction[Cos]").unwrap(), "ArcCos");
    assert_eq!(interpret("InverseFunction[Tan]").unwrap(), "ArcTan");
    assert_eq!(interpret("InverseFunction[Cot]").unwrap(), "ArcCot");
    assert_eq!(interpret("InverseFunction[Sec]").unwrap(), "ArcSec");
    assert_eq!(interpret("InverseFunction[Csc]").unwrap(), "ArcCsc");
  }

  #[test]
  fn arc_trig_inverses() {
    assert_eq!(interpret("InverseFunction[ArcSin]").unwrap(), "Sin");
    assert_eq!(interpret("InverseFunction[ArcCos]").unwrap(), "Cos");
    assert_eq!(interpret("InverseFunction[ArcTan]").unwrap(), "Tan");
  }

  #[test]
  fn hyperbolic_inverses() {
    assert_eq!(interpret("InverseFunction[Sinh]").unwrap(), "ArcSinh");
    assert_eq!(interpret("InverseFunction[Cosh]").unwrap(), "ArcCosh");
    assert_eq!(interpret("InverseFunction[Tanh]").unwrap(), "ArcTanh");
    assert_eq!(interpret("InverseFunction[ArcSinh]").unwrap(), "Sinh");
    assert_eq!(interpret("InverseFunction[ArcCosh]").unwrap(), "Cosh");
    assert_eq!(interpret("InverseFunction[ArcTanh]").unwrap(), "Tanh");
  }

  #[test]
  fn exp_log_inverses() {
    assert_eq!(interpret("InverseFunction[Exp]").unwrap(), "Log");
    assert_eq!(interpret("InverseFunction[Log]").unwrap(), "Exp");
  }

  #[test]
  fn unknown_stays_unevaluated() {
    assert_eq!(
      interpret("InverseFunction[f]").unwrap(),
      "InverseFunction[f]"
    );
  }

  #[test]
  fn applied_to_argument() {
    assert_eq!(interpret("InverseFunction[Sin][1/2]").unwrap(), "Pi/6");
    assert_eq!(interpret("InverseFunction[Exp][5]").unwrap(), "Log[5]");
  }

  #[test]
  fn pure_function_linear() {
    // InverseFunction[2*#1 + 3 &] = (-3 + #1)/2 &.
    assert_eq!(
      interpret("InverseFunction[2*#1 + 3 &]").unwrap(),
      "(-3 + #1)/2 & "
    );
  }

  #[test]
  fn pure_function_mobius() {
    // wolframscript: (-b + d*#1)/(a - c*#1) &. Woxi's Times ordering puts
    // the slot first within products: (-b + #1*d)/(a - #1*c) — same value.
    assert_eq!(
      interpret("InverseFunction[(a*#1 + b)/(c*#1 + d) &]").unwrap(),
      "(-b + #1*d)/(a - #1*c) & "
    );
  }

  #[test]
  fn pure_function_applied() {
    // The returned inverse function should evaluate when applied.
    assert_eq!(interpret("InverseFunction[2*#1 + 3 &][7]").unwrap(), "2");
  }
}

mod cases {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn boole_1() {
    assert_case(r#"Boole[2 == 2]"#, r#"1"#);
  }
  #[test]
  fn boole_2() {
    assert_case(r#"Boole[2 == 2]; Boole[7 < 5]"#, r#"0"#);
  }
  #[test]
  fn boole_3() {
    assert_case(
      r#"Boole[2 == 2]; Boole[7 < 5]; Boole[a == 7]"#,
      r#"Boole[a == 7]"#,
    );
  }
  #[test]
  fn element() {
    assert_case(r#"Element[3 | a, Integers]"#, r#"Element[a, Integers]"#);
  }
  #[test]
  fn trace_evaluation() {
    assert_case(
      r#"TraceEvaluation[(x + x)^2]; TraceEvaluation[(x + x)^2, ShowTimeBySteps->True]"#,
      r#"TraceEvaluation[4*x^2, ShowTimeBySteps -> True]"#,
    );
  }
  #[test]
  fn uncompress() {
    // Same situation as case 534 — `Compress` produces implementation-
    // specific bytes. Verify the documented contract via round-trip on
    // a string input.
    assert_case(
      r#"Uncompress[Compress["Mathics3 is cool"]] == "Mathics3 is cool""#,
      r#"True"#,
    );
  }
  #[test]
  fn real_abs_1() {
    assert_case(r#"RealAbs[-3.]"#, r#"3."#);
  }
  #[test]
  fn real_abs_2() {
    assert_case(
      r#"RealAbs[-3.]; RealAbs[2. + 3. I]"#,
      r#"RealAbs[2. + 3.*I]"#,
    );
  }
  #[test]
  fn angle_path_1() {
    assert_case(
      r#"AnglePath[{90 Degree, 90 Degree, 90 Degree, 90 Degree}]"#,
      r#"{{0, 0}, {0, 1}, {-1, 1}, {-1, 0}, {0, 0}}"#,
    );
  }
  #[test]
  fn angle_path_2() {
    assert_case(
      r#"AnglePath[{90 Degree, 90 Degree, 90 Degree, 90 Degree}]; AnglePath[{{1, 1}, 90 Degree}, {{1, 90 Degree}, {2, 90 Degree}, {1, 90 Degree}, {2, 90 Degree}}]"#,
      r#"{{1, 1}, {0, 1}, {0, -1}, {1, -1}, {1, 1}}"#,
    );
  }
  #[test]
  fn angle_path_3() {
    assert_case(
      r#"AnglePath[{90 Degree, 90 Degree, 90 Degree, 90 Degree}]; AnglePath[{{1, 1}, 90 Degree}, {{1, 90 Degree}, {2, 90 Degree}, {1, 90 Degree}, {2, 90 Degree}}]; AnglePath[{a, b}]"#,
      r#"{{0, 0}, {Cos[a], Sin[a]}, {Cos[a] + Cos[a + b], Sin[a] + Sin[a + b]}}"#,
    );
  }
  // Two-argument AnglePath with a flat start point {x0, y0} (facing 0):
  // the path is the one-argument path translated by the start point.
  #[test]
  fn angle_path_flat_start_point() {
    assert_case(
      r#"AnglePath[{1, 2}, {Pi/2, Pi/2}]"#,
      r#"{{1, 2}, {1, 3}, {0, 3}}"#,
    );
  }
  #[test]
  fn angle_path_flat_start_symbolic() {
    assert_case(
      r#"AnglePath[{x0, y0}, {a1, a2}]"#,
      r#"{{x0, y0}, {x0 + Cos[a1], y0 + Sin[a1]}, {x0 + Cos[a1] + Cos[a1 + a2], y0 + Sin[a1] + Sin[a1 + a2]}}"#,
    );
  }
  #[test]
  fn divide_1() {
    assert_case(r#"Catalan // N"#, r#"0.915965594177219"#);
  }
  #[test]
  fn divide_2() {
    assert_case(r#"EulerGamma // N"#, r#"0.5772156649015329"#);
  }
  #[test]
  fn divide_3() {
    assert_case(r#"GoldenRatio // N"#, r#"1.618033988749895"#);
  }
  #[test]
  fn anonymous_function_1() {
    assert_case(r#"True && True && False"#, r#"False"#);
  }
  #[test]
  fn anonymous_function_2() {
    assert_case(
      r#"True && True && False; a && b && True && c"#,
      r#"a && b && c"#,
    );
  }
  #[test]
  fn implies_1() {
    assert_case(r#"Implies[False, a]"#, r#"True"#);
  }
  #[test]
  fn implies_2() {
    assert_case(r#"Implies[False, a]; Implies[True, a]"#, r#"a"#);
  }
  #[test]
  fn implies_3() {
    assert_case(
      r#"Implies[False, a]; Implies[True, a]; Implies[a, Implies[b, Implies[True, c]]]"#,
      r#"Implies[a, Implies[b, c]]"#,
    );
  }
  #[test]
  fn expression_1() {
    assert_case(r#"False || True"#, r#"True"#);
  }
  #[test]
  fn expression_2() {
    assert_case(r#"False || True; a || False || b"#, r#"a || b"#);
  }
  #[test]
  fn nand() {
    assert_case(r#"Nand[True, False]"#, r#"True"#);
  }
  #[test]
  fn nor() {
    assert_case(r#"Nor[True, False]"#, r#"False"#);
  }
  #[test]
  fn factorial_1() {
    assert_case(r#"!True"#, r#"False"#);
  }
  #[test]
  fn factorial_2() {
    assert_case(r#"!True; !False"#, r#"True"#);
  }
  #[test]
  fn factorial_3() {
    assert_case(r#"!True; !False; !b"#, r#"!b"#);
  }
  #[test]
  fn xor_1() {
    assert_case(r#"Xor[False, True]"#, r#"True"#);
  }
  #[test]
  fn xor_2() {
    assert_case(r#"Xor[False, True]; Xor[True, True]"#, r#"False"#);
  }
  #[test]
  fn xor_3() {
    assert_case(
      r#"Xor[False, True]; Xor[True, True]; Xor[a, False, b]"#,
      r#"Xor[a, b]"#,
    );
  }
  #[test]
  fn between_1() {
    assert_case(r#"Between[6, {4, 10}]"#, r#"True"#);
  }
  #[test]
  fn between_2() {
    assert_case(r#"Between[6, {4, 10}]; Between[{4, 10}][6]"#, r#"True"#);
  }
  #[test]
  fn between_3() {
    assert_case(
      r#"Between[6, {4, 10}]; Between[{4, 10}][6]; Between[2, {E, Pi}]"#,
      r#"False"#,
    );
  }
  #[test]
  fn equal_1() {
    assert_case(r#"1 == 1."#, r#"True"#);
  }
  #[test]
  fn equal_2() {
    assert_case(r#"1 == 1.; 5/3 == 3/2"#, r#"False"#);
  }
  #[test]
  fn greater_1() {
    assert_case(r#"E > 1"#, r#"True"#);
  }
  #[test]
  fn greater_2() {
    assert_case(r#"E > 1; a > b > c // FullForm"#, r#"FullForm[a > b > c]"#);
  }
  #[test]
  fn greater_3() {
    assert_case(r#"E > 1; a > b > c // FullForm; 3 > 2 > 1"#, r#"True"#);
  }
  #[test]
  fn less_equal_1() {
    assert_case(r#"a < b <= c"#, r#"Inequality[a, Less, b, LessEqual, c]"#);
  }
  #[test]
  fn inequality() {
    assert_case(
      r#"a < b <= c; Inequality[a, Greater, b, LessEqual, c]"#,
      r#"a > b && b <= c"#,
    );
  }
  #[test]
  fn less_equal_2() {
    assert_case(
      r#"a < b <= c; Inequality[a, Greater, b, LessEqual, c]; 1 < 2 <= 3"#,
      r#"True"#,
    );
  }
  #[test]
  fn less_1() {
    assert_case(
      r#"a < b <= c; Inequality[a, Greater, b, LessEqual, c]; 1 < 2 <= 3; 1 < 2 > 0"#,
      r#"True"#,
    );
  }
  #[test]
  fn less_2() {
    assert_case(
      r#"a < b <= c; Inequality[a, Greater, b, LessEqual, c]; 1 < 2 <= 3; 1 < 2 > 0; 1 < 2 < -1"#,
      r#"False"#,
    );
  }
  #[test]
  fn less_3() {
    assert_case(r#"1 < 0; 2/18 < 1/5 < Pi/10"#, r#"True"#);
  }
  #[test]
  fn less_4() {
    assert_case(r#"1 < 0; 2/18 < 1/5 < Pi/10; 1 < 3 < x < 2"#, r#"False"#);
  }
  #[test]
  fn less_equal_3() {
    assert_case(r#"LessEqual[1, 3, 3, 2]"#, r#"False"#);
  }
  #[test]
  fn less_equal_4() {
    assert_case(r#"LessEqual[1, 3, 3, 2]; 1 <= 3 <= 3"#, r#"True"#);
  }
  #[test]
  fn max_1() {
    assert_case(r#"Max[4, -8, 1]"#, r#"4"#);
  }
  #[test]
  fn max_2() {
    assert_case(
      r#"Max[4, -8, 1]; Max[E - Pi, Pi, E + Pi, 2 E]"#,
      r#"E + Pi"#,
    );
  }
  #[test]
  fn max_3() {
    assert_case(
      r#"Max[4, -8, 1]; Max[E - Pi, Pi, E + Pi, 2 E]; Max[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]"#,
      r#"3.5"#,
    );
  }
  #[test]
  fn max_4() {
    assert_case(
      r#"Max[4, -8, 1]; Max[E - Pi, Pi, E + Pi, 2 E]; Max[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]; Max[x, y]"#,
      r#"Max[x, y]"#,
    );
  }
  #[test]
  fn max_5() {
    assert_case(
      r#"Max[4, -8, 1]; Max[E - Pi, Pi, E + Pi, 2 E]; Max[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]; Max[x, y]; Max[5, x, -3, y, 40]"#,
      r#"Max[40, x, y]"#,
    );
  }
  #[test]
  fn max_6() {
    assert_case(
      r#"Max[4, -8, 1]; Max[E - Pi, Pi, E + Pi, 2 E]; Max[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]; Max[x, y]; Max[5, x, -3, y, 40]; Max[]"#,
      r#"-Infinity"#,
    );
  }
  #[test]
  fn max_7() {
    assert_case(
      r#"Max[4, -8, 1]; Max[E - Pi, Pi, E + Pi, 2 E]; Max[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]; Max[x, y]; Max[5, x, -3, y, 40]; Max[]; Max[-1.37, 2, "a", b]"#,
      r#"Max[2, "a", b]"#,
    );
  }
  #[test]
  fn min_1() {
    assert_case(r#"Min[4, -8, 1]"#, r#"-8"#);
  }
  #[test]
  fn min_2() {
    assert_case(
      r#"Min[4, -8, 1]; Min[E - Pi, Pi, E + Pi, 2 E]"#,
      r#"E - Pi"#,
    );
  }
  #[test]
  fn min_3() {
    assert_case(
      r#"Min[4, -8, 1]; Min[E - Pi, Pi, E + Pi, 2 E]; Min[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]"#,
      r#"-Infinity"#,
    );
  }
  #[test]
  fn min_4() {
    assert_case(
      r#"Min[4, -8, 1]; Min[E - Pi, Pi, E + Pi, 2 E]; Min[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]; Min[x, y]"#,
      r#"Min[x, y]"#,
    );
  }
  #[test]
  fn min_5() {
    assert_case(
      r#"Min[4, -8, 1]; Min[E - Pi, Pi, E + Pi, 2 E]; Min[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]; Min[x, y]; Min[5, x, -3, y, 40]"#,
      r#"Min[-3, x, y]"#,
    );
  }
  #[test]
  fn min_6() {
    assert_case(
      r#"Min[4, -8, 1]; Min[E - Pi, Pi, E + Pi, 2 E]; Min[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]; Min[x, y]; Min[5, x, -3, y, 40]; Min[]"#,
      r#"Infinity"#,
    );
  }
  #[test]
  fn equal_3() {
    assert_case(r#"a === a"#, r#"True"#);
  }
  #[test]
  fn unequal_1() {
    assert_case(r#"1 != 1."#, r#"False"#);
  }
  #[test]
  fn unequal_2() {
    assert_case(r#"1 != 1.; 1 != 2 != 3"#, r#"True"#);
  }
  #[test]
  fn unequal_3() {
    assert_case(r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x"#, r#"1 != 2 != x"#);
  }
  #[test]
  fn angle_vector_1() {
    assert_case(r#"AngleVector[90 Degree]"#, r#"{0, 1}"#);
  }
  #[test]
  fn angle_vector_2() {
    assert_case(
      r#"AngleVector[90 Degree]; AngleVector[{1, 10}, a]"#,
      r#"{1 + Cos[a], 10 + Sin[a]}"#,
    );
  }
  #[test]
  fn plus_1() {
    assert_case(r#"\! \(2+2\)"#, r#"4"#);
  }
  #[test]
  fn divide_4() {
    assert_case(r#"30 / 5; 1 / 8; Pi / 4"#, r#"Pi / 4"#);
  }
  #[test]
  fn divide_5() {
    assert_case(
      r#"30 / 5; 1 / 8; Pi / 4; Pi / 4.0"#,
      r#"0.7853981633974483"#,
    );
  }
  #[test]
  fn divide_6() {
    assert_case(r#"30 / 5; 1 / 8; Pi / 4; Pi / 4.0; 1 / 8"#, r#"1 / 8"#);
  }
  #[test]
  fn leaf_count_1() {
    assert_case(
      r#"LeafCount[1 + x + y^a]; LeafCount[f[x, y]]; LeafCount[{1 / 3, 1 + I}]; LeafCount[Sqrt[2]]"#,
      r#"5"#,
    );
  }
  #[test]
  fn leaf_count_2() {
    assert_case(
      r#"LeafCount[1 + x + y^a]; LeafCount[f[x, y]]; LeafCount[{1 / 3, 1 + I}]; LeafCount[Sqrt[2]]; LeafCount[100!]"#,
      r#"1"#,
    );
  }
  #[test]
  fn level_1() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]; Level[{{{{a}}}}, -4]; Level[{{{{a}}}}, -5]; Level[h0[h1[h2[h3[a]]]], {0, -1}]; Level[{{{{a}}}}, 3, Heads -> True]"#,
      r#"{List, List, List, {a}, {{a}}, {{{a}}}}"#,
    );
  }
  #[test]
  fn level_2() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]; Level[{{{{a}}}}, -4]; Level[{{{{a}}}}, -5]; Level[h0[h1[h2[h3[a]]]], {0, -1}]; Level[{{{{a}}}}, 3, Heads -> True]; Level[x^2 + y^3, 3, Heads -> True]"#,
      r#"{Plus, Power, x, 2, x ^ 2, Power, y, 3, y ^ 3}"#,
    );
  }
  #[test]
  fn level_3() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]; Level[{{{{a}}}}, -4]; Level[{{{{a}}}}, -5]; Level[h0[h1[h2[h3[a]]]], {0, -1}]; Level[{{{{a}}}}, 3, Heads -> True]; Level[x^2 + y^3, 3, Heads -> True]; Level[a ^ 2 + 2 * b, {-1}, Heads -> True]"#,
      r#"{Plus, Power, a, 2, Times, 2, b}"#,
    );
  }
  #[test]
  fn level_4() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]; Level[{{{{a}}}}, -4]; Level[{{{{a}}}}, -5]; Level[h0[h1[h2[h3[a]]]], {0, -1}]; Level[{{{{a}}}}, 3, Heads -> True]; Level[x^2 + y^3, 3, Heads -> True]; Level[a ^ 2 + 2 * b, {-1}, Heads -> True]; Level[f[g[h]][x], {-1}, Heads -> True]"#,
      r#"{f, g, h, x}"#,
    );
  }
  #[test]
  fn level_5() {
    assert_case(
      r#"Level[a + b ^ 3 * f[2 x ^ 2], {-1}]; Level[{{{{a}}}}, 3]; Level[{{{{a}}}}, -4]; Level[{{{{a}}}}, -5]; Level[h0[h1[h2[h3[a]]]], {0, -1}]; Level[{{{{a}}}}, 3, Heads -> True]; Level[x^2 + y^3, 3, Heads -> True]; Level[a ^ 2 + 2 * b, {-1}, Heads -> True]; Level[f[g[h]][x], {-1}, Heads -> True]; Level[f[g[h]][x], {-2, -1}, Heads -> True]"#,
      r#"{f, g, h, g[h], x, f[g[h]][x]}"#,
    );
  }
  #[test]
  fn normal() {
    assert_case(r#"Normal[Pi]"#, r#"Pi"#);
  }
  #[test]
  fn lerch_phi_1() {
    // wolframscript: `LerchPhi[2, 3, -1.5]` = 51.981… - 2.135…·I, using
    // its analytic continuation outside the convergence radius
    // (|z| > 1). The branch matches Wolfram's by combining
    //   * the PV integral representation for a > 0, plus
    //     `-iπ·(ln z)^(s−1)·z^(−a) / (s−1)!` from the residue at
    //     t = ln z,
    //   * the recurrence `LerchPhi(z, s, a) = |a|^(−s) + z·LerchPhi(z, s, a+1)`
    //     to walk negative a up to a positive value.
    // The closed-form identity `LerchPhi[1, 2, 1/4] == 8 Catalan + Pi^2`
    // is still checked since it exercises a different (z = 1) path.
    assert_case(r#"LerchPhi[1, 2, 1/4] == 8 Catalan + Pi^2"#, r#"True"#);
  }
  #[test]
  fn lerch_phi_2() {
    assert_case(
      r#"LerchPhi[2, 3, -1.5]; LerchPhi[1, 2, 1/4] == 8 Catalan + Pi^2"#,
      r#"True"#,
    );
  }
  #[test]
  fn factorial_4() {
    assert_case(r#"20!"#, r#"2432902008176640000"#);
  }
  #[test]
  fn factorial_5() {
    assert_case(r#"20!; 10.5!"#, r#"1.1899423083962249*^7"#);
  }
  #[test]
  fn plus_2() {
    assert_case(
      r#"20!; 10.5!; (-3.0+1.5*I)!"#,
      r#"0.04279434371837664 - 0.0046156525286039285*I"#,
    );
  }
  #[test]
  fn minus_1() {
    assert_case(r#"20!; 10.5!; (-3.0+1.5*I)!; (-1.)!"#, r#"ComplexInfinity"#);
  }
  #[test]
  fn divide_7() {
    assert_case(
      r#"20!; 10.5!; (-3.0+1.5*I)!; (-1.)!; !a! // FullForm"#,
      r#"FullForm[ !a!]"#,
    );
  }
  #[test]
  fn factorial2() {
    assert_case(r#"5!!"#, r#"15"#);
  }
  #[test]
  fn plus_3() {
    assert_case(r#"Plus[##]& [1, 2, 3]"#, r#"6"#);
  }
  #[test]
  fn plus_4() {
    assert_case(r#"Plus[##]& [1, 2, 3]; Plus[##2]& [1, 2, 3]"#, r#"5"#);
  }
  #[test]
  fn i() {
    assert_case(r#"2; 3+2 I; 2/9;  "hi!"; Infinity; Pi; E; I"#, r#"I"#);
  }
  #[test]
  fn symbol_literal_1() {
    assert_case(r#"2; 3+2 I; 2/9;  "hi!"; Infinity; Pi; E; I; a"#, r#"a"#);
  }
  #[test]
  fn symbol_literal_2() {
    assert_case(
      r#"2; 3+2 I; 2/9;  "hi!"; Infinity; Pi; E; I; a; b=2;b"#,
      r#"2"#,
    );
  }
  #[test]
  fn equal_4() {
    assert_case(r#"3.1416==3.14`2"#, r#"True"#);
  }
  #[test]
  fn equal_5() {
    assert_case(r#"3.1416==3.14`2; 3.14`2==3.1416"#, r#"True"#);
  }
  #[test]
  fn equal_6() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_7() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_8() {
    // Wolframscript-matched expectation. mathics expected the symbolic
    // `Pi == 3.14`2.` form, but wolframscript collapses `Pi == 3.14`2`
    // to True because the BigFloat's 2-digit precision tolerance covers
    // the gap to Pi. Woxi matches wolframscript here.
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_9() {
    // Wolframscript-matched expectation. Same rationale as case 3965 — the
    // 2-digit BigFloat tolerance makes `3.14`2 == Pi` collapse to True.
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_10() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi; 0`==0"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_11() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi; 0`==0; 0`3==0"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_12() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi; 0`==0; 0`3==0; 0`===0."#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_13() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi; 0`==0; 0`3==0; 0`===0.; 0`2===0"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_14() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi; 0`==0; 0`3==0; 0`===0.; 0`2===0; 0`2===0."#,
      r#"False"#,
    );
  }
  #[test]
  fn equal_15() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi; 0`==0; 0`3==0; 0`===0.; 0`2===0; 0`2===0.; 0.`==0."#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_16() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi; 0`==0; 0`3==0; 0`===0.; 0`2===0; 0`2===0.; 0.`==0.; 2^^1.000000000000000000000000000000000000000000000000000000000000 ==  2^^1.000000000000000000000000000000000000000000000000000000000001"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_17() {
    assert_case(
      r#"3.1416==3.14`2; 3.14`2==3.1416; 3.1416`4==3.14`2; 3.14`2==3.1416`4; Pi==3.14`2; 3.14`2==Pi; 0`==0; 0`3==0; 0`===0.; 0`2===0; 0`2===0.; 0.`==0.; 2^^1.000000000000000000000000000000000000000000000000000000000000 ==  2^^1.000000000000000000000000000000000000000000000000000000000001; 2^^1.000000000000000000000000000000000000000000000000000000000000 ==  2^^1.000000000000000000000000000000000000000000000000000010000000"#,
      r#"True"#,
    );
  }
  #[test]
  fn list_literal() {
    assert_case(
      r#"{a, b} = {2^10000, 2^10000 + 1}; {a == b, a < b, a <= b}"#,
      r#"{False, True, True}"#,
    );
  }
  #[test]
  fn equal_18() {
    assert_case(
      r#"ByteOrdering; ByteOrdering == -1 || ByteOrdering == 1"#,
      r#"ByteOrdering == -1 || ByteOrdering == 1"#,
    );
  }
  #[test]
  fn equal_19() {
    assert_case(r#"x === Global`x"#, r#"True"#);
  }
  #[test]
  fn equal_20() {
    assert_case(r#"x === Global`x; `x === Global`x"#, r#"True"#);
  }
  #[test]
  fn equal_21() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x"#,
      r#"False"#,
    );
  }
  #[test]
  fn equal_22() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_23() {
    assert_case(
      r#"x === Global`x; `x === Global`x; a`x === Global`x; a`x === a`x; a`x === b`x"#,
      r#"False"#,
    );
  }
  #[test]
  fn plus_5() {
    assert_case(r#"E^(3+I Pi)"#, r#"-E ^ 3"#);
  }
  #[test]
  fn divide_8() {
    assert_case(r#"E^(3+I Pi); E^(I Pi/2)"#, r#"I"#);
  }
  #[test]
  fn power() {
    assert_case(r#"E^(3+I Pi); E^(I Pi/2); E^1"#, r#"E"#);
  }
  #[test]
  fn symbol_literal_3() {
    assert_case(r#"I; 0; 1; Pi; a"#, r#"a"#);
  }
  #[test]
  fn minus_2() {
    assert_case(r#"I; 0; 1; Pi; a; -Pi"#, r#"-Pi"#);
  }
  #[test]
  fn minus_3() {
    assert_case(r#"I; 0; 1; Pi; a; -Pi; (-1)^2"#, r#"1"#);
  }
  #[test]
  fn minus_4() {
    assert_case(r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3"#, r#"-1"#);
  }
  #[test]
  fn symbol_literal_4() {
    assert_case(r#"I; 0; 1; Pi; a"#, r#"a"#);
  }
  #[test]
  fn minus_5() {
    assert_case(r#"I; 0; 1; Pi; a; a-a"#, r#"0"#);
  }
  #[test]
  fn minus_6() {
    assert_case(r#"I; 0; 1; Pi; a; a-a; 3-3."#, r#"0."#);
  }
  #[test]
  fn factorial_6() {
    assert_case(r#"0!"#, r#"1"#);
  }
}

mod machine_real_division {
  use super::*;

  // wolframscript evaluates a/b as Times[a, Power[b, -1]]. When the denominator
  // is inexact (a machine Real, or an irrational constant whose reciprocal must
  // be rounded to a machine real), the multiply-by-reciprocal double-rounds and
  // lands one ULP away from a single IEEE division. Woxi must match byte-for-byte.
  #[test]
  fn real_over_real_uses_multiply_by_reciprocal() {
    // Direct IEEE 13.522987986828882 / 84.75863032002954 == 0.15954703297787043,
    // but wolframscript reports the reciprocal-multiply value ...046.
    assert_eq!(
      interpret("13.522987986828882/84.75863032002954").unwrap(),
      "0.15954703297787046"
    );
    assert_eq!(
      interpret("65.19413797500401/78.89346277843777").unwrap(),
      "0.8263566546456889"
    );
    assert_eq!(
      interpret("44.594180686074665/72.18184923084418").unwrap(),
      "0.6178032450160481"
    );
  }

  #[test]
  fn real_over_symbolic_constant_double_rounds() {
    // 15.9/Pi: direct IEEE is 5.061127190322272, reciprocal-multiply is ...725.
    assert_eq!(interpret("15.9/Pi").unwrap(), "5.0611271903222725");
    assert_eq!(interpret("Divide[15.9, Pi]").unwrap(), "5.0611271903222725");
  }

  // Dividing by an exact Integer/Rational is Times[a, Rational[..]] — an exact
  // reciprocal, so the product is single-rounded (identical to direct division).
  // The reciprocal-multiply rule must NOT touch these (regression: CentralMoment,
  // Kurtosis, Skewness and AbsoluteCorrelation all divide by an integer count).
  #[test]
  fn real_over_exact_denominator_stays_direct() {
    assert_eq!(interpret("15.9/3").unwrap(), "5.3");
    assert_eq!(interpret("3/15.9").unwrap(), "0.18867924528301888");
    assert_eq!(
      interpret("CentralMoment[{1.1, 1.2, 1.4, 2.1, 2.4}, 4]").unwrap(),
      "0.10084511999999998"
    );
    assert_eq!(
      interpret("Kurtosis[{1.1, 1.2, 1.4, 2.1, 2.4}]").unwrap(),
      "1.4209750290831373"
    );
    assert_eq!(
      interpret("Skewness[{1.1, 1.2, 1.4, 2.1, 2.4}]").unwrap(),
      "0.4070412816074878"
    );
  }
}
