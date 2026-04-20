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
    assert_eq!(
      interpret("FullForm[b y^3]").unwrap(),
      "Times[b, Power[y, 3]]"
    );
  }

  #[test]
  fn two_x_squared_y_cubed() {
    assert_eq!(
      interpret("FullForm[2 x^2 y^3]").unwrap(),
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
      "Hold[Times[2, Part[f[x], 1]]]"
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
      "Hold[Times[Optional[Pattern[c, Blank[]]], Power[Pattern[x, Blank[]], 2]]]"
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
    assert_eq!(
      interpret("SixJSymbol[{1, 2, 3}, {4, 5, 12}]").unwrap(),
      "0"
    );
  }

  // For the degenerate valid case the symbol is returned unchanged —
  // we haven't implemented the general Wigner 3-j formula yet.
  #[test]
  fn three_j_valid_returns_unchanged() {
    assert_eq!(
      interpret("ThreeJSymbol[{2, 0}, {6, 0}, {4, 0}]").unwrap(),
      "ThreeJSymbol[{2, 0}, {6, 0}, {4, 0}]"
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
    assert_eq!(interpret("Log[I]").unwrap(), "(I/2)*Pi");
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
}
