use super::*;

mod degree_constant {
  use super::*;

  #[test]
  fn degree_symbolic() {
    assert_eq!(interpret("Degree").unwrap(), "Degree");
  }

  #[test]
  fn degree_numeric() {
    assert_eq!(interpret("N[Degree]").unwrap(), "0.017453292519943295");
  }

  #[test]
  fn degree_arithmetic() {
    assert_eq!(interpret("Sin[30 Degree]").unwrap(), "1/2");
  }

  #[test]
  fn sin_exact_values() {
    assert_eq!(interpret("Sin[0]").unwrap(), "0");
    assert_eq!(interpret("Sin[Pi/6]").unwrap(), "1/2");
    assert_eq!(interpret("Sin[Pi/4]").unwrap(), "1/Sqrt[2]");
    assert_eq!(interpret("Sin[Pi/3]").unwrap(), "Sqrt[3]/2");
    assert_eq!(interpret("Sin[Pi/2]").unwrap(), "1");
    assert_eq!(interpret("Sin[Pi]").unwrap(), "0");
    assert_eq!(interpret("Sin[2 Pi]").unwrap(), "0");
  }

  #[test]
  fn sin_exact_negative() {
    assert_eq!(interpret("Sin[210 Degree]").unwrap(), "-1/2");
    assert_eq!(interpret("Sin[270 Degree]").unwrap(), "-1");
  }

  #[test]
  fn cos_exact_values() {
    assert_eq!(interpret("Cos[0]").unwrap(), "1");
    assert_eq!(interpret("Cos[Pi/6]").unwrap(), "Sqrt[3]/2");
    assert_eq!(interpret("Cos[Pi/4]").unwrap(), "1/Sqrt[2]");
    assert_eq!(interpret("Cos[Pi/3]").unwrap(), "1/2");
    assert_eq!(interpret("Cos[Pi/2]").unwrap(), "0");
    assert_eq!(interpret("Cos[Pi]").unwrap(), "-1");
    assert_eq!(interpret("Cos[2 Pi]").unwrap(), "1");
  }

  #[test]
  fn cos_exact_negative() {
    assert_eq!(interpret("Cos[120 Degree]").unwrap(), "-1/2");
    assert_eq!(interpret("Cos[180 Degree]").unwrap(), "-1");
  }

  #[test]
  fn tan_exact_values() {
    assert_eq!(interpret("Tan[0]").unwrap(), "0");
    assert_eq!(interpret("Tan[Pi/6]").unwrap(), "1/Sqrt[3]");
    assert_eq!(interpret("Tan[Pi/4]").unwrap(), "1");
    assert_eq!(interpret("Tan[Pi/3]").unwrap(), "Sqrt[3]");
    assert_eq!(interpret("Tan[Pi]").unwrap(), "0");
  }

  #[test]
  fn tan_exact_negative() {
    assert_eq!(interpret("Tan[120 Degree]").unwrap(), "-Sqrt[3]");
    assert_eq!(interpret("Tan[135 Degree]").unwrap(), "-1");
  }

  #[test]
  fn tan_complex_infinity() {
    assert_eq!(interpret("Tan[Pi/2]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Tan[90 Degree]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Tan[270 Degree]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Tan[450 Degree]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Tan[3 Pi/2]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn sin_degree_all_quadrants() {
    assert_eq!(interpret("Sin[45 Degree]").unwrap(), "1/Sqrt[2]");
    assert_eq!(interpret("Sin[90 Degree]").unwrap(), "1");
    assert_eq!(interpret("Sin[120 Degree]").unwrap(), "Sqrt[3]/2");
    assert_eq!(interpret("Sin[135 Degree]").unwrap(), "1/Sqrt[2]");
    assert_eq!(interpret("Sin[150 Degree]").unwrap(), "1/2");
    assert_eq!(interpret("Sin[180 Degree]").unwrap(), "0");
    assert_eq!(interpret("Sin[360 Degree]").unwrap(), "0");
  }
}

mod degree_trig_functions {
  use super::*;

  #[test]
  fn sin_degrees_exact() {
    assert_eq!(interpret("SinDegrees[0]").unwrap(), "0");
    assert_eq!(interpret("SinDegrees[30]").unwrap(), "1/2");
    assert_eq!(interpret("SinDegrees[45]").unwrap(), "1/Sqrt[2]");
    assert_eq!(interpret("SinDegrees[60]").unwrap(), "Sqrt[3]/2");
    assert_eq!(interpret("SinDegrees[90]").unwrap(), "1");
    assert_eq!(interpret("SinDegrees[180]").unwrap(), "0");
    assert_eq!(interpret("SinDegrees[360]").unwrap(), "0");
    assert_eq!(interpret("SinDegrees[-30]").unwrap(), "-1/2");
  }

  #[test]
  fn cos_degrees_exact() {
    assert_eq!(interpret("CosDegrees[0]").unwrap(), "1");
    assert_eq!(interpret("CosDegrees[60]").unwrap(), "1/2");
    assert_eq!(interpret("CosDegrees[90]").unwrap(), "0");
    assert_eq!(interpret("CosDegrees[180]").unwrap(), "-1");
    assert_eq!(interpret("CosDegrees[360]").unwrap(), "1");
  }

  #[test]
  fn tan_degrees_exact() {
    assert_eq!(interpret("TanDegrees[0]").unwrap(), "0");
    assert_eq!(interpret("TanDegrees[45]").unwrap(), "1");
    assert_eq!(interpret("TanDegrees[60]").unwrap(), "Sqrt[3]");
    assert_eq!(interpret("TanDegrees[180]").unwrap(), "0");
  }

  #[test]
  fn cot_degrees_exact() {
    assert_eq!(interpret("CotDegrees[45]").unwrap(), "1");
    assert_eq!(interpret("CotDegrees[90]").unwrap(), "0");
  }

  #[test]
  fn sec_degrees_exact() {
    assert_eq!(interpret("SecDegrees[0]").unwrap(), "1");
    assert_eq!(interpret("SecDegrees[60]").unwrap(), "2");
  }

  #[test]
  fn csc_degrees_exact() {
    assert_eq!(interpret("CscDegrees[30]").unwrap(), "2");
    assert_eq!(interpret("CscDegrees[90]").unwrap(), "1");
  }

  #[test]
  fn sin_degrees_numeric() {
    assert_eq!(interpret("SinDegrees[1.0]").unwrap(), "0.01745240643728351");
  }

  #[test]
  fn sin_degrees_symbolic() {
    assert_eq!(interpret("SinDegrees[x]").unwrap(), "SinDegrees[x]");
    assert_eq!(interpret("CosDegrees[x]").unwrap(), "CosDegrees[x]");
    assert_eq!(interpret("TanDegrees[x]").unwrap(), "TanDegrees[x]");
  }

  #[test]
  fn sin_degrees_evaluated_arg() {
    assert_eq!(interpret("SinDegrees[30 + 60]").unwrap(), "1");
  }

  #[test]
  fn arcsin_degrees_exact() {
    assert_eq!(interpret("ArcSinDegrees[1/2]").unwrap(), "30");
    assert_eq!(interpret("ArcSinDegrees[1]").unwrap(), "90");
    assert_eq!(interpret("ArcSinDegrees[0]").unwrap(), "0");
  }

  #[test]
  fn arccos_degrees_exact() {
    assert_eq!(interpret("ArcCosDegrees[1/2]").unwrap(), "60");
    assert_eq!(interpret("ArcCosDegrees[1]").unwrap(), "0");
    assert_eq!(interpret("ArcCosDegrees[0]").unwrap(), "90");
  }

  #[test]
  fn arctan_infinity() {
    assert_eq!(interpret("ArcTan[Infinity]").unwrap(), "Pi/2");
    assert_eq!(interpret("ArcTan[-Infinity]").unwrap(), "-1/2*Pi");
  }

  #[test]
  fn arctan_degrees_exact() {
    assert_eq!(interpret("ArcTanDegrees[1]").unwrap(), "45");
    assert_eq!(interpret("ArcTanDegrees[0]").unwrap(), "0");
  }

  #[test]
  fn arccot_degrees_exact() {
    assert_eq!(interpret("ArcCotDegrees[1]").unwrap(), "45");
  }

  #[test]
  fn arcsec_degrees_exact() {
    assert_eq!(interpret("ArcSecDegrees[2]").unwrap(), "60");
  }

  #[test]
  fn arccsc_degrees_exact() {
    assert_eq!(interpret("ArcCscDegrees[2]").unwrap(), "30");
  }

  #[test]
  fn arc_degrees_symbolic() {
    assert_eq!(interpret("ArcSinDegrees[x]").unwrap(), "ArcSinDegrees[x]");
    assert_eq!(interpret("ArcCosDegrees[x]").unwrap(), "ArcCosDegrees[x]");
    assert_eq!(interpret("ArcTanDegrees[x]").unwrap(), "ArcTanDegrees[x]");
  }

  #[test]
  fn arc_degrees_float() {
    assert_eq!(interpret("ArcTanDegrees[1.0]").unwrap(), "45.");
    assert_eq!(interpret("ArcCotDegrees[1.0]").unwrap(), "45.");
  }

  #[test]
  fn degrees_listable() {
    assert_eq!(interpret("SinDegrees[{0, 30, 90}]").unwrap(), "{0, 1/2, 1}");
    assert_eq!(interpret("CosDegrees[{0, 60, 90}]").unwrap(), "{1, 1/2, 0}");
  }
}

mod e_constant {
  use super::*;

  #[test]
  fn e_symbolic() {
    assert_eq!(interpret("E").unwrap(), "E");
  }

  #[test]
  fn e_numeric() {
    assert_eq!(interpret("N[E]").unwrap(), "2.718281828459045");
  }

  #[test]
  fn e_comparison() {
    assert_eq!(interpret("E > 2").unwrap(), "True");
  }

  #[test]
  fn log_e_is_one() {
    assert_eq!(interpret("Log[E]").unwrap(), "1");
  }

  #[test]
  fn log_e_power() {
    assert_eq!(interpret("Log[E^3]").unwrap(), "3");
  }

  #[test]
  fn log_e_power_symbolic() {
    assert_eq!(interpret("Log[E^n]").unwrap(), "Log[E^n]");
  }

  #[test]
  fn numeric_q_e() {
    assert_eq!(interpret("NumericQ[E]").unwrap(), "True");
  }

  #[test]
  fn numeric_q_pi() {
    assert_eq!(interpret("NumericQ[Pi]").unwrap(), "True");
  }

  #[test]
  fn numeric_q_sqrt_pi() {
    assert_eq!(interpret("NumericQ[Sqrt[Pi]]").unwrap(), "True");
  }

  #[test]
  fn numeric_q_sin_integer() {
    assert_eq!(interpret("NumericQ[Sin[1]]").unwrap(), "True");
  }

  #[test]
  fn numeric_q_compound() {
    assert_eq!(interpret("NumericQ[Log[2] + Sin[3]]").unwrap(), "True");
  }

  #[test]
  fn numeric_q_symbol() {
    assert_eq!(interpret("NumericQ[x]").unwrap(), "False");
  }

  #[test]
  fn numeric_q_i() {
    assert_eq!(interpret("NumericQ[I]").unwrap(), "True");
  }

  #[test]
  fn numeric_q_downvalue_true() {
    clear_state();
    assert_eq!(
      interpret("NumericQ[a]=True; NumericQ[Sqrt[a]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn numeric_q_downvalue_false() {
    clear_state();
    assert_eq!(
      interpret("NumericQ[a]=False; NumericQ[Sqrt[a]]").unwrap(),
      "False"
    );
  }

  #[test]
  fn e_plus_e() {
    assert_eq!(interpret("E + E").unwrap(), "2*E");
  }

  #[test]
  fn e_plus_e_plus_e() {
    assert_eq!(interpret("E + E + E").unwrap(), "3*E");
  }

  #[test]
  fn e_like_term_collection() {
    assert_eq!(interpret("3*E + 2*E").unwrap(), "5*E");
  }

  #[test]
  fn e_plus_pi() {
    assert_eq!(interpret("E + Pi + E").unwrap(), "2*E + Pi");
  }

  #[test]
  fn e_times_numeric() {
    assert_eq!(interpret("3*E").unwrap(), "3*E");
  }

  #[test]
  fn e_power() {
    assert_eq!(interpret("E^2").unwrap(), "E^2");
  }

  #[test]
  fn n_e_high_precision() {
    // First 50 significant digits of E must be correct
    let result = interpret("N[E, 50]").unwrap();
    assert!(
      result.starts_with("2.7182818284590452353602874713526624977572470936999"),
      "N[E, 50] = {}",
      result
    );
    assert!(result.ends_with("`50."));
  }
}

mod pi_symbolic {
  use super::*;

  #[test]
  fn pi_stays_symbolic() {
    assert_eq!(interpret("Pi").unwrap(), "Pi");
  }

  #[test]
  fn pi_comparison() {
    assert_eq!(interpret("Pi > 3").unwrap(), "True");
  }

  #[test]
  fn pi_less() {
    assert_eq!(interpret("Pi < 4").unwrap(), "True");
  }
}

mod directed_infinity {
  use super::*;

  #[test]
  fn positive() {
    assert_eq!(interpret("DirectedInfinity[1]").unwrap(), "Infinity");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("DirectedInfinity[-1]").unwrap(), "-Infinity");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("DirectedInfinity[0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn no_arg() {
    assert_eq!(interpret("DirectedInfinity[]").unwrap(), "ComplexInfinity");
  }
}

mod infinity_arithmetic {
  use super::*;

  #[test]
  fn infinity_plus_neg_infinity() {
    assert_eq!(
      interpret("Infinity + (-Infinity)").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn directed_infinity_add() {
    assert_eq!(
      interpret("DirectedInfinity[1] + DirectedInfinity[-1]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn directed_infinity_normalization() {
    assert_eq!(
      interpret("DirectedInfinity[1 + I]").unwrap(),
      "DirectedInfinity[(1 + I)/Sqrt[2]]"
    );
  }

  #[test]
  fn finite_divided_by_directed_infinity() {
    assert_eq!(interpret("1 / DirectedInfinity[1 + I]").unwrap(), "0");
  }

  #[test]
  fn finite_divided_by_infinity() {
    assert_eq!(interpret("1 / Infinity").unwrap(), "0");
  }

  #[test]
  fn directed_infinity_positive_real() {
    assert_eq!(interpret("DirectedInfinity[5]").unwrap(), "Infinity");
  }

  #[test]
  fn directed_infinity_negative_real() {
    assert_eq!(interpret("DirectedInfinity[-3]").unwrap(), "-Infinity");
  }
}

mod constant_real_arithmetic {
  use super::*;

  #[test]
  fn pi_divided_by_real() {
    assert_eq!(interpret("Pi / 4.0").unwrap(), "0.7853981633974483");
  }

  #[test]
  fn pi_times_real() {
    assert_eq!(interpret("Pi * 2.0").unwrap(), "6.283185307179586");
  }

  #[test]
  fn pi_times_integer_stays_symbolic() {
    assert_eq!(interpret("2 * Pi").unwrap(), "2*Pi");
  }
}
