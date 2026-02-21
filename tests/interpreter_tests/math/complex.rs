use super::*;

mod sign_complex_tests {
  use woxi::interpret;

  #[test]
  fn sign_positive_integer() {
    assert_eq!(interpret("Sign[19]").unwrap(), "1");
  }

  #[test]
  fn sign_negative_integer() {
    assert_eq!(interpret("Sign[-6]").unwrap(), "-1");
  }

  #[test]
  fn sign_zero() {
    assert_eq!(interpret("Sign[0]").unwrap(), "0");
  }

  #[test]
  fn sign_list() {
    assert_eq!(
      interpret("Sign[{-5, -10, 15, 20, 0}]").unwrap(),
      "{-1, -1, 1, 1, 0}"
    );
  }

  #[test]
  fn sign_complex_pythagorean() {
    // Sign[3 - 4*I] = (3 - 4I) / 5
    assert_eq!(interpret("Sign[3 - 4*I]").unwrap(), "3/5 - (4*I)/5");
  }

  #[test]
  fn sign_complex_positive_imaginary() {
    assert_eq!(interpret("Sign[3 + 4*I]").unwrap(), "3/5 + (4*I)/5");
  }

  #[test]
  fn sign_pure_imaginary() {
    assert_eq!(interpret("Sign[I]").unwrap(), "I");
  }

  #[test]
  fn sign_negative_imaginary() {
    assert_eq!(interpret("Sign[-I]").unwrap(), "-I");
  }

  #[test]
  fn sign_complex_irrational_abs() {
    // Sign[1 + I] = (1 + I) / Sqrt[2]
    assert_eq!(interpret("Sign[1 + I]").unwrap(), "(1 + I)/Sqrt[2]");
  }

  #[test]
  fn sign_complex_irrational_abs_negative() {
    assert_eq!(interpret("Sign[1 - I]").unwrap(), "(1 - I)/Sqrt[2]");
  }

  #[test]
  fn sign_complex_2_plus_i() {
    assert_eq!(interpret("Sign[2 + I]").unwrap(), "(2 + I)/Sqrt[5]");
  }
}

mod abs_complex_tests {
  use woxi::interpret;

  #[test]
  fn abs_complex_pythagorean() {
    assert_eq!(interpret("Abs[3 + 4*I]").unwrap(), "5");
  }

  #[test]
  fn abs_complex_pythagorean_negative() {
    assert_eq!(interpret("Abs[3 - 4*I]").unwrap(), "5");
  }

  #[test]
  fn abs_pure_imaginary() {
    assert_eq!(interpret("Abs[I]").unwrap(), "1");
  }

  #[test]
  fn abs_complex_irrational() {
    assert_eq!(interpret("Abs[1 + I]").unwrap(), "Sqrt[2]");
  }

  #[test]
  fn abs_negative_complex() {
    assert_eq!(interpret("Abs[-3 - 4*I]").unwrap(), "5");
  }

  #[test]
  fn abs_float_complex() {
    assert_eq!(interpret("Abs[3.0 + I]").unwrap(), "3.1622776601683795");
  }

  #[test]
  fn abs_i_infinity() {
    assert_eq!(interpret("Abs[I Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn abs_infinity_equality() {
    assert_eq!(
      interpret("Abs[Infinity] == Abs[I Infinity] == Abs[ComplexInfinity]")
        .unwrap(),
      "True"
    );
  }
}

mod conjugate_tests {
  use woxi::interpret;

  #[test]
  fn conjugate_integer() {
    assert_eq!(interpret("Conjugate[3]").unwrap(), "3");
  }

  #[test]
  fn conjugate_negative_integer() {
    assert_eq!(interpret("Conjugate[-5]").unwrap(), "-5");
  }

  #[test]
  fn conjugate_rational() {
    assert_eq!(interpret("Conjugate[3/4]").unwrap(), "3/4");
  }

  #[test]
  fn conjugate_complex_integer() {
    assert_eq!(interpret("Conjugate[3 + 4*I]").unwrap(), "3 - 4*I");
  }

  #[test]
  fn conjugate_complex_negative_imag() {
    assert_eq!(interpret("Conjugate[3 - 4*I]").unwrap(), "3 + 4*I");
  }

  #[test]
  fn conjugate_pure_imaginary() {
    assert_eq!(interpret("Conjugate[4*I]").unwrap(), "-4*I");
  }

  #[test]
  fn conjugate_i() {
    assert_eq!(interpret("Conjugate[I]").unwrap(), "-I");
  }

  #[test]
  fn conjugate_negative_i() {
    assert_eq!(interpret("Conjugate[-I]").unwrap(), "I");
  }

  #[test]
  fn conjugate_complex_float() {
    assert_eq!(interpret("Conjugate[1.5 + 2.5*I]").unwrap(), "1.5 - 2.5*I");
  }

  #[test]
  fn conjugate_complex_rational() {
    assert_eq!(
      interpret("Conjugate[1/2 + 3/4*I]").unwrap(),
      "1/2 - (3*I)/4"
    );
  }

  #[test]
  fn conjugate_zero() {
    assert_eq!(interpret("Conjugate[0]").unwrap(), "0");
  }

  #[test]
  fn conjugate_symbolic() {
    assert_eq!(interpret("Conjugate[x]").unwrap(), "Conjugate[x]");
  }

  #[test]
  fn conjugate_symbolic_plus_imaginary() {
    // Distributes over Plus, conjugates I factor
    assert_eq!(
      interpret("Conjugate[a + b * I]").unwrap(),
      "Conjugate[a] - I*Conjugate[b]"
    );
  }

  #[test]
  fn conjugate_distribute_over_list() {
    assert_eq!(
      interpret("Conjugate[{1, 2, a}]").unwrap(),
      "{1, 2, Conjugate[a]}"
    );
  }

  #[test]
  fn conjugate_times_i() {
    // Conjugate[I*a] = -I*Conjugate[a]
    assert_eq!(interpret("Conjugate[I*a]").unwrap(), "-I*Conjugate[a]");
  }

  #[test]
  fn conjugate_times_real() {
    // Real coefficient passes through
    assert_eq!(interpret("Conjugate[2*a]").unwrap(), "2*Conjugate[a]");
  }

  #[test]
  fn conjugate_negate_symbolic() {
    // Conjugate[-a] = -Conjugate[a]
    assert_eq!(interpret("Conjugate[-a]").unwrap(), "-Conjugate[a]");
  }

  #[test]
  fn conjugate_nested_complex_list() {
    // Distributes over nested lists with mixed elements
    assert_eq!(
      interpret("Conjugate[{{1, 2 + I 4, a + I b}, {I}}]").unwrap(),
      "{{1, 2 - 4*I, Conjugate[a] - I*Conjugate[b]}, {-I}}"
    );
  }

  #[test]
  fn conjugate_numeric_plus_symbolic() {
    // Conjugate[3 + I*b] = 3 - I*Conjugate[b]
    assert_eq!(
      interpret("Conjugate[3 + I*b]").unwrap(),
      "3 - I*Conjugate[b]"
    );
  }

  #[test]
  fn conjugate_real_plus_symbol() {
    // Conjugate[a + 2] = 2 + Conjugate[a]
    assert_eq!(interpret("Conjugate[a + 2]").unwrap(), "2 + Conjugate[a]");
  }
}

mod re_tests {
  use woxi::interpret;

  #[test]
  fn re_integer() {
    assert_eq!(interpret("Re[3]").unwrap(), "3");
  }

  #[test]
  fn re_real() {
    assert_eq!(interpret("Re[3.14]").unwrap(), "3.14");
  }

  #[test]
  fn re_complex() {
    assert_eq!(interpret("Re[3 + 4*I]").unwrap(), "3");
  }

  #[test]
  fn re_complex_negative_imag() {
    assert_eq!(interpret("Re[3 - 4*I]").unwrap(), "3");
  }

  #[test]
  fn re_pure_imaginary() {
    assert_eq!(interpret("Re[4*I]").unwrap(), "0");
  }

  #[test]
  fn re_i() {
    assert_eq!(interpret("Re[I]").unwrap(), "0");
  }

  #[test]
  fn re_negative_i() {
    assert_eq!(interpret("Re[-I]").unwrap(), "0");
  }

  #[test]
  fn re_zero() {
    assert_eq!(interpret("Re[0]").unwrap(), "0");
  }

  #[test]
  fn re_rational_complex() {
    assert_eq!(interpret("Re[1/2 + 3/4*I]").unwrap(), "1/2");
  }

  #[test]
  fn re_float_complex() {
    assert_eq!(interpret("Re[1.5 + 2.5*I]").unwrap(), "1.5");
  }

  #[test]
  fn re_symbolic() {
    assert_eq!(interpret("Re[x]").unwrap(), "Re[x]");
  }
}

mod im_tests {
  use woxi::interpret;

  #[test]
  fn im_integer() {
    assert_eq!(interpret("Im[3]").unwrap(), "0");
  }

  #[test]
  fn im_real() {
    assert_eq!(interpret("Im[3.14]").unwrap(), "0");
  }

  #[test]
  fn im_complex() {
    assert_eq!(interpret("Im[3 + 4*I]").unwrap(), "4");
  }

  #[test]
  fn im_complex_negative_imag() {
    assert_eq!(interpret("Im[3 - 4*I]").unwrap(), "-4");
  }

  #[test]
  fn im_pure_imaginary() {
    assert_eq!(interpret("Im[4*I]").unwrap(), "4");
  }

  #[test]
  fn im_i() {
    assert_eq!(interpret("Im[I]").unwrap(), "1");
  }

  #[test]
  fn im_negative_i() {
    assert_eq!(interpret("Im[-I]").unwrap(), "-1");
  }

  #[test]
  fn im_zero() {
    assert_eq!(interpret("Im[0]").unwrap(), "0");
  }

  #[test]
  fn im_rational_complex() {
    assert_eq!(interpret("Im[1/2 + 3/4*I]").unwrap(), "3/4");
  }

  #[test]
  fn im_float_complex() {
    assert_eq!(interpret("Im[1.5 + 2.5*I]").unwrap(), "2.5");
  }

  #[test]
  fn im_symbolic() {
    assert_eq!(interpret("Im[x]").unwrap(), "Im[x]");
  }

  // ── Arg ──────────────────────────────────────────────────

  #[test]
  fn arg_positive_integer() {
    assert_eq!(interpret("Arg[3]").unwrap(), "0");
  }

  #[test]
  fn arg_negative_integer() {
    assert_eq!(interpret("Arg[-3]").unwrap(), "Pi");
  }

  #[test]
  fn arg_zero() {
    assert_eq!(interpret("Arg[0]").unwrap(), "0");
  }

  #[test]
  fn arg_positive_rational() {
    assert_eq!(interpret("Arg[1/2]").unwrap(), "0");
  }

  #[test]
  fn arg_negative_rational() {
    assert_eq!(interpret("Arg[-1/2]").unwrap(), "Pi");
  }

  #[test]
  fn arg_pure_imaginary_positive() {
    assert_eq!(interpret("Arg[I]").unwrap(), "Pi/2");
  }

  #[test]
  fn arg_pure_imaginary_negative() {
    assert_eq!(interpret("Arg[-I]").unwrap(), "-1/2*Pi");
  }

  #[test]
  fn arg_first_quadrant() {
    assert_eq!(interpret("Arg[1+I]").unwrap(), "Pi/4");
  }

  #[test]
  fn arg_fourth_quadrant() {
    assert_eq!(interpret("Arg[1-I]").unwrap(), "-1/4*Pi");
  }

  #[test]
  fn arg_second_quadrant() {
    assert_eq!(interpret("Arg[-1+I]").unwrap(), "(3*Pi)/4");
  }

  #[test]
  fn arg_third_quadrant() {
    assert_eq!(interpret("Arg[-1-I]").unwrap(), "(-3*Pi)/4");
  }

  #[test]
  fn arg_scaled_complex() {
    assert_eq!(interpret("Arg[2+2I]").unwrap(), "Pi/4");
  }

  #[test]
  fn arg_non_standard_angle() {
    assert_eq!(interpret("Arg[3-4I]").unwrap(), "-ArcTan[4/3]");
  }

  #[test]
  fn arg_non_standard_second_quadrant() {
    assert_eq!(interpret("Arg[-3+4I]").unwrap(), "Pi - ArcTan[4/3]");
  }

  #[test]
  fn arg_non_standard_third_quadrant() {
    assert_eq!(interpret("Arg[-3-4I]").unwrap(), "-Pi + ArcTan[4/3]");
  }

  #[test]
  fn arg_positive_real() {
    assert_eq!(interpret("Arg[5.0]").unwrap(), "0");
  }

  #[test]
  fn arg_negative_real() {
    assert_eq!(interpret("Arg[-2.5]").unwrap(), "Pi");
  }

  #[test]
  fn arg_symbolic() {
    assert_eq!(interpret("Arg[x]").unwrap(), "Arg[x]");
  }

  // ── RealValuedNumberQ ────────────────────────────────────

  #[test]
  fn real_valued_number_q_integer() {
    assert_eq!(interpret("RealValuedNumberQ[10]").unwrap(), "True");
  }

  #[test]
  fn real_valued_number_q_real() {
    assert_eq!(interpret("RealValuedNumberQ[4.0]").unwrap(), "True");
  }

  #[test]
  fn real_valued_number_q_rational() {
    assert_eq!(interpret("RealValuedNumberQ[3/4]").unwrap(), "True");
  }

  #[test]
  fn real_valued_number_q_complex() {
    assert_eq!(interpret("RealValuedNumberQ[1+I]").unwrap(), "False");
  }

  #[test]
  fn real_valued_number_q_zero_times_i() {
    assert_eq!(interpret("RealValuedNumberQ[0*I]").unwrap(), "True");
  }

  #[test]
  fn real_valued_number_q_pi() {
    assert_eq!(interpret("RealValuedNumberQ[Pi]").unwrap(), "False");
  }

  #[test]
  fn real_valued_number_q_symbol() {
    assert_eq!(interpret("RealValuedNumberQ[x]").unwrap(), "False");
  }

  #[test]
  fn real_valued_number_q_approx_zero_times_i() {
    // 0.0 * I → Complex, not real-valued
    assert_eq!(interpret("RealValuedNumberQ[0.0 * I]").unwrap(), "False");
  }

  #[test]
  fn real_valued_number_q_underflow_overflow() {
    assert_eq!(
      interpret(
        "{RealValuedNumberQ[Underflow[]], RealValuedNumberQ[Overflow[]]}"
      )
      .unwrap(),
      "{True, True}"
    );
  }

  // ── Exp ──────────────────────────────────────────────────

  #[test]
  fn exp_zero() {
    assert_eq!(interpret("Exp[0]").unwrap(), "1");
  }

  #[test]
  fn exp_one() {
    assert_eq!(interpret("Exp[1]").unwrap(), "E");
  }

  // ── Log2 ─────────────────────────────────────────────────

  #[test]
  fn log2_power_of_two() {
    assert_eq!(interpret("Log2[1024]").unwrap(), "10");
  }

  #[test]
  fn log2_large_power() {
    assert_eq!(interpret("Log2[4^8]").unwrap(), "16");
  }

  #[test]
  fn log2_non_power() {
    assert_eq!(interpret("Log2[3]").unwrap(), "Log2[3]");
  }

  // ── Log10 ────────────────────────────────────────────────

  #[test]
  fn log10_power_of_ten() {
    assert_eq!(interpret("Log10[1000]").unwrap(), "3");
  }

  #[test]
  fn log10_million() {
    assert_eq!(interpret("Log10[1000000]").unwrap(), "6");
  }

  #[test]
  fn log10_non_power() {
    assert_eq!(interpret("Log10[7]").unwrap(), "Log10[7]");
  }
}

mod complex_number {
  use super::*;

  #[test]
  fn head_of_complex() {
    assert_eq!(interpret("Head[2 + 3*I]").unwrap(), "Complex");
  }

  #[test]
  fn head_of_i() {
    assert_eq!(interpret("Head[I]").unwrap(), "Complex");
  }

  #[test]
  fn head_of_pure_imaginary() {
    assert_eq!(interpret("Head[3 I]").unwrap(), "Complex");
  }

  #[test]
  fn complex_constructor() {
    assert_eq!(interpret("Complex[1, 2/3]").unwrap(), "1 + (2*I)/3");
  }

  #[test]
  fn complex_constructor_zero_imag() {
    assert_eq!(interpret("Complex[5, 0]").unwrap(), "5");
  }

  #[test]
  fn complex_constructor_zero_real() {
    assert_eq!(interpret("Complex[0, 3]").unwrap(), "3*I");
  }

  #[test]
  fn complex_constructor_i() {
    assert_eq!(interpret("Complex[0, 1]").unwrap(), "I");
  }

  #[test]
  fn abs_complex() {
    assert_eq!(interpret("Abs[Complex[3, 4]]").unwrap(), "5");
  }

  #[test]
  fn complex_conjugate_product() {
    assert_eq!(interpret("(3+I)*(3-I)").unwrap(), "10");
  }

  #[test]
  fn complex_multiplication() {
    assert_eq!(interpret("(2+3*I)*(4+5*I)").unwrap(), "-7 + 22*I");
  }

  #[test]
  fn pure_imaginary_multiplication() {
    assert_eq!(interpret("(2*I)*(3*I)").unwrap(), "-6");
  }

  #[test]
  fn exp_complex() {
    // E^(I*0.5) should give cos(0.5) + I*sin(0.5)
    let result = interpret("E^(I*0.5)").unwrap();
    assert!(result.contains("0.8775825618903728"));
    assert!(result.contains("0.479425538604203"));
  }

  #[test]
  fn im_exp_complex() {
    assert_eq!(interpret("Im[E^(I*0.5)]").unwrap(), "0.479425538604203");
  }

  #[test]
  fn re_exp_complex() {
    assert_eq!(interpret("Re[E^(I*0.5)]").unwrap(), "0.8775825618903728");
  }
}
