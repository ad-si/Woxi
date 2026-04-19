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

  #[test]
  fn sign_infinity() {
    assert_eq!(interpret("Sign[Infinity]").unwrap(), "1");
  }

  #[test]
  fn sign_negative_infinity() {
    assert_eq!(interpret("Sign[-Infinity]").unwrap(), "-1");
  }

  #[test]
  fn sign_complex_infinity() {
    assert_eq!(interpret("Sign[ComplexInfinity]").unwrap(), "Indeterminate");
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

  #[test]
  fn re_i_times_numeric() {
    // Re[I * Sinh[1]] = 0 since Sinh[1] is a real numeric value
    assert_eq!(interpret("Re[I*Sinh[1]]").unwrap(), "0");
  }

  #[test]
  fn re_i_times_log() {
    assert_eq!(interpret("Re[I*Log[2]]").unwrap(), "0");
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

  #[test]
  fn im_i_times_numeric() {
    // Im[I * Sinh[1]] = Sinh[1] since I*Sinh[1] is purely imaginary
    assert_eq!(interpret("Im[I*Sinh[1]]").unwrap(), "Sinh[1]");
  }

  #[test]
  fn im_i_times_log() {
    assert_eq!(interpret("Im[I*Log[2]]").unwrap(), "Log[2]");
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

  // ── Exp with imaginary multiples of Pi (Euler's formula) ──

  #[test]
  fn exp_i_pi() {
    assert_eq!(interpret("Exp[I Pi]").unwrap(), "-1");
  }

  #[test]
  fn exp_neg_i_pi() {
    assert_eq!(interpret("Exp[-I Pi]").unwrap(), "-1");
  }

  #[test]
  fn exp_i_pi_half() {
    assert_eq!(interpret("Exp[I Pi / 2]").unwrap(), "I");
  }

  #[test]
  fn exp_neg_i_pi_half() {
    assert_eq!(interpret("Exp[-I Pi / 2]").unwrap(), "-I");
  }

  #[test]
  fn exp_2_i_pi() {
    assert_eq!(interpret("Exp[2 I Pi]").unwrap(), "1");
  }

  #[test]
  fn exp_i_pi_third() {
    // Wolfram keeps Exp[I*Pi/3] unevaluated (only evaluates for denom 1 or 2)
    assert_eq!(interpret("Exp[I Pi / 3]").unwrap(), "E^((I/3)*Pi)");
  }

  #[test]
  fn exp_i_pi_sixth() {
    assert_eq!(interpret("Exp[I Pi / 6]").unwrap(), "E^((I/6)*Pi)");
  }

  #[test]
  fn exp_i_pi_fourth() {
    assert_eq!(interpret("Exp[I Pi / 4]").unwrap(), "E^((I/4)*Pi)");
  }

  #[test]
  fn exp_2_i_pi_third() {
    assert_eq!(interpret("Exp[2 I Pi / 3]").unwrap(), "E^(((2*I)/3)*Pi)");
  }

  #[test]
  fn e_to_i_pi() {
    // E^(I*Pi) should also work via Power syntax
    assert_eq!(interpret("E^(I Pi)").unwrap(), "-1");
  }

  #[test]
  fn e_to_i_pi_over_two() {
    // E^(I*Pi/2) = I (Euler's identity, quarter turn).
    assert_eq!(interpret("E^(I Pi/2)").unwrap(), "I");
  }

  #[test]
  fn e_to_quarter_i_pi_real_exponent() {
    // With a machine-real exponent (.25), E^(.25 I Pi) evaluates to the
    // complex machine-real Cos[Pi/4] + I*Sin[Pi/4].
    assert_eq!(
      interpret("E^(.25 I Pi)").unwrap(),
      "0.7071067811865476 + 0.7071067811865475*I"
    );
  }

  #[test]
  fn chop_negates_real_log_sum_with_i_pi() {
    // E^(Log[2.] + I Pi) = -2 + 0.I numerically; Chop removes the tiny
    // imaginary residue, leaving -2.
    assert_eq!(
      interpret("log2=Log[2.]; Chop[E^(log2+I Pi)]").unwrap(),
      "-2."
    );
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
    // Log2[x] for non-power-of-2 returns change-of-base formula (matches Wolfram)
    assert_eq!(interpret("Log2[3]").unwrap(), "Log[3]/Log[2]");
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
    // Log10[x] for non-power-of-10 returns change-of-base formula (matches Wolfram)
    assert_eq!(interpret("Log10[7]").unwrap(), "Log[7]/Log[10]");
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

mod complex_power_tests {
  use woxi::interpret;

  #[test]
  fn complex_power_3_4i_squared() {
    assert_eq!(interpret("(3 + 4I)^2").unwrap(), "-7 + 24*I");
  }

  #[test]
  fn complex_power_3_4i_10() {
    assert_eq!(interpret("(3 + 4I)^10").unwrap(), "-9653287 + 1476984*I");
  }

  #[test]
  fn complex_power_1_i_squared() {
    assert_eq!(interpret("(1 + I)^2").unwrap(), "2*I");
  }

  #[test]
  fn complex_power_1_i_4() {
    assert_eq!(interpret("(1 + I)^4").unwrap(), "-4");
  }

  #[test]
  fn complex_power_2_3i_cubed() {
    assert_eq!(interpret("(2 + 3I)^3").unwrap(), "-46 + 9*I");
  }

  #[test]
  fn complex_power_rational_base() {
    assert_eq!(interpret("(1/2 + I/3)^4").unwrap(), "-119/1296 + (5*I)/54");
  }
}

mod complex_power_numeric {
  use woxi::interpret;

  #[test]
  fn n_i_to_the_i() {
    // I^I = e^(-Pi/2) ≈ 0.2078795763... (preserves complex form with 0.*I)
    assert_eq!(interpret("N[I^I]").unwrap(), "0.20787957635076193 + 0.*I");
  }

  #[test]
  fn n_2_to_the_i() {
    assert_eq!(
      interpret("N[2^I]").unwrap(),
      "0.7692389013639721 + 0.6389612763136348*I"
    );
  }

  #[test]
  fn n_1_plus_i_to_the_1_plus_i() {
    assert_eq!(
      interpret("N[(1 + I)^(1 + I)]").unwrap(),
      "0.2739572538301211 + 0.5837007587586147*I"
    );
  }

  #[test]
  fn n_sqrt_i() {
    assert_eq!(
      interpret("N[Sqrt[I]]").unwrap(),
      "0.7071067811865476 + 0.7071067811865475*I"
    );
  }

  #[test]
  fn n_i_to_the_one_half() {
    assert_eq!(
      interpret("N[I^(1/2)]").unwrap(),
      "0.7071067811865476 + 0.7071067811865475*I"
    );
  }

  #[test]
  fn n_neg1_to_the_one_third() {
    assert_eq!(
      interpret("N[(-1)^(1/3)]").unwrap(),
      "0.5000000000000001 + 0.8660254037844386*I"
    );
  }

  #[test]
  fn i_to_float_exponent() {
    // Direct float exponent (no N wrapper needed)
    assert_eq!(
      interpret("I^0.5").unwrap(),
      "0.7071067811865476 + 0.7071067811865475*I"
    );
  }

  #[test]
  fn complex_float_power() {
    assert_eq!(
      interpret("(1.0 + I)^(2.0 + 3.0 I)").unwrap(),
      "-0.163450932107355 + 0.09600498360894891*I"
    );
  }
}

mod re_im_constants {
  use woxi::interpret;

  #[test]
  fn re_pi() {
    assert_eq!(interpret("Re[Pi]").unwrap(), "Pi");
  }

  #[test]
  fn im_pi() {
    assert_eq!(interpret("Im[Pi]").unwrap(), "0");
  }

  #[test]
  fn re_e() {
    assert_eq!(interpret("Re[E]").unwrap(), "E");
  }

  #[test]
  fn im_e() {
    assert_eq!(interpret("Im[E]").unwrap(), "0");
  }

  #[test]
  fn re_euler_gamma() {
    assert_eq!(interpret("Re[EulerGamma]").unwrap(), "EulerGamma");
  }

  #[test]
  fn im_euler_gamma() {
    assert_eq!(interpret("Im[EulerGamma]").unwrap(), "0");
  }

  #[test]
  fn re_golden_ratio() {
    assert_eq!(interpret("Re[GoldenRatio]").unwrap(), "GoldenRatio");
  }

  #[test]
  fn im_golden_ratio() {
    assert_eq!(interpret("Im[GoldenRatio]").unwrap(), "0");
  }

  #[test]
  fn re_infinity() {
    assert_eq!(interpret("Re[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn im_infinity() {
    assert_eq!(interpret("Im[Infinity]").unwrap(), "0");
  }

  #[test]
  fn re_integer() {
    assert_eq!(interpret("Re[5]").unwrap(), "5");
  }

  #[test]
  fn im_integer() {
    assert_eq!(interpret("Im[5]").unwrap(), "0");
  }

  #[test]
  fn re_complex() {
    assert_eq!(interpret("Re[3 + 4 I]").unwrap(), "3");
  }

  #[test]
  fn im_complex() {
    assert_eq!(interpret("Im[3 + 4 I]").unwrap(), "4");
  }
}

mod re_im_listable {
  use woxi::interpret;

  #[test]
  fn re_threads_over_list() {
    assert_eq!(interpret("Re[{1 + 2 I, 3 - 4 I}]").unwrap(), "{1, 3}");
  }

  #[test]
  fn im_threads_over_list() {
    assert_eq!(interpret("Im[{1 + 2 I, 3 - 4 I}]").unwrap(), "{2, -4}");
  }

  #[test]
  fn conjugate_threads_over_list() {
    assert_eq!(
      interpret("Conjugate[{1 + I, 2 - 3 I}]").unwrap(),
      "{1 - I, 2 + 3*I}"
    );
  }

  #[test]
  fn arg_threads_over_list() {
    assert_eq!(interpret("Arg[{1, -1, I}]").unwrap(), "{0, Pi, Pi/2}");
  }
}

mod arctan_two_arg {
  use woxi::interpret;

  #[test]
  fn arctan2_positive_positive() {
    assert_eq!(interpret("ArcTan[1, 1]").unwrap(), "Pi/4");
  }

  #[test]
  fn arctan2_positive_negative() {
    assert_eq!(interpret("ArcTan[1, -1]").unwrap(), "-1/4*Pi");
  }

  #[test]
  fn arctan2_negative_positive() {
    assert_eq!(interpret("ArcTan[-1, 1]").unwrap(), "(3*Pi)/4");
  }

  #[test]
  fn arctan2_negative_negative() {
    assert_eq!(interpret("ArcTan[-1, -1]").unwrap(), "(-3*Pi)/4");
  }

  #[test]
  fn arctan2_positive_x_axis() {
    assert_eq!(interpret("ArcTan[1, 0]").unwrap(), "0");
  }

  #[test]
  fn arctan2_negative_x_axis() {
    assert_eq!(interpret("ArcTan[-1, 0]").unwrap(), "Pi");
  }

  #[test]
  fn arctan2_positive_y_axis() {
    assert_eq!(interpret("ArcTan[0, 1]").unwrap(), "Pi/2");
  }

  #[test]
  fn arctan2_negative_y_axis() {
    assert_eq!(interpret("ArcTan[0, -1]").unwrap(), "-1/2*Pi");
  }

  #[test]
  fn arctan2_origin_indeterminate() {
    assert_eq!(interpret("ArcTan[0, 0]").unwrap(), "Indeterminate");
  }

  #[test]
  fn arctan2_numeric_negative_negative() {
    assert_eq!(
      interpret("N[ArcTan[-1, -1]]").unwrap(),
      "-2.356194490192345"
    );
  }

  #[test]
  fn arctan2_positive_x_with_symbolic_y() {
    // When x > 0, ArcTan[x, y] = ArcTan[y/x]; exact angles should
    // reduce via the single-argument ArcTan.
    assert_eq!(interpret("ArcTan[1, Sqrt[3]]").unwrap(), "Pi/3");
    assert_eq!(interpret("ArcTan[Sqrt[3], 1]").unwrap(), "Pi/6");
    assert_eq!(interpret("ArcTan[Sqrt[3], 3]").unwrap(), "Pi/3");
  }
}
