use super::*;

mod integer_part {
  use super::*;

  #[test]
  fn positive_float() {
    assert_eq!(interpret("IntegerPart[3.7]").unwrap(), "3");
  }

  #[test]
  fn negative_float() {
    assert_eq!(interpret("IntegerPart[-3.7]").unwrap(), "-3");
  }

  #[test]
  fn integer() {
    assert_eq!(interpret("IntegerPart[5]").unwrap(), "5");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("IntegerPart[7/3]").unwrap(), "2");
  }

  #[test]
  fn negative_rational() {
    assert_eq!(interpret("IntegerPart[-7/3]").unwrap(), "-2");
  }
}

mod fractional_part {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("FractionalPart[5]").unwrap(), "0");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("FractionalPart[7/3]").unwrap(), "1/3");
  }

  #[test]
  fn negative_rational() {
    assert_eq!(interpret("FractionalPart[-7/3]").unwrap(), "-1/3");
  }

  #[test]
  fn symbolic_pi() {
    assert_eq!(interpret("FractionalPart[Pi]").unwrap(), "-3 + Pi");
  }

  #[test]
  fn symbolic_e() {
    assert_eq!(interpret("FractionalPart[E]").unwrap(), "-2 + E");
  }

  #[test]
  fn symbolic_golden_ratio() {
    assert_eq!(
      interpret("FractionalPart[GoldenRatio]").unwrap(),
      "-1 + GoldenRatio"
    );
  }

  #[test]
  fn symbolic_euler_gamma() {
    // EulerGamma ≈ 0.577, so FractionalPart is just EulerGamma itself
    assert_eq!(
      interpret("FractionalPart[EulerGamma]").unwrap(),
      "EulerGamma"
    );
  }
}

mod mixed_fraction_parts {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("MixedFractionParts[5]").unwrap(), "{5, 0}");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("MixedFractionParts[7/3]").unwrap(), "{2, 1/3}");
  }

  #[test]
  fn negative_rational() {
    assert_eq!(interpret("MixedFractionParts[-7/3]").unwrap(), "{-2, -1/3}");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("MixedFractionParts[0]").unwrap(), "{0, 0}");
  }

  #[test]
  fn proper_fraction() {
    assert_eq!(interpret("MixedFractionParts[1/3]").unwrap(), "{0, 1/3}");
  }

  #[test]
  fn negative_proper_fraction() {
    assert_eq!(interpret("MixedFractionParts[-1/3]").unwrap(), "{0, -1/3}");
  }

  #[test]
  fn listable() {
    assert_eq!(
      interpret("MixedFractionParts[{7/3, 5, 1/2}]").unwrap(),
      "{{2, 1/3}, {5, 0}, {0, 1/2}}"
    );
  }
}

mod floor {
  use super::*;

  #[test]
  fn positive_float() {
    assert_eq!(interpret("Floor[3.7]").unwrap(), "3");
  }

  #[test]
  fn negative_float() {
    assert_eq!(interpret("Floor[-2.3]").unwrap(), "-3");
  }

  #[test]
  fn integer_unchanged() {
    assert_eq!(interpret("Floor[5]").unwrap(), "5");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("Floor[0]").unwrap(), "0");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("Floor[x]").unwrap(), "Floor[x]");
  }

  // `Floor[Pi * 10^20]` must produce the 21-digit integer exactly. Using
  // f64 only preserves ~16 integer digits, so the last digits would drift;
  // Woxi falls back to an arbitrary-precision BigFloat path for values
  // beyond 1e15.
  #[test]
  fn pi_times_10_pow_20() {
    assert_eq!(
      interpret("Floor[Pi * 10^20]").unwrap(),
      "314159265358979323846"
    );
  }

  // Regression for mathics numbers/linalg.py `DigitCount[Floor[Pi *
  // 10^100]]`: the i128 saturation at 2^127-1 used to give a 39-digit
  // integer; we now get the true 101-digit integer via BigFloat.
  #[test]
  fn pi_times_10_pow_100_full_digits() {
    assert_eq!(
      interpret("DigitCount[Floor[Pi * 10^100]]").unwrap(),
      "{8, 12, 12, 10, 8, 9, 8, 12, 14, 8}"
    );
  }

  // Negative arbitrary-precision case: Floor rounds toward -Infinity.
  #[test]
  fn neg_pi_times_10_pow_20() {
    assert_eq!(
      interpret("Floor[-Pi * 10^20]").unwrap(),
      "-314159265358979323847"
    );
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("Floor[7/3]").unwrap(), "2");
    assert_eq!(interpret("Floor[-7/3]").unwrap(), "-3");
    assert_eq!(interpret("Floor[3/2]").unwrap(), "1");
    assert_eq!(interpret("Floor[-3/2]").unwrap(), "-2");
  }

  #[test]
  fn two_arg_integer_step() {
    assert_eq!(interpret("Floor[10, 3]").unwrap(), "9");
    assert_eq!(interpret("Floor[7, 2]").unwrap(), "6");
    assert_eq!(interpret("Floor[5.8, 2]").unwrap(), "4");
    assert_eq!(interpret("Floor[-5.5, 2]").unwrap(), "-6");
  }

  #[test]
  fn two_arg_rational_step() {
    assert_eq!(interpret("Floor[7/2, 1/3]").unwrap(), "10/3");
  }

  // An exact rational step keeps the result exact even when the first
  // argument is a symbolic constant — Floor[Pi, 1/10] is 31/10, not 3.1.
  #[test]
  fn two_arg_exact_step_symbolic_arg() {
    assert_eq!(interpret("Floor[Pi, 1/10]").unwrap(), "31/10");
    assert_eq!(interpret("Floor[E, 1/10]").unwrap(), "27/10");
    assert_eq!(interpret("Floor[Sqrt[2], 1/10]").unwrap(), "7/5");
    assert_eq!(interpret("Floor[-Pi, 1/10]").unwrap(), "-16/5");
  }

  // The result follows the (exact) step, so a machine-float first argument
  // still yields an exact rational: Floor[2.7, 1/10] is 27/10, not 2.7.
  #[test]
  fn two_arg_exact_step_float_arg() {
    assert_eq!(interpret("Floor[2.7, 1/10]").unwrap(), "27/10");
    assert_eq!(interpret("Floor[3.14159, 1/100]").unwrap(), "157/50");
  }

  #[test]
  fn two_arg_float_step() {
    assert_eq!(interpret("Floor[5.5, 0.5]").unwrap(), "5.5");
    assert_eq!(interpret("Floor[10, 3.]").unwrap(), "9.");
  }

  #[test]
  fn two_arg_list() {
    assert_eq!(interpret("Floor[{2.5, 3.7}, 2]").unwrap(), "{2, 2}");
  }
}

mod ceiling_two_arg {
  use super::*;

  #[test]
  fn integer_step() {
    assert_eq!(interpret("Ceiling[10, 3]").unwrap(), "12");
    assert_eq!(interpret("Ceiling[5.8, 2]").unwrap(), "6");
    assert_eq!(interpret("Ceiling[-5.5, 2]").unwrap(), "-4");
  }

  #[test]
  fn rational_step() {
    assert_eq!(interpret("Ceiling[7/2, 1/3]").unwrap(), "11/3");
  }

  // Exact step keeps the result exact for symbolic and float first arguments.
  #[test]
  fn exact_step_symbolic_and_float_arg() {
    assert_eq!(interpret("Ceiling[Pi, 1/10]").unwrap(), "16/5");
    assert_eq!(interpret("Ceiling[Sqrt[2], 1/100]").unwrap(), "71/50");
    assert_eq!(interpret("Ceiling[2.71, 1/10]").unwrap(), "14/5");
  }

  #[test]
  fn float_step() {
    assert_eq!(interpret("Ceiling[10, 3.]").unwrap(), "12.");
  }

  #[test]
  fn list() {
    assert_eq!(interpret("Ceiling[{2.5, 3.7}, 2]").unwrap(), "{4, 4}");
  }
}

mod round {
  use super::*;

  #[test]
  fn round_integer() {
    assert_eq!(interpret("Round[3]").unwrap(), "3");
  }

  #[test]
  fn round_real() {
    assert_eq!(interpret("Round[2.6]").unwrap(), "3");
  }

  #[test]
  fn round_two_args() {
    assert_eq!(interpret("Round[3.14159, 0.01]").unwrap(), "3.14");
  }

  #[test]
  fn round_to_tens() {
    assert_eq!(interpret("Round[37, 10]").unwrap(), "40");
  }

  #[test]
  fn round_two_args_bankers_rounding() {
    // Banker's rounding: ties round to even
    assert_eq!(interpret("Round[2.5, 1]").unwrap(), "2");
    assert_eq!(interpret("Round[3.5, 1]").unwrap(), "4");
    assert_eq!(interpret("Round[4.5, 1]").unwrap(), "4");
  }

  #[test]
  fn round_two_args_bankers_decimal() {
    assert_eq!(interpret("Round[3.45, 0.1]").unwrap(), "3.4000000000000004");
  }

  #[test]
  fn round_two_args_integer_step_returns_integer() {
    // When step is Integer, result should be Integer
    assert_eq!(interpret("Round[2.7, 1]").unwrap(), "3");
    assert_eq!(interpret("Round[2.3, 1]").unwrap(), "2");
  }

  // Round[a + b I, c] rounds the real and imaginary parts componentwise.
  #[test]
  fn round_two_args_complex() {
    // Float step keeps the result a machine real.
    assert_eq!(interpret("Round[1.2 + 3.8 I, 0.5]").unwrap(), "1. + 4.*I");
    // Rational step gives an exact result.
    assert_eq!(interpret("Round[1.2 + 3.8 I, 1/2]").unwrap(), "1 + 4*I");
    assert_eq!(
      interpret("Round[2.3 + 4.7 I, 1/10]").unwrap(),
      "23/10 + (47*I)/10"
    );
    // Already a multiple of the step is unchanged.
    assert_eq!(
      interpret("Round[5/2 + 7/2 I, 1/2]").unwrap(),
      "5/2 + (7*I)/2"
    );
  }
}

mod floor_ceiling_two_arg_complex {
  use super::*;

  // Floor/Ceiling[a + b I, c] act componentwise on the real and imaginary parts.
  #[test]
  fn floor_complex() {
    assert_eq!(interpret("Floor[2.7 + 3.2 I, 0.5]").unwrap(), "2.5 + 3.*I");
  }

  #[test]
  fn ceiling_complex() {
    assert_eq!(
      interpret("Ceiling[1.1 + 2.9 I, 0.5]").unwrap(),
      "1.5 + 3.*I"
    );
    // Integer step yields integer Gaussian components.
    assert_eq!(interpret("Ceiling[2.2 + 3.3 I, 1]").unwrap(), "3 + 4*I");
  }

  // Symbolic-real components (Pi + E I) resolve through the scalar path.
  #[test]
  fn symbolic_components() {
    assert_eq!(interpret("Floor[Pi + E I, 1/2]").unwrap(), "3 + (5*I)/2");
  }
}

mod cube_root {
  use super::*;

  #[test]
  fn perfect_cube() {
    assert_eq!(interpret("CubeRoot[8]").unwrap(), "2");
  }

  #[test]
  fn negative_perfect_cube() {
    assert_eq!(interpret("CubeRoot[-27]").unwrap(), "-3");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("CubeRoot[0]").unwrap(), "0");
  }

  #[test]
  fn one() {
    assert_eq!(interpret("CubeRoot[1]").unwrap(), "1");
  }

  #[test]
  fn large_cube() {
    assert_eq!(interpret("CubeRoot[1000]").unwrap(), "10");
  }

  #[test]
  fn non_perfect_cube_symbolic() {
    // Non-perfect cubes return n^(1/3)
    assert_eq!(interpret("CubeRoot[2]").unwrap(), "2^(1/3)");
  }

  #[test]
  fn non_perfect_cube_with_factor() {
    // CubeRoot[16] = CubeRoot[8*2] = 2 * 2^(1/3)
    assert_eq!(interpret("CubeRoot[16]").unwrap(), "2*2^(1/3)");
  }

  #[test]
  fn cube_factor_54() {
    // CubeRoot[54] = CubeRoot[27*2] = 3 * 2^(1/3)
    assert_eq!(interpret("CubeRoot[54]").unwrap(), "3*2^(1/3)");
  }

  #[test]
  fn negative_with_factor() {
    // CubeRoot[-16] = -2 * 2^(1/3)
    assert_eq!(interpret("CubeRoot[-16]").unwrap(), "-2*2^(1/3)");
  }
}

mod sqrt_rational {
  use woxi::interpret;

  #[test]
  fn perfect_square_denominator() {
    // Sqrt[13297/4] should simplify to Sqrt[13297]/2
    assert_eq!(interpret("Sqrt[13297/4]").unwrap(), "Sqrt[13297]/2");
  }

  #[test]
  fn both_perfect_squares() {
    assert_eq!(interpret("Sqrt[9/4]").unwrap(), "3/2");
  }

  #[test]
  fn perfect_square_numerator() {
    assert_eq!(interpret("Sqrt[4/7]").unwrap(), "2/Sqrt[7]");
  }
}

mod rational_power {
  use super::*;

  #[test]
  fn negative_rational_negative_power() {
    assert_eq!(interpret("(-2/3)^(-3)").unwrap(), "-27/8");
  }

  #[test]
  fn rational_positive_power() {
    assert_eq!(interpret("(2/3)^3").unwrap(), "8/27");
  }

  #[test]
  fn rational_power_simplifies() {
    assert_eq!(interpret("(1/2)^4").unwrap(), "1/16");
  }
}

mod cases {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn round_1() {
    assert_case(r#"Round[10.6]"#, r#"11"#);
  }
  #[test]
  fn round_2() {
    assert_case(r#"Round[10.6]; Round[0.06, 0.1]"#, r#"0.1"#);
  }
  #[test]
  fn round_3() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]"#,
      r#"0."#,
    );
  }
  #[test]
  fn round_4() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]"#,
      r#"3."#,
    );
  }
  #[test]
  fn round_5() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]; Round[Pi^2]"#,
      r#"10"#,
    );
  }
  #[test]
  fn round_6() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]; Round[Pi^2]; Round[2.6, 1/3]"#,
      r#"8 / 3"#,
    );
  }
  #[test]
  fn round_7() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]; Round[Pi^2]; Round[2.6, 1/3]; Round[10, Pi]"#,
      r#"3*Pi"#,
    );
  }
  #[test]
  fn round_8() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]; Round[Pi^2]; Round[2.6, 1/3]; Round[10, Pi]; Round[6/(2 + 3 I)]"#,
      r#"1 - I"#,
    );
  }
  #[test]
  fn round_9() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]; Round[Pi^2]; Round[2.6, 1/3]; Round[10, Pi]; Round[6/(2 + 3 I)]; Round[1 + 2 I, 2 I]"#,
      r#"2*I"#,
    );
  }
  #[test]
  fn round_10() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]; Round[Pi^2]; Round[2.6, 1/3]; Round[10, Pi]; Round[6/(2 + 3 I)]; Round[1 + 2 I, 2 I]; Round[-1.4]"#,
      r#"-1"#,
    );
  }
  #[test]
  fn round_11() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]; Round[Pi^2]; Round[2.6, 1/3]; Round[10, Pi]; Round[6/(2 + 3 I)]; Round[1 + 2 I, 2 I]; Round[-1.4]; Round[x]"#,
      r#"Round[x]"#,
    );
  }
  #[test]
  fn round_12() {
    assert_case(
      r#"Round[10.6]; Round[0.06, 0.1]; Round[0.04, 0.1]; Round[Pi, .5]; Round[Pi^2]; Round[2.6, 1/3]; Round[10, Pi]; Round[6/(2 + 3 I)]; Round[1 + 2 I, 2 I]; Round[-1.4]; Round[x]; Round[1.5, k]"#,
      r#"Round[1.5, k]"#,
    );
  }
  #[test]
  fn ceiling_1() {
    assert_case(r#"Ceiling[1.2]"#, r#"2"#);
  }
  #[test]
  fn ceiling_2() {
    assert_case(r#"Ceiling[1.2]; Ceiling[3/2]"#, r#"2"#);
  }
  #[test]
  fn ceiling_3() {
    assert_case(
      r#"Ceiling[1.2]; Ceiling[3/2]; Ceiling[1.3 + 0.7 I]"#,
      r#"2 + I"#,
    );
  }
  #[test]
  fn floor_1() {
    assert_case(r#"Floor[10.4]"#, r#"10"#);
  }
  #[test]
  fn floor_2() {
    assert_case(r#"Floor[10.4]; Floor[10/3]"#, r#"3"#);
  }
  #[test]
  fn floor_3() {
    assert_case(r#"Floor[10.4]; Floor[10/3]; Floor[10]"#, r#"10"#);
  }
  #[test]
  fn floor_4() {
    assert_case(
      r#"Floor[10.4]; Floor[10/3]; Floor[10]; Floor[21, 2]"#,
      r#"20"#,
    );
  }
  #[test]
  fn floor_5() {
    assert_case(
      r#"Floor[10.4]; Floor[10/3]; Floor[10]; Floor[21, 2]; Floor[2.6, 0.5]"#,
      r#"2.5"#,
    );
  }
  #[test]
  fn floor_6() {
    assert_case(
      r#"Floor[10.4]; Floor[10/3]; Floor[10]; Floor[21, 2]; Floor[2.6, 0.5]; Floor[-10.4]"#,
      r#"-11"#,
    );
  }
  #[test]
  fn floor_7() {
    assert_case(
      r#"Floor[10.4]; Floor[10/3]; Floor[10]; Floor[21, 2]; Floor[2.6, 0.5]; Floor[-10.4]; Floor[1.5 + 2.7 I]"#,
      r#"1 + 2*I"#,
    );
  }
  #[test]
  fn floor_8() {
    assert_case(
      r#"Floor[10.4]; Floor[10/3]; Floor[10]; Floor[21, 2]; Floor[2.6, 0.5]; Floor[-10.4]; Floor[1.5 + 2.7 I]; Floor[10.4, -1]"#,
      r#"11"#,
    );
  }
  #[test]
  fn floor_9() {
    assert_case(
      r#"Floor[10.4]; Floor[10/3]; Floor[10]; Floor[21, 2]; Floor[2.6, 0.5]; Floor[-10.4]; Floor[1.5 + 2.7 I]; Floor[10.4, -1]; Floor[-10.4, -1]"#,
      r#"-10"#,
    );
  }
  #[test]
  fn fractional_part_1() {
    assert_case(r#"FractionalPart[4.1]"#, r#"0.09999999999999964"#);
  }
  #[test]
  fn fractional_part_2() {
    assert_case(r#"FractionalPart[4.1]; FractionalPart[-5.25]"#, r#"-0.25"#);
  }
  #[test]
  fn integer_part_1() {
    assert_case(r#"IntegerPart[4.1]"#, r#"4"#);
  }
  #[test]
  fn integer_part_2() {
    assert_case(r#"IntegerPart[4.1]; IntegerPart[-5.25]"#, r#"-5"#);
  }
  #[test]
  fn mod_1() {
    assert_case(
      r#"Mod[$RandomState, 10^100]"#,
      r#"4741994566655706294890138869165136649510315974360597429933393392703942354819024473254400806573416326"#,
    );
  }
  #[test]
  fn math_ml_form() {
    // mathics rendered the contents to MathML XML; wolframscript -code
    // returns the unevaluated wrapper `MathMLForm[HoldForm[Sqrt[a^3]]]`
    // verbatim. Woxi matches wolframscript.
    assert_case(
      r#"MathMLForm[HoldForm[Sqrt[a^3]]]"#,
      r#"MathMLForm[HoldForm[Sqrt[a^3]]]"#,
    );
  }
  #[test]
  fn te_x_form() {
    // mathics rendered the contents as `\sqrt{a^3}`; wolframscript -code
    // returns the unevaluated wrapper `TeXForm[HoldForm[Sqrt[a^3]]]`
    // verbatim. Woxi matches wolframscript.
    assert_case(
      r#"TeXForm[HoldForm[Sqrt[a^3]]]"#,
      r#"TeXForm[HoldForm[Sqrt[a^3]]]"#,
    );
  }
  #[test]
  fn cube_root_1() {
    assert_case(r#"CubeRoot[16]"#, r#"2*2^(1/3)"#);
  }
  #[test]
  fn sqrt_1() {
    assert_case(r#"Sqrt[4]"#, r#"2"#);
  }
  #[test]
  fn sqrt_2() {
    assert_case(r#"Sqrt[4]; Sqrt[5]"#, r#"Sqrt[5]"#);
  }
  #[test]
  fn sqrt_3() {
    assert_case(r#"Sqrt[4]; Sqrt[5]; Sqrt[5] // N"#, r#"2.23606797749979"#);
  }
  #[test]
  fn sqrt_4() {
    assert_case(r#"Sqrt[4]; Sqrt[5]; Sqrt[5] // N; Sqrt[a]^2"#, r#"a"#);
  }
  #[test]
  fn sqrt_5() {
    assert_case(
      r#"Sqrt[4]; Sqrt[5]; Sqrt[5] // N; Sqrt[a]^2; Sqrt[-4]"#,
      r#"2*I"#,
    );
  }
  #[test]
  fn equal() {
    assert_case(
      r#"Sqrt[4]; Sqrt[5]; Sqrt[5] // N; Sqrt[a]^2; Sqrt[-4]; I == Sqrt[-1]"#,
      r#"True"#,
    );
  }
  #[test]
  fn mod_2() {
    assert_case(r#"Mod[14, 6]"#, r#"2"#);
  }
  #[test]
  fn mod_3() {
    assert_case(r#"Mod[14, 6]; Mod[-3, 4]"#, r#"1"#);
  }
  #[test]
  fn mod_4() {
    assert_case(r#"Mod[14, 6]; Mod[-3, 4]; Mod[-3, -4]"#, r#"-3"#);
  }
  #[test]
  fn mod_5() {
    assert_case(
      r#"ModularInverse[2, 3]; k = 2; n = 3; Mod[ModularInverse[k, n] * k, n] == 1"#,
      r#"True"#,
    );
  }
  // Gaussian Mod: Mod[m, n] = m - n*Round[m/n] with the complex quotient
  // rounded component-wise (round-half-to-even).
  #[test]
  fn mod_gaussian_integer() {
    assert_case(r#"Mod[7 + 3 I, 2]"#, r#"-1 - I"#);
  }
  #[test]
  fn mod_gaussian_complex_modulus() {
    assert_case(r#"Mod[1 + 6 I, 2 + I]"#, r#"-1"#);
  }
  #[test]
  fn mod_real_by_gaussian() {
    assert_case(r#"Mod[5, 2 + I]"#, r#"0"#);
  }
  #[test]
  fn mod_gaussian_rational() {
    assert_case(r#"Mod[7/2 + 3 I, 2]"#, r#"-1/2 - I"#);
  }
  #[test]
  fn sqrt_6() {
    assert_case(r#"1; Sqrt[2]"#, r#"Sqrt[2]"#);
  }
  #[test]
  fn divide_1() {
    assert_case(r#"1; Sqrt[2]; 2/9"#, r#"2/9"#);
  }
  #[test]
  fn pi_1() {
    assert_case(r#"1; Sqrt[2]; 2/9; Pi"#, r#"Pi"#);
  }
  #[test]
  fn sqrt_7() {
    assert_case(r#"1; 2/9; Sqrt[2]"#, r#"Sqrt[2]"#);
  }
  #[test]
  fn pi_2() {
    assert_case(r#"1; 2/9; Sqrt[2]; Pi"#, r#"Pi"#);
  }
  #[test]
  fn sqrt_8() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes"#,
      r#"SqrtBox["a"]"#,
    );
  }
  #[test]
  fn plus_1() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes; a + b * c//MakeBoxes"#,
      r#"RowBox[{"a", "+", RowBox[{"b", " ", "c"}]}]"#,
    );
  }
  #[test]
  fn plus_2() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes; a + b * c//MakeBoxes; a + b / c//MakeBoxes"#,
      r#"RowBox[{"a", "+", FractionBox["b", "c"]}]"#,
    );
  }
  #[test]
  fn plus_3() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes; a + b * c//MakeBoxes; a + b / c//MakeBoxes; a + b * c // InputForm//MakeBoxes"#,
      r#"InterpretationBox[StyleBox["a + b*c", ShowStringCharacters -> True, NumberMarks -> True], InputForm[a + b*c], Editable -> True, AutoDelete -> True]"#,
    );
  }
  #[test]
  fn plus_4() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes; a + b * c//MakeBoxes; a + b / c//MakeBoxes; a + b * c // InputForm//MakeBoxes; a + b / c //InputForm//MakeBoxes"#,
      r#"InterpretationBox[StyleBox["a + b/c", ShowStringCharacters -> True, NumberMarks -> True], InputForm[a + b/c], Editable -> True, AutoDelete -> True]"#,
    );
  }
  #[test]
  fn plus_5() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes; a + b * c//MakeBoxes; a + b / c//MakeBoxes; a + b * c // InputForm//MakeBoxes; a + b / c //InputForm//MakeBoxes; a + b * c // OutputForm//MakeBoxes"#,
      // wolframscript prints the rendered text as a quoted
      // String (one set of quotes baked in) and wraps the
      // expression-arg of InterpretationBox in OutputForm[…].
      r#"InterpretationBox[PaneBox["a + b c", BaselinePosition -> Baseline], OutputForm[a + b*c], Editable -> False]"#,
    );
  }
  #[test]
  fn plus_6() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes; a + b * c//MakeBoxes; a + b / c//MakeBoxes; a + b * c // InputForm//MakeBoxes; a + b / c //InputForm//MakeBoxes; a + b * c // OutputForm//MakeBoxes; a + b / c // OutputForm//MakeBoxes"#,
      // wolframscript prints this with `\n` escapes for the
      // newlines (InputForm string display); Woxi's top-level
      // output renders them as literal newlines but the
      // underlying String content is identical. Expected matches
      // wolframscript's logical content, modulo newline display.
      r#"InterpretationBox[PaneBox["    b
a + -
    c", BaselinePosition -> Baseline], OutputForm[a + b/c], Editable -> False]"#,
    );
  }
  #[test]
  fn plus_7() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes; a + b * c//MakeBoxes; a + b / c//MakeBoxes; a + b * c // InputForm//MakeBoxes; a + b / c //InputForm//MakeBoxes; a + b * c // OutputForm//MakeBoxes; a + b / c // OutputForm//MakeBoxes; a + b * c // FullForm//MakeBoxes"#,
      r#"TagBox[StyleBox[RowBox[{"Plus", "[", RowBox[{"a", ",", RowBox[{"Times", "[", RowBox[{"b", ",", "c"}], "]"}]}], "]"}], ShowSpecialCharacters -> False, ShowStringCharacters -> True, NumberMarks -> True], FullForm]"#,
    );
  }
  #[test]
  fn plus_8() {
    assert_case(
      r#""Hola"; "Hola
qué tal?"; a/b//MakeBoxes; Sqrt[a]//MakeBoxes; a + b * c//MakeBoxes; a + b / c//MakeBoxes; a + b * c // InputForm//MakeBoxes; a + b / c //InputForm//MakeBoxes; a + b * c // OutputForm//MakeBoxes; a + b / c // OutputForm//MakeBoxes; a + b * c // FullForm//MakeBoxes; a + b / c // FullForm//MakeBoxes"#,
      r#"TagBox[StyleBox[RowBox[{"Plus", "[", RowBox[{"a", ",", RowBox[{"Times", "[", RowBox[{"b", ",", RowBox[{"Power", "[", RowBox[{"c", ",", RowBox[{"-", "1"}]}], "]"}]}], "]"}]}], "]"}], ShowSpecialCharacters -> False, ShowStringCharacters -> True, NumberMarks -> True], FullForm]"#,
    );
  }
  #[test]
  fn round_13() {
    assert_case(r#"Round[a, b]"#, r#"Round[a, b]"#);
  }
  #[test]
  fn round_14() {
    assert_case(r#"Round[a, b]; Round[a, b]"#, r#"Round[a, b]"#);
  }
  #[test]
  fn minus_1() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q); a (- 2 a) ( 3 Sqrt[a] q)"#,
      r#"-6*a^(5/2)*q"#,
    );
  }
  #[test]
  fn plus_9() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q); a (- 2 a) ( 3 Sqrt[a] q); a (5+ a+ 2 b) (3 a q)"#,
      r#"3*a^2*(5 + a + 2*b)*q"#,
    );
  }
  #[test]
  fn plus_10() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q); a (- 2 a) ( 3 Sqrt[a] q); a (5+ a+ 2 b) (3 a q); a (- 2 (5+ a+ 2 b)) * (3 a q)"#,
      r#"-6*a^2*(5 + a + 2*b)*q"#,
    );
  }
  #[test]
  fn divide_2() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q); a (- 2 a) ( 3 Sqrt[a] q); a (5+ a+ 2 b) (3 a q); a (- 2 (5+ a+ 2 b)) * (3 a q); a  b a^2 / (2 a)^(3/2)"#,
      r#"(a^(3/2)*b)/(2*Sqrt[2])"#,
    );
  }
  #[test]
  fn divide_3() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q); a (- 2 a) ( 3 Sqrt[a] q); a (5+ a+ 2 b) (3 a q); a (- 2 (5+ a+ 2 b)) * (3 a q); a  b a^2 / (2 a)^(3/2); a  b a^2 / (a)^(3/2)"#,
      r#"a^(3/2)*b"#,
    );
  }
  #[test]
  fn divide_4() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q); a (- 2 a) ( 3 Sqrt[a] q); a (5+ a+ 2 b) (3 a q); a (- 2 (5+ a+ 2 b)) * (3 a q); a  b a^2 / (2 a)^(3/2); a  b a^2 / (a)^(3/2); a  b a^2 / (a b)^(3/2)"#,
      r#"(a^3*b)/(a*b)^(3/2)"#,
    );
  }
  #[test]
  fn minus_2() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q); a (- 2 a) ( 3 Sqrt[a] q); a (5+ a+ 2 b) (3 a q); a (- 2 (5+ a+ 2 b)) * (3 a q); a  b a^2 / (2 a)^(3/2); a  b a^2 / (a)^(3/2); a  b a^2 / (a b)^(3/2); a  b a ^ 2  (a b)^(-3 / 2)"#,
      r#"(a^3*b)/(a*b)^(3/2)"#,
    );
  }
  #[test]
  fn expression() {
    assert_case(
      r#"1.  2.  3.; 1 * 2/3 * 3/5; 1 (- 2/3) ( 3/5); 1. (- 2/3) ( 3 / 5); 1 (- 2/3) (2 I); 1. (- 2/3) (2 I); a ( 2 a) ( 3 a q); a (- 2 a) ( 3 Sqrt[a] q); a (5+ a+ 2 b) (3 a q); a (- 2 (5+ a+ 2 b)) * (3 a q); a  b a^2 / (2 a)^(3/2); a  b a^2 / (a)^(3/2); a  b a^2 / (a b)^(3/2); a  b a ^ 2  (a b)^(-3 / 2); a  b Infinity"#,
      r#"a*b*Infinity"#,
    );
  }
  #[test]
  fn cube_root_2() {
    assert_case(r#"CubeRoot[-5]"#, r#"-5 ^ (1 / 3)"#);
  }
  #[test]
  fn cube_root_3() {
    assert_case(r#"CubeRoot[-5]; CubeRoot[-510000]"#, r#"-10*510^(1/3)"#);
  }
  #[test]
  fn cube_root_4() {
    assert_case(
      r#"CubeRoot[-5]; CubeRoot[-510000]; CubeRoot[-5.1]"#,
      r#"-1.7213006207263157"#,
    );
  }
  #[test]
  fn cube_root_5() {
    assert_case(
      r#"CubeRoot[-5]; CubeRoot[-510000]; CubeRoot[-5.1]; CubeRoot[b]"#,
      r#"Surd[b, 3]"#,
    );
  }
  #[test]
  fn cube_root_6() {
    assert_case(
      r#"CubeRoot[-5]; CubeRoot[-510000]; CubeRoot[-5.1]; CubeRoot[b]; CubeRoot[-0.5]"#,
      r#"-0.7937005259840998"#,
    );
  }
  #[test]
  fn sqrt_9() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]"#,
      r#"Sqrt[2]"#,
    );
  }
  #[test]
  fn sqrt_10() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]"#,
      r#"I*Sqrt[2]"#,
    );
  }
  #[test]
  fn minus_3() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2)"#,
      r#"I*Sqrt[2]"#,
    );
  }
  #[test]
  fn divide_5() {
    assert_case(
      r#"I; 0; 1; Pi; a; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2)"#,
      r#"Sqrt[2]"#,
    );
  }
  #[test]
  fn minus_4() {
    assert_case(r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]"#, r#"0"#);
  }
  #[test]
  fn minus_5() {
    assert_case(r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi"#, r#"-Pi"#);
  }
  #[test]
  fn minus_6() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2"#,
      r#"1"#,
    );
  }
  #[test]
  fn minus_7() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3"#,
      r#"-1"#,
    );
  }
  #[test]
  fn sqrt_11() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]"#,
      r#"Sqrt[2]"#,
    );
  }
  #[test]
  fn sqrt_12() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]"#,
      r#"I*Sqrt[2]"#,
    );
  }
  #[test]
  fn minus_8() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2)"#,
      r#"I*Sqrt[2]"#,
    );
  }
  #[test]
  fn divide_6() {
    assert_case(
      r#"I; 0; 1; Pi; a; a-a; 3-3.; 2-Sqrt[4]; -Pi; (-1)^2; (-1)^3; Sqrt[2]; Sqrt[-2]; (-2)^(1/2); (2)^(1/2)"#,
      r#"Sqrt[2]"#,
    );
  }
}
