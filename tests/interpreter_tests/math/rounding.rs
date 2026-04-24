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
