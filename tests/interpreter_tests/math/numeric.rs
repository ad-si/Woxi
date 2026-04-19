use super::*;

mod n_arbitrary_precision {
  use super::*;

  #[test]
  fn n_default_precision_unchanged() {
    // N[expr] with 1 arg should still return f64 precision
    assert_eq!(interpret("N[Pi]").unwrap(), "3.141592653589793");
    assert_eq!(interpret("N[E]").unwrap(), "2.718281828459045");
    assert_eq!(interpret("N[Degree]").unwrap(), "0.017453292519943295");
  }

  #[test]
  fn n_high_precision_literal_truncates_to_machine() {
    // A literal Real with more digits than f64 can hold is stored as f64;
    // N[literal] echoes that machine-precision value (matches wolframscript).
    assert_eq!(
      interpret("N[1.01234567890123456789]").unwrap(),
      "1.0123456789012346"
    );
  }

  #[test]
  fn n_machine_precision_backtick_literal() {
    // Explicit `-precision marker on a high-digit literal still collapses
    // to machine precision (matches wolframscript).
    assert_eq!(
      interpret("N[1.01234567890123456789`]").unwrap(),
      "1.0123456789012346"
    );
  }

  #[test]
  fn n_of_two_ninths_at_machine_precision() {
    // N[2/9] at machine precision gives 16 significant digits.
    assert_eq!(interpret("N[2/9]").unwrap(), "0.2222222222222222");
  }

  #[test]
  fn n_pi_arbitrary_first_digits() {
    // N[Pi, 50] — check that first 50 significant digits are correct
    let result = interpret("N[Pi, 50]").unwrap();
    assert!(
      result
        .starts_with("3.14159265358979323846264338327950288419716939937510")
    );
    assert!(result.ends_with("`50."));
  }

  #[test]
  fn n_e_arbitrary_first_digits() {
    // N[E, 30] — check first 30 significant digits
    let result = interpret("N[E, 30]").unwrap();
    assert!(result.starts_with("2.7182818284590452353602874713"));
    assert!(result.ends_with("`30."));
  }

  #[test]
  fn n_integer_arbitrary() {
    // N[100, 20] should return 100.`20.
    assert_eq!(interpret("N[100, 20]").unwrap(), "100.`20.");
    // N[7, 20] should return 7.`20.
    assert_eq!(interpret("N[7, 20]").unwrap(), "7.`20.");
  }

  #[test]
  fn n_rational_arbitrary() {
    // N[1/3, 20] — should start with 0.3333...
    let result = interpret("N[1/3, 20]").unwrap();
    assert!(result.starts_with("0.3333333333333333333"));
    assert!(result.ends_with("`20."));
  }

  #[test]
  fn n_sqrt_arbitrary() {
    // N[Sqrt[2], 20] — check first 20 digits
    let result = interpret("N[Sqrt[2], 20]").unwrap();
    assert!(result.starts_with("1.414213562373095048801688724"));
    assert!(result.ends_with("`20."));
  }

  #[test]
  fn n_arbitrary_with_symbolic_parts() {
    // N[expr, prec] should evaluate numeric parts and leave symbols as-is
    // sqrt(a) is parsed as Times[sqrt, a], not Sqrt[a]
    let result = interpret("a = 4/433; N[sqrt(a), 10]").unwrap();
    assert!(result.contains("sqrt"));
    assert!(result.contains("`10."));
  }

  #[test]
  fn n_pi_10000_digits() {
    // N[Pi, 10000] — the main todo item
    let result = interpret("N[Pi, 10000]").unwrap();
    // Check the suffix
    assert!(result.ends_with("`10000."));
    // Check first 50 digits of Pi
    assert!(
      result
        .starts_with("3.14159265358979323846264338327950288419716939937510")
    );
    // Check digit count: should have > 10000 sig digits
    let digits_part = result.split('`').next().unwrap();
    let sig_digits: usize =
      digits_part.chars().filter(|c| c.is_ascii_digit()).count();
    assert!(sig_digits >= 10000);
  }

  #[test]
  fn n_pi_100_digits() {
    // N[Pi, 100] — check first 100 digits
    let result = interpret("N[Pi, 100]").unwrap();
    assert!(result.starts_with("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651"));
    assert!(result.ends_with("`100."));
  }

  #[test]
  fn n_head_is_real() {
    // Head[N[Pi, 50]] should be Real
    assert_eq!(interpret("Head[N[Pi, 50]]").unwrap(), "Real");
  }

  #[test]
  fn n_number_q() {
    // NumberQ[N[Pi, 50]] should be True
    assert_eq!(interpret("NumberQ[N[Pi, 50]]").unwrap(), "True");
  }

  #[test]
  fn n_exp_negative_integer() {
    // N[Exp[-2], 10] — uses E^(-2) with integer power
    let result = interpret("N[Exp[-2], 10]").unwrap();
    assert!(result.starts_with("0.13533528323661269"));
    assert!(result.ends_with("`10."));

    // N[Exp[-2], 20]
    let result = interpret("N[Exp[-2], 20]").unwrap();
    assert!(result.starts_with("0.13533528323661269189"));
    assert!(result.ends_with("`20."));

    // N[Exp[-1], 10]
    let result = interpret("N[Exp[-1], 10]").unwrap();
    assert!(result.starts_with("0.36787944117144232"));
    assert!(result.ends_with("`10."));

    // N[Exp[-3], 10]
    let result = interpret("N[Exp[-3], 10]").unwrap();
    assert!(result.starts_with("0.049787068367863942"));
    assert!(result.ends_with("`10."));
  }

  #[test]
  fn n_exp_positive_integer_power() {
    // N[E^2, 10] — integer power of E
    let result = interpret("N[E^2, 10]").unwrap();
    assert!(result.starts_with("7.38905609893065022"));
    assert!(result.ends_with("`10."));

    // N[Pi^2, 10] — integer power of Pi
    let result = interpret("N[Pi^2, 10]").unwrap();
    assert!(result.starts_with("9.86960440108935861"));
    assert!(result.ends_with("`10."));
  }

  #[test]
  fn n_exp_default_precision() {
    // N[Exp[-2]] — should return f64 precision
    assert_eq!(interpret("N[Exp[-2]]").unwrap(), "0.1353352832366127");
  }

  #[test]
  fn n_bigfloat_scientific_notation_large() {
    // N[Exp[1000], 10] should use *^ scientific notation (value >= 1e6)
    let result = interpret("N[Exp[1000], 10]").unwrap();
    assert!(
      result.starts_with("1.9700711140170469"),
      "Expected mantissa starting with 1.97..., got: {}",
      result
    );
    assert!(
      result.contains("`10.*^434"),
      "Expected scientific notation *^434, got: {}",
      result
    );
  }

  #[test]
  fn n_bigfloat_scientific_notation_small() {
    // N[Exp[-1000], 10] should use *^ scientific notation (value < 1e-5)
    let result = interpret("N[Exp[-1000], 10]").unwrap();
    assert!(
      result.starts_with("5.0759588975494"),
      "Expected mantissa starting with 5.07..., got: {}",
      result
    );
    assert!(
      result.contains("`10.*^-435"),
      "Expected scientific notation *^-435, got: {}",
      result
    );
  }

  #[test]
  fn n_bigfloat_scientific_notation_medium() {
    // N[Exp[100], 10] should use *^ (value ~ 2.69e43)
    let result = interpret("N[Exp[100], 10]").unwrap();
    assert!(
      result.starts_with("2.688117141816"),
      "Expected mantissa starting with 2.688..., got: {}",
      result
    );
    assert!(
      result.contains("`10.*^43"),
      "Expected scientific notation *^43, got: {}",
      result
    );
  }

  #[test]
  fn n_bigfloat_no_scientific_notation() {
    // N[Exp[10], 10] — value ~ 22026, should NOT use *^ (< 1e6)
    let result = interpret("N[Exp[10], 10]").unwrap();
    assert!(
      result.starts_with("22026.465794806"),
      "Expected normal notation, got: {}",
      result
    );
    assert!(
      result.ends_with("`10."),
      "Expected plain backtick notation, got: {}",
      result
    );
    assert!(
      !result.contains("*^"),
      "Should not use scientific notation, got: {}",
      result
    );
  }
}

mod real_precision {
  use super::*;

  #[test]
  fn full_precision_when_needed() {
    // Power[1.5, 2.5] needs full precision
    assert_eq!(interpret("Power[1.5, 2.5]").unwrap(), "2.7556759606310752");
  }

  #[test]
  fn short_precision_when_clean() {
    // Simple addition should round cleanly
    assert_eq!(interpret("1.5 + 2.7").unwrap(), "4.2");
  }
}

mod precision {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("Precision[1]").unwrap(), "Infinity");
  }

  #[test]
  fn integer_ten() {
    // Any integer (non-zero or zero) has infinite precision.
    assert_eq!(interpret("Precision[10]").unwrap(), "Infinity");
  }

  #[test]
  fn integer_zero() {
    // Precision[0] — exact integer zero has infinite precision.
    assert_eq!(interpret("Precision[0]").unwrap(), "Infinity");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("Precision[1/2]").unwrap(), "Infinity");
  }

  #[test]
  fn machine_real() {
    assert_eq!(interpret("Precision[0.5]").unwrap(), "MachinePrecision");
  }

  #[test]
  fn machine_real_zero_with_trailing_zeros() {
    // 0.00 still has MachinePrecision — the trailing zeros don't change the
    // underlying f64 representation.
    assert_eq!(interpret("Precision[0.00]").unwrap(), "MachinePrecision");
  }

  #[test]
  fn machine_real_zero() {
    // 0. — a plain machine-real zero has MachinePrecision.
    assert_eq!(interpret("Precision[0.]").unwrap(), "MachinePrecision");
  }

  #[test]
  fn machine_real_ten() {
    // 10. — a machine-real has MachinePrecision.
    assert_eq!(interpret("Precision[10.]").unwrap(), "MachinePrecision");
  }

  #[test]
  fn gaussian_integer() {
    // 2 + 3 I — exact Gaussian integer has infinite precision.
    assert_eq!(interpret("Precision[2 + 3 I]").unwrap(), "Infinity");
  }

  #[test]
  fn precision_of_real_part_machine_complex() {
    // Re[0.5 + 2.3 I] extracts the machine-real 0.5 with MachinePrecision.
    assert_eq!(
      interpret("Precision[Re[0.5+2.3 I]]").unwrap(),
      "MachinePrecision"
    );
  }

  #[test]
  fn precision_of_real_part_integer_plus_machine_imaginary() {
    // Re[1 + 2.3 I] is the machine-real 1. (integer real part picks up the
    // machine precision of the imaginary summand).
    assert_eq!(
      interpret("Precision[Re[1+2.3 I]]").unwrap(),
      "MachinePrecision"
    );
  }

  #[test]
  fn precision_of_imaginary_part_machine_complex() {
    // Im[0.5 + 2.3 I] extracts the machine-real 2.3 with MachinePrecision.
    assert_eq!(
      interpret("Precision[Im[0.5+2.3 I]]").unwrap(),
      "MachinePrecision"
    );
  }

  #[test]
  fn arbitrary_precision_literal() {
    // 1.23`10 is a literal with explicit precision of 10 digits.
    assert_eq!(interpret("Precision[1.23`10]").unwrap(), "10.");
  }

  #[test]
  fn arbitrary_precision_literal_4_digits() {
    // 3.1413`4 is a literal with explicit precision of 4 digits.
    assert_eq!(interpret("Precision[3.1413`4]").unwrap(), "4.");
  }

  #[test]
  fn dollar_machine_precision_value() {
    // $MachinePrecision is Log10[2^53] ≈ 15.9546 (matches wolframscript).
    assert_eq!(
      interpret("$MachinePrecision").unwrap(),
      "15.954589770191003"
    );
  }

  #[test]
  fn dollar_machine_epsilon_value() {
    // $MachineEpsilon is the smallest x where 1. + x != 1. on f64.
    assert_eq!(
      interpret("$MachineEpsilon").unwrap(),
      "2.220446049250313*^-16"
    );
  }

  #[test]
  fn system_word_length() {
    assert_eq!(interpret("$SystemWordLength").unwrap(), "64");
  }

  #[test]
  fn system_word_length_head_is_integer() {
    assert_eq!(
      interpret("Head[$SystemWordLength] == Integer").unwrap(),
      "True"
    );
  }

  #[test]
  fn session_id_head_is_integer() {
    assert_eq!(interpret("Head[$SessionID] == Integer").unwrap(), "True");
  }

  #[test]
  fn process_id_head_is_integer() {
    assert_eq!(interpret("Head[$ProcessID] == Integer").unwrap(), "True");
  }

  #[cfg(unix)]
  #[test]
  fn parent_process_id_head_is_integer() {
    assert_eq!(
      interpret("Head[$ParentProcessID] == Integer").unwrap(),
      "True"
    );
  }

  #[cfg(unix)]
  #[test]
  fn machine_name_head_is_string() {
    assert_eq!(interpret("Head[$MachineName] == String").unwrap(), "True");
  }

  #[test]
  fn user_name_head_is_string() {
    assert_eq!(interpret("Head[$UserName] == String").unwrap(), "True");
  }

  #[test]
  fn system_id_head_is_string() {
    assert_eq!(interpret("Head[$SystemID] == String").unwrap(), "True");
  }

  #[test]
  fn processor_type_head_is_string() {
    assert_eq!(interpret("Head[$ProcessorType] == String").unwrap(), "True");
  }

  #[cfg(any(target_os = "macos", target_os = "linux"))]
  #[test]
  fn system_memory_head_is_integer() {
    assert_eq!(interpret("Head[$SystemMemory] == Integer").unwrap(), "True");
  }

  #[cfg(any(target_os = "macos", target_os = "linux"))]
  #[test]
  fn system_memory_is_positive() {
    assert_eq!(interpret("$SystemMemory > 0").unwrap(), "True");
  }

  #[test]
  fn environment_returns_string_for_set_var() {
    assert_eq!(
      interpret("Head[Environment[\"PATH\"]] == String").unwrap(),
      "True"
    );
  }

  #[test]
  fn environment_returns_failed_for_unset_var() {
    assert_eq!(
      interpret("Environment[\"NONEXISTENT_VAR_XYZ_12345\"]").unwrap(),
      "$Failed"
    );
  }

  #[test]
  fn get_environment_string_returns_rule() {
    // Head of the result of GetEnvironment["PATH"] is Rule
    assert_eq!(
      interpret("Head[GetEnvironment[\"PATH\"]] == Rule").unwrap(),
      "True"
    );
  }

  #[test]
  fn get_environment_list_returns_list_of_rules() {
    // Length of GetEnvironment[{"PATH", "HOME"}] is 2
    assert_eq!(
      interpret("Length[GetEnvironment[{\"PATH\", \"HOME\"}]]").unwrap(),
      "2"
    );
  }

  #[test]
  fn version_number() {
    // $VersionNumber should return the git-describe output of the Woxi repo
    // (e.g. "v0.1.0-1234-gabcdef" or similar). Just check it's non-empty.
    let result = interpret("$VersionNumber").unwrap();
    assert!(!result.is_empty());
    assert_ne!(result, "$VersionNumber");
  }

  #[test]
  fn command_line_head_is_list() {
    // $CommandLine should return a list (of string args).
    assert_eq!(interpret("Head[$CommandLine] == List").unwrap(), "True");
  }

  #[test]
  fn script_command_line_head_is_list() {
    // $ScriptCommandLine should return a list (of string args).
    assert_eq!(
      interpret("Head[$ScriptCommandLine] == List").unwrap(),
      "True"
    );
  }
}

mod accuracy {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("Accuracy[1]").unwrap(), "Infinity");
  }

  #[test]
  fn gaussian_integer() {
    // 2 + 3 I is an exact Gaussian integer — infinite accuracy.
    assert_eq!(interpret("Accuracy[2 + 3 I]").unwrap(), "Infinity");
  }

  #[test]
  fn integer_ten() {
    // Any integer has infinite accuracy.
    assert_eq!(interpret("Accuracy[10]").unwrap(), "Infinity");
  }

  #[test]
  fn symbol() {
    assert_eq!(interpret("Accuracy[A]").unwrap(), "Infinity");
  }

  #[test]
  fn machine_real() {
    // Accuracy[0.5] ≈ 16.2556...
    let result = interpret("Accuracy[0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 16.2556).abs() < 0.01);
  }
}

mod exponent {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Exponent[(x^3 + 1)^2 + 1, x]").unwrap(), "6");
  }

  #[test]
  fn zero_expr() {
    assert_eq!(interpret("Exponent[0, x]").unwrap(), "-Infinity");
  }

  #[test]
  fn min_form() {
    assert_eq!(interpret("Exponent[(x^2 + 1)^3 - 1, x, Min]").unwrap(), "2");
  }
}

mod overflow_safety {
  use super::*;

  #[test]
  fn large_product_no_panic() {
    let result = interpret("IntegerLength[Times@@Range[5000]]");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "16326");
  }

  #[test]
  fn rationalize_no_panic() {
    let result = interpret("Rationalize[N[Pi], 0]");
    assert!(result.is_ok());
  }

  #[test]
  fn rationalize_zero_tolerance() {
    assert_eq!(
      interpret("Rationalize[N[Pi], 0]").unwrap(),
      "245850922/78256779"
    );
  }

  #[test]
  fn rationalize_symbolic_constant_passes_through() {
    assert_eq!(interpret("Rationalize[Pi]").unwrap(), "Pi");
  }

  #[test]
  fn rationalize_symbolic_constant_with_tolerance() {
    assert_eq!(interpret("Rationalize[Pi, 0.01]").unwrap(), "22/7");
  }

  #[test]
  fn rationalize_exact_rational_passes_through() {
    assert_eq!(interpret("Rationalize[1/3]").unwrap(), "1/3");
  }

  #[test]
  fn rationalize_integer_passes_through() {
    assert_eq!(interpret("Rationalize[5]").unwrap(), "5");
  }

  #[test]
  fn rationalize_undefined_symbol_passes_through() {
    assert_eq!(interpret("Rationalize[x]").unwrap(), "x");
  }

  #[test]
  fn n_erf_evaluates_numerically() {
    // N[Erf[1], 20] should produce a numeric result, not stay symbolic
    let result = interpret("N[Erf[1], 20]").unwrap();
    assert!(
      result.starts_with("0.8427007929497148693412"),
      "N[Erf[1],20] should start with correct first 22 digits: {}",
      result
    );
    assert!(
      result.ends_with("`20."),
      "Should have precision marker `20.: {}",
      result
    );
  }

  #[test]
  fn n_erf_zero() {
    let result = interpret("N[Erf[0], 20]").unwrap();
    assert!(
      result.starts_with("0"),
      "N[Erf[0],20] should be 0: {}",
      result
    );
  }

  #[test]
  fn n_erfc_evaluates_numerically() {
    // Erfc[1] = 1 - Erf[1] ≈ 0.1572992...
    let result = interpret("N[Erfc[1], 20]").unwrap();
    assert!(
      result.starts_with("0.1572992070502851306587"),
      "N[Erfc[1],20] should start correctly: {}",
      result
    );
  }

  #[test]
  fn n_exp_integral_ei_evaluates_numerically() {
    // ExpIntegralEi[1] ≈ 1.895117816355937...
    let result = interpret("N[ExpIntegralEi[1], 20]").unwrap();
    assert!(
      result.starts_with("1.89511781635593675546"),
      "N[ExpIntegralEi[1],20] should start correctly: {}",
      result
    );
  }
}
