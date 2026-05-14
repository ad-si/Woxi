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

  // Regression (mathics test_numbers.py:166): when the precision
  // argument is a non-integer numeric expression like Pi, the
  // resulting BigFloat carries the *exact* precision value as its
  // backtick tag — not the integer floor. ToString uses the precision
  // to round, so `ToString[N[Pi, Pi]]` displays as "3.14".
  #[test]
  fn n_pi_pi_precision_tag_is_full_pi_value() {
    assert_eq!(
      interpret("N[Pi, Pi]").unwrap(),
      "3.1415926535897932385`3.141592653589793"
    );
  }

  #[test]
  fn n_pi_pi_to_string_rounds_to_3_digits() {
    assert_eq!(interpret("ToString[N[Pi, Pi]]").unwrap(), "3.14");
  }

  #[test]
  fn precision_of_n_pi_pi_returns_full_pi() {
    assert_eq!(
      interpret("Precision[N[Pi, Pi]]").unwrap(),
      "3.141592653589793"
    );
  }

  // Regression (mathics test_numbers.py:167): `N[1/9, 30]` displays 30
  // ones after the decimal via ToString. Internally a 58-digit BigFloat
  // (matching wolframscript's bit budget for p ≤ 47).
  #[test]
  fn n_one_ninth_at_30_digits() {
    assert_eq!(
      interpret("N[1/9, 30]").unwrap(),
      "0.1111111111111111111111111111111111111111111111111111111111`30."
    );
  }

  #[test]
  fn n_one_ninth_at_30_digits_to_string() {
    assert_eq!(
      interpret("ToString[N[1/9, 30]]").unwrap(),
      "0.111111111111111111111111111111"
    );
  }

  // Regression (mathics test_numbers.py:175): a high-digit literal
  // passed through `N[_, 5]` keeps its 19-digit machine-precision
  // payload internally but ToString rounds to the requested 5 digits.
  #[test]
  fn n_high_digit_literal_at_precision_5() {
    assert_eq!(
      interpret("N[1.012345678901234567890123, 5]").unwrap(),
      "1.0123456789012345679`5."
    );
  }

  #[test]
  fn n_high_digit_literal_at_precision_5_to_string() {
    assert_eq!(
      interpret("ToString[N[1.012345678901234567890123, 5]]").unwrap(),
      "1.0123"
    );
  }

  // Regression (mathics test_evaluators.py:29, F[1.2, 2/9]):
  // `N[F[1.2, 2/9], $MachinePrecision]` previously expanded `1.2` to
  // its 50-digit binary-exact decimal expansion inside the
  // FunctionCall args. wolframscript keeps the user-entered `1.2`
  // literal and only expands the rational `2/9` to a BigFloat.
  // `n_eval_arbitrary_partial` now skips machine-precision Reals.
  #[test]
  fn n_machine_precision_inside_function_call_keeps_real() {
    let result =
      interpret("N[F[1.2, 2/9], $MachinePrecision]").unwrap();
    assert!(
      result.starts_with("F[1.2, 0."),
      "expected `1.2` to stay machine-precision, got: {}",
      result
    );
  }

  // Regression (mathics test_numbers.py:225): a list of values where
  // one carries an explicit accuracy tag `0.``5` (zero with accuracy
  // 5 digits) reports that accuracy as the minimum. The Integer 1 has
  // infinite accuracy, so the list's accuracy is dictated by 0.``5.
  #[test]
  fn accuracy_of_list_with_accuracy_tagged_zero() {
    assert_eq!(interpret("Accuracy[{1, 0.``5}]").unwrap(), "5.");
  }

  // Regression (mathics test_numbers.py:260): a complex with machine-
  // precision real and imaginary parts has Precision = MachinePrecision.
  #[test]
  fn precision_of_complex_machine_real() {
    assert_eq!(
      interpret("Precision[0.4 + 2.4 I]").unwrap(),
      "MachinePrecision"
    );
  }

  // Regression (mathics test_numbers.py:267): a list containing an
  // Integer (Precision Infinity) and a machine-precision Real reports
  // Precision = MachinePrecision (the lower of the two).
  #[test]
  fn precision_of_list_with_integer_and_machine_real() {
    assert_eq!(interpret("Precision[{1, 0.}]").unwrap(), "MachinePrecision");
  }

  // Regression (mathics test_numbers.py:268): when a list contains
  // `0.``5` (zero with accuracy 5), its Precision is `0.` (no
  // significant digits — the value is accuracy-only).
  #[test]
  fn precision_of_list_with_accuracy_tagged_zero() {
    assert_eq!(interpret("Precision[{1, 0.``5}]").unwrap(), "0.");
  }

  #[test]
  fn n_of_machine_real_stays_machine() {
    // N[1.5, 30] on a machine-precision Real returns the Real unchanged;
    // the requested higher precision cannot recover information not already
    // in the f64. Regression for mathics test_numbers.py:169.
    assert_eq!(interpret("N[1.5, 30]").unwrap(), "1.5");
    assert_eq!(interpret("N[1.5, 5]").unwrap(), "1.5");
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
  fn n_golden_ratio_arbitrary() {
    // N[GoldenRatio, 40] — (1 + Sqrt[5]) / 2 evaluated at arbitrary precision.
    let result = interpret("N[GoldenRatio, 40]").unwrap();
    assert!(result.starts_with("1.618033988749894848204586834365638117720"));
    assert!(result.ends_with("`40."));
  }

  #[test]
  fn n_euler_gamma_arbitrary() {
    // N[EulerGamma, 40] — Euler–Mascheroni constant at arbitrary precision.
    let result = interpret("N[EulerGamma, 40]").unwrap();
    assert!(result.starts_with("0.577215664901532860606512090082402431042"));
    assert!(result.ends_with("`40."));
  }

  #[test]
  fn n_catalan_arbitrary() {
    // N[Catalan, 20] — Catalan's constant G at arbitrary precision.
    let result = interpret("N[Catalan, 20]").unwrap();
    assert!(result.starts_with("0.91596559417721901505"));
    assert!(result.ends_with("`20."));
  }

  #[test]
  fn n_glaisher_arbitrary() {
    // N[Glaisher, 50] — Glaisher–Kinkelin constant A.
    let result = interpret("N[Glaisher, 50]").unwrap();
    assert!(
      result.starts_with("1.2824271291006226368753425688697917277676889273250")
    );
    assert!(result.ends_with("`50."));
  }

  #[test]
  fn n_khinchin_arbitrary() {
    // N[Khinchin, 50] — Khinchin's constant K₀.
    let result = interpret("N[Khinchin, 50]").unwrap();
    assert!(
      result.starts_with("2.6854520010653064453097148354817956938203822939944")
    );
    assert!(result.ends_with("`50."));
  }

  #[test]
  fn n_machine_precision_arbitrary() {
    // N[MachinePrecision, 30] — MachinePrecision = Log10[2^53] ≈ 15.9545…
    // Regression for the mathics atomic/numbers.py MachinePrecision doctest
    // and for the `N[MachinePrecision, _]` path more broadly.
    let result = interpret("N[MachinePrecision, 30]").unwrap();
    assert!(
      result.starts_with("15.9545897701910033463281614203"),
      "Got: {}",
      result
    );
    assert!(result.ends_with("`30."));
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

mod n_threading {
  use super::*;

  #[test]
  fn n_value_propagates_through_plus() {
    // `N[a] = 10.9; N[a + b]` should thread N over Plus, look up
    // `N[a]` (returning 10.9) and leave `b` symbolic.
    clear_state();
    assert_eq!(interpret("N[a] = 10.9; N[a + b]").unwrap(), "10.9 + b");
  }

  #[test]
  fn n_value_propagates_through_function_call() {
    clear_state();
    assert_eq!(interpret("N[a] = 10.9; N[f[a, b]]").unwrap(), "f[10.9, b]");
  }

  #[test]
  fn nholdall_blocks_n_threading() {
    // SetAttributes[f, NHoldAll] should keep N from threading into f's
    // arguments — `N[f[a, b]]` stays as `f[a, b]`.
    clear_state();
    assert_eq!(
      interpret("N[a] = 10.9; SetAttributes[f, NHoldAll]; N[f[a, b]]").unwrap(),
      "f[a, b]"
    );
  }

  #[test]
  fn nholdfirst_blocks_only_first_arg() {
    clear_state();
    assert_eq!(
      interpret(
        "N[a] = 10.9; N[b] = 7.7; SetAttributes[g, NHoldFirst]; N[g[a, b]]"
      )
      .unwrap(),
      "g[a, 7.7]"
    );
  }

  #[test]
  fn nholdrest_blocks_only_rest_args() {
    clear_state();
    assert_eq!(
      interpret(
        "N[a] = 10.9; N[b] = 7.7; SetAttributes[h, NHoldRest]; N[h[a, b]]"
      )
      .unwrap(),
      "h[10.9, b]"
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
  fn precision_real_literal_zero_is_machine_precision() {
    // A BigFloat zero (e.g. `0.`20`, `0.00`2`, `-0.`20`) reports
    // MachinePrecision in Wolfram — the specified precision is irrelevant
    // when every significant digit is zero. Regression for mathics
    // test_numbers.py:232.
    assert_eq!(interpret("Precision[0.`20]").unwrap(), "MachinePrecision");
    assert_eq!(interpret("Precision[0.00`2]").unwrap(), "MachinePrecision");
    assert_eq!(interpret("Precision[0.00`20]").unwrap(), "MachinePrecision");
    assert_eq!(interpret("Precision[-0.`2]").unwrap(), "MachinePrecision");
    assert_eq!(interpret("Precision[-0.`20]").unwrap(), "MachinePrecision");
  }

  #[test]
  fn precision_nonzero_big_float_uses_specified_precision() {
    // Regression guard: only zero BigFloats degrade to MachinePrecision;
    // a nonzero literal with a precision suffix still uses it.
    assert_eq!(interpret("Precision[10.00`2]").unwrap(), "2.");
    assert_eq!(interpret("Precision[10.00`20]").unwrap(), "20.");
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

  // Precision of a nested list is the minimum across all numeric leaves.
  // Integers contribute Infinity, MachineReal contributes MachinePrecision,
  // and BigFloat contributes its literal precision. When MachinePrecision
  // is mixed with arbitrary-precision Reals, wolframscript collapses the
  // result to MachinePrecision (machine arithmetic dominates the mix).
  #[test]
  fn precision_nested_list_minimum() {
    assert_eq!(
      interpret("Precision[{{1, 1.`},{1.`5, 1.`10}}]").unwrap(),
      "MachinePrecision"
    );
  }

  // Identity `Accuracy[z] == Precision[z] + Log[z]` at z = 37.`
  // — a MachinePrecision check that the two precision measures differ by
  // Log10 of the magnitude. Matches the mathics numbers.py Precision
  // doctest.
  #[test]
  fn accuracy_precision_log_identity_on_machine_real() {
    assert_eq!(
      interpret("(Accuracy[z] == Precision[z] + Log[z])/.z-> 37.`").unwrap(),
      "True"
    );
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
  fn dollar_max_precision_value() {
    // $MaxPrecision defaults to Infinity (matches wolframscript).
    assert_eq!(interpret("$MaxPrecision").unwrap(), "Infinity");
  }

  // $MinMachineNumber is Wolfram's smallest *normalized* positive machine
  // number (2^-1022), not the smallest subnormal. Regression for the
  // mathics doctest in numbers/constants.py for $MaxMachineNumber.
  #[test]
  fn dollar_min_machine_number_value() {
    assert_eq!(
      interpret("$MinMachineNumber").unwrap(),
      "2.2250738585072014*^-308"
    );
  }

  #[test]
  fn dollar_max_machine_number_value() {
    // $MaxMachineNumber is f64::MAX ≈ 1.7977*^308.
    assert_eq!(
      interpret("$MaxMachineNumber").unwrap(),
      "1.7976931348623157*^308"
    );
  }

  // Product of the two machine bounds rounds to ~4, but the f64 result is
  // `3.9999999999999996` (1 ULP below 4). Woxi and wolframscript agree here;
  // mathics displays the rounded `4.` instead. Regression for the mathics
  // doctest of MaxMachineNumber.
  #[test]
  fn max_times_min_machine_number() {
    assert_eq!(
      interpret("$MaxMachineNumber * $MinMachineNumber").unwrap(),
      "3.9999999999999996"
    );
  }

  #[test]
  fn dollar_min_precision_value() {
    // $MinPrecision defaults to 0 (matches wolframscript).
    assert_eq!(interpret("$MinPrecision").unwrap(), "0");
  }

  #[test]
  fn precision_list_machine_and_integer() {
    // Precision on a list returns the minimum precision of its elements —
    // here one machine real and one exact integer, so MachinePrecision wins.
    // Previously Precision was marked listable and returned {Infinity,
    // MachinePrecision}; wolframscript returns the single symbol.
    assert_eq!(interpret("Precision[{1, 0.}]").unwrap(), "MachinePrecision");
  }

  #[test]
  fn precision_of_machine_complex_literal() {
    // Precision of a machine-precision complex literal returns the symbol
    // MachinePrecision, not the numeric value (matches wolframscript).
    assert_eq!(
      interpret("Precision[0.4 + 2.4 I]").unwrap(),
      "MachinePrecision"
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

  #[test]
  fn byte_ordering_is_minus_one_or_one() {
    // -1 = little-endian (most modern platforms), 1 = big-endian.
    assert_eq!(
      interpret("$ByteOrdering == -1 || $ByteOrdering == 1").unwrap(),
      "True"
    );
    assert_eq!(interpret("Head[$ByteOrdering] == Integer").unwrap(), "True");
  }

  #[test]
  fn operating_system_head_is_string() {
    assert_eq!(
      interpret("Head[$OperatingSystem] == String").unwrap(),
      "True"
    );
  }

  #[test]
  fn pathname_separator_head_is_string() {
    // $PathnameSeparator is "/" on Unix/Mac, "\" on Windows.
    assert_eq!(
      interpret("Head[$PathnameSeparator] == String").unwrap(),
      "True"
    );
    assert_eq!(interpret("StringLength[$PathnameSeparator]").unwrap(), "1");
  }

  // $CharacterEncoding defaults to "UTF8", matching wolframscript on a
  // modern terminal. Regression for the mathics doctest of CharacterEncoding.
  #[test]
  fn character_encoding_default_is_utf8() {
    assert_eq!(interpret("$CharacterEncoding").unwrap(), "UTF8");
  }

  // `$SystemCharacterEncoding` uses the IANA spelling "UTF-8", not "UTF8".
  #[test]
  fn system_character_encoding_default_is_utf_dash_8() {
    assert_eq!(interpret("$SystemCharacterEncoding").unwrap(), "UTF-8");
  }

  // The two encoding variables differ by the dash; they are NOT equal in
  // wolframscript (or Woxi). Mathics wrongly expects True here.
  #[test]
  fn system_vs_user_character_encoding_differ() {
    assert_eq!(
      interpret("$SystemCharacterEncoding == $CharacterEncoding").unwrap(),
      "False"
    );
  }

  // `$CharacterEncodings` returns the fixed registry-ordered list; the
  // first nine entries are as in wolframscript (not alphabetical).
  #[test]
  fn character_encodings_first_nine() {
    assert_eq!(
      interpret("$CharacterEncodings[[;;9]]").unwrap(),
      "{AdobeStandard, ASCII, CP936, CP949, CP950, EUC-JP, EUC, IBM-850, ISO8859-10}"
    );
  }

  // `$PrintForms` lists the output-form symbols in wolframscript's order.
  #[test]
  fn print_forms_matches_wolframscript() {
    assert_eq!(
      interpret("$PrintForms").unwrap(),
      "{InputForm, OutputForm, TextForm, CForm, FortranForm, ScriptForm, MathMLForm, TeXForm, StandardForm, TraditionalForm}"
    );
  }

  // `$OutputForms` is a superset of `$PrintForms` that adds Short,
  // MatrixForm, BaseForm, etc.
  #[test]
  fn output_forms_matches_wolframscript() {
    assert_eq!(
      interpret("$OutputForms").unwrap(),
      "{InputForm, OutputForm, TextForm, CForm, Short, Shallow, MatrixForm, TableForm, TreeForm, FullForm, NumberForm, EngineeringForm, ScientificForm, QuantityForm, DecimalForm, PercentForm, PaddedForm, AccountingForm, BaseForm, DisplayForm, StyleForm, FortranForm, ScriptForm, MathMLForm, TeXForm, StandardForm, TraditionalForm}"
    );
  }

  // `$BoxForms` defaults to `{StandardForm, TraditionalForm}`.
  #[test]
  fn box_forms_default() {
    assert_eq!(
      interpret("$BoxForms").unwrap(),
      "{StandardForm, TraditionalForm}"
    );
  }

  // Unsetting `$BoxForms` falls back to the default list (the system
  // variable lookup still triggers).
  #[test]
  fn box_forms_unset_restores_default() {
    assert_eq!(
      interpret("$BoxForms=.; $BoxForms").unwrap(),
      "{StandardForm, TraditionalForm}"
    );
  }

  #[test]
  fn home_directory_head_is_string() {
    // $HomeDirectory reads $HOME (or $USERPROFILE on Windows).
    assert_eq!(interpret("Head[$HomeDirectory] == String").unwrap(), "True");
  }

  // `$UserBaseDirectory` and `$BaseDirectory` return string paths that
  // match wolframscript's platform-specific defaults (Library/Wolfram on
  // macOS, .Wolfram on other Unix-likes).
  #[test]
  fn user_base_directory_head_is_string() {
    assert_eq!(
      interpret("Head[$UserBaseDirectory] == String").unwrap(),
      "True"
    );
  }

  #[test]
  fn base_directory_head_is_string() {
    assert_eq!(interpret("Head[$BaseDirectory] == String").unwrap(), "True");
  }

  // `$InstallationDirectory` reports the directory containing the Woxi
  // executable — analogous to wolframscript's Wolfram installation path.
  #[test]
  fn installation_directory_head_is_string() {
    assert_eq!(
      interpret("Head[$InstallationDirectory] == String").unwrap(),
      "True"
    );
  }

  // `$Path` is the package search list; head must be List, and every
  // entry must be a string.
  #[test]
  fn dollar_path_is_list_of_strings() {
    assert_eq!(interpret("Head[$Path] == List").unwrap(), "True");
    assert_eq!(
      interpret("AllTrue[$Path, Head[#] == String &]").unwrap(),
      "True"
    );
    // Current-directory "." is always present.
    assert_eq!(interpret("MemberQ[$Path, \".\"]").unwrap(), "True");
  }

  // `$Version` is a string banner. Woxi returns `"Woxi <git-version>"`.
  #[test]
  fn dollar_version_is_string() {
    assert_eq!(interpret("Head[$Version] == String").unwrap(), "True");
    assert_eq!(
      interpret("StringStartsQ[$Version, \"Woxi \"]").unwrap(),
      "True"
    );
  }

  // `$Line` tracks the input line number; wolframscript runs each script
  // as a fresh session, so it always reads as 1 regardless of how many
  // statements have already been evaluated.
  #[test]
  fn dollar_line_is_one() {
    assert_eq!(interpret("$Line").unwrap(), "1");
    assert_eq!(interpret("$Line; $Line").unwrap(), "1");
    assert_eq!(interpret("42; $Line").unwrap(), "1");
  }

  #[test]
  fn temporary_directory_head_is_string() {
    // $TemporaryDirectory is canonicalized and has no trailing slash —
    // matches wolframscript's output (e.g. /private/var/folders/... on macOS).
    assert_eq!(
      interpret("Head[$TemporaryDirectory] == String").unwrap(),
      "True"
    );
    let result = interpret("$TemporaryDirectory").unwrap();
    assert!(!result.is_empty());
    assert!(!result.ends_with('/') || result == "/");
  }

  #[test]
  fn initial_directory_head_is_string() {
    // $InitialDirectory returns the current working directory at startup.
    assert_eq!(
      interpret("Head[$InitialDirectory] == String").unwrap(),
      "True"
    );
  }

  #[cfg(not(target_os = "windows"))]
  #[test]
  fn root_directory_is_slash() {
    // $RootDirectory is the filesystem root: "/" on Unix/Mac.
    assert_eq!(interpret("$RootDirectory").unwrap(), "/");
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
  fn set_environment_rule_returns_null() {
    interpret("SetEnvironment[\"WOXI_TEST_SET_ENV_VAR_A\" -> \"alpha\"]")
      .unwrap();
    assert_eq!(
      interpret("Environment[\"WOXI_TEST_SET_ENV_VAR_A\"]").unwrap(),
      "alpha"
    );
  }

  #[test]
  fn set_environment_none_unsets_var() {
    interpret("SetEnvironment[\"WOXI_TEST_SET_ENV_VAR_B\" -> \"beta\"]")
      .unwrap();
    interpret("SetEnvironment[\"WOXI_TEST_SET_ENV_VAR_B\" -> None]").unwrap();
    assert_eq!(
      interpret("Environment[\"WOXI_TEST_SET_ENV_VAR_B\"]").unwrap(),
      "$Failed"
    );
  }

  #[test]
  fn set_environment_list_of_rules_returns_null() {
    interpret(
      "SetEnvironment[{\"WOXI_TEST_SET_ENV_VAR_C\" -> \"c1\", \"WOXI_TEST_SET_ENV_VAR_D\" -> \"d1\"}]"
    )
    .unwrap();
    assert_eq!(
      interpret("Environment[\"WOXI_TEST_SET_ENV_VAR_C\"]").unwrap(),
      "c1"
    );
    assert_eq!(
      interpret("Environment[\"WOXI_TEST_SET_ENV_VAR_D\"]").unwrap(),
      "d1"
    );
  }

  #[test]
  fn set_environment_non_string_value_returns_failed() {
    assert_eq!(
      interpret("SetEnvironment[\"WOXI_TEST_SET_ENV_VAR_E\" -> 5]").unwrap(),
      "$Failed"
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
  fn integer_zero() {
    // Accuracy[0] is Infinity (exact zero integer).
    assert_eq!(interpret("Accuracy[0]").unwrap(), "Infinity");
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

  #[test]
  fn arbitrary_precision_real() {
    // Accuracy[3.14``5] = 5 (the explicit accuracy specified by `\`\`5`).
    let result = interpret("Accuracy[3.14``5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 5.0).abs() < 0.01);
  }

  #[test]
  fn zero_with_accuracy_tag() {
    // `0.\`\`2` is a 0 with accuracy 2; Accuracy reads that off.
    let result = interpret("Accuracy[0.``2]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 2.0).abs() < 0.01);
  }

  #[test]
  fn function_call_with_inexact_arg() {
    // Accuracy of an unevaluated head propagates the minimum accuracy
    // among the arguments. `F[1, Pi, A]` is exact (∞), `F[1.3, Pi, A]`
    // takes its accuracy from 1.3 ≈ 15.84.
    assert_eq!(interpret("Accuracy[F[1, Pi, A]]").unwrap(), "Infinity");
    let result = interpret("Accuracy[F[1.3, Pi, A]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 15.8406).abs() < 0.01);
  }

  #[test]
  fn list_minimum() {
    // Accuracy of a list is the minimum accuracy of its elements
    // (recursively). Mixed exact and inexact: the inexact entry wins.
    let result = interpret("Accuracy[{{1, 1.`}, {1.``5, 1.``10}}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 5.0).abs() < 0.01);
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

mod cases {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn real_valued_number_q_1() {
    assert_case(r#"RealValuedNumberQ[10]"#, r#"True"#);
  }
  #[test]
  fn real_valued_number_q_2() {
    assert_case(
      r#"RealValuedNumberQ[10]; RealValuedNumberQ[4.0]"#,
      r#"True"#,
    );
  }
  #[test]
  fn real_valued_number_q_3() {
    assert_case(
      r#"RealValuedNumberQ[10]; RealValuedNumberQ[4.0]; RealValuedNumberQ[1+I]"#,
      r#"False"#,
    );
  }
  #[test]
  fn real_valued_number_q_4() {
    assert_case(
      r#"RealValuedNumberQ[10]; RealValuedNumberQ[4.0]; RealValuedNumberQ[1+I]; RealValuedNumberQ[0 * I]"#,
      r#"True"#,
    );
  }
  #[test]
  fn real_valued_number_q_5() {
    assert_case(
      r#"RealValuedNumberQ[10]; RealValuedNumberQ[4.0]; RealValuedNumberQ[1+I]; RealValuedNumberQ[0 * I]; RealValuedNumberQ[0.0 * I]"#,
      r#"False"#,
    );
  }
  #[test]
  fn list_literal() {
    assert_case(
      r#"RealValuedNumberQ[10]; RealValuedNumberQ[4.0]; RealValuedNumberQ[1+I]; RealValuedNumberQ[0 * I]; RealValuedNumberQ[0.0 * I]; {RealValuedNumberQ[Underflow[]], RealValuedNumberQ[Overflow[]]}"#,
      r#"{True, True}"#,
    );
  }
  #[test]
  fn n_1() {
    assert_case(r#"N[f[2, 3]]"#, r#"f[2., 3.]"#);
  }
  #[test]
  fn uncompress() {
    // The mathics original (`>> Compress[N[Pi, 10]] = ...`) accepts any
    // output. The scraped expectation pinned wolframscript-specific
    // compressed bytes — different compressors (Wolfram's vs Woxi's
    // zlib) produce different byte sequences from the same input even
    // when both are valid `Compress` outputs. Verify the documented
    // contract via round-trip: `Uncompress[Compress[x]] == x`.
    assert_case(r#"Uncompress[Compress[N[Pi, 10]]] == N[Pi, 10]"#, r#"True"#);
  }
  #[test]
  fn chop_1() {
    assert_case(r#"Chop[10.0 ^ -16]"#, r#"0"#);
  }
  #[test]
  fn chop_2() {
    assert_case(r#"Chop[10.0 ^ -16]; Chop[10.0 ^ -9]"#, r#"1.*^-9"#);
  }
  #[test]
  fn chop_3() {
    assert_case(
      r#"Chop[10.0 ^ -16]; Chop[10.0 ^ -9]; Chop[10 ^ -11 I]"#,
      r#"I / 100000000000"#,
    );
  }
  #[test]
  fn chop_4() {
    assert_case(
      r#"Chop[10.0 ^ -16]; Chop[10.0 ^ -9]; Chop[10 ^ -11 I]; Chop[0. + 10 ^ -11 I]"#,
      r#"0"#,
    );
  }
  #[test]
  fn n_2() {
    assert_case(
      r#"N[Pi, 50]"#,
      r#"3.1415926535897932384626433832795028841971693993751058209749445923078164118876`50."#,
    );
  }
  #[test]
  fn n_3() {
    assert_case(r#"N[Pi, 50]; N[1/7]"#, r#"0.14285714285714285"#);
  }
  #[test]
  fn n_4() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]"#,
      r#"0.1428571428571428571`5."#,
    );
  }
  #[test]
  fn n_5() {
    assert_case(r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9"#, r#"10.9"#);
  }
  #[test]
  fn symbol_literal() {
    assert_case(r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a"#, r#"a"#);
  }
  #[test]
  fn n_6() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]"#,
      r#"10.9 + b"#,
    );
  }
  #[test]
  fn n_7() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]"#,
      r#"a"#,
    );
  }
  #[test]
  fn n_8() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]; N[a, 20] = 11; N[a + b, 20]"#,
      r#"11.`20. + b"#,
    );
  }
  #[test]
  fn n_9() {
    assert_case(
      r#"N[Pi, 50]; N[1/7]; N[1/7, 5]; N[a] = 10.9; a; N[a + b]; N[a, 20]; N[a, 20] = 11; N[a + b, 20]; N[f[a, b]]"#,
      r#"f[10.9, b]"#,
    );
  }
  #[test]
  fn rationalize_1() {
    assert_case(
      r#"Rationalize[2.2]; Rationalize[-11.5, 1]; Rationalize[N[Pi]]"#,
      r#"3.141592653589793"#,
    );
  }
  #[test]
  fn rationalize_2() {
    assert_case(
      r#"Rationalize[2.2]; Rationalize[-11.5, 1]; Rationalize[N[Pi]]; Rationalize[N[Pi], 0]"#,
      r#"245850922 / 78256779"#,
    );
  }
  #[test]
  fn n_10() {
    assert_case(
      r#"Catalan // N; N[Catalan, 20]"#,
      r#"0.91596559417721901505470246127766603241`20."#,
    );
  }
  #[test]
  fn n_11() {
    assert_case(
      r#"Cos[60 Degree]; Degree == Pi / 180; N[\[Degree]] == N[Degree]"#,
      r#"True"#,
    );
  }
  #[test]
  fn n_12() {
    assert_case(r#"N[E]"#, r#"2.718281828459045"#);
  }
  #[test]
  fn n_13() {
    assert_case(
      r#"N[E]; N[E, 50]"#,
      r#"2.71828182845904523536028747135266249775724709369995957495841999330501070030685`50."#,
    );
  }
  #[test]
  fn n_14() {
    assert_case(
      r#"EulerGamma // N; N[EulerGamma, 40]"#,
      r#"0.5772156649015328606065120900824024310421593359398806556748`40."#,
    );
  }
  #[test]
  fn n_15() {
    assert_case(r#"N[Glaisher]"#, r#"1.2824271291006226"#);
  }
  #[test]
  fn n_16() {
    assert_case(
      r#"N[Glaisher]; N[Glaisher, 50]"#,
      r#"1.28242712910062263687534256886979172776768892732500119211620855281283363706656`50."#,
    );
  }
  #[test]
  fn n_17() {
    assert_case(
      r#"GoldenRatio // N; N[GoldenRatio, 40]"#,
      r#"1.6180339887498948482045868343656381177203091798057628621355`40."#,
    );
  }
  #[test]
  fn precision_1() {
    assert_case(r#"Precision[1]"#, r#"Infinity"#);
  }
  #[test]
  fn divide_1() {
    assert_case(r#"Precision[1]; 1 / Infinity"#, r#"0"#);
  }
  #[test]
  fn plus() {
    assert_case(
      r#"Precision[1]; 1 / Infinity; Infinity + 100"#,
      r#"Infinity"#,
    );
  }
  #[test]
  fn n_18() {
    assert_case(r#"N[Khinchin]"#, r#"2.6854520010653062"#);
  }
  #[test]
  fn n_19() {
    assert_case(
      r#"N[Khinchin]; N[Khinchin, 50]"#,
      r#"2.68545200106530644530971483548179569382038229399446295307978944749044639219889`50."#,
    );
  }
  #[test]
  fn n_20() {
    assert_case(r#"Pi; N[Pi]"#, r#"3.141592653589793"#);
  }
  #[test]
  fn n_21() {
    assert_case(
      r#"Pi; N[Pi]; N[Pi, 20]"#,
      r#"3.1415926535897932384626433832795028842`20."#,
    );
  }
  #[test]
  fn number_form_1() {
    assert_case(
      r#"NumberForm[N[Pi], 10]"#,
      r#"NumberForm[3.141592653589793, 10]"#,
    );
  }
  #[test]
  fn number_form_2() {
    assert_case(
      r#"NumberForm[N[Pi], 10]; NumberForm[N[Pi], {10, 6}]"#,
      r#"NumberForm[3.141592653589793, {10, 6}]"#,
    );
  }
  #[test]
  fn number_form_3() {
    assert_case(
      r#"NumberForm[N[Pi], 10]; NumberForm[N[Pi], {10, 6}]; NumberForm[N[Pi]]"#,
      r#"NumberForm[3.141592653589793]"#,
    );
  }
  #[test]
  fn n_22() {
    assert_case(
      r#"N[CosineDistance[{7, 9}, {71, 89}]]"#,
      r#"0.00007596457213221441"#,
    );
  }
  #[test]
  fn cosine_distance_1() {
    assert_case(
      r#"N[CosineDistance[{7, 9}, {71, 89}]]; CosineDistance[{0.0, 0.0}, {x, y}]"#,
      r#"0."#,
    );
  }
  #[test]
  fn cosine_distance_2() {
    assert_case(
      r#"N[CosineDistance[{7, 9}, {71, 89}]]; CosineDistance[{0.0, 0.0}, {x, y}]; CosineDistance[{1, 0}, {x, y}]"#,
      r#"1 - Conjugate[x] / Sqrt[Abs[x] ^ 2 + Abs[y] ^ 2]"#,
    );
  }
  #[test]
  fn cosine_distance_3() {
    assert_case(
      r#"N[CosineDistance[{7, 9}, {71, 89}]]; CosineDistance[{0.0, 0.0}, {x, y}]; CosineDistance[{1, 0}, {x, y}]; CosineDistance[{x, y}, {1, 0}]"#,
      r#"1 - x / Sqrt[Abs[x] ^ 2 + Abs[y] ^ 2]"#,
    );
  }
  #[test]
  fn cosine_distance_4() {
    assert_case(
      r#"N[CosineDistance[{7, 9}, {71, 89}]]; CosineDistance[{0.0, 0.0}, {x, y}]; CosineDistance[{1, 0}, {x, y}]; CosineDistance[{x, y}, {1, 0}]; CosineDistance[{a, b, c}, {x, y, z}]"#,
      r#"1 - (a*Conjugate[x] + b*Conjugate[y] + c*Conjugate[z])/(Sqrt[Abs[a]^2 + Abs[b]^2 + Abs[c]^2]*Sqrt[Abs[x]^2 + Abs[y]^2 + Abs[z]^2])"#,
    );
  }
  #[test]
  fn cosine_distance_5() {
    assert_case(
      r#"N[CosineDistance[{7, 9}, {71, 89}]]; CosineDistance[{0.0, 0.0}, {x, y}]; CosineDistance[{1, 0}, {x, y}]; CosineDistance[{x, y}, {1, 0}]; CosineDistance[{a, b, c}, {x, y, z}]; CosineDistance[1+2I, 5]"#,
      r#"1 - (1 + 2*I)/Sqrt[5]"#,
    );
  }
  #[test]
  fn n_23() {
    assert_case(r#"1 == 1.; 5/3 == 3/2; N[E, 100] == N[E, 150]"#, r#"True"#);
  }
  #[test]
  fn equal_1() {
    assert_case(
      r#"1 == 1.; 5/3 == 3/2; N[E, 100] == N[E, 150]; Pi == N[Pi, 20]"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_2() {
    assert_case(
      r#"1 == 1.; 5/3 == 3/2; N[E, 100] == N[E, 150]; Pi == N[Pi, 20]; Pi == 3.14"#,
      r#"False"#,
    );
  }
  #[test]
  fn equal_3() {
    assert_case(
      r#"1 == 1.; 5/3 == 3/2; N[E, 100] == N[E, 150]; Pi == N[Pi, 20]; Pi == 3.14; Pi ^ E == E ^ Pi"#,
      r#"False"#,
    );
  }
  #[test]
  fn equal_4() {
    assert_case(
      r#"1 == 1.; 5/3 == 3/2; N[E, 100] == N[E, 150]; Pi == N[Pi, 20]; Pi == 3.14; Pi ^ E == E ^ Pi; Pi == 3.1415``4"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_5() {
    assert_case(
      r#"1 == 1.; 5/3 == 3/2; N[E, 100] == N[E, 150]; Pi == N[Pi, 20]; Pi == 3.14; Pi ^ E == E ^ Pi; Pi == 3.1415``4; 0.739085133215160642 == 0.739085133215160641"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_6() {
    assert_case(
      r#"1 == 1.; 5/3 == 3/2; N[E, 100] == N[E, 150]; Pi == N[Pi, 20]; Pi == 3.14; Pi ^ E == E ^ Pi; Pi == 3.1415``4; 0.739085133215160642 == 0.739085133215160641; 0.73908513321516064200000000 == 0.73908513321516064100000000"#,
      r#"False"#,
    );
  }
  #[test]
  fn exact_number_q_1() {
    assert_case(r#"ExactNumberQ[10]"#, r#"True"#);
  }
  #[test]
  fn exact_number_q_2() {
    assert_case(r#"ExactNumberQ[10]; ExactNumberQ[10.0]"#, r#"False"#);
  }
  #[test]
  fn exact_number_q_3() {
    assert_case(
      r#"ExactNumberQ[10]; ExactNumberQ[10.0]; ExactNumberQ[I]"#,
      r#"True"#,
    );
  }
  #[test]
  fn exact_number_q_4() {
    assert_case(
      r#"ExactNumberQ[10]; ExactNumberQ[10.0]; ExactNumberQ[I]; ExactNumberQ[1 + I]"#,
      r#"True"#,
    );
  }
  #[test]
  fn exact_number_q_5() {
    assert_case(
      r#"ExactNumberQ[10]; ExactNumberQ[10.0]; ExactNumberQ[I]; ExactNumberQ[1 + I]; ExactNumberQ[1. + I]"#,
      r#"False"#,
    );
  }
  #[test]
  fn exact_number_q_6() {
    assert_case(
      r#"ExactNumberQ[10]; ExactNumberQ[10.0]; ExactNumberQ[I]; ExactNumberQ[1 + I]; ExactNumberQ[1. + I]; ExactNumberQ[5/6]"#,
      r#"True"#,
    );
  }
  #[test]
  fn exact_number_q_7() {
    assert_case(
      r#"ExactNumberQ[10]; ExactNumberQ[10.0]; ExactNumberQ[I]; ExactNumberQ[1 + I]; ExactNumberQ[1. + I]; ExactNumberQ[5/6]; ExactNumberQ[4 * I + 5/6]"#,
      r#"True"#,
    );
  }
  #[test]
  fn inexact_number_q_1() {
    assert_case(r#"InexactNumberQ[a]"#, r#"False"#);
  }
  #[test]
  fn inexact_number_q_2() {
    assert_case(r#"InexactNumberQ[a]; InexactNumberQ[3.0]"#, r#"True"#);
  }
  #[test]
  fn inexact_number_q_3() {
    assert_case(
      r#"InexactNumberQ[a]; InexactNumberQ[3.0]; InexactNumberQ[2/3]"#,
      r#"False"#,
    );
  }
  #[test]
  fn inexact_number_q_4() {
    assert_case(
      r#"InexactNumberQ[a]; InexactNumberQ[3.0]; InexactNumberQ[2/3]; InexactNumberQ[4.0+I]"#,
      r#"True"#,
    );
  }
  #[test]
  fn machine_number_q_1() {
    assert_case(r#"MachineNumberQ[3.14159265358979324]"#, r#"False"#);
  }
  #[test]
  fn machine_number_q_2() {
    assert_case(
      r#"MachineNumberQ[3.14159265358979324]; MachineNumberQ[1.5 + 2.3 I]"#,
      r#"True"#,
    );
  }
  #[test]
  fn machine_number_q_3() {
    assert_case(
      r#"MachineNumberQ[3.14159265358979324]; MachineNumberQ[1.5 + 2.3 I]; MachineNumberQ[2.71828182845904524 + 3.14159265358979324 I]"#,
      r#"False"#,
    );
  }
  #[test]
  fn accuracy_1() {
    assert_case(r#"Accuracy[3.1416`2]"#, r#"1.5028491117376408"#);
  }
  #[test]
  fn accuracy_2() {
    assert_case(r#"Accuracy[3.1416`2]; Accuracy[1]"#, r#"Infinity"#);
  }
  #[test]
  fn accuracy_3() {
    assert_case(
      r#"Accuracy[3.1416`2]; Accuracy[1]; Accuracy[A]"#,
      r#"Infinity"#,
    );
  }
  #[test]
  fn accuracy_4() {
    assert_case(
      r#"Accuracy[3.1416`2]; Accuracy[1]; Accuracy[A]; z=Complex[3.00``2, 4.00``2]; Accuracy[z] == -Log[10, Sqrt[10^(-2 Accuracy[Re[z]]) + 10^(-2 Accuracy[Im[z]])]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn n_24() {
    assert_case(r#"N[MachinePrecision]"#, r#"15.954589770191003"#);
  }
  #[test]
  fn precision_2() {
    assert_case(r#"Precision[1]"#, r#"Infinity"#);
  }
  #[test]
  fn precision_3() {
    assert_case(r#"Precision[1]; Precision[1/2]"#, r#"Infinity"#);
  }
  #[test]
  fn precision_4() {
    assert_case(
      r#"Precision[1]; Precision[1/2]; Precision[1.23`10]"#,
      r#"10."#,
    );
  }
  #[test]
  fn precision_5() {
    assert_case(
      r#"Precision[1]; Precision[1/2]; Precision[1.23`10]; Precision[0.5]"#,
      r#"MachinePrecision"#,
    );
  }
  #[test]
  fn n_25() {
    assert_case(
      r#"30 / 5; 1 / 8; Pi / 4; Pi / 4.0; 1 / 8; N[%]"#,
      r#"Out[0]"#,
    );
  }
  #[test]
  fn divide_2() {
    assert_case(
      r#"30 / 5; 1 / 8; Pi / 4; Pi / 4.0; 1 / 8; N[%]; a / b / c"#,
      r#"a/(b*c)"#,
    );
  }
  #[test]
  fn divide_3() {
    assert_case(
      r#"30 / 5; 1 / 8; Pi / 4; Pi / 4.0; 1 / 8; N[%]; a / b / c; a / (b / c)"#,
      r#"(a*c)/b"#,
    );
  }
  #[test]
  fn divide_4() {
    assert_case(
      r#"30 / 5; 1 / 8; Pi / 4; Pi / 4.0; 1 / 8; N[%]; a / b / c; a / (b / c); a / b / (c / (d / e))"#,
      r#"(a*d)/(b*c*e)"#,
    );
  }
  #[test]
  fn times() {
    assert_case(
      r#"30 / 5; 1 / 8; Pi / 4; Pi / 4.0; 1 / 8; N[%]; a / b / c; a / (b / c); a / b / (c / (d / e)); a / (b ^ 2 * c ^ 3 / e)"#,
      r#"(a*e)/(b^2*c^3)"#,
    );
  }
  #[test]
  fn n_26() {
    assert_case(
      r#"5!!; Factorial2[-3]; I!! + 1; N[Pi!!, 6]"#,
      r#"3.3523681241546551093`6."#,
    );
  }
  #[test]
  fn n_27() {
    assert_case(r#"N[3^200]"#, r#"2.6561398887587478*^95"#);
  }
  #[test]
  fn n_28() {
    assert_case(r#"N[3^200]; N[2^1023]"#, r#"8.98846567431158*^307"#);
  }
  #[test]
  fn n_29() {
    assert_case(
      r#"N[3^200]; N[2^1023]; N[2^1024]"#,
      r#"1.79769313486231590772930519078902473362`15.954589770191005*^308"#,
    );
  }
  #[test]
  fn set() {
    assert_case(
      r#"N[3^200]; N[2^1023]; N[2^1024]; p=N[Pi,100]"#,
      r#"3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865191976`100."#,
    );
  }
  #[test]
  fn machine_number_q_4() {
    assert_case(r#"MachineNumberQ[1.5 + 3.14159265358979324 I]"#, r#"True"#);
  }
  #[test]
  fn machine_number_q_5() {
    assert_case(
      r#"MachineNumberQ[1.5 + 3.14159265358979324 I]; MachineNumberQ[1.5 + 5 I]"#,
      r#"True"#,
    );
  }
  #[test]
  fn n_30() {
    assert_case(
      r#"ArcTan[ComplexInfinity]; ArcTan[-1, 1]; ArcTan[1, -1]; ArcTan[-1, -1]; ArcTan[1, 0]; ArcTan[-1, 0]; ArcTan[0, 1]; ArcTan[0, -1]; Cos[1.5 Pi]; N[Sin[1], 40]"#,
      r#"0.8414709848078965066525023216302989996225630607983710656728`40."#,
    );
  }
  #[test]
  fn n_31() {
    assert_case(r#"N[Sqrt[2], 41]//Precision"#, r#"41."#);
  }
  #[test]
  fn n_32() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision"#,
      r#"40."#,
    );
  }
  #[test]
  fn n_33() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision"#,
      r#"41."#,
    );
  }
  #[test]
  fn n_34() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision"#,
      r#"40."#,
    );
  }
  #[test]
  fn n_35() {
    assert_case(
      r#"N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]//Precision; N[Sqrt[2], 40]//Precision; N[Sqrt[2], 41]"#,
      r#"1.4142135623730950488016887242096980785696718753769480731767`41."#,
    );
  }
  #[test]
  fn chop_5() {
    assert_case(
      r#"E^(3+I Pi); E^(I Pi/2); E^1; log2=Log[2.]; E^log2; log2=Log[2.]; Chop[E^(log2+I Pi)]"#,
      r#"-2."#,
    );
  }
}
