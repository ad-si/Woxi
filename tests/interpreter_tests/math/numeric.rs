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
  // and BigFloat contributes its literal precision. For
  // `{{1, 1.`}, {1.`5, 1.`10}}`, the minimum is 5. Regression for the
  // mathics numbers.py doctest `Precision[{{1, 1.`},{1.`5, 1.`10}}] == 5.`.
  #[test]
  fn precision_nested_list_minimum() {
    assert_eq!(
      interpret("Precision[{{1, 1.`},{1.`5, 1.`10}}]").unwrap(),
      "5."
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
