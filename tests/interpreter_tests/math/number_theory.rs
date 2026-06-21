use super::*;

mod factor_integer {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("FactorInteger[12]").unwrap(), "{{2, 2}, {3, 1}}");
    assert_eq!(
      interpret("FactorInteger[360]").unwrap(),
      "{{2, 3}, {3, 2}, {5, 1}}"
    );
  }

  #[test]
  fn one() {
    // Regression: FactorInteger[1] was returning {} instead of {{1, 1}}
    assert_eq!(interpret("FactorInteger[1]").unwrap(), "{{1, 1}}");
  }

  // wolframscript: FactorInteger[0] = {{0, 1}} (0 treated as 0^1) rather
  // than raising an error.
  #[test]
  fn zero() {
    assert_eq!(interpret("FactorInteger[0]").unwrap(), "{{0, 1}}");
  }

  // Unit-numerator rationals must not carry a spurious {1, 1} factor.
  #[test]
  fn unit_numerator_rational() {
    assert_eq!(
      interpret("FactorInteger[1/6]").unwrap(),
      "{{2, -1}, {3, -1}}"
    );
    assert_eq!(interpret("FactorInteger[1/2]").unwrap(), "{{2, -1}}");
    assert_eq!(
      interpret("FactorInteger[1/12]").unwrap(),
      "{{2, -2}, {3, -1}}"
    );
  }

  #[test]
  fn negative() {
    assert_eq!(
      interpret("FactorInteger[-12]").unwrap(),
      "{{-1, 1}, {2, 2}, {3, 1}}"
    );
    assert_eq!(interpret("FactorInteger[-1]").unwrap(), "{{-1, 1}}");
  }

  #[test]
  fn prime() {
    assert_eq!(interpret("FactorInteger[7]").unwrap(), "{{7, 1}}");
    assert_eq!(interpret("FactorInteger[97]").unwrap(), "{{97, 1}}");
  }

  #[test]
  fn prime_power() {
    assert_eq!(interpret("FactorInteger[8]").unwrap(), "{{2, 3}}");
    assert_eq!(interpret("FactorInteger[27]").unwrap(), "{{3, 3}}");
  }

  #[test]
  fn rational() {
    assert_eq!(
      interpret("FactorInteger[2010 / 2011]").unwrap(),
      "{{2, 1}, {3, 1}, {5, 1}, {67, 1}, {2011, -1}}"
    );
    assert_eq!(
      interpret("FactorInteger[12/35]").unwrap(),
      "{{2, 2}, {3, 1}, {5, -1}, {7, -1}}"
    );
    assert_eq!(
      interpret("FactorInteger[-12/35]").unwrap(),
      "{{-1, 1}, {2, 2}, {3, 1}, {5, -1}, {7, -1}}"
    );
  }
}

mod divisible {
  use super::*;

  #[test]
  fn divisible_true() {
    assert_eq!(interpret("Divisible[10, 2]").unwrap(), "True");
    assert_eq!(interpret("Divisible[15, 3]").unwrap(), "True");
    assert_eq!(interpret("Divisible[15, 5]").unwrap(), "True");
    assert_eq!(interpret("Divisible[100, 10]").unwrap(), "True");
  }

  #[test]
  fn divisible_false() {
    assert_eq!(interpret("Divisible[10, 3]").unwrap(), "False");
    assert_eq!(interpret("Divisible[7, 2]").unwrap(), "False");
    assert_eq!(interpret("Divisible[11, 5]").unwrap(), "False");
  }

  #[test]
  fn divisible_by_one() {
    assert_eq!(interpret("Divisible[5, 1]").unwrap(), "True");
    assert_eq!(interpret("Divisible[0, 1]").unwrap(), "True");
  }

  #[test]
  fn divisible_zero() {
    assert_eq!(interpret("Divisible[0, 5]").unwrap(), "True");
  }

  #[test]
  fn divisible_non_integer() {
    // Wolfram returns unevaluated for non-exact numbers
    assert_eq!(interpret("Divisible[5.5, 2]").unwrap(), "Divisible[5.5, 2]");
    assert_eq!(
      interpret("Divisible[10, 2.5]").unwrap(),
      "Divisible[10, 2.5]"
    );
  }

  #[test]
  fn divisible_division_by_zero() {
    assert!(interpret("Divisible[5, 0]").is_err());
  }
}

mod fibonacci_builtin {
  use super::*;

  #[test]
  fn fibonacci_zero() {
    assert_eq!(interpret("Fibonacci[0]").unwrap(), "0");
  }

  #[test]
  fn fibonacci_one() {
    assert_eq!(interpret("Fibonacci[1]").unwrap(), "1");
  }

  #[test]
  fn fibonacci_small() {
    assert_eq!(interpret("Fibonacci[9]").unwrap(), "34");
    assert_eq!(interpret("Fibonacci[10]").unwrap(), "55");
  }

  #[test]
  fn fibonacci_negative() {
    assert_eq!(interpret("Fibonacci[-1]").unwrap(), "1");
    assert_eq!(interpret("Fibonacci[-2]").unwrap(), "-1");
    assert_eq!(interpret("Fibonacci[-3]").unwrap(), "2");
    assert_eq!(interpret("Fibonacci[-4]").unwrap(), "-3");
  }

  #[test]
  fn fibonacci_symbolic() {
    assert_eq!(interpret("Fibonacci[x]").unwrap(), "Fibonacci[x]");
  }

  // Real arguments use the analytic continuation
  // Fibonacci[x] = (phi^x - Cos[pi x] phi^-x)/Sqrt[5].
  #[test]
  fn fibonacci_real_argument() {
    assert_eq!(interpret("Fibonacci[2.0]").unwrap(), "1.");
    assert_eq!(interpret("Fibonacci[3.0]").unwrap(), "2.");
    assert_eq!(interpret("Fibonacci[2.5]").unwrap(), "1.4893065462657091");
    assert_eq!(interpret("Fibonacci[0.5]").unwrap(), "0.5688644810057831");
    assert_eq!(interpret("Fibonacci[-0.5]").unwrap(), "0.35157758425414287");
    // N[Fibonacci[1/2]] numericizes the index, then evaluates.
    assert_eq!(
      interpret("N[Fibonacci[1/2]]").unwrap(),
      "0.5688644810057831"
    );
  }

  // LucasL[x] = phi^x + Cos[pi x] phi^-x for real arguments.
  #[test]
  fn lucas_l_real_argument() {
    assert_eq!(interpret("LucasL[0.5]").unwrap(), "1.272019649514069");
    assert_eq!(interpret("N[LucasL[1/2]]").unwrap(), "1.272019649514069");
  }

  #[test]
  fn fibonacci_polynomial_n_one() {
    // F_1(x) = 1
    assert_eq!(interpret("Fibonacci[1, x]").unwrap(), "1");
  }

  #[test]
  fn fibonacci_polynomial_n_zero() {
    // F_0(x) = 0
    assert_eq!(interpret("Fibonacci[0, x]").unwrap(), "0");
  }

  #[test]
  fn fibonacci_polynomial_small() {
    // F_2(x) = x; F_3(x) = 1 + x^2; F_4(x) = 2 x + x^3
    assert_eq!(interpret("Fibonacci[2, x]").unwrap(), "x");
    assert_eq!(interpret("Fibonacci[3, x]").unwrap(), "1 + x^2");
    assert_eq!(interpret("Fibonacci[4, x]").unwrap(), "2*x + x^3");
  }

  #[test]
  fn fibonacci_polynomial_six() {
    // F_6(x) = 3 x + 4 x^3 + x^5
    assert_eq!(interpret("Fibonacci[6, x]").unwrap(), "3*x + 4*x^3 + x^5");
  }

  #[test]
  fn fibonacci_polynomial_ten() {
    // F_10(x) = 5 x + 20 x^3 + 21 x^5 + 8 x^7 + x^9
    assert_eq!(
      interpret("Fibonacci[10, x]").unwrap(),
      "5*x + 20*x^3 + 21*x^5 + 8*x^7 + x^9"
    );
  }

  #[test]
  fn fibonacci_polynomial_at_one_matches_fibonacci() {
    // Fibonacci[n, 1] == Fibonacci[n]
    assert_eq!(interpret("Fibonacci[10, 1]").unwrap(), "55");
  }

  #[test]
  fn fibonacci_polynomial_at_two() {
    // Fibonacci[5, 2] = 29 (Pell numbers)
    assert_eq!(interpret("Fibonacci[5, 2]").unwrap(), "29");
  }

  #[test]
  fn fibonacci_polynomial_negative_index() {
    // F_{-n}(x) = (-1)^{n+1} F_n(x)
    // F_{-3}(x) = F_3(x) = 1 + x^2
    assert_eq!(interpret("Fibonacci[-3, x]").unwrap(), "1 + x^2");
  }
}

mod lucas_l_builtin {
  use super::*;

  #[test]
  fn lucas_small() {
    assert_eq!(interpret("LucasL[0]").unwrap(), "2");
    assert_eq!(interpret("LucasL[1]").unwrap(), "1");
    assert_eq!(interpret("LucasL[10]").unwrap(), "123");
  }

  // L_{-n} = (-1)^n L_n.
  #[test]
  fn lucas_negative_index() {
    assert_eq!(interpret("LucasL[-1]").unwrap(), "-1");
    assert_eq!(interpret("LucasL[-2]").unwrap(), "3");
    assert_eq!(interpret("LucasL[-3]").unwrap(), "-4");
    assert_eq!(interpret("LucasL[-4]").unwrap(), "7");
    assert_eq!(interpret("LucasL[-10]").unwrap(), "123");
  }

  #[test]
  fn lucas_negative_index_table() {
    assert_eq!(
      interpret("Table[LucasL[-n], {n, 1, 8}]").unwrap(),
      "{-1, 3, -4, 7, -11, 18, -29, 47}"
    );
  }

  // Large negative index still uses big-integer arithmetic.
  #[test]
  fn lucas_large_negative_index() {
    assert_eq!(interpret("LucasL[-100]").unwrap(), "792070839848372253127");
  }

  #[test]
  fn lucas_polynomial_small() {
    assert_eq!(interpret("LucasL[0, x]").unwrap(), "2");
    assert_eq!(interpret("LucasL[1, x]").unwrap(), "x");
    assert_eq!(interpret("LucasL[5, 3]").unwrap(), "393");
  }

  // L_{-n}(x) = (-1)^n L_n(x).
  #[test]
  fn lucas_polynomial_negative_index() {
    assert_eq!(interpret("LucasL[-1, x]").unwrap(), "-x");
    assert_eq!(interpret("LucasL[-2, x]").unwrap(), "2 + x^2");
    assert_eq!(interpret("LucasL[-3, x]").unwrap(), "-3*x - x^3");
    assert_eq!(interpret("LucasL[-5, 3]").unwrap(), "-393");
  }

  #[test]
  fn lucas_symbolic_unevaluated() {
    assert_eq!(interpret("LucasL[n]").unwrap(), "LucasL[n]");
  }
}

mod catalan_number_builtin {
  use super::*;

  #[test]
  fn non_negative() {
    assert_eq!(interpret("CatalanNumber[0]").unwrap(), "1");
    assert_eq!(interpret("CatalanNumber[1]").unwrap(), "1");
    assert_eq!(interpret("CatalanNumber[5]").unwrap(), "42");
    assert_eq!(interpret("CatalanNumber[10]").unwrap(), "16796");
  }

  // The analytic continuation collapses at negative integers:
  // CatalanNumber[-1] = -1 and CatalanNumber[-n] = 0 for n >= 2.
  #[test]
  fn negative_index() {
    assert_eq!(interpret("CatalanNumber[-1]").unwrap(), "-1");
    assert_eq!(interpret("CatalanNumber[-2]").unwrap(), "0");
    assert_eq!(interpret("CatalanNumber[-3]").unwrap(), "0");
    assert_eq!(interpret("CatalanNumber[-100]").unwrap(), "0");
  }

  #[test]
  fn negative_index_table() {
    assert_eq!(
      interpret("Table[CatalanNumber[-n], {n, 1, 6}]").unwrap(),
      "{-1, 0, 0, 0, 0, 0}"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("CatalanNumber[n]").unwrap(), "CatalanNumber[n]");
  }

  // A real index evaluates numerically. An integer-valued real gives the
  // exact integer rounded to a machine real (not the unevaluated form);
  // negative integer-valued reals follow the same collapse as exact ones.
  // Verified against wolframscript.
  #[test]
  fn integer_valued_real_index() {
    assert_eq!(interpret("CatalanNumber[3.0]").unwrap(), "5.");
    assert_eq!(interpret("CatalanNumber[5.0]").unwrap(), "42.");
    assert_eq!(interpret("CatalanNumber[0.0]").unwrap(), "1.");
    assert_eq!(interpret("CatalanNumber[10.0]").unwrap(), "16796.");
    assert_eq!(interpret("CatalanNumber[-1.0]").unwrap(), "-1.");
    assert_eq!(interpret("CatalanNumber[-2.0]").unwrap(), "0.");
  }
}

mod euler_phi {
  use super::*;

  #[test]
  fn basic_values() {
    assert_eq!(interpret("EulerPhi[1]").unwrap(), "1");
    assert_eq!(interpret("EulerPhi[10]").unwrap(), "4");
    assert_eq!(interpret("EulerPhi[12]").unwrap(), "4");
    assert_eq!(interpret("EulerPhi[100]").unwrap(), "40");
  }

  #[test]
  fn prime_argument() {
    // EulerPhi[p] = p - 1 for prime p
    assert_eq!(interpret("EulerPhi[7]").unwrap(), "6");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("EulerPhi[0]").unwrap(), "0");
  }

  #[test]
  fn negative_integers() {
    // EulerPhi[-n] == EulerPhi[n]
    assert_eq!(interpret("EulerPhi[-1]").unwrap(), "1");
    assert_eq!(interpret("EulerPhi[-12]").unwrap(), "4");
    assert_eq!(interpret("EulerPhi[-11] == EulerPhi[11]").unwrap(), "True");
  }

  #[test]
  fn list_argument() {
    assert_eq!(
      interpret("EulerPhi[{1, 2, 3, 4, 5}]").unwrap(),
      "{1, 1, 2, 2, 4}"
    );
  }

  #[test]
  fn big_integer_argument() {
    // EulerPhi[40!] exceeds i128 and requires BigInt arithmetic.
    assert_eq!(
      interpret("EulerPhi[40!]").unwrap(),
      "121343746763281707274905415180804423680000000000"
    );
  }
}

mod integer_exponent {
  use super::*;

  #[test]
  fn default_base_is_10() {
    // Regression: IntegerExponent[n] used to use base 2 instead of 10.
    assert_eq!(interpret("IntegerExponent[100]").unwrap(), "2");
    assert_eq!(interpret("IntegerExponent[1000]").unwrap(), "3");
  }

  #[test]
  fn with_base() {
    assert_eq!(interpret("IntegerExponent[40, 2]").unwrap(), "3");
    assert_eq!(interpret("IntegerExponent[125, 5]").unwrap(), "3");
  }

  #[test]
  fn big_integer() {
    // Regression: IntegerExponent[100!, 10] used to return unevaluated
    // because the factorial overflows i128.
    assert_eq!(interpret("IntegerExponent[100!, 10]").unwrap(), "24");
  }

  #[test]
  fn zero_gives_infinity() {
    assert_eq!(interpret("IntegerExponent[0, 2]").unwrap(), "Infinity");
  }

  // A base that is not an integer greater than 1 emits IntegerExponent::ibase
  // and stays unevaluated (matching wolframscript).
  #[test]
  fn invalid_base_stays_unevaluated() {
    assert_eq!(
      interpret("IntegerExponent[12, 1]").unwrap(),
      "IntegerExponent[12, 1]"
    );
    assert_eq!(
      interpret("IntegerExponent[12, 0]").unwrap(),
      "IntegerExponent[12, 0]"
    );
    assert_eq!(
      interpret("IntegerExponent[12, -2]").unwrap(),
      "IntegerExponent[12, -2]"
    );
    assert_eq!(
      interpret("IntegerExponent[12, 3/2]").unwrap(),
      "IntegerExponent[12, 3/2]"
    );
  }

  // The invalid-base check runs before the n == 0 short-circuit, so this is
  // ibase rather than Infinity.
  #[test]
  fn invalid_base_beats_zero() {
    assert_eq!(
      interpret("IntegerExponent[0, 1]").unwrap(),
      "IntegerExponent[0, 1]"
    );
  }
}

mod integer_length {
  use super::*;

  #[test]
  fn integer_length_basic() {
    assert_eq!(interpret("IntegerLength[12345]").unwrap(), "5");
    assert_eq!(interpret("IntegerLength[0]").unwrap(), "0");
    assert_eq!(interpret("IntegerLength[1]").unwrap(), "1");
    assert_eq!(interpret("IntegerLength[-99]").unwrap(), "2");
  }

  #[test]
  fn integer_length_base() {
    assert_eq!(interpret("IntegerLength[255, 16]").unwrap(), "2");
    assert_eq!(interpret("IntegerLength[100, 10]").unwrap(), "3");
    assert_eq!(interpret("IntegerLength[255, 2]").unwrap(), "8");
    assert_eq!(interpret("IntegerLength[12345, 2]").unwrap(), "14");
  }

  #[test]
  fn integer_length_big_integer() {
    assert_eq!(interpret("IntegerLength[10^10000]").unwrap(), "10001");
    assert_eq!(interpret("IntegerLength[-10^1000]").unwrap(), "1001");
    assert_eq!(interpret("IntegerLength[10^100]").unwrap(), "101");
  }

  #[test]
  fn integer_length_mapped_over_big_powers() {
    // Regression: `10 ^ Range[100]` used to thread Power via f64::powf,
    // collapsing 10^40, 10^41, ... to lossy Reals so IntegerLength couldn't
    // evaluate them.
    assert_eq!(
      interpret("IntegerLength /@ (10 ^ Range[100]) == Range[2, 101]").unwrap(),
      "True"
    );
  }
}

mod integer_reverse {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("IntegerReverse[1234]").unwrap(), "4321");
  }

  #[test]
  fn base_2() {
    assert_eq!(interpret("IntegerReverse[1022, 2]").unwrap(), "511");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("IntegerReverse[-123]").unwrap(), "321");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("IntegerReverse[0]").unwrap(), "0");
  }
}

mod harmonic_number {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("HarmonicNumber[5]").unwrap(), "137/60");
    assert_eq!(interpret("HarmonicNumber[10]").unwrap(), "7381/2520");
  }

  #[test]
  fn edge_cases() {
    assert_eq!(interpret("HarmonicNumber[0]").unwrap(), "0");
    assert_eq!(interpret("HarmonicNumber[1]").unwrap(), "1");
  }

  #[test]
  fn generalized() {
    assert_eq!(interpret("HarmonicNumber[5, 2]").unwrap(), "5269/3600");
  }

  #[test]
  fn negative_order() {
    // HarmonicNumber[n, -r] = Sum[k^r, {k, 1, n}]
    assert_eq!(interpret("HarmonicNumber[3, -1]").unwrap(), "6");
    assert_eq!(interpret("HarmonicNumber[3, -2]").unwrap(), "14");
    assert_eq!(interpret("HarmonicNumber[4, -1]").unwrap(), "10");
    assert_eq!(interpret("HarmonicNumber[5, -2]").unwrap(), "55");
    assert_eq!(interpret("HarmonicNumber[0, -1]").unwrap(), "0");
  }

  // The limiting value of the (generalized) harmonic series at Infinity.
  #[test]
  fn at_infinity() {
    assert_eq!(interpret("HarmonicNumber[Infinity, 2]").unwrap(), "Pi^2/6");
    assert_eq!(interpret("HarmonicNumber[Infinity, 3]").unwrap(), "Zeta[3]");
    assert_eq!(interpret("HarmonicNumber[Infinity]").unwrap(), "Infinity");
    assert_eq!(
      interpret("HarmonicNumber[Infinity, 1]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("HarmonicNumber[Infinity, 1/2]").unwrap(),
      "Infinity"
    );
    assert_eq!(
      interpret("HarmonicNumber[Infinity, -1]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn real_argument() {
    assert_eq!(
      interpret("HarmonicNumber[3.8]").unwrap(),
      "2.0380634056306492"
    );
  }
}

mod integer_name {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("IntegerName[0]").unwrap(), "zero");
  }

  #[test]
  fn small_numbers() {
    assert_eq!(interpret("IntegerName[1]").unwrap(), "one");
    assert_eq!(interpret("IntegerName[10]").unwrap(), "ten");
    assert_eq!(interpret("IntegerName[11]").unwrap(), "eleven");
    assert_eq!(interpret("IntegerName[15]").unwrap(), "fifteen");
    assert_eq!(interpret("IntegerName[19]").unwrap(), "nineteen");
  }

  #[test]
  fn tens() {
    assert_eq!(interpret("IntegerName[20]").unwrap(), "twenty");
    assert_eq!(interpret("IntegerName[21]").unwrap(), "twenty\u{2010}one");
    assert_eq!(interpret("IntegerName[50]").unwrap(), "fifty");
    assert_eq!(interpret("IntegerName[99]").unwrap(), "ninety\u{2010}nine");
  }

  #[test]
  fn hundreds() {
    assert_eq!(interpret("IntegerName[100]").unwrap(), "one hundred");
    assert_eq!(interpret("IntegerName[101]").unwrap(), "one hundred one");
    assert_eq!(
      interpret("IntegerName[123]").unwrap(),
      "one hundred twenty\u{2010}three"
    );
    assert_eq!(
      interpret("IntegerName[999]").unwrap(),
      "nine hundred ninety\u{2010}nine"
    );
  }

  #[test]
  fn thousands() {
    assert_eq!(interpret("IntegerName[1000]").unwrap(), "1 thousand");
    assert_eq!(interpret("IntegerName[1001]").unwrap(), "1 thousand 1");
    assert_eq!(interpret("IntegerName[1100]").unwrap(), "1 thousand 100");
    assert_eq!(interpret("IntegerName[2500]").unwrap(), "2 thousand 500");
    assert_eq!(interpret("IntegerName[10000]").unwrap(), "10 thousand");
    assert_eq!(
      interpret("IntegerName[123456]").unwrap(),
      "123 thousand 456"
    );
  }

  #[test]
  fn millions_and_above() {
    assert_eq!(interpret("IntegerName[1000000]").unwrap(), "1 million");
    assert_eq!(
      interpret("IntegerName[987654321]").unwrap(),
      "987 million 654 thousand 321"
    );
    assert_eq!(
      interpret("IntegerName[1000000000000]").unwrap(),
      "1 trillion"
    );
    assert_eq!(
      interpret("IntegerName[1000000000000000]").unwrap(),
      "1 quadrillion"
    );
    assert_eq!(
      interpret("IntegerName[1000000000000000000]").unwrap(),
      "1 quintillion"
    );
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("IntegerName[-5]").unwrap(), "negative five");
    assert_eq!(
      interpret("IntegerName[-999]").unwrap(),
      "negative nine hundred ninety\u{2010}nine"
    );
    assert_eq!(
      interpret("IntegerName[-1000]").unwrap(),
      "negative 1 thousand"
    );
  }

  #[test]
  fn non_integer_returns_symbolic() {
    assert_eq!(interpret("IntegerName[x]").unwrap(), "IntegerName[x]");
  }

  #[test]
  fn two_arg_form() {
    // IntegerName[n, "Words"] is accepted (second arg is a format hint)
    assert_eq!(
      interpret(r#"IntegerName[42, "Words"]"#).unwrap(),
      "forty\u{2010}two"
    );
  }

  // IntegerName[n, "Ordinal"] gives the ordinal name (regular hyphen).
  #[test]
  fn ordinal_units() {
    assert_eq!(interpret(r#"IntegerName[1, "Ordinal"]"#).unwrap(), "first");
    assert_eq!(interpret(r#"IntegerName[3, "Ordinal"]"#).unwrap(), "third");
    assert_eq!(interpret(r#"IntegerName[8, "Ordinal"]"#).unwrap(), "eighth");
    assert_eq!(interpret(r#"IntegerName[9, "Ordinal"]"#).unwrap(), "ninth");
    assert_eq!(
      interpret(r#"IntegerName[12, "Ordinal"]"#).unwrap(),
      "twelfth"
    );
  }

  #[test]
  fn ordinal_tens() {
    assert_eq!(
      interpret(r#"IntegerName[20, "Ordinal"]"#).unwrap(),
      "twentieth"
    );
    assert_eq!(
      interpret(r#"IntegerName[42, "Ordinal"]"#).unwrap(),
      "forty-second"
    );
    assert_eq!(
      interpret(r#"IntegerName[99, "Ordinal"]"#).unwrap(),
      "ninety-ninth"
    );
  }

  #[test]
  fn ordinal_hundreds() {
    assert_eq!(
      interpret(r#"IntegerName[100, "Ordinal"]"#).unwrap(),
      "one hundredth"
    );
    assert_eq!(
      interpret(r#"IntegerName[142, "Ordinal"]"#).unwrap(),
      "one hundred forty-second"
    );
  }

  #[test]
  fn ordinal_list() {
    assert_eq!(
      interpret(r#"IntegerName[{1, 2, 3}, "Ordinal"]"#).unwrap(),
      "{first, second, third}"
    );
  }
}

mod roman_numeral {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("RomanNumeral[0]").unwrap(), "N");
  }

  #[test]
  fn basic_values() {
    assert_eq!(interpret("RomanNumeral[1]").unwrap(), "I");
    assert_eq!(interpret("RomanNumeral[5]").unwrap(), "V");
    assert_eq!(interpret("RomanNumeral[10]").unwrap(), "X");
    assert_eq!(interpret("RomanNumeral[50]").unwrap(), "L");
    assert_eq!(interpret("RomanNumeral[100]").unwrap(), "C");
    assert_eq!(interpret("RomanNumeral[500]").unwrap(), "D");
    assert_eq!(interpret("RomanNumeral[1000]").unwrap(), "M");
  }

  #[test]
  fn subtractive_forms() {
    assert_eq!(interpret("RomanNumeral[4]").unwrap(), "IV");
    assert_eq!(interpret("RomanNumeral[9]").unwrap(), "IX");
    assert_eq!(interpret("RomanNumeral[40]").unwrap(), "XL");
    assert_eq!(interpret("RomanNumeral[90]").unwrap(), "XC");
    assert_eq!(interpret("RomanNumeral[400]").unwrap(), "CD");
    assert_eq!(interpret("RomanNumeral[900]").unwrap(), "CM");
  }

  #[test]
  fn complex_numbers() {
    assert_eq!(interpret("RomanNumeral[14]").unwrap(), "XIV");
    assert_eq!(interpret("RomanNumeral[49]").unwrap(), "XLIX");
    assert_eq!(interpret("RomanNumeral[99]").unwrap(), "XCIX");
    assert_eq!(interpret("RomanNumeral[444]").unwrap(), "CDXLIV");
    assert_eq!(interpret("RomanNumeral[1999]").unwrap(), "MCMXCIX");
    assert_eq!(interpret("RomanNumeral[2025]").unwrap(), "MMXXV");
    assert_eq!(interpret("RomanNumeral[3999]").unwrap(), "MMMCMXCIX");
  }

  #[test]
  fn extended_range() {
    assert_eq!(interpret("RomanNumeral[4000]").unwrap(), "MMMM");
    assert_eq!(interpret("RomanNumeral[4999]").unwrap(), "MMMMCMXCIX");
  }

  #[test]
  fn negative_integers() {
    assert_eq!(interpret("RomanNumeral[-1]").unwrap(), "I");
    assert_eq!(interpret("RomanNumeral[-5]").unwrap(), "V");
    assert_eq!(interpret("RomanNumeral[-2025]").unwrap(), "MMXXV");
  }

  #[test]
  fn non_integer_returns_unevaluated() {
    assert_eq!(interpret("RomanNumeral[x]").unwrap(), "RomanNumeral[x]");
    assert_eq!(interpret("RomanNumeral[1.5]").unwrap(), "RomanNumeral[1.5]");
  }
}

mod from_digits {
  use super::*;

  #[test]
  fn base_10() {
    assert_eq!(interpret("FromDigits[{1, 2, 3}]").unwrap(), "123");
  }

  #[test]
  fn base_2() {
    assert_eq!(interpret("FromDigits[{1, 0, 1}, 2]").unwrap(), "5");
  }

  #[test]
  fn base_16() {
    assert_eq!(interpret("FromDigits[{1, 0}, 16]").unwrap(), "16");
  }

  #[test]
  fn single_digit() {
    assert_eq!(interpret("FromDigits[{7}]").unwrap(), "7");
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("FromDigits[{}]").unwrap(), "0");
  }

  #[test]
  fn symbolic_base_numeric_digits() {
    assert_eq!(
      interpret("FromDigits[{1, 2, 3}, x]").unwrap(),
      "3 + 2*x + x^2"
    );
  }

  #[test]
  fn symbolic_base_with_zero_digit() {
    assert_eq!(interpret("FromDigits[{1, 0, 1}, x]").unwrap(), "1 + x^2");
  }

  #[test]
  fn symbolic_base_symbolic_digits() {
    assert_eq!(
      interpret("FromDigits[{a, b, c}, x]").unwrap(),
      "c + b*x + a*x^2"
    );
  }

  #[test]
  fn symbolic_base_single_digit() {
    assert_eq!(interpret("FromDigits[{3}, x]").unwrap(), "3");
  }

  #[test]
  fn string_with_overflow_digits() {
    // "a" = 10, "0" = 0, so 10*10 + 0 = 100
    assert_eq!(interpret(r#"FromDigits["a0"]"#).unwrap(), "100");
  }

  #[test]
  fn string_numeric() {
    assert_eq!(interpret(r#"FromDigits["1234"]"#).unwrap(), "1234");
  }
}

mod integer_string {
  use super::*;

  #[test]
  fn base_10() {
    assert_eq!(interpret("IntegerString[42]").unwrap(), "42");
  }

  #[test]
  fn base_16() {
    assert_eq!(interpret("IntegerString[255, 16]").unwrap(), "ff");
  }

  #[test]
  fn base_2() {
    assert_eq!(interpret("IntegerString[255, 2]").unwrap(), "11111111");
  }

  #[test]
  fn with_padding() {
    assert_eq!(interpret("IntegerString[255, 16, 4]").unwrap(), "00ff");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("IntegerString[0]").unwrap(), "0");
  }

  #[test]
  fn zero_base_2() {
    assert_eq!(interpret("IntegerString[0, 2]").unwrap(), "0");
  }
}

mod binomial {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Binomial[10, 3]").unwrap(), "120");
  }

  #[test]
  fn zero_k() {
    assert_eq!(interpret("Binomial[5, 0]").unwrap(), "1");
  }

  #[test]
  fn k_equals_n() {
    assert_eq!(interpret("Binomial[5, 5]").unwrap(), "1");
  }

  #[test]
  fn k_greater_than_n() {
    assert_eq!(interpret("Binomial[3, 5]").unwrap(), "0");
  }

  #[test]
  fn negative_n() {
    assert_eq!(interpret("Binomial[-3, 2]").unwrap(), "6");
  }

  // Negative k is zero when n >= 0.
  #[test]
  fn negative_k_nonneg_n() {
    assert_eq!(interpret("Binomial[2, -1]").unwrap(), "0");
    assert_eq!(interpret("Binomial[5, -3]").unwrap(), "0");
  }

  // Both arguments negative: Binomial[n, k] = (-1)^(n+k) Binomial[-k-1, -n-1].
  #[test]
  fn negative_n_negative_k() {
    assert_eq!(interpret("Binomial[-3, -3]").unwrap(), "1");
    assert_eq!(interpret("Binomial[-1, -1]").unwrap(), "1");
    assert_eq!(interpret("Binomial[-3, -5]").unwrap(), "6");
    assert_eq!(interpret("Binomial[-1, -3]").unwrap(), "1");
    // Zero when -k-1 < -n-1 (i.e. k > n).
    assert_eq!(interpret("Binomial[-2, -1]").unwrap(), "0");
    assert_eq!(interpret("Binomial[-5, -2]").unwrap(), "0");
  }

  #[test]
  fn zero_zero() {
    assert_eq!(interpret("Binomial[0, 0]").unwrap(), "1");
  }

  #[test]
  fn choose_one() {
    assert_eq!(interpret("Binomial[7, 1]").unwrap(), "7");
  }

  #[test]
  fn large_n() {
    assert_eq!(interpret("Binomial[20, 10]").unwrap(), "184756");
  }

  #[test]
  fn rational_first_arg() {
    assert_eq!(interpret("Binomial[1/2, 3]").unwrap(), "1/16");
    assert_eq!(interpret("Binomial[1/2, 4]").unwrap(), "-5/128");
    assert_eq!(interpret("Binomial[3/2, 2]").unwrap(), "3/8");
  }

  #[test]
  fn symbolic_first_arg() {
    assert_eq!(
      interpret("Binomial[n, 3]").unwrap(),
      "((-2 + n)*(-1 + n)*n)/6"
    );
  }
}

mod bernstein_basis {
  use super::*;

  #[test]
  fn numeric_half() {
    // BernsteinBasis[4, 3, 0.5] = Binomial[4,3]*0.5^3*0.5^1 = 4*1/16 = 0.25
    assert_eq!(interpret("BernsteinBasis[4, 3, 0.5]").unwrap(), "0.25");
  }

  #[test]
  fn n_equals_zero() {
    // BernsteinBasis[4, 0, x] = (1-x)^4 — but we only evaluate numerically.
    assert_eq!(interpret("BernsteinBasis[4, 0, 0.5]").unwrap(), "0.0625");
  }

  #[test]
  fn out_of_range_stays_unevaluated() {
    // n < 0 or n > d → Wolfram emits a message and keeps the expression
    // unevaluated.
    assert_eq!(
      interpret("BernsteinBasis[3, 4, 0.5]").unwrap(),
      "BernsteinBasis[3, 4, 0.5]"
    );
    assert_eq!(
      interpret("BernsteinBasis[3, -1, 2]").unwrap(),
      "BernsteinBasis[3, -1, 2]"
    );
  }

  #[test]
  fn symbolic_stays_unevaluated() {
    // With symbolic x, we leave unevaluated (matches wolframscript).
    assert_eq!(
      interpret("BernsteinBasis[3, 1, x]").unwrap(),
      "BernsteinBasis[3, 1, x]"
    );
  }
}

mod multinomial {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Multinomial[2, 3, 4]").unwrap(), "1260");
  }

  #[test]
  fn two_args() {
    // Multinomial[a, b] = Binomial[a+b, a]
    assert_eq!(interpret("Multinomial[3, 4]").unwrap(), "35");
  }

  #[test]
  fn single_arg() {
    assert_eq!(interpret("Multinomial[5]").unwrap(), "1");
  }

  #[test]
  fn empty() {
    assert_eq!(interpret("Multinomial[]").unwrap(), "1");
  }

  #[test]
  fn all_ones() {
    // Multinomial[1,1,1] = 3! / (1!*1!*1!) = 6
    assert_eq!(interpret("Multinomial[1, 1, 1]").unwrap(), "6");
  }

  // Negative integer arguments are handled by the cumulative binomial
  // product rather than erroring; e.g. a (-1)! pole in the denominator
  // sends the result to 0, but other cases stay finite.
  #[test]
  fn negative_integer_args() {
    assert_eq!(interpret("Multinomial[1, 2, -1]").unwrap(), "0");
    assert_eq!(interpret("Multinomial[-1, 1]").unwrap(), "0");
    assert_eq!(interpret("Multinomial[-2, 5]").unwrap(), "0");
    assert_eq!(interpret("Multinomial[-3, 1]").unwrap(), "-2");
    assert_eq!(interpret("Multinomial[1, 1, -5]").unwrap(), "12");
    assert_eq!(interpret("Multinomial[0, -1]").unwrap(), "1");
  }

  #[test]
  fn two_symbolic_args() {
    // Multinomial[a, b] = Binomial[a + b, b]
    assert_eq!(
      interpret("Multinomial[a, b]").unwrap(),
      "Binomial[a + b, b]"
    );
  }

  #[test]
  fn two_args_one_numeric() {
    // Multinomial[3, x] = Binomial[3 + x, x]
    assert_eq!(
      interpret("Multinomial[3, x]").unwrap(),
      "Binomial[3 + x, x]"
    );
  }

  #[test]
  fn three_symbolic_args_stay_unevaluated() {
    // Higher-arity symbolic forms stay as Multinomial (matches wolframscript).
    assert_eq!(
      interpret("Multinomial[a, b, c]").unwrap(),
      "Multinomial[a, b, c]"
    );
  }

  #[test]
  fn rational_two_args_exact() {
    // Multinomial[3, 1/2] = Binomial[7/2, 3] = 35/16 (exact, not 2.1875).
    assert_eq!(interpret("Multinomial[3, 1/2]").unwrap(), "35/16");
  }

  #[test]
  fn one_symbolic_two_integers() {
    // Multinomial[3, 4, c] = Binomial[c+7, c] * Multinomial[3, 4]
    //                     = Binomial[7+c, c] * 35
    assert_eq!(
      interpret("Multinomial[3, 4, c]").unwrap(),
      "35*Binomial[7 + c, c]"
    );
  }

  #[test]
  fn one_symbolic_with_rational() {
    // Multinomial[x, 3, 1/2] = Binomial[x+7/2, x] * Multinomial[3, 1/2]
    //                       = Binomial[x+7/2, x] * 35/16
    assert_eq!(
      interpret("Multinomial[x, 3, 1/2]").unwrap(),
      "(35*Binomial[7/2 + x, x])/16"
    );
  }

  #[test]
  fn two_symbolic_with_integer_stays_unevaluated() {
    // Orderless: args are canonically sorted (integers before symbols).
    assert_eq!(
      interpret("Multinomial[a, 3, b]").unwrap(),
      "Multinomial[3, a, b]"
    );
  }
}

mod power_mod {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("PowerMod[2, 10, 1000]").unwrap(), "24");
  }

  #[test]
  fn modular_inverse() {
    assert_eq!(interpret("PowerMod[3, -1, 7]").unwrap(), "5");
  }

  #[test]
  fn large_exponent() {
    assert_eq!(interpret("PowerMod[2, 100, 13]").unwrap(), "3");
  }

  #[test]
  fn base_zero() {
    assert_eq!(interpret("PowerMod[0, 5, 7]").unwrap(), "0");
  }

  #[test]
  fn exponent_zero() {
    assert_eq!(interpret("PowerMod[5, 0, 7]").unwrap(), "1");
  }

  #[test]
  fn negative_base() {
    assert_eq!(interpret("PowerMod[-2, 3, 7]").unwrap(), "6");
  }

  #[test]
  fn large_modulus() {
    // 2^127 - 1 is a Mersenne prime; by Fermat's little theorem, 2^(p-1) ≡ 1 (mod p)
    // so 2^p ≡ 2 (mod p), meaning PowerMod[2, 2^127-1, 2^127-1] = 2
    assert_eq!(interpret("PowerMod[2, 2^127 - 1, 2^127 - 1]").unwrap(), "2");
  }

  // PowerMod[a, 1/2, m] is the smallest nonnegative modular square root.
  #[test]
  fn sqrt_prime_modulus() {
    assert_eq!(interpret("PowerMod[3, 1/2, 11]").unwrap(), "5");
    assert_eq!(interpret("PowerMod[2, 1/2, 7]").unwrap(), "3");
    assert_eq!(interpret("PowerMod[5, 1/2, 11]").unwrap(), "4");
    assert_eq!(interpret("PowerMod[2, 1/2, 17]").unwrap(), "6");
  }

  #[test]
  fn sqrt_composite_modulus() {
    // 4 distinct roots {2, 5, 16, 19}; the smallest is returned.
    assert_eq!(interpret("PowerMod[4, 1/2, 21]").unwrap(), "2");
    assert_eq!(interpret("PowerMod[9, 1/2, 16]").unwrap(), "3");
    assert_eq!(interpret("PowerMod[1, 1/2, 24]").unwrap(), "1");
  }

  #[test]
  fn sqrt_reduces_base_mod_m() {
    // 16 ≡ 2 (mod 7); -1 ≡ 4 (mod 5).
    assert_eq!(interpret("PowerMod[16, 1/2, 7]").unwrap(), "3");
    assert_eq!(interpret("PowerMod[-1, 1/2, 5]").unwrap(), "2");
  }

  #[test]
  fn sqrt_base_zero() {
    assert_eq!(interpret("PowerMod[0, 1/2, 12]").unwrap(), "0");
  }

  #[test]
  fn sqrt_no_solution_stays_unevaluated() {
    // 3 is a quadratic non-residue mod 7.
    assert_eq!(
      interpret("PowerMod[3, 1/2, 7]").unwrap(),
      "PowerMod[3, 1/2, 7]"
    );
  }
}

mod prime_pi {
  use super::*;

  #[test]
  fn small_values() {
    assert_eq!(interpret("PrimePi[1]").unwrap(), "0");
    assert_eq!(interpret("PrimePi[2]").unwrap(), "1");
    assert_eq!(interpret("PrimePi[10]").unwrap(), "4");
    assert_eq!(interpret("PrimePi[100]").unwrap(), "25");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("PrimePi[0]").unwrap(), "0");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("PrimePi[-5]").unwrap(), "0");
  }

  #[test]
  fn at_prime() {
    // PrimePi[7] = 4 (primes: 2, 3, 5, 7)
    assert_eq!(interpret("PrimePi[7]").unwrap(), "4");
  }

  #[test]
  fn symbolic_constants() {
    // PrimePi[E] = 1 (only prime 2 <= E ≈ 2.718)
    assert_eq!(interpret("PrimePi[E]").unwrap(), "1");
    // PrimePi[Pi] = 2 (primes 2, 3 <= Pi ≈ 3.14159)
    assert_eq!(interpret("PrimePi[Pi]").unwrap(), "2");
  }
}

mod next_prime {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("NextPrime[7]").unwrap(), "11");
  }

  #[test]
  fn from_composite() {
    assert_eq!(interpret("NextPrime[10]").unwrap(), "11");
  }

  #[test]
  fn from_one() {
    assert_eq!(interpret("NextPrime[1]").unwrap(), "2");
  }

  #[test]
  fn from_two() {
    assert_eq!(interpret("NextPrime[2]").unwrap(), "3");
  }

  #[test]
  fn from_zero() {
    assert_eq!(interpret("NextPrime[0]").unwrap(), "2");
  }

  #[test]
  fn from_negative() {
    assert_eq!(interpret("NextPrime[-5]").unwrap(), "-3");
  }

  #[test]
  fn big_integer_10_pow_100() {
    assert_eq!(
      interpret("NextPrime[10^100]").unwrap(),
      "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000267"
    );
  }

  #[test]
  fn big_integer_10_pow_50() {
    assert_eq!(
      interpret("NextPrime[10^50]").unwrap(),
      "100000000000000000000000000000000000000000000000151"
    );
  }

  #[test]
  fn big_integer_result_is_prime() {
    assert_eq!(interpret("PrimeQ[NextPrime[10^100]]").unwrap(), "True");
  }

  #[test]
  fn prime_q_big_integer() {
    assert_eq!(interpret("PrimeQ[10^100]").unwrap(), "False");
  }

  #[test]
  fn prime_q_big_prime() {
    assert_eq!(
        interpret("PrimeQ[10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000267]").unwrap(),
        "True"
      );
  }

  #[test]
  fn prime_q_large_integer_mersenne() {
    // 2^127 - 1 fits in i128 but is too large for trial division
    assert_eq!(interpret("PrimeQ[2^127 - 1]").unwrap(), "True");
  }

  #[test]
  fn prime_q_large_integer_composite() {
    // 2^67 - 1 = 193707721 * 761838257287 — fits in i128 but is composite
    assert_eq!(interpret("PrimeQ[2^67 - 1]").unwrap(), "False");
  }

  #[test]
  fn prime_q_negative_integers() {
    // In Wolfram Language, PrimeQ tests the absolute value
    assert_eq!(interpret("PrimeQ[-7]").unwrap(), "True");
    assert_eq!(interpret("PrimeQ[-2]").unwrap(), "True");
    assert_eq!(interpret("PrimeQ[-1]").unwrap(), "False");
    assert_eq!(interpret("PrimeQ[-4]").unwrap(), "False");
    assert_eq!(interpret("PrimeQ[-100]").unwrap(), "False");
    assert_eq!(interpret("PrimeQ[0]").unwrap(), "False");
  }

  #[test]
  fn two_arg_positive_k() {
    assert_eq!(interpret("NextPrime[10, 2]").unwrap(), "13");
    assert_eq!(interpret("NextPrime[7, 1]").unwrap(), "11");
    assert_eq!(interpret("NextPrime[1, 1]").unwrap(), "2");
  }

  #[test]
  fn two_arg_negative_k() {
    assert_eq!(interpret("NextPrime[10, -1]").unwrap(), "7");
    assert_eq!(interpret("NextPrime[10, -2]").unwrap(), "5");
    assert_eq!(interpret("NextPrime[3, -1]").unwrap(), "2");
  }

  #[test]
  fn two_arg_crossing_zero() {
    assert_eq!(interpret("NextPrime[2, -1]").unwrap(), "-2");
    assert_eq!(interpret("NextPrime[1, -1]").unwrap(), "-2");
    assert_eq!(interpret("NextPrime[-2, 1]").unwrap(), "2");
    assert_eq!(interpret("NextPrime[-2, -1]").unwrap(), "-3");
    assert_eq!(interpret("NextPrime[-3, -1]").unwrap(), "-5");
  }
}

mod modular_inverse {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("ModularInverse[3, 7]").unwrap(), "5");
  }

  #[test]
  fn negative_base() {
    assert_eq!(interpret("ModularInverse[-3, 7]").unwrap(), "2");
  }

  #[test]
  fn not_invertible() {
    assert_eq!(
      interpret("ModularInverse[2, 10]").unwrap(),
      "ModularInverse[2, 10]"
    );
    assert_eq!(
      interpret("ModularInverse[0, 5]").unwrap(),
      "ModularInverse[0, 5]"
    );
  }

  #[test]
  fn various() {
    assert_eq!(interpret("ModularInverse[2, 7]").unwrap(), "4");
    assert_eq!(interpret("ModularInverse[1, 100]").unwrap(), "1");
    assert_eq!(interpret("ModularInverse[17, 19]").unwrap(), "9");
  }
}

mod multiplicative_order {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("MultiplicativeOrder[2, 7]").unwrap(), "3");
    assert_eq!(interpret("MultiplicativeOrder[3, 10]").unwrap(), "4");
    assert_eq!(interpret("MultiplicativeOrder[2, 9]").unwrap(), "6");
  }

  // Modulo 1 the ring is trivial, so the order is 1 for any a.
  #[test]
  fn modulus_one() {
    assert_eq!(interpret("MultiplicativeOrder[3, 1]").unwrap(), "1");
    assert_eq!(interpret("MultiplicativeOrder[1, 1]").unwrap(), "1");
    assert_eq!(interpret("MultiplicativeOrder[0, 1]").unwrap(), "1");
    assert_eq!(interpret("MultiplicativeOrder[-3, 1]").unwrap(), "1");
    assert_eq!(interpret("MultiplicativeOrder[100, 1]").unwrap(), "1");
  }

  // Not coprime / zero base → no order exists, stays unevaluated.
  #[test]
  fn non_coprime_unevaluated() {
    assert_eq!(
      interpret("MultiplicativeOrder[4, 6]").unwrap(),
      "MultiplicativeOrder[4, 6]"
    );
    assert_eq!(
      interpret("MultiplicativeOrder[0, 5]").unwrap(),
      "MultiplicativeOrder[0, 5]"
    );
  }

  #[test]
  fn negative_base() {
    assert_eq!(interpret("MultiplicativeOrder[-1, 5]").unwrap(), "2");
  }
}

mod bit_length {
  use super::*;

  #[test]
  fn basic_values() {
    assert_eq!(interpret("BitLength[0]").unwrap(), "0");
    assert_eq!(interpret("BitLength[1]").unwrap(), "1");
    assert_eq!(interpret("BitLength[2]").unwrap(), "2");
    assert_eq!(interpret("BitLength[7]").unwrap(), "3");
    assert_eq!(interpret("BitLength[8]").unwrap(), "4");
    assert_eq!(interpret("BitLength[255]").unwrap(), "8");
    assert_eq!(interpret("BitLength[256]").unwrap(), "9");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("BitLength[-1]").unwrap(), "0");
    assert_eq!(interpret("BitLength[-255]").unwrap(), "8");
  }
}

mod bitwise_ops {
  use super::*;

  #[test]
  fn bit_and() {
    assert_eq!(interpret("BitAnd[5, 3]").unwrap(), "1");
    assert_eq!(interpret("BitAnd[15, 6, 3]").unwrap(), "2");
  }

  #[test]
  fn bit_or() {
    assert_eq!(interpret("BitOr[5, 3]").unwrap(), "7");
    assert_eq!(interpret("BitOr[1, 2, 4]").unwrap(), "7");
  }

  #[test]
  fn bit_xor() {
    assert_eq!(interpret("BitXor[5, 3]").unwrap(), "6");
    assert_eq!(interpret("BitXor[7, 3]").unwrap(), "4");
  }

  #[test]
  fn empty_returns_identity_element() {
    // BitAnd[]/BitOr[]/BitXor[] return their identity element.
    assert_eq!(interpret("BitAnd[]").unwrap(), "-1");
    assert_eq!(interpret("BitOr[]").unwrap(), "0");
    assert_eq!(interpret("BitXor[]").unwrap(), "0");
  }

  #[test]
  fn single_argument_is_one_identity() {
    // OneIdentity: a single argument returns itself, numeric or symbolic.
    assert_eq!(interpret("BitAnd[12]").unwrap(), "12");
    assert_eq!(interpret("BitAnd[x]").unwrap(), "x");
    assert_eq!(interpret("BitOr[x]").unwrap(), "x");
    assert_eq!(interpret("BitXor[a + b]").unwrap(), "a + b");
  }

  #[test]
  fn bit_not() {
    assert_eq!(interpret("BitNot[5]").unwrap(), "-6");
    assert_eq!(interpret("BitNot[0]").unwrap(), "-1");
  }

  #[test]
  fn bit_shift_right() {
    assert_eq!(interpret("BitShiftRight[8, 2]").unwrap(), "2");
    assert_eq!(interpret("BitShiftRight[255, 4]").unwrap(), "15");
    assert_eq!(interpret("BitShiftRight[1024]").unwrap(), "512");
    assert_eq!(interpret("BitShiftRight[-8, 2]").unwrap(), "-2");
    assert_eq!(
      interpret("BitShiftRight[x, 2]").unwrap(),
      "BitShiftRight[x, 2]"
    );
  }

  #[test]
  fn bit_shift_left() {
    assert_eq!(interpret("BitShiftLeft[1, 4]").unwrap(), "16");
    assert_eq!(interpret("BitShiftLeft[3]").unwrap(), "6");
    assert_eq!(interpret("BitShiftLeft[5, 0]").unwrap(), "5");
  }

  #[test]
  fn bit_ops_thread_over_lists() {
    // Bit{And,Or,Xor,Not,ShiftRight,ShiftLeft,Length} are all Listable.
    assert_eq!(
      interpret("BitShiftRight[{8, 16, 32}, 2]").unwrap(),
      "{2, 4, 8}"
    );
    assert_eq!(
      interpret("BitAnd[{1, 5, 12}, {3, 6, 9}]").unwrap(),
      "{1, 4, 8}"
    );
    assert_eq!(interpret("BitNot[{0, 1, 2}]").unwrap(), "{-1, -2, -3}");
  }
}

mod gcd {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("GCD[12, 8]").unwrap(), "4");
  }

  #[test]
  fn coprime() {
    assert_eq!(interpret("GCD[17, 19]").unwrap(), "1");
  }

  #[test]
  fn with_zero() {
    assert_eq!(interpret("GCD[0, 5]").unwrap(), "5");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("GCD[-12, 8]").unwrap(), "4");
  }

  #[test]
  fn multiple_args() {
    assert_eq!(interpret("GCD[24, 36, 60]").unwrap(), "12");
  }

  #[test]
  fn empty() {
    assert_eq!(interpret("GCD[]").unwrap(), "0");
  }

  #[test]
  fn big_integer() {
    assert_eq!(interpret("GCD[2^200, 123436216212]").unwrap(), "4");
  }

  #[test]
  fn big_integer_both() {
    assert_eq!(
      interpret("GCD[2^100, 2^150]").unwrap(),
      interpret("2^100").unwrap()
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("GCD[x, 5]").unwrap(), "GCD[5, x]");
  }

  #[test]
  fn rationals() {
    // GCD[a/b, c/d] = GCD[a, c] / LCM[b, d]
    assert_eq!(interpret("GCD[1/2, 1/3]").unwrap(), "1/6");
    assert_eq!(interpret("GCD[2/3, 4/9]").unwrap(), "2/9");
  }

  #[test]
  fn three_rationals() {
    assert_eq!(interpret("GCD[3/4, 5/6, 1/2]").unwrap(), "1/12");
  }

  #[test]
  fn mixed_integer_and_rational() {
    // 2 = 2/1, so GCD[2, 1/3] = GCD[2, 1] / LCM[1, 3] = 1/3
    assert_eq!(interpret("GCD[2, 1/3]").unwrap(), "1/3");
  }

  #[test]
  fn lcm_no_args_warns_argm() {
    use woxi::interpret_with_stdout;
    // LCM requires at least one argument: wolframscript emits LCM::argm and
    // returns unevaluated. GCD[] is the identity 0 and warns nothing.
    let l = interpret_with_stdout("LCM[]").unwrap();
    assert_eq!(l.result, "LCM[]");
    assert!(l.warnings[0].contains(
      "LCM::argm: LCM called with 0 arguments; 1 or more arguments are expected."
    ));
    let g = interpret_with_stdout("GCD[]").unwrap();
    assert_eq!(g.result, "0");
    assert!(g.warnings.is_empty());
  }

  #[test]
  fn inexact_argument_warns() {
    use woxi::interpret_with_stdout;
    // A Real argument keeps GCD unevaluated and warns ::exact, naming the
    // first inexact argument in the Orderless-sorted form.
    let r = interpret_with_stdout("GCD[12.0, 8]").unwrap();
    assert_eq!(r.result, "GCD[8, 12.]");
    assert!(r.warnings[0].contains(
      "GCD::exact: Argument 12. in GCD[8, 12.] is not an exact number."
    ));
    // LCM behaves the same way.
    let l = interpret_with_stdout("LCM[2.5, 5]").unwrap();
    assert_eq!(l.result, "LCM[2.5, 5]");
    assert!(l.warnings[0].contains(
      "LCM::exact: Argument 2.5 in LCM[2.5, 5] is not an exact number."
    ));
    // A purely symbolic argument stays unevaluated WITHOUT a warning.
    let s = interpret_with_stdout("GCD[Pi, 2]").unwrap();
    assert_eq!(s.result, "GCD[2, Pi]");
    assert!(s.warnings.is_empty());
  }
}

mod extended_gcd {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("ExtendedGCD[6, 10]").unwrap(), "{2, {2, -1}}");
  }

  #[test]
  fn coprime() {
    assert_eq!(interpret("ExtendedGCD[2, 3]").unwrap(), "{1, {-1, 1}}");
  }

  #[test]
  fn same_divisor() {
    assert_eq!(interpret("ExtendedGCD[12, 8]").unwrap(), "{4, {1, -1}}");
  }

  #[test]
  fn with_zero() {
    assert_eq!(interpret("ExtendedGCD[0, 5]").unwrap(), "{5, {0, 1}}");
  }

  #[test]
  fn both_zero() {
    assert_eq!(interpret("ExtendedGCD[0, 0]").unwrap(), "{0, {0, 0}}");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("ExtendedGCD[-6, 10]").unwrap(), "{2, {-2, -1}}");
  }

  #[test]
  fn three_args() {
    // Verify: 6*(-14) + 10*7 + 15*1 = -84 + 70 + 15 = 1
    let result = interpret("ExtendedGCD[6, 10, 15]").unwrap();
    assert!(result.starts_with("{1, {"));
  }

  #[test]
  fn bezout_identity() {
    // Verify a*s + b*t == gcd
    assert_eq!(
      interpret("Module[{r = ExtendedGCD[6, 10]}, 6*r[[2, 1]] + 10*r[[2, 2]]]")
        .unwrap(),
      "2"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("ExtendedGCD[x, 5]").unwrap(), "ExtendedGCD[x, 5]");
  }
}

mod lcm {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("LCM[4, 6]").unwrap(), "12");
  }

  #[test]
  fn with_zero() {
    assert_eq!(interpret("LCM[0, 5]").unwrap(), "0");
  }

  #[test]
  fn multiple_args() {
    assert_eq!(interpret("LCM[2, 3, 5]").unwrap(), "30");
  }

  #[test]
  fn big_integer() {
    assert_eq!(
      interpret("LCM[2^200, 123436216212]").unwrap(),
      "49588587967610287244150772077232914068094816317781627045866288529276928"
    );
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("LCM[-4, 6]").unwrap(), "12");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("LCM[x, 5]").unwrap(), "LCM[5, x]");
  }

  #[test]
  fn rationals() {
    // LCM[a/b, c/d] = LCM[a, c] / GCD[b, d]
    assert_eq!(interpret("LCM[1/2, 1/3]").unwrap(), "1");
    assert_eq!(interpret("LCM[2/3, 4/9]").unwrap(), "4/3");
  }

  #[test]
  fn mixed_integer_and_rational() {
    // 6 = 6/1, LCM[6, 2/3] = LCM[6, 2] / GCD[1, 3] = 6/1 = 6
    assert_eq!(interpret("LCM[6, 2/3]").unwrap(), "6");
  }
}

mod jacobi_symbol {
  use super::*;

  #[test]
  fn basic_legendre() {
    assert_eq!(interpret("JacobiSymbol[1, 5]").unwrap(), "1");
    assert_eq!(interpret("JacobiSymbol[2, 5]").unwrap(), "-1");
    assert_eq!(interpret("JacobiSymbol[3, 5]").unwrap(), "-1");
    assert_eq!(interpret("JacobiSymbol[4, 5]").unwrap(), "1");
  }

  #[test]
  fn zero_numerator() {
    assert_eq!(interpret("JacobiSymbol[0, 5]").unwrap(), "0");
    assert_eq!(interpret("JacobiSymbol[0, 1]").unwrap(), "1");
  }

  #[test]
  fn denominator_one() {
    assert_eq!(interpret("JacobiSymbol[7, 1]").unwrap(), "1");
    assert_eq!(interpret("JacobiSymbol[0, 1]").unwrap(), "1");
  }

  #[test]
  fn negative_numerator() {
    // (-1/p) = (-1)^((p-1)/2)
    assert_eq!(interpret("JacobiSymbol[-1, 3]").unwrap(), "-1");
    assert_eq!(interpret("JacobiSymbol[-1, 5]").unwrap(), "1");
    assert_eq!(interpret("JacobiSymbol[-1, 7]").unwrap(), "-1");
  }

  #[test]
  fn composite_denominator() {
    assert_eq!(interpret("JacobiSymbol[2, 15]").unwrap(), "1");
    assert_eq!(interpret("JacobiSymbol[7, 15]").unwrap(), "-1");
  }

  #[test]
  fn large_values() {
    assert_eq!(interpret("JacobiSymbol[1001, 9907]").unwrap(), "-1");
  }

  #[test]
  fn table_mod_5() {
    // Complete table of Jacobi symbols for (n/5), n=0..4
    assert_eq!(
      interpret("Table[JacobiSymbol[n, 5], {n, 0, 4}]").unwrap(),
      "{0, 1, -1, -1, 1}"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("JacobiSymbol[x, 5]").unwrap(),
      "JacobiSymbol[x, 5]"
    );
  }
}

mod real_digits {
  use super::*;

  #[test]
  fn pi_20_digits() {
    assert_eq!(
      interpret("RealDigits[Pi, 10, 20]").unwrap(),
      "{{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4}, 1}"
    );
  }

  #[test]
  fn pi_base_260_5_digits() {
    // Regression for mathics test_realdigits (atomic/test_numbers.py:70).
    // Non-decimal base on Pi must route through the arbitrary-base path.
    assert_eq!(
      interpret("RealDigits[Pi, 260, 5]").unwrap(),
      "{{3, 36, 211, 172, 124}, 1}"
    );
  }

  #[test]
  fn pi_base_2_and_16() {
    assert_eq!(
      interpret("RealDigits[Pi, 2, 10]").unwrap(),
      "{{1, 1, 0, 0, 1, 0, 0, 1, 0, 0}, 2}"
    );
    assert_eq!(
      interpret("RealDigits[Pi, 16, 8]").unwrap(),
      "{{3, 2, 4, 3, 15, 6, 10, 8}, 1}"
    );
  }

  #[test]
  fn non_numeric_symbol_stays_unevaluated() {
    // Bare symbols can't be converted to digits; return unevaluated
    // (matches wolframscript).
    assert_eq!(interpret("RealDigits[abc]").unwrap(), "RealDigits[abc]");
  }

  #[test]
  fn real_in_non_decimal_base_is_padded() {
    // RealDigits[123.45, 40] should produce ~10 base-40 digits of 123.45,
    // matching wolframscript byte-for-byte.
    assert_eq!(
      interpret("RealDigits[123.45, 40]").unwrap(),
      "{{3, 3, 18, 0, 0, 0, 0, 0, 0, 0}, 2}"
    );
  }

  #[test]
  fn real_base_10_uses_decimal_literal() {
    // RealDigits[123.55555] must reflect the decimal the user typed and pad
    // with zeros, not expose the f64 rounding tail (…5,4,9,9,9,…). Regression
    // for mathics atomic/numbers.py:359.
    assert_eq!(
      interpret("RealDigits[123.55555]").unwrap(),
      "{{1, 2, 3, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0}, 3}"
    );
  }

  #[test]
  fn four_argument_start_position() {
    // RealDigits[Pi, 10, 11, -3] returns 11 digits starting at the 10^-3
    // place. Regression for mathics atomic/numbers.py:371.
    assert_eq!(
      interpret("RealDigits[Pi, 10, 11, -3]").unwrap(),
      "{{1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7}, -2}"
    );
  }

  #[test]
  fn four_argument_deep_position() {
    // The 500th decimal digit of Pi is 2.
    assert_eq!(
      interpret("RealDigits[Pi, 10, 1, -500]").unwrap(),
      "{{2}, -499}"
    );
  }

  #[test]
  fn real_digits_beyond_machine_precision_are_indeterminate() {
    // A machine-precision Real carries ~16 decimal digits; anything past that
    // is unknowable and must be reported as Indeterminate. Regression for
    // mathics atomic/numbers.py:375.
    assert_eq!(
      interpret("RealDigits[123.45, 10, 18]").unwrap(),
      "{{1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Indeterminate, \
       Indeterminate}, 3}"
    );
  }

  #[test]
  fn real_digits_base2_exact_dyadic_pads_with_zero() {
    // A machine Real carries ~53 bits, i.e. 53 base-2 digits. 23.34375 is an
    // exact dyadic (10111.01011), so the remaining digits are genuinely 0
    // (not Indeterminate), and the point sits after 5 digits. Regression for
    // RealDigits padding exact values with Indeterminate in non-decimal bases.
    assert_eq!(
      interpret("RealDigits[23.34375, 2]").unwrap(),
      "{{1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
       0, 0, 0, 0, 0, 0, 0}, 5}"
    );
  }

  #[test]
  fn integer_ten() {
    // RealDigits[10] yields digits {1, 0} with exponent 2.
    assert_eq!(interpret("RealDigits[10]").unwrap(), "{{1, 0}, 2}");
  }

  #[test]
  fn one_over_197_base_260_5_digits() {
    // First 5 base-260 digits of 1/197 with zero integer exponent.
    assert_eq!(
      interpret("RealDigits[1/197, 260, 5]").unwrap(),
      "{{1, 83, 38, 71, 69}, 0}"
    );
  }

  // Regression (mathics test_numbers.py:102): `RealDigits[x, b, len, p]`
  // returns digits at positions p, p-1, …, p-len+1 (exp = p + 1). The
  // rational long-division path previously ignored the `p` argument.
  #[test]
  fn one_over_197_base_260_5_digits_start_pos_neg_6() {
    assert_eq!(
      interpret("RealDigits[1/197, 260, 5, -6]").unwrap(),
      "{{246, 208, 137, 67, 80}, -5}"
    );
  }

  #[test]
  fn one_over_197_base_260_3_digits_start_pos_neg_2() {
    assert_eq!(
      interpret("RealDigits[1/197, 260, 3, -2]").unwrap(),
      "{{83, 38, 71}, -1}"
    );
  }

  #[test]
  fn one_over_197_base_260_start_pos_zero_pads_msd() {
    // start_pos = 0 includes the (zero) integer-part digit at the front.
    assert_eq!(
      interpret("RealDigits[1/197, 260, 3, 0]").unwrap(),
      "{{0, 1, 83}, 1}"
    );
  }

  #[test]
  fn one_over_197_base_260_start_pos_above_msd_zeros() {
    // start_pos = 2 is two positions above the MSD; all returned digits
    // are zero.
    assert_eq!(
      interpret("RealDigits[1/197, 260, 3, 2]").unwrap(),
      "{{0, 0, 0}, 3}"
    );
  }

  #[test]
  fn pi_part_extraction() {
    assert_eq!(
      interpret("RealDigits[Pi, 10, 10][[1]]").unwrap(),
      "{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}"
    );
  }

  #[test]
  fn pi_exponent() {
    assert_eq!(interpret("RealDigits[Pi, 10, 5][[2]]").unwrap(), "1");
  }

  #[test]
  fn e_constant() {
    assert_eq!(
      interpret("RealDigits[E, 10, 10]").unwrap(),
      "{{2, 7, 1, 8, 2, 8, 1, 8, 2, 8}, 1}"
    );
  }

  #[test]
  fn integer_value() {
    assert_eq!(
      interpret("RealDigits[42, 10, 5]").unwrap(),
      "{{4, 2, 0, 0, 0}, 2}"
    );
  }

  #[test]
  fn round_integer() {
    assert_eq!(
      interpret("RealDigits[100, 10, 5]").unwrap(),
      "{{1, 0, 0, 0, 0}, 3}"
    );
  }

  #[test]
  fn rational() {
    assert_eq!(
      interpret("RealDigits[1/7, 10, 12]").unwrap(),
      "{{1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7}, 0}"
    );
  }

  #[test]
  fn one_third() {
    assert_eq!(
      interpret("RealDigits[1/3, 10, 10]").unwrap(),
      "{{3, 3, 3, 3, 3, 3, 3, 3, 3, 3}, 0}"
    );
  }

  #[test]
  fn zero() {
    assert_eq!(
      interpret("RealDigits[0, 10, 5]").unwrap(),
      "{{0, 0, 0, 0, 0}, 0}"
    );
  }

  #[test]
  fn negative_number() {
    // RealDigits uses absolute value for digits
    assert_eq!(
      interpret("RealDigits[-Pi, 10, 5]").unwrap(),
      "{{3, 1, 4, 1, 5}, 1}"
    );
  }

  #[test]
  fn small_number() {
    assert_eq!(
      interpret("RealDigits[345/100000, 10, 5]").unwrap(),
      "{{3, 4, 5, 0, 0}, -2}"
    );
  }

  #[test]
  fn large_precision() {
    // Test 1000 digits of Pi - verify first and last few digits
    assert_eq!(
      interpret("Take[RealDigits[Pi, 10, 1000][[1]], 5]").unwrap(),
      "{3, 1, 4, 1, 5}"
    );
    assert_eq!(
      interpret("Take[RealDigits[Pi, 10, 1000][[1]], -5]").unwrap(),
      "{2, 0, 1, 9, 8}"
    );
  }

  #[test]
  fn rational_repeating_one_seventh() {
    // 1/7 = 0.142857142857... (all repeating)
    assert_eq!(
      interpret("RealDigits[1/7]").unwrap(),
      "{{{1, 4, 2, 8, 5, 7}}, 0}"
    );
  }

  #[test]
  fn rational_repeating_one_third() {
    // 1/3 = 0.333... (single repeating digit)
    assert_eq!(interpret("RealDigits[1/3]").unwrap(), "{{{3}}, 0}");
  }

  #[test]
  fn rational_mixed_repeating() {
    // 7/12 = 0.58333... (non-repeating "58" then repeating "3")
    assert_eq!(interpret("RealDigits[7/12]").unwrap(), "{{5, 8, {3}}, 0}");
  }

  #[test]
  fn rational_mixed_repeating_one_sixth() {
    // 1/6 = 0.1666... (non-repeating "1" then repeating "6")
    assert_eq!(interpret("RealDigits[1/6]").unwrap(), "{{1, {6}}, 0}");
  }

  #[test]
  fn rational_terminating() {
    // 1/4 = 0.25 (terminating decimal)
    assert_eq!(interpret("RealDigits[1/4]").unwrap(), "{{2, 5}, 0}");
  }

  #[test]
  fn rational_greater_than_one() {
    // 22/7 = 3.142857142857... (integer part 3, then repeating)
    assert_eq!(
      interpret("RealDigits[22/7]").unwrap(),
      "{{3, {1, 4, 2, 8, 5, 7}}, 1}"
    );
  }

  #[test]
  fn rational_with_explicit_digits_no_repeat() {
    // With explicit num_digits, repeating pattern should NOT be detected
    assert_eq!(
      interpret("RealDigits[1/7, 10, 12]").unwrap(),
      "{{1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7}, 0}"
    );
  }
}

mod coprime_q {
  use super::*;

  #[test]
  fn three_coprime() {
    assert_eq!(interpret("CoprimeQ[2, 3, 5]").unwrap(), "True");
  }

  #[test]
  fn three_not_coprime() {
    assert_eq!(interpret("CoprimeQ[2, 4, 5]").unwrap(), "False");
  }

  // Single argument tests whether n is a unit (|n| == 1).
  #[test]
  fn single_argument() {
    assert_eq!(interpret("CoprimeQ[1]").unwrap(), "True");
    assert_eq!(interpret("CoprimeQ[-1]").unwrap(), "True");
    assert_eq!(interpret("CoprimeQ[5]").unwrap(), "False");
    assert_eq!(interpret("CoprimeQ[0]").unwrap(), "False");
    assert_eq!(interpret("CoprimeQ[x]").unwrap(), "False");
  }

  // CoprimeQ is Listable: it threads element-wise over list arguments.
  #[test]
  fn listable_single_list() {
    assert_eq!(interpret("CoprimeQ[{6, 35}]").unwrap(), "{False, False}");
    assert_eq!(
      interpret("CoprimeQ[{1, 2, 3}]").unwrap(),
      "{True, False, False}"
    );
  }

  #[test]
  fn listable_multiple_lists() {
    // Threads pairwise: {CoprimeQ[2,4], CoprimeQ[3,9]}.
    assert_eq!(
      interpret("CoprimeQ[{2, 3}, {4, 9}]").unwrap(),
      "{False, False}"
    );
    // Scalar broadcasts: {CoprimeQ[2,5], CoprimeQ[3,5]}.
    assert_eq!(interpret("CoprimeQ[{2, 3}, 5]").unwrap(), "{True, True}");
    assert_eq!(
      interpret("CoprimeQ[{2, 4}, {3, 9}, {5, 25}]").unwrap(),
      "{True, True}"
    );
  }
}

mod prime_power_q {
  use super::*;

  #[test]
  fn prime_power_9() {
    assert_eq!(interpret("PrimePowerQ[9]").unwrap(), "True");
  }

  #[test]
  fn prime_power_52142() {
    assert_eq!(interpret("PrimePowerQ[52142]").unwrap(), "False");
  }

  #[test]
  fn prime_power_neg8() {
    assert_eq!(interpret("PrimePowerQ[-8]").unwrap(), "True");
  }

  #[test]
  fn prime_power_371293() {
    assert_eq!(interpret("PrimePowerQ[371293]").unwrap(), "True");
  }
}

mod factorial {
  use super::*;

  #[test]
  fn factorial_small() {
    assert_eq!(interpret("Factorial[0]").unwrap(), "1");
    assert_eq!(interpret("Factorial[1]").unwrap(), "1");
    assert_eq!(interpret("Factorial[2]").unwrap(), "2");
    assert_eq!(interpret("Factorial[3]").unwrap(), "6");
    assert_eq!(interpret("Factorial[4]").unwrap(), "24");
    assert_eq!(interpret("Factorial[5]").unwrap(), "120");
  }

  #[test]
  fn factorial_medium() {
    assert_eq!(interpret("Factorial[10]").unwrap(), "3628800");
    assert_eq!(interpret("Factorial[12]").unwrap(), "479001600");
    assert_eq!(interpret("Factorial[20]").unwrap(), "2432902008176640000");
  }

  #[test]
  fn factorial_big() {
    assert_eq!(
      interpret("Factorial[50]").unwrap(),
      "30414093201713378043612608166064768844377641568960512000000000000"
    );
  }

  #[test]
  fn factorial_100() {
    assert_eq!(
      interpret("Factorial[100]").unwrap(),
      "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000"
    );
  }

  #[test]
  fn factorial_500_string_length() {
    // 500! has 1135 decimal digits.
    assert_eq!(
      interpret("500! // ToString // StringLength").unwrap(),
      "1135"
    );
  }

  #[test]
  fn factorial_symbolic() {
    assert_eq!(interpret("Factorial[n]").unwrap(), "n!");
  }

  #[test]
  fn factorial_of_plus_parenthesizes() {
    // `n!` binds only to the immediately preceding atom, so when Factorial's
    // argument is a Plus/Times the whole expression needs parentheses.
    assert_eq!(interpret("(a + b)!").unwrap(), "(a + b)!");
    assert_eq!(interpret("(-1 + z)!").unwrap(), "(-1 + z)!");
  }

  #[test]
  fn factorial_of_times_parenthesizes() {
    assert_eq!(interpret("(a * b)!").unwrap(), "(a*b)!");
  }

  #[test]
  fn factorial_of_unary_minus_parenthesizes() {
    assert_eq!(interpret("(-a)!").unwrap(), "(-a)!");
  }

  #[test]
  fn factorial_negative_integers() {
    // Factorial of negative integers is ComplexInfinity
    assert_eq!(interpret("Factorial[-1]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Factorial[-2]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("Factorial[-5]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("(-1)!").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("(-3)!").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn factorial_half_integer_rational() {
    // Factorial[1/2] = Gamma[3/2] = Sqrt[Pi]/2
    assert_eq!(interpret("Factorial[1/2]").unwrap(), "Sqrt[Pi]/2");
    // Factorial[3/2] = Gamma[5/2] = (3*Sqrt[Pi])/4
    assert_eq!(interpret("Factorial[3/2]").unwrap(), "(3*Sqrt[Pi])/4");
    // Factorial[-1/2] = Gamma[1/2] = Sqrt[Pi]
    assert_eq!(interpret("Factorial[-1/2]").unwrap(), "Sqrt[Pi]");
    // Factorial[-3/2] = Gamma[-1/2] = -2*Sqrt[Pi]
    assert_eq!(interpret("Factorial[-3/2]").unwrap(), "-2*Sqrt[Pi]");
  }

  #[test]
  fn factorial_float() {
    // Factorial[0.5] = Gamma[1.5] ≈ 0.886...
    let result = interpret("Factorial[0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.886226925452758).abs() < 1e-12);
    // Factorial[1.5] = Gamma[2.5] ≈ 1.329...
    let result = interpret("Factorial[1.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.329340388179137).abs() < 1e-12);
    // Factorial[-0.5] = Gamma[0.5] = Sqrt[Pi] ≈ 1.7724...
    let result = interpret("Factorial[-0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.772453850905516).abs() < 1e-12);
  }

  // Prefix Not binds looser than postfix Factorial, so `!a!` must parse as
  // Not[Factorial[a]] rather than Factorial[Not[a]]. wolframscript's REPL
  // preserves the FullForm wrapper for Not, printing `FullForm[ !a!]`
  // (with the leading space disambiguating the postfix Factorial). The
  // bare head form is reachable via `ToString[FullForm[…]]`.
  #[test]
  fn not_binds_looser_than_factorial() {
    assert_eq!(interpret("!a! // FullForm").unwrap(), "FullForm[ !a!]");
    assert_eq!(
      interpret("ToString[FullForm[!a!]]").unwrap(),
      "Not[Factorial[a]]"
    );
  }

  #[test]
  fn series_at_zero_order_0() {
    // x! = 1 + O(x).
    assert_eq!(
      interpret("Series[x!, {x, 0, 0}]").unwrap(),
      "SeriesData[x, 0, {1}, 0, 1, 1]"
    );
  }

  #[test]
  fn series_at_zero_order_1() {
    // x! = 1 - EulerGamma x + O(x^2).
    assert_eq!(
      interpret("Series[x!, {x, 0, 1}]").unwrap(),
      "SeriesData[x, 0, {1, -EulerGamma}, 0, 2, 1]"
    );
  }

  #[test]
  fn series_at_zero_order_2() {
    // wolframscript:
    //   SeriesData[x, 0, {1, -EulerGamma, (6*EulerGamma^2 + Pi^2)/12}, 0, 3, 1]
    // Woxi's Plus canonicalisation reorders to (Pi^2 + 6*EulerGamma^2)/12.
    assert_eq!(
      interpret("Series[x!, {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, -EulerGamma, (Pi^2 + 6*EulerGamma^2)/12}, 0, 3, 1]"
    );
  }
}

mod factorial2 {
  use super::*;

  #[test]
  fn series_at_zero_order_0() {
    // x!! = 1 + O(x).
    assert_eq!(
      interpret("Series[x!!, {x, 0, 0}]").unwrap(),
      "SeriesData[x, 0, {1}, 0, 1, 1]"
    );
  }

  #[test]
  fn series_at_zero_order_1() {
    // x!! = 1 + (-EulerGamma + Log[2])/2 * x + O(x^2).
    assert_eq!(
      interpret("Series[x!!, {x, 0, 1}]").unwrap(),
      "SeriesData[x, 0, {1, (-EulerGamma + Log[2])/2}, 0, 2, 1]"
    );
  }

  #[test]
  fn series_at_zero_order_2() {
    // wolframscript:
    //   SeriesData[x, 0, {1, (-EulerGamma + Log[2])/2,
    //                     (6 (EulerGamma - Log[2])^2 + Pi^2 (1 + Log[64] - 6 Log[Pi]))/48}, 0, 3, 1]
    // Woxi's Plus canonicalisation reorders the numerator terms.
    assert_eq!(
      interpret("Series[x!!, {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {1, (-EulerGamma + Log[2])/2, \
       (Pi^2*(1 + Log[64] - 6*Log[Pi]) + 6*(EulerGamma - Log[2])^2)/48}, 0, 3, 1]"
    );
  }

  #[test]
  fn double_factorial_odd() {
    assert_eq!(interpret("Factorial2[5]").unwrap(), "15");
    assert_eq!(interpret("Factorial2[7]").unwrap(), "105");
  }

  #[test]
  fn double_factorial_postfix_syntax() {
    // 5!! is postfix sugar for Factorial2[5].
    assert_eq!(interpret("5!!").unwrap(), "15");
  }

  #[test]
  fn double_factorial_even() {
    assert_eq!(interpret("Factorial2[6]").unwrap(), "48");
    assert_eq!(interpret("Factorial2[10]").unwrap(), "3840");
  }

  #[test]
  fn double_factorial_edge() {
    assert_eq!(interpret("Factorial2[0]").unwrap(), "1");
    assert_eq!(interpret("Factorial2[-1]").unwrap(), "1");
    assert_eq!(interpret("Factorial2[1]").unwrap(), "1");
  }
}

mod subfactorial {
  use super::*;

  #[test]
  fn subfactorial_basic() {
    assert_eq!(interpret("Subfactorial[0]").unwrap(), "1");
    assert_eq!(interpret("Subfactorial[1]").unwrap(), "0");
    assert_eq!(interpret("Subfactorial[2]").unwrap(), "1");
    assert_eq!(interpret("Subfactorial[3]").unwrap(), "2");
    assert_eq!(interpret("Subfactorial[5]").unwrap(), "44");
  }
}

mod pochhammer {
  use super::*;

  #[test]
  fn pochhammer_basic() {
    assert_eq!(interpret("Pochhammer[4, 8]").unwrap(), "6652800");
    assert_eq!(interpret("Pochhammer[1, 5]").unwrap(), "120");
    assert_eq!(interpret("Pochhammer[3, 0]").unwrap(), "1");
  }

  #[test]
  fn pochhammer_symbolic_zero() {
    // Pochhammer[a, 0] = 1 for any a, even symbolic
    assert_eq!(interpret("Pochhammer[a, 0]").unwrap(), "1");
    assert_eq!(interpret("Pochhammer[x + y, 0]").unwrap(), "1");
  }

  #[test]
  fn pochhammer_symbolic_positive_n() {
    // Pochhammer[a, n] expands for positive integer n
    assert_eq!(interpret("Pochhammer[a, 1]").unwrap(), "a");
    assert_eq!(interpret("Pochhammer[a, 3]").unwrap(), "a*(1 + a)*(2 + a)");
    assert_eq!(
      interpret("Pochhammer[a, 4]").unwrap(),
      "a*(1 + a)*(2 + a)*(3 + a)"
    );
  }

  #[test]
  fn pochhammer_symbolic_negative_n() {
    // Pochhammer[a, -n] = 1/((a-1)(a-2)...(a-n))
    assert_eq!(interpret("Pochhammer[a, -1]").unwrap(), "(-1 + a)^(-1)");
    assert_eq!(
      interpret("Pochhammer[a, -2]").unwrap(),
      "1/((-2 + a)*(-1 + a))"
    );
  }

  #[test]
  fn pochhammer_numeric_negative_n() {
    // Pochhammer[5, -2] = 1/(4*3) = 1/12
    assert_eq!(interpret("Pochhammer[5, -2]").unwrap(), "1/12");
  }

  #[test]
  fn pochhammer_symbolic_n_unevaluated() {
    // When n is also symbolic, stay unevaluated
    assert_eq!(interpret("Pochhammer[a, n]").unwrap(), "Pochhammer[a, n]");
  }

  #[test]
  fn pochhammer_half_series_order_1() {
    // Pochhammer[x, 1/2] = Sqrt[Pi] * x + O(x^2).
    assert_eq!(
      interpret("Series[Pochhammer[x, 1/2], {x, 0, 1}]").unwrap(),
      "SeriesData[x, 0, {Sqrt[Pi]}, 1, 2, 1]"
    );
  }

  #[test]
  fn pochhammer_half_series_order_2() {
    // wolframscript: SeriesData[x, 0, {Sqrt[Pi], -(Sqrt[Pi]*Log[4])}, 1, 3, 1].
    assert_eq!(
      interpret("Series[Pochhammer[x, 1/2], {x, 0, 2}]").unwrap(),
      "SeriesData[x, 0, {Sqrt[Pi], -(Sqrt[Pi]*Log[4])}, 1, 3, 1]"
    );
  }
}

mod bell_b {
  use super::*;

  #[test]
  fn bell_10() {
    assert_eq!(interpret("BellB[10]").unwrap(), "115975");
  }

  #[test]
  fn bell_0() {
    assert_eq!(interpret("BellB[0]").unwrap(), "1");
  }

  #[test]
  fn bell_1() {
    assert_eq!(interpret("BellB[1]").unwrap(), "1");
  }
}

mod partitions_p {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("PartitionsP[0]").unwrap(), "1");
  }

  #[test]
  fn one() {
    assert_eq!(interpret("PartitionsP[1]").unwrap(), "1");
  }

  #[test]
  fn small_values() {
    assert_eq!(
      interpret("Table[PartitionsP[k], {k, 0, 12}]").unwrap(),
      "{1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77}"
    );
  }

  #[test]
  fn fifty() {
    assert_eq!(interpret("PartitionsP[50]").unwrap(), "204226");
  }

  #[test]
  fn hundred() {
    assert_eq!(interpret("PartitionsP[100]").unwrap(), "190569292");
  }

  #[test]
  fn two_hundred() {
    assert_eq!(interpret("PartitionsP[200]").unwrap(), "3972999029388");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("PartitionsP[-1]").unwrap(), "0");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("PartitionsP[x]").unwrap(), "PartitionsP[x]");
  }

  #[test]
  fn non_integer() {
    assert_eq!(interpret("PartitionsP[1.5]").unwrap(), "PartitionsP[1.5]");
  }
}

mod partitions_q {
  use super::*;

  #[test]
  fn base_cases() {
    assert_eq!(interpret("PartitionsQ[0]").unwrap(), "1");
    assert_eq!(interpret("PartitionsQ[1]").unwrap(), "1");
    assert_eq!(interpret("PartitionsQ[2]").unwrap(), "1");
  }

  #[test]
  fn small_values() {
    assert_eq!(interpret("PartitionsQ[3]").unwrap(), "2");
    assert_eq!(interpret("PartitionsQ[4]").unwrap(), "2");
    assert_eq!(interpret("PartitionsQ[5]").unwrap(), "3");
  }

  #[test]
  fn medium_values() {
    assert_eq!(interpret("PartitionsQ[10]").unwrap(), "10");
    assert_eq!(interpret("PartitionsQ[20]").unwrap(), "64");
    assert_eq!(interpret("PartitionsQ[50]").unwrap(), "3658");
  }

  #[test]
  fn large_value() {
    assert_eq!(interpret("PartitionsQ[100]").unwrap(), "444793");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("PartitionsQ[-1]").unwrap(), "0");
    assert_eq!(interpret("PartitionsQ[-10]").unwrap(), "0");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("PartitionsQ[x]").unwrap(), "PartitionsQ[x]");
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[PartitionsQ]").unwrap(),
      "{Listable, Protected}"
    );
  }
}

mod arithmetic_geometric_mean {
  use super::*;

  #[test]
  fn numeric() {
    let result = interpret("ArithmeticGeometricMean[1., 2.]")
      .unwrap()
      .parse::<f64>()
      .unwrap();
    assert!((result - 1.4567910310469068).abs() < 1e-10);
  }

  #[test]
  fn zero_arg() {
    assert_eq!(interpret("ArithmeticGeometricMean[0, 5]").unwrap(), "0");
    assert_eq!(interpret("ArithmeticGeometricMean[5, 0]").unwrap(), "0");
  }

  #[test]
  fn equal_args() {
    assert_eq!(interpret("ArithmeticGeometricMean[3, 3]").unwrap(), "3");
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("ArithmeticGeometricMean[a, b]").unwrap(),
      "ArithmeticGeometricMean[a, b]"
    );
  }

  #[test]
  fn exact_stays_unevaluated() {
    assert_eq!(
      interpret("ArithmeticGeometricMean[1, 2]").unwrap(),
      "ArithmeticGeometricMean[1, 2]"
    );
  }

  #[test]
  fn attributes() {
    assert_eq!(
      interpret("Attributes[ArithmeticGeometricMean]").unwrap(),
      "{Listable, NumericFunction, Orderless, Protected, ReadProtected}"
    );
  }

  #[test]
  fn zero_arg_preserves_type() {
    // A machine-real zero yields a machine-real zero; an exact zero yields 0.
    assert_eq!(interpret("ArithmeticGeometricMean[0., 5.]").unwrap(), "0.");
    assert_eq!(interpret("ArithmeticGeometricMean[3, 0.]").unwrap(), "0.");
    assert_eq!(interpret("ArithmeticGeometricMean[5., 0]").unwrap(), "0");
    assert_eq!(interpret("ArithmeticGeometricMean[0., 0.]").unwrap(), "0.");
  }

  #[test]
  fn equal_args_with_real_is_real() {
    assert_eq!(interpret("ArithmeticGeometricMean[1, 1.]").unwrap(), "1.");
    assert_eq!(
      interpret("ArithmeticGeometricMean[2.0, 2.0]").unwrap(),
      "2."
    );
  }

  #[test]
  fn negative_real_args() {
    // AGM[-a, -b] = -AGM[a, b] for same-sign reals.
    assert_eq!(
      interpret("ArithmeticGeometricMean[-2., -8.]").unwrap(),
      "-4.486057160575205"
    );
    assert_eq!(
      interpret("ArithmeticGeometricMean[-1., -1.]").unwrap(),
      "-1."
    );
  }

  #[test]
  fn orderless_sorts_arguments() {
    // The Orderless attribute canonicalizes the argument order.
    assert_eq!(
      interpret("ArithmeticGeometricMean[24, 6]").unwrap(),
      "ArithmeticGeometricMean[6, 24]"
    );
    assert_eq!(
      interpret("ArithmeticGeometricMean[b, a]").unwrap(),
      "ArithmeticGeometricMean[a, b]"
    );
    assert_eq!(
      interpret("ArithmeticGeometricMean[x, 2]").unwrap(),
      "ArithmeticGeometricMean[2, x]"
    );
    assert_eq!(
      interpret("ArithmeticGeometricMean[1/2, 1/8]").unwrap(),
      "ArithmeticGeometricMean[1/8, 1/2]"
    );
    assert_eq!(
      interpret("ArithmeticGeometricMean[-2, -8]").unwrap(),
      "ArithmeticGeometricMean[-8, -2]"
    );
  }

  #[test]
  fn machine_real_values() {
    assert_eq!(
      interpret("ArithmeticGeometricMean[1.8, 1.2]").unwrap(),
      "1.4848082617417828"
    );
    assert_eq!(
      interpret("ArithmeticGeometricMean[3.0, 7.0]").unwrap(),
      "4.789013583140951"
    );
  }
}

mod random_prime {
  use super::*;

  #[test]
  fn with_max() {
    // Seed for determinism
    interpret("SeedRandom[42]").unwrap();
    let result: i128 = interpret("RandomPrime[100]").unwrap().parse().unwrap();
    assert!(result >= 2 && result <= 100);
    assert!(woxi::is_prime(result as usize));
  }

  #[test]
  fn with_range() {
    interpret("SeedRandom[42]").unwrap();
    let result: i128 =
      interpret("RandomPrime[{10, 30}]").unwrap().parse().unwrap();
    assert!(result >= 10 && result <= 30);
    assert!(woxi::is_prime(result as usize));
  }

  #[test]
  fn list_of_primes() {
    interpret("SeedRandom[42]").unwrap();
    assert_eq!(interpret("Length[RandomPrime[100, 10]]").unwrap(), "10");
  }

  #[test]
  fn all_results_are_prime() {
    interpret("SeedRandom[42]").unwrap();
    assert_eq!(
      interpret("AllTrue[RandomPrime[100, 20], PrimeQ]").unwrap(),
      "True"
    );
  }

  #[test]
  fn dim_list_produces_matrix() {
    // RandomPrime[range, {2, 5}] yields a 2x5 matrix of primes (matches
    // wolframscript). Previously we rejected the dim-list with
    // "second argument must be a positive integer".
    interpret("SeedRandom[42]").unwrap();
    let result =
      interpret("Dimensions[RandomPrime[{10, 30}, {2, 5}]]").unwrap();
    assert_eq!(result, "{2, 5}");
    interpret("SeedRandom[42]").unwrap();
    let all_prime =
      interpret("AllTrue[Flatten[RandomPrime[{10, 30}, {2, 5}]], PrimeQ]")
        .unwrap();
    assert_eq!(all_prime, "True");
  }

  #[test]
  fn all_results_in_range() {
    interpret("SeedRandom[42]").unwrap();
    assert_eq!(
      interpret(
        "AllTrue[RandomPrime[{10, 50}, 20], Function[x, 10 <= x <= 50]]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn range_with_single_prime() {
    interpret("SeedRandom[42]").unwrap();
    // Only prime in {14, 17} is 17
    assert_eq!(interpret("RandomPrime[{14, 17}]").unwrap(), "17");
  }

  #[test]
  fn no_primes_in_range() {
    assert!(interpret("RandomPrime[{14, 16}]").is_err());
  }

  #[test]
  fn max_less_than_2() {
    assert!(interpret("RandomPrime[1]").is_err());
  }

  #[test]
  fn symbolic_stays_unevaluated() {
    assert_eq!(interpret("RandomPrime[x]").unwrap(), "RandomPrime[x]");
  }

  #[test]
  fn ten_digit_prime() {
    interpret("SeedRandom[42]").unwrap();
    let result: i128 = interpret("RandomPrime[{10^9, 10^10}]")
      .unwrap()
      .parse()
      .unwrap();
    assert!(result >= 1_000_000_000 && result <= 10_000_000_000);
  }
}

mod divisors_tests {
  use super::*;

  #[test]
  fn positive_integer() {
    assert_eq!(interpret("Divisors[12]").unwrap(), "{1, 2, 3, 4, 6, 12}");
  }

  #[test]
  fn negative_integer_same_as_positive() {
    assert_eq!(interpret("Divisors[-12]").unwrap(), "{1, 2, 3, 4, 6, 12}");
  }

  #[test]
  fn zero_stays_unevaluated() {
    // Divisors[0] is mathematically undefined; Wolfram leaves it
    // unevaluated.
    assert_eq!(interpret("Divisors[0]").unwrap(), "Divisors[0]");
  }

  #[test]
  fn symbolic_stays_unevaluated() {
    assert_eq!(interpret("Divisors[x]").unwrap(), "Divisors[x]");
  }

  #[test]
  fn listable_threads_over_list() {
    assert_eq!(
      interpret("Divisors[{6, 12}]").unwrap(),
      "{{1, 2, 3, 6}, {1, 2, 3, 4, 6, 12}}"
    );
  }

  #[test]
  fn listable_multiple_elements() {
    assert_eq!(
      interpret("Divisors[{1, 2, 3, 4, 5}]").unwrap(),
      "{{1}, {1, 2}, {1, 3}, {1, 2, 4}, {1, 5}}"
    );
  }

  #[test]
  fn divisors_one() {
    assert_eq!(interpret("Divisors[1]").unwrap(), "{1}");
  }

  #[test]
  fn divisors_prime() {
    assert_eq!(interpret("Divisors[7]").unwrap(), "{1, 7}");
  }
}

mod divisor_sigma {
  use super::*;

  #[test]
  fn sum_of_divisors() {
    assert_eq!(interpret("DivisorSigma[1, 12]").unwrap(), "28");
    assert_eq!(interpret("DivisorSigma[0, 12]").unwrap(), "6");
    assert_eq!(interpret("DivisorSigma[2, 10]").unwrap(), "130");
  }

  // Sigma uses |n|, so it is defined on negatives.
  #[test]
  fn negative_uses_absolute_value() {
    assert_eq!(interpret("DivisorSigma[1, -12]").unwrap(), "28");
  }

  // 0 has no finite divisor sum; wolframscript leaves it unevaluated
  // rather than raising an error.
  #[test]
  fn zero_unevaluated() {
    assert_eq!(
      interpret("DivisorSigma[1, 0]").unwrap(),
      "DivisorSigma[1, 0]"
    );
    assert_eq!(
      interpret("DivisorSigma[0, 0]").unwrap(),
      "DivisorSigma[0, 0]"
    );
    assert_eq!(
      interpret("DivisorSigma[2, 0]").unwrap(),
      "DivisorSigma[2, 0]"
    );
  }

  #[test]
  fn symbolic_modulus_unevaluated() {
    assert_eq!(
      interpret("DivisorSigma[0, x]").unwrap(),
      "DivisorSigma[0, x]"
    );
  }

  // Negative order: DivisorSigma[-p, n] = sigma_p(n)/n^p, an exact rational.
  #[test]
  fn negative_order() {
    assert_eq!(interpret("DivisorSigma[-1, 12]").unwrap(), "7/3");
    assert_eq!(interpret("DivisorSigma[-1, 6]").unwrap(), "2");
    assert_eq!(interpret("DivisorSigma[-2, 12]").unwrap(), "35/24");
    assert_eq!(interpret("DivisorSigma[-3, 4]").unwrap(), "73/64");
    assert_eq!(interpret("DivisorSigma[-1, 7]").unwrap(), "8/7");
    assert_eq!(interpret("DivisorSigma[-1, 1]").unwrap(), "1");
  }

  // Negative order uses |n| as well.
  #[test]
  fn negative_order_negative_modulus() {
    assert_eq!(interpret("DivisorSigma[-1, -12]").unwrap(), "7/3");
  }

  // Negative order at 0 still has no divisor sum → unevaluated.
  #[test]
  fn negative_order_zero_modulus() {
    assert_eq!(
      interpret("DivisorSigma[-1, 0]").unwrap(),
      "DivisorSigma[-1, 0]"
    );
  }
}

mod carmichael_lambda {
  use super::*;

  #[test]
  fn small_values() {
    assert_eq!(
      interpret("Table[CarmichaelLambda[n], {n, 1, 20}]").unwrap(),
      "{1, 1, 2, 2, 4, 2, 6, 2, 6, 4, 10, 2, 12, 6, 4, 4, 16, 6, 18, 4}"
    );
  }

  #[test]
  fn one() {
    assert_eq!(interpret("CarmichaelLambda[1]").unwrap(), "1");
  }

  #[test]
  fn prime() {
    // For a prime p, CarmichaelLambda[p] = p - 1
    assert_eq!(interpret("CarmichaelLambda[7]").unwrap(), "6");
    assert_eq!(interpret("CarmichaelLambda[13]").unwrap(), "12");
  }

  #[test]
  fn prime_power() {
    // CarmichaelLambda[p^k] = p^(k-1)(p-1) for odd prime p
    assert_eq!(interpret("CarmichaelLambda[9]").unwrap(), "6"); // 3^2
    assert_eq!(interpret("CarmichaelLambda[27]").unwrap(), "18"); // 3^3
  }

  #[test]
  fn power_of_two() {
    assert_eq!(interpret("CarmichaelLambda[2]").unwrap(), "1");
    assert_eq!(interpret("CarmichaelLambda[4]").unwrap(), "2");
    assert_eq!(interpret("CarmichaelLambda[8]").unwrap(), "2");
    assert_eq!(interpret("CarmichaelLambda[16]").unwrap(), "4");
    assert_eq!(interpret("CarmichaelLambda[32]").unwrap(), "8");
  }

  #[test]
  fn composite() {
    assert_eq!(interpret("CarmichaelLambda[100]").unwrap(), "20");
    assert_eq!(interpret("CarmichaelLambda[12]").unwrap(), "2");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("CarmichaelLambda[0]").unwrap(), "0");
  }

  #[test]
  fn negative() {
    // CarmichaelLambda[-n] = CarmichaelLambda[n]
    assert_eq!(interpret("CarmichaelLambda[-5]").unwrap(), "4");
    assert_eq!(interpret("CarmichaelLambda[-12]").unwrap(), "2");
  }

  #[test]
  fn listable() {
    assert_eq!(
      interpret("CarmichaelLambda[{6, 12, 100}]").unwrap(),
      "{2, 2, 20}"
    );
  }
}

mod mersenne_prime_exponent {
  use super::*;

  #[test]
  fn first_ten() {
    assert_eq!(
      interpret("Table[MersennePrimeExponent[n], {n, 10}]").unwrap(),
      "{2, 3, 5, 7, 13, 17, 19, 31, 61, 89}"
    );
  }

  #[test]
  fn single_values() {
    assert_eq!(interpret("MersennePrimeExponent[1]").unwrap(), "2");
    assert_eq!(interpret("MersennePrimeExponent[5]").unwrap(), "13");
    assert_eq!(interpret("MersennePrimeExponent[20]").unwrap(), "4423");
  }

  #[test]
  fn large_index() {
    assert_eq!(interpret("MersennePrimeExponent[51]").unwrap(), "82589933");
  }
}

mod mersenne_prime_exponent_q {
  use super::*;

  // n with 2^n - 1 prime (known Mersenne prime exponents).
  #[test]
  fn exponents_are_true() {
    assert_eq!(interpret("MersennePrimeExponentQ[2]").unwrap(), "True");
    assert_eq!(interpret("MersennePrimeExponentQ[7]").unwrap(), "True");
    assert_eq!(interpret("MersennePrimeExponentQ[13]").unwrap(), "True");
    assert_eq!(interpret("MersennePrimeExponentQ[127]").unwrap(), "True");
    assert_eq!(interpret("MersennePrimeExponentQ[521]").unwrap(), "True");
  }

  // Primes whose Mersenne number is composite, and non-primes.
  #[test]
  fn non_exponents_are_false() {
    assert_eq!(interpret("MersennePrimeExponentQ[11]").unwrap(), "False");
    assert_eq!(interpret("MersennePrimeExponentQ[23]").unwrap(), "False");
    assert_eq!(interpret("MersennePrimeExponentQ[1]").unwrap(), "False");
    assert_eq!(interpret("MersennePrimeExponentQ[4]").unwrap(), "False");
    assert_eq!(interpret("MersennePrimeExponentQ[0]").unwrap(), "False");
  }

  // Non-integers and symbols are False (not unevaluated), matching
  // wolframscript.
  #[test]
  fn non_integer_is_false() {
    assert_eq!(interpret("MersennePrimeExponentQ[x]").unwrap(), "False");
    assert_eq!(interpret("MersennePrimeExponentQ[-3]").unwrap(), "False");
    assert_eq!(interpret("MersennePrimeExponentQ[7/2]").unwrap(), "False");
    assert_eq!(interpret("MersennePrimeExponentQ[5.0]").unwrap(), "False");
  }
}

mod cases {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn bit_length_1() {
    assert_case(r#"BitLength[1023]"#, r#"10"#);
  }
  #[test]
  fn bit_length_2() {
    assert_case(r#"BitLength[1023]; BitLength[100]"#, r#"7"#);
  }
  #[test]
  fn bit_length_3() {
    assert_case(r#"BitLength[1023]; BitLength[100]; BitLength[-5]"#, r#"3"#);
  }
  #[test]
  fn bit_length_4() {
    assert_case(
      r#"BitLength[1023]; BitLength[100]; BitLength[-5]; BitLength[0]"#,
      r#"0"#,
    );
  }
  #[test]
  fn digit_count_1() {
    assert_case(r#"DigitCount[1022]"#, r#"{1, 2, 0, 0, 0, 0, 0, 0, 0, 1}"#);
  }
  #[test]
  fn digit_count_2() {
    assert_case(
      r#"DigitCount[1022]; DigitCount[Floor[Pi * 10^100]]"#,
      r#"{8, 12, 12, 10, 8, 9, 8, 12, 14, 8}"#,
    );
  }
  #[test]
  fn digit_count_3() {
    assert_case(
      r#"DigitCount[1022]; DigitCount[Floor[Pi * 10^100]]; DigitCount[1022, 2]"#,
      r#"{9, 1}"#,
    );
  }
  #[test]
  fn digit_count_4() {
    assert_case(
      r#"DigitCount[1022]; DigitCount[Floor[Pi * 10^100]]; DigitCount[1022, 2]; DigitCount[1022, 2, 1]"#,
      r#"9"#,
    );
  }
  #[test]
  fn from_digits_1() {
    assert_case(r#"FromDigits["123"]"#, r#"123"#);
  }
  #[test]
  fn from_digits_2() {
    assert_case(r#"FromDigits["123"]; FromDigits[{1, 2, 3}]"#, r#"123"#);
  }
  #[test]
  fn from_digits_3() {
    assert_case(
      r#"FromDigits["123"]; FromDigits[{1, 2, 3}]; FromDigits[{1, 0, 1}, 1000]"#,
      r#"1000001"#,
    );
  }
  #[test]
  fn from_digits_4() {
    assert_case(
      r#"FromDigits["123"]; FromDigits[{1, 2, 3}]; FromDigits[{1, 0, 1}, 1000]; FromDigits[{a, b, c}, 5]"#,
      r#"5*(5*a + b) + c"#,
    );
  }
  #[test]
  fn from_digits_5() {
    assert_case(
      r#"FromDigits["123"]; FromDigits[{1, 2, 3}]; FromDigits[{1, 0, 1}, 1000]; FromDigits[{a, b, c}, 5]; FromDigits["a0"]"#,
      r#"100"#,
    );
  }
  #[test]
  fn from_digits_6() {
    assert_case(
      r#"FromDigits["123"]; FromDigits[{1, 2, 3}]; FromDigits[{1, 0, 1}, 1000]; FromDigits[{a, b, c}, 5]; FromDigits["a0"]; FromDigits["a0", 16]"#,
      r#"160"#,
    );
  }
  #[test]
  fn from_digits_7() {
    assert_case(
      r#"FromDigits["123"]; FromDigits[{1, 2, 3}]; FromDigits[{1, 0, 1}, 1000]; FromDigits[{a, b, c}, 5]; FromDigits["a0"]; FromDigits["a0", 16]; FromDigits[{}]"#,
      r#"0"#,
    );
  }
  #[test]
  fn from_digits_8() {
    assert_case(
      r#"FromDigits["123"]; FromDigits[{1, 2, 3}]; FromDigits[{1, 0, 1}, 1000]; FromDigits[{a, b, c}, 5]; FromDigits["a0"]; FromDigits["a0", 16]; FromDigits[{}]; FromDigits[""]"#,
      r#"0"#,
    );
  }
  #[test]
  fn integer_digits_1() {
    assert_case(r#"IntegerDigits[76543]"#, r#"{7, 6, 5, 4, 3}"#);
  }
  #[test]
  fn integer_digits_2() {
    assert_case(
      r#"IntegerDigits[76543]; IntegerDigits[76543, 10]"#,
      r#"{7, 6, 5, 4, 3}"#,
    );
  }
  #[test]
  fn integer_digits_3() {
    assert_case(
      r#"IntegerDigits[76543]; IntegerDigits[76543, 10]; IntegerDigits[-76543]"#,
      r#"{7, 6, 5, 4, 3}"#,
    );
  }
  #[test]
  fn integer_digits_4() {
    assert_case(
      r#"IntegerDigits[76543]; IntegerDigits[76543, 10]; IntegerDigits[-76543]; IntegerDigits[76543, 10, 3]"#,
      r#"{5, 4, 3}"#,
    );
  }
  #[test]
  fn integer_digits_5() {
    assert_case(
      r#"IntegerDigits[76543]; IntegerDigits[76543, 10]; IntegerDigits[-76543]; IntegerDigits[76543, 10, 3]; IntegerDigits[25, 8]"#,
      r#"{3, 1}"#,
    );
  }
  #[test]
  fn integer_reverse_1() {
    assert_case(r#"IntegerReverse[1234]"#, r#"4321"#);
  }
  #[test]
  fn integer_reverse_2() {
    assert_case(r#"IntegerReverse[1234]; IntegerReverse[1022, 2]"#, r#"511"#);
  }
  #[test]
  fn integer_reverse_3() {
    assert_case(
      r#"IntegerReverse[1234]; IntegerReverse[1022, 2]; IntegerReverse[-123]"#,
      r#"321"#,
    );
  }
  #[test]
  fn continued_fraction_1() {
    assert_case(
      r#"ContinuedFraction[Pi, 10]"#,
      r#"{3, 7, 15, 1, 292, 1, 1, 1, 2, 1}"#,
    );
  }
  #[test]
  fn continued_fraction_2() {
    assert_case(
      r#"ContinuedFraction[Pi, 10]; ContinuedFraction[(1 + 2 Sqrt[3])/5]"#,
      r#"{0, 1, {8, 3, 34, 3}}"#,
    );
  }
  #[test]
  fn continued_fraction_3() {
    assert_case(
      r#"ContinuedFraction[Pi, 10]; ContinuedFraction[(1 + 2 Sqrt[3])/5]; ContinuedFraction[Sqrt[70]]"#,
      r#"{8, {2, 1, 2, 1, 2, 16}}"#,
    );
  }
  #[test]
  fn continued_fraction_real_1() {
    // Machine reals return only precision-justified terms.
    assert_case(r#"ContinuedFraction[3.245]"#, r#"{3, 4, 12}"#);
  }
  #[test]
  fn continued_fraction_real_2() {
    assert_case(r#"ContinuedFraction[2.5]"#, r#"{2}"#);
  }
  #[test]
  fn continued_fraction_real_3() {
    assert_case(
      r#"ContinuedFraction[0.1]; ContinuedFraction[-3.245]"#,
      r#"{-3, -4, -12}"#,
    );
  }
  #[test]
  fn continued_fraction_real_4() {
    assert_case(
      r#"ContinuedFraction[3.14159]"#,
      r#"{3, 7, 15, 1, 25, 1, 7}"#,
    );
  }
  #[test]
  fn continued_fraction_real_with_count() {
    // The count caps the precision-justified terms (3.245 has only 3).
    assert_case(r#"ContinuedFraction[3.245, 2]"#, r#"{3, 4}"#);
    assert_case(r#"ContinuedFraction[3.245, 4]"#, r#"{3, 4, 12}"#);
    assert_case(r#"ContinuedFraction[2.5, 3]"#, r#"{2}"#);
  }
  #[test]
  fn continued_fraction_whole_real_unevaluated() {
    // A whole-number Real has no fractional CF terms; Wolfram leaves it as is.
    assert_case(r#"ContinuedFraction[3.]"#, r#"ContinuedFraction[3.]"#);
  }
  #[test]
  fn divisors_1() {
    assert_case(r#"Divisors[20]"#, r#"{1, 2, 4, 5, 10, 20}"#);
  }
  #[test]
  fn divisors_2() {
    assert_case(r#"Divisors[20]"#, r#"{1, 2, 4, 5, 10, 20}"#);
  }
  #[test]
  fn divisors_3() {
    assert_case(
      r#"Divisors[20]; Divisors[704]"#,
      r#"{1, 2, 4, 8, 11, 16, 22, 32, 44, 64, 88, 176, 352, 704}"#,
    );
  }
  #[test]
  fn divisors_4() {
    assert_case(
      r#"Divisors[20]; Divisors[704]; Divisors[{87, 106, 202, 305}]"#,
      r#"{{1, 3, 29, 87}, {1, 2, 53, 106}, {1, 2, 101, 202}, {1, 5, 61, 305}}"#,
    );
  }
  #[test]
  fn from_continued_fraction() {
    assert_case(
      r#"FromContinuedFraction[{3, 7, 15, 1, 292, 1, 1, 1, 2, 1}]"#,
      r#"1146408 / 364913"#,
    );
  }
  #[test]
  fn from_continued_fraction_periodic() {
    // A trailing sublist denotes the repeating block; the value is an exact
    // quadratic surd in wolframscript's canonical (P + S Sqrt[D])/Q form.
    assert_case(r#"FromContinuedFraction[{1, {2}}]"#, r#"Sqrt[2]"#);
    assert_case(r#"FromContinuedFraction[{{1}}]"#, r#"(1 + Sqrt[5])/2"#);
    assert_case(r#"FromContinuedFraction[{{2}}]"#, r#"1 + Sqrt[2]"#);
    assert_case(
      r#"FromContinuedFraction[{1, 2, {3, 4}}]"#,
      r#"(4 + Sqrt[3])/4"#,
    );
    assert_case(r#"FromContinuedFraction[{0, {1}}]"#, r#"(-1 + Sqrt[5])/2"#);
    assert_case(r#"FromContinuedFraction[{-1, {2}}]"#, r#"-2 + Sqrt[2]"#);
    assert_case(r#"FromContinuedFraction[{{2, 3}}]"#, r#"(3 + Sqrt[15])/3"#);
  }
  #[test]
  fn from_continued_fraction_periodic_invalid() {
    // The repeating block must be the last element; otherwise unevaluated.
    assert_case(
      r#"FromContinuedFraction[{1, {2}, 3}]"#,
      r#"FromContinuedFraction[{1, {2}, 3}]"#,
    );
  }
  #[test]
  fn next_prime_1() {
    assert_case(r#"NextPrime[100]"#, r#"101"#);
  }
  #[test]
  fn next_prime_2() {
    assert_case(r#"NextPrime[100]; NextPrime[100.5, 2]"#, r#"103"#);
  }
  #[test]
  fn next_prime_3() {
    assert_case(
      r#"NextPrime[100]; NextPrime[100.5, 2]; NextPrime[100, 2.5]"#,
      r#"NextPrime[100, 2.5]"#,
    );
  }
  #[test]
  fn next_prime_4() {
    assert_case(
      r#"NextPrime[100]; NextPrime[100.5, 2]; NextPrime[100, 2.5]; NextPrime[100, -1]"#,
      r#"97"#,
    );
  }
  #[test]
  fn powers_representations_1() {
    assert_case(
      r#"PowersRepresentations[1729, 2, 3]"#,
      r#"{{1, 12}, {9, 10}}"#,
    );
  }
  #[test]
  fn powers_representations_2() {
    assert_case(
      r#"PowersRepresentations[1729, 2, 3]; PowersRepresentations[25, 2, 2]"#,
      r#"{{0, 5}, {3, 4}}"#,
    );
  }
  #[test]
  fn powers_representations_3() {
    assert_case(
      r#"PowersRepresentations[1729, 2, 3]; PowersRepresentations[25, 2, 2]; PowersRepresentations[25, 3, 2]"#,
      r#"{{0, 0, 5}, {0, 3, 4}}"#,
    );
  }
  #[test]
  fn prime_1() {
    assert_case(r#"Prime[1]"#, r#"2"#);
  }
  #[test]
  fn prime_2() {
    assert_case(r#"Prime[1]; Prime[167]"#, r#"991"#);
  }
  #[test]
  fn prime_3() {
    assert_case(
      r#"Prime[1]; Prime[167]; Prime[{5, 10, 15}]"#,
      r#"{11, 29, 47}"#,
    );
  }
  // Prime requires an exact positive integer; wolframscript rejects any Real
  // index (even integer-valued like 3.0), leaving the call unevaluated rather
  // than computing Prime[3] == 5.
  #[test]
  fn prime_real_index_unevaluated() {
    assert_case(r#"Prime[3.0]"#, r#"Prime[3.]"#);
    assert_case(r#"Prime[2.5]"#, r#"Prime[2.5]"#);
  }
  #[test]
  fn prime_pi_1() {
    assert_case(r#"PrimePi[2]"#, r#"1"#);
  }
  #[test]
  fn prime_pi_2() {
    assert_case(r#"PrimePi[2]; PrimePi[100]"#, r#"25"#);
  }
  #[test]
  fn prime_pi_3() {
    assert_case(r#"PrimePi[2]; PrimePi[100]; PrimePi[-1]"#, r#"0"#);
  }
  #[test]
  fn prime_pi_4() {
    assert_case(
      r#"PrimePi[2]; PrimePi[100]; PrimePi[-1]; PrimePi[3.5]"#,
      r#"2"#,
    );
  }
  #[test]
  fn prime_pi_5() {
    assert_case(
      r#"PrimePi[2]; PrimePi[100]; PrimePi[-1]; PrimePi[3.5]; PrimePi[E]"#,
      r#"1"#,
    );
  }
  #[test]
  fn integer_length_1() {
    assert_case(
      r#"Mod[$RandomState, 10^100]; IntegerLength[$RandomState]"#,
      r#"578"#,
    );
  }
  #[test]
  fn integer_exponent_1() {
    assert_case(r#"IntegerExponent[16, 2]"#, r#"4"#);
  }
  #[test]
  fn integer_exponent_2() {
    assert_case(
      r#"IntegerExponent[16, 2]; IntegerExponent[-510000]"#,
      r#"4"#,
    );
  }
  #[test]
  fn integer_exponent_3() {
    assert_case(
      r#"IntegerExponent[16, 2]; IntegerExponent[-510000]; IntegerExponent[10, b]"#,
      r#"IntegerExponent[10, b]"#,
    );
  }
  #[test]
  fn integer_length_2() {
    assert_case(r#"IntegerLength[123456]"#, r#"6"#);
  }
  #[test]
  fn integer_length_3() {
    assert_case(
      r#"IntegerLength[123456]; IntegerLength[10^10000]"#,
      r#"10001"#,
    );
  }
  #[test]
  fn integer_length_4() {
    assert_case(
      r#"IntegerLength[123456]; IntegerLength[10^10000]; IntegerLength[-10^1000]"#,
      r#"1001"#,
    );
  }
  #[test]
  fn integer_length_5() {
    assert_case(
      r#"IntegerLength[123456]; IntegerLength[10^10000]; IntegerLength[-10^1000]; IntegerLength[8, 2]"#,
      r#"4"#,
    );
  }
  #[test]
  fn number_digit_1() {
    assert_case(r#"NumberDigit[210.345, 2]"#, r#"2"#);
  }
  #[test]
  fn number_digit_2() {
    assert_case(
      r#"NumberDigit[210.345, 2]; NumberDigit[210.345, -1]"#,
      r#"3"#,
    );
  }
  #[test]
  fn real_digits_1() {
    assert_case(
      r#"RealDigits[123.55555]"#,
      r#"{{1, 2, 3, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0}, 3}"#,
    );
  }
  #[test]
  fn real_digits_2() {
    assert_case(
      r#"RealDigits[123.55555]; RealDigits[19 / 7]"#,
      r#"{{2, {7, 1, 4, 2, 8, 5}}, 1}"#,
    );
  }
  #[test]
  fn real_digits_3() {
    assert_case(
      r#"RealDigits[123.55555]; RealDigits[19 / 7]; RealDigits[Pi, 10, 1, -500]; RealDigits[Pi, 10, 11, -3]"#,
      r#"{{1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7}, -2}"#,
    );
  }
  #[test]
  fn real_digits_4() {
    assert_case(
      r#"RealDigits[123.55555]; RealDigits[19 / 7]; RealDigits[Pi, 10, 1, -500]; RealDigits[Pi, 10, 11, -3]; RealDigits[123.45, 10, 18]"#,
      r#"{{1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Indeterminate, Indeterminate}, 3}"#,
    );
  }
  #[test]
  fn real_digits_5() {
    assert_case(
      r#"RealDigits[123.55555]; RealDigits[19 / 7]; RealDigits[Pi, 10, 1, -500]; RealDigits[Pi, 10, 11, -3]; RealDigits[123.45, 10, 18]; RealDigits[Pi, 10, 25]"#,
      r#"{{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3}, 1}"#,
    );
  }
  #[test]
  fn real_digits_6() {
    assert_case(
      r#"RealDigits[123.55555]; RealDigits[19 / 7]; RealDigits[Pi, 10, 1, -500]; RealDigits[Pi, 10, 11, -3]; RealDigits[123.45, 10, 18]; RealDigits[Pi, 10, 25]; RealDigits[10]"#,
      r#"{{1, 0}, 2}"#,
    );
  }
  #[test]
  fn factorial2() {
    assert_case(r#"5!!; Factorial2[-3]"#, r#"-1"#);
  }
  #[test]
  fn plus() {
    assert_case(r#"5!!; Factorial2[-3]; I!! + 1"#, r#"1 + I!!"#);
  }
  #[test]
  fn pochhammer_1() {
    assert_case(r#"Pochhammer[1, 3]"#, r#"6"#);
  }
  #[test]
  fn pochhammer_2() {
    assert_case(
      r#"Pochhammer[1, 3]; Pochhammer[1, 3] == Pochhammer[2, 2]"#,
      r#"True"#,
    );
  }
  #[test]
  fn subfactorial_1() {
    assert_case(r#"Subfactorial[{0, 1, 2, 3}]"#, r#"{1, 0, 1, 2}"#);
  }
  #[test]
  fn subfactorial_2() {
    assert_case(
      r#"Subfactorial[{0, 1, 2, 3}]; Subfactorial[6.0]"#,
      r#"265."#,
    );
  }
  #[test]
  fn bell_b_1() {
    assert_case(r#"BellB[10]"#, r#"115975"#);
  }
  #[test]
  fn bell_b_2() {
    assert_case(
      r#"BellB[10]; BellB[5, x]"#,
      r#"x + 15*x^2 + 25*x^3 + 10*x^4 + x^5"#,
    );
  }
  #[test]
  fn binomial_1() {
    assert_case(r#"Binomial[5, 3]"#, r#"10"#);
  }
  #[test]
  fn binomial_2() {
    assert_case(
      r#"Binomial[5, 3]; Binomial[10.5,3.2]"#,
      r#"165.28610936725698"#,
    );
  }
  #[test]
  fn binomial_3() {
    assert_case(
      r#"Binomial[5, 3]; Binomial[10.5,3.2]; Binomial[10, -2]"#,
      r#"0"#,
    );
  }
  #[test]
  fn binomial_4() {
    assert_case(
      r#"Binomial[5, 3]; Binomial[10.5,3.2]; Binomial[10, -2]; Binomial[-10.5, -3.5]"#,
      r#"0"#,
    );
  }
  #[test]
  fn multinomial_1() {
    assert_case(r#"Multinomial[2, 3, 4, 5]"#, r#"2522520"#);
  }
  #[test]
  fn multinomial_2() {
    assert_case(r#"Multinomial[2, 3, 4, 5]; Multinomial[]"#, r#"1"#);
  }
  #[test]
  fn multinomial_3() {
    assert_case(
      r#"Multinomial[2, 3, 4, 5]; Multinomial[]; Multinomial[a, b, c]"#,
      r#"Multinomial[a, b, c]"#,
    );
  }
  #[test]
  fn multinomial_4() {
    assert_case(
      r#"Multinomial[2, 3, 4, 5]; Multinomial[]; Multinomial[a, b, c]; Multinomial[2, 3]"#,
      r#"10"#,
    );
  }
  #[test]
  fn bernoulli_b() {
    assert_case(r#"BernoulliB[42]"#, r#"1520097643918070802691 / 1806"#);
  }
  #[test]
  fn fibonacci_1() {
    assert_case(r#"Fibonacci[0]"#, r#"0"#);
  }
  #[test]
  fn fibonacci_2() {
    assert_case(r#"Fibonacci[0]; Fibonacci[1]"#, r#"1"#);
  }
  #[test]
  fn fibonacci_3() {
    assert_case(r#"Fibonacci[0]; Fibonacci[1]; Fibonacci[10]"#, r#"55"#);
  }
  #[test]
  fn fibonacci_4() {
    assert_case(
      r#"Fibonacci[0]; Fibonacci[1]; Fibonacci[10]; Fibonacci[200]"#,
      r#"280571172992510140037611932413038677189525"#,
    );
  }
  #[test]
  fn fibonacci_5() {
    assert_case(
      r#"Fibonacci[0]; Fibonacci[1]; Fibonacci[10]; Fibonacci[200]; Fibonacci[7, x]"#,
      r#"1 + 6*x^2 + 5*x^4 + x^6"#,
    );
  }
  #[test]
  fn linear_recurrence_1() {
    assert_case(
      r#"LinearRecurrence[{1, 1}, {1, 1}, 10]"#,
      r#"{1, 1, 2, 3, 5, 8, 13, 21, 34, 55}"#,
    );
  }
  #[test]
  fn linear_recurrence_2() {
    assert_case(
      r#"LinearRecurrence[{1, 1}, {1, 1}, 10]; LinearRecurrence[{1, 1}, {1, 1}, {3, 5}]"#,
      r#"{2, 3, 5}"#,
    );
  }
  #[test]
  fn linear_recurrence_3() {
    assert_case(
      r#"LinearRecurrence[{1, 1}, {1, 1}, 10]; LinearRecurrence[{1, 1}, {1, 1}, {3, 5}]; LinearRecurrence[{1, 1}, {1, 1}, {6}]"#,
      r#"{8}"#,
    );
  }
  #[test]
  fn divisible_1() {
    assert_case(r#"Divisible[10, 2]"#, r#"True"#);
  }
  #[test]
  fn divisible_2() {
    assert_case(r#"Divisible[10, 2]; Divisible[2, 10]"#, r#"False"#);
  }
  #[test]
  fn gcd_1() {
    assert_case(r#"GCD[20, 30]"#, r#"10"#);
  }
  #[test]
  fn gcd_2() {
    assert_case(r#"GCD[20, 30]; GCD[10, y]"#, r#"GCD[10, y]"#);
  }
  #[test]
  fn gcd_3() {
    assert_case(
      r#"GCD[20, 30]; GCD[10, y]; GCD[4, {10, 11, 12, 13, 14}]"#,
      r#"{2, 1, 4, 1, 2}"#,
    );
  }
  #[test]
  fn lcm_1() {
    assert_case(r#"LCM[15, 20]"#, r#"60"#);
  }
  #[test]
  fn lcm_2() {
    assert_case(r#"LCM[15, 20]; LCM[20, 30, 40, 50]"#, r#"600"#);
  }
  #[test]
  fn modular_inverse() {
    assert_case(r#"ModularInverse[2, 3]"#, r#"2"#);
  }
  #[test]
  fn power_mod() {
    assert_case(r#"PowerMod[7, 2, 5]"#, r#"4"#);
  }
  #[test]
  fn divisors_5() {
    assert_case(r#"Divisors[0]"#, r#"Divisors[0]"#);
  }
  #[test]
  fn divisors_6() {
    assert_case(
      r#"Divisors[0]; Divisors[{-206, -502, -1702, 9}]"#,
      r#"{{1, 2, 103, 206}, {1, 2, 251, 502}, {1, 2, 23, 37, 46, 74, 851, 1702}, {1, 3, 9}}"#,
    );
  }
  #[test]
  fn binomial_5() {
    assert_case(r#"Binomial[-10, -3.5]"#, r#"ComplexInfinity"#);
  }
}

mod farey_sequence {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn full_sequences() {
    assert_case(
      r#"FareySequence[5]"#,
      r#"{0, 1/5, 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 1}"#,
    );
    assert_case(r#"FareySequence[1]"#, r#"{0, 1}"#);
    assert_case(
      r#"FareySequence[8]"#,
      r#"{0, 1/8, 1/7, 1/6, 1/5, 1/4, 2/7, 1/3, 3/8, 2/5, 3/7, 1/2, 4/7, 3/5, 5/8, 2/3, 5/7, 3/4, 4/5, 5/6, 6/7, 7/8, 1}"#,
    );
  }

  #[test]
  fn indexed_elements() {
    assert_case(r#"FareySequence[6, 3]"#, r#"1/5"#);
    assert_case(r#"FareySequence[5, 11]"#, r#"1"#);
    assert_case(r#"FareySequence[2, 3]"#, r#"1"#);
  }

  #[test]
  fn non_positive_order_messages_null() {
    // FareySequence::intpm message, result Null (1-argument form only)
    assert_case(r#"FareySequence[0]"#, r#"Null"#);
    assert_case(r#"FareySequence[-1]"#, r#"Null"#);
  }

  #[test]
  fn silently_unevaluated_forms() {
    // Out-of-range positive rank messages FareySequence::rank
    assert_case(r#"FareySequence[5, 100]"#, r#"FareySequence[5, 100]"#);
    // Rank 0, non-positive order in rank form, and symbolic/rational
    // arguments all stay quiet
    assert_case(r#"FareySequence[5, 0]"#, r#"FareySequence[5, 0]"#);
    assert_case(r#"FareySequence[0, 1]"#, r#"FareySequence[0, 1]"#);
    assert_case(r#"FareySequence[n]"#, r#"FareySequence[n]"#);
    assert_case(r#"FareySequence[5/2]"#, r#"FareySequence[5/2]"#);
    assert_case(r#"FareySequence[3, x]"#, r#"FareySequence[3, x]"#);
  }
}

mod number_expand {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn decimal_place_values() {
    assert_case(r#"NumberExpand[325]"#, r#"{300, 20, 5}"#);
    assert_case(r#"NumberExpand[12345]"#, r#"{10000, 2000, 300, 40, 5}"#);
    // Zero digits stay in the list
    assert_case(r#"NumberExpand[305]"#, r#"{300, 0, 5}"#);
    assert_case(r#"NumberExpand[1000]"#, r#"{1000, 0, 0, 0}"#);
    assert_case(r#"NumberExpand[0]"#, r#"{0}"#);
    // Each term carries the sign
    assert_case(r#"NumberExpand[-325]"#, r#"{-300, -20, -5}"#);
  }

  #[test]
  fn other_bases() {
    assert_case(
      r#"NumberExpand[325, 2]"#,
      r#"{256, 0, 64, 0, 0, 0, 4, 0, 1}"#,
    );
    assert_case(r#"NumberExpand[7, 2]"#, r#"{4, 2, 1}"#);
    assert_case(r#"NumberExpand[-7, 2]"#, r#"{-4, -2, -1}"#);
    assert_case(r#"NumberExpand[325, 16]"#, r#"{256, 64, 5}"#);
    assert_case(r#"NumberExpand[0, 2]"#, r#"{0}"#);
  }

  #[test]
  fn rationals_stay_whole() {
    // No positional expansion for exact non-integers, base ignored
    assert_case(r#"NumberExpand[1/3]"#, r#"{1/3}"#);
    assert_case(r#"NumberExpand[-1/3]"#, r#"{-1/3}"#);
    assert_case(r#"NumberExpand[1/3, 2]"#, r#"{1/3}"#);
    assert_case(r#"NumberExpand[5/2]"#, r#"{5/2}"#);
  }

  #[test]
  fn invalid_input_handling() {
    // Integer base below 2 messages NumberExpand::rbase (even when the
    // value itself would not expand)
    assert_case(r#"NumberExpand[325, 1]"#, r#"NumberExpand[325, 1]"#);
    assert_case(r#"NumberExpand[1/3, 1]"#, r#"NumberExpand[1/3, 1]"#);
    // Symbolic value or base stays silently unevaluated
    assert_case(r#"NumberExpand[x]"#, r#"NumberExpand[x]"#);
    assert_case(r#"NumberExpand[325, x]"#, r#"NumberExpand[325, x]"#);
  }
}

mod number_decompose {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn greedy_decomposition() {
    assert_case(r#"NumberDecompose[325, {100, 10, 1}]"#, r#"{3, 2, 5}"#);
    assert_case(
      r#"NumberDecompose[327/100, {1, 1/10, 1/100}]"#,
      r#"{3, 2, 7}"#,
    );
    // Greedy, not optimal: 17 = 3*5 + 1*2 + 0*1
    assert_case(r#"NumberDecompose[17, {5, 2, 1}]"#, r#"{3, 1, 0}"#);
    assert_case(r#"NumberDecompose[0, {100, 10, 1}]"#, r#"{0, 0, 0}"#);
    // Quotients truncate toward zero, so signs carry through
    assert_case(r#"NumberDecompose[-325, {100, 10, 1}]"#, r#"{-3, -2, -5}"#);
    // Equal adjacent units are allowed (nonincreasing)
    assert_case(r#"NumberDecompose[25, {10, 10, 1}]"#, r#"{2, 0, 5}"#);
  }

  #[test]
  fn last_entry_is_remainder_quotient() {
    // The final element is rem/unit, not floored
    assert_case(r#"NumberDecompose[7/2, {1}]"#, r#"{7/2}"#);
    // ...and becomes a machine real when any input is real
    assert_case(r#"NumberDecompose[10.25, {10, 1, 0.25}]"#, r#"{1, 0, 1.}"#);
    assert_case(r#"NumberDecompose[10, {3, 1.}]"#, r#"{3, 1.}"#);
  }

  #[test]
  fn unit_validation() {
    // Non-positive, increasing, or symbolic units message
    // NumberDecompose::psv and stay unevaluated
    assert_case(
      r#"NumberDecompose[325, {10, 100, 1}]"#,
      r#"NumberDecompose[325, {10, 100, 1}]"#,
    );
    assert_case(
      r#"NumberDecompose[325, {0, 10}]"#,
      r#"NumberDecompose[325, {0, 10}]"#,
    );
    assert_case(
      r#"NumberDecompose[325, {-10, 1}]"#,
      r#"NumberDecompose[325, {-10, 1}]"#,
    );
    assert_case(
      r#"NumberDecompose[325, {100, y}]"#,
      r#"NumberDecompose[325, {100, y}]"#,
    );
  }

  #[test]
  fn symbolic_value_skips_unit_validation() {
    // No psv message when the value is not numeric
    assert_case(
      r#"NumberDecompose[x, {2, 3}]"#,
      r#"NumberDecompose[x, {2, 3}]"#,
    );
    assert_case(
      r#"NumberDecompose[x, {100, 10}]"#,
      r#"NumberDecompose[x, {100, 10}]"#,
    );
    assert_case(r#"NumberDecompose[325, x]"#, r#"NumberDecompose[325, x]"#);
  }
}

mod number_compose {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn reconstructs_from_coefficients_and_units() {
    assert_case(r#"NumberCompose[{1, 2, 3}, {100, 10, 1}]"#, r#"123"#);
    assert_case(
      r#"NumberCompose[{1, 2, 3, 4}, {100, 10, 1, 0.1}]"#,
      r#"123.4"#,
    );
    // Exact rational arithmetic.
    assert_case(r#"NumberCompose[{2, 5, 3}, {12, 1, 1/12}]"#, r#"117/4"#);
    // Equal adjacent units are allowed (nonincreasing).
    assert_case(r#"NumberCompose[{1, 2}, {10, 10}]"#, r#"30"#);
    // Negative coefficients carry through.
    assert_case(r#"NumberCompose[{1, -2, 3}, {100, 10, 1}]"#, r#"83"#);
  }

  #[test]
  fn round_trips_with_number_decompose() {
    assert_case(
      r#"NumberCompose[NumberDecompose[123.456, {100, 10, 1, 0.1}], {100, 10, 1, 0.1}]"#,
      r#"123.456"#,
    );
  }

  #[test]
  fn shorter_coefficients_align_to_the_trailing_units() {
    // {1, 2} pairs with the last two units {10, 1}: 1*10 + 2*1 = 12.
    assert_case(r#"NumberCompose[{1, 2}, {100, 10, 1}]"#, r#"12"#);
    assert_case(r#"NumberCompose[{3}, {100, 10, 1}]"#, r#"3"#);
    assert_case(r#"NumberCompose[{}, {10, 1}]"#, r#"0"#);
  }

  #[test]
  fn symbolic_coefficients() {
    assert_case(
      r#"NumberCompose[{a, b, c}, {100, 10, 1}]"#,
      r#"100*a + 10*b + c"#,
    );
  }

  #[test]
  fn too_many_coefficients_stays_unevaluated() {
    // The coefficient list cannot be longer than the unit list (ulen),
    // checked before unit validation.
    assert_case(
      r#"NumberCompose[{1, 2, 3, 4}, {100, 10, 1}]"#,
      r#"NumberCompose[{1, 2, 3, 4}, {100, 10, 1}]"#,
    );
    assert_case(
      r#"NumberCompose[{1, 2, 3}, {1, 10}]"#,
      r#"NumberCompose[{1, 2, 3}, {1, 10}]"#,
    );
  }

  #[test]
  fn invalid_units_stay_unevaluated() {
    // Units must be nonincreasing positive numbers (psv).
    assert_case(
      r#"NumberCompose[{1, 2}, {1, 10}]"#,
      r#"NumberCompose[{1, 2}, {1, 10}]"#,
    );
    assert_case(
      r#"NumberCompose[{1, 2}, {10, 0}]"#,
      r#"NumberCompose[{1, 2}, {10, 0}]"#,
    );
    assert_case(
      r#"NumberCompose[{1, 2}, {10, -1}]"#,
      r#"NumberCompose[{1, 2}, {10, -1}]"#,
    );
    assert_case(
      r#"NumberCompose[{1, 2}, {10, y}]"#,
      r#"NumberCompose[{1, 2}, {10, y}]"#,
    );
  }
}

mod fibonorial {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn small_values() {
    assert_case(
      r#"Fibonorial /@ Range[0, 10]"#,
      r#"{1, 1, 1, 2, 6, 30, 240, 3120, 65520, 2227680, 122522400}"#,
    );
  }

  #[test]
  fn big_integer_values() {
    assert_case(
      r#"Fibonorial[20]"#,
      r#"9692987370815489224102512784450560000"#,
    );
  }

  #[test]
  fn negative_integers_are_complex_infinity() {
    assert_case(r#"Fibonorial[-1]"#, r#"ComplexInfinity"#);
    assert_case(r#"Fibonorial[-4]"#, r#"ComplexInfinity"#);
  }

  #[test]
  fn non_integer_input() {
    // Rationals and reals emit Fibonorial::intnm and stay unevaluated;
    // symbols stay quiet
    assert_case(r#"Fibonorial[5/2]"#, r#"Fibonorial[5/2]"#);
    assert_case(r#"Fibonorial[2.5]"#, r#"Fibonorial[2.5]"#);
    assert_case(r#"Fibonorial[x]"#, r#"Fibonorial[x]"#);
  }
}

mod dirichlet_character {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn cyclic_moduli() {
    assert_case(
      r#"Table[DirichletCharacter[5, j, n], {j, 4}, {n, 5}]"#,
      r#"{{1, 1, 1, 1, 0}, {1, I, -I, -1, 0}, {1, -1, -1, 1, 0}, {1, -I, I, -1, 0}}"#,
    );
    // Order-6 values keep their exponential print forms
    assert_case(
      r#"Table[DirichletCharacter[9, j, 2], {j, 6}]"#,
      r#"{1, E^(I/3*Pi), E^((2*I)/3*Pi), -1, E^((-2*I)/3*Pi), E^((-1/3*I)*Pi)}"#,
    );
    assert_case(r#"DirichletCharacter[7, 2, 5]"#, r#"E^((-1/3*I)*Pi)"#);
  }

  #[test]
  fn two_power_moduli() {
    // (Z/8)* = <-1> x <5>: the 5-part index runs fastest
    assert_case(
      r#"Table[DirichletCharacter[8, j, n], {j, 4}, {n, 8}]"#,
      r#"{{1, 0, 1, 0, 1, 0, 1, 0}, {1, 0, -1, 0, -1, 0, 1, 0}, {1, 0, -1, 0, 1, 0, -1, 0}, {1, 0, 1, 0, -1, 0, -1, 0}}"#,
    );
    assert_case(
      r#"Table[DirichletCharacter[16, j, 3], {j, 8}]"#,
      r#"{1, -I, -1, I, -1, I, 1, -I}"#,
    );
  }

  #[test]
  fn composite_moduli() {
    // CRT factors in ascending prime order, last factor fastest
    assert_case(
      r#"Table[DirichletCharacter[12, j, n], {j, 4}, {n, 12}]"#,
      r#"{{1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, -1, 0, 1, 0, 0, 0, -1, 0}, {1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0}, {1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0}}"#,
    );
    assert_case(
      r#"Table[DirichletCharacter[15, j, 2], {j, 8}]"#,
      r#"{1, I, -1, -I, -1, -I, 1, I}"#,
    );
    // Coefficient stays outside the exponential: -1 * e^(2 i pi/3)
    assert_case(
      r#"Table[DirichletCharacter[21, j, 2], {j, 12}]"#,
      r#"{1, E^((2*I)/3*Pi), E^((-2*I)/3*Pi), 1, E^((2*I)/3*Pi), E^((-2*I)/3*Pi), -1, -E^((2*I)/3*Pi), -E^((-2*I)/3*Pi), -1, -E^((2*I)/3*Pi), -E^((-2*I)/3*Pi)}"#,
    );
    // +-I coefficients multiply positive exponentials but divide
    // negative ones (Times with a negative power prints as division)
    assert_case(
      r#"Table[DirichletCharacter[45, j, 2], {j, 24}]"#,
      r#"{1, I, -1, -I, E^(I/3*Pi), I*E^(I/3*Pi), -E^(I/3*Pi), -I*E^(I/3*Pi), E^((2*I)/3*Pi), I*E^((2*I)/3*Pi), -E^((2*I)/3*Pi), -I*E^((2*I)/3*Pi), -1, -I, 1, I, E^((-2*I)/3*Pi), I/E^((2*I)/3*Pi), -E^((-2*I)/3*Pi), -I/E^((2*I)/3*Pi), E^((-1/3*I)*Pi), I/E^(I/3*Pi), -E^((-1/3*I)*Pi), -I/E^(I/3*Pi)}"#,
    );
  }

  #[test]
  fn trivial_moduli_and_periodicity() {
    // k = 1: identically 1, even at n = 0
    assert_case(
      r#"Table[DirichletCharacter[1, 1, n], {n, 0, 4}]"#,
      r#"{1, 1, 1, 1, 1}"#,
    );
    assert_case(
      r#"Table[DirichletCharacter[2, 1, n], {n, 0, 4}]"#,
      r#"{0, 1, 0, 1, 0}"#,
    );
    // chi is periodic: -3 == 2 mod 5
    assert_case(r#"DirichletCharacter[5, 2, -3]"#, r#"I"#);
    assert_case(r#"DirichletCharacter[5, 2, 0]"#, r#"0"#);
  }

  #[test]
  fn invalid_index_handling() {
    // j > EulerPhi[k] messages DirichletCharacter::invl; j <= 0
    // messages DirichletCharacter::intp; symbolic n stays quiet
    assert_case(
      r#"DirichletCharacter[5, 5, 2]"#,
      r#"DirichletCharacter[5, 5, 2]"#,
    );
    assert_case(
      r#"DirichletCharacter[5, 0, 2]"#,
      r#"DirichletCharacter[5, 0, 2]"#,
    );
    assert_case(
      r#"DirichletCharacter[5, 2, x]"#,
      r#"DirichletCharacter[5, 2, x]"#,
    );
  }
}

mod minkowski_question_mark {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn positive_rationals() {
    assert_case(
      r#"MinkowskiQuestionMark /@ {0, 1, 1/2, 1/3, 2/3, 3/7, 5/8}"#,
      r#"{0, 1, 1/2, 1/4, 3/4, 7/16, 11/16}"#,
    );
    // ?(x + 1) = ?(x) + 1 on the positive side
    assert_case(r#"MinkowskiQuestionMark[7/3]"#, r#"9/4"#);
  }

  #[test]
  fn negative_rationals_use_wolframs_expansion() {
    // Wolfram feeds its termwise-negated ContinuedFraction into the
    // dyadic formula, so negative quotients become positive exponents:
    // ?(-1/2) = 2/2^(-2) = 8
    assert_case(
      r#"MinkowskiQuestionMark /@ {-1/2, -1/3, -2/3, -1, -2, -5/8, -7/3}"#,
      r#"{8, 16, -12, -1, -2, -52, 14}"#,
    );
    assert_case(r#"MinkowskiQuestionMark[-3/7]"#, r#"-56"#);
  }

  #[test]
  fn quadratic_irrationals() {
    // Periodic tails sum as geometric series
    assert_case(r#"MinkowskiQuestionMark[Sqrt[2]]"#, r#"7/5"#);
    assert_case(r#"MinkowskiQuestionMark[GoldenRatio]"#, r#"5/3"#);
    assert_case(r#"MinkowskiQuestionMark[1 + Sqrt[2]]"#, r#"12/5"#);
    assert_case(r#"MinkowskiQuestionMark[Sqrt[2]/2]"#, r#"4/5"#);
    // ...even formally divergent ones (ratio -4 here)
    assert_case(r#"MinkowskiQuestionMark[-Sqrt[2]]"#, r#"3/5"#);
  }

  #[test]
  fn unevaluated_forms() {
    assert_case(r#"MinkowskiQuestionMark[x]"#, r#"MinkowskiQuestionMark[x]"#);
    // Machine reals use a limited-precision expansion in wolframscript
    // whose noise is not reproduced; stay symbolic instead
    assert_case(
      r#"MinkowskiQuestionMark[0.3]"#,
      r#"MinkowskiQuestionMark[0.3]"#,
    );
  }
}

mod continued_fraction_negative_convention {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn negative_rationals_negate_termwise() {
    // Regression: floor-based expansion {-1, 2} is wrong; Wolfram
    // negates the expansion of |x| termwise
    assert_case(r#"ContinuedFraction[-1/2]"#, r#"{0, -2}"#);
    assert_case(r#"ContinuedFraction[-2/3]"#, r#"{0, -1, -2}"#);
    assert_case(r#"ContinuedFraction[-7/3]"#, r#"{-2, -3}"#);
    assert_case(r#"ContinuedFraction[-151/77]"#, r#"{-1, -1, -24, -1, -2}"#);
  }

  #[test]
  fn golden_ratio_and_reciprocal_sqrt() {
    // Regression: both stayed unevaluated before
    assert_case(r#"ContinuedFraction[GoldenRatio]"#, r#"{1, {1}}"#);
    assert_case(r#"ContinuedFraction[Sqrt[2]/2]"#, r#"{0, 1, {2}}"#);
  }
}

mod from_roman_numeral {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn standard_numerals() {
    assert_case(r#"FromRomanNumeral["MCMXCIV"]"#, r#"1994"#);
    // Case-insensitive
    assert_case(r#"FromRomanNumeral["mcmxciv"]"#, r#"1994"#);
    assert_case(r#"FromRomanNumeral["McM"]"#, r#"1900"#);
    // N denotes zero (RomanNumeral[0] is "N"), alone or repeated
    assert_case(r#"FromRomanNumeral["N"]"#, r#"0"#);
    assert_case(r#"FromRomanNumeral["NN"]"#, r#"0"#);
    assert_case(r#"FromRomanNumeral[""]"#, r#"0"#);
  }

  #[test]
  fn non_canonical_forms() {
    // Generic pairwise subtractive rule, not strict syntax
    assert_case(r#"FromRomanNumeral["IIII"]"#, r#"4"#);
    assert_case(r#"FromRomanNumeral["XIIX"]"#, r#"20"#);
    assert_case(r#"FromRomanNumeral["VX"]"#, r#"5"#);
    assert_case(r#"FromRomanNumeral["IM"]"#, r#"999"#);
    assert_case(r#"FromRomanNumeral["MMMMM"]"#, r#"5000"#);
  }

  #[test]
  fn listability_and_round_trip() {
    assert_case(r#"FromRomanNumeral[{"III", "IX"}]"#, r#"{3, 9}"#);
    assert_case(
      r#"FromRomanNumeral[RomanNumeral[Range[0, 30]]] == Range[0, 30]"#,
      r#"True"#,
    );
  }

  #[test]
  fn invalid_input() {
    // Invalid characters emit FromRomanNumeral::nrom; non-strings emit
    // FromRomanNumeral::string
    assert_case(r#"FromRomanNumeral["ABC"]"#, r#"FromRomanNumeral[ABC]"#);
    assert_case(r#"FromRomanNumeral["I I"]"#, r#"FromRomanNumeral[I I]"#);
    assert_case(r#"FromRomanNumeral[5]"#, r#"FromRomanNumeral[5]"#);
    assert_case(r#"FromRomanNumeral[x]"#, r#"FromRomanNumeral[x]"#);
  }
}

mod dirichlet_l {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn modulus_one_is_zeta() {
    assert_case(r#"DirichletL[1, 1, 2]"#, r#"Pi^2/6"#);
    assert_case(r#"DirichletL[1, 1, 4]"#, r#"Pi^4/90"#);
    assert_case(r#"DirichletL[1, 1, 3]"#, r#"Zeta[3]"#);
    assert_case(r#"DirichletL[1, 1, 1]"#, r#"ComplexInfinity"#);
    // ...but symbolic s stays unevaluated even for k = 1
    assert_case(r#"DirichletL[1, 1, s]"#, r#"DirichletL[1, 1, s]"#);
  }

  #[test]
  fn classic_values_at_one() {
    // Leibniz series and its mod-3 sibling
    assert_case(r#"DirichletL[4, 2, 1]"#, r#"Pi/4"#);
    assert_case(r#"DirichletL[3, 2, 1]"#, r#"Pi/(3*Sqrt[3])"#);
  }

  #[test]
  fn non_positive_integers_via_generalized_bernoulli() {
    // Real characters give rationals: L(1-n, chi) = -B_{n,chi}/n
    assert_case(r#"DirichletL[4, 2, 0]"#, r#"1/2"#);
    assert_case(r#"DirichletL[4, 2, -1]"#, r#"0"#);
    assert_case(r#"DirichletL[4, 2, -2]"#, r#"-1/2"#);
    assert_case(r#"DirichletL[3, 2, 0]"#, r#"1/3"#);
    assert_case(r#"DirichletL[3, 2, -2]"#, r#"-2/9"#);
    assert_case(r#"DirichletL[8, 3, 0]"#, r#"1/2"#);
    assert_case(r#"DirichletL[12, 4, 0]"#, r#"0"#);
    // Quartic characters give Gaussian rationals
    assert_case(r#"DirichletL[5, 2, 0]"#, r#"3/5 + I/5"#);
    assert_case(r#"DirichletL[5, 2, -2]"#, r#"-4/5 - (2*I)/5"#);
    assert_case(r#"DirichletL[5, 2, -4]"#, r#"148/25 + (86*I)/25"#);
    assert_case(r#"DirichletL[5, 3, 0]"#, r#"0"#);
  }

  #[test]
  fn unevaluated_cases() {
    // Principal characters never evaluate (s = 1 diverges, the rest
    // stays symbolic in wolframscript too)
    assert_case(r#"DirichletL[4, 1, 1]"#, r#"DirichletL[4, 1, 1]"#);
    assert_case(r#"DirichletL[4, 1, 2]"#, r#"DirichletL[4, 1, 2]"#);
    // s >= 2 stays symbolic for k > 1 (wolframscript leaves even the
    // Catalan value DirichletL[4, 2, 2] unevaluated)
    assert_case(r#"DirichletL[4, 2, 2]"#, r#"DirichletL[4, 2, 2]"#);
    // Symbolic and non-integer s
    assert_case(r#"DirichletL[4, 2, s]"#, r#"DirichletL[4, 2, s]"#);
    assert_case(r#"DirichletL[5, 2, 3/2]"#, r#"DirichletL[5, 2, 3/2]"#);
  }

  #[test]
  fn invalid_index_messages() {
    // DirichletL::invl / DirichletL::intp, mirroring DirichletCharacter
    assert_case(r#"DirichletL[5, 9, 1]"#, r#"DirichletL[5, 9, 1]"#);
    assert_case(r#"DirichletL[5, 0, 1]"#, r#"DirichletL[5, 0, 1]"#);
  }
}

mod dirichlet_convolve {
  use super::*;

  #[test]
  fn numeric_modulus_expands_divisor_sum() {
    // Sum over divisors d of m of f(d) g(m/d)
    assert_eq!(interpret("DirichletConvolve[n, n, n, 6]").unwrap(), "24");
    assert_eq!(interpret("DirichletConvolve[1, 1, n, 12]").unwrap(), "6");
    assert_eq!(interpret("DirichletConvolve[n, 1, n, 12]").unwrap(), "28");
    assert_eq!(
      interpret("DirichletConvolve[MoebiusMu[n], n, n, 12]").unwrap(),
      "4"
    );
  }

  #[test]
  fn numeric_modulus_with_unknown_functions() {
    assert_eq!(
      interpret("DirichletConvolve[f[n], g[n], n, 4]").unwrap(),
      "f[4]*g[1] + f[2]*g[2] + f[1]*g[4]"
    );
    assert_eq!(
      interpret("DirichletConvolve[f[n], g[n], n, 1]").unwrap(),
      "f[1]*g[1]"
    );
  }

  #[test]
  fn symbolic_power_identities() {
    // n^j * n^k -> m^min(j,k) DivisorSigma[|j-k|, m]
    assert_eq!(
      interpret("DirichletConvolve[1, 1, n, m]").unwrap(),
      "DivisorSigma[0, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[n, 1, n, m]").unwrap(),
      "DivisorSigma[1, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[n^2, 1, n, m]").unwrap(),
      "DivisorSigma[2, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[n, n, n, m]").unwrap(),
      "m*DivisorSigma[0, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[n^2, n, n, m]").unwrap(),
      "m*DivisorSigma[1, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[n^3, n, n, m]").unwrap(),
      "m*DivisorSigma[2, m]"
    );
  }

  #[test]
  fn symbolic_moebius_and_totient_identities() {
    assert_eq!(
      interpret("DirichletConvolve[MoebiusMu[n], n, n, m]").unwrap(),
      "EulerPhi[m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[MoebiusMu[n], 1, n, m]").unwrap(),
      "KroneckerDelta[1 - m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[1, MoebiusMu[n], n, m]").unwrap(),
      "KroneckerDelta[1 - m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[EulerPhi[n], 1, n, m]").unwrap(),
      "m"
    );
  }

  #[test]
  fn linearity_and_coefficients() {
    assert_eq!(
      interpret("DirichletConvolve[n + 1, 1, n, m]").unwrap(),
      "DivisorSigma[0, m] + DivisorSigma[1, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[2, 3, n, m]").unwrap(),
      "6*DivisorSigma[0, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[2 n, 1, n, m]").unwrap(),
      "2*DivisorSigma[1, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[a n, 1, n, m]").unwrap(),
      "a*DivisorSigma[1, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[2 MoebiusMu[n], 1, n, m]").unwrap(),
      "2*KroneckerDelta[1 - m]"
    );
    // Variable not appearing in f, g: both are constants
    assert_eq!(
      interpret("DirichletConvolve[n, n, k, m]").unwrap(),
      "n^2*DivisorSigma[0, m]"
    );
  }

  #[test]
  fn unit_second_argument_rewrites_to_divisor_sum() {
    // g == 1 with irreducible f: whole f becomes a DivisorSum body
    assert_eq!(
      interpret("DirichletConvolve[f[n], 1, n, m]").unwrap(),
      "DivisorSum[m, f[#1] & ]"
    );
    assert_eq!(
      interpret("DirichletConvolve[f[n] + n, 1, n, m]").unwrap(),
      "DivisorSum[m, f[#1] + #1 & ]"
    );
    assert_eq!(
      interpret("DirichletConvolve[n f[n], 1, n, m]").unwrap(),
      "DivisorSum[m, f[#1]*#1 & ]"
    );
    assert_eq!(
      interpret("DirichletConvolve[f[n] + g[n], 1, n, m]").unwrap(),
      "DivisorSum[m, f[#1] + g[#1] & ]"
    );
  }

  #[test]
  fn unit_first_argument_splits_linearly() {
    // f == 1 splits g term-by-term (asymmetric with the g == 1 path)
    assert_eq!(
      interpret("DirichletConvolve[1, f[n], n, m]").unwrap(),
      "DivisorSum[m, f[#1] & ]"
    );
    assert_eq!(
      interpret("DirichletConvolve[1, f[n] + n, n, m]").unwrap(),
      "DivisorSigma[1, m] + DivisorSum[m, f[#1] & ]"
    );
  }

  #[test]
  fn partial_reduction_keeps_inert_pairs() {
    assert_eq!(
      interpret("DirichletConvolve[n + MoebiusMu[n], MoebiusMu[n], n, m]")
        .unwrap(),
      "DirichletConvolve[MoebiusMu[n], MoebiusMu[n], n, m] + EulerPhi[m]"
    );
    // A sum in f commutes to the second position before splitting
    assert_eq!(
      interpret("DirichletConvolve[f[n] + n, MoebiusMu[n], n, m]").unwrap(),
      "DirichletConvolve[MoebiusMu[n], f[n], n, m] + EulerPhi[m]"
    );
  }

  #[test]
  fn irreducible_cases_stay_unevaluated() {
    assert_eq!(
      interpret("DirichletConvolve[f[n], g[n], n, m]").unwrap(),
      "DirichletConvolve[f[n], g[n], n, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[MoebiusMu[n], MoebiusMu[n], n, m]").unwrap(),
      "DirichletConvolve[MoebiusMu[n], MoebiusMu[n], n, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[MoebiusMu[n], n^2, n, m]").unwrap(),
      "DirichletConvolve[MoebiusMu[n], n^2, n, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[f[n], EulerPhi[n], n, m]").unwrap(),
      "DirichletConvolve[f[n], EulerPhi[n], n, m]"
    );
    assert_eq!(
      interpret("DirichletConvolve[1, 1, n, 2.5]").unwrap(),
      "DirichletConvolve[1, 1, n, 2.5]"
    );
  }

  #[test]
  fn symbolic_compound_modulus() {
    assert_eq!(
      interpret("DirichletConvolve[1, 1, n, 2 m]").unwrap(),
      "DivisorSigma[0, 2*m]"
    );
  }

  #[test]
  fn wrong_arg_count_emits_argrx() {
    assert_eq!(
      interpret("DirichletConvolve[1, 1, n]").unwrap(),
      "DirichletConvolve[1, 1, n]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "DirichletConvolve::argrx: DirichletConvolve called with 3 arguments; 4 arguments are expected."
      )),
      "expected argrx message, got {:?}",
      msgs
    );
  }

  #[test]
  fn moebius_mu_table_via_convolution() {
    assert_eq!(
      interpret("Table[DirichletConvolve[MoebiusMu[n], 1, n, m], {m, 1, 8}]")
        .unwrap(),
      "{1, 0, 0, 0, 0, 0, 0, 0}"
    );
  }
}

mod moebius_mu_symbolic {
  use super::*;

  #[test]
  fn symbolic_argument_stays_unevaluated() {
    // Regression: this used to be a hard evaluation error
    assert_eq!(interpret("MoebiusMu[n]").unwrap(), "MoebiusMu[n]");
    assert_eq!(interpret("MoebiusMu[2.5]").unwrap(), "MoebiusMu[2.5]");
  }

  #[test]
  fn zero_and_negative_arguments() {
    assert_eq!(interpret("MoebiusMu[0]").unwrap(), "0");
    // Negative arguments use the absolute value
    assert_eq!(interpret("MoebiusMu[-6]").unwrap(), "1");
    assert_eq!(interpret("MoebiusMu[-4]").unwrap(), "0");
    assert_eq!(interpret("MoebiusMu[-5]").unwrap(), "-1");
  }
}

mod cyclic_group_and_cycle_index {
  use super::*;

  #[test]
  fn cyclic_group_is_inert_and_consumed() {
    assert_eq!(interpret("CyclicGroup[4]").unwrap(), "CyclicGroup[4]");
    assert_eq!(interpret("GroupOrder[CyclicGroup[6]]").unwrap(), "6");
    assert_eq!(
      interpret("GroupGenerators[CyclicGroup[4]]").unwrap(),
      "{Cycles[{{1, 2, 3, 4}}]}"
    );
    assert_eq!(interpret("GroupGenerators[CyclicGroup[1]]").unwrap(), "{}");
  }

  #[test]
  fn cyclic_group_elements() {
    assert_eq!(
      interpret("GroupElements[CyclicGroup[4]]").unwrap(),
      "{Cycles[{}], Cycles[{{1, 2, 3, 4}}], Cycles[{{1, 3}, {2, 4}}], Cycles[{{1, 4, 3, 2}}]}"
    );
    assert_eq!(
      interpret("GroupElements[CyclicGroup[1]]").unwrap(),
      "{Cycles[{}]}"
    );
    assert_eq!(
      interpret("GroupElements[CyclicGroup[0]]").unwrap(),
      "{Cycles[{}]}"
    );
  }

  #[test]
  fn symmetric_group_elements() {
    // Lexicographic image-list order, matching wolframscript
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[3]]").unwrap(),
      "{Cycles[{}], Cycles[{{2, 3}}], Cycles[{{1, 2}}], Cycles[{{1, 2, 3}}], Cycles[{{1, 3, 2}}], Cycles[{{1, 3}}]}"
    );
    assert_eq!(
      interpret("GroupElements[SymmetricGroup[0]]").unwrap(),
      "{Cycles[{}]}"
    );
  }

  #[test]
  fn cycle_index_named_groups() {
    assert_eq!(
      interpret(
        "CycleIndexPolynomial[CyclicGroup[4], {x[1], x[2], x[3], x[4]}]"
      )
      .unwrap(),
      "x[1]^4/4 + x[2]^2/4 + x[4]/2"
    );
    assert_eq!(
      interpret("CycleIndexPolynomial[SymmetricGroup[3], {a, b, c}]").unwrap(),
      "a^3/6 + (a*b)/2 + c/3"
    );
    assert_eq!(
      interpret("CycleIndexPolynomial[DihedralGroup[4], {a, b, c, d}]")
        .unwrap(),
      "a^4/8 + (a^2*b)/4 + (3*b^2)/8 + d/4"
    );
    assert_eq!(
      interpret("CycleIndexPolynomial[AlternatingGroup[4], {a, b, c, d}]")
        .unwrap(),
      "a^4/12 + b^2/4 + (2*a*c)/3"
    );
    assert_eq!(
      interpret("CycleIndexPolynomial[SymmetricGroup[4], {a, b, c, d}]")
        .unwrap(),
      "a^4/24 + (a^2*b)/4 + b^2/8 + (a*c)/3 + d/4"
    );
    assert_eq!(
      interpret("CycleIndexPolynomial[AbelianGroup[{2, 2}], {a, b}]").unwrap(),
      "a^4/4 + (a^2*b)/2 + b^2/4"
    );
  }

  #[test]
  fn cycle_index_permutation_group_closure() {
    assert_eq!(
      interpret(
        "CycleIndexPolynomial[PermutationGroup[{Cycles[{{1, 2}}]}], {a, b}]"
      )
      .unwrap(),
      "a^2/2 + b/2"
    );
    // Generators of S3
    assert_eq!(
      interpret(
        "CycleIndexPolynomial[PermutationGroup[{Cycles[{{1, 2, 3}}], Cycles[{{1, 2}}]}], {a, b, c}]"
      )
      .unwrap(),
      "a^3/6 + (a*b)/2 + c/3"
    );
  }

  #[test]
  fn cycle_index_short_vars_become_one() {
    assert_eq!(
      interpret("CycleIndexPolynomial[SymmetricGroup[3], {a, b}]").unwrap(),
      "1/3 + a^3/6 + (a*b)/2"
    );
  }

  #[test]
  fn cycle_index_indexed_variable_ordering() {
    // Regression: products of indexed variables previously sorted after
    // single powers in Plus (x[1]^2*x[2]^2 came after x[3]^2)
    assert_eq!(
      interpret(
        "CycleIndexPolynomial[DihedralGroup[6], {x[1], x[2], x[3], x[4], x[5], x[6]}]"
      )
      .unwrap(),
      "x[1]^6/12 + (x[1]^2*x[2]^2)/4 + x[2]^3/3 + x[3]^2/6 + x[6]/6"
    );
    // Indexed-variable arguments compare numerically: x[5] before x[10]
    assert_eq!(
      interpret("CycleIndexPolynomial[CyclicGroup[10], Array[x, 10]]").unwrap(),
      "x[1]^10/10 + x[2]^5/10 + (2*x[5]^2)/5 + (2*x[10])/5"
    );
    assert_eq!(
      interpret("x[1]^6 + x[2]^3 + x[1]^2 x[2]^2").unwrap(),
      "x[1]^6 + x[1]^2*x[2]^2 + x[2]^3"
    );
    // Plain symbol names stay lexicographic (x10 before x2)
    assert_eq!(interpret("x2 + x10").unwrap(), "x10 + x2");
  }

  #[test]
  fn cycle_index_error_messages() {
    assert_eq!(
      interpret("CycleIndexPolynomial[x, {a, b}]").unwrap(),
      "CycleIndexPolynomial[x, {a, b}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(
        |m| m.contains("CycleIndexPolynomial::grp: x is not a valid group.")
      ),
      "expected grp message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("CycleIndexPolynomial[CyclicGroup[3], x]").unwrap(),
      "CycleIndexPolynomial[CyclicGroup[3], x]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "CycleIndexPolynomial::list: List expected at position 2 in CycleIndexPolynomial[CyclicGroup[3], x]."
      )),
      "expected list message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("CycleIndexPolynomial[CyclicGroup[3]]").unwrap(),
      "CycleIndexPolynomial[CyclicGroup[3]]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "CycleIndexPolynomial::argtu: CycleIndexPolynomial called with 1 argument; 2 or 3 arguments are expected."
      )),
      "expected argtu message, got {:?}",
      msgs
    );
  }
}

mod group_stabilizer_and_table {
  use super::*;

  #[test]
  fn symmetric_group_stabilizers() {
    // Canonical generators on the remaining points: a transposition of
    // the first two plus the full cycle
    assert_eq!(
      interpret("GroupStabilizer[SymmetricGroup[4], {1}]").unwrap(),
      "PermutationGroup[{Cycles[{{2, 3}}], Cycles[{{2, 3, 4}}]}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[SymmetricGroup[4], {2}]").unwrap(),
      "PermutationGroup[{Cycles[{{1, 3}}], Cycles[{{1, 3, 4}}]}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[SymmetricGroup[4], {1, 2}]").unwrap(),
      "PermutationGroup[{Cycles[{{3, 4}}]}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[SymmetricGroup[5], {1}]").unwrap(),
      "PermutationGroup[{Cycles[{{2, 3}}], Cycles[{{2, 3, 4, 5}}]}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[SymmetricGroup[3], {1, 2, 3}]").unwrap(),
      "PermutationGroup[{}]"
    );
  }

  #[test]
  fn alternating_group_stabilizers() {
    // Consecutive 3-cycles on the remaining points
    assert_eq!(
      interpret("GroupStabilizer[AlternatingGroup[4], {1}]").unwrap(),
      "PermutationGroup[{Cycles[{{2, 3, 4}}]}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[AlternatingGroup[5], {1}]").unwrap(),
      "PermutationGroup[{Cycles[{{2, 3, 4}}], Cycles[{{3, 4, 5}}]}]"
    );
  }

  #[test]
  fn cyclic_and_dihedral_stabilizers() {
    assert_eq!(
      interpret("GroupStabilizer[CyclicGroup[6], {2}]").unwrap(),
      "PermutationGroup[{}]"
    );
    // The reflection through the fixed vertex
    assert_eq!(
      interpret("GroupStabilizer[DihedralGroup[4], {1}]").unwrap(),
      "PermutationGroup[{Cycles[{{2, 4}}]}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[DihedralGroup[6], {1}]").unwrap(),
      "PermutationGroup[{Cycles[{{2, 6}, {3, 5}}]}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[DihedralGroup[4], {1, 3}]").unwrap(),
      "PermutationGroup[{Cycles[{{2, 4}}]}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[DihedralGroup[4], {1, 2}]").unwrap(),
      "PermutationGroup[{}]"
    );
  }

  #[test]
  fn stabilizer_edge_cases() {
    // Empty point list: the whole group
    assert_eq!(
      interpret("GroupStabilizer[SymmetricGroup[4], {}]").unwrap(),
      "SymmetricGroup[4]"
    );
    // Forced (at most one nontrivial element) PermutationGroup stabilizers
    assert_eq!(
      interpret(
        "GroupStabilizer[PermutationGroup[{Cycles[{{1, 2, 3}}], Cycles[{{1, 2}}]}], {3}]"
      )
      .unwrap(),
      "PermutationGroup[{Cycles[{{1, 2}}]}]"
    );
    assert_eq!(
      interpret(
        "GroupStabilizer[PermutationGroup[{Cycles[{{1, 2, 3}}]}], {1}]"
      )
      .unwrap(),
      "PermutationGroup[{}]"
    );
    assert_eq!(
      interpret("GroupStabilizer[x, {1}]").unwrap(),
      "GroupStabilizer[x, {1}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs
        .iter()
        .any(|m| m.contains("GroupStabilizer::grp: x is not a valid group.")),
      "expected grp message, got {:?}",
      msgs
    );
  }

  #[test]
  fn multiplication_tables() {
    assert_eq!(
      interpret("GroupMultiplicationTable[CyclicGroup[4]]").unwrap(),
      "{{1, 2, 3, 4}, {2, 3, 4, 1}, {3, 4, 1, 2}, {4, 1, 2, 3}}"
    );
    assert_eq!(
      interpret("GroupMultiplicationTable[SymmetricGroup[3]]").unwrap(),
      "{{1, 2, 3, 4, 5, 6}, {2, 1, 4, 3, 6, 5}, {3, 5, 1, 6, 2, 4}, {4, 6, 2, 5, 1, 3}, {5, 3, 6, 1, 4, 2}, {6, 4, 5, 2, 3, 1}}"
    );
    // DihedralGroup[3] is isomorphic to S3 with the same element order
    assert_eq!(
      interpret("GroupMultiplicationTable[DihedralGroup[3]]").unwrap(),
      "{{1, 2, 3, 4, 5, 6}, {2, 1, 4, 3, 6, 5}, {3, 5, 1, 6, 2, 4}, {4, 6, 2, 5, 1, 3}, {5, 3, 6, 1, 4, 2}, {6, 4, 5, 2, 3, 1}}"
    );
    assert_eq!(
      interpret(
        "GroupMultiplicationTable[PermutationGroup[{Cycles[{{1, 2}}]}]]"
      )
      .unwrap(),
      "{{1, 2}, {2, 1}}"
    );
    assert_eq!(
      interpret("GroupMultiplicationTable[CyclicGroup[1]]").unwrap(),
      "{{1}}"
    );
    assert_eq!(
      interpret("GroupMultiplicationTable[x]").unwrap(),
      "GroupMultiplicationTable[x]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m
        .contains("GroupMultiplicationTable::grp: x is not a valid group.")),
      "expected grp message, got {:?}",
      msgs
    );
  }
}

mod integer_digits_string_messages {
  use super::*;

  #[test]
  fn invalid_bases_emit_ibase() {
    // Regression: bases below 2 raised hard errors
    assert_eq!(
      interpret("IntegerDigits[10, 1]").unwrap(),
      "IntegerDigits[10, 1]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "IntegerDigits::ibase: Base 1 is not an integer greater than 1."
      )),
      "expected ibase message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("IntegerDigits[10, 0]").unwrap(),
      "IntegerDigits[10, 0]"
    );
    assert_eq!(
      interpret("IntegerDigits[10, 2.5]").unwrap(),
      "IntegerDigits[10, 2.5]"
    );
    // Symbolic bases stay silently unevaluated
    assert_eq!(
      interpret("IntegerDigits[10, x]").unwrap(),
      "IntegerDigits[10, x]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.is_empty(),
      "symbolic base must stay silent, got {:?}",
      msgs
    );
  }

  #[test]
  fn non_integer_subjects_emit_int() {
    // Regression: silent unevaluated return without the ::int message
    assert_eq!(
      interpret("IntegerDigits[2.5]").unwrap(),
      "IntegerDigits[2.5]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "IntegerDigits::int: Integer expected at position 1 in IntegerDigits[2.5]."
      )),
      "expected int message, got {:?}",
      msgs
    );
    // Integral reals are still not integers
    assert_eq!(
      interpret("IntegerDigits[2.0]").unwrap(),
      "IntegerDigits[2.]"
    );
    assert_eq!(
      interpret("IntegerString[2.0]").unwrap(),
      "IntegerString[2.]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "IntegerString::int: Integer expected at position 1 in IntegerString[2.]."
      )),
      "expected int message, got {:?}",
      msgs
    );
  }

  #[test]
  fn invalid_lengths_emit_intnm() {
    // Regression: negative lengths raised hard errors
    assert_eq!(
      interpret("IntegerDigits[5, 2, -1]").unwrap(),
      "IntegerDigits[5, 2, -1]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "IntegerDigits::intnm: Non-negative machine-sized integer expected at position 3 in IntegerDigits[5, 2, -1]."
      )),
      "expected intnm message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("IntegerDigits[5, 2, 2.5]").unwrap(),
      "IntegerDigits[5, 2, 2.5]"
    );
    // Symbolic lengths stay silently unevaluated
    assert_eq!(
      interpret("IntegerDigits[5, 2, m]").unwrap(),
      "IntegerDigits[5, 2, m]"
    );
  }

  #[test]
  fn integer_string_invalid_bases_emit_basf() {
    // Regression: bases outside 2..36 raised hard errors
    assert_eq!(
      interpret("IntegerString[255, 1]").unwrap(),
      "IntegerString[255, 1]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "IntegerString::basf: Requested base 1 should be an integer between 2 and 36."
      )),
      "expected basf message, got {:?}",
      msgs
    );
    assert_eq!(
      interpret("IntegerString[255, 40]").unwrap(),
      "IntegerString[255, 40]"
    );
    // Symbolic arguments stay silently unevaluated
    assert_eq!(
      interpret("IntegerString[255, x]").unwrap(),
      "IntegerString[255, x]"
    );
    assert_eq!(interpret("IntegerString[n]").unwrap(), "IntegerString[n]");
    // Valid forms still work
    assert_eq!(interpret("IntegerString[255, 16]").unwrap(), "ff");
    assert_eq!(interpret("IntegerDigits[255, 16]").unwrap(), "{15, 15}");
  }
}
