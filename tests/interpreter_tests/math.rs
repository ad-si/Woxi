use super::*;

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
  fn real_argument() {
    assert_eq!(
      interpret("HarmonicNumber[3.8]").unwrap(),
      "2.0380634056306492"
    );
  }
}

mod coefficient_list {
  use super::*;

  #[test]
  fn basic_polynomial() {
    assert_eq!(
      interpret("CoefficientList[x^2 + 3 x + 1, x]").unwrap(),
      "{1, 3, 1}"
    );
  }

  #[test]
  fn expanded_polynomial() {
    assert_eq!(
      interpret("CoefficientList[(x + 1)^3, x]").unwrap(),
      "{1, 3, 3, 1}"
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
    assert_eq!(interpret("PrimeQ[-7]").unwrap(), "True");
    assert_eq!(interpret("PrimeQ[-2]").unwrap(), "True");
    assert_eq!(interpret("PrimeQ[-1]").unwrap(), "False");
    assert_eq!(interpret("PrimeQ[-4]").unwrap(), "False");
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
  fn bit_not() {
    assert_eq!(interpret("BitNot[5]").unwrap(), "-6");
    assert_eq!(interpret("BitNot[0]").unwrap(), "-1");
  }
}

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
}

mod sign_predicates {
  use super::*;

  #[test]
  fn positive() {
    assert_eq!(interpret("Positive[5]").unwrap(), "True");
    assert_eq!(interpret("Positive[-3]").unwrap(), "False");
    assert_eq!(interpret("Positive[0]").unwrap(), "False");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("Negative[-5]").unwrap(), "True");
    assert_eq!(interpret("Negative[3]").unwrap(), "False");
    assert_eq!(interpret("Negative[0]").unwrap(), "False");
  }

  #[test]
  fn non_positive() {
    assert_eq!(interpret("NonPositive[-5]").unwrap(), "True");
    assert_eq!(interpret("NonPositive[0]").unwrap(), "True");
    assert_eq!(interpret("NonPositive[3]").unwrap(), "False");
  }

  #[test]
  fn non_negative() {
    assert_eq!(interpret("NonNegative[5]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[0]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[-3]").unwrap(), "False");
  }

  #[test]
  fn positive_constants() {
    assert_eq!(interpret("Positive[Pi]").unwrap(), "True");
    assert_eq!(interpret("Positive[E]").unwrap(), "True");
    assert_eq!(interpret("Positive[Infinity]").unwrap(), "True");
    assert_eq!(interpret("Positive[-Pi]").unwrap(), "False");
    assert_eq!(interpret("Positive[-E]").unwrap(), "False");
  }

  #[test]
  fn negative_constants() {
    assert_eq!(interpret("Negative[-Pi]").unwrap(), "True");
    assert_eq!(interpret("Negative[-E]").unwrap(), "True");
    assert_eq!(interpret("Negative[Pi]").unwrap(), "False");
    assert_eq!(interpret("Negative[E]").unwrap(), "False");
    assert_eq!(interpret("Negative[Infinity]").unwrap(), "False");
  }

  #[test]
  fn non_positive_constants() {
    assert_eq!(interpret("NonPositive[-Pi]").unwrap(), "True");
    assert_eq!(interpret("NonPositive[Pi]").unwrap(), "False");
  }

  #[test]
  fn non_negative_constants() {
    assert_eq!(interpret("NonNegative[Pi]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[-Pi]").unwrap(), "False");
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

mod chop {
  use super::*;

  #[test]
  fn small_number() {
    assert_eq!(interpret("Chop[0.00000000001]").unwrap(), "0");
  }

  #[test]
  fn normal_number() {
    assert_eq!(interpret("Chop[1.5]").unwrap(), "1.5");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("Chop[0]").unwrap(), "0");
  }

  #[test]
  fn negative_small() {
    assert_eq!(interpret("Chop[-0.00000000001]").unwrap(), "0");
  }

  #[test]
  fn list() {
    assert_eq!(
      interpret("Chop[{0.00000000001, 1.5, -0.000000000001}]").unwrap(),
      "{0, 1.5, 0}"
    );
  }

  #[test]
  fn custom_tolerance() {
    assert_eq!(interpret("Chop[0.05, 0.1]").unwrap(), "0");
    assert_eq!(interpret("Chop[0.5, 0.1]").unwrap(), "0.5");
  }
}

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

mod symbolic_product {
  use super::*;

  #[test]
  fn product_i_squared_symbolic() {
    // Product[i^2, {i, 1, n}] = n!^2
    assert_eq!(interpret("Product[i^2, {i, 1, n}]").unwrap(), "n!^2");
  }

  #[test]
  fn product_i_symbolic() {
    // Product[i, {i, 1, n}] = n!
    assert_eq!(interpret("Product[i, {i, 1, n}]").unwrap(), "n!");
  }

  #[test]
  fn product_numeric() {
    // Numeric bounds should still work
    assert_eq!(interpret("Product[i^2, {i, 1, 6}]").unwrap(), "518400");
  }

  #[test]
  fn factorial_formatting() {
    // Factorial[n] should display as n!
    assert_eq!(interpret("Factorial[n]").unwrap(), "n!");
    assert_eq!(interpret("Factorial[5]").unwrap(), "120");
  }

  #[test]
  fn factorial_suffix_syntax() {
    // n! suffix syntax should parse as Factorial[n]
    assert_eq!(interpret("Factorial[5]").unwrap(), "120");
    assert_eq!(
      interpret("Factorial[50]").unwrap(),
      "30414093201713378043612608166064768844377641568960512000000000000"
    );
  }

  #[test]
  fn factorial_with_arithmetic() {
    // 2 + 3! should be 2 + 6 = 8 (factorial binds tighter)
    assert_eq!(interpret("2 + Factorial[3]").unwrap(), "8");
  }

  #[test]
  fn product_symbolic_lower_bound() {
    assert_eq!(
      interpret("Product[k, {k, i, n}]").unwrap(),
      "Pochhammer[i, 1 - i + n]"
    );
  }

  #[test]
  fn product_concrete_lower_symbolic_upper() {
    assert_eq!(interpret("Product[k, {k, 3, n}]").unwrap(), "n!/2");
  }

  #[test]
  fn product_power_symbolic() {
    assert_eq!(
      interpret("Product[2 ^ i, {i, 1, n}]").unwrap(),
      "2^((n*(1 + n))/2)"
    );
  }
}

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

mod sum {
  use super::*;

  #[test]
  fn finite_sum_integers() {
    assert_eq!(interpret("Sum[i, {i, 1, 10}]").unwrap(), "55");
  }

  #[test]
  fn finite_sum_squares() {
    assert_eq!(interpret("Sum[i^2, {i, 1, 5}]").unwrap(), "55");
  }

  #[test]
  fn finite_sum_with_explicit_min() {
    assert_eq!(interpret("Sum[i, {i, 5, 10}]").unwrap(), "45");
  }

  #[test]
  fn infinite_sum_zeta_2() {
    assert_eq!(interpret("Sum[1/n^2, {n, 1, Infinity}]").unwrap(), "Pi^2/6");
  }

  #[test]
  fn infinite_sum_zeta_4() {
    assert_eq!(
      interpret("Sum[1/n^4, {n, 1, Infinity}]").unwrap(),
      "Pi^4/90"
    );
  }

  #[test]
  fn infinite_sum_zeta_6() {
    assert_eq!(
      interpret("Sum[1/n^6, {n, 1, Infinity}]").unwrap(),
      "Pi^6/945"
    );
  }

  #[test]
  fn infinite_sum_zeta_8() {
    assert_eq!(
      interpret("Sum[1/n^8, {n, 1, Infinity}]").unwrap(),
      "Pi^8/9450"
    );
  }

  #[test]
  fn infinite_sum_zeta_10() {
    assert_eq!(
      interpret("Sum[1/n^10, {n, 1, Infinity}]").unwrap(),
      "Pi^10/93555"
    );
  }

  #[test]
  fn infinite_sum_zeta_12() {
    assert_eq!(
      interpret("Sum[1/n^12, {n, 1, Infinity}]").unwrap(),
      "(691*Pi^12)/638512875"
    );
  }

  #[test]
  fn infinite_sum_zeta_odd_returns_zeta() {
    assert_eq!(
      interpret("Sum[1/n^3, {n, 1, Infinity}]").unwrap(),
      "Zeta[3]"
    );
    assert_eq!(
      interpret("Sum[1/n^5, {n, 1, Infinity}]").unwrap(),
      "Zeta[5]"
    );
  }

  #[test]
  fn infinite_sum_negative_power_form() {
    // n^(-2) form should also work
    assert_eq!(
      interpret("Sum[n^(-2), {n, 1, Infinity}]").unwrap(),
      "Pi^2/6"
    );
  }

  #[test]
  fn infinite_sum_harmonic_unevaluated() {
    // Sum[1/n, {n, 1, Infinity}] diverges — should return unevaluated
    assert_eq!(
      interpret("Sum[1/n, {n, 1, Infinity}]").unwrap(),
      "Sum[1/n, {n, 1, Infinity}]"
    );
  }

  #[test]
  fn sum_with_step() {
    // Sum[i, {i, 1, 10, 2}] -> 1 + 3 + 5 + 7 + 9 = 25
    assert_eq!(interpret("Sum[i, {i, 1, 10, 2}]").unwrap(), "25");
  }

  #[test]
  fn sum_with_step_3() {
    // Sum[i, {i, 1, 10, 3}] -> 1 + 4 + 7 + 10 = 22
    assert_eq!(interpret("Sum[i, {i, 1, 10, 3}]").unwrap(), "22");
  }

  #[test]
  fn sum_with_negative_step() {
    // Sum[i, {i, 10, 1, -1}] -> 10 + 9 + ... + 1 = 55
    assert_eq!(interpret("Sum[i, {i, 10, 1, -1}]").unwrap(), "55");
  }

  #[test]
  fn sum_multi_dimensional() {
    // Sum[i*j, {i, 1, 2}, {j, 1, 3}] = 1*1+1*2+1*3+2*1+2*2+2*3 = 18
    assert_eq!(interpret("Sum[i*j, {i, 1, 2}, {j, 1, 3}]").unwrap(), "18");
  }

  #[test]
  fn sum_multi_dimensional_addition() {
    // Sum[i + j, {i, 1, 3}, {j, 1, 2}] = 21
    assert_eq!(interpret("Sum[i + j, {i, 1, 3}, {j, 1, 2}]").unwrap(), "21");
  }

  #[test]
  fn sum_explicit_list() {
    // Sum[x^i, {i, {1, 3, 5}}] = x + x^3 + x^5
    assert_eq!(
      interpret("Sum[x^i, {i, {1, 3, 5}}]").unwrap(),
      "x + x^3 + x^5"
    );
  }

  #[test]
  fn sum_single_arg_unevaluated() {
    // Sum[{a, b, c}] is invalid in Wolfram — requires 2+ args
    assert_eq!(interpret("Sum[{a, b, c}]").unwrap(), "Sum[{a, b, c}]");
  }

  #[test]
  fn sum_k_symbolic() {
    assert_eq!(interpret("Sum[k, {k, 1, n}]").unwrap(), "(n*(1 + n))/2");
  }

  #[test]
  fn sum_geometric_symbolic() {
    assert_eq!(
      interpret("Sum[1 / 2 ^ i, {i, 1, k}]").unwrap(),
      "(-1 + 2^k)/2^k"
    );
  }

  #[test]
  fn sum_geometric_infinite() {
    assert_eq!(interpret("Sum[1 / 2 ^ i, {i, 1, Infinity}]").unwrap(), "1");
  }

  #[test]
  fn sum_k_symbolic_both_bounds() {
    assert_eq!(interpret("Sum[k, {k, n, 2 n}]").unwrap(), "(3*n*(1 + n))/2");
  }

  #[test]
  fn sum_real_upper_bound() {
    assert_eq!(interpret("Sum[i, {i, 1, 2.5}]").unwrap(), "3");
  }

  #[test]
  fn sum_real_both_bounds() {
    assert_eq!(interpret("Sum[i, {i, 1.1, 2.5}]").unwrap(), "3.2");
  }

  #[test]
  fn sum_squared_symbolic_identity() {
    assert_eq!(
      interpret("Sum[x ^ 2, {x, 1, y}] - y * (y + 1) * (2 * y + 1) / 6")
        .unwrap(),
      "0"
    );
  }

  #[test]
  fn sum_harmonic_number() {
    assert_eq!(
      interpret("Sum[1 / k ^ 2, {k, 1, n}]").unwrap(),
      "HarmonicNumber[n, 2]"
    );
  }

  #[test]
  fn sum_leibniz_formula() {
    assert_eq!(
      interpret("Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]").unwrap(),
      "Pi/4"
    );
  }

  #[test]
  fn sum_complex_bounds() {
    assert_eq!(interpret("Sum[k, {k, I, I + 1}]").unwrap(), "1 + 2*I");
  }

  #[test]
  fn sum_complex_bounds_real_range() {
    assert_eq!(interpret("Sum[k, {k, I, I + 1.5}]").unwrap(), "1 + 2*I");
  }
}

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
}

mod fixed_point {
  use super::*;

  #[test]
  fn fixed_point_cos() {
    assert_eq!(
      interpret("FixedPoint[Cos, 1.0]").unwrap(),
      "0.7390851332151607"
    );
  }

  #[test]
  fn fixed_point_sqrt2_newton() {
    // Newton's method for sqrt(2): f(x) = (x + 2/x) / 2
    assert_eq!(
      interpret("FixedPoint[N[(# + 2/#)/2] &, 1.]").unwrap(),
      "1.414213562373095"
    );
  }

  #[test]
  fn fixed_point_identity() {
    // FixedPoint on a value that's already a fixed point
    assert_eq!(interpret("FixedPoint[# &, 5]").unwrap(), "5");
  }

  #[test]
  fn fixed_point_with_max_iterations() {
    // FixedPoint with explicit max iterations
    assert_eq!(
      interpret("FixedPoint[Cos, 1.0, 100]").unwrap(),
      "0.7390851332151607"
    );
  }

  #[test]
  fn fixed_point_floor_halving() {
    // Floor[#/2]& converges to 0
    assert_eq!(interpret("FixedPoint[Floor[#/2] &, 100]").unwrap(), "0");
  }

  #[test]
  fn fixed_point_list_collatz() {
    // Regression test: SetDelayed with literal arg (collatz[1] := 1) must
    // take priority over general pattern (collatz[x_] := 3 x + 1)
    assert_eq!(
      interpret(
        "collatz[1] := 1; collatz[x_ ? EvenQ] := x / 2; collatz[x_] := 3 x + 1; FixedPointList[collatz, 14]"
      )
      .unwrap(),
      "{14, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 1}"
    );
  }
}

mod subdivide {
  use super::*;

  #[test]
  fn unit_interval() {
    assert_eq!(interpret("Subdivide[4]").unwrap(), "{0, 1/4, 1/2, 3/4, 1}");
  }

  #[test]
  fn two_parts() {
    assert_eq!(interpret("Subdivide[2]").unwrap(), "{0, 1/2, 1}");
  }

  #[test]
  fn one_part() {
    assert_eq!(interpret("Subdivide[1]").unwrap(), "{0, 1}");
  }

  #[test]
  fn custom_range() {
    assert_eq!(
      interpret("Subdivide[0, 10, 5]").unwrap(),
      "{0, 2, 4, 6, 8, 10}"
    );
  }

  #[test]
  fn two_arg_form() {
    assert_eq!(
      interpret("Subdivide[10, 5]").unwrap(),
      "{0, 2, 4, 6, 8, 10}"
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

mod composition {
  use super::*;

  #[test]
  fn composition_apply() {
    assert_eq!(
      interpret("Composition[StringLength, ToString][12345]").unwrap(),
      "5"
    );
  }

  #[test]
  fn composition_symbolic() {
    assert_eq!(interpret("Composition[f, g]").unwrap(), "f @* g");
  }

  #[test]
  fn composition_variable() {
    assert_eq!(
      interpret("f = Composition[StringLength, ToString]; f[12345]").unwrap(),
      "5"
    );
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
    assert_eq!(interpret("GCD[x, 5]").unwrap(), "GCD[x, 5]");
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
    assert_eq!(interpret("LCM[x, 5]").unwrap(), "LCM[x, 5]");
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
}

mod listable {
  use super::*;

  #[test]
  fn fibonacci_list() {
    assert_eq!(
      interpret("Fibonacci[{1, 2, 3, 4, 5, 6}]").unwrap(),
      "{1, 1, 2, 3, 5, 8}"
    );
  }

  #[test]
  fn sin_list() {
    assert_eq!(interpret("Sin[{0, Pi/2, Pi}]").unwrap(), "{0, 1, 0}");
  }

  #[test]
  fn power_list_scalar() {
    assert_eq!(interpret("Power[{2, 3, 4}, 2]").unwrap(), "{4, 9, 16}");
  }

  #[test]
  fn power_scalar_list() {
    assert_eq!(interpret("Power[2, {1, 2, 3}]").unwrap(), "{2, 4, 8}");
  }

  #[test]
  fn power_exponent_one_simplifies() {
    assert_eq!(interpret("x^1").unwrap(), "x");
    assert_eq!(interpret("Power[y, 1]").unwrap(), "y");
    assert_eq!(interpret("(a + b)^1").unwrap(), "a + b");
  }

  #[test]
  fn mod_basic() {
    assert_eq!(interpret("Mod[10, 3]").unwrap(), "1");
    assert_eq!(interpret("Mod[7, 4]").unwrap(), "3");
    assert_eq!(interpret("Mod[15, 5]").unwrap(), "0");
    assert_eq!(interpret("Mod[0, 5]").unwrap(), "0");
  }

  #[test]
  fn mod_negative_args() {
    assert_eq!(interpret("Mod[-10, 3]").unwrap(), "2");
    assert_eq!(interpret("Mod[-5, 3]").unwrap(), "1");
    assert_eq!(interpret("Mod[10, -3]").unwrap(), "-2");
    assert_eq!(interpret("Mod[-10, -3]").unwrap(), "-1");
    assert_eq!(interpret("Mod[-1, 3]").unwrap(), "2");
  }

  #[test]
  fn mod_rational() {
    assert_eq!(interpret("Mod[5/2, 1]").unwrap(), "1/2");
    assert_eq!(interpret("Mod[7/3, 2/3]").unwrap(), "1/3");
  }

  #[test]
  fn mod_float() {
    assert_eq!(interpret("Mod[5.5, 2]").unwrap(), "1.5");
    assert_eq!(interpret("Mod[7.5, 2]").unwrap(), "1.5");
  }

  #[test]
  fn mod_division_by_zero() {
    assert_eq!(interpret("Mod[10, 0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Mod[0, 0]").unwrap(), "Indeterminate");
  }

  #[test]
  fn mod_three_args() {
    assert_eq!(interpret("Mod[10, 3, 1]").unwrap(), "1");
    assert_eq!(interpret("Mod[10, 3, -1]").unwrap(), "1");
    assert_eq!(interpret("Mod[10, 3, 2]").unwrap(), "4");
    assert_eq!(interpret("Mod[-5, 3, 1]").unwrap(), "1");
    assert_eq!(interpret("Mod[10, 3, 0]").unwrap(), "1");
  }

  #[test]
  fn mod_three_args_rational() {
    assert_eq!(interpret("Mod[5/2, 3/2, 0]").unwrap(), "1");
    assert_eq!(interpret("Mod[5/2, 3/2, 1]").unwrap(), "1");
  }

  #[test]
  fn mod_symbolic() {
    assert_eq!(interpret("Mod[x, 3]").unwrap(), "Mod[x, 3]");
  }

  #[test]
  fn mod_list_scalar() {
    assert_eq!(interpret("Mod[{10, 20, 30}, 7]").unwrap(), "{3, 6, 2}");
  }

  #[test]
  fn plus_two_lists() {
    assert_eq!(
      interpret("Plus[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "{5, 7, 9}"
    );
  }

  #[test]
  fn plus_list_scalar() {
    assert_eq!(interpret("{1, 2, 3} + 10").unwrap(), "{11, 12, 13}");
  }

  #[test]
  fn times_two_lists() {
    assert_eq!(interpret("{1, 2, 3} * {4, 5, 6}").unwrap(), "{4, 10, 18}");
  }

  #[test]
  fn plus_three_lists() {
    assert_eq!(
      interpret("Plus[{1, 2}, {3, 4}, {5, 6}]").unwrap(),
      "{9, 12}"
    );
  }

  #[test]
  fn mismatched_lengths_no_thread() {
    // Mismatched list lengths should not thread — function returns unevaluated
    assert_eq!(interpret("Sin[{1, 2}] + Sin[{3, 4, 5}]").is_err(), true);
  }

  #[test]
  fn nested_listable() {
    assert_eq!(interpret("Abs[{-1, 2, -3, 4}]").unwrap(), "{1, 2, 3, 4}");
  }

  #[test]
  fn user_defined_listable() {
    assert_eq!(
      interpret("SetAttributes[f, Listable]; f[{1, 2, 3}]").unwrap(),
      "{f[1], f[2], f[3]}"
    );
  }

  #[test]
  fn evenq_list() {
    assert_eq!(
      interpret("EvenQ[{1, 2, 3, 4}]").unwrap(),
      "{False, True, False, True}"
    );
  }

  #[test]
  fn floor_list() {
    assert_eq!(interpret("Floor[{1.2, 2.7, 3.5}]").unwrap(), "{1, 2, 3}");
  }
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
    assert_eq!(interpret("Sin[x] Sin[3y]").unwrap(), "Sin[3*y]*Sin[x]");
  }

  #[test]
  fn function_call_implicit_times_evaluates() {
    assert_eq!(interpret("Sin[0] Sin[Pi/2]").unwrap(), "0");
  }
}

mod gudermannian {
  use super::*;

  #[test]
  fn zero() {
    assert_eq!(interpret("Gudermannian[0]").unwrap(), "0");
  }

  #[test]
  fn zero_float() {
    assert_eq!(interpret("Gudermannian[0.]").unwrap(), "0.");
  }

  #[test]
  fn numeric() {
    assert_eq!(
      interpret("Gudermannian[4.2]").unwrap(),
      "1.5408074208608435"
    );
  }

  #[test]
  fn infinity() {
    assert_eq!(interpret("Gudermannian[Infinity]").unwrap(), "Pi/2");
  }

  #[test]
  fn negative_infinity() {
    assert_eq!(interpret("Gudermannian[-Infinity]").unwrap(), "-1/2*Pi");
  }

  #[test]
  fn complex_infinity() {
    assert_eq!(
      interpret("Gudermannian[ComplexInfinity]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn undefined() {
    assert_eq!(interpret("Gudermannian[Undefined]").unwrap(), "Undefined");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("Gudermannian[z]").unwrap(), "Gudermannian[z]");
  }
}

mod precision {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("Precision[1]").unwrap(), "Infinity");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("Precision[1/2]").unwrap(), "Infinity");
  }

  #[test]
  fn machine_real() {
    assert_eq!(interpret("Precision[0.5]").unwrap(), "MachinePrecision");
  }
}

mod accuracy {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("Accuracy[1]").unwrap(), "Infinity");
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

mod unsame_q_multi {
  use super::*;

  #[test]
  fn three_args_with_duplicate() {
    assert_eq!(interpret("UnsameQ[1, 1, 2]").unwrap(), "False");
  }

  #[test]
  fn three_args_all_different() {
    assert_eq!(interpret("UnsameQ[1, 2, 3]").unwrap(), "True");
  }
}

mod real_abs {
  use super::*;

  #[test]
  fn real_negative() {
    assert_eq!(interpret("RealAbs[-3.]").unwrap(), "3.");
  }

  #[test]
  fn integer_negative() {
    assert_eq!(interpret("RealAbs[-3]").unwrap(), "3");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("RealAbs[x]").unwrap(), "RealAbs[x]");
  }
}

mod abs_infinity {
  use super::*;

  #[test]
  fn infinity() {
    assert_eq!(interpret("Abs[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn neg_infinity() {
    assert_eq!(interpret("Abs[-Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn complex_infinity() {
    assert_eq!(interpret("Abs[ComplexInfinity]").unwrap(), "Infinity");
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

mod element {
  use super::*;

  #[test]
  fn integer_in_integers() {
    assert_eq!(interpret("Element[3, Integers]").unwrap(), "True");
  }

  #[test]
  fn real_not_in_integers() {
    assert_eq!(interpret("Element[3.5, Integers]").unwrap(), "False");
  }

  #[test]
  fn alternatives_with_known_member() {
    assert_eq!(
      interpret("Element[3 | a, Integers]").unwrap(),
      "Element[a, Integers]"
    );
  }

  #[test]
  fn symbolic_in_reals() {
    assert_eq!(interpret("Element[a, Reals]").unwrap(), "Element[a, Reals]");
  }

  #[test]
  fn integer_in_reals() {
    assert_eq!(interpret("Element[5, Reals]").unwrap(), "True");
  }

  #[test]
  fn prime_in_primes() {
    assert_eq!(interpret("Element[7, Primes]").unwrap(), "True");
  }

  #[test]
  fn non_prime_in_primes() {
    assert_eq!(interpret("Element[4, Primes]").unwrap(), "False");
  }
}

mod conditional_expression {
  use super::*;

  #[test]
  fn true_condition() {
    assert_eq!(
      interpret("ConditionalExpression[x^2, True]").unwrap(),
      "x^2"
    );
  }

  #[test]
  fn false_condition() {
    assert_eq!(
      interpret("ConditionalExpression[x^2, False]").unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn symbolic_condition() {
    assert_eq!(
      interpret("ConditionalExpression[x^2, x > 0]").unwrap(),
      "ConditionalExpression[x^2, x > 0]"
    );
  }
}

mod assuming {
  use super::*;

  #[test]
  fn assuming_matching_condition_simplifies() {
    assert_eq!(
      interpret("Assuming[y>0, ConditionalExpression[y x^2, y>0]//Simplify]")
        .unwrap(),
      "x^2*y"
    );
  }

  #[test]
  fn assuming_negated_condition_gives_undefined() {
    assert_eq!(
      interpret(
        "Assuming[Not[y>0], ConditionalExpression[y x^2, y>0]//Simplify]"
      )
      .unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn simplify_conditional_without_assumptions() {
    assert_eq!(
      interpret("ConditionalExpression[y x ^ 2, y > 0]//Simplify").unwrap(),
      "ConditionalExpression[x^2*y, y > 0]"
    );
  }

  #[test]
  fn assuming_returns_body_value() {
    assert_eq!(interpret("Assuming[x > 0, 1 + 2]").unwrap(), "3");
  }

  #[test]
  fn assumptions_restored_after_assuming() {
    assert_eq!(interpret("Assuming[x > 0, $Assumptions]").unwrap(), "x > 0");
    assert_eq!(interpret("$Assumptions").unwrap(), "True");
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

mod attributes {
  use super::*;

  #[test]
  fn plus() {
    assert_eq!(
      interpret("Attributes[Plus]").unwrap(),
      "{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"
    );
  }

  #[test]
  fn hold() {
    assert_eq!(
      interpret("Attributes[Hold]").unwrap(),
      "{HoldAll, Protected}"
    );
  }

  #[test]
  fn if_func() {
    assert_eq!(
      interpret("Attributes[If]").unwrap(),
      "{HoldRest, Protected}"
    );
  }

  #[test]
  fn set_func() {
    assert_eq!(
      interpret("Attributes[Set]").unwrap(),
      "{HoldFirst, Protected, SequenceHold}"
    );
  }

  #[test]
  fn and_func() {
    assert_eq!(
      interpret("Attributes[And]").unwrap(),
      "{Flat, HoldAll, OneIdentity, Protected}"
    );
  }

  #[test]
  fn constant_e() {
    assert_eq!(
      interpret("Attributes[E]").unwrap(),
      "{Constant, Protected, ReadProtected}"
    );
  }

  #[test]
  fn sin_func() {
    assert_eq!(
      interpret("Attributes[Sin]").unwrap(),
      "{Listable, NumericFunction, Protected}"
    );
  }

  #[test]
  fn unknown_func() {
    assert_eq!(interpret("Attributes[unknownfunc]").unwrap(), "{}");
  }

  #[test]
  fn string_arg() {
    assert_eq!(
      interpret("Attributes[\"Plus\"]").unwrap(),
      "{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"
    );
  }

  #[test]
  fn non_symbol_arg_returns_unevaluated() {
    assert_eq!(
      interpret("Attributes[a + b + c]").unwrap(),
      "Attributes[a + b + c]"
    );
  }

  #[test]
  fn hold_complete() {
    assert_eq!(
      interpret("Attributes[HoldComplete]").unwrap(),
      "{HoldAllComplete, Protected}"
    );
  }

  #[test]
  fn unevaluated() {
    assert_eq!(
      interpret("Attributes[Unevaluated]").unwrap(),
      "{HoldAllComplete, Protected}"
    );
  }
}

mod options {
  use super::*;

  #[test]
  fn set_and_get() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f]").unwrap(),
      "{a -> 1, b -> 2}"
    );
  }

  #[test]
  fn get_specific_option() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, a]").unwrap(),
      "{a -> 1}"
    );
  }

  #[test]
  fn get_second_option() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, b]").unwrap(),
      "{b -> 2}"
    );
  }

  #[test]
  fn unknown_function() {
    assert_eq!(interpret("Options[unknownfunc]").unwrap(), "{}");
  }

  #[test]
  fn option_not_found() {
    assert_eq!(
      interpret("Options[f] = {a -> 1, b -> 2}; Options[f, c]").unwrap(),
      "{}"
    );
  }

  #[test]
  fn overwrite_options() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> 1}; Options[f] = {a -> 10, b -> 20}; Options[f]"
      )
      .unwrap(),
      "{a -> 10, b -> 20}"
    );
  }

  #[test]
  fn single_rule() {
    assert_eq!(
      interpret("Options[g] = {x -> 42}; Options[g]").unwrap(),
      "{x -> 42}"
    );
  }

  #[test]
  fn multiple_functions() {
    assert_eq!(
      interpret(
        "Options[f] = {a -> 1}; Options[g] = {b -> 2}; {Options[f], Options[g]}"
      )
      .unwrap(),
      "{{a -> 1}, {b -> 2}}"
    );
  }
}

mod symbol_q {
  use super::*;

  #[test]
  fn symbol_is_true() {
    assert_eq!(interpret("SymbolQ[a]").unwrap(), "True");
  }

  #[test]
  fn integer_is_false() {
    assert_eq!(interpret("SymbolQ[1]").unwrap(), "False");
  }

  #[test]
  fn expr_is_false() {
    assert_eq!(interpret("SymbolQ[a + b]").unwrap(), "False");
  }

  #[test]
  fn string_is_false() {
    assert_eq!(interpret("SymbolQ[\"abc\"]").unwrap(), "False");
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
}

mod same_q_unsame_q {
  use super::*;

  #[test]
  fn same_q_empty() {
    assert_eq!(interpret("SameQ[]").unwrap(), "True");
  }

  #[test]
  fn same_q_single() {
    assert_eq!(interpret("SameQ[a]").unwrap(), "True");
  }

  #[test]
  fn unsame_q_empty() {
    assert_eq!(interpret("UnsameQ[]").unwrap(), "True");
  }

  #[test]
  fn unsame_q_single() {
    assert_eq!(interpret("UnsameQ[a]").unwrap(), "True");
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
}

mod composition_edge_cases {
  use super::*;

  #[test]
  fn empty_is_identity() {
    assert_eq!(interpret("Composition[]").unwrap(), "Identity");
  }

  #[test]
  fn single_is_function() {
    assert_eq!(interpret("Composition[f]").unwrap(), "f");
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

mod polynomial_q_1arg {
  use super::*;

  #[test]
  fn polynomial_expr() {
    assert_eq!(interpret("PolynomialQ[x^2]").unwrap(), "True");
  }

  #[test]
  fn constant() {
    assert_eq!(interpret("PolynomialQ[2]").unwrap(), "True");
  }

  #[test]
  fn non_polynomial() {
    assert_eq!(interpret("PolynomialQ[x^2 + x/y]").unwrap(), "False");
  }
}

mod equivalent_logic {
  use super::*;

  #[test]
  fn all_true() {
    assert_eq!(interpret("Equivalent[True, True]").unwrap(), "True");
  }

  #[test]
  fn mixed_true_false() {
    assert_eq!(interpret("Equivalent[True, True, False]").unwrap(), "False");
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("Equivalent[a, b, c]").unwrap(),
      "Equivalent[a, b, c]"
    );
  }
}

mod list_equality {
  use super::*;

  #[test]
  fn nested_lists_equal() {
    assert_eq!(interpret("{{1}, {2}} == {{1}, {2}}").unwrap(), "True");
  }

  #[test]
  fn flat_lists_equal() {
    assert_eq!(interpret("{1, 2} == {1, 2}").unwrap(), "True");
  }

  #[test]
  fn lists_not_equal() {
    assert_eq!(interpret("{1, 2} == {1, 3}").unwrap(), "False");
  }

  #[test]
  fn different_length_lists() {
    assert_eq!(interpret("{1, 2} == {1, 2, 3}").unwrap(), "False");
  }
}

mod equal_edge_cases {
  use super::*;

  #[test]
  fn equal_zero_args() {
    assert_eq!(interpret("Equal[]").unwrap(), "True");
  }

  #[test]
  fn equal_one_arg() {
    assert_eq!(interpret("Equal[x]").unwrap(), "True");
  }

  #[test]
  fn equal_one_arg_list() {
    assert_eq!(
      interpret("{Equal[x], Equal[1], Equal[\"a\"]}").unwrap(),
      "{True, True, True}"
    );
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

mod mixed_coefficient_combining {
  use super::*;

  #[test]
  fn real_and_integer_coefficients() {
    assert_eq!(
      interpret("a + b + 4.5 + a + b + a + 2 + 1.5 b").unwrap(),
      "6.5 + 3*a + 3.5*b"
    );
  }
}

mod through {
  use super::*;

  #[test]
  fn through_list_head() {
    assert_eq!(interpret("Through[{f, g}[x]]").unwrap(), "{f[x], g[x]}");
  }

  #[test]
  fn through_list_head_multiple_args() {
    assert_eq!(
      interpret("Through[{f, g}[x, y]]").unwrap(),
      "{f[x, y], g[x, y]}"
    );
  }

  #[test]
  fn through_function_head() {
    assert_eq!(interpret("Through[f[g][x]]").unwrap(), "f[g[x]]");
  }

  #[test]
  fn through_plus_head() {
    assert_eq!(interpret("Through[Plus[f, g][x]]").unwrap(), "f[x] + g[x]");
  }

  #[test]
  fn through_times_head() {
    assert_eq!(interpret("Through[Times[f, g][x]]").unwrap(), "f[x]*g[x]");
  }

  #[test]
  fn through_simple_call_unevaluated() {
    assert_eq!(interpret("Through[f[x]]").unwrap(), "Through[f[x]]");
  }

  #[test]
  fn through_non_call_unevaluated() {
    assert_eq!(interpret("Through[x]").unwrap(), "Through[x]");
  }

  #[test]
  fn through_with_matching_head_filter() {
    assert_eq!(
      interpret("Through[{f, g}[x], List]").unwrap(),
      "{f[x], g[x]}"
    );
  }

  #[test]
  fn through_with_non_matching_head_filter() {
    assert_eq!(interpret("Through[{f, g}[x], Plus]").unwrap(), "{f, g}[x]");
  }
}

mod curried_call_preservation {
  use super::*;

  #[test]
  fn symbolic_curried_call_stays() {
    assert_eq!(interpret("f[g][x]").unwrap(), "f[g][x]");
  }

  #[test]
  fn list_head_stays() {
    assert_eq!(interpret("{f, g}[x]").unwrap(), "{f, g}[x]");
  }
}

mod anonymous_function_precedence {
  use super::*;

  #[test]
  fn ampersand_captures_full_expression() {
    assert_eq!(interpret("#1+#2&[4, 5]").unwrap(), "9");
  }

  #[test]
  fn ampersand_with_operator_after() {
    assert_eq!(interpret("#^2& @ 3").unwrap(), "9");
  }

  #[test]
  fn slot2_standalone() {
    assert_eq!(interpret("#2&[4, 5]").unwrap(), "5");
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

mod apply_head_replacement {
  use super::*;

  #[test]
  fn apply_replaces_plus_head() {
    assert_eq!(interpret("f @@ (a + b + c)").unwrap(), "f[a, b, c]");
  }

  #[test]
  fn apply_operator_form() {
    assert_eq!(interpret("Apply[f][a + b + c]").unwrap(), "f[a, b, c]");
  }
}

mod factorial2 {
  use super::*;

  #[test]
  fn double_factorial_odd() {
    assert_eq!(interpret("Factorial2[5]").unwrap(), "15");
    assert_eq!(interpret("Factorial2[7]").unwrap(), "105");
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
}

mod xor_single {
  use super::*;

  #[test]
  fn xor_single_arg() {
    assert_eq!(interpret("Xor[True]").unwrap(), "True");
    assert_eq!(interpret("Xor[False]").unwrap(), "False");
  }
}

mod power_expand {
  use super::*;

  #[test]
  fn power_of_power() {
    assert_eq!(interpret("PowerExpand[(a ^ b) ^ c]").unwrap(), "a^(b*c)");
  }

  #[test]
  fn power_of_product() {
    assert_eq!(interpret("PowerExpand[(a * b) ^ c]").unwrap(), "a^c*b^c");
  }

  #[test]
  fn sqrt_of_square() {
    assert_eq!(interpret("PowerExpand[(x ^ 2) ^ (1/2)]").unwrap(), "x");
  }
}

mod variables {
  use super::*;

  #[test]
  fn variables_polynomial() {
    assert_eq!(
      interpret("Variables[a x^2 + b x + c]").unwrap(),
      "{a, b, c, x}"
    );
  }

  #[test]
  fn variables_list() {
    assert_eq!(
      interpret("Variables[{a + b x, c y^2 + x/2}]").unwrap(),
      "{a, b, c, x, y}"
    );
  }

  #[test]
  fn variables_with_function() {
    assert_eq!(interpret("Variables[x + Sin[y]]").unwrap(), "{x, Sin[y]}");
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
}

mod atom_q {
  use super::*;

  #[test]
  fn atom_q_rational() {
    assert_eq!(interpret("AtomQ[1/2]").unwrap(), "True");
  }

  #[test]
  fn atom_q_integer() {
    assert_eq!(interpret("AtomQ[5]").unwrap(), "True");
  }

  #[test]
  fn atom_q_expression() {
    assert_eq!(interpret("AtomQ[x + y]").unwrap(), "False");
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

mod bessel_j {
  use super::*;

  #[test]
  fn zero_order_at_origin() {
    assert_eq!(interpret("BesselJ[0, 0]").unwrap(), "1");
  }

  #[test]
  fn nonzero_order_at_origin() {
    assert_eq!(interpret("BesselJ[1, 0]").unwrap(), "0");
    assert_eq!(interpret("BesselJ[2, 0]").unwrap(), "0");
  }

  #[test]
  fn zero_order_real_origin() {
    assert_eq!(interpret("BesselJ[0, 0.]").unwrap(), "1.");
  }

  #[test]
  fn numeric_zero_order() {
    let result: f64 = interpret("BesselJ[0, 5.2]").unwrap().parse().unwrap();
    assert!((result - (-0.11029043979098728)).abs() < 1e-10);
  }

  #[test]
  fn numeric_first_order() {
    let result: f64 = interpret("BesselJ[1, 3.0]").unwrap().parse().unwrap();
    assert!((result - 0.3390589585259365).abs() < 1e-10);
  }

  #[test]
  fn numeric_second_order() {
    let result: f64 = interpret("BesselJ[2, 1.5]").unwrap().parse().unwrap();
    assert!((result - 0.23208767214421472).abs() < 1e-10);
  }

  #[test]
  fn negative_order() {
    let result: f64 = interpret("BesselJ[-1, 3.0]").unwrap().parse().unwrap();
    assert!((result - (-0.3390589585259365)).abs() < 1e-10);
  }

  #[test]
  fn symbolic_returns_unevaluated() {
    assert_eq!(interpret("BesselJ[0, x]").unwrap(), "BesselJ[0, x]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[BesselJ[0, 5]]").unwrap().parse().unwrap();
    assert!((result - (-0.17759677131433846)).abs() < 1e-10);
  }
}

mod hypergeometric2f1 {
  use super::*;

  #[test]
  fn z_zero() {
    assert_eq!(interpret("Hypergeometric2F1[1, 2, 3, 0]").unwrap(), "1");
  }

  #[test]
  fn numeric_basic() {
    let result: f64 = interpret("Hypergeometric2F1[1, 2, 3, 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.5451774444795618).abs() < 1e-10);
  }

  #[test]
  fn numeric_log_related() {
    let result: f64 = interpret("Hypergeometric2F1[1, 1, 2, 0.5]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.38629436111989).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("Hypergeometric2F1[1, 2, 3, x]").unwrap(),
      "Hypergeometric2F1[1, 2, 3, x]"
    );
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[Hypergeometric2F1[1, 2, 3, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.5451774444795618).abs() < 1e-10);
  }
}

mod elliptic_k {
  use super::*;

  #[test]
  fn zero_arg() {
    assert_eq!(interpret("EllipticK[0]").unwrap(), "Pi/2");
  }

  #[test]
  fn one_arg() {
    assert_eq!(interpret("EllipticK[1]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn real_one_arg() {
    assert_eq!(interpret("EllipticK[1.0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn real_zero_arg() {
    let result: f64 = interpret("EllipticK[0.0]").unwrap().parse().unwrap();
    assert!((result - 1.5707963267948966).abs() < 1e-10);
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("EllipticK[0.5]").unwrap().parse().unwrap();
    assert!((result - 1.8540746773013717).abs() < 1e-10);
  }

  #[test]
  fn numeric_near_one() {
    let result: f64 = interpret("EllipticK[0.9]").unwrap().parse().unwrap();
    assert!((result - 2.578092113348173).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result: f64 = interpret("EllipticK[-0.5]").unwrap().parse().unwrap();
    assert!((result - 1.415737208425956).abs() < 1e-10);
  }

  #[test]
  fn symbolic_exact() {
    assert_eq!(interpret("EllipticK[1/2]").unwrap(), "EllipticK[1/2]");
  }

  #[test]
  fn symbolic_variable() {
    assert_eq!(interpret("EllipticK[x]").unwrap(), "EllipticK[x]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[EllipticK[1/2]]").unwrap().parse().unwrap();
    assert!((result - 1.8540746773013717).abs() < 1e-10);
  }
}

mod polylog {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("PolyLog[2, 0]").unwrap(), "0");
    assert_eq!(interpret("PolyLog[3, 0]").unwrap(), "0");
  }

  #[test]
  fn at_one_gives_zeta() {
    assert_eq!(interpret("PolyLog[2, 1]").unwrap(), "Pi^2/6");
    assert_eq!(interpret("PolyLog[3, 1]").unwrap(), "Zeta[3]");
    assert_eq!(interpret("PolyLog[4, 1]").unwrap(), "Pi^4/90");
  }

  #[test]
  fn at_neg_one() {
    assert_eq!(interpret("PolyLog[2, -1]").unwrap(), "-1/12*Pi^2");
    assert_eq!(interpret("PolyLog[3, -1]").unwrap(), "(-3*Zeta[3])/4");
    assert_eq!(interpret("PolyLog[4, -1]").unwrap(), "(-7*Pi^4)/720");
  }

  #[test]
  fn s_one_symbolic() {
    assert_eq!(interpret("PolyLog[1, x]").unwrap(), "-Log[1 - x]");
  }

  #[test]
  fn s_one_rational() {
    assert_eq!(interpret("PolyLog[1, 1/2]").unwrap(), "Log[2]");
  }

  #[test]
  fn s_zero() {
    assert_eq!(interpret("PolyLog[0, x]").unwrap(), "x/(1 - x)");
    assert_eq!(interpret("PolyLog[0, 0]").unwrap(), "0");
    assert_eq!(interpret("PolyLog[0, 1]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn negative_s() {
    assert_eq!(interpret("PolyLog[-1, x]").unwrap(), "x/(1 - x)^2");
    assert_eq!(interpret("PolyLog[-2, x]").unwrap(), "(x + x^2)/(1 - x)^3");
    assert_eq!(
      interpret("PolyLog[-3, x]").unwrap(),
      "(x + 4*x^2 + x^3)/(1 - x)^4"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("PolyLog[2, x]").unwrap(), "PolyLog[2, x]");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("PolyLog[2, 0.5]").unwrap().parse().unwrap();
    assert!((result - 0.5822405264650125).abs() < 1e-8);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[PolyLog[3, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 0.5372131936080402).abs() < 1e-8);
  }
}

mod log_rational_simplification {
  use super::*;

  #[test]
  fn log_half() {
    assert_eq!(interpret("Log[1/2]").unwrap(), "-Log[2]");
  }

  #[test]
  fn log_two_thirds() {
    assert_eq!(interpret("Log[2/3]").unwrap(), "-Log[3/2]");
  }
}

mod legendre_p {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("LegendreP[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("LegendreP[1, x]").unwrap(), "x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("LegendreP[2, x]").unwrap(), "(-1 + 3*x^2)/2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(interpret("LegendreP[3, x]").unwrap(), "(-3*x + 5*x^3)/2");
  }

  #[test]
  fn degree_four() {
    assert_eq!(
      interpret("LegendreP[4, x]").unwrap(),
      "(3 - 30*x^2 + 35*x^4)/8"
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("LegendreP[2, 0]").unwrap(), "-1/2");
    assert_eq!(interpret("LegendreP[3, 0]").unwrap(), "0");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("LegendreP[2, 1]").unwrap(), "1");
    assert_eq!(interpret("LegendreP[3, 1]").unwrap(), "1");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("LegendreP[2, 0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.125)).abs() < 1e-10);
  }

  #[test]
  fn at_rational() {
    assert_eq!(interpret("LegendreP[3, 1/2]").unwrap(), "-7/16");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[LegendreP[3, 1/2]]").unwrap().parse().unwrap();
    assert!((result - (-0.4375)).abs() < 1e-10);
  }

  #[test]
  fn high_degree() {
    assert_eq!(
      interpret("LegendreP[10, x]").unwrap(),
      "(-63 + 3465*x^2 - 30030*x^4 + 90090*x^6 - 109395*x^8 + 46189*x^10)/256"
    );
  }
}

mod polygamma {
  use super::*;

  #[test]
  fn digamma_one() {
    assert_eq!(interpret("PolyGamma[1]").unwrap(), "-EulerGamma");
  }

  #[test]
  fn digamma_two() {
    assert_eq!(interpret("PolyGamma[2]").unwrap(), "1 - EulerGamma");
  }

  #[test]
  fn digamma_three() {
    assert_eq!(interpret("PolyGamma[3]").unwrap(), "3/2 - EulerGamma");
  }

  #[test]
  fn digamma_five() {
    assert_eq!(interpret("PolyGamma[0, 5]").unwrap(), "25/12 - EulerGamma");
  }

  #[test]
  fn trigamma_one() {
    assert_eq!(interpret("PolyGamma[1, 1]").unwrap(), "Pi^2/6");
  }

  #[test]
  fn trigamma_two() {
    assert_eq!(interpret("PolyGamma[1, 2]").unwrap(), "-1 + Pi^2/6");
  }

  #[test]
  fn trigamma_three() {
    assert_eq!(interpret("PolyGamma[1, 3]").unwrap(), "-5/4 + Pi^2/6");
  }

  #[test]
  fn tetragamma_unevaluated() {
    assert_eq!(interpret("PolyGamma[2, 1]").unwrap(), "PolyGamma[2, 1]");
  }

  #[test]
  fn polygamma_3_1() {
    assert_eq!(interpret("PolyGamma[3, 1]").unwrap(), "Pi^4/15");
  }

  #[test]
  fn polygamma_5_1() {
    assert_eq!(interpret("PolyGamma[5, 1]").unwrap(), "(8*Pi^6)/63");
  }

  #[test]
  fn polygamma_3_3_factored() {
    assert_eq!(
      interpret("PolyGamma[3, 3]").unwrap(),
      "6*(-17/16 + Pi^4/90)"
    );
  }

  #[test]
  fn pole_at_zero() {
    assert_eq!(interpret("PolyGamma[0, 0]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[0, -1]").unwrap(), "ComplexInfinity");
    assert_eq!(interpret("PolyGamma[1, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("PolyGamma[x]").unwrap(), "PolyGamma[0, x]");
    assert_eq!(interpret("PolyGamma[1/2]").unwrap(), "PolyGamma[0, 1/2]");
  }

  #[test]
  fn numeric_digamma() {
    let result: f64 = interpret("PolyGamma[1.0]").unwrap().parse().unwrap();
    assert!((result - (-0.5772156649015329)).abs() < 1e-10);
  }

  #[test]
  fn numeric_trigamma() {
    let result: f64 = interpret("PolyGamma[1, 1.0]").unwrap().parse().unwrap();
    assert!((result - 1.6449340668482264).abs() < 1e-8);
  }

  #[test]
  fn n_evaluates_polygamma() {
    let result: f64 = interpret("N[PolyGamma[2, 1]]").unwrap().parse().unwrap();
    assert!((result - (-2.4041138063191885)).abs() < 1e-8);
  }
}

mod zeta {
  use super::*;

  #[test]
  fn pole_at_one() {
    assert_eq!(interpret("Zeta[1]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("Zeta[0]").unwrap(), "-1/2");
  }

  #[test]
  fn positive_even_2() {
    assert_eq!(interpret("Zeta[2]").unwrap(), "Pi^2/6");
  }

  #[test]
  fn positive_even_4() {
    assert_eq!(interpret("Zeta[4]").unwrap(), "Pi^4/90");
  }

  #[test]
  fn positive_even_6() {
    assert_eq!(interpret("Zeta[6]").unwrap(), "Pi^6/945");
  }

  #[test]
  fn positive_even_12() {
    assert_eq!(interpret("Zeta[12]").unwrap(), "(691*Pi^12)/638512875");
  }

  #[test]
  fn positive_even_20() {
    assert_eq!(
      interpret("Zeta[20]").unwrap(),
      "(174611*Pi^20)/1531329465290625"
    );
  }

  #[test]
  fn positive_odd_unevaluated() {
    assert_eq!(interpret("Zeta[3]").unwrap(), "Zeta[3]");
    assert_eq!(interpret("Zeta[5]").unwrap(), "Zeta[5]");
  }

  #[test]
  fn negative_even_trivial_zeros() {
    assert_eq!(interpret("Zeta[-2]").unwrap(), "0");
    assert_eq!(interpret("Zeta[-4]").unwrap(), "0");
    assert_eq!(interpret("Zeta[-6]").unwrap(), "0");
  }

  #[test]
  fn negative_odd() {
    assert_eq!(interpret("Zeta[-1]").unwrap(), "-1/12");
    assert_eq!(interpret("Zeta[-3]").unwrap(), "1/120");
    assert_eq!(interpret("Zeta[-5]").unwrap(), "-1/252");
    assert_eq!(interpret("Zeta[-7]").unwrap(), "1/240");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("Zeta[x]").unwrap(), "Zeta[x]");
    assert_eq!(interpret("Zeta[1/2]").unwrap(), "Zeta[1/2]");
  }

  #[test]
  fn numeric_real_positive() {
    let result: f64 = interpret("Zeta[2.0]").unwrap().parse().unwrap();
    assert!((result - 1.6449340668482264).abs() < 1e-10);
  }

  #[test]
  fn numeric_real_half() {
    let result: f64 = interpret("Zeta[0.5]").unwrap().parse().unwrap();
    assert!((result - (-1.460354508809588)).abs() < 1e-6);
  }

  #[test]
  fn n_evaluates_zeta() {
    let result: f64 = interpret("N[Zeta[2]]").unwrap().parse().unwrap();
    assert!((result - 1.6449340668482264).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_zeta_3() {
    let result: f64 = interpret("N[Zeta[3]]").unwrap().parse().unwrap();
    assert!((result - 1.2020569031595942).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative_real() {
    let result: f64 = interpret("Zeta[-0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.20788622497735454)).abs() < 1e-6);
  }
}

mod jacobi_dn {
  use super::*;

  #[test]
  fn at_zero_u() {
    assert_eq!(interpret("JacobiDN[0, m]").unwrap(), "1");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiDN[u, 0]").unwrap(), "1");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiDN[u, 1]").unwrap(), "Sech[u]");
  }

  #[test]
  fn even_symmetry() {
    assert_eq!(interpret("JacobiDN[-u, m]").unwrap(), "JacobiDN[u, m]");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("JacobiDN[4., 2/3]").unwrap().parse().unwrap();
    assert!((result - 0.9988832842546814).abs() < 1e-10);
  }

  #[test]
  fn numeric_real_2() {
    let result: f64 = interpret("JacobiDN[0.5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 0.9656789647459513).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiDN[u, m]").unwrap(), "JacobiDN[u, m]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[JacobiDN[1, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 0.8231610016315964).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_2() {
    let result: f64 =
      interpret("N[JacobiDN[2, 1/3]]").unwrap().parse().unwrap();
    assert!((result - 0.8259983048005796).abs() < 1e-10);
  }
}

mod jacobi_sn {
  use super::*;

  #[test]
  fn at_zero_u() {
    assert_eq!(interpret("JacobiSN[0, m]").unwrap(), "0");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiSN[u, 0]").unwrap(), "Sin[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiSN[u, 1]").unwrap(), "Tanh[u]");
  }

  #[test]
  fn odd_symmetry() {
    assert_eq!(interpret("JacobiSN[-u, m]").unwrap(), "-JacobiSN[u, m]");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("JacobiSN[0.5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 0.47421562271182044).abs() < 1e-10);
  }

  #[test]
  fn numeric_real_2() {
    let result: f64 = interpret("JacobiSN[4., 2/3]").unwrap().parse().unwrap();
    assert!((result - 0.05786429516439241).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiSN[u, m]").unwrap(), "JacobiSN[u, m]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[JacobiSN[1, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 0.8030018248956442).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_2() {
    let result: f64 =
      interpret("N[JacobiSN[2, 1/3]]").unwrap().parse().unwrap();
    assert!((result - 0.9763095827654809).abs() < 1e-10);
  }
}

mod elliptic_theta {
  use super::*;

  #[test]
  fn theta1_at_q_zero() {
    assert_eq!(interpret("EllipticTheta[1, z, 0]").unwrap(), "0");
  }

  #[test]
  fn theta2_at_q_zero() {
    assert_eq!(interpret("EllipticTheta[2, z, 0]").unwrap(), "0");
  }

  #[test]
  fn theta3_at_q_zero() {
    assert_eq!(interpret("EllipticTheta[3, z, 0]").unwrap(), "1");
  }

  #[test]
  fn theta4_at_q_zero() {
    assert_eq!(interpret("EllipticTheta[4, z, 0]").unwrap(), "1");
  }

  #[test]
  fn theta1_numeric() {
    let result: f64 = interpret("EllipticTheta[1, 0.5, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.5279836054564474).abs() < 1e-10);
  }

  #[test]
  fn theta2_numeric() {
    let result: f64 = interpret("EllipticTheta[2, 0.5, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.9877965496358908).abs() < 1e-10);
  }

  #[test]
  fn theta3_numeric() {
    let result: f64 = interpret("EllipticTheta[3, 0.5, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.1079772298263333).abs() < 1e-10);
  }

  #[test]
  fn theta4_numeric() {
    let result: f64 = interpret("EllipticTheta[4, 0.5, 0.1]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 0.8918563114390474).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("EllipticTheta[3, z, q]").unwrap(),
      "EllipticTheta[3, z, q]"
    );
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[EllipticTheta[3, 0, 1/2]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 2.128936827211877).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_theta1() {
    let result: f64 = interpret("N[EllipticTheta[1, 1, 1/3]]")
      .unwrap()
      .parse()
      .unwrap();
    assert!((result - 1.2529300675643478).abs() < 1e-10);
  }
}

mod exp_integral_ei {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("ExpIntegralEi[0]").unwrap(), "-Infinity");
  }

  #[test]
  fn at_infinity() {
    assert_eq!(interpret("ExpIntegralEi[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn at_neg_infinity() {
    assert_eq!(interpret("ExpIntegralEi[-Infinity]").unwrap(), "0");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("ExpIntegralEi[x]").unwrap(), "ExpIntegralEi[x]");
  }

  #[test]
  fn numeric_positive() {
    let result: f64 = interpret("ExpIntegralEi[0.5]").unwrap().parse().unwrap();
    assert!((result - 0.4542199048631736).abs() < 1e-10);
  }

  #[test]
  fn numeric_negative() {
    let result: f64 =
      interpret("ExpIntegralEi[-0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.5597735947761607)).abs() < 1e-10);
  }

  #[test]
  fn numeric_larger() {
    let result: f64 = interpret("ExpIntegralEi[2.0]").unwrap().parse().unwrap();
    assert!((result - 4.95423435600189).abs() < 1e-8);
  }

  #[test]
  fn n_evaluates_positive() {
    let result: f64 =
      interpret("N[ExpIntegralEi[1]]").unwrap().parse().unwrap();
    assert!((result - 1.8951178163559368).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_negative() {
    let result: f64 =
      interpret("N[ExpIntegralEi[-1]]").unwrap().parse().unwrap();
    assert!((result - (-0.21938393439552026)).abs() < 1e-10);
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

mod jacobi_cn {
  use super::*;

  #[test]
  fn at_zero_u() {
    assert_eq!(interpret("JacobiCN[0, m]").unwrap(), "1");
  }

  #[test]
  fn m_zero() {
    assert_eq!(interpret("JacobiCN[u, 0]").unwrap(), "Cos[u]");
  }

  #[test]
  fn m_one() {
    assert_eq!(interpret("JacobiCN[u, 1]").unwrap(), "Sech[u]");
  }

  #[test]
  fn even_symmetry() {
    assert_eq!(interpret("JacobiCN[-u, m]").unwrap(), "JacobiCN[u, m]");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("JacobiCN[0.5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 0.8804087364264626).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("JacobiCN[u, m]").unwrap(), "JacobiCN[u, m]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[JacobiCN[1, 1/2]]").unwrap().parse().unwrap();
    assert!((result - 0.5959765676721407).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_2() {
    let result: f64 =
      interpret("N[JacobiCN[2, 1/3]]").unwrap().parse().unwrap();
    assert!((result - (-0.21637836906745786)).abs() < 1e-10);
  }
}

mod bessel_i {
  use super::*;

  #[test]
  fn at_zero_order_zero() {
    assert_eq!(interpret("BesselI[0, 0]").unwrap(), "1");
  }

  #[test]
  fn at_zero_order_nonzero() {
    assert_eq!(interpret("BesselI[1, 0]").unwrap(), "0");
  }

  #[test]
  fn numeric_order_zero() {
    let result: f64 = interpret("BesselI[0, 1.5]").unwrap().parse().unwrap();
    assert!((result - 1.6467231897728904).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_one() {
    let result: f64 = interpret("BesselI[1, 2.0]").unwrap().parse().unwrap();
    assert!((result - 1.5906368546373288).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("BesselI[n, z]").unwrap(), "BesselI[n, z]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[BesselI[0, 1]]").unwrap().parse().unwrap();
    assert!((result - 1.2660658777520082).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_order_two() {
    let result: f64 = interpret("N[BesselI[2, 1]]").unwrap().parse().unwrap();
    assert!((result - 0.13574766976703828).abs() < 1e-10);
  }
}

mod bessel_k {
  use super::*;

  #[test]
  fn at_zero_order_zero() {
    assert_eq!(interpret("BesselK[0, 0]").unwrap(), "Infinity");
  }

  #[test]
  fn at_zero_order_nonzero() {
    assert_eq!(interpret("BesselK[1, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn numeric_order_zero() {
    let result: f64 = interpret("BesselK[0, 1.5]").unwrap().parse().unwrap();
    assert!((result - 0.2138055626475258).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_one() {
    let result: f64 = interpret("BesselK[1, 2.0]").unwrap().parse().unwrap();
    assert!((result - 0.13986588181652243).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("BesselK[n, z]").unwrap(), "BesselK[n, z]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[BesselK[0, 1]]").unwrap().parse().unwrap();
    assert!((result - 0.42102443824070834).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates_order_two() {
    let result: f64 = interpret("N[BesselK[2, 1]]").unwrap().parse().unwrap();
    assert!((result - 1.6248388986351778).abs() < 1e-10);
  }
}

mod chebyshev_t {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("ChebyshevT[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("ChebyshevT[1, x]").unwrap(), "x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("ChebyshevT[2, x]").unwrap(), "-1 + 2*x^2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(interpret("ChebyshevT[3, x]").unwrap(), "-3*x + 4*x^3");
  }

  #[test]
  fn degree_four() {
    assert_eq!(interpret("ChebyshevT[4, x]").unwrap(), "1 - 8*x^2 + 8*x^4");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("ChebyshevT[5, 1]").unwrap(), "1");
  }

  #[test]
  fn at_rational() {
    assert_eq!(interpret("ChebyshevT[3, 1/2]").unwrap(), "-1");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("ChebyshevT[5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 0.9988800000000002).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("ChebyshevT[n, x]").unwrap(), "ChebyshevT[n, x]");
  }

  #[test]
  fn at_rational_exact() {
    assert_eq!(interpret("ChebyshevT[4, 1/3]").unwrap(), "17/81");
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("ChebyshevT[4, 0]").unwrap(), "1");
  }
}

mod laguerre_l {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("LaguerreL[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("LaguerreL[1, x]").unwrap(), "1 - x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("LaguerreL[2, x]").unwrap(), "(2 - 4*x + x^2)/2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(
      interpret("LaguerreL[3, x]").unwrap(),
      "(6 - 18*x + 9*x^2 - x^3)/6"
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("LaguerreL[4, 0]").unwrap(), "1");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("LaguerreL[3, 1]").unwrap(), "-2/3");
  }

  #[test]
  fn at_rational() {
    assert_eq!(interpret("LaguerreL[3, 1/2]").unwrap(), "-7/48");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("LaguerreL[5, 0.3]").unwrap().parse().unwrap();
    assert!((result - (-0.09333274999999995)).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("LaguerreL[n, x]").unwrap(), "LaguerreL[n, x]");
  }
}

mod beta_fn {
  use super::*;

  #[test]
  fn both_one() {
    assert_eq!(interpret("Beta[1, 1]").unwrap(), "1");
  }

  #[test]
  fn positive_integers() {
    assert_eq!(interpret("Beta[2, 3]").unwrap(), "1/12");
  }

  #[test]
  fn positive_integers_2() {
    assert_eq!(interpret("Beta[3, 4]").unwrap(), "1/60");
  }

  #[test]
  fn half_half() {
    assert_eq!(interpret("Beta[1/2, 1/2]").unwrap(), "Pi");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("Beta[1.5, 2.5]").unwrap().parse().unwrap();
    assert!((result - 0.19634954084936207).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("Beta[a, b]").unwrap(), "Beta[a, b]");
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[Beta[2, 3]]").unwrap().parse().unwrap();
    assert!((result - 0.08333333333333333).abs() < 1e-10);
  }
}

mod log_integral {
  use super::*;

  #[test]
  fn at_zero() {
    assert_eq!(interpret("LogIntegral[0]").unwrap(), "0");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("LogIntegral[1]").unwrap(), "-Infinity");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("LogIntegral[x]").unwrap(), "LogIntegral[x]");
  }

  #[test]
  fn numeric_two() {
    let result: f64 = interpret("LogIntegral[2.0]").unwrap().parse().unwrap();
    assert!((result - 1.0451637801174924).abs() < 1e-10);
  }

  #[test]
  fn numeric_ten() {
    let result: f64 = interpret("LogIntegral[10.0]").unwrap().parse().unwrap();
    assert!((result - 6.165599504787296).abs() < 1e-10);
  }

  #[test]
  fn numeric_half() {
    let result: f64 = interpret("LogIntegral[0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.37867104306108795)).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[LogIntegral[2]]").unwrap().parse().unwrap();
    assert!((result - 1.0451637801174924).abs() < 1e-10);
  }
}

mod hermite_h {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("HermiteH[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("HermiteH[1, x]").unwrap(), "2*x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("HermiteH[2, x]").unwrap(), "-2 + 4*x^2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(interpret("HermiteH[3, x]").unwrap(), "-12*x + 8*x^3");
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("HermiteH[4, 0]").unwrap(), "12");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("HermiteH[3, 1]").unwrap(), "-4");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("HermiteH[5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 31.757759999999998).abs() < 1e-8);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("HermiteH[n, x]").unwrap(), "HermiteH[n, x]");
  }
}

mod chebyshev_u {
  use super::*;

  #[test]
  fn degree_zero() {
    assert_eq!(interpret("ChebyshevU[0, x]").unwrap(), "1");
  }

  #[test]
  fn degree_one() {
    assert_eq!(interpret("ChebyshevU[1, x]").unwrap(), "2*x");
  }

  #[test]
  fn degree_two() {
    assert_eq!(interpret("ChebyshevU[2, x]").unwrap(), "-1 + 4*x^2");
  }

  #[test]
  fn degree_three() {
    assert_eq!(interpret("ChebyshevU[3, x]").unwrap(), "-4*x + 8*x^3");
  }

  #[test]
  fn degree_four() {
    assert_eq!(
      interpret("ChebyshevU[4, x]").unwrap(),
      "1 - 12*x^2 + 16*x^4"
    );
  }

  #[test]
  fn at_zero() {
    assert_eq!(interpret("ChebyshevU[3, 0]").unwrap(), "0");
    assert_eq!(interpret("ChebyshevU[4, 0]").unwrap(), "1");
  }

  #[test]
  fn at_one() {
    assert_eq!(interpret("ChebyshevU[3, 1]").unwrap(), "4");
  }

  #[test]
  fn numeric_real() {
    let result: f64 = interpret("ChebyshevU[5, 0.3]").unwrap().parse().unwrap();
    assert!((result - 1.01376).abs() < 1e-4);
  }

  #[test]
  fn rational_arg() {
    let result = interpret("N[ChebyshevU[3, 1/2]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - (-1.0)).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 =
      interpret("N[ChebyshevU[2, 3/4]]").unwrap().parse().unwrap();
    // U_2(3/4) = -1 + 4*(9/16) = -1 + 9/4 = 5/4 = 1.25
    assert!((result - 1.25).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("ChebyshevU[n, x]").unwrap(), "ChebyshevU[n, x]");
  }
}

mod bessel_y {
  use super::*;

  #[test]
  fn at_zero_order_zero() {
    assert_eq!(interpret("BesselY[0, 0]").unwrap(), "-Infinity");
  }

  #[test]
  fn at_zero_order_one() {
    assert_eq!(interpret("BesselY[1, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn at_zero_order_two() {
    assert_eq!(interpret("BesselY[2, 0]").unwrap(), "ComplexInfinity");
  }

  #[test]
  fn numeric_order_zero() {
    let result: f64 = interpret("BesselY[0, 2.5]").unwrap().parse().unwrap();
    assert!((result - 0.49807035961523194).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_one() {
    let result: f64 = interpret("BesselY[1, 2.5]").unwrap().parse().unwrap();
    assert!((result - 0.14591813796678565).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_two() {
    let result: f64 = interpret("BesselY[2, 3.0]").unwrap().parse().unwrap();
    assert!((result - (-0.16040039348492371)).abs() < 1e-10);
  }

  #[test]
  fn numeric_order_three() {
    let result: f64 = interpret("BesselY[3, 5.0]").unwrap().parse().unwrap();
    assert!((result - 0.14626716269319232).abs() < 1e-10);
  }

  #[test]
  fn negative_order() {
    // BesselY[-1, z] = -BesselY[1, z]
    let result: f64 = interpret("BesselY[-1, 2.5]").unwrap().parse().unwrap();
    assert!((result - (-0.14591813796678565)).abs() < 1e-10);
  }

  #[test]
  fn numeric_small_arg() {
    let result: f64 = interpret("BesselY[0, 0.5]").unwrap().parse().unwrap();
    assert!((result - (-0.4445187335067067)).abs() < 1e-10);
  }

  #[test]
  fn numeric_large_arg() {
    let result: f64 = interpret("BesselY[0, 10.0]").unwrap().parse().unwrap();
    assert!((result - 0.055671167283599395).abs() < 1e-10);
  }

  #[test]
  fn n_evaluates() {
    let result: f64 = interpret("N[BesselY[0, 5/2]]").unwrap().parse().unwrap();
    assert!((result - 0.49807035961523194).abs() < 1e-10);
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("BesselY[0, x]").unwrap(), "BesselY[0, x]");
  }
}
