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
    assert_eq!(interpret("LCM[x, 5]").unwrap(), "LCM[x, 5]");
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
}
