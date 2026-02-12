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
}

mod minus_wrong_arity {
  use super::*;

  #[test]
  fn minus_single_arg_negates() {
    assert_eq!(interpret("Minus[5]").unwrap(), "-5");
  }

  #[test]
  fn minus_two_args_returns_unevaluated() {
    // Minus[5, 2] should print warning and return 5 - 2
    let result = interpret("Minus[5, 2]").unwrap();
    assert_eq!(result, "5 - 2");
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
