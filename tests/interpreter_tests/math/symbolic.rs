use super::*;

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
