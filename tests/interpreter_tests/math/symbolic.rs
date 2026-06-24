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

  // Monomial body c*k^p: c == 1 keeps the bare factorial, while a coefficient
  // switches to Gamma[1+n] (matching wolframscript). A unit-fraction
  // coefficient renders as a denominator power.
  #[test]
  fn product_monomial_unit_coeff() {
    assert_eq!(interpret("Product[1/k, {k, 1, n}]").unwrap(), "n!^(-1)");
    assert_eq!(interpret("Product[1/k^2, {k, 1, n}]").unwrap(), "n!^(-2)");
  }

  #[test]
  fn product_monomial_with_coeff() {
    assert_eq!(
      interpret("Product[2 k, {k, 1, n}]").unwrap(),
      "2^n*Gamma[1 + n]"
    );
    assert_eq!(
      interpret("Product[3 k^2, {k, 1, n}]").unwrap(),
      "3^n*Gamma[1 + n]^2"
    );
    assert_eq!(
      interpret("Product[c k, {k, 1, n}]").unwrap(),
      "c^n*Gamma[1 + n]"
    );
    assert_eq!(
      interpret("Product[k/2, {k, 1, n}]").unwrap(),
      "Gamma[1 + n]/2^n"
    );
    assert_eq!(
      interpret("Product[2 k/3, {k, 1, n}]").unwrap(),
      "(2/3)^n*Gamma[1 + n]"
    );
  }

  // Telescoping rational products Product[(k+a)/(k+b), {k, 1, n}] with
  // non-negative integer shifts collapse to a finite product of linear factors
  // in n (matching wolframscript).
  #[test]
  fn product_rational_telescoping() {
    assert_eq!(interpret("Product[1 + 1/k, {k, 1, n}]").unwrap(), "1 + n");
    assert_eq!(interpret("Product[(k + 1)/k, {k, 1, n}]").unwrap(), "1 + n");
    assert_eq!(
      interpret("Product[k/(k + 1), {k, 1, n}]").unwrap(),
      "(1 + n)^(-1)"
    );
    assert_eq!(
      interpret("Product[(k + 2)/k, {k, 1, n}]").unwrap(),
      "((1 + n)*(2 + n))/2"
    );
    assert_eq!(
      interpret("Product[(k + 1)/(k + 2), {k, 1, n}]").unwrap(),
      "2/(2 + n)"
    );
    assert_eq!(
      interpret("Product[(k + 3)/(k + 1), {k, 1, n}]").unwrap(),
      "((2 + n)*(3 + n))/6"
    );
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

  // Constant body (independent of the product variable):
  //   Product[c, {k, min, max}] = c^(max - min + 1)
  #[test]
  fn product_constant_symbol() {
    assert_eq!(interpret("Product[x, {k, 1, n}]").unwrap(), "x^n");
  }

  #[test]
  fn product_constant_number() {
    assert_eq!(interpret("Product[2, {k, 1, n}]").unwrap(), "2^n");
  }

  #[test]
  fn product_constant_from_zero() {
    assert_eq!(interpret("Product[x, {k, 0, n}]").unwrap(), "x^(1 + n)");
  }

  #[test]
  fn product_constant_symbolic_bounds() {
    assert_eq!(interpret("Product[c, {k, m, n}]").unwrap(), "c^(1 - m + n)");
  }

  #[test]
  fn product_constant_shifted_lower() {
    assert_eq!(interpret("Product[x, {k, 2, n}]").unwrap(), "x^(-1 + n)");
  }

  // Product[k, {k, 2, n}] = n! (the (min-1)! = 1 denominator is dropped).
  #[test]
  fn product_var_from_two_is_factorial() {
    assert_eq!(interpret("Product[k, {k, 2, n}]").unwrap(), "n!");
  }

  // Infinite product of the iteration variable stays unevaluated (not n!).
  #[test]
  fn product_var_to_infinity_unevaluated() {
    assert_eq!(
      interpret("Product[k, {k, 1, Infinity}]").unwrap(),
      "Product[k, {k, 1, Infinity}]"
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

  // Convergent rational summands with simple integer poles telescope to an
  // exact rational, evaluated in closed form via residues.
  #[test]
  fn infinite_sum_telescoping_rational() {
    assert_eq!(
      interpret("Sum[1/(n (n + 1)), {n, 1, Infinity}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Sum[1/(n (n + 2)), {n, 1, Infinity}]").unwrap(),
      "3/4"
    );
    assert_eq!(
      interpret("Sum[1/(n^2 + n), {n, 1, Infinity}]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("Sum[1/(n (n + 1) (n + 2)), {n, 1, Infinity}]").unwrap(),
      "1/4"
    );
    assert_eq!(
      interpret("Sum[1/((n + 1) (n + 3)), {n, 1, Infinity}]").unwrap(),
      "5/12"
    );
    // A non-unit lower bound is handled too.
    assert_eq!(
      interpret("Sum[1/(n (n + 1)), {n, 2, Infinity}]").unwrap(),
      "1/2"
    );
  }

  // Infinite geometric series with a numeric ratio and a lower bound >= 2
  // reduce to the min = 1 series minus the head terms, giving an exact rational.
  #[test]
  fn infinite_geometric_lower_bound_above_one() {
    assert_eq!(interpret("Sum[(1/3)^n, {n, 2, Infinity}]").unwrap(), "1/6");
    assert_eq!(interpret("Sum[(1/3)^n, {n, 3, Infinity}]").unwrap(), "1/18");
    assert_eq!(interpret("Sum[(1/2)^n, {n, 2, Infinity}]").unwrap(), "1/2");
    assert_eq!(interpret("Sum[(2/3)^n, {n, 2, Infinity}]").unwrap(), "4/3");
  }

  // The same head-subtraction reduction applies to any series whose min = 1
  // form is known in closed form, e.g. the Basel-type zeta sums.
  #[test]
  fn infinite_zeta_lower_bound_above_one() {
    assert_eq!(
      interpret("Sum[1/n^2, {n, 2, Infinity}]").unwrap(),
      "-1 + Pi^2/6"
    );
    assert_eq!(
      interpret("Sum[1/n^2, {n, 3, Infinity}]").unwrap(),
      "-5/4 + Pi^2/6"
    );
  }

  // A divergent series (harmonic) stays unevaluated.
  #[test]
  fn infinite_sum_lower_bound_above_one_stays_unevaluated() {
    assert_eq!(
      interpret("Sum[1/n, {n, 2, Infinity}]").unwrap(),
      "Sum[n^(-1), {n, 2, Infinity}]"
    );
  }

  // A symbolic geometric series with lower bound > 1 closes to
  // -(x^m/(-1 + x)), matching wolframscript.
  #[test]
  fn infinite_geometric_symbolic_lower_bound_above_one() {
    assert_eq!(
      interpret("Sum[x^n, {n, 2, Infinity}]").unwrap(),
      "-(x^2/(-1 + x))"
    );
  }

  // The c^(-n) form (an integer base with a negated exponent) is recognized as
  // a geometric series with ratio 1/c, like the equivalent (1/c)^n form.
  #[test]
  fn infinite_geometric_negative_exponent() {
    assert_eq!(interpret("Sum[2^(-n), {n, 1, Infinity}]").unwrap(), "1");
    assert_eq!(interpret("Sum[3^(-n), {n, 1, Infinity}]").unwrap(), "1/2");
    assert_eq!(interpret("Sum[2^(-n), {n, 0, Infinity}]").unwrap(), "2");
    // Combined with the lower-bound reduction.
    assert_eq!(interpret("Sum[2^(-n), {n, 2, Infinity}]").unwrap(), "1/2");
    // An integer exponent multiple folds into the effective ratio 1/4.
    assert_eq!(interpret("Sum[2^(-2 n), {n, 1, Infinity}]").unwrap(), "1/3");
    // A coefficient is carried through.
    assert_eq!(interpret("Sum[5 2^(-n), {n, 1, Infinity}]").unwrap(), "5");
  }

  // A geometric series with |ratio| >= 1 diverges and must stay unevaluated
  // (regression: the min = 0 closed form previously fired without a
  // convergence check, e.g. Sum[(3/2)^n, {n, 0, Infinity}] wrongly gave -2).
  #[test]
  fn divergent_geometric_stays_unevaluated() {
    assert_eq!(
      interpret("Sum[(3/2)^n, {n, 0, Infinity}]").unwrap(),
      "Sum[(3/2)^n, {n, 0, Infinity}]"
    );
    assert_eq!(
      interpret("Sum[2^(2 n), {n, 0, Infinity}]").unwrap(),
      "Sum[2^(2*n), {n, 0, Infinity}]"
    );
  }

  // A convergent geometric series with a negative ratio is still recognized.
  #[test]
  fn infinite_geometric_negative_ratio() {
    assert_eq!(interpret("Sum[(-1/2)^n, {n, 0, Infinity}]").unwrap(), "2/3");
  }

  #[test]
  fn infinite_sum_short_iterator_form() {
    // `{n, max}` is shorthand for `{n, 1, max}`, including when max is Infinity
    // — previously failed with "Sum: iterator bounds must be integers".
    assert_eq!(interpret("Sum[1/n^2, {n, Infinity}]").unwrap(), "Pi^2/6");
    assert_eq!(interpret("Sum[k, {k, 10}]").unwrap(), "55");
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
      "Sum[n^(-1), {n, 1, Infinity}]"
    );
  }

  // Symbolic geometric series Sum[c base^k, {k, 0, Infinity}] = c/(1 - base).
  #[test]
  fn infinite_sum_geometric_symbolic() {
    assert_eq!(
      interpret("Sum[r^k, {k, 0, Infinity}]").unwrap(),
      "(1 - r)^(-1)"
    );
    assert_eq!(
      interpret("Sum[x^n, {n, 0, Infinity}]").unwrap(),
      "(1 - x)^(-1)"
    );
  }

  #[test]
  fn infinite_sum_geometric_with_coefficient() {
    assert_eq!(
      interpret("Sum[a r^k, {k, 0, Infinity}]").unwrap(),
      "a/(1 - r)"
    );
    assert_eq!(
      interpret("Sum[3 r^k, {k, 0, Infinity}]").unwrap(),
      "3/(1 - r)"
    );
  }

  #[test]
  fn infinite_sum_geometric_with_divisor() {
    assert_eq!(
      interpret("Sum[r^k/5, {k, 0, Infinity}]").unwrap(),
      "1/(5*(1 - r))"
    );
  }

  // A linear coefficient is an arithmetico-geometric series, not a plain
  // geometric one: Sum[k r^k, {k, 0, Infinity}] = r/(-1 + r)^2 (the k = 0 term
  // is zero, so it equals the k = 1 sum), matching wolframscript.
  #[test]
  fn infinite_sum_arith_geometric_from_zero() {
    assert_eq!(
      interpret("Sum[k r^k, {k, 0, Infinity}]").unwrap(),
      "r/(-1 + r)^2"
    );
  }

  // Logarithmic (Mercator) series Sum[base^k/k, {k, 1, Infinity}] = -Log[1-base].
  #[test]
  fn infinite_sum_logarithmic() {
    assert_eq!(
      interpret("Sum[x^k/k, {k, 1, Infinity}]").unwrap(),
      "-Log[1 - x]"
    );
    assert_eq!(
      interpret("Sum[(-1)^k x^k/k, {k, 1, Infinity}]").unwrap(),
      "-Log[1 + x]"
    );
    assert_eq!(
      interpret("Sum[2 x^k/k, {k, 1, Infinity}]").unwrap(),
      "-2*Log[1 - x]"
    );
    // Convergent numeric base evaluates to a number.
    assert_eq!(
      interpret("Sum[(1/2)^k/k, {k, 1, Infinity}]").unwrap(),
      "Log[2]"
    );
    assert_eq!(
      interpret("Sum[(1/3)^k/k, {k, 1, Infinity}]").unwrap(),
      "Log[3/2]"
    );
  }

  // The Mercator series written with the base in the denominator: 1/(b^k k) and
  // b^(-k)/k must be recognised as base 1/b, e.g. Sum[1/(2^k k)] = Log[2].
  #[test]
  fn infinite_sum_logarithmic_reciprocal_base() {
    assert_eq!(
      interpret("Sum[1/(2^k k), {k, 1, Infinity}]").unwrap(),
      "Log[2]"
    );
    assert_eq!(
      interpret("Sum[1/(k 2^k), {k, 1, Infinity}]").unwrap(),
      "Log[2]"
    );
    assert_eq!(
      interpret("Sum[2^(-k)/k, {k, 1, Infinity}]").unwrap(),
      "Log[2]"
    );
    assert_eq!(
      interpret("Sum[1/(3^k k), {k, 1, Infinity}]").unwrap(),
      "Log[3/2]"
    );
    assert_eq!(
      interpret("Sum[1/(5^k k), {k, 1, Infinity}]").unwrap(),
      "Log[5/4]"
    );
    // A constant coefficient is preserved: 3 * Log[2].
    assert_eq!(
      interpret("Sum[3/(2^k k), {k, 1, Infinity}]").unwrap(),
      "3*Log[2]"
    );
  }

  // Exponential series Sum[base^k/k!, {k, 0, Infinity}] = E^base.
  #[test]
  fn infinite_sum_exponential() {
    assert_eq!(interpret("Sum[x^k/k!, {k, 0, Infinity}]").unwrap(), "E^x");
    assert_eq!(interpret("Sum[1/k!, {k, 0, Infinity}]").unwrap(), "E");
    assert_eq!(interpret("Sum[2^k/k!, {k, 0, Infinity}]").unwrap(), "E^2");
    assert_eq!(
      interpret("Sum[(-1)^k/k!, {k, 0, Infinity}]").unwrap(),
      "E^(-1)"
    );
  }

  #[test]
  fn infinite_sum_exponential_with_coefficient() {
    assert_eq!(
      interpret("Sum[3 x^k/k!, {k, 0, Infinity}]").unwrap(),
      "3*E^x"
    );
  }

  // Starting at k = 1 drops the constant k = 0 term: E^base - 1.
  #[test]
  fn infinite_sum_exponential_from_one() {
    assert_eq!(interpret("Sum[1/k!, {k, 1, Infinity}]").unwrap(), "-1 + E");
    assert_eq!(
      interpret("Sum[x^k/k!, {k, 1, Infinity}]").unwrap(),
      "-1 + E^x"
    );
  }

  // A non-constant coefficient (k/k!) is not the c base^k/k! shape.
  #[test]
  fn infinite_sum_exponential_not_misapplied() {
    assert_eq!(
      interpret("Sum[k/k!, {k, 1, Infinity}]").unwrap(),
      "Sum[k/k!, {k, 1, Infinity}]"
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
  fn sum_constant_symbolic() {
    // Sum of constant body (not dependent on iteration variable)
    assert_eq!(interpret("Sum[1, {i, 1, n}]").unwrap(), "n");
  }

  #[test]
  fn sum_constant_c_symbolic() {
    assert_eq!(interpret("Sum[c, {i, 1, n}]").unwrap(), "c*n");
  }

  #[test]
  fn sum_k_symbolic() {
    assert_eq!(interpret("Sum[k, {k, 1, n}]").unwrap(), "(n*(1 + n))/2");
  }

  #[test]
  fn sum_k_cubed_symbolic() {
    assert_eq!(
      interpret("Sum[k^3, {k, 1, n}]").unwrap(),
      "(n^2*(1 + n)^2)/4"
    );
  }

  #[test]
  fn sum_k_cubed_numeric() {
    assert_eq!(interpret("Sum[k^3, {k, 1, 10}]").unwrap(), "3025");
  }

  #[test]
  fn sum_k_fourth_symbolic() {
    assert_eq!(
      interpret("Sum[k^4, {k, 1, n}]").unwrap(),
      "(n*(1 + n)*(1 + 2*n)*(-1 + 3*n + 3*n^2))/30"
    );
  }

  #[test]
  fn sum_k_fourth_numeric() {
    assert_eq!(interpret("Sum[k^4, {k, 1, 5}]").unwrap(), "979");
  }

  #[test]
  fn sum_k_fifth_symbolic() {
    assert_eq!(
      interpret("Sum[k^5, {k, 1, n}]").unwrap(),
      "(n^2*(1 + n)^2*(-1 + 2*n + 2*n^2))/12"
    );
  }

  #[test]
  fn sum_k_fifth_numeric() {
    assert_eq!(interpret("Sum[k^5, {k, 1, 5}]").unwrap(), "4425");
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
  fn sum_geometric_infinite_from_zero() {
    assert_eq!(interpret("Sum[1/2^n, {n, 0, Infinity}]").unwrap(), "2");
  }

  #[test]
  fn sum_k_symbolic_both_bounds() {
    assert_eq!(interpret("Sum[k, {k, n, 2 n}]").unwrap(), "(3*n*(1 + n))/2");
  }

  #[test]
  fn sum_indefinite_linear() {
    // Sum[i, i] = antidifference of i = ∑_{k=1}^{i-1} k = (i-1)*i/2.
    assert_eq!(interpret("Sum[i, i]").unwrap(), "((-1 + i)*i)/2");
  }

  #[test]
  fn sum_indefinite_cubic() {
    // Sum[i^3, i] = ((i-1)*i/2)^2.
    assert_eq!(interpret("Sum[i^3, i]").unwrap(), "((-1 + i)^2*i^2)/4");
  }

  #[test]
  fn sum_indefinite_constant() {
    // Sum[1, i] = i (antidifference of 1).
    assert_eq!(interpret("Sum[1, i]").unwrap(), "i");
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
  fn sum_harmonic_number_order_1() {
    // Sum[1/k, {k, 1, n}] should give HarmonicNumber[n], not HarmonicNumber[n, 1]
    assert_eq!(
      interpret("Sum[1/k, {k, 1, n}]").unwrap(),
      "HarmonicNumber[n]"
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

  #[test]
  fn rational_coefficients_in_other_variable() {
    // Matches wolframscript: coefficient of x^0 is 2/(y-3), coefficient of
    // x^1 is 1/(y-3) + 1/(y-2). Rendered with ^(-1) because Woxi's default
    // output uses that form for simple reciprocals of Plus expressions.
    assert_eq!(
      interpret("CoefficientList[(x + 2)/(y - 3) + x/(y - 2), x]").unwrap(),
      "{2/(-3 + y), (-3 + y)^(-1) + (-2 + y)^(-1)}"
    );
  }

  #[test]
  fn series_data_input_is_normalized() {
    // Regression for mathics algebra.py:909 — CoefficientList applied to a
    // SeriesData must reduce it to the underlying polynomial first and then
    // trim trailing zeros, matching wolframscript.
    assert_eq!(
      interpret("CoefficientList[Series[2x, {x, 0, 9}], x]").unwrap(),
      "{0, 2}"
    );
    assert_eq!(
      interpret("CoefficientList[Series[Log[1-x], {x, 0, 9}], x]").unwrap(),
      "{0, -1, -1/2, -1/3, -1/4, -1/5, -1/6, -1/7, -1/8, -1/9}"
    );
  }

  #[test]
  fn multivariate_two_vars() {
    // CoefficientList[poly, {x, y}] produces a rectangular matrix m where
    // m[[i, j]] is the coefficient of x^(i-1) y^(j-1). Regression for
    // mathics algebra.py:CoefficientList.
    assert_eq!(
      interpret("CoefficientList[a x^2 + b y^3 + c x + d y + 5, {x, y}]")
        .unwrap(),
      "{{5, d, 0, b}, {c, 0, 0, 0}, {a, 0, 0, 0}}"
    );
  }

  #[test]
  fn multivariate_three_vars() {
    // 3-variable case produces a 3-dimensional rectangular array.
    assert_eq!(
      interpret("CoefficientList[(x - 2 y + 3 z)^3, {x, y, z}]").unwrap(),
      "{{{0, 0, 0, 27}, {0, 0, -54, 0}, {0, 36, 0, 0}, {-8, 0, 0, 0}}, \
       {{0, 0, 27, 0}, {0, -36, 0, 0}, {12, 0, 0, 0}, {0, 0, 0, 0}}, \
       {{0, 9, 0, 0}, {-6, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}, \
       {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}}"
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
    // Variables preserves first-appearance order in the canonical expression form
    assert_eq!(
      interpret("Variables[a x^2 + b x + c]").unwrap(),
      "{c, b, x, a}"
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

  #[test]
  fn log_of_product() {
    assert_eq!(
      interpret("PowerExpand[Log[x*y]]").unwrap(),
      "Log[x] + Log[y]"
    );
  }

  #[test]
  fn log_of_power() {
    assert_eq!(interpret("PowerExpand[Log[x^y]]").unwrap(), "y*Log[x]");
  }

  #[test]
  fn log_of_product_three() {
    assert_eq!(
      interpret("PowerExpand[Log[x*y*z]]").unwrap(),
      "Log[x] + Log[y] + Log[z]"
    );
  }

  #[test]
  fn log_of_quotient() {
    assert_eq!(
      interpret("PowerExpand[Log[x/y]]").unwrap(),
      "Log[x] - Log[y]"
    );
  }

  #[test]
  fn log_of_sqrt() {
    assert_eq!(interpret("PowerExpand[Log[Sqrt[x]]]").unwrap(), "Log[x]/2");
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

mod log_base_exact_power {
  use super::*;

  // Log[base, 1/base^k] collapses to the negative exponent.
  #[test]
  fn reciprocal_power() {
    assert_eq!(interpret("Log[2, 1/8]").unwrap(), "-3");
    assert_eq!(interpret("Log[2, 1/4]").unwrap(), "-2");
    assert_eq!(interpret("Log[3, 1/9]").unwrap(), "-2");
    assert_eq!(interpret("Log[10, 1/1000]").unwrap(), "-3");
  }

  // The exponent can be a fraction when the base is itself a power.
  #[test]
  fn fractional_exponent() {
    assert_eq!(interpret("Log[4, 1/2]").unwrap(), "-1/2");
    assert_eq!(interpret("Log[8, 2]").unwrap(), "1/3");
    assert_eq!(interpret("Log[9, 1/3]").unwrap(), "-1/2");
  }

  // Integer powers still collapse, non-powers stay as the log ratio.
  #[test]
  fn integer_and_non_power() {
    assert_eq!(interpret("Log[2, 8]").unwrap(), "3");
    assert_eq!(interpret("Log[2, 1/3]").unwrap(), "-(Log[3]/Log[2])");
    assert_eq!(interpret("Log[2, 3]").unwrap(), "Log[3]/Log[2]");
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

mod cases {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn product_1() {
    assert_case(r#"Product[k, {k, i, n}]"#, r#"Pochhammer[i, 1 - i + n]"#);
  }
  #[test]
  fn product_2() {
    assert_case(r#"Product[k, {k, i, n}]; Product[k, {k, 1, n}]"#, r#"n!"#);
  }
  #[test]
  fn product_3() {
    assert_case(
      r#"Product[k, {k, i, n}]; Product[k, {k, 1, n}]; Product[k, {k, n}]"#,
      r#"n!"#,
    );
  }
  #[test]
  fn product_4() {
    assert_case(
      r#"Product[k, {k, i, n}]; Product[k, {k, 1, n}]; Product[k, {k, n}]; Product[k, {k, 3, n}]"#,
      r#"n! / 2"#,
    );
  }
  #[test]
  fn product_5() {
    assert_case(
      r#"Product[k, {k, i, n}]; Product[k, {k, 1, n}]; Product[k, {k, n}]; Product[k, {k, 3, n}]; Product[x^k, {k, 2, 20, 2}]"#,
      r#"x ^ 110"#,
    );
  }
  #[test]
  fn product_6() {
    assert_case(
      r#"Product[k, {k, i, n}]; Product[k, {k, 1, n}]; Product[k, {k, n}]; Product[k, {k, 3, n}]; Product[x^k, {k, 2, 20, 2}]; Product[2 ^ i, {i, 1, n}]"#,
      r#"2^((n*(1 + n))/2)"#,
    );
  }
  #[test]
  fn product_7() {
    assert_case(
      r#"Product[k, {k, i, n}]; Product[k, {k, 1, n}]; Product[k, {k, n}]; Product[k, {k, 3, n}]; Product[x^k, {k, 2, 20, 2}]; Product[2 ^ i, {i, 1, n}]; Product[f[i], {i, 1, 7}]"#,
      r#"f[1]*f[2]*f[3]*f[4]*f[5]*f[6]*f[7]"#,
    );
  }
  #[test]
  fn primorial() {
    assert_case(
      r#"Product[k, {k, i, n}]; Product[k, {k, 1, n}]; Product[k, {k, n}]; Product[k, {k, 3, n}]; Product[x^k, {k, 2, 20, 2}]; Product[2 ^ i, {i, 1, n}]; Product[f[i], {i, 1, 7}]; Primorial[0] = 1; Primorial[n_Integer] := Product[Prime[k], {k, 1, n}]; Primorial[12]"#,
      r#"7420738134810"#,
    );
  }
  #[test]
  fn sum_1() {
    assert_case(r#"Sum[k, {k, 1, 10}]"#, r#"55"#);
  }
  #[test]
  fn sum_2() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]"#,
      r#"(n*(1 + n))/2"#,
    );
  }
  #[test]
  fn sum_3() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]"#,
      r#"(-1 + 2^k)/2^k"#,
    );
  }
  #[test]
  fn sum_4() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]"#,
      r#"1"#,
    );
  }
  #[test]
  fn sum_5() {
    assert_case(
      r#"Sum[k, {k, 1, 10}]; Sum[k, {k, 1, n}]; Sum[1 / 2 ^ i, {i, 1, k}]; Sum[1 / 2 ^ i, {i, 1, Infinity}]; Sum[1 / ((-1)^k (2k + 1)), {k, 0, Infinity}]"#,
      r#"Pi / 4"#,
    );
  }
  #[test]
  fn r_solve_1() {
    assert_case(r#"RSolve[a[n] == a[n+1], a[n], n]"#, r#"{{a[n] -> C[1]}}"#);
  }
  #[test]
  fn r_solve_2() {
    assert_case(
      r#"RSolve[a[n] == a[n+1], a[n], n]; RSolve[{a[n + 2] == a[n]}, a, n]"#,
      r#"{{a -> Function[{n}, C[1] + (-1)^n*C[2]]}}"#,
    );
  }
  #[test]
  fn r_solve_3() {
    assert_case(
      r#"RSolve[a[n] == a[n+1], a[n], n]; RSolve[{a[n + 2] == a[n]}, a, n]; RSolve[{a[n + 2] == a[n], a[0] == 1}, a, n]"#,
      r#"{{a -> Function[{n}, (-1)^n + C[1] - (-1)^n*C[1]]}}"#,
    );
  }
  #[test]
  fn r_solve_4() {
    assert_case(
      r#"RSolve[a[n] == a[n+1], a[n], n]; RSolve[{a[n + 2] == a[n]}, a, n]; RSolve[{a[n + 2] == a[n], a[0] == 1}, a, n]; RSolve[{a[n + 2] == a[n], a[0] == 1, a[1] == 4}, a, n]"#,
      r#"{{a -> Function[{n}, (5 - 3*(-1)^n)/2]}}"#,
    );
  }
  #[test]
  fn r_solve_integer_coefficient() {
    // Regression: `2*a[n]` arrives as FunctionCall Times after evaluation
    // and used to be silently dropped, leaving RSolve unevaluated.
    assert_case(
      r#"RSolve[{a[n + 1] == 2 a[n], a[0] == 1}, a, n]"#,
      r#"{{a -> Function[{n}, 2^n]}}"#,
    );
  }
  // First-order homogeneous recurrence with one initial condition. The closed
  // form is `v * r^(n - k)`, with the coefficient's factors of r folded into
  // the exponent the way wolframscript does (e.g. a[1]==6, r=2 → 3*2^n, not
  // the value-equal 6*2^(-1 + n); a[2]==5, r=2 → 5*2^(-2 + n) rather than
  // 5*2^n/4).
  #[test]
  fn r_solve_first_order_with_ic() {
    assert_case(
      r#"RSolve[{a[n] == 2 a[n-1], a[1] == 1}, a[n], n]"#,
      r#"{{a[n] -> 2^(-1 + n)}}"#,
    );
    assert_case(
      r#"RSolve[{a[n] == 2 a[n-1], a[0] == 3}, a[n], n]"#,
      r#"{{a[n] -> 3*2^n}}"#,
    );
    assert_case(
      r#"RSolve[{a[n] == 2 a[n-1], a[2] == 5}, a[n], n]"#,
      r#"{{a[n] -> 5*2^(-2 + n)}}"#,
    );
    assert_case(
      r#"RSolve[{a[n] == 2 a[n-1], a[1] == 6}, a[n], n]"#,
      r#"{{a[n] -> 3*2^n}}"#,
    );
    assert_case(
      r#"RSolve[{a[n] == 3 a[n-1], a[1] == 1}, a[n], n]"#,
      r#"{{a[n] -> 3^(-1 + n)}}"#,
    );
  }
  // The first-order general solution anchors its constant at n = 1, so the
  // single root carries the exponent n-1 (matching wolframscript). Higher-order
  // solutions keep the exponent n.
  #[test]
  fn r_solve_first_order_general() {
    assert_case(
      r#"RSolve[a[n] == 2 a[n-1], a[n], n]"#,
      r#"{{a[n] -> 2^(-1 + n)*C[1]}}"#,
    );
    assert_case(
      r#"RSolve[a[n] == -a[n-1], a[n], n]"#,
      r#"{{a[n] -> (-1)^(-1 + n)*C[1]}}"#,
    );
  }
  // A repeated characteristic root of multiplicity m contributes m terms whose
  // j-th occurrence carries the factor n^j (j = 0, 1, …). Previously the second
  // term dropped the n factor, collapsing to a one-parameter family.
  #[test]
  fn r_solve_repeated_roots() {
    // Double root 2.
    assert_case(
      r#"RSolve[a[n] == 4 a[n-1] - 4 a[n-2], a[n], n]"#,
      r#"{{a[n] -> 2^n*C[1] + 2^n*n*C[2]}}"#,
    );
    // Double root 1.
    assert_case(
      r#"RSolve[a[n] == 2 a[n-1] - a[n-2], a[n], n]"#,
      r#"{{a[n] -> C[1] + n*C[2]}}"#,
    );
    // Triple root 1.
    assert_case(
      r#"RSolve[a[n] == 3 a[n-1] - 3 a[n-2] + a[n-3], a[n], n]"#,
      r#"{{a[n] -> C[1] + n*C[2] + n^2*C[3]}}"#,
    );
  }
  #[test]
  fn r_solve_second_order_distinct_roots() {
    assert_case(
      r#"RSolve[{a[n + 2] == 5 a[n + 1] - 6 a[n], a[0] == 0, a[1] == 1}, a, n]"#,
      r#"{{a -> Function[{n}, -2^n + 3^n]}}"#,
    );
  }
  #[test]
  fn r_solve_inhomogeneous_stays_unevaluated() {
    // Regression: the constant/`n` forcing term used to be silently
    // dropped, returning the (wrong) homogeneous solution {{a[n] -> C[1]}}.
    assert_case(
      r#"RSolve[a[n] == a[n+1] + n, a[n], n]"#,
      r#"RSolve[a[n] == n + a[1 + n], a[n], n]"#,
    );
  }
  #[test]
  fn r_solve_value_function() {
    assert_case(
      r#"RSolveValue[{a[n + 1] == 2 a[n], a[0] == 1}, a, n]"#,
      r#"Function[{n}, 2^n]"#,
    );
  }
  #[test]
  fn r_solve_value_expression() {
    assert_case(
      r#"RSolveValue[{a[n + 1] == 3 a[n], a[0] == 2}, a[n], n]"#,
      r#"2*3^n"#,
    );
  }
  #[test]
  fn r_solve_value_at_point() {
    assert_case(
      r#"RSolveValue[{a[n + 1] == 2 a[n], a[0] == 1}, a[3], n]"#,
      r#"8"#,
    );
  }
  #[test]
  fn r_solve_value_general_solution() {
    assert_case(r#"RSolveValue[a[n] == a[n+1], a[n], n]"#, r#"C[1]"#);
  }
  #[test]
  fn r_solve_value_second_order() {
    assert_case(
      r#"RSolveValue[{a[n + 2] == a[n], a[0] == 1, a[1] == 4}, a[n], n]"#,
      r#"(5 - 3*(-1)^n)/2"#,
    );
  }
  #[test]
  fn sum_6() {
    assert_case(
      r#"Precision[1]; 1 / Infinity; Infinity + 100; Sum[1/x^2, {x, 1, Infinity}]"#,
      r#"Pi ^ 2 / 6"#,
    );
  }
  #[test]
  fn product_8() {
    assert_case(r#"Product[1 + 1 / i ^ 2, {i, Infinity}]"#, r#"Sinh[Pi]/Pi"#);
  }
  #[test]
  fn product_minus_one_over_quartic_from_two_to_infinity() {
    // wolframscript: Product[1 - 1/i^4, {i, 2, Infinity}] = Sinh[Pi]/(4*Pi).
    assert_case(
      r#"Product[1 - 1/i^4, {i, 2, Infinity}]"#,
      r#"Sinh[Pi]/(4*Pi)"#,
    );
  }
  #[test]
  fn product_minus_one_over_quartic_finite() {
    // Finite case still uses the standard numeric path.
    assert_case(r#"Product[1 - 1/i^4, {i, 2, 5}]"#, r#"221/240"#);
  }

  // Infinite products of rational functions that split over the integers.
  // The convergent ones use the Gamma-ratio closed form
  // ∏_j Γ(n0-s_j) / ∏_i Γ(n0-r_i).
  #[test]
  fn product_rational_telescoping_one_minus_recip_square() {
    assert_case(r#"Product[1 - 1/n^2, {n, 2, Infinity}]"#, r#"1/2"#);
    assert_case(r#"Product[1 - 1/n^2, {n, 3, Infinity}]"#, r#"2/3"#);
  }

  #[test]
  fn product_rational_telescoping_reciprocal() {
    assert_case(r#"Product[n^2/(n^2 - 1), {n, 2, Infinity}]"#, r#"2"#);
    assert_case(r#"Product[1/(1 - 1/n^2), {n, 2, Infinity}]"#, r#"2"#);
  }

  #[test]
  fn product_rational_telescoping_shifted_factors() {
    assert_case(
      r#"Product[((n + 2)(n - 2))/((n + 1)(n - 1)), {n, 3, Infinity}]"#,
      r#"1/4"#,
    );
  }

  // Σ(numerator roots) > Σ(denominator roots) → the product converges to 0.
  #[test]
  fn product_rational_converges_to_zero() {
    assert_case(r#"Product[(n - 1)/n, {n, 2, Infinity}]"#, r#"0"#);
    assert_case(r#"Product[(n + 1)/(n + 2), {n, 1, Infinity}]"#, r#"0"#);
  }

  // Σ(numerator roots) < Σ(denominator roots) → divergent, left unevaluated.
  #[test]
  fn product_rational_divergent_unevaluated() {
    assert_case(
      r#"Product[(2 + n)/(1 + n), {n, 1, Infinity}]"#,
      r#"Product[(2 + n)/(1 + n), {n, 1, Infinity}]"#,
    );
  }

  // Constant body over an infinite range.
  #[test]
  fn product_constant_infinite() {
    assert_case(r#"Product[1, {k, 1, Infinity}]"#, r#"1"#);
    assert_case(r#"Product[1/2, {k, 1, Infinity}]"#, r#"0"#);
    assert_case(r#"Product[0, {k, 1, Infinity}]"#, r#"0"#);
  }

  #[test]
  fn product_constant_infinite_divergent() {
    assert_case(
      r#"Product[2, {k, 1, Infinity}]"#,
      r#"Product[2, {k, 1, Infinity}]"#,
    );
  }
}
