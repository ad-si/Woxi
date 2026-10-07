use super::assert_reduces;

#[test]
fn coefficients_larger_than_i128_remain_exact() {
  assert_reduces(
    "Reduce[100000000000000000000000000000000000000000*x == \
     200000000000000000000000000000000000000000, x, Rationals]",
    "x == 2",
  );
}

#[test]
fn negation_flips_strictness_without_approximation() {
  assert_reduces("Reduce[!(3*x < 1), x, Reals]", "x >= 1/3");
}

#[test]
fn real_domain_drops_non_real_polynomial_roots() {
  assert_reduces("Reduce[x^3 == 1, x, Reals]", "x == 1");
  assert_reduces("Reduce[x^2 == -1, x, Reals]", "False");
  assert_reduces(
    "Reduce[x^5 - x - 1 == 0, x, Reals]",
    "x == Root[-1 - #1 + #1^5 & , 1, 0]",
  );
}

#[test]
fn real_domain_filters_high_degree_polynomial_roots() {
  // Degree-100 sparse polynomial with a single real root; the numeric
  // root finders used to overflow and report every Root[…] as real.
  assert_reduces(
    "Length[{ToRules[Reduce[150 - 129*x^14 + 510*x^17 - 298*x^36 - \
     17*x^84 - 24*x^94 + 588*x^96 + 650*x^98 - 841*x^99 - 6*x^100 == 0, \
     x, Reals]]}]",
    "1",
  );
}
