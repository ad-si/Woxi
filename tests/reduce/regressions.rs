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
