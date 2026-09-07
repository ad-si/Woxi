use proptest::prelude::*;
use proptest::test_runner::RngSeed;

use super::assert_reduces;

fn gcd(mut left: i64, mut right: i64) -> i64 {
  left = left.abs();
  right = right.abs();
  while right != 0 {
    (left, right) = (right, left % right);
  }
  left.max(1)
}

fn rational_text(mut numerator: i64, mut denominator: i64) -> String {
  if denominator < 0 {
    numerator = -numerator;
    denominator = -denominator;
  }
  let divisor = gcd(numerator, denominator);
  numerator /= divisor;
  denominator /= divisor;
  if denominator == 1 {
    numerator.to_string()
  } else {
    format!("{numerator}/{denominator}")
  }
}

fn deterministic_config() -> ProptestConfig {
  let cases = std::env::var("WOXI_REDUCE_PROPTEST_CASES")
    .ok()
    .and_then(|value| value.parse().ok())
    .unwrap_or(256);
  ProptestConfig {
    cases,
    rng_seed: RngSeed::Fixed(0x5eed_c0de_2026_0815),
    ..ProptestConfig::default()
  }
}

proptest! {
  #![proptest_config(deterministic_config())]

  #[test]
  fn exact_linear_equations_have_the_exact_root(
    coefficient in -20_i64..=20,
    constant in -50_i64..=50,
  ) {
    prop_assume!(coefficient != 0);
    let input = format!(
      "Reduce[({coefficient})*x + ({constant}) == 0, x, Rationals]"
    );
    let root = rational_text(-constant, coefficient);
    assert_reduces(&input, &format!("x == {root}"));
  }

  #[test]
  fn strict_integer_lower_bounds_shift_exactly(bound in -100_i64..=100) {
    assert_reduces(
      &format!("Reduce[x > {bound}, x, Integers]"),
      &format!("Element[x, Integers] && x >= {}", bound + 1),
    );
  }
}
