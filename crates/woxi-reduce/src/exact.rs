//! Exact arithmetic primitives shared by the linear decision procedures.

use std::cmp::Ordering;

use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, Zero};

/// A canonical arbitrary-precision rational.
///
/// The denominator is positive, numerator and denominator are coprime, and
/// zero is represented only as `0/1`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rational {
  pub numerator: BigInt,
  pub denominator: BigInt,
}

impl Rational {
  pub fn new(mut numerator: BigInt, mut denominator: BigInt) -> Option<Self> {
    if denominator.is_zero() {
      return None;
    }
    if numerator.is_zero() {
      return Some(Self::integer(BigInt::zero()));
    }
    if denominator.sign() == Sign::Minus {
      numerator = -numerator;
      denominator = -denominator;
    }
    let divisor = gcd(numerator.clone(), denominator.clone());
    Some(Self {
      numerator: numerator / &divisor,
      denominator: denominator / divisor,
    })
  }

  pub fn integer(value: BigInt) -> Self {
    Self {
      numerator: value,
      denominator: BigInt::one(),
    }
  }

  pub fn reciprocal(&self) -> Option<Self> {
    Self::new(self.denominator.clone(), self.numerator.clone())
  }

  pub fn zero() -> Self {
    Self::integer(BigInt::zero())
  }

  pub fn one() -> Self {
    Self::integer(BigInt::one())
  }

  pub fn is_zero(&self) -> bool {
    self.numerator.is_zero()
  }

  pub fn is_integer(&self) -> bool {
    self.denominator.is_one()
  }

  pub fn negated(&self) -> Self {
    Self {
      numerator: -self.numerator.clone(),
      denominator: self.denominator.clone(),
    }
  }

  pub fn add(&self, other: &Self) -> Self {
    Self::new(
      &self.numerator * &other.denominator
        + &other.numerator * &self.denominator,
      &self.denominator * &other.denominator,
    )
    .expect("a sum of valid rationals has a nonzero denominator")
  }

  pub fn subtract(&self, other: &Self) -> Self {
    self.add(&other.negated())
  }

  pub fn multiply(&self, other: &Self) -> Self {
    Self::new(
      &self.numerator * &other.numerator,
      &self.denominator * &other.denominator,
    )
    .expect("a product of valid rationals has a nonzero denominator")
  }

  pub fn checked_divide(&self, other: &Self) -> Option<Self> {
    Some(self.multiply(&other.reciprocal()?))
  }

  pub fn numeric_cmp(&self, other: &Self) -> Ordering {
    (&self.numerator * &other.denominator)
      .cmp(&(&other.numerator * &self.denominator))
  }
}

pub fn gcd(mut left: BigInt, mut right: BigInt) -> BigInt {
  left = left.abs();
  right = right.abs();
  while !right.is_zero() {
    let remainder = &left % &right;
    left = right;
    right = remainder;
  }
  left
}

pub fn lcm(left: &BigInt, right: &BigInt) -> BigInt {
  if left.is_zero() || right.is_zero() {
    return BigInt::zero();
  }
  (left / gcd(left.clone(), right.clone()) * right).abs()
}

pub fn extended_gcd(left: &BigInt, right: &BigInt) -> (BigInt, BigInt, BigInt) {
  let (mut old_remainder, mut remainder) = (left.clone(), right.clone());
  let (mut old_left, mut left_coefficient) = (BigInt::one(), BigInt::zero());
  let (mut old_right, mut right_coefficient) = (BigInt::zero(), BigInt::one());
  while !remainder.is_zero() {
    let quotient = &old_remainder / &remainder;
    (old_remainder, remainder) =
      (remainder.clone(), old_remainder - &quotient * &remainder);
    (old_left, left_coefficient) = (
      left_coefficient.clone(),
      old_left - &quotient * &left_coefficient,
    );
    (old_right, right_coefficient) = (
      right_coefficient.clone(),
      old_right - quotient * &right_coefficient,
    );
  }
  if old_remainder.is_negative() {
    (-old_remainder, -old_left, -old_right)
  } else {
    (old_remainder, old_left, old_right)
  }
}

pub fn euclidean_mod(value: BigInt, modulus: &BigInt) -> BigInt {
  debug_assert!(modulus.is_positive());
  let mut residue = value % modulus;
  if residue.is_negative() {
    residue += modulus;
  }
  residue
}

/// Solves `coefficient*x == value (mod modulus)` as one residue and its
/// reduced positive modulus.
pub fn solve_linear_congruence(
  coefficient: &BigInt,
  value: &BigInt,
  modulus: &BigInt,
) -> Option<(BigInt, BigInt)> {
  if !modulus.is_positive() {
    return None;
  }
  let divisor = gcd(coefficient.clone(), modulus.clone());
  if value % &divisor != BigInt::zero() {
    return None;
  }
  let reduced_coefficient = coefficient / &divisor;
  let reduced_value = value / &divisor;
  let reduced_modulus = modulus / divisor;
  let (_, inverse, _) = extended_gcd(&reduced_coefficient, &reduced_modulus);
  Some((
    euclidean_mod(inverse * reduced_value, &reduced_modulus),
    reduced_modulus,
  ))
}

/// Generalized CRT for positive, not-necessarily-coprime moduli.
pub fn crt_pair(
  left_residue: &BigInt,
  left_modulus: &BigInt,
  right_residue: &BigInt,
  right_modulus: &BigInt,
) -> Option<(BigInt, BigInt)> {
  if !left_modulus.is_positive() || !right_modulus.is_positive() {
    return None;
  }
  let (divisor, left_inverse, _) = extended_gcd(left_modulus, right_modulus);
  let difference = right_residue - left_residue;
  if &difference % &divisor != BigInt::zero() {
    return None;
  }
  let right_factor = right_modulus / &divisor;
  let step = euclidean_mod(difference / &divisor * left_inverse, &right_factor);
  let combined_modulus = left_modulus * &right_factor;
  Some((
    euclidean_mod(left_residue + left_modulus * step, &combined_modulus),
    combined_modulus,
  ))
}

#[cfg(test)]
mod tests {
  use proptest::prelude::*;
  use proptest::test_runner::RngSeed;

  use super::*;

  #[test]
  fn rational_is_reduced_and_has_positive_denominator() {
    let value = Rational::new(BigInt::from(-42), BigInt::from(-30)).unwrap();
    assert_eq!(value.numerator, BigInt::from(7));
    assert_eq!(value.denominator, BigInt::from(5));
  }

  #[test]
  fn rational_zero_is_unique() {
    for denominator in [-17, -1, 1, 23] {
      let value =
        Rational::new(BigInt::zero(), BigInt::from(denominator)).unwrap();
      assert_eq!(value, Rational::integer(BigInt::zero()));
    }
    assert!(Rational::new(BigInt::zero(), BigInt::zero()).is_none());
  }

  #[test]
  fn reciprocal_normalizes_its_sign() {
    let value = Rational::new(BigInt::from(-3), BigInt::from(7)).unwrap();
    let reciprocal = value.reciprocal().unwrap();
    assert_eq!(reciprocal.numerator, BigInt::from(-7));
    assert_eq!(reciprocal.denominator, BigInt::from(3));
    assert!(Rational::integer(BigInt::zero()).reciprocal().is_none());
  }

  #[test]
  fn rational_reduction_is_not_fixed_width() {
    let common = BigInt::from(10).pow(80_u32);
    let value = Rational::new(&common * 6, common * 15).unwrap();
    assert_eq!(value.numerator, BigInt::from(2));
    assert_eq!(value.denominator, BigInt::from(5));
  }

  #[test]
  fn rational_arithmetic_stays_canonical() {
    let half = Rational::new(BigInt::one(), BigInt::from(2)).unwrap();
    let third = Rational::new(BigInt::one(), BigInt::from(3)).unwrap();
    assert_eq!(
      half.add(&third),
      Rational::new(BigInt::from(5), BigInt::from(6)).unwrap()
    );
    assert_eq!(
      half.subtract(&third),
      Rational::new(BigInt::one(), BigInt::from(6)).unwrap()
    );
    assert_eq!(
      half.multiply(&third),
      Rational::new(BigInt::one(), BigInt::from(6)).unwrap()
    );
    assert_eq!(
      half.checked_divide(&third).unwrap(),
      Rational::new(BigInt::from(3), BigInt::from(2)).unwrap()
    );
    assert!(half.checked_divide(&Rational::zero()).is_none());
  }

  #[test]
  fn gcd_and_lcm_are_positive_and_exact() {
    assert_eq!(gcd(BigInt::from(-18), BigInt::from(24)), BigInt::from(6));
    assert_eq!(lcm(&BigInt::from(-18), &BigInt::from(24)), BigInt::from(72));
    assert_eq!(lcm(&BigInt::zero(), &BigInt::from(24)), BigInt::zero());
  }

  #[test]
  fn extended_gcd_returns_a_checked_bezout_identity() {
    for (left, right) in [(240, 46), (-240, 46), (240, -46), (0, 17)] {
      let left = BigInt::from(left);
      let right = BigInt::from(right);
      let (divisor, x, y) = extended_gcd(&left, &right);
      assert_eq!(&left * x + &right * y, divisor);
      assert_eq!(divisor, gcd(left, right));
    }
  }

  #[test]
  fn linear_congruence_reduces_by_the_coefficient_gcd() {
    assert_eq!(
      solve_linear_congruence(
        &BigInt::from(6),
        &BigInt::from(8),
        &BigInt::from(14),
      ),
      Some((BigInt::from(6), BigInt::from(7)))
    );
    assert!(
      solve_linear_congruence(
        &BigInt::from(6),
        &BigInt::from(9),
        &BigInt::from(14),
      )
      .is_none()
    );
  }

  #[test]
  fn generalized_crt_handles_non_coprime_and_inconsistent_moduli() {
    assert_eq!(
      crt_pair(
        &BigInt::from(1),
        &BigInt::from(4),
        &BigInt::from(3),
        &BigInt::from(6),
      ),
      Some((BigInt::from(9), BigInt::from(12)))
    );
    assert!(
      crt_pair(
        &BigInt::from(1),
        &BigInt::from(4),
        &BigInt::from(2),
        &BigInt::from(6),
      )
      .is_none()
    );
  }

  proptest! {
    #![proptest_config(ProptestConfig {
      cases: 512,
      rng_seed: RngSeed::Fixed(0x5eed_0001),
      ..ProptestConfig::default()
    })]

    #[test]
    fn construction_is_canonical_and_idempotent(
      numerator in i64::MIN..=i64::MAX,
      denominator in i64::MIN..=i64::MAX,
    ) {
      prop_assume!(denominator != 0);
      let value = Rational::new(
        BigInt::from(numerator),
        BigInt::from(denominator),
      ).unwrap();
      prop_assert!(value.denominator.is_positive());
      prop_assert_eq!(
        gcd(value.numerator.clone(), value.denominator.clone()),
        BigInt::one(),
      );
      prop_assert_eq!(
        Rational::new(value.numerator.clone(), value.denominator.clone()),
        Some(value),
      );
    }
  }
}
