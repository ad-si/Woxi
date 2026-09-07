//! Deterministic, table-driven dense-linear acceptance corpus.
//!
//! Each loop is a family of independently asserted public evaluator cases.
//! The case counter is intentional: Gate 2 requires at least 200 curated
//! formulas in addition to generated property and development-oracle tests.

use super::assert_reduces;

#[test]
fn curated_dense_linear_matrix_has_240_public_cases() {
  let mut cases = 0_usize;

  // Exact integral equality pivots, including negative roots and coefficients.
  for root in -20_i64..20 {
    let coefficient = root.unsigned_abs() as i64 + 2;
    assert_reduces(
      &format!(
        "Reduce[{coefficient}*x == {}, x, Reals]",
        coefficient * root
      ),
      &format!("x == {root}"),
    );
    cases += 1;
  }

  // Exact rational pivots exercise GCD and denominator-sign normalization.
  for numerator in -20_i64..20 {
    let denominator = numerator.unsigned_abs() as i64 + 2;
    let divisor = gcd_i64(numerator, denominator);
    let reduced_numerator = numerator / divisor;
    let reduced_denominator = denominator / divisor;
    let expected = if reduced_denominator == 1 {
      reduced_numerator.to_string()
    } else {
      format!("{reduced_numerator}/{reduced_denominator}")
    };
    assert_reduces(
      &format!("Reduce[{denominator}*x == {numerator}, x, Rationals]"),
      &format!("x == {expected}"),
    );
    cases += 1;
  }

  // Positive and negative pivots verify exact order reversal.
  for boundary in -10_i64..10 {
    let coefficient = boundary.unsigned_abs() as i64 + 2;
    assert_reduces(
      &format!(
        "Reduce[{coefficient}*x < {}, x, Reals]",
        coefficient * boundary
      ),
      &format!("x < {boundary}"),
    );
    cases += 1;
    assert_reduces(
      &format!(
        "Reduce[-{coefficient}*x <= {}, x, Reals]",
        -coefficient * boundary
      ),
      &format!("x >= {boundary}"),
    );
    cases += 1;
  }

  // Target-independent clauses must survive equality substitution.
  for parameter_bound in -10_i64..10 {
    let root = parameter_bound + 3;
    assert_reduces(
      &format!("Reduce[x == {root} && a > {parameter_bound}, x, Reals]"),
      &format!("a > {parameter_bound} && x == {root}"),
    );
    cases += 1;
  }

  // Complementary strict/closed bounds are exact contradictions.
  for boundary in -10_i64..10 {
    assert_reduces(
      &format!("Reduce[x < {boundary} && x >= {boundary}, x, Reals]"),
      "False",
    );
    cases += 1;
  }

  // One-variable Fourier-Motzkin projection.
  for boundary in -10_i64..10 {
    assert_reduces(
      &format!("Reduce[Exists[y, x < y && y < {boundary}], x, Reals]"),
      &format!("x < {boundary}"),
    );
    cases += 1;
  }

  // Two eliminated variables exercise sequential projection.
  for boundary in -10_i64..10 {
    assert_reduces(
      &format!(
        "Reduce[Exists[{{y, z}}, x < y && y < z && z < {boundary}], x, Reals]"
      ),
      &format!("x < {boundary}"),
    );
    cases += 1;
  }

  // Universal duality and depth-three alternation.
  for offset in -10_i64..10 {
    assert_reduces(
      &format!(
        "Reduce[ForAll[y, y < x + {offset} || y >= x + {offset}], x, Reals]"
      ),
      "True",
    );
    cases += 1;
    assert_reduces(
      &format!(
        "Reduce[Exists[y, ForAll[z, Exists[w, w == z + y + {offset}]]], x, Reals]"
      ),
      "True",
    );
    cases += 1;
  }

  assert_eq!(cases, 240, "the curated Gate 2 corpus count is contractual");
}

fn gcd_i64(mut left: i64, mut right: i64) -> i64 {
  left = left.abs();
  right = right.abs();
  while right != 0 {
    (left, right) = (right, left % right);
  }
  left.max(1)
}
