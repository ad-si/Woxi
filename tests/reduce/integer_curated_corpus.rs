//! Deterministic, table-driven Presburger acceptance corpus.
//!
//! These are public evaluator cases, separate from the 25,000-formula exact
//! exhaustive oracle and the development-only Z3 comparison.

use super::assert_reduces;

#[test]
fn curated_presburger_matrix_has_320_public_cases() {
  let mut cases = 0_usize;

  // Strict half-integer bounds must round inward for positive and negative
  // values without conversion through binary floating point.
  for integer_part in -20_i64..20 {
    let numerator = 2 * integer_part + 1;
    assert_reduces(
      &format!("Reduce[x > {numerator}/2, x, Integers]"),
      &format!("Element[x, Integers] && x >= {}", integer_part + 1),
    );
    cases += 1;
    assert_reduces(
      &format!("Reduce[x < {numerator}/2, x, Integers]"),
      &format!("Element[x, Integers] && x <= {integer_part}"),
    );
    cases += 1;
  }

  // Finite intervals are result materialization, not the decision procedure.
  for lower in -20_i64..20 {
    let values = (lower..=lower + 4).collect::<Vec<_>>();
    assert_reduces(
      &format!("Reduce[{lower} <= x <= {}, x, Integers]", lower + 4),
      &equality_disjunction("x", &values),
    );
    cases += 1;
  }

  // Unbounded canonical residue classes.
  for index in 0_i64..40 {
    let modulus = index + 2;
    let residue = (3 * index + 1).rem_euclid(modulus);
    assert_reduces(
      &format!("Reduce[Mod[x, {modulus}] == {residue}, x, Integers]"),
      &format!("Element[x, Integers] && Mod[x, {modulus}] == {residue}"),
    );
    cases += 1;
  }

  // Coefficient-GCD reduction produces the primitive residue class.
  for index in 0_i64..40 {
    let modulus = index + 2;
    let residue = (5 * index + 1).rem_euclid(modulus);
    assert_reduces(
      &format!(
        "Reduce[Mod[2*x, {}] == {}, x, Integers]",
        2 * modulus,
        2 * residue
      ),
      &format!("Element[x, Integers] && Mod[x, {modulus}] == {residue}"),
    );
    cases += 1;
  }

  // Bounded residue classes use exact progression materialization.
  for index in 0_i64..40 {
    let modulus = index + 2;
    let residue = (7 * index + 1).rem_euclid(modulus);
    let upper = 3 * modulus;
    let final_step = (upper - residue) / modulus;
    let values = (0_i64..=final_step)
      .map(|step| residue + step * modulus)
      .collect::<Vec<_>>();
    assert_reduces(
      &format!(
        "Reduce[0 <= x <= {upper} && Mod[x, {modulus}] == {residue}, x, Integers]"
      ),
      &equality_disjunction("x", &values),
    );
    cases += 1;
  }

  // Inconsistent non-coprime congruences are decided without a search cap.
  for offset in 0_i64..20 {
    let even_residue = 2 * offset;
    let odd_residue = 2 * offset + 1;
    assert_reduces(
      &format!(
        "Resolve[Exists[x, Mod[x, 4] == {} && Mod[x, 6] == {}], Integers]",
        even_residue.rem_euclid(4),
        odd_residue.rem_euclid(6)
      ),
      "False",
    );
    cases += 1;
  }

  // Projection of an affine equality retains exactly the parity condition.
  for offset in -10_i64..10 {
    let residue = offset.rem_euclid(2);
    assert_reduces(
      &format!("Reduce[Exists[y, x == 2*y + {offset}], x, Integers]"),
      &format!("Element[x, Integers] && Mod[x, 2] == {residue}"),
    );
    cases += 1;
  }

  // Universal duality over integer order partitions.
  for boundary in -10_i64..10 {
    assert_reduces(
      &format!(
        "Resolve[ForAll[x, x <= {boundary} || x > {boundary}], Integers]"
      ),
      "True",
    );
    cases += 1;
  }

  // Boolean DNF shaping materializes a finite interval with one excluded point.
  for lower in -10_i64..10 {
    let excluded = lower + 2;
    let values = (lower..=lower + 4)
      .filter(|value| *value != excluded)
      .collect::<Vec<_>>();
    assert_reduces(
      &format!(
        "Reduce[{lower} <= x <= {} && x != {excluded}, x, Integers]",
        lower + 4
      ),
      &equality_disjunction("x", &values),
    );
    cases += 1;
  }

  assert_eq!(cases, 320, "the curated Gate 3 corpus count is contractual");
}

fn equality_disjunction(variable: &str, values: &[i64]) -> String {
  if values.is_empty() {
    return "False".to_string();
  }
  values
    .iter()
    .map(|value| format!("{variable} == {value}"))
    .collect::<Vec<_>>()
    .join(" || ")
}
