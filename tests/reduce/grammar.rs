//! One explicit harness case for every row of the scoped grammar.
//!
//! Cases marked ignored are executable specifications for later roadmap gates,
//! not claims about the current implementation.

use super::assert_reduces;

#[test]
fn exact_rational_affine_terms() {
  assert_reduces("Reduce[3*x + 2 == 11, x, Reals]", "x == 3");
  assert_reduces("Reduce[x == 1/3, x, Rationals]", "x == 1/3");
}

#[test]
fn all_order_relations_and_disequality() {
  for (relation, expected) in [
    ("==", "x == 1"),
    ("!=", "x != 1"),
    ("<", "x < 1"),
    ("<=", "x <= 1"),
    (">", "x > 1"),
    (">=", "x >= 1"),
  ] {
    assert_reduces(&format!("Reduce[x {relation} 1, x, Reals]"), expected);
  }
}

#[test]
fn booleans_and_not() {
  assert_reduces("Reduce[True, x, Reals]", "True");
  assert_reduces("Reduce[False, x, Reals]", "False");
  assert_reduces(
    "Reduce[x < 1 && x > -1, x, Reals]",
    "Inequality[-1, Less, x, Less, 1]",
  );
  assert_reduces("Reduce[x < 0 || x > 1, x, Reals]", "x < 0 || x > 1");
  assert_reduces("Reduce[!(x < 1), x, Reals]", "x >= 1");
}

#[test]
fn all_scoped_domains_have_a_surface_case() {
  assert_reduces("Reduce[x == 1, x, Reals]", "x == 1");
  assert_reduces("Reduce[x == 1, x, Rationals]", "x == 1");
  assert_reduces("Reduce[x == 1, x, Integers]", "x == 1");
}

#[test]
fn divisibility_and_mod_surface_forms() {
  assert_reduces(
    "Reduce[Mod[x, 3] == 1, x, Integers]",
    "Element[x, Integers] && Mod[x, 3] == 1",
  );
}

#[test]
fn xor_formula() {
  assert_reduces("Reduce[Xor[x < 0, x > 1], x, Reals]", "x < 0 || x > 1");
}

#[test]
fn exists_formula() {
  assert_reduces("Reduce[Exists[y, x < y && y < 1], x, Reals]", "x < 1");
}

#[test]
fn forall_formula() {
  assert_reduces("Reduce[ForAll[y, y < x || y >= x], x, Reals]", "True");
}
