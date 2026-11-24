//! `CylindricalDecomposition` of linear real formulas: each variable is
//! bounded only by the variables listed before it.

use super::assert_reduces;

#[test]
fn later_variable_is_bounded_by_earlier_ones() {
  assert_reduces(
    "CylindricalDecomposition[x + y > 1 && x < 2, {x, y}]",
    "x < 2 && y > 1 - x",
  );
  assert_reduces(
    "CylindricalDecomposition[x + y > 1 && x < 2, {y, x}]",
    "y > -1 && Inequality[1 - y, Less, x, Less, 2]",
  );
}

#[test]
fn competing_bounds_split_the_earlier_variable() {
  assert_reduces(
    "CylindricalDecomposition[y > x && y > -x, {x, y}]",
    "(x < 0 && y > -x) || (x >= 0 && y > x)",
  );
  assert_reduces(
    "CylindricalDecomposition[(a > b + 1 && b < 3 && c > a) || \
     (a < -5 && c == 2), {a, b, c}]",
    "(a < -5 && c == 2) || (a < 4 && b < -1 + a && c > a) || \
     (a >= 4 && b < 3 && c > a)",
  );
}

#[test]
fn closed_bounds_that_meet_give_a_point_cell() {
  assert_reduces(
    "CylindricalDecomposition[x >= 0 && x <= 1 && y >= x && y <= 1, {x, y}]",
    "(Inequality[0, LessEqual, x, Less, 1] && \
     Inequality[x, LessEqual, y, LessEqual, 1]) || (x == 1 && y == 1)",
  );
}

#[test]
fn equations_fix_a_variable() {
  assert_reduces(
    "CylindricalDecomposition[x == 2 y && y > 1, {x, y}]",
    "x > 2 && y == x/2",
  );
}

#[test]
fn unequal_splits_and_trivial_formulas_collapse() {
  assert_reduces("CylindricalDecomposition[x != 1, {x}]", "x < 1 || x > 1");
  assert_reduces("CylindricalDecomposition[x > 0 || x <= 0, {x}]", "True");
  assert_reduces("CylindricalDecomposition[x > 1 && x < 0, {x, y}]", "False");
}

#[test]
fn quantifiers_are_eliminated_first() {
  assert_reduces(
    "CylindricalDecomposition[Exists[y, x < y && y < 1], {x}]",
    "x < 1",
  );
}

#[test]
fn decomposition_drives_find_instance() {
  // Every instance must satisfy the decomposed condition.
  assert_reduces(
    "cd = CylindricalDecomposition[(a > b + 1 && b < 3 && c > a) || \
     (a < -5 && c == 2), {a, b, c}]; \
     And @@ (cd /. FindInstance[cd, {a, b, c}, Reals, 5])",
    "True",
  );
}
