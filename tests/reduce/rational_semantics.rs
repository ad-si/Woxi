use super::assert_reduces;

#[test]
fn exact_linear_equation_and_bounds() {
  assert_reduces("Reduce[2*x + 1 == 0, x, Reals]", "x == -1/2");
  assert_reduces(
    "Reduce[1/3 < x && x <= 5/7, x, Rationals]",
    "Inequality[1/3, Less, x, LessEqual, 5/7]",
  );
}

#[test]
fn target_independent_condition_survives() {
  assert_reduces("Reduce[x == 2 && a > 0, x, Reals]", "a > 0 && x == 2");
}

#[test]
fn projected_open_polyhedron() {
  assert_reduces(
    "Reduce[Exists[y, 0 < y && y < x && x < 2], x, Rationals]",
    "Inequality[0, Less, x, Less, 2]",
  );
}

#[test]
fn universally_true_partition() {
  assert_reduces("Resolve[ForAll[x, x <= a || x > a], Reals]", "True");
}
