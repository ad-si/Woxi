use super::assert_reduces;

#[test]
fn exact_integer_bound_tightening() {
  assert_reduces(
    "Reduce[x > 7/3, x, Integers]",
    "Element[x, Integers] && x >= 3",
  );
  assert_reduces(
    "Reduce[x <= -7/3, x, Integers]",
    "Element[x, Integers] && x <= -3",
  );
}

#[test]
fn bounded_integer_interval_regression() {
  assert_reduces("Reduce[2 < x < 5, x, Integers]", "x == 3 || x == 4");
}

#[test]
fn parity_projection() {
  assert_reduces(
    "Reduce[Exists[y, x == 2*y + 1], x, Integers]",
    "Element[x, Integers] && Mod[x, 2] == 1",
  );
}

#[test]
fn inconsistent_congruences() {
  assert_reduces(
    "Resolve[Exists[x, Mod[x, 4] == 1 && Mod[x, 6] == 2], Integers]",
    "False",
  );
}
