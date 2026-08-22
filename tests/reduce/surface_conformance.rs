use super::assert_reduces;

#[test]
fn canonical_surface_examples() {
  for (input, expected) in [
    ("Reduce[x^2 == 4, x]", "x == -2 || x == 2"),
    (
      "Reduce[x > 2, x, Integers]",
      "Element[x, Integers] && x >= 3",
    ),
    (
      "Reduce[-1 < x < 1, x, Reals]",
      "Inequality[-1, Less, x, Less, 1]",
    ),
  ] {
    assert_reduces(input, expected);
  }
}
