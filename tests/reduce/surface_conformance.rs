use super::assert_reduces;
use woxi::interpret;

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

#[test]
fn rational_domain_membership() {
  assert_eq!(
    interpret("Reduce[x > 0, x, Rationals]").unwrap(),
    "Element[x, Rationals] && x > 0"
  );
  assert_eq!(
    interpret("Reduce[x == 1/2, x, Rationals]").unwrap(),
    "x == 1/2"
  );
  assert_eq!(
    interpret("Reduce[x == 1 || x == 2, x, Rationals]").unwrap(),
    "x == 1 || x == 2"
  );
  assert_eq!(
    interpret("Reduce[x == 1 || x > 2, x, Rationals]").unwrap(),
    "Element[x, Rationals] && (x == 1 || x > 2)"
  );
  assert_eq!(
    interpret("Reduce[x > 0 && y < 1, {x, y}, Rationals]").unwrap(),
    "Element[x | y, Rationals] && x > 0 && y < 1"
  );
  assert_eq!(
    interpret("Reduce[x == 1 && y > 0, {x, y}, Rationals]").unwrap(),
    "Element[y, Rationals] && x == 1 && y > 0"
  );
  assert_eq!(
    interpret("Reduce[True, x, Rationals]").unwrap(),
    "Element[x, Rationals]"
  );
  assert_eq!(
    interpret("Reduce[True, {x, y}, Rationals]").unwrap(),
    "Element[x, Rationals] && Element[y, Rationals]"
  );
  assert_eq!(
    interpret("Reduce[False, {x, y}, Rationals]").unwrap(),
    "False"
  );
}

#[test]
fn finite_rational_solutions_sort_numerically() {
  assert_eq!(
    interpret("Reduce[x == 2/3 || x == -1/2 || x == 1/3, x, Rationals]")
      .unwrap(),
    "x == -1/2 || x == 1/3 || x == 2/3"
  );
}
