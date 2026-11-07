use woxi::interpret;

#[test]
fn canonical_surface_examples() {
  assert_eq!(
    interpret("Reduce[x^2 == 4, x]").unwrap(),
    "x == -2 || x == 2"
  );
  assert_eq!(
    interpret("Reduce[x > 2, x, Integers]").unwrap(),
    "Element[x, Integers] && x >= 3"
  );
  assert_eq!(
    interpret("Reduce[-1 < x < 1, x, Reals]").unwrap(),
    "Inequality[-1, Less, x, Less, 1]"
  );
}

#[test]
fn linear_surface_examples() {
  assert_eq!(
    interpret("Reduce[2*x + 1 == 0, x, Reals]").unwrap(),
    "x == -1/2"
  );
  assert_eq!(
    interpret("Reduce[1/3 < x && x <= 5/7, x, Rationals]").unwrap(),
    "Element[x, Rationals] && Inequality[1/3, Less, x, LessEqual, 5/7]"
  );
  assert_eq!(
    interpret("Reduce[2 < x < 5, x, Integers]").unwrap(),
    "x == 3 || x == 4"
  );
  assert_eq!(
    interpret("Reduce[!(3*x < 1), x, Reals]").unwrap(),
    "x >= 1/3"
  );
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
