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
    "Element[C[1], Integers] && x == 1 + 2*C[1]",
  );
}

/// A residue class is reported as a parametrization, never as a congruence —
/// wolframscript's shape, verified against it on 2026-09-10. A bound on the
/// variable becomes a bound on the parameter, a denied congruence splits into
/// the classes it leaves, and every constrained target gets its own
/// parameter.
#[test]
fn residue_classes_are_parametrized() {
  assert_reduces(
    "Reduce[Mod[x, 6] == 4 && x > 10, x, Integers]",
    "Element[C[1], Integers] && C[1] >= 2 && x == 4 + 6*C[1]",
  );
  assert_reduces(
    "Reduce[Mod[x, 2] == 1 && x < 0, x, Integers]",
    "Element[C[1], Integers] && C[1] <= -1 && x == 1 + 2*C[1]",
  );
  assert_reduces(
    "Reduce[Mod[x, 3] != 1, x, Integers]",
    "Element[C[1], Integers] && (x == 3*C[1] || x == 2 + 3*C[1])",
  );
  assert_reduces(
    "Reduce[Mod[x, 3] != 1 && x > 0, x, Integers]",
    "(Element[C[1], Integers] && C[1] >= 1 && x == 3*C[1]) \
     || (Element[C[1], Integers] && C[1] >= 0 && x == 2 + 3*C[1])",
  );
  assert_reduces(
    "Reduce[Mod[x, 2] == 1 && Mod[x, 3] == 2, x, Integers]",
    "Element[C[1], Integers] && x == 5 + 6*C[1]",
  );
  assert_reduces(
    "Reduce[Mod[x, 2] == 1 && y > 0, {x, y}, Integers]",
    "Element[y | C[1], Integers] && x == 1 + 2*C[1] && y >= 1",
  );
  assert_reduces(
    "Reduce[Mod[x, 2] == 1 && y > 0 && Mod[y, 3] == 2, {x, y}, Integers]",
    "Element[C[1] | C[2], Integers] && C[2] >= 0 && x == 1 + 2*C[1] \
     && y == 2 + 3*C[2]",
  );
}

/// A branch with no congruence keeps its bounded shape, and one that pins the
/// variable is reported by its value — including next to a parametrized
/// branch, where wolframscript writes `x == 8`, not `8 <= x <= 8`.
#[test]
fn unparametrized_branches_keep_their_shape() {
  assert_reduces(
    "Reduce[x > 0, x, Integers]",
    "Element[x, Integers] && x >= 1",
  );
  assert_reduces(
    "Reduce[Mod[x, 2] == 0 || x > 5, x, Integers]",
    "(Element[x, Integers] && x >= 6) \
     || (Element[C[1], Integers] && x == 2*C[1])",
  );
  assert_reduces(
    "Reduce[Exists[y, x == 2*y && y > 3], x, Integers]",
    "x == 8 || (Element[C[1], Integers] && C[1] >= 4 && x == 2*C[1])",
  );
  // An equation pinning the variable outranks its congruence.
  assert_reduces("Reduce[Mod[x, 2] == 1 && x == 5, x, Integers]", "x == 5");
  assert_reduces(
    "Reduce[Mod[x, 2] == 1 && x > 0 && x < 9, x, Integers]",
    "x == 1 || x == 3 || x == 5 || x == 7",
  );
}

#[test]
fn inconsistent_congruences() {
  assert_reduces(
    "Resolve[Exists[x, Mod[x, 4] == 1 && Mod[x, 6] == 2], Integers]",
    "False",
  );
}
