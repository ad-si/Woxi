use super::assert_reduces;

#[test]
fn coefficients_larger_than_i128_remain_exact() {
  assert_reduces(
    "Reduce[100000000000000000000000000000000000000000*x == \
     200000000000000000000000000000000000000000, x, Rationals]",
    "x == 2",
  );
}

#[test]
fn negation_flips_strictness_without_approximation() {
  assert_reduces("Reduce[!(3*x < 1), x, Reals]", "x >= 1/3");
}

// An integer equation whose coefficient gcd does not divide its constant has
// no solution, however it is combined with other conditions. Before this was
// checked, the engine restated the equation as a degenerate two-sided bound
// containing the non-integer 5/2.
#[test]
fn integer_equations_without_solutions_are_false_in_any_context() {
  assert_reduces(
    "Reduce[2 x + 4 y == 5 && x >= 0, {x, y}, Integers]",
    "False",
  );
  assert_reduces(
    "Reduce[2 x + 4 y == 5 || x > 100, {x, y}, Integers]",
    "Element[x | y, Integers] && x >= 101",
  );
  assert_reduces("Reduce[2 x == 4 && Mod[x, 3] == 1, x, Integers]", "False");
}

// Equations between the requested variables are solved, over both domains,
// instead of being printed as pairs of half-spaces.
#[test]
fn determined_linear_systems_are_solved() {
  assert_reduces(
    "Reduce[x + y == 3 && x - y == 1, {x, y}, Integers]",
    "x == 2 && y == 1",
  );
  assert_reduces(
    "Reduce[x + y == 3 && x - y == 1, {x, y}, Reals]",
    "x == 2 && y == 1",
  );
  assert_reduces("Reduce[x <= 2 && x >= 2, x, Reals]", "x == 2");
  // Later variables are solved in terms of earlier ones.
  assert_reduces(
    "Reduce[x + y == 3 && x >= 0 && y >= 0, {x, y}, Reals]",
    "Inequality[0, LessEqual, x, LessEqual, 3] && y == 3 - x",
  );
  // An integer equation is solved for a variable with a unit coefficient.
  assert_reduces(
    "Reduce[2 x == 4 y && x >= 0, {x, y}, Integers]",
    "Element[x | y, Integers] && x == 2*y && y >= 0",
  );
}

// A finite single-variable result is enumerated even when one branch is an
// equation rather than a bounded interval.
#[test]
fn finite_integer_branches_with_equations_are_enumerated() {
  assert_reduces(
    "Reduce[x == 2 || 5 <= x <= 7, x, Integers]",
    "x == 2 || x == 5 || x == 6 || x == 7",
  );
  assert_reduces(
    "Reduce[x > 2 || x == 0, x, Integers]",
    "Element[x, Integers] && (x == 0 || x >= 3)",
  );
}

// Output follows Wolfram's ordering conventions: constants lead a sum,
// regions are listed in increasing order, relations are solved for the last
// requested variable they mention, and two one-sided bounds on one variable
// merge into a single chain even inside a longer conjunction.
#[test]
fn emitted_forms_follow_wolfram_ordering() {
  assert_reduces("Reduce[x == 2 || x == -2, x, Reals]", "x == -2 || x == 2");
  assert_reduces("Reduce[x + y < 1, {x, y}, Reals]", "y < 1 - x");
  assert_reduces("Reduce[x > y, {x, y}, Reals]", "y < x");
  assert_reduces("Reduce[x > 1 && x < y, {x, y}, Reals]", "x > 1 && y > x");
  assert_reduces(
    "Reduce[x > 1 && x < 5 && y > 2, {x, y}, Reals]",
    "Inequality[1, Less, x, Less, 5] && y > 2",
  );
  assert_reduces(
    "Reduce[Mod[x + 2 y, 4] == 3, {x, y}, Integers]",
    "Element[x | y, Integers] && Mod[x + 2*y, 4] == 3",
  );
}

// Cooper elimination instantiates the body once per residue of the lcm of
// all moduli. A request that would need billions of instances is declined
// and left unevaluated instead of running forever.
#[test]
fn oversized_congruence_periods_are_declined_promptly() {
  assert_reduces(
    "Resolve[Exists[x, Mod[x, 1000003] == 1 && Mod[x, 1000033] == 2], Integers]",
    "Resolve[Exists[x, Mod[x, 1000003] == 1 && Mod[x, 1000033] == 2], Integers]",
  );
}
