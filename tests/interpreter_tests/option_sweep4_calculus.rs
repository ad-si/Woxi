// Argument forms of ODE, recurrence and calculus functions found missing by
// the reference-page parameter sweep (round 4).

use super::*;

mod difference_delta_symbolic_order {
  use super::*;

  #[test]
  fn exponential_with_symbolic_order() {
    assert_eq!(
      interpret("DifferenceDelta[a^i, {i, n}]").unwrap(),
      "(-1 + a)^n*a^i"
    );
  }

  #[test]
  fn symbolic_order_with_step_and_linear_exponent() {
    // Each difference multiplies b^(c i) by b^(c h) - 1.
    assert_eq!(
      interpret("DifferenceDelta[2^(3 i), {i, n, h}]").unwrap(),
      "2^(3*i)*(-1 + 2^(3*h))^n"
    );
  }

  #[test]
  fn symbolic_order_agrees_with_integer_orders() {
    assert_eq!(
      interpret(
        "Table[Simplify[(DifferenceDelta[a^i, {i, n}] /. n -> k) - \
         DifferenceDelta[a^i, {i, k}]], {k, 1, 3}]"
      )
      .unwrap(),
      "{0, 0, 0}"
    );
  }

  #[test]
  fn non_exponential_symbolic_order_stays_unevaluated() {
    assert_eq!(
      interpret("DifferenceDelta[i^2, {i, n}]").unwrap(),
      "DifferenceDelta[i^2, {i, n}]"
    );
    assert_eq!(
      interpret("DifferenceDelta[a^i^2, {i, n}]").unwrap(),
      "DifferenceDelta[a^i^2, {i, n}]"
    );
  }
}

mod times_order_of_symbolic_powers {
  use super::*;

  // A power with a symbol exponent orders against a sum in its base like
  // the base itself: `(-1 + x)*x` and so `(-1 + x)*x^n`, but `x^n*(1 + x)`.
  #[test]
  fn negative_constant_sum_leads() {
    assert_eq!(interpret("x^n (-1 + x)").unwrap(), "(-1 + x)*x^n");
    assert_eq!(interpret("a^i (-1 + a)^n").unwrap(), "(-1 + a)^n*a^i");
  }

  #[test]
  fn positive_constant_sum_follows() {
    assert_eq!(interpret("(1 + x) x^n").unwrap(), "x^n*(1 + x)");
    assert_eq!(interpret("(1 + x^b) x^a").unwrap(), "x^a*(1 + x^b)");
  }
}

mod dsolve_systems {
  use super::*;

  #[test]
  fn decoupled_system_solves_each_equation() {
    assert_eq!(
      interpret("DSolve[{y'[x] == x^2 y[x], z'[x] == 5 z[x]}, {y, z}, x]")
        .unwrap(),
      "{{y -> Function[{x}, E^(x^3/3)*C[1]], z -> Function[{x}, E^(5*x)*C[2]]}}"
    );
    assert_eq!(
      interpret(
        "DSolveValue[{y'[x] == x^2 y[x], z'[x] == 5 z[x]}, {y[x], z[x]}, x]"
      )
      .unwrap(),
      "{E^(x^3/3)*C[1], E^(5*x)*C[2]}"
    );
  }

  #[test]
  fn decoupled_system_numbers_constants_in_order() {
    assert_eq!(
      interpret("DSolve[{y''[x] == -y[x], z'[x] == z[x]}, {y[x], z[x]}, x]")
        .unwrap(),
      "{{y[x] -> C[1]*Cos[x] + C[2]*Sin[x], z[x] -> E^x*C[3]}}"
    );
  }

  // A system the solvers cannot handle stays unevaluated instead of
  // aborting with "cannot convert … to a numeric value".
  #[test]
  fn unsupported_system_stays_unevaluated() {
    assert_eq!(
      interpret("DSolve[{y'[x] == a y[x] + z[x], z'[x] == y[x]}, {y, z}, x]")
        .unwrap(),
      "DSolve[{Derivative[1][y][x] == a*y[x] + z[x], \
       Derivative[1][z][x] == y[x]}, {y, z}, x]"
    );
    assert_eq!(
      interpret(
        "DSolveValue[{y'[x] == x y[x] + z[x], z'[x] == y[x]}, {y[x], z[x]}, x]"
      )
      .unwrap(),
      "DSolveValue[{Derivative[1][y][x] == x*y[x] + z[x], \
       Derivative[1][z][x] == y[x]}, {y[x], z[x]}, x]"
    );
  }
}

mod recurrence_table_algebraic_system {
  use super::*;

  // The coupled step and the algebraic constraint are solved together at
  // each index.
  #[test]
  fn coupled_step_with_constraint() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{y[n + 1] + z[n + 1] == y[n] - 3 z[n] + 7, \
         y[n] + 2 z[n] == 1, y[0] == 1/2, z[0] == 1/4}, {y, z}, {n, 0, 4}]"
      )
      .unwrap(),
      "{{1/2, 1/4}, {25/2, -23/4}, {145/2, -143/4}, {745/2, -743/4}, \
       {3745/2, -3743/4}}"
    );
  }

  #[test]
  fn machine_initial_values() {
    assert_eq!(
      interpret(
        "RecurrenceTable[{y[n + 1] + z[n + 1] == y[n] - 3 z[n] + 7, \
         y[n] + 2 z[n] == 1, y[0] == 0.5, z[0] == 0.25}, {y, z}, {n, 0, 2}]"
      )
      .unwrap(),
      "{{0.5, 0.25}, {12.5, -5.75}, {72.5, -35.75}}"
    );
  }
}

mod dsolve_differential_algebraic {
  use super::*;

  // The algebraic equation eliminates z; the reduced ODE is solved and z
  // recovered from it (DSolve reference page).
  #[test]
  fn algebraic_constraint_is_eliminated() {
    assert_eq!(
      interpret(
        "DSolve[{y'[x] - 3 z[x] == Sin[x], y[x] + z[x] == 1/5, \
         y[Pi/2] == 1/2}, {y, z}, x]"
      )
      .unwrap(),
      "{{y -> Function[{x}, (2 - Cos[x] + 3*Sin[x])/10], \
       z -> Function[{x}, (Cos[x] - 3*Sin[x])/10]}}"
    );
    assert_eq!(
      interpret(
        "DSolveValue[{y'[x] - 3 z[x] == Sin[x], y[x] + z[x] == 1/5, \
         y[Pi/2] == 1/2}, {y, z}, x]"
      )
      .unwrap(),
      "{Function[{x}, (2 - Cos[x] + 3*Sin[x])/10], \
       Function[{x}, (Cos[x] - 3*Sin[x])/10]}"
    );
  }

  #[test]
  fn eliminated_solution_satisfies_the_system() {
    assert_eq!(
      interpret(
        "sol = First[DSolve[{y'[x] == z[x], y[x] - 2 z[x] == 0}, \
         {y[x], z[x]}, x]]; Simplify[{D[y[x] /. sol, x] - (z[x] /. sol), \
         (y[x] /. sol) - 2 (z[x] /. sol)}]"
      )
      .unwrap(),
      "{0, 0}"
    );
  }
}

mod nested_limits {
  use super::*;

  // Limit[f, {x1 -> a1, x2 -> a2}] is lim_{x1 -> a1} lim_{x2 -> a2} f: the
  // last rule is taken first.
  #[test]
  fn last_rule_is_taken_first() {
    assert_eq!(
      interpret("Limit[RealAbs[x] Tan[y]^2, {y -> Pi/2, x -> 0}]").unwrap(),
      "0"
    );
    assert_eq!(
      interpret("Limit[ArcTan[y/x], {y -> Infinity, x -> Infinity}]")
        .unwrap(),
      "0"
    );
  }

  #[test]
  fn continuous_function() {
    assert_eq!(interpret("Limit[x y, {x -> 2, y -> 3}]").unwrap(), "6");
  }
}

mod limits_through_branch_cuts {
  use super::*;

  // From the left of 0, Log[x] = Log[-x] + I Pi: Log[x]/x heads to
  // +Infinity there and to -Infinity from the right.
  #[test]
  fn log_over_x_two_sided_is_indeterminate() {
    assert_eq!(interpret("Limit[Log[x]/x, x -> 0]").unwrap(), "Indeterminate");
    assert_eq!(
      interpret("Limit[Log[x]^2/x, x -> 0]").unwrap(),
      "Indeterminate"
    );
  }

  // The real part dominates the left-hand side, so both sides agree.
  #[test]
  fn both_sides_agree() {
    assert_eq!(interpret("Limit[Log[x]/x^2, x -> 0]").unwrap(), "-Infinity");
    assert_eq!(interpret("Limit[x Log[x], x -> 0]").unwrap(), "0");
  }

  #[test]
  fn at_negative_infinity() {
    assert_eq!(
      interpret("Limit[Log[x]/Sqrt[x], x -> -Infinity]").unwrap(),
      "0"
    );
    assert_eq!(interpret("Limit[Log[x]/x, x -> -Infinity]").unwrap(), "0");
    assert_eq!(
      interpret("Limit[x^(1/3)/x, x -> -Infinity]").unwrap(),
      "0"
    );
  }

  #[test]
  fn complex_constant_factor() {
    assert_eq!(
      interpret("Limit[-I Pi/t, t -> 0, Direction -> \"FromAbove\"]")
        .unwrap(),
      "DirectedInfinity[-I]"
    );
    assert_eq!(
      interpret("Limit[(I Pi + Log[t])/Sqrt[t], t -> Infinity]").unwrap(),
      "0"
    );
  }
}

mod curl_of_rank_n_minus_1 {
  use super::*;

  // A rank-2 field in 3D has a scalar curl,
  // (1/2) Σ ε[j, i, k] D[f[[i, k]], x_j] (Curl reference page).
  #[test]
  fn rank_two_in_three_dimensions() {
    assert_eq!(
      interpret(
        "Curl[{{x y, x y^2, x y^3}, {x^2 y, x^2 y^2, x^2 y^3}, \
         {x^3 y, x^3 y^2, x^3 y^3}}, {x, y, z}]"
      )
      .unwrap(),
      "(x^3 - 3*x*y^2 - 3*x^2*y^2 + 2*x*y^3)/2"
    );
  }

  #[test]
  fn antisymmetric_part_only() {
    assert_eq!(
      interpret("Curl[{{0, z, -y}, {-z, 0, x}, {y, -x, 0}}, {x, y, z}]")
        .unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Curl[{{x, y, z}, {y, z, x}, {z, x, y}}, {x, y, z}]").unwrap(),
      "0"
    );
  }
}

mod function_domain_several_variables {
  use super::*;

  #[test]
  fn nonzero_denominator() {
    assert_eq!(
      interpret("FunctionDomain[(x + y)/(x^2 - y^2), {x, y}]").unwrap(),
      "x^2 - y^2 != 0"
    );
  }

  #[test]
  fn no_restriction() {
    assert_eq!(interpret("FunctionDomain[x + y, {x, y}]").unwrap(), "True");
  }

  #[test]
  fn other_restrictions_stay_unevaluated() {
    assert_eq!(
      interpret("FunctionDomain[Sqrt[x - y], {x, y}]").unwrap(),
      "FunctionDomain[Sqrt[x - y], {x, y}]"
    );
  }
}

mod nintegrate_regions_and_complex {
  use super::*;

  fn assert_close(expr: &str, expected: &str, tol: f64) {
    assert_eq!(
      interpret(&format!("Abs[{expr} - ({expected})] < {tol}")).unwrap(),
      "True",
      "{expr} gave {}",
      interpret(expr).unwrap()
    );
  }

  // The `&&` condition is compiled, and each comparison's crossings are
  // located on their own, so the sliver where the line meets the ellipse
  // is not lost (3 Pi/16; NIntegrate reference page: 0.589049).
  #[test]
  fn boole_of_a_conjunction() {
    assert_close(
      "NIntegrate[Boole[1/4 <= x^2 + (2 y)^2 <= 1 && y > x], {x, -1, 1}, \
       {y, -1, 1}]",
      "3 Pi/16",
      1e-10,
    );
    assert_close(
      "NIntegrate[Boole[x^2 + y^2 <= 1 && y > x], {x, -1, 1}, {y, -1, 1}]",
      "Pi/2",
      1e-10,
    );
  }

  #[test]
  fn exclusions_split_the_interval() {
    assert_close(
      "NIntegrate[1/Sqrt[Abs[Cos[y]]], {y, 0, Pi}, Exclusions -> Cos[y] == 0]",
      "Sqrt[Pi] Gamma[1/4]/Gamma[3/4]",
      1e-6,
    );
  }

  // NIntegrate reference page: 21.5381 - 19.5357 I.
  #[test]
  fn complex_integrand_with_exclusions() {
    assert_close(
      "NIntegrate[1/Sqrt[Cos[x y]], {x, 0, 5}, {y, 0, 5}, \
       Exclusions -> Cos[x y] == 0]",
      "21.5381 - 19.5357 I",
      1e-4,
    );
  }

  #[test]
  fn complex_integrand() {
    assert_close(
      "NIntegrate[Sqrt[x - y], {x, 0, 1}, {y, 0, 1}]",
      "4/15 + 4 I/15",
      1e-8,
    );
  }
}

mod nsum_complex {
  use super::*;

  // -Zeta'[2 + 2 I]; NSum reference page: -0.182175 - 0.136618 I.
  #[test]
  fn complex_summand() {
    assert_eq!(
      interpret(
        "Abs[NSum[Log[x]/x^(2 + 2 I), {x, 1, Infinity}] - \
         (-0.1821754027 - 0.1366183169 I)] < 10^-8"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn agrees_with_zeta() {
    assert_eq!(
      interpret(
        "Abs[NSum[1/x^(1.2 + I), {x, 1, Infinity}] - Zeta[1.2 + I]] < 10^-9"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("Abs[NSum[I/n^2, {n, 1, 10}] - N[I HarmonicNumber[10, 2]]] < 10^-12")
        .unwrap(),
      "True"
    );
  }
}
