use super::*;

mod composition {
  use super::*;

  #[test]
  fn composition_apply() {
    assert_eq!(
      interpret("Composition[StringLength, ToString][12345]").unwrap(),
      "5"
    );
  }

  #[test]
  fn composition_symbolic() {
    assert_eq!(interpret("Composition[f, g]").unwrap(), "f @* g");
  }

  #[test]
  fn composition_variable() {
    assert_eq!(
      interpret("f = Composition[StringLength, ToString]; f[12345]").unwrap(),
      "5"
    );
  }
}

mod through {
  use super::*;

  #[test]
  fn through_list_head() {
    assert_eq!(interpret("Through[{f, g}[x]]").unwrap(), "{f[x], g[x]}");
  }

  #[test]
  fn through_list_head_multiple_args() {
    assert_eq!(
      interpret("Through[{f, g}[x, y]]").unwrap(),
      "{f[x, y], g[x, y]}"
    );
  }

  #[test]
  fn through_function_head() {
    assert_eq!(interpret("Through[f[g][x]]").unwrap(), "f[g[x]]");
  }

  #[test]
  fn through_plus_head() {
    assert_eq!(interpret("Through[Plus[f, g][x]]").unwrap(), "f[x] + g[x]");
  }

  #[test]
  fn through_times_head() {
    assert_eq!(interpret("Through[Times[f, g][x]]").unwrap(), "f[x]*g[x]");
  }

  #[test]
  fn through_simple_call_unevaluated() {
    assert_eq!(interpret("Through[f[x]]").unwrap(), "Through[f[x]]");
  }

  #[test]
  fn through_non_call_unevaluated() {
    assert_eq!(interpret("Through[x]").unwrap(), "Through[x]");
  }

  #[test]
  fn through_with_matching_head_filter() {
    assert_eq!(
      interpret("Through[{f, g}[x], List]").unwrap(),
      "{f[x], g[x]}"
    );
  }

  #[test]
  fn through_with_non_matching_head_filter() {
    assert_eq!(interpret("Through[{f, g}[x], Plus]").unwrap(), "{f, g}[x]");
  }
}

mod curried_call_preservation {
  use super::*;

  #[test]
  fn symbolic_curried_call_stays() {
    assert_eq!(interpret("f[g][x]").unwrap(), "f[g][x]");
  }

  #[test]
  fn list_head_stays() {
    assert_eq!(interpret("{f, g}[x]").unwrap(), "{f, g}[x]");
  }
}

mod anonymous_function_precedence {
  use super::*;

  #[test]
  fn ampersand_captures_full_expression() {
    assert_eq!(interpret("#1+#2&[4, 5]").unwrap(), "9");
  }

  #[test]
  fn ampersand_with_operator_after() {
    assert_eq!(interpret("#^2& @ 3").unwrap(), "9");
  }

  #[test]
  fn slot2_standalone() {
    assert_eq!(interpret("#2&[4, 5]").unwrap(), "5");
  }
}

mod apply_head_replacement {
  use super::*;

  #[test]
  fn apply_replaces_plus_head() {
    assert_eq!(interpret("f @@ (a + b + c)").unwrap(), "f[a, b, c]");
  }

  #[test]
  fn apply_operator_form() {
    assert_eq!(interpret("Apply[f][a + b + c]").unwrap(), "f[a, b, c]");
  }
}

mod fixed_point {
  use super::*;

  #[test]
  fn fixed_point_cos() {
    assert_eq!(
      interpret("FixedPoint[Cos, 1.0]").unwrap(),
      "0.7390851332151607"
    );
  }

  #[test]
  fn fixed_point_sqrt2_newton() {
    // Newton's method for sqrt(2): f(x) = (x + 2/x) / 2
    assert_eq!(
      interpret("FixedPoint[N[(# + 2/#)/2] &, 1.]").unwrap(),
      "1.414213562373095"
    );
  }

  #[test]
  fn fixed_point_identity() {
    // FixedPoint on a value that's already a fixed point
    assert_eq!(interpret("FixedPoint[# &, 5]").unwrap(), "5");
  }

  #[test]
  fn fixed_point_with_max_iterations() {
    // FixedPoint with explicit max iterations
    assert_eq!(
      interpret("FixedPoint[Cos, 1.0, 100]").unwrap(),
      "0.7390851332151607"
    );
  }

  #[test]
  fn fixed_point_floor_halving() {
    // Floor[#/2]& converges to 0
    assert_eq!(interpret("FixedPoint[Floor[#/2] &, 100]").unwrap(), "0");
  }

  #[test]
  fn fixed_point_list_collatz() {
    // Regression test: SetDelayed with literal arg (collatz[1] := 1) must
    // take priority over general pattern (collatz[x_] := 3 x + 1)
    assert_eq!(
      interpret(
        "collatz[1] := 1; collatz[x_ ? EvenQ] := x / 2; collatz[x_] := 3 x + 1; FixedPointList[collatz, 14]"
      )
      .unwrap(),
      "{14, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 1}"
    );
  }
}

mod subdivide {
  use super::*;

  #[test]
  fn unit_interval() {
    assert_eq!(interpret("Subdivide[4]").unwrap(), "{0, 1/4, 1/2, 3/4, 1}");
  }

  #[test]
  fn two_parts() {
    assert_eq!(interpret("Subdivide[2]").unwrap(), "{0, 1/2, 1}");
  }

  #[test]
  fn one_part() {
    assert_eq!(interpret("Subdivide[1]").unwrap(), "{0, 1}");
  }

  #[test]
  fn custom_range() {
    assert_eq!(
      interpret("Subdivide[0, 10, 5]").unwrap(),
      "{0, 2, 4, 6, 8, 10}"
    );
  }

  #[test]
  fn two_arg_form() {
    assert_eq!(
      interpret("Subdivide[10, 5]").unwrap(),
      "{0, 2, 4, 6, 8, 10}"
    );
  }
}

mod composition_edge_cases {
  use super::*;

  #[test]
  fn empty_is_identity() {
    assert_eq!(interpret("Composition[]").unwrap(), "Identity");
  }

  #[test]
  fn single_is_function() {
    assert_eq!(interpret("Composition[f]").unwrap(), "f");
  }
}

mod listable {
  use super::*;

  #[test]
  fn fibonacci_list() {
    assert_eq!(
      interpret("Fibonacci[{1, 2, 3, 4, 5, 6}]").unwrap(),
      "{1, 1, 2, 3, 5, 8}"
    );
  }

  #[test]
  fn sin_list() {
    assert_eq!(interpret("Sin[{0, Pi/2, Pi}]").unwrap(), "{0, 1, 0}");
  }

  #[test]
  fn power_list_scalar() {
    assert_eq!(interpret("Power[{2, 3, 4}, 2]").unwrap(), "{4, 9, 16}");
  }

  #[test]
  fn power_scalar_list() {
    assert_eq!(interpret("Power[2, {1, 2, 3}]").unwrap(), "{2, 4, 8}");
  }

  #[test]
  fn power_exponent_one_simplifies() {
    assert_eq!(interpret("x^1").unwrap(), "x");
    assert_eq!(interpret("Power[y, 1]").unwrap(), "y");
    assert_eq!(interpret("(a + b)^1").unwrap(), "a + b");
  }

  #[test]
  fn mod_basic() {
    assert_eq!(interpret("Mod[10, 3]").unwrap(), "1");
    assert_eq!(interpret("Mod[7, 4]").unwrap(), "3");
    assert_eq!(interpret("Mod[15, 5]").unwrap(), "0");
    assert_eq!(interpret("Mod[0, 5]").unwrap(), "0");
  }

  #[test]
  fn mod_negative_args() {
    assert_eq!(interpret("Mod[-10, 3]").unwrap(), "2");
    assert_eq!(interpret("Mod[-5, 3]").unwrap(), "1");
    assert_eq!(interpret("Mod[10, -3]").unwrap(), "-2");
    assert_eq!(interpret("Mod[-10, -3]").unwrap(), "-1");
    assert_eq!(interpret("Mod[-1, 3]").unwrap(), "2");
  }

  #[test]
  fn mod_rational() {
    assert_eq!(interpret("Mod[5/2, 1]").unwrap(), "1/2");
    assert_eq!(interpret("Mod[7/3, 2/3]").unwrap(), "1/3");
  }

  #[test]
  fn mod_float() {
    assert_eq!(interpret("Mod[5.5, 2]").unwrap(), "1.5");
    assert_eq!(interpret("Mod[7.5, 2]").unwrap(), "1.5");
  }

  #[test]
  fn mod_division_by_zero() {
    assert_eq!(interpret("Mod[10, 0]").unwrap(), "Indeterminate");
    assert_eq!(interpret("Mod[0, 0]").unwrap(), "Indeterminate");
  }

  #[test]
  fn mod_three_args() {
    assert_eq!(interpret("Mod[10, 3, 1]").unwrap(), "1");
    assert_eq!(interpret("Mod[10, 3, -1]").unwrap(), "1");
    assert_eq!(interpret("Mod[10, 3, 2]").unwrap(), "4");
    assert_eq!(interpret("Mod[-5, 3, 1]").unwrap(), "1");
    assert_eq!(interpret("Mod[10, 3, 0]").unwrap(), "1");
  }

  #[test]
  fn mod_three_args_rational() {
    assert_eq!(interpret("Mod[5/2, 3/2, 0]").unwrap(), "1");
    assert_eq!(interpret("Mod[5/2, 3/2, 1]").unwrap(), "1");
  }

  #[test]
  fn mod_symbolic() {
    assert_eq!(interpret("Mod[x, 3]").unwrap(), "Mod[x, 3]");
  }

  #[test]
  fn mod_list_scalar() {
    assert_eq!(interpret("Mod[{10, 20, 30}, 7]").unwrap(), "{3, 6, 2}");
  }

  #[test]
  fn plus_two_lists() {
    assert_eq!(
      interpret("Plus[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "{5, 7, 9}"
    );
  }

  #[test]
  fn plus_list_scalar() {
    assert_eq!(interpret("{1, 2, 3} + 10").unwrap(), "{11, 12, 13}");
  }

  #[test]
  fn times_two_lists() {
    assert_eq!(interpret("{1, 2, 3} * {4, 5, 6}").unwrap(), "{4, 10, 18}");
  }

  #[test]
  fn plus_three_lists() {
    assert_eq!(
      interpret("Plus[{1, 2}, {3, 4}, {5, 6}]").unwrap(),
      "{9, 12}"
    );
  }

  #[test]
  fn mismatched_lengths_no_thread() {
    // Mismatched list lengths should not thread — function returns unevaluated
    assert_eq!(interpret("Sin[{1, 2}] + Sin[{3, 4, 5}]").is_err(), true);
  }

  #[test]
  fn nested_listable() {
    assert_eq!(interpret("Abs[{-1, 2, -3, 4}]").unwrap(), "{1, 2, 3, 4}");
  }

  #[test]
  fn user_defined_listable() {
    assert_eq!(
      interpret("SetAttributes[f, Listable]; f[{1, 2, 3}]").unwrap(),
      "{f[1], f[2], f[3]}"
    );
  }

  #[test]
  fn evenq_list() {
    assert_eq!(
      interpret("EvenQ[{1, 2, 3, 4}]").unwrap(),
      "{False, True, False, True}"
    );
  }

  #[test]
  fn floor_list() {
    assert_eq!(interpret("Floor[{1.2, 2.7, 3.5}]").unwrap(), "{1, 2, 3}");
  }
}
