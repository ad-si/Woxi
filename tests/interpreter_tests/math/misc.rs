use super::*;

mod minus_wrong_arity {
  use super::*;

  #[test]
  fn minus_single_arg_negates() {
    assert_eq!(interpret("Minus[5]").unwrap(), "-5");
  }

  #[test]
  fn minus_two_args_returns_unevaluated() {
    // Minus[5, 2] should print warning and return 5 − 2 (Unicode minus, matching Wolfram)
    let result = interpret("Minus[5, 2]").unwrap();
    assert_eq!(result, "5 \u{2212} 2");
  }
}

mod therefore {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("Therefore[a, b]").unwrap(), "a \u{2234} b");
  }

  #[test]
  fn three_args() {
    assert_eq!(
      interpret("Therefore[a, b, c]").unwrap(),
      "a \u{2234} b \u{2234} c"
    );
  }

  #[test]
  fn one_arg() {
    assert_eq!(interpret("Therefore[a]").unwrap(), "Therefore[a]");
  }

  #[test]
  fn zero_args() {
    assert_eq!(interpret("Therefore[]").unwrap(), "Therefore[]");
  }

  #[test]
  fn args_evaluated() {
    assert_eq!(interpret("Therefore[1+2, 3]").unwrap(), "3 \u{2234} 3");
  }
}

mod because {
  use super::*;

  #[test]
  fn two_args() {
    assert_eq!(interpret("Because[a, b]").unwrap(), "a \u{2235} b");
  }

  #[test]
  fn three_args() {
    assert_eq!(
      interpret("Because[a, b, c]").unwrap(),
      "a \u{2235} b \u{2235} c"
    );
  }

  #[test]
  fn one_arg() {
    assert_eq!(interpret("Because[a]").unwrap(), "Because[a]");
  }

  #[test]
  fn args_evaluated() {
    assert_eq!(interpret("Because[1+2, 3]").unwrap(), "3 \u{2235} 3");
  }
}

mod implicit_multiply_power_precedence {
  use super::*;

  #[test]
  fn b_y_cubed() {
    assert_eq!(
      interpret("FullForm[b y^3]").unwrap(),
      "FullForm[b*y^3]"
    );
  }

  #[test]
  fn two_x_squared_y_cubed() {
    assert_eq!(
      interpret("FullForm[2 x^2 y^3]").unwrap(),
      "FullForm[2*x^2*y^3]"
    );
  }

  #[test]
  fn coefficient_with_implicit_multiply() {
    assert_eq!(
      interpret("Coefficient[a x^2 + b y^3 + c x + d y + 5, y, 3]").unwrap(),
      "b"
    );
  }

  #[test]
  fn function_call_implicit_times() {
    // Regression: Sin[x] Sin[y] was not parsed as implicit multiplication
    assert_eq!(interpret("Sin[x] Cos[y]").unwrap(), "Cos[y]*Sin[x]");
  }

  #[test]
  fn function_call_implicit_times_three_factors() {
    assert_eq!(
      interpret("Sin[x] Cos[y] Tan[z]").unwrap(),
      "Cos[y]*Sin[x]*Tan[z]"
    );
  }

  #[test]
  fn function_call_implicit_times_with_number() {
    assert_eq!(interpret("2 Sin[x]").unwrap(), "2*Sin[x]");
  }

  #[test]
  fn function_call_implicit_times_with_implicit_arg() {
    // Sin[3y] should parse 3y as implicit multiplication inside the argument
    assert_eq!(interpret("Sin[x] Sin[3y]").unwrap(), "Sin[x]*Sin[3*y]");
  }

  #[test]
  fn function_call_implicit_times_evaluates() {
    assert_eq!(interpret("Sin[0] Sin[Pi/2]").unwrap(), "0");
  }
}

mod max_symbolic {
  use super::*;

  #[test]
  fn filters_numeric_keeps_symbolic() {
    assert_eq!(interpret("Max[5, x, -3, y, 40]").unwrap(), "Max[40, x, y]");
  }

  #[test]
  fn all_numeric() {
    assert_eq!(interpret("Max[5, 3, 8, 1]").unwrap(), "8");
  }
}

mod min_symbolic {
  use super::*;

  #[test]
  fn filters_numeric_keeps_symbolic() {
    assert_eq!(interpret("Min[5, x, -3, y, 40]").unwrap(), "Min[-3, x, y]");
  }
}

mod max_min_flatten {
  use super::*;

  #[test]
  fn max_flattens_nested_lists() {
    assert_eq!(
      interpret("Max[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]").unwrap(),
      "3.5"
    );
  }

  #[test]
  fn min_flattens_nested_lists() {
    assert_eq!(
      interpret("Min[{1,2},3,{-3,3.5,-Infinity},{{1/2}}]").unwrap(),
      "-Infinity"
    );
  }
}

mod diagonal {
  use super::*;

  #[test]
  fn main_diagonal() {
    assert_eq!(
      interpret("Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "{1, 5, 9}"
    );
  }

  #[test]
  fn superdiagonal() {
    assert_eq!(
      interpret("Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]").unwrap(),
      "{2, 6}"
    );
  }

  #[test]
  fn subdiagonal() {
    assert_eq!(
      interpret("Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, -1]").unwrap(),
      "{4, 8}"
    );
  }

  #[test]
  fn rectangular() {
    assert_eq!(
      interpret("Diagonal[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "{1, 5}"
    );
  }
}

mod pauli_matrix {
  use super::*;

  #[test]
  fn pauli_1() {
    assert_eq!(interpret("PauliMatrix[1]").unwrap(), "{{0, 1}, {1, 0}}");
  }

  #[test]
  fn pauli_2() {
    assert_eq!(interpret("PauliMatrix[2]").unwrap(), "{{0, -I}, {I, 0}}");
  }

  #[test]
  fn pauli_3() {
    assert_eq!(interpret("PauliMatrix[3]").unwrap(), "{{1, 0}, {0, -1}}");
  }

  #[test]
  fn pauli_table() {
    assert_eq!(
      interpret("Table[PauliMatrix[i], {i, 1, 3}]").unwrap(),
      "{{{0, 1}, {1, 0}}, {{0, -I}, {I, 0}}, {{1, 0}, {0, -1}}}"
    );
  }
}

mod curl {
  use super::*;

  #[test]
  fn curl_2d() {
    assert_eq!(interpret("Curl[{y, -x}, {x, y}]").unwrap(), "-2");
  }

  #[test]
  fn curl_3d() {
    assert_eq!(
      interpret("Curl[{y, -x, 2 z}, {x, y, z}]").unwrap(),
      "{0, 0, -2}"
    );
  }
}

mod log {
  use super::*;

  #[test]
  fn log_zero() {
    assert_eq!(interpret("Log[0]").unwrap(), "-Infinity");
  }

  #[test]
  fn log_two_arg_exact() {
    assert_eq!(interpret("Log[2, 8]").unwrap(), "3");
    assert_eq!(interpret("Log[2, 16]").unwrap(), "4");
    assert_eq!(interpret("Log[3, 9]").unwrap(), "2");
  }

  #[test]
  fn log_two_arg_symbolic() {
    assert_eq!(interpret("Log[2, 5]").unwrap(), "Log[5]/Log[2]");
  }
}

mod linear_recurrence {
  use super::*;

  #[test]
  fn fibonacci_via_recurrence() {
    assert_eq!(
      interpret("LinearRecurrence[{1, 1}, {1, 1}, 10]").unwrap(),
      "{1, 1, 2, 3, 5, 8, 13, 21, 34, 55}"
    );
  }
}

mod zero_divided_by_symbolic {
  use super::*;

  #[test]
  fn zero_over_symbolic() {
    assert_eq!(interpret("0/x").unwrap(), "0");
    assert_eq!(interpret("0/(2*Pi)").unwrap(), "0");
    assert_eq!(interpret("0/Sqrt[2]").unwrap(), "0");
  }

  #[test]
  fn zero_real_over_symbolic() {
    assert_eq!(interpret("0.0/x").unwrap(), "0.");
    assert_eq!(interpret("0.0/Pi").unwrap(), "0.");
  }

  #[test]
  fn zero_over_integer() {
    assert_eq!(interpret("0/5").unwrap(), "0");
  }
}

mod eigensystem {
  use super::*;

  #[test]
  fn two_by_two() {
    assert_eq!(
      interpret("Eigensystem[{{1, 2}, {3, 4}}]").unwrap(),
      "{{(5 + Sqrt[33])/2, (5 - Sqrt[33])/2}, {{(-3 + Sqrt[33])/6, 1}, {(-3 - Sqrt[33])/6, 1}}}"
    );
  }

  #[test]
  fn diagonal() {
    assert_eq!(
      interpret("Eigensystem[{{2, 0}, {0, 3}}]").unwrap(),
      "{{3, 2}, {{0, 1}, {1, 0}}}"
    );
  }

  #[test]
  fn three_by_three_diagonal() {
    assert_eq!(
      interpret("Eigensystem[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "{{3, 2, 1}, {{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}}"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(interpret("Eigensystem[m]").unwrap(), "Eigensystem[m]");
  }
}
