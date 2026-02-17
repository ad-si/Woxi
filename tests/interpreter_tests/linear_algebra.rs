use super::*;

mod dot {
  use super::*;

  #[test]
  fn vector_dot_product() {
    assert_eq!(interpret("Dot[{1, 2, 3}, {4, 5, 6}]").unwrap(), "32");
  }

  #[test]
  fn matrix_vector() {
    assert_eq!(
      interpret("Dot[{{1, 0}, {0, 1}}, {5, 6}]").unwrap(),
      "{5, 6}"
    );
  }

  #[test]
  fn matrix_matrix() {
    assert_eq!(
      interpret("Dot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}]").unwrap(),
      "{{19, 22}, {43, 50}}"
    );
  }

  #[test]
  fn dot_simple() {
    assert_eq!(interpret("Dot[{1, 2}, {3, 4}]").unwrap(), "11");
  }

  #[test]
  fn infix_dot_vector() {
    // Infix . operator should evaluate the same as Dot[]
    assert_eq!(interpret("{1, 2, 3} . {4, 5, 6}").unwrap(), "32");
  }

  #[test]
  fn infix_dot_matrix_vector() {
    assert_eq!(interpret("{{1, 2}, {3, 4}} . {5, 6}").unwrap(), "{17, 39}");
  }

  #[test]
  fn infix_dot_matrix_matrix() {
    assert_eq!(
      interpret("{{1, 2}, {3, 4}} . {{5, 6}, {7, 8}}").unwrap(),
      "{{19, 22}, {43, 50}}"
    );
  }

  #[test]
  fn infix_dot_symbolic() {
    // Symbolic dot returns unevaluated in infix form
    assert_eq!(interpret("a . b").unwrap(), "a . b");
  }

  #[test]
  fn dot_display_infix() {
    // Dot[a, b] should display as infix a . b
    assert_eq!(interpret("Dot[a, b]").unwrap(), "a . b");
  }

  #[test]
  fn identity_matrix_dot_symbolic_vector() {
    assert_eq!(
      interpret("Dot[IdentityMatrix[2], {x, y}]").unwrap(),
      "{x, y}"
    );
    assert_eq!(
      interpret("Dot[IdentityMatrix[3], {a, b, c}]").unwrap(),
      "{a, b, c}"
    );
  }

  #[test]
  fn matrix_dot_symbolic_vector() {
    assert_eq!(
      interpret("Dot[{{2, 0}, {0, 3}}, {x, y}]").unwrap(),
      "{2*x, 3*y}"
    );
  }
}

mod det {
  use super::*;

  #[test]
  fn det_2x2() {
    assert_eq!(interpret("Det[{{1, 2}, {3, 4}}]").unwrap(), "-2");
  }

  #[test]
  fn det_3x3() {
    assert_eq!(
      interpret("Det[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "6"
    );
  }

  #[test]
  fn det_identity() {
    assert_eq!(interpret("Det[{{1, 0}, {0, 1}}]").unwrap(), "1");
  }
}

mod inverse {
  use super::*;

  #[test]
  fn inverse_2x2() {
    assert_eq!(
      interpret("Inverse[{{1, 2}, {3, 4}}]").unwrap(),
      "{{-2, 1}, {3/2, -1/2}}"
    );
  }

  #[test]
  fn inverse_identity() {
    assert_eq!(
      interpret("Inverse[{{1, 0}, {0, 1}}]").unwrap(),
      "{{1, 0}, {0, 1}}"
    );
  }

  #[test]
  fn inverse_symbolic_2x2() {
    assert_eq!(
      interpret("Inverse[{{a, b}, {c, d}}]").unwrap(),
      "{{d/(-(b*c) + a*d), -(b/(-(b*c) + a*d))}, {-(c/(-(b*c) + a*d)), a/(-(b*c) + a*d)}}"
    );
  }

  #[test]
  fn inverse_times_identity() {
    // A . Inverse[A] should give identity matrix for numeric matrices
    assert_eq!(
      interpret("{{1, 2}, {3, 4}} . Inverse[{{1, 2}, {3, 4}}]").unwrap(),
      "{{1, 0}, {0, 1}}"
    );
  }
}

mod tr {
  use super::*;

  #[test]
  fn trace_2x2() {
    assert_eq!(interpret("Tr[{{1, 2}, {3, 4}}]").unwrap(), "5");
  }

  #[test]
  fn trace_3x3() {
    assert_eq!(
      interpret("Tr[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "6"
    );
  }

  #[test]
  fn trace_vector() {
    assert_eq!(interpret("Tr[{1, 2, 3}]").unwrap(), "6");
  }
}

mod identity_matrix {
  use super::*;

  #[test]
  fn identity_3() {
    assert_eq!(
      interpret("IdentityMatrix[3]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn identity_1() {
    assert_eq!(interpret("IdentityMatrix[1]").unwrap(), "{{1}}");
  }
}

mod diagonal_matrix {
  use super::*;

  #[test]
  fn diagonal_basic() {
    assert_eq!(
      interpret("DiagonalMatrix[{1, 2, 3}]").unwrap(),
      "{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}"
    );
  }
}

mod cross {
  use super::*;

  #[test]
  fn cross_basic() {
    assert_eq!(
      interpret("Cross[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "{-3, 6, -3}"
    );
  }

  #[test]
  fn cross_unit_vectors() {
    assert_eq!(
      interpret("Cross[{1, 0, 0}, {0, 1, 0}]").unwrap(),
      "{0, 0, 1}"
    );
  }

  #[test]
  fn cross_symbolic() {
    // Negated products should be parenthesized: -(c*e) not -c*e
    let result = interpret("Cross[{a, b, c}, {d, e, f}]").unwrap();
    assert!(result.contains("-(c*e)"), "Expected -(c*e) in {}", result);
    assert!(result.contains("b*f"), "Expected b*f in {}", result);
    assert!(result.contains("-(b*d)"), "Expected -(b*d) in {}", result);
    assert!(result.contains("a*e"), "Expected a*e in {}", result);
  }

  #[test]
  fn cross_unit_vectors_yz() {
    assert_eq!(
      interpret("Cross[{0, 1, 0}, {0, 0, 1}]").unwrap(),
      "{1, 0, 0}"
    );
  }

  #[test]
  fn cross_unit_vectors_zx() {
    assert_eq!(
      interpret("Cross[{0, 0, 1}, {1, 0, 0}]").unwrap(),
      "{0, 1, 0}"
    );
  }

  #[test]
  fn cross_symbolic_return() {
    // Non-3-vectors should return unevaluated
    assert_eq!(
      interpret("Cross[{1, 2}, {3, 4}]").unwrap(),
      "Cross[{1, 2}, {3, 4}]"
    );
  }

  #[test]
  fn cross_2d() {
    assert_eq!(interpret("Cross[{x, y}]").unwrap(), "{-y, x}");
  }

  #[test]
  fn cross_2d_numeric() {
    assert_eq!(interpret("Cross[{3, 4}]").unwrap(), "{-4, 3}");
  }

  #[test]
  fn cross_2d_sqrt() {
    assert_eq!(interpret("Cross[{1, Sqrt[3]}]").unwrap(), "{-Sqrt[3], 1}");
  }
}

mod projection {
  use super::*;

  #[test]
  fn projection_basic() {
    assert_eq!(
      interpret("Projection[{5, 6, 7}, {1, 0, 0}]").unwrap(),
      "{5, 0, 0}"
    );
  }

  #[test]
  fn projection_2d() {
    assert_eq!(
      interpret("Projection[{2, 3}, {1, 2}]").unwrap(),
      "{8/5, 16/5}"
    );
  }
}

mod eigenvalues {
  use super::*;

  #[test]
  fn eigenvalues_1x1() {
    assert_eq!(interpret("Eigenvalues[{{5}}]").unwrap(), "{5}");
  }

  #[test]
  fn eigenvalues_2x2_integer() {
    assert_eq!(
      interpret("Eigenvalues[{{2, 1}, {1, 2}}]").unwrap(),
      "{3, 1}"
    );
  }

  #[test]
  fn eigenvalues_2x2_diagonal() {
    assert_eq!(
      interpret("Eigenvalues[{{2, 0}, {0, 3}}]").unwrap(),
      "{3, 2}"
    );
  }

  #[test]
  fn eigenvalues_2x2_zero() {
    assert_eq!(
      interpret("Eigenvalues[{{0, 0}, {0, 0}}]").unwrap(),
      "{0, 0}"
    );
  }

  #[test]
  fn eigenvalues_2x2_with_negative() {
    assert_eq!(
      interpret("Eigenvalues[{{4, -2}, {1, 1}}]").unwrap(),
      "{3, 2}"
    );
  }

  #[test]
  fn eigenvalues_2x2_irrational() {
    assert_eq!(
      interpret("Eigenvalues[{{1, 2}, {3, 4}}]").unwrap(),
      "{(5 + Sqrt[33])/2, (5 - Sqrt[33])/2}"
    );
  }

  #[test]
  fn eigenvalues_3x3_diagonal() {
    assert_eq!(
      interpret("Eigenvalues[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "{3, 2, 1}"
    );
  }

  #[test]
  fn eigenvalues_3x3_with_zero() {
    // {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}} has eigenvalues involving Sqrt[33]
    assert_eq!(
      interpret("Eigenvalues[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "{(3*(5 + Sqrt[33]))/2, (3*(5 - Sqrt[33]))/2, 0}"
    );
  }
}
