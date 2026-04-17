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
  fn vector_matrix() {
    // Regression: Dot[vector, matrix] used to return unevaluated
    assert_eq!(
      interpret("Dot[{1, 2, 3}, IdentityMatrix[3]]").unwrap(),
      "{1, 2, 3}"
    );
    assert_eq!(
      interpret("{1, 2} . {{1, 2, 3}, {4, 5, 6}}").unwrap(),
      "{9, 12, 15}"
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

  #[test]
  fn det_symbolic_2x2() {
    assert_eq!(interpret("Det[{{a, b}, {c, d}}]").unwrap(), "-(b*c) + a*d");
  }

  #[test]
  fn det_symbolic_3x3() {
    assert_eq!(
      interpret("Det[{{a, b, c}, {d, e, f}, {g, h, i}}]").unwrap(),
      "-(c*e*g) + b*f*g + c*d*h - a*f*h - b*d*i + a*e*i"
    );
  }

  #[test]
  fn det_singular_3x3() {
    assert_eq!(
      interpret("Det[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "0"
    );
  }

  #[test]
  fn det_1x1() {
    assert_eq!(interpret("Det[{{7}}]").unwrap(), "7");
  }

  #[test]
  fn det_rational() {
    assert_eq!(interpret("Det[{{1/2, 1/3}, {1/4, 1/5}}]").unwrap(), "1/60");
  }

  #[test]
  fn det_unevaluated() {
    assert_eq!(interpret("Det[x]").unwrap(), "Det[x]");
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

  #[test]
  fn inverse_rational_matrix() {
    // Inverse of Hilbert-like matrix with rational entries must stay exact
    assert_eq!(
      interpret("Inverse[Table[1 / (i + j + 1), {i, 3}, {j, 3}]]").unwrap(),
      "{{300, -900, 630}, {-900, 2880, -2100}, {630, -2100, 1575}}"
    );
  }

  #[test]
  fn inverse_rational_2x2() {
    // Inverse of a 2x2 matrix with rational entries
    assert_eq!(
      interpret("Inverse[{{1/2, 1/3}, {1/4, 1/5}}]").unwrap(),
      "{{12, -20}, {-15, 30}}"
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

  #[test]
  fn trace_with_times() {
    assert_eq!(
      interpret("Tr[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, Times]").unwrap(),
      "45"
    );
  }

  #[test]
  fn trace_symbolic_with_times() {
    assert_eq!(interpret("Tr[{{a, b}, {c, d}}, Times]").unwrap(), "a*d");
  }

  #[test]
  fn trace_vector_with_times() {
    assert_eq!(interpret("Tr[{2, 3, 4}, Times]").unwrap(), "24");
  }

  #[test]
  fn trace_matrix_with_list_returns_diagonal() {
    // Tr[m, List] returns the list of diagonal entries.
    assert_eq!(
      interpret("Tr[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, List]").unwrap(),
      "{1, 5, 9}"
    );
    // Non-square matrix: take min(rows, cols) diagonal entries.
    assert_eq!(
      interpret("Tr[{{1, 2, 3}, {4, 5, 6}}, List]").unwrap(),
      "{1, 5}"
    );
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

  #[test]
  fn superdiagonal() {
    assert_eq!(
      interpret("DiagonalMatrix[{a, b}, 1]").unwrap(),
      "{{0, a, 0}, {0, 0, b}, {0, 0, 0}}"
    );
  }

  #[test]
  fn subdiagonal() {
    assert_eq!(
      interpret("DiagonalMatrix[{a, b}, -1]").unwrap(),
      "{{0, 0, 0}, {a, 0, 0}, {0, b, 0}}"
    );
  }

  #[test]
  fn explicit_zero_offset() {
    assert_eq!(
      interpret("DiagonalMatrix[{1, 2, 3}, 0]").unwrap(),
      "{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}"
    );
  }

  #[test]
  fn superdiagonal_three_elements() {
    assert_eq!(
      interpret("DiagonalMatrix[{a, b, c}, 1]").unwrap(),
      "{{0, a, 0, 0}, {0, 0, b, 0}, {0, 0, 0, c}, {0, 0, 0, 0}}"
    );
  }

  #[test]
  fn subdiagonal_three_elements() {
    assert_eq!(
      interpret("DiagonalMatrix[{a, b, c}, -1]").unwrap(),
      "{{0, 0, 0, 0}, {a, 0, 0, 0}, {0, b, 0, 0}, {0, 0, c, 0}}"
    );
  }

  #[test]
  fn offset_2() {
    assert_eq!(
      interpret("DiagonalMatrix[{x}, 2]").unwrap(),
      "{{0, 0, x}, {0, 0, 0}, {0, 0, 0}}"
    );
  }

  #[test]
  fn offset_neg2() {
    assert_eq!(
      interpret("DiagonalMatrix[{x}, -2]").unwrap(),
      "{{0, 0, 0}, {0, 0, 0}, {x, 0, 0}}"
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

  // Regression: the cross product operator (⨯, U+2A2F) should desugar to
  // Cross[a, b]. Also accepts the PUA form U+F3C4 and the `\[Cross]` escape.
  #[test]
  fn cross_operator_unicode_u2a2f() {
    assert_eq!(
      interpret("{1, 2, 3} \u{2A2F} {4, 5, 6}").unwrap(),
      "{-3, 6, -3}"
    );
  }

  #[test]
  fn cross_operator_symbolic() {
    assert_eq!(
      interpret("{a, b, c} \u{2A2F} {d, e, f}").unwrap(),
      "{-(c*e) + b*f, c*d - a*f, -(b*d) + a*e}"
    );
  }

  #[test]
  fn cross_operator_named_character() {
    assert_eq!(
      interpret("{1, 2, 3} \\[Cross] {4, 5, 6}").unwrap(),
      "{-3, 6, -3}"
    );
  }

  #[test]
  fn cross_operator_pua() {
    // U+F3C4 is Mathematica's PUA representation of \[Cross].
    assert_eq!(
      interpret("{1, 2, 3} \u{F3C4} {4, 5, 6}").unwrap(),
      "{-3, 6, -3}"
    );
  }

  #[test]
  fn cross_operator_parses_to_cross_function_call() {
    // Flat/associative: a ⨯ b ⨯ c → Cross[a, b, c]
    assert_eq!(
      interpret("FullForm[Hold[a \u{2A2F} b \u{2A2F} c]]").unwrap(),
      "Hold[Cross[a, b, c]]"
    );
  }

  #[test]
  fn cross_operator_inside_hold() {
    assert_eq!(
      interpret("FullForm[Hold[{1, 2, 3} \u{2A2F} {4, 5, 6}]]").unwrap(),
      "Hold[Cross[List[1, 2, 3], List[4, 5, 6]]]"
    );
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

mod conjugate_transpose {
  use super::*;

  #[test]
  fn conjugate_transpose_complex() {
    assert_eq!(
      interpret("ConjugateTranspose[{{0, I}, {0, 0}}]").unwrap(),
      "{{0, 0}, {-I, 0}}"
    );
  }

  #[test]
  fn conjugate_transpose_real() {
    assert_eq!(
      interpret("ConjugateTranspose[{{1, 2}, {3, 4}}]").unwrap(),
      "{{1, 3}, {2, 4}}"
    );
  }

  #[test]
  fn transpose_postfix_named_char() {
    // \[Transpose] is a postfix operator: expr \[Transpose] → Transpose[expr]
    assert_eq!(
      interpret("{{1,2,3}, {4,5,6}} \\[Transpose]").unwrap(),
      "{{1, 4}, {2, 5}, {3, 6}}"
    );
  }

  #[test]
  fn transpose_postfix_on_identifier() {
    assert_eq!(
      interpret("m = {{1,2,3}, {4,5,6}}; m \\[Transpose]").unwrap(),
      "{{1, 4}, {2, 5}, {3, 6}}"
    );
  }

  #[test]
  fn conjugate_transpose_postfix_named_char() {
    // \[ConjugateTranspose] is the postfix form of ConjugateTranspose
    assert_eq!(
      interpret("{{0, I}, {0, 0}} \\[ConjugateTranspose]").unwrap(),
      "{{0, 0}, {-I, 0}}"
    );
  }
}

mod box_matrix {
  use super::*;

  #[test]
  fn box_matrix_1() {
    assert_eq!(
      interpret("BoxMatrix[1]").unwrap(),
      "{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}"
    );
  }

  #[test]
  fn box_matrix_0() {
    assert_eq!(interpret("BoxMatrix[0]").unwrap(), "{{1}}");
  }
}

mod row_reduce {
  use super::*;

  #[test]
  fn row_reduce_2x3() {
    assert_eq!(
      interpret("RowReduce[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "{{1, 0, -1}, {0, 1, 2}, {0, 0, 0}}"
    );
  }

  #[test]
  fn row_reduce_symbolic() {
    assert_eq!(
      interpret("RowReduce[{{1, 0, a}, {1, 1, b}}]").unwrap(),
      "{{1, 0, a}, {0, 1, -a + b}}"
    );
  }

  #[test]
  fn row_reduce_identity() {
    assert_eq!(
      interpret("RowReduce[{{1, 0}, {0, 1}}]").unwrap(),
      "{{1, 0}, {0, 1}}"
    );
  }
}

mod matrix_rank {
  use super::*;

  #[test]
  fn rank_3x3_deficient() {
    assert_eq!(
      interpret("MatrixRank[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn rank_3x3_full() {
    assert_eq!(
      interpret("MatrixRank[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]").unwrap(),
      "3"
    );
  }

  #[test]
  fn rank_symbolic() {
    assert_eq!(interpret("MatrixRank[{{a, b}, {3 a, 3 b}}]").unwrap(), "1");
  }
}

mod null_space {
  use super::*;

  #[test]
  fn null_space_3x3() {
    assert_eq!(
      interpret("NullSpace[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "{{1, -2, 1}}"
    );
  }

  #[test]
  fn null_space_full_rank() {
    assert_eq!(interpret("NullSpace[{{1, 0}, {0, 1}}]").unwrap(), "{}");
  }
}

mod fit {
  use super::*;

  #[test]
  fn linear_fit_implicit_x() {
    assert_eq!(
      interpret("Fit[{1, 2, 3, 4, 5}, {1, x}, x]").unwrap(),
      "0. + 1.*x"
    );
  }

  #[test]
  fn linear_fit_explicit_xy_pairs() {
    assert_eq!(
      interpret("Fit[{{0, 1}, {1, 0}, {3, 2}, {5, 4}}, {1, x}, x]").unwrap(),
      "0.18644067796610186 + 0.6949152542372881*x"
    );
  }

  #[test]
  fn exact_linear_fit() {
    assert_eq!(
      interpret("Fit[{{1, 3}, {2, 5}, {3, 7}, {4, 9}}, {1, x}, x]").unwrap(),
      "1. + 2.*x"
    );
  }

  #[test]
  fn quadratic_fit() {
    assert_eq!(
      interpret("Fit[{1.2, 2.5, 3.7, 4.1, 5.8}, {1, x, x^2}, x]").unwrap(),
      "0.2199999999999992 + 1.0800000000000003*x - 4.450793765416313*^-17*x^2"
    );
  }

  #[test]
  fn sin_basis() {
    assert_eq!(
      interpret("Fit[{1, 2, 3}, {1, Sin[x]}, x]").unwrap(),
      "3.220973081736608 - 1.9361180115491952*Sin[x]"
    );
  }

  #[test]
  fn log_factorial_fit() {
    assert_eq!(
      interpret("Fit[N[Log[Table[Factorial[n], {n, 1, 20}]]], {1, x, x^2}, x]")
        .unwrap(),
      "-2.0296326640742994 + 1.1790183565363432*x + 0.053116572938742876*x^2"
    );
  }

  #[test]
  fn single_basis_function() {
    assert_eq!(interpret("Fit[{2, 4, 6}, {x}, x]").unwrap(), "2.*x");
  }

  #[test]
  fn two_data_points_linear() {
    assert_eq!(interpret("Fit[{1, 2}, {1, x}, x]").unwrap(), "0. + 1.*x");
  }
}

mod linear_solve {
  use super::*;

  #[test]
  fn solve_2x2() {
    // LinearSolve[{{1, 2}, {3, 4}}, {5, 6}] = {-4, 9/2}
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {3, 4}}, {5, 6}]").unwrap(),
      "{-4, 9/2}"
    );
  }

  #[test]
  fn solve_diagonal() {
    // Diagonal matrix
    assert_eq!(
      interpret("LinearSolve[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, {1, 1, 1}]")
        .unwrap(),
      "{1, 1/2, 1/3}"
    );
  }

  #[test]
  fn solve_3x3() {
    assert_eq!(
      interpret("LinearSolve[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}, {1, 0, 0}]")
        .unwrap(),
      "{-2/3, -2/3, 1}"
    );
  }

  #[test]
  fn solve_identity() {
    // Identity matrix returns the vector itself
    assert_eq!(
      interpret("LinearSolve[{{1, 0}, {0, 1}}, {3, 7}]").unwrap(),
      "{3, 7}"
    );
  }

  #[test]
  fn solve_1x1() {
    assert_eq!(interpret("LinearSolve[{{5}}, {10}]").unwrap(), "{2}");
  }

  #[test]
  fn solve_with_rationals() {
    // Result should be exact rational
    assert_eq!(
      interpret("LinearSolve[{{2, 1}, {1, 3}}, {1, 1}]").unwrap(),
      "{2/5, 1/5}"
    );
  }

  #[test]
  fn solve_matrix_rhs_2x2() {
    // When b is a matrix, the result must stay in exact rational form,
    // not be converted to Reals. Regression test for a bug where list
    // threading in Plus/Times/Divide pushed values through f64.
    assert_eq!(
      interpret(
        "LinearSolve[{{1, 9, 8}, {1, 2, 7}, {3, 8, 4}}, \
         {{6, 5}, {4, 3}, {2, 1}}]"
      )
      .unwrap(),
      "{{-82/121, -109/121}, {24/121, 26/121}, {74/121, 60/121}}"
    );
  }

  #[test]
  fn solve_matrix_rhs_3x3_identity() {
    // Solving m.x = I gives the inverse of m, with exact rational entries.
    assert_eq!(
      interpret(
        "LinearSolve[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}, \
         {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]"
      )
      .unwrap(),
      "{{-2/3, -4/3, 1}, {-2/3, 11/3, -2}, {1, -2, 1}}"
    );
  }
}

mod eigenvectors {
  use super::*;

  #[test]
  fn matrix_1x1() {
    assert_eq!(interpret("Eigenvectors[{{5}}]").unwrap(), "{{1}}");
  }

  #[test]
  fn diagonal_2x2() {
    assert_eq!(
      interpret("Eigenvectors[{{2, 0}, {0, 3}}]").unwrap(),
      "{{0, 1}, {1, 0}}"
    );
  }

  #[test]
  fn identity_2x2() {
    assert_eq!(
      interpret("Eigenvectors[{{1, 0}, {0, 1}}]").unwrap(),
      "{{0, 1}, {1, 0}}"
    );
  }

  #[test]
  fn symbolic_2x2() {
    assert_eq!(
      interpret("Eigenvectors[{{1, 2}, {3, 4}}]").unwrap(),
      "{{(-3 + Sqrt[33])/6, 1}, {(-3 - Sqrt[33])/6, 1}}"
    );
  }

  #[test]
  fn upper_triangular_3x3() {
    assert_eq!(
      interpret("Eigenvectors[{{1, 2, 3}, {0, 4, 5}, {0, 0, 6}}]").unwrap(),
      "{{16, 25, 10}, {2, 3, 0}, {1, 0, 0}}"
    );
  }

  #[test]
  fn defective_matrix() {
    assert_eq!(
      interpret("Eigenvectors[{{1, 1}, {0, 1}}]").unwrap(),
      "{{1, 0}, {0, 0}}"
    );
  }

  #[test]
  fn symbolic_variable() {
    assert_eq!(interpret("Eigenvectors[x]").unwrap(), "Eigenvectors[x]");
  }
}

mod signature {
  use super::*;

  #[test]
  fn identity() {
    assert_eq!(interpret("Signature[{1, 2, 3}]").unwrap(), "1");
  }

  #[test]
  fn even_permutation() {
    assert_eq!(interpret("Signature[{2, 3, 1}]").unwrap(), "1");
  }

  #[test]
  fn odd_permutation() {
    assert_eq!(interpret("Signature[{2, 1, 3}]").unwrap(), "-1");
  }

  #[test]
  fn reverse() {
    assert_eq!(interpret("Signature[{3, 2, 1}]").unwrap(), "-1");
  }

  #[test]
  fn duplicate_elements() {
    assert_eq!(interpret("Signature[{1, 2, 1}]").unwrap(), "0");
  }

  #[test]
  fn single_element() {
    assert_eq!(interpret("Signature[{1}]").unwrap(), "1");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("Signature[{c, a, b}]").unwrap(), "1");
  }

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Signature[x]").unwrap(), "Signature[x]");
  }
}

mod minors {
  use super::*;

  #[test]
  fn three_by_three() {
    assert_eq!(
      interpret("Minors[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "{{-3, -6, -3}, {-6, -12, -6}, {-3, -6, -3}}"
    );
  }

  #[test]
  fn three_by_three_k2() {
    assert_eq!(
      interpret("Minors[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 2]").unwrap(),
      "{{-3, -6, -3}, {-6, -12, -6}, {-3, -6, -3}}"
    );
  }

  #[test]
  fn three_by_three_k1() {
    assert_eq!(
      interpret("Minors[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}"
    );
  }

  #[test]
  fn two_by_two() {
    assert_eq!(
      interpret("Minors[{{1, 2}, {3, 4}}]").unwrap(),
      "{{1, 2}, {3, 4}}"
    );
  }

  #[test]
  fn two_by_two_k2() {
    assert_eq!(interpret("Minors[{{1, 2}, {3, 4}}, 2]").unwrap(), "{{-2}}");
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("Minors[{{a, b}, {c, d}}, 2]").unwrap(),
      "{{-(b*c) + a*d}}"
    );
  }

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("Minors[x]").unwrap(), "Minors[x]");
  }
}

mod pseudo_inverse {
  use super::*;

  #[test]
  fn invertible_2x2() {
    assert_eq!(
      interpret("PseudoInverse[{{1, 2}, {3, 4}}]").unwrap(),
      "{{-2, 1}, {3/2, -1/2}}"
    );
  }

  #[test]
  fn invertible_1x1() {
    assert_eq!(interpret("PseudoInverse[{{5}}]").unwrap(), "{{1/5}}");
  }

  #[test]
  fn rectangular_2x3() {
    assert_eq!(
      interpret("PseudoInverse[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "{{-17/18, 4/9}, {-1/9, 1/9}, {13/18, -2/9}}"
    );
  }

  #[test]
  fn row_vector() {
    assert_eq!(
      interpret("PseudoInverse[{{1, 2, 3}}]").unwrap(),
      "{{1/14}, {1/7}, {3/14}}"
    );
  }

  #[test]
  fn column_vector() {
    assert_eq!(
      interpret("PseudoInverse[{{1}, {2}, {3}}]").unwrap(),
      "{{1/14, 1/7, 3/14}}"
    );
  }

  #[test]
  fn zero_matrix() {
    assert_eq!(
      interpret("PseudoInverse[{{0, 0}, {0, 0}}]").unwrap(),
      "{{0, 0}, {0, 0}}"
    );
  }

  #[test]
  fn identity_property_row_vector() {
    // M.M+.M == M for a row vector
    assert_eq!(
      interpret("{{1, 2, 3}}.PseudoInverse[{{1, 2, 3}}].{{1, 2, 3}}").unwrap(),
      "{{1, 2, 3}}"
    );
  }

  #[test]
  fn identity_property_rectangular() {
    // M+.M.M+ == M+ for a rectangular matrix
    let pi = interpret("PseudoInverse[{{1, 2, 3}, {4, 5, 6}}]").unwrap();
    let check =
      interpret(&format!("{}.{{{{1, 2, 3}}, {{4, 5, 6}}}}.{}", pi, pi))
        .unwrap();
    assert_eq!(check, pi);
  }

  #[test]
  fn unevaluated() {
    assert_eq!(interpret("PseudoInverse[x]").unwrap(), "PseudoInverse[x]");
  }
}

mod lattice_reduce {
  use super::*;

  #[test]
  fn basic_2x2() {
    assert_eq!(
      interpret("LatticeReduce[{{1, 2}, {3, 4}}]").unwrap(),
      "{{1, 0}, {0, 2}}"
    );
  }

  #[test]
  fn single_vector() {
    assert_eq!(
      interpret("LatticeReduce[{{1, 2, 3}}]").unwrap(),
      "{{1, 2, 3}}"
    );
  }

  #[test]
  fn removes_dependent() {
    assert_eq!(
      interpret("LatticeReduce[{{1, 0}, {0, 1}, {2, 3}}]").unwrap(),
      "{{1, 0}, {0, 1}}"
    );
  }

  #[test]
  fn removes_dependent_3d() {
    assert_eq!(
      interpret("LatticeReduce[{{1, 0, 0}, {0, 1, 0}, {1, 1, 0}}]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}}"
    );
  }

  #[test]
  fn rank_deficient_3x3() {
    // Matrix with rank 2: last row is sum of first two
    let result =
      interpret("LatticeReduce[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap();
    // Should produce 2 vectors (rank 2)
    assert!(result.starts_with("{{") && result.ends_with("}}"));
    // Count the number of inner lists
    let inner = &result[1..result.len() - 1];
    let count = inner.matches("}, {").count() + 1;
    assert_eq!(count, 2);
  }

  #[test]
  fn unevaluated_symbolic() {
    assert_eq!(interpret("LatticeReduce[x]").unwrap(), "LatticeReduce[x]");
  }
}

mod kronecker_product {
  use super::*;

  #[test]
  fn two_by_two_numeric() {
    assert_eq!(
      interpret("KroneckerProduct[{{1, 2}, {3, 4}}, {{0, 5}, {6, 7}}]")
        .unwrap(),
      "{{0, 5, 0, 10}, {6, 7, 12, 14}, {0, 15, 0, 20}, {18, 21, 24, 28}}"
    );
  }

  #[test]
  fn identity_with_symbolic() {
    assert_eq!(
      interpret("KroneckerProduct[{{1, 0}, {0, 1}}, {{a, b}, {c, d}}]")
        .unwrap(),
      "{{a, b, 0, 0}, {c, d, 0, 0}, {0, 0, a, b}, {0, 0, c, d}}"
    );
  }

  #[test]
  fn one_by_one() {
    assert_eq!(
      interpret("KroneckerProduct[{{2}}, {{3}}]").unwrap(),
      "{{6}}"
    );
  }
}

mod find_fit {
  use super::*;

  #[test]
  fn quadratic() {
    let result =
      interpret("FindFit[{{1,1},{2,4},{3,9}}, a*x^2 + b, {a,b}, x]").unwrap();
    // a should be ~1.0, b should be ~0.0
    assert!(
      result.contains("a -> "),
      "Expected a rule for a, got: {}",
      result
    );
    assert!(
      result.contains("b -> "),
      "Expected a rule for b, got: {}",
      result
    );
    // Parse a value
    let a_val: f64 = result
      .split("a -> ")
      .nth(1)
      .unwrap()
      .split(',')
      .next()
      .unwrap()
      .trim()
      .parse()
      .unwrap();
    assert!(
      (a_val - 1.0).abs() < 0.001,
      "Expected a ≈ 1.0, got {}",
      a_val
    );
  }

  #[test]
  fn simple_linear() {
    // Fit a*x to data {1,2,3} (x=1,2,3)
    let result = interpret("FindFit[{1,4,9}, a*x^2, {a}, x]").unwrap();
    assert!(
      result.contains("a -> 1."),
      "Expected a -> 1., got: {}",
      result
    );
  }

  #[test]
  fn exponential_model() {
    let result = interpret(
      "FindFit[{{0,1},{1,2.7},{2,7.4},{3,20.1}}, a*E^(b*x), {a,b}, x]",
    )
    .unwrap();
    // a ≈ 1, b ≈ 1
    assert!(result.contains("a -> "), "Expected a rule, got: {}", result);
    assert!(result.contains("b -> "), "Expected b rule, got: {}", result);
  }
}

mod lu_decomposition {
  use super::*;

  #[test]
  fn basic_2x2() {
    assert_eq!(
      interpret("LUDecomposition[{{1, 2}, {3, 4}}]").unwrap(),
      "{{{1, 2}, {3, -2}}, {1, 2}, 0}"
    );
  }

  #[test]
  fn basic_3x3() {
    assert_eq!(
      interpret("LUDecomposition[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}]").unwrap(),
      "{{{1, 2, 3}, {4, -3, -6}, {7, 2, 1}}, {1, 2, 3}, 0}"
    );
  }

  #[test]
  fn with_pivoting() {
    assert_eq!(
      interpret("LUDecomposition[{{0, 1}, {1, 0}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {2, 1}, 0}"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("LUDecomposition[{{a, b}, {c, d}}]").unwrap(),
      "{{{a, b}, {c/a, -((b*c)/a) + d}}, {1, 2}, 0}"
    );
  }

  #[test]
  fn identity() {
    assert_eq!(
      interpret("LUDecomposition[{{1, 0}, {0, 1}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {1, 2}, 0}"
    );
  }
}

mod vector_angle {
  use super::*;

  #[test]
  fn orthogonal_2d() {
    assert_eq!(interpret("VectorAngle[{1, 0}, {0, 1}]").unwrap(), "Pi/2");
  }

  #[test]
  fn parallel_2d() {
    assert_eq!(interpret("VectorAngle[{1, 1}, {1, 1}]").unwrap(), "0");
  }

  #[test]
  fn antiparallel_2d() {
    assert_eq!(interpret("VectorAngle[{3, 4}, {-3, -4}]").unwrap(), "Pi");
  }

  #[test]
  fn pi_over_4() {
    assert_eq!(interpret("VectorAngle[{1, 1}, {1, 0}]").unwrap(), "Pi/4");
  }

  #[test]
  fn orthogonal_3d() {
    assert_eq!(
      interpret("VectorAngle[{1, 0, 0}, {0, 1, 0}]").unwrap(),
      "Pi/2"
    );
  }

  #[test]
  fn perpendicular_2d() {
    assert_eq!(interpret("VectorAngle[{1, 2}, {-2, 1}]").unwrap(), "Pi/2");
  }

  #[test]
  fn numeric_3d() {
    // VectorAngle[{1,2,3},{4,5,6}] = ArcCos[32/(Sqrt[14]*Sqrt[77])]
    let result = interpret("N[VectorAngle[{1, 2, 3}, {4, 5, 6}]]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.2257261285527342).abs() < 1e-10);
  }

  #[test]
  fn numeric_float_input() {
    let result = interpret("VectorAngle[{1.0, 0}, {0, 1.0}]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
  }

  #[test]
  fn zero_vector_indeterminate() {
    assert_eq!(
      interpret("VectorAngle[{0, 0}, {1, 0}]").unwrap(),
      "Indeterminate"
    );
  }

  #[test]
  fn different_lengths_unevaluated() {
    assert_eq!(
      interpret("VectorAngle[{1, 2}, {3, 4, 5}]").unwrap(),
      "VectorAngle[{1, 2}, {3, 4, 5}]"
    );
  }

  #[test]
  fn non_list_args_unevaluated() {
    assert_eq!(interpret("VectorAngle[x, y]").unwrap(), "VectorAngle[x, y]");
  }

  #[test]
  fn one_d_vectors() {
    assert_eq!(interpret("VectorAngle[{1}, {2}]").unwrap(), "0");
  }
}

mod upper_triangularize {
  use super::*;

  #[test]
  fn basic_3x3() {
    assert_eq!(
      interpret("UpperTriangularize[{{1,2,3},{4,5,6},{7,8,9}}]").unwrap(),
      "{{1, 2, 3}, {0, 5, 6}, {0, 0, 9}}"
    );
  }

  #[test]
  fn with_offset_1() {
    assert_eq!(
      interpret("UpperTriangularize[{{1,2,3},{4,5,6},{7,8,9}}, 1]").unwrap(),
      "{{0, 2, 3}, {0, 0, 6}, {0, 0, 0}}"
    );
  }

  #[test]
  fn with_offset_neg1() {
    assert_eq!(
      interpret("UpperTriangularize[{{1,2,3},{4,5,6},{7,8,9}}, -1]").unwrap(),
      "{{1, 2, 3}, {4, 5, 6}, {0, 8, 9}}"
    );
  }

  #[test]
  fn symbolic_2x2() {
    assert_eq!(
      interpret("UpperTriangularize[{{a,b},{c,d}}]").unwrap(),
      "{{a, b}, {0, d}}"
    );
  }

  #[test]
  fn non_square() {
    assert_eq!(
      interpret("UpperTriangularize[{{1,2},{3,4},{5,6}}]").unwrap(),
      "{{1, 2}, {0, 4}, {0, 0}}"
    );
  }
}

mod lower_triangularize {
  use super::*;

  #[test]
  fn basic_3x3() {
    assert_eq!(
      interpret("LowerTriangularize[{{1,2,3},{4,5,6},{7,8,9}}]").unwrap(),
      "{{1, 0, 0}, {4, 5, 0}, {7, 8, 9}}"
    );
  }

  #[test]
  fn with_offset_1() {
    assert_eq!(
      interpret("LowerTriangularize[{{1,2,3},{4,5,6},{7,8,9}}, 1]").unwrap(),
      "{{1, 2, 0}, {4, 5, 6}, {7, 8, 9}}"
    );
  }

  #[test]
  fn symbolic_2x2() {
    assert_eq!(
      interpret("LowerTriangularize[{{a,b},{c,d}}]").unwrap(),
      "{{a, 0}, {c, d}}"
    );
  }
}

mod linear_model_fit {
  use super::*;

  #[test]
  fn normal_returns_fitted_expression() {
    let result = interpret(
      "lm = LinearModelFit[{{0, 1}, {1, 0}, {3, 2}, {5, 4}}, x, x]; Normal[lm]",
    )
    .unwrap();
    // Should return something like 0.186... + 0.694...*x
    assert!(result.contains("+"));
    assert!(result.contains("x"));
  }

  #[test]
  fn evaluate_at_point() {
    let result = interpret(
      "lm = LinearModelFit[{{0, 1}, {1, 0}, {3, 2}, {5, 4}}, x, x]; lm[2.3]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 1.7847457627118644).abs() < 1e-10);
  }

  #[test]
  fn best_fit_parameters() {
    let result = interpret(
      "lm = LinearModelFit[{{0, 1}, {1, 0}, {3, 2}, {5, 4}}, x, x]; lm[\"BestFitParameters\"]",
    )
    .unwrap();
    assert!(result.starts_with("{"));
    assert!(result.contains("0.186"));
    assert!(result.contains("0.694"));
  }

  #[test]
  fn fit_residuals() {
    let result = interpret(
      "lm = LinearModelFit[{{0, 1}, {1, 0}, {3, 2}, {5, 4}}, x, x]; lm[\"FitResiduals\"]",
    )
    .unwrap();
    assert!(result.starts_with("{"));
    assert!(result.contains("0.813"));
    assert!(result.contains("-0.881"));
  }

  #[test]
  fn r_squared() {
    let result = interpret(
      "lm = LinearModelFit[{{0, 1}, {1, 0}, {3, 2}, {5, 4}}, x, x]; lm[\"RSquared\"]",
    )
    .unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 0.814043583535109).abs() < 1e-10);
  }

  #[test]
  fn with_basis_list() {
    // LinearModelFit with explicit basis {1, x, x^2}
    let result = interpret(
      "lm = LinearModelFit[{{1, 1}, {2, 4}, {3, 9}}, {1, x, x^2}, x]; Normal[lm]",
    )
    .unwrap();
    assert!(result.contains("x"));
  }

  #[test]
  fn simple_data() {
    // LinearModelFit with y-only data (implicit x = 1, 2, ...)
    let result =
      interpret("lm = LinearModelFit[{1, 2, 3, 4, 5}, x, x]; lm[3]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 3.0).abs() < 1e-10);
  }
}

mod translation_transform {
  use super::*;

  #[test]
  fn creates_transformation_function_2d() {
    assert_eq!(
      interpret("TranslationTransform[{1, 2}]").unwrap(),
      "TransformationFunction[{{1, 0, 1}, {0, 1, 2}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn creates_transformation_function_3d() {
    assert_eq!(
      interpret("TranslationTransform[{1, 2, 3}]").unwrap(),
      "TransformationFunction[{{1, 0, 0, 1}, {0, 1, 0, 2}, {0, 0, 1, 3}, {0, 0, 0, 1}}]"
    );
  }

  #[test]
  fn apply_2d() {
    assert_eq!(
      interpret("TranslationTransform[{1, 2}][{3, 4}]").unwrap(),
      "{4, 6}"
    );
  }

  #[test]
  fn apply_3d() {
    assert_eq!(
      interpret("TranslationTransform[{1, 2, 3}][{0, 0, 0}]").unwrap(),
      "{1, 2, 3}"
    );
  }

  #[test]
  fn apply_symbolic() {
    assert_eq!(
      interpret("TranslationTransform[{a, b}][{x, y}]").unwrap(),
      "{a + x, b + y}"
    );
  }

  #[test]
  fn apply_origin() {
    assert_eq!(
      interpret("TranslationTransform[{5, -3}][{0, 0}]").unwrap(),
      "{5, -3}"
    );
  }
}

mod transformation_function_apply {
  use super::*;

  #[test]
  fn rotation_apply() {
    assert_eq!(
      interpret("RotationTransform[Pi/2][{1, 0}]").unwrap(),
      "{0, 1}"
    );
  }

  #[test]
  fn rotation_apply_diagonal() {
    assert_eq!(
      interpret("RotationTransform[Pi/2][{0, 1}]").unwrap(),
      "{-1, 0}"
    );
  }
}

mod tensor_rank {
  use super::*;

  #[test]
  fn scalar() {
    assert_eq!(interpret("TensorRank[42]").unwrap(), "0");
    assert_eq!(interpret("TensorRank[3.14]").unwrap(), "0");
    assert_eq!(interpret("TensorRank[2/3]").unwrap(), "0");
  }

  #[test]
  fn vector() {
    assert_eq!(interpret("TensorRank[{1, 2, 3}]").unwrap(), "1");
    assert_eq!(interpret("TensorRank[{}]").unwrap(), "1");
  }

  #[test]
  fn matrix() {
    assert_eq!(interpret("TensorRank[{{1, 2}, {3, 4}}]").unwrap(), "2");
    assert_eq!(interpret("TensorRank[{{}}]").unwrap(), "2");
  }

  #[test]
  fn higher() {
    assert_eq!(
      interpret("TensorRank[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}]").unwrap(),
      "3"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("TensorRank[x]").unwrap(), "TensorRank[x]");
    assert_eq!(interpret("TensorRank[f[x]]").unwrap(), "TensorRank[f[x]]");
  }
}

mod array_depth {
  use super::*;

  #[test]
  fn scalar() {
    assert_eq!(interpret("ArrayDepth[42]").unwrap(), "0");
  }

  #[test]
  fn vector() {
    assert_eq!(interpret("ArrayDepth[{1, 2, 3}]").unwrap(), "1");
  }

  #[test]
  fn matrix() {
    assert_eq!(interpret("ArrayDepth[{{1, 2}, {3, 4}}]").unwrap(), "2");
  }

  #[test]
  fn rank3() {
    assert_eq!(
      interpret("ArrayDepth[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}]").unwrap(),
      "3"
    );
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("ArrayDepth[{}]").unwrap(), "1");
  }

  #[test]
  fn ragged_different_lengths() {
    // Sublists have different lengths — not rectangular at level 2
    assert_eq!(interpret("ArrayDepth[{{1, 2}, {3}}]").unwrap(), "1");
  }

  #[test]
  fn ragged_mixed_types() {
    // First element is atomic, second is a list — not rectangular at level 2
    assert_eq!(interpret("ArrayDepth[{1, {2, 3}}]").unwrap(), "1");
  }

  #[test]
  fn ragged_deep() {
    // Sublists at level 2 differ in structure
    assert_eq!(
      interpret("ArrayDepth[{{{1, 2}, {3, 4}}, {{5, 6}}}]").unwrap(),
      "1"
    );
  }

  #[test]
  fn ragged_at_level_3() {
    // Rectangular at level 2 but ragged at level 3
    assert_eq!(
      interpret("ArrayDepth[{{{1}, {2}}, {{3, 4}, {5}}}]").unwrap(),
      "2"
    );
  }

  #[test]
  fn symbols_are_atoms() {
    assert_eq!(interpret("ArrayDepth[{a, b, c}]").unwrap(), "1");
  }

  #[test]
  fn wide_matrix() {
    assert_eq!(
      interpret("ArrayDepth[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "2"
    );
  }
}

mod symmetric_matrix_q {
  use super::*;

  #[test]
  fn symmetric_true() {
    assert_eq!(
      interpret("SymmetricMatrixQ[{{1, 2}, {2, 3}}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("SymmetricMatrixQ[{{1, 2, 3}, {2, 5, 6}, {3, 6, 9}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn symmetric_false() {
    assert_eq!(
      interpret("SymmetricMatrixQ[{{1, 2}, {3, 4}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn non_square() {
    assert_eq!(
      interpret("SymmetricMatrixQ[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn not_a_matrix() {
    assert_eq!(interpret("SymmetricMatrixQ[42]").unwrap(), "False");
    assert_eq!(interpret("SymmetricMatrixQ[x]").unwrap(), "False");
    assert_eq!(interpret("SymmetricMatrixQ[{1, 2, 3}]").unwrap(), "False");
    assert_eq!(interpret("SymmetricMatrixQ[{}]").unwrap(), "False");
  }

  #[test]
  fn single_element() {
    assert_eq!(interpret("SymmetricMatrixQ[{{1}}]").unwrap(), "True");
  }

  #[test]
  fn symbolic_entries() {
    assert_eq!(
      interpret("SymmetricMatrixQ[{{a, b}, {b, c}}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("SymmetricMatrixQ[{{a, b}, {c, d}}]").unwrap(),
      "False"
    );
  }
}

#[cfg(test)]
mod tensor_wedge {
  use super::*;

  #[test]
  fn two_vectors_2d() {
    // TensorWedge[{a, b}, {c, d}] = {{0, a*d - b*c}, {-(a*d - b*c), 0}}
    assert_eq!(
      interpret("TensorWedge[{1, 0}, {0, 1}]").unwrap(),
      "{{0, 1}, {-1, 0}}"
    );
  }

  #[test]
  fn two_vectors_3d_numeric() {
    // TensorWedge[{1, 0, 0}, {0, 1, 0}]
    // = {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}
    assert_eq!(
      interpret("TensorWedge[{1, 0, 0}, {0, 1, 0}]").unwrap(),
      "{{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}"
    );
  }

  #[test]
  fn antisymmetry() {
    // TensorWedge[u, v] = -TensorWedge[v, u]
    let uv = interpret("TensorWedge[{1, 2, 3}, {4, 5, 6}]").unwrap();
    let vu = interpret("TensorWedge[{4, 5, 6}, {1, 2, 3}]").unwrap();
    // Parse and check negation
    let uv_neg =
      interpret(&format!("Map[Function[x, -x], {}, {{2}}]", uv)).unwrap();
    assert_eq!(uv_neg, vu, "TensorWedge should be antisymmetric");
  }

  #[test]
  fn single_vector() {
    // TensorWedge[v] = v
    assert_eq!(interpret("TensorWedge[{1, 2, 3}]").unwrap(), "{1, 2, 3}");
  }

  #[test]
  fn parallel_vectors_zero() {
    // Wedge of parallel vectors is zero
    assert_eq!(
      interpret("TensorWedge[{1, 2}, {2, 4}]").unwrap(),
      "{{0, 0}, {0, 0}}"
    );
  }

  #[test]
  fn three_vectors_3d() {
    // TensorWedge[{1,0,0}, {0,1,0}, {0,0,1}] should give a rank-3 tensor
    // with the only nonzero entries being ±1 at even/odd permutations of (0,1,2)
    let result =
      interpret("TensorWedge[{1, 0, 0}, {0, 1, 0}, {0, 0, 1}]").unwrap();
    // The (0,1,2) entry should be 1 and (1,0,2) should be -1, etc.
    assert!(result.contains("1"), "should contain nonzero entries");
  }

  #[test]
  fn symbolic_unevaluated() {
    // Non-list args should return unevaluated
    assert_eq!(interpret("TensorWedge[a, b]").unwrap(), "TensorWedge[a, b]");
  }
}

#[cfg(test)]
mod logit_model_fit {
  use super::*;

  #[test]
  fn basic_logistic_regression() {
    // Simple dataset: low x -> y=0, high x -> y=1
    let result = interpret(
      "LogitModelFit[{{0, 0}, {1, 0}, {2, 0}, {3, 1}, {4, 1}, {5, 1}}, {1, x}, x]",
    )
    .unwrap();
    assert!(
      result.contains("FittedModel"),
      "expected FittedModel, got {}",
      result
    );
  }

  #[test]
  fn fitted_model_evaluation() {
    clear_state();
    // Fit a logistic model and evaluate at a point
    interpret(
      "model = LogitModelFit[{{0, 0}, {1, 0}, {2, 0}, {3, 1}, {4, 1}, {5, 1}}, {1, x}, x]",
    )
    .unwrap();

    // Evaluate at x=2.5 — should be near 0.5 (transition point)
    let result = interpret("model[2.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!(
      val > 0.2 && val < 0.8,
      "expected probability near 0.5 at transition, got {}",
      val
    );
  }

  #[test]
  fn extreme_predictions() {
    clear_state();
    interpret(
      "model = LogitModelFit[{{0, 0}, {1, 0}, {2, 0}, {3, 1}, {4, 1}, {5, 1}}, {1, x}, x]",
    )
    .unwrap();

    // At x=-5, should be close to 0
    let low = interpret("model[-5.0]").unwrap();
    let low_val: f64 = low.parse().unwrap();
    assert!(low_val < 0.1, "expected p < 0.1 at x=-5, got {}", low_val);

    // At x=10, should be close to 1
    let high = interpret("model[10.0]").unwrap();
    let high_val: f64 = high.parse().unwrap();
    assert!(high_val > 0.9, "expected p > 0.9 at x=10, got {}", high_val);
  }

  #[test]
  fn best_fit_parameters() {
    clear_state();
    interpret(
      "model = LogitModelFit[{{0, 0}, {1, 0}, {2, 0}, {3, 1}, {4, 1}, {5, 1}}, {1, x}, x]",
    )
    .unwrap();

    let params = interpret("model[\"BestFitParameters\"]").unwrap();
    assert!(
      params.starts_with('{'),
      "expected list of params, got {}",
      params
    );
  }
}

mod design_matrix {
  use super::*;

  #[test]
  fn linear_single_var() {
    assert_eq!(
      interpret("DesignMatrix[{{1, 1}, {2, 2}, {3, 3}}, x, x]").unwrap(),
      "{{1, 1}, {1, 2}, {1, 3}}"
    );
  }

  #[test]
  fn polynomial_basis() {
    assert_eq!(
      interpret("DesignMatrix[{{1, 1}, {2, 4}, {3, 9}}, {1, x, x^2}, x]")
        .unwrap(),
      "{{1, 1, 1}, {1, 2, 4}, {1, 3, 9}}"
    );
  }

  #[test]
  fn single_function() {
    assert_eq!(
      interpret("DesignMatrix[{{1, 1}, {2, 4}, {3, 9}}, x^2, x]").unwrap(),
      "{{1, 1}, {1, 4}, {1, 9}}"
    );
  }

  #[test]
  fn multivariate() {
    assert_eq!(
      interpret(
        "DesignMatrix[{{1, 2, 10}, {3, 4, 20}}, {1, x1, x2}, {x1, x2}]"
      )
      .unwrap(),
      "{{1, 1, 2}, {1, 3, 4}}"
    );
  }
}

mod find_integer_null_vector {
  use super::*;

  #[test]
  fn simple_integers() {
    assert_eq!(
      interpret("FindIntegerNullVector[{1, 2, 3}]").unwrap(),
      "{1, 1, -1}"
    );
  }

  #[test]
  fn rationals() {
    assert_eq!(
      interpret("FindIntegerNullVector[{1.0, 0.5, 0.25}]").unwrap(),
      "{1, -2, 0}"
    );
  }

  #[test]
  fn no_relation() {
    // Sqrt[2] and 1 are linearly independent over integers
    assert_eq!(
      interpret("FindIntegerNullVector[{1, Sqrt[2]}]").unwrap(),
      "FindIntegerNullVector[{1, Sqrt[2]}]"
    );
  }

  #[test]
  fn with_zero() {
    assert_eq!(
      interpret("FindIntegerNullVector[{Pi, 0, 1}]").unwrap(),
      "{0, 1, 0}"
    );
  }

  #[test]
  fn two_equal_values() {
    assert_eq!(
      interpret("FindIntegerNullVector[{3, 3}]").unwrap(),
      "{1, -1}"
    );
  }

  #[test]
  fn proportional_values() {
    // 2 and 6: 3*2 - 1*6 = 0
    assert_eq!(
      interpret("FindIntegerNullVector[{2, 6}]").unwrap(),
      "{3, -1}"
    );
  }

  #[test]
  fn sqrt2_relation() {
    // Sqrt[2]^2 = 2, so {Sqrt[2], Sqrt[2], -1} should work
    // i.e., FindIntegerNullVector[{2, 1}] = {1, -2}
    assert_eq!(
      interpret("FindIntegerNullVector[{2, 1}]").unwrap(),
      "{1, -2}"
    );
  }
}

mod vector_less {
  use super::*;

  #[test]
  fn all_less() {
    assert_eq!(interpret("VectorLess[{{1,2},{3,4}}]").unwrap(), "True");
  }

  #[test]
  fn not_all_less() {
    assert_eq!(interpret("VectorLess[{{3,2},{1,4}}]").unwrap(), "False");
  }

  #[test]
  fn equal_elements_strict() {
    assert_eq!(interpret("VectorLess[{{1,2},{1,3}}]").unwrap(), "False");
  }

  #[test]
  fn all_equal_strict() {
    assert_eq!(interpret("VectorLess[{{1,2},{1,2}}]").unwrap(), "False");
  }

  #[test]
  fn reverse_order() {
    assert_eq!(interpret("VectorLess[{{5,6},{3,4}}]").unwrap(), "False");
  }

  #[test]
  fn negative_values() {
    assert_eq!(interpret("VectorLess[{{-1,-2},{-3,-4}}]").unwrap(), "False");
  }

  #[test]
  fn zero_to_positive() {
    assert_eq!(interpret("VectorLess[{{0,0},{1,1}}]").unwrap(), "True");
  }

  #[test]
  fn three_vectors() {
    assert_eq!(
      interpret("VectorLess[{{1,2},{3,4},{5,6}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn three_vectors_fail() {
    assert_eq!(
      interpret("VectorLess[{{1,2},{3,4},{2,6}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn four_vectors() {
    assert_eq!(
      interpret("VectorLess[{{1,2},{3,4},{5,6},{7,8}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn scalars() {
    assert_eq!(interpret("VectorLess[{1,2}]").unwrap(), "True");
  }

  #[test]
  fn scalars_chain() {
    assert_eq!(interpret("VectorLess[{1,2,3}]").unwrap(), "True");
  }

  #[test]
  fn scalars_equal_strict() {
    assert_eq!(interpret("VectorLess[{1,1}]").unwrap(), "False");
  }

  #[test]
  fn three_dim_vectors() {
    assert_eq!(interpret("VectorLess[{{1,2,3},{4,5,6}}]").unwrap(), "True");
  }

  #[test]
  fn single_element_vectors() {
    assert_eq!(interpret("VectorLess[{{1},{2}}]").unwrap(), "True");
  }

  #[test]
  fn empty_vectors() {
    assert_eq!(interpret("VectorLess[{{},{}}]").unwrap(), "True");
  }

  #[test]
  fn mismatched_lengths() {
    assert_eq!(interpret("VectorLess[{{1,2},{3}}]").unwrap(), "False");
  }

  #[test]
  fn rationals() {
    assert_eq!(
      interpret("VectorLess[{{1/2, 3/4}, {5/6, 7/8}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("VectorLess[{{a,b},{c,d}}]").unwrap(),
      "VectorLess[{{a, b}, {c, d}}]"
    );
  }
}

mod vector_less_equal {
  use super::*;

  #[test]
  fn all_less() {
    assert_eq!(interpret("VectorLessEqual[{{1,2},{3,4}}]").unwrap(), "True");
  }

  #[test]
  fn not_all_less() {
    assert_eq!(
      interpret("VectorLessEqual[{{3,2},{1,4}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn equal_elements() {
    assert_eq!(interpret("VectorLessEqual[{{1,2},{1,3}}]").unwrap(), "True");
  }

  #[test]
  fn all_equal() {
    assert_eq!(interpret("VectorLessEqual[{{1,2},{1,2}}]").unwrap(), "True");
  }

  #[test]
  fn reverse_order() {
    assert_eq!(
      interpret("VectorLessEqual[{{5,6},{3,4}}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn three_vectors() {
    assert_eq!(
      interpret("VectorLessEqual[{{1,2},{3,4},{5,6}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn three_vectors_with_equal() {
    assert_eq!(
      interpret("VectorLessEqual[{{1,2},{3,4},{3,6}}]").unwrap(),
      "True"
    );
  }

  #[test]
  fn scalars() {
    assert_eq!(interpret("VectorLessEqual[{1,2}]").unwrap(), "True");
  }

  #[test]
  fn scalars_equal() {
    assert_eq!(interpret("VectorLessEqual[{1,1}]").unwrap(), "True");
  }

  #[test]
  fn empty_vectors() {
    assert_eq!(interpret("VectorLessEqual[{{},{}}]").unwrap(), "True");
  }

  #[test]
  fn mismatched_lengths() {
    assert_eq!(interpret("VectorLessEqual[{{1,2},{3}}]").unwrap(), "False");
  }

  #[test]
  fn symbolic_unevaluated() {
    assert_eq!(
      interpret("VectorLessEqual[{{1,2},{3,a}}]").unwrap(),
      "VectorLessEqual[{{1, 2}, {3, a}}]"
    );
  }
}
