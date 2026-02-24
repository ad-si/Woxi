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
