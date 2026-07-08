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

mod array_dot {
  use super::*;

  #[test]
  fn vectors_contract_one() {
    // Contracting the single dimension of two vectors is the dot product.
    assert_eq!(
      interpret("ArrayDot[{1, 2, 3}, {4, 5, 6}, 1]").unwrap(),
      "32"
    );
  }

  #[test]
  fn matrices_contract_one() {
    // k = 1 over two matrices is ordinary matrix multiplication.
    assert_eq!(
      interpret("ArrayDot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, 1]").unwrap(),
      "{{19, 22}, {43, 50}}"
    );
  }

  #[test]
  fn matrices_full_contraction_scalar() {
    // k = 2 contracts every dimension, giving the Frobenius inner product.
    assert_eq!(
      interpret("ArrayDot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, 2]").unwrap(),
      "70"
    );
  }

  #[test]
  fn zero_contraction_is_outer_product() {
    assert_eq!(
      interpret("ArrayDot[{1, 2}, {3, 4}, 0]").unwrap(),
      "{{3, 4}, {6, 8}}"
    );
    assert_eq!(
      interpret("ArrayDot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, 0]").unwrap(),
      "{{{{5, 6}, {7, 8}}, {{10, 12}, {14, 16}}}, \
       {{{15, 18}, {21, 24}}, {{20, 24}, {28, 32}}}}"
    );
  }

  #[test]
  fn non_square_matrices() {
    assert_eq!(
      interpret("ArrayDot[{{1, 2, 3}}, {{1}, {2}, {3}}, 1]").unwrap(),
      "{{14}}"
    );
    assert_eq!(
      interpret("ArrayDot[{{1, 2}, {3, 4}}, {{5, 6, 7}, {8, 9, 10}}, 1]")
        .unwrap(),
      "{{21, 24, 27}, {47, 54, 61}}"
    );
  }

  #[test]
  fn symbolic_entries() {
    assert_eq!(
      interpret("ArrayDot[{a, b}, {c, d}, 1]").unwrap(),
      "a*c + b*d"
    );
  }

  #[test]
  fn higher_rank_dimensions() {
    assert_eq!(
      interpret(
        "Dimensions[ArrayDot[Array[a, {2, 3, 4}], Array[b, {4, 5}], 1]]"
      )
      .unwrap(),
      "{2, 3, 5}"
    );
    assert_eq!(
      interpret(
        "Dimensions[ArrayDot[Array[a, {2, 3, 4}], Array[b, {3, 4, 5}], 2]]"
      )
      .unwrap(),
      "{2, 5}"
    );
  }

  #[test]
  fn depth_exceeded_is_unevaluated() {
    // k larger than the array depth stays unevaluated (emits ArrayDot::kspec).
    assert_eq!(
      interpret("ArrayDot[{1, 2}, {3, 4}, 2]").unwrap(),
      "ArrayDot[{1, 2}, {3, 4}, 2]"
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

  #[test]
  fn det_large_integer_matrix_is_fast_and_exact() {
    // Regression: cofactor expansion is O(n!), so a 20x20 integer determinant
    // used to hang. The fraction-free Bareiss path computes it instantly.
    // {i + j} has rank 2, so its determinant is 0 for n >= 3.
    assert_eq!(
      interpret("Det[Table[i + j, {i, 20}, {j, 20}]]").unwrap(),
      "0"
    );
    // A non-singular larger matrix gives the exact value.
    assert_eq!(
      interpret("Det[Table[If[i == j, 2, 1], {i, 10}, {j, 10}]]").unwrap(),
      "11"
    );
    assert_eq!(interpret("Det[IdentityMatrix[25]]").unwrap(), "1");
    // Exact rational entries stay exact via fraction-free elimination.
    assert_eq!(
      interpret("Det[HilbertMatrix[5]]").unwrap(),
      "1/266716800000"
    );
  }
}

mod permanent {
  use super::*;

  #[test]
  fn numeric_matrices() {
    // Like the determinant but with all-positive signs: 1*4 + 2*3 = 10.
    assert_eq!(interpret("Permanent[{{1, 2}, {3, 4}}]").unwrap(), "10");
    assert_eq!(
      interpret("Permanent[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "450"
    );
    assert_eq!(
      interpret(
        "Permanent[{{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}}]"
      )
      .unwrap(),
      "576"
    );
    assert_eq!(interpret("Permanent[{{5}}]").unwrap(), "5");
    assert_eq!(interpret("Permanent[IdentityMatrix[3]]").unwrap(), "1");
  }

  #[test]
  fn symbolic_matrix() {
    assert_eq!(
      interpret("Permanent[{{a, b}, {c, d}}]").unwrap(),
      "b*c + a*d"
    );
  }

  #[test]
  fn non_square_stays_unevaluated() {
    assert_eq!(
      interpret("Permanent[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "Permanent[{{1, 2, 3}, {4, 5, 6}}]"
    );
  }

  #[test]
  fn large_numeric_matrix_uses_ryser() {
    // Regression: the O(n!) permutation sum hung for n >= 12. Ryser's
    // O(2^n n^2) formula computes it instantly. Exact values match
    // wolframscript.
    assert_eq!(
      interpret("Permanent[Table[i + j, {i, 12}, {j, 12}]]").unwrap(),
      "4158410247782904833280"
    );
    // Exact rational entries stay exact.
    assert_eq!(
      interpret("Permanent[HilbertMatrix[4]]").unwrap(),
      "32547/224000"
    );
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

  #[test]
  fn inverse_singular_returns_unevaluated() {
    // Matches wolframscript: a singular matrix emits Inverse::sing and
    // returns the expression unevaluated, not a hard error.
    assert_eq!(
      interpret("Inverse[{{1, 0}, {0, 0}}]").unwrap(),
      "Inverse[{{1, 0}, {0, 0}}]"
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

  // Tr[array] for a rank >= 3 array sums the super-diagonal a[[i, i, ..., i]]
  // over i = 1..min(dimensions), producing a scalar. Previously Woxi only
  // contracted the first two levels and returned a list.

  #[test]
  fn trace_rank_three_cube() {
    assert_eq!(
      interpret("Tr[Array[a, {2, 2, 2}]]").unwrap(),
      "a[1, 1, 1] + a[2, 2, 2]"
    );
  }

  #[test]
  fn trace_rank_three_3x3x3() {
    assert_eq!(
      interpret("Tr[Array[a, {3, 3, 3}]]").unwrap(),
      "a[1, 1, 1] + a[2, 2, 2] + a[3, 3, 3]"
    );
  }

  #[test]
  fn trace_rank_three_non_cubic_uses_min_dimension() {
    // The diagonal runs i = 1..min(2, 3, 4) = 2.
    assert_eq!(
      interpret("Tr[Array[a, {2, 3, 4}]]").unwrap(),
      "a[1, 1, 1] + a[2, 2, 2]"
    );
  }

  #[test]
  fn trace_rank_four() {
    assert_eq!(
      interpret("Tr[Array[a, {2, 2, 2, 2}]]").unwrap(),
      "a[1, 1, 1, 1] + a[2, 2, 2, 2]"
    );
  }

  #[test]
  fn trace_rank_three_with_times() {
    assert_eq!(
      interpret("Tr[Array[a, {2, 2, 2}], Times]").unwrap(),
      "a[1, 1, 1]*a[2, 2, 2]"
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

  #[test]
  fn identity_rectangular_wide() {
    // IdentityMatrix[{m, n}] gives the m×n matrix with 1s on the leading
    // diagonal and 0s elsewhere.
    assert_eq!(
      interpret("IdentityMatrix[{2, 3}]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}}"
    );
  }

  #[test]
  fn identity_rectangular_tall() {
    assert_eq!(
      interpret("IdentityMatrix[{3, 2}]").unwrap(),
      "{{1, 0}, {0, 1}, {0, 0}}"
    );
  }

  #[test]
  fn identity_rectangular_square() {
    // {n, n} matches IdentityMatrix[n].
    assert_eq!(
      interpret("IdentityMatrix[{3, 3}]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
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

  #[test]
  fn inexact_diagonal_fills_with_real_zero() {
    // An inexact diagonal makes the whole matrix inexact: the off-diagonal
    // fill is the machine real 0., not the exact Integer 0.
    assert_eq!(
      interpret("DiagonalMatrix[{1., 2.}]").unwrap(),
      "{{1., 0.}, {0., 2.}}"
    );
    assert_eq!(
      interpret("Head[DiagonalMatrix[{1., 2.}][[1, 2]]]").unwrap(),
      "Real"
    );
  }

  #[test]
  fn mixed_diagonal_promotes_exact_numbers() {
    // A single inexact entry promotes the exact numeric entries to Real,
    // but leaves symbolic entries untouched.
    assert_eq!(
      interpret("DiagonalMatrix[{1, 2.}]").unwrap(),
      "{{1., 0.}, {0., 2.}}"
    );
    assert_eq!(
      interpret("DiagonalMatrix[{1/2, 2.}]").unwrap(),
      "{{0.5, 0.}, {0., 2.}}"
    );
    assert_eq!(
      interpret("DiagonalMatrix[{a, 2.}]").unwrap(),
      "{{a, 0.}, {0., 2.}}"
    );
  }

  #[test]
  fn inexact_diagonal_with_offset() {
    assert_eq!(
      interpret("DiagonalMatrix[{1., 2.}, 1]").unwrap(),
      "{{0., 1., 0.}, {0., 0., 2.}, {0., 0., 0.}}"
    );
  }

  #[test]
  fn exact_diagonal_stays_exact() {
    // A purely exact diagonal keeps Integer zeros.
    assert_eq!(
      interpret("DiagonalMatrix[{1, 2}]").unwrap(),
      "{{1, 0}, {0, 2}}"
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
      "FullForm[Hold[Cross[a, b, c]]]"
    );
  }

  #[test]
  fn cross_operator_inside_hold() {
    assert_eq!(
      interpret("FullForm[Hold[{1, 2, 3} \u{2A2F} {4, 5, 6}]]").unwrap(),
      "FullForm[Hold[Cross[{1, 2, 3}, {4, 5, 6}]]]"
    );
  }
}

// The legacy `VectorAnalysis` package functions — DotProduct, CrossProduct,
// ScalarTripleProduct, Coordinates, SetCoordinates, CoordinatesToCartesian,
// CoordinatesFromCartesian — are deliberately left unevaluated to match
// wolframscript's `-code` mode behaviour, which does not load that legacy
// package.
mod vector_analysis_unevaluated {
  use super::*;

  #[test]
  fn dot_product_stays_unevaluated() {
    assert_eq!(
      interpret("DotProduct[{1, 2, 3}, {4, 5, 6}]").unwrap(),
      "DotProduct[{1, 2, 3}, {4, 5, 6}]"
    );
  }

  #[test]
  fn cross_product_stays_unevaluated() {
    assert_eq!(
      interpret("CrossProduct[{1, 0, 0}, {0, 1, 0}]").unwrap(),
      "CrossProduct[{1, 0, 0}, {0, 1, 0}]"
    );
  }

  #[test]
  fn scalar_triple_product_stays_unevaluated() {
    assert_eq!(
      interpret("ScalarTripleProduct[{-2, 3, 1}, {0, 4, 0}, {-1, 3, 3}]")
        .unwrap(),
      "ScalarTripleProduct[{-2, 3, 1}, {0, 4, 0}, {-1, 3, 3}]"
    );
  }

  #[test]
  fn coordinates_stays_unevaluated() {
    assert_eq!(interpret("Coordinates[]").unwrap(), "Coordinates[]");
  }

  #[test]
  fn set_coordinates_stays_unevaluated() {
    assert_eq!(
      interpret("SetCoordinates[Spherical]").unwrap(),
      "SetCoordinates[Spherical]"
    );
  }

  #[test]
  fn coordinates_to_cartesian_stays_unevaluated() {
    assert_eq!(
      interpret("CoordinatesToCartesian[{2, Pi, 3}, Spherical]").unwrap(),
      "CoordinatesToCartesian[{2, Pi, 3}, Spherical]"
    );
  }

  #[test]
  fn coordinates_from_cartesian_stays_unevaluated() {
    assert_eq!(
      interpret("CoordinatesFromCartesian[{0, 0, -2}, Spherical]").unwrap(),
      "CoordinatesFromCartesian[{0, 0, -2}, Spherical]"
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

  // Projection uses the conjugate inner product, so for complex vectors
  // the formula is `(Conjugate[v].u / Conjugate[v].v) * v`. Regression
  // for mathics doctest `Projection[{3 + I, 2, 2 - I}, {2, 4, 5 I}]`.
  #[test]
  fn projection_complex_vectors() {
    assert_eq!(
      interpret("Projection[{3 + I, 2, 2 - I}, {2, 4, 5 I}]").unwrap(),
      "{2/5 - (16*I)/45, 4/5 - (32*I)/45, 8/9 + I}"
    );
  }

  #[test]
  fn projection_onto_complex_vector() {
    assert_eq!(
      interpret("Projection[{1, 2}, {1 + I, 2}]").unwrap(),
      "{1 + (2*I)/3, 5/3 - I/3}"
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

  // Complex (Hermitian) rotation-pattern matrices [[a, b], [-b, a]] with an
  // imaginary off-diagonal: the imaginary unit must reduce (previously the
  // results kept unevaluated `I^2`/double negatives, e.g. {2 - I^2, 2 + I^2}),
  // and the real eigenvalues are ordered like wolframscript.
  #[test]
  fn eigenvalues_2x2_complex_hermitian() {
    assert_eq!(interpret("Eigenvalues[PauliMatrix[2]]").unwrap(), "{-1, 1}");
    assert_eq!(
      interpret("Eigenvalues[{{0, I}, {-I, 0}}]").unwrap(),
      "{-1, 1}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{2, I}, {-I, 2}}]").unwrap(),
      "{3, 1}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{1, -I}, {I, 1}}]").unwrap(),
      "{2, 0}"
    );
    // A purely real rotation keeps its imaginary eigenvalues in a-I*b order.
    assert_eq!(
      interpret("Eigenvalues[{{0, -2}, {2, 0}}]").unwrap(),
      "{2*I, -2*I}"
    );
    // Genuinely symbolic off-diagonal is left in the closed a ± I*b form.
    assert_eq!(
      interpret("Eigenvalues[{{a, b}, {-b, a}}]").unwrap(),
      "{a - I*b, a + I*b}"
    );
  }

  // Complex Hermitian matrices with a *distinct* diagonal go through the generic
  // 2×2 path; their real eigenvalues must also be ordered like wolframscript
  // (decreasing magnitude), not left in the closed minus/plus branch order.
  #[test]
  fn eigenvalues_2x2_complex_hermitian_distinct_diagonal() {
    assert_eq!(
      interpret("Eigenvalues[{{2, 2 I}, {-2 I, 5}}]").unwrap(),
      "{6, 1}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{4, I}, {-I, 1}}]").unwrap(),
      "{(5 + Sqrt[13])/2, (5 - Sqrt[13])/2}"
    );
    // Genuinely complex eigenvalues keep the closed minus/plus order.
    assert_eq!(
      interpret("Eigenvalues[{{1, 2}, {-2, 1}}]").unwrap(),
      "{1 + 2*I, 1 - 2*I}"
    );
    // Fully symbolic matrices are unchanged.
    assert_eq!(
      interpret("Eigenvalues[{{a, b}, {c, d}}]").unwrap(),
      "{(a + d - Sqrt[a^2 + 4*b*c - 2*a*d + d^2])/2, \
       (a + d + Sqrt[a^2 + 4*b*c - 2*a*d + d^2])/2}"
    );
  }

  // Symbolic triangular/diagonal matrices: the eigenvalues are the diagonal
  // entries in matrix order (not the unsimplified quadratic formula).
  #[test]
  fn eigenvalues_symbolic_diagonal() {
    assert_eq!(
      interpret("Eigenvalues[{{a, 0}, {0, b}}]").unwrap(),
      "{a, b}"
    );
    assert_eq!(
      interpret("Eigenvalues[DiagonalMatrix[{x, y, z}]]").unwrap(),
      "{x, y, z}"
    );
  }

  #[test]
  fn eigenvalues_symbolic_triangular() {
    assert_eq!(
      interpret("Eigenvalues[{{a, b}, {0, d}}]").unwrap(),
      "{a, d}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{a, 0}, {c, d}}]").unwrap(),
      "{a, d}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{a, b, c}, {0, d, e}, {0, 0, f}}]").unwrap(),
      "{a, d, f}"
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
  fn eigenvalues_2x2_complex_pure_imaginary() {
    // Regression: the negative discriminant lost its sign and the rotation
    // matrix got real eigenvalues {-Sqrt[1], Sqrt[1]} instead of {I, -I}
    assert_eq!(
      interpret("Eigenvalues[{{0, -1}, {1, 0}}]").unwrap(),
      "{I, -I}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{0, -4}, {1, 0}}]").unwrap(),
      "{2*I, -2*I}"
    );
  }

  #[test]
  fn eigenvalues_2x2_complex_with_real_part() {
    assert_eq!(
      interpret("Eigenvalues[{{1, -1}, {1, 1}}]").unwrap(),
      "{1 + I, 1 - I}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{3, -2}, {4, -1}}]").unwrap(),
      "{1 + 2*I, 1 - 2*I}"
    );
  }

  #[test]
  fn eigenvalues_2x2_complex_irrational() {
    assert_eq!(
      interpret("Eigenvalues[{{1, -2}, {1, 1}}]").unwrap(),
      "{1 + I*Sqrt[2], 1 - I*Sqrt[2]}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{2, -5}, {1, 2}}]").unwrap(),
      "{2 + I*Sqrt[5], 2 - I*Sqrt[5]}"
    );
  }

  #[test]
  fn eigenvalues_2x2_complex_half_integer() {
    // Odd trace keeps the quotient form
    assert_eq!(
      interpret("Eigenvalues[{{2, -1}, {3, 1}}]").unwrap(),
      "{(3 + I*Sqrt[11])/2, (3 - I*Sqrt[11])/2}"
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

  #[test]
  fn eigenvalues_2x2_symbolic() {
    // For a fully symbolic 2x2 matrix, the closed-form eigenvalues are
    // (a + d ± Sqrt[(a - d)^2 + 4*b*c]) / 2.
    assert_eq!(
      interpret("Eigenvalues[{{a, b}, {c, d}}]").unwrap(),
      "{(a + d - Sqrt[a^2 + 4*b*c - 2*a*d + d^2])/2, (a + d + Sqrt[a^2 + 4*b*c - 2*a*d + d^2])/2}"
    );
  }

  #[test]
  fn eigenvalues_tiebreak_ascending() {
    // Eigenvalues with equal magnitude (±1 here) tiebreak ascending —
    // {2, -1, 1}, not {2, 1, -1}. Matches wolframscript.
    assert_eq!(
      interpret("Eigenvalues[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]").unwrap(),
      "{2, -1, 1}"
    );
  }

  // Eigenvalues[m, k] takes the k largest-magnitude eigenvalues (k < 0 takes
  // the smallest |k|), like Take.
  #[test]
  fn eigenvalues_count() {
    assert_eq!(
      interpret("Eigenvalues[{{2, 0, 0}, {0, 3, 0}, {0, 0, 1}}, 2]").unwrap(),
      "{3, 2}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{2, 0, 0}, {0, 3, 0}, {0, 0, 1}}, 1]").unwrap(),
      "{3}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{1, 2}, {3, 4}}, 1]").unwrap(),
      "{(5 + Sqrt[33])/2}"
    );
    assert_eq!(
      interpret("Eigenvalues[{{1, 2}, {3, 4}}, -1]").unwrap(),
      "{(5 - Sqrt[33])/2}"
    );
  }

  #[test]
  fn eigenvalues_count_overflow_emits_take() {
    assert_eq!(
      interpret("Eigenvalues[{{1, 2}, {3, 4}}, 3]").unwrap(),
      "Eigenvalues[{{1, 2}, {3, 4}}, 3]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Eigenvalues::take: Cannot take eigenvalues 1 through 3 out of the total of 2 eigenvalues."
      )),
      "expected take message, got {:?}",
      msgs
    );
  }

  #[test]
  fn diagonalization_identity() {
    // With T = Transpose[Eigenvectors[A]], Inverse[T] . A . T should equal
    // DiagonalMatrix[Eigenvalues[A]] — the textbook diagonalization
    // identity. Exercises Eigenvalues, Eigenvectors, Transpose, Inverse,
    // Dot, DiagonalMatrix and their ordering consistency.
    let expr = "A = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}; \
                T = Transpose[Eigenvectors[A]]; \
                Inverse[T] . A . T == DiagonalMatrix[Eigenvalues[A]]";
    assert_eq!(interpret(expr).unwrap(), "True");
  }

  #[test]
  fn eigenvalues_float_3x3_audit_case() {
    // Audit case: Eigenvalues of a 3x3 float matrix.
    // Expected (descending magnitude):
    //   {6.606744130165772, 4.525355330602706, 0.6679005392315176}
    let result = interpret(
      "Eigenvalues[{{1.1, 2.2, 3.25}, {0.76, 4.6, 5}, {0.1, 0.1, 6.1}}]",
    )
    .unwrap();
    let stripped = result.trim_start_matches('{').trim_end_matches('}');
    let vals: Vec<f64> = stripped
      .split(", ")
      .map(|s| s.parse::<f64>().unwrap())
      .collect();
    let expected = [6.606744130165772, 4.525355330602706, 0.6679005392315176];
    assert_eq!(vals.len(), 3);
    for (v, e) in vals.iter().zip(expected.iter()) {
      assert!((v - e).abs() < 1e-9, "got {} expected {}", v, e);
    }
  }

  #[test]
  fn eigenvalues_float_3x3_simple() {
    // Simpler diagonal-ish 3x3.
    let result = interpret(
      "Eigenvalues[{{2.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 5.0}}]",
    )
    .unwrap();
    // Sorted by descending magnitude: 5, 3, 2
    let stripped = result.trim_start_matches('{').trim_end_matches('}');
    let vals: Vec<f64> = stripped
      .split(", ")
      .map(|s| s.parse::<f64>().unwrap())
      .collect();
    assert_eq!(vals.len(), 3);
    assert!((vals[0] - 5.0).abs() < 1e-10);
    assert!((vals[1] - 3.0).abs() < 1e-10);
    assert!((vals[2] - 2.0).abs() < 1e-10);
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

// RankDecomposition[m] = {C, F} with C the pivot columns of m and F the
// nonzero rows of RowReduce[m], so that C.F == m. Verified vs wolframscript.
mod rank_decomposition {
  use super::*;

  #[test]
  fn rank_one() {
    assert_eq!(
      interpret("RankDecomposition[{{1, 2}, {2, 4}}]").unwrap(),
      "{{{1}, {2}}, {{1, 2}}}"
    );
  }

  #[test]
  fn rank_two_wide() {
    assert_eq!(
      interpret("RankDecomposition[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "{{{1, 2}, {4, 5}}, {{1, 0, -1}, {0, 1, 2}}}"
    );
  }

  #[test]
  fn rank_deficient_square() {
    assert_eq!(
      interpret("RankDecomposition[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]")
        .unwrap(),
      "{{{1, 2}, {4, 5}, {7, 8}}, {{1, 0, -1}, {0, 1, 2}}}"
    );
  }

  #[test]
  fn full_rank_returns_identity_factor() {
    assert_eq!(
      interpret("RankDecomposition[{{1, 2}, {3, 4}}]").unwrap(),
      "{{{1, 2}, {3, 4}}, {{1, 0}, {0, 1}}}"
    );
  }

  // Reconstruction: C.F recovers the original matrix.
  #[test]
  fn reconstructs_original() {
    assert_eq!(
      interpret("RankDecomposition[{{1, 2, 3}, {4, 5, 6}}] /. {c_, f_} :> c.f")
        .unwrap(),
      "{{1, 2, 3}, {4, 5, 6}}"
    );
  }

  // A rank-0 (all-zero) matrix has no decomposition; a non-matrix argument is
  // a usage error. Both stay unevaluated.
  #[test]
  fn unevaluated_forms() {
    assert_eq!(
      interpret("RankDecomposition[{{0, 0}, {0, 0}}]").unwrap(),
      "RankDecomposition[{{0, 0}, {0, 0}}]"
    );
    assert_eq!(
      interpret("RankDecomposition[{1, 2, 3}]").unwrap(),
      "RankDecomposition[{1, 2, 3}]"
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

  // Multiple basis vectors are emitted in order of decreasing free-variable
  // index, matching Wolfram.
  #[test]
  fn null_space_multiple_vectors_ordering() {
    assert_eq!(
      interpret("NullSpace[{{1, 1, 1}}]").unwrap(),
      "{{-1, 0, 1}, {-1, 1, 0}}"
    );
    assert_eq!(
      interpret("NullSpace[{{1, 2, 3, 4}}]").unwrap(),
      "{{-4, 0, 0, 1}, {-3, 0, 1, 0}, {-2, 1, 0, 0}}"
    );
    assert_eq!(
      interpret("NullSpace[{{1, 1, 0, 0}, {0, 0, 1, 1}}]").unwrap(),
      "{{0, 0, -1, 1}, {-1, 1, 0, 0}}"
    );
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

  // Operator form: LinearSolve[m][b] == LinearSolve[m, b], and it can be
  // mapped over several right-hand sides.
  #[test]
  fn solve_operator_form() {
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {3, 4}}][{5, 6}]").unwrap(),
      "{-4, 9/2}"
    );
    assert_eq!(
      interpret("Map[LinearSolve[{{1, 2}, {3, 4}}], {{5, 6}, {1, 0}}]")
        .unwrap(),
      "{{-4, 9/2}, {-2, 3/2}}"
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

  // A target whose length differs from the (square) coefficient matrix emits
  // LinearSolve::lslc and stays unevaluated, rather than leaking an error.
  #[test]
  fn dimension_mismatch_emits_lslc() {
    clear_state();
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {3, 4}}, {1, 2, 3}]").unwrap(),
      "LinearSolve[{{1, 2}, {3, 4}}, {1, 2, 3}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "LinearSolve::lslc: Coefficient matrix and target vector(s) or matrix do not have the same dimensions."
      )),
      "expected LinearSolve::lslc, got {msgs:?}"
    );
  }

  // A singular system with no solution emits LinearSolve::nosol.
  #[test]
  fn no_solution_emits_nosol() {
    clear_state();
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {2, 4}}, {1, 3}]").unwrap(),
      "LinearSolve[{{1, 2}, {2, 4}}, {1, 3}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "LinearSolve::nosol: Linear equation encountered that has no solution."
      )),
      "expected LinearSolve::nosol, got {msgs:?}"
    );
  }

  // A singular but consistent system still solves (free variable -> 0).
  #[test]
  fn singular_consistent_still_solves() {
    clear_state();
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {2, 4}}, {2, 4}]").unwrap(),
      "{2, 0}"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().all(|m| !m.contains("nosol")),
      "unexpected nosol message: {msgs:?}"
    );
  }

  // Rectangular (non-square) systems: a particular solution with the free
  // (non-pivot) variables set to 0, via the RREF of the augmented matrix.
  #[test]
  fn underdetermined_rectangular() {
    assert_eq!(
      interpret("LinearSolve[{{1, 2, 3}, {4, 5, 6}}, {1, 2}]").unwrap(),
      "{-1/3, 2/3, 0}"
    );
    assert_eq!(
      interpret("LinearSolve[{{1, 1, 1}, {1, 2, 3}}, {6, 14}]").unwrap(),
      "{-2, 8, 0}"
    );
  }

  // An overdetermined but consistent system solves; an inconsistent one emits
  // LinearSolve::nosol and stays unevaluated.
  #[test]
  fn overdetermined_rectangular() {
    assert_eq!(interpret("LinearSolve[{{1}, {2}}, {3, 6}]").unwrap(), "{3}");
    clear_state();
    assert_eq!(
      interpret("LinearSolve[{{1}, {1}}, {1, 2}]").unwrap(),
      "LinearSolve[{{1}, {1}}, {1, 2}]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains("LinearSolve::nosol")),
      "expected LinearSolve::nosol, got {msgs:?}"
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

  #[test]
  fn singular_consistent_returns_particular_solution() {
    // Regression: LinearSolve used to reject every singular matrix. For a
    // singular-but-consistent system it should return one particular
    // solution with free variables set to 0, matching wolframscript.
    assert_eq!(
      interpret("LinearSolve[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1, 1}]")
        .unwrap(),
      "{-1, 1, 0}"
    );
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {3, 6}}, {2, 6}]").unwrap(),
      "{2, 0}"
    );
  }
}

mod eigenvectors {
  use super::*;

  #[test]
  fn matrix_1x1() {
    assert_eq!(interpret("Eigenvectors[{{5}}]").unwrap(), "{{1}}");
  }

  // Eigenvectors[m, k] takes the eigenvectors for the k largest-magnitude
  // eigenvalues (k < 0 takes the smallest |k|).
  #[test]
  fn count_form() {
    assert_eq!(
      interpret("Eigenvectors[{{2, 0, 0}, {0, 3, 0}, {0, 0, 1}}, 2]").unwrap(),
      "{{0, 1, 0}, {1, 0, 0}}"
    );
    assert_eq!(
      interpret("Eigenvectors[{{1, 2}, {3, 4}}, 1]").unwrap(),
      "{{(-3 + Sqrt[33])/6, 1}}"
    );
    assert_eq!(
      interpret("Eigenvectors[{{1, 2}, {3, 4}}, -1]").unwrap(),
      "{{(-3 - Sqrt[33])/6, 1}}"
    );
  }

  #[test]
  fn count_overflow_emits_take() {
    assert_eq!(
      interpret("Eigenvectors[{{1, 2}, {3, 4}}, 3]").unwrap(),
      "Eigenvectors[{{1, 2}, {3, 4}}, 3]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains("Eigenvectors::take")),
      "expected take message, got {:?}",
      msgs
    );
  }

  #[test]
  fn diagonal_2x2() {
    assert_eq!(
      interpret("Eigenvectors[{{2, 0}, {0, 3}}]").unwrap(),
      "{{0, 1}, {1, 0}}"
    );
  }

  #[test]
  fn diagonal_3x3_with_zero_entry() {
    // A diagonal matrix's eigenvectors are the standard basis vectors.
    assert_eq!(
      interpret("Eigenvectors[{{2, 0, 0}, {0, -1, 0}, {0, 0, 0}}]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn rank_two_3x3_with_repeated_eigenvalue() {
    // Two 1-eigenvectors and one 0-eigenvector — wolframscript orders the
    // 1-eigenvectors as e2, e1 (reverse).
    assert_eq!(
      interpret("Eigenvectors[{{1, 0, 0}, {0, 1, 0}, {0, 0, 0}}]").unwrap(),
      "{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}}"
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

  #[test]
  fn cyclic_3x3_eigenvectors() {
    // Matches wolframscript: the eigenvector ordering follows the
    // Eigenvalues order {2, -1, 1}.
    assert_eq!(
      interpret("Eigenvectors[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]").unwrap(),
      "{{1, 1, 1}, {1, -2, 1}, {-1, 0, 1}}"
    );
  }

  #[test]
  fn rank_deficient_3x3_returns_three_vectors() {
    // {{1,2,3},{4,5,6},{7,8,9}}: rank-deficient integer matrix with
    // eigenvalues {(15+3 Sqrt[33])/2, (15-3 Sqrt[33])/2, 0}. Previously
    // returned unevaluated; now must return three eigenvectors with the
    // zero-eigenvalue kernel as `{1, -2, 1}`.
    let result =
      interpret("Eigenvectors[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap();
    assert!(
      !result.starts_with("Eigenvectors["),
      "still unevaluated: {result}"
    );
    assert!(
      result.ends_with("{1, -2, 1}}"),
      "missing zero-eigenvalue kernel {{1, -2, 1}}: {result}"
    );
  }

  #[test]
  fn rank_deficient_3x3_eigenvector_relation() {
    // For each eigenvector v_i with eigenvalue λ_i, A . v_i == λ_i v_i.
    // After Simplify, the residual A.v - λ v must vanish to {0, 0, 0}.
    let m = "{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}";
    let code = format!(
      "Module[{{m = {m}, vs, ls}}, vs = Eigenvectors[m]; ls = Eigenvalues[m]; Simplify[Table[m . vs[[i]] - ls[[i]] * vs[[i]], {{i, 1, 3}}]]]"
    );
    assert_eq!(
      interpret(&code).unwrap(),
      "{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}"
    );
  }
}

mod eigensystem {
  use super::*;

  #[test]
  fn cyclic_3x3() {
    // Eigensystem returns {eigenvalues, eigenvectors} in aligned order.
    assert_eq!(
      interpret("Eigensystem[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]").unwrap(),
      "{{2, -1, 1}, {{1, 1, 1}, {1, -2, 1}, {-1, 0, 1}}}"
    );
  }

  // Eigensystem[m, k] takes the k largest-magnitude pairs from each part.
  #[test]
  fn count_form() {
    assert_eq!(
      interpret("Eigensystem[{{2, 0, 0}, {0, 3, 0}, {0, 0, 1}}, 2]").unwrap(),
      "{{3, 2}, {{0, 1, 0}, {1, 0, 0}}}"
    );
    assert_eq!(
      interpret("Eigensystem[{{1, 2}, {3, 4}}, 1]").unwrap(),
      "{{(5 + Sqrt[33])/2}, {{(-3 + Sqrt[33])/6, 1}}}"
    );
  }

  #[test]
  fn count_overflow_emits_take() {
    assert_eq!(
      interpret("Eigensystem[{{1, 2}, {3, 4}}, 3]").unwrap(),
      "Eigensystem[{{1, 2}, {3, 4}}, 3]"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains("Eigensystem::take")),
      "expected take message, got {:?}",
      msgs
    );
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

  // Signature operates on any non-atomic head, not just List: it uses the
  // level-1 parts as the sequence. A held `Cycles[...]` has a single part, so
  // its signature is 1 (Signature does NOT interpret it as a permutation).
  #[test]
  fn general_head() {
    assert_eq!(interpret("Signature[f[2, 1]]").unwrap(), "-1");
    assert_eq!(interpret("Signature[f[3, 1, 2]]").unwrap(), "1");
    assert_eq!(interpret("Signature[f[b, a]]").unwrap(), "-1");
    assert_eq!(interpret("Signature[Cycles[{{1, 2, 3}}]]").unwrap(), "1");
    assert_eq!(interpret("Signature[Cycles[{{1, 2}}]]").unwrap(), "1");
  }

  // Inversions use the canonical (numeric) Order, not string comparison, so a
  // multi-digit element sorts by value: {10, 2} needs one swap → -1.
  #[test]
  fn numeric_ordering_not_string() {
    assert_eq!(interpret("Signature[{10, 2}]").unwrap(), "-1");
    assert_eq!(interpret("Signature[{2, 10, 3}]").unwrap(), "-1");
  }

  // 1 and 1.0 are distinct under canonical order, so they are not duplicates.
  #[test]
  fn int_and_real_not_duplicate() {
    assert_eq!(interpret("Signature[{1, 1.0}]").unwrap(), "1");
  }

  // An atomic argument emits Signature::normal and stays unevaluated.
  #[test]
  fn atomic_emits_normal() {
    assert_eq!(interpret("Signature[5]").unwrap(), "Signature[5]");
    let msgs = woxi::get_captured_messages_raw();
    assert!(
      msgs.iter().any(|m| m.contains(
        "Signature::normal: Nonatomic expression expected at position 1 in Signature[5]."
      )),
      "expected normal message, got {:?}",
      msgs
    );
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
  fn rank_deficient_square() {
    // Singular (rank-deficient) square matrices have an exact Moore-Penrose
    // pseudoinverse computed via rank factorization — no Inverse::sing error.
    assert_eq!(
      interpret("PseudoInverse[{{1, 0}, {0, 0}}]").unwrap(),
      "{{1, 0}, {0, 0}}"
    );
    assert_eq!(
      interpret("PseudoInverse[{{1, 2}, {2, 4}}]").unwrap(),
      "{{1/25, 2/25}, {2/25, 4/25}}"
    );
    assert_eq!(
      interpret("PseudoInverse[{{1, 1}, {1, 1}}]").unwrap(),
      "{{1/4, 1/4}, {1/4, 1/4}}"
    );
  }

  #[test]
  fn rank_deficient_rectangular() {
    assert_eq!(
      interpret("PseudoInverse[{{1, 2, 3}, {2, 4, 6}}]").unwrap(),
      "{{1/70, 1/35}, {1/35, 2/35}, {3/70, 3/35}}"
    );
  }

  #[test]
  fn rank_deficient_no_singular_warning() {
    use woxi::interpret_with_stdout;
    // The internal singular probes must not leak an Inverse::sing message.
    let r = interpret_with_stdout("PseudoInverse[{{1, 0}, {0, 0}}]").unwrap();
    assert_eq!(r.result, "{{1, 0}, {0, 0}}");
    assert!(r.warnings.is_empty());
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

// DrazinInverse[m] = A^D, the unique matrix with A^D A A^D = A^D, A A^D = A^D A,
// and A^(k+1) A^D = A^k (k = matrix index). Verified against wolframscript 15.0.
mod drazin_inverse {
  use super::*;

  // An invertible matrix reduces to the ordinary inverse (index 0).
  #[test]
  fn invertible_is_inverse() {
    assert_eq!(
      interpret("DrazinInverse[{{1, 2}, {3, 4}}]").unwrap(),
      "{{-2, 1}, {3/2, -1/2}}"
    );
    assert_eq!(
      interpret("DrazinInverse[{{2, 0}, {0, 3}}]").unwrap(),
      "{{1/2, 0}, {0, 1/3}}"
    );
    assert_eq!(interpret("DrazinInverse[{{2}}]").unwrap(), "{{1/2}}");
  }

  // A nilpotent matrix has the zero matrix as its Drazin inverse.
  #[test]
  fn nilpotent_is_zero() {
    assert_eq!(
      interpret("DrazinInverse[{{0, 1}, {0, 0}}]").unwrap(),
      "{{0, 0}, {0, 0}}"
    );
    assert_eq!(interpret("DrazinInverse[{{0}}]").unwrap(), "{{0}}");
  }

  // Index-1 singular matrices: the core is inverted, the nilpotent part zeroed.
  #[test]
  fn index_one_singular() {
    // A projection is its own Drazin inverse.
    assert_eq!(
      interpret("DrazinInverse[{{1, 0}, {0, 0}}]").unwrap(),
      "{{1, 0}, {0, 0}}"
    );
    // Core and nilpotent parts are not axis-aligned here.
    assert_eq!(
      interpret("DrazinInverse[{{2, 1}, {0, 0}}]").unwrap(),
      "{{1/2, 1/4}, {0, 0}}"
    );
    assert_eq!(
      interpret("DrazinInverse[{{4, 0, 0}, {0, 0, 0}, {0, 0, 2}}]").unwrap(),
      "{{1/4, 0, 0}, {0, 0, 0}, {0, 0, 1/2}}"
    );
  }

  // Index-2 matrix with a nonzero core eigenvalue and a size-2 nilpotent block.
  #[test]
  fn index_two() {
    assert_eq!(
      interpret("DrazinInverse[{{2, 0, 0}, {0, 0, 1}, {0, 0, 0}}]").unwrap(),
      "{{1/2, 0, 0}, {0, 0, 0}, {0, 0, 0}}"
    );
  }

  // Symbolic entries are handled (generic rank).
  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("DrazinInverse[{{a, 0}, {0, 0}}]").unwrap(),
      "{{a^(-1), 0}, {0, 0}}"
    );
  }

  // A non-square (or non-matrix) argument stays unevaluated.
  #[test]
  fn non_square_unevaluated() {
    assert_eq!(
      interpret("DrazinInverse[{{1, 2, 3}, {4, 5, 6}}]").unwrap(),
      "DrazinInverse[{{1, 2, 3}, {4, 5, 6}}]"
    );
    assert_eq!(
      interpret("DrazinInverse[{1, 2, 3}]").unwrap(),
      "DrazinInverse[{1, 2, 3}]"
    );
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

  #[test]
  fn two_by_two_fully_symbolic() {
    // Indexed symbols a_ij and b_ij — matches wolframscript byte-for-byte.
    assert_eq!(
      interpret(
        "KroneckerProduct[\
           {{a11, a12}, {a21, a22}}, {{b11, b12}, {b21, b22}}\
         ]"
      )
      .unwrap(),
      "{{a11*b11, a11*b12, a12*b11, a12*b12}, \
        {a11*b21, a11*b22, a12*b21, a12*b22}, \
        {a21*b11, a21*b12, a22*b11, a22*b12}, \
        {a21*b21, a21*b22, a22*b21, a22*b22}}"
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

  // Wolfram 15 returns the separate L and U factors and the row permutation as
  // structured-array objects, followed by the ∞-norm condition number.

  #[test]
  fn basic_2x2() {
    assert_eq!(
      interpret("LUDecomposition[{{1, 2}, {3, 4}}]").unwrap(),
      "{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, \
       {{1, 0}, {3, 1}}]], UpperTriangularMatrix[StructuredArray`\
       StructuredData[{2, 2}, {{1, 2}, {0, -2}}]], \
       PermutationMatrix[StructuredArray`StructuredData[{2, 2}, \
       {Cycles[{}], Infinity}]], 0}"
    );
  }

  #[test]
  fn basic_3x3() {
    assert_eq!(
      interpret("LUDecomposition[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}]").unwrap(),
      "{LowerTriangularMatrix[StructuredArray`StructuredData[{3, 3}, \
       {{1, 0, 0}, {4, 1, 0}, {7, 2, 1}}]], \
       UpperTriangularMatrix[StructuredArray`StructuredData[{3, 3}, \
       {{1, 2, 3}, {0, -3, -6}, {0, 0, 1}}]], \
       PermutationMatrix[StructuredArray`StructuredData[{3, 3}, \
       {Cycles[{}], Infinity}]], 0}"
    );
  }

  #[test]
  fn with_pivoting() {
    assert_eq!(
      interpret("LUDecomposition[{{0, 1}, {1, 0}}]").unwrap(),
      "{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, \
       {{1, 0}, {0, 1}}]], UpperTriangularMatrix[StructuredArray`\
       StructuredData[{2, 2}, {{1, 0}, {0, 1}}]], \
       PermutationMatrix[StructuredArray`StructuredData[{2, 2}, \
       {Cycles[{{1, 2}}], Infinity}]], 0}"
    );
  }

  // A machine matrix uses magnitude-based partial pivoting and reports the
  // ∞-norm condition number ‖A‖∞·‖A⁻¹‖∞ = 4·0.8 = 3.2 as the last element.
  #[test]
  fn numeric_pivoting_and_condition_number() {
    assert_eq!(
      interpret("LUDecomposition[{{1., 3.}, {2., 1.}}]").unwrap(),
      "{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, \
       {{1., 0.}, {0.5, 1.}}]], UpperTriangularMatrix[StructuredArray`\
       StructuredData[{2, 2}, {{2., 1.}, {0., 2.5}}]], \
       PermutationMatrix[StructuredArray`StructuredData[{2, 2}, \
       {Cycles[{{1, 2}}], Infinity}]], 3.2}"
    );
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("LUDecomposition[{{a, b}, {c, d}}]").unwrap(),
      "{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, \
       {{1, 0}, {c/a, 1}}]], UpperTriangularMatrix[StructuredArray`\
       StructuredData[{2, 2}, {{a, b}, {0, -((b*c)/a) + d}}]], \
       PermutationMatrix[StructuredArray`StructuredData[{2, 2}, \
       {Cycles[{}], Infinity}]], 0}"
    );
  }

  #[test]
  fn identity() {
    assert_eq!(
      interpret("LUDecomposition[{{1, 0}, {0, 1}}]").unwrap(),
      "{LowerTriangularMatrix[StructuredArray`StructuredData[{2, 2}, \
       {{1, 0}, {0, 1}}]], UpperTriangularMatrix[StructuredArray`\
       StructuredData[{2, 2}, {{1, 0}, {0, 1}}]], \
       PermutationMatrix[StructuredArray`StructuredData[{2, 2}, \
       {Cycles[{}], Infinity}]], 0}"
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

mod vector_order {
  use super::*;

  // VectorGreater[{u, v}] is the element-wise strict comparison, And-reduced.
  #[test]
  fn pairwise() {
    assert_eq!(
      interpret("VectorGreater[{{2, 3}, {1, 2}}]").unwrap(),
      "True"
    );
    // One failing component makes the whole comparison False.
    assert_eq!(
      interpret("VectorGreater[{{2, 3}, {1, 5}}]").unwrap(),
      "False"
    );
    // Strict comparison: equal components are not greater.
    assert_eq!(
      interpret("VectorGreater[{{2, 3}, {2, 3}}]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("VectorGreaterEqual[{{2, 2}, {1, 2}}]").unwrap(),
      "True"
    );
    assert_eq!(interpret("VectorLess[{{1, 2}, {3, 4}}]").unwrap(), "True");
    assert_eq!(
      interpret("VectorLessEqual[{{1, 2}, {1, 3}}]").unwrap(),
      "True"
    );
  }

  // More than two vectors chains the comparison v1 > v2 > ... > vn.
  #[test]
  fn chained() {
    assert_eq!(
      interpret("VectorGreater[{{3, 4}, {2, 3}, {1, 2}}]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("VectorGreater[{{3, 4}, {2, 5}, {1, 2}}]").unwrap(),
      "False"
    );
  }

  // Symbolic comparisons stay unevaluated; mismatched lengths are False.
  #[test]
  fn symbolic_and_mismatched() {
    assert_eq!(
      interpret("VectorGreater[{{a, b}, {c, d}}]").unwrap(),
      "VectorGreater[{{a, b}, {c, d}}]"
    );
    assert_eq!(
      interpret("VectorGreater[{{2, 3, 1}, {1, 2}}]").unwrap(),
      "False"
    );
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
  fn creates_transformation_function_with_symbolic_entries() {
    assert_eq!(
      interpret("TranslationTransform[{x0, y0}]").unwrap(),
      "TransformationFunction[{{1, 0, x0}, {0, 1, y0}, {0, 0, 1}}]"
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

  // Two-vector form RotationTransform[{u, v}] (2D): rotation taking the
  // direction of u to the direction of v. Only the directions matter, not
  // the magnitudes.
  #[test]
  fn rotation_two_vectors_quarter_turn() {
    assert_eq!(
      interpret("RotationTransform[{{1, 0}, {0, 1}}][{1, 0}]").unwrap(),
      "{0, 1}"
    );
  }

  #[test]
  fn rotation_two_vectors_diagonal() {
    assert_eq!(
      interpret("RotationTransform[{{1, 0}, {1, 1}}][{1, 0}]").unwrap(),
      "{1/Sqrt[2], 1/Sqrt[2]}"
    );
  }

  #[test]
  fn rotation_two_vectors_ignores_magnitude() {
    // u = {2,0} (dir {1,0}), v = {0,3} (dir {0,1}) → 90° rotation.
    assert_eq!(
      interpret("RotationTransform[{{2, 0}, {0, 3}}][{1, 1}]").unwrap(),
      "{-1, 1}"
    );
  }

  #[test]
  fn rotation_two_vectors_matrix() {
    assert_eq!(
      interpret("TransformationMatrix[RotationTransform[{{1, 0}, {1, 1}}]]")
        .unwrap(),
      "{{1/Sqrt[2], -(1/Sqrt[2]), 0}, {1/Sqrt[2], 1/Sqrt[2], 0}, {0, 0, 1}}"
    );
  }
}

// TransformationMatrix extracts the homogeneous matrix from a
// TransformationFunction; other arguments stay unevaluated.
mod transformation_matrix {
  use super::*;

  #[test]
  fn from_rotation() {
    assert_eq!(
      interpret("TransformationMatrix[RotationTransform[Pi/2]]").unwrap(),
      "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn from_scaling() {
    assert_eq!(
      interpret("TransformationMatrix[ScalingTransform[{2, 3}]]").unwrap(),
      "{{2, 0, 0}, {0, 3, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn from_translation() {
    assert_eq!(
      interpret("TransformationMatrix[TranslationTransform[{1, 2}]]").unwrap(),
      "{{1, 0, 1}, {0, 1, 2}, {0, 0, 1}}"
    );
  }

  #[test]
  fn from_reflection() {
    assert_eq!(
      interpret("TransformationMatrix[ReflectionTransform[{1, 0}]]").unwrap(),
      "{{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn non_transform_stays_unevaluated() {
    assert_eq!(
      interpret("TransformationMatrix[5]").unwrap(),
      "TransformationMatrix[5]"
    );
    // A plain matrix is not a TransformationFunction.
    assert_eq!(
      interpret("TransformationMatrix[{{1, 2}, {3, 4}}]").unwrap(),
      "TransformationMatrix[{{1, 2}, {3, 4}}]"
    );
  }
}

// ReflectionTransform[v] reflects in the hyperplane through the origin
// perpendicular to v: linear part I - 2 (v⊗v)/(v·v) in a homogeneous matrix.
mod reflection_transform {
  use super::*;

  #[test]
  fn axis_aligned() {
    assert_eq!(
      interpret("ReflectionTransform[{1, 0}]").unwrap(),
      "TransformationFunction[{{-1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]"
    );
    assert_eq!(
      interpret("ReflectionTransform[{0, 1}]").unwrap(),
      "TransformationFunction[{{1, 0, 0}, {0, -1, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn diagonal_vector() {
    assert_eq!(
      interpret("ReflectionTransform[{1, 1}]").unwrap(),
      "TransformationFunction[{{0, -1, 0}, {-1, 0, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn rational_entries() {
    assert_eq!(
      interpret("ReflectionTransform[{3, 4}]").unwrap(),
      "TransformationFunction[{{7/25, -24/25, 0}, {-24/25, -7/25, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn three_dimensional() {
    assert_eq!(
      interpret("ReflectionTransform[{1, 0, 0}]").unwrap(),
      "TransformationFunction[{{-1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}]"
    );
  }

  #[test]
  fn through_point() {
    // Reflection in the hyperplane perpendicular to v through pt.
    assert_eq!(
      interpret("ReflectionTransform[{1, 0}, {2, 3}]").unwrap(),
      "TransformationFunction[{{-1, 0, 4}, {0, 1, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn applied_to_point() {
    assert_eq!(
      interpret("ReflectionTransform[{1, 1}][{2, 3}]").unwrap(),
      "{-3, -2}"
    );
    assert_eq!(
      interpret("ReflectionTransform[{0, 1}][{4, 5}]").unwrap(),
      "{4, -5}"
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

  #[test]
  fn tensor_contract_single_pair() {
    // TensorContract[T, {{1, 3}}] reduces the rank by 2.
    assert_eq!(
      interpret("TensorRank[TensorContract[T, {{1, 3}}]]").unwrap(),
      "-2 + TensorRank[T]"
    );
  }

  #[test]
  fn tensor_contract_two_pairs() {
    assert_eq!(
      interpret("TensorRank[TensorContract[T, {{1, 3}, {2, 4}}]]").unwrap(),
      "-4 + TensorRank[T]"
    );
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

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn array_depth_1() {
    assert_case(r#"ArrayDepth[{{a,b},{c,d}}]"#, r#"2"#);
  }
  #[test]
  fn array_depth_2() {
    assert_case(r#"ArrayDepth[{{a,b},{c,d}}]; ArrayDepth[x]"#, r#"0"#);
  }
  #[test]
  fn dimensions_1() {
    assert_case(r#"Dimensions[{a, b, c}]"#, r#"{3}"#);
  }
  #[test]
  fn dimensions_2() {
    assert_case(
      r#"Dimensions[{a, b, c}]; Dimensions[{{a, b}, {c, d}, {e, f}}]"#,
      r#"{3, 2}"#,
    );
  }
  #[test]
  fn dimensions_3() {
    assert_case(
      r#"Dimensions[{a, b, c}]; Dimensions[{{a, b}, {c, d}, {e, f}}]; Dimensions[{{a, b}, {b, c}, {c, d, e}}]"#,
      r#"{3}"#,
    );
  }
  #[test]
  fn dimensions_4() {
    assert_case(
      r#"Dimensions[{a, b, c}]; Dimensions[{{a, b}, {c, d}, {e, f}}]; Dimensions[{{a, b}, {b, c}, {c, d, e}}]; Dimensions[f[f[a, b, c]]]"#,
      r#"{1, 3}"#,
    );
  }
  #[test]
  fn list_literal_1() {
    assert_case(r#"{a, b, c} . {x, y, z}"#, r#"a*x + b*y + c*z"#);
  }
  #[test]
  fn list_literal_2() {
    assert_case(
      r#"{a, b, c} . {x, y, z}; {{a, b}, {c, d}} . {x, y}"#,
      r#"{a*x + b*y, c*x + d*y}"#,
    );
  }
  #[test]
  fn list_literal_3() {
    assert_case(
      r#"{a, b, c} . {x, y, z}; {{a, b}, {c, d}} . {x, y}; {{a, b}, {c, d}} . {{r, s}, {t, u}}"#,
      r#"{{a*r + b*t, a*s + b*u}, {c*r + d*t, c*s + d*u}}"#,
    );
  }
  #[test]
  fn expression() {
    assert_case(
      r#"{a, b, c} . {x, y, z}; {{a, b}, {c, d}} . {x, y}; {{a, b}, {c, d}} . {{r, s}, {t, u}}; a . b"#,
      r#"a . b"#,
    );
  }
  #[test]
  fn inner_1() {
    assert_case(r#"Inner[f, {a, b}, {x, y}, g]"#, r#"g[f[a, x], f[b, y]]"#);
  }
  #[test]
  fn inner_2() {
    assert_case(
      r#"Inner[f, {a, b}, {x, y}, g]; Inner[Times, {a, b}, {c, d}, Plus] == {a, b} . {c, d}"#,
      r#"True"#,
    );
  }
  #[test]
  fn inner_3() {
    assert_case(
      r#"Inner[f, {a, b}, {x, y}, g]; Inner[Times, {a, b}, {c, d}, Plus] == {a, b} . {c, d}; Inner[And, {{False, False}, {False, True}}, {{True, False}, {True, True}}, Or]"#,
      r#"{{False, False}, {True, True}}"#,
    );
  }
  #[test]
  fn inner_4() {
    assert_case(
      r#"Inner[f, {a, b}, {x, y}, g]; Inner[Times, {a, b}, {c, d}, Plus] == {a, b} . {c, d}; Inner[And, {{False, False}, {False, True}}, {{True, False}, {True, True}}, Or]; Inner[f, {{{a, b}}, {{c, d}}}, {{1}, {2}}, g]"#,
      r#"{{{g[f[a, 1], f[b, 2]]}}, {{g[f[c, 1], f[d, 2]]}}}"#,
    );
  }
  #[test]
  fn outer_1() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]"#,
      r#"{{f[a, 1], f[a, 2], f[a, 3]}, {f[b, 1], f[b, 2], f[b, 3]}}"#,
    );
  }
  #[test]
  fn outer_2() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]"#,
      r#"{{{{a, 2*a}, {3*a, 4*a}}, {{b, 2*b}, {3*b, 4*b}}}, {{{c, 2*c}, {3*c, 4*c}}, {{d, 2*d}, {3*d, 4*d}}}}"#,
    );
  }
  #[test]
  fn outer_3() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]"#,
      r#"SparseArray[Automatic, {2, 2, 2, 2}, 0, {1, {{0, 2, 4}, {{2, 1, 2}, {2, 2, 1}, {1, 1, 2}, {1, 2, 1}}}, {a*c, a*d, b*c, b*d}}]"#,
    );
  }
  #[test]
  fn outer_4() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[f, {a, b}, {x, y, z}, {1, 2}]"#,
      r#"{{{f[a, x, 1], f[a, x, 2]}, {f[a, y, 1], f[a, y, 2]}, {f[a, z, 1], f[a, z, 2]}}, {{f[b, x, 1], f[b, x, 2]}, {f[b, y, 1], f[b, y, 2]}, {f[b, z, 1], f[b, z, 2]}}}"#,
    );
  }
  #[test]
  fn outer_5() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[f, {a, b}, {x, y, z}, {1, 2}]; Outer[f, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]"#,
      r#"{{SparseArray[Automatic, {2, 2}, f[0, 0], {1, {{0, 1, 2}, {{2}, {1}}}, {f[0, c], f[0, d]}}], SparseArray[Automatic, {2, 2}, f[a, 0], {1, {{0, 1, 2}, {{2}, {1}}}, {f[a, c], f[a, d]}}]}, {SparseArray[Automatic, {2, 2}, f[b, 0], {1, {{0, 1, 2}, {{2}, {1}}}, {f[b, c], f[b, d]}}], SparseArray[Automatic, {2, 2}, f[0, 0], {1, {{0, 1, 2}, {{2}, {1}}}, {f[0, c], f[0, d]}}]}}"#,
    );
  }
  #[test]
  fn outer_6() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[f, {a, b}, {x, y, z}, {1, 2}]; Outer[f, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], {c, d}]"#,
      r#"SparseArray[Automatic, {2, 2, 2}, 0, {1, {{0, 2, 4}, {{2, 1}, {2, 2}, {1, 1}, {1, 2}}}, {a*c, a*d, b*c, b*d}}]"#,
    );
  }
  #[test]
  fn outer_7() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[f, {a, b}, {x, y, z}, {1, 2}]; Outer[f, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], {c, d}]; Outer[Times, {{1, 2}}, {{a, b}, {c, d, e}}]"#,
      r#"{{{{a, b}, {c, d, e}}, {{2*a, 2*b}, {2*c, 2*d, 2*e}}}}"#,
    );
  }
  #[test]
  fn outer_8() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[f, {a, b}, {x, y, z}, {1, 2}]; Outer[f, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], {c, d}]; Outer[Times, {{1, 2}}, {{a, b}, {c, d, e}}]; Outer[StringJoin, {"", "re", "un"}, {"cover", "draw", "wind"}, {"", "ing", "s"}] // InputForm"#,
      r#"InputForm[{{{"cover", "covering", "covers"}, {"draw", "drawing", "draws"}, {"wind", "winding", "winds"}}, {{"recover", "recovering", "recovers"}, {"redraw", "redrawing", "redraws"}, {"rewind", "rewinding", "rewinds"}}, {{"uncover", "uncovering", "uncovers"}, {"undraw", "undrawing", "undraws"}, {"unwind", "unwinding", "unwinds"}}}]"#,
    );
  }
  #[test]
  fn set_1() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[f, {a, b}, {x, y, z}, {1, 2}]; Outer[f, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], {c, d}]; Outer[Times, {{1, 2}}, {{a, b}, {c, d, e}}]; Outer[StringJoin, {"", "re", "un"}, {"cover", "draw", "wind"}, {"", "ing", "s"}] // InputForm; trigs = Outer[Composition, {Sin, Cos, Tan}, {ArcSin, ArcCos, ArcTan}]"#,
      r#"{{Sin @* ArcSin, Sin @* ArcCos, Sin @* ArcTan}, {Cos @* ArcSin, Cos @* ArcCos, Cos @* ArcTan}, {Tan @* ArcSin, Tan @* ArcCos, Tan @* ArcTan}}"#,
    );
  }
  #[test]
  fn map() {
    assert_case(
      r#"Outer[f, {a, b}, {1, 2, 3}]; Outer[Times, {{a, b}, {c, d}}, {{1, 2}, {3, 4}}]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[f, {a, b}, {x, y, z}, {1, 2}]; Outer[f, SparseArray[{{1, 2} -> a, {2, 1} -> b}], SparseArray[{{1, 2} -> c, {2, 1} -> d}]]; Outer[Times, SparseArray[{{1, 2} -> a, {2, 1} -> b}], {c, d}]; Outer[Times, {{1, 2}}, {{a, b}, {c, d, e}}]; Outer[StringJoin, {"", "re", "un"}, {"cover", "draw", "wind"}, {"", "ing", "s"}] // InputForm; trigs = Outer[Composition, {Sin, Cos, Tan}, {ArcSin, ArcCos, ArcTan}]; Map[#[0] &, trigs, {2}]"#,
      r#"{{0, 1, 0}, {1, 0, 1}, {0, ComplexInfinity, 0}}"#,
    );
  }
  #[test]
  fn transpose_1() {
    assert_case(
      r#"square = {{1, 2, 3}, {4, 5, 6}}; Transpose[square]"#,
      r#"{{1, 4}, {2, 5}, {3, 6}}"#,
    );
  }
  #[test]
  fn matrix_form_1() {
    assert_case(
      r#"square = {{1, 2, 3}, {4, 5, 6}}; Transpose[square]; MatrixForm[%]"#,
      r#"MatrixForm[Out[0]]"#,
    );
  }
  #[test]
  fn matrix_form_2() {
    assert_case(
      r#"square = {{1, 2, 3}, {4, 5, 6}}; Transpose[square]; MatrixForm[%]; matrix = {{1, 2}, {3, 4}, {5, 6}}; MatrixForm[Transpose[matrix]]"#,
      r#"MatrixForm[{{1, 3, 5}, {2, 4, 6}}]"#,
    );
  }
  #[test]
  fn transpose_2() {
    assert_case(
      r#"square = {{1, 2, 3}, {4, 5, 6}}; Transpose[square]; MatrixForm[%]; matrix = {{1, 2}, {3, 4}, {5, 6}}; MatrixForm[Transpose[matrix]]; matrix3D = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}; Transpose[matrix3D]"#,
      r#"{{{1, 2}, {5, 6}}, {{3, 4}, {7, 8}}}"#,
    );
  }
  #[test]
  fn transpose_3() {
    assert_case(
      r#"square = {{1, 2, 3}, {4, 5, 6}}; Transpose[square]; MatrixForm[%]; matrix = {{1, 2}, {3, 4}, {5, 6}}; MatrixForm[Transpose[matrix]]; matrix3D = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}; Transpose[matrix3D]; Transpose[Transpose[matrix]] == matrix"#,
      r#"True"#,
    );
  }
  #[test]
  fn transpose_4() {
    assert_case(
      r#"square = {{1, 2, 3}, {4, 5, 6}}; Transpose[square]; MatrixForm[%]; matrix = {{1, 2}, {3, 4}, {5, 6}}; MatrixForm[Transpose[matrix]]; matrix3D = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}; Transpose[matrix3D]; Transpose[Transpose[matrix]] == matrix; Transpose[Transpose[matrix3D]] == matrix3D"#,
      r#"True"#,
    );
  }
  #[test]
  fn transpose_5() {
    assert_case(
      r#"square = {{1, 2, 3}, {4, 5, 6}}; Transpose[square]; MatrixForm[%]; matrix = {{1, 2}, {3, 4}, {5, 6}}; MatrixForm[Transpose[matrix]]; matrix3D = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}; Transpose[matrix3D]; Transpose[Transpose[matrix]] == matrix; Transpose[Transpose[matrix3D]] == matrix3D; Transpose[{}]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn transpose_6() {
    assert_case(
      r#"square = {{1, 2, 3}, {4, 5, 6}}; Transpose[square]; MatrixForm[%]; matrix = {{1, 2}, {3, 4}, {5, 6}}; MatrixForm[Transpose[matrix]]; matrix3D = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}; Transpose[matrix3D]; Transpose[Transpose[matrix]] == matrix; Transpose[Transpose[matrix3D]] == matrix3D; Transpose[{}]; Transpose[{a, b, c}]"#,
      r#"{a, b, c}"#,
    );
  }
  #[test]
  fn conjugate_transpose_1() {
    assert_case(
      r#"ConjugateTranspose[{{0, I}, {0, 0}}]"#,
      r#"{{0, 0}, {-I, 0}}"#,
    );
  }
  #[test]
  fn conjugate_transpose_2() {
    assert_case(
      r#"ConjugateTranspose[{{0, I}, {0, 0}}]; ConjugateTranspose[{{1, 2 I, 3}, {3 + 4 I, 5, I}}]"#,
      r#"{{1, 3 - 4*I}, {-2*I, 5}, {3, -I}}"#,
    );
  }
  #[test]
  fn levi_civita_tensor_1() {
    assert_case(
      r#"LeviCivitaTensor[3]"#,
      r#"SparseArray[Automatic, {3, 3, 3}, 0, {1, {{0, 2, 4, 6}, {{2, 3}, {3, 2}, {1, 3}, {3, 1}, {1, 2}, {2, 1}}}, {1, -1, -1, 1, 1, -1}}]"#,
    );
  }
  #[test]
  fn levi_civita_tensor_2() {
    assert_case(
      r#"LeviCivitaTensor[3]; LeviCivitaTensor[3, List]"#,
      r#"{{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}, {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}}, {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}}"#,
    );
  }
  #[test]
  fn sparse_array_1() {
    assert_case(
      r#"SparseArray[{{1, 2} -> 1, {2, 1} -> 1}]"#,
      r#"SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, {1, 1}}]"#,
    );
  }
  #[test]
  fn sparse_array_2() {
    assert_case(
      r#"SparseArray[{{1, 2} -> 1, {2, 1} -> 1}]; SparseArray[{{1, 2} -> 1, {2, 1} -> 1}, {3, 3}]"#,
      r#"SparseArray[Automatic, {3, 3}, 0, {1, {{0, 1, 2, 2}, {{2}, {1}}}, {1, 1}}]"#,
    );
  }
  #[test]
  fn set_2() {
    assert_case(
      r#"SparseArray[{{1, 2} -> 1, {2, 1} -> 1}]; SparseArray[{{1, 2} -> 1, {2, 1} -> 1}, {3, 3}]; M=SparseArray[{{0, a}, {b, 0}}]"#,
      r#"SparseArray[Automatic, {2, 2}, 0, {1, {{0, 1, 2}, {{2}, {1}}}, {a, b}}]"#,
    );
  }
  #[test]
  fn divide() {
    assert_case(
      r#"SparseArray[{{1, 2} -> 1, {2, 1} -> 1}]; SparseArray[{{1, 2} -> 1, {2, 1} -> 1}, {3, 3}]; M=SparseArray[{{0, a}, {b, 0}}]; M //Normal"#,
      r#"{{0, a}, {b, 0}}"#,
    );
  }
  #[test]
  fn det_1() {
    assert_case(r#"Det[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]"#, r#"-2"#);
  }
  #[test]
  fn det_2() {
    assert_case(
      r#"Det[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]; Det[{{a, b, c}, {d, e, f}, {g, h, i}}]"#,
      r#"-(c*e*g) + b*f*g + c*d*h - a*f*h - b*d*i + a*e*i"#,
    );
  }
  #[test]
  fn eigensystem() {
    assert_case(
      r#"Eigensystem[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]"#,
      r#"{{2, -1, 1}, {{1, 1, 1}, {1, -2, 1}, {-1, 0, 1}}}"#,
    );
  }
  #[test]
  fn eigenvalues_1() {
    assert_case(
      r#"Eigenvalues[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]"#,
      r#"{2, -1, 1}"#,
    );
  }
  #[test]
  fn eigenvalues_2() {
    assert_case(
      r#"Eigenvalues[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]; Eigenvalues[{{Cos[theta],Sin[theta],0},{-Sin[theta],Cos[theta],0},{0,0,1}}] // Sort"#,
      r#"{1, Cos[theta] - I*Sin[theta], Cos[theta] + I*Sin[theta]}"#,
    );
  }
  #[test]
  fn eigenvalues_3() {
    assert_case(
      r#"Eigenvalues[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]; Eigenvalues[{{Cos[theta],Sin[theta],0},{-Sin[theta],Cos[theta],0},{0,0,1}}] // Sort; Eigenvalues[{{7, 1}, {-4, 3}}]"#,
      r#"{5, 5}"#,
    );
  }
  #[test]
  fn eigenvectors_1() {
    assert_case(
      r#"Eigenvectors[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]"#,
      r#"{{1, 1, 1}, {1, -2, 1}, {-1, 0, 1}}"#,
    );
  }
  #[test]
  fn eigenvectors_2() {
    assert_case(
      r#"Eigenvectors[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]; Eigenvectors[{{1, 0, 0}, {0, 1, 0}, {0, 0, 0}}]"#,
      r#"{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}}"#,
    );
  }
  #[test]
  fn eigenvectors_3() {
    assert_case(
      r#"Eigenvectors[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]; Eigenvectors[{{1, 0, 0}, {0, 1, 0}, {0, 0, 0}}]; Eigenvectors[{{2, 0, 0}, {0, -1, 0}, {0, 0, 0}}]"#,
      r#"{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"#,
    );
  }
  #[test]
  fn eigenvectors_4() {
    assert_case(
      r#"Eigenvectors[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]; Eigenvectors[{{1, 0, 0}, {0, 1, 0}, {0, 0, 0}}]; Eigenvectors[{{2, 0, 0}, {0, -1, 0}, {0, 0, 0}}]; Eigenvectors[{{0.1, 0.2}, {0.8, 0.5}}]"#,
      r#"{{-0.29524180884432627, -0.9554225632202384}, {-0.6289601696450942, 0.777437524821136}}"#,
    );
  }
  #[test]
  fn inverse_1() {
    assert_case(
      r#"Inverse[{{1, 2, 0}, {2, 3, 0}, {3, 4, 1}}]"#,
      r#"{{-3, 2, 0}, {2, -1, 0}, {1, -2, 1}}"#,
    );
  }
  #[test]
  fn least_squares() {
    assert_case(
      r#"LeastSquares[{{1, 2}, {2, 3}, {5, 6}}, {1, 5, 3}]"#,
      r#"{-28 / 13, 31 / 13}"#,
    );
  }
  #[test]
  fn linear_solve_1() {
    assert_case(
      r#"LinearSolve[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}, {1, 2, 3}]"#,
      r#"{0, 1, 2}"#,
    );
  }
  #[test]
  fn list_literal_4() {
    assert_case(
      r#"LinearSolve[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}, {1, 2, 3}]; {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}} . {0, 1, 2}"#,
      r#"{1, 2, 3}"#,
    );
  }
  #[test]
  fn linear_solve_2() {
    assert_case(
      r#"LinearSolve[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}, {1, 2, 3}]; {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}} . {0, 1, 2}; LinearSolve[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1, 1}]"#,
      r#"{-1, 1, 0}"#,
    );
  }
  #[test]
  fn matrix_exp_1() {
    assert_case(
      r#"MatrixExp[{{0, 2}, {0, 1}}]"#,
      r#"{{1, 2*(-1 + E)}, {0, E}}"#,
    );
  }
  #[test]
  fn matrix_exp_2() {
    assert_case(
      r#"MatrixExp[{{0, 2}, {0, 1}}]; MatrixExp[{{1.5, 0.5}, {0.5, 2.0}}]"#,
      r#"{{5.162660242762233, 3.0295198346219987}, {3.0295198346219983, 8.192180077384233}}"#,
    );
  }
  #[test]
  fn matrix_exp_repeated_eigenvalue() {
    // Jordan form f(M) = f'(λ)·M + (f(λ) − λ·f'(λ))·I; previously stayed
    // unevaluated for defective matrices
    assert_case(r#"MatrixExp[{{1, 1}, {0, 1}}]"#, r#"{{E, E}, {0, E}}"#);
    assert_case(
      r#"MatrixExp[{{4, 1}, {0, 4}}]"#,
      r#"{{E^4, E^4}, {0, E^4}}"#,
    );
  }

  // A zero-trace real 2x2 matrix has pure-imaginary eigenvalues ±i·b; its
  // exponential is the rotation matrix in Cos/Sin form, not a tangle of
  // complex exponentials.
  #[test]
  fn matrix_exp_pure_imaginary_eigenvalues() {
    assert_case(
      r#"MatrixExp[{{0, -1}, {1, 0}}]"#,
      r#"{{Cos[1], -Sin[1]}, {Sin[1], Cos[1]}}"#,
    );
    assert_case(
      r#"MatrixExp[{{0, -2}, {2, 0}}]"#,
      r#"{{Cos[2], -Sin[2]}, {Sin[2], Cos[2]}}"#,
    );
  }
  // MatrixExp[m, v] computes the action MatrixExp[m].v on a vector.
  #[test]
  fn matrix_exp_on_vector() {
    // Diagonal matrix: each component is scaled by Exp of its eigenvalue.
    assert_case(r#"MatrixExp[{{1, 0}, {0, 2}}, {3, 5}]"#, r#"{3*E, 5*E^2}"#);
    assert_case(r#"MatrixExp[{{2, 0}, {0, 3}}, {1, 1}]"#, r#"{E^2, E^3}"#);
    // A nilpotent matrix gives an exact integer result.
    assert_case(r#"MatrixExp[{{0, 1}, {0, 0}}, {1, 1}]"#, r#"{2, 1}"#);
    // The zero matrix acts as the identity.
    assert_case(r#"MatrixExp[{{0, 0}, {0, 0}}, {2, 3}]"#, r#"{2, 3}"#);
  }
  // A non-square first argument emits MatrixExp::matsq and stays unevaluated.
  #[test]
  fn matrix_exp_on_vector_nonsquare() {
    assert_case(
      r#"MatrixExp[{{1, 2, 3}}, {1, 2}]"#,
      r#"MatrixExp[{{1, 2, 3}}, {1, 2}]"#,
    );
  }
  // A non-vector second argument emits MatrixExp::vector and stays unevaluated.
  #[test]
  fn matrix_exp_on_vector_nonvector() {
    assert_case(
      r#"MatrixExp[{{1, 0}, {0, 2}}, {{1, 0}, {0, 1}}]"#,
      r#"MatrixExp[{{1, 0}, {0, 2}}, {{1, 0}, {0, 1}}]"#,
    );
  }
  // MatrixFunction[f, m] applies a named scalar function to the matrix.
  #[test]
  fn matrix_function_diagonal() {
    assert_case(
      r#"MatrixFunction[Sqrt, {{2, 0}, {0, 8}}]"#,
      r#"{{Sqrt[2], 0}, {0, 2*Sqrt[2]}}"#,
    );
    assert_case(
      r#"MatrixFunction[Exp, {{1, 0}, {0, 2}}]"#,
      r#"{{E, 0}, {0, E^2}}"#,
    );
    assert_case(
      r#"MatrixFunction[Sin, {{0, 0}, {0, Pi/2}}]"#,
      r#"{{0, 0}, {0, 1}}"#,
    );
    assert_case(
      r#"MatrixFunction[Log, {{1, 0}, {0, E}}]"#,
      r#"{{0, 0}, {0, 1}}"#,
    );
  }
  // A 2x2 with distinct eigenvalues uses the Sylvester formula; an integer
  // result matches wolframscript exactly.
  #[test]
  fn matrix_function_sylvester_integer() {
    assert_case(
      r#"MatrixFunction[Sqrt, {{5, 4}, {4, 5}}]"#,
      r#"{{2, 1}, {1, 2}}"#,
    );
  }
  // A defective (repeated-eigenvalue) matrix with Exp uses the Jordan form.
  #[test]
  fn matrix_function_nilpotent_exp() {
    assert_case(
      r#"MatrixFunction[Exp, {{0, 1}, {0, 0}}]"#,
      r#"{{1, 1}, {0, 1}}"#,
    );
  }
  // A pure function applied to a diagonal matrix.
  #[test]
  fn matrix_function_pure_diagonal() {
    assert_case(
      r#"MatrixFunction[#^2 &, {{1, 0}, {0, 2}}]"#,
      r#"{{1, 0}, {0, 4}}"#,
    );
    assert_case(
      r#"MatrixFunction[1/# &, {{2, 0}, {0, 4}}]"#,
      r#"{{1/2, 0}, {0, 1/4}}"#,
    );
  }
  // A non-square argument emits MatrixFunction::matsq and stays unevaluated.
  #[test]
  fn matrix_function_nonsquare() {
    assert_case(
      r#"MatrixFunction[Exp, {{1, 2}, {3}}]"#,
      r#"MatrixFunction[Exp, {{1, 2}, {3}}]"#,
    );
  }
  #[test]
  fn matrix_log_diagonal() {
    assert_case(
      r#"MatrixLog[{{2, 0}, {0, 3}}]"#,
      r#"{{Log[2], 0}, {0, Log[3]}}"#,
    );
    assert_case(r#"MatrixLog[{{E, 0}, {0, E^2}}]"#, r#"{{1, 0}, {0, 2}}"#);
  }
  #[test]
  fn matrix_log_nilpotent_offset() {
    assert_case(r#"MatrixLog[{{1, 1}, {0, 1}}]"#, r#"{{0, 1}, {0, 0}}"#);
    assert_case(
      r#"MatrixLog[{{4, 1}, {0, 4}}]"#,
      r#"{{Log[4], 1/4}, {0, Log[4]}}"#,
    );
  }
  #[test]
  fn matrix_log_symmetric() {
    assert_case(
      r#"MatrixLog[{{2, 1}, {1, 2}}]"#,
      r#"{{Log[3]/2, Log[3]/2}, {Log[3]/2, Log[3]/2}}"#,
    );
  }
  #[test]
  fn matrix_log_singular_stays_unevaluated() {
    // Singular matrix: MatrixLog::fnand message, unevaluated result
    assert_case(
      r#"MatrixLog[{{1, 1}, {1, 1}}]"#,
      r#"MatrixLog[{{1, 1}, {1, 1}}]"#,
    );
  }
  #[test]
  fn matrix_power_1() {
    assert_case(
      r#"MatrixPower[{{1, 2}, {1, 1}}, 10]"#,
      r#"{{3363, 4756}, {2378, 3363}}"#,
    );
  }
  #[test]
  fn matrix_power_2() {
    assert_case(
      r#"MatrixPower[{{1, 2}, {1, 1}}, 10]; MatrixPower[{{1, 2}, {2, 5}}, -3]"#,
      r#"{{169, -70}, {-70, 29}}"#,
    );
  }
  #[test]
  fn matrix_power_diagonal_symbolic_n() {
    assert_case(
      r#"MatrixPower[{{2, 0}, {0, 3}}, n]"#,
      r#"{{2^n, 0}, {0, 3^n}}"#,
    );
  }
  #[test]
  fn matrix_power_diagonal_symbolic_entries_and_n() {
    assert_case(
      r#"MatrixPower[{{a, 0}, {0, b}}, n]"#,
      r#"{{a^n, 0}, {0, b^n}}"#,
    );
  }
  #[test]
  fn matrix_power_3x3_diagonal_symbolic_n() {
    assert_case(
      r#"MatrixPower[{{2, 0, 0}, {0, 3, 0}, {0, 0, 5}}, n]"#,
      r#"{{2^n, 0, 0}, {0, 3^n, 0}, {0, 0, 5^n}}"#,
    );
  }
  #[test]
  fn matrix_power_non_diagonal_symbolic_n_stays_unevaluated() {
    // 2x2 single-block input with messy eigenvalues stays unevaluated:
    // wolframscript closes it with Sqrt[33] terms, which the block
    // fast path below intentionally avoids.
    assert_case(
      r#"MatrixPower[{{1, 2}, {3, 4}}, n]"#,
      r#"MatrixPower[{{1, 2}, {3, 4}}, n]"#,
    );
  }

  // ─── Block-decomposable MatrixPower with symbolic n ───────────────
  //
  // A square matrix that splits into multiple connected components
  // (i.e. is block-diagonal after a row/column permutation) can be
  // raised to a symbolic power by handling each block independently:
  //
  //   - size-1 blocks: entry ↦ entry^n
  //   - size-2 blocks: closed-form
  //       M^n[i,j] = M[i,j] · (λ_1^n − λ_2^n) / (λ_1 − λ_2)   (i ≠ j)
  //       M^n[i,i] = (M[i,i] − λ_2)·λ_1^n / (λ_1 − λ_2)
  //                + (λ_1 − M[i,i])·λ_2^n / (λ_1 − λ_2)

  #[test]
  fn matrix_power_audit_3x3_block_symbolic_n() {
    // Audit case: 1×1 block at index 1 (eigenvalue 2) and a 2×2 block
    // {{0, 1}, {-1, 0}} on indices 0, 2 (eigenvalues ±I).
    //
    // Woxi's terms are logically equivalent to wolframscript's
    // (`I*I^n = I^(n+1)`, Plus-term ordering differs cosmetically) but
    // not byte-identical, so test against Woxi's canonical form.
    assert_case(
      r#"MatrixPower[{{0, 0, 1}, {0, 2, 0}, {-1, 0, 0}}, n]"#,
      r#"{{I^n/2 + (-I)^n/2, 0, -1/2*I^(1 + n) + I/2*(-I)^n}, {0, 2^n, 0}, {I^(1 + n)/2 - I/2*(-I)^n, 0, I^n/2 + (-I)^n/2}}"#,
    );
  }

  #[test]
  fn matrix_power_block_2x2_2x2_symbolic_n() {
    // Two disjoint 2×2 blocks at indices {0, 1} and {2, 3}.
    // Each 2×2 has trace 0, det 1 ⇒ eigenvalues ±I, same closed form.
    assert_case(
      r#"MatrixPower[{{0, 1, 0, 0}, {-1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, -1, 0}}, n]"#,
      r#"{{I^n/2 + (-I)^n/2, -1/2*I^(1 + n) + I/2*(-I)^n, 0, 0}, {I^(1 + n)/2 - I/2*(-I)^n, I^n/2 + (-I)^n/2, 0, 0}, {0, 0, I^n/2 + (-I)^n/2, -1/2*I^(1 + n) + I/2*(-I)^n}, {0, 0, I^(1 + n)/2 - I/2*(-I)^n, I^n/2 + (-I)^n/2}}"#,
    );
  }

  // Verifying that substituting integer values into the symbolic result
  // gives the same answer as `MatrixPower` with that integer directly.
  // This sanity-checks that the closed-form formula is correct even when
  // its display differs from wolframscript's.
  #[test]
  fn matrix_power_audit_consistent_at_n_4() {
    assert_case(
      r#"r = MatrixPower[{{0, 0, 1}, {0, 2, 0}, {-1, 0, 0}}, n]; r /. n -> 4"#,
      r#"{{1, 0, 0}, {0, 16, 0}, {0, 0, 1}}"#,
    );
  }

  #[test]
  fn matrix_power_audit_consistent_at_n_5() {
    assert_case(
      r#"r = MatrixPower[{{0, 0, 1}, {0, 2, 0}, {-1, 0, 0}}, n]; r /. n -> 5"#,
      r#"{{0, 0, 1}, {0, 32, 0}, {-1, 0, 0}}"#,
    );
  }
  #[test]
  fn matrix_rank_1() {
    assert_case(r#"MatrixRank[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]"#, r#"2"#);
  }
  #[test]
  fn matrix_rank_2() {
    assert_case(
      r#"MatrixRank[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]; MatrixRank[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]"#,
      r#"3"#,
    );
  }
  #[test]
  fn matrix_rank_3() {
    assert_case(
      r#"MatrixRank[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]; MatrixRank[{{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}]; MatrixRank[{{a, b}, {3 a, 3 b}}]"#,
      r#"1"#,
    );
  }
  #[test]
  fn null_space_1() {
    assert_case(
      r#"NullSpace[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]"#,
      r#"{{1, -2, 1}}"#,
    );
  }
  #[test]
  fn null_space_2() {
    assert_case(
      r#"NullSpace[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]; A = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}; NullSpace[A]"#,
      r#"{}"#,
    );
  }
  #[test]
  fn matrix_rank_4() {
    assert_case(
      r#"NullSpace[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]; A = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}}; NullSpace[A]; MatrixRank[A]"#,
      r#"3"#,
    );
  }
  #[test]
  fn pseudo_inverse_1() {
    assert_case(
      r#"PseudoInverse[{{1, 2}, {2, 3}, {3, 4}}]"#,
      r#"{{-11 / 6, -1 / 3, 7 / 6}, {4 / 3, 1 / 3, -2 / 3}}"#,
    );
  }
  #[test]
  fn pseudo_inverse_wide_matrix() {
    // Regression: a wide (ncols > nrows) full-row-rank matrix used the right
    // one-sided formula M^T (M M^T)^{-1}. Previously the left formula ran
    // first, inverting the singular ncols×ncols product — yielding all
    // Indeterminate (and hanging for large ncols).
    assert_case(
      r#"PseudoInverse[{{1, 0, 0}, {0, 1, 0}}]"#,
      r#"{{1, 0}, {0, 1}, {0, 0}}"#,
    );
  }
  #[test]
  fn pseudo_inverse_2() {
    assert_case(
      r#"PseudoInverse[{{1, 2}, {2, 3}, {3, 4}}]; PseudoInverse[{{1, 2, 0}, {2, 3, 0}, {3, 4, 1}}]"#,
      r#"{{-3, 2, 0}, {2, -1, 0}, {1, -2, 1}}"#,
    );
  }
  #[test]
  fn pseudo_inverse_3() {
    assert_case(
      r#"PseudoInverse[{{1, 2}, {2, 3}, {3, 4}}]; PseudoInverse[{{1, 2, 0}, {2, 3, 0}, {3, 4, 1}}]; PseudoInverse[{{1.0, 2.5}, {2.5, 1.0}}]"#,
      r#"{{-0.19047619047619055, 0.4761904761904761}, {0.4761904761904762, -0.1904761904761904}}"#,
    );
  }
  #[test]
  fn qr_decomposition() {
    assert_case(
      r#"QRDecomposition[{{1, 2}, {3, 4}, {5, 6}}]"#,
      r#"{{{1/Sqrt[35], 3/Sqrt[35], Sqrt[5/7]}, {13/Sqrt[210], 2*Sqrt[2/105], -Sqrt[5/42]}}, {{Sqrt[35], 44/Sqrt[35]}, {0, 2*Sqrt[6/35]}}}"#,
    );
  }
  #[test]
  fn row_reduce_1() {
    assert_case(
      r#"RowReduce[{{1, 0, a}, {1, 1, b}}]"#,
      r#"{{1, 0, a}, {0, 1, -a + b}}"#,
    );
  }
  #[test]
  fn row_reduce_2() {
    assert_case(
      r#"RowReduce[{{1, 0, a}, {1, 1, b}}]; RowReduce[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}] // MatrixForm"#,
      r#"MatrixForm[{{1, 0, -1}, {0, 1, 2}, {0, 0, 0}}]"#,
    );
  }
  #[test]
  fn singular_value_decomposition() {
    assert_case(
      r#"SingularValueDecomposition[{{1.5, 2.0}, {2.5, 3.0}}]"#,
      r#"{{{-0.5389535334972083, 0.8423354965397537}, {-0.8423354965397537, -0.5389535334972083}}, {{4.63555452966064, 0.}, {0., 0.10786196059193043}}, {{-0.6286775450376474, -0.7776660879615599}, {-0.7776660879615598, 0.6286775450376474}}}"#,
    );
  }
  #[test]
  fn singular_value_list_exact() {
    assert_case(
      r#"SingularValueList[{{1, 2}, {3, 4}}]"#,
      r#"{Sqrt[15 + Sqrt[221]], Sqrt[15 - Sqrt[221]]}"#,
    );
  }
  #[test]
  fn singular_value_list_drops_zero() {
    // Rank-2 matrix: only the nonzero singular values are returned
    assert_case(
      r#"SingularValueList[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]"#,
      r#"{Sqrt[(3*(95 + Sqrt[8881]))/2], Sqrt[(3*(95 - Sqrt[8881]))/2]}"#,
    );
  }
  #[test]
  fn singular_value_list_rank_one() {
    assert_case(r#"SingularValueList[{{1, 1}, {1, 1}}]"#, r#"{2}"#);
  }
  #[test]
  fn singular_value_list_integer_results() {
    assert_case(r#"SingularValueList[{{3, 0}, {0, 2}}]"#, r#"{3, 2}"#);
    assert_case(r#"SingularValueList[{{0, -2}, {1, 0}}]"#, r#"{2, 1}"#);
  }
  #[test]
  fn singular_value_list_rectangular() {
    assert_case(
      r#"SingularValueList[{{1, 2, 3}, {4, 5, 6}}]"#,
      r#"{Sqrt[(91 + Sqrt[8065])/2], Sqrt[(91 - Sqrt[8065])/2]}"#,
    );
  }
  #[test]
  fn singular_value_list_machine_precision() {
    assert_case(r#"SingularValueList[{{3., 0.}, {0., 4.}}]"#, r#"{4., 3.}"#);
    // Numerically singular: the zero singular value is dropped
    assert_case(r#"SingularValueList[{{1., 2.}, {2., 4.}}]"#, r#"{5.}"#);
  }
  #[test]
  fn singular_value_list_k_largest() {
    assert_case(
      r#"SingularValueList[{{1, 2}, {3, 4}}, 1]"#,
      r#"{Sqrt[15 + Sqrt[221]]}"#,
    );
    // Negative k: the |k| smallest
    assert_case(
      r#"SingularValueList[{{1, 2}, {3, 4}}, -1]"#,
      r#"{Sqrt[15 - Sqrt[221]]}"#,
    );
  }
  #[test]
  fn tr_1() {
    assert_case(r#"Tr[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]"#, r#"15"#);
  }
  #[test]
  fn tr_2() {
    assert_case(
      r#"Tr[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]; Tr[{{a, b, c}, {d, e, f}, {g, h, i}}]"#,
      r#"a + e + i"#,
    );
  }
  // Tr of a SparseArray traces its dense form.
  #[test]
  fn tr_sparse_array() {
    assert_case(
      r#"Tr[SparseArray[{{1, 1} -> 1, {2, 2} -> 5}, {2, 2}]]"#,
      r#"6"#,
    );
    assert_case(
      r#"Tr[SparseArray[{{1, 1} -> 1}, {2, 2}], List]"#,
      r#"{1, 0}"#,
    );
  }
  #[test]
  fn matrix_q_1() {
    assert_case(r#"MatrixQ[{{1, 3}, {4.0, 3/2}}, NumberQ]"#, r#"True"#);
  }
  #[test]
  fn matrix_q_2() {
    assert_case(
      r#"MatrixQ[{{1, 3}, {4.0, 3/2}}, NumberQ]; MatrixQ[{{1}, {1, 2}}] (* first row should have length two *)"#,
      r#"False"#,
    );
  }
  #[test]
  fn matrix_q_3() {
    assert_case(
      r#"MatrixQ[{{1, 3}, {4.0, 3/2}}, NumberQ]; MatrixQ[{{1}, {1, 2}}] (* first row should have length two *); MatrixQ[Array[a, {1, 1, 2}]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn matrix_q_4() {
    assert_case(
      r#"MatrixQ[{{1, 3}, {4.0, 3/2}}, NumberQ]; MatrixQ[{{1}, {1, 2}}] (* first row should have length two *); MatrixQ[Array[a, {1, 1, 2}]]; MatrixQ[{{1, 2}, {3, 4 + 5}}, Positive]"#,
      r#"True"#,
    );
  }
  #[test]
  fn matrix_q_5() {
    assert_case(
      r#"MatrixQ[{{1, 3}, {4.0, 3/2}}, NumberQ]; MatrixQ[{{1}, {1, 2}}] (* first row should have length two *); MatrixQ[Array[a, {1, 1, 2}]]; MatrixQ[{{1, 2}, {3, 4 + 5}}, Positive]; MatrixQ[{{1, 2 I}, {3, 4 + 5}}, Positive]"#,
      r#"False"#,
    );
  }
  #[test]
  fn vector_q() {
    assert_case(r#"VectorQ[{a, b, c}]"#, r#"True"#);
  }
  #[test]
  fn cross_1() {
    assert_case(
      r#"Cross[{x1, y1, z1}, {x2, y2, z2}]"#,
      r#"{-(y2*z1) + y1*z2, x2*z1 - x1*z2, -(x2*y1) + x1*y2}"#,
    );
  }
  #[test]
  fn cross_2() {
    assert_case(
      r#"Cross[{x1, y1, z1}, {x2, y2, z2}]; Cross[{x, y}]"#,
      r#"{-y, x}"#,
    );
  }
  #[test]
  fn set_3() {
    assert_case(
      r#"Cross[{x1, y1, z1}, {x2, y2, z2}]; Cross[{x, y}]; v1 = {1, Sqrt[3]}; v2 = Cross[v1]"#,
      r#"{-Sqrt[3], 1}"#,
    );
  }
  #[test]
  fn norm_1() {
    assert_case(
      r#"Norm[{x, y, z}]"#,
      r#"Sqrt[Abs[x] ^ 2 + Abs[y] ^ 2 + Abs[z] ^ 2]"#,
    );
  }
  #[test]
  fn norm_2() {
    assert_case(r#"Norm[{x, y, z}]; Norm[{3, 4}, 2]"#, r#"5"#);
  }
  #[test]
  fn norm_3() {
    assert_case(
      r#"Norm[{x, y, z}]; Norm[{3, 4}, 2]; Norm[{10, 100, 200}, 1]"#,
      r#"310"#,
    );
  }
  #[test]
  fn norm_4() {
    assert_case(
      r#"Norm[{x, y, z}]; Norm[{3, 4}, 2]; Norm[{10, 100, 200}, 1]; Norm[{x, y, z}, Infinity]"#,
      r#"Max[Abs[x], Abs[y], Abs[z]]"#,
    );
  }
  #[test]
  fn norm_5() {
    assert_case(
      r#"Norm[{x, y, z}]; Norm[{3, 4}, 2]; Norm[{10, 100, 200}, 1]; Norm[{x, y, z}, Infinity]; Norm[{-100, 2, 3, 4}, Infinity]"#,
      r#"100"#,
    );
  }
  #[test]
  fn norm_6() {
    assert_case(
      r#"Norm[{x, y, z}]; Norm[{3, 4}, 2]; Norm[{10, 100, 200}, 1]; Norm[{x, y, z}, Infinity]; Norm[{-100, 2, 3, 4}, Infinity]; Norm[1 + I]"#,
      r#"Sqrt[2]"#,
    );
  }
  #[test]
  fn norm_7() {
    // Norm[matrix, "Frobenius"] sums Abs[a_ij]^2 over every entry. Also
    // exercises Array's pure-function call path so `Subscript[a, ##] &`
    // expands to `Subscript[a, i, j]`.
    assert_case(
      r#"Norm[{x, y, z}]; Norm[{3, 4}, 2]; Norm[{10, 100, 200}, 1]; Norm[{x, y, z}, Infinity]; Norm[{-100, 2, 3, 4}, Infinity]; Norm[1 + I]; Norm[Array[Subscript[a, ##] &, {2, 2}], "Frobenius"]"#,
      r#"Sqrt[Abs[Subscript[a, 1, 1]]^2 + Abs[Subscript[a, 1, 2]]^2 + Abs[Subscript[a, 2, 1]]^2 + Abs[Subscript[a, 2, 2]]^2]"#,
    );
  }
  #[test]
  fn norm_empty_vector_emits_nvm() {
    // Norm[{}] has no value: it stays unevaluated (with Norm::nvm) rather
    // than collapsing to 0. (wolframscript parity)
    assert_case(r#"Norm[{}]"#, r#"Norm[{}]"#);
  }
  #[test]
  fn norm_rank_three_tensor_emits_nvm() {
    // Norm accepts only scalars, vectors and matrices. A rank-3 array is
    // rejected with Norm::nvm and stays unevaluated rather than threading the
    // matrix-norm formula over the inner vectors. (wolframscript parity)
    assert_case(r#"Norm[{{{1, 2}}}]"#, r#"Norm[{{{1, 2}}}]"#);
  }
  #[test]
  fn norm_rank_four_tensor_emits_nvm() {
    assert_case(r#"Norm[{{{{1}}}}]"#, r#"Norm[{{{{1}}}}]"#);
  }
  // An inexact (Real) vector gives an inexact norm even at a whole-number
  // value: Norm[{3., 4.}] is 5., not the exact 5. Exact vectors are
  // unchanged. Per wolframscript.
  #[test]
  fn norm_inexact_vector_stays_real() {
    assert_case(r#"Norm[{3., 4.}]"#, r#"5."#);
    assert_case(r#"Head[Norm[{3., 4.}]]"#, r#"Real"#);
    assert_case(r#"Norm[{6., 8.}, 1]"#, r#"14."#);
    assert_case(r#"Norm[{3., 4.}, Infinity]"#, r#"4."#);
    // An exact vector still gives an exact integer.
    assert_case(r#"Norm[{3, 4}]"#, r#"5"#);
  }
  #[test]
  fn kronecker_product_1() {
    assert_case(
      r#"a = {{a11, a12}, {a21, a22}}; b = {{b11, b12}, {b21, b22}}; KroneckerProduct[a, b]"#,
      r#"{{a11*b11, a11*b12, a12*b11, a12*b12}, {a11*b21, a11*b22, a12*b21, a12*b22}, {a21*b11, a21*b12, a22*b11, a22*b12}, {a21*b21, a21*b22, a22*b21, a22*b22}}"#,
    );
  }
  #[test]
  fn kronecker_product_2() {
    assert_case(
      r#"a = {{a11, a12}, {a21, a22}}; b = {{b11, b12}, {b21, b22}}; KroneckerProduct[a, b]; a = {{0, 1}, {-1, 0}}; b = {{1, 2}, {3, 4}}; KroneckerProduct[a, b] // MatrixForm"#,
      r#"MatrixForm[{{0, 0, 1, 2}, {0, 0, 3, 4}, {-1, -2, 0, 0}, {-3, -4, 0, 0}}]"#,
    );
  }
  #[test]
  fn normalize_1() {
    assert_case(
      r#"Normalize[{1, 1, 1, 1}]"#,
      r#"{1 / 2, 1 / 2, 1 / 2, 1 / 2}"#,
    );
  }
  #[test]
  fn normalize_2() {
    assert_case(
      r#"Normalize[{1, 1, 1, 1}]; Normalize[1 + I]"#,
      r#"(1 + I)/Sqrt[2]"#,
    );
  }
  #[test]
  fn projection_1() {
    assert_case(r#"Projection[{5, 6, 7}, {1, 0, 0}]"#, r#"{5, 0, 0}"#);
  }
  #[test]
  fn projection_2() {
    assert_case(
      r#"Projection[{5, 6, 7}, {1, 0, 0}]; Projection[{2, 3}, {1, 2}]"#,
      r#"{8 / 5, 16 / 5}"#,
    );
  }
  #[test]
  fn projection_3() {
    assert_case(
      r#"Projection[{5, 6, 7}, {1, 0, 0}]; Projection[{2, 3}, {1, 2}]; Projection[{1.3, 2.1, 3.1}, {-0.3, 4.2, 5.3}]"#,
      r#"{-0.16276735050196423, 2.278742907027499, 2.8755565255347015}"#,
    );
  }
  #[test]
  fn projection_4() {
    assert_case(
      r#"Projection[{5, 6, 7}, {1, 0, 0}]; Projection[{2, 3}, {1, 2}]; Projection[{1.3, 2.1, 3.1}, {-0.3, 4.2, 5.3}]; Projection[{3 + I, 2, 2 - I}, {2, 4, 5 I}]"#,
      r#"{2/5 - (16*I)/45, 4/5 - (32*I)/45, 8/9 + I}"#,
    );
  }
  #[test]
  fn projection_5() {
    assert_case(
      r#"Projection[{5, 6, 7}, {1, 0, 0}]; Projection[{2, 3}, {1, 2}]; Projection[{1.3, 2.1, 3.1}, {-0.3, 4.2, 5.3}]; Projection[{3 + I, 2, 2 - I}, {2, 4, 5 I}]; Projection[{a, b, c}, {1, 1, 1}]"#,
      r#"{(a + b + c) / 3, (a + b + c) / 3, (a + b + c) / 3}"#,
    );
  }
  #[test]
  fn projection_symbolic_vectors() {
    // Projection[u, v] for unbound symbolic vectors expands to the
    // Hermitian projection formula:
    //   (Conjugate[v] . u / Conjugate[v] . v) * v
    // Wolframscript renders this as
    //   (v*Conjugate[v] . u)/Conjugate[v] . v
    assert_case(
      r#"Projection[u, v]"#,
      r#"(v*Conjugate[v] . u) / Conjugate[v] . v"#,
    );
  }
  #[test]
  fn unit_vector_1() {
    assert_case(r#"UnitVector[2]"#, r#"{0, 1}"#);
  }
  #[test]
  fn unit_vector_2() {
    assert_case(r#"UnitVector[2]; UnitVector[4, 3]"#, r#"{0, 0, 1, 0}"#);
  }
  #[test]
  fn vector_angle_1() {
    assert_case(r#"VectorAngle[{1, 0}, {0, 1}]"#, r#"Pi / 2"#);
  }
  #[test]
  fn vector_angle_2() {
    assert_case(
      r#"VectorAngle[{1, 0}, {0, 1}]; VectorAngle[{1, 2}, {3, 1}]"#,
      r#"Pi / 4"#,
    );
  }
  #[test]
  fn vector_angle_3() {
    assert_case(
      r#"VectorAngle[{1, 0}, {0, 1}]; VectorAngle[{1, 2}, {3, 1}]; VectorAngle[{1, 1, 0}, {1, 0, 1}]"#,
      r#"Pi / 3"#,
    );
  }
  #[test]
  fn matrix_exp_3() {
    // Wolframscript-matched expectation. mathics rendered the φ symbol
    // as the `\[Phi]` named-character escape and parenthesised `(I/2)`;
    // wolframscript -code emits the literal Unicode `ϕ` (U+03D5) and
    // drops the redundant `(I/2)` parens because `*` already binds
    // tighter than `/`.
    assert_case(
      r#"Table[PauliMatrix[i], {i, 1, 3}]; PauliMatrix[1] . PauliMatrix[2] == I PauliMatrix[3]; MatrixExp[I \[Phi]/2 PauliMatrix[3]]"#,
      "{{E^(I/2*\u{03D5}), 0}, {0, E^((-1/2*I)*\u{03D5})}}",
    );
  }
  #[test]
  fn greater() {
    assert_case(
      r#"Table[PauliMatrix[i], {i, 1, 3}]; PauliMatrix[1] . PauliMatrix[2] == I PauliMatrix[3]; MatrixExp[I \[Phi]/2 PauliMatrix[3]]; % /. \[Phi] -> 2 Pi"#,
      r#"Out[0]"#,
    );
  }
  #[test]
  fn box_matrix() {
    assert_case(
      r#"BoxMatrix[3]"#,
      r#"{{1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}}"#,
    );
  }
  // BoxMatrix accepts a list of radii for a rectangular / n-D box of ones.
  #[test]
  fn box_matrix_radii_list() {
    // {1, 2}: 3 rows by 5 columns.
    assert_case(
      r#"BoxMatrix[{1, 2}]"#,
      r#"{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}"#,
    );
    // A zero radius collapses that dimension to length 1.
    assert_case(r#"BoxMatrix[{0, 1}]"#, r#"{{1, 1, 1}}"#);
    // Three radii give a 3D box of shape (2ri+1).
    assert_case(r#"Dimensions[BoxMatrix[{1, 2, 1}]]"#, r#"{3, 5, 3}"#);
  }
  // CrossMatrix is the n-dimensional cross structuring element.
  #[test]
  fn cross_matrix_scalar() {
    // A scalar radius gives a 2D cross in both directions.
    assert_case(r#"CrossMatrix[1]"#, r#"{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}"#);
    assert_case(r#"CrossMatrix[0]"#, r#"{{1}}"#);
    // A real radius is rounded.
    assert_case(
      r#"CrossMatrix[2.0]"#,
      r#"{{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}}"#,
    );
  }
  #[test]
  fn cross_matrix_radii_list() {
    // Different vertical and horizontal radii.
    assert_case(
      r#"CrossMatrix[{2, 1}]"#,
      r#"{{0, 1, 0}, {0, 1, 0}, {1, 1, 1}, {0, 1, 0}, {0, 1, 0}}"#,
    );
    // Three radii give a 3D cross.
    assert_case(
      r#"CrossMatrix[{1, 1, 1}]"#,
      r#"{{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}, {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}}"#,
    );
  }
  #[test]
  fn diagonal_matrix_1() {
    assert_case(
      r#"DiagonalMatrix[{1, 2, 3}]"#,
      r#"{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}"#,
    );
  }
  #[test]
  fn matrix_form_3() {
    assert_case(
      r#"DiagonalMatrix[{1, 2, 3}]; MatrixForm[%]"#,
      r#"MatrixForm[Out[0]]"#,
    );
  }
  #[test]
  fn diamond_matrix() {
    assert_case(
      r#"DiamondMatrix[3]"#,
      r#"{{0, 0, 0, 1, 0, 0, 0}, {0, 0, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 0, 0, 0}}"#,
    );
  }
  // DiamondMatrix accepts a list of radii for a rectangular / n-D L1 ball.
  #[test]
  fn diamond_matrix_radii_list() {
    // Wider horizontally than vertically.
    assert_case(
      r#"DiamondMatrix[{2, 3}]"#,
      r#"{{0, 0, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1, 0}, {0, 0, 0, 1, 0, 0, 0}}"#,
    );
    // Taller than wide.
    assert_case(
      r#"DiamondMatrix[{3, 2}]"#,
      r#"{{0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}}"#,
    );
    // Equal radii reduce to the square diamond.
    assert_case(
      r#"DiamondMatrix[{2, 2}]"#,
      r#"{{0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}}"#,
    );
    // Three radii give a 3D diamond (here the 3D cross).
    assert_case(
      r#"DiamondMatrix[{1, 1, 1}]"#,
      r#"{{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}, {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}}"#,
    );
  }
  #[test]
  fn disk_matrix() {
    assert_case(
      r#"DiskMatrix[3]"#,
      r#"{{0, 0, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1, 0}, {0, 0, 1, 1, 1, 0, 0}}"#,
    );
  }
  // DiskMatrix accepts a list of radii for an elliptical / n-D L2 ball.
  #[test]
  fn disk_matrix_radii_list() {
    // Wider than tall.
    assert_case(
      r#"DiskMatrix[{2, 3}]"#,
      r#"{{0, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {0, 1, 1, 1, 1, 1, 0}}"#,
    );
    // Tall and narrow.
    assert_case(
      r#"DiskMatrix[{1, 2}]"#,
      r#"{{0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}}"#,
    );
    // Equal radii reduce to the round disk.
    assert_case(
      r#"DiskMatrix[{2, 2}]"#,
      r#"{{0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}}"#,
    );
    // Three radii give a 3D ball.
    assert_case(
      r#"DiskMatrix[{1, 1, 1}]"#,
      r#"{{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}}"#,
    );
  }
  #[test]
  fn identity_matrix() {
    assert_case(
      r#"IdentityMatrix[3]"#,
      r#"{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"#,
    );
  }
  #[test]
  fn diagonal_1() {
    assert_case(
      r#"Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]"#,
      r#"{1, 5, 9}"#,
    );
  }
  #[test]
  fn diagonal_2() {
    assert_case(
      r#"Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]; Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]"#,
      r#"{2, 6}"#,
    );
  }
  #[test]
  fn diagonal_3() {
    assert_case(
      r#"Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]; Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]; Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, -1]"#,
      r#"{4, 8}"#,
    );
  }
  #[test]
  fn diagonal_4() {
    assert_case(
      r#"Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]; Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]; Diagonal[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, -1]; Diagonal[{{1, 2, 3}, {4, 5, 6}}]"#,
      r#"{1, 5}"#,
    );
  }
  #[test]
  fn diagonal_matrix_2() {
    assert_case(r#"DiagonalMatrix[a + b]"#, r#"DiagonalMatrix[a + b]"#);
  }
  #[test]
  fn dimensions_5() {
    assert_case(r#"Dimensions[{}]"#, r#"{0}"#);
  }
  #[test]
  fn dimensions_6() {
    assert_case(r#"Dimensions[{}]; Dimensions[{{}}]"#, r#"{1, 0}"#);
  }
  #[test]
  fn set_4() {
    assert_case(
      r#"Dimensions[{}]; Dimensions[{{}}]; A = {{ b ^ ( -1 / 2), 0}, {a * b ^ ( -1 / 2 ), b ^ ( 1 / 2 )}}"#,
      r#"{{1 / Sqrt[b], 0}, {a / Sqrt[b], Sqrt[b]}}"#,
    );
  }
  #[test]
  fn expr() {
    assert_case(
      r#"Dimensions[{}]; Dimensions[{{}}]; A = {{ b ^ ( -1 / 2), 0}, {a * b ^ ( -1 / 2 ), b ^ ( 1 / 2 )}}; A . Inverse[A]"#,
      r#"{{1, 0}, {0, 1}}"#,
    );
  }
  #[test]
  fn a() {
    assert_case(
      r#"Dimensions[{}]; Dimensions[{{}}]; A = {{ b ^ ( -1 / 2), 0}, {a * b ^ ( -1 / 2 ), b ^ ( 1 / 2 )}}; A . Inverse[A]; A"#,
      r#"{{1 / Sqrt[b], 0}, {a / Sqrt[b], Sqrt[b]}}"#,
    );
  }
  #[test]
  fn transpose_7() {
    assert_case(
      r#"Dimensions[{}]; Dimensions[{{}}]; A = {{ b ^ ( -1 / 2), 0}, {a * b ^ ( -1 / 2 ), b ^ ( 1 / 2 )}}; A . Inverse[A]; A; Transpose[x]"#,
      r#"Transpose[x]"#,
    );
  }
  #[test]
  fn numeric_array() {
    // wolframscript prints NumericArray with the dimensions inline
    // (`NumericArray[<2,2>, UnsignedInteger8]`) rather than spelling out
    // the data list as the mathics expectation did.
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]"#,
      r#"NumericArray[<2,2>, UnsignedInteger8]"#,
    );
  }
  #[test]
  fn to_string() {
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]; ToString[NumericArray[{{1,2},{3,4}}]]"#,
      r#""NumericArray[<2,2>, UnsignedInteger8]""#,
    );
  }
  #[test]
  fn head() {
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]; ToString[NumericArray[{{1,2},{3,4}}]]; Head[NumericArray[{1,2}]]"#,
      r#"NumericArray"#,
    );
  }
  #[test]
  fn atom_q() {
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]; ToString[NumericArray[{{1,2},{3,4}}]]; Head[NumericArray[{1,2}]]; AtomQ[NumericArray[{1,2}]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn first_1() {
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]; ToString[NumericArray[{{1,2},{3,4}}]]; Head[NumericArray[{1,2}]]; AtomQ[NumericArray[{1,2}]]; First[NumericArray[{1,2,3}]]"#,
      r#"1"#,
    );
  }
  #[test]
  fn first_2() {
    // First on a multi-dim NumericArray returns a sub-NumericArray, which
    // wolframscript renders in the same `<dim>` shorthand.
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]; ToString[NumericArray[{{1,2},{3,4}}]]; Head[NumericArray[{1,2}]]; AtomQ[NumericArray[{1,2}]]; First[NumericArray[{1,2,3}]]; First[NumericArray[{{1,2}, {3,4}}]]"#,
      r#"NumericArray[<2>, UnsignedInteger8]"#,
    );
  }
  #[test]
  fn last_1() {
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]; ToString[NumericArray[{{1,2},{3,4}}]]; Head[NumericArray[{1,2}]]; AtomQ[NumericArray[{1,2}]]; First[NumericArray[{1,2,3}]]; First[NumericArray[{{1,2}, {3,4}}]]; Last[NumericArray[{1,2,3}]]"#,
      r#"3"#,
    );
  }
  #[test]
  fn last_2() {
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]; ToString[NumericArray[{{1,2},{3,4}}]]; Head[NumericArray[{1,2}]]; AtomQ[NumericArray[{1,2}]]; First[NumericArray[{1,2,3}]]; First[NumericArray[{{1,2}, {3,4}}]]; Last[NumericArray[{1,2,3}]]; Last[NumericArray[{{1,2}, {3,4}}]]"#,
      r#"NumericArray[<2>, UnsignedInteger8]"#,
    );
  }
  #[test]
  fn normal() {
    assert_case(
      r#"NumericArray[{{1,2},{3,4}}]; ToString[NumericArray[{{1,2},{3,4}}]]; Head[NumericArray[{1,2}]]; AtomQ[NumericArray[{1,2}]]; First[NumericArray[{1,2,3}]]; First[NumericArray[{{1,2}, {3,4}}]]; Last[NumericArray[{1,2,3}]]; Last[NumericArray[{{1,2}, {3,4}}]]; Normal[NumericArray[{{1,2}, {3,4}}]]"#,
      r#"{{1, 2}, {3, 4}}"#,
    );
  }
  #[test]
  fn inverse_2() {
    assert_case(r#"Inverse[{{0, 2},{2, 0}}]"#, r#"{{0, 1 / 2},{1 / 2, 0}}"#);
  }
  #[test]
  fn inverse_3() {
    assert_case(
      r#"Inverse[{{0, 2},{2, 0}}]; Inverse[{{0, 2.},{2, 0}}]"#,
      r#"{{0., 0.5}, {0.5, 0.}}"#,
    );
  }
  #[test]
  fn inverse_4() {
    assert_case(
      r#"Inverse[{{0, 2},{2, 0}}]; Inverse[{{0, 2.},{2, 0}}]; Inverse[{{0, 2., 0},{2, 0, 0}, {0, 0, a}}]"#,
      r#"{{0., 0.5, 0.}, {0.5, 0., 0.}, {0., 0., 1./a}}"#,
    );
  }
  #[test]
  fn inverse_5() {
    assert_case(
      r#"Inverse[{{0, 2},{2, 0}}]; Inverse[{{0, 2.},{2, 0}}]; Inverse[{{0, 2., 0},{2, 0, 0}, {0, 0, a}}]; Inverse[{{a, b},{c, d}}].{{a, b},{c, d}}"#,
      r#"{{-((b*c)/(-(b*c) + a*d)) + (a*d)/(-(b*c) + a*d), 0}, {0, -((b*c)/(-(b*c) + a*d)) + (a*d)/(-(b*c) + a*d)}}"#,
    );
  }
  #[test]
  fn inverse_6() {
    assert_case(
      r#"Inverse[{{0, 2},{2, 0}}]; Inverse[{{0, 2.},{2, 0}}]; Inverse[{{0, 2., 0},{2, 0, 0}, {0, 0, a}}]; Inverse[{{a, b},{c, d}}].{{a, b},{c, d}}; Inverse[{{a, b},{c, d}}].{{a, b},{c, d}}//Simplify"#,
      r#"{{1, 0},{0, 1}}"#,
    );
  }
  #[test]
  fn inverse_7() {
    assert_case(
      r#"Inverse[{{0, 2},{2, 0}}]; Inverse[{{0, 2.},{2, 0}}]; Inverse[{{0, 2., 0},{2, 0, 0}, {0, 0, a}}]; Inverse[{{a, b},{c, d}}].{{a, b},{c, d}}; Inverse[{{a, b},{c, d}}].{{a, b},{c, d}}//Simplify; Inverse[{{g[a], g[b]},{g[c], g[d]}}].{{g[a], g[b]},{g[c], g[d]}}//Simplify"#,
      r#"{{1, 0},{0, 1}}"#,
    );
  }
  #[test]
  fn normalize_3() {
    assert_case(r#"Normalize[0]"#, r#"0"#);
  }
  #[test]
  fn normalize_4() {
    assert_case(r#"Normalize[0]; Normalize[{0}]"#, r#"{0}"#);
  }
  #[test]
  fn normalize_5() {
    assert_case(r#"Normalize[0]; Normalize[{0}]; Normalize[{}]"#, r#"{}"#);
  }
  #[test]
  fn vector_angle_4() {
    assert_case(
      r#"Normalize[0]; Normalize[{0}]; Normalize[{}]; VectorAngle[{0, 1}, {0, 1}]"#,
      r#"0"#,
    );
  }
}

mod euler_matrix {
  use super::*;

  #[test]
  fn zero_angles_identity() {
    assert_eq!(
      interpret("EulerMatrix[{0, 0, 0}]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn default_zyz_convention() {
    // Default axis sequence is {3, 2, 3} (ZYZ).
    assert_eq!(
      interpret("EulerMatrix[{Pi/2, 0, 0}]").unwrap(),
      "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn explicit_xyz_convention() {
    // The symbolic entries are sums whose canonical Plus ordering differs
    // harmlessly between engines (e.g. Cos[a]*Sin[c] + Cos[c]*Sin[a]*Sin[b]).
    // Evaluate at numeric angles and Round to pin down every entry exactly
    // while absorbing last-ULP floating-point noise — both engines agree.
    assert_eq!(
      interpret("Round[EulerMatrix[{1., 2., 3.}, {1, 2, 3}], 10^-6]").unwrap(),
      "{{205991/500000, 58727/1000000, 909297/1000000}, \
       {-681243/1000000, -642873/1000000, 14007/40000}, \
       {605127/1000000, -381859/500000, -44969/200000}}"
    );
  }

  #[test]
  fn explicit_zxz_convention() {
    assert_eq!(
      interpret("Round[EulerMatrix[{1., 2., 3.}, {3, 1, 3}], 10^-6]").unwrap(),
      "{{-242739/500000, -422919/1000000, 765147/1000000}, \
       {-43239/50000, 103847/1000000, -98259/200000}, \
       {401/3125, -450099/500000, -416147/1000000}}"
    );
  }

  #[test]
  fn numeric_z_rotation() {
    // Rotation by Pi about z (axis 3): diag-ish with -1 on first two diagonal.
    assert_eq!(
      interpret("EulerMatrix[{Pi, 0, 0}]").unwrap(),
      "{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}}"
    );
  }
}

mod affine_transform {
  use super::*;

  #[test]
  fn matrix_only_builds_transformation_function() {
    // AffineTransform[m] augments m with a zero translation column.
    assert_eq!(
      interpret("AffineTransform[{{1, 2}, {3, 4}}]").unwrap(),
      "TransformationFunction[{{1, 2, 0}, {3, 4, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn matrix_and_vector_builds_transformation_function() {
    // AffineTransform[{m, v}] puts v into the last column.
    assert_eq!(
      interpret("AffineTransform[{{{1, 2}, {3, 4}}, {5, 6}}]").unwrap(),
      "TransformationFunction[{{1, 2, 5}, {3, 4, 6}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn three_by_three_matrix() {
    assert_eq!(
      interpret("AffineTransform[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]").unwrap(),
      "TransformationFunction[\
       {{1, 2, 3, 0}, {4, 5, 6, 0}, {7, 8, 9, 0}, {0, 0, 0, 1}}]"
    );
  }

  #[test]
  fn identity_with_translation_applied_to_origin() {
    // AffineTransform[{I, v}] applied to the origin yields v.
    assert_eq!(
      interpret("AffineTransform[{{{1, 0}, {0, 1}}, {2, 3}}][{0, 0}]").unwrap(),
      "{2, 3}"
    );
  }

  #[test]
  fn applied_to_point_computes_m_dot_p_plus_v() {
    // {{2,0},{0,3}}.{4,5} + {1,1} = {8,15} + {1,1} = {9,16}
    assert_eq!(
      interpret("AffineTransform[{{{2, 0}, {0, 3}}, {1, 1}}][{4, 5}]").unwrap(),
      "{9, 16}"
    );
  }

  #[test]
  fn matrix_only_applied_to_point() {
    // {{1,2},{3,4}}.{1,1} = {3, 7}
    assert_eq!(
      interpret("AffineTransform[{{1, 2}, {3, 4}}][{1, 1}]").unwrap(),
      "{3, 7}"
    );
  }

  #[test]
  fn rescaling_transform_applied_to_point() {
    // (5-0)/10 = 1/2, (5-0)/5 = 1.
    assert_eq!(
      interpret("RescalingTransform[{{0, 10}, {0, 5}}][{5, 5}]").unwrap(),
      "{1/2, 1}"
    );
  }

  #[test]
  fn rescaling_transform_nonzero_min() {
    assert_eq!(
      interpret("RescalingTransform[{{1, 3}, {2, 6}, {0, 4}}][{2, 4, 1}]")
        .unwrap(),
      "{1/2, 1/2, 1/4}"
    );
  }

  #[test]
  fn rescaling_transform_builds_unit_matrix() {
    assert_eq!(
      interpret("RescalingTransform[{{0, 10}, {0, 5}}]").unwrap(),
      "TransformationFunction[{{1/10, 0, 0}, {0, 1/5, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn rescaling_transform_with_target_box_matrix() {
    assert_eq!(
      interpret("RescalingTransform[{{0, 10}, {0, 5}}, {{2, 4}, {0, 100}}]")
        .unwrap(),
      "TransformationFunction[{{1/5, 0, 2}, {0, 20, 0}, {0, 0, 1}}]"
    );
  }

  #[test]
  fn rescaling_transform_with_target_box_applied() {
    assert_eq!(
      interpret(
        "RescalingTransform[{{0, 10}, {0, 5}}, {{2, 4}, {0, 100}}][{5, 5}]"
      )
      .unwrap(),
      "{3, 100}"
    );
  }

  #[test]
  fn rescaling_transform_real_inputs() {
    assert_eq!(
      interpret("RescalingTransform[{{0., 10.}, {0., 5.}}][{5, 5}]").unwrap(),
      "{0.5, 1.}"
    );
  }
}

// LinearFractionalTransform[{m, v, w, b}] is the projective transform
// p |-> (m.p + v)/(w.p + b), built as the augmented matrix {{m, v}, {w, b}}.
mod linear_fractional_transform {
  use super::*;

  // Affine special case (w = 0, b = 1): denominator is 1, so it reduces to
  // m.p + v.
  #[test]
  fn affine_special_case() {
    assert_eq!(
      interpret(
        "LinearFractionalTransform[{{{1, 2}, {3, 4}}, {1, 1}, {0, 0}, 1}][{1, 1}]"
      )
      .unwrap(),
      "{4, 8}"
    );
  }

  // Genuine projective transform: the result is divided by w.p + b.
  #[test]
  fn projective_divide() {
    // m.p = {6, 8}, +v = {7, 9}; w.p + b = 3 + 4 + 2 = 9; {7,9}/9 = {7/9, 1}.
    assert_eq!(
      interpret(
        "LinearFractionalTransform[{{{2, 0}, {0, 2}}, {1, 1}, {1, 1}, 2}][{3, 4}]"
      )
      .unwrap(),
      "{7/9, 1}"
    );
  }

  // The augmented homogeneous matrix is {{m, v}, {w, b}}.
  #[test]
  fn transformation_matrix() {
    assert_eq!(
      interpret(
        "TransformationMatrix[LinearFractionalTransform[{{{1, 2}, {3, 4}}, {1, 1}, {0, 0}, 1}]]"
      )
      .unwrap(),
      "{{1, 2, 1}, {3, 4, 1}, {0, 0, 1}}"
    );
  }

  // A standalone projective TransformationFunction also divides by the
  // homogeneous coordinate from its last row.
  #[test]
  fn transformation_function_projective_divide() {
    assert_eq!(
      interpret(
        "TransformationFunction[{{2, 0, 1}, {0, 2, 1}, {1, 1, 2}}][{3, 4}]"
      )
      .unwrap(),
      "{7/9, 1}"
    );
  }
}

mod orthogonalize {
  use super::*;

  #[test]
  fn already_orthogonal_pair() {
    assert_eq!(
      interpret("Orthogonalize[{{1, 0}, {1, 1}}]").unwrap(),
      "{{1, 0}, {0, 1}}"
    );
  }

  #[test]
  fn rational_orthonormal_pair() {
    assert_eq!(
      interpret("Orthogonalize[{{3, 4}, {1, 0}}]").unwrap(),
      "{{3/5, 4/5}, {4/5, -3/5}}"
    );
  }

  #[test]
  fn radical_orthonormal_pair() {
    assert_eq!(
      interpret("Orthogonalize[{{1, 1}, {1, 0}}]").unwrap(),
      "{{1/Sqrt[2], 1/Sqrt[2]}, {1/Sqrt[2], -(1/Sqrt[2])}}"
    );
  }

  #[test]
  fn linearly_dependent_collapses_to_zero() {
    assert_eq!(
      interpret("Orthogonalize[{{1, 1}, {2, 2}}]").unwrap(),
      "{{1/Sqrt[2], 1/Sqrt[2]}, {0, 0}}"
    );
  }

  #[test]
  fn three_dimensional_pair() {
    assert_eq!(
      interpret("Orthogonalize[{{1, 2, 2}, {2, 1, 2}}]").unwrap(),
      "{{1/3, 2/3, 2/3}, {10/(3*Sqrt[17]), -7/(3*Sqrt[17]), 2/(3*Sqrt[17])}}"
    );
  }

  #[test]
  fn radical_pair_in_three_d() {
    assert_eq!(
      interpret("Orthogonalize[{{1, 1, 0}, {0, 1, 1}}]").unwrap(),
      "{{1/Sqrt[2], 1/Sqrt[2], 0}, {-(1/Sqrt[6]), 1/Sqrt[6], Sqrt[2/3]}}"
    );
  }

  #[test]
  fn zero_vector_stays_zero() {
    assert_eq!(
      interpret("Orthogonalize[{{0, 0}, {1, 0}}]").unwrap(),
      "{{0, 0}, {1, 0}}"
    );
  }

  #[test]
  fn axis_aligned_basis_is_identity() {
    assert_eq!(
      interpret("Orthogonalize[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
  }
}

mod jordan_decomposition {
  use super::*;

  #[test]
  fn diagonal_matrices() {
    // Eigenvalues in Eigenvalues[m] order; basis-vector columns permute
    assert_eq!(
      interpret("JordanDecomposition[{{2, 0}, {0, 3}}]").unwrap(),
      "{{{0, 1}, {1, 0}}, {{3, 0}, {0, 2}}}"
    );
    assert_eq!(
      interpret("JordanDecomposition[{{3, 0}, {0, 2}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {{3, 0}, {0, 2}}}"
    );
    assert_eq!(
      interpret("JordanDecomposition[{{2, 0}, {0, 2}}]").unwrap(),
      "{{{0, 1}, {1, 0}}, {{2, 0}, {0, 2}}}"
    );
  }

  #[test]
  fn distinct_eigenvalues() {
    assert_eq!(
      interpret("JordanDecomposition[{{2, 1}, {1, 2}}]").unwrap(),
      "{{{1, -1}, {1, 1}}, {{3, 0}, {0, 1}}}"
    );
    assert_eq!(
      interpret("JordanDecomposition[{{5, 4}, {1, 2}}]").unwrap(),
      "{{{4, -1}, {1, 1}}, {{6, 0}, {0, 1}}}"
    );
    // Triangular conventions
    assert_eq!(
      interpret("JordanDecomposition[{{2, 1}, {0, 3}}]").unwrap(),
      "{{{1, 1}, {1, 0}}, {{3, 0}, {0, 2}}}"
    );
    assert_eq!(
      interpret("JordanDecomposition[{{5, 0}, {1, 2}}]").unwrap(),
      "{{{3, 0}, {1, 1}}, {{5, 0}, {0, 2}}}"
    );
  }

  #[test]
  fn irrational_eigenvalues() {
    assert_eq!(
      interpret("JordanDecomposition[{{1, 2}, {3, 4}}]").unwrap(),
      "{{{(-3 + Sqrt[33])/6, (-3 - Sqrt[33])/6}, {1, 1}}, {{(5 + Sqrt[33])/2, 0}, {0, (5 - Sqrt[33])/2}}}"
    );
  }

  #[test]
  fn complex_eigenvalues() {
    assert_eq!(
      interpret("JordanDecomposition[{{0, -1}, {1, 0}}]").unwrap(),
      "{{{I, -I}, {1, 1}}, {{I, 0}, {0, -I}}}"
    );
    assert_eq!(
      interpret("JordanDecomposition[{{1, -2}, {1, 1}}]").unwrap(),
      "{{{I*Sqrt[2], -I*Sqrt[2]}, {1, 1}}, {{1 + I*Sqrt[2], 0}, {0, 1 - I*Sqrt[2]}}}"
    );
  }

  #[test]
  fn defective_matrices() {
    assert_eq!(
      interpret("JordanDecomposition[{{1, 1}, {0, 1}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {{1, 1}, {0, 1}}}"
    );
    assert_eq!(
      interpret("JordanDecomposition[{{4, 2}, {0, 4}}]").unwrap(),
      "{{{1, 0}, {0, 1/2}}, {{4, 1}, {0, 4}}}"
    );
    assert_eq!(
      interpret("JordanDecomposition[{{3, 1}, {-1, 1}}]").unwrap(),
      "{{{-1, -1}, {1, 0}}, {{2, 1}, {0, 2}}}"
    );
    assert_eq!(
      interpret("JordanDecomposition[{{2, 0}, {-1, 2}}]").unwrap(),
      "{{{0, -1}, {1, 0}}, {{2, 1}, {0, 2}}}"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    // 3x3 and machine-precision matrices are out of scope
    assert_eq!(
      interpret("JordanDecomposition[{{1, 1, 0}, {0, 1, 0}, {0, 0, 2}}]")
        .unwrap(),
      "JordanDecomposition[{{1, 1, 0}, {0, 1, 0}, {0, 0, 2}}]"
    );
  }
}

mod coordinate_transform {
  use super::*;

  #[test]
  fn polar_cartesian() {
    assert_eq!(
      interpret("CoordinateTransform[\"Polar\" -> \"Cartesian\", {r, t}]")
        .unwrap(),
      "{r*Cos[t], r*Sin[t]}"
    );
    assert_eq!(
      interpret("CoordinateTransform[\"Cartesian\" -> \"Polar\", {x, y}]")
        .unwrap(),
      "{Sqrt[x^2 + y^2], ArcTan[x, y]}"
    );
  }

  #[test]
  fn spherical_cartesian() {
    assert_eq!(
      interpret(
        "CoordinateTransform[\"Spherical\" -> \"Cartesian\", {r, t, p}]"
      )
      .unwrap(),
      "{r*Cos[p]*Sin[t], r*Sin[p]*Sin[t], r*Cos[t]}"
    );
    assert_eq!(
      interpret(
        "CoordinateTransform[\"Cartesian\" -> \"Spherical\", {x, y, z}]"
      )
      .unwrap(),
      "{Sqrt[x^2 + y^2 + z^2], ArcTan[z, Sqrt[x^2 + y^2]], ArcTan[x, y]}"
    );
  }

  #[test]
  fn cylindrical_cartesian() {
    assert_eq!(
      interpret(
        "CoordinateTransform[\"Cylindrical\" -> \"Cartesian\", {r, t, z}]"
      )
      .unwrap(),
      "{r*Cos[t], r*Sin[t], z}"
    );
    assert_eq!(
      interpret(
        "CoordinateTransform[\"Cartesian\" -> \"Cylindrical\", {x, y, z}]"
      )
      .unwrap(),
      "{Sqrt[x^2 + y^2], ArcTan[x, y], z}"
    );
  }

  #[test]
  fn numeric_points_fold() {
    assert_eq!(
      interpret("CoordinateTransform[\"Polar\" -> \"Cartesian\", {1, Pi/4}]")
        .unwrap(),
      "{1/Sqrt[2], 1/Sqrt[2]}"
    );
    assert_eq!(
      interpret("CoordinateTransform[\"Cartesian\" -> \"Polar\", {1, 1}]")
        .unwrap(),
      "{Sqrt[2], Pi/4}"
    );
    assert_eq!(
      interpret(
        "CoordinateTransform[\"Spherical\" -> \"Cartesian\", {2, Pi/2, 0}]"
      )
      .unwrap(),
      "{2, 0, 0}"
    );
  }

  #[test]
  fn unsupported_stays_unevaluated() {
    assert_eq!(
      interpret("CoordinateTransform[\"Bogus\" -> \"Cartesian\", {x, y}]")
        .unwrap(),
      "CoordinateTransform[Bogus -> Cartesian, {x, y}]"
    );
  }
}

mod times_unit_factor {
  use super::*;

  #[test]
  fn unit_factor_preserves_structure() {
    // Regression: Times[1, Cos[Pi/4]] used to re-normalize the result
    // into (Sqrt[2])^(-1) instead of returning 1/Sqrt[2] unchanged
    assert_eq!(interpret("1*Cos[Pi/4]").unwrap(), "1/Sqrt[2]");
    assert_eq!(interpret("1*x").unwrap(), "x");
  }
}

mod hermite_decomposition {
  use super::*;

  #[test]
  fn full_rank_square() {
    assert_eq!(
      interpret("HermiteDecomposition[{{2, 4}, {3, 5}}]").unwrap(),
      "{{{-1, 1}, {3, -2}}, {{1, 1}, {0, 2}}}"
    );
    assert_eq!(
      interpret("HermiteDecomposition[{{6, 4}, {2, 8}}]").unwrap(),
      "{{{0, 1}, {-1, 3}}, {{2, 8}, {0, 20}}}"
    );
    assert_eq!(
      interpret("HermiteDecomposition[{{-3, 5}, {7, -2}}]").unwrap(),
      "{{{2, 1}, {7, 3}}, {{1, 8}, {0, 29}}}"
    );
    // Already in HNF: u is the identity
    assert_eq!(
      interpret("HermiteDecomposition[{{4, 0}, {0, 6}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {{4, 0}, {0, 6}}}"
    );
    assert_eq!(
      interpret("HermiteDecomposition[{{5}}]").unwrap(),
      "{{{1}}, {{5}}}"
    );
    assert_eq!(
      interpret("HermiteDecomposition[{{0, 3}, {2, 0}}]").unwrap(),
      "{{{0, 1}, {1, 0}}, {{2, 0}, {0, 3}}}"
    );
  }

  #[test]
  fn singular_and_rectangular() {
    // Square singular: a zero row appears, the kernel row lands in u
    assert_eq!(
      interpret("HermiteDecomposition[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]")
        .unwrap(),
      "{{{1, 0, 0}, {4, -1, 0}, {1, -2, 1}}, {{1, 2, 3}, {0, 3, 6}, {0, 0, 0}}}"
    );
    // Wide
    assert_eq!(
      interpret("HermiteDecomposition[{{2, 4, 6}}]").unwrap(),
      "{{{1}}, {{2, 4, 6}}}"
    );
    assert_eq!(
      interpret("HermiteDecomposition[{{12, 18, 6}, {8, 4, 10}}]").unwrap(),
      "{{{1, -1}, {2, -3}}, {{4, 14, -4}, {0, 24, -18}}}"
    );
    // Tall rank-1
    assert_eq!(
      interpret("HermiteDecomposition[{{1, 2}, {2, 4}, {3, 6}}]").unwrap(),
      "{{{1, 0, 0}, {-2, 1, 0}, {-3, 0, 1}}, {{1, 2}, {0, 0}, {0, 0}}}"
    );
    // Tall full-column-rank
    assert_eq!(
      interpret("HermiteDecomposition[{{3, 1}, {1, 2}, {4, 3}}]").unwrap(),
      "{{{0, 1, 0}, {-1, 3, 0}, {-1, -1, 1}}, {{1, 2}, {0, 5}, {0, 0}}}"
    );
    assert_eq!(
      interpret("HermiteDecomposition[{{0, 0}, {0, 0}}]").unwrap(),
      "{{{1, 0}, {0, 1}}, {{0, 0}, {0, 0}}}"
    );
  }

  #[test]
  fn non_integer_stays_unevaluated() {
    assert_eq!(
      interpret("HermiteDecomposition[x]").unwrap(),
      "HermiteDecomposition[x]"
    );
    assert_eq!(
      interpret("HermiteDecomposition[{{1.5, 2}, {3, 4}}]").unwrap(),
      "HermiteDecomposition[{{1.5, 2}, {3, 4}}]"
    );
  }
}

mod matsq_messages {
  use super::*;

  // Det/Inverse/Eigenvalues/Eigenvectors emit <F>::matsq for a non-square or
  // non-matrix concrete argument and stay unevaluated, rather than leaking an
  // internal error. Symbolic arguments (x, a + b) stay held with no message.
  fn assert_matsq(input: &str, call: &str, head: &str) {
    clear_state();
    assert_eq!(interpret(input).unwrap(), call);
    let msgs = woxi::get_captured_messages_raw();
    let expected = format!(
      "{head}::matsq: Argument {} at position 1 is not a nonempty square matrix.",
      &call[head.len() + 1..call.len() - 1]
    );
    assert!(
      msgs.iter().any(|m| m.contains(&expected)),
      "expected {expected:?} for {input}, got {msgs:?}"
    );
  }

  #[test]
  fn det_non_square_emits_matsq() {
    assert_matsq(
      "Det[{{1, 2, 3}, {4, 5, 6}}]",
      "Det[{{1, 2, 3}, {4, 5, 6}}]",
      "Det",
    );
  }

  #[test]
  fn det_empty_vector_scalar_emit_matsq() {
    assert_matsq("Det[{}]", "Det[{}]", "Det");
    assert_matsq("Det[{1, 2, 3}]", "Det[{1, 2, 3}]", "Det");
    assert_matsq("Det[5]", "Det[5]", "Det");
    assert_matsq("Det[{{1, 2}, {3}}]", "Det[{{1, 2}, {3}}]", "Det");
  }

  #[test]
  fn eigenvalues_eigenvectors_inverse_non_square() {
    assert_matsq(
      "Eigenvalues[{{1, 2, 3}, {4, 5, 6}}]",
      "Eigenvalues[{{1, 2, 3}, {4, 5, 6}}]",
      "Eigenvalues",
    );
    assert_matsq(
      "Eigenvectors[{{1, 2, 3}, {4, 5, 6}}]",
      "Eigenvectors[{{1, 2, 3}, {4, 5, 6}}]",
      "Eigenvectors",
    );
    assert_matsq("Inverse[{}]", "Inverse[{}]", "Inverse");
  }

  #[test]
  fn symbolic_argument_stays_held_without_message() {
    for input in ["Det[x]", "Det[a + b]", "Eigenvalues[m]", "Inverse[x]"] {
      clear_state();
      let _ = interpret(input).unwrap();
      let msgs = woxi::get_captured_messages_raw();
      assert!(
        msgs.iter().all(|m| !m.contains("::matsq")),
        "unexpected matsq for {input}: {msgs:?}"
      );
    }
  }

  #[test]
  fn valid_square_matrices_unaffected() {
    clear_state();
    assert_eq!(interpret("Det[{{1, 2}, {3, 4}}]").unwrap(), "-2");
    assert_eq!(
      interpret("Eigenvalues[{{2, 0}, {0, 3}}]").unwrap(),
      "{3, 2}"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.iter().all(|m| !m.contains("::matsq")), "{msgs:?}");
  }

  // MatrixPower takes two arguments, so the matsq message reports only the
  // matrix (position 1), not the exponent.
  #[test]
  fn matrix_power_non_square_and_scalar() {
    for (input, arg) in [
      (
        "MatrixPower[{{1, 2, 3}, {4, 5, 6}}, 2]",
        "{{1, 2, 3}, {4, 5, 6}}",
      ),
      ("MatrixPower[{1, 2, 3}, 2]", "{1, 2, 3}"),
      ("MatrixPower[5, 2]", "5"),
    ] {
      clear_state();
      assert_eq!(interpret(input).unwrap(), input);
      let expected = format!(
        "MatrixPower::matsq: Argument {arg} at position 1 is not a nonempty square matrix."
      );
      assert!(
        woxi::get_captured_messages_raw()
          .iter()
          .any(|m| m.contains(&expected)),
        "expected {expected:?} for {input}"
      );
    }
  }

  #[test]
  fn matrix_exp_and_log_non_square() {
    assert_matsq(
      "MatrixExp[{{1, 2, 3}, {4, 5, 6}}]",
      "MatrixExp[{{1, 2, 3}, {4, 5, 6}}]",
      "MatrixExp",
    );
    assert_matsq("MatrixExp[{1, 2}]", "MatrixExp[{1, 2}]", "MatrixExp");
    assert_matsq("MatrixLog[{1, 2, 3}]", "MatrixLog[{1, 2, 3}]", "MatrixLog");
  }

  #[test]
  fn matrix_power_symbolic_and_valid_unaffected() {
    clear_state();
    // A symbol holds (no message); a valid square matrix computes.
    assert_eq!(interpret("MatrixPower[x, 2]").unwrap(), "MatrixPower[x, 2]");
    assert_eq!(
      interpret("MatrixPower[{{1, 2}, {3, 4}}, 2]").unwrap(),
      "{{7, 10}, {15, 22}}"
    );
    assert_eq!(
      interpret("MatrixExp[{{0, 0}, {0, 0}}]").unwrap(),
      "{{1, 0}, {0, 1}}"
    );
    let msgs = woxi::get_captured_messages_raw();
    assert!(msgs.iter().all(|m| !m.contains("::matsq")), "{msgs:?}");
  }
}

mod matrix_minimal_polynomial {
  use super::*;

  // The minimal polynomial is the monic polynomial of least degree annihilating
  // the matrix. Verified against wolframscript.
  #[test]
  fn scalar_multiple_of_identity_is_degree_one() {
    // 2*I has minimal polynomial x - 2 even though its characteristic
    // polynomial is (x - 2)^2.
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{2, 0}, {0, 2}}, x]").unwrap(),
      "-2 + x"
    );
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{5}}, x]").unwrap(),
      "-5 + x"
    );
  }

  #[test]
  fn defective_and_distinct_eigenvalues() {
    // Jordan block: minimal polynomial equals the characteristic polynomial.
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{2, 1}, {0, 2}}, x]").unwrap(),
      "4 - 4*x + x^2"
    );
    // Distinct eigenvalues 1, 2.
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{1, 0}, {0, 2}}, x]").unwrap(),
      "2 - 3*x + x^2"
    );
    // Repeated eigenvalue with one Jordan block of size 2 → degree 2, not 3.
    assert_eq!(
      interpret(
        "MatrixMinimalPolynomial[{{1, 1, 0}, {0, 1, 0}, {0, 0, 1}}, x]"
      )
      .unwrap(),
      "1 - 2*x + x^2"
    );
  }

  #[test]
  fn general_and_rational_entries() {
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{1, 2}, {3, 4}}, x]").unwrap(),
      "-2 - 5*x + x^2"
    );
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{0, 1}, {-1, 0}}, x]").unwrap(),
      "1 + x^2"
    );
    // Rational matrix entries are handled exactly.
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{1, 1/2}, {0, 1}}, x]").unwrap(),
      "1 - 2*x + x^2"
    );
  }

  #[test]
  fn variable_other_than_x() {
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{1, 2}, {3, 4}}, y]").unwrap(),
      "-2 - 5*y + y^2"
    );
  }

  #[test]
  fn non_square_argument_emits_matsq() {
    // A non-(square matrix) first argument — including a bare symbol — emits the
    // matsq message and stays unevaluated. (Unlike MatrixPower, a symbol here is
    // not treated as a potential matrix.)
    for (input, arg) in [
      (
        "MatrixMinimalPolynomial[{{1, 2, 3}, {4, 5, 6}}, x]",
        "{{1, 2, 3}, {4, 5, 6}}",
      ),
      ("MatrixMinimalPolynomial[{1, 2, 3}, x]", "{1, 2, 3}"),
      ("MatrixMinimalPolynomial[5, x]", "5"),
      ("MatrixMinimalPolynomial[a, x]", "a"),
      ("MatrixMinimalPolynomial[{}, x]", "{}"),
    ] {
      clear_state();
      assert_eq!(interpret(input).unwrap(), input);
      let expected = format!(
        "MatrixMinimalPolynomial::matsq: Argument {arg} at position 1 is not \
         a nonempty square matrix."
      );
      assert!(
        woxi::get_captured_messages_raw()
          .iter()
          .any(|m| m.contains(&expected)),
        "expected {expected:?} for {input}"
      );
    }
  }

  // Matrices with symbolic entries return the fully expanded polynomial
  // (matching wolframscript), rather than a factored coefficient form.
  #[test]
  fn symbolic_entries_are_expanded() {
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{a, b}, {c, d}}, x]").unwrap(),
      "-(b*c) + a*d - a*x - d*x + x^2"
    );
    assert_eq!(
      interpret("MatrixMinimalPolynomial[{{a, 1}, {0, a}}, x]").unwrap(),
      "a^2 - 2*a*x + x^2"
    );
  }

  // Regression: a 3x3 matrix with symbolic entries used to overflow i128 in the
  // polynomial-division path and panic. It now returns the expanded cubic.
  #[test]
  fn symbolic_3x3_does_not_overflow() {
    assert_eq!(
      interpret(
        "MatrixMinimalPolynomial[{{p, q, 0}, {0, p, 0}, {0, 0, r}}, x]"
      )
      .unwrap(),
      "-(p^2*r) + p^2*x + 2*p*r*x - 2*p*x^2 - r*x^2 + x^3"
    );
  }
}

mod permutation_matrix {
  use super::*;

  // PermutationMatrix[perm] has a 1 at (i, perm[i]). Woxi returns the dense
  // matrix (wolframscript returns a sparse StructuredArray whose Normal agrees).
  #[test]
  fn permutation_list() {
    assert_eq!(
      interpret("Normal[PermutationMatrix[{2, 3, 1}]]").unwrap(),
      "{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}"
    );
    assert_eq!(
      interpret("Normal[PermutationMatrix[{3, 1, 2}]]").unwrap(),
      "{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}"
    );
    assert_eq!(
      interpret("Normal[PermutationMatrix[{2, 1}]]").unwrap(),
      "{{0, 1}, {1, 0}}"
    );
  }

  // The identity permutation gives the identity matrix.
  #[test]
  fn identity_permutation() {
    assert_eq!(
      interpret("Normal[PermutationMatrix[{1, 2, 3}]]").unwrap(),
      "{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}"
    );
  }

  // A Cycles object is accepted and converted via PermutationList.
  #[test]
  fn cycles_input() {
    assert_eq!(
      interpret("Normal[PermutationMatrix[Cycles[{{1, 2, 3}}]]]").unwrap(),
      "{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}"
    );
    assert_eq!(
      interpret("Normal[PermutationMatrix[Cycles[{{1, 2}}]]]").unwrap(),
      "{{0, 1}, {1, 0}}"
    );
  }

  // Multiplying by the matrix permutes coordinates: P . {a,b,c} reorders.
  #[test]
  fn permutes_vector() {
    assert_eq!(
      interpret("PermutationMatrix[{2, 3, 1}] . {a, b, c}").unwrap(),
      "{b, c, a}"
    );
  }
}

mod vandermonde_matrix {
  use super::*;

  // VandermondeMatrix[{x1,...,xn}] has entry (i, j) equal to xi^(j-1). Woxi
  // returns the dense matrix (wolframscript returns a StructuredArray whose
  // Normal agrees).
  #[test]
  fn symbolic_points() {
    assert_eq!(
      interpret("Normal[VandermondeMatrix[{a, b, c}]]").unwrap(),
      "{{1, a, a^2}, {1, b, b^2}, {1, c, c^2}}"
    );
    assert_eq!(
      interpret("Normal[VandermondeMatrix[{x, y}]]").unwrap(),
      "{{1, x}, {1, y}}"
    );
  }

  #[test]
  fn numeric_points() {
    assert_eq!(
      interpret("Normal[VandermondeMatrix[{2, 3, 5}]]").unwrap(),
      "{{1, 2, 4}, {1, 3, 9}, {1, 5, 25}}"
    );
    assert_eq!(
      interpret("Normal[VandermondeMatrix[{1, 2, 3, 4}]]").unwrap(),
      "{{1, 1, 1, 1}, {1, 2, 4, 8}, {1, 3, 9, 27}, {1, 4, 16, 64}}"
    );
  }

  #[test]
  fn single_point() {
    assert_eq!(
      interpret("Normal[VandermondeMatrix[{7}]]").unwrap(),
      "{{1}}"
    );
  }

  // The determinant follows the Vandermonde product formula
  // Prod_{i<j} (xj - xi); here (3-2)(5-2)(5-3) = 6.
  #[test]
  fn determinant() {
    assert_eq!(interpret("Det[VandermondeMatrix[{2, 3, 5}]]").unwrap(), "6");
    assert_eq!(
      interpret("Det[VandermondeMatrix[{1, 2, 4, 8}]]").unwrap(),
      "1008"
    );
  }
}

mod companion_matrix {
  use super::*;

  // CompanionMatrix[{c0,...,c_{n-1}}] is the companion matrix of the monic
  // polynomial x^n + c_{n-1} x^{n-1} + ... + c0: 1's on the subdiagonal and
  // -c_i down the last column.
  #[test]
  fn coefficient_list() {
    assert_eq!(
      interpret("CompanionMatrix[{2, 3, 1}]").unwrap(),
      "{{0, 0, -2}, {1, 0, -3}, {0, 1, -1}}"
    );
    assert_eq!(
      interpret("CompanionMatrix[{1, 1}]").unwrap(),
      "{{0, -1}, {1, -1}}"
    );
    assert_eq!(
      interpret("CompanionMatrix[{6, -5, 0, 1}]").unwrap(),
      "{{0, 0, 0, -6}, {1, 0, 0, 5}, {0, 1, 0, 0}, {0, 0, 1, -1}}"
    );
  }

  #[test]
  fn single_coefficient() {
    assert_eq!(interpret("CompanionMatrix[{5}]").unwrap(), "{{-5}}");
  }

  #[test]
  fn symbolic_coefficients() {
    assert_eq!(
      interpret("CompanionMatrix[{a, b, c}]").unwrap(),
      "{{0, 0, -a}, {1, 0, -b}, {0, 1, -c}}"
    );
  }

  // The characteristic polynomial recovers the defining coefficients.
  #[test]
  fn characteristic_polynomial() {
    assert_eq!(
      interpret("CharacteristicPolynomial[CompanionMatrix[{2, 3, 1}], x]")
        .unwrap(),
      "-2 - 3*x - x^2 - x^3"
    );
  }
}

mod roll_pitch_yaw_matrix_tests {
  use woxi::interpret;

  // R_x(γ).R_y(β).R_z(α) — rotations applied in reverse order of the angle
  // list (wolframscript's default {3, 2, 1} convention).
  #[test]
  fn exact_angles() {
    assert_eq!(
      interpret("RollPitchYawMatrix[{Pi/2, 0, Pi/2}]").unwrap(),
      "{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}}"
    );
    assert_eq!(
      interpret("RollPitchYawMatrix[{Pi/4, 0, 0}]").unwrap(),
      "{{1/Sqrt[2], -(1/Sqrt[2]), 0}, {1/Sqrt[2], 1/Sqrt[2], 0}, {0, 0, 1}}"
    );
  }

  #[test]
  fn explicit_axis_sequence() {
    assert_eq!(
      interpret("RollPitchYawMatrix[{Pi/2, Pi/2, 0}, {1, 2, 3}]").unwrap(),
      "{{0, 1, 0}, {0, 0, -1}, {-1, 0, 0}}"
    );
  }

  // The identity RollPitchYawMatrix[{α,β,γ}, {p,q,r}] =
  // EulerMatrix[{γ,β,α}, {r,q,p}], verified symbolically in wolframscript.
  #[test]
  fn euler_matrix_delegation() {
    assert_eq!(
      interpret(
        "RollPitchYawMatrix[{a, b, c}] === EulerMatrix[{c, b, a}, {1, 2, 3}]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "RollPitchYawMatrix[{a, b, c}, {3, 1, 3}] === EulerMatrix[{c, b, a}, {3, 1, 3}]"
      )
      .unwrap(),
      "True"
    );
  }

  // A non-list argument emits ::ang and echoes.
  #[test]
  fn invalid_angles() {
    assert_eq!(
      interpret("RollPitchYawMatrix[a]").unwrap(),
      "RollPitchYawMatrix[a]"
    );
  }
}

mod lyapunov_solve_tests {
  use woxi::interpret;

  // LyapunovSolve[a, c] solves a.x + x.aᵀ == c exactly over the rationals.
  #[test]
  fn continuous() {
    assert_eq!(
      interpret("LyapunovSolve[{{1, 2}, {0, 3}}, {{1, 0}, {0, 1}}]").unwrap(),
      "{{2/3, -1/12}, {-1/12, 1/6}}"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{-1, 0}, {0, -2}}, {{4, 2}, {2, 4}}]").unwrap(),
      "{{-2, -2/3}, {-2/3, -1}}"
    );
    // Machine reals convert exactly and round back, matching wolframscript
    // to the last digit.
    assert_eq!(
      interpret("LyapunovSolve[{{1., 2.}, {0., 3.}}, {{1., 0.}, {0., 1.}}]")
        .unwrap(),
      "{{0.6666666666666666, -0.08333333333333333}, {-0.08333333333333333, 0.16666666666666666}}"
    );
  }

  // The three-argument form is the general Sylvester equation
  // a.x + x.b == c (b untransposed), allowing rectangular c.
  #[test]
  fn sylvester() {
    assert_eq!(
      interpret(
        "LyapunovSolve[{{1, 2}, {0, 3}}, {{4, 5}, {6, 7}}, {{1, 0}, {0, 1}}]"
      )
      .unwrap(),
      "{{5/4, -33/40}, {-3/20, 7/40}}"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{1, 0}, {0, 2}}, {{3}}, {{1}, {1}}]").unwrap(),
      "{{1/4}, {1/5}}"
    );
  }

  // DiscreteLyapunovSolve[a, c] solves a.x.aᵀ - x == c; the 3-argument form
  // solves a.x.b - x == c.
  #[test]
  fn discrete() {
    assert_eq!(
      interpret(
        "DiscreteLyapunovSolve[{{1/2, 0}, {0, 1/3}}, {{1, 0}, {0, 1}}]"
      )
      .unwrap(),
      "{{-4/3, 0}, {0, -9/8}}"
    );
    assert_eq!(
      interpret("DiscreteLyapunovSolve[{{0, 1}, {-1/2, 0}}, {{1, 2}, {2, 1}}]")
        .unwrap(),
      "{{-8/3, -4/3}, {-4/3, -5/3}}"
    );
    assert_eq!(
      interpret(
        "DiscreteLyapunovSolve[{{1/2, 0}, {0, 1/3}}, {{1/4, 0}, {0, 1/5}}, {{1, 0}, {0, 1}}]"
      )
      .unwrap(),
      "{{-8/7, 0}, {0, -15/14}}"
    );
  }

  // Singular systems emit ::nosol; shape errors emit ::matsq/::ndims;
  // non-diagonal symbolic matrices stay unevaluated (wolframscript's
  // general symbolic path produces large unsimplified Conjugate quotients
  // that are out of scope).
  #[test]
  fn error_forms() {
    assert_eq!(
      interpret("LyapunovSolve[{{1, -1}, {1, -1}}, {{1, 0}, {0, 1}}]").unwrap(),
      "LyapunovSolve[{{1, -1}, {1, -1}}, {{1, 0}, {0, 1}}]"
    );
    assert_eq!(
      interpret("DiscreteLyapunovSolve[{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}]")
        .unwrap(),
      "DiscreteLyapunovSolve[{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}]"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{1, 2}}, {{1, 0}, {0, 1}}]").unwrap(),
      "LyapunovSolve[{{1, 2}}, {{1, 0}, {0, 1}}]"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{1, 2}, {0, 3}}, {{1, 0}}]").unwrap(),
      "LyapunovSolve[{{1, 2}, {0, 3}}, {{1, 0}}]"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{a, 1}, {0, b}}, {{1, 0}, {0, 1}}]").unwrap(),
      "LyapunovSolve[{{a, 1}, {0, b}}, {{1, 0}, {0, 1}}]"
    );
  }

  // A diagonal symbolic first matrix decouples entrywise into
  // wolframscript's Conjugate closed forms.
  #[test]
  fn symbolic_diagonal() {
    assert_eq!(
      interpret("LyapunovSolve[{{a, 0}, {0, b}}, {{1, 0}, {0, 1}}]").unwrap(),
      "{{(a + Conjugate[a])^(-1), 0}, {0, (b + Conjugate[b])^(-1)}}"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{a, 0}, {0, b}}, {{c, 0}, {0, d}}]").unwrap(),
      "{{c/(a + Conjugate[a]), 0}, {0, d/(b + Conjugate[b])}}"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{a, 0}, {0, b}}, {{0, 1}, {2, 0}}]").unwrap(),
      "{{0, (a + Conjugate[b])^(-1)}, {2/(b + Conjugate[a]), 0}}"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{a}}, {{c}}]").unwrap(),
      "{{c/(a + Conjugate[a])}}"
    );
    assert_eq!(
      interpret("LyapunovSolve[{{2, 0}, {0, 3}}, {{c, 0}, {0, d}}]").unwrap(),
      "{{c/4, 0}, {0, d/6}}"
    );
    assert_eq!(
      interpret("DiscreteLyapunovSolve[{{a, 0}, {0, b}}, {{1, 0}, {0, 1}}]")
        .unwrap(),
      "{{(-1 + a*Conjugate[a])^(-1), 0}, {0, (-1 + b*Conjugate[b])^(-1)}}"
    );
  }
}

mod modulus_option {
  use super::*;

  #[test]
  fn det_reduces_mod_m() {
    assert_eq!(
      interpret("Det[{{1, 2}, {3, 4}}, Modulus -> 5]").unwrap(),
      "3"
    );
    assert_eq!(
      interpret("Det[{{2, 3, 1}, {0, 4, 6}, {1, 1, 1}}, Modulus -> 5]")
        .unwrap(),
      "0"
    );
    // Composite moduli are fine for Det, and Modulus -> 0 means
    // ordinary arithmetic
    assert_eq!(
      interpret("Det[{{1, 2}, {3, 4}}, Modulus -> 6]").unwrap(),
      "4"
    );
    assert_eq!(
      interpret("Det[{{1, 2}, {3, 4}}, Modulus -> 0]").unwrap(),
      "-2"
    );
  }

  #[test]
  fn inverse_mod_m() {
    assert_eq!(
      interpret("Inverse[{{1, 2}, {3, 4}}, Modulus -> 5]").unwrap(),
      "{{3, 1}, {4, 2}}"
    );
    // Composite modulus works when the determinant is invertible
    assert_eq!(
      interpret("Inverse[{{1, 2}, {3, 4}}, Modulus -> 9]").unwrap(),
      "{{7, 1}, {6, 4}}"
    );
  }

  // A singular matrix emits `sing` showing the mod-reduced matrix.
  #[test]
  fn inverse_singular_emits_sing() {
    let result =
      woxi::interpret_with_stdout("Inverse[{{1, 3}, {2, 6}}, Modulus -> 5]")
        .unwrap();
    assert_eq!(result.result, "Inverse[{{1, 3}, {2, 6}}, Modulus -> 5]");
    assert!(
      result.warnings.iter().any(
        |w| w.contains("Inverse::sing: Matrix {{1, 3}, {2, 1}} is singular.")
      ),
      "expected sing, got {:?}",
      result.warnings
    );
  }

  #[test]
  fn row_reduce_mod_p() {
    assert_eq!(
      interpret("RowReduce[{{1, 2, 3}, {4, 5, 6}}, Modulus -> 7]").unwrap(),
      "{{1, 0, 6}, {0, 1, 2}}"
    );
    assert_eq!(
      interpret("RowReduce[{{2, 4}, {1, 3}}, Modulus -> 5]").unwrap(),
      "{{1, 0}, {0, 1}}"
    );
    assert_eq!(
      interpret("RowReduce[{{0, 0}, {0, 0}}, Modulus -> 3]").unwrap(),
      "{{0, 0}, {0, 0}}"
    );
  }

  #[test]
  fn row_reduce_rejects_composite_modulus() {
    let result =
      woxi::interpret_with_stdout("RowReduce[{{2, 4}, {1, 3}}, Modulus -> 6]")
        .unwrap();
    assert_eq!(result.result, "RowReduce[{{2, 4}, {1, 3}}, Modulus -> 6]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "RowReduce::nmod: {{2, 4}, {1, 3}} cannot be reduced in non-prime \
         modulus 6."
      )),
      "expected nmod, got {:?}",
      result.warnings
    );
  }

  #[test]
  fn null_space_mod_m() {
    assert_eq!(
      interpret("NullSpace[{{1, 2}, {2, 4}}, Modulus -> 5]").unwrap(),
      "{{3, 1}}"
    );
    assert_eq!(
      interpret("NullSpace[{{1, 2, 3}, {2, 4, 6}, {0, 1, 1}}, Modulus -> 7]")
        .unwrap(),
      "{{6, 6, 1}}"
    );
    assert_eq!(
      interpret("NullSpace[{{1, 0}, {0, 1}}, Modulus -> 5]").unwrap(),
      "{}"
    );
    // Free columns are taken in reverse order with the free variable 1
    assert_eq!(
      interpret("NullSpace[{{1, 2, 3}}, Modulus -> 5]").unwrap(),
      "{{2, 0, 1}, {3, 1, 0}}"
    );
    // Composite moduli work when the pivots stay invertible
    assert_eq!(
      interpret("NullSpace[{{1, 2}, {2, 4}}, Modulus -> 6]").unwrap(),
      "{{4, 1}}"
    );
  }

  #[test]
  fn linear_solve_mod_p() {
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {3, 4}}, {5, 6}, Modulus -> 7]").unwrap(),
      "{3, 1}"
    );
    // Underdetermined systems set the free variables to 0
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {2, 4}}, {3, 6}, Modulus -> 5]").unwrap(),
      "{3, 0}"
    );
    assert_eq!(
      interpret("LinearSolve[{{1, 2, 3}}, {4}, Modulus -> 5]").unwrap(),
      "{4, 0, 0}"
    );
    // A matrix right-hand side solves column-wise
    assert_eq!(
      interpret("LinearSolve[{{1, 2}, {2, 4}}, {{3}, {6}}, Modulus -> 5]")
        .unwrap(),
      "{{3}, {0}}"
    );
  }

  #[test]
  fn linear_solve_inconsistent_emits_nosol() {
    let result = woxi::interpret_with_stdout(
      "LinearSolve[{{1, 2}, {2, 4}}, {3, 7}, Modulus -> 5]",
    )
    .unwrap();
    assert_eq!(
      result.result,
      "LinearSolve[{{1, 2}, {2, 4}}, {3, 7}, Modulus -> 5]"
    );
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "LinearSolve::nosol: Linear equation encountered that has no \
         solution."
      )),
      "expected nosol, got {:?}",
      result.warnings
    );
  }

  #[test]
  fn matrix_rank_mod_p() {
    assert_eq!(
      interpret("MatrixRank[{{1, 2}, {3, 6}}, Modulus -> 3]").unwrap(),
      "1"
    );
    assert_eq!(
      interpret("MatrixRank[{{1, 2}, {3, 4}}, Modulus -> 5]").unwrap(),
      "2"
    );
    assert_eq!(
      interpret("MatrixRank[{{0, 0}, {0, 0}}, Modulus -> 3]").unwrap(),
      "0"
    );
  }

  #[test]
  fn matrix_rank_rejects_composite_modulus() {
    let result =
      woxi::interpret_with_stdout("MatrixRank[{{2, 4}, {1, 3}}, Modulus -> 6]")
        .unwrap();
    assert_eq!(result.result, "MatrixRank[{{2, 4}, {1, 3}}, Modulus -> 6]");
    assert!(
      result.warnings.iter().any(|w| w.contains(
        "MatrixRank::modp: The value of the option Modulus -> 6 should be \
         a prime number or zero."
      )),
      "expected modp, got {:?}",
      result.warnings
    );
  }
}
