// Miscellaneous structural and arithmetic argument forms found missing by the
// reference-page parameter sweep (round 4). Expected values are the ones the
// Wolfram reference pages show, or follow from the definitions.

use super::*;

mod inactivate_head_pattern {
  use super::*;

  #[test]
  fn alternatives_of_heads() {
    assert_eq!(
      interpret("Inactivate[f[g[h[x]], y, g[z]], g | h]").unwrap(),
      "f[Inactive[g][Inactive[h][x]], y, Inactive[g][z]]"
    );
  }

  #[test]
  fn except_a_head() {
    assert_eq!(
      interpret("Inactivate[Integrate[Sin[x] + Cos[x], x], Except[Integrate]]")
        .unwrap(),
      "Integrate[Inactive[Plus][Inactive[Sin][x], Inactive[Cos][x]], x]"
    );
    assert_eq!(
      interpret(
        "Activate[Inactivate[Integrate[Sin[x] + Cos[x], x], Except[Integrate]]]"
      )
      .unwrap(),
      "-Cos[x] + Sin[x]"
    );
  }

  #[test]
  fn a_single_head_still_works() {
    assert_eq!(
      interpret("Inactivate[f[g[x], h[g[y]]], g]").unwrap(),
      "f[Inactive[g][x], h[Inactive[g][y]]]"
    );
  }

  #[test]
  fn a_blank_pattern_inactivates_every_head() {
    assert_eq!(
      interpret("Inactivate[f[g[x]], _]").unwrap(),
      "Inactive[f][Inactive[g][x]]"
    );
  }
}

mod odd_rounding_functions {
  use super::*;

  #[test]
  fn negated_argument_is_pulled_out() {
    assert_eq!(interpret("IntegerPart[-x]").unwrap(), "-IntegerPart[x]");
    assert_eq!(
      interpret("FractionalPart[-x]").unwrap(),
      "-FractionalPart[x]"
    );
    assert_eq!(interpret("Round[-x]").unwrap(), "-Round[x]");
    assert_eq!(interpret("Round[-2 x]").unwrap(), "-Round[2*x]");
    assert_eq!(interpret("IntegerPart[-x/3]").unwrap(), "-IntegerPart[x/3]");
  }

  #[test]
  fn odd_function_identities_hold() {
    assert_eq!(
      interpret("IntegerPart[-x] == -IntegerPart[x]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("FractionalPart[-x] == -FractionalPart[x]").unwrap(),
      "True"
    );
    assert_eq!(interpret("Round[-x] == -Round[x]").unwrap(), "True");
  }

  #[test]
  fn numbers_keep_their_values() {
    assert_eq!(interpret("Round[-5/2]").unwrap(), "-2");
    assert_eq!(interpret("Round[-Pi]").unwrap(), "-3");
    assert_eq!(interpret("IntegerPart[-7/2]").unwrap(), "-3");
    assert_eq!(interpret("FractionalPart[-7/2]").unwrap(), "-1/2");
    assert_eq!(interpret("Round[-2.5]").unwrap(), "-2");
  }

  #[test]
  fn a_non_negated_symbol_stays() {
    assert_eq!(interpret("Round[a - b]").unwrap(), "Round[a - b]");
    assert_eq!(interpret("IntegerPart[x]").unwrap(), "IntegerPart[x]");
  }
}

mod length_while_any_head {
  use super::*;

  #[test]
  fn counts_leading_arguments_of_a_normal_expression() {
    assert_eq!(
      interpret("LengthWhile[f[1, 1, 2, 3, a, 8, 13, 21], IntegerQ]").unwrap(),
      "4"
    );
    assert_eq!(interpret("LengthWhile[g[a, 1], IntegerQ]").unwrap(), "0");
  }
}

mod permutation_product_symbolic {
  use super::*;

  #[test]
  fn a_permutation_cancels_its_inverse() {
    assert_eq!(
      interpret("PermutationProduct[a, InversePermutation[a]]").unwrap(),
      "Cycles[{}]"
    );
    assert_eq!(
      interpret("PermutationProduct[a, b, a, a, a, a, InversePermutation[a]]")
        .unwrap(),
      "PermutationProduct[a, b, PermutationPower[a, 3]]"
    );
  }

  #[test]
  fn adjacent_cycles_compose() {
    assert_eq!(
      interpret(
        "PermutationProduct[a, b, Cycles[{{1, 2, 3}}], Cycles[{{2, 3}}], c]"
      )
      .unwrap(),
      "PermutationProduct[a, b, Cycles[{{1, 3}}], c]"
    );
  }

  #[test]
  fn irreducible_products_stay() {
    assert_eq!(
      interpret("PermutationProduct[a, b]").unwrap(),
      "PermutationProduct[a, b]"
    );
  }
}

mod minors_through_det {
  use super::*;

  #[test]
  fn full_order_minor_is_the_determinant() {
    assert_eq!(
      interpret(
        "m = Array[Subscript[x, ##] &, {4, 4}]; Minors[m, 4] == {{Det[m]}}"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "m = Array[Subscript[x, ##] &, {4, 4}]; Minors[m, 3] == Minors[m]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn a_pure_function_is_applied_to_each_submatrix() {
    assert_eq!(
      interpret("Minors[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}, 2, Total[#, 2] &]")
        .unwrap(),
      "{{12, 14, 16}, {18, 21, 23}, {24, 27, 29}}"
    );
    assert_eq!(
      interpret("Minors[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}, 2, Tr]").unwrap(),
      "{{6, 7, 8}, {9, 11, 12}, {12, 14, 15}}"
    );
  }

  #[test]
  fn numeric_minors_are_unchanged() {
    assert_eq!(
      interpret("Minors[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}]").unwrap(),
      "{{-3, -6, -3}, {-6, -11, -4}, {-3, -2, 2}}"
    );
  }
}

mod subgraph_directed {
  use super::*;

  #[test]
  fn rule_list_is_a_graph() {
    assert_eq!(
      interpret(
        "Sort[EdgeList[Subgraph[{1 -> 3, 2 -> 1, 3 -> 6, 4 -> 6, 1 -> 5, \
         5 -> 4, 6 -> 1}, {1, 3, 6}]]]"
      )
      .unwrap(),
      interpret(
        "Sort[{DirectedEdge[1, 3], DirectedEdge[3, 6], DirectedEdge[6, 1]}]"
      )
      .unwrap()
    );
    assert_eq!(
      interpret(
        "VertexList[Subgraph[{1 -> 3, 2 -> 1, 3 -> 6, 4 -> 6, 1 -> 5, \
         5 -> 4, 6 -> 1}, {1, 3, 6}]]"
      )
      .unwrap(),
      "{1, 3, 6}"
    );
  }

  #[test]
  fn directed_edges_of_a_graph_are_kept() {
    assert_eq!(
      interpret(
        "EdgeCount[Subgraph[Graph[{1 -> 2, 2 -> 3, 3 -> 1, 3 -> 4}], {1, 2, 3}]]"
      )
      .unwrap(),
      "3"
    );
    assert_eq!(
      interpret(
        "DirectedGraphQ[Subgraph[Graph[{1 -> 2, 2 -> 3, 3 -> 4}], {1, 2}]]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod string_template_defaults {
  use super::*;

  #[test]
  fn defaults_fill_missing_named_slots() {
    assert_eq!(
      interpret(
        "t = StringTemplate[\"a is `a`, b is `b`\", <|\"a\" -> 5|>]; \
         TemplateApply[t, <|\"b\" -> 2|>]"
      )
      .unwrap(),
      "a is 5, b is 2"
    );
    assert_eq!(
      interpret(
        "t = StringTemplate[\"a is `a`, b is `b`\", <|\"a\" -> 5|>]; \
         TemplateApply[t, <|\"a\" -> 1, \"b\" -> 2|>]"
      )
      .unwrap(),
      "a is 1, b is 2"
    );
    assert_eq!(
      interpret(
        "t = StringTemplate[\"`a` and `b`\", <|\"a\" -> 5, \"b\" -> 6|>]; \
         TemplateApply[t]"
      )
      .unwrap(),
      "5 and 6"
    );
    assert_eq!(
      interpret(
        "t = StringTemplate[\"x `a` `b`\", <|\"a\" -> 5|>]; t[<|\"b\" -> 3|>]"
      )
      .unwrap(),
      "x 5 3"
    );
  }

  #[test]
  fn delayed_defaults_are_evaluated_on_application() {
    assert_eq!(
      interpret(
        "SeedRandom[1]; t = StringTemplate[\"a is `a`, b is `b`\", \
         <|\"a\" :> RandomInteger[{7, 7}]|>]; TemplateApply[t, <|\"b\" -> 2|>]"
      )
      .unwrap(),
      "a is 7, b is 2"
    );
    assert_eq!(
      interpret(
        "t = StringTemplate[\"a is `a`, b is `b`\", <|\"a\" :> RandomReal[]|>]; \
         StringMatchQ[TemplateApply[t, <|\"b\" -> 2|>], \
         \"a is 0.\" ~~ DigitCharacter.. ~~ \", b is 2\"]"
      )
      .unwrap(),
      "True"
    );
  }

  #[test]
  fn machine_reals_are_inserted_as_text() {
    assert_eq!(
      interpret("TemplateApply[\"`a`\", <|\"a\" -> 0.123456789|>]").unwrap(),
      "0.123457"
    );
    assert_eq!(
      interpret("StringTemplate[\"`a`\"][<|\"a\" -> 0.123456789|>]").unwrap(),
      "0.123457"
    );
  }
}

mod orthogonalize_inner_product {
  use super::*;

  #[test]
  fn legendre_polynomials_from_an_integral_inner_product() {
    let expected = "{1/Sqrt[2], Sqrt[3/2]*x, (3*Sqrt[5/2]*(-1/3 + x^2))/2, \
                    (5*Sqrt[7/2]*((-3*x)/5 + x^3))/2}";
    assert_eq!(
      interpret(
        "Orthogonalize[{1, x, x^2, x^3}, Integrate[#1 #2, {x, -1, 1}] &]"
      )
      .unwrap(),
      expected
    );
    assert_eq!(
      interpret(
        "ip[f_, g_] := Integrate[f g, {x, -1, 1}]; \
         Orthogonalize[{1, x, x^2, x^3}, ip]"
      )
      .unwrap(),
      expected
    );
  }

  #[test]
  fn dot_and_a_pure_dot_match_the_default_on_real_vectors() {
    let expected = "{{1/Sqrt[5], 2/Sqrt[5]}, {2/Sqrt[5], -(1/Sqrt[5])}}";
    assert_eq!(
      interpret("Orthogonalize[{{1, 2}, {3, 4}}, Dot]").unwrap(),
      expected
    );
    assert_eq!(
      interpret("Orthogonalize[{{1, 2}, {3, 4}}, #1 . #2 &]").unwrap(),
      expected
    );
  }

  #[test]
  fn a_dependent_vector_becomes_zero() {
    assert_eq!(
      interpret("Orthogonalize[{{1, 2}, {2, 4}}, #1 . #2 &]").unwrap(),
      "{{1/Sqrt[5], 2/Sqrt[5]}, {0, 0}}"
    );
  }
}

mod correlation_complex {
  use super::*;

  #[test]
  fn exact_complex_vectors() {
    // Cov = -7/3 + 56 I/3, Var[x] = 34/3, Var[y] = 134/3.
    assert_eq!(
      interpret(
        "Simplify[Correlation[{2 + I, 3 - 2 I, 5 + 4 I}, {I, 1 + 2 I, \
         10 - 5 I}] - (-7 + 56 I)/(2 Sqrt[1139])]"
      )
      .unwrap(),
      "0"
    );
    assert_eq!(
      interpret(
        "Chop[N[Correlation[{2 + I, 3 - 2 I, 5 + 4 I}, {I, 1 + 2 I, \
         10 - 5 I}]] - (-0.10370654891554296 + 0.8296523913243437 I)]"
      )
      .unwrap(),
      "0"
    );
  }

  #[test]
  fn real_vectors_are_unchanged() {
    assert_eq!(
      interpret("Correlation[{1, 2, 3}, {1, 5, 7}]").unwrap(),
      "(3*Sqrt[3/7])/2"
    );
  }
}

mod dispatch_association {
  use super::*;

  #[test]
  fn an_association_dispatch_table_replaces() {
    assert_eq!(
      interpret("{a, c, e} /. Dispatch[<|a -> b, c -> d|>]").unwrap(),
      "{b, d, e}"
    );
    assert_eq!(
      interpret(
        "{1, 2, 3.5} /. Dispatch[{1 -> a, 2 -> b, c_?NumericQ -> Infinity}]"
      )
      .unwrap(),
      "{a, b, Infinity}"
    );
  }
}

mod kronecker_delta_lists {
  use super::*;

  #[test]
  fn lists_compare_as_wholes() {
    assert_eq!(interpret("KroneckerDelta[{1, 0}, {0, 1}]").unwrap(), "0");
    assert_eq!(interpret("KroneckerDelta[{1, 0}, {1, 0}]").unwrap(), "1");
    assert_eq!(interpret("KroneckerDelta[{1, 0}, {1}]").unwrap(), "0");
  }

  #[test]
  fn undecidable_symbolic_lists_stay() {
    assert_eq!(
      interpret("KroneckerDelta[{a}, {b}]").unwrap(),
      "KroneckerDelta[{a}, {b}]"
    );
  }
}

mod nothing_as_a_head {
  use super::*;

  #[test]
  fn reduces_to_nothing() {
    assert_eq!(interpret("Nothing[a]").unwrap(), "Nothing");
    assert_eq!(interpret("Nothing[a, b]").unwrap(), "Nothing");
    assert_eq!(interpret("{1, Nothing[2], 3}").unwrap(), "{1, 3}");
  }
}

mod dispatch_of_association {
  use super::*;

  #[test]
  fn becomes_a_rule_list_dispatch() {
    assert_eq!(
      interpret("Dispatch[<|a -> b, c -> d|>]").unwrap(),
      "Dispatch[{a -> b, c -> d}]"
    );
    assert_eq!(
      interpret("Dispatch[<|a :> b|>]").unwrap(),
      "Dispatch[{a :> b}]"
    );
  }
}

mod template_unfilled_slot {
  use super::*;

  #[test]
  fn contributes_nothing() {
    assert_eq!(
      interpret(
        "TemplateApply[StringTemplate[\"a is `a`, b is `b`\", \
         <|\"a\" -> 1|>]]"
      )
      .unwrap(),
      "a is 1, b is "
    );
    assert_eq!(
      interpret("TemplateApply[\"`a` and `b`\", <|\"a\" -> 1|>]").unwrap(),
      "1 and "
    );
  }
}

mod rescale_symbolic_bounds {
  use super::*;

  #[test]
  fn value_at_zero_and_at_the_endpoint() {
    assert_eq!(interpret("Rescale[0, {a, b}]").unwrap(), "-(a/(-a + b))");
    assert_eq!(interpret("Rescale[a, {a, b}]").unwrap(), "0");
    assert_eq!(interpret("Rescale[a, {b, a}]").unwrap(), "1");
    assert_eq!(
      interpret("Rescale[0, {a, b}, {c, d}]").unwrap(),
      "-((-(b*c) + a*d)/(-a + b))"
    );
  }

  #[test]
  fn complex_bounds() {
    assert_eq!(
      interpret("Rescale[1 + 2 I, {0, 1 + I}]").unwrap(),
      "3/2 + I/2"
    );
  }

  #[test]
  fn numeric_bounds_are_unchanged() {
    assert_eq!(interpret("Rescale[3, {1, 5}]").unwrap(), "1/2");
    assert_eq!(interpret("Rescale[3, {1, 5}, {0, 10}]").unwrap(), "5");
  }
}

mod orthogonalize_dot_unsimplified {
  use super::*;

  #[test]
  fn projections_use_the_unnormalized_vectors() {
    // Wolfram's symbolic Dot result keeps `x - (a (a x + b y))/(a^2 + b^2)`
    // unsimplified; check the components that do not involve the
    // (differently ordered) norm.
    assert_eq!(
      interpret("Orthogonalize[{{a, b}, {x, y}}, Dot][[1]]").unwrap(),
      "{a/Sqrt[a^2 + b^2], b/Sqrt[a^2 + b^2]}"
    );
    assert_eq!(
      interpret("Numerator[Orthogonalize[{{a, b}, {x, y}}, Dot][[2, 1]]]")
        .unwrap(),
      "x - (a*(a*x + b*y))/(a^2 + b^2)"
    );
  }
}

mod permutation_power_runs {
  use super::*;

  #[test]
  fn equal_factors_become_a_power() {
    assert_eq!(
      interpret("PermutationProduct[a, a]").unwrap(),
      "PermutationPower[a, 2]"
    );
  }
}
