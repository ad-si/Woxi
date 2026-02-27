use super::*;

mod sign_predicates {
  use super::*;

  #[test]
  fn positive() {
    assert_eq!(interpret("Positive[5]").unwrap(), "True");
    assert_eq!(interpret("Positive[-3]").unwrap(), "False");
    assert_eq!(interpret("Positive[0]").unwrap(), "False");
  }

  #[test]
  fn negative() {
    assert_eq!(interpret("Negative[-5]").unwrap(), "True");
    assert_eq!(interpret("Negative[3]").unwrap(), "False");
    assert_eq!(interpret("Negative[0]").unwrap(), "False");
  }

  #[test]
  fn non_positive() {
    assert_eq!(interpret("NonPositive[-5]").unwrap(), "True");
    assert_eq!(interpret("NonPositive[0]").unwrap(), "True");
    assert_eq!(interpret("NonPositive[3]").unwrap(), "False");
  }

  #[test]
  fn non_negative() {
    assert_eq!(interpret("NonNegative[5]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[0]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[-3]").unwrap(), "False");
  }

  #[test]
  fn positive_constants() {
    assert_eq!(interpret("Positive[Pi]").unwrap(), "True");
    assert_eq!(interpret("Positive[E]").unwrap(), "True");
    assert_eq!(interpret("Positive[Infinity]").unwrap(), "True");
    assert_eq!(interpret("Positive[-Pi]").unwrap(), "False");
    assert_eq!(interpret("Positive[-E]").unwrap(), "False");
  }

  #[test]
  fn negative_constants() {
    assert_eq!(interpret("Negative[-Pi]").unwrap(), "True");
    assert_eq!(interpret("Negative[-E]").unwrap(), "True");
    assert_eq!(interpret("Negative[Pi]").unwrap(), "False");
    assert_eq!(interpret("Negative[E]").unwrap(), "False");
    assert_eq!(interpret("Negative[Infinity]").unwrap(), "False");
  }

  #[test]
  fn non_positive_constants() {
    assert_eq!(interpret("NonPositive[-Pi]").unwrap(), "True");
    assert_eq!(interpret("NonPositive[Pi]").unwrap(), "False");
  }

  #[test]
  fn non_negative_constants() {
    assert_eq!(interpret("NonNegative[Pi]").unwrap(), "True");
    assert_eq!(interpret("NonNegative[-Pi]").unwrap(), "False");
  }
}

mod chop {
  use super::*;

  #[test]
  fn small_number() {
    assert_eq!(interpret("Chop[0.00000000001]").unwrap(), "0");
  }

  #[test]
  fn normal_number() {
    assert_eq!(interpret("Chop[1.5]").unwrap(), "1.5");
  }

  #[test]
  fn zero() {
    assert_eq!(interpret("Chop[0]").unwrap(), "0");
  }

  #[test]
  fn negative_small() {
    assert_eq!(interpret("Chop[-0.00000000001]").unwrap(), "0");
  }

  #[test]
  fn list() {
    assert_eq!(
      interpret("Chop[{0.00000000001, 1.5, -0.000000000001}]").unwrap(),
      "{0, 1.5, 0}"
    );
  }

  #[test]
  fn custom_tolerance() {
    assert_eq!(interpret("Chop[0.05, 0.1]").unwrap(), "0");
    assert_eq!(interpret("Chop[0.5, 0.1]").unwrap(), "0.5");
  }
}

mod element {
  use super::*;

  #[test]
  fn integer_in_integers() {
    assert_eq!(interpret("Element[3, Integers]").unwrap(), "True");
  }

  #[test]
  fn real_not_in_integers() {
    assert_eq!(interpret("Element[3.5, Integers]").unwrap(), "False");
  }

  #[test]
  fn alternatives_with_known_member() {
    assert_eq!(
      interpret("Element[3 | a, Integers]").unwrap(),
      "Element[a, Integers]"
    );
  }

  #[test]
  fn symbolic_in_reals() {
    assert_eq!(interpret("Element[a, Reals]").unwrap(), "Element[a, Reals]");
  }

  #[test]
  fn integer_in_reals() {
    assert_eq!(interpret("Element[5, Reals]").unwrap(), "True");
  }

  #[test]
  fn prime_in_primes() {
    assert_eq!(interpret("Element[7, Primes]").unwrap(), "True");
  }

  #[test]
  fn non_prime_in_primes() {
    assert_eq!(interpret("Element[4, Primes]").unwrap(), "False");
  }

  #[test]
  fn pi_in_reals() {
    assert_eq!(interpret("Element[Pi, Reals]").unwrap(), "True");
  }

  #[test]
  fn e_in_reals() {
    assert_eq!(interpret("Element[E, Reals]").unwrap(), "True");
  }

  #[test]
  fn euler_gamma_in_reals() {
    assert_eq!(interpret("Element[EulerGamma, Reals]").unwrap(), "True");
  }

  #[test]
  fn pi_not_in_integers() {
    assert_eq!(interpret("Element[Pi, Integers]").unwrap(), "False");
  }

  #[test]
  fn pi_not_in_rationals() {
    assert_eq!(interpret("Element[Pi, Rationals]").unwrap(), "False");
  }

  #[test]
  fn rational_not_in_integers() {
    assert_eq!(interpret("Element[1/2, Integers]").unwrap(), "False");
  }

  #[test]
  fn rational_in_rationals() {
    assert_eq!(interpret("Element[3/4, Rationals]").unwrap(), "True");
  }

  #[test]
  fn i_not_in_reals() {
    assert_eq!(interpret("Element[I, Reals]").unwrap(), "False");
  }

  #[test]
  fn i_in_complexes() {
    assert_eq!(interpret("Element[I, Complexes]").unwrap(), "True");
  }

  #[test]
  fn pi_in_complexes() {
    assert_eq!(interpret("Element[Pi, Complexes]").unwrap(), "True");
  }

  #[test]
  fn true_in_booleans() {
    assert_eq!(interpret("Element[True, Booleans]").unwrap(), "True");
  }

  #[test]
  fn list_all_known() {
    // Element[{2, 3}, Integers] → True (all known members)
    assert_eq!(interpret("Element[{2, 3}, Integers]").unwrap(), "True");
  }

  #[test]
  fn list_with_non_member() {
    // Element[{2, 1/2}, Integers] → False (1/2 is not an integer)
    assert_eq!(interpret("Element[{2, 1/2}, Integers]").unwrap(), "False");
  }

  #[test]
  fn list_with_symbolic() {
    // Element[{x, y}, Reals] → Element[x | y, Reals] (symbolic remains)
    assert_eq!(
      interpret("Element[{x, y}, Reals]").unwrap(),
      "Element[x | y, Reals]"
    );
  }

  #[test]
  fn element_attributes() {
    assert_eq!(interpret("Attributes[Element]").unwrap(), "{Protected}");
  }
}

mod conditional_expression {
  use super::*;

  #[test]
  fn true_condition() {
    assert_eq!(
      interpret("ConditionalExpression[x^2, True]").unwrap(),
      "x^2"
    );
  }

  #[test]
  fn false_condition() {
    assert_eq!(
      interpret("ConditionalExpression[x^2, False]").unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn symbolic_condition() {
    assert_eq!(
      interpret("ConditionalExpression[x^2, x > 0]").unwrap(),
      "ConditionalExpression[x^2, x > 0]"
    );
  }
}

mod assuming {
  use super::*;

  #[test]
  fn assuming_matching_condition_simplifies() {
    assert_eq!(
      interpret("Assuming[y>0, ConditionalExpression[y x^2, y>0]//Simplify]")
        .unwrap(),
      "x^2*y"
    );
  }

  #[test]
  fn assuming_negated_condition_gives_undefined() {
    assert_eq!(
      interpret(
        "Assuming[Not[y>0], ConditionalExpression[y x^2, y>0]//Simplify]"
      )
      .unwrap(),
      "Undefined"
    );
  }

  #[test]
  fn simplify_conditional_without_assumptions() {
    assert_eq!(
      interpret("ConditionalExpression[y x ^ 2, y > 0]//Simplify").unwrap(),
      "ConditionalExpression[x^2*y, y > 0]"
    );
  }

  #[test]
  fn assuming_returns_body_value() {
    assert_eq!(interpret("Assuming[x > 0, 1 + 2]").unwrap(), "3");
  }

  #[test]
  fn assumptions_restored_after_assuming() {
    assert_eq!(interpret("Assuming[x > 0, $Assumptions]").unwrap(), "x > 0");
    assert_eq!(interpret("$Assumptions").unwrap(), "True");
  }
}

mod symbol_q {
  use super::*;

  // SymbolQ is not a standard Wolfram built-in (it's from GeneralUtilities package),
  // so it returns unevaluated in standard Wolfram Language — matching our behavior.

  #[test]
  fn symbol_is_true() {
    // Wolfram: SymbolQ[a] → SymbolQ[a] (unevaluated, not a standard built-in)
    assert_eq!(interpret("SymbolQ[a]").unwrap(), "SymbolQ[a]");
  }

  #[test]
  fn integer_is_false() {
    assert_eq!(interpret("SymbolQ[1]").unwrap(), "SymbolQ[1]");
  }

  #[test]
  fn expr_is_false() {
    assert_eq!(interpret("SymbolQ[a + b]").unwrap(), "SymbolQ[a + b]");
  }

  #[test]
  fn string_is_false() {
    assert_eq!(interpret("SymbolQ[\"abc\"]").unwrap(), "SymbolQ[abc]");
  }
}

mod same_q_unsame_q {
  use super::*;

  #[test]
  fn same_q_empty() {
    assert_eq!(interpret("SameQ[]").unwrap(), "True");
  }

  #[test]
  fn same_q_single() {
    assert_eq!(interpret("SameQ[a]").unwrap(), "True");
  }

  #[test]
  fn unsame_q_empty() {
    assert_eq!(interpret("UnsameQ[]").unwrap(), "True");
  }

  #[test]
  fn unsame_q_single() {
    assert_eq!(interpret("UnsameQ[a]").unwrap(), "True");
  }
}

mod equivalent_logic {
  use super::*;

  #[test]
  fn all_true() {
    assert_eq!(interpret("Equivalent[True, True]").unwrap(), "True");
  }

  #[test]
  fn mixed_true_false() {
    assert_eq!(interpret("Equivalent[True, True, False]").unwrap(), "False");
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("Equivalent[a, b, c]").unwrap(),
      "Equivalent[a, b, c]"
    );
  }
}

mod list_equality {
  use super::*;

  #[test]
  fn nested_lists_equal() {
    assert_eq!(interpret("{{1}, {2}} == {{1}, {2}}").unwrap(), "True");
  }

  #[test]
  fn flat_lists_equal() {
    assert_eq!(interpret("{1, 2} == {1, 2}").unwrap(), "True");
  }

  #[test]
  fn lists_not_equal() {
    assert_eq!(interpret("{1, 2} == {1, 3}").unwrap(), "False");
  }

  #[test]
  fn different_length_lists() {
    assert_eq!(interpret("{1, 2} == {1, 2, 3}").unwrap(), "False");
  }
}

mod equal_edge_cases {
  use super::*;

  #[test]
  fn equal_zero_args() {
    assert_eq!(interpret("Equal[]").unwrap(), "True");
  }

  #[test]
  fn equal_one_arg() {
    assert_eq!(interpret("Equal[x]").unwrap(), "True");
  }

  #[test]
  fn equal_one_arg_list() {
    assert_eq!(
      interpret("{Equal[x], Equal[1], Equal[\"a\"]}").unwrap(),
      "{True, True, True}"
    );
  }

  #[test]
  fn equal_funccall_var_symbolic() {
    // n[1] + 2 == 3 should stay symbolic, not evaluate to False
    assert_eq!(interpret("n[1] + 2 == 3").unwrap(), "2 + n[1] == 3");
  }

  #[test]
  fn equal_dot_product_symbolic() {
    // Dot product of Array vars with weights == target should stay symbolic
    assert_eq!(
      interpret("vars = Array[n, 2]; vars . {3, 5} == 10").unwrap(),
      "3*n[1] + 5*n[2] == 10"
    );
  }
}

mod unsame_q_multi {
  use super::*;

  #[test]
  fn three_args_with_duplicate() {
    assert_eq!(interpret("UnsameQ[1, 1, 2]").unwrap(), "False");
  }

  #[test]
  fn three_args_all_different() {
    assert_eq!(interpret("UnsameQ[1, 2, 3]").unwrap(), "True");
  }
}

mod atom_q {
  use super::*;

  #[test]
  fn atom_q_rational() {
    assert_eq!(interpret("AtomQ[1/2]").unwrap(), "True");
  }

  #[test]
  fn atom_q_integer() {
    assert_eq!(interpret("AtomQ[5]").unwrap(), "True");
  }

  #[test]
  fn atom_q_expression() {
    assert_eq!(interpret("AtomQ[x + y]").unwrap(), "False");
  }
}

mod xor_single {
  use super::*;

  #[test]
  fn xor_single_arg() {
    assert_eq!(interpret("Xor[True]").unwrap(), "True");
    assert_eq!(interpret("Xor[False]").unwrap(), "False");
  }
}

mod real_abs {
  use super::*;

  #[test]
  fn real_negative() {
    assert_eq!(interpret("RealAbs[-3.]").unwrap(), "3.");
  }

  #[test]
  fn integer_negative() {
    assert_eq!(interpret("RealAbs[-3]").unwrap(), "3");
  }

  #[test]
  fn symbolic() {
    assert_eq!(interpret("RealAbs[x]").unwrap(), "RealAbs[x]");
  }
}

mod abs_infinity {
  use super::*;

  #[test]
  fn infinity() {
    assert_eq!(interpret("Abs[Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn neg_infinity() {
    assert_eq!(interpret("Abs[-Infinity]").unwrap(), "Infinity");
  }

  #[test]
  fn complex_infinity() {
    assert_eq!(interpret("Abs[ComplexInfinity]").unwrap(), "Infinity");
  }
}
