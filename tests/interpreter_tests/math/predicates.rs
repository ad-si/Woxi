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
  fn positive_rational() {
    assert_eq!(interpret("Positive[3/4]").unwrap(), "True");
    assert_eq!(interpret("Positive[-3/4]").unwrap(), "False");
  }

  #[test]
  fn negative_rational() {
    assert_eq!(interpret("Negative[-3/4]").unwrap(), "True");
    assert_eq!(interpret("Negative[3/4]").unwrap(), "False");
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

  #[test]
  fn positive_symbolic_returns_unevaluated() {
    assert_eq!(interpret("Positive[x]").unwrap(), "Positive[x]");
  }

  #[test]
  fn non_negative_symbolic_returns_unevaluated() {
    assert_eq!(interpret("NonNegative[x]").unwrap(), "NonNegative[x]");
  }

  #[test]
  fn non_positive_symbolic_returns_unevaluated() {
    assert_eq!(interpret("NonPositive[x]").unwrap(), "NonPositive[x]");
  }

  #[test]
  fn negative_symbolic_returns_unevaluated() {
    assert_eq!(interpret("Negative[x]").unwrap(), "Negative[x]");
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

  #[test]
  fn complex_small_imaginary_dropped() {
    // The imaginary part is below tolerance so it should be chopped,
    // leaving just the real part.
    assert_eq!(interpret("Chop[1.0 + 1.*^-15 I]").unwrap(), "1.");
  }

  #[test]
  fn complex_normal_preserved() {
    assert_eq!(interpret("Chop[1. + 2. I]").unwrap(), "1. + 2.*I");
  }

  #[test]
  fn list_of_complex_numbers() {
    assert_eq!(
      interpret("Chop[{1.0 + 1.*^-15 I, 2.0 + 1.*^-5 I}]").unwrap(),
      "{1., 2. + 0.00001*I}"
    );
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
  fn one_not_in_primes() {
    // 1 is not prime
    assert_eq!(interpret("Element[1, Primes]").unwrap(), "False");
  }

  #[test]
  fn negative_not_in_primes() {
    assert_eq!(interpret("Element[-3, Primes]").unwrap(), "False");
  }

  #[test]
  fn real_not_in_primes() {
    assert_eq!(interpret("Element[3.5, Primes]").unwrap(), "False");
  }

  #[test]
  fn symbolic_in_primes_stays_unevaluated() {
    assert_eq!(
      interpret("Element[x, Primes]").unwrap(),
      "Element[x, Primes]"
    );
  }

  #[test]
  fn prime_list_in_primes() {
    // All members → True
    assert_eq!(interpret("Element[{2, 3, 5}, Primes]").unwrap(), "True");
  }

  #[test]
  fn mixed_list_in_primes() {
    // 4 is not prime → False
    assert_eq!(interpret("Element[{2, 3, 4}, Primes]").unwrap(), "False");
  }

  #[test]
  fn primes_symbol_unevaluated() {
    // The domain constant itself stays symbolic
    assert_eq!(interpret("Primes").unwrap(), "Primes");
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

  #[test]
  fn element_infix_named_char_pi_reals() {
    assert_eq!(interpret("Pi \\[Element] Reals").unwrap(), "True");
  }

  #[test]
  fn element_infix_named_char_integer() {
    assert_eq!(interpret("3 \\[Element] Integers").unwrap(), "True");
  }

  #[test]
  fn element_infix_named_char_rational() {
    assert_eq!(interpret("1/2 \\[Element] Rationals").unwrap(), "True");
  }

  #[test]
  fn element_infix_named_char_false() {
    assert_eq!(interpret("Pi \\[Element] Integers").unwrap(), "False");
  }

  #[test]
  fn element_infix_named_char_symbolic() {
    assert_eq!(
      interpret("x \\[Element] Reals").unwrap(),
      "Element[x, Reals]"
    );
  }

  #[test]
  fn element_infix_unicode_pi_reals() {
    assert_eq!(interpret("Pi \u{2208} Reals").unwrap(), "True");
  }

  #[test]
  fn element_infix_unicode_integer() {
    assert_eq!(interpret("3 \u{2208} Integers").unwrap(), "True");
  }

  #[test]
  fn element_infix_unicode_false() {
    assert_eq!(interpret("Pi \u{2208} Integers").unwrap(), "False");
  }
}

mod not_element {
  use super::*;

  #[test]
  fn integer_not_in_primes() {
    // 4 is not prime
    assert_eq!(interpret("NotElement[4, Primes]").unwrap(), "True");
  }

  #[test]
  fn prime_in_primes() {
    // 5 is prime, so NotElement returns False
    assert_eq!(interpret("NotElement[5, Primes]").unwrap(), "False");
  }

  #[test]
  fn integer_not_in_reals() {
    // 5 is real, so NotElement returns False
    assert_eq!(interpret("NotElement[5, Reals]").unwrap(), "False");
  }

  #[test]
  fn pi_not_in_integers() {
    assert_eq!(interpret("NotElement[Pi, Integers]").unwrap(), "True");
  }

  #[test]
  fn symbolic_stays_unevaluated() {
    assert_eq!(
      interpret("NotElement[x, Reals]").unwrap(),
      "NotElement[x, Reals]"
    );
  }

  #[test]
  fn infix_named_char() {
    assert_eq!(interpret("5 \\[NotElement] Primes").unwrap(), "False");
  }

  #[test]
  fn infix_named_char_true() {
    assert_eq!(interpret("Pi \\[NotElement] Integers").unwrap(), "True");
  }

  #[test]
  fn infix_unicode() {
    assert_eq!(interpret("5 \u{2209} Primes").unwrap(), "False");
  }

  #[test]
  fn infix_unicode_true() {
    assert_eq!(interpret("Pi \u{2209} Integers").unwrap(), "True");
  }

  #[test]
  fn attributes() {
    assert_eq!(interpret("Attributes[NotElement]").unwrap(), "{Protected}");
  }
}

mod reverse_element {
  use super::*;

  #[test]
  fn function_form_stays_unevaluated() {
    assert_eq!(
      interpret("ReverseElement[Reals, 5]").unwrap(),
      "Reals \u{220B} 5"
    );
  }

  #[test]
  fn infix_named_char() {
    assert_eq!(
      interpret("Reals \\[ReverseElement] 5").unwrap(),
      "Reals \u{220B} 5"
    );
  }

  #[test]
  fn infix_unicode() {
    assert_eq!(interpret("Reals \u{220B} 5").unwrap(), "Reals \u{220B} 5");
  }

  #[test]
  fn symbolic() {
    assert_eq!(
      interpret("Reals \\[ReverseElement] x").unwrap(),
      "Reals \u{220B} x"
    );
  }
}

mod named_characters {
  use super::*;

  #[test]
  fn epsilon_named_char() {
    assert_eq!(interpret("\\[Epsilon]").unwrap(), "\u{03F5}");
  }

  #[test]
  fn epsilon_unicode() {
    assert_eq!(interpret("\u{03F5}").unwrap(), "\u{03F5}");
  }

  #[test]
  fn epsilon_as_variable() {
    assert_eq!(interpret("\\[Epsilon] = 42; \\[Epsilon]").unwrap(), "42");
  }

  #[test]
  fn euro_named_char() {
    assert_eq!(interpret("\\[Euro]").unwrap(), "\u{20AC}");
  }

  #[test]
  fn euro_unicode() {
    assert_eq!(interpret("\u{20AC}").unwrap(), "\u{20AC}");
  }

  #[test]
  fn euro_as_variable() {
    assert_eq!(interpret("\\[Euro] = 100; \\[Euro]").unwrap(), "100");
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

  // SameQ on a complex integer versus various unrelated values: always
  // False, since SameQ compares structure not value. Only the identical
  // pair returns True.
  #[test]
  fn same_q_complex_integer_vs_real() {
    assert_eq!(interpret("SameQ[3 + 2 I, .25]").unwrap(), "False");
  }

  #[test]
  fn same_q_distinct_symbols() {
    assert_eq!(interpret("A === B").unwrap(), "False");
  }

  #[test]
  fn same_q_symbol_vs_integer() {
    assert_eq!(interpret("A === 1").unwrap(), "False");
  }

  #[test]
  fn same_q_complex_integer_vs_sqrt() {
    assert_eq!(interpret("SameQ[3 + 2 I, Sqrt[2]]").unwrap(), "False");
  }

  #[test]
  fn same_q_complex_integer_vs_bessel() {
    assert_eq!(interpret("SameQ[3 + 2 I, BesselJ[0, 2]]").unwrap(), "False");
  }

  #[test]
  fn same_q_complex_integer_equal_self() {
    assert_eq!(interpret("SameQ[3 + 2 I, 3 + 2 I]").unwrap(), "True");
  }

  // Regression (mathics test_comparison.py:159-167, where mathics's own
  // expectations are too strict — these match wolframscript): SameQ
  // between a low-precision tagged Real and a machine-precision Real
  // rounds both to the lower precision before comparing.
  #[test]
  fn same_q_precision_4_vs_machine_real_below_truncation() {
    // .222 differs from N[2/9, 4] = 0.2222 at the 4th digit → False.
    assert_eq!(interpret("N[2/9, 4] === .222").unwrap(), "False");
  }

  #[test]
  fn same_q_precision_4_vs_machine_real_at_truncation() {
    // .2222 rounds to 0.2222 at precision 4 → True (mathics disagrees;
    // wolframscript agrees).
    assert_eq!(interpret("N[2/9, 4] === .2222").unwrap(), "True");
  }

  #[test]
  fn same_q_precision_4_vs_machine_real_above_truncation() {
    // .22222 rounds to 0.2222 at precision 4 → True.
    assert_eq!(interpret("N[2/9, 4] === .22222").unwrap(), "True");
  }

  #[test]
  fn same_q_precision_4_vs_precision_3() {
    // Both BigFloats; min precision is 3, both round to 0.222 → True.
    assert_eq!(interpret("N[2/9, 4] === .222`3").unwrap(), "True");
  }

  #[test]
  fn same_q_machine_real_vs_precision_4() {
    // 2./9. (machine precision) vs N[2/9, 4]: round to precision 4,
    // both equal 0.2222 → True (wolframscript) — Woxi used to return
    // False for any precision below the machine band.
    assert_eq!(interpret("2./9. === N[2/9, 4]").unwrap(), "True");
  }

  #[test]
  fn same_q_machine_real_vs_machine_real_strict() {
    // Two machine reals with no shared bits still compare strictly:
    // 0.22221 ≠ 0.2222 bit-for-bit, so SameQ is False.
    assert_eq!(interpret(".22221 === .2222").unwrap(), "False");
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
    // Wolfram renders Equivalent with the U+29E6 infix operator.
    assert_eq!(
      interpret("Equivalent[a, b, c]").unwrap(),
      "a \u{29e6} b \u{29e6} c"
    );
  }

  // A True among symbolic terms reduces the equivalence to And: every
  // remaining term must also be True for the whole to hold.
  #[test]
  fn true_with_symbolic_reduces_to_and() {
    assert_eq!(
      interpret("Equivalent[a, b, True, c]").unwrap(),
      "a && b && c"
    );
  }

  // A False among symbolic terms reduces to And of the negations: each
  // remaining term must be False.
  #[test]
  fn false_with_symbolic_reduces_to_not() {
    assert_eq!(interpret("Equivalent[a, False]").unwrap(), "Not[a]");
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

  // Machine-precision Reals: Equal ignores the last ~7 bits of f64 mantissa,
  // matching wolframscript. Values within 2^-46 relative distance compare
  // True; SameQ stays strict.
  #[test]
  fn equal_tolerates_machine_real_rounding() {
    assert_eq!(
      interpret("Pochhammer[1, 3.001] == Pochhammer[2, 2.001]").unwrap(),
      "True"
    );
    assert_eq!(interpret("1.0 == 1.00000000000001").unwrap(), "True");
    assert_eq!(interpret("1.0 == 1.0000001").unwrap(), "False");
  }

  #[test]
  fn same_q_stays_strict_for_machine_reals() {
    assert_eq!(
      interpret("SameQ[6.007542293946962, 6.007542293946958]").unwrap(),
      "False"
    );
  }

  // SameQ between two BigFloats with explicit precision tags compares at
  // the *lower* of the two precisions. `.2222222\`6` and `.2222\`3` both
  // round to `0.222` once you only keep 3 significant digits, so SameQ
  // is True; `.22\`3` rounds to a different number and stays False.
  #[test]
  fn same_q_bigfloats_agree_at_lower_precision() {
    assert_eq!(interpret(".2222222`6 === .2222`3").unwrap(), "True");
    assert_eq!(interpret(".2222222`6 === .222`3").unwrap(), "True");
  }

  #[test]
  fn same_q_bigfloats_disagree_below_lower_precision() {
    assert_eq!(interpret(".2222222`6 === .22`3").unwrap(), "False");
    assert_eq!(interpret(".2222`3 === .333`3").unwrap(), "False");
  }

  #[test]
  fn equal_of_non_comparable_function_calls_stays_symbolic() {
    // g[2] == g[3] — no assumption that g is injective; stays symbolic.
    assert_eq!(interpret("g[2]==g[3]").unwrap(), "g[2] == g[3]");
  }

  #[test]
  fn unequal_of_non_comparable_function_calls_stays_symbolic() {
    // g[2] != g[3] stays symbolic for the same reason.
    assert_eq!(interpret("g[2]!=g[3]").unwrap(), "g[2] != g[3]");
  }

  #[test]
  fn equal_dot_product_symbolic() {
    // Dot product of Array vars with weights == target should stay symbolic
    assert_eq!(
      interpret("vars = Array[n, 2]; vars . {3, 5} == 10").unwrap(),
      "3*n[1] + 5*n[2] == 10"
    );
  }

  // ── Mixed-operator comparison chains split into pairwise `&&` ──────
  #[test]
  fn mixed_eq_neq_splits() {
    // `a == b != c` → `a == b && b != c` (matches wolframscript).
    assert_eq!(
      interpret("g[1] == g[2] != g[3]").unwrap(),
      "g[1] == g[2] && g[2] != g[3]"
    );
  }

  #[test]
  fn mixed_less_greater_splits() {
    assert_eq!(interpret("a < b > c").unwrap(), "a < b && b > c");
  }

  #[test]
  fn homogeneous_equal_chain_stays_whole() {
    assert_eq!(interpret("a == b == c").unwrap(), "a == b == c");
  }

  #[test]
  fn homogeneous_less_chain_stays_whole() {
    assert_eq!(interpret("a < b < c").unwrap(), "a < b < c");
  }

  #[test]
  fn equal_mixed_with_less_keeps_inequality() {
    // `a == b <= c` stays as a chain (Equal + Less-direction, no Unequal
    // and no opposite direction) — wolframscript displays the head form
    // `Inequality[a, Equal, b, LessEqual, c]`.
    assert_eq!(
      interpret("a == b <= c").unwrap(),
      "Inequality[a, Equal, b, LessEqual, c]"
    );
  }

  #[test]
  fn inequality_fn_splits_opposite_directions() {
    // `Inequality[a, Greater, b, LessEqual, c]` splits into pairwise `&&`
    // because the chain mixes a Greater-direction op with a Less-direction op.
    assert_eq!(
      interpret("Inequality[a, Greater, b, LessEqual, c]").unwrap(),
      "a > b && b <= c"
    );
  }

  #[test]
  fn inequality_fn_same_direction_stays_head() {
    // Same-direction Less/LessEqual — keep as Inequality head so
    // ToString[..., InputForm] can still print it as Inequality[...].
    assert_eq!(
      interpret("Inequality[0, LessEqual, x, LessEqual, 1]").unwrap(),
      "0 <= x <= 1"
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

  // ByteArray is an atomic data type in Wolfram — even though it has a
  // payload, its structure is opaque to Parts and other list operations.
  #[test]
  fn atom_q_byte_array() {
    assert_eq!(interpret("AtomQ[ByteArray[{4, 2}]]").unwrap(), "True");
  }

  // NumericArray likewise is atomic despite wrapping a tensor of numbers.
  #[test]
  fn atom_q_numeric_array() {
    assert_eq!(interpret("AtomQ[NumericArray[{1, 2, 3}]]").unwrap(), "True");
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

  #[test]
  fn listable() {
    assert_eq!(interpret("RealAbs[{-3, 0, 5}]").unwrap(), "{3, 0, 5}");
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

mod abs_exact {
  use super::*;

  #[test]
  fn rational_negative() {
    assert_eq!(interpret("Abs[-3/4]").unwrap(), "3/4");
  }

  #[test]
  fn rational_positive() {
    assert_eq!(interpret("Abs[3/4]").unwrap(), "3/4");
  }

  #[test]
  fn rational_negative_improper() {
    assert_eq!(interpret("Abs[-7/3]").unwrap(), "7/3");
  }

  #[test]
  fn integer_negative() {
    assert_eq!(interpret("Abs[-5]").unwrap(), "5");
  }
}

mod possible_zero_q {
  use super::*;

  #[test]
  fn literal_zero() {
    assert_eq!(interpret("PossibleZeroQ[0]").unwrap(), "True");
  }

  #[test]
  fn real_zero() {
    assert_eq!(interpret("PossibleZeroQ[0.0]").unwrap(), "True");
  }

  #[test]
  fn nonzero_integer() {
    assert_eq!(interpret("PossibleZeroQ[1]").unwrap(), "False");
  }

  #[test]
  fn nonzero_negative() {
    assert_eq!(interpret("PossibleZeroQ[-3]").unwrap(), "False");
  }

  #[test]
  fn nonzero_real() {
    assert_eq!(interpret("PossibleZeroQ[1.5]").unwrap(), "False");
  }

  #[test]
  fn nonzero_rational() {
    assert_eq!(interpret("PossibleZeroQ[1/2]").unwrap(), "False");
  }

  #[test]
  fn rational_cancel_to_zero() {
    assert_eq!(interpret("PossibleZeroQ[3/4 - 3/4]").unwrap(), "True");
  }

  #[test]
  fn symbolic_cancel() {
    assert_eq!(interpret("PossibleZeroQ[x - x]").unwrap(), "True");
  }

  #[test]
  fn symbolic_cancel_with_coeff() {
    assert_eq!(interpret("PossibleZeroQ[2*a - 2*a]").unwrap(), "True");
  }

  #[test]
  fn symbolic_cancel_power() {
    assert_eq!(interpret("PossibleZeroQ[a^2 - a^2]").unwrap(), "True");
  }

  #[test]
  fn symbolic_unknown_false() {
    assert_eq!(interpret("PossibleZeroQ[x]").unwrap(), "False");
  }

  #[test]
  fn sin_zero() {
    assert_eq!(interpret("PossibleZeroQ[Sin[0]]").unwrap(), "True");
  }

  #[test]
  fn sin_pi() {
    assert_eq!(interpret("PossibleZeroQ[Sin[Pi]]").unwrap(), "True");
  }

  #[test]
  fn cos_pi_half() {
    assert_eq!(interpret("PossibleZeroQ[Cos[Pi/2]]").unwrap(), "True");
  }

  #[test]
  fn cos_zero_minus_one() {
    assert_eq!(interpret("PossibleZeroQ[Cos[0] - 1]").unwrap(), "True");
  }

  #[test]
  fn log_one() {
    assert_eq!(interpret("PossibleZeroQ[Log[1]]").unwrap(), "True");
  }

  #[test]
  fn sqrt_cancel() {
    assert_eq!(
      interpret("PossibleZeroQ[Sqrt[2] - Sqrt[2]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn constant_subtract() {
    assert_eq!(interpret("PossibleZeroQ[E - E]").unwrap(), "True");
  }

  #[test]
  fn nonzero_constant_pi() {
    assert_eq!(interpret("PossibleZeroQ[Pi]").unwrap(), "False");
  }

  #[test]
  fn nonzero_sum() {
    assert_eq!(interpret("PossibleZeroQ[2 + 3]").unwrap(), "False");
  }

  #[test]
  fn complex_i_false() {
    assert_eq!(interpret("PossibleZeroQ[I]").unwrap(), "False");
  }

  #[test]
  fn complex_zero() {
    assert_eq!(interpret("PossibleZeroQ[0 + 0*I]").unwrap(), "True");
  }

  #[test]
  fn zero_times_i() {
    assert_eq!(interpret("PossibleZeroQ[0*I]").unwrap(), "True");
  }

  #[test]
  fn infinity_false() {
    assert_eq!(interpret("PossibleZeroQ[Infinity]").unwrap(), "False");
  }

  #[test]
  fn neg_infinity_false() {
    assert_eq!(interpret("PossibleZeroQ[-Infinity]").unwrap(), "False");
  }

  #[test]
  fn complex_infinity_false() {
    assert_eq!(
      interpret("PossibleZeroQ[ComplexInfinity]").unwrap(),
      "False"
    );
  }

  #[test]
  fn boolean_false() {
    assert_eq!(interpret("PossibleZeroQ[True]").unwrap(), "False");
  }

  #[test]
  fn string_false() {
    assert_eq!(interpret("PossibleZeroQ[\"hello\"]").unwrap(), "False");
  }

  #[test]
  fn x_squared_plus_one_false() {
    assert_eq!(interpret("PossibleZeroQ[x^2 + 1]").unwrap(), "False");
  }

  #[test]
  fn wrong_arg_count_unevaluated() {
    assert_eq!(interpret("PossibleZeroQ[]").unwrap(), "PossibleZeroQ[]");
    assert_eq!(
      interpret("PossibleZeroQ[1, 2]").unwrap(),
      "PossibleZeroQ[1, 2]"
    );
  }
}

mod tautology_q {
  use super::*;

  #[test]
  fn simple_tautology() {
    assert_eq!(interpret("TautologyQ[Or[a, Not[a]]]").unwrap(), "True");
  }

  #[test]
  fn not_a_tautology() {
    assert_eq!(interpret("TautologyQ[Or[a, b]]").unwrap(), "False");
    assert_eq!(interpret("TautologyQ[And[a, b]]").unwrap(), "False");
  }

  #[test]
  fn implies_tautology() {
    assert_eq!(
      interpret("TautologyQ[Implies[And[a, b], a]]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("TautologyQ[Implies[a, Or[a, b]]]").unwrap(),
      "True"
    );
  }

  #[test]
  fn constant_values() {
    assert_eq!(interpret("TautologyQ[True]").unwrap(), "True");
    assert_eq!(interpret("TautologyQ[False]").unwrap(), "False");
  }

  #[test]
  fn de_morgan() {
    // Not[And[a, b]] <=> Or[Not[a], Not[b]]
    assert_eq!(
      interpret("TautologyQ[Equivalent[Not[And[a, b]], Or[Not[a], Not[b]]]]")
        .unwrap(),
      "True"
    );
  }
}

mod all_match {
  use super::*;

  #[test]
  fn all_integers() {
    assert_eq!(interpret("AllMatch[{1, 2, 3}, _Integer]").unwrap(), "True");
  }

  #[test]
  fn mixed_types() {
    assert_eq!(
      interpret("AllMatch[{1, 2, \"a\"}, _Integer]").unwrap(),
      "False"
    );
  }

  #[test]
  fn pattern_matching() {
    assert_eq!(interpret("AllMatch[{x^2, x^3, x^5}, x^_]").unwrap(), "True");
    assert_eq!(
      interpret("AllMatch[{x^2, y^3, x^5}, x^_]").unwrap(),
      "False"
    );
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("AllMatch[{}, _Integer]").unwrap(), "True");
  }

  #[test]
  fn non_list() {
    assert_eq!(interpret("AllMatch[x, _Integer]").unwrap(), "True");
  }

  #[test]
  fn level_spec() {
    assert_eq!(
      interpret("AllMatch[{{1, 2}, {3, 4}}, _Integer, 2]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AllMatch[{{1, 2}, {3, 4}}, _Integer, 1]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("AllMatch[{{1, 2}, {3, 4}}, _List, 1]").unwrap(),
      "True"
    );
  }

  #[test]
  fn operator_form() {
    assert_eq!(interpret("AllMatch[_Integer][{1, 2, 3}]").unwrap(), "True");
    assert_eq!(
      interpret("AllMatch[_Integer][{1, \"a\", 3}]").unwrap(),
      "False"
    );
  }

  #[test]
  fn association() {
    assert_eq!(
      interpret("AllMatch[<|\"a\" -> 1, \"b\" -> 2|>, _Integer]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("AllMatch[<|\"a\" -> 1, \"b\" -> \"x\"|>, _Integer]").unwrap(),
      "False"
    );
  }
}

mod all_same_by {
  use super::*;

  #[test]
  fn same_parity() {
    assert_eq!(
      interpret("AllSameBy[{1, 3, 7, 5}, Mod[#, 2] &]").unwrap(),
      "True"
    );
  }

  #[test]
  fn different_parity() {
    assert_eq!(
      interpret("AllSameBy[{1, 3, 7, 4}, Mod[#, 2] &]").unwrap(),
      "False"
    );
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("AllSameBy[{}, f]").unwrap(), "True");
  }

  #[test]
  fn single_element() {
    assert_eq!(interpret("AllSameBy[{1}, f]").unwrap(), "True");
  }

  #[test]
  fn all_same_identity() {
    assert_eq!(interpret("AllSameBy[{1, 1, 1}, # &]").unwrap(), "True");
  }

  #[test]
  fn same_string_length() {
    assert_eq!(
      interpret("AllSameBy[{\"abc\", \"def\", \"ghi\"}, StringLength]")
        .unwrap(),
      "True"
    );
  }
}

mod any_match {
  use super::*;

  #[test]
  fn some_match() {
    assert_eq!(
      interpret("AnyMatch[{1, \"a\", \"b\"}, _Integer]").unwrap(),
      "True"
    );
  }

  #[test]
  fn none_match() {
    assert_eq!(
      interpret("AnyMatch[{\"a\", \"b\"}, _Integer]").unwrap(),
      "False"
    );
  }

  #[test]
  fn pattern_matching() {
    assert_eq!(interpret("AnyMatch[{x^2, y^3}, x^_]").unwrap(), "True");
  }

  #[test]
  fn empty_list() {
    assert_eq!(interpret("AnyMatch[{}, _Integer]").unwrap(), "False");
  }

  #[test]
  fn operator_form() {
    assert_eq!(
      interpret("AnyMatch[_Integer][{\"a\", 2, \"c\"}]").unwrap(),
      "True"
    );
  }
}

mod cases {
  use super::super::super::case_helpers::assert_case;

  #[test]
  fn assuming() {
    assert_case(
      r#"$Assumptions = { x > 0 }; Assuming[y>0, ConditionalExpression[y x^2, y>0]//Simplify]; Assuming[Not[y>0], ConditionalExpression[y x^2, y>0]//Simplify]"#,
      r#"Undefined"#,
    );
  }
  #[test]
  fn conditional_expression_1() {
    assert_case(
      r#"$Assumptions = { x > 0 }; Assuming[y>0, ConditionalExpression[y x^2, y>0]//Simplify]; Assuming[Not[y>0], ConditionalExpression[y x^2, y>0]//Simplify]; ConditionalExpression[y x ^ 2, y > 0]//Simplify"#,
      r#"ConditionalExpression[x^2*y, y > 0]"#,
    );
  }
  #[test]
  fn conditional_expression_2() {
    assert_case(r#"ConditionalExpression[x^2, True]"#, r#"x ^ 2"#);
  }
  #[test]
  fn conditional_expression_3() {
    assert_case(
      r#"ConditionalExpression[x^2, True]; ConditionalExpression[x^2, False]"#,
      r#"Undefined"#,
    );
  }
  #[test]
  fn greater_1() {
    assert_case(
      r#"ConditionalExpression[x^2, True]; ConditionalExpression[x^2, False]; f = ConditionalExpression[x^2, x>0]"#,
      r#"ConditionalExpression[x ^ 2, x > 0]"#,
    );
  }
  #[test]
  fn greater_2() {
    assert_case(
      r#"ConditionalExpression[x^2, True]; ConditionalExpression[x^2, False]; f = ConditionalExpression[x^2, x>0]; f /. x -> 2"#,
      r#"4"#,
    );
  }
  #[test]
  fn greater_3() {
    assert_case(
      r#"ConditionalExpression[x^2, True]; ConditionalExpression[x^2, False]; f = ConditionalExpression[x^2, x>0]; f /. x -> 2; f /. x -> -2"#,
      r#"Undefined"#,
    );
  }
  #[test]
  fn conditional_expression_4() {
    assert_case(r#"ConditionalExpression[a, False]"#, r#"Undefined"#);
  }
  #[test]
  fn prime_power_q_1() {
    assert_case(r#"PrimePowerQ[9]"#, r#"True"#);
  }
  #[test]
  fn prime_power_q_2() {
    assert_case(r#"PrimePowerQ[9]; PrimePowerQ[52142]"#, r#"False"#);
  }
  #[test]
  fn prime_power_q_3() {
    assert_case(
      r#"PrimePowerQ[9]; PrimePowerQ[52142]; PrimePowerQ[-8]"#,
      r#"True"#,
    );
  }
  #[test]
  fn prime_power_q_4() {
    assert_case(
      r#"PrimePowerQ[9]; PrimePowerQ[52142]; PrimePowerQ[-8]; PrimePowerQ[371293]"#,
      r#"True"#,
    );
  }
  #[test]
  fn equivalent_1() {
    assert_case(r#"Equivalent[True, True, False]"#, r#"False"#);
  }
  #[test]
  fn equivalent_2() {
    // Wolframscript-matched expectation. mathics rendered as the head
    // form `Equivalent[a, b, c]`; wolframscript -code uses the infix
    // `\[Equivalent]` (`⧦`) glyph, which is what Woxi emits too.
    assert_case(
      r#"Equivalent[True, True, False]; Equivalent[a, b, c]"#,
      "a \u{29E6} b \u{29E6} c",
    );
  }
  #[test]
  fn equivalent_3() {
    assert_case(
      r#"Equivalent[True, True, False]; Equivalent[a, b, c]; Equivalent[a, b, True, c]"#,
      r#"a && b && c"#,
    );
  }
  #[test]
  fn same_q() {
    assert_case(r#"a === a; SameQ[a] === SameQ[] === True"#, r#"True"#);
  }
  #[test]
  fn list_literal_1() {
    assert_case(
      r#"a === a; SameQ[a] === SameQ[] === True; {1==1., 1===1.}"#,
      r#"{True, False}"#,
    );
  }
  #[test]
  fn equal_1() {
    assert_case(
      r#"a === a; SameQ[a] === SameQ[] === True; {1==1., 1===1.}; 2./9. === .2222222222222222`15.9546"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_2() {
    assert_case(
      r#"a === a; SameQ[a] === SameQ[] === True; {1==1., 1===1.}; 2./9. === .2222222222222222`15.9546; .2222222`6 === .2222`3"#,
      r#"True"#,
    );
  }
  #[test]
  fn equal_3() {
    assert_case(
      r#"a === a; SameQ[a] === SameQ[] === True; {1==1., 1===1.}; 2./9. === .2222222222222222`15.9546; .2222222`6 === .2222`3; .2222222`6 === .222`3"#,
      r#"True"#,
    );
  }
  #[test]
  fn true_q_1() {
    assert_case(r#"TrueQ[True]"#, r#"True"#);
  }
  #[test]
  fn true_q_2() {
    assert_case(r#"TrueQ[True]; TrueQ[False]"#, r#"False"#);
  }
  #[test]
  fn true_q_3() {
    assert_case(r#"TrueQ[True]; TrueQ[False]; TrueQ[a]"#, r#"False"#);
  }
  #[test]
  fn unequal_1() {
    assert_case(
      r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x; Unequal["11", "11"]"#,
      r#"False"#,
    );
  }
  #[test]
  fn unequal_2() {
    assert_case(
      r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x; Unequal["11", "11"]; Unequal[11, "11"]"#,
      r#"True"#,
    );
  }
  #[test]
  fn list_literal_2() {
    assert_case(
      r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x; Unequal["11", "11"]; Unequal[11, "11"]; {1} != {2}"#,
      r#"True"#,
    );
  }
  #[test]
  fn list_literal_3() {
    assert_case(
      r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x; Unequal["11", "11"]; Unequal[11, "11"]; {1} != {2}; {1, 2} != {1, 2}"#,
      r#"False"#,
    );
  }
  #[test]
  fn list_literal_4() {
    assert_case(
      r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x; Unequal["11", "11"]; Unequal[11, "11"]; {1} != {2}; {1, 2} != {1, 2}; {a} != {a}"#,
      r#"False"#,
    );
  }
  #[test]
  fn string_literal_1() {
    assert_case(
      r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x; Unequal["11", "11"]; Unequal[11, "11"]; {1} != {2}; {1, 2} != {1, 2}; {a} != {a}; "a" != "b""#,
      r#"True"#,
    );
  }
  #[test]
  fn string_literal_2() {
    assert_case(
      r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x; Unequal["11", "11"]; Unequal[11, "11"]; {1} != {2}; {1, 2} != {1, 2}; {a} != {a}; "a" != "b"; "a" != "a""#,
      r#"False"#,
    );
  }
  #[test]
  fn list_literal_5() {
    assert_case(
      r#"1 != 1.; 1 != 2 != 3; 1 != 2 != x; Unequal["11", "11"]; Unequal[11, "11"]; {1} != {2}; {1, 2} != {1, 2}; {a} != {a}; "a" != "b"; "a" != "a"; {Unequal[], Unequal[x], Unequal[1]}"#,
      r#"{True, True, True}"#,
    );
  }
  #[test]
  fn coprime_q_1() {
    assert_case(r#"CoprimeQ[7, 9]"#, r#"True"#);
  }
  #[test]
  fn coprime_q_2() {
    assert_case(r#"CoprimeQ[7, 9]; CoprimeQ[-4, 9]"#, r#"True"#);
  }
  #[test]
  fn coprime_q_3() {
    assert_case(
      r#"CoprimeQ[7, 9]; CoprimeQ[-4, 9]; CoprimeQ[12, 15]"#,
      r#"False"#,
    );
  }
  #[test]
  fn coprime_q_4() {
    assert_case(
      r#"CoprimeQ[7, 9]; CoprimeQ[-4, 9]; CoprimeQ[12, 15]; CoprimeQ[1+2I, 1-I]; CoprimeQ[4+2I, 6+3I]; CoprimeQ[2, 3, 5]"#,
      r#"True"#,
    );
  }
  #[test]
  fn coprime_q_5() {
    assert_case(
      r#"CoprimeQ[7, 9]; CoprimeQ[-4, 9]; CoprimeQ[12, 15]; CoprimeQ[1+2I, 1-I]; CoprimeQ[4+2I, 6+3I]; CoprimeQ[2, 3, 5]; CoprimeQ[2, 4, 5]"#,
      r#"False"#,
    );
  }
  #[test]
  fn even_q_1() {
    assert_case(r#"EvenQ[4]"#, r#"True"#);
  }
  #[test]
  fn even_q_2() {
    assert_case(r#"EvenQ[4]; EvenQ[-3]"#, r#"False"#);
  }
  #[test]
  fn even_q_3() {
    assert_case(r#"EvenQ[4]; EvenQ[-3]; EvenQ[n]"#, r#"False"#);
  }
  #[test]
  fn integer_q_1() {
    assert_case(r#"IntegerQ[3]"#, r#"True"#);
  }
  #[test]
  fn integer_q_2() {
    assert_case(r#"IntegerQ[3]; IntegerQ[Pi]"#, r#"False"#);
  }
  #[test]
  fn negative_1() {
    assert_case(r#"Negative[0]"#, r#"False"#);
  }
  #[test]
  fn negative_2() {
    assert_case(r#"Negative[0]; Negative[-3]"#, r#"True"#);
  }
  #[test]
  fn negative_3() {
    assert_case(r#"Negative[0]; Negative[-3]; Negative[10/7]"#, r#"False"#);
  }
  #[test]
  fn negative_4() {
    assert_case(
      r#"Negative[0]; Negative[-3]; Negative[10/7]; Negative[1+2I]"#,
      r#"False"#,
    );
  }
  #[test]
  fn negative_5() {
    assert_case(
      r#"Negative[0]; Negative[-3]; Negative[10/7]; Negative[1+2I]; Negative[a + b]"#,
      r#"Negative[a + b]"#,
    );
  }
  #[test]
  fn list_literal_6() {
    assert_case(r#"{Positive[0], NonNegative[0]}"#, r#"{False, True}"#);
  }
  #[test]
  fn list_literal_7() {
    assert_case(r#"{Negative[0], NonPositive[0]}"#, r#"{False, True}"#);
  }
  #[test]
  fn number_q_1() {
    assert_case(r#"NumberQ[3+I]"#, r#"True"#);
  }
  #[test]
  fn number_q_2() {
    assert_case(r#"NumberQ[3+I]; NumberQ[5!]"#, r#"True"#);
  }
  #[test]
  fn number_q_3() {
    assert_case(r#"NumberQ[3+I]; NumberQ[5!]; NumberQ[Pi]"#, r#"False"#);
  }
  #[test]
  fn numeric_q_1() {
    assert_case(r#"NumericQ[2]"#, r#"True"#);
  }
  #[test]
  fn numeric_q_2() {
    assert_case(r#"NumericQ[2]; NumericQ[Sqrt[Pi]]"#, r#"True"#);
  }
  #[test]
  fn number_q_4() {
    assert_case(
      r#"NumericQ[2]; NumericQ[Sqrt[Pi]]; NumberQ[Sqrt[Pi]]"#,
      r#"False"#,
    );
  }
  #[test]
  fn numeric_q_3() {
    assert_case(
      r#"NumericQ[2]; NumericQ[Sqrt[Pi]]; NumberQ[Sqrt[Pi]]; NumericQ[a]=True"#,
      r#"True"#,
    );
  }
  #[test]
  fn numeric_q_4() {
    assert_case(
      r#"NumericQ[2]; NumericQ[Sqrt[Pi]]; NumberQ[Sqrt[Pi]]; NumericQ[a]=True; NumericQ[a]"#,
      r#"True"#,
    );
  }
  #[test]
  fn numeric_q_5() {
    assert_case(
      r#"NumericQ[2]; NumericQ[Sqrt[Pi]]; NumberQ[Sqrt[Pi]]; NumericQ[a]=True; NumericQ[a]; NumericQ[Sin[a]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn odd_q_1() {
    assert_case(r#"OddQ[-3]"#, r#"True"#);
  }
  #[test]
  fn odd_q_2() {
    assert_case(r#"OddQ[-3]; OddQ[0]"#, r#"False"#);
  }
  #[test]
  fn possible_zero_q_1() {
    assert_case(r#"PossibleZeroQ[E^(I Pi/4) - (-1)^(1/4)]"#, r#"True"#);
  }
  #[test]
  fn possible_zero_q_2() {
    assert_case(
      r#"PossibleZeroQ[E^(I Pi/4) - (-1)^(1/4)]; PossibleZeroQ[(x + 1) (x - 1) - x^2 + 1]"#,
      r#"True"#,
    );
  }
  #[test]
  fn possible_zero_q_3() {
    assert_case(
      r#"PossibleZeroQ[E^(I Pi/4) - (-1)^(1/4)]; PossibleZeroQ[(x + 1) (x - 1) - x^2 + 1]; PossibleZeroQ[(E + Pi)^2 - E^2 - Pi^2 - 2 E Pi]"#,
      r#"True"#,
    );
  }
  #[test]
  fn possible_zero_q_4() {
    assert_case(
      r#"PossibleZeroQ[E^(I Pi/4) - (-1)^(1/4)]; PossibleZeroQ[(x + 1) (x - 1) - x^2 + 1]; PossibleZeroQ[(E + Pi)^2 - E^2 - Pi^2 - 2 E Pi]; PossibleZeroQ[E^Pi - Pi^E]"#,
      r#"False"#,
    );
  }
  #[test]
  fn possible_zero_q_5() {
    assert_case(
      r#"PossibleZeroQ[E^(I Pi/4) - (-1)^(1/4)]; PossibleZeroQ[(x + 1) (x - 1) - x^2 + 1]; PossibleZeroQ[(E + Pi)^2 - E^2 - Pi^2 - 2 E Pi]; PossibleZeroQ[E^Pi - Pi^E]; PossibleZeroQ[1/x + 1/y - (x + y)/(x y)]"#,
      r#"True"#,
    );
  }
  #[test]
  fn possible_zero_q_6() {
    assert_case(
      r#"PossibleZeroQ[E^(I Pi/4) - (-1)^(1/4)]; PossibleZeroQ[(x + 1) (x - 1) - x^2 + 1]; PossibleZeroQ[(E + Pi)^2 - E^2 - Pi^2 - 2 E Pi]; PossibleZeroQ[E^Pi - Pi^E]; PossibleZeroQ[1/x + 1/y - (x + y)/(x y)]; PossibleZeroQ[2^(2 I) - 2^(-2 I) - 2 I Sin[Log[4]]]"#,
      r#"True"#,
    );
  }
  #[test]
  fn possible_zero_q_7() {
    assert_case(
      r#"PossibleZeroQ[E^(I Pi/4) - (-1)^(1/4)]; PossibleZeroQ[(x + 1) (x - 1) - x^2 + 1]; PossibleZeroQ[(E + Pi)^2 - E^2 - Pi^2 - 2 E Pi]; PossibleZeroQ[E^Pi - Pi^E]; PossibleZeroQ[1/x + 1/y - (x + y)/(x y)]; PossibleZeroQ[2^(2 I) - 2^(-2 I) - 2 I Sin[Log[4]]]; PossibleZeroQ[Sqrt[x^2] - x]"#,
      r#"False"#,
    );
  }
  #[test]
  fn positive_1() {
    assert_case(r#"Positive[1]"#, r#"True"#);
  }
  #[test]
  fn positive_2() {
    assert_case(r#"Positive[1]; Positive[0]"#, r#"False"#);
  }
  #[test]
  fn positive_3() {
    assert_case(r#"Positive[1]; Positive[0]; Positive[1 + 2 I]"#, r#"False"#);
  }
  #[test]
  fn prime_q_1() {
    assert_case(r#"PrimeQ[2]"#, r#"True"#);
  }
  #[test]
  fn prime_q_2() {
    assert_case(r#"PrimeQ[2]; PrimeQ[-3]"#, r#"True"#);
  }
  #[test]
  fn prime_q_3() {
    assert_case(r#"PrimeQ[2]; PrimeQ[-3]; PrimeQ[137]"#, r#"True"#);
  }
  #[test]
  fn prime_q_4() {
    assert_case(
      r#"PrimeQ[2]; PrimeQ[-3]; PrimeQ[137]; PrimeQ[2 ^ 127 - 1]"#,
      r#"True"#,
    );
  }
  #[test]
  fn symbol_literal() {
    assert_case(
      r#"2; 3+2 I; 2/9;  "hi!"; Infinity; Pi; E; I; a; b=2;b; $MachinePrecision; NumericQ[a]=True; a"#,
      r#"a"#,
    );
  }
  #[test]
  fn pi() {
    assert_case(
      r#"2; 3+2 I; 2/9;  "hi!"; Infinity; Pi; E; I; a; b=2;b; $MachinePrecision; NumericQ[a]=True; a; NumericQ[Pi]=False; Pi"#,
      r#"Pi"#,
    );
  }
  #[test]
  fn print() {
    assert_case(
      r#"2; 3+2 I; 2/9;  "hi!"; Infinity; Pi; E; I; a; b=2;b; $MachinePrecision; NumericQ[a]=True; a; NumericQ[Pi]=False; Pi; Print"#,
      r#"Print"#,
    );
  }
  #[test]
  fn sin() {
    assert_case(
      r#"2; 3+2 I; 2/9;  "hi!"; Infinity; Pi; E; I; a; b=2;b; $MachinePrecision; NumericQ[a]=True; a; NumericQ[Pi]=False; Pi; Print; Sin"#,
      r#"Sin"#,
    );
  }
  #[test]
  fn numeric_q_6() {
    assert_case(r#"NumericQ[a]=True;NumericQ[a]"#, r#"True"#);
  }
  #[test]
  fn numeric_q_7() {
    assert_case(
      r#"NumericQ[a]=True;NumericQ[a]; NumericQ[a]=False;NumericQ[a]"#,
      r#"False"#,
    );
  }
  #[test]
  fn unsame_q_1() {
    assert_case(r#"UnsameQ[]"#, r#"True"#);
  }
  #[test]
  fn unsame_q_2() {
    assert_case(r#"UnsameQ[]; UnsameQ[expr]"#, r#"True"#);
  }
  #[test]
  fn unequal_3() {
    assert_case(r#"UnsameQ[]; UnsameQ[expr]; x =!= x"#, r#"False"#);
  }
  #[test]
  fn unequal_4() {
    assert_case(r#"UnsameQ[]; UnsameQ[expr]; x =!= x; x =!= y"#, r#"True"#);
  }
  #[test]
  fn unequal_5() {
    assert_case(
      r#"UnsameQ[]; UnsameQ[expr]; x =!= x; x =!= y; 1 =!= 2 =!= 3 =!= 4"#,
      r#"True"#,
    );
  }
  #[test]
  fn unequal_6() {
    assert_case(
      r#"UnsameQ[]; UnsameQ[expr]; x =!= x; x =!= y; 1 =!= 2 =!= 3 =!= 4; 1 =!= 2 =!= 1 =!= 4"#,
      r#"False"#,
    );
  }
  #[test]
  fn unsame_q_3() {
    assert_case(
      r#"UnsameQ[]; UnsameQ[expr]; x =!= x; x =!= y; 1 =!= 2 =!= 3 =!= 4; 1 =!= 2 =!= 1 =!= 4; UnsameQ[10, 5, 2, 1, 0]"#,
      r#"True"#,
    );
  }
  #[test]
  fn unsame_q_4() {
    assert_case(
      r#"UnsameQ[]; UnsameQ[expr]; x =!= x; x =!= y; 1 =!= 2 =!= 3 =!= 4; 1 =!= 2 =!= 1 =!= 4; UnsameQ[10, 5, 2, 1, 0]; UnsameQ[10, 5, 2, 1, 0, 0]"#,
      r#"False"#,
    );
  }
  #[test]
  fn negative_6() {
    assert_case(
      r#"MachineNumberQ[1.5 + 3.14159265358979324 I]; MachineNumberQ[1.5 + 5 I]; Negative[-E]"#,
      r#"True"#,
    );
  }
  #[test]
  fn negative_7() {
    assert_case(
      r#"MachineNumberQ[1.5 + 3.14159265358979324 I]; MachineNumberQ[1.5 + 5 I]; Negative[-E]; Negative[Sin[{11, 14}]]"#,
      r#"{True, False}"#,
    );
  }
  #[test]
  fn positive_4() {
    assert_case(
      r#"MachineNumberQ[1.5 + 3.14159265358979324 I]; MachineNumberQ[1.5 + 5 I]; Negative[-E]; Negative[Sin[{11, 14}]]; Positive[Pi]"#,
      r#"True"#,
    );
  }
  #[test]
  fn positive_5() {
    assert_case(
      r#"MachineNumberQ[1.5 + 3.14159265358979324 I]; MachineNumberQ[1.5 + 5 I]; Negative[-E]; Negative[Sin[{11, 14}]]; Positive[Pi]; Positive[x]"#,
      r#"Positive[x]"#,
    );
  }
  #[test]
  fn positive_6() {
    assert_case(
      r#"MachineNumberQ[1.5 + 3.14159265358979324 I]; MachineNumberQ[1.5 + 5 I]; Negative[-E]; Negative[Sin[{11, 14}]]; Positive[Pi]; Positive[x]; Positive[Sin[{11, 14}]]"#,
      r#"{False, True}"#,
    );
  }
  #[test]
  fn prime_q_5() {
    assert_case(
      r#"MachineNumberQ[1.5 + 3.14159265358979324 I]; MachineNumberQ[1.5 + 5 I]; Negative[-E]; Negative[Sin[{11, 14}]]; Positive[Pi]; Positive[x]; Positive[Sin[{11, 14}]]; PrimeQ[1]"#,
      r#"False"#,
    );
  }
  #[test]
  fn prime_q_6() {
    assert_case(
      r#"MachineNumberQ[1.5 + 3.14159265358979324 I]; MachineNumberQ[1.5 + 5 I]; Negative[-E]; Negative[Sin[{11, 14}]]; Positive[Pi]; Positive[x]; Positive[Sin[{11, 14}]]; PrimeQ[1]; PrimeQ[2 ^ 255 - 1]"#,
      r#"False"#,
    );
  }
}

mod equal_accuracy_form {
  use super::*;

  // wolframscript's `Equal` is more lenient than
  // `|a - b| < 10^-p` for precision-tagged operands: it allows
  // up to roughly an extra decade of slack so that low-precision
  // near-equal literals match. Regression for mathics 1-Manual
  // `13.1416``4 == 13.1413``4 → True` row. Stored precision is
  // ~5.12 sig digits, |Δ| = 3e-4 ~ between 10^-5.12 and
  // 10^-(5.12 - 1) ≈ 7.6e-4.
  #[test]
  fn accuracy_four_equal_when_within_one_extra_decade() {
    assert_eq!(interpret("13.1416``4 == 13.1413``4").unwrap(), "True");
  }

  // Beyond the widened tolerance the comparison still returns
  // False, matching wolframscript.
  #[test]
  fn accuracy_four_unequal_when_difference_is_large() {
    assert_eq!(interpret("13.1416``4 == 13.5``4").unwrap(), "False");
  }
}
