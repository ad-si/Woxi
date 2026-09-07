use num_bigint::{BigInt, Sign};
use num_traits::Zero;
use woxi_reduce::presburger::*;
use woxi_reduce::*;

#[cfg(test)]
mod tests {
  use std::collections::{BTreeMap, BTreeSet};

  use proptest::prelude::*;
  use proptest::test_runner::RngSeed;

  use super::super::lower::formula_from_expr;
  use super::super::try_linear_integer_reduce;
  use super::*;
  use crate::syntax::Expr;

  fn parse(source: &str) -> Formula {
    formula_from_expr(&crate::parse_to_expr(source).unwrap()).unwrap()
  }

  // A symbol with the Constant attribute is a number, not an unknown, and
  // the exact rational fragment has no value for it. Treating `Pi` as a free
  // parameter would make `Exists[x, x > 4 && x < Pi]` come out `True`.
  #[test]
  fn constant_symbols_are_not_lowered_as_variables() {
    for source in ["x < Pi", "x > E && x < 3", "Degree * x == 1"] {
      assert!(
        formula_from_expr(&crate::parse_to_expr(source).unwrap()).is_none(),
        "{source}"
      );
    }
    assert!(
      formula_from_expr(&crate::parse_to_expr("x < a").unwrap()).is_some()
    );
  }

  fn eliminate(source: &str) -> Formula {
    eliminate_quantifiers(parse(source)).unwrap()
  }

  #[test]
  fn elimination_order_minimizes_cooper_instances_and_is_stable() {
    fn selection(source: &str) -> ((BigInt, usize), Formula) {
      let normalized = normalize_integer_formula(parse(source)).unwrap();
      let Formula::Quantified(_, variables, body) = &normalized else {
        panic!("test input must contain one quantifier block");
      };
      let selected = choose_elimination_variable(
        body,
        &variables.iter().cloned().collect::<BTreeSet<_>>(),
      )
      .unwrap();
      (
        presburger_elimination_cost(body, &selected),
        eliminate_quantifiers(normalized).unwrap(),
      )
    }

    let body = "x >= a && x <= b && Mod[x, 11] == 0 && Mod[y, 2] == 0";
    let first = selection(&format!("Exists[{{x, y}}, {body}]"));
    let permuted = selection(&format!("Exists[{{y, x}}, {body}]"));
    assert_eq!(first.0, (BigInt::from(2), 1));
    assert_eq!(first.0, permuted.0);
    assert_eq!(first.1, permuted.1);
  }

  #[test]
  fn repeated_integer_subformulas_use_the_shared_memo() {
    let repeated = parse("Exists[x, Mod[x, 2] == 1]");
    let source = normalize_integer_formula(Formula::Or(vec![
      Formula::And(vec![repeated.clone(), parse("a < 0")]),
      Formula::And(vec![repeated, parse("a >= 0")]),
    ]))
    .unwrap();
    let mut memo = FormulaMemo::default();
    let result = eliminate_recursive(source, &mut memo).unwrap().normalized();
    assert!(!result.contains_quantifier());
    assert!(memo.hits() > 0);
  }

  #[test]
  fn integer_strictness_and_equality_are_exact() {
    assert_eq!(eliminate("Exists[x, 2*x == 1]"), Formula::False);
    assert_eq!(eliminate("Exists[x, 2*x == 4]"), Formula::True);
    assert_eq!(eliminate("Exists[x, x > 3 && x < 4]"), Formula::False);
    assert_eq!(eliminate("Exists[x, x >= 3 && x <= 3]"), Formula::True);
  }

  #[test]
  fn unbounded_congruence_formulas_do_not_search_a_finite_interval() {
    assert_eq!(
      eliminate("Exists[x, x > a && Mod[x, 2] == 0]"),
      Formula::True
    );
    assert_eq!(
      eliminate("Exists[x, x < a && Mod[x, 3] == 1]"),
      Formula::True
    );
  }

  #[test]
  fn inconsistent_congruences_are_false() {
    assert_eq!(
      eliminate("Exists[x, Mod[x, 4] == 1 && Mod[x, 6] == 2]"),
      Formula::False
    );
  }

  #[test]
  fn integer_universal_duality_and_alternation() {
    assert_eq!(eliminate("ForAll[x, x <= a || x > a]"), Formula::True);
    assert_eq!(eliminate("ForAll[x, Exists[y, y > x]]"), Formula::True);
    assert_eq!(eliminate("Exists[x, ForAll[y, y <= x]]"), Formula::False);
  }

  // A lone linear equation in several integer unknowns is routed to the
  // lattice-parametrizing Diophantine reducer, not to Cooper elimination,
  // which would only restate the equation as a degenerate two-sided bound.
  #[test]
  fn underdetermined_linear_equations_are_left_to_the_diophantine_reducer() {
    fn args(source: &str) -> Vec<Expr> {
      let call = crate::parse_to_expr(source).unwrap();
      let Expr::FunctionCall { args, .. } = &call else {
        panic!("test input must be a Reduce call");
      };
      args.to_vec()
    }
    for source in [
      "Reduce[2 x == 4 y, {x, y}, Integers]",
      "Reduce[3 x + 5 y == 1, {x, y}, Integers]",
      "Reduce[2 x + 4 y == 5, {x, y}, Integers]",
      "Reduce[2 x == 4, {x, y}, Integers]",
    ] {
      assert!(
        try_linear_integer_reduce(&args(source)).is_none(),
        "{source}"
      );
    }
    // A single unknown, or anything beyond a lone equation, stays here.
    assert!(
      try_linear_integer_reduce(&args("Reduce[2 x == 4, x, Integers]"))
        .is_some()
    );
    assert!(
      try_linear_integer_reduce(&args(
        "Reduce[2 x == 4 y && x >= 0, {x, y}, Integers]"
      ))
      .is_some()
    );
  }

  #[test]
  fn parity_projection_eliminates_the_bound_variable() {
    let result = eliminate("Exists[y, x == 2*y + 1]");
    assert!(!result.contains_quantifier());
    assert!(
      result
        .all_variables()
        .iter()
        .all(|variable| variable.binder.is_none())
    );
  }

  fn evaluate_integer_term(
    term: &AffineTerm,
    environment: &BTreeMap<Variable, BigInt>,
  ) -> BigInt {
    assert!(term.constant.is_integer());
    term.coefficients.iter().fold(
      term.constant.numerator.clone(),
      |value, (variable, coefficient)| {
        assert!(coefficient.is_integer());
        value
          + &coefficient.numerator
            * environment
              .get(variable)
              .expect("every variable must have an integer valuation")
      },
    )
  }

  fn evaluate_integer_formula(
    formula: &Formula,
    environment: &mut BTreeMap<Variable, BigInt>,
  ) -> bool {
    match formula {
      Formula::True => true,
      Formula::False => false,
      Formula::Atom(Atom::Relation(relation, term)) => {
        let sign = evaluate_integer_term(term, environment).sign();
        match relation {
          Relation::Equal => sign == Sign::NoSign,
          Relation::NotEqual => sign != Sign::NoSign,
          Relation::Less => sign == Sign::Minus,
          Relation::LessEqual => sign != Sign::Plus,
          Relation::Greater => sign == Sign::Plus,
          Relation::GreaterEqual => sign != Sign::Minus,
        }
      }
      Formula::Atom(Atom::Divides {
        modulus,
        term,
        negated,
      }) => {
        let divides =
          (evaluate_integer_term(term, environment) % modulus).is_zero();
        if *negated { !divides } else { divides }
      }
      Formula::And(children) => children
        .iter()
        .all(|child| evaluate_integer_formula(child, environment)),
      Formula::Or(children) => children
        .iter()
        .any(|child| evaluate_integer_formula(child, environment)),
      Formula::Not(inner) => !evaluate_integer_formula(inner, environment),
      Formula::Quantified(quantifier, variables, body) => {
        assert_eq!(variables.len(), 1);
        let variable = &variables[0];
        let values = -3_i8..=3;
        match quantifier {
          Quantifier::Exists => values.into_iter().any(|value| {
            environment.insert(variable.clone(), BigInt::from(value));
            let result = evaluate_integer_formula(body, environment);
            environment.remove(variable);
            result
          }),
          Quantifier::ForAll => values.into_iter().all(|value| {
            environment.insert(variable.clone(), BigInt::from(value));
            let result = evaluate_integer_formula(body, environment);
            environment.remove(variable);
            result
          }),
        }
      }
    }
  }

  fn generated_integer_atom(
    bound: &Variable,
    free: &Variable,
    bound_coefficient: i8,
    free_coefficient: i8,
    constant: i8,
    kind: u8,
    modulus: u8,
  ) -> Formula {
    let term = AffineTerm::variable(bound.clone())
      .scaled(&Rational::integer(BigInt::from(bound_coefficient)))
      .add(
        &AffineTerm::variable(free.clone())
          .scaled(&Rational::integer(BigInt::from(free_coefficient))),
      )
      .add(&AffineTerm::constant(Rational::integer(BigInt::from(
        constant,
      ))));
    if kind < 6 {
      Formula::Atom(Atom::Relation(
        match kind {
          0 => Relation::Equal,
          1 => Relation::NotEqual,
          2 => Relation::Less,
          3 => Relation::LessEqual,
          4 => Relation::Greater,
          _ => Relation::GreaterEqual,
        },
        term,
      ))
    } else {
      Formula::Atom(
        Atom::divides(BigInt::from(modulus), term, kind == 7).unwrap(),
      )
    }
  }

  fn bounded_quantifier(
    body: Formula,
    variable: Variable,
    universal: bool,
  ) -> Formula {
    let lower_term = AffineTerm::variable(variable.clone())
      .add(&AffineTerm::constant(Rational::integer(BigInt::from(3))));
    let upper_term = AffineTerm::variable(variable.clone())
      .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(3))));
    if universal {
      Formula::Quantified(
        Quantifier::ForAll,
        vec![variable],
        Box::new(Formula::Or(vec![
          Formula::Atom(Atom::Relation(Relation::Less, lower_term)),
          Formula::Atom(Atom::Relation(Relation::Greater, upper_term)),
          body,
        ])),
      )
    } else {
      Formula::Quantified(
        Quantifier::Exists,
        vec![variable],
        Box::new(Formula::And(vec![
          Formula::Atom(Atom::Relation(Relation::GreaterEqual, lower_term)),
          Formula::Atom(Atom::Relation(Relation::LessEqual, upper_term)),
          body,
        ])),
      )
    }
  }

  proptest! {
    #![proptest_config(ProptestConfig {
      cases: std::env::var("WOXI_REDUCE_PRESBURGER_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(25_000),
      rng_seed: RngSeed::Fixed(0x5eed_3000),
      max_shrink_iters: 30_000,
      ..ProptestConfig::default()
    })]

    #[test]
    #[ignore = "25,000-case self-contained Presburger fuzz gate"]
    fn generated_bounded_formulas_agree_with_exhaustive_integer_evaluation(
      atoms in proptest::collection::vec(
        (-4_i8..=4, -4_i8..=4, -7_i8..=7, 0_u8..8, 2_u8..=5),
        1..=4,
      ),
      connective_bits in any::<u8>(),
      universal in any::<bool>(),
    ) {
      let bound = Variable::bound("y", 0);
      let free = Variable::free("x");
      let mut children = atoms.into_iter().map(|(a, b, c, kind, modulus)| {
        generated_integer_atom(&bound, &free, a, b, c, kind, modulus)
      });
      let mut body = children.next().unwrap();
      for (index, child) in children.enumerate() {
        body = if connective_bits & (1 << index) == 0 {
          Formula::And(vec![body, child])
        } else {
          Formula::Or(vec![body, child])
        };
      }
      let original = bounded_quantifier(body, bound, universal).normalized();
      let eliminated = eliminate_quantifiers(original.clone()).unwrap();
      prop_assert!(!eliminated.contains_quantifier());
      for value in -2_i8..=2 {
        let mut environment =
          BTreeMap::from([(free.clone(), BigInt::from(value))]);
        let before = evaluate_integer_formula(&original, &mut environment);
        let after = evaluate_integer_formula(&eliminated, &mut environment);
        prop_assert_eq!(before, after);
      }
    }
  }
}
