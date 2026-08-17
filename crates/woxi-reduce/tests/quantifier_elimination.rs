use std::collections::{BTreeMap, BTreeSet};

use num_bigint::{BigInt, Sign};
use num_traits::Zero;
use woxi_reduce::presburger;
use woxi_reduce::rational_qe;
use woxi_reduce::{
  AffineTerm, Atom, Formula, FormulaMemo, Quantifier, Rational, Relation,
  Variable,
};

fn integer(value: i64) -> Rational {
  Rational::integer(BigInt::from(value))
}

fn variable_term(variable: &Variable) -> AffineTerm {
  AffineTerm::variable(variable.clone())
}

fn relation(relation: Relation, term: AffineTerm) -> Formula {
  Formula::Atom(Atom::Relation(relation, term))
}

fn rational_term_value(
  term: &AffineTerm,
  environment: &BTreeMap<Variable, Rational>,
) -> Rational {
  term.coefficients.iter().fold(
    term.constant.clone(),
    |value, (variable, coefficient)| {
      value.add(&coefficient.multiply(&environment[variable]))
    },
  )
}

fn rational_formula_value(
  formula: &Formula,
  environment: &BTreeMap<Variable, Rational>,
) -> bool {
  match formula {
    Formula::True => true,
    Formula::False => false,
    Formula::Atom(Atom::Relation(relation, term)) => {
      let sign = rational_term_value(term, environment).numerator.sign();
      match relation {
        Relation::Equal => sign == Sign::NoSign,
        Relation::NotEqual => sign != Sign::NoSign,
        Relation::Less => sign == Sign::Minus,
        Relation::LessEqual => sign != Sign::Plus,
        Relation::Greater => sign == Sign::Plus,
        Relation::GreaterEqual => sign != Sign::Minus,
      }
    }
    Formula::And(children) => children
      .iter()
      .all(|child| rational_formula_value(child, environment)),
    Formula::Or(children) => children
      .iter()
      .any(|child| rational_formula_value(child, environment)),
    Formula::Atom(Atom::Divides { .. })
    | Formula::Not(_)
    | Formula::Quantified(_, _, _) => {
      panic!("dense elimination must return quantifier-free relations")
    }
  }
}

fn integer_term_value(
  term: &AffineTerm,
  environment: &BTreeMap<Variable, BigInt>,
) -> BigInt {
  assert!(term.constant.is_integer());
  term.coefficients.iter().fold(
    term.constant.numerator.clone(),
    |value, (variable, coefficient)| {
      assert!(coefficient.is_integer());
      value + &coefficient.numerator * &environment[variable]
    },
  )
}

fn integer_formula_value(
  formula: &Formula,
  environment: &BTreeMap<Variable, BigInt>,
) -> bool {
  match formula {
    Formula::True => true,
    Formula::False => false,
    Formula::Atom(Atom::Relation(relation, term)) => {
      let sign = integer_term_value(term, environment).sign();
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
      let divides = (integer_term_value(term, environment) % modulus).is_zero();
      if *negated { !divides } else { divides }
    }
    Formula::And(children) => children
      .iter()
      .all(|child| integer_formula_value(child, environment)),
    Formula::Or(children) => children
      .iter()
      .any(|child| integer_formula_value(child, environment)),
    Formula::Not(_) | Formula::Quantified(_, _, _) => {
      panic!("Presburger elimination must return normalized quantifier-free IR")
    }
  }
}

#[test]
fn dense_projection_preserves_an_open_parameterized_interval() {
  let x = Variable::bound("x", 0);
  let a = Variable::free("a");
  let formula = Formula::Quantified(
    Quantifier::Exists,
    vec![x.clone()],
    Box::new(Formula::And(vec![
      relation(Relation::Greater, variable_term(&x)),
      relation(
        Relation::Less,
        variable_term(&x).subtract(&variable_term(&a)),
      ),
    ])),
  );
  let result = rational_qe::eliminate_quantifiers(formula).unwrap();
  assert!(!result.contains_quantifier());
  assert_eq!(result.free_variables(), BTreeSet::from([a.clone()]));
  for value in -2_i64..=2 {
    assert_eq!(
      rational_formula_value(
        &result,
        &BTreeMap::from([(a.clone(), integer(value))]),
      ),
      value > 0,
    );
  }
}

#[test]
fn dense_universal_duality_decides_a_total_order_tautology() {
  let x = Variable::bound("x", 0);
  let a = Variable::free("a");
  let difference = variable_term(&x).subtract(&variable_term(&a));
  let formula = Formula::Quantified(
    Quantifier::ForAll,
    vec![x],
    Box::new(Formula::Or(vec![
      relation(Relation::LessEqual, difference.clone()),
      relation(Relation::Greater, difference),
    ])),
  );
  assert_eq!(
    rational_qe::eliminate_quantifiers(formula),
    Some(Formula::True)
  );
}

#[test]
fn dense_order_prefers_an_equality_pivot() {
  let x = Variable::bound("x", 0);
  let y = Variable::bound("y", 1);
  let a = Variable::free("a");
  let body = Formula::And(vec![
    relation(Relation::Greater, variable_term(&x)),
    relation(
      Relation::Less,
      variable_term(&x).subtract(&variable_term(&a)),
    ),
    relation(
      Relation::Equal,
      variable_term(&y).subtract(&variable_term(&a)),
    ),
  ]);
  assert_eq!(
    rational_qe::choose_elimination_variable(
      &body,
      &BTreeSet::from([x, y.clone()]),
    ),
    Some(y)
  );
}

#[test]
fn repeated_dense_subformula_is_memoized() {
  let x = Variable::bound("x", 0);
  let repeated = Formula::Quantified(
    Quantifier::Exists,
    vec![x.clone()],
    Box::new(relation(
      Relation::Equal,
      variable_term(&x).subtract(&AffineTerm::constant(integer(1))),
    )),
  );
  let source = Formula::Or(vec![
    Formula::And(vec![repeated.clone(), Formula::True]),
    Formula::And(vec![repeated, Formula::False]),
  ]);
  let mut memo = FormulaMemo::default();
  assert!(rational_qe::eliminate_recursive(source, &mut memo).is_some());
  assert!(memo.hits() > 0);
}

#[test]
fn presburger_rejects_a_nonintegral_equality() {
  let x = Variable::bound("x", 0);
  let formula = Formula::Quantified(
    Quantifier::Exists,
    vec![x.clone()],
    Box::new(relation(
      Relation::Equal,
      variable_term(&x)
        .scaled(&integer(2))
        .subtract(&AffineTerm::constant(integer(1))),
    )),
  );
  assert_eq!(
    presburger::eliminate_quantifiers(formula),
    Some(Formula::False)
  );
}

#[test]
fn presburger_projects_parity_without_a_search_bound() {
  let y = Variable::bound("y", 0);
  let x = Variable::free("x");
  let formula = Formula::Quantified(
    Quantifier::Exists,
    vec![y.clone()],
    Box::new(relation(
      Relation::Equal,
      variable_term(&x)
        .subtract(&variable_term(&y).scaled(&integer(2)))
        .subtract(&AffineTerm::constant(integer(1))),
    )),
  );
  let result = presburger::eliminate_quantifiers(formula).unwrap();
  assert!(!result.contains_quantifier());
  assert_eq!(result.free_variables(), BTreeSet::from([x.clone()]));
  for value in -5_i64..=5 {
    assert_eq!(
      integer_formula_value(
        &result,
        &BTreeMap::from([(x.clone(), BigInt::from(value))]),
      ),
      value.rem_euclid(2) == 1,
    );
  }
}

#[test]
fn cooper_order_uses_the_smaller_residue_period() {
  let x = Variable::bound("x", 0);
  let y = Variable::bound("y", 1);
  let body = Formula::And(vec![
    Formula::Atom(
      Atom::divides(BigInt::from(11), variable_term(&x), false).unwrap(),
    ),
    Formula::Atom(
      Atom::divides(BigInt::from(2), variable_term(&y), false).unwrap(),
    ),
  ]);
  assert_eq!(
    presburger::choose_elimination_variable(
      &body,
      &BTreeSet::from([x, y.clone()]),
    ),
    Some(y)
  );
}
