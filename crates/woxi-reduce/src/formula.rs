//! Binder-safe linear formula representation and theory-neutral normalization.

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::BigInt;
use num_traits::{Signed, Zero};

use super::affine::{AffineTerm, Variable};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Relation {
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
}

impl Relation {
  pub fn negated(self) -> Self {
    match self {
      Self::Equal => Self::NotEqual,
      Self::NotEqual => Self::Equal,
      Self::Less => Self::GreaterEqual,
      Self::LessEqual => Self::Greater,
      Self::Greater => Self::LessEqual,
      Self::GreaterEqual => Self::Less,
    }
  }
}

/// Atomic linear predicates. Relations are stored as `term relation 0`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
  Relation(Relation, AffineTerm),
  Divides {
    modulus: BigInt,
    term: AffineTerm,
    negated: bool,
  },
}

impl Atom {
  pub fn divides(
    modulus: BigInt,
    term: AffineTerm,
    negated: bool,
  ) -> Option<Self> {
    if modulus.is_zero()
      || !term.constant.is_integer()
      || term
        .coefficients
        .values()
        .any(|coefficient| !coefficient.is_integer())
    {
      return None;
    }
    Some(Self::Divides {
      modulus: modulus.abs(),
      term,
      negated,
    })
  }

  pub fn negated(&self) -> Self {
    match self {
      Self::Relation(relation, term) => {
        Self::Relation(relation.negated(), term.clone())
      }
      Self::Divides {
        modulus,
        term,
        negated,
      } => Self::Divides {
        modulus: modulus.clone(),
        term: term.clone(),
        negated: !negated,
      },
    }
  }

  pub fn variables(&self) -> BTreeSet<Variable> {
    match self {
      Self::Relation(_, term) | Self::Divides { term, .. } => term.variables(),
    }
  }

  pub fn substitute(
    &self,
    variable: &Variable,
    replacement: &AffineTerm,
  ) -> Self {
    match self {
      Self::Relation(relation, term) => {
        Self::Relation(*relation, term.substitute(variable, replacement))
      }
      Self::Divides {
        modulus,
        term,
        negated,
      } => Self::Divides {
        modulus: modulus.clone(),
        term: term.substitute(variable, replacement),
        negated: *negated,
      },
    }
  }

  fn constant_truth(&self) -> Option<bool> {
    match self {
      Self::Relation(relation, term) if term.is_constant() => {
        let sign = term.constant.numerator.sign();
        Some(match relation {
          Relation::Equal => sign == num_bigint::Sign::NoSign,
          Relation::NotEqual => sign != num_bigint::Sign::NoSign,
          Relation::Less => sign == num_bigint::Sign::Minus,
          Relation::LessEqual => sign != num_bigint::Sign::Plus,
          Relation::Greater => sign == num_bigint::Sign::Plus,
          Relation::GreaterEqual => sign != num_bigint::Sign::Minus,
        })
      }
      Self::Divides {
        modulus,
        term,
        negated,
      } if term.is_constant() => {
        let divides = (&term.constant.numerator % modulus).is_zero();
        Some(if *negated { !divides } else { divides })
      }
      Self::Relation(_, _) | Self::Divides { .. } => None,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Quantifier {
  Exists,
  ForAll,
}

impl Quantifier {
  fn negated(self) -> Self {
    match self {
      Self::Exists => Self::ForAll,
      Self::ForAll => Self::Exists,
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Formula {
  True,
  False,
  Atom(Atom),
  And(Vec<Self>),
  Or(Vec<Self>),
  Not(Box<Self>),
  Quantified(Quantifier, Vec<Variable>, Box<Self>),
}

/// Per-elimination memo for structurally repeated subformulas.
///
/// Normalization removes duplicate siblings, but the same quantified subtree
/// can still occur below distinct Boolean parents. Both theory engines use
/// this one cache rather than maintaining independent sharing mechanisms.
#[derive(Default)]
pub struct FormulaMemo {
  results: BTreeMap<Formula, Formula>,
  hits: usize,
}

impl FormulaMemo {
  pub fn get(&mut self, formula: &Formula) -> Option<Formula> {
    let result = self.results.get(formula).cloned();
    if result.is_some() {
      self.hits += 1;
    }
    result
  }

  pub fn insert(&mut self, source: Formula, result: Formula) {
    self.results.insert(source, result);
  }

  pub fn hits(&self) -> usize {
    self.hits
  }
}

impl Formula {
  pub fn into_nnf(self) -> Self {
    self.into_nnf_with_polarity(false)
  }

  fn into_nnf_with_polarity(self, negated: bool) -> Self {
    match self {
      Self::True => {
        if negated {
          Self::False
        } else {
          Self::True
        }
      }
      Self::False => {
        if negated {
          Self::True
        } else {
          Self::False
        }
      }
      Self::Atom(atom) => {
        Self::Atom(if negated { atom.negated() } else { atom })
      }
      Self::Not(inner) => inner.into_nnf_with_polarity(!negated),
      Self::And(children) => {
        let children = children
          .into_iter()
          .map(|child| child.into_nnf_with_polarity(negated))
          .collect();
        if negated {
          Self::Or(children)
        } else {
          Self::And(children)
        }
      }
      Self::Or(children) => {
        let children = children
          .into_iter()
          .map(|child| child.into_nnf_with_polarity(negated))
          .collect();
        if negated {
          Self::And(children)
        } else {
          Self::Or(children)
        }
      }
      Self::Quantified(quantifier, variables, body) => Self::Quantified(
        if negated {
          quantifier.negated()
        } else {
          quantifier
        },
        variables,
        Box::new(body.into_nnf_with_polarity(negated)),
      ),
    }
  }

  /// Applies only theory-neutral equivalence rewrites and stable ordering.
  pub fn normalized(self) -> Self {
    match self {
      Self::Not(_) => self.into_nnf().normalized(),
      Self::And(children) => Self::normalized_and(children),
      Self::Or(children) => Self::normalized_or(children),
      Self::Quantified(quantifier, mut variables, body) => {
        variables.sort();
        variables.dedup();
        let body = body.normalized();
        let body_variables = body.all_variables();
        variables.retain(|variable| body_variables.contains(variable));
        if variables.is_empty() {
          body
        } else {
          Self::Quantified(quantifier, variables, Box::new(body))
        }
      }
      Self::Atom(atom) => match atom.constant_truth() {
        Some(true) => Self::True,
        Some(false) => Self::False,
        None => Self::Atom(atom),
      },
      leaf => leaf,
    }
  }

  fn normalized_and(children: Vec<Self>) -> Self {
    let mut flattened = Vec::new();
    for child in children {
      match child.normalized() {
        Self::False => return Self::False,
        Self::True => {}
        Self::And(nested) => flattened.extend(nested),
        child => flattened.push(child),
      }
    }
    flattened.sort();
    flattened.dedup();
    let simplify_affine = within_affine_simplification_budget(&flattened);
    if simplify_affine {
      for (index, left) in flattened.iter().enumerate() {
        if flattened[index + 1..]
          .iter()
          .any(|right| formulas_contradict(left, right))
        {
          return Self::False;
        }
      }
    }
    // Remove an affine half-space when another conjunct is provably stronger.
    // Equivalent but differently scaled atoms retain the structurally first
    // representative, keeping normalization deterministic and idempotent.
    if simplify_affine {
      let original = flattened.clone();
      flattened = original
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
          !original.iter().enumerate().any(|(other_index, other)| {
            if index == &other_index {
              return false;
            }
            formula_implies(other, candidate)
              && (!formula_implies(candidate, other) || other_index < *index)
          })
        })
        .map(|(_, child)| child.clone())
        .collect();
    }
    match flattened.len() {
      0 => Self::True,
      1 => flattened.pop().unwrap(),
      _ => Self::And(flattened),
    }
  }

  fn normalized_or(children: Vec<Self>) -> Self {
    let mut flattened = Vec::new();
    for child in children {
      match child.normalized() {
        Self::True => return Self::True,
        Self::False => {}
        Self::Or(nested) => flattened.extend(nested),
        child => flattened.push(child),
      }
    }
    flattened.sort();
    flattened.dedup();
    let simplify_affine = within_affine_simplification_budget(&flattened);
    if simplify_affine {
      for (index, left) in flattened.iter().enumerate() {
        if flattened[index + 1..]
          .iter()
          .any(|right| formulas_cover_universe(left, right))
        {
          return Self::True;
        }
      }
    }
    // In a disjunction, a branch that implies another branch is redundant.
    // Equivalent branches retain the structurally first representative.
    if simplify_affine {
      let original = flattened.clone();
      flattened = original
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
          !original.iter().enumerate().any(|(other_index, other)| {
            if index == &other_index {
              return false;
            }
            formula_implies(candidate, other)
              && (!formula_implies(other, candidate) || other_index < *index)
          })
        })
        .map(|(_, child)| child.clone())
        .collect();
    }
    match flattened.len() {
      0 => Self::False,
      1 => flattened.pop().unwrap(),
      _ => Self::Or(flattened),
    }
  }

  pub fn all_variables(&self) -> BTreeSet<Variable> {
    let mut output = BTreeSet::new();
    self.collect_all_variables(&mut output);
    output
  }

  fn collect_all_variables(&self, output: &mut BTreeSet<Variable>) {
    match self {
      Self::True | Self::False => {}
      Self::Atom(atom) => output.extend(atom.variables()),
      Self::And(children) | Self::Or(children) => {
        for child in children {
          child.collect_all_variables(output);
        }
      }
      Self::Not(inner) => inner.collect_all_variables(output),
      Self::Quantified(_, variables, body) => {
        output.extend(variables.iter().cloned());
        body.collect_all_variables(output);
      }
    }
  }

  pub fn free_variables(&self) -> BTreeSet<Variable> {
    fn collect(
      formula: &Formula,
      bound: &mut BTreeSet<Variable>,
      output: &mut BTreeSet<Variable>,
    ) {
      match formula {
        Formula::True | Formula::False => {}
        Formula::Atom(atom) => output.extend(
          atom
            .variables()
            .into_iter()
            .filter(|variable| !bound.contains(variable)),
        ),
        Formula::And(children) | Formula::Or(children) => {
          for child in children {
            collect(child, bound, output);
          }
        }
        Formula::Not(inner) => collect(inner, bound, output),
        Formula::Quantified(_, variables, body) => {
          let newly_bound = variables
            .iter()
            .filter(|variable| bound.insert((*variable).clone()))
            .cloned()
            .collect::<Vec<_>>();
          collect(body, bound, output);
          for variable in newly_bound {
            bound.remove(&variable);
          }
        }
      }
    }

    let mut output = BTreeSet::new();
    collect(self, &mut BTreeSet::new(), &mut output);
    output
  }

  pub fn contains_variable(&self, variable: &Variable) -> bool {
    self.all_variables().contains(variable)
  }

  pub fn substitute(
    &self,
    variable: &Variable,
    replacement: &AffineTerm,
  ) -> Self {
    match self {
      Self::True | Self::False => self.clone(),
      Self::Atom(atom) => Self::Atom(atom.substitute(variable, replacement)),
      Self::And(children) => Self::And(
        children
          .iter()
          .map(|child| child.substitute(variable, replacement))
          .collect(),
      ),
      Self::Or(children) => Self::Or(
        children
          .iter()
          .map(|child| child.substitute(variable, replacement))
          .collect(),
      ),
      Self::Not(inner) => {
        Self::Not(Box::new(inner.substitute(variable, replacement)))
      }
      Self::Quantified(quantifier, variables, body)
        if variables.contains(variable) =>
      {
        Self::Quantified(*quantifier, variables.clone(), body.clone())
      }
      Self::Quantified(quantifier, variables, body) => Self::Quantified(
        *quantifier,
        variables.clone(),
        Box::new(body.substitute(variable, replacement)),
      ),
    }
  }

  pub fn is_nnf(&self) -> bool {
    match self {
      Self::Not(_) => false,
      Self::And(children) | Self::Or(children) => {
        children.iter().all(Self::is_nnf)
      }
      Self::Quantified(_, _, body) => body.is_nnf(),
      Self::True | Self::False | Self::Atom(_) => true,
    }
  }

  pub fn contains_quantifier(&self) -> bool {
    match self {
      Self::Quantified(_, _, _) => true,
      Self::And(children) | Self::Or(children) => {
        children.iter().any(Self::contains_quantifier)
      }
      Self::Not(inner) => inner.contains_quantifier(),
      Self::True | Self::False | Self::Atom(_) => false,
    }
  }

  pub fn contains_divisibility(&self) -> bool {
    match self {
      Self::Atom(Atom::Divides { .. }) => true,
      Self::And(children) | Self::Or(children) => {
        children.iter().any(Self::contains_divisibility)
      }
      Self::Not(inner) | Self::Quantified(_, _, inner) => {
        inner.contains_divisibility()
      }
      Self::True | Self::False | Self::Atom(Atom::Relation(_, _)) => false,
    }
  }
}

/// Exact compositional implication is intentionally a small-formula
/// canonicalization pass. Applying its pairwise recursion to the large
/// divisibility disjunctions produced by Cooper elimination is semantically
/// unnecessary and can dominate the decision procedure.
fn within_affine_simplification_budget(children: &[Formula]) -> bool {
  children.len() <= 64
    && children.iter().all(|child| !child.contains_divisibility())
    && children.iter().map(formula_node_count).sum::<usize>() <= 512
}

fn formula_node_count(formula: &Formula) -> usize {
  match formula {
    Formula::And(children) | Formula::Or(children) => {
      1 + children.iter().map(formula_node_count).sum::<usize>()
    }
    Formula::Not(inner) | Formula::Quantified(_, _, inner) => {
      1 + formula_node_count(inner)
    }
    Formula::True | Formula::False | Formula::Atom(_) => 1,
  }
}

fn formula_implies(left: &Formula, right: &Formula) -> bool {
  if left == right
    || matches!(left, Formula::False)
    || matches!(right, Formula::True)
  {
    return true;
  }
  match (left, right) {
    (Formula::Or(children), _) => {
      children.iter().all(|child| formula_implies(child, right))
    }
    (_, Formula::And(children)) => {
      children.iter().all(|child| formula_implies(left, child))
    }
    (_, Formula::Or(children)) => {
      children.iter().any(|child| formula_implies(left, child))
    }
    (Formula::And(children), _) => {
      children.iter().any(|child| formula_implies(child, right))
    }
    (
      Formula::Atom(Atom::Relation(left_relation, left_term)),
      Formula::Atom(Atom::Relation(right_relation, right_term)),
    ) => {
      order_atom_implies(*left_relation, left_term, *right_relation, right_term)
    }
    _ => false,
  }
}

fn formulas_contradict(left: &Formula, right: &Formula) -> bool {
  let (
    Formula::Atom(Atom::Relation(left_relation, left_term)),
    Formula::Atom(Atom::Relation(right_relation, right_term)),
  ) = (left, right)
  else {
    return false;
  };
  let Some((factor, offset, left_strict, right_strict)) =
    half_space_relationship(
      *left_relation,
      left_term,
      *right_relation,
      right_term,
    )
  else {
    return false;
  };
  if factor.numerator.sign() != num_bigint::Sign::Minus {
    return false;
  }
  match offset.numerator.sign() {
    num_bigint::Sign::Plus => true,
    num_bigint::Sign::NoSign => left_strict || right_strict,
    num_bigint::Sign::Minus => false,
  }
}

fn formulas_cover_universe(left: &Formula, right: &Formula) -> bool {
  let (
    Formula::Atom(Atom::Relation(left_relation, left_term)),
    Formula::Atom(Atom::Relation(right_relation, right_term)),
  ) = (left, right)
  else {
    return false;
  };
  let Some((factor, offset, left_strict, right_strict)) =
    half_space_relationship(
      *left_relation,
      left_term,
      *right_relation,
      right_term,
    )
  else {
    return false;
  };
  if factor.numerator.sign() != num_bigint::Sign::Minus {
    return false;
  }
  match offset.numerator.sign() {
    num_bigint::Sign::Minus => true,
    num_bigint::Sign::NoSign => !left_strict || !right_strict,
    num_bigint::Sign::Plus => false,
  }
}

/// Decides implication between two proportional open/closed affine
/// half-spaces. Both predicates are oriented as `term < 0` or `term <= 0`.
fn order_atom_implies(
  left_relation: Relation,
  left_term: &AffineTerm,
  right_relation: Relation,
  right_term: &AffineTerm,
) -> bool {
  let Some((factor, offset, left_strict, right_strict)) =
    half_space_relationship(
      left_relation,
      left_term,
      right_relation,
      right_term,
    )
  else {
    return false;
  };
  if factor.numerator.sign() != num_bigint::Sign::Plus {
    return false;
  }

  // right_term = factor * left_term + offset. Under left_term <= 0,
  // a negative offset proves either right strictness; a zero offset requires
  // the left predicate to be at least as strict as the right predicate.
  match offset.numerator.sign() {
    num_bigint::Sign::Minus => true,
    num_bigint::Sign::NoSign => left_strict || !right_strict,
    num_bigint::Sign::Plus => false,
  }
}

fn half_space_relationship(
  left_relation: Relation,
  left_term: &AffineTerm,
  right_relation: Relation,
  right_term: &AffineTerm,
) -> Option<(super::exact::Rational, super::exact::Rational, bool, bool)> {
  let (left_term, left_strict) = upper_half_space(left_relation, left_term)?;
  let (right_term, right_strict) =
    upper_half_space(right_relation, right_term)?;
  let (_, left_coefficient) = left_term.coefficients.first_key_value()?;
  let (_, right_coefficient) = right_term.coefficients.first_key_value()?;
  let factor = right_coefficient.checked_divide(left_coefficient)?;
  if right_term.coefficients != left_term.scaled(&factor).coefficients {
    return None;
  }
  let offset = right_term
    .constant
    .subtract(&left_term.constant.multiply(&factor));
  Some((factor, offset, left_strict, right_strict))
}

fn upper_half_space(
  relation: Relation,
  term: &AffineTerm,
) -> Option<(AffineTerm, bool)> {
  match relation {
    Relation::Less => Some((term.clone(), true)),
    Relation::LessEqual => Some((term.clone(), false)),
    Relation::Greater => Some((
      term.scaled(&super::exact::Rational::integer((-1).into())),
      true,
    )),
    Relation::GreaterEqual => Some((
      term.scaled(&super::exact::Rational::integer((-1).into())),
      false,
    )),
    Relation::Equal | Relation::NotEqual => None,
  }
}

#[cfg(test)]
mod tests {
  use num_bigint::BigInt;
  use proptest::prelude::*;
  use proptest::test_runner::RngSeed;

  use super::super::exact::Rational;
  use super::*;

  fn relation(variable: Variable, relation: Relation, value: i64) -> Formula {
    Formula::Atom(Atom::Relation(
      relation,
      AffineTerm::variable(variable).subtract(&AffineTerm::constant(
        Rational::integer(BigInt::from(value)),
      )),
    ))
  }

  #[test]
  fn nnf_swaps_connectives_quantifiers_and_atoms() {
    let x = Variable::bound("x", 1);
    let formula = Formula::Not(Box::new(Formula::Quantified(
      Quantifier::Exists,
      vec![x.clone()],
      Box::new(Formula::And(vec![
        relation(x.clone(), Relation::Less, 0),
        relation(x, Relation::NotEqual, 2),
      ])),
    )));
    let nnf = formula.into_nnf();
    assert!(nnf.is_nnf());
    let Formula::Quantified(Quantifier::ForAll, _, body) = nnf else {
      panic!("negation must swap Exists to ForAll");
    };
    let Formula::Or(children) = *body else {
      panic!("negation must swap And to Or");
    };
    assert!(matches!(
      &children[0],
      Formula::Atom(Atom::Relation(Relation::GreaterEqual, _))
    ));
    assert!(matches!(
      &children[1],
      Formula::Atom(Atom::Relation(Relation::Equal, _))
    ));
  }

  #[test]
  fn normalization_is_idempotent_and_order_independent() {
    let x = Variable::free("x");
    let low = relation(x.clone(), Relation::Greater, 0);
    let high = relation(x, Relation::LessEqual, 4);
    let first = Formula::And(vec![
      Formula::True,
      high.clone(),
      Formula::And(vec![low.clone(), high]),
    ])
    .normalized();
    let second = Formula::And(vec![low, first.clone()]).normalized();
    assert_eq!(first, second);
    assert_eq!(first.clone().normalized(), first);
  }

  #[test]
  fn normalization_removes_weaker_proportional_half_spaces() {
    let x = Variable::free("x");
    let strongest_upper = relation(x.clone(), Relation::Less, 0);
    let weaker_upper = relation(x.clone(), Relation::LessEqual, 1);
    let strongest_lower = relation(x.clone(), Relation::Greater, 1);
    let weaker_lower = relation(x, Relation::GreaterEqual, 0);
    assert_eq!(
      Formula::And(vec![weaker_upper, strongest_upper.clone()]).normalized(),
      strongest_upper
    );
    assert_eq!(
      Formula::And(vec![weaker_lower, strongest_lower.clone()]).normalized(),
      strongest_lower
    );
  }

  #[test]
  fn normalization_detects_complementary_bounds_and_subsumes_branches() {
    let x = Variable::free("x");
    let below_zero = relation(x.clone(), Relation::Less, 0);
    let at_least_zero = relation(x.clone(), Relation::GreaterEqual, 0);
    assert_eq!(
      Formula::And(vec![below_zero.clone(), at_least_zero.clone()])
        .normalized(),
      Formula::False
    );
    assert_eq!(
      Formula::Or(vec![below_zero.clone(), at_least_zero]).normalized(),
      Formula::True
    );

    let below_one = relation(x, Relation::Less, 1);
    let narrow_branch = Formula::And(vec![
      below_zero,
      relation(Variable::free("y"), Relation::Greater, 0),
    ]);
    assert_eq!(
      Formula::Or(vec![narrow_branch, below_one.clone()]).normalized(),
      below_one
    );
  }

  #[test]
  fn free_variables_respect_shadowing_identity() {
    let free_x = Variable::free("x");
    let bound_x = Variable::bound("x", 9);
    let formula = Formula::Quantified(
      Quantifier::Exists,
      vec![bound_x.clone()],
      Box::new(Formula::And(vec![
        relation(bound_x, Relation::Equal, 0),
        relation(free_x.clone(), Relation::Greater, 1),
      ])),
    );
    assert_eq!(formula.free_variables(), BTreeSet::from([free_x]));
  }

  #[test]
  fn divisibility_requires_integral_affine_coefficients() {
    let x = Variable::free("x");
    let integral = AffineTerm::variable(x.clone())
      .scaled(&Rational::integer(BigInt::from(2)));
    assert!(Atom::divides(BigInt::from(-6), integral, false).is_some());

    let half = Rational::new(BigInt::from(1), BigInt::from(2)).unwrap();
    let fractional = AffineTerm::variable(x).scaled(&half);
    assert!(Atom::divides(BigInt::from(6), fractional, false).is_none());
    assert!(Atom::divides(BigInt::zero(), AffineTerm::zero(), false).is_none());
  }

  #[test]
  fn substitution_removes_only_the_requested_identity() {
    let x = Variable::free("x");
    let shadow = Variable::bound("x", 1);
    let formula = Formula::And(vec![
      relation(x.clone(), Relation::Equal, 1),
      relation(shadow.clone(), Relation::Equal, 2),
    ]);
    let substituted = formula.substitute(
      &x,
      &AffineTerm::constant(Rational::integer(BigInt::from(1))),
    );
    assert!(!substituted.contains_variable(&x));
    assert!(substituted.contains_variable(&shadow));
  }

  #[test]
  fn constant_atoms_and_unused_binders_fold() {
    let true_relation = Formula::Atom(Atom::Relation(
      Relation::Less,
      AffineTerm::constant(Rational::integer(BigInt::from(-1))),
    ));
    assert_eq!(true_relation.normalized(), Formula::True);

    let false_divisibility = Formula::Atom(
      Atom::divides(
        BigInt::from(4),
        AffineTerm::constant(Rational::integer(BigInt::from(6))),
        false,
      )
      .unwrap(),
    );
    assert_eq!(false_divisibility.normalized(), Formula::False);

    let unused = Variable::bound("unused", 99);
    assert_eq!(
      Formula::Quantified(
        Quantifier::Exists,
        vec![unused],
        Box::new(Formula::True),
      )
      .normalized(),
      Formula::True
    );
  }

  proptest! {
    #![proptest_config(ProptestConfig {
      cases: 256,
      rng_seed: RngSeed::Fixed(0x5eed_0002),
      ..ProptestConfig::default()
    })]

    #[test]
    fn normalization_is_a_permutation_invariant_fixed_point(
      bounds in proptest::collection::vec(-50_i64..=50, 0..24),
    ) {
      let x = Variable::free("x");
      let mut forward = bounds
        .iter()
        .map(|bound| relation(x.clone(), Relation::LessEqual, *bound))
        .collect::<Vec<_>>();
      let mut reverse = forward.clone();
      reverse.reverse();
      let forward = Formula::And(std::mem::take(&mut forward)).normalized();
      let reverse = Formula::And(reverse).normalized();
      prop_assert_eq!(&forward, &reverse);
      prop_assert_eq!(forward.clone().normalized(), forward);
    }

    #[test]
    fn bound_source_names_are_alpha_invariant(
      first in "[a-z][a-z0-9]{0,12}",
      second in "[a-z][a-z0-9]{0,12}",
      binder in 0_u32..10_000,
    ) {
      prop_assert_eq!(
        Variable::bound(first, binder),
        Variable::bound(second, binder),
      );
    }
  }
}
