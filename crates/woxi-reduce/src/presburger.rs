//! Cooper-style quantifier elimination for first-order linear integer arithmetic.
//!
//! Instead of relying on a search bound, existential elimination enumerates
//! the finite set of inequality transition points and divisibility residues.
//! This is a direct finite-cell form of Cooper's boundary/period argument.

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, Zero};

use super::affine::{AffineTerm, Variable};
use super::exact::{Rational, lcm};
use super::formula::{Atom, Formula, FormulaMemo, Quantifier, Relation};

pub fn eliminate_quantifiers(formula: Formula) -> Option<Formula> {
  let normalized = normalize_integer_formula(formula)?;
  let mut memo = FormulaMemo::default();
  eliminate_recursive(normalized, &mut memo).map(simplify_presburger)
}

pub fn normalize_integer_formula(formula: Formula) -> Option<Formula> {
  normalize_integer_nnf(formula.into_nnf()).map(Formula::normalized)
}

fn normalize_integer_nnf(formula: Formula) -> Option<Formula> {
  match formula {
    Formula::True | Formula::False => Some(formula),
    Formula::Atom(Atom::Relation(relation, term)) => {
      let term = clear_denominators(&term);
      let one = AffineTerm::constant(Rational::one());
      let negative = term.scaled(&Rational::integer((-1).into()));
      Some(match relation {
        Relation::Equal => Formula::And(vec![
          Formula::Atom(Atom::Relation(Relation::LessEqual, term)),
          Formula::Atom(Atom::Relation(Relation::LessEqual, negative)),
        ]),
        Relation::NotEqual => Formula::Or(vec![
          Formula::Atom(Atom::Relation(Relation::LessEqual, term.add(&one))),
          Formula::Atom(Atom::Relation(
            Relation::LessEqual,
            negative.add(&one),
          )),
        ]),
        Relation::Less => {
          Formula::Atom(Atom::Relation(Relation::LessEqual, term.add(&one)))
        }
        Relation::LessEqual => {
          Formula::Atom(Atom::Relation(Relation::LessEqual, term))
        }
        Relation::Greater => {
          Formula::Atom(Atom::Relation(Relation::LessEqual, negative.add(&one)))
        }
        Relation::GreaterEqual => {
          Formula::Atom(Atom::Relation(Relation::LessEqual, negative))
        }
      })
    }
    Formula::Atom(atom @ Atom::Divides { .. }) => Some(Formula::Atom(atom)),
    Formula::And(children) => Some(Formula::And(
      children
        .into_iter()
        .map(normalize_integer_nnf)
        .collect::<Option<Vec<_>>>()?,
    )),
    Formula::Or(children) => Some(Formula::Or(
      children
        .into_iter()
        .map(normalize_integer_nnf)
        .collect::<Option<Vec<_>>>()?,
    )),
    Formula::Quantified(quantifier, variables, body) => {
      Some(Formula::Quantified(
        quantifier,
        variables,
        Box::new(normalize_integer_nnf(*body)?),
      ))
    }
    Formula::Not(_) => unreachable!("input is converted to NNF first"),
  }
}

fn clear_denominators(term: &AffineTerm) -> AffineTerm {
  let denominator = term
    .coefficients
    .values()
    .fold(term.constant.denominator.clone(), |common, coefficient| {
      lcm(&common, &coefficient.denominator)
    });
  term.scaled(&Rational::integer(denominator))
}

#[doc(hidden)]
pub fn eliminate_recursive(
  formula: Formula,
  memo: &mut FormulaMemo,
) -> Option<Formula> {
  if let Some(result) = memo.get(&formula) {
    return Some(result);
  }
  let source = formula.clone();
  let result = match formula {
    Formula::True | Formula::False | Formula::Atom(_) => Some(formula),
    Formula::Not(_) => {
      eliminate_recursive(normalize_integer_formula(formula)?, memo)
    }
    Formula::And(children) => Some(
      Formula::And(
        children
          .into_iter()
          .map(|child| eliminate_recursive(child, memo))
          .collect::<Option<Vec<_>>>()?,
      )
      .normalized(),
    ),
    Formula::Or(children) => Some(
      Formula::Or(
        children
          .into_iter()
          .map(|child| eliminate_recursive(child, memo))
          .collect::<Option<Vec<_>>>()?,
      )
      .normalized(),
    ),
    Formula::Quantified(quantifier, variables, body) => {
      let mut result = eliminate_recursive(*body, memo)?;
      let mut remaining = variables.into_iter().collect::<BTreeSet<_>>();
      while let Some(variable) =
        choose_elimination_variable(&result, &remaining)
      {
        remaining.remove(&variable);
        result = match quantifier {
          Quantifier::Exists => eliminate_exists(result, &variable)?,
          Quantifier::ForAll => eliminate_forall(result, &variable)?,
        };
        debug_assert!(!result.contains_variable(&variable));
      }
      Some(result.normalized())
    }
  };
  if let Some(result) = &result {
    memo.insert(source, result.clone());
  }
  result
}

/// Estimates Cooper's finite instantiation count for each variable. The main
/// term is `period * (1 + transition_count)` after accounting for coefficient
/// normalization; atom occurrences and the variable identity break ties.
#[doc(hidden)]
pub fn choose_elimination_variable(
  formula: &Formula,
  variables: &BTreeSet<Variable>,
) -> Option<Variable> {
  variables
    .iter()
    .min_by_key(|variable| presburger_elimination_cost(formula, variable))
    .cloned()
}

#[doc(hidden)]
pub fn presburger_elimination_cost(
  formula: &Formula,
  variable: &Variable,
) -> (BigInt, usize) {
  fn visit(
    formula: &Formula,
    variable: &Variable,
    coefficient_lcm: &mut BigInt,
    period: &mut BigInt,
    transitions: &mut usize,
    occurrences: &mut usize,
  ) {
    match formula {
      Formula::Atom(Atom::Relation(_, term)) => {
        let coefficient = term.coefficient(variable);
        if !coefficient.is_zero() {
          *occurrences = occurrences.saturating_add(1);
          *transitions = transitions.saturating_add(1);
          if coefficient.is_integer() {
            *coefficient_lcm =
              lcm(coefficient_lcm, &coefficient.numerator.abs());
          }
        }
      }
      Formula::Atom(Atom::Divides { modulus, term, .. }) => {
        let coefficient = term.coefficient(variable);
        if !coefficient.is_zero() {
          *occurrences = occurrences.saturating_add(1);
          *period = lcm(period, modulus);
          if coefficient.is_integer() {
            *coefficient_lcm =
              lcm(coefficient_lcm, &coefficient.numerator.abs());
          }
        }
      }
      Formula::And(children) | Formula::Or(children) => {
        for child in children {
          visit(
            child,
            variable,
            coefficient_lcm,
            period,
            transitions,
            occurrences,
          );
        }
      }
      Formula::Not(inner) => visit(
        inner,
        variable,
        coefficient_lcm,
        period,
        transitions,
        occurrences,
      ),
      Formula::Quantified(_, _, body) => visit(
        body,
        variable,
        coefficient_lcm,
        period,
        transitions,
        occurrences,
      ),
      Formula::True | Formula::False => {}
    }
  }

  let mut coefficient_lcm = BigInt::one();
  let mut period = BigInt::one();
  let mut transitions = 0_usize;
  let mut occurrences = 0_usize;
  visit(
    formula,
    variable,
    &mut coefficient_lcm,
    &mut period,
    &mut transitions,
    &mut occurrences,
  );
  let instances =
    period * coefficient_lcm * BigInt::from(transitions.saturating_add(1));
  (instances, occurrences)
}

fn eliminate_forall(body: Formula, variable: &Variable) -> Option<Formula> {
  let negated = normalize_integer_formula(Formula::Not(Box::new(body)))?;
  let exists = eliminate_exists(negated, variable)?;
  normalize_integer_formula(Formula::Not(Box::new(exists)))
}

fn eliminate_exists(body: Formula, variable: &Variable) -> Option<Formula> {
  let (body, coefficient_lcm) = normalize_coefficients(body, variable)?;
  let body = if coefficient_lcm > BigInt::one() {
    Formula::And(vec![
      body,
      Formula::Atom(Atom::divides(
        coefficient_lcm,
        AffineTerm::variable(variable.clone()),
        false,
      )?),
    ])
    .normalized()
  } else {
    body
  };

  let period = divisibility_period(&body, variable);
  let transitions = transition_points(&body, variable)?;
  let mut disjuncts = Vec::new();
  let mut residue = BigInt::zero();
  while residue < period {
    let residue_term = AffineTerm::constant(Rational::integer(residue.clone()));
    disjuncts.push(negative_infinity_instance(&body, variable, &residue_term)?);
    for transition in &transitions {
      let candidate = transition.add(&residue_term);
      disjuncts.push(body.substitute(variable, &candidate).normalized());
    }
    residue += 1;
  }
  let result = simplify_presburger(Formula::Or(disjuncts));
  debug_assert!(!result.contains_variable(variable));
  Some(result)
}

fn simplify_presburger(formula: Formula) -> Formula {
  match formula {
    Formula::And(children) => {
      let children = children
        .into_iter()
        .map(simplify_presburger)
        .collect::<Vec<_>>();
      let normalized = Formula::And(children).normalized();
      let Formula::And(children) = &normalized else {
        return normalized;
      };
      let mut positive = BTreeSet::new();
      let mut negative = BTreeSet::new();
      for child in children {
        if let Formula::Atom(Atom::Divides {
          modulus,
          term,
          negated,
        }) = child
        {
          let key = (modulus.clone(), term.clone());
          if *negated {
            negative.insert(key);
          } else {
            positive.insert(key);
          }
        }
      }
      if positive.iter().any(|atom| negative.contains(atom)) {
        Formula::False
      } else {
        normalized
      }
    }
    Formula::Or(children) => {
      let children = children
        .into_iter()
        .map(simplify_presburger)
        .collect::<Vec<_>>();
      let normalized = Formula::Or(children).normalized();
      let Formula::Or(children) = &normalized else {
        return normalized;
      };
      let mut positive = BTreeSet::new();
      let mut negative = BTreeSet::new();
      let mut residue_groups: BTreeMap<
        (BigInt, BTreeMap<Variable, Rational>),
        BTreeSet<BigInt>,
      > = BTreeMap::new();
      for child in children {
        if let Formula::Atom(Atom::Divides {
          modulus,
          term,
          negated,
        }) = child
        {
          let key = (modulus.clone(), term.clone());
          if *negated {
            negative.insert(key);
          } else {
            positive.insert(key);
            let mut residue = &term.constant.numerator % modulus;
            if residue.is_negative() {
              residue += modulus;
            }
            residue_groups
              .entry((modulus.clone(), term.coefficients.clone()))
              .or_default()
              .insert(residue);
          }
        }
      }
      if positive.iter().any(|atom| negative.contains(atom))
        || residue_groups.iter().any(|((modulus, _), residues)| {
          BigInt::from(residues.len()) == *modulus
        })
      {
        Formula::True
      } else {
        normalized
      }
    }
    Formula::Not(inner) => Formula::Not(Box::new(simplify_presburger(*inner)))
      .into_nnf()
      .normalized(),
    Formula::Quantified(quantifier, variables, body) => Formula::Quantified(
      quantifier,
      variables,
      Box::new(simplify_presburger(*body)),
    )
    .normalized(),
    Formula::True | Formula::False | Formula::Atom(_) => formula.normalized(),
  }
}

/// Makes every nonzero coefficient of `variable` equal to `-1` or `1` by
/// introducing the scaled variable `L*variable`. The caller adds `L | variable`.
fn normalize_coefficients(
  formula: Formula,
  variable: &Variable,
) -> Option<(Formula, BigInt)> {
  let mut coefficient_lcm = BigInt::one();
  collect_coefficients(&formula, variable, &mut coefficient_lcm)?;
  let transformed =
    transform_coefficients(formula, variable, &coefficient_lcm)?;
  Some((transformed.normalized(), coefficient_lcm))
}

fn collect_coefficients(
  formula: &Formula,
  variable: &Variable,
  output: &mut BigInt,
) -> Option<()> {
  match formula {
    Formula::Atom(
      Atom::Relation(Relation::LessEqual, term) | Atom::Divides { term, .. },
    ) => {
      let coefficient = term.coefficient(variable);
      if !coefficient.is_integer() {
        return None;
      }
      if !coefficient.is_zero() {
        *output = lcm(output, &coefficient.numerator.abs());
      }
      Some(())
    }
    Formula::And(children) | Formula::Or(children) => {
      for child in children {
        collect_coefficients(child, variable, output)?;
      }
      Some(())
    }
    Formula::True | Formula::False => Some(()),
    Formula::Atom(Atom::Relation(_, _))
    | Formula::Not(_)
    | Formula::Quantified(_, _, _) => None,
  }
}

fn transform_coefficients(
  formula: Formula,
  variable: &Variable,
  coefficient_lcm: &BigInt,
) -> Option<Formula> {
  match formula {
    Formula::Atom(Atom::Relation(Relation::LessEqual, term)) => {
      let (term, _) = transform_term(term, variable, coefficient_lcm)?;
      Some(Formula::Atom(Atom::Relation(Relation::LessEqual, term)))
    }
    Formula::Atom(Atom::Divides {
      modulus,
      term,
      negated,
    }) => {
      let (term, scale) = transform_term(term, variable, coefficient_lcm)?;
      Some(Formula::Atom(Atom::Divides {
        modulus: modulus * scale,
        term,
        negated,
      }))
    }
    Formula::And(children) => Some(Formula::And(
      children
        .into_iter()
        .map(|child| transform_coefficients(child, variable, coefficient_lcm))
        .collect::<Option<Vec<_>>>()?,
    )),
    Formula::Or(children) => Some(Formula::Or(
      children
        .into_iter()
        .map(|child| transform_coefficients(child, variable, coefficient_lcm))
        .collect::<Option<Vec<_>>>()?,
    )),
    Formula::True | Formula::False => Some(formula),
    Formula::Atom(Atom::Relation(_, _))
    | Formula::Not(_)
    | Formula::Quantified(_, _, _) => None,
  }
}

fn transform_term(
  term: AffineTerm,
  variable: &Variable,
  coefficient_lcm: &BigInt,
) -> Option<(AffineTerm, BigInt)> {
  let coefficient = term.coefficient(variable);
  if coefficient.is_zero() {
    return Some((term, BigInt::one()));
  }
  if !coefficient.is_integer() {
    return None;
  }
  let scale = coefficient_lcm / coefficient.numerator.abs();
  let mut transformed = term.scaled(&Rational::integer(scale.clone()));
  transformed.coefficients.insert(
    variable.clone(),
    Rational::integer(match coefficient.numerator.sign() {
      Sign::Plus => BigInt::one(),
      Sign::Minus => -BigInt::one(),
      Sign::NoSign => unreachable!(),
    }),
  );
  Some((transformed, scale))
}

fn divisibility_period(formula: &Formula, variable: &Variable) -> BigInt {
  match formula {
    Formula::Atom(Atom::Divides { modulus, term, .. })
      if !term.coefficient(variable).is_zero() =>
    {
      modulus.clone()
    }
    Formula::And(children) | Formula::Or(children) => {
      children.iter().fold(BigInt::one(), |period, child| {
        lcm(&period, &divisibility_period(child, variable))
      })
    }
    Formula::True
    | Formula::False
    | Formula::Atom(_)
    | Formula::Not(_)
    | Formula::Quantified(_, _, _) => BigInt::one(),
  }
}

fn transition_points(
  formula: &Formula,
  variable: &Variable,
) -> Option<BTreeSet<AffineTerm>> {
  let mut output = BTreeSet::new();
  collect_transition_points(formula, variable, &mut output)?;
  Some(output)
}

fn collect_transition_points(
  formula: &Formula,
  variable: &Variable,
  output: &mut BTreeSet<AffineTerm>,
) -> Option<()> {
  match formula {
    Formula::Atom(Atom::Relation(Relation::LessEqual, term)) => {
      let coefficient = term.coefficient(variable);
      if coefficient.is_zero() {
        return Some(());
      }
      let mut rest = term.clone();
      rest.coefficients.remove(variable);
      match coefficient.numerator.sign() {
        // x + rest <= 0 changes truth at -rest + 1.
        Sign::Plus => {
          output.insert(
            rest
              .scaled(&Rational::integer((-1).into()))
              .add(&AffineTerm::constant(Rational::one())),
          );
        }
        // -x + rest <= 0 changes truth at rest.
        Sign::Minus => {
          output.insert(rest);
        }
        Sign::NoSign => {}
      }
      Some(())
    }
    Formula::Atom(Atom::Divides { .. }) | Formula::True | Formula::False => {
      Some(())
    }
    Formula::And(children) | Formula::Or(children) => {
      for child in children {
        collect_transition_points(child, variable, output)?;
      }
      Some(())
    }
    Formula::Atom(Atom::Relation(_, _))
    | Formula::Not(_)
    | Formula::Quantified(_, _, _) => None,
  }
}

fn negative_infinity_instance(
  formula: &Formula,
  variable: &Variable,
  residue: &AffineTerm,
) -> Option<Formula> {
  Some(
    match formula {
      Formula::True | Formula::False => formula.clone(),
      Formula::Atom(Atom::Relation(Relation::LessEqual, term)) => {
        match term.coefficient(variable).numerator.sign() {
          Sign::Plus => Formula::True,
          Sign::Minus => Formula::False,
          Sign::NoSign => {
            Formula::Atom(Atom::Relation(Relation::LessEqual, term.clone()))
          }
        }
      }
      Formula::Atom(atom @ Atom::Divides { .. }) => {
        Formula::Atom(atom.substitute(variable, residue))
      }
      Formula::And(children) => Formula::And(
        children
          .iter()
          .map(|child| negative_infinity_instance(child, variable, residue))
          .collect::<Option<Vec<_>>>()?,
      ),
      Formula::Or(children) => Formula::Or(
        children
          .iter()
          .map(|child| negative_infinity_instance(child, variable, residue))
          .collect::<Option<Vec<_>>>()?,
      ),
      Formula::Atom(Atom::Relation(_, _))
      | Formula::Not(_)
      | Formula::Quantified(_, _, _) => return None,
    }
    .normalized(),
  )
}
