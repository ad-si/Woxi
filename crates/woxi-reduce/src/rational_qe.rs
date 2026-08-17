//! Quantifier elimination for exact dense linear arithmetic.
//!
//! The implementation is a correctness-first Fourier-Motzkin baseline. It
//! expands Boolean structure to DNF for existential projection; later sharing
//! and recursive projection optimizations must be checked against this path.

use std::collections::BTreeSet;

use num_bigint::Sign;

use super::affine::{AffineTerm, Variable};
use super::exact::Rational;
use super::formula::{Atom, Formula, FormulaMemo, Quantifier, Relation};

pub fn eliminate_quantifiers(formula: Formula) -> Option<Formula> {
  let mut memo = FormulaMemo::default();
  eliminate_recursive(formula.into_nnf().normalized(), &mut memo)
    .map(Formula::normalized)
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
    Formula::Not(_) => eliminate_recursive(formula.into_nnf(), memo),
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

/// Chooses the variable with the smallest estimated Fourier-Motzkin growth.
/// Equality pivots are cheapest; otherwise the principal cost is the number
/// of lower/upper pairs, followed by disequality splits and atom occurrences.
/// `Variable` is the final key, making ties stable across input permutations.
#[doc(hidden)]
pub fn choose_elimination_variable(
  formula: &Formula,
  variables: &BTreeSet<Variable>,
) -> Option<Variable> {
  variables
    .iter()
    .min_by_key(|variable| rational_elimination_cost(formula, variable))
    .cloned()
}

#[doc(hidden)]
pub fn rational_elimination_cost(
  formula: &Formula,
  variable: &Variable,
) -> (bool, usize, usize, usize) {
  fn visit(
    formula: &Formula,
    variable: &Variable,
    equality: &mut bool,
    disequalities: &mut usize,
    lowers: &mut usize,
    uppers: &mut usize,
    occurrences: &mut usize,
  ) {
    match formula {
      Formula::Atom(Atom::Relation(relation, term)) => {
        let coefficient = term.coefficient(variable);
        if coefficient.is_zero() {
          return;
        }
        *occurrences = occurrences.saturating_add(1);
        match relation {
          Relation::Equal => *equality = true,
          Relation::NotEqual => {
            *disequalities = disequalities.saturating_add(1);
          }
          Relation::Less | Relation::LessEqual => {
            if coefficient.numerator.sign() == Sign::Plus {
              *uppers = uppers.saturating_add(1);
            } else {
              *lowers = lowers.saturating_add(1);
            }
          }
          Relation::Greater | Relation::GreaterEqual => {
            if coefficient.numerator.sign() == Sign::Plus {
              *lowers = lowers.saturating_add(1);
            } else {
              *uppers = uppers.saturating_add(1);
            }
          }
        }
      }
      Formula::Atom(Atom::Divides { term, .. }) => {
        if !term.coefficient(variable).is_zero() {
          *occurrences = occurrences.saturating_add(1);
        }
      }
      Formula::And(children) | Formula::Or(children) => {
        for child in children {
          visit(
            child,
            variable,
            equality,
            disequalities,
            lowers,
            uppers,
            occurrences,
          );
        }
      }
      Formula::Not(inner) => visit(
        inner,
        variable,
        equality,
        disequalities,
        lowers,
        uppers,
        occurrences,
      ),
      Formula::Quantified(_, _, body) => visit(
        body,
        variable,
        equality,
        disequalities,
        lowers,
        uppers,
        occurrences,
      ),
      Formula::True | Formula::False => {}
    }
  }

  let mut equality = false;
  let mut disequalities = 0_usize;
  let mut lowers = 0_usize;
  let mut uppers = 0_usize;
  let mut occurrences = 0_usize;
  visit(
    formula,
    variable,
    &mut equality,
    &mut disequalities,
    &mut lowers,
    &mut uppers,
    &mut occurrences,
  );
  (
    !equality,
    lowers.saturating_mul(uppers),
    disequalities,
    occurrences,
  )
}

fn eliminate_forall(body: Formula, variable: &Variable) -> Option<Formula> {
  let negated = Formula::Not(Box::new(body)).into_nnf().normalized();
  let exists = eliminate_exists(negated, variable)?;
  Some(Formula::Not(Box::new(exists)).into_nnf().normalized())
}

fn eliminate_exists(body: Formula, variable: &Variable) -> Option<Formula> {
  let body = split_variable_disequalities(body, variable).normalized();
  let branches = to_dnf(body)?;
  let projected = branches
    .into_iter()
    .map(|branch| project_conjunction(branch, variable))
    .collect::<Option<Vec<_>>>()?;
  Some(Formula::Or(projected).normalized())
}

fn split_variable_disequalities(
  formula: Formula,
  variable: &Variable,
) -> Formula {
  match formula {
    Formula::Atom(Atom::Relation(Relation::NotEqual, term))
      if !term.coefficient(variable).is_zero() =>
    {
      Formula::Or(vec![
        Formula::Atom(Atom::Relation(Relation::Less, term.clone())),
        Formula::Atom(Atom::Relation(Relation::Greater, term)),
      ])
    }
    Formula::And(children) => Formula::And(
      children
        .into_iter()
        .map(|child| split_variable_disequalities(child, variable))
        .collect(),
    ),
    Formula::Or(children) => Formula::Or(
      children
        .into_iter()
        .map(|child| split_variable_disequalities(child, variable))
        .collect(),
    ),
    Formula::Quantified(_, _, _) | Formula::Not(_) => {
      unreachable!("inner quantifiers and negations are eliminated first")
    }
    leaf => leaf,
  }
}

/// Returns a disjunction of conjunctions, represented as `Vec<Vec<Atom>>`.
fn to_dnf(formula: Formula) -> Option<Vec<Vec<Atom>>> {
  match formula {
    Formula::True => Some(vec![Vec::new()]),
    Formula::False => Some(Vec::new()),
    Formula::Atom(atom) => Some(vec![vec![atom]]),
    Formula::Or(children) => {
      let mut output = Vec::new();
      for child in children {
        output.extend(to_dnf(child)?);
      }
      Some(output)
    }
    Formula::And(children) => {
      let mut product = vec![Vec::new()];
      for child in children {
        let alternatives = to_dnf(child)?;
        let mut next = Vec::new();
        for prefix in &product {
          for alternative in &alternatives {
            let mut conjunction = prefix.clone();
            conjunction.extend(alternative.iter().cloned());
            next.push(conjunction);
          }
        }
        product = next;
      }
      Some(product)
    }
    Formula::Not(_) | Formula::Quantified(_, _, _) => None,
  }
}

#[derive(Clone)]
struct Bound {
  value: AffineTerm,
  strict: bool,
}

fn project_conjunction(
  atoms: Vec<Atom>,
  variable: &Variable,
) -> Option<Formula> {
  if atoms
    .iter()
    .any(|atom| matches!(atom, Atom::Divides { .. }))
  {
    return None;
  }

  if let Some((pivot_index, replacement)) =
    atoms.iter().enumerate().find_map(|(index, atom)| {
      equality_solution(atom, variable).map(|r| (index, r))
    })
  {
    return Some(
      Formula::And(
        atoms
          .iter()
          .enumerate()
          .filter(|(index, _)| *index != pivot_index)
          .map(|(_, atom)| {
            Formula::Atom(atom.substitute(variable, &replacement))
          })
          .collect(),
      )
      .normalized(),
    );
  }

  let mut independent = Vec::new();
  let mut lower_bounds = Vec::new();
  let mut upper_bounds = Vec::new();
  for atom in atoms {
    let Atom::Relation(relation, term) = atom else {
      return None;
    };
    let coefficient = term.coefficient(variable);
    if coefficient.is_zero() {
      independent.push(Formula::Atom(Atom::Relation(relation, term)));
      continue;
    }
    if matches!(relation, Relation::Equal | Relation::NotEqual) {
      return None;
    }
    let boundary = solve_boundary(&term, variable, &coefficient)?;
    let strict = matches!(relation, Relation::Less | Relation::Greater);
    let coefficient_positive = coefficient.numerator.sign() == Sign::Plus;
    let is_upper = match relation {
      Relation::Less | Relation::LessEqual => coefficient_positive,
      Relation::Greater | Relation::GreaterEqual => !coefficient_positive,
      Relation::Equal | Relation::NotEqual => unreachable!(),
    };
    let bound = Bound {
      value: boundary,
      strict,
    };
    if is_upper {
      upper_bounds.push(bound);
    } else {
      lower_bounds.push(bound);
    }
  }

  for lower in &lower_bounds {
    for upper in &upper_bounds {
      independent.push(Formula::Atom(Atom::Relation(
        if lower.strict || upper.strict {
          Relation::Less
        } else {
          Relation::LessEqual
        },
        lower.value.subtract(&upper.value),
      )));
    }
  }
  Some(Formula::And(independent).normalized())
}

fn equality_solution(atom: &Atom, variable: &Variable) -> Option<AffineTerm> {
  let Atom::Relation(Relation::Equal, term) = atom else {
    return None;
  };
  let coefficient = term.coefficient(variable);
  if coefficient.is_zero() {
    return None;
  }
  solve_boundary(term, variable, &coefficient)
}

/// Solves `coefficient * variable + rest == 0` for the variable.
fn solve_boundary(
  term: &AffineTerm,
  variable: &Variable,
  coefficient: &Rational,
) -> Option<AffineTerm> {
  let mut rest = term.clone();
  rest.coefficients.remove(variable);
  let factor = Rational::integer((-1).into()).checked_divide(coefficient)?;
  Some(rest.scaled(&factor))
}
