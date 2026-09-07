//! Complete finite-model search for quantifier-free affine integer formulas.
//!
//! This is deliberately a client of the same exact lowering and normalization
//! used by `Reduce`. It only claims a result after proving that every requested
//! variable is bounded in every Boolean branch; an affine but unbounded formula
//! is `Unsupported`, never an empty solution set.

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::{BigInt, Sign};
use num_traits::{One, ToPrimitive, Zero};
use woxi_reduce::{AffineTerm, Atom, Formula, Relation, Variable, gcd};

use crate::syntax::Expr;

const MAX_DNF_BRANCHES: usize = 4096;
const MAX_ENUMERATED_POINTS: u64 = 1_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FiniteIntegerSolve {
  /// The expression is outside quantifier-free affine integer arithmetic.
  NotApplicable,
  /// The expression is affine integer arithmetic, but its models were not
  /// proved finite (or a deterministic resource guard was reached).
  Unsupported,
  /// Every model, in the requested variable order.
  Solved(Vec<Vec<BigInt>>),
  /// The supported finite formula was proved to have no models.
  Infeasible,
}

#[derive(Clone, Debug, Default)]
struct Bounds {
  lower: Option<BigInt>,
  upper: Option<BigInt>,
}

#[derive(Clone, Debug)]
struct EnumerationDomain {
  start: BigInt,
  upper: BigInt,
  step: BigInt,
  count: BigInt,
}

pub(crate) fn solve_finite_integer(
  constraints: &Expr,
  variables: &Expr,
) -> FiniteIntegerSolve {
  let Some(targets) = target_variables(variables) else {
    return FiniteIntegerSolve::NotApplicable;
  };
  if targets.is_empty() {
    return FiniteIntegerSolve::NotApplicable;
  }

  let Some(formula) = constraints_formula(constraints) else {
    return FiniteIntegerSolve::NotApplicable;
  };
  if formula == Formula::False {
    return FiniteIntegerSolve::Infeasible;
  }
  if formula == Formula::True {
    return FiniteIntegerSolve::NotApplicable;
  }
  let target_set = targets.iter().cloned().collect::<BTreeSet<_>>();
  let free = formula.free_variables();
  if !free.is_subset(&target_set) || !target_set.is_subset(&free) {
    return FiniteIntegerSolve::Unsupported;
  }

  let Some(formula) =
    woxi_reduce::presburger::normalize_integer_formula(formula)
  else {
    return FiniteIntegerSolve::NotApplicable;
  };
  let Some(branches) = dnf_branches(formula, MAX_DNF_BRANCHES) else {
    return FiniteIntegerSolve::Unsupported;
  };

  let mut solutions = BTreeSet::new();
  for atoms in branches {
    match solve_conjunction(&atoms, &targets) {
      BranchSolve::Unsupported => return FiniteIntegerSolve::Unsupported,
      BranchSolve::Infeasible => {}
      BranchSolve::Solved(branch_solutions) => {
        solutions.extend(branch_solutions);
      }
    }
  }

  if solutions.is_empty() {
    FiniteIntegerSolve::Infeasible
  } else {
    FiniteIntegerSolve::Solved(solutions.into_iter().collect())
  }
}

fn target_variables(expr: &Expr) -> Option<Vec<Variable>> {
  let names = match expr {
    Expr::Identifier(name) => vec![name.clone()],
    Expr::List(items) => items
      .iter()
      .map(|item| match item {
        Expr::Identifier(name) => Some(name.clone()),
        _ => None,
      })
      .collect::<Option<Vec<_>>>()?,
    _ => return None,
  };
  if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
    return None;
  }
  Some(names.into_iter().map(Variable::free).collect())
}

fn constraints_formula(expr: &Expr) -> Option<Formula> {
  match expr {
    Expr::List(items) => Some(
      Formula::And(
        items
          .iter()
          .map(super::lower::formula_from_expr)
          .collect::<Option<Vec<_>>>()?,
      )
      .normalized(),
    ),
    _ => super::lower::formula_from_expr(expr),
  }
}

/// Converts NNF to explicit conjunction branches transactionally. The guard
/// prevents adversarial Boolean products from causing exponential growth.
fn dnf_branches(formula: Formula, limit: usize) -> Option<Vec<Vec<Atom>>> {
  match formula {
    Formula::True => Some(vec![Vec::new()]),
    Formula::False => Some(Vec::new()),
    Formula::Atom(atom) => Some(vec![vec![atom]]),
    Formula::Or(children) => {
      let mut result = Vec::new();
      for child in children {
        result.extend(dnf_branches(child, limit)?);
        if result.len() > limit {
          return None;
        }
      }
      Some(result)
    }
    Formula::And(children) => {
      let mut result = vec![Vec::new()];
      for child in children {
        let alternatives = dnf_branches(child, limit)?;
        if alternatives.is_empty() {
          return Some(Vec::new());
        }
        if result.len().saturating_mul(alternatives.len()) > limit {
          return None;
        }
        let mut product = Vec::with_capacity(result.len() * alternatives.len());
        for prefix in &result {
          for suffix in &alternatives {
            let mut branch = prefix.clone();
            branch.extend(suffix.iter().cloned());
            product.push(branch);
          }
        }
        result = product;
      }
      Some(result)
    }
    Formula::Not(_) | Formula::Quantified(_, _, _) => None,
  }
}

enum BranchSolve {
  Unsupported,
  Infeasible,
  Solved(Vec<Vec<BigInt>>),
}

fn solve_conjunction(atoms: &[Atom], targets: &[Variable]) -> BranchSolve {
  let mut bounds = targets
    .iter()
    .cloned()
    .map(|variable| (variable, Bounds::default()))
    .collect::<BTreeMap<_, _>>();

  // Interval propagation is monotone: each successful update tightens a
  // bound. Iterate until a full pass makes no progress.
  loop {
    let mut changed = false;
    for atom in atoms {
      let Atom::Relation(Relation::LessEqual, term) = atom else {
        continue;
      };
      if !term.constant.is_integer()
        || term
          .coefficients
          .values()
          .any(|coefficient| !coefficient.is_integer())
      {
        return BranchSolve::Unsupported;
      }
      for target in targets {
        let coefficient = term.coefficient(target).numerator;
        if coefficient.is_zero() {
          continue;
        }
        let Some(minimum_rest) = minimum_rest(term, target, &bounds) else {
          continue;
        };
        let candidate = if coefficient.sign() == Sign::Plus {
          BoundCandidate::Upper(floor_div(&(-minimum_rest), &coefficient))
        } else {
          BoundCandidate::Lower(ceil_div(&(-minimum_rest), &coefficient))
        };
        changed |= update_bound(bounds.get_mut(target).unwrap(), candidate);
      }
    }
    if bounds.values().any(bounds_are_empty) {
      return BranchSolve::Infeasible;
    }
    if !changed {
      break;
    }
  }

  if bounds
    .values()
    .any(|bound| bound.lower.is_none() || bound.upper.is_none())
  {
    return BranchSolve::Unsupported;
  }

  let mut domains = BTreeMap::new();
  let mut total = BigInt::one();
  for target in targets {
    let Some(domain) = enumeration_domain(target, &bounds[target], atoms)
    else {
      return BranchSolve::Infeasible;
    };
    total *= &domain.count;
    if total > BigInt::from(MAX_ENUMERATED_POINTS) {
      return BranchSolve::Unsupported;
    }
    domains.insert(target.clone(), domain);
  }

  let mut assignment = BTreeMap::new();
  let mut solutions = Vec::new();
  enumerate(0, targets, &domains, atoms, &mut assignment, &mut solutions);
  if solutions.is_empty() {
    BranchSolve::Infeasible
  } else {
    BranchSolve::Solved(solutions)
  }
}

enum BoundCandidate {
  Lower(BigInt),
  Upper(BigInt),
}

fn update_bound(bound: &mut Bounds, candidate: BoundCandidate) -> bool {
  match candidate {
    BoundCandidate::Lower(value)
      if bound.lower.as_ref().is_none_or(|old| value > *old) =>
    {
      bound.lower = Some(value);
      true
    }
    BoundCandidate::Upper(value)
      if bound.upper.as_ref().is_none_or(|old| value < *old) =>
    {
      bound.upper = Some(value);
      true
    }
    BoundCandidate::Lower(_) | BoundCandidate::Upper(_) => false,
  }
}

fn bounds_are_empty(bound: &Bounds) -> bool {
  matches!((&bound.lower, &bound.upper), (Some(lower), Some(upper)) if lower > upper)
}

fn minimum_rest(
  term: &AffineTerm,
  omitted: &Variable,
  bounds: &BTreeMap<Variable, Bounds>,
) -> Option<BigInt> {
  let mut value = term.constant.numerator.clone();
  for (variable, coefficient) in &term.coefficients {
    if variable == omitted || coefficient.is_zero() {
      continue;
    }
    let bound = bounds.get(variable)?;
    let extremum = if coefficient.numerator.sign() == Sign::Plus {
      bound.lower.as_ref()?
    } else {
      bound.upper.as_ref()?
    };
    value += &coefficient.numerator * extremum;
  }
  Some(value)
}

fn floor_div(numerator: &BigInt, denominator: &BigInt) -> BigInt {
  let quotient = numerator / denominator;
  let remainder = numerator % denominator;
  if !remainder.is_zero() && numerator.sign() != denominator.sign() {
    quotient - BigInt::one()
  } else {
    quotient
  }
}

fn ceil_div(numerator: &BigInt, denominator: &BigInt) -> BigInt {
  let quotient = numerator / denominator;
  let remainder = numerator % denominator;
  if !remainder.is_zero() && numerator.sign() == denominator.sign() {
    quotient + BigInt::one()
  } else {
    quotient
  }
}

/// Uses the strongest one-variable divisibility atom as an enumeration stride.
/// Remaining congruences are still checked by `atom_holds`, so this is purely a
/// completeness-preserving search reduction.
fn enumeration_domain(
  variable: &Variable,
  bounds: &Bounds,
  atoms: &[Atom],
) -> Option<EnumerationDomain> {
  let lower = bounds.lower.as_ref().unwrap();
  let upper = bounds.upper.as_ref().unwrap();
  let mut stride = (BigInt::zero(), BigInt::one());
  for atom in atoms {
    if let Some(candidate) = unary_congruence(atom, variable) {
      let candidate = candidate?;
      if candidate.1 > stride.1 {
        stride = candidate;
      }
    }
  }
  let (residue, step) = stride;
  let start = lower + positive_mod(&(residue - lower), &step);
  if &start > upper {
    return None;
  }
  let count = (upper - &start) / &step + BigInt::one();
  Some(EnumerationDomain {
    start,
    upper: upper.clone(),
    step,
    count,
  })
}

/// Returns `Some(None)` when the unary congruence itself is inconsistent.
fn unary_congruence(
  atom: &Atom,
  variable: &Variable,
) -> Option<Option<(BigInt, BigInt)>> {
  let Atom::Divides {
    modulus,
    term,
    negated: false,
  } = atom
  else {
    return None;
  };
  if !term.constant.is_integer()
    || term.coefficients.len() != 1
    || !term.coefficients.contains_key(variable)
  {
    return None;
  }
  let coefficient = &term.coefficients[variable];
  if !coefficient.is_integer() || coefficient.is_zero() {
    return None;
  }

  let divisor = gcd(coefficient.numerator.clone(), modulus.clone());
  let right = -term.constant.numerator.clone();
  if !(&right % &divisor).is_zero() {
    return Some(None);
  }
  let reduced_modulus = modulus / &divisor;
  if reduced_modulus.is_one() {
    return Some(Some((BigInt::zero(), BigInt::one())));
  }
  let reduced_coefficient = &coefficient.numerator / &divisor;
  let reduced_right = right / divisor;
  let coefficient_mod = positive_mod(&reduced_coefficient, &reduced_modulus);
  let (_, inverse, _) = extended_gcd(coefficient_mod, reduced_modulus.clone());
  let residue = positive_mod(&(reduced_right * inverse), &reduced_modulus);
  Some(Some((residue, reduced_modulus)))
}

fn positive_mod(value: &BigInt, modulus: &BigInt) -> BigInt {
  let residue = value % modulus;
  if residue.sign() == Sign::Minus {
    residue + modulus
  } else {
    residue
  }
}

fn extended_gcd(left: BigInt, right: BigInt) -> (BigInt, BigInt, BigInt) {
  if right.is_zero() {
    return (left, BigInt::one(), BigInt::zero());
  }
  let quotient = &left / &right;
  let remainder = &left % &right;
  let (gcd, x, y) = extended_gcd(right, remainder);
  (gcd, y.clone(), x - quotient * y)
}

fn enumerate(
  index: usize,
  targets: &[Variable],
  domains: &BTreeMap<Variable, EnumerationDomain>,
  atoms: &[Atom],
  assignment: &mut BTreeMap<Variable, BigInt>,
  solutions: &mut Vec<Vec<BigInt>>,
) {
  if index == targets.len() {
    if atoms.iter().all(|atom| atom_holds(atom, assignment)) {
      solutions.push(
        targets
          .iter()
          .map(|target| assignment[target].clone())
          .collect(),
      );
    }
    return;
  }

  let variable = &targets[index];
  let domain = &domains[variable];
  let mut value = domain.start.clone();
  while value <= domain.upper {
    assignment.insert(variable.clone(), value.clone());
    enumerate(index + 1, targets, domains, atoms, assignment, solutions);
    value += &domain.step;
  }
  assignment.remove(variable);
}

fn atom_holds(atom: &Atom, assignment: &BTreeMap<Variable, BigInt>) -> bool {
  match atom {
    Atom::Relation(Relation::LessEqual, term) => {
      evaluate_term(term, assignment)
        .is_some_and(|value| value <= BigInt::zero())
    }
    Atom::Divides {
      modulus,
      term,
      negated,
    } => evaluate_term(term, assignment).is_some_and(|value| {
      let divides = (&value % modulus).is_zero();
      if *negated { !divides } else { divides }
    }),
    Atom::Relation(_, _) => false,
  }
}

fn evaluate_term(
  term: &AffineTerm,
  assignment: &BTreeMap<Variable, BigInt>,
) -> Option<BigInt> {
  if !term.constant.is_integer()
    || term
      .coefficients
      .values()
      .any(|coefficient| !coefficient.is_integer())
  {
    return None;
  }
  let mut value = term.constant.numerator.clone();
  for (variable, coefficient) in &term.coefficients {
    value += &coefficient.numerator * assignment.get(variable)?;
  }
  Some(value)
}

pub(crate) fn integer_expr(value: BigInt) -> Expr {
  value
    .to_i128()
    .map_or_else(|| Expr::BigInteger(value), Expr::Integer)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn signed_integer_division_rounds_outward() {
    assert_eq!(floor_div(&BigInt::from(7), &BigInt::from(3)), 2.into());
    assert_eq!(floor_div(&BigInt::from(-7), &BigInt::from(3)), (-3).into());
    assert_eq!(ceil_div(&BigInt::from(7), &BigInt::from(3)), 3.into());
    assert_eq!(ceil_div(&BigInt::from(-7), &BigInt::from(3)), (-2).into());
    assert_eq!(ceil_div(&BigInt::from(7), &BigInt::from(-3)), (-2).into());
  }
}
