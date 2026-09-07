//! Makes equations explicit in a quantifier-free elimination result.
//!
//! Both theory engines decide formulas over `<=`, `<` and divisibility atoms,
//! so an equation `t == 0` over the integers reaches the result as the pair
//! `t <= 0 && -t <= 0`, and an equation between free variables is never
//! solved. This pass restores the shape a reader expects:
//!
//! - complementary half-spaces in one conjunction fold back into `t == 0`;
//! - over the integers an equation is divided by the gcd of its coefficients,
//!   and a conjunction whose equation cannot be satisfied by integers
//!   (`2 x + 4 y == 5`) becomes `False`;
//! - an equation that can be solved for a target variable is substituted into
//!   the other conjuncts, so `x + y == 3 && x - y == 1` becomes
//!   `x == 2 && y == 1`, and `x + y == 3 && x >= 0 && y >= 0` becomes
//!   `0 <= x <= 3 && y == 3 - x`.
//!
//! Only target variables are ever solved for; other symbols are parameters.
//! Later targets are solved in terms of earlier ones, which is the order
//! Wolfram reports.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use super::affine::{AffineTerm, Variable};
use super::exact::{Rational, gcd};
use super::formula::{Atom, Formula, Relation};

/// Rewrites `formula` as described in the module documentation. With
/// `integral` set, equations are normalized and solved as integer equations;
/// otherwise any nonzero coefficient may serve as a pivot.
pub fn fold_equalities(
  formula: Formula,
  targets: &[Variable],
  integral: bool,
) -> Formula {
  match formula {
    Formula::And(children) => fold_conjunction(children, targets, integral),
    Formula::Or(children) => Formula::Or(
      children
        .into_iter()
        .map(|child| fold_equalities(child, targets, integral))
        .collect(),
    )
    .normalized(),
    Formula::Not(inner) => {
      Formula::Not(Box::new(fold_equalities(*inner, targets, integral)))
        .normalized()
    }
    Formula::Quantified(quantifier, variables, body) => Formula::Quantified(
      quantifier,
      variables,
      Box::new(fold_equalities(*body, targets, integral)),
    )
    .normalized(),
    Formula::Atom(Atom::Relation(Relation::Equal, term)) => {
      match primitive_equation(term, integral) {
        Some(term) => {
          Formula::Atom(Atom::Relation(Relation::Equal, term)).normalized()
        }
        None => Formula::False,
      }
    }
    leaf => leaf,
  }
}

fn fold_conjunction(
  children: Vec<Formula>,
  targets: &[Variable],
  integral: bool,
) -> Formula {
  let mut remaining = children
    .into_iter()
    .map(|child| fold_equalities(child, targets, integral))
    .collect::<Vec<_>>();
  let mut solved: Vec<(Variable, AffineTerm)> = Vec::new();
  loop {
    let Some(mut conjuncts) = conjuncts(Formula::And(remaining).normalized())
    else {
      return Formula::False;
    };
    let Some(equations) = fold_half_spaces(&mut conjuncts, integral) else {
      return Formula::False;
    };
    let Some(pivot) = equations.iter().find_map(|(index, term)| {
      pivot(term, targets, integral).map(|v| (*index, v))
    }) else {
      remaining = conjuncts;
      break;
    };
    let (index, variable) = pivot;
    let Formula::Atom(Atom::Relation(Relation::Equal, term)) =
      conjuncts.remove(index)
    else {
      unreachable!("equation indices come from the conjunct list");
    };
    let replacement = solve_for(&term, &variable);
    for (_, earlier) in &mut solved {
      *earlier = earlier.substitute(&variable, &replacement);
    }
    remaining = conjuncts
      .into_iter()
      .map(|child| child.substitute(&variable, &replacement))
      .collect();
    solved.push((variable, replacement));
  }
  let target_index =
    |variable: &Variable| targets.iter().position(|target| target == variable);
  solved.sort_by_key(|(variable, _)| target_index(variable));
  remaining.extend(solved.into_iter().map(|(variable, replacement)| {
    Formula::Atom(Atom::Relation(
      Relation::Equal,
      AffineTerm::variable(variable).subtract(&replacement),
    ))
  }));
  Formula::And(remaining).normalized()
}

/// The conjuncts of a normalized formula, or `None` for `False`.
fn conjuncts(formula: Formula) -> Option<Vec<Formula>> {
  match formula {
    Formula::False => None,
    Formula::True => Some(Vec::new()),
    Formula::And(children) => Some(children),
    other => Some(vec![other]),
  }
}

/// Replaces every complementary pair `t <= 0`, `-t <= 0` by `t == 0` and
/// normalizes every equation in place. Returns the positions of the equations
/// among the conjuncts, or `None` when an equation has no integer solution.
fn fold_half_spaces(
  conjuncts: &mut Vec<Formula>,
  integral: bool,
) -> Option<Vec<(usize, AffineTerm)>> {
  let mut folded = Vec::new();
  let mut index = 0;
  while index < conjuncts.len() {
    let Formula::Atom(Atom::Relation(Relation::LessEqual, term)) =
      &conjuncts[index]
    else {
      index += 1;
      continue;
    };
    let negated = term.scaled(&Rational::integer((-1).into()));
    let partner = conjuncts.iter().enumerate().position(|(other, child)| {
      other != index
        && matches!(
          child,
          Formula::Atom(Atom::Relation(Relation::LessEqual, candidate))
            if *candidate == negated
        )
    });
    match partner {
      Some(partner) => {
        let term = term.clone();
        conjuncts[index] = Formula::Atom(Atom::Relation(Relation::Equal, term));
        conjuncts.remove(partner);
        if partner > index {
          index += 1;
        }
      }
      None => index += 1,
    }
  }
  for (position, child) in conjuncts.iter_mut().enumerate() {
    if let Formula::Atom(Atom::Relation(Relation::Equal, term)) = child {
      let term = primitive_equation(term.clone(), integral)?;
      *child = Formula::Atom(Atom::Relation(Relation::Equal, term.clone()));
      folded.push((position, term));
    }
  }
  Some(folded)
}

/// Gives an equation a canonical sign (positive leading coefficient) and,
/// when `integral`, primitive integer coefficients. Returns `None` for an
/// integer equation whose coefficient gcd does not divide its constant, which
/// no integers satisfy.
fn primitive_equation(term: AffineTerm, integral: bool) -> Option<AffineTerm> {
  let Some((_, leading)) = term.coefficients.first_key_value() else {
    return Some(term);
  };
  let term = if leading.numerator.is_negative() {
    term.scaled(&Rational::integer((-1).into()))
  } else {
    term
  };
  if !integral {
    return Some(term);
  }
  let denominator = term
    .coefficients
    .values()
    .fold(term.constant.denominator.clone(), |common, coefficient| {
      super::exact::lcm(&common, &coefficient.denominator)
    });
  let term = term.scaled(&Rational::integer(denominator));
  let divisor = term
    .coefficients
    .values()
    .fold(BigInt::zero(), |common, coefficient| {
      gcd(common, coefficient.numerator.clone())
    });
  if (&term.constant.numerator % &divisor) != BigInt::zero() {
    return None;
  }
  if divisor.is_one() {
    return Some(term);
  }
  Some(term.scaled(&Rational::new(BigInt::one(), divisor).unwrap()))
}

/// The latest target that the equation can be solved for.
fn pivot(
  term: &AffineTerm,
  targets: &[Variable],
  integral: bool,
) -> Option<Variable> {
  targets
    .iter()
    .rev()
    .find(|target| {
      let coefficient = term.coefficient(target);
      if coefficient.is_zero() {
        return false;
      }
      !integral || coefficient.numerator.abs() == BigInt::one()
    })
    .cloned()
}

/// Solves `coefficient * variable + rest == 0` for the variable.
fn solve_for(term: &AffineTerm, variable: &Variable) -> AffineTerm {
  let coefficient = term.coefficient(variable);
  let mut rest = term.clone();
  rest.coefficients.remove(variable);
  rest.scaled(
    &Rational::integer((-1).into())
      .checked_divide(&coefficient)
      .expect("a pivot coefficient is nonzero"),
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  fn free(name: &str) -> Variable {
    Variable::free(name)
  }

  fn term(pairs: &[(&str, i64)], constant: i64) -> AffineTerm {
    let mut result = AffineTerm::constant(Rational::integer(constant.into()));
    for (name, coefficient) in pairs {
      result = result.add(
        &AffineTerm::variable(free(name))
          .scaled(&Rational::integer((*coefficient).into())),
      );
    }
    result
  }

  fn less_equal(term: AffineTerm) -> Formula {
    Formula::Atom(Atom::Relation(Relation::LessEqual, term))
  }

  fn equal(term: AffineTerm) -> Formula {
    Formula::Atom(Atom::Relation(Relation::Equal, term))
  }

  #[test]
  fn complementary_half_spaces_fold_into_an_equation() {
    let split = Formula::And(vec![
      less_equal(term(&[("x", 2), ("y", -4)], 0)),
      less_equal(term(&[("x", -2), ("y", 4)], 0)),
    ])
    .normalized();
    assert_eq!(
      fold_equalities(split, &[free("x"), free("y")], true),
      equal(term(&[("x", 1), ("y", -2)], 0)).normalized()
    );
  }

  #[test]
  fn an_integer_equation_with_no_solution_is_false() {
    let formula = Formula::And(vec![
      equal(term(&[("x", 2), ("y", 4)], -5)),
      less_equal(term(&[("x", -1)], 0)),
    ])
    .normalized();
    assert_eq!(
      fold_equalities(formula, &[free("x"), free("y")], true),
      Formula::False
    );
  }

  #[test]
  fn determined_systems_are_solved_completely() {
    let system = Formula::And(vec![
      equal(term(&[("x", 1), ("y", 1)], -3)),
      equal(term(&[("x", 1), ("y", -1)], -1)),
    ])
    .normalized();
    assert_eq!(
      fold_equalities(system, &[free("x"), free("y")], true),
      Formula::And(vec![
        equal(term(&[("x", 1)], -2)),
        equal(term(&[("y", 1)], -1)),
      ])
      .normalized()
    );
  }

  #[test]
  fn later_targets_are_solved_in_terms_of_earlier_ones() {
    let formula = Formula::And(vec![
      equal(term(&[("x", 1), ("y", 1)], -3)),
      less_equal(term(&[("x", -1)], 0)),
      less_equal(term(&[("y", -1)], 0)),
    ])
    .normalized();
    assert_eq!(
      fold_equalities(formula, &[free("x"), free("y")], true),
      Formula::And(vec![
        less_equal(term(&[("x", -1)], 0)),
        less_equal(term(&[("x", 1)], -3)),
        equal(term(&[("x", 1), ("y", 1)], -3)),
      ])
      .normalized()
    );
  }

  #[test]
  fn parameters_are_never_solved_for() {
    let formula = equal(term(&[("a", 1), ("x", 2)], -1)).normalized();
    assert_eq!(
      fold_equalities(formula.clone(), &[free("x")], true),
      formula
    );
    assert_eq!(
      fold_equalities(formula, &[free("x")], false),
      equal(term(&[("a", 1), ("x", 2)], -1)).normalized()
    );
  }
}
