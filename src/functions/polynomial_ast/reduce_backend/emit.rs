//! Deterministic conversion from the linear IR back to Woxi expressions.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use woxi_reduce::{
  AffineTerm, Atom, Formula, Quantifier, Rational, Relation, Variable,
  ceil_div, crt_pair, euclidean_mod, floor_div, lcm, solve_linear_congruence,
};

use crate::helpers::call;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr, UnaryOperator};

use super::{bigint_expr, rational_expr};

/// Finite single-variable results are materialized from the disjunctive
/// normal form of the decided formula; past this many branches the result is
/// printed structurally instead.
const MAX_FINITE_DNF_BRANCHES: usize = 100_000;

pub(super) fn formula_expr(formula: &Formula) -> Expr {
  formula_expr_for_targets(formula, &[], false)
}

/// Prints a decided formula solved for `targets`. `integral` says the result
/// describes integers, which changes which variable an equation is solved
/// for.
pub(super) fn formula_expr_for_targets(
  formula: &Formula,
  targets: &[Variable],
  integral: bool,
) -> Expr {
  match formula {
    Formula::True => Expr::Identifier("True".to_string()),
    Formula::False => Expr::Identifier("False".to_string()),
    Formula::Atom(atom) => atom_expr(atom, targets, integral),
    Formula::And(children) => conjunction_expr(children, targets, integral),
    Formula::Or(children) => {
      fold_binary(children, BinaryOperator::Or, targets, integral)
    }
    Formula::Not(inner) => Expr::UnaryOp {
      op: UnaryOperator::Not,
      operand: Box::new(formula_expr_for_targets(inner, targets, integral)),
    },
    Formula::Quantified(quantifier, variables, body) => call(
      match quantifier {
        Quantifier::Exists => "Exists",
        Quantifier::ForAll => "ForAll",
      },
      vec![
        variables_expr(variables),
        formula_expr_for_targets(body, targets, integral),
      ],
    ),
  }
}

/// Emits a finite unary integer result by deriving its interval and residue
/// class exactly. Iteration here materializes the required finite output; it
/// is not used to decide satisfiability or to search for a witness.
pub(super) fn finite_integer_target_expr(
  formula: &Formula,
  target: &Variable,
) -> Option<Expr> {
  let branches = formula.clone().dnf_branches(MAX_FINITE_DNF_BRANCHES)?;
  let mut values = BTreeSet::new();
  for branch in branches {
    values.extend(finite_integer_branch_values(&branch, target)?);
    if values.len() > 1_000_000 {
      return None;
    }
  }
  Some(fold_owned_binary(
    values
      .into_iter()
      .map(|value| Expr::Comparison {
        operands: vec![
          Expr::Identifier(target.name.clone()),
          bigint_expr(&value),
        ],
        operators: vec![ComparisonOp::Equal],
      })
      .collect(),
    BinaryOperator::Or,
  ))
}

fn finite_integer_branch_values(
  atoms: &[Atom],
  target: &Variable,
) -> Option<Vec<BigInt>> {
  let mut lower: Option<BigInt> = None;
  let mut upper: Option<BigInt> = None;
  let mut congruence = (BigInt::zero(), BigInt::one());
  let mut forbidden = Vec::new();

  for atom in atoms {
    match atom {
      Atom::Relation(Relation::Equal, term) => {
        if term.coefficients.len() != 1
          || !term.constant.is_integer()
          || !term.coefficient(target).is_integer()
        {
          return None;
        }
        let coefficient = term.coefficient(target).numerator;
        if coefficient.is_zero() {
          return None;
        }
        let value = -term.constant.numerator.clone();
        if !(&value % &coefficient).is_zero() {
          return Some(Vec::new());
        }
        let value = value / coefficient;
        upper = Some(match upper {
          Some(old) => std::cmp::min(old, value.clone()),
          None => value.clone(),
        });
        lower = Some(match lower {
          Some(old) => std::cmp::max(old, value),
          None => value,
        });
      }
      Atom::Relation(Relation::LessEqual, term) => {
        if term.coefficients.len() != 1
          || !term.constant.is_integer()
          || !term.coefficient(target).is_integer()
        {
          return None;
        }
        let coefficient = term.coefficient(target).numerator;
        if coefficient.is_positive() {
          let bound =
            floor_div(&(-term.constant.numerator.clone()), &coefficient);
          upper = Some(match upper {
            Some(old) => std::cmp::min(old, bound),
            None => bound,
          });
        } else if coefficient.is_negative() {
          let bound =
            ceil_div(&(-term.constant.numerator.clone()), &coefficient);
          lower = Some(match lower {
            Some(old) => std::cmp::max(old, bound),
            None => bound,
          });
        } else {
          return None;
        }
      }
      Atom::Divides {
        modulus,
        term,
        negated,
      } => {
        if term.coefficients.len() != 1
          || !term.constant.is_integer()
          || !term.coefficient(target).is_integer()
        {
          return None;
        }
        let solved = solve_linear_congruence(
          &term.coefficient(target).numerator,
          &(-term.constant.numerator.clone()),
          modulus,
        );
        if *negated {
          if let Some(solved) = solved {
            forbidden.push(solved);
          }
        } else {
          let Some(solved) = solved else {
            return Some(Vec::new());
          };
          let Some(combined) =
            crt_pair(&congruence.0, &congruence.1, &solved.0, &solved.1)
          else {
            return Some(Vec::new());
          };
          congruence = combined;
        }
      }
      Atom::Relation(..) => return None,
    }
  }

  let (lower, upper) = (lower?, upper?);
  if lower > upper {
    return Some(Vec::new());
  }
  let combined_period = forbidden
    .iter()
    .fold(congruence.1.clone(), |period, (_, forbidden_period)| {
      lcm(&period, forbidden_period)
    });
  let residue_count = &combined_period / &congruence.1;
  if residue_count > BigInt::from(1_000_000_u32) {
    return None;
  }

  let mut output = Vec::new();
  let mut residue_index = BigInt::zero();
  while residue_index < residue_count {
    let residue = euclidean_mod(
      &congruence.0 + &residue_index * &congruence.1,
      &combined_period,
    );
    residue_index += 1;
    if forbidden.iter().any(|(forbidden_residue, period)| {
      euclidean_mod(residue.clone(), period) == *forbidden_residue
    }) {
      continue;
    }
    let mut value = &residue
      + ceil_div(&(&lower - &residue), &combined_period) * &combined_period;
    let output_size = if value > upper {
      BigInt::zero()
    } else {
      floor_div(&(&upper - &value), &combined_period) + 1
    };
    if BigInt::from(output.len()) + output_size > BigInt::from(1_000_000_u32) {
      return None;
    }
    while value <= upper {
      output.push(value.clone());
      value += &combined_period;
    }
  }
  Some(output)
}

/// Rounds concrete unary integer bounds inward and reduces concrete unary
/// congruences to a unit-coefficient canonical residue.
pub(super) fn canonical_integer_formula(
  formula: &Formula,
  targets: &[Variable],
) -> Formula {
  match formula {
    Formula::Atom(Atom::Relation(Relation::LessEqual, term))
      if term.coefficients.len() == 1
        && term.constant.is_integer()
        && targets.contains(term.coefficients.first_key_value().unwrap().0) =>
    {
      let (variable, coefficient) =
        term.coefficients.first_key_value().unwrap();
      if !coefficient.is_integer() || coefficient.is_zero() {
        return formula.clone();
      }
      let positive = coefficient.numerator.is_positive();
      let (coefficient, bound) = if positive {
        (
          Rational::one(),
          floor_div(
            &(-term.constant.numerator.clone()),
            &coefficient.numerator,
          ),
        )
      } else {
        (
          Rational::integer(BigInt::from(-1)),
          ceil_div(&(-term.constant.numerator.clone()), &coefficient.numerator),
        )
      };
      let term = AffineTerm::variable(variable.clone())
        .scaled(&coefficient)
        .add(&AffineTerm::constant(if positive {
          Rational::integer(-bound)
        } else {
          Rational::integer(bound)
        }));
      Formula::Atom(Atom::Relation(Relation::LessEqual, term))
    }
    Formula::Atom(Atom::Divides {
      modulus,
      term,
      negated,
    }) if term.coefficients.len() == 1
      && term.constant.is_integer()
      && targets.contains(term.coefficients.first_key_value().unwrap().0) =>
    {
      let (variable, coefficient) =
        term.coefficients.first_key_value().unwrap();
      if !coefficient.is_integer() {
        return formula.clone();
      }
      let Some((residue, period)) = solve_linear_congruence(
        &coefficient.numerator,
        &(-term.constant.numerator.clone()),
        modulus,
      ) else {
        return if *negated {
          Formula::True
        } else {
          Formula::False
        };
      };
      Formula::Atom(
        Atom::divides(
          period,
          AffineTerm::variable(variable.clone())
            .subtract(&AffineTerm::constant(Rational::integer(residue))),
          *negated,
        )
        .expect("a solved congruence is integral with positive modulus"),
      )
    }
    Formula::And(children) => Formula::And(
      children
        .iter()
        .map(|child| canonical_integer_formula(child, targets))
        .collect(),
    )
    .normalized(),
    Formula::Or(children) => Formula::Or(
      children
        .iter()
        .map(|child| canonical_integer_formula(child, targets))
        .collect(),
    )
    .normalized(),
    Formula::Not(inner) => {
      Formula::Not(Box::new(canonical_integer_formula(inner, targets)))
        .normalized()
    }
    Formula::Quantified(quantifier, variables, body) => Formula::Quantified(
      *quantifier,
      variables.clone(),
      Box::new(canonical_integer_formula(body, targets)),
    )
    .normalized(),
    Formula::True
    | Formula::False
    | Formula::Atom(Atom::Relation(_, _) | Atom::Divides { .. }) => {
      formula.clone()
    }
  }
}

fn atom_expr(atom: &Atom, targets: &[Variable], integral: bool) -> Expr {
  match atom {
    Atom::Relation(relation, term) => {
      relation_expr(*relation, term, targets, integral)
    }
    Atom::Divides {
      modulus,
      term,
      negated,
    } => congruence_expr(modulus, term, *negated),
  }
}

/// `m | (t + c)` prints as the congruence `Mod[t, m] == r` it was most likely
/// written as, with the residue `r = -c mod m`.
fn congruence_expr(modulus: &BigInt, term: &AffineTerm, negated: bool) -> Expr {
  let residue = euclidean_mod(-term.constant.numerator.clone(), modulus);
  let mut variable_part = term.clone();
  variable_part.constant = Rational::zero();
  Expr::Comparison {
    operands: vec![
      call("Mod", vec![term_expr(&variable_part), bigint_expr(modulus)]),
      bigint_expr(&residue),
    ],
    operators: vec![if negated {
      ComparisonOp::NotEqual
    } else {
      ComparisonOp::Equal
    }],
  }
}

fn relation_expr(
  relation: Relation,
  term: &AffineTerm,
  targets: &[Variable],
  integral: bool,
) -> Expr {
  // Wolfram solves each relation for the last of the requested variables it
  // mentions, treating the earlier ones as the outer coordinates. An integer
  // equation is solved for a variable with a unit coefficient when it has
  // one, so `x - 2 y == 0` prints as `x == 2 y` rather than `y == x/2`.
  let mut mentioned = targets
    .iter()
    .rev()
    .filter(|target| !term.coefficient(target).is_zero());
  let isolated = if integral && relation == Relation::Equal {
    mentioned
      .clone()
      .find(|target| term.coefficient(target).numerator.abs().is_one())
      .or_else(|| mentioned.next())
  } else {
    mentioned.next()
  }
  .cloned()
  .or_else(|| {
    (term.coefficients.len() == 1)
      .then(|| term.coefficients.keys().next().unwrap().clone())
  });
  if let Some(variable) = isolated
    && let Some((relation, boundary)) =
      isolate_relation(relation, term, &variable)
  {
    return Expr::Comparison {
      operands: vec![Expr::Identifier(variable.name), term_expr(&boundary)],
      operators: vec![comparison_op(relation)],
    };
  }
  Expr::Comparison {
    operands: vec![term_expr(term), Expr::Integer(0)],
    operators: vec![comparison_op(relation)],
  }
}

fn isolate_relation(
  relation: Relation,
  term: &AffineTerm,
  variable: &Variable,
) -> Option<(Relation, AffineTerm)> {
  let coefficient = term.coefficient(variable);
  if coefficient.is_zero() {
    return None;
  }
  let mut rest = term.clone();
  rest.coefficients.remove(variable);
  let factor = Rational::integer((-1).into()).checked_divide(&coefficient)?;
  let boundary = rest.scaled(&factor);
  let relation = if coefficient.numerator.sign() == num_bigint::Sign::Minus {
    reverse_order(relation)
  } else {
    relation
  };
  Some((relation, boundary))
}

/// Emits a conjunction. Two one-sided bounds on the same variable whose
/// boundaries involve no requested variable merge into one `Inequality`
/// chain, or into an equation when they coincide. Conjuncts are ordered as
/// Wolfram orders them: conditions on parameters first, then on the requested
/// variables in the order they were asked for.
fn conjunction_expr(
  children: &[Formula],
  targets: &[Variable],
  integral: bool,
) -> Expr {
  let mut consumed = vec![false; children.len()];
  let mut items: Vec<(Option<usize>, Expr)> = Vec::new();
  for (position, target) in targets.iter().enumerate() {
    let mut lower = None;
    let mut upper = None;
    for (index, child) in children.iter().enumerate() {
      if consumed[index] {
        continue;
      }
      let Formula::Atom(Atom::Relation(relation, term)) = child else {
        continue;
      };
      let Some((relation, boundary)) =
        isolate_relation(*relation, term, target)
      else {
        continue;
      };
      if targets
        .iter()
        .any(|other| !boundary.coefficient(other).is_zero())
      {
        continue;
      }
      match relation {
        Relation::Greater | Relation::GreaterEqual if lower.is_none() => {
          lower = Some((index, relation, boundary));
        }
        Relation::Less | Relation::LessEqual if upper.is_none() => {
          upper = Some((index, relation, boundary));
        }
        _ => {}
      }
    }
    if let (Some(lower), Some(upper)) = (lower, upper) {
      consumed[lower.0] = true;
      consumed[upper.0] = true;
      items.push((
        Some(position),
        bounds_expr(target, lower.1, &lower.2, upper.1, &upper.2),
      ));
    }
  }
  for (index, child) in children.iter().enumerate() {
    if consumed[index] {
      continue;
    }
    let key = targets
      .iter()
      .rposition(|target| child.contains_variable(target));
    items.push((key, formula_expr_for_targets(child, targets, integral)));
  }
  items.sort_by_key(|(key, _)| *key);
  fold_owned_binary(
    items
      .into_iter()
      .map(|(_, expression)| expression)
      .collect(),
    BinaryOperator::And,
  )
}

fn bounds_expr(
  target: &Variable,
  lower_relation: Relation,
  lower_bound: &AffineTerm,
  upper_relation: Relation,
  upper_bound: &AffineTerm,
) -> Expr {
  if lower_bound == upper_bound {
    let closed = lower_relation == Relation::GreaterEqual
      && upper_relation == Relation::LessEqual;
    return if closed {
      Expr::Comparison {
        operands: vec![
          Expr::Identifier(target.name.clone()),
          term_expr(lower_bound),
        ],
        operators: vec![ComparisonOp::Equal],
      }
    } else {
      Expr::Identifier("False".to_string())
    };
  }
  call(
    "Inequality",
    vec![
      term_expr(lower_bound),
      Expr::Identifier(
        match lower_relation {
          Relation::Greater => "Less",
          Relation::GreaterEqual => "LessEqual",
          _ => unreachable!(),
        }
        .to_string(),
      ),
      Expr::Identifier(target.name.clone()),
      Expr::Identifier(
        match upper_relation {
          Relation::Less => "Less",
          Relation::LessEqual => "LessEqual",
          _ => unreachable!(),
        }
        .to_string(),
      ),
      term_expr(upper_bound),
    ],
  )
}

fn reverse_order(relation: Relation) -> Relation {
  match relation {
    Relation::Less => Relation::Greater,
    Relation::LessEqual => Relation::GreaterEqual,
    Relation::Greater => Relation::Less,
    Relation::GreaterEqual => Relation::LessEqual,
    Relation::Equal | Relation::NotEqual => relation,
  }
}

/// Prints a term with its constant first, the order `Plus` sorts to.
pub(super) fn term_expr(term: &AffineTerm) -> Expr {
  let mut summands = Vec::new();
  if !term.constant.is_zero() || term.coefficients.is_empty() {
    summands.push(rational_expr(&term.constant));
  }
  for (variable, coefficient) in &term.coefficients {
    summands.push(coefficient_variable_expr(coefficient, variable));
  }
  fold_owned_binary(summands, BinaryOperator::Plus)
}

/// Whether the formula fixes `target` to one integer on every branch, so a
/// domain-membership conjunct for it would be redundant.
pub(super) fn target_is_pinned(formula: &Formula, target: &Variable) -> bool {
  match formula {
    Formula::Atom(Atom::Relation(Relation::Equal, term)) => {
      term.coefficients.len() == 1
        && term.coefficient(target) == Rational::one()
        && term.constant.is_integer()
    }
    Formula::And(children) => {
      children.iter().any(|child| target_is_pinned(child, target))
    }
    Formula::Or(children) => {
      children.iter().all(|child| target_is_pinned(child, target))
    }
    _ => false,
  }
}

fn coefficient_variable_expr(
  coefficient: &Rational,
  variable: &Variable,
) -> Expr {
  let variable = Expr::Identifier(variable.name.clone());
  if coefficient.numerator == BigInt::one() && coefficient.denominator.is_one()
  {
    variable
  } else if coefficient.numerator == -BigInt::one()
    && coefficient.denominator.is_one()
  {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(variable),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(rational_expr(coefficient)),
      right: Box::new(variable),
    }
  }
}

fn variables_expr(variables: &[Variable]) -> Expr {
  if variables.len() == 1 {
    Expr::Identifier(variables[0].name.clone())
  } else {
    Expr::List(
      variables
        .iter()
        .map(|variable| Expr::Identifier(variable.name.clone()))
        .collect::<Vec<_>>()
        .into(),
    )
  }
}

fn fold_binary(
  children: &[Formula],
  operator: BinaryOperator,
  targets: &[Variable],
  integral: bool,
) -> Expr {
  let mut ordered = children.iter().collect::<Vec<_>>();
  if operator == BinaryOperator::Or {
    // Regions along the first variable's axis in increasing order.
    ordered.sort_by(|left, right| {
      compare_disjuncts(
        disjunct_key(left, targets),
        disjunct_key(right, targets),
      )
    });
  }
  fold_owned_binary(
    ordered
      .into_iter()
      .map(|child| formula_expr_for_targets(child, targets, integral))
      .collect::<Vec<_>>(),
    operator,
  )
}

/// The first requested variable a branch mentions and, when the branch bounds
/// it by a constant, that boundary.
fn disjunct_key(
  formula: &Formula,
  targets: &[Variable],
) -> Option<(usize, Option<Rational>)> {
  let index = targets
    .iter()
    .position(|target| formula.contains_variable(target))?;
  Some((index, constant_boundary(formula, &targets[index])))
}

fn constant_boundary(formula: &Formula, target: &Variable) -> Option<Rational> {
  match formula {
    Formula::Atom(Atom::Relation(relation, term)) => {
      let (_, boundary) = isolate_relation(*relation, term, target)?;
      boundary.is_constant().then_some(boundary.constant)
    }
    Formula::And(children) => children
      .iter()
      .find_map(|child| constant_boundary(child, target)),
    _ => None,
  }
}

fn compare_disjuncts(
  left: Option<(usize, Option<Rational>)>,
  right: Option<(usize, Option<Rational>)>,
) -> Ordering {
  match (left, right) {
    (None, None) => Ordering::Equal,
    (None, Some(_)) => Ordering::Less,
    (Some(_), None) => Ordering::Greater,
    (Some((left_index, left_bound)), Some((right_index, right_bound))) => {
      left_index.cmp(&right_index).then_with(|| {
        match (left_bound, right_bound) {
          (Some(left), Some(right)) => left.numeric_cmp(&right),
          (None, None) => Ordering::Equal,
          (None, Some(_)) => Ordering::Greater,
          (Some(_), None) => Ordering::Less,
        }
      })
    }
  }
}

fn fold_owned_binary(expressions: Vec<Expr>, operator: BinaryOperator) -> Expr {
  let mut expressions = expressions.into_iter();
  let Some(first) = expressions.next() else {
    return match operator {
      BinaryOperator::And => Expr::Identifier("True".to_string()),
      BinaryOperator::Or => Expr::Identifier("False".to_string()),
      BinaryOperator::Plus => Expr::Integer(0),
      _ => unreachable!("only associative n-ary operators are folded"),
    };
  };
  expressions.fold(first, |left, right| Expr::BinaryOp {
    op: operator,
    left: Box::new(left),
    right: Box::new(right),
  })
}

fn comparison_op(relation: Relation) -> ComparisonOp {
  match relation {
    Relation::Equal => ComparisonOp::Equal,
    Relation::NotEqual => ComparisonOp::NotEqual,
    Relation::Less => ComparisonOp::Less,
    Relation::LessEqual => ComparisonOp::LessEqual,
    Relation::Greater => ComparisonOp::Greater,
    Relation::GreaterEqual => ComparisonOp::GreaterEqual,
  }
}

#[cfg(test)]
mod tests {
  use std::collections::BTreeMap;

  use super::*;
  use crate::syntax::expr_to_string;

  #[test]
  fn affine_emission_is_stable_and_exact() {
    let term = AffineTerm {
      constant: Rational::new(BigInt::from(2), BigInt::from(3)).unwrap(),
      coefficients: BTreeMap::from([
        (
          Variable::free("y"),
          Rational::new(BigInt::from(-1), BigInt::from(2)).unwrap(),
        ),
        (Variable::free("x"), Rational::one()),
      ]),
    };
    assert_eq!(expr_to_string(&term_expr(&term)), "2/3 + x - y/2");
  }

  #[test]
  fn formula_emission_preserves_boolean_and_divisibility_structure() {
    let x = Variable::free("x");
    let less = Formula::Atom(Atom::Relation(
      Relation::Less,
      AffineTerm::variable(x.clone()),
    ));
    let odd = Formula::Atom(
      Atom::divides(
        BigInt::from(2),
        AffineTerm::variable(x)
          .subtract(&AffineTerm::constant(Rational::one())),
        false,
      )
      .unwrap(),
    );
    let formula = Formula::Or(vec![less, odd]).normalized();
    assert_eq!(
      expr_to_string(&formula_expr(&formula)),
      "x < 0 || Mod[x, 2] == 1"
    );
  }

  #[test]
  fn target_relation_is_isolated_and_negative_coefficients_reverse_order() {
    let x = Variable::free("x");
    let term = AffineTerm::variable(x.clone())
      .scaled(&Rational::integer(BigInt::from(-2)))
      .add(&AffineTerm::constant(Rational::integer(BigInt::from(3))));
    let formula = Formula::Atom(Atom::Relation(Relation::LessEqual, term));
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(&formula, &[x], false)),
      "x >= 3/2"
    );
  }

  #[test]
  fn two_target_bounds_emit_as_an_inequality_chain() {
    let x = Variable::free("x");
    let formula = Formula::And(vec![
      Formula::Atom(Atom::Relation(
        Relation::Greater,
        AffineTerm::variable(x.clone()),
      )),
      Formula::Atom(Atom::Relation(
        Relation::Less,
        AffineTerm::variable(x.clone())
          .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(2)))),
      )),
    ])
    .normalized();
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(&formula, &[x], false)),
      "Inequality[0, Less, x, Less, 2]"
    );
  }

  #[test]
  fn unit_target_congruence_emits_a_canonical_mod_residue() {
    let x = Variable::free("x");
    let formula = Formula::Atom(
      Atom::divides(
        BigInt::from(6),
        AffineTerm::variable(x.clone())
          .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(7)))),
        false,
      )
      .unwrap(),
    );
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(&formula, &[x], true)),
      "Mod[x, 6] == 1"
    );
  }

  #[test]
  fn finite_integer_interval_and_residue_emit_an_arithmetic_progression() {
    let x = Variable::free("x");
    let formula = Formula::And(vec![
      Formula::Atom(Atom::Relation(
        Relation::LessEqual,
        AffineTerm::variable(x.clone())
          .scaled(&Rational::integer(BigInt::from(-1))),
      )),
      Formula::Atom(Atom::Relation(
        Relation::LessEqual,
        AffineTerm::variable(x.clone())
          .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(10)))),
      )),
      Formula::Atom(
        Atom::divides(
          BigInt::from(3),
          AffineTerm::variable(x.clone()).subtract(&AffineTerm::constant(
            Rational::integer(BigInt::from(1)),
          )),
          false,
        )
        .unwrap(),
      ),
    ])
    .normalized();
    assert_eq!(
      expr_to_string(&finite_integer_target_expr(&formula, &x).unwrap()),
      "x == 1 || x == 4 || x == 7 || x == 10"
    );
  }

  #[test]
  fn integer_output_rounds_bounds_and_reduces_linear_congruences() {
    let x = Variable::free("x");
    let lower = Formula::Atom(Atom::Relation(
      Relation::LessEqual,
      AffineTerm::variable(x.clone())
        .scaled(&Rational::integer(BigInt::from(-3)))
        .add(&AffineTerm::constant(Rational::integer(BigInt::from(8)))),
    ));
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(
        &canonical_integer_formula(&lower, std::slice::from_ref(&x)),
        std::slice::from_ref(&x),
        true,
      )),
      "x >= 3"
    );

    let congruence = Formula::Atom(
      Atom::divides(
        BigInt::from(10),
        AffineTerm::variable(x.clone())
          .scaled(&Rational::integer(BigInt::from(6)))
          .add(&AffineTerm::constant(Rational::integer(BigInt::from(4)))),
        false,
      )
      .unwrap(),
    );
    assert_eq!(
      expr_to_string(&formula_expr_for_targets(
        &canonical_integer_formula(&congruence, std::slice::from_ref(&x)),
        std::slice::from_ref(&x),
        true,
      )),
      "Mod[x, 5] == 1"
    );
  }

  #[test]
  fn finite_integer_dnf_materializes_negated_residue_classes() {
    let x = Variable::free("x");
    let formula = Formula::And(vec![
      Formula::Atom(Atom::Relation(
        Relation::LessEqual,
        AffineTerm::variable(x.clone())
          .scaled(&Rational::integer(BigInt::from(-1))),
      )),
      Formula::Atom(Atom::Relation(
        Relation::LessEqual,
        AffineTerm::variable(x.clone())
          .subtract(&AffineTerm::constant(Rational::integer(BigInt::from(4)))),
      )),
      Formula::Atom(
        Atom::divides(BigInt::from(2), AffineTerm::variable(x.clone()), true)
          .unwrap(),
      ),
    ])
    .normalized();
    assert_eq!(
      expr_to_string(&finite_integer_target_expr(&formula, &x).unwrap()),
      "x == 1 || x == 3"
    );
  }
}
