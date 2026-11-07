//! Self-contained exact linear decision procedures for `Reduce` and `Resolve`.
//!
//! Parser-specific expressions are lowered transactionally into a binder-safe
//! affine formula IR. Dense linear arithmetic uses exact Fourier-Motzkin
//! elimination; integer arithmetic uses Cooper elimination. Unsupported
//! theories fall through to Woxi's established specialized reducers without
//! invoking an external process.

use super::*;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use woxi_reduce::{Atom, Formula, Rational, Relation};

use crate::helpers::call;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

#[allow(dead_code)]
mod emit;
mod integer_solve;
#[allow(dead_code)]
mod lower;
#[cfg(test)]
mod presburger_tests;
#[cfg(test)]
mod rational_qe_tests;

pub(crate) use integer_solve::{
  FiniteIntegerSolve, integer_expr, solve_finite_integer,
};

/// Runs the self-contained dense linear engine for every completely lowered
/// explicit real/rational request. Out-of-scope terms fall through without any
/// partial interpretation.
pub(super) fn try_linear_rational_reduce(args: &[Expr]) -> Option<Expr> {
  let request = lower::request_from_args(args)?;
  if !matches!(
    request.domain,
    ReduceDomain::Reals | ReduceDomain::Rationals
  ) {
    return None;
  }
  let result =
    woxi_reduce::rational_qe::eliminate_quantifiers(request.formula)?;
  let expression = emit::formula_expr_for_targets(&result, &request.targets);
  if request.domain != ReduceDomain::Rationals
    || matches!(result, Formula::False)
  {
    return Some(expression);
  }
  // Dense elimination describes the bounds over Q but does not encode domain
  // membership in the formula. Restore it unless every branch fixes the target
  // to an exact rational constant, which already implies that membership.
  fn fixes_rational_value(
    formula: &Formula,
    target: &woxi_reduce::Variable,
  ) -> bool {
    match formula {
      Formula::False => true,
      Formula::Atom(Atom::Relation(Relation::Equal, term)) => {
        term.coefficients.len() == 1 && term.coefficients.contains_key(target)
      }
      Formula::And(parts) => {
        parts.iter().any(|part| fixes_rational_value(part, target))
      }
      Formula::Or(parts) => {
        parts.iter().all(|part| fixes_rational_value(part, target))
      }
      _ => false,
    }
  }
  let members: Vec<_> = request
    .targets
    .iter()
    .filter(|target| !fixes_rational_value(&result, target))
    .map(|target| Expr::Identifier(target.name.clone()))
    .collect();
  let membership = |member| call("Element", vec![member, id_expr("Rationals")]);
  if matches!(result, Formula::True) {
    return Some(
      members
        .into_iter()
        .map(membership)
        .reduce(|a, b| binop(BinaryOperator::And, a, b))
        .unwrap_or(expression),
    );
  }
  let Some(members) = members
    .into_iter()
    .reduce(|a, b| binop(BinaryOperator::Alternatives, a, b))
  else {
    return Some(expression);
  };
  Some(binop(BinaryOperator::And, membership(members), expression))
}

/// Runs the self-contained Presburger engine for every completely lowered
/// explicit integer request.
pub(super) fn try_linear_integer_reduce(args: &[Expr]) -> Option<Expr> {
  let request = lower::request_from_args(args)?;
  if request.domain != ReduceDomain::Integers
    || is_underdetermined_linear_equation(&request)
  {
    return None;
  }
  let result = emit::canonical_integer_formula(
    &woxi_reduce::presburger::eliminate_quantifiers(request.formula)?,
    &request.targets,
  );
  if let [target] = request.targets.as_slice()
    && let Some(expression) = emit::finite_integer_target_expr(&result, target)
  {
    return Some(expression);
  }
  // A residue class is reported as a parametrization, never as a congruence.
  if let Some(expression) =
    emit::parametrized_integer_expr(&result, &request.targets)
  {
    return Some(expression);
  }
  let mut expression =
    emit::formula_expr_for_targets(&result, &request.targets);
  for target in request.targets.iter().rev() {
    if result.contains_variable(target) {
      expression = binop(
        BinaryOperator::And,
        call(
          "Element",
          vec![Expr::Identifier(target.name.clone()), id_expr("Integers")],
        ),
        expression,
      );
    }
  }
  Some(expression)
}

/// A linear equation (or a system of them, possibly with congruences) in
/// several integer unknowns, e.g.
/// `Reduce[2 x == 4 y, {x, y}, Integers]`. Its solution set is an unbounded
/// lattice that wolframscript reports parametrized with fresh integer
/// parameters (`C[1] ∈ Integers && x == 2 C[1] && y == C[1]`). Cooper
/// elimination cannot express that shape — it would only restate the equation
/// as a degenerate two-sided bound on one unknown — so such requests are
/// left to the extended-Euclidean Diophantine reducer in `reduce.rs`, which
/// is the single canonical implementation of that parametrization.
fn is_underdetermined_linear_equation(
  request: &lower::LinearReduceRequest,
) -> bool {
  fn is_equation(formula: &Formula) -> bool {
    matches!(formula, Formula::Atom(Atom::Relation(Relation::Equal, _)))
  }
  // A congruence `Mod[e, m] == r` joins a system of equations; on their own,
  // congruences are residue classes, reported as such below.
  fn is_congruence(formula: &Formula) -> bool {
    matches!(formula, Formula::Atom(Atom::Divides { negated: false, .. }))
  }
  request.targets.len() >= 2
    && match &request.formula {
      Formula::And(parts) => {
        parts.iter().any(is_equation)
          && parts.iter().all(|p| is_equation(p) || is_congruence(p))
      }
      formula => is_equation(formula),
    }
}

pub(crate) fn try_linear_rational_resolve(args: &[Expr]) -> Option<Expr> {
  if args.len() != 2
    || !matches!(
      &args[1],
      Expr::Identifier(domain) if domain == "Reals" || domain == "Rationals"
    )
  {
    return None;
  }
  let formula = lower::formula_from_expr(&args[0])?;
  let result = woxi_reduce::rational_qe::eliminate_quantifiers(formula)?;
  Some(emit::formula_expr(&result))
}

pub(crate) fn try_linear_integer_resolve(args: &[Expr]) -> Option<Expr> {
  if args.len() != 2
    || !matches!(&args[1], Expr::Identifier(domain) if domain == "Integers")
  {
    return None;
  }
  let formula = lower::formula_from_expr(&args[0])?;
  let result = woxi_reduce::presburger::eliminate_quantifiers(formula)?;
  Some(emit::formula_expr(&result))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReduceDomain {
  Default,
  Reals,
  Integers,
  Rationals,
  Complexes,
  Modulus,
  Unknown,
}

fn integer_value(expression: &Expr) -> Option<BigInt> {
  match expression {
    Expr::Integer(value) => Some(BigInt::from(*value)),
    Expr::BigInteger(value) => Some(value.clone()),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => integer_value(operand).map(std::ops::Neg::neg),
    _ => None,
  }
}

fn rational_value(expression: &Expr) -> Option<Rational> {
  if let Some(value) = integer_value(expression) {
    return Some(Rational::integer(value));
  }
  match expression {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      Rational::new(integer_value(&args[0])?, integer_value(&args[1])?)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => Rational::new(integer_value(left)?, integer_value(right)?),
    _ => None,
  }
}

fn request_domain(args: &[Expr]) -> ReduceDomain {
  if args.len() < 3 {
    return ReduceDomain::Default;
  }
  match &args[2] {
    Expr::Identifier(name) => match name.as_str() {
      "Reals" => ReduceDomain::Reals,
      "Integers" => ReduceDomain::Integers,
      "Rationals" => ReduceDomain::Rationals,
      "Complexes" => ReduceDomain::Complexes,
      _ => ReduceDomain::Unknown,
    },
    Expr::Rule { pattern, .. } if matches!(pattern.as_ref(), Expr::Identifier(name) if name == "Modulus") => {
      ReduceDomain::Modulus
    }
    _ => ReduceDomain::Unknown,
  }
}

fn bigint_expr(value: &BigInt) -> Expr {
  value
    .to_i128()
    .map_or_else(|| Expr::BigInteger(value.clone()), Expr::Integer)
}

fn rational_expr(value: &Rational) -> Expr {
  if value.denominator.is_one() {
    bigint_expr(&value.numerator)
  } else {
    call(
      "Rational",
      vec![
        bigint_expr(&value.numerator),
        bigint_expr(&value.denominator),
      ],
    )
  }
}
