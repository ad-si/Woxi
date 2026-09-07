//! Self-contained exact linear decision procedures for `Reduce` and `Resolve`.
//!
//! Parser-specific expressions are lowered transactionally into a binder-safe
//! affine formula IR. Dense linear arithmetic uses exact Fourier-Motzkin
//! elimination; integer arithmetic uses Cooper elimination. Unsupported
//! theories fall through to Woxi's established specialized reducers without
//! invoking an external process.

use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use woxi_reduce::{Atom, Formula, Rational, Relation};

use crate::helpers::call;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

mod emit;
mod integer_solve;
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
  let result = woxi_reduce::equalities::fold_equalities(
    woxi_reduce::rational_qe::eliminate_quantifiers(request.formula)?,
    &request.targets,
    false,
  );
  Some(emit::formula_expr_for_targets(
    &result,
    &request.targets,
    false,
  ))
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
    &woxi_reduce::equalities::fold_equalities(
      woxi_reduce::presburger::eliminate_quantifiers(request.formula)?,
      &request.targets,
      true,
    ),
    &request.targets,
  );
  if let [target] = request.targets.as_slice()
    && let Some(expression) = emit::finite_integer_target_expr(&result, target)
  {
    return Some(expression);
  }
  let expression =
    emit::formula_expr_for_targets(&result, &request.targets, true);
  if matches!(result, Formula::True | Formula::False) {
    return Some(expression);
  }
  // Every requested variable that the result does not fix to an integer
  // keeps its domain: `Element[x | y, Integers] && ...`, the form
  // wolframscript uses for several variables.
  let membership = request
    .targets
    .iter()
    .filter(|target| !emit::target_is_pinned(&result, target))
    .map(|target| Expr::Identifier(target.name.clone()))
    .reduce(|left, right| Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left: Box::new(left),
      right: Box::new(right),
    });
  Some(match membership {
    Some(members) => Expr::BinaryOp {
      op: BinaryOperator::And,
      left: Box::new(call(
        "Element",
        vec![members, Expr::Identifier("Integers".to_string())],
      )),
      right: Box::new(expression),
    },
    None => expression,
  })
}

/// A lone linear equation in several integer unknowns, e.g.
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
  request.targets.len() >= 2
    && matches!(
      request.formula,
      Formula::Atom(Atom::Relation(Relation::Equal, _))
    )
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
