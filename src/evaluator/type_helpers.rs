#[allow(unused_imports)]
use super::*;

/// Convert an Expr to a number if possible
pub fn expr_to_number(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_f64()
    }
    Expr::Real(f) => Some(*f),
    Expr::BigFloat(digits, _) => digits.parse::<f64>().ok(),
    Expr::Constant(name) => constant_to_f64(name),
    _ => None,
  }
}

pub use crate::functions::math_ast::expr_to_i128;

pub use crate::functions::math_ast::needs_bigint_arithmetic as needs_bigint;
pub use crate::functions::math_ast::{
  constant_to_f64, expr_to_bigint, num_to_expr,
};

/// Check if an expression contains the imaginary unit I
pub fn contains_i(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) if s == "I" => true,
    Expr::BinaryOp { left, right, .. } => contains_i(left) || contains_i(right),
    Expr::FunctionCall { args, .. } => args.iter().any(contains_i),
    Expr::List(items) => items.iter().any(contains_i),
    Expr::UnaryOp { operand, .. } => contains_i(operand),
    _ => false,
  }
}

/// Apply a binary operation when at least one operand is a BigInteger or large Integer
pub fn bigint_binary_op<F>(left: &Expr, right: &Expr, op: F) -> Option<Expr>
where
  F: FnOnce(num_bigint::BigInt, num_bigint::BigInt) -> num_bigint::BigInt,
{
  if !needs_bigint(left) && !needs_bigint(right) {
    return None;
  }
  let l = expr_to_bigint(left)?;
  let r = expr_to_bigint(right)?;
  Some(crate::functions::math_ast::bigint_to_expr(op(l, r)))
}
