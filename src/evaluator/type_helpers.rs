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
    Expr::Constant(name) => constant_to_f64(name),
    _ => None,
  }
}

/// Extract an i128 from Integer or BigInteger (if it fits)
pub fn expr_to_i128(expr: &Expr) -> Option<i128> {
  use num_traits::ToPrimitive;
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => n.to_i128(),
    _ => None,
  }
}

/// Resolve a named constant to its numeric f64 value.
/// Constants (Pi, E, Degree) are kept symbolic — use try_eval_to_f64 for numeric evaluation.
pub fn constant_to_f64(_name: &str) -> Option<f64> {
  None
}

/// Convert a number to an appropriate Expr (Integer if whole, Real otherwise)
pub fn num_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
}

/// Convert an Expr to BigInt if it's an integer type
pub fn expr_to_bigint(expr: &Expr) -> Option<num_bigint::BigInt> {
  match expr {
    Expr::Integer(n) => Some(num_bigint::BigInt::from(*n)),
    Expr::BigInteger(n) => Some(n.clone()),
    _ => None,
  }
}

/// Check if an expression requires BigInt arithmetic (exceeds f64 precision).
/// f64 can only represent integers exactly up to 2^53.
pub fn needs_bigint(expr: &Expr) -> bool {
  match expr {
    Expr::BigInteger(_) => true,
    Expr::Integer(n) => n.unsigned_abs() > (1u128 << 53),
    _ => false,
  }
}

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
