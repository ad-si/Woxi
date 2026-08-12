use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

pub fn neg1(e: Expr) -> Expr {
  Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand: Box::new(e),
  }
}

/// Helper to build a binary operation expression
pub fn binop(op: BinaryOperator, a: Expr, b: Expr) -> Expr {
  Expr::BinaryOp {
    op,
    left: Box::new(a),
    right: Box::new(b),
  }
}

pub fn plus2(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Plus, a, b)
}

pub fn minus2(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Minus, a, b)
}

pub fn pow2(b: Expr, e: Expr) -> Expr {
  binop(BinaryOperator::Power, b, e)
}

pub fn times2(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Times, a, b)
}

pub fn div2(a: Expr, b: Expr) -> Expr {
  binop(BinaryOperator::Divide, a, b)
}

/// Build the boolean symbol `True` or `False`.
pub fn bool_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}
