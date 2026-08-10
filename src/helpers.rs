use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

pub fn neg1(e: Expr) -> Expr {
  Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand: Box::new(e),
  }
}

pub fn pow2(b: Expr, e: Expr) -> Expr {
  Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(b),
    right: Box::new(e),
  }
}

pub fn div2(a: Expr, b: Expr) -> Expr {
  Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(a),
    right: Box::new(b),
  }
}
