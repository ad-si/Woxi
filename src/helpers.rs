use crate::syntax::{BinaryOperator, Expr};

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
