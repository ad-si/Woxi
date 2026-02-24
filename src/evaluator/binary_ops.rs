#[allow(unused_imports)]
use super::*;

/// Thread a binary operation over lists (e.g., {1,2,3} + 2 -> {3,4,5})
pub fn thread_binary_op(
  left: &Expr,
  right: &Expr,
  op: BinaryOperator,
) -> Result<Expr, InterpreterError> {
  // Helper to apply the binary operation to two evaluated expressions
  fn apply_op(
    l: &Expr,
    r: &Expr,
    op: BinaryOperator,
  ) -> Result<Expr, InterpreterError> {
    // Recursively thread over nested lists
    let has_nested_list =
      matches!(l, Expr::List(_)) || matches!(r, Expr::List(_));
    if has_nested_list {
      return thread_binary_op(l, r, op);
    }
    // Check if BigInt arithmetic is needed
    if (needs_bigint(l) || needs_bigint(r))
      && let (Some(lb), Some(rb)) = (expr_to_bigint(l), expr_to_bigint(r))
    {
      let result = match op {
        BinaryOperator::Plus => lb + rb,
        BinaryOperator::Minus => lb - rb,
        BinaryOperator::Times => lb * rb,
        _ => {
          // For Divide/Power, fall through to f64 path
          let ln = expr_to_number(l);
          let rn = expr_to_number(r);
          if let (Some(a), Some(b)) = (ln, rn) {
            return Ok(num_to_expr(a / b));
          }
          return Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          });
        }
      };
      return Ok(crate::functions::math_ast::bigint_to_expr(result));
    }
    let ln = expr_to_number(l);
    let rn = expr_to_number(r);
    let any_real = matches!(l, Expr::Real(_)) || matches!(r, Expr::Real(_));
    match op {
      BinaryOperator::Plus => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a + b))
          } else {
            Ok(num_to_expr(a + b))
          }
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      BinaryOperator::Minus => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a - b))
          } else {
            Ok(num_to_expr(a - b))
          }
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      BinaryOperator::Times => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a * b))
          } else {
            Ok(num_to_expr(a * b))
          }
        } else if matches!(l, Expr::Integer(0)) || matches!(r, Expr::Integer(0))
        {
          Ok(Expr::Integer(0))
        } else if matches!(l, Expr::Integer(1)) {
          Ok(r.clone())
        } else if matches!(r, Expr::Integer(1)) {
          Ok(l.clone())
        } else {
          // Apply Orderless canonical ordering for Times
          let ord = crate::functions::list_helpers_ast::compare_exprs(l, r);
          if ord < 0 {
            // Swap to put smaller expression first
            Ok(Expr::BinaryOp {
              op,
              left: Box::new(r.clone()),
              right: Box::new(l.clone()),
            })
          } else {
            Ok(Expr::BinaryOp {
              op,
              left: Box::new(l.clone()),
              right: Box::new(r.clone()),
            })
          }
        }
      }
      BinaryOperator::Divide => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if b == 0.0 {
            Err(InterpreterError::EvaluationError("Division by zero".into()))
          } else if any_real {
            Ok(Expr::Real(a / b))
          } else {
            Ok(num_to_expr(a / b))
          }
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      BinaryOperator::Power => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a.powf(b)))
          } else {
            Ok(num_to_expr(a.powf(b)))
          }
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      _ => Ok(Expr::BinaryOp {
        op,
        left: Box::new(l.clone()),
        right: Box::new(r.clone()),
      }),
    }
  }

  match (left, right) {
    (Expr::List(left_items), Expr::List(right_items)) => {
      // Both lists - element-wise operation
      if left_items.len() != right_items.len() {
        return Err(InterpreterError::EvaluationError(
          "Lists must have the same length for element-wise operations".into(),
        ));
      }
      let results: Result<Vec<Expr>, _> = left_items
        .iter()
        .zip(right_items.iter())
        .map(|(l, r)| apply_op(l, r, op))
        .collect();
      Ok(Expr::List(results?))
    }
    (Expr::List(items), scalar) => {
      // List op scalar - broadcast scalar
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_op(item, scalar, op))
        .collect();
      Ok(Expr::List(results?))
    }
    (scalar, Expr::List(items)) => {
      // Scalar op list - broadcast scalar
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_op(scalar, item, op))
        .collect();
      Ok(Expr::List(results?))
    }
    _ => apply_op(left, right, op),
  }
}

/// Extract raw string content from an Expr (without quotes for strings)
pub fn expr_to_raw_string(expr: &Expr) -> String {
  match expr {
    Expr::String(s) => s.clone(),
    _ => expr_to_string(expr),
  }
}

/// Parse a string to Expr (wrapper for syntax::string_to_expr)
pub fn string_to_expr(s: &str) -> Result<Expr, InterpreterError> {
  crate::syntax::string_to_expr(s)
}
