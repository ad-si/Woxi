#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

use crate::functions::calculus_ast::is_constant_wrt;

// ─── PolynomialQ ────────────────────────────────────────────────────

/// PolynomialQ[expr, var] - Tests if expr is a polynomial in var
pub fn polynomial_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PolynomialQ expects 1 or 2 arguments".into(),
    ));
  }

  if args.len() == 1 {
    // 1-arg form: check if expr is a polynomial in all its variables
    let mut vars = std::collections::HashSet::new();
    collect_poly_vars(&args[0], &mut vars);
    if vars.is_empty() {
      // A constant is a polynomial
      return Ok(bool_expr(true));
    }
    return Ok(bool_expr(vars.iter().all(|v| is_polynomial(&args[0], v))));
  }

  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of PolynomialQ must be a symbol".into(),
      ));
    }
  };
  Ok(bool_expr(is_polynomial(&args[0], var)))
}

/// Collect variables that appear in polynomial context only (not inside functions like Sin)
pub fn collect_poly_vars(
  expr: &Expr,
  vars: &mut std::collections::HashSet<String>,
) {
  match expr {
    Expr::Identifier(name)
      if name != "True"
        && name != "False"
        && name != "Null"
        && name != "I"
        && name != "Pi"
        && name != "E"
        && name != "Infinity" =>
    {
      vars.insert(name.clone());
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_poly_vars(left, vars);
      collect_poly_vars(right, vars);
    }
    Expr::UnaryOp { operand, .. } => collect_poly_vars(operand, vars),
    Expr::FunctionCall { name, args } => {
      if name == "Plus" || name == "Times" || name == "Power" {
        for a in args {
          collect_poly_vars(a, vars);
        }
      }
      // For other functions like Sin[x], don't collect x as a polynomial variable
    }
    _ => {}
  }
}

/// Recursively check whether an expression is a polynomial in `var`.
pub fn is_polynomial(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => true,
    Expr::Identifier(_) => true, // either it IS the variable or a constant symbol – both ok
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Times => {
        is_polynomial(left, var) && is_polynomial(right, var)
      }
      BinaryOperator::Power => {
        // base must contain the variable and exponent must be a non-negative integer
        if is_constant_wrt(right, var) {
          if let Expr::Integer(n) = right.as_ref() {
            *n >= 0 && is_polynomial(left, var)
          } else {
            // non-integer exponent like x^y where y is a symbol ≠ var
            // Only polynomial if base is constant w.r.t. var
            is_constant_wrt(left, var)
          }
        } else {
          false
        }
      }
      BinaryOperator::Divide => {
        // polynomial / constant-in-var is still polynomial
        is_polynomial(left, var) && is_constant_wrt(right, var)
      }
      _ => is_constant_wrt(expr, var),
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_polynomial(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" | "Times" => args.iter().all(|a| is_polynomial(a, var)),
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[1], var) {
          if let Expr::Integer(n) = &args[1] {
            *n >= 0 && is_polynomial(&args[0], var)
          } else {
            is_constant_wrt(&args[0], var)
          }
        } else {
          false
        }
      }
      _ => is_constant_wrt(expr, var),
    },
    Expr::List(_) => false,
    _ => is_constant_wrt(expr, var),
  }
}
