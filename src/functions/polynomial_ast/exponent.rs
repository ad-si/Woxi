#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

use crate::functions::calculus_ast::is_constant_wrt;

// ─── Exponent ───────────────────────────────────────────────────────

/// Exponent[expr, var] - Returns the maximum power of var in expr
pub fn exponent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Exponent expects 2 or 3 arguments".into(),
    ));
  }
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of Exponent must be a symbol".into(),
      ));
    }
  };

  // Exponent[0, x] -> -Infinity
  if matches!(&args[0], Expr::Integer(0)) {
    return Ok(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("Infinity".to_string())),
    });
  }

  // Expand and combine like terms first to handle things like (x^2+1)^3-1
  let expanded = expand_and_combine(&args[0]);

  // Determine if we need Max (default) or Min
  let use_min =
    args.len() == 3 && matches!(&args[2], Expr::Identifier(s) if s == "Min");

  if use_min {
    match min_power(&expanded, var) {
      Some(n) => Ok(Expr::Integer(n)),
      None => Ok(Expr::FunctionCall {
        name: "Exponent".to_string(),
        args: args.to_vec(),
      }),
    }
  } else {
    match max_power(&expanded, var) {
      Some(n) => Ok(Expr::Integer(n)),
      None => Ok(Expr::FunctionCall {
        name: "Exponent".to_string(),
        args: args.to_vec(),
      }),
    }
  }
}

/// Find the maximum power of `var` in `expr`.  Returns None for non-polynomial forms.
pub fn max_power(expr: &Expr, var: &str) -> Option<i128> {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      Some(0)
    }
    Expr::Identifier(name) => {
      if name == var {
        Some(1)
      } else {
        Some(0)
      }
    }
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus => {
        let l = max_power(left, var)?;
        let r = max_power(right, var)?;
        Some(l.max(r))
      }
      BinaryOperator::Times => {
        let l = max_power(left, var)?;
        let r = max_power(right, var)?;
        Some(l + r)
      }
      BinaryOperator::Power => {
        if is_constant_wrt(left, var) {
          Some(0)
        } else if is_constant_wrt(right, var) {
          if let Expr::Integer(n) = right.as_ref() {
            let base_pow = max_power(left, var)?;
            Some(base_pow * n)
          } else {
            None
          }
        } else {
          None
        }
      }
      BinaryOperator::Divide => {
        if is_constant_wrt(right, var) {
          max_power(left, var)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(0)
        } else {
          None
        }
      }
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => max_power(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let mut m: i128 = 0;
        for a in args {
          m = m.max(max_power(a, var)?);
        }
        Some(m)
      }
      "Times" => {
        let mut s: i128 = 0;
        for a in args {
          s += max_power(a, var)?;
        }
        Some(s)
      }
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[0], var) {
          Some(0)
        } else if let Expr::Integer(n) = &args[1] {
          let base_pow = max_power(&args[0], var)?;
          Some(base_pow * n)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(0)
        } else {
          None
        }
      }
    },
    _ => {
      if is_constant_wrt(expr, var) {
        Some(0)
      } else {
        None
      }
    }
  }
}

/// Find the minimum power of `var` in `expr`.
pub fn min_power(expr: &Expr, var: &str) -> Option<i128> {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) | Expr::String(_) => {
      Some(0)
    }
    Expr::Identifier(name) => {
      if name == var {
        Some(1)
      } else {
        Some(0)
      }
    }
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus => {
        let l = min_power(left, var)?;
        let r = min_power(right, var)?;
        Some(l.min(r))
      }
      BinaryOperator::Times => {
        let l = min_power(left, var)?;
        let r = min_power(right, var)?;
        Some(l + r)
      }
      BinaryOperator::Power => {
        if is_constant_wrt(left, var) {
          Some(0)
        } else if is_constant_wrt(right, var) {
          if let Expr::Integer(n) = right.as_ref() {
            let base_pow = min_power(left, var)?;
            Some(base_pow * n)
          } else {
            None
          }
        } else {
          None
        }
      }
      BinaryOperator::Divide => {
        if is_constant_wrt(right, var) {
          min_power(left, var)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(0)
        } else {
          None
        }
      }
    },
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => min_power(operand, var),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Plus" => {
        let mut m: Option<i128> = None;
        for a in args {
          let p = min_power(a, var)?;
          m = Some(match m {
            None => p,
            Some(prev) => prev.min(p),
          });
        }
        m
      }
      "Times" => {
        let mut s: i128 = 0;
        for a in args {
          s += min_power(a, var)?;
        }
        Some(s)
      }
      "Power" if args.len() == 2 => {
        if is_constant_wrt(&args[0], var) {
          Some(0)
        } else if let Expr::Integer(n) = &args[1] {
          let base_pow = min_power(&args[0], var)?;
          Some(base_pow * n)
        } else {
          None
        }
      }
      _ => {
        if is_constant_wrt(expr, var) {
          Some(0)
        } else {
          None
        }
      }
    },
    _ => {
      if is_constant_wrt(expr, var) {
        Some(0)
      } else {
        None
      }
    }
  }
}
