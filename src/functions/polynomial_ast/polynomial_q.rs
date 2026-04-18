#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::evaluator::pattern_matching::expr_equal;
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
    // First check for negative powers or non-polynomial structure
    if has_negative_powers(&args[0]) {
      return Ok(bool_expr(false));
    }
    let mut vars = std::collections::HashSet::new();
    collect_poly_vars(&args[0], &mut vars);
    if vars.is_empty() {
      // A constant is a polynomial
      return Ok(bool_expr(true));
    }
    return Ok(bool_expr(vars.iter().all(|v| is_polynomial(&args[0], v))));
  }

  match &args[1] {
    Expr::Identifier(name) => Ok(bool_expr(is_polynomial(&args[0], name))),
    Expr::List(vars) => {
      // PolynomialQ[expr, {v1, v2, ...}] — polynomial in all listed vars.
      // Each vi may be a symbol or a sub-expression treated as an atomic variable.
      for v in vars {
        if !is_polynomial_in_var(&args[0], v) {
          return Ok(bool_expr(false));
        }
      }
      Ok(bool_expr(true))
    }
    other => {
      // Non-symbol expression like f[a] — treat as atomic variable.
      Ok(bool_expr(is_polynomial_in_var(&args[0], other)))
    }
  }
}

/// Check if `expr` is a polynomial in `var_expr`, which may be a symbol or a
/// sub-expression (e.g. f[a]) treated as an atomic variable. Implemented by
/// substituting every occurrence of `var_expr` in `expr` with a fresh
/// identifier, then delegating to the symbol-based `is_polynomial`.
fn is_polynomial_in_var(expr: &Expr, var_expr: &Expr) -> bool {
  if let Expr::Identifier(name) = var_expr {
    return is_polynomial(expr, name);
  }
  const FRESH: &str = "$__PolyVar__";
  let fresh = Expr::Identifier(FRESH.to_string());
  let substituted = replace_subexpr(expr, var_expr, &fresh);
  if has_negative_powers(&substituted) {
    return false;
  }
  is_polynomial(&substituted, FRESH)
}

/// Structurally replace every occurrence of `target` in `expr` with `replacement`.
fn replace_subexpr(expr: &Expr, target: &Expr, replacement: &Expr) -> Expr {
  if expr_equal(expr, target) {
    return replacement.clone();
  }
  match expr {
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: op.clone(),
      left: Box::new(replace_subexpr(left, target, replacement)),
      right: Box::new(replace_subexpr(right, target, replacement)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: op.clone(),
      operand: Box::new(replace_subexpr(operand, target, replacement)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| replace_subexpr(a, target, replacement))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| replace_subexpr(a, target, replacement))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// Check if an expression contains any negative powers (indicating non-polynomial structure)
fn has_negative_powers(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::Constant(_)
    | Expr::Identifier(_) => false,
    Expr::BinaryOp { op, left, right } => match op {
      BinaryOperator::Power => {
        if let Expr::Integer(n) = right.as_ref()
          && *n < 0
        {
          return true;
        }
        has_negative_powers(left) || has_negative_powers(right)
      }
      BinaryOperator::Divide => {
        // a / b — check if b contains variables (non-constant denominator)
        if !matches!(right.as_ref(), Expr::Integer(_) | Expr::Real(_)) {
          return true;
        }
        has_negative_powers(left)
      }
      _ => has_negative_powers(left) || has_negative_powers(right),
    },
    Expr::UnaryOp { operand, .. } => has_negative_powers(operand),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Power" if args.len() == 2 => {
        if let Expr::Integer(n) = &args[1]
          && *n < 0
        {
          return true;
        }
        args.iter().any(has_negative_powers)
      }
      "Times" | "Plus" => args.iter().any(has_negative_powers),
      "Rational" => false,
      _ => false, // Function calls (Sin, Cos, x[1]) don't count
    },
    _ => false,
  }
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
