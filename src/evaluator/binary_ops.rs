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
    // BigFloat precision-tracked arithmetic
    if (matches!(l, Expr::BigFloat(_, _)) || matches!(r, Expr::BigFloat(_, _)))
      && let Some(result) = bigfloat_binary_op(l, r, op)
    {
      return Ok(result);
    }
    let ln = expr_to_number(l);
    let rn = expr_to_number(r);
    let any_real = matches!(l, Expr::Real(_) | Expr::BigFloat(_, _))
      || matches!(r, Expr::Real(_) | Expr::BigFloat(_, _));
    match op {
      BinaryOperator::Plus => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a + b))
          } else {
            Ok(num_to_expr(a + b))
          }
        } else {
          // Evaluate through the function-level Plus which handles Rationals
          crate::functions::math_ast::plus_ast(&[l.clone(), r.clone()])
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
          // Subtract: a - b = a + (-1)*b
          crate::functions::math_ast::plus_ast(&[
            l.clone(),
            crate::functions::math_ast::times_ast(&[
              Expr::Integer(-1),
              r.clone(),
            ])?,
          ])
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
          // Evaluate through the function-level Times which handles Rationals
          crate::functions::math_ast::times_ast(&[l.clone(), r.clone()])
        }
      }
      BinaryOperator::Divide => {
        // Delegate to divide_two which properly handles Rationals,
        // BigIntegers, Reals, Complex, etc.
        crate::functions::math_ast::divide_two(l, r)
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
        // Emit Thread::tdlen warning and return the expression unevaluated,
        // matching wolframscript behavior.
        let unevaluated = Expr::BinaryOp {
          op,
          left: Box::new(left.clone()),
          right: Box::new(right.clone()),
        };
        crate::emit_message(&format!(
          "Thread::tdlen: Objects of unequal length in {} cannot be combined.",
          crate::syntax::expr_to_string(&unevaluated)
        ));
        return Ok(unevaluated);
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

/// Extract (f64_value, precision_in_digits) from a numeric expression.
/// For BigFloat, precision comes from the stored value.
/// For Integer/BigInteger, precision is effectively infinite (use f64::INFINITY).
/// For Real, precision is machine precision (~16 digits).
fn bigfloat_value_prec(expr: &Expr) -> Option<(f64, f64)> {
  match expr {
    Expr::BigFloat(digits, prec) => {
      let v: f64 = digits.parse().unwrap_or(0.0);
      Some((v, *prec))
    }
    Expr::Real(f) => Some((*f, 16.0)),
    Expr::Integer(n) => Some((*n as f64, f64::INFINITY)),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_f64().map(|v| (v, f64::INFINITY))
    }
    _ => None,
  }
}

/// Compute result precision for addition/subtraction using error propagation.
fn precision_for_add(lv: f64, lp: f64, rv: f64, rp: f64, result: f64) -> f64 {
  let le = if lp.is_finite() {
    lv.abs() * 10f64.powf(-lp)
  } else {
    0.0
  };
  let re = if rp.is_finite() {
    rv.abs() * 10f64.powf(-rp)
  } else {
    0.0
  };
  let total_error = le + re;
  if result.abs() < 1e-300 || total_error <= 0.0 {
    if lp.is_finite() && rp.is_finite() {
      lp.min(rp).max(1.0)
    } else if lp.is_finite() {
      lp.max(1.0)
    } else {
      rp.max(1.0)
    }
  } else {
    let p = result.abs().log10() - total_error.log10();
    p.max(1.0)
  }
}

/// Format an f64 value as a BigFloat string with the given number of significant digits.
fn format_bigfloat_value(value: f64, sig_digits: usize) -> String {
  if value == 0.0 {
    return "0.".to_string();
  }
  let sign = if value < 0.0 { "-" } else { "" };
  let abs_val = value.abs();
  let magnitude = abs_val.log10().floor() as i32;
  let decimal_places = ((sig_digits as i32) - magnitude - 1).max(0) as usize;
  let formatted = format!("{}{:.prec$}", sign, abs_val, prec = decimal_places);
  // Ensure trailing dot if no decimal point
  if !formatted.contains('.') {
    format!("{}.", formatted)
  } else {
    formatted
  }
}

/// Perform a binary operation on BigFloat operands with precision tracking.
/// Returns None if operands are not numeric, so the caller can fall through.
fn bigfloat_binary_op(l: &Expr, r: &Expr, op: BinaryOperator) -> Option<Expr> {
  let (lv, lp) = bigfloat_value_prec(l)?;
  let (rv, rp) = bigfloat_value_prec(r)?;

  // If either operand is machine-precision Real (not BigFloat), produce Real
  if matches!(l, Expr::Real(_)) || matches!(r, Expr::Real(_)) {
    let result = match op {
      BinaryOperator::Plus => lv + rv,
      BinaryOperator::Minus => lv - rv,
      BinaryOperator::Times => lv * rv,
      BinaryOperator::Divide if rv != 0.0 => lv / rv,
      BinaryOperator::Power => lv.powf(rv),
      _ => return None,
    };
    return Some(Expr::Real(result));
  }

  let result_value = match op {
    BinaryOperator::Plus => lv + rv,
    BinaryOperator::Minus => lv - rv,
    BinaryOperator::Times => lv * rv,
    BinaryOperator::Divide if rv != 0.0 => lv / rv,
    BinaryOperator::Power => lv.powf(rv),
    _ => return None,
  };

  let result_prec = match op {
    BinaryOperator::Plus | BinaryOperator::Minus => {
      precision_for_add(lv, lp, rv, rp, result_value)
    }
    _ => {
      // For Times/Divide/Power, use min precision
      let p = if lp.is_finite() && rp.is_finite() {
        lp.min(rp)
      } else if lp.is_finite() {
        lp
      } else {
        rp
      };
      p.max(1.0)
    }
  };

  let display_prec = (result_prec.round() as usize).max(1);
  let result_str = format_bigfloat_value(result_value, display_prec);
  Some(Expr::BigFloat(result_str, result_prec))
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
