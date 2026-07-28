//! `CaputoD` — the Caputo fractional differintegral.
//!
//! For a power the answer is a single Gamma ratio,
//! `Gamma[p + 1]/Gamma[p - α + 1] t^(p - α)`, and the operator is linear, so a
//! polynomial is handled term by term. What sets Caputo apart from
//! Riemann–Liouville is the constant: it differentiates `⌈α⌉` times first, so
//! a constant vanishes for any positive order.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, Expr, UnaryOperator, unevaluated};

/// The same expression with its arithmetic written as calls, so one matcher
/// covers both spellings the parser produces.
fn as_calls(expr: &Expr) -> Expr {
  let call = |name: &str, args: Vec<Expr>| Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  };
  match expr {
    Expr::BinaryOp { op, left, right } => {
      let (l, r) = (as_calls(left), as_calls(right));
      match op {
        BinaryOperator::Plus => call("Plus", vec![l, r]),
        BinaryOperator::Minus => {
          call("Plus", vec![l, call("Times", vec![Expr::Integer(-1), r])])
        }
        BinaryOperator::Times => call("Times", vec![l, r]),
        BinaryOperator::Divide => {
          call("Times", vec![l, call("Power", vec![r, Expr::Integer(-1)])])
        }
        BinaryOperator::Power => call("Power", vec![l, r]),
        _ => expr.clone(),
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => call("Times", vec![Expr::Integer(-1), as_calls(operand)]),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(as_calls).collect::<Vec<_>>().into(),
    },
    other => other.clone(),
  }
}

/// `f[args…]`, evaluated.
fn eval_call(name: &str, args: Vec<Expr>) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  })
}

/// Whether `expr` mentions the variable anywhere.
fn mentions(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(name) => name == var,
    Expr::List(items) => items.iter().any(|e| mentions(e, var)),
    Expr::FunctionCall { args, .. } => args.iter().any(|e| mentions(e, var)),
    Expr::BinaryOp { left, right, .. } => {
      mentions(left, var) || mentions(right, var)
    }
    Expr::UnaryOp { operand, .. } => mentions(operand, var),
    _ => false,
  }
}

/// One term of the expanded input as a coefficient and the power of the
/// variable it carries. `None` for a term that is not a power of it.
fn term_parts(term: &Expr, var: &str) -> Option<(Expr, Expr)> {
  if !mentions(term, var) {
    return Some((term.clone(), Expr::Integer(0)));
  }
  match term {
    Expr::Identifier(name) if name == var => {
      Some((Expr::Integer(1), Expr::Integer(1)))
    }
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && matches!(&args[0], Expr::Identifier(b) if b == var)
        && !mentions(&args[1], var) =>
    {
      Some((Expr::Integer(1), args[1].clone()))
    }
    // A product: exactly one factor may carry the variable.
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut coefficients = Vec::new();
      let mut power = None;
      for factor in args.iter() {
        if mentions(factor, var) {
          if power.is_some() {
            return None;
          }
          power = Some(term_parts(factor, var)?.1);
        } else {
          coefficients.push(factor.clone());
        }
      }
      let coefficient = match coefficients.len() {
        0 => Expr::Integer(1),
        1 => coefficients.into_iter().next().unwrap(),
        _ => Expr::FunctionCall {
          name: "Times".to_string(),
          args: coefficients.into(),
        },
      };
      Some((coefficient, power?))
    }
    _ => None,
  }
}

/// The addends of an expanded expression.
fn addends(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().cloned().collect()
    }
    other => vec![other.clone()],
  }
}

/// The value of an expression when it is a plain number.
fn as_number(expr: &Expr) -> Option<f64> {
  crate::functions::math_ast::try_eval_to_f64(expr)
}

/// Whether `expr` is a non-negative whole number, and which one.
fn whole_number(expr: &Expr) -> Option<i64> {
  let v = as_number(expr)?;
  (v.fract() == 0.0 && v >= 0.0).then_some(v as i64)
}

/// `CaputoD[f, {x, α}]` — the Caputo fractional differintegral of `f`.
pub fn caputo_d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let uneval = || Ok(unevaluated("CaputoD", args));
  let sing = || {
    crate::emit_message(
      "CaputoD::sing: Caputo fractional derivative of given order is not \
       available for the input function.",
    );
    uneval()
  };
  if args.len() != 2 {
    return uneval();
  }
  let Expr::List(spec) = &args[1] else {
    return uneval();
  };
  if spec.len() != 2 {
    return uneval();
  }
  let Expr::Identifier(var) = &spec[0] else {
    return uneval();
  };
  let order = &spec[1];
  // Order zero asks for nothing at all.
  if matches!(order, Expr::Integer(0)) {
    return Ok(args[0].clone());
  }
  let order_value = as_number(order);
  // With a symbolic order the Gamma ratio only stands for a whole power of at
  // least two; below that the answer splits on the sign of the order, and for
  // a fractional power wolframscript reports it cannot do it.
  let symbolic_order = order_value.is_none();
  // `⌈α⌉` is how many times the function is differentiated before the
  // remaining order is integrated away.
  let derivatives = match order_value {
    Some(_) => {
      match whole_number(&eval_call("Ceiling", vec![order.clone()])?) {
        Some(n) => n,
        // A negative order integrates instead, and differentiates nothing.
        None => 0,
      }
    }
    None => 0,
  };

  let expanded = as_calls(&eval_call("Expand", vec![args[0].clone()])?);
  let mut terms = Vec::new();
  for term in addends(&expanded) {
    let Some((coefficient, power)) = term_parts(&term, var) else {
      return uneval();
    };
    let whole_power = whole_number(&power);
    if symbolic_order {
      match whole_power {
        Some(p) if p >= 2 => {}
        Some(_) => return uneval(),
        None => return sing(),
      }
    } else {
      match whole_power {
        // A constant vanishes under any positive order — the property the
        // Caputo derivative is chosen for.
        Some(0) if derivatives >= 1 => continue,
        // wolframscript works out no answer for a power the differentiation
        // step already flattens, even though it is plainly zero.
        Some(p) if p >= 1 && p < derivatives => return uneval(),
        Some(_) => {}
        // A power that is not a whole one is differentiated `⌈α⌉` times and
        // then integrated, and that integral only converges above
        // `t^(-1)` — so `p` has to clear `⌈α⌉ - 1`.
        None => {
          let converges = as_number(&power)
            .is_some_and(|p| derivatives < 1 || p > (derivatives - 1) as f64);
          if !converges {
            return sing();
          }
        }
      }
    }
    // Gamma[p + 1] / Gamma[p - α + 1] * var^(p - α).
    let plus = |a: Expr, b: Expr| Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![a, b].into(),
    };
    let shifted = eval_call(
      "Subtract",
      vec![plus(power.clone(), Expr::Integer(1)), order.clone()],
    )?;
    let ratio = eval_call(
      "Divide",
      vec![
        eval_call("Gamma", vec![plus(power.clone(), Expr::Integer(1))])?,
        eval_call("Gamma", vec![shifted])?,
      ],
    )?;
    let exponent = eval_call("Subtract", vec![power, order.clone()])?;
    terms.push(eval_call(
      "Times",
      vec![
        coefficient,
        ratio,
        eval_call("Power", vec![Expr::Identifier(var.to_string()), exponent])?,
      ],
    )?);
  }
  match terms.len() {
    0 => Ok(Expr::Integer(0)),
    1 => Ok(terms.into_iter().next().unwrap()),
    _ => eval_call("Plus", terms),
  }
}
