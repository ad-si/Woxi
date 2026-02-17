//! AST-native calculus functions (D, Integrate).
//!
//! These functions work directly with `Expr` AST nodes for symbolic differentiation
//! and integration.

use crate::InterpreterError;
use crate::syntax::Expr;

/// D[expr, var] or D[expr, {var, n}] - Symbolic differentiation
pub fn d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "D expects exactly 2 arguments".into(),
    ));
  }

  // Thread over lists in the first argument: D[{f1, f2, ...}, var] -> {D[f1, var], D[f2, var], ...}
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| d_ast(&[item.clone(), args[1].clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  // Handle D[expr, {var, n}] for higher-order derivatives
  if let Expr::List(items) = &args[1] {
    if items.len() == 2 {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Variable specification in D must be a symbol".into(),
          ));
        }
      };
      let n = match &items[1] {
        Expr::Integer(n) if *n >= 0 => *n as usize,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Derivative order in D must be a non-negative integer".into(),
          ));
        }
      };
      // Apply differentiation n times
      let mut result = args[0].clone();
      for _ in 0..n {
        result = differentiate(&result, &var_name)?;
        result = simplify(result);
      }
      return Ok(result);
    }
  }

  // Get the variable name
  let var_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of D must be a symbol".into(),
      ));
    }
  };

  // Differentiate the expression
  differentiate(&args[0], &var_name)
}

/// Integrate[expr, var] or Integrate[expr, {var, lo, hi}] - Symbolic integration
pub fn integrate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Integrate expects exactly 2 arguments".into(),
    ));
  }

  // Check if the second argument is {var, lo, hi} (definite integral)
  if let Expr::List(items) = &args[1]
    && items.len() == 3
  {
    let var_name = match &items[0] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Integrate".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let lo = &items[1];
    let hi = &items[2];

    // Try known definite integrals first
    if let Some(result) = try_definite_integral(&args[0], &var_name, lo, hi) {
      return Ok(result);
    }

    // Fall back: compute indefinite integral and evaluate at bounds
    if let Some(antideriv) = integrate(&args[0], &var_name) {
      let antideriv = simplify(antideriv);
      // F(hi) - F(lo)
      let at_hi = crate::syntax::substitute_variable(&antideriv, &var_name, hi);
      let at_lo = crate::syntax::substitute_variable(&antideriv, &var_name, lo);
      let at_hi = crate::evaluator::evaluate_expr_to_expr(&at_hi)?;
      let at_lo = crate::evaluator::evaluate_expr_to_expr(&at_lo)?;
      return Ok(simplify(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(at_hi),
        right: Box::new(at_lo),
      }));
    }

    // Return unevaluated
    return Ok(Expr::FunctionCall {
      name: "Integrate".to_string(),
      args: args.to_vec(),
    });
  }

  // Indefinite integral: Integrate[expr, var]
  let var_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Integrate".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Integrate the expression
  match integrate(&args[0], &var_name) {
    Some(result) => Ok(simplify(result)),
    None => Ok(Expr::FunctionCall {
      name: "Integrate".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Check if an expression represents Infinity
fn is_infinity(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(name) if name == "Infinity")
}

/// Check if an expression represents -Infinity (via UnaryOp::Minus or Times[-1, Infinity])
fn is_negative_infinity(expr: &Expr) -> bool {
  match expr {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => is_infinity(operand),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      (matches!(left.as_ref(), Expr::Integer(-1)) && is_infinity(right))
        || (matches!(right.as_ref(), Expr::Integer(-1)) && is_infinity(left))
    }
    Expr::Integer(n) if *n < 0 => false, // negative number, not -Infinity
    _ => false,
  }
}

/// Try to evaluate a definite integral using known closed-form results
fn try_definite_integral(
  integrand: &Expr,
  var: &str,
  lo: &Expr,
  hi: &Expr,
) -> Option<Expr> {
  // Gaussian integral: ∫_{-∞}^{∞} E^(-a*x^2) dx = Sqrt[Pi/a]
  if is_negative_infinity(lo)
    && is_infinity(hi)
    && let Some(coeff) = match_gaussian(integrand, var)
  {
    // coeff is 'a' in E^(-a*x^2): result = Sqrt[Pi/a]
    return Some(match coeff {
      Expr::Integer(1) => Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Constant("Pi".to_string())],
      },
      _ => Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::Constant("Pi".to_string())),
          right: Box::new(coeff),
        }],
      },
    });
  }

  // Half-Gaussian: ∫_0^{∞} E^(-a*x^2) dx = Sqrt[Pi/a]/2
  if matches!(lo, Expr::Integer(0))
    && is_infinity(hi)
    && let Some(coeff) = match_gaussian(integrand, var)
  {
    let sqrt_part = match coeff {
      Expr::Integer(1) => Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Constant("Pi".to_string())],
      },
      _ => Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::Constant("Pi".to_string())),
          right: Box::new(coeff),
        }],
      },
    };
    return Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(sqrt_part),
      right: Box::new(Expr::Integer(2)),
    });
  }

  None
}

/// Try to match an expression as E^(-a*x^2) where a is a positive constant.
/// Returns Some(a) if it matches, None otherwise.
fn match_gaussian(expr: &Expr, var: &str) -> Option<Expr> {
  // Match E^(exponent) where E is the constant
  let exponent = match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
        Some(right.as_ref())
      } else {
        None
      }
    }
    _ => None,
  }?;

  // Match -a*x^2 or -(x^2) forms in the exponent
  match_neg_a_x_squared(exponent, var)
}

/// Match an exponent expression as -a*x^2 and return 'a'.
/// Handles forms: -x^2, -(x^2), -a*x^2, Times[-1, x, x], etc.
fn match_neg_a_x_squared(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    // UnaryOp::Minus wrapping something
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      // -x^2 => a=1
      if is_var_squared(operand, var) {
        return Some(Expr::Integer(1));
      }
      // -(a*x^2) => a
      if let Some(coeff) = match_a_x_squared(operand, var) {
        return Some(coeff);
      }
      None
    }
    // Times[-1, x^2] or Times[x^2, -1]
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      // -1 * x^2
      if matches!(left.as_ref(), Expr::Integer(-1))
        && is_var_squared(right, var)
      {
        return Some(Expr::Integer(1));
      }
      if matches!(right.as_ref(), Expr::Integer(-1))
        && is_var_squared(left, var)
      {
        return Some(Expr::Integer(1));
      }
      // -1 * (a * x^2)
      if matches!(left.as_ref(), Expr::Integer(-1))
        && let Some(coeff) = match_a_x_squared(right, var)
      {
        return Some(coeff);
      }
      if matches!(right.as_ref(), Expr::Integer(-1))
        && let Some(coeff) = match_a_x_squared(left, var)
      {
        return Some(coeff);
      }
      // (-a) * x^2 where a is a negative integer
      if let Expr::Integer(n) = left.as_ref()
        && *n < 0
        && is_var_squared(right, var)
      {
        return Some(Expr::Integer(-*n));
      }
      if let Expr::Integer(n) = right.as_ref()
        && *n < 0
        && is_var_squared(left, var)
      {
        return Some(Expr::Integer(-*n));
      }
      None
    }
    // FunctionCall("Times", [...]) form
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Times[-1, x^2] => a=1
      // Times[-a, x^2] => a (where a is positive)
      // Times[-1, a, x^2] => a
      // Find a negative integer factor and check remaining is a*x^2
      for (i, arg) in args.iter().enumerate() {
        if let Expr::Integer(n) = arg
          && *n < 0
        {
          let mut rest: Vec<Expr> = args
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, e)| e.clone())
            .collect();
          let rest_expr = if rest.len() == 1 {
            rest.remove(0)
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            }
          };
          if *n == -1 {
            // Times[-1, x^2] => a=1
            if is_var_squared(&rest_expr, var) {
              return Some(Expr::Integer(1));
            }
            // Times[-1, a*x^2] => a
            if let Some(coeff) = match_a_x_squared(&rest_expr, var) {
              return Some(coeff);
            }
          } else {
            // Times[-a, x^2] => a
            if is_var_squared(&rest_expr, var) {
              return Some(Expr::Integer(-*n));
            }
          }
        }
      }
      None
    }
    // BinaryOp::Minus: 0 - x^2 or similar (unlikely but handle)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(0)) {
        if is_var_squared(right, var) {
          return Some(Expr::Integer(1));
        }
        if let Some(coeff) = match_a_x_squared(right, var) {
          return Some(coeff);
        }
      }
      None
    }
    _ => None,
  }
}

/// Check if expr is x^2 (where x is the variable)
fn is_var_squared(expr: &Expr, var: &str) -> bool {
  matches!(
    expr,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      && matches!(right.as_ref(), Expr::Integer(2))
  )
}

/// Match a*x^2 and return 'a'
fn match_a_x_squared(expr: &Expr, var: &str) -> Option<Expr> {
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    // a * x^2
    if is_constant_wrt(left, var) && is_var_squared(right, var) {
      return Some(*left.clone());
    }
    // x^2 * a
    if is_var_squared(left, var) && is_constant_wrt(right, var) {
      return Some(*right.clone());
    }
  }
  // FunctionCall("Times", [...]) form
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
  {
    // Find x^2 factor, rest is 'a'
    for (i, arg) in args.iter().enumerate() {
      if is_var_squared(arg, var) {
        let rest: Vec<Expr> = args
          .iter()
          .enumerate()
          .filter(|(j, _)| *j != i)
          .map(|(_, e)| e.clone())
          .collect();
        if rest.iter().all(|a| is_constant_wrt(a, var)) {
          return if rest.len() == 1 {
            Some(rest[0].clone())
          } else {
            Some(Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            })
          };
        }
      }
    }
  }
  None
}

/// Check if expression is constant with respect to a variable
pub fn is_constant_wrt(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::String(_) | Expr::Constant(_) => {
      true
    }
    Expr::Identifier(name) => name != var,
    Expr::List(items) => items.iter().all(|e| is_constant_wrt(e, var)),
    Expr::BinaryOp { left, right, .. } => {
      is_constant_wrt(left, var) && is_constant_wrt(right, var)
    }
    Expr::UnaryOp { operand, .. } => is_constant_wrt(operand, var),
    Expr::FunctionCall { args, .. } => {
      args.iter().all(|e| is_constant_wrt(e, var))
    }
    _ => false,
  }
}

/// Differentiate an expression with respect to a variable
/// Public wrapper for differentiate - used by Derivative[n][f][x] evaluation
pub fn differentiate_expr(
  expr: &Expr,
  var: &str,
) -> Result<Expr, InterpreterError> {
  differentiate(expr, var)
}

fn differentiate(expr: &Expr, var: &str) -> Result<Expr, InterpreterError> {
  match expr {
    // Constants
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => {
      Ok(Expr::Integer(0))
    }

    // Variable
    Expr::Identifier(name) => {
      if name == var {
        Ok(Expr::Integer(1))
      } else {
        Ok(Expr::Integer(0))
      }
    }

    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator::*;
      match op {
        Plus => {
          // d/dx[a + b] = d/dx[a] + d/dx[b]
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Plus,
            left: Box::new(da),
            right: Box::new(db),
          }))
        }
        Minus => {
          // d/dx[a - b] = d/dx[a] - d/dx[b]
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Minus,
            left: Box::new(da),
            right: Box::new(db),
          }))
        }
        Times => {
          // Product rule: d/dx[a * b] = a' * b + a * b'
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Plus,
            left: Box::new(Expr::BinaryOp {
              op: Times,
              left: Box::new(da),
              right: right.clone(),
            }),
            right: Box::new(Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(db),
            }),
          }))
        }
        Divide => {
          // Quotient rule: d/dx[a / b] = (a' * b - a * b') / b^2
          let da = differentiate(left, var)?;
          let db = differentiate(right, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: Divide,
            left: Box::new(Expr::BinaryOp {
              op: Minus,
              left: Box::new(Expr::BinaryOp {
                op: Times,
                left: Box::new(da),
                right: right.clone(),
              }),
              right: Box::new(Expr::BinaryOp {
                op: Times,
                left: left.clone(),
                right: Box::new(db),
              }),
            }),
            right: Box::new(Expr::BinaryOp {
              op: Power,
              left: right.clone(),
              right: Box::new(Expr::Integer(2)),
            }),
          }))
        }
        Power => {
          // Power rule for x^n: n * x^(n-1) * x'
          // Use Plus[-1, n] to match Wolfram's canonical form (-1 + n)
          if is_constant_wrt(right, var) {
            let df = differentiate(left, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::BinaryOp {
                op: Times,
                left: right.clone(),
                right: Box::new(Expr::BinaryOp {
                  op: Power,
                  left: left.clone(),
                  right: Box::new(Expr::BinaryOp {
                    op: Plus,
                    left: Box::new(Expr::Integer(-1)),
                    right: right.clone(),
                  }),
                }),
              }),
              right: Box::new(df),
            }))
          } else if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
            // d/dx[E^g(x)] = E^g(x) * g'(x)  (since Log[E] = 1)
            let dg = differentiate(right, var)?;
            Ok(simplify(Expr::BinaryOp {
              op: Times,
              left: Box::new(expr.clone()),
              right: Box::new(dg),
            }))
          } else if is_constant_wrt(left, var) {
            // d/dx[a^g(x)] = a^g(x) * ln(a) * g'(x)
            let dg = differentiate(right, var)?;
            Ok(Expr::BinaryOp {
              op: Times,
              left: Box::new(Expr::BinaryOp {
                op: Times,
                left: Box::new(expr.clone()),
                right: Box::new(Expr::FunctionCall {
                  name: "Log".to_string(),
                  args: vec![*left.clone()],
                }),
              }),
              right: Box::new(dg),
            })
          } else {
            // General case - return unevaluated
            Ok(Expr::FunctionCall {
              name: "D".to_string(),
              args: vec![expr.clone(), Expr::Identifier(var.to_string())],
            })
          }
        }
        _ => Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        }),
      }
    }

    // Unary minus
    Expr::UnaryOp { op, operand } => {
      use crate::syntax::UnaryOperator;
      if matches!(op, UnaryOperator::Minus) {
        let d = differentiate(operand, var)?;
        Ok(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(d),
        })
      } else {
        Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        })
      }
    }

    // Function calls
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Sin" if args.len() == 1 => {
          // d/dx[sin(f(x))] = cos(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Cos" if args.len() == 1 => {
          // d/dx[cos(f(x))] = -sin(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::FunctionCall {
                name: "Sin".to_string(),
                args: args.clone(),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Tan" if args.len() == 1 => {
          // d/dx[tan(f(x))] = sec^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(Expr::FunctionCall {
                name: "Sec".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(df),
          }))
        }
        "Sec" if args.len() == 1 => {
          // d/dx[sec(f(x))] = sec(f(x)) * tan(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(Expr::FunctionCall {
                name: "Sec".to_string(),
                args: args.clone(),
              }),
              right: Box::new(Expr::FunctionCall {
                name: "Tan".to_string(),
                args: args.clone(),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Csc" if args.len() == 1 => {
          // d/dx[csc(f(x))] = -csc(f(x)) * cot(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::FunctionCall {
                  name: "Csc".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::FunctionCall {
                  name: "Cot".to_string(),
                  args: args.clone(),
                }),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Cot" if args.len() == 1 => {
          // d/dx[cot(f(x))] = -csc^2(f(x)) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Csc".to_string(),
                  args: args.clone(),
                }),
                right: Box::new(Expr::Integer(2)),
              }),
            }),
            right: Box::new(df),
          }))
        }
        "Exp" if args.len() == 1 => {
          // d/dx[e^f(x)] = e^f(x) * f'(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            }),
            right: Box::new(df),
          }))
        }
        "Log" if args.len() == 1 => {
          // d/dx[ln(f(x))] = f'(x) / f(x)
          let df = differentiate(&args[0], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(df),
            right: Box::new(args[0].clone()),
          }))
        }
        // Handle evaluated Plus[a, b, ...] (FunctionCall form of +)
        "Plus" if args.len() >= 2 => {
          let mut result = differentiate(&args[0], var)?;
          for arg in &args[1..] {
            let d = differentiate(arg, var)?;
            result = simplify(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(result),
              right: Box::new(d),
            });
          }
          Ok(result)
        }
        // Handle evaluated Times[a, b, ...] (FunctionCall form of *)
        "Times" if args.len() >= 2 => {
          // For n factors, treat as a * rest and apply product rule recursively
          let a = &args[0];
          let rest = if args.len() == 2 {
            args[1].clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: args[1..].to_vec(),
            }
          };
          let da = differentiate(a, var)?;
          let drest = differentiate(&rest, var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(da),
              right: Box::new(rest.clone()),
            }),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(a.clone()),
              right: Box::new(drest),
            }),
          }))
        }
        // Handle evaluated Power[base, exp] (FunctionCall form of ^)
        "Power" if args.len() == 2 => differentiate(
          &Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(args[0].clone()),
            right: Box::new(args[1].clone()),
          },
          var,
        ),
        // Handle Rational[n, d] as constant
        "Rational" if args.len() == 2 => Ok(Expr::Integer(0)),
        _ => Ok(Expr::FunctionCall {
          name: "D".to_string(),
          args: vec![expr.clone(), Expr::Identifier(var.to_string())],
        }),
      }
    }

    _ => Ok(Expr::FunctionCall {
      name: "D".to_string(),
      args: vec![expr.clone(), Expr::Identifier(var.to_string())],
    }),
  }
}

/// Build `expr / divisor`, simplifying to just `expr` when `divisor == 1`.
/// For integer divisors, produces `Rational[1, n] * expr` to match Wolfram output.
fn make_divided(expr: Expr, divisor: Expr) -> Expr {
  match &divisor {
    Expr::Integer(1) => expr,
    _ => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(expr),
      right: Box::new(divisor),
    },
  }
}

/// Build `-expr / divisor`, simplifying to `-expr` when `divisor == 1`.
/// For integer divisors, produces `Rational[-1, n] * expr` to match Wolfram output.
fn make_neg_divided(expr: Expr, divisor: Expr) -> Expr {
  match &divisor {
    Expr::Integer(1) => Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(expr),
    },
    Expr::Integer(n) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(crate::functions::math_ast::make_rational_pub(-1, *n)),
      right: Box::new(expr),
    },
    _ => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(expr),
      }),
      right: Box::new(divisor),
    },
  }
}

/// Build the antiderivative of Exp[-a*x^2]:
///   Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]  (general a)
///   (Sqrt[Pi]*Erf[x])/2            (when a=1)
fn make_gaussian_antiderivative(var: &str, coeff: &Expr) -> Expr {
  let var_expr = Expr::Identifier(var.to_string());
  let (erf_arg, prefix) = match coeff {
    Expr::Integer(1) => {
      // a=1: Erf[x], prefix = Sqrt[Pi]
      (
        var_expr,
        Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![Expr::Constant("Pi".to_string())],
        },
      )
    }
    _ => {
      // general a: Erf[Sqrt[a]*x], prefix = Sqrt[Pi/a]
      let sqrt_a = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![coeff.clone()],
      };
      let erf_arg = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(sqrt_a),
        right: Box::new(var_expr),
      };
      let prefix = Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::Constant("Pi".to_string())),
          right: Box::new(coeff.clone()),
        }],
      };
      (erf_arg, prefix)
    }
  };
  let erf_expr = Expr::FunctionCall {
    name: "Erf".to_string(),
    args: vec![erf_arg],
  };
  // (prefix * Erf[...]) / 2
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(prefix),
      right: Box::new(erf_expr),
    }),
    right: Box::new(Expr::Integer(2)),
  }
}

/// Try to match an expression as `a*var` where `a` is constant w.r.t. `var`,
/// or just `var` (returning `Integer(1)`).
/// Returns Some(a) if it matches, None otherwise.
fn try_match_linear_arg(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    Expr::Identifier(name) if name == var => Some(Expr::Integer(1)),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if is_constant_wrt(left, var)
        && matches!(right.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*left.clone())
      } else if is_constant_wrt(right, var)
        && matches!(left.as_ref(), Expr::Identifier(name) if name == var)
      {
        Some(*right.clone())
      } else {
        None
      }
    }
    // FunctionCall("Times", [coeff, var]) form
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Find the variable factor and collect the rest as coefficient
      let mut var_idx = None;
      for (i, arg) in args.iter().enumerate() {
        if matches!(arg, Expr::Identifier(n) if n == var) {
          var_idx = Some(i);
          break;
        }
      }
      if let Some(vi) = var_idx {
        let rest: Vec<Expr> = args
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != vi)
          .map(|(_, e)| e.clone())
          .collect();
        if rest.iter().all(|a| is_constant_wrt(a, var)) {
          if rest.len() == 1 {
            Some(rest[0].clone())
          } else {
            Some(Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            })
          }
        } else {
          None
        }
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Try to integrate Sin[f(x)]^2 or Cos[f(x)]^2 using power-reduction identities.
/// sin²(a*x) = x/2 - sin(2*a*x)/(4*a)
/// cos²(a*x) = x/2 + sin(2*a*x)/(4*a)
fn try_integrate_trig_squared(base: &Expr, var: &str) -> Option<Expr> {
  if let Expr::FunctionCall { name, args } = base
    && args.len() == 1
  {
    let is_sin = name == "Sin";
    let is_cos = name == "Cos";
    if !is_sin && !is_cos {
      return None;
    }
    let coeff = try_match_linear_arg(&args[0], var)?;
    // Build: 2*a*x
    let double_arg = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(args[0].clone()),
    });
    // sin(2*a*x)
    let sin_double = Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![double_arg],
    };
    // 4*a
    let four_a = simplify(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(4)),
      right: Box::new(coeff),
    });
    // x/2
    let x_half = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Identifier(var.to_string())),
      right: Box::new(Expr::Integer(2)),
    };
    // sin(2*a*x)/(4*a)
    let sin_term = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(sin_double),
      right: Box::new(four_a),
    };
    if is_sin {
      // x/2 - sin(2*a*x)/(4*a)
      Some(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(x_half),
        right: Box::new(sin_term),
      })
    } else {
      // x/2 + sin(2*a*x)/(4*a)
      Some(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(x_half),
        right: Box::new(sin_term),
      })
    }
  } else {
    None
  }
}

/// Integrate an expression with respect to a variable
fn integrate(expr: &Expr, var: &str) -> Option<Expr> {
  match expr {
    // Constant: ∫ c dx = c*x
    Expr::Integer(n) => Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Integer(*n)),
      right: Box::new(Expr::Identifier(var.to_string())),
    }),
    Expr::Real(f) => Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Real(*f)),
      right: Box::new(Expr::Identifier(var.to_string())),
    }),

    // Variable: ∫ x dx = x^2/2, ∫ c dx = c*x
    Expr::Identifier(name) => {
      if name == var {
        Some(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(Expr::Identifier(var.to_string())),
            right: Box::new(Expr::Integer(2)),
          }),
          right: Box::new(Expr::Integer(2)),
        })
      } else {
        // Constant * x
        Some(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Identifier(name.clone())),
          right: Box::new(Expr::Identifier(var.to_string())),
        })
      }
    }

    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator::*;
      match op {
        Plus => {
          // ∫ (a + b) dx = ∫ a dx + ∫ b dx
          let int_a = integrate(left, var)?;
          let int_b = integrate(right, var)?;
          Some(Expr::BinaryOp {
            op: Plus,
            left: Box::new(int_a),
            right: Box::new(int_b),
          })
        }
        Minus => {
          // ∫ (a - b) dx = ∫ a dx - ∫ b dx
          let int_a = integrate(left, var)?;
          let int_b = integrate(right, var)?;
          Some(Expr::BinaryOp {
            op: Minus,
            left: Box::new(int_a),
            right: Box::new(int_b),
          })
        }
        Times => {
          // c * f(x) where c is constant
          if is_constant_wrt(left, var) {
            let int_b = integrate(right, var)?;
            Some(Expr::BinaryOp {
              op: Times,
              left: left.clone(),
              right: Box::new(int_b),
            })
          } else if is_constant_wrt(right, var) {
            let int_a = integrate(left, var)?;
            Some(Expr::BinaryOp {
              op: Times,
              left: right.clone(),
              right: Box::new(int_a),
            })
          } else {
            None
          }
        }
        Divide => {
          // f(x) / c where c is constant
          if is_constant_wrt(right, var) {
            let int_a = integrate(left, var)?;
            Some(Expr::BinaryOp {
              op: Divide,
              left: Box::new(int_a),
              right: right.clone(),
            })
          } else {
            None
          }
        }
        Power => {
          // ∫ x^n dx = x^(n+1)/(n+1) where n is constant
          if let Expr::Identifier(name) = left.as_ref()
            && name == var
            && is_constant_wrt(right, var)
          {
            let new_exp = simplify(Expr::BinaryOp {
              op: Plus,
              left: right.clone(),
              right: Box::new(Expr::Integer(1)),
            });
            return Some(Expr::BinaryOp {
              op: Divide,
              left: Box::new(Expr::BinaryOp {
                op: Power,
                left: left.clone(),
                right: Box::new(new_exp.clone()),
              }),
              right: Box::new(new_exp),
            });
          }
          // ∫ E^x dx = E^x, ∫ E^(a*x) dx = E^(a*x)/a,
          // ∫ E^(-a*x^2) dx = Gaussian integral
          if matches!(left.as_ref(), Expr::Constant(c) if c == "E") {
            let exp_arg = right.as_ref();
            // ∫ E^x dx = E^x
            if let Expr::Identifier(n) = exp_arg
              && n == var
            {
              return Some(expr.clone());
            }
            // ∫ E^(a*x) dx = E^(a*x)/a
            if let Some(coeff) = try_match_linear_arg(exp_arg, var) {
              return Some(make_divided(expr.clone(), coeff));
            }
            // ∫ E^(-a*x^2) dx = Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]
            if let Some(coeff) = match_neg_a_x_squared(exp_arg, var) {
              return Some(make_gaussian_antiderivative(var, &coeff));
            }
          }
          // ∫ Sin[x]^2 dx = x/2 - Sin[2*x]/4
          // ∫ Cos[x]^2 dx = x/2 + Sin[2*x]/4
          if matches!(right.as_ref(), Expr::Integer(2))
            && let Some(result) = try_integrate_trig_squared(left, var)
          {
            return Some(result);
          }
          None
        }
        _ => None,
      }
    }

    // Function calls
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Plus" if args.len() >= 2 => {
          // ∫ (a + b + ...) dx = ∫ a dx + ∫ b dx + ...
          let integrals: Option<Vec<Expr>> =
            args.iter().map(|arg| integrate(arg, var)).collect();
          integrals.map(|ints| Expr::FunctionCall {
            name: "Plus".to_string(),
            args: ints,
          })
        }
        "Sin" if args.len() == 1 => {
          // ∫ sin(a*x) dx = -cos(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let cos_expr = Expr::FunctionCall {
              name: "Cos".to_string(),
              args: args.clone(),
            };
            return Some(make_neg_divided(cos_expr, coeff));
          }
          None
        }
        "Cos" if args.len() == 1 => {
          // ∫ cos(a*x) dx = sin(a*x)/a
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let sin_expr = Expr::FunctionCall {
              name: "Sin".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(sin_expr, coeff));
          }
          None
        }
        "Exp" if args.len() == 1 => {
          // ∫ e^x dx = e^x
          if let Expr::Identifier(n) = &args[0]
            && n == var
          {
            return Some(Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            });
          }
          // ∫ e^(a*x) dx = e^(a*x)/a  (linear argument)
          if let Some(coeff) = try_match_linear_arg(&args[0], var) {
            let exp_expr = Expr::FunctionCall {
              name: "Exp".to_string(),
              args: args.clone(),
            };
            return Some(make_divided(exp_expr, coeff));
          }
          // ∫ Exp[-a*x^2] dx = Sqrt[Pi/a]/2 * Erf[Sqrt[a]*x]
          // (when a=1: Sqrt[Pi]/2 * Erf[x])
          if let Some(coeff) = match_neg_a_x_squared(&args[0], var) {
            return Some(make_gaussian_antiderivative(var, &coeff));
          }
          None
        }
        _ => None,
      }
    }

    // Unary minus
    Expr::UnaryOp { op, operand } => {
      use crate::syntax::UnaryOperator;
      if matches!(op, UnaryOperator::Minus) {
        let int = integrate(operand, var)?;
        Some(Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(int),
        })
      } else {
        None
      }
    }

    _ => None,
  }
}

/// Simplify an expression
pub fn simplify(expr: Expr) -> Expr {
  match expr {
    Expr::BinaryOp { op, left, right } => {
      let left = simplify(*left);
      let right = simplify(*right);

      use crate::syntax::BinaryOperator::*;
      match (&op, &left, &right) {
        // 0 + x = x
        (Plus, Expr::Integer(0), _) => return right,
        // x + 0 = x
        (Plus, _, Expr::Integer(0)) => return left,
        // 0 * x = 0
        (Times, Expr::Integer(0), _) | (Times, _, Expr::Integer(0)) => {
          return Expr::Integer(0);
        }
        // 1 * x = x
        (Times, Expr::Integer(1), _) => return right,
        // x * 1 = x
        (Times, _, Expr::Integer(1)) => return left,
        // x - 0 = x
        (Minus, _, Expr::Integer(0)) => return left,
        // 0 - n = -n  (for integers)
        (Minus, Expr::Integer(0), Expr::Integer(n)) => {
          return Expr::Integer(-n);
        }
        // 0 - (-x) = x
        (
          Minus,
          Expr::Integer(0),
          Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand,
          },
        ) => {
          return *operand.clone();
        }
        // 0 - x = -x  (general)
        (Minus, Expr::Integer(0), _) => {
          return Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(right),
          };
        }
        // x / 1 = x
        (Divide, _, Expr::Integer(1)) => return left,
        // x^0 = 1
        (Power, _, Expr::Integer(0)) => return Expr::Integer(1),
        // x^1 = x
        (Power, _, Expr::Integer(1)) => return left,
        // 0^n = 0 (for n > 0)
        (Power, Expr::Integer(0), Expr::Integer(n)) if *n > 0 => {
          return Expr::Integer(0);
        }
        // 1^n = 1
        (Power, Expr::Integer(1), _) => return Expr::Integer(1),
        // Numeric simplification
        (Plus, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a + b);
        }
        (Minus, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a - b);
        }
        (Times, Expr::Integer(a), Expr::Integer(b)) => {
          return Expr::Integer(a * b);
        }
        _ => {}
      }

      // For Times, delegate to times_ast for proper flattening and sorting
      if matches!(op, Times)
        && let Ok(result) =
          crate::functions::math_ast::times_ast(&[left.clone(), right.clone()])
      {
        return result;
      }
      // For Plus, delegate to plus_ast for proper sorting
      if matches!(op, Plus)
        && let Ok(result) =
          crate::functions::math_ast::plus_ast(&[left.clone(), right.clone()])
      {
        return result;
      }

      Expr::BinaryOp {
        op,
        left: Box::new(left),
        right: Box::new(right),
      }
    }
    Expr::UnaryOp { op, operand } => {
      let operand = simplify(*operand);
      use crate::syntax::UnaryOperator;
      if matches!(&op, UnaryOperator::Minus) {
        if let Expr::Integer(0) = operand {
          return Expr::Integer(0);
        }
        if let Expr::Integer(n) = operand {
          return Expr::Integer(-n);
        }
      }
      Expr::UnaryOp {
        op,
        operand: Box::new(operand),
      }
    }
    _ => expr,
  }
}

/// Extract base and exponent from Power expressions (both BinaryOp and FunctionCall forms)
fn extract_power(expr: &Expr) -> Option<(Expr, Expr)> {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => Some((*left.clone(), *right.clone())),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((args[0].clone(), args[1].clone()))
    }
    _ => None,
  }
}

/// Check if an expression approaches 1 when var -> Infinity
fn eval_at_infinity_is_one(expr: &Expr, var: &str) -> bool {
  // Substitute a large value and check if close to 1
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Integer(1_000_000));
  if let Ok(val) = crate::evaluator::evaluate_expr_to_expr(&subst)
    && let Some(f) = crate::functions::math_ast::try_eval_to_f64(&val)
  {
    return (f - 1.0).abs() < 0.01;
  }
  false
}

/// Check if an expression diverges to infinity when var -> Infinity
fn eval_at_infinity_diverges(expr: &Expr, var: &str) -> Option<bool> {
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Integer(1_000_000));
  if let Ok(val) = crate::evaluator::evaluate_expr_to_expr(&subst)
    && let Some(f) = crate::functions::math_ast::try_eval_to_f64(&val)
  {
    if f > 1e5 {
      return Some(true); // positive infinity
    }
    if f < -1e5 {
      return Some(false); // negative infinity (returns Some(false) for sign)
    }
  }
  None
}

/// Handle limits at infinity.
/// Strategies:
/// 1. If expr is constant wrt var, return it directly
/// 2. Direct substitution heuristic (evaluate at large n) to classify the limit
/// 3. For f^g where f->1, g->inf: Limit = E^(Limit[g*(f-1)])
/// 4. For expressions going to 0 or a constant: detect via structure
fn limit_at_infinity(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
) -> Result<Expr, InterpreterError> {
  // If the expression is constant wrt the variable, return it
  if is_constant_wrt(expr, var_name) {
    return crate::evaluator::evaluate_expr_to_expr(expr);
  }

  // Handle var itself: Limit[n, n -> Infinity] = Infinity
  if let Expr::Identifier(name) = expr
    && name == var_name
  {
    return Ok(point.clone());
  }

  // Handle f^g form (e.g., (1 + 1/n)^n -> E)
  if let Some((base, exp)) = extract_power(expr) {
    // Check if base -> 1 and exponent -> Infinity (1^Infinity indeterminate form)
    if eval_at_infinity_is_one(&base, var_name)
      && eval_at_infinity_diverges(&exp, var_name).is_some()
    {
      // Use identity: Limit[f^g] = E^(Limit[g * (f - 1)]) when f -> 1, g -> inf
      let f_minus_1 = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(base),
        right: Box::new(Expr::Integer(1)),
      };
      let product = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(exp),
        right: Box::new(f_minus_1),
      };
      // Take the limit of g * (f - 1) as var -> Infinity
      let rule = Expr::Rule {
        pattern: Box::new(Expr::Identifier(var_name.to_string())),
        replacement: Box::new(point.clone()),
      };
      let exponent_limit = limit_ast(&[product, rule])?;

      // If the exponent limit is a clean value, return E^limit
      if is_clean_value(&exponent_limit) {
        let result = simplify(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(Expr::Constant("E".to_string())),
          right: Box::new(exponent_limit),
        });
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
    }
  }

  // For simple expressions, try evaluating at two large values to detect convergence
  let val1 = eval_at_large_n(expr, var_name, 1_000_000);
  let val2 = eval_at_large_n(expr, var_name, 10_000_000);
  if let (Some(f1), Some(f2)) = (val1, val2) {
    // Both diverging to +infinity
    if f1 > 1e5 && f2 > f1 {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    // Both diverging to -infinity
    if f1 < -1e5 && f2 < f1 {
      return Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
    // Approaching zero: both values small and getting smaller
    if f2.abs() < 1e-4 && f2.abs() < f1.abs() {
      return Ok(Expr::Integer(0));
    }
    // Check convergence: values should be close (relative difference)
    let diff = (f1 - f2).abs();
    let scale = f1.abs().max(f2.abs()).max(1e-15);
    if diff / scale < 0.01 {
      // Converging to a nonzero limit — determine the value
      // Check if the limit is a known integer
      let rounded = f2.round();
      if (f2 - rounded).abs() < 1e-4 {
        return Ok(Expr::Integer(rounded as i128));
      }
      // Check for known constants
      if (f2 - std::f64::consts::E).abs() < 1e-3 {
        return Ok(Expr::Constant("E".to_string()));
      }
      if (f2 - std::f64::consts::PI).abs() < 1e-3 {
        return Ok(Expr::Constant("Pi".to_string()));
      }
    }
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Limit".to_string(),
    args: vec![
      expr.clone(),
      Expr::Rule {
        pattern: Box::new(Expr::Identifier(var_name.to_string())),
        replacement: Box::new(point.clone()),
      },
    ],
  })
}

/// Evaluate an expression numerically at var = n
fn eval_at_large_n(expr: &Expr, var: &str, n: i128) -> Option<f64> {
  let subst = crate::syntax::substitute_variable(expr, var, &Expr::Integer(n));
  let val = crate::evaluator::evaluate_expr_to_expr(&subst).ok()?;
  crate::functions::math_ast::try_eval_to_f64(&val)
}

/// Check if an expression is a "clean" value (integer, real, constant, or rational)
fn is_clean_value(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      true
    }
    _ => crate::functions::math_ast::try_eval_to_f64(expr).is_some(),
  }
}

/// Direction for one-sided limits
#[derive(Debug, Clone, Copy, PartialEq)]
enum LimitDirection {
  /// Two-sided limit (default)
  TwoSided,
  /// From above (x -> x0+), i.e. from the right
  FromAbove,
  /// From below (x -> x0-), i.e. from the left
  FromBelow,
}

/// Parse the Direction option from a Rule like `Direction -> "FromAbove"`
fn parse_direction(option: &Expr) -> Option<LimitDirection> {
  if let Expr::Rule {
    pattern,
    replacement,
  } = option
    && let Expr::Identifier(name) = pattern.as_ref()
    && name == "Direction"
  {
    match replacement.as_ref() {
      Expr::String(s) if s == "FromAbove" => {
        return Some(LimitDirection::FromAbove);
      }
      Expr::String(s) if s == "FromBelow" => {
        return Some(LimitDirection::FromBelow);
      }
      _ => {}
    }
  }
  None
}

/// Evaluate an expression numerically at var = point + delta
fn eval_near_point(
  expr: &Expr,
  var: &str,
  point: &Expr,
  delta: f64,
) -> Option<f64> {
  let point_val = crate::functions::math_ast::try_eval_to_f64(point)?;
  let val_at = point_val + delta;
  let subst =
    crate::syntax::substitute_variable(expr, var, &Expr::Real(val_at));
  let result = crate::evaluator::evaluate_expr_to_expr(&subst).ok()?;
  crate::functions::math_ast::try_eval_to_f64(&result)
}

/// Compute a one-sided limit numerically by evaluating at points approaching x0
fn numerical_one_sided_limit(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
  direction: LimitDirection,
) -> Option<Expr> {
  let sign = match direction {
    LimitDirection::FromAbove => 1.0,
    LimitDirection::FromBelow => -1.0,
    LimitDirection::TwoSided => return None,
  };

  // Evaluate at decreasing distances from the point
  let deltas = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12];
  let mut vals = Vec::new();
  for &d in &deltas {
    if let Some(v) = eval_near_point(expr, var_name, point, sign * d) {
      if v.is_nan() {
        return None;
      }
      vals.push(v);
    } else {
      return None;
    }
  }

  // Check for immediate infinity (even at the first sample point)
  if vals.iter().any(|v| v.is_infinite()) {
    // Determine sign from the first non-infinite value, or from the infinite ones
    let sign_positive = vals
      .iter()
      .find(|v| !v.is_infinite())
      .map(|v| *v > 0.0)
      .unwrap_or_else(|| vals[0].is_sign_positive());
    if sign_positive {
      return Some(Expr::Identifier("Infinity".to_string()));
    } else {
      return Some(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
  }

  // Check if the values are monotonically diverging (sign consistent, magnitude increasing)
  let all_positive = vals.iter().all(|&v| v > 0.0);
  let all_negative = vals.iter().all(|&v| v < 0.0);
  let magnitudes_increasing = vals.windows(2).all(|w| w[1].abs() > w[0].abs());

  if magnitudes_increasing && (all_positive || all_negative) {
    // Check that the growth is unbounded (magnitude at least doubles over the range)
    if vals.last().unwrap().abs() > 2.0 * vals.first().unwrap().abs() {
      if all_positive {
        return Some(Expr::Identifier("Infinity".to_string()));
      } else {
        return Some(Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        });
      }
    }
  }

  // Check convergence to a finite value
  let last = *vals.last().unwrap();
  let second_last = vals[vals.len() - 2];
  let diff = (last - second_last).abs();
  let scale = last.abs().max(second_last.abs()).max(1e-15);
  if diff / scale < 0.01 || diff < 1e-10 {
    // Converging — determine the value
    let rounded = last.round();
    if (last - rounded).abs() < 1e-6 {
      return Some(Expr::Integer(rounded as i128));
    }
    // Check for known constants
    if (last - std::f64::consts::E).abs() < 1e-4 {
      return Some(Expr::Constant("E".to_string()));
    }
    if (last - std::f64::consts::PI).abs() < 1e-4 {
      return Some(Expr::Constant("Pi".to_string()));
    }
    return Some(Expr::Real(last));
  }

  None
}

/// Compute a two-sided limit numerically by checking both sides agree
fn numerical_two_sided_limit(
  expr: &Expr,
  var_name: &str,
  point: &Expr,
) -> Option<Expr> {
  let from_above =
    numerical_one_sided_limit(expr, var_name, point, LimitDirection::FromAbove);
  let from_below =
    numerical_one_sided_limit(expr, var_name, point, LimitDirection::FromBelow);

  match (from_above, from_below) {
    (Some(a), Some(b)) => {
      // Check if both sides agree
      let a_val = crate::functions::math_ast::try_eval_to_f64(&a);
      let b_val = crate::functions::math_ast::try_eval_to_f64(&b);
      match (a_val, b_val) {
        (Some(av), Some(bv)) => {
          let diff = (av - bv).abs();
          let scale = av.abs().max(bv.abs()).max(1e-15);
          if diff / scale < 0.01 || diff < 1e-10 {
            // Both sides converge to the same value
            return Some(a);
          }
          // Sides disagree — indeterminate
          Some(Expr::Identifier("Indeterminate".to_string()))
        }
        _ => {
          // At least one side is infinite — check if they match symbolically
          let a_str = crate::syntax::expr_to_string(&a);
          let b_str = crate::syntax::expr_to_string(&b);
          if a_str == b_str {
            return Some(a);
          }
          // Different infinities — indeterminate
          Some(Expr::Identifier("Indeterminate".to_string()))
        }
      }
    }
    _ => None,
  }
}

/// Limit[expr, x -> x0] - Compute the limit of expr as x approaches x0
/// Limit[expr, x -> x0, Direction -> "FromAbove"] - One-sided limit
pub fn limit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Limit expects 2 or 3 arguments".into(),
    ));
  }

  // Parse optional Direction from 3rd argument
  let direction = if args.len() == 3 {
    parse_direction(&args[2]).unwrap_or(LimitDirection::TwoSided)
  } else {
    LimitDirection::TwoSided
  };

  // Second argument must be a Rule: x -> x0
  let (var_name, point) = match &args[1] {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let name = match pattern.as_ref() {
        Expr::Identifier(n) => n.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Limit".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (name, replacement.as_ref().clone())
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Limit".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Handle limits at Infinity
  if is_infinity(&point) || is_negative_infinity(&point) {
    return limit_at_infinity(&args[0], &var_name, &point);
  }

  // Strategy: try direct substitution first
  let substituted =
    crate::syntax::substitute_variable(&args[0], &var_name, &point);
  let result = crate::evaluator::evaluate_expr_to_expr(&substituted);

  match result {
    Ok(ref val) => {
      // Check if the result is a valid numeric value (not Indeterminate, ComplexInfinity, etc.)
      match val {
        Expr::Integer(_) | Expr::Real(_) | Expr::Constant(_) => {
          return result;
        }
        Expr::FunctionCall { name, args: fargs }
          if name == "Rational" && fargs.len() == 2 =>
        {
          // Check for 0/0 indeterminate form
          if matches!(&fargs[1], Expr::Integer(0)) {
            // Fall through to L'Hôpital
          } else {
            return result;
          }
        }
        Expr::FunctionCall { name, .. }
          if name == "DirectedInfinity" || name == "Indeterminate" =>
        {
          // Fall through to try other methods
        }
        _ => {
          // Check if it evaluates to a number via N[]
          if crate::functions::math_ast::try_eval_to_f64(val).is_some() {
            return result;
          }
          // Return unevaluated if substitution doesn't yield a clean result
        }
      }
    }
    Err(_) => {
      // Substitution failed (e.g., division by zero)
    }
  }

  // Try L'Hôpital's rule for 0/0 forms
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: num,
    right: den,
  } = &args[0]
  {
    let num_at_point =
      crate::syntax::substitute_variable(num, &var_name, &point);
    let den_at_point =
      crate::syntax::substitute_variable(den, &var_name, &point);
    let num_val = crate::evaluator::evaluate_expr_to_expr(&num_at_point);
    let den_val = crate::evaluator::evaluate_expr_to_expr(&den_at_point);

    if let (Ok(Expr::Integer(0)), Ok(Expr::Integer(0))) = (&num_val, &den_val) {
      // Apply L'Hôpital: Limit[f'/g', x -> x0]
      if let (Ok(df), Ok(dg)) =
        (differentiate(num, &var_name), differentiate(den, &var_name))
      {
        let new_expr = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(simplify(df)),
          right: Box::new(simplify(dg)),
        };
        return limit_ast(&[new_expr, args[1].clone()]);
      }
    }
  }

  // Numerical approach for one-sided or two-sided limits
  match direction {
    LimitDirection::FromAbove | LimitDirection::FromBelow => {
      if let Some(result) =
        numerical_one_sided_limit(&args[0], &var_name, &point, direction)
      {
        return Ok(result);
      }
    }
    LimitDirection::TwoSided => {
      if let Some(result) =
        numerical_two_sided_limit(&args[0], &var_name, &point)
      {
        return Ok(result);
      }
    }
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Limit".to_string(),
    args: args.to_vec(),
  })
}

/// Rational arithmetic helpers for coefficient-based series computation
/// Represents a rational number as (numerator, denominator) with denominator > 0
fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  let num = a.0 * b.1 + b.0 * a.1;
  let den = a.1 * b.1;
  rat_reduce(num, den)
}

fn rat_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  rat_reduce(a.0 * b.0, a.1 * b.1)
}

fn rat_div(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  if b.0 < 0 {
    rat_reduce(-a.0 * b.1, a.1 * -b.0)
  } else {
    rat_reduce(a.0 * b.1, a.1 * b.0)
  }
}

fn rat_reduce(num: i128, den: i128) -> (i128, i128) {
  if num == 0 {
    return (0, 1);
  }
  let g = gcd(num.abs(), den.abs());
  let (n, d) = (num / g, den / g);
  if d < 0 { (-n, -d) } else { (n, d) }
}

fn rat_to_expr(r: (i128, i128)) -> Expr {
  if r.1 == 1 {
    Expr::Integer(r.0)
  } else {
    crate::functions::math_ast::make_rational_pub(r.0, r.1)
  }
}

/// Cauchy product of two coefficient vectors (convolution)
fn cauchy_product(
  a: &[(i128, i128)],
  b: &[(i128, i128)],
  n: usize,
) -> (i128, i128) {
  let mut sum = (0i128, 1i128);
  for k in 0..=n {
    if k < a.len() && (n - k) < b.len() {
      sum = rat_add(sum, rat_mul(a[k], b[n - k]));
    }
  }
  sum
}

/// Compute Tan[x] series coefficients around x=0 up to order n
/// Uses recurrence: tan'(x) = 1 + tan²(x), t(0) = 0
fn tan_series_coeffs(order: usize) -> Vec<(i128, i128)> {
  let mut t: Vec<(i128, i128)> = vec![(0, 1)]; // t_0 = 0
  for n in 0..order {
    // (n+1)*t_{n+1} = delta_{n,0} + sum_{k=0..n} t_k * t_{n-k}
    let mut rhs = cauchy_product(&t, &t, n);
    if n == 0 {
      rhs = rat_add(rhs, (1, 1));
    }
    // t_{n+1} = rhs / (n+1)
    t.push(rat_div(rhs, ((n + 1) as i128, 1)));
  }
  t
}

/// Compute Sec[x] series coefficients around x=0 up to order n
/// Uses: sec'(x) = sec(x)*tan(x), sec(0) = 1
/// If s = sec, t = tan: s'= s*t, t' = 1 + t^2 = s^2
fn sec_series_coeffs(order: usize) -> Vec<(i128, i128)> {
  let mut s: Vec<(i128, i128)> = vec![(1, 1)]; // s_0 = 1
  let mut t: Vec<(i128, i128)> = vec![(0, 1)]; // t_0 = 0
  for n in 0..order {
    // (n+1)*t_{n+1} = sum_{k=0..n} s_k * s_{n-k}
    let t_rhs = cauchy_product(&s, &s, n);
    t.push(rat_div(t_rhs, ((n + 1) as i128, 1)));
    // (n+1)*s_{n+1} = sum_{k=0..n} s_k * t_{n-k}
    let s_rhs = cauchy_product(&s, &t, n);
    s.push(rat_div(s_rhs, ((n + 1) as i128, 1)));
  }
  s
}

/// Compute Csc[x] series coefficients around x=0 up to order n
/// csc(x) = 1/sin(x), has a pole at x=0 so series starts at x^{-1}
/// Returns (coefficients, nmin) where nmin is the starting power
fn csc_series_coeffs(order: usize) -> (Vec<(i128, i128)>, i128) {
  // csc(x) = 1/x + x/6 + 7*x^3/360 + ...
  // Use: sin(x)*csc(x) = 1, solve for csc coefficients
  // sin coefficients: s_1 = 1, s_3 = -1/6, s_5 = 1/120, ...
  let total = order + 2; // need extra terms since csc starts at x^{-1}
  let mut sin_c: Vec<(i128, i128)> = Vec::new();
  let mut factorial = 1i128;
  for k in 0..=total {
    if k % 2 == 0 {
      sin_c.push((0, 1));
    } else {
      let sign = if (k / 2) % 2 == 0 { 1 } else { -1 };
      sin_c.push((sign, factorial));
    }
    if k < total {
      factorial *= (k + 1) as i128;
    }
  }
  // Recompute: sin_c[k] = coefficient of x^k in sin(x)
  // We need sin_c as rationals properly
  let mut sin_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut fact = 1i128;
  for k in 0..=total {
    if k > 1 {
      fact *= k as i128;
    }
    if k % 2 == 0 {
      sin_coeffs.push((0, 1));
    } else {
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      sin_coeffs.push(rat_reduce(sign, fact));
    }
  }

  // csc(x) = c_{-1}/x + c_1*x + c_3*x^3 + ...
  // sin(x)*csc(x) = 1
  // Shift: let csc(x) = (1/x) * C(x) where C(x) = c_0 + c_1*x + c_2*x^2 + ...
  // Then sin(x) * C(x) / x = 1, so sin(x)/x * C(x) = 1
  // sinc(x) = sin(x)/x = 1 - x^2/6 + x^4/120 - ...
  // sinc coefficients: sinc_k = sin_coeffs[k+1] (shift by one)
  let mut sinc: Vec<(i128, i128)> = Vec::new();
  for k in 0..=total {
    if k + 1 < sin_coeffs.len() {
      sinc.push(sin_coeffs[k + 1]);
    } else {
      sinc.push((0, 1));
    }
  }

  // C(x) = 1/sinc(x), so sinc*C = 1
  // c_0 = 1/sinc_0 = 1
  // c_n = -(1/sinc_0) * sum_{k=1..n} sinc_k * c_{n-k}
  let mut c: Vec<(i128, i128)> = vec![(1, 1)];
  for n in 1..=total {
    let mut s = (0i128, 1i128);
    for k in 1..=n {
      if k < sinc.len() && (n - k) < c.len() {
        s = rat_add(s, rat_mul(sinc[k], c[n - k]));
      }
    }
    c.push(rat_reduce(-s.0, s.1));
  }

  // csc(x) = c_0/x + c_1 + c_2*x + ... = c_0*x^{-1} + c_1*x^0 + ...
  // In SeriesData format, nmin = -1, coefficients are c_0, c_1, c_2, ...
  // But only keep up to order
  let keep = (order as i128 + 2) as usize; // from x^{-1} to x^{order}
  let coeffs: Vec<(i128, i128)> = c.into_iter().take(keep).collect();
  (coeffs, -1)
}

/// Compute Cot[x] series coefficients around x=0
/// Uses: cot'(x) = -1 - cot²(x) (but cot has pole at 0)
/// cot(x) = cos(x)/sin(x) = 1/x - x/3 - x^3/45 - ...
/// Returns (coefficients, nmin)
fn cot_series_coeffs(order: usize) -> (Vec<(i128, i128)>, i128) {
  // Use cos(x)/sin(x) = cot(x)
  // cos(x) = cot(x)*sin(x)
  // Similar to csc: let cot(x) = (1/x)*C(x)
  // cos(x) = C(x)*sin(x)/x = C(x)*sinc(x)
  let total = order + 2;

  let mut sin_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut cos_coeffs: Vec<(i128, i128)> = Vec::new();
  let mut fact = 1i128;
  for k in 0..=total {
    if k > 1 {
      fact *= k as i128;
    }
    if k % 2 == 0 {
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      cos_coeffs.push(rat_reduce(sign, fact));
      sin_coeffs.push((0, 1));
    } else {
      cos_coeffs.push((0, 1));
      let sign = if (k / 2) % 2 == 0 { 1i128 } else { -1 };
      sin_coeffs.push(rat_reduce(sign, fact));
    }
  }

  // sinc = sin(x)/x
  let mut sinc: Vec<(i128, i128)> = Vec::new();
  for k in 0..=total {
    if k + 1 < sin_coeffs.len() {
      sinc.push(sin_coeffs[k + 1]);
    } else {
      sinc.push((0, 1));
    }
  }

  // C(x)*sinc(x) = cos(x), where cot(x) = C(x)/x
  // c_0 = cos_0/sinc_0 = 1
  // c_n = (cos_n - sum_{k=1..n} sinc_k * c_{n-k}) / sinc_0
  let mut c: Vec<(i128, i128)> = vec![(1, 1)];
  for n in 1..=total {
    let mut s = if n < cos_coeffs.len() {
      cos_coeffs[n]
    } else {
      (0, 1)
    };
    for k in 1..=n {
      if k < sinc.len() && (n - k) < c.len() {
        s = rat_add(s, rat_mul((-sinc[k].0, sinc[k].1), c[n - k]));
      }
    }
    c.push(s);
  }

  let keep = (order as i128 + 2) as usize;
  let coeffs: Vec<(i128, i128)> = c.into_iter().take(keep).collect();
  (coeffs, -1)
}

/// Try coefficient-based series for known functions around x=0.
/// Returns Some((coefficients, nmin)) if handled, None otherwise.
fn try_fast_series(
  expr: &Expr,
  var_name: &str,
  x0: &Expr,
  order: i128,
) -> Option<(Vec<(i128, i128)>, i128)> {
  // Only for expansion around 0
  if !matches!(x0, Expr::Integer(0)) {
    return None;
  }

  // Only for f[var] form
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      // Check inner arg is the variable
      if !matches!(&args[0], Expr::Identifier(v) if v == var_name) {
        return None;
      }
      let n = order as usize;
      match name.as_str() {
        "Tan" => Some((tan_series_coeffs(n), 0)),
        "Sec" => Some((sec_series_coeffs(n), 0)),
        "Csc" => Some(csc_series_coeffs(n)),
        "Cot" => Some(cot_series_coeffs(n)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Series[expr, {x, x0, n}] - Taylor series expansion
pub fn series_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Series expects exactly 2 arguments".into(),
    ));
  }

  // Second argument: {x, x0, n}
  let (var_name, x0, order) = match &args[1] {
    Expr::List(items) if items.len() == 3 => {
      let name = match &items[0] {
        Expr::Identifier(n) => n.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Series".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let order = match &items[2] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Series".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (name, items[1].clone(), order)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Series".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Try fast coefficient-based computation for known functions
  if let Some((rat_coeffs, nmin)) =
    try_fast_series(&args[0], &var_name, &x0, order)
  {
    let mut coefficients: Vec<Expr> =
      rat_coeffs.iter().map(|r| rat_to_expr(*r)).collect();
    let mut actual_nmin = nmin;

    // Strip leading zeros
    while !coefficients.is_empty()
      && matches!(coefficients[0], Expr::Integer(0))
    {
      coefficients.remove(0);
      actual_nmin += 1;
    }

    // Strip trailing zeros
    while coefficients.len() > 1
      && matches!(coefficients.last(), Some(Expr::Integer(0)))
    {
      coefficients.pop();
    }

    if coefficients.is_empty() {
      return Ok(Expr::Integer(0));
    }

    return Ok(Expr::FunctionCall {
      name: "SeriesData".to_string(),
      args: vec![
        Expr::Identifier(var_name),
        x0,
        Expr::List(coefficients),
        Expr::Integer(actual_nmin),
        Expr::Integer(order + 1),
        Expr::Integer(1),
      ],
    });
  }

  // Compute Taylor coefficients: f^(k)(x0) / k!
  let mut coefficients = Vec::new();
  let mut current_expr = args[0].clone();

  for k in 0..=order {
    // Evaluate current derivative at x0
    let substituted =
      crate::syntax::substitute_variable(&current_expr, &var_name, &x0);
    let value = crate::evaluator::evaluate_expr_to_expr(&substituted)?;

    // Compute k!
    let mut factorial = 1i128;
    for i in 2..=k {
      factorial *= i;
    }

    // Coefficient = value / k!
    let coeff = if matches!(&value, Expr::Integer(0)) {
      Expr::Integer(0)
    } else if factorial == 1 {
      value
    } else {
      // value / factorial
      match &value {
        Expr::Integer(n) => {
          let g = gcd(n.abs(), factorial);
          let (num, den) = (n / g, factorial / g);
          if den == 1 {
            Expr::Integer(num)
          } else {
            crate::functions::math_ast::make_rational_pub(num, den)
          }
        }
        _ => Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(value),
          right: Box::new(Expr::Integer(factorial)),
        },
      }
    };

    coefficients.push(coeff);

    // Differentiate for the next iteration (unless this is the last)
    if k < order {
      current_expr = match differentiate(&current_expr, &var_name) {
        Ok(d) => simplify(d),
        Err(_) => {
          return Ok(Expr::FunctionCall {
            name: "Series".to_string(),
            args: args.to_vec(),
          });
        }
      };
    }
  }

  // Strip leading zero coefficients and adjust nmin
  let mut nmin: i128 = 0;
  while !coefficients.is_empty() && matches!(coefficients[0], Expr::Integer(0))
  {
    coefficients.remove(0);
    nmin += 1;
  }

  // If all coefficients are zero, return 0
  if coefficients.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Build SeriesData[x, x0, {c0, c1, ...}, nmin, nmax, 1]
  Ok(Expr::FunctionCall {
    name: "SeriesData".to_string(),
    args: vec![
      Expr::Identifier(var_name),
      x0,
      Expr::List(coefficients),
      Expr::Integer(nmin),
      Expr::Integer(order + 1),
      Expr::Integer(1),
    ],
  })
}

fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// NIntegrate[expr, {var, lo, hi}] - Numerical integration
pub fn nintegrate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "NIntegrate expects exactly 2 arguments".into(),
    ));
  }

  // Second argument must be {var, lo, hi}
  let (var_name, lo, hi) = match &args[1] {
    Expr::List(items) if items.len() == 3 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "NIntegrate: first element of integration range must be a symbol"
              .into(),
          ));
        }
      };
      // Evaluate bounds to numeric values
      let lo_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
      let hi_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
      let lo = crate::functions::math_ast::try_eval_to_f64(&lo_expr)
        .ok_or_else(|| {
          InterpreterError::EvaluationError(
            "NIntegrate: lower bound must be numeric".into(),
          )
        })?;
      let hi = crate::functions::math_ast::try_eval_to_f64(&hi_expr)
        .ok_or_else(|| {
          InterpreterError::EvaluationError(
            "NIntegrate: upper bound must be numeric".into(),
          )
        })?;
      (var_name, lo, hi)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "NIntegrate expects {var, lo, hi} as second argument".into(),
      ));
    }
  };

  let integrand = &args[0];

  // Evaluate the integrand at a point
  let eval_at = |x: f64| -> Option<f64> {
    let substituted =
      crate::syntax::substitute_variable(integrand, &var_name, &Expr::Real(x));
    let evaluated =
      crate::evaluator::evaluate_expr_to_expr(&substituted).ok()?;
    crate::functions::math_ast::try_eval_to_f64(&evaluated)
  };

  let result = adaptive_simpson(&eval_at, lo, hi, 1e-12, 50);

  match result {
    Some(val) => Ok(Expr::Real(val)),
    None => Err(InterpreterError::EvaluationError(
      "NIntegrate: failed to converge or integrand is not numeric".into(),
    )),
  }
}

/// Adaptive Simpson's quadrature
fn adaptive_simpson(
  f: &dyn Fn(f64) -> Option<f64>,
  a: f64,
  b: f64,
  tol: f64,
  max_depth: u32,
) -> Option<f64> {
  let fa = f(a)?;
  let fb = f(b)?;
  let m = (a + b) / 2.0;
  let fm = f(m)?;
  let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
  adaptive_simpson_rec(f, a, b, tol, whole, fa, fm, fb, max_depth)
}

fn adaptive_simpson_rec(
  f: &dyn Fn(f64) -> Option<f64>,
  a: f64,
  b: f64,
  tol: f64,
  whole: f64,
  fa: f64,
  fm: f64,
  fb: f64,
  depth: u32,
) -> Option<f64> {
  let m = (a + b) / 2.0;
  let m1 = (a + m) / 2.0;
  let m2 = (m + b) / 2.0;
  let fm1 = f(m1)?;
  let fm2 = f(m2)?;
  let h = b - a;
  let left = h / 12.0 * (fa + 4.0 * fm1 + fm);
  let right = h / 12.0 * (fm + 4.0 * fm2 + fb);
  let refined = left + right;
  let error = (refined - whole) / 15.0;

  if depth == 0 || error.abs() < tol {
    Some(refined + error)
  } else {
    let left_result =
      adaptive_simpson_rec(f, a, m, tol / 2.0, left, fa, fm1, fm, depth - 1)?;
    let right_result =
      adaptive_simpson_rec(f, m, b, tol / 2.0, right, fm, fm2, fb, depth - 1)?;
    Some(left_result + right_result)
  }
}

/// Curl[{f1, f2}, {x1, x2}] - 2D curl (scalar)
/// Curl[{f1, f2, f3}, {x1, x2, x3}] - 3D curl (vector)
pub fn curl_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Curl expects exactly 2 arguments".into(),
    ));
  }
  let field = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Curl".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let vars = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Curl".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if field.len() == 2 && vars.len() == 2 {
    // 2D curl: dF2/dx1 - dF1/dx2
    let var1 = match &vars[0] {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Curl".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let var2 = match &vars[1] {
      Expr::Identifier(s) => s,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Curl".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let df2_dx1 = differentiate_expr(&field[1], var1)?;
    let df1_dx2 = differentiate_expr(&field[0], var2)?;
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(df2_dx1),
      right: Box::new(df1_dx2),
    };
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else if field.len() == 3 && vars.len() == 3 {
    // 3D curl: {dF3/dx2 - dF2/dx3, dF1/dx3 - dF3/dx1, dF2/dx1 - dF1/dx2}
    let var_names: Vec<&str> = vars
      .iter()
      .map(|v| match v {
        Expr::Identifier(s) => Ok(s.as_str()),
        _ => Err(InterpreterError::EvaluationError(
          "Curl: variables must be symbols".into(),
        )),
      })
      .collect::<Result<Vec<_>, _>>()?;

    let mut components = Vec::new();
    // Curl component i = dF_{(i+2)%3}/dx_{(i+1)%3} - dF_{(i+1)%3}/dx_{(i+2)%3}
    for i in 0..3 {
      let j = (i + 1) % 3;
      let k = (i + 2) % 3;
      let d1 = differentiate_expr(&field[k], var_names[j])?;
      let d2 = differentiate_expr(&field[j], var_names[k])?;
      let comp = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(d1),
        right: Box::new(d2),
      };
      components.push(crate::evaluator::evaluate_expr_to_expr(&comp)?);
    }
    Ok(Expr::List(components))
  } else {
    Ok(Expr::FunctionCall {
      name: "Curl".to_string(),
      args: args.to_vec(),
    })
  }
}
