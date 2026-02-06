//! AST-native calculus functions (D, Integrate).
//!
//! These functions work directly with `Expr` AST nodes for symbolic differentiation
//! and integration.

use crate::InterpreterError;
use crate::syntax::Expr;

/// D[expr, var] - Symbolic differentiation
pub fn d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "D expects exactly 2 arguments".into(),
    ));
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

/// Integrate[expr, var] - Symbolic integration
pub fn integrate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Integrate expects exactly 2 arguments".into(),
    ));
  }

  // Get the variable name
  let var_name = match &args[1] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of Integrate must be a symbol".into(),
      ));
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
          // ∫ sin(x) dx = -cos(x)
          if let Expr::Identifier(n) = &args[0]
            && n == var
          {
            return Some(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(Expr::FunctionCall {
                name: "Cos".to_string(),
                args: args.clone(),
              }),
            });
          }
          None
        }
        "Cos" if args.len() == 1 => {
          // ∫ cos(x) dx = sin(x)
          if let Expr::Identifier(n) = &args[0]
            && n == var
          {
            return Some(Expr::FunctionCall {
              name: "Sin".to_string(),
              args: args.clone(),
            });
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
