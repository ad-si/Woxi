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
        // Handle evaluated Times[a, b] (FunctionCall form of *)
        "Times" if args.len() == 2 => {
          // Product rule: d/dx[a * b] = a' * b + a * b'
          let da = differentiate(&args[0], var)?;
          let db = differentiate(&args[1], var)?;
          Ok(simplify(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(da),
              right: Box::new(args[1].clone()),
            }),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(args[0].clone()),
              right: Box::new(db),
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

/// Limit[expr, x -> x0] - Compute the limit of expr as x approaches x0
pub fn limit_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Limit expects exactly 2 arguments".into(),
    ));
  }

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

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Limit".to_string(),
    args: args.to_vec(),
  })
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
