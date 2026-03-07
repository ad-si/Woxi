#[allow(unused_imports)]
use super::*;

pub fn dispatch_calculus_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Derivative" if args.len() == 3 => {
      if let (Expr::Integer(n), Expr::Identifier(func_name)) =
        (&args[0], &args[1])
      {
        let n = *n as usize;
        let overloads = crate::FUNC_DEFS.with(|m| {
          let defs = m.borrow();
          defs.get(func_name).cloned()
        });
        if let Some(overloads) = overloads {
          for (
            params,
            _conditions,
            _defaults,
            _heads,
            _blank_types,
            body_expr,
          ) in &overloads
          {
            if params.len() == 1 {
              let param = &params[0];
              let mut deriv = body_expr.clone();
              for _ in 0..n {
                deriv = match crate::functions::calculus_ast::differentiate_expr(
                  &deriv, param,
                ) {
                  Ok(v) => v,
                  Err(e) => return Some(Err(e)),
                };
              }
              let substituted =
                crate::syntax::substitute_variable(&deriv, param, &args[2]);
              return Some(evaluate_expr_to_expr(&substituted));
            }
          }
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      }));
    }
    "Derivative" if args.len() <= 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      }));
    }
    "D" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::d_ast(args));
    }
    "Dt" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::dt_ast(args));
    }
    "Curl" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::curl_ast(args));
    }
    "Grad" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::grad_ast(args));
    }
    "Integrate" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::integrate_ast(args));
    }
    "NIntegrate" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::nintegrate_ast(args));
    }
    "Limit" if (2..=3).contains(&args.len()) => {
      return Some(crate::functions::calculus_ast::limit_ast(args));
    }
    "Series" if args.len() >= 2 => {
      return Some(crate::functions::calculus_ast::series_ast(args));
    }
    "RSolve" if args.len() == 3 => {
      return Some(crate::functions::rsolve_ast::rsolve_ast(args));
    }
    "RecurrenceTable" if args.len() == 3 => {
      return Some(crate::functions::rsolve_ast::recurrence_table_ast(args));
    }
    "DSolve" if args.len() == 3 => {
      return Some(crate::functions::ode_ast::dsolve_ast(args));
    }
    "NDSolve" if args.len() == 3 => {
      return Some(crate::functions::ode_ast::ndsolve_ast(args));
    }
    "LaplaceTransform" if args.len() == 3 => {
      return Some(laplace_transform(&args[0], &args[1], &args[2]));
    }
    _ => {}
  }
  None
}

/// Compute the Laplace transform of expr with respect to variable t, yielding a function of s.
fn laplace_transform(
  expr: &Expr,
  t_expr: &Expr,
  s_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let t = match t_expr {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LaplaceTransform".to_string(),
        args: vec![expr.clone(), t_expr.clone(), s_expr.clone()],
      });
    }
  };

  if let Some(result) = laplace_transform_inner(expr, t, s_expr) {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    // Return unevaluated
    Ok(Expr::FunctionCall {
      name: "LaplaceTransform".to_string(),
      args: vec![expr.clone(), t_expr.clone(), s_expr.clone()],
    })
  }
}

/// Check if an expression depends on variable t
fn depends_on(expr: &Expr, t: &str) -> bool {
  match expr {
    Expr::Identifier(name) => name == t,
    Expr::Integer(_) | Expr::Real(_) | Expr::String(_) => false,
    Expr::FunctionCall { args, .. } => args.iter().any(|a| depends_on(a, t)),
    Expr::List(items) => items.iter().any(|a| depends_on(a, t)),
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => depends_on(pattern, t) || depends_on(replacement, t),
    Expr::BinaryOp { left, right, .. } => {
      depends_on(left, t) || depends_on(right, t)
    }
    Expr::UnaryOp { operand, .. } => depends_on(operand, t),
    _ => true, // conservative: assume depends
  }
}

/// Extract the function name and args from an expression, handling both
/// FunctionCall and BinaryOp forms uniformly.
fn as_func_args(expr: &Expr) -> Option<(&str, Vec<&Expr>)> {
  match expr {
    Expr::FunctionCall { name, args } => {
      Some((name.as_str(), args.iter().collect()))
    }
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator;
      let name = match op {
        BinaryOperator::Plus => "Plus",
        BinaryOperator::Minus => "Plus", // a - b is Plus[a, Times[-1, b]]
        BinaryOperator::Times => "Times",
        BinaryOperator::Divide => "Times", // a / b is Times[a, Power[b, -1]]
        BinaryOperator::Power => "Power",
        _ => return None,
      };
      Some((name, vec![left.as_ref(), right.as_ref()]))
    }
    _ => None,
  }
}

/// Try to compute Laplace transform symbolically. Returns None if not recognized.
fn laplace_transform_inner(expr: &Expr, t: &str, s: &Expr) -> Option<Expr> {
  // L[constant, t, s] = constant/s (if expr doesn't depend on t)
  if !depends_on(expr, t) {
    return Some(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        expr.clone(),
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![s.clone(), Expr::Integer(-1)],
        },
      ],
    });
  }

  // L[t, t, s] = 1/s^2
  if let Expr::Identifier(name) = expr
    && name == t
  {
    return Some(Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![s.clone(), Expr::Integer(-2)],
    });
  }

  if let Some((fname, fargs)) = as_func_args(expr) {
    // L[t^n, t, s] = n! / s^(n+1) for integer n >= 0
    if fname == "Power" && fargs.len() == 2 {
      // Check if base is the variable t (as Identifier)
      if let Expr::Identifier(base) = fargs[0]
        && base == t
        && !depends_on(fargs[1], t)
      {
        // t^n → Gamma[n+1] * s^(-n-1)
        let n = fargs[1];
        return Some(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Gamma".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![n.clone(), Expr::Integer(1)],
              }],
            },
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![
                s.clone(),
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![
                    Expr::Integer(-1),
                    Expr::FunctionCall {
                      name: "Times".to_string(),
                      args: vec![Expr::Integer(-1), n.clone()],
                    },
                  ],
                },
              ],
            },
          ],
        });
      }
      // L[E^(a*t), t, s] = 1/(s - a)  — E can be Identifier("E") or Constant("E")
      let is_e = matches!(fargs[0], Expr::Identifier(b) if b == "E")
        || matches!(fargs[0], Expr::Constant(b) if b == "E");
      if is_e && let Some(a) = extract_linear_coeff(fargs[1], t) {
        return Some(Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                s.clone(),
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![Expr::Integer(-1), a],
                },
              ],
            },
            Expr::Integer(-1),
          ],
        });
      }
    }

    // L[Sin[a*t], t, s] = a/(s^2 + a^2)
    if fname == "Sin"
      && fargs.len() == 1
      && let Some(a) = extract_linear_coeff(fargs[0], t)
    {
      return Some(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          a.clone(),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![s.clone(), Expr::Integer(2)],
                  },
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![a, Expr::Integer(2)],
                  },
                ],
              },
              Expr::Integer(-1),
            ],
          },
        ],
      });
    }

    // L[Cos[a*t], t, s] = s/(s^2 + a^2)
    if fname == "Cos"
      && fargs.len() == 1
      && let Some(a) = extract_linear_coeff(fargs[0], t)
    {
      return Some(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          s.clone(),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![s.clone(), Expr::Integer(2)],
                  },
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![a, Expr::Integer(2)],
                  },
                ],
              },
              Expr::Integer(-1),
            ],
          },
        ],
      });
    }

    // Linearity: L[a + b, t, s] = L[a, t, s] + L[b, t, s]
    if fname == "Plus" {
      let mut terms = Vec::new();
      for arg in &fargs {
        if let Some(lt) = laplace_transform_inner(arg, t, s) {
          terms.push(lt);
        } else {
          return None;
        }
      }
      return Some(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms,
      });
    }

    // Linearity: L[c * f(t), t, s] = c * L[f(t), t, s] where c doesn't depend on t
    if fname == "Times" && fargs.len() >= 2 {
      let mut constants = Vec::new();
      let mut t_dependent = Vec::new();
      for arg in &fargs {
        if depends_on(arg, t) {
          t_dependent.push((*arg).clone());
        } else {
          constants.push((*arg).clone());
        }
      }
      if !constants.is_empty() && !t_dependent.is_empty() {
        let t_part = if t_dependent.len() == 1 {
          t_dependent[0].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: t_dependent,
          }
        };
        if let Some(lt) = laplace_transform_inner(&t_part, t, s) {
          constants.push(lt);
          return Some(Expr::FunctionCall {
            name: "Times".to_string(),
            args: constants,
          });
        }
      }
    }
  }

  None
}

/// Extract coefficient `a` from expression of the form `a*t` or just `t` (returns 1).
/// Returns None if the expression is not linear in t.
fn extract_linear_coeff(expr: &Expr, t: &str) -> Option<Expr> {
  // Just t → coefficient is 1
  if let Expr::Identifier(name) = expr
    && name == t
  {
    return Some(Expr::Integer(1));
  }
  if let Some((fname, fargs)) = as_func_args(expr) {
    if fname == "Times" && fargs.len() == 2 {
      if let Expr::Identifier(v) = fargs[1]
        && v == t
        && !depends_on(fargs[0], t)
      {
        return Some(fargs[0].clone());
      }
      if let Expr::Identifier(v) = fargs[0]
        && v == t
        && !depends_on(fargs[1], t)
      {
        return Some(fargs[1].clone());
      }
    }
    // Handle Times with more than 2 args
    if fname == "Times" && fargs.len() > 2 {
      let t_idx = fargs
        .iter()
        .position(|a| matches!(a, Expr::Identifier(n) if n == t));
      if let Some(idx) = t_idx {
        let mut rest: Vec<_> = fargs
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != idx)
          .map(|(_, a)| (*a).clone())
          .collect();
        if rest.iter().all(|a| !depends_on(a, t)) {
          return Some(if rest.len() == 1 {
            rest.remove(0)
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            }
          });
        }
      }
    }
  }
  None
}
