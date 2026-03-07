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
    "Laplacian" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::laplacian_ast(args));
    }
    "Div" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::div_ast(args));
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
    "InverseLaplaceTransform" if args.len() == 3 => {
      return Some(inverse_laplace_transform(&args[0], &args[1], &args[2]));
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

/// Compute the inverse Laplace transform of expr(s) with respect to s, yielding f(t).
fn inverse_laplace_transform(
  expr: &Expr,
  s_expr: &Expr,
  t_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let s = match s_expr {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "InverseLaplaceTransform".to_string(),
        args: vec![expr.clone(), s_expr.clone(), t_expr.clone()],
      });
    }
  };
  let t = match t_expr {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "InverseLaplaceTransform".to_string(),
        args: vec![expr.clone(), s_expr.clone(), t_expr.clone()],
      });
    }
  };

  // Normalize the expression by converting BinaryOps to FunctionCall form
  let normalized = normalize_to_func_calls(expr);
  let normalized =
    crate::evaluator::evaluate_expr_to_expr(&normalized).unwrap_or(normalized);
  if let Some(result) = inverse_laplace_inner(&normalized, s, t) {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(Expr::FunctionCall {
      name: "InverseLaplaceTransform".to_string(),
      args: vec![expr.clone(), s_expr.clone(), t_expr.clone()],
    })
  }
}

/// Try to compute inverse Laplace transform symbolically.
fn inverse_laplace_inner(expr: &Expr, s: &str, t: &str) -> Option<Expr> {
  // If expr doesn't depend on s, it's a constant * DiracDelta(t) — not commonly needed
  // Just skip this case and return None for constants

  if let Some((fname, fargs)) = as_func_args(expr) {
    // L^-1[s^(-n)] = t^(n-1) / Gamma[n]
    if fname == "Power" && fargs.len() == 2 {
      if let Expr::Identifier(base) = fargs[0]
        && base == s
      {
        // s^(-n) → t^(n-1) / Gamma[n]  (where n > 0)
        // The exponent is fargs[1], which should be negative
        if let Expr::Integer(exp) = fargs[1]
          && *exp < 0
        {
          let n = -exp; // positive
          if n == 1 {
            // s^(-1) → 1
            return Some(Expr::Integer(1));
          }
          // s^(-n) → t^(n-1) / (n-1)!
          return Some(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![
                  Expr::Identifier(t.to_string()),
                  Expr::Integer(n - 1),
                ],
              },
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![
                  Expr::FunctionCall {
                    name: "Gamma".to_string(),
                    args: vec![Expr::Integer(n)],
                  },
                  Expr::Integer(-1),
                ],
              },
            ],
          });
        }
      }

      // L^-1[(s^2 + a^2)^(-1)] = Sin[a*t] / a
      if matches!(fargs[1], Expr::Integer(-1))
        && let Some(a_squared) = extract_s_squared_plus_const(fargs[0], s)
      {
        let a = sqrt_of_expr(&a_squared);
        return Some(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![a.clone(), Expr::Integer(-1)],
            },
            Expr::FunctionCall {
              name: "Sin".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![a, Expr::Identifier(t.to_string())],
              }],
            },
          ],
        });
      }

      // L^-1[(s - a)^(-1)] = E^(a*t) or L^-1[(s + a)^(-1)] = E^(-a*t)
      if let Expr::Integer(-1) = fargs[1] {
        // Check if fargs[0] is (s + something) or (s - something)
        if let Some(neg_a) = extract_linear_s_offset(fargs[0], s) {
          // (s + neg_a)^(-1) → E^(-neg_a * t)
          let exponent = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(-1),
              neg_a,
              Expr::Identifier(t.to_string()),
            ],
          };
          return Some(Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![Expr::Constant("E".to_string()), exponent],
          });
        }
      }
    }

    // L^-1[a/(s^2 + a^2)] = Sin[a*t] and L^-1[s/(s^2 + a^2)] = Cos[a*t]
    if fname == "Times" && fargs.len() == 2 {
      // Check for pattern: X * (s^2 + a^2)^(-1)
      let (numerator, denom_base) = if is_power_neg1(fargs[1]) {
        (fargs[0], get_power_base(fargs[1]))
      } else if is_power_neg1(fargs[0]) {
        (fargs[1], get_power_base(fargs[0]))
      } else {
        (fargs[0], None)
      };

      if let Some(denom) = denom_base {
        // Check if denom is s^2 + a^2
        if let Some(a_squared) = extract_s_squared_plus_const(denom, s) {
          // numerator / (s^2 + a^2)
          if let Expr::Identifier(n) = numerator
            && n == s
          {
            // s / (s^2 + a^2) → Cos[a * t]
            let a = sqrt_of_expr(&a_squared);
            return Some(Expr::FunctionCall {
              name: "Cos".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![a, Expr::Identifier(t.to_string())],
              }],
            });
          }
          // For numerator/(s^2 + a^2) → (numerator/a) * Sin[a*t]
          if !depends_on(numerator, s) {
            let a = sqrt_of_expr(&a_squared);
            return Some(Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                numerator.clone(),
                Expr::FunctionCall {
                  name: "Power".to_string(),
                  args: vec![a.clone(), Expr::Integer(-1)],
                },
                Expr::FunctionCall {
                  name: "Sin".to_string(),
                  args: vec![Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![a, Expr::Identifier(t.to_string())],
                  }],
                },
              ],
            });
          }
        }
      }
    }

    // Linearity: L^-1[a + b] = L^-1[a] + L^-1[b]
    if fname == "Plus" {
      let mut terms = Vec::new();
      for arg in &fargs {
        if let Some(inv) = inverse_laplace_inner(arg, s, t) {
          terms.push(inv);
        } else {
          return None;
        }
      }
      return Some(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms,
      });
    }

    // Linearity: L^-1[c * F(s)] = c * L^-1[F(s)] where c doesn't depend on s
    if fname == "Times" && fargs.len() >= 2 {
      let mut constants = Vec::new();
      let mut s_dependent = Vec::new();
      for arg in &fargs {
        if depends_on(arg, s) {
          s_dependent.push((*arg).clone());
        } else {
          constants.push((*arg).clone());
        }
      }
      if !constants.is_empty() && !s_dependent.is_empty() {
        let s_part = if s_dependent.len() == 1 {
          s_dependent[0].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: s_dependent,
          }
        };
        if let Some(inv) = inverse_laplace_inner(&s_part, s, t) {
          constants.push(inv);
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

/// Extract offset from s + c or Plus[s, c] form. Returns the constant part.
fn extract_linear_s_offset(expr: &Expr, s: &str) -> Option<Expr> {
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "Plus"
    && fargs.len() == 2
  {
    if let Expr::Identifier(v) = fargs[0]
      && v == s
      && !depends_on(fargs[1], s)
    {
      return Some(fargs[1].clone());
    }
    if let Expr::Identifier(v) = fargs[1]
      && v == s
      && !depends_on(fargs[0], s)
    {
      return Some(fargs[0].clone());
    }
  }
  None
}

/// Check if expr is X^(-1)
fn is_power_neg1(expr: &Expr) -> bool {
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "Power"
    && fargs.len() == 2
  {
    return matches!(fargs[1], Expr::Integer(-1));
  }
  false
}

/// Get base from X^(-1), returning X
fn get_power_base(expr: &Expr) -> Option<&Expr> {
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "Power"
    && fargs.len() == 2
    && matches!(fargs[1], Expr::Integer(-1))
  {
    return Some(fargs[0]);
  }
  None
}

/// Try to extract the "a" from a^2 (i.e. Power[a, 2] → a, integer n → sqrt as integer if perfect square)
fn sqrt_of_expr(expr: &Expr) -> Expr {
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "Power"
    && fargs.len() == 2
    && matches!(fargs[1], Expr::Integer(2))
  {
    return fargs[0].clone();
  }
  if let Expr::Integer(n) = expr {
    let root = (*n as f64).sqrt();
    if root == root.floor() && root >= 0.0 {
      return Expr::Integer(root as i128);
    }
  }
  // Fallback: return Sqrt[expr]
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![
      expr.clone(),
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)],
      },
    ],
  }
}

/// Check if expr is s^2 + c where c doesn't depend on s. Returns c.
fn extract_s_squared_plus_const(expr: &Expr, s: &str) -> Option<Expr> {
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "Plus"
    && fargs.len() == 2
  {
    // Check each arg: one should be s^2, other should be constant
    for i in 0..2 {
      let other = 1 - i;
      if is_s_squared(fargs[i], s) && !depends_on(fargs[other], s) {
        return Some(fargs[other].clone());
      }
    }
  }
  None
}

/// Recursively convert BinaryOp forms to FunctionCall forms
fn normalize_to_func_calls(expr: &Expr) -> Expr {
  match expr {
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator;
      let left = normalize_to_func_calls(left);
      let right = normalize_to_func_calls(right);
      match op {
        BinaryOperator::Plus => Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![left, right],
        },
        BinaryOperator::Minus => Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            left,
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), right],
            },
          ],
        },
        BinaryOperator::Times => Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![left, right],
        },
        BinaryOperator::Divide => Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            left,
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![right, Expr::Integer(-1)],
            },
          ],
        },
        BinaryOperator::Power => Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![left, right],
        },
        _ => expr.clone(),
      }
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let inner = normalize_to_func_calls(operand);
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), inner],
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(normalize_to_func_calls).collect(),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(normalize_to_func_calls).collect())
    }
    _ => expr.clone(),
  }
}

/// Check if expr is s^2
fn is_s_squared(expr: &Expr, s: &str) -> bool {
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "Power"
    && fargs.len() == 2
    && let Expr::Identifier(base) = fargs[0]
  {
    return base == s && matches!(fargs[1], Expr::Integer(2));
  }
  false
}
