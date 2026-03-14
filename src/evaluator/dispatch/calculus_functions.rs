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
    "Wronskian" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::wronskian_ast(args));
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
    "SeriesCoefficient" if args.len() == 2 => {
      // SeriesCoefficient[f, {x, x0, n}]
      if let Expr::List(spec) = &args[1]
        && spec.len() == 3
        && let Some(n_val) = crate::functions::math_ast::expr_to_i128(&spec[2])
        && n_val >= 0
      {
        let n = n_val as usize;
        // Compute Series[f, {x, x0, n}]
        let series_result = crate::functions::calculus_ast::series_ast(&[
          args[0].clone(),
          Expr::List(spec.clone()),
        ]);
        match series_result {
          Ok(Expr::FunctionCall {
            ref name,
            args: ref sargs,
          }) if name == "SeriesData" && sargs.len() >= 6 => {
            // SeriesData[x, x0, coeffs, nmin, nmax, den]
            if let Expr::List(ref coeffs) = sargs[2]
              && let Some(nmin) =
                crate::functions::math_ast::expr_to_i128(&sargs[3])
              && let Some(den) =
                crate::functions::math_ast::expr_to_i128(&sargs[5])
              && den == 1
            {
              let idx = (n as i128 - nmin) as usize;
              if idx < coeffs.len() {
                return Some(Ok(coeffs[idx].clone()));
              } else {
                return Some(Ok(Expr::Integer(0)));
              }
            }
          }
          Ok(_) => {}
          Err(e) => return Some(Err(e)),
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "SeriesCoefficient".to_string(),
        args: args.to_vec(),
      }));
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
    "DSolveValue" if args.len() == 3 => {
      // DSolveValue[eqs, y[x], x] = value from DSolve[eqs, y[x], x]
      let result = crate::functions::ode_ast::dsolve_ast(args);
      return Some(match result {
        Ok(result_expr) => extract_value_from_solve_result(&result_expr)
          .map(Ok)
          .unwrap_or(Ok(result_expr)),
        err => err,
      });
    }
    "Interpolation" | "ListInterpolation" if !args.is_empty() => {
      return Some(crate::functions::ode_ast::interpolation_ast(args));
    }
    "NDSolve" if args.len() == 3 => {
      return Some(crate::functions::ode_ast::ndsolve_ast(args));
    }
    "NDSolveValue" if args.len() == 3 => {
      let result = crate::functions::ode_ast::ndsolve_ast(args);
      return Some(match result {
        Ok(result_expr) => extract_value_from_solve_result(&result_expr)
          .map(Ok)
          .unwrap_or(Ok(result_expr)),
        err => err,
      });
    }
    "LaplaceTransform" if args.len() == 3 => {
      return Some(laplace_transform(&args[0], &args[1], &args[2]));
    }
    "InverseLaplaceTransform" if args.len() == 3 => {
      return Some(inverse_laplace_transform(&args[0], &args[1], &args[2]));
    }
    "FourierTransform" if args.len() == 3 => {
      return Some(fourier_transform(&args[0], &args[1], &args[2]));
    }
    "InverseFourierTransform" if args.len() == 3 => {
      return Some(inverse_fourier_transform(&args[0], &args[1], &args[2]));
    }
    "FunctionDomain" if args.len() >= 2 && args.len() <= 3 => {
      return Some(function_domain_ast(args));
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

    // L[BesselJ[n, a*t], t, s] = a^n / (Sqrt[a^2 + s^2] * (s + Sqrt[a^2 + s^2])^n)
    if fname == "BesselJ"
      && fargs.len() == 2
      && !depends_on(fargs[0], t)
      && let Some(a) = extract_linear_coeff(fargs[1], t)
    {
      let n = fargs[0];
      // sqrt_term = Sqrt[a^2 + s^2]
      let sqrt_term = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![a.clone(), Expr::Integer(2)],
              },
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![s.clone(), Expr::Integer(2)],
              },
            ],
          },
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)],
          },
        ],
      };
      // result = a^n / (sqrt_term * (s + sqrt_term)^n)
      //        = Times[Power[a, n], Power[sqrt_term, -1], Power[Plus[s, sqrt_term], Times[-1, n]]]
      return Some(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![a, n.clone()],
          },
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![sqrt_term.clone(), Expr::Integer(-1)],
          },
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![s.clone(), sqrt_term],
              },
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), n.clone()],
              },
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

// ── FourierTransform implementation ──────────────────────────────────
// Convention: F[f(t), t, ω] = (1/√(2π)) ∫_{-∞}^{∞} f(t) e^{iωt} dt

/// Helper to build Sqrt[expr]
fn make_sqrt(expr: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![
      expr,
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)],
      },
    ],
  }
}

/// Helper to build Times[args...]
fn make_times(args: Vec<Expr>) -> Expr {
  if args.len() == 1 {
    args.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args,
    }
  }
}

/// Helper to build Plus[args...]
fn make_plus(args: Vec<Expr>) -> Expr {
  if args.len() == 1 {
    args.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args,
    }
  }
}

/// Helper to build Power[base, exp]
fn make_power(base: Expr, exp: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![base, exp],
  }
}

/// Helper to build DiracDelta[x]
fn make_dirac_delta(arg: Expr) -> Expr {
  Expr::FunctionCall {
    name: "DiracDelta".to_string(),
    args: vec![arg],
  }
}

/// Compute Fourier transform of expr w.r.t. variable t, into variable w.
fn fourier_transform(
  expr: &Expr,
  t_expr: &Expr,
  w_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let t = match t_expr {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FourierTransform".to_string(),
        args: vec![expr.clone(), t_expr.clone(), w_expr.clone()],
      });
    }
  };

  // Normalize expression
  let normalized = normalize_to_func_calls(expr);
  let normalized =
    crate::evaluator::evaluate_expr_to_expr(&normalized).unwrap_or(normalized);

  if let Some(result) = fourier_transform_inner(&normalized, t, w_expr) {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(Expr::FunctionCall {
      name: "FourierTransform".to_string(),
      args: vec![expr.clone(), t_expr.clone(), w_expr.clone()],
    })
  }
}

/// Try to compute Fourier transform symbolically. Returns None if not recognized.
fn fourier_transform_inner(expr: &Expr, t: &str, w: &Expr) -> Option<Expr> {
  // F[constant] = constant * Sqrt[2*Pi] * DiracDelta[w]
  if !depends_on(expr, t) {
    return Some(make_times(vec![
      expr.clone(),
      make_sqrt(make_times(vec![
        Expr::Integer(2),
        Expr::Constant("Pi".to_string()),
      ])),
      make_dirac_delta(w.clone()),
    ]));
  }

  // F[DiracDelta[t]] = 1/Sqrt[2*Pi]
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "DiracDelta"
    && fargs.len() == 1
  {
    if let Expr::Identifier(v) = fargs[0]
      && v == t
    {
      return Some(make_power(
        make_times(vec![Expr::Integer(2), Expr::Constant("Pi".to_string())]),
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-1), Expr::Integer(2)],
        },
      ));
    }
    // F[DiracDelta[t - a]] = E^(i*a*w) / Sqrt[2*Pi]
    if let Some(neg_a) = extract_linear_s_offset(fargs[0], t) {
      // DiracDelta[t + neg_a] means shift by -neg_a
      // F = E^(i*(-neg_a)*w) / Sqrt[2*Pi]
      return Some(make_times(vec![
        make_power(
          Expr::Constant("E".to_string()),
          make_times(vec![
            Expr::Constant("I".to_string()),
            make_times(vec![Expr::Integer(-1), neg_a]),
            w.clone(),
          ]),
        ),
        make_power(
          make_times(vec![Expr::Integer(2), Expr::Constant("Pi".to_string())]),
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-1), Expr::Integer(2)],
          },
        ),
      ]));
    }
  }

  if let Some((fname, fargs)) = as_func_args(expr) {
    // F[E^(-a*t^2)] = E^(-w^2/(4a)) / Sqrt[2a]
    if fname == "Power" && fargs.len() == 2 {
      let is_e = matches!(fargs[0], Expr::Identifier(b) if b == "E")
        || matches!(fargs[0], Expr::Constant(b) if b == "E");
      if is_e {
        if let Some(a) = match_neg_a_t_squared(fargs[1], t) {
          // F[E^(-a*t^2)] = E^(-w^2/(4a)) / Sqrt[2a]
          return Some(make_times(vec![
            make_power(
              Expr::Constant("E".to_string()),
              make_times(vec![
                Expr::Integer(-1),
                make_power(w.clone(), Expr::Integer(2)),
                make_power(
                  make_times(vec![Expr::Integer(4), a.clone()]),
                  Expr::Integer(-1),
                ),
              ]),
            ),
            make_power(
              make_times(vec![Expr::Integer(2), a]),
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(-1), Expr::Integer(2)],
              },
            ),
          ]));
        }
        // F[E^(-a*|t|)] = Sqrt[2/Pi] * a / (a^2 + w^2)
        if let Some(a) = match_neg_a_abs_t(fargs[1], t) {
          return Some(make_times(vec![
            make_sqrt(make_times(vec![
              Expr::Integer(2),
              make_power(Expr::Constant("Pi".to_string()), Expr::Integer(-1)),
            ])),
            a.clone(),
            make_power(
              make_plus(vec![
                make_power(a, Expr::Integer(2)),
                make_power(w.clone(), Expr::Integer(2)),
              ]),
              Expr::Integer(-1),
            ),
          ]));
        }
      }
    }

    // F[Sin[a*t]] = I*Sqrt[Pi/2] * (DiracDelta[w-a] - DiracDelta[w+a])
    // Wait, Mathematica gives: I*Sqrt[Pi/2]*DiracDelta[-a + w] - I*Sqrt[Pi/2]*DiracDelta[a + w]
    if fname == "Sin"
      && fargs.len() == 1
      && let Some(a) = extract_linear_coeff(fargs[0], t)
    {
      let coeff = make_times(vec![
        Expr::Constant("I".to_string()),
        make_sqrt(make_times(vec![
          Expr::Constant("Pi".to_string()),
          make_power(Expr::Integer(2), Expr::Integer(-1)),
        ])),
      ]);
      return Some(make_plus(vec![
        make_times(vec![
          coeff.clone(),
          make_dirac_delta(make_plus(vec![
            make_times(vec![Expr::Integer(-1), a.clone()]),
            w.clone(),
          ])),
        ]),
        make_times(vec![
          Expr::Integer(-1),
          coeff,
          make_dirac_delta(make_plus(vec![a, w.clone()])),
        ]),
      ]));
    }

    // F[Cos[a*t]] = Sqrt[Pi/2] * (DiracDelta[w-a] + DiracDelta[w+a])
    if fname == "Cos"
      && fargs.len() == 1
      && let Some(a) = extract_linear_coeff(fargs[0], t)
    {
      let coeff = make_sqrt(make_times(vec![
        Expr::Constant("Pi".to_string()),
        make_power(Expr::Integer(2), Expr::Integer(-1)),
      ]));
      return Some(make_plus(vec![
        make_times(vec![
          coeff.clone(),
          make_dirac_delta(make_plus(vec![
            make_times(vec![Expr::Integer(-1), a.clone()]),
            w.clone(),
          ])),
        ]),
        make_times(vec![
          coeff,
          make_dirac_delta(make_plus(vec![a, w.clone()])),
        ]),
      ]));
    }

    // F[t] => special: derivative of delta => not commonly useful, skip
    // F[1/t] = I*Sqrt[Pi/2] * Sign[w]
    if fname == "Power"
      && fargs.len() == 2
      && let Expr::Identifier(base) = fargs[0]
      && base == t
      && matches!(fargs[1], Expr::Integer(-1))
    {
      return Some(make_times(vec![
        Expr::Constant("I".to_string()),
        make_sqrt(make_times(vec![
          Expr::Constant("Pi".to_string()),
          make_power(Expr::Integer(2), Expr::Integer(-1)),
        ])),
        Expr::FunctionCall {
          name: "Sign".to_string(),
          args: vec![w.clone()],
        },
      ]));
    }

    // Linearity: F[a + b] = F[a] + F[b]
    if fname == "Plus" {
      let mut terms = Vec::new();
      for arg in &fargs {
        if let Some(ft) = fourier_transform_inner(arg, t, w) {
          terms.push(ft);
        } else {
          return None;
        }
      }
      return Some(make_plus(terms));
    }

    // Linearity: F[c * f(t)] = c * F[f(t)] where c doesn't depend on t
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
        if let Some(ft) = fourier_transform_inner(&t_part, t, w) {
          constants.push(ft);
          return Some(make_times(constants));
        }
      }
    }
  }

  // F[t] (just the variable itself)
  if let Expr::Identifier(name) = expr
    && name == t
  {
    // F[t] = -I * Sqrt[2*Pi] * DiracDelta'[w]
    // This involves derivative of DiracDelta, return unevaluated
    return None;
  }

  None
}

/// Match exponent of form -a*t^2 where a > 0 and return a.
fn match_neg_a_t_squared(exp: &Expr, t: &str) -> Option<Expr> {
  // After normalization, -t^2 might appear as Times[-1, Power[t, 2]]
  // and -a*t^2 as Times[-1, a, Power[t, 2]] or Times[Times[-1, a], Power[t, 2]]
  if let Some((fname, fargs)) = as_func_args(exp) {
    if fname == "Times" {
      // Find t^2 factor and collect the rest
      let mut t_sq_idx = None;
      for (i, arg) in fargs.iter().enumerate() {
        if let Some(("Power", pargs)) = as_func_args(arg)
          && pargs.len() == 2
          && let Expr::Identifier(v) = pargs[0]
          && v == t
          && matches!(pargs[1], Expr::Integer(2))
        {
          t_sq_idx = Some(i);
          break;
        }
      }
      if let Some(idx) = t_sq_idx {
        let rest: Vec<_> = fargs
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != idx)
          .map(|(_, a)| (*a).clone())
          .collect();
        if rest.iter().all(|a| !depends_on(a, t)) {
          // The coefficient of t^2 is the product of rest
          // We need it to be negative (i.e. -a where a > 0)
          // Return a = -coefficient
          let coeff = if rest.len() == 1 {
            rest[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            }
          };
          // a = -coeff (negate)
          return Some(make_times(vec![Expr::Integer(-1), coeff]));
        }
      }
    }
  }
  None
}

/// Match exponent of form -a*Abs[t] and return a.
fn match_neg_a_abs_t(exp: &Expr, t: &str) -> Option<Expr> {
  if let Some((fname, fargs)) = as_func_args(exp) {
    if fname == "Times" {
      // Find Abs[t] factor
      let mut abs_idx = None;
      for (i, arg) in fargs.iter().enumerate() {
        if let Some(("Abs", aargs)) = as_func_args(arg)
          && aargs.len() == 1
          && let Expr::Identifier(v) = aargs[0]
          && v == t
        {
          abs_idx = Some(i);
          break;
        }
      }
      if let Some(idx) = abs_idx {
        let rest: Vec<_> = fargs
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != idx)
          .map(|(_, a)| (*a).clone())
          .collect();
        if rest.iter().all(|a| !depends_on(a, t)) {
          let coeff = if rest.len() == 1 {
            rest[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            }
          };
          return Some(make_times(vec![Expr::Integer(-1), coeff]));
        }
      }
    }
  }
  None
}

// ── InverseFourierTransform implementation ───────────────────────────
// Convention: F^-1[g(w), w, t] = (1/√(2π)) ∫_{-∞}^{∞} g(w) e^{-iwt} dw

fn inverse_fourier_transform(
  expr: &Expr,
  w_expr: &Expr,
  t_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let w = match w_expr {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "InverseFourierTransform".to_string(),
        args: vec![expr.clone(), w_expr.clone(), t_expr.clone()],
      });
    }
  };

  let normalized = normalize_to_func_calls(expr);
  let normalized =
    crate::evaluator::evaluate_expr_to_expr(&normalized).unwrap_or(normalized);

  if let Some(result) = inverse_fourier_inner(&normalized, w, t_expr) {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(Expr::FunctionCall {
      name: "InverseFourierTransform".to_string(),
      args: vec![expr.clone(), w_expr.clone(), t_expr.clone()],
    })
  }
}

fn inverse_fourier_inner(expr: &Expr, w: &str, t: &Expr) -> Option<Expr> {
  // F^-1[constant] = constant * Sqrt[2*Pi] * DiracDelta[t]
  if !depends_on(expr, w) {
    return Some(make_times(vec![
      expr.clone(),
      make_sqrt(make_times(vec![
        Expr::Integer(2),
        Expr::Constant("Pi".to_string()),
      ])),
      make_dirac_delta(t.clone()),
    ]));
  }

  // F^-1[DiracDelta[w]] = 1/Sqrt[2*Pi]
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "DiracDelta"
    && fargs.len() == 1
    && let Expr::Identifier(v) = fargs[0]
    && v == w
  {
    return Some(make_power(
      make_times(vec![Expr::Integer(2), Expr::Constant("Pi".to_string())]),
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(-1), Expr::Integer(2)],
      },
    ));
  }

  if let Some((fname, fargs)) = as_func_args(expr) {
    // F^-1[E^(-a*w^2)] = E^(-t^2/(4a)) / Sqrt[2a]
    if fname == "Power" && fargs.len() == 2 {
      let is_e = matches!(fargs[0], Expr::Identifier(b) if b == "E")
        || matches!(fargs[0], Expr::Constant(b) if b == "E");
      if is_e {
        if let Some(a) = match_neg_a_t_squared(fargs[1], w) {
          return Some(make_times(vec![
            make_power(
              Expr::Constant("E".to_string()),
              make_times(vec![
                Expr::Integer(-1),
                make_power(t.clone(), Expr::Integer(2)),
                make_power(
                  make_times(vec![Expr::Integer(4), a.clone()]),
                  Expr::Integer(-1),
                ),
              ]),
            ),
            make_power(
              make_times(vec![Expr::Integer(2), a]),
              Expr::FunctionCall {
                name: "Rational".to_string(),
                args: vec![Expr::Integer(-1), Expr::Integer(2)],
              },
            ),
          ]));
        }
      }
    }

    // Linearity
    if fname == "Plus" {
      let mut terms = Vec::new();
      for arg in &fargs {
        if let Some(ft) = inverse_fourier_inner(arg, w, t) {
          terms.push(ft);
        } else {
          return None;
        }
      }
      return Some(make_plus(terms));
    }

    if fname == "Times" && fargs.len() >= 2 {
      let mut constants = Vec::new();
      let mut w_dependent = Vec::new();
      for arg in &fargs {
        if depends_on(arg, w) {
          w_dependent.push((*arg).clone());
        } else {
          constants.push((*arg).clone());
        }
      }
      if !constants.is_empty() && !w_dependent.is_empty() {
        let w_part = if w_dependent.len() == 1 {
          w_dependent[0].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: w_dependent,
          }
        };
        if let Some(inv) = inverse_fourier_inner(&w_part, w, t) {
          constants.push(inv);
          return Some(make_times(constants));
        }
      }
    }
  }

  None
}

/// Extract the value from a DSolve-like result: {{y[x] -> value}} -> value
fn extract_value_from_solve_result(expr: &Expr) -> Option<Expr> {
  if let Expr::List(outer) = expr
    && !outer.is_empty()
    && let Expr::List(inner) = &outer[0]
    && inner.len() == 1
  {
    match &inner[0] {
      Expr::Rule { replacement, .. } => {
        return Some(replacement.as_ref().clone());
      }
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        return Some(args[1].clone());
      }
      _ => {}
    }
  }
  None
}

// ── FunctionDomain implementation ────────────────────────────────────

/// FunctionDomain[f, x] or FunctionDomain[f, x, domain]
/// Finds the domain of f as a function of x.
fn function_domain_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let f = &args[0];
  let var = match &args[1] {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FunctionDomain".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Collect constraints on var for f to be defined
  let mut constraints = Vec::new();
  collect_domain_constraints(f, var, &mut constraints);

  if constraints.is_empty() {
    // No restrictions → domain is all reals
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Simplify: combine constraints with And
  let result = if constraints.len() == 1 {
    constraints.pop().unwrap()
  } else {
    // Join with &&
    let mut combined = constraints[0].clone();
    for c in &constraints[1..] {
      combined = Expr::FunctionCall {
        name: "And".to_string(),
        args: vec![combined, c.clone()],
      };
    }
    combined
  };

  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Collect constraints that must hold for expr to be defined.
fn collect_domain_constraints(
  expr: &Expr,
  var: &str,
  constraints: &mut Vec<Expr>,
) {
  match expr {
    // Division: denominator != 0
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      collect_domain_constraints(left, var, constraints);
      collect_domain_constraints(right, var, constraints);
      // Add constraint: right != 0
      if contains_variable(right, var) {
        constraints.push(Expr::Comparison {
          operands: vec![right.as_ref().clone(), Expr::Integer(0)],
          operators: vec![crate::syntax::ComparisonOp::NotEqual],
        });
      }
    }
    // Power with negative exponent: base != 0
    // Power with fractional exponent (sqrt): base >= 0
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      collect_domain_constraints(left, var, constraints);
      collect_domain_constraints(right, var, constraints);

      if contains_variable(left, var) {
        // Check for x^(-n) → x != 0
        let is_negative_power = match right.as_ref() {
          Expr::Integer(n) => *n < 0,
          Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            ..
          } => true,
          _ => false,
        };
        if is_negative_power {
          constraints.push(Expr::Comparison {
            operands: vec![left.as_ref().clone(), Expr::Integer(0)],
            operators: vec![crate::syntax::ComparisonOp::NotEqual],
          });
        }

        // Check for x^(1/2) i.e. Sqrt → x >= 0
        let is_half_power = match right.as_ref() {
          Expr::FunctionCall { name, args }
            if name == "Rational" && args.len() == 2 =>
          {
            matches!((&args[0], &args[1]), (Expr::Integer(1), Expr::Integer(2)))
          }
          _ => false,
        };
        if is_half_power {
          constraints.push(Expr::Comparison {
            operands: vec![left.as_ref().clone(), Expr::Integer(0)],
            operators: vec![crate::syntax::ComparisonOp::GreaterEqual],
          });
        }
      }
    }
    // Log[x] → x > 0
    Expr::FunctionCall { name, args }
      if (name == "Log" || name == "Log2" || name == "Log10")
        && args.len() >= 1 =>
    {
      let arg = args.last().unwrap();
      for a in args {
        collect_domain_constraints(a, var, constraints);
      }
      if contains_variable(arg, var) {
        constraints.push(Expr::Comparison {
          operands: vec![arg.clone(), Expr::Integer(0)],
          operators: vec![crate::syntax::ComparisonOp::Greater],
        });
      }
    }
    // Sqrt[x] → x >= 0
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      collect_domain_constraints(&args[0], var, constraints);
      if contains_variable(&args[0], var) {
        constraints.push(Expr::Comparison {
          operands: vec![args[0].clone(), Expr::Integer(0)],
          operators: vec![crate::syntax::ComparisonOp::GreaterEqual],
        });
      }
    }
    // Recurse into other structures
    Expr::BinaryOp { left, right, .. } => {
      collect_domain_constraints(left, var, constraints);
      collect_domain_constraints(right, var, constraints);
    }
    Expr::UnaryOp { operand, .. } => {
      collect_domain_constraints(operand, var, constraints);
    }
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_domain_constraints(a, var, constraints);
      }
    }
    _ => {}
  }
}

fn contains_variable(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(name) => name == var,
    Expr::Integer(_) | Expr::Real(_) | Expr::String(_) => false,
    Expr::BinaryOp { left, right, .. } => {
      contains_variable(left, var) || contains_variable(right, var)
    }
    Expr::UnaryOp { operand, .. } => contains_variable(operand, var),
    Expr::FunctionCall { name, args } => {
      if name == "Rational" {
        return false;
      }
      args.iter().any(|a| contains_variable(a, var))
    }
    Expr::List(items) => items.iter().any(|a| contains_variable(a, var)),
    _ => false,
  }
}
