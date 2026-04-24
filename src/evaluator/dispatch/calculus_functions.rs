#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::is_sqrt;

/// Check if the result of differentiation contains a
/// `Derivative[...][func_name][...]` pattern (as CurriedCall),
/// indicating the derivative couldn't be fully resolved.
fn contains_unresolved_derivative(expr: &Expr, func_name: &str) -> bool {
  match expr {
    // Match CurriedCall { func: CurriedCall { func: FunctionCall("Derivative", _), args: [Identifier(func_name)] }, args: _ }
    Expr::CurriedCall { func, args } => {
      let is_derivative_of_func = matches!(
        func.as_ref(),
        Expr::CurriedCall {
          func: inner_func,
          args: inner_args,
        } if matches!(inner_func.as_ref(), Expr::FunctionCall { name, .. } if name == "Derivative")
          && inner_args.iter().any(|a| matches!(a, Expr::Identifier(id) if id == func_name))
      );
      is_derivative_of_func
        || contains_unresolved_derivative(func, func_name)
        || args
          .iter()
          .any(|a| contains_unresolved_derivative(a, func_name))
    }
    Expr::FunctionCall { args, .. } | Expr::List(args) => args
      .iter()
      .any(|a| contains_unresolved_derivative(a, func_name)),
    Expr::BinaryOp { left, right, .. } => {
      contains_unresolved_derivative(left, func_name)
        || contains_unresolved_derivative(right, func_name)
    }
    Expr::UnaryOp { operand, .. } => {
      contains_unresolved_derivative(operand, func_name)
    }
    _ => false,
  }
}

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
      // For built-in functions (not in FUNC_DEFS), try symbolic differentiation
      if let (Expr::Integer(n), Expr::Identifier(func_name)) =
        (&args[0], &args[1])
      {
        let n = *n as usize;
        let dummy_var = "__derivative_var__".to_string();
        let dummy_ident = Expr::Identifier(dummy_var.clone());
        let mut deriv = Expr::FunctionCall {
          name: func_name.clone(),
          args: vec![dummy_ident.clone()],
        };
        let mut resolved = true;
        for _ in 0..n {
          deriv = match crate::functions::calculus_ast::differentiate_expr(
            &deriv, &dummy_var,
          ) {
            Ok(v) => v,
            Err(_) => {
              resolved = false;
              break;
            }
          };
          // If differentiation returned a Derivative[...][func][var] form,
          // no actual symbolic derivative was computed (e.g. unknown function).
          // In that case, return unevaluated to avoid corrupting DSolve inputs.
          if contains_unresolved_derivative(&deriv, func_name) {
            resolved = false;
            break;
          }
        }
        if resolved {
          let substituted =
            crate::syntax::substitute_variable(&deriv, &dummy_var, &args[2]);
          return Some(evaluate_expr_to_expr(&substituted));
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      }));
    }
    "Derivative" if args.len() == 2 => {
      // Derivative[n][f] → compute the nth derivative symbolically
      // and return as a pure function body&
      if let Expr::Integer(n) = &args[0] {
        let n = *n as usize;
        if let Expr::Identifier(func_name) = &args[1] {
          // Use a dummy variable, differentiate, then replace with Slot[1]
          let dummy = "__d_slot__";
          let dummy_expr = Expr::Identifier(dummy.to_string());
          let mut deriv = Expr::FunctionCall {
            name: func_name.clone(),
            args: vec![dummy_expr.clone()],
          };
          let mut resolved = true;
          for _ in 0..n {
            deriv = match crate::functions::calculus_ast::differentiate_expr(
              &deriv, dummy,
            ) {
              Ok(v) => v,
              Err(_) => {
                resolved = false;
                break;
              }
            };
            if contains_unresolved_derivative(&deriv, func_name) {
              resolved = false;
              break;
            }
          }
          if resolved {
            let simplified = crate::functions::calculus_ast::simplify(deriv);
            // Replace dummy variable with Slot(1)
            let with_slot = crate::syntax::substitute_variable(
              &simplified,
              dummy,
              &Expr::Slot(1),
            );
            return Some(Ok(Expr::Function {
              body: Box::new(with_slot),
            }));
          }
        }
        // Handle pure function: Derivative[n][body&]
        if let Expr::Function { body } = &args[1] {
          let dummy = "__d_slot__";
          // Replace Slot(1) with dummy variable for differentiation
          let with_dummy = crate::syntax::substitute_slots(
            body,
            &[Expr::Identifier(dummy.to_string())],
          );
          let mut deriv = with_dummy;
          let mut resolved = true;
          for _ in 0..n {
            deriv = match crate::functions::calculus_ast::differentiate_expr(
              &deriv, dummy,
            ) {
              Ok(v) => v,
              Err(_) => {
                resolved = false;
                break;
              }
            };
          }
          if resolved {
            let simplified = crate::functions::calculus_ast::simplify(deriv);
            let with_slot = crate::syntax::substitute_variable(
              &simplified,
              dummy,
              &Expr::Slot(1),
            );
            return Some(Ok(Expr::Function {
              body: Box::new(with_slot),
            }));
          }
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      }));
    }
    "Derivative" if args.len() == 1 => {
      return Some(Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      }));
    }
    "D" if args.len() >= 2 => {
      return Some(crate::functions::calculus_ast::d_ast(args));
    }
    "Dt" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::dt_ast(args));
    }
    "Curl" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::curl_ast(args));
    }
    "FrenetSerretSystem" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::frenet_serret_system_ast(
        args,
      ));
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
    "Integrate" if args.len() >= 2 => {
      return Some(crate::functions::calculus_ast::integrate_ast(args));
    }
    "NIntegrate" if args.len() >= 2 => {
      return Some(crate::functions::calculus_ast::nintegrate_ast(args));
    }
    "Limit" if (2..=3).contains(&args.len()) => {
      return Some(crate::functions::calculus_ast::limit_ast(args));
    }
    // DiscreteLimit[expr, n -> val] / DiscreteLimit[expr, {m -> ..., n -> ...}]
    // For smooth rational sequences this matches the continuous Limit;
    // delegate to Limit, and for a list of substitutions apply them
    // outside-in (matches wolframscript's iterated-limit semantics for
    // simple rational expressions).
    "DiscreteLimit" if args.len() == 2 => {
      let inner = &args[1];
      let subs: Vec<Expr> = match inner {
        Expr::List(items) => items.clone(),
        _ => vec![inner.clone()],
      };
      let mut current = args[0].clone();
      let mut all_resolved = true;
      for sub in subs.into_iter().rev() {
        let next = match crate::functions::calculus_ast::limit_ast(&[
          current.clone(),
          sub.clone(),
        ]) {
          Ok(v) => v,
          Err(e) => return Some(Err(e)),
        };
        // If Limit didn't simplify (still a Limit[...] wrapper), keep the
        // original DiscreteLimit unevaluated rather than replacing it with
        // a spurious Limit[] form.
        if matches!(&next, Expr::FunctionCall { name, .. } if name == "Limit") {
          all_resolved = false;
          break;
        }
        current = next;
      }
      if all_resolved {
        return Some(Ok(current));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "DiscreteLimit".to_string(),
        args: args.to_vec(),
      }));
    }
    "MaxLimit" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::max_limit_ast(args));
    }
    "MinLimit" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::min_limit_ast(args));
    }
    "Series" if args.len() >= 2 => {
      return Some(crate::functions::calculus_ast::series_ast(args));
    }
    "AsymptoticSolve" if args.len() == 3 => {
      return Some(crate::functions::calculus_ast::asymptotic_solve_ast(args));
    }
    "DiscreteConvolve" if args.len() == 4 => {
      return Some(crate::functions::calculus_ast::discrete_convolve_ast(args));
    }
    "SeriesCoefficient" if args.len() == 2 => {
      // SeriesCoefficient[SeriesData[x, x0, coeffs, nmin, nmax, den], q]
      // — query a SeriesData directly. `q` may be an Integer or a
      // `Rational[p, r]`; the target index into `coeffs` is
      // `q * den - nmin`. If that isn't an integer (q's numerator·den
      // isn't divisible by q's denominator) the series has no term at
      // that exponent → 0. If the index falls below nmin the series
      // starts later → 0. If the index is at/beyond the tracked range
      // (`coeffs.len()`) we don't know the coefficient → Indeterminate.
      if let Expr::FunctionCall {
        name: sd_name,
        args: sd_args,
      } = &args[0]
        && sd_name == "SeriesData"
        && sd_args.len() >= 6
        && let Expr::List(coeffs) = &sd_args[2]
        && let Some(nmin) = crate::functions::math_ast::expr_to_i128(&sd_args[3])
        && let Some(den) = crate::functions::math_ast::expr_to_i128(&sd_args[5])
        && den > 0
      {
        // Normalise q to (num, denom) with denom > 0.
        let (q_num, q_den): (i128, i128) = match &args[1] {
          Expr::Integer(n) => (*n, 1),
          Expr::FunctionCall { name: rn, args: ra }
            if rn == "Rational" && ra.len() == 2 =>
          {
            match (&ra[0], &ra[1]) {
              (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => (*n, *d),
              _ => return None,
            }
          }
          _ => return None,
        };
        // target_n = q * den = (q_num * den) / q_den must be integer.
        let scaled = q_num * den;
        if scaled % q_den != 0 {
          return Some(Ok(Expr::Integer(0)));
        }
        let target_n = scaled / q_den;
        let idx = target_n - nmin;
        if idx < 0 {
          return Some(Ok(Expr::Integer(0)));
        }
        let idx_u = idx as usize;
        if idx_u < coeffs.len() {
          return Some(Ok(coeffs[idx_u].clone()));
        } else {
          return Some(Ok(Expr::Identifier("Indeterminate".to_string())));
        }
      }
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
    "FourierSinTransform" if args.len() == 3 => {
      return Some(fourier_sin_transform(&args[0], &args[1], &args[2]));
    }
    "FourierCosTransform" if args.len() == 3 => {
      return Some(fourier_cos_transform(&args[0], &args[1], &args[2]));
    }
    "InverseFourierTransform" if args.len() == 3 => {
      return Some(inverse_fourier_transform(&args[0], &args[1], &args[2]));
    }
    "FunctionDomain" if args.len() >= 2 && args.len() <= 3 => {
      return Some(function_domain_ast(args));
    }
    "AsymptoticIntegrate" if args.len() == 3 => {
      return Some(crate::functions::calculus_ast::asymptotic_integrate_ast(
        args,
      ));
    }
    "ArcCurvature" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::arc_curvature_ast(args));
    }
    "DifferenceDelta" if !args.is_empty() => {
      return Some(crate::functions::calculus_ast::difference_delta_ast(args));
    }
    "DifferenceQuotient" if !args.is_empty() => {
      return Some(crate::functions::calculus_ast::difference_quotient_ast(
        args,
      ));
    }
    "GeneratingFunction" if args.len() == 3 => {
      return Some(generating_function(&args[0], &args[1], &args[2]));
    }
    "ExponentialGeneratingFunction" if args.len() == 3 => {
      return Some(exponential_generating_function(
        &args[0], &args[1], &args[2],
      ));
    }
    // InverseFunction[f] — returns the inverse of known functions
    "InverseFunction" if args.len() == 1 => {
      if let Expr::Identifier(func_name) = &args[0] {
        let inverse = match func_name.as_str() {
          // Trig -> ArcTrig
          "Sin" => Some("ArcSin"),
          "Cos" => Some("ArcCos"),
          "Tan" => Some("ArcTan"),
          "Cot" => Some("ArcCot"),
          "Sec" => Some("ArcSec"),
          "Csc" => Some("ArcCsc"),
          // Hyperbolic -> ArcHyperbolic
          "Sinh" => Some("ArcSinh"),
          "Cosh" => Some("ArcCosh"),
          "Tanh" => Some("ArcTanh"),
          "Coth" => Some("ArcCoth"),
          "Sech" => Some("ArcSech"),
          "Csch" => Some("ArcCsch"),
          // ArcTrig -> Trig
          "ArcSin" => Some("Sin"),
          "ArcCos" => Some("Cos"),
          "ArcTan" => Some("Tan"),
          "ArcCot" => Some("Cot"),
          "ArcSec" => Some("Sec"),
          "ArcCsc" => Some("Csc"),
          // ArcHyperbolic -> Hyperbolic
          "ArcSinh" => Some("Sinh"),
          "ArcCosh" => Some("Cosh"),
          "ArcTanh" => Some("Tanh"),
          "ArcCoth" => Some("Coth"),
          "ArcSech" => Some("Sech"),
          "ArcCsch" => Some("Csch"),
          // Exp <-> Log
          "Exp" => Some("Log"),
          "Log" => Some("Exp"),
          _ => None,
        };
        if let Some(inv_name) = inverse {
          return Some(Ok(Expr::Identifier(inv_name.to_string())));
        }
      }
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

/// Collect like terms that share the same DiracDelta and structural factors.
/// E.g. I*Sqrt[Pi/2]*DiracDelta[a] + Sqrt[Pi/2]*DiracDelta[a]
///   → (1 + I)*Sqrt[Pi/2]*DiracDelta[a]
fn collect_dirac_like_terms(terms: Vec<Expr>) -> Vec<Expr> {
  use std::collections::BTreeMap;

  /// Check if a factor is a "scalar" coefficient (numeric or I)
  fn is_scalar(f: &Expr) -> bool {
    match f {
      Expr::Integer(_) | Expr::Real(_) => true,
      Expr::Constant(s) if s == "I" => true,
      Expr::FunctionCall { name, .. } if name == "Rational" => true,
      _ => false,
    }
  }

  /// Extract from a term: (base_key, scalar_factors, base_factors_with_dirac)
  /// base = structural factors + DiracDelta, scalar = numeric/I coefficients
  fn extract_base_and_scalar(
    term: &Expr,
  ) -> Option<(String, Vec<Expr>, Vec<Expr>)> {
    let factors: Vec<Expr> = match term {
      Expr::FunctionCall { name, args } if name == "Times" => {
        // Flatten nested Times
        let mut flat = Vec::new();
        for a in args {
          if let Expr::FunctionCall {
            name: n,
            args: inner,
          } = a
            && n == "Times"
          {
            flat.extend(inner.iter().cloned());
            continue;
          }
          flat.push(a.clone());
        }
        flat
      }
      Expr::FunctionCall { name, .. } if name == "DiracDelta" => {
        vec![term.clone()]
      }
      _ => return None,
    };

    // Must contain DiracDelta
    if !factors.iter().any(
      |f| matches!(f, Expr::FunctionCall { name, .. } if name == "DiracDelta"),
    ) {
      return None;
    }

    let mut scalar_parts: Vec<Expr> = Vec::new();
    let mut base_parts: Vec<Expr> = Vec::new();

    for f in factors {
      if is_scalar(&f) {
        scalar_parts.push(f);
      } else {
        base_parts.push(f);
      }
    }

    let base_key = base_parts
      .iter()
      .map(crate::syntax::expr_to_string)
      .collect::<Vec<_>>()
      .join("*");

    Some((base_key, scalar_parts, base_parts))
  }

  // Group by base (structural + DiracDelta)
  let mut groups: BTreeMap<String, (Vec<Expr>, Vec<Vec<Expr>>)> =
    BTreeMap::new();
  let mut non_dirac: Vec<Expr> = Vec::new();

  for term in terms {
    if let Some((key, scalar, base)) = extract_base_and_scalar(&term) {
      groups
        .entry(key)
        .or_insert_with(|| (base, Vec::new()))
        .1
        .push(scalar);
    } else {
      non_dirac.push(term);
    }
  }

  let mut result = non_dirac;
  for (_, (base_factors, scalar_groups)) in groups {
    if scalar_groups.len() == 1 {
      // Only one term - reconstruct as-is
      let mut factors = scalar_groups.into_iter().next().unwrap();
      factors.extend(base_factors);
      result.push(if factors.len() == 1 {
        factors.into_iter().next().unwrap()
      } else {
        make_times(factors)
      });
    } else {
      // Multiple terms - sum the scalar parts
      let scalar_terms: Vec<Expr> = scalar_groups
        .into_iter()
        .map(|factors| {
          if factors.is_empty() {
            Expr::Integer(1)
          } else if factors.len() == 1 {
            factors.into_iter().next().unwrap()
          } else {
            make_times(factors)
          }
        })
        .collect();
      let scalar_sum = make_plus(scalar_terms);
      let scalar_eval = crate::evaluator::evaluate_expr_to_expr(&scalar_sum)
        .unwrap_or(scalar_sum);
      let mut all_factors = vec![scalar_eval];
      all_factors.extend(base_factors);
      result.push(make_times(all_factors));
    }
  }

  result
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
    // F[1/t] = (I*Pi*Sign[w])/Sqrt[2*Pi]
    if fname == "Power"
      && fargs.len() == 2
      && let Expr::Identifier(base) = fargs[0]
      && base == t
      && matches!(fargs[1], Expr::Integer(-1))
    {
      return Some(make_times(vec![
        Expr::Constant("I".to_string()),
        Expr::Constant("Pi".to_string()),
        Expr::FunctionCall {
          name: "Sign".to_string(),
          args: vec![w.clone()],
        },
        make_power(
          make_times(vec![Expr::Integer(2), Expr::Constant("Pi".to_string())]),
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-1), Expr::Integer(2)],
          },
        ),
      ]));
    }

    // Linearity: F[a + b] = F[a] + F[b]
    if fname == "Plus" {
      let mut terms = Vec::new();
      for arg in &fargs {
        if let Some(ft) = fourier_transform_inner(arg, t, w) {
          // Flatten nested Plus results
          match ft {
            Expr::FunctionCall { ref name, ref args } if name == "Plus" => {
              terms.extend(args.iter().cloned());
            }
            _ => terms.push(ft),
          }
        } else {
          return None;
        }
      }
      // Collect like terms that share the same DiracDelta factor
      let terms = collect_dirac_like_terms(terms);
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
  if let Some((fname, fargs)) = as_func_args(exp)
    && fname == "Times"
  {
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
  None
}

/// Match exponent of form -a*Abs[t] and return a.
fn match_neg_a_abs_t(exp: &Expr, t: &str) -> Option<Expr> {
  if let Some((fname, fargs)) = as_func_args(exp)
    && fname == "Times"
  {
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
      if is_e && let Some(a) = match_neg_a_t_squared(fargs[1], w) {
        // F^-1[E^(-a*w^2)] = E^(-t^2/(4a)) / Sqrt[2a]
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

  // Simplify each constraint and convert to interval representation
  let mut simplified: Vec<Expr> = Vec::new();
  for c in &constraints {
    let s = simplify_domain_constraint(c, var);
    simplified.push(s);
  }

  // Combine constraints with And
  let result = if simplified.len() == 1 {
    simplified.pop().unwrap()
  } else {
    let mut combined = simplified[0].clone();
    for c in &simplified[1..] {
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
        && !args.is_empty() =>
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
    expr if is_sqrt(expr).is_some() => {
      let sqrt_arg = is_sqrt(expr).unwrap();
      collect_domain_constraints(sqrt_arg, var, constraints);
      if contains_variable(sqrt_arg, var) {
        constraints.push(Expr::Comparison {
          operands: vec![sqrt_arg.clone(), Expr::Integer(0)],
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

/// Simplify a domain constraint for display, matching Wolfram behavior.
/// - `x != 0` → `x < 0 || x > 0`
/// - `-1 + x >= 0` → `x >= 1`
/// - `x^2 - 1 != 0` → `x < -1 || Inequality[-1, Less, x, Less, 1] || x > 1`
fn simplify_domain_constraint(constraint: &Expr, var: &str) -> Expr {
  use crate::syntax::ComparisonOp;

  if let Expr::Comparison {
    operands,
    operators,
  } = constraint
    && operands.len() == 2
    && operators.len() == 1
  {
    let lhs = &operands[0];
    let rhs = &operands[1];
    let op = &operators[0];

    match op {
      ComparisonOp::NotEqual => {
        // Try to solve lhs == rhs for roots
        let diff = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Minus,
          left: Box::new(lhs.clone()),
          right: Box::new(rhs.clone()),
        };
        let diff_eval =
          crate::evaluator::evaluate_expr_to_expr(&diff).unwrap_or(diff);

        // Use Solve to find roots
        let eq = Expr::Comparison {
          operands: vec![diff_eval, Expr::Integer(0)],
          operators: vec![ComparisonOp::Equal],
        };
        let var_expr = Expr::Identifier(var.to_string());
        crate::push_quiet();
        let solve_result =
          crate::functions::polynomial_ast::solve::solve_ast(&[
            eq,
            var_expr.clone(),
          ]);
        crate::pop_quiet();

        if let Ok(Expr::List(ref solutions)) = solve_result {
          // Extract numeric root values
          let mut roots: Vec<f64> = Vec::new();
          let mut root_exprs: Vec<Expr> = Vec::new();
          for sol in solutions {
            if let Expr::List(rules) = sol
              && rules.len() == 1
            {
              if let Expr::Rule { replacement, .. } = &rules[0] {
                if let Some(val) = expr_to_f64(replacement) {
                  roots.push(val);
                  root_exprs.push(replacement.as_ref().clone());
                }
              } else if let Expr::FunctionCall { name, args } = &rules[0]
                && name == "Rule"
                && args.len() == 2
                && let Some(val) = expr_to_f64(&args[1])
              {
                roots.push(val);
                root_exprs.push(args[1].clone());
              }
            }
          }

          if !roots.is_empty() {
            // Sort roots
            let mut indexed: Vec<(f64, Expr)> =
              roots.into_iter().zip(root_exprs).collect();
            indexed.sort_by(|a, b| {
              a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Build interval complement: x < r1 || r1 < x < r2 || ... || x > rN
            let mut intervals: Vec<Expr> = Vec::new();

            // x < first root
            intervals.push(Expr::Comparison {
              operands: vec![var_expr.clone(), indexed[0].1.clone()],
              operators: vec![ComparisonOp::Less],
            });

            // Between consecutive roots: ri < x < r(i+1)
            for i in 0..indexed.len() - 1 {
              intervals.push(Expr::FunctionCall {
                name: "Inequality".to_string(),
                args: vec![
                  indexed[i].1.clone(),
                  Expr::Identifier("Less".to_string()),
                  var_expr.clone(),
                  Expr::Identifier("Less".to_string()),
                  indexed[i + 1].1.clone(),
                ],
              });
            }

            // x > last root
            intervals.push(Expr::Comparison {
              operands: vec![
                var_expr.clone(),
                indexed.last().unwrap().1.clone(),
              ],
              operators: vec![ComparisonOp::Greater],
            });

            if intervals.len() == 1 {
              return intervals.pop().unwrap();
            }
            return Expr::FunctionCall {
              name: "Or".to_string(),
              args: intervals,
            };
          }
        }

        // Fallback: simple x != value case
        if matches!(lhs, Expr::Identifier(name) if name == var) {
          return Expr::FunctionCall {
            name: "Or".to_string(),
            args: vec![
              Expr::Comparison {
                operands: vec![lhs.clone(), rhs.clone()],
                operators: vec![ComparisonOp::Less],
              },
              Expr::Comparison {
                operands: vec![lhs.clone(), rhs.clone()],
                operators: vec![ComparisonOp::Greater],
              },
            ],
          };
        }
      }
      ComparisonOp::GreaterEqual
      | ComparisonOp::Greater
      | ComparisonOp::LessEqual
      | ComparisonOp::Less
        // Try to solve for x: isolate x by solving lhs - rhs == 0
        // then adjust the inequality
        if contains_variable(lhs, var) && !contains_variable(rhs, var) => {
          // Try to solve lhs = value for x
          let eq = Expr::Comparison {
            operands: vec![lhs.clone(), rhs.clone()],
            operators: vec![ComparisonOp::Equal],
          };
          let var_expr = Expr::Identifier(var.to_string());
          crate::push_quiet();
          let solve_result =
            crate::functions::polynomial_ast::solve::solve_ast(&[
              eq,
              var_expr.clone(),
            ]);
          crate::pop_quiet();

          if let Ok(Expr::List(ref solutions)) = solve_result
            && solutions.len() == 1
            && let Expr::List(rules) = &solutions[0]
            && rules.len() == 1
          {
            let value = match &rules[0] {
              Expr::Rule { replacement, .. } => {
                Some(replacement.as_ref().clone())
              }
              Expr::FunctionCall { name, args }
                if name == "Rule" && args.len() == 2 =>
              {
                Some(args[1].clone())
              }
              _ => None,
            };
            if let Some(val) = value {
              // Check if the coefficient of x is positive (preserve direction)
              // For simple cases, just return x op val
              return Expr::Comparison {
                operands: vec![var_expr, val],
                operators: vec![*op],
              };
            }
          }
        }
      _ => {}
    }
  }

  // Fallback: return the constraint as-is, evaluated
  crate::evaluator::evaluate_expr_to_expr(constraint)
    .unwrap_or_else(|_| constraint.clone())
}

/// Try to convert an expression to f64 (for sorting roots)
fn expr_to_f64(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => expr_to_f64(operand).map(|v| -v),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(*n as f64 / *d as f64)
      } else {
        None
      }
    }
    _ => None,
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

/// GeneratingFunction[a_n, n, x] = Sum[a_n * x^n, {n, 0, Infinity}]
fn generating_function(
  expr: &Expr,
  n_expr: &Expr,
  x_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let n = match n_expr {
    Expr::Identifier(name) => name.as_str(),
    // Multivariate: GeneratingFunction[a, {n, m}, {x, y}]
    Expr::List(ns) => {
      if let Expr::List(xs) = x_expr
        && ns.len() == xs.len()
        && ns.len() >= 2
      {
        return generating_function_multivariate(expr, ns, xs);
      }
      return Ok(Expr::FunctionCall {
        name: "GeneratingFunction".to_string(),
        args: vec![expr.clone(), n_expr.clone(), x_expr.clone()],
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "GeneratingFunction".to_string(),
        args: vec![expr.clone(), n_expr.clone(), x_expr.clone()],
      });
    }
  };

  if let Some(result) = gf_inner(expr, n, x_expr)? {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(Expr::FunctionCall {
      name: "GeneratingFunction".to_string(),
      args: vec![expr.clone(), n_expr.clone(), x_expr.clone()],
    })
  }
}

/// Multivariate generating function: reduce from right to left
fn generating_function_multivariate(
  expr: &Expr,
  ns: &[Expr],
  xs: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Process innermost variable first (rightmost), then work outward
  let mut result = expr.clone();
  for i in (0..ns.len()).rev() {
    let n_var = &ns[i];
    let x_var = &xs[i];
    result = generating_function(&result, n_var, x_var)?;
    // If it came back unevaluated, we can't continue
    if let Expr::FunctionCall { ref name, .. } = result
      && name == "GeneratingFunction"
    {
      return Ok(Expr::FunctionCall {
        name: "GeneratingFunction".to_string(),
        args: vec![
          expr.clone(),
          Expr::List(ns.to_vec()),
          Expr::List(xs.to_vec()),
        ],
      });
    }
  }
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Core pattern matching for generating functions.
/// Returns Some(result) if a closed form is found, None otherwise.
fn gf_inner(
  expr: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Case 0: expr doesn't depend on n => constant * 1/(1-x)
  if !depends_on(expr, n) {
    // c/(1-x)
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(expr.clone()),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(x.clone()),
      }),
    }));
  }

  // Case 1: expr = n (just the variable)
  if matches!(expr, Expr::Identifier(name) if name == n) {
    // x/(-1+x)^2  (canonical Wolfram form; (1-x)^2 = (-1+x)^2 since power is even)
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(x.clone()),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(x.clone()),
        }),
        right: Box::new(Expr::Integer(2)),
      }),
    }));
  }

  // Handle Divide explicitly (before as_func_args normalizes it)
  // Also handle canonical Times[..., Power[..., -1]] and Power[..., -1] forms
  {
    let (num, den) =
      crate::functions::polynomial_ast::together::extract_num_den(expr);
    if !matches!(&den, Expr::Integer(1))
      && let Some(result) = gf_divide(&num, &den, n, x)?
    {
      return Ok(Some(result));
    }
  }

  // Handle Minus: a - b => a + (-b)
  if let Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left,
    right,
  } = expr
  {
    let neg_right = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: right.clone(),
    };
    let as_plus = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: left.clone(),
      right: Box::new(neg_right),
    };
    return gf_inner(&as_plus, n, x);
  }

  // Check for function call / binary op patterns
  if let Some((fname, fargs)) = as_func_args(expr) {
    match fname {
      "Power" if fargs.len() == 2 => {
        return gf_power(fargs[0], fargs[1], n, x);
      }
      "Plus" => {
        return gf_plus(expr, n, x);
      }
      "Times" => {
        return gf_times(expr, n, x);
      }
      "Factorial" if fargs.len() == 1 => {
        // 1/n! case is handled in gf_times via Power[Factorial[n], -1]
      }
      "Binomial" if fargs.len() == 2 => {
        return gf_binomial(fargs[0], fargs[1], n, x);
      }
      _ => {}
    }
  }

  // Case: f[n+k] — shifted sequence
  if let Expr::FunctionCall {
    name: fname,
    args: fargs,
  } = expr
    && fargs.len() == 1
    && let Some((shift, inner_var)) = extract_shift(&fargs[0], n)
    && shift > 0
  {
    // GeneratingFunction[f[n+k], n, x] = (1/x^k) * (GF[f[n],n,x] - Sum[f[i]*x^i, {i,0,k-1}])
    let base_expr = Expr::FunctionCall {
      name: fname.clone(),
      args: vec![Expr::Identifier(inner_var.to_string())],
    };
    let gf_base = Expr::FunctionCall {
      name: "GeneratingFunction".to_string(),
      args: vec![base_expr, Expr::Identifier(n.to_string()), x.clone()],
    };
    // Subtract the first k terms
    let mut subtract_terms = Vec::new();
    for i in 0..shift {
      let fi = Expr::FunctionCall {
        name: fname.clone(),
        args: vec![Expr::Integer(i as i128)],
      };
      let term = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(fi),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(x.clone()),
          right: Box::new(Expr::Integer(i as i128)),
        }),
      };
      subtract_terms.push(term);
    }
    let subtracted =
      subtract_terms.into_iter().reduce(|acc, t| Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(acc),
        right: Box::new(t),
      });
    let numerator = if let Some(sub) = subtracted {
      Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(gf_base),
        right: Box::new(sub),
      }
    } else {
      gf_base
    };
    let result = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(shift as i128)),
      }),
    };
    return Ok(Some(result));
  }

  Ok(None)
}

/// Extract shift from expression like n+1, n+2, etc.
/// Returns (shift, variable_name) if the expression is var + constant.
fn extract_shift<'a>(expr: &'a Expr, var: &str) -> Option<(i64, &'a str)> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      if let Expr::Identifier(name) = left.as_ref()
        && name == var
        && let Expr::Integer(k) = right.as_ref()
      {
        return Some((*k as i64, name.as_str()));
      }
      if let Expr::Identifier(name) = right.as_ref()
        && name == var
        && let Expr::Integer(k) = left.as_ref()
      {
        return Some((*k as i64, name.as_str()));
      }
      None
    }
    // Handle FunctionCall Plus[k, n] form
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      if let Expr::Identifier(id) = &args[0]
        && id == var
        && let Expr::Integer(k) = &args[1]
      {
        return Some((*k as i64, id.as_str()));
      }
      if let Expr::Identifier(id) = &args[1]
        && id == var
        && let Expr::Integer(k) = &args[0]
      {
        return Some((*k as i64, id.as_str()));
      }
      None
    }
    _ => None,
  }
}

/// Handle Power[base, exp] in generating function context
fn gf_power(
  base: &Expr,
  exp: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Case: a^n where a doesn't depend on n => 1/(1 - a*x)
  if matches!(exp, Expr::Identifier(name) if name == n) && !depends_on(base, n)
  {
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(base.clone()),
          right: Box::new(x.clone()),
        }),
      }),
      right: Box::new(Expr::Integer(-1)),
    }));
  }

  // Case: n^k where k is a positive integer => Eulerian number formula
  if matches!(base, Expr::Identifier(name) if name == n)
    && !depends_on(exp, n)
    && let Expr::Integer(k) = exp
  {
    let k = *k;
    if k >= 2 {
      return gf_n_power_k(k, x);
    }
  }

  // Case: Power[Factorial[n], -1] => 1/n! => E^x
  if let Expr::Integer(-1) = exp
    && let Expr::FunctionCall { name, args } = base
    && name == "Factorial"
    && args.len() == 1
    && matches!(&args[0], Expr::Identifier(var) if var == n)
  {
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Constant("E".to_string())),
      right: Box::new(x.clone()),
    }));
  }

  // Case: Power[Factorial[n], -2] => 1/(n!)^2 => BesselI[0, 2*Sqrt[x]]
  if let Expr::Integer(-2) = exp
    && let Expr::FunctionCall { name, args } = base
    && name == "Factorial"
    && args.len() == 1
    && matches!(&args[0], Expr::Identifier(var) if var == n)
  {
    return Ok(Some(Expr::FunctionCall {
      name: "BesselI".to_string(),
      args: vec![
        Expr::Integer(0),
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(crate::functions::math_ast::make_sqrt(x.clone())),
        },
      ],
    }));
  }

  Ok(None)
}

/// Generating function for n^k: Sum[n^k * x^n, {n, 0, inf}]
/// Uses the formula involving Eulerian numbers: result = Sum[A(k,j) * x^(j+1), j=0..k-1] / (1-x)^(k+1)
/// where A(k,j) are the Eulerian numbers.
fn gf_n_power_k(k: i128, x: &Expr) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Compute Eulerian numbers A(k, j) for j = 0..k-1
  let k_usize = k as usize;
  let eulerian = compute_eulerian_numbers(k_usize);

  // Build numerator polynomial: Sum[A(k,j) * x^(j+1), j=0..k-1]
  // Using (1-x)^(k+1) as denominator for consistency with GF[n] = x/(1-x)^2
  let sign = 1i128;
  let mut num_terms: Vec<Expr> = Vec::new();
  for (j, &coeff) in eulerian.iter().enumerate() {
    let signed_coeff = sign * coeff;
    if signed_coeff == 0 {
      continue;
    }
    let power = (j + 1) as i128;
    let x_pow = if power == 1 {
      x.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(power)),
      }
    };
    if signed_coeff == 1 {
      num_terms.push(x_pow);
    } else if signed_coeff == -1 {
      num_terms.push(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(x_pow),
      });
    } else {
      num_terms.push(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(signed_coeff)),
        right: Box::new(x_pow),
      });
    }
  }

  let numerator = if num_terms.len() == 1 {
    num_terms.into_iter().next().unwrap()
  } else {
    num_terms
      .into_iter()
      .reduce(|acc, t| Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(acc),
        right: Box::new(t),
      })
      .unwrap()
  };

  // Denominator: (-1+x)^(k+1) — canonical Wolfram form.
  // Since (1-x) = -(-1+x), we have (1-x)^(k+1) = (-1)^(k+1) * (-1+x)^(k+1).
  // When (k+1) is odd, the numerator must be negated to compensate.
  let denominator = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(x.clone()),
    }),
    right: Box::new(Expr::Integer(k + 1)),
  };

  let final_numerator = if (k + 1) % 2 != 0 {
    // Odd power: negate numerator
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(numerator),
    }
  } else {
    numerator
  };

  Ok(Some(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(final_numerator),
    right: Box::new(denominator),
  }))
}

/// Compute Eulerian numbers A(k, j) for j = 0..k-1
fn compute_eulerian_numbers(k: usize) -> Vec<i128> {
  // A(k, j) = sum_{i=0}^{j} (-1)^i * C(k+1, i) * (j+1-i)^k
  let mut result = Vec::with_capacity(k);
  for j in 0..k {
    let mut val: i128 = 0;
    for i in 0..=j {
      let sign = if i % 2 == 0 { 1i128 } else { -1 };
      let binom = binomial_coeff(k as i128 + 1, i as i128);
      let base = (j + 1 - i) as i128;
      let power = base.pow(k as u32);
      val += sign * binom * power;
    }
    result.push(val);
  }
  result
}

/// Compute binomial coefficient C(n, k)
fn binomial_coeff(n: i128, k: i128) -> i128 {
  if k < 0 || k > n {
    return 0;
  }
  if k == 0 || k == n {
    return 1;
  }
  let k = k.min(n - k);
  let mut result: i128 = 1;
  for i in 0..k {
    result = result * (n - i) / (i + 1);
  }
  result
}

/// Handle Plus (sum) in generating function — linearity
fn gf_plus(
  expr: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Collect all terms
  let terms = collect_plus_terms(expr);

  let mut result_terms = Vec::new();
  for term in &terms {
    if let Some(gf) = gf_inner(term, n, x)? {
      result_terms.push(gf);
    } else {
      return Ok(None); // Can't evaluate one term, give up
    }
  }

  let result = result_terms
    .into_iter()
    .reduce(|acc, t| Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(acc),
      right: Box::new(t),
    })
    .unwrap();

  Ok(Some(result))
}

/// Collect all additive terms from a Plus expression tree
fn collect_plus_terms(expr: &Expr) -> Vec<&Expr> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut terms = collect_plus_terms(left);
      terms.extend(collect_plus_terms(right));
      terms
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().collect()
    }
    _ => vec![expr],
  }
}

/// Handle Times (product) in generating function
fn gf_times(
  expr: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Collect all multiplicative factors
  let factors = collect_times_factors(expr);

  // Separate factors into n-dependent and constant
  let mut constants: Vec<&Expr> = Vec::new();
  let mut n_dependent: Vec<&Expr> = Vec::new();

  for factor in &factors {
    if depends_on(factor, n) {
      n_dependent.push(factor);
    } else {
      constants.push(factor);
    }
  }

  // If there's a constant factor, factor it out
  if !constants.is_empty() && !n_dependent.is_empty() {
    let const_product = if constants.len() == 1 {
      (*constants[0]).clone()
    } else {
      constants
        .iter()
        .cloned()
        .cloned()
        .reduce(|acc, t| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(acc),
          right: Box::new(t),
        })
        .unwrap()
    };

    let n_product = if n_dependent.len() == 1 {
      (*n_dependent[0]).clone()
    } else {
      n_dependent
        .iter()
        .cloned()
        .cloned()
        .reduce(|acc, t| Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(acc),
          right: Box::new(t),
        })
        .unwrap()
    };

    if let Some(inner_gf) = gf_inner(&n_product, n, x)? {
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(const_product),
        right: Box::new(inner_gf),
      }));
    }
  }

  // All factors depend on n: recombine and try as a single expression
  // Handle common combined patterns:

  // c * a^n pattern (all together)
  let recombined = if n_dependent.len() >= 2 {
    n_dependent
      .iter()
      .cloned()
      .cloned()
      .reduce(|acc, t| Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(acc),
        right: Box::new(t),
      })
      .unwrap()
  } else if n_dependent.len() == 1 {
    (*n_dependent[0]).clone()
  } else {
    return Ok(None);
  };

  // Try specific combined patterns like (-1)^n which is Power[-1, n]
  if let Some((fname, fargs)) = as_func_args(&recombined)
    && fname == "Power"
    && fargs.len() == 2
  {
    return gf_power(fargs[0], fargs[1], n, x);
  }

  // Try expanding the product and handling as a sum
  // e.g. n*(-1+n) → -n + n^2
  let expanded = crate::functions::polynomial_ast::expand_expr(&recombined);
  if crate::syntax::expr_to_string(&expanded)
    != crate::syntax::expr_to_string(&recombined)
  {
    return gf_inner(&expanded, n, x);
  }

  Ok(None)
}

/// Collect all multiplicative factors from a Times expression tree
fn collect_times_factors(expr: &Expr) -> Vec<&Expr> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let mut factors = collect_times_factors(left);
      factors.extend(collect_times_factors(right));
      factors
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().collect()
    }
    _ => vec![expr],
  }
}

/// Handle Binomial[expr1, expr2] patterns
fn gf_binomial(
  top: &Expr,
  bottom: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Binomial[n, k] where k is constant: x^k/(-1+x)^(k+1) (with sign adjustment)
  // Canonical Wolfram form uses (-1+x) as base.
  // (1-x)^(k+1) = (-1)^(k+1) * (-1+x)^(k+1), so negate numerator when (k+1) is odd.
  if matches!(top, Expr::Identifier(name) if name == n)
    && !depends_on(bottom, n)
    && let Expr::Integer(k) = bottom
  {
    let power = k + 1;
    let x_pow = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(x.clone()),
      right: Box::new(Expr::Integer(*k)),
    };
    let num = if power % 2 != 0 {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(x_pow),
      }
    } else {
      x_pow
    };
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(num),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(x.clone()),
        }),
        right: Box::new(Expr::Integer(power)),
      }),
    }));
  }

  // Binomial[2n, n] => 1/Sqrt[1-4x]
  if !depends_on(bottom, n) {
    return Ok(None);
  }
  if matches!(bottom, Expr::Identifier(name) if name == n) {
    // Check if top is 2*n
    if let Some((fname, fargs)) = as_func_args(top)
      && fname == "Times"
      && fargs.len() == 2
    {
      let is_2n = (matches!(fargs[0], Expr::Integer(2))
        && matches!(fargs[1], Expr::Identifier(name) if name == n))
        || (matches!(fargs[1], Expr::Integer(2))
          && matches!(fargs[0], Expr::Identifier(name) if name == n));
      if is_2n {
        // 1/Sqrt[1 - 4*x]
        return Ok(Some(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(4)),
              right: Box::new(x.clone()),
            }),
          }),
          right: Box::new(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-1), Expr::Integer(2)],
          }),
        }));
      }
    }
  }

  Ok(None)
}

/// Handle Divide in generating function context
fn gf_divide(
  num: &Expr,
  den: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // 1/(n+1) => -Log[1-x]/x
  if matches!(num, Expr::Integer(1)) {
    if let Some((fname, fargs)) = as_func_args(den)
      && fname == "Plus"
      && fargs.len() == 2
    {
      let is_n_plus_1 = (matches!(fargs[0], Expr::Identifier(name) if name == n)
        && matches!(fargs[1], Expr::Integer(1)))
        || (matches!(fargs[1], Expr::Identifier(name) if name == n)
          && matches!(fargs[0], Expr::Integer(1)));
      if is_n_plus_1 {
        return Ok(Some(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![Expr::BinaryOp {
                op: BinaryOperator::Minus,
                left: Box::new(Expr::Integer(1)),
                right: Box::new(x.clone()),
              }],
            }),
          }),
          right: Box::new(x.clone()),
        }));
      }
    }

    // 1/Factorial[n] => E^x
    if let Expr::FunctionCall { name, args } = den
      && name == "Factorial"
      && args.len() == 1
      && matches!(&args[0], Expr::Identifier(var) if var == n)
    {
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Constant("E".to_string())),
        right: Box::new(x.clone()),
      }));
    }
  }

  // General: numerator / denominator — try to handle as num * den^(-1)
  if !depends_on(num, n) && depends_on(den, n) {
    // const / f(n) — rewrite as const * f(n)^(-1) and try
    let inv_den = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(den.clone()),
      right: Box::new(Expr::Integer(-1)),
    };
    let product = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(num.clone()),
      right: Box::new(inv_den),
    };
    return gf_inner(&product, n, x);
  }

  Ok(None)
}

// ─── ExponentialGeneratingFunction ──────────────────────────────────────

/// ExponentialGeneratingFunction[a_n, n, x] = Sum[a_n * x^n / n!, {n, 0, Infinity}]
fn exponential_generating_function(
  expr: &Expr,
  n_expr: &Expr,
  x_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let n = match n_expr {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ExponentialGeneratingFunction".to_string(),
        args: vec![expr.clone(), n_expr.clone(), x_expr.clone()],
      });
    }
  };

  if let Some(result) = egf_inner(expr, n, x_expr)? {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(Expr::FunctionCall {
      name: "ExponentialGeneratingFunction".to_string(),
      args: vec![expr.clone(), n_expr.clone(), x_expr.clone()],
    })
  }
}

/// Try to compute the polynomial part P(x) such that EGF = E^x * P(x).
/// Returns Some(polynomial) if the expression has this form, None otherwise.
/// This allows combining polynomial parts before multiplying by E^x,
/// producing properly factored output like E^x*(1+x) instead of E^x + E^x*x.
fn egf_poly_part(
  expr: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Constant (doesn't depend on n) => P(x) = constant
  if !depends_on(expr, n) {
    return Ok(Some(expr.clone()));
  }

  // n => P(x) = x
  if matches!(expr, Expr::Identifier(name) if name == n) {
    return Ok(Some(x.clone()));
  }

  // Handle Minus: a - b => a + (-b)
  if let Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left,
    right,
  } = expr
  {
    let neg_right = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: right.clone(),
    };
    let as_plus = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: left.clone(),
      right: Box::new(neg_right),
    };
    return egf_poly_part(&as_plus, n, x);
  }

  if let Some((fname, fargs)) = as_func_args(expr) {
    match fname {
      "Plus" => {
        // Sum of polynomials
        let terms = egf_collect_plus_terms(expr);
        let mut poly_parts = Vec::new();
        for term in &terms {
          if let Some(p) = egf_poly_part(term, n, x)? {
            poly_parts.push(p);
          } else {
            return Ok(None);
          }
        }
        let sum = if poly_parts.len() == 1 {
          poly_parts.remove(0)
        } else {
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: poly_parts,
          }
        };
        return Ok(Some(sum));
      }
      "Times" => {
        // c * f(n) where c doesn't depend on n => P(x) = c * Pf(x)
        let factors = egf_collect_times_factors(expr);
        let mut constants = Vec::new();
        let mut n_dependent = Vec::new();
        for factor in &factors {
          if depends_on(factor, n) {
            n_dependent.push((*factor).clone());
          } else {
            constants.push((*factor).clone());
          }
        }
        if !constants.is_empty() && !n_dependent.is_empty() {
          let rest = if n_dependent.len() == 1 {
            n_dependent.remove(0)
          } else {
            let mut product = n_dependent.remove(0);
            for f in n_dependent {
              product = Expr::BinaryOp {
                op: BinaryOperator::Times,
                left: Box::new(product),
                right: Box::new(f),
              };
            }
            product
          };
          if let Some(inner_poly) = egf_poly_part(&rest, n, x)? {
            let c = if constants.len() == 1 {
              constants.remove(0)
            } else {
              let mut product = constants.remove(0);
              for ci in constants {
                product = Expr::BinaryOp {
                  op: BinaryOperator::Times,
                  left: Box::new(product),
                  right: Box::new(ci),
                };
              }
              product
            };
            return Ok(Some(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(c),
              right: Box::new(inner_poly),
            }));
          }
        }
      }
      "Power" if fargs.len() == 2 => {
        // n^k where k is a non-negative integer => P(x) = factored Stirling polynomial
        if matches!(fargs[0], Expr::Identifier(name) if name == n)
          && let Some(k) = egf_expr_to_nonneg_int(fargs[1])
        {
          return Ok(Some(egf_stirling_polynomial(k, x)));
        }
      }
      _ => {}
    }
  }

  // n^k via BinaryOp::Power
  if let Expr::BinaryOp {
    op: BinaryOperator::Power,
    left,
    right,
  } = expr
    && matches!(left.as_ref(), Expr::Identifier(name) if name == n)
    && let Some(k) = egf_expr_to_nonneg_int(right)
  {
    return Ok(Some(egf_stirling_polynomial(k, x)));
  }

  Ok(None)
}

/// Core EGF pattern matching.
/// Returns Some(result) if a closed form is found, None otherwise.
fn egf_inner(
  expr: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // First try the polynomial approach: EGF = E^x * P(x)
  // This produces properly factored results like E^x*(1+x) instead of E^x + E^x*x
  if let Some(poly) = egf_poly_part(expr, n, x)? {
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Constant("E".to_string())),
        right: Box::new(x.clone()),
      }),
      right: Box::new(poly),
    }));
  }

  // Handle Minus: a - b => a + (-b)
  if let Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left,
    right,
  } = expr
  {
    let neg_right = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: right.clone(),
    };
    let as_plus = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: left.clone(),
      right: Box::new(neg_right),
    };
    return egf_inner(&as_plus, n, x);
  }

  // Handle Divide: a / b => a * b^(-1)
  if let Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left,
    right,
  } = expr
  {
    let as_times = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: left.clone(),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: right.clone(),
        right: Box::new(Expr::Integer(-1)),
      }),
    };
    return egf_inner(&as_times, n, x);
  }

  if let Some((fname, fargs)) = as_func_args(expr) {
    match fname {
      "Plus" => {
        return egf_plus(expr, n, x);
      }
      "Times" => {
        return egf_times(expr, n, x);
      }
      "Power" if fargs.len() == 2 => {
        return egf_power(fargs[0], fargs[1], n, x);
      }
      "Factorial" if fargs.len() == 1 => {
        // EGF[n!, n, x] = 1/(1-x)
        if matches!(fargs[0], Expr::Identifier(name) if name == n) {
          return Ok(Some(Expr::BinaryOp {
            op: BinaryOperator::Power,
            left: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(x.clone()),
            }),
            right: Box::new(Expr::Integer(-1)),
          }));
        }
      }
      // EGF[Sin[n], n, x] = E^(x*Cos[1]) * Sin[x*Sin[1]]
      // Derived from: Sin[n] = Im[e^(in)], so EGF = Im[e^(x*e^i)]
      //   = Im[e^(x*(cos1 + i*sin1))] = e^(x*cos1) * sin(x*sin1)
      // Wolfram outputs: Sin[x*Sin[1]]*(Cosh[x*Cos[1]] + Sinh[x*Cos[1]])
      // which equals e^(x*Cos[1]) * Sin[x*Sin[1]] since Cosh+Sinh = E^x
      "Sin"
        if fargs.len() == 1
          && matches!(fargs[0], Expr::Identifier(name) if name == n) =>
      {
        // Build: Sin[x*Sin[1]] * (Cosh[x*Cos[1]] + Sinh[x*Cos[1]])
        let cos1 = Expr::FunctionCall {
          name: "Cos".to_string(),
          args: vec![Expr::Integer(1)],
        };
        let sin1 = Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![Expr::Integer(1)],
        };
        let x_cos1 = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(x.clone()),
          right: Box::new(cos1),
        };
        let x_sin1 = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(x.clone()),
          right: Box::new(sin1),
        };
        let sin_part = Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![x_sin1],
        };
        let cosh_part = Expr::FunctionCall {
          name: "Cosh".to_string(),
          args: vec![x_cos1.clone()],
        };
        let sinh_part = Expr::FunctionCall {
          name: "Sinh".to_string(),
          args: vec![x_cos1],
        };
        let exp_part = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(cosh_part),
          right: Box::new(sinh_part),
        };
        return Ok(Some(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(sin_part),
          right: Box::new(exp_part),
        }));
      }
      // EGF[Cos[n], n, x] = E^(x*Cos[1]) * Cos[x*Sin[1]]
      "Cos"
        if fargs.len() == 1
          && matches!(fargs[0], Expr::Identifier(name) if name == n) =>
      {
        let cos1 = Expr::FunctionCall {
          name: "Cos".to_string(),
          args: vec![Expr::Integer(1)],
        };
        let sin1 = Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![Expr::Integer(1)],
        };
        let x_cos1 = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(x.clone()),
          right: Box::new(cos1),
        };
        let x_sin1 = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(x.clone()),
          right: Box::new(sin1),
        };
        let cos_part = Expr::FunctionCall {
          name: "Cos".to_string(),
          args: vec![x_sin1],
        };
        let cosh_part = Expr::FunctionCall {
          name: "Cosh".to_string(),
          args: vec![x_cos1.clone()],
        };
        let sinh_part = Expr::FunctionCall {
          name: "Sinh".to_string(),
          args: vec![x_cos1],
        };
        let exp_part = Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(cosh_part),
          right: Box::new(sinh_part),
        };
        return Ok(Some(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(cos_part),
          right: Box::new(exp_part),
        }));
      }
      _ => {}
    }
  }

  Ok(None)
}

/// EGF for Plus: linearity (fallback when poly_part doesn't work)
fn egf_plus(
  expr: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;
  let terms = egf_collect_plus_terms(expr);
  let mut results = Vec::new();
  for term in &terms {
    if let Some(r) = egf_inner(term, n, x)? {
      results.push(r);
    } else {
      return Ok(None);
    }
  }
  let mut sum = results.remove(0);
  for r in results {
    sum = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(sum),
      right: Box::new(r),
    };
  }
  Ok(Some(sum))
}

/// EGF for Times: factor out constants, handle c * f(n)
fn egf_times(
  expr: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;
  let factors = egf_collect_times_factors(expr);

  let mut constants = Vec::new();
  let mut n_dependent = Vec::new();
  for factor in &factors {
    if depends_on(factor, n) {
      n_dependent.push((*factor).clone());
    } else {
      constants.push((*factor).clone());
    }
  }

  if !constants.is_empty() && !n_dependent.is_empty() {
    let c = if constants.len() == 1 {
      constants.remove(0)
    } else {
      let mut product = constants.remove(0);
      for ci in constants {
        product = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(product),
          right: Box::new(ci),
        };
      }
      product
    };
    let rest = if n_dependent.len() == 1 {
      n_dependent.remove(0)
    } else {
      let mut product = n_dependent.remove(0);
      for f in n_dependent {
        product = Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(product),
          right: Box::new(f),
        };
      }
      product
    };
    if let Some(inner) = egf_inner(&rest, n, x)? {
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(c),
        right: Box::new(inner),
      }));
    }
    return Ok(None);
  }

  Ok(None)
}

/// EGF for Power[base, exp]
fn egf_power(
  base: &Expr,
  exp: &Expr,
  n: &str,
  x: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Case: c^n where c doesn't depend on n => e^(c*x)
  if !depends_on(base, n) && matches!(exp, Expr::Identifier(name) if name == n)
  {
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Constant("E".to_string())),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(base.clone()),
        right: Box::new(x.clone()),
      }),
    }));
  }

  // Case: n^k where k is a non-negative integer
  // EGF[n^k, n, x] = e^x * Sum[S(k,j) * x^j, {j=0..k}]
  if matches!(base, Expr::Identifier(name) if name == n)
    && let Some(k) = egf_expr_to_nonneg_int(exp)
  {
    let poly = egf_stirling_polynomial(k, x);
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Constant("E".to_string())),
        right: Box::new(x.clone()),
      }),
      right: Box::new(poly),
    }));
  }

  Ok(None)
}

/// Collect all terms from a Plus expression.
fn egf_collect_plus_terms(expr: &Expr) -> Vec<&Expr> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut terms = egf_collect_plus_terms(left);
      terms.extend(egf_collect_plus_terms(right));
      terms
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().collect()
    }
    _ => vec![expr],
  }
}

/// Collect all factors from a Times expression.
fn egf_collect_times_factors(expr: &Expr) -> Vec<&Expr> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let mut factors = egf_collect_times_factors(left);
      factors.extend(egf_collect_times_factors(right));
      factors
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().collect()
    }
    _ => vec![expr],
  }
}

/// Extract a non-negative integer from an expression.
fn egf_expr_to_nonneg_int(expr: &Expr) -> Option<usize> {
  match expr {
    Expr::Integer(n) if *n >= 0 => Some(*n as usize),
    _ => None,
  }
}

/// Compute the Stirling polynomial: Sum[S(k,j) * x^j, {j=0..k}]
/// For k >= 1, factors out x: x * (S(k,1) + S(k,2)*x + ... + S(k,k)*x^(k-1))
/// to match Wolfram's canonical form (e.g. E^x*x*(1+x) instead of E^x*(x+x^2)).
fn egf_stirling_polynomial(k: usize, x: &Expr) -> Expr {
  use crate::syntax::BinaryOperator;

  let stirling = egf_stirling_numbers(k);

  // k=0: S(0,0)=1, polynomial is just 1
  if k == 0 {
    return Expr::Integer(1);
  }

  // For k >= 1, S(k,0) = 0, so all terms have j >= 1.
  // Factor out x: build inner = S(k,1) + S(k,2)*x + ... + S(k,k)*x^(k-1)
  let mut inner_terms: Vec<Expr> = Vec::new();
  for j in 1..=k {
    let s = stirling[j];
    if s == 0 {
      continue;
    }
    // shifted power: j-1
    let shifted = j - 1;
    let term = if shifted == 0 {
      Expr::Integer(s as i128)
    } else if shifted == 1 {
      if s == 1 {
        x.clone()
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(s as i128)),
          right: Box::new(x.clone()),
        }
      }
    } else {
      let x_power = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(x.clone()),
        right: Box::new(Expr::Integer(shifted as i128)),
      };
      if s == 1 {
        x_power
      } else {
        Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(s as i128)),
          right: Box::new(x_power),
        }
      }
    };
    inner_terms.push(term);
  }

  if inner_terms.is_empty() {
    return Expr::Integer(0);
  }

  let inner = if inner_terms.len() == 1 {
    inner_terms.remove(0)
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: inner_terms,
    }
  };

  // x * inner (or just x if inner is 1)
  if matches!(&inner, Expr::Integer(1)) {
    x.clone()
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(x.clone()),
      right: Box::new(inner),
    }
  }
}

/// Compute Stirling numbers of the second kind S(k, j) for j = 0..k.
fn egf_stirling_numbers(k: usize) -> Vec<u64> {
  if k == 0 {
    return vec![1];
  }
  let mut prev = vec![1u64];
  for i in 1..=k {
    let mut cur = vec![0u64; i + 1];
    for j in 0..=i {
      if j < prev.len() {
        cur[j] += (j as u64) * prev[j];
      }
      if j > 0 && j - 1 < prev.len() {
        cur[j] += prev[j - 1];
      }
    }
    prev = cur;
  }
  prev
}

// ── FourierSinTransform / FourierCosTransform ───────────────────────
// FourierSinTransform[f, t, w] = sqrt(2/pi) * Integrate[f * Sin[w*t], {t, 0, Infinity}]
// FourierCosTransform[f, t, w] = sqrt(2/pi) * Integrate[f * Cos[w*t], {t, 0, Infinity}]

fn fourier_sin_transform(
  expr: &Expr,
  t_expr: &Expr,
  w_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let t = match t_expr {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FourierSinTransform".to_string(),
        args: vec![expr.clone(), t_expr.clone(), w_expr.clone()],
      });
    }
  };

  let normalized = normalize_to_func_calls(expr);
  let normalized =
    crate::evaluator::evaluate_expr_to_expr(&normalized).unwrap_or(normalized);

  if let Some(result) =
    fourier_sin_cos_transform_inner(&normalized, t, w_expr, true)
  {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(Expr::FunctionCall {
      name: "FourierSinTransform".to_string(),
      args: vec![expr.clone(), t_expr.clone(), w_expr.clone()],
    })
  }
}

fn fourier_cos_transform(
  expr: &Expr,
  t_expr: &Expr,
  w_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  let t = match t_expr {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FourierCosTransform".to_string(),
        args: vec![expr.clone(), t_expr.clone(), w_expr.clone()],
      });
    }
  };

  let normalized = normalize_to_func_calls(expr);
  let normalized =
    crate::evaluator::evaluate_expr_to_expr(&normalized).unwrap_or(normalized);

  if let Some(result) =
    fourier_sin_cos_transform_inner(&normalized, t, w_expr, false)
  {
    crate::evaluator::evaluate_expr_to_expr(&result)
  } else {
    Ok(Expr::FunctionCall {
      name: "FourierCosTransform".to_string(),
      args: vec![expr.clone(), t_expr.clone(), w_expr.clone()],
    })
  }
}

/// Try to compute Fourier sine or cosine transform symbolically.
/// is_sin: true for sine transform, false for cosine transform.
fn fourier_sin_cos_transform_inner(
  expr: &Expr,
  t: &str,
  w: &Expr,
  is_sin: bool,
) -> Option<Expr> {
  // sqrt(2/Pi) prefactor
  let prefactor = make_sqrt(make_times(vec![
    Expr::Integer(2),
    make_power(Expr::Constant("Pi".to_string()), Expr::Integer(-1)),
  ]));

  // Handle linearity: c * f(t) where c doesn't depend on t
  if let Some((fname, fargs)) = as_func_args(expr) {
    if fname == "Times" {
      let mut coeff_parts = Vec::new();
      let mut t_parts = Vec::new();
      for a in &fargs {
        if depends_on(a, t) {
          t_parts.push((*a).clone());
        } else {
          coeff_parts.push((*a).clone());
        }
      }
      if !coeff_parts.is_empty() && !t_parts.is_empty() {
        let inner = if t_parts.len() == 1 {
          t_parts.into_iter().next().unwrap()
        } else {
          make_times(t_parts)
        };
        if let Some(result) =
          fourier_sin_cos_transform_inner(&inner, t, w, is_sin)
        {
          let mut all = coeff_parts;
          all.push(result);
          return Some(make_times(all));
        }
      }
    } else if fname == "Plus" {
      let mut results = Vec::new();
      for a in &fargs {
        if let Some(r) = fourier_sin_cos_transform_inner(a, t, w, is_sin) {
          results.push(r);
        } else {
          return None;
        }
      }
      return Some(make_plus(results));
    }
  }

  // FST[E^(-a*t), t, w] = sqrt(2/Pi) * w / (a^2 + w^2) (for a > 0)
  // FCT[E^(-a*t), t, w] = sqrt(2/Pi) * a / (a^2 + w^2) (for a > 0)
  if let Some(exp_arg) = is_exp_of(expr) {
    // Check if exp_arg = -a*t (negative linear in t)
    if let Some(neg_coeff) = extract_neg_linear_coeff(exp_arg, t) {
      let a = neg_coeff;
      let a2_plus_w2 = make_plus(vec![
        make_power(a.clone(), Expr::Integer(2)),
        make_power(w.clone(), Expr::Integer(2)),
      ]);
      let result = if is_sin {
        // sqrt(2/Pi) * w / (a^2 + w^2)
        make_times(vec![
          prefactor,
          w.clone(),
          make_power(a2_plus_w2, Expr::Integer(-1)),
        ])
      } else {
        // sqrt(2/Pi) * a / (a^2 + w^2)
        make_times(vec![
          prefactor,
          a,
          make_power(a2_plus_w2, Expr::Integer(-1)),
        ])
      };
      return Some(result);
    }
  }

  // FST[t, t, w] — just the variable itself
  if let Expr::Identifier(name) = expr
    && name == t
    && is_sin
  {
    // FST[t, t, w] = sqrt(2/Pi) * ... not well-defined (diverges)
    return None;
  }

  // FST[1/t, t, w] = sqrt(Pi/2) (for w > 0)
  if is_sin
    && let Some((fname, fargs)) = as_func_args(expr)
    && fname == "Power"
    && fargs.len() == 2
    && matches!(&fargs[0], Expr::Identifier(v) if v == t)
    && matches!(&fargs[1], Expr::Integer(-1))
  {
    return Some(make_sqrt(make_times(vec![
      Expr::Constant("Pi".to_string()),
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)],
      },
    ])));
  }

  None
}

/// Check if expr is E^(something) or Exp[something], return the exponent.
fn is_exp_of(expr: &Expr) -> Option<&Expr> {
  if let Some((fname, fargs)) = as_func_args(expr) {
    if fname == "Exp" && fargs.len() == 1 {
      return Some(fargs[0]);
    }
    if fname == "Power"
      && fargs.len() == 2
      && matches!(&fargs[0], Expr::Identifier(e) | Expr::Constant(e) if e == "E")
    {
      return Some(fargs[1]);
    }
  }
  None
}

/// If expr = -a * t where a doesn't depend on t, return a.
fn extract_neg_linear_coeff(expr: &Expr, t: &str) -> Option<Expr> {
  if let Some((fname, fargs)) = as_func_args(expr)
    && fname == "Times"
  {
    let mut has_t = false;
    let mut has_neg = false;
    let mut coeffs = Vec::new();
    for a in fargs {
      if let Expr::Identifier(v) = a
        && v == t
      {
        has_t = true;
        continue;
      }
      if matches!(a, Expr::Integer(-1)) {
        has_neg = true;
        continue;
      }
      if !depends_on(a, t) {
        coeffs.push(a.clone());
      } else {
        return None;
      }
    }
    if has_t && has_neg {
      return Some(if coeffs.is_empty() {
        Expr::Integer(1)
      } else if coeffs.len() == 1 {
        coeffs.into_iter().next().unwrap()
      } else {
        make_times(coeffs)
      });
    }
  }
  // Check if it's -t directly: BinaryOp Times(-1, t)
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left,
    right,
  } = expr
    && matches!(left.as_ref(), Expr::Integer(-1))
    && matches!(right.as_ref(), Expr::Identifier(v) if v == t)
  {
    return Some(Expr::Integer(1));
  }
  None
}
