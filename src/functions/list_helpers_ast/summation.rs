#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AnglePath[{θ1, θ2, ...}] - path with unit steps and cumulative turning angles.
/// AnglePath[{{r1, θ1}, {r2, θ2}, ...}] - path with specified step lengths.
pub fn angle_path_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "AnglePath expects 1 or 2 arguments".into(),
    ));
  }

  // AnglePath[{{x0, y0}, θ0}, steps] form: consume initial position and
  // starting angle from the first argument, then treat the second argument
  // as the step list.
  let (start_pos, start_angle, items_vec): (
    Option<Expr>,
    Option<Expr>,
    Vec<Expr>,
  ) = if args.len() == 2 {
    let (pos, theta) = parse_initial_spec(&args[0])?;
    let step_items = match &args[1] {
      Expr::List(items) => items.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "AnglePath".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    (Some(pos), Some(theta), step_items.to_vec())
  } else {
    let step_items = match &args[0] {
      Expr::List(items) => items.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "AnglePath".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    (None, None, step_items.to_vec())
  };
  let items = &items_vec;

  // Start at origin
  let mut x = 0.0_f64;
  let mut y = 0.0_f64;
  let mut angle = 0.0_f64;
  let mut points: Vec<Expr> = Vec::with_capacity(items.len() + 1);

  // Check if all items are numeric (pure angles) or {step, angle} pairs
  let is_pair_form = items
    .first()
    .is_some_and(|item| matches!(item, Expr::List(_)));

  // Use numeric mode only when every input converts to a float AND at least
  // one input is an explicit Real. Integer-only inputs stay symbolic so
  // Cos/Sin are preserved, and any non-numeric input (e.g. `{a, b}`) stays
  // symbolic so the output can still be produced.
  let any_real = if is_pair_form {
    items.iter().any(|item| {
      if let Expr::List(pair) = item {
        pair.iter().any(|p| matches!(p, Expr::Real(_)))
      } else {
        matches!(item, Expr::Real(_))
      }
    })
  } else {
    items.iter().any(|item| matches!(item, Expr::Real(_)))
  };
  let all_floatable = if is_pair_form {
    items.iter().all(|item| {
      if let Expr::List(pair) = item {
        pair.len() == 2
          && matches!(&pair[0], Expr::Integer(_) | Expr::Real(_))
          && matches!(&pair[1], Expr::Integer(_) | Expr::Real(_))
      } else {
        matches!(item, Expr::Integer(_) | Expr::Real(_))
      }
    })
  } else {
    items
      .iter()
      .all(|item| matches!(item, Expr::Integer(_) | Expr::Real(_)))
  };
  let use_numeric = all_floatable && any_real;

  if !use_numeric {
    // Symbolic mode: keep exact Cos/Sin
    let mut cum_terms_x: Vec<Expr> = Vec::new();
    let mut cum_terms_y: Vec<Expr> = Vec::new();
    let mut cum_angle = start_angle.clone().unwrap_or(Expr::Integer(0));

    let (start_x, start_y) = if let Some(Expr::List(pair)) = &start_pos {
      if pair.len() == 2 {
        (pair[0].clone(), pair[1].clone())
      } else {
        (Expr::Integer(0), Expr::Integer(0))
      }
    } else {
      (Expr::Integer(0), Expr::Integer(0))
    };
    let start_x_expr = start_x.clone();
    let start_y_expr = start_y.clone();
    points.push(Expr::List(vec![start_x.clone(), start_y.clone()].into()));
    if !matches!(start_x, Expr::Integer(0)) {
      cum_terms_x.push(start_x_expr);
    }
    if !matches!(start_y, Expr::Integer(0)) {
      cum_terms_y.push(start_y_expr);
    }

    for item in items {
      let (step, theta) = if is_pair_form {
        if let Expr::List(pair) = item {
          if pair.len() != 2 {
            return Err(InterpreterError::EvaluationError(
              "AnglePath: each element must be a number or {step, angle} pair"
                .into(),
            ));
          }
          (pair[0].clone(), pair[1].clone())
        } else {
          let theta = item.clone();
          (Expr::Integer(1), theta)
        }
      } else {
        (Expr::Integer(1), item.clone())
      };

      // cum_angle += theta
      cum_angle = crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[cum_angle, theta],
      )?;

      // cos_term = step * Cos[cum_angle], sin_term = step * Sin[cum_angle]
      let cos_val = crate::evaluator::evaluate_function_call_ast(
        "Cos",
        &[cum_angle.clone()],
      )?;
      let sin_val = crate::evaluator::evaluate_function_call_ast(
        "Sin",
        &[cum_angle.clone()],
      )?;
      let cos_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[step.clone(), cos_val],
      )?;
      let sin_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[step, sin_val],
      )?;

      cum_terms_x.push(cos_term);
      cum_terms_y.push(sin_term);

      let px = if cum_terms_x.len() == 1 {
        cum_terms_x[0].clone()
      } else {
        crate::evaluator::evaluate_function_call_ast("Plus", &cum_terms_x)?
      };
      let py = if cum_terms_y.len() == 1 {
        cum_terms_y[0].clone()
      } else {
        crate::evaluator::evaluate_function_call_ast("Plus", &cum_terms_y)?
      };

      points.push(Expr::List(vec![px, py].into()));
    }
  } else {
    // Numeric mode
    if let Some(Expr::List(pair)) = &start_pos
      && pair.len() == 2
    {
      x = expr_to_f64(&pair[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "AnglePath: starting x must be numeric".into(),
        )
      })?;
      y = expr_to_f64(&pair[1]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "AnglePath: starting y must be numeric".into(),
        )
      })?;
    }
    if let Some(theta_expr) = &start_angle {
      angle = expr_to_f64(theta_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "AnglePath: starting angle must be numeric".into(),
        )
      })?;
    }
    points.push(Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()));

    for item in items {
      let (step, theta) = if is_pair_form {
        if let Expr::List(pair) = item {
          if pair.len() != 2 {
            return Err(InterpreterError::EvaluationError(
              "AnglePath: each element must be a number or {step, angle} pair"
                .into(),
            ));
          }
          let s = expr_to_f64(&pair[0]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "AnglePath: step must be numeric".into(),
            )
          })?;
          let t = expr_to_f64(&pair[1]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "AnglePath: angle must be numeric".into(),
            )
          })?;
          (s, t)
        } else {
          let t = expr_to_f64(item).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "AnglePath: angle must be numeric".into(),
            )
          })?;
          (1.0, t)
        }
      } else {
        let t = expr_to_f64(item).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "AnglePath: angle must be numeric".into(),
          )
        })?;
        (1.0, t)
      };

      angle += theta;
      x += step * angle.cos();
      y += step * angle.sin();
      points.push(Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()));
    }
  }

  Ok(Expr::List(points.into()))
}

/// Parse the 2-argument form's first argument: `{{x0, y0}, θ0}`.
/// Returns (position, initial_angle).
fn parse_initial_spec(expr: &Expr) -> Result<(Expr, Expr), InterpreterError> {
  if let Expr::List(items) = expr
    && items.len() == 2
    && let Expr::List(pos) = &items[0]
    && pos.len() == 2
  {
    return Ok((Expr::List(pos.clone()), items[1].clone()));
  }
  Err(InterpreterError::EvaluationError(
    "AnglePath: first argument must be {{x, y}, angle}".into(),
  ))
}

/// AST-based Product: product of list elements or iterator product.
/// Recognise the integrand `1 + 1/var^2` across the AST shapes Woxi
/// produces for that input. Used by the infinite-product closed form
/// `Product[1 + 1/k², {k, 1, ∞}] = Sinh[π]/π` so we don't depend on the
/// exact canonical form of the body.
/// True if `body` represents `1 - 1/var^4` in any canonical AST form Woxi
/// parses: `Plus[1, Times[-1, Power[var, -4]]]`, `1 + Power[var, -4]*-1`,
/// `1 - 1/var^4` (BinaryOp), etc.
fn body_is_one_minus_one_over_var_quartic(body: &Expr, var_name: &str) -> bool {
  use crate::syntax::BinaryOperator;
  let is_var_to_neg_four = |e: &Expr| -> bool {
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
      && matches!(right.as_ref(), Expr::Integer(-4))
    {
      return true;
    }
    if let Expr::FunctionCall { name, args } = e
      && name == "Power"
      && args.len() == 2
      && matches!(&args[0], Expr::Identifier(s) if s == var_name)
      && matches!(&args[1], Expr::Integer(-4))
    {
      return true;
    }
    false
  };
  // `var^4` (positive exponent), used in `1 - 1/var^4` BinaryOp form.
  let is_var_to_four = |e: &Expr| -> bool {
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
      && matches!(right.as_ref(), Expr::Integer(4))
    {
      return true;
    }
    if let Expr::FunctionCall { name, args } = e
      && name == "Power"
      && args.len() == 2
      && matches!(&args[0], Expr::Identifier(s) if s == var_name)
      && matches!(&args[1], Expr::Integer(4))
    {
      return true;
    }
    false
  };
  // Recognise `-1 * var^-4` or `var^-4 * -1`.
  let is_neg_one_over_var_quartic = |e: &Expr| -> bool {
    // `Times[-1, var^-4]`.
    if let Expr::FunctionCall { name, args } = e
      && name == "Times"
      && args.len() == 2
      && ((matches!(&args[0], Expr::Integer(-1))
        && is_var_to_neg_four(&args[1]))
        || (matches!(&args[1], Expr::Integer(-1))
          && is_var_to_neg_four(&args[0])))
    {
      return true;
    }
    // `-1/var^4` as BinaryOp Times.
    if let Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } = e
      && ((matches!(left.as_ref(), Expr::Integer(-1))
        && is_var_to_neg_four(right))
        || (matches!(right.as_ref(), Expr::Integer(-1))
          && is_var_to_neg_four(left)))
    {
      return true;
    }
    // `Divide[-1, var^4]`.
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Integer(-1))
      && is_var_to_four(right)
    {
      return true;
    }
    false
  };
  // Match `Plus[1, -1 * var^-4]` in either FunctionCall or BinaryOp form.
  if let Expr::FunctionCall { name, args } = body
    && name == "Plus"
    && args.len() == 2
  {
    return (matches!(&args[0], Expr::Integer(1))
      && is_neg_one_over_var_quartic(&args[1]))
      || (matches!(&args[1], Expr::Integer(1))
        && is_neg_one_over_var_quartic(&args[0]));
  }
  if let Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left,
    right,
  } = body
  {
    return (matches!(left.as_ref(), Expr::Integer(1))
      && is_neg_one_over_var_quartic(right))
      || (matches!(right.as_ref(), Expr::Integer(1))
        && is_neg_one_over_var_quartic(left));
  }
  // `1 - 1/var^4` BinaryOp Minus.
  if let Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left,
    right,
  } = body
    && matches!(left.as_ref(), Expr::Integer(1))
  {
    // right side must be `1/var^4`.
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: rl,
      right: rr,
    } = right.as_ref()
      && matches!(rl.as_ref(), Expr::Integer(1))
      && is_var_to_four(rr)
    {
      return true;
    }
    if is_var_to_neg_four(right) {
      return true;
    }
  }
  false
}

fn body_is_one_plus_one_over_var_squared(body: &Expr, var_name: &str) -> bool {
  use crate::syntax::BinaryOperator;
  let is_var_squared = |e: &Expr| -> bool {
    matches!(
      e,
      Expr::BinaryOp { op: BinaryOperator::Power, left, right }
        if matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
          && matches!(right.as_ref(), Expr::Integer(2))
    ) || matches!(
      e,
      Expr::FunctionCall { name, args }
        if name == "Power"
          && args.len() == 2
          && matches!(&args[0], Expr::Identifier(s) if s == var_name)
          && matches!(&args[1], Expr::Integer(2))
    )
  };
  let is_one_over_var_squared = |e: &Expr| -> bool {
    // `1 / var^2`
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Integer(1))
      && is_var_squared(right.as_ref())
    {
      return true;
    }
    // `var^(-2)` (canonical form for reciprocal squares)
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } = e
      && matches!(left.as_ref(), Expr::Identifier(s) if s == var_name)
      && matches!(right.as_ref(), Expr::Integer(-2))
    {
      return true;
    }
    if let Expr::FunctionCall { name, args } = e
      && name == "Power"
      && args.len() == 2
      && matches!(&args[0], Expr::Identifier(s) if s == var_name)
      && matches!(&args[1], Expr::Integer(-2))
    {
      return true;
    }
    false
  };
  // Match the Plus shapes: Plus[1, 1/var^2] or Plus[1/var^2, 1] in either
  // BinaryOp::Plus or FunctionCall["Plus", …] form.
  if let Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left,
    right,
  } = body
  {
    return (matches!(left.as_ref(), Expr::Integer(1))
      && is_one_over_var_squared(right))
      || (matches!(right.as_ref(), Expr::Integer(1))
        && is_one_over_var_squared(left));
  }
  if let Expr::FunctionCall { name, args } = body
    && name == "Plus"
    && args.len() == 2
  {
    return (matches!(&args[0], Expr::Integer(1))
      && is_one_over_var_squared(&args[1]))
      || (matches!(&args[1], Expr::Integer(1))
        && is_one_over_var_squared(&args[0]));
  }
  false
}

pub fn product_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Product[{a, b, c}] -> a * b * c
    let items = match &args[0] {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Product".to_string(),
          args: args.to_vec().into(),
        });
      }
    };

    let mut product = 1.0;
    for item in items {
      if let Some(n) = expr_to_f64(item) {
        product *= n;
      } else {
        return Ok(Expr::FunctionCall {
          name: "Product".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
    return Ok(f64_to_expr(product));
  }

  if args.len() == 2 {
    // Product[expr, {i, min, max}] -> multiply expr for each i
    let body = &args[0];
    let iter_spec = &args[1];

    match iter_spec {
      Expr::List(items) if items.len() >= 2 => {
        let var_name = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Product".to_string(),
              args: args.to_vec().into(),
            });
          }
        };

        // Check for list iteration form: {i, list}
        if items.len() == 2 {
          let evaluated_second =
            crate::evaluator::evaluate_expr_to_expr(&items[1])?;
          if let Expr::List(list_items) = &evaluated_second {
            // Product[expr, {i, list}] -> iterate over list elements
            let mut product = 1.0;
            for item in list_items {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, item);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              if let Some(n) = expr_to_f64(&val) {
                product *= n;
              } else {
                return Ok(Expr::FunctionCall {
                  name: "Product".to_string(),
                  args: args.to_vec().into(),
                });
              }
            }
            return Ok(f64_to_expr(product));
          }
        }

        // Check if bounds are numeric
        let bounds = if items.len() == 2 {
          expr_to_i128(&items[1]).map(|max| (1i128, max))
        } else {
          match (expr_to_i128(&items[1]), expr_to_i128(&items[2])) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
          }
        };

        // If bounds are symbolic, try to compute symbolic product
        if bounds.is_none() {
          let min_concrete = if items.len() == 2 {
            Some(1i128) // {i, n} implies min = 1
          } else {
            expr_to_i128(&items[1])
          };
          let max_concrete = if items.len() == 2 {
            expr_to_i128(&items[1])
          } else {
            expr_to_i128(&items[2])
          };
          let max_expr = if items.len() == 2 {
            &items[1]
          } else {
            &items[2]
          };
          let min_expr = if items.len() == 2 {
            &Expr::Integer(1)
          } else {
            &items[1]
          };

          let max_is_infinity =
            matches!(max_expr, Expr::Identifier(s) if s == "Infinity");

          // Body independent of the iteration variable:
          //   Product[c, {k, min, max}] = c^(max - min + 1)
          if !crate::functions::polynomial_ast::contains_var(body, &var_name)
            && !max_is_infinity
          {
            let count =
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![
                  Expr::Integer(1),
                  max_expr.clone(),
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![Expr::Integer(-1), min_expr.clone()].into(),
                  },
                ]
                .into(),
              })?;
            let power = Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(body.clone()),
              right: Box::new(count),
            };
            return crate::evaluator::evaluate_expr_to_expr(&power);
          }

          // Body is the iteration variable itself: Product[k, {k, ...}]
          // (the closed forms below use Factorial/Pochhammer, which are only
          // valid for a finite upper limit).
          if matches!(body, Expr::Identifier(name) if name == &var_name)
            && !max_is_infinity
          {
            if let Some(min_val) = min_concrete {
              if max_concrete.is_none() {
                // Product[k, {k, concrete_min, symbolic_max}]
                // = max! / (min-1)!
                let n_factorial = Expr::FunctionCall {
                  name: "Factorial".to_string(),
                  args: vec![max_expr.clone()].into(),
                };
                if min_val == 1 {
                  return Ok(n_factorial);
                }
                // Compute (min-1)! as a concrete integer
                let mut denom: i128 = 1;
                for j in 2..min_val {
                  denom *= j;
                }
                // (min-1)! == 1 (min <= 2): the product is just max!.
                if denom == 1 {
                  return Ok(n_factorial);
                }
                return Ok(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Divide,
                  left: Box::new(n_factorial),
                  right: Box::new(Expr::Integer(denom)),
                });
              }
            } else if max_concrete.is_none() {
              // Product[k, {k, sym_min, sym_max}]
              // = Pochhammer[min, 1 - min + max]
              return Ok(Expr::FunctionCall {
                name: "Pochhammer".to_string(),
                args: vec![
                  min_expr.clone(),
                  // 1 - min + max
                  Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Plus,
                    left: Box::new(Expr::BinaryOp {
                      op: crate::syntax::BinaryOperator::Minus,
                      left: Box::new(Expr::Integer(1)),
                      right: Box::new(min_expr.clone()),
                    }),
                    right: Box::new(max_expr.clone()),
                  },
                ]
                .into(),
              });
            }
          }

          // Body is c^var: Product[c^i, {i, 1, n}] = c^(n*(1+n)/2)
          if let Some(1) = min_concrete
            && max_concrete.is_none()
            && let Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: base,
              right: exp,
            } = body
          {
            if matches!(exp.as_ref(), Expr::Identifier(name) if name == &var_name)
            {
              // Product[c^i, {i, 1, n}] = c^((n*(1+n))/2)
              let n = max_expr.clone();
              let exponent = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(n.clone()),
                  right: Box::new(Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Plus,
                    left: Box::new(Expr::Integer(1)),
                    right: Box::new(n),
                  }),
                }),
                right: Box::new(Expr::Integer(2)),
              };
              return Ok(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: base.clone(),
                right: Box::new(exponent),
              });
            }

            // Product[i^k, {i, 1, n}] = n!^k
            if matches!(base.as_ref(), Expr::Identifier(name) if name == &var_name)
            {
              return Ok(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Factorial".to_string(),
                  args: vec![max_expr.clone()].into(),
                }),
                right: exp.clone(),
              });
            }
          }

          // Monomial body c*var^p (p a nonzero integer, c free of var):
          //   Product[c var^p, {k, 1, n}] = c^n * n!^p.
          // wolframscript keeps the bare factorial when c == 1
          // (Product[1/k] -> n!^(-1)) but switches to Gamma[1+n] once a
          // coefficient is present (Product[2 k] -> 2^n Gamma[1+n]).
          if min_concrete == Some(1)
            && max_concrete.is_none()
            && !max_is_infinity
            && crate::functions::polynomial_ast::contains_var(body, &var_name)
            && let Ok(dbody) =
              crate::functions::calculus_ast::differentiate_expr(
                body, &var_name,
              )
          {
            use crate::syntax::BinaryOperator;
            // p = (d body / d var) * var / body — the exponent of a monomial.
            let p_expr = Expr::FunctionCall {
              name: "Simplify".to_string(),
              args: vec![Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  dbody,
                  Expr::Identifier(var_name.clone()),
                  Expr::FunctionCall {
                    name: "Power".to_string(),
                    args: vec![body.clone(), Expr::Integer(-1)].into(),
                  },
                ]
                .into(),
              }]
              .into(),
            };
            let p_val = crate::evaluator::evaluate_expr_to_expr(&p_expr).ok();
            if let Some(p) = p_val.as_ref().and_then(expr_to_i128)
              && p != 0
            {
              // c = body with var -> 1.
              let c = crate::evaluator::evaluate_expr_to_expr(
                &crate::syntax::substitute_variable(
                  body,
                  &var_name,
                  &Expr::Integer(1),
                ),
              )?;
              if !crate::functions::polynomial_ast::contains_var(&c, &var_name)
              {
                let c_is_one = matches!(&c, Expr::Integer(1));
                let base = if c_is_one {
                  Expr::FunctionCall {
                    name: "Factorial".to_string(),
                    args: vec![max_expr.clone()].into(),
                  }
                } else {
                  Expr::FunctionCall {
                    name: "Gamma".to_string(),
                    args: vec![Expr::BinaryOp {
                      op: BinaryOperator::Plus,
                      left: Box::new(Expr::Integer(1)),
                      right: Box::new(max_expr.clone()),
                    }]
                    .into(),
                  }
                };
                let pow_part = if p == 1 {
                  base
                } else {
                  Expr::BinaryOp {
                    op: BinaryOperator::Power,
                    left: Box::new(base),
                    right: Box::new(Expr::Integer(p)),
                  }
                };
                // Coefficient c^n. wolframscript renders a *unit fraction* 1/b as
                // a denominator power (Product[k/2] -> Gamma[1+n]/2^n), but keeps
                // any other coefficient as c^n (Product[2k/3] -> (2/3)^n Gamma).
                let pow_n = |b: &Expr| Expr::BinaryOp {
                  op: BinaryOperator::Power,
                  left: Box::new(b.clone()),
                  right: Box::new(max_expr.clone()),
                };
                let unit_fraction_den = match &c {
                  Expr::FunctionCall { name, args }
                    if name == "Rational" && args.len() == 2 =>
                  {
                    match (&args[0], &args[1]) {
                      (Expr::Integer(1), Expr::Integer(b)) => Some(*b),
                      _ => None,
                    }
                  }
                  _ => None,
                };
                let result = if c_is_one {
                  pow_part
                } else if let Some(b) = unit_fraction_den {
                  Expr::BinaryOp {
                    op: BinaryOperator::Divide,
                    left: Box::new(pow_part),
                    right: Box::new(pow_n(&Expr::Integer(b))),
                  }
                } else {
                  Expr::FunctionCall {
                    name: "Times".to_string(),
                    args: vec![pow_n(&c), pow_part].into(),
                  }
                };
                return crate::evaluator::evaluate_expr_to_expr(&result);
              }
            }
          }

          // Closed form for the classical infinite product
          //   ∏_{k=1}^∞ (1 + 1/k²) = Sinh[π] / π
          // (the `x = π` case of the Weierstrass factorisation
          //   sinh(x)/x = ∏_{k=1}^∞ (1 + x²/(kπ)²)).
          // Recognised when min == 1, max == Infinity, and the body is
          // `1 + 1/var^2` in any of the canonical AST shapes.
          if let Some(1) = min_concrete
            && matches!(max_expr, Expr::Identifier(s) if s == "Infinity")
            && body_is_one_plus_one_over_var_squared(body, &var_name)
          {
            return Ok(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![Expr::Constant("Pi".to_string())].into(),
              }),
              right: Box::new(Expr::Constant("Pi".to_string())),
            });
          }

          // Closed form for ∏_{k=2}^∞ (1 - 1/k⁴) = Sinh[π] / (4 π).
          // Comes from (1 - x²/k²)(1 + x²/k²) at x = 1 over k ≥ 2: the k = 1
          // factor is 0, but starting at k = 2 the residual product is
          // Sinh[π] · sin[π] / π² which is 0/π² = 0 — instead the standard
          // result is obtained by L'Hôpital at x → 1 of the truncated form.
          if let Some(2) = min_concrete
            && matches!(max_expr, Expr::Identifier(s) if s == "Infinity")
            && body_is_one_minus_one_over_var_quartic(body, &var_name)
          {
            return Ok(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(Expr::FunctionCall {
                name: "Sinh".to_string(),
                args: vec![Expr::Constant("Pi".to_string())].into(),
              }),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::Integer(4)),
                right: Box::new(Expr::Constant("Pi".to_string())),
              }),
            });
          }

          // For other symbolic cases, return unevaluated
          return Ok(Expr::FunctionCall {
            name: "Product".to_string(),
            args: args.to_vec().into(),
          });
        }

        let (min, max) = bounds.unwrap();

        let step = if items.len() >= 4 {
          expr_to_i128(&items[3]).unwrap_or(1)
        } else {
          1
        };

        // Collect evaluated values for each iteration
        let mut values: Vec<Expr> = Vec::new();
        let mut i = min;
        while (step > 0 && i <= max) || (step < 0 && i >= max) {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          values.push(val);
          i += step;
        }

        // Try numeric product first
        let mut numeric_product = 1.0;
        let mut all_numeric = true;
        for val in &values {
          if let Some(n) = expr_to_f64(val) {
            numeric_product *= n;
          } else {
            all_numeric = false;
            break;
          }
        }

        if all_numeric {
          return Ok(f64_to_expr(numeric_product));
        }

        // For symbolic values, build a Times expression
        if values.is_empty() {
          return Ok(Expr::Integer(1));
        }
        if values.len() == 1 {
          return Ok(values.into_iter().next().unwrap());
        }
        // Fold into nested Times
        let mut result = values.remove(0);
        for val in values {
          result = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(result),
            right: Box::new(val),
          };
        }
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
      _ => {}
    }
  }

  Ok(Expr::FunctionCall {
    name: "Product".to_string(),
    args: args.to_vec().into(),
  })
}

/// AST-based Sum: sum of list elements or iterator sum.
pub fn sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    // Sum requires at least 2 arguments
    return Ok(Expr::FunctionCall {
      name: "Sum".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Indefinite sum: Sum[f[i], i] → ∑_{k=0}^{i-1} f[k] (the antidifference F
  // where F(i+1) - F(i) = f(i), with F(0) = 0).
  // wolframscript: Sum[1, i] = i, Sum[i, i] = ((-1 + i)*i)/2,
  // Sum[i^3, i] = ((-1+i)^2*i^2)/4.
  //
  // Implementation: compute ∑_{k=1}^{i-1} f[k] (the path with proven symbolic
  // support) and add f(0). For f=1 this gives 1 + (i-1) = i; for f=i it gives
  // 0 + i(i-1)/2 = i(i-1)/2.
  if args.len() == 2
    && let Expr::Identifier(var_name) = &args[1]
  {
    let fresh_name = format!("$sum_indef_{}_$", var_name);
    let fresh = Expr::Identifier(fresh_name.clone());
    let body_in_fresh =
      crate::syntax::substitute_variable(&args[0], var_name, &fresh);
    // Upper bound: var - 1, built as `(-1) + var` for canonical Plus form.
    let upper = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier(var_name.clone())].into(),
    };
    let iter_spec =
      Expr::List(vec![fresh.clone(), Expr::Integer(1), upper].into());
    let inner_sum = sum_ast(&[body_in_fresh, iter_spec])?;
    // Evaluate f(0)
    let f_at_zero =
      crate::syntax::substitute_variable(&args[0], var_name, &Expr::Integer(0));
    let f_at_zero_eval = crate::evaluator::evaluate_expr_to_expr(&f_at_zero)?;
    return crate::functions::math_ast::plus_ast(&[f_at_zero_eval, inner_sum]);
  }

  // Multi-dimensional Sum: Sum[expr, {i,...}, {j,...}, ...] => Sum[Sum[expr, {j,...}], {i,...}]
  if args.len() > 2 {
    // Evaluate innermost sum first (last iterator), then wrap outward
    let body = &args[0];
    let inner_iter = &args[args.len() - 1];
    let inner_sum = sum_ast(&[body.clone(), inner_iter.clone()])?;
    if args.len() == 3 {
      return sum_ast(&[inner_sum, args[1].clone()]);
    } else {
      let mut new_args = vec![inner_sum];
      new_args.extend_from_slice(&args[1..args.len() - 1]);
      return sum_ast(&new_args);
    }
  }

  // Sum[expr, {i, min, max}] or variants
  let body = &args[0];
  let iter_spec = &args[1];

  match iter_spec {
    Expr::List(items) if items.len() >= 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: args.to_vec().into(),
          });
        }
      };

      // Check for list iteration form: {i, list}
      if items.len() == 2 {
        let evaluated_second =
          crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        if let Expr::List(list_items) = &evaluated_second {
          // Sum[expr, {i, {v1, v2, ...}}] -> iterate over list elements
          let mut acc = Expr::Integer(0);
          for item in list_items {
            let substituted =
              crate::syntax::substitute_variable(body, &var_name, item);
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          }
          return Ok(acc);
        }
      }

      // Short form `Sum[expr, {i, max}]` is sugar for `Sum[expr, {i, 1, max}]`.
      // Forward to the 3-element form so symbolic / Infinity bounds work.
      if items.len() == 2 {
        let new_iter = Expr::List(
          vec![items[0].clone(), Expr::Integer(1), items[1].clone()].into(),
        );
        return sum_ast(&[body.clone(), new_iter]);
      }

      // Check for infinite sum: {i, min, Infinity}
      if items.len() == 3
        && let Expr::Identifier(s) = &items[2]
        && s == "Infinity"
      {
        let min_val = expr_to_i128(&items[1]).unwrap_or(1);
        if let Some(result) = try_infinite_sum(body, &var_name, min_val)? {
          return Ok(result);
        }
        // Could not evaluate symbolically — return unevaluated
        return Ok(Expr::FunctionCall {
          name: "Sum".to_string(),
          args: args.to_vec().into(),
        });
      }

      // Fast path: Sum[Factorial[var], {var, min, max(, step)}] with integer
      // bounds, min >= 0, step >= 1. Uses a running product so each factorial
      // is built incrementally instead of recomputed from scratch.
      if (items.len() == 3 || items.len() == 4)
        && let Expr::FunctionCall {
          name: fname,
          args: fargs,
        } = body
        && fname == "Factorial"
        && fargs.len() == 1
        && matches!(&fargs[0], Expr::Identifier(fv) if fv == &var_name)
      {
        let min_int = expr_to_i128(&items[1]);
        let max_int = expr_to_i128(&items[2]);
        let step_int = if items.len() == 4 {
          expr_to_i128(&items[3])
        } else {
          Some(1)
        };
        if let (Some(min), Some(max), Some(step)) = (min_int, max_int, step_int)
          && step >= 1
          && min >= 0
          && max >= min
        {
          let mut fact = num_bigint::BigInt::from(1);
          for k in 2..=min {
            fact *= num_bigint::BigInt::from(k);
          }
          let mut sum = num_bigint::BigInt::from(0);
          let mut i = min;
          while i <= max {
            sum += &fact;
            for s in 1..=step {
              let next = i + s;
              if next >= 2 {
                fact *= num_bigint::BigInt::from(next);
              }
            }
            i += step;
          }
          return Ok(crate::functions::math_ast::bigint_to_expr(sum));
        }
      }

      // Try real-valued iteration when bounds are numeric but not integers
      if items.len() == 3 {
        let min_int = expr_to_i128(&items[1]);
        let max_int = expr_to_i128(&items[2]);
        if min_int.is_none() || max_int.is_none() {
          // Check if bounds are numeric reals
          let min_f = crate::functions::math_ast::try_eval_to_f64(&items[1]);
          let max_f = crate::functions::math_ast::try_eval_to_f64(&items[2]);
          if let (Some(min_val), Some(max_val)) = (min_f, max_f) {
            // Iterate with step=1, substituting real values
            let mut acc = Expr::Integer(0);
            let mut i = min_val;
            while i <= max_val + 1e-10 {
              let sub_val =
                if (i - i.round()).abs() < 1e-10 && min_int.is_some() {
                  Expr::Integer(i.round() as i128)
                } else {
                  Expr::Real(i)
                };
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &sub_val);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
              i += 1.0;
            }
            return Ok(acc);
          }
        }
      }

      // Try iterating when the difference between bounds is real
      // (handles complex bounds like {k, I, I+1})
      if items.len() == 3 {
        let diff = crate::functions::math_ast::plus_ast(&[
          items[2].clone(),
          Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(items[1].clone()),
          },
        ]);
        if let Ok(diff_expr) = diff {
          let diff_eval = crate::evaluator::evaluate_expr_to_expr(&diff_expr);
          if let Ok(ref de) = diff_eval
            && let Some(range) = crate::functions::math_ast::try_eval_to_f64(de)
            && (0.0..10000.0).contains(&range)
          {
            let n_iters = range.floor() as i128 + 1;
            let min_eval = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
            let mut acc = Expr::Integer(0);
            for j in 0..n_iters {
              let iter_val = if j == 0 {
                min_eval.clone()
              } else {
                crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Plus,
                  left: Box::new(min_eval.clone()),
                  right: Box::new(Expr::Integer(j)),
                })?
              };
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &iter_val);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
            }
            return Ok(acc);
          }
        }
      }

      // Try symbolic Sum when bounds are not both concrete integers
      if items.len() == 3 {
        let min_concrete = expr_to_i128(&items[1]);
        let max_concrete = expr_to_i128(&items[2]);
        if min_concrete.is_none() || max_concrete.is_none() {
          if let Some(result) = try_symbolic_sum(
            body,
            &var_name,
            &items[1],
            &items[2],
            min_concrete,
            max_concrete,
          )? {
            // Evaluate to simplify the symbolic result
            return crate::evaluator::evaluate_expr_to_expr(&result);
          }
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: args.to_vec().into(),
          });
        }
      }

      // Extract min, max, step
      let (min, max, step) = if items.len() == 2 {
        let max_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        (1i128, max_val, 1i128)
      } else if items.len() == 3 {
        let min_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let max_val = expr_to_i128(&items[2]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        (min_val, max_val, 1i128)
      } else {
        // items.len() == 4: {i, min, max, step}
        let min_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let max_val = expr_to_i128(&items[2]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let step_val = expr_to_i128(&items[3]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: step must be an integer".into(),
          )
        })?;
        if step_val == 0 {
          return Err(InterpreterError::EvaluationError(
            "Sum: step cannot be zero".into(),
          ));
        }
        (min_val, max_val, step_val)
      };

      // Closed-form fast paths to avoid pathological iteration for large
      // bounds (e.g. Sum[k, {k, 3, 10^20-1, 3}]).
      let n_terms: Option<i128> = if step > 0 && max >= min {
        Some((max - min) / step + 1)
      } else if step < 0 && max <= min {
        Some((min - max) / (-step) + 1)
      } else if min == max {
        Some(1)
      } else {
        // Empty sum
        Some(0)
      };
      if let Some(n) = n_terms {
        if n == 0 {
          return Ok(Expr::Integer(0));
        }
        // Sum[c, {k, ...}] when body doesn't reference k: c * n_terms.
        if !crate::functions::polynomial_ast::contains_var(body, &var_name) {
          let val = crate::evaluator::evaluate_expr_to_expr(body)?;
          return crate::functions::math_ast::times_ast(&[
            val,
            Expr::Integer(n),
          ]);
        }
        // Sum[k, {k, a, b, c}] = n_terms * (first + last) / 2.
        if matches!(body, Expr::Identifier(name) if name == &var_name) {
          let last = min + step * (n - 1);
          let sum_num = num_bigint::BigInt::from(n)
            * (num_bigint::BigInt::from(min) + num_bigint::BigInt::from(last));
          let sum = sum_num / num_bigint::BigInt::from(2);
          return Ok(crate::functions::math_ast::bigint_to_expr(sum));
        }
      }

      let mut acc = Expr::Integer(0);
      let mut i = min;
      if step > 0 {
        while i <= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          i += step;
        }
      } else {
        while i >= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          i += step;
        }
      }
      return Ok(acc);
    }
    _ => {}
  }

  Ok(Expr::FunctionCall {
    name: "Sum".to_string(),
    args: args.to_vec().into(),
  })
}

/// Try to evaluate a known infinite series Sum[body, {var, min, Infinity}].
/// Returns Some(result) if a closed form is found, None otherwise.
/// Try to evaluate a symbolic Sum where at least one bound is not a concrete integer.
/// Returns Some(expr) if a known closed form is found, None otherwise.
/// Flatten the factors of a product, descending through both `Times[...]`
/// (FunctionCall) and `BinaryOp::Times` spellings.
fn collect_times_factors(e: &Expr, out: &mut Vec<Expr>) {
  match e {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      collect_times_factors(left, out);
      collect_times_factors(right, out);
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args.iter() {
        collect_times_factors(a, out);
      }
    }
    _ => out.push(e.clone()),
  }
}

/// If `f` is `base^var` (in either Power spelling), return `base`.
fn power_with_exponent_var(f: &Expr, var_name: &str) -> Option<Expr> {
  match f {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } if matches!(right.as_ref(), Expr::Identifier(s) if s == var_name) => {
      Some((**left).clone())
    }
    Expr::FunctionCall { name, args }
      if name == "Power"
        && args.len() == 2
        && matches!(&args[1], Expr::Identifier(s) if s == var_name) =>
    {
      Some(args[0].clone())
    }
    _ => None,
  }
}

/// Binomial theorem: `Sum[c Binomial[N, k] r^k, {k, 0, N}] = c (1 + r)^N`,
/// where the upper limit equals the Binomial's first argument and `c`, `r` are
/// free of `k`. `r = -1` collapses to `KroneckerDelta[N]` to match Wolfram
/// (e.g. `Sum[(-1)^k Binomial[n, k], {k, 0, n}]`). Returns None when the body
/// is not of this shape.
fn try_binomial_theorem_sum(
  body: &Expr,
  var_name: &str,
  max_expr: &Expr,
) -> Option<Expr> {
  use crate::functions::polynomial_ast::contains_var;
  use crate::syntax::BinaryOperator;

  let mut factors = Vec::new();
  collect_times_factors(body, &mut factors);

  let mut binom_n: Option<Expr> = None;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  let mut r_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    // Binomial[N, k] with N free of k.
    if let Expr::FunctionCall { name, args } = f
      && name == "Binomial"
      && args.len() == 2
      && matches!(&args[1], Expr::Identifier(s) if s == var_name)
      && !contains_var(&args[0], var_name)
    {
      if binom_n.is_some() {
        return None; // only a single Binomial factor is supported
      }
      binom_n = Some(args[0].clone());
      continue;
    }
    // r^k with r free of k.
    if let Some(base) = power_with_exponent_var(f, var_name)
      && !contains_var(&base, var_name)
    {
      r_factors.push(base);
      continue;
    }
    // Plain constant factor (free of k) folds into the coefficient.
    if !contains_var(f, var_name) {
      coeff_factors.push(f.clone());
      continue;
    }
    return None; // a k-dependent factor we cannot fold into the theorem
  }

  let n = binom_n?;
  // A leftover constant coefficient (e.g. `2 Binomial[n, k]`) would give a
  // c (1+r)^N form that Wolfram further folds (2*2^n -> 2^(1+n)); avoid the
  // form divergence by leaving those unevaluated.
  if !coeff_factors.is_empty() {
    return None;
  }
  // The upper limit must be exactly the Binomial's first argument.
  if crate::syntax::expr_to_string(&n)
    != crate::syntax::expr_to_string(max_expr)
  {
    return None;
  }

  // r = product of the r^k bases (default 1 when there is no power factor).
  let r = match r_factors.len() {
    0 => Expr::Integer(1),
    1 => r_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: r_factors.into(),
    },
  };
  let one_plus_r = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(Expr::Integer(1)),
    right: Box::new(r),
  };
  let base = crate::evaluator::evaluate_expr_to_expr(&one_plus_r).ok()?;
  // 1 + r == 0 (r == -1): the alternating sum is KroneckerDelta[N].
  let term = if matches!(base, Expr::Integer(0)) {
    Expr::FunctionCall {
      name: "KroneckerDelta".to_string(),
      args: vec![n].into(),
    }
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base),
      right: Box::new(n),
    }
  };

  Some(term)
}

fn try_symbolic_sum(
  body: &Expr,
  var_name: &str,
  min_expr: &Expr,
  max_expr: &Expr,
  min_concrete: Option<i128>,
  _max_concrete: Option<i128>,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // If body doesn't contain the iteration variable, it's a constant sum:
  // Sum[c, {var, min, max}] = c * (max - min + 1)
  if !crate::functions::polynomial_ast::contains_var(body, var_name) {
    let count = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(max_expr.clone()),
        right: Box::new(min_expr.clone()),
      }),
    };
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(body.clone()),
      right: Box::new(count),
    }));
  }

  // Linearity over a constant factor: Sum[c * f(k), ...] = c * Sum[f(k), ...].
  {
    let factors: Option<Vec<Expr>> = match body {
      Expr::FunctionCall { name, args } if name == "Times" => {
        Some(args.iter().cloned().collect())
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => Some(vec![(**left).clone(), (**right).clone()]),
      // f / c → f * c^-1, so a constant denominator pulls out (Sum[k/2]).
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => Some(vec![
        (**left).clone(),
        Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        },
      ]),
      _ => None,
    };
    if let Some(factors) = factors {
      let (const_factors, var_factors): (Vec<Expr>, Vec<Expr>) =
        factors.into_iter().partition(|f| {
          !crate::functions::polynomial_ast::contains_var(f, var_name)
        });
      if !const_factors.is_empty() && !var_factors.is_empty() {
        let inner_body = if var_factors.len() == 1 {
          var_factors[0].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: var_factors.into(),
          }
        };
        if let Some(inner_sum) = try_symbolic_sum(
          &inner_body,
          var_name,
          min_expr,
          max_expr,
          min_concrete,
          _max_concrete,
        )? {
          let mut all = const_factors;
          all.push(inner_sum);
          let result = Expr::FunctionCall {
            name: "Times".to_string(),
            args: all.into(),
          };
          return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
        }
      }
    }
  }

  // Binomial theorem: Sum[c Binomial[N, k] r^k, {k, 0, N}] = c (1 + r)^N.
  if let Some(0) = min_concrete
    && let Some(result) = try_binomial_theorem_sum(body, var_name, max_expr)
  {
    return Ok(Some(result));
  }

  // Sum[k, {k, 1, n}] = n*(1 + n)/2
  if let Some(1) = min_concrete {
    if matches!(body, Expr::Identifier(name) if name == var_name) {
      let n = max_expr.clone();
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(n.clone()),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(n),
          }),
        }),
        right: Box::new(Expr::Integer(2)),
      }));
    }

    // Sum[k^2, {k, 1, n}] = n*(1 + n)*(1 + 2*n)/6
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(2))
    {
      let n = max_expr.clone();
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(n.clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(n.clone()),
            }),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(n),
            }),
          }),
        }),
        right: Box::new(Expr::Integer(6)),
      }));
    }

    // Sum[k^3, {k, 1, n}] = (n*(1 + n)/2)^2
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(3))
    {
      let n = max_expr.clone();
      // (n*(1+n)/2)^2
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(n.clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(n),
            }),
          }),
          right: Box::new(Expr::Integer(2)),
        }),
        right: Box::new(Expr::Integer(2)),
      }));
    }

    // Sum[k^4, {k, 1, n}] = n*(1+n)*(1+2*n)*(-1+3*n+3*n^2)/30
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(4))
    {
      let n = max_expr.clone();
      // n*(1+n)*(1+2*n)*(-1+3*n+3*n^2)/30
      let n_plus_1 = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(n.clone()),
      };
      let one_plus_2n = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(2)),
          right: Box::new(n.clone()),
        }),
      };
      let neg1_plus_3n_plus_3n2 = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(3)),
            right: Box::new(n.clone()),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(3)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(n.clone()),
              right: Box::new(Expr::Integer(2)),
            }),
          }),
        }),
      };
      let numerator = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![n, n_plus_1, one_plus_2n, neg1_plus_3n_plus_3n2].into(),
      };
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numerator),
        right: Box::new(Expr::Integer(30)),
      }));
    }

    // Sum[k^5, {k, 1, n}] = n^2*(1+n)^2*(-1+2*n+2*n^2)/12
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(5))
    {
      let n = max_expr.clone();
      // n^2*(1+n)^2*(-1+2*n+2*n^2)/12
      let n_sq = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(n.clone()),
        right: Box::new(Expr::Integer(2)),
      };
      let n_plus_1_sq = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(n.clone()),
        }),
        right: Box::new(Expr::Integer(2)),
      };
      let neg1_plus_2n_plus_2n2 = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(n.clone()),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(2)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(n),
              right: Box::new(Expr::Integer(2)),
            }),
          }),
        }),
      };
      let numerator = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![n_sq, n_plus_1_sq, neg1_plus_2n_plus_2n2].into(),
      };
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numerator),
        right: Box::new(Expr::Integer(12)),
      }));
    }

    // Sum[1/k^s, {k, 1, n}] = HarmonicNumber[n, s]
    if let Some(s) = match_reciprocal_power(body, var_name)
      && s >= 1
    {
      return Ok(Some(Expr::FunctionCall {
        name: "HarmonicNumber".to_string(),
        args: if s == 1 {
          vec![max_expr.clone()].into()
        } else {
          vec![max_expr.clone(), Expr::Integer(s as i128)].into()
        },
      }));
    }

    // Sum[c^i, {i, 1, n}] = c*(c^n - 1)/(c - 1) (geometric series)
    // In Divide form: Sum[1/c^i, {i, 1, n}] = (c^n - 1)/(c^n * (c - 1))
    // or equivalently: (-1 + c^n)/c^n
    // Detect body = 1/c^var (Divide or Power with negative exponent)
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } = body
      && matches!(left.as_ref(), Expr::Integer(1))
    {
      // 1 / c^var
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } = right.as_ref()
        && matches!(exp.as_ref(), Expr::Identifier(name) if name == var_name)
      {
        // Sum[1/c^i, {i, 1, n}] = (-1 + c^n)/(c^n*(c-1))
        // For c=2: (-1 + 2^n)/2^n
        let c = base.as_ref();
        let n = max_expr.clone();
        let c_to_n = Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(c.clone()),
          right: Box::new(n),
        };
        return Ok(Some(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(c_to_n.clone()),
          }),
          right: Box::new(c_to_n),
        }));
      }
    }
  }

  // Sum[c^(q*var), {var, 0, n}] = (c^(q*(n+1)) - 1) / (c^q - 1) — generalised
  // geometric series with a coefficient `q` in the exponent (constant w.r.t.
  // `var`). Handles e.g. `Sum[a^(k*n), {k, 0, m-1}]` →
  // `(a^(m*n) - 1) / (a^n - 1)` after simplifying `q*((m-1)+1) = q*m`.
  if matches!(min_concrete, Some(0))
    && let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
    && crate::functions::polynomial_ast::contains_var(exp, var_name)
    && !crate::functions::polynomial_ast::contains_var(base, var_name)
  {
    // Try to write `exp` as `q * var` with `q` constant w.r.t. `var`.
    let q_opt: Option<Expr> = match exp.as_ref() {
      Expr::Identifier(n) if n == var_name => Some(Expr::Integer(1)),
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        let l_is_var =
          matches!(left.as_ref(), Expr::Identifier(n) if n == var_name);
        let r_is_var =
          matches!(right.as_ref(), Expr::Identifier(n) if n == var_name);
        if l_is_var
          && !crate::functions::polynomial_ast::contains_var(right, var_name)
        {
          Some(*right.clone())
        } else if r_is_var
          && !crate::functions::polynomial_ast::contains_var(left, var_name)
        {
          Some(*left.clone())
        } else {
          None
        }
      }
      _ => None,
    };
    if let Some(q) = q_opt {
      let one_plus_max = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(max_expr.clone()),
      };
      let c_to_q = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(*base.clone()),
        right: Box::new(q.clone()),
      };
      let c_to_q_times_top = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(*base.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(q),
          right: Box::new(one_plus_max),
        }),
      };
      let numer = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(c_to_q_times_top),
      };
      let denom = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(c_to_q),
      };
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(numer),
        right: Box::new(denom),
      };
      return Ok(Some(
        crate::evaluator::evaluate_expr_to_expr(&result).unwrap_or(result),
      ));
    }
  }

  // Sum[k, {k, a, n}] where a is symbolic
  if min_concrete.is_none()
    && matches!(body, Expr::Identifier(name) if name == var_name)
  {
    // Sum[k, {k, a, n}] = (a+n)*(n-a+1)/2
    let a = min_expr.clone();
    let n = max_expr.clone();
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(a.clone()),
          right: Box::new(n.clone()),
        }),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(n),
            right: Box::new(a),
          }),
          right: Box::new(Expr::Integer(1)),
        }),
      }),
      right: Box::new(Expr::Integer(2)),
    }));
  }

  Ok(None)
}

/// Check if a body expression matches the Leibniz series: (-1)^k / (2k+1).
/// We verify by evaluating at k=0,1,2,3,4 and checking against expected values.
fn is_leibniz_body(body: &Expr, var_name: &str) -> bool {
  // Expected values: f(0)=1, f(1)=-1/3, f(2)=1/5, f(3)=-1/7, f(4)=1/9
  let expected: [(i128, f64); 5] = [
    (0, 1.0),
    (1, -1.0 / 3.0),
    (2, 1.0 / 5.0),
    (3, -1.0 / 7.0),
    (4, 1.0 / 9.0),
  ];
  for (k, exp_val) in &expected {
    let substituted =
      crate::syntax::substitute_variable(body, var_name, &Expr::Integer(*k));
    if let Ok(result) = crate::evaluator::evaluate_expr_to_expr(&substituted) {
      if let Some(val) = crate::functions::math_ast::try_eval_to_f64(&result) {
        if (val - exp_val).abs() > 1e-12 {
          return false;
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  true
}

/// Match a geometric-series body `c * base^var` where `base` is free of `var`
/// and symbolic (not a plain number), and `var` appears only as the exponent.
/// Returns `(coefficient, base)`.
fn match_geometric_base(body: &Expr, var_name: &str) -> Option<(Expr, Expr)> {
  use crate::syntax::BinaryOperator;

  // Flatten the multiplicative factors of `body`. A `Divide` contributes its
  // denominator as a reciprocal factor `denominator^(-1)`.
  fn collect(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect(left, out);
        collect(right, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        collect(left, out);
        out.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  // Is `f` exactly `base^var` (var only as the exponent)? Returns the base.
  fn power_base<'a>(f: &'a Expr, var_name: &str) -> Option<&'a Expr> {
    let (base, exp) = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (left.as_ref(), right.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return None,
    };
    match exp {
      Expr::Identifier(n) if n == var_name => Some(base),
      _ => None,
    }
  }
  let is_plain_number = |e: &Expr| -> bool {
    matches!(e, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };

  let mut factors = Vec::new();
  collect(body, &mut factors);

  let mut base: Option<&Expr> = None;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    if base.is_none()
      && let Some(b) = power_base(f, var_name)
    {
      base = Some(b);
      continue;
    }
    // Every other factor must be free of the summation variable.
    if !crate::functions::calculus_ast::is_constant_wrt(f, var_name) {
      return None;
    }
    coeff_factors.push(f.clone());
  }

  let base = base?;
  // The base must be free of the variable and genuinely symbolic; a numeric
  // base needs the existing convergence-aware handlers instead.
  if !crate::functions::calculus_ast::is_constant_wrt(base, var_name)
    || is_plain_number(base)
  {
    return None;
  }

  let coeff = match coeff_factors.len() {
    0 => Expr::Integer(1),
    1 => coeff_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff_factors.into(),
    },
  };
  Some((coeff, base.clone()))
}

/// Match a logarithmic-series body `coeff * base^var / var`, i.e. a geometric
/// term over the first power of the summation variable. Returns `(coeff, base)`
/// with `base` the product of all `base^var` factors. Used for the Mercator
/// series Sum[base^k/k, {k,1,Infinity}] = -Log[1 - base].
fn match_log_geometric(body: &Expr, var_name: &str) -> Option<(Expr, Expr)> {
  use crate::syntax::BinaryOperator;
  fn collect(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect(left, out);
        collect(right, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        collect(left, out);
        out.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  fn power_base(f: &Expr, var_name: &str) -> Option<Expr> {
    let (base, exp) = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (left.as_ref().clone(), right.as_ref().clone()),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (args[0].clone(), args[1].clone())
      }
      _ => return None,
    };
    match &exp {
      Expr::Identifier(n) if n == var_name => Some(base),
      _ => None,
    }
  }
  let is_recip_var = |f: &Expr| -> bool {
    let (base, exp) = match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (left.as_ref(), right.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return false,
    };
    matches!(base, Expr::Identifier(n) if n == var_name)
      && matches!(exp, Expr::Integer(-1))
  };

  let mut factors = Vec::new();
  collect(body, &mut factors);
  let mut bases: Vec<Expr> = Vec::new();
  let mut seen_recip_var = false;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Some(b) = power_base(f, var_name) {
      bases.push(b);
      continue;
    }
    if !seen_recip_var && is_recip_var(f) {
      seen_recip_var = true;
      continue;
    }
    if !crate::functions::calculus_ast::is_constant_wrt(f, var_name) {
      return None;
    }
    coeff_factors.push(f.clone());
  }
  if !seen_recip_var || bases.is_empty() {
    return None;
  }
  if bases
    .iter()
    .any(|b| !crate::functions::calculus_ast::is_constant_wrt(b, var_name))
  {
    return None;
  }
  let base = if bases.len() == 1 {
    bases.pop().unwrap()
  } else {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: bases.into(),
    })
    .ok()?
  };
  let coeff = match coeff_factors.len() {
    0 => Expr::Integer(1),
    1 => coeff_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff_factors.into(),
    },
  };
  Some((coeff, base))
}

/// Match an exponential-series body `c * base^var / var!` where `base` is free
/// of `var` (numeric or symbolic) and `var` appears only as the exponent of
/// `base` and inside the factorial. Returns `(coefficient, base)`; the base
/// defaults to `1` when there is no explicit `base^var` factor (e.g. `1/k!`).
fn match_exponential_base(body: &Expr, var_name: &str) -> Option<(Expr, Expr)> {
  use crate::syntax::BinaryOperator;

  fn collect(e: &Expr, out: &mut Vec<Expr>) {
    match e {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        collect(left, out);
        collect(right, out);
      }
      Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left,
        right,
      } => {
        collect(left, out);
        out.push(Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
      }
      Expr::FunctionCall { name, args } if name == "Times" => {
        for a in args.iter() {
          collect(a, out);
        }
      }
      other => out.push(other.clone()),
    }
  }
  // Split a power factor into (base, exponent).
  fn as_power(f: &Expr) -> Option<(&Expr, &Expr)> {
    match f {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => Some((left.as_ref(), right.as_ref())),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((&args[0], &args[1]))
      }
      _ => None,
    }
  }
  let is_factorial_of_var = |e: &Expr| -> bool {
    matches!(e, Expr::FunctionCall { name, args }
      if name == "Factorial" && args.len() == 1
        && matches!(&args[0], Expr::Identifier(n) if n == var_name))
  };

  let mut factors = Vec::new();
  collect(body, &mut factors);

  let mut have_factorial = false;
  let mut base: Option<&Expr> = None;
  let mut coeff_factors: Vec<Expr> = Vec::new();
  for f in &factors {
    if let Some((b, e)) = as_power(f) {
      // The `1/var!` factor.
      if matches!(e, Expr::Integer(-1)) && is_factorial_of_var(b) {
        if have_factorial {
          return None; // more than one factorial factor
        }
        have_factorial = true;
        continue;
      }
      // The `base^var` factor.
      if base.is_none()
        && matches!(e, Expr::Identifier(n) if n == var_name)
        && crate::functions::calculus_ast::is_constant_wrt(b, var_name)
      {
        base = Some(b);
        continue;
      }
    }
    // Any remaining factor must be free of the summation variable.
    if !crate::functions::calculus_ast::is_constant_wrt(f, var_name) {
      return None;
    }
    coeff_factors.push(f.clone());
  }

  if !have_factorial {
    return None;
  }
  let base = base.cloned().unwrap_or(Expr::Integer(1));
  let coeff = match coeff_factors.len() {
    0 => Expr::Integer(1),
    1 => coeff_factors.into_iter().next().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: coeff_factors.into(),
    },
  };
  Some((coeff, base))
}

/// gcd of two non-negative integers.
fn tr_gcd(mut a: i128, mut b: i128) -> i128 {
  a = a.abs();
  b = b.abs();
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Reduce a rational (num, den) to lowest terms with a positive denominator.
fn tr_reduce(n: i128, d: i128) -> (i128, i128) {
  if d == 0 {
    return (n, 0);
  }
  let g = tr_gcd(n, d).max(1);
  let (mut n, mut d) = (n / g, d / g);
  if d < 0 {
    n = -n;
    d = -d;
  }
  (n, d)
}

fn tr_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  tr_reduce(a.0 * b.1 + b.0 * a.1, a.1 * b.1)
}
fn tr_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
  tr_reduce(a.0 * b.0, a.1 * b.1)
}

/// Parse an evaluated expression as an exact rational (n, d).
fn tr_as_rat(e: &Expr) -> Option<(i128, i128)> {
  match e {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(d)) => Some((*n, *d)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// `CoefficientList[op[body], var]` as a vector of exact rationals, or None if
/// the result isn't a list of rational coefficients (e.g. non-polynomial).
fn tr_coeff_list(
  body: &Expr,
  var_name: &str,
  op: &str,
) -> Option<Vec<(i128, i128)>> {
  let part = Expr::FunctionCall {
    name: op.to_string(),
    args: vec![body.clone()].into(),
  };
  let cl = Expr::FunctionCall {
    name: "CoefficientList".to_string(),
    args: vec![part, Expr::Identifier(var_name.to_string())].into(),
  };
  let evaled = crate::evaluator::evaluate_expr_to_expr(&cl).ok()?;
  let Expr::List(items) = &evaled else {
    return None;
  };
  items.iter().map(tr_as_rat).collect()
}

/// Horner evaluation of a polynomial (rational coefficients, ascending degree)
/// at an integer point.
fn tr_poly_eval(coeffs: &[(i128, i128)], x: i128) -> (i128, i128) {
  let mut acc = (0i128, 1i128);
  for c in coeffs.iter().rev() {
    acc = tr_add(tr_mul(acc, (x, 1)), *c);
  }
  acc
}

/// Evaluate the derivative of a polynomial at an integer point.
fn tr_poly_deriv_eval(coeffs: &[(i128, i128)], x: i128) -> (i128, i128) {
  // d/dx sum c_k x^k = sum k c_k x^(k-1).
  let deriv: Vec<(i128, i128)> = coeffs
    .iter()
    .enumerate()
    .skip(1)
    .map(|(k, c)| tr_mul(*c, (k as i128, 1)))
    .collect();
  tr_poly_eval(&deriv, x)
}

/// Exact harmonic number H_a = sum_{k=1}^{a} 1/k (H_0 = 0).
fn tr_harmonic(a: i128) -> (i128, i128) {
  let mut h = (0i128, 1i128);
  for k in 1..=a {
    h = tr_add(h, (1, k));
  }
  h
}

/// Sum of a convergent rational summand with simple integer poles, evaluated
/// in closed form via residues:
/// `Sum[P(n)/Q(n), {n, min, Infinity}] = -sum_r residue_r * H_{min-1-r}`,
/// where r ranges over the (integer, simple) roots of Q. Returns None unless
/// every pole is a simple integer at or below `min-1` and the series converges
/// (the residues sum to zero, i.e. the summand decays like 1/n^2).
fn try_telescoping_rational_sum(
  body: &Expr,
  var_name: &str,
  min: i128,
) -> Result<Option<Expr>, InterpreterError> {
  // Trim trailing zero coefficients to get true degrees.
  let trim = |mut v: Vec<(i128, i128)>| {
    while v.len() > 1 && v.last() == Some(&(0, 1)) {
      v.pop();
    }
    v
  };
  let mut num = tr_coeff_list(body, var_name, "Numerator")
    .map(&trim)
    .unwrap_or_default();
  let mut den = tr_coeff_list(body, var_name, "Denominator")
    .map(&trim)
    .unwrap_or_default();
  // Woxi's Numerator/Denominator don't split a reciprocal power such as
  // Power[n^2 + n, -1]; in that case the numerator comes back empty. Recover
  // the form 1/Q by inverting the body: if body^-1 is a polynomial Q, the
  // summand is 1/Q.
  if num.is_empty() {
    let recip = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![body.clone(), Expr::Integer(-1)].into(),
    };
    if let Some(q) = tr_coeff_list(&recip, var_name, "Together").map(&trim)
      && q.len() > 1
    {
      num = vec![(1, 1)];
      den = q;
    }
  }
  // An empty coefficient list means the numerator/denominator isn't a genuine
  // polynomial in `var`; leave it to other paths.
  if num.is_empty() || den.is_empty() {
    return Ok(None);
  }
  let deg_num = num.len() - 1;
  let deg_den = den.len() - 1;
  // Need a genuine rational function that decays at least like 1/n^2.
  if deg_den < deg_num + 2 || deg_den == 0 {
    return Ok(None);
  }

  // Find the integer roots of Q with multiplicity. A root at 0 shows up as a
  // run of leading zero coefficients; other integer roots divide the lowest
  // non-zero coefficient (rational root theorem, integer candidates only).
  let mut roots: Vec<i128> = Vec::new();
  // Strip integer denominators so the candidate search works on integers.
  let lcm_den = den.iter().fold(1i128, |acc, &(_, d)| {
    let g = tr_gcd(acc, d);
    acc / g * d
  });
  let int_coeffs: Vec<i128> =
    den.iter().map(|&(n, d)| n * (lcm_den / d)).collect();
  // Roots at 0 (trailing zero constant terms).
  let mut lowest = 0usize;
  while lowest < int_coeffs.len() && int_coeffs[lowest] == 0 {
    roots.push(0);
    lowest += 1;
  }
  if lowest >= int_coeffs.len() {
    return Ok(None);
  }
  let reduced = &int_coeffs[lowest..];
  let c0 = reduced[0];
  // Candidate integer roots r divide c0 (leading coefficient need not be 1, but
  // for these telescoping cases the relevant roots are integer divisors of c0).
  let c0a = c0.abs();
  for cand in 1..=c0a {
    if c0a % cand != 0 {
      continue;
    }
    for r in [cand, -cand] {
      // Horner test on the integer polynomial.
      let mut acc = 0i128;
      for c in reduced.iter().rev() {
        acc = acc * r + c;
      }
      if acc == 0 {
        roots.push(r);
      }
    }
  }
  // Every pole must be simple and the roots must account for the full degree
  // (Q factors completely over the integers with no repeats).
  if roots.len() != deg_den {
    return Ok(None);
  }
  let mut sorted = roots.clone();
  sorted.sort_unstable();
  if sorted.windows(2).any(|w| w[0] == w[1]) {
    return Ok(None); // a repeated pole is not handled here
  }
  // No pole may lie inside the summation range [min, Infinity).
  if roots.iter().any(|&r| r > min - 1) {
    return Ok(None);
  }

  let mut total = (0i128, 1i128);
  let mut residue_sum = (0i128, 1i128);
  for &r in &roots {
    let pr = tr_poly_eval(&num, r);
    let dpr = tr_poly_deriv_eval(&den, r);
    if dpr.0 == 0 {
      return Ok(None);
    }
    // residue = P(r) / Q'(r)
    let c = tr_mul(pr, (dpr.1, dpr.0));
    residue_sum = tr_add(residue_sum, c);
    let h = tr_harmonic(min - 1 - r);
    total = tr_add(total, tr_mul((-c.0, c.1), h));
  }
  // Convergence requires the 1/n coefficient (sum of residues) to vanish.
  if residue_sum.0 != 0 {
    return Ok(None);
  }
  let (n, d) = tr_reduce(total.0, total.1);
  Ok(Some(crate::functions::math_ast::make_rational(n, d)))
}

fn try_infinite_sum(
  body: &Expr,
  var_name: &str,
  min: i128,
) -> Result<Option<Expr>, InterpreterError> {
  // Geometric series Sum[c base^var, {var, 0, Infinity}] = c / (1 - base) for
  // a symbolic base. Only the min == 0 form is matched, because wolframscript
  // canonicalizes the min >= 1 result to a different (though equivalent) form.
  if min == 0
    && let Some((coeff, base)) = match_geometric_base(body, var_name)
  {
    use crate::syntax::BinaryOperator;
    let one_minus_base = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(base),
      }),
    };
    let closed = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(coeff),
      right: Box::new(one_minus_base),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&closed)?));
  }

  // Exponential series Sum[c base^var / var!, {var, m, Infinity}]. The base
  // may be numeric or symbolic (the series converges everywhere).
  //   m == 0:                 c E^base
  //   m == 1 with c == 1:     E^base - 1
  // Larger m, or m == 1 with a coefficient, are skipped because
  // wolframscript canonicalizes those results to a different (though
  // equivalent) form.
  if let Some((coeff, base)) = match_exponential_base(body, var_name) {
    let e_to_base = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Constant("E".to_string()), base].into(),
    };
    if min == 0 {
      let result = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![coeff, e_to_base].into(),
      };
      return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
    }
    if min == 1 && matches!(coeff, Expr::Integer(1)) {
      let result = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![e_to_base, Expr::Integer(-1)].into(),
      };
      return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&result)?));
    }
  }

  // Logarithmic series Sum[base^k/k, {k, 1, Infinity}] = -Log[1 - base]
  // (the Mercator/Taylor series for -Log[1-x]). A numeric base needs
  // |base| < 1 to converge; a symbolic base yields the formal result.
  if min == 1
    && let Some((coeff, base)) = match_log_geometric(body, var_name)
  {
    use crate::syntax::BinaryOperator;
    if let Some(b) = crate::functions::math_ast::try_eval_to_f64(&base)
      && !(b.abs() < 1.0)
    {
      return Ok(None);
    }
    // -coeff * Log[1 - base]
    let log_term = Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Integer(1)),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(base),
        }),
      }]
      .into(),
    };
    let closed = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), coeff, log_term].into(),
    };
    return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&closed)?));
  }

  // Try Leibniz formula: Sum[(-1)^k / (2k+1), {k, 0, Infinity}] = Pi/4
  if min == 0 && is_leibniz_body(body, var_name) {
    return Ok(Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Constant("Pi".to_string())),
      right: Box::new(Expr::Integer(4)),
    }));
  }

  // For min < 1, compute initial terms and delegate to min=1 case:
  // Sum[f(n), {n, min, Infinity}] = f(min) + f(min+1) + ... + f(0) + Sum[f(n), {n, 1, Infinity}]
  if min < 1 {
    if let Some(tail_sum) = try_infinite_sum(body, var_name, 1)? {
      let mut acc = tail_sum;
      for k in min..1 {
        let substituted =
          crate::syntax::substitute_variable(body, var_name, &Expr::Integer(k));
        let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
        acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
      }
      return Ok(Some(acc));
    }
    return Ok(None);
  }

  // Convergent rational summand with simple integer poles telescopes to an
  // exact rational, e.g. Sum[1/(n(n+1)), {n, 1, Infinity}] = 1. Handled for
  // any finite lower bound >= 1.
  if let Some(result) = try_telescoping_rational_sum(body, var_name, min)? {
    return Ok(Some(result));
  }

  if min != 1 {
    return Ok(None);
  }

  // Try to detect the pattern 1/var^s (i.e., var^(-s))
  // The body for 1/n^2 is: Times[1, Power[Power[n, 2], -1]]
  // which evaluates/simplifies to Power[n, -2] conceptually,
  // but in practice we need to match the AST structure.
  if let Some(s) = match_reciprocal_power(body, var_name) {
    if s >= 2 && s % 2 == 0 {
      // Zeta(s) for even s: (-1)^(s/2+1) * B_s * (2*Pi)^s / (2 * s!)
      return Ok(Some(zeta_even(s)?));
    }
    // Odd s >= 3: no known closed form in terms of Pi (returns Zeta[s])
    if s >= 3 && s % 2 == 1 {
      return Ok(Some(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: vec![Expr::Integer(s as i128)].into(),
      }));
    }
  }

  // Sum[1/c^i, {i, 1, Infinity}] = 1/(c-1) for integer c > 1
  // Detect body = 1/c^var (Divide form)
  use crate::syntax::BinaryOperator;
  if let Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left,
    right,
  } = body
    && matches!(left.as_ref(), Expr::Integer(1))
    && let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = right.as_ref()
    && matches!(exp.as_ref(), Expr::Identifier(name) if name == var_name)
    && let Some(c) = expr_to_i128(base)
    && c > 1
  {
    // Sum = 1/(c-1)
    return Ok(Some(crate::functions::math_ast::make_rational(1, c - 1)));
  }

  Ok(None)
}

/// Match the pattern `1/var^s` in the body expression.
/// Returns Some(s) if the body is equivalent to var^(-s) with s a positive integer.
fn match_reciprocal_power(body: &Expr, var_name: &str) -> Option<i64> {
  use crate::syntax::BinaryOperator;

  match body {
    // Direct Power[var, -s]
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Identifier(name) = left.as_ref()
        && name == var_name
        && let Some(exp) = get_integer(right)
        && exp < 0
      {
        return Some(-exp as i64);
      }
      // Power[Power[var, s], -1]
      match_power_inverse(body, var_name)
    }
    // Divide[1, Power[var, s]] or Divide[1, var]  (how 1/var^s is stored internally)
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      if is_one(left) {
        // 1 / var^s
        if let Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: base,
          right: exp,
        } = right.as_ref()
          && let Expr::Identifier(name) = base.as_ref()
          && name == var_name
          && let Some(s) = get_integer(exp)
          && s > 0
        {
          return Some(s as i64);
        }
        // 1 / var => s = 1
        if let Expr::Identifier(name) = right.as_ref()
          && name == var_name
        {
          return Some(1);
        }
      }
      None
    }
    // Times[1, Power[Power[var, s], -1]]  (FullForm representation)
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_one(left) {
        return match_power_inverse(right, var_name);
      }
      if is_one(right) {
        return match_power_inverse(left, var_name);
      }
      None
    }
    _ => match_power_inverse(body, var_name),
  }
}

/// Match Power[Power[var, s], -1] or Power[var, -s]
fn match_power_inverse(expr: &Expr, var_name: &str) -> Option<i64> {
  use crate::syntax::BinaryOperator;

  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      // Power[something, -1] where something = Power[var, s]
      if let Some(-1) = get_integer(right) {
        if let Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: inner_left,
          right: inner_right,
        } = left.as_ref()
          && let Expr::Identifier(name) = inner_left.as_ref()
          && name == var_name
          && let Some(s) = get_integer(inner_right)
          && s > 0
        {
          return Some(s as i64);
        }
        // Power[var, -1] => s = 1
        if let Expr::Identifier(name) = left.as_ref()
          && name == var_name
        {
          return Some(1);
        }
      }
      // Power[var, -s] directly
      if let Expr::Identifier(name) = left.as_ref()
        && name == var_name
        && let Some(exp) = get_integer(right)
        && exp < 0
      {
        return Some(-exp as i64);
      }
      None
    }
    _ => None,
  }
}

/// Get an integer value from an Expr
fn get_integer(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i128()
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Integer(n) => Some(-n),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        (-n).to_i128()
      }
      _ => None,
    },
    _ => None,
  }
}

fn is_one(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(1))
    || matches!(expr, Expr::BigInteger(n) if *n == num_bigint::BigInt::from(1))
}

/// Compute ζ(2k) = |B_{2k}| * (2π)^{2k} / (2 * (2k)!) as a symbolic expression.
/// Returns Pi^(2k) * rational_coefficient.
fn zeta_even(s: i64) -> Result<Expr, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Get B_s using bernoulli_b_ast
  let b_s =
    crate::functions::math_ast::bernoulli_b_ast(&[Expr::Integer(s as i128)])?;

  // Extract the rational value of B_s as (num, den)
  let (b_num, b_den) = match &b_s {
    Expr::Integer(n) => (*n, 1i128),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      match n.to_i128() {
        Some(v) => (v, 1i128),
        None => {
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: vec![].into(),
          });
        }
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
        (Some(n), Some(d)) => (n, d),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: vec![].into(),
          });
        }
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Sum".to_string(),
        args: vec![].into(),
      });
    }
  };

  // ζ(s) = (-1)^(s/2+1) * B_s * (2π)^s / (2 * s!)
  // Since B_s for even s alternates sign: B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, ...
  // (-1)^(s/2+1) * B_s = |B_s| always positive
  // So ζ(s) = |B_s| * (2π)^s / (2 * s!)

  // Compute (2^s) * |B_s_num| / (2 * s! * |B_s_den|)
  // = 2^(s-1) * |B_s_num| / (s! * |B_s_den|)
  let abs_b_num = b_num.abs();

  // Compute 2^(s-1) and s!
  let two_pow = 2i128.checked_pow((s - 1) as u32).unwrap_or(i128::MAX);
  let mut factorial: i128 = 1;
  for i in 2..=s as i128 {
    factorial = factorial.checked_mul(i).unwrap_or(i128::MAX);
  }

  // The coefficient of Pi^s is: 2^(s-1) * |B_s_num| / (s! * B_s_den)
  let coeff_num = two_pow * abs_b_num;
  let coeff_den = factorial * b_den.abs();

  // Simplify the fraction
  let g = gcd_i128(coeff_num.abs(), coeff_den.abs());
  let final_num = coeff_num / g;
  let final_den = coeff_den / g;

  // Build the expression: (final_num / final_den) * Pi^s
  let pi_power = if s == 1 {
    Expr::Identifier("Pi".to_string())
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Identifier("Pi".to_string())),
      right: Box::new(Expr::Integer(s as i128)),
    }
  };

  if final_num == 1 && final_den == 1 {
    Ok(pi_power)
  } else if final_den == 1 {
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(final_num)),
      right: Box::new(pi_power),
    })
  } else if final_num == 1 {
    // 1/d * Pi^s => Pi^s / d
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(pi_power),
      right: Box::new(Expr::Integer(final_den)),
    })
  } else {
    // n/d * Pi^s => (n * Pi^s) / d
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(final_num)),
        right: Box::new(pi_power),
      }),
      right: Box::new(Expr::Integer(final_den)),
    })
  }
}
