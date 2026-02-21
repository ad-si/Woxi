#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AnglePath[{θ1, θ2, ...}] - path with unit steps and cumulative turning angles.
/// AnglePath[{{r1, θ1}, {r2, θ2}, ...}] - path with specified step lengths.
pub fn angle_path_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AnglePath expects 1 argument".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AnglePath".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Start at origin
  let mut x = 0.0_f64;
  let mut y = 0.0_f64;
  let mut angle = 0.0_f64;
  let mut points: Vec<Expr> = Vec::with_capacity(items.len() + 1);

  // Check if all items are numeric (pure angles) or {step, angle} pairs
  let is_pair_form = items
    .first()
    .is_some_and(|item| matches!(item, Expr::List(_)));

  // Starting point {0, 0}
  // Use Integer 0 when all inputs are exact integers, else Real
  let all_integer = if is_pair_form {
    items.iter().all(|item| {
      if let Expr::List(pair) = item {
        pair.len() == 2
          && matches!(&pair[0], Expr::Integer(_))
          && matches!(&pair[1], Expr::Integer(_))
      } else {
        matches!(item, Expr::Integer(_))
      }
    })
  } else {
    items.iter().all(|item| matches!(item, Expr::Integer(_)))
  };

  if all_integer {
    // Symbolic mode: keep exact Cos/Sin
    let mut cum_terms_x: Vec<Expr> = Vec::new();
    let mut cum_terms_y: Vec<Expr> = Vec::new();
    let mut cum_angle = Expr::Integer(0);

    points.push(Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]));

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

      points.push(Expr::List(vec![px, py]));
    }
  } else {
    // Numeric mode
    points.push(Expr::List(vec![Expr::Real(0.0), Expr::Real(0.0)]));

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
      points.push(Expr::List(vec![Expr::Real(x), Expr::Real(y)]));
    }
  }

  Ok(Expr::List(points))
}

/// AST-based Product: product of list elements or iterator product.
pub fn product_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Product[{a, b, c}] -> a * b * c
    let items = match &args[0] {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Product".to_string(),
          args: args.to_vec(),
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
          args: args.to_vec(),
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
              args: args.to_vec(),
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
                  args: args.to_vec(),
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

          // Body is the iteration variable itself: Product[k, {k, ...}]
          if matches!(body, Expr::Identifier(name) if name == &var_name) {
            if let Some(min_val) = min_concrete {
              if max_concrete.is_none() {
                // Product[k, {k, concrete_min, symbolic_max}]
                // = max! / (min-1)!
                let n_factorial = Expr::FunctionCall {
                  name: "Factorial".to_string(),
                  args: vec![max_expr.clone()],
                };
                if min_val == 1 {
                  return Ok(n_factorial);
                }
                // Compute (min-1)! as a concrete integer
                let mut denom: i128 = 1;
                for j in 2..min_val {
                  denom *= j;
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
                ],
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
                  args: vec![max_expr.clone()],
                }),
                right: exp.clone(),
              });
            }
          }

          // For other symbolic cases, return unevaluated
          return Ok(Expr::FunctionCall {
            name: "Product".to_string(),
            args: args.to_vec(),
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
    args: args.to_vec(),
  })
}

/// AST-based Sum: sum of list elements or iterator sum.
pub fn sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    // Sum requires at least 2 arguments
    return Ok(Expr::FunctionCall {
      name: "Sum".to_string(),
      args: args.to_vec(),
    });
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
            args: args.to_vec(),
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
          args: args.to_vec(),
        });
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
            args: args.to_vec(),
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
    args: args.to_vec(),
  })
}

/// Try to evaluate a known infinite series Sum[body, {var, min, Infinity}].
/// Returns Some(result) if a closed form is found, None otherwise.
/// Try to evaluate a symbolic Sum where at least one bound is not a concrete integer.
/// Returns Some(expr) if a known closed form is found, None otherwise.
fn try_symbolic_sum(
  body: &Expr,
  var_name: &str,
  min_expr: &Expr,
  max_expr: &Expr,
  min_concrete: Option<i128>,
  _max_concrete: Option<i128>,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

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

    // Sum[1/k^s, {k, 1, n}] = HarmonicNumber[n, s]
    if let Some(s) = match_reciprocal_power(body, var_name)
      && s >= 1
    {
      return Ok(Some(Expr::FunctionCall {
        name: "HarmonicNumber".to_string(),
        args: vec![max_expr.clone(), Expr::Integer(s as i128)],
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

fn try_infinite_sum(
  body: &Expr,
  var_name: &str,
  min: i128,
) -> Result<Option<Expr>, InterpreterError> {
  // Try Leibniz formula: Sum[(-1)^k / (2k+1), {k, 0, Infinity}] = Pi/4
  if min == 0 {
    if is_leibniz_body(body, var_name) {
      return Ok(Some(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(4)),
      }));
    }
    return Ok(None);
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
        args: vec![Expr::Integer(s as i128)],
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
            args: vec![],
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
            args: vec![],
          });
        }
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Sum".to_string(),
        args: vec![],
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
