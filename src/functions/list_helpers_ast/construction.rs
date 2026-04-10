#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based Table: generate a table of values.
/// Table[expr, {i, min, max}] -> {expr with i=min, ..., expr with i=max}
/// Table[expr, {i, max}] -> {expr with i=1, ..., expr with i=max}
/// Table[expr, {i, {list}}] -> {expr with i=elem1, expr with i=elem2, ...}
/// Table[expr, n] -> {expr, expr, ..., expr} (n times)
/// Multi-dimensional Table: Table[expr, iter1, iter2, ...]
/// Recursively nests Table from last iterator to first.
pub fn table_multi_ast(
  body: &Expr,
  iters: &[Expr],
) -> Result<Expr, InterpreterError> {
  if iters.len() == 1 {
    return table_ast(body, &iters[0]);
  }
  // Nest: the body for the outer iterator is Table[body, inner_iters...]
  // We build from the innermost outward
  let inner_body = Expr::FunctionCall {
    name: "Table".to_string(),
    args: std::iter::once(body.clone())
      .chain(iters[1..].iter().cloned())
      .collect(),
  };
  table_ast(&inner_body, &iters[0])
}

pub fn table_ast(
  body: &Expr,
  iter_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  match iter_spec {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      // Simple form: Table[expr, n]
      let n = expr_to_i128(iter_spec).ok_or_else(|| {
        InterpreterError::EvaluationError("Table: count too large".into())
      })?;
      if n < 0 {
        return Err(InterpreterError::EvaluationError(
          "Table: count must be non-negative".into(),
        ));
      }
      let mut results = Vec::new();
      for _ in 0..n {
        let val = crate::evaluator::evaluate_expr_to_expr(body)?;
        results.push(val);
      }
      Ok(Expr::List(results))
    }
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::List(vec![]));
      }

      // Handle {n} form (single element = just repeat count, no variable)
      if items.len() == 1 {
        let evaluated = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
        let n = expr_to_i128(&evaluated).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Table: iterator bound must be an integer".into(),
          )
        })?;
        let mut results = Vec::new();
        for _ in 0..n {
          let val = crate::evaluator::evaluate_expr_to_expr(body)?;
          results.push(val);
        }
        return Ok(Expr::List(results));
      }

      // Extract iterator variable
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Table: iterator variable must be an identifier".into(),
          ));
        }
      };

      if items.len() == 2 {
        // Check if second element is a list (iterate over list)
        let mut second = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        match &mut second {
          Expr::List(list_items) => {
            // {i, {a, b, c}} form - iterate over list elements
            let mut results = Vec::new();
            for item in list_items.iter() {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, item);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
            }
            return Ok(Expr::List(results));
          }
          _ => {
            // {i, max} form - iterate from 1 to max
            let max_val = expr_to_i128(&second).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "Table: iterator bound must be an integer".into(),
              )
            })?;
            let mut results = Vec::new();
            for i in 1..=max_val {
              let substituted = crate::syntax::substitute_variable(
                body,
                &var_name,
                &Expr::Integer(i),
              );
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
            }
            return Ok(Expr::List(results));
          }
        }
      } else if items.len() >= 3 {
        // {i, min, max} or {i, min, max, step} form
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;

        // Get step (default is 1)
        let step_expr = if items.len() >= 4 {
          crate::evaluator::evaluate_expr_to_expr(&items[3])?
        } else {
          Expr::Integer(1)
        };

        // Keep exact integer iteration behavior when possible.
        if let (Some(min_val), Some(max_val), Some(step_val)) = (
          expr_to_i128(&min_expr),
          expr_to_i128(&max_expr),
          expr_to_i128(&step_expr),
        ) {
          if step_val == 0 {
            return Err(InterpreterError::EvaluationError(
              "Table: step cannot be zero".into(),
            ));
          }

          let mut results = Vec::new();
          let mut i = min_val;
          if step_val > 0 {
            while i <= max_val {
              let substituted = crate::syntax::substitute_variable(
                body,
                &var_name,
                &Expr::Integer(i),
              );
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
              i += step_val;
            }
          } else {
            // Negative step
            while i >= max_val {
              let substituted = crate::syntax::substitute_variable(
                body,
                &var_name,
                &Expr::Integer(i),
              );
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
              i += step_val;
            }
          }
          return Ok(Expr::List(results));
        }

        // Fallback: numeric iteration for symbolic numeric bounds/step (e.g. Pi/4).
        crate::functions::math_ast::try_eval_to_f64(&min_expr).ok_or_else(
          || {
            InterpreterError::EvaluationError(
              "Table: iterator bound must be numeric".into(),
            )
          },
        )?;
        let max_num = crate::functions::math_ast::try_eval_to_f64(&max_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Table: iterator bound must be numeric".into(),
            )
          })?;
        let step_num = crate::functions::math_ast::try_eval_to_f64(&step_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Table: step must be numeric".into(),
            )
          })?;

        if step_num.abs() <= f64::EPSILON {
          return Err(InterpreterError::EvaluationError(
            "Table: step cannot be zero".into(),
          ));
        }

        let mut results = Vec::new();
        let mut current_expr = min_expr.clone();
        let mut safety_counter: usize = 0;
        if step_num > 0.0 {
          loop {
            let current_num =
              crate::functions::math_ast::try_eval_to_f64(&current_expr)
                .ok_or_else(|| {
                  InterpreterError::EvaluationError(
                    "Table: iterator value became non-numeric".into(),
                  )
                })?;
            if current_num > max_num + f64::EPSILON {
              break;
            }
            let substituted = crate::syntax::substitute_variable(
              body,
              &var_name,
              &current_expr,
            );
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            results.push(val);
            current_expr =
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![current_expr, step_expr.clone()],
              })?;
            safety_counter += 1;
            if safety_counter > 1_000_000 {
              return Err(InterpreterError::EvaluationError(
                "Table: iterator exceeded maximum iterations".into(),
              ));
            }
          }
        } else {
          loop {
            let current_num =
              crate::functions::math_ast::try_eval_to_f64(&current_expr)
                .ok_or_else(|| {
                  InterpreterError::EvaluationError(
                    "Table: iterator value became non-numeric".into(),
                  )
                })?;
            if current_num < max_num - f64::EPSILON {
              break;
            }
            let substituted = crate::syntax::substitute_variable(
              body,
              &var_name,
              &current_expr,
            );
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            results.push(val);
            current_expr =
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![current_expr, step_expr.clone()],
              })?;
            safety_counter += 1;
            if safety_counter > 1_000_000 {
              return Err(InterpreterError::EvaluationError(
                "Table: iterator exceeded maximum iterations".into(),
              ));
            }
          }
        }
        return Ok(Expr::List(results));
      }

      Err(InterpreterError::EvaluationError(
        "Table: invalid iterator specification".into(),
      ))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Table: invalid iterator specification".into(),
    )),
  }
}

/// Extract a rational (numerator, denominator) from an Expr.
/// Returns Some((n, d)) for Integer, Rational, None for anything else.
fn expr_to_rational(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// AST-based Range: generate a range of numbers.
/// Range[n] -> {1, 2, ..., n}
/// Range[min, max] -> {min, ..., max}
/// Range[min, max, step] -> {min, min+step, ..., max}
pub fn range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "Range".to_string(),
      args: args.to_vec(),
    });
  }

  // Try exact rational arithmetic first
  let all_rational = args.iter().all(|a| expr_to_rational(a).is_some());

  if all_rational {
    let (min_n, min_d, max_n, max_d, step_n, step_d) = if args.len() == 1 {
      let (mn, md) = expr_to_rational(&args[0]).unwrap();
      (1_i128, 1_i128, mn, md, 1_i128, 1_i128)
    } else if args.len() == 2 {
      let (min_n, min_d) = expr_to_rational(&args[0]).unwrap();
      let (max_n, max_d) = expr_to_rational(&args[1]).unwrap();
      (min_n, min_d, max_n, max_d, 1_i128, 1_i128)
    } else {
      let (min_n, min_d) = expr_to_rational(&args[0]).unwrap();
      let (max_n, max_d) = expr_to_rational(&args[1]).unwrap();
      let (step_n, step_d) = expr_to_rational(&args[2]).unwrap();
      (min_n, min_d, max_n, max_d, step_n, step_d)
    };

    if step_n == 0 {
      return Err(InterpreterError::EvaluationError(
        "Range: step cannot be zero".into(),
      ));
    }

    let mut results = Vec::new();
    // Current value as (numerator, denominator)
    let mut cur_n = min_n;
    let mut cur_d = min_d;
    let step_positive = (step_n > 0) == (step_d > 0);

    loop {
      // Compare cur vs max: cur_n/cur_d vs max_n/max_d
      // cur_n * max_d vs max_n * cur_d (careful with sign of denominators)
      let lhs = cur_n * max_d;
      let rhs = max_n * cur_d;
      let denom_sign = (cur_d > 0) == (max_d > 0);

      if step_positive {
        // For positive step: stop when cur > max
        if denom_sign && lhs > rhs {
          break;
        }
        if !denom_sign && lhs < rhs {
          break;
        }
      } else {
        // For negative step: stop when cur < max
        if denom_sign && lhs < rhs {
          break;
        }
        if !denom_sign && lhs > rhs {
          break;
        }
      }

      results.push(crate::functions::math_ast::make_rational_pub(cur_n, cur_d));

      // cur += step: cur_n/cur_d + step_n/step_d = (cur_n*step_d + step_n*cur_d) / (cur_d*step_d)
      cur_n = cur_n * step_d + step_n * cur_d;
      cur_d *= step_d;

      // Simplify to avoid overflow
      let g = gcd_i128(cur_n.abs(), cur_d.abs());
      if g > 1 {
        cur_n /= g;
        cur_d /= g;
      }

      if results.len() > 1_000_000 {
        return Err(InterpreterError::EvaluationError(
          "Range: result too large".into(),
        ));
      }
    }

    return Ok(Expr::List(results));
  }

  // Try symbolic range when arguments contain symbolic constants (e.g. Pi)
  let try_numeric = |e: &Expr| -> Option<f64> {
    if let Some(f) = expr_to_f64(e) {
      return Some(f);
    }
    // Try evaluating N[expr] to get a float
    let n_expr = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![e.clone()],
    };
    if let Ok(evaled) = crate::evaluator::evaluate_expr_to_expr(&n_expr) {
      return expr_to_f64(&evaled);
    }
    None
  };

  let (min_expr, max_expr, step_expr) = if args.len() == 1 {
    (Expr::Integer(1), args[0].clone(), Expr::Integer(1))
  } else if args.len() == 2 {
    (args[0].clone(), args[1].clone(), Expr::Integer(1))
  } else {
    (args[0].clone(), args[1].clone(), args[2].clone())
  };

  // Check if any arg is non-numeric (symbolic)
  let has_symbolic = expr_to_f64(&min_expr).is_none()
    || expr_to_f64(&max_expr).is_none()
    || expr_to_f64(&step_expr).is_none();

  if has_symbolic {
    // Try to evaluate numerically to determine count
    let min_f = try_numeric(&min_expr);
    let max_f = try_numeric(&max_expr);
    let step_f = try_numeric(&step_expr);

    if let (Some(min_val), Some(max_val), Some(step_val)) =
      (min_f, max_f, step_f)
    {
      if step_val == 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Range: step cannot be zero".into(),
        ));
      }
      let count = ((max_val - min_val) / step_val).floor() as i128 + 1;
      if count <= 0 {
        return Ok(Expr::List(vec![]));
      }
      if count > 1_000_000 {
        return Err(InterpreterError::EvaluationError(
          "Range: result too large".into(),
        ));
      }
      let mut results = Vec::with_capacity(count as usize);
      for k in 0..count {
        // Generate min_expr + k * step_expr symbolically
        let elem = if k == 0 {
          min_expr.clone()
        } else {
          let k_times_step = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(k)),
            right: Box::new(step_expr.clone()),
          };
          let sum = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(min_expr.clone()),
            right: Box::new(k_times_step),
          };
          crate::evaluator::evaluate_expr_to_expr(&sum)?
        };
        results.push(elem);
      }
      return Ok(Expr::List(results));
    }

    return Err(InterpreterError::EvaluationError(
      "Range: argument must be numeric".into(),
    ));
  }

  // Fallback to f64 for Real arguments
  let (min, max, step) = (
    expr_to_f64(&min_expr).unwrap(),
    expr_to_f64(&max_expr).unwrap(),
    expr_to_f64(&step_expr).unwrap(),
  );

  if step == 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Range: step cannot be zero".into(),
    ));
  }

  // Check if any input is Real - if so, all outputs should be Real
  let any_real = args.iter().any(|a| matches!(a, Expr::Real(_)));

  let mut results = Vec::new();
  let mut val = min;
  if step > 0.0 {
    while val <= max + f64::EPSILON {
      results.push(if any_real {
        Expr::Real(val)
      } else {
        f64_to_expr(val)
      });
      val += step;
    }
  } else {
    while val >= max - f64::EPSILON {
      results.push(if any_real {
        Expr::Real(val)
      } else {
        f64_to_expr(val)
      });
      val += step;
    }
  }

  Ok(Expr::List(results))
}

/// PowerRange[min, max] generates {min, min*10, min*100, ...} up to max.
/// PowerRange[min, max, r] uses factor r instead of 10.
pub fn power_range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "PowerRange".to_string(),
      args: args.to_vec(),
    });
  }

  // Try rational arithmetic
  let all_rational = args.iter().all(|a| expr_to_rational(a).is_some());

  if all_rational {
    let (min_n, min_d) = expr_to_rational(&args[0]).unwrap();
    let (max_n, max_d) = expr_to_rational(&args[1]).unwrap();
    let (fac_n, fac_d) = if args.len() == 3 {
      expr_to_rational(&args[2]).unwrap()
    } else {
      (10, 1)
    };

    if fac_n == 0 {
      return Err(InterpreterError::EvaluationError(
        "PowerRange: factor cannot be zero".into(),
      ));
    }

    let mut results = Vec::new();
    let mut cur_n = min_n;
    let mut cur_d = min_d;

    // Compare min vs max to determine direction
    // min_val = min_n / min_d, max_val = max_n / max_d
    let min_cmp_max = (min_n * max_d).cmp(&(max_n * min_d));
    // Adjust for negative denominators
    let min_cmp_max = if (min_d > 0) != (max_d > 0) {
      min_cmp_max.reverse()
    } else {
      min_cmp_max
    };
    let growing = matches!(
      min_cmp_max,
      std::cmp::Ordering::Less | std::cmp::Ordering::Equal
    );

    loop {
      // Compare cur vs max
      let cmp_val = cur_n * max_d;
      let cmp_ref = max_n * cur_d;
      let same_sign = (cur_d > 0) == (max_d > 0);

      let past_max = if growing {
        if same_sign {
          cmp_val > cmp_ref
        } else {
          cmp_val < cmp_ref
        }
      } else if same_sign {
        cmp_val < cmp_ref
      } else {
        cmp_val > cmp_ref
      };

      if past_max {
        break;
      }

      results.push(crate::functions::math_ast::make_rational_pub(cur_n, cur_d));

      // cur *= factor: (cur_n/cur_d) * (fac_n/fac_d) = (cur_n*fac_n) / (cur_d*fac_d)
      cur_n *= fac_n;
      cur_d *= fac_d;

      // Simplify
      let g = gcd_i128(cur_n.abs(), cur_d.abs());
      if g > 1 {
        cur_n /= g;
        cur_d /= g;
      }

      if results.len() > 1_000_000 {
        return Err(InterpreterError::EvaluationError(
          "PowerRange: result too large".into(),
        ));
      }
    }

    return Ok(Expr::List(results));
  }

  Ok(Expr::FunctionCall {
    name: "PowerRange".to_string(),
    args: args.to_vec(),
  })
}

/// AST-based ConstantArray: create array filled with constant.
/// ConstantArray[c, n] -> {c, c, ..., c} (n times)
/// ConstantArray[c, {n1, n2}] -> nested array
pub fn constant_array_ast(
  elem: &Expr,
  dims: &Expr,
) -> Result<Expr, InterpreterError> {
  match dims {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let n = expr_to_i128(dims).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "ConstantArray: dimension too large".into(),
        )
      })?;
      if n < 0 {
        return Err(InterpreterError::EvaluationError(
          "ConstantArray: dimension must be non-negative".into(),
        ));
      }
      Ok(Expr::List(vec![elem.clone(); n as usize]))
    }
    Expr::List(dim_list) => {
      if dim_list.is_empty() {
        return Ok(elem.clone());
      }
      let first_dim = expr_to_i128(&dim_list[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "ConstantArray: dimensions must be integers".into(),
        )
      })?;
      if dim_list.len() == 1 {
        Ok(Expr::List(vec![elem.clone(); first_dim as usize]))
      } else {
        let rest_dims = Expr::List(dim_list[1..].to_vec());
        let inner = constant_array_ast(elem, &rest_dims)?;
        Ok(Expr::List(vec![inner; first_dim as usize]))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "ConstantArray".to_string(),
      args: vec![elem.clone(), dims.clone()],
    }),
  }
}

/// AST-based Do: execute expression multiple times.
/// Do[expr, n] -> execute expr n times
/// Do[expr, {i, max}] -> execute with i from 1 to max
/// Do[expr, {i, min, max}] -> execute with i from min to max
pub fn do_ast(body: &Expr, iter_spec: &Expr) -> Result<Expr, InterpreterError> {
  match iter_spec {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let n = expr_to_i128(iter_spec).unwrap_or(0);
      for _ in 0..n {
        match crate::evaluator::evaluate_expr_to_expr(body) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
          Err(e) => return Err(e),
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    Expr::List(items) if items.len() == 1 => {
      // Do[body, {n}] — repeat n times without iterator variable
      let n_expr = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
      let n = expr_to_i128(&n_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Do: repeat count must be an integer".into(),
        )
      })?;
      for _ in 0..n {
        match crate::evaluator::evaluate_expr_to_expr(body) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
          Err(e) => return Err(e),
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    Expr::List(items) if items.len() >= 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Do: iterator variable must be an identifier".into(),
          ));
        }
      };

      // Handle list iterator: Do[body, {i, {a, b, c}}]
      if items.len() == 2 {
        let val_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        if let Expr::List(list_items) = &val_expr {
          for item in list_items {
            let substituted =
              crate::syntax::substitute_variable(body, &var_name, item);
            match crate::evaluator::evaluate_expr_to_expr(&substituted) {
              Ok(_) => {}
              Err(InterpreterError::BreakSignal) => break,
              Err(InterpreterError::ContinueSignal) => {}
              Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
              Err(e) => return Err(e),
            }
          }
          return Ok(Expr::Identifier("Null".to_string()));
        }
      }

      let (min, max, step) = if items.len() == 2 {
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_val = expr_to_i128(&max_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        (1i128, max_val, 1i128)
      } else if items.len() >= 3 {
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
        let min_val = expr_to_i128(&min_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        let max_val = expr_to_i128(&max_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        let step_val = if items.len() >= 4 {
          let step_expr = crate::evaluator::evaluate_expr_to_expr(&items[3])?;
          expr_to_i128(&step_expr).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: step must be an integer".into(),
            )
          })?
        } else {
          1i128
        };
        (min_val, max_val, step_val)
      } else {
        return Err(InterpreterError::EvaluationError(
          "Do: invalid iterator specification".into(),
        ));
      };

      if step == 0 {
        return Err(InterpreterError::EvaluationError(
          "Do: step cannot be zero".into(),
        ));
      }

      let mut i = min;
      if step > 0 {
        while i <= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          match crate::evaluator::evaluate_expr_to_expr(&substituted) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
            Err(e) => return Err(e),
          }
          i += step;
        }
      } else {
        while i >= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          match crate::evaluator::evaluate_expr_to_expr(&substituted) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
            Err(e) => return Err(e),
          }
          i += step;
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Do: invalid iterator specification".into(),
    )),
  }
}

/// Array[f, n] - creates a list by applying f to indices 1..n
pub fn array_ast(func: &Expr, n: i128) -> Result<Expr, InterpreterError> {
  let mut result = Vec::new();
  for i in 1..=n {
    let arg = Expr::Integer(i);
    let val = apply_func_ast(func, &arg)?;
    result.push(val);
  }
  Ok(Expr::List(result))
}

/// Array[f, {n1, n2, ...}] - multi-dimensional array
/// Array[f, dims, origin] - with custom origin
/// Array[f, dims, origin, head] - with custom head instead of List
pub fn array_multi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let func = &args[0];
  let dims_expr = &args[1];

  let dims: Vec<i128> = match dims_expr {
    Expr::List(items) => {
      let mut d = Vec::new();
      for item in items {
        if let Some(n) = expr_to_i128(item) {
          d.push(n);
        } else {
          return Ok(Expr::FunctionCall {
            name: "Array".to_string(),
            args: args.to_vec(),
          });
        }
      }
      d
    }
    _ => {
      if let Some(n) = expr_to_i128(dims_expr) {
        vec![n]
      } else {
        return Ok(Expr::FunctionCall {
          name: "Array".to_string(),
          args: args.to_vec(),
        });
      }
    }
  };

  // Parse per-dimension index ranges. Each entry is (start, step).
  // Wolfram supports several forms for the origin/range argument:
  //   Array[f, n]                  → indices 1..n        (start=1, step=1)
  //   Array[f, n, r]               → indices r..r+n-1    (start=r, step=1)
  //   Array[f, n, {a, b}]          → n values from a..b  (start=a, step=(b-a)/(n-1))
  //   Array[f, {n1,...}, {r1,...}] → per-dim origin
  //   Array[f, {n1,...}, {{a1,b1},...}] → per-dim range
  let ranges: Vec<(f64, f64)> = if args.len() >= 3 {
    // Determine if the origin arg is a per-dimension list or a single spec.
    // For a 1-D array (dims.len()==1), a list of length 2 with integer
    // elements is ambiguous: treat it as {a, b} (range) rather than two
    // per-dim origins, matching Wolfram's behaviour.
    let origin = &args[2];
    match origin {
      Expr::List(items)
        if dims.len() == 1
          && items.len() == 2
          && items.iter().all(|i| expr_to_i128(i).is_some()) =>
      {
        // Single-dim {a, b} range form
        let a = expr_to_i128(&items[0]).unwrap() as f64;
        let b = expr_to_i128(&items[1]).unwrap() as f64;
        let n = dims[0];
        let step = if n > 1 {
          (b - a) / ((n - 1) as f64)
        } else {
          0.0
        };
        vec![(a, step)]
      }
      Expr::List(items) => items
        .iter()
        .map(|item| {
          // Each item can be an integer (start) or a 2-list {a, b}
          if let Some(r) = expr_to_i128(item) {
            (r as f64, 1.0)
          } else if let Expr::List(pair) = item {
            if pair.len() == 2 {
              let a = expr_to_i128(&pair[0]).unwrap_or(1) as f64;
              let b = expr_to_i128(&pair[1]).unwrap_or(1) as f64;
              (a, b - a) // step calculated later using dims
            } else {
              (1.0, 1.0)
            }
          } else {
            (1.0, 1.0)
          }
        })
        .enumerate()
        .map(|(i, (start, raw_step))| {
          // If the item was a pair, raw_step is (b - a); convert to per-index step
          if let Some(Expr::List(pair)) = items.get(i)
            && pair.len() == 2
            && expr_to_i128(&pair[0]).is_some()
          {
            let n = dims[i];
            let step = if n > 1 {
              raw_step / ((n - 1) as f64)
            } else {
              0.0
            };
            return (start, step);
          }
          (start, 1.0)
        })
        .collect(),
      _ => {
        let o = expr_to_i128(origin).unwrap_or(1) as f64;
        vec![(o, 1.0); dims.len()]
      }
    }
  } else {
    vec![(1.0, 1.0); dims.len()]
  };

  // Parse optional head (4th arg)
  let head: Option<&str> = if args.len() >= 4 {
    match &args[3] {
      Expr::Identifier(h) => Some(h.as_str()),
      _ => None,
    }
  } else {
    None
  };

  // Build the array recursively.
  // `ranges[depth]` is (start, step) — index values are start, start+step, ...
  fn build_array(
    func: &Expr,
    dims: &[i128],
    ranges: &[(f64, f64)],
    depth: usize,
    indices: &mut Vec<Expr>,
  ) -> Result<Expr, InterpreterError> {
    if depth >= dims.len() {
      let index_args: Vec<Expr> = indices.clone();
      if index_args.len() == 1 {
        apply_func_ast(func, &index_args[0])
      } else {
        // For multi-arg, evaluate f[i1, i2, ...]
        match func {
          Expr::Identifier(name) => {
            crate::evaluator::evaluate_function_call_ast(name, &index_args)
          }
          _ => {
            // For pure functions, we need to call with Sequence of args
            // Build f[i1, i2, ...] manually
            let func_call = Expr::FunctionCall {
              name: crate::syntax::expr_to_string(func),
              args: index_args,
            };
            crate::evaluator::evaluate_expr_to_expr(&func_call)
          }
        }
      }
    } else {
      let n = dims[depth];
      let (start, step) = ranges[depth];
      let mut items = Vec::new();
      for i in 0..n {
        let v = start + step * (i as f64);
        // Use Integer if the value is a whole number and step is integer-typed;
        // otherwise use Real.
        let index_expr = if step.fract() == 0.0 && start.fract() == 0.0 {
          Expr::Integer(v as i128)
        } else {
          Expr::Real(v)
        };
        indices.push(index_expr);
        items.push(build_array(func, dims, ranges, depth + 1, indices)?);
        indices.pop();
      }
      Ok(Expr::List(items))
    }
  }

  let result = build_array(func, &dims, &ranges, 0, &mut Vec::new())?;

  // If head is specified, flatten and wrap with head
  if let Some(h) = head {
    fn collect_leaves(expr: &Expr, leaves: &mut Vec<Expr>) {
      match expr {
        Expr::List(items) => {
          for item in items {
            collect_leaves(item, leaves);
          }
        }
        _ => leaves.push(expr.clone()),
      }
    }
    let mut leaves = Vec::new();
    collect_leaves(&result, &mut leaves);
    crate::evaluator::evaluate_function_call_ast(h, &leaves)
  } else {
    Ok(result)
  }
}

/// Parse a dimension specification into a list of sizes. Accepts a List of
/// non-negative integers, or a single non-negative integer for 1D.
fn parse_sparse_dims(expr: &Expr) -> Option<Vec<usize>> {
  match expr {
    Expr::List(items) => {
      let mut result = Vec::with_capacity(items.len());
      for item in items {
        let n = expr_to_i128(item)?;
        if n < 0 {
          return None;
        }
        result.push(n as usize);
      }
      Some(result)
    }
    _ => {
      let n = expr_to_i128(expr)?;
      if n < 0 {
        return None;
      }
      Some(vec![n as usize])
    }
  }
}

/// Parse a list of position-value rules. Positions may be scalar integers
/// (1D) or lists of integers (k-D); all rules must share the same rank.
/// Returns the rules as (position, value) pairs and the max-per-axis dims.
fn parse_sparse_rules(
  items: &[Expr],
) -> Option<(Vec<(Vec<i128>, Expr)>, Vec<usize>)> {
  let mut rules: Vec<(Vec<i128>, Expr)> = Vec::with_capacity(items.len());
  let mut max_pos: Vec<usize> = Vec::new();
  let mut rank: Option<usize> = None;

  for item in items {
    let (pattern, replacement) = match item {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => return None,
    };
    let pos_vec: Vec<i128> = match pattern {
      Expr::List(pos) => {
        let mut v = Vec::with_capacity(pos.len());
        for p in pos {
          v.push(expr_to_i128(p)?);
        }
        v
      }
      _ => vec![expr_to_i128(pattern)?],
    };
    // Reject empty position, non-positive indices and rank mismatches.
    if pos_vec.is_empty() || pos_vec.iter().any(|&p| p < 1) {
      return None;
    }
    match rank {
      None => {
        rank = Some(pos_vec.len());
        max_pos = vec![0; pos_vec.len()];
      }
      Some(r) if r != pos_vec.len() => return None,
      _ => {}
    }
    for (i, &p) in pos_vec.iter().enumerate() {
      max_pos[i] = max_pos[i].max(p as usize);
    }
    rules.push((pos_vec, replacement.clone()));
  }

  Some((rules, max_pos))
}

/// Return the shape of a dense, rectangular nested list. For a scalar leaf,
/// returns an empty vector.
fn dense_shape(expr: &Expr) -> Option<Vec<usize>> {
  match expr {
    Expr::List(items) => {
      if items.is_empty() {
        return Some(vec![0]);
      }
      let first_shape = dense_shape(&items[0])?;
      for item in &items[1..] {
        if dense_shape(item)? != first_shape {
          return None;
        }
      }
      let mut dims = vec![items.len()];
      dims.extend(first_shape);
      Some(dims)
    }
    _ => Some(Vec::new()),
  }
}

/// Walk a dense nested list and record every leaf that is not equal to the
/// default value as a (position, value) rule.
fn collect_dense_rules(
  expr: &Expr,
  indices: &mut Vec<i128>,
  rules: &mut Vec<(Vec<i128>, Expr)>,
  default_str: &str,
) {
  match expr {
    Expr::List(items) => {
      for (i, item) in items.iter().enumerate() {
        indices.push((i + 1) as i128);
        collect_dense_rules(item, indices, rules, default_str);
        indices.pop();
      }
    }
    _ => {
      if crate::syntax::expr_to_string(expr) != default_str {
        rules.push((indices.clone(), expr.clone()));
      }
    }
  }
}

/// Parse the first argument of a SparseArray call, producing (rules, inferred
/// dimensions). The first argument may be a list of rules, a dense nested
/// list, or a single Rule expression.
fn parse_sparse_data(
  data: &Expr,
  default: &Expr,
) -> Option<(Vec<(Vec<i128>, Expr)>, Vec<usize>)> {
  // A bare single rule is treated as a one-element rule list.
  if matches!(data, Expr::Rule { .. }) {
    return parse_sparse_rules(std::slice::from_ref(data));
  }

  let items = match data {
    Expr::List(items) => items,
    _ => return None,
  };

  if items.is_empty() {
    return Some((Vec::new(), vec![0]));
  }

  let any_rule = items.iter().any(|it| matches!(it, Expr::Rule { .. }));
  let all_rules = items.iter().all(|it| matches!(it, Expr::Rule { .. }));

  if all_rules {
    return parse_sparse_rules(items);
  }
  if any_rule {
    // Mixed rule / non-rule entries are ambiguous.
    return None;
  }

  // Dense nested list: must be rectangular.
  let shape = dense_shape(data)?;
  let default_str = crate::syntax::expr_to_string(default);
  let mut rules = Vec::new();
  let mut indices = Vec::new();
  collect_dense_rules(data, &mut indices, &mut rules, &default_str);
  Some((rules, shape))
}

/// Normalize SparseArray arguments to the canonical form
/// `SparseArray[Automatic, dims, default, rules]`. Accepts:
///   SparseArray[rules]                — infer dims from positions
///   SparseArray[list]                 — dense list, default 0
///   SparseArray[data, dims]           — explicit dims, default 0
///   SparseArray[data, dims, default]  — explicit default
/// Already-canonical calls are returned unchanged. If normalization is not
/// possible the original FunctionCall is returned unevaluated.
pub fn sparse_array_normalize_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Already in canonical form: SparseArray[Automatic, dims, default, rules].
  if args.len() == 4
    && matches!(&args[0], Expr::Identifier(s) if s == "Automatic")
    && matches!(&args[1], Expr::List(_))
    && matches!(&args[3], Expr::List(_))
  {
    return Ok(Expr::FunctionCall {
      name: "SparseArray".to_string(),
      args: args.to_vec(),
    });
  }

  if args.is_empty() || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "SparseArray".to_string(),
      args: args.to_vec(),
    });
  }

  let data = &args[0];
  let default = if args.len() >= 3 {
    args[2].clone()
  } else {
    Expr::Integer(0)
  };

  let (parsed_rules, inferred_dims) = match parse_sparse_data(data, &default) {
    Some(x) => x,
    None => {
      return Ok(Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Determine final dimensions. When explicit dims are given they must have
  // the same rank as the parsed positions (unless there are no rules).
  let dims: Vec<usize> = if args.len() >= 2 {
    match parse_sparse_dims(&args[1]) {
      Some(d) => {
        if !parsed_rules.is_empty() && d.len() != parsed_rules[0].0.len() {
          return Ok(Expr::FunctionCall {
            name: "SparseArray".to_string(),
            args: args.to_vec(),
          });
        }
        d
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "SparseArray".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    if inferred_dims.is_empty() {
      return Ok(Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: args.to_vec(),
      });
    }
    inferred_dims
  };

  // Drop rules that are out of bounds or equal to the default value. Later
  // rules for the same position override earlier ones (Wolfram semantics),
  // and BTreeMap gives us deterministic lexicographic ordering.
  let default_str = crate::syntax::expr_to_string(&default);
  let mut dedup: std::collections::BTreeMap<Vec<i128>, Expr> =
    std::collections::BTreeMap::new();
  for (pos, val) in parsed_rules {
    if pos.len() != dims.len() {
      continue;
    }
    if !pos
      .iter()
      .enumerate()
      .all(|(i, &p)| p >= 1 && (p as usize) <= dims[i])
    {
      continue;
    }
    if crate::syntax::expr_to_string(&val) == default_str {
      dedup.remove(&pos);
      continue;
    }
    dedup.insert(pos, val);
  }

  let rules_expr: Vec<Expr> = dedup
    .into_iter()
    .map(|(pos, val)| Expr::Rule {
      pattern: Box::new(Expr::List(
        pos.into_iter().map(Expr::Integer).collect(),
      )),
      replacement: Box::new(val),
    })
    .collect();

  let dims_expr =
    Expr::List(dims.iter().map(|&n| Expr::Integer(n as i128)).collect());

  Ok(Expr::FunctionCall {
    name: "SparseArray".to_string(),
    args: vec![
      Expr::Identifier("Automatic".to_string()),
      dims_expr,
      default,
      Expr::List(rules_expr),
    ],
  })
}

/// Build a dense k-D nested list from a canonical rule list.
fn build_dense_from_rules(
  dims: &[usize],
  default: &Expr,
  rules: &[(Vec<i128>, Expr)],
) -> Expr {
  if dims.is_empty() {
    return default.clone();
  }
  if dims.len() == 1 {
    let n = dims[0];
    let mut arr = vec![default.clone(); n];
    for (pos, val) in rules {
      if pos.len() == 1 && pos[0] >= 1 && (pos[0] as usize) <= n {
        arr[(pos[0] - 1) as usize] = val.clone();
      }
    }
    return Expr::List(arr);
  }
  let d0 = dims[0];
  let rest_dims = &dims[1..];
  let mut result = Vec::with_capacity(d0);
  for i in 1..=d0 {
    let sub_rules: Vec<(Vec<i128>, Expr)> = rules
      .iter()
      .filter_map(|(pos, val)| {
        if !pos.is_empty() && pos[0] == i as i128 {
          Some((pos[1..].to_vec(), val.clone()))
        } else {
          None
        }
      })
      .collect();
    result.push(build_dense_from_rules(rest_dims, default, &sub_rules));
  }
  Expr::List(result)
}

/// Expand a SparseArray (any recognized form) into a dense nested list.
/// Called by `Normal[SparseArray[...]]`.
pub fn sparse_array_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let normalized = sparse_array_normalize_ast(args)?;
  let sa_args = match &normalized {
    Expr::FunctionCall { name, args } if name == "SparseArray" => args,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Normal".to_string(),
        args: vec![normalized],
      });
    }
  };
  // After normalize, either canonical (4 args starting with Automatic) or
  // still the original form (normalization failed).
  if sa_args.len() != 4
    || !matches!(&sa_args[0], Expr::Identifier(s) if s == "Automatic")
  {
    return Ok(Expr::FunctionCall {
      name: "Normal".to_string(),
      args: vec![normalized],
    });
  }
  let dims: Vec<usize> = match &sa_args[1] {
    Expr::List(items) => {
      let mut d = Vec::with_capacity(items.len());
      for it in items {
        match expr_to_i128(it) {
          Some(n) if n >= 0 => d.push(n as usize),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Normal".to_string(),
              args: vec![normalized],
            });
          }
        }
      }
      d
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Normal".to_string(),
        args: vec![normalized],
      });
    }
  };
  let default = &sa_args[2];
  let rules_list: Vec<(Vec<i128>, Expr)> = match &sa_args[3] {
    Expr::List(items) => items
      .iter()
      .filter_map(|r| match r {
        Expr::Rule {
          pattern,
          replacement,
        } => {
          let pos = match pattern.as_ref() {
            Expr::List(ps) => {
              let mut v = Vec::with_capacity(ps.len());
              for p in ps {
                v.push(expr_to_i128(p)?);
              }
              v
            }
            other => vec![expr_to_i128(other)?],
          };
          Some((pos, replacement.as_ref().clone()))
        }
        _ => None,
      })
      .collect(),
    _ => Vec::new(),
  };
  Ok(build_dense_from_rules(&dims, default, &rules_list))
}

/// Tuples[list, n] - Generate all n-tuples from elements of list (Cartesian product).
pub fn tuples_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Tuples[{list1, list2, ...}] - Cartesian product of multiple lists
    // Each element can be a List or a FunctionCall (extract args as elements)
    let outer_items = match &args[0] {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Tuples".to_string(),
          args: args.to_vec(),
        });
      }
    };

    // Extract elements from each sublist/expression
    let mut lists: Vec<Vec<Expr>> = Vec::new();
    for item in outer_items {
      match item {
        Expr::List(items) => lists.push(items.clone()),
        Expr::FunctionCall { args: fc_args, .. } => {
          lists.push(fc_args.clone());
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Tuples".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }

    // Cartesian product of all lists
    let mut result: Vec<Vec<Expr>> = vec![vec![]];
    for list in &lists {
      let mut new_result = Vec::new();
      for tuple in &result {
        for item in list {
          let mut new_tuple = tuple.clone();
          new_tuple.push(item.clone());
          new_result.push(new_tuple);
        }
      }
      result = new_result;
    }

    return Ok(Expr::List(result.into_iter().map(Expr::List).collect()));
  }

  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Tuples expects 1 or 2 arguments".into(),
    ));
  }

  // Tuples[list, n] or Tuples[f[a,b,...], n]
  let (items, head_name): (Vec<Expr>, Option<String>) = match &args[0] {
    Expr::List(items) => (items.clone(), None),
    Expr::FunctionCall {
      name,
      args: fc_args,
    } => (fc_args.clone(), Some(name.clone())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tuples".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = match &args[1] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tuples".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if n == 0 {
    let empty = if let Some(ref h) = head_name {
      Expr::FunctionCall {
        name: h.clone(),
        args: vec![],
      }
    } else {
      Expr::List(vec![])
    };
    return Ok(Expr::List(vec![empty]));
  }

  // Iterative Cartesian product
  let mut result: Vec<Vec<Expr>> = vec![vec![]];

  for _ in 0..n {
    let mut new_result = Vec::new();
    for tuple in &result {
      for item in &items {
        let mut new_tuple = tuple.clone();
        new_tuple.push(item.clone());
        new_result.push(new_tuple);
      }
    }
    result = new_result;
  }

  let wrap = |elems: Vec<Expr>| -> Expr {
    if let Some(ref h) = head_name {
      Expr::FunctionCall {
        name: h.clone(),
        args: elems,
      }
    } else {
      Expr::List(elems)
    }
  };

  Ok(Expr::List(result.into_iter().map(wrap).collect()))
}
