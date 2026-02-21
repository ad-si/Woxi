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
        let second = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        match second {
          Expr::List(list_items) => {
            // {i, {a, b, c}} form - iterate over list elements
            let mut results = Vec::new();
            for item in list_items {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &item);
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

  // Fallback to f64 for Real arguments
  let (min, max, step) = if args.len() == 1 {
    let max_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (1.0, max_val, 1.0)
  } else if args.len() == 2 {
    let min_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let max_val = expr_to_f64(&args[1]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (min_val, max_val, 1.0)
  } else {
    let min_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let max_val = expr_to_f64(&args[1]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let step_val = expr_to_f64(&args[2]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (min_val, max_val, step_val)
  };

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

  // Parse origins
  let origins: Vec<i128> = if args.len() >= 3 {
    match &args[2] {
      Expr::List(items) => items
        .iter()
        .map(|item| expr_to_i128(item).unwrap_or(1))
        .collect(),
      _ => {
        let o = expr_to_i128(&args[2]).unwrap_or(1);
        vec![o; dims.len()]
      }
    }
  } else {
    vec![1i128; dims.len()]
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

  // Build the array recursively
  fn build_array(
    func: &Expr,
    dims: &[i128],
    origins: &[i128],
    depth: usize,
    indices: &mut Vec<i128>,
  ) -> Result<Expr, InterpreterError> {
    if depth >= dims.len() {
      // Apply function to indices
      let index_args: Vec<Expr> =
        indices.iter().map(|&i| Expr::Integer(i)).collect();
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
      let origin = origins[depth];
      let mut items = Vec::new();
      for i in 0..n {
        indices.push(origin + i);
        items.push(build_array(func, dims, origins, depth + 1, indices)?);
        indices.pop();
      }
      Ok(Expr::List(items))
    }
  }

  let result = build_array(func, &dims, &origins, 0, &mut Vec::new())?;

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

/// AST-based SparseArray: create a matrix from position rules.
/// SparseArray[rules, {rows, cols}, default] -> evaluates rules and creates matrix
pub fn sparse_array_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Ok(Expr::FunctionCall {
      name: "SparseArray".to_string(),
      args: args.to_vec(),
    });
  }

  let rules = &args[0];
  let dims = &args[1];
  let default = &args[2];

  // Extract dimensions
  let dim_values = match dims {
    Expr::List(items) => {
      let mut result = Vec::new();
      for item in items {
        match expr_to_i128(item) {
          Some(n) => result.push(n as usize),
          None => {
            return Ok(Expr::FunctionCall {
              name: "SparseArray".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
      result
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if dim_values.len() == 2 {
    let rows = dim_values[0];
    let cols = dim_values[1];

    // Initialize matrix with default value
    let mut matrix: Vec<Vec<Expr>> = vec![vec![default.clone(); cols]; rows];

    // Process rules: {pos -> val, pos -> val, ...}
    let rule_list = match rules {
      Expr::List(items) => items.clone(),
      _ => vec![rules.clone()],
    };

    for rule in &rule_list {
      match rule {
        Expr::Rule {
          pattern,
          replacement,
        } => {
          // pattern should be {row, col} (1-indexed)
          if let Expr::List(pos) = pattern.as_ref()
            && pos.len() == 2
            && let (Expr::Integer(r), Expr::Integer(c)) = (&pos[0], &pos[1])
          {
            let ri = (*r - 1) as usize;
            let ci = (*c - 1) as usize;
            if ri < rows && ci < cols {
              matrix[ri][ci] = replacement.as_ref().clone();
            }
          }
        }
        _ => {} // skip non-rules
      }
    }

    // Convert to nested list
    let result: Vec<Expr> = matrix.into_iter().map(Expr::List).collect();
    return Ok(Expr::List(result));
  }

  // For non-2D arrays, return symbolic
  Ok(Expr::FunctionCall {
    name: "SparseArray".to_string(),
    args: args.to_vec(),
  })
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
