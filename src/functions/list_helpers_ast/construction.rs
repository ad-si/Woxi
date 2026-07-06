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

/// A count/bound value: an exact integer, or a real/rational truncated toward
/// zero (wolframscript's `Table[q, {5.5}]` yields 5 copies). Returns None for a
/// non-numeric (symbolic) value.
fn bound_to_count(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(_) | Expr::BigInteger(_) => expr_to_i128(e),
    _ => crate::functions::math_ast::try_eval_to_f64(e).map(|f| f as i128),
  }
}

/// The unevaluated single-iterator `Table[body, iter_spec]`.
fn table_unevaluated(body: &Expr, iter_spec: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Table".to_string(),
    args: vec![body.clone(), iter_spec.clone()].into(),
  }
}

/// If any iterator in a `Table`/`Do`-style call has bounds that don't resolve
/// to real numbers, return the message to emit before leaving the whole call
/// unevaluated (matching wolframscript's `::iterb` / `::nliter`). `name` is the
/// enclosing head (e.g. "Table", "Do"). `iters` are the iterator specs (the
/// arguments after the body).
pub fn table_iterators_invalid(name: &str, iters: &[Expr]) -> Option<String> {
  let iterb = |iter: &Expr| {
    Some(format!(
      "{}::iterb: Iterator {} does not have appropriate bounds.",
      name,
      crate::syntax::format_expr(iter, crate::syntax::ExprForm::Output)
    ))
  };
  // Variables introduced by earlier iterators: a later bound that references
  // one of them (e.g. the inner `{j, i}` of `Table[j, {i, 1, 3}, {j, i}]`)
  // becomes numeric during iteration, so it must not be flagged.
  let mut bound_vars: Vec<String> = Vec::new();
  // A bound "resolves" if it is already numeric or references an earlier
  // iterator variable.
  let resolves = |e: &Expr, vars: &[String]| -> bool {
    if expr_references_any(e, vars) {
      return true;
    }
    let ev =
      crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
    crate::functions::math_ast::try_eval_to_f64_with_infinity(&ev).is_some()
  };
  for (idx, iter) in iters.iter().enumerate() {
    let position = idx + 2; // the body is argument 1
    match iter {
      // A bare count `Table[expr, n]`: valid when it resolves.
      _ if !matches!(iter, Expr::List(_)) && resolves(iter, &bound_vars) => {}
      Expr::List(items) => {
        if items.is_empty() {
          continue;
        }
        if items.len() == 1 {
          if !resolves(&items[0], &bound_vars) {
            return iterb(iter);
          }
          continue;
        }
        // `{i, ...}`: items[0] is the iterator variable (introduced for later
        // iterators). The remaining entries must be a single list, or all
        // resolve to numbers / earlier iterator variables.
        if let Expr::Identifier(v) = &items[0] {
          bound_vars.push(v.clone());
        }
        let second = crate::evaluator::evaluate_expr_to_expr(&items[1])
          .unwrap_or_else(|_| items[1].clone());
        if matches!(second, Expr::List(_)) {
          continue;
        }
        // The variable of this iterator is in scope only for later ones, so
        // check the bounds against the earlier variables (all but the last).
        let earlier = &bound_vars[..bound_vars.len().saturating_sub(1)];
        if items[1..].iter().all(|b| resolves(b, earlier)) {
          continue;
        }
        // A `{i, min, max, step}` range with symbolic bounds is still valid
        // when its count (max - min)/step is a definite non-negative integer
        // (e.g. `{x, a, a + 5 n, n}` -> 6 values).
        if items.len() >= 3
          && symbolic_count_definite(&items[1], &items[2], items.get(3))
        {
          continue;
        }
        return iterb(iter);
      }
      // A non-list, non-numeric iterator such as the `i` in `Table[i, i]`.
      _ => {
        return Some(format!(
          "{}::nliter: Nonlist iterator {} at position {} does not evaluate to a real numeric value.",
          name,
          crate::syntax::expr_to_string(iter),
          position
        ));
      }
    }
  }
  None
}

/// Whether the range `{min, max, step}` has a definite non-negative integer
/// count `(max - min)/step`, so `Table` can iterate it symbolically even when
/// the bounds themselves are symbolic (e.g. `{a, a + 5 n, n}` -> count 5).
fn symbolic_count_definite(
  min: &Expr,
  max: &Expr,
  step: Option<&Expr>,
) -> bool {
  let eval = |e: &Expr| {
    crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone())
  };
  let (min_e, max_e) = (eval(min), eval(max));
  let step_e = step.map(eval).unwrap_or(Expr::Integer(1));
  let Ok(diff) =
    crate::evaluator::evaluate_function_call_ast("Subtract", &[max_e, min_e])
  else {
    return false;
  };
  let Ok(ratio) =
    crate::evaluator::evaluate_function_call_ast("Divide", &[diff, step_e])
  else {
    return false;
  };
  matches!(ratio, Expr::Integer(n) if n >= 0)
}

/// Whether `expr` references any identifier named in `vars`.
fn expr_references_any(expr: &Expr, vars: &[String]) -> bool {
  if vars.is_empty() {
    return false;
  }
  match expr {
    Expr::Identifier(n) => vars.iter().any(|v| v == n),
    Expr::BinaryOp { left, right, .. } => {
      expr_references_any(left, vars) || expr_references_any(right, vars)
    }
    Expr::UnaryOp { operand, .. } => expr_references_any(operand, vars),
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| expr_references_any(a, vars))
    }
    Expr::List(items) => items.iter().any(|a| expr_references_any(a, vars)),
    _ => false,
  }
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
        if !is_nothing(&val) {
          results.push(val);
        }
      }
      Ok(Expr::List(results.into()))
    }
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::List(vec![].into()));
      }

      // Handle {n} form (single element = just repeat count, no variable)
      if items.len() == 1 {
        let evaluated = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
        // A non-numeric count leaves the call unevaluated (the dispatch-level
        // check normally intercepts this; kept here as a safety net). A real
        // count is truncated toward zero, matching wolframscript's
        // Table[q, {5.5}] -> 5 copies.
        let n = match bound_to_count(&evaluated) {
          Some(n) => n,
          None => return Ok(table_unevaluated(body, iter_spec)),
        };
        let mut results = Vec::new();
        for _ in 0..n {
          let val = crate::evaluator::evaluate_expr_to_expr(body)?;
          if !is_nothing(&val) {
            results.push(val);
          }
        }
        return Ok(Expr::List(results.into()));
      }

      // Extract iterator variable
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        other => {
          // A non-symbol iterator (Table[i, {3, 1, 5}]) is a raw object that
          // cannot be used as an iterator: wolframscript emits Table::itraw
          // and returns the call unevaluated rather than raising an error.
          crate::emit_message(&format!(
            "Table::itraw: Raw object {} cannot be used as an iterator.",
            crate::syntax::expr_to_string(other)
          ));
          return Ok(Expr::FunctionCall {
            name: "Table".to_string(),
            args: vec![body.clone(), iter_spec.clone()].into(),
          });
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
              if !is_nothing(&val) {
                results.push(val);
              }
            }
            return Ok(Expr::List(results.into()));
          }
          _ => {
            // {i, max} form - iterate from 1 to max (real max truncated).
            let max_val = match bound_to_count(&second) {
              Some(n) => n,
              None => return Ok(table_unevaluated(body, iter_spec)),
            };
            let mut results = Vec::new();
            for i in 1..=max_val {
              let substituted = crate::syntax::substitute_variable(
                body,
                &var_name,
                &Expr::Integer(i),
              );
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              if !is_nothing(&val) {
                results.push(val);
              }
            }
            return Ok(Expr::List(results.into()));
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
              if !is_nothing(&val) {
                results.push(val);
              }
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
              if !is_nothing(&val) {
                results.push(val);
              }
              i += step_val;
            }
          }
          return Ok(Expr::List(results.into()));
        }

        // Symbolic path: if (max - min) / step simplifies to a non-negative
        // EXACT integer count, iterate symbolically as min, min+step, …,
        // min+n*step. Handles cases like Table[x, {x, a, a+5n, n}] where the
        // bounds are symbolic but the iteration count is well-defined.
        // Accept only Expr::Integer so Real arithmetic (e.g. 10/0.2 → 50.)
        // keeps using the existing f64 accumulation path, preserving
        // snapshot-sensitive rounding behavior.
        let diff = crate::evaluator::evaluate_function_call_ast(
          "Subtract",
          &[max_expr.clone(), min_expr.clone()],
        )?;
        let ratio = crate::evaluator::evaluate_function_call_ast(
          "Divide",
          &[diff, step_expr.clone()],
        )?;
        if let Expr::Integer(n) = ratio
          && (0..=1_000_000).contains(&n)
        {
          let mut results = Vec::with_capacity((n + 1) as usize);
          for k in 0..=n {
            let current = if k == 0 {
              min_expr.clone()
            } else {
              let k_step = crate::evaluator::evaluate_function_call_ast(
                "Times",
                &[Expr::Integer(k), step_expr.clone()],
              )?;
              crate::evaluator::evaluate_function_call_ast(
                "Plus",
                &[min_expr.clone(), k_step],
              )?
            };
            let substituted =
              crate::syntax::substitute_variable(body, &var_name, &current);
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            if !is_nothing(&val) {
              results.push(val);
            }
          }
          return Ok(Expr::List(results.into()));
        }

        // Fallback: numeric iteration for symbolic numeric bounds/step (e.g.
        // Pi/4). Non-numeric bounds/step leave the call unevaluated (the
        // dispatch-level check normally intercepts this first).
        if crate::functions::math_ast::try_eval_to_f64(&min_expr).is_none() {
          return Ok(table_unevaluated(body, iter_spec));
        }
        let max_num =
          match crate::functions::math_ast::try_eval_to_f64(&max_expr) {
            Some(v) => v,
            None => return Ok(table_unevaluated(body, iter_spec)),
          };
        let step_num =
          match crate::functions::math_ast::try_eval_to_f64_with_infinity(
            &step_expr,
          ) {
            Some(v) => v,
            None => return Ok(table_unevaluated(body, iter_spec)),
          };

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
              crate::functions::math_ast::try_eval_to_f64_with_infinity(
                &current_expr,
              )
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
            if !is_nothing(&val) {
              results.push(val);
            }
            current_expr =
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![current_expr, step_expr.clone()].into(),
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
              crate::functions::math_ast::try_eval_to_f64_with_infinity(
                &current_expr,
              )
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
            if !is_nothing(&val) {
              results.push(val);
            }
            current_expr =
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![current_expr, step_expr.clone()].into(),
              })?;
            safety_counter += 1;
            if safety_counter > 1_000_000 {
              return Err(InterpreterError::EvaluationError(
                "Table: iterator exceeded maximum iterations".into(),
              ));
            }
          }
        }
        return Ok(Expr::List(results.into()));
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
      args: args.to_vec().into(),
    });
  }

  // Listable-style threading: Range[{a,b}, ...] => {Range[a, ...], Range[b, ...]}.
  // Mathematica threads Range over any list argument when the other
  // arguments are scalars, so e.g. Range[0, 999, {3, 5}] returns
  // {Range[0,999,3], Range[0,999,5]}.
  for (idx, arg) in args.iter().enumerate() {
    if let Expr::List(items) = arg {
      let mut threaded = Vec::with_capacity(items.len());
      for item in items {
        let mut new_args: Vec<Expr> = args.to_vec();
        new_args[idx] = item.clone();
        threaded.push(range_ast(&new_args)?);
      }
      return Ok(Expr::List(threaded.into()));
    }
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
      let call = Expr::FunctionCall {
        name: "Range".to_string(),
        args: args.to_vec().into(),
      };
      crate::emit_message(&format!(
        "Range::range: Range specification in {} does not have appropriate bounds.",
        crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
      ));
      return Ok(call);
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

    return Ok(Expr::List(results.into()));
  }

  // Try symbolic range when arguments contain symbolic constants (e.g. Pi)
  let try_numeric = |e: &Expr| -> Option<f64> {
    if let Some(f) = expr_to_f64(e) {
      return Some(f);
    }
    // Try evaluating N[expr] to get a float
    let n_expr = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![e.clone()].into(),
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

    // If the endpoints are purely symbolic (e.g. `Range[a, a+5]`) the
    // difference `max - min` — or the ratio `(max - min)/step` when the
    // step is itself symbolic (e.g. `Range[a, b, (b - a)/4]`) — may still
    // simplify to a numeric value. In that case we can enumerate the
    // range using the numeric count and render each element symbolically.
    if min_f.is_none() || max_f.is_none() {
      if let Some(0.0) = step_f {
        let call = Expr::FunctionCall {
          name: "Range".to_string(),
          args: args.to_vec().into(),
        };
        crate::emit_message(&format!(
          "Range::range: Range specification in {} does not have appropriate bounds.",
          crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
        ));
        return Ok(call);
      }
      // ratio = (max - min) / step
      let ratio_expr = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              max_expr.clone(),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), min_expr.clone()].into(),
              },
            ]
            .into(),
          },
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![step_expr.clone(), Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      };
      // Accept rationals (e.g. `5/2`), floats, and symbolic reals like
      // `Pi` — fall through to `try_numeric` (which calls `N[expr]`)
      // when the ratio doesn't reduce to a literal Integer/Rational/Real.
      let evaled_ratio =
        crate::evaluator::evaluate_expr_to_expr(&ratio_expr).ok();
      let ratio_val = evaled_ratio.as_ref().and_then(|e| {
        expr_to_rational(e)
          .map(|(n, d)| n as f64 / d as f64)
          .or_else(|| try_numeric(e))
      });
      if let Some(ratio_val) = ratio_val
        && ratio_val.is_finite()
      {
        let count = ratio_val.floor() as i128 + 1;
        if count <= 0 {
          return Ok(Expr::List(vec![].into()));
        }
        if count > 1_000_000 {
          return Err(InterpreterError::EvaluationError(
            "Range: result too large".into(),
          ));
        }
        let mut results = Vec::with_capacity(count as usize);
        for k in 0..count {
          let elem = if k == 0 {
            min_expr.clone()
          } else {
            let k_times_step = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(k), step_expr.clone()].into(),
            };
            let sum = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![min_expr.clone(), k_times_step].into(),
            };
            crate::evaluator::evaluate_expr_to_expr(&sum)?
          };
          results.push(elem);
        }
        return Ok(Expr::List(results.into()));
      }
      // Fully symbolic with no way to determine length — leave
      // unevaluated like Mathematica.
      return Ok(Expr::FunctionCall {
        name: "Range".to_string(),
        args: args.to_vec().into(),
      });
    }

    if let (Some(min_val), Some(max_val), Some(step_val)) =
      (min_f, max_f, step_f)
    {
      if step_val == 0.0 {
        let call = Expr::FunctionCall {
          name: "Range".to_string(),
          args: args.to_vec().into(),
        };
        crate::emit_message(&format!(
          "Range::range: Range specification in {} does not have appropriate bounds.",
          crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
        ));
        return Ok(call);
      }
      let count = ((max_val - min_val) / step_val).floor() as i128 + 1;
      if count <= 0 {
        return Ok(Expr::List(vec![].into()));
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
      return Ok(Expr::List(results.into()));
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
    let call = Expr::FunctionCall {
      name: "Range".to_string(),
      args: args.to_vec().into(),
    };
    crate::emit_message(&format!(
      "Range::range: Range specification in {} does not have appropriate bounds.",
      crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
    ));
    return Ok(call);
  }

  // Output elements are Real only when `min` or `step` is Real.
  // `max` only controls termination, so e.g. Range[3.5] yields integers.
  let any_real =
    matches!(&min_expr, Expr::Real(_)) || matches!(&step_expr, Expr::Real(_));

  // Compute each element as `min + k*step` rather than by repeated addition.
  // Accumulation compounds the rounding error of an inexact step, so e.g.
  // Range[10, 11, 0.1] would end at 10.999999999999996; `min + k*step` matches
  // wolframscript's 11. (The termination tolerance is unchanged.)
  let mut results = Vec::new();
  let mut k: i128 = 0;
  loop {
    let val = min + (k as f64) * step;
    let within = if step > 0.0 {
      val <= max + f64::EPSILON
    } else {
      val >= max - f64::EPSILON
    };
    if !within {
      break;
    }
    results.push(if any_real {
      Expr::Real(val)
    } else {
      f64_to_expr(val)
    });
    k += 1;
    if k > 1_000_000 {
      return Err(InterpreterError::EvaluationError(
        "Range: result too large".into(),
      ));
    }
  }

  Ok(Expr::List(results.into()))
}

/// PowerRange[min, max] generates {min, min*10, min*100, ...} up to max.
/// PowerRange[min, max, r] uses factor r instead of 10.
pub fn power_range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "PowerRange".to_string(),
      args: args.to_vec().into(),
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

    return Ok(Expr::List(results.into()));
  }

  Ok(Expr::FunctionCall {
    name: "PowerRange".to_string(),
    args: args.to_vec().into(),
  })
}

/// AST-based ConstantArray: create array filled with constant.
/// ConstantArray[c, n] -> {c, c, ..., c} (n times)
/// ConstantArray[c, {n1, n2}] -> nested array
pub fn constant_array_ast(
  elem: &Expr,
  dims: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ConstantArray".to_string(),
      args: vec![elem.clone(), dims.clone()].into(),
    })
  };
  // A dimension must be a non-negative machine integer; anything else
  // (negative, non-integer, symbolic, or too large for i128) leaves the call
  // unevaluated rather than raising an error — matching wolframscript, which
  // returns the unevaluated form (or a SymbolicZerosArray for symbolic dims).
  let nonneg_dim = |e: &Expr| -> Option<usize> {
    match e {
      Expr::Integer(n) if *n >= 0 => usize::try_from(*n).ok(),
      Expr::BigInteger(_) => expr_to_i128(e)
        .filter(|&n| n >= 0)
        .and_then(|n| usize::try_from(n).ok()),
      _ => None,
    }
  };
  // Flatten the dimension specifier into a list (a scalar dimension counts as
  // a single-element list, matching wolframscript's `SymbolicZerosArray[{n}]`).
  let dim_exprs: Vec<Expr> = match dims {
    Expr::List(dim_list) => dim_list.iter().cloned().collect(),
    other => vec![other.clone()],
  };

  // Classify each dimension. A concrete non-negative machine integer is a
  // usable dimension; a concrete number that isn't (negative, non-integer,
  // too large) makes wolframscript emit `::ilsmn` and leave the call
  // unevaluated; anything else is symbolic.
  let mut concrete_dims: Vec<usize> = Vec::with_capacity(dim_exprs.len());
  let mut has_symbolic = false;
  for d in &dim_exprs {
    match nonneg_dim(d) {
      Some(n) => concrete_dims.push(n),
      None => {
        // A concrete but invalid dimension is an error, so leave unevaluated;
        // a symbolic dimension yields a SymbolicZeros/OnesArray placeholder.
        if crate::functions::predicate_ast::is_numeric_q_pub(d) {
          return unevaluated();
        }
        has_symbolic = true;
      }
    }
  }

  if !has_symbolic {
    // Build the nested constant array from the innermost dimension outward.
    let mut result = elem.clone();
    for &n in concrete_dims.iter().rev() {
      result = Expr::List(vec![result; n].into());
    }
    return Ok(result);
  }

  // At least one symbolic dimension (and no invalid concrete ones): produce a
  // SymbolicZerosArray/SymbolicOnesArray placeholder like wolframscript.
  //   ConstantArray[0, dims] -> SymbolicZerosArray[dims]
  //   ConstantArray[1, dims] -> SymbolicOnesArray[dims]
  //   ConstantArray[c, dims] -> c*SymbolicOnesArray[dims]
  let dims_list = Expr::List(dim_exprs.into());
  let ones = || Expr::FunctionCall {
    name: "SymbolicOnesArray".to_string(),
    args: vec![dims_list.clone()].into(),
  };
  let result = match elem {
    Expr::Integer(0) => Expr::FunctionCall {
      name: "SymbolicZerosArray".to_string(),
      args: vec![dims_list.clone()].into(),
    },
    Expr::Integer(1) => ones(),
    _ => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(elem.clone()),
      right: Box::new(ones()),
    },
  };
  Ok(result)
}

/// True when `body` references `var_name` as the head of a `FunctionCall`
/// anywhere in the tree. The substitute-variable path turns such a call
/// into `CurriedCall { func: <value>, args }`, so it works when `var` is
/// bound to a value that isn't a function (e.g. an Association). The
/// ENV-binding fast path can't reproduce that — it would dispatch the
/// call by literal name. When this returns true we have to fall back to
/// the substitute path to preserve semantics.
fn body_uses_var_as_function_head(body: &Expr, var_name: &str) -> bool {
  match body {
    Expr::FunctionCall { name, args } => {
      name == var_name
        || args
          .iter()
          .any(|a| body_uses_var_as_function_head(a, var_name))
    }
    Expr::List(items) => items
      .iter()
      .any(|a| body_uses_var_as_function_head(a, var_name)),
    Expr::BinaryOp { left, right, .. } => {
      body_uses_var_as_function_head(left, var_name)
        || body_uses_var_as_function_head(right, var_name)
    }
    Expr::UnaryOp { operand, .. } => {
      body_uses_var_as_function_head(operand, var_name)
    }
    Expr::Comparison { operands, .. } => operands
      .iter()
      .any(|a| body_uses_var_as_function_head(a, var_name)),
    Expr::CompoundExpr(exprs) => exprs
      .iter()
      .any(|a| body_uses_var_as_function_head(a, var_name)),
    Expr::CurriedCall { func, args } => {
      body_uses_var_as_function_head(func, var_name)
        || args
          .iter()
          .any(|a| body_uses_var_as_function_head(a, var_name))
    }
    Expr::Part { expr, index } => {
      body_uses_var_as_function_head(expr, var_name)
        || body_uses_var_as_function_head(index, var_name)
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      body_uses_var_as_function_head(pattern, var_name)
        || body_uses_var_as_function_head(replacement, var_name)
    }
    Expr::ReplaceAll { expr: e, rules }
    | Expr::ReplaceRepeated { expr: e, rules } => {
      body_uses_var_as_function_head(e, var_name)
        || body_uses_var_as_function_head(rules, var_name)
    }
    Expr::Map { func, list }
    | Expr::Apply { func, list }
    | Expr::MapApply { func, list } => {
      body_uses_var_as_function_head(func, var_name)
        || body_uses_var_as_function_head(list, var_name)
    }
    Expr::Association(items) => items.iter().any(|(k, v)| {
      body_uses_var_as_function_head(k, var_name)
        || body_uses_var_as_function_head(v, var_name)
    }),
    _ => false,
  }
}

/// Bind a Do-loop iterator variable in ENV. Reuses the existing slot when
/// possible to avoid cloning `var_name` into a fresh HashMap key on every
/// iteration.
fn bind_loop_var(var_name: &str, value: Expr) {
  crate::ENV.with(|e| {
    let mut env = e.borrow_mut();
    if let Some(slot) = env.get_mut(var_name) {
      *slot = crate::StoredValue::ExprVal(value);
    } else {
      env.insert(var_name.to_string(), crate::StoredValue::ExprVal(value));
    }
  });
}

/// Restore the previous binding of a loop iterator variable (or remove
/// the slot if the variable was previously unbound).
fn restore_loop_var(var_name: &str, prev: Option<crate::StoredValue>) {
  crate::ENV.with(|e| {
    let mut env = e.borrow_mut();
    if let Some(v) = prev {
      env.insert(var_name.to_string(), v);
    } else {
      env.remove(var_name);
    }
  });
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
        other => {
          // A non-symbol iterator (Do[body, {3, 1, 5}]) is a raw object that
          // cannot be used as an iterator: wolframscript emits Do::itraw and
          // returns the call unevaluated rather than raising an error.
          crate::emit_message(&format!(
            "Do::itraw: Raw object {} cannot be used as an iterator.",
            crate::syntax::expr_to_string(other)
          ));
          return Ok(Expr::FunctionCall {
            name: "Do".to_string(),
            args: vec![body.clone(), iter_spec.clone()].into(),
          });
        }
      };

      // Handle list iterator: Do[body, {i, {a, b, c}}]
      if items.len() == 2 {
        let needs_substitute = body_uses_var_as_function_head(body, &var_name);

        // Fast path: `Do[body, {c, Characters[s]}]` — iterate the
        // string's chars directly instead of allocating an N-element
        // Expr::List of single-char Expr::Strings. The accumulating
        // splitLines loop pays one such call per ~75 KB input.
        if !needs_substitute
          && let Expr::FunctionCall {
            name: head,
            args: head_args,
          } = &items[1]
          && head == "Characters"
          && head_args.len() == 1
        {
          let s_expr = crate::evaluator::evaluate_expr_to_expr(&head_args[0])?;
          if let Expr::String(ref s) = s_expr {
            let prev = crate::ENV.with(|e| e.borrow_mut().remove(&var_name));
            let mut early_return: Option<Expr> = None;
            let mut error: Option<InterpreterError> = None;
            let mut buf = [0u8; 4];
            for ch in s.chars() {
              let cs = ch.encode_utf8(&mut buf).to_string();
              bind_loop_var(&var_name, Expr::String(cs));
              match crate::evaluator::evaluate_expr_to_expr(body) {
                Ok(_) => {}
                Err(InterpreterError::BreakSignal) => break,
                Err(InterpreterError::ContinueSignal) => continue,
                Err(InterpreterError::ReturnValue(val)) => {
                  early_return = Some(*val);
                  break;
                }
                Err(e) => {
                  error = Some(e);
                  break;
                }
              }
            }
            restore_loop_var(&var_name, prev);
            if let Some(e) = error {
              return Err(e);
            }
            if let Some(v) = early_return {
              return Ok(v);
            }
            return Ok(Expr::Identifier("Null".to_string()));
          }
          // Fall through to the generic path if Characters[s] didn't
          // produce a String (e.g. threaded over a list).
        }

        let val_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        if let Expr::List(list_items) = &val_expr {
          // Bind the iterator in ENV instead of cloning + substituting
          // the body on every iteration. The substitute path used to
          // walk the entire body AST per step, which made tight loops
          // (e.g. `Do[…, {c, Characters[s]}]` over 75 KB inputs) pay
          // O(body_size) work per iteration. ENV binding makes that
          // cost O(1) by resolving `var_name` through the normal
          // identifier-lookup path.
          if !needs_substitute {
            let prev = crate::ENV.with(|e| e.borrow_mut().remove(&var_name));
            let mut early_return: Option<Expr> = None;
            let mut error: Option<InterpreterError> = None;
            for item in list_items {
              bind_loop_var(&var_name, item.clone());
              match crate::evaluator::evaluate_expr_to_expr(body) {
                Ok(_) => {}
                Err(InterpreterError::BreakSignal) => break,
                Err(InterpreterError::ContinueSignal) => continue,
                Err(InterpreterError::ReturnValue(val)) => {
                  early_return = Some(*val);
                  break;
                }
                Err(e) => {
                  error = Some(e);
                  break;
                }
              }
            }
            restore_loop_var(&var_name, prev);
            if let Some(e) = error {
              return Err(e);
            }
            if let Some(v) = early_return {
              return Ok(v);
            }
            return Ok(Expr::Identifier("Null".to_string()));
          }
          // Fall back to substitute when body uses var as a function head.
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
        let max_val = super::utilities::expr_to_i128_floor(&max_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: iterator bound must be an integer".into(),
            )
          })?;
        (1i128, max_val, 1i128)
      } else if items.len() >= 3 {
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
        let min_val = super::utilities::expr_to_i128_floor(&min_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: iterator bound must be an integer".into(),
            )
          })?;
        let max_val = super::utilities::expr_to_i128_floor(&max_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: iterator bound must be an integer".into(),
            )
          })?;
        let step_val = if items.len() >= 4 {
          let step_expr = crate::evaluator::evaluate_expr_to_expr(&items[3])?;
          super::utilities::expr_to_i128_floor(&step_expr).ok_or_else(|| {
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

      let needs_substitute = body_uses_var_as_function_head(body, &var_name);
      if needs_substitute {
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
        return Ok(Expr::Identifier("Null".to_string()));
      }
      // ENV-binding fast path.
      let prev = crate::ENV.with(|e| e.borrow_mut().remove(&var_name));
      let mut early_return: Option<Expr> = None;
      let mut error: Option<InterpreterError> = None;
      let mut i = min;
      if step > 0 {
        while i <= max {
          crate::ENV.with(|e| {
            e.borrow_mut().insert(
              var_name.clone(),
              crate::StoredValue::ExprVal(Expr::Integer(i)),
            );
          });
          match crate::evaluator::evaluate_expr_to_expr(body) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(InterpreterError::ReturnValue(val)) => {
              early_return = Some(*val);
              break;
            }
            Err(e) => {
              error = Some(e);
              break;
            }
          }
          i += step;
        }
      } else {
        while i >= max {
          crate::ENV.with(|e| {
            e.borrow_mut().insert(
              var_name.clone(),
              crate::StoredValue::ExprVal(Expr::Integer(i)),
            );
          });
          match crate::evaluator::evaluate_expr_to_expr(body) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(InterpreterError::ReturnValue(val)) => {
              early_return = Some(*val);
              break;
            }
            Err(e) => {
              error = Some(e);
              break;
            }
          }
          i += step;
        }
      }
      crate::ENV.with(|e| {
        let mut env = e.borrow_mut();
        if let Some(v) = prev {
          env.insert(var_name.clone(), v);
        } else {
          env.remove(&var_name);
        }
      });
      if let Some(e) = error {
        return Err(e);
      }
      if let Some(v) = early_return {
        return Ok(v);
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Do: invalid iterator specification".into(),
    )),
  }
}

/// Multi-iterator Do: `Do[body, iter1, iter2, ...]`.
///
/// In Wolfram, a multi-iterator `Do` is a single construct: `Break[]` and
/// `Return[]` exit the entire `Do`, not just the innermost iterator. We
/// implement this by recursing over the iterator list, evaluating `body`
/// at the innermost level. The recursion is performed on a special inner
/// helper that does NOT catch `Break`/`Return`; only this outer wrapper
/// catches them so they propagate through all levels.
pub fn do_multi_ast(
  body: &Expr,
  iter_specs: &[Expr],
) -> Result<Expr, InterpreterError> {
  // A non-symbol iterator in any spec (Do[body, {3, 1, 5}, {j, 1, 2}]) is a
  // raw object that cannot be used as an iterator: emit Do::itraw and return
  // the call unevaluated, matching wolframscript, rather than crashing.
  for spec in iter_specs {
    if let Expr::List(items) = spec
      && items.len() >= 2
      && !matches!(&items[0], Expr::Identifier(_))
    {
      crate::emit_message(&format!(
        "Do::itraw: Raw object {} cannot be used as an iterator.",
        crate::syntax::expr_to_string(&items[0])
      ));
      return Ok(Expr::FunctionCall {
        name: "Do".to_string(),
        args: std::iter::once(body.clone())
          .chain(iter_specs.iter().cloned())
          .collect(),
      });
    }
  }
  match do_multi_inner(body, iter_specs) {
    Ok(_) => Ok(Expr::Identifier("Null".to_string())),
    Err(InterpreterError::BreakSignal) => {
      Ok(Expr::Identifier("Null".to_string()))
    }
    Err(InterpreterError::ReturnValue(val)) => Ok(*val),
    Err(e) => Err(e),
  }
}

fn do_multi_inner(
  body: &Expr,
  iter_specs: &[Expr],
) -> Result<(), InterpreterError> {
  if iter_specs.is_empty() {
    // Evaluate body; let Break/Return/Continue propagate up.
    match crate::evaluator::evaluate_expr_to_expr(body) {
      Ok(_) => Ok(()),
      Err(InterpreterError::ContinueSignal) => Ok(()),
      Err(e) => Err(e),
    }
  } else {
    let iter_spec = &iter_specs[0];
    let rest = &iter_specs[1..];
    iterate_spec(iter_spec, &mut |_| do_multi_inner(body, rest))
  }
}

/// Helper: drive a single iterator spec, calling `step` once per iteration.
/// `step` receives the current iteration index (0-based) for informational
/// purposes; the iterator variable (if any) is bound in ENV before each call.
/// Errors from `step` (Break/Return/etc.) propagate up.
fn iterate_spec<F>(
  iter_spec: &Expr,
  step: &mut F,
) -> Result<(), InterpreterError>
where
  F: FnMut(usize) -> Result<(), InterpreterError>,
{
  match iter_spec {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let n = expr_to_i128(iter_spec).unwrap_or(0);
      for i in 0..n {
        step(i as usize)?;
      }
      Ok(())
    }
    Expr::List(items) if items.len() == 1 => {
      let n_expr = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
      let n = expr_to_i128(&n_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Do: repeat count must be an integer".into(),
        )
      })?;
      for i in 0..n {
        step(i as usize)?;
      }
      Ok(())
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

      // Iterator over an explicit list: Do[..., {v, {a, b, c}}]
      if items.len() == 2 {
        let val_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        if let Expr::List(list_items) = &val_expr {
          let prev = crate::ENV.with(|e| e.borrow_mut().remove(&var_name));
          let mut err: Option<InterpreterError> = None;
          for (i, item) in list_items.iter().enumerate() {
            bind_loop_var(&var_name, item.clone());
            match step(i) {
              Ok(()) => {}
              Err(e) => {
                err = Some(e);
                break;
              }
            }
          }
          restore_loop_var(&var_name, prev);
          if let Some(e) = err {
            return Err(e);
          }
          return Ok(());
        }
      }

      // Numeric range iterator: {v, max} or {v, min, max} or {v, min, max, step}.
      let (min, max, step_val) = if items.len() == 2 {
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_val = super::utilities::expr_to_i128_floor(&max_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: iterator bound must be an integer".into(),
            )
          })?;
        (1i128, max_val, 1i128)
      } else {
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
        let min_val = super::utilities::expr_to_i128_floor(&min_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: iterator bound must be an integer".into(),
            )
          })?;
        let max_val = super::utilities::expr_to_i128_floor(&max_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: iterator bound must be an integer".into(),
            )
          })?;
        let s = if items.len() >= 4 {
          let s_expr = crate::evaluator::evaluate_expr_to_expr(&items[3])?;
          super::utilities::expr_to_i128_floor(&s_expr).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: step must be an integer".into(),
            )
          })?
        } else {
          1i128
        };
        (min_val, max_val, s)
      };

      if step_val == 0 {
        return Err(InterpreterError::EvaluationError(
          "Do: step cannot be zero".into(),
        ));
      }

      let prev = crate::ENV.with(|e| e.borrow_mut().remove(&var_name));
      let mut err: Option<InterpreterError> = None;
      let mut i = min;
      let mut idx = 0usize;
      if step_val > 0 {
        while i <= max {
          crate::ENV.with(|e| {
            e.borrow_mut().insert(
              var_name.clone(),
              crate::StoredValue::ExprVal(Expr::Integer(i)),
            );
          });
          match step(idx) {
            Ok(()) => {}
            Err(e) => {
              err = Some(e);
              break;
            }
          }
          i += step_val;
          idx += 1;
        }
      } else {
        while i >= max {
          crate::ENV.with(|e| {
            e.borrow_mut().insert(
              var_name.clone(),
              crate::StoredValue::ExprVal(Expr::Integer(i)),
            );
          });
          match step(idx) {
            Ok(()) => {}
            Err(e) => {
              err = Some(e);
              break;
            }
          }
          i += step_val;
          idx += 1;
        }
      }
      restore_loop_var(&var_name, prev);
      if let Some(e) = err {
        return Err(e);
      }
      Ok(())
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
    if !is_nothing(&val) {
      result.push(val);
    }
  }
  Ok(Expr::List(result.into()))
}

/// Build the sequence of index values for a dimension given an integer
/// starting offset. Indices are `start, start+1, ..., start+n-1`, built as
/// Expr values so the result preserves integer exactness.
fn build_offset_indices(
  n: i128,
  start: &Expr,
) -> Result<Vec<Expr>, InterpreterError> {
  let mut result = Vec::with_capacity(n.max(0) as usize);
  for i in 0..n {
    let val = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![start.clone(), Expr::Integer(i)].into(),
    };
    result.push(crate::evaluator::evaluate_expr_to_expr(&val)?);
  }
  Ok(result)
}

/// Build the sequence of index values for a dimension given a range
/// `{a, b}`. Produces `n` values evenly spaced from `a` to `b`, using exact
/// rational arithmetic when the endpoints are exact.
fn build_range_indices(
  n: i128,
  a: &Expr,
  b: &Expr,
) -> Result<Vec<Expr>, InterpreterError> {
  if n <= 0 {
    return Ok(Vec::new());
  }
  if n == 1 {
    // Wolfram gives the midpoint (a + b) / 2 when requesting a single
    // sample over a range, matching Array[f, 1, {a, b}] → {f[(a+b)/2]}.
    let half = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![Expr::Integer(2), Expr::Integer(-1)].into(),
    };
    let mid = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        half,
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![a.clone(), b.clone()].into(),
        },
      ]
      .into(),
    };
    return Ok(vec![crate::evaluator::evaluate_expr_to_expr(&mid)?]);
  }
  let mut result = Vec::with_capacity(n as usize);
  // value_i = a + i * (b - a) / (n - 1)
  let diff = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      b.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), a.clone()].into(),
      },
    ]
    .into(),
  };
  let inv_denom = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![Expr::Integer(n - 1), Expr::Integer(-1)].into(),
  };
  for i in 0..n {
    let term = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(i), diff.clone(), inv_denom.clone()].into(),
    };
    let val = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![a.clone(), term].into(),
    };
    result.push(crate::evaluator::evaluate_expr_to_expr(&val)?);
  }
  Ok(result)
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
            args: args.to_vec().into(),
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
          args: args.to_vec().into(),
        });
      }
    }
  };

  // Build the list of index Exprs per dimension using exact arithmetic so
  // integer/rational inputs stay exact.
  //
  // Wolfram supports several forms for the origin/range argument:
  //   Array[f, n]                  → indices 1..n        (start=1, step=1)
  //   Array[f, n, r]               → indices r..r+n-1    (start=r, step=1)
  //   Array[f, n, {a, b}]          → n values from a..b  (start=a, step=(b-a)/(n-1))
  //   Array[f, {n1,...}, {r1,...}] → per-dim origin
  //   Array[f, {n1,...}, {{a1,b1},...}] → per-dim range
  let dim_indices: Vec<Vec<Expr>> = if args.len() >= 3 {
    let origin = &args[2];
    match origin {
      // Single-dim {a, b} range form. Ambiguous with two per-dim origins,
      // so only treat as a range when there is exactly one dimension.
      Expr::List(items) if dims.len() == 1 && items.len() == 2 => {
        vec![build_range_indices(dims[0], &items[0], &items[1])?]
      }
      Expr::List(items) => items
        .iter()
        .enumerate()
        .map(|(i, item)| match item {
          Expr::List(pair) if pair.len() == 2 => {
            build_range_indices(dims[i], &pair[0], &pair[1])
          }
          _ => build_offset_indices(dims[i], item),
        })
        .collect::<Result<Vec<_>, _>>()?,
      _ => dims
        .iter()
        .map(|&n| build_offset_indices(n, origin))
        .collect::<Result<Vec<_>, _>>()?,
    }
  } else {
    dims
      .iter()
      .map(|&n| build_offset_indices(n, &Expr::Integer(1)))
      .collect::<Result<Vec<_>, _>>()?
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
  // `dim_indices[depth]` is the list of index values for that dimension.
  fn build_array(
    func: &Expr,
    dim_indices: &[Vec<Expr>],
    depth: usize,
    indices: &mut Vec<Expr>,
  ) -> Result<Expr, InterpreterError> {
    if depth >= dim_indices.len() {
      let index_args: Vec<Expr> = indices.clone();
      if index_args.len() == 1 {
        apply_func_ast(func, &index_args[0])
      } else {
        // For multi-arg, evaluate f[i1, i2, ...]
        match func {
          Expr::Identifier(name) => {
            crate::evaluator::evaluate_function_call_ast(name, &index_args)
          }
          // `Subscript[a, ##] &` — substitute each `#k`/`##` slot with the
          // index args, then evaluate the body so e.g.
          // `Array[Subscript[a, ##] &, {2, 2}]` produces concrete
          // `Subscript[a, i, j]` cells.
          Expr::Function { body } => {
            let substituted =
              crate::syntax::substitute_slots(body, &index_args);
            crate::evaluator::evaluate_expr_to_expr(&substituted)
          }
          // `Function[{i, j, …}, body]` — bind named params positionally.
          Expr::NamedFunction { params, body, .. } => {
            let bindings: Vec<(&str, &Expr)> = params
              .iter()
              .zip(index_args.iter())
              .map(|(p, a)| (p.as_str(), a))
              .collect();
            let substituted =
              crate::syntax::substitute_variables(body, &bindings);
            crate::evaluator::evaluate_expr_to_expr(&substituted)
          }
          _ => {
            // Fall back to invoking the function via the standard call form
            // (e.g. Composition, named symbols stored as values, …).
            let func_str = crate::syntax::expr_to_string(func);
            crate::evaluator::evaluate_function_call_ast(&func_str, &index_args)
          }
        }
      }
    } else {
      let mut items = Vec::new();
      for index_expr in &dim_indices[depth] {
        indices.push(index_expr.clone());
        items.push(build_array(func, dim_indices, depth + 1, indices)?);
        indices.pop();
      }
      Ok(Expr::List(items.into()))
    }
  }

  let result = build_array(func, &dim_indices, 0, &mut Vec::new())?;

  // If a custom head is specified, replace List at every level of the
  // generated result with that head and evaluate the substituted
  // expression. When head == "List" this is a no-op.
  if let Some(h) = head {
    if h == "List" {
      Ok(result)
    } else {
      fn replace_list_head(
        expr: &Expr,
        head: &str,
      ) -> Result<Expr, InterpreterError> {
        match expr {
          Expr::List(items) => {
            let new_items: Result<Vec<Expr>, InterpreterError> =
              items.iter().map(|it| replace_list_head(it, head)).collect();
            crate::evaluator::evaluate_function_call_ast(head, &new_items?)
          }
          _ => Ok(expr.clone()),
        }
      }
      replace_list_head(&result, h)
    }
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

/// Expand `Band[{i, j}, ...] -> v` rules into concrete `{i+k, j+k} -> v`
/// rules using `dims`. Other rules pass through unchanged. Returns None
/// when nothing in `data` references Band (so the caller can use the
/// original expression).
fn expand_band_rules(data: &Expr, dims: &[usize]) -> Option<Expr> {
  let is_band_rule = |it: &Expr| {
    matches!(it, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::FunctionCall { name, .. } if name == "Band"))
  };
  // Accept either a list of rules or a single bare `Band[...] -> v` rule.
  let items: Vec<Expr> = match data {
    Expr::List(items) => items.to_vec(),
    rule if is_band_rule(rule) => vec![rule.clone()],
    _ => return None,
  };
  if !items.iter().any(is_band_rule) {
    return None;
  }
  let mut expanded: Vec<Expr> = Vec::with_capacity(items.len());
  for item in items.iter() {
    if let Expr::Rule {
      pattern,
      replacement,
    } = item
      && let Expr::FunctionCall { name, args } = pattern.as_ref()
      && name == "Band"
      && !args.is_empty()
    {
      let Some(start) = list_of_positive_ints(&args[0], dims.len()) else {
        return None;
      };
      let end: Vec<usize> = if args.len() >= 2 {
        match list_of_positive_ints(&args[1], dims.len()) {
          Some(e) => e
            .into_iter()
            .enumerate()
            .map(|(i, v)| v.min(dims[i]))
            .collect(),
          None => return None,
        }
      } else {
        dims.to_vec()
      };
      let steps_possible: usize = (0..start.len())
        .map(|i| {
          if start[i] > end[i] {
            0
          } else {
            end[i] - start[i] + 1
          }
        })
        .min()
        .unwrap_or(0);
      for k in 0..steps_possible {
        let pos: Vec<Expr> = start
          .iter()
          .map(|&s| Expr::Integer((s + k) as i128))
          .collect();
        expanded.push(Expr::Rule {
          pattern: Box::new(Expr::List(pos.into())),
          replacement: replacement.clone(),
        });
      }
      continue;
    }
    expanded.push(item.clone());
  }
  Some(Expr::List(expanded.into()))
}

/// Expand a pattern rule such as `{i_} :> i` or `{i_, j_} :> i + j` (with an
/// optional `/; cond`) into explicit `{i, j, ...} -> value` rules over the
/// grid `dims`. The rule's left-hand side must be a `List` pattern (after
/// stripping any outer `Condition`) whose length equals the rank of `dims`.
/// Each position in the grid is tested with `MatchQ` and, when it matches,
/// the value is obtained with `Replace` (which binds the pattern variables to
/// the indices). Returns None when `data` is not such a pattern rule, leaving
/// other constructor forms untouched.
fn expand_pattern_rule(data: &Expr, dims: &[usize]) -> Option<Expr> {
  let lhs: &Expr = match data {
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => pattern,
    _ => return None,
  };
  // Strip an outer `Condition[inner, test]` to inspect the structural part.
  let structural: &Expr = match lhs {
    Expr::FunctionCall { name, args }
      if name == "Condition" && args.len() == 2 =>
    {
      &args[0]
    }
    other => other,
  };
  // Only the list-pattern form is handled (e.g. `{i_}`, `{i_, j_}`).
  let Expr::List(pats) = structural else {
    return None;
  };
  if pats.len() != dims.len() || dims.is_empty() {
    return None;
  }
  // It must actually be a pattern (contain a Blank), otherwise it is an
  // ordinary explicit position rule that the normal pipeline handles.
  if !pats.iter().any(expr_contains_pattern) {
    return None;
  }
  let total: usize = dims.iter().product();
  // Guard against pathological sizes.
  if total == 0 || total > 1_000_000 {
    return None;
  }
  let mut rules: Vec<Expr> = Vec::new();
  for flat in 0..total {
    let mut rem = flat;
    let mut pos = vec![0i128; dims.len()];
    for k in (0..dims.len()).rev() {
      pos[k] = (rem % dims[k]) as i128 + 1;
      rem /= dims[k];
    }
    let pos_list = Expr::List(pos.iter().map(|&p| Expr::Integer(p)).collect());
    let match_q = Expr::FunctionCall {
      name: "MatchQ".to_string(),
      args: vec![pos_list.clone(), lhs.clone()].into(),
    };
    let matched = matches!(
      crate::evaluator::evaluate_expr_to_expr(&match_q),
      Ok(Expr::Identifier(ref s)) if s == "True"
    );
    if !matched {
      continue;
    }
    let replaced = Expr::FunctionCall {
      name: "Replace".to_string(),
      args: vec![pos_list.clone(), data.clone()].into(),
    };
    let val = crate::evaluator::evaluate_expr_to_expr(&replaced).ok()?;
    rules.push(Expr::Rule {
      pattern: Box::new(pos_list),
      replacement: Box::new(val),
    });
  }
  Some(Expr::List(rules.into()))
}

/// Whether `expr` contains a pattern (Blank) node anywhere within it.
fn expr_contains_pattern(expr: &Expr) -> bool {
  match expr {
    Expr::Pattern { .. }
    | Expr::PatternOptional { .. }
    | Expr::PatternTest { .. } => true,
    Expr::List(items) => items.iter().any(expr_contains_pattern),
    Expr::FunctionCall { args, .. } => args.iter().any(expr_contains_pattern),
    Expr::BinaryOp { left, right, .. } => {
      expr_contains_pattern(left) || expr_contains_pattern(right)
    }
    Expr::UnaryOp { operand, .. } => expr_contains_pattern(operand),
    _ => false,
  }
}

fn list_of_positive_ints(e: &Expr, rank: usize) -> Option<Vec<usize>> {
  let Expr::List(items) = e else { return None };
  if items.len() != rank {
    return None;
  }
  let mut out = Vec::with_capacity(rank);
  for it in items.iter() {
    let n = expr_to_i128(it)?;
    if n < 1 {
      return None;
    }
    out.push(n as usize);
  }
  Some(out)
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
      args: args.to_vec().into(),
    });
  }

  if args.is_empty() || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "SparseArray".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Band[{i, j}] / Band[{i, j}, {iMax, jMax}] rules need to know the
  // dimensions to expand into concrete positions. When dims are
  // explicit, pre-expand and substitute back into args[0].
  let explicit_dims: Option<Vec<usize>> =
    args.get(1).and_then(parse_sparse_dims);
  let expanded_data: Expr;
  let data: &Expr = if let Some(ref dims) = explicit_dims {
    if let Some(expanded) = expand_pattern_rule(&args[0], dims)
      .or_else(|| expand_band_rules(&args[0], dims))
    {
      expanded_data = expanded;
      &expanded_data
    } else {
      &args[0]
    }
  } else {
    &args[0]
  };
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
        args: args.to_vec().into(),
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
            args: args.to_vec().into(),
          });
        }
        d
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "SparseArray".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    if inferred_dims.is_empty() {
      return Ok(Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: args.to_vec().into(),
      });
    }
    inferred_dims
  };

  // Drop rules that are out of bounds or equal to the default value. When
  // the same position appears more than once, the *first* rule wins — this
  // matches wolframscript:
  //   Normal[SparseArray[{1 -> 5, 1 -> 9}, 3]] == {5, 0, 0}
  // BTreeMap gives us deterministic lexicographic ordering.
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
      continue;
    }
    dedup.entry(pos).or_insert(val);
  }

  // Build the CSR-style fourth argument that wolframscript exposes for
  // SparseArray:
  //   {1, {row_ptr, inner_positions}, values}
  // where `row_ptr` groups entries by their first coordinate and
  // `inner_positions` lists the remaining coordinates for each entry.
  let sorted: Vec<(Vec<i128>, Expr)> = dedup.into_iter().collect();
  let dims_expr =
    Expr::List(dims.iter().map(|&n| Expr::Integer(n as i128)).collect());
  let rank = dims.len();

  // row_ptr length: for rank 1 Wolfram uses {0, nnz}; for rank >= 2 it
  // groups by the first axis, giving length dims[0] + 1.
  let row_ptr: Vec<i128> = if rank <= 1 {
    vec![0, sorted.len() as i128]
  } else {
    let mut ptr = vec![0i128; dims[0] + 1];
    for (pos, _) in &sorted {
      let row = pos[0] as usize; // 1-based row index
      for slot in &mut ptr[row..] {
        *slot += 1;
      }
    }
    ptr
  };

  let inner_positions: Vec<Expr> = sorted
    .iter()
    .map(|(pos, _)| {
      let tail: Vec<Expr> = if rank <= 1 {
        // 1D arrays still show a single-element inner position.
        pos.iter().cloned().map(Expr::Integer).collect()
      } else {
        pos[1..].iter().cloned().map(Expr::Integer).collect()
      };
      Expr::List(tail.into())
    })
    .collect();
  let values_list: Vec<Expr> = sorted.into_iter().map(|(_, v)| v).collect();

  let row_ptr_expr =
    Expr::List(row_ptr.into_iter().map(Expr::Integer).collect());
  let structure = Expr::List(
    vec![
      Expr::Integer(1),
      Expr::List(vec![row_ptr_expr, Expr::List(inner_positions.into())].into()),
      Expr::List(values_list.into()),
    ]
    .into(),
  );

  Ok(Expr::FunctionCall {
    name: "SparseArray".to_string(),
    args: vec![
      Expr::Identifier("Automatic".to_string()),
      dims_expr,
      default,
      structure,
    ]
    .into(),
  })
}

/// Extract (position, value) pairs from the fourth argument of a canonical
/// SparseArray. Accepts either the legacy list-of-rules form or the
/// CSR-style `{1, {row_ptr, inner_positions}, values}` form used by
/// wolframscript.
pub fn sparse_array_extract_rules(
  dims: &[usize],
  arg: &Expr,
) -> Vec<(Vec<i128>, Expr)> {
  let Expr::List(items) = arg else {
    return Vec::new();
  };

  // Legacy list-of-rules form
  if items.iter().all(|i| matches!(i, Expr::Rule { .. })) {
    return items
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
      .collect();
  }

  // CSR form: {1, {row_ptr, inner_positions}, values}
  if items.len() == 3
    && matches!(&items[0], Expr::Integer(1))
    && let (Expr::List(structure), Expr::List(values)) = (&items[1], &items[2])
    && structure.len() == 2
    && let (Expr::List(row_ptr), Expr::List(inner_list)) =
      (&structure[0], &structure[1])
  {
    let rank = dims.len();
    let row_ptr_vals: Option<Vec<i128>> =
      row_ptr.iter().map(expr_to_i128).collect();
    let Some(row_ptr_vals) = row_ptr_vals else {
      return Vec::new();
    };
    let mut out: Vec<(Vec<i128>, Expr)> = Vec::with_capacity(values.len());
    // Determine row for each entry from row_ptr cumulative counts.
    let num_rows = if rank <= 1 { 1 } else { dims[0] };
    let mut entry_idx: usize = 0;
    for row in 0..num_rows {
      let count = (row_ptr_vals.get(row + 1).copied().unwrap_or(0)
        - row_ptr_vals.get(row).copied().unwrap_or(0))
        as usize;
      for _ in 0..count {
        if entry_idx >= values.len() || entry_idx >= inner_list.len() {
          break;
        }
        let Expr::List(tail_exprs) = &inner_list[entry_idx] else {
          entry_idx += 1;
          continue;
        };
        let tail: Option<Vec<i128>> =
          tail_exprs.iter().map(expr_to_i128).collect();
        let Some(tail) = tail else {
          entry_idx += 1;
          continue;
        };
        let mut pos = Vec::with_capacity(rank);
        if rank >= 2 {
          pos.push((row + 1) as i128);
          pos.extend(tail);
        } else {
          pos.extend(tail);
        }
        out.push((pos, values[entry_idx].clone()));
        entry_idx += 1;
      }
    }
    return out;
  }

  Vec::new()
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
    return Expr::List(arr.into());
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
  Expr::List(result.into())
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
        args: vec![normalized].into(),
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
      args: vec![normalized].into(),
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
              args: vec![normalized.clone()].into(),
            });
          }
        }
      }
      d
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Normal".to_string(),
        args: vec![normalized].into(),
      });
    }
  };
  let default = &sa_args[2];
  let rules_list = sparse_array_extract_rules(&dims, &sa_args[3]);
  Ok(build_dense_from_rules(&dims, default, &rules_list))
}

/// Unified Tuples: `Tuples[{list1, list2, …}]` (one element from each
/// list, tuples take the outer head), `Tuples[list, n]` (all n-tuples,
/// tuples take list's head) and `Tuples[list, {n1, n2, …}]` (all
/// n1×n2×… arrays, list's head at every level; `{}` yields scalars).
/// Atomic expressions emit ::normal (with `{1, i}` positions for atomic
/// elements in the one-argument form) and invalid specs emit ::ilsmn.
pub fn tuples_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Expr::FunctionCall {
    name: "Tuples".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);
  let emit_normal = |position: &str| {
    crate::emit_message(&format!(
      "Tuples::normal: Nonatomic expression expected at position {} in {}.",
      position,
      show(&original())
    ));
  };

  // Elements and head of a nonatomic expression (None for the List head).
  fn parts(e: &Expr) -> Option<(&[Expr], Option<&str>)> {
    match e {
      Expr::List(items) => Some((items.as_slice(), None)),
      Expr::FunctionCall { name, args } => {
        Some((args.as_slice(), Some(name.as_str())))
      }
      _ => None,
    }
  }
  let wrap = |head: Option<&str>, elems: Vec<Expr>| -> Expr {
    match head {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: elems.into(),
      },
      None => Expr::List(elems.into()),
    }
  };

  if args.len() == 1 {
    // Tuples[{list1, list2, ...}] - one element from each list
    let Some((outer_items, outer_head)) = parts(&args[0]) else {
      emit_normal("1");
      return Ok(original());
    };
    let mut lists: Vec<&[Expr]> = Vec::new();
    for (i, item) in outer_items.iter().enumerate() {
      let Some((elems, _)) = parts(item) else {
        emit_normal(&format!("{{1, {}}}", i + 1));
        return Ok(original());
      };
      lists.push(elems);
    }

    let mut result: Vec<Vec<Expr>> = vec![vec![]];
    for list in &lists {
      let mut new_result = Vec::new();
      for tuple in &result {
        for item in *list {
          let mut new_tuple = tuple.clone();
          new_tuple.push(item.clone());
          new_result.push(new_tuple);
        }
      }
      result = new_result;
    }
    return Ok(Expr::List(
      result
        .into_iter()
        .map(|v| wrap(outer_head, v))
        .collect::<Vec<_>>()
        .into(),
    ));
  }

  // Tuples[list, n] or Tuples[list, {n1, n2, ...}]
  let Some((items, head)) = parts(&args[0]) else {
    emit_normal("1");
    return Ok(original());
  };

  let machine_nonneg = |e: &Expr| -> Option<usize> {
    match e {
      Expr::Integer(n) if (0..=i64::MAX as i128).contains(n) => {
        Some(*n as usize)
      }
      _ => None,
    }
  };
  let dims: Option<Vec<usize>> = match &args[1] {
    Expr::List(spec) => spec.iter().map(machine_nonneg).collect(),
    e => machine_nonneg(e).map(|n| vec![n]),
  };
  let Some(dims) = dims else {
    crate::emit_message(&format!(
      "Tuples::ilsmn: Single or list of non-negative machine-sized integers expected at position 2 of {}.",
      show(&original())
    ));
    return Ok(original());
  };

  // All flat tuples of length n1*n2*..., in lexicographic order.
  let total: usize = dims.iter().product();
  let mut flat_tuples: Vec<Vec<Expr>> = vec![vec![]];
  for _ in 0..total {
    let mut new_result = Vec::new();
    for tuple in &flat_tuples {
      for item in items {
        let mut new_tuple = tuple.clone();
        new_tuple.push(item.clone());
        new_result.push(new_tuple);
      }
    }
    flat_tuples = new_result;
  }

  // Reshape a flat tuple into nested arrays of the given shape, applying
  // the subject's head at every level. An empty shape yields the bare
  // element (so a scalar n behaves identically to the shape {n}).
  fn reshape(
    flat: &[Expr],
    dims: &[usize],
    wrap: &dyn Fn(Option<&str>, Vec<Expr>) -> Expr,
    head: Option<&str>,
  ) -> Expr {
    if dims.is_empty() {
      return flat[0].clone();
    }
    let chunk: usize = dims[1..].iter().product();
    let elems: Vec<Expr> = (0..dims[0])
      .map(|i| {
        reshape(&flat[i * chunk..(i + 1) * chunk], &dims[1..], wrap, head)
      })
      .collect();
    wrap(head, elems)
  }

  Ok(Expr::List(
    flat_tuples
      .into_iter()
      .map(|t| reshape(&t, &dims, &wrap, head))
      .collect::<Vec<_>>()
      .into(),
  ))
}

/// DistanceMatrix[{v1, v2, ...}] - matrix of pairwise distances.
///
/// Returns a symmetric matrix with a zero diagonal where entry (i, j) is the
/// distance between vi and vj. The default distance is EuclideanDistance.
/// An optional `DistanceFunction -> f` rule selects a different distance
/// function (e.g. ManhattanDistance, SquaredEuclideanDistance, or any binary
/// function f).
pub fn distance_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Return the unevaluated form for any argument shape that DistanceMatrix
  // cannot process (matching wolframscript, which leaves it symbolic).
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "DistanceMatrix".to_string(),
      args: args.to_vec().into(),
    })
  };

  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }

  let points = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => return unevaluated(),
  };

  // Determine the distance function: default EuclideanDistance, or the
  // replacement of an optional `DistanceFunction -> f` rule.
  let mut dist_fn = Expr::Identifier("EuclideanDistance".to_string());
  if let Some(opt) = args.get(1) {
    match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(
        pattern.as_ref(),
        Expr::Identifier(name) if name == "DistanceFunction"
      ) =>
      {
        dist_fn = (**replacement).clone();
      }
      _ => return unevaluated(),
    }
  }

  let n = points.len();
  let mut rows: Vec<Expr> = Vec::with_capacity(n);
  for i in 0..n {
    let mut row: Vec<Expr> = Vec::with_capacity(n);
    for j in 0..n {
      if i == j {
        row.push(Expr::Integer(0));
        continue;
      }
      let dist = apply_func_to_two_args(&dist_fn, &points[i], &points[j])?;
      row.push(dist);
    }
    rows.push(Expr::List(row.into()));
  }

  Ok(Expr::List(rows.into()))
}
