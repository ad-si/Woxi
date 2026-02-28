#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;
use std::cmp::Ordering;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Extract sub-intervals from an Interval expression.
/// Returns Some(vec of (lo, hi) pairs) if the expr is Interval[...].
pub fn is_interval(expr: &Expr) -> Option<Vec<(&Expr, &Expr)>> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Interval"
  {
    let mut spans = Vec::new();
    for arg in args {
      if let Expr::List(pair) = arg
        && pair.len() == 2
      {
        spans.push((&pair[0], &pair[1]));
      } else {
        return None;
      }
    }
    return Some(spans);
  }
  None
}

/// Check if any expression in the slice is an Interval
fn has_interval(args: &[Expr]) -> bool {
  args.iter().any(|a| is_interval(a).is_some())
}

/// Compare two numeric expressions. Returns None if not comparable.
fn compare_numeric(a: &Expr, b: &Expr) -> Option<Ordering> {
  let fa = expr_to_f64(a)?;
  let fb = expr_to_f64(b)?;
  fa.partial_cmp(&fb)
}

/// Try to convert an expr to f64, handling Infinity, rationals, etc.
fn expr_to_f64(expr: &Expr) -> Option<f64> {
  crate::functions::math_ast::try_eval_to_f64_with_infinity(expr)
}

/// Compute the numeric minimum of two expressions.
fn numeric_min(a: &Expr, b: &Expr) -> Expr {
  match compare_numeric(a, b) {
    Some(Ordering::Less | Ordering::Equal) => a.clone(),
    Some(Ordering::Greater) => b.clone(),
    None => a.clone(), // fallback
  }
}

/// Compute the numeric maximum of two expressions.
fn numeric_max(a: &Expr, b: &Expr) -> Expr {
  match compare_numeric(a, b) {
    Some(Ordering::Greater | Ordering::Equal) => a.clone(),
    Some(Ordering::Less) => b.clone(),
    None => a.clone(), // fallback
  }
}

/// Sort endpoints: ensure lo <= hi.
fn sort_endpoints(lo: Expr, hi: Expr) -> (Expr, Expr) {
  match compare_numeric(&lo, &hi) {
    Some(Ordering::Greater) => (hi, lo),
    _ => (lo, hi),
  }
}

/// Normalize a list of (lo, hi) spans: sort by lower bound, merge overlapping/adjacent.
fn normalize_intervals(mut spans: Vec<(Expr, Expr)>) -> Vec<(Expr, Expr)> {
  if spans.is_empty() {
    return spans;
  }

  // Sort endpoints within each span
  spans = spans
    .into_iter()
    .map(|(lo, hi)| sort_endpoints(lo, hi))
    .collect();

  // Sort spans by lower bound
  spans.sort_by(|a, b| compare_numeric(&a.0, &b.0).unwrap_or(Ordering::Equal));

  let mut merged: Vec<(Expr, Expr)> = Vec::new();
  merged.push(spans.remove(0));

  for (lo, hi) in spans {
    let last = merged.last_mut().unwrap();
    // Check if current span overlaps or is adjacent to the last merged span
    // Overlapping: lo <= last.hi (using <=, not <, for adjacency)
    match compare_numeric(&lo, &last.1) {
      Some(Ordering::Less | Ordering::Equal) => {
        // Merge: extend the upper bound if needed
        last.1 = numeric_max(&last.1, &hi);
      }
      _ => {
        merged.push((lo, hi));
      }
    }
  }

  merged
}

/// Build an Interval expression from normalized spans.
fn make_interval(spans: Vec<(Expr, Expr)>) -> Expr {
  let args: Vec<Expr> = spans
    .into_iter()
    .map(|(lo, hi)| Expr::List(vec![lo, hi]))
    .collect();
  Expr::FunctionCall {
    name: "Interval".to_string(),
    args,
  }
}

/// Evaluate an arithmetic expression on two endpoints using the evaluator.
fn eval_binop(a: &Expr, b: &Expr, op: &str) -> Result<Expr, InterpreterError> {
  match op {
    "Plus" => crate::functions::math_ast::plus_ast(&[a.clone(), b.clone()]),
    "Times" => crate::functions::math_ast::times_ast(&[a.clone(), b.clone()]),
    "Power" => crate::functions::math_ast::power_two(a, b),
    "Divide" => crate::functions::math_ast::divide_two(a, b),
    _ => unreachable!(),
  }
}

/// Min/max of four expressions
fn min_of_four(a: &Expr, b: &Expr, c: &Expr, d: &Expr) -> Expr {
  numeric_min(&numeric_min(a, b), &numeric_min(c, d))
}

fn max_of_four(a: &Expr, b: &Expr, c: &Expr, d: &Expr) -> Expr {
  numeric_max(&numeric_max(a, b), &numeric_max(c, d))
}

// ─── Construction ───────────────────────────────────────────────────────────

/// Interval[{a,b}, {c,d}, ...] — construct an interval, normalizing spans.
pub fn interval_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Interval[] → Interval[] (empty interval)
  if args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Interval".to_string(),
      args: vec![],
    });
  }

  let mut spans = Vec::new();
  for arg in args {
    if let Expr::List(pair) = arg
      && pair.len() == 2
    {
      spans.push((pair[0].clone(), pair[1].clone()));
    } else {
      // Invalid argument — return unevaluated
      return Ok(Expr::FunctionCall {
        name: "Interval".to_string(),
        args: args.to_vec(),
      });
    }
  }

  let normalized = normalize_intervals(spans);
  Ok(make_interval(normalized))
}

// ─── Set Operations ─────────────────────────────────────────────────────────

/// IntervalUnion[i1, i2, ...] — union of intervals.
pub fn interval_union_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut all_spans = Vec::new();
  for arg in args {
    if let Some(spans) = is_interval(arg) {
      for (lo, hi) in spans {
        all_spans.push((lo.clone(), hi.clone()));
      }
    } else {
      // Non-interval arg: return unevaluated
      return Ok(Expr::FunctionCall {
        name: "IntervalUnion".to_string(),
        args: args.to_vec(),
      });
    }
  }
  let normalized = normalize_intervals(all_spans);
  Ok(make_interval(normalized))
}

/// IntervalIntersection[i1, i2, ...] — intersection of intervals.
pub fn interval_intersection_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Interval".to_string(),
      args: vec![],
    });
  }

  // Start with spans of first argument
  let first_spans = match is_interval(&args[0]) {
    Some(spans) => spans
      .into_iter()
      .map(|(lo, hi)| (lo.clone(), hi.clone()))
      .collect::<Vec<_>>(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "IntervalIntersection".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut current = first_spans;

  for arg in &args[1..] {
    let other = match is_interval(arg) {
      Some(spans) => spans
        .into_iter()
        .map(|(lo, hi)| (lo.clone(), hi.clone()))
        .collect::<Vec<_>>(),
      None => {
        return Ok(Expr::FunctionCall {
          name: "IntervalIntersection".to_string(),
          args: args.to_vec(),
        });
      }
    };

    // Pairwise intersection of current spans with other spans
    let mut new_spans = Vec::new();
    for (a_lo, a_hi) in &current {
      for (b_lo, b_hi) in &other {
        let lo = numeric_max(a_lo, b_lo);
        let hi = numeric_min(a_hi, b_hi);
        // Only include if lo <= hi (valid span)
        if let Some(Ordering::Less | Ordering::Equal) =
          compare_numeric(&lo, &hi)
        {
          new_spans.push((lo, hi));
        }
      }
    }
    current = normalize_intervals(new_spans);
  }

  Ok(make_interval(current))
}

/// IntervalMemberQ[interval, x] — test membership.
pub fn interval_member_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "IntervalMemberQ expects exactly 2 arguments".into(),
    ));
  }

  let spans = match is_interval(&args[0]) {
    Some(s) => s,
    None => {
      return Ok(Expr::FunctionCall {
        name: "IntervalMemberQ".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Second arg could be a point or an Interval
  if let Some(sub_spans) = is_interval(&args[1]) {
    // Sub-interval check: every sub-span of args[1] must be contained in some span of args[0]
    for (sub_lo, sub_hi) in &sub_spans {
      let mut contained = false;
      for (lo, hi) in &spans {
        if let (
          Some(Ordering::Less | Ordering::Equal),
          Some(Ordering::Less | Ordering::Equal),
        ) = (compare_numeric(lo, sub_lo), compare_numeric(sub_hi, hi))
        {
          contained = true;
          break;
        }
      }
      if !contained {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
    Ok(Expr::Identifier("True".to_string()))
  } else {
    // Point membership
    let point = &args[1];
    for (lo, hi) in &spans {
      if let (
        Some(Ordering::Less | Ordering::Equal),
        Some(Ordering::Less | Ordering::Equal),
      ) = (compare_numeric(lo, point), compare_numeric(point, hi))
      {
        return Ok(Expr::Identifier("True".to_string()));
      }
    }
    Ok(Expr::Identifier("False".to_string()))
  }
}

// ─── Arithmetic Hooks ───────────────────────────────────────────────────────

/// Hook for Plus: Interval + Interval, Interval + scalar, scalar + Interval.
pub fn try_interval_plus(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if !has_interval(args) {
    return None;
  }

  // Separate intervals and scalars, wrapping scalars as Interval[{x, x}]
  let mut current_spans: Vec<(Expr, Expr)> = vec![];
  let mut first = true;

  for arg in args {
    let arg_spans = if let Some(spans) = is_interval(arg) {
      spans
        .into_iter()
        .map(|(lo, hi)| (lo.clone(), hi.clone()))
        .collect::<Vec<_>>()
    } else {
      // Scalar: treat as Interval[{x, x}]
      vec![(arg.clone(), arg.clone())]
    };

    if first {
      current_spans = arg_spans;
      first = false;
    } else {
      // Add each pair of spans
      let mut new_spans = Vec::new();
      for (a_lo, a_hi) in &current_spans {
        for (b_lo, b_hi) in &arg_spans {
          match (
            eval_binop(a_lo, b_lo, "Plus"),
            eval_binop(a_hi, b_hi, "Plus"),
          ) {
            (Ok(lo), Ok(hi)) => new_spans.push((lo, hi)),
            (Err(e), _) | (_, Err(e)) => return Some(Err(e)),
          }
        }
      }
      current_spans = normalize_intervals(new_spans);
    }
  }

  Some(Ok(make_interval(normalize_intervals(current_spans))))
}

/// Hook for Times: Interval * Interval, scalar * Interval, etc.
pub fn try_interval_times(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if !has_interval(args) {
    return None;
  }

  let mut current_spans: Vec<(Expr, Expr)> = vec![];
  let mut first = true;

  for arg in args {
    let arg_spans = if let Some(spans) = is_interval(arg) {
      spans
        .into_iter()
        .map(|(lo, hi)| (lo.clone(), hi.clone()))
        .collect::<Vec<_>>()
    } else {
      vec![(arg.clone(), arg.clone())]
    };

    if first {
      current_spans = arg_spans;
      first = false;
    } else {
      let mut new_spans = Vec::new();
      for (a_lo, a_hi) in &current_spans {
        for (b_lo, b_hi) in &arg_spans {
          // Compute all four products and take min/max
          match (
            eval_binop(a_lo, b_lo, "Times"),
            eval_binop(a_lo, b_hi, "Times"),
            eval_binop(a_hi, b_lo, "Times"),
            eval_binop(a_hi, b_hi, "Times"),
          ) {
            (Ok(p1), Ok(p2), Ok(p3), Ok(p4)) => {
              let lo = min_of_four(&p1, &p2, &p3, &p4);
              let hi = max_of_four(&p1, &p2, &p3, &p4);
              new_spans.push((lo, hi));
            }
            _ => {
              // If any product fails, return unevaluated
              return None;
            }
          }
        }
      }
      current_spans = normalize_intervals(new_spans);
    }
  }

  Some(Ok(make_interval(normalize_intervals(current_spans))))
}

/// Hook for Divide: a / Interval or Interval / b.
pub fn try_interval_divide(
  a: &Expr,
  b: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let a_int = is_interval(a);
  let b_int = is_interval(b);

  if a_int.is_none() && b_int.is_none() {
    return None;
  }

  // Convert b to reciprocal interval, then multiply
  let recip_spans = if let Some(spans) = &b_int {
    let mut recip = Vec::new();
    for (lo, hi) in spans {
      // Check if interval spans zero — undefined
      let lo_f = expr_to_f64(lo)?;
      let hi_f = expr_to_f64(hi)?;
      if lo_f <= 0.0 && hi_f >= 0.0 && !(lo_f == 0.0 && hi_f == 0.0) {
        // Interval contains zero — can't compute reciprocal cleanly
        // For now, return unevaluated if it fully spans zero
        if lo_f < 0.0 && hi_f > 0.0 {
          return None;
        }
      }
      match (
        eval_binop(&Expr::Integer(1), lo, "Divide"),
        eval_binop(&Expr::Integer(1), hi, "Divide"),
      ) {
        (Ok(r_lo), Ok(r_hi)) => {
          recip.push(sort_endpoints(r_lo, r_hi));
        }
        _ => return None,
      }
    }
    recip
  } else {
    // b is scalar, reciprocal is {1/b, 1/b}
    match eval_binop(&Expr::Integer(1), b, "Divide") {
      Ok(r) => vec![(r.clone(), r)],
      Err(_) => return None,
    }
  };

  let a_spans = if let Some(spans) = &a_int {
    spans
      .iter()
      .map(|(lo, hi)| ((*lo).clone(), (*hi).clone()))
      .collect::<Vec<_>>()
  } else {
    vec![(a.clone(), a.clone())]
  };

  // Multiply a_spans by recip_spans
  let mut result_spans = Vec::new();
  for (a_lo, a_hi) in &a_spans {
    for (b_lo, b_hi) in &recip_spans {
      match (
        eval_binop(a_lo, b_lo, "Times"),
        eval_binop(a_lo, b_hi, "Times"),
        eval_binop(a_hi, b_lo, "Times"),
        eval_binop(a_hi, b_hi, "Times"),
      ) {
        (Ok(p1), Ok(p2), Ok(p3), Ok(p4)) => {
          let lo = min_of_four(&p1, &p2, &p3, &p4);
          let hi = max_of_four(&p1, &p2, &p3, &p4);
          result_spans.push((lo, hi));
        }
        _ => return None,
      }
    }
  }

  Some(Ok(make_interval(normalize_intervals(result_spans))))
}

/// Hook for Power: Interval^n.
pub fn try_interval_power(
  base: &Expr,
  exp: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let spans = is_interval(base)?;

  // exp must be numeric
  let exp_f = expr_to_f64(exp)?;

  // Negative exponent: compute reciprocal of base^|exp|
  if exp_f < 0.0 {
    let pos_exp = match exp {
      Expr::Integer(n) => Expr::Integer(-n),
      Expr::Real(f) => Expr::Real(-f),
      _ => return None,
    };
    // First compute base^|exp|
    let powered = try_interval_power(base, &pos_exp)?;
    let powered = match powered {
      Ok(p) => p,
      Err(e) => return Some(Err(e)),
    };
    // Then compute reciprocal
    return try_interval_divide(&Expr::Integer(1), &powered);
  }

  let exp_int = if let Expr::Integer(n) = exp {
    Some(*n)
  } else {
    None
  };
  let is_even = exp_int.map(|n| n % 2 == 0).unwrap_or(false);

  let mut result_spans = Vec::new();
  for (lo, hi) in &spans {
    if is_even {
      // Even power: need to handle sign carefully
      let lo_f = expr_to_f64(lo)?;
      let hi_f = expr_to_f64(hi)?;

      if lo_f >= 0.0 {
        // Both non-negative: lo^n .. hi^n
        match (eval_binop(lo, exp, "Power"), eval_binop(hi, exp, "Power")) {
          (Ok(r_lo), Ok(r_hi)) => result_spans.push((r_lo, r_hi)),
          _ => return None,
        }
      } else if hi_f <= 0.0 {
        // Both non-positive: hi^n .. lo^n (reversed because even power)
        match (eval_binop(hi, exp, "Power"), eval_binop(lo, exp, "Power")) {
          (Ok(r_lo), Ok(r_hi)) => result_spans.push((r_lo, r_hi)),
          _ => return None,
        }
      } else {
        // Spans zero: 0 .. max(lo^n, hi^n)
        match (eval_binop(lo, exp, "Power"), eval_binop(hi, exp, "Power")) {
          (Ok(r_lo), Ok(r_hi)) => {
            let max_val = numeric_max(&r_lo, &r_hi);
            result_spans.push((Expr::Integer(0), max_val));
          }
          _ => return None,
        }
      }
    } else {
      // Odd power or non-integer: lo^n .. hi^n (preserves order)
      match (eval_binop(lo, exp, "Power"), eval_binop(hi, exp, "Power")) {
        (Ok(r_lo), Ok(r_hi)) => {
          result_spans.push(sort_endpoints(r_lo, r_hi));
        }
        _ => return None,
      }
    }
  }

  Some(Ok(make_interval(normalize_intervals(result_spans))))
}

/// Hook for Min: Min[..., Interval[{a,b}], ...] → extract lowest endpoint.
pub fn try_interval_min(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if !has_interval(args) {
    return None;
  }

  // Collect all "minimum candidate" values
  let mut candidates: Vec<Expr> = Vec::new();
  for arg in args {
    if let Some(spans) = is_interval(arg) {
      // The min of an interval is its lowest lower bound
      for (lo, _hi) in spans {
        candidates.push(lo.clone());
      }
    } else {
      candidates.push(arg.clone());
    }
  }

  // Find the minimum among candidates
  if candidates.is_empty() {
    return None;
  }
  let mut best = candidates[0].clone();
  for c in &candidates[1..] {
    best = numeric_min(&best, c);
  }

  Some(Ok(best))
}

/// Hook for Max: Max[..., Interval[{a,b}], ...] → extract highest endpoint.
pub fn try_interval_max(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if !has_interval(args) {
    return None;
  }

  let mut candidates: Vec<Expr> = Vec::new();
  for arg in args {
    if let Some(spans) = is_interval(arg) {
      for (_lo, hi) in spans {
        candidates.push(hi.clone());
      }
    } else {
      candidates.push(arg.clone());
    }
  }

  if candidates.is_empty() {
    return None;
  }
  let mut best = candidates[0].clone();
  for c in &candidates[1..] {
    best = numeric_max(&best, c);
  }

  Some(Ok(best))
}

// ─── Comparison Hooks ───────────────────────────────────────────────────────

/// Check interval comparison for Less/Greater/LessEqual/GreaterEqual.
/// Returns Some(true/false) for definite results, None for unevaluated (overlapping).
pub fn try_interval_compare(
  args: &[Expr],
  cmp_name: &str,
) -> Option<Result<Expr, InterpreterError>> {
  if !has_interval(args) {
    return None;
  }

  if args.len() != 2 {
    return None;
  }

  // Extract or wrap as intervals
  let a_spans = if let Some(spans) = is_interval(&args[0]) {
    spans
      .into_iter()
      .map(|(lo, hi)| (lo.clone(), hi.clone()))
      .collect::<Vec<_>>()
  } else {
    let val = &args[0];
    // Must be numeric for comparison
    expr_to_f64(val)?;
    vec![(val.clone(), val.clone())]
  };

  let b_spans = if let Some(spans) = is_interval(&args[1]) {
    spans
      .into_iter()
      .map(|(lo, hi)| (lo.clone(), hi.clone()))
      .collect::<Vec<_>>()
  } else {
    let val = &args[1];
    expr_to_f64(val)?;
    vec![(val.clone(), val.clone())]
  };

  // Get overall bounds
  let a_lo = a_spans
    .iter()
    .map(|(lo, _)| lo)
    .fold(a_spans[0].0.clone(), |acc, x| numeric_min(&acc, x));
  let a_hi = a_spans
    .iter()
    .map(|(_, hi)| hi)
    .fold(a_spans[0].1.clone(), |acc, x| numeric_max(&acc, x));
  let b_lo = b_spans
    .iter()
    .map(|(lo, _)| lo)
    .fold(b_spans[0].0.clone(), |acc, x| numeric_min(&acc, x));
  let b_hi = b_spans
    .iter()
    .map(|(_, hi)| hi)
    .fold(b_spans[0].1.clone(), |acc, x| numeric_max(&acc, x));

  match cmp_name {
    "Less" => {
      // True if a_hi < b_lo (all of a is less than all of b)
      if let Some(Ordering::Less) = compare_numeric(&a_hi, &b_lo) {
        return Some(Ok(Expr::Identifier("True".to_string())));
      }
      // False if a_lo >= b_hi (some of a is >= some of b)
      if let Some(Ordering::Greater | Ordering::Equal) =
        compare_numeric(&a_lo, &b_hi)
      {
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      // Overlapping: return unevaluated
      Some(Ok(Expr::FunctionCall {
        name: "Less".to_string(),
        args: args.to_vec(),
      }))
    }
    "Greater" => {
      // True if a_lo > b_hi
      if let Some(Ordering::Greater) = compare_numeric(&a_lo, &b_hi) {
        return Some(Ok(Expr::Identifier("True".to_string())));
      }
      // False if a_hi <= b_lo
      if let Some(Ordering::Less | Ordering::Equal) =
        compare_numeric(&a_hi, &b_lo)
      {
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      Some(Ok(Expr::FunctionCall {
        name: "Greater".to_string(),
        args: args.to_vec(),
      }))
    }
    "LessEqual" => {
      // True if a_hi <= b_lo
      if let Some(Ordering::Less | Ordering::Equal) =
        compare_numeric(&a_hi, &b_lo)
      {
        return Some(Ok(Expr::Identifier("True".to_string())));
      }
      // False if a_lo > b_hi
      if let Some(Ordering::Greater) = compare_numeric(&a_lo, &b_hi) {
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      Some(Ok(Expr::FunctionCall {
        name: "LessEqual".to_string(),
        args: args.to_vec(),
      }))
    }
    "GreaterEqual" => {
      // True if a_lo >= b_hi
      if let Some(Ordering::Greater | Ordering::Equal) =
        compare_numeric(&a_lo, &b_hi)
      {
        return Some(Ok(Expr::Identifier("True".to_string())));
      }
      // False if a_hi < b_lo
      if let Some(Ordering::Less) = compare_numeric(&a_hi, &b_lo) {
        return Some(Ok(Expr::Identifier("False".to_string())));
      }
      Some(Ok(Expr::FunctionCall {
        name: "GreaterEqual".to_string(),
        args: args.to_vec(),
      }))
    }
    _ => None,
  }
}
