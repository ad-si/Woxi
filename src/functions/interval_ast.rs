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

/// Apply a monotonic function (`Floor`, `Ceiling`, `Round`, `IntegerPart`,
/// `Sqrt`, `Exp`, `Log`, …) to an `Interval[...]`, mapping each span's
/// endpoints and renormalizing (which re-sorts, so both increasing and
/// decreasing monotonic functions are handled). Returns `None` unless `expr`
/// is an interval whose endpoints — and their images — are real and numeric,
/// so an out-of-domain endpoint (e.g. `Sqrt` of a negative bound, giving a
/// complex image) leaves the call unevaluated rather than producing a bogus
/// interval.
pub fn map_monotonic_interval(head: &str, expr: &Expr) -> Option<Expr> {
  let spans = is_interval(expr)?;
  let mut out: Vec<(Expr, Expr)> = Vec::with_capacity(spans.len());
  for (a, b) in spans {
    expr_to_f64(a)?;
    expr_to_f64(b)?;
    let fa =
      crate::evaluator::evaluate_function_call_ast(head, &[a.clone()]).ok()?;
    let fb =
      crate::evaluator::evaluate_function_call_ast(head, &[b.clone()]).ok()?;
    // Require real-numeric images so the endpoints stay orderable.
    expr_to_f64(&fa)?;
    expr_to_f64(&fb)?;
    out.push((fa, fb));
  }
  Some(make_interval(normalize_intervals(out)))
}

/// True if some `offset + period*k` (integer `k`) lies in `[af, bf]`.
fn critical_point_in(offset: f64, period: f64, af: f64, bf: f64) -> bool {
  let eps = 1e-9;
  let k_lo = ((af - offset) / period - eps).ceil();
  let k_hi = ((bf - offset) / period + eps).floor();
  k_lo <= k_hi
}

/// `Sin[Interval[...]]` / `Cos[Interval[...]]` — the range of the (bounded)
/// trig function over each span. The bounds are the endpoint images unless a
/// maximum point (value 1) or minimum point (value -1) falls inside the span.
/// Returns `None` unless every endpoint and its image is real-numeric.
pub fn trig_interval(head: &str, expr: &Expr) -> Option<Expr> {
  // (offset of a maximum, offset of a minimum), period 2*Pi.
  let (max_offset, min_offset) = match head {
    "Sin" => (std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2),
    "Cos" => (0.0, std::f64::consts::PI),
    _ => return None,
  };
  let period = std::f64::consts::TAU;
  let spans = is_interval(expr)?;
  let mut out: Vec<(Expr, Expr)> = Vec::with_capacity(spans.len());
  for (a, b) in spans {
    let af = expr_to_f64(a)?;
    let bf = expr_to_f64(b)?;
    // Endpoint images, kept symbolic (1/2, Sqrt[3]/2, …) but required numeric.
    let fa =
      crate::evaluator::evaluate_function_call_ast(head, &[a.clone()]).ok()?;
    let fb =
      crate::evaluator::evaluate_function_call_ast(head, &[b.clone()]).ok()?;
    expr_to_f64(&fa)?;
    expr_to_f64(&fb)?;
    let lo = if critical_point_in(min_offset, period, af, bf) {
      Expr::Integer(-1)
    } else {
      numeric_min(&fa, &fb)
    };
    let hi = if critical_point_in(max_offset, period, af, bf) {
      Expr::Integer(1)
    } else {
      numeric_max(&fa, &fb)
    };
    out.push((lo, hi));
  }
  Some(make_interval(normalize_intervals(out)))
}

/// `Abs[Interval[...]]` — the interval of absolute values. For each span
/// `[a, b]`: if it contains 0 the result runs from 0 to max(|a|, |b|);
/// otherwise from min(|a|, |b|) to max(|a|, |b|). Spans are renormalized
/// (merging any that now overlap). Returns `None` unless every endpoint is
/// numeric.
pub fn abs_interval(expr: &Expr) -> Option<Expr> {
  let spans = is_interval(expr)?;
  let zero = Expr::Integer(0);
  let mut out: Vec<(Expr, Expr)> = Vec::with_capacity(spans.len());
  for (a, b) in spans {
    // Require numeric endpoints; otherwise leave Abs unevaluated.
    expr_to_f64(a)?;
    expr_to_f64(b)?;
    let abs_a =
      crate::evaluator::evaluate_function_call_ast("Abs", &[a.clone()]).ok()?;
    let abs_b =
      crate::evaluator::evaluate_function_call_ast("Abs", &[b.clone()]).ok()?;
    let contains_zero = matches!(
      compare_numeric(a, &zero),
      Some(Ordering::Less | Ordering::Equal)
    ) && matches!(
      compare_numeric(b, &zero),
      Some(Ordering::Greater | Ordering::Equal)
    );
    let lo = if contains_zero {
      zero.clone()
    } else {
      numeric_min(&abs_a, &abs_b)
    };
    let hi = numeric_max(&abs_a, &abs_b);
    out.push((lo, hi));
  }
  Some(make_interval(normalize_intervals(out)))
}

/// `Cosh[Interval[...]]` — the range of `Cosh` over each span. `Cosh` is even
/// with a minimum value of `1` at `x = 0`, so a span containing `0` runs from
/// `1` to `max(Cosh[a], Cosh[b])`; a span on one side of `0` is monotonic, so
/// it runs between the endpoint images. Returns `None` unless every endpoint
/// is numeric.
pub fn cosh_interval(expr: &Expr) -> Option<Expr> {
  let spans = is_interval(expr)?;
  let zero = Expr::Integer(0);
  let mut out: Vec<(Expr, Expr)> = Vec::with_capacity(spans.len());
  for (a, b) in spans {
    expr_to_f64(a)?;
    expr_to_f64(b)?;
    let ca = crate::evaluator::evaluate_function_call_ast("Cosh", &[a.clone()])
      .ok()?;
    let cb = crate::evaluator::evaluate_function_call_ast("Cosh", &[b.clone()])
      .ok()?;
    expr_to_f64(&ca)?;
    expr_to_f64(&cb)?;
    let contains_zero = matches!(
      compare_numeric(a, &zero),
      Some(Ordering::Less | Ordering::Equal)
    ) && matches!(
      compare_numeric(b, &zero),
      Some(Ordering::Greater | Ordering::Equal)
    );
    let lo = if contains_zero {
      Expr::Integer(1)
    } else {
      numeric_min(&ca, &cb)
    };
    let hi = numeric_max(&ca, &cb);
    out.push((lo, hi));
  }
  Some(make_interval(normalize_intervals(out)))
}

/// Count the poles `offset + k*period` (integer `k`) strictly inside `(af, bf)`.
fn count_poles_in(offset: f64, period: f64, af: f64, bf: f64) -> i64 {
  let eps = 1e-9;
  let lo_k = (af - offset) / period;
  let hi_k = (bf - offset) / period;
  let first = (lo_k + eps).floor() as i64 + 1; // smallest k strictly > lo_k
  let last = (hi_k - eps).ceil() as i64 - 1; // largest k strictly < hi_k
  (last - first + 1).max(0)
}

/// `Tan[Interval[...]]` / `Cot[Interval[...]]` — the range over each span,
/// accounting for the poles at `Pi/2 + k Pi` (Tan) or `k Pi` (Cot). Both are
/// strictly monotonic between consecutive poles (Tan increasing, Cot
/// decreasing). For a span with no pole inside the result is the endpoint
/// images; with one pole the branch runs out to ±Infinity on each side of the
/// pole (the two pieces merge to all reals when they meet); with two or more
/// poles the function already covers all reals. Returns `None` unless every
/// endpoint and its image is real-numeric.
pub fn tan_cot_interval(head: &str, expr: &Expr) -> Option<Expr> {
  let (offset, increasing) = match head {
    "Tan" => (std::f64::consts::FRAC_PI_2, true),
    "Cot" => (0.0, false),
    _ => return None,
  };
  let period = std::f64::consts::PI;
  let inf = || Expr::Identifier("Infinity".to_string());
  let neg_inf = || Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(Expr::Identifier("Infinity".to_string())),
  };

  let spans = is_interval(expr)?;
  let mut out: Vec<(Expr, Expr)> = Vec::with_capacity(spans.len());
  for (a, b) in spans {
    let af = expr_to_f64(a)?;
    let bf = expr_to_f64(b)?;
    let fa =
      crate::evaluator::evaluate_function_call_ast(head, &[a.clone()]).ok()?;
    let fb =
      crate::evaluator::evaluate_function_call_ast(head, &[b.clone()]).ok()?;
    expr_to_f64(&fa)?;
    expr_to_f64(&fb)?;
    match count_poles_in(offset, period, af, bf) {
      0 => out.push((fa, fb)), // monotonic; normalize sorts the endpoints
      1 => {
        // One pole: the lower branch runs to +Infinity (Tan) / the upper to
        // +Infinity (Cot), the other side coming from -Infinity.
        if increasing {
          out.push((fa, inf()));
          out.push((neg_inf(), fb));
        } else {
          out.push((neg_inf(), fa));
          out.push((fb, inf()));
        }
      }
      _ => out.push((neg_inf(), inf())), // ≥2 poles: all reals
    }
  }
  crate::evaluator::evaluate_expr_to_expr(&make_interval(normalize_intervals(
    out,
  )))
  .ok()
}

/// `Sec[Interval[...]]` / `Csc[Interval[...]]` — the range over each span.
/// These have poles (`Pi/2 + k Pi` for Sec, `k Pi` for Csc) AND extrema, so a
/// pole-free piece is not monotonic: each branch carries a single extremum of
/// value `+1` (a minimum, on "U" branches) or `-1` (a maximum, on "∩" branches).
/// Split at the poles; on each piece collect the finite endpoint images, the
/// branch extremum value when it lies strictly inside, and ±Infinity for a
/// pole-adjacent side. Returns `None` unless every endpoint and finite image is
/// real-numeric.
pub fn sec_csc_interval(head: &str, expr: &Expr) -> Option<Expr> {
  use std::f64::consts::{FRAC_PI_2, PI};
  // (pole offset, extremum offset): poles at pole_off + k*Pi, extrema (values
  // ±1) at ext_off + k*Pi with value (-1)^k.
  let (pole_off, ext_off) = match head {
    "Sec" => (FRAC_PI_2, 0.0),
    "Csc" => (0.0, FRAC_PI_2),
    _ => return None,
  };
  let inf = || Expr::Identifier("Infinity".to_string());
  let neg_inf = || Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(Expr::Identifier("Infinity".to_string())),
  };
  let fold_min =
    |v: &[Expr]| v.iter().cloned().reduce(|a, b| numeric_min(&a, &b));
  let fold_max =
    |v: &[Expr]| v.iter().cloned().reduce(|a, b| numeric_max(&a, &b));

  let spans = is_interval(expr)?;
  let mut out: Vec<(Expr, Expr)> = Vec::with_capacity(spans.len());
  for (a, b) in spans {
    let af = expr_to_f64(a)?;
    let bf = expr_to_f64(b)?;
    let eps = 1e-9;
    // Boundary marker for an original endpoint: `None` if the endpoint is
    // itself a pole (image ±Infinity), else its finite image (required real).
    let mk_bound = |pos: f64, e: &Expr| -> Option<(f64, Option<Expr>)> {
      let k = ((pos - pole_off) / PI).round();
      if (pos - (pole_off + k * PI)).abs() < eps {
        return Some((pos, None)); // endpoint lands on a pole
      }
      let fe = crate::evaluator::evaluate_function_call_ast(head, &[e.clone()])
        .ok()?;
      expr_to_f64(&fe)?;
      Some((pos, Some(fe)))
    };
    // Boundaries: original endpoints plus the poles strictly inside.
    let mut bnds: Vec<(f64, Option<Expr>)> =
      vec![mk_bound(af, a)?, mk_bound(bf, b)?];
    let k_lo = ((af - pole_off) / PI + eps).floor() as i64 + 1;
    let k_hi = ((bf - pole_off) / PI - eps).ceil() as i64 - 1;
    for k in k_lo..=k_hi {
      bnds.push((pole_off + k as f64 * PI, None));
    }
    bnds.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal));
    // Each consecutive pair is one continuous piece on a single branch.
    for w in bnds.windows(2) {
      let (cp, fc) = (&w[0].0, &w[0].1);
      let (dp, fd) = (&w[1].0, &w[1].1);
      let left_pole = fc.is_none();
      let right_pole = fd.is_none();
      // The branch's extremum: nearest lattice point to the piece midpoint.
      let mid = (cp + dp) / 2.0;
      let kk = ((mid - ext_off) / PI).round() as i64;
      let e_pos = ext_off + kk as f64 * PI;
      let v = if kk.rem_euclid(2) == 0 { 1 } else { -1 };
      let e_inside = *cp < e_pos - eps && e_pos < *dp - eps;
      if v == 1 {
        // U branch (minimum +1): finite below, +Infinity toward a pole.
        let mut lows = Vec::new();
        if e_inside {
          lows.push(Expr::Integer(1));
        }
        if let Some(f) = fc {
          lows.push(f.clone());
        }
        if let Some(f) = fd {
          lows.push(f.clone());
        }
        let lo = fold_min(&lows)?;
        let hi = if left_pole || right_pole {
          inf()
        } else {
          numeric_max(fc.as_ref().unwrap(), fd.as_ref().unwrap())
        };
        out.push((lo, hi));
      } else {
        // ∩ branch (maximum -1): finite above, -Infinity toward a pole.
        let mut highs = Vec::new();
        if e_inside {
          highs.push(Expr::Integer(-1));
        }
        if let Some(f) = fc {
          highs.push(f.clone());
        }
        if let Some(f) = fd {
          highs.push(f.clone());
        }
        let hi = fold_max(&highs)?;
        let lo = if left_pole || right_pole {
          neg_inf()
        } else {
          numeric_min(fc.as_ref().unwrap(), fd.as_ref().unwrap())
        };
        out.push((lo, hi));
      }
    }
  }
  crate::evaluator::evaluate_expr_to_expr(&make_interval(normalize_intervals(
    out,
  )))
  .ok()
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
    .map(|(lo, hi)| Expr::List(vec![lo, hi].into()))
    .collect();
  Expr::FunctionCall {
    name: "Interval".to_string(),
    args: args.into(),
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
      args: vec![].into(),
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
        args: args.to_vec().into(),
      });
    }
  }

  let normalized = normalize_intervals(spans);
  Ok(make_interval(normalized))
}

// ─── Set Operations ─────────────────────────────────────────────────────────

/// IntervalUnion[i1, i2, ...] — union of intervals.
pub fn interval_union_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // CenteredInterval inputs: combine into the smallest enclosing
  // axis-aligned box (a CenteredInterval is itself such a box, even
  // when c and r are complex). Wolfram returns a CenteredInterval with
  // a precision-tracked internal representation — we return the same
  // bounding box in surface form.
  if args.iter().all(centered_interval_extract_is_some) {
    return Ok(centered_interval_box_op(args, BoxOp::Union));
  }
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
        args: args.to_vec().into(),
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
      args: vec![].into(),
    });
  }

  // CenteredInterval inputs: take the per-axis intersection of the
  // bounding boxes. If any axis is empty, return Interval[].
  if args.iter().all(centered_interval_extract_is_some) {
    return Ok(centered_interval_box_op(args, BoxOp::Intersection));
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
        args: args.to_vec().into(),
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
          args: args.to_vec().into(),
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
        args: args.to_vec().into(),
      });
    }
  };

  // A list of test points threads: IntervalMemberQ[iv, {p1, p2, …}] returns
  // {IntervalMemberQ[iv, p1], …}.
  if let Expr::List(points) = &args[1] {
    let results: Result<Vec<Expr>, InterpreterError> = points
      .iter()
      .map(|p| interval_member_q_ast(&[args[0].clone(), p.clone()]))
      .collect();
    return Ok(Expr::List(results?.into()));
  }

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

  // Interval ^ Interval. For a non-negative base, x^y is monotonic in each
  // variable separately (∂/∂x = y x^(y-1) and ∂/∂y = x^y ln x each keep a
  // fixed sign over the box), so the image is bounded by the four corner
  // values. Negative bases are left unevaluated, matching wolframscript.
  if let Some(exp_spans) = is_interval(exp) {
    let mut result_spans = Vec::new();
    for (blo, bhi) in &spans {
      if expr_to_f64(blo)? < 0.0 {
        return None;
      }
      for (elo, ehi) in &exp_spans {
        let corners = [
          eval_binop(blo, elo, "Power").ok()?,
          eval_binop(blo, ehi, "Power").ok()?,
          eval_binop(bhi, elo, "Power").ok()?,
          eval_binop(bhi, ehi, "Power").ok()?,
        ];
        // Every corner must be a finite real (e.g. 0^0 / 0^-1 bail out).
        for c in &corners {
          expr_to_f64(c)?;
        }
        let mut lo = corners[0].clone();
        let mut hi = corners[0].clone();
        for c in &corners[1..] {
          lo = numeric_min(&lo, c);
          hi = numeric_max(&hi, c);
        }
        result_spans.push((lo, hi));
      }
    }
    return Some(Ok(make_interval(normalize_intervals(result_spans))));
  }

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
        args: args.to_vec().into(),
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
        args: args.to_vec().into(),
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
        args: args.to_vec().into(),
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
        args: args.to_vec().into(),
      }))
    }
    _ => None,
  }
}

// ─── CenteredInterval helpers ───────────────────────────────────────────

/// Box operations on CenteredInterval inputs (treated as axis-aligned
/// rectangles in C — real centre and radius produce a degenerate
/// rectangle on the real axis).
#[derive(Copy, Clone)]
enum BoxOp {
  Union,
  Intersection,
}

fn centered_interval_extract_is_some(e: &Expr) -> bool {
  centered_interval_box(e).is_some()
}

/// Split `e` into ((re_c, im_c), (re_r, im_r)) where `e` is
/// `CenteredInterval[c, r]` with arithmetic-evaluable `c` and `r`.
fn centered_interval_box(e: &Expr) -> Option<((Expr, Expr), (Expr, Expr))> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "CenteredInterval" || args.len() != 2 {
    return None;
  }
  let (cr, ci) = split_real_imag(&args[0])?;
  let (rr, ri) = split_real_imag(&args[1])?;
  Some(((cr, ci), (rr, ri)))
}

/// Split a numeric/complex expression into its real and imaginary
/// parts as separate Expr values. Symbolic terms inside Plus/Times are
/// preserved; the integers / rationals split cleanly so that
/// `Re[2 + 3 I]` etc. survive output without further simplification.
fn split_real_imag(e: &Expr) -> Option<(Expr, Expr)> {
  // Real-only fast path.
  let is_real_atom = matches!(e, Expr::Integer(_) | Expr::Real(_))
    || matches!(
      e,
      Expr::FunctionCall { name, .. } if name == "Rational"
    );
  if is_real_atom {
    return Some((e.clone(), Expr::Integer(0)));
  }
  let re_expr =
    crate::evaluator::evaluate_function_call_ast("Re", std::slice::from_ref(e))
      .ok()?;
  let im_expr =
    crate::evaluator::evaluate_function_call_ast("Im", std::slice::from_ref(e))
      .ok()?;
  Some((re_expr, im_expr))
}

fn centered_interval_box_op(args: &[Expr], op: BoxOp) -> Expr {
  // Collect (Re_lo, Re_hi, Im_lo, Im_hi) per box.
  let mut re_lo_lst: Vec<Expr> = Vec::new();
  let mut re_hi_lst: Vec<Expr> = Vec::new();
  let mut im_lo_lst: Vec<Expr> = Vec::new();
  let mut im_hi_lst: Vec<Expr> = Vec::new();
  for arg in args {
    let Some(((cr, ci), (rr, ri))) = centered_interval_box(arg) else {
      return Expr::FunctionCall {
        name: match op {
          BoxOp::Union => "IntervalUnion".to_string(),
          BoxOp::Intersection => "IntervalIntersection".to_string(),
        },
        args: args.to_vec().into(),
      };
    };
    re_lo_lst.push(eval_sub(&cr, &rr));
    re_hi_lst.push(eval_add(&cr, &rr));
    im_lo_lst.push(eval_sub(&ci, &ri));
    im_hi_lst.push(eval_add(&ci, &ri));
  }
  let (re_lo, re_hi) = match op {
    BoxOp::Union => (extremum(&re_lo_lst, false), extremum(&re_hi_lst, true)),
    BoxOp::Intersection => {
      (extremum(&re_lo_lst, true), extremum(&re_hi_lst, false))
    }
  };
  let (im_lo, im_hi) = match op {
    BoxOp::Union => (extremum(&im_lo_lst, false), extremum(&im_hi_lst, true)),
    BoxOp::Intersection => {
      (extremum(&im_lo_lst, true), extremum(&im_hi_lst, false))
    }
  };
  if matches!(op, BoxOp::Intersection) {
    // Empty on any axis ⇒ empty interval.
    if compare_numeric(&re_lo, &re_hi) == Some(Ordering::Greater)
      || compare_numeric(&im_lo, &im_hi) == Some(Ordering::Greater)
    {
      return Expr::FunctionCall {
        name: "Interval".to_string(),
        args: vec![].into(),
      };
    }
  }
  let two = Expr::Integer(2);
  let cre = eval_divide(&eval_add(&re_lo, &re_hi), &two);
  let cim = eval_divide(&eval_add(&im_lo, &im_hi), &two);
  let rre = eval_divide(&eval_sub(&re_hi, &re_lo), &two);
  let rim = eval_divide(&eval_sub(&im_hi, &im_lo), &two);
  let centre = combine_complex(&cre, &cim);
  let radius = combine_complex(&rre, &rim);
  Expr::FunctionCall {
    name: "CenteredInterval".to_string(),
    args: vec![centre, radius].into(),
  }
}

fn extremum(xs: &[Expr], maximum: bool) -> Expr {
  let mut iter = xs.iter().cloned();
  let mut best = iter.next().expect("non-empty list");
  for x in iter {
    let ord = compare_numeric(&x, &best);
    let pick_x = matches!(
      (ord, maximum),
      (Some(Ordering::Greater), true) | (Some(Ordering::Less), false)
    );
    if pick_x {
      best = x;
    }
  }
  best
}

fn combine_complex(re: &Expr, im: &Expr) -> Expr {
  let is_zero = matches!(im, Expr::Integer(0))
    || matches!(im, Expr::Real(v) if v.abs() < 1e-300);
  if is_zero {
    return re.clone();
  }
  let im_term = eval_mul(im, &Expr::Identifier("I".to_string()));
  eval_add(re, &im_term)
}

fn eval_add(a: &Expr, b: &Expr) -> Expr {
  crate::evaluator::evaluate_function_call_ast("Plus", &[a.clone(), b.clone()])
    .unwrap_or_else(|_| Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![a.clone(), b.clone()].into(),
    })
}

fn eval_sub(a: &Expr, b: &Expr) -> Expr {
  let neg_b = crate::evaluator::evaluate_function_call_ast(
    "Times",
    &[Expr::Integer(-1), b.clone()],
  )
  .unwrap_or_else(|_| Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(-1), b.clone()].into(),
  });
  eval_add(a, &neg_b)
}

fn eval_mul(a: &Expr, b: &Expr) -> Expr {
  crate::evaluator::evaluate_function_call_ast("Times", &[a.clone(), b.clone()])
    .unwrap_or_else(|_| Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![a.clone(), b.clone()].into(),
    })
}

fn eval_divide(num: &Expr, den: &Expr) -> Expr {
  let inv = crate::evaluator::evaluate_function_call_ast(
    "Power",
    &[den.clone(), Expr::Integer(-1)],
  )
  .unwrap_or_else(|_| Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![den.clone(), Expr::Integer(-1)].into(),
  });
  eval_mul(num, &inv)
}
