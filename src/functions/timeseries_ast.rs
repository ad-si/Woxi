//! Temporal data: `TemporalData`, `TimeSeries`, and `TimeSeriesResample`.
//!
//! A time series is normalized to the canonical inert form
//! `TimeSeries[{{date, value}, ...}]`, where each `date` is a date list
//! `{y, m, d, h, min, sec}`. `TemporalData[TimeSeries, {values, {spec}, ...}]`
//! (the internal full form produced by, e.g., `CompressedData`-backed data) is
//! rebuilt into this canonical form by pairing the value path with the dates
//! generated from the embedded date specification.

use super::*;
use crate::functions::datetime_ast::{
  date_to_absolute_seconds, day_of_week, days_in_month,
  extract_date_components, weekday_index,
};

fn as_f64(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(i) => Some(*i as f64),
    Expr::Real(r) => Some(*r),
    Expr::BigInteger(b) => {
      use num_traits::ToPrimitive;
      b.to_f64()
    }
    // A rescaled series carries rational stamps, so they count as times too.
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      Some(as_f64(&args[0])? / as_f64(&args[1])?)
    }
    _ => None,
  }
}

/// Extract `{y, m, d, h, min, sec}` numeric components from a date list,
/// padding missing trailing fields with zero (defaulting month/day to 1).
fn date_components(e: &Expr) -> Option<[f64; 6]> {
  let items = match e {
    Expr::List(items) => items.iter().collect::<Vec<_>>(),
    _ => return None,
  };
  let mut out = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
  for (i, slot) in out.iter_mut().enumerate() {
    if let Some(v) = items.get(i) {
      *slot = as_f64(v)?;
    }
  }
  Some(out)
}

fn make_date_list(y: i64, m: i64, d: i64, h: i64, min: i64, sec: f64) -> Expr {
  Expr::List(
    vec![
      Expr::Integer(y as i128),
      Expr::Integer(m as i128),
      Expr::Integer(d as i128),
      Expr::Integer(h as i128),
      Expr::Integer(min as i128),
      Expr::Real(sec),
    ]
    .into(),
  )
}

/// Step size in seconds for fixed-length calendar units. Returns `None` for the
/// variable-length units (handled separately by month/year arithmetic).
fn unit_seconds(unit: &str) -> Option<f64> {
  match unit {
    "Second" => Some(1.0),
    "Minute" => Some(60.0),
    "Hour" => Some(3600.0),
    "Day" => Some(86400.0),
    "Week" => Some(7.0 * 86400.0),
    _ => None,
  }
}

/// Advance `{y, m, d, h, min, sec}` by `n` whole months (used for the
/// variable-length "Month"/"Year"/"Quarter" step units).
fn add_months(c: [f64; 6], n: i64) -> [f64; 6] {
  let mut y = c[0] as i64;
  let mut m = c[1] as i64 - 1 + n;
  y += m.div_euclid(12);
  m = m.rem_euclid(12) + 1;
  let dim = days_in_month(y, m);
  let d = (c[2] as i64).min(dim);
  [y as f64, m as f64, d as f64, c[3], c[4], c[5]]
}

/// Generate `count` dates starting at `start`, advancing by `step` ({amount,
/// unit}). Each returned date is a `{y,m,d,h,min,sec}` list.
fn generate_dates(
  start: [f64; 6],
  step_amount: f64,
  step_unit: &str,
  count: usize,
) -> Vec<Expr> {
  let mut dates = Vec::with_capacity(count);
  if let Some(unit_secs) = unit_seconds(step_unit) {
    let base = date_to_absolute_seconds(
      start[0] as i64,
      start[1] as i64,
      start[2] as i64,
      start[3] as i64,
      start[4] as i64,
      start[5],
    );
    for k in 0..count {
      let secs = base + (k as f64) * step_amount * unit_secs;
      let (y, m, d, h, mi, s) =
        crate::functions::datetime_ast::absolute_seconds_to_date(secs);
      dates.push(make_date_list(y, m, d, h, mi, s));
    }
  } else {
    // Month / Quarter / Year: step by whole months.
    let months_per = match step_unit {
      "Month" => 1,
      "Quarter" => 3,
      "Year" => 12,
      _ => 1,
    };
    for k in 0..count {
      let c = add_months(start, (k as i64) * (step_amount as i64) * months_per);
      dates.push(make_date_list(
        c[0] as i64,
        c[1] as i64,
        c[2] as i64,
        c[3] as i64,
        c[4] as i64,
        c[5],
      ));
    }
  }
  dates
}

/// Pull `(start, step_amount, step_unit)` out of a date specification such as
/// `DateSpecification[{2013,4,1,..}, {2013,9,1,..}, {1, "Day"}]` (the head may
/// be context-qualified, e.g. `TemporalData`DateSpecification`).
fn parse_date_spec(spec: &Expr) -> Option<([f64; 6], f64, String)> {
  let args = match spec {
    Expr::FunctionCall { name, args }
      if name.ends_with("DateSpecification") =>
    {
      args.iter().collect::<Vec<_>>()
    }
    _ => return None,
  };
  let start = date_components(args.first()?)?;
  // Step is the last argument, a {amount, unit} pair.
  let (amount, unit) = match args.last()? {
    Expr::List(items) => {
      let v = items.iter().collect::<Vec<_>>();
      let amount = as_f64(v.first()?)?;
      let unit = match v.get(1)? {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => "Day".to_string(),
      };
      (amount, unit)
    }
    _ => (1.0, "Day".to_string()),
  };
  Some((start, amount, unit))
}

/// Parse a step descriptor for a date range: `"Day"`, `{amount, "Day"}`, or
/// `Quantity[amount, "Day"]`. Returns `(amount, unit)`.
fn parse_step_descriptor(e: &Expr) -> Option<(f64, String)> {
  let is_unit = |s: &str| {
    unit_seconds(s).is_some() || matches!(s, "Month" | "Quarter" | "Year")
  };
  match e {
    Expr::String(s) | Expr::Identifier(s) if is_unit(s) => {
      Some((1.0, s.clone()))
    }
    Expr::List(items) => {
      let v: Vec<&Expr> = items.iter().collect();
      let amount = as_f64(v.first()?)?;
      let unit = match v.get(1)? {
        Expr::String(s) | Expr::Identifier(s) if is_unit(s) => s.clone(),
        _ => return None,
      };
      Some((amount, unit))
    }
    Expr::FunctionCall { name, args } if name == "Quantity" => {
      let v: Vec<&Expr> = args.iter().collect();
      let amount = as_f64(v.first()?).unwrap_or(1.0);
      match v.get(1)? {
        Expr::String(s) | Expr::Identifier(s) if is_unit(s) => {
          Some((amount, s.clone()))
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// Detect the `{start, step}` / `{start, end, step}` date-range form of the
/// TimeSeries time argument and return `(start, amount, unit)`. Only fires when
/// the last element is a recognizable step unit, so it never captures a list of
/// explicit numeric or date-list stamps.
fn parse_range_spec(times: &[&Expr]) -> Option<([f64; 6], f64, String)> {
  if times.len() != 2 && times.len() != 3 {
    return None;
  }
  let (amount, unit) = parse_step_descriptor(times.last()?)?;
  let start = date_components(times[0])?;
  Some((start, amount, unit))
}

/// Pair a value path with the regular numeric times `t0, t0 + dt, …` given by
/// the `{{t0, t1, dt}}` specification of an `EventSeries` field list, whose
/// values come as `{{v1, v2, …}}` (a list holding the one path).
fn build_pairs_from_event_spec(fields: &[Expr]) -> Option<Vec<Expr>> {
  let Expr::List(paths) = fields.first()? else {
    return None;
  };
  // The internal form holds a list of paths; only a single path is a series.
  let [Expr::List(values)] = paths.iter().collect::<Vec<_>>()[..] else {
    return None;
  };
  let Expr::List(specs) = fields.get(1)? else {
    return None;
  };
  let Some(Expr::List(spec)) = specs.iter().next() else {
    return None;
  };
  let [t0, _, dt] = spec.iter().collect::<Vec<_>>()[..] else {
    return None;
  };
  let time_at = |i: usize| -> Option<Expr> {
    match (t0, dt) {
      (Expr::Integer(a), Expr::Integer(d)) => {
        Some(Expr::Integer(a + d * i as i128))
      }
      _ => Some(Expr::Real(as_f64(t0)? + as_f64(dt)? * i as f64)),
    }
  };
  values
    .iter()
    .enumerate()
    .map(|(i, v)| Some(Expr::List(vec![time_at(i)?, v.clone()].into())))
    .collect()
}

/// Build canonical `{{date, value}, ...}` pairs from a value path and the date
/// specification embedded in `TemporalData`'s field list.
fn build_pairs_from_temporal(fields: &[Expr]) -> Option<Vec<Expr>> {
  let values: Vec<Expr> = match fields.first()? {
    // Wolfram's internal form wraps the single value path in a list.
    Expr::List(items)
      if items.len() == 1
        && matches!(items.first(), Some(Expr::List(p)) if p.len() > 1) =>
    {
      match items.first() {
        Some(Expr::List(p)) => p.iter().cloned().collect(),
        _ => return None,
      }
    }
    Expr::List(items) => items.iter().cloned().collect(),
    _ => return None,
  };
  // The date spec sits in a singleton list at field index 1.
  let spec = match fields.get(1)? {
    Expr::List(items) => items.iter().next()?.clone(),
    other => other.clone(),
  };
  let (start, amount, unit) = parse_date_spec(&spec)?;
  let dates = generate_dates(start, amount, &unit, values.len());
  Some(
    dates
      .into_iter()
      .zip(values)
      .map(|(d, v)| Expr::List(vec![d, v].into()))
      .collect(),
  )
}

/// `MovingAverage[ts, n]` — the mean of each window of `n` values, stamped
/// with the last time of the window.
pub fn time_series_moving_average_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("MovingAverage", args));
  let Some(pairs) = series_pairs_of(&args[0]) else {
    return echo();
  };
  let Some(n) = crate::functions::math_ast::expr_to_i128(&args[1]) else {
    return echo();
  };
  if n < 1 || (n as usize) > pairs.len() {
    return echo();
  }
  let n = n as usize;
  let mut out = Vec::with_capacity(pairs.len() + 1 - n);
  for window in pairs.windows(n) {
    let values: Vec<Expr> = window.iter().map(|(_, v)| v.clone()).collect();
    let mean = crate::evaluator::evaluate_expr_to_expr(&call(
      "N",
      vec![call1("Mean", Expr::List(values.into()))],
    ))?;
    out.push(Expr::List(vec![window[n - 1].0.clone(), mean].into()));
  }
  Ok(rebuild_series(&args[0], out))
}

/// Arithmetic on a series works on its values and keeps the time stamps:
/// `ts + 1` shifts every value, `2 ts` scales them, and two series combine
/// point by point. Returns `None` when no argument is a series.
pub fn try_series_arithmetic(head: &str, args: &[Expr]) -> Option<Expr> {
  let series: Vec<Option<Vec<(Expr, Expr)>>> =
    args.iter().map(series_pairs_of).collect();
  let first = series.iter().position(std::option::Option::is_some)?;
  let base = series[first].as_ref()?;
  let mut out = Vec::with_capacity(base.len());
  for (i, (time, _)) in base.iter().enumerate() {
    let mut point_args = Vec::with_capacity(args.len());
    for (arg, pairs) in args.iter().zip(series.iter()) {
      match pairs {
        Some(p) => point_args.push(p.get(i)?.1.clone()),
        None => point_args.push(arg.clone()),
      }
    }
    let combined =
      crate::evaluator::evaluate_expr_to_expr(&call(head, point_args)).ok()?;
    out.push(Expr::List(vec![time.clone(), combined].into()));
  }
  Some(rebuild_series(&args[first], out))
}

/// `TimeSeriesShift[ts, dt]` — every time stamp moves by `dt`.
pub fn time_series_shift_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("TimeSeriesShift", args));
  if args.len() != 2 {
    return echo();
  }
  let Some(pairs) = series_pairs_of(&args[0]) else {
    return echo();
  };
  let mut out = Vec::with_capacity(pairs.len());
  for (time, value) in pairs {
    let shifted = arith("Plus", &time, &args[1])?;
    out.push(Expr::List(vec![shifted, value].into()));
  }
  Ok(rebuild_series(&args[0], out))
}

/// `MovingMap[f, series, n]` — `f` applied to the values in each time window
/// `[t - n, t]`, stamped at `t`. `n` is a width in time, not a count, and a
/// window that would reach back past the start of the series is not one, so
/// the result is shorter at the front. `None` when the arguments are not a
/// series and a positive width, leaving the call to the list handling.
pub fn moving_map_series_ast(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() != 3 {
    return None;
  }
  let pairs = series_pairs_of(&args[1])?;
  // A one-element window spec means the same as the bare width.
  let width_expr = match &args[2] {
    Expr::List(spec) if spec.len() == 1 => &spec[0],
    other => other,
  };
  let width = to_time(width_expr)?;
  // Negated on purpose: a NaN width is not a valid width either.
  #[allow(clippy::neg_cmp_op_on_partial_ord)]
  if !(width > 0.0) {
    return None;
  }
  let times: Vec<f64> = pairs
    .iter()
    .map(|(t, _)| to_time(t))
    .collect::<Option<Vec<_>>>()?;
  let first = *times.first()?;
  // Times are stored to the precision of a float, so compare with the slack
  // an exact stamp would otherwise fail by.
  let slack = 1e-9 * (1.0 + width.abs() + first.abs());
  let mut out = Vec::new();
  for (i, (time, _)) in pairs.iter().enumerate() {
    let start = times[i] - width;
    if start < first - slack {
      continue;
    }
    let window: Vec<Expr> = pairs
      .iter()
      .zip(times.iter())
      .filter(|(_, t)| **t >= start - slack && **t <= times[i] + slack)
      .map(|((_, v), _)| v.clone())
      .collect();
    let applied = match crate::evaluator::apply_function_to_arg(
      &args[0],
      &Expr::List(window.into()),
    ) {
      Ok(v) => v,
      Err(e) => return Some(Err(e)),
    };
    out.push(Expr::List(vec![time.clone(), applied].into()));
  }
  Some(Ok(rebuild_series(&args[1], out)))
}

/// `TimeSeriesRescale[ts, {tmin, tmax}]` — the same values, with the time
/// stamps carried linearly onto the given span.
pub fn time_series_rescale_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("TimeSeriesRescale", args));
  if args.len() != 2 {
    let n = args.len();
    let noun = if n == 1 { "argument" } else { "arguments" };
    crate::emit_message(&format!(
      "TimeSeriesRescale::argr: TimeSeriesRescale called with {n} {noun}; \
       2 arguments are expected."
    ));
    return echo();
  }
  let Some(pairs) = series_pairs_of(&args[0]) else {
    return echo();
  };
  // A span written as anything but a pair leaves the series as it was.
  let Expr::List(span) = &args[1] else {
    return Ok(args[0].clone());
  };
  if span.len() != 2 {
    return Ok(args[0].clone());
  }
  let (Some(low), Some(high)) = (to_time(&span[0]), to_time(&span[1])) else {
    return echo();
  };
  // Negated on purpose: NaN endpoints are not strictly increasing either.
  #[allow(clippy::neg_cmp_op_on_partial_ord)]
  if !(high > low) {
    crate::emit_message(&format!(
      "TimeSeriesRescale::trng: The argument {} is not a valid pair of \
       strictly increasing time points.",
      expr_to_string(&args[1])
    ));
    return echo();
  }
  // The stamps keep their spacing, so the span they already cover is what
  // maps onto the new one.
  let times: Vec<f64> = pairs.iter().filter_map(|(t, _)| to_time(t)).collect();
  if times.len() != pairs.len() {
    return echo();
  }
  let (first, last) = (
    times.iter().copied().fold(f64::INFINITY, f64::min),
    times.iter().copied().fold(f64::NEG_INFINITY, f64::max),
  );
  // Negated on purpose: NaN times give no usable span either.
  #[allow(clippy::neg_cmp_op_on_partial_ord)]
  if !(last > first) {
    return echo();
  }
  let start = &pairs
    .iter()
    .min_by(|a, b| {
      to_time(&a.0)
        .unwrap_or(f64::NAN)
        .total_cmp(&to_time(&b.0).unwrap_or(f64::NAN))
    })
    .expect("a series with a span has points")
    .0;
  let end = &pairs
    .iter()
    .max_by(|a, b| {
      to_time(&a.0)
        .unwrap_or(f64::NAN)
        .total_cmp(&to_time(&b.0).unwrap_or(f64::NAN))
    })
    .expect("a series with a span has points")
    .0;
  let width = arith("Subtract", end, start)?;
  let reach = arith("Subtract", &span[1], &span[0])?;
  let mut out = Vec::with_capacity(pairs.len());
  for (time, value) in &pairs {
    // tmin + (t - first) / (last - first) * (tmax - tmin), kept exact.
    let offset = arith("Subtract", time, start)?;
    let fraction = arith("Divide", &offset, &width)?;
    let scaled = arith("Times", &fraction, &reach)?;
    let moved = arith("Plus", &span[0], &scaled)?;
    out.push(Expr::List(vec![moved, value.clone()].into()));
  }
  Ok(rebuild_series(&args[0], out))
}

/// `TimeSeriesMap[f, ts]` — apply `f` to every value, keeping the time stamps.
pub fn time_series_map_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("TimeSeriesMap", args));
  if args.len() != 2 {
    return echo();
  }
  let Some(pairs) = series_pairs_of(&args[1]) else {
    return echo();
  };
  let mut out = Vec::with_capacity(pairs.len());
  for (time, value) in pairs {
    let applied = crate::evaluator::apply_function_to_arg(&args[0], &value)?;
    out.push(Expr::List(vec![time, applied].into()));
  }
  Ok(rebuild_series(&args[1], out))
}

/// `TimeSeriesThread[f, {ts1, ts2, …}]` — apply `f` to the list of values the
/// series share at each time stamp.
pub fn time_series_thread_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("TimeSeriesThread", args));
  if args.len() != 2 {
    return echo();
  }
  let Expr::List(series) = &args[1] else {
    return echo();
  };
  let mut all: Vec<Vec<(Expr, Expr)>> = Vec::with_capacity(series.len());
  for s in series {
    match series_pairs_of(s) {
      Some(p) => all.push(p),
      None => return echo(),
    }
  }
  let Some(first) = all.first() else {
    return echo();
  };
  // Only the time stamps every series has take part.
  let mut out = Vec::new();
  for (time, _) in first {
    let key = to_time(time);
    let mut values = Vec::with_capacity(all.len());
    for series in &all {
      let found = series.iter().find(|(t, _)| match (to_time(t), key) {
        (Some(a), Some(b)) => a == b,
        _ => expr_to_string(t) == expr_to_string(time),
      });
      match found {
        Some((_, v)) => values.push(v.clone()),
        None => break,
      }
    }
    if values.len() != all.len() {
      continue;
    }
    let applied = crate::evaluator::apply_function_to_arg(
      &args[0],
      &Expr::List(values.into()),
    )?;
    out.push(Expr::List(vec![time.clone(), applied].into()));
  }
  Ok(rebuild_series(&series[0], out))
}

/// `RegularlySampledQ[ts]` — True when the time stamps are evenly spaced.
/// Fewer than three stamps are trivially even.
pub fn regularly_sampled_q_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("RegularlySampledQ", args));
  if args.len() != 1 {
    return echo();
  }
  let Some(pairs) = series_pairs_of(&args[0]) else {
    return echo();
  };
  let times: Option<Vec<f64>> = pairs.iter().map(|(t, _)| to_time(t)).collect();
  let Some(times) = times else { return echo() };
  if times.len() < 3 {
    return Ok(bool_expr(true));
  }
  let step = times[1] - times[0];
  let even = times
    .windows(2)
    .all(|w| (w[1] - w[0] - step).abs() <= 1e-9 * step.abs().max(1.0));
  Ok(bool_expr(even))
}

/// `TimeSeriesInsert[ts, {t, v}]` — add a point, keeping the path sorted.
pub fn time_series_insert_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("TimeSeriesInsert", args));
  if args.len() != 2 {
    return echo();
  }
  let Some(pairs) = series_pairs_of(&args[0]) else {
    return echo();
  };
  let Expr::List(point) = &args[1] else {
    return echo();
  };
  if point.len() != 2 {
    return echo();
  }
  let mut all: Vec<(Expr, Expr)> = pairs;
  all.push((point[0].clone(), point[1].clone()));
  all.sort_by(|a, b| match (to_time(&a.0), to_time(&b.0)) {
    (Some(x), Some(y)) => {
      x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
    }
    _ => std::cmp::Ordering::Equal,
  });
  Ok(rebuild_series(
    &args[0],
    all
      .into_iter()
      .map(|(t, v)| Expr::List(vec![t, v].into()))
      .collect(),
  ))
}

/// The `{time, value}` pairs of a `TimeSeries` or an `EventSeries`.
pub fn series_pairs_of(expr: &Expr) -> Option<Vec<(Expr, Expr)>> {
  if let Some(p) = time_series_pairs(expr) {
    return Some(p);
  }
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "EventSeries" || args.len() != 1 {
    return None;
  }
  let Expr::List(items) = &args[0] else {
    return None;
  };
  let mut out = Vec::with_capacity(items.len());
  for item in items {
    match item {
      Expr::List(kv) if kv.len() == 2 => {
        out.push((kv[0].clone(), kv[1].clone()));
      }
      _ => return None,
    }
  }
  Some(out)
}

/// Wrap a path in the same head the source series had, so an EventSeries stays
/// one.
fn rebuild_series(source: &Expr, path: Vec<Expr>) -> Expr {
  let head = match source {
    Expr::FunctionCall { name, .. } if name == "EventSeries" => "EventSeries",
    _ => "TimeSeries",
  };
  call1(head, Expr::List(path.into()))
}

fn time_series(mut pairs: Vec<Expr>) -> Expr {
  // A series runs in time order however its stamps were given.
  let key = |e: &Expr| match e {
    Expr::List(kv) if !kv.is_empty() => to_time(&kv[0]),
    _ => None,
  };
  if pairs.iter().all(|p| key(p).is_some()) {
    pairs.sort_by(|a, b| {
      key(a)
        .unwrap_or(f64::NAN)
        .total_cmp(&key(b).unwrap_or(f64::NAN))
    });
  }
  call1("TimeSeries", Expr::List(pairs.into()))
}

/// One realization of `DiscreteMarkovProcess[i0 | p0, m]` over the integer
/// times `{tmin, tmax}`, as `{{t, state}, …}` pairs. The state at `tmin` is
/// `i0` (or drawn from the initial distribution `p0`); each later state is
/// drawn from the row of `m` belonging to the previous one. Rows are
/// normalized by their sum, so non-exact floating-point input still works.
fn discrete_markov_path(dargs: &[Expr], spec: &[Expr]) -> Option<Vec<Expr>> {
  let (p0, matrix, n) = crate::functions::math_ast::dmp_parts(dargs)?;
  let [Expr::Integer(tmin), Expr::Integer(tmax)] = spec else {
    return None;
  };
  if tmax < tmin {
    return None;
  }
  let to_weights = |items: &[Expr]| -> Option<Vec<f64>> {
    let w: Vec<f64> = items
      .iter()
      .map(|e| try_eval_to_f64(e).filter(|x| *x >= 0.0))
      .collect::<Option<_>>()?;
    (w.iter().sum::<f64>() > 0.0).then_some(w)
  };
  let Expr::List(rows) = &matrix else {
    return None;
  };
  let rows: Vec<Vec<f64>> = rows
    .iter()
    .map(|r| match r {
      Expr::List(c) => to_weights(c),
      _ => None,
    })
    .collect::<Option<_>>()?;
  let draw = |w: &[f64]| -> usize {
    use rand::RngExt;
    let total: f64 = w.iter().sum();
    let mut r = crate::with_rng(|rng| rng.random_range(0.0..total));
    for (k, wk) in w.iter().enumerate() {
      if r < *wk {
        return k;
      }
      r -= wk;
    }
    w.len() - 1
  };
  let mut state = draw(&to_weights(&p0)?);
  debug_assert!(state < n);
  let mut pairs = Vec::with_capacity((tmax - tmin + 1) as usize);
  for t in *tmin..=*tmax {
    if t > *tmin {
      state = draw(&rows[state]);
    }
    pairs.push(Expr::List(
      vec![Expr::Integer(t), Expr::Integer(state as i128 + 1)].into(),
    ));
  }
  Some(pairs)
}

/// `RandomFunction[proc, {t0, t1, dt}]` — one realization of a random
/// process, sampled at `t0, t0 + dt, …` up to `t1`. Returned as a
/// `TimeSeries[{{t0, v0}, …}]` object (Woxi's canonical internal form for
/// `TemporalData`, the same one `TemporalData[values, times]` itself
/// normalizes to — see the module doc comment), so every existing
/// `["Values"]`/`["Times"]`/… property accessor already works on it, plus
/// the `["States"]` accessor added above for wolframscript's own
/// process-realization report.
///
/// A continuous-time Markov process's value one step later depends only on
/// its current value, not on how it got there, so simulating a full path is
/// just repeatedly drawing from the process's own one-step transition
/// distribution — no discretization error, unlike an Euler–Maruyama
/// approximation. That transition distribution is implemented here for the
/// two process types Woxi already recognizes as one-dimensional continuous
/// processes ([`crate::functions::math_ast::distributions::process_slice_distribution`]
/// gives their unconditional time-`t` slice the same way): `WienerProcess[m,
/// s]` and `OrnsteinUhlenbeckProcess[m, s, th]`/`OrnsteinUhlenbeckProcess[m,
/// s, th, x0]` — plus the counting process `PoissonProcess[mu]`, which starts
/// at 0 and grows by a `PoissonDistribution[mu dt]` count each step. Any
/// other process, or a malformed spec, is left unevaluated.
pub fn random_function_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unchanged = || Ok(unevaluated("RandomFunction", args));
  // `RandomFunction[proc, tspec, n]` draws `n` independent realizations.
  let count = match args {
    [_, _] => 1,
    [_, _, Expr::Integer(n)] if *n >= 1 => *n as usize,
    _ => return unchanged(),
  };
  let mut paths = Vec::with_capacity(count);
  for _ in 0..count {
    match one_realization(&args[0], &args[1]) {
      Some(path) => paths.push(path),
      None => return unchanged(),
    }
  }
  if count == 1 {
    return Ok(time_series(paths.pop().unwrap()));
  }
  Ok(multi_path_temporal_data(&paths))
}

/// Several realizations sharing one time axis, as the multi-path
/// `TemporalData[{values1, …}, {{t1, …}}]` that [`temporal_paths`] reads.
fn multi_path_temporal_data(paths: &[Vec<Expr>]) -> Expr {
  let split = |pair: &Expr, i: usize| match pair {
    Expr::List(kv) => kv[i].clone(),
    _ => unreachable!(),
  };
  let times: Vec<Expr> = paths[0].iter().map(|p| split(p, 0)).collect();
  let values: Vec<Expr> = paths
    .iter()
    .map(|path| Expr::List(path.iter().map(|p| split(p, 1)).collect()))
    .collect();
  call(
    "TemporalData",
    vec![
      Expr::List(values.into()),
      Expr::List(vec![Expr::List(times.into())].into()),
    ],
  )
}

/// One realization of `proc` over `tspec` as `{{t, value}, …}` pairs, or
/// `None` for a process or time specification Woxi does not simulate.
fn one_realization(proc: &Expr, tspec: &Expr) -> Option<Vec<Expr>> {
  let Expr::FunctionCall {
    name: proc_name,
    args: dargs,
  } = proc
  else {
    return None;
  };
  let Expr::List(spec) = tspec else {
    return None;
  };
  if proc_name == "DiscreteMarkovProcess" {
    return discrete_markov_path(dargs, spec);
  }
  if let Some(path) = discrete_linear_process_path(proc, spec) {
    return path;
  }
  let [t0, t1, dt] = spec.as_slice() else {
    return None;
  };
  let (Some(t0), Some(t1), Some(dt)) = (
    try_eval_to_f64(t0),
    try_eval_to_f64(t1),
    try_eval_to_f64(dt),
  ) else {
    return None;
  };
  if dt.is_nan() || dt <= 0.0 || t1 < t0 {
    return None;
  }

  use rand_distr::{Distribution, Normal};
  let n_steps = ((t1 - t0) / dt).round() as i64;
  if n_steps < 0 {
    return None;
  }

  if let ("PoissonProcess", [mu]) = (proc_name.as_str(), dargs.as_slice()) {
    let Some(mu) = try_eval_to_f64(mu).filter(|m| *m > 0.0) else {
      if !crate::functions::predicate_ast::is_numeric_q(mu) {
        crate::emit_message(&format!(
          "PoissonProcess::realprm: Parameter {} at position 1 in {} is expected to be real.",
          crate::syntax::expr_to_output(mu),
          crate::syntax::expr_to_output(proc)
        ));
      }
      return None;
    };
    let mut count: i128 = 0;
    let mut t = t0;
    let mut pairs = Vec::with_capacity(n_steps as usize + 1);
    pairs.push(Expr::List(vec![Expr::Real(t0), Expr::Integer(0)].into()));
    for _ in 0..n_steps {
      let this_step = dt.min(t1 - t);
      let Ok(increments) = rand_distr::Poisson::new(mu * this_step) else {
        return None;
      };
      count += crate::with_rng(|rng| increments.sample(rng)) as i128;
      t += this_step;
      pairs.push(Expr::List(vec![Expr::Real(t), Expr::Integer(count)].into()));
    }
    return Some(pairs);
  }

  let sample_normal = |mean: f64, sd: f64| -> Option<f64> {
    let n = Normal::new(mean, sd.max(0.0)).ok()?;
    Some(crate::with_rng(|rng| n.sample(rng)))
  };

  // The distribution of the process one step of size `step` after being at
  // value `x` — the process's own transition kernel.
  let step_params = |x: f64, step: f64| -> Option<(f64, f64)> {
    match (proc_name.as_str(), dargs.as_slice()) {
      ("WienerProcess", [m, s]) => {
        let (m, s) = (try_eval_to_f64(m)?, try_eval_to_f64(s)?);
        Some((x + m * step, s * step.sqrt()))
      }
      ("OrnsteinUhlenbeckProcess", [m, s, th] | [m, s, th, _]) => {
        let (m, s, th) = (
          try_eval_to_f64(m)?,
          try_eval_to_f64(s)?,
          try_eval_to_f64(th)?,
        );
        let decay = (-th * step).exp();
        let mean = m + (x - m) * decay;
        let var = s * s * (1.0 - decay * decay) / (2.0 * th);
        Some((mean, var.max(0.0).sqrt()))
      }
      _ => None,
    }
  };

  // The process's value at t0: `WienerProcess` always starts at 0;
  // `OrnsteinUhlenbeckProcess`'s 3-argument form starts in its stationary
  // distribution (mean `m`, variance `s^2 / (2 th)`), and the 4-argument
  // form starts at the given `x0`.
  let initial = match (proc_name.as_str(), dargs.as_slice()) {
    ("WienerProcess", [_, _]) => Some(0.0),
    ("OrnsteinUhlenbeckProcess", [m, s, th]) => {
      match (try_eval_to_f64(m), try_eval_to_f64(s), try_eval_to_f64(th)) {
        (Some(m), Some(s), Some(th)) => sample_normal(m, s / (2.0 * th).sqrt()),
        _ => None,
      }
    }
    ("OrnsteinUhlenbeckProcess", [_, _, _, x0]) => try_eval_to_f64(x0),
    _ => None,
  };
  let mut x = initial?;

  let mut pairs = Vec::with_capacity(n_steps as usize + 1);
  pairs.push(Expr::List(vec![Expr::Real(t0), Expr::Real(x)].into()));
  let mut t = t0;
  for _ in 0..n_steps {
    let this_step = dt.min(t1 - t);
    let (mean, sd) = step_params(x, this_step)?;
    x = sample_normal(mean, sd)?;
    t += this_step;
    pairs.push(Expr::List(vec![Expr::Real(t), Expr::Real(x)].into()));
  }

  Some(pairs)
}

/// A discrete-time linear process in the canonical form
/// `ar(L) (1 - L)^d (1 - L^s)^sd x[t] = c + ma(L) e[t]` with Gaussian white
/// noise `e` of variance `v`. `ar` and `ma` hold the coefficients of `L^1,
/// L^2, …` after multiplying out any seasonal factors, with the signs of the
/// recursion `w[t] = c + Σ ar[i] w[t-i] + e[t] + Σ ma[j] e[t-j]`.
pub struct LinearProcess {
  pub c: f64,
  pub ar: Vec<f64>,
  pub ma: Vec<f64>,
  pub v: f64,
  pub d: usize,
  pub season: usize,
  pub seasonal_d: usize,
}

fn f64_list(e: &Expr) -> Option<Vec<f64>> {
  match e {
    Expr::List(items) => items.iter().map(try_eval_to_f64).collect(),
    _ => None,
  }
}

fn small_order(e: &Expr) -> Option<usize> {
  match e {
    Expr::Integer(n) if (0..=64).contains(n) => Some(*n as usize),
    _ => None,
  }
}

/// Coefficients of `(1 + Σ a[i] L^i)(1 + Σ b[j] L^(s j))` beyond the
/// constant term; `sign` is -1 for AR polynomials (`1 - Σ a L^i`).
fn seasonal_product(a: &[f64], b: &[f64], s: usize, sign: f64) -> Vec<f64> {
  let mut p = vec![0.0; 1 + a.len() + s * b.len()];
  let left: Vec<f64> = std::iter::once(1.0)
    .chain(a.iter().map(|x| sign * x))
    .collect();
  let mut right = vec![0.0; 1 + s * b.len()];
  right[0] = 1.0;
  for (j, x) in b.iter().enumerate() {
    right[s * (j + 1)] = sign * x;
  }
  for (i, l) in left.iter().enumerate() {
    for (j, r) in right.iter().enumerate() {
      p[i + j] += l * r;
    }
  }
  p[1..].iter().map(|x| sign * x).collect()
}

/// Read an `ARProcess`, `MAProcess`, `ARMAProcess`, `ARIMAProcess`,
/// `SARMAProcess` or `SARIMAProcess` with numeric scalar parameters (an
/// optional leading constant, no initial data) into a [`LinearProcess`].
pub fn linear_process(proc: &Expr) -> Option<LinearProcess> {
  let Expr::FunctionCall { name, args } = proc else {
    return None;
  };
  let args: Vec<Expr> = args.iter().cloned().collect();
  let arity = match name.as_str() {
    "ARProcess" | "MAProcess" => 2,
    "ARMAProcess" => 3,
    "ARIMAProcess" | "SARMAProcess" => 4,
    "SARIMAProcess" => 5,
    _ => return None,
  };
  // An optional leading scalar constant shifts the remaining parameters.
  let (c, rest) = match args.len() {
    n if n == arity => (0.0, &args[..]),
    n if n == arity + 1 && !matches!(args[0], Expr::List(_)) => {
      (try_eval_to_f64(&args[0])?, &args[1..])
    }
    _ => return None,
  };
  let v = try_eval_to_f64(rest.last()?)?;
  if v < 0.0 {
    return None;
  }
  let mut model = LinearProcess {
    c,
    ar: vec![],
    ma: vec![],
    v,
    d: 0,
    season: 1,
    seasonal_d: 0,
  };
  match name.as_str() {
    "ARProcess" => model.ar = f64_list(&rest[0])?,
    "MAProcess" => model.ma = f64_list(&rest[0])?,
    "ARMAProcess" => {
      model.ar = f64_list(&rest[0])?;
      model.ma = f64_list(&rest[1])?;
    }
    "ARIMAProcess" => {
      model.ar = f64_list(&rest[0])?;
      model.d = small_order(&rest[1])?;
      model.ma = f64_list(&rest[2])?;
    }
    "SARMAProcess" | "SARIMAProcess" => {
      let integrated = name == "SARIMAProcess";
      let ar = f64_list(&rest[0])?;
      if integrated {
        model.d = small_order(&rest[1])?;
      }
      let ma = f64_list(&rest[if integrated { 2 } else { 1 }])?;
      let Expr::List(seasonal) = &rest[if integrated { 3 } else { 2 }] else {
        return None;
      };
      let seasonal: Vec<Expr> = seasonal.iter().cloned().collect();
      let (s, sar, sd, sma) = match (integrated, seasonal.as_slice()) {
        (true, [s, sar, sd, sma]) => (s, sar, small_order(sd)?, sma),
        (false, [s, sar, sma]) => (s, sar, 0, sma),
        _ => return None,
      };
      let s = small_order(s).filter(|s| *s >= 1)?;
      model.season = s;
      model.seasonal_d = sd;
      model.ar = seasonal_product(&ar, &f64_list(sar)?, s, -1.0);
      model.ma = seasonal_product(&ma, &f64_list(sma)?, s, 1.0);
    }
    _ => return None,
  }
  Some(model)
}

/// Whether every root of `1 - Σ ar[i] z^i` lies outside the unit circle,
/// i.e. the AR recursion forgets its starting values.
pub fn ar_is_stationary(ar: &[f64]) -> bool {
  let p = match ar.iter().rposition(|x| *x != 0.0) {
    Some(i) => i + 1,
    None => return true,
  };
  // Roots of 1 - Σ ar[i] z^i (coefficients in ascending powers).
  let coeffs: Vec<f64> = std::iter::once(1.0)
    .chain(ar[..p].iter().map(|a| -a))
    .collect();
  crate::functions::polynomial_ast::solve::durand_kerner_roots(&coeffs)
    .iter()
    .all(|(re, im)| re.hypot(*im) > 1.0 + 1e-12)
}

/// A realization of a white-noise or ARMA-family process over the integer
/// times `{tmin, tmax}` (or `{tmax}`, starting at 0). `None` means `proc` is
/// not such a process; `Some(None)` that it is but cannot be simulated here.
fn discrete_linear_process_path(
  proc: &Expr,
  spec: &[Expr],
) -> Option<Option<Vec<Expr>>> {
  let Expr::FunctionCall { name, args: pargs } = proc else {
    return None;
  };
  let noise: Option<(f64, f64)> = match (name.as_str(), pargs.as_slice()) {
    ("WhiteNoiseProcess", []) => Some((0.0, 1.0)),
    ("WhiteNoiseProcess", [sd]) => match try_eval_to_f64(sd) {
      Some(sd) if sd >= 0.0 => Some((0.0, sd)),
      _ => {
        // A distribution argument draws its values from that distribution.
        let Expr::FunctionCall { .. } = sd else {
          return Some(None);
        };
        return Some(white_noise_from_distribution(sd, spec));
      }
    },
    ("WhiteNoiseProcess", [m, sd]) => {
      match (try_eval_to_f64(m), try_eval_to_f64(sd)) {
        (Some(m), Some(sd)) if sd >= 0.0 => Some((m, sd)),
        _ => return Some(None),
      }
    }
    _ => None,
  };
  let model = match noise {
    Some(_) => None,
    None => Some(linear_process(proc).or_else(|| {
      // A recognized family member with symbolic or malformed parameters.
      matches!(
        name.as_str(),
        "ARProcess"
          | "MAProcess"
          | "ARMAProcess"
          | "ARIMAProcess"
          | "SARMAProcess"
          | "SARIMAProcess"
      )
      .then_some(None)?
    })?),
  };
  if noise.is_none() && model.is_none() {
    return None;
  }
  let Some((tmin, tmax)) = integer_time_range(spec) else {
    return Some(None);
  };
  let n = (tmax - tmin + 1) as usize;

  use rand_distr::{Distribution, Normal};
  let values: Vec<f64> = if let Some((m, sd)) = noise {
    let Ok(dist) = Normal::new(m, sd) else {
      return Some(None);
    };
    crate::with_rng(|rng| (0..n).map(|_| dist.sample(rng)).collect())
  } else {
    let model = model.unwrap();
    let Ok(dist) = Normal::new(0.0, model.v.sqrt()) else {
      return Some(None);
    };
    // A stationary recursion starts from its stationary regime by running
    // it in for a while first; otherwise it starts from zero history.
    let burn = if ar_is_stationary(&model.ar) {
      1000 + 50 * (model.ar.len() + model.ma.len())
    } else {
      0
    };
    let total = burn + n;
    let noise: Vec<f64> = crate::with_rng(|rng| {
      (0..total + model.ma.len())
        .map(|_| dist.sample(rng))
        .collect()
    });
    let q = model.ma.len();
    let mut w: Vec<f64> = Vec::with_capacity(total);
    for t in 0..total {
      let mut x = model.c + noise[t + q];
      for (i, a) in model.ar.iter().enumerate() {
        if t > i {
          x += a * w[t - 1 - i];
        }
      }
      for (j, b) in model.ma.iter().enumerate() {
        x += b * noise[t + q - 1 - j];
      }
      w.push(x);
    }
    let mut x: Vec<f64> = w.split_off(burn);
    // Undo the differencing, starting from zero history.
    for _ in 0..model.d {
      for t in 1..x.len() {
        x[t] += x[t - 1];
      }
    }
    for _ in 0..model.seasonal_d {
      for t in model.season..x.len() {
        x[t] += x[t - model.season];
      }
    }
    x
  };
  Some(Some(
    values
      .into_iter()
      .enumerate()
      .map(|(i, v)| {
        Expr::List(vec![Expr::Integer(tmin + i as i128), Expr::Real(v)].into())
      })
      .collect(),
  ))
}

/// `WeakStationarity[proc]` for an ARMA-family process with numeric
/// parameters: moving-average parts are always weakly stationary, an
/// integrated one never is, and an autoregressive one is when every root of
/// its AR polynomial lies outside the unit circle.
pub fn weak_stationarity_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let [proc] = args else {
    return Ok(unevaluated("WeakStationarity", args));
  };
  if let Expr::FunctionCall { name, .. } = proc
    && name == "WhiteNoiseProcess"
  {
    return Ok(bool_expr(true));
  }
  let Some(model) = linear_process(proc) else {
    return Ok(unevaluated("WeakStationarity", args));
  };
  if model.d > 0 || model.seasonal_d > 0 {
    return Ok(bool_expr(false));
  }
  let stationary = ar_is_stationary(&model.ar);
  if !stationary && let Expr::FunctionCall { name, .. } = proc {
    crate::emit_message(&format!(
      "{name}::nonwkst: The process is not weakly stationary."
    ));
  }
  Ok(bool_expr(stationary))
}

/// `{tmin, tmax}` or `{tmax}` (from 0) as an integer time range.
fn integer_time_range(spec: &[Expr]) -> Option<(i128, i128)> {
  let (tmin, tmax) = match spec {
    [Expr::Integer(tmax)] => (0, *tmax),
    [Expr::Integer(tmin), Expr::Integer(tmax)] => (*tmin, *tmax),
    _ => return None,
  };
  (tmax >= tmin && tmax - tmin < 100_000_000).then_some((tmin, tmax))
}

/// `WhiteNoiseProcess[dist]`: independent draws from `dist`.
fn white_noise_from_distribution(
  dist: &Expr,
  spec: &[Expr],
) -> Option<Vec<Expr>> {
  let (tmin, tmax) = integer_time_range(spec)?;
  let n = tmax - tmin + 1;
  let draws = crate::evaluator::evaluate_expr_to_expr(&call(
    "RandomVariate",
    vec![dist.clone(), Expr::Integer(n)],
  ))
  .ok()?;
  let Expr::List(ref draws) = draws else {
    return None;
  };
  if draws.len() as i128 != n {
    return None;
  }
  Some(
    draws
      .iter()
      .enumerate()
      .map(|(i, v)| {
        Expr::List(vec![Expr::Integer(tmin + i as i128), v.clone()].into())
      })
      .collect(),
  )
}

/// Return the `{{date, value}, ...}` pairs of a canonical `TimeSeries`.
pub fn time_series_pairs(expr: &Expr) -> Option<Vec<(Expr, Expr)>> {
  let pairs = match expr {
    Expr::FunctionCall { name, args }
      if name == "TimeSeries" && args.len() == 1 =>
    {
      match args.iter().next()? {
        Expr::List(items) => items,
        _ => return None,
      }
    }
    _ => return None,
  };
  let mut out = Vec::new();
  for p in pairs {
    if let Expr::List(kv) = p {
      let kv: Vec<_> = kv.iter().collect();
      if kv.len() == 2 {
        out.push((kv[0].clone(), kv[1].clone()));
        continue;
      }
    }
    return None;
  }
  Some(out)
}

/// `TemporalData[TimeSeries, {values, {spec}, ...}, ...]` →
/// canonical `TimeSeries[{{date, value}, ...}]`.
///
/// Also handles the user-facing constructor `TemporalData[values, {times}]`: a
/// flat list of scalar values is a single path and normalizes to a canonical
/// `TimeSeries`, while a list of value paths (a list of lists) is a multi-path
/// object that stays inert in the canonical `TemporalData[{p1, …}, {{t…}}]` form
/// — its paths are recovered by [`temporal_paths`] for plotting and queries.
pub fn temporal_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let (Some(Expr::Identifier(tag)), Some(Expr::List(fields))) =
    (args.first(), args.get(1))
    && tag == "TimeSeries"
  {
    let fields: Vec<Expr> = fields.iter().cloned().collect();
    if let Some(pairs) = build_pairs_from_temporal(&fields) {
      return Ok(time_series(pairs));
    }
    // Wolfram's own form: a list holding the one value path, sampled at the
    // regular numeric times `{{t0, t1, dt}}`.
    if let Some(pairs) = build_pairs_from_event_spec(&fields) {
      return Ok(time_series(pairs));
    }
  }

  // `TemporalData[EventSeries, {values, {{t0, t1, dt}}, …}]`: a value path
  // sampled at the regular numeric times t0, t0 + dt, …
  if let (Some(Expr::Identifier(tag)), Some(Expr::List(fields))) =
    (args.first(), args.get(1))
    && tag == "EventSeries"
    && let Some(pairs) = build_pairs_from_event_spec(fields)
  {
    return Ok(time_series(pairs));
  }

  // Constructor form `TemporalData[values, times, …]`.
  if let [Expr::List(values), Expr::List(_times), ..] = args {
    let is_multi_path =
      !values.is_empty() && values.iter().all(|v| matches!(v, Expr::List(_)));
    if is_multi_path {
      return Ok(unevaluated("TemporalData", args));
    }
    // Single scalar path → reuse the TimeSeries constructor.
    return time_series_ast(args);
  }

  Ok(unevaluated("TemporalData", args))
}

/// When every value of a single time path is a component association
/// `<|key -> component, …|>` (from `ComponentKeys`), split it into one path per
/// component, in the association's key order. Otherwise the path is returned
/// unchanged as a single element.
fn split_component_paths(pairs: Vec<(Expr, Expr)>) -> Vec<Vec<(Expr, Expr)>> {
  let all_assoc = !pairs.is_empty()
    && pairs.iter().all(|(_, v)| matches!(v, Expr::Association(_)));
  if !all_assoc {
    return vec![pairs];
  }
  let keys: Vec<Expr> = match &pairs[0].1 {
    Expr::Association(kv) => kv.iter().map(|(k, _)| k.clone()).collect(),
    _ => return vec![pairs],
  };
  keys
    .iter()
    .map(|key| {
      pairs
        .iter()
        .filter_map(|(t, v)| match v {
          Expr::Association(kv) => kv
            .iter()
            .find(|(k, _)| expr_to_output(k) == expr_to_output(key))
            .map(|(_, val)| (t.clone(), val.clone())),
          _ => None,
        })
        .collect()
    })
    .collect()
}

/// All value paths of a temporal object as `(time, value)` pair lists. A
/// `TimeSeries` (or single-path object) yields one path; a multi-path
/// `TemporalData[{p1, …}, {{t…}}]` yields one path per component, each sharing
/// the common time axis. Returns `None` for non-temporal expressions.
pub fn temporal_paths(expr: &Expr) -> Option<Vec<Vec<(Expr, Expr)>>> {
  if let Some(pairs) = time_series_pairs(expr) {
    return Some(split_component_paths(pairs));
  }
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "TemporalData" {
    return None;
  }
  let Expr::List(paths) = args.first()? else {
    return None;
  };
  // Times are wrapped as `{{t1, …}}`; unwrap the singleton path-list, otherwise
  // take the stamps directly.
  let path_len = match paths.iter().next() {
    Some(Expr::List(vals)) => vals.len(),
    _ => 0,
  };
  let times: Vec<Expr> = match args.get(1)? {
    Expr::List(items) => match (items.len(), items.iter().next()) {
      (1, Some(Expr::List(inner))) => inner.iter().cloned().collect(),
      // `{t0}`: unit steps from the start time t0.
      (1, Some(t0 @ (Expr::Integer(_) | Expr::Real(_)))) => (0..path_len)
        .map(|i| match t0 {
          Expr::Integer(t) => Expr::Integer(t + i as i128),
          Expr::Real(t) => Expr::Real(t + i as f64),
          _ => unreachable!(),
        })
        .collect(),
      _ => items.iter().cloned().collect(),
    },
    _ => return None,
  };
  let mut out = Vec::with_capacity(paths.len());
  for p in paths {
    let Expr::List(vals) = p else { return None };
    out.push(
      times
        .iter()
        .cloned()
        .zip(vals.iter().cloned())
        .collect::<Vec<_>>(),
    );
  }
  Some(out)
}

/// `TimeSeries[values]` / `TimeSeries[values, dates]` constructor. A list of
/// `{date, value}` pairs is already canonical and is returned unchanged.
pub fn time_series_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || unevaluated("TimeSeries", args);
  match args {
    [Expr::List(items)] => {
      let elems: Vec<&Expr> = items.iter().collect();
      // Already a list of {date, value} pairs → canonical, leave inert.
      let is_pairs = !elems.is_empty()
        && elems
          .iter()
          .all(|e| matches!(e, Expr::List(kv) if kv.len() == 2));
      if is_pairs {
        // A series runs in time order however its points were written, so
        // out-of-order stamps are sorted here rather than left as given.
        let mut pairs: Vec<Expr> = elems.iter().map(|e| (*e).clone()).collect();
        let ordered = pairs
          .iter()
          .all(|p| matches!(p, Expr::List(kv) if to_time(&kv[0]).is_some()));
        if ordered {
          pairs.sort_by(|a, b| {
            let key = |e: &Expr| match e {
              Expr::List(kv) => to_time(&kv[0]).unwrap_or(f64::NAN),
              _ => f64::NAN,
            };
            key(a).total_cmp(&key(b))
          });
          return Ok(call1("TimeSeries", Expr::List(pairs.into())));
        }
        return Ok(echo());
      }
      // Bare value path → assign integer times 1, 2, 3, …
      let pairs = elems
        .iter()
        .enumerate()
        .map(|(i, v)| {
          Expr::List(vec![Expr::Integer((i + 1) as i128), (*v).clone()].into())
        })
        .collect();
      Ok(time_series(pairs))
    }
    // `TimeSeries[values, tspec]` and `TimeSeries[values, tspec, keys]`. A
    // trailing list of string keys names the components of each (vector) value,
    // turning that value into an `<|key -> component, …|>` association (WL 15).
    [Expr::List(values), Expr::List(times), rest @ ..] => {
      let keys = component_keys(rest);
      let values: Vec<Expr> = values
        .iter()
        .map(|v| match &keys {
          Some(k) => apply_keys(v, k),
          None => v.clone(),
        })
        .collect();
      let times: Vec<&Expr> = times.iter().collect();

      // `{start, step}` / `{start, end, step}` date-range spec → auto-generate
      // dates. Checked before explicit-stamp pairing because a 3-element spec
      // like `{{2013,1,1}, Automatic, "Day"}` can share the values' length.
      if let Some((start, amount, unit)) = parse_range_spec(&times) {
        let dates = generate_dates(start, amount, &unit, values.len());
        let pairs = dates
          .into_iter()
          .zip(values.iter())
          .map(|(d, v)| Expr::List(vec![d, v.clone()].into()))
          .collect();
        return Ok(time_series(pairs));
      }

      // `{tmin, tmax, dt}` describing exactly one stamp per value is a
      // regular range, even when there happen to be three values.
      if let [t0, t1, dt] = times[..]
        && let (Some(a), Some(b), Some(d)) =
          (as_f64(t0), as_f64(t1), as_f64(dt))
        && d > 0.0
        && ((b - a) / d).round() as usize + 1 == values.len()
        && ((b - a) / d - ((b - a) / d).round()).abs() < 1e-9
      {
        let exact = matches!((t0, dt), (Expr::Integer(_), Expr::Integer(_)));
        let pairs = values
          .iter()
          .enumerate()
          .map(|(i, v)| {
            let t = if exact {
              match (t0, dt) {
                (Expr::Integer(a), Expr::Integer(d)) => {
                  Expr::Integer(a + d * i as i128)
                }
                _ => unreachable!(),
              }
            } else {
              Expr::Real(a + d * i as f64)
            };
            Expr::List(vec![t, v.clone()].into())
          })
          .collect();
        return Ok(time_series(pairs));
      }

      // Explicit time stamps, one per value → direct pairing.
      if times.len() == values.len() {
        let pairs = times
          .iter()
          .zip(values.iter())
          .map(|(t, v)| Expr::List(vec![(*t).clone(), v.clone()].into()))
          .collect();
        return Ok(time_series(pairs));
      }

      // A single starting specification → auto-generate the time stamps.
      if times.len() == 1 {
        match times[0] {
          // `{{t1, t2, …}}` — one path's explicit times wrapped in a list.
          Expr::List(inner) => {
            let pairs = inner
              .iter()
              .zip(values.iter())
              .map(|(t, v)| Expr::List(vec![t.clone(), v.clone()].into()))
              .collect();
            return Ok(time_series(pairs));
          }
          // `{DateObject[…]}` — daily-spaced dates from the start date.
          Expr::FunctionCall { name, .. } if name == "DateObject" => {
            if let Some(c) = extract_date_components(times[0]) {
              let dates =
                generate_dates(pad_components(&c), 1.0, "Day", values.len());
              let pairs = dates
                .into_iter()
                .zip(values.iter())
                .map(|(d, v)| Expr::List(vec![d, v.clone()].into()))
                .collect();
              return Ok(time_series(pairs));
            }
          }
          // `{n}` — numeric start, advancing by 1: n, n+1, n+2, …
          _ if as_f64(times[0]).is_some() => {
            let start = as_f64(times[0]).unwrap();
            let pairs = values
              .iter()
              .enumerate()
              .map(|(i, v)| {
                Expr::List(
                  vec![real_or_int(start + i as f64), v.clone()].into(),
                )
              })
              .collect();
            return Ok(time_series(pairs));
          }
          _ => {}
        }
      }

      // Unsupported spec → leave the constructor unevaluated.
      Ok(echo())
    }
    _ => Ok(echo()),
  }
}

/// `TimeSeriesResample[ts, weekday]` — when the second argument is a weekday
/// symbol/string (Monday … Sunday), keep only the points falling on that
/// weekday. The result is a `TimeSeries` over the matching points.
pub fn time_series_resample_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("TimeSeriesResample", args));
  if args.len() != 2 {
    return echo();
  }
  // Normalize the first argument to canonical pairs.
  let pairs = match time_series_pairs(&args[0]) {
    Some(p) => p,
    None => match &args[0] {
      Expr::FunctionCall { name, args: ta } if name == "TemporalData" => {
        let ta: Vec<Expr> = ta.iter().cloned().collect();
        let ts = temporal_data_ast(&ta)?;
        match time_series_pairs(&ts) {
          Some(p) => p,
          None => return echo(),
        }
      }
      _ => return echo(),
    },
  };

  if let Expr::Identifier(s) | Expr::String(s) = &args[1]
    && let Some(target) = weekday_index(s)
  {
    let filtered: Vec<Expr> = pairs
      .into_iter()
      .filter(|(date, _)| {
        date_components(date).is_some_and(|c| {
          day_of_week(c[0] as i64, c[1] as i64, c[2] as i64) == target
        })
      })
      .map(|(d, v)| Expr::List(vec![d, v].into()))
      .collect();
    return Ok(time_series(filtered));
  }

  match resample_times(&pairs, Some(&args[1]))? {
    Some(times) => resample_at(&pairs, &times),
    None => echo(),
  }
}

/// `TimeSeriesResample[ts]` — resample at the series' own minimum time
/// increment, over its full time span.
pub fn time_series_resample_default(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("TimeSeriesResample", args));
  let Some(pairs) = time_series_pairs(&args[0]) else {
    return echo();
  };
  match resample_times(&pairs, None)? {
    Some(times) => resample_at(&pairs, &times),
    None => echo(),
  }
}

/// The smallest gap between consecutive (numeric) time stamps.
fn minimum_increment(pairs: &[(Expr, Expr)]) -> Option<Expr> {
  let mut best: Option<(f64, Expr)> = None;
  for w in pairs.windows(2) {
    let (a, b) = (to_time(&w[0].0)?, to_time(&w[1].0)?);
    let gap = b - a;
    if gap > 0.0 && best.as_ref().is_none_or(|(g, _)| gap < *g) {
      best = Some((gap, arith("Subtract", &w[1].0, &w[0].0).ok()?));
    }
  }
  best.map(|(_, e)| e)
}

/// Evaluate a two-argument arithmetic head on expressions, so resampled times
/// and interpolated values stay exact when the inputs are.
fn arith(head: &str, a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(&call(
    head,
    vec![a.clone(), b.clone()],
  ))
}

/// The sample times a resampling specification asks for. `None` for the spec
/// means "the series' own minimum increment over its full span". Returns
/// `Ok(None)` when the spec is not one of the recognised numeric forms, so the
/// caller can leave the call unevaluated.
///
/// Recognised: a bare step `dt`, `{tmin, tmax}`, `{tmin, tmax, dt}`, and an
/// explicit list of stamps `{{t1, t2, …}}`.
fn resample_times(
  pairs: &[(Expr, Expr)],
  spec: Option<&Expr>,
) -> Result<Option<Vec<Expr>>, InterpreterError> {
  if pairs.len() < 2 {
    return Ok(None);
  }
  // Only numeric time axes are resampled here; date stamps keep the weekday
  // form above.
  if pairs
    .iter()
    .any(|(t, _)| !matches!(t, Expr::Integer(_) | Expr::Real(_)))
  {
    return Ok(None);
  }

  // An explicit `{{t1, t2, …}}` list of stamps needs no stepping.
  if let Some(Expr::List(outer)) = spec
    && outer.len() == 1
    && let Expr::List(stamps) = &outer[0]
  {
    return Ok(Some(stamps.iter().cloned().collect()));
  }

  let first = pairs[0].0.clone();
  let last = pairs[pairs.len() - 1].0.clone();
  let (start, end, step) = match spec {
    None => (first, last, minimum_increment(pairs)),
    Some(Expr::List(range)) => match range.len() {
      2 => (range[0].clone(), range[1].clone(), minimum_increment(pairs)),
      3 => (range[0].clone(), range[1].clone(), Some(range[2].clone())),
      _ => return Ok(None),
    },
    Some(dt @ (Expr::Integer(_) | Expr::Real(_))) => {
      (first, last, Some(dt.clone()))
    }
    _ => return Ok(None),
  };
  let Some(step) = step else { return Ok(None) };

  let (Some(t0), Some(t1), Some(dt)) =
    (to_time(&start), to_time(&end), to_time(&step))
  else {
    return Ok(None);
  };
  if dt <= 0.0 || t1 < t0 {
    return Ok(None);
  }

  // Build the stamps by exact arithmetic (`start + k*step`) so an integer
  // step keeps integer stamps, while the count comes from the numeric span.
  let count = ((t1 - t0) / dt + 1e-9).floor() as usize;
  let mut times = Vec::with_capacity(count + 1);
  for k in 0..=count {
    // An exact `0*step` would vanish, but a Real step keeps the whole
    // time axis approximate, first stamp included.
    let offset = match &step {
      Expr::Real(_) if k == 0 => Expr::Real(0.0),
      _ => arith("Times", &Expr::Integer(k as i128), &step)?,
    };
    times.push(arith("Plus", &start, &offset)?);
  }
  Ok(Some(times))
}

/// Sample `pairs` at each of `times` by linear interpolation, exactly.
fn resample_at(
  pairs: &[(Expr, Expr)],
  times: &[Expr],
) -> Result<Expr, InterpreterError> {
  let mut out = Vec::with_capacity(times.len());
  for t in times {
    let Some(value) = interpolate_exact(pairs, t)? else {
      return Ok(unevaluated(
        "TimeSeriesResample",
        &[time_series(
          pairs
            .iter()
            .map(|(d, v)| Expr::List(vec![d.clone(), v.clone()].into()))
            .collect(),
        )],
      ));
    };
    out.push(Expr::List(vec![t.clone(), value].into()));
  }
  Ok(time_series(out))
}

/// The value of the piecewise-linear path through `pairs` at time `q`, kept
/// exact: interpolating integer data at an integer stamp gives an integer.
/// Returns `None` when a time or value is not numeric.
fn interpolate_exact(
  pairs: &[(Expr, Expr)],
  q: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  let Some(qt) = to_time(q) else {
    return Ok(None);
  };
  let mut times = Vec::with_capacity(pairs.len());
  for (t, _) in pairs {
    match to_time(t) {
      Some(v) => times.push(v),
      None => return Ok(None),
    }
  }
  // A stamp that lands on a data point returns the stored value. An inexact
  // stamp still numericizes it, so a Real time axis carries Real values —
  // the same precision the interpolated stamps get from the arithmetic.
  if let Some(i) = times.iter().position(|t| (t - qt).abs() < 1e-9) {
    let value = &pairs[i].1;
    if matches!(q, Expr::Real(_)) {
      return Ok(Some(crate::evaluator::evaluate_expr_to_expr(&call1(
        "N",
        value.clone(),
      ))?));
    }
    return Ok(Some(value.clone()));
  }
  // Outside the sampled span the path is held flat at its end value; Wolfram
  // clamps rather than extrapolating the end segment's slope.
  let last = times.len() - 1;
  if qt < times[0] {
    return Ok(Some(pairs[0].1.clone()));
  }
  if qt > times[last] {
    return Ok(Some(pairs[last].1.clone()));
  }
  let seg = times
    .windows(2)
    .position(|w| qt <= w[1])
    .unwrap_or(last - 1);
  let (t0, y0) = (&pairs[seg].0, &pairs[seg].1);
  let (t1, y1) = (&pairs[seg + 1].0, &pairs[seg + 1].1);
  let slope = arith(
    "Divide",
    &arith("Subtract", y1, y0)?,
    &arith("Subtract", t1, t0)?,
  )?;
  let offset = arith("Times", &slope, &arith("Subtract", q, t0)?)?;
  Ok(Some(arith("Plus", y0, &offset)?))
}

/// `TimeSeriesWindow[ts, {tmin, tmax}]` — the points of `ts` whose time stamps
/// lie in `[tmin, tmax]`, both ends included. The bounds may be given in either
/// order and may be `±Infinity`. A window that catches nothing yields an empty
/// TimeSeries and reports `TimeSeriesWindow::tswndt`.
pub fn time_series_window_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Ok(unevaluated("TimeSeriesWindow", args));
  if args.len() != 2 {
    return echo();
  }
  // A TemporalData argument is normalized through the TimeSeries constructor
  // first, exactly as TimeSeriesResample does.
  let pairs = match time_series_pairs(&args[0]) {
    Some(p) => p,
    None => match &args[0] {
      Expr::FunctionCall { name, args: ta } if name == "TemporalData" => {
        let ta: Vec<Expr> = ta.iter().cloned().collect();
        let ts = temporal_data_ast(&ta)?;
        match time_series_pairs(&ts) {
          Some(p) => p,
          None => return echo(),
        }
      }
      _ => return echo(),
    },
  };

  let Expr::List(bounds) = &args[1] else {
    return echo();
  };
  let bounds: Vec<&Expr> = bounds.iter().collect();
  let [lo, hi] = bounds[..] else { return echo() };
  let (Some(lo), Some(hi)) = (window_bound(lo), window_bound(hi)) else {
    return echo();
  };
  let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };

  let kept: Vec<Expr> = pairs
    .into_iter()
    .filter(|(time, _)| to_time(time).is_some_and(|t| t >= lo && t <= hi))
    .map(|(t, v)| Expr::List(vec![t, v].into()))
    .collect();

  if kept.is_empty() {
    crate::emit_message(&format!(
      "TimeSeriesWindow::tswndt: The window {} contains no values on the path(s) {}.",
      crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output),
      crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
    ));
  }
  Ok(time_series(kept))
}

/// A window endpoint: a time stamp, or `±Infinity` for an open end.
fn window_bound(e: &Expr) -> Option<f64> {
  if matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity") {
    return Some(f64::INFINITY);
  }
  if expr_to_string(e) == "-Infinity" {
    return Some(f64::NEG_INFINITY);
  }
  to_time(e)
}

/// If `expr` is a `TimeSeries`, return the list of its values (the value path),
/// for descriptive statistics such as `Mean`, `Total`, `Min`, `Max`.
pub fn time_series_values(expr: &Expr) -> Option<Expr> {
  let pairs = time_series_pairs(expr)?;
  Some(Expr::List(
    pairs.into_iter().map(|(_, v)| v).collect::<Vec<_>>().into(),
  ))
}

/// A trailing constructor argument that names the components of each vector
/// value (WL 15): either a bare list of string keys, or the option
/// `ComponentKeys -> {"a", …}`. Anything else (other options, etc.) is not
/// treated as component keys.
fn component_keys(rest: &[Expr]) -> Option<Vec<String>> {
  let key_list = rest.iter().find_map(|e| match e {
    Expr::List(_) => Some(e),
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(),
      Expr::Identifier(n) if n == "ComponentKeys") =>
    {
      Some(replacement.as_ref())
    }
    _ => None,
  })?;
  let Expr::List(items) = key_list else {
    return None;
  };
  let keys: Option<Vec<String>> = items
    .iter()
    .map(|e| match e {
      Expr::String(s) => Some(s.clone()),
      _ => None,
    })
    .collect();
  keys.filter(|k| !k.is_empty())
}

/// Turn a vector `value` into an `<|key -> component, …|>` association when its
/// length matches `keys`; otherwise leave the value untouched.
fn apply_keys(value: &Expr, keys: &[String]) -> Expr {
  match value {
    Expr::List(items) if items.len() == keys.len() => Expr::Association(
      keys
        .iter()
        .cloned()
        .zip(items.iter().cloned())
        .map(|(k, v)| (Expr::String(k), v))
        .collect(),
    ),
    _ => value.clone(),
  }
}

/// `Values[ts]` — the value path. A component-keyed series (whose values are
/// associations) materializes as a `Tabular`, matching WL 15; otherwise the
/// plain list of values is returned.
pub fn time_series_values_output(expr: &Expr) -> Option<Expr> {
  let pairs = time_series_pairs(expr)?;
  let values: Vec<Expr> = pairs.into_iter().map(|(_, v)| v).collect();
  if !values.is_empty()
    && values.iter().all(|v| matches!(v, Expr::Association(_)))
  {
    return Some(crate::functions::tabular_ast::tabular_ast(&[Expr::List(
      values.into(),
    )]));
  }
  Some(Expr::List(values.into()))
}

/// Pad a component vector to `{y, m, d, h, min, sec}`, defaulting month/day to 1
/// and the time fields to 0.
fn pad_components(c: &[f64]) -> [f64; 6] {
  let mut out = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
  for (i, slot) in out.iter_mut().enumerate() {
    if let Some(v) = c.get(i) {
      *slot = *v;
    }
  }
  out
}

/// Render a time value as an `Integer` when whole, otherwise a `Real`.
fn real_or_int(t: f64) -> Expr {
  if t.fract() == 0.0 && t.abs() < 9.007e15 {
    Expr::Integer(t as i128)
  } else {
    Expr::Real(t)
  }
}

/// Convert a time stamp (a plain number, a date list, or a `DateObject`) to a
/// scalar time: numeric stamps pass through; dates become AbsoluteTime seconds.
pub fn to_time(e: &Expr) -> Option<f64> {
  if let Some(n) = as_f64(e) {
    return Some(n);
  }
  let c = pad_components(&extract_date_components(e)?);
  Some(date_to_absolute_seconds(
    c[0] as i64,
    c[1] as i64,
    c[2] as i64,
    c[3] as i64,
    c[4] as i64,
    c[5],
  ))
}

/// Convert a stored time stamp to the `DateObject[{y,m,d,h,min,sec}, Instant,
/// Gregorian, 0.]` form that `Normal`, `FirstDate`, and `LastDate` expose. The
/// component list is padded to six fields with integer zeros, preserving the
/// original element types — so a `{…, 0, 0, 0.}` date list keeps its Real
/// seconds while a `DateObject[{y, m, d}, Day]` pads with integer zeros, exactly
/// as WL does. Returns `None` for a non-date (numeric) stamp.
fn instant_date_object(date: &Expr) -> Option<Expr> {
  let mut comps: Vec<Expr> = match date {
    Expr::FunctionCall { name, args } if name == "DateObject" => {
      match args.first()? {
        Expr::List(items) => items.iter().cloned().collect(),
        _ => return None,
      }
    }
    Expr::List(items) => items.iter().cloned().collect(),
    _ => return None,
  };
  // Must be a numeric date list, not e.g. a value vector like {0.1, "cat"}.
  if comps.is_empty() || !comps.iter().all(|c| as_f64(c).is_some()) {
    return None;
  }
  while comps.len() < 6 {
    comps.push(Expr::Integer(0));
  }
  Some(call(
    "DateObject",
    vec![
      Expr::List(comps.into()),
      str_expr("Instant"),
      str_expr("Gregorian"),
      Expr::Real(0.0),
    ],
  ))
}

/// `Normal[ts]` — the explicit `{{date, value}, …}` list, with each date stamp
/// surfaced as an `Instant`-granularity `DateObject`. A non-date (numeric) stamp
/// is left unchanged.
pub fn time_series_normal(ts: &Expr) -> Option<Expr> {
  let pairs = series_pairs_of(ts)?;
  Some(Expr::List(
    pairs
      .into_iter()
      .map(|(date, value)| {
        let d = instant_date_object(&date).unwrap_or(date);
        Expr::List(vec![d, value].into())
      })
      .collect(),
  ))
}

/// Apply a `TimeSeries` to an argument: `ts["property"]` returns a path
/// component, and `ts[t]` (a date or number) returns the value at time `t`,
/// linearly interpolating between — and extrapolating beyond — the data points.
pub fn apply_time_series(
  ts: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::CurriedCall {
      func: Box::new(ts.clone()),
      args: vec![arg.clone()],
    })
  };
  // An EventSeries answers the same property queries.
  let Some(pairs) = series_pairs_of(ts) else {
    return unevaluated();
  };

  // Numeric times paired with the stored value expressions, kept in input
  // order (which the constructors already produce ascending).
  let mut points: Vec<(f64, &Expr)> = Vec::with_capacity(pairs.len());
  for (date, value) in &pairs {
    match to_time(date) {
      Some(t) => points.push((t, value)),
      None => return unevaluated(),
    }
  }

  // Property access: ts["Path"], ts["Values"], ts["Times"], ts["FirstDate"], …
  // These stay meaningful on an empty series, which an empty window produces.
  if let Expr::String(prop) = arg {
    return apply_property(&pairs, &points, prop).map_or_else(unevaluated, Ok);
  }

  // A value lookup needs at least one point to read or interpolate from.
  if pairs.is_empty() {
    return unevaluated();
  }

  // Value lookup at a time stamp.
  let Some(q) = to_time(arg) else {
    return unevaluated();
  };

  let _ = q;
  // Piecewise-linear lookup, kept exact and clamped at both ends.
  match interpolate_exact(&pairs, arg)? {
    Some(value) => Ok(value),
    None => unevaluated(),
  }
}

/// Apply a multi-path `TemporalData[{p1, …}, tspec]` to an argument. The
/// path-list properties report every path; the rest answer for the first
/// path, as Wolfram does, and a time slice is the empirical distribution
/// of the paths' values there.
pub fn apply_multi_path_temporal_data(
  td: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::CurriedCall {
      func: Box::new(td.clone()),
      args: vec![arg.clone()],
    })
  };
  let Some(paths) = temporal_paths(td).filter(|p| !p.is_empty()) else {
    return unevaluated();
  };
  let as_series = |path: &Vec<(Expr, Expr)>| {
    call1(
      "TimeSeries",
      Expr::List(
        path
          .iter()
          .map(|(t, v)| Expr::List(vec![t.clone(), v.clone()].into()))
          .collect(),
      ),
    )
  };
  let each = |prop: &str| -> Result<Expr, InterpreterError> {
    let mut out = Vec::with_capacity(paths.len());
    for path in &paths {
      out.push(apply_time_series(&as_series(path), &str_expr(prop))?);
    }
    Ok(Expr::List(out.into()))
  };
  if let Expr::String(prop) = arg {
    return match prop.as_str() {
      "Paths" => each("Path"),
      "ValueList" | "States" => each("Values"),
      "TimeList" => each("Times"),
      "PathCount" => Ok(Expr::Integer(paths.len() as i128)),
      _ => apply_time_series(&as_series(&paths[0]), arg),
    };
  }
  // A time slice is the empirical distribution of the paths' values.
  let mut out = Vec::with_capacity(paths.len());
  for path in &paths {
    out.push(apply_time_series(&as_series(path), arg)?);
  }
  if out.iter().any(|v| matches!(v, Expr::CurriedCall { .. })) {
    return unevaluated();
  }
  crate::evaluator::evaluate_expr_to_expr(&call1(
    "EmpiricalDistribution",
    Expr::List(out.into()),
  ))
}

/// Resolve a string property access on a `TimeSeries`.
fn apply_property(
  pairs: &[(Expr, Expr)],
  points: &[(f64, &Expr)],
  prop: &str,
) -> Option<Expr> {
  let date_object = |date: &Expr| instant_date_object(date);
  // Numeric stamps echo verbatim; date stamps surface as AbsoluteTime Reals.
  let time_stamp = |date: &Expr, t: f64| match date {
    Expr::Integer(_) | Expr::Real(_) => date.clone(),
    // A rescaled series carries rational stamps; those are numbers too and
    // stay exact rather than turning into their AbsoluteTime.
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      date.clone()
    }
    _ => Expr::Real(t),
  };
  match prop {
    "Values" => {
      Some(Expr::List(pairs.iter().map(|(_, v)| v.clone()).collect()))
    }
    // A numeric time stamp is reported verbatim; a date stamp becomes its
    // AbsoluteTime (a Real), matching how WL exposes the time axis.
    "Path" => Some(Expr::List(
      pairs
        .iter()
        .zip(points)
        .map(|((date, v), (t, _))| {
          Expr::List(vec![time_stamp(date, *t), v.clone()].into())
        })
        .collect(),
    )),
    "Times" => Some(Expr::List(
      pairs
        .iter()
        .zip(points)
        .map(|((date, _), (t, _))| time_stamp(date, *t))
        .collect(),
    )),
    "FirstDate" | "MinDate" => date_object(&pairs.first()?.0),
    "LastDate" | "MaxDate" => date_object(&pairs.last()?.0),
    "FirstValue" => Some(pairs.first()?.1.clone()),
    "LastValue" => Some(pairs.last()?.1.clone()),
    "PathLength" => Some(Expr::Integer(pairs.len() as i128)),
    "FirstTime" | "MinTime" => {
      let (date, (t, _)) = (&pairs.first()?.0, points.first()?);
      Some(time_stamp(date, *t))
    }
    "LastTime" | "MaxTime" => {
      let (date, (t, _)) = (&pairs.last()?.0, points.last()?);
      Some(time_stamp(date, *t))
    }
    // The rank of each value: 1 for scalars, the list length for vectors.
    "ValueDimensions" => match &pairs.first()?.1 {
      Expr::List(items) => Some(Expr::Integer(items.len() as i128)),
      _ => Some(Expr::Integer(1)),
    },
    // `RandomFunction[…]["States"]` — wolframscript reports every
    // realization the same way whether one or several were requested, so a
    // single realization (the only kind Woxi's `RandomFunction` produces) is
    // still wrapped as a one-element list of paths: `[[1]]` then extracts it.
    "States" => Some(Expr::List(
      vec![Expr::List(pairs.iter().map(|(_, v)| v.clone()).collect())].into(),
    )),
    _ => None,
  }
}

/// EventSeriesQ[expr] — True only for an `EventSeries` object. A `TimeSeries`
/// (including the one `EventSeriesAccumulate` returns) is not one.
pub fn event_series_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let is_event_series = matches!(&args[0], Expr::FunctionCall { name, args: a }
    if name == "EventSeries" && a.len() == 1)
    && series_pairs_of(&args[0]).is_some();
  Ok(bool_expr(is_event_series))
}

/// EventSeriesLookup[series, t] — the events nearest to `t`, as `{time, value}`
/// pairs. Every event at the minimal distance is returned, so a `t` exactly
/// between two of them gives both.
pub fn event_series_lookup_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unchanged = || Ok(unevaluated("EventSeriesLookup", args));
  let (Some(pairs), Some(target)) =
    (series_pairs_of(&args[0]), to_time(&args[1]))
  else {
    return unchanged();
  };
  let mut best: Option<f64> = None;
  let mut distances = Vec::with_capacity(pairs.len());
  for (stamp, _) in &pairs {
    let Some(t) = to_time(stamp) else {
      return unchanged();
    };
    let d = (t - target).abs();
    distances.push(d);
    if best.is_none_or(|b| d < b) {
      best = Some(d);
    }
  }
  let Some(best) = best else {
    return unchanged();
  };
  let events: Vec<Expr> = pairs
    .into_iter()
    .zip(distances)
    .filter(|(_, d)| *d == best)
    .map(|((stamp, value), _)| Expr::List(vec![stamp, value].into()))
    .collect();
  Ok(Expr::List(events.into()))
}

/// EventSeriesAccumulate[series] — the running count of events, as a
/// `TimeSeries` stamped at the event times. The values themselves play no
/// part: what accumulates is how many events have occurred.
pub fn event_series_accumulate_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Some(pairs) = series_pairs_of(&args[0]) else {
    return Ok(unevaluated("EventSeriesAccumulate", args));
  };
  let counted: Vec<Expr> = pairs
    .into_iter()
    .enumerate()
    .map(|(i, (stamp, _))| {
      Expr::List(vec![stamp, Expr::Integer(i as i128 + 1)].into())
    })
    .collect();
  Ok(call1("TimeSeries", Expr::List(counted.into())))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn date(y: i64, m: i64, d: i64) -> [f64; 6] {
    [y as f64, m as f64, d as f64, 0.0, 0.0, 0.0]
  }

  fn render(e: &Expr) -> String {
    expr_to_string(e)
  }

  #[test]
  fn day_step_advances_one_day() {
    let dates = generate_dates(date(2013, 4, 1), 1.0, "Day", 3);
    assert_eq!(dates.len(), 3);
    // third date is 2013-04-03
    assert_eq!(render(&dates[2]), "{2013, 4, 3, 0, 0, 0.}");
  }

  #[test]
  fn month_step_rolls_over_year() {
    let dates = generate_dates(date(2013, 11, 15), 1.0, "Month", 3);
    // November + 2 months → January of the next year
    assert_eq!(render(&dates[2]), "{2014, 1, 15, 0, 0, 0.}");
  }
}
