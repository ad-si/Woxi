//! Temporal data: `TemporalData`, `TimeSeries`, and `TimeSeriesResample`.
//!
//! A time series is normalized to the canonical inert form
//! `TimeSeries[{{date, value}, ...}]`, where each `date` is a date list
//! `{y, m, d, h, min, sec}`. `TemporalData[TimeSeries, {values, {spec}, ...}]`
//! (the internal full form produced by, e.g., `CompressedData`-backed data) is
//! rebuilt into this canonical form by pairing the value path with the dates
//! generated from the embedded date specification.

use crate::InterpreterError;
use crate::functions::datetime_ast::{
  date_to_absolute_seconds, day_of_week, days_in_month,
  extract_date_components, weekday_index,
};
use crate::syntax::{Expr, unevaluated};

fn as_f64(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(i) => Some(*i as f64),
    Expr::Real(r) => Some(*r),
    Expr::BigInteger(b) => {
      use num_traits::ToPrimitive;
      b.to_f64()
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

/// Build canonical `{{date, value}, ...}` pairs from a value path and the date
/// specification embedded in `TemporalData`'s field list.
fn build_pairs_from_temporal(fields: &[Expr]) -> Option<Vec<Expr>> {
  let values: Vec<Expr> = match fields.first()? {
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

fn time_series(pairs: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: "TimeSeries".to_string(),
    args: vec![Expr::List(pairs.into())].into(),
  }
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
  for p in pairs.iter() {
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

/// All value paths of a temporal object as `(time, value)` pair lists. A
/// `TimeSeries` (or single-path object) yields one path; a multi-path
/// `TemporalData[{p1, …}, {{t…}}]` yields one path per component, each sharing
/// the common time axis. Returns `None` for non-temporal expressions.
pub fn temporal_paths(expr: &Expr) -> Option<Vec<Vec<(Expr, Expr)>>> {
  if let Some(pairs) = time_series_pairs(expr) {
    return Some(vec![pairs]);
  }
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "TemporalData" {
    return None;
  }
  let paths = match args.first()? {
    Expr::List(p) => p,
    _ => return None,
  };
  // Times are wrapped as `{{t1, …}}`; unwrap the singleton path-list, otherwise
  // take the stamps directly.
  let times: Vec<Expr> = match args.get(1)? {
    Expr::List(items) => match (items.len(), items.iter().next()) {
      (1, Some(Expr::List(inner))) => inner.iter().cloned().collect(),
      _ => items.iter().cloned().collect(),
    },
    _ => return None,
  };
  let mut out = Vec::with_capacity(paths.len());
  for p in paths.iter() {
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

  let target = match &args[1] {
    Expr::Identifier(s) | Expr::String(s) => weekday_index(s),
    _ => None,
  };
  let Some(target) = target else { return echo() };

  let filtered: Vec<Expr> = pairs
    .into_iter()
    .filter(|(date, _)| {
      date_components(date)
        .map(|c| day_of_week(c[0] as i64, c[1] as i64, c[2] as i64) == target)
        .unwrap_or(false)
    })
    .map(|(d, v)| Expr::List(vec![d, v].into()))
    .collect();

  Ok(time_series(filtered))
}

/// If `expr` is a `TimeSeries`, return the list of its values (the value path),
/// for descriptive statistics such as `Mean`, `Total`, `Min`, `Max`.
pub fn time_series_values(expr: &Expr) -> Option<Expr> {
  let pairs = time_series_pairs(expr)?;
  Some(Expr::List(
    pairs.into_iter().map(|(_, v)| v).collect::<Vec<_>>().into(),
  ))
}

/// A trailing constructor argument that is a non-empty list of string keys names
/// the components of each vector value (WL 15). Anything else (options, etc.) is
/// not treated as component keys.
fn component_keys(rest: &[Expr]) -> Option<Vec<String>> {
  let Expr::List(items) = rest.first()? else {
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
fn to_time(e: &Expr) -> Option<f64> {
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
  Some(Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![
      Expr::List(comps.into()),
      Expr::String("Instant".to_string()),
      Expr::String("Gregorian".to_string()),
      Expr::Real(0.0),
    ]
    .into(),
  })
}

/// `Normal[ts]` — the explicit `{{date, value}, …}` list, with each date stamp
/// surfaced as an `Instant`-granularity `DateObject`. A non-date (numeric) stamp
/// is left unchanged.
pub fn time_series_normal(ts: &Expr) -> Option<Expr> {
  let pairs = time_series_pairs(ts)?;
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
  let Some(pairs) = time_series_pairs(ts) else {
    return unevaluated();
  };
  if pairs.is_empty() {
    return unevaluated();
  }

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
  if let Expr::String(prop) = arg {
    return apply_property(&pairs, &points, prop).map_or_else(unevaluated, Ok);
  }

  // Value lookup at a time stamp.
  let Some(q) = to_time(arg) else {
    return unevaluated();
  };

  // Exact hit returns the stored value expression unchanged.
  for (t, v) in &points {
    if (*t - q).abs() < 1e-6 {
      return Ok((*v).clone());
    }
  }

  // Linear interpolation / extrapolation needs the numeric values.
  let ys: Vec<f64> = match points.iter().map(|(_, v)| as_f64(v)).collect() {
    Some(ys) => ys,
    None => return unevaluated(),
  };
  if points.len() == 1 {
    return Ok(Expr::Real(ys[0]));
  }
  let lerp = |i: usize, j: usize| {
    let (t0, t1) = (points[i].0, points[j].0);
    ys[i] + (ys[j] - ys[i]) * (q - t0) / (t1 - t0)
  };
  let last = points.len() - 1;
  let y = if q < points[0].0 {
    lerp(0, 1) // extrapolate below the first point
  } else if q > points[last].0 {
    lerp(last - 1, last) // extrapolate above the last point
  } else {
    let seg = points
      .windows(2)
      .position(|w| q <= w[1].0)
      .unwrap_or(last - 1);
    lerp(seg, seg + 1)
  };
  Ok(Expr::Real(y))
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
    _ => None,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn date(y: i64, m: i64, d: i64) -> [f64; 6] {
    [y as f64, m as f64, d as f64, 0.0, 0.0, 0.0]
  }

  fn render(e: &Expr) -> String {
    crate::syntax::expr_to_string(e)
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
