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
};
use crate::syntax::Expr;

/// Monday = 0 … Sunday = 6, matching `day_of_week`.
fn weekday_index(name: &str) -> Option<i64> {
  match name {
    "Monday" => Some(0),
    "Tuesday" => Some(1),
    "Wednesday" => Some(2),
    "Thursday" => Some(3),
    "Friday" => Some(4),
    "Saturday" => Some(5),
    "Sunday" => Some(6),
    _ => None,
  }
}

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
  Ok(Expr::FunctionCall {
    name: "TemporalData".to_string(),
    args: args.to_vec().into(),
  })
}

/// `TimeSeries[values]` / `TimeSeries[values, dates]` constructor. A list of
/// `{date, value}` pairs is already canonical and is returned unchanged.
pub fn time_series_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let echo = || Expr::FunctionCall {
    name: "TimeSeries".to_string(),
    args: args.to_vec().into(),
  };
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
    [Expr::List(values), Expr::List(times)] => {
      let pairs = times
        .iter()
        .zip(values.iter())
        .map(|(t, v)| Expr::List(vec![t.clone(), v.clone()].into()))
        .collect();
      Ok(time_series(pairs))
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
  let echo = || {
    Ok(Expr::FunctionCall {
      name: "TimeSeriesResample".to_string(),
      args: args.to_vec().into(),
    })
  };
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
