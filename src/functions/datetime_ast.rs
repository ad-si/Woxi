//! AST-native date/time functions.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Parse an ISO-style date/time string into its integer components, e.g.
/// `"2024-03-15"` → `{2024, 3, 15}` and `"2024-03-15 14:30:00"` →
/// `{2024, 3, 15, 14, 30, 0}`. The date and time may be separated by a space or
/// `T`. Returns `None` for anything that is not a plain `Y[-M[-D]][ HH:MM:SS]`
/// string (e.g. natural-language dates), so the caller can leave it unparsed.
pub fn parse_iso_date_components(s: &str) -> Option<Vec<Expr>> {
  let s = s.trim();
  if s.is_empty() {
    return None;
  }
  let (date_part, time_part) = match s.find(['T', ' ']) {
    Some(i) => (&s[..i], Some(s[i + 1..].trim())),
    None => (s, None),
  };

  // Date: YYYY[-MM[-DD]] with each field a positive integer in range.
  let mut comps: Vec<i64> = Vec::new();
  for piece in date_part.split('-') {
    comps.push(piece.trim().parse::<i64>().ok()?);
  }
  let valid_date = match comps.as_slice() {
    [y] => (1..=9999).contains(y),
    [y, m] => (1..=9999).contains(y) && (1..=12).contains(m),
    [y, m, d] => {
      (1..=9999).contains(y) && (1..=12).contains(m) && (1..=31).contains(d)
    }
    _ => false,
  };
  if !valid_date {
    return None;
  }

  // Time: HH[:MM[:SS]] — only meaningful with a full Y-M-D date.
  if let Some(t) = time_part {
    if comps.len() != 3 || t.is_empty() {
      return None;
    }
    for piece in t.split(':') {
      // Seconds may be fractional; truncate toward zero.
      comps.push(piece.trim().parse::<f64>().ok()? as i64);
    }
  }

  Some(
    comps
      .into_iter()
      .map(|n| Expr::Integer(n as i128))
      .collect(),
  )
}

fn is_leap_year(year: i64) -> bool {
  (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i64, month: i64) -> i64 {
  match month {
    1 => 31,
    2 => {
      if is_leap_year(year) {
        29
      } else {
        28
      }
    }
    3 => 31,
    4 => 30,
    5 => 31,
    6 => 30,
    7 => 31,
    8 => 31,
    9 => 30,
    10 => 31,
    11 => 30,
    12 => 31,
    _ => 30,
  }
}

fn days_in_year(year: i64) -> i64 {
  if is_leap_year(year) { 366 } else { 365 }
}

/// DayCount[d1, d2, weekday]: count occurrences of `weekday` in the interval
/// between the two dates, excluding the earlier endpoint and including the
/// later (half-open `(min, max]`). Symmetric for reversed dates. Returns None
/// when an argument can't be interpreted (caller leaves it unevaluated).
pub fn day_count_weekday_ast(
  d1: &Expr,
  d2: &Expr,
  weekday: &Expr,
) -> Option<Expr> {
  let target = match weekday {
    Expr::Identifier(s) | Expr::String(s) => weekday_index(s)?,
    _ => return None,
  };
  let ymd = |e: &Expr| -> Option<(i64, i64, i64)> {
    let c = extract_date_components(e)?;
    Some((
      *c.first()? as i64,
      c.get(1).map(|v| *v as i64).unwrap_or(1),
      c.get(2).map(|v| *v as i64).unwrap_or(1),
    ))
  };
  let (y1, m1, dd1) = ymd(d1)?;
  let (y2, m2, dd2) = ymd(d2)?;
  let abs1 = date_to_absolute_days(y1, m1, dd1);
  let abs2 = date_to_absolute_days(y2, m2, dd2);
  let dow1 = day_of_week(y1, m1, dd1);
  let dow2 = day_of_week(y2, m2, dd2);
  let (lo, hi, lo_dow) = if abs1 <= abs2 {
    (abs1, abs2, dow1)
  } else {
    (abs2, abs1, dow2)
  };
  let n = hi - lo; // number of days in (lo, hi]
  // Count k in [1, n] with (lo_dow + k) ≡ target (mod 7), i.e. k ≡ r (mod 7).
  let r = ((target - lo_dow) % 7 + 7) % 7;
  let count = if r == 0 {
    n / 7
  } else if r <= n {
    (n - r) / 7 + 1
  } else {
    0
  };
  Some(Expr::Integer(count as i128))
}

/// Convert a date {y,m,d} to absolute days since 1900-01-01 (day 0)
fn date_to_absolute_days(year: i64, month: i64, day: i64) -> i64 {
  let mut total_days: i64 = 0;

  if year >= 1900 {
    for y in 1900..year {
      total_days += days_in_year(y);
    }
  } else {
    for y in year..1900 {
      total_days -= days_in_year(y);
    }
  }

  for m in 1..month {
    total_days += days_in_month(year, m);
  }

  total_days += day - 1; // day 1 of month = 0 additional days
  total_days
}

/// Convert absolute days since 1900-01-01 to {year, month, day}
fn absolute_days_to_date(mut days: i64) -> (i64, i64, i64) {
  let mut year: i64 = 1900;

  if days >= 0 {
    loop {
      let dy = days_in_year(year);
      if days < dy {
        break;
      }
      days -= dy;
      year += 1;
    }
  } else {
    while days < 0 {
      year -= 1;
      days += days_in_year(year);
    }
  }

  let mut month: i64 = 1;
  loop {
    let dm = days_in_month(year, month);
    if days < dm {
      break;
    }
    days -= dm;
    month += 1;
  }

  (year, month, days + 1)
}

/// Convert {y,m,d,h,min,sec} to total seconds since 1900-01-01 00:00:00
pub(crate) fn date_to_absolute_seconds(
  year: i64,
  month: i64,
  day: i64,
  hour: i64,
  minute: i64,
  second: f64,
) -> f64 {
  let total_days = date_to_absolute_days(year, month, day);
  (total_days as f64) * 86400.0
    + (hour as f64) * 3600.0
    + (minute as f64) * 60.0
    + second
}

/// Convert total seconds to {year, month, day, hour, minute, second}
pub(crate) fn absolute_seconds_to_date(
  total_seconds: f64,
) -> (i64, i64, i64, i64, i64, f64) {
  let total_days = (total_seconds / 86400.0).floor() as i64;
  let remaining = total_seconds - (total_days as f64) * 86400.0;

  let (year, month, day) = absolute_days_to_date(total_days);

  let hour = (remaining / 3600.0).floor() as i64;
  let remaining = remaining - (hour as f64) * 3600.0;
  let minute = (remaining / 60.0).floor() as i64;
  let second = remaining - (minute as f64) * 60.0;

  (year, month, day, hour, minute, second)
}

/// Normalize a date list, handling fractional days, overflow months, etc.
/// Input: {y, m, d, h, min, sec} where some values can be fractional or out of range
fn normalize_date(components: &[f64]) -> (i64, i64, i64, i64, i64, f64) {
  let year = if !components.is_empty() {
    components[0] as i64
  } else {
    1900
  };
  let month = if components.len() > 1 {
    components[1]
  } else {
    1.0
  };
  let day = if components.len() > 2 {
    components[2]
  } else {
    1.0
  };
  let hour = if components.len() > 3 {
    components[3]
  } else {
    0.0
  };
  let minute = if components.len() > 4 {
    components[4]
  } else {
    0.0
  };
  let second = if components.len() > 5 {
    components[5]
  } else {
    0.0
  };

  // Convert everything to seconds, then back
  // First handle integer month (can overflow)
  let mut y = year;
  let mut m = month.floor() as i64;
  let frac_month = month - month.floor();

  // Normalize month to 1-12
  while m < 1 {
    m += 12;
    y -= 1;
  }
  while m > 12 {
    m -= 12;
    y += 1;
  }

  // Convert fractional month to days
  let extra_days_from_month = frac_month * (days_in_month(y, m) as f64);

  let total_seconds = date_to_absolute_seconds(
    y,
    m,
    1,
    0,
    0,
    (day - 1.0 + extra_days_from_month) * 86400.0
      + hour * 3600.0
      + minute * 60.0
      + second,
  );

  absolute_seconds_to_date(total_seconds)
}

/// Extract a date component list from an Expr
pub(crate) fn extract_date_components(expr: &Expr) -> Option<Vec<f64>> {
  match expr {
    Expr::List(items) => {
      let mut components = Vec::new();
      for item in items {
        let evaluated = crate::evaluator::evaluate_expr_to_expr(item).ok()?;
        match &evaluated {
          Expr::Integer(n) => components.push(*n as f64),
          Expr::Real(f) => components.push(*f),
          _ => return None,
        }
      }
      Some(components)
    }
    // DateObject[{y, m, d, ...}, ...] — extract the date list from first arg
    Expr::FunctionCall { name, args }
      if name == "DateObject" && !args.is_empty() =>
    {
      extract_date_components(&args[0])
    }
    _ => None,
  }
}

/// Parse a natural language date string like "6 June 1991"
fn parse_date_string(s: &str) -> Option<(i64, i64, i64)> {
  let months = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
  ];

  let parts: Vec<&str> = s.split_whitespace().collect();
  if parts.len() == 3 {
    // Try "Day Month Year" format
    if let Ok(day) = parts[0].parse::<i64>() {
      let month_lower = parts[1].to_lowercase();
      if let Some(month_idx) =
        months.iter().position(|m| m.starts_with(&*month_lower))
        && let Ok(year) = parts[2].parse::<i64>()
      {
        return Some((year, (month_idx + 1) as i64, day));
      }
    }
    // Try "Month Day Year" format
    let month_lower = parts[0].to_lowercase();
    if let Some(month_idx) =
      months.iter().position(|m| m.starts_with(&*month_lower))
    {
      // Remove trailing comma from day if present
      let day_str = parts[1].trim_end_matches(',');
      if let Ok(day) = day_str.parse::<i64>()
        && let Ok(year) = parts[2].parse::<i64>()
      {
        return Some((year, (month_idx + 1) as i64, day));
      }
    }
  }

  // Try common date formats: DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD
  let s = s.trim();

  // Try YYYY-MM-DD
  if s.len() >= 10 && s.chars().nth(4) == Some('-') {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() >= 3
      && let (Ok(y), Ok(m), Ok(d)) = (
        parts[0].parse::<i64>(),
        parts[1].parse::<i64>(),
        parts[2].parse::<i64>(),
      )
      && (1..=12).contains(&m)
      && (1..=31).contains(&d)
    {
      return Some((y, m, d));
    }
  }

  // Try DD/MM/YYYY
  for sep in &['/', '-', '.'] {
    let parts: Vec<&str> = s.split(*sep).collect();
    if parts.len() == 3
      && let (Ok(a), Ok(b), Ok(c)) = (
        parts[0].parse::<i64>(),
        parts[1].parse::<i64>(),
        parts[2].parse::<i64>(),
      )
    {
      // If c > 31 it's likely the year
      if c > 31 {
        // a/b/c = day/month/year
        if a > 12 {
          return Some((c, b, a)); // day/month/year with day > 12
        }
        return Some((c, b, a)); // Assume day/month/year
      }
    }
  }

  None
}

/// Parse a date string using a format specification
fn parse_date_with_format(
  date_str: &str,
  format: &[String],
) -> Option<(i64, i64, i64)> {
  let mut pos = 0;
  let bytes = date_str.as_bytes();
  let mut day: Option<i64> = None;
  let mut month: Option<i64> = None;
  let mut year: Option<i64> = None;

  for spec in format {
    // Skip non-digit separators before reading a numeric field
    if matches!(spec.as_str(), "Day" | "Month" | "Year" | "YearShort") {
      while pos < bytes.len() && !bytes[pos].is_ascii_digit() {
        pos += 1;
      }
    }
    match spec.as_str() {
      "Day" => {
        // Read digits
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
          pos += 1;
        }
        if start < pos {
          day = Some(date_str[start..pos].parse().ok()?);
        }
      }
      "Month" => {
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
          pos += 1;
        }
        if start < pos {
          month = Some(date_str[start..pos].parse().ok()?);
        }
      }
      "Year" => {
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
          pos += 1;
        }
        if start < pos {
          year = Some(date_str[start..pos].parse().ok()?);
        }
      }
      "YearShort" => {
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
          pos += 1;
        }
        if start < pos {
          let short_year: i64 = date_str[start..pos].parse().ok()?;
          year = Some(if short_year >= 69 {
            1900 + short_year
          } else {
            2000 + short_year
          });
        }
      }
      _ => {
        // Literal separator — skip matching characters
        let sep = spec.as_bytes();
        for &b in sep {
          if pos < bytes.len() && bytes[pos] == b {
            pos += 1;
          } else if pos < bytes.len() && !bytes[pos].is_ascii_digit() {
            // Skip any non-digit separator character
            pos += 1;
          }
        }
      }
    }
  }

  Some((year.unwrap_or(1900), month.unwrap_or(1), day.unwrap_or(1)))
}

/// Day of week for a given date (0=Monday, 1=Tuesday, ..., 6=Sunday)
fn day_of_week(year: i64, month: i64, day: i64) -> i64 {
  // Zeller's / Tomohiko Sakamoto's algorithm
  let t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
  let mut y = year;
  if month < 3 {
    y -= 1;
  }
  let dow = (y + y / 4 - y / 100 + y / 400 + t[(month - 1) as usize] + day) % 7;
  // Result: 0=Sunday, 1=Monday, ...
  // Convert to: 0=Monday, 1=Tuesday, ..., 6=Sunday
  (dow + 6) % 7
}

fn day_name(dow: i64) -> &'static str {
  match dow {
    0 => "Monday",
    1 => "Tuesday",
    2 => "Wednesday",
    3 => "Thursday",
    4 => "Friday",
    5 => "Saturday",
    6 => "Sunday",
    _ => "Monday",
  }
}

fn day_name_short(dow: i64) -> &'static str {
  match dow {
    0 => "Mon",
    1 => "Tue",
    2 => "Wed",
    3 => "Thu",
    4 => "Fri",
    5 => "Sat",
    6 => "Sun",
    _ => "Mon",
  }
}

fn month_name(month: i64) -> &'static str {
  match month {
    1 => "January",
    2 => "February",
    3 => "March",
    4 => "April",
    5 => "May",
    6 => "June",
    7 => "July",
    8 => "August",
    9 => "September",
    10 => "October",
    11 => "November",
    12 => "December",
    _ => "January",
  }
}

fn month_name_short(month: i64) -> &'static str {
  match month {
    1 => "Jan",
    2 => "Feb",
    3 => "Mar",
    4 => "Apr",
    5 => "May",
    6 => "Jun",
    7 => "Jul",
    8 => "Aug",
    9 => "Sep",
    10 => "Oct",
    11 => "Nov",
    12 => "Dec",
    _ => "Jan",
  }
}

/// AbsoluteTime[] or AbsoluteTime[date]
pub fn absolute_time_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    // Current time
    use std::time::{SystemTime, UNIX_EPOCH};
    let unix_secs = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_secs_f64();
    // Unix epoch is 1970-01-01; Wolfram epoch is 1900-01-01
    // Difference: 70 years = 2208988800 seconds (accounting for leap years)
    let wolfram_secs = unix_secs + 2208988800.0;
    return Ok(Expr::Real(wolfram_secs));
  }

  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "AbsoluteTime".to_string(),
      args: args.to_vec().into(),
    });
  }

  let arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;

  match &arg {
    Expr::List(_) => {
      if let Some(components) = extract_date_components(&arg) {
        let year = components[0] as i64;
        let month = if components.len() > 1 {
          components[1] as i64
        } else {
          1
        };
        let day = if components.len() > 2 {
          components[2] as i64
        } else {
          1
        };
        let hour = if components.len() > 3 {
          components[3] as i64
        } else {
          0
        };
        let minute = if components.len() > 4 {
          components[4] as i64
        } else {
          0
        };
        let second = if components.len() > 5 {
          components[5]
        } else {
          0.0
        };
        let total =
          date_to_absolute_seconds(year, month, day, hour, minute, second);
        // Integer date components always produce an integer result
        if total == total.floor() {
          Ok(Expr::Integer(total as i128))
        } else {
          Ok(Expr::Real(total))
        }
      } else {
        // Might be a date string with format spec
        // AbsoluteTime[{"string", {"format", ...}}]
        if let Expr::List(items) = &arg
          && items.len() == 2
          && let Expr::String(date_str) = &items[0]
          && let Expr::List(fmt_items) = &items[1]
        {
          let format: Vec<String> = fmt_items
            .iter()
            .filter_map(|item| {
              if let Expr::String(s) = item {
                Some(s.clone())
              } else {
                None
              }
            })
            .collect();
          if let Some((y, m, d)) = parse_date_with_format(date_str, &format) {
            let total = date_to_absolute_seconds(y, m, d, 0, 0, 0.0);
            return Ok(Expr::Real(total));
          }
        }
        Ok(Expr::FunctionCall {
          name: "AbsoluteTime".to_string(),
          args: vec![arg].into(),
        })
      }
    }
    Expr::String(s) => {
      if let Some((y, m, d)) = parse_date_string(s) {
        let total = date_to_absolute_seconds(y, m, d, 0, 0, 0.0);
        Ok(Expr::Integer(total as i128))
      } else {
        Ok(Expr::FunctionCall {
          name: "AbsoluteTime".to_string(),
          args: vec![arg].into(),
        })
      }
    }
    Expr::Integer(n) => {
      // AbsoluteTime[n] returns n (already absolute time)
      Ok(Expr::Integer(*n))
    }
    _ => Ok(Expr::FunctionCall {
      name: "AbsoluteTime".to_string(),
      args: vec![arg].into(),
    }),
  }
}

/// DateList[n] — convert absolute seconds to {y,m,d,h,min,sec.}
/// DateList[{y,m,d,...}] — normalize a date list
/// DateList["string"] — parse a date string
pub fn date_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    // Current date
    return absolute_time_ast(&[]).and_then(|t| date_list_ast(&[t]));
  }

  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "DateList".to_string(),
      args: args.to_vec().into(),
    });
  }

  let arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;

  match &arg {
    Expr::Integer(n) => {
      let (y, m, d, h, min, sec) = absolute_seconds_to_date(*n as f64);
      Ok(make_date_list(y, m, d, h, min, sec))
    }
    Expr::Real(f) => {
      let (y, m, d, h, min, sec) = absolute_seconds_to_date(*f);
      Ok(make_date_list(y, m, d, h, min, sec))
    }
    Expr::List(_) => {
      if let Some(components) = extract_date_components(&arg) {
        let (y, m, d, h, min, sec) = normalize_date(&components);
        Ok(make_date_list(y, m, d, h, min, sec))
      } else {
        // Might be a date string with format spec
        if let Expr::List(items) = &arg
          && items.len() == 2
          && let Expr::String(date_str) = &items[0]
          && let Expr::List(fmt_items) = &items[1]
        {
          let format: Vec<String> = fmt_items
            .iter()
            .filter_map(|item| {
              if let Expr::String(s) = item {
                Some(s.clone())
              } else {
                None
              }
            })
            .collect();
          if let Some((y, m, d)) = parse_date_with_format(date_str, &format) {
            return Ok(make_date_list(y, m, d, 0, 0, 0.0));
          }
        }
        Ok(Expr::FunctionCall {
          name: "DateList".to_string(),
          args: vec![arg].into(),
        })
      }
    }
    Expr::String(s) => {
      if let Some((y, m, d)) = parse_date_string(s) {
        Ok(make_date_list(y, m, d, 0, 0, 0.0))
      } else {
        Ok(Expr::FunctionCall {
          name: "DateList".to_string(),
          args: vec![arg].into(),
        })
      }
    }
    Expr::FunctionCall {
      name,
      args: fn_args,
    } if name == "DateObject" && !fn_args.is_empty() => {
      if let Some(date_list) = resolve_date_to_list(&arg) {
        Ok(date_list)
      } else {
        Ok(Expr::FunctionCall {
          name: "DateList".to_string(),
          args: vec![arg].into(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "DateList".to_string(),
      args: vec![arg].into(),
    }),
  }
}

fn make_date_list(y: i64, m: i64, d: i64, h: i64, min: i64, sec: f64) -> Expr {
  // Keep the raw f64 seconds: wolframscript exposes the floating-point
  // residual from converting the date to absolute seconds and back (e.g.
  // `46.019999980926514` for `{2003, 5, 0.5, 0.1, 0.767}`).
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

/// DatePlus[date, n] — add n days to a date
/// DatePlus[date, {{n1, "unit1"}, ...}] — add with units
pub fn date_plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "DatePlus".to_string(),
      args: args.to_vec().into(),
    });
  }

  let date_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let delta_arg = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  // Track whether the input is a DateObject so we can wrap the result
  let input_is_date_object = matches!(&date_arg, Expr::FunctionCall { name, .. } if name == "DateObject");

  let components = match extract_date_components(&date_arg) {
    Some(c) => c,
    None => {
      return Ok(Expr::FunctionCall {
        name: "DatePlus".to_string(),
        args: vec![date_arg, delta_arg].into(),
      });
    }
  };

  let year = components[0] as i64;
  let month = if components.len() > 1 {
    components[1] as i64
  } else {
    1
  };
  let day = if components.len() > 2 {
    components[2] as i64
  } else {
    1
  };
  let input_len = components.len();

  // Calculate total days to add
  let total_days = match &delta_arg {
    Expr::Integer(n) => *n as i64,
    Expr::Real(f) => *f as i64,
    Expr::List(items) => {
      // A delta is either a single {n, "unit"} pair or a list of such pairs
      // ({{1, "Month"}, {-1, "Day"}}). Normalize to a list of pairs, then
      // apply each in order to a running date so that, e.g.,
      // {{1, "Month"}, {-1, "Day"}} adds a month then subtracts a day (the
      // last day of the month). Month/Year increments adjust calendar fields
      // (clamping the day to the new month length); Day/Week increments shift
      // by absolute days.
      let unit_of = |e: &Expr| -> Option<String> {
        match e {
          Expr::String(s) | Expr::Identifier(s) => Some(s.clone()),
          _ => None,
        }
      };
      let pairs: Vec<(i64, String)> = if items.len() == 2
        && matches!(&items[0], Expr::Integer(_) | Expr::Real(_))
        && unit_of(&items[1]).is_some()
      {
        // Single {n, "unit"} pair.
        let n = match &items[0] {
          Expr::Integer(n) => *n as i64,
          Expr::Real(f) => *f as i64,
          _ => 0,
        };
        vec![(n, unit_of(&items[1]).unwrap())]
      } else {
        // List of {n, "unit"} pairs.
        items
          .iter()
          .filter_map(|item| match item {
            Expr::List(pair) if pair.len() == 2 => {
              let n = match &pair[0] {
                Expr::Integer(n) => *n as i64,
                Expr::Real(f) => *f as i64,
                _ => 0,
              };
              unit_of(&pair[1]).map(|u| (n, u))
            }
            _ => None,
          })
          .collect()
      };

      let mut y = year;
      let mut m = month;
      let mut d = day;
      for (n, unit) in &pairs {
        let n = *n;
        match unit.as_str() {
          "Day" | "Week" => {
            let shift = if unit == "Week" { n * 7 } else { n };
            let abs = date_to_absolute_days(y, m, d) + shift;
            let (ny, nm, nd) = absolute_days_to_date(abs);
            y = ny;
            m = nm;
            d = nd;
          }
          "Month" => {
            m += n;
            while m > 12 {
              m -= 12;
              y += 1;
            }
            while m < 1 {
              m += 12;
              y -= 1;
            }
            d = d.min(days_in_month(y, m));
          }
          "Year" => {
            y += n;
            d = d.min(days_in_month(y, m));
          }
          _ => {}
        }
      }
      return Ok(make_date_result(y, m, d, input_len, input_is_date_object));
    }
    // Quantity[n, "unit"] — treat as a single {n, "unit"} pair
    Expr::FunctionCall {
      name: fname,
      args: qargs,
    } if fname == "Quantity" && qargs.len() == 2 => {
      let n = match &qargs[0] {
        Expr::Integer(n) => *n as i64,
        Expr::Real(f) => *f as i64,
        _ => 0,
      };
      let unit = match &qargs[1] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => String::new(),
      };
      match unit.as_str() {
        "Day" | "Days" | "days" | "day" => n,
        "Week" | "Weeks" | "weeks" | "week" => n * 7,
        "Month" | "Months" | "months" | "month" => {
          let mut new_month = month + n;
          let mut new_year = year;
          while new_month > 12 {
            new_month -= 12;
            new_year += 1;
          }
          while new_month < 1 {
            new_month += 12;
            new_year -= 1;
          }
          let new_day = day.min(days_in_month(new_year, new_month));
          return Ok(make_date_result(
            new_year,
            new_month,
            new_day,
            input_len,
            input_is_date_object,
          ));
        }
        "Year" | "Years" | "years" | "year" => {
          let new_year = year + n;
          let new_day = day.min(days_in_month(new_year, month));
          return Ok(make_date_result(
            new_year,
            month,
            new_day,
            input_len,
            input_is_date_object,
          ));
        }
        _ => n,
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DatePlus".to_string(),
        args: vec![date_arg, delta_arg].into(),
      });
    }
  };

  let abs_days = date_to_absolute_days(year, month, day) + total_days;
  let (ny, nm, nd) = absolute_days_to_date(abs_days);
  Ok(make_date_result(
    ny,
    nm,
    nd,
    input_len,
    input_is_date_object,
  ))
}

fn make_date_result(
  y: i64,
  m: i64,
  d: i64,
  input_len: usize,
  as_date_object: bool,
) -> Expr {
  if as_date_object {
    Expr::FunctionCall {
      name: "DateObject".to_string(),
      args: vec![
        Expr::List(
          vec![
            Expr::Integer(y as i128),
            Expr::Integer(m as i128),
            Expr::Integer(d as i128),
          ]
          .into(),
        ),
        Expr::String("Day".to_string()),
      ]
      .into(),
    }
  } else if input_len <= 3 {
    Expr::List(
      vec![
        Expr::Integer(y as i128),
        Expr::Integer(m as i128),
        Expr::Integer(d as i128),
      ]
      .into(),
    )
  } else {
    make_date_list(y, m, d, 0, 0, 0.0)
  }
}

/// DateDifference[date1, date2] — difference in days
/// DateDifference[date1, date2, "unit"] — difference in given unit
pub fn date_difference_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "DateDifference".to_string(),
      args: args.to_vec().into(),
    });
  }

  let date1 = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let date2 = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  let c1 = match extract_date_components(&date1) {
    Some(c) => c,
    None => {
      return Ok(Expr::FunctionCall {
        name: "DateDifference".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let c2 = match extract_date_components(&date2) {
    Some(c) => c,
    None => {
      return Ok(Expr::FunctionCall {
        name: "DateDifference".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let s1 = date_to_absolute_seconds(
    c1[0] as i64,
    if c1.len() > 1 { c1[1] as i64 } else { 1 },
    if c1.len() > 2 { c1[2] as i64 } else { 1 },
    if c1.len() > 3 { c1[3] as i64 } else { 0 },
    if c1.len() > 4 { c1[4] as i64 } else { 0 },
    if c1.len() > 5 { c1[5] } else { 0.0 },
  );
  let s2 = date_to_absolute_seconds(
    c2[0] as i64,
    if c2.len() > 1 { c2[1] as i64 } else { 1 },
    if c2.len() > 2 { c2[2] as i64 } else { 1 },
    if c2.len() > 3 { c2[3] as i64 } else { 0 },
    if c2.len() > 4 { c2[4] as i64 } else { 0 },
    if c2.len() > 5 { c2[5] } else { 0.0 },
  );

  let diff_seconds = s2 - s1;
  let diff_days = diff_seconds / 86400.0;

  if args.len() == 2 {
    // Return Quantity[n, "Days"]
    let n = if diff_days == diff_days.floor() {
      Expr::Integer(diff_days as i128)
    } else {
      Expr::Real(diff_days)
    };
    return Ok(Expr::FunctionCall {
      name: "Quantity".to_string(),
      args: vec![n, Expr::String("Days".to_string())].into(),
    });
  }

  // With unit specification
  let unit_arg = crate::evaluator::evaluate_expr_to_expr(&args[2])?;
  let unit = match &unit_arg {
    Expr::String(s) => s.clone(),
    Expr::Identifier(s) => s.clone(),
    Expr::List(_) => {
      // Multi-unit format like {"Week", "Day"}
      return date_difference_multi_unit(&c1, &c2, &unit_arg);
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DateDifference".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let (value, unit_name) = match unit.as_str() {
    "Year" => {
      // Wolfram computes: complete_years + remaining_days / 365
      let y1 = c1[0] as i64;
      let m1 = if c1.len() > 1 { c1[1] as i64 } else { 1 };
      let d1 = if c1.len() > 2 { c1[2] as i64 } else { 1 };
      let y2 = c2[0] as i64;
      let m2 = if c2.len() > 1 { c2[1] as i64 } else { 1 };
      let d2 = if c2.len() > 2 { c2[2] as i64 } else { 1 };
      let mut complete_years = y2 - y1;
      // Check if the anniversary hasn't happened yet
      if (m2, d2) < (m1, d1) {
        complete_years -= 1;
      }
      // Calculate remaining days after complete years
      let anniversary_abs = date_to_absolute_days(y1 + complete_years, m1, d1);
      let end_abs = date_to_absolute_days(y2, m2, d2);
      let remaining_days = end_abs - anniversary_abs;
      (
        complete_years as f64 + remaining_days as f64 / 365.0,
        "Years",
      )
    }
    "Month" => (diff_days / 30.436875, "Months"),
    "Week" => (diff_days / 7.0, "Weeks"),
    "Day" => (diff_days, "Days"),
    "Hour" => (diff_seconds / 3600.0, "Hours"),
    "Minute" => (diff_seconds / 60.0, "Minutes"),
    "Second" => (diff_seconds, "Seconds"),
    _ => (diff_days, "Days"),
  };

  let n = if value == value.floor() {
    Expr::Integer(value as i128)
  } else {
    Expr::Real(value)
  };
  Ok(Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![n, Expr::String(unit_name.to_string())].into(),
  })
}

fn date_difference_multi_unit(
  c1: &[f64],
  c2: &[f64],
  units: &Expr,
) -> Result<Expr, InterpreterError> {
  // Multi-unit decomposition like {"Week", "Day"} → MixedMagnitude[{9, 6}].
  // Walk the unit list largest-to-smallest, taking the integer part of each
  // bucket and passing the remainder down. Returns the unevaluated form for
  // unit lists containing "Year" or "Month" — those need calendar-based
  // logic that this simple linear conversion can't produce.
  let unit_strs: Vec<String> = match units {
    Expr::List(items) => {
      let mut v = Vec::with_capacity(items.len());
      for item in items {
        match item {
          Expr::String(s) => v.push(s.clone()),
          Expr::Identifier(s) => v.push(s.clone()),
          _ => return Ok(unevaluated(c1, c2, units)),
        }
      }
      v
    }
    _ => return Ok(unevaluated(c1, c2, units)),
  };
  if unit_strs.is_empty() {
    return Ok(unevaluated(c1, c2, units));
  }
  // Bail out for calendar-aware units that don't reduce to fixed seconds.
  for u in &unit_strs {
    if u == "Year" || u == "Month" {
      return Ok(unevaluated(c1, c2, units));
    }
  }
  let s1 = date_to_absolute_seconds(
    c1[0] as i64,
    if c1.len() > 1 { c1[1] as i64 } else { 1 },
    if c1.len() > 2 { c1[2] as i64 } else { 1 },
    if c1.len() > 3 { c1[3] as i64 } else { 0 },
    if c1.len() > 4 { c1[4] as i64 } else { 0 },
    if c1.len() > 5 { c1[5] } else { 0.0 },
  );
  let s2 = date_to_absolute_seconds(
    c2[0] as i64,
    if c2.len() > 1 { c2[1] as i64 } else { 1 },
    if c2.len() > 2 { c2[2] as i64 } else { 1 },
    if c2.len() > 3 { c2[3] as i64 } else { 0 },
    if c2.len() > 4 { c2[4] as i64 } else { 0 },
    if c2.len() > 5 { c2[5] } else { 0.0 },
  );
  let mut remaining = s2 - s1;
  let last_idx = unit_strs.len() - 1;
  let mut magnitudes: Vec<Expr> = Vec::with_capacity(unit_strs.len());
  let mut plurals: Vec<Expr> = Vec::with_capacity(unit_strs.len());
  for (i, u) in unit_strs.iter().enumerate() {
    let secs_per_unit = match u.as_str() {
      "Week" => 7.0 * 86400.0,
      "Day" => 86400.0,
      "Hour" => 3600.0,
      "Minute" => 60.0,
      "Second" => 1.0,
      _ => return Ok(unevaluated(c1, c2, units)),
    };
    let plural = match u.as_str() {
      "Week" => "Weeks",
      "Day" => "Days",
      "Hour" => "Hours",
      "Minute" => "Minutes",
      "Second" => "Seconds",
      _ => unreachable!(),
    };
    plurals.push(Expr::String(plural.to_string()));
    if i == last_idx {
      // Final bucket keeps the leftover (may be fractional).
      let value = remaining / secs_per_unit;
      magnitudes.push(if value == value.floor() {
        Expr::Integer(value as i128)
      } else {
        Expr::Real(value)
      });
    } else {
      let count = (remaining / secs_per_unit).trunc();
      remaining -= count * secs_per_unit;
      magnitudes.push(Expr::Integer(count as i128));
    }
  }
  Ok(Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "MixedMagnitude".to_string(),
        args: vec![Expr::List(magnitudes.into())].into(),
      },
      Expr::FunctionCall {
        name: "MixedUnit".to_string(),
        args: vec![Expr::List(plurals.into())].into(),
      },
    ]
    .into(),
  })
}

fn unevaluated(c1: &[f64], c2: &[f64], units: &Expr) -> Expr {
  let to_list = |c: &[f64]| -> Expr {
    Expr::List(
      c.iter()
        .map(|&v| {
          if v == v.floor() {
            Expr::Integer(v as i128)
          } else {
            Expr::Real(v)
          }
        })
        .collect(),
    )
  };
  Expr::FunctionCall {
    name: "DateDifference".to_string(),
    args: vec![to_list(c1), to_list(c2), units.clone()].into(),
  }
}

/// DateString[date, format] — format a date as a string
pub fn date_string_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty()
    || (args.len() == 1
      && matches!(&args[0], Expr::Identifier(s) if s == "Now"))
  {
    // Current date/time
    let abs_time = absolute_time_ast(&[])?;
    let date_list = date_list_ast(&[abs_time])?;
    if let Some(c) = extract_date_components(&date_list) {
      let (y, m, d, h, min, sec) = normalize_date(&c);
      let dow = day_of_week(y, m, d);
      return Ok(Expr::String(format!(
        "{} {} {} {} {:02}:{:02}:{:02}",
        day_name_short(dow),
        d,
        month_name_short(m),
        y,
        h,
        min,
        sec as i64
      )));
    }
  }

  let date_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;

  // Handle Now
  if matches!(&date_arg, Expr::Identifier(s) if s == "Now") {
    return date_string_ast(&[]);
  }

  // DateString["string"] with no format spec returns the string as-is (Wolfram behavior)
  if args.len() == 1
    && let Expr::String(s) = &date_arg
  {
    return Ok(Expr::String(s.clone()));
  }

  // Extract date components — handle DateObject[{y,m,d,...}, ...] by extracting first arg
  // Also detect granularity from DateObject (e.g. "Day", "Instant", "Hour", etc.)
  let mut granularity: Option<&str> = None;
  let date_expr = if let Expr::FunctionCall { name, args: dargs } = &date_arg {
    if name == "DateObject" && !dargs.is_empty() {
      // Check for granularity in the second argument
      if dargs.len() >= 2 {
        if let Expr::String(g) = &dargs[1] {
          granularity = Some(match g.as_str() {
            "Day" => "Day",
            "Month" => "Month",
            "Year" => "Year",
            "Hour" => "Hour",
            "Minute" => "Minute",
            "Second" | "Instant" => "Second",
            _ => "Day",
          });
        } else if let Expr::Identifier(g) = &dargs[1] {
          granularity = Some(match g.as_str() {
            "Day" => "Day",
            "Month" => "Month",
            "Year" => "Year",
            "Hour" => "Hour",
            "Minute" => "Minute",
            "Second" | "Instant" => "Second",
            _ => "Day",
          });
        }
      }
      &dargs[0]
    } else {
      &date_arg
    }
  } else {
    &date_arg
  };

  let components = match extract_date_components(date_expr) {
    Some(c) => c,
    None => {
      // Try parsing a date string like "2025-09-24" or "6 June 1991"
      if let Expr::String(s) = date_expr {
        if let Some((y, m, d)) = parse_date_string(s) {
          vec![y as f64, m as f64, d as f64, 0.0, 0.0, 0.0]
        } else {
          return Ok(Expr::FunctionCall {
            name: "DateString".to_string(),
            args: args.to_vec().into(),
          });
        }
      } else {
        return Ok(Expr::FunctionCall {
          name: "DateString".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  };

  // Determine whether to include time in default format output.
  // Only omit time when DateObject has explicit Day/Month/Year granularity.
  // For plain lists or DateObjects with time granularity, always include time.
  let has_time = match granularity {
    Some("Day" | "Month" | "Year") => false,
    Some(_) => true,
    None => true, // plain list or no DateObject: always include time (Wolfram behavior)
  };

  let (y, m, d, h, min, sec) = normalize_date(&components);

  if args.len() == 1 {
    let dow = day_of_week(y, m, d);
    if has_time {
      // Format with time: "DayNameShort DD MonthNameShort YYYY HH:MM:SS"
      return Ok(Expr::String(format!(
        "{} {} {} {} {:02}:{:02}:{:02}",
        day_name_short(dow),
        d,
        month_name_short(m),
        y,
        h,
        min,
        sec as i64
      )));
    } else {
      // Format without time: "DayNameShort DD MonthNameShort YYYY"
      return Ok(Expr::String(format!(
        "{} {} {} {}",
        day_name_short(dow),
        d,
        month_name_short(m),
        y,
      )));
    }
  }

  let fmt_arg = crate::evaluator::evaluate_expr_to_expr(&args[1])?;
  let format_specs = match &fmt_arg {
    Expr::List(items) => items.clone(),
    Expr::String(s) => {
      // Named date format specifications — handle directly for correct padding
      let dow = day_of_week(y, m, d);
      match s.as_str() {
        "ISODateTime" => {
          return Ok(Expr::String(format!(
            "{}-{:02}-{:02}T{:02}:{:02}:{:02}",
            y, m, d, h, min, sec as i64
          )));
        }
        "ISODate" => {
          return Ok(Expr::String(format!("{}-{:02}-{:02}", y, m, d)));
        }
        "DateTime" => {
          return Ok(Expr::String(format!(
            "{} {} {} {} {:02}:{:02}:{:02}",
            day_name(dow),
            d,
            month_name(m),
            y,
            h,
            min,
            sec as i64
          )));
        }
        "DateTimeShort" => {
          return Ok(Expr::String(format!(
            "{} {} {} {} {:02}:{:02}:{:02}",
            day_name_short(dow),
            d,
            month_name_short(m),
            y,
            h,
            min,
            sec as i64
          )));
        }
        "Date" => {
          return Ok(Expr::String(format!(
            "{} {} {} {}",
            day_name(dow),
            d,
            month_name(m),
            y
          )));
        }
        "DateShort" => {
          return Ok(Expr::String(format!(
            "{} {} {} {}",
            day_name_short(dow),
            d,
            month_name_short(m),
            y
          )));
        }
        "Time" => {
          return Ok(Expr::String(format!(
            "{:02}:{:02}:{:02}",
            h, min, sec as i64
          )));
        }
        // Single format element as a string (e.g. DateString[Now, "Year"])
        _ => vec![Expr::String(s.clone())],
      }
      .into()
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DateString".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut result = String::new();
  for spec in &format_specs {
    if let Expr::String(s) = spec {
      match s.as_str() {
        "Year" => result.push_str(&format!("{}", y)),
        "YearShort" => result.push_str(&format!("{:02}", y % 100)),
        "Month" => result.push_str(&format!("{:02}", m)),
        "MonthName" => result.push_str(month_name(m)),
        "MonthNameShort" => result.push_str(month_name_short(m)),
        "Day" => result.push_str(&format!("{:02}", d)),
        "DayName" => result.push_str(day_name(day_of_week(y, m, d))),
        "DayNameShort" => result.push_str(day_name_short(day_of_week(y, m, d))),
        "Hour" => result.push_str(&format!("{:02}", h)),
        "Minute" => result.push_str(&format!("{:02}", min)),
        "Second" => result.push_str(&format!("{:02}", sec as i64)),
        _ => result.push_str(s), // literal separator
      }
    }
  }

  Ok(Expr::String(result))
}

/// DayName[{year, month, day}] - return the name of the day of the week
/// DayName[DateObject[{year, month, day}]] - also accepted
pub fn day_name_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "DayName".to_string(),
      args: args.to_vec().into(),
    });
  }

  let arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;

  // Handle DateObject[{y,m,d,...}]
  let date_expr = match &arg {
    Expr::FunctionCall { name, args: dargs }
      if name == "DateObject" && !dargs.is_empty() =>
    {
      &dargs[0]
    }
    _ => &arg,
  };

  if let Some(components) = extract_date_components(date_expr)
    && components.len() >= 3
  {
    let year = components[0] as i64;
    let month = components[1] as i64;
    let day = components[2] as i64;
    let dow = day_of_week(year, month, day);
    return Ok(Expr::Identifier(day_name(dow).to_string()));
  }

  Ok(Expr::FunctionCall {
    name: "DayName".to_string(),
    args: args.to_vec().into(),
  })
}

/// Day of the year (1–366) for a Gregorian date.
fn day_of_year(year: i64, month: i64, day: i64) -> i64 {
  let mut doy = day;
  for m in 1..month {
    doy += days_in_month(year, m);
  }
  doy
}

/// Number of ISO-8601 weeks in a year (52 or 53).
fn iso_weeks_in_year(year: i64) -> i64 {
  let p = |y: i64| ((y + y / 4 - y / 100 + y / 400) % 7 + 7) % 7;
  if p(year) == 4 || p(year - 1) == 3 {
    53
  } else {
    52
  }
}

/// ISO-8601 week number of a Gregorian date.
fn iso_week(year: i64, month: i64, day: i64) -> i64 {
  let doy = day_of_year(year, month, day);
  // ISO weekday: Monday = 1 … Sunday = 7 (day_of_week returns Monday = 0).
  let iso_weekday = day_of_week(year, month, day) + 1;
  let week = (doy - iso_weekday + 10) / 7;
  if week < 1 {
    iso_weeks_in_year(year - 1)
  } else if week > iso_weeks_in_year(year) {
    1
  } else {
    week
  }
}

/// ISO-8601 week-numbering year of a Gregorian date. Differs from the calendar
/// year near January 1 / December 31 (e.g. 2023-01-01 belongs to ISO year
/// 2022, week 52).
fn iso_week_year(year: i64, month: i64, day: i64) -> i64 {
  let doy = day_of_year(year, month, day);
  let iso_weekday = day_of_week(year, month, day) + 1;
  let week = (doy - iso_weekday + 10) / 7;
  if week < 1 {
    year - 1
  } else if week > iso_weeks_in_year(year) {
    year + 1
  } else {
    year
  }
}

/// DateValue[date, property] — a named component of a date.
pub fn date_value_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "DateValue".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    return unevaluated();
  }
  let Some(date_list) = resolve_date_to_list(&args[0]) else {
    return unevaluated();
  };
  let Some(comps) = extract_date_components(&date_list) else {
    return unevaluated();
  };
  if comps.len() < 3 {
    return unevaluated();
  }
  let y = comps[0] as i64;
  let mo = comps[1] as i64;
  let d = comps[2] as i64;
  let h = comps.get(3).map(|v| *v as i64).unwrap_or(0);
  let mi = comps.get(4).map(|v| *v as i64).unwrap_or(0);
  let sec = comps.get(5).map(|v| *v as i64).unwrap_or(0);

  let property = |prop: &str| -> Option<Expr> {
    let int = |n: i64| Expr::Integer(n as i128);
    Some(match prop {
      "Year" => int(y),
      "Month" => int(mo),
      "Day" => int(d),
      "Hour" => int(h),
      "Minute" => int(mi),
      "Second" => int(sec),
      "DayName" => {
        Expr::Identifier(day_name(day_of_week(y, mo, d)).to_string())
      }
      "MonthName" => Expr::Identifier(month_name(mo).to_string()),
      "DayNameShort" => {
        Expr::String(day_name_short(day_of_week(y, mo, d)).to_string())
      }
      "Quarter" => int((mo - 1) / 3 + 1),
      "DayOfYear" => int(day_of_year(y, mo, d)),
      "ISOWeekDay" => int(day_of_week(y, mo, d) + 1),
      "Week" => int(iso_week(y, mo, d)),
      "ISOWeek" => int(iso_week(y, mo, d)),
      "ISOWeekYear" => int(iso_week_year(y, mo, d)),
      _ => return None,
    })
  };

  match &args[1] {
    Expr::String(prop) => property(prop).map_or_else(unevaluated, Ok),
    Expr::List(props) => {
      let mut values = Vec::with_capacity(props.len());
      for p in props.iter() {
        match p {
          Expr::String(s) => match property(s) {
            Some(v) => values.push(v),
            None => return unevaluated(),
          },
          _ => return unevaluated(),
        }
      }
      Ok(Expr::List(values.into()))
    }
    _ => unevaluated(),
  }
}

/// DayMatchQ[date, daytype] — whether a date matches a day-of-week
/// specification. Supports the weekday names (as symbols or strings) and the
/// string categories "Weekday"/"Weekend". Calendar-dependent specs
/// ("BusinessDay", "Holiday"), bare category symbols, and lists are left
/// unevaluated, matching Wolfram.
pub fn day_match_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "DayMatchQ".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    return unevaluated();
  }
  let Some(date_list) = resolve_date_to_list(&args[0]) else {
    return unevaluated();
  };
  let Some(comps) = extract_date_components(&date_list) else {
    return unevaluated();
  };
  if comps.len() < 3 {
    return unevaluated();
  }
  // day_of_week: Monday = 0 … Sunday = 6.
  let dow = day_of_week(comps[0] as i64, comps[1] as i64, comps[2] as i64);
  let weekday_name = day_name(dow);

  const DAY_NAMES: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
  ];

  let boolean = |b: bool| {
    Ok(Expr::Identifier(
      if b { "True" } else { "False" }.to_string(),
    ))
  };

  match &args[1] {
    // A weekday name (symbol or string) matches that day of the week.
    Expr::Identifier(s) | Expr::String(s)
      if DAY_NAMES.contains(&s.as_str()) =>
    {
      boolean(s == weekday_name)
    }
    // String-only categories.
    Expr::String(s) if s == "Weekend" => boolean(dow >= 5),
    Expr::String(s) if s == "Weekday" => boolean(dow <= 4),
    _ => unevaluated(),
  }
}

/// The seven weekday names, indexed to match `day_of_week` (Monday = 0).
const WEEKDAY_NAMES: [&str; 7] = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
];

/// DayRound[date, daytype] — round `date` to the nearest day of the given
/// weekday using the default next-day convention (the next occurrence on or
/// after `date`). Returns a `DateObject[…, Day]`. Only the weekday-name form
/// (symbol or string) is supported; calendar categories and the 3-argument
/// rounding form are left unevaluated, matching Wolfram's package-only forms.
pub fn day_round_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "DayRound".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 2 {
    return unevaluated();
  }
  let target_dow = match &args[1] {
    Expr::Identifier(s) | Expr::String(s) => {
      match WEEKDAY_NAMES.iter().position(|n| n == s) {
        Some(i) => i as i64,
        None => return unevaluated(),
      }
    }
    _ => return unevaluated(),
  };
  let Some(date_list) = resolve_date_to_list(&args[0]) else {
    return unevaluated();
  };
  let Some(comps) = extract_date_components(&date_list) else {
    return unevaluated();
  };
  if comps.len() < 3 {
    return unevaluated();
  }
  let (y, m, d) = (comps[0] as i64, comps[1] as i64, comps[2] as i64);
  let current_dow = day_of_week(y, m, d);
  let days_forward = (target_dow - current_dow).rem_euclid(7);
  let (ny, nm, nd) =
    absolute_days_to_date(date_to_absolute_days(y, m, d) + days_forward);

  // Build DateObject[{ny, nm, nd}] and let it normalize to Day granularity.
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![Expr::List(
      vec![
        Expr::Integer(ny as i128),
        Expr::Integer(nm as i128),
        Expr::Integer(nd as i128),
      ]
      .into(),
    )]
    .into(),
  })
}

/// Resolve a date expression (date list, date string, DateObject) to a
/// normalized {year, month, day, hour, min, sec} Expr::List.
/// Returns None if the expression cannot be interpreted as a date.
pub fn resolve_date_to_list(expr: &Expr) -> Option<Expr> {
  let evaluated = crate::evaluator::evaluate_expr_to_expr(expr).ok()?;
  match &evaluated {
    Expr::List(_) => {
      let components = extract_date_components(&evaluated)?;
      let (y, m, d, h, min, sec) = normalize_date(&components);
      Some(make_date_list(y, m, d, h, min, sec))
    }
    Expr::String(s) => {
      let (y, m, d) = parse_date_string(s)?;
      Some(make_date_list(y, m, d, 0, 0, 0.0))
    }
    Expr::FunctionCall { name, args }
      if name == "DateObject" && !args.is_empty() =>
    {
      // DateObject[{y, m, d, ...}] — extract the date list
      resolve_date_to_list(&args[0])
    }
    _ => None,
  }
}

// ─── DayPlus ────────────────────────────────────────────────────────

/// DayPlus[date, n] - adds n days to a date and returns DateObject
/// DayPlus[date, n, "BusinessDay"] - adds n business days (Mon-Fri)
pub fn day_plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "DayPlus".to_string(),
      args: args.to_vec().into(),
    });
  }

  let date_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let n_arg = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  let components = match extract_date_components(&date_arg) {
    Some(c) => c,
    None => {
      return Ok(Expr::FunctionCall {
        name: "DayPlus".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let year = components[0] as i64;
  let month = if components.len() > 1 {
    components[1] as i64
  } else {
    1
  };
  let day = if components.len() > 2 {
    components[2] as i64
  } else {
    1
  };

  let n = match &n_arg {
    Expr::Integer(n) => *n as i64,
    Expr::Real(f) => *f as i64,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DayPlus".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Check for "BusinessDay" mode
  let is_business = if args.len() == 3 {
    matches!(&args[2], Expr::String(s) if s == "BusinessDay")
  } else {
    false
  };

  let (ny, nm, nd) = if is_business {
    // Add business days (skip weekends)
    let mut abs = date_to_absolute_days(year, month, day);
    let step = if n >= 0 { 1 } else { -1 };
    let mut remaining = n.abs();
    while remaining > 0 {
      abs += step;
      let (y, m, d) = absolute_days_to_date(abs);
      let dow = day_of_week(y, m, d);
      // Skip Saturday (5) and Sunday (6) — dow uses 0=Monday convention
      if dow != 5 && dow != 6 {
        remaining -= 1;
      }
    }
    absolute_days_to_date(abs)
  } else {
    let abs = date_to_absolute_days(year, month, day) + n;
    absolute_days_to_date(abs)
  };

  // Return DateObject[{y, m, d}, Day]
  Ok(Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![
      Expr::List(
        vec![
          Expr::Integer(ny as i128),
          Expr::Integer(nm as i128),
          Expr::Integer(nd as i128),
        ]
        .into(),
      ),
      Expr::String("Day".to_string()),
    ]
    .into(),
  })
}

// ─── DayRange ───────────────────────────────────────────────────────

/// DayRange[date1, date2] — list of DateObject "Day" values from date1 to date2.
/// If date1 > date2 the order is normalized so the range is always ascending.
///
/// Note: the three-argument day-type form (e.g. "BusinessDay") is intentionally
/// left unevaluated. wolframscript's day-type filtering is holiday-aware (it
/// consults CalendarData, so e.g. New Year's Day is excluded from
/// "BusinessDay"), which cannot be reproduced without that data. Producing a
/// non-holiday-aware result would diverge from wolframscript.
/// Map a weekday symbol name to `day_of_week`'s index (0=Monday … 6=Sunday).
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

pub fn day_range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "DayRange".to_string(),
      args: args.to_vec().into(),
    });
  }

  // Optional 3rd argument: a weekday (or list of weekdays) to keep, e.g.
  // `DayRange[start, end, Sunday]`.
  let weekday_filter: Option<Vec<i64>> = if args.len() == 3 {
    let spec = crate::evaluator::evaluate_expr_to_expr(&args[2])?;
    let names: Vec<&Expr> = match &spec {
      Expr::List(items) => items.iter().collect(),
      other => vec![other],
    };
    let mut idxs = Vec::new();
    for e in names {
      match e {
        Expr::Identifier(n) | Expr::Constant(n) => match weekday_index(n) {
          Some(i) => idxs.push(i),
          None => {
            return Ok(Expr::FunctionCall {
              name: "DayRange".to_string(),
              args: args.to_vec().into(),
            });
          }
        },
        _ => {
          return Ok(Expr::FunctionCall {
            name: "DayRange".to_string(),
            args: args.to_vec().into(),
          });
        }
      }
    }
    Some(idxs)
  } else {
    None
  };

  let start_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let end_arg = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  // Accept partial date specs: `{y}` and `{y, m}` default the missing month
  // and day to 1, matching wolframscript (`DayRange[{2013, 1}, …]`).
  let pad_date = |c: Vec<f64>| -> Option<Vec<f64>> {
    if c.is_empty() {
      return None;
    }
    let mut out = c;
    while out.len() < 3 {
      out.push(1.0);
    }
    Some(out)
  };
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "DayRange".to_string(),
      args: args.to_vec().into(),
    })
  };

  let start_comp = match extract_date_components(&start_arg).and_then(pad_date)
  {
    Some(c) => c,
    None => return unevaluated(),
  };
  let end_comp = match extract_date_components(&end_arg).and_then(pad_date) {
    Some(c) => c,
    None => return unevaluated(),
  };

  let mut abs_start = date_to_absolute_days(
    start_comp[0] as i64,
    start_comp[1] as i64,
    start_comp[2] as i64,
  );
  let mut abs_end = date_to_absolute_days(
    end_comp[0] as i64,
    end_comp[1] as i64,
    end_comp[2] as i64,
  );

  // Normalize so the range is always ascending.
  if abs_start > abs_end {
    std::mem::swap(&mut abs_start, &mut abs_end);
  }

  let mut result = Vec::new();
  let mut abs = abs_start;
  while abs <= abs_end {
    let (y, m, d) = absolute_days_to_date(abs);
    let keep = match &weekday_filter {
      Some(idxs) => idxs.contains(&day_of_week(y, m, d)),
      None => true,
    };
    if keep {
      result.push(Expr::FunctionCall {
        name: "DateObject".to_string(),
        args: vec![
          Expr::List(
            vec![
              Expr::Integer(y as i128),
              Expr::Integer(m as i128),
              Expr::Integer(d as i128),
            ]
            .into(),
          ),
          Expr::String("Day".to_string()),
        ]
        .into(),
      });
    }
    abs += 1;
  }

  Ok(Expr::List(result.into()))
}

/// JulianDate[] / JulianDate[{y, m, d, h, min, s}] - Julian date of the
/// current instant or of a proleptic-Gregorian date list. Wolfram has
/// no input year zero: 0 and -1 both denote 1 BC (astronomical year 0),
/// so negative years shift by one. Date lists are taken as-is with no
/// time-zone adjustment, matching wolframscript.
pub fn julian_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "JulianDate".to_string(),
    args: args.to_vec().into(),
  };

  if args.is_empty() {
    #[cfg(not(target_arch = "wasm32"))]
    {
      use std::time::{SystemTime, UNIX_EPOCH};
      let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
      return Ok(Expr::Real(2440587.5 + unix / 86400.0));
    }
    #[cfg(target_arch = "wasm32")]
    return Ok(unevaluated(args));
  }
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() && items.len() <= 6 => items,
    _ => return Ok(unevaluated(args)),
  };
  let mut parts: Vec<f64> = Vec::with_capacity(6);
  for item in items.iter() {
    match item {
      Expr::Integer(v) => parts.push(*v as f64),
      Expr::Real(v) => parts.push(*v),
      Expr::FunctionCall { name, args } if name == "Rational" => {
        match (&args[0], &args[1]) {
          (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
            parts.push(*p as f64 / *q as f64)
          }
          _ => return Ok(unevaluated(&[Expr::List(items.clone())])),
        }
      }
      _ => return Ok(unevaluated(&[Expr::List(items.clone())])),
    }
  }
  while parts.len() < 3 {
    parts.push(1.0);
  }
  while parts.len() < 6 {
    parts.push(0.0);
  }

  let y_input = parts[0] as i64;
  // No input year zero: 0 and -1 are both 1 BC (astronomical 0)
  let y = if y_input < 0 { y_input + 1 } else { y_input };
  let m = parts[1] as i64;
  let d = parts[2] as i64;

  // Proleptic Gregorian JDN (floor divisions for negative years)
  let a = (14 - m).div_euclid(12);
  let y2 = y + 4800 - a;
  let m2 = m + 12 * a - 3;
  let jdn = d + (153 * m2 + 2).div_euclid(5) + 365 * y2 + y2.div_euclid(4)
    - y2.div_euclid(100)
    + y2.div_euclid(400)
    - 32045;

  // Single division keeps the float rounding identical to wolframscript
  let seconds = (parts[3] - 12.0) * 3600.0 + parts[4] * 60.0 + parts[5];
  Ok(Expr::Real(jdn as f64 + seconds / 86400.0))
}
