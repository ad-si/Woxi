//! AST-native date/time functions.

use crate::InterpreterError;
use crate::syntax::{Expr, unevaluated};

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

pub(crate) fn days_in_month(year: i64, month: i64) -> i64 {
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

/// Normalize the components of a DateObject date list the way wolframscript
/// does: out-of-range values carry into the next-larger unit, cascading from
/// seconds up through minutes, hours, days, and months (so
/// `{2026, 13, 45}` becomes `{2027, 2, 14}` and `{2026, 1, 0}` rolls back to
/// `{2025, 12, 31}`). Only the final component (the seconds of a 6-element
/// list) may be a Real; it keeps its type. Returns `None` — leaving the list
/// untouched — when the shape or component types don't qualify.
pub(crate) fn normalize_date_components(items: &[Expr]) -> Option<Vec<Expr>> {
  if items.is_empty() || items.len() > 6 {
    return None;
  }
  let as_i64 = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => i64::try_from(*n).ok(),
      _ => None,
    }
  };
  let mut y = as_i64(&items[0])?;
  let mut mo = items.get(1).map(&as_i64).unwrap_or(Some(1))?;
  let mut d = items.get(2).map(&as_i64).unwrap_or(Some(1))?;
  let mut h = items.get(3).map(&as_i64).unwrap_or(Some(0))?;
  let mut mi = items.get(4).map(&as_i64).unwrap_or(Some(0))?;
  let (mut s, s_is_real) = match items.get(5) {
    None => (0.0, false),
    Some(Expr::Integer(n)) => (i64::try_from(*n).ok()? as f64, false),
    Some(Expr::Real(v)) if v.is_finite() => (*v, true),
    Some(_) => return None,
  };

  // Carry each unit into the next-larger one, smallest first.
  let s_carry = (s / 60.0).floor() as i64;
  s -= 60.0 * s_carry as f64;
  mi += s_carry;
  h += mi.div_euclid(60);
  mi = mi.rem_euclid(60);
  d += h.div_euclid(24);
  h = h.rem_euclid(24);
  // Months before days: the month determines each month's length.
  y += (mo - 1).div_euclid(12);
  mo = (mo - 1).rem_euclid(12) + 1;
  while d > days_in_month(y, mo) {
    d -= days_in_month(y, mo);
    mo += 1;
    if mo > 12 {
      mo = 1;
      y += 1;
    }
  }
  while d < 1 {
    mo -= 1;
    if mo < 1 {
      mo = 12;
      y -= 1;
    }
    d += days_in_month(y, mo);
  }

  let ints = [y, mo, d, h, mi];
  let mut out: Vec<Expr> = ints[..items.len().min(5)]
    .iter()
    .map(|&n| Expr::Integer(n as i128))
    .collect();
  if items.len() == 6 {
    out.push(if s_is_real {
      Expr::Real(s)
    } else {
      Expr::Integer(s as i128)
    });
  }
  Some(out)
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

/// Julian Day Number (JDN) for a proleptic-Gregorian calendar date. Standard
/// integer-arithmetic conversion; exact for year >= 1 (the range where the
/// truncating division matches floor division).
fn gregorian_to_jdn(year: i64, month: i64, day: i64) -> i64 {
  let a = (14 - month) / 12;
  let y = year + 4800 - a;
  let m = month + 12 * a - 3;
  day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045
}

/// Convert a Julian Day Number back to a Julian-calendar {year, month, day}.
fn jdn_to_julian(jdn: i64) -> (i64, i64, i64) {
  let c = jdn + 32082;
  let d = (4 * c + 3) / 1461;
  let e = c - (1461 * d) / 4;
  let m = (5 * e + 2) / 153;
  let day = e - (153 * m + 2) / 5 + 1;
  let month = m + 3 - 12 * (m / 10);
  let year = d - 4800 + m / 10;
  (year, month, day)
}

/// CalendarConvert[dateobj, calendar] — reinterpret a (Gregorian) DateObject
/// in another calendar system. Currently the Julian calendar is supported
/// (plus the Gregorian identity). The source must be a Gregorian DateObject
/// (the default); a DateObject already tagged with a non-Gregorian calendar
/// stays unevaluated, matching wolframscript. Non-Julian target calendars are
/// left unevaluated.
pub fn calendar_convert_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("CalendarConvert", args));
  if args.len() != 2 {
    return unevaluated();
  }
  fn token(e: &Expr) -> Option<&str> {
    match e {
      Expr::String(s) => Some(s.as_str()),
      Expr::Identifier(s) => Some(s.as_str()),
      _ => None,
    }
  }
  let Some(target) = token(&args[1]) else {
    return unevaluated();
  };

  // The source must be a DateObject whose date list is the first element.
  let Expr::FunctionCall { name, args: dargs } = &args[0] else {
    return unevaluated();
  };
  if name != "DateObject" || dargs.is_empty() {
    return unevaluated();
  }
  let Expr::List(items) = &dargs[0] else {
    return unevaluated();
  };
  if items.is_empty() {
    return unevaluated();
  }

  // Recognised calendar and granularity tags among the trailing DateObject
  // elements. A non-Gregorian source calendar isn't supported → unevaluated.
  let is_calendar = |s: &str| {
    matches!(
      s,
      "Gregorian"
        | "Julian"
        | "Islamic"
        | "Hebrew"
        | "Jewish"
        | "Coptic"
        | "Persian"
        | "ArithmeticPersian"
        | "Ethiopic"
        | "Chinese"
        | "Mayan"
    )
  };
  let is_granularity = |s: &str| {
    matches!(
      s,
      "Year"
        | "Quarter"
        | "Month"
        | "Week"
        | "Day"
        | "Hour"
        | "Minute"
        | "Second"
        | "Instant"
    )
  };
  let mut granularity = "Day";
  for a in dargs[1..].iter() {
    if let Some(t) = token(a) {
      if is_calendar(t) && t != "Gregorian" {
        return unevaluated();
      }
      if is_granularity(t) {
        granularity = match t {
          "Year" => "Year",
          "Quarter" => "Quarter",
          "Month" => "Month",
          "Week" => "Week",
          "Day" => "Day",
          "Hour" => "Hour",
          "Minute" => "Minute",
          "Second" => "Second",
          _ => "Instant",
        };
      }
    }
  }

  let as_i64 = |e: &Expr| -> Option<i64> {
    if let Expr::Integer(n) = e {
      i64::try_from(*n).ok()
    } else {
      None
    }
  };
  let Some(y) = as_i64(&items[0]) else {
    return unevaluated();
  };
  let m = items.get(1).and_then(as_i64).unwrap_or(1);
  let d = items.get(2).and_then(as_i64).unwrap_or(1);

  let make_date = |yy: i64, mm: i64, dd: i64, calendar: Option<&str>| {
    let mut oargs = vec![
      Expr::List(
        vec![
          Expr::Integer(yy as i128),
          Expr::Integer(mm as i128),
          Expr::Integer(dd as i128),
        ]
        .into(),
      ),
      Expr::String(granularity.to_string()),
    ];
    if let Some(cal) = calendar {
      oargs.push(Expr::String(cal.to_string()));
    }
    Ok(Expr::FunctionCall {
      name: "DateObject".to_string(),
      args: oargs.into(),
    })
  };

  match target {
    "Gregorian" => make_date(y, m, d, None),
    "Julian" => {
      let (jy, jm, jd) = jdn_to_julian(gregorian_to_jdn(y, m, d));
      make_date(jy, jm, jd, Some("Julian"))
    }
    _ => unevaluated(),
  }
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
pub(crate) fn parse_date_string(s: &str) -> Option<(i64, i64, i64)> {
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

  // Parse a day token, stripping a trailing comma and any ordinal suffix
  // ("4", "8th," and "1st" all yield the numeric day).
  let parse_day = |tok: &str| -> Option<i64> {
    let t = tok.trim_end_matches(',');
    let t = t
      .strip_suffix("st")
      .or_else(|| t.strip_suffix("nd"))
      .or_else(|| t.strip_suffix("rd"))
      .or_else(|| t.strip_suffix("th"))
      .unwrap_or(t);
    t.parse::<i64>().ok()
  };

  let parts: Vec<&str> = s.split_whitespace().collect();
  if parts.len() == 3 {
    // Try "Day Month Year" format
    if let Some(day) = parse_day(parts[0]) {
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
      && let Some(day) = parse_day(parts[1])
      && let Ok(year) = parts[2].parse::<i64>()
    {
      return Some((year, (month_idx + 1) as i64, day));
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
pub(crate) fn day_of_week(year: i64, month: i64, day: i64) -> i64 {
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
    use web_time::{SystemTime, UNIX_EPOCH};
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
    return Ok(unevaluated("AbsoluteTime", args));
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
    return Ok(unevaluated("DateList", args));
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
      // The ISO parser also handles an optional time part
      // ("2022-11-23 14:30:00"); fall back to the date-only parser for other
      // numeric layouts (DD/MM/YYYY, …).
      if let Some(comps) = parse_iso_date_components(s) {
        let get = |i: usize, default: i64| -> i64 {
          match comps.get(i) {
            Some(Expr::Integer(n)) => *n as i64,
            _ => default,
          }
        };
        return Ok(make_date_list(
          get(0, 1900),
          get(1, 1),
          get(2, 1),
          get(3, 0),
          get(4, 0),
          get(5, 0) as f64,
        ));
      }
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
/// DateBounds[{date1, date2, …}] — the earliest and latest of the dates,
/// returned in their original representation (DateObjects stay DateObjects,
/// date lists stay date lists). Dates are ordered by their padded calendar
/// components (year, month, day, hour, minute, second).
pub fn date_bounds_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DateBounds", args));
  let Expr::List(items) = &args[0] else {
    return unevaluated();
  };
  if items.is_empty() {
    return unevaluated();
  }
  // Order key: calendar components padded to {y, m=1, d=1, 0, 0, 0}.
  let key = |e: &Expr| -> Option<[f64; 6]> {
    let c = extract_date_components(e)?;
    let mut k = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    for (i, v) in c.iter().enumerate().take(6) {
      k[i] = *v;
    }
    Some(k)
  };
  let keys: Option<Vec<[f64; 6]>> = items.iter().map(key).collect();
  let Some(keys) = keys else {
    return unevaluated();
  };
  let (mut min_i, mut max_i) = (0usize, 0usize);
  for i in 1..keys.len() {
    if keys[i]
      .partial_cmp(&keys[min_i])
      .is_some_and(|o| o == std::cmp::Ordering::Less)
    {
      min_i = i;
    }
    if keys[i]
      .partial_cmp(&keys[max_i])
      .is_some_and(|o| o == std::cmp::Ordering::Greater)
    {
      max_i = i;
    }
  }
  Ok(Expr::List(
    vec![items[min_i].clone(), items[max_i].clone()].into(),
  ))
}

/// TimeZoneConvert[date, tz] — convert a DateObject between time zones.
/// Numeric offsets work everywhere; named IANA zones resolve through
/// chrono-tz (CLI builds only — the WASM build leaves named zones
/// unevaluated). One-argument form converts to $TimeZone (0.).
pub fn time_zone_convert_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("TimeZoneConvert", args));
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }

  let Expr::FunctionCall {
    name: dname,
    args: dargs,
  } = &args[0]
  else {
    crate::emit_message(&format!(
      "TimeZoneConvert::nodobj: First argument {} in TimeZoneConvert is not a DateObject.",
      crate::syntax::expr_to_output(&args[0])
    ));
    return unevaluated();
  };
  if dname != "DateObject" || dargs.is_empty() {
    crate::emit_message(&format!(
      "TimeZoneConvert::nodobj: First argument {} in TimeZoneConvert is not a DateObject.",
      crate::syntax::expr_to_output(&args[0])
    ));
    return unevaluated();
  }
  let Expr::List(components) = &dargs[0] else {
    return unevaluated();
  };

  // Source zone: canonical 4th argument, or a TimeZone -> tz rule, or 0.
  let mut source_tz = Expr::Real(0.0);
  if dargs.len() == 4 && !matches!(&dargs[3], Expr::Rule { .. }) {
    source_tz = dargs[3].clone();
  } else {
    for a in dargs.iter().skip(1) {
      if let Expr::Rule {
        pattern,
        replacement,
      } = a
        && matches!(pattern.as_ref(), Expr::Identifier(p) if p == "TimeZone")
      {
        source_tz = (**replacement).clone();
      }
    }
  }
  // Granularity and calendar to copy into the result.
  let granularity = match dargs.get(1) {
    Some(Expr::Identifier(g)) | Some(Expr::String(g)) => g.clone(),
    _ => {
      if components.len() >= 4 {
        "Instant".to_string()
      } else {
        ["Year", "Month", "Day"][components.len() - 1].to_string()
      }
    }
  };
  let calendar = match dargs.get(2) {
    Some(Expr::Identifier(c)) | Some(Expr::String(c)) => c.clone(),
    _ => "Gregorian".to_string(),
  };

  let target_tz = args.get(1).cloned().unwrap_or(Expr::Real(0.0));
  // Displayed zone value: numeric offsets print as Reals, names stay.
  let tz_display = |tz: &Expr| -> Option<Expr> {
    match tz {
      Expr::Integer(n) => Some(Expr::Real(*n as f64)),
      Expr::Real(v) => Some(Expr::Real(*v)),
      Expr::String(name) => {
        zone_exists(name).then(|| Expr::String(name.clone()))
      }
      _ => None,
    }
  };
  let Some(target_display) = tz_display(&target_tz) else {
    crate::emit_message(&format!(
      "DateObject::zone: Time zone specification {} should be a real number, integer or time zone string.",
      crate::syntax::expr_to_output(&target_tz)
    ));
    return unevaluated();
  };

  let rebuild = |list: Vec<Expr>, tz: Expr| -> Expr {
    Expr::FunctionCall {
      name: "DateObject".to_string(),
      args: vec![
        Expr::List(list.into()),
        Expr::String(granularity.clone()),
        Expr::String(calendar.clone()),
        tz,
      ]
      .into(),
    }
  };

  // Date-granularity objects only swap the zone label.
  if components.len() <= 3 {
    return Ok(rebuild(components.to_vec(), target_display));
  }

  // Numeric components for offset resolution.
  let as_f64 = |e: &Expr| -> Option<f64> {
    match e {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(v) => Some(*v),
      _ => None,
    }
  };
  let nums: Vec<f64> = match components.iter().map(as_f64).collect() {
    Some(v) => v,
    None => return unevaluated(),
  };

  let source_offset = match zone_offset_hours(&source_tz, &nums, true) {
    Some(v) => v,
    None => return unevaluated(),
  };
  let utc: Vec<f64> = {
    let mut u = nums.clone();
    while u.len() < 6 {
      u.push(0.0);
    }
    u[5] -= source_offset * 3600.0;
    u
  };
  let Some(target_offset) = zone_offset_hours(&target_tz, &utc, false) else {
    crate::emit_message(&format!(
      "DateObject::zone: Time zone specification {} should be a real number, integer or time zone string.",
      crate::syntax::expr_to_output(&target_tz)
    ));
    return unevaluated();
  };

  let shift = ((target_offset - source_offset) * 3600.0).round() as i64;
  let mut list: Vec<Expr> = components.to_vec();
  while list.len() < 6 {
    list.push(Expr::Integer(0));
  }
  // Add the shift to the seconds component, preserving its Real-ness.
  list[5] = match &list[5] {
    Expr::Integer(n) => Expr::Integer(n + shift as i128),
    Expr::Real(v) => Expr::Real(v + shift as f64),
    other => other.clone(),
  };
  let Some(normalized) = normalize_date_components(&list) else {
    return unevaluated();
  };
  Ok(rebuild(normalized, target_display))
}

/// Whether a named IANA zone exists (always false in WASM builds).
#[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
fn zone_exists(name: &str) -> bool {
  name.parse::<chrono_tz::Tz>().is_ok()
}
#[cfg(not(all(feature = "cli", not(target_arch = "wasm32"))))]
fn zone_exists(_name: &str) -> bool {
  false
}

/// Whether a zone name is definitely invalid. Only decidable on CLI builds
/// (which bundle the chrono-tz database); WASM builds return false so a
/// possibly-valid name stays silently unevaluated instead of erroring.
#[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
pub(crate) fn zone_name_invalid(name: &str) -> bool {
  !zone_exists(name)
}
#[cfg(not(all(feature = "cli", not(target_arch = "wasm32"))))]
pub(crate) fn zone_name_invalid(_name: &str) -> bool {
  false
}

/// UTC offset in hours of a named IANA zone at a reference date (a
/// date-like expression) or, when `date` is None, at the current instant.
/// Returns None when the zone database is unavailable (WASM), the name is
/// unknown, or the date can't be resolved.
#[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
pub(crate) fn named_zone_offset_at(
  name: &str,
  date: Option<&Expr>,
) -> Option<f64> {
  let comps: Vec<f64> = match date {
    Some(d) => extract_date_components(d)?,
    None => {
      use chrono::{Datelike, Timelike};
      let now = chrono::Utc::now();
      vec![
        now.year() as f64,
        now.month() as f64,
        now.day() as f64,
        now.hour() as f64,
        now.minute() as f64,
        now.second() as f64,
      ]
    }
  };
  named_zone_offset(name, &comps, false)
}
#[cfg(not(all(feature = "cli", not(target_arch = "wasm32"))))]
pub(crate) fn named_zone_offset_at(
  _name: &str,
  _date: Option<&Expr>,
) -> Option<f64> {
  None
}

/// UTC offset in hours for a zone spec. For named zones, `local` selects
/// whether the components are zone-local wall time (source side) or UTC
/// (target side).
fn zone_offset_hours(
  tz: &Expr,
  components: &[f64],
  local: bool,
) -> Option<f64> {
  match tz {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(v) => Some(*v),
    Expr::String(name) => named_zone_offset(name, components, local),
    _ => None,
  }
}

#[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
fn named_zone_offset(name: &str, c: &[f64], local: bool) -> Option<f64> {
  use chrono::{Offset, TimeZone};
  let tz: chrono_tz::Tz = name.parse().ok()?;
  let (y, mo, d) = (
    c[0] as i32,
    c.get(1).copied().unwrap_or(1.0) as u32,
    c.get(2).copied().unwrap_or(1.0) as u32,
  );
  let h = c.get(3).copied().unwrap_or(0.0);
  let mi = c.get(4).copied().unwrap_or(0.0);
  let sec = c.get(5).copied().unwrap_or(0.0);
  // Roll sub-day components into whole seconds from midnight; the date
  // itself may need normalizing when the UTC shift crossed midnight.
  let total = (h * 3600.0 + mi * 60.0 + sec).floor() as i64;
  let date = chrono::NaiveDate::from_ymd_opt(y, mo, d)?;
  let datetime = date.and_hms_opt(0, 0, 0)? + chrono::Duration::seconds(total);
  let offset_secs = if local {
    tz.from_local_datetime(&datetime)
      .earliest()?
      .offset()
      .fix()
      .local_minus_utc()
  } else {
    tz.offset_from_utc_datetime(&datetime)
      .fix()
      .local_minus_utc()
  };
  Some(offset_secs as f64 / 3600.0)
}
#[cfg(not(all(feature = "cli", not(target_arch = "wasm32"))))]
fn named_zone_offset(_name: &str, _c: &[f64], _local: bool) -> Option<f64> {
  None
}

pub fn date_plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("DatePlus", args));
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
        // Units accept singular and plural spellings ("Day"/"Days").
        match unit.to_ascii_lowercase().trim_end_matches('s') {
          "day" | "week" => {
            let shift = if unit.to_ascii_lowercase().starts_with("week") {
              n * 7
            } else {
              n
            };
            let abs = date_to_absolute_days(y, m, d) + shift;
            let (ny, nm, nd) = absolute_days_to_date(abs);
            y = ny;
            m = nm;
            d = nd;
          }
          "month" => {
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
          "year" => {
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
        // Sub-day time units are valid calendar increments in Wolfram, but
        // Woxi does not yet carry the time-of-day component, so it keeps the
        // (approximate) day-based behavior rather than rejecting them.
        "Hour" | "Hours" | "hour" | "hours" | "Minute" | "Minutes"
        | "minute" | "minutes" | "Second" | "Seconds" | "second"
        | "seconds" => n,
        // Any other unit (e.g. "Meters", "Kilograms") is not a calendar
        // increment: emit the incompatible-unit messages and stay unevaluated,
        // rather than silently treating the magnitude as a number of days.
        _ => {
          crate::emit_message(&format!(
            "UnitConvert::compat: {} and MixedUnit[{{Years, Months, Days, \
             Hours, Minutes, Seconds}}] are incompatible units.",
            unit
          ));
          crate::emit_message(&format!(
            "DatePlus::inc: {} is not a recognized calendar increment \
             specification for DatePlus.",
            quantity_increment_text(&qargs[0], &unit)
          ));
          return Ok(Expr::FunctionCall {
            name: "DatePlus".to_string(),
            args: vec![date_arg, delta_arg].into(),
          });
        }
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

/// Renders a `Quantity` magnitude and unit the way Wolfram names it in the
/// `DatePlus::inc` message: the lowercased unit, singular when the magnitude
/// is exactly 1 and plural otherwise, e.g. `5 meters`, `1 meter`.
fn quantity_increment_text(magnitude: &Expr, unit: &str) -> String {
  let unit_lower = unit.to_ascii_lowercase();
  let is_one = matches!(magnitude, Expr::Integer(1));
  let word = if is_one {
    unit_lower
      .strip_suffix('s')
      .unwrap_or(&unit_lower)
      .to_string()
  } else if unit_lower.ends_with('s') {
    unit_lower
  } else {
    format!("{unit_lower}s")
  };
  format!("{} {}", crate::syntax::expr_to_string(magnitude), word)
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

/// Advance `(y, m, d)` by `k` months, clamping the day to the last valid day
/// of the resulting month (matching DatePlus / Wolfram calendar arithmetic).
/// `k` may be negative.
fn add_months_clamped(y: i64, m: i64, d: i64, k: i64) -> (i64, i64, i64) {
  let total = (m - 1) + k;
  let new_y = y + total.div_euclid(12);
  let new_m = total.rem_euclid(12) + 1;
  let new_d = d.min(days_in_month(new_y, new_m));
  (new_y, new_m, new_d)
}

/// Calendar-based difference in whole steps plus a fractional part, matching
/// Wolfram's `DateDifference[d1, d2, "Month" | "Year"]`. Each step is
/// `step_months` months (1 for "Month", 12 for "Year"). The result is
///   k + (days(d2) - days(start)) / (days(next) - days(start))
/// where `k` is the largest number of whole steps with `d1 + k*step <= d2`,
/// `start = d1 + k*step`, and `next = d1 + (k+1)*step`; i.e. the fractional
/// part divides the leftover days by the actual length of the partial
/// month/year. Negative when `d2` precedes `d1`.
fn calendar_step_difference(
  y1: i64,
  m1: i64,
  d1: i64,
  y2: i64,
  m2: i64,
  d2: i64,
  step_months: i64,
) -> f64 {
  let days1 = date_to_absolute_days(y1, m1, d1);
  let days2 = date_to_absolute_days(y2, m2, d2);
  // Work forward from the earlier date and negate if the order is reversed.
  if days2 < days1 {
    return -calendar_step_difference(y2, m2, d2, y1, m1, d1, step_months);
  }
  let add_k = |k: i64| -> i64 {
    let (yy, mm, dd) = add_months_clamped(y1, m1, d1, k * step_months);
    date_to_absolute_days(yy, mm, dd)
  };
  // Initial estimate of the whole-step count, then adjust by ±1.
  let mut k = ((y2 - y1) * 12 + (m2 - m1)) / step_months;
  if k < 0 {
    k = 0;
  }
  while add_k(k + 1) <= days2 {
    k += 1;
  }
  while k > 0 && add_k(k) > days2 {
    k -= 1;
  }
  let start = add_k(k);
  let next = add_k(k + 1);
  let period = next - start;
  k as f64 + (days2 - start) as f64 / period as f64
}

/// DateDifference[date1, date2] — difference in days
/// DateDifference[date1, date2, "unit"] — difference in given unit
/// A date argument that Wolfram reports as un-interpretable via a `::date`
/// message: a bare symbol or an arbitrary (non date-producing) function call.
/// Numbers, strings and lists are left to the normal date extractors.
fn is_uninterpretable_date_spec(e: &Expr) -> bool {
  match e {
    Expr::Identifier(name) => {
      !matches!(name.as_str(), "Today" | "Now" | "Yesterday" | "Tomorrow")
    }
    Expr::FunctionCall { name, .. } => !matches!(
      name.as_str(),
      "DateObject"
        | "DateList"
        | "DateString"
        | "AbsoluteTime"
        | "FromAbsoluteTime"
        | "FromUnixTime"
        | "TimeObject"
        | "Today"
        | "Now"
        | "Yesterday"
        | "Tomorrow"
        | "Quantity"
    ),
    _ => false,
  }
}

/// If any of the given date-position arguments (evaluated) cannot be
/// interpreted as a date specification and is clearly non-date (a symbol or an
/// arbitrary function call), emit the `head::date` message for the first such
/// argument and return the whole call `head[args...]` unevaluated. Returns
/// `None` when every date argument looks interpretable, so the caller can
/// proceed with its normal computation. Mirrors Wolfram, which reports the
/// first un-interpretable date and leaves the expression unevaluated.
pub fn date_spec_error(
  head: &str,
  args: &[Expr],
  date_args: &[Expr],
) -> Option<Expr> {
  for a in date_args {
    let ev =
      crate::evaluator::evaluate_expr_to_expr(a).unwrap_or_else(|_| a.clone());
    if is_uninterpretable_date_spec(&ev) {
      crate::emit_message(&format!(
        "{}::date: Expression {} cannot be interpreted as a date specification.",
        head,
        crate::syntax::format_expr(&ev, crate::syntax::ExprForm::Output)
      ));
      return Some(unevaluated(head, args));
    }
  }
  None
}

pub fn date_difference_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(unevaluated("DateDifference", args));
  }

  if let Some(unevaluated) = date_spec_error("DateDifference", args, &args[..2])
  {
    return Ok(unevaluated);
  }

  let date1 = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let date2 = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  let c1 = match extract_date_components(&date1) {
    Some(c) => c,
    None => {
      return Ok(unevaluated("DateDifference", args));
    }
  };
  let c2 = match extract_date_components(&date2) {
    Some(c) => c,
    None => {
      return Ok(unevaluated("DateDifference", args));
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
      return Ok(unevaluated("DateDifference", args));
    }
  };

  // Calendar-based helpers use whole (y, m, d) components; the leading three
  // components default to 1 when absent (e.g. a bare {year}).
  let ymd = |c: &[f64]| -> (i64, i64, i64) {
    (
      c[0] as i64,
      if c.len() > 1 { c[1] as i64 } else { 1 },
      if c.len() > 2 { c[2] as i64 } else { 1 },
    )
  };

  let (value, unit_name) = match unit.as_str() {
    "Year" => {
      let (y1, m1, d1) = ymd(&c1);
      let (y2, m2, d2) = ymd(&c2);
      (
        calendar_step_difference(y1, m1, d1, y2, m2, d2, 12),
        "Years",
      )
    }
    "Month" => {
      let (y1, m1, d1) = ymd(&c1);
      let (y2, m2, d2) = ymd(&c2);
      (
        calendar_step_difference(y1, m1, d1, y2, m2, d2, 1),
        "Months",
      )
    }
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
          _ => return Ok(unevaluated_date_difference(c1, c2, units)),
        }
      }
      v
    }
    _ => return Ok(unevaluated_date_difference(c1, c2, units)),
  };
  if unit_strs.is_empty() {
    return Ok(unevaluated_date_difference(c1, c2, units));
  }
  // Bail out for calendar-aware units that don't reduce to fixed seconds.
  for u in &unit_strs {
    if u == "Year" || u == "Month" {
      return Ok(unevaluated_date_difference(c1, c2, units));
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
      _ => return Ok(unevaluated_date_difference(c1, c2, units)),
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

fn unevaluated_date_difference(c1: &[f64], c2: &[f64], units: &Expr) -> Expr {
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
/// True if `s` is a recognized DateString format name — either a compound
/// named format ("ISODate", "DateTime", …) or a single format element
/// ("Year", "DayName", …). Used so `DateString[fmt]` formats the current date.
fn is_known_date_format(s: &str) -> bool {
  matches!(
    s,
    "ISODateTime"
      | "ISODate"
      | "DateTime"
      | "DateTimeShort"
      | "Date"
      | "DateShort"
      | "Time"
      | "Year"
      | "YearShort"
      | "Month"
      | "MonthName"
      | "MonthNameShort"
      | "Day"
      | "DayName"
      | "DayNameShort"
      | "Hour"
      | "Minute"
      | "Second"
  )
}

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

  // A numeric first argument is an absolute time (seconds since 1900-01-01).
  // Convert it to a date list and re-dispatch, so DateString[3155673600]
  // yields "Sat 1 Jan 2000 00:00:00" like wolframscript.
  if matches!(
    &date_arg,
    Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_)
  ) {
    let date_list = date_list_ast(std::slice::from_ref(&date_arg))?;
    let mut new_args = vec![date_list];
    new_args.extend(args[1..].iter().cloned());
    return date_string_ast(&new_args);
  }

  // DateString[fmt] with a single string argument: if `fmt` is a recognized
  // date-format name (e.g. "ISODate", "DateTime", "Year"), it formats the
  // CURRENT date, i.e. DateString[fmt] == DateString[Now, fmt]. Any other
  // string is returned unchanged (matching wolframscript, e.g.
  // DateString["hello"] -> "hello").
  if args.len() == 1
    && let Expr::String(s) = &date_arg
  {
    if is_known_date_format(s) {
      let abs_time = absolute_time_ast(&[])?;
      let date_list = date_list_ast(&[abs_time])?;
      return date_string_ast(&[date_list, args[0].clone()]);
    }
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
          return Ok(unevaluated("DateString", args));
        }
      } else {
        return Ok(unevaluated("DateString", args));
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
      return Ok(unevaluated("DateString", args));
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
        // "Hour" is the 24-hour clock by default; "Hour24" is its explicit
        // 2-digit form.
        "Hour" | "Hour24" => result.push_str(&format!("{:02}", h)),
        "Minute" => result.push_str(&format!("{:02}", min)),
        "Second" => result.push_str(&format!("{:02}", sec as i64)),
        // ISO day of the week: Monday = 1, ..., Sunday = 7.
        "ISOWeekDay" => {
          result.push_str(&format!("{}", day_of_week(y, m, d) + 1))
        }
        // "Short" variants omit the leading zero.
        "MonthShort" => result.push_str(&format!("{}", m)),
        "DayShort" => result.push_str(&format!("{}", d)),
        "HourShort" | "Hour24Short" => result.push_str(&format!("{}", h)),
        "MinuteShort" => result.push_str(&format!("{}", min)),
        "SecondShort" => result.push_str(&format!("{}", sec as i64)),
        // 12-hour clock: 0 and 12 both map to 12, 13..23 map to 1..11.
        "Hour12" => {
          result.push_str(&format!("{:02}", (h + 11).rem_euclid(12) + 1))
        }
        "Hour12Short" => {
          result.push_str(&format!("{}", (h + 11).rem_euclid(12) + 1))
        }
        "AMPM" => result.push_str(if h < 12 { "AM" } else { "PM" }),
        "AMPMLowerCase" => result.push_str(if h < 12 { "am" } else { "pm" }),
        "Quarter" => result.push_str(&format!("{}", (m - 1) / 3 + 1)),
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
    return Ok(unevaluated("DayName", args));
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

  Ok(unevaluated("DayName", args))
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
  let unevaluated = || Ok(unevaluated("DateValue", args));
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
      // "…Short" component forms. For most fields these equal the plain
      // integer component (the "Short" only suppresses zero-padding in
      // DateString), but "YearShort" is the year modulo 100.
      "YearShort" => int(y.rem_euclid(100)),
      "MonthShort" => int(mo),
      "DayShort" => int(d),
      "HourShort" => int(h),
      "MinuteShort" => int(mi),
      "SecondShort" => int(sec),
      // 12-hour clock: 0 and 12 both map to 12, 13..23 map to 1..11.
      "Hour12" => int((h + 11).rem_euclid(12) + 1),
      "AMPM" => Expr::String(if h < 12 { "AM" } else { "PM" }.to_string()),
      // DayName is a Symbol (`Saturday`), not a String — matching `DayName[…]`
      // and wolframscript. The other name keys (MonthName, *Short) are Strings.
      "DayName" => {
        Expr::Identifier(day_name(day_of_week(y, mo, d)).to_string())
      }
      "MonthName" => Expr::String(month_name(mo).to_string()),
      "MonthNameShort" => Expr::String(month_name_short(mo).to_string()),
      "DayNameShort" => {
        Expr::String(day_name_short(day_of_week(y, mo, d)).to_string())
      }
      "Quarter" => int((mo - 1) / 3 + 1),
      // The ISO 8601 ordinal date counts calendar days from the start of the
      // year (1..366), the same value as DayOfYear.
      "DayOfYear" | "ISOYearDay" => int(day_of_year(y, mo, d)),
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
  let unevaluated = || Ok(unevaluated("DayMatchQ", args));
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

/// The time-zone offset a `DateObject` carries, if any. A DateObject built from
/// a date-time list (e.g. `DateObject[{2024, 7, 8, 14, 30}]`) canonicalizes to
/// the 4-argument form `DateObject[list, granularity, calendar, tz]`; a
/// date-only object has no time zone. `DayRound` retains the calendar and time
/// zone of such an input, so the rounded result reads
/// `DateObject[{…}, Day, Gregorian, 0.]` rather than the bare `…, Day` form.
fn date_object_timezone(expr: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "DateObject" || args.len() < 4 {
    return None;
  }
  match &args[3] {
    tz @ (Expr::Real(_) | Expr::Integer(_)) => Some(tz.clone()),
    _ => None,
  }
}

/// Build the `DateObject` result of a DayRound, carrying the calendar and time
/// zone (`Day, Gregorian, tz`) when the input object had one, otherwise the
/// bare `DateObject[{…}, Day]` form.
fn day_round_result(
  y: i128,
  m: i128,
  d: i128,
  tz: Option<Expr>,
) -> Result<Expr, InterpreterError> {
  let mut date_args = vec![Expr::List(
    vec![Expr::Integer(y), Expr::Integer(m), Expr::Integer(d)].into(),
  )];
  if let Some(tz) = tz {
    date_args.push(Expr::String("Day".to_string()));
    date_args.push(Expr::String("Gregorian".to_string()));
    date_args.push(tz);
  }
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: date_args.into(),
  })
}

/// DayRound[date, daytype] — round `date` to the nearest day of the given day
/// type using the default next-day convention (the next occurrence on or after
/// `date`). Returns a `DateObject[…, Day]`. Supports weekday names (symbol or
/// string) and the calendar classes "Day", "Weekday", and "Weekend"; holiday-
/// dependent classes (e.g. "BusinessDay") and the 3-argument rounding form are
/// left unevaluated.
pub fn day_round_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DayRound", args));
  // DayRound[date] — floor the date to its containing day, returning a
  // DateObject with Day granularity (the daytype defaults to every day).
  if args.len() == 1 {
    let Some(date_list) = resolve_date_to_list(&args[0]) else {
      return unevaluated();
    };
    let Some(comps) = extract_date_components(&date_list) else {
      return unevaluated();
    };
    if comps.len() < 3 {
      return unevaluated();
    }
    return day_round_result(
      comps[0] as i128,
      comps[1] as i128,
      comps[2] as i128,
      date_object_timezone(&args[0]),
    );
  }
  if args.len() != 2 {
    return unevaluated();
  }
  let (Expr::Identifier(daytype) | Expr::String(daytype)) = &args[1] else {
    return unevaluated();
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
  // day_of_week gives 0 = Monday … 4 = Friday, 5 = Saturday, 6 = Sunday.
  let current_dow = day_of_week(y, m, d);
  // Days to advance onto the next day of the requested type on or after `date`.
  // The day-type classes "Day"/"Weekday"/"Weekend" and any weekday name are
  // supported; holiday-dependent classes (e.g. "BusinessDay") are not.
  let days_forward = if daytype == "Day" {
    0
  } else if daytype == "Weekday" {
    (0i64..7).find(|k| (current_dow + k) % 7 <= 4).unwrap()
  } else if daytype == "Weekend" {
    (0i64..7).find(|k| (current_dow + k) % 7 >= 5).unwrap()
  } else if let Some(target_dow) =
    WEEKDAY_NAMES.iter().position(|n| n == daytype)
  {
    (target_dow as i64 - current_dow).rem_euclid(7)
  } else {
    return unevaluated();
  };
  let (ny, nm, nd) =
    absolute_days_to_date(date_to_absolute_days(y, m, d) + days_forward);

  // Build the DateObject, retaining the input object's calendar and time zone
  // (`Day, Gregorian, tz`) when it carried one.
  day_round_result(
    ny as i128,
    nm as i128,
    nd as i128,
    date_object_timezone(&args[0]),
  )
}

/// The Day/Month/Year unit one step after (`forward`) or before a date given by
/// its calendar components, as a `DateObject` at that granularity. Other
/// granularity strings return `None`.
fn date_granularity_step(
  comps: &[f64],
  gran: &str,
  forward: bool,
) -> Option<Result<Expr, InterpreterError>> {
  let sign: i64 = if forward { 1 } else { -1 };
  let y = *comps.first()? as i64;
  let make = |v: Vec<i128>| {
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "DateObject".to_string(),
      args: vec![Expr::List(
        v.into_iter().map(Expr::Integer).collect::<Vec<_>>().into(),
      )]
      .into(),
    })
  };
  match gran {
    "Day" => {
      let m = *comps.get(1).unwrap_or(&1.0) as i64;
      let d = *comps.get(2).unwrap_or(&1.0) as i64;
      let (ny, nm, nd) =
        absolute_days_to_date(date_to_absolute_days(y, m, d) + sign);
      Some(make(vec![ny as i128, nm as i128, nd as i128]))
    }
    "Month" => {
      let m = *comps.get(1).unwrap_or(&1.0) as i64;
      let (ny, nm) = if forward {
        if m >= 12 { (y + 1, 1) } else { (y, m + 1) }
      } else if m <= 1 {
        (y - 1, 12)
      } else {
        (y, m - 1)
      };
      Some(make(vec![ny as i128, nm as i128]))
    }
    "Year" => Some(make(vec![(y + sign) as i128])),
    _ => None,
  }
}

/// NextDate[date, weekday] — the next occurrence of the given weekday strictly
/// after `date`, returned as a `DateObject[…, Day]`. Also NextDate[date, gran]
/// for gran = "Day"/"Month"/"Year" gives the next such calendar unit. Other
/// forms are left unevaluated.
pub fn next_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("NextDate", args));
  if args.len() != 2 {
    return unevaluated();
  }
  // Granularity form: NextDate[date, "Day" | "Month" | "Year"].
  if let Expr::Identifier(s) | Expr::String(s) = &args[1]
    && matches!(s.as_str(), "Day" | "Month" | "Year")
  {
    let Some(date_list) = resolve_date_to_list(&args[0]) else {
      return unevaluated();
    };
    let Some(comps) = extract_date_components(&date_list) else {
      return unevaluated();
    };
    return match date_granularity_step(&comps, s, true) {
      Some(r) => r,
      None => unevaluated(),
    };
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
  // Strictly after `date`: the offset is in 1..=7 (a same-weekday date jumps a
  // full week rather than staying put).
  let days_forward = (target_dow - current_dow + 6).rem_euclid(7) + 1;
  let (ny, nm, nd) =
    absolute_days_to_date(date_to_absolute_days(y, m, d) + days_forward);

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

/// PreviousDate[date, weekday] — the previous occurrence of the given weekday
/// strictly before `date`, returned as a `DateObject[…, Day]`. The mirror of
/// `NextDate`: only the weekday-name form (symbol or string Monday…Sunday) is
/// supported; other forms are left unevaluated.
pub fn previous_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("PreviousDate", args));
  if args.len() != 2 {
    return unevaluated();
  }
  // Granularity form: PreviousDate[date, "Day" | "Month" | "Year"].
  if let Expr::Identifier(s) | Expr::String(s) = &args[1]
    && matches!(s.as_str(), "Day" | "Month" | "Year")
  {
    let Some(date_list) = resolve_date_to_list(&args[0]) else {
      return unevaluated();
    };
    let Some(comps) = extract_date_components(&date_list) else {
      return unevaluated();
    };
    return match date_granularity_step(&comps, s, false) {
      Some(r) => r,
      None => unevaluated(),
    };
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
  // Strictly before `date`: the offset is in 1..=7 (a same-weekday date jumps a
  // full week back rather than staying put).
  let days_back = (current_dow - target_dow + 6).rem_euclid(7) + 1;
  let (ny, nm, nd) =
    absolute_days_to_date(date_to_absolute_days(y, m, d) - days_back);

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

/// A numeric ordering key for a `DateObject` (absolute seconds since the epoch)
/// or a `TimeObject` (seconds since midnight), tagged with a kind (0 = date,
/// 1 = time) so the two domains are never ordered against each other. Used by
/// the `<`/`>`/`<=`/`>=` comparisons and Max/Min. Returns None for anything else.
pub fn datetime_order_key(e: &Expr) -> Option<(f64, u8)> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name == "DateObject" {
    let resolved = resolve_date_to_list(e)?;
    let Expr::List(items) = &resolved else {
      return None;
    };
    let nums: Vec<f64> = items
      .iter()
      .map(|it| match it {
        Expr::Integer(n) => Some(*n as f64),
        Expr::Real(r) => Some(*r),
        _ => None,
      })
      .collect::<Option<Vec<_>>>()?;
    let g = |i: usize, d: f64| nums.get(i).copied().unwrap_or(d);
    return Some((
      date_to_absolute_seconds(
        g(0, 0.0) as i64,
        g(1, 1.0) as i64,
        g(2, 1.0) as i64,
        g(3, 0.0) as i64,
        g(4, 0.0) as i64,
        g(5, 0.0),
      ),
      0,
    ));
  }
  if name == "TimeObject"
    && !args.is_empty()
    && let Expr::List(items) = &args[0]
    && (1..=3).contains(&items.len())
  {
    let mut comps = Vec::with_capacity(items.len());
    for it in items.iter() {
      comps.push(match it {
        Expr::Integer(n) => *n as f64,
        Expr::Real(r) => *r,
        _ => return None,
      });
    }
    let secs = comps.first().copied().unwrap_or(0.0) * 3600.0
      + comps.get(1).copied().unwrap_or(0.0) * 60.0
      + comps.get(2).copied().unwrap_or(0.0);
    return Some((secs, 1));
  }
  None
}

// ─── DayPlus ────────────────────────────────────────────────────────

/// DayPlus[date, n] - adds n days to a date and returns DateObject
/// DayPlus[date, n, "BusinessDay"] - adds n business days (Mon-Fri)
pub fn day_plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(unevaluated("DayPlus", args));
  }

  let date_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let n_arg = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  let components = match extract_date_components(&date_arg) {
    Some(c) => c,
    None => {
      return Ok(unevaluated("DayPlus", args));
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
      return Ok(unevaluated("DayPlus", args));
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
pub(crate) fn weekday_index(name: &str) -> Option<i64> {
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
    return Ok(unevaluated("DayRange", args));
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
            return Ok(unevaluated("DayRange", args));
          }
        },
        _ => {
          return Ok(unevaluated("DayRange", args));
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
  let unevaluated = || Ok(unevaluated("DayRange", args));

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

/// TimeObject[{h}] / TimeObject[{h, m}] / TimeObject[{h, m, s}]
///
/// Normalizes a time-of-day list and tags it with a granularity, matching
/// wolframscript:
///   * `{h}`       → `TimeObject[{h}, Hour]`     (whole hours)
///   * `{h, m}`    → `TimeObject[{h, m}, Minute]` (whole minutes)
///   * `{h, m, s}` → `TimeObject[{h, m, s}, Instant]` (fractional seconds kept)
///
/// Components carry over (75 minutes → +1 hour, 15 minutes) and the whole
/// time wraps modulo 24 hours, so `{25, 30}` becomes `{1, 30}` and `{-1, 30}`
/// becomes `{23, 30}`. The precision of the least-significant field is set by
/// the granularity: hours and minutes are truncated to whole numbers, while
/// the Instant form keeps a fractional seconds value.
pub fn time_object_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("TimeObject", args));

  // Only the single date-list argument form is handled here; option forms
  // (TimeZone, string parsing, …) fall through unevaluated.
  if args.len() != 1 {
    return unevaluated();
  }

  let arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let items = match &arg {
    Expr::List(items) if !items.is_empty() && items.len() <= 3 => items,
    _ => return unevaluated(),
  };

  let mut comps: Vec<f64> = Vec::with_capacity(items.len());
  for item in items.iter() {
    match item {
      Expr::Integer(n) => comps.push(*n as f64),
      Expr::Real(r) => comps.push(*r),
      _ => return unevaluated(),
    }
  }
  let n = comps.len();

  // Total seconds since midnight, then wrap into a single day.
  let mut total = comps.first().copied().unwrap_or(0.0) * 3600.0
    + comps.get(1).copied().unwrap_or(0.0) * 60.0
    + comps.get(2).copied().unwrap_or(0.0);
  total = total.rem_euclid(86400.0);

  // Build a field that is an Integer when whole and a Real otherwise.
  let field = |v: f64| -> Expr {
    if v.fract() == 0.0 {
      Expr::Integer(v as i128)
    } else {
      Expr::Real(v)
    }
  };

  let (fields, granularity): (Vec<Expr>, &str) = match n {
    1 => {
      let h = (total / 3600.0).floor();
      (vec![Expr::Integer(h as i128)], "Hour")
    }
    2 => {
      let total_min = (total / 60.0).floor();
      let h = (total_min / 60.0).floor();
      let m = total_min - h * 60.0;
      (
        vec![Expr::Integer(h as i128), Expr::Integer(m as i128)],
        "Minute",
      )
    }
    _ => {
      let h = (total / 3600.0).floor();
      let rem = total - h * 3600.0;
      let m = (rem / 60.0).floor();
      let s = rem - m * 60.0;
      (
        vec![Expr::Integer(h as i128), Expr::Integer(m as i128), field(s)],
        "Instant",
      )
    }
  };

  Ok(Expr::FunctionCall {
    name: "TimeObject".to_string(),
    args: vec![
      Expr::List(fields.into()),
      Expr::String(granularity.to_string()),
    ]
    .into(),
  })
}

/// DateRange[start, end] / DateRange[start, end, increment]
///
/// Returns the list of dates from `start` to `end` (inclusive), each rendered
/// as a six-element date list `{y, m, d, h, min, sec.}` (the seconds field is
/// always a Real, matching wolframscript). The default increment is one day.
///
/// The increment may be a bare integer (interpreted as days), a `Quantity[n,
/// unit]` (units "Seconds", "Minutes", "Hours", "Days", "Weeks", "Months",
/// "Years"), or a string unit name (increment of one). "Months" and "Years"
/// step by the calendar; the other units step by a fixed number of seconds.
/// If `start` is after `end` the result is the empty list.
pub fn date_range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DateRange", args));

  if args.len() != 2 && args.len() != 3 {
    return unevaluated();
  }

  let start_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let end_arg = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  // Pad a partial date spec to six components {y, m, d, h, min, sec}.
  let pad_full = |c: Vec<f64>| -> Option<[f64; 6]> {
    if c.is_empty() {
      return None;
    }
    let mut out = [0.0_f64; 6];
    out[1] = 1.0; // default month
    out[2] = 1.0; // default day
    for (i, v) in c.into_iter().take(6).enumerate() {
      out[i] = v;
    }
    Some(out)
  };

  let start = match extract_date_components(&start_arg).and_then(pad_full) {
    Some(c) => c,
    None => return unevaluated(),
  };
  let end = match extract_date_components(&end_arg).and_then(pad_full) {
    Some(c) => c,
    None => return unevaluated(),
  };

  // Parse the optional increment into either a fixed number of seconds or a
  // calendar step expressed as (months, years).
  enum Step {
    Seconds(f64),
    Calendar { months: i64 },
  }

  let unit_to_step = |unit: &str, qty: f64| -> Option<Step> {
    match unit {
      "Second" | "Seconds" => Some(Step::Seconds(qty)),
      "Minute" | "Minutes" => Some(Step::Seconds(qty * 60.0)),
      "Hour" | "Hours" => Some(Step::Seconds(qty * 3600.0)),
      "Day" | "Days" => Some(Step::Seconds(qty * 86400.0)),
      "Week" | "Weeks" => Some(Step::Seconds(qty * 7.0 * 86400.0)),
      "Month" | "Months" => Some(Step::Calendar { months: qty as i64 }),
      "Year" | "Years" => Some(Step::Calendar {
        months: qty as i64 * 12,
      }),
      _ => None,
    }
  };

  let step = if args.len() == 3 {
    let inc = crate::evaluator::evaluate_expr_to_expr(&args[2])?;
    match &inc {
      // Bare integer/real → number of days.
      Expr::Integer(n) => Step::Seconds(*n as f64 * 86400.0),
      Expr::Real(r) => Step::Seconds(*r * 86400.0),
      // String unit name → step of one of that unit.
      Expr::String(u) => match unit_to_step(u, 1.0) {
        Some(s) => s,
        None => return unevaluated(),
      },
      // Quantity[n, "Unit"]
      Expr::FunctionCall { name, args: qargs }
        if name == "Quantity" && qargs.len() == 2 =>
      {
        let qty = match &qargs[0] {
          Expr::Integer(n) => *n as f64,
          Expr::Real(r) => *r,
          _ => return unevaluated(),
        };
        let unit = match &qargs[1] {
          Expr::String(u) => u.clone(),
          _ => return unevaluated(),
        };
        match unit_to_step(&unit, qty) {
          Some(s) => s,
          None => return unevaluated(),
        }
      }
      _ => return unevaluated(),
    }
  } else {
    Step::Seconds(86400.0)
  };

  let make_elem = |y: i64, m: i64, d: i64, h: i64, min: i64, sec: f64| {
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
  };

  let start_secs = date_to_absolute_seconds(
    start[0] as i64,
    start[1] as i64,
    start[2] as i64,
    start[3] as i64,
    start[4] as i64,
    start[5],
  );
  let end_secs = date_to_absolute_seconds(
    end[0] as i64,
    end[1] as i64,
    end[2] as i64,
    end[3] as i64,
    end[4] as i64,
    end[5],
  );

  let mut result = Vec::new();

  match step {
    Step::Seconds(step_secs) => {
      if step_secs <= 0.0 {
        return unevaluated();
      }
      // Guard against pathologically large ranges.
      let count = ((end_secs - start_secs) / step_secs).floor();
      if count > 1_000_000.0 {
        return unevaluated();
      }
      let mut secs = start_secs;
      while secs <= end_secs + 1e-6 {
        let (y, m, d, h, min, sec) = absolute_seconds_to_date(secs);
        result.push(make_elem(y, m, d, h, min, sec));
        secs += step_secs;
      }
    }
    Step::Calendar { months } => {
      if months <= 0 {
        return unevaluated();
      }
      // Step by whole months, preserving the day/time of `start` and clamping
      // the day to the number of days available in the target month.
      let base_y = start[0] as i64;
      let base_m0 = start[1] as i64 - 1; // zero-based month index
      let base_d = start[2] as i64;
      let h = start[3] as i64;
      let min = start[4] as i64;
      let sec = start[5];
      let mut k: i64 = 0;
      loop {
        let total_m0 = base_m0 + k * months;
        let y = base_y + total_m0.div_euclid(12);
        let m = total_m0.rem_euclid(12) + 1;
        let d = base_d.min(days_in_month(y, m));
        let secs = date_to_absolute_seconds(y, m, d, h, min, sec);
        if secs > end_secs + 1e-6 {
          break;
        }
        result.push(make_elem(y, m, d, h, min, sec));
        k += 1;
        if k > 1_000_000 {
          return unevaluated();
        }
      }
    }
  }

  Ok(Expr::List(result.into()))
}

/// JulianDate[] / JulianDate[{y, m, d, h, min, s}] - Julian date of the
/// current instant or of a proleptic-Gregorian date list. Wolfram has
/// no input year zero: 0 and -1 both denote 1 BC (astronomical year 0),
/// so negative years shift by one. Date lists are taken as-is with no
/// time-zone adjustment, matching wolframscript.
pub fn julian_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("JulianDate", args);

  if args.is_empty() {
    use web_time::{SystemTime, UNIX_EPOCH};
    let unix = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .map(|d| d.as_secs_f64())
      .unwrap_or(0.0);
    return Ok(Expr::Real(2440587.5 + unix / 86400.0));
  }
  if args.len() != 1 {
    return Ok(unevaluated(args));
  }

  // A DateObject argument contributes its component list directly (all
  // Woxi date instants carry TimeZone 0, matching JulianDate's UTC base).
  if let Expr::FunctionCall { name, args: dargs } = &args[0]
    && name == "DateObject"
    && !dargs.is_empty()
    && matches!(&dargs[0], Expr::List(_))
  {
    return julian_date_ast(std::slice::from_ref(&dargs[0]));
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

// ─── MidDate ────────────────────────────────────────────────────────

/// Calendar granularity names accepted by MidDate's second argument and
/// carried by DateObject/DateInterval expressions.
fn is_granularity_name(s: &str) -> bool {
  matches!(
    s,
    "Year"
      | "Quarter"
      | "Month"
      | "Week"
      | "Day"
      | "Hour"
      | "Minute"
      | "Second"
      | "Instant"
  )
}

/// The granularity implied by the number of components in a raw date list,
/// mirroring DateObject's tagging (1 → Year, …, 6 → Instant).
fn granularity_from_component_count(len: usize) -> &'static str {
  match len {
    1 => "Year",
    2 => "Month",
    3 => "Day",
    4 => "Hour",
    5 => "Minute",
    _ => "Instant",
  }
}

/// The real span in seconds of one granularity unit starting at the given
/// calendar date (a "Month" at 2024-02 is 29 days, at 2024-04 is 30).
fn granularity_actual_seconds(y: i64, m: i64, gran: &str) -> f64 {
  match gran {
    "Year" => days_in_year(y) as f64 * 86400.0,
    "Quarter" => {
      let qm = (m - 1) / 3 * 3 + 1;
      (days_in_month(y, qm)
        + days_in_month(y, qm + 1)
        + days_in_month(y, qm + 2)) as f64
        * 86400.0
    }
    "Month" => days_in_month(y, m) as f64 * 86400.0,
    "Week" => 7.0 * 86400.0,
    "Day" => 86400.0,
    "Hour" => 3600.0,
    "Minute" => 60.0,
    "Second" => 1.0,
    _ => 0.0, // "Instant"
  }
}

/// The average span in seconds of one granularity unit, used as the weight
/// in MidDate's default "GranularMean" method: all months weigh the mean
/// Gregorian month (30.436875 days) regardless of their actual length.
fn granularity_nominal_seconds(gran: &str) -> f64 {
  match gran {
    "Year" => 365.2425 * 86400.0,
    "Quarter" => 365.2425 / 4.0 * 86400.0,
    "Month" => 30.436875 * 86400.0,
    "Week" => 7.0 * 86400.0,
    "Day" => 86400.0,
    "Hour" => 3600.0,
    "Minute" => 60.0,
    "Second" => 1.0,
    _ => 0.0, // "Instant"
  }
}

/// The implicit interval covered by a raw component list with the given (or
/// implied) granularity, as (start, end, nominal_weight) in absolute seconds.
fn date_component_interval(
  list: &Expr,
  gran: Option<&str>,
) -> Option<(f64, f64, f64)> {
  let comps = extract_date_components(list)?;
  if comps.is_empty() {
    return None;
  }
  let gran = gran.map(str::to_string).unwrap_or_else(|| {
    granularity_from_component_count(comps.len()).to_string()
  });
  let mut y = comps[0] as i64;
  let mut m = if comps.len() > 1 { comps[1] as i64 } else { 1 };
  let mut d = if comps.len() > 2 { comps[2] as i64 } else { 1 };
  let h = if comps.len() > 3 { comps[3] as i64 } else { 0 };
  let mi = if comps.len() > 4 { comps[4] as i64 } else { 0 };
  let s = if comps.len() > 5 { comps[5] } else { 0.0 };
  match gran.as_str() {
    // A Week-granular date denotes the week containing it, starting Monday.
    "Week" => {
      let abs = date_to_absolute_days(y, m, d) - day_of_week(y, m, d);
      let (ny, nm, nd) = absolute_days_to_date(abs);
      y = ny;
      m = nm;
      d = nd;
    }
    // A Quarter-granular date denotes the quarter containing its month.
    "Quarter" => {
      m = (m - 1) / 3 * 3 + 1;
      d = 1;
    }
    _ => {}
  }
  let start = date_to_absolute_seconds(y, m, d, h, mi, s);
  let end = start + granularity_actual_seconds(y, m, &gran);
  Some((start, end, granularity_nominal_seconds(&gran)))
}

/// The implicit time interval covered by a single date specification
/// (DateObject, DateInterval, date list, or date string), as
/// (start, end, nominal_weight) in absolute seconds since 1900-01-01.
fn date_spec_interval(expr: &Expr) -> Option<(f64, f64, f64)> {
  let evaluated = crate::evaluator::evaluate_expr_to_expr(expr).ok()?;
  match &evaluated {
    // Canonical form: DateInterval[{{start, end}, …}, gran, cal, tz].
    // The interval runs from the start of its first date to the *end* of
    // its last granular unit (a Day interval covers the whole end day).
    Expr::FunctionCall { name, args }
      if name == "DateInterval" && !args.is_empty() =>
    {
      let gran = match args.get(1) {
        Some(Expr::String(g)) | Some(Expr::Identifier(g))
          if is_granularity_name(g) =>
        {
          g.clone()
        }
        _ => "Day".to_string(),
      };
      let Expr::List(pairs) = &args[0] else {
        return None;
      };
      let mut lo = f64::INFINITY;
      let mut hi = f64::NEG_INFINITY;
      for pair in pairs.iter() {
        let Expr::List(se) = pair else {
          return None;
        };
        if se.len() != 2 {
          return None;
        }
        let (start, _, _) = date_component_interval(&se[0], Some(&gran))?;
        let (_, end, _) = date_component_interval(&se[1], Some(&gran))?;
        lo = lo.min(start);
        hi = hi.max(end);
      }
      if !lo.is_finite() || !hi.is_finite() {
        return None;
      }
      Some((lo, hi, hi - lo))
    }
    Expr::FunctionCall { name, args }
      if name == "DateObject" && !args.is_empty() =>
    {
      let gran = match args.get(1) {
        Some(Expr::String(g)) | Some(Expr::Identifier(g))
          if is_granularity_name(g) =>
        {
          Some(g.as_str())
        }
        _ => None,
      };
      date_component_interval(&args[0], gran)
    }
    Expr::List(_) => date_component_interval(&evaluated, None),
    Expr::String(s) => {
      let (y, m, d) = parse_date_string(s)?;
      let start = date_to_absolute_seconds(y, m, d, 0, 0, 0.0);
      Some((start, start + 86400.0, 86400.0))
    }
    _ => None,
  }
}

/// Build the DateObject for an instant truncated to the given granularity.
fn instant_to_granular_date_object(seconds: f64, gran: &str) -> Expr {
  // Round to microseconds so float noise cannot flip a calendar boundary.
  let seconds = (seconds * 1e6).round() / 1e6;
  let (y, m, d, h, mi, s) = absolute_seconds_to_date(seconds);
  let int = |v: i64| Expr::Integer(v as i128);
  let with_time_zone = |fields: Vec<Expr>, gran: &str| Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![
      Expr::List(fields.into()),
      Expr::String(gran.to_string()),
      Expr::String("Gregorian".to_string()),
      Expr::Real(0.0),
    ]
    .into(),
  };
  let date_only = |fields: Vec<Expr>, gran: &str| Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![Expr::List(fields.into()), Expr::String(gran.to_string())]
      .into(),
  };
  match gran {
    "Year" => date_only(vec![int(y)], "Year"),
    "Quarter" => date_only(vec![int(y), int((m - 1) / 3 * 3 + 1)], "Quarter"),
    "Month" => date_only(vec![int(y), int(m)], "Month"),
    "Week" => {
      let abs = date_to_absolute_days(y, m, d) - day_of_week(y, m, d);
      let (wy, wm, wd) = absolute_days_to_date(abs);
      date_only(vec![int(wy), int(wm), int(wd)], "Week")
    }
    "Day" => date_only(vec![int(y), int(m), int(d)], "Day"),
    "Hour" => with_time_zone(vec![int(y), int(m), int(d), int(h)], "Hour"),
    "Minute" => {
      with_time_zone(vec![int(y), int(m), int(d), int(h), int(mi)], "Minute")
    }
    "Second" => with_time_zone(
      vec![
        int(y),
        int(m),
        int(d),
        int(h),
        int(mi),
        int(s.floor() as i64),
      ],
      "Second",
    ),
    _ => with_time_zone(
      vec![int(y), int(m), int(d), int(h), int(mi), Expr::Real(s)],
      "Instant",
    ),
  }
}

/// MidDate[datespec] — the instant at the middle of the implicit interval
///   covered by a date (a "Month" date covers its whole month, and so on).
/// MidDate[{date1, date2, …}] — the mean of the dates' interval midpoints,
///   weighted by the nominal length of each date's granularity (Wolfram's
///   default "GranularMean" method); instants get an unweighted mean.
/// MidDate[dateint] — the midpoint of a DateInterval.
/// MidDate[datespec, gran] — the result truncated to calendar granularity
///   gran ("Year", "Quarter", "Month", "Week", "Day", "Hour", "Minute",
///   "Second", or "Instant").
/// MidDate[datespec, gran, x] — the date at fraction x (instead of 1/2) of
///   the interval; for a list of dates x spans their overall bounds.
pub fn mid_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("MidDate", args));
  if args.is_empty() || args.len() > 3 {
    return unevaluated();
  }

  let out_gran: Option<String> = match args.get(1) {
    None => None,
    Some(g) => match &crate::evaluator::evaluate_expr_to_expr(g)? {
      Expr::String(s) | Expr::Identifier(s) if is_granularity_name(s) => {
        Some(s.clone())
      }
      Expr::Identifier(s) if s == "Automatic" => None,
      _ => return unevaluated(),
    },
  };

  let fraction: Option<f64> = match args.get(2) {
    None => None,
    Some(x) => match &crate::evaluator::evaluate_expr_to_expr(x)? {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(r) => Some(*r),
      Expr::FunctionCall { name, args: rargs }
        if name == "Rational" && rargs.len() == 2 =>
      {
        match (&rargs[0], &rargs[1]) {
          (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
            Some(*p as f64 / *q as f64)
          }
          _ => return unevaluated(),
        }
      }
      _ => return unevaluated(),
    },
  };

  let date_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;

  // A list argument is a collection of dates unless every element is a
  // number, in which case it is a single date list like {2024, 10, 1}.
  let specs: Vec<Expr> = match &date_arg {
    Expr::List(items)
      if !items.is_empty()
        && !items
          .iter()
          .all(|e| matches!(e, Expr::Integer(_) | Expr::Real(_))) =>
    {
      items.to_vec()
    }
    Expr::Association(pairs) if !pairs.is_empty() => {
      pairs.iter().map(|(_, v)| v.clone()).collect()
    }
    other => vec![other.clone()],
  };

  let mut intervals: Vec<(f64, f64, f64)> = Vec::with_capacity(specs.len());
  for spec in &specs {
    match date_spec_interval(spec) {
      Some(iv) => intervals.push(iv),
      None => return unevaluated(),
    }
  }

  let point = if let Some(x) = fraction {
    // Fraction x of the overall bounds (for a single date, its interval).
    let lo = intervals
      .iter()
      .map(|iv| iv.0)
      .fold(f64::INFINITY, f64::min);
    let hi = intervals
      .iter()
      .map(|iv| iv.1)
      .fold(f64::NEG_INFINITY, f64::max);
    lo + x * (hi - lo)
  } else if intervals.len() == 1 {
    let (start, end, _) = intervals[0];
    (start + end) / 2.0
  } else {
    // "GranularMean": midpoints weighted by nominal granularity length.
    let total_weight: f64 = intervals.iter().map(|iv| iv.2).sum();
    if total_weight > 0.0 {
      intervals
        .iter()
        .map(|(s, e, w)| (s + e) / 2.0 * w)
        .sum::<f64>()
        / total_weight
    } else {
      // All instants: plain mean.
      intervals.iter().map(|(s, e, _)| (s + e) / 2.0).sum::<f64>()
        / intervals.len() as f64
    }
  };

  Ok(instant_to_granular_date_object(
    point,
    out_gran.as_deref().unwrap_or("Instant"),
  ))
}

// ── Notebook-style DateObject panel ─────────────────────────────────────

/// Render `DateObject[…]` as the framed date panel Wolfram notebooks show:
/// a small calendar icon followed by the formatted date text and — for
/// instants carrying an explicit numeric time zone — a muted `GMT±h`
/// suffix. Returns `None` when the date cannot be formatted (symbolic
/// components), so those stay symbolic.
pub fn date_object_panel_svg(expr: &Expr) -> Option<String> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "DateObject" {
    return None;
  }
  let text = match date_string_ast(std::slice::from_ref(expr)) {
    Ok(Expr::String(ref s)) => s.clone(),
    _ => return None,
  };

  // Time-zone suffix, shown only when the date has a time of day and an
  // explicit numeric zone (like the notebook's `… 16:37:38 GMT+0`).
  let tz_label = match (args.first(), args.last()) {
    (Some(Expr::List(comps)), Some(tz))
      if args.len() >= 2 && comps.len() >= 4 =>
    {
      crate::functions::math_ast::try_eval_to_f64(tz).map(|z| {
        if z == z.trunc() {
          format!("GMT{}{}", if z < 0.0 { "-" } else { "+" }, z.abs() as i64)
        } else {
          format!("GMT{}{}", if z < 0.0 { "-" } else { "+" }, z.abs())
        }
      })
    }
    _ => None,
  };

  let theme = crate::functions::graphics::theme();
  let font_size = 14.0;
  // Atkinson Hyperlegible Mono (the monospace face all visual hosts map
  // `monospace` to) advances 0.632 em per glyph.
  let char_w = font_size * 0.632;
  let tz_font_size = 11.0;
  let tz_char_w = char_w * tz_font_size / font_size;
  let pad_x = 10.0;
  let icon_w = 14.0;
  let gap = 7.0;
  let height = 30.0;

  let text_w = text.chars().count() as f64 * char_w;
  let tz_w = tz_label
    .as_ref()
    .map(|t| 5.0 + t.chars().count() as f64 * tz_char_w)
    .unwrap_or(0.0);
  let width = (2.0 * pad_x + icon_w + gap + text_w + tz_w).ceil();

  let mut svg = format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
     viewBox=\"0 0 {w} {h}\">",
    w = width,
    h = height,
  );
  // Panel frame
  svg.push_str(&format!(
    "<rect x=\"0.5\" y=\"0.5\" width=\"{}\" height=\"{}\" rx=\"4\" \
     fill=\"{}\" stroke=\"{}\"/>",
    width - 1.0,
    height - 1.0,
    theme.table_header_bg,
    theme.framed_border,
  ));
  // Calendar icon: body, header separator, and two binder rings
  let ix = pad_x;
  let iy = (height - 14.0) / 2.0;
  svg.push_str(&format!(
    "<g stroke=\"{c}\" stroke-width=\"1.2\" fill=\"none\">\
     <rect x=\"{bx:.1}\" y=\"{by:.1}\" width=\"13\" height=\"11.5\" rx=\"1.5\"/>\
     <line x1=\"{bx:.1}\" y1=\"{hy:.1}\" x2=\"{bx2:.1}\" y2=\"{hy:.1}\"/>\
     <line x1=\"{r1:.1}\" y1=\"{ry:.1}\" x2=\"{r1:.1}\" y2=\"{ry2:.1}\"/>\
     <line x1=\"{r2:.1}\" y1=\"{ry:.1}\" x2=\"{r2:.1}\" y2=\"{ry2:.1}\"/>\
     </g>",
    c = theme.text_secondary,
    bx = ix + 0.5,
    by = iy + 2.5,
    bx2 = ix + 13.5,
    hy = iy + 5.5,
    r1 = ix + 4.0,
    r2 = ix + 10.0,
    ry = iy + 0.5,
    ry2 = iy + 4.0,
  ));
  // Date text (dates contain no XML-special characters). The time-zone
  // label rides along as a tspan so it always starts right after the date
  // glyphs, whatever monospace face the host actually resolves.
  let tx = pad_x + icon_w + gap;
  let tz_span = tz_label
    .map(|tz| {
      format!(
        "<tspan dx=\"5\" font-size=\"{tz_font_size}\" fill=\"{}\">{}</tspan>",
        theme.text_muted, tz,
      )
    })
    .unwrap_or_default();
  svg.push_str(&format!(
    "<text x=\"{tx:.1}\" y=\"{ty:.1}\" font-family=\"monospace\" \
     font-size=\"{font_size}\" fill=\"{}\" xml:space=\"preserve\">{}{}</text>",
    theme.text_primary,
    text,
    tz_span,
    ty = height / 2.0 + font_size * 0.35,
  ));
  svg.push_str("</svg>");
  Some(svg)
}

/// DateWithinQ[container, date] — True when `date`'s calendar span lies
/// inside `container`'s span. Spans are half-open, so the instant at a day's
/// closing midnight is not within that day, while its opening midnight is.
/// Instant containers (and TimeObject arguments) stay silently unevaluated;
/// non-date arguments emit `::arg` and echo.
pub fn date_within_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DateWithinQ", args));
  if args.len() != 2 {
    return unevaluated();
  }

  // Numeric calendar components of a date list (padded to y, mo, d, h, mi, s).
  fn comps(items: &[Expr]) -> Option<(i64, i64, i64, i64, i64, f64)> {
    if items.is_empty() || items.len() > 6 {
      return None;
    }
    let mut c = [0.0f64; 6];
    for (i, item) in items.iter().enumerate() {
      c[i] = match item {
        Expr::Integer(n) => *n as f64,
        Expr::Real(v) => *v,
        _ => return None,
      };
    }
    Some((
      c[0] as i64,
      if items.len() > 1 { c[1] as i64 } else { 1 },
      if items.len() > 2 { c[2] as i64 } else { 1 },
      if items.len() > 3 { c[3] as i64 } else { 0 },
      if items.len() > 4 { c[4] as i64 } else { 0 },
      if items.len() > 5 { c[5] } else { 0.0 },
    ))
  }

  // The instant that closes a span given its opening components and the
  // granularity level (1 = year … 6 = second/instant).
  fn unit_end(
    (y, mo, d, h, mi, s): (i64, i64, i64, i64, i64, f64),
    level: usize,
  ) -> f64 {
    match level {
      1 => date_to_absolute_seconds(y + 1, 1, 1, 0, 0, 0.0),
      2 => {
        let (ny, nm) = if mo == 12 { (y + 1, 1) } else { (y, mo + 1) };
        date_to_absolute_seconds(ny, nm, 1, 0, 0, 0.0)
      }
      3 => date_to_absolute_seconds(y, mo, d, h, mi, s) + 86400.0,
      4 => date_to_absolute_seconds(y, mo, d, h, mi, s) + 3600.0,
      5 => date_to_absolute_seconds(y, mo, d, h, mi, s) + 60.0,
      _ => date_to_absolute_seconds(y, mo, d, h, mi, s), // instant: a point
    }
  }

  // A DateInterval's granularity label mapped to its level.
  fn granularity_level(e: Option<&Expr>) -> usize {
    let name = match e {
      Some(Expr::String(s)) => s.as_str(),
      Some(Expr::Identifier(s)) => s.as_str(),
      _ => "Day",
    };
    match name {
      "Year" => 1,
      "Month" | "Quarter" | "Week" => 2,
      "Hour" => 4,
      "Minute" => 5,
      "Second" | "Instant" => 6,
      _ => 3,
    }
  }

  // The half-open [start, end) span of a DateObject (by component-list length)
  // or the overall span of a DateInterval (first sub-interval start … last
  // sub-interval end).
  fn span(e: &Expr) -> Option<(f64, f64)> {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    match name.as_str() {
      "DateObject" if !args.is_empty() => {
        let Expr::List(items) = &args[0] else {
          return None;
        };
        let c = comps(items)?;
        let start = date_to_absolute_seconds(c.0, c.1, c.2, c.3, c.4, c.5);
        Some((start, unit_end(c, items.len())))
      }
      "DateInterval" if !args.is_empty() => {
        let Expr::List(pairs) = &args[0] else {
          return None;
        };
        let level = granularity_level(args.get(1));
        let (mut min_start, mut max_end) = (f64::INFINITY, f64::NEG_INFINITY);
        for pair in pairs.iter() {
          let Expr::List(ends) = pair else {
            return None;
          };
          if ends.len() != 2 {
            return None;
          }
          let (Expr::List(sl), Expr::List(el)) = (&ends[0], &ends[1]) else {
            return None;
          };
          let sc = comps(sl)?;
          let ec = comps(el)?;
          let start =
            date_to_absolute_seconds(sc.0, sc.1, sc.2, sc.3, sc.4, sc.5);
          min_start = min_start.min(start);
          max_end = max_end.max(unit_end(ec, level));
        }
        if min_start.is_infinite() {
          return None;
        }
        Some((min_start, max_end))
      }
      _ => None,
    }
  }

  // Date-like but unresolvable arguments (TimeObject, instant containers)
  // stay silently unevaluated, matching wolframscript.
  let is_time_object = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "TimeObject");
  if is_time_object(&args[0]) || is_time_object(&args[1]) {
    return unevaluated();
  }
  let arg_error = |e: &Expr| {
    crate::emit_message(&format!(
      "DateWithinQ::arg: Argument {} is not a valid date object expression.",
      crate::syntax::expr_to_output(e)
    ));
  };
  let Some((s1, e1)) = span(&args[0]) else {
    arg_error(&args[0]);
    return unevaluated();
  };
  let Some((s2, e2)) = span(&args[1]) else {
    arg_error(&args[1]);
    return unevaluated();
  };
  // An instant container is a point; wolframscript leaves those calls
  // unevaluated rather than answering.
  if s1 == e1 {
    return unevaluated();
  }
  let within = s2 >= s1 && e2 <= e1 && s2 < e1;
  Ok(Expr::Identifier(
    if within { "True" } else { "False" }.to_string(),
  ))
}

/// DateSelect[list, crit] — the dates of `list` for which `crit[date]` is
/// True (exactly Select). DateSelect[DateInterval[...], crit] first enumerates
/// the interval's constituent days (Day granularity) as DateObjects. Other
/// granularities and non-list/interval arguments are left unevaluated.
pub fn date_select_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DateSelect", args));
  if args.len() != 2 {
    return unevaluated();
  }

  // Build the list of candidate dates.
  let list = match &args[0] {
    Expr::List(_) => args[0].clone(),
    Expr::FunctionCall { name, args: dargs }
      if name == "DateInterval" && dargs.len() >= 2 =>
    {
      // Only Day-granularity intervals are enumerated here.
      let is_day = matches!(&dargs[1], Expr::String(s) if s == "Day")
        || matches!(&dargs[1], Expr::Identifier(s) if s == "Day");
      if !is_day {
        return unevaluated();
      }
      let Expr::List(pairs) = &dargs[0] else {
        return unevaluated();
      };
      // Read the first three (y, m, d) components of a date list.
      let ymd = |e: &Expr| -> Option<(i64, i64, i64)> {
        let Expr::List(c) = e else { return None };
        if c.len() < 3 {
          return None;
        }
        let g = |x: &Expr| match x {
          Expr::Integer(n) => Some(*n as i64),
          Expr::Real(v) => Some(*v as i64),
          _ => None,
        };
        Some((g(&c[0])?, g(&c[1])?, g(&c[2])?))
      };
      let mut days: Vec<Expr> = Vec::new();
      for pair in pairs.iter() {
        let Expr::List(ends) = pair else {
          return unevaluated();
        };
        if ends.len() != 2 {
          return unevaluated();
        }
        let (Some(s), Some(e)) = (ymd(&ends[0]), ymd(&ends[1])) else {
          return unevaluated();
        };
        let start = date_to_absolute_days(s.0, s.1, s.2);
        let end = date_to_absolute_days(e.0, e.1, e.2);
        for abs in start..=end {
          let (y, m, d) = absolute_days_to_date(abs);
          let obj =
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "DateObject".to_string(),
              args: vec![Expr::List(
                vec![
                  Expr::Integer(y as i128),
                  Expr::Integer(m as i128),
                  Expr::Integer(d as i128),
                ]
                .into(),
              )]
              .into(),
            })?;
          days.push(obj);
        }
      }
      Expr::List(days.into())
    }
    _ => return unevaluated(),
  };

  // Filtering is exactly Select.
  crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Select".to_string(),
    args: vec![list, args[1].clone()].into(),
  })
}

/// DateOverlapsQ[date1, date2] — True when the calendar spans of two
/// DateObjects or DateIntervals share a common sub-span. Spans are half-open
/// [start, end) in absolute seconds, so adjacent calendar units (e.g. two
/// consecutive days) do not overlap, while intervals that share an interior
/// unit do. Non-date arguments (or instant points) stay unevaluated.
pub fn date_overlaps_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("DateOverlapsQ", args));
  if args.len() != 2 {
    return unevaluated();
  }

  // Numeric calendar components of a date list (padded to y, mo, d, h, mi, s).
  fn components(list: &[Expr]) -> Option<(i64, i64, i64, i64, i64, f64)> {
    if list.is_empty() || list.len() > 6 {
      return None;
    }
    let mut c = [0.0f64; 6];
    for (i, item) in list.iter().enumerate() {
      c[i] = match item {
        Expr::Integer(n) => *n as f64,
        Expr::Real(v) => *v,
        _ => return None,
      };
    }
    Some((
      c[0] as i64,
      if list.len() > 1 { c[1] as i64 } else { 1 },
      if list.len() > 2 { c[2] as i64 } else { 1 },
      if list.len() > 3 { c[3] as i64 } else { 0 },
      if list.len() > 4 { c[4] as i64 } else { 0 },
      if list.len() > 5 { c[5] } else { 0.0 },
    ))
  }

  // The instant that closes a span, given its opening components and the
  // granularity level (1 = year … 6 = second).
  fn end_of_unit(
    (y, mo, d, h, mi, s): (i64, i64, i64, i64, i64, f64),
    level: usize,
  ) -> f64 {
    match level {
      1 => date_to_absolute_seconds(y + 1, 1, 1, 0, 0, 0.0),
      2 => {
        let (ny, nm) = if mo == 12 { (y + 1, 1) } else { (y, mo + 1) };
        date_to_absolute_seconds(ny, nm, 1, 0, 0, 0.0)
      }
      3 => date_to_absolute_seconds(y, mo, d, h, mi, s) + 86400.0,
      4 => date_to_absolute_seconds(y, mo, d, h, mi, s) + 3600.0,
      5 => date_to_absolute_seconds(y, mo, d, h, mi, s) + 60.0,
      _ => date_to_absolute_seconds(y, mo, d, h, mi, s) + 1.0,
    }
  }

  // Map a granularity label to its level.
  fn granularity_level(e: Option<&Expr>) -> usize {
    let name = match e {
      Some(Expr::String(s)) => s.as_str(),
      Some(Expr::Identifier(s)) => s.as_str(),
      _ => "Day",
    };
    match name {
      "Year" => 1,
      "Month" | "Quarter" | "Week" => 2,
      "Hour" => 4,
      "Minute" => 5,
      "Second" | "Instant" => 6,
      _ => 3, // Day (default)
    }
  }

  // All half-open spans covered by a DateObject (one) or DateInterval (one
  // per sub-interval). `None` for anything that is not such an object.
  fn spans(e: &Expr) -> Option<Vec<(f64, f64)>> {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    match name.as_str() {
      "DateObject" if !args.is_empty() => {
        let Expr::List(items) = &args[0] else {
          return None;
        };
        let comps = components(items)?;
        let start = date_to_absolute_seconds(
          comps.0, comps.1, comps.2, comps.3, comps.4, comps.5,
        );
        Some(vec![(start, end_of_unit(comps, items.len()))])
      }
      "DateInterval" if !args.is_empty() => {
        let Expr::List(pairs) = &args[0] else {
          return None;
        };
        let level = granularity_level(args.get(1));
        let mut out = Vec::new();
        for pair in pairs.iter() {
          let Expr::List(ends) = pair else {
            return None;
          };
          if ends.len() != 2 {
            return None;
          }
          let (Expr::List(sl), Expr::List(el)) = (&ends[0], &ends[1]) else {
            return None;
          };
          let sc = components(sl)?;
          let ec = components(el)?;
          let start =
            date_to_absolute_seconds(sc.0, sc.1, sc.2, sc.3, sc.4, sc.5);
          out.push((start, end_of_unit(ec, level)));
        }
        if out.is_empty() {
          return None;
        }
        Some(out)
      }
      _ => None,
    }
  }

  let arg_error = |e: &Expr| {
    crate::emit_message(&format!(
      "DateOverlapsQ::arg: Argument {} is not a valid date object expression.",
      crate::syntax::expr_to_output(e)
    ));
  };
  let (Some(a), Some(b)) = (spans(&args[0]), spans(&args[1])) else {
    if spans(&args[0]).is_none() {
      arg_error(&args[0]);
    } else {
      arg_error(&args[1]);
    }
    return unevaluated();
  };
  // Two half-open spans overlap when they share positive-length interior.
  let overlaps = a
    .iter()
    .any(|&(s1, e1)| b.iter().any(|&(s2, e2)| s1.max(s2) < e1.min(e2)));
  Ok(Expr::Identifier(
    if overlaps { "True" } else { "False" }.to_string(),
  ))
}

/// FromJulianDate[jd] — the DateObject instant of a Julian date (days
/// since noon on -4714-11-24, proleptic Gregorian). Exact input gives
/// exact time components (integer seconds when whole); Real input gives a
/// Real seconds component. Non-numeric arguments emit ::arg.
pub fn from_julian_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("FromJulianDate", args));
  if args.len() != 1 {
    return unevaluated();
  }
  // jd as exact (num, den) plus realness; reals convert via their exact
  // binary fraction so the calendar arithmetic is exact either way.
  let (num, den, is_real): (i128, i128, bool) = match &args[0] {
    Expr::Integer(n) => (*n, 1, false),
    Expr::FunctionCall { name, args: ra }
      if name == "Rational" && ra.len() == 2 =>
    {
      match (&ra[0], &ra[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q > 0 => (*p, *q, false),
        _ => return unevaluated(),
      }
    }
    Expr::Real(v) if v.is_finite() => {
      let bits = v.to_bits();
      let sign = if bits >> 63 == 0 { 1i128 } else { -1 };
      let exp = ((bits >> 52) & 0x7ff) as i64;
      let frac = (bits & ((1u64 << 52) - 1)) as i128;
      let (m, e2) = if exp == 0 {
        (frac, -1074i64)
      } else {
        (frac + (1i128 << 52), exp - 1075)
      };
      if *v == 0.0 {
        (0, 1, true)
      } else if (0..70).contains(&e2) {
        (sign * (m << e2), 1, true)
      } else if e2 < 0 && e2 > -100 {
        (sign * m, 1i128 << (-e2), true)
      } else {
        return unevaluated();
      }
    }
    _ => {
      crate::emit_message(&format!(
        "FromJulianDate::arg: Argument {} cannot be interpreted as a Julian date input.",
        crate::syntax::expr_to_output(&args[0])
      ));
      return unevaluated();
    }
  };

  // Split jd + 1/2 into whole days (the Julian day number at the civil
  // date) and the fraction of the day since midnight.
  let shifted_num = num
    .checked_mul(2)
    .and_then(|x| x.checked_add(den))
    .map(|x| (x, den * 2));
  let Some((sn, sd)) = shifted_num else {
    return unevaluated();
  };
  let days = sn.div_euclid(sd);
  let frac_num = sn.rem_euclid(sd); // fraction = frac_num / sd of a day

  // Fliegel–Van Flandern: Julian day number → proleptic Gregorian date.
  let jdn = days;
  let a = jdn + 68569;
  let b = (4 * a).div_euclid(146097);
  let c = a - (146097 * b + 3).div_euclid(4);
  let d = (4000 * (c + 1)).div_euclid(1461001);
  let e = c - (1461 * d).div_euclid(4) + 31;
  let m = (80 * e).div_euclid(2447);
  let day = e - (2447 * m).div_euclid(80);
  let f = m.div_euclid(11);
  let month = m + 2 - 12 * f;
  let mut year = 100 * (b - 49) + d + f;
  // Fliegel–Van Flandern yields astronomical years (with a year 0);
  // wolframscript's DateObject uses historical numbering, so astronomical
  // 0 is -1, -4713 is -4714, and so on.
  if year <= 0 {
    year -= 1;
  }

  // Time of day from the exact fraction: seconds = frac * 86400.
  let total_secs_num = frac_num * 86400; // / sd
  let whole_secs = total_secs_num.div_euclid(sd);
  let rem_num = total_secs_num.rem_euclid(sd);
  let hour = whole_secs.div_euclid(3600);
  let minute = (whole_secs.rem_euclid(3600)).div_euclid(60);
  let sec_whole = whole_secs.rem_euclid(60);
  let second: Expr = if is_real {
    Expr::Real(sec_whole as f64 + rem_num as f64 / sd as f64)
  } else if rem_num == 0 {
    Expr::Integer(sec_whole)
  } else {
    let g = {
      let (mut x, mut y) = (rem_num, sd);
      while y != 0 {
        (x, y) = (y, x % y);
      }
      x.max(1)
    };
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Integer(sec_whole),
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(rem_num / g), Expr::Integer(sd / g)].into(),
        },
      ]
      .into(),
    })?
  };

  Ok(Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![
      Expr::List(
        vec![
          Expr::Integer(year),
          Expr::Integer(month),
          Expr::Integer(day),
          Expr::Integer(hour),
          Expr::Integer(minute),
          second,
        ]
        .into(),
      ),
      Expr::String("Instant".to_string()),
      Expr::String("Gregorian".to_string()),
      Expr::Real(0.0),
    ]
    .into(),
  })
}
