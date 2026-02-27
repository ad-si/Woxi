//! AST-native date/time functions.

use crate::InterpreterError;
use crate::syntax::Expr;

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
fn date_to_absolute_seconds(
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
fn absolute_seconds_to_date(
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
fn extract_date_components(expr: &Expr) -> Option<Vec<f64>> {
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
        months.iter().position(|m| month_lower.starts_with(m))
        && let Ok(year) = parts[2].parse::<i64>()
      {
        return Some((year, (month_idx + 1) as i64, day));
      }
    }
    // Try "Month Day Year" format
    let month_lower = parts[0].to_lowercase();
    if let Some(month_idx) =
      months.iter().position(|m| month_lower.starts_with(m))
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
      args: args.to_vec(),
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
          args: vec![arg],
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
          args: vec![arg],
        })
      }
    }
    Expr::Integer(n) => {
      // AbsoluteTime[n] returns n (already absolute time)
      Ok(Expr::Integer(*n))
    }
    _ => Ok(Expr::FunctionCall {
      name: "AbsoluteTime".to_string(),
      args: vec![arg],
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
      args: args.to_vec(),
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
          args: vec![arg],
        })
      }
    }
    Expr::String(s) => {
      if let Some((y, m, d)) = parse_date_string(s) {
        Ok(make_date_list(y, m, d, 0, 0, 0.0))
      } else {
        Ok(Expr::FunctionCall {
          name: "DateList".to_string(),
          args: vec![arg],
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "DateList".to_string(),
      args: vec![arg],
    }),
  }
}

fn make_date_list(y: i64, m: i64, d: i64, h: i64, min: i64, sec: f64) -> Expr {
  // Round seconds to avoid floating point artifacts
  let sec_rounded = (sec * 100.0).round() / 100.0;
  Expr::List(vec![
    Expr::Integer(y as i128),
    Expr::Integer(m as i128),
    Expr::Integer(d as i128),
    Expr::Integer(h as i128),
    Expr::Integer(min as i128),
    Expr::Real(sec_rounded),
  ])
}

/// DatePlus[date, n] — add n days to a date
/// DatePlus[date, {{n1, "unit1"}, ...}] — add with units
pub fn date_plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "DatePlus".to_string(),
      args: args.to_vec(),
    });
  }

  let date_arg = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let delta_arg = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  let components = match extract_date_components(&date_arg) {
    Some(c) => c,
    None => {
      return Ok(Expr::FunctionCall {
        name: "DatePlus".to_string(),
        args: vec![date_arg, delta_arg],
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
      // List of {n, "unit"} pairs
      let mut total = 0i64;
      for item in items {
        if let Expr::List(pair) = item
          && pair.len() == 2
        {
          let n = match &pair[0] {
            Expr::Integer(n) => *n as i64,
            Expr::Real(f) => *f as i64,
            _ => 0,
          };
          let unit = match &pair[1] {
            Expr::String(s) => s.clone(),
            Expr::Identifier(s) => s.clone(),
            _ => String::new(),
          };
          total += match unit.as_str() {
            "Day" => n,
            "Week" => n * 7,
            "Month" => {
              // Add months by adjusting the date
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
                new_year, new_month, new_day, input_len,
              ));
            }
            "Year" => {
              let new_year = year + n;
              let new_day = day.min(days_in_month(new_year, month));
              return Ok(make_date_result(new_year, month, new_day, input_len));
            }
            _ => n,
          };
        }
      }
      total
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DatePlus".to_string(),
        args: vec![date_arg, delta_arg],
      });
    }
  };

  let abs_days = date_to_absolute_days(year, month, day) + total_days;
  let (ny, nm, nd) = absolute_days_to_date(abs_days);
  Ok(make_date_result(ny, nm, nd, input_len))
}

fn make_date_result(y: i64, m: i64, d: i64, input_len: usize) -> Expr {
  if input_len <= 3 {
    Expr::List(vec![
      Expr::Integer(y as i128),
      Expr::Integer(m as i128),
      Expr::Integer(d as i128),
    ])
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
      args: args.to_vec(),
    });
  }

  let date1 = crate::evaluator::evaluate_expr_to_expr(&args[0])?;
  let date2 = crate::evaluator::evaluate_expr_to_expr(&args[1])?;

  let c1 = match extract_date_components(&date1) {
    Some(c) => c,
    None => {
      return Ok(Expr::FunctionCall {
        name: "DateDifference".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let c2 = match extract_date_components(&date2) {
    Some(c) => c,
    None => {
      return Ok(Expr::FunctionCall {
        name: "DateDifference".to_string(),
        args: args.to_vec(),
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
      args: vec![n, Expr::String("Days".to_string())],
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
        args: args.to_vec(),
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
    args: vec![n, Expr::String(unit_name.to_string())],
  })
}

fn date_difference_multi_unit(
  _c1: &[f64],
  _c2: &[f64],
  _units: &Expr,
) -> Result<Expr, InterpreterError> {
  // Multi-unit differences like {"Month", "Day"} are complex
  // Return unevaluated for now
  Ok(Expr::FunctionCall {
    name: "DateDifference".to_string(),
    args: vec![],
  })
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

  // Extract date components — handle DateObject[{y,m,d,...}, ...] by extracting first arg
  let date_expr = if let Expr::FunctionCall { name, args: dargs } = &date_arg {
    if name == "DateObject" && !dargs.is_empty() {
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
      return Ok(Expr::FunctionCall {
        name: "DateString".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let (y, m, d, h, min, sec) = normalize_date(&components);

  if args.len() == 1 {
    // Default format: "DayNameShort DD MonthNameShort YYYY HH:MM:SS"
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

  let fmt_arg = crate::evaluator::evaluate_expr_to_expr(&args[1])?;
  let format_specs = match &fmt_arg {
    Expr::List(items) => items.clone(),
    Expr::String(s) => {
      // Named date format specifications
      match s.as_str() {
        "ISODateTime" => vec![
          Expr::String("Year".to_string()),
          Expr::String("-".to_string()),
          Expr::String("Month".to_string()),
          Expr::String("-".to_string()),
          Expr::String("Day2".to_string()),
          Expr::String("T".to_string()),
          Expr::String("Hour".to_string()),
          Expr::String(":".to_string()),
          Expr::String("Minute".to_string()),
          Expr::String(":".to_string()),
          Expr::String("Second".to_string()),
        ],
        "ISODate" => vec![
          Expr::String("Year".to_string()),
          Expr::String("-".to_string()),
          Expr::String("Month".to_string()),
          Expr::String("-".to_string()),
          Expr::String("Day2".to_string()),
        ],
        "DateTime" => vec![
          Expr::String("DayName".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Day".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("MonthName".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Year".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Hour".to_string()),
          Expr::String(":".to_string()),
          Expr::String("Minute".to_string()),
          Expr::String(":".to_string()),
          Expr::String("Second".to_string()),
        ],
        "DateTimeShort" => vec![
          Expr::String("DayNameShort".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Day".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("MonthNameShort".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Year".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Hour".to_string()),
          Expr::String(":".to_string()),
          Expr::String("Minute".to_string()),
          Expr::String(":".to_string()),
          Expr::String("Second".to_string()),
        ],
        "Date" => vec![
          Expr::String("DayName".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Day".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("MonthName".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Year".to_string()),
        ],
        "DateShort" => vec![
          Expr::String("DayNameShort".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Day".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("MonthNameShort".to_string()),
          Expr::String(" ".to_string()),
          Expr::String("Year".to_string()),
        ],
        "Time" => vec![
          Expr::String("Hour".to_string()),
          Expr::String(":".to_string()),
          Expr::String("Minute".to_string()),
          Expr::String(":".to_string()),
          Expr::String("Second".to_string()),
        ],
        // Single format element as a string (e.g. DateString[Now, "Year"])
        _ => vec![Expr::String(s.clone())],
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DateString".to_string(),
        args: args.to_vec(),
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
        "Day" => result.push_str(&format!("{}", d)),
        "Day2" => result.push_str(&format!("{:02}", d)),
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
