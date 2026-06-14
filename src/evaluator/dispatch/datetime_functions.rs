#[allow(unused_imports)]
use super::*;

pub fn dispatch_datetime_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "AbsoluteTime" => {
      return Some(crate::functions::datetime_ast::absolute_time_ast(args));
    }
    "DateList" => {
      return Some(crate::functions::datetime_ast::date_list_ast(args));
    }
    "DatePlus" if args.len() == 2 => {
      return Some(crate::functions::datetime_ast::date_plus_ast(args));
    }
    "DayPlus" if args.len() >= 2 && args.len() <= 3 => {
      return Some(crate::functions::datetime_ast::day_plus_ast(args));
    }
    "DayRange" if args.len() == 2 || args.len() == 3 => {
      return Some(crate::functions::datetime_ast::day_range_ast(args));
    }
    "DateDifference" if args.len() >= 2 => {
      return Some(crate::functions::datetime_ast::date_difference_ast(args));
    }
    "DateString" => {
      return Some(crate::functions::datetime_ast::date_string_ast(args));
    }
    "SessionTime" if args.is_empty() => {
      return Some(Ok(Expr::Real(crate::session_time())));
    }
    "TimeUsed" if args.is_empty() => {
      return Some(Ok(Expr::Real(crate::functions::memory::cpu_time_used())));
    }
    "DayName" if args.len() == 1 => {
      return Some(crate::functions::datetime_ast::day_name_ast(args));
    }
    // DateInterval[{date1, date2}] — create a date interval
    "DateInterval" if args.len() == 1 => {
      if let Expr::List(dates) = &args[0]
        && dates.len() == 2
      {
        let d1 =
          crate::functions::datetime_ast::resolve_date_to_list(&dates[0]);
        let d2 =
          crate::functions::datetime_ast::resolve_date_to_list(&dates[1]);
        if let (Some(start), Some(end)) = (d1, d2) {
          return Some(Ok(Expr::FunctionCall {
            name: "DateInterval".to_string(),
            args: vec![
              Expr::List(vec![Expr::List(vec![start, end].into())].into()),
              Expr::String("Day".to_string()),
              Expr::String("Gregorian".to_string()),
              Expr::Identifier("None".to_string()),
            ]
            .into(),
          }));
        }
      }
      crate::emit_message(&format!(
        "DateInterval::arg: Argument {} cannot be interpreted as a date or time input.",
        crate::syntax::expr_to_string(&args[0])
      ));
      return Some(Ok(Expr::FunctionCall {
        name: "DateInterval".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // DateObject is a data container — normalize granularity
    "DateObject" => {
      // DateObject[] → current instant (same shape as `Now`)
      if args.is_empty() {
        return Some(crate::evaluator::evaluate_expr_to_expr(
          &Expr::Identifier("Now".to_string()),
        ));
      }
      // DateObject[n] — n is absolute seconds since Wolfram's epoch
      // (1900-01-01 00:00:00 UTC). Convert to a calendar instant.
      #[cfg(feature = "cli")]
      if args.len() == 1
        && let Some(secs) = match &args[0] {
          Expr::Integer(n) => Some(*n as f64),
          Expr::Real(r) => Some(*r),
          _ => None,
        }
      {
        let (y, mo, d, h, mi, s) = wolfram_seconds_to_ymdhms(secs);
        let date_list = Expr::List(
          vec![
            Expr::Integer(y as i128),
            Expr::Integer(mo as i128),
            Expr::Integer(d as i128),
            Expr::Integer(h as i128),
            Expr::Integer(mi as i128),
            Expr::Integer(s as i128),
          ]
          .into(),
        );
        return Some(Ok(Expr::FunctionCall {
          name: "DateObject".to_string(),
          args: vec![
            date_list,
            Expr::String("Instant".to_string()),
            Expr::String("Gregorian".to_string()),
            Expr::Real(0.0),
          ]
          .into(),
        }));
      }
      // DateObject[{y, ...}] adds a granularity tag based on list length.
      //   1 → Year, 2 → Month, 3 → Day
      //   4 → Hour, Gregorian, 0., 5 → Minute, Gregorian, 0.,
      //   6 → Instant, Gregorian, 0.
      if args.len() == 1
        && let Expr::List(items) = &args[0]
      {
        let mut extra: Vec<Expr> = Vec::new();
        match items.len() {
          1 => extra.push(Expr::String("Year".to_string())),
          2 => extra.push(Expr::String("Month".to_string())),
          3 => extra.push(Expr::String("Day".to_string())),
          4..=6 => {
            let gran = match items.len() {
              4 => "Hour",
              5 => "Minute",
              _ => "Instant",
            };
            extra.push(Expr::String(gran.to_string()));
            extra.push(Expr::String("Gregorian".to_string()));
            extra.push(Expr::Real(0.0));
          }
          _ => {
            return Some(Ok(Expr::FunctionCall {
              name: "DateObject".to_string(),
              args: args.to_vec().into(),
            }));
          }
        }
        let mut new_args = vec![args[0].clone()];
        new_args.extend(extra);
        return Some(Ok(Expr::FunctionCall {
          name: "DateObject".to_string(),
          args: new_args.into(),
        }));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "DateObject".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // DayCount[date1, date2] — number of days between two dates
    "DayCount" if args.len() == 2 => {
      // Use DateDifference which returns Quantity[n, "Days"]
      let result = crate::functions::datetime_ast::date_difference_ast(args);
      if let Ok(Expr::FunctionCall {
        name: ref qname,
        args: ref qargs,
      }) = result
        && qname == "Quantity"
        && qargs.len() == 2
      {
        return Some(Ok(qargs[0].clone()));
      }
      return Some(result);
    }
    "UnixTime" if args.is_empty() => {
      use std::time::{SystemTime, UNIX_EPOCH};
      let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
      return Some(Ok(Expr::Integer(secs as i128)));
    }
    // UnixTime[date] — seconds since the Unix epoch for a date list or a
    // DateObject. A DateObject's embedded time-zone offset (its last element,
    // when a Real) shifts the calendar time back to UTC. Computed from
    // AbsoluteTime (Wolfram epoch is 2_208_988_800 s before the Unix epoch).
    "UnixTime" if args.len() == 1 => {
      const WOLFRAM_EPOCH_TO_UNIX: f64 = 2_208_988_800.0;
      // Resolve the date spec and any time-zone offset (in hours).
      let (date_spec, tz) = match &args[0] {
        Expr::FunctionCall { name, args: dargs }
          if name == "DateObject" && !dargs.is_empty() =>
        {
          let tz = match dargs.last() {
            Some(Expr::Real(r)) => *r,
            Some(Expr::Integer(n)) => *n as f64,
            _ => 0.0,
          };
          (dargs[0].clone(), tz)
        }
        other => (other.clone(), 0.0),
      };
      let abs =
        crate::functions::datetime_ast::absolute_time_ast(&[date_spec.clone()]);
      let abs_secs = match abs {
        Ok(Expr::Integer(n)) => Some(n as f64),
        Ok(Expr::Real(r)) => Some(r),
        _ => None,
      };
      if let Some(abs_secs) = abs_secs {
        let unix = abs_secs - WOLFRAM_EPOCH_TO_UNIX - tz * 3600.0;
        return Some(Ok(Expr::Integer(unix.round() as i128)));
      }
      return Some(Ok(Expr::FunctionCall {
        name: "UnixTime".to_string(),
        args: args.to_vec().into(),
      }));
    }
    // FromUnixTime[n] / FromUnixTime[n, TimeZone -> tz] — the instant `n`
    // seconds after the Unix epoch, as a DateObject. The displayed calendar
    // time is shifted by the time-zone offset (in hours); without the option
    // the local offset is used.
    "FromUnixTime" if args.len() == 1 || args.len() == 2 => {
      let Some(unix_secs) = (match &args[0] {
        Expr::Integer(n) => Some(*n as f64),
        Expr::Real(r) => Some(*r),
        _ => None,
      }) else {
        return Some(Ok(Expr::FunctionCall {
          name: "FromUnixTime".to_string(),
          args: args.to_vec().into(),
        }));
      };
      // Resolve the time-zone offset in hours.
      let tz = if args.len() == 2 {
        match parse_timezone_option(&args[1]) {
          Some(tz) => tz,
          None => {
            return Some(Ok(Expr::FunctionCall {
              name: "FromUnixTime".to_string(),
              args: args.to_vec().into(),
            }));
          }
        }
      } else {
        local_timezone_offset_hours()
      };
      #[cfg(feature = "cli")]
      {
        const WOLFRAM_EPOCH_TO_UNIX: f64 = 2_208_988_800.0;
        let wolfram_secs = unix_secs + WOLFRAM_EPOCH_TO_UNIX + tz * 3600.0;
        let (y, mo, d, h, mi, s) = wolfram_seconds_to_ymdhms(wolfram_secs);
        let date_list = Expr::List(
          vec![
            Expr::Integer(y as i128),
            Expr::Integer(mo as i128),
            Expr::Integer(d as i128),
            Expr::Integer(h as i128),
            Expr::Integer(mi as i128),
            Expr::Integer(s as i128),
          ]
          .into(),
        );
        return Some(Ok(Expr::FunctionCall {
          name: "DateObject".to_string(),
          args: vec![
            date_list,
            Expr::String("Instant".to_string()),
            Expr::String("Gregorian".to_string()),
            Expr::Real(tz),
          ]
          .into(),
        }));
      }
      #[cfg(not(feature = "cli"))]
      {
        let _ = (unix_secs, tz);
        return Some(Ok(Expr::FunctionCall {
          name: "FromUnixTime".to_string(),
          args: args.to_vec().into(),
        }));
      }
    }
    "JulianDate" if args.len() <= 1 => {
      return Some(crate::functions::datetime_ast::julian_date_ast(args));
    }
    _ => {}
  }
  None
}

/// Convert Wolfram's absolute-seconds count (seconds since
/// 1900-01-01 00:00:00 UTC, no leap seconds) to a calendar
/// `(year, month, day, hour, minute, second)` tuple in UTC. Uses
/// `chrono` via the Unix-epoch offset (Wolfram epoch is 70 calendar
/// years and 17 leap days before Unix epoch — 2_208_988_800 seconds).
#[cfg(feature = "cli")]
fn wolfram_seconds_to_ymdhms(secs: f64) -> (i64, u32, u32, u32, u32, u32) {
  use chrono::{Datelike, TimeZone, Timelike, Utc};
  const WOLFRAM_EPOCH_TO_UNIX: f64 = 2_208_988_800.0;
  let unix_secs = secs - WOLFRAM_EPOCH_TO_UNIX;
  let total = unix_secs.round() as i64;
  let dt = Utc
    .timestamp_opt(total, 0)
    .single()
    .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().unwrap());
  (
    dt.year() as i64,
    dt.month(),
    dt.day(),
    dt.hour(),
    dt.minute(),
    dt.second(),
  )
}

/// Parse a `TimeZone -> tz` option, returning the offset in hours. Accepts the
/// `Expr::Rule` and `Rule[...]`/`RuleDelayed[...]` forms with an Integer, Real,
/// or Rational right-hand side.
fn parse_timezone_option(opt: &Expr) -> Option<f64> {
  let (lhs, rhs) = match opt {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref()),
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      (&args[0], &args[1])
    }
    _ => return None,
  };
  if !matches!(lhs, Expr::Identifier(s) if s == "TimeZone") {
    return None;
  }
  match rhs {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
          Some(*p as f64 / *q as f64)
        }
        _ => None,
      }
    }
    _ => None,
  }
}

/// The local system time-zone offset in hours, used as FromUnixTime's default.
/// Falls back to 0 (UTC) outside the `cli` feature.
fn local_timezone_offset_hours() -> f64 {
  #[cfg(feature = "cli")]
  {
    use chrono::{Local, Offset};
    Local::now().offset().fix().local_minus_utc() as f64 / 3600.0
  }
  #[cfg(not(feature = "cli"))]
  {
    0.0
  }
}
