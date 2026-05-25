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
          4 | 5 | 6 => {
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
    _ => {}
  }
  None
}
