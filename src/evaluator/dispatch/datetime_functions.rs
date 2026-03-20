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
    "DateDifference" if args.len() >= 2 => {
      return Some(crate::functions::datetime_ast::date_difference_ast(args));
    }
    "DateString" => {
      return Some(crate::functions::datetime_ast::date_string_ast(args));
    }
    "SessionTime" if args.is_empty() => {
      return Some(Ok(Expr::Real(crate::session_time())));
    }
    "DayName" if args.len() == 1 => {
      return Some(crate::functions::datetime_ast::day_name_ast(args));
    }
    // DateObject is a data container — return as-is
    "DateObject" => {
      return Some(Ok(Expr::FunctionCall {
        name: "DateObject".to_string(),
        args: args.to_vec(),
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
    _ => {}
  }
  None
}
