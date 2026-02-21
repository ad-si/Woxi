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
    _ => {}
  }
  None
}
