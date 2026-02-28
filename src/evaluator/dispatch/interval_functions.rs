#[allow(unused_imports)]
use super::*;

pub fn dispatch_interval_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Interval" => {
      return Some(crate::functions::interval_ast::interval_ast(args));
    }
    "IntervalUnion" => {
      return Some(crate::functions::interval_ast::interval_union_ast(args));
    }
    "IntervalIntersection" => {
      return Some(crate::functions::interval_ast::interval_intersection_ast(
        args,
      ));
    }
    "IntervalMemberQ" => {
      return Some(crate::functions::interval_ast::interval_member_q_ast(args));
    }
    _ => {}
  }
  None
}
