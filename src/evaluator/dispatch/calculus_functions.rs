#[allow(unused_imports)]
use super::*;

pub fn dispatch_calculus_functions(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Derivative" if args.len() == 3 => {
      if let (Expr::Integer(n), Expr::Identifier(func_name)) =
        (&args[0], &args[1])
      {
        let n = *n as usize;
        let overloads = crate::FUNC_DEFS.with(|m| {
          let defs = m.borrow();
          defs.get(func_name).cloned()
        });
        if let Some(overloads) = overloads {
          for (params, _conditions, _defaults, _heads, body_expr) in &overloads
          {
            if params.len() == 1 {
              let param = &params[0];
              let mut deriv = body_expr.clone();
              for _ in 0..n {
                deriv = match crate::functions::calculus_ast::differentiate_expr(
                  &deriv, param,
                ) {
                  Ok(v) => v,
                  Err(e) => return Some(Err(e)),
                };
              }
              let substituted =
                crate::syntax::substitute_variable(&deriv, param, &args[2]);
              return Some(evaluate_expr_to_expr(&substituted));
            }
          }
        }
      }
      return Some(Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      }));
    }
    "Derivative" if args.len() <= 2 => {
      return Some(Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      }));
    }
    "D" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::d_ast(args));
    }
    "Dt" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::dt_ast(args));
    }
    "Curl" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::curl_ast(args));
    }
    "Integrate" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::integrate_ast(args));
    }
    "NIntegrate" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::nintegrate_ast(args));
    }
    "Limit" if (2..=3).contains(&args.len()) => {
      return Some(crate::functions::calculus_ast::limit_ast(args));
    }
    "Series" if args.len() == 2 => {
      return Some(crate::functions::calculus_ast::series_ast(args));
    }
    _ => {}
  }
  None
}
