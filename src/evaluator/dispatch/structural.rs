#[allow(unused_imports)]
use super::*;
use crate::syntax::unevaluated;

pub fn dispatch_structural(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "Function" => match args.len() {
      1 => {
        return Some(Ok(Expr::Function {
          body: Box::new(args[0].clone()),
        }));
      }
      2 => {
        let (params, bracketed) = match &args[0] {
          Expr::Identifier(name) => (vec![name.clone()], false),
          Expr::List(items) => (
            items
              .iter()
              .filter_map(|item| {
                if let Expr::Identifier(n) = item {
                  Some(n.clone())
                } else {
                  None
                }
              })
              .collect(),
            true,
          ),
          _ => {
            return Some(Ok(unevaluated("Function", args)));
          }
        };
        return Some(Ok(Expr::NamedFunction {
          params,
          body: Box::new(args[1].clone()),
          bracketed,
        }));
      }
      _ => {
        return Some(Ok(unevaluated("Function", args)));
      }
    },
    "Compile" if args.len() == 2 => {
      let vars = match &args[0] {
        Expr::List(items) => {
          let mut var_names = Vec::new();
          for item in items {
            match item {
              Expr::Identifier(name) => var_names.push(name.clone()),
              Expr::List(inner) if !inner.is_empty() => {
                if let Expr::Identifier(name) = &inner[0] {
                  var_names.push(name.clone());
                }
              }
              _ => {}
            }
          }
          var_names
        }
        Expr::Identifier(name) => vec![name.clone()],
        _ => {
          return Some(Ok(unevaluated("Compile", args)));
        }
      };
      let var_exprs: Vec<Expr> =
        vars.iter().map(|v| Expr::Identifier(v.clone())).collect();
      return Some(Ok(Expr::FunctionCall {
        name: "CompiledFunction".to_string(),
        args: vec![Expr::List(var_exprs.into()), args[1].clone()].into(),
      }));
    }
    "Rational" if args.len() == 2 => {
      if let (Some(n), Some(d)) =
        (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
      {
        if d == 0 {
          return Some(Ok(Expr::Identifier("ComplexInfinity".to_string())));
        }
        return Some(Ok(crate::functions::math_ast::make_rational(n, d)));
      }
      return Some(Ok(unevaluated("Rational", args)));
    }
    "Module" => return Some(module_ast(args)),
    "Block" => return Some(block_ast(args)),
    "Assuming" if args.len() == 2 => return Some(assuming_ast(args)),
    "With" if args.len() == 2 => return Some(with_ast(args)),
    "Set" if args.len() == 2 => {
      return Some(set_ast(&args[0], &args[1]));
    }
    _ => {}
  }
  None
}
