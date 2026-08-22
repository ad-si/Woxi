#[allow(unused_imports)]
use super::*;

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
    "Compile" if args.len() >= 2 => {
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
      let _ = &vars;
      // Keep the argument specs verbatim so the declared element types
      // survive: `{s, _Integer, 0}` must bind an integer, not the Real every
      // argument used to be coerced to. A bare-name spec list (`{x, y}`) is
      // already its own name list, so the untyped form is unchanged.
      return Some(Ok(call(
        "CompiledFunction",
        vec![args[0].clone(), args[1].clone()],
      )));
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
    // `DynamicModule` scopes its locals the way `Module` does. Wolfram
    // keeps the wrapper around the result because the front end owns the
    // local state; we hand back the body's value instead, so every
    // display path (a Grid, a Graphics, a Column) renders it as itself.
    "Module" | "DynamicModule" => return Some(module_ast(args)),
    "Block" => return Some(block_ast(args)),
    "BlockRandom" => return Some(block_random_ast(args)),
    "Assuming" if args.len() == 2 => return Some(assuming_ast(args)),
    "With" if args.len() >= 2 => return Some(with_ast(args)),
    "Set" if args.len() == 2 => {
      return Some(set_ast(&args[0], &args[1]));
    }
    _ => {}
  }
  None
}
