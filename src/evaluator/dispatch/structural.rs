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
      //
      // `RuntimeAttributes -> {Listable}` makes the *compiled function
      // itself* Listable — a call like `f[{1, 2, 3}, y]` then threads
      // element-wise over the list argument, the same as
      // `SetAttributes[f, Listable]` would for an ordinary function —
      // instead of passing the whole list into the body as one opaque
      // value. Woxi has no bytecode VM to honor `Parallelization` or
      // `RuntimeOptions`, so those are accepted (like real Mathematica
      // silently degrading on a machine that can't parallelize) but only
      // `Listable` changes evaluation. Record it as a 3rd `CompiledFunction`
      // arg — `apply_curried_call` reads it to decide whether to thread.
      let listable = args[2..].iter().any(|opt| {
        matches!(opt, Expr::Rule { pattern, replacement }
          if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "RuntimeAttributes")
            && rule_value_mentions_listable(replacement))
      });
      let mut compiled_args = vec![args[0].clone(), args[1].clone()];
      if listable {
        compiled_args.push(Expr::Identifier("Listable".to_string()));
      }
      return Some(Ok(call("CompiledFunction", compiled_args)));
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

/// Whether a `RuntimeAttributes -> …` value names `Listable`, either bare
/// (`RuntimeAttributes -> Listable`) or inside the usual attribute list
/// (`RuntimeAttributes -> {Listable}`).
fn rule_value_mentions_listable(value: &Expr) -> bool {
  match value {
    Expr::Identifier(n) => n == "Listable",
    Expr::List(items) => items
      .iter()
      .any(|i| matches!(i, Expr::Identifier(n) if n == "Listable")),
    _ => false,
  }
}
