#[allow(unused_imports)]
use super::*;

/// MapAll[f, expr] - apply f to every subexpression in expr (bottom-up)
pub fn map_all_ast(f: &Expr, expr: &Expr) -> Result<Expr, InterpreterError> {
  // First, recursively apply to subexpressions
  let mapped = match expr {
    Expr::FunctionCall { name, args } => {
      // Map over the arguments
      let new_args: Vec<Expr> = args
        .iter()
        .map(|a| map_all_ast(f, a))
        .collect::<Result<Vec<_>, _>>()?;
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      }
    }
    Expr::List(items) => {
      let new_items: Vec<Expr> = items
        .iter()
        .map(|a| map_all_ast(f, a))
        .collect::<Result<Vec<_>, _>>()?;
      Expr::List(new_items)
    }
    // Atoms: just return as-is (will be wrapped by f below)
    _ => expr.clone(),
  };

  // Then apply f to the result
  apply_function_to_arg(f, &mapped)
}

/// AST-based Distribute implementation
/// Distribute[f[x1, x2, ...]] distributes f over Plus in the xi
/// Distribute[expr, g] distributes over g instead of Plus
/// Distribute[expr, g, f] only distributes if outer head is f
pub fn distribute_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let expr = &args[0];

  // Determine the inner head to distribute over (default: Plus/addition)
  let distribute_over = if args.len() >= 2 {
    match &args[1] {
      Expr::Identifier(name) => name.clone(),
      _ => "Plus".to_string(),
    }
  } else {
    "Plus".to_string()
  };

  // Get the outer function call
  let (outer_name, outer_args) = match expr {
    Expr::FunctionCall { name, args } => (name.clone(), args.clone()),
    _ => {
      // Not a function call - return as-is
      return Ok(expr.clone());
    }
  };

  // If 3 args provided, check that outer head matches
  if args.len() == 3 {
    let required_head = match &args[2] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Distribute".to_string(),
          args: args.to_vec(),
        });
      }
    };
    if outer_name != required_head {
      return Ok(expr.clone());
    }
  }

  // Split each argument into its parts based on distribute_over head
  let mut arg_lists: Vec<Vec<Expr>> = Vec::new();
  for arg in &outer_args {
    let parts = split_by_head(arg, &distribute_over);
    arg_lists.push(parts);
  }

  // Compute cartesian product
  let mut combinations: Vec<Vec<Expr>> = vec![vec![]];
  for parts in &arg_lists {
    let mut new_combinations = Vec::new();
    for combo in &combinations {
      for part in parts {
        let mut new_combo = combo.clone();
        new_combo.push(part.clone());
        new_combinations.push(new_combo);
      }
    }
    combinations = new_combinations;
  }

  // Build result: wrap each combination in outer_name, then combine with distribute_over
  let terms: Vec<Expr> = combinations
    .into_iter()
    .map(|combo| Expr::FunctionCall {
      name: outer_name.clone(),
      args: combo,
    })
    .collect();

  if terms.len() == 1 {
    return evaluate_expr_to_expr(&terms[0]);
  }

  let result = if distribute_over == "List" {
    Expr::List(terms)
  } else {
    Expr::FunctionCall {
      name: distribute_over,
      args: terms,
    }
  };

  evaluate_expr_to_expr(&result)
}

/// Split an expression by its head. E.g., split_by_head(a + b, "Plus") = [a, b]
pub fn split_by_head(expr: &Expr, head: &str) -> Vec<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == head => args.clone(),
    _ => vec![expr.clone()],
  }
}

/// Apply Map operation on AST (func /@ list)
pub fn apply_map_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_function_to_arg(func, item))
        .collect();
      Ok(Expr::List(results?))
    }
    Expr::Association(items) => {
      // Map over association applies function to values only
      let results: Result<Vec<(Expr, Expr)>, InterpreterError> = items
        .iter()
        .map(|(key, val)| {
          let new_val = apply_function_to_arg(func, val)?;
          Ok((key.clone(), new_val))
        })
        .collect();
      Ok(Expr::Association(results?))
    }
    _ => {
      // Not a list or association, return unevaluated
      Ok(Expr::Map {
        func: Box::new(func.clone()),
        list: Box::new(list.clone()),
      })
    }
  }
}

/// Apply Apply operation on AST (func @@ list)
pub fn apply_apply_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    // Apply replaces the head of any expression: f @@ Plus[a, b, c] → f[a, b, c]
    Expr::FunctionCall { args, .. } => args.clone(),
    _ => {
      // Not a list or function call, return unevaluated
      return Ok(Expr::Apply {
        func: Box::new(func.clone()),
        list: Box::new(list.clone()),
      });
    }
  };

  // Apply converts List[a, b, c] to func[a, b, c]
  match func {
    Expr::Identifier(func_name) => {
      // Resolve variable holding a function name: f = Plus; f @@ {1,2} → 3
      if let Some(resolved) = resolve_identifier_to_func_name(func_name) {
        return evaluate_function_call_ast(&resolved, &items);
      }
      evaluate_function_call_ast(func_name, &items)
    }
    Expr::FunctionCall {
      name: func_name, ..
    } => evaluate_function_call_ast(func_name, &items),
    Expr::Function { body } => {
      // Anonymous function applied to a list
      // For single-arg anonymous functions, apply to first element
      if items.len() == 1 {
        apply_function_to_arg(func, &items[0])
      } else {
        // Multiple args - substitute each slot
        let substituted = crate::syntax::substitute_slots(body, &items);
        evaluate_expr_to_expr(&substituted)
      }
    }
    Expr::NamedFunction { params, body } => {
      // Named-parameter function applied to list items
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(items.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      evaluate_expr_to_expr(&substituted)
    }
    _ => Ok(Expr::Apply {
      func: Box::new(func.clone()),
      list: Box::new(list.clone()),
    }),
  }
}

/// Apply MapApply operation on AST (f @@@ {{a, b}, {c, d}} -> {f[a, b], f[c, d]})
pub fn apply_map_apply_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      // Not a list, return unevaluated
      return Ok(Expr::MapApply {
        func: Box::new(func.clone()),
        list: Box::new(list.clone()),
      });
    }
  };

  // MapApply applies func to each sublist
  let results: Result<Vec<Expr>, InterpreterError> = items
    .iter()
    .map(|item| apply_apply_ast(func, item))
    .collect();

  Ok(Expr::List(results?))
}

/// Apply Postfix operation on AST (expr // func)
pub fn apply_postfix_ast(
  expr: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  apply_function_to_arg(func, expr)
}

/// Apply a function to an argument (helper for Map, Postfix, etc.)
pub fn apply_function_to_arg(
  func: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      // Check if this identifier is a variable holding another function/value
      let resolved = ENV.with(|e| e.borrow().get(name).cloned());
      match &resolved {
        Some(StoredValue::ExprVal(expr)) if !matches!(expr, Expr::Identifier(n) if n == name) =>
        {
          return apply_function_to_arg(expr, arg);
        }
        _ => {}
      }
      // Resolve variable holding a function name: t = Flatten; t @ x → Flatten[x]
      if let Some(resolved_name) = resolve_identifier_to_func_name(name) {
        return evaluate_function_call_ast(&resolved_name, &[arg.clone()]);
      }
      // Simple function name: f applied to arg
      evaluate_function_call_ast(name, &[arg.clone()])
    }
    Expr::Function { body } => {
      // Anonymous function: substitute # with arg and evaluate
      let substituted = crate::syntax::substitute_slots(body, &[arg.clone()]);
      evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body } => {
      // Named-parameter function: substitute params with arg
      let mut substituted = (**body).clone();
      if let Some(param) = params.first() {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args } => {
      // Curried function: f[a] applied to b becomes f[a, b]
      // Special case: operator forms where f[x][y] becomes f[y, x]
      // (the applied argument becomes the first parameter)
      if matches!(
        name.as_str(),
        "ReplaceAll"
          | "ReplaceRepeated"
          | "StringStartsQ"
          | "StringEndsQ"
          | "StringContainsQ"
          | "StringMatchQ"
          | "StringReplace"
          | "StringCases"
          | "MemberQ"
          | "Select"
      ) && args.len() == 1
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![arg.clone(), args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else {
        let mut new_args = args.clone();
        new_args.push(arg.clone());
        evaluate_function_call_ast(name, &new_args)
      }
    }
    _ => {
      // Fallback: create a function call expression
      let func_str = expr_to_string(func);
      if let Some(name) = func_str.strip_suffix('&') {
        // It's an anonymous function like "#^2&"
        let body = string_to_expr(name)?;
        let substituted =
          crate::syntax::substitute_slots(&body, &[arg.clone()]);
        evaluate_expr_to_expr(&substituted)
      } else {
        // Treat as a function name
        evaluate_function_call_ast(&func_str, &[arg.clone()])
      }
    }
  }
}

/// Apply a curried call: f[a][b, c] applies function result f[a] to args [b, c]
pub fn apply_curried_call(
  func: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      // Simple function name applied to args
      evaluate_function_call_ast(name, args)
    }
    Expr::Function { body } => {
      // Anonymous function: substitute # with args and evaluate
      let substituted = crate::syntax::substitute_slots(body, args);
      evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body } => {
      // Named-parameter function: substitute each param with corresponding arg
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(args.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "CompiledFunction" && func_args.len() == 2 => {
      // CompiledFunction[{x, y, ...}, body][args...] — substitute and evaluate numerically
      let params: Vec<String> = match &func_args[0] {
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(n) = item {
              Some(n.clone())
            } else {
              None
            }
          })
          .collect(),
        Expr::Identifier(n) => vec![n.clone()],
        _ => vec![],
      };
      let body = &func_args[1];
      // Convert all arguments to Real (compiled functions work numerically)
      let num_args: Vec<Expr> = args
        .iter()
        .map(|a| match a {
          Expr::Integer(n) => Expr::Real(*n as f64),
          Expr::BigInteger(n) => {
            use std::str::FromStr;
            Expr::Real(f64::from_str(&n.to_string()).unwrap_or(f64::NAN))
          }
          other => other.clone(),
        })
        .collect();
      let mut substituted = body.clone();
      for (param, arg) in params.iter().zip(num_args.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      let result = evaluate_expr_to_expr(&substituted)?;
      // Ensure the result is numerical
      match &result {
        Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
        _ => Ok(result),
      }
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } => {
      // Curried function: f[a][b] becomes f[a, b]
      // Special case: operator forms where f[x][y] becomes f[y, x]
      if matches!(
        name.as_str(),
        "ReplaceAll"
          | "ReplaceRepeated"
          | "StringStartsQ"
          | "StringEndsQ"
          | "StringContainsQ"
          | "StringMatchQ"
          | "StringReplace"
          | "StringCases"
          | "MemberQ"
          | "Select"
          | "SortBy"
          | "GroupBy"
          | "CountsBy"
          | "MaximalBy"
          | "MinimalBy"
          | "Cases"
          | "DeleteCases"
          | "Position"
          | "FreeQ"
          | "MatchQ"
          | "Count"
      ) && func_args.len() == 1
        && args.len() == 1
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![args[0].clone(), func_args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if name == "Composition" && !func_args.is_empty() {
        // Composition[f, g, h][x] applies functions right-to-left: f[g[h[x]]]
        let mut result = args.to_vec();
        for f in func_args.iter().rev() {
          let intermediate = apply_curried_call(f, &result)?;
          result = vec![intermediate];
        }
        Ok(result.into_iter().next().unwrap())
      } else if name == "MapAt" && func_args.len() == 2 && args.len() == 1 {
        // MapAt[f, pos][expr] -> MapAt[f, expr, pos]
        let new_args =
          vec![func_args[0].clone(), args[0].clone(), func_args[1].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if matches!(
        name.as_str(),
        "Derivative"
          | "Apply"
          | "Map"
          | "MapThread"
          | "Scan"
          | "Append"
          | "Prepend"
          | "Take"
          | "Drop"
          | "Between"
      ) {
        // Known operator-form functions: flatten curried call
        let mut new_args = func_args.clone();
        new_args.extend(args.iter().cloned());
        evaluate_function_call_ast(name, &new_args)
      } else {
        // Unknown/symbolic curried call: preserve the CurriedCall form
        // e.g. f[g][x] stays as f[g][x], not f[g, x]
        Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        })
      }
    }
    Expr::List(_) => {
      // {f, g}[x] stays unevaluated as a CurriedCall in Wolfram Language
      Ok(Expr::CurriedCall {
        func: Box::new(func.clone()),
        args: args.to_vec(),
      })
    }
    _ => {
      // Fallback: try to convert to string and evaluate
      let func_str = expr_to_string(func);
      if let Some(name) = func_str.strip_suffix('&') {
        // It's an anonymous function like "#^2&"
        let body = string_to_expr(name)?;
        let substituted = crate::syntax::substitute_slots(&body, args);
        evaluate_expr_to_expr(&substituted)
      } else if args.len() == 1 {
        // Treat as a function name with single arg
        evaluate_function_call_ast(&func_str, args)
      } else {
        // Multiple args - treat as curried
        evaluate_function_call_ast(&func_str, args)
      }
    }
  }
}
