#[allow(unused_imports)]
use super::*;

/// Get the innermost base of nested Part expressions
pub fn get_part_base(expr: &Expr) -> &Expr {
  match expr {
    Expr::Part { expr: inner, .. } => get_part_base(inner),
    _ => expr,
  }
}

/// Apply a chain of Part indices to an expression, handling All by mapping over elements
pub fn apply_part_indices(
  expr: &Expr,
  indices: &[Expr],
) -> Result<Expr, InterpreterError> {
  if indices.is_empty() {
    return Ok(expr.clone());
  }
  let idx = &indices[0];
  let rest = &indices[1..];
  if matches!(idx, Expr::Identifier(s) if s == "All") {
    // All: map remaining indices over each element
    if rest.is_empty() {
      // Part[expr, All] with no more indices — return as-is
      return Ok(expr.clone());
    }
    match expr {
      Expr::List(items) => {
        let mapped: Result<Vec<Expr>, InterpreterError> = items
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::List(mapped?))
      }
      Expr::FunctionCall { name, args } => {
        let mapped: Result<Vec<Expr>, InterpreterError> = args
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::FunctionCall {
          name: name.clone(),
          args: mapped?,
        })
      }
      _ => apply_part_indices(expr, rest),
    }
  } else {
    // Normal index: extract, then continue with remaining indices
    let extracted = extract_part_ast(expr, idx)?;
    apply_part_indices(&extracted, rest)
  }
}

/// Evaluate the base expression of a Part, with optimization for identifiers in ENV
pub fn eval_part_base(e: &Expr) -> Result<Expr, InterpreterError> {
  if let Expr::Identifier(var_name) = e {
    let env_result = ENV.with(|env| {
      let env = env.borrow();
      if let Some(StoredValue::ExprVal(stored)) = env.get(var_name) {
        Some(Ok(stored.clone()))
      } else {
        None
      }
    });
    if let Some(r) = env_result {
      return r;
    }
  }
  evaluate_expr_to_expr(e)
}

/// Extract part from expression on AST (expr[[index]])
pub fn extract_part_ast(
  expr: &Expr,
  index: &Expr,
) -> Result<Expr, InterpreterError> {
  // For associations, handle key-based lookup (can be any Expr)
  if let Expr::Association(items) = expr {
    // Try to find a matching key
    let index_str = crate::syntax::expr_to_string(index);
    for (key, val) in items {
      let key_str = crate::syntax::expr_to_string(key);
      // Compare keys (strip quotes from strings for comparison)
      let key_cmp = key_str.trim_matches('"');
      let index_cmp = index_str.trim_matches('"');
      if key_cmp == index_cmp {
        return Ok(val.clone());
      }
    }
    // Key not found - return unevaluated Part
    return Ok(Expr::Part {
      expr: Box::new(expr.clone()),
      index: Box::new(index.clone()),
    });
  }

  let idx = match index {
    Expr::Integer(n) => *n as i64,
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      match n.to_i64() {
        Some(v) => v,
        None => {
          return Ok(Expr::Part {
            expr: Box::new(expr.clone()),
            index: Box::new(index.clone()),
          });
        }
      }
    }
    Expr::Real(f) => *f as i64,
    Expr::List(indices) => {
      // Part[expr, {i1, i2, ...}] → {Part[expr, i1], Part[expr, i2], ...}
      let mut results = Vec::new();
      for idx_expr in indices {
        results.push(extract_part_ast(expr, idx_expr)?);
      }
      return Ok(Expr::List(results));
    }
    _ => {
      return Ok(Expr::Part {
        expr: Box::new(expr.clone()),
        index: Box::new(index.clone()),
      });
    }
  };

  match expr {
    Expr::List(items) => {
      let len = items.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(items[actual_idx as usize].clone())
      } else {
        // Print warning to stderr and return unevaluated Part expression
        let expr_str = crate::syntax::expr_to_string(expr);
        eprintln!();
        eprintln!("Part::partw: Part {} of {} does not exist.", idx, expr_str);
        Ok(Expr::Part {
          expr: Box::new(expr.clone()),
          index: Box::new(index.clone()),
        })
      }
    }
    Expr::FunctionCall { name, args } => {
      if idx == 0 {
        // Part[f[...], 0] returns the head
        return Ok(Expr::Identifier(name.clone()));
      }
      let len = args.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(args[actual_idx as usize].clone())
      } else {
        let expr_str = crate::syntax::expr_to_string(expr);
        eprintln!();
        eprintln!("Part::partw: Part {} of {} does not exist.", idx, expr_str);
        Ok(Expr::Part {
          expr: Box::new(expr.clone()),
          index: Box::new(index.clone()),
        })
      }
    }
    Expr::String(s) => {
      let chars: Vec<char> = s.chars().collect();
      let len = chars.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(Expr::String(chars[actual_idx as usize].to_string()))
      } else {
        // Print warning to stderr and return unevaluated Part expression
        eprintln!();
        eprintln!("Part::partw: Part {} of \"{}\" does not exist.", idx, s);
        Ok(Expr::Part {
          expr: Box::new(expr.clone()),
          index: Box::new(index.clone()),
        })
      }
    }
    _ => Ok(Expr::Part {
      expr: Box::new(expr.clone()),
      index: Box::new(index.clone()),
    }),
  }
}
