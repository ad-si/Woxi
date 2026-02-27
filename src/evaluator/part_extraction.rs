#[allow(unused_imports)]
use super::*;

fn expr_to_i64(expr: &Expr) -> Option<i64> {
  match expr {
    Expr::Integer(n) => Some(*n as i64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i64()
    }
    Expr::Real(f) => Some(*f as i64),
    _ => None,
  }
}

fn normalize_span_pos(pos: i64, len: i64) -> Option<i64> {
  if pos == 0 {
    return None;
  }
  let normalized = if pos < 0 { len + pos + 1 } else { pos };
  if (1..=len).contains(&normalized) {
    Some(normalized)
  } else {
    None
  }
}

fn part_take_unevaluated(expr: &Expr, index: &Expr) -> Expr {
  Expr::Part {
    expr: Box::new(expr.clone()),
    index: Box::new(index.clone()),
  }
}

fn part_take_warn(expr: &Expr, start: i64, end: i64) {
  let expr_str = crate::syntax::expr_to_string(expr);
  eprintln!();
  eprintln!(
    "Part::take: Cannot take positions {} through {} in {}.",
    start, end, expr_str
  );
}

fn extract_span_from_items(
  expr: &Expr,
  index: &Expr,
  items: &[Expr],
  start_expr: &Expr,
  end_expr: &Expr,
  step_expr: Option<&Expr>,
  rebuild: impl FnOnce(Vec<Expr>) -> Expr,
) -> Result<Expr, InterpreterError> {
  let len = items.len() as i64;
  let start_raw = if matches!(start_expr, Expr::Identifier(s) if s == "All") {
    1
  } else if let Some(v) = expr_to_i64(start_expr) {
    v
  } else {
    return Ok(part_take_unevaluated(expr, index));
  };
  let step = match step_expr {
    Some(e) => match expr_to_i64(e) {
      Some(v) => v,
      None => return Ok(part_take_unevaluated(expr, index)),
    },
    None => 1,
  };
  if step == 0 {
    let end_raw = if matches!(end_expr, Expr::Identifier(s) if s == "All") {
      len
    } else if let Some(v) = expr_to_i64(end_expr) {
      v
    } else {
      return Ok(part_take_unevaluated(expr, index));
    };
    part_take_warn(expr, start_raw, end_raw);
    return Ok(part_take_unevaluated(expr, index));
  }
  let end_raw = if matches!(end_expr, Expr::Identifier(s) if s == "All") {
    if step < 0 { -1 } else { len }
  } else if let Some(v) = expr_to_i64(end_expr) {
    v
  } else {
    return Ok(part_take_unevaluated(expr, index));
  };

  let start = if let Some(v) = normalize_span_pos(start_raw, len) {
    v
  } else {
    part_take_warn(expr, start_raw, end_raw);
    return Ok(part_take_unevaluated(expr, index));
  };
  let end = if let Some(v) = normalize_span_pos(end_raw, len) {
    v
  } else {
    part_take_warn(expr, start_raw, end_raw);
    return Ok(part_take_unevaluated(expr, index));
  };

  if (step > 0 && start > end) || (step < 0 && start < end) {
    part_take_warn(expr, start_raw, end_raw);
    return Ok(part_take_unevaluated(expr, index));
  }

  let mut out = Vec::new();
  let mut i = start;
  loop {
    if (step > 0 && i > end) || (step < 0 && i < end) {
      break;
    }
    out.push(items[(i - 1) as usize].clone());
    i += step;
  }
  Ok(rebuild(out))
}

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

  // Span[start, end] or Span[start, end, step]
  if let Expr::FunctionCall { name, args } = index
    && name == "Span"
    && (args.len() == 2 || args.len() == 3)
  {
    match expr {
      Expr::List(items) => {
        return extract_span_from_items(
          expr,
          index,
          items,
          &args[0],
          &args[1],
          args.get(2),
          Expr::List,
        );
      }
      Expr::FunctionCall {
        name,
        args: fn_args,
      } => {
        return extract_span_from_items(
          expr,
          index,
          fn_args,
          &args[0],
          &args[1],
          args.get(2),
          |selected| Expr::FunctionCall {
            name: name.clone(),
            args: selected,
          },
        );
      }
      _ => return Ok(part_take_unevaluated(expr, index)),
    }
  }

  let idx = match index {
    Expr::Integer(n) => *n as i64,
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      match n.to_i64() {
        Some(v) => v,
        None => return Ok(part_take_unevaluated(expr, index)),
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
    _ => return Ok(part_take_unevaluated(expr, index)),
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
        Ok(part_take_unevaluated(expr, index))
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
        Ok(part_take_unevaluated(expr, index))
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      // Rule[a, b] has Part 0 = Rule/RuleDelayed, Part 1 = a, Part 2 = b
      if idx == 0 {
        let head_name = if matches!(expr, Expr::Rule { .. }) {
          "Rule"
        } else {
          "RuleDelayed"
        };
        return Ok(Expr::Identifier(head_name.to_string()));
      }
      let parts: [&Expr; 2] = [pattern, replacement];
      let len = 2i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(parts[actual_idx as usize].clone())
      } else {
        let expr_str = crate::syntax::expr_to_string(expr);
        eprintln!();
        eprintln!("Part::partw: Part {} of {} does not exist.", idx, expr_str);
        Ok(part_take_unevaluated(expr, index))
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
        Ok(part_take_unevaluated(expr, index))
      }
    }
    _ => Ok(part_take_unevaluated(expr, index)),
  }
}
