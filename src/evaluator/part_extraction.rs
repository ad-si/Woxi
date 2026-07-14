#[allow(unused_imports)]
use super::*;
use crate::syntax::{BinaryOperator, UnaryOperator};

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

/// `a | b | c` (Alternatives) is a flat, associative head in WL but is stored
/// as a nested BinaryOp chain. Rewrite it into a flattened
/// `Alternatives[a, b, c]` FunctionCall so positional operations (Part, Take,
/// …) see all operands as siblings, matching wolframscript. Returns `None`
/// when `expr` is not such a chain.
pub fn flatten_alternatives_binop(expr: &Expr) -> Option<Expr> {
  fn gather(e: &Expr, out: &mut Vec<Expr>) -> bool {
    if let Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left,
      right,
    } = e
    {
      gather(left, out);
      gather(right, out);
      true
    } else {
      out.push(e.clone());
      false
    }
  }
  let mut parts = Vec::new();
  if gather(expr, &mut parts) {
    Some(Expr::FunctionCall {
      name: "Alternatives".to_string(),
      args: parts.into(),
    })
  } else {
    None
  }
}

fn part_take_warn(expr: &Expr, start: i64, end: i64) {
  let expr_str = crate::syntax::expr_to_string(expr);
  crate::emit_message(&format!(
    "Part::take: Cannot take positions {} through {} in {}.",
    start, end, expr_str
  ));
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
  } else if start_raw == len + 1 {
    // A start one past the end is a valid empty span: `{a}[[2;;]]` → `{}`,
    // `{a,b,c}[[4;;]]` → `{}` (matching wolfram). The empty-result branch
    // below (end == start - 1) then yields the empty list.
    len + 1
  } else {
    part_take_warn(expr, start_raw, end_raw);
    return Ok(part_take_unevaluated(expr, index));
  };
  let end = if let Some(v) = normalize_span_pos(end_raw, len) {
    v
  } else {
    // An end one before the start is a valid empty span: `{x}[[;; -2]]` →
    // `{}` (end normalizes to 0). The empty-result branch below handles it.
    let e = if end_raw < 0 {
      len + end_raw + 1
    } else {
      end_raw
    };
    if e == start - 1 {
      e
    } else {
      part_take_warn(expr, start_raw, end_raw);
      return Ok(part_take_unevaluated(expr, index));
    }
  };

  if (step > 0 && start > end) || (step < 0 && start < end) {
    // Reverse-direction span. wolframscript treats the adjacent case
    // (`end == start - step`) as an empty result — so `{a,b,c}[[-1;;-2]]`
    // and `{…}[[3;;2]]` produce `{}`, and `Plus[a,b,c,d][[-1;;-2]]`
    // folds to `0`. Any wider inversion (e.g. `5;;2`) is an error.
    if (step > 0 && end == start - 1) || (step < 0 && end == start + 1) {
      let empty = rebuild(Vec::new());
      return crate::evaluator::evaluate_expr_to_expr(&empty);
    }
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
        Ok(Expr::List(mapped?.into()))
      }
      Expr::FunctionCall { name, args } => {
        let mapped: Result<Vec<Expr>, InterpreterError> = args
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::FunctionCall {
          name: name.clone(),
          args: mapped?.into(),
        })
      }
      _ => apply_part_indices(expr, rest),
    }
  } else if let Expr::List(sub_indices) = idx {
    // Part[expr, {i1, i2, ...}, rest...] → {Part[expr, i1, rest...], ...}
    // (map remaining indices over each extracted element, like All does)
    if rest.is_empty() {
      let extracted = extract_part_ast(expr, idx)?;
      return Ok(extracted);
    }
    let mut results = Vec::with_capacity(sub_indices.len());
    for sub_idx in sub_indices {
      let extracted = extract_part_ast(expr, sub_idx)?;
      results.push(apply_part_indices(&extracted, rest)?);
    }
    Ok(Expr::List(results.into()))
  } else if let Expr::FunctionCall { name, .. } = idx
    && name == "Span"
    && !rest.is_empty()
  {
    // Part[expr, Span[...], rest...] → map remaining indices over each
    // element of the span-extracted sub-list.
    let extracted = extract_part_ast(expr, idx)?;
    match &extracted {
      Expr::List(items) => {
        let mapped: Result<Vec<Expr>, InterpreterError> = items
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::List(mapped?.into()))
      }
      Expr::FunctionCall {
        name: head,
        args: fn_args,
      } => {
        let mapped: Result<Vec<Expr>, InterpreterError> = fn_args
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::FunctionCall {
          name: head.clone(),
          args: mapped?.into(),
        })
      }
      _ => apply_part_indices(&extracted, rest),
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

/// Number of positionally-indexable parts of `expr`, or `None` for atoms and
/// expressions (like ByteArray) whose part count differs from their argument
/// count. Used to validate a list-of-positions spec atomically.
fn positional_length(expr: &Expr) -> Option<usize> {
  match expr {
    Expr::List(items) => Some(items.len()),
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => Some(2),
    Expr::FunctionCall { name, args } if name != "ByteArray" => {
      Some(args.len())
    }
    _ => None,
  }
}

/// Rewrap the elements selected by a list-of-positions spec with the head of
/// the original expression: `f[a,b,c][[{1,3}]]` is `f[a, c]`, `(x->y)[[{1,2}]]`
/// is `x -> y`, and a List keeps the List head. Only used on expressions that
/// `positional_length` accepts.
fn rewrap_part_list(expr: &Expr, elems: Vec<Expr>) -> Expr {
  match expr {
    Expr::FunctionCall { name, .. } => Expr::FunctionCall {
      name: name.clone(),
      args: elems.into(),
    },
    Expr::Rule { .. } if elems.len() == 2 => Expr::Rule {
      pattern: Box::new(elems[0].clone()),
      replacement: Box::new(elems[1].clone()),
    },
    Expr::RuleDelayed { .. } if elems.len() == 2 => Expr::RuleDelayed {
      pattern: Box::new(elems[0].clone()),
      replacement: Box::new(elems[1].clone()),
    },
    _ => Expr::List(elems.into()),
  }
}

/// Concrete integer position from an index expression (Integer, integer-valued
/// Real, or BigInteger), or `None` for anything else.
fn simple_position(e: &Expr) -> Option<i64> {
  match e {
    Expr::Integer(n) => Some(*n as i64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i64()
    }
    Expr::Real(f) if f.is_finite() && f.fract() == 0.0 => Some(*f as i64),
    _ => None,
  }
}

/// Extract part from expression on AST (expr[[index]])
pub fn extract_part_ast(
  expr: &Expr,
  index: &Expr,
) -> Result<Expr, InterpreterError> {
  // Normalize a nested Alternatives chain (`a | b | c`) into a flat
  // Alternatives[a, b, c] so positional access matches wolframscript.
  if let Some(flat) = flatten_alternatives_binop(expr) {
    return extract_part_ast(&flat, index);
  }

  // A 1-D SparseArray indexed by an integer yields the scalar entry at that
  // position. (Spans/higher-rank keep sub-arrays sparse and fall through.)
  if matches!(index, Expr::Integer(_) | Expr::BigInteger(_))
    && let Expr::FunctionCall { name, args: sa } = expr
    && name == "SparseArray"
    && sa.len() == 4
    && matches!(&sa[1], Expr::List(d) if d.len() == 1)
  {
    let dense = crate::functions::list_helpers_ast::sparse_array_ast(sa)?;
    if matches!(&dense, Expr::List(_)) {
      return extract_part_ast(&dense, index);
    }
  }

  // For associations, handle key-based lookup and integer position indexing
  if let Expr::Association(items) = expr {
    // Positional lookup shared by the single-integer and list-index forms.
    // Returns the (key, value) entry, or None when out of bounds (the caller
    // emits Part::partw).
    let entry_at = |i: i128| -> Option<&(Expr, Expr)> {
      if i == 0 {
        return None;
      }
      let idx = if i > 0 {
        (i as usize) - 1
      } else {
        let pos = items.len() as i128 + i;
        if pos < 0 {
          return None;
        }
        pos as usize
      };
      items.get(idx)
    };
    let missing_for = |key: &Expr| Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![Expr::String("KeyAbsent".to_string()), key.clone()].into(),
    };
    // Key-based lookup shared by the string, Key[...], and list-index forms.
    // String indices match only string keys; Key[k] matches any key exactly.
    let lookup = |key: &Expr| -> Option<&Expr> {
      items
        .iter()
        .find(|(k, _)| crate::evaluator::pattern_matching::expr_equal(k, key))
        .map(|(_, v)| v)
    };
    match index {
      // Integer index: access by position (return value, not rule)
      Expr::Integer(i) => {
        if *i == 0 {
          return Ok(Expr::Identifier("Association".to_string()));
        }
        if let Some((_, v)) = entry_at(*i) {
          return Ok(v.clone());
        }
        crate::emit_message(&format!(
          "Part::partw: Part {} of {} does not exist.",
          i,
          crate::syntax::format_expr(expr, crate::syntax::ExprForm::Output)
        ));
        return Ok(part_take_unevaluated(expr, index));
      }
      // String index: lookup among string keys; absent -> Missing[KeyAbsent, key]
      Expr::String(_) => {
        if let Some(v) = lookup(index) {
          return Ok(v.clone());
        }
        return Ok(missing_for(index));
      }
      // Key[k] index: exact key-expression lookup (works for non-string keys)
      Expr::FunctionCall { name, args } if name == "Key" && args.len() == 1 => {
        if let Some(v) = lookup(&args[0]) {
          return Ok(v.clone());
        }
        return Ok(missing_for(index));
      }
      // Span: positional slice preserving the key -> value entries
      Expr::FunctionCall { name, args }
        if name == "Span" && (args.len() == 2 || args.len() == 3) =>
      {
        let rule_items: Vec<Expr> = items
          .iter()
          .map(|(k, v)| Expr::Rule {
            pattern: Box::new(k.clone()),
            replacement: Box::new(v.clone()),
          })
          .collect();
        return extract_span_from_items(
          expr,
          index,
          &rule_items,
          &args[0],
          &args[1],
          args.get(2),
          |selected| {
            Expr::Association(
              selected
                .into_iter()
                .map(|r| match &r {
                  Expr::Rule {
                    pattern,
                    replacement,
                  } => ((**pattern).clone(), (**replacement).clone()),
                  _ => (r.clone(), Expr::Identifier("Null".to_string())),
                })
                .collect(),
            )
          },
        );
      }
      // List index: sub-association of the selected entries
      Expr::List(idxs) => {
        let mut selected: Vec<(Expr, Expr)> = Vec::with_capacity(idxs.len());
        for ix in idxs.iter() {
          match ix {
            Expr::Integer(i) => {
              if let Some((k, v)) = entry_at(*i) {
                selected.push((k.clone(), v.clone()));
              } else {
                crate::emit_message(&format!(
                  "Part::partw: Part {} of {} does not exist.",
                  i,
                  crate::syntax::format_expr(
                    expr,
                    crate::syntax::ExprForm::Output
                  )
                ));
                return Ok(part_take_unevaluated(expr, index));
              }
            }
            Expr::String(_) => match lookup(ix) {
              Some(v) => selected.push((ix.clone(), v.clone())),
              None => selected.push((ix.clone(), missing_for(ix))),
            },
            Expr::FunctionCall { name, args }
              if name == "Key" && args.len() == 1 =>
            {
              match lookup(&args[0]) {
                Some(v) => selected.push((args[0].clone(), v.clone())),
                None => selected.push((args[0].clone(), missing_for(ix))),
              }
            }
            _ => {
              crate::emit_message(&format!(
                "Part::pkspec1: The expression {} cannot be used as a part specification.",
                crate::syntax::format_expr(ix, crate::syntax::ExprForm::Output)
              ));
              return Ok(part_take_unevaluated(expr, index));
            }
          }
        }
        return Ok(Expr::Association(selected));
      }
      // Anything else is not a valid association part specification
      _ => {
        crate::emit_message(&format!(
          "Part::pkspec1: The expression {} cannot be used as a part specification.",
          crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
        ));
        return Ok(part_take_unevaluated(expr, index));
      }
    }
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
          |v: Vec<Expr>| Expr::List(v.into()),
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
            args: selected.into(),
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
    // Only integer-valued Reals are valid part specs (e.g. 2.0 -> 2).
    // A non-integer Real (1.5) or a Rational (3/2) errors Part::pkspec1
    // (matching wolframscript), rather than silently truncating 1.5 -> 1.
    Expr::Real(f) if f.is_finite() && f.fract() == 0.0 => *f as i64,
    Expr::Real(_) => {
      crate::emit_message(&format!(
        "Part::pkspec1: The expression {} cannot be used as a part specification.",
        crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
      ));
      return Ok(part_take_unevaluated(expr, index));
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      crate::emit_message(&format!(
        "Part::pkspec1: The expression {} cannot be used as a part specification.",
        crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
      ));
      return Ok(part_take_unevaluated(expr, index));
    }
    Expr::List(indices) => {
      // Part[expr, {i1, i2, ...}] collects the elements at each position. When
      // every position is a concrete integer and the expression has a known
      // length, validate the bounds up front: a single out-of-range position
      // fails the WHOLE spec with one Part::partw naming the full index list
      // (matching wolframscript), instead of returning a partially-resolved
      // list with a per-index message.
      if let Some(len) = positional_length(expr) {
        let positions: Option<Vec<i64>> =
          indices.iter().map(simple_position).collect();
        if let Some(positions) = positions {
          let len = len as i64;
          let in_range = |i: i64| {
            i == 0 || (1..=len).contains(&i) || (-len..=-1).contains(&i)
          };
          if !positions.iter().all(|&i| in_range(i)) {
            crate::emit_message(&format!(
              "Part::partw: Part {} of {} does not exist.",
              crate::syntax::format_expr(
                index,
                crate::syntax::ExprForm::Output
              ),
              crate::syntax::format_expr(expr, crate::syntax::ExprForm::Output),
            ));
            return Ok(part_take_unevaluated(expr, index));
          }
          // All positions valid: extract each part and keep the head of the
          // original expression (f[a,b,c][[{1,3}]] -> f[a, c]).
          let mut results = Vec::with_capacity(indices.len());
          for idx_expr in indices.iter() {
            results.push(extract_part_ast(expr, idx_expr)?);
          }
          return Ok(rewrap_part_list(expr, results));
        }
      }
      // Fallback: non-positional expression (e.g. ByteArray) or non-integer
      // index entries — extract each part under a List head (legacy behavior).
      let mut results = Vec::new();
      for idx_expr in indices {
        results.push(extract_part_ast(expr, idx_expr)?);
      }
      return Ok(Expr::List(results.into()));
    }
    _ => return Ok(part_take_unevaluated(expr, index)),
  };

  match expr {
    Expr::List(items) => {
      if idx == 0 {
        // Part[{...}, 0] returns the head, which is List
        return Ok(Expr::Identifier("List".to_string()));
      }
      let len = items.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(items[actual_idx as usize].clone())
      } else {
        // Print warning to stderr and return unevaluated Part expression
        let expr_str = crate::syntax::expr_to_string(expr);
        crate::emit_message(&format!(
          "Part::partw: Part {} of {} does not exist.",
          idx, expr_str
        ));
        Ok(part_take_unevaluated(expr, index))
      }
    }
    Expr::FunctionCall { name, args } => {
      if idx == 0 {
        // Part[f[...], 0] returns the head
        return Ok(Expr::Identifier(name.clone()));
      }
      // ByteArray["base64"] indexes into the decoded byte sequence,
      // matching wolframscript: ByteArray[{1, 25, 3}][[2]] -> 25.
      if name == "ByteArray"
        && args.len() == 1
        && let Expr::String(b64) = &args[0]
      {
        use base64::Engine;
        let engine = base64::engine::general_purpose::STANDARD;
        if let Ok(bytes) = engine.decode(b64) {
          let len = bytes.len() as i64;
          let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
          if actual_idx >= 0 && actual_idx < len {
            return Ok(Expr::Integer(bytes[actual_idx as usize] as i128));
          } else {
            let expr_str = crate::syntax::expr_to_string(expr);
            crate::emit_message(&format!(
              "Part::partw: Part {} of {} does not exist.",
              idx, expr_str
            ));
            return Ok(part_take_unevaluated(expr, index));
          }
        }
      }
      let len = args.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(args[actual_idx as usize].clone())
      } else {
        let expr_str = crate::syntax::expr_to_string(expr);
        crate::emit_message(&format!(
          "Part::partw: Part {} of {} does not exist.",
          idx, expr_str
        ));
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
        crate::emit_message(&format!(
          "Part::partw: Part {} of {} does not exist.",
          idx, expr_str
        ));
        Ok(part_take_unevaluated(expr, index))
      }
    }
    // Strings are atoms in Wolfram Language — Part[string, n] for n ≠ 0
    // returns unevaluated (handled by the fallback arm below).
    Expr::BinaryOp { op, left, right } => {
      // Decompose BinaryOp into head + args consistent with Head[]
      let (head_name, parts): (&str, Vec<Expr>) = match op {
        BinaryOperator::Plus => ("Plus", vec![*left.clone(), *right.clone()]),
        BinaryOperator::Minus => {
          // a - b = Plus[a, Times[-1, b]]
          let neg_right = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), *right.clone()].into(),
          };
          ("Plus", vec![*left.clone(), neg_right])
        }
        BinaryOperator::Times => ("Times", vec![*left.clone(), *right.clone()]),
        BinaryOperator::Divide => {
          if matches!(left.as_ref(), Expr::Integer(1)) {
            // 1/b = Power[b, -1]
            ("Power", vec![*right.clone(), Expr::Integer(-1)])
          } else {
            // a/b = Times[a, Power[b, -1]]
            let inv_right = Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![*right.clone(), Expr::Integer(-1)].into(),
            };
            ("Times", vec![*left.clone(), inv_right])
          }
        }
        BinaryOperator::Power => ("Power", vec![*left.clone(), *right.clone()]),
        BinaryOperator::And => ("And", vec![*left.clone(), *right.clone()]),
        BinaryOperator::Or => ("Or", vec![*left.clone(), *right.clone()]),
        BinaryOperator::StringJoin => {
          ("StringJoin", vec![*left.clone(), *right.clone()])
        }
        BinaryOperator::Alternatives => {
          ("Alternatives", vec![*left.clone(), *right.clone()])
        }
      };
      if idx == 0 {
        return Ok(Expr::Identifier(head_name.to_string()));
      }
      let len = parts.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(parts[actual_idx as usize].clone())
      } else {
        let expr_str = crate::syntax::expr_to_string(expr);
        crate::emit_message(&format!(
          "Part::partw: Part {} of {} does not exist.",
          idx, expr_str
        ));
        Ok(part_take_unevaluated(expr, index))
      }
    }
    Expr::UnaryOp { op, operand } => {
      let (head_name, parts): (&str, Vec<Expr>) = match op {
        UnaryOperator::Minus => {
          ("Times", vec![Expr::Integer(-1), *operand.clone()])
        }
        UnaryOperator::Not => ("Not", vec![*operand.clone()]),
      };
      if idx == 0 {
        return Ok(Expr::Identifier(head_name.to_string()));
      }
      let len = parts.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(parts[actual_idx as usize].clone())
      } else {
        let expr_str = crate::syntax::expr_to_string(expr);
        crate::emit_message(&format!(
          "Part::partw: Part {} of {} does not exist.",
          idx, expr_str
        ));
        Ok(part_take_unevaluated(expr, index))
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      // Index into the WL structure: uniform `a op b op c` is Op[a, b, c];
      // mixed `a < b <= c` is Inequality[a, Less, b, LessEqual, c].
      let (head, parts) =
        crate::syntax::comparison_head_and_args(operands, operators);
      if idx == 0 {
        return Ok(Expr::Identifier(head));
      }
      let len = parts.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(parts[actual_idx as usize].clone())
      } else {
        let expr_str = crate::syntax::expr_to_string(expr);
        crate::emit_message(&format!(
          "Part::partw: Part {} of {} does not exist.",
          idx, expr_str
        ));
        Ok(part_take_unevaluated(expr, index))
      }
    }
    _ => {
      // Atoms: Part[atom, 0] returns the Head of the atom
      if idx == 0 {
        return crate::functions::predicate_ast::head_ast(&[expr.clone()]);
      }
      Ok(part_take_unevaluated(expr, index))
    }
  }
}
