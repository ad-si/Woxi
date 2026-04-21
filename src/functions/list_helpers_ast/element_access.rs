#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// Decompose a BinaryOp or UnaryOp expression into canonical Wolfram
/// (head_name, args) form so that First/Rest/Part/etc. can operate on them.
///
/// Returns None for atoms and already-handled types (List, FunctionCall).
pub fn expr_to_head_args(expr: &Expr) -> Option<(String, Vec<Expr>)> {
  use crate::syntax::{BinaryOperator, UnaryOperator};
  match expr {
    Expr::BinaryOp { op, left, right } => {
      let (head, args) = match op {
        BinaryOperator::Plus => ("Plus", vec![*left.clone(), *right.clone()]),
        BinaryOperator::Minus => {
          // a - b  =  Plus[a, Times[-1, b]]
          let neg_right = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), *right.clone()],
          };
          ("Plus", vec![*left.clone(), neg_right])
        }
        BinaryOperator::Times => ("Times", vec![*left.clone(), *right.clone()]),
        BinaryOperator::Divide => {
          // a / b  =  Times[a, Power[b, -1]]
          let inv_right = Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![*right.clone(), Expr::Integer(-1)],
          };
          ("Times", vec![*left.clone(), inv_right])
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
      Some((head.to_string(), args))
    }
    Expr::UnaryOp { op, operand } => match op {
      UnaryOperator::Minus => Some((
        "Times".to_string(),
        vec![Expr::Integer(-1), *operand.clone()],
      )),
      UnaryOperator::Not => Some(("Not".to_string(), vec![*operand.clone()])),
    },
    Expr::Rule {
      pattern,
      replacement,
    } => Some((
      "Rule".to_string(),
      vec![*pattern.clone(), *replacement.clone()],
    )),
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some((
      "RuleDelayed".to_string(),
      vec![*pattern.clone(), *replacement.clone()],
    )),
    _ => None,
  }
}

/// AST-based First: return first element of list.
/// First[list] or First[list, default] - returns default if list is empty.
pub fn first_ast(
  list: &Expr,
  default: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  match list {
    Expr::Association(pairs) => {
      if pairs.is_empty() {
        if let Some(d) = default {
          return Ok(d.clone());
        }
        return Ok(Expr::FunctionCall {
          name: "First".to_string(),
          args: vec![list.clone()],
        });
      }
      Ok(pairs[0].1.clone())
    }
    Expr::List(items) => {
      if items.is_empty() {
        if let Some(d) = default {
          Ok(d.clone())
        } else {
          let expr_str = crate::syntax::expr_to_string(list);
          crate::emit_message(&format!(
            "{} has zero length and no first element.",
            expr_str
          ));
          Ok(Expr::FunctionCall {
            name: "First".to_string(),
            args: vec![list.clone()],
          })
        }
      } else {
        Ok(items[0].clone())
      }
    }
    Expr::FunctionCall { name, args } => {
      // NumericArray / ByteArray wrap a single list payload that should be
      // indexed as if it were the sequence itself.
      if (name == "NumericArray" || name == "ByteArray")
        && args.len() == 1
        && let Expr::List(items) = &args[0]
      {
        if items.is_empty() {
          if let Some(d) = default {
            return Ok(d.clone());
          }
        } else {
          return Ok(items[0].clone());
        }
      }
      if args.is_empty() {
        if let Some(d) = default {
          Ok(d.clone())
        } else {
          let expr_str = crate::syntax::expr_to_string(list);
          crate::emit_message(&format!(
            "{} has zero length and no first element.",
            expr_str
          ));
          Ok(Expr::FunctionCall {
            name: "First".to_string(),
            args: vec![list.clone()],
          })
        }
      } else {
        Ok(args[0].clone())
      }
    }
    _ => {
      // Try decomposing BinaryOp/UnaryOp to canonical form
      if let Some((_head, args)) = expr_to_head_args(list) {
        if args.is_empty() {
          if let Some(d) = default {
            return Ok(d.clone());
          }
        } else {
          return Ok(args[0].clone());
        }
      }
      if let Some(d) = default {
        Ok(d.clone())
      } else {
        Ok(Expr::FunctionCall {
          name: "First".to_string(),
          args: vec![list.clone()],
        })
      }
    }
  }
}

/// AST-based Last: return last element of list.
/// Last[list] or Last[list, default] - returns default if list is empty.
pub fn last_ast(
  list: &Expr,
  default: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  match list {
    Expr::Association(pairs) => {
      if pairs.is_empty() {
        if let Some(d) = default {
          return Ok(d.clone());
        }
        return Ok(Expr::FunctionCall {
          name: "Last".to_string(),
          args: vec![list.clone()],
        });
      }
      Ok(pairs[pairs.len() - 1].1.clone())
    }
    Expr::List(items) => {
      if items.is_empty() {
        if let Some(d) = default {
          Ok(d.clone())
        } else {
          let expr_str = crate::syntax::expr_to_string(list);
          crate::emit_message(&format!(
            "{} has zero length and no last element.",
            expr_str
          ));
          Ok(Expr::FunctionCall {
            name: "Last".to_string(),
            args: vec![list.clone()],
          })
        }
      } else {
        Ok(items[items.len() - 1].clone())
      }
    }
    Expr::FunctionCall { name, args } => {
      // NumericArray / ByteArray wrap a single list payload that should be
      // indexed as if it were the sequence itself.
      if (name == "NumericArray" || name == "ByteArray")
        && args.len() == 1
        && let Expr::List(items) = &args[0]
      {
        if items.is_empty() {
          if let Some(d) = default {
            return Ok(d.clone());
          }
        } else {
          return Ok(items[items.len() - 1].clone());
        }
      }
      if args.is_empty() {
        if let Some(d) = default {
          Ok(d.clone())
        } else {
          let expr_str = crate::syntax::expr_to_string(list);
          crate::emit_message(&format!(
            "{} has zero length and no last element.",
            expr_str
          ));
          Ok(Expr::FunctionCall {
            name: "Last".to_string(),
            args: vec![list.clone()],
          })
        }
      } else {
        Ok(args[args.len() - 1].clone())
      }
    }
    _ => {
      // Try decomposing BinaryOp/UnaryOp to canonical form
      if let Some((_head, args)) = expr_to_head_args(list)
        && !args.is_empty()
      {
        return Ok(args[args.len() - 1].clone());
      }
      if let Some(d) = default {
        Ok(d.clone())
      } else {
        Ok(Expr::FunctionCall {
          name: "Last".to_string(),
          args: vec![list.clone()],
        })
      }
    }
  }
}

/// AST-based Rest: return all but first element.
pub fn rest_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      if items.is_empty() {
        let expr_str = crate::syntax::expr_to_string(list);
        crate::emit_message(&format!(
          "Cannot take Rest of expression {} with length zero.",
          expr_str
        ));
        Ok(Expr::FunctionCall {
          name: "Rest".to_string(),
          args: vec![list.clone()],
        })
      } else {
        Ok(Expr::List(items[1..].to_vec()))
      }
    }
    Expr::FunctionCall { name, args } => {
      if args.is_empty() {
        let expr_str = crate::syntax::expr_to_string(list);
        crate::emit_message(&format!(
          "Cannot take Rest of expression {} with length zero.",
          expr_str
        ));
        Ok(Expr::FunctionCall {
          name: "Rest".to_string(),
          args: vec![list.clone()],
        })
      } else {
        // Evaluate so e.g. Times[x] reduces to x
        crate::evaluator::evaluate_function_call_ast(name, &args[1..])
      }
    }
    _ => {
      // Try decomposing BinaryOp/UnaryOp to canonical form
      if let Some((head, args)) = expr_to_head_args(list) {
        if args.is_empty() {
          let expr_str = crate::syntax::expr_to_string(list);
          crate::emit_message(&format!(
            "Cannot take Rest of expression {} with length zero.",
            expr_str
          ));
          return Ok(Expr::FunctionCall {
            name: "Rest".to_string(),
            args: vec![list.clone()],
          });
        }
        // Evaluate so e.g. Times[x] reduces to x
        return crate::evaluator::evaluate_function_call_ast(&head, &args[1..]);
      }
      Err(InterpreterError::EvaluationError(format!(
        "Nonatomic expression expected at position 1 in Rest[{}].",
        crate::syntax::expr_to_string(list)
      )))
    }
  }
}

/// AST-based Most: return all but last element.
pub fn most_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      if items.is_empty() {
        Err(InterpreterError::EvaluationError(
          "Most: list is empty".into(),
        ))
      } else {
        Ok(Expr::List(items[..items.len() - 1].to_vec()))
      }
    }
    Expr::FunctionCall { name, args } => {
      if args.is_empty() {
        Err(InterpreterError::EvaluationError(format!(
          "Cannot take Most of expression {}[] with length zero.",
          name
        )))
      } else {
        crate::evaluator::evaluate_function_call_ast(
          name,
          &args[..args.len() - 1],
        )
      }
    }
    _ => {
      // Try decomposing BinaryOp/UnaryOp to canonical form
      if let Some((head, args)) = expr_to_head_args(list) {
        if args.is_empty() {
          return Err(InterpreterError::EvaluationError(
            "Most: expression has length zero".into(),
          ));
        }
        return crate::evaluator::evaluate_function_call_ast(
          &head,
          &args[..args.len() - 1],
        );
      }
      Ok(Expr::FunctionCall {
        name: "Most".to_string(),
        args: vec![list.clone()],
      })
    }
  }
}

/// AST-based Take: take first n elements.
/// Returns unevaluated if n exceeds list length (to let fallback handle error).
/// Multi-dimensional Take: Take[list, spec1, spec2, ...]
pub fn take_multi_ast(
  list: &Expr,
  specs: &[Expr],
) -> Result<Expr, InterpreterError> {
  if specs.is_empty() {
    return Ok(list.clone());
  }

  // Apply the first spec at this level
  let result = take_ast(list, &specs[0])?;

  // If there are more specs, apply them recursively to each element
  if specs.len() == 1 {
    return Ok(result);
  }

  match &result {
    Expr::List(items) => {
      let mut new_items = Vec::new();
      for item in items {
        new_items.push(take_multi_ast(item, &specs[1..])?);
      }
      Ok(Expr::List(new_items))
    }
    _ => Ok(result),
  }
}

pub fn take_ast(list: &Expr, n: &Expr) -> Result<Expr, InterpreterError> {
  // Handle All: return the list unchanged
  if matches!(n, Expr::Identifier(name) if name == "All") {
    return Ok(list.clone());
  }

  // Handle associations: Take on values, reconstruct association
  if let Expr::Association(pairs) = list {
    let rules: Vec<Expr> = pairs
      .iter()
      .map(|(k, v)| Expr::Rule {
        pattern: Box::new(k.clone()),
        replacement: Box::new(v.clone()),
      })
      .collect();
    let result = take_ast(&Expr::List(rules), n)?;
    // Convert result back to association
    if let Expr::List(items) = &result {
      let mut new_pairs = Vec::new();
      for item in items {
        match item {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => {
            new_pairs.push((*pattern.clone(), *replacement.clone()));
          }
          _ => return Ok(result),
        }
      }
      return Ok(Expr::Association(new_pairs));
    }
    return Ok(result);
  }

  let (items, head): (&[Expr], Option<&str>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name: h, args } => (args.as_slice(), Some(h.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  // Helper to wrap result in the original head.
  let wrap = |v: Vec<Expr>| -> Expr {
    match head {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: v,
      },
      None => Expr::List(v),
    }
  };

  // Handle Take[list, {start, end}] and Take[list, {start, end, step}]
  if let Expr::List(spec) = n {
    if spec.len() == 1 {
      if let Some(idx) = expr_to_i128(&spec[0]) {
        let len = items.len() as i128;
        let real_idx = if idx < 0 { len + idx + 1 } else { idx };
        if real_idx >= 1 && real_idx <= len {
          return Ok(wrap(vec![items[(real_idx - 1) as usize].clone()]));
        }
      }
    } else if spec.len() >= 2 {
      let len = items.len() as i128;
      if let (Some(start), Some(end)) =
        (expr_to_i128(&spec[0]), expr_to_i128(&spec[1]))
      {
        let step = if spec.len() == 3 {
          expr_to_i128(&spec[2]).unwrap_or(1)
        } else {
          1
        };
        let real_start = if start < 0 { len + start + 1 } else { start };
        let real_end = if end < 0 { len + end + 1 } else { end };
        if real_start >= 1
          && real_end >= 1
          && real_start <= len
          && real_end <= len
          && step != 0
        {
          let mut result = Vec::new();
          let mut i = real_start;
          while (step > 0 && i <= real_end) || (step < 0 && i >= real_end) {
            result.push(items[(i - 1) as usize].clone());
            i += step;
          }
          return Ok(wrap(result));
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Take".to_string(),
      args: vec![list.clone(), n.clone()],
    });
  }

  // Handle UpTo[n]: take up to n elements (clamp to list length)
  if let Expr::FunctionCall {
    name: up_name,
    args: up_args,
  } = n
    && up_name == "UpTo"
    && up_args.len() == 1
    && let Some(max_count) = expr_to_i128(&up_args[0])
  {
    let len = items.len() as i128;
    let actual = max_count.min(len);
    if actual >= 0 {
      return Ok(wrap(items[..actual as usize].to_vec()));
    }
  }

  let count = match expr_to_i128(n) {
    Some(i) => i,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  let len = items.len() as i128;
  if count >= 0 {
    if count > len {
      // Print warning to stderr and return unevaluated
      let list_str = crate::syntax::expr_to_string(list);
      crate::emit_message(&format!(
        "Take::take: Cannot take positions 1 through {} in {}.",
        count, list_str
      ));
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
    Ok(wrap(items[..count as usize].to_vec()))
  } else {
    if -count > len {
      // Print warning to stderr and return unevaluated
      let list_str = crate::syntax::expr_to_string(list);
      crate::emit_message(&format!(
        "Take::take: Cannot take positions {} through -1 in {}.",
        count, list_str
      ));
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
    Ok(wrap(items[items.len() - (-count) as usize..].to_vec()))
  }
}

/// AST-based Drop: drop first n elements.
pub fn drop_ast(list: &Expr, n: &Expr) -> Result<Expr, InterpreterError> {
  // Handle associations: convert to rules, apply Drop, convert back
  if let Expr::Association(pairs) = list {
    let rules: Vec<Expr> = pairs
      .iter()
      .map(|(k, v)| Expr::Rule {
        pattern: Box::new(k.clone()),
        replacement: Box::new(v.clone()),
      })
      .collect();
    let result = drop_ast(&Expr::List(rules), n)?;
    if let Expr::List(items) = &result {
      let mut new_pairs = Vec::new();
      for item in items {
        match item {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => {
            new_pairs.push((*pattern.clone(), *replacement.clone()));
          }
          _ => return Ok(result),
        }
      }
      return Ok(Expr::Association(new_pairs));
    }
    return Ok(result);
  }

  let (items, drop_head): (&[Expr], Option<&str>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name: h, args } => (args.as_slice(), Some(h.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Drop".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  let wrap_drop = |v: Vec<Expr>| -> Expr {
    match drop_head {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: v,
      },
      None => Expr::List(v),
    }
  };

  let len = items.len() as i128;

  // Drop[list, {m, n}] - drop elements m through n
  if let Expr::List(spec) = n {
    if spec.len() == 2
      && let (Some(m), Some(n_end)) =
        (expr_to_i128(&spec[0]), expr_to_i128(&spec[1]))
    {
      let start = if m > 0 { m - 1 } else { len + m };
      let end = if n_end > 0 { n_end - 1 } else { len + n_end };
      let start = start.max(0) as usize;
      let end = (end + 1).max(0).min(len) as usize;
      if start >= end {
        return Ok(list.clone());
      }
      let mut result = items[..start].to_vec();
      result.extend_from_slice(&items[end..]);
      return Ok(wrap_drop(result));
    }
    // Drop[list, {n}] - drop the nth element
    if spec.len() == 1
      && let Some(n_val) = expr_to_i128(&spec[0])
    {
      let idx = if n_val > 0 { n_val - 1 } else { len + n_val };
      if idx < 0 || idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "Drop: index {} out of range for list of length {}",
          n_val, len
        )));
      }
      let idx = idx as usize;
      let mut result = items[..idx].to_vec();
      result.extend_from_slice(&items[idx + 1..]);
      return Ok(wrap_drop(result));
    }
    // Drop[list, {m, n, s}] - drop elements m, m+s, m+2s, ..., up to n.
    if spec.len() == 3
      && let (Some(m), Some(n_end), Some(step)) = (
        expr_to_i128(&spec[0]),
        expr_to_i128(&spec[1]),
        expr_to_i128(&spec[2]),
      )
    {
      if step == 0 {
        return Err(InterpreterError::EvaluationError(
          "Drop: step cannot be zero".into(),
        ));
      }
      let start_idx = if m > 0 { m - 1 } else { len + m };
      let end_idx = if n_end > 0 { n_end - 1 } else { len + n_end };
      let mut to_drop = std::collections::HashSet::new();
      let mut i = start_idx;
      if step > 0 {
        while i <= end_idx {
          if i < 0 || i >= len {
            return Err(InterpreterError::EvaluationError(format!(
              "Drop: position {} is out of range for list of length {}",
              i + 1,
              len
            )));
          }
          to_drop.insert(i as usize);
          i += step;
        }
      } else {
        while i >= end_idx {
          if i < 0 || i >= len {
            return Err(InterpreterError::EvaluationError(format!(
              "Drop: position {} is out of range for list of length {}",
              i + 1,
              len
            )));
          }
          to_drop.insert(i as usize);
          i += step;
        }
      }
      let result: Vec<Expr> = items
        .iter()
        .enumerate()
        .filter_map(|(idx, item)| {
          if to_drop.contains(&idx) {
            None
          } else {
            Some(item.clone())
          }
        })
        .collect();
      return Ok(wrap_drop(result));
    }
    return Ok(Expr::FunctionCall {
      name: "Drop".to_string(),
      args: vec![list.clone(), n.clone()],
    });
  }

  let count = match expr_to_i128(n) {
    Some(i) => i,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Drop".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  if count >= 0 {
    let drop_count = count.min(len) as usize;
    Ok(wrap_drop(items[drop_count..].to_vec()))
  } else {
    let keep = (len + count).max(0) as usize;
    Ok(wrap_drop(items[..keep].to_vec()))
  }
}

/// Drop[list, rows, cols] — 3-argument Drop.
/// When `rows` is `None`, all rows are kept and `cols` is applied to each row.
pub fn drop_multi_ast(
  list: &Expr,
  rows: &Expr,
  cols: &Expr,
) -> Result<Expr, InterpreterError> {
  // First apply row dropping
  let after_rows = if matches!(rows, Expr::Identifier(s) if s == "None") {
    list.clone()
  } else {
    drop_ast(list, rows)?
  };

  // Then apply column dropping to each element
  if matches!(cols, Expr::Identifier(s) if s == "None") {
    return Ok(after_rows);
  }

  match &after_rows {
    Expr::List(items) => {
      let mut result = Vec::with_capacity(items.len());
      for item in items {
        result.push(drop_ast(item, cols)?);
      }
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Drop".to_string(),
      args: vec![list.clone(), rows.clone(), cols.clone()],
    }),
  }
}

/// Part[list, i] or list[[i]] - Extract element at position i (1-indexed)
pub fn part_ast(list: &Expr, index: &Expr) -> Result<Expr, InterpreterError> {
  // Try decomposing BinaryOp/UnaryOp to canonical form first
  if let Some((head_name, ha_args)) = expr_to_head_args(list) {
    let canonical = Expr::FunctionCall {
      name: head_name,
      args: ha_args,
    };
    return part_ast(&canonical, index);
  }

  // Handle Association integer indexing: Part[<|a->1, b->2|>, 2] => 2
  if let Expr::FunctionCall { name, args } = list
    && name == "Association"
    && let Expr::Integer(i) = index
  {
    if *i == 0 {
      return Ok(Expr::Identifier("Association".to_string()));
    }
    let idx = if *i > 0 {
      (*i as usize) - 1
    } else {
      let len = args.len() as i128;
      (len + *i) as usize
    };
    if idx >= args.len() {
      return Err(InterpreterError::EvaluationError(
        "Part: index out of bounds".into(),
      ));
    }
    // Extract value from Rule[key, value]
    return match &args[idx] {
      Expr::FunctionCall {
        name: rname,
        args: rargs,
      } if rname == "Rule" && rargs.len() == 2 => Ok(rargs[1].clone()),
      Expr::Rule { replacement, .. } => Ok(*replacement.clone()),
      other => Ok(other.clone()),
    };
  }

  let (items, head) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Part".to_string(),
        args: vec![list.clone(), index.clone()],
      });
    }
  };

  match index {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let i = expr_to_i128(index).ok_or_else(|| {
        InterpreterError::EvaluationError("Part: index too large".into())
      })?;
      if i == 0 {
        // Part[expr, 0] returns the head
        return Ok(Expr::Identifier(head.unwrap_or("List").to_string()));
      }
      if i < 0 {
        // Negative indexing: count from end
        let len = items.len() as i128;
        let idx = len + i;
        if idx < 0 || idx >= len {
          return Err(InterpreterError::EvaluationError(
            "Part: index out of bounds".into(),
          ));
        }
        return Ok(items[idx as usize].clone());
      }
      let idx = (i as usize) - 1;
      if idx >= items.len() {
        return Err(InterpreterError::EvaluationError(
          "Part: index out of bounds".into(),
        ));
      }
      Ok(items[idx].clone())
    }
    Expr::List(indices) => {
      // Multiple indices: Part[list, {i1, i2, ...}]
      let mut results = Vec::new();
      for idx_expr in indices {
        if let Expr::Integer(i) = idx_expr {
          if *i < 1 {
            return Err(InterpreterError::EvaluationError(
              "Part: index must be a positive integer".into(),
            ));
          }
          let idx = (*i as usize) - 1;
          if idx >= items.len() {
            return Err(InterpreterError::EvaluationError(
              "Part: index out of bounds".into(),
            ));
          }
          results.push(items[idx].clone());
        } else {
          return Err(InterpreterError::EvaluationError(
            "Part: indices must be integers".into(),
          ));
        }
      }
      Ok(Expr::List(results))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Part".to_string(),
      args: vec![list.clone(), index.clone()],
    }),
  }
}

/// Insert[list, elem, n] - Insert element at position n (1-indexed).
/// Threads through any non-atomic head, so e.g. Insert[f[a,b], x, 2] returns
/// f[a, x, b].
pub fn insert_ast(
  list: &Expr,
  elem: &Expr,
  pos: &Expr,
) -> Result<Expr, InterpreterError> {
  let (items, head): (Vec<Expr>, Option<String>) = match list {
    Expr::List(items) => (items.clone(), None),
    Expr::FunctionCall { name, args } => (args.clone(), Some(name.clone())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Insert".to_string(),
        args: vec![list.clone(), elem.clone(), pos.clone()],
      });
    }
  };

  let len = items.len() as i128;

  // Collect all insertion positions. Supports:
  //   - Integer:              Insert[list, x, 2]
  //   - {n}:                  Insert[list, x, {2}]     (single-level position)
  //   - {{p1}, {p2}, ...}:    Insert[list, x, {{2}, {4}}]  (multiple positions)
  let positions: Vec<i128> = match pos {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      vec![expr_to_i128(pos).unwrap()]
    }
    Expr::List(inner) => {
      // Two shapes:
      // {n}          — single position at top level
      // {{p1}, ...}  — list of (top-level) positions
      if inner.iter().all(|i| matches!(i, Expr::List(_))) && !inner.is_empty() {
        let mut ps = Vec::with_capacity(inner.len());
        for sub in inner {
          if let Expr::List(sub_items) = sub {
            if sub_items.len() != 1 {
              return Err(InterpreterError::EvaluationError(
                "Insert: nested position specs are not supported".into(),
              ));
            }
            match expr_to_i128(&sub_items[0]) {
              Some(n) => ps.push(n),
              None => {
                return Err(InterpreterError::EvaluationError(
                  "Insert: position must be an integer".into(),
                ));
              }
            }
          }
        }
        ps
      } else if inner.len() == 1 {
        match expr_to_i128(&inner[0]) {
          Some(n) => vec![n],
          None => {
            return Err(InterpreterError::EvaluationError(
              "Insert: position must be an integer".into(),
            ));
          }
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "Insert: position must be an integer".into(),
        ));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Insert: position must be an integer".into(),
      ));
    }
  };

  // Normalize each position to a 0-based insert index into the ORIGINAL list.
  // Wolfram semantics: when multiple positions are given they all refer to
  // positions in the original list, not the list as it grows.
  let mut insert_indices: Vec<usize> = Vec::with_capacity(positions.len());
  for n in &positions {
    let idx = if *n > 0 {
      let p = (*n - 1) as usize;
      if p > items.len() {
        return Err(InterpreterError::EvaluationError(
          "Insert: position out of bounds".into(),
        ));
      }
      p
    } else if *n < 0 {
      if *n < -(len + 1) {
        return Err(InterpreterError::EvaluationError(
          "Insert: position out of bounds".into(),
        ));
      }
      (len + 1 + *n) as usize
    } else {
      return Err(InterpreterError::EvaluationError(
        "Insert: position cannot be 0".into(),
      ));
    };
    insert_indices.push(idx);
  }

  // Insert from rightmost to leftmost so earlier indices stay valid.
  insert_indices.sort_unstable();
  let mut result = items;
  for idx in insert_indices.into_iter().rev() {
    result.insert(idx, elem.clone());
  }
  match head {
    Some(name) => Ok(Expr::FunctionCall { name, args: result }),
    None => Ok(Expr::List(result)),
  }
}

/// Extract[list, n] - extracts element at position n
/// Extract[list, {n1, n2, ...}] - extracts element at nested position
pub fn extract_ast(
  list: &Expr,
  index: &Expr,
  head: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let result = extract_ast_inner(list, index, head)?;
  // If head is provided and the result is NOT from a multi-position spec
  // (which already wraps each element), wrap the result with head.
  if let Some(h) = head {
    // For multi-position specs (list of lists), wrapping is already done inside.
    // For single-position specs, wrap here.
    let is_multi = matches!(index, Expr::List(indices)
      if !indices.is_empty() && indices.iter().all(|i| matches!(i, Expr::List(_))));
    if !is_multi {
      return Ok(wrap_with_head(h, result));
    }
  }
  Ok(result)
}

fn wrap_with_head(head: &Expr, val: Expr) -> Expr {
  match head {
    Expr::Identifier(name) => Expr::FunctionCall {
      name: name.clone(),
      args: vec![val],
    },
    _ => Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![head.clone(), val],
    },
  }
}

fn extract_ast_inner(
  list: &Expr,
  index: &Expr,
  head: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  match index {
    Expr::Integer(_) | Expr::BigInteger(_) => part_ast(list, index),
    Expr::List(indices) => {
      // Check if this is a list of position specs (list of lists)
      let all_lists = !indices.is_empty()
        && indices.iter().all(|i| matches!(i, Expr::List(_)));
      if all_lists {
        // Multiple positions: Extract[expr, {{p1}, {p2, p3}, ...}]
        let mut results = Vec::new();
        for pos_spec in indices {
          let val = extract_ast_inner(list, pos_spec, None)?;
          if let Some(h) = head {
            results.push(wrap_with_head(h, val));
          } else {
            results.push(val);
          }
        }
        return Ok(Expr::List(results));
      }
      // Nested extraction: Extract[expr, {i, j, ...}]
      let mut current = list.clone();
      for idx in indices {
        current = part_ast(&current, idx)?;
      }
      Ok(current)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Extract: invalid index".into(),
    )),
  }
}

/// ReplacePart[list, n -> val] - replaces element at position n
pub fn replace_part_ast(
  expr: &Expr,
  rule: &Expr,
) -> Result<Expr, InterpreterError> {
  // Handle list of rules: ReplacePart[expr, {pos1 -> val1, pos2 -> val2}]
  if let Expr::List(rules) = rule
    && rules
      .iter()
      .all(|r| matches!(r, Expr::Rule { .. } | Expr::RuleDelayed { .. }) || matches!(r, Expr::FunctionCall { name, .. } if name == "Rule" || name == "RuleDelayed"))
    {
      let mut result = expr.clone();
      for r in rules {
        result = replace_part_ast(&result, r)?;
      }
      return Ok(result);
    }

  // Extract position and value from the rule
  let (pos, val, is_delayed) = match rule {
    Expr::Rule {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref(), false),
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref(), true),
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      (&args[0], &args[1], name == "RuleDelayed")
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ReplacePart".to_string(),
        args: vec![expr.clone(), rule.clone()],
      });
    }
  };
  // For RuleDelayed, evaluate the replacement each time it's used
  let eval_val = |v: &Expr| -> Result<Expr, InterpreterError> {
    if is_delayed {
      crate::evaluator::evaluate_expr_to_expr(v)
    } else {
      Ok(v.clone())
    }
  };

  // Handle multiple positions: ReplacePart[expr, {{1}, {3}} -> val]
  if let Expr::List(pos_list) = pos
    && !pos_list.is_empty()
    && pos_list.iter().all(|p| matches!(p, Expr::List(_)))
  {
    let mut result = expr.clone();
    for p in pos_list {
      let ev = eval_val(val)?;
      let sub_rule = Expr::Rule {
        pattern: Box::new(p.clone()),
        replacement: Box::new(ev),
      };
      result = replace_part_ast(&result, &sub_rule)?;
    }
    return Ok(result);
  }

  // Determine the position specification
  let positions: Vec<i128> = match pos {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      vec![expr_to_i128(pos).unwrap_or(0)]
    }
    Expr::List(indices) => indices.iter().filter_map(expr_to_i128).collect(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ReplacePart".to_string(),
        args: vec![expr.clone(), rule.clone()],
      });
    }
  };

  if positions.is_empty() {
    return Ok(expr.clone());
  }

  let ev = eval_val(val)?;

  // Single flat position
  if positions.len() == 1 {
    let p = positions[0];
    return replace_at_position(expr, p, &ev);
  }

  // Multi-part position {i, j, ...}
  replace_at_deep_pos(expr, &positions, &ev)
}

/// Replace at a single position in an expression
fn replace_at_position(
  expr: &Expr,
  pos: i128,
  val: &Expr,
) -> Result<Expr, InterpreterError> {
  let (items, head_name) = match expr {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ReplacePart".to_string(),
        args: vec![
          expr.clone(),
          Expr::Rule {
            pattern: Box::new(Expr::Integer(pos)),
            replacement: Box::new(val.clone()),
          },
        ],
      });
    }
  };

  if pos == 0 {
    // Replace the head
    let new_head = match val {
      Expr::Identifier(s) => s.clone(),
      Expr::FunctionCall { name, .. } => name.clone(),
      _ => crate::syntax::expr_to_string(val),
    };
    return Ok(Expr::FunctionCall {
      name: new_head,
      args: items.to_vec(),
    });
  }

  let len = items.len() as i128;
  let idx = if pos > 0 && pos <= len {
    (pos - 1) as usize
  } else if pos < 0 && -pos <= len {
    (len + pos) as usize
  } else {
    // Out-of-range position: silently return the original expression
    // (matches wolframscript behavior).
    return Ok(expr.clone());
  };

  let mut result = items.to_vec();
  result[idx] = val.clone();
  match head_name {
    Some(h) => Ok(Expr::FunctionCall {
      name: h.to_string(),
      args: result,
    }),
    None => Ok(Expr::List(result)),
  }
}

/// Replace at a deep multi-part position {i, j, ...}
fn replace_at_deep_pos(
  expr: &Expr,
  positions: &[i128],
  val: &Expr,
) -> Result<Expr, InterpreterError> {
  if positions.is_empty() {
    return Ok(val.clone());
  }

  let pos = positions[0];
  let (items, head_name) = match expr {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => return Ok(expr.clone()),
  };

  if pos == 0 {
    // For position 0 at intermediate level, this doesn't make sense
    return Ok(expr.clone());
  }

  let len = items.len() as i128;
  let idx = if pos > 0 && pos <= len {
    (pos - 1) as usize
  } else if pos < 0 && -pos <= len {
    (len + pos) as usize
  } else {
    return Ok(expr.clone());
  };

  let mut result = items.to_vec();
  result[idx] = replace_at_deep_pos(&items[idx], &positions[1..], val)?;
  match head_name {
    Some(h) => Ok(Expr::FunctionCall {
      name: h.to_string(),
      args: result,
    }),
    None => Ok(Expr::List(result)),
  }
}

/// Delete[list, pos] - Delete an element at a position
pub fn delete_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Delete expects exactly 2 arguments".into(),
    ));
  }

  // Extract items and optional head name from List or FunctionCall
  let (items, head_name) = match &args[0] {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args: fargs } => {
      (fargs.as_slice(), Some(name.as_str()))
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Delete".to_string(),
        args: args.to_vec(),
      });
    }
  };

  match &args[1] {
    // Delete[expr, n] - delete at position n
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let pos = match expr_to_i128(&args[1]) {
        Some(n) => n,
        None => return Ok(args[0].clone()),
      };
      // Out-of-range: emit Delete::partw and return unevaluated (matches
      // wolframscript). Position 0 deletes the head, which is always valid.
      let len = items.len() as i128;
      if pos != 0 && (pos > len || pos < -len) {
        crate::emit_message(&format!(
          "Delete::partw: Part {{{}}} of {} does not exist.",
          pos,
          crate::syntax::expr_to_string(&args[0])
        ));
        return Ok(Expr::FunctionCall {
          name: "Delete".to_string(),
          args: args.to_vec(),
        });
      }
      return delete_at_position_general(items, pos, head_name);
    }
    Expr::List(pos_list) => {
      // Determine if this is a multi-part position {i, j, ...} or multiple positions {{p1}, {p2}, ...}
      let is_multiple_positions =
        pos_list.iter().all(|p| matches!(p, Expr::List(_)));

      if is_multiple_positions && !pos_list.is_empty() {
        // Multiple positions: Delete[expr, {{p1}, {p2}, ...}]
        let mut positions: Vec<Vec<i128>> = Vec::new();
        for p in pos_list {
          if let Expr::List(inner) = p {
            let pos: Vec<i128> =
              inner.iter().filter_map(expr_to_i128).collect();
            if pos.len() == inner.len() {
              positions.push(pos);
            }
          }
        }
        let mut result = args[0].clone();
        for pos in positions.iter().rev() {
          if pos.len() == 1 {
            let cur_items = match &result {
              Expr::List(items) => items.as_slice(),
              Expr::FunctionCall { args: fargs, .. } => fargs.as_slice(),
              _ => return Ok(result),
            };
            result = delete_at_position_general(cur_items, pos[0], head_name)?;
          } else {
            result = delete_at_deep_position(&result, pos)?;
          }
        }
        return Ok(result);
      } else {
        // Multi-part position: Delete[expr, {i, j, ...}]
        let pos: Vec<i128> = pos_list.iter().filter_map(expr_to_i128).collect();
        if pos.len() == pos_list.len() {
          if pos.len() == 1 {
            return delete_at_position_general(items, pos[0], head_name);
          } else {
            return delete_at_deep_position(&args[0], &pos);
          }
        }
      }
    }
    _ => {}
  }

  Ok(Expr::FunctionCall {
    name: "Delete".to_string(),
    args: args.to_vec(),
  })
}

/// Delete element at a single flat position in an expression
fn delete_at_position_general(
  items: &[Expr],
  pos: i128,
  head_name: Option<&str>,
) -> Result<Expr, InterpreterError> {
  let wrap = |result_items: Vec<Expr>| -> Expr {
    match head_name {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: result_items,
      },
      None => Expr::List(result_items),
    }
  };
  let len = items.len() as i128;
  let idx = if pos > 0 {
    (pos - 1) as usize
  } else if pos < 0 {
    (len + pos) as usize
  } else {
    // Position 0 = delete the head, return Sequence[args...]
    return Ok(Expr::FunctionCall {
      name: "Sequence".to_string(),
      args: items.to_vec(),
    });
  };
  if idx >= items.len() {
    return Ok(wrap(items.to_vec()));
  }
  let mut result = items.to_vec();
  result.remove(idx);
  Ok(wrap(result))
}

/// Delete element at a deep multi-part position {i, j, ...}
fn delete_at_deep_position(
  expr: &Expr,
  pos: &[i128],
) -> Result<Expr, InterpreterError> {
  if pos.is_empty() {
    return Ok(expr.clone());
  }
  // Extract the current level's items + wrapping (List or FunctionCall[head]).
  let (items, head): (Vec<Expr>, Option<String>) = match expr {
    Expr::List(items) => (items.clone(), None),
    Expr::FunctionCall { name, args } => (args.clone(), Some(name.clone())),
    _ => return Ok(expr.clone()),
  };

  // Position 0 at this level → delete the head. Mathematica folds an
  // "unheaded" argument list into the parent by turning it into a
  // Sequence so the outer head absorbs it.
  if pos[0] == 0 {
    let unheaded = Expr::FunctionCall {
      name: "Sequence".to_string(),
      args: items,
    };
    if pos.len() == 1 {
      return Ok(unheaded);
    }
    // Further indexing past a removed head doesn't make sense for this
    // simple positional API; leave the expression alone.
    return Ok(expr.clone());
  }

  let len = items.len() as i128;
  let idx = if pos[0] > 0 {
    (pos[0] - 1) as usize
  } else {
    (len + pos[0]) as usize
  };
  if idx >= items.len() {
    return Ok(expr.clone());
  }

  let rebuild = |new_items: Vec<Expr>| -> Expr {
    match &head {
      Some(h) => flatten_sequences(Expr::FunctionCall {
        name: h.clone(),
        args: new_items,
      }),
      None => flatten_sequences(Expr::List(new_items)),
    }
  };

  if pos.len() == 1 {
    let mut result = items;
    result.remove(idx);
    Ok(rebuild(result))
  } else {
    let mut result = items.clone();
    result[idx] = delete_at_deep_position(&items[idx], &pos[1..])?;
    Ok(rebuild(result))
  }
}

/// Splice any top-level `Sequence[...]` args into their parent — matches
/// Mathematica's post-evaluation Sequence-flattening rule.
fn flatten_sequences(expr: Expr) -> Expr {
  let splice = |items: &[Expr]| -> Vec<Expr> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
      if let Expr::FunctionCall {
        name: inner_name,
        args: inner_args,
      } = item
        && inner_name == "Sequence"
      {
        out.extend(inner_args.iter().cloned());
      } else {
        out.push(item.clone());
      }
    }
    out
  };
  match &expr {
    Expr::List(items) => Expr::List(splice(items)),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: splice(args),
    },
    _ => expr,
  }
}
