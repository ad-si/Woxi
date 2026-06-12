#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// Decompose a BinaryOp or UnaryOp expression into canonical Wolfram
/// (head_name, args) form so that First/Rest/Part/etc. can operate on them.
///
/// Returns None for atoms and already-handled types (List, FunctionCall).
/// Flatten a chained binary operator (Plus/Times/And/Or/StringJoin/Alternatives)
/// into its full n-ary argument list. e.g. `(((1+2)+3)+4)` → `[1, 2, 3, 4]`.
fn flatten_assoc_binop(
  op: crate::syntax::BinaryOperator,
  left: &crate::syntax::Expr,
  right: &crate::syntax::Expr,
) -> Vec<crate::syntax::Expr> {
  let mut out = flatten_assoc_one(op, left);
  out.extend(flatten_assoc_one(op, right));
  out
}

fn flatten_assoc_one(
  op: crate::syntax::BinaryOperator,
  expr: &crate::syntax::Expr,
) -> Vec<crate::syntax::Expr> {
  if let crate::syntax::Expr::BinaryOp {
    op: inner_op,
    left,
    right,
  } = expr
    && *inner_op == op
  {
    flatten_assoc_binop(op, left, right)
  } else {
    vec![expr.clone()]
  }
}

pub fn expr_to_head_args(expr: &Expr) -> Option<(String, Vec<Expr>)> {
  use crate::syntax::{BinaryOperator, UnaryOperator};
  match expr {
    Expr::BinaryOp { op, left, right } => {
      let (head, args) = match op {
        BinaryOperator::Plus => (
          "Plus",
          flatten_assoc_binop(BinaryOperator::Plus, left, right),
        ),
        BinaryOperator::Minus => {
          // a - b  =  Plus[a, Times[-1, b]]
          let neg_right = Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), *right.clone()].into(),
          };
          // Still flatten any further chained Plus on the left.
          let mut args = flatten_assoc_one(BinaryOperator::Plus, left.as_ref());
          args.push(neg_right);
          ("Plus", args)
        }
        BinaryOperator::Times => (
          "Times",
          flatten_assoc_binop(BinaryOperator::Times, left, right),
        ),
        BinaryOperator::Divide => {
          // a / b  =  Times[a, Power[b, -1]]
          let inv_right = Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![*right.clone(), Expr::Integer(-1)].into(),
          };
          let mut args =
            flatten_assoc_one(BinaryOperator::Times, left.as_ref());
          args.push(inv_right);
          ("Times", args)
        }
        BinaryOperator::Power => ("Power", vec![*left.clone(), *right.clone()]),
        BinaryOperator::And => {
          ("And", flatten_assoc_binop(BinaryOperator::And, left, right))
        }
        BinaryOperator::Or => {
          ("Or", flatten_assoc_binop(BinaryOperator::Or, left, right))
        }
        BinaryOperator::StringJoin => (
          "StringJoin",
          flatten_assoc_binop(BinaryOperator::StringJoin, left, right),
        ),
        BinaryOperator::Alternatives => (
          "Alternatives",
          flatten_assoc_binop(BinaryOperator::Alternatives, left, right),
        ),
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
          args: vec![list.clone()].into(),
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
            args: vec![list.clone()].into(),
          })
        }
      } else {
        Ok(items[0].clone())
      }
    }
    Expr::FunctionCall { name, args } => {
      // NumericArray / ByteArray wrap a single list payload that should be
      // indexed as if it were the sequence itself. The 2-arg
      // `NumericArray[list, dtype]` form is what the dispatcher emits;
      // re-wrap a nested list element so multi-dim arrays return a typed
      // sub-array (`First[NumericArray[<2,2>, …]] → NumericArray[<2>, …]`).
      if (name == "NumericArray" || name == "ByteArray")
        && (args.len() == 1 || args.len() == 2)
        && let Expr::List(items) = &args[0]
      {
        if items.is_empty() {
          if let Some(d) = default {
            return Ok(d.clone());
          }
        } else {
          let elem = items[0].clone();
          if name == "NumericArray"
            && args.len() == 2
            && matches!(elem, Expr::List(_))
          {
            return Ok(Expr::FunctionCall {
              name: "NumericArray".to_string(),
              args: vec![elem, args[1].clone()].into(),
            });
          }
          return Ok(elem);
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
            args: vec![list.clone()].into(),
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
          args: vec![list.clone()].into(),
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
          args: vec![list.clone()].into(),
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
            args: vec![list.clone()].into(),
          })
        }
      } else {
        Ok(items[items.len() - 1].clone())
      }
    }
    Expr::FunctionCall { name, args } => {
      // NumericArray / ByteArray wrap a single list payload that should be
      // indexed as if it were the sequence itself. Mirrors the First handler
      // above — the 2-arg `NumericArray[list, dtype]` form re-wraps nested
      // list results so `Last[NumericArray[<2,2>, …]]` returns a typed
      // sub-array.
      if (name == "NumericArray" || name == "ByteArray")
        && (args.len() == 1 || args.len() == 2)
        && let Expr::List(items) = &args[0]
      {
        if items.is_empty() {
          if let Some(d) = default {
            return Ok(d.clone());
          }
        } else {
          let elem = items[items.len() - 1].clone();
          if name == "NumericArray"
            && args.len() == 2
            && matches!(elem, Expr::List(_))
          {
            return Ok(Expr::FunctionCall {
              name: "NumericArray".to_string(),
              args: vec![elem, args[1].clone()].into(),
            });
          }
          return Ok(elem);
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
            args: vec![list.clone()].into(),
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
          args: vec![list.clone()].into(),
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
          args: vec![list.clone()].into(),
        })
      } else {
        Ok(Expr::List(items[1..].to_vec().into()))
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
          args: vec![list.clone()].into(),
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
            args: vec![list.clone()].into(),
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
        Ok(Expr::List(items[..items.len() - 1].to_vec().into()))
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
        args: vec![list.clone()].into(),
      })
    }
  }
}

/// AST-based Take: take first n elements.
/// Returns unevaluated if n exceeds list length (to let fallback handle error).
/// Multi-dimensional Take: Take[list, spec1, spec2, ...]

/// Emit the Take::take / Drop::drop message for an invalid position range.
fn take_drop_message(fname: &str, from: i128, to: i128, list: &Expr) {
  let (tag, verb) = if fname == "Take" {
    ("take", "take")
  } else {
    ("drop", "drop")
  };
  crate::emit_message(&format!(
    "{}::{}: Cannot {} positions {} through {} in {}.",
    fname,
    tag,
    verb,
    from,
    to,
    crate::syntax::format_expr(list, crate::syntax::ExprForm::Output)
  ));
}

/// The (from, to) display range of a Take/Drop spec, for messages.
/// The count carried by a valid `UpTo[n]` wrapper: a non-negative
/// integer, or effectively unbounded for `UpTo[Infinity]`.
fn upto_count(e: &Expr) -> Option<i128> {
  if let Expr::FunctionCall { name, args } = e
    && name == "UpTo"
    && args.len() == 1
  {
    return match &args[0] {
      Expr::Integer(v) if *v >= 0 => Some(*v),
      Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity" => {
        Some(i128::MAX)
      }
      Expr::FunctionCall { name: dn, args: da }
        if dn == "DirectedInfinity"
          && da.len() == 1
          && matches!(&da[0], Expr::Integer(1)) =>
      {
        Some(i128::MAX)
      }
      _ => None,
    };
  }
  None
}

/// Whether an expression has the shape of a valid Take/Drop sequence
/// specification: ±n, All, None, UpTo[n], {i}, {i, j} or {i, j, s} with
/// machine-sized integer entries, a nonzero step, and UpTo allowed for
/// the i/j endpoints. Used to decide when to emit Take::seqs/Drop::seqs.
pub fn seq_spec_shape_ok(spec: &Expr) -> bool {
  const M: i128 = i64::MAX as i128;
  let machine =
    |e: &Expr| matches!(e, Expr::Integer(v) if (-M..=M).contains(v));
  let endpoint = |e: &Expr| machine(e) || upto_count(e).is_some();
  match spec {
    Expr::Identifier(s) if s == "All" || s == "None" => true,
    e if machine(e) || upto_count(e).is_some() => true,
    Expr::List(parts) => match parts.as_slice() {
      [i] => machine(i),
      [i, j] => endpoint(i) && endpoint(j),
      [i, j, s] => {
        endpoint(i)
          && endpoint(j)
          && machine(s)
          && !matches!(s, Expr::Integer(0))
      }
      _ => false,
    },
    _ => false,
  }
}

fn spec_range(n: &Expr) -> Option<(i128, i128)> {
  match n {
    Expr::Integer(v) if *v >= 0 => Some((1, *v)),
    Expr::Integer(v) => Some((*v, -1)),
    Expr::List(spec) => {
      let nums: Option<Vec<i128>> =
        spec.iter().map(super::utilities::expr_to_i128).collect();
      match nums.as_deref() {
        Some([i]) => Some((*i, *i)),
        Some([i, j]) | Some([i, j, _]) => Some((*i, *j)),
        _ => None,
      }
    }
    _ => None,
  }
}

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
      Ok(Expr::List(new_items.into()))
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
    let result = take_ast(&Expr::List(rules.into()), n)?;
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
          _ => return Ok(result.clone()),
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
      if let Some((from, to)) = spec_range(n) {
        take_drop_message("Take", from, to, list);
      }
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()].into(),
      });
    }
  };

  // Helper to wrap result in the original head.
  let wrap = |v: Vec<Expr>| -> Expr {
    match head {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: v.into(),
      },
      None => Expr::List(v.into()),
    }
  };

  // Handle None: an empty take
  if matches!(n, Expr::Identifier(name) if name == "None") {
    return Ok(wrap(Vec::new()));
  }

  // Handle Take[list, {start, end}] and Take[list, {start, end, step}]
  if let Expr::List(spec) = n {
    if spec.len() == 1 {
      if let Some(idx) = expr_to_i128(&spec[0]) {
        let len = items.len() as i128;
        let real_idx = if idx < 0 { len + idx + 1 } else { idx };
        if real_idx >= 1 && real_idx <= len {
          return Ok(wrap(vec![items[(real_idx - 1) as usize].clone()]));
        }
        take_drop_message("Take", idx, idx, list);
        return Ok(Expr::FunctionCall {
          name: "Take".to_string(),
          args: vec![list.clone(), n.clone()].into(),
        });
      }
    } else if spec.len() >= 2 {
      let len = items.len() as i128;
      // UpTo[k] endpoints clamp to the list length.
      let endpoint = |e: &Expr| -> Option<i128> {
        upto_count(e)
          .map(|k| k.min(len))
          .or_else(|| expr_to_i128(e))
      };
      if let (Some(start), Some(end)) = (endpoint(&spec[0]), endpoint(&spec[1]))
      {
        let step = if spec.len() == 3 {
          expr_to_i128(&spec[2]).unwrap_or(1)
        } else {
          1
        };
        let real_start = if start < 0 { len + start + 1 } else { start };
        let real_end = if end < 0 { len + end + 1 } else { end };
        // An adjacent reversed range (end == start - step) is an empty
        // take; anything further reversed or out of range is an error
        let adjacent_empty = step != 0 && real_end == real_start - step;
        let in_range = real_start >= 1
          && real_end >= 1
          && real_start <= len
          && real_end <= len;
        let proper = step != 0
          && ((step > 0 && real_end >= real_start)
            || (step < 0 && real_end <= real_start));
        if in_range && step != 0 && (proper || adjacent_empty) {
          let mut result = Vec::new();
          let mut i = real_start;
          while (step > 0 && i <= real_end) || (step < 0 && i >= real_end) {
            result.push(items[(i - 1) as usize].clone());
            i += step;
          }
          return Ok(wrap(result));
        }
        if step != 0 {
          take_drop_message("Take", start, end, list);
          return Ok(Expr::FunctionCall {
            name: "Take".to_string(),
            args: vec![list.clone(), n.clone()].into(),
          });
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Take".to_string(),
      args: vec![list.clone(), n.clone()].into(),
    });
  }

  // Handle UpTo[n]: take up to n elements (clamp to list length)
  if let Some(max_count) = upto_count(n) {
    let actual = max_count.min(items.len() as i128) as usize;
    return Ok(wrap(items[..actual].to_vec()));
  }

  let count = match expr_to_i128(n) {
    Some(i) => i,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()].into(),
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
        args: vec![list.clone(), n.clone()].into(),
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
        args: vec![list.clone(), n.clone()].into(),
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
    let result = drop_ast(&Expr::List(rules.into()), n)?;
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
          _ => return Ok(result.clone()),
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
      if let Some((from, to)) = spec_range(n) {
        take_drop_message("Drop", from, to, list);
      }
      return Ok(Expr::FunctionCall {
        name: "Drop".to_string(),
        args: vec![list.clone(), n.clone()].into(),
      });
    }
  };

  let wrap_drop = |v: Vec<Expr>| -> Expr {
    match drop_head {
      Some(h) => Expr::FunctionCall {
        name: h.to_string(),
        args: v.into(),
      },
      None => Expr::List(v.into()),
    }
  };

  let len = items.len() as i128;

  // Handle None (drop nothing) and All (drop everything)
  if matches!(n, Expr::Identifier(name) if name == "None") {
    return Ok(list.clone());
  }
  if matches!(n, Expr::Identifier(name) if name == "All") {
    return Ok(wrap_drop(Vec::new()));
  }

  // UpTo[k] endpoints clamp to the list length.
  let endpoint = |e: &Expr| -> Option<i128> {
    upto_count(e)
      .map(|k| k.min(len))
      .or_else(|| expr_to_i128(e))
  };

  // Drop[list, {m, n}] - drop elements m through n
  if let Expr::List(spec) = n {
    if spec.len() == 2
      && let (Some(m), Some(n_end)) = (endpoint(&spec[0]), endpoint(&spec[1]))
    {
      let real_start = if m > 0 { m } else { len + m + 1 };
      let real_end = if n_end > 0 { n_end } else { len + n_end + 1 };
      // The adjacent reversed range drops nothing; anything further
      // reversed or out of range is an error
      if real_end == real_start - 1 {
        return Ok(list.clone());
      }
      if real_start < 1
        || real_end < 1
        || real_start > len
        || real_end > len
        || real_end < real_start
      {
        take_drop_message("Drop", m, n_end, list);
        return Ok(Expr::FunctionCall {
          name: "Drop".to_string(),
          args: vec![list.clone(), n.clone()].into(),
        });
      }
      let start = (real_start - 1) as usize;
      let end = real_end as usize;
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
        take_drop_message("Drop", n_val, n_val, list);
        return Ok(Expr::FunctionCall {
          name: "Drop".to_string(),
          args: vec![list.clone(), n.clone()].into(),
        });
      }
      let idx = idx as usize;
      let mut result = items[..idx].to_vec();
      result.extend_from_slice(&items[idx + 1..]);
      return Ok(wrap_drop(result));
    }
    // Drop[list, {m, n, s}] - drop elements m, m+s, m+2s, ..., up to n.
    if spec.len() == 3
      && let (Some(m), Some(n_end), Some(step)) = (
        endpoint(&spec[0]),
        endpoint(&spec[1]),
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
      args: vec![list.clone(), n.clone()].into(),
    });
  }

  // Handle UpTo[n]: drop min(n, len) elements from the front.
  if let Some(max_count) = upto_count(n) {
    let drop_count = max_count.min(len) as usize;
    return Ok(wrap_drop(items[drop_count..].to_vec()));
  }

  let count = match expr_to_i128(n) {
    Some(i) => i,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Drop".to_string(),
        args: vec![list.clone(), n.clone()].into(),
      });
    }
  };

  if count.unsigned_abs() > len.unsigned_abs() {
    let (from, to) = if count >= 0 { (1, count) } else { (count, -1) };
    take_drop_message("Drop", from, to, list);
    return Ok(Expr::FunctionCall {
      name: "Drop".to_string(),
      args: vec![list.clone(), n.clone()].into(),
    });
  }
  if count >= 0 {
    Ok(wrap_drop(items[count as usize..].to_vec()))
  } else {
    let keep = (len + count) as usize;
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
      Ok(Expr::List(result.into()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Drop".to_string(),
      args: vec![list.clone(), rows.clone(), cols.clone()].into(),
    }),
  }
}

/// Part[list, i] or list[[i]] - Extract element at position i (1-indexed)
pub fn part_ast(list: &Expr, index: &Expr) -> Result<Expr, InterpreterError> {
  // Try decomposing BinaryOp/UnaryOp to canonical form first
  if let Some((head_name, ha_args)) = expr_to_head_args(list) {
    let canonical = Expr::FunctionCall {
      name: head_name,
      args: ha_args.into(),
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
        args: vec![list.clone(), index.clone()].into(),
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
      Ok(Expr::List(results.into()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Part".to_string(),
      args: vec![list.clone(), index.clone()].into(),
    }),
  }
}

/// Insert[list, elem, n] - Insert element at position n (1-indexed).
/// Threads through any non-atomic head, so e.g. Insert[f[a,b], x, 2] returns
/// f[a, x, b].
fn insert_ins_message(path: &[i128], list: &Expr) {
  let pos = format!(
    "{{{}}}",
    path
      .iter()
      .map(|p| p.to_string())
      .collect::<Vec<_>>()
      .join(", ")
  );
  // wolframscript's Insert::ins text has no trailing period
  crate::emit_message(&format!(
    "Insert::ins: Cannot insert at position {} in {}",
    pos,
    crate::syntax::format_expr(list, crate::syntax::ExprForm::Output)
  ));
}

/// Insert `elem` along nested `paths` (lists of part indices; the last
/// index is the insertion point). All positions refer to the original
/// expression. Returns Err(failing_path) on any invalid position.
fn insert_paths(
  expr: &Expr,
  elem: &Expr,
  paths: &[Vec<i128>],
) -> Result<Expr, Vec<i128>> {
  let (items, head): (Vec<Expr>, Option<String>) = match expr {
    Expr::List(items) => (items.iter().cloned().collect(), None),
    Expr::FunctionCall { name, args } => {
      (args.iter().cloned().collect(), Some(name.clone()))
    }
    _ => return Err(paths[0].clone()),
  };
  let len = items.len() as i128;

  // Paths that descend deeper, grouped by their (0-based) first index
  let mut deeper: std::collections::BTreeMap<usize, Vec<Vec<i128>>> =
    Default::default();
  // Insertion indices at this level (0-based slots in the original)
  let mut here: Vec<usize> = Vec::new();
  for path in paths {
    let n = path[0];
    if path.len() == 1 {
      let idx = if n > 0 {
        if n - 1 > len {
          return Err(path.clone());
        }
        (n - 1) as usize
      } else if n < 0 {
        if n < -(len + 1) {
          return Err(path.clone());
        }
        (len + 1 + n) as usize
      } else {
        return Err(path.clone());
      };
      here.push(idx);
    } else {
      // Part index for descent
      let idx = if n > 0 {
        if n > len {
          return Err(path.clone());
        }
        (n - 1) as usize
      } else if n < 0 {
        if -n > len {
          return Err(path.clone());
        }
        (len + n) as usize
      } else {
        return Err(path.clone());
      };
      deeper.entry(idx).or_default().push(path[1..].to_vec());
    }
  }

  let mut result = items;
  // Recurse first (uses original child contents), tracking failures with
  // the full path restored
  for (idx, sub_paths) in &deeper {
    match insert_paths(&result[*idx], elem, sub_paths) {
      Ok(new_child) => result[*idx] = new_child,
      Err(mut failing) => {
        // Restore the leading index of the failing path
        let lead = paths
          .iter()
          .find(|p| p.len() > 1 && p[1..] == failing[..])
          .map(|p| p[0])
          .unwrap_or((*idx + 1) as i128);
        failing.insert(0, lead);
        return Err(failing);
      }
    }
  }
  // Insert at this level from rightmost to leftmost
  here.sort_unstable();
  for idx in here.into_iter().rev() {
    result.insert(idx, elem.clone());
  }
  Ok(match head {
    Some(name) => Expr::FunctionCall {
      name,
      args: result.into(),
    },
    None => Expr::List(result.into()),
  })
}

pub fn insert_ast(
  list: &Expr,
  elem: &Expr,
  pos: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "Insert".to_string(),
    args: vec![list.clone(), elem.clone(), pos.clone()].into(),
  };
  // Parse the position specification:
  //   n              single top-level position
  //   {p1, ..., pk}  one nested path
  //   {{...}, ...}   several paths
  let paths: Vec<Vec<i128>> = match pos {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      vec![vec![expr_to_i128(pos).unwrap()]]
    }
    Expr::List(inner)
      if !inner.is_empty()
        && inner.iter().all(|i| matches!(i, Expr::List(_))) =>
    {
      let mut ps = Vec::with_capacity(inner.len());
      for sub in inner.iter() {
        let Expr::List(sub_items) = sub else {
          unreachable!()
        };
        let path: Option<Vec<i128>> =
          sub_items.iter().map(expr_to_i128).collect();
        match path {
          Some(p) if !p.is_empty() => ps.push(p),
          _ => {
            crate::emit_message(&format!(
              "Insert::psl: Position specification {} in {} is not a machine-sized integer or a list of machine-sized integers.",
              crate::syntax::format_expr(pos, crate::syntax::ExprForm::Output),
              crate::syntax::format_expr(
                &unevaluated(),
                crate::syntax::ExprForm::Output
              )
            ));
            return Ok(unevaluated());
          }
        }
      }
      ps
    }
    Expr::List(inner) if !inner.is_empty() => {
      match inner
        .iter()
        .map(expr_to_i128)
        .collect::<Option<Vec<i128>>>()
      {
        Some(p) => vec![p],
        None => {
          crate::emit_message(&format!(
            "Insert::psl: Position specification {} in {} is not a machine-sized integer or a list of machine-sized integers.",
            crate::syntax::format_expr(pos, crate::syntax::ExprForm::Output),
            crate::syntax::format_expr(
              &unevaluated(),
              crate::syntax::ExprForm::Output
            )
          ));
          return Ok(unevaluated());
        }
      }
    }
    _ => {
      crate::emit_message(&format!(
        "Insert::psl: Position specification {} in {} is not a machine-sized integer or a list of machine-sized integers.",
        crate::syntax::format_expr(pos, crate::syntax::ExprForm::Output),
        crate::syntax::format_expr(
          &unevaluated(),
          crate::syntax::ExprForm::Output
        )
      ));
      return Ok(unevaluated());
    }
  };
  match insert_paths(list, elem, &paths) {
    Ok(result) => Ok(result),
    Err(failing) => {
      insert_ins_message(&failing, list);
      Ok(unevaluated())
    }
  }
}

/// Extract[list, n] - extracts element at position n
/// Extract[list, {n1, n2, ...}] - extracts element at nested position
/// A single component of an Extract path: a part index or an
/// association key.
enum ExtractComp {
  Idx(i128),
  Key(Expr),
}

/// Outcome of resolving one path: the part, or which message to emit.
enum ExtractOutcome {
  Found(Expr),
  /// Out-of-range index; Extract::partw always reports the *first*
  /// component of the failing path and the original subject.
  Partw,
  /// Path descends below the depth of the object.
  Partd,
  /// Missing association key (with the inner association it was
  /// looked up in); the result is Missing[KeyAbsent, Key[k]].
  Keyw(Expr, Expr),
}

fn extract_resolve(subject: &Expr, path: &[ExtractComp]) -> ExtractOutcome {
  let mut current = subject.clone();
  for comp in path {
    match comp {
      ExtractComp::Idx(0) => {
        current = match &current {
          Expr::CurriedCall { func, .. } => (**func).clone(),
          other => Expr::Identifier(
            crate::evaluator::pattern_matching::get_expr_head(other),
          ),
        };
      }
      ExtractComp::Idx(n) => {
        use crate::functions::expr_form::{ExprForm, decompose_expr};
        let items: Vec<Expr> = match &current {
          Expr::List(items) => items.to_vec(),
          Expr::FunctionCall { args, .. } => args.to_vec(),
          Expr::CurriedCall { args, .. } => args.clone(),
          Expr::Association(pairs) => {
            pairs.iter().map(|(_, v)| v.clone()).collect()
          }
          // Operator nodes (Plus, Power, …) decompose to head + children.
          other => match decompose_expr(other) {
            ExprForm::Composite { children, .. } => children,
            ExprForm::Atom(_) => return ExtractOutcome::Partd,
          },
        };
        let len = items.len() as i128;
        let idx = if *n < 0 { len + n } else { n - 1 };
        if idx < 0 || idx >= len {
          return ExtractOutcome::Partw;
        }
        current = items[idx as usize].clone();
      }
      ExtractComp::Key(key) => {
        let Expr::Association(pairs) = &current else {
          return ExtractOutcome::Partd;
        };
        let key_str = crate::syntax::expr_to_string(key);
        match pairs
          .iter()
          .find(|(k, _)| crate::syntax::expr_to_string(k) == key_str)
        {
          Some((_, v)) => {
            let value = v.clone();
            current = value;
          }
          None => {
            return ExtractOutcome::Keyw(key.clone(), current.clone());
          }
        }
      }
    }
  }
  ExtractOutcome::Found(current)
}

/// Unified Extract: `Extract[expr, pos]`, `Extract[expr, {pos1, …}]`,
/// and `Extract[expr, pos, h]` (h wraps each part and evaluates).
/// Paths may contain integers (0 is the head, negatives count from the
/// end), `Key[k]`, and strings (association keys). Emits ::psl1 for
/// inapplicable specs, ::partw / ::partd for bad paths, and ::keyw
/// (yielding Missing[KeyAbsent, …]) for absent keys.
pub fn extract_unified_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Expr::FunctionCall {
    name: "Extract".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);
  let subject = &args[0];
  let spec = &args[1];
  let head = args.get(2);

  let parse_comp = |e: &Expr| -> Option<ExtractComp> {
    match e {
      Expr::Integer(n) => Some(ExtractComp::Idx(*n)),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i128().map(ExtractComp::Idx)
      }
      Expr::String(_) => Some(ExtractComp::Key(e.clone())),
      Expr::FunctionCall { name, args } if name == "Key" && args.len() == 1 => {
        Some(ExtractComp::Key(args[0].clone()))
      }
      _ => None,
    }
  };
  let psl1 = || {
    crate::emit_message(&format!(
      "Extract::psl1: Position specification {} in {} is not applicable.",
      show(spec),
      show(&original())
    ));
    Ok(original())
  };

  // Parse the spec into one or many paths. A list of lists is a
  // multi-path spec; the empty list extracts nothing.
  enum Paths {
    Single(Vec<ExtractComp>),
    Multi(Vec<Vec<ExtractComp>>),
  }
  let paths = match spec {
    Expr::List(items) if items.iter().all(|e| matches!(e, Expr::List(_))) => {
      let mut multi = Vec::new();
      for item in items {
        let Expr::List(comps) = item else {
          unreachable!()
        };
        match comps.iter().map(parse_comp).collect::<Option<Vec<_>>>() {
          Some(path) => multi.push(path),
          None => return psl1(),
        }
      }
      Paths::Multi(multi)
    }
    Expr::List(items) => {
      match items.iter().map(parse_comp).collect::<Option<Vec<_>>>() {
        Some(path) => Paths::Single(path),
        None => return psl1(),
      }
    }
    other => match parse_comp(other) {
      Some(comp @ ExtractComp::Idx(_)) => Paths::Single(vec![comp]),
      _ => return psl1(),
    },
  };

  let wrap = |part: Expr| -> Result<Expr, InterpreterError> {
    match head {
      None => Ok(part),
      Some(Expr::Identifier(h)) => {
        crate::evaluator::evaluate_function_call_ast(h, &[part])
      }
      Some(h) => crate::evaluator::evaluate_expr_to_expr(&Expr::CurriedCall {
        func: Box::new(h.clone()),
        args: vec![part],
      }),
    }
  };

  let resolve_one =
    |path: &[ExtractComp]| -> Result<Option<Expr>, InterpreterError> {
      match extract_resolve(subject, path) {
        ExtractOutcome::Found(part) => Ok(Some(wrap(part)?)),
        ExtractOutcome::Partw => {
          let first = match path.first() {
            Some(ExtractComp::Idx(n)) => Expr::Integer(*n),
            Some(ExtractComp::Key(k)) => Expr::FunctionCall {
              name: "Key".to_string(),
              args: vec![k.clone()].into(),
            },
            None => Expr::Integer(0),
          };
          crate::emit_message(&format!(
            "Extract::partw: Part {} of {} does not exist.",
            show(&first),
            show(subject)
          ));
          Ok(None)
        }
        ExtractOutcome::Partd => {
          let path_expr = Expr::List(
            path
              .iter()
              .map(|c| match c {
                ExtractComp::Idx(n) => Expr::Integer(*n),
                ExtractComp::Key(k) => Expr::FunctionCall {
                  name: "Key".to_string(),
                  args: vec![k.clone()].into(),
                },
              })
              .collect(),
          );
          crate::emit_message(&format!(
            "Extract::partd: Part specification {} is longer than depth of object.",
            show(&path_expr)
          ));
          Ok(None)
        }
        ExtractOutcome::Keyw(key, container) => {
          crate::emit_message(&format!(
            "Extract::keyw: Key {} does not exist in {}.",
            show(&key),
            show(&container)
          ));
          let missing = Expr::FunctionCall {
            name: "Missing".to_string(),
            args: vec![
              Expr::String("KeyAbsent".to_string()),
              Expr::FunctionCall {
                name: "Key".to_string(),
                args: vec![key].into(),
              },
            ]
            .into(),
          };
          Ok(Some(missing))
        }
      }
    };

  match paths {
    Paths::Single(path) => match resolve_one(&path)? {
      Some(part) => Ok(part),
      None => Ok(original()),
    },
    Paths::Multi(multi) => {
      let mut out = Vec::with_capacity(multi.len());
      for path in &multi {
        match resolve_one(path)? {
          Some(part) => out.push(part),
          None => return Ok(original()),
        }
      }
      Ok(Expr::List(out.into()))
    }
  }
}

/// ReplacePart[list, n -> val] - replaces element at position n
/// ReplacePart[expr, rules] — replace parts by position. Positions may be
/// integers, nested paths, lists of paths, or patterns matching index
/// sequences (i_ binds the index; {_, 1} replaces the first element at
/// every depth-2 branch). Each part position takes the FIRST matching
/// rule; replaced parts are not re-examined. Atomic subjects come back
/// unchanged; invalid rule specifications emit ::reps.
pub fn replace_part_ast(
  expr: &Expr,
  rule: &Expr,
) -> Result<Expr, InterpreterError> {
  // Normalize the rule spec into an ordered list of (lhs, rhs, delayed)
  let as_rule = |r: &Expr| -> Option<(Expr, Expr, bool)> {
    match r {
      Expr::Rule {
        pattern,
        replacement,
      } => Some(((**pattern).clone(), (**replacement).clone(), false)),
      Expr::RuleDelayed {
        pattern,
        replacement,
      } => Some(((**pattern).clone(), (**replacement).clone(), true)),
      Expr::FunctionCall { name, args }
        if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
      {
        Some((args[0].clone(), args[1].clone(), name == "RuleDelayed"))
      }
      _ => None,
    }
  };
  let rules: Vec<(Expr, Expr, bool)> = match rule {
    Expr::List(items) if items.iter().all(|r| as_rule(r).is_some()) => {
      items.iter().map(|r| as_rule(r).unwrap()).collect()
    }
    single => match as_rule(single) {
      Some(r) => vec![r],
      None => {
        crate::emit_message(&format!(
          "ReplacePart::reps: {} is neither a list of replacement rules nor a valid dispatch table, and so cannot be used for replacing.",
          crate::syntax::format_expr(rule, crate::syntax::ExprForm::Output)
        ));
        return Ok(Expr::FunctionCall {
          name: "ReplacePart".to_string(),
          args: vec![expr.clone(), rule.clone()].into(),
        });
      }
    },
  };

  // Expand each rule LHS into one or more path matchers (each a list of
  // per-level components: integers or patterns)
  #[derive(Clone)]
  enum Comp {
    Index(i128),
    Pattern(Expr),
  }
  let to_comp = |e: &Expr| -> Comp {
    match e {
      Expr::Integer(n) => Comp::Index(*n),
      other => Comp::Pattern(other.clone()),
    }
  };
  // matcher -> (path components, rule index)
  let mut matchers: Vec<(Vec<Comp>, usize)> = Vec::new();
  for (ri, (lhs, _, _)) in rules.iter().enumerate() {
    match lhs {
      Expr::List(items)
        if !items.is_empty()
          && items.iter().all(|i| matches!(i, Expr::List(_))) =>
      {
        // {{p1...}, {p2...}}: several paths for the same rule
        for sub in items.iter() {
          if let Expr::List(parts) = sub {
            matchers.push((parts.iter().map(to_comp).collect(), ri));
          }
        }
      }
      Expr::List(items) if !items.is_empty() => {
        matchers.push((items.iter().map(to_comp).collect(), ri));
      }
      other => matchers.push((vec![to_comp(other)], ri)),
    }
  }
  let max_depth = matchers.iter().map(|(m, _)| m.len()).max().unwrap_or(0);

  // Walk the expression; at each child position try the matchers in rule
  // order against the full path so far
  fn walk(
    expr: &Expr,
    path: &mut Vec<(i128, i128)>, // (1-based index, level length)
    matchers: &[(Vec<Comp>, usize)],
    rules: &[(Expr, Expr, bool)],
    max_depth: usize,
  ) -> Result<Expr, InterpreterError> {
    let (items, head): (&[Expr], Option<&str>) = match expr {
      Expr::List(items) => (items.as_slice(), None),
      Expr::FunctionCall { name, args } => {
        (args.as_slice(), Some(name.as_str()))
      }
      _ => return Ok(expr.clone()),
    };
    let len = items.len() as i128;
    // Position 0 at this level replaces the head
    let mut new_head: Option<String> = None;
    for (comps, ri) in matchers {
      if comps.len() != path.len() + 1 {
        continue;
      }
      let mut all_match = true;
      for (comp, &(p_idx, p_len)) in comps.iter().zip(path.iter()) {
        if let Comp::Index(n) = comp {
          let norm = if *n >= 0 { *n } else { p_len + *n + 1 };
          if norm != p_idx {
            all_match = false;
            break;
          }
        } else {
          all_match = false; // patterns over head positions: unsupported
          break;
        }
      }
      if all_match && matches!(comps.last(), Some(Comp::Index(0))) {
        let (_, rhs, _) = &rules[*ri];
        let value = crate::evaluator::evaluate_expr_to_expr(rhs)?;
        if let Expr::Identifier(h) = &value {
          new_head = Some(h.clone());
          break;
        }
      }
    }
    let len = len;
    let mut out: Vec<Expr> = Vec::with_capacity(items.len());
    'next_item: for (i0, item) in items.iter().enumerate() {
      let idx = (i0 + 1) as i128;
      path.push((idx, len));
      // Try each matcher whose depth equals the current path length
      for (comps, ri) in matchers {
        if comps.len() != path.len() {
          continue;
        }
        let mut bindings: Vec<(String, Expr)> = Vec::new();
        let mut ok = true;
        for (comp, &(p_idx, p_len)) in comps.iter().zip(path.iter()) {
          match comp {
            Comp::Index(n) => {
              let norm = if *n >= 0 { *n } else { p_len + *n + 1 };
              if norm != p_idx {
                ok = false;
                break;
              }
            }
            Comp::Pattern(pat) => {
              match crate::evaluator::pattern_matching::match_pattern(
                &Expr::Integer(p_idx),
                pat,
              ) {
                Some(bs) => {
                  // Repeated pattern variables must bind consistently
                  for (name, val) in bs {
                    match bindings.iter().find(|(n, _)| *n == name) {
                      Some((_, prev))
                        if !crate::evaluator::pattern_matching::expr_equal(
                          prev, &val,
                        ) =>
                      {
                        ok = false;
                        break;
                      }
                      Some(_) => {}
                      None => bindings.push((name, val)),
                    }
                  }
                  if !ok {
                    break;
                  }
                }
                None => {
                  ok = false;
                  break;
                }
              }
            }
          }
        }
        if ok {
          let (_, rhs, _) = &rules[*ri];
          let mut value = rhs.clone();
          for (name, bound) in &bindings {
            value = crate::syntax::substitute_variable(&value, name, bound);
          }
          let value = crate::evaluator::evaluate_expr_to_expr(&value)?;
          out.push(value);
          path.pop();
          continue 'next_item;
        }
      }
      // No rule applies here: recurse if deeper matchers exist
      if path.len() < max_depth {
        let replaced = walk(item, path, matchers, rules, max_depth)?;
        out.push(replaced);
      } else {
        out.push(item.clone());
      }
      path.pop();
    }
    Ok(match (new_head, head) {
      (Some(h), _) => Expr::FunctionCall {
        name: h,
        args: out.into(),
      },
      (None, Some(h)) => Expr::FunctionCall {
        name: h.to_string(),
        args: out.into(),
      },
      (None, None) => Expr::List(out.into()),
    })
  }

  if !matches!(expr, Expr::List(_) | Expr::FunctionCall { .. }) {
    // Atomic subjects come back unchanged
    return Ok(expr.clone());
  }
  walk(expr, &mut Vec::new(), &matchers, &rules, max_depth)
}

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
      // Atomic subject: wolframscript shows the scalar position without
      // braces here (Part 1 of y does not exist.)
      crate::emit_message(&format!(
        "Delete::partw: Part {} of {} does not exist.",
        crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output),
        crate::syntax::format_expr(&args[0], crate::syntax::ExprForm::Output)
      ));
      return Ok(Expr::FunctionCall {
        name: "Delete".to_string(),
        args: args.to_vec().into(),
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
          args: args.to_vec().into(),
        });
      }
      return delete_at_position_general(items, pos, head_name);
    }
    Expr::List(pos_list) => {
      // Determine if this is a multi-part position {i, j, ...} or multiple positions {{p1}, {p2}, ...}
      let is_multiple_positions =
        pos_list.iter().all(|p| matches!(p, Expr::List(_)));

      if is_multiple_positions && !pos_list.is_empty() {
        // Multiple positions: Delete[expr, {{p1}, {p2}, ...}]. Positions
        // refer to the original expression: normalize negatives against
        // it, deduplicate (deleting the same part twice deletes once),
        // and apply deepest/rightmost first.
        let partw_path = |path: &[i128]| {
          let shown = if path.len() == 1 {
            format!("{{{}}}", path[0])
          } else {
            format!(
              "{{{}}}",
              path
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(", ")
            )
          };
          crate::emit_message(&format!(
            "Delete::partw: Part {} of {} does not exist.",
            shown,
            crate::syntax::format_expr(
              &args[0],
              crate::syntax::ExprForm::Output
            )
          ));
        };
        let mut positions: Vec<Vec<i128>> = Vec::new();
        for p in pos_list {
          if let Expr::List(inner) = p {
            let raw: Vec<i128> =
              inner.iter().filter_map(expr_to_i128).collect();
            if raw.len() != inner.len() || raw.is_empty() {
              return Ok(Expr::FunctionCall {
                name: "Delete".to_string(),
                args: args.to_vec().into(),
              });
            }
            // Normalize each index against the original structure
            let mut cursor = &args[0];
            let mut norm = Vec::with_capacity(raw.len());
            for (depth, &ix) in raw.iter().enumerate() {
              let sub_items: &[Expr] = match cursor {
                Expr::List(v) => v.as_slice(),
                Expr::FunctionCall { args: fa, .. } => fa.as_slice(),
                _ => {
                  crate::emit_message(&format!(
                    "Delete::partw: Part {} of {} does not exist.",
                    raw[depth],
                    crate::syntax::format_expr(
                      cursor,
                      crate::syntax::ExprForm::Output
                    )
                  ));
                  return Ok(Expr::FunctionCall {
                    name: "Delete".to_string(),
                    args: args.to_vec().into(),
                  });
                }
              };
              let len = sub_items.len() as i128;
              let pos_ix = if ix > 0 { ix } else { len + ix + 1 };
              if pos_ix < 1 || pos_ix > len {
                partw_path(&raw);
                return Ok(Expr::FunctionCall {
                  name: "Delete".to_string(),
                  args: args.to_vec().into(),
                });
              }
              norm.push(pos_ix);
              if depth + 1 < raw.len() {
                cursor = &sub_items[(pos_ix - 1) as usize];
              }
            }
            positions.push(norm);
          }
        }
        positions.sort();
        positions.dedup();
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
            match delete_at_deep_position(&result, pos)? {
              Ok(v) => result = v,
              Err(_) => {
                partw_path(pos);
                return Ok(Expr::FunctionCall {
                  name: "Delete".to_string(),
                  args: args.to_vec().into(),
                });
              }
            }
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
            match delete_at_deep_position(&args[0], &pos)? {
              Ok(v) => return Ok(v),
              Err(fail) => {
                // Descent into an atom names the inner subject (Part 1
                // of b); an out-of-range final index names the full path
                let atomic = !matches!(
                  fail.subject,
                  Expr::List(_) | Expr::FunctionCall { .. }
                );
                if atomic {
                  crate::emit_message(&format!(
                    "Delete::partw: Part {} of {} does not exist.",
                    fail.index,
                    crate::syntax::format_expr(
                      &fail.subject,
                      crate::syntax::ExprForm::Output
                    )
                  ));
                } else {
                  crate::emit_message(&format!(
                    "Delete::partw: Part {{{}}} of {} does not exist.",
                    pos
                      .iter()
                      .map(|p| p.to_string())
                      .collect::<Vec<_>>()
                      .join(", "),
                    crate::syntax::format_expr(
                      &args[0],
                      crate::syntax::ExprForm::Output
                    )
                  ));
                }
                return Ok(Expr::FunctionCall {
                  name: "Delete".to_string(),
                  args: args.to_vec().into(),
                });
              }
            }
          }
        }
      }
    }
    _ => {
      crate::emit_message(&format!(
        "Delete::pkspec: The expression {} cannot be used as a part specification. Use Key[{}] instead.",
        crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output),
        crate::syntax::format_expr(&args[1], crate::syntax::ExprForm::Output)
      ));
    }
  }

  Ok(Expr::FunctionCall {
    name: "Delete".to_string(),
    args: args.to_vec().into(),
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
        args: result_items.into(),
      },
      None => Expr::List(result_items.into()),
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
      args: items.to_vec().into(),
    });
  };
  if idx >= items.len() {
    return Ok(wrap(items.to_vec()));
  }
  let mut result = items.to_vec();
  result.remove(idx);
  Ok(wrap(result))
}

/// Delete element at a deep multi-part position {i, j, ...}.
/// On an out-of-range or atom-descent failure, returns the failing index
/// together with the sub-expression it could not index into (for
/// wolframscript-matching error messages).
pub(crate) struct DeepDeleteFailure {
  pub(crate) index: i128,
  pub(crate) subject: Expr,
}

fn delete_at_deep_position(
  expr: &Expr,
  pos: &[i128],
) -> Result<Result<Expr, DeepDeleteFailure>, InterpreterError> {
  if pos.is_empty() {
    return Ok(Ok(expr.clone()));
  }
  let (items, head): (Vec<Expr>, Option<String>) = match expr {
    Expr::List(items) => (items.to_vec(), None),
    Expr::FunctionCall { name, args } => (args.to_vec(), Some(name.clone())),
    _ => {
      return Ok(Err(DeepDeleteFailure {
        index: pos[0],
        subject: expr.clone(),
      }));
    }
  };

  // Position 0 at this level → delete the head. Mathematica folds an
  // "unheaded" argument list into the parent by turning it into a
  // Sequence so the outer head absorbs it.
  if pos[0] == 0 {
    let unheaded = Expr::FunctionCall {
      name: "Sequence".to_string(),
      args: items.into(),
    };
    if pos.len() == 1 {
      return Ok(Ok(unheaded));
    }
    return Ok(Err(DeepDeleteFailure {
      index: pos[1],
      subject: unheaded,
    }));
  }

  let len = items.len() as i128;
  let idx = if pos[0] > 0 && pos[0] <= len {
    (pos[0] - 1) as usize
  } else if pos[0] < 0 && (-pos[0]) <= len {
    (len + pos[0]) as usize
  } else {
    return Ok(Err(DeepDeleteFailure {
      index: pos[0],
      subject: expr.clone(),
    }));
  };

  let rebuild = |new_items: Vec<Expr>| -> Expr {
    match &head {
      Some(h) => flatten_sequences(Expr::FunctionCall {
        name: h.clone(),
        args: new_items.into(),
      }),
      None => flatten_sequences(Expr::List(new_items.into())),
    }
  };

  if pos.len() == 1 {
    let mut result = items;
    result.remove(idx);
    Ok(Ok(rebuild(result)))
  } else {
    match delete_at_deep_position(&items[idx], &pos[1..])? {
      Ok(sub) => {
        let mut result = items.clone();
        result[idx] = sub;
        Ok(Ok(rebuild(result)))
      }
      Err(fail) => Ok(Err(fail)),
    }
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
    Expr::List(items) => Expr::List(splice(items).into()),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: splice(args).into(),
    },
    _ => expr,
  }
}
