//! AST-based list helper functions.
//!
//! These functions work directly with `Expr` AST nodes, avoiding the string
//! round-trips and re-parsing that the original `list_helpers.rs` functions use.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Convert an Expr to a boolean value.
/// Returns Some(true) for Identifier("True"), Some(false) for Identifier("False").
pub fn expr_to_bool(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Identifier(s) if s == "True" => Some(true),
    Expr::Identifier(s) if s == "False" => Some(false),
    _ => None,
  }
}

/// Convert a boolean to an Expr.
pub fn bool_to_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}

/// Apply a function/predicate to an argument and return the resulting Expr.
/// Uses the existing apply_function_to_arg from evaluator.
pub fn apply_func_ast(
  func: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  crate::evaluator::apply_function_to_arg(func, arg)
}

/// AST-based Map: apply function to each element of a list or association.
/// Map[f, {a, b, c}] -> {f[a], f[b], f[c]}
/// Map[f, <|a -> 1, b -> 2|>] -> <|a -> f[1], b -> f[2]|>
pub fn map_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_func_ast(func, item))
        .collect();
      Ok(Expr::List(results?))
    }
    Expr::Association(items) => {
      // Map over association applies function to values only
      let results: Result<Vec<(Expr, Expr)>, InterpreterError> = items
        .iter()
        .map(|(key, val)| {
          let new_val = apply_func_ast(func, val)?;
          Ok((key.clone(), new_val))
        })
        .collect();
      Ok(Expr::Association(results?))
    }
    _ => {
      // Not a list or association, return unevaluated
      Ok(Expr::FunctionCall {
        name: "Map".to_string(),
        args: vec![func.clone(), list.clone()],
      })
    }
  }
}

/// AST-based Select: filter elements where predicate returns True.
/// Select[{a, b, c}, pred] -> elements where pred[elem] is True
/// Select[{a, b, c}, pred, n] -> first n elements where pred[elem] is True
pub fn select_ast(
  list: &Expr,
  pred: &Expr,
  n: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  // Select works on any expression with arguments, preserving the head
  let (items, head_name): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
    _ => {
      let mut args = vec![list.clone(), pred.clone()];
      if let Some(limit) = n {
        args.push(limit.clone());
      }
      return Ok(Expr::FunctionCall {
        name: "Select".to_string(),
        args,
      });
    }
  };

  let limit = match n {
    Some(expr) => match expr {
      Expr::Integer(i) => Some(*i as usize),
      _ => None,
    },
    None => None,
  };

  let mut kept = Vec::new();
  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      kept.push(item.clone());
      if let Some(lim) = limit
        && kept.len() >= lim
      {
        break;
      }
    }
  }

  // Preserve the original head
  match head_name {
    Some(name) => Ok(Expr::FunctionCall { name, args: kept }),
    None => Ok(Expr::List(kept)),
  }
}

/// AST-based AllTrue: check if predicate is true for all elements.
/// AllTrue[{a, b, c}, pred] -> True if pred[x] is True for all x
pub fn all_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AllTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) != Some(true) {
      return Ok(bool_to_expr(false));
    }
  }

  Ok(bool_to_expr(true))
}

/// AST-based AnyTrue: check if predicate is true for any element.
/// AnyTrue[{a, b, c}, pred] -> True if pred[x] is True for any x
pub fn any_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "AnyTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      return Ok(bool_to_expr(true));
    }
  }

  Ok(bool_to_expr(false))
}

/// AST-based NoneTrue: check if predicate is false for all elements.
/// NoneTrue[{a, b, c}, pred] -> True if pred[x] is False for all x
pub fn none_true_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NoneTrue".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      return Ok(bool_to_expr(false));
    }
  }

  Ok(bool_to_expr(true))
}

/// AST-based Fold/FoldList: fold a function over a list.
/// Fold[f, x, {a, b, c}] -> f[f[f[x, a], b], c]
pub fn fold_ast(
  func: &Expr,
  init: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Fold".to_string(),
        args: vec![func.clone(), init.clone(), list.clone()],
      });
    }
  };

  let mut acc = init.clone();
  for item in items {
    // Apply func[acc, item]
    acc = apply_func_to_two_args(func, &acc, item)?;
  }

  Ok(acc)
}

/// Apply a binary function to two arguments.
fn apply_func_to_two_args(
  func: &Expr,
  arg1: &Expr,
  arg2: &Expr,
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => crate::evaluator::evaluate_function_call_ast(
      name,
      &[arg1.clone(), arg2.clone()],
    ),
    Expr::Function { body } => {
      // Anonymous function with two slots
      let substituted =
        crate::syntax::substitute_slots(body, &[arg1.clone(), arg2.clone()]);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body } => {
      let mut substituted = (**body).clone();
      let args_vec = [arg1, arg2];
      for (param, arg) in params.iter().zip(args_vec.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args } => {
      // Curried function: f[a] applied to (b, c) becomes f[a, b, c]
      let mut new_args = args.clone();
      new_args.push(arg1.clone());
      new_args.push(arg2.clone());
      crate::evaluator::evaluate_function_call_ast(name, &new_args)
    }
    _ => {
      let func_str = crate::syntax::expr_to_string(func);
      crate::evaluator::evaluate_function_call_ast(
        &func_str,
        &[arg1.clone(), arg2.clone()],
      )
    }
  }
}

/// AST-based CountBy: count elements by the value of a function.
/// CountBy[{a, b, c}, f] -> association of f[x] -> count
pub fn count_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CountBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, i128> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if let Some(count) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, 1);
    }
  }

  // Build association preserving order
  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let count = counts[&k];
      let key_expr = crate::syntax::string_to_expr(&k).unwrap_or(Expr::Raw(k));
      (key_expr, Expr::Integer(count))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// AST-based GroupBy: group elements by the value of a function.
/// GroupBy[{a, b, c}, f] -> association of f[x] -> {elements with that f value}
pub fn group_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "GroupBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut groups: HashMap<String, Vec<Expr>> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if let Some(group) = groups.get_mut(&key_str) {
      group.push(item.clone());
    } else {
      order.push(key_str.clone());
      groups.insert(key_str, vec![item.clone()]);
    }
  }

  // Build association preserving order
  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let items = groups.remove(&k).unwrap();
      let key_expr = crate::syntax::string_to_expr(&k).unwrap_or(Expr::Raw(k));
      (key_expr, Expr::List(items))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// Wolfram canonical ordering for expressions.
/// For strings: case-insensitive first, then lowercase before uppercase for ties.
/// For numbers: numeric comparison.
/// Mixed: numbers before strings.
/// Extract (real, imaginary) parts from a numeric expression for sorting.
/// Returns None for non-numeric expressions.
fn expr_to_complex_parts(e: &Expr) -> Option<(f64, f64)> {
  use crate::functions::math_ast::try_eval_to_f64;
  // Pure real number
  if let Some(v) = try_eval_to_f64(e) {
    return Some((v, 0.0));
  }
  // Check if expression contains I (complex unit)
  let s = crate::syntax::expr_to_string(e);
  if !s.contains('I') {
    return None;
  }
  match e {
    // Pure imaginary: n*I (BinaryOp form)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(right.as_ref(), Expr::Identifier(name) if name == "I")
        && let Some(im) = try_eval_to_f64(left)
      {
        return Some((0.0, im));
      }
      if matches!(left.as_ref(), Expr::Identifier(name) if name == "I")
        && let Some(im) = try_eval_to_f64(right)
      {
        return Some((0.0, im));
      }
      None
    }
    // a + b*I (BinaryOp form)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left,
      right,
    } => {
      if let Some(re) = try_eval_to_f64(left)
        && let Some((_, im)) = expr_to_complex_parts(right)
      {
        return Some((re, im));
      }
      None
    }
    // a - b*I (BinaryOp form)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left,
      right,
    } => {
      if let Some(re) = try_eval_to_f64(left)
        && let Some((_, im)) = expr_to_complex_parts(right)
      {
        return Some((re, -im));
      }
      None
    }
    // FunctionCall Plus[re, Times[im, I]]
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      if let Some(re) = try_eval_to_f64(&args[0])
        && let Some((_, im)) = expr_to_complex_parts(&args[1])
      {
        return Some((re, im));
      }
      if let Some(re) = try_eval_to_f64(&args[1])
        && let Some((_, im)) = expr_to_complex_parts(&args[0])
      {
        return Some((0.0 + im, re)); // im is imaginary coefficient
      }
      None
    }
    // FunctionCall Times[n, I]
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if matches!(&args[1], Expr::Identifier(n) if n == "I")
        && let Some(im) = try_eval_to_f64(&args[0])
      {
        return Some((0.0, im));
      }
      if matches!(&args[0], Expr::Identifier(n) if n == "I")
        && let Some(im) = try_eval_to_f64(&args[1])
      {
        return Some((0.0, im));
      }
      None
    }
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      if let (Some(re), Some(im)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
      {
        return Some((re, im));
      }
      None
    }
    // Just I
    Expr::Identifier(name) if name == "I" => Some((0.0, 1.0)),
    // Negated: -I, -(a+bI)
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      if let Some((re, im)) = expr_to_complex_parts(operand) {
        return Some((-re, -im));
      }
      None
    }
    _ => None,
  }
}

pub fn canonical_cmp(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  // Try numeric comparison (including complex numbers)
  let a_num = expr_to_complex_parts(a);
  let b_num = expr_to_complex_parts(b);

  match (a_num, b_num) {
    (Some((a_re, a_im)), Some((b_re, b_im))) => {
      // Both numeric: compare by real part first, then imaginary part
      match a_re.partial_cmp(&b_re).unwrap_or(std::cmp::Ordering::Equal) {
        std::cmp::Ordering::Equal => {
          // Same real part: pure reals (im=0) come first
          if a_im == 0.0 && b_im != 0.0 {
            return std::cmp::Ordering::Less;
          }
          if a_im != 0.0 && b_im == 0.0 {
            return std::cmp::Ordering::Greater;
          }
          a_im.partial_cmp(&b_im).unwrap_or(std::cmp::Ordering::Equal)
        }
        other => other,
      }
    }
    (Some(_), None) => std::cmp::Ordering::Less, // numbers before non-numbers
    (None, Some(_)) => std::cmp::Ordering::Greater,
    (None, None) => {
      // Non-numeric: string comparison
      let sa = crate::syntax::expr_to_string(a);
      let sb = crate::syntax::expr_to_string(b);
      let la = sa.to_lowercase();
      let lb = sb.to_lowercase();
      match la.cmp(&lb) {
        std::cmp::Ordering::Equal => {
          for (ca, cb) in sa.chars().zip(sb.chars()) {
            if ca != cb {
              if ca.to_lowercase().eq(cb.to_lowercase()) {
                if ca.is_lowercase() {
                  return std::cmp::Ordering::Less;
                } else {
                  return std::cmp::Ordering::Greater;
                }
              }
              return ca.cmp(&cb);
            }
          }
          sa.len().cmp(&sb.len())
        }
        other => other,
      }
    }
  }
}

/// AST-based SortBy: sort elements by the value of a function.
/// SortBy[{a, b, c}, f] -> elements sorted by f[x]
pub fn sort_by_ast(list: &Expr, func: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SortBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  // Compute keys for each element
  let mut keyed: Vec<(Expr, Expr)> = items
    .into_iter()
    .map(|item| {
      let key = apply_func_ast(func, &item)?;
      Ok((item, key))
    })
    .collect::<Result<_, InterpreterError>>()?;

  // Sort by key, using canonical ordering as tiebreaker
  keyed.sort_by(|a, b| {
    let key_ord = canonical_cmp(&a.1, &b.1);
    if key_ord == std::cmp::Ordering::Equal {
      canonical_cmp(&a.0, &b.0)
    } else {
      key_ord
    }
  });

  Ok(Expr::List(
    keyed.into_iter().map(|(item, _)| item).collect(),
  ))
}

///// Ordering[list
pub fn ordering_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Ordering expects 1 or 2 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Ordering".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut indexed: Vec<(usize, &Expr)> = items.iter().enumerate().collect();

  indexed.sort_by(|a, b| {
    let va = crate::syntax::expr_to_string(a.1);
    let vb = crate::syntax::expr_to_string(b.1);
    if let (Ok(na), Ok(nb)) = (va.parse::<f64>(), vb.parse::<f64>()) {
      na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
    } else {
      va.cmp(&vb)
    }
  });

  let mut result: Vec<Expr> = indexed
    .iter()
    .map(|(idx, _)| Expr::Integer((*idx + 1) as i128))
    .collect();

  if args.len() == 2
    && let Some(n) = expr_to_i128(&args[1])
  {
    let n = n;
    if n >= 0 {
      result.truncate(n as usize);
    } else {
      // Negative n: take last |n| elements (largest positions)
      let abs_n = n.unsigned_abs() as usize;
      if abs_n <= result.len() {
        result = result.split_off(result.len() - abs_n);
      }
    }
  }

  Ok(Expr::List(result))
}

/// MinimalBy[list, f] - Returns all elements that minimize f
pub fn minimal_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => return Ok(Expr::List(vec![])),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MinimalBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  let keyed: Vec<(Expr, Expr)> = items
    .iter()
    .map(|item| {
      let key = apply_func_ast(func, item)?;
      Ok((item.clone(), key))
    })
    .collect::<Result<_, InterpreterError>>()?;

  let min_key = keyed
    .iter()
    .map(|(_, k)| k)
    .min_by(|a, b| {
      let ka = crate::syntax::expr_to_string(a);
      let kb = crate::syntax::expr_to_string(b);
      if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
      } else {
        ka.cmp(&kb)
      }
    })
    .cloned();

  if let Some(min_k) = min_key {
    let min_str = crate::syntax::expr_to_string(&min_k);
    let result: Vec<Expr> = keyed
      .into_iter()
      .filter(|(_, k)| crate::syntax::expr_to_string(k) == min_str)
      .map(|(item, _)| item)
      .collect();
    Ok(Expr::List(result))
  } else {
    Ok(Expr::List(vec![]))
  }
}

/// MaximalBy[list, f] - Returns all elements that maximize f
pub fn maximal_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => return Ok(Expr::List(vec![])),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MaximalBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  let keyed: Vec<(Expr, Expr)> = items
    .iter()
    .map(|item| {
      let key = apply_func_ast(func, item)?;
      Ok((item.clone(), key))
    })
    .collect::<Result<_, InterpreterError>>()?;

  let max_key = keyed
    .iter()
    .map(|(_, k)| k)
    .max_by(|a, b| {
      let ka = crate::syntax::expr_to_string(a);
      let kb = crate::syntax::expr_to_string(b);
      if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
      } else {
        ka.cmp(&kb)
      }
    })
    .cloned();

  if let Some(max_k) = max_key {
    let max_str = crate::syntax::expr_to_string(&max_k);
    let result: Vec<Expr> = keyed
      .into_iter()
      .filter(|(_, k)| crate::syntax::expr_to_string(k) == max_str)
      .map(|(item, _)| item)
      .collect();
    Ok(Expr::List(result))
  } else {
    Ok(Expr::List(vec![]))
  }
}

/// MapAt[f, list, pos] - Apply function at specific positions
/// Supports single integer, list of integers, and negative indices
pub fn map_at_ast(
  func: &Expr,
  list: &Expr,
  pos_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapAt".to_string(),
        args: vec![func.clone(), list.clone(), pos_spec.clone()],
      });
    }
  };

  let len = items.len() as i128;

  // Collect positions to modify
  // Wolfram uses {{1}, {3}} for multiple positions (list of single-element lists)
  let positions: Vec<i128> = match pos_spec {
    Expr::Integer(n) => vec![*n],
    Expr::BigInteger(_) => match expr_to_i128(pos_spec) {
      Some(n) => vec![n],
      None => {
        return Ok(Expr::FunctionCall {
          name: "MapAt".to_string(),
          args: vec![func.clone(), list.clone(), pos_spec.clone()],
        });
      }
    },
    Expr::List(pos_list) => {
      // Each element must be a single-element list like {1}
      let mut positions = Vec::new();
      for p in pos_list {
        match p {
          Expr::List(inner) if inner.len() == 1 => {
            if let Some(n) = expr_to_i128(&inner[0]) {
              positions.push(n);
            }
          }
          _ => {
            return Ok(Expr::FunctionCall {
              name: "MapAt".to_string(),
              args: vec![func.clone(), list.clone(), pos_spec.clone()],
            });
          }
        }
      }
      positions
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapAt".to_string(),
        args: vec![func.clone(), list.clone(), pos_spec.clone()],
      });
    }
  };

  let mut indices = std::collections::HashSet::new();
  for p in positions {
    let idx = if p < 0 {
      (len + p) as usize
    } else {
      (p - 1) as usize
    };
    if idx < items.len() {
      indices.insert(idx);
    }
  }

  let result: Result<Vec<Expr>, _> = items
    .iter()
    .enumerate()
    .map(|(i, item)| {
      if indices.contains(&i) {
        apply_func_ast(func, item)
      } else {
        Ok(item.clone())
      }
    })
    .collect();

  Ok(Expr::List(result?))
}

/// AST-based Nest: apply a function n times.
/// Nest[f, x, n] -> f[f[f[...f[x]...]]] (n times)
pub fn nest_ast(
  func: &Expr,
  init: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Nest requires non-negative count".into(),
    ));
  }

  let mut result = init.clone();
  for _ in 0..n {
    result = apply_func_ast(func, &result)?;
  }

  Ok(result)
}

/// AST-based NestList: build a list by repeatedly applying a function.
/// NestList[f, x, n] -> {x, f[x], f[f[x]], ..., f^n[x]}
pub fn nest_list_ast(
  func: &Expr,
  init: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "NestList requires non-negative count".into(),
    ));
  }

  let mut results = vec![init.clone()];
  let mut current = init.clone();
  for _ in 0..n {
    current = apply_func_ast(func, &current)?;
    results.push(current.clone());
  }

  Ok(Expr::List(results))
}

/// AST-based FixedPoint: apply function until result stops changing.
/// FixedPoint[f, x] -> fixed point of f starting from x
pub fn fixed_point_ast(
  func: &Expr,
  init: &Expr,
  max_iterations: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut current = init.clone();

  for _ in 0..max {
    let next = apply_func_ast(func, &current)?;
    let current_str = crate::syntax::expr_to_string(&current);
    let next_str = crate::syntax::expr_to_string(&next);
    if current_str == next_str {
      return Ok(current);
    }
    current = next;
  }

  Ok(current)
}

/// AST-based Cases: select elements matching a pattern.
pub fn cases_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Cases".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let mut kept = Vec::new();
  for item in items {
    if matches_pattern_ast(item, pattern) {
      kept.push(item.clone());
    } else {
      // Fall back to string matching for compatibility
      let item_str = crate::syntax::expr_to_string(item);
      let pattern_str = crate::syntax::expr_to_string(pattern);
      if matches_pattern_simple(&item_str, &pattern_str) {
        kept.push(item.clone());
      }
    }
  }

  Ok(Expr::List(kept))
}

/// Simple pattern matching for Cases.
/// This handles basic patterns like x_, _Integer, etc.
fn matches_pattern_simple(value: &str, pattern: &str) -> bool {
  // Match any value
  if pattern == "_" {
    return true;
  }

  // Named blank pattern like x_
  if pattern.ends_with('_')
    && !pattern.contains("_Integer")
    && !pattern.contains("_Real")
  {
    return true;
  }

  // Type patterns
  if pattern == "_Integer" {
    return value.parse::<i128>().is_ok();
  }
  if pattern == "_Real" {
    return value.parse::<f64>().is_ok() && value.contains('.');
  }
  if pattern == "_String" {
    return value.starts_with('"') && value.ends_with('"');
  }
  if pattern == "_List" {
    return value.starts_with('{') && value.ends_with('}');
  }

  // Literal match
  value == pattern
}

/// AST-based pattern matching for expressions.
/// Supports: Blank (_), named patterns (x_), head patterns (_Integer, _List, etc.),
/// Except, Alternatives, and literal matching.
pub fn matches_pattern_ast(expr: &Expr, pattern: &Expr) -> bool {
  match pattern {
    // Blank pattern: _ matches anything
    Expr::Pattern {
      name: _,
      head: None,
    } => true,
    // Head-constrained pattern: _Integer, _List, etc.
    Expr::Pattern {
      name: _,
      head: Some(h),
    } => get_expr_head_str(expr) == h,
    // Identifier patterns like "_", "_Integer", "_List", etc.
    Expr::Identifier(s) if s == "_" => true,
    Expr::Identifier(s) if s.starts_with('_') => {
      let head = &s[1..];
      get_expr_head_str(expr) == head
    }
    // Except[pattern] - matches anything that doesn't match the inner pattern
    Expr::FunctionCall { name, args }
      if name == "Except" && args.len() == 1 =>
    {
      !matches_pattern_ast(expr, &args[0])
    }
    // Alternatives: a | b - matches if either side matches
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      left,
      right,
    } => matches_pattern_ast(expr, left) || matches_pattern_ast(expr, right),
    // Structural matching for lists: {_, _} matches {1, 2}
    Expr::List(pat_items) => {
      if let Expr::List(expr_items) = expr {
        pat_items.len() == expr_items.len()
          && pat_items
            .iter()
            .zip(expr_items.iter())
            .all(|(p, e)| matches_pattern_ast(e, p))
      } else {
        false
      }
    }
    // Structural matching for function calls: f[_
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } => {
      if pat_name == "Except"
        || pat_name == "PatternTest"
        || pat_name == "Condition"
      {
        // Already handled above or not a structural match
        let pattern_str = crate::syntax::expr_to_string(pattern);
        let expr_str = crate::syntax::expr_to_string(expr);
        expr_str == pattern_str
      } else if let Expr::FunctionCall {
        name: expr_name,
        args: expr_args,
      } = expr
      {
        pat_name == expr_name
          && pat_args.len() == expr_args.len()
          && pat_args
            .iter()
            .zip(expr_args.iter())
            .all(|(p, e)| matches_pattern_ast(e, p))
      } else {
        false
      }
    }
    // Literal comparison
    _ => {
      let pattern_str = crate::syntax::expr_to_string(pattern);
      let expr_str = crate::syntax::expr_to_string(expr);
      expr_str == pattern_str
    }
  }
}

/// Get the head of an expression as a string
fn get_expr_head_str(expr: &Expr) -> &str {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => "Integer",
    Expr::Real(_) | Expr::BigFloat(_, _) => "Real",
    Expr::String(_) => "String",
    Expr::List(_) => "List",
    Expr::FunctionCall { name, .. } => name,
    Expr::Association(_) => "Association",
    _ => "Symbol",
  }
}

/// Cases with level specification: Cases[list, pattern, {level}]
pub fn cases_with_level_ast(
  list: &Expr,
  pattern: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Parse level spec: {n} means exactly level n
  let level = match level_spec {
    Expr::List(items) if items.len() == 1 => {
      expr_to_i128(&items[0]).unwrap_or(1) as usize
    }
    _ => 1,
  };

  let mut results = Vec::new();
  collect_at_level(list, pattern, level, 0, &mut results);
  Ok(Expr::List(results))
}

/// Recursively collect elements matching pattern at a specific level
fn collect_at_level(
  expr: &Expr,
  pattern: &Expr,
  target_level: usize,
  current_level: usize,
  results: &mut Vec<Expr>,
) {
  if current_level == target_level {
    if matches_pattern_ast(expr, pattern) {
      results.push(expr.clone());
    }
    return;
  }

  // Recurse into sublists/subexpressions
  match expr {
    Expr::List(items) => {
      for item in items {
        collect_at_level(
          item,
          pattern,
          target_level,
          current_level + 1,
          results,
        );
      }
    }
    Expr::FunctionCall { args, .. } => {
      for arg in args {
        collect_at_level(
          arg,
          pattern,
          target_level,
          current_level + 1,
          results,
        );
      }
    }
    _ => {}
  }
}

/// AST-based Position: find positions of elements matching a pattern.
pub fn position_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Position".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let pattern_str = crate::syntax::expr_to_string(pattern);
  let mut positions = Vec::new();

  for (i, item) in items.iter().enumerate() {
    let item_str = crate::syntax::expr_to_string(item);
    if matches_pattern_simple(&item_str, &pattern_str) {
      // 1-indexed
      positions.push(Expr::List(vec![Expr::Integer((i + 1) as i128)]));
    }
  }

  Ok(Expr::List(positions))
}

/// FirstPosition[list, pattern] - finds the position of the first element matching pattern
/// Returns {index} or Missing["NotFound"] if not found
pub fn first_position_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "FirstPosition expects at least 2 arguments".into(),
    ));
  }
  let default = if args.len() >= 3 {
    args[2].clone()
  } else {
    Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![Expr::String("NotFound".to_string())],
    }
  };

  fn find_first(
    expr: &Expr,
    pattern: &Expr,
    path: &mut Vec<i128>,
  ) -> Option<Vec<i128>> {
    let pattern_str = crate::syntax::expr_to_string(pattern);
    let expr_str = crate::syntax::expr_to_string(expr);
    if matches_pattern_simple(&expr_str, &pattern_str)
      || matches_pattern_ast(expr, pattern)
    {
      return Some(path.clone());
    }
    if let Expr::List(items) = expr {
      for (i, item) in items.iter().enumerate() {
        path.push((i + 1) as i128);
        if let Some(result) = find_first(item, pattern, path) {
          return Some(result);
        }
        path.pop();
      }
    }
    None
  }

  let mut path = Vec::new();
  match find_first(&args[0], &args[1], &mut path) {
    Some(indices) => {
      Ok(Expr::List(indices.into_iter().map(Expr::Integer).collect()))
    }
    None => Ok(default),
  }
}

/// AST-based MapIndexed: apply function with index to each element.
/// MapIndexed[f, {a, b, c}] -> {f[a, {1}], f[b, {2}], f[c, {3}]}
pub fn map_indexed_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapIndexed".to_string(),
        args: vec![func.clone(), list.clone()],
      });
    }
  };

  let results: Result<Vec<Expr>, _> = items
    .iter()
    .enumerate()
    .map(|(i, item)| {
      let index = Expr::List(vec![Expr::Integer((i + 1) as i128)]);
      apply_func_to_two_args(func, item, &index)
    })
    .collect();

  Ok(Expr::List(results?))
}

/// AST-based Tally: count occurrences of each element.
/// Tally[{a, b, a, c, b, a}] -> {{a, 3}, {b, 2}, {c, 1}}
pub fn tally_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tally".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, (Expr, i128)> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key_str = crate::syntax::expr_to_string(item);
    if let Some((_, count)) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, (item.clone(), 1));
    }
  }

  let pairs: Vec<Expr> = order
    .into_iter()
    .map(|k| {
      let (expr, count) = counts.remove(&k).unwrap();
      Expr::List(vec![expr, Expr::Integer(count)])
    })
    .collect();

  Ok(Expr::List(pairs))
}

///// Counts[list] - Returns association of distinct elements with their counts
/// Counts[{a, b, a, c, b, a}] -> <|a -> 3, b -> 2, c -> 1|>
pub fn counts_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Counts".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  use std::collections::HashMap;
  let mut counts: HashMap<String, (Expr, i128)> = HashMap::new();
  let mut order: Vec<String> = Vec::new();

  for item in items {
    let key_str = crate::syntax::expr_to_string(item);
    if let Some((_, count)) = counts.get_mut(&key_str) {
      *count += 1;
    } else {
      order.push(key_str.clone());
      counts.insert(key_str, (item.clone(), 1));
    }
  }

  let pairs: Vec<(Expr, Expr)> = order
    .into_iter()
    .map(|k| {
      let (expr, count) = counts.remove(&k).unwrap();
      (expr, Expr::Integer(count))
    })
    .collect();

  Ok(Expr::Association(pairs))
}

/// AST-based DeleteDuplicates: remove duplicate elements.
/// DeleteDuplicates[{a, b, a, c, b}] -> {a, b, c}
pub fn delete_duplicates_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteDuplicates".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  use std::collections::HashSet;
  let mut seen: HashSet<String> = HashSet::new();
  let mut result = Vec::new();

  for item in items {
    let key_str = crate::syntax::expr_to_string(item);
    if seen.insert(key_str) {
      result.push(item.clone());
    }
  }

  Ok(Expr::List(result))
}

/// AST-based Union: combine lists and remove duplicates.
pub fn union_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  use std::collections::HashSet;
  let mut seen: HashSet<String> = HashSet::new();
  let mut result = Vec::new();

  for list in lists {
    let items = match list {
      Expr::List(items) => items,
      _ => continue,
    };
    for item in items {
      let key_str = crate::syntax::expr_to_string(item);
      if seen.insert(key_str) {
        result.push(item.clone());
      }
    }
  }

  // Union sorts its result in Mathematica
  result.sort_by(|a, b| {
    let ka = crate::syntax::expr_to_string(a);
    let kb = crate::syntax::expr_to_string(b);
    // Try numeric comparison first
    if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
      na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
    } else {
      ka.cmp(&kb)
    }
  });

  Ok(Expr::List(result))
}

/// AST-based Intersection: find common elements.
pub fn intersection_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  use std::collections::HashSet;

  // Start with elements from first list
  let first_items = match &lists[0] {
    Expr::List(items) => items,
    _ => return Ok(Expr::List(vec![])),
  };

  let mut common: HashSet<String> = first_items
    .iter()
    .map(crate::syntax::expr_to_string)
    .collect();

  // Intersect with each subsequent list
  for list in lists.iter().skip(1) {
    let items = match list {
      Expr::List(items) => items,
      _ => continue,
    };
    let list_set: HashSet<String> =
      items.iter().map(crate::syntax::expr_to_string).collect();
    common = common.intersection(&list_set).cloned().collect();
  }

  // Collect matching elements and sort canonically (like Mathematica)
  let mut result: Vec<Expr> = first_items
    .iter()
    .filter(|item| common.contains(&crate::syntax::expr_to_string(item)))
    .cloned()
    .collect();
  result.sort_by(|a, b| {
    let ord = compare_exprs(a, b);
    // compare_exprs returns 1 if a precedes b (canonical order), -1 if b precedes a
    if ord > 0 {
      std::cmp::Ordering::Less
    } else if ord < 0 {
      std::cmp::Ordering::Greater
    } else {
      std::cmp::Ordering::Equal
    }
  });
  result.dedup_by(|a, b| {
    crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
  });

  Ok(Expr::List(result))
}

/// AST-based Complement: elements in first list not in others.
pub fn complement_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  use std::collections::HashSet;

  // Extract items from any expression (List or FunctionCall)
  fn get_items(expr: &Expr) -> Option<(&[Expr], Option<&str>)> {
    match expr {
      Expr::List(items) => Some((items.as_slice(), None)),
      Expr::FunctionCall { name, args } => {
        Some((args.as_slice(), Some(name.as_str())))
      }
      _ => None,
    }
  }

  let (first_items, head_name) = match get_items(&lists[0]) {
    Some(r) => r,
    None => return Ok(Expr::List(vec![])),
  };

  // Get elements to exclude from all lists after the first
  let mut exclude: HashSet<String> = HashSet::new();
  for list in lists.iter().skip(1) {
    if let Some((items, _)) = get_items(list) {
      for item in items {
        exclude.insert(crate::syntax::expr_to_string(item));
      }
    }
  }

  // Filter first list, also remove duplicates and sort
  let mut seen = HashSet::new();
  let mut result: Vec<Expr> = first_items
    .iter()
    .filter(|item| {
      let s = crate::syntax::expr_to_string(item);
      !exclude.contains(&s) && seen.insert(s)
    })
    .cloned()
    .collect();

  // Sort by string representation (Wolfram sorts Complement output)
  result.sort_by(|a, b| {
    crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
  });

  match head_name {
    Some(name) => Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: result,
    }),
    None => Ok(Expr::List(result)),
  }
}

/// AST-based Table: generate a table of values.
/// Table[expr, {i, min, max}] -> {expr with i=min, ..., expr with i=max}
/// Table[expr, {i, max}] -> {expr with i=1, ..., expr with i=max}
/// Table[expr, {i, {list}}] -> {expr with i=elem1, expr with i=elem2, ...}
/// Table[expr, n] -> {expr, expr, ..., expr} (n times)
/// Multi-dimensional Table: Table[expr, iter1, iter2, ...]
/// Recursively nests Table from last iterator to first.
pub fn table_multi_ast(
  body: &Expr,
  iters: &[Expr],
) -> Result<Expr, InterpreterError> {
  if iters.len() == 1 {
    return table_ast(body, &iters[0]);
  }
  // Nest: the body for the outer iterator is Table[body, inner_iters...]
  // We build from the innermost outward
  let inner_body = Expr::FunctionCall {
    name: "Table".to_string(),
    args: std::iter::once(body.clone())
      .chain(iters[1..].iter().cloned())
      .collect(),
  };
  table_ast(&inner_body, &iters[0])
}

pub fn table_ast(
  body: &Expr,
  iter_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  match iter_spec {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      // Simple form: Table[expr, n]
      let n = expr_to_i128(iter_spec).ok_or_else(|| {
        InterpreterError::EvaluationError("Table: count too large".into())
      })?;
      if n < 0 {
        return Err(InterpreterError::EvaluationError(
          "Table: count must be non-negative".into(),
        ));
      }
      let mut results = Vec::new();
      for _ in 0..n {
        let val = crate::evaluator::evaluate_expr_to_expr(body)?;
        results.push(val);
      }
      Ok(Expr::List(results))
    }
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::List(vec![]));
      }

      // Handle {n} form (single element = just repeat count, no variable)
      if items.len() == 1 {
        let evaluated = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
        let n = expr_to_i128(&evaluated).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Table: iterator bound must be an integer".into(),
          )
        })?;
        let mut results = Vec::new();
        for _ in 0..n {
          let val = crate::evaluator::evaluate_expr_to_expr(body)?;
          results.push(val);
        }
        return Ok(Expr::List(results));
      }

      // Extract iterator variable
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Table: iterator variable must be an identifier".into(),
          ));
        }
      };

      if items.len() == 2 {
        // Check if second element is a list (iterate over list)
        let second = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        match second {
          Expr::List(list_items) => {
            // {i, {a, b, c}} form - iterate over list elements
            let mut results = Vec::new();
            for item in list_items {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &item);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
            }
            return Ok(Expr::List(results));
          }
          _ => {
            // {i, max} form - iterate from 1 to max
            let max_val = expr_to_i128(&second).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "Table: iterator bound must be an integer".into(),
              )
            })?;
            let mut results = Vec::new();
            for i in 1..=max_val {
              let substituted = crate::syntax::substitute_variable(
                body,
                &var_name,
                &Expr::Integer(i),
              );
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
            }
            return Ok(Expr::List(results));
          }
        }
      } else if items.len() >= 3 {
        // {i, min, max} or {i, min, max, step} form
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;

        // Get step (default is 1)
        let step_expr = if items.len() >= 4 {
          crate::evaluator::evaluate_expr_to_expr(&items[3])?
        } else {
          Expr::Integer(1)
        };

        // Keep exact integer iteration behavior when possible.
        if let (Some(min_val), Some(max_val), Some(step_val)) = (
          expr_to_i128(&min_expr),
          expr_to_i128(&max_expr),
          expr_to_i128(&step_expr),
        ) {
          if step_val == 0 {
            return Err(InterpreterError::EvaluationError(
              "Table: step cannot be zero".into(),
            ));
          }

          let mut results = Vec::new();
          let mut i = min_val;
          if step_val > 0 {
            while i <= max_val {
              let substituted = crate::syntax::substitute_variable(
                body,
                &var_name,
                &Expr::Integer(i),
              );
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
              i += step_val;
            }
          } else {
            // Negative step
            while i >= max_val {
              let substituted = crate::syntax::substitute_variable(
                body,
                &var_name,
                &Expr::Integer(i),
              );
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              results.push(val);
              i += step_val;
            }
          }
          return Ok(Expr::List(results));
        }

        // Fallback: numeric iteration for symbolic numeric bounds/step (e.g. Pi/4).
        crate::functions::math_ast::try_eval_to_f64(&min_expr).ok_or_else(
          || {
            InterpreterError::EvaluationError(
              "Table: iterator bound must be numeric".into(),
            )
          },
        )?;
        let max_num = crate::functions::math_ast::try_eval_to_f64(&max_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Table: iterator bound must be numeric".into(),
            )
          })?;
        let step_num = crate::functions::math_ast::try_eval_to_f64(&step_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Table: step must be numeric".into(),
            )
          })?;

        if step_num.abs() <= f64::EPSILON {
          return Err(InterpreterError::EvaluationError(
            "Table: step cannot be zero".into(),
          ));
        }

        let mut results = Vec::new();
        let mut current_expr = min_expr.clone();
        let mut safety_counter: usize = 0;
        if step_num > 0.0 {
          loop {
            let current_num =
              crate::functions::math_ast::try_eval_to_f64(&current_expr)
                .ok_or_else(|| {
                  InterpreterError::EvaluationError(
                    "Table: iterator value became non-numeric".into(),
                  )
                })?;
            if current_num > max_num + f64::EPSILON {
              break;
            }
            let substituted = crate::syntax::substitute_variable(
              body,
              &var_name,
              &current_expr,
            );
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            results.push(val);
            current_expr =
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![current_expr, step_expr.clone()],
              })?;
            safety_counter += 1;
            if safety_counter > 1_000_000 {
              return Err(InterpreterError::EvaluationError(
                "Table: iterator exceeded maximum iterations".into(),
              ));
            }
          }
        } else {
          loop {
            let current_num =
              crate::functions::math_ast::try_eval_to_f64(&current_expr)
                .ok_or_else(|| {
                  InterpreterError::EvaluationError(
                    "Table: iterator value became non-numeric".into(),
                  )
                })?;
            if current_num < max_num - f64::EPSILON {
              break;
            }
            let substituted = crate::syntax::substitute_variable(
              body,
              &var_name,
              &current_expr,
            );
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            results.push(val);
            current_expr =
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![current_expr, step_expr.clone()],
              })?;
            safety_counter += 1;
            if safety_counter > 1_000_000 {
              return Err(InterpreterError::EvaluationError(
                "Table: iterator exceeded maximum iterations".into(),
              ));
            }
          }
        }
        return Ok(Expr::List(results));
      }

      Err(InterpreterError::EvaluationError(
        "Table: invalid iterator specification".into(),
      ))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Table: invalid iterator specification".into(),
    )),
  }
}

/// Helper to extract i128 from Expr
fn expr_to_i128(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i128()
    }
    Expr::Real(f) if f.fract() == 0.0 => Some(*f as i128),
    _ => None,
  }
}

/// AST-based MapThread: apply function to corresponding elements.
/// MapThread[f, {{a, b}, {c, d}}] -> {f[a, c], f[b, d]}
pub fn map_thread_ast(
  func: &Expr,
  lists: &Expr,
  level: Option<usize>,
) -> Result<Expr, InterpreterError> {
  let outer_items = match lists {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MapThread".to_string(),
        args: vec![func.clone(), lists.clone()],
      });
    }
  };

  if outer_items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // Get each sublist
  let mut sublists: Vec<Vec<Expr>> = Vec::new();
  for item in outer_items {
    match item {
      Expr::List(items) => sublists.push(items.clone()),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "MapThread: second argument must be a list of lists".into(),
        ));
      }
    }
  }

  // Check all sublists have the same length
  let len = sublists[0].len();
  for sublist in &sublists {
    if sublist.len() != len {
      return Err(InterpreterError::EvaluationError(
        "MapThread: all lists must have the same length".into(),
      ));
    }
  }

  let depth = level.unwrap_or(1);

  if depth <= 1 {
    // Apply function to corresponding elements
    let mut results = Vec::new();
    for i in 0..len {
      let args: Vec<Expr> = sublists.iter().map(|sl| sl[i].clone()).collect();
      let result = apply_func_to_n_args(func, &args)?;
      results.push(result);
    }
    Ok(Expr::List(results))
  } else {
    // Recurse: thread at this level, then recurse into sublists
    let mut results = Vec::new();
    for i in 0..len {
      let inner_lists: Vec<Expr> =
        sublists.iter().map(|sl| sl[i].clone()).collect();
      let inner_arg = Expr::List(inner_lists);
      let result = map_thread_ast(func, &inner_arg, Some(depth - 1))?;
      results.push(result);
    }
    Ok(Expr::List(results))
  }
}

/// Apply a function to n arguments.
fn apply_func_to_n_args(
  func: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, args)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, args);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body } => {
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(args.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args: fa } => {
      let mut new_args = fa.clone();
      new_args.extend(args.iter().cloned());
      crate::evaluator::evaluate_function_call_ast(name, &new_args)
    }
    _ => {
      let func_str = crate::syntax::expr_to_string(func);
      crate::evaluator::evaluate_function_call_ast(&func_str, args)
    }
  }
}

/// AST-based Partition: break list into sublists of length n.
/// Partition[{a, b, c, d, e}, 2] -> {{a, b}, {c, d}}
pub fn partition_ast(
  list: &Expr,
  n: i128,
  d: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Partition".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  if n <= 0 {
    return Err(InterpreterError::EvaluationError(
      "Partition: size must be positive".into(),
    ));
  }

  let n_usize = n as usize;
  let d_usize = d.unwrap_or(n) as usize;
  if d_usize == 0 {
    return Err(InterpreterError::EvaluationError(
      "Partition: offset must be positive".into(),
    ));
  }

  let mut results = Vec::new();
  let mut i = 0;
  while i + n_usize <= items.len() {
    results.push(Expr::List(items[i..i + n_usize].to_vec()));
    i += d_usize;
  }

  Ok(Expr::List(results))
}

/// AST-based First: return first element of list.
/// First[list] or First[list, default] - returns default if list is empty.
pub fn first_ast(
  list: &Expr,
  default: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      if items.is_empty() {
        if let Some(d) = default {
          Ok(d.clone())
        } else {
          Err(InterpreterError::EvaluationError(
            "First: list is empty".into(),
          ))
        }
      } else {
        Ok(items[0].clone())
      }
    }
    Expr::FunctionCall { args, .. } => {
      if args.is_empty() {
        if let Some(d) = default {
          Ok(d.clone())
        } else {
          Err(InterpreterError::EvaluationError(
            "First: expression has no elements".into(),
          ))
        }
      } else {
        Ok(args[0].clone())
      }
    }
    _ => {
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
    Expr::List(items) => {
      if items.is_empty() {
        if let Some(d) = default {
          Ok(d.clone())
        } else {
          Err(InterpreterError::EvaluationError(
            "Last: list is empty".into(),
          ))
        }
      } else {
        Ok(items[items.len() - 1].clone())
      }
    }
    Expr::FunctionCall { args, .. } => {
      if args.is_empty() {
        if let Some(d) = default {
          Ok(d.clone())
        } else {
          Err(InterpreterError::EvaluationError(
            "Last: expression has no elements".into(),
          ))
        }
      } else {
        Ok(args[args.len() - 1].clone())
      }
    }
    _ => {
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
        Err(InterpreterError::EvaluationError(
          "Cannot take Rest of expression {} with length zero.".into(),
        ))
      } else {
        Ok(Expr::List(items[1..].to_vec()))
      }
    }
    Expr::FunctionCall { name, args } => {
      if args.is_empty() {
        Err(InterpreterError::EvaluationError(format!(
          "Cannot take Rest of expression {}[] with length zero.",
          name
        )))
      } else {
        Ok(Expr::FunctionCall {
          name: name.clone(),
          args: args[1..].to_vec(),
        })
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "Nonatomic expression expected at position 1 in Rest[{}].",
      crate::syntax::expr_to_string(list)
    ))),
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
    _ => Ok(Expr::FunctionCall {
      name: "Most".to_string(),
      args: vec![list.clone()],
    }),
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

  match result {
    Expr::List(items) => {
      let mut new_items = Vec::new();
      for item in &items {
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

  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
  };

  // Handle Take[list, {start, end}] and Take[list, {start, end, step}]
  if let Expr::List(spec) = n {
    if spec.len() == 1 {
      if let Some(idx) = expr_to_i128(&spec[0]) {
        let len = items.len() as i128;
        let real_idx = if idx < 0 { len + idx + 1 } else { idx };
        if real_idx >= 1 && real_idx <= len {
          return Ok(Expr::List(vec![items[(real_idx - 1) as usize].clone()]));
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
          return Ok(Expr::List(result));
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Take".to_string(),
      args: vec![list.clone(), n.clone()],
    });
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
      eprintln!();
      eprintln!(
        "Take::take: Cannot take positions 1 through {} in {}.",
        count, list_str
      );
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
    Ok(Expr::List(items[..count as usize].to_vec()))
  } else {
    if -count > len {
      // Print warning to stderr and return unevaluated
      let list_str = crate::syntax::expr_to_string(list);
      eprintln!();
      eprintln!(
        "Take::take: Cannot take positions {} through -1 in {}.",
        count, list_str
      );
      return Ok(Expr::FunctionCall {
        name: "Take".to_string(),
        args: vec![list.clone(), n.clone()],
      });
    }
    Ok(Expr::List(
      items[items.len() - (-count) as usize..].to_vec(),
    ))
  }
}

/// AST-based Drop: drop first n elements.
pub fn drop_ast(list: &Expr, n: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Drop".to_string(),
        args: vec![list.clone(), n.clone()],
      });
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
      return Ok(Expr::List(result));
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
      return Ok(Expr::List(result));
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
    let drop = count.min(len) as usize;
    Ok(Expr::List(items[drop..].to_vec()))
  } else {
    let keep = (len + count).max(0) as usize;
    Ok(Expr::List(items[..keep].to_vec()))
  }
}

/// AST-based Flatten: flatten nested lists.
pub fn flatten_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  fn flatten_recursive(expr: &Expr, result: &mut Vec<Expr>) {
    match expr {
      Expr::List(items) => {
        for item in items {
          flatten_recursive(item, result);
        }
      }
      _ => result.push(expr.clone()),
    }
  }

  match list {
    Expr::List(_) => {
      let mut result = Vec::new();
      flatten_recursive(list, &mut result);
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Flatten".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// Flatten[list, n] - flatten a list to depth n
pub fn flatten_level_ast(
  list: &Expr,
  depth: i128,
) -> Result<Expr, InterpreterError> {
  fn flatten_to_depth(expr: &Expr, depth: i128, result: &mut Vec<Expr>) {
    if depth <= 0 {
      result.push(expr.clone());
      return;
    }
    match expr {
      Expr::List(items) => {
        for item in items {
          flatten_to_depth(item, depth - 1, result);
        }
      }
      _ => result.push(expr.clone()),
    }
  }

  match list {
    Expr::List(items) => {
      let mut result = Vec::new();
      for item in items {
        flatten_to_depth(item, depth, &mut result);
      }
      Ok(Expr::List(result))
    }
    _ => Ok(list.clone()),
  }
}

/// AST-based Reverse: reverse a list.
pub fn reverse_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut reversed = items.clone();
      reversed.reverse();
      Ok(Expr::List(reversed))
    }
    Expr::FunctionCall { name, args } => {
      let mut reversed = args.clone();
      reversed.reverse();
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: reversed,
      })
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: replacement.clone(),
      replacement: pattern.clone(),
    }),
    _ => Ok(Expr::FunctionCall {
      name: "Reverse".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// AST-based Sort: sort a list.
pub fn sort_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut sorted = items.clone();
      sorted.sort_by(canonical_cmp);
      Ok(Expr::List(sorted))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Sort".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// Extract a rational (numerator, denominator) from an Expr.
/// Returns Some((n, d)) for Integer, Rational, None for anything else.
fn expr_to_rational(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// AST-based Range: generate a range of numbers.
/// Range[n] -> {1, 2, ..., n}
/// Range[min, max] -> {min, ..., max}
/// Range[min, max, step] -> {min, min+step, ..., max}
pub fn range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Ok(Expr::FunctionCall {
      name: "Range".to_string(),
      args: args.to_vec(),
    });
  }

  // Try exact rational arithmetic first
  let all_rational = args.iter().all(|a| expr_to_rational(a).is_some());

  if all_rational {
    let (min_n, min_d, max_n, max_d, step_n, step_d) = if args.len() == 1 {
      let (mn, md) = expr_to_rational(&args[0]).unwrap();
      (1_i128, 1_i128, mn, md, 1_i128, 1_i128)
    } else if args.len() == 2 {
      let (min_n, min_d) = expr_to_rational(&args[0]).unwrap();
      let (max_n, max_d) = expr_to_rational(&args[1]).unwrap();
      (min_n, min_d, max_n, max_d, 1_i128, 1_i128)
    } else {
      let (min_n, min_d) = expr_to_rational(&args[0]).unwrap();
      let (max_n, max_d) = expr_to_rational(&args[1]).unwrap();
      let (step_n, step_d) = expr_to_rational(&args[2]).unwrap();
      (min_n, min_d, max_n, max_d, step_n, step_d)
    };

    if step_n == 0 {
      return Err(InterpreterError::EvaluationError(
        "Range: step cannot be zero".into(),
      ));
    }

    let mut results = Vec::new();
    // Current value as (numerator, denominator)
    let mut cur_n = min_n;
    let mut cur_d = min_d;
    let step_positive = (step_n > 0) == (step_d > 0);

    loop {
      // Compare cur vs max: cur_n/cur_d vs max_n/max_d
      // cur_n * max_d vs max_n * cur_d (careful with sign of denominators)
      let lhs = cur_n * max_d;
      let rhs = max_n * cur_d;
      let denom_sign = (cur_d > 0) == (max_d > 0);

      if step_positive {
        // For positive step: stop when cur > max
        if denom_sign && lhs > rhs {
          break;
        }
        if !denom_sign && lhs < rhs {
          break;
        }
      } else {
        // For negative step: stop when cur < max
        if denom_sign && lhs < rhs {
          break;
        }
        if !denom_sign && lhs > rhs {
          break;
        }
      }

      results.push(crate::functions::math_ast::make_rational_pub(cur_n, cur_d));

      // cur += step: cur_n/cur_d + step_n/step_d = (cur_n*step_d + step_n*cur_d) / (cur_d*step_d)
      cur_n = cur_n * step_d + step_n * cur_d;
      cur_d *= step_d;

      // Simplify to avoid overflow
      let g = gcd_i128(cur_n.abs(), cur_d.abs());
      if g > 1 {
        cur_n /= g;
        cur_d /= g;
      }

      if results.len() > 1_000_000 {
        return Err(InterpreterError::EvaluationError(
          "Range: result too large".into(),
        ));
      }
    }

    return Ok(Expr::List(results));
  }

  // Fallback to f64 for Real arguments
  let (min, max, step) = if args.len() == 1 {
    let max_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (1.0, max_val, 1.0)
  } else if args.len() == 2 {
    let min_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let max_val = expr_to_f64(&args[1]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (min_val, max_val, 1.0)
  } else {
    let min_val = expr_to_f64(&args[0]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let max_val = expr_to_f64(&args[1]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    let step_val = expr_to_f64(&args[2]).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Range: argument must be numeric".into(),
      )
    })?;
    (min_val, max_val, step_val)
  };

  if step == 0.0 {
    return Err(InterpreterError::EvaluationError(
      "Range: step cannot be zero".into(),
    ));
  }

  // Check if any input is Real - if so, all outputs should be Real
  let any_real = args.iter().any(|a| matches!(a, Expr::Real(_)));

  let mut results = Vec::new();
  let mut val = min;
  if step > 0.0 {
    while val <= max + f64::EPSILON {
      results.push(if any_real {
        Expr::Real(val)
      } else {
        f64_to_expr(val)
      });
      val += step;
    }
  } else {
    while val >= max - f64::EPSILON {
      results.push(if any_real {
        Expr::Real(val)
      } else {
        f64_to_expr(val)
      });
      val += step;
    }
  }

  Ok(Expr::List(results))
}

/// Helper to extract f64 from Expr
fn expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_f64()
    }
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

/// Helper to convert f64 to appropriate Expr
fn f64_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
}

/// AST-based Accumulate: cumulative sums.
pub fn accumulate_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Accumulate".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  // Check if any element is Real - result should preserve Real type
  let has_real = items.iter().any(|item| matches!(item, Expr::Real(_)));

  let mut sum = 0.0;
  let mut results = Vec::new();
  for item in items {
    if let Some(n) = expr_to_f64(item) {
      sum += n;
      if has_real {
        results.push(Expr::Real(sum));
      } else {
        results.push(f64_to_expr(sum));
      }
    } else {
      return Ok(Expr::FunctionCall {
        name: "Accumulate".to_string(),
        args: vec![list.clone()],
      });
    }
  }

  Ok(Expr::List(results))
}

/// AST-based Differences: successive differences.
pub fn differences_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Differences".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if items.len() <= 1 {
    return Ok(Expr::List(vec![]));
  }

  let mut results = Vec::new();
  for i in 1..items.len() {
    if let (Some(a), Some(b)) =
      (expr_to_f64(&items[i - 1]), expr_to_f64(&items[i]))
    {
      results.push(f64_to_expr(b - a));
    } else {
      return Ok(Expr::FunctionCall {
        name: "Differences".to_string(),
        args: vec![list.clone()],
      });
    }
  }

  Ok(Expr::List(results))
}

/// AST-based Scan: apply function to each element for side effects.
/// Returns Null but evaluates function on each element.
pub fn scan_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Scan".to_string(),
        args: vec![func.clone(), list.clone()],
      });
    }
  };

  for item in items {
    apply_func_ast(func, item)?;
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// AST-based FoldList: fold showing intermediate values.
/// FoldList[f, x, {a, b, c}] -> {x, f[x, a], f[f[x, a], b], ...}
pub fn fold_list_ast(
  func: &Expr,
  init: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FoldList".to_string(),
        args: vec![func.clone(), init.clone(), list.clone()],
      });
    }
  };

  let mut results = vec![init.clone()];
  let mut acc = init.clone();
  for item in items {
    acc = apply_func_to_two_args(func, &acc, item)?;
    results.push(acc.clone());
  }

  Ok(Expr::List(results))
}

/// AST-based FixedPointList: list of values until fixed point.
pub fn fixed_point_list_ast(
  func: &Expr,
  init: &Expr,
  max_iterations: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut results = vec![init.clone()];
  let mut current = init.clone();

  for _ in 0..max {
    let next = apply_func_ast(func, &current)?;
    let current_str = crate::syntax::expr_to_string(&current);
    let next_str = crate::syntax::expr_to_string(&next);
    results.push(next.clone());
    if current_str == next_str {
      break;
    }
    current = next;
  }

  Ok(Expr::List(results))
}

/// AST-based Transpose: transpose a matrix (list of lists).
pub fn transpose_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let rows = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Transpose".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if rows.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // If it's a 1D list (no sub-lists), return it unchanged
  if !rows.iter().any(|r| matches!(r, Expr::List(_))) {
    return Ok(list.clone());
  }

  // Get dimensions
  let num_rows = rows.len();
  let num_cols = match &rows[0] {
    Expr::List(items) => items.len(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Transpose: argument must be a matrix".into(),
      ));
    }
  };

  // Verify all rows have the same length
  for row in rows {
    if let Expr::List(items) = row {
      if items.len() != num_cols {
        return Err(InterpreterError::EvaluationError(
          "Transpose: all rows must have the same length".into(),
        ));
      }
    } else {
      return Err(InterpreterError::EvaluationError(
        "Transpose: argument must be a matrix".into(),
      ));
    }
  }

  // Build transposed matrix
  let mut result = Vec::new();
  for j in 0..num_cols {
    let mut new_row = Vec::new();
    for i in 0..num_rows {
      if let Expr::List(items) = &rows[i] {
        new_row.push(items[j].clone());
      }
    }
    result.push(Expr::List(new_row));
  }

  Ok(Expr::List(result))
}

/// AST-based Riffle: interleave elements with separator.
/// Riffle[{a, b, c}, x] -> {a, x, b, x, c}
pub fn riffle_ast(list: &Expr, sep: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Riffle".to_string(),
        args: vec![list.clone(), sep.clone()],
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  // If sep is a list, interleave element-wise: Riffle[{a,b,c}, {x,y,z}] -> {a,x,b,y,c,z}
  if let Expr::List(sep_items) = sep {
    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
      result.push(item.clone());
      if i < sep_items.len() {
        result.push(sep_items[i].clone());
      }
    }
    return Ok(Expr::List(result));
  }

  let mut result = Vec::new();
  for (i, item) in items.iter().enumerate() {
    result.push(item.clone());
    if i < items.len() - 1 {
      result.push(sep.clone());
    }
  }

  Ok(Expr::List(result))
}

/// AST-based RotateLeft: rotate list left by n positions.
pub fn rotate_left_ast(list: &Expr, n: i128) -> Result<Expr, InterpreterError> {
  let (items, head_name): (&[Expr], Option<&str>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.as_str())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RotateLeft".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  if items.is_empty() {
    return match head_name {
      Some(name) => Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: vec![],
      }),
      None => Ok(Expr::List(vec![])),
    };
  }

  let len = items.len() as i128;
  let shift = ((n % len) + len) % len;
  let shift_usize = shift as usize;

  let mut result = items[shift_usize..].to_vec();
  result.extend_from_slice(&items[..shift_usize]);

  match head_name {
    Some(name) => Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: result,
    }),
    None => Ok(Expr::List(result)),
  }
}

/// AST-based RotateRight: rotate list right by n positions.
pub fn rotate_right_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  rotate_left_ast(list, -n)
}

/// AST-based PadLeft: pad list on the left to length n.
/// If n < len, truncates from the left.
pub fn pad_left_ast(
  list: &Expr,
  n: i128,
  pad: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PadLeft".to_string(),
        args: vec![list.clone(), Expr::Integer(n), pad.clone()],
      });
    }
  };

  let len = items.len() as i128;
  if n <= 0 {
    return Ok(Expr::List(vec![]));
  }

  if n < len {
    // Truncate from the left
    let skip = (len - n) as usize;
    return Ok(Expr::List(items[skip..].to_vec()));
  }

  if n == len {
    return Ok(list.clone());
  }

  let needed = (n - len) as usize;
  let mut result = vec![pad.clone(); needed];
  result.extend(items.iter().cloned());

  Ok(Expr::List(result))
}

/// AST-based PadRight: pad list on the right to length n.
/// If n < len, truncates from the right.
pub fn pad_right_ast(
  list: &Expr,
  n: i128,
  pad: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PadRight".to_string(),
        args: vec![list.clone(), Expr::Integer(n), pad.clone()],
      });
    }
  };

  let len = items.len() as i128;
  if n <= 0 {
    return Ok(Expr::List(vec![]));
  }

  if n < len {
    // Truncate from the right
    return Ok(Expr::List(items[..n as usize].to_vec()));
  }

  if n == len {
    return Ok(list.clone());
  }

  let needed = (n - len) as usize;
  let mut result = items.clone();
  result.extend(vec![pad.clone(); needed]);

  Ok(Expr::List(result))
}

/// AST-based Join: join multiple lists.
pub fn join_ast(lists: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut result = Vec::new();
  for list in lists {
    match list {
      Expr::List(items) => result.extend(items.iter().cloned()),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Join".to_string(),
          args: lists.to_vec(),
        });
      }
    }
  }
  Ok(Expr::List(result))
}

/// AST-based Append: append element to list.
pub fn append_ast(list: &Expr, elem: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut result = items.clone();
      result.push(elem.clone());
      Ok(Expr::List(result))
    }
    Expr::FunctionCall { name, args } => {
      let mut new_args = args.clone();
      new_args.push(elem.clone());
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Append".to_string(),
      args: vec![list.clone(), elem.clone()],
    }),
  }
}

/// AST-based Prepend: prepend element to list.
pub fn prepend_ast(list: &Expr, elem: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut result = vec![elem.clone()];
      result.extend(items.iter().cloned());
      Ok(Expr::List(result))
    }
    Expr::FunctionCall { name, args } => {
      let mut new_args = vec![elem.clone()];
      new_args.extend(args.iter().cloned());
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Prepend".to_string(),
      args: vec![list.clone(), elem.clone()],
    }),
  }
}

/// AST-based DeleteDuplicatesBy: remove duplicates by key function.
pub fn delete_duplicates_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteDuplicatesBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  use std::collections::HashSet;
  let mut seen: HashSet<String> = HashSet::new();
  let mut result = Vec::new();

  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    if seen.insert(key_str) {
      result.push(item.clone());
    }
  }

  Ok(Expr::List(result))
}

/// AST-based Median: calculate median of a list.
pub fn median_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Median".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if items.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Median: list is empty".into(),
    ));
  }

  // Check for list-of-lists (matrix) input → columnwise median
  if items.iter().all(|item| matches!(item, Expr::List(_))) {
    let rows: Vec<&Vec<Expr>> = items
      .iter()
      .filter_map(|item| {
        if let Expr::List(row) = item {
          Some(row)
        } else {
          None
        }
      })
      .collect();
    if !rows.is_empty() {
      let ncols = rows[0].len();
      if rows.iter().all(|r| r.len() == ncols) {
        let mut result = Vec::new();
        for col in 0..ncols {
          let column: Vec<Expr> = rows.iter().map(|r| r[col].clone()).collect();
          let col_median = median_ast(&Expr::List(column))?;
          result.push(col_median);
        }
        return Ok(Expr::List(result));
      }
    }
  }

  // Check if all items are integers
  let all_integers = items.iter().all(|i| matches!(i, Expr::Integer(_)));

  if all_integers {
    // Sort integer values
    let mut int_values: Vec<i128> = items
      .iter()
      .filter_map(|i| {
        if let Expr::Integer(n) = i {
          Some(*n)
        } else {
          None
        }
      })
      .collect();
    int_values.sort();

    let len = int_values.len();
    if len % 2 == 1 {
      Ok(Expr::Integer(int_values[len / 2]))
    } else {
      // Average of two middle values
      let a = int_values[len / 2 - 1];
      let b = int_values[len / 2];
      let sum = a + b;
      if sum % 2 == 0 {
        Ok(Expr::Integer(sum / 2))
      } else {
        // Return as Rational
        fn gcd(a: i128, b: i128) -> i128 {
          if b == 0 { a } else { gcd(b, a % b) }
        }
        let g = gcd(sum.abs(), 2);
        Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sum / g), Expr::Integer(2 / g)],
        })
      }
    }
  } else {
    // Check if any Real inputs - preserve Real type
    let has_real = items.iter().any(|i| matches!(i, Expr::Real(_)));

    // Extract numeric values as f64
    let mut values: Vec<f64> = Vec::new();
    for item in items {
      if let Some(n) = expr_to_f64(item) {
        values.push(n);
      } else {
        return Ok(Expr::FunctionCall {
          name: "Median".to_string(),
          args: vec![list.clone()],
        });
      }
    }

    values
      .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = values.len();
    let result = if len % 2 == 1 {
      values[len / 2]
    } else {
      (values[len / 2 - 1] + values[len / 2]) / 2.0
    };

    // Preserve Real type if inputs had Real
    if has_real {
      Ok(Expr::Real(result))
    } else {
      Ok(f64_to_expr(result))
    }
  }
}

/// AST-based Count: count elements equal to pattern.
pub fn count_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Count".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let count = items
    .iter()
    .filter(|item| matches_pattern_ast(item, pattern))
    .count();

  Ok(Expr::Integer(count as i128))
}

/// AST-based ConstantArray: create array filled with constant.
/// ConstantArray[c, n] -> {c, c, ..., c} (n times)
/// ConstantArray[c, {n1, n2}] -> nested array
pub fn constant_array_ast(
  elem: &Expr,
  dims: &Expr,
) -> Result<Expr, InterpreterError> {
  match dims {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let n = expr_to_i128(dims).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "ConstantArray: dimension too large".into(),
        )
      })?;
      if n < 0 {
        return Err(InterpreterError::EvaluationError(
          "ConstantArray: dimension must be non-negative".into(),
        ));
      }
      Ok(Expr::List(vec![elem.clone(); n as usize]))
    }
    Expr::List(dim_list) => {
      if dim_list.is_empty() {
        return Ok(elem.clone());
      }
      let first_dim = expr_to_i128(&dim_list[0]).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "ConstantArray: dimensions must be integers".into(),
        )
      })?;
      if dim_list.len() == 1 {
        Ok(Expr::List(vec![elem.clone(); first_dim as usize]))
      } else {
        let rest_dims = Expr::List(dim_list[1..].to_vec());
        let inner = constant_array_ast(elem, &rest_dims)?;
        Ok(Expr::List(vec![inner; first_dim as usize]))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "ConstantArray".to_string(),
      args: vec![elem.clone(), dims.clone()],
    }),
  }
}

/// AST-based NestWhile: nest while condition is true.
pub fn nest_while_ast(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  max_iterations: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut current = init.clone();

  for _ in 0..max {
    let test_result = apply_func_ast(test, &current)?;
    if expr_to_bool(&test_result) != Some(true) {
      break;
    }
    current = apply_func_ast(func, &current)?;
  }

  Ok(current)
}

/// AST-based NestWhileList: like NestWhile but returns list.
pub fn nest_while_list_ast(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  max_iterations: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut results = vec![init.clone()];
  let mut current = init.clone();

  for _ in 0..max {
    let test_result = apply_func_ast(test, &current)?;
    if expr_to_bool(&test_result) != Some(true) {
      break;
    }
    current = apply_func_ast(func, &current)?;
    results.push(current.clone());
  }

  Ok(Expr::List(results))
}

/// AST-based Product: product of list elements or iterator product.
pub fn product_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Product[{a, b, c}] -> a * b * c
    let items = match &args[0] {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Product".to_string(),
          args: args.to_vec(),
        });
      }
    };

    let mut product = 1.0;
    for item in items {
      if let Some(n) = expr_to_f64(item) {
        product *= n;
      } else {
        return Ok(Expr::FunctionCall {
          name: "Product".to_string(),
          args: args.to_vec(),
        });
      }
    }
    return Ok(f64_to_expr(product));
  }

  if args.len() == 2 {
    // Product[expr, {i, min, max}] -> multiply expr for each i
    let body = &args[0];
    let iter_spec = &args[1];

    match iter_spec {
      Expr::List(items) if items.len() >= 2 => {
        let var_name = match &items[0] {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Product".to_string(),
              args: args.to_vec(),
            });
          }
        };

        // Check for list iteration form: {i, list}
        if items.len() == 2 {
          let evaluated_second =
            crate::evaluator::evaluate_expr_to_expr(&items[1])?;
          if let Expr::List(list_items) = &evaluated_second {
            // Product[expr, {i, list}] -> iterate over list elements
            let mut product = 1.0;
            for item in list_items {
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, item);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              if let Some(n) = expr_to_f64(&val) {
                product *= n;
              } else {
                return Ok(Expr::FunctionCall {
                  name: "Product".to_string(),
                  args: args.to_vec(),
                });
              }
            }
            return Ok(f64_to_expr(product));
          }
        }

        // Check if bounds are numeric
        let bounds = if items.len() == 2 {
          expr_to_i128(&items[1]).map(|max| (1i128, max))
        } else {
          match (expr_to_i128(&items[1]), expr_to_i128(&items[2])) {
            (Some(min), Some(max)) => Some((min, max)),
            _ => None,
          }
        };

        // If bounds are symbolic, try to compute symbolic product
        if bounds.is_none() {
          let min_concrete = if items.len() == 2 {
            Some(1i128) // {i, n} implies min = 1
          } else {
            expr_to_i128(&items[1])
          };
          let max_concrete = if items.len() == 2 {
            expr_to_i128(&items[1])
          } else {
            expr_to_i128(&items[2])
          };
          let max_expr = if items.len() == 2 {
            &items[1]
          } else {
            &items[2]
          };
          let min_expr = if items.len() == 2 {
            &Expr::Integer(1)
          } else {
            &items[1]
          };

          // Body is the iteration variable itself: Product[k, {k, ...}]
          if matches!(body, Expr::Identifier(name) if name == &var_name) {
            if let Some(min_val) = min_concrete {
              if max_concrete.is_none() {
                // Product[k, {k, concrete_min, symbolic_max}]
                // = max! / (min-1)!
                let n_factorial = Expr::FunctionCall {
                  name: "Factorial".to_string(),
                  args: vec![max_expr.clone()],
                };
                if min_val == 1 {
                  return Ok(n_factorial);
                }
                // Compute (min-1)! as a concrete integer
                let mut denom: i128 = 1;
                for j in 2..min_val {
                  denom *= j;
                }
                return Ok(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Divide,
                  left: Box::new(n_factorial),
                  right: Box::new(Expr::Integer(denom)),
                });
              }
            } else if max_concrete.is_none() {
              // Product[k, {k, sym_min, sym_max}]
              // = Pochhammer[min, 1 - min + max]
              return Ok(Expr::FunctionCall {
                name: "Pochhammer".to_string(),
                args: vec![
                  min_expr.clone(),
                  // 1 - min + max
                  Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Plus,
                    left: Box::new(Expr::BinaryOp {
                      op: crate::syntax::BinaryOperator::Minus,
                      left: Box::new(Expr::Integer(1)),
                      right: Box::new(min_expr.clone()),
                    }),
                    right: Box::new(max_expr.clone()),
                  },
                ],
              });
            }
          }

          // Body is c^var: Product[c^i, {i, 1, n}] = c^(n*(1+n)/2)
          if let Some(1) = min_concrete
            && max_concrete.is_none()
            && let Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: base,
              right: exp,
            } = body
          {
            if matches!(exp.as_ref(), Expr::Identifier(name) if name == &var_name)
            {
              // Product[c^i, {i, 1, n}] = c^((n*(1+n))/2)
              let n = max_expr.clone();
              let exponent = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(n.clone()),
                  right: Box::new(Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Plus,
                    left: Box::new(Expr::Integer(1)),
                    right: Box::new(n),
                  }),
                }),
                right: Box::new(Expr::Integer(2)),
              };
              return Ok(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: base.clone(),
                right: Box::new(exponent),
              });
            }

            // Product[i^k, {i, 1, n}] = n!^k
            if matches!(base.as_ref(), Expr::Identifier(name) if name == &var_name)
            {
              return Ok(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(Expr::FunctionCall {
                  name: "Factorial".to_string(),
                  args: vec![max_expr.clone()],
                }),
                right: exp.clone(),
              });
            }
          }

          // For other symbolic cases, return unevaluated
          return Ok(Expr::FunctionCall {
            name: "Product".to_string(),
            args: args.to_vec(),
          });
        }

        let (min, max) = bounds.unwrap();

        let step = if items.len() >= 4 {
          expr_to_i128(&items[3]).unwrap_or(1)
        } else {
          1
        };

        // Collect evaluated values for each iteration
        let mut values: Vec<Expr> = Vec::new();
        let mut i = min;
        while (step > 0 && i <= max) || (step < 0 && i >= max) {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          values.push(val);
          i += step;
        }

        // Try numeric product first
        let mut numeric_product = 1.0;
        let mut all_numeric = true;
        for val in &values {
          if let Some(n) = expr_to_f64(val) {
            numeric_product *= n;
          } else {
            all_numeric = false;
            break;
          }
        }

        if all_numeric {
          return Ok(f64_to_expr(numeric_product));
        }

        // For symbolic values, build a Times expression
        if values.is_empty() {
          return Ok(Expr::Integer(1));
        }
        if values.len() == 1 {
          return Ok(values.into_iter().next().unwrap());
        }
        // Fold into nested Times
        let mut result = values.remove(0);
        for val in values {
          result = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(result),
            right: Box::new(val),
          };
        }
        return crate::evaluator::evaluate_expr_to_expr(&result);
      }
      _ => {}
    }
  }

  Ok(Expr::FunctionCall {
    name: "Product".to_string(),
    args: args.to_vec(),
  })
}

/// AST-based Sum: sum of list elements or iterator sum.
pub fn sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    // Sum requires at least 2 arguments
    return Ok(Expr::FunctionCall {
      name: "Sum".to_string(),
      args: args.to_vec(),
    });
  }

  // Multi-dimensional Sum: Sum[expr, {i,...}, {j,...}, ...] => Sum[Sum[expr, {j,...}], {i,...}]
  if args.len() > 2 {
    // Evaluate innermost sum first (last iterator), then wrap outward
    let body = &args[0];
    let inner_iter = &args[args.len() - 1];
    let inner_sum = sum_ast(&[body.clone(), inner_iter.clone()])?;
    if args.len() == 3 {
      return sum_ast(&[inner_sum, args[1].clone()]);
    } else {
      let mut new_args = vec![inner_sum];
      new_args.extend_from_slice(&args[1..args.len() - 1]);
      return sum_ast(&new_args);
    }
  }

  // Sum[expr, {i, min, max}] or variants
  let body = &args[0];
  let iter_spec = &args[1];

  match iter_spec {
    Expr::List(items) if items.len() >= 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: args.to_vec(),
          });
        }
      };

      // Check for list iteration form: {i, list}
      if items.len() == 2 {
        let evaluated_second =
          crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        if let Expr::List(list_items) = &evaluated_second {
          // Sum[expr, {i, {v1, v2, ...}}] -> iterate over list elements
          let mut acc = Expr::Integer(0);
          for item in list_items {
            let substituted =
              crate::syntax::substitute_variable(body, &var_name, item);
            let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
            acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          }
          return Ok(acc);
        }
      }

      // Check for infinite sum: {i, min, Infinity}
      if items.len() == 3
        && let Expr::Identifier(s) = &items[2]
        && s == "Infinity"
      {
        let min_val = expr_to_i128(&items[1]).unwrap_or(1);
        if let Some(result) = try_infinite_sum(body, &var_name, min_val)? {
          return Ok(result);
        }
        // Could not evaluate symbolically — return unevaluated
        return Ok(Expr::FunctionCall {
          name: "Sum".to_string(),
          args: args.to_vec(),
        });
      }

      // Try real-valued iteration when bounds are numeric but not integers
      if items.len() == 3 {
        let min_int = expr_to_i128(&items[1]);
        let max_int = expr_to_i128(&items[2]);
        if min_int.is_none() || max_int.is_none() {
          // Check if bounds are numeric reals
          let min_f = crate::functions::math_ast::try_eval_to_f64(&items[1]);
          let max_f = crate::functions::math_ast::try_eval_to_f64(&items[2]);
          if let (Some(min_val), Some(max_val)) = (min_f, max_f) {
            // Iterate with step=1, substituting real values
            let mut acc = Expr::Integer(0);
            let mut i = min_val;
            while i <= max_val + 1e-10 {
              let sub_val =
                if (i - i.round()).abs() < 1e-10 && min_int.is_some() {
                  Expr::Integer(i.round() as i128)
                } else {
                  Expr::Real(i)
                };
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &sub_val);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
              i += 1.0;
            }
            return Ok(acc);
          }
        }
      }

      // Try iterating when the difference between bounds is real
      // (handles complex bounds like {k, I, I+1})
      if items.len() == 3 {
        let diff = crate::functions::math_ast::plus_ast(&[
          items[2].clone(),
          Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(items[1].clone()),
          },
        ]);
        if let Ok(diff_expr) = diff {
          let diff_eval = crate::evaluator::evaluate_expr_to_expr(&diff_expr);
          if let Ok(ref de) = diff_eval
            && let Some(range) = crate::functions::math_ast::try_eval_to_f64(de)
            && (0.0..10000.0).contains(&range)
          {
            let n_iters = range.floor() as i128 + 1;
            let min_eval = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
            let mut acc = Expr::Integer(0);
            for j in 0..n_iters {
              let iter_val = if j == 0 {
                min_eval.clone()
              } else {
                crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Plus,
                  left: Box::new(min_eval.clone()),
                  right: Box::new(Expr::Integer(j)),
                })?
              };
              let substituted =
                crate::syntax::substitute_variable(body, &var_name, &iter_val);
              let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
              acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
            }
            return Ok(acc);
          }
        }
      }

      // Try symbolic Sum when bounds are not both concrete integers
      if items.len() == 3 {
        let min_concrete = expr_to_i128(&items[1]);
        let max_concrete = expr_to_i128(&items[2]);
        if min_concrete.is_none() || max_concrete.is_none() {
          if let Some(result) = try_symbolic_sum(
            body,
            &var_name,
            &items[1],
            &items[2],
            min_concrete,
            max_concrete,
          )? {
            // Evaluate to simplify the symbolic result
            return crate::evaluator::evaluate_expr_to_expr(&result);
          }
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: args.to_vec(),
          });
        }
      }

      // Extract min, max, step
      let (min, max, step) = if items.len() == 2 {
        let max_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        (1i128, max_val, 1i128)
      } else if items.len() == 3 {
        let min_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let max_val = expr_to_i128(&items[2]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        (min_val, max_val, 1i128)
      } else {
        // items.len() == 4: {i, min, max, step}
        let min_val = expr_to_i128(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let max_val = expr_to_i128(&items[2]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: iterator bounds must be integers".into(),
          )
        })?;
        let step_val = expr_to_i128(&items[3]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Sum: step must be an integer".into(),
          )
        })?;
        if step_val == 0 {
          return Err(InterpreterError::EvaluationError(
            "Sum: step cannot be zero".into(),
          ));
        }
        (min_val, max_val, step_val)
      };

      let mut acc = Expr::Integer(0);
      let mut i = min;
      if step > 0 {
        while i <= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          i += step;
        }
      } else {
        while i >= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
          acc = crate::functions::math_ast::plus_ast(&[acc, val])?;
          i += step;
        }
      }
      return Ok(acc);
    }
    _ => {}
  }

  Ok(Expr::FunctionCall {
    name: "Sum".to_string(),
    args: args.to_vec(),
  })
}

/// Try to evaluate a known infinite series Sum[body, {var, min, Infinity}].
/// Returns Some(result) if a closed form is found, None otherwise.
/// Try to evaluate a symbolic Sum where at least one bound is not a concrete integer.
/// Returns Some(expr) if a known closed form is found, None otherwise.
fn try_symbolic_sum(
  body: &Expr,
  var_name: &str,
  min_expr: &Expr,
  max_expr: &Expr,
  min_concrete: Option<i128>,
  _max_concrete: Option<i128>,
) -> Result<Option<Expr>, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Sum[k, {k, 1, n}] = n*(1 + n)/2
  if let Some(1) = min_concrete {
    if matches!(body, Expr::Identifier(name) if name == var_name) {
      let n = max_expr.clone();
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(n.clone()),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(n),
          }),
        }),
        right: Box::new(Expr::Integer(2)),
      }));
    }

    // Sum[k^2, {k, 1, n}] = n*(1 + n)*(1 + 2*n)/6
    if let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = body
      && matches!(base.as_ref(), Expr::Identifier(name) if name == var_name)
      && matches!(exp.as_ref(), Expr::Integer(2))
    {
      let n = max_expr.clone();
      return Ok(Some(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(n.clone()),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(Expr::Integer(1)),
              right: Box::new(n.clone()),
            }),
          }),
          right: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(1)),
            right: Box::new(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(2)),
              right: Box::new(n),
            }),
          }),
        }),
        right: Box::new(Expr::Integer(6)),
      }));
    }

    // Sum[1/k^s, {k, 1, n}] = HarmonicNumber[n, s]
    if let Some(s) = match_reciprocal_power(body, var_name)
      && s >= 1
    {
      return Ok(Some(Expr::FunctionCall {
        name: "HarmonicNumber".to_string(),
        args: vec![max_expr.clone(), Expr::Integer(s as i128)],
      }));
    }

    // Sum[c^i, {i, 1, n}] = c*(c^n - 1)/(c - 1) (geometric series)
    // In Divide form: Sum[1/c^i, {i, 1, n}] = (c^n - 1)/(c^n * (c - 1))
    // or equivalently: (-1 + c^n)/c^n
    // Detect body = 1/c^var (Divide or Power with negative exponent)
    if let Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } = body
      && matches!(left.as_ref(), Expr::Integer(1))
    {
      // 1 / c^var
      if let Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: base,
        right: exp,
      } = right.as_ref()
        && matches!(exp.as_ref(), Expr::Identifier(name) if name == var_name)
      {
        // Sum[1/c^i, {i, 1, n}] = (-1 + c^n)/(c^n*(c-1))
        // For c=2: (-1 + 2^n)/2^n
        let c = base.as_ref();
        let n = max_expr.clone();
        let c_to_n = Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(c.clone()),
          right: Box::new(n),
        };
        return Ok(Some(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(c_to_n.clone()),
          }),
          right: Box::new(c_to_n),
        }));
      }
    }
  }

  // Sum[k, {k, a, n}] where a is symbolic
  if min_concrete.is_none()
    && matches!(body, Expr::Identifier(name) if name == var_name)
  {
    // Sum[k, {k, a, n}] = (a+n)*(n-a+1)/2
    let a = min_expr.clone();
    let n = max_expr.clone();
    return Ok(Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(a.clone()),
          right: Box::new(n.clone()),
        }),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(n),
            right: Box::new(a),
          }),
          right: Box::new(Expr::Integer(1)),
        }),
      }),
      right: Box::new(Expr::Integer(2)),
    }));
  }

  Ok(None)
}

/// Check if a body expression matches the Leibniz series: (-1)^k / (2k+1).
/// We verify by evaluating at k=0,1,2,3,4 and checking against expected values.
fn is_leibniz_body(body: &Expr, var_name: &str) -> bool {
  // Expected values: f(0)=1, f(1)=-1/3, f(2)=1/5, f(3)=-1/7, f(4)=1/9
  let expected: [(i128, f64); 5] = [
    (0, 1.0),
    (1, -1.0 / 3.0),
    (2, 1.0 / 5.0),
    (3, -1.0 / 7.0),
    (4, 1.0 / 9.0),
  ];
  for (k, exp_val) in &expected {
    let substituted =
      crate::syntax::substitute_variable(body, var_name, &Expr::Integer(*k));
    if let Ok(result) = crate::evaluator::evaluate_expr_to_expr(&substituted) {
      if let Some(val) = crate::functions::math_ast::try_eval_to_f64(&result) {
        if (val - exp_val).abs() > 1e-12 {
          return false;
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  true
}

fn try_infinite_sum(
  body: &Expr,
  var_name: &str,
  min: i128,
) -> Result<Option<Expr>, InterpreterError> {
  // Try Leibniz formula: Sum[(-1)^k / (2k+1), {k, 0, Infinity}] = Pi/4
  if min == 0 {
    if is_leibniz_body(body, var_name) {
      return Ok(Some(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(4)),
      }));
    }
    return Ok(None);
  }

  if min != 1 {
    return Ok(None);
  }

  // Try to detect the pattern 1/var^s (i.e., var^(-s))
  // The body for 1/n^2 is: Times[1, Power[Power[n, 2], -1]]
  // which evaluates/simplifies to Power[n, -2] conceptually,
  // but in practice we need to match the AST structure.
  if let Some(s) = match_reciprocal_power(body, var_name) {
    if s >= 2 && s % 2 == 0 {
      // Zeta(s) for even s: (-1)^(s/2+1) * B_s * (2*Pi)^s / (2 * s!)
      return Ok(Some(zeta_even(s)?));
    }
    // Odd s >= 3: no known closed form in terms of Pi (returns Zeta[s])
    if s >= 3 && s % 2 == 1 {
      return Ok(Some(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: vec![Expr::Integer(s as i128)],
      }));
    }
  }

  // Sum[1/c^i, {i, 1, Infinity}] = 1/(c-1) for integer c > 1
  // Detect body = 1/c^var (Divide form)
  use crate::syntax::BinaryOperator;
  if let Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left,
    right,
  } = body
    && matches!(left.as_ref(), Expr::Integer(1))
    && let Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: base,
      right: exp,
    } = right.as_ref()
    && matches!(exp.as_ref(), Expr::Identifier(name) if name == var_name)
    && let Some(c) = expr_to_i128(base)
    && c > 1
  {
    // Sum = 1/(c-1)
    return Ok(Some(crate::functions::math_ast::make_rational(1, c - 1)));
  }

  Ok(None)
}

/// Match the pattern `1/var^s` in the body expression.
/// Returns Some(s) if the body is equivalent to var^(-s) with s a positive integer.
fn match_reciprocal_power(body: &Expr, var_name: &str) -> Option<i64> {
  use crate::syntax::BinaryOperator;

  match body {
    // Direct Power[var, -s]
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Identifier(name) = left.as_ref()
        && name == var_name
        && let Some(exp) = get_integer(right)
        && exp < 0
      {
        return Some(-exp as i64);
      }
      // Power[Power[var, s], -1]
      match_power_inverse(body, var_name)
    }
    // Divide[1, Power[var, s]] or Divide[1, var]  (how 1/var^s is stored internally)
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      if is_one(left) {
        // 1 / var^s
        if let Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: base,
          right: exp,
        } = right.as_ref()
          && let Expr::Identifier(name) = base.as_ref()
          && name == var_name
          && let Some(s) = get_integer(exp)
          && s > 0
        {
          return Some(s as i64);
        }
        // 1 / var => s = 1
        if let Expr::Identifier(name) = right.as_ref()
          && name == var_name
        {
          return Some(1);
        }
      }
      None
    }
    // Times[1, Power[Power[var, s], -1]]  (FullForm representation)
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_one(left) {
        return match_power_inverse(right, var_name);
      }
      if is_one(right) {
        return match_power_inverse(left, var_name);
      }
      None
    }
    _ => match_power_inverse(body, var_name),
  }
}

/// Match Power[Power[var, s], -1] or Power[var, -s]
fn match_power_inverse(expr: &Expr, var_name: &str) -> Option<i64> {
  use crate::syntax::BinaryOperator;

  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => {
      // Power[something, -1] where something = Power[var, s]
      if let Some(-1) = get_integer(right) {
        if let Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: inner_left,
          right: inner_right,
        } = left.as_ref()
          && let Expr::Identifier(name) = inner_left.as_ref()
          && name == var_name
          && let Some(s) = get_integer(inner_right)
          && s > 0
        {
          return Some(s as i64);
        }
        // Power[var, -1] => s = 1
        if let Expr::Identifier(name) = left.as_ref()
          && name == var_name
        {
          return Some(1);
        }
      }
      // Power[var, -s] directly
      if let Expr::Identifier(name) = left.as_ref()
        && name == var_name
        && let Some(exp) = get_integer(right)
        && exp < 0
      {
        return Some(-exp as i64);
      }
      None
    }
    _ => None,
  }
}

/// Get an integer value from an Expr
fn get_integer(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i128()
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Integer(n) => Some(-n),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        (-n).to_i128()
      }
      _ => None,
    },
    _ => None,
  }
}

fn is_one(expr: &Expr) -> bool {
  matches!(expr, Expr::Integer(1))
    || matches!(expr, Expr::BigInteger(n) if *n == num_bigint::BigInt::from(1))
}

/// Compute ζ(2k) = |B_{2k}| * (2π)^{2k} / (2 * (2k)!) as a symbolic expression.
/// Returns Pi^(2k) * rational_coefficient.
fn zeta_even(s: i64) -> Result<Expr, InterpreterError> {
  use crate::syntax::BinaryOperator;

  // Get B_s using bernoulli_b_ast
  let b_s =
    crate::functions::math_ast::bernoulli_b_ast(&[Expr::Integer(s as i128)])?;

  // Extract the rational value of B_s as (num, den)
  let (b_num, b_den) = match &b_s {
    Expr::Integer(n) => (*n, 1i128),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      match n.to_i128() {
        Some(v) => (v, 1i128),
        None => {
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: vec![],
          });
        }
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
        (Some(n), Some(d)) => (n, d),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Sum".to_string(),
            args: vec![],
          });
        }
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Sum".to_string(),
        args: vec![],
      });
    }
  };

  // ζ(s) = (-1)^(s/2+1) * B_s * (2π)^s / (2 * s!)
  // Since B_s for even s alternates sign: B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, ...
  // (-1)^(s/2+1) * B_s = |B_s| always positive
  // So ζ(s) = |B_s| * (2π)^s / (2 * s!)

  // Compute (2^s) * |B_s_num| / (2 * s! * |B_s_den|)
  // = 2^(s-1) * |B_s_num| / (s! * |B_s_den|)
  let abs_b_num = b_num.abs();

  // Compute 2^(s-1) and s!
  let two_pow = 2i128.checked_pow((s - 1) as u32).unwrap_or(i128::MAX);
  let mut factorial: i128 = 1;
  for i in 2..=s as i128 {
    factorial = factorial.checked_mul(i).unwrap_or(i128::MAX);
  }

  // The coefficient of Pi^s is: 2^(s-1) * |B_s_num| / (s! * B_s_den)
  let coeff_num = two_pow * abs_b_num;
  let coeff_den = factorial * b_den.abs();

  // Simplify the fraction
  let g = gcd_i128(coeff_num.abs(), coeff_den.abs());
  let final_num = coeff_num / g;
  let final_den = coeff_den / g;

  // Build the expression: (final_num / final_den) * Pi^s
  let pi_power = if s == 1 {
    Expr::Identifier("Pi".to_string())
  } else {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Identifier("Pi".to_string())),
      right: Box::new(Expr::Integer(s as i128)),
    }
  };

  if final_num == 1 && final_den == 1 {
    Ok(pi_power)
  } else if final_den == 1 {
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(final_num)),
      right: Box::new(pi_power),
    })
  } else if final_num == 1 {
    // 1/d * Pi^s => Pi^s / d
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(pi_power),
      right: Box::new(Expr::Integer(final_den)),
    })
  } else {
    // n/d * Pi^s => (n * Pi^s) / d
    Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(final_num)),
        right: Box::new(pi_power),
      }),
      right: Box::new(Expr::Integer(final_den)),
    })
  }
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// AST-based Thread: thread a function over lists.
/// Thread[f[{a, b}, {c, d}]] -> {f[a, c], f[b, d]}
pub fn thread_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => {
      // Find which args are lists
      let mut list_indices: Vec<usize> = Vec::new();
      let mut list_len: Option<usize> = None;

      for (i, arg) in args.iter().enumerate() {
        if let Expr::List(items) = arg {
          if let Some(len) = list_len {
            if items.len() != len {
              return Err(InterpreterError::EvaluationError(
                "Thread: all lists must have the same length".into(),
              ));
            }
          } else {
            list_len = Some(items.len());
          }
          list_indices.push(i);
        }
      }

      if list_indices.is_empty() {
        return Ok(expr.clone());
      }

      let len = list_len.unwrap();
      let mut results = Vec::new();

      for j in 0..len {
        let new_args: Vec<Expr> = args
          .iter()
          .enumerate()
          .map(|(i, arg)| {
            if list_indices.contains(&i) {
              if let Expr::List(items) = arg {
                items[j].clone()
              } else {
                arg.clone()
              }
            } else {
              arg.clone()
            }
          })
          .collect();
        let result =
          crate::evaluator::evaluate_function_call_ast(name, &new_args)?;
        results.push(result);
      }

      Ok(Expr::List(results))
    }
    _ => Ok(expr.clone()),
  }
}

/// AST-based Through: apply multiple functions.
/// Through[{f, g}[x]] -> {f[x], g[x]}
/// Through[f[g][x]] -> f[g[x]]
/// Through[Plus[f, g][x]] -> f[x] + g[x]
pub fn through_ast(
  expr: &Expr,
  head_filter: Option<&str>,
) -> Result<Expr, InterpreterError> {
  // Through operates on CurriedCall: h[f1, f2, ...][args...]
  // It threads the args through each fi, wrapping the result in h.
  match expr {
    Expr::CurriedCall { func, args } => {
      // func is the head expression, e.g. f[g], {f, g}, Plus[f, g]
      // args are the outer arguments to thread through
      let (head_name, functions) = match func.as_ref() {
        Expr::FunctionCall { name, args: fns } => {
          (name.as_str(), fns.as_slice())
        }
        Expr::List(items) => ("List", items.as_slice()),
        _ => {
          // Not a compound head - return unevaluated
          return Ok(Expr::FunctionCall {
            name: "Through".to_string(),
            args: vec![expr.clone()],
          });
        }
      };

      // Check head filter if provided
      if let Some(filter) = head_filter
        && head_name != filter
      {
        // Head doesn't match filter - return the inner expression unchanged
        return Ok(expr.clone());
      }

      // Thread: apply each function to the outer args
      let threaded: Vec<Expr> = functions
        .iter()
        .map(|f| Expr::FunctionCall {
          name: crate::syntax::expr_to_string(f),
          args: args.clone(),
        })
        .collect();

      // Wrap in the head
      if head_name == "List" {
        Ok(Expr::List(threaded))
      } else {
        Ok(Expr::FunctionCall {
          name: head_name.to_string(),
          args: threaded,
        })
      }
    }
    _ => {
      if head_filter.is_some() {
        // With head filter and non-CurriedCall: return expression as-is
        Ok(expr.clone())
      } else {
        // Not a curried call - return unevaluated
        Ok(Expr::FunctionCall {
          name: "Through".to_string(),
          args: vec![expr.clone()],
        })
      }
    }
  }
}

/// AST-based TakeLargest: take n largest elements.
pub fn take_largest_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  // Extract numeric values with indices
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for item in items {
    if let Some(v) = expr_to_f64(item) {
      keyed.push((v, item.clone()));
    } else {
      return Ok(Expr::FunctionCall {
        name: "TakeLargest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  }

  keyed
    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

  let take = (n as usize).min(keyed.len());
  let result: Vec<Expr> =
    keyed.into_iter().take(take).map(|(_, e)| e).collect();

  Ok(Expr::List(result))
}

/// AST-based TakeSmallest: take n smallest elements.
pub fn take_smallest_ast(
  list: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  };

  // Extract numeric values with indices
  let mut keyed: Vec<(f64, Expr)> = Vec::new();
  for item in items {
    if let Some(v) = expr_to_f64(item) {
      keyed.push((v, item.clone()));
    } else {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallest".to_string(),
        args: vec![list.clone(), Expr::Integer(n)],
      });
    }
  }

  keyed
    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

  let take = (n as usize).min(keyed.len());
  let result: Vec<Expr> =
    keyed.into_iter().take(take).map(|(_, e)| e).collect();

  Ok(Expr::List(result))
}

/// AST-based ArrayDepth: compute depth of nested lists.
pub fn array_depth_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  fn compute_depth(expr: &Expr) -> i128 {
    match expr {
      Expr::List(items) => {
        if items.is_empty() {
          1
        } else {
          1 + items.iter().map(compute_depth).min().unwrap_or(0)
        }
      }
      _ => 0,
    }
  }

  Ok(Expr::Integer(compute_depth(list)))
}

/// ArrayQ[expr] - True if expr is a full array (rectangular at all levels).
pub fn array_q_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  fn is_full_array(expr: &Expr) -> bool {
    match expr {
      Expr::List(items) => {
        if items.is_empty() {
          return true;
        }
        // All items must have the same structure
        let depths: Vec<Vec<usize>> =
          items.iter().map(get_dimensions).collect();
        // All items must have the same dimensions
        depths.iter().all(|d| d == &depths[0])
      }
      _ => false,
    }
  }
  Ok(if is_full_array(expr) {
    Expr::Identifier("True".to_string())
  } else {
    Expr::Identifier("False".to_string())
  })
}

/// Get the dimensions of a rectangular array
fn get_dimensions(expr: &Expr) -> Vec<usize> {
  match expr {
    Expr::List(items) => {
      let mut dims = vec![items.len()];
      if !items.is_empty() {
        let sub_dims: Vec<Vec<usize>> =
          items.iter().map(get_dimensions).collect();
        // Check all sublists have the same dimensions
        if sub_dims.iter().all(|d| d == &sub_dims[0]) && !sub_dims[0].is_empty()
        {
          dims.extend_from_slice(&sub_dims[0]);
        }
      }
      dims
    }
    _ => vec![],
  }
}

/// VectorQ[expr] - True if expr is a list of non-list elements.
pub fn vector_q_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(items) => {
      Ok(if items.iter().all(|i| !matches!(i, Expr::List(_))) {
        Expr::Identifier("True".to_string())
      } else {
        Expr::Identifier("False".to_string())
      })
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

/// MatrixQ[expr] - True if expr is a list of equal-length lists (2D rectangular array).
pub fn matrix_q_ast(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::List(rows) => {
      if rows.is_empty() {
        return Ok(Expr::Identifier("True".to_string()));
      }
      // Each row must be a list
      let mut ncols = None;
      for row in rows {
        match row {
          Expr::List(cols) => {
            if let Some(expected) = ncols {
              if cols.len() != expected {
                return Ok(Expr::Identifier("False".to_string()));
              }
            } else {
              ncols = Some(cols.len());
            }
            // Each element must not be a list (must be a scalar)
            if cols.iter().any(|c| matches!(c, Expr::List(_))) {
              return Ok(Expr::Identifier("False".to_string()));
            }
          }
          _ => return Ok(Expr::Identifier("False".to_string())),
        }
      }
      Ok(Expr::Identifier("True".to_string()))
    }
    _ => Ok(Expr::Identifier("False".to_string())),
  }
}

/// AST-based TakeWhile: take elements while predicate is true.
pub fn take_while_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeWhile".to_string(),
        args: vec![list.clone(), pred.clone()],
      });
    }
  };

  let mut result = Vec::new();
  for item in items {
    let test_result = apply_func_ast(pred, item)?;
    if expr_to_bool(&test_result) == Some(true) {
      result.push(item.clone());
    } else {
      break;
    }
  }

  Ok(Expr::List(result))
}

/// AST-based Do: execute expression multiple times.
/// Do[expr, n] -> execute expr n times
/// Do[expr, {i, max}] -> execute with i from 1 to max
/// Do[expr, {i, min, max}] -> execute with i from min to max
pub fn do_ast(body: &Expr, iter_spec: &Expr) -> Result<Expr, InterpreterError> {
  match iter_spec {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let n = expr_to_i128(iter_spec).unwrap_or(0);
      for _ in 0..n {
        match crate::evaluator::evaluate_expr_to_expr(body) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
          Err(e) => return Err(e),
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    Expr::List(items) if items.len() == 1 => {
      // Do[body, {n}] — repeat n times without iterator variable
      let n_expr = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
      let n = expr_to_i128(&n_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Do: repeat count must be an integer".into(),
        )
      })?;
      for _ in 0..n {
        match crate::evaluator::evaluate_expr_to_expr(body) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
          Err(e) => return Err(e),
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    Expr::List(items) if items.len() >= 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Do: iterator variable must be an identifier".into(),
          ));
        }
      };

      let (min, max, step) = if items.len() == 2 {
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_val = expr_to_i128(&max_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        (1i128, max_val, 1i128)
      } else if items.len() >= 3 {
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
        let min_val = expr_to_i128(&min_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        let max_val = expr_to_i128(&max_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "Do: iterator bound must be an integer".into(),
          )
        })?;
        let step_val = if items.len() >= 4 {
          let step_expr = crate::evaluator::evaluate_expr_to_expr(&items[3])?;
          expr_to_i128(&step_expr).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Do: step must be an integer".into(),
            )
          })?
        } else {
          1i128
        };
        (min_val, max_val, step_val)
      } else {
        return Err(InterpreterError::EvaluationError(
          "Do: invalid iterator specification".into(),
        ));
      };

      if step == 0 {
        return Err(InterpreterError::EvaluationError(
          "Do: step cannot be zero".into(),
        ));
      }

      let mut i = min;
      if step > 0 {
        while i <= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          match crate::evaluator::evaluate_expr_to_expr(&substituted) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
            Err(e) => return Err(e),
          }
          i += step;
        }
      } else {
        while i >= max {
          let substituted = crate::syntax::substitute_variable(
            body,
            &var_name,
            &Expr::Integer(i),
          );
          match crate::evaluator::evaluate_expr_to_expr(&substituted) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
            Err(e) => return Err(e),
          }
          i += step;
        }
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Do: invalid iterator specification".into(),
    )),
  }
}

/// AST-based DeleteCases: remove elements matching pattern.
pub fn delete_cases_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  delete_cases_with_count_ast(list, pattern, None)
}

/// DeleteCases[list, pattern, levelspec, n] - delete at most n matches
pub fn delete_cases_with_count_ast(
  list: &Expr,
  pattern: &Expr,
  max_count: Option<i128>,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteCases".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let pattern_str = crate::syntax::expr_to_string(pattern);
  let mut removed = 0i128;
  let result: Vec<Expr> = items
    .iter()
    .filter(|item| {
      if let Some(max) = max_count
        && removed >= max
      {
        return true; // keep remaining items
      }
      let item_str = crate::syntax::expr_to_string(item);
      if matches_pattern_simple(&item_str, &pattern_str) {
        removed += 1;
        false
      } else {
        true
      }
    })
    .cloned()
    .collect();

  Ok(Expr::List(result))
}

/// AST-based MinMax: return {min, max} of a list.
pub fn min_max_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MinMax".to_string(),
        args: vec![list.clone()],
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![
      Expr::Identifier("Infinity".to_string()),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      },
    ]));
  }

  let mut min_val = f64::INFINITY;
  let mut max_val = f64::NEG_INFINITY;

  for item in items {
    if let Some(n) = expr_to_f64(item) {
      if n < min_val {
        min_val = n;
      }
      if n > max_val {
        max_val = n;
      }
    } else {
      return Ok(Expr::FunctionCall {
        name: "MinMax".to_string(),
        args: vec![list.clone()],
      });
    }
  }

  Ok(Expr::List(vec![f64_to_expr(min_val), f64_to_expr(max_val)]))
}

/// Part[list, i] or list[[i]] - Extract element at position i (1-indexed)
pub fn part_ast(list: &Expr, index: &Expr) -> Result<Expr, InterpreterError> {
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

/// Insert[list, elem, n] - Insert element at position n (1-indexed)
pub fn insert_ast(
  list: &Expr,
  elem: &Expr,
  pos: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Insert".to_string(),
        args: vec![list.clone(), elem.clone(), pos.clone()],
      });
    }
  };

  let n = match expr_to_i128(pos) {
    Some(n) => n,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Insert: position must be an integer".into(),
      ));
    }
  };

  let len = items.len() as i128;

  // Handle positive and negative indices
  let insert_pos = if n > 0 {
    let pos = (n - 1) as usize;
    if pos > items.len() {
      return Err(InterpreterError::EvaluationError(
        "Insert: position out of bounds".into(),
      ));
    }
    pos
  } else if n < 0 {
    let pos = (len + 1 + n) as usize;
    if n < -(len + 1) {
      return Err(InterpreterError::EvaluationError(
        "Insert: position out of bounds".into(),
      ));
    }
    pos
  } else {
    return Err(InterpreterError::EvaluationError(
      "Insert: position cannot be 0".into(),
    ));
  };

  let mut result = items;
  result.insert(insert_pos, elem.clone());
  Ok(Expr::List(result))
}

/// Array[f, n] - creates a list by applying f to indices 1..n
pub fn array_ast(func: &Expr, n: i128) -> Result<Expr, InterpreterError> {
  let mut result = Vec::new();
  for i in 1..=n {
    let arg = Expr::Integer(i);
    let val = apply_func_ast(func, &arg)?;
    result.push(val);
  }
  Ok(Expr::List(result))
}

/// Gather[list] - gathers elements into sublists of identical elements
pub fn gather_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Gather expects a list argument".into(),
      ));
    }
  };
  let mut groups: Vec<Vec<Expr>> = Vec::new();
  for item in items {
    let found = groups.iter_mut().find(|g| {
      crate::syntax::expr_to_string(&g[0])
        == crate::syntax::expr_to_string(item)
    });
    if let Some(group) = found {
      group.push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// GatherBy[list, f] - gathers elements into sublists by applying f
pub fn gather_by_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "GatherBy expects a list as first argument".into(),
      ));
    }
  };
  let mut groups: Vec<(String, Vec<Expr>)> = Vec::new();
  for item in items {
    let key = apply_func_ast(func, item)?;
    let key_str = crate::syntax::expr_to_string(&key);
    let found = groups.iter_mut().find(|(k, _)| *k == key_str);
    if let Some((_, group)) = found {
      group.push(item.clone());
    } else {
      groups.push((key_str, vec![item.clone()]));
    }
  }
  Ok(Expr::List(
    groups.into_iter().map(|(_, v)| Expr::List(v)).collect(),
  ))
}

/// Split[list] - splits into sublists of identical consecutive elements
pub fn split_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Split expects a list argument".into(),
      ));
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let last_group = groups.last().unwrap();
    if crate::syntax::expr_to_string(&last_group[0])
      == crate::syntax::expr_to_string(item)
    {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// Split[list, test] - splits list where consecutive elements satisfy test function
pub fn split_with_test_ast(
  list: &Expr,
  test: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Split".to_string(),
        args: vec![list.clone(), test.clone()],
      });
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let last_item = groups.last().unwrap().last().unwrap();
    let test_result = apply_func_to_two_args(test, last_item, item)?;
    let passes = matches!(
      &test_result,
      Expr::Identifier(name) if name == "True"
    );
    if passes {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// SplitBy[list, f] - splits into sublists of consecutive elements with same f value
pub fn split_by_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "SplitBy expects a list as first argument".into(),
      ));
    }
  };
  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }
  let mut prev_key = apply_func_ast(func, &items[0])?;
  let mut groups: Vec<Vec<Expr>> = vec![vec![items[0].clone()]];
  for item in items.iter().skip(1) {
    let key = apply_func_ast(func, item)?;
    if crate::syntax::expr_to_string(&key)
      == crate::syntax::expr_to_string(&prev_key)
    {
      groups.last_mut().unwrap().push(item.clone());
    } else {
      groups.push(vec![item.clone()]);
      prev_key = key;
    }
  }
  Ok(Expr::List(groups.into_iter().map(Expr::List).collect()))
}

/// Extract[list, n] - extracts element at position n
/// Extract[list, {n1, n2, ...}] - extracts element at nested position
pub fn extract_ast(
  list: &Expr,
  index: &Expr,
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
          results.push(extract_ast(list, pos_spec)?);
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

/// Catenate[{list1, list2, ...}] - concatenates lists
pub fn catenate_ast(list_of_lists: &Expr) -> Result<Expr, InterpreterError> {
  let outer = match list_of_lists {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Catenate expects a list of lists".into(),
      ));
    }
  };
  let mut result = Vec::new();
  for item in outer {
    match item {
      Expr::List(inner) => result.extend(inner.clone()),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Catenate expects all elements to be lists".into(),
        ));
      }
    }
  }
  Ok(Expr::List(result))
}

/// Apply[f, list] - applies f to the elements of list (f @@ list)
pub fn apply_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    Expr::FunctionCall { args, .. } => args.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Apply expects a list or expression as second argument".into(),
      ));
    }
  };
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, &items)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, &items);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body } => {
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(items.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Apply: first argument must be a function".into(),
    )),
  }
}

/// Identity[x] - returns x unchanged
pub fn identity_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  Ok(arg.clone())
}

/// Outer[f, list1, list2, ...] - generalized outer product
pub fn outer_ast(
  func: &Expr,
  lists: &[Expr],
) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Outer expects at least one list argument".into(),
    ));
  }
  outer_impl(func, &lists[0], &lists[1..], &[])
}

fn outer_impl(
  func: &Expr,
  current: &Expr,
  remaining: &[Expr],
  accumulated: &[Expr],
) -> Result<Expr, InterpreterError> {
  if let Expr::List(items) = current {
    // Thread through list elements
    let mut results = Vec::new();
    for item in items {
      results.push(outer_impl(func, item, remaining, accumulated)?);
    }
    Ok(Expr::List(results))
  } else {
    // Atomic element: add to accumulated args
    let mut new_acc = accumulated.to_vec();
    new_acc.push(current.clone());
    if remaining.is_empty() {
      // All lists consumed: apply f to accumulated args
      apply_func_to_n_args(func, &new_acc)
    } else {
      // Move to next list
      outer_impl(func, &remaining[0], &remaining[1..], &new_acc)
    }
  }
}

/// Inner[f, list1, list2, g] - generalized inner product
pub fn inner_ast(
  f: &Expr,
  list1: &Expr,
  list2: &Expr,
  g: &Expr,
) -> Result<Expr, InterpreterError> {
  inner_recursive(f, list1, list2, g)
}

fn inner_recursive(
  f: &Expr,
  a: &Expr,
  b: &Expr,
  g: &Expr,
) -> Result<Expr, InterpreterError> {
  let a_depth = list_depth(a);
  let b_depth = list_depth(b);

  if a_depth == 1 && b_depth == 1 {
    // Base case: both are flat lists - do pairwise f then combine with g
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let items_b = match b {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    if items_a.len() != items_b.len() {
      return Err(InterpreterError::EvaluationError(
        "Inner: incompatible dimensions".into(),
      ));
    }
    let mut products = Vec::new();
    for (x, y) in items_a.iter().zip(items_b.iter()) {
      let val = apply_func_to_two_args(f, x, y)?;
      products.push(val);
    }
    apply_func_to_n_args(g, &products)
  } else if a_depth > 1 {
    // Map over the first dimension of a
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let mut results = Vec::new();
    for row in items_a {
      results.push(inner_recursive(f, row, b, g)?);
    }
    Ok(Expr::List(results))
  } else {
    // a_depth == 1, b_depth > 1
    // Contract a (vector) with first dimension of b
    // Need to iterate over the inner structure of b
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let items_b = match b {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    if items_a.len() != items_b.len() {
      return Err(InterpreterError::EvaluationError(
        "Inner: incompatible dimensions".into(),
      ));
    }
    // Each items_b[k] should be a list. Extract their elements column-wise.
    let inner_len = match &items_b[0] {
      Expr::List(inner) => inner.len(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner: incompatible dimensions".into(),
        ));
      }
    };
    let mut results = Vec::new();
    for j in 0..inner_len {
      // Build a "column" vector from b: [b[0][j], b[1][j], ...]
      let mut col = Vec::new();
      for bk in items_b {
        match bk {
          Expr::List(inner) => {
            if j < inner.len() {
              col.push(inner[j].clone());
            } else {
              return Err(InterpreterError::EvaluationError(
                "Inner: incompatible dimensions".into(),
              ));
            }
          }
          _ => {
            return Err(InterpreterError::EvaluationError(
              "Inner: incompatible dimensions".into(),
            ));
          }
        }
      }
      let col_expr = Expr::List(col);
      results.push(inner_recursive(f, a, &col_expr, g)?);
    }
    Ok(Expr::List(results))
  }
}

fn list_depth(expr: &Expr) -> usize {
  match expr {
    Expr::List(items) => {
      if items.is_empty() {
        1
      } else {
        1 + items.iter().map(list_depth).min().unwrap_or(0)
      }
    }
    _ => 0,
  }
}

/// ReplacePart[list, n -> val] - replaces element at position n
pub fn replace_part_ast(
  list: &Expr,
  rule: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ReplacePart expects a list as first argument".into(),
      ));
    }
  };
  let (pos, val) = match rule {
    Expr::Rule {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ReplacePart expects a rule as second argument".into(),
      ));
    }
  };
  let idx = match expr_to_i128(pos) {
    Some(n) => {
      let len = items.len() as i128;
      if n > 0 && n <= len {
        (n - 1) as usize
      } else if n < 0 && -n <= len {
        (len + n) as usize
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "ReplacePart: position {} out of range",
          n
        )));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ReplacePart: position must be an integer".into(),
      ));
    }
  };
  let mut result = items;
  result[idx] = val.clone();
  Ok(Expr::List(result))
}

/// AST-based Permutations: generate all permutations of a list.
/// Permutations[{a, b, c}] -> all permutations
/// Permutations[{a, b, c}, {k}] -> all permutations of length k
pub fn permutations_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let list = &args[0];
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Permutations".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let k = if args.len() >= 2 {
    // Second arg should be {k} — a list with one integer
    match &args[1] {
      Expr::List(spec) if spec.len() == 1 => {
        expr_to_i128(&spec[0]).unwrap_or(items.len() as i128) as usize
      }
      Expr::Integer(_) | Expr::BigInteger(_) => {
        expr_to_i128(&args[1]).unwrap_or(items.len() as i128) as usize
      }
      _ => items.len(),
    }
  } else {
    items.len()
  };

  let n = items.len();
  if k > n {
    return Ok(Expr::List(vec![]));
  }

  let mut result = Vec::new();
  let indices: Vec<usize> = (0..n).collect();
  generate_k_permutations(
    &indices,
    k,
    &mut vec![],
    &mut vec![false; n],
    &items,
    &mut result,
  );
  Ok(Expr::List(result))
}

/// Helper to generate k-permutations
fn generate_k_permutations(
  _indices: &[usize],
  k: usize,
  current: &mut Vec<usize>,
  used: &mut Vec<bool>,
  items: &[Expr],
  result: &mut Vec<Expr>,
) {
  if current.len() == k {
    let perm: Vec<Expr> = current.iter().map(|&i| items[i].clone()).collect();
    result.push(Expr::List(perm));
    return;
  }
  for i in 0..items.len() {
    if !used[i] {
      used[i] = true;
      current.push(i);
      generate_k_permutations(_indices, k, current, used, items, result);
      current.pop();
      used[i] = false;
    }
  }
}

/// AST-based Subsets: generate subsets of a list.
/// Subsets[{a, b, c}] -> all subsets
/// Subsets[{a, b, c}, {k}] -> all subsets of size k
pub fn subsets_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let list = &args[0];
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Subsets".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() >= 2 {
    match &args[1] {
      // Subsets[list, {k}] - subsets of exactly size k
      Expr::List(spec) if spec.len() == 1 => {
        if let Some(k) = expr_to_i128(&spec[0]) {
          let k = k.max(0) as usize;
          let mut result = Vec::new();
          generate_combinations(&items, k, 0, &mut vec![], &mut result);
          // Handle optional max count argument
          if args.len() >= 3
            && let Some(max) = expr_to_i128(&args[2])
          {
            result.truncate(max.max(0) as usize);
          }
          return Ok(Expr::List(result));
        }
      }
      // Subsets[list, {min, max}] - subsets of sizes min through max
      Expr::List(spec) if spec.len() == 2 => {
        if let (Some(min), Some(max)) =
          (expr_to_i128(&spec[0]), expr_to_i128(&spec[1]))
        {
          let min = min.max(0) as usize;
          let max = max.min(items.len() as i128).max(0) as usize;
          let mut result = Vec::new();
          for k in min..=max {
            generate_combinations(&items, k, 0, &mut vec![], &mut result);
          }
          return Ok(Expr::List(result));
        }
      }
      // Subsets[list, {min, max, step}] - subsets of sizes min, min+step, ...
      Expr::List(spec) if spec.len() == 3 => {
        if let (Some(min), Some(max), Some(step)) = (
          expr_to_i128(&spec[0]),
          expr_to_i128(&spec[1]),
          expr_to_i128(&spec[2]),
        ) {
          let min = min.max(0) as usize;
          let max = max.min(items.len() as i128).max(0) as usize;
          let step = step.max(1) as usize;
          let mut result = Vec::new();
          let mut k = min;
          while k <= max {
            generate_combinations(&items, k, 0, &mut vec![], &mut result);
            k += step;
          }
          return Ok(Expr::List(result));
        }
      }
      // Subsets[list, n] - all subsets up to size n
      _ => {
        if let Some(max_k) = expr_to_i128(&args[1]) {
          let max_k = max_k.min(items.len() as i128).max(0) as usize;
          let mut result = Vec::new();
          for k in 0..=max_k {
            generate_combinations(&items, k, 0, &mut vec![], &mut result);
          }
          return Ok(Expr::List(result));
        }
      }
    }
  }

  // Subsets[list] - all subsets
  let n = items.len();
  let mut result = Vec::new();
  for k in 0..=n {
    generate_combinations(&items, k, 0, &mut vec![], &mut result);
  }
  Ok(Expr::List(result))
}

/// Helper to generate combinations (subsets of size k)
fn generate_combinations(
  items: &[Expr],
  k: usize,
  start: usize,
  current: &mut Vec<Expr>,
  result: &mut Vec<Expr>,
) {
  if current.len() == k {
    result.push(Expr::List(current.clone()));
    return;
  }
  for i in start..items.len() {
    current.push(items[i].clone());
    generate_combinations(items, k, i + 1, current, result);
    current.pop();
  }
}

/// AST-based SparseArray: create a matrix from position rules.
/// SparseArray[rules, {rows, cols}, default] -> evaluates rules and creates matrix
pub fn sparse_array_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Ok(Expr::FunctionCall {
      name: "SparseArray".to_string(),
      args: args.to_vec(),
    });
  }

  let rules = &args[0];
  let dims = &args[1];
  let default = &args[2];

  // Extract dimensions
  let dim_values = match dims {
    Expr::List(items) => {
      let mut result = Vec::new();
      for item in items {
        match expr_to_i128(item) {
          Some(n) => result.push(n as usize),
          None => {
            return Ok(Expr::FunctionCall {
              name: "SparseArray".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
      result
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if dim_values.len() == 2 {
    let rows = dim_values[0];
    let cols = dim_values[1];

    // Initialize matrix with default value
    let mut matrix: Vec<Vec<Expr>> = vec![vec![default.clone(); cols]; rows];

    // Process rules: {pos -> val, pos -> val, ...}
    let rule_list = match rules {
      Expr::List(items) => items.clone(),
      _ => vec![rules.clone()],
    };

    for rule in &rule_list {
      match rule {
        Expr::Rule {
          pattern,
          replacement,
        } => {
          // pattern should be {row, col} (1-indexed)
          if let Expr::List(pos) = pattern.as_ref()
            && pos.len() == 2
            && let (Expr::Integer(r), Expr::Integer(c)) = (&pos[0], &pos[1])
          {
            let ri = (*r - 1) as usize;
            let ci = (*c - 1) as usize;
            if ri < rows && ci < cols {
              matrix[ri][ci] = replacement.as_ref().clone();
            }
          }
        }
        _ => {} // skip non-rules
      }
    }

    // Convert to nested list
    let result: Vec<Expr> = matrix.into_iter().map(Expr::List).collect();
    return Ok(Expr::List(result));
  }

  // For non-2D arrays, return symbolic
  Ok(Expr::FunctionCall {
    name: "SparseArray".to_string(),
    args: args.to_vec(),
  })
}

/// Tuples[list, n] - Generate all n-tuples from elements of list (Cartesian product).
pub fn tuples_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Tuples[{list1, list2, ...}] - Cartesian product of multiple lists
    // Each element can be a List or a FunctionCall (extract args as elements)
    let outer_items = match &args[0] {
      Expr::List(items) => items,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Tuples".to_string(),
          args: args.to_vec(),
        });
      }
    };

    // Extract elements from each sublist/expression
    let mut lists: Vec<Vec<Expr>> = Vec::new();
    for item in outer_items {
      match item {
        Expr::List(items) => lists.push(items.clone()),
        Expr::FunctionCall { args: fc_args, .. } => {
          lists.push(fc_args.clone());
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Tuples".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }

    // Cartesian product of all lists
    let mut result: Vec<Vec<Expr>> = vec![vec![]];
    for list in &lists {
      let mut new_result = Vec::new();
      for tuple in &result {
        for item in list {
          let mut new_tuple = tuple.clone();
          new_tuple.push(item.clone());
          new_result.push(new_tuple);
        }
      }
      result = new_result;
    }

    return Ok(Expr::List(result.into_iter().map(Expr::List).collect()));
  }

  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Tuples expects 1 or 2 arguments".into(),
    ));
  }

  // Tuples[list, n] or Tuples[f[a,b,...], n]
  let (items, head_name): (Vec<Expr>, Option<String>) = match &args[0] {
    Expr::List(items) => (items.clone(), None),
    Expr::FunctionCall {
      name,
      args: fc_args,
    } => (fc_args.clone(), Some(name.clone())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tuples".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = match &args[1] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Tuples".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if n == 0 {
    let empty = if let Some(ref h) = head_name {
      Expr::FunctionCall {
        name: h.clone(),
        args: vec![],
      }
    } else {
      Expr::List(vec![])
    };
    return Ok(Expr::List(vec![empty]));
  }

  // Iterative Cartesian product
  let mut result: Vec<Vec<Expr>> = vec![vec![]];

  for _ in 0..n {
    let mut new_result = Vec::new();
    for tuple in &result {
      for item in &items {
        let mut new_tuple = tuple.clone();
        new_tuple.push(item.clone());
        new_result.push(new_tuple);
      }
    }
    result = new_result;
  }

  let wrap = |elems: Vec<Expr>| -> Expr {
    if let Some(ref h) = head_name {
      Expr::FunctionCall {
        name: h.clone(),
        args: elems,
      }
    } else {
      Expr::List(elems)
    }
  };

  Ok(Expr::List(result.into_iter().map(wrap).collect()))
}

/// Dimensions[list] - Returns the dimensions of a nested list
pub fn dimensions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Dimensions expects exactly 1 argument".into(),
    ));
  }

  fn get_dimensions(expr: &Expr) -> Vec<i128> {
    match expr {
      Expr::List(items) => {
        let mut dims = vec![items.len() as i128];
        if !items.is_empty() {
          // Check if all sub-elements have the same dimensions
          let sub_dims: Vec<Vec<i128>> =
            items.iter().map(get_dimensions).collect();
          if !sub_dims.is_empty() && sub_dims.iter().all(|d| d == &sub_dims[0])
          {
            dims.extend(sub_dims[0].iter());
          }
        }
        dims
      }
      Expr::FunctionCall { name, args } => {
        let mut dims = vec![args.len() as i128];
        if !args.is_empty() {
          // Check if all sub-elements are function calls with the same head and dimensions
          let sub_dims: Vec<Vec<i128>> =
            args.iter().map(get_dimensions).collect();
          if !sub_dims.is_empty()
            && sub_dims.iter().all(|d| d == &sub_dims[0])
            && args.iter().all(|a| {
              matches!(a, Expr::FunctionCall { name: n, .. } if n == name)
                || matches!(a, Expr::List(_))
            })
          {
            dims.extend(sub_dims[0].iter());
          }
        }
        dims
      }
      _ => vec![],
    }
  }

  let dims = get_dimensions(&args[0]);
  Ok(Expr::List(dims.into_iter().map(Expr::Integer).collect()))
}

/// Delete[list, pos] - Delete an element at a position
pub fn delete_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Delete expects exactly 2 arguments".into(),
    ));
  }

  if let Expr::List(items) = &args[0] {
    match &args[1] {
      // Delete[list, n] - delete at position n
      Expr::Integer(_) | Expr::BigInteger(_) => {
        let pos = match expr_to_i128(&args[1]) {
          Some(n) => n,
          None => return Ok(args[0].clone()),
        };
        return delete_at_position(items, pos);
      }
      Expr::List(pos_list) => {
        // Determine if this is a multi-part position {i, j, ...} or multiple positions {{p1}, {p2}, ...}
        let is_multiple_positions =
          pos_list.iter().all(|p| matches!(p, Expr::List(_)));

        if is_multiple_positions && !pos_list.is_empty() {
          // Multiple positions: Delete[list, {{p1}, {p2}, ...}]
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
          // For single-element position lists, collect flat indices
          let mut result = args[0].clone();
          // Sort positions in reverse to avoid index shifting issues
          // For multi-part positions, we need to handle them one at a time
          for pos in positions.iter().rev() {
            if pos.len() == 1 {
              if let Expr::List(items) = &result {
                result = delete_at_position(items, pos[0])?;
              }
            } else {
              result = delete_at_deep_position(&result, pos)?;
            }
          }
          return Ok(result);
        } else {
          // Multi-part position: Delete[list, {i, j, ...}]
          let pos: Vec<i128> =
            pos_list.iter().filter_map(expr_to_i128).collect();
          if pos.len() == pos_list.len() {
            if pos.len() == 1 {
              return delete_at_position(items, pos[0]);
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
  } else {
    Ok(Expr::FunctionCall {
      name: "Delete".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Delete element at a single flat position in a list
fn delete_at_position(
  items: &[Expr],
  pos: i128,
) -> Result<Expr, InterpreterError> {
  let len = items.len() as i128;
  let idx = if pos > 0 {
    (pos - 1) as usize
  } else if pos < 0 {
    (len + pos) as usize
  } else {
    return Ok(Expr::List(items.to_vec()));
  };
  if idx >= items.len() {
    return Ok(Expr::List(items.to_vec()));
  }
  let mut result = items.to_vec();
  result.remove(idx);
  Ok(Expr::List(result))
}

/// Delete element at a deep multi-part position {i, j, ...}
fn delete_at_deep_position(
  expr: &Expr,
  pos: &[i128],
) -> Result<Expr, InterpreterError> {
  if pos.is_empty() {
    return Ok(expr.clone());
  }
  if let Expr::List(items) = expr {
    let len = items.len() as i128;
    let idx = if pos[0] > 0 {
      (pos[0] - 1) as usize
    } else if pos[0] < 0 {
      (len + pos[0]) as usize
    } else {
      return Ok(expr.clone());
    };
    if idx >= items.len() {
      return Ok(expr.clone());
    }
    if pos.len() == 1 {
      let mut result = items.to_vec();
      result.remove(idx);
      Ok(Expr::List(result))
    } else {
      let mut result = items.to_vec();
      result[idx] = delete_at_deep_position(&items[idx], &pos[1..])?;
      Ok(Expr::List(result))
    }
  } else {
    Ok(expr.clone())
  }
}

/// OrderedQ[list] - Tests if a list is in sorted (non-decreasing) order
pub fn ordered_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "OrderedQ expects exactly 1 argument".into(),
    ));
  }

  if let Expr::List(items) = &args[0] {
    if items.len() <= 1 {
      return Ok(Expr::Identifier("True".to_string()));
    }
    for i in 0..items.len() - 1 {
      if !expr_le(&items[i], &items[i + 1]) {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
    Ok(Expr::Identifier("True".to_string()))
  } else {
    Ok(Expr::FunctionCall {
      name: "OrderedQ".to_string(),
      args: args.to_vec(),
    })
  }
}

// ─── DeleteAdjacentDuplicates ──────────────────────────────────────

/// DeleteAdjacentDuplicates[list] - removes consecutive duplicate elements
pub fn delete_adjacent_duplicates_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "DeleteAdjacentDuplicates expects exactly 1 argument".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DeleteAdjacentDuplicates".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let mut result = vec![items[0].clone()];
  for item in items.iter().skip(1) {
    if crate::syntax::expr_to_string(item)
      != crate::syntax::expr_to_string(result.last().unwrap())
    {
      result.push(item.clone());
    }
  }
  Ok(Expr::List(result))
}

// ─── Commonest ─────────────────────────────────────────────────────

/// Commonest[list] - returns the most common element(s)
/// Commonest[list, n] - returns the n most common elements
pub fn commonest_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Commonest expects 1 or 2 arguments".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Commonest".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let n = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n >= 1 => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Commonest".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    1
  };

  // Count occurrences, preserving order of first appearance
  let mut counts: Vec<(String, &Expr, usize)> = Vec::new();
  for item in items {
    let key = crate::syntax::expr_to_string(item);
    if let Some(entry) = counts.iter_mut().find(|(k, _, _)| k == &key) {
      entry.2 += 1;
    } else {
      counts.push((key, item, 1));
    }
  }

  // Sort by count descending (stable sort preserves insertion order for ties)
  counts.sort_by(|a, b| b.2.cmp(&a.2));

  // Take top n distinct count levels
  let mut result = Vec::new();
  let mut distinct_counts = 0;
  let mut last_count = 0;
  for (_, item, count) in &counts {
    if *count != last_count {
      distinct_counts += 1;
      if distinct_counts > n {
        break;
      }
      last_count = *count;
    }
    result.push((*item).clone());
  }

  Ok(Expr::List(result))
}

// ─── ComposeList ──────────────────────────────────────────────────

/// ComposeList[{f, g, h}, x] -> {x, f[x], g[f[x]], h[g[f[x]]]}
pub fn compose_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ComposeList expects exactly 2 arguments".into(),
    ));
  }
  let funcs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ComposeList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut result = vec![args[1].clone()];
  let mut current = args[1].clone();
  for func in funcs {
    current = apply_func_ast(func, &current)?;
    result.push(current.clone());
  }
  Ok(Expr::List(result))
}

/// Compare two Expr values for canonical ordering.
/// Returns 1 if a < b, -1 if a > b, 0 if equal (Wolfram Order convention).
pub fn compare_exprs(a: &Expr, b: &Expr) -> i64 {
  // Try numeric comparison first
  let a_num = expr_to_f64(a);
  let b_num = expr_to_f64(b);
  if let (Some(an), Some(bn)) = (a_num, b_num) {
    return if an < bn {
      1
    } else if an > bn {
      -1
    } else {
      0
    };
  }
  // Numbers come before non-numbers
  if a_num.is_some() {
    return 1;
  }
  if b_num.is_some() {
    return -1;
  }

  // Wolfram canonical ordering: symbols and compounds are compared structurally
  let a_is_atom = is_atom_expr(a);
  let b_is_atom = is_atom_expr(b);

  match (a_is_atom, b_is_atom) {
    (true, true) => {
      // Both atoms: alphabetical comparison
      let a_str = crate::syntax::expr_to_string(a);
      let b_str = crate::syntax::expr_to_string(b);
      wolfram_string_order(&a_str, &b_str)
    }
    (true, false) => {
      // Atom vs compound: compare atom with compound's sort key
      let b_key = expr_sort_key(b);
      let a_str = crate::syntax::expr_to_string(a);
      let cmp = wolfram_string_order(&a_str, &b_key);
      if cmp == 0 {
        1 // atom comes before compound with same key
      } else {
        cmp
      }
    }
    (false, true) => {
      // Compound vs atom: reverse of above
      let a_key = expr_sort_key(a);
      let b_str = crate::syntax::expr_to_string(b);
      let cmp = wolfram_string_order(&a_key, &b_str);
      if cmp == 0 {
        -1 // compound comes after atom with same key
      } else {
        cmp
      }
    }
    (false, false) => {
      // Both compounds: compare sort keys, then by full string
      let a_key = expr_sort_key(a);
      let b_key = expr_sort_key(b);
      let cmp = wolfram_string_order(&a_key, &b_key);
      if cmp != 0 {
        return cmp;
      }
      let a_str = crate::syntax::expr_to_string(a);
      let b_str = crate::syntax::expr_to_string(b);
      wolfram_string_order(&a_str, &b_str)
    }
  }
}

/// Check if an expression is an atomic (non-compound) expression
fn is_atom_expr(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(_) | Expr::Constant(_) | Expr::String(_))
}

/// Extract the sort key for a compound expression.
/// For Plus/Times: the last (largest) symbolic argument
/// For Power: the base
/// For other functions: the last argument, or the function name
fn expr_sort_key(e: &Expr) -> String {
  match e {
    Expr::FunctionCall { name, args } if !args.is_empty() => {
      // For Orderless functions (Plus, Times), use the last argument as sort key
      if let Some(last) = args.last()
        && is_atom_expr(last)
      {
        return crate::syntax::expr_to_string(last);
      }
      // Fallback: use function name
      name.clone()
    }
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator;
      match op {
        BinaryOperator::Power => {
          // Power: sort key is the base
          crate::syntax::expr_to_string(left)
        }
        BinaryOperator::Plus | BinaryOperator::Times => {
          // For binary plus/times: use the "larger" operand
          let l = crate::syntax::expr_to_string(left);
          let r = crate::syntax::expr_to_string(right);
          if wolfram_string_order(&l, &r) >= 0 {
            r
          } else {
            l
          }
        }
        _ => crate::syntax::expr_to_string(e),
      }
    }
    _ => crate::syntax::expr_to_string(e),
  }
}

/// Wolfram canonical string ordering: case-insensitive alphabetical, then lowercase < uppercase
fn wolfram_string_order(a: &str, b: &str) -> i64 {
  let a_chars: Vec<char> = a.chars().collect();
  let b_chars: Vec<char> = b.chars().collect();

  for (ac, bc) in a_chars.iter().zip(b_chars.iter()) {
    let al = ac.to_lowercase().next().unwrap_or(*ac);
    let bl = bc.to_lowercase().next().unwrap_or(*bc);
    if al != bl {
      // Case-insensitive comparison first
      return if al < bl { 1 } else { -1 };
    }
    // Same letter, different case: lowercase comes first
    if ac != bc {
      // lowercase < uppercase in Wolfram ordering
      if ac.is_lowercase() && bc.is_uppercase() {
        return 1;
      } else if ac.is_uppercase() && bc.is_lowercase() {
        return -1;
      }
    }
  }
  // If all compared chars are equal, shorter string comes first
  match a_chars.len().cmp(&b_chars.len()) {
    std::cmp::Ordering::Less => 1,
    std::cmp::Ordering::Greater => -1,
    std::cmp::Ordering::Equal => 0,
  }
}

/// Helper: compare two Expr values for ordering (less-or-equal)
fn expr_le(a: &Expr, b: &Expr) -> bool {
  compare_exprs(a, b) >= 0
}

/// Subsequences[list] - all contiguous subsequences
/// Subsequences[list, {n}] - contiguous subsequences of length n
/// Subsequences[list, {nmin, nmax}] - lengths in range
/// Subsequences[{a, b, c}] => {{}, {a}, {b}, {c}, {a, b}, {b, c}, {a, b, c}}
pub fn subsequences_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Subsequences expects 1 or 2 arguments".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Subsequences".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = items.len();

  // Determine min and max lengths
  let (min_len, max_len) = if args.len() == 2 {
    match &args[1] {
      Expr::List(spec) => {
        if spec.len() == 1 {
          // {n} - exactly length n
          if let Expr::Integer(k) = &spec[0] {
            let k = *k as usize;
            (k, k)
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsequences".to_string(),
              args: args.to_vec(),
            });
          }
        } else if spec.len() == 2 {
          // {nmin, nmax}
          if let (Expr::Integer(lo), Expr::Integer(hi)) = (&spec[0], &spec[1]) {
            (*lo as usize, *hi as usize)
          } else {
            return Ok(Expr::FunctionCall {
              name: "Subsequences".to_string(),
              args: args.to_vec(),
            });
          }
        } else {
          return Ok(Expr::FunctionCall {
            name: "Subsequences".to_string(),
            args: args.to_vec(),
          });
        }
      }
      Expr::Integer(_) | Expr::BigInteger(_) => {
        let k = expr_to_i128(&args[1]).unwrap_or(0) as usize;
        (k, k)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Subsequences".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    (0, n)
  };

  let mut result = Vec::new();
  for len in min_len..=max_len.min(n) {
    if len == 0 {
      result.push(Expr::List(vec![]));
    } else {
      for start in 0..=(n - len) {
        result.push(Expr::List(items[start..start + len].to_vec()));
      }
    }
  }
  Ok(Expr::List(result))
}

/// BinCounts[data, {min, max, dx}] - count data points in equal-width bins
/// Bins are [min, min+dx), [min+dx, min+2dx), ..., [max-dx, max)
pub fn bin_counts_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BinCounts".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract numeric values from data, skip non-numeric
  let values: Vec<f64> = data.iter().filter_map(expr_to_f64).collect();

  let (min_val, max_val, dx) = if args.len() == 1 {
    // BinCounts[data] - default dx=1, aligned to integer boundaries
    if values.is_empty() {
      return Ok(Expr::List(vec![]));
    }
    let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let dx = 1.0;
    let mut lo = (data_min / dx).floor() * dx;
    if (data_min - lo).abs() < 1e-12 {
      lo -= dx;
    }
    let mut hi = (data_max / dx).ceil() * dx;
    if (data_max - hi).abs() < 1e-12 {
      hi += dx;
    }
    (lo, hi, dx)
  } else if args.len() == 2 {
    match &args[1] {
      // BinCounts[data, dx]
      Expr::Integer(dx_int) => {
        if values.is_empty() {
          return Ok(Expr::List(vec![]));
        }
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dx = *dx_int as f64;
        let mut lo = (data_min / dx).floor() * dx;
        if (data_min - lo).abs() < 1e-12 {
          lo -= dx;
        }
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 {
          hi += dx;
        }
        (lo, hi, dx)
      }
      Expr::Real(dx_f) => {
        if values.is_empty() {
          return Ok(Expr::List(vec![]));
        }
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let dx = *dx_f;
        let mut lo = (data_min / dx).floor() * dx;
        if (data_min - lo).abs() < 1e-12 {
          lo -= dx;
        }
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 {
          hi += dx;
        }
        (lo, hi, dx)
      }
      // BinCounts[data, {min, max, dx}]
      Expr::List(spec) if spec.len() == 3 => {
        let min_v = match expr_to_f64(&spec[0]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let max_v = match expr_to_f64(&spec[1]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let dx = match expr_to_f64(&spec[2]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "BinCounts".to_string(),
              args: args.to_vec(),
            });
          }
        };
        (min_v, max_v, dx)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "BinCounts".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    return Ok(Expr::FunctionCall {
      name: "BinCounts".to_string(),
      args: args.to_vec(),
    });
  };

  if dx <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "BinCounts: bin width must be positive".into(),
    ));
  }

  let num_bins = ((max_val - min_val) / dx).round() as usize;
  let mut counts = vec![0i128; num_bins];

  for &v in &values {
    if v >= min_val && v < max_val {
      let bin = ((v - min_val) / dx) as usize;
      let bin = bin.min(num_bins - 1);
      counts[bin] += 1;
    }
  }

  Ok(Expr::List(counts.into_iter().map(Expr::Integer).collect()))
}

/// Round a positive value to the nearest "nice" number (1, 2, 5, 10, 20, 50, …)
/// using log-scale distance to decide which is closest.
fn nice_number(x: f64) -> f64 {
  if x <= 0.0 {
    return 1.0;
  }
  let exp = x.log10().floor() as i32;
  let base = 10f64.powi(exp);
  let candidates = [1.0 * base, 2.0 * base, 5.0 * base, 10.0 * base];
  let log_x = x.ln();
  candidates
    .iter()
    .copied()
    .min_by(|a, b| {
      (a.ln() - log_x)
        .abs()
        .partial_cmp(&(b.ln() - log_x).abs())
        .unwrap()
    })
    .unwrap()
}

/// Compute the interquartile range (Q3 - Q1) of a sorted slice.
fn interquartile_range(sorted: &[f64]) -> f64 {
  let n = sorted.len();
  if n < 2 {
    return 0.0;
  }
  let q1 = wolfram_quantile(sorted, 0.25);
  let q3 = wolfram_quantile(sorted, 0.75);
  q3 - q1
}

/// Wolfram's default Quantile: h = n*p, j = floor(h), g = h - j.
/// If g > 0: x_{j+1}, else x_j (1-based indexing).
fn wolfram_quantile(sorted: &[f64], p: f64) -> f64 {
  let n = sorted.len();
  if n == 0 {
    return 0.0;
  }
  let h = n as f64 * p;
  let j = h.floor() as usize;
  let g = h - j as f64;
  if g > 1e-12 {
    // x_{j+1}, 0-based: sorted[j]
    sorted[j.min(n - 1)]
  } else {
    // x_j, 0-based: sorted[j-1], but j could be 0
    if j == 0 {
      sorted[0]
    } else {
      sorted[(j - 1).min(n - 1)]
    }
  }
}

/// HistogramList[data] - returns {bin_edges, counts}
/// HistogramList[data, {dx}] - explicit bin width
/// HistogramList[data, {min, max, dx}] - explicit bin specification
pub fn histogram_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "HistogramList".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let values: Vec<f64> = data.iter().filter_map(expr_to_f64).collect();

  if values.is_empty() {
    return Ok(Expr::List(vec![Expr::List(vec![]), Expr::List(vec![])]));
  }

  let (min_val, max_val, dx) = if args.len() == 1 {
    // Auto-binning: Freedman-Diaconis rule with nice number rounding
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let data_min = sorted[0];
    let data_max = sorted[sorted.len() - 1];
    let n = sorted.len() as f64;

    let iqr = interquartile_range(&sorted);
    let dx = if iqr > 0.0 {
      nice_number(2.0 * iqr / n.cbrt())
    } else if data_max > data_min {
      // Fallback: use range / Sturges' rule
      let sturges_bins = (n.log2() + 1.0).ceil().max(1.0);
      nice_number((data_max - data_min) / sturges_bins)
    } else {
      // All values identical
      let v = data_min.abs();
      if v == 0.0 { 1.0 } else { nice_number(v) }
    };

    let lo = (data_min / dx).floor() * dx;
    let mut hi = (data_max / dx).ceil() * dx;
    // Ensure data_max is strictly inside the last bin
    if (data_max - hi).abs() < 1e-12 * dx.max(1.0) {
      hi += dx;
    }
    (lo, hi, dx)
  } else if args.len() == 2 {
    match &args[1] {
      // HistogramList[data, {dx}]
      Expr::List(spec) if spec.len() == 1 => {
        let dx = match expr_to_f64(&spec[0]) {
          Some(v) if v > 0.0 => v,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let data_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = (data_min / dx).floor() * dx;
        let mut hi = (data_max / dx).ceil() * dx;
        if (data_max - hi).abs() < 1e-12 * dx.max(1.0) {
          hi += dx;
        }
        (lo, hi, dx)
      }
      // HistogramList[data, {min, max, dx}]
      Expr::List(spec) if spec.len() == 3 => {
        let min_v = match expr_to_f64(&spec[0]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let max_v = match expr_to_f64(&spec[1]) {
          Some(v) => v,
          None => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec(),
            });
          }
        };
        let dx = match expr_to_f64(&spec[2]) {
          Some(v) if v > 0.0 => v,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "HistogramList".to_string(),
              args: args.to_vec(),
            });
          }
        };
        (min_v, max_v, dx)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "HistogramList".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    return Ok(Expr::FunctionCall {
      name: "HistogramList".to_string(),
      args: args.to_vec(),
    });
  };

  if dx <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "HistogramList: bin width must be positive".into(),
    ));
  }

  let num_bins = ((max_val - min_val) / dx).round() as usize;
  if num_bins == 0 {
    return Ok(Expr::List(vec![Expr::List(vec![]), Expr::List(vec![])]));
  }

  let mut counts = vec![0i128; num_bins];
  for &v in &values {
    if v >= min_val && v <= max_val {
      let bin = ((v - min_val) / dx) as usize;
      let bin = bin.min(num_bins - 1);
      counts[bin] += 1;
    }
  }

  // Build bin edges list
  let mut edges = Vec::with_capacity(num_bins + 1);
  for i in 0..=num_bins {
    edges.push(f64_to_expr(min_val + i as f64 * dx));
  }

  Ok(Expr::List(vec![
    Expr::List(edges),
    Expr::List(counts.into_iter().map(Expr::Integer).collect()),
  ]))
}

/// ContainsOnly[list, elems] - True if every element of list is in elems
pub fn contains_only_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ContainsOnly expects exactly 2 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ContainsOnly".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let elems = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ContainsOnly".to_string(),
        args: args.to_vec(),
      });
    }
  };

  use std::collections::HashSet;
  let allowed: HashSet<String> =
    elems.iter().map(crate::syntax::expr_to_string).collect();

  for item in list {
    if !allowed.contains(&crate::syntax::expr_to_string(item)) {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// LengthWhile[list, crit] - gives the number of contiguous elements at the start that satisfy crit
pub fn length_while_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "LengthWhile expects exactly 2 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LengthWhile".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let crit = &args[1];
  let mut count: i128 = 0;
  for item in list {
    let test = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![crit.clone(), Expr::List(vec![item.clone()])],
    })?;
    match &test {
      Expr::Identifier(s) if s == "True" => count += 1,
      _ => break,
    }
  }
  Ok(Expr::Integer(count))
}

/// TakeLargestBy[list, f, n] - take the n largest elements sorted by f
pub fn take_largest_by_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "TakeLargestBy expects exactly 3 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargestBy".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let f = &args[1];
  let n = match &args[2] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeLargestBy".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Compute f[item] for each item
  let mut with_keys: Vec<(Expr, Expr)> = Vec::new();
  for item in list {
    let key = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![f.clone(), Expr::List(vec![item.clone()])],
    })?;
    with_keys.push((key, item.clone()));
  }
  // Sort descending by key (largest first)
  // compare_exprs returns 1 if first < second in canonical order
  with_keys.sort_by(|a, b| {
    let ord = compare_exprs(&a.0, &b.0);
    // ord > 0 means a.key < b.key, so b should come first (descending)
    if ord > 0 {
      std::cmp::Ordering::Greater
    } else if ord < 0 {
      std::cmp::Ordering::Less
    } else {
      std::cmp::Ordering::Equal
    }
  });
  let result: Vec<Expr> =
    with_keys.into_iter().take(n).map(|(_, v)| v).collect();
  Ok(Expr::List(result))
}

/// TakeSmallestBy[list, f, n] - take the n smallest elements sorted by f
pub fn take_smallest_by_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "TakeSmallestBy expects exactly 3 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallestBy".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let f = &args[1];
  let n = match &args[2] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeSmallestBy".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Compute f[item] for each item
  let mut with_keys: Vec<(Expr, Expr)> = Vec::new();
  for item in list {
    let key = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![f.clone(), Expr::List(vec![item.clone()])],
    })?;
    with_keys.push((key, item.clone()));
  }
  // Sort ascending by key (smallest first)
  // compare_exprs returns 1 if first < second in canonical order
  with_keys.sort_by(|a, b| {
    let ord = compare_exprs(&a.0, &b.0);
    // ord > 0 means a.key < b.key, so a should come first (ascending)
    if ord > 0 {
      std::cmp::Ordering::Less
    } else if ord < 0 {
      std::cmp::Ordering::Greater
    } else {
      std::cmp::Ordering::Equal
    }
  });
  let result: Vec<Expr> =
    with_keys.into_iter().take(n).map(|(_, v)| v).collect();
  Ok(Expr::List(result))
}

/// Pick[list, sel] - pick elements where selector is True
/// Pick[list, sel, pattern] - pick elements where selector matches pattern
pub fn pick_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Pick expects 2 or 3 arguments".into(),
    ));
  }
  let list = &args[0];
  let sel = &args[1];
  let pattern = if args.len() == 3 {
    Some(&args[2])
  } else {
    None
  };

  pick_recursive(list, sel, pattern)
}

fn pick_recursive(
  list: &Expr,
  sel: &Expr,
  pattern: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  match (list, sel) {
    (
      Expr::FunctionCall {
        name,
        args: list_args,
      },
      Expr::List(sel_items),
    ) if list_args.len() == sel_items.len() => {
      let mut result = Vec::new();
      for (item, s) in list_args.iter().zip(sel_items.iter()) {
        if let (Expr::List(_), Expr::List(_)) = (item, s) {
          let picked = pick_recursive(item, s, pattern)?;
          result.push(picked);
        } else if let (Expr::FunctionCall { .. }, Expr::List(_)) = (item, s) {
          let picked = pick_recursive(item, s, pattern)?;
          result.push(picked);
        } else if matches_selector(s, pattern) {
          result.push(item.clone());
        }
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: result,
      })
    }
    (Expr::List(list_items), Expr::List(sel_items))
      if list_items.len() == sel_items.len() =>
    {
      let mut result = Vec::new();
      for (item, s) in list_items.iter().zip(sel_items.iter()) {
        if let (Expr::List(_), Expr::List(_)) = (item, s) {
          let picked = pick_recursive(item, s, pattern)?;
          result.push(picked);
        } else if let (Expr::FunctionCall { .. }, Expr::List(_)) = (item, s) {
          let picked = pick_recursive(item, s, pattern)?;
          result.push(picked);
        } else if matches_selector(s, pattern) {
          result.push(item.clone());
        }
      }
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Pick".to_string(),
      args: if let Some(p) = pattern {
        vec![list.clone(), sel.clone(), p.clone()]
      } else {
        vec![list.clone(), sel.clone()]
      },
    }),
  }
}

/// Level[expr, levelspec] - gives a list of all subexpressions at the specified levels.
/// Level[expr, levelspec, Heads -> True] - also includes heads.
///
/// Each node has a positive level (distance from root) and a negative level (-Depth[node]).
/// Level specs: n means {1,n}, {n} means exactly n, {n1,n2} means range.
/// Positive level values refer to positive level, negative values refer to negative level.
pub fn level_ast(
  expr: &Expr,
  level_spec: &Expr,
  include_heads: bool,
) -> Result<Expr, InterpreterError> {
  let (min_level, max_level) = parse_level_spec(level_spec)?;

  let mut results = Vec::new();
  level_traverse(expr, 0, min_level, max_level, include_heads, &mut results);
  Ok(Expr::List(results))
}

/// Check if a node matches the level spec.
/// pos_level: distance from root (0 = root itself)
/// neg_level: -Depth[node] (-1 for atoms, -2 for f[atom], etc.)
fn matches_level(
  pos_level: i64,
  neg_level: i64,
  min_level: i64,
  max_level: i64,
) -> bool {
  // Check min condition
  let min_ok = if min_level >= 0 {
    pos_level >= min_level
  } else {
    neg_level >= min_level
  };

  // Check max condition
  let max_ok = if max_level >= 0 {
    pos_level <= max_level
  } else {
    neg_level <= max_level
  };

  min_ok && max_ok
}

/// Parse level spec into (min, max) raw values (positive or negative).
fn parse_level_spec(spec: &Expr) -> Result<(i64, i64), InterpreterError> {
  match spec {
    Expr::List(items) if items.len() == 1 => {
      let n = level_value(&items[0])?;
      Ok((n, n))
    }
    Expr::List(items) if items.len() == 2 => {
      let n1 = level_value(&items[0])?;
      let n2 = level_value(&items[1])?;
      Ok((n1, n2))
    }
    Expr::Identifier(s) if s == "Infinity" => Ok((1, i64::MAX)),
    _ => {
      let n = level_value(spec)?;
      Ok((1, n))
    }
  }
}

fn level_value(expr: &Expr) -> Result<i64, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(*n as i64),
    Expr::Identifier(s) if s == "Infinity" => Ok(i64::MAX),
    Expr::FunctionCall { name, args } if name == "Minus" && args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        Ok(-(*n as i64))
      } else {
        Err(InterpreterError::EvaluationError(
          "Invalid level specification".into(),
        ))
      }
    }
    _ => Err(InterpreterError::EvaluationError(
      "Invalid level specification".to_string(),
    )),
  }
}

/// Get head name for a BinaryOperator
fn binary_op_head(op: &crate::syntax::BinaryOperator) -> &'static str {
  use crate::syntax::BinaryOperator;
  match op {
    BinaryOperator::Plus | BinaryOperator::Minus => "Plus",
    BinaryOperator::Times | BinaryOperator::Divide => "Times",
    BinaryOperator::Power => "Power",
    BinaryOperator::And => "And",
    BinaryOperator::Or => "Or",
    BinaryOperator::StringJoin => "StringJoin",
    BinaryOperator::Alternatives => "Alternatives",
  }
}

/// Traverse expression tree in post-order, collecting matching elements.
/// Returns the Mathematica Depth of the expression.
fn level_traverse(
  expr: &Expr,
  pos_level: i64,
  min_level: i64,
  max_level: i64,
  include_heads: bool,
  results: &mut Vec<Expr>,
) -> i64 {
  // Helper: traverse children, emit head first if applicable, return max child depth
  let traverse_compound = |head_name: &str,
                           children: &[&Expr],
                           pos_level: i64,
                           results: &mut Vec<Expr>|
   -> i64 {
    // Head symbol is an atom (depth 1, neg_level = -1)
    if include_heads && matches_level(pos_level + 1, -1, min_level, max_level) {
      results.push(Expr::Identifier(head_name.to_string()));
    }

    let mut max_child_depth: i64 = 0;
    for child in children {
      let child_depth = level_traverse(
        child,
        pos_level + 1,
        min_level,
        max_level,
        include_heads,
        results,
      );
      max_child_depth = max_child_depth.max(child_depth);
    }
    max_child_depth
  };

  match expr {
    Expr::List(items) => {
      let children: Vec<&Expr> = items.iter().collect();
      let max_child_depth =
        traverse_compound("List", &children, pos_level, results);
      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    Expr::FunctionCall { name, args, .. } => {
      let children: Vec<&Expr> = args.iter().collect();
      let max_child_depth =
        traverse_compound(name, &children, pos_level, results);
      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    Expr::BinaryOp { op, left, right } => {
      let head = binary_op_head(op);
      let children = [left.as_ref(), right.as_ref()];
      let max_child_depth =
        traverse_compound(head, &children, pos_level, results);
      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    Expr::CurriedCall { func, args } => {
      // CurriedCall: head is the func expr, children are the args
      // For Heads->True, the head (func) is traversed as a sub-expression
      if include_heads {
        // Traverse the head expression (func) for matching sub-parts
        let _head_depth = level_traverse(
          func,
          pos_level + 1,
          min_level,
          max_level,
          include_heads,
          results,
        );
      }

      // Depth of CurriedCall is based on args only (not head), matching Mathematica behavior
      let mut max_child_depth: i64 = 0;
      for arg in args {
        let child_depth = level_traverse(
          arg,
          pos_level + 1,
          min_level,
          max_level,
          include_heads,
          results,
        );
        max_child_depth = max_child_depth.max(child_depth);
      }

      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    Expr::UnaryOp { op, operand } => {
      let head = match op {
        crate::syntax::UnaryOperator::Minus => "Times",
        crate::syntax::UnaryOperator::Not => "Not",
      };
      let children = [operand.as_ref()];
      let max_child_depth =
        traverse_compound(head, &children, pos_level, results);
      let depth = 1 + max_child_depth;
      if matches_level(pos_level, -depth, min_level, max_level) {
        results.push(expr.clone());
      }
      depth
    }
    _ => {
      // Atom: depth 1, neg_level = -1
      if matches_level(pos_level, -1, min_level, max_level) {
        results.push(expr.clone());
      }
      1
    }
  }
}

fn matches_selector(sel: &Expr, pattern: Option<&Expr>) -> bool {
  match pattern {
    None => {
      matches!(sel, Expr::Identifier(s) if s == "True")
    }
    Some(pat) => {
      match crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "MatchQ".to_string(),
        args: vec![sel.clone(), pat.clone()],
      }) {
        Ok(Expr::Identifier(s)) => s == "True",
        _ => false,
      }
    }
  }
}
