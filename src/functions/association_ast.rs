//! AST-native association functions.
//!
//! These functions work directly with `Expr` AST nodes.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Whether an expression is a valid association entry: a single rule, a
/// list of rules, or an association. Used by AssociationMap to decide when
/// applying a function yields a result that can't form an association.
fn is_valid_rule_result(expr: &Expr) -> bool {
  match expr {
    Expr::Rule { .. } | Expr::RuleDelayed { .. } | Expr::Association(_) => true,
    Expr::List(items) => items
      .iter()
      .all(|e| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. })),
    _ => false,
  }
}

/// Emit `<F>::<tag>: The argument <subject> is not a valid <what>.` and
/// build the unevaluated call, matching wolframscript's message family
/// for association functions applied to invalid subjects.
fn invalid_subject_message(
  fname: &str,
  tag: &str,
  what: &str,
  subject: &Expr,
  args: &[Expr],
) -> Expr {
  crate::emit_message(&format!(
    "{}::{}: The argument {} is not a valid {}.",
    fname,
    tag,
    crate::syntax::format_expr(subject, crate::syntax::ExprForm::Output),
    what
  ));
  Expr::FunctionCall {
    name: fname.to_string(),
    args: args.to_vec().into(),
  }
}

/// Helper to extract key from a rule expression
fn extract_rule_key(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::Rule { pattern, .. } => Some(*pattern.clone()),
    Expr::RuleDelayed { pattern, .. } => Some(*pattern.clone()),
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      Some(args[0].clone())
    }
    _ => None,
  }
}

/// Helper to extract value from a rule expression
fn extract_rule_value(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::Rule { replacement, .. } => Some(*replacement.clone()),
    Expr::RuleDelayed { replacement, .. } => Some(*replacement.clone()),
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      Some(args[1].clone())
    }
    _ => None,
  }
}

/// Recursively extract keys from an expression (handles nested lists/associations)
fn keys_recursive(expr: &Expr) -> Expr {
  match expr {
    Expr::Association(items) => {
      let keys: Vec<Expr> = items.iter().map(|(k, _)| k.clone()).collect();
      Expr::List(keys.into())
    }
    Expr::List(items) => {
      let results: Vec<Expr> = items
        .iter()
        .map(|item| {
          if let Some(k) = extract_rule_key(item) {
            k
          } else {
            // Recurse into nested structures
            keys_recursive(item)
          }
        })
        .collect();
      Expr::List(results.into())
    }
    _ => expr.clone(),
  }
}

/// Recursively extract values from an expression (handles nested lists/associations)
fn values_recursive(expr: &Expr) -> Expr {
  match expr {
    Expr::Association(items) => {
      let values: Vec<Expr> = items.iter().map(|(_, v)| v.clone()).collect();
      Expr::List(values.into())
    }
    Expr::List(items) => {
      let results: Vec<Expr> = items
        .iter()
        .map(|item| {
          if let Some(v) = extract_rule_value(item) {
            v
          } else {
            // Recurse into nested structures
            values_recursive(item)
          }
        })
        .collect();
      Expr::List(results.into())
    }
    _ => expr.clone(),
  }
}

/// Keys[assoc] - Returns a list of keys from an association or list of rules
/// Apply an optional head `h` to a single extracted key/value leaf, wrapping it
/// as `h[leaf]` and evaluating. With no head the leaf is returned unchanged.
fn apply_optional_head(
  head: Option<&Expr>,
  leaf: Expr,
) -> Result<Expr, InterpreterError> {
  match head {
    None => Ok(leaf),
    Some(Expr::Identifier(name)) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: name.clone(),
        args: vec![leaf].into(),
      })
    }
    Some(h) => crate::evaluator::evaluate_expr_to_expr(&Expr::CurriedCall {
      func: Box::new(h.clone()),
      args: vec![leaf],
    }),
  }
}

/// Wrap every leaf of a (possibly nested) extraction result with `head`,
/// preserving the list nesting produced by `keys_recursive`/`values_recursive`.
fn wrap_leaves(
  expr: Expr,
  head: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  if head.is_none() {
    return Ok(expr);
  }
  if let Expr::List(items) = &expr {
    let mut out = Vec::with_capacity(items.len());
    for item in items.iter() {
      out.push(wrap_leaves(item.clone(), head)?);
    }
    return Ok(Expr::List(out.into()));
  }
  apply_optional_head(head, expr)
}

pub fn keys_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Keys expects 1 or 2 arguments".into(),
    ));
  }
  // Optional second argument: a head wrapped around each key before evaluation.
  let head = args.get(1);
  match &args[0] {
    Expr::Association(items) => {
      let mut keys = Vec::with_capacity(items.len());
      for (k, _) in items.iter() {
        keys.push(apply_optional_head(head, k.clone())?);
      }
      Ok(Expr::List(keys.into()))
    }
    Expr::List(_) => wrap_leaves(keys_recursive(&args[0]), head),
    // Keys[k -> v] = k, Keys[k :> v] = k
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
      apply_optional_head(head, pattern.as_ref().clone())
    }
    Expr::FunctionCall { name, args: rargs }
      if (name == "Rule" || name == "RuleDelayed") && rargs.len() == 2 =>
    {
      apply_optional_head(head, rargs[0].clone())
    }
    _ => Ok(invalid_subject_message(
      "Keys",
      "invrl",
      "Association or a list of rules",
      &args[0],
      args,
    )),
  }
}

/// Values[assoc] - Returns a list of values from an association or list of rules
pub fn values_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Values expects 1 or 2 arguments".into(),
    ));
  }
  // Optional second argument: a head wrapped around each value before evaluation.
  let head = args.get(1);
  match &args[0] {
    Expr::Association(items) => {
      let mut values = Vec::with_capacity(items.len());
      for (_, v) in items.iter() {
        values.push(apply_optional_head(head, v.clone())?);
      }
      Ok(Expr::List(values.into()))
    }
    Expr::List(_) => wrap_leaves(values_recursive(&args[0]), head),
    Expr::Rule { replacement, .. } | Expr::RuleDelayed { replacement, .. } => {
      apply_optional_head(head, replacement.as_ref().clone())
    }
    Expr::FunctionCall { name, args: rargs }
      if (name == "Rule" || name == "RuleDelayed") && rargs.len() == 2 =>
    {
      apply_optional_head(head, rargs[1].clone())
    }
    _ => Ok(invalid_subject_message(
      "Values",
      "invrl",
      "Association or a list of rules",
      &args[0],
      args,
    )),
  }
}

/// KeyDropFrom[assoc, key] - Returns a new association with the specified key dropped
pub fn key_drop_from_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KeyDropFrom expects exactly 2 arguments".into(),
    ));
  }
  let key_str = crate::syntax::expr_to_string(&args[1]);
  let key_cmp = key_str.trim_matches('"');

  match &args[0] {
    Expr::Association(items) => {
      let filtered: Vec<(Expr, Expr)> = items
        .iter()
        .filter(|(k, _)| {
          let k_str = crate::syntax::expr_to_string(k);
          let k_cmp = k_str.trim_matches('"');
          k_cmp != key_cmp
        })
        .cloned()
        .collect();
      Ok(Expr::Association(filtered))
    }
    Expr::Identifier(sym) => {
      crate::emit_message(&format!(
        "KeyDropFrom::blnoval: The symbol {} at position 1 should have an immediate value defined.",
        sym
      ));
      Ok(Expr::FunctionCall {
        name: "KeyDropFrom".to_string(),
        args: args.to_vec().into(),
      })
    }
    other => {
      crate::emit_message(&format!(
        "KeyDropFrom::rvalue: {} is not a variable with a value, so its value cannot be changed.",
        crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
      ));
      Ok(Expr::FunctionCall {
        name: "KeyDropFrom".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// KeyExistsQ[assoc, key] - Returns True if key is present
pub fn key_exists_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KeyExistsQ expects exactly 2 arguments".into(),
    ));
  }
  let key_str = crate::syntax::expr_to_string(&args[1]);
  let key_cmp = key_str.trim_matches('"');

  match &args[0] {
    Expr::Association(items) => {
      for (k, _) in items {
        let k_str = crate::syntax::expr_to_string(k);
        let k_cmp = k_str.trim_matches('"');
        if k_cmp == key_cmp {
          return Ok(Expr::Identifier("True".to_string()));
        }
      }
      Ok(Expr::Identifier("False".to_string()))
    }
    // A list of rules is also a valid subject.
    Expr::List(items)
      if items.iter().all(|e| extract_rule_key(e).is_some()) =>
    {
      for item in items.iter() {
        if let Some(k) = extract_rule_key(item) {
          let k_str = crate::syntax::expr_to_string(&k);
          if k_str.trim_matches('"') == key_cmp {
            return Ok(Expr::Identifier("True".to_string()));
          }
        }
      }
      Ok(Expr::Identifier("False".to_string()))
    }
    other => {
      // Unlike its siblings, KeyExistsQ answers False after the message.
      invalid_subject_message(
        "KeyExistsQ",
        "invrl",
        "Association or a list of rules",
        other,
        args,
      );
      Ok(Expr::Identifier("False".to_string()))
    }
  }
}

/// Lookup[assoc, key] - Returns the value for a key or Missing["KeyAbsent", key]
pub fn lookup_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Lookup expects at least 2 arguments".into(),
    ));
  }

  // If the second argument is a list of keys, look up each one
  if let Expr::List(keys) = &args[1]
    && let Expr::Association(_) = &args[0]
  {
    let results: Result<Vec<Expr>, InterpreterError> = keys
      .iter()
      .map(|key| {
        let mut new_args = vec![args[0].clone(), key.clone()];
        if args.len() >= 3 {
          new_args.push(args[2].clone());
        }
        lookup_ast(&new_args)
      })
      .collect();
    return Ok(Expr::List(results?.into()));
  }

  let key_str = crate::syntax::expr_to_string(&args[1]);
  let key_cmp = key_str.trim_matches('"');

  match &args[0] {
    Expr::Association(items) => {
      for (k, v) in items {
        let k_str = crate::syntax::expr_to_string(k);
        let k_cmp = k_str.trim_matches('"');
        if k_cmp == key_cmp {
          return Ok(v.clone());
        }
      }
      // Default value if provided
      if args.len() >= 3 {
        return Ok(args[2].clone());
      }
      // Return Missing["KeyAbsent", key]
      Ok(Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![Expr::String("KeyAbsent".to_string()), args[1].clone()]
          .into(),
      })
    }
    Expr::List(items) => {
      // Thread Lookup over a list of associations
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|item| {
          let mut new_args = vec![item.clone()];
          new_args.extend(args[1..].iter().cloned());
          lookup_ast(&new_args)
        })
        .collect();
      Ok(Expr::List(results?.into()))
    }
    _ => Ok(invalid_subject_message(
      "Lookup",
      "invrl",
      "Association or a list of rules",
      &args[0],
      args,
    )),
  }
}

/// AssociateTo[symbol, rule] - Adds a key-value pair to an association (in-place)
pub fn associate_to_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "AssociateTo expects exactly 2 arguments".into(),
    ));
  }

  // First arg should be identifier (variable name)
  let var_name = match &args[0] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "AssociateTo first argument must be a symbol".into(),
      ));
    }
  };

  // Second arg should be a rule
  let (key, val) = match &args[1] {
    Expr::Rule {
      pattern,
      replacement,
    } => (pattern.as_ref().clone(), replacement.as_ref().clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "AssociateTo second argument must be a rule".into(),
      ));
    }
  };

  // Get the existing association from the environment
  let stored = crate::ENV.with(|e| e.borrow().get(&var_name).cloned());

  let mut items = match stored {
    Some(crate::StoredValue::Association(pairs)) => pairs
      .into_iter()
      .map(|(k, v)| {
        let key_expr =
          crate::syntax::string_to_expr(&k).unwrap_or(Expr::String(k));
        let val_expr =
          crate::syntax::string_to_expr(&v).unwrap_or(Expr::String(v));
        (key_expr, val_expr)
      })
      .collect::<Vec<_>>(),
    Some(crate::StoredValue::ExprVal(mut expr)) => {
      if let Expr::Association(ref mut items) = expr {
        std::mem::take(items)
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{} is not defined",
          var_name
        )));
      }
    }
    Some(crate::StoredValue::Raw(s)) => {
      match crate::syntax::string_to_expr(&s) {
        Ok(mut expr) => {
          if let Expr::Association(ref mut items) = expr {
            std::mem::take(items)
          } else {
            return Err(InterpreterError::EvaluationError(format!(
              "{} is not an association",
              var_name
            )));
          }
        }
        Err(_) => {
          return Err(InterpreterError::EvaluationError(format!(
            "{} is not an association",
            var_name
          )));
        }
      }
    }
    None => {
      return Err(InterpreterError::EvaluationError(format!(
        "{} is not defined",
        var_name
      )));
    }
  };

  // Update or add the key
  let key_str = crate::syntax::expr_to_string(&key);
  let key_cmp = key_str.trim_matches('"');
  let mut found = false;
  for (k, v) in &mut items {
    let k_str = crate::syntax::expr_to_string(k);
    let k_cmp = k_str.trim_matches('"');
    if k_cmp == key_cmp {
      *v = val.clone();
      found = true;
      break;
    }
  }
  if !found {
    items.push((key, val));
  }

  // Store back to environment
  let pairs: Vec<(String, String)> = items
    .iter()
    .map(|(k, v)| {
      (
        crate::syntax::expr_to_string(k),
        crate::syntax::expr_to_string(v),
      )
    })
    .collect();
  crate::ENV.with(|e| {
    e.borrow_mut()
      .insert(var_name, crate::StoredValue::Association(pairs));
  });

  Ok(Expr::Association(items))
}

/// KeySort[assoc] - Sorts an association by its keys
pub fn key_sort_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "KeySort expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Association(items) => {
      let mut sorted = items.clone();
      sorted.sort_by(|a, b| {
        let ka = crate::syntax::expr_to_string(&a.0);
        let kb = crate::syntax::expr_to_string(&b.0);
        if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
          na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
        } else {
          ka.cmp(&kb)
        }
      });
      Ok(Expr::Association(sorted))
    }
    _ => Ok(invalid_subject_message(
      "KeySort",
      "invrl",
      "Association or a list of rules",
      &args[0],
      args,
    )),
  }
}

/// AssociationMap[f, assoc] - Maps f over each rule in the association
/// AssociationMap[f, <|a -> 1, b -> 2|>] applies f to each Rule
pub fn association_map_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "AssociationMap expects exactly 2 arguments".into(),
    ));
  }

  let func = &args[0];

  match &args[1] {
    Expr::Association(items) => {
      // Apply f once to each key -> value rule.
      let mut applied: Vec<(Expr, Expr)> = Vec::new(); // (rule, f[rule])
      for (key, value) in items {
        let rule = Expr::Rule {
          pattern: Box::new(key.clone()),
          replacement: Box::new(value.clone()),
        };
        let result = crate::evaluator::apply_function_to_arg(func, &rule)?;
        applied.push((rule, result));
      }
      // If every result is a plain rule, build the association directly.
      if applied
        .iter()
        .all(|(_, r)| matches!(r, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
      {
        let new_items: Vec<(Expr, Expr)> = applied
          .iter()
          .map(|(_, r)| match r {
            Expr::Rule {
              pattern,
              replacement,
            }
            | Expr::RuleDelayed {
              pattern,
              replacement,
            } => ((**pattern).clone(), (**replacement).clone()),
            _ => unreachable!(),
          })
          .collect();
        return Ok(Expr::Association(new_items));
      }
      // Otherwise the results don't form a valid association: emit
      // AssociationMap::invrlf for each invalid result (matching
      // wolframscript) and keep the unevaluated `Association[f[…], …]`.
      let func_str =
        crate::syntax::format_expr(func, crate::syntax::ExprForm::Output);
      for (rule, result) in &applied {
        if !is_valid_rule_result(result) {
          crate::emit_message(&format!(
            "AssociationMap::invrlf: Applying {} to {} yields {}, which is not a valid rule, list of rules or association.",
            func_str,
            crate::syntax::format_expr(rule, crate::syntax::ExprForm::Output),
            crate::syntax::format_expr(result, crate::syntax::ExprForm::Output),
          ));
        }
      }
      Ok(Expr::FunctionCall {
        name: "Association".to_string(),
        args: applied
          .into_iter()
          .map(|(_, r)| r)
          .collect::<Vec<_>>()
          .into(),
      })
    }
    Expr::List(items) => {
      // AssociationMap[f, {e1, e2, ...}] creates <|e1 -> f[e1], e2 -> f[e2], ...|>
      let mut new_items = Vec::new();
      for item in items {
        let result = crate::evaluator::apply_function_to_arg(func, item)?;
        new_items.push((item.clone(), result));
      }
      Ok(Expr::Association(new_items))
    }
    _ => Ok(invalid_subject_message(
      "AssociationMap",
      "invrp",
      "Association or a list",
      &args[1],
      args,
    )),
  }
}

/// AssociationThread[keys, values] - Creates an association from lists of keys and values
/// AssociationThread[keys -> values] - Rule form
pub fn association_thread_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (keys_expr, values_expr) = if args.len() == 2 {
    (&args[0], &args[1])
  } else if args.len() == 1 {
    // Rule form: AssociationThread[{keys} -> {values}]
    match &args[0] {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "AssociationThread expects a rule or two arguments".into(),
        ));
      }
    }
  } else {
    return Err(InterpreterError::EvaluationError(
      "AssociationThread expects 1 or 2 arguments".into(),
    ));
  };

  let keys = match keys_expr {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "AssociationThread: keys must be a list".into(),
      ));
    }
  };
  let values = match values_expr {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "AssociationThread: values must be a list".into(),
      ));
    }
  };

  if keys.len() != values.len() {
    return Err(InterpreterError::EvaluationError(
      "AssociationThread: lists must have the same length".into(),
    ));
  }

  let items: Vec<(Expr, Expr)> = keys.into_iter().zip(values).collect();
  Ok(Expr::Association(items))
}

/// Merge[{assoc1, assoc2, ...}, f] - Merges associations, applying f to conflicting values
pub fn merge_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Merge expects exactly 2 arguments".into(),
    ));
  }

  let assocs = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(invalid_subject_message(
        "Merge",
        "list1",
        "list of Associations or rules or lists of rules",
        &args[0],
        args,
      ));
    }
  };
  let func = &args[1];

  // Collect all key-value pairs, grouping values by key
  let mut key_values: Vec<(String, Expr, Vec<Expr>)> = Vec::new(); // (key_str, key_expr, values)

  // Each element may be an Association, a single rule (`k -> v`), or a list
  // of rules. Collect the (key, value) pairs from all of them, preserving
  // first-seen key order. wolframscript accepts all three shapes.
  let push_pair = |key_values: &mut Vec<(String, Expr, Vec<Expr>)>,
                   k: &Expr,
                   v: &Expr| {
    let k_str = crate::syntax::expr_to_string(k);
    if let Some(entry) = key_values.iter_mut().find(|(ks, _, _)| *ks == k_str) {
      entry.2.push(v.clone());
    } else {
      key_values.push((k_str, k.clone(), vec![v.clone()]));
    }
  };

  fn rule_parts(item: &Expr) -> Option<(&Expr, &Expr)> {
    match item {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => Some((pattern, replacement)),
      _ => None,
    }
  }

  for item in &assocs {
    match item {
      Expr::Association(items) => {
        for (k, v) in items {
          push_pair(&mut key_values, k, v);
        }
      }
      _ if rule_parts(item).is_some() => {
        let (k, v) = rule_parts(item).unwrap();
        push_pair(&mut key_values, k, v);
      }
      Expr::List(items) if items.iter().all(|i| rule_parts(i).is_some()) => {
        for i in items.iter() {
          let (k, v) = rule_parts(i).unwrap();
          push_pair(&mut key_values, k, v);
        }
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Merge: all elements must be associations".into(),
        ));
      }
    }
  }

  // Apply the merge function to each group of values
  let mut result = Vec::new();
  for (_, key_expr, values) in key_values {
    let merged_value = crate::evaluator::apply_function_to_arg(
      func,
      &Expr::List(values.into()),
    )?;
    result.push((key_expr, merged_value));
  }

  Ok(Expr::Association(result))
}

/// KeyMap[f, assoc] - Maps f over the keys of an association
pub fn key_map_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KeyMap expects exactly 2 arguments".into(),
    ));
  }

  let func = &args[0];

  match &args[1] {
    Expr::Association(items) => {
      let results: Result<Vec<(Expr, Expr)>, InterpreterError> = items
        .iter()
        .map(|(k, v)| {
          let new_key = crate::evaluator::apply_function_to_arg(func, k)?;
          Ok((new_key, v.clone()))
        })
        .collect();
      Ok(Expr::Association(results?))
    }
    _ => Ok(invalid_subject_message(
      "KeyMap",
      "invak",
      "Association",
      &args[1],
      args,
    )),
  }
}

/// KeySelect[assoc, f] - Selects entries where f[key] returns True
pub fn key_select_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KeySelect expects exactly 2 arguments".into(),
    ));
  }

  let func = &args[1];

  match &args[0] {
    Expr::Association(items) => {
      let mut result = Vec::new();
      for (k, v) in items {
        let test = crate::evaluator::apply_function_to_arg(func, k)?;
        if matches!(&test, Expr::Identifier(s) if s == "True") {
          result.push((k.clone(), v.clone()));
        }
      }
      Ok(Expr::Association(result))
    }
    _ => Ok(invalid_subject_message(
      "KeySelect",
      "invru",
      "Association or a list of rules",
      &args[0],
      args,
    )),
  }
}

/// KeyTake[assoc, keys] - Takes only the specified keys
pub fn key_take_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KeyTake expects exactly 2 arguments".into(),
    ));
  }

  let keep_keys: Vec<String> = match &args[1] {
    Expr::List(items) => items
      .iter()
      .map(|k| {
        let s = crate::syntax::expr_to_string(k);
        s.trim_matches('"').to_string()
      })
      .collect(),
    other => vec![
      crate::syntax::expr_to_string(other)
        .trim_matches('"')
        .to_string(),
    ],
  };

  match &args[0] {
    Expr::Association(items) => {
      let mut result = Vec::new();
      for desired_key in &keep_keys {
        for (k, v) in items {
          let k_str = crate::syntax::expr_to_string(k);
          let k_cmp = k_str.trim_matches('"');
          if k_cmp == desired_key {
            result.push((k.clone(), v.clone()));
            break;
          }
        }
      }
      Ok(Expr::Association(result))
    }
    _ => Ok(invalid_subject_message(
      "KeyTake",
      "invrl",
      "Association or a list of rules",
      &args[0],
      args,
    )),
  }
}

/// KeyDrop[assoc, keys] - Drops the specified keys
pub fn key_drop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KeyDrop expects exactly 2 arguments".into(),
    ));
  }

  let drop_keys: Vec<String> = match &args[1] {
    Expr::List(items) => items
      .iter()
      .map(|k| {
        let s = crate::syntax::expr_to_string(k);
        s.trim_matches('"').to_string()
      })
      .collect(),
    other => vec![
      crate::syntax::expr_to_string(other)
        .trim_matches('"')
        .to_string(),
    ],
  };

  match &args[0] {
    Expr::Association(items) => {
      let result: Vec<(Expr, Expr)> = items
        .iter()
        .filter(|(k, _)| {
          let k_str = crate::syntax::expr_to_string(k);
          let k_cmp = k_str.trim_matches('"');
          !drop_keys.iter().any(|dk| dk == k_cmp)
        })
        .cloned()
        .collect();
      Ok(Expr::Association(result))
    }
    _ => Ok(invalid_subject_message(
      "KeyDrop",
      "invrl",
      "Association or a list of rules",
      &args[0],
      args,
    )),
  }
}

/// KeyValueMap[f, assoc] - Maps a function over key-value pairs, returning a list
/// KeyValueMap[f, <|a -> 1, b -> 2|>] -> {f[a, 1], f[b, 2]}
pub fn key_value_map_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "KeyValueMap expects exactly 2 arguments".into(),
    ));
  }

  let func = &args[0];

  match &args[1] {
    Expr::Association(items) => {
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|(key, value)| match func {
          Expr::Identifier(name) if name == "List" => {
            Ok(Expr::List(vec![key.clone(), value.clone()].into()))
          }
          Expr::Identifier(name) => {
            crate::evaluator::evaluate_function_call_ast(
              name,
              &[key.clone(), value.clone()],
            )
          }
          Expr::Function { body } => {
            let substituted = crate::syntax::substitute_slots(
              body,
              &[key.clone(), value.clone()],
            );
            crate::evaluator::evaluate_expr_to_expr(&substituted)
          }
          Expr::NamedFunction { params, body, .. } => {
            let mut substituted = (**body).clone();
            let args_vec = [&key, &value];
            for (param, arg) in params.iter().zip(args_vec.iter()) {
              substituted =
                crate::syntax::substitute_variable(&substituted, param, arg);
            }
            crate::evaluator::evaluate_expr_to_expr(&substituted)
          }
          Expr::FunctionCall { name, args: fargs } => {
            let mut new_args = fargs.clone();
            new_args.push(key.clone());
            new_args.push(value.clone());
            crate::evaluator::evaluate_function_call_ast(name, &new_args)
          }
          _ => Ok(Expr::FunctionCall {
            name: "KeyValueMap".to_string(),
            args: vec![func.clone(), Expr::Association(items.clone())].into(),
          }),
        })
        .collect();
      Ok(Expr::List(results?.into()))
    }
    _ => Ok(invalid_subject_message(
      "KeyValueMap",
      "invak",
      "Association",
      &args[1],
      args,
    )),
  }
}

// ─── KeyUnion ───────────────────────────────────────────────────────

/// KeyUnion[{assoc1, assoc2, ...}] - extends each association with Missing values
/// for keys that appear in other associations but not in that one.
/// KeyUnion[{assoc1, assoc2, ...}, f] - uses f[key] instead of Missing[KeyAbsent, key].
pub fn key_union_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "KeyUnion expects 1 or 2 arguments".into(),
    ));
  }

  let assocs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(invalid_subject_message(
        "KeyUnion",
        "invar",
        "list of Associations or rules",
        &args[0],
        args,
      ));
    }
  };

  let default_fn = if args.len() >= 2 {
    Some(&args[1])
  } else {
    None
  };

  // Extract all associations as Vec<Vec<(key, value)>>
  let mut all_assocs: Vec<Vec<(Expr, Expr)>> = Vec::new();
  for assoc in assocs {
    match assoc {
      Expr::Association(items) => {
        all_assocs.push(items.clone());
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "KeyUnion: each element must be an association".into(),
        ));
      }
    }
  }

  // Collect all unique keys in order of first appearance
  let mut all_keys: Vec<Expr> = Vec::new();
  let mut seen_keys: Vec<String> = Vec::new();
  for assoc in &all_assocs {
    for (key, _) in assoc {
      let key_str = crate::syntax::expr_to_string(key);
      if !seen_keys.contains(&key_str) {
        seen_keys.push(key_str);
        all_keys.push(key.clone());
      }
    }
  }

  // Build result: for each association, include all keys
  let mut result = Vec::new();
  for assoc in &all_assocs {
    let mut new_items: Vec<(Expr, Expr)> = Vec::new();
    for key in &all_keys {
      let key_str = crate::syntax::expr_to_string(key);
      let value = assoc
        .iter()
        .find(|(k, _)| crate::syntax::expr_to_string(k) == key_str)
        .map(|(_, v)| v.clone());
      match value {
        Some(v) => new_items.push((key.clone(), v)),
        None => {
          let missing = match default_fn {
            Some(func) => {
              let call = Expr::FunctionCall {
                name: crate::syntax::expr_to_string(func),
                args: vec![key.clone()].into(),
              };
              crate::evaluator::evaluate_expr_to_expr(&call)?
            }
            None => Expr::FunctionCall {
              name: "Missing".to_string(),
              args: vec![Expr::String("KeyAbsent".to_string()), key.clone()]
                .into(),
            },
          };
          new_items.push((key.clone(), missing));
        }
      }
    }
    result.push(Expr::Association(new_items));
  }

  Ok(Expr::List(result.into()))
}

/// Extract the inner associations from a `KeyComplement`/`KeyIntersection`
/// argument: a single list of associations. Returns the `::invar` message form
/// when the argument is not a list of associations.
fn key_set_op_assocs(
  fname: &str,
  args: &[Expr],
) -> Result<Vec<Vec<(Expr, Expr)>>, Expr> {
  let assocs = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Err(invalid_subject_message(
        fname,
        "invar",
        "list of Associations or rules",
        &args[0],
        args,
      ));
    }
  };
  let mut all_assocs: Vec<Vec<(Expr, Expr)>> = Vec::new();
  for assoc in assocs {
    match assoc {
      Expr::Association(items) => all_assocs.push(items.clone()),
      _ => {
        return Err(invalid_subject_message(
          fname,
          "invar",
          "list of Associations or rules",
          &args[0],
          args,
        ));
      }
    }
  }
  Ok(all_assocs)
}

/// KeyIntersection[{assoc1, assoc2, ...}] restricts every association to the
/// keys common to all of them. The common keys are ordered as they appear in
/// the first association, and each result keeps that association's own values.
pub fn key_intersection_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "KeyIntersection expects 1 argument".into(),
    ));
  }
  let all_assocs = match key_set_op_assocs("KeyIntersection", args) {
    Ok(a) => a,
    Err(e) => return Ok(e),
  };
  if all_assocs.is_empty() {
    return Ok(Expr::List(vec![].into()));
  }

  // Common keys, ordered by the first association.
  let common_keys: Vec<Expr> = all_assocs[0]
    .iter()
    .map(|(k, _)| k.clone())
    .filter(|k| {
      let ks = crate::syntax::expr_to_string(k);
      all_assocs[1..].iter().all(|a| {
        a.iter()
          .any(|(k2, _)| crate::syntax::expr_to_string(k2) == ks)
      })
    })
    .collect();

  let result: Vec<Expr> = all_assocs
    .iter()
    .map(|assoc| {
      let items: Vec<(Expr, Expr)> = common_keys
        .iter()
        .map(|k| {
          let ks = crate::syntax::expr_to_string(k);
          let v = assoc
            .iter()
            .find(|(k2, _)| crate::syntax::expr_to_string(k2) == ks)
            .map(|(_, v)| v.clone())
            .unwrap();
          (k.clone(), v)
        })
        .collect();
      Expr::Association(items)
    })
    .collect();

  Ok(Expr::List(result.into()))
}

/// KeyComplement[{assoc1, assoc2, ...}] returns the first association
/// restricted to the keys that appear in none of the others, keeping the first
/// association's order and values.
pub fn key_complement_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "KeyComplement expects 1 argument".into(),
    ));
  }
  let all_assocs = match key_set_op_assocs("KeyComplement", args) {
    Ok(a) => a,
    Err(e) => return Ok(e),
  };
  if all_assocs.is_empty() {
    return Ok(Expr::Association(vec![]));
  }

  let other_keys: Vec<String> = all_assocs[1..]
    .iter()
    .flat_map(|a| a.iter().map(|(k, _)| crate::syntax::expr_to_string(k)))
    .collect();

  let items: Vec<(Expr, Expr)> = all_assocs[0]
    .iter()
    .filter(|(k, _)| !other_keys.contains(&crate::syntax::expr_to_string(k)))
    .cloned()
    .collect();

  Ok(Expr::Association(items))
}
