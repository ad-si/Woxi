//! AST-native association functions.
//!
//! These functions work directly with `Expr` AST nodes.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Keys[assoc] - Returns a list of keys from an association
pub fn keys_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Keys expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Association(items) => {
      let keys: Vec<Expr> = items.iter().map(|(k, _)| k.clone()).collect();
      Ok(Expr::List(keys))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Keys expects an association".into(),
    )),
  }
}

/// Values[assoc] - Returns a list of values from an association
pub fn values_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Values expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Association(items) => {
      let values: Vec<Expr> = items.iter().map(|(_, v)| v.clone()).collect();
      Ok(Expr::List(values))
    }
    _ => Err(InterpreterError::EvaluationError(
      "Values expects an association".into(),
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
    _ => Err(InterpreterError::EvaluationError(
      "KeyDropFrom expects an association as first argument".into(),
    )),
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
    _ => Err(InterpreterError::EvaluationError(
      "KeyExistsQ expects an association as first argument".into(),
    )),
  }
}

/// Lookup[assoc, key] - Returns the value for a key or Missing["KeyAbsent", key]
pub fn lookup_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Lookup expects at least 2 arguments".into(),
    ));
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
        args: vec![Expr::String("KeyAbsent".to_string()), args[1].clone()],
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "Lookup expects an association as first argument".into(),
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
    Some(crate::StoredValue::ExprVal(Expr::Association(items))) => items,
    Some(crate::StoredValue::Raw(s)) => {
      if let Ok(Expr::Association(items)) = crate::syntax::string_to_expr(&s) {
        items
      } else {
        return Err(InterpreterError::EvaluationError(format!(
          "{} is not an association",
          var_name
        )));
      }
    }
    None | Some(crate::StoredValue::ExprVal(_)) => {
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
    _ => Err(InterpreterError::EvaluationError(
      "KeySort expects an association".into(),
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
      let mut new_items = Vec::new();
      for (key, value) in items {
        let rule = Expr::Rule {
          pattern: Box::new(key.clone()),
          replacement: Box::new(value.clone()),
        };
        let result = crate::evaluator::apply_function_to_arg(func, &rule)?;
        match result {
          Expr::Rule {
            pattern,
            replacement,
          } => {
            new_items.push((*pattern, *replacement));
          }
          _ => {
            // When f doesn't produce rules, collect all results as Association args
            // Wolfram returns Association[f[...], f[...]] (not a plain List)
            let results: Result<Vec<Expr>, InterpreterError> = items
              .iter()
              .map(|(k, v)| {
                let r = Expr::Rule {
                  pattern: Box::new(k.clone()),
                  replacement: Box::new(v.clone()),
                };
                crate::evaluator::apply_function_to_arg(func, &r)
              })
              .collect();
            return Ok(Expr::FunctionCall {
              name: "Association".to_string(),
              args: results?,
            });
          }
        }
      }
      Ok(Expr::Association(new_items))
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
    _ => Err(InterpreterError::EvaluationError(
      "AssociationMap expects an association or list as second argument".into(),
    )),
  }
}

/// AssociationThread[keys, values] - Creates an association from lists of keys and values
pub fn association_thread_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "AssociationThread expects exactly 2 arguments".into(),
    ));
  }

  let keys = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "AssociationThread: first argument must be a list".into(),
      ));
    }
  };
  let values = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "AssociationThread: second argument must be a list".into(),
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
      return Err(InterpreterError::EvaluationError(
        "Merge: first argument must be a list of associations".into(),
      ));
    }
  };
  let func = &args[1];

  // Collect all key-value pairs, grouping values by key
  let mut key_values: Vec<(String, Expr, Vec<Expr>)> = Vec::new(); // (key_str, key_expr, values)

  for assoc in &assocs {
    if let Expr::Association(items) = assoc {
      for (k, v) in items {
        let k_str = crate::syntax::expr_to_string(k);
        if let Some(entry) =
          key_values.iter_mut().find(|(ks, _, _)| *ks == k_str)
        {
          entry.2.push(v.clone());
        } else {
          key_values.push((k_str, k.clone(), vec![v.clone()]));
        }
      }
    } else {
      return Err(InterpreterError::EvaluationError(
        "Merge: all elements must be associations".into(),
      ));
    }
  }

  // Apply the merge function to each group of values
  let mut result = Vec::new();
  for (_, key_expr, values) in key_values {
    let merged_value =
      crate::evaluator::apply_function_to_arg(func, &Expr::List(values))?;
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
    _ => Err(InterpreterError::EvaluationError(
      "KeyMap expects an association as second argument".into(),
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
    _ => Err(InterpreterError::EvaluationError(
      "KeySelect expects an association as first argument".into(),
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
    _ => Err(InterpreterError::EvaluationError(
      "KeyTake expects an association as first argument".into(),
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
    _ => Err(InterpreterError::EvaluationError(
      "KeyDrop expects an association as first argument".into(),
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
            Ok(Expr::List(vec![key.clone(), value.clone()]))
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
          Expr::NamedFunction { params, body } => {
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
            args: vec![func.clone(), Expr::Association(items.clone())],
          }),
        })
        .collect();
      Ok(Expr::List(results?))
    }
    _ => Err(InterpreterError::EvaluationError(
      "KeyValueMap expects an association as second argument".into(),
    )),
  }
}
