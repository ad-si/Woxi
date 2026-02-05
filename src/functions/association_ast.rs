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
