#[allow(unused_imports)]
use super::*;

/// Helper for Attributes[f] = value / Attributes[f] := value
/// Extracts attribute symbols from value, validates, and sets them on the symbol.
pub fn set_attributes_from_value(
  sym_name: &str,
  rhs_value: &Expr,
) -> Result<Expr, InterpreterError> {
  // Check if symbol is locked
  let is_locked = crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(sym_name)
      .is_some_and(|attrs| attrs.contains(&"Locked".to_string()))
  });
  if is_locked {
    eprintln!("Attributes::locked: Symbol {} is locked.", sym_name);
    return Ok(rhs_value.clone());
  }

  // Extract attribute names from the value
  let attr_exprs = match rhs_value {
    Expr::List(items) => items.clone(),
    Expr::Identifier(_) => vec![rhs_value.clone()],
    _ => vec![rhs_value.clone()],
  };

  let mut valid_attrs = Vec::new();
  let mut has_error = false;
  for attr_expr in &attr_exprs {
    if let Expr::Identifier(attr_name) = attr_expr {
      valid_attrs.push(attr_name.clone());
    } else {
      // Non-symbol attribute — emit warning
      let attr_str = expr_to_string(attr_expr);
      eprintln!("Attributes::attnf: {} is not a known attribute.", attr_str);
      has_error = true;
    }
  }

  if has_error {
    return Ok(Expr::Identifier("$Failed".to_string()));
  }

  // Replace all user-defined attributes for this symbol
  crate::FUNC_ATTRS.with(|m| {
    m.borrow_mut().insert(sym_name.to_string(), valid_attrs);
  });

  Ok(rhs_value.clone())
}

/// Helper for Options[f] = value — set options for symbol f
pub fn set_options_from_value(
  sym_name: &str,
  rhs_value: &Expr,
) -> Result<Expr, InterpreterError> {
  // Extract rules from the value
  let rules = match rhs_value {
    Expr::List(items) => items.clone(),
    _ => vec![rhs_value.clone()],
  };

  crate::FUNC_OPTIONS.with(|m| {
    m.borrow_mut().insert(sym_name.to_string(), rules);
  });

  Ok(rhs_value.clone())
}

/// AST-based Set implementation to handle Part assignment on associations and lists
pub fn set_ast(lhs: &Expr, rhs: &Expr) -> Result<Expr, InterpreterError> {
  // Handle Part assignment: var[[indices]] = value
  if let Expr::Part { .. } = lhs {
    // Flatten nested Part to get base variable and list of indices
    let mut indices = Vec::new();
    let mut current = lhs;
    while let Expr::Part { expr, index } = current {
      indices.push(index.as_ref().clone());
      current = expr.as_ref();
    }
    indices.reverse();

    let var_name = match current {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment requires a variable name".into(),
        ));
      }
    };

    // Check Protected attribute
    if is_symbol_protected(&var_name) {
      let rhs_value = evaluate_expr_to_expr(rhs)?;
      eprintln!("Part::wrsym: Symbol {} is Protected.", var_name);
      return Ok(rhs_value);
    }

    // Evaluate indices
    let mut eval_indices = Vec::new();
    for idx in &indices {
      eval_indices.push(evaluate_expr_to_expr(idx)?);
    }

    // Evaluate the RHS
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Single-index association assignment: myHash[["key"]] = value
    if eval_indices.len() == 1 {
      let is_assoc = crate::ENV.with(|e| {
        let env = e.borrow();
        matches!(env.get(&var_name), Some(StoredValue::Association(_)))
      });
      if is_assoc {
        // Use expr_to_string to match the storage format (preserves string quotes)
        let key = expr_to_string(&eval_indices[0]);
        crate::ENV.with(|e| {
          let mut env = e.borrow_mut();
          if let Some(StoredValue::Association(pairs)) = env.get_mut(&var_name)
          {
            if let Some(pair) = pairs.iter_mut().find(|(k, _)| k == &key) {
              pair.1 = expr_to_string(&rhs_value);
            } else {
              pairs.push((key, expr_to_string(&rhs_value)));
            }
          }
        });
        return Ok(rhs_value);
      }
    }

    // General Part assignment: modify in-place if ExprVal, otherwise parse/modify/store
    let modified_in_place = crate::ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(StoredValue::ExprVal(expr)) = env.get_mut(&var_name) {
        // Modify directly in place — no clone needed
        set_part_deep(expr, &eval_indices, &rhs_value)
      } else {
        Err(InterpreterError::EvaluationError("not ExprVal".into()))
      }
    });
    if modified_in_place.is_ok() {
      return Ok(rhs_value);
    }

    // Fallback: parse stored string, modify, store back as ExprVal
    let stored_str = crate::ENV.with(|e| {
      let env = e.borrow();
      match env.get(&var_name) {
        Some(StoredValue::Raw(s)) => Some(s.clone()),
        _ => None,
      }
    });
    if let Some(stored_str) = stored_str {
      let mut stored_expr =
        string_to_expr(&stored_str).unwrap_or(Expr::Raw(stored_str));
      set_part_deep(&mut stored_expr, &eval_indices, &rhs_value)?;
      crate::ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name, StoredValue::ExprVal(stored_expr))
      });
      return Ok(rhs_value);
    }

    return Err(InterpreterError::EvaluationError(format!(
      "Variable {} not found",
      var_name
    )));
  }

  // Handle simple identifier assignment: x = value
  if let Expr::Identifier(var_name) = lhs {
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Check Protected attribute
    if is_symbol_protected(var_name) {
      eprintln!("Set::wrsym: Symbol {} is Protected.", var_name);
      return Ok(rhs_value);
    }

    // Check if RHS is an association
    if let Expr::Association(items) = &rhs_value {
      let pairs: Vec<(String, String)> = items
        .iter()
        .map(|(k, v)| {
          // Use expr_to_string for keys to preserve type info
          // (e.g. Expr::String("x") → "\"x\"", Expr::Identifier("x") → "x")
          (expr_to_string(k), expr_to_string(v))
        })
        .collect();
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::Association(pairs))
      });
    } else if matches!(
      &rhs_value,
      Expr::List(_)
        | Expr::FunctionCall { .. }
        | Expr::String(_)
        | Expr::Function { .. }
        | Expr::NamedFunction { .. }
        | Expr::Image { .. }
    ) {
      // Store lists, function calls, functions, and strings as ExprVal for faithful roundtrip
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::ExprVal(rhs_value.clone()))
      });
    } else {
      ENV.with(|e| {
        e.borrow_mut().insert(
          var_name.clone(),
          StoredValue::Raw(expr_to_string(&rhs_value)),
        )
      });
    }

    return Ok(rhs_value);
  }

  // Handle Options[f] = value — set options on symbol f
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Options"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    return set_options_from_value(sym_name, &rhs_value);
  }

  // Handle Attributes[f] = value — set attributes on symbol f
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Attributes"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    return set_attributes_from_value(sym_name, &rhs_value);
  }

  // Handle DownValues: f[val1, val2, ...] = rhs
  // Store as a function definition with literal-match conditions
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Check user-defined Protected attribute for DownValues
    // (builtin Protected is not checked for DownValues, matching wolframscript behavior)
    let is_user_protected = crate::FUNC_ATTRS.with(|m| {
      m.borrow()
        .get(func_name.as_str())
        .is_some_and(|attrs| attrs.contains(&"Protected".to_string()))
    });
    if is_user_protected {
      eprintln!("Set::wrsym: Symbol {} is Protected.", func_name);
      return Ok(rhs_value);
    }

    // Build param names and conditions for each argument
    let mut params = Vec::new();
    let mut conditions: Vec<Option<Expr>> = Vec::new();
    let defaults = vec![None; lhs_args.len()];
    let heads = vec![None; lhs_args.len()];

    for (i, arg) in lhs_args.iter().enumerate() {
      let param_name = format!("_dv{}", i);
      // Evaluate the literal argument value
      let eval_arg = evaluate_expr_to_expr(arg)?;
      // Condition: _dvN === eval_arg (using SameQ for exact matching)
      conditions.push(Some(Expr::Comparison {
        operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
        operators: vec![crate::syntax::ComparisonOp::SameQ],
      }));
      params.push(param_name);
    }

    crate::FUNC_DEFS.with(|m| {
      let mut defs = m.borrow_mut();
      let entry = defs.entry(func_name.clone()).or_insert_with(Vec::new);
      // Insert at the beginning so specific values take priority over general patterns
      entry.insert(0, (params, conditions, defaults, heads, rhs_value.clone()));
    });

    return Ok(rhs_value);
  }

  Err(InterpreterError::EvaluationError(
    "First argument of Set must be an identifier, part extract, or function call".into(),
  ))
}

/// Handle SetDelayed[f[patterns...], body] — stores a function definition.
/// This handles cases that the PEG FunctionDefinition rule doesn't parse,
/// such as list-pattern arguments: f[{x_Integer, y_Integer}] := body.
pub fn set_delayed_ast(
  lhs: &Expr,
  body: &Expr,
) -> Result<Expr, InterpreterError> {
  // Handle Attributes[f] := value — set attributes on symbol f
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Attributes"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    // SetDelayed still evaluates the RHS for Attributes
    let rhs_value = evaluate_expr_to_expr(body)?;
    return set_attributes_from_value(sym_name, &rhs_value);
  }

  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
  {
    // Check user-defined Protected attribute for DownValues
    let is_user_protected = crate::FUNC_ATTRS.with(|m| {
      m.borrow()
        .get(func_name.as_str())
        .is_some_and(|attrs| attrs.contains(&"Protected".to_string()))
    });
    if is_user_protected {
      eprintln!("SetDelayed::wrsym: Symbol {} is Protected.", func_name);
      return Ok(Expr::Identifier("Null".to_string()));
    }

    let mut params = Vec::new();
    let mut conditions: Vec<Option<Expr>> = Vec::new();
    let mut defaults: Vec<Option<Expr>> = Vec::new();
    let mut heads: Vec<Option<String>> = Vec::new();
    // We also need to track substitutions for list-pattern destructuring
    let mut body_substitutions: Vec<(String, Vec<(String, Option<String>)>)> =
      Vec::new();

    for (i, arg) in lhs_args.iter().enumerate() {
      match arg {
        // List pattern: {x_Integer, y_Integer} — destructure a list argument
        Expr::List(patterns) => {
          let param_name = format!("_lp{}", i);
          // Condition: argument must be a list with the right length
          conditions.push(Some(Expr::Comparison {
            operands: vec![
              Expr::FunctionCall {
                name: "Length".to_string(),
                args: vec![Expr::Identifier(param_name.clone())],
              },
              Expr::Integer(patterns.len() as i128),
            ],
            operators: vec![crate::syntax::ComparisonOp::SameQ],
          }));
          // Extract pattern names and head constraints from list elements
          let mut element_bindings = Vec::new();
          for pat in patterns {
            let (pat_name, head) = extract_pattern_info(pat);
            element_bindings.push((pat_name, head));
          }
          body_substitutions.push((param_name.clone(), element_bindings));
          params.push(param_name);
          defaults.push(None);
          heads.push(Some("List".to_string()));
        }
        // Simple pattern: x_ or x_Head
        _ => {
          let (pat_name, head) = extract_pattern_info(arg);
          if pat_name.is_empty() && head.is_none() {
            // Literal value (not a pattern) — create a SameQ condition
            // e.g., f[1] := ... should only match when arg === 1
            let param_name = format!("_dv{}", i);
            let eval_arg = evaluate_expr_to_expr(arg)?;
            conditions.push(Some(Expr::Comparison {
              operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
              operators: vec![crate::syntax::ComparisonOp::SameQ],
            }));
            params.push(param_name);
          } else {
            params.push(pat_name);
            conditions.push(None);
          }
          defaults.push(None);
          heads.push(head);
        }
      }
    }

    // Build the body with list-destructuring substitutions.
    // For each list-pattern param, replace references to element names
    // with Part[param, index] expressions.
    let mut final_body = body.clone();
    for (param_name, element_bindings) in &body_substitutions {
      for (idx, (elem_name, _head)) in element_bindings.iter().enumerate() {
        if !elem_name.is_empty() {
          // Replace elem_name with Part[param_name, idx+1]
          let part_expr = Expr::FunctionCall {
            name: "Part".to_string(),
            args: vec![
              Expr::Identifier(param_name.clone()),
              Expr::Integer((idx + 1) as i128),
            ],
          };
          final_body = crate::syntax::substitute_variable(
            &final_body,
            elem_name,
            &part_expr,
          );
        }
      }
    }

    // Check if all args are literal (non-pattern) — if so, insert at beginning
    // for priority over general patterns (matching Mathematica specificity ordering)
    let has_literal_conditions = conditions.iter().any(|c| {
      if let Some(Expr::Comparison { operators, .. }) = c {
        operators
          .iter()
          .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
      } else {
        false
      }
    });

    crate::FUNC_DEFS.with(|m| {
      let mut defs = m.borrow_mut();
      let entry = defs.entry(func_name.clone()).or_insert_with(Vec::new);
      if has_literal_conditions {
        // Literal-match definitions go first for priority
        entry.insert(0, (params, conditions, defaults, heads, final_body));
      } else {
        entry.push((params, conditions, defaults, heads, final_body));
      }
    });

    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Handle simple identifier assignment: a := expr (OwnValues)
  if let Expr::Identifier(var_name) = lhs {
    if is_symbol_protected(var_name) {
      eprintln!("SetDelayed::wrsym: Symbol {} is Protected.", var_name);
      return Ok(Expr::Identifier("Null".to_string()));
    }
    // Store the unevaluated body — it will be re-evaluated each time the symbol is accessed
    ENV.with(|e| {
      e.borrow_mut()
        .insert(var_name.clone(), StoredValue::ExprVal(body.clone()))
    });
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Fallback: return symbolic form
  Ok(Expr::FunctionCall {
    name: "SetDelayed".to_string(),
    args: vec![lhs.clone(), body.clone()],
  })
}

/// Extract a pattern name and optional head constraint from a pattern expression.
/// e.g., x_Integer -> ("x", Some("Integer")), x_ -> ("x", None)
pub fn extract_pattern_info(expr: &Expr) -> (String, Option<String>) {
  match expr {
    Expr::Identifier(name) => {
      // Could be a pattern like "x_Integer" or "x_" in text form
      if let Some(pos) = name.find('_') {
        let pat_name = name[..pos].to_string();
        let head = &name[pos + 1..];
        if head.is_empty() {
          (pat_name, None)
        } else {
          (pat_name, Some(head.to_string()))
        }
      } else {
        (name.clone(), None)
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      // Pattern[name, Blank[head]] or Pattern[name, Blank[]]
      if let Expr::Identifier(pat_name) = &args[0]
        && let Expr::FunctionCall {
          name: blank_name,
          args: blank_args,
        } = &args[1]
        && blank_name == "Blank"
      {
        let head = blank_args.first().and_then(|a| {
          if let Expr::Identifier(h) = a {
            Some(h.clone())
          } else {
            None
          }
        });
        return (pat_name.clone(), head);
      }
      (String::new(), None)
    }
    _ => {
      // Try to extract from string representation
      let s = expr_to_string(expr);
      if let Some(pos) = s.find('_') {
        let pat_name = s[..pos].to_string();
        let head = &s[pos + 1..];
        if head.is_empty() {
          (pat_name, None)
        } else {
          (pat_name, Some(head.to_string()))
        }
      } else {
        (String::new(), None)
      }
    }
  }
}

/// Handle TagSetDelayed[tag, lhs, rhs] — stores an upvalue definition.
/// When evaluate_rhs is true, acts as TagSet (evaluates the RHS first).
pub fn tag_set_delayed_ast(
  tag: &Expr,
  lhs: &Expr,
  body: &Expr,
  evaluate_rhs: bool,
) -> Result<Expr, InterpreterError> {
  let tag_name = match tag {
    Expr::Identifier(s) => s.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TagSetDelayed: first argument must be a symbol".into(),
      ));
    }
  };

  let body = if evaluate_rhs {
    evaluate_expr_to_expr(body)?
  } else {
    body.clone()
  };

  // Extract outer function name and args from the LHS
  let (outer_func, lhs_args) = match lhs {
    Expr::FunctionCall { name, args } => (name.clone(), args.clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TagSetDelayed: second argument must be a function call".into(),
      ));
    }
  };

  // Process each argument in the LHS to extract patterns
  let mut params = Vec::new();
  let mut conditions: Vec<Option<Expr>> = Vec::new();
  let mut defaults: Vec<Option<Expr>> = Vec::new();
  let mut heads: Vec<Option<String>> = Vec::new();
  let mut final_body = body.clone();

  for (i, arg) in lhs_args.iter().enumerate() {
    match arg {
      Expr::FunctionCall {
        name: arg_func_name,
        args: inner_args,
      } => {
        let param_name = format!("_up{}", i);
        heads.push(Some(arg_func_name.clone()));

        if !inner_args.is_empty() {
          conditions.push(Some(Expr::Comparison {
            operands: vec![
              Expr::FunctionCall {
                name: "Length".to_string(),
                args: vec![Expr::Identifier(param_name.clone())],
              },
              Expr::Integer(inner_args.len() as i128),
            ],
            operators: vec![crate::syntax::ComparisonOp::SameQ],
          }));
        } else {
          conditions.push(None);
        }

        for (j, inner_arg) in inner_args.iter().enumerate() {
          let (pat_name, _pat_head) = extract_pattern_info(inner_arg);
          if !pat_name.is_empty() {
            let part_expr = Expr::FunctionCall {
              name: "Part".to_string(),
              args: vec![
                Expr::Identifier(param_name.clone()),
                Expr::Integer((j + 1) as i128),
              ],
            };
            final_body = crate::syntax::substitute_variable(
              &final_body,
              &pat_name,
              &part_expr,
            );
          }
        }

        params.push(param_name);
        defaults.push(None);
      }
      _ => {
        let (pat_name, head) = extract_pattern_info(arg);
        if pat_name.is_empty() && head.is_none() {
          let param_name = format!("_up{}", i);
          let eval_arg = evaluate_expr_to_expr(arg)?;
          conditions.push(Some(Expr::Comparison {
            operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
            operators: vec![crate::syntax::ComparisonOp::SameQ],
          }));
          params.push(param_name);
        } else {
          params.push(pat_name);
          conditions.push(None);
        }
        defaults.push(None);
        heads.push(head);
      }
    }
  }

  // Store in UPVALUES for introspection and cleanup
  crate::UPVALUES.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(tag_name).or_insert_with(Vec::new);
    entry.push((
      outer_func.clone(),
      params.clone(),
      conditions.clone(),
      defaults.clone(),
      heads.clone(),
      final_body.clone(),
    ));
  });

  // Store in FUNC_DEFS under the outer function name
  crate::FUNC_DEFS.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(outer_func).or_insert_with(Vec::new);
    entry.insert(0, (params, conditions, defaults, heads, final_body));
  });

  Ok(Expr::Identifier("Null".to_string()))
}

/// UpSet[lhs, rhs] — automatically assigns upvalues to all symbols in the arguments of lhs.
/// f[g] ^= 5 stores an upvalue for g such that f[g] evaluates to 5.
pub fn upset_ast(lhs: &Expr, rhs: &Expr) -> Result<Expr, InterpreterError> {
  // LHS must be a function call
  let (_, lhs_args) = match lhs {
    Expr::FunctionCall { name, args } => (name.clone(), args.clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "UpSet::normal: Nonatomic expression expected at position 1 in {} ^= {}",
        crate::syntax::expr_to_string(lhs),
        crate::syntax::expr_to_string(rhs)
      )));
    }
  };

  // Evaluate the RHS
  let eval_rhs = evaluate_expr_to_expr(rhs)?;

  // Find all tag symbols in the arguments
  let mut tags = Vec::new();
  for arg in &lhs_args {
    match arg {
      Expr::Identifier(s) => tags.push(s.clone()),
      Expr::FunctionCall { name, .. } => tags.push(name.clone()),
      _ => {} // Skip non-symbol arguments (integers, etc.)
    }
  }

  if tags.is_empty() {
    return Err(InterpreterError::EvaluationError(format!(
      "UpSet::nosym: {} does not contain a symbol to attach a rule to.",
      crate::syntax::expr_to_string(lhs)
    )));
  }

  // Store upvalue for each tag
  for tag in &tags {
    tag_set_delayed_ast(&Expr::Identifier(tag.clone()), lhs, &eval_rhs, true)?;
  }

  // UpSet returns the evaluated RHS
  Ok(eval_rhs)
}

/// UpSetDelayed[lhs, rhs] — like UpSet but with delayed evaluation (RHS not evaluated).
/// f[g] ^:= body stores a delayed upvalue for g such that f[g] evaluates body each time.
pub fn upset_delayed_ast(
  lhs: &Expr,
  rhs: &Expr,
) -> Result<Expr, InterpreterError> {
  // LHS must be a function call
  let (_, lhs_args) = match lhs {
    Expr::FunctionCall { name, args } => (name.clone(), args.clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "UpSetDelayed::normal: Nonatomic expression expected at position 1 in {} ^:= {}",
        crate::syntax::expr_to_string(lhs),
        crate::syntax::expr_to_string(rhs)
      )));
    }
  };

  // Find all tag symbols in the arguments
  let mut tags = Vec::new();
  for arg in &lhs_args {
    match arg {
      Expr::Identifier(s) => tags.push(s.clone()),
      Expr::FunctionCall { name, .. } => tags.push(name.clone()),
      _ => {} // Skip non-symbol arguments
    }
  }

  if tags.is_empty() {
    return Err(InterpreterError::EvaluationError(format!(
      "UpSetDelayed::nosym: {} does not contain a symbol to attach a rule to.",
      crate::syntax::expr_to_string(lhs)
    )));
  }

  // Store delayed upvalue for each tag (evaluate_rhs=false for delayed)
  for tag in &tags {
    tag_set_delayed_ast(&Expr::Identifier(tag.clone()), lhs, rhs, false)?;
  }

  // UpSetDelayed returns Null
  Ok(Expr::Identifier("Null".to_string()))
}
