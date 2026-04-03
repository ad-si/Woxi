#[allow(unused_imports)]
use super::*;

/// Compute a specificity score for a pattern rule based on its conditions,
/// blank types, and head constraints. Lower score = more specific = should be
/// tried first. Ordering: literal (SameQ) > head-constrained Blank > Blank >
/// BlankSequence > BlankNullSequence.
pub fn pattern_specificity_score(
  blank_types: &[u8],
  heads: &[Option<String>],
  conditions: &[Option<Expr>],
) -> u32 {
  // Literal definitions (SameQ conditions) are most specific
  let is_literal = conditions.iter().any(|c| {
    if let Some(Expr::Comparison { operators, .. }) = c {
      operators
        .iter()
        .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
    } else {
      false
    }
  });
  if is_literal {
    return 0;
  }
  // Use the maximum blank_type as primary score (higher = less specific)
  let max_blank = blank_types.iter().copied().max().unwrap_or(1) as u32;
  // Head constraints make a pattern more specific (subtract 1 for each)
  let head_bonus = heads.iter().filter(|h| h.is_some()).count() as u32;
  // Conditions (PatternTest, Condition) also add specificity
  let cond_bonus = conditions.iter().filter(|c| c.is_some()).count() as u32;
  // Score: higher blank_type dominates, head/condition constraints reduce score
  max_blank * 10 - head_bonus - cond_bonus
}

/// Collect all operands for an associative binary operator (Plus, Times, Alternatives),
/// flattening nested applications of the same operator.
fn collect_binary_children(
  expr: &Expr,
  target_op: &crate::syntax::BinaryOperator,
) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp { op, left, right } if op == target_op => {
      let mut parts = collect_binary_children(left, target_op);
      parts.extend(collect_binary_children(right, target_op));
      parts
    }
    _ => vec![expr.clone()],
  }
}

/// Collect all pattern variable names from an expression.
/// Returns tuples of (name, head, is_optional) for each Pattern/PatternOptional node found.
fn collect_pattern_vars(expr: &Expr) -> Vec<(String, Option<String>, bool)> {
  let mut vars = Vec::new();
  collect_pattern_vars_inner(expr, &mut vars);
  vars
}

fn collect_pattern_vars_inner(
  expr: &Expr,
  vars: &mut Vec<(String, Option<String>, bool)>,
) {
  match expr {
    Expr::Pattern { name, head, .. } => {
      if !vars.iter().any(|(n, _, _)| n == name) {
        vars.push((name.clone(), head.clone(), false));
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => {
      if !vars.iter().any(|(n, _, _)| n == name) {
        vars.push((name.clone(), head.clone(), true));
      }
      if let Some(d) = default {
        collect_pattern_vars_inner(d, vars);
      }
    }
    Expr::PatternTest { name, test, .. } => {
      if !vars.iter().any(|(n, _, _)| n == name) {
        vars.push((name.clone(), None, false));
      }
      collect_pattern_vars_inner(test, vars);
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_pattern_vars_inner(left, vars);
      collect_pattern_vars_inner(right, vars);
    }
    Expr::UnaryOp { operand, .. } => {
      collect_pattern_vars_inner(operand, vars);
    }
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_pattern_vars_inner(a, vars);
      }
    }
    Expr::List(items) => {
      for item in items {
        collect_pattern_vars_inner(item, vars);
      }
    }
    _ => {}
  }
}

/// Replace pattern variables with placeholder identifiers in an expression.
/// Returns the substituted expression.
fn replace_patterns_with_placeholders(
  expr: &Expr,
  vars: &[(String, Option<String>, bool)],
) -> Expr {
  match expr {
    Expr::Pattern { name, .. } | Expr::PatternOptional { name, .. } => {
      if vars.iter().any(|(n, _, _)| n == name) {
        Expr::Identifier(format!("__patvar{}__", name))
      } else {
        expr.clone()
      }
    }
    Expr::PatternTest { name, .. } => {
      if vars.iter().any(|(n, _, _)| n == name) {
        Expr::Identifier(format!("__patvar{}__", name))
      } else {
        expr.clone()
      }
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(replace_patterns_with_placeholders(left, vars)),
      right: Box::new(replace_patterns_with_placeholders(right, vars)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_patterns_with_placeholders(operand, vars)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| replace_patterns_with_placeholders(a, vars))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| replace_patterns_with_placeholders(a, vars))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// Replace placeholder identifiers back with Pattern or PatternOptional nodes.
fn replace_placeholders_with_patterns(
  expr: &Expr,
  vars: &[(String, Option<String>, bool)],
) -> Expr {
  match expr {
    Expr::Identifier(name) => {
      if let Some(stripped) = name
        .strip_prefix("__patvar")
        .and_then(|s| s.strip_suffix("__"))
        && let Some((pat_name, head, is_optional)) =
          vars.iter().find(|(n, _, _)| n == stripped)
      {
        if *is_optional {
          return Expr::PatternOptional {
            name: pat_name.clone(),
            head: head.clone(),
            default: None, // system-determined default
          };
        } else {
          return Expr::Pattern {
            name: pat_name.clone(),
            head: head.clone(),
            blank_type: 1,
          };
        }
      }
      expr.clone()
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(replace_placeholders_with_patterns(left, vars)),
      right: Box::new(replace_placeholders_with_patterns(right, vars)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_placeholders_with_patterns(operand, vars)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| replace_placeholders_with_patterns(a, vars))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| replace_placeholders_with_patterns(a, vars))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// Normalize a structural pattern by evaluating it with placeholder variables.
/// E.g., `1/x_` (BinaryOp Divide) → `Power[x_, -1]` (canonical form).
fn normalize_structural_pattern(pattern: &Expr) -> Expr {
  let vars = collect_pattern_vars(pattern);
  if vars.is_empty() {
    return pattern.clone();
  }
  let with_placeholders = replace_patterns_with_placeholders(pattern, &vars);
  match evaluate_expr_to_expr(&with_placeholders) {
    Ok(evaluated) => {
      // Convert BinaryOp::Divide to canonical Times[..., Power[..., -1]] form
      // so that patterns match regardless of how the expression was written
      // (e.g., 1/(a*b) vs (a*b)^-1 should both match the same pattern).
      // This is done before replacing placeholders back, since the arithmetic
      // functions (power_two, times_ast) need plain symbols, not pattern nodes.
      let evaluated = canonicalize_divide_in_expr(&evaluated);
      let result = replace_placeholders_with_patterns(&evaluated, &vars);
      // For Orderless functions (Times, Plus), reorder top-level args so
      // PatternOptional args come last. This ensures non-optional patterns
      // match earlier canonical args (e.g., numbers before symbols),
      // following Wolfram's convention for Orderless matching.
      reorder_orderless_pattern_args(result)
    }
    Err(_) => pattern.clone(), // fallback to raw pattern
  }
}

/// Recursively convert BinaryOp::Divide to canonical Times[num, Power[den, -1]]
/// form. Uses power_two/times_ast so nested structures are properly distributed
/// (e.g., 1/(a*Sqrt[b]) → Times[Power[a,-1], Power[b,-1/2]]).
pub fn canonicalize_divide_in_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      let left = canonicalize_divide_in_expr(left);
      let right = canonicalize_divide_in_expr(right);
      match crate::functions::math_ast::power_two(&right, &Expr::Integer(-1)) {
        Ok(den_inv) => {
          if matches!(left, Expr::Integer(1)) {
            den_inv
          } else {
            crate::functions::math_ast::times_ast(&[left.clone(), den_inv])
              .unwrap_or_else(|_| Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(left),
                right: Box::new(right),
              })
          }
        }
        Err(_) => Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(left),
          right: Box::new(right),
        },
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(canonicalize_divide_in_expr).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(canonicalize_divide_in_expr(left)),
      right: Box::new(canonicalize_divide_in_expr(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(canonicalize_divide_in_expr(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(canonicalize_divide_in_expr).collect())
    }
    other => other.clone(),
  }
}

/// For FunctionCall patterns of Orderless functions, move PatternOptional
/// args to the end so non-optional patterns get matched first.
fn reorder_orderless_pattern_args(pattern: Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = &pattern {
    let is_orderless = crate::evaluator::listable::is_builtin_orderless(name);
    if is_orderless
      && args.len() >= 2
      && args
        .iter()
        .any(|a| matches!(a, Expr::PatternOptional { .. }))
    {
      let mut sorted_args = args.clone();
      sorted_args.sort_by_key(|a| {
        if matches!(a, Expr::PatternOptional { .. }) {
          1
        } else {
          0
        }
      });
      return Expr::FunctionCall {
        name: name.clone(),
        args: sorted_args,
      };
    }
  }
  pattern
}

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
    crate::emit_message(&format!(
      "Attributes::locked: Symbol {} is locked.",
      sym_name
    ));
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
      crate::emit_message(&format!(
        "Attributes::attnf: {} is not a known attribute.",
        attr_str
      ));
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
  // Handle Entity property mutation: Entity["type", "name"]["property"] = value
  if let Expr::CurriedCall { func, args } = lhs
    && let Expr::FunctionCall {
      name,
      args: entity_args,
    } = func.as_ref()
    && name == "Entity"
    && entity_args.len() == 2
  {
    return crate::functions::entity_ast::entity_property_set(
      entity_args,
      args,
      rhs,
    );
  }

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
      crate::emit_message(&format!(
        "Part::wrsym: Symbol {} is Protected.",
        var_name
      ));
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
      crate::emit_message(&format!(
        "Set::wrsym: Symbol {} is Protected.",
        var_name
      ));
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
        | Expr::Graphics { .. }
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
      crate::emit_message(&format!(
        "Set::wrsym: Symbol {} is Protected.",
        func_name
      ));
      return Ok(rhs_value);
    }

    // Build param names and conditions for each argument
    let mut params = Vec::new();
    let mut conditions: Vec<Option<Expr>> = Vec::new();
    let mut defaults = Vec::new();
    let mut heads = Vec::new();
    let mut blank_types: Vec<u8> = Vec::new();

    for (i, arg) in lhs_args.iter().enumerate() {
      let param_name = format!("_dv{}", i);

      // Check if arg is a pattern (Blank, BlankSequence, named pattern, etc.)
      let is_pattern = match arg {
        Expr::Pattern { .. }
        | Expr::PatternOptional { .. }
        | Expr::PatternTest { .. } => true,
        Expr::Identifier(name) => name.contains('_'),
        _ => crate::evaluator::pattern_matching::contains_pattern(arg),
      };

      if is_pattern {
        let (pat_name, head, blank_type) = extract_pattern_info(arg);
        let final_name = if pat_name.is_empty() {
          param_name
        } else {
          pat_name
        };
        params.push(final_name);
        conditions.push(None);
        heads.push(head);
        blank_types.push(blank_type);
      } else {
        // Evaluate the literal argument value
        let eval_arg = evaluate_expr_to_expr(arg)?;
        // Condition: _dvN === eval_arg (using SameQ for exact matching)
        conditions.push(Some(Expr::Comparison {
          operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
          operators: vec![crate::syntax::ComparisonOp::SameQ],
        }));
        params.push(param_name);
        heads.push(None);
        blank_types.push(1);
      }
      defaults.push(None);
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
        // Literal-match definitions go before pattern definitions but after
        // existing literal definitions (preserving definition order).
        let pos = entry
          .iter()
          .position(|(_, c, _, _, _, _)| {
            // Find the first non-literal (pattern) definition
            !c.iter().any(|cond| {
              if let Some(Expr::Comparison { operators, .. }) = cond {
                operators
                  .iter()
                  .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
              } else {
                false
              }
            })
          })
          .unwrap_or(entry.len());
        entry.insert(
          pos,
          (
            params,
            conditions,
            defaults,
            heads,
            blank_types,
            rhs_value.clone(),
          ),
        );
      } else {
        // Insert by pattern specificity: Blank < BlankSequence < BlankNullSequence
        let score =
          pattern_specificity_score(&blank_types, &heads, &conditions);
        let pos = entry
          .iter()
          .position(|(_, c, _, h, bt, _)| {
            pattern_specificity_score(bt, h, c) > score
          })
          .unwrap_or(entry.len());
        entry.insert(
          pos,
          (
            params,
            conditions,
            defaults,
            heads,
            blank_types,
            rhs_value.clone(),
          ),
        );
      }
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
  // Unwrap Condition: f[x_] := body /; test is parsed as
  // SetDelayed[f[x_], Condition[body, test]]. Extract the body and condition.
  let (body, body_condition) = if let Expr::FunctionCall { name, args } = body
    && name == "Condition"
    && args.len() == 2
  {
    (&args[0], Some(&args[1]))
  } else {
    (body, None)
  };

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
      crate::emit_message(&format!(
        "SetDelayed::wrsym: Symbol {} is Protected.",
        func_name
      ));
      return Ok(Expr::Identifier("Null".to_string()));
    }

    let mut params = Vec::new();
    let mut conditions: Vec<Option<Expr>> = Vec::new();
    let mut defaults: Vec<Option<Expr>> = Vec::new();
    let mut heads: Vec<Option<String>> = Vec::new();
    let mut blank_types: Vec<u8> = Vec::new();
    // We also need to track substitutions for list-pattern destructuring
    let mut body_substitutions: Vec<(String, Vec<(String, Option<String>)>)> =
      Vec::new();
    let mut inline_opts_defaults: Option<Vec<Expr>> = None;

    for (i, arg) in lhs_args.iter().enumerate() {
      match arg {
        // OptionsPattern[] or OptionsPattern[{defaults...}] — matches zero or more Rule arguments
        Expr::FunctionCall {
          name: fn_name,
          args: op_args,
        } if fn_name == "OptionsPattern" => {
          let param_name = format!("__opts{}", i);
          params.push(param_name);
          conditions.push(None);
          defaults.push(None);
          heads.push(None);
          blank_types.push(3); // BlankNullSequence - matches 0 or more args
          // Extract inline defaults from OptionsPattern[{a -> a0, ...}]
          if op_args.len() == 1
            && let Expr::List(rules) = &op_args[0]
          {
            inline_opts_defaults = Some(rules.clone());
          }
        }
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
            let (pat_name, head, _blank_type) = extract_pattern_info(pat);
            element_bindings.push((pat_name, head));
          }
          body_substitutions.push((param_name.clone(), element_bindings));
          params.push(param_name);
          defaults.push(None);
          heads.push(Some("List".to_string()));
          blank_types.push(1);
        }
        // Simple pattern: x_ or x_Head
        _ => {
          let (pat_name, head, blank_type) = extract_pattern_info(arg);
          // Check for anonymous pattern identifiers (_, __, ___)
          let is_anonymous_pattern = pat_name.is_empty()
            && head.is_none()
            && matches!(arg, Expr::Identifier(name) if name.starts_with('_'));
          if is_anonymous_pattern {
            let param_name = format!("_dv{}", i);
            params.push(param_name);
            conditions.push(None);
            blank_types.push(blank_type);
          } else if pat_name.is_empty() && head.is_none() {
            if crate::evaluator::pattern_matching::contains_pattern(arg) {
              // Structural pattern (e.g., 1/x_, a_ + b_) — normalize and store
              // the pattern AST in a __StructuralPattern__ marker for dispatch-time matching.
              let param_name = format!("__sp{}", i);
              let normalized = normalize_structural_pattern(arg);
              conditions.push(Some(Expr::FunctionCall {
                name: "__StructuralPattern__".to_string(),
                args: vec![Expr::Identifier(param_name.clone()), normalized],
              }));
              params.push(param_name);
              blank_types.push(1);
            } else {
              // Literal value (not a pattern) — create a SameQ condition
              // e.g., f[1] := ... should only match when arg === 1
              let param_name = format!("_dv{}", i);
              let eval_arg = evaluate_expr_to_expr(arg)?;
              conditions.push(Some(Expr::Comparison {
                operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
                operators: vec![crate::syntax::ComparisonOp::SameQ],
              }));
              params.push(param_name);
              blank_types.push(1);
            }
          } else {
            params.push(pat_name);
            conditions.push(None);
            blank_types.push(blank_type);
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

    // If there's a body-level condition (from /;), attach it to a condition slot
    if let Some(body_cond) = body_condition {
      let mut attached = false;
      for c in conditions.iter_mut() {
        if c.is_none() {
          *c = Some(body_cond.clone());
          attached = true;
          break;
        }
      }
      if !attached && !conditions.is_empty() {
        // All slots have conditions - combine with first non-structural-pattern using And
        let combine_idx = conditions.iter().position(|c| {
          !matches!(
            c,
            Some(Expr::FunctionCall { name, .. }) if name == "__StructuralPattern__"
          )
        });
        if let Some(idx) = combine_idx {
          let existing = conditions[idx].take().unwrap();
          conditions[idx] = Some(Expr::FunctionCall {
            name: "And".to_string(),
            args: vec![existing, body_cond.clone()],
          });
        } else {
          // All conditions are structural patterns — append as extra condition
          conditions.push(Some(body_cond.clone()));
          params.push(String::new());
          defaults.push(None);
          heads.push(None);
          blank_types.push(1);
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
      let insert_pos = if has_literal_conditions {
        // Literal-match definitions go before pattern definitions but after
        // existing literal definitions (preserving definition order).
        entry
          .iter()
          .position(|(_, c, _, _, _, _)| {
            !c.iter().any(|cond| {
              if let Some(Expr::Comparison { operators, .. }) = cond {
                operators
                  .iter()
                  .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
              } else {
                false
              }
            })
          })
          .unwrap_or(entry.len())
      } else {
        // Insert by pattern specificity: Blank < BlankSequence < BlankNullSequence
        let score =
          pattern_specificity_score(&blank_types, &heads, &conditions);
        entry
          .iter()
          .position(|(_, c, _, h, bt, _)| {
            pattern_specificity_score(bt, h, c) > score
          })
          .unwrap_or(entry.len())
      };
      entry.insert(
        insert_pos,
        (params, conditions, defaults, heads, blank_types, final_body),
      );
      // Store inline OptionsPattern defaults, keeping in sync with FUNC_DEFS entries
      crate::FUNC_OPTS_INLINE.with(|oi| {
        let mut inline_map = oi.borrow_mut();
        let inline_entry =
          inline_map.entry(func_name.clone()).or_insert_with(Vec::new);
        // Ensure the inline_entry has the same length as the FUNC_DEFS entry
        while inline_entry.len() < entry.len() {
          inline_entry.push(None);
        }
        // Set the inline defaults for this overload at the correct position
        if let Some(ref opts) = inline_opts_defaults {
          inline_entry[insert_pos] = Some(opts.clone());
        }
      });
    });

    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Handle simple identifier assignment: a := expr (OwnValues)
  if let Expr::Identifier(var_name) = lhs {
    if is_symbol_protected(var_name) {
      crate::emit_message(&format!(
        "SetDelayed::wrsym: Symbol {} is Protected.",
        var_name
      ));
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

/// Extract a pattern name, optional head constraint, and blank type from a pattern expression.
/// Returns (name, head, blank_type) where blank_type is:
///   1 = Blank (_), 2 = BlankSequence (__), 3 = BlankNullSequence (___)
/// e.g., x_Integer -> ("x", Some("Integer"), 1), x__ -> ("x", None, 2)
pub fn extract_pattern_info(expr: &Expr) -> (String, Option<String>, u8) {
  match expr {
    // AST Pattern node: Pattern { name: "x", head: Some("Integer"), blank_type: 1 }
    Expr::Pattern {
      name,
      head,
      blank_type,
    } => (name.clone(), head.clone(), *blank_type),
    // AST PatternOptional node: x_:default or x_Head:default
    Expr::PatternOptional { name, head, .. } => (name.clone(), head.clone(), 1),
    Expr::Identifier(name) => {
      // Could be a pattern like "x_Integer", "x_", "x__", "x___" in text form
      if let Some(pos) = name.find('_') {
        let pat_name = name[..pos].to_string();
        let rest = &name[pos..];
        // Count consecutive underscores (1=Blank, 2=BlankSequence, 3=BlankNullSequence)
        let num_underscores = rest.chars().take_while(|c| *c == '_').count();
        let blank_type = num_underscores.min(3) as u8;
        let head_str = &rest[num_underscores..];
        if head_str.is_empty() {
          (pat_name, None, blank_type)
        } else {
          (pat_name, Some(head_str.to_string()), blank_type)
        }
      } else {
        (name.clone(), None, 1)
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      // Pattern[name, Blank[head]], Pattern[name, BlankSequence[head]], Pattern[name, BlankNullSequence[head]]
      if let Expr::Identifier(pat_name) = &args[0]
        && let Expr::FunctionCall {
          name: blank_name,
          args: blank_args,
        } = &args[1]
      {
        let blank_type = match blank_name.as_str() {
          "Blank" => 1,
          "BlankSequence" => 2,
          "BlankNullSequence" => 3,
          _ => 1,
        };
        let head = blank_args.first().and_then(|a| {
          if let Expr::Identifier(h) = a {
            Some(h.clone())
          } else {
            None
          }
        });
        return (pat_name.clone(), head, blank_type);
      }
      (String::new(), None, 1)
    }
    _ => {
      // Structural patterns (e.g., BinaryOp containing patterns) are not
      // simple named patterns — return empty to signal special handling.
      (String::new(), None, 1)
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
  // Handles FunctionCall directly, and converts BinaryOp/UnaryOp/Comparison
  // to their canonical function call form (e.g. Plus[a, b] for a + b).
  let (outer_func, lhs_args) = match lhs {
    Expr::FunctionCall { name, args } => (name.clone(), args.clone()),
    Expr::BinaryOp { op, left, right } => {
      let (name, args) = match op {
        crate::syntax::BinaryOperator::Plus => {
          ("Plus".to_string(), collect_binary_children(lhs, op))
        }
        crate::syntax::BinaryOperator::Times => {
          ("Times".to_string(), collect_binary_children(lhs, op))
        }
        crate::syntax::BinaryOperator::Alternatives => {
          ("Alternatives".to_string(), collect_binary_children(lhs, op))
        }
        crate::syntax::BinaryOperator::Minus => (
          "Plus".to_string(),
          vec![
            left.as_ref().clone(),
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: right.clone(),
            },
          ],
        ),
        crate::syntax::BinaryOperator::Divide => (
          "Times".to_string(),
          vec![
            left.as_ref().clone(),
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: right.clone(),
              right: Box::new(Expr::Integer(-1)),
            },
          ],
        ),
        crate::syntax::BinaryOperator::Power => (
          "Power".to_string(),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        ),
        crate::syntax::BinaryOperator::And => (
          "And".to_string(),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        ),
        crate::syntax::BinaryOperator::Or => (
          "Or".to_string(),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        ),
        crate::syntax::BinaryOperator::StringJoin => (
          "StringJoin".to_string(),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        ),
      };
      (name, args)
    }
    Expr::UnaryOp { op, operand } => {
      let (name, args) = match op {
        crate::syntax::UnaryOperator::Minus => (
          "Times".to_string(),
          vec![Expr::Integer(-1), operand.as_ref().clone()],
        ),
        crate::syntax::UnaryOperator::Not => {
          ("Not".to_string(), vec![operand.as_ref().clone()])
        }
      };
      (name, args)
    }
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
  // Track first occurrence of each pattern variable for SameQ conditions
  // on repeated pattern variables (e.g. v_ appearing in multiple args).
  let mut seen_pattern_vars: std::collections::HashMap<String, Expr> =
    std::collections::HashMap::new();
  // Extra conditions for repeated pattern variables
  let mut extra_conditions: Vec<Expr> = Vec::new();

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
          let (pat_name, _pat_head, _blank_type) =
            extract_pattern_info(inner_arg);
          if !pat_name.is_empty() {
            let part_expr = Expr::FunctionCall {
              name: "Part".to_string(),
              args: vec![
                Expr::Identifier(param_name.clone()),
                Expr::Integer((j + 1) as i128),
              ],
            };
            // If this pattern variable was already seen, add a SameQ
            // condition to ensure both occurrences match the same value.
            if let Some(prev_expr) = seen_pattern_vars.get(&pat_name) {
              extra_conditions.push(Expr::Comparison {
                operands: vec![prev_expr.clone(), part_expr.clone()],
                operators: vec![crate::syntax::ComparisonOp::SameQ],
              });
            } else {
              seen_pattern_vars.insert(pat_name.clone(), part_expr.clone());
            }
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
        let (pat_name, head, _blank_type) = extract_pattern_info(arg);
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

  // Add extra conditions for repeated pattern variables to the last
  // parameter's condition slot (merging with any existing condition).
  if !extra_conditions.is_empty() && !conditions.is_empty() {
    let last_idx = conditions.len() - 1;
    let mut all_conds = extra_conditions;
    if let Some(existing) = conditions[last_idx].take() {
      all_conds.insert(0, existing);
    }
    // Combine all conditions with And
    let combined = if all_conds.len() == 1 {
      all_conds.remove(0)
    } else {
      Expr::FunctionCall {
        name: "And".to_string(),
        args: all_conds,
      }
    };
    conditions[last_idx] = Some(combined);
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
      lhs.clone(),
      body.clone(),
    ));
  });

  // Store in FUNC_DEFS under the outer function name
  let blank_types = vec![1u8; params.len()];
  crate::FUNC_DEFS.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(outer_func).or_insert_with(Vec::new);
    entry.insert(
      0,
      (params, conditions, defaults, heads, blank_types, final_body),
    );
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
