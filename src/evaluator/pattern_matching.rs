#[allow(unused_imports)]
use super::*;

/// Perform nested access on an association: assoc["a", "b"] -> assoc["a"]["b"]
pub fn association_nested_access(
  var_name: &str,
  keys: &[Expr],
) -> Result<Expr, InterpreterError> {
  if keys.is_empty() {
    // Return the association itself
    return ENV.with(|e| {
      if let Some(StoredValue::Association(pairs)) = e.borrow().get(var_name) {
        let items: Vec<(Expr, Expr)> = pairs
          .iter()
          .map(|(k, v)| {
            (
              string_to_expr(k).unwrap_or(Expr::Identifier(k.clone())),
              string_to_expr(v).unwrap_or(Expr::Raw(v.clone())),
            )
          })
          .collect();
        Ok(Expr::Association(items))
      } else {
        Err(InterpreterError::EvaluationError(format!(
          "{} is not an association",
          var_name
        )))
      }
    });
  }

  // Get the association
  let assoc = ENV.with(|e| e.borrow().get(var_name).cloned());
  match assoc {
    Some(StoredValue::Association(pairs)) => {
      // Perform nested access
      let mut current_val: Option<String> = None;
      let mut current_pairs = pairs;

      for key in keys {
        // Use expr_to_string to match storage format (preserves string quotes)
        let key_str = expr_to_string(key);

        // Look up key in current association
        if let Some((_, val)) =
          current_pairs.iter().find(|(k, _)| k == &key_str)
        {
          // Check if val is a nested association
          if val.starts_with("<|") && val.ends_with("|>") {
            // Parse the nested association
            match crate::interpret(&format!("Keys[{}]", val)) {
              Ok(_) => {
                // It's an association - we need to continue drilling down
                // Parse the association into pairs
                match parse_association_string(val) {
                  Ok(nested_pairs) => {
                    current_pairs = nested_pairs;
                    current_val = None;
                  }
                  Err(_) => {
                    current_val = Some(val.clone());
                  }
                }
              }
              Err(_) => {
                current_val = Some(val.clone());
              }
            }
          } else {
            current_val = Some(val.clone());
          }
        } else {
          // Key not found
          return Ok(Expr::FunctionCall {
            name: var_name.to_string(),
            args: keys.to_vec(),
          });
        }
      }

      // Return the final value
      if let Some(val) = current_val {
        string_to_expr(&val).or(Ok(Expr::Raw(val)))
      } else {
        // Return remaining association
        let items: Vec<(Expr, Expr)> = current_pairs
          .iter()
          .map(|(k, v)| {
            (
              Expr::Identifier(k.clone()),
              string_to_expr(v).unwrap_or(Expr::Raw(v.clone())),
            )
          })
          .collect();
        Ok(Expr::Association(items))
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{} is not an association",
      var_name
    ))),
  }
}

/// Parse an association string like "<|a -> 1, b -> 2|>" into pairs
pub fn parse_association_string(
  s: &str,
) -> Result<Vec<(String, String)>, InterpreterError> {
  if !s.starts_with("<|") || !s.ends_with("|>") {
    return Err(InterpreterError::EvaluationError(
      "Not an association".into(),
    ));
  }
  let inner = &s[2..s.len() - 2]; // Strip <| and |>
  let mut pairs = Vec::new();

  // Simple parsing - split by ", " and then by " -> "
  for item in split_association_items(inner) {
    if let Some(arrow_pos) = item.find(" -> ") {
      let key = item[..arrow_pos].trim().to_string();
      let val = item[arrow_pos + 4..].trim().to_string();
      pairs.push((key, val));
    }
  }

  Ok(pairs)
}

/// Split association items handling nested associations
pub fn split_association_items(s: &str) -> Vec<String> {
  let mut items = Vec::new();
  let mut current = String::new();
  let mut depth = 0;

  for c in s.chars() {
    match c {
      '<' => {
        depth += 1;
        current.push(c);
      }
      '>' => {
        depth -= 1;
        current.push(c);
      }
      ',' if depth == 0 => {
        items.push(current.trim().to_string());
        current = String::new();
      }
      _ => current.push(c),
    }
  }
  if !current.trim().is_empty() {
    items.push(current.trim().to_string());
  }
  items
}

/// Check if a pattern Expr contains any Expr::Pattern nodes (named blanks like n_).
pub fn contains_pattern(expr: &Expr) -> bool {
  match expr {
    Expr::Pattern { .. }
    | Expr::PatternOptional { .. }
    | Expr::PatternTest { .. } => true,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      ..
    } => true,
    Expr::FunctionCall { name, .. } if name == "Alternatives" => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_pattern(left) || contains_pattern(right)
    }
    Expr::FunctionCall { args, .. } => args.iter().any(contains_pattern),
    Expr::List(items) => items.iter().any(contains_pattern),
    Expr::UnaryOp { operand, .. } => contains_pattern(operand),
    _ => false,
  }
}

/// Try AST-based structural pattern matching for a single rule on an expression.
/// Returns Some(result) if the pattern matched and was replaced.
pub fn try_ast_pattern_replace(
  expr: &Expr,
  pattern: &Expr,
  replacement: &Expr,
  condition: Option<&str>,
) -> Result<Option<Expr>, InterpreterError> {
  // Check if pattern contains any Expr::Pattern nodes
  if !contains_pattern(pattern) {
    return Ok(None);
  }

  // First try matching the entire expression (top-level match)
  if let Some(result) =
    try_ast_pattern_replace_single(expr, pattern, replacement, condition)?
  {
    return Ok(Some(result));
  }

  // If top-level didn't match, recurse into sub-expressions
  match expr {
    Expr::List(items) => {
      let mut results = Vec::new();
      let mut any_matched = false;
      for item in items {
        if let Some(result) =
          try_ast_pattern_replace(item, pattern, replacement, condition)?
        {
          results.push(result);
          any_matched = true;
        } else {
          results.push(item.clone());
        }
      }
      if any_matched {
        Ok(Some(Expr::List(results)))
      } else {
        Ok(None)
      }
    }
    Expr::FunctionCall { name, args } => {
      let mut new_args = Vec::new();
      let mut any_matched = false;
      for arg in args {
        if let Some(result) =
          try_ast_pattern_replace(arg, pattern, replacement, condition)?
        {
          new_args.push(result);
          any_matched = true;
        } else {
          new_args.push(arg.clone());
        }
      }
      if any_matched {
        Ok(Some(Expr::FunctionCall {
          name: name.clone(),
          args: new_args,
        }))
      } else {
        Ok(None)
      }
    }
    _ => Ok(None),
  }
}

/// Check if a function has the OneIdentity attribute (builtin or user-defined).
/// Check if a symbol has the Protected attribute (builtin or user-defined).
pub fn is_symbol_protected(name: &str) -> bool {
  let builtin = get_builtin_attributes(name);
  if builtin.contains(&"Protected") {
    return true;
  }
  crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(name)
      .is_some_and(|attrs| attrs.contains(&"Protected".to_string()))
  })
}

pub fn has_one_identity(name: &str) -> bool {
  let builtin = get_builtin_attributes(name);
  if builtin.contains(&"OneIdentity") {
    return true;
  }
  crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(name)
      .is_some_and(|attrs| attrs.contains(&"OneIdentity".to_string()))
  })
}

/// Try OneIdentity matching: when a pattern is f[args...] and f has OneIdentity,
/// match a non-f expression by filling in defaults for PatternOptional args
/// and matching the expression against the remaining required pattern slot.
pub fn try_one_identity_match(
  expr: &Expr,
  pat_name: &str,
  pat_args: &[Expr],
) -> Option<Vec<(String, Expr)>> {
  if !has_one_identity(pat_name) {
    return None;
  }

  // Separate args into optional (with defaults) and required patterns
  let mut required_indices = Vec::new();
  let mut bindings = Vec::new();

  for (i, arg) in pat_args.iter().enumerate() {
    match arg {
      Expr::PatternOptional { name, default, .. } => {
        // Bind optional pattern to its default value
        bindings.push((name.clone(), *default.clone()));
      }
      Expr::Pattern { .. } | Expr::Identifier(_) => {
        required_indices.push(i);
      }
      _ => {
        required_indices.push(i);
      }
    }
  }

  // OneIdentity: if there's exactly one required (non-optional) pattern slot,
  // try matching the expression against it
  if required_indices.len() == 1 {
    let req_pat = &pat_args[required_indices[0]];
    if let Some(mut req_bindings) = match_pattern(expr, req_pat) {
      req_bindings.extend(bindings);
      return Some(req_bindings);
    }
  }

  None
}

/// Try to match a single expression against a structural pattern.
pub fn try_ast_pattern_replace_single(
  value: &Expr,
  pattern: &Expr,
  replacement: &Expr,
  condition: Option<&str>,
) -> Result<Option<Expr>, InterpreterError> {
  // First try normal structural match
  let bindings_opt = match match_pattern(value, pattern) {
    Some(bindings) => Some(bindings),
    None => {
      // Try OneIdentity matching as fallback
      if let Expr::FunctionCall {
        name: pat_name,
        args: pat_args,
      } = pattern
      {
        try_one_identity_match(value, pat_name, pat_args)
      } else {
        None
      }
    }
  };

  if let Some(bindings) = bindings_opt {
    // Check condition if present
    if let Some(cond_str) = condition {
      // Substitute bindings into condition and evaluate
      let mut substituted_cond = cond_str.to_string();
      for (var, val) in &bindings {
        substituted_cond =
          replace_var_with_value(&substituted_cond, var, &expr_to_string(val));
      }
      match interpret(&substituted_cond) {
        Ok(result) if result == "True" => {}
        _ => return Ok(None), // Condition not satisfied
      }
    }
    // Substitute bindings into replacement using apply_bindings
    return Ok(Some(apply_bindings(replacement, &bindings)?));
  }
  Ok(None)
}

/// Extract the pattern Expr and optional /; condition string from a rule's pattern field.
/// Handles Expr::Raw("pattern_str /; condition_str") by parsing the pattern part
/// and returning the condition string separately.
pub fn extract_pattern_and_condition(pattern: &Expr) -> (Expr, Option<String>) {
  match pattern {
    Expr::Raw(s) if s.contains(" /; ") => {
      // Split on " /; " to get pattern and condition
      if let Some(idx) = s.find(" /; ") {
        let pattern_str = &s[..idx];
        let condition_str = &s[idx + 4..];
        if let Ok(pattern_expr) = crate::syntax::string_to_expr(pattern_str) {
          return (pattern_expr, Some(condition_str.to_string()));
        }
      }
      (pattern.clone(), None)
    }
    _ => (pattern.clone(), None),
  }
}

/// Apply Replace operation on AST - only matches at the top level (not subexpressions)
/// Replace[expr, rules] tries to apply rules only to the entire expression
/// Replace[expr, {{rule1}, {rule2}}] returns a list of results
pub fn apply_replace_ast(
  expr: &Expr,
  rules: &Expr,
) -> Result<Expr, InterpreterError> {
  // Check if rules is a list of rule-lists: Replace[x, {{x -> 1}, {x -> 2}}]
  if let Expr::List(outer_items) = rules
    && !outer_items.is_empty()
    && outer_items.iter().all(|item| matches!(item, Expr::List(_)))
  {
    // Each inner list is a set of rules to try
    let results: Result<Vec<Expr>, _> = outer_items
      .iter()
      .map(|rule_set| apply_replace_ast(expr, rule_set))
      .collect();
    return Ok(Expr::List(results?));
  }

  // Try to apply single rule or list of rules at top level only
  match rules {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      // Try structural pattern matching first
      let (pat_expr, condition) = extract_pattern_and_condition(pattern);
      if contains_pattern(&pat_expr) {
        if let Some(result) = try_ast_pattern_replace_single(
          expr,
          &pat_expr,
          replacement,
          condition.as_deref(),
        )? {
          return Ok(result);
        }
        return Ok(expr.clone());
      }
      // Simple string-based matching at top level
      let expr_str = expr_to_string(expr);
      let pattern_str = expr_to_string(pattern);
      if expr_str == pattern_str {
        return Ok((**replacement).clone());
      }
      Ok(expr.clone())
    }
    Expr::List(items) if !items.is_empty() => {
      // Multiple rules - try each in order, use first match
      for rule in items {
        match rule {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => {
            let (pat_expr, condition) = extract_pattern_and_condition(pattern);
            if contains_pattern(&pat_expr) {
              if let Some(result) = try_ast_pattern_replace_single(
                expr,
                &pat_expr,
                replacement,
                condition.as_deref(),
              )? {
                return Ok(result);
              }
              continue;
            }
            let expr_str = expr_to_string(expr);
            let pattern_str = expr_to_string(pattern);
            if expr_str == pattern_str {
              return Ok((**replacement).clone());
            }
          }
          _ => {}
        }
      }
      Ok(expr.clone())
    }
    _ => Ok(expr.clone()),
  }
}

/// Find a subset of `sub_len` arguments from `args` at `indices` that matches the pattern args
/// when wrapped in a function call. Returns the matched indices and bindings.
pub fn find_orderless_subset_match(
  func_name: &str,
  args: &[Expr],
  indices: &[usize],
  pat_args: &[Expr],
  sub_len: usize,
) -> Option<(Vec<usize>, Vec<(String, Expr)>)> {
  // Generate all combinations of sub_len indices from the available indices
  let combos = combinations(indices, sub_len);
  for combo in combos {
    let sub_args: Vec<Expr> = combo.iter().map(|&i| args[i].clone()).collect();
    // Try all permutations of the selected subset against pattern args
    let perms = permutations(&sub_args);
    for perm in perms {
      let sub_expr = Expr::FunctionCall {
        name: func_name.to_string(),
        args: perm,
      };
      if let Some(bindings) = match_pattern(
        &sub_expr,
        &Expr::FunctionCall {
          name: func_name.to_string(),
          args: pat_args.to_vec(),
        },
      ) {
        return Some((combo, bindings));
      }
    }
  }
  None
}

/// Generate all combinations of k elements from the slice.
pub fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
  if k == 0 {
    return vec![vec![]];
  }
  if items.len() < k {
    return vec![];
  }
  let mut result = Vec::new();
  for (i, &item) in items.iter().enumerate() {
    let rest = &items[i + 1..];
    for mut combo in combinations(rest, k - 1) {
      combo.insert(0, item);
      result.push(combo);
    }
  }
  result
}

/// Generate all permutations of a slice.
pub fn permutations(items: &[Expr]) -> Vec<Vec<Expr>> {
  if items.len() <= 1 {
    return vec![items.to_vec()];
  }
  let mut result = Vec::new();
  for i in 0..items.len() {
    let rest: Vec<Expr> = items
      .iter()
      .enumerate()
      .filter(|&(j, _)| j != i)
      .map(|(_, e)| e.clone())
      .collect();
    for mut perm in permutations(&rest) {
      perm.insert(0, items[i].clone());
      result.push(perm);
    }
  }
  result
}

/// Try Flat subsequence replacement. For a Flat function f, matches f[a,b] within f[a,b,c]
/// by trying contiguous subsequences. Recursively applies to subexpressions.
pub fn try_flat_replace_all(
  expr: &Expr,
  pattern: &Expr,
  replacement: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => {
      let has_flat = is_builtin_flat(name)
        || crate::FUNC_ATTRS.with(|m| {
          m.borrow()
            .get(name.as_str())
            .is_some_and(|attrs| attrs.contains(&"Flat".to_string()))
        });
      if has_flat
        && let Expr::FunctionCall {
          name: pat_name,
          args: pat_args,
        } = pattern
        && pat_name == name
        && pat_args.len() < args.len()
      {
        let has_orderless = is_builtin_orderless(name)
          || crate::FUNC_ATTRS.with(|m| {
            m.borrow()
              .get(name.as_str())
              .is_some_and(|attrs| attrs.contains(&"Orderless".to_string()))
          });

        if has_orderless {
          // For Flat+Orderless: try all combinations of sub_len args
          let sub_len = pat_args.len();
          let indices: Vec<usize> = (0..args.len()).collect();
          if let Some((matched_indices, bindings)) =
            find_orderless_subset_match(name, args, &indices, pat_args, sub_len)
          {
            let replaced = apply_bindings(replacement, &bindings)?;
            let mut new_args: Vec<Expr> = args
              .iter()
              .enumerate()
              .filter(|(i, _)| !matched_indices.contains(i))
              .map(|(_, a)| a.clone())
              .collect();
            new_args.push(replaced);
            if new_args.len() == 1 {
              return Ok(Some(new_args.into_iter().next().unwrap()));
            }
            return Ok(Some(Expr::FunctionCall {
              name: name.clone(),
              args: new_args,
            }));
          }
        } else {
          // For Flat only: try contiguous subsequences
          let sub_len = pat_args.len();
          for start in 0..=(args.len() - sub_len) {
            let sub_expr = Expr::FunctionCall {
              name: name.clone(),
              args: args[start..start + sub_len].to_vec(),
            };
            if let Some(bindings) = match_pattern(&sub_expr, pattern) {
              let replaced = apply_bindings(replacement, &bindings)?;
              let mut new_args = args[..start].to_vec();
              new_args.push(replaced);
              new_args.extend_from_slice(&args[start + sub_len..]);
              if new_args.len() == 1 {
                return Ok(Some(new_args.into_iter().next().unwrap()));
              }
              return Ok(Some(Expr::FunctionCall {
                name: name.clone(),
                args: new_args,
              }));
            }
          }
        }
      }
      // Recurse into subexpressions
      let mut changed = false;
      let mut new_args = Vec::with_capacity(args.len());
      for arg in args {
        if let Some(new_arg) = try_flat_replace_all(arg, pattern, replacement)?
        {
          new_args.push(new_arg);
          changed = true;
        } else {
          new_args.push(arg.clone());
        }
      }
      if changed {
        Ok(Some(Expr::FunctionCall {
          name: name.clone(),
          args: new_args,
        }))
      } else {
        Ok(None)
      }
    }
    Expr::List(items) => {
      let mut changed = false;
      let mut new_items = Vec::with_capacity(items.len());
      for item in items {
        if let Some(new_item) =
          try_flat_replace_all(item, pattern, replacement)?
        {
          new_items.push(new_item);
          changed = true;
        } else {
          new_items.push(item.clone());
        }
      }
      if changed {
        Ok(Some(Expr::List(new_items)))
      } else {
        Ok(None)
      }
    }
    _ => Ok(None),
  }
}

/// Apply ReplaceAll operation on AST (expr /. rules)
/// Uses AST-based structural pattern matching for patterns containing blanks (n_, x_Head, etc.),
/// falls back to string-based matching for simple patterns.
pub fn apply_replace_all_ast(
  expr: &Expr,
  rules: &Expr,
) -> Result<Expr, InterpreterError> {
  // Try AST-based structural pattern matching first for single rules
  match rules {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      let (pat_expr, condition) = extract_pattern_and_condition(pattern);
      if contains_pattern(&pat_expr)
        && let Some(result) = try_ast_pattern_replace(
          expr,
          &pat_expr,
          replacement,
          condition.as_deref(),
        )?
      {
        return Ok(result);
      }
    }
    _ => {}
  }

  // Try Flat subsequence matching at AST level
  if let Expr::Rule {
    pattern,
    replacement,
  }
  | Expr::RuleDelayed {
    pattern,
    replacement,
  } = rules
    && let Some(result) = try_flat_replace_all(expr, pattern, replacement)?
  {
    return Ok(result);
  }

  // Extract pattern and replacement strings for string-based matching
  let (pattern_str, replacement_str) = match rules {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => (expr_to_string(pattern), expr_to_string(replacement)),
    Expr::List(items) if !items.is_empty() => {
      // Multiple rules - apply at AST level for correctness and performance.
      // Collect (pattern_expr, replacement_expr) pairs.
      let rule_exprs: Vec<(&Expr, &Expr)> = items
        .iter()
        .filter_map(|rule| match rule {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => Some((pattern.as_ref(), replacement.as_ref())),
          _ => None,
        })
        .collect();
      return apply_replace_all_multi_ast(expr, &rule_exprs);
    }
    _ => return Ok(expr.clone()),
  };

  // Fall back to string-based function for simple patterns
  let expr_str = expr_to_string(expr);
  let result =
    apply_replace_all_direct(&expr_str, &pattern_str, &replacement_str)?;
  string_to_expr(&result)
}

/// Apply ReplaceRepeated operation on AST (expr //. rules)
/// Routes to string-based implementation for correct pattern matching support
pub fn apply_replace_repeated_ast(
  expr: &Expr,
  rules: &Expr,
) -> Result<Expr, InterpreterError> {
  match rules {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      let pattern_str = expr_to_string(pattern);
      let replacement_str = expr_to_string(replacement);
      let expr_str = expr_to_string(expr);
      let result = apply_replace_repeated_direct(
        &expr_str,
        &pattern_str,
        &replacement_str,
      )?;
      string_to_expr(&result)
    }
    Expr::List(items) if !items.is_empty() => {
      // Multiple rules - repeatedly try each rule in order, using first match
      let rule_pairs: Vec<(String, String)> = items
        .iter()
        .filter_map(|rule| match rule {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => Some((expr_to_string(pattern), expr_to_string(replacement))),
          _ => None,
        })
        .collect();
      let mut current = expr_to_string(expr);
      let max_iterations = 1000;
      for _ in 0..max_iterations {
        let next = apply_replace_all_direct_multi_rules(&current, &rule_pairs)?;
        if next == current {
          break;
        }
        current = next;
      }
      string_to_expr(&current)
    }
    _ => Ok(expr.clone()),
  }
}

/// Check if two Expr values are structurally equal
#[allow(dead_code)]
pub fn expr_equal(a: &Expr, b: &Expr) -> bool {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => x == y,
    (Expr::Real(x), Expr::Real(y)) => x == y,
    (Expr::String(x), Expr::String(y)) => x == y,
    (Expr::Identifier(x), Expr::Identifier(y)) => x == y,
    (Expr::Slot(x), Expr::Slot(y)) => x == y,
    (Expr::SlotSequence(x), Expr::SlotSequence(y)) => x == y,
    (Expr::Constant(x), Expr::Constant(y)) => x == y,
    (Expr::List(xs), Expr::List(ys)) => {
      xs.len() == ys.len()
        && xs.iter().zip(ys.iter()).all(|(x, y)| expr_equal(x, y))
    }
    (
      Expr::FunctionCall { name: n1, args: a1 },
      Expr::FunctionCall { name: n2, args: a2 },
    ) => {
      n1 == n2
        && a1.len() == a2.len()
        && a1.iter().zip(a2.iter()).all(|(x, y)| expr_equal(x, y))
    }
    _ => expr_to_string(a) == expr_to_string(b),
  }
}

/// Apply a list of rules once to an expression
#[allow(dead_code)]
pub fn apply_rules_once(
  expr: &Expr,
  rules: &[(&Expr, &Expr)],
) -> Result<Expr, InterpreterError> {
  // Try to match each rule against the expression
  for (pattern, replacement) in rules {
    if let Some(bindings) = match_pattern(expr, pattern) {
      return apply_bindings(replacement, &bindings);
    }
  }

  // No rule matched at the top level, try to apply rules to subexpressions
  match expr {
    Expr::List(items) => {
      let new_items: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_rules_once(item, rules))
        .collect();
      Ok(Expr::List(new_items?))
    }
    Expr::FunctionCall { name, args } => {
      // For Flat functions, try subsequence matching before recursing
      let has_flat = is_builtin_flat(name)
        || crate::FUNC_ATTRS.with(|m| {
          m.borrow()
            .get(name.as_str())
            .is_some_and(|attrs| attrs.contains(&"Flat".to_string()))
        });
      if has_flat {
        for (pattern, replacement) in rules {
          if let Expr::FunctionCall {
            name: pat_name,
            args: pat_args,
          } = pattern
            && pat_name == name
            && pat_args.len() < args.len()
          {
            // Try matching contiguous subsequences of args
            let sub_len = pat_args.len();
            for start in 0..=(args.len() - sub_len) {
              let sub_expr = Expr::FunctionCall {
                name: name.clone(),
                args: args[start..start + sub_len].to_vec(),
              };
              if let Some(bindings) = match_pattern(&sub_expr, pattern) {
                let replaced = apply_bindings(replacement, &bindings)?;
                let mut new_args = args[..start].to_vec();
                new_args.push(replaced);
                new_args.extend_from_slice(&args[start + sub_len..]);
                if new_args.len() == 1 {
                  return Ok(new_args.into_iter().next().unwrap());
                }
                return Ok(Expr::FunctionCall {
                  name: name.clone(),
                  args: new_args,
                });
              }
            }
          }
        }
      }

      let new_args: Result<Vec<Expr>, _> = args
        .iter()
        .map(|arg| apply_rules_once(arg, rules))
        .collect();
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args?,
      })
    }
    Expr::BinaryOp { op, left, right } => Ok(Expr::BinaryOp {
      op: *op,
      left: Box::new(apply_rules_once(left, rules)?),
      right: Box::new(apply_rules_once(right, rules)?),
    }),
    _ => Ok(expr.clone()),
  }
}

/// Match a pattern against an expression, returning bindings if successful
pub fn match_pattern(
  expr: &Expr,
  pattern: &Expr,
) -> Option<Vec<(String, Expr)>> {
  match pattern {
    Expr::Pattern { name, head } => {
      // Check head constraint if present
      if let Some(h) = head {
        let expr_head = get_expr_head(expr);
        if expr_head != *h {
          return None;
        }
      }
      Some(vec![(name.clone(), expr.clone())])
    }
    Expr::PatternOptional { name, head, .. } => {
      // When a value is present, PatternOptional matches like a regular Pattern
      if let Some(h) = head {
        let expr_head = get_expr_head(expr);
        if expr_head != *h {
          return None;
        }
      }
      Some(vec![(name.clone(), expr.clone())])
    }
    Expr::PatternTest { name, test } => {
      // _?test or x_?test — matches if test[expr] is True
      let test_result = match test.as_ref() {
        Expr::Identifier(func_name) => {
          let call = Expr::FunctionCall {
            name: func_name.clone(),
            args: vec![expr.clone()],
          };
          evaluate_expr_to_expr(&call).ok()
        }
        Expr::Function { body } => {
          // Anonymous function: substitute slots in the body (not the Function wrapper)
          let substituted =
            crate::syntax::substitute_slots(body, &[expr.clone()]);
          evaluate_expr_to_expr(&substituted).ok()
        }
        _ => {
          // General expression used as function: call (test)[expr]
          let call_str =
            format!("({})[{}]", expr_to_string(test), expr_to_string(expr));
          interpret(&call_str).ok().map(|r| {
            if r == "True" {
              Expr::Identifier("True".to_string())
            } else {
              Expr::Identifier("False".to_string())
            }
          })
        }
      };
      match test_result {
        Some(Expr::Identifier(s)) if s == "True" => {
          if name.is_empty() {
            Some(vec![])
          } else {
            Some(vec![(name.clone(), expr.clone())])
          }
        }
        _ => None,
      }
    }
    // Blank[] or Blank[h] as FunctionCall
    Expr::FunctionCall { name, args } if name == "Blank" => match args.len() {
      0 => Some(vec![]),
      1 => {
        if let Expr::Identifier(h) = &args[0] {
          let expr_head = get_expr_head(expr);
          if expr_head == *h { Some(vec![]) } else { None }
        } else {
          None
        }
      }
      _ => None,
    },
    // Pattern[name, blank] as FunctionCall
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      let pattern_name = if let Expr::Identifier(n) = &args[0] {
        n.clone()
      } else {
        return None;
      };
      if let Some(mut bindings) = match_pattern(expr, &args[1]) {
        bindings.push((pattern_name, expr.clone()));
        Some(bindings)
      } else {
        None
      }
    }
    Expr::Identifier(name) if name.ends_with('_') => {
      // Blank pattern like x_
      let var_name = name.trim_end_matches('_');
      Some(vec![(var_name.to_string(), expr.clone())])
    }
    Expr::Identifier(name) if name.starts_with('_') && name.len() > 1 => {
      // Head-constrained blank: _Head matches expressions with head Head
      let head = &name[1..];
      let expr_head = get_expr_head(expr);
      if expr_head == head {
        Some(vec![])
      } else {
        None
      }
    }
    Expr::Integer(n) => {
      if matches!(expr, Expr::Integer(m) if m == n) {
        Some(vec![])
      } else if let Expr::BigInteger(m) = expr {
        use num_traits::ToPrimitive;
        if m.to_i128() == Some(*n) {
          Some(vec![])
        } else {
          None
        }
      } else {
        None
      }
    }
    Expr::BigInteger(n) => {
      if let Expr::BigInteger(m) = expr {
        if m == n { Some(vec![]) } else { None }
      } else if let Expr::Integer(m) = expr {
        if num_bigint::BigInt::from(*m) == *n {
          Some(vec![])
        } else {
          None
        }
      } else {
        None
      }
    }
    Expr::Real(f) => {
      if matches!(expr, Expr::Real(g) if (f - g).abs() < f64::EPSILON) {
        Some(vec![])
      } else {
        None
      }
    }
    Expr::Identifier(name) => {
      if matches!(expr, Expr::Identifier(n) if n == name) {
        Some(vec![])
      } else {
        None
      }
    }
    Expr::String(s) => {
      if matches!(expr, Expr::String(t) if t == s) {
        Some(vec![])
      } else {
        None
      }
    }
    Expr::List(pat_items) => {
      if let Expr::List(expr_items) = expr {
        if pat_items.len() != expr_items.len() {
          return None;
        }
        let mut bindings = Vec::new();
        for (p, e) in pat_items.iter().zip(expr_items.iter()) {
          if let Some(b) = match_pattern(e, p) {
            bindings.extend(b);
          } else {
            return None;
          }
        }
        Some(bindings)
      } else {
        None
      }
    }
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } if pat_name == "Alternatives" && !pat_args.is_empty() => {
      // Alternatives as FunctionCall: try each alternative
      for alt in pat_args {
        if let Some(b) = match_pattern(expr, alt) {
          return Some(b);
        }
      }
      None
    }
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } => {
      if let Expr::FunctionCall {
        name: expr_name,
        args: expr_args,
      } = expr
      {
        if pat_name != expr_name || pat_args.len() != expr_args.len() {
          return None;
        }
        let mut bindings = Vec::new();
        for (p, e) in pat_args.iter().zip(expr_args.iter()) {
          if let Some(b) = match_pattern(e, p) {
            bindings.extend(b);
          } else {
            return None;
          }
        }
        Some(bindings)
      } else {
        None
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      left: alt_left,
      right: alt_right,
    } => {
      // Alternatives pattern: try each alternative
      if let Some(b) = match_pattern(expr, alt_left) {
        return Some(b);
      }
      match_pattern(expr, alt_right)
    }
    Expr::BinaryOp {
      op: pat_op,
      left: pat_left,
      right: pat_right,
    } => {
      if let Expr::BinaryOp {
        op: expr_op,
        left: expr_left,
        right: expr_right,
      } = expr
      {
        if pat_op != expr_op {
          return None;
        }
        let mut bindings = Vec::new();
        if let Some(b) = match_pattern(expr_left, pat_left) {
          bindings.extend(b);
        } else {
          return None;
        }
        if let Some(b) = match_pattern(expr_right, pat_right) {
          bindings.extend(b);
        } else {
          return None;
        }
        Some(bindings)
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: pat_op,
      operand: pat_operand,
    } => {
      if let Expr::UnaryOp {
        op: expr_op,
        operand: expr_operand,
      } = expr
      {
        if pat_op != expr_op {
          return None;
        }
        match_pattern(expr_operand, pat_operand)
      } else {
        None
      }
    }
    _ => {
      // For other patterns, check structural equality
      if expr_equal(expr, pattern) {
        Some(vec![])
      } else {
        None
      }
    }
  }
}

/// Get the head of an expression (for pattern matching with head constraints)
pub fn get_expr_head(expr: &Expr) -> String {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => "Integer".to_string(),
    Expr::Real(_) | Expr::BigFloat(_, _) => "Real".to_string(),
    Expr::String(_) => "String".to_string(),
    Expr::List(_) => "List".to_string(),
    Expr::FunctionCall { name, .. } => name.clone(),
    Expr::Association(_) => "Association".to_string(),
    _ => "Symbol".to_string(),
  }
}

/// Get the head of an expression from its string representation (for string-based pattern matching)
pub fn get_string_expr_head(expr: &str) -> String {
  let expr = expr.trim();
  if expr.starts_with('"') && expr.ends_with('"') {
    "String".to_string()
  } else if expr.starts_with('{') && expr.ends_with('}') {
    "List".to_string()
  } else if expr.starts_with("<|") && expr.ends_with("|>") {
    "Association".to_string()
  } else if expr.contains('[') && expr.ends_with(']') {
    // FunctionCall: extract the function name
    let bracket_pos = expr.find('[').unwrap();
    expr[..bracket_pos].to_string()
  } else if expr.contains('.') && expr.parse::<f64>().is_ok() {
    "Real".to_string()
  } else if expr.parse::<i64>().is_ok() {
    "Integer".to_string()
  } else {
    "Symbol".to_string()
  }
}

/// Apply bindings to a replacement expression
pub fn apply_bindings(
  replacement: &Expr,
  bindings: &[(String, Expr)],
) -> Result<Expr, InterpreterError> {
  let mut result = replacement.clone();
  for (name, value) in bindings {
    result = crate::syntax::substitute_variable(&result, name, value);
  }
  // Evaluate the result after substitution
  evaluate_expr_to_expr(&result)
}

/// Resolve an identifier to a function name if it's a variable holding a symbol.
/// Returns Some(resolved_name) if the variable holds a different identifier or Raw symbol name.
/// Returns None if the variable doesn't exist, holds the same name, or holds a non-identifier value.
pub fn resolve_identifier_to_func_name(name: &str) -> Option<String> {
  ENV.with(|e| {
    let env = e.borrow();
    match env.get(name) {
      Some(StoredValue::ExprVal(Expr::Identifier(resolved)))
        if resolved != name =>
      {
        Some(resolved.clone())
      }
      Some(StoredValue::Raw(s))
        if s != name
          && !s.is_empty()
          && s
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '$')
          && s
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$') =>
      {
        Some(s.clone())
      }
      _ => None,
    }
  })
}

/// Apply ReplaceAll with multiple rules at the AST level.
/// For each sub-expression, try each rule; the first matching rule wins.
/// If no rule matches the whole node, recurse into children.
pub fn apply_replace_all_multi_ast(
  expr: &Expr,
  rules: &[(&Expr, &Expr)],
) -> Result<Expr, InterpreterError> {
  // Try each rule against the whole expression
  for (pattern, replacement) in rules {
    // Handle Expr::Raw patterns (conditional patterns like n_ /; EvenQ[n])
    if let Expr::Raw(pat_str) = pattern
      && let Some(wolfram_pattern) = parse_wolfram_pattern(pat_str)
    {
      let expr_str = expr_to_string(expr);
      let repl_str = expr_to_string(replacement);
      if let Ok(Some(result_str)) =
        apply_wolfram_pattern(&expr_str, &wolfram_pattern, &repl_str)
      {
        let evaluated = evaluate_fullform(&result_str).unwrap_or(result_str);
        return string_to_expr(&evaluated);
      }
      continue;
    }
    if let Some(bindings) = match_pattern(expr, pattern) {
      // Substitute bindings into the replacement
      let mut result = (*replacement).clone();
      for (name, val) in &bindings {
        if !name.is_empty() {
          result = crate::syntax::substitute_variable(&result, name, val);
        }
      }
      return evaluate_expr_to_expr(&result);
    }
  }

  // No rule matched the whole expression — recurse into sub-expressions
  match expr {
    Expr::List(items) => {
      let new_items: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_replace_all_multi_ast(item, rules))
        .collect();
      Ok(Expr::List(new_items?))
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Result<Vec<Expr>, _> = args
        .iter()
        .map(|arg| apply_replace_all_multi_ast(arg, rules))
        .collect();
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args?,
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let new_left = apply_replace_all_multi_ast(left, rules)?;
      let new_right = apply_replace_all_multi_ast(right, rules)?;
      Ok(Expr::BinaryOp {
        op: *op,
        left: Box::new(new_left),
        right: Box::new(new_right),
      })
    }
    // Atoms and other nodes without children — return unchanged
    _ => Ok(expr.clone()),
  }
}
