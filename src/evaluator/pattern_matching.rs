#[allow(unused_imports)]
use super::*;
use std::cell::RefCell;

// Thread-local stack of accumulated bindings from outer FunctionCall arg loops.
// Used by Orderless matching with Optional patterns to check compatibility
// with already-bound variables from outer pattern contexts.
thread_local! {
  static MATCH_CONTEXT: RefCell<Vec<Vec<(String, Expr)>>> = const { RefCell::new(Vec::new()) };
}

fn push_match_context(bindings: &[(String, Expr)]) {
  MATCH_CONTEXT.with(|ctx| ctx.borrow_mut().push(bindings.to_vec()));
}

fn pop_match_context() {
  MATCH_CONTEXT.with(|ctx| ctx.borrow_mut().pop());
}

/// Public wrappers for use from dispatch code.
pub fn push_match_context_pub(bindings: &[(String, Expr)]) {
  push_match_context(bindings);
}

pub fn pop_match_context_pub() {
  pop_match_context();
}

/// Check if bindings are compatible with all outer context bindings.
fn bindings_compatible_with_context(bindings: &[(String, Expr)]) -> bool {
  MATCH_CONTEXT.with(|ctx| {
    let stack = ctx.borrow();
    for level in stack.iter() {
      for (name, val) in bindings {
        if name.is_empty() {
          continue;
        }
        if let Some((_, ctx_val)) = level.iter().find(|(n, _)| n == name)
          && !expr_equal(val, ctx_val)
        {
          return false;
        }
      }
    }
    true
  })
}

/// Merge new bindings into existing bindings, checking for consistency.
/// If a variable name already has a binding, the new value must be
/// structurally equal. Returns false if there is a conflict.
pub(crate) fn merge_bindings(
  existing: &mut Vec<(String, Expr)>,
  new: Vec<(String, Expr)>,
) -> bool {
  for (name, value) in new {
    if name.is_empty() {
      continue;
    }
    if let Some((_, existing_value)) = existing.iter().find(|(n, _)| *n == name)
    {
      if !expr_equal(existing_value, &value) {
        return false;
      }
      // Already bound to the same value, skip duplicate
    } else {
      existing.push((name, value));
    }
  }
  true
}

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
  stacker::maybe_grow(2 * 1024 * 1024, 4 * 1024 * 1024, || {
    try_ast_pattern_replace_impl(expr, pattern, replacement, condition)
  })
}

fn try_ast_pattern_replace_impl(
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

/// Map a BinaryOperator to the corresponding Wolfram Language function name.
pub fn binary_op_to_func_name(
  op: &crate::syntax::BinaryOperator,
) -> &'static str {
  use crate::syntax::BinaryOperator;
  match op {
    BinaryOperator::Plus => "Plus",
    BinaryOperator::Times => "Times",
    BinaryOperator::Power => "Power",
    BinaryOperator::And => "And",
    BinaryOperator::Or => "Or",
    _ => "",
  }
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
        if let Some(d) = default {
          bindings.push((name.clone(), *d.clone()));
        } else if let Some(def) =
          crate::evaluator::dispatch::builtin_default_value_at_position(
            pat_name,
            i + 1,
          )
        {
          // System-determined default (_.): use Default[f, position]
          bindings.push((name.clone(), def));
        } else if let Some(def) =
          crate::evaluator::dispatch::builtin_default_value(pat_name)
        {
          // Fallback to position-independent Default[f]
          bindings.push((name.clone(), def));
        }
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
    if let Some(mut req_bindings) = match_pattern(expr, req_pat)
      && merge_bindings(&mut req_bindings, bindings)
    {
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

/// Apply Replace with level specification: Replace[expr, rules, levelspec]
/// Traverses the expression and applies rules at the specified levels.
/// Supports both positive levels (distance from root) and negative levels
/// (counted from the leaves: -1 = atoms, -2 = depth-2 subtrees, etc.).
pub fn apply_replace_with_level_ast(
  expr: &Expr,
  rules: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Parse level spec: n = {0, n}, {n} = exactly level n, {min, max} = range.
  // Negative endpoints are interpreted relative to the depth of the subtree.
  let level_val = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => Some(*n as i64),
      Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity" => {
        Some(i64::MAX)
      }
      Expr::FunctionCall { name, args }
        if name == "Minus" && args.len() == 1 =>
      {
        if let Expr::Integer(n) = &args[0] {
          Some(-(*n as i64))
        } else {
          None
        }
      }
      _ => crate::evaluator::type_helpers::expr_to_i128(e).map(|n| n as i64),
    }
  };
  let (min_level, max_level) = match level_spec {
    Expr::Integer(n) => (0i64, *n as i64),
    // `All` means every level including the head — equivalent to {0, Infinity}.
    Expr::Identifier(s) | Expr::Constant(s) if s == "All" => (0i64, i64::MAX),
    Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity" => {
      (1i64, i64::MAX)
    }
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = level_val(&items[0]) {
        (n, n)
      } else {
        return Ok(Expr::FunctionCall {
          name: "Replace".to_string(),
          args: vec![expr.clone(), rules.clone(), level_spec.clone()],
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      let min = level_val(&items[0]).unwrap_or(0);
      let max = level_val(&items[1]).unwrap_or(0);
      (min, max)
    }
    _ => match level_val(level_spec) {
      Some(n) => (0, n),
      None => {
        return Ok(Expr::FunctionCall {
          name: "Replace".to_string(),
          args: vec![expr.clone(), rules.clone(), level_spec.clone()],
        });
      }
    },
  };

  let (result, _depth) =
    replace_at_depth(expr, rules, 0, min_level, max_level)?;
  Ok(result)
}

/// Recursively apply Replace rules at the specified levels (bottom-up, like
/// Mathematica). Returns the rewritten expression along with its Mathematica
/// `Depth` (1 for atoms, 1 + max(child Depth) otherwise) so that negative
/// level endpoints can be evaluated.
fn replace_at_depth(
  expr: &Expr,
  rules: &Expr,
  pos_level: i64,
  min_level: i64,
  max_level: i64,
) -> Result<(Expr, i64), InterpreterError> {
  // First recurse into children, tracking the maximum child Depth so we can
  // compute Depth[expr] for negative-level matching.
  let (recursed, max_child_depth) = match expr {
    Expr::List(items) => {
      let mut mapped = Vec::with_capacity(items.len());
      let mut max_depth: i64 = 0;
      for item in items {
        let (e, d) =
          replace_at_depth(item, rules, pos_level + 1, min_level, max_level)?;
        max_depth = max_depth.max(d);
        mapped.push(e);
      }
      (Expr::List(mapped), max_depth)
    }
    Expr::FunctionCall { name, args } => {
      let mut mapped = Vec::with_capacity(args.len());
      let mut max_depth: i64 = 0;
      for item in args {
        let (e, d) =
          replace_at_depth(item, rules, pos_level + 1, min_level, max_level)?;
        max_depth = max_depth.max(d);
        mapped.push(e);
      }
      (
        Expr::FunctionCall {
          name: name.clone(),
          args: mapped,
        },
        max_depth,
      )
    }
    _ => (expr.clone(), 0),
  };

  let depth = 1 + max_child_depth; // Mathematica Depth (atoms = 1)
  let neg_level = -depth;

  // Check both endpoints against the appropriate axis (positive or negative).
  let min_ok = if min_level >= 0 {
    pos_level >= min_level
  } else {
    neg_level >= min_level
  };
  let max_ok = if max_level >= 0 {
    pos_level <= max_level
  } else {
    neg_level <= max_level
  };

  let result = if min_ok && max_ok {
    apply_replace_ast(&recursed, rules)?
  } else {
    recursed
  };

  Ok((result, depth))
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

/// Check if a pattern tree contains any PatternOptional nodes.
fn pattern_contains_optional(pat: &Expr) -> bool {
  match pat {
    Expr::PatternOptional { .. } => true,
    Expr::FunctionCall { args, .. } => {
      args.iter().any(pattern_contains_optional)
    }
    Expr::BinaryOp { left, right, .. } => {
      pattern_contains_optional(left) || pattern_contains_optional(right)
    }
    Expr::UnaryOp { operand, .. } => pattern_contains_optional(operand),
    _ => false,
  }
}

/// Count how many PatternOptional variables in a pattern tree have their
/// default value in the given bindings.  Used to prefer Orderless matches
/// that maximise default usage (Wolfram semantics).
fn count_optional_defaults_used(
  _outer_func: &str,
  pat_args: &[Expr],
  bindings: &[(String, Expr)],
) -> usize {
  let mut count = 0;
  for pat in pat_args {
    count += count_defaults_in_pat(pat, _outer_func, bindings);
  }
  count
}

fn count_defaults_in_pat(
  pat: &Expr,
  context_func: &str,
  bindings: &[(String, Expr)],
) -> usize {
  match pat {
    Expr::PatternOptional { name, default, .. } => {
      // Determine the default value
      let default_val = match default {
        Some(d) => Some(d.as_ref().clone()),
        None => crate::evaluator::builtin_default_value(context_func),
      };
      if let Some(def) = default_val {
        if let Some((_, val)) = bindings.iter().find(|(n, _)| n == name) {
          if expr_equal(val, &def) { 1 } else { 0 }
        } else {
          0
        }
      } else {
        0
      }
    }
    Expr::FunctionCall { name, args } => args
      .iter()
      .map(|a| count_defaults_in_pat(a, name, bindings))
      .sum(),
    Expr::BinaryOp { op, left, right } => {
      let func = crate::evaluator::pattern_matching::binary_op_to_func_name(op);
      let ctx = if func.is_empty() { context_func } else { func };
      count_defaults_in_pat(left, ctx, bindings)
        + count_defaults_in_pat(right, ctx, bindings)
    }
    _ => 0,
  }
}

/// Try simple symbol replacement at the AST level.
/// Handles cases like `List -> Sequence` where `Expr::List` has head "List"
/// but doesn't contain the literal string "List" in its printed form `{...}`.
/// Returns `Some(new_expr)` if any replacement was made, `None` otherwise.
fn try_symbol_replace_all(
  expr: &Expr,
  pattern_sym: &str,
  replacement: &Expr,
) -> Option<Expr> {
  match expr {
    // Direct symbol match: replace identifiers that match the pattern
    Expr::Identifier(name) if name == pattern_sym => Some(replacement.clone()),

    // List head replacement: List -> X turns {a,b} into X[a,b]
    Expr::List(items) if pattern_sym == "List" => {
      // Recurse into items first
      let new_items: Vec<Expr> = items
        .iter()
        .map(|item| {
          try_symbol_replace_all(item, pattern_sym, replacement)
            .unwrap_or_else(|| item.clone())
        })
        .collect();
      // Replace the head
      match replacement {
        Expr::Identifier(new_head) => Some(Expr::FunctionCall {
          name: new_head.clone(),
          args: new_items,
        }),
        _ => {
          // For non-symbol replacements, convert to FullForm-style
          let head_str = expr_to_string(replacement);
          Some(Expr::FunctionCall {
            name: head_str,
            args: new_items,
          })
        }
      }
    }

    // Recurse into List items (when pattern is not "List")
    Expr::List(items) => {
      let mut new_items = Vec::with_capacity(items.len());
      let mut any_changed = false;
      for item in items {
        if let Some(new_item) =
          try_symbol_replace_all(item, pattern_sym, replacement)
        {
          new_items.push(new_item);
          any_changed = true;
        } else {
          new_items.push(item.clone());
        }
      }
      if any_changed {
        Some(Expr::List(new_items))
      } else {
        None
      }
    }

    // FunctionCall head replacement: f -> g turns f[a,b] into g[a,b]
    Expr::FunctionCall { name, args } => {
      let head_changed = name == pattern_sym;
      // Recurse into args
      let mut new_args = Vec::with_capacity(args.len());
      let mut any_arg_changed = false;
      for arg in args {
        if let Some(new_arg) =
          try_symbol_replace_all(arg, pattern_sym, replacement)
        {
          new_args.push(new_arg);
          any_arg_changed = true;
        } else {
          new_args.push(arg.clone());
        }
      }
      if head_changed {
        match replacement {
          Expr::Identifier(new_head) => Some(Expr::FunctionCall {
            name: new_head.clone(),
            args: new_args,
          }),
          // Non-symbol replacement: create CurriedCall (f -> expr turns f[a,b] into expr[a,b])
          _ => Some(Expr::CurriedCall {
            func: Box::new(replacement.clone()),
            args: new_args,
          }),
        }
      } else if any_arg_changed {
        Some(Expr::FunctionCall {
          name: name.clone(),
          args: new_args,
        })
      } else {
        None
      }
    }

    // Recurse into BinaryOp (Plus, Times, Divide, Power, etc.)
    Expr::BinaryOp { op, left, right } => {
      let new_left = try_symbol_replace_all(left, pattern_sym, replacement);
      let new_right = try_symbol_replace_all(right, pattern_sym, replacement);
      if new_left.is_some() || new_right.is_some() {
        Some(Expr::BinaryOp {
          op: *op,
          left: Box::new(new_left.unwrap_or_else(|| left.as_ref().clone())),
          right: Box::new(new_right.unwrap_or_else(|| right.as_ref().clone())),
        })
      } else {
        None
      }
    }

    // Recurse into UnaryOp (Minus, Not, etc.)
    Expr::UnaryOp { op, operand } => {
      try_symbol_replace_all(operand, pattern_sym, replacement).map(
        |new_operand| Expr::UnaryOp {
          op: *op,
          operand: Box::new(new_operand),
        },
      )
    }

    // Recurse into Rule
    Expr::Rule {
      pattern: pat,
      replacement: repl,
    } => {
      let new_pat = try_symbol_replace_all(pat, pattern_sym, replacement);
      let new_repl = try_symbol_replace_all(repl, pattern_sym, replacement);
      if new_pat.is_some() || new_repl.is_some() {
        Some(Expr::Rule {
          pattern: Box::new(new_pat.unwrap_or_else(|| pat.as_ref().clone())),
          replacement: Box::new(
            new_repl.unwrap_or_else(|| repl.as_ref().clone()),
          ),
        })
      } else {
        None
      }
    }

    // Recurse into RuleDelayed
    Expr::RuleDelayed {
      pattern: pat,
      replacement: repl,
    } => {
      let new_pat = try_symbol_replace_all(pat, pattern_sym, replacement);
      let new_repl = try_symbol_replace_all(repl, pattern_sym, replacement);
      if new_pat.is_some() || new_repl.is_some() {
        Some(Expr::RuleDelayed {
          pattern: Box::new(new_pat.unwrap_or_else(|| pat.as_ref().clone())),
          replacement: Box::new(
            new_repl.unwrap_or_else(|| repl.as_ref().clone()),
          ),
        })
      } else {
        None
      }
    }

    // Recurse into CurriedCall
    Expr::CurriedCall { func, args } => {
      let new_func = try_symbol_replace_all(func, pattern_sym, replacement);
      let mut new_args = Vec::with_capacity(args.len());
      let mut any_arg_changed = false;
      for arg in args {
        if let Some(new_arg) =
          try_symbol_replace_all(arg, pattern_sym, replacement)
        {
          new_args.push(new_arg);
          any_arg_changed = true;
        } else {
          new_args.push(arg.clone());
        }
      }
      if new_func.is_some() || any_arg_changed {
        Some(Expr::CurriedCall {
          func: Box::new(new_func.unwrap_or_else(|| func.as_ref().clone())),
          args: new_args,
        })
      } else {
        None
      }
    }

    // Recurse into Part
    Expr::Part { expr: e, index } => {
      let new_expr = try_symbol_replace_all(e, pattern_sym, replacement);
      let new_index = try_symbol_replace_all(index, pattern_sym, replacement);
      if new_expr.is_some() || new_index.is_some() {
        Some(Expr::Part {
          expr: Box::new(new_expr.unwrap_or_else(|| e.as_ref().clone())),
          index: Box::new(new_index.unwrap_or_else(|| index.as_ref().clone())),
        })
      } else {
        None
      }
    }

    // Recurse into Comparison
    Expr::Comparison {
      operands,
      operators,
    } => {
      let mut new_operands = Vec::with_capacity(operands.len());
      let mut any_changed = false;
      for operand in operands {
        if let Some(new_op) =
          try_symbol_replace_all(operand, pattern_sym, replacement)
        {
          new_operands.push(new_op);
          any_changed = true;
        } else {
          new_operands.push(operand.clone());
        }
      }
      if any_changed {
        Some(Expr::Comparison {
          operands: new_operands,
          operators: operators.clone(),
        })
      } else {
        None
      }
    }

    // Recurse into Association
    Expr::Association(pairs) => {
      let mut new_pairs = Vec::with_capacity(pairs.len());
      let mut any_changed = false;
      for (k, v) in pairs {
        let new_k = try_symbol_replace_all(k, pattern_sym, replacement);
        let new_v = try_symbol_replace_all(v, pattern_sym, replacement);
        if new_k.is_some() || new_v.is_some() {
          any_changed = true;
        }
        new_pairs.push((
          new_k.unwrap_or_else(|| k.clone()),
          new_v.unwrap_or_else(|| v.clone()),
        ));
      }
      if any_changed {
        Some(Expr::Association(new_pairs))
      } else {
        None
      }
    }

    _ => None,
  }
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
/// Unwrap a top-level `HoldPattern[p]` from a pattern expression. HoldPattern
/// is transparent to matching — it only prevents evaluation of the wrapped
/// expression when parsing.
fn strip_hold_pattern(pattern: &Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = pattern
    && name == "HoldPattern"
    && args.len() == 1
  {
    return args[0].clone();
  }
  pattern.clone()
}

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
      let stripped = strip_hold_pattern(pattern);
      let (pat_expr, condition) = extract_pattern_and_condition(&stripped);
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

  // Try AST-level symbol replacement (handles Expr::List head replacement for List -> X)
  if let Expr::Rule {
    pattern,
    replacement,
  }
  | Expr::RuleDelayed {
    pattern,
    replacement,
  } = rules
  {
    let stripped = strip_hold_pattern(pattern);
    if let Expr::Identifier(pat_sym) = &stripped
      && let Some(result) = try_symbol_replace_all(expr, pat_sym, replacement)
    {
      return Ok(result);
    }
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
      // Check if this is a list of rule-lists: {{x->1}, {x->2}}
      // In Wolfram, expr /. {{r1}, {r2}} applies each sub-list independently
      // and returns a list of results: {expr/.{r1}, expr/.{r2}}
      let is_list_of_rule_lists = items.iter().all(|item| {
        matches!(item, Expr::List(sub) if sub.iter().all(|r|
          matches!(r, Expr::Rule { .. } | Expr::RuleDelayed { .. })
        ))
      });
      if is_list_of_rule_lists {
        let results: Result<Vec<Expr>, _> = items
          .iter()
          .map(|sub_rules| apply_replace_all_ast(expr, sub_rules))
          .collect();
        return results.map(Expr::List);
      }
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
/// Applies ReplaceAll repeatedly until the expression stops changing.
pub fn apply_replace_repeated_ast(
  expr: &Expr,
  rules: &Expr,
) -> Result<Expr, InterpreterError> {
  let max_iterations = 65536;
  let mut current = expr.clone();
  for _ in 0..max_iterations {
    let next = apply_replace_all_ast(&current, rules)?;
    if expr_equal(&next, &current) {
      break;
    }
    // Re-evaluate after substitution so e.g. 3^2 becomes 9
    current = evaluate_expr_to_expr(&next)?;
  }
  Ok(current)
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

/// Sequence pattern info: name, head constraint, min/max count, optional test function.
struct SeqInfo {
  name: String,
  head: Option<String>,
  min_count: usize,
  max_count: Option<usize>, // None means unlimited
  test: Option<Box<Expr>>,
  /// For Repeated/RepeatedNull: the element pattern to match against each arg.
  element_pattern: Option<Box<Expr>>,
  /// Condition test to evaluate after the sequence is matched (for x__ /; cond patterns).
  condition: Option<Box<Expr>>,
}

/// Check if a pattern is a sequence pattern (BlankSequence or BlankNullSequence).
fn get_sequence_info(pattern: &Expr) -> Option<SeqInfo> {
  match pattern {
    Expr::Pattern {
      name,
      head,
      blank_type,
    } if *blank_type >= 2 => {
      let min = if *blank_type == 2 { 1 } else { 0 };
      Some(SeqInfo {
        name: name.clone(),
        head: head.clone(),
        min_count: min,
        max_count: None,
        test: None,
        element_pattern: None,
        condition: None,
      })
    }
    // Condition wrapping a sequence pattern: x__Integer /; test
    Expr::FunctionCall {
      name: cond_name,
      args: cond_args,
    } if cond_name == "Condition" && cond_args.len() == 2 => {
      // Recurse on the inner pattern and attach the condition
      let mut seq = get_sequence_info(&cond_args[0])?;
      seq.condition = Some(Box::new(cond_args[1].clone()));
      Some(seq)
    }
    // PatternTest with BlankSequence: x__?IntegerQ or __?IntegerQ
    Expr::PatternTest {
      name,
      head,
      blank_type,
      test,
    } if *blank_type >= 2 => {
      let min = if *blank_type == 2 { 1 } else { 0 };
      Some(SeqInfo {
        name: name.clone(),
        head: head.clone(),
        min_count: min,
        max_count: None,
        test: Some(test.clone()),
        element_pattern: None,
        condition: None,
      })
    }
    // BlankSequence[] or BlankSequence[h] as FunctionCall
    Expr::FunctionCall { name, args }
      if name == "BlankSequence" || name == "BlankNullSequence" =>
    {
      let head = if args.len() == 1 {
        if let Expr::Identifier(h) = &args[0] {
          Some(h.clone())
        } else {
          None
        }
      } else {
        None
      };
      let min = if name == "BlankSequence" { 1 } else { 0 };
      Some(SeqInfo {
        name: String::new(),
        head,
        min_count: min,
        max_count: None,
        test: None,
        element_pattern: None,
        condition: None,
      })
    }
    // Repeated[pat] or Repeated[pat, {min, max}] — matches 1+ elements each matching pat
    // RepeatedNull[pat] or RepeatedNull[pat, {min, max}] — matches 0+ elements
    Expr::FunctionCall { name, args }
      if (name == "Repeated" || name == "RepeatedNull")
        && !args.is_empty()
        && args.len() <= 2 =>
    {
      let is_null = name == "RepeatedNull";
      let (mut min, mut max): (usize, Option<usize>) =
        if is_null { (0, None) } else { (1, None) };

      // Parse optional count spec: {n}, {min, max}, or just n
      if args.len() == 2 {
        let spec_items: Option<&[Expr]> = match &args[1] {
          Expr::List(items) => Some(items),
          Expr::FunctionCall {
            name: list_name,
            args: spec,
          } if list_name == "List" => Some(spec),
          _ => None,
        };
        if let Some(items) = spec_items {
          match items.len() {
            1 => {
              if let Expr::Integer(n) = &items[0] {
                let n = *n as usize;
                min = n;
                max = Some(n);
              }
            }
            2 => {
              if let Expr::Integer(lo) = &items[0] {
                min = *lo as usize;
              }
              if let Expr::Integer(hi) = &items[1] {
                max = Some(*hi as usize);
              }
            }
            _ => {}
          }
        } else if let Expr::Integer(n) = &args[1] {
          let n = *n as usize;
          min = n;
          max = Some(n);
        }
      }

      Some(SeqInfo {
        name: String::new(),
        head: None,
        min_count: min,
        max_count: max,
        test: None,
        element_pattern: Some(Box::new(args[0].clone())),
        condition: None,
      })
    }
    _ => None,
  }
}

/// Apply a PatternTest function to an expression, returning true if it passes.
fn apply_pattern_test(test: &Expr, elem: &Expr) -> bool {
  let test_result = match test {
    Expr::Identifier(func_name) => {
      let call = Expr::FunctionCall {
        name: func_name.clone(),
        args: vec![elem.clone()],
      };
      evaluate_expr_to_expr(&call).ok()
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, &[elem.clone()]);
      evaluate_expr_to_expr(&substituted).ok()
    }
    _ => {
      let call_str =
        format!("({})[{}]", expr_to_string(test), expr_to_string(elem));
      interpret(&call_str).ok().map(|r| {
        if r == "True" {
          Expr::Identifier("True".to_string())
        } else {
          Expr::Identifier("False".to_string())
        }
      })
    }
  };
  matches!(test_result, Some(Expr::Identifier(ref s)) if s == "True")
}

/// Compute the minimum number of expression args required by a slice of patterns.
fn min_args_for_patterns(pats: &[Expr]) -> usize {
  pats
    .iter()
    .map(|p| {
      if let Some(seq) = get_sequence_info(p) {
        seq.min_count
      } else {
        1 // Non-sequence patterns always need exactly 1 arg
      }
    })
    .sum()
}

/// Match a slice of expression args against a slice of pattern args,
/// handling BlankSequence (__) and BlankNullSequence (___) patterns
/// that can consume variable numbers of arguments.
fn match_args_with_sequences(
  expr_args: &[Expr],
  pat_args: &[Expr],
) -> Option<Vec<(String, Expr)>> {
  // Base case: no more patterns
  if pat_args.is_empty() {
    return if expr_args.is_empty() {
      Some(vec![])
    } else {
      None
    };
  }

  let pat = &pat_args[0];
  let rest_pats = &pat_args[1..];

  if let Some(seq) = get_sequence_info(pat) {
    // Sequence pattern: try consuming different numbers of args
    let rest_min = min_args_for_patterns(rest_pats);
    let mut max_count = if expr_args.len() >= rest_min {
      expr_args.len() - rest_min
    } else {
      return None;
    };

    // Apply explicit max_count from Repeated[pat, {min, max}]
    if let Some(explicit_max) = seq.max_count
      && explicit_max < max_count
    {
      max_count = explicit_max;
    }

    if max_count < seq.min_count {
      return None;
    }

    for count in seq.min_count..=max_count {
      let seq_args = &expr_args[..count];

      // Check head constraints for all elements
      if let Some(ref h) = seq.head
        && !seq_args.iter().all(|a| get_expr_head(a) == *h)
      {
        continue;
      }

      // Check PatternTest for all elements
      if let Some(ref test) = seq.test
        && !seq_args.iter().all(|a| apply_pattern_test(test, a))
      {
        continue;
      }

      // Check element pattern for Repeated/RepeatedNull
      if let Some(ref elem_pat) = seq.element_pattern {
        let mut all_match = true;
        let mut elem_bindings: Vec<(String, Expr)> = vec![];
        for arg in seq_args {
          if let Some(b) = match_pattern(arg, elem_pat) {
            if !merge_bindings(&mut elem_bindings, b) {
              all_match = false;
              break;
            }
          } else {
            all_match = false;
            break;
          }
        }
        if !all_match {
          continue;
        }
        // Recursively match the rest
        if let Some(rest_bindings) =
          match_args_with_sequences(&expr_args[count..], rest_pats)
          && merge_bindings(&mut elem_bindings, rest_bindings)
        {
          // Add binding for this sequence name (if any)
          if !seq.name.is_empty() {
            let bound_value = if count == 0 {
              Expr::FunctionCall {
                name: "Sequence".to_string(),
                args: vec![],
              }
            } else if count == 1 {
              seq_args[0].clone()
            } else {
              Expr::FunctionCall {
                name: "Sequence".to_string(),
                args: seq_args.to_vec(),
              }
            };
            elem_bindings.insert(0, (seq.name.clone(), bound_value));
          }
          // Check Condition if present
          if let Some(ref cond) = seq.condition {
            let test_expr =
              apply_bindings(cond, &elem_bindings).unwrap_or(*cond.clone());
            match evaluate_expr_to_expr(&test_expr) {
              Ok(Expr::Identifier(ref s)) if s == "True" => {}
              _ => continue,
            }
          }
          return Some(elem_bindings);
        }
        continue;
      }

      // Recursively match the rest
      if let Some(mut rest_bindings) =
        match_args_with_sequences(&expr_args[count..], rest_pats)
      {
        // Add binding for this sequence
        if !seq.name.is_empty() {
          let bound_value = if count == 0 {
            Expr::FunctionCall {
              name: "Sequence".to_string(),
              args: vec![],
            }
          } else if count == 1 {
            seq_args[0].clone()
          } else {
            Expr::FunctionCall {
              name: "Sequence".to_string(),
              args: seq_args.to_vec(),
            }
          };
          rest_bindings.insert(0, (seq.name.clone(), bound_value));
        }
        // Check Condition if present
        if let Some(ref cond) = seq.condition {
          let test_expr =
            apply_bindings(cond, &rest_bindings).unwrap_or(*cond.clone());
          match evaluate_expr_to_expr(&test_expr) {
            Ok(Expr::Identifier(ref s)) if s == "True" => {}
            _ => continue,
          }
        }
        return Some(rest_bindings);
      }
    }
    None
  } else {
    // Non-sequence pattern: must match exactly one expr arg
    if expr_args.is_empty() {
      return None;
    }
    if let Some(mut bindings) = match_pattern(&expr_args[0], pat)
      && let Some(rest_bindings) =
        match_args_with_sequences(&expr_args[1..], rest_pats)
      && merge_bindings(&mut bindings, rest_bindings)
    {
      return Some(bindings);
    }
    None
  }
}

/// Match a pattern against an expression, returning bindings if successful
pub fn match_pattern(
  expr: &Expr,
  pattern: &Expr,
) -> Option<Vec<(String, Expr)>> {
  stacker::maybe_grow(2 * 1024 * 1024, 4 * 1024 * 1024, || {
    match_pattern_impl(expr, pattern)
  })
}

fn match_pattern_impl(
  expr: &Expr,
  pattern: &Expr,
) -> Option<Vec<(String, Expr)>> {
  // HoldPattern[p] is transparent to pattern matching: it only prevents
  // evaluation of the wrapped expression when parsed, and the matcher
  // should treat it as if it were just `p`.
  if let Expr::FunctionCall { name, args } = pattern
    && name == "HoldPattern"
    && args.len() == 1
  {
    return match_pattern_impl(expr, &args[0]);
  }
  match pattern {
    Expr::Pattern { name, head, .. } => {
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
    Expr::PatternTest {
      name, head, test, ..
    } => {
      // _?test or x_?test or x_Head?test — matches if head matches and test[expr] is True
      // Check head constraint first
      if let Some(h) = head {
        let expr_head = get_expr_head(expr);
        if expr_head != *h {
          return None;
        }
      }
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
        Some(Expr::Identifier(ref s)) if s == "True" => {
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
        // Check if any pattern item is a sequence pattern
        let has_sequence =
          pat_items.iter().any(|p| get_sequence_info(p).is_some());
        if has_sequence {
          match_args_with_sequences(expr_items, pat_items)
        } else {
          if pat_items.len() != expr_items.len() {
            return None;
          }
          let mut bindings = Vec::new();
          for (p, e) in pat_items.iter().zip(expr_items.iter()) {
            if let Some(b) = match_pattern(e, p) {
              if !merge_bindings(&mut bindings, b) {
                return None;
              }
            } else {
              return None;
            }
          }
          Some(bindings)
        }
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
    // Verbatim[expr] - matches literally, not treating contents as patterns
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } if pat_name == "Verbatim" && pat_args.len() == 1 => {
      if expr_to_string(expr) == expr_to_string(&pat_args[0]) {
        Some(vec![])
      } else {
        None
      }
    }
    // Except[c] - matches anything that doesn't match c
    // Except[c, pattern] - matches pattern but not c
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } if pat_name == "Except"
      && (pat_args.len() == 1 || pat_args.len() == 2) =>
    {
      if pat_args.len() == 2 {
        // Except[c, pattern]: pattern must match, c must NOT match
        if match_pattern(expr, &pat_args[0]).is_some() {
          None
        } else {
          match_pattern(expr, &pat_args[1])
        }
      } else if match_pattern(expr, &pat_args[0]).is_some() {
        None
      } else {
        Some(vec![])
      }
    }
    // Condition[pattern, test] - matches if pattern matches AND test evaluates to True
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } if pat_name == "Condition" && pat_args.len() == 2 => {
      // First match the pattern part
      if let Some(bindings) = match_pattern(expr, &pat_args[0]) {
        // Substitute bindings into the test expression and evaluate
        let test_expr = apply_bindings(&pat_args[1], &bindings)
          .unwrap_or(pat_args[1].clone());
        match evaluate_expr_to_expr(&test_expr) {
          Ok(Expr::Identifier(ref s)) if s == "True" => Some(bindings),
          _ => None,
        }
      } else {
        None
      }
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
        if pat_name != expr_name {
          return None;
        }
        // Check if any pattern arg is a sequence pattern
        let has_sequence =
          pat_args.iter().any(|p| get_sequence_info(p).is_some());
        if has_sequence {
          match_args_with_sequences(expr_args, pat_args)
        } else {
          if pat_args.len() != expr_args.len() {
            return None;
          }
          // For Orderless functions (Times, Plus), try all permutations
          let is_orderless =
            crate::evaluator::listable::is_builtin_orderless(pat_name);
          if is_orderless && pat_args.len() >= 2 {
            // Try all permutations of expression args against pattern args.
            // When Optional patterns are present, prefer matches where more
            // Optional patterns use their default values (Wolfram semantics).
            let perms = permutations(expr_args);
            let has_optionals = pat_args.iter().any(pattern_contains_optional);
            let mut best_match: Option<(Vec<(String, Expr)>, usize)> = None;
            for perm in perms {
              let mut bindings = Vec::new();
              let mut matched = true;
              for (p, e) in pat_args.iter().zip(perm.iter()) {
                push_match_context(&bindings);
                let result = match_pattern(e, p);
                pop_match_context();
                if let Some(b) = result {
                  if !merge_bindings(&mut bindings, b) {
                    matched = false;
                    break;
                  }
                } else {
                  matched = false;
                  break;
                }
              }
              if matched {
                if !has_optionals {
                  // No optionals — return first match immediately
                  return Some(bindings);
                }
                // Check compatibility with outer context bindings
                if !bindings_compatible_with_context(&bindings) {
                  continue;
                }
                let score =
                  count_optional_defaults_used(pat_name, pat_args, &bindings);
                if let Some((_, best_score)) = &best_match {
                  if score > *best_score {
                    best_match = Some((bindings, score));
                  }
                } else {
                  best_match = Some((bindings, score));
                }
              }
            }
            // If no context-compatible match found with Optional scoring,
            // fall back to first match without context check
            if best_match.is_none() {
              for perm in permutations(expr_args) {
                let mut bindings = Vec::new();
                let mut matched = true;
                for (p, e) in pat_args.iter().zip(perm.iter()) {
                  push_match_context(&bindings);
                  let result = match_pattern(e, p);
                  pop_match_context();
                  if let Some(b) = result {
                    if !merge_bindings(&mut bindings, b) {
                      matched = false;
                      break;
                    }
                  } else {
                    matched = false;
                    break;
                  }
                }
                if matched {
                  return Some(bindings);
                }
              }
              return None;
            }
            best_match.map(|(b, _)| b)
          } else {
            let mut bindings = Vec::new();
            for (p, e) in pat_args.iter().zip(expr_args.iter()) {
              push_match_context(&bindings);
              let result = match_pattern(e, p);
              pop_match_context();
              if let Some(b) = result {
                if !merge_bindings(&mut bindings, b) {
                  return None;
                }
              } else {
                return None;
              }
            }
            Some(bindings)
          }
        }
      } else {
        // Expression is not a FunctionCall with the same name;
        // try OneIdentity matching as fallback
        try_one_identity_match(expr, pat_name, pat_args)
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
        push_match_context(&bindings);
        let left_result = match_pattern(expr_left, pat_left);
        pop_match_context();
        if let Some(b) = left_result {
          if !merge_bindings(&mut bindings, b) {
            return None;
          }
        } else {
          return None;
        }
        push_match_context(&bindings);
        let right_result = match_pattern(expr_right, pat_right);
        pop_match_context();
        if let Some(b) = right_result {
          if !merge_bindings(&mut bindings, b) {
            return None;
          }
        } else {
          return None;
        }
        Some(bindings)
      } else {
        // Expression is not the same BinaryOp; try OneIdentity matching
        let func_name = binary_op_to_func_name(pat_op);
        if !func_name.is_empty() {
          try_one_identity_match(
            expr,
            func_name,
            &[*pat_left.clone(), *pat_right.clone()],
          )
        } else {
          None
        }
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
  use crate::syntax::{BinaryOperator, UnaryOperator};
  // Check for complex numbers before general matching,
  // so that e.g. 2I (stored as Times[2, I]) is recognized as Complex
  if crate::functions::predicate_ast::is_complex_number(expr) {
    return "Complex".to_string();
  }
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => "Integer".to_string(),
    Expr::Real(_) | Expr::BigFloat(_, _) => "Real".to_string(),
    Expr::String(_) => "String".to_string(),
    Expr::List(_) => "List".to_string(),
    Expr::FunctionCall { name, .. } => name.clone(),
    Expr::Association(_) => "Association".to_string(),
    Expr::BinaryOp { op, .. } => match op {
      BinaryOperator::Plus | BinaryOperator::Minus => "Plus".to_string(),
      BinaryOperator::Times => "Times".to_string(),
      BinaryOperator::Divide => "Times".to_string(),
      BinaryOperator::Power => "Power".to_string(),
      BinaryOperator::And => "And".to_string(),
      BinaryOperator::Or => "Or".to_string(),
      BinaryOperator::StringJoin => "StringJoin".to_string(),
      BinaryOperator::Alternatives => "Alternatives".to_string(),
    },
    Expr::UnaryOp { op, .. } => match op {
      UnaryOperator::Minus => "Times".to_string(),
      UnaryOperator::Not => "Not".to_string(),
    },
    Expr::Comparison { .. } => "Comparison".to_string(),
    Expr::CompoundExpr(_) => "CompoundExpression".to_string(),
    Expr::Rule { .. } => "Rule".to_string(),
    Expr::RuleDelayed { .. } => "RuleDelayed".to_string(),
    Expr::Map { .. } => "Map".to_string(),
    Expr::Apply { .. } => "Apply".to_string(),
    Expr::ReplaceAll { .. } => "ReplaceAll".to_string(),
    Expr::ReplaceRepeated { .. } => "ReplaceRepeated".to_string(),
    Expr::Function { .. } => "Function".to_string(),
    Expr::Part { .. } => "Part".to_string(),
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
  // Use simultaneous substitution to prevent variable name leakage
  let binding_refs: Vec<(&str, &Expr)> = bindings
    .iter()
    .map(|(name, value)| (name.as_str(), value))
    .collect();
  let result = crate::syntax::substitute_variables(replacement, &binding_refs);
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
      // Use simultaneous substitution to prevent variable name leakage
      let binding_refs: Vec<(&str, &Expr)> = bindings
        .iter()
        .filter(|(name, _)| !name.is_empty())
        .map(|(name, val)| (name.as_str(), val))
        .collect();
      let result =
        crate::syntax::substitute_variables(replacement, &binding_refs);
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
      let new_items = new_items?;
      // Check if any rule replaces the "List" head
      for (pattern, replacement) in rules {
        if let Expr::Identifier(sym) = pattern
          && sym == "List"
        {
          let new_head = match replacement {
            Expr::Identifier(h) => h.clone(),
            _ => expr_to_string(replacement),
          };
          return Ok(Expr::FunctionCall {
            name: new_head,
            args: new_items,
          });
        }
      }
      Ok(Expr::List(new_items))
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Result<Vec<Expr>, _> = args
        .iter()
        .map(|arg| apply_replace_all_multi_ast(arg, rules))
        .collect();
      let new_args = new_args?;
      // Check if any rule replaces the function head
      for (pattern, replacement) in rules {
        if let Expr::Identifier(sym) = pattern
          && sym == name
        {
          return match replacement {
            Expr::Identifier(h) => evaluate_expr_to_expr(&Expr::FunctionCall {
              name: h.clone(),
              args: new_args,
            }),
            // Non-symbol replacement: create CurriedCall
            _ => Ok(Expr::CurriedCall {
              func: Box::new((*replacement).clone()),
              args: new_args,
            }),
          };
        }
      }
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
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
