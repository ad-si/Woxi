#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// AST-based Select: filter elements where predicate returns True.
/// Select[{a, b, c}, pred] -> elements where pred[elem] is True
/// Select[{a, b, c}, pred, n] -> first n elements where pred[elem] is True
pub fn select_ast(
  list: &Expr,
  pred: &Expr,
  n: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let limit = match n {
    Some(expr) => match expr {
      Expr::Integer(i) => Some(*i as usize),
      _ => None,
    },
    None => None,
  };

  // Handle associations: Select on values, preserve key-value pairs
  if let Expr::Association(pairs) = list {
    let mut kept = Vec::new();
    for (key, val) in pairs {
      let result = apply_func_ast(pred, val)?;
      if expr_to_bool(&result) == Some(true) {
        kept.push((key.clone(), val.clone()));
        if let Some(lim) = limit
          && kept.len() >= lim
        {
          break;
        }
      }
    }
    return Ok(Expr::Association(kept));
  }

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

/// AST-based SelectFirst: first element where predicate returns True.
/// SelectFirst[list, pred] -> first matching element or Missing["NotFound"]
/// SelectFirst[list, pred, default] -> first matching element or default
pub fn select_first_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "SelectFirst expects 2 or 3 arguments".into(),
    ));
  }

  let list = &args[0];
  let pred = &args[1];
  let default = args.get(2);

  let items: &[Expr] = match list {
    Expr::List(items) => items.as_slice(),
    Expr::FunctionCall { args, .. } => args.as_slice(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SelectFirst".to_string(),
        args: args.to_vec(),
      });
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      return Ok(item.clone());
    }
  }

  // No match found
  match default {
    Some(d) => Ok(d.clone()),
    None => Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![Expr::String("NotFound".to_string())],
    }),
  }
}

/// AST-based FirstCase: return first element matching a pattern.
/// FirstCase[list, pattern] — returns first match or Missing["NotFound"]
/// FirstCase[list, pattern, default] — returns first match or default
/// FirstCase[list, pattern :> rhs] — returns rhs with bindings from first match
pub fn first_case_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let list = &args[0];
  let pattern = &args[1];
  let default = if args.len() >= 3 {
    Some(&args[2])
  } else {
    None
  };

  let items = match list {
    Expr::List(items) => items,
    _ => {
      // Non-list: return Missing["NotFound"] or default
      return Ok(default.cloned().unwrap_or_else(|| Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![Expr::String("NotFound".to_string())],
      }));
    }
  };

  // Check if pattern is a Rule or RuleDelayed: lhs -> rhs or lhs :> rhs
  let (match_pat, replacement) = extract_rule_parts(pattern);

  for item in items {
    if let Some(repl) = replacement {
      // Rule/RuleDelayed form: match against LHS, return RHS with bindings
      if let Some(bindings) =
        crate::evaluator::pattern_matching::match_pattern(item, match_pat)
      {
        let result =
          crate::evaluator::pattern_matching::apply_bindings(repl, &bindings)?;
        return Ok(result);
      }
    } else if matches_pattern_ast(item, match_pat) {
      return Ok(item.clone());
    }
  }

  // No match found
  Ok(default.cloned().unwrap_or_else(|| Expr::FunctionCall {
    name: "Missing".to_string(),
    args: vec![Expr::String("NotFound".to_string())],
  }))
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

  // Check if pattern is a Rule or RuleDelayed: lhs -> rhs or lhs :> rhs
  let (match_pat, replacement) = extract_rule_parts(pattern);

  let mut kept = Vec::new();
  for item in items {
    if let Some(repl) = replacement {
      // Rule/RuleDelayed form: match against LHS, return RHS with bindings
      if let Some(bindings) =
        crate::evaluator::pattern_matching::match_pattern(item, match_pat)
      {
        let result =
          crate::evaluator::pattern_matching::apply_bindings(repl, &bindings)?;
        kept.push(result);
      }
    } else if matches_pattern_ast(item, pattern) {
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

/// Extract pattern and optional replacement from a Rule or RuleDelayed expression.
/// Returns (pattern, Some(replacement)) for rules, or (original, None) for plain patterns.
fn extract_rule_parts(expr: &Expr) -> (&Expr, Option<&Expr>) {
  match expr {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => (pattern.as_ref(), Some(replacement.as_ref())),
    _ => (expr, None),
  }
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

/// Sequence pattern info for boolean matching.
struct SeqInfoBool {
  head: Option<String>,
  min_count: usize,
  max_count: Option<usize>,
  test: Option<Box<Expr>>,
  /// For Repeated/RepeatedNull: the element pattern to match against each arg.
  element_pattern: Option<Box<Expr>>,
}

/// Check if a pattern is a sequence pattern (BlankSequence or BlankNullSequence).
fn get_sequence_info_bool(pattern: &Expr) -> Option<SeqInfoBool> {
  match pattern {
    Expr::Pattern {
      head, blank_type, ..
    } if *blank_type >= 2 => {
      let min = if *blank_type == 2 { 1 } else { 0 };
      Some(SeqInfoBool {
        head: head.clone(),
        min_count: min,
        max_count: None,
        test: None,
        element_pattern: None,
      })
    }
    Expr::PatternTest {
      blank_type, test, ..
    } if *blank_type >= 2 => {
      let min = if *blank_type == 2 { 1 } else { 0 };
      Some(SeqInfoBool {
        head: None,
        min_count: min,
        max_count: None,
        test: Some(test.clone()),
        element_pattern: None,
      })
    }
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
      Some(SeqInfoBool {
        head,
        min_count: min,
        max_count: None,
        test: None,
        element_pattern: None,
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

      Some(SeqInfoBool {
        head: None,
        min_count: min,
        max_count: max,
        test: None,
        element_pattern: Some(Box::new(args[0].clone())),
      })
    }
    _ => None,
  }
}

/// Apply a PatternTest function to an expression.
fn apply_test_bool(test: &Expr, elem: &Expr) -> bool {
  let test_result = match test {
    Expr::Identifier(func_name) => {
      let call = Expr::FunctionCall {
        name: func_name.clone(),
        args: vec![elem.clone()],
      };
      crate::evaluator::evaluate_expr_to_expr(&call).ok()
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, &[elem.clone()]);
      crate::evaluator::evaluate_expr_to_expr(&substituted).ok()
    }
    _ => {
      let call_str = format!(
        "({})[{}]",
        crate::syntax::expr_to_string(test),
        crate::syntax::expr_to_string(elem)
      );
      crate::interpret(&call_str).ok().map(|r| {
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

/// Check if expression args match pattern args, handling sequence patterns.
fn args_match_with_sequences(expr_args: &[Expr], pat_args: &[Expr]) -> bool {
  if pat_args.is_empty() {
    return expr_args.is_empty();
  }

  let pat = &pat_args[0];
  let rest_pats = &pat_args[1..];

  if let Some(seq) = get_sequence_info_bool(pat) {
    let rest_min: usize = rest_pats
      .iter()
      .map(|p| {
        if let Some(s) = get_sequence_info_bool(p) {
          s.min_count
        } else {
          1
        }
      })
      .sum();
    let mut max_count = if expr_args.len() >= rest_min {
      expr_args.len() - rest_min
    } else {
      return false;
    };

    // Apply explicit max_count from Repeated[pat, {min, max}]
    if let Some(explicit_max) = seq.max_count
      && explicit_max < max_count
    {
      max_count = explicit_max;
    }

    if max_count < seq.min_count {
      return false;
    }

    for count in seq.min_count..=max_count {
      let seq_args = &expr_args[..count];
      // Check head constraints
      if let Some(ref h) = seq.head
        && !seq_args.iter().all(|a| get_expr_head_str(a) == h)
      {
        continue;
      }
      // Check PatternTest for all elements
      if let Some(ref test) = seq.test
        && !seq_args.iter().all(|a| apply_test_bool(test, a))
      {
        continue;
      }
      // Check element pattern for Repeated/RepeatedNull
      if let Some(ref elem_pat) = seq.element_pattern
        && !seq_args.iter().all(|a| matches_pattern_ast(a, elem_pat))
      {
        continue;
      }
      if args_match_with_sequences(&expr_args[count..], rest_pats) {
        return true;
      }
    }
    false
  } else {
    if expr_args.is_empty() {
      return false;
    }
    matches_pattern_ast(&expr_args[0], pat)
      && args_match_with_sequences(&expr_args[1..], rest_pats)
  }
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
      ..
    } => true,
    // Head-constrained pattern: _Integer, _List, etc.
    Expr::Pattern {
      name: _,
      head: Some(h),
      ..
    } => get_expr_head_str(expr) == h,
    // PatternTest: _?test or x_?test — matches if test[expr] is True
    Expr::PatternTest { test, .. } => {
      let test_result = match test.as_ref() {
        Expr::Identifier(func_name) => {
          let call = Expr::FunctionCall {
            name: func_name.clone(),
            args: vec![expr.clone()],
          };
          crate::evaluator::evaluate_expr_to_expr(&call).ok()
        }
        Expr::Function { body } => {
          // Anonymous function: substitute slots in the body (not the Function wrapper)
          let substituted =
            crate::syntax::substitute_slots(body, &[expr.clone()]);
          crate::evaluator::evaluate_expr_to_expr(&substituted).ok()
        }
        _ => {
          // General expression used as function: call (test)[expr]
          let call_str = format!(
            "({})[{}]",
            crate::syntax::expr_to_string(test),
            crate::syntax::expr_to_string(expr)
          );
          crate::interpret(&call_str).ok().map(|r| {
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
    // Blank[] or Blank[h] as FunctionCall (unevaluated form)
    Expr::FunctionCall { name, args } if name == "Blank" => match args.len() {
      0 => true,
      1 => {
        if let Expr::Identifier(h) = &args[0] {
          get_expr_head_str(expr) == h
        } else {
          false
        }
      }
      _ => false,
    },
    // Pattern[name, blank] as FunctionCall (unevaluated form)
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      matches_pattern_ast(expr, &args[1])
    }
    // Identifier patterns like "_", "_Integer", "_List", etc.
    Expr::Identifier(s) if s == "_" => true,
    Expr::Identifier(s) if s.starts_with('_') => {
      let head = &s[1..];
      get_expr_head_str(expr) == head
    }
    // Verbatim[expr] - matches literally, not treating contents as patterns
    Expr::FunctionCall { name, args }
      if name == "Verbatim" && args.len() == 1 =>
    {
      crate::syntax::expr_to_string(expr)
        == crate::syntax::expr_to_string(&args[0])
    }
    // Except[c] - matches anything that doesn't match c
    // Except[c, pattern] - matches pattern but not c
    Expr::FunctionCall { name, args }
      if name == "Except" && (args.len() == 1 || args.len() == 2) =>
    {
      if args.len() == 2 {
        // Except[c, pattern] - matches pattern but not c
        matches_pattern_ast(expr, &args[1])
          && !matches_pattern_ast(expr, &args[0])
      } else {
        !matches_pattern_ast(expr, &args[0])
      }
    }
    // Alternatives: a | b - matches if either side matches
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Alternatives,
      left,
      right,
    } => matches_pattern_ast(expr, left) || matches_pattern_ast(expr, right),
    // Alternatives[a, b, ...] as FunctionCall - matches if any alternative matches
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } if pat_name == "Alternatives" && !pat_args.is_empty() => {
      pat_args.iter().any(|alt| matches_pattern_ast(expr, alt))
    }
    // Structural matching for lists: {_, _} matches {1, 2}
    Expr::List(pat_items) => {
      if let Expr::List(expr_items) = expr {
        let has_seq = pat_items
          .iter()
          .any(|p| get_sequence_info_bool(p).is_some());
        if has_seq {
          args_match_with_sequences(expr_items, pat_items)
        } else {
          pat_items.len() == expr_items.len()
            && pat_items
              .iter()
              .zip(expr_items.iter())
              .all(|(p, e)| matches_pattern_ast(e, p))
        }
      } else {
        false
      }
    }
    // Condition[pattern, test] - matches if pattern matches AND test evaluates to True
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } if pat_name == "Condition" && pat_args.len() == 2 => {
      // Delegate to match_pattern which handles bindings and condition evaluation
      crate::evaluator::pattern_matching::match_pattern(expr, pattern).is_some()
    }
    // Structural matching for function calls: f[_
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } => {
      if pat_name == "Except" || pat_name == "PatternTest" {
        // Already handled above or not a structural match
        let pattern_str = crate::syntax::expr_to_string(pattern);
        let expr_str = crate::syntax::expr_to_string(expr);
        expr_str == pattern_str
      } else if let Expr::FunctionCall {
        name: expr_name,
        args: expr_args,
      } = expr
      {
        if pat_name != expr_name {
          return false;
        }
        // If any pattern arg is a Condition wrapping a sequence pattern,
        // delegate to the full match_pattern which handles condition evaluation
        // with proper sequence bindings.
        let has_condition_on_seq = pat_args.iter().any(|p| {
          if let Expr::FunctionCall { name: cn, args: ca } = p
            && cn == "Condition"
            && ca.len() == 2
          {
            get_sequence_info_bool(&ca[0]).is_some()
          } else {
            false
          }
        });
        if has_condition_on_seq {
          return crate::evaluator::pattern_matching::match_pattern(
            expr, pattern,
          )
          .is_some();
        }
        let has_seq =
          pat_args.iter().any(|p| get_sequence_info_bool(p).is_some());
        if has_seq {
          args_match_with_sequences(expr_args, pat_args)
        } else {
          pat_args.len() == expr_args.len()
            && pat_args
              .iter()
              .zip(expr_args.iter())
              .all(|(p, e)| matches_pattern_ast(e, p))
        }
      } else {
        false
      }
    }
    // Structural matching for BinaryOp: x^n_ matches x^2, etc.
    // (Alternatives BinaryOp is already handled above)
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
        pat_op == expr_op
          && matches_pattern_ast(expr_left, pat_left)
          && matches_pattern_ast(expr_right, pat_right)
      } else {
        false
      }
    }
    // Structural matching for UnaryOp
    Expr::UnaryOp {
      op: pat_op,
      operand: pat_operand,
    } => {
      if let Expr::UnaryOp {
        op: expr_op,
        operand: expr_operand,
      } = expr
      {
        pat_op == expr_op && matches_pattern_ast(expr_operand, pat_operand)
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

/// Cases with level specification: Cases[list, pattern, levelspec]
pub fn cases_with_level_ast(
  list: &Expr,
  pattern: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  let (min_level, max_level) = parse_level_spec(level_spec)?;

  let mut results = Vec::new();
  collect_at_level_range(
    list,
    pattern,
    0,
    min_level as usize,
    max_level as usize,
    &mut results,
  );
  Ok(Expr::List(results))
}

/// Recursively collect elements matching pattern within a level range
fn collect_at_level_range(
  expr: &Expr,
  pattern: &Expr,
  current_level: usize,
  min_level: usize,
  max_level: usize,
  results: &mut Vec<Expr>,
) {
  if current_level >= min_level
    && current_level <= max_level
    && matches_pattern_ast(expr, pattern)
  {
    results.push(expr.clone());
  }

  if current_level >= max_level {
    return;
  }

  // Recurse into sublists/subexpressions
  match expr {
    Expr::List(items) => {
      for item in items {
        collect_at_level_range(
          item,
          pattern,
          current_level + 1,
          min_level,
          max_level,
          results,
        );
      }
    }
    Expr::FunctionCall { args, .. } => {
      for arg in args {
        collect_at_level_range(
          arg,
          pattern,
          current_level + 1,
          min_level,
          max_level,
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
    Expr::FunctionCall { args, .. } => args.as_slice(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Position".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let mut positions = Vec::new();
  let mut path = Vec::new();
  position_recursive(items, pattern, &mut path, &mut positions);

  Ok(Expr::List(positions))
}

/// Recursively find positions of elements matching pattern
fn position_recursive(
  items: &[Expr],
  pattern: &Expr,
  path: &mut Vec<i128>,
  positions: &mut Vec<Expr>,
) {
  for (i, item) in items.iter().enumerate() {
    let idx = (i + 1) as i128;
    path.push(idx);

    // Check if this item matches
    if matches_pattern_ast(item, pattern) {
      positions
        .push(Expr::List(path.iter().map(|p| Expr::Integer(*p)).collect()));
    }

    // Recurse into sublists and function calls
    match item {
      Expr::List(sub_items) => {
        position_recursive(sub_items, pattern, path, positions);
      }
      Expr::FunctionCall { args, .. } => {
        position_recursive(args, pattern, path, positions);
      }
      _ => {}
    }

    path.pop();
  }
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

/// AST-based Count: count elements equal to pattern.
pub fn count_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  count_ast_level(list, pattern, None)
}

pub fn count_ast_level(
  list: &Expr,
  pattern: &Expr,
  level_spec: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let assoc_values: Vec<Expr>;
  let items = match list {
    Expr::List(items) => items.as_slice(),
    Expr::Association(pairs) => {
      assoc_values = pairs.iter().map(|(_, v)| v.clone()).collect();
      assoc_values.as_slice()
    }
    _ => {
      let mut args = vec![list.clone(), pattern.clone()];
      if let Some(ls) = level_spec {
        args.push(ls.clone());
      }
      return Ok(Expr::FunctionCall {
        name: "Count".to_string(),
        args,
      });
    }
  };

  // Parse level spec
  let (min_level, max_level) = match level_spec {
    None => (1usize, 1usize),
    Some(ls) => {
      let (min, max) = parse_level_spec(ls)?;
      (
        min.max(0) as usize,
        if max == i64::MAX {
          usize::MAX
        } else {
          max as usize
        },
      )
    }
  };

  let count = count_at_level(items, pattern, 1, min_level, max_level);
  Ok(Expr::Integer(count as i128))
}

fn count_at_level(
  items: &[Expr],
  pattern: &Expr,
  current_level: usize,
  min_level: usize,
  max_level: usize,
) -> usize {
  let mut count = 0;
  for item in items {
    if current_level >= min_level
      && current_level <= max_level
      && matches_pattern_ast(item, pattern)
    {
      count += 1;
    }
    // Recurse into sublists if we haven't reached max_level
    if current_level < max_level
      && let Expr::List(sub_items) = item
    {
      count += count_at_level(
        sub_items,
        pattern,
        current_level + 1,
        min_level,
        max_level,
      );
    }
  }
  count
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

  let mut removed = 0i128;
  let result: Vec<Expr> = items
    .iter()
    .filter(|item| {
      if let Some(max) = max_count
        && removed >= max
      {
        return true; // keep remaining items
      }
      if matches_pattern_ast(item, pattern) {
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

/// DeleteCases[list, pattern, levelspec] - delete matching elements at specified levels
pub fn delete_cases_with_level_ast(
  list: &Expr,
  pattern: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  let (min_level, max_level) = parse_level_spec(level_spec)?;
  let min = min_level.max(0) as usize;
  let max = if max_level == i64::MAX {
    usize::MAX
  } else {
    max_level as usize
  };
  Ok(delete_at_level_range(list, pattern, 0, min, max))
}

/// Recursively delete elements matching pattern within a level range
fn delete_at_level_range(
  expr: &Expr,
  pattern: &Expr,
  current_level: usize,
  min_level: usize,
  max_level: usize,
) -> Expr {
  match expr {
    Expr::List(items) => {
      let filtered: Vec<Expr> = items
        .iter()
        .filter(|item| {
          // Check if this item should be deleted (matches at current_level+1)
          let child_level = current_level + 1;
          !(child_level >= min_level
            && child_level <= max_level
            && matches_pattern_ast(item, pattern))
        })
        .map(|item| {
          // Recurse into sublists if we haven't reached max level
          if current_level + 1 < max_level {
            delete_at_level_range(
              item,
              pattern,
              current_level + 1,
              min_level,
              max_level,
            )
          } else {
            item.clone()
          }
        })
        .collect();
      Expr::List(filtered)
    }
    Expr::FunctionCall { name, args } => {
      let filtered: Vec<Expr> = args
        .iter()
        .filter(|item| {
          let child_level = current_level + 1;
          !(child_level >= min_level
            && child_level <= max_level
            && matches_pattern_ast(item, pattern))
        })
        .map(|item| {
          if current_level + 1 < max_level {
            delete_at_level_range(
              item,
              pattern,
              current_level + 1,
              min_level,
              max_level,
            )
          } else {
            item.clone()
          }
        })
        .collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: filtered,
      }
    }
    _ => expr.clone(),
  }
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
    Expr::Constant(s) if s == "Infinity" => Ok((1, i64::MAX)),
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
    Expr::Constant(s) if s == "Infinity" => Ok(i64::MAX),
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
        Ok(Expr::Identifier(ref s)) => s == "True",
        _ => false,
      }
    }
  }
}

// ─── PeakDetect ─────────────────────────────────────────────────────

/// PeakDetect[data] or PeakDetect[data, s]
/// Returns a list of 0s and 1s indicating peak positions.
/// Parameter s (default 0) controls sharpness filtering:
/// with s=0, finds all local peaks; higher s filters out less prominent peaks.
pub fn peak_detect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PeakDetect expects 1 or 2 arguments".into(),
    ));
  }

  let data = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PeakDetect".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let sharpness: usize = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(s) if *s >= 0 => *s as usize,
      _ => 0,
    }
  } else {
    0
  };

  let n = data.len();
  if n == 0 {
    return Ok(Expr::List(vec![]));
  }

  // Convert to f64 values
  let values: Vec<f64> = data
    .iter()
    .map(|e| crate::functions::math_ast::try_eval_to_f64(e).unwrap_or(f64::NAN))
    .collect();

  // Step 1: Find all local peaks
  // A point is a peak if it's >= both immediate neighbors AND
  // the plateau it belongs to (if any) has strictly lower values on both sides.
  let mut is_peak = vec![false; n];
  for i in 0..n {
    let val = values[i];
    let left_ok = i == 0 || val >= values[i - 1];
    let right_ok = i == n - 1 || val >= values[i + 1];

    if !(left_ok && right_ok) {
      continue;
    }

    // Find the extent of the plateau (consecutive equal values)
    let mut plateau_left = i;
    while plateau_left > 0 && values[plateau_left - 1] == val {
      plateau_left -= 1;
    }
    let mut plateau_right = i;
    while plateau_right < n - 1 && values[plateau_right + 1] == val {
      plateau_right += 1;
    }

    // Check that outside the plateau, values are strictly lower on at least one side
    let left_strict = plateau_left == 0 || values[plateau_left - 1] < val;
    let right_strict =
      plateau_right == n - 1 || values[plateau_right + 1] < val;
    // Both sides must be strictly lower (or at boundary)
    is_peak[i] = left_strict
      && right_strict
      && (plateau_left > 0 || plateau_right < n - 1); // not all equal
  }

  // Step 2: Apply sharpness filter — compute prominence and filter
  if sharpness > 0 {
    // Compute prominence for each peak
    let mut prominences: Vec<f64> = vec![0.0; n];
    for i in 0..n {
      if !is_peak[i] {
        continue;
      }
      let val = values[i];
      // Find the minimum of the highest points on each side
      // between this peak and the next higher peak (or boundary)
      let mut left_min = val;
      for j in (0..i).rev() {
        left_min = left_min.min(values[j]);
        if values[j] > val {
          break;
        }
      }
      let mut right_min = val;
      for j in (i + 1)..n {
        right_min = right_min.min(values[j]);
        if values[j] > val {
          break;
        }
      }
      prominences[i] = val - left_min.max(right_min);
    }

    // Sort prominences to find threshold
    let mut sorted_proms: Vec<f64> =
      prominences.iter().copied().filter(|p| *p > 0.0).collect();
    sorted_proms
      .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if !sorted_proms.is_empty() {
      // Use sharpness to determine how many peaks to keep
      // Higher sharpness removes less prominent peaks
      let threshold_idx = (sharpness).min(sorted_proms.len() - 1);
      let threshold = sorted_proms[threshold_idx];
      for i in 0..n {
        if is_peak[i] && prominences[i] < threshold {
          is_peak[i] = false;
        }
      }
    }
  }

  let result = is_peak
    .iter()
    .map(|&p| Expr::Integer(if p { 1 } else { 0 }))
    .collect();
  Ok(Expr::List(result))
}

/// Helper: generate index combinations of size k from 0..n, in lexicographic order.
fn generate_index_combinations(
  n: usize,
  k: usize,
  start: usize,
  current: &mut Vec<usize>,
  result: &mut Vec<Vec<usize>>,
) {
  if current.len() == k {
    result.push(current.clone());
    return;
  }
  for i in start..n {
    current.push(i);
    generate_index_combinations(n, k, i + 1, current, result);
    current.pop();
  }
}

/// Generate all permutations of a slice of indices.
fn permutations(indices: &[usize]) -> Vec<Vec<usize>> {
  if indices.len() <= 1 {
    return vec![indices.to_vec()];
  }
  let mut result = Vec::new();
  for (i, &idx) in indices.iter().enumerate() {
    let rest: Vec<usize> = indices
      .iter()
      .enumerate()
      .filter(|&(j, _)| j != i)
      .map(|(_, &v)| v)
      .collect();
    for mut perm in permutations(&rest) {
      perm.insert(0, idx);
      result.push(perm);
    }
  }
  result
}

/// Get the pattern size from a subset pattern (list length, or 1 for non-list patterns).
fn subset_pattern_size(pattern: &Expr) -> usize {
  match pattern {
    Expr::List(items) => items.len(),
    // Condition[{...}, test] — extract list length from inside Condition
    Expr::FunctionCall { name, args }
      if name == "Condition" && args.len() == 2 =>
    {
      subset_pattern_size(&args[0])
    }
    _ => 1,
  }
}

/// Build a list expr from items at given indices, for matching against subset pattern.
fn build_subset_expr(
  items: &[Expr],
  indices: &[usize],
  pattern_is_list: bool,
) -> Expr {
  if pattern_is_list {
    Expr::List(indices.iter().map(|&i| items[i].clone()).collect())
  } else {
    // Single-element pattern: match against the element directly
    items[indices[0]].clone()
  }
}

/// Check if a subset pattern is a list pattern (or Condition wrapping a list).
fn subset_pattern_is_list(pattern: &Expr) -> bool {
  match pattern {
    Expr::List(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Condition" && args.len() == 2 =>
    {
      subset_pattern_is_list(&args[0])
    }
    _ => false,
  }
}

/// Check if a subset (given by indices into items) matches pattern in some permutation.
/// Returns Some(matching_perm) if matched, None otherwise.
fn find_matching_permutation(
  items: &[Expr],
  indices: &[usize],
  pattern: &Expr,
  is_list: bool,
) -> Option<Vec<usize>> {
  if indices.len() <= 1 {
    // Only one permutation possible
    let subset_expr = build_subset_expr(items, indices, is_list);
    if matches_pattern_ast(&subset_expr, pattern)
      || crate::evaluator::pattern_matching::match_pattern(
        &subset_expr,
        pattern,
      )
      .is_some()
    {
      return Some(indices.to_vec());
    }
    return None;
  }

  // First try sorted order (most common case)
  let subset_expr = build_subset_expr(items, indices, is_list);
  if matches_pattern_ast(&subset_expr, pattern)
    || crate::evaluator::pattern_matching::match_pattern(&subset_expr, pattern)
      .is_some()
  {
    return Some(indices.to_vec());
  }

  // Try all other permutations
  for perm in permutations(indices) {
    if perm == indices {
      continue; // Already tried sorted order
    }
    let subset_expr = build_subset_expr(items, &perm, is_list);
    if matches_pattern_ast(&subset_expr, pattern)
      || crate::evaluator::pattern_matching::match_pattern(
        &subset_expr,
        pattern,
      )
      .is_some()
    {
      return Some(perm);
    }
  }
  None
}

/// SubsetPosition[list, pattern] — find positions of all subsets matching pattern.
/// Returns all matching subset index-lists (1-indexed), in the order that matched.
pub fn subset_position_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "SubsetPosition".to_string(),
        args: vec![list.clone(), pattern.clone()],
      });
    }
  };

  let k = subset_pattern_size(pattern);
  let is_list = subset_pattern_is_list(pattern);

  let mut index_combos = Vec::new();
  generate_index_combinations(
    items.len(),
    k,
    0,
    &mut vec![],
    &mut index_combos,
  );

  let mut results = Vec::new();
  for indices in &index_combos {
    if let Some(perm) =
      find_matching_permutation(items, indices, pattern, is_list)
    {
      // Return 1-indexed positions in the matching permutation order
      results.push(Expr::List(
        perm
          .iter()
          .map(|&i| Expr::Integer((i + 1) as i128))
          .collect(),
      ));
    }
  }

  Ok(Expr::List(results))
}

/// SubsetCases[list, pattern] — find non-overlapping subsets matching pattern (greedy).
/// SubsetCases[list, pattern, n] — find at most n non-overlapping matches.
pub fn subset_cases_ast(
  list: &Expr,
  pattern: &Expr,
  max_count: Option<usize>,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items,
    _ => {
      let mut a = vec![list.clone(), pattern.clone()];
      if let Some(n) = max_count {
        a.push(Expr::Integer(n as i128));
      }
      return Ok(Expr::FunctionCall {
        name: "SubsetCases".to_string(),
        args: a,
      });
    }
  };

  let k = subset_pattern_size(pattern);
  let is_list = subset_pattern_is_list(pattern);
  let limit = max_count.unwrap_or(usize::MAX);

  // Generate all subsets of size k in lexicographic order
  let mut index_combos = Vec::new();
  generate_index_combinations(
    items.len(),
    k,
    0,
    &mut vec![],
    &mut index_combos,
  );

  // Greedily select non-overlapping matches
  let mut used = vec![false; items.len()];
  let mut results = Vec::new();

  for indices in &index_combos {
    if results.len() >= limit {
      break;
    }
    // Skip if any index is already used
    if indices.iter().any(|&i| used[i]) {
      continue;
    }
    if let Some(perm) =
      find_matching_permutation(items, indices, pattern, is_list)
    {
      // Mark indices as used
      for &i in indices {
        used[i] = true;
      }
      // Return the subset elements in the matching permutation order
      results
        .push(Expr::List(perm.iter().map(|&i| items[i].clone()).collect()));
    }
  }

  Ok(Expr::List(results))
}

/// SubsetCount[list, pattern] — count non-overlapping subsets matching pattern.
pub fn subset_count_ast(
  list: &Expr,
  pattern: &Expr,
) -> Result<Expr, InterpreterError> {
  let result = subset_cases_ast(list, pattern, None)?;
  match &result {
    Expr::List(items) => Ok(Expr::Integer(items.len() as i128)),
    _ => Ok(Expr::Integer(0)),
  }
}
