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
        args: args.into(),
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
    Some(name) => Ok(Expr::FunctionCall {
      name,
      args: kept.into(),
    }),
    None => Ok(Expr::List(kept.into())),
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
        args: args.to_vec().into(),
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
      args: vec![Expr::String("NotFound".to_string())].into(),
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

  // 4-argument form FirstCase[expr, patt, default, levelspec]: the first
  // element matching `patt` at the given levels (or `default`). Delegate the
  // level traversal to Cases and take its first result.
  if args.len() == 4 {
    let cases = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Cases".to_string(),
      args: vec![list.clone(), pattern.clone(), args[3].clone()].into(),
    })?;
    if let Expr::List(matches) = &cases
      && let Some(first) = matches.first()
    {
      return Ok(first.clone());
    }
    return Ok(args[2].clone());
  }

  let default = if args.len() >= 3 {
    Some(&args[2])
  } else {
    None
  };

  // Scan the elements of a list or, for an association, its values.
  let items: Vec<&Expr> = match list {
    Expr::List(items) => items.iter().collect(),
    Expr::Association(pairs) => pairs.iter().map(|(_, v)| v).collect(),
    _ => {
      // Other heads: return Missing["NotFound"] or the default.
      return Ok(default.cloned().unwrap_or_else(|| Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![Expr::String("NotFound".to_string())].into(),
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
    args: vec![Expr::String("NotFound".to_string())].into(),
  }))
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
      head,
      blank_type,
      test,
      ..
    } if *blank_type >= 2 => {
      let min = if *blank_type == 2 { 1 } else { 0 };
      Some(SeqInfoBool {
        head: head.clone(),
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
        args: vec![elem.clone()].into(),
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
            args: vec![expr.clone()].into(),
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

/// AST-based Position: find positions of elements matching a pattern,
/// optionally restricted to a levelspec, and optionally limited to the
/// first `n` matches in scan order.
/// Wolfram-style depth of an expression: atoms have depth 1, composites
/// 1 + the maximum child depth (used for negative level specifications).
fn position_depth(expr: &Expr) -> i64 {
  use crate::functions::expr_form::{ExprForm, decompose_expr};
  match expr {
    Expr::Association(pairs) => {
      1 + pairs
        .iter()
        .map(|(_, v)| position_depth(v))
        .max()
        .unwrap_or(0)
    }
    _ => match decompose_expr(expr) {
      ExprForm::Atom(_) => 1,
      ExprForm::Composite { children, .. } => {
        1 + children.iter().map(position_depth).max().unwrap_or(0)
      }
    },
  }
}

/// Does a part at positive level `len` whose subexpression has Wolfram
/// depth `depth` fall within the level bounds? Negative bounds compare
/// against the negative level -depth.
fn position_level_match(len: usize, depth: i64, min: i64, max: i64) -> bool {
  let l = len as i64;
  let lower_ok = if min >= 0 { l >= min } else { depth <= -min };
  let upper_ok = if max >= 0 { l <= max } else { depth >= -max };
  lower_ok && upper_ok
}

/// Visit `expr` in Position's canonical order — head (when enabled),
/// children left to right, then the expression itself — recording the
/// paths of pattern matches. Returns false once the match limit is hit.
#[allow(clippy::too_many_arguments)]
fn position_visit(
  expr: &Expr,
  pattern: &Expr,
  path: &mut Vec<Expr>,
  out: &mut Vec<Expr>,
  min: i64,
  max: i64,
  heads: bool,
  limit: Option<usize>,
) -> bool {
  use crate::functions::expr_form::{ExprForm, decompose_expr};
  let full = |out: &Vec<Expr>| limit.is_some_and(|n| out.len() >= n);

  // Decompose into head + children; associations key their children.
  enum Kids {
    None,
    Indexed(Vec<Expr>),
    Keyed(Vec<(Expr, Expr)>),
  }
  let (head, kids): (Option<Expr>, Kids) = match expr {
    Expr::Association(pairs) => (
      Some(Expr::Identifier("Association".to_string())),
      Kids::Keyed(pairs.clone()),
    ),
    Expr::CurriedCall { func, args } => {
      (Some((**func).clone()), Kids::Indexed(args.clone()))
    }
    _ => match decompose_expr(expr) {
      ExprForm::Atom(_) => (None, Kids::None),
      ExprForm::Composite { head, children } => {
        (Some(Expr::Identifier(head)), Kids::Indexed(children))
      }
    },
  };

  if heads && let Some(h) = &head {
    path.push(Expr::Integer(0));
    let go = position_visit(h, pattern, path, out, min, max, heads, limit);
    path.pop();
    if !go {
      return false;
    }
  }

  match kids {
    Kids::None => {}
    Kids::Indexed(children) => {
      for (i, child) in children.iter().enumerate() {
        path.push(Expr::Integer((i + 1) as i128));
        let go =
          position_visit(child, pattern, path, out, min, max, heads, limit);
        path.pop();
        if !go {
          return false;
        }
      }
    }
    Kids::Keyed(pairs) => {
      for (key, value) in &pairs {
        path.push(Expr::FunctionCall {
          name: "Key".to_string(),
          args: vec![key.clone()].into(),
        });
        let go =
          position_visit(value, pattern, path, out, min, max, heads, limit);
        path.pop();
        if !go {
          return false;
        }
      }
    }
  }

  if position_level_match(path.len(), position_depth(expr), min, max)
    && matches_pattern_ast(expr, pattern)
  {
    out.push(Expr::List(path.clone().into()));
    if full(out) {
      return false;
    }
  }
  true
}

/// Visit `expr` in canonical order — head (when enabled), children left
/// to right, then the expression itself — collecting matched values (or
/// rule right-hand sides) at the requested levels. Returns false once
/// the match limit is hit.
#[allow(clippy::too_many_arguments)]
fn cases_visit(
  expr: &Expr,
  match_pat: &Expr,
  replacement: Option<&Expr>,
  level: usize,
  out: &mut Vec<Expr>,
  min: i64,
  max: i64,
  heads: bool,
  limit: Option<usize>,
) -> Result<bool, InterpreterError> {
  use crate::functions::expr_form::{ExprForm, decompose_expr};

  let (head, kids): (Option<Expr>, Vec<Expr>) = match expr {
    Expr::Association(pairs) => (
      Some(Expr::Identifier("Association".to_string())),
      pairs.iter().map(|(_, v)| v.clone()).collect(),
    ),
    Expr::CurriedCall { func, args } => (Some((**func).clone()), args.clone()),
    _ => match decompose_expr(expr) {
      ExprForm::Atom(_) => (None, Vec::new()),
      ExprForm::Composite { head, children } => {
        (Some(Expr::Identifier(head)), children)
      }
    },
  };

  if heads && let Some(h) = &head {
    let go = cases_visit(
      h,
      match_pat,
      replacement,
      level + 1,
      out,
      min,
      max,
      heads,
      limit,
    )?;
    if !go {
      return Ok(false);
    }
  }
  for child in &kids {
    let go = cases_visit(
      child,
      match_pat,
      replacement,
      level + 1,
      out,
      min,
      max,
      heads,
      limit,
    )?;
    if !go {
      return Ok(false);
    }
  }

  if position_level_match(level, position_depth(expr), min, max)
    && let Some(bindings) =
      crate::evaluator::pattern_matching::match_pattern(expr, match_pat)
  {
    match replacement {
      Some(repl) => out.push(
        crate::evaluator::pattern_matching::apply_bindings(repl, &bindings)?,
      ),
      None => out.push(expr.clone()),
    }
    if limit.is_some_and(|n| out.len() >= n) {
      return Ok(false);
    }
  }
  Ok(true)
}

/// Unified Cases covering the full argument space:
/// - `Cases[expr, pattern]` — level-1 parts matching the pattern
/// - `Cases[expr, pattern :> rhs]` — transformed matches
/// - `Cases[expr, pattern, levelspec]` — n, {n}, {m, n}, negative levels
/// - `Cases[expr, pattern, levelspec, n]` — at most n matches
/// - a trailing `Heads -> …` option (default Heads -> False)
///
/// Matches are emitted in canonical order (children before the
/// enclosing expression). Invalid level specs emit `::level`, invalid
/// counts `::innf`; both return the call unevaluated.
pub fn cases_unified_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Expr::FunctionCall {
    name: "Cases".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  // Strip a trailing Heads -> … option. Cases defaults to Heads -> False.
  let heads_setting = |e: &Expr| -> Option<bool> {
    let (lhs, rhs) = match e {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return None,
    };
    if matches!(lhs, Expr::Identifier(s) if s == "Heads") {
      Some(matches!(rhs, Expr::Identifier(v) if v == "True"))
    } else {
      None
    }
  };
  let mut heads = false;
  let mut positional: &[Expr] = args;
  // The pattern argument itself may be a rule, so only strip an option
  // from positions 3 and beyond.
  if positional.len() > 2
    && let Some(last) = positional.last()
    && let Some(h) = heads_setting(last)
  {
    heads = h;
    positional = &positional[..positional.len() - 1];
  }
  if positional.len() < 2 || positional.len() > 4 {
    return Ok(original());
  }
  let subject = &positional[0];
  let pattern = &positional[1];
  let (match_pat, replacement) = extract_rule_parts(pattern);

  let strict_int = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => i64::try_from(*n).ok(),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i64()
      }
      _ => None,
    }
  };
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };
  let bound = |e: &Expr| -> Option<i64> {
    if is_infinity(e) {
      Some(i64::MAX)
    } else {
      strict_int(e)
    }
  };

  let (min_level, max_level): (i64, i64) = match positional.get(2) {
    None => (1, 1),
    Some(spec) => {
      let parsed = match spec {
        Expr::List(items) if items.len() == 1 => {
          bound(&items[0]).map(|n| (n, n))
        }
        Expr::List(items) if items.len() == 2 => {
          match (bound(&items[0]), bound(&items[1])) {
            (Some(m), Some(n)) => Some((m, n)),
            _ => None,
          }
        }
        other => bound(other).map(|n| (1, n)),
      };
      match parsed {
        Some(b) => b,
        None => {
          crate::emit_message(&format!(
            "Cases::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
            show(spec)
          ));
          return Ok(original());
        }
      }
    }
  };

  let limit: Option<usize> = match positional.get(3) {
    None => None,
    Some(e) if is_infinity(e) => None,
    Some(e) => match strict_int(e) {
      Some(n) if n >= 0 => Some(n as usize),
      _ => {
        crate::emit_message(&format!(
          "Cases::innf: Non-negative integer or Infinity expected at position 4 in {}.",
          show(&original())
        ));
        return Ok(original());
      }
    },
  };
  if limit == Some(0) {
    return Ok(Expr::List(Vec::new().into()));
  }

  let mut out: Vec<Expr> = Vec::new();
  cases_visit(
    subject,
    match_pat,
    replacement,
    0,
    &mut out,
    min_level,
    max_level,
    heads,
    limit,
  )?;
  Ok(Expr::List(out.into()))
}

/// Unified Count: `Count[expr, pattern]`, `Count[expr, pattern,
/// levelspec]`, plus a trailing `Heads -> …` option (default False).
/// Counts matches on the same canonical walk as Cases. Invalid level
/// specs emit `::level`; a non-option fourth argument emits `::nonopt`.
pub fn count_unified_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Expr::FunctionCall {
    name: "Count".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  let heads_setting = |e: &Expr| -> Option<bool> {
    let (lhs, rhs) = match e {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return None,
    };
    if matches!(lhs, Expr::Identifier(s) if s == "Heads") {
      Some(matches!(rhs, Expr::Identifier(v) if v == "True"))
    } else {
      None
    }
  };
  let mut heads = false;
  let mut positional: &[Expr] = args;
  if positional.len() > 2
    && let Some(last) = positional.last()
    && let Some(h) = heads_setting(last)
  {
    heads = h;
    positional = &positional[..positional.len() - 1];
  }
  if positional.len() > 3 {
    crate::emit_message(&format!(
      "Count::nonopt: Options expected (instead of {}) beyond position 3 in {}. An option must be a rule or a list of rules.",
      show(&positional[positional.len() - 1]),
      show(&original())
    ));
    return Ok(original());
  }
  if positional.len() < 2 {
    return Ok(original());
  }
  let subject = &positional[0];
  let pattern = &positional[1];
  let (match_pat, replacement) = extract_rule_parts(pattern);

  let strict_int = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => i64::try_from(*n).ok(),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i64()
      }
      _ => None,
    }
  };
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };
  let bound = |e: &Expr| -> Option<i64> {
    if is_infinity(e) {
      Some(i64::MAX)
    } else {
      strict_int(e)
    }
  };

  let (min_level, max_level): (i64, i64) = match positional.get(2) {
    None => (1, 1),
    Some(spec) => {
      let parsed = match spec {
        Expr::List(items) if items.len() == 1 => {
          bound(&items[0]).map(|n| (n, n))
        }
        Expr::List(items) if items.len() == 2 => {
          match (bound(&items[0]), bound(&items[1])) {
            (Some(m), Some(n)) => Some((m, n)),
            _ => None,
          }
        }
        other => bound(other).map(|n| (1, n)),
      };
      match parsed {
        Some(b) => b,
        None => {
          crate::emit_message(&format!(
            "Count::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
            show(spec)
          ));
          return Ok(original());
        }
      }
    }
  };

  let mut out: Vec<Expr> = Vec::new();
  cases_visit(
    subject,
    match_pat,
    replacement,
    0,
    &mut out,
    min_level,
    max_level,
    heads,
    None,
  )?;
  Ok(Expr::Integer(out.len() as i128))
}

/// Unified Level: `Level[expr, levelspec]`, `Level[expr, levelspec, f]`
/// (f wraps the collected sequence and evaluates), plus a trailing
/// `Heads -> …` option (default False). Subexpressions are emitted in
/// canonical order (children before the enclosing expression). Invalid
/// level specs emit `::level` and return the call unevaluated.
pub fn level_unified_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Expr::FunctionCall {
    name: "Level".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  let heads_setting = |e: &Expr| -> Option<bool> {
    let (lhs, rhs) = match e {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return None,
    };
    if matches!(lhs, Expr::Identifier(s) if s == "Heads") {
      Some(matches!(rhs, Expr::Identifier(v) if v == "True"))
    } else {
      None
    }
  };
  let mut heads = false;
  let mut positional: &[Expr] = args;
  if positional.len() > 2
    && let Some(last) = positional.last()
    && let Some(h) = heads_setting(last)
  {
    heads = h;
    positional = &positional[..positional.len() - 1];
  }
  if positional.len() < 2 || positional.len() > 3 {
    return Ok(original());
  }
  let subject = &positional[0];
  let spec = &positional[1];
  let wrap = positional.get(2);

  let strict_int = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => i64::try_from(*n).ok(),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i64()
      }
      _ => None,
    }
  };
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };
  let bound = |e: &Expr| -> Option<i64> {
    if is_infinity(e) {
      Some(i64::MAX)
    } else {
      strict_int(e)
    }
  };
  let parsed = match spec {
    Expr::List(items) if items.len() == 1 => bound(&items[0]).map(|n| (n, n)),
    Expr::List(items) if items.len() == 2 => {
      match (bound(&items[0]), bound(&items[1])) {
        (Some(m), Some(n)) => Some((m, n)),
        _ => None,
      }
    }
    Expr::List(_) => None,
    other => bound(other).map(|n| (1, n)),
  };
  let Some((min_level, max_level)) = parsed else {
    crate::emit_message(&format!(
      "Level::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
      show(spec)
    ));
    return Ok(original());
  };

  let blank = Expr::FunctionCall {
    name: "Blank".to_string(),
    args: vec![].into(),
  };
  let mut out: Vec<Expr> = Vec::new();
  cases_visit(
    subject, &blank, None, 0, &mut out, min_level, max_level, heads, None,
  )?;

  match wrap {
    None => Ok(Expr::List(out.into())),
    Some(Expr::Identifier(f)) => {
      crate::evaluator::evaluate_function_call_ast(f, &out)
    }
    Some(f) => crate::evaluator::evaluate_expr_to_expr(&Expr::CurriedCall {
      func: Box::new(f.clone()),
      args: out,
    }),
  }
}

/// Rebuild `expr` with pattern matches at the requested levels removed.
/// A node is checked before its children (a deleted parent prunes the
/// subtree); returns None when the node itself is deleted.
fn delete_cases_walk(
  expr: &Expr,
  match_pat: &Expr,
  level: usize,
  min: i64,
  max: i64,
  remaining: &mut Option<usize>,
) -> Option<Expr> {
  use crate::functions::expr_form::{ExprForm, decompose_expr};

  if remaining.is_none_or(|n| n > 0)
    && position_level_match(level, position_depth(expr), min, max)
    && crate::evaluator::pattern_matching::match_pattern(expr, match_pat)
      .is_some()
  {
    if let Some(n) = remaining.as_mut() {
      *n -= 1;
    }
    return None;
  }

  match expr {
    Expr::Association(pairs) => {
      let kept: Vec<(Expr, Expr)> = pairs
        .iter()
        .filter_map(|(k, v)| {
          delete_cases_walk(v, match_pat, level + 1, min, max, remaining)
            .map(|nv| (k.clone(), nv))
        })
        .collect();
      Some(Expr::Association(kept))
    }
    Expr::CurriedCall { func, args } => {
      let kept: Vec<Expr> = args
        .iter()
        .filter_map(|a| {
          delete_cases_walk(a, match_pat, level + 1, min, max, remaining)
        })
        .collect();
      Some(Expr::CurriedCall {
        func: func.clone(),
        args: kept,
      })
    }
    _ => match decompose_expr(expr) {
      ExprForm::Atom(_) => Some(expr.clone()),
      ExprForm::Composite { head, children } => {
        let kept: Vec<Expr> = children
          .iter()
          .filter_map(|c| {
            delete_cases_walk(c, match_pat, level + 1, min, max, remaining)
          })
          .collect();
        if head == "List" {
          Some(Expr::List(kept.into()))
        } else {
          Some(Expr::FunctionCall {
            name: head,
            args: kept.into(),
          })
        }
      }
    },
  }
}

/// Unified DeleteCases: `DeleteCases[expr, pattern]`,
/// `DeleteCases[expr, pattern, levelspec]`, and
/// `DeleteCases[expr, pattern, levelspec, n]` (at most n deletions).
/// Deleting the whole expression (level 0) yields `Sequence[]`.
/// Invalid level specs emit `::level`, invalid counts `::innf`.
pub fn delete_cases_unified_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || Expr::FunctionCall {
    name: "DeleteCases".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  if args.len() < 2 || args.len() > 4 {
    return Ok(original());
  }
  let subject = &args[0];
  let pattern = &args[1];
  let (match_pat, _) = extract_rule_parts(pattern);

  let strict_int = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => i64::try_from(*n).ok(),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i64()
      }
      _ => None,
    }
  };
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };
  let bound = |e: &Expr| -> Option<i64> {
    if is_infinity(e) {
      Some(i64::MAX)
    } else {
      strict_int(e)
    }
  };

  let (min_level, max_level): (i64, i64) = match args.get(2) {
    None => (1, 1),
    Some(spec) => {
      let parsed = match spec {
        Expr::List(items) if items.len() == 1 => {
          bound(&items[0]).map(|n| (n, n))
        }
        Expr::List(items) if items.len() == 2 => {
          match (bound(&items[0]), bound(&items[1])) {
            (Some(m), Some(n)) => Some((m, n)),
            _ => None,
          }
        }
        other => bound(other).map(|n| (1, n)),
      };
      match parsed {
        Some(b) => b,
        None => {
          crate::emit_message(&format!(
            "DeleteCases::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
            show(spec)
          ));
          return Ok(original());
        }
      }
    }
  };

  let mut remaining: Option<usize> = match args.get(3) {
    None => None,
    Some(e) if is_infinity(e) => None,
    Some(e) => match strict_int(e) {
      Some(n) if n >= 0 => Some(n as usize),
      _ => {
        crate::emit_message(&format!(
          "DeleteCases::innf: Non-negative integer or Infinity expected at position 4 in {}.",
          show(&original())
        ));
        return Ok(original());
      }
    },
  };

  match delete_cases_walk(
    subject,
    match_pat,
    0,
    min_level,
    max_level,
    &mut remaining,
  ) {
    Some(result) => Ok(result),
    None => Ok(Expr::FunctionCall {
      name: "Sequence".to_string(),
      args: vec![].into(),
    }),
  }
}

/// Unified Position covering the full argument space:
/// - `Position[expr, pattern]` — all levels including the head ({0}) and
///   the whole expression ({}); Heads -> True is the default
/// - `Position[expr, pattern, levelspec]` — n, {n}, {m, n}, negative
///   levels counted from the leaves
/// - `Position[expr, pattern, levelspec, n]` — at most n positions
/// - a trailing `Heads -> …` option in any form
///
/// Invalid level specs emit `::level`, invalid counts `::innf`; both
/// return the call unevaluated.
pub fn position_unified_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Expr::FunctionCall {
    name: "Position".to_string(),
    args: args.to_vec().into(),
  };
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  // Strip a trailing Heads -> … option (default is Heads -> True).
  let heads_setting = |e: &Expr| -> Option<bool> {
    let (lhs, rhs) = match e {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => return None,
    };
    if matches!(lhs, Expr::Identifier(s) if s == "Heads") {
      Some(matches!(rhs, Expr::Identifier(v) if v == "True"))
    } else {
      None
    }
  };
  let mut heads = true;
  let mut positional: &[Expr] = args;
  if let Some(last) = positional.last()
    && let Some(h) = heads_setting(last)
  {
    heads = h;
    positional = &positional[..positional.len() - 1];
  }
  if positional.len() < 2 || positional.len() > 4 {
    return Ok(original());
  }
  let subject = &positional[0];
  let pattern = &positional[1];

  let strict_int = |e: &Expr| -> Option<i64> {
    match e {
      Expr::Integer(n) => i64::try_from(*n).ok(),
      Expr::BigInteger(n) => {
        use num_traits::ToPrimitive;
        n.to_i64()
      }
      _ => None,
    }
  };
  let is_infinity = |e: &Expr| -> bool {
    matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "DirectedInfinity" && args.len() == 1
        && matches!(&args[0], Expr::Integer(1)))
  };
  let bound = |e: &Expr| -> Option<i64> {
    if is_infinity(e) {
      Some(i64::MAX)
    } else {
      strict_int(e)
    }
  };

  let (min_level, max_level): (i64, i64) = match positional.get(2) {
    None => (0, i64::MAX),
    Some(spec) => {
      let parsed = match spec {
        Expr::List(items) if items.len() == 1 => {
          bound(&items[0]).map(|n| (n, n))
        }
        Expr::List(items) if items.len() == 2 => {
          match (bound(&items[0]), bound(&items[1])) {
            (Some(m), Some(n)) => Some((m, n)),
            _ => None,
          }
        }
        other => bound(other).map(|n| (1, n)),
      };
      match parsed {
        Some(b) => b,
        None => {
          crate::emit_message(&format!(
            "Position::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
            show(spec)
          ));
          return Ok(original());
        }
      }
    }
  };

  let limit: Option<usize> = match positional.get(3) {
    None => None,
    Some(e) if is_infinity(e) => None,
    Some(e) => match strict_int(e) {
      Some(n) if n >= 0 => Some(n as usize),
      _ => {
        crate::emit_message(&format!(
          "Position::innf: Non-negative integer or Infinity expected at position 4 in {}.",
          show(&original())
        ));
        return Ok(original());
      }
    },
  };
  if limit == Some(0) {
    return Ok(Expr::List(Vec::new().into()));
  }

  let mut out: Vec<Expr> = Vec::new();
  let mut path: Vec<Expr> = Vec::new();
  position_visit(
    subject, pattern, &mut path, &mut out, min_level, max_level, heads, limit,
  );
  Ok(Expr::List(out.into()))
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
      args: vec![Expr::String("NotFound".to_string())].into(),
    }
  };

  // Level bounds (lmin, lmax) measured as depth from the root (0 = the whole
  // expression). The default searches all levels; a 4th argument restricts
  // them: `n` -> levels 1..n, `{n}` -> exactly level n, `{a, b}` -> a..b,
  // `Infinity` -> all.
  const INF: i128 = i128::MAX;
  let (lmin, lmax) = if args.len() >= 4 {
    match parse_level_bounds(&args[3]) {
      Some(b) => b,
      None => (0, INF),
    }
  } else {
    (0, INF)
  };

  #[allow(clippy::too_many_arguments)]
  fn find_first(
    expr: &Expr,
    pattern: &Expr,
    path: &mut Vec<i128>,
    depth: i128,
    lmin: i128,
    lmax: i128,
  ) -> Option<Vec<i128>> {
    if depth >= lmin && depth <= lmax {
      let pattern_str = crate::syntax::expr_to_string(pattern);
      let expr_str = crate::syntax::expr_to_string(expr);
      if matches_pattern_simple(&expr_str, &pattern_str)
        || matches_pattern_ast(expr, pattern)
      {
        return Some(path.clone());
      }
    }
    if depth >= lmax {
      return None;
    }
    // Recurse into every structurally composite expression — List args,
    // FunctionCall args, BinaryOp operands, UnaryOp operand — so patterns
    // like `x^2` can be found inside `1 + x^2` (a Plus) at position {1, 2}.
    let recurse = |item: &Expr, idx: i128, path: &mut Vec<i128>| {
      path.push(idx);
      let r = find_first(item, pattern, path, depth + 1, lmin, lmax);
      path.pop();
      r
    };
    match expr {
      Expr::List(items) => {
        for (i, item) in items.iter().enumerate() {
          if let Some(result) = recurse(item, (i + 1) as i128, path) {
            return Some(result);
          }
        }
      }
      Expr::FunctionCall { args, .. } => {
        for (i, item) in args.iter().enumerate() {
          if let Some(result) = recurse(item, (i + 1) as i128, path) {
            return Some(result);
          }
        }
      }
      Expr::BinaryOp { left, right, .. } => {
        if let Some(result) = recurse(left, 1, path) {
          return Some(result);
        }
        if let Some(result) = recurse(right, 2, path) {
          return Some(result);
        }
      }
      Expr::UnaryOp { operand, .. } => {
        if let Some(result) = recurse(operand, 1, path) {
          return Some(result);
        }
      }
      _ => {}
    }
    None
  }

  let mut path = Vec::new();
  match find_first(&args[0], &args[1], &mut path, 0, lmin, lmax) {
    Some(indices) => {
      Ok(Expr::List(indices.into_iter().map(Expr::Integer).collect()))
    }
    None => Ok(default),
  }
}

/// Parse a Wolfram level specification into inclusive `(min, max)` depth
/// bounds. Returns `None` for unrecognized forms.
fn parse_level_bounds(spec: &Expr) -> Option<(i128, i128)> {
  const INF: i128 = i128::MAX;
  let as_level = |e: &Expr| -> Option<i128> {
    match e {
      Expr::Integer(n) => Some(*n),
      Expr::Identifier(s) if s == "Infinity" => Some(INF),
      _ => None,
    }
  };
  match spec {
    Expr::Integer(n) => Some((1, *n)),
    Expr::Identifier(s) if s == "Infinity" => Some((1, INF)),
    Expr::List(items) if items.len() == 1 => {
      let n = as_level(&items[0])?;
      Some((n, n))
    }
    Expr::List(items) if items.len() == 2 => {
      Some((as_level(&items[0])?, as_level(&items[1])?))
    }
    _ => None,
  }
}

/// AST-based TakeWhile: take elements while predicate is true.
pub fn take_while_ast(
  list: &Expr,
  pred: &Expr,
) -> Result<Expr, InterpreterError> {
  // On an association, the predicate is tested against each value and the
  // leading run of key->value pairs is kept.
  if let Expr::Association(pairs) = list {
    let mut result: Vec<(Expr, Expr)> = Vec::new();
    for (k, v) in pairs {
      let test_result = apply_func_ast(pred, v)?;
      if expr_to_bool(&test_result) == Some(true) {
        result.push((k.clone(), v.clone()));
      } else {
        break;
      }
    }
    return Ok(Expr::Association(result));
  }

  let items = match list {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TakeWhile".to_string(),
        args: vec![list.clone(), pred.clone()].into(),
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

  Ok(Expr::List(result.into()))
}

/// ContainsOnly[list, elems] - True if every element of list is in elems
pub fn contains_only_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "ContainsOnly expects 2 or 3 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ContainsOnly".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let elems = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ContainsOnly".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  // Extract optional SameTest option from the 3rd argument. Accepts either
  // {SameTest -> f} or a plain Rule SameTest -> f.
  let same_test: Option<Expr> = args.get(2).and_then(|opt| {
    let rules: Vec<&Expr> = match opt {
      Expr::List(items) => items.iter().collect(),
      Expr::Rule { .. } | Expr::RuleDelayed { .. } => vec![opt],
      _ => return None,
    };
    for r in rules {
      if let Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } = r
        && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "SameTest")
      {
        return Some((**replacement).clone());
      }
    }
    None
  });

  if let Some(test) = same_test {
    // With a SameTest, each list element must match at least one elem via
    // test[item, elem] evaluating to True.
    for item in list {
      let mut found = false;
      for elem in elems {
        let call = Expr::FunctionCall {
          name: crate::syntax::expr_to_string(&test),
          args: vec![item.clone(), elem.clone()].into(),
        };
        // Try treating `test` as a callable expression (e.g. Equal).
        let applied = match &test {
          Expr::Identifier(name) => Expr::FunctionCall {
            name: name.clone(),
            args: vec![item.clone(), elem.clone()].into(),
          },
          _ => call,
        };
        let result = crate::evaluator::evaluate_expr_to_expr(&applied)?;
        if matches!(&result, Expr::Identifier(s) if s == "True") {
          found = true;
          break;
        }
      }
      if !found {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
    return Ok(Expr::Identifier("True".to_string()));
  }

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
  let crit = &args[1];
  // On an association, count the leading run whose *values* satisfy the
  // criterion.
  let items: Vec<Expr> = match &args[0] {
    Expr::List(items) => items.to_vec(),
    Expr::Association(pairs) => pairs.iter().map(|(_, v)| v.clone()).collect(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LengthWhile".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  let mut count: i128 = 0;
  for item in &items {
    let test = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![crit.clone(), Expr::List(vec![item.clone()].into())].into(),
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
        args: result.into(),
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
      Ok(Expr::List(result.into()))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Pick".to_string(),
      args: if let Some(p) = pattern {
        vec![list.clone(), sel.clone(), p.clone()].into()
      } else {
        vec![list.clone(), sel.clone()].into()
      },
    }),
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
        args: vec![sel.clone(), pat.clone()].into(),
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
        args: args.to_vec().into(),
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
    return Ok(Expr::List(vec![].into()));
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
        args: vec![list.clone(), pattern.clone()].into(),
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

  Ok(Expr::List(results.into()))
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
        args: a.into(),
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

  Ok(Expr::List(results.into()))
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

/// CommonestFilter[data, r] - replace every element by the commonest
/// value in its radius-r neighborhood (clamped at the edges; 2D
/// rectangular arrays use a square window). Ties keep the center value
/// when it is among the maxima and otherwise take the first-occurring
/// maximum in window order. Nonpositive radii leave the data unchanged.
pub fn commonest_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| Expr::FunctionCall {
    name: "CommonestFilter".to_string(),
    args: args.to_vec().into(),
  };
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      crate::emit_message(&format!(
        "CommonestFilter::arg1: The first argument {} should be a rectangular array, image or video.",
        crate::syntax::expr_to_string(&args[0])
      ));
      return Ok(unevaluated(args));
    }
  };
  let r = match &args[1] {
    Expr::Integer(r) => *r,
    _ => {
      crate::emit_message(&format!(
        "CommonestFilter::bdrad: {} is not a valid neighborhood range specification.",
        crate::syntax::expr_to_string(&args[1])
      ));
      return Ok(unevaluated(args));
    }
  };
  if r <= 0 {
    return Ok(args[0].clone());
  }
  let r = r as usize;

  // Commonest in `window` with the tie rules; elements compare by
  // their printed form
  let pick = |window: &[&Expr], center: &Expr| -> Expr {
    let keys: Vec<String> = window
      .iter()
      .map(|e| crate::syntax::expr_to_string(e))
      .collect();
    let count_of = |key: &str| -> usize {
      keys.iter().filter(|k| k.as_str() == key).count()
    };
    let max_count = keys.iter().map(|k| count_of(k)).max().unwrap_or(0);
    let center_key = crate::syntax::expr_to_string(center);
    if count_of(&center_key) == max_count {
      return center.clone();
    }
    for (i, k) in keys.iter().enumerate() {
      if count_of(k) == max_count {
        return window[i].clone();
      }
    }
    center.clone()
  };

  // 2D rectangular array
  let is_matrix =
    !items.is_empty() && items.iter().all(|row| matches!(row, Expr::List(_)));
  if is_matrix {
    let rows: Vec<&[Expr]> = items
      .iter()
      .map(|row| match row {
        Expr::List(cells) => cells.as_slice(),
        _ => unreachable!(),
      })
      .collect();
    let width = rows[0].len();
    if rows.iter().any(|row| row.len() != width) {
      crate::emit_message(&format!(
        "CommonestFilter::arg1: The first argument {} should be a rectangular array, image or video.",
        crate::syntax::expr_to_string(&args[0])
      ));
      return Ok(unevaluated(args));
    }
    let result: Vec<Expr> = (0..rows.len())
      .map(|i| {
        let cells: Vec<Expr> = (0..width)
          .map(|j| {
            let mut window: Vec<&Expr> = Vec::new();
            for wi in i.saturating_sub(r)..=(i + r).min(rows.len() - 1) {
              for wj in j.saturating_sub(r)..=(j + r).min(width - 1) {
                window.push(&rows[wi][wj]);
              }
            }
            pick(&window, &rows[i][j])
          })
          .collect();
        Expr::List(cells.into())
      })
      .collect();
    return Ok(Expr::List(result.into()));
  }

  let n = items.len();
  let result: Vec<Expr> = (0..n)
    .map(|i| {
      let window: Vec<&Expr> = items[i.saturating_sub(r)..=(i + r).min(n - 1)]
        .iter()
        .collect();
      pick(&window, &items[i])
    })
    .collect();
  Ok(Expr::List(result.into()))
}
