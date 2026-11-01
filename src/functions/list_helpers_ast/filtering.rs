#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;
use std::borrow::Cow;

/// The optional count of `Select[list, crit, n]` / `Discard[list, crit, n]`:
/// `Ok(None)` when absent or `Infinity`, `Ok(Some(n))` for a non-negative
/// integer. Anything else (a negative or inexact number, `-Infinity`, a
/// symbol) yields `Err` with the unevaluated call, which is what WL returns,
/// after emitting `::innf` — unless the first argument is atomic, where WL
/// reports `::normal` instead.
fn filter_count_limit(
  name: &str,
  list: &Expr,
  crit: &Expr,
  n: Option<&Expr>,
) -> Result<Option<usize>, Expr> {
  use num_traits::{Signed, ToPrimitive};
  let Some(n) = n else { return Ok(None) };
  match n {
    Expr::Integer(i) if *i >= 0 => {
      return Ok(Some(usize::try_from(*i).unwrap_or(usize::MAX)));
    }
    Expr::BigInteger(i) if !i.is_negative() => {
      return Ok(Some(i.to_usize().unwrap_or(usize::MAX)));
    }
    Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity" => {
      return Ok(None);
    }
    Expr::FunctionCall { name, args }
      if name == "DirectedInfinity"
        && args.len() == 1
        && matches!(args[0], Expr::Integer(1)) =>
    {
      return Ok(None);
    }
    _ => {}
  }
  let args = vec![list.clone(), crit.clone(), n.clone()];
  if is_atomic_arg(list) {
    emit_nonatomic_normal_message(name, &args);
  } else {
    crate::emit_message(&format!(
      "{name}::innf: Non-negative integer or Infinity expected at position 3 in {}.",
      crate::syntax::format_expr(
        &call(name, args.clone()),
        crate::syntax::ExprForm::Output
      )
    ));
  }
  Err(call(name, args))
}

/// AST-based Select: filter elements where predicate returns True.
/// Select[{a, b, c}, pred] -> elements where pred[elem] is True
/// Select[{a, b, c}, pred, n] -> first n elements where pred[elem] is True
pub fn select_ast(
  list: &Expr,
  pred: &Expr,
  n: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let limit = match filter_count_limit("Select", list, pred, n) {
    Ok(limit) => limit,
    Err(original) => return Ok(original),
  };

  // Handle associations: Select on values, preserve key-value pairs
  if let Expr::Association(pairs) = list {
    let mut kept = Vec::new();
    for (key, val) in pairs {
      if limit.is_some_and(|lim| kept.len() >= lim) {
        break;
      }
      let result = apply_func_value(pred, val)?;
      if expr_to_bool(&result) == Some(true) {
        kept.push((key.clone(), val.clone()));
      }
    }
    return Ok(Expr::Association(kept));
  }

  // Select works on any expression with arguments, preserving the head.
  // Pure functions are ordinary expressions with head Function, so their
  // raw parts (Slots and parameter names intact) are filtered like any
  // other head's arguments: Select[(# > 0 &), crit] tests crit[#1 > 0].
  let owned_items: Vec<Expr>;
  let (items, head_name): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
    Expr::Function { body } => {
      owned_items = vec![(**body).clone()];
      (owned_items.as_slice(), Some("Function".to_string()))
    }
    Expr::NamedFunction {
      params,
      body,
      bracketed,
    } => {
      let mut parts: Vec<Expr> = Vec::with_capacity(params.len() + 1);
      if *bracketed {
        // Function[{x, y}, body] has two parts: the parameter list and
        // the body.
        parts.push(Expr::List(
          params
            .iter()
            .map(|p| Expr::Identifier(p.clone()))
            .collect::<Vec<_>>()
            .into(),
        ));
      } else {
        parts.extend(params.iter().map(|p| Expr::Identifier(p.clone())));
      }
      parts.push((**body).clone());
      owned_items = parts;
      (owned_items.as_slice(), Some("Function".to_string()))
    }
    _ => {
      let mut args = vec![list.clone(), pred.clone()];
      if let Some(limit) = n {
        args.push(limit.clone());
      }
      // An atomic first argument is invalid: emit ::normal, matching WL.
      if is_atomic_arg(list) {
        emit_nonatomic_normal_message("Select", &args);
      }
      return Ok(call("Select", args));
    }
  };

  let mut kept = Vec::new();
  for item in items {
    if limit.is_some_and(|lim| kept.len() >= lim) {
      break;
    }
    let result = apply_func_value(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      kept.push(item.clone());
    }
  }

  // Preserve the original head, then evaluate the rebuilt expression so a
  // head with its own rules reduces (e.g. Select[a + b + 2, NumberQ] keeps
  // Plus[2], which evaluates to 2). Inert heads stay symbolic.
  match head_name {
    Some(name) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name,
        args: kept.into(),
      })
    }
    None => Ok(Expr::List(kept.into())),
  }
}

/// AST-based Discard: the complement of Select — keep elements for which the
/// criterion is NOT True.
/// Discard[{a, b, c}, crit] -> elements where crit[elem] is not True
/// Discard[{a, b, c}, crit, n] -> discard only the first n matching elements
pub fn discard_ast(
  list: &Expr,
  crit: &Expr,
  n: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let limit = match filter_count_limit("Discard", list, crit, n) {
    Ok(limit) => limit,
    Err(original) => return Ok(original),
  };

  // Associations: test the criterion against the values, keep non-matching
  // key-value pairs.
  if let Expr::Association(pairs) = list {
    let mut kept = Vec::new();
    let mut discarded = 0usize;
    for (key, val) in pairs {
      let is_match = expr_to_bool(&apply_func_ast(crit, val)?) == Some(true);
      let drop = is_match && limit.is_none_or(|lim| discarded < lim);
      if drop {
        discarded += 1;
      } else {
        kept.push((key.clone(), val.clone()));
      }
    }
    return Ok(Expr::Association(kept));
  }

  // Works on any expression with arguments, preserving the head.
  let (items, head_name): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
    _ => {
      let mut args = vec![list.clone(), crit.clone()];
      if let Some(limit) = n {
        args.push(limit.clone());
      }
      if is_atomic_arg(list) {
        emit_nonatomic_normal_message("Discard", &args);
      }
      return Ok(call("Discard", args));
    }
  };

  let mut kept = Vec::new();
  let mut discarded = 0usize;
  for item in items {
    let is_match = expr_to_bool(&apply_func_ast(crit, item)?) == Some(true);
    let drop = is_match && limit.is_none_or(|lim| discarded < lim);
    if drop {
      discarded += 1;
    } else {
      kept.push(item.clone());
    }
  }

  match head_name {
    Some(name) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name,
        args: kept.into(),
      })
    }
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

  let not_found = || match default {
    Some(d) => d.clone(),
    None => call1("Missing", str_expr("NotFound")),
  };

  // On an association, the predicate is tested against the values and the first
  // matching value is returned.
  if let Expr::Association(pairs) = list {
    for (_k, v) in pairs {
      if expr_to_bool(&apply_func_ast(pred, v)?) == Some(true) {
        return Ok(v.clone());
      }
    }
    return Ok(not_found());
  }

  let items: &[Expr] = match list {
    Expr::List(items) => items.as_slice(),
    Expr::FunctionCall { args, .. } => args.as_slice(),
    _ => {
      return Ok(unevaluated("SelectFirst", args));
    }
  };

  for item in items {
    let result = apply_func_ast(pred, item)?;
    if expr_to_bool(&result) == Some(true) {
      return Ok(item.clone());
    }
  }

  Ok(not_found())
}

/// AST-based FirstCase: return first element matching a pattern.
/// FirstCase[list, pattern] — returns first match or Missing["NotFound"]
/// FirstCase[list, pattern, default] — returns first match or default
/// FirstCase[list, pattern :> rhs] — returns rhs with bindings from first match
/// Whether `e` is an atom, a list or an association — anything else is a
/// compound expression whose parts FirstCase searches like a list's.
fn is_atom_or_list(e: &Expr) -> bool {
  matches!(
    e,
    Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(..)
      | Expr::String(_)
      | Expr::Identifier(_)
      | Expr::List(_)
      | Expr::Association(_)
  )
}

/// Whether `e` is a `Heads -> …` option.
fn is_heads_option(e: &Expr) -> bool {
  matches!(e, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
    if matches!(pattern.as_ref(), Expr::Identifier(h) if h == "Heads"))
}

pub fn first_case_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let list = &args[0];
  let pattern = &args[1];

  // FirstCase[expr, patt, default, levelspec, Heads -> …]: the first element
  // matching `patt` at the given levels (by default {1}), or `default`. A
  // levelspec, a `Heads` option or a non-list expression delegates the level
  // traversal to Cases and takes its first result.
  let n_opts = args[2..]
    .iter()
    .rev()
    .take_while(|a| is_heads_option(a))
    .count();
  let (positional, opts) = args.split_at(args.len() - n_opts);
  if positional.len() == 4 || !opts.is_empty() || !is_atom_or_list(list) {
    let levelspec = positional
      .get(3)
      .cloned()
      .unwrap_or_else(|| Expr::List(vec![Expr::Integer(1)].into()));
    let mut cases_args = vec![list.clone(), pattern.clone(), levelspec];
    cases_args.extend(opts.iter().cloned());
    let cases =
      crate::evaluator::evaluate_expr_to_expr(&call("Cases", cases_args))?;
    if let Expr::List(matches) = &cases
      && let Some(first) = matches.first()
    {
      return Ok(first.clone());
    }
    return Ok(
      positional
        .get(2)
        .cloned()
        .unwrap_or_else(|| call1("Missing", str_expr("NotFound"))),
    );
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
      return Ok(
        default
          .cloned()
          .unwrap_or_else(|| call1("Missing", str_expr("NotFound"))),
      );
    }
  };

  // Check if pattern is a Rule or RuleDelayed: lhs -> rhs or lhs :> rhs
  let (match_pat, replacement) = extract_rule_parts(pattern);

  for item in items {
    if let Some(repl) = replacement {
      // Rule/RuleDelayed form: match against LHS, return RHS with bindings
      if let Some(bindings) =
        crate::evaluator::pattern_matching::match_pattern(item, match_pat)
        && let Some(result) =
          crate::evaluator::pattern_matching::instantiate_replacement(
            repl, &bindings,
          )?
      {
        return Ok(result);
      }
    } else if matches_pattern_ast(item, match_pat) {
      return Ok(item.clone());
    }
  }

  // No match found
  Ok(
    default
      .cloned()
      .unwrap_or_else(|| call1("Missing", str_expr("NotFound"))),
  )
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
      let min = usize::from(*blank_type == 2);
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
      let min = usize::from(*blank_type == 2);
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
      let min = usize::from(name == "BlankSequence");
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
      let substituted =
        crate::syntax::substitute_slots(body, std::slice::from_ref(elem));
      crate::evaluator::evaluate_expr_to_expr(&substituted).ok()
    }
    _ => {
      let call_str =
        format!("({})[{}]", expr_to_string(test), expr_to_string(elem));
      crate::interpret(&call_str).ok().map(|r| {
        if r == "True" {
          bool_expr(true)
        } else {
          bool_expr(false)
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
            crate::syntax::substitute_slots(body, std::slice::from_ref(expr));
          crate::evaluator::evaluate_expr_to_expr(&substituted).ok()
        }
        _ => {
          // General expression used as function: call (test)[expr]
          let call_str =
            format!("({})[{}]", expr_to_string(test), expr_to_string(expr));
          crate::interpret(&call_str).ok().map(|r| {
            if r == "True" {
              bool_expr(true)
            } else {
              bool_expr(false)
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
    // HoldPattern[p] matches what p matches; it only keeps p from evaluating
    Expr::FunctionCall { name, args }
      if name == "HoldPattern" && args.len() == 1 =>
    {
      matches_pattern_ast(expr, &args[0])
    }
    // Verbatim[expr] - matches literally, not treating contents as patterns
    Expr::FunctionCall { name, args }
      if name == "Verbatim" && args.len() == 1 =>
    {
      expr_to_string(expr) == expr_to_string(&args[0])
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
      op: BinaryOperator::Alternatives,
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
      if pat_name == "PatternTest" && pat_args.len() == 2 {
        // `Except[0]?NumericQ` and friends — any pattern constrained by a
        // test. Delegate rather than re-deriving the rules here.
        crate::evaluator::pattern_matching::match_pattern(expr, pattern)
          .is_some()
      } else if pat_name == "Except" || pat_name == "PatternTest" {
        // Already handled above or not a structural match
        let pattern_str = expr_to_string(pattern);
        let expr_str = expr_to_string(expr);
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
    // Any other shape holding a pattern (`_ -> 2`, `x_ == y_`, an
    // association or a curried call with blanks) goes to the full matcher;
    // without one it is a literal, compared as written.
    _ if crate::evaluator::pattern_matching::contains_pattern(pattern) => {
      crate::evaluator::pattern_matching::match_pattern(expr, pattern).is_some()
    }
    _ => {
      let pattern_str = expr_to_string(pattern);
      let expr_str = expr_to_string(expr);
      expr_str == pattern_str
    }
  }
}

/// AST-based Position: find positions of elements matching a pattern,
/// optionally restricted to a levelspec, and optionally limited to the
/// first `n` matches in scan order.
/// Wolfram-style depth of an expression: atoms have depth 1, composites
/// 1 + the maximum child depth (used for negative level specifications).
/// Rational and Complex numbers are atomic for Level / Depth / Cases /
/// Position purposes (matching Wolfram's AtomQ), even though Woxi stores a
/// rational as a `Rational[n, d]` call and a complex as a Plus-Times tree.
/// They must therefore not be descended into when enumerating levels. The
/// packed-array objects (SparseArray, ByteArray, NumericArray) are AtomQ too,
/// so structural traversal must not index into their stored representation
/// (e.g. Position/Count/Cases on a SparseArray look at no parts).
fn is_atomic_number(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, .. }
    if name == "Rational"
      || name == "SparseArray"
      || name == "ByteArray"
      || name == "NumericArray")
    || crate::functions::predicate_ast::is_complex_number(expr)
}

fn position_depth(expr: &Expr) -> i64 {
  match expr {
    Expr::Association(pairs) => {
      1 + pairs
        .iter()
        .map(|(_, v)| position_depth(v))
        .max()
        .unwrap_or(0)
    }
    _ => match walk_parts(expr, true) {
      (None, _) => 1,
      // An empty `f[]` is as deep as `Depth[f[]]` says (2): the empty
      // maximum is the depth of an atom, not 0.
      (Some(_), children) => {
        1 + children.iter().map(position_depth).max().unwrap_or(1)
      }
    },
  }
}

/// The head (when `want_head`; always `Some` for a compound, so `None`
/// means an atom) and the indexed children of a non-association `expr`, as
/// the Cases/Position walks see them. A list's or call's arguments are lent
/// rather than copied: copying every subtree at every level made the walks
/// quadratic in the depth.
fn walk_parts(expr: &Expr, want_head: bool) -> (Option<Expr>, Cow<'_, [Expr]>) {
  use crate::functions::expr_form::{ExprForm, decompose_expr};
  let head = |name: &str| {
    Some(Expr::Identifier(
      if want_head { name } else { "" }.to_string(),
    ))
  };
  match expr {
    // Rational / Complex are atoms: do not descend into their parts.
    _ if is_atomic_number(expr) => (None, Cow::Borrowed(&[])),
    Expr::CurriedCall { func, args } => {
      (Some((**func).clone()), Cow::Owned(args.clone()))
    }
    Expr::List(items) => (head("List"), Cow::Borrowed(&items[..])),
    // `-Infinity`, stored as `Times[-1, Infinity]`, is the atom-like
    // `DirectedInfinity[-1]` to `decompose_expr`.
    Expr::FunctionCall { name, args }
      if !(name == "Times"
        && args.len() == 2
        && matches!(args[0], Expr::Integer(-1))
        && matches!(&args[1], Expr::Identifier(s) | Expr::Constant(s) if s == "Infinity")) =>
    {
      (head(name), Cow::Borrowed(&args[..]))
    }
    _ => match decompose_expr(expr) {
      ExprForm::Atom(_) => (None, Cow::Borrowed(&[])),
      ExprForm::Composite { head, children } => {
        (Some(Expr::Identifier(head)), Cow::Owned(children))
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
  let full = |out: &Vec<Expr>| limit.is_some_and(|n| out.len() >= n);

  // Decompose into head + children; associations key their children.
  enum Kids<'a> {
    Indexed(Cow<'a, [Expr]>),
    Keyed(&'a [(Expr, Expr)]),
  }
  let (head, kids): (Option<Expr>, Kids) =
    if let Expr::Association(pairs) = expr {
      (Some(id_expr("Association")), Kids::Keyed(pairs))
    } else {
      let (head, children) = walk_parts(expr, heads);
      (head, Kids::Indexed(children))
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
      for (key, value) in pairs {
        path.push(call1("Key", key.clone()));
        let go =
          position_visit(value, pattern, path, out, min, max, heads, limit);
        path.pop();
        if !go {
          return false;
        }
      }
    }
  }

  // The depth (a walk of the whole subtree) only matters to a negative bound.
  let depth = if min < 0 || max < 0 {
    position_depth(expr)
  } else {
    0
  };
  if position_level_match(path.len(), depth, min, max)
    && crate::evaluator::pattern_matching::match_pattern(expr, pattern)
      .is_some()
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
  // Parts deeper than a non-negative `max` level can never match, so don't
  // even decompose (clone) them. Negative bounds count from the leaves and
  // need the full walk.
  let too_deep = max >= 0 && level as i64 >= max;
  let (head, kids): (Option<Expr>, Cow<[Expr]>) = match expr {
    _ if too_deep => (None, Cow::Borrowed(&[])),
    Expr::Association(pairs) => (
      Some(id_expr("Association")),
      pairs.iter().map(|(_, v)| v.clone()).collect(),
    ),
    _ => walk_parts(expr, heads),
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
  for child in kids.iter() {
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

  // The depth (a walk of the whole subtree) only matters to a negative bound.
  let depth = if min < 0 || max < 0 {
    position_depth(expr)
  } else {
    0
  };
  if position_level_match(level, depth, min, max)
    && let Some(bindings) =
      crate::evaluator::pattern_matching::match_pattern(expr, match_pat)
  {
    match replacement {
      Some(repl) => {
        if let Some(result) =
          crate::evaluator::pattern_matching::instantiate_replacement(
            repl, &bindings,
          )?
        {
          out.push(result);
        }
      }
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
  let original = || unevaluated("Cases", args);
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
  // A rendered graphic (a plot's `Graphics[…]`) is searched through the
  // symbolic expression it was built from, like `Part`/`First`/`Length`
  // do — `Cases[Plot[…], Line[u_] -> u, Infinity]` finds the curve points.
  let graphic_form = if let Expr::Graphics { .. } = &positional[0] {
    crate::evaluator::part_extraction::graphics_symbolic_form(&positional[0])
  } else {
    None
  };
  let subject = graphic_form.as_ref().unwrap_or(&positional[0]);
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
      if let Some(b) = parsed {
        b
      } else {
        crate::emit_message(&format!(
          "Cases::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
          show(spec)
        ));
        return Ok(original());
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
  let original = || unevaluated("Count", args);
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
  // A rendered graphic is counted through its symbolic expression, as
  // `Cases` searches it.
  let graphic_form = if let Expr::Graphics { .. } = &positional[0] {
    crate::evaluator::part_extraction::graphics_symbolic_form(&positional[0])
  } else {
    None
  };
  let subject = graphic_form.as_ref().unwrap_or(&positional[0]);
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
      if let Some(b) = parsed {
        b
      } else {
        crate::emit_message(&format!(
          "Count::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
          show(spec)
        ));
        return Ok(original());
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
  let original = || unevaluated("Level", args);
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

  let blank = call0("Blank");
  let mut out: Vec<Expr> = Vec::new();
  // A packed array object, a tree or a dataset is an atom: nothing inside it
  // is a level. The object itself sits at level 0 counted from the top and at
  // level -1 counted from the bottom, so a spec covering either reports it.
  if crate::functions::predicate_ast::is_atomic_traversal_object(subject) {
    let covers = |l: i64| min_level <= l && max_level >= l;
    if covers(0) || covers(-1) {
      out.push(subject.clone());
    }
  } else {
    cases_visit(
      subject, &blank, None, 0, &mut out, min_level, max_level, heads, None,
    )?;
  }

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

/// Outcome of walking one part in DeleteCases.
enum DeleteWalk {
  /// The part matched and is removed.
  Deleted,
  /// The part is kept (possibly with deletions inside it).
  Keep(Expr),
  /// The part's head matched (`Heads -> True`): its arguments are spliced
  /// into the parent, as `Sequence[args]` would be.
  Splice(Vec<Expr>),
}

/// Walk each child of a node one level down, collecting what is kept.
fn delete_cases_children(
  children: &[Expr],
  match_pat: &Expr,
  level: usize,
  min: i64,
  max: i64,
  heads: bool,
  remaining: &mut Option<usize>,
) -> Vec<Expr> {
  let mut kept = Vec::with_capacity(children.len());
  for c in children {
    match delete_cases_walk(c, match_pat, level + 1, min, max, heads, remaining)
    {
      DeleteWalk::Deleted => {}
      DeleteWalk::Keep(e) => kept.push(e),
      DeleteWalk::Splice(items) => kept.extend(items),
    }
  }
  kept
}

/// Rebuild `expr` with pattern matches at the requested levels removed.
/// A node is checked before its head and children (a deleted parent
/// prunes the subtree). With `heads`, a matching head (one level below
/// its node) is deleted too, leaving the node's arguments as a sequence.
fn delete_cases_walk(
  expr: &Expr,
  match_pat: &Expr,
  level: usize,
  min: i64,
  max: i64,
  heads: bool,
  remaining: &mut Option<usize>,
) -> DeleteWalk {
  use crate::functions::expr_form::{ExprForm, decompose_expr};

  if remaining.is_none_or(|n| n > 0)
    && position_level_match(level, position_depth(expr), min, max)
    && crate::evaluator::pattern_matching::match_pattern(expr, match_pat)
      .is_some()
  {
    if let Some(n) = remaining.as_mut() {
      *n -= 1;
    }
    return DeleteWalk::Deleted;
  }

  match expr {
    Expr::Association(pairs) => {
      let mut kept: Vec<(Expr, Expr)> = Vec::with_capacity(pairs.len());
      for (k, v) in pairs {
        if let DeleteWalk::Keep(nv) =
          delete_cases_walk(v, match_pat, level + 1, min, max, heads, remaining)
        {
          kept.push((k.clone(), nv));
        }
      }
      DeleteWalk::Keep(Expr::Association(kept))
    }
    Expr::CurriedCall { func, args } => {
      let new_head = if heads {
        delete_cases_walk(
          func,
          match_pat,
          level + 1,
          min,
          max,
          heads,
          remaining,
        )
      } else {
        DeleteWalk::Keep((**func).clone())
      };
      let kept = delete_cases_children(
        args, match_pat, level, min, max, heads, remaining,
      );
      match new_head {
        DeleteWalk::Keep(h) => DeleteWalk::Keep(Expr::CurriedCall {
          func: Box::new(h),
          args: kept,
        }),
        _ => DeleteWalk::Splice(kept),
      }
    }
    _ => match decompose_expr(expr) {
      ExprForm::Atom(_) => DeleteWalk::Keep(expr.clone()),
      ExprForm::Composite { head, children } => {
        let head_deleted = heads
          && remaining.is_none_or(|n| n > 0)
          && position_level_match(level + 1, 1, min, max)
          && crate::evaluator::pattern_matching::match_pattern(
            &Expr::Identifier(head.clone()),
            match_pat,
          )
          .is_some();
        if head_deleted && let Some(n) = remaining.as_mut() {
          *n -= 1;
        }
        let kept = delete_cases_children(
          &children, match_pat, level, min, max, heads, remaining,
        );
        if head_deleted {
          DeleteWalk::Splice(kept)
        } else {
          // Rebuilt in its own form: a rule stays a rule.
          DeleteWalk::Keep(crate::functions::expr_form::compose_expr(
            &head, &kept,
          ))
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
  let original = || unevaluated("DeleteCases", args);
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);

  // A trailing `Heads -> True | False` option.
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
    match (lhs, rhs) {
      (Expr::Identifier(h), Expr::Identifier(v)) if h == "Heads" => {
        match v.as_str() {
          "True" => Some(true),
          "False" => Some(false),
          _ => None,
        }
      }
      _ => None,
    }
  };
  let (args, heads) = match args.last().and_then(heads_setting) {
    Some(h) if args.len() >= 3 => (&args[..args.len() - 1], h),
    _ => (args, false),
  };

  if args.len() < 2 || args.len() > 4 {
    return Ok(original());
  }
  let subject = &args[0];
  // Unlike Cases, DeleteCases has no `pattern -> rhs` form: a rule is an
  // ordinary expression to match (e.g. deleting `1 -> 0` from a rule list).
  let match_pat = &args[1];

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
      if let Some(b) = parsed {
        b
      } else {
        crate::emit_message(&format!(
          "DeleteCases::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
          show(spec)
        ));
        return Ok(original());
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
    heads,
    &mut remaining,
  ) {
    DeleteWalk::Keep(result) => Ok(result),
    DeleteWalk::Deleted => Ok(call0("Sequence")),
    DeleteWalk::Splice(items) => Ok(call("Sequence", items)),
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
  let original = || unevaluated("Position", args);
  // A packed array object or a tree is an atom: nothing inside it has a
  // position, so only a match of the whole object could be reported — and
  // Position never reports the empty path for a pattern match at level 0
  // unless the level specification asks for it.
  if args.len() >= 2
    && crate::functions::predicate_ast::is_atomic_object(&args[0])
  {
    return Ok(Expr::List(Vec::new().into()));
  }
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
      if let Some(b) = parsed {
        b
      } else {
        crate::emit_message(&format!(
          "Position::level: Level specification {} is not of the form n, {{n}} or {{m, n}}.",
          show(spec)
        ));
        return Ok(original());
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
/// `FirstPosition[expr, patt]` / `[expr, patt, default]` /
/// `[expr, patt, default, levelspec]` — the first position `Position` reports,
/// or the default when there is none. Delegating keeps the two in step: a
/// head counts as position `{0}`, so `FirstPosition[{1, a, 2}, _Symbol]` is
/// `{0}` (the `List` head), not `{2}`.
pub fn first_position_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "FirstPosition expects at least 2 arguments".into(),
    ));
  }
  // Trailing `Heads -> …` options are Position's too.
  let n_opts = args[2..]
    .iter()
    .rev()
    .take_while(|a| is_heads_option(a))
    .count();
  let (positional, opts) = args.split_at(args.len() - n_opts);
  let default = if positional.len() >= 3 {
    positional[2].clone()
  } else {
    call1("Missing", str_expr("NotFound"))
  };
  let mut position_args = vec![args[0].clone(), args[1].clone()];
  if positional.len() >= 4 {
    position_args.push(positional[3].clone());
  }
  position_args.extend(positional.iter().skip(4).cloned());
  position_args.extend(opts.iter().cloned());
  match position_unified_ast(&position_args)? {
    Expr::List(ref found) => Ok(found.first().cloned().unwrap_or(default)),
    // Position left itself unevaluated (a malformed level spec, say).
    _ => Ok(unevaluated("FirstPosition", args)),
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

  // TakeWhile works on any expression with parts, preserving the head
  // (TakeWhile[f[2, 4, 1, 6], EvenQ] -> f[2, 4]).
  let (items, head_name): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
    _ => {
      let call_args = vec![list.clone(), pred.clone()];
      // An atomic first argument is invalid: emit ::normal, matching WL.
      if is_atomic_arg(list) {
        emit_nonatomic_normal_message("TakeWhile", &call_args);
      }
      return Ok(call("TakeWhile", call_args));
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

  // Preserve the original head, then evaluate the rebuilt expression so a
  // head with its own rules reduces; inert heads stay symbolic.
  match head_name {
    Some(name) => {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name,
        args: result.into(),
      })
    }
    None => Ok(Expr::List(result.into())),
  }
}

/// ContainsOnly[list, elems] - True if every element of list is in elems
pub fn contains_only_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "ContainsOnly expects 2 or 3 arguments".into(),
    ));
  }
  let Expr::List(list) = &args[0] else {
    return Ok(unevaluated("ContainsOnly", args));
  };
  let Expr::List(elems) = &args[1] else {
    return Ok(unevaluated("ContainsOnly", args));
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
          name: expr_to_string(&test),
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
        return Ok(bool_expr(false));
      }
    }
    return Ok(bool_expr(true));
  }

  use std::collections::HashSet;
  let allowed: HashSet<String> = elems.iter().map(expr_to_string).collect();

  for item in list {
    if !allowed.contains(&expr_to_string(item)) {
      return Ok(bool_expr(false));
    }
  }
  Ok(bool_expr(true))
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
    // Any other normal expression: count its leading arguments, whatever
    // the head (`LengthWhile[f[1, 2, a], IntegerQ]` is 2).
    Expr::FunctionCall { args: fargs, .. } => fargs.to_vec(),
    _ => {
      return Ok(unevaluated("LengthWhile", args));
    }
  };
  let mut count: i128 = 0;
  for item in &items {
    let test = crate::evaluator::evaluate_expr_to_expr(&call(
      "Apply",
      vec![crit.clone(), Expr::List(vec![item.clone()].into())],
    ))?;
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

/// The element slice of any non-atomic expression. The selector's own head is
/// irrelevant to Pick (only its structure/length matters); the result always
/// keeps the *first* argument's head, so we extract children uniformly from
/// Lists and general `head[...]` expressions.
fn pick_children(expr: &Expr) -> Option<&[Expr]> {
  match expr {
    Expr::List(items) => Some(items),
    Expr::FunctionCall { args, .. } => Some(args),
    _ => None,
  }
}

fn is_container(expr: &Expr) -> bool {
  matches!(expr, Expr::List(_) | Expr::FunctionCall { .. })
}

fn pick_recursive(
  list: &Expr,
  sel: &Expr,
  pattern: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  // Pick[assoc, sel] / Pick[assoc, sel, patt] — the selector is parallel to the
  // association's values; keep the matching key->value pairs as a
  // sub-association.
  if let Expr::Association(pairs) = list
    && let Some(sel_items) = pick_children(sel)
    && pairs.len() == sel_items.len()
  {
    let mut result: Vec<(Expr, Expr)> = Vec::new();
    for ((k, v), s) in pairs.iter().zip(sel_items.iter()) {
      if matches_selector(s, pattern) {
        result.push((k.clone(), v.clone()));
      }
    }
    return Ok(Expr::Association(result));
  }

  // General case: `list` is a List or any head[...] and `sel` has the same
  // length. Elements are matched by position; a nested selector element (itself
  // a container) recurses, otherwise the element is kept when the selector
  // matches. The result keeps `list`'s head.
  if let (Some(list_items), Some(sel_items)) =
    (pick_children(list), pick_children(sel))
    && list_items.len() == sel_items.len()
  {
    let mut result = Vec::new();
    for (item, s) in list_items.iter().zip(sel_items.iter()) {
      if is_container(item) && is_container(s) {
        result.push(pick_recursive(item, s, pattern)?);
      } else if matches_selector(s, pattern) {
        result.push(item.clone());
      }
    }
    return Ok(match list {
      Expr::FunctionCall { name, .. } => Expr::FunctionCall {
        name: name.clone(),
        args: result.into(),
      },
      _ => Expr::List(result.into()),
    });
  }

  Ok(call(
    "Pick",
    if let Some(p) = pattern {
      vec![list.clone(), sel.clone(), p.clone()]
    } else {
      vec![list.clone(), sel.clone()]
    },
  ))
}

fn matches_selector(sel: &Expr, pattern: Option<&Expr>) -> bool {
  match pattern {
    None => {
      matches!(sel, Expr::Identifier(s) if s == "True")
    }
    Some(pat) => {
      match crate::evaluator::evaluate_expr_to_expr(&call(
        "MatchQ",
        vec![sel.clone(), pat.clone()],
      )) {
        Ok(Expr::Identifier(ref s)) => s == "True",
        _ => false,
      }
    }
  }
}

// ─── PeakDetect ─────────────────────────────────────────────────────

/// PeakDetect[data, 0, sharpness, threshold, Padding -> p]: 1 on every
/// element of a peak run found at scale 0. `None` for forms outside that.
fn peak_detect_unsmoothed(args: &[Expr], positional: usize) -> Option<Expr> {
  use crate::functions::math_ast::try_eval_to_f64;
  if !(2..=4).contains(&positional) {
    return None;
  }
  let Ok(padding) = super::functional::peak_padding(&args[positional..]) else {
    return None;
  };
  let Expr::List(data) = &args[0] else {
    return None;
  };
  // A cut-off given bare or as `{value, 0}` (measured at scale 0).
  let unsmoothed = |e: &Expr| match e {
    Expr::List(p) if p.len() == 2 && try_eval_to_f64(&p[1]) == Some(0.0) => {
      try_eval_to_f64(&p[0])
    }
    Expr::List(_) => None,
    other => try_eval_to_f64(other),
  };
  if try_eval_to_f64(&args[1]) != Some(0.0) {
    return None;
  }
  let sharpness = match args.get(2).filter(|_| positional > 2) {
    None => 0.0,
    Some(e) => match unsmoothed(e) {
      Some(v) if v >= 0.0 => v,
      _ => return None,
    },
  };
  let threshold = match args.get(3).filter(|_| positional > 3) {
    None => f64::NEG_INFINITY,
    Some(e) => unsmoothed(e)?,
  };
  let vals = data
    .iter()
    .map(try_eval_to_f64)
    .collect::<Option<Vec<f64>>>()?;
  let mut marks = vec![0i128; vals.len()];
  for run in super::functional::peak_runs(&vals, sharpness, threshold, padding)
  {
    for m in &mut marks[run.start..=run.end] {
      *m = 1;
    }
  }
  Some(Expr::List(marks.into_iter().map(Expr::Integer).collect()))
}

/// PeakDetect[data] or PeakDetect[data, s]
/// Returns a list of 0s and 1s indicating peak positions.
/// Parameter s (default 0) controls sharpness filtering:
/// with s=0, finds all local peaks; higher s filters out less prominent peaks.
pub fn peak_detect_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // PeakDetect[data, 0, sharpness, threshold] (and Padding -> p) at the
  // unsmoothed scale marks the runs FindPeaks reports. Smoothing scales
  // beyond 0 are only supported by the two-argument heuristic below.
  let positional = args
    .iter()
    .take_while(|a| !matches!(a, Expr::Rule { .. }))
    .count();
  if positional < args.len() || positional > 2 {
    return Ok(
      peak_detect_unsmoothed(args, positional)
        .unwrap_or_else(|| unevaluated("PeakDetect", args)),
    );
  }
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PeakDetect expects 1 or 2 arguments".into(),
    ));
  }

  let Expr::List(data) = &args[0] else {
    return Ok(unevaluated("PeakDetect", args));
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
    .map(|&p| Expr::Integer(i128::from(p)))
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
  let Expr::List(items) = list else {
    return Ok(call("SubsetPosition", vec![list.clone(), pattern.clone()]));
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
  let Expr::List(items) = list else {
    let mut a = vec![list.clone(), pattern.clone()];
    if let Some(n) = max_count {
      a.push(Expr::Integer(n as i128));
    }
    return Ok(call("SubsetCases", a));
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
  let unevaluated = |args: &[Expr]| unevaluated("CommonestFilter", args);
  if args.len() != 2 {
    return Ok(unevaluated(args));
  }
  // An image is filtered channel by channel, as a matrix of its values.
  if let Expr::Image {
    width,
    height,
    channels,
    data,
    image_type,
    color_space,
    display_size,
  } = &args[0]
  {
    let (w, h, ch) = (*width as usize, *height as usize, *channels as usize);
    let mut out = vec![0.0; w * h * ch];
    for c in 0..ch {
      let matrix = Expr::List(
        (0..h)
          .map(|y| {
            Expr::List(
              (0..w)
                .map(|x| Expr::Real(data[(y * w + x) * ch + c]))
                .collect::<Vec<_>>()
                .into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      );
      let filtered = commonest_filter_ast(&[matrix, args[1].clone()])?;
      let Expr::List(rows) = &filtered else {
        return Ok(unevaluated(args));
      };
      for (y, row) in rows.iter().enumerate() {
        let Expr::List(cells) = row else {
          return Ok(unevaluated(args));
        };
        for (x, v) in cells.iter().enumerate() {
          let Expr::Real(v) = v else {
            return Ok(unevaluated(args));
          };
          out[(y * w + x) * ch + c] = *v;
        }
      }
    }
    return Ok(Expr::Image {
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(out),
      image_type: *image_type,
      color_space: *color_space,
      display_size: *display_size,
    });
  }
  let Expr::List(items) = &args[0] else {
    crate::emit_message(&format!(
      "CommonestFilter::arg1: The first argument {} should be a rectangular array, image or video.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated(args));
  };
  // The range rounds up and ignores its sign, as it does for MinFilter
  // and MaxFilter, so 1.5 is a radius of 2 and -1 a radius of 1.
  let Some(r) = crate::functions::image_ast::neighborhood_radius(
    "CommonestFilter",
    &args[1],
    crate::functions::image_ast::NeighborhoodRange::RoundedUp,
    true,
  ) else {
    return Ok(unevaluated(args));
  };
  if r == 0 {
    return Ok(args[0].clone());
  }

  // Commonest in `window` with the tie rules; elements compare by
  // their printed form
  let pick = |window: &[&Expr], center: &Expr| -> Expr {
    let keys: Vec<String> = window.iter().map(|e| expr_to_string(e)).collect();
    let count_of = |key: &str| -> usize {
      keys.iter().filter(|k| k.as_str() == key).count()
    };
    let max_count = keys.iter().map(|k| count_of(k)).max().unwrap_or(0);
    let center_key = expr_to_string(center);
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
        expr_to_string(&args[0])
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
