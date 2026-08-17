//! AST-native control flow functions.
//!
//! Switch, Piecewise, Quiet.

#[allow(unused_imports)]
use super::*;
use crate::evaluator::{apply_function_to_arg, evaluate_expr_to_expr};
use crate::functions::expr_form::{ExprForm, decompose_expr};

/// Switch[expr, pat1, val1, pat2, val2, ..., default?]
/// Evaluates expr, then finds first matching pattern and returns corresponding value.
/// Uses lazy evaluation — only matched branch is evaluated.
pub fn switch_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Switch[expr, p1, v1, p2, v2, ...] requires an odd number of arguments.
  // An even count is an error in wolframscript (there is no trailing default;
  // a catch-all uses the `_` pattern), reported as Switch::argct.
  if args.len().is_multiple_of(2) {
    crate::emit_message(&format!(
      "Switch::argct: Switch called with {} arguments. \
       Switch must be called with an odd number of arguments.",
      args.len()
    ));
    return Ok(unevaluated("Switch", args));
  }

  // Evaluate the test expression
  let test = evaluate_expr_to_expr(&args[0])?;
  let test_str = crate::syntax::expr_to_string(&test);

  // Iterate pattern-value pairs (the argument list is odd, so `rest` is even
  // and every pattern has a value — there is no leftover default argument).
  let rest = &args[1..];
  let mut i = 0;
  while i + 1 < rest.len() {
    // Every candidate is evaluated as it is tried — `Switch[2, 1 + 1, "two"]`
    // gives "two" — while the ones after the match are never touched. Patterns
    // proper (`_Integer`, `x_ /; x > 1`, …) evaluate to themselves.
    let pattern = evaluate_expr_to_expr(&rest[i])?;
    let value = &rest[i + 1];

    // Check if pattern matches
    if pattern_matches(&test, &pattern, &test_str) {
      return evaluate_expr_to_expr(value);
    }
    i += 2;
  }

  // No match — return unevaluated
  Ok(unevaluated("Switch", args))
}

/// Check if `test` matches `pattern`.
fn pattern_matches(test: &Expr, pattern: &Expr, _test_str: &str) -> bool {
  crate::evaluator::match_pattern(test, pattern).is_some()
}

/// Piecewise[{{val1, cond1}, {val2, cond2}, ...}] or
/// Piecewise[{{val1, cond1}, {val2, cond2}, ...}, default]
pub fn piecewise_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Piecewise expects 1 or 2 arguments".into(),
    ));
  }

  // `Piecewise` holds its arguments: the value of a piece whose condition is
  // False must never be evaluated, so that a guard like
  // `Piecewise[{{f[x], 0 <= x <= xmax}}, 0]` protects `f` from being called
  // out of range at all (the loop below evaluates each value only once its
  // condition has selected the piece). A first argument that is *already* a
  // list of pairs is therefore walked as written; only an indirect one (a
  // symbol holding the list, `Piecewise[Table[…]]`, …) is evaluated first to
  // get at the pairs.
  let evaluated_first = match &args[0] {
    Expr::List(_) => args[0].clone(),
    other => evaluate_expr_to_expr(other)?,
  };
  let pairs = match &evaluated_first {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "First argument of Piecewise must be a list of {value, condition} pairs".into(),
      ));
    }
  };
  let pairs = &pairs;

  let default = if args.len() == 2 {
    Some(&args[1])
  } else {
    None
  };

  // Wolfram semantics: conditions evaluate in order; False pieces are
  // removed; the values of surviving pieces (and the default) evaluate;
  // a True condition after symbolic ones turns its value into the new
  // default and drops everything behind it.
  let mut kept: Vec<Expr> = Vec::new(); // evaluated {value, cond} pairs
  let mut true_default: Option<Expr> = None;

  for pair in pairs {
    // A piece given indirectly (a symbol holding `{value, condition}`)
    // resolves here; a literal pair keeps its value held until its condition
    // selects it.
    let resolved;
    let pair = match pair {
      Expr::List(items) if items.len() == 2 => pair,
      other => {
        resolved = evaluate_expr_to_expr(other)?;
        &resolved
      }
    };
    match pair {
      Expr::List(items) if items.len() == 2 => {
        let cond = evaluate_expr_to_expr(&items[1])?;
        match &cond {
          Expr::Identifier(s) if s == "True" => {
            let val = evaluate_expr_to_expr(&items[0])?;
            if kept.is_empty() {
              // First reachable piece — Piecewise collapses to it
              return Ok(val);
            }
            true_default = Some(val);
            break;
          }
          Expr::Identifier(s) if s == "False" => {}
          _ => {
            let val = evaluate_expr_to_expr(&items[0])?;
            kept.push(Expr::List(vec![val, cond].into()));
          }
        }
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Each element of Piecewise list must be a {value, condition} pair"
            .into(),
        ));
      }
    }
  }

  let default_val = match true_default {
    Some(v) => v,
    None => match default {
      Some(d) => evaluate_expr_to_expr(d)?,
      None => Expr::Integer(0),
    },
  };

  if kept.is_empty() {
    // No condition can still become True — collapse to the default
    return Ok(default_val);
  }

  // wolframscript merges *consecutive* clauses whose values are structurally
  // identical, OR-ing their distinct conditions:
  //   Piecewise[{{a, c1}, {a, c2}}] -> Piecewise[{{a, c1 || c2}}].
  // Identical clauses collapse to one (c1 || c1 -> c1).
  let mut merged: Vec<(Expr, Vec<Expr>)> = Vec::new();
  for pair in &kept {
    if let Expr::List(items) = pair
      && items.len() == 2
    {
      let val = items[0].clone();
      let cond = items[1].clone();
      if let Some((prev_val, conds)) = merged.last_mut()
        && expr_to_string(prev_val) == expr_to_string(&val)
      {
        let cond_str = expr_to_string(&cond);
        if !conds.iter().any(|c| expr_to_string(c) == cond_str) {
          conds.push(cond);
        }
        continue;
      }
      merged.push((val, vec![cond]));
    }
  }

  // Rebuild the {value, condition} pairs, OR-combining multi-condition groups.
  let mut clauses: Vec<Expr> = Vec::with_capacity(merged.len());
  for (val, conds) in merged {
    let cond = if conds.len() == 1 {
      conds.into_iter().next().unwrap()
    } else {
      evaluate_expr_to_expr(&call("Or", conds))?
    };
    clauses.push(Expr::List(vec![val, cond].into()));
  }

  // wolframscript drops trailing clauses whose value equals the default
  // (a clause that only restates the fallback is redundant).
  let default_str = expr_to_string(&default_val);
  while let Some(Expr::List(items)) = clauses.last() {
    if items.len() == 2 && expr_to_string(&items[0]) == default_str {
      clauses.pop();
    } else {
      break;
    }
  }

  if clauses.is_empty() {
    return Ok(default_val);
  }

  Ok(call(
    "Piecewise",
    vec![Expr::List(clauses.into()), default_val],
  ))
}

/// Parse the message-off specification for Quiet.
/// Returns None for All (suppress everything), Some(vec) for specific messages.
/// Each message is a (symbol, tag) pair like ("Power", "infy").
fn parse_quiet_spec(spec: &Expr) -> Option<Vec<(String, String)>> {
  match spec {
    Expr::Identifier(s) if s == "All" => None,
    Expr::Identifier(s) if s == "None" => Some(vec![]),
    Expr::List(items) => {
      let mut msgs = Vec::new();
      for item in items {
        if let Some((sym, tag)) = parse_message_name(item) {
          msgs.push((sym, tag));
        }
      }
      Some(msgs)
    }
    // Single message name
    _ => {
      if let Some((sym, tag)) = parse_message_name(spec) {
        Some(vec![(sym, tag)])
      } else {
        // Unrecognized — treat as All
        None
      }
    }
  }
}

/// Extract (symbol, tag) from a MessageName expression like Power::infy.
/// MessageName[Power, "infy"] → ("Power", "infy")
fn parse_message_name(expr: &Expr) -> Option<(String, String)> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "MessageName" && args.len() == 2 =>
    {
      let sym = match &args[0] {
        Expr::Identifier(s) => s.clone(),
        _ => expr_to_string(&args[0]),
      };
      let tag = match &args[1] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => expr_to_string(&args[1]),
      };
      Some((sym, tag))
    }
    _ => None,
  }
}

/// Check if a warning message string matches a (symbol, tag) spec.
/// Warning format: "Symbol::tag: ..."
fn message_matches(warning: &str, specs: &[(String, String)]) -> bool {
  for (sym, tag) in specs {
    let prefix = format!("{sym}::{tag}: ");
    if warning.starts_with(&prefix) {
      return true;
    }
    // Also match without trailing space (in case message is just "Symbol::tag:")
    let prefix2 = format!("{sym}::{tag}:");
    if warning.starts_with(&prefix2) {
      return true;
    }
  }
  false
}

/// Quiet[expr] — evaluate expr, suppress all messages
/// Quiet[expr, {msg1, msg2, ...}] — suppress only specific messages
/// Quiet[expr, All] — suppress all messages
/// Quiet[expr, None] — suppress nothing
/// Quiet[expr, moff, mon] — suppress moff, enable mon
pub fn quiet_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Parse the message specification
  let suppress_spec = if args.len() == 1 {
    // Quiet[expr] — suppress all
    None // None means "all"
  } else {
    // Quiet[expr, spec] or Quiet[expr, moff, mon]
    parse_quiet_spec(&args[1])
  };

  // For 3-arg form Quiet[expr, moff, mon], mon re-enables messages within moff
  // For simplicity: if moff is None (All), mon still re-enables specific messages
  // But the common case is just Quiet[expr] or Quiet[expr, {msgs}]
  let enable_spec = if args.len() == 3 {
    parse_quiet_spec(&args[2])
  } else {
    Some(vec![]) // Nothing explicitly re-enabled
  };

  let suppress_all = suppress_spec.is_none();
  let suppress_msgs = suppress_spec.unwrap_or_default();

  // Determine effective suppression:
  // - If suppress_all and enable is None (All): suppress nothing (all re-enabled)
  // - If suppress_all and enable is Some(specific): suppress all except those in enable
  // - If suppress specific: suppress those, minus any in enable

  let effective_suppress_all = if suppress_all {
    match &enable_spec {
      None => false, // Quiet[expr, All, All] = everything re-enabled
      Some(enables) => enables.is_empty(), // No re-enables: suppress all
    }
  } else {
    false
  };

  // If None suppression (suppress nothing), just evaluate normally
  if suppress_msgs.is_empty() && !suppress_all {
    return evaluate_expr_to_expr(&args[0]);
  }

  // Save warning state
  let snapshot = crate::snapshot_warnings();

  // Suppress message printing
  if effective_suppress_all {
    crate::push_quiet();
  }

  // Evaluate the expression
  let result = evaluate_expr_to_expr(&args[0]);

  // Restore quiet level
  if effective_suppress_all {
    crate::pop_quiet();
  }

  // Handle warning cleanup based on suppression spec
  if effective_suppress_all {
    // Suppress all: restore all buffers to snapshot (discard everything added during eval)
    crate::restore_warnings(snapshot);
  } else if !suppress_msgs.is_empty() {
    // Suppress specific messages: remove only matching entries from the
    // messages buffer and from `$MessageList` (entries already there when the
    // block started are kept regardless of the filter).
    let snap_msgs = snapshot.messages().len();
    let snap_list = snapshot.message_list().len();
    let new_msgs: Vec<String> = crate::get_captured_messages_raw()
      .into_iter()
      .enumerate()
      .filter(|(i, w)| *i < snap_msgs || !message_matches(w, &suppress_msgs))
      .map(|(_, w)| w)
      .collect();
    // `$MessageList` holds bare `Head::tag` names, so the filter matches
    // against the name with a colon appended.
    let new_list: Vec<String> = crate::message_list_names()
      .into_iter()
      .enumerate()
      .filter(|(i, n)| {
        *i < snap_list || !message_matches(&format!("{n}:"), &suppress_msgs)
      })
      .map(|(_, n)| n)
      .collect();
    crate::restore_warnings_filtered(
      snapshot.unimplemented().to_vec(),
      new_msgs,
      new_list,
    );
  }

  result
}

/// Wrap an expression in HoldForm.
fn hold_form(expr: &Expr) -> Expr {
  call1("HoldForm", expr.clone())
}

/// What to do with each sub-expression a trace visits.
enum TraceSink<'a> {
  /// TraceScan[f, …] — apply `f` to the HoldForm-wrapped sub-expression.
  Apply(&'a Expr),
  /// TracePrint[…] — print the sub-expression wrapped in HoldCompleteForm,
  /// indented by its depth in the evaluation (the nesting `Trace` produces).
  Print,
}

/// Deliver an expression to the trace sink, unconditionally.
/// `depth` is the expression's nesting level in the evaluation (1 for the
/// traced expression itself) and is only used for TracePrint's indentation.
fn do_trace(
  expr: &Expr,
  sink: &TraceSink,
  depth: usize,
) -> Result<(), InterpreterError> {
  match sink {
    TraceSink::Apply(f) => {
      let wrapped = hold_form(expr);
      apply_function_to_arg(f, &wrapped)?;
    }
    TraceSink::Print => {
      // Each step is printed wrapped in HoldCompleteForm, the same way
      // TraceScan's steps reach the scanning function wrapped in HoldForm.
      let wrapped = call1("HoldCompleteForm", expr.clone());
      let line = format!(
        "{}{}",
        " ".repeat(depth),
        crate::syntax::expr_to_output(&wrapped)
      );
      if !crate::is_quiet_print() {
        println!("{line}");
      }
      crate::capture_stdout(&line);
    }
  }
  Ok(())
}

/// Deliver an expression to the trace sink if it matches the form filter.
/// Returns true if the expression was traced.
fn maybe_trace(
  expr: &Expr,
  sink: &TraceSink,
  form: Option<&Expr>,
  depth: usize,
) -> Result<bool, InterpreterError> {
  if let Some(form_pat) = form
    && crate::evaluator::match_pattern(expr, form_pat).is_none()
  {
    return Ok(false);
  }
  do_trace(expr, sink, depth)?;
  Ok(true)
}

/// Rebuild an expression from a head name and its evaluated children.
/// Uses the canonical `Expr` variant for the head so the rebuilt step prints
/// like the original (`{2, 4}`, not `List[2, 4]`).
fn rebuild_from_head(head: &str, children: &[Expr]) -> Expr {
  crate::functions::expr_form::compose_expr(head, children)
}

/// Recursively trace-evaluate an expression, handing every sub-expression to
/// `sink`. `depth` is the nesting level of `expr` in the evaluation; the head
/// and the arguments sit one level deeper, while the rebuilt expression and
/// the result stay at the same level (mirroring the list nesting of `Trace`).
fn trace_eval(
  expr: &Expr,
  sink: &TraceSink,
  form: Option<&Expr>,
  depth: usize,
) -> Result<Expr, InterpreterError> {
  // Trace the input expression
  maybe_trace(expr, sink, form, depth)?;

  match decompose_expr(expr) {
    ExprForm::Atom(_) => {
      // Atoms evaluate to themselves (or to their value)
      let result = evaluate_expr_to_expr(expr)?;
      if expr_to_string(&result) != expr_to_string(expr) {
        maybe_trace(&result, sink, form, depth)?;
      }
      Ok(result)
    }
    ExprForm::Composite { head, children } => {
      // Trace the head; remember if it matched the form
      let head_expr = Expr::Identifier(head.clone());
      let head_matched = maybe_trace(&head_expr, sink, form, depth + 1)?;

      // Recursively trace-evaluate each child. Arguments the head holds are
      // reported but left untouched — evaluating them here would defeat the
      // Hold attribute (`TracePrint[x = x + 1]` must not turn the assignment
      // target into `0 = 1`).
      let mut evaluated_children = Vec::new();
      let mut children_changed = false;
      for (index, child) in children.iter().enumerate() {
        let eval_child = if crate::evaluator::holds_argument_at(&head, index) {
          maybe_trace(child, sink, form, depth + 1)?;
          child.clone()
        } else {
          trace_eval(child, sink, form, depth + 1)?
        };
        if expr_to_string(&eval_child) != expr_to_string(child) {
          children_changed = true;
        }
        evaluated_children.push(eval_child);
      }

      // Rebuild with evaluated children
      let rebuilt = rebuild_from_head(&head, &evaluated_children);
      if children_changed {
        // When head matched, always trace rebuilt; otherwise check form
        if head_matched {
          do_trace(&rebuilt, sink, depth)?;
        } else {
          maybe_trace(&rebuilt, sink, form, depth)?;
        }
      }

      // Final evaluation (apply the head function to evaluated args)
      let result = evaluate_expr_to_expr(&rebuilt)?;
      let rebuilt_str = expr_to_string(&rebuilt);
      let result_str = expr_to_string(&result);
      if result_str != rebuilt_str {
        // When head matched, always trace result; otherwise check form
        if head_matched {
          do_trace(&result, sink, depth)?;
        } else {
          maybe_trace(&result, sink, form, depth)?;
        }
      }

      Ok(result)
    }
  }
}

/// TraceScan[f, expr] — apply f to each sub-expression during evaluation.
/// TraceScan[f, expr, form] — apply f only to sub-expressions matching form.
/// Returns the evaluated result of expr.
pub fn trace_scan_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Evaluate the function argument (first arg)
  let f = evaluate_expr_to_expr(&args[0])?;
  let expr = &args[1];
  let form = if args.len() == 3 {
    Some(&args[2])
  } else {
    None
  };

  trace_eval(expr, &TraceSink::Apply(&f), form, 1)
}

/// TracePrint[expr] — print every sub-expression used while evaluating expr,
/// indented by one space per level of the evaluation.
/// TracePrint[expr, form] — print only sub-expressions matching form.
/// Returns the evaluated result of expr.
pub fn trace_print_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let expr = &args[0];
  let form = args.get(1);

  trace_eval(expr, &TraceSink::Print, form, 1)
}
