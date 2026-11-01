use super::utilities::expr_to_f64;
#[allow(unused_imports)]
use super::utilities::*;
use super::*;
use crate::functions::math_ast::rat_reduce;

/// Longest list `Range` (and the geometric `PowerRange`) will build.  Wolfram
/// caps these only by available memory — `Range[10^7]` is routine in
/// Project-Euler-style code — so the limit is a runaway guard, not a
/// language-level restriction.
const MAX_RANGE_LENGTH: usize = 100_000_000;

/// AST-based Table: generate a table of values.
/// Table[expr, {i, min, max}] -> {expr with i=min, ..., expr with i=max}
/// Table[expr, {i, max}] -> {expr with i=1, ..., expr with i=max}
/// Table[expr, {i, {list}}] -> {expr with i=elem1, expr with i=elem2, ...}
/// Table[expr, n] -> {expr, expr, ..., expr} (n times)
/// Multi-dimensional Table: Table[expr, iter1, iter2, ...]
/// Recursively nests Table from last iterator to first.
pub fn table_multi_ast(
  body: &Expr,
  iters: &[Expr],
) -> Result<Expr, InterpreterError> {
  if iters.len() == 1 {
    return table_ast(body, &iters[0]);
  }
  // Nest: the body for the outer iterator is Table[body, inner_iters...]
  // We build from the innermost outward
  let inner_body = call(
    "Table",
    std::iter::once(body.clone())
      .chain(iters[1..].iter().cloned())
      .collect(),
  );
  table_ast(&inner_body, &iters[0])
}

/// A count/bound value: an exact integer, or a real/rational truncated toward
/// zero (wolframscript's `Table[q, {5.5}]` yields 5 copies). Returns None for a
/// non-numeric (symbolic) value.
fn bound_to_count(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(_) | Expr::BigInteger(_) => expr_to_i128(e),
    _ => crate::functions::math_ast::try_eval_to_f64(e).map(|f| f as i128),
  }
}

/// The unevaluated single-iterator `Table[body, iter_spec]`.
fn table_unevaluated(body: &Expr, iter_spec: &Expr) -> Expr {
  call("Table", vec![body.clone(), iter_spec.clone()])
}

/// If any iterator in a `Table`/`Do`-style call has bounds that don't resolve
/// to real numbers, return the message to emit before leaving the whole call
/// unevaluated (matching wolframscript's `::iterb` / `::nliter`). `name` is the
/// enclosing head (e.g. "Table", "Do"). `iters` are the iterator specs (the
/// arguments after the body).
/// The iterator specs with every bound evaluated once — except bounds that
/// reference an earlier iterator's variable, which only mean something
/// during iteration. The validity check and the iteration both read these,
/// so an expensive bound (`{x, someLongComputation}`) is not computed twice,
/// and a bound with side effects runs once, as in Wolfram.
pub fn table_preevaluate_iterators(iters: &[Expr]) -> Vec<Expr> {
  table_counted_iterators(table_evaluate_iterators(iters))
}

/// `{list}` runs once per element of the list, like an iterator `{i, list}`
/// whose variable the body never names: as a count.
pub fn table_counted_iterators(iters: Vec<Expr>) -> Vec<Expr> {
  iters
    .into_iter()
    .map(|iter| match &iter {
      Expr::List(items) if items.len() == 1 => match &items[0] {
        Expr::List(values) => {
          Expr::List(vec![Expr::Integer(values.len() as i128)].into())
        }
        _ => iter,
      },
      _ => iter,
    })
    .collect()
}

/// The iterator specs with their bounds evaluated (see
/// [`table_preevaluate_iterators`]), a `{list}` still holding its list.
pub fn table_evaluate_iterators(iters: &[Expr]) -> Vec<Expr> {
  let eval = |e: &Expr| {
    crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone())
  };
  let mut bound_vars: Vec<String> = Vec::new();
  iters
    .iter()
    .map(|iter| match iter {
      Expr::List(items) if items.len() >= 2 => {
        let earlier = bound_vars.clone();
        let mut out = Vec::with_capacity(items.len());
        out.push(items[0].clone());
        for b in &items[1..] {
          out.push(if expr_references_any(b, &earlier) {
            b.clone()
          } else {
            eval(b)
          });
        }
        if let Expr::Identifier(v) = &items[0] {
          bound_vars.push(v.clone());
        }
        Expr::List(out.into())
      }
      Expr::List(items) if items.len() == 1 => {
        if expr_references_any(&items[0], &bound_vars) {
          iter.clone()
        } else {
          Expr::List(vec![eval(&items[0])].into())
        }
      }
      Expr::List(_) => iter.clone(),
      other if expr_references_any(other, &bound_vars) => other.clone(),
      other => eval(other),
    })
    .collect()
}

pub fn table_iterators_invalid(name: &str, iters: &[Expr]) -> Option<String> {
  table_iterator_failure(name, iters).map(|(msg, _)| msg)
}

/// Like [`table_iterators_invalid`], with the position of the offending
/// iterator in `iters`.
pub fn table_iterator_failure(
  name: &str,
  iters: &[Expr],
) -> Option<(String, usize)> {
  let mut failing = 0usize;
  let failure = table_iterator_failure_message(name, iters, &mut failing)?;
  Some((failure, failing))
}

/// Whether an iterator bound is a number that cannot count: complex, or an
/// infinity. `Table` reports such an iterator twice, as wolframscript does
/// (`Table[x, {I}]`); a symbolic bound once.
pub fn iterator_has_nonreal_numeric_bound(iter: &Expr) -> bool {
  let bounds: &[Expr] = match iter {
    Expr::List(items) if items.len() == 1 => items,
    Expr::List(items) if items.len() > 1 => &items[1..],
    _ => return false,
  };
  bounds.iter().any(|b| {
    let infinite = matches!(
      crate::functions::math_ast::try_eval_to_f64_with_infinity(b),
      Some(f) if f.is_infinite()
    );
    let complex_infinity = matches!(b, Expr::Identifier(s) if s == "ComplexInfinity")
      || matches!(b, Expr::FunctionCall { name, args } if name == "DirectedInfinity" && args.is_empty());
    infinite
      || complex_infinity
      || (crate::functions::predicate_ast::is_numeric_q(b)
        && crate::functions::math_ast::try_eval_to_f64(b).is_none())
  })
}

fn table_iterator_failure_message(
  name: &str,
  iters: &[Expr],
  failing: &mut usize,
) -> Option<String> {
  let iterb = |iter: &Expr| {
    Some(format!(
      "{}::iterb: Iterator {} does not have appropriate bounds.",
      name,
      crate::syntax::format_expr(iter, crate::syntax::ExprForm::Output)
    ))
  };
  // `Do` runs an infinite iterator until a `Break[]`; `Table` cannot.
  let allow_infinite = !matches!(name, "Table" | "ParallelTable");
  // Variables introduced by earlier iterators: a later bound that references
  // one of them (e.g. the inner `{j, i}` of `Table[j, {i, 1, 3}, {j, i}]`)
  // becomes numeric during iteration, so it must not be flagged.
  let mut bound_vars: Vec<String> = Vec::new();
  // A bound "resolves" if it is already numeric or references an earlier
  // iterator variable.
  let resolves = |e: &Expr, vars: &[String]| -> bool {
    if expr_references_any(e, vars) {
      return true;
    }
    let ev =
      crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
    match crate::functions::math_ast::try_eval_to_f64_with_infinity(&ev) {
      Some(f) => allow_infinite || f.is_finite(),
      None => false,
    }
  };
  for (idx, iter) in iters.iter().enumerate() {
    *failing = idx;
    let position = idx + 2; // the body is argument 1
    match iter {
      // A bare count `Table[expr, n]`: valid when it resolves.
      _ if !matches!(iter, Expr::List(_)) && resolves(iter, &bound_vars) => {}
      Expr::List(items) => {
        if items.is_empty() {
          continue;
        }
        if items.len() == 1 {
          if !resolves(&items[0], &bound_vars) {
            return iterb(iter);
          }
          continue;
        }
        // `{i, ...}`: items[0] is the iterator variable (introduced for later
        // iterators). The remaining entries must be a single list, or all
        // resolve to numbers / earlier iterator variables.
        // Bounds that reference an earlier iterator variable are only
        // meaningful once that variable is bound, so they are not evaluated
        // here (doing so would e.g. emit `Part::pkspec1` for `l[[i]]`).
        let earlier_vars = bound_vars.clone();
        let eval_bound = |e: &Expr| {
          if expr_references_any(e, &earlier_vars) {
            e.clone()
          } else {
            crate::evaluator::evaluate_expr_to_expr(e)
              .unwrap_or_else(|_| e.clone())
          }
        };
        if let Expr::Identifier(v) = &items[0] {
          bound_vars.push(v.clone());
        }
        // A zero step `{i, min, max, 0}` never terminates: Table/Do reject it
        // with iterb rather than looping forever (matching wolframscript).
        if items.len() >= 4 {
          let step = eval_bound(&items[3]);
          if crate::functions::math_ast::try_eval_to_f64(&step) == Some(0.0) {
            return iterb(iter);
          }
        }
        let second = eval_bound(&items[1]);
        if matches!(second, Expr::List(_)) {
          continue;
        }
        // The variable of this iterator is in scope only for later ones, so
        // check the bounds against the earlier variables (all but the last).
        let earlier = &bound_vars[..bound_vars.len().saturating_sub(1)];
        if items[1..].iter().all(|b| resolves(b, earlier)) {
          continue;
        }
        // A `{i, min, max, step}` range with symbolic bounds is still valid
        // when its count (max - min)/step is a definite non-negative integer
        // (e.g. `{x, a, a + 5 n, n}` -> 6 values).
        if items.len() >= 3
          && symbolic_count_definite(&items[1], &items[2], items.get(3))
        {
          continue;
        }
        return iterb(iter);
      }
      // A non-list, non-numeric iterator such as the `i` in `Table[i, i]`.
      _ => {
        return Some(format!(
          "{}::nliter: Nonlist iterator {} at position {} does not evaluate to a real numeric value.",
          name,
          expr_to_string(iter),
          position
        ));
      }
    }
  }
  None
}

/// Whether the range `{min, max, step}` has a definite real count
/// `(max - min)/step`, so `Table` can iterate it symbolically even when the
/// bounds themselves are symbolic (e.g. `{a, a + 5 n, n}` -> count 5).
fn symbolic_count_definite(
  min: &Expr,
  max: &Expr,
  step: Option<&Expr>,
) -> bool {
  let eval = |e: &Expr| {
    crate::evaluator::evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone())
  };
  let (min_e, max_e) = (eval(min), eval(max));
  let step_e = step.map_or(Expr::Integer(1), eval);
  let Ok(diff) =
    crate::evaluator::evaluate_function_call_ast("Subtract", &[max_e, min_e])
  else {
    return false;
  };
  let Ok(ratio) =
    crate::evaluator::evaluate_function_call_ast("Divide", &[diff, step_e])
  else {
    return false;
  };
  // Any real count runs Floor[count] + 1 times (none when negative), as for
  // numeric bounds: `{θ, Pi + s, 3.4557 + s, Pi/50}` has the count 5.
  crate::functions::math_ast::try_eval_to_f64(&ratio).is_some()
}

/// Whether `expr` references any identifier named in `vars`, looking through
/// every syntactic wrapper (`a[[i]]`, `f@x`, `f/@list`, `f@@list`, …) so a
/// later iterator's bound is recognised as depending on an earlier iterator
/// variable no matter how the reference is written — e.g. the inner
/// `{x, 0, nl[[dn1]] - 1}` of `Table[…, {dn1, ...}, {x, 0, nl[[dn1]] - 1}]`
/// depends on `dn1` through a `Part` node, not a bare identifier. Exhaustive
/// (no wildcard arm) so a new `Expr` variant forces this to be updated too.
fn expr_references_any(expr: &Expr, vars: &[String]) -> bool {
  if vars.is_empty() {
    return false;
  }
  match expr {
    Expr::Identifier(n) => vars.iter().any(|v| v == n),
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _)
    | Expr::String(_)
    | Expr::Slot(_)
    | Expr::SlotSequence(_)
    | Expr::Pattern { .. }
    | Expr::Constant(_)
    | Expr::Raw(_)
    | Expr::Image { .. }
    | Expr::Graphics { .. } => false,
    Expr::List(items) => items.iter().any(|a| expr_references_any(a, vars)),
    Expr::CompoundExpr(items) => {
      items.iter().any(|a| expr_references_any(a, vars))
    }
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| expr_references_any(a, vars))
    }
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|a| expr_references_any(a, vars))
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_references_any(left, vars) || expr_references_any(right, vars)
    }
    Expr::Rule {
      pattern: a,
      replacement: b,
    }
    | Expr::RuleDelayed {
      pattern: a,
      replacement: b,
    }
    | Expr::ReplaceAll { expr: a, rules: b }
    | Expr::ReplaceRepeated { expr: a, rules: b }
    | Expr::Map { func: a, list: b }
    | Expr::Apply { func: a, list: b }
    | Expr::MapApply { func: a, list: b }
    | Expr::PrefixApply { func: a, arg: b }
    | Expr::Postfix { expr: a, func: b }
    | Expr::Part { expr: a, index: b } => {
      expr_references_any(a, vars) || expr_references_any(b, vars)
    }
    Expr::UnaryOp { operand, .. } => expr_references_any(operand, vars),
    Expr::Function { body } => expr_references_any(body, vars),
    Expr::NamedFunction { body, .. } => expr_references_any(body, vars),
    Expr::PatternOptional { default, .. } => default
      .as_deref()
      .is_some_and(|d| expr_references_any(d, vars)),
    Expr::PatternTest { test, .. } => expr_references_any(test, vars),
    Expr::CurriedCall { func, args } => {
      expr_references_any(func, vars)
        || args.iter().any(|a| expr_references_any(a, vars))
    }
    Expr::Association(pairs) => pairs.iter().any(|(k, v)| {
      expr_references_any(k, vars) || expr_references_any(v, vars)
    }),
  }
}

pub fn table_ast(
  body: &Expr,
  iter_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // The body is substituted once per iteration: its large arrays are
  // shared first, so those substitutions copy pointers, not arrays.
  let mut shared_body = body.clone();
  crate::evaluator::share_large_lists(&mut shared_body, 0);
  let body = &shared_body;
  // A bare (non-list) iterator spec may be a symbol or expression that
  // evaluates to a repeat count (`Table[x, n]`, `Table[x, 2 + 3]`), not just
  // a literal integer. Resolve it before dispatching so the arms below only
  // have to deal with already-evaluated forms; the caller (table_iterators_
  // invalid) has already confirmed a bare spec resolves to a number.
  let resolved_spec;
  let iter_spec: &Expr = if matches!(
    iter_spec,
    Expr::List(_) | Expr::Integer(_) | Expr::BigInteger(_)
  ) {
    iter_spec
  } else {
    resolved_spec = crate::evaluator::evaluate_expr_to_expr(iter_spec)?;
    &resolved_spec
  };
  match iter_spec {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      // Simple form: Table[expr, n]
      let n = expr_to_i128(iter_spec).ok_or_else(|| {
        InterpreterError::EvaluationError("Table: count too large".into())
      })?;
      if n < 0 {
        return Err(InterpreterError::EvaluationError(
          "Table: count must be non-negative".into(),
        ));
      }
      let mut results = Vec::new();
      for _ in 0..n {
        let val = crate::evaluator::evaluate_value(body)?;
        crate::evaluator::listable::push_list_element(&mut results, val);
      }
      Ok(Expr::List(results.into()))
    }
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::List(vec![].into()));
      }

      // Handle {n} form (single element = just repeat count, no variable)
      if items.len() == 1 {
        let evaluated = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
        // A non-numeric count leaves the call unevaluated (the dispatch-level
        // check normally intercepts this; kept here as a safety net). A real
        // count is truncated toward zero, matching wolframscript's
        // Table[q, {5.5}] -> 5 copies.
        let Some(n) = bound_to_count(&evaluated) else {
          return Ok(table_unevaluated(body, iter_spec));
        };
        let mut results = Vec::new();
        for _ in 0..n {
          let val = crate::evaluator::evaluate_value(body)?;
          crate::evaluator::listable::push_list_element(&mut results, val);
        }
        return Ok(Expr::List(results.into()));
      }

      // Extract iterator variable
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        other => {
          // A non-symbol iterator (Table[i, {3, 1, 5}]) is a raw object that
          // cannot be used as an iterator: wolframscript emits Table::itraw
          // and returns the call unevaluated rather than raising an error.
          crate::emit_message(&format!(
            "Table::itraw: Raw object {} cannot be used as an iterator.",
            expr_to_string(other)
          ));
          return Ok(call("Table", vec![body.clone(), iter_spec.clone()]));
        }
      };

      // Wolfram's Table localizes the iterator the way Block does: the symbol
      // is given a value for the duration of one iteration instead of being
      // textually replaced throughout the body. The two only agree where the
      // body is fully evaluated — every held position tells them apart.
      // `Table[tile[i, 0] :> i^2, {i, 3}]` keeps `i^2` on the right of each
      // rule (RuleDelayed holds it) and `Table[Hold[i], {i, 2}]` is
      // `{Hold[i], Hold[i]}`; substitution would burn the loop counter in.
      // The substitute path stays for a body that uses the iterator as a
      // function head, which an ENV binding cannot express.
      let substitute = body_uses_var_as_function_head(body, &var_name);
      let prev = if substitute {
        None
      } else {
        crate::ENV.with(|e| e.borrow_mut_own_value(&var_name).remove(&var_name))
      };
      let result =
        table_over_iterator(body, &var_name, items, iter_spec, substitute);
      if !substitute {
        restore_loop_var(&var_name, prev);
      }
      result
    }
    _ => Err(InterpreterError::EvaluationError(
      "Table: invalid iterator specification".into(),
    )),
  }
}

/// One `Table` iterator `{i, …}`: run `var_name` over the spec in `items` and
/// collect the body's value at each step.
///
/// `substitute` picks the textual-substitution path instead of the dynamic
/// binding the caller has already installed — see the note there for when
/// that is still needed.
fn table_over_iterator(
  body: &Expr,
  var_name: &str,
  items: &[Expr],
  iter_spec: &Expr,
  substitute: bool,
) -> Result<Expr, InterpreterError> {
  // The body's value with the iterator standing at `current`.
  let eval_at = |current: &Expr| -> Result<Expr, InterpreterError> {
    if substitute {
      let substituted =
        crate::syntax::substitute_variable(body, var_name, current);
      crate::evaluator::evaluate_value(&substituted)
    } else {
      bind_loop_var(var_name, current.clone());
      crate::evaluator::evaluate_value(body)
    }
  };

  if items.len() == 2 {
    // Check if second element is a list (iterate over list)
    let mut second = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
    if let Expr::List(list_items) = &mut second {
      // {i, {a, b, c}} form - iterate over list elements
      let mut results = Vec::new();
      for item in list_items.iter() {
        let val = eval_at(item)?;
        crate::evaluator::listable::push_list_element(&mut results, val);
      }
      return Ok(Expr::List(results.into()));
    }
    // {i, max} form - iterate from 1 to max (real max truncated).
    let Some(max_val) = bound_to_count(&second) else {
      return Ok(table_unevaluated(body, iter_spec));
    };
    let mut results = Vec::new();
    for i in 1..=max_val {
      let val = eval_at(&Expr::Integer(i))?;
      crate::evaluator::listable::push_list_element(&mut results, val);
    }
    return Ok(Expr::List(results.into()));
  } else if items.len() >= 3 {
    // {i, min, max} or {i, min, max, step} form
    let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
    let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;

    // Get step (default is 1)
    let step_expr = if items.len() >= 4 {
      crate::evaluator::evaluate_expr_to_expr(&items[3])?
    } else {
      Expr::Integer(1)
    };

    // Keep exact integer iteration behavior when possible.
    if let (Some(min_val), Some(max_val), Some(step_val)) = (
      expr_to_exact_i128(&min_expr),
      expr_to_exact_i128(&max_expr),
      expr_to_exact_i128(&step_expr),
    ) {
      if step_val == 0 {
        return Err(InterpreterError::EvaluationError(
          "Table: step cannot be zero".into(),
        ));
      }

      let mut results = Vec::new();
      let mut i = min_val;
      if step_val > 0 {
        while i <= max_val {
          let val = eval_at(&Expr::Integer(i))?;
          crate::evaluator::listable::push_list_element(&mut results, val);
          i += step_val;
        }
      } else {
        // Negative step
        while i >= max_val {
          let val = eval_at(&Expr::Integer(i))?;
          crate::evaluator::listable::push_list_element(&mut results, val);
          i += step_val;
        }
      }
      return Ok(Expr::List(results.into()));
    }

    // Symbolic path: if (max - min) / step simplifies to a real count,
    // iterate symbolically as min, min+step, …, min+n*step with
    // n = Floor[count] (no values when negative). Handles cases like
    // Table[x, {x, a, a+5n, n}] and {θ, Pi + s, 3.4557 + s, Pi/50} where
    // the bounds are symbolic but the iteration count is well-defined.
    let diff = crate::evaluator::evaluate_function_call_ast(
      "Subtract",
      &[max_expr.clone(), min_expr.clone()],
    )?;
    let ratio = crate::evaluator::evaluate_function_call_ast(
      "Divide",
      &[diff, step_expr.clone()],
    )?;
    let count = match &ratio {
      Expr::Integer(n) => Some(*n),
      other => crate::functions::math_ast::try_eval_to_f64(other)
        .filter(|f| f.is_finite())
        .map(range_step_count),
    };
    // Numeric bounds keep the machine-real path below.
    let numeric =
      |e: &Expr| crate::functions::math_ast::try_eval_to_f64(e).is_some();
    let count = count.filter(|_| {
      matches!(ratio, Expr::Integer(_))
        || !(numeric(&min_expr) && numeric(&max_expr))
    });
    if let Some(n) = count
      && n < 0
    {
      return Ok(Expr::List(vec![].into()));
    }
    if let Some(n) = count
      && (0..=1_000_000).contains(&n)
    {
      let mut results = Vec::with_capacity((n + 1) as usize);
      for k in 0..=n {
        // A machine-real step makes even the first value min + 0.0 a real.
        let current = if k == 0 && !matches!(step_expr, Expr::Real(_)) {
          min_expr.clone()
        } else {
          let k_step = crate::evaluator::evaluate_function_call_ast(
            "Times",
            &[Expr::Integer(k), step_expr.clone()],
          )?;
          crate::evaluator::evaluate_function_call_ast(
            "Plus",
            &[min_expr.clone(), k_step],
          )?
        };
        let val = eval_at(&current)?;
        crate::evaluator::listable::push_list_element(&mut results, val);
      }
      return Ok(Expr::List(results.into()));
    }

    // Fallback: numeric iteration for symbolic numeric bounds/step (e.g.
    // Pi/4). Non-numeric bounds/step leave the call unevaluated (the
    // dispatch-level check normally intercepts this first).
    if crate::functions::math_ast::try_eval_to_f64(&min_expr).is_none() {
      return Ok(table_unevaluated(body, iter_spec));
    }
    let Some(max_num) = crate::functions::math_ast::try_eval_to_f64(&max_expr)
    else {
      return Ok(table_unevaluated(body, iter_spec));
    };
    let Some(step_num) =
      crate::functions::math_ast::try_eval_to_f64_with_infinity(&step_expr)
    else {
      return Ok(table_unevaluated(body, iter_spec));
    };

    if step_num == 0.0 {
      return Err(InterpreterError::EvaluationError(
        "Table: step cannot be zero".into(),
      ));
    }

    let mut results = Vec::new();
    // `min + 0*step`: a machine-real step makes even the first value a
    // machine real (`{i, 1, 2, 1.}` starts at `1.`).
    let mut current_expr = nth_iterator_value(&min_expr, &step_expr, 0)?;
    let mut safety_counter: usize = 0;
    // Accumulated rounding (0.25 + 14 * 0.05 = 0.9500000000000003) must not
    // push the last value past `max`, so compare with a slack proportional
    // to the step rather than to machine epsilon.
    let slack = 1e-9 * step_num.abs();
    if step_num > 0.0 {
      loop {
        let current_num =
          crate::functions::math_ast::try_eval_to_f64_with_infinity(
            &current_expr,
          )
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Table: iterator value became non-numeric".into(),
            )
          })?;
        if current_num > max_num + slack {
          break;
        }
        let val = eval_at(&current_expr)?;
        crate::evaluator::listable::push_list_element(&mut results, val);
        safety_counter += 1;
        current_expr =
          nth_iterator_value(&min_expr, &step_expr, safety_counter)?;
        if safety_counter > 1_000_000 {
          return Err(InterpreterError::EvaluationError(
            "Table: iterator exceeded maximum iterations".into(),
          ));
        }
      }
    } else {
      loop {
        let current_num =
          crate::functions::math_ast::try_eval_to_f64_with_infinity(
            &current_expr,
          )
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Table: iterator value became non-numeric".into(),
            )
          })?;
        if current_num < max_num - slack {
          break;
        }
        let val = eval_at(&current_expr)?;
        crate::evaluator::listable::push_list_element(&mut results, val);
        safety_counter += 1;
        current_expr =
          nth_iterator_value(&min_expr, &step_expr, safety_counter)?;
        if safety_counter > 1_000_000 {
          return Err(InterpreterError::EvaluationError(
            "Table: iterator exceeded maximum iterations".into(),
          ));
        }
      }
    }
    return Ok(Expr::List(results.into()));
  }

  Err(InterpreterError::EvaluationError(
    "Table: invalid iterator specification".into(),
  ))
}

use crate::functions::math_ast::expr_to_rational;

/// The `k`-th iterator value `min + k*step`. Computing it from `min` each
/// time (rather than adding `step` to the previous value) keeps machine-real
/// steps free of accumulated rounding: `Table[t, {t, 0, 20, 0.1}]` ends
/// exactly at `20.`, as `Range` does, not at `20.000000000000014`.
pub(crate) fn nth_iterator_value(
  min_expr: &Expr,
  step_expr: &Expr,
  k: usize,
) -> Result<Expr, InterpreterError> {
  // Machine reals: one fused multiply-add, as Wolfram computes it
  // (`{c, 0.1, 3.1, 0.1}` reaches 1.8, not 1.8000000000000003), an exact
  // rational bound or step numericized first.
  let machine = super::utilities::numeric_expr_to_f64;
  if (matches!(min_expr, Expr::Real(_)) || matches!(step_expr, Expr::Real(_)))
    && let (Some(min), Some(step)) = (machine(min_expr), machine(step_expr))
  {
    return Ok(Expr::Real((k as f64).mul_add(step, min)));
  }
  // The start itself (an infinite step would make it 0*Infinity).
  if k == 0 {
    return crate::evaluator::evaluate_expr_to_expr(min_expr);
  }
  crate::evaluator::evaluate_expr_to_expr(&plus(vec![
    min_expr.clone(),
    crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(k as i128), step_expr.clone()],
    )?,
  ]))
}

/// Number of steps from `min` to `max` implied by `ratio = (max - min) /
/// step`, tolerant of floating-point rounding at an exact integer
/// boundary. Used by the symbolic/machine-real branches of [`range_ast`]
/// below, where `min`/`max`/`step` are only reachable as `f64` (e.g. via
/// `N[…]` on a `Pi`-based bound).
///
/// A ratio that is mathematically exactly integral can still land a hair
/// under it in `f64` — e.g. `Range[Pi/n, 2 Pi - Pi/n, 2 Pi/n]` has a ratio
/// of exactly `n - 1`, but computing it through `Pi`-based machine reals
/// can produce `n - 1 - epsilon`. A plain `.floor()` then reads that as
/// one step short and the range silently drops its last element (`Range[
/// Pi/360, 2 Pi - Pi/360, 2 Pi/360]` returned 359 points instead of 360
/// before this rounding tolerance was added). Snapping a ratio that is
/// within a relative `1e-9` of an integer to that integer fixes the
/// boundary without affecting a ratio that is genuinely non-integral
/// (which still truncates towards `min` exactly as `Range` requires).
pub(crate) fn range_step_count(ratio: f64) -> i128 {
  let rounded = ratio.round();
  if (ratio - rounded).abs() <= 1e-9 * rounded.abs().max(1.0) {
    rounded as i128
  } else {
    ratio.floor() as i128
  }
}

/// AST-based Range: generate a range of numbers.
/// Range[n] -> {1, 2, ..., n}
/// Range[min, max] -> {min, ..., max}
/// Range[min, max, step] -> {min, min+step, ..., max}
pub fn range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Ok(unevaluated("Range", args));
  }

  // Listable-style threading: Range[{a,b}, ...] => {Range[a, ...], Range[b, ...]}.
  // Mathematica threads Range over any list argument when the other
  // arguments are scalars, so e.g. Range[0, 999, {3, 5}] returns
  // {Range[0,999,3], Range[0,999,5]}.
  for (idx, arg) in args.iter().enumerate() {
    if let Expr::List(items) = arg {
      let mut threaded = Vec::with_capacity(items.len());
      for item in items {
        let mut new_args: Vec<Expr> = args.to_vec();
        new_args[idx] = item.clone();
        threaded.push(range_ast(&new_args)?);
      }
      return Ok(Expr::List(threaded.into()));
    }
  }

  // Try exact rational arithmetic first
  let all_rational = args.iter().all(|a| expr_to_rational(a).is_some());

  if all_rational {
    let (min_n, min_d, max_n, max_d, step_n, step_d) = if args.len() == 1 {
      let (mn, md) = expr_to_rational(&args[0]).unwrap();
      (1_i128, 1_i128, mn, md, 1_i128, 1_i128)
    } else if args.len() == 2 {
      let (min_n, min_d) = expr_to_rational(&args[0]).unwrap();
      let (max_n, max_d) = expr_to_rational(&args[1]).unwrap();
      (min_n, min_d, max_n, max_d, 1_i128, 1_i128)
    } else {
      let (min_n, min_d) = expr_to_rational(&args[0]).unwrap();
      let (max_n, max_d) = expr_to_rational(&args[1]).unwrap();
      let (step_n, step_d) = expr_to_rational(&args[2]).unwrap();
      (min_n, min_d, max_n, max_d, step_n, step_d)
    };

    if step_n == 0 {
      let call = unevaluated("Range", args);
      crate::emit_message(&format!(
        "Range::range: Range specification in {} does not have appropriate bounds.",
        crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
      ));
      return Ok(call);
    }

    let mut results = Vec::new();
    // Current value as (numerator, denominator)
    let mut cur_n = min_n;
    let mut cur_d = min_d;
    let step_positive = (step_n > 0) == (step_d > 0);

    loop {
      // Compare cur vs max: cur_n/cur_d vs max_n/max_d
      // cur_n * max_d vs max_n * cur_d (careful with sign of denominators)
      let lhs = cur_n * max_d;
      let rhs = max_n * cur_d;
      let denom_sign = (cur_d > 0) == (max_d > 0);

      if step_positive {
        // For positive step: stop when cur > max
        if denom_sign && lhs > rhs {
          break;
        }
        if !denom_sign && lhs < rhs {
          break;
        }
      } else {
        // For negative step: stop when cur < max
        if denom_sign && lhs < rhs {
          break;
        }
        if !denom_sign && lhs > rhs {
          break;
        }
      }

      results.push(crate::functions::math_ast::make_rational(cur_n, cur_d));

      // cur += step: cur_n/cur_d + step_n/step_d = (cur_n*step_d + step_n*cur_d) / (cur_d*step_d)
      cur_n = cur_n * step_d + step_n * cur_d;
      cur_d *= step_d;

      // Simplify to avoid overflow
      (cur_n, cur_d) = rat_reduce(cur_n, cur_d);

      if results.len() > MAX_RANGE_LENGTH {
        return Err(InterpreterError::EvaluationError(
          "Range: result too large".into(),
        ));
      }
    }

    return Ok(Expr::List(results.into()));
  }

  // Try symbolic range when arguments contain symbolic constants (e.g. Pi)
  let try_numeric = |e: &Expr| -> Option<f64> {
    if let Some(f) = expr_to_f64(e) {
      return Some(f);
    }
    // Try evaluating N[expr] to get a float
    let n_expr = call1("N", e.clone());
    if let Ok(evaled) = crate::evaluator::evaluate_expr_to_expr(&n_expr) {
      return expr_to_f64(&evaled);
    }
    None
  };

  let (min_expr, max_expr, step_expr) = if args.len() == 1 {
    (Expr::Integer(1), args[0].clone(), Expr::Integer(1))
  } else if args.len() == 2 {
    (args[0].clone(), args[1].clone(), Expr::Integer(1))
  } else {
    (args[0].clone(), args[1].clone(), args[2].clone())
  };

  // Check if any arg is non-numeric (symbolic). Exact rationals among
  // machine reals are numbers too: the elements are machine reals then.
  let literal_number = |e: &Expr| {
    matches!(e, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Rational")
  };
  let machine_literals = [&min_expr, &max_expr, &step_expr]
    .iter()
    .all(|e| literal_number(e))
    && (matches!(&min_expr, Expr::Real(_))
      || matches!(&step_expr, Expr::Real(_)));
  let has_symbolic = !machine_literals
    && (expr_to_f64(&min_expr).is_none()
      || expr_to_f64(&max_expr).is_none()
      || expr_to_f64(&step_expr).is_none());

  if has_symbolic {
    // Numeric contagion: Range[start, stop, di] is computed as
    // start + k*di for k = 0, 1, 2, .... When `di` (or `start` itself)
    // is an inexact machine real, every element becomes inexact —
    // including k=0, since `start + 0` still numericizes `start` via
    // arithmetic with an inexact operand (`0*di` is inexact `0.`).
    // `max`/`stop` never affects exactness — it only controls where the
    // range terminates.
    let any_real =
      matches!(&min_expr, Expr::Real(_)) || matches!(&step_expr, Expr::Real(_));
    let first_elem = |min_expr: &Expr| -> Result<Expr, InterpreterError> {
      if any_real {
        let n_expr = call1("N", min_expr.clone());
        crate::evaluator::evaluate_expr_to_expr(&n_expr)
      } else {
        Ok(min_expr.clone())
      }
    };

    // Try to evaluate numerically to determine count
    let min_f = try_numeric(&min_expr);
    let max_f = try_numeric(&max_expr);
    let step_f = try_numeric(&step_expr);

    // If the endpoints are purely symbolic (e.g. `Range[a, a+5]`) the
    // difference `max - min` — or the ratio `(max - min)/step` when the
    // step is itself symbolic (e.g. `Range[a, b, (b - a)/4]`) — may still
    // simplify to a numeric value. In that case we can enumerate the
    // range using the numeric count and render each element symbolically.
    if min_f.is_none() || max_f.is_none() {
      if let Some(0.0) = step_f {
        let call = unevaluated("Range", args);
        crate::emit_message(&format!(
          "Range::range: Range specification in {} does not have appropriate bounds.",
          crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
        ));
        return Ok(call);
      }
      // ratio = (max - min) / step
      let ratio_expr = times(vec![
        plus(vec![max_expr.clone(), neg(min_expr.clone())]),
        pow(step_expr.clone(), Expr::Integer(-1)),
      ]);
      // Accept rationals (e.g. `5/2`), floats, and symbolic reals like
      // `Pi` — fall through to `try_numeric` (which calls `N[expr]`)
      // when the ratio doesn't reduce to a literal Integer/Rational/Real.
      let evaled_ratio =
        crate::evaluator::evaluate_expr_to_expr(&ratio_expr).ok();
      let ratio_val = evaled_ratio.as_ref().and_then(|e| {
        expr_to_rational(e)
          .map(|(n, d)| n as f64 / d as f64)
          .or_else(|| try_numeric(e))
      });
      if let Some(ratio_val) = ratio_val
        && ratio_val.is_finite()
      {
        let count = range_step_count(ratio_val) + 1;
        if count <= 0 {
          return Ok(Expr::List(vec![].into()));
        }
        if count > MAX_RANGE_LENGTH as i128 {
          return Err(InterpreterError::EvaluationError(
            "Range: result too large".into(),
          ));
        }
        let mut results = Vec::with_capacity(count as usize);
        for k in 0..count {
          let elem = if k == 0 {
            first_elem(&min_expr)?
          } else {
            let k_times_step = times(vec![Expr::Integer(k), step_expr.clone()]);
            let sum = plus(vec![min_expr.clone(), k_times_step]);
            crate::evaluator::evaluate_expr_to_expr(&sum)?
          };
          results.push(elem);
        }
        return Ok(Expr::List(results.into()));
      }
      // Fully symbolic with no way to determine length — leave
      // unevaluated like Mathematica, with its message.
      let call = unevaluated("Range", args);
      crate::emit_message(&format!(
        "Range::range: Range specification in {} does not have appropriate bounds.",
        crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
      ));
      return Ok(call);
    }

    if let (Some(min_val), Some(max_val), Some(step_val)) =
      (min_f, max_f, step_f)
    {
      if step_val == 0.0 {
        let call = unevaluated("Range", args);
        crate::emit_message(&format!(
          "Range::range: Range specification in {} does not have appropriate bounds.",
          crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
        ));
        return Ok(call);
      }
      let count = range_step_count((max_val - min_val) / step_val) + 1;
      if count <= 0 {
        return Ok(Expr::List(vec![].into()));
      }
      if count > MAX_RANGE_LENGTH as i128 {
        return Err(InterpreterError::EvaluationError(
          "Range: result too large".into(),
        ));
      }
      let mut results = Vec::with_capacity(count as usize);
      for k in 0..count {
        // Generate min_expr + k * step_expr symbolically
        let elem = if k == 0 {
          first_elem(&min_expr)?
        } else {
          let k_times_step = times2(Expr::Integer(k), step_expr.clone());
          let sum = plus2(min_expr.clone(), k_times_step);
          crate::evaluator::evaluate_expr_to_expr(&sum)?
        };
        results.push(elem);
      }
      return Ok(Expr::List(results.into()));
    }

    // A bound or step that is no number (`Range[0, 1, f[x]]`).
    let call = unevaluated("Range", args);
    crate::emit_message(&format!(
      "Range::range: Range specification in {} does not have appropriate bounds.",
      crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
    ));
    return Ok(call);
  }

  // Fallback to f64 for Real arguments
  let as_f64 = |e: &Expr| {
    expr_to_f64(e)
      .or_else(|| crate::functions::math_ast::try_eval_to_f64(e))
      .unwrap_or(f64::NAN)
  };
  let (min, max, step) =
    (as_f64(&min_expr), as_f64(&max_expr), as_f64(&step_expr));

  if step == 0.0 {
    let call = unevaluated("Range", args);
    crate::emit_message(&format!(
      "Range::range: Range specification in {} does not have appropriate bounds.",
      crate::syntax::format_expr(&call, crate::syntax::ExprForm::Output)
    ));
    return Ok(call);
  }

  // Output elements are Real only when `min` or `step` is Real.
  // `max` only controls termination, so e.g. Range[3.5] yields integers.
  let any_real =
    matches!(&min_expr, Expr::Real(_)) || matches!(&step_expr, Expr::Real(_));

  // As wolframscript does: the count is `(max - min)/step` rounded down,
  // allowing for that ratio to land one rounding step under an integer
  // (Range[-0.8, 3.8, 0.01] has 461 elements, although the ratio is
  // 459.99999999999994); each element is `min + k*step` with a single
  // rounding, and a last element past `max` is `max` itself.
  let ratio = (max - min) / step;
  if !ratio.is_finite() {
    return Ok(Expr::List(vec![].into()));
  }
  let steps = (ratio * (1.0 + f64::EPSILON)).floor();
  if steps < 0.0 {
    return Ok(Expr::List(vec![].into()));
  }
  if steps >= MAX_RANGE_LENGTH as f64 {
    return Err(InterpreterError::EvaluationError(
      "Range: result too large".into(),
    ));
  }
  let count = steps as usize + 1;
  let mut results = Vec::with_capacity(count);
  for k in 0..count {
    let mut val = (k as f64).mul_add(step, min);
    if (step > 0.0 && val > max) || (step < 0.0 && val < max) {
      val = max;
    }
    results.push(if any_real {
      Expr::Real(val)
    } else {
      f64_to_expr(val)
    });
  }

  Ok(Expr::List(results.into()))
}

/// PowerRange[min, max] generates {min, min*10, min*100, ...} up to max.
/// PowerRange[min, max, r] uses factor r instead of 10.
pub fn power_range_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(unevaluated("PowerRange", args));
  }

  // Try rational arithmetic
  let all_rational = args.iter().all(|a| expr_to_rational(a).is_some());

  if all_rational {
    let (min_n, min_d) = expr_to_rational(&args[0]).unwrap();
    let (max_n, max_d) = expr_to_rational(&args[1]).unwrap();
    let (fac_n, fac_d) = if args.len() == 3 {
      expr_to_rational(&args[2]).unwrap()
    } else {
      (10, 1)
    };

    // Repeatedly multiplying can only stay between the bounds if both have
    // the same sign and neither is zero.
    if (min_n == 0) || (max_n == 0) || ((min_n > 0) != (max_n > 0)) {
      crate::emit_message(&format!(
        "PowerRange::range: Range specification in {} does not have \
         appropriate bounds.",
        crate::syntax::format_expr(
          &unevaluated("PowerRange", args),
          crate::syntax::ExprForm::Output
        )
      ));
      return Ok(unevaluated("PowerRange", args));
    }
    // A factor of 0 collapses and a factor of 1 never advances.
    if fac_n == 0 || (fac_n == fac_d) {
      crate::emit_message(&format!(
        "PowerRange::factor: Factor cannot be {}.",
        i32::from(fac_n != 0)
      ));
      return Ok(unevaluated("PowerRange", args));
    }

    // Terms are compared by magnitude, and the direction of travel comes
    // from the factor: a factor bigger than 1 grows towards the bound and
    // a smaller one shrinks towards it. Comparing magnitudes is what stops
    // a negative factor one term early -- `PowerRange[1, 5, -2]` is
    // `{1, -2, 4}`, not `{1, -2, 4, -8}` -- and what makes a range whose
    // bounds run against the factor come back empty.
    let magnitude = |n: i128, d: i128| (n.unsigned_abs(), d.unsigned_abs());
    let (max_mag_n, max_mag_d) = magnitude(max_n, max_d);
    let (fac_mag_n, fac_mag_d) = magnitude(fac_n, fac_d);
    let growing = fac_mag_n > fac_mag_d;
    let mut results = Vec::new();
    let (mut cur_n, mut cur_d) = (min_n, min_d);
    loop {
      let (cur_mag_n, cur_mag_d) = magnitude(cur_n, cur_d);
      let (Some(left), Some(right)) = (
        cur_mag_n.checked_mul(max_mag_d),
        max_mag_n.checked_mul(cur_mag_d),
      ) else {
        break;
      };
      if if growing { left > right } else { left < right } {
        break;
      }
      results.push(crate::functions::math_ast::make_rational(cur_n, cur_d));

      // cur *= factor, stopping rather than wrapping if it no longer fits.
      let (Some(next_n), Some(next_d)) =
        (cur_n.checked_mul(fac_n), cur_d.checked_mul(fac_d))
      else {
        break;
      };
      (cur_n, cur_d) = rat_reduce(next_n, next_d);

      if results.len() > MAX_RANGE_LENGTH {
        return Err(InterpreterError::EvaluationError(
          "PowerRange: result too large".into(),
        ));
      }
    }

    return Ok(Expr::List(results.into()));
  }

  Ok(unevaluated("PowerRange", args))
}

/// AST-based ConstantArray: create array filled with constant.
/// ConstantArray[c, n] -> {c, c, ..., c} (n times)
/// ConstantArray[c, {n1, n2}] -> nested array
pub fn constant_array_ast(
  elem: &Expr,
  dims: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated =
    || Ok(call("ConstantArray", vec![elem.clone(), dims.clone()]));
  // A dimension must be a non-negative machine integer; anything else
  // (negative, non-integer, symbolic, or too large for i128) leaves the call
  // unevaluated rather than raising an error — matching wolframscript, which
  // returns the unevaluated form (or a SymbolicZerosArray for symbolic dims).
  let nonneg_dim = |e: &Expr| -> Option<usize> {
    match e {
      Expr::Integer(n) if *n >= 0 => usize::try_from(*n).ok(),
      Expr::BigInteger(_) => expr_to_i128(e)
        .filter(|&n| n >= 0)
        .and_then(|n| usize::try_from(n).ok()),
      _ => None,
    }
  };
  // Flatten the dimension specifier into a list (a scalar dimension counts as
  // a single-element list, matching wolframscript's `SymbolicZerosArray[{n}]`).
  let dim_exprs: Vec<Expr> = match dims {
    Expr::List(dim_list) => dim_list.iter().cloned().collect(),
    other => vec![other.clone()],
  };

  // Classify each dimension. A concrete non-negative machine integer is a
  // usable dimension; a concrete number that isn't (negative, non-integer,
  // too large) makes wolframscript emit `::ilsmn` and leave the call
  // unevaluated; anything else is symbolic.
  let mut concrete_dims: Vec<usize> = Vec::with_capacity(dim_exprs.len());
  let mut has_symbolic = false;
  for d in &dim_exprs {
    if let Some(n) = nonneg_dim(d) {
      concrete_dims.push(n);
    } else {
      // A concrete but invalid dimension (negative, non-integer, too large)
      // makes wolframscript emit `::ilsmn` and leave the call unevaluated; a
      // symbolic dimension yields a SymbolicZeros/OnesArray placeholder.
      if crate::functions::predicate_ast::is_numeric_q(d) {
        crate::emit_message(&format!(
          "ConstantArray::ilsmn: Single or list of non-negative machine-sized integers expected at position 2 of {}.",
          expr_to_string(&call(
            "ConstantArray",
            vec![elem.clone(), dims.clone()]
          ))
        ));
        return unevaluated();
      }
      has_symbolic = true;
    }
  }

  if !has_symbolic {
    // Build the nested constant array from the innermost dimension outward.
    let mut result = elem.clone();
    for &n in concrete_dims.iter().rev() {
      result = Expr::List(vec![result; n].into());
    }
    return Ok(result);
  }

  // At least one symbolic dimension (and no invalid concrete ones): produce a
  // SymbolicZerosArray/SymbolicOnesArray placeholder like wolframscript.
  //   ConstantArray[0, dims] -> SymbolicZerosArray[dims]
  //   ConstantArray[1, dims] -> SymbolicOnesArray[dims]
  //   ConstantArray[c, dims] -> c*SymbolicOnesArray[dims]
  let dims_list = Expr::List(dim_exprs.into());
  let ones = || call1("SymbolicOnesArray", dims_list.clone());
  let result = match elem {
    Expr::Integer(0) => call1("SymbolicZerosArray", dims_list.clone()),
    Expr::Integer(1) => ones(),
    _ => times2(elem.clone(), ones()),
  };
  Ok(result)
}

/// True when `body` references `var_name` as the head of a `FunctionCall`
/// anywhere in the tree. The substitute-variable path turns such a call
/// into `CurriedCall { func: <value>, args }`, so it works when `var` is
/// bound to a value that isn't a function (e.g. an Association). The
/// ENV-binding fast path can't reproduce that — it would dispatch the
/// call by literal name. When this returns true we have to fall back to
/// the substitute path to preserve semantics.
fn body_uses_var_as_function_head(body: &Expr, var_name: &str) -> bool {
  match body {
    Expr::FunctionCall { name, args } => {
      name == var_name
        || args
          .iter()
          .any(|a| body_uses_var_as_function_head(a, var_name))
    }
    Expr::List(items) => items
      .iter()
      .any(|a| body_uses_var_as_function_head(a, var_name)),
    Expr::BinaryOp { left, right, .. } => {
      body_uses_var_as_function_head(left, var_name)
        || body_uses_var_as_function_head(right, var_name)
    }
    Expr::UnaryOp { operand, .. } => {
      body_uses_var_as_function_head(operand, var_name)
    }
    Expr::Comparison { operands, .. } => operands
      .iter()
      .any(|a| body_uses_var_as_function_head(a, var_name)),
    Expr::CompoundExpr(exprs) => exprs
      .iter()
      .any(|a| body_uses_var_as_function_head(a, var_name)),
    Expr::CurriedCall { func, args } => {
      body_uses_var_as_function_head(func, var_name)
        || args
          .iter()
          .any(|a| body_uses_var_as_function_head(a, var_name))
    }
    Expr::Part { expr, index } => {
      body_uses_var_as_function_head(expr, var_name)
        || body_uses_var_as_function_head(index, var_name)
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      body_uses_var_as_function_head(pattern, var_name)
        || body_uses_var_as_function_head(replacement, var_name)
    }
    Expr::ReplaceAll { expr: e, rules }
    | Expr::ReplaceRepeated { expr: e, rules } => {
      body_uses_var_as_function_head(e, var_name)
        || body_uses_var_as_function_head(rules, var_name)
    }
    Expr::Map { func, list }
    | Expr::Apply { func, list }
    | Expr::MapApply { func, list } => {
      body_uses_var_as_function_head(func, var_name)
        || body_uses_var_as_function_head(list, var_name)
    }
    Expr::Association(items) => items.iter().any(|(k, v)| {
      body_uses_var_as_function_head(k, var_name)
        || body_uses_var_as_function_head(v, var_name)
    }),
    _ => false,
  }
}

/// Bind a Do-loop iterator variable in ENV. Reuses the existing slot when
/// possible to avoid cloning `var_name` into a fresh HashMap key on every
/// iteration.
fn bind_loop_var(var_name: &str, value: Expr) {
  crate::ENV.with(|e| {
    let mut env = e.borrow_mut_own_value(var_name);
    if let Some(slot) = env.get_mut(var_name) {
      *slot = crate::StoredValue::ExprVal(value);
    } else {
      env.insert(var_name.to_string(), crate::StoredValue::ExprVal(value));
    }
  });
}

/// Restore the previous binding of a loop iterator variable (or remove
/// the slot if the variable was previously unbound).
fn restore_loop_var(var_name: &str, prev: Option<crate::StoredValue>) {
  crate::ENV.with(|e| {
    let mut env = e.borrow_mut_own_value(var_name);
    if let Some(v) = prev {
      env.insert(var_name.to_string(), v);
    } else {
      env.remove(var_name);
    }
  });
}

/// AST-based Do: execute expression multiple times.
/// Do[expr, n] -> execute expr n times
/// Do[expr, {i, max}] -> execute with i from 1 to max
/// Do[expr, {i, min, max}] -> execute with i from min to max
pub fn do_ast(body: &Expr, iter_spec: &Expr) -> Result<Expr, InterpreterError> {
  // A bare (non-list) iterator spec may be a symbol or expression that
  // evaluates to a repeat count (`Do[expr, n]`, `Do[expr, 500 - iStart]`),
  // not just a literal integer. Resolve it before dispatching, mirroring
  // table_ast; the caller (table_iterators_invalid) has already confirmed a
  // bare spec resolves to a number.
  let resolved_spec;
  let iter_spec: &Expr = if matches!(
    iter_spec,
    Expr::List(_) | Expr::Integer(_) | Expr::BigInteger(_)
  ) {
    iter_spec
  } else {
    resolved_spec = crate::evaluator::evaluate_expr_to_expr(iter_spec)?;
    &resolved_spec
  };
  match iter_spec {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let n = expr_to_i128(iter_spec).unwrap_or(0);
      for _ in 0..n {
        match crate::evaluator::evaluate_expr_to_expr(body) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
          Err(e) => return Err(e),
        }
      }
      Ok(null_expr())
    }
    Expr::List(items) if items.len() == 1 => {
      // Do[body, {n}] — repeat n times without iterator variable
      let n_expr = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
      let n = repeat_count(&n_expr)?;
      let mut done: i128 = 0;
      while n.is_none_or(|n| done < n) {
        done += 1;
        match crate::evaluator::evaluate_expr_to_expr(body) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
          Err(e) => return Err(e),
        }
      }
      Ok(null_expr())
    }
    Expr::List(items) if items.len() >= 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        other => {
          // A non-symbol iterator (Do[body, {3, 1, 5}]) is a raw object that
          // cannot be used as an iterator: wolframscript emits Do::itraw and
          // returns the call unevaluated rather than raising an error.
          crate::emit_message(&format!(
            "Do::itraw: Raw object {} cannot be used as an iterator.",
            expr_to_string(other)
          ));
          return Ok(call("Do", vec![body.clone(), iter_spec.clone()]));
        }
      };

      // Handle list iterator: Do[body, {i, {a, b, c}}]
      if items.len() == 2 {
        let needs_substitute = body_uses_var_as_function_head(body, &var_name);

        // Fast path: `Do[body, {c, Characters[s]}]` — iterate the
        // string's chars directly instead of allocating an N-element
        // Expr::List of single-char Expr::Strings. The accumulating
        // splitLines loop pays one such call per ~75 KB input.
        if !needs_substitute
          && let Expr::FunctionCall {
            name: head,
            args: head_args,
          } = &items[1]
          && head == "Characters"
          && head_args.len() == 1
        {
          let s_expr = crate::evaluator::evaluate_expr_to_expr(&head_args[0])?;
          if let Expr::String(ref s) = s_expr {
            let prev = crate::ENV
              .with(|e| e.borrow_mut_own_value(&var_name).remove(&var_name));
            let mut early_return: Option<Expr> = None;
            let mut error: Option<InterpreterError> = None;
            let mut buf = [0u8; 4];
            for ch in s.chars() {
              let cs = ch.encode_utf8(&mut buf).to_string();
              bind_loop_var(&var_name, Expr::String(cs));
              match crate::evaluator::evaluate_expr_to_expr(body) {
                Ok(_) => {}
                Err(InterpreterError::BreakSignal) => break,
                Err(InterpreterError::ContinueSignal) => {}
                Err(InterpreterError::ReturnValue(val)) => {
                  early_return = Some(*val);
                  break;
                }
                Err(e) => {
                  error = Some(e);
                  break;
                }
              }
            }
            restore_loop_var(&var_name, prev);
            if let Some(e) = error {
              return Err(e);
            }
            if let Some(v) = early_return {
              return Ok(v);
            }
            return Ok(null_expr());
          }
          // Fall through to the generic path if Characters[s] didn't
          // produce a String (e.g. threaded over a list).
        }

        let val_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        if let Expr::List(list_items) = &val_expr {
          // Bind the iterator in ENV instead of cloning + substituting
          // the body on every iteration. The substitute path used to
          // walk the entire body AST per step, which made tight loops
          // (e.g. `Do[…, {c, Characters[s]}]` over 75 KB inputs) pay
          // O(body_size) work per iteration. ENV binding makes that
          // cost O(1) by resolving `var_name` through the normal
          // identifier-lookup path.
          if !needs_substitute {
            let prev = crate::ENV
              .with(|e| e.borrow_mut_own_value(&var_name).remove(&var_name));
            let mut early_return: Option<Expr> = None;
            let mut error: Option<InterpreterError> = None;
            for item in list_items {
              bind_loop_var(&var_name, item.clone());
              match crate::evaluator::evaluate_expr_to_expr(body) {
                Ok(_) => {}
                Err(InterpreterError::BreakSignal) => break,
                Err(InterpreterError::ContinueSignal) => {}
                Err(InterpreterError::ReturnValue(val)) => {
                  early_return = Some(*val);
                  break;
                }
                Err(e) => {
                  error = Some(e);
                  break;
                }
              }
            }
            restore_loop_var(&var_name, prev);
            if let Some(e) = error {
              return Err(e);
            }
            if let Some(v) = early_return {
              return Ok(v);
            }
            return Ok(null_expr());
          }
          // Fall back to substitute when body uses var as a function head.
          for item in list_items {
            let substituted =
              crate::syntax::substitute_variable(body, &var_name, item);
            match crate::evaluator::evaluate_expr_to_expr(&substituted) {
              Ok(_) => {}
              Err(InterpreterError::BreakSignal) => break,
              Err(InterpreterError::ContinueSignal) => {}
              Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
              Err(e) => return Err(e),
            }
          }
          return Ok(null_expr());
        }
      }

      let (min_expr, max_expr, step_expr) = if items.len() == 2 {
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        (Expr::Integer(1), max_expr, Expr::Integer(1))
      } else if items.len() >= 3 {
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
        let step_expr = if items.len() >= 4 {
          crate::evaluator::evaluate_expr_to_expr(&items[3])?
        } else {
          Expr::Integer(1)
        };
        (min_expr, max_expr, step_expr)
      } else {
        return Err(InterpreterError::EvaluationError(
          "Do: invalid iterator specification".into(),
        ));
      };

      let needs_substitute = body_uses_var_as_function_head(body, &var_name);
      if needs_substitute {
        let mut early_return: Option<Expr> = None;
        super::utilities::for_each_numeric_step(
          min_expr,
          &max_expr,
          step_expr,
          |current| {
            let substituted =
              crate::syntax::substitute_variable(body, &var_name, &current);
            match crate::evaluator::evaluate_expr_to_expr(&substituted) {
              Ok(_) => Ok(true),
              Err(InterpreterError::BreakSignal) => Ok(false),
              Err(InterpreterError::ContinueSignal) => Ok(true),
              Err(InterpreterError::ReturnValue(val)) => {
                early_return = Some(*val);
                Ok(false)
              }
              Err(e) => Err(e),
            }
          },
        )?;
        if let Some(v) = early_return {
          return Ok(v);
        }
        return Ok(null_expr());
      }
      // ENV-binding fast path.
      let prev = crate::ENV
        .with(|e| e.borrow_mut_own_value(&var_name).remove(&var_name));
      let mut early_return: Option<Expr> = None;
      let result = super::utilities::for_each_numeric_step(
        min_expr,
        &max_expr,
        step_expr,
        |current| {
          crate::ENV.with(|e| {
            e.borrow_mut_own_value(&var_name)
              .insert(var_name.clone(), crate::StoredValue::ExprVal(current));
          });
          match crate::evaluator::evaluate_expr_to_expr(body) {
            Ok(_) => Ok(true),
            Err(InterpreterError::BreakSignal) => Ok(false),
            Err(InterpreterError::ContinueSignal) => Ok(true),
            Err(InterpreterError::ReturnValue(val)) => {
              early_return = Some(*val);
              Ok(false)
            }
            Err(e) => Err(e),
          }
        },
      );
      crate::ENV.with(|e| {
        let mut env = e.borrow_mut_own_value(&var_name);
        if let Some(v) = prev {
          env.insert(var_name.clone(), v);
        } else {
          env.remove(&var_name);
        }
      });
      result?;
      if let Some(v) = early_return {
        return Ok(v);
      }
      Ok(null_expr())
    }
    _ => Err(InterpreterError::EvaluationError(
      "Do: invalid iterator specification".into(),
    )),
  }
}

/// Multi-iterator Do: `Do[body, iter1, iter2, ...]`.
///
/// In Wolfram, a multi-iterator `Do` is a single construct: `Break[]` and
/// `Return[]` exit the entire `Do`, not just the innermost iterator. We
/// implement this by recursing over the iterator list, evaluating `body`
/// at the innermost level. The recursion is performed on a special inner
/// helper that does NOT catch `Break`/`Return`; only this outer wrapper
/// catches them so they propagate through all levels.
pub fn do_multi_ast(
  body: &Expr,
  iter_specs: &[Expr],
) -> Result<Expr, InterpreterError> {
  // A non-symbol iterator in any spec (Do[body, {3, 1, 5}, {j, 1, 2}]) is a
  // raw object that cannot be used as an iterator: emit Do::itraw and return
  // the call unevaluated, matching wolframscript, rather than crashing.
  for spec in iter_specs {
    if let Expr::List(items) = spec
      && items.len() >= 2
      && !matches!(&items[0], Expr::Identifier(_))
    {
      crate::emit_message(&format!(
        "Do::itraw: Raw object {} cannot be used as an iterator.",
        expr_to_string(&items[0])
      ));
      return Ok(call(
        "Do",
        std::iter::once(body.clone())
          .chain(iter_specs.iter().cloned())
          .collect(),
      ));
    }
  }
  match do_multi_inner(body, iter_specs) {
    Ok(()) => Ok(null_expr()),
    Err(InterpreterError::BreakSignal) => Ok(null_expr()),
    Err(InterpreterError::ReturnValue(val)) => Ok(*val),
    Err(e) => Err(e),
  }
}

fn do_multi_inner(
  body: &Expr,
  iter_specs: &[Expr],
) -> Result<(), InterpreterError> {
  if iter_specs.is_empty() {
    // Evaluate body; let Break/Return/Continue propagate up.
    match crate::evaluator::evaluate_expr_to_expr(body) {
      Ok(_) => Ok(()),
      Err(InterpreterError::ContinueSignal) => Ok(()),
      Err(e) => Err(e),
    }
  } else {
    let iter_spec = &iter_specs[0];
    let rest = &iter_specs[1..];
    iterate_spec(iter_spec, &mut |_| do_multi_inner(body, rest))
  }
}

/// The number of repetitions of a `{n}` iterator: a real `n` runs
/// `Floor[n]` times (`Do[body, {2.7}]` runs twice) and `Infinity` until a
/// `Break[]` (`None`).
fn repeat_count(n: &Expr) -> Result<Option<i128>, InterpreterError> {
  if let Some(k) = expr_to_i128(n) {
    return Ok(Some(k));
  }
  match crate::functions::math_ast::try_eval_to_f64_with_infinity(n) {
    Some(f) if f == f64::INFINITY => Ok(None),
    Some(f) if f.is_finite() => Ok(Some(f.floor() as i128)),
    _ => Err(InterpreterError::EvaluationError(
      "Do: repeat count must be a real number".into(),
    )),
  }
}

/// Helper: drive a single iterator spec, calling `step` once per iteration.
/// `step` receives the current iteration index (0-based) for informational
/// purposes; the iterator variable (if any) is bound in ENV before each call.
/// Errors from `step` (Break/Return/etc.) propagate up.
fn iterate_spec<F>(
  iter_spec: &Expr,
  step: &mut F,
) -> Result<(), InterpreterError>
where
  F: FnMut(usize) -> Result<(), InterpreterError>,
{
  match iter_spec {
    Expr::Integer(_) | Expr::BigInteger(_) => {
      let n = expr_to_i128(iter_spec).unwrap_or(0);
      for i in 0..n {
        step(i as usize)?;
      }
      Ok(())
    }
    Expr::List(items) if items.len() == 1 => {
      let n_expr = crate::evaluator::evaluate_expr_to_expr(&items[0])?;
      let n = repeat_count(&n_expr)?;
      let mut i: i128 = 0;
      while n.is_none_or(|n| i < n) {
        step(i as usize)?;
        i += 1;
      }
      Ok(())
    }
    Expr::List(items) if items.len() >= 2 => {
      let var_name = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Do: iterator variable must be an identifier".into(),
          ));
        }
      };

      // Iterator over an explicit list: Do[..., {v, {a, b, c}}]
      if items.len() == 2 {
        let val_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        if let Expr::List(list_items) = &val_expr {
          let prev = crate::ENV
            .with(|e| e.borrow_mut_own_value(&var_name).remove(&var_name));
          let mut err: Option<InterpreterError> = None;
          for (i, item) in list_items.iter().enumerate() {
            bind_loop_var(&var_name, item.clone());
            match step(i) {
              Ok(()) => {}
              Err(e) => {
                err = Some(e);
                break;
              }
            }
          }
          restore_loop_var(&var_name, prev);
          if let Some(e) = err {
            return Err(e);
          }
          return Ok(());
        }
      }

      // Numeric range iterator: {v, max} or {v, min, max} or {v, min, max, step}.
      let (min_expr, max_expr, step_expr) = if items.len() == 2 {
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        (Expr::Integer(1), max_expr, Expr::Integer(1))
      } else {
        let min_expr = crate::evaluator::evaluate_expr_to_expr(&items[1])?;
        let max_expr = crate::evaluator::evaluate_expr_to_expr(&items[2])?;
        let step_expr = if items.len() >= 4 {
          crate::evaluator::evaluate_expr_to_expr(&items[3])?
        } else {
          Expr::Integer(1)
        };
        (min_expr, max_expr, step_expr)
      };

      let prev = crate::ENV
        .with(|e| e.borrow_mut_own_value(&var_name).remove(&var_name));
      let mut err: Option<InterpreterError> = None;
      let mut idx = 0usize;
      let result = super::utilities::for_each_numeric_step(
        min_expr,
        &max_expr,
        step_expr,
        |current| {
          crate::ENV.with(|e| {
            e.borrow_mut_own_value(&var_name)
              .insert(var_name.clone(), crate::StoredValue::ExprVal(current));
          });
          let r = step(idx);
          idx += 1;
          match r {
            Ok(()) => Ok(true),
            Err(e) => {
              err = Some(e);
              Ok(false)
            }
          }
        },
      );
      restore_loop_var(&var_name, prev);
      if let Some(e) = err {
        return Err(e);
      }
      result?;
      Ok(())
    }
    _ => Err(InterpreterError::EvaluationError(
      "Do: invalid iterator specification".into(),
    )),
  }
}

/// Array[f, n] - creates a list by applying f to indices 1..n
pub fn array_ast(func: &Expr, n: i128) -> Result<Expr, InterpreterError> {
  let mut result = Vec::new();
  for i in 1..=n {
    let arg = Expr::Integer(i);
    let val = apply_func_ast(func, &arg)?;
    crate::evaluator::listable::push_list_element(&mut result, val);
  }
  Ok(Expr::List(result.into()))
}

/// Build the sequence of index values for a dimension given an integer
/// starting offset. Indices are `start, start+1, ..., start+n-1`, built as
/// Expr values so the result preserves integer exactness.
fn build_offset_indices(
  n: i128,
  start: &Expr,
) -> Result<Vec<Expr>, InterpreterError> {
  let mut result = Vec::with_capacity(n.max(0) as usize);
  for i in 0..n {
    let val = plus(vec![start.clone(), Expr::Integer(i)]);
    result.push(crate::evaluator::evaluate_expr_to_expr(&val)?);
  }
  Ok(result)
}

/// Build the sequence of index values for a dimension given a range
/// `{a, b}`. Produces `n` values evenly spaced from `a` to `b`, using exact
/// rational arithmetic when the endpoints are exact.
fn build_range_indices(
  n: i128,
  a: &Expr,
  b: &Expr,
) -> Result<Vec<Expr>, InterpreterError> {
  if n <= 0 {
    return Ok(Vec::new());
  }
  if n == 1 {
    // Wolfram gives the midpoint (a + b) / 2 when requesting a single
    // sample over a range, matching Array[f, 1, {a, b}] → {f[(a+b)/2]}.
    let half = pow(Expr::Integer(2), Expr::Integer(-1));
    let mid = times(vec![half, plus(vec![a.clone(), b.clone()])]);
    return Ok(vec![crate::evaluator::evaluate_expr_to_expr(&mid)?]);
  }
  let mut result = Vec::with_capacity(n as usize);
  // value_i = a + i * (b - a) / (n - 1)
  let diff = plus(vec![b.clone(), neg(a.clone())]);
  let inv_denom = pow(Expr::Integer(n - 1), Expr::Integer(-1));
  for i in 0..n {
    let term = times(vec![Expr::Integer(i), diff.clone(), inv_denom.clone()]);
    let val = plus(vec![a.clone(), term]);
    result.push(crate::evaluator::evaluate_expr_to_expr(&val)?);
  }
  Ok(result)
}

/// Array[f, {n1, n2, ...}] - multi-dimensional array
/// Array[f, dims, origin] - with custom origin
/// Array[f, dims, origin, head] - with custom head instead of List
pub fn array_multi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let func = &args[0];
  let dims_expr = &args[1];

  let dims: Vec<i128> = match dims_expr {
    Expr::List(items) => {
      let mut d = Vec::new();
      for item in items {
        if let Some(n) = expr_to_i128(item) {
          d.push(n);
        } else {
          return Ok(unevaluated("Array", args));
        }
      }
      d
    }
    _ => {
      if let Some(n) = expr_to_i128(dims_expr) {
        vec![n]
      } else {
        return Ok(unevaluated("Array", args));
      }
    }
  };

  // Build the list of index Exprs per dimension using exact arithmetic so
  // integer/rational inputs stay exact.
  //
  // Wolfram supports several forms for the origin/range argument:
  //   Array[f, n]                  → indices 1..n        (start=1, step=1)
  //   Array[f, n, r]               → indices r..r+n-1    (start=r, step=1)
  //   Array[f, n, {a, b}]          → n values from a..b  (start=a, step=(b-a)/(n-1))
  //   Array[f, {n1,...}, {r1,...}] → per-dim origin
  //   Array[f, {n1,...}, {{a1,b1},...}] → per-dim range
  let dim_indices: Vec<Vec<Expr>> = if args.len() >= 3 {
    let origin = &args[2];
    match origin {
      // Single-dim {a, b} range form. Ambiguous with two per-dim origins,
      // so only treat as a range when there is exactly one dimension.
      Expr::List(items) if dims.len() == 1 && items.len() == 2 => {
        vec![build_range_indices(dims[0], &items[0], &items[1])?]
      }
      Expr::List(items) => items
        .iter()
        .enumerate()
        .map(|(i, item)| match item {
          Expr::List(pair) if pair.len() == 2 => {
            build_range_indices(dims[i], &pair[0], &pair[1])
          }
          _ => build_offset_indices(dims[i], item),
        })
        .collect::<Result<Vec<_>, _>>()?,
      _ => dims
        .iter()
        .map(|&n| build_offset_indices(n, origin))
        .collect::<Result<Vec<_>, _>>()?,
    }
  } else {
    dims
      .iter()
      .map(|&n| build_offset_indices(n, &Expr::Integer(1)))
      .collect::<Result<Vec<_>, _>>()?
  };

  // Parse optional head (4th arg)
  let head: Option<&str> = if args.len() >= 4 {
    match &args[3] {
      Expr::Identifier(h) => Some(h.as_str()),
      _ => None,
    }
  } else {
    None
  };

  // Build the array recursively.
  // `dim_indices[depth]` is the list of index values for that dimension.
  fn build_array(
    func: &Expr,
    dim_indices: &[Vec<Expr>],
    depth: usize,
    indices: &mut Vec<Expr>,
  ) -> Result<Expr, InterpreterError> {
    if depth >= dim_indices.len() {
      let index_args: Vec<Expr> = indices.clone();
      if index_args.len() == 1 {
        apply_func_ast(func, &index_args[0])
      } else {
        // For multi-arg, evaluate f[i1, i2, ...]
        match func {
          Expr::Identifier(name) => {
            crate::evaluator::evaluate_function_call_ast(name, &index_args)
          }
          // `Subscript[a, ##] &` — substitute each `#k`/`##` slot with the
          // index args, then evaluate the body so e.g.
          // `Array[Subscript[a, ##] &, {2, 2}]` produces concrete
          // `Subscript[a, i, j]` cells.
          Expr::Function { body } => {
            let substituted =
              crate::syntax::substitute_slots(body, &index_args);
            crate::evaluator::evaluate_expr_to_expr(&substituted)
          }
          // `Function[{i, j, …}, body]` — bind named params positionally.
          Expr::NamedFunction { params, body, .. } => {
            let bindings: Vec<(&str, &Expr)> = params
              .iter()
              .zip(index_args.iter())
              .map(|(p, a)| (p.as_str(), a))
              .collect();
            let substituted =
              crate::syntax::substitute_variables(body, &bindings);
            crate::evaluator::evaluate_expr_to_expr(&substituted)
          }
          _ => {
            // Fall back to invoking the function via the standard call form
            // (e.g. Composition, named symbols stored as values, …).
            let func_str = expr_to_string(func);
            crate::evaluator::evaluate_function_call_ast(&func_str, &index_args)
          }
        }
      }
    } else {
      let mut items = Vec::new();
      for index_expr in &dim_indices[depth] {
        indices.push(index_expr.clone());
        let val = build_array(func, dim_indices, depth + 1, indices)?;
        // At the innermost dimension `val` is `f`'s leaf application result,
        // which may be `Nothing`/`Sequence[…]` (e.g. `Array[If[# != 2, #,
        // ##&[]] &, {5}]`); at any outer dimension it's always a full
        // `Expr::List` sub-array, which `push_list_element` just pushes
        // through unchanged. Routing every depth through it uniformly keeps
        // `Array`'s multi-dimensional forms in sync with the bare-integer
        // `Array[f, n]` form (`array_ast`, above), which does the same.
        crate::evaluator::listable::push_list_element(&mut items, val);
        indices.pop();
      }
      Ok(Expr::List(items.into()))
    }
  }

  let result = build_array(func, &dim_indices, 0, &mut Vec::new())?;

  // If a custom head is specified, replace List at every level of the
  // generated result with that head and evaluate the substituted
  // expression. When head == "List" this is a no-op.
  if let Some(h) = head {
    if h == "List" {
      Ok(result)
    } else {
      fn replace_list_head(
        expr: &Expr,
        head: &str,
      ) -> Result<Expr, InterpreterError> {
        match expr {
          Expr::List(items) => {
            let new_items: Result<Vec<Expr>, InterpreterError> =
              items.iter().map(|it| replace_list_head(it, head)).collect();
            crate::evaluator::evaluate_function_call_ast(head, &new_items?)
          }
          _ => Ok(expr.clone()),
        }
      }
      replace_list_head(&result, h)
    }
  } else {
    Ok(result)
  }
}

/// Parse a dimension specification into a list of sizes. Accepts a List of
/// non-negative integers, or a single non-negative integer for 1D.
/// Rank of the positions in a rule list (`{i, j} -> v`, `Band[{i, j}] -> v`).
fn sparse_rule_rank(data: &Expr) -> Option<usize> {
  let first = match data {
    Expr::List(items) => items.first()?,
    rule => rule,
  };
  let (Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }) = first
  else {
    return None;
  };
  match pattern.as_ref() {
    Expr::List(p) if !p.iter().any(|x| matches!(x, Expr::List(_))) => {
      Some(p.len())
    }
    Expr::FunctionCall { name, args } if name == "Band" => {
      match args.first()? {
        Expr::List(p) => Some(p.len()),
        _ => None,
      }
    }
    _ => None,
  }
}

fn parse_sparse_dims(expr: &Expr) -> Option<Vec<usize>> {
  if let Expr::List(items) = expr {
    let mut result = Vec::with_capacity(items.len());
    for item in items {
      let n = expr_to_i128(item)?;
      if n < 0 {
        return None;
      }
      result.push(n as usize);
    }
    Some(result)
  } else {
    let n = expr_to_i128(expr)?;
    if n < 0 {
      return None;
    }
    Some(vec![n as usize])
  }
}

/// When `SparseArray[rules]` is given with no explicit dimensions and some
/// rule is a bare `Band[start] -> value`, infer the matrix dimensions that
/// rule implies (and the dimensions any plain position rules imply) so the
/// bands can be expanded before falling back to the normal dims-inference
/// path. A band's extent is taken from `value`'s length when no explicit
/// end (`Band[start, end] -> value`) is given. Returns `None` when `data`
/// has no bare `Band` rule, so the caller keeps its previous behavior.
fn infer_dims_from_band_rules(data: &Expr) -> Option<Vec<usize>> {
  let items: Vec<Expr> = match data {
    Expr::List(items) => items.to_vec(),
    rule @ Expr::Rule { .. } => vec![rule.clone()],
    _ => return None,
  };
  if items.is_empty() || !items.iter().all(|it| matches!(it, Expr::Rule { .. }))
  {
    return None;
  }
  let is_band_rule = |it: &Expr| {
    matches!(it, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::FunctionCall { name, .. } if name == "Band"))
  };
  if !items.iter().any(is_band_rule) {
    return None;
  }

  let mut rank: Option<usize> = None;
  let mut max_pos: Vec<usize> = Vec::new();
  let bump_rank =
    |r: usize, rank: &mut Option<usize>, max_pos: &mut Vec<usize>| -> bool {
      match rank {
        None => {
          *rank = Some(r);
          *max_pos = vec![0usize; r];
          true
        }
        Some(rk) => *rk == r,
      }
    };

  for item in &items {
    let Expr::Rule {
      pattern,
      replacement,
    } = item
    else {
      return None;
    };
    if let Expr::FunctionCall {
      name,
      args: band_args,
    } = pattern.as_ref()
      && name == "Band"
      && !band_args.is_empty()
    {
      let start: Vec<i128> = match &band_args[0] {
        Expr::List(pos) => {
          pos.iter().map(expr_to_i128).collect::<Option<_>>()?
        }
        other => vec![expr_to_i128(other)?],
      };
      if start.is_empty() || start.iter().any(|&p| p < 1) {
        return None;
      }
      if !bump_rank(start.len(), &mut rank, &mut max_pos) {
        return None;
      }
      let end: Vec<i128> = if band_args.len() >= 2 {
        match &band_args[1] {
          Expr::List(pos) if pos.len() == start.len() => {
            pos.iter().map(expr_to_i128).collect::<Option<_>>()?
          }
          _ => return None,
        }
      } else if start.len() >= 2
        && let Some(blocks) = band_blocks(replacement, start.len())
      {
        let mut end = start.clone();
        for (shape, _) in &blocks {
          let extent = band_block_extent(shape, start.len());
          for (e, &n) in end.iter_mut().zip(&extent) {
            *e += n as i128;
          }
        }
        end.iter().map(|e| e - 1).collect()
      } else if let Expr::List(vals) = replacement.as_ref() {
        let len = vals.len() as i128;
        start.iter().map(|&s| s + len - 1).collect()
      } else {
        return None;
      };
      for (i, &e) in end.iter().enumerate() {
        if e < 1 {
          return None;
        }
        max_pos[i] = max_pos[i].max(e as usize);
      }
    } else {
      let pos_vec: Vec<i128> = match pattern.as_ref() {
        Expr::List(pos) => {
          pos.iter().map(expr_to_i128).collect::<Option<_>>()?
        }
        other => vec![expr_to_i128(other)?],
      };
      if pos_vec.is_empty() || pos_vec.iter().any(|&p| p < 1) {
        return None;
      }
      if !bump_rank(pos_vec.len(), &mut rank, &mut max_pos) {
        return None;
      }
      for (i, &p) in pos_vec.iter().enumerate() {
        max_pos[i] = max_pos[i].max(p as usize);
      }
    }
  }
  Some(max_pos)
}

/// The blocks of an array-valued band of rank `rank`, with their shapes: an
/// array of rank 2 … `rank` is one block, a list of them a block-diagonal
/// run. A block of lower rank spreads over the extra axes as
/// `band_block_extent` describes.
fn band_blocks(value: &Expr, rank: usize) -> Option<Vec<(Vec<usize>, &Expr)>> {
  let block = |e: &Expr| {
    dense_shape(e)
      .filter(|shape| (2..=rank).contains(&shape.len()) && !shape.contains(&0))
  };
  if let Some(shape) = block(value) {
    return Some(vec![(shape, value)]);
  }
  let Expr::List(items) = value else {
    return None;
  };
  if items.is_empty() {
    return None;
  }
  items
    .iter()
    .map(|item| block(item).map(|shape| (shape, item)))
    .collect()
}

/// The extent of a band block of `shape` over all `rank` axes. The block's
/// q levels span the first q axes, and every further axis advances with
/// all of them: `Band[{1, 1, 1}] -> {{1, 2}, {3, 4}}` puts 3 at {2, 1, 2}
/// and 4 at {2, 2, 3}.
fn band_block_extent(shape: &[usize], rank: usize) -> Vec<usize> {
  let diagonal = shape.iter().map(|n| n - 1).sum::<usize>() + 1;
  (0..rank)
    .map(|i| shape.get(i).copied().unwrap_or(diagonal))
    .collect()
}

/// The 0-based offset over all `rank` axes of the block element at the
/// 1-based `index` (see `band_block_extent`).
fn band_block_offset(index: &[i128], rank: usize) -> Vec<usize> {
  let offsets: Vec<usize> = index.iter().map(|&i| i as usize - 1).collect();
  let diagonal: usize = offsets.iter().sum();
  (0..rank)
    .map(|i| offsets.get(i).copied().unwrap_or(diagonal))
    .collect()
}

/// Expand `Band[{i, j}, ...] -> v` rules into concrete `{i+k, j+k} -> v`
/// rules using `dims`. Other rules pass through unchanged. Returns None
/// when nothing in `data` references Band (so the caller can use the
/// original expression).
fn expand_band_rules(data: &Expr, dims: &[usize]) -> Option<Expr> {
  let is_band_rule = |it: &Expr| {
    matches!(it, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::FunctionCall { name, .. } if name == "Band"))
  };
  // Accept either a list of rules or a single bare `Band[...] -> v` rule.
  let items: Vec<Expr> = match data {
    Expr::List(items) => items.to_vec(),
    rule if is_band_rule(rule) => vec![rule.clone()],
    _ => return None,
  };
  if !items.iter().any(is_band_rule) {
    return None;
  }
  let mut expanded: Vec<Expr> = Vec::with_capacity(items.len());
  for item in &items {
    if let Expr::Rule {
      pattern,
      replacement,
    } = item
      && let Expr::FunctionCall { name, args } = pattern.as_ref()
      && name == "Band"
      && !args.is_empty()
    {
      let start = list_of_positive_ints(&args[0], dims.len())?;
      let end: Vec<usize> = if args.len() >= 2 {
        list_of_positive_ints(&args[1], dims.len())?
          .into_iter()
          .enumerate()
          .map(|(i, v)| v.min(dims[i]))
          .collect()
      } else {
        dims.to_vec()
      };
      let steps_possible: usize = (0..start.len())
        .map(|i| {
          if start[i] > end[i] {
            0
          } else {
            end[i] - start[i] + 1
          }
        })
        .min()
        .unwrap_or(0);
      // An array value of the band's rank is one block with its corner at
      // the band's start — tiled over the whole start…end box when the band
      // names its end — and a list of them the blocks of a block-diagonal
      // band, each starting where the previous one ends, cycling until the
      // named end. A block reaching past the box is cut (`::bndtr`).
      if dims.len() >= 2
        && let Some(blocks) = band_blocks(replacement, dims.len())
      {
        let push = |expanded: &mut Vec<Expr>, pos: &[usize], v: Expr| {
          expanded.push(rule_expr(
            Expr::List(pos.iter().map(|&p| Expr::Integer(p as i128)).collect()),
            v,
          ));
        };
        let explicit_end = args.len() >= 2;
        let single = matches!(blocks.as_slice(), [(_, b)] if std::ptr::eq(*b, replacement.as_ref()));
        if single {
          let (shape, block) = &blocks[0];
          let shape = band_block_extent(shape, start.len());
          let span: Vec<usize> = start
            .iter()
            .zip(&end)
            .map(|(s, e)| (e + 1).saturating_sub(*s))
            .collect();
          if shape.iter().zip(&span).any(|(n, m)| n > m) {
            let ints = |v: &[usize]| {
              Expr::List(v.iter().map(|&x| Expr::Integer(x as i128)).collect())
            };
            let band = Expr::FunctionCall {
              name: "Band".to_string(),
              args: vec![ints(&start), ints(&end), ints(&vec![1; start.len()])]
                .into(),
            };
            crate::emit_message(&format!(
              "SparseArray::bndtr: There are insufficient positions available \
               in the {} to fit all the values {} so some of the values will \
               be omitted in the specification {}.",
              crate::syntax::format_expr(
                &band,
                crate::syntax::ExprForm::Output
              ),
              crate::syntax::format_expr(
                block,
                crate::syntax::ExprForm::Output
              ),
              crate::syntax::format_expr(
                &rule_expr(band.clone(), (*block).clone()),
                crate::syntax::ExprForm::Output
              ),
            ));
          }
          let mut entries = Vec::new();
          collect_dense_rules(block, &mut Vec::new(), &mut entries, "\u{0}");
          let reps: Vec<usize> = shape
            .iter()
            .zip(&span)
            .map(|(n, m)| if explicit_end { m.div_ceil(*n) } else { 1 })
            .collect();
          let total: usize = reps.iter().product();
          for r in 0..total {
            let mut rest = r;
            let tile: Vec<usize> = reps
              .iter()
              .rev()
              .map(|n| {
                let t = rest % n;
                rest /= n;
                t
              })
              .collect::<Vec<_>>()
              .into_iter()
              .rev()
              .collect();
            for (index, v) in &entries {
              let offset = band_block_offset(index, start.len());
              let pos: Vec<usize> = (0..start.len())
                .map(|k| start[k] + tile[k] * shape[k] + offset[k])
                .collect();
              if pos.iter().zip(&end).all(|(p, e)| p <= e) {
                push(&mut expanded, &pos, v.clone());
              }
            }
          }
          continue;
        }
        let mut corner = start.clone();
        'cycle: loop {
          for (shape, block) in &blocks {
            if corner.iter().zip(&end).any(|(c, e)| c > e) {
              break 'cycle;
            }
            let mut entries = Vec::new();
            collect_dense_rules(block, &mut Vec::new(), &mut entries, "\u{0}");
            for (index, v) in entries {
              let pos: Vec<usize> = corner
                .iter()
                .zip(band_block_offset(&index, corner.len()))
                .map(|(&c, o)| c + o)
                .collect();
              if pos.iter().zip(&end).all(|(p, e)| p <= e) {
                push(&mut expanded, &pos, v);
              }
            }
            let extent = band_block_extent(shape, corner.len());
            for (c, n) in corner.iter_mut().zip(&extent) {
              *c += n;
            }
          }
          if !explicit_end {
            break;
          }
        }
        continue;
      }
      // `v` supplies one value per band position (in order) when it's a
      // vector — only as far as it reaches; otherwise every position gets
      // `v` itself (`Band[{i,j}] -> c` broadcasts the scalar `c`).
      let per_position_value: Option<&crate::ExprList> =
        match replacement.as_ref() {
          Expr::List(vals)
            if !vals.iter().any(|v| matches!(v, Expr::List(_))) =>
          {
            Some(vals)
          }
          _ => None,
        };
      let steps_possible = match per_position_value {
        Some(vals) => steps_possible.min(vals.len()),
        None => steps_possible,
      };
      for k in 0..steps_possible {
        let pos: Vec<Expr> = start
          .iter()
          .map(|&s| Expr::Integer((s + k) as i128))
          .collect();
        let value = match per_position_value {
          Some(vals) => vals[k].clone(),
          None => replacement.as_ref().clone(),
        };
        expanded.push(rule_expr(Expr::List(pos.into()), value));
      }
      continue;
    }
    expanded.push(item.clone());
  }
  Some(Expr::List(expanded.into()))
}

/// Rewrite negative integer coordinates in explicit position rules (counted
/// from the end of each axis, so `{-1, 1}` in a 3x3 array is `{3, 1}`) into
/// positive ones. Pattern positions and out-of-range coordinates are left
/// unchanged. Returns None when nothing needed rewriting.
fn resolve_negative_positions(data: &Expr, dims: &[usize]) -> Option<Expr> {
  let fix_rule = |item: &Expr| -> Option<Expr> {
    let (pattern, replacement, delayed) = match item {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern, replacement, false),
      Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern, replacement, true),
      _ => return None,
    };
    let fix_coord = |c: &Expr, axis: usize| -> Option<Expr> {
      let v = expr_to_i128(c)?;
      let len = *dims.get(axis)? as i128;
      (v < 0 && -v <= len).then(|| Expr::Integer(len + 1 + v))
    };
    let new_pattern = match pattern.as_ref() {
      Expr::List(ps) if ps.len() == dims.len() => {
        let mut changed = false;
        let fixed: Vec<Expr> = ps
          .iter()
          .enumerate()
          .map(|(k, p)| match fix_coord(p, k) {
            Some(n) => {
              changed = true;
              n
            }
            None => p.clone(),
          })
          .collect();
        changed.then(|| Expr::List(fixed.into()))?
      }
      other if dims.len() == 1 => fix_coord(other, 0)?,
      _ => return None,
    };
    Some(if delayed {
      Expr::RuleDelayed {
        pattern: Box::new(new_pattern),
        replacement: replacement.clone(),
      }
    } else {
      Expr::Rule {
        pattern: Box::new(new_pattern),
        replacement: replacement.clone(),
      }
    })
  };
  match data {
    Expr::List(items) => {
      let mut any = false;
      let new_items: Vec<Expr> = items
        .iter()
        .map(|it| match fix_rule(it) {
          Some(n) => {
            any = true;
            n
          }
          None => it.clone(),
        })
        .collect();
      any.then(|| Expr::List(new_items.into()))
    }
    single => fix_rule(single),
  }
}

/// Expand a pattern rule such as `{i_} :> i` or `{i_, j_} :> i + j` (with an
/// optional `/; cond`) into explicit `{i, j, ...} -> value` rules over the
/// grid `dims`. The rule's left-hand side must be a `List` pattern (after
/// stripping any outer `Condition`) whose length equals the rank of `dims`.
/// Each position in the grid is tested with `MatchQ` and, when it matches,
/// the value is obtained with `Replace` (which binds the pattern variables to
/// the indices). Returns None when `data` is not such a pattern rule, leaving
/// other constructor forms untouched.
fn expand_pattern_rule(data: &Expr, dims: &[usize]) -> Option<Expr> {
  let lhs: &Expr = match data {
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => pattern,
    _ => return None,
  };
  // Strip an outer `Condition[inner, test]` to inspect the structural part.
  let structural: &Expr = match lhs {
    Expr::FunctionCall { name, args }
      if name == "Condition" && args.len() == 2 =>
    {
      &args[0]
    }
    other => other,
  };
  // Only the list-pattern form is handled (e.g. `{i_}`, `{i_, j_}`).
  let Expr::List(pats) = structural else {
    return None;
  };
  if pats.len() != dims.len() || dims.is_empty() {
    return None;
  }
  // It must actually be a pattern (contain a Blank), otherwise it is an
  // ordinary explicit position rule that the normal pipeline handles.
  if !pats.iter().any(expr_contains_pattern) {
    return None;
  }
  let total: usize = dims.iter().product();
  // Guard against pathological sizes.
  if total == 0 || total > 1_000_000 {
    return None;
  }
  let mut rules: Vec<Expr> = Vec::new();
  for flat in 0..total {
    let mut rem = flat;
    let mut pos = vec![0i128; dims.len()];
    for k in (0..dims.len()).rev() {
      pos[k] = (rem % dims[k]) as i128 + 1;
      rem /= dims[k];
    }
    let pos_list = Expr::List(pos.iter().map(|&p| Expr::Integer(p)).collect());
    let match_q = call("MatchQ", vec![pos_list.clone(), lhs.clone()]);
    let matched = matches!(
      crate::evaluator::evaluate_expr_to_expr(&match_q),
      Ok(Expr::Identifier(ref s)) if s == "True"
    );
    if !matched {
      continue;
    }
    let replaced = call("Replace", vec![pos_list.clone(), data.clone()]);
    let val = crate::evaluator::evaluate_expr_to_expr(&replaced).ok()?;
    rules.push(rule_expr(pos_list, val));
  }
  Some(Expr::List(rules.into()))
}

/// Expand a LIST of position rules that mixes explicit positions with
/// pattern rules (`{i_, i_} :> …`) and/or Alternatives-of-position rules
/// (`{1, 2} | {3, 4} | … -> v`) into a flat list of explicit rules.
/// wolframscript semantics: for every position, the FIRST rule (in list
/// order) whose LHS matches wins. Returns None when no rule needs pattern
/// matching (the ordinary pipeline handles plain rule lists).
fn expand_pattern_rule_list(data: &Expr, dims: &[usize]) -> Option<Expr> {
  let Expr::List(items) = data else {
    return None;
  };
  if dims.is_empty() || items.is_empty() {
    return None;
  }

  enum Matcher {
    Explicit(Vec<i128>, Expr),
    Alternatives(std::collections::HashSet<Vec<i128>>, Expr),
    /// The whole rule, applied per position via MatchQ + Replace.
    Pattern(Expr),
  }

  let explicit_pos = |e: &Expr| -> Option<Vec<i128>> {
    let pos: Vec<i128> = match e {
      Expr::List(ps) => ps.iter().map(expr_to_i128).collect::<Option<_>>()?,
      other => vec![expr_to_i128(other)?],
    };
    (pos.len() == dims.len()).then_some(pos)
  };

  let mut matchers: Vec<Matcher> = Vec::with_capacity(items.len());
  let mut needs_matching = false;
  for item in items {
    let (lhs, rhs, delayed) = match item {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref(), false),
      Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref(), true),
      _ => return None,
    };
    // A delayed RHS is evaluated once per expanded position; for the
    // position-independent matchers evaluate it here, once.
    let value = || -> Option<Expr> {
      if delayed {
        crate::evaluator::evaluate_expr_to_expr(rhs).ok()
      } else {
        Some(rhs.clone())
      }
    };
    if let Some(pos) = explicit_pos(lhs) {
      matchers.push(Matcher::Explicit(pos, value()?));
    } else if let Expr::FunctionCall { name, args } = lhs
      && name == "Alternatives"
      && let Some(set) = args
        .iter()
        .map(explicit_pos)
        .collect::<Option<std::collections::HashSet<_>>>()
    {
      needs_matching = true;
      matchers.push(Matcher::Alternatives(set, value()?));
    } else if expr_contains_pattern(lhs) {
      needs_matching = true;
      matchers.push(Matcher::Pattern(item.clone()));
    } else {
      return None;
    }
  }
  if !needs_matching {
    return None;
  }

  let total: usize = dims.iter().product();
  if total == 0 {
    return None;
  }
  // When every pattern rule can name the few positions it may match (a
  // diagonal or a band), only those — plus the explicit ones — are visited;
  // otherwise every position of the array is tried.
  let mut candidates: Option<Vec<Vec<i128>>> = Some(Vec::new());
  for m in &matchers {
    let found = match m {
      Matcher::Explicit(p, _) => Some(vec![p.clone()]),
      Matcher::Alternatives(set, _) => Some(set.iter().cloned().collect()),
      Matcher::Pattern(rule) => match rule {
        Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
          pattern_rule_candidates(pattern, dims)
        }
        _ => None,
      },
    };
    match (candidates.as_mut(), found) {
      (Some(all), Some(found)) => all.extend(found),
      _ => candidates = None,
    }
  }
  let positions: Box<dyn Iterator<Item = Vec<i128>>> = match candidates {
    Some(mut list) => {
      list.sort();
      list.dedup();
      Box::new(list.into_iter())
    }
    None if total > 1_000_000 => return None,
    None => {
      let dims = dims.to_vec();
      Box::new((0..total).map(move |flat| {
        let mut rem = flat;
        let mut pos = vec![0i128; dims.len()];
        for k in (0..dims.len()).rev() {
          pos[k] = (rem % dims[k]) as i128 + 1;
          rem /= dims[k];
        }
        pos
      }))
    }
  };
  let mut rules: Vec<Expr> = Vec::new();
  for pos in positions {
    for m in &matchers {
      match m {
        Matcher::Explicit(p, v) if *p == pos => {
          rules.push(explicit_rule(&pos, v.clone()));
        }
        Matcher::Alternatives(set, v) if set.contains(&pos) => {
          rules.push(explicit_rule(&pos, v.clone()));
        }
        Matcher::Pattern(rule) => {
          let pos_list =
            Expr::List(pos.iter().map(|&p| Expr::Integer(p)).collect());
          let match_q = call(
            "MatchQ",
            vec![
              pos_list.clone(),
              match rule {
                Expr::Rule { pattern, .. }
                | Expr::RuleDelayed { pattern, .. } => (**pattern).clone(),
                _ => unreachable!(),
              },
            ],
          );
          if !matches!(
            crate::evaluator::evaluate_expr_to_expr(&match_q),
            Ok(Expr::Identifier(ref s)) if s == "True"
          ) {
            continue;
          }
          let replaced = call("Replace", vec![pos_list, rule.clone()]);
          let val = crate::evaluator::evaluate_expr_to_expr(&replaced).ok()?;
          rules.push(explicit_rule(&pos, val));
        }
        _ => continue,
      }
      break;
    }
  }
  Some(Expr::List(rules.into()))
}

/// The positions of a rank-2 array that a pattern rule's left side can
/// match, when they are few and easy to name: `{i_, i_}` is the diagonal, and
/// `{i_, j_} /; cond` is whatever `Solve[cond, j]` gives along each row
/// (`Abs[i - j] == 1` is the band either side of it). Every candidate is
/// still checked by the matcher; `None` means "try every position".
fn pattern_rule_candidates(
  lhs: &Expr,
  dims: &[usize],
) -> Option<Vec<Vec<i128>>> {
  let [rows, cols] = *dims else {
    return None;
  };
  let (inner, cond) = match lhs {
    Expr::FunctionCall { name, args }
      if name == "Condition" && args.len() == 2 =>
    {
      (&args[0], Some(&args[1]))
    }
    other => (other, None),
  };
  let Expr::List(slots) = inner else {
    return None;
  };
  let names: Vec<&str> = slots
    .iter()
    .map(|s| match s {
      Expr::Pattern {
        name,
        head: None,
        blank_type: 1,
      } if !name.is_empty() => Some(name.as_str()),
      _ => None,
    })
    .collect::<Option<_>>()?;
  let [a, b] = names.as_slice() else {
    return None;
  };
  if a == b {
    return Some((1..=rows.min(cols) as i128).map(|i| vec![i, i]).collect());
  }
  let cond = cond?;
  // Solving only finds every position when the condition is arithmetic in
  // the two indices. One that reads data (`w[[i]] == S`) or calls a
  // function would be evaluated with symbolic indices and come back with
  // no solutions, silently dropping positions that do match.
  if !is_index_arithmetic(cond, &[a, b]) {
    return None;
  }
  let solved = {
    crate::push_quiet();
    let r = crate::evaluator::evaluate_expr_to_expr(&call(
      "Solve",
      vec![cond.clone(), Expr::Identifier((*b).to_string())],
    ));
    crate::pop_quiet();
    r.ok()?
  };
  let Expr::List(solutions) = &solved else {
    return None;
  };
  let mut columns_of = Vec::new();
  for sol in solutions {
    let Expr::List(rules) = sol else {
      return None;
    };
    let [Expr::Rule { replacement, .. }] = rules.as_slice() else {
      return None;
    };
    columns_of.push(replacement.as_ref().clone());
  }
  let mut out = Vec::new();
  for i in 1..=rows as i128 {
    for f in &columns_of {
      let at = crate::syntax::substitute_variable(f, a, &Expr::Integer(i));
      match crate::evaluator::evaluate_expr_to_expr(&at) {
        Ok(Expr::Integer(j)) => {
          if (1..=cols as i128).contains(&j) {
            out.push(vec![i, j]);
          }
        }
        // A column that is not an integer is no array position.
        Ok(Expr::Real(_) | Expr::FunctionCall { .. }) => {}
        _ => return None,
      }
    }
  }
  Some(out)
}

/// Whether `cond` is built only from the index variables `vars`, numbers and
/// arithmetic, comparison and logical operators — the conditions `Solve`
/// can turn into the complete set of matching positions.
fn is_index_arithmetic(cond: &Expr, vars: &[&str]) -> bool {
  let all = |items: &[Expr]| items.iter().all(|e| is_index_arithmetic(e, vars));
  match cond {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => true,
    Expr::Identifier(name) => vars.contains(&name.as_str()),
    Expr::BinaryOp { left, right, .. } => {
      is_index_arithmetic(left, vars) && is_index_arithmetic(right, vars)
    }
    Expr::UnaryOp { operand, .. } => is_index_arithmetic(operand, vars),
    Expr::Comparison { operands, .. } => all(operands),
    Expr::FunctionCall { name, args } => {
      matches!(
        name.as_str(),
        "Plus"
          | "Times"
          | "Power"
          | "Subtract"
          | "Minus"
          | "Divide"
          | "Rational"
          | "Abs"
          | "Equal"
          | "Unequal"
          | "Less"
          | "LessEqual"
          | "Greater"
          | "GreaterEqual"
          | "And"
          | "Or"
          | "Not"
      ) && all(args)
    }
    _ => false,
  }
}

/// The dimensions a rule list's explicit integer positions span, when the
/// list also holds pattern rules (which say nothing about the size).
fn infer_dims_from_explicit_positions(data: &Expr) -> Option<Vec<usize>> {
  let Expr::List(items) = data else {
    return None;
  };
  let mut dims: Option<Vec<usize>> = None;
  let mut has_pattern = false;
  for item in items {
    let lhs = match item {
      Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
        pattern.as_ref()
      }
      _ => return None,
    };
    let Expr::List(pos) = lhs else {
      if expr_contains_pattern(lhs) {
        has_pattern = true;
        continue;
      }
      return None;
    };
    let explicit: Option<Vec<usize>> = pos
      .iter()
      .map(|p| expr_to_i128(p).filter(|n| *n > 0).map(|n| n as usize))
      .collect();
    match explicit {
      Some(p) => match dims.as_mut() {
        Some(d) if d.len() == p.len() => {
          for (dk, pk) in d.iter_mut().zip(p) {
            *dk = (*dk).max(pk);
          }
        }
        Some(_) => return None,
        None => dims = Some(p),
      },
      None if expr_contains_pattern(lhs) => has_pattern = true,
      None => return None,
    }
  }
  if has_pattern { dims } else { None }
}

/// Build `{p1, p2, …} -> value` for an explicit integer position.
fn explicit_rule(pos: &[i128], value: Expr) -> Expr {
  rule_expr(
    Expr::List(pos.iter().map(|&p| Expr::Integer(p)).collect()),
    value,
  )
}

/// Whether `expr` contains a pattern (Blank) node anywhere within it.
fn expr_contains_pattern(expr: &Expr) -> bool {
  // The one pattern detector, which also knows the long forms a reloaded
  // definition carries (`Pattern[i, Blank[]]` for `i_`)
  crate::evaluator::pattern_matching::contains_pattern(expr)
}

fn list_of_positive_ints(e: &Expr, rank: usize) -> Option<Vec<usize>> {
  let Expr::List(items) = e else { return None };
  if items.len() != rank {
    return None;
  }
  let mut out = Vec::with_capacity(rank);
  for it in items {
    let n = expr_to_i128(it)?;
    if n < 1 {
      return None;
    }
    out.push(n as usize);
  }
  Some(out)
}

/// Parse a list of position-value rules. Positions may be scalar integers
/// (1D) or lists of integers (k-D); all rules must share the same rank.
/// Returns the rules as (position, value) pairs and the max-per-axis dims.
fn parse_sparse_rules(
  items: &[Expr],
) -> Option<(Vec<(Vec<i128>, Expr)>, Vec<usize>)> {
  let mut rules: Vec<(Vec<i128>, Expr)> = Vec::with_capacity(items.len());
  let mut max_pos: Vec<usize> = Vec::new();
  let mut rank: Option<usize> = None;

  for item in items {
    let (pattern, replacement) = match item {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => return None,
    };
    let pos_vec: Vec<i128> = match pattern {
      Expr::List(pos) => {
        let mut v = Vec::with_capacity(pos.len());
        for p in pos {
          v.push(expr_to_i128(p)?);
        }
        v
      }
      _ => vec![expr_to_i128(pattern)?],
    };
    // Reject empty position, non-positive indices and rank mismatches.
    if pos_vec.is_empty() || pos_vec.iter().any(|&p| p < 1) {
      return None;
    }
    match rank {
      None => {
        rank = Some(pos_vec.len());
        max_pos = vec![0; pos_vec.len()];
      }
      Some(r) if r != pos_vec.len() => return None,
      _ => {}
    }
    for (i, &p) in pos_vec.iter().enumerate() {
      max_pos[i] = max_pos[i].max(p as usize);
    }
    rules.push((pos_vec, replacement.clone()));
  }

  Some((rules, max_pos))
}

/// Return the shape of a dense, rectangular nested list. For a scalar leaf,
/// returns an empty vector.
fn dense_shape(expr: &Expr) -> Option<Vec<usize>> {
  match expr {
    Expr::List(items) => {
      if items.is_empty() {
        return Some(vec![0]);
      }
      let first_shape = dense_shape(&items[0])?;
      for item in &items[1..] {
        if dense_shape(item)? != first_shape {
          return None;
        }
      }
      let mut dims = vec![items.len()];
      dims.extend(first_shape);
      Some(dims)
    }
    _ => Some(Vec::new()),
  }
}

/// Walk a dense nested list and record every leaf that is not equal to the
/// default value as a (position, value) rule.
fn collect_dense_rules(
  expr: &Expr,
  indices: &mut Vec<i128>,
  rules: &mut Vec<(Vec<i128>, Expr)>,
  default_str: &str,
) {
  match expr {
    Expr::List(items) => {
      for (i, item) in items.iter().enumerate() {
        indices.push((i + 1) as i128);
        collect_dense_rules(item, indices, rules, default_str);
        indices.pop();
      }
    }
    _ => {
      if expr_to_string(expr) != default_str {
        rules.push((indices.clone(), expr.clone()));
      }
    }
  }
}

/// Parse the first argument of a SparseArray call, producing (rules, inferred
/// dimensions). The first argument may be a list of rules, a dense nested
/// list, or a single Rule expression.
fn parse_sparse_data(
  data: &Expr,
  default: &Expr,
) -> Option<(Vec<(Vec<i128>, Expr)>, Vec<usize>)> {
  // A bare single rule is treated as a one-element rule list.
  if matches!(data, Expr::Rule { .. }) {
    return parse_sparse_rules(std::slice::from_ref(data));
  }

  let Expr::List(items) = data else { return None };

  if items.is_empty() {
    return Some((Vec::new(), vec![0]));
  }

  let any_rule = items.iter().any(|it| matches!(it, Expr::Rule { .. }));
  let all_rules = items.iter().all(|it| matches!(it, Expr::Rule { .. }));

  if all_rules {
    return parse_sparse_rules(items);
  }
  if any_rule {
    // Mixed rule / non-rule entries are ambiguous.
    return None;
  }

  // A list of SparseArrays (`SparseArray[{s1, s2}]`) is the array of their
  // dense forms.
  fn densify_items(e: &Expr) -> Expr {
    match e {
      Expr::List(items) => {
        Expr::List(items.iter().map(densify_items).collect())
      }
      other => densify_sparse_array(other).unwrap_or_else(|| other.clone()),
    }
  }
  let densified = densify_items(data);
  let data = &densified;
  // Dense nested list: must be rectangular.
  let shape = dense_shape(data)?;
  let default_str = expr_to_string(default);
  let mut rules = Vec::new();
  let mut indices = Vec::new();
  collect_dense_rules(data, &mut indices, &mut rules, &default_str);
  Some((rules, shape))
}

/// Rewrite the single-argument rule forms of `SparseArray` into a plain list
/// of `position -> value` rules plus an optional background value taken from
/// a `{_, …, _} -> value` rule. Returns `None` when `data` uses neither form.
fn sparse_rules_with_background(data: &Expr) -> Option<(Expr, Option<Expr>)> {
  let as_rule = |e: &Expr| -> Option<(Expr, Expr)> {
    match e {
      Expr::Rule {
        pattern,
        replacement,
      } => Some((pattern.as_ref().clone(), replacement.as_ref().clone())),
      _ => None,
    }
  };
  let mut changed = false;
  let rules: Vec<Expr> = match data {
    Expr::Rule { .. } => {
      let (lhs, rhs) = as_rule(data)?;
      match (&lhs, &rhs) {
        (Expr::List(keys), Expr::List(vals))
          if keys.len() == vals.len()
            && !keys.is_empty()
            && (keys.iter().all(|k| matches!(k, Expr::List(_)))
              // `{i1, i2, …} -> {v1, v2, …}` threads over vector positions.
              || keys.iter().all(|k| matches!(k, Expr::Integer(_)))) =>
        {
          changed = true;
          keys
            .iter()
            .zip(vals.iter())
            .map(|(k, v)| Expr::Rule {
              pattern: Box::new(k.clone()),
              replacement: Box::new(v.clone()),
            })
            .collect()
        }
        // `{pos1, pos2, …} -> v` gives every position the same value.
        (Expr::List(keys), value)
          if !matches!(value, Expr::List(_))
            && !keys.is_empty()
            && keys.iter().all(|k| matches!(k, Expr::List(_))) =>
        {
          changed = true;
          keys
            .iter()
            .map(|k| Expr::Rule {
              pattern: Box::new(k.clone()),
              replacement: Box::new(value.clone()),
            })
            .collect()
        }
        _ => return None,
      }
    }
    Expr::List(items) if items.iter().all(|i| as_rule(i).is_some()) => {
      items.to_vec()
    }
    _ => return None,
  };
  let mut background = None;
  let mut kept = Vec::with_capacity(rules.len());
  for rule in rules {
    let (lhs, rhs) = as_rule(&rule)?;
    let is_blank_position = matches!(&lhs, Expr::List(pos)
      if !pos.is_empty() && pos.iter().all(|p| expr_to_string(p) == "_"));
    if is_blank_position {
      // It matches every position, so — the first matching rule winning —
      // no rule after it can take effect, nor count towards the
      // dimensions: `SparseArray[{{1, 1} -> 1, {_, _} -> 5, {2, 2} -> 3}]`
      // is the 1×1 array {{1}}.
      changed = true;
      background = Some(rhs);
      break;
    }
    kept.push(rule);
  }
  changed.then(|| (Expr::List(kept.into()), background))
}

/// Normalize SparseArray arguments to the canonical form
/// `SparseArray[Automatic, dims, default, rules]`. Accepts:
///   SparseArray[rules]                — infer dims from positions
///   SparseArray[list]                 — dense list, default 0
///   SparseArray[data, dims]           — explicit dims, default 0
///   SparseArray[data, dims, default]  — explicit default
/// Already-canonical calls are returned unchanged. If normalization is not
/// possible the original FunctionCall is returned unevaluated.
pub fn sparse_array_normalize_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Already in canonical form: SparseArray[Automatic, dims, default, rules].
  if args.len() == 4
    && matches!(&args[0], Expr::Identifier(s) if s == "Automatic")
    && matches!(&args[1], Expr::List(_))
    && matches!(&args[3], Expr::List(_))
  {
    return Ok(unevaluated("SparseArray", args));
  }

  if args.is_empty() || args.len() > 3 {
    return Ok(unevaluated("SparseArray", args));
  }

  // No rules with explicit dimensions: the empty array is built directly,
  // never as a dense list (`SparseArray[{}, {8192, 8192}]`).
  if (2..=3).contains(&args.len())
    && matches!(&args[0], Expr::List(rules) if rules.is_empty())
    && let Some(dims) = parse_sparse_dims(&args[1])
    && !dims.is_empty()
    && dims.iter().all(|&d| d > 0)
  {
    let default = args.get(2).cloned().unwrap_or(Expr::Integer(0));
    return Ok(
      crate::evaluator::dispatch::list_operations::build_sparse_array_csr(
        &dims,
        &default,
        &[],
      ),
    );
  }

  // `{} -> v` names no positions: no rules at all.
  if args.len() >= 2
    && matches!(&args[0], Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::List(keys) if keys.is_empty()))
  {
    let mut rewritten = args.to_vec();
    rewritten[0] = Expr::List(Vec::new().into());
    return sparse_array_normalize_ast(&rewritten);
  }

  // With explicit dimensions, a `{pos1, pos2, …} -> values` rule expands
  // into its rule list the same way.
  if args.len() >= 2
    && matches!(&args[0], Expr::Rule { .. })
    && let Some((rules, None)) = sparse_rules_with_background(&args[0])
  {
    let mut rewritten = args.to_vec();
    rewritten[0] = rules;
    return sparse_array_normalize_ast(&rewritten);
  }

  // A single dimension `n` for rank-r positions means `{n, n, …}`.
  if args.len() >= 2
    && matches!(&args[1], Expr::Integer(_))
    && let Some(rank) = sparse_rule_rank(&args[0])
    && rank > 1
  {
    let mut rewritten = args.to_vec();
    rewritten[1] = Expr::List(vec![args[1].clone(); rank].into());
    return sparse_array_normalize_ast(&rewritten);
  }

  // `SparseArray[{pos1, pos2, …} -> {val1, val2, …}]` pairs positions with
  // values; a `{_, …, _} -> v` rule sets the background of a call without an
  // explicit default. Both reduce to the plain rule-list form.
  if args.len() == 1
    && let Some(normalized) = sparse_rules_with_background(&args[0])
  {
    let mut rewritten = args.to_vec();
    rewritten.truncate(1);
    let (rules, background) = normalized;
    rewritten[0] = rules;
    if let Some(bg) = background {
      rewritten.push(Expr::Identifier("Automatic".to_string()));
      rewritten.push(bg);
    }
    if rewritten.len() == 3 {
      // dims are inferred from the remaining rules
      if matches!(&rewritten[0], Expr::List(kept) if kept.is_empty()) {
        crate::emit_message(
          "SparseArray::exdims: The dimensions cannot be determined from \
           the positions {}.",
        );
        return Ok(unevaluated("SparseArray", args));
      }
      let Some((_, inferred)) = parse_sparse_data(&rewritten[0], &rewritten[2])
      else {
        return Ok(unevaluated("SparseArray", args));
      };
      rewritten[1] = Expr::List(
        inferred.iter().map(|&n| Expr::Integer(n as i128)).collect(),
      );
    }
    return sparse_array_normalize_ast(&rewritten);
  }

  // Band[{i, j}] / Band[{i, j}, {iMax, jMax}] rules need to know the
  // dimensions to expand into concrete positions. When dims are
  // explicit, pre-expand and substitute back into args[0].
  let explicit_dims: Option<Vec<usize>> =
    args.get(1).and_then(parse_sparse_dims);
  // With no explicit dims, a bare `Band[start] -> list` rule still implies
  // dimensions (from its start position and the value list's length) —
  // infer them so the band can be expanded the same way an explicit-dims
  // call would.
  let band_inferred_dims: Option<Vec<usize>> = if explicit_dims.is_none() {
    infer_dims_from_band_rules(&args[0])
  } else {
    None
  };
  // Pattern rules (`{i_, i_} -> v`) mixed with explicit positions take the
  // dimensions from those positions: `SparseArray[{{1, 1} -> 1, {i_, i_}
  // -> 5, {3, 3} -> 3}]` is 3×3.
  let pattern_inferred_dims: Option<Vec<usize>> =
    if explicit_dims.is_none() && band_inferred_dims.is_none() {
      infer_dims_from_explicit_positions(&args[0])
    } else {
      None
    };
  let resolved_arg0: Option<Expr> = explicit_dims
    .as_ref()
    .and_then(|dims| resolve_negative_positions(&args[0], dims));
  let arg0: &Expr = resolved_arg0.as_ref().unwrap_or(&args[0]);
  let expanded_data: Expr;
  let data: &Expr = if let Some(ref dims) = explicit_dims {
    if let Some(expanded) = expand_pattern_rule(arg0, dims) {
      expanded_data = expanded;
      &expanded_data
    } else {
      // Band and pattern rules can be mixed in one rule list
      // (`{{i_, i_} -> i, Band[{1, 2}] -> r}`), so expand both in turn.
      let banded = expand_band_rules(arg0, dims);
      let after_band = banded.as_ref().unwrap_or(arg0);
      match expand_pattern_rule_list(after_band, dims) {
        Some(expanded) => {
          expanded_data = expanded;
          &expanded_data
        }
        None => match banded {
          Some(expanded) => {
            expanded_data = expanded;
            &expanded_data
          }
          None => arg0,
        },
      }
    }
  } else if let Some(ref dims) = band_inferred_dims {
    if let Some(expanded) = expand_band_rules(&args[0], dims) {
      expanded_data = expanded;
      &expanded_data
    } else {
      &args[0]
    }
  } else if let Some(ref dims) = pattern_inferred_dims
    && let Some(expanded) = expand_pattern_rule_list(&args[0], dims)
  {
    expanded_data = expanded;
    &expanded_data
  } else {
    &args[0]
  };
  let default = if args.len() >= 3 {
    args[2].clone()
  } else {
    Expr::Integer(0)
  };

  let Some((parsed_rules, inferred_dims)) = parse_sparse_data(data, &default)
  else {
    return Ok(unevaluated("SparseArray", args));
  };

  // Determine final dimensions. When explicit dims are given they must have
  // the same rank as the parsed positions (unless there are no rules).
  let dims: Vec<usize> = if args.len() >= 2 {
    match parse_sparse_dims(&args[1]) {
      Some(d) => {
        if !parsed_rules.is_empty() && d.len() != parsed_rules[0].0.len() {
          return Ok(unevaluated("SparseArray", args));
        }
        d
      }
      None => {
        return Ok(unevaluated("SparseArray", args));
      }
    }
  } else {
    if inferred_dims.is_empty() {
      return Ok(unevaluated("SparseArray", args));
    }
    inferred_dims
  };

  // Drop rules that are out of bounds or equal to the default value. When
  // the same position appears more than once, the *first* rule wins — this
  // matches wolframscript:
  //   Normal[SparseArray[{1 -> 5, 1 -> 9}, 3]] == {5, 0, 0}
  // BTreeMap gives us deterministic lexicographic ordering.
  let default_str = expr_to_string(&default);
  let mut dedup: std::collections::BTreeMap<Vec<i128>, Expr> =
    std::collections::BTreeMap::new();
  for (pos, val) in parsed_rules {
    if pos.len() != dims.len() {
      continue;
    }
    if !pos
      .iter()
      .enumerate()
      .all(|(i, &p)| p >= 1 && (p as usize) <= dims[i])
    {
      continue;
    }
    if expr_to_string(&val) == default_str {
      continue;
    }
    dedup.entry(pos).or_insert(val);
  }

  // Build the CSR-style fourth argument that wolframscript exposes for
  // SparseArray:
  //   {1, {row_ptr, inner_positions}, values}
  // where `row_ptr` groups entries by their first coordinate and
  // `inner_positions` lists the remaining coordinates for each entry.
  let sorted: Vec<(Vec<i128>, Expr)> = dedup.into_iter().collect();
  let dims_expr =
    Expr::List(dims.iter().map(|&n| Expr::Integer(n as i128)).collect());
  let rank = dims.len();

  // row_ptr length: for rank 1 Wolfram uses {0, nnz}; for rank >= 2 it
  // groups by the first axis, giving length dims[0] + 1.
  let row_ptr: Vec<i128> = if rank <= 1 {
    vec![0, sorted.len() as i128]
  } else {
    let mut ptr = vec![0i128; dims[0] + 1];
    for (pos, _) in &sorted {
      let row = pos[0] as usize; // 1-based row index
      for slot in &mut ptr[row..] {
        *slot += 1;
      }
    }
    ptr
  };

  let inner_positions: Vec<Expr> = sorted
    .iter()
    .map(|(pos, _)| {
      let tail: Vec<Expr> = if rank <= 1 {
        // 1D arrays still show a single-element inner position.
        pos.iter().copied().map(Expr::Integer).collect()
      } else {
        pos[1..].iter().copied().map(Expr::Integer).collect()
      };
      Expr::List(tail.into())
    })
    .collect();
  let values_list: Vec<Expr> = sorted.into_iter().map(|(_, v)| v).collect();

  let row_ptr_expr =
    Expr::List(row_ptr.into_iter().map(Expr::Integer).collect());
  let structure = Expr::List(
    vec![
      Expr::Integer(1),
      Expr::List(vec![row_ptr_expr, Expr::List(inner_positions.into())].into()),
      Expr::List(values_list.into()),
    ]
    .into(),
  );

  Ok(call(
    "SparseArray",
    vec![id_expr("Automatic"), dims_expr, default, structure],
  ))
}

/// Extract (position, value) pairs from the fourth argument of a canonical
/// SparseArray. Accepts either the legacy list-of-rules form or the
/// CSR-style `{1, {row_ptr, inner_positions}, values}` form used by
/// wolframscript.
pub fn sparse_array_extract_rules(
  dims: &[usize],
  arg: &Expr,
) -> Vec<(Vec<i128>, Expr)> {
  let Expr::List(items) = arg else {
    return Vec::new();
  };

  // Legacy list-of-rules form
  if items.iter().all(|i| matches!(i, Expr::Rule { .. })) {
    return items
      .iter()
      .filter_map(|r| match r {
        Expr::Rule {
          pattern,
          replacement,
        } => {
          let pos = match pattern.as_ref() {
            Expr::List(ps) => {
              let mut v = Vec::with_capacity(ps.len());
              for p in ps {
                v.push(expr_to_i128(p)?);
              }
              v
            }
            other => vec![expr_to_i128(other)?],
          };
          Some((pos, replacement.as_ref().clone()))
        }
        _ => None,
      })
      .collect();
  }

  // CSR form: {1, {row_ptr, inner_positions}, values}. The values slot may
  // also be the marker `Pattern`, which a "PatternArray" query leaves there:
  // every stored entry is then a blank.
  let pattern_values: Option<Vec<Expr>> = match (&items.get(1), &items.get(2)) {
    (Some(Expr::List(structure)), Some(Expr::Identifier(marker)))
      if marker == "Pattern" && structure.len() == 2 =>
    {
      match &structure[1] {
        Expr::List(inner) => Some(vec![call0("Blank"); inner.len()]),
        _ => None,
      }
    }
    _ => None,
  };
  let pattern_values_expr = pattern_values.map(|v| Expr::List(v.into()));
  let items: Vec<Expr> = match &pattern_values_expr {
    Some(values) => vec![items[0].clone(), items[1].clone(), values.clone()],
    None => items.to_vec(),
  };
  let items = items.as_slice();
  if items.len() == 3
    && matches!(&items[0], Expr::Integer(1))
    && let (Expr::List(structure), Expr::List(values)) = (&items[1], &items[2])
    && structure.len() == 2
    && let (Expr::List(row_ptr), Expr::List(inner_list)) =
      (&structure[0], &structure[1])
  {
    let rank = dims.len();
    let row_ptr_vals: Option<Vec<i128>> =
      row_ptr.iter().map(expr_to_i128).collect();
    let Some(row_ptr_vals) = row_ptr_vals else {
      return Vec::new();
    };
    let mut out: Vec<(Vec<i128>, Expr)> = Vec::with_capacity(values.len());
    // Determine row for each entry from row_ptr cumulative counts.
    let num_rows = if rank <= 1 { 1 } else { dims[0] };
    let mut entry_idx: usize = 0;
    for row in 0..num_rows {
      let count = (row_ptr_vals.get(row + 1).copied().unwrap_or(0)
        - row_ptr_vals.get(row).copied().unwrap_or(0))
        as usize;
      for _ in 0..count {
        if entry_idx >= values.len() || entry_idx >= inner_list.len() {
          break;
        }
        let Expr::List(tail_exprs) = &inner_list[entry_idx] else {
          entry_idx += 1;
          continue;
        };
        let tail: Option<Vec<i128>> =
          tail_exprs.iter().map(expr_to_i128).collect();
        let Some(tail) = tail else {
          entry_idx += 1;
          continue;
        };
        let mut pos = Vec::with_capacity(rank);
        if rank >= 2 {
          pos.push((row + 1) as i128);
          pos.extend(tail);
        } else {
          pos.extend(tail);
        }
        out.push((pos, values[entry_idx].clone()));
        entry_idx += 1;
      }
    }
    return out;
  }

  Vec::new()
}

/// Build a dense k-D nested list from a canonical rule list.
fn build_dense_from_rules(
  dims: &[usize],
  default: &Expr,
  rules: &[(Vec<i128>, Expr)],
) -> Expr {
  if dims.is_empty() {
    return default.clone();
  }
  if dims.len() == 1 {
    let n = dims[0];
    let mut arr = vec![default.clone(); n];
    for (pos, val) in rules {
      if pos.len() == 1 && pos[0] >= 1 && (pos[0] as usize) <= n {
        arr[(pos[0] - 1) as usize] = val.clone();
      }
    }
    return Expr::List(arr.into());
  }
  let d0 = dims[0];
  let rest_dims = &dims[1..];
  let mut result = Vec::with_capacity(d0);
  for i in 1..=d0 {
    let sub_rules: Vec<(Vec<i128>, Expr)> = rules
      .iter()
      .filter_map(|(pos, val)| {
        if !pos.is_empty() && pos[0] == i as i128 {
          Some((pos[1..].to_vec(), val.clone()))
        } else {
          None
        }
      })
      .collect();
    result.push(build_dense_from_rules(rest_dims, default, &sub_rules));
  }
  Expr::List(result.into())
}

/// If `expr` is a `SparseArray[...]`, return its dense nested-list form so
/// callers (statistics functions, etc.) can operate on the expanded array.
/// Returns `None` when `expr` is not a SparseArray or when densification
/// fails to reduce it to a non-SparseArray (so the caller leaves it alone).
pub fn densify_sparse_array(expr: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args: sa } = expr else {
    return None;
  };
  if name != "SparseArray" {
    return None;
  }
  let dense = sparse_array_ast(sa).ok()?;
  if matches!(&dense, Expr::FunctionCall { name, .. } if name == "SparseArray")
  {
    return None;
  }
  Some(dense)
}

/// The background (default) element of a canonical SparseArray.
pub(crate) fn sparse_array_background(expr: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "SparseArray" || args.len() != 4 {
    return None;
  }
  matches!(&args[0], Expr::Identifier(s) if s == "Automatic")
    .then(|| args[2].clone())
}

/// Shape of a rectangular nested list, as a `List` of integers.
fn nested_list_dims(expr: &Expr) -> Option<Expr> {
  let mut dims = Vec::new();
  let mut cur = expr;
  while let Expr::List(items) = cur {
    dims.push(Expr::Integer(items.len() as i128));
    match items.first() {
      Some(first) => cur = first,
      None => break,
    }
  }
  (!dims.is_empty()).then(|| Expr::List(dims.into()))
}

/// Row-major index paths (1-based) of the entries of a dense nested list that
/// differ from `background` — i.e. the positions the SparseArray stores.
fn sparse_stored_paths(dense: &Expr, background: &Expr) -> Vec<Vec<usize>> {
  fn walk(
    e: &Expr,
    bg: &str,
    path: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
  ) {
    if let Expr::List(items) = e {
      for (i, item) in items.iter().enumerate() {
        path.push(i + 1);
        walk(item, bg, path, out);
        path.pop();
      }
      return;
    }
    if expr_to_string(e) != bg {
      out.push(path.clone());
    }
  }
  let mut out = Vec::new();
  walk(
    dense,
    &expr_to_string(background),
    &mut Vec::new(),
    &mut out,
  );
  out
}

/// Element of a dense nested list at a 1-based index path.
fn nested_list_at(dense: &Expr, path: &[usize]) -> Option<Expr> {
  let mut cur = dense;
  for &i in path {
    let Expr::List(items) = cur else { return None };
    cur = items.get(i - 1)?;
  }
  Some(cur.clone())
}

/// Elementwise arithmetic (`Plus`, `Times`, `Power`, …) over operands of which
/// at least one is a SparseArray. Wolfram keeps the result sparse as long as
/// the other operands are scalars or SparseArrays — the background is the head
/// applied to the operands' backgrounds, so `sa + 1` shifts the background to
/// 1 rather than densifying. A plain dense list operand makes the result
/// dense, so those are left to the ordinary Listable path.
///
/// Returns `None` when no operand is a SparseArray, when one cannot be
/// densified, or when a dense list is involved.
pub fn try_sparse_array_arithmetic(head: &str, args: &[Expr]) -> Option<Expr> {
  let is_sparse = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "SparseArray");
  if !args.iter().any(is_sparse)
    || args.iter().any(|a| matches!(a, Expr::List(_)))
  {
    return None;
  }

  // Each operand contributes its dense form (SparseArray) or itself (scalar);
  // the background column uses the SparseArray backgrounds instead.
  let mut dense_args = Vec::with_capacity(args.len());
  let mut background_args = Vec::with_capacity(args.len());
  for a in args {
    if is_sparse(a) {
      dense_args.push(Some(densify_sparse_array(a)?));
      background_args.push(sparse_array_background(a)?);
    } else {
      dense_args.push(None);
      background_args.push(a.clone());
    }
  }

  let apply = |operands: Vec<Expr>| -> Option<Expr> {
    crate::evaluator::evaluate_expr_to_expr(&call(head, operands)).ok()
  };

  // Like Wolfram, evaluate the head once on the backgrounds and once per
  // stored position — never on the whole dense array. `1/SparseArray[…]`
  // therefore reports its `Power::infy` exactly once, from the background.
  let background = apply(background_args.clone())?;
  let dims = nested_list_dims(dense_args.iter().flatten().next()?)?;

  // The union of the operands' stored positions: anywhere every operand sits
  // at its background, the result sits at the result background too.
  // (A sorted set: a linear membership test made this quadratic in the
  // number of stored entries.)
  let mut paths: std::collections::BTreeSet<Vec<usize>> =
    std::collections::BTreeSet::new();
  for (dense, bg) in dense_args.iter().zip(background_args.iter()) {
    let Some(dense) = dense else { continue };
    paths.extend(sparse_stored_paths(dense, bg));
  }

  // One evaluation over the lists of stored values (the head threads over
  // them, a scalar operand broadcasting), not one per position.
  let paths: Vec<Vec<usize>> = paths.into_iter().collect();
  let mut operands = Vec::with_capacity(args.len());
  for (dense, bg) in dense_args.iter().zip(background_args.iter()) {
    operands.push(match dense {
      Some(d) => Expr::List(
        paths
          .iter()
          .map(|path| nested_list_at(d, path))
          .collect::<Option<Vec<_>>>()?
          .into(),
      ),
      None => bg.clone(),
    });
  }
  let values = if paths.is_empty() {
    Vec::new()
  } else {
    match apply(operands)? {
      Expr::List(ref values) if values.len() == paths.len() => values.to_vec(),
      _ => return None,
    }
  };
  let background_str = expr_to_string(&background);
  let mut rules = Vec::with_capacity(paths.len());
  for (path, value) in paths.into_iter().zip(values) {
    if expr_to_string(&value) == background_str {
      continue;
    }
    rules.push(rule_expr(
      Expr::List(path.iter().map(|&i| Expr::Integer(i as i128)).collect()),
      value,
    ));
  }

  sparse_array_normalize_ast(&[Expr::List(rules.into()), dims, background]).ok()
}

/// `a / b` with a SparseArray operand. Routed as `a * b^-1` rather than as a
/// `Divide`, so that dividing by a SparseArray whose background is 0 reports
/// `Power::infy` — the tag wolframscript uses for the `1/sparse` syntax.
pub fn try_sparse_array_divide(a: &Expr, b: &Expr) -> Option<Expr> {
  let is_sparse = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "SparseArray");
  if !is_sparse(a) && !is_sparse(b) {
    return None;
  }
  let minus_one = Expr::Integer(-1);
  let inverse = if is_sparse(b) {
    try_sparse_array_arithmetic("Power", &[b.clone(), minus_one])?
  } else {
    crate::evaluator::evaluate_expr_to_expr(&pow(b.clone(), minus_one)).ok()?
  };
  try_sparse_array_arithmetic("Times", &[a.clone(), inverse])
}

/// Expand a SparseArray (any recognized form) into a dense nested list.
/// Called by `Normal[SparseArray[...]]`. A SparseArray that cannot be built
/// comes back as itself, which is what `Normal` gives for it.
pub fn sparse_array_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let normalized = sparse_array_normalize_ast(args)?;
  let sa_args = match &normalized {
    Expr::FunctionCall { name, args } if name == "SparseArray" => args,
    _ => {
      return Ok(normalized);
    }
  };
  // After normalize, either canonical (4 args starting with Automatic) or
  // still the original form (normalization failed).
  if sa_args.len() != 4
    || !matches!(&sa_args[0], Expr::Identifier(s) if s == "Automatic")
  {
    return Ok(normalized);
  }
  let dims: Vec<usize> = match &sa_args[1] {
    Expr::List(items) => {
      let mut d = Vec::with_capacity(items.len());
      for it in items {
        match expr_to_i128(it) {
          Some(n) if n >= 0 => d.push(n as usize),
          _ => {
            return Ok(normalized.clone());
          }
        }
      }
      d
    }
    _ => {
      return Ok(normalized);
    }
  };
  let default = &sa_args[2];
  let rules_list = sparse_array_extract_rules(&dims, &sa_args[3]);
  Ok(build_dense_from_rules(&dims, default, &rules_list))
}

/// `Take[SparseArray[…], spec1, spec2, …]` on the stored rules, so a large
/// array is never densified. Returns None for specs this does not cover
/// (and for out-of-range ones, whose message the dense path emits).
pub fn sparse_array_take(
  sa_args: &[Expr],
  specs: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if sa_args.len() != 4
    || !matches!(&sa_args[0], Expr::Identifier(s) if s == "Automatic")
  {
    return None;
  }
  let Expr::List(dim_items) = &sa_args[1] else {
    return None;
  };
  let dims: Vec<usize> = dim_items
    .iter()
    .map(|d| match d {
      Expr::Integer(n) if *n >= 0 => Some(*n as usize),
      _ => None,
    })
    .collect::<Option<_>>()?;
  if specs.is_empty() || specs.len() > dims.len() {
    return None;
  }
  // Each spec as an inclusive 1-based range with a positive step.
  let resolve = |i: i128, len: i128| if i < 0 { len + i + 1 } else { i };
  let mut ranges: Vec<(i128, i128, i128)> = Vec::with_capacity(specs.len());
  for (spec, &len) in specs.iter().zip(&dims) {
    let len = len as i128;
    let (start, end, step) = match spec {
      Expr::Integer(n) if *n >= 0 => (1, *n, 1),
      Expr::Integer(n) => (len + n + 1, len, 1),
      Expr::Identifier(a) if a == "All" => (1, len, 1),
      Expr::FunctionCall { name, args }
        if name == "UpTo" && args.len() == 1 =>
      {
        match &args[0] {
          Expr::Integer(n) if *n >= 0 => (1, (*n).min(len), 1),
          _ => return None,
        }
      }
      Expr::List(items) => match items.as_slice() {
        [Expr::Integer(m)] => (resolve(*m, len), resolve(*m, len), 1),
        [Expr::Integer(m), Expr::Integer(n)] => {
          (resolve(*m, len), resolve(*n, len), 1)
        }
        [Expr::Integer(m), Expr::Integer(n), Expr::Integer(k)] if *k > 0 => {
          (resolve(*m, len), resolve(*n, len), *k)
        }
        _ => return None,
      },
      _ => return None,
    };
    if start < 1 || end > len || end < start - 1 {
      if !ranges.is_empty() {
        return None;
      }
      let (from, to) = match spec {
        Expr::Integer(n) if *n >= 0 => (1, *n),
        Expr::Integer(n) => (*n, -1),
        Expr::List(items) => match items.as_slice() {
          [Expr::Integer(m)] => (*m, *m),
          [Expr::Integer(m), Expr::Integer(n), ..] => (*m, *n),
          _ => return None,
        },
        _ => return None,
      };
      let sa = Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: sa_args.to_vec().into(),
      };
      super::take_drop_message("Take", from, to, &sa);
      let mut take_args = vec![sa];
      take_args.extend_from_slice(specs);
      return Some(Ok(crate::helpers::call("Take", take_args)));
    }
    ranges.push((start, end, step));
  }

  let mut new_dims = dims.clone();
  for (d, &(start, end, step)) in new_dims.iter_mut().zip(&ranges) {
    *d = if end < start {
      0
    } else {
      ((end - start) / step + 1) as usize
    };
  }
  let rules: Vec<Expr> = sparse_array_extract_rules(&dims, &sa_args[3])
    .into_iter()
    .filter_map(|(mut pos, val)| {
      for (p, &(start, end, step)) in pos.iter_mut().zip(&ranges) {
        if *p < start || *p > end || (*p - start) % step != 0 {
          return None;
        }
        *p = (*p - start) / step + 1;
      }
      Some(Expr::Rule {
        pattern: Box::new(Expr::List(
          pos.into_iter().map(Expr::Integer).collect(),
        )),
        replacement: Box::new(val),
      })
    })
    .collect();
  Some(crate::evaluator::evaluate_function_call_ast(
    "SparseArray",
    &[
      Expr::List(rules.into()),
      Expr::List(new_dims.iter().map(|&n| Expr::Integer(n as i128)).collect()),
      sa_args[2].clone(),
    ],
  ))
}

/// Unified Tuples: `Tuples[{list1, list2, …}]` (one element from each
/// list, tuples take the outer head), `Tuples[list, n]` (all n-tuples,
/// tuples take list's head) and `Tuples[list, {n1, n2, …}]` (all
/// n1×n2×… arrays, list's head at every level; `{}` yields scalars).
/// Atomic expressions emit ::normal (with `{1, i}` positions for atomic
/// elements in the one-argument form) and invalid specs emit ::ilsmn.
pub fn tuples_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("Tuples", args);
  let show =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);
  let emit_normal = |position: &str| {
    crate::emit_message(&format!(
      "Tuples::normal: Nonatomic expression expected at position {} in {}.",
      position,
      show(&original())
    ));
  };

  // Elements and head of a nonatomic expression (None for the List head).
  fn parts(e: &Expr) -> Option<(&[Expr], Option<&str>)> {
    match e {
      Expr::List(items) => Some((items.as_slice(), None)),
      Expr::FunctionCall { name, args } => {
        Some((args.as_slice(), Some(name.as_str())))
      }
      _ => None,
    }
  }
  let wrap = |head: Option<&str>, elems: Vec<Expr>| -> Expr {
    match head {
      Some(h) => call(h, elems),
      None => Expr::List(elems.into()),
    }
  };

  if args.len() == 1 {
    // Tuples[{list1, list2, ...}] - one element from each list
    let Some((outer_items, outer_head)) = parts(&args[0]) else {
      emit_normal("1");
      return Ok(original());
    };
    let mut lists: Vec<&[Expr]> = Vec::new();
    for (i, item) in outer_items.iter().enumerate() {
      let Some((elems, _)) = parts(item) else {
        emit_normal(&format!("{{1, {}}}", i + 1));
        return Ok(original());
      };
      lists.push(elems);
    }

    let mut result: Vec<Vec<Expr>> = vec![vec![]];
    for list in &lists {
      let mut new_result = Vec::new();
      for tuple in &result {
        for item in *list {
          let mut new_tuple = tuple.clone();
          new_tuple.push(item.clone());
          new_result.push(new_tuple);
        }
      }
      result = new_result;
    }
    return Ok(Expr::List(
      result
        .into_iter()
        .map(|v| wrap(outer_head, v))
        .collect::<Vec<_>>()
        .into(),
    ));
  }

  // Tuples[list, n] or Tuples[list, {n1, n2, ...}]
  let Some((items, head)) = parts(&args[0]) else {
    emit_normal("1");
    return Ok(original());
  };

  let machine_nonneg = |e: &Expr| -> Option<usize> {
    match e {
      Expr::Integer(n) if (0..=i64::MAX as i128).contains(n) => {
        Some(*n as usize)
      }
      _ => None,
    }
  };
  let dims: Option<Vec<usize>> = match &args[1] {
    Expr::List(spec) => spec.iter().map(machine_nonneg).collect(),
    e => machine_nonneg(e).map(|n| vec![n]),
  };
  let Some(dims) = dims else {
    crate::emit_message(&format!(
      "Tuples::ilsmn: Single or list of non-negative machine-sized integers expected at position 2 of {}.",
      show(&original())
    ));
    return Ok(original());
  };

  // Reshape a flat tuple into nested arrays of the given shape, applying
  // the subject's head at every level. An empty shape yields the bare
  // element (so a scalar n behaves identically to the shape {n}).
  fn reshape(
    flat: &[Expr],
    dims: &[usize],
    wrap: &dyn Fn(Option<&str>, Vec<Expr>) -> Expr,
    head: Option<&str>,
  ) -> Expr {
    if dims.is_empty() {
      return flat[0].clone();
    }
    let chunk: usize = dims[1..].iter().product();
    let elems: Vec<Expr> = (0..dims[0])
      .map(|i| {
        reshape(&flat[i * chunk..(i + 1) * chunk], &dims[1..], wrap, head)
      })
      .collect();
    wrap(head, elems)
  }

  // All flat tuples of length n1*n2*..., in lexicographic order, generated
  // one at a time from a mixed-radix counter (building every prefix level
  // first held several copies of the result: 2.6 GB for
  // Tuples[{-1, 0, 1}, 12]).
  let total: usize = dims.iter().product();
  let k = items.len();
  let Some(count) = u32::try_from(total).ok().and_then(|t| k.checked_pow(t))
  else {
    return Ok(original());
  };
  let mut digits = vec![0usize; total];
  let mut tuples = Vec::with_capacity(count);
  for _ in 0..count {
    let flat: Vec<Expr> = digits.iter().map(|&d| items[d].clone()).collect();
    tuples.push(if dims.len() == 1 {
      wrap(head, flat)
    } else {
      reshape(&flat, &dims, &wrap, head)
    });
    for digit in digits.iter_mut().rev() {
      *digit += 1;
      if *digit < k {
        break;
      }
      *digit = 0;
    }
  }
  Ok(Expr::List(tuples.into()))
}

/// DistanceMatrix[{v1, v2, ...}] - matrix of pairwise distances.
///
/// Returns a symmetric matrix with a zero diagonal where entry (i, j) is the
/// distance between vi and vj. The default distance is EuclideanDistance.
/// An optional `DistanceFunction -> f` rule selects a different distance
/// function (e.g. ManhattanDistance, SquaredEuclideanDistance, or any binary
/// function f).
pub fn distance_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Return the unevaluated form for any argument shape that DistanceMatrix
  // cannot process (matching wolframscript, which leaves it symbolic).
  let unevaluated = || Ok(unevaluated("DistanceMatrix", args));

  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }

  let points = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => return unevaluated(),
  };

  // Determine the distance function: default EuclideanDistance, or the
  // replacement of an optional `DistanceFunction -> f` rule.
  let mut dist_fn = id_expr("EuclideanDistance");
  if let Some(opt) = args.get(1) {
    match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(
        pattern.as_ref(),
        Expr::Identifier(name) if name == "DistanceFunction"
      ) =>
      {
        dist_fn = (**replacement).clone();
      }
      _ => return unevaluated(),
    }
  }

  let n = points.len();
  let mut rows: Vec<Expr> = Vec::with_capacity(n);
  for i in 0..n {
    let mut row: Vec<Expr> = Vec::with_capacity(n);
    for j in 0..n {
      if i == j {
        row.push(Expr::Integer(0));
        continue;
      }
      let dist = apply_func_to_two_args(&dist_fn, &points[i], &points[j])?;
      row.push(dist);
    }
    rows.push(Expr::List(row.into()));
  }

  Ok(Expr::List(rows.into()))
}

/// Answer a `SparseArray[…]["property"]` query. Returns `None` for an
/// unrecognized property so the caller can report `SparseArray::nomthd`.
pub fn sparse_array_property(sa_args: &[Expr], prop: &str) -> Option<Expr> {
  if sa_args.len() != 4
    || !matches!(&sa_args[0], Expr::Identifier(s) if s == "Automatic")
  {
    return None;
  }
  let Expr::List(dim_items) = &sa_args[1] else {
    return None;
  };
  let dims: Vec<usize> = dim_items
    .iter()
    .map(|d| expr_to_i128(d).and_then(|n| usize::try_from(n).ok()))
    .collect::<Option<Vec<usize>>>()?;
  let background = &sa_args[2];
  let entries = sparse_array_extract_rules(&dims, &sa_args[3]);
  let total: usize = dims.iter().product();
  // The CSR triple `{1, {rowPointers, columnIndices}, values}`.
  let csr = match &sa_args[3] {
    Expr::List(items) if items.len() == 3 => match &items[1] {
      Expr::List(structure) if structure.len() == 2 => {
        Some((structure[0].clone(), structure[1].clone()))
      }
      _ => None,
    },
    _ => None,
  };
  let positions = || {
    Expr::List(
      entries
        .iter()
        .map(|(pos, _)| {
          Expr::List(pos.iter().map(|p| Expr::Integer(*p)).collect())
        })
        .collect(),
    )
  };

  match prop {
    "ExplicitValues" | "NonzeroValues" => {
      Some(Expr::List(entries.iter().map(|(_, v)| v.clone()).collect()))
    }
    "ExplicitPositions" | "NonzeroPositions" => Some(positions()),
    "ExplicitLength" => Some(Expr::Integer(entries.len() as i128)),
    "ImplicitValue" | "Background" => Some(background.clone()),
    "Dimensions" => Some(sa_args[1].clone()),
    "Density" => {
      if total == 0 {
        return Some(Expr::Real(0.0));
      }
      Some(Expr::Real(entries.len() as f64 / total as f64))
    }
    "RowPointers" => csr.map(|(rp, _)| rp),
    "ColumnIndices" => csr.map(|(_, ci)| ci),
    // One flat list of indices for a vector, one list per row otherwise.
    "AdjacencyLists" => {
      if dims.len() <= 1 {
        return Some(Expr::List(
          entries
            .iter()
            .filter_map(|(pos, _)| pos.first().map(|p| Expr::Integer(*p)))
            .collect(),
        ));
      }
      let mut rows: Vec<Vec<Expr>> = vec![Vec::new(); dims[0]];
      for (pos, _) in &entries {
        let (Some(row), Some(col)) = (pos.first(), pos.get(1)) else {
          continue;
        };
        let row = usize::try_from(*row).ok()?;
        if row >= 1 && row <= dims[0] {
          rows[row - 1].push(Expr::Integer(*col));
        }
      }
      Some(Expr::List(
        rows.into_iter().map(|r| Expr::List(r.into())).collect(),
      ))
    }
    // The same array with `_` wherever a value is stored.
    "PatternArray" => {
      let Expr::List(structure) = &sa_args[3] else {
        return None;
      };
      Some(call(
        "SparseArray",
        vec![
          sa_args[0].clone(),
          sa_args[1].clone(),
          sa_args[2].clone(),
          Expr::List(
            vec![
              structure[0].clone(),
              structure[1].clone(),
              id_expr("Pattern"),
            ]
            .into(),
          ),
        ],
      ))
    }
    "Properties" => Some(Expr::List(
      [
        "AdjacencyLists",
        "ColumnIndices",
        "Density",
        "ExplicitLength",
        "ExplicitPositions",
        "ExplicitValues",
        "ImplicitValue",
        "RowPointers",
        "ReplaceValues",
      ]
      .iter()
      .map(|p| str_expr(p))
      .collect(),
    )),
    _ => None,
  }
}
