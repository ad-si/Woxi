#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;
use crate::syntax::{BinaryOperator, UnaryOperator};

/// AST-based Fold/FoldList: fold a function over a list.
/// Fold[f, x, {a, b, c}] -> f[f[f[x, a], b], c]
/// Threads through any non-atomic head (e.g. Fold[f, x, g[a,b]] works).
pub fn fold_ast(
  func: &Expr,
  init: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items: &[Expr] = match list {
    Expr::List(items) => items.as_slice(),
    Expr::FunctionCall { args, .. } => args.as_slice(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Fold".to_string(),
        args: vec![func.clone(), init.clone(), list.clone()].into(),
      });
    }
  };

  let mut acc = init.clone();
  for item in items {
    acc = apply_func_to_two_args(func, &acc, item)?;
  }

  Ok(acc)
}

/// AST-based FoldWhile: fold `func` over `items` starting from `init`,
/// applying `test` to the most recent accumulator value(s) before each
/// further fold. Returns the first accumulator value for which `test`
/// does not yield `True` (or the final value if `test` never fails).
///
/// `FoldWhile[f, x, {a1, …}, test]`               (m = 1)
/// `FoldWhile[f, x, {a1, …}, test, m]`            — test gets last m values
/// `FoldWhile[f, x, {a1, …}, test, m, n]`         — n extra folds after the
///                                                   test fails (or `-n` to
///                                                   step back n values)
pub fn fold_while_ast(
  func: &Expr,
  init: &Expr,
  items: &[Expr],
  test: &Expr,
  m: NestWhileM,
  extra_n: i128,
) -> Result<Expr, InterpreterError> {
  // Build the accumulator history: r0 = init, r1 = f[r0, a1], …
  // Wolfram tests the current value(s) before each further fold and, unlike
  // NestWhile, tests immediately even before m values are available.
  let mut history = vec![init.clone()];
  let mut consumed = 0usize;
  loop {
    let window: Vec<Expr> = match m {
      NestWhileM::Last(n) => {
        history[history.len().saturating_sub(n)..].to_vec()
      }
      NestWhileM::All => history.clone(),
    };
    let test_result = apply_func_to_n_args(test, &window)?;
    if expr_to_bool(&test_result) != Some(true) {
      break;
    }
    if consumed >= items.len() {
      break;
    }
    let next =
      apply_func_to_two_args(func, history.last().unwrap(), &items[consumed])?;
    history.push(next);
    consumed += 1;
  }
  // `history.last()` is the fail point (or the final value if the list was
  // exhausted). Apply the optional extra-step offset.
  let target = (history.len() as i128 - 1) + extra_n;
  if target < 0 {
    return Ok(history[0].clone());
  }
  if (target as usize) < history.len() {
    return Ok(history[target as usize].clone());
  }
  // Positive offset: keep folding with any remaining list elements.
  let mut current = history.last().cloned().unwrap_or_else(|| init.clone());
  let extra = (target as usize) - (history.len() - 1);
  for i in 0..extra {
    let idx = consumed + i;
    if idx >= items.len() {
      break;
    }
    current = apply_func_to_two_args(func, &current, &items[idx])?;
  }
  Ok(current)
}

/// AST-based Nest: apply a function n times.
/// Nest[f, x, n] -> f[f[f[...f[x]...]]] (n times)
pub fn nest_ast(
  func: &Expr,
  init: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "Nest requires non-negative count".into(),
    ));
  }

  let mut result = init.clone();
  for _ in 0..n {
    result = apply_func_ast(func, &result)?;
  }

  Ok(result)
}

/// AST-based NestList: build a list by repeatedly applying a function.
/// NestList[f, x, n] -> {x, f[x], f[f[x]], ..., f^n[x]}
pub fn nest_list_ast(
  func: &Expr,
  init: &Expr,
  n: i128,
) -> Result<Expr, InterpreterError> {
  if n < 0 {
    return Err(InterpreterError::EvaluationError(
      "NestList requires non-negative count".into(),
    ));
  }

  let mut results = vec![init.clone()];
  let mut current = init.clone();
  for _ in 0..n {
    current = apply_func_ast(func, &current)?;
    results.push(current.clone());
  }

  Ok(Expr::List(results.into()))
}

/// AST-based FixedPoint: apply function until result stops changing.
/// FixedPoint[f, x] -> fixed point of f starting from x
pub fn fixed_point_ast(
  func: &Expr,
  init: &Expr,
  max_iterations: Option<i128>,
  same_test: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut current = init.clone();

  for _ in 0..max {
    let next = apply_func_ast(func, &current)?;
    if fixed_point_converged(same_test, &current, &next)? {
      return Ok(next);
    }
    current = next;
  }

  Ok(current)
}

/// Convergence check shared by FixedPoint and FixedPointList: the
/// `SameTest -> f` option applied to {previous, next}, or the default
/// SameQ comparison (which treats two machine reals differing by at most
/// one ULP as identical).
fn fixed_point_converged(
  same_test: Option<&Expr>,
  current: &Expr,
  next: &Expr,
) -> Result<bool, InterpreterError> {
  match same_test {
    None => Ok(same_q_default(current, next)),
    Some(test) => {
      let result =
        apply_func_to_n_args(test, &[current.clone(), next.clone()])?;
      Ok(expr_to_bool(&result) == Some(true))
    }
  }
}

/// FixedPoint/FixedPointList default comparison: SameQ semantics.
fn same_q_default(a: &Expr, b: &Expr) -> bool {
  crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
    || crate::functions::boolean_ast::same_q_real_bigfloat(a, b)
}

/// AST-based Accumulate: cumulative sums.
/// Threads through any non-atomic head, preserving it (e.g. Accumulate[g[1,2,3]]
/// returns g[1, 3, 6]).
/// FindPeaks[list] — the local maxima of a numeric list.
///
/// Each maximal run of equal values that strictly exceeds its real neighbors
/// (with the list boundaries acting as -Infinity) and does not span the whole
/// list is a peak. It is reported as `{position, value}`, where `position` is
/// the mean of the run's 1-based indices (so a flat plateau yields a
/// half-integer center, matching wolframscript: FindPeaks[{1,3,3,1}] ->
/// {{5/2, 3}}). A list with no interior boundary (e.g. `{5}` or `{3,3,3}`)
/// has no peaks.
pub fn find_peaks_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "FindPeaks".to_string(),
    args: vec![list.clone()].into(),
  };
  // Anything that is not a consistent list of real values emits FindPeaks::arg
  // and stays unevaluated, matching wolframscript.
  let bail = || {
    crate::emit_message(&format!(
      "FindPeaks::arg: The argument {} at position 1 is not a consistent list of real values.",
      crate::syntax::format_expr(list, crate::syntax::ExprForm::Output)
    ));
    Ok(unevaluated())
  };
  let items: &[Expr] = match list {
    Expr::List(items) => items.as_slice(),
    _ => return bail(),
  };
  // Every element must be a real numeric value.
  let mut vals: Vec<f64> = Vec::with_capacity(items.len());
  for it in items {
    match crate::functions::math_ast::try_eval_to_f64(it) {
      Some(v) => vals.push(v),
      None => return bail(),
    }
  }

  let n = vals.len();
  let mut peaks: Vec<Expr> = Vec::new();
  let mut i = 0usize;
  while i < n {
    // Extend a maximal run of equal values [i..=j] (0-based).
    let mut j = i;
    while j + 1 < n && vals[j + 1] == vals[i] {
      j += 1;
    }
    let v = vals[i];
    let left_ok = i == 0 || vals[i - 1] < v;
    let right_ok = j + 1 == n || vals[j + 1] < v;
    let spans_whole_list = i == 0 && j + 1 == n;
    if left_ok && right_ok && !spans_whole_list {
      // Center = mean of the 1-based positions (i+1 .. j+1).
      let pos_sum = (i + 1 + j + 1) as i128;
      let pos = if pos_sum % 2 == 0 {
        Expr::Integer(pos_sum / 2)
      } else {
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Divide".to_string(),
          args: vec![Expr::Integer(pos_sum), Expr::Integer(2)].into(),
        })?
      };
      peaks.push(Expr::List(vec![pos, items[i].clone()].into()));
    }
    i = j + 1;
  }
  Ok(Expr::List(peaks.into()))
}

pub fn accumulate_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  // Determine the head to wrap the result in and the items to accumulate over.
  let (items, head): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    // Rational/Complex are atoms — their internal args must not accumulate.
    Expr::FunctionCall { name, args }
      if !crate::functions::list_helpers_ast::sorting::is_atomic_arg(list) =>
    {
      (args.as_slice(), Some(name.clone()))
    }
    _ => {
      crate::functions::list_helpers_ast::sorting::emit_nonatomic_normal_message(
        "Accumulate",
        std::slice::from_ref(list),
      );
      return Ok(Expr::FunctionCall {
        name: "Accumulate".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  };

  let wrap = |elems: Vec<Expr>| -> Expr {
    match &head {
      Some(name) => Expr::FunctionCall {
        name: name.clone(),
        args: elems.into(),
      },
      None => Expr::List(elems.into()),
    }
  };

  if items.is_empty() {
    return Ok(wrap(Vec::new()));
  }

  // Accumulate via Plus so each partial sum keeps exact arithmetic and is
  // promoted element-by-element: a prefix summing only integers stays an
  // integer, and a Real (or Rational) only affects the sums it enters —
  // matching Wolfram's Accumulate[{4, -50, 20.0, 12.6}] = {4, -46, -26., -13.4}.
  let mut results = Vec::new();
  let mut running_sum = items[0].clone();
  results.push(running_sum.clone());
  for item in &items[1..] {
    running_sum = crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[running_sum, item.clone()],
    )?;
    results.push(running_sum.clone());
  }
  Ok(wrap(results))
}

/// AST-based Differences: successive differences.
pub fn differences_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  differences_n_ast(list, 1)
}

/// Differences[list, n] - n-th order differences
pub fn differences_n_ast(
  list: &Expr,
  n: usize,
) -> Result<Expr, InterpreterError> {
  differences_step_ast(list, n, 1)
}

/// Differences[list, n, s] - n-th order differences with step `s`.
///
/// Each pass replaces the list with `Drop[#, s] - Drop[#, -s]`, i.e.
/// `{list[s+1] - list[1], list[s+2] - list[2], ...}`, applied `n` times.
/// `s = 1` gives the ordinary successive differences.
pub fn differences_step_ast(
  list: &Expr,
  n: usize,
  s: usize,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Differences".to_string(),
        args: vec![list.clone()].into(),
      });
    }
  };

  let s = s.max(1);
  let mut current = items;
  for _ in 0..n {
    if current.len() <= s {
      return Ok(Expr::List(vec![].into()));
    }
    let mut next = Vec::new();
    for i in s..current.len() {
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[current[i].clone(), current[i - s].clone()],
      )?;
      next.push(diff);
    }
    current = next.into();
  }

  Ok(Expr::List(current))
}

/// Differences[list, {n1, n2, ...}] - apply `ni` differences at level `i`.
///
/// At the top level we take `n1` successive differences. Each element of
/// the resulting list is then recursively passed to
/// `differences_spec_ast(_, {n2, ...})` which applies the next level of
/// differencing. A single-element spec `{n}` is equivalent to
/// `Differences[list, n]`.
pub fn differences_spec_ast(
  list: &Expr,
  spec: &[usize],
) -> Result<Expr, InterpreterError> {
  if spec.is_empty() {
    return Ok(list.clone());
  }
  let first = spec[0];
  let outer = differences_n_ast(list, first)?;
  if spec.len() == 1 {
    return Ok(outer);
  }
  let rest = &spec[1..];
  let items: Vec<Expr> = match &outer {
    Expr::List(items) => items.to_vec(),
    _ => return Ok(outer),
  };
  let mut result = Vec::with_capacity(items.len());
  for item in items {
    result.push(differences_spec_ast(&item, rest)?);
  }
  Ok(Expr::List(result.into()))
}

/// AST-based Scan: apply function to each element for side effects.
/// Returns Null but evaluates function on each element.
pub fn scan_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      for item in items {
        apply_func_ast(func, item)?;
      }
    }
    // Rational / Complex are atoms — no parts to scan over.
    _ if crate::functions::predicate_ast::is_atomic_number(list) => {}
    _ => {
      // For any compound expression, decompose into head + children,
      // and apply func to each child for side effects.
      // E.g. Scan[f, Power[x, 2]] applies f[x] and f[2]
      use crate::functions::expr_form::{ExprForm, decompose_expr};
      match decompose_expr(list) {
        ExprForm::Composite { children, .. } => {
          for child in &children {
            apply_func_ast(func, child)?;
          }
        }
        ExprForm::Atom(_) => {
          // Atoms have no parts to scan over
        }
      }
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// Scan[f, expr, levelspec] — apply `func` (for side effects) to each part of
/// `expr` at the given levels, in the same order as Level[expr, levelspec],
/// returning Null.
pub fn scan_levelspec_ast(
  func: &Expr,
  expr: &Expr,
  levelspec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Reuse the Level machinery to enumerate the parts in the correct order.
  let level_call = Expr::FunctionCall {
    name: "Level".to_string(),
    args: vec![expr.clone(), levelspec.clone()].into(),
  };
  let parts = crate::evaluator::evaluate_expr_to_expr(&level_call)?;
  match parts {
    Expr::List(ref items) => {
      for item in items.iter() {
        apply_func_ast(func, item)?;
      }
      Ok(Expr::Identifier("Null".to_string()))
    }
    // Level could not interpret the spec — keep the call unevaluated.
    _ => Ok(Expr::FunctionCall {
      name: "Scan".to_string(),
      args: vec![func.clone(), expr.clone(), levelspec.clone()].into(),
    }),
  }
}

/// AST-based FoldList: fold showing intermediate values.
/// FoldList[f, x, {a, b, c}] -> {x, f[x, a], f[f[x, a], b], ...}
/// Threads through any non-atomic head: FoldList[f, x, g[a,b]]
/// returns g[x, f[x, a], f[f[x, a], b]].
pub fn fold_list_ast(
  func: &Expr,
  init: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let (items, head): (&[Expr], Option<String>) = match list {
    Expr::List(items) => (items.as_slice(), None),
    Expr::FunctionCall { name, args } => (args.as_slice(), Some(name.clone())),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FoldList".to_string(),
        args: vec![func.clone(), init.clone(), list.clone()].into(),
      });
    }
  };

  let mut results = vec![init.clone()];
  let mut acc = init.clone();
  for item in items {
    acc = apply_func_to_two_args(func, &acc, item)?;
    results.push(acc.clone());
  }

  match head {
    Some(name) => Ok(Expr::FunctionCall {
      name,
      args: results.into(),
    }),
    None => Ok(Expr::List(results.into())),
  }
}

/// AST-based SequenceFoldList.
///
/// `SequenceFoldList[f, {x1,…,xn}, {a1,a2,…}]` keeps a running history seeded
/// with the n initial values. At each step it applies `f` to the last n
/// history values followed by the next element of the a-list, appending the
/// result to the history; it returns the full history.
///
/// `SequenceFoldList[f, {x1,…,xn}, {a1,…}, k]` instead feeds `f` the last n
/// history values plus a sliding window of `k - n` a-values (the window
/// advances by one each step). The default `k` is `n + 1` (window size 1).
/// SequenceFold / SequenceFoldList require their 2nd (initial values) and 3rd
/// (sequence) arguments to be lists. A non-list emits `fname::invl` naming the
/// offending argument and the call stays unevaluated. Returns the unevaluated
/// form if invalid, else None. The 2nd argument is checked before the 3rd.
fn sequence_fold_invl(fname: &str, args: &[Expr]) -> Option<Expr> {
  for idx in [1usize, 2] {
    if let Some(arg) = args.get(idx)
      && !matches!(arg, Expr::List(_))
    {
      crate::emit_message(&format!(
        "{fname}::invl: The argument {} is not a list.",
        crate::syntax::format_expr(arg, crate::syntax::ExprForm::Output)
      ));
      return Some(Expr::FunctionCall {
        name: fname.to_string(),
        args: args.to_vec().into(),
      });
    }
  }
  None
}

pub fn sequence_fold_list_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "SequenceFoldList".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() < 3 || args.len() > 4 {
    return unevaluated();
  }
  if let Some(uneval) = sequence_fold_invl("SequenceFoldList", args) {
    return Ok(uneval);
  }
  let func = &args[0];
  let (Expr::List(x_init), Expr::List(a_items)) = (&args[1], &args[2]) else {
    return unevaluated();
  };
  let n = x_init.len();
  if n == 0 {
    return unevaluated();
  }
  // Window of a-values consumed per step: k - n, default 1 (k = n + 1).
  let window: usize = if args.len() == 4 {
    match expr_to_i128(&args[3]) {
      Some(k) if k > n as i128 => (k - n as i128) as usize,
      _ => return unevaluated(),
    }
  } else {
    1
  };

  let a: Vec<Expr> = a_items.iter().cloned().collect();
  let mut history: Vec<Expr> = x_init.iter().cloned().collect();
  if window <= a.len() {
    let num_steps = a.len() - window + 1;
    for i in 0..num_steps {
      let mut call_args: Vec<Expr> = history[history.len() - n..].to_vec();
      call_args.extend_from_slice(&a[i..i + window]);
      let next = apply_func_to_args(func, &call_args)?;
      history.push(next);
    }
  }
  Ok(Expr::List(history.into()))
}

/// AST-based SequenceFold: the last element of the SequenceFoldList history.
pub fn sequence_fold_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Validate under the SequenceFold name first, so the invl message names this
  // function rather than the internal SequenceFoldList delegate.
  if (3..=4).contains(&args.len())
    && let Some(uneval) = sequence_fold_invl("SequenceFold", args)
  {
    return Ok(uneval);
  }
  match sequence_fold_list_ast(args)? {
    Expr::List(ref items) if !items.is_empty() => {
      Ok(items.last().cloned().unwrap())
    }
    // SequenceFoldList stayed unevaluated → keep SequenceFold unevaluated.
    _ => Ok(Expr::FunctionCall {
      name: "SequenceFold".to_string(),
      args: args.to_vec().into(),
    }),
  }
}

/// AST-based FixedPointList: list of values until fixed point.
pub fn fixed_point_list_ast(
  func: &Expr,
  init: &Expr,
  max_iterations: Option<i128>,
  same_test: Option<&Expr>,
) -> Result<Expr, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut results = vec![init.clone()];
  let mut current = init.clone();

  for _ in 0..max {
    let next = apply_func_ast(func, &current)?;
    let converged = fixed_point_converged(same_test, &current, &next)?;
    results.push(next.clone());
    if converged {
      break;
    }
    current = next;
  }

  Ok(Expr::List(results.into()))
}

/// AST-based NestWhile: nest while condition is true.
///
/// `extra_n` corresponds to Wolfram's 6th argument: after the test stops
/// being True (or `max_iterations` is reached), apply `func` an additional
/// `n` times when `n > 0`, or return the result `|n|` iterations earlier
/// when `n < 0`.
/// Selector for the `m` argument of NestWhile.
#[derive(Clone, Copy, Debug)]
pub enum NestWhileM {
  /// Pass last `n` history values to the test predicate.
  Last(usize),
  /// Pass *all* history values to the test predicate.
  All,
}

pub fn nest_while_ast(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  m: NestWhileM,
  max_iterations: Option<i128>,
  extra_n: i128,
) -> Result<Expr, InterpreterError> {
  // We always materialise the full history when `extra_n` could need it,
  // so that negative `extra_n` can step back through previously seen values.
  let history = nest_while_history(func, init, test, m, max_iterations)?;
  let len = history.len() as i128;
  let target = (len - 1) + extra_n;
  if target < 0 {
    // Asked for a step before the initial value — Wolfram returns the
    // initial value in this case as the closest available result.
    return Ok(history[0].clone());
  }
  if (target as usize) < history.len() {
    return Ok(history[target as usize].clone());
  }
  // Need to keep applying `func` past the point where the test stopped.
  let mut current = history.last().cloned().unwrap_or_else(|| init.clone());
  let extra = (target as usize) - (history.len() - 1);
  for _ in 0..extra {
    current = apply_func_ast(func, &current)?;
  }
  Ok(current)
}

/// AST-based NestWhileList: like NestWhile but returns list.
pub fn nest_while_list_ast(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  m: NestWhileM,
  max_iterations: Option<i128>,
  extra_n: i128,
) -> Result<Expr, InterpreterError> {
  let mut history = nest_while_history(func, init, test, m, max_iterations)?;
  if extra_n > 0 {
    // Continue applying `func` for the requested extra iterations.
    let mut current = history.last().cloned().unwrap_or_else(|| init.clone());
    for _ in 0..extra_n {
      current = apply_func_ast(func, &current)?;
      history.push(current.clone());
    }
  } else if extra_n < 0 {
    // Drop the trailing `|extra_n|` values, but never below the initial.
    let drop = (-extra_n) as usize;
    let new_len = history.len().saturating_sub(drop).max(1);
    history.truncate(new_len);
  }
  Ok(Expr::List(history.into()))
}

/// Iterate `func` from `init` while `test` is True, returning the entire
/// history (initial value first). Stops at `max_iterations` if provided.
/// `m` selects which history values are supplied to `test`.
fn nest_while_history(
  func: &Expr,
  init: &Expr,
  test: &Expr,
  m: NestWhileM,
  max_iterations: Option<i128>,
) -> Result<Vec<Expr>, InterpreterError> {
  let max = max_iterations.unwrap_or(10000);
  let mut history = vec![init.clone()];
  for _ in 0..max {
    // `NestWhile[f, expr, test, m]` supplies the m most recent results to
    // `test`, so `f` must be applied at least m-1 times before the first test
    // can run (e.g. `NestWhile[f, x, test, 2]` -> `f[x]`). While fewer than m
    // results are available, keep applying `f` without testing.
    let window_full = match m {
      NestWhileM::Last(n) => history.len() >= n,
      NestWhileM::All => true,
    };
    if window_full {
      let test_args: Vec<Expr> = match m {
        NestWhileM::Last(n) => {
          let start = history.len().saturating_sub(n);
          history[start..].to_vec()
        }
        NestWhileM::All => history.clone(),
      };
      let test_result = apply_func_to_n_args(test, &test_args)?;
      if expr_to_bool(&test_result) != Some(true) {
        break;
      }
    }
    let next = apply_func_ast(func, history.last().unwrap())?;
    history.push(next);
  }
  Ok(history)
}

/// Extract the children of any expression in Wolfram canonical form.
/// Returns None for atomic expressions (Integer, Real, String, Symbol).
fn expr_children(expr: &Expr) -> Option<Vec<Expr>> {
  // Rational and Complex are atoms: Apply on them returns them unchanged.
  if crate::functions::predicate_ast::is_atomic_number(expr) {
    return None;
  }
  match expr {
    Expr::List(items) => Some(items.to_vec()),
    Expr::FunctionCall { args, .. } => Some(args.to_vec()),
    Expr::BinaryOp { op, left, right } => {
      match op {
        // a - b → Plus[a, Times[-1, b]]
        BinaryOperator::Minus => Some(vec![
          left.as_ref().clone(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), right.as_ref().clone()].into(),
          },
        ]),
        // 1/b → Power[b, -1]; a/b → Times[a, Power[b, -1]]
        BinaryOperator::Divide => {
          if matches!(left.as_ref(), Expr::Integer(1)) {
            Some(vec![right.as_ref().clone(), Expr::Integer(-1)])
          } else {
            Some(vec![
              left.as_ref().clone(),
              Expr::FunctionCall {
                name: "Power".to_string(),
                args: vec![right.as_ref().clone(), Expr::Integer(-1)].into(),
              },
            ])
          }
        }
        // Flat (associative) operators such as Alternatives (`a | b | c`) are
        // stored as nested BinaryOp chains but are conceptually a single
        // n-ary head, so flatten same-operator nestings into siblings —
        // matching WS where `a | b | c` is Alternatives[a, b, c]. Power is
        // right-associative and genuinely binary (`a^b^c` = a^(b^c)), so it
        // keeps its two operands.
        BinaryOperator::Plus
        | BinaryOperator::Times
        | BinaryOperator::And
        | BinaryOperator::Or
        | BinaryOperator::StringJoin
        | BinaryOperator::Alternatives => {
          fn flatten(op: &BinaryOperator, e: &Expr, out: &mut Vec<Expr>) {
            if let Expr::BinaryOp {
              op: inner,
              left,
              right,
            } = e
              && inner == op
            {
              flatten(op, left, out);
              flatten(op, right, out);
              return;
            }
            out.push(e.clone());
          }
          let mut parts = Vec::new();
          flatten(op, left, &mut parts);
          flatten(op, right, &mut parts);
          Some(parts)
        }
        // Power (right-associative, binary) and any remaining operators.
        _ => Some(vec![left.as_ref().clone(), right.as_ref().clone()]),
      }
    }
    Expr::UnaryOp { operand, .. } => Some(vec![operand.as_ref().clone()]),
    // Comparison chains decompose to Op[a, b, c] (uniform) or
    // Inequality[a, Less, b, ...] (mixed), matching wolframscript.
    Expr::Comparison {
      operands,
      operators,
    } => {
      let (_, args) =
        crate::syntax::comparison_head_and_args(operands, operators);
      Some(args)
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some(vec![pattern.as_ref().clone(), replacement.as_ref().clone()]),
    Expr::Association(pairs) => Some(
      pairs
        .iter()
        .map(|(k, v)| Expr::Rule {
          pattern: Box::new(k.clone()),
          replacement: Box::new(v.clone()),
        })
        .collect(),
    ),
    // Atomic expressions have no children
    _ => None,
  }
}

/// Apply[f, list] - applies f to the elements of list (f @@ list)
pub fn apply_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  // For associations, Apply operates on values (not rules)
  let items = if let Expr::Association(pairs) = list {
    pairs.iter().map(|(_, v)| v.clone()).collect()
  } else {
    match expr_children(list) {
      Some(items) => items,
      None => {
        // Atoms have no children; Apply on an atom returns the atom unchanged
        return Ok(list.clone());
      }
    }
  };
  match func {
    Expr::Identifier(name) => {
      // Build `name[items]` and evaluate it through the full pipeline so the
      // new head's attributes govern argument evaluation: a non-holding head
      // (List, Times, …) evaluates items that the source head kept unevaluated
      // (e.g. Apply[List, Hold[1 + 1]] -> {2}), while a holding head (Hold,
      // HoldForm) leaves them untouched (Apply[Hold, Hold[1 + 1]] -> Hold[1 + 1]).
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: name.clone(),
        args: items.into(),
      })
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, &items);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(items.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    // Apply replaces the head of `list` with `func`, whatever `func` is:
    //   Apply[{g, h}, {1, 2}]          -> {g, h}[1, 2]
    //   Apply[3, {1, 2}]               -> 3[1, 2]
    //   Apply[f[a], {1, 2}]            -> f[a][1, 2]
    //   Apply[Composition[f, g], {1, 2}] -> f[g[1, 2]]
    // Build the curried application and evaluate it so applicable heads
    // (Composition, pure functions, …) reduce while inert heads stay symbolic.
    _ => crate::evaluator::apply_curried_call(func, &items),
  }
}

/// Apply[f, expr, levelspec] - replace heads at specified levels.
pub fn apply_at_level_ast(
  func: &Expr,
  expr: &Expr,
  level_spec: &Expr,
) -> Result<Expr, InterpreterError> {
  // Compute depth so we can resolve negative level values:
  //   `-k` selects subexpressions whose Depth is k, i.e. level (D - k)
  // where D = Depth[expr] (atoms have Depth 1).
  let depth = expr_depth(expr);
  let resolve = |n: i128| -> usize {
    if n < 0 {
      let k = -n;
      let pos = depth as i128 - k;
      if pos < 0 { 0 } else { pos as usize }
    } else {
      n as usize
    }
  };

  // Parse level spec. For the integer form, `Apply[f, expr, n]` is
  // equivalent to `Apply[f, expr, {1, n}]` — levels 1 through n — not
  // {0, n} as Woxi previously assumed.
  let (min_level, max_level) = match level_spec {
    Expr::Identifier(s) if s == "Infinity" => (1usize, usize::MAX),
    Expr::Integer(n) => (1usize, resolve(*n)),
    Expr::List(levels) => {
      if levels.len() == 1 {
        if let Some(n) = expr_to_i128(&levels[0]) {
          let r = resolve(n);
          (r, r)
        } else {
          (1, 1)
        }
      } else if levels.len() == 2 {
        let min = match expr_to_i128(&levels[0]) {
          Some(n) => resolve(n),
          None => 0,
        };
        let max = match &levels[1] {
          Expr::Identifier(s) if s == "Infinity" => usize::MAX,
          other => match expr_to_i128(other) {
            Some(n) => resolve(n),
            None => 1,
          },
        };
        (min, max)
      } else {
        (1, 1)
      }
    }
    _ => (1, 1),
  };

  apply_at_level_recursive(func, expr, 0, min_level, max_level)
}

/// Mathematica-style Depth: atoms have depth 1, compound expressions are
/// 1 + max(depth of children).
fn expr_depth(expr: &Expr) -> usize {
  let children: Vec<&Expr> = match expr {
    Expr::List(items) => items.iter().collect(),
    Expr::FunctionCall { args, .. } => args.iter().collect(),
    Expr::BinaryOp { left, right, .. } => vec![left.as_ref(), right.as_ref()],
    Expr::UnaryOp { operand, .. } => vec![operand.as_ref()],
    _ => return 1,
  };
  if children.is_empty() {
    return 1;
  }
  1 + children.iter().map(|c| expr_depth(c)).max().unwrap_or(0)
}

fn apply_at_level_recursive(
  func: &Expr,
  expr: &Expr,
  current_level: usize,
  min_level: usize,
  max_level: usize,
) -> Result<Expr, InterpreterError> {
  // Normalize BinaryOp / UnaryOp into a FunctionCall so Apply can descend
  // into Plus/Times/Power and other operators that the parser sometimes
  // leaves as Expr::BinaryOp rather than FunctionCall.
  let normalized: Expr = match expr {
    Expr::BinaryOp { op, left, right } => Expr::FunctionCall {
      name: binary_op_to_name(*op).to_string(),
      args: vec![(**left).clone(), (**right).clone()].into(),
    },
    Expr::UnaryOp { op, operand } => Expr::FunctionCall {
      name: unary_op_to_name(*op).to_string(),
      args: vec![(**operand).clone()].into(),
    },
    other => other.clone(),
  };
  let (items, is_list, head_name) = match &normalized {
    Expr::List(items) => (items.clone(), true, None),
    Expr::FunctionCall { name, args } => {
      (args.clone(), false, Some(name.clone()))
    }
    _ => return Ok(expr.clone()),
  };

  // Recurse into children first if we haven't reached max_level
  let new_items: Vec<Expr> = if current_level < max_level {
    items
      .iter()
      .map(|item| {
        apply_at_level_recursive(
          func,
          item,
          current_level + 1,
          min_level,
          max_level,
        )
      })
      .collect::<Result<Vec<_>, _>>()?
  } else {
    items.to_vec()
  };

  // Replace head at this level if in range
  if current_level >= min_level && current_level <= max_level {
    apply_func_as_head(func, &new_items)
  } else if is_list {
    Ok(Expr::List(new_items.into()))
  } else if let Some(name) = head_name {
    crate::evaluator::evaluate_function_call_ast(&name, &new_items)
  } else {
    Ok(expr.clone())
  }
}

fn binary_op_to_name(op: BinaryOperator) -> &'static str {
  use BinaryOperator::*;
  match op {
    Plus => "Plus",
    Minus => "Subtract",
    Times => "Times",
    Divide => "Divide",
    Power => "Power",
    And => "And",
    Or => "Or",
    StringJoin => "StringJoin",
    Alternatives => "Alternatives",
  }
}

fn unary_op_to_name(op: UnaryOperator) -> &'static str {
  use UnaryOperator::*;
  match op {
    Minus => "Times",
    Not => "Not",
  }
}

fn apply_func_as_head(
  func: &Expr,
  items: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, items)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, items);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(items.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    // Apply replaces the head with `func` regardless of what `func` is.
    _ => crate::evaluator::apply_curried_call(func, items),
  }
}

/// Outer[f, list1, list2, ...] - generalized outer product.
/// Threads through any shared head (not just `List`). All arguments must have
/// the same head; if they differ, the call is returned unevaluated.
fn outer_ast(func: &Expr, lists: &[Expr]) -> Result<Expr, InterpreterError> {
  outer_ast_with_levels(func, lists, &[])
}

/// TensorProduct[a, b, …]. The head is Flat, so nested TensorProducts are
/// flattened first. Rank-0 numeric quantities (NumericQ, e.g. `2`, `Pi`,
/// `Sqrt[2]`) are scalars: they factor out and multiply the rest via `Times`.
/// Among the remaining factors, maximal runs of explicit arrays contract via
/// `Outer[Times, …]`; a non-numeric symbolic atom (`a`) cannot be contracted
/// and is kept as a separate factor, so the result stays a symbolic
/// `TensorProduct` (rendered with the U+F3DA operator). Matches wolframscript:
/// `TensorProduct[2, a, {1,2}]` → `2 (a ⊗ {1, 2})`,
/// `TensorProduct[{1,2}, {a,b}, c]` → `{{a, b}, {2 a, 2 b}} ⊗ c`.
pub fn tensor_product_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Flatten nested TensorProduct (Flat attribute).
  let mut flat: Vec<Expr> = Vec::new();
  for a in args {
    if let Expr::FunctionCall { name, args: inner } = a
      && name == "TensorProduct"
    {
      flat.extend(inner.iter().cloned());
    } else {
      flat.push(a.clone());
    }
  }
  if flat.len() == 1 {
    return Ok(flat.into_iter().next().unwrap());
  }

  let times = Expr::Identifier("Times".to_string());
  let is_scalar = |e: &Expr| -> bool {
    !matches!(e, Expr::List(_))
      && crate::functions::predicate_ast::is_numeric_q(e)
  };

  // Pull scalars out; they distribute over the tensor part via Times.
  let mut scalars: Vec<Expr> = Vec::new();
  let mut rest: Vec<Expr> = Vec::new();
  for a in flat {
    if is_scalar(&a) {
      scalars.push(a);
    } else {
      rest.push(a);
    }
  }

  // Among the non-scalar factors, contract maximal runs of explicit arrays.
  let mut factors: Vec<Expr> = Vec::new();
  let mut group: Vec<Expr> = Vec::new();
  let flush = |group: &mut Vec<Expr>,
               factors: &mut Vec<Expr>|
   -> Result<(), InterpreterError> {
    if group.len() == 1 {
      factors.push(group[0].clone());
    } else if group.len() > 1 {
      factors.push(outer_ast(&times, group)?);
    }
    group.clear();
    Ok(())
  };
  for a in &rest {
    if matches!(a, Expr::List(_)) {
      group.push(a.clone());
    } else {
      flush(&mut group, &mut factors)?;
      factors.push(a.clone());
    }
  }
  flush(&mut group, &mut factors)?;

  // Assemble the tensor part from the surviving factors.
  let tensor_part = match factors.len() {
    0 => None,
    1 => Some(factors.into_iter().next().unwrap()),
    _ => Some(Expr::FunctionCall {
      name: "TensorProduct".to_string(),
      args: factors.into(),
    }),
  };

  // Multiply the scalar product back in (Times distributes over arrays).
  match (scalars.is_empty(), tensor_part) {
    (true, Some(tp)) => Ok(tp),
    (false, Some(tp)) => {
      let mut prod = scalars;
      prod.push(tp);
      crate::evaluator::evaluate_function_call_ast("Times", &prod)
    }
    (_, None) => {
      crate::evaluator::evaluate_function_call_ast("Times", &scalars)
    }
  }
}

pub fn outer_ast_with_levels(
  func: &Expr,
  lists: &[Expr],
  levels: &[usize],
) -> Result<Expr, InterpreterError> {
  if lists.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Outer expects at least one list argument".into(),
    ));
  }

  // Determine the common head. All arguments must share the same head.
  let head = expr_head_name(&lists[0]);
  if head.is_none() {
    // First argument is atomic — apply f directly.
    return apply_func_to_n_args(func, lists);
  }
  let head = head.unwrap();
  for arg in &lists[1..] {
    match expr_head_name(arg) {
      Some(h) if h == head => {}
      _ => {
        // Mismatched or atomic — return unevaluated.
        let mut call_args = vec![func.clone()];
        call_args.extend(lists.iter().cloned());
        return Ok(Expr::FunctionCall {
          name: "Outer".to_string(),
          args: call_args.into(),
        });
      }
    }
  }

  if levels.is_empty() {
    outer_impl(func, &lists[0], &lists[1..], &[], &head)
  } else {
    // Build per-list level specs
    let per_list_levels: Vec<usize> = if levels.len() == 1 {
      // Single level spec applies to all lists
      vec![levels[0]; lists.len()]
    } else {
      // One level per list
      let mut v = levels.to_vec();
      while v.len() < lists.len() {
        v.push(*v.last().unwrap_or(&1));
      }
      v
    };
    outer_impl_leveled(func, lists, 0, &per_list_levels, &[], &head)
  }
}

/// Return the head name for an expression that can be threaded by Outer.
fn expr_head_name(expr: &Expr) -> Option<String> {
  match expr {
    Expr::List(_) => Some("List".to_string()),
    Expr::FunctionCall { name, .. } => Some(name.clone()),
    _ => None,
  }
}

/// Get child items from a List or FunctionCall.
fn expr_items(expr: &Expr) -> Option<&[Expr]> {
  match expr {
    Expr::List(items) => Some(items.as_slice()),
    Expr::FunctionCall { args, .. } => Some(args.as_slice()),
    _ => None,
  }
}

/// Wrap items in the given head.
fn wrap_in_head(head: &str, items: Vec<Expr>) -> Expr {
  if head == "List" {
    Expr::List(items.into())
  } else {
    Expr::FunctionCall {
      name: head.to_string(),
      args: items.into(),
    }
  }
}

fn outer_impl(
  func: &Expr,
  current: &Expr,
  remaining: &[Expr],
  accumulated: &[Expr],
  head: &str,
) -> Result<Expr, InterpreterError> {
  if expr_head_name(current).as_deref() == Some(head)
    && let Some(items) = expr_items(current)
  {
    // Thread through child elements, wrapping the result in the common head.
    // Only descend into sub-expressions sharing the common head: a Rational,
    // Complex, or other non-matching FunctionCall is an atom, not a level.
    let mut results = Vec::new();
    for item in items {
      results.push(outer_impl(func, item, remaining, accumulated, head)?);
    }
    Ok(wrap_in_head(head, results))
  } else {
    // Atomic element: add to accumulated args
    let mut new_acc = accumulated.to_vec();
    new_acc.push(current.clone());
    if remaining.is_empty() {
      // All lists consumed: apply f to accumulated args
      apply_func_to_n_args(func, &new_acc)
    } else {
      // Move to next list
      outer_impl(func, &remaining[0], &remaining[1..], &new_acc, head)
    }
  }
}

/// Level-aware Outer: descend only `levels[list_idx]` levels into each list.
fn outer_impl_leveled(
  func: &Expr,
  lists: &[Expr],
  list_idx: usize,
  levels: &[usize],
  accumulated: &[Expr],
  head: &str,
) -> Result<Expr, InterpreterError> {
  if list_idx >= lists.len() {
    // All lists consumed: apply f to accumulated args
    return apply_func_to_n_args(func, accumulated);
  }
  let depth = levels[list_idx];
  outer_descend(
    func,
    &lists[list_idx],
    lists,
    list_idx,
    levels,
    accumulated,
    head,
    depth,
  )
}

fn outer_descend(
  func: &Expr,
  current: &Expr,
  lists: &[Expr],
  list_idx: usize,
  levels: &[usize],
  accumulated: &[Expr],
  head: &str,
  depth_remaining: usize,
) -> Result<Expr, InterpreterError> {
  if depth_remaining > 0
    && expr_head_name(current).as_deref() == Some(head)
    && let Some(items) = expr_items(current)
  {
    let mut results = Vec::new();
    for item in items {
      results.push(outer_descend(
        func,
        item,
        lists,
        list_idx,
        levels,
        accumulated,
        head,
        depth_remaining - 1,
      )?);
    }
    return Ok(wrap_in_head(head, results));
  }
  // Reached target depth or hit an atom: treat as a single element
  let mut new_acc = accumulated.to_vec();
  new_acc.push(current.clone());
  outer_impl_leveled(func, lists, list_idx + 1, levels, &new_acc, head)
}

/// Inner[f, list1, list2, g] - generalized inner product
pub fn inner_ast(
  f: &Expr,
  list1: &Expr,
  list2: &Expr,
  g: &Expr,
) -> Result<Expr, InterpreterError> {
  match inner_recursive(f, list1, list2, g) {
    Err(InterpreterError::EvaluationError(msg))
      if msg.contains("incompatible dimensions") =>
    {
      let l1_len = match list1 {
        Expr::List(items) => items.len(),
        _ => 0,
      };
      let l2_len = match list2 {
        Expr::List(items) => items.len(),
        _ => 0,
      };
      crate::emit_message(&format!(
        "Inner::incom: Length {} of dimension 1 in {} is incommensurate with length {} of dimension 1 in {}.",
        l1_len,
        crate::syntax::expr_to_string(list1),
        l2_len,
        crate::syntax::expr_to_string(list2),
      ));
      Ok(Expr::FunctionCall {
        name: "Inner".to_string(),
        args: vec![f.clone(), list1.clone(), list2.clone(), g.clone()].into(),
      })
    }
    other => other,
  }
}

fn inner_recursive(
  f: &Expr,
  a: &Expr,
  b: &Expr,
  g: &Expr,
) -> Result<Expr, InterpreterError> {
  let a_depth = list_depth(a);
  let b_depth = list_depth(b);

  if a_depth == 1 && b_depth == 1 {
    // Base case: both are flat lists - do pairwise f then combine with g
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let items_b = match b {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    if items_a.len() != items_b.len() {
      return Err(InterpreterError::EvaluationError(
        "Inner: incompatible dimensions".into(),
      ));
    }
    let mut products = Vec::new();
    for (x, y) in items_a.iter().zip(items_b.iter()) {
      let val = apply_func_to_two_args(f, x, y)?;
      products.push(val);
    }
    apply_func_to_n_args(g, &products)
  } else if a_depth > 1 {
    // Map over the first dimension of a
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let mut results = Vec::new();
    for row in items_a {
      results.push(inner_recursive(f, row, b, g)?);
    }
    Ok(Expr::List(results.into()))
  } else {
    // a_depth == 1, b_depth > 1
    // Contract a (vector) with first dimension of b
    // Need to iterate over the inner structure of b
    let items_a = match a {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    let items_b = match b {
      Expr::List(items) => items,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner expects lists as arguments".into(),
        ));
      }
    };
    if items_a.len() != items_b.len() {
      return Err(InterpreterError::EvaluationError(
        "Inner: incompatible dimensions".into(),
      ));
    }
    // Each items_b[k] should be a list. Extract their elements column-wise.
    let inner_len = match &items_b[0] {
      Expr::List(inner) => inner.len(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Inner: incompatible dimensions".into(),
        ));
      }
    };
    let mut results = Vec::new();
    for j in 0..inner_len {
      // Build a "column" vector from b: [b[0][j], b[1][j], ...]
      let mut col = Vec::new();
      for bk in items_b {
        match bk {
          Expr::List(inner) if j < inner.len() => {
            col.push(inner[j].clone());
          }
          _ => {
            return Err(InterpreterError::EvaluationError(
              "Inner: incompatible dimensions".into(),
            ));
          }
        }
      }
      let col_expr = Expr::List(col.into());
      results.push(inner_recursive(f, a, &col_expr, g)?);
    }
    Ok(Expr::List(results.into()))
  }
}

/// Full rectangular dimensions of a nested-list array, or `None` if the
/// expression is ragged or not a list. A 1-D vector returns `[n]`.
fn full_array_dims(expr: &Expr) -> Option<Vec<usize>> {
  match expr {
    Expr::List(items) => {
      if items.is_empty() {
        return Some(vec![0]);
      }
      let sub: Vec<Option<Vec<usize>>> =
        items.iter().map(full_array_dims).collect();
      // All children atoms -> 1-D vector.
      if sub.iter().all(Option::is_none) {
        return Some(vec![items.len()]);
      }
      // Otherwise every child must be a list with identical dimensions.
      let first = sub[0].clone()?;
      if sub.iter().all(|s| s.as_ref() == Some(&first)) {
        let mut dims = vec![items.len()];
        dims.extend(first);
        Some(dims)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Fetch the element of a nested-list array at a full 0-indexed multi-index.
fn array_elem_at(array: &Expr, index: &[usize]) -> Option<Expr> {
  let mut cur = array;
  for &i in index {
    match cur {
      Expr::List(items) => cur = items.get(i)?,
      _ => return None,
    }
  }
  Some(cur.clone())
}

/// Row-major Cartesian product of the given axis sizes. Empty `sizes` yields a
/// single empty index (the scalar case).
fn cartesian_indices(sizes: &[usize]) -> Vec<Vec<usize>> {
  let mut result = vec![Vec::new()];
  for &s in sizes {
    let mut next = Vec::with_capacity(result.len() * s);
    for prefix in &result {
      for i in 0..s {
        let mut p = prefix.clone();
        p.push(i);
        next.push(p);
      }
    }
    result = next;
  }
  result
}

/// Reshape a flat list into a nested list of the given shape. Empty shape
/// returns the single scalar element.
fn reshape_flat(flat: &[Expr], shape: &[usize]) -> Expr {
  if shape.is_empty() {
    return flat[0].clone();
  }
  let chunk: usize = shape[1..].iter().product();
  let mut out = Vec::with_capacity(shape[0]);
  for i in 0..shape[0] {
    out.push(reshape_flat(&flat[i * chunk..(i + 1) * chunk], &shape[1..]));
  }
  Expr::List(out.into())
}

/// ArrayReduce[f, array, levels] reduces a full rectangular array over the
/// dimension(s) given by `levels`, applying `f` to the flat list of elements
/// gathered along those dimensions (row-major order). The result is indexed by
/// the remaining dimensions, in ascending order.
///
///   ArrayReduce[Total, {{1, 2}, {3, 4}}, 1]      -> {4, 6}
///   ArrayReduce[Total, {{1, 2}, {3, 4}}, 2]      -> {3, 7}
///   ArrayReduce[Total, {{1, 2}, {3, 4}}, {1, 2}] -> 10
pub fn array_reduce_ast(
  func: &Expr,
  array: &Expr,
  levels: &Expr,
) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ArrayReduce".to_string(),
      args: vec![func.clone(), array.clone(), levels.clone()].into(),
    })
  };

  let dims = match full_array_dims(array) {
    Some(d) if !d.is_empty() && !d.contains(&0) => d,
    _ => return unevaluated(),
  };
  let rank = dims.len();

  // Parse the level spec into a list of raw 1-indexed dims (any integer).
  let raw_levels: Vec<i128> = match levels {
    Expr::Integer(n) => vec![*n],
    Expr::List(items) => {
      let mut v = Vec::with_capacity(items.len());
      for it in items {
        match it {
          Expr::Integer(n) => v.push(*n),
          _ => return unevaluated(),
        }
      }
      v
    }
    _ => return unevaluated(),
  };
  // Validate: non-positive -> arlowlev, beyond the array depth -> arhighlev.
  // Match wolframscript's message and the unevaluated return.
  for &l in &raw_levels {
    if l < 1 {
      crate::emit_message(&format!(
        "ArrayReduce::arlowlev: Level specification {} should be positive.",
        l
      ));
      return unevaluated();
    }
    if l as usize > rank {
      crate::emit_message(&format!(
        "ArrayReduce::arhighlev: Level specification {} is higher than array depth.",
        l
      ));
      return unevaluated();
    }
  }
  let mut level_set: Vec<usize> =
    raw_levels.iter().map(|&l| l as usize).collect();
  level_set.sort_unstable();
  level_set.dedup();

  let remaining: Vec<usize> =
    (1..=rank).filter(|d| !level_set.contains(d)).collect();
  let remaining_sizes: Vec<usize> =
    remaining.iter().map(|&d| dims[d - 1]).collect();
  let level_sizes: Vec<usize> =
    level_set.iter().map(|&d| dims[d - 1]).collect();

  let remaining_indices = cartesian_indices(&remaining_sizes);
  let level_indices = cartesian_indices(&level_sizes);

  let mut flat_results = Vec::with_capacity(remaining_indices.len());
  for r_idx in &remaining_indices {
    let mut gathered = Vec::with_capacity(level_indices.len());
    for l_idx in &level_indices {
      let mut full = vec![0usize; rank];
      for (pos, &dim) in remaining.iter().enumerate() {
        full[dim - 1] = r_idx[pos];
      }
      for (pos, &dim) in level_set.iter().enumerate() {
        full[dim - 1] = l_idx[pos];
      }
      match array_elem_at(array, &full) {
        Some(e) => gathered.push(e),
        None => return unevaluated(),
      }
    }
    let applied = apply_func_ast(func, &Expr::List(gathered.into()))?;
    flat_results.push(applied);
  }

  Ok(reshape_flat(&flat_results, &remaining_sizes))
}

/// Smallest period `p` in `[1, len]` such that element `i` equals element
/// `i % p` for every `i`, and the pattern fits at least `min_reps` whole times
/// (`len / p >= min_reps`). Returns `None` when no such period exists.
fn find_repeat_period(strs: &[String], min_reps: i128) -> Option<usize> {
  let len = strs.len();
  for p in 1..=len {
    let tiles = (0..len).all(|i| strs[i] == strs[i % p]);
    if tiles && (len / p) as i128 >= min_reps {
      return Some(p);
    }
  }
  None
}

/// FindRepeat[seq] / FindRepeat[seq, n] finds the shortest sub-sequence that,
/// tiled, reproduces `seq` (a partial final repetition is allowed). The
/// optional `n` requires the pattern to repeat at least `n` whole times;
/// when no such period exists the empty sequence is returned. Works on lists,
/// strings, and associations (over their values, keeping the keys).
///
///   FindRepeat[{1, 2, 3, 1, 2, 3}]    -> {1, 2, 3}
///   FindRepeat[{1, 2, 3, 4, 5}]       -> {1, 2, 3, 4, 5}
///   FindRepeat[{1, 2, 3, 1, 2, 3}, 3] -> {}
///   FindRepeat["abcabc"]              -> "abc"
pub fn find_repeat_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let arg = &args[0];
  let min_reps: i128 = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(n) if n >= 1 => n,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FindRepeat".to_string(),
          args: args.to_vec().into(),
        });
      }
    }
  } else {
    1
  };

  match arg {
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::List(vec![].into()));
      }
      let strs: Vec<String> =
        items.iter().map(crate::syntax::expr_to_string).collect();
      match find_repeat_period(&strs, min_reps) {
        Some(p) => Ok(Expr::List(items[..p].to_vec().into())),
        None => Ok(Expr::List(vec![].into())),
      }
    }
    Expr::String(s) => {
      let chars: Vec<char> = s.chars().collect();
      if chars.is_empty() {
        return Ok(Expr::String(String::new()));
      }
      let strs: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
      match find_repeat_period(&strs, min_reps) {
        Some(p) => Ok(Expr::String(chars[..p].iter().collect())),
        None => Ok(Expr::String(String::new())),
      }
    }
    Expr::Association(pairs) => {
      if pairs.is_empty() {
        return Ok(Expr::Association(vec![]));
      }
      let strs: Vec<String> = pairs
        .iter()
        .map(|(_, v)| crate::syntax::expr_to_string(v))
        .collect();
      match find_repeat_period(&strs, min_reps) {
        Some(p) => Ok(Expr::Association(pairs[..p].to_vec())),
        None => Ok(Expr::Association(vec![])),
      }
    }
    _ => {
      crate::emit_message(&format!(
        "FindRepeat::arg1: The first argument {} to FindRepeat is expected to be a list, an association or a string.",
        crate::syntax::expr_to_string(arg)
      ));
      Ok(Expr::FunctionCall {
        name: "FindRepeat".to_string(),
        args: args.to_vec().into(),
      })
    }
  }
}

/// Split `strs` into `(transient_len, period)` for FindTransientRepeat: `period`
/// is the length of the shortest end-cycle that repeats at least `n` whole
/// times; the choice minimizes the transient length, tie-breaking on the
/// smallest period. `period == 0` means no cycle repeats `n` times (the whole
/// sequence is the transient).
fn transient_repeat_split(strs: &[String], n: i128) -> (usize, usize) {
  let len = strs.len();
  let mut best: Option<(usize, usize)> = None; // (transient_len, period)
  for p in 1..=len {
    // Count how many whole copies of the final p-length block tile the end.
    let mut k = 1usize;
    while (k + 1) * p <= len {
      let blk = len - (k + 1) * p;
      let c = len - p;
      if (0..p).all(|j| strs[blk + j] == strs[c + j]) {
        k += 1;
      } else {
        break;
      }
    }
    if k as i128 >= n {
      let cand = (len - k * p, p);
      if best.is_none_or(|b| cand < b) {
        best = Some(cand);
      }
    }
  }
  best.unwrap_or((len, 0))
}

/// FindTransientRepeat[seq, n] splits `seq` into `{transient, repeat}` where the
/// repeat is the shortest end-cycle occurring at least `n` times. Works on
/// lists, strings, and associations; both parts keep the input head.
pub fn find_transient_repeat_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Expr::FunctionCall {
    name: "FindTransientRepeat".to_string(),
    args: args.to_vec().into(),
  };
  // The second argument must be a positive integer.
  let n: i128 = match expr_to_i128(&args[1]) {
    Some(n) if n >= 1 => n,
    _ => {
      crate::emit_message(&format!(
        "FindTransientRepeat::intp: Positive integer expected at position 2 in {}.",
        crate::syntax::expr_to_string(&unevaluated())
      ));
      return Ok(unevaluated());
    }
  };

  match &args[0] {
    Expr::List(items) => {
      let strs: Vec<String> =
        items.iter().map(crate::syntax::expr_to_string).collect();
      let (t, p) = transient_repeat_split(&strs, n);
      Ok(Expr::List(
        vec![
          Expr::List(items[..t].to_vec().into()),
          Expr::List(items[t..t + p].to_vec().into()),
        ]
        .into(),
      ))
    }
    Expr::String(s) => {
      let chars: Vec<char> = s.chars().collect();
      let strs: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
      let (t, p) = transient_repeat_split(&strs, n);
      Ok(Expr::List(
        vec![
          Expr::String(chars[..t].iter().collect()),
          Expr::String(chars[t..t + p].iter().collect()),
        ]
        .into(),
      ))
    }
    Expr::Association(pairs) => {
      let strs: Vec<String> = pairs
        .iter()
        .map(|(_, v)| crate::syntax::expr_to_string(v))
        .collect();
      let (t, p) = transient_repeat_split(&strs, n);
      Ok(Expr::List(
        vec![
          Expr::Association(pairs[..t].to_vec()),
          Expr::Association(pairs[t..t + p].to_vec()),
        ]
        .into(),
      ))
    }
    _ => Ok(unevaluated()),
  }
}
