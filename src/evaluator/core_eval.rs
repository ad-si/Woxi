#[allow(unused_imports)]
use super::*;
use crate::syntax::{BinaryOperator, ComparisonOp};

thread_local! {
  /// Symbols currently being looked up — prevents infinite recursion when
  /// a stored OwnValue references the same symbol (e.g. `s = {a, s}`).
  static SYMBOLS_BEING_EVALUATED:
    std::cell::RefCell<std::collections::HashSet<String>> =
      std::cell::RefCell::new(std::collections::HashSet::new());
}

/// Does `expr` mention any Identifier (other than `self_name`) whose
/// current stored OwnValue might produce a different result on
/// re-evaluation? Used to skip wasteful re-evaluation of stored values
/// that are already at fixpoint.
/// Compare two operands as exact integers when both are `Integer` /
/// `BigInteger`. f64 conversion silently rounds 1-ULP differences to
/// a tie above ~2^53, so `2^60 < 2^60 + 1` would otherwise return
/// False. Returns `None` if either side isn't an exact integer, in
/// which case the f64 path takes over.
fn exact_integer_ord(a: &Expr, b: &Expr) -> Option<std::cmp::Ordering> {
  fn as_bigint(e: &Expr) -> Option<num_bigint::BigInt> {
    match e {
      Expr::Integer(n) => Some(num_bigint::BigInt::from(*n)),
      Expr::BigInteger(n) => Some(n.clone()),
      _ => None,
    }
  }
  Some(as_bigint(a)?.cmp(&as_bigint(b)?))
}

fn needs_reevaluation(expr: &Expr, self_name: &str) -> bool {
  match expr {
    Expr::Identifier(n) => {
      n != self_name && ENV.with(|e| e.borrow().contains_key(n))
    }
    Expr::List(items) => items.iter().any(|a| needs_reevaluation(a, self_name)),
    // FunctionCall / BinaryOp / UnaryOp / CurriedCall stored via SetDelayed
    // (`x := expr`) must be re-evaluated on each lookup — that is the whole
    // point of `:=`. Set (`x = expr`) evaluates the RHS first, so the stored
    // value would only be a FunctionCall when evaluation didn't reduce it
    // (e.g. `x = Sin[a]` with `a` unbound); re-evaluating those on lookup is
    // also the right behaviour (matches wolframscript: when `a` later gets a
    // value, `x` picks it up).
    Expr::FunctionCall { .. } => true,
    Expr::BinaryOp { .. } => true,
    Expr::UnaryOp { .. } => true,
    Expr::CurriedCall { .. } => true,
    _ => false,
  }
}

/// Check if a function has a specific Hold attribute (built-in or user-defined).
fn has_hold_attribute(name: &str, attr: &str) -> bool {
  get_builtin_attributes(name).contains(&attr)
    || crate::FUNC_ATTRS.with(|m| {
      m.borrow()
        .get(name)
        .is_some_and(|attrs| attrs.contains(&attr.to_string()))
    })
}

/// Evaluate function arguments respecting Hold attributes.
/// Returns the evaluated (or held) arguments based on the function's attributes.
/// Held arguments still honour `Evaluate[expr]`: it forces evaluation of the
/// wrapped expression even though the surrounding head holds. HoldAllComplete
/// is the exception — it suppresses Evaluate too.
fn evaluate_args_with_hold(
  name: &str,
  args: &[Expr],
) -> Result<Vec<Expr>, InterpreterError> {
  let hold_all_complete = has_hold_attribute(name, "HoldAllComplete");
  let hold_all = has_hold_attribute(name, "HoldAll") || hold_all_complete;
  let hold_first = has_hold_attribute(name, "HoldFirst");
  let hold_rest = has_hold_attribute(name, "HoldRest");

  let process_held = |arg: &Expr| -> Result<Expr, InterpreterError> {
    if hold_all_complete {
      Ok(arg.clone())
    } else {
      unwrap_top_level_evaluate(arg)
    }
  };

  // HoldAllComplete keeps `Sequence[…]` un-spliced (Wolfram preserves the
  // wrapper); other hold attributes still splice so multi-arg
  // `Evaluate[…]` produces a flat sequence in the surrounding context.
  let maybe_splice = |xs: Vec<Expr>| -> Vec<Expr> {
    if hold_all_complete {
      xs
    } else {
      splice_top_level_sequences(xs)
    }
  };
  if hold_all {
    let raw: Result<Vec<Expr>, _> = args.iter().map(process_held).collect();
    Ok(maybe_splice(raw?))
  } else if hold_first && !args.is_empty() {
    let mut result = vec![process_held(&args[0])?];
    for arg in &args[1..] {
      result.push(evaluate_expr_to_expr(arg)?);
    }
    Ok(maybe_splice(result))
  } else if hold_rest && !args.is_empty() {
    let mut result = vec![evaluate_expr_to_expr(&args[0])?];
    for arg in &args[1..] {
      result.push(process_held(arg)?);
    }
    Ok(maybe_splice(result))
  } else {
    args
      .iter()
      .map(evaluate_expr_to_expr)
      .collect::<Result<_, _>>()
  }
}

/// If `expr` is `Unevaluated[inner]`, return `inner`; otherwise return `expr`
/// unchanged. Used by select built-ins (Length, Sqrt, etc.) that consume
/// the Unevaluated wrapper before computing.
pub fn strip_unevaluated(expr: &Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Unevaluated"
    && args.len() == 1
  {
    return args[0].clone();
  }
  expr.clone()
}

/// If `arg` is `Evaluate[expr]`, evaluate `expr`. Otherwise return `arg`
/// unchanged. (Only the top-level Evaluate is unwrapped — nested calls
/// keep their normal evaluation rules.) Multi-argument
/// `Evaluate[a, b, …]` becomes `Sequence[a-evaluated, b-evaluated, …]`,
/// which the caller splices into the surrounding args.
fn unwrap_top_level_evaluate(arg: &Expr) -> Result<Expr, InterpreterError> {
  if let Expr::FunctionCall { name, args } = arg
    && name == "Evaluate"
  {
    if args.len() == 1 {
      return evaluate_expr_to_expr(&args[0]);
    }
    let evaluated: Result<Vec<Expr>, _> =
      args.iter().map(evaluate_expr_to_expr).collect();
    return Ok(Expr::FunctionCall {
      name: "Sequence".to_string(),
      args: evaluated?.into(),
    });
  }
  Ok(arg.clone())
}

/// Splice any top-level `Sequence[...]` elements in `args` into the list.
/// Required so that `Hold[Evaluate[1, 2]]` flattens to `Hold[1, 2]` (the
/// `Evaluate` produces a `Sequence` that gets spliced into Hold's args).
fn splice_top_level_sequences(args: Vec<Expr>) -> Vec<Expr> {
  let mut out = Vec::with_capacity(args.len());
  for a in args {
    if let Expr::FunctionCall { name, args: inner } = &a
      && name == "Sequence"
    {
      out.extend(inner.iter().cloned());
    } else {
      out.push(a);
    }
  }
  out
}

/// Helper to construct an RGBColor Expr from numeric values.
/// Uses Integer for whole numbers (0, 1) and Real for fractional values.
fn rgb_color_expr(r: f64, g: f64, b: f64) -> Expr {
  fn num(v: f64) -> Expr {
    if v == 0.0 {
      Expr::Integer(0)
    } else if v == 1.0 {
      Expr::Integer(1)
    } else {
      Expr::Real(v)
    }
  }
  Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![num(r), num(g), num(b)].into(),
  }
}

/// Helper to construct a GrayLevel Expr.
fn gray_level_expr(g: f64) -> Expr {
  fn num(v: f64) -> Expr {
    if v == 0.0 {
      Expr::Integer(0)
    } else if v == 1.0 {
      Expr::Integer(1)
    } else {
      Expr::Real(v)
    }
  }
  Expr::FunctionCall {
    name: "GrayLevel".to_string(),
    args: vec![num(g)].into(),
  }
}

/// Map named color identifiers to their Wolfram Language evaluation result.
/// Basic colors → RGBColor[r, g, b] or GrayLevel[g]
/// Light* colors → RGBColor[r, g, b] or GrayLevel[g]
pub fn named_color_expr(name: &str) -> Option<Expr> {
  Some(match name {
    // Basic colors
    "Red" => rgb_color_expr(1.0, 0.0, 0.0),
    "Green" => rgb_color_expr(0.0, 1.0, 0.0),
    "Blue" => rgb_color_expr(0.0, 0.0, 1.0),
    "Black" => gray_level_expr(0.0),
    "White" => gray_level_expr(1.0),
    "Gray" => gray_level_expr(0.5),
    "Cyan" => rgb_color_expr(0.0, 1.0, 1.0),
    "Magenta" => rgb_color_expr(1.0, 0.0, 1.0),
    "Yellow" => rgb_color_expr(1.0, 1.0, 0.0),
    "Brown" => rgb_color_expr(0.6, 0.4, 0.2),
    "Orange" => rgb_color_expr(1.0, 0.5, 0.0),
    "Pink" => rgb_color_expr(1.0, 0.5, 0.5),
    "Purple" => rgb_color_expr(0.5, 0.0, 0.5),
    // Light colors
    "LightRed" => rgb_color_expr(1.0, 0.85, 0.85),
    "LightBlue" => rgb_color_expr(0.87, 0.94, 1.0),
    "LightGreen" => rgb_color_expr(0.88, 1.0, 0.88),
    "LightGray" => gray_level_expr(0.85),
    "LightOrange" => rgb_color_expr(1.0, 0.9, 0.8),
    "LightYellow" => rgb_color_expr(1.0, 1.0, 0.85),
    "LightPurple" => rgb_color_expr(0.94, 0.88, 0.94),
    "LightCyan" => rgb_color_expr(0.9, 1.0, 1.0),
    "LightMagenta" => rgb_color_expr(1.0, 0.9, 1.0),
    "LightBrown" => rgb_color_expr(0.94, 0.91, 0.88),
    "LightPink" => rgb_color_expr(1.0, 0.925, 0.925),
    _ => return None,
  })
}

/// Find the index of a Label[tag] in a list of expressions where `tag` matches `goto_tag`.
/// Both the goto tag and each label's tag are compared after evaluation,
/// since neither Goto nor Label has HoldAll.
pub fn find_label_index(exprs: &[Expr], goto_tag: &Expr) -> Option<usize> {
  let tag_str = expr_to_string(goto_tag);
  for (i, expr) in exprs.iter().enumerate() {
    if let Expr::FunctionCall { name, args } = expr
      && name == "Label"
      && args.len() == 1
    {
      // Evaluate the label's tag to match Goto's evaluated tag
      let label_tag =
        evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());
      if expr_to_string(&label_tag) == tag_str {
        return Some(i);
      }
    }
  }
  None
}

/// Prepare arguments for iterating functions (Sum, Product, NSum).
/// The body (args[0]) is kept unevaluated to preserve the iteration variable.
/// Iterator specs (args[1..]) have their bounds evaluated but variable names preserved.
fn prepare_iterating_function_args(
  args: &[Expr],
) -> Result<Vec<Expr>, InterpreterError> {
  let mut result = Vec::new();

  // Body stays unevaluated
  result.push(args[0].clone());

  // Process iterator specs
  for arg in &args[1..] {
    if let Expr::List(items) = arg {
      if items.is_empty() {
        result.push(arg.clone());
        continue;
      }
      let mut new_items = Vec::new();
      // First element is the variable name — keep unevaluated
      new_items.push(items[0].clone());
      // Remaining elements are bounds — evaluate them
      for item in &items[1..] {
        new_items.push(evaluate_expr_to_expr(item)?);
      }
      result.push(Expr::List(new_items.into()));
    } else {
      result.push(evaluate_expr_to_expr(arg)?);
    }
  }

  Ok(result)
}

/// Early dispatch for FunctionCall in evaluate_expr_to_expr — handles held functions
/// before argument evaluation. Returns Some(result) if handled, None otherwise.
#[inline(never)]
fn evaluate_expr_to_expr_early_dispatch(
  name: &str,
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  match name {
    "Protect" | "Unprotect" | "Condition" | "MessageName" | "Attributes" => {
      return Ok(Some(evaluate_function_call_ast(name, args)?));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Save" if args.len() == 2 => {
      // HoldRest: evaluate first arg (filename), hold second (symbols)
      let filename_expr = evaluate_expr_to_expr(&args[0])?;
      return Ok(Some(evaluate_function_call_ast(
        name,
        &[filename_expr, args[1].clone()],
      )?));
    }
    "CompoundExpression" => {
      if args.is_empty() {
        return Ok(Some(Expr::Identifier("Null".to_string())));
      }
      let mut result = Expr::Identifier("Null".to_string());
      let mut start_index = 0;
      'goto_loop: loop {
        for i in start_index..args.len() {
          match evaluate_expr_to_expr(&args[i]) {
            Ok(val) => {
              // A `Return[val]` produced by Block/Module/While/For
              // short-circuits the remaining statements — matching
              // wolframscript's CompoundExpression behavior.
              if let Expr::FunctionCall { name: n, args: a } = &val
                && n == "Return"
                && a.len() == 1
              {
                return Ok(Some(val));
              }
              result = val;
            }
            Err(InterpreterError::GotoSignal(tag)) => {
              if let Some(label_idx) = find_label_index(args, &tag) {
                start_index = label_idx + 1;
                continue 'goto_loop;
              } else {
                return Err(InterpreterError::GotoSignal(tag));
              }
            }
            Err(e) => return Err(e),
          }
        }
        break;
      }
      // If the final value is a `Sequence[...]`, splice and take the last
      // spliced element — wolframscript yields `2` for
      // `a; Sequence[1, 2]`, not the whole Sequence. Empty sequences
      // collapse to Null.
      if let Expr::FunctionCall {
        name: n,
        args: seq_args,
      } = &result
        && n == "Sequence"
      {
        result = seq_args
          .last()
          .cloned()
          .unwrap_or_else(|| Expr::Identifier("Null".to_string()));
      }
      return Ok(Some(result));
    }
    "Pattern" if args.len() == 2 => {
      return Ok(Some(evaluate_pattern_function_ast(args)?));
    }
    "RuleDelayed" if args.len() == 2 => {
      return Ok(Some(evaluate_rule_delayed_ast(args)?));
    }
    "AbsoluteTiming" | "Timing" if args.len() == 1 => {
      let start = web_time::Instant::now();
      let result = evaluate_expr_to_expr(&args[0])?;
      let elapsed = start.elapsed().as_secs_f64();
      return Ok(Some(Expr::List(vec![Expr::Real(elapsed), result].into())));
    }
    "RepeatedTiming" if args.len() == 1 => {
      let mut times = Vec::new();
      let mut last_result = Expr::Identifier("Null".to_string());
      let overall_start = web_time::Instant::now();
      for _ in 0..100 {
        let start = web_time::Instant::now();
        last_result = evaluate_expr_to_expr(&args[0])?;
        times.push(start.elapsed().as_secs_f64());
        if times.len() >= 3 && overall_start.elapsed().as_secs_f64() > 0.5 {
          break;
        }
      }
      times.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let median = times[times.len() / 2];
      return Ok(Some(Expr::List(
        vec![Expr::Real(median), last_result].into(),
      )));
    }
    "Sum" | "ParallelSum" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      return Ok(Some(crate::functions::list_helpers_ast::sum_ast(
        &prepared,
      )?));
    }
    "Product" | "ParallelProduct" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      return Ok(Some(crate::functions::list_helpers_ast::product_ast(
        &prepared,
      )?));
    }
    "NSum" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      return Ok(Some(crate::functions::math_ast::nsum_ast(&prepared)?));
    }
    "NProduct" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      return Ok(Some(crate::functions::math_ast::nproduct_ast(&prepared)?));
    }
    _ => {}
  }
  Ok(None)
}

/// Whether an expression is a valid rules slot for ReplaceAll /
/// ReplaceRepeated: a single Rule/RuleDelayed, or a (possibly nested)
/// list of such rules. Anything else triggers ReplaceAll::reps in
/// Wolfram.
fn is_valid_replace_rules(expr: &Expr) -> bool {
  match expr {
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => true,
    Expr::FunctionCall { name, .. }
      if name == "Rule" || name == "RuleDelayed" =>
    {
      true
    }
    Expr::List(items) => items.iter().all(is_valid_replace_rules),
    _ => false,
  }
}

/// Evaluate an Expr AST and return a new Expr (not a string).
/// This is the core function for AST-based evaluation without string round-trips.
pub fn evaluate_expr_to_expr(expr: &Expr) -> Result<Expr, InterpreterError> {
  // Dynamically grow the stack when running low. This prevents stack
  // overflows in debug builds where the massive evaluate_function_call_ast_inner
  // function uses large stack frames.  The 2 MB red zone triggers a 4 MB
  // growth so that even deep user-defined recursion (e.g. naive Fibonacci,
  // n-queens) succeeds regardless of the initial thread stack size.
  stacker::maybe_grow(2 * 1024 * 1024, 4 * 1024 * 1024, || {
    evaluate_expr_to_expr_impl(expr)
  })
}

fn evaluate_expr_to_expr_impl(expr: &Expr) -> Result<Expr, InterpreterError> {
  // $RecursionLimit enforcement: prevent infinite recursion from
  // user-defined functions.  When the limit is exceeded, return the
  // expression unevaluated (matching Wolfram behavior).
  let depth = crate::RECURSION_DEPTH.with(|d| {
    let cur = d.get();
    d.set(cur + 1);
    cur
  });
  struct DepthGuard;
  impl Drop for DepthGuard {
    fn drop(&mut self) {
      crate::RECURSION_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
  }
  let _guard = DepthGuard;

  const RECURSION_LIMIT: usize = 1024;
  if depth > RECURSION_LIMIT {
    return Ok(expr.clone());
  }

  // Trampoline loop: tail-recursive calls return TailCall instead of
  // recursing, so the call stack stays flat regardless of recursion depth.
  // The vast majority of evaluations don't return TailCall, so avoid the
  // up-front Expr::clone() and only allocate the owned `current` once a
  // TailCall is actually observed.
  match evaluate_expr_to_expr_inner(expr) {
    Err(InterpreterError::TailCall(next)) => {
      let mut current = *next;
      loop {
        match evaluate_expr_to_expr_inner(&current) {
          Err(InterpreterError::TailCall(next)) => current = *next,
          result => return result,
        }
      }
    }
    result => result,
  }
}

pub fn evaluate_expr_to_expr_inner(
  expr: &Expr,
) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(Expr::Integer(*n)),
    Expr::BigInteger(n) => Ok(Expr::BigInteger(n.clone())),
    Expr::Real(f) => Ok(Expr::Real(*f)),
    Expr::BigFloat(d, p) => Ok(Expr::BigFloat(d.clone(), *p)),
    Expr::String(s) => Ok(Expr::String(s.clone())),
    Expr::Identifier(name) => {
      // Look up in environment
      if let Some(stored) = ENV.with(|e| e.borrow().get(name).cloned()) {
        match stored {
          // Re-evaluate the stored value so that later changes to free
          // symbols propagate, matching Wolfram's OwnValues semantics
          // (`z = 1 + x; x = 2; z` returns 3). Only re-evaluate when the
          // stored expression mentions a free Identifier that itself has
          // a stored value — otherwise the stored value is already at
          // fixpoint and re-evaluating is wasted work (plus risks
          // quadratic blowup for growing lists like `k = {k, 1}`).
          StoredValue::ExprVal(e) => {
            // `a /; cond := body` stored the OwnValue as
            // `Condition[body, cond]` so the guard can be re-checked at
            // lookup time. Evaluate the test; if it isn't True the rule
            // doesn't fire and the symbol returns as itself.
            if let Expr::FunctionCall {
              name: cond_name,
              args: cond_args,
            } = &e
              && cond_name == "Condition"
              && cond_args.len() == 2
            {
              let test_eval = evaluate_expr_to_expr(&cond_args[1]);
              let passes = matches!(
                test_eval,
                Ok(Expr::Identifier(ref s)) if s == "True"
              );
              if !passes {
                return Ok(Expr::Identifier(name.clone()));
              }
              return evaluate_expr_to_expr(&cond_args[0]);
            }
            if matches!(&e, Expr::Identifier(s) if s == name) {
              return Ok(e);
            }
            if !needs_reevaluation(&e, name) {
              return Ok(e);
            }
            let already =
              SYMBOLS_BEING_EVALUATED.with(|s| s.borrow().contains(name));
            if already {
              return Ok(e);
            }
            SYMBOLS_BEING_EVALUATED
              .with(|s| s.borrow_mut().insert(name.clone()));
            let result = evaluate_expr_to_expr(&e);
            SYMBOLS_BEING_EVALUATED.with(|s| s.borrow_mut().remove(name));
            result
          }
          StoredValue::Raw(val) => {
            // Parse the stored value back to Expr. If the parsed result is
            // itself a free symbol bound to another value (e.g. `a = b;
            // b = 4` stored a as Raw("b")), recursively evaluate so the
            // chain resolves all the way through.
            let parsed = string_to_expr(&val)?;
            if matches!(&parsed, Expr::Identifier(s) if s == name) {
              return Ok(parsed);
            }
            if !needs_reevaluation(&parsed, name) {
              return Ok(parsed);
            }
            let already =
              SYMBOLS_BEING_EVALUATED.with(|s| s.borrow().contains(name));
            if already {
              return Ok(parsed);
            }
            SYMBOLS_BEING_EVALUATED
              .with(|s| s.borrow_mut().insert(name.clone()));
            let result = evaluate_expr_to_expr(&parsed);
            SYMBOLS_BEING_EVALUATED.with(|s| s.borrow_mut().remove(name));
            result
          }
          StoredValue::Association(items) => {
            // Convert stored association back to Expr::Association
            let expr_items: Vec<(Expr, Expr)> = items
              .into_iter()
              .map(|(k, v)| {
                let key_expr = string_to_expr(&k).unwrap_or(Expr::String(k));
                let val_expr = string_to_expr(&v).unwrap_or(Expr::String(v));
                (key_expr, val_expr)
              })
              .collect();
            Ok(Expr::Association(expr_items))
          }
        }
      } else {
        #[cfg(not(target_arch = "wasm32"))]
        if name == "Now" {
          use chrono::Local;
          let now = Local::now();
          let seconds = now
            .format("%S%.f")
            .to_string()
            .parse::<f64>()
            .unwrap_or(0.0);
          let tz_offset_hours = now.offset().local_minus_utc() as f64 / 3600.0;
          return Ok(Expr::FunctionCall {
            name: "DateObject".to_string(),
            args: vec![
              Expr::List(
                vec![
                  Expr::Integer(
                    now.format("%Y").to_string().parse::<i128>().unwrap(),
                  ),
                  Expr::Integer(
                    now.format("%m").to_string().parse::<i128>().unwrap(),
                  ),
                  Expr::Integer(
                    now.format("%d").to_string().parse::<i128>().unwrap(),
                  ),
                  Expr::Integer(
                    now.format("%H").to_string().parse::<i128>().unwrap(),
                  ),
                  Expr::Integer(
                    now.format("%M").to_string().parse::<i128>().unwrap(),
                  ),
                  Expr::Real(seconds),
                ]
                .into(),
              ),
              Expr::String("Instant".to_string()),
              Expr::String("Gregorian".to_string()),
              Expr::Real(tz_offset_hours),
            ]
            .into(),
          });
        }
        #[cfg(target_arch = "wasm32")]
        if name == "Now" {
          let now = js_sys::Date::new_0();
          let seconds =
            now.get_seconds() as f64 + now.get_milliseconds() as f64 / 1000.0;
          let tz_offset_hours = -(now.get_timezone_offset() / 60.0);
          return Ok(Expr::FunctionCall {
            name: "DateObject".to_string(),
            args: vec![
              Expr::List(
                vec![
                  Expr::Integer(now.get_full_year() as i128),
                  Expr::Integer((now.get_month() + 1) as i128),
                  Expr::Integer(now.get_date() as i128),
                  Expr::Integer(now.get_hours() as i128),
                  Expr::Integer(now.get_minutes() as i128),
                  Expr::Real(seconds),
                ]
                .into(),
              ),
              Expr::String("Instant".to_string()),
              Expr::String("Gregorian".to_string()),
              Expr::Real(tz_offset_hours),
            ]
            .into(),
          });
        }
        // Handle Today/Tomorrow/Yesterday → DateObject[{y, m, d}, Day]
        #[cfg(not(target_arch = "wasm32"))]
        if name == "Today" || name == "Tomorrow" || name == "Yesterday" {
          use chrono::{Duration, Local};
          let now = Local::now();
          let date = match name.as_str() {
            "Tomorrow" => now + Duration::days(1),
            "Yesterday" => now - Duration::days(1),
            _ => now,
          };
          return Ok(Expr::FunctionCall {
            name: "DateObject".to_string(),
            args: vec![
              Expr::List(
                vec![
                  Expr::Integer(
                    date.format("%Y").to_string().parse::<i128>().unwrap(),
                  ),
                  Expr::Integer(
                    date.format("%m").to_string().parse::<i128>().unwrap(),
                  ),
                  Expr::Integer(
                    date.format("%d").to_string().parse::<i128>().unwrap(),
                  ),
                ]
                .into(),
              ),
              Expr::String("Day".to_string()),
            ]
            .into(),
          });
        }
        #[cfg(target_arch = "wasm32")]
        if name == "Today" || name == "Tomorrow" || name == "Yesterday" {
          let now = js_sys::Date::new_0();
          let offset_days: i32 = match name.as_str() {
            "Tomorrow" => 1,
            "Yesterday" => -1,
            _ => 0,
          };
          let ms = now.get_time() + (offset_days as f64) * 86_400_000.0;
          let d = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms));
          return Ok(Expr::FunctionCall {
            name: "DateObject".to_string(),
            args: vec![
              Expr::List(
                vec![
                  Expr::Integer(d.get_full_year() as i128),
                  Expr::Integer((d.get_month() + 1) as i128),
                  Expr::Integer(d.get_date() as i128),
                ]
                .into(),
              ),
              Expr::String("Day".to_string()),
            ]
            .into(),
          });
        }
        // Handle system $ variables (both unqualified `$Foo` and
        // System-context-qualified `System`$Foo` forms).
        if name.starts_with('$')
          && let Some(val) = get_system_variable(name)
        {
          return Ok(val);
        }
        if let Some(short) = name.strip_prefix("System`")
          && short.starts_with('$')
          && let Some(val) = get_system_variable(short)
        {
          return Ok(val);
        }
        // Handle named colors (Red → RGBColor[1, 0, 0], etc.)
        if let Some(color_expr) = named_color_expr(name) {
          return Ok(color_expr);
        }
        // Thick → Thickness[Large]
        if name == "Thick" {
          return Ok(Expr::FunctionCall {
            name: "Thickness".to_string(),
            args: vec![Expr::Identifier("Large".to_string())].into(),
          });
        }
        // Thin → Thickness[Tiny]
        if name == "Thin" {
          return Ok(Expr::FunctionCall {
            name: "Thickness".to_string(),
            args: vec![Expr::Identifier("Tiny".to_string())].into(),
          });
        }
        // Dashed → Dashing[{Small, Small}]
        if name == "Dashed" {
          return Ok(Expr::FunctionCall {
            name: "Dashing".to_string(),
            args: vec![Expr::List(
              vec![
                Expr::Identifier("Small".to_string()),
                Expr::Identifier("Small".to_string()),
              ]
              .into(),
            )]
            .into(),
          });
        }
        // Dotted → Dashing[{0, Small}]
        if name == "Dotted" {
          return Ok(Expr::FunctionCall {
            name: "Dashing".to_string(),
            args: vec![Expr::List(
              vec![Expr::Integer(0), Expr::Identifier("Small".to_string())]
                .into(),
            )]
            .into(),
          });
        }
        // DotDashed → Dashing[{0, Small, Small, Small}]
        if name == "DotDashed" {
          return Ok(Expr::FunctionCall {
            name: "Dashing".to_string(),
            args: vec![Expr::List(
              vec![
                Expr::Integer(0),
                Expr::Identifier("Small".to_string()),
                Expr::Identifier("Small".to_string()),
                Expr::Identifier("Small".to_string()),
              ]
              .into(),
            )]
            .into(),
          });
        }
        // Return as symbolic identifier
        Ok(Expr::Identifier(name.clone()))
      }
    }
    Expr::Slot(n) => {
      // Slots should be replaced before evaluation
      Ok(Expr::Slot(*n))
    }
    Expr::SlotSequence(n) => Ok(Expr::SlotSequence(*n)),
    Expr::Constant(name) => {
      // After `Unprotect[c]; c = value`, the user has reassigned the
      // built-in constant. Honour the new OwnValue if one was stored —
      // matching wolframscript's `Unprotect[Pi]; Pi = 3; Pi → 3`.
      if let Some(stored) = ENV.with(|e| e.borrow().get(name).cloned()) {
        match stored {
          StoredValue::Raw(val) => return string_to_expr(&val),
          StoredValue::ExprVal(e) => return Ok(e),
          _ => {}
        }
      }
      Ok(Expr::Constant(name.clone()))
    }
    Expr::List(items) => {
      let mut evaluated: Vec<Expr> = Vec::with_capacity(items.len());
      // Snapshot FUNC_DEFS state before evaluation so we can detect
      // whether any list element added or removed DownValues. Wolfram's
      // evaluator iterates each expression to a fixed point; if a later
      // element introduces a rule, earlier elements that didn't match
      // before are re-evaluated against the new rule.
      //
      // The key insight: we only need to detect *whether* the rule set
      // changed, not collect the names. Comparing FUNC_DEFS length
      // before/after is O(1); only when the length differs do we need
      // the full re-evaluation pass. Length-equal-but-different (a
      // remove + insert of the same name) doesn't happen via normal
      // element evaluation, so the cheap check is sufficient.
      let len_before = crate::FUNC_DEFS.with(|m| m.borrow().len());
      for item in items {
        let val = evaluate_expr_to_expr(item)?;
        if matches!(&val, Expr::Identifier(s) if s == "Nothing") {
          continue;
        }
        // Flatten Sequence in lists
        if let Expr::FunctionCall { name, args } = &val
          && name == "Sequence"
        {
          evaluated.extend(args.iter().cloned());
          continue;
        }
        // Flatten Splice in lists: Splice[{...}] (1-arg, default List head)
        // or Splice[{...}, List] (explicit List head)
        if let Expr::FunctionCall { name, args } = &val
          && name == "Splice"
          && (args.len() == 1
            || (args.len() == 2
              && matches!(&args[1], Expr::Identifier(h) if h == "List")))
          && let Expr::List(splice_items) = &args[0]
        {
          evaluated.extend(splice_items.iter().cloned());
          continue;
        }
        evaluated.push(val);
      }
      // If any new DownValues were introduced during element evaluation,
      // re-evaluate the elements once so that earlier items pick up
      // rules defined by later items (matches wolframscript:
      // `{F[a, b], F[x__]:=H[x]; F[a,b], F=.; F[a,b]}` →
      // `{H[a, b], H[a, b], H[a, b]}`). Limited to one re-evaluation pass
      // to avoid infinite loops; deeper fixed-point semantics would need
      // pervasive iteration in the evaluator itself.
      let len_after = crate::FUNC_DEFS.with(|m| m.borrow().len());
      if len_after != len_before {
        let mut re_eval: Vec<Expr> = Vec::with_capacity(evaluated.len());
        for item in &evaluated {
          re_eval.push(evaluate_expr_to_expr(item)?);
        }
        return Ok(Expr::List(re_eval.into()));
      }
      Ok(Expr::List(evaluated.into()))
    }
    Expr::FunctionCall { name, args } => {
      // Track function calls on the evaluation stack for stack traces.
      crate::push_eval_stack(name);
      let result: Result<Expr, InterpreterError> = (|| {
        // Special handling for If - lazy evaluation of branches
        if name == "If" && (args.len() == 2 || args.len() == 3) {
          let cond = evaluate_expr_to_expr(&args[0])?;
          if matches!(&cond, Expr::Identifier(s) if s == "True") {
            return Err(InterpreterError::TailCall(Box::new(args[1].clone())));
          } else if matches!(&cond, Expr::Identifier(s) if s == "False") {
            if args.len() == 3 {
              return Err(InterpreterError::TailCall(Box::new(
                args[2].clone(),
              )));
            } else {
              return Ok(Expr::Identifier("Null".to_string()));
            }
          } else {
            // Condition didn't evaluate to True/False - return unevaluated
            let mut new_args = vec![cond];
            for arg in args.iter().skip(1) {
              new_args.push(arg.clone());
            }
            return Ok(Expr::FunctionCall {
              name: name.clone(),
              args: new_args.into(),
            });
          }
        }
        // Special handling for Module/Block/Assuming - don't evaluate args (body needs local bindings first)
        if name == "Module" {
          return module_ast(args);
        }
        if name == "Block" {
          return block_ast(args);
        }
        if name == "Assuming" && args.len() == 2 {
          return assuming_ast(args);
        }
        // Special handling for Set - first arg must be identifier or Part, second gets evaluated
        if name == "Set" && args.len() == 2 {
          return set_ast(&args[0], &args[1]);
        }
        // Special handling for SetDelayed - stores function definitions
        if name == "SetDelayed" && args.len() == 2 {
          // Before applying the SetDelayed normally, check if any
          // user-defined SetDelayed rule (typically an UpValue installed
          // via `(F[x_] := s_) ^:= Q[x, s]`) matches the call. This is
          // what lets `F[1] := 2` evaluate to `Q[1, 2]` for case 4709 —
          // Wolfram's UpValues-on-SetDelayed mechanism.
          let has_setdelayed_rules = crate::FUNC_DEFS.with(|m| {
            m.borrow().get("SetDelayed").is_some_and(|v| !v.is_empty())
          });
          if has_setdelayed_rules {
            let result =
              crate::evaluator::dispatch::evaluate_function_call_ast(
                "SetDelayed",
                args,
              )?;
            // If a user-defined rule matched, the result will differ from
            // the original `SetDelayed[…]` shell. Otherwise fall through
            // to the built-in handler. `Expr` doesn't impl `PartialEq` so
            // compare via the canonical InputForm rendering.
            let original = Expr::FunctionCall {
              name: "SetDelayed".to_string(),
              args: args.to_vec().into(),
            };
            let unchanged = crate::syntax::expr_to_string(&result)
              == crate::syntax::expr_to_string(&original);
            if !unchanged {
              return Ok(result);
            }
          }
          return set_delayed_ast(&args[0], &args[1]);
        }
        // Special handling for TagSetDelayed - stores upvalue definitions
        if name == "TagSetDelayed" && args.len() == 3 {
          return tag_set_delayed_ast(&args[0], &args[1], &args[2], false);
        }
        // Special handling for TagSet - stores evaluated upvalue definitions
        if name == "TagSet" && args.len() == 3 {
          return tag_set_delayed_ast(&args[0], &args[1], &args[2], true);
        }
        // Special handling for TagUnset - removes upvalue definitions
        if name == "TagUnset" && args.len() == 2 {
          return tag_unset_ast(&args[0], &args[1]);
        }
        // Special handling for UpSet - automatically determines tags from LHS
        if name == "UpSet" && args.len() == 2 {
          return upset_ast(&args[0], &args[1]);
        }
        // Special handling for UpSetDelayed - like UpSet but delayed (no RHS evaluation)
        if name == "UpSetDelayed" && args.len() == 2 {
          return upset_delayed_ast(&args[0], &args[1]);
        }
        // Special handling for Increment/Decrement - x++ / x--
        // and PreIncrement/PreDecrement - ++x / --x
        if (name == "Increment"
          || name == "Decrement"
          || name == "PreIncrement"
          || name == "PreDecrement")
          && args.len() == 1
        {
          if let Expr::Identifier(var_name) = &args[0] {
            let current = ENV.with(|e| e.borrow().get(var_name).cloned());
            let current_val = match current {
              Some(StoredValue::ExprVal(e)) => e,
              Some(StoredValue::Raw(s)) => {
                crate::syntax::string_to_expr(&s).unwrap_or(Expr::Integer(0))
              }
              _ => {
                // Match Mathematica: unset variable → emit rvalue warning
                // and leave the Increment/Decrement/PreIncrement/PreDecrement
                // call unevaluated.
                crate::emit_message(&format!(
                  "{}::rvalue: {} is not a variable with a value, so its value cannot be changed.",
                  name, var_name
                ));
                return Ok(Expr::FunctionCall {
                  name: name.clone(),
                  args: args.clone(),
                });
              }
            };
            let delta = if name == "Increment" || name == "PreIncrement" {
              Expr::Integer(1)
            } else {
              Expr::Integer(-1)
            };
            let new_val = evaluate_expr_to_expr(&Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(current_val.clone()),
              right: Box::new(delta),
            })?;
            ENV.with(|e| {
              e.borrow_mut().insert(
                var_name.clone(),
                StoredValue::Raw(crate::syntax::expr_to_string(&new_val)),
              );
            });
            // Post-increment/decrement returns old value; pre returns new value
            if name == "PreIncrement" || name == "PreDecrement" {
              return Ok(new_val);
            }
            return Ok(current_val);
          }
          // Handle Part expressions: ++x[[i]], --x[[i]], x[[i]]++, x[[i]]--
          if let Expr::Part { .. } = &args[0] {
            // Get the current value at the part position
            let current_val = evaluate_expr_to_expr(&args[0])?;
            let delta = if name == "Increment" || name == "PreIncrement" {
              Expr::Integer(1)
            } else {
              Expr::Integer(-1)
            };
            let new_val = evaluate_expr_to_expr(&Expr::BinaryOp {
              op: BinaryOperator::Plus,
              left: Box::new(current_val.clone()),
              right: Box::new(delta),
            })?;
            // Use Set to assign the new value back to the part
            crate::evaluator::assignment::set_ast(&args[0], &new_val)?;
            if name == "PreIncrement" || name == "PreDecrement" {
              return Ok(new_val);
            }
            return Ok(current_val);
          }
          // Target is a non-assignable literal (number, string, expression).
          // Match wolframscript: emit an rvalue message and leave the call
          // unevaluated — e.g. `--5` → "PreDecrement::rvalue: 5 is not a
          // variable...".
          crate::emit_message(&format!(
            "{}::rvalue: {} is not a variable with a value, so its value cannot be changed.",
            name,
            crate::syntax::expr_to_string(&args[0])
          ));
          return Ok(Expr::FunctionCall {
            name: name.clone(),
            args: args.clone(),
          });
        }
        // Special handling for Unset - x =. (removes definition)
        if name == "Unset" && args.len() == 1 {
          // Thread over lists: '{a, {b}} =.' → {Unset[a], Unset[{b}]}.
          if let Expr::List(items) = &args[0] {
            let mut results = Vec::with_capacity(items.len());
            for it in items {
              let r = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Unset".to_string(),
                args: vec![it.clone()].into(),
              })?;
              results.push(r);
            }
            return Ok(Expr::List(results.into()));
          }
          if let Expr::Identifier(var_name) = &args[0] {
            let had_value = ENV.with(|e| e.borrow_mut().remove(var_name));
            if had_value.is_none() {
              // No OwnValue was set. Mathematica still returns Null for
              // 'foo =.' even if foo was never defined, so do the same.
              return Ok(Expr::Identifier("Null".to_string()));
            }
            return Ok(Expr::Identifier("Null".to_string()));
          }
          // OwnValues[sym] =. clears the OwnValue for sym (equivalent to
          // sym =.). Wolfram returns Null whether or not the value was set.
          if let Expr::FunctionCall {
            name: head,
            args: lhs_args,
          } = &args[0]
            && head == "OwnValues"
            && lhs_args.len() == 1
            && let Expr::Identifier(var_name) = &lhs_args[0]
          {
            ENV.with(|e| e.borrow_mut().remove(var_name));
            return Ok(Expr::Identifier("Null".to_string()));
          }
          // SubValues[sym] =. and DownValues[sym] =. clear the corresponding
          // function definitions for sym.
          if let Expr::FunctionCall {
            name: head,
            args: lhs_args,
          } = &args[0]
            && (head == "DownValues" || head == "SubValues")
            && lhs_args.len() == 1
            && let Expr::Identifier(sym_name) = &lhs_args[0]
          {
            crate::FUNC_DEFS.with(|m| {
              m.borrow_mut().remove(sym_name);
            });
            return Ok(Expr::Identifier("Null".to_string()));
          }
          // UpValues[sym] =. removes every upvalue rule attached to
          // `sym`. Each rule lives in two places: `UPVALUES[sym]` (for
          // introspection) and `FUNC_DEFS[outer_head]` (where dispatch
          // looks). Drop both. (Mirrors the upvalue-cleanup branch of
          // `ClearAll[sym]`.)
          if let Expr::FunctionCall {
            name: head,
            args: lhs_args,
          } = &args[0]
            && head == "UpValues"
            && lhs_args.len() == 1
            && let Expr::Identifier(sym_name) = &lhs_args[0]
          {
            let up_defs =
              crate::UPVALUES.with(|m| m.borrow_mut().remove(sym_name));
            if let Some(up_defs) = up_defs {
              for (
                outer_func,
                params,
                _conds,
                _defaults,
                _heads,
                body,
                _orig_lhs,
                _orig_body,
              ) in &up_defs
              {
                let body_str = crate::syntax::expr_to_string(body);
                crate::FUNC_DEFS.with(|m| {
                  if let Some(entry) = m.borrow_mut().get_mut(outer_func) {
                    entry.retain(|(p, _, _, _, _, b)| {
                      !(p == params
                        && crate::syntax::expr_to_string(b) == body_str)
                    });
                  }
                });
              }
            }
            return Ok(Expr::Identifier("Null".to_string()));
          }
          // Messages[sym] =. — Woxi has no per-symbol message storage
          // yet; treat as a no-op success.
          if let Expr::FunctionCall {
            name: head,
            args: lhs_args,
          } = &args[0]
            && head == "Messages"
            && lhs_args.len() == 1
            && matches!(&lhs_args[0], Expr::Identifier(_))
          {
            return Ok(Expr::Identifier("Null".to_string()));
          }
          // Pattern-based unset: f[args] =.
          // Requires a matching DownValue in FUNC_DEFS; otherwise Mathematica
          // emits an 'Unset::norep' warning and returns $Failed.
          if let Expr::FunctionCall {
            name: head,
            args: lhs_args,
          } = &args[0]
          {
            let had_any = crate::FUNC_DEFS.with(|m| {
              m.borrow().get(head).is_some_and(|defs| !defs.is_empty())
            });
            if !had_any {
              let lhs_str = crate::syntax::expr_to_string(&args[0]);
              crate::emit_message(&format!(
                "Unset::norep: Assignment on {} for {} not found.",
                head, lhs_str
              ));
              return Ok(Expr::Identifier("$Failed".to_string()));
            }
            // Reconstruct each entry's LHS pattern and remove only the entry
            // whose LHS matches the unset pattern. Two entries with the same
            // outer head can have different inner patterns (e.g.
            // `MakeBoxes[F[x__], fmt_]` vs `MakeBoxes[G[x___], fmt_]`); we
            // need per-entry comparison so unsetting one keeps the other.
            let target_lhs_str = crate::syntax::expr_to_string(&args[0]);
            let mut removed_any = false;
            crate::FUNC_DEFS.with(|m| {
              let mut map = m.borrow_mut();
              if let Some(entries) = map.get_mut(head) {
                let original_len = entries.len();
                entries.retain(
                  |(params, conds, _defaults, heads, blank_types, _body)| {
                    let pattern_args: Vec<Expr> = params
                      .iter()
                      .enumerate()
                      .map(|(i, p)| {
                        // Structural pattern: pull the original pattern AST
                        // out of the __StructuralPattern__ marker.
                        if let Some(Some(Expr::FunctionCall {
                          name: cn,
                          args: ca,
                        })) = conds.get(i)
                          && cn == "__StructuralPattern__"
                          && ca.len() == 2
                        {
                          return ca[1].clone();
                        }
                        // Literal-match (SameQ): use the literal value.
                        if let Some(Some(Expr::Comparison {
                          operands,
                          operators,
                        })) = conds.get(i)
                          && operators
                            .iter()
                            .any(|op| matches!(op, ComparisonOp::SameQ))
                          && let Some(literal_val) = operands.get(1)
                        {
                          return literal_val.clone();
                        }
                        Expr::Pattern {
                          name: p.clone(),
                          head: heads.get(i).and_then(|h| h.clone()),
                          blank_type: blank_types.get(i).copied().unwrap_or(1),
                        }
                      })
                      .collect();
                    let entry_lhs = Expr::FunctionCall {
                      name: head.clone(),
                      args: pattern_args.into(),
                    };
                    let entry_lhs_str =
                      crate::syntax::expr_to_string(&entry_lhs);
                    entry_lhs_str != target_lhs_str
                  },
                );
                removed_any = entries.len() < original_len;
                if entries.is_empty() {
                  map.remove(head);
                }
              }
            });
            if !removed_any {
              let lhs_str = crate::syntax::expr_to_string(&args[0]);
              crate::emit_message(&format!(
                "Unset::norep: Assignment on {} for {} not found.",
                head, lhs_str
              ));
              return Ok(Expr::Identifier("$Failed".to_string()));
            }
            let _ = lhs_args;
            return Ok(Expr::Identifier("Null".to_string()));
          }
          return Ok(Expr::Identifier("Null".to_string()));
        }
        // Definition and FullDefinition have HoldAll in Wolfram, so their
        // argument stays unevaluated. Information has no Hold attribute —
        // its argument is evaluated like any other call. The `?symbol` REPL
        // shortcut wraps the arg in `Unevaluated` to preserve symbol
        // inspection (handled in the Information dispatcher).
        if (name == "Definition" || name == "FullDefinition") && args.len() == 1
        {
          return dispatch::evaluate_function_call_ast(name, args);
        }
        // ApplyTo[x, f] (x //= f): set x to f[x] and return the new
        // value. HoldFirst; an unset variable emits rvalue.
        if name == "ApplyTo" && args.len() == 2 {
          if let Expr::Identifier(var_name) = &args[0] {
            let current = ENV.with(|e| e.borrow().get(var_name).cloned());
            let current_val = match current {
              Some(StoredValue::ExprVal(e)) => e,
              Some(StoredValue::Raw(s)) => {
                crate::syntax::string_to_expr(&s).unwrap_or(Expr::Integer(0))
              }
              _ => {
                crate::emit_message(&format!(
                  "ApplyTo::rvalue: {} is not a variable with a value, so its value cannot be changed.",
                  var_name
                ));
                return Ok(Expr::FunctionCall {
                  name: name.clone(),
                  args: args.clone(),
                });
              }
            };
            let func = evaluate_expr_to_expr(&args[1])?;
            let new_val =
              crate::evaluator::function_application::apply_function_to_arg(
                &func,
                &current_val,
              )?;
            ENV.with(|e| {
              e.borrow_mut().insert(
                var_name.clone(),
                StoredValue::ExprVal(new_val.clone()),
              );
            });
            return Ok(new_val);
          }
          crate::emit_message(&format!(
            "ApplyTo::rvalue: {} is not a variable with a value, so its value cannot be changed.",
            crate::syntax::expr_to_string(&args[0])
          ));
          return Ok(Expr::FunctionCall {
            name: name.clone(),
            args: args.clone(),
          });
        }
        // Special handling for AddTo, SubtractFrom, TimesBy, DivideBy - x += y, x -= y, etc.
        if (name == "AddTo"
          || name == "SubtractFrom"
          || name == "TimesBy"
          || name == "DivideBy")
          && args.len() == 2
        {
          let op = match name.as_str() {
            "AddTo" => BinaryOperator::Plus,
            "SubtractFrom" => BinaryOperator::Minus,
            "TimesBy" => BinaryOperator::Times,
            "DivideBy" => BinaryOperator::Divide,
            _ => unreachable!(),
          };
          if let Expr::Identifier(var_name) = &args[0] {
            let current = ENV.with(|e| e.borrow().get(var_name).cloned());
            let current_val = match current {
              Some(StoredValue::ExprVal(e)) => e,
              Some(StoredValue::Raw(s)) => {
                crate::syntax::string_to_expr(&s).unwrap_or(Expr::Integer(0))
              }
              _ => {
                crate::emit_message(&format!(
                  "{}::rvalue: {} is not a variable with a value, so its value cannot be changed.",
                  name, var_name
                ));
                // Match Mathematica: leave the whole call unevaluated.
                return Ok(Expr::FunctionCall {
                  name: name.clone(),
                  args: args.clone(),
                });
              }
            };
            let rhs = evaluate_expr_to_expr(&args[1])?;
            let new_val = evaluate_expr_to_expr(&Expr::BinaryOp {
              op,
              left: Box::new(current_val),
              right: Box::new(rhs),
            })?;
            ENV.with(|e| {
              e.borrow_mut().insert(
                var_name.clone(),
                StoredValue::ExprVal(new_val.clone()),
              );
            });
            return Ok(new_val);
          }
          if let Expr::Part { .. } = &args[0] {
            let rhs = evaluate_expr_to_expr(&args[1])?;
            let current_val = evaluate_expr_to_expr(&args[0])?;
            let new_val = evaluate_expr_to_expr(&Expr::BinaryOp {
              op,
              left: Box::new(current_val),
              right: Box::new(rhs),
            })?;
            crate::evaluator::assignment::set_ast(&args[0], &new_val)?;
            return Ok(new_val);
          }
        }
        // AppendTo/PrependTo on a Part target: AppendTo[x[[i]], v]
        // Evaluate the current value at the Part location, append, then
        // write back via the existing Part-assignment machinery. This is
        // how wolframscript handles `AppendTo[nums[[k]], i]`.
        if (name == "AppendTo" || name == "PrependTo")
          && args.len() == 2
          && let Expr::Part { .. } = &args[0]
        {
          let elem = evaluate_expr_to_expr(&args[1])?;
          let is_append = name == "AppendTo";
          let mut current = evaluate_expr_to_expr(&args[0])?;
          let new_val = match &mut current {
            Expr::List(items) => {
              let mut items = std::mem::take(items);
              if is_append {
                items.push(elem);
              } else {
                items.insert(0, elem);
              }
              Expr::List(items)
            }
            Expr::FunctionCall {
              name: head,
              args: fa,
            } => {
              let head = std::mem::take(head);
              let mut fa = std::mem::take(fa);
              if is_append {
                fa.push(elem);
              } else {
                fa.insert(0, elem);
              }
              Expr::FunctionCall {
                name: head,
                args: fa,
              }
            }
            _ => {
              return Err(InterpreterError::EvaluationError(format!(
                "{} requires a list-valued target",
                name
              )));
            }
          };
          crate::evaluator::assignment::set_ast(&args[0], &new_val)?;
          return Ok(new_val);
        }
        // Special handling for AppendTo, PrependTo - x = Append[x, elem]
        if (name == "AppendTo" || name == "PrependTo")
          && args.len() == 2
          && let Expr::Identifier(var_name) = &args[0]
        {
          let elem = evaluate_expr_to_expr(&args[1])?;
          let is_append = name == "AppendTo";
          // Fast path: stored value is already an ExprVal(List) or
          // ExprVal(FunctionCall). Mutate the stored ExprList in place
          // and clone the result once for the return value. Avoids the
          // O(N) clone-on-read that the previous get(...).cloned() path
          // paid on every iteration of a `Do[AppendTo[xs, …], …]` loop.
          let mutated = ENV.with(|e| -> Option<Expr> {
            let mut env = e.borrow_mut();
            match env.get_mut(var_name) {
              Some(StoredValue::ExprVal(stored)) => match stored {
                Expr::List(items) => {
                  if is_append {
                    items.push_back(elem.clone());
                  } else {
                    items.push_front(elem.clone());
                  }
                  Some(stored.clone())
                }
                Expr::FunctionCall {
                  name: _,
                  args: fn_args,
                } => {
                  if is_append {
                    fn_args.push_back(elem.clone());
                  } else {
                    fn_args.push_front(elem.clone());
                  }
                  Some(stored.clone())
                }
                _ => None,
              },
              _ => None,
            }
          });
          if let Some(new_val) = mutated {
            // Register the appended/prepended symbol in $PrintForms /
            // $OutputForms when the target is `$BoxForms`. wolframscript
            // exposes user-added box forms in those lists even after a
            // subsequent `$BoxForms=.` clears the OwnValue.
            if var_name == "$BoxForms"
              && let Expr::Identifier(form_name) = &args[1]
            {
              crate::evaluator::assignment::register_user_print_form(form_name);
            }
            return Ok(new_val);
          }

          // Slow path: stored value is Raw / Association / not yet bound /
          // a non-list ExprVal. Read, decode, mutate, store back.
          let current = ENV.with(|e| e.borrow().get(var_name).cloned());
          let mut current_val = match current {
            Some(StoredValue::ExprVal(e)) => e,
            Some(StoredValue::Raw(s)) => crate::syntax::string_to_expr(&s)
              .unwrap_or(Expr::List(vec![].into())),
            // System variables like `$BoxForms` aren't in ENV until written —
            // first AppendTo seeds the env from the built-in default.
            _ => {
              match crate::evaluator::listable::get_system_variable(var_name) {
                Some(default) => default,
                None => {
                  return Err(InterpreterError::EvaluationError(format!(
                    "{} requires a variable with a list value",
                    name
                  )));
                }
              }
            }
          };
          let new_val = match &mut current_val {
            Expr::List(items) => {
              let mut items = std::mem::take(items);
              if is_append {
                items.push(elem);
              } else {
                items.insert(0, elem);
              }
              Expr::List(items)
            }
            // AppendTo/PrependTo also work on any FunctionCall head
            // (matches wolframscript: AppendTo[f[a,b], x] -> f[a,b,x]).
            Expr::FunctionCall {
              name: head,
              args: fn_args,
            } => {
              let mut new_args = std::mem::take(fn_args);
              if is_append {
                new_args.push(elem);
              } else {
                new_args.insert(0, elem);
              }
              Expr::FunctionCall {
                name: head.clone(),
                args: new_args,
              }
            }
            _ => {
              return Err(InterpreterError::EvaluationError(format!(
                "{}: {} is not a list",
                name, var_name
              )));
            }
          };
          // Store as ExprVal rather than re-serializing to a Raw string.
          // The Raw form forces the next AppendTo/PrependTo on the same
          // variable to re-parse the entire list, which makes a tight
          // `Do[AppendTo[xs, …], …]` loop quadratic. ExprVal keeps the
          // updated list as an Expr in place, so the loop is amortized
          // O(n).
          ENV.with(|e| {
            e.borrow_mut()
              .insert(var_name.clone(), StoredValue::ExprVal(new_val.clone()));
          });
          if var_name == "$BoxForms"
            && let Expr::Identifier(form_name) = &args[1]
          {
            crate::evaluator::assignment::register_user_print_form(form_name);
          }
          return Ok(new_val);
        }
        // Special handling for AssociateTo - x = Append[x, key -> val]
        if name == "AssociateTo"
          && args.len() == 2
          && let Expr::Identifier(var_name) = &args[0]
        {
          let rule = evaluate_expr_to_expr(&args[1])?;
          // Collect the key→value pairs to add. The second argument may be a
          // single rule (`a -> b`), an association, or a list of rules and/or
          // associations (`{k1 -> v1, k2 -> v2}`) — all documented forms.
          fn collect_pairs(e: &Expr, out: &mut Vec<(Expr, Expr)>) -> bool {
            match e {
              Expr::Rule {
                pattern,
                replacement,
              } => {
                out.push((
                  pattern.as_ref().clone(),
                  replacement.as_ref().clone(),
                ));
                true
              }
              Expr::Association(items) => {
                out.extend(items.iter().cloned());
                true
              }
              Expr::List(items) => {
                items.iter().all(|it| collect_pairs(it, out))
              }
              _ => false,
            }
          }
          let mut new_pairs: Vec<(Expr, Expr)> = Vec::new();
          if !collect_pairs(&rule, &mut new_pairs) {
            // Not a recognized key-value form — return unevaluated.
            return Ok(Expr::FunctionCall {
              name: "AssociateTo".to_string(),
              args: vec![args[0].clone(), rule].into(),
            });
          }
          let current = ENV.with(|e| e.borrow().get(var_name).cloned());
          let mut items = match &current {
            Some(StoredValue::Association(pairs)) => pairs
              .iter()
              .map(|(k, v)| {
                let ke = crate::syntax::string_to_expr(k)
                  .unwrap_or(Expr::String(k.clone()));
                let ve = crate::syntax::string_to_expr(v)
                  .unwrap_or(Expr::String(v.clone()));
                (ke, ve)
              })
              .collect::<Vec<_>>(),
            Some(StoredValue::ExprVal(Expr::Association(items))) => {
              items.clone()
            }
            Some(StoredValue::Raw(s)) => {
              match crate::syntax::string_to_expr(s) {
                Ok(Expr::Association(ref items)) => items.clone(),
                _ => {
                  return Ok(Expr::FunctionCall {
                    name: "AssociateTo".to_string(),
                    args: vec![args[0].clone(), rule].into(),
                  });
                }
              }
            }
            _ => {
              return Ok(Expr::FunctionCall {
                name: "AssociateTo".to_string(),
                args: vec![args[0].clone(), rule].into(),
              });
            }
          };
          for (key, val) in new_pairs {
            if let Some(pos) = items.iter().position(|(k, _)| {
              crate::evaluator::pattern_matching::expr_equal(k, &key)
            }) {
              items[pos].1 = val;
            } else {
              items.push((key, val));
            }
          }
          let new_val = Expr::Association(items);
          let pairs: Vec<(String, String)> = match &new_val {
            Expr::Association(items) => items
              .iter()
              .map(|(k, v)| {
                (
                  crate::syntax::expr_to_string(k),
                  crate::syntax::expr_to_string(v),
                )
              })
              .collect(),
            _ => unreachable!(),
          };
          ENV.with(|e| {
            e.borrow_mut()
              .insert(var_name.clone(), StoredValue::Association(pairs));
          });
          return Ok(new_val);
        }
        // Special handling for KeyDropFrom - removes key from association variable
        if name == "KeyDropFrom"
          && args.len() == 2
          && let Expr::Identifier(var_name) = &args[0]
        {
          let key = evaluate_expr_to_expr(&args[1])?;
          let current = ENV.with(|e| e.borrow().get(var_name).cloned());
          let items = match &current {
            Some(StoredValue::Association(pairs)) => pairs
              .iter()
              .map(|(k, v)| {
                let ke = crate::syntax::string_to_expr(k)
                  .unwrap_or(Expr::String(k.clone()));
                let ve = crate::syntax::string_to_expr(v)
                  .unwrap_or(Expr::String(v.clone()));
                (ke, ve)
              })
              .collect::<Vec<_>>(),
            Some(StoredValue::ExprVal(Expr::Association(items))) => {
              items.clone()
            }
            Some(StoredValue::Raw(s)) => {
              match crate::syntax::string_to_expr(s) {
                Ok(Expr::Association(ref items)) => items.clone(),
                _ => {
                  return Ok(Expr::FunctionCall {
                    name: "KeyDropFrom".to_string(),
                    args: vec![args[0].clone(), key].into(),
                  });
                }
              }
            }
            _ => {
              // Undefined symbol: wolframscript emits ::blnoval.
              crate::emit_message(&format!(
                "KeyDropFrom::blnoval: The symbol {} at position 1 should have an immediate value defined.",
                var_name
              ));
              return Ok(Expr::FunctionCall {
                name: "KeyDropFrom".to_string(),
                args: vec![args[0].clone(), key].into(),
              });
            }
          };
          let filtered: Vec<(Expr, Expr)> = items
            .into_iter()
            .filter(|(k, _)| {
              !crate::evaluator::pattern_matching::expr_equal(k, &key)
            })
            .collect();
          let new_val = Expr::Association(filtered);
          let pairs: Vec<(String, String)> = match &new_val {
            Expr::Association(items) => items
              .iter()
              .map(|(k, v)| {
                (
                  crate::syntax::expr_to_string(k),
                  crate::syntax::expr_to_string(v),
                )
              })
              .collect(),
            _ => unreachable!(),
          };
          ENV.with(|e| {
            e.borrow_mut()
              .insert(var_name.clone(), StoredValue::Association(pairs));
          });
          return Ok(new_val);
        }
        // Special handling for Return - raises ReturnValue to short-circuit evaluation
        if name == "Return" {
          let val = if args.is_empty() {
            Expr::Identifier("Null".to_string())
          } else {
            evaluate_expr_to_expr(&args[0])?
          };
          return Err(InterpreterError::ReturnValue(Box::new(val)));
        }
        // Special handling for Break[] - raises BreakSignal
        if name == "Break" && args.is_empty() {
          return Err(InterpreterError::BreakSignal);
        }
        // Special handling for Continue[] - raises ContinueSignal
        if name == "Continue" && args.is_empty() {
          return Err(InterpreterError::ContinueSignal);
        }
        // Label[tag] is not evaluated — it stays symbolic.
        // CompoundExpression handles it via find_label_index on the AST.
        // Special handling for Goto[tag] — evaluates the tag then raises GotoSignal
        if name == "Goto" && args.len() == 1 {
          let tag = evaluate_expr_to_expr(&args[0])?;
          return Err(InterpreterError::GotoSignal(Box::new(tag)));
        }
        // Special handling for Throw[value] and Throw[value, tag]
        if name == "Throw" && !args.is_empty() && args.len() <= 2 {
          let val = evaluate_expr_to_expr(&args[0])?;
          let tag = if args.len() == 2 {
            Some(Box::new(evaluate_expr_to_expr(&args[1])?))
          } else {
            None
          };
          return Err(InterpreterError::ThrowValue(Box::new(val), tag));
        }
        // Special handling for Catch[expr], Catch[expr, form] and
        // Catch[expr, form, f]. The form is matched as a pattern (so `a | b`
        // and `_` work); with the 3-argument form, `f[value, tag]` is returned
        // instead of the bare thrown value.
        if name == "Catch" && !args.is_empty() && args.len() <= 3 {
          let tag_pattern = if args.len() >= 2 {
            Some(evaluate_expr_to_expr(&args[1])?)
          } else {
            None
          };
          match evaluate_expr_to_expr(&args[0]) {
            Ok(result) => return Ok(result),
            Err(InterpreterError::ThrowValue(val, thrown_tag)) => {
              let matched = match &tag_pattern {
                // No form: only untagged Throws are caught — a tagged
                // Throw passes through Catch[expr] to the top level
                // (Throw::nocatch), matching wolframscript.
                None => thrown_tag.is_none(),
                // A form only matches a tagged Throw whose tag matches it.
                Some(pattern) => match &thrown_tag {
                  Some(tag) => {
                    crate::functions::list_helpers_ast::matches_pattern_ast(
                      tag, pattern,
                    )
                  }
                  None => false,
                },
              };
              if !matched {
                // Tag doesn't match - re-throw.
                return Err(InterpreterError::ThrowValue(val, thrown_tag));
              }
              // Caught. With a third argument, apply f to value and tag.
              if args.len() == 3 {
                let f = evaluate_expr_to_expr(&args[2])?;
                let tag_expr = thrown_tag
                  .map(|t| *t)
                  .unwrap_or_else(|| Expr::Identifier("Null".to_string()));
                let application = if let Expr::Identifier(fname) = &f {
                  Expr::FunctionCall {
                    name: fname.clone(),
                    args: vec![*val, tag_expr].into(),
                  }
                } else {
                  Expr::CurriedCall {
                    func: Box::new(f),
                    args: vec![*val, tag_expr],
                  }
                };
                return evaluate_expr_to_expr(&application);
              }
              return Ok(*val);
            }
            Err(e) => return Err(e),
          }
        }
        // Special handling for Abort[] - abort computation
        if name == "Abort" && args.is_empty() {
          return Err(InterpreterError::Abort);
        }
        // Quit[] / Exit[] - terminate the process
        if (name == "Quit" || name == "Exit") && args.len() <= 1 {
          #[cfg(not(target_arch = "wasm32"))]
          {
            let code = if args.len() == 1 {
              if let Ok(val) = evaluate_expr_to_expr(&args[0]) {
                crate::functions::math_ast::try_eval_to_f64(&val)
                  .map(|f| f as i32)
                  .unwrap_or(0)
              } else {
                0
              }
            } else {
              0
            };
            std::process::exit(code);
          }
          #[cfg(target_arch = "wasm32")]
          {
            return Err(InterpreterError::Abort);
          }
        }
        // Interrupt[] behaves like Abort[] in batch mode
        if name == "Interrupt" && args.is_empty() {
          return Err(InterpreterError::Abort);
        }
        // Pause[n] - sleep for n seconds (wall clock) and return Null.
        // A negative or non-numeric n emits Pause::numnm and leaves the
        // call unevaluated, matching wolframscript.
        if name == "Pause" && args.len() == 1 {
          let evaluated = evaluate_expr_to_expr(&args[0]).ok();
          let secs = evaluated
            .as_ref()
            .and_then(crate::functions::math_ast::try_eval_to_f64);
          match secs {
            Some(s) if s >= 0.0 => {
              if s > 0.0 {
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::sleep(std::time::Duration::from_secs_f64(s));
                #[cfg(target_arch = "wasm32")]
                crate::wasm::sleep_seconds(s);
              }
              return Ok(Expr::Identifier("Null".to_string()));
            }
            _ => {
              let shown = evaluated.as_ref().unwrap_or(&args[0]);
              let call_str =
                crate::syntax::expr_to_string(&Expr::FunctionCall {
                  name: "Pause".to_string(),
                  args: vec![shown.clone()].into(),
                });
              crate::emit_message(&format!(
                "Pause::numnm: Non-negative machine-sized number expected at position 1 in {}.",
                call_str
              ));
              return Ok(Expr::FunctionCall {
                name: "Pause".to_string(),
                args: vec![shown.clone()].into(),
              });
            }
          }
        }
        // AbortProtect[expr] evaluates expr and returns its result (the body
        // runs transparently; the deferred-abort nuance is not modeled).
        if name == "AbortProtect" && args.len() == 1 {
          return evaluate_expr_to_expr(&args[0]);
        }
        // Special handling for CheckAbort[expr, failexpr]
        if name == "CheckAbort" && args.len() == 2 {
          match evaluate_expr_to_expr(&args[0]) {
            Ok(result) => return Ok(result),
            Err(InterpreterError::Abort) => {
              return Err(InterpreterError::TailCall(Box::new(
                args[1].clone(),
              )));
            }
            Err(e) => return Err(e),
          }
        }
        // Special handling for Check[expr, failexpr] and the tag-filtered
        // Check[expr, failexpr, s1::t1, ...] form (tags may also come in
        // lists). With tags, only the listed messages trigger failexpr.
        if name == "Check" && args.len() >= 2 {
          fn collect_message_tags(e: &Expr, out: &mut Vec<String>) {
            match e {
              Expr::FunctionCall { name, args }
                if name == "MessageName" && args.len() == 2 =>
              {
                let sym = crate::syntax::expr_to_string(&args[0]);
                let tag = match &args[1] {
                  Expr::String(s) => s.clone(),
                  other => crate::syntax::expr_to_string(other),
                };
                out.push(format!("{sym}::{tag}"));
              }
              Expr::List(items) => {
                for item in items {
                  collect_message_tags(item, out);
                }
              }
              _ => {}
            }
          }
          let warnings_before = crate::get_captured_warnings().len();
          let msgs_before = crate::get_captured_messages_raw().len();
          match evaluate_expr_to_expr(&args[0]) {
            Ok(result) => {
              let triggered = if args.len() == 2 {
                crate::get_captured_warnings().len() > warnings_before
              } else {
                let mut tags: Vec<String> = Vec::new();
                for spec in &args[2..] {
                  collect_message_tags(spec, &mut tags);
                }
                let msgs = crate::get_captured_messages_raw();
                msgs[msgs_before.min(msgs.len())..].iter().any(|m| {
                  crate::message_name_of(m).is_some_and(|n| tags.contains(&n))
                })
              };
              if triggered {
                return Err(InterpreterError::TailCall(Box::new(
                  args[1].clone(),
                )));
              }
              return Ok(result);
            }
            Err(_) => {
              return Err(InterpreterError::TailCall(Box::new(
                args[1].clone(),
              )));
            }
          }
        }
        // Special handling for Association[args]: if any raw arg is
        // structurally non-normalizable (e.g. a List containing a non-Rule
        // item), short-circuit to the unevaluated `Association[args]` form
        // BEFORE evaluating args. wolframscript's `Association` has
        // `HoldAllComplete`, so when normalization can't proceed it keeps
        // the inner Association literals (with their empty `{}`/`<||>`
        // entries) intact rather than collapsing them via standalone
        // evaluation.
        if name == "Association" && !args.is_empty() {
          fn is_normalizable_assoc_arg(e: &Expr) -> bool {
            match e {
              Expr::Rule { .. }
              | Expr::RuleDelayed { .. }
              | Expr::Association(_) => true,
              Expr::FunctionCall { name: n, args: ca }
                if n == "Rule" && ca.len() == 2 =>
              {
                true
              }
              Expr::FunctionCall { name: n, args: ca }
                if n == "RuleDelayed" && ca.len() == 2 =>
              {
                true
              }
              Expr::FunctionCall {
                name: n,
                args: aargs,
              } if n == "Association" => {
                aargs.iter().all(is_normalizable_assoc_arg)
              }
              Expr::List(items) => items.iter().all(is_normalizable_assoc_arg),
              _ => false,
            }
          }
          if !args.iter().all(is_normalizable_assoc_arg) {
            // HoldAllComplete, but Association still evaluates its arguments
            // to see whether they normalize to a valid association structure
            // — e.g. Association[Table[k -> v, ...]] builds from the resulting
            // list of rules. If the evaluated form is a valid structure, build
            // from it; otherwise keep the ORIGINAL held arguments
            // (Association[Range[3]] and Association[x] stay unevaluated).
            let evaluated: Vec<Expr> = args
              .iter()
              .map(crate::evaluator::evaluate_expr_to_expr)
              .collect::<Result<_, _>>()?;
            if evaluated.iter().all(is_normalizable_assoc_arg) {
              return crate::evaluator::evaluate_expr_to_expr(
                &Expr::FunctionCall {
                  name: "Association".to_string(),
                  args: evaluated.into(),
                },
              );
            }
            return Ok(Expr::FunctionCall {
              name: name.to_string(),
              args: args.to_vec().into(),
            });
          }
        }
        // Special handling for Quiet[expr], Quiet[expr, msgs], Quiet[expr, moff, mon]
        if name == "Quiet" {
          if (args.is_empty() || args.len() > 3)
            && let Some(result) =
              dispatch::arg_count::check_arg_count(name, args)
          {
            return result;
          }
          return crate::functions::control_flow_ast::quiet_ast(args);
        }
        // Special handling for Switch - lazy evaluation of branches.
        // `>= 2` (rather than `>= 3`) so an even argument count like
        // `Switch[2, 1]` reaches switch_ast, which reports Switch::argct.
        if name == "Switch" && args.len() >= 2 {
          return crate::functions::control_flow_ast::switch_ast(args);
        }
        // Special handling for Piecewise - lazy evaluation of branches
        if name == "Piecewise" && !args.is_empty() && args.len() <= 2 {
          return crate::functions::control_flow_ast::piecewise_ast(args);
        }
        // Special handling for TraceScan - traces evaluation of sub-expressions
        if name == "TraceScan" && args.len() >= 2 && args.len() <= 3 {
          return crate::functions::control_flow_ast::trace_scan_ast(args);
        }
        // Special handling for Table, Do, With - don't evaluate args (body needs iteration/bindings)
        // These functions take unevaluated expressions as first argument
        if name == "Table"
        || name == "Do"
        // ParallelDo holds its body like Do so side-effecting expressions
        // (e.g. Print[i]) aren't evaluated once before iteration begins.
        || name == "ParallelDo"
        || name == "With"
        || name == "Block"
        || name == "Function"
        || name == "For"
        || name == "While"
        || name == "ClearAll"
        || name == "Clear"
        || name == "Hold"
        || name == "HoldForm"
        || name == "HoldComplete"
        || name == "Unevaluated"
        || name == "ValueQ"
        || name == "Reap"
        || name == "Plot"
        || name == "Plot3D"
        // Note: Graphics is NOT held – its args are evaluated normally
        // (so Thick → Thickness[Large], Red → RGBColor[1,0,0], etc.)
        || name == "ParametricPlot"
        || name == "PolarPlot"
        || name == "DensityPlot"
        || name == "ContourPlot"
        || name == "RegionPlot"
        || name == "StreamPlot"
        || name == "VectorPlot"
        || name == "StreamDensityPlot"
        || name == "Show"
        || name == "GraphicsRow"
        || name == "GraphicsColumn"
        || name == "GraphicsGrid"
        || name == "PlotGrid"
        || name == "BooleanTable"
        // Manipulate holds its body and variable specs so that controls
        // like Plot[Sin[a*x], {x, 0, 6}] aren't evaluated with `a` free
        // (matching Wolfram Language notebook behavior).
        || name == "Manipulate"
        // Animate is Manipulate that auto-plays; it holds its body and
        // variable spec for the same reason. (ListAnimate is deliberately
        // NOT held: its argument is a precomputed list of frames that must
        // evaluate first, matching Wolfram Language.)
        || name == "Animate"
        // Control likewise holds its variable spec so that the bound symbol
        // and any symbolic values stay intact for interactive rendering.
        || name == "Control"
        // LocatorPane holds its body graphic so it isn't evaluated with the
        // locator point free; ClickPane holds its handler so it stays an
        // un-applied function until fed a click position.
        || name == "LocatorPane"
        || name == "ClickPane"
        {
          // Flatten Sequence even in held args (unless SequenceHold)
          let flattened = flatten_sequences(name, args);
          // Honour Evaluate[...] inside HoldAll wrappers like Hold and
          // HoldForm — Evaluate forces evaluation of its argument even
          // through a hold. HoldComplete suppresses it (matching Wolfram).
          // The plotting heads are included so the common
          // `Plot[Table[…] // Evaluate, …]` idiom yields the evaluated
          // list of curves instead of a whole-body expression the
          // samplers can't decompose.
          let args: std::borrow::Cow<[Expr]> = if matches!(
            name.as_str(),
            "Hold"
              | "HoldForm"
              | "Function"
              | "Reap"
              | "Manipulate"
              | "Animate"
              | "Plot"
              | "Plot3D"
              | "ParametricPlot"
              | "ParametricPlot3D"
              | "PolarPlot"
              | "DensityPlot"
              | "ContourPlot"
              | "RegionPlot"
              | "StreamPlot"
              | "VectorPlot"
              | "StreamDensityPlot"
          ) {
            let raw: Vec<Expr> = flattened
              .iter()
              .map(unwrap_top_level_evaluate)
              .collect::<Result<Vec<Expr>, _>>()?;
            // Multi-arg Evaluate yields a Sequence that must splice into
            // the surrounding hold context (e.g. `Hold[Evaluate[1, 2]]`
            // becomes `Hold[1, 2]`).
            std::borrow::Cow::Owned(splice_top_level_sequences(raw))
          } else {
            flattened
          };
          // Pass unevaluated args to the function dispatcher
          return evaluate_function_call_ast(name, &args);
        }
        // Check if name is a variable holding a callable value (Function, FunctionCall like Composition)
        let var_val = ENV.with(|e| e.borrow().get(name).cloned());
        if let Some(StoredValue::ExprVal(stored_expr)) = &var_val {
          match stored_expr {
            Expr::Function { .. }
            | Expr::NamedFunction { .. }
            | Expr::FunctionCall { .. } => {
              // For Function[{params}, body, attrs] with HoldAll/HoldFirst/
              // HoldRest, suppress argument evaluation so the function body
              // sees the unevaluated forms.
              let hold_attrs = function_hold_attributes(stored_expr);
              let mut prepared: Vec<Expr> = Vec::with_capacity(args.len());
              for (i, a) in args.iter().enumerate() {
                let hold = hold_attrs.0
                  || (hold_attrs.1 && i == 0)
                  || (hold_attrs.2 && i > 0);
                if hold {
                  prepared.push(a.clone());
                } else {
                  prepared.push(evaluate_expr_to_expr(a)?);
                }
              }
              return apply_curried_call(stored_expr, &prepared);
            }
            _ => {}
          }
        }
        // Variable holds a symbol name (e.g. t = Flatten) — re-dispatch as that function
        if let Some(resolved_name) = resolve_identifier_to_func_name(name) {
          return Err(InterpreterError::TailCall(Box::new(
            Expr::FunctionCall {
              name: resolved_name,
              args: args.to_vec().into(),
            },
          )));
        }
        // Check if name is a variable holding an association (for nested access: assoc["a", "b"])
        if let Some(StoredValue::Association(_)) = var_val {
          // Evaluate arguments and perform nested access
          let evaluated_args: Vec<Expr> = args
            .iter()
            .map(evaluate_expr_to_expr)
            .collect::<Result<_, _>>()?;
          return association_nested_access(name, &evaluated_args);
        }

        // Try early dispatch for held functions
        if let Some(result) = evaluate_expr_to_expr_early_dispatch(name, args)?
        {
          return Ok(result);
        }

        // Evaluate arguments (respecting Hold attributes)
        let evaluated_args = evaluate_args_with_hold(name, args)?;
        // Flatten Sequence arguments (unless function has SequenceHold attribute)
        let evaluated_args = flatten_sequences(name, &evaluated_args);
        // Dispatch to function implementation
        evaluate_function_call_ast(name, &evaluated_args)
      })(); // end of closure for stack tracking
      // Capture stack trace before popping, so errors carry context
      if matches!(&result, Err(InterpreterError::EvaluationError(_))) {
        crate::capture_error_trace();
      }
      crate::pop_eval_stack();
      result
    }
    Expr::BinaryOp { op, left, right } => {
      // Short-circuit evaluation for And (&&) and Or (||):
      // Evaluate left side first, and only evaluate right side if needed.
      match op {
        BinaryOperator::And => {
          let left_val = evaluate_expr_to_expr(left)?;
          if matches!(&left_val, Expr::Identifier(s) if s == "False") {
            return Ok(Expr::Identifier("False".to_string()));
          }
          let right_val = evaluate_expr_to_expr(right)?;
          let has_user_rules =
            crate::FUNC_DEFS.with(|m| m.borrow().contains_key("And"));
          if has_user_rules {
            return evaluate_function_call_ast("And", &[left_val, right_val]);
          }
          return crate::functions::boolean_ast::and_ast(&[
            left_val, right_val,
          ]);
        }
        BinaryOperator::Or => {
          let left_val = evaluate_expr_to_expr(left)?;
          if matches!(&left_val, Expr::Identifier(s) if s == "True") {
            return Ok(Expr::Identifier("True".to_string()));
          }
          let right_val = evaluate_expr_to_expr(right)?;
          let has_user_rules =
            crate::FUNC_DEFS.with(|m| m.borrow().contains_key("Or"));
          if has_user_rules {
            return evaluate_function_call_ast("Or", &[left_val, right_val]);
          }
          return crate::functions::boolean_ast::or_ast(&[left_val, right_val]);
        }
        BinaryOperator::StringJoin => {
          // StringJoin is Flat: evaluate the whole `a <> b <> c` chain as one
          // StringJoin so string_join_ast sees every operand at once (correct
          // message positions and a flat result), rather than evaluating each
          // nested BinaryOp separately and emitting premature messages.
          fn gather(e: &Expr, out: &mut Vec<Expr>) {
            if let Expr::BinaryOp {
              op: BinaryOperator::StringJoin,
              left,
              right,
            } = e
            {
              gather(left, out);
              gather(right, out);
            } else {
              out.push(e.clone());
            }
          }
          let mut operands = Vec::new();
          gather(left, &mut operands);
          gather(right, &mut operands);
          let mut evaled = Vec::with_capacity(operands.len());
          for o in &operands {
            evaled.push(evaluate_expr_to_expr(o)?);
          }
          return crate::functions::string_ast::string_join_ast(&evaled);
        }
        _ => {}
      }

      let left_val = evaluate_expr_to_expr(left)?;
      let right_val = evaluate_expr_to_expr(right)?;

      // Splice Sequence operands into the corresponding n-ary operation, e.g.
      // `Sequence[1, 2] + Sequence[3, 4]` -> `Plus[1, 2, 3, 4]` = 10. Map
      // subtraction and division onto Plus/Times (with a negated/reciprocal
      // second operand) so the spliced arguments combine the same way Wolfram
      // does (`Sequence[1, 2] - 3` -> `Plus[1, 2, -3]` = 0). Re-evaluating the
      // function-call form runs `flatten_sequences`, which performs the splice.
      {
        let is_seq = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "Sequence");
        if is_seq(&left_val) || is_seq(&right_val) {
          let neg = |e: Expr| Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), e].into(),
          };
          let recip = |e: Expr| Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![e, Expr::Integer(-1)].into(),
          };
          let spliced: Option<(&str, Expr, Expr)> = match op {
            BinaryOperator::Plus => {
              Some(("Plus", left_val.clone(), right_val.clone()))
            }
            BinaryOperator::Times => {
              Some(("Times", left_val.clone(), right_val.clone()))
            }
            BinaryOperator::Power => {
              Some(("Power", left_val.clone(), right_val.clone()))
            }
            BinaryOperator::StringJoin => {
              Some(("StringJoin", left_val.clone(), right_val.clone()))
            }
            BinaryOperator::Minus => {
              Some(("Plus", left_val.clone(), neg(right_val.clone())))
            }
            BinaryOperator::Divide => {
              Some(("Times", left_val.clone(), recip(right_val.clone())))
            }
            _ => None,
          };
          if let Some((name, l, r)) = spliced {
            return evaluate_expr_to_expr(&Expr::FunctionCall {
              name: name.to_string(),
              args: vec![l, r].into(),
            });
          }
        }
      }

      // For operators with corresponding function names (Plus, Times, Power, etc.),
      // check if there are user-defined rules (e.g. upvalues from TagSetDelayed).
      // If so, route through evaluate_function_call_ast which checks FUNC_DEFS first.
      let func_name = match op {
        BinaryOperator::Plus => Some("Plus"),
        BinaryOperator::Times => Some("Times"),
        BinaryOperator::Power => Some("Power"),
        BinaryOperator::StringJoin => Some("StringJoin"),
        _ => None,
      };
      if let Some(name) = func_name {
        let has_user_rules =
          crate::FUNC_DEFS.with(|m| m.borrow().contains_key(name));
        if has_user_rules {
          return evaluate_function_call_ast(name, &[left_val, right_val]);
        }
        // If the head symbol has an OwnValue (e.g. `Unprotect[Plus]; Plus=Q`),
        // substitute the head and re-evaluate as `Q[a, b]` rather than the
        // built-in plus_ast path. Matches Wolfram: `Plus = Q; a + b` → `Q[a, b]`.
        let own_value_head: Option<String> = ENV.with(|e| {
          let env = e.borrow();
          match env.get(name) {
            Some(StoredValue::ExprVal(Expr::Identifier(s))) => Some(s.clone()),
            // OwnValues set via `Sym = simple_identifier` are stored as Raw
            // text (the assignment shortcut). Treat a single-identifier
            // textual value the same as a head substitution.
            Some(StoredValue::Raw(s))
              if s.chars().all(|c| c.is_alphanumeric() || c == '$') =>
            {
              Some(s.clone())
            }
            _ => None,
          }
        });
        if let Some(new_head) = own_value_head
          && new_head != name
        {
          return evaluate_function_call_ast(&new_head, &[left_val, right_val]);
        }
      }

      // Check for list threading (arithmetic operations thread over lists)
      let has_list = matches!(&left_val, Expr::List(_))
        || matches!(&right_val, Expr::List(_));

      if has_list
        && matches!(
          op,
          BinaryOperator::Plus
            | BinaryOperator::Minus
            | BinaryOperator::Times
            | BinaryOperator::Divide
            | BinaryOperator::Power
        )
      {
        return thread_binary_op(&left_val, &right_val, *op);
      }

      // Try numeric evaluation
      let left_num = expr_to_number(&left_val);
      let right_num = expr_to_number(&right_val);

      match op {
        BinaryOperator::Plus => {
          // Use plus_ast for proper symbolic handling and canonical ordering
          crate::functions::math_ast::plus_ast(&[left_val, right_val])
        }
        BinaryOperator::Minus => {
          // Handle BigInteger arithmetic
          if let Some(result) =
            bigint_binary_op(&left_val, &right_val, |a, b| a - b)
          {
            Ok(result)
          } else if matches!(&left_val, Expr::BigFloat(_, _))
            || matches!(&right_val, Expr::BigFloat(_, _))
          {
            // Arbitrary-precision operands: route through subtract_ast
            // (a + (-1)*b) so precision is tracked, instead of collapsing to
            // an f64 Real. Matches `Subtract[N[Pi,30], 3]`.
            crate::functions::math_ast::subtract_ast(&[left_val, right_val])
          } else if let (Some(l), Some(r)) = (left_num, right_num) {
            if matches!(&left_val, Expr::Real(_))
              || matches!(&right_val, Expr::Real(_))
            {
              Ok(Expr::Real(l - r))
            } else {
              Ok(num_to_expr(l - r))
            }
          } else if matches!(&left_val, Expr::Integer(0)) {
            // 0 - x => -x via times_ast for proper distribution
            crate::functions::math_ast::times_ast(&[
              Expr::Integer(-1),
              right_val,
            ])
          } else if crate::functions::quantity_ast::is_quantity(&left_val)
            .is_some()
            || crate::functions::quantity_ast::is_quantity(&right_val).is_some()
          {
            // Quantity subtraction: delegate to subtract_ast → Plus[a, Times[-1, b]]
            crate::functions::math_ast::subtract_ast(&[left_val, right_val])
          } else {
            // Use subtract_ast for proper distribution of -1 over Plus
            crate::functions::math_ast::subtract_ast(&[left_val, right_val])
          }
        }
        BinaryOperator::Times => {
          // Use times_ast for proper symbolic handling and canonical ordering
          crate::functions::math_ast::times_ast(&[left_val, right_val])
        }
        BinaryOperator::Divide => {
          // Delegate to divide_ast for proper handling of Rational, Real, etc.
          crate::functions::math_ast::divide_ast(&[left_val, right_val])
        }
        BinaryOperator::Power => {
          // Delegate to power_ast for proper handling of Rational, Real, etc.
          crate::functions::math_ast::power_ast(&[left_val, right_val])
        }
        BinaryOperator::And | BinaryOperator::Or => {
          // Handled above with short-circuit evaluation
          unreachable!()
        }
        BinaryOperator::StringJoin => {
          // Route through string_join_ast so that non-string arguments emit
          // the StringJoin::string warning and return unevaluated (matching
          // wolframscript), instead of silently coercing identifiers/integers.
          crate::functions::string_ast::string_join_ast(&[left_val, right_val])
        }
        BinaryOperator::Alternatives => {
          // `a | b | c` (the `|` operator) is a flat Alternatives head in WL:
          // Length[a | b | c] == 3. The parser builds it as a nested BinaryOp
          // chain, so canonicalize the evaluated chain into a flat
          // Alternatives[...] FunctionCall. This makes structural operations
          // (Part, Sort, MemberQ, Append, …) see all operands as siblings,
          // matching wolframscript. Held patterns keep their BinaryOp form and
          // are handled directly by the pattern matcher, which accepts both.
          fn push_alt(e: &Expr, out: &mut Vec<Expr>) {
            match e {
              Expr::FunctionCall { name, args } if name == "Alternatives" => {
                out.extend(args.iter().cloned());
              }
              Expr::BinaryOp {
                op: BinaryOperator::Alternatives,
                left,
                right,
              } => {
                push_alt(left, out);
                push_alt(right, out);
              }
              other => out.push(other.clone()),
            }
          }
          let mut parts = Vec::new();
          push_alt(&left_val, &mut parts);
          push_alt(&right_val, &mut parts);
          Ok(Expr::FunctionCall {
            name: "Alternatives".to_string(),
            args: parts.into(),
          })
        }
      }
    }
    Expr::UnaryOp { op, operand } => {
      let val = evaluate_expr_to_expr(operand)?;
      match op {
        UnaryOperator::Minus => {
          // Use times_ast for proper distribution: -(a+b) → -a - b
          crate::functions::math_ast::times_ast(&[Expr::Integer(-1), val])
        }
        UnaryOperator::Not => {
          if matches!(&val, Expr::Identifier(s) if s == "True") {
            Ok(Expr::Identifier("False".to_string()))
          } else if matches!(&val, Expr::Identifier(s) if s == "False") {
            Ok(Expr::Identifier("True".to_string()))
          } else {
            Ok(Expr::UnaryOp {
              op: *op,
              operand: Box::new(val),
            })
          }
        }
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      if operands.len() < 2 || operators.is_empty() {
        return Ok(Expr::Identifier("True".to_string()));
      }

      let values: Vec<Expr> = operands
        .iter()
        .map(evaluate_expr_to_expr)
        .collect::<Result<_, _>>()?;

      // Mixed-operator chains (e.g. `a == b != c`, `a < b > c`) split into
      // pairwise `a op1 b && b op2 c && …` per wolframscript. Homogeneous
      // chains stay intact so downstream code can keep the `Comparison`
      // node and decide numerically.
      if operators.len() >= 2 && !all_same_comparison_family(operators) {
        let mut terms: Vec<Expr> = Vec::with_capacity(operators.len());
        for i in 0..operators.len() {
          terms.push(Expr::Comparison {
            operands: vec![values[i].clone(), values[i + 1].clone()],
            operators: vec![operators[i]],
          });
        }
        let and_expr = Expr::FunctionCall {
          name: "And".to_string(),
          args: terms.into(),
        };
        return evaluate_expr_to_expr(&and_expr);
      }

      // Comparisons with complex infinities: the four order relations are
      // invalid — emit `<Head>::nord` and stay unevaluated like
      // wolframscript — and Equal/Unequal between TWO complex infinities
      // is indeterminate (unevaluated, no message). Real-directed
      // infinities (DirectedInfinity[±1]) stay comparable.
      {
        let complex_inf_display = |e: &Expr| -> Option<String> {
          match e {
            Expr::Identifier(s) | Expr::Constant(s)
              if s == "ComplexInfinity" =>
            {
              Some("ComplexInfinity".to_string())
            }
            Expr::FunctionCall { name, args } if name == "DirectedInfinity" => {
              match args.len() {
                0 => Some("ComplexInfinity".to_string()),
                1 => {
                  let real = matches!(
                    &args[0],
                    Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_)
                  ) || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Rational");
                  if real {
                    None
                  } else {
                    Some(format!(
                      "{} Infinity",
                      crate::syntax::expr_to_string(&args[0])
                    ))
                  }
                }
                _ => None,
              }
            }
            _ => None,
          }
        };
        for i in 0..operators.len() {
          let op = operators[i];
          let display = complex_inf_display(&values[i])
            .or_else(|| complex_inf_display(&values[i + 1]));
          let Some(display) = display else { continue };
          let head = match op {
            ComparisonOp::Less => Some("Less"),
            ComparisonOp::LessEqual => Some("LessEqual"),
            ComparisonOp::Greater => Some("Greater"),
            ComparisonOp::GreaterEqual => Some("GreaterEqual"),
            _ => None,
          };
          if let Some(head) = head {
            crate::emit_message(&format!(
              "{head}::nord: Invalid comparison with {display} attempted."
            ));
            return Ok(Expr::Comparison {
              operands: values,
              operators: operators.clone(),
            });
          }
          let _ = display;
        }

        // Equal/Unequal with infinities (wolframscript-verified):
        // any infinity vs a finite numeric quantity → False / True;
        // two explicit directions compare by direction (Infinity ==
        // DirectedInfinity[I] → False, DirectedInfinity[I] ==
        // DirectedInfinity[I] → True); ComplexInfinity (unknown
        // direction) vs any infinity stays unevaluated.
        for i in 0..operators.len() {
          let op = operators[i];
          if !matches!(op, ComparisonOp::Equal | ComparisonOp::NotEqual) {
            continue;
          }
          let verdict = crate::functions::boolean_ast::infinity_equal_verdict(
            &values[i],
            &values[i + 1],
          );
          match verdict {
            None => {}
            Some(Some(eq)) => {
              let result = eq == matches!(op, ComparisonOp::Equal);
              // A false pair settles an Equal chain (and an equal
              // adjacent pair settles an Unequal chain) of any length;
              // a true pair is only decisive for a 2-operand comparison.
              if !result || operands.len() == 2 {
                return Ok(Expr::Identifier(
                  if result { "True" } else { "False" }.to_string(),
                ));
              }
            }
            Some(None) => {
              return Ok(Expr::Comparison {
                operands: values,
                operators: operators.clone(),
              });
            }
          }
        }
      }

      // UnsameQ with 3+ operands is NOT transitive: it requires ALL pairs
      // to be distinct, not just adjacent ones. Delegate to unsame_q_ast,
      // which implements the correct all-pairs check.
      if operators.len() >= 2
        && operators.iter().all(|op| *op == ComparisonOp::UnsameQ)
      {
        return crate::functions::boolean_ast::unsame_q_ast(&values);
      }

      // Unequal (!=) with 3+ operands is also all-pairs, not adjacent
      // pairs. "abc" != "def" != "abc" must return False.
      if operators.len() >= 2
        && operators.iter().all(|op| *op == ComparisonOp::NotEqual)
      {
        let result = crate::functions::boolean_ast::unequal_ast(&values)?;
        // Preserve the chained-comparison form for symbolic results so
        // that 1 != 2 != x prints as `1 != 2 != x`, not Unequal[...].
        if matches!(&result, Expr::FunctionCall { name, .. } if name == "Unequal")
        {
          return Ok(Expr::Comparison {
            operands: values,
            operators: operators.clone(),
          });
        }
        return Ok(result);
      }

      // For homogeneous monotone chains (all Less, all LessEqual, all
      // Greater, all GreaterEqual) detect contradictions across non-adjacent
      // numeric values. `1 < 3 < x < 2` is False because 3 < ... < 2 cannot
      // hold even though x is symbolic; without this scan the pair (3, x)
      // would short-circuit to an unevaluated form.
      use crate::functions::math_ast::try_eval_to_f64_with_infinity as try_eval_to_f64;
      let monotone_op = operators.first().filter(|_| {
        operators.iter().all(|op| {
          matches!(
            op,
            ComparisonOp::Less
              | ComparisonOp::LessEqual
              | ComparisonOp::Greater
              | ComparisonOp::GreaterEqual
          )
        }) && operators.iter().skip(1).all(|op| op == &operators[0])
      });
      if let Some(op) = monotone_op {
        let nums: Vec<(usize, f64)> = values
          .iter()
          .enumerate()
          .filter_map(|(i, e)| try_eval_to_f64(e).map(|n| (i, n)))
          .collect();
        for w in nums.windows(2) {
          // f64 conversion silently rounds 1-ULP differences above
          // ~2^53 to a tie — `2^60 < 2^60 + 1` would otherwise
          // short-circuit to False. Use the BigInt path when both
          // endpoints are exact integers; let the f64 fallback handle
          // any pair that mixes a non-integer.
          let exact = exact_integer_ord(&values[w[0].0], &values[w[1].0]);
          let ok = match (exact, op) {
            (Some(o), ComparisonOp::Less) => {
              matches!(o, std::cmp::Ordering::Less)
            }
            (Some(o), ComparisonOp::LessEqual) => {
              !matches!(o, std::cmp::Ordering::Greater)
            }
            (Some(o), ComparisonOp::Greater) => {
              matches!(o, std::cmp::Ordering::Greater)
            }
            (Some(o), ComparisonOp::GreaterEqual) => {
              !matches!(o, std::cmp::Ordering::Less)
            }
            (None, ComparisonOp::Less) => w[0].1 < w[1].1,
            (None, ComparisonOp::LessEqual) => w[0].1 <= w[1].1,
            (None, ComparisonOp::Greater) => w[0].1 > w[1].1,
            (None, ComparisonOp::GreaterEqual) => w[0].1 >= w[1].1,
            _ => true,
          };
          if !ok {
            return Ok(Expr::Identifier("False".to_string()));
          }
        }
      }

      // Evaluate comparison chain
      for i in 0..operators.len() {
        let left = &values[i];
        let right = &values[i + 1];
        let op = &operators[i];

        // Interval comparisons: Interval[{a,b}] < Interval[{c,d}], a scalar
        // counting as the degenerate interval {s,s}. A determinable result is
        // used directly; an indeterminate (overlapping) one falls through to
        // the normal logic, which leaves the comparison unevaluated.
        if (crate::functions::interval_ast::is_interval(left).is_some()
          || crate::functions::interval_ast::is_interval(right).is_some())
          && let Some(b) = interval_compare(op, left, right)
        {
          if !b {
            return Ok(Expr::Identifier("False".to_string()));
          }
          continue;
        }

        // Ordering comparisons on DateObject / TimeObject: compare by absolute
        // time (dates) or seconds-since-midnight (times of day). Both operands
        // must be the same kind. A determinable result is used; anything else
        // falls through to the normal (unevaluated) handling.
        if matches!(
          op,
          ComparisonOp::Less
            | ComparisonOp::LessEqual
            | ComparisonOp::Greater
            | ComparisonOp::GreaterEqual
        ) && let (Some((lv, lk)), Some((rv, rk))) = (
          crate::functions::datetime_ast::datetime_order_key(left),
          crate::functions::datetime_ast::datetime_order_key(right),
        ) && lk == rk
        {
          let ok = match op {
            ComparisonOp::Less => lv < rv,
            ComparisonOp::LessEqual => lv <= rv,
            ComparisonOp::Greater => lv > rv,
            ComparisonOp::GreaterEqual => lv >= rv,
            _ => true,
          };
          if !ok {
            return Ok(Expr::Identifier("False".to_string()));
          }
          continue;
        }

        // Enharmonic MusicPitch equality (WL 15): two `MusicPitch` objects
        // compare equal by their sounding MIDI pitch, so `MusicPitch["C#"]`
        // equals `MusicPitch["Db"]`. Only `Equal` has this rule — Wolfram
        // leaves `!=` on distinct pitch objects unevaluated.
        if matches!(op, ComparisonOp::Equal)
          && matches!(left, Expr::FunctionCall { name, .. } if name == "MusicPitch")
          && matches!(right, Expr::FunctionCall { name, .. } if name == "MusicPitch")
          && let (Some(lm), Some(rm)) = (
            crate::functions::music_ast::music_pitch_midi(left),
            crate::functions::music_ast::music_pitch_midi(right),
          )
        {
          if lm != rm {
            return Ok(Expr::Identifier("False".to_string()));
          }
          continue;
        }

        // Any other (in)equality touching a MusicPitch stays unevaluated
        // (after the structurally-identical collapse below), matching
        // Wolfram: `MusicPitch["C#"] != MusicPitch["Db"]` and
        // `MusicPitch["C"] == 5` both echo back.
        if matches!(op, ComparisonOp::Equal | ComparisonOp::NotEqual)
          && (matches!(left, Expr::FunctionCall { name, .. } if name == "MusicPitch")
            || matches!(right, Expr::FunctionCall { name, .. } if name == "MusicPitch"))
        {
          if expr_to_string(left) == expr_to_string(right) {
            if matches!(op, ComparisonOp::NotEqual) {
              return Ok(Expr::Identifier("False".to_string()));
            }
            continue;
          }
          return Ok(Expr::Comparison {
            operands: values,
            operators: operators.clone(),
          });
        }

        let result = match op {
          ComparisonOp::SameQ => {
            // SameQ tests structural identity, not numeric equality.
            // Integer[1] !== Real[1.] even though they are numerically equal.
            // Special case: a machine-precision Real and a BigFloat with
            // a precision tag inside the machine-precision band collapse
            // to the same f64 — Wolfram treats them as SameQ.
            expr_to_string(left) == expr_to_string(right)
              || crate::functions::boolean_ast::same_q_real_bigfloat(
                left, right,
              )
          }
          ComparisonOp::Equal => {
            if let Some(ord) = exact_integer_ord(left, right) {
              matches!(ord, std::cmp::Ordering::Equal)
            } else {
              // Two BigFloats with > ~16 digits of precision can carry
              // distinct stored digit strings even though both collapse
              // to the same f64. Compare the digit strings truncated to
              // the *shared* precision so genuinely-different literals
              // are False but values that agree to the shared precision
              // (e.g. `N[E, 100]` vs `N[E, 150]`) stay equal.
              if let (Expr::BigFloat(d_l, p_l), Expr::BigFloat(d_r, p_r)) =
                (left, right)
                && p_l.min(*p_r) > 16.0
              {
                // Compare to (shared - 1) digits so a 1-ULP difference at
                // the last shared digit stays equal-within-tolerance.
                let shared = (p_l.min(*p_r).floor() as usize).saturating_sub(1);
                if shared > 0
                  && !crate::functions::boolean_ast::bigfloat_digits_match_to(
                    d_l, d_r, shared,
                  )
                {
                  return Ok(Expr::Identifier("False".to_string()));
                }
              }
              if let (Some(l), Some(r)) =
                (try_eval_to_f64(left), try_eval_to_f64(right))
              {
                // Machine-precision Reals are compared up to the last ~7 bits
                // (matches wolframscript, which treats the f64 guard bits as
                // "insignificant"). Exact-vs-exact comparisons stay strict.
                // When a BigFloat with low precision p is involved, also
                // widen the tolerance to `10^-p · max(|l|, |r|)` so e.g.
                // `3.1416 == 3.14`2` is True (the shorter operand only
                // commits to ~2 significant decimals).
                let involves_real =
                  matches!(left, Expr::Real(_) | Expr::BigFloat(_, _))
                    || matches!(right, Expr::Real(_) | Expr::BigFloat(_, _))
                    || matches!(left, Expr::UnaryOp { operand, .. }
                if matches!(operand.as_ref(), Expr::Real(_)))
                    || matches!(right, Expr::UnaryOp { operand, .. }
                  if matches!(operand.as_ref(), Expr::Real(_)));
                if involves_real {
                  let mut tol =
                    f64::max(l.abs(), r.abs()) * (2.0_f64).powi(-46);
                  let bigfloat_prec = [left, right]
                    .iter()
                    .filter_map(|e| match e {
                      Expr::BigFloat(_, p) => Some(*p),
                      _ => None,
                    })
                    .reduce(f64::min);
                  if let Some(p) = bigfloat_prec {
                    // wolframscript's `Equal` is more lenient than
                    // `|a - b| < 10^-p`: it returns True whenever
                    // the precision of the difference is below 1
                    // (no significant digit). Practically that's
                    // roughly an extra decade of slack — use
                    // `10^-(p - 1)` so accuracy-form literals like
                    // `13.1416``4 == 13.1413``4` match (diff
                    // 3e-4 within 10^-4.12 ≈ 7.6e-5? no, within
                    // 10^-(5.12 - 1) ≈ 7.6e-4).
                    let widened_p = (p - 1.0).max(0.0);
                    let prec_tol =
                      f64::max(l.abs(), r.abs()) * 10.0_f64.powf(-widened_p);
                    if prec_tol > tol {
                      tol = prec_tol;
                    }
                  }
                  (l - r).abs() <= tol
                } else {
                  l == r
                }
              } else if expr_to_string(left) == expr_to_string(right) {
                true
              } else if let Some(ord) =
                crate::functions::quantity_ast::try_quantity_compare(
                  left, right,
                )
              {
                ord == std::cmp::Ordering::Equal
              } else if crate::functions::boolean_ast::all_components_equal(
                left, right,
              ) {
                // Same head and arity with every leaf determinably equal,
                // e.g. `RGBColor[0., 0., 1.] == RGBColor[0, 0, 1]`.
                true
              } else if has_free_symbols(left) || has_free_symbols(right) {
                // Symbolic: return unevaluated
                return Ok(Expr::Comparison {
                  operands: values,
                  operators: operators.clone(),
                });
              } else {
                false
              }
            }
          }
          ComparisonOp::UnsameQ => {
            // UnsameQ tests structural non-identity, not numeric inequality.
            expr_to_string(left) != expr_to_string(right)
          }
          ComparisonOp::NotEqual => {
            if let Some(ord) = exact_integer_ord(left, right) {
              !matches!(ord, std::cmp::Ordering::Equal)
            } else if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l != r
            } else if expr_to_string(left) == expr_to_string(right) {
              false
            } else if crate::functions::boolean_ast::all_components_equal(
              left, right,
            ) {
              // Determinably equal component-wise → `!=` is False.
              false
            } else if has_free_symbols(left) || has_free_symbols(right) {
              // Symbolic: return unevaluated
              return Ok(Expr::Comparison {
                operands: values,
                operators: operators.clone(),
              });
            } else {
              true
            }
          }
          ComparisonOp::Less => {
            if let Some(ord) = exact_integer_ord(left, right) {
              matches!(ord, std::cmp::Ordering::Less)
            } else if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l < r
            } else if let Some(ord) =
              crate::functions::quantity_ast::try_quantity_compare(left, right)
            {
              ord == std::cmp::Ordering::Less
            } else {
              // Return unevaluated comparison
              return Ok(Expr::Comparison {
                operands: values,
                operators: operators.clone(),
              });
            }
          }
          ComparisonOp::LessEqual => {
            if let Some(ord) = exact_integer_ord(left, right) {
              !matches!(ord, std::cmp::Ordering::Greater)
            } else if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l <= r
            } else if let Some(ord) =
              crate::functions::quantity_ast::try_quantity_compare(left, right)
            {
              ord != std::cmp::Ordering::Greater
            } else {
              return Ok(Expr::Comparison {
                operands: values,
                operators: operators.clone(),
              });
            }
          }
          ComparisonOp::Greater => {
            if let Some(ord) = exact_integer_ord(left, right) {
              matches!(ord, std::cmp::Ordering::Greater)
            } else if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l > r
            } else if let Some(ord) =
              crate::functions::quantity_ast::try_quantity_compare(left, right)
            {
              ord == std::cmp::Ordering::Greater
            } else {
              return Ok(Expr::Comparison {
                operands: values,
                operators: operators.clone(),
              });
            }
          }
          ComparisonOp::GreaterEqual => {
            if let Some(ord) = exact_integer_ord(left, right) {
              !matches!(ord, std::cmp::Ordering::Less)
            } else if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l >= r
            } else if let Some(ord) =
              crate::functions::quantity_ast::try_quantity_compare(left, right)
            {
              ord != std::cmp::Ordering::Less
            } else {
              return Ok(Expr::Comparison {
                operands: values,
                operators: operators.clone(),
              });
            }
          }
        };

        if !result {
          return Ok(Expr::Identifier("False".to_string()));
        }
      }
      Ok(Expr::Identifier("True".to_string()))
    }
    Expr::CompoundExpr(exprs) => {
      let mut result = Expr::Identifier("Null".to_string());
      let mut start_index = 0;
      'goto_loop: loop {
        for i in start_index..exprs.len() {
          match evaluate_expr_to_expr(&exprs[i]) {
            Ok(val) => result = val,
            Err(InterpreterError::GotoSignal(tag)) => {
              if let Some(label_idx) = find_label_index(exprs, &tag) {
                start_index = label_idx + 1;
                continue 'goto_loop;
              } else {
                return Err(InterpreterError::GotoSignal(tag));
              }
            }
            Err(e) => return Err(e),
          }
        }
        break;
      }
      // Trailing `Sequence[...]` splices into the surrounding context;
      // wolframscript yields `2` for `a; Sequence[1, 2]`. Empty sequences
      // collapse to Null.
      if let Expr::FunctionCall {
        name: n,
        args: seq_args,
      } = &result
        && n == "Sequence"
      {
        result = seq_args
          .last()
          .cloned()
          .unwrap_or_else(|| Expr::Identifier("Null".to_string()));
      }
      Ok(result)
    }
    Expr::Association(items) => {
      let mut deduped: Vec<(Expr, Expr)> = Vec::new();
      for (k, v) in items {
        let key = evaluate_expr_to_expr(k)?;
        let val = evaluate_expr_to_expr(v)?;
        if let Some(pos) = deduped.iter().position(|(ek, _)| {
          crate::evaluator::pattern_matching::expr_equal(ek, &key)
        }) {
          deduped[pos].1 = val;
        } else {
          deduped.push((key, val));
        }
      }
      Ok(Expr::Association(deduped))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let p = evaluate_expr_to_expr(pattern)?;
      let r = evaluate_expr_to_expr(replacement)?;
      Ok(Expr::Rule {
        pattern: Box::new(p),
        replacement: Box::new(r),
      })
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      // RuleDelayed has HoldRest: evaluate the pattern (LHS) but hold the
      // replacement (RHS) so it re-evaluates per match.
      let p = evaluate_expr_to_expr(pattern)?;
      Ok(Expr::RuleDelayed {
        pattern: Box::new(p),
        replacement: replacement.clone(),
      })
    }
    Expr::ReplaceAll { expr: e, rules } => {
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      let evaluated_rules = evaluate_expr_to_expr(rules)?;
      // Unwrap Dispatch[rules] → use inner rules
      let unwrapped =
        if let Expr::FunctionCall { name, args } = &evaluated_rules {
          if name == "Dispatch" && args.len() == 1 {
            &args[0]
          } else {
            &evaluated_rules
          }
        } else {
          &evaluated_rules
        };
      // Validate rules shape. Wolfram emits ReplaceAll::reps and returns
      // the call unevaluated when the rules slot isn't a Rule/RuleDelayed
      // (or list of such, possibly nested for multi-solution forms).
      // Mirror that here so `eqs /. sol` with `sol` unbound surfaces the
      // same diagnostic instead of silently returning `eqs`.
      if !is_valid_replace_rules(unwrapped) {
        crate::emit_message(&format!(
          "ReplaceAll::reps: {{{}}} is neither a list of replacement rules nor a valid dispatch table, and so cannot be used for replacing.",
          crate::syntax::expr_to_output(unwrapped),
        ));
        return Ok(Expr::ReplaceAll {
          expr: Box::new(evaluated_expr),
          rules: Box::new(evaluated_rules),
        });
      }
      let result = apply_replace_all_ast(&evaluated_expr, unwrapped)?;
      Err(InterpreterError::TailCall(Box::new(result)))
    }
    Expr::ReplaceRepeated { expr: e, rules } => {
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      let evaluated_rules = evaluate_expr_to_expr(rules)?;
      // Unwrap Dispatch[rules] → use inner rules
      let unwrapped =
        if let Expr::FunctionCall { name, args } = &evaluated_rules {
          if name == "Dispatch" && args.len() == 1 {
            &args[0]
          } else {
            &evaluated_rules
          }
        } else {
          &evaluated_rules
        };
      // Same `reps` shape-validation as ReplaceAll: emit the matching
      // ReplaceRepeated::reps message and return the call unevaluated
      // when the rules slot isn't a Rule / RuleDelayed (or list of
      // such, possibly nested for multi-solution forms).
      if !is_valid_replace_rules(unwrapped) {
        crate::emit_message(&format!(
          "ReplaceRepeated::reps: {{{}}} is neither a list of replacement rules nor a valid dispatch table, and so cannot be used for replacing.",
          crate::syntax::expr_to_output(unwrapped),
        ));
        return Ok(Expr::ReplaceRepeated {
          expr: Box::new(evaluated_expr),
          rules: Box::new(evaluated_rules),
        });
      }
      apply_replace_repeated_ast(&evaluated_expr, unwrapped)
    }
    Expr::Map { func, list } => {
      let evaluated_list = evaluate_expr_to_expr(list)?;
      apply_map_ast(func, &evaluated_list)
    }
    Expr::Apply { func, list } => {
      // Apply does not hold its head, so evaluate it (matching the function
      // form `Apply[f, list]`). This normalizes a compound head such as
      // `f + g` to `Plus[f, g]`, so `(f + g) @@ {1, 2}` yields `(f + g)[1, 2]`
      // rather than mis-associating as `f + g[1, 2]`.
      let evaluated_func = evaluate_expr_to_expr(func)?;
      let evaluated_list = evaluate_expr_to_expr(list)?;
      apply_apply_ast(&evaluated_func, &evaluated_list)
    }
    Expr::MapApply { func, list } => {
      let evaluated_list = evaluate_expr_to_expr(list)?;
      apply_map_apply_ast(func, &evaluated_list)
    }
    Expr::PrefixApply { func, arg } => {
      // f @ x is equivalent to f[x]. Per Wolfram semantics the head is
      // evaluated first, so a dynamic head like `If[cond, A, B]@x` reduces
      // to `B[x]` rather than being treated as a curried call
      // `If[cond, A, B, x]`.
      let evaluated_arg = evaluate_expr_to_expr(arg)?;
      let evaluated_func = if matches!(func.as_ref(), Expr::FunctionCall { .. })
      {
        evaluate_expr_to_expr(func)?
      } else {
        (**func).clone()
      };
      apply_function_to_arg(&evaluated_func, &evaluated_arg)
    }
    Expr::Postfix { expr: e, func } => {
      // `expr // f` is `f[expr]`. If `f` is a head with HoldAll/
      // HoldAllComplete (e.g. `Hold`, `FullForm`), pre-evaluating expr
      // would defeat the hold and let side effects fire. Normalize the
      // Postfix chain to a `FunctionCall` and let the regular evaluator
      // apply f's hold attributes — including for chained postfix like
      // `e // OutputForm // MakeBoxes`, where the inner Postfix needs
      // to surface as `OutputForm[e]` (a FunctionCall) before MakeBoxes
      // sees it.
      let func_holds_all = matches!(
        func.as_ref(),
        Expr::Identifier(name)
          if has_hold_attribute(name, "HoldAll")
            || has_hold_attribute(name, "HoldAllComplete")
      );
      if func_holds_all && let Expr::Identifier(name) = func.as_ref() {
        // Convert any nested `Postfix(e, f)` to `FunctionCall { name: f,
        // args: [e] }` recursively so the held argument is in normal
        // structural form.
        fn normalize_postfix(e: &Expr) -> Expr {
          if let Expr::Postfix {
            expr: inner_e,
            func: inner_f,
          } = e
            && let Expr::Identifier(inner_name) = inner_f.as_ref()
          {
            Expr::FunctionCall {
              name: inner_name.clone(),
              args: vec![normalize_postfix(inner_e)].into(),
            }
          } else {
            e.clone()
          }
        }
        let arg = normalize_postfix(e);
        return evaluate_expr_to_expr(&Expr::FunctionCall {
          name: name.clone(),
          args: vec![arg].into(),
        });
      }
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      // `expr // f` is `f[expr]`. Like PrefixApply, the head must be
      // evaluated first so a dynamic head like `expr // If[cond, A, B]`
      // reduces before application.
      let evaluated_func = if matches!(func.as_ref(), Expr::FunctionCall { .. })
      {
        evaluate_expr_to_expr(func)?
      } else {
        (**func).clone()
      };
      apply_postfix_ast(&evaluated_expr, &evaluated_func)
    }
    Expr::Part { expr: e, index } => {
      // Fast path: `var[[k]]` where `var` is an Identifier bound to a
      // List in ENV and `k` is an Integer literal (or evaluates to one
      // without needing the surrounding Part machinery). Clone only the
      // indexed element instead of the whole stored list.
      //
      // The general path below cleans `eval_part_base` -> `stored.clone()`
      // -> `extract_part_ast` and pays the full O(N) list copy on every
      // invocation. asciiLess in build_summary.wls hits this path on
      // every comparison (`a[[i]]`, `b[[i]]` over ~30 chars × thousands
      // of compares).
      if let Expr::Identifier(var_name) = e.as_ref() {
        let idx_expr = match index.as_ref() {
          Expr::Integer(n) => Some(*n),
          other => match evaluate_expr_to_expr(other) {
            Ok(Expr::Integer(n)) => Some(n),
            _ => None,
          },
        };
        if let Some(i) = idx_expr {
          let elem = ENV.with(|env| -> Option<Expr> {
            let env = env.borrow();
            match env.get(var_name) {
              Some(StoredValue::ExprVal(Expr::List(items))) => {
                let len = items.len() as i128;
                let pos = if i > 0 {
                  if i > len {
                    return None;
                  }
                  (i as usize) - 1
                } else if i < 0 {
                  let p = len + i;
                  if p < 0 {
                    return None;
                  }
                  p as usize
                } else {
                  return None;
                };
                items.get(pos).cloned()
              }
              _ => None,
            }
          });
          if let Some(v) = elem {
            return Ok(v);
          }
        }
      }

      // Track Part nesting depth for Part::partd warnings
      PART_DEPTH.with(|d| *d.borrow_mut() += 1);

      // Collect the full chain of Part indices (innermost first)
      // e.g. Part[Part[Part[base, i], j], k] -> base, [i, j, k]
      let mut indices_unevaluated = vec![index.as_ref().clone()];
      let mut base_expr = e.as_ref();
      while let Expr::Part {
        expr: inner_e,
        index: inner_idx,
      } = base_expr
      {
        indices_unevaluated.push(inner_idx.as_ref().clone());
        base_expr = inner_e.as_ref();
      }
      indices_unevaluated.reverse(); // now [i, j, k] order (outermost to innermost)

      // Evaluate all indices
      let mut indices = Vec::with_capacity(indices_unevaluated.len());
      for idx in &indices_unevaluated {
        if let Expr::FunctionCall { name, args } = idx
          && name == "Span"
          && (args.len() == 2 || args.len() == 3)
        {
          let mut evaluated_args = Vec::with_capacity(args.len());
          for arg in args {
            if matches!(arg, Expr::Identifier(s) if s == "All") {
              evaluated_args.push(arg.clone());
            } else {
              evaluated_args.push(evaluate_expr_to_expr(arg)?);
            }
          }
          indices.push(Expr::FunctionCall {
            name: "Span".to_string(),
            args: evaluated_args.into(),
          });
        } else {
          indices.push(evaluate_expr_to_expr(idx)?);
        }
      }

      // Check if any index needs the recursive apply_part_indices path:
      // - `All` always needs it (even alone, to return the expression itself)
      // - `List` / `Span` in non-last position need it (to map remaining
      //   indices over each extracted element rather than extract sequentially)
      let needs_mapping = indices.iter().enumerate().any(|(i, idx)| {
        matches!(idx, Expr::Identifier(s) if s == "All")
          || (i + 1 < indices.len()
            && (matches!(idx, Expr::List(_))
              || matches!(idx, Expr::FunctionCall { name, .. } if name == "Span")))
      });

      // Set when a multi-index extraction reaches an atom with indices left
      // over — the spec is deeper than the object (Part::partd), as opposed
      // to an out-of-bounds index (Part::partw, already emitted).
      let mut hit_atom_mid = false;
      let result = if needs_mapping {
        // All requires collecting indices and mapping — must clone base
        let base_val = eval_part_base(base_expr)?;
        apply_part_indices(&base_val, &indices)?
      } else {
        // Fast path: no All, use original optimized approach
        let base_val = eval_part_base(base_expr)?;
        let mut result = extract_part_ast(&base_val, &indices[0])?;

        if indices.len() > 1 {
          // Multi-index Part: if extraction returns an unevaluated Part at
          // any level, the spec is deeper than the object — return the full
          // Part expression unevaluated (matching wolframscript Part::partd).
          let mut part_too_deep = false;
          for idx in &indices[1..] {
            if let Expr::Part { .. } = &result {
              part_too_deep = true;
              break;
            }
            if matches!(
              &result,
              Expr::Identifier(_)
                | Expr::Integer(_)
                | Expr::BigInteger(_)
                | Expr::Real(_)
                | Expr::String(_)
            ) {
              part_too_deep = true;
              hit_atom_mid = true;
              break;
            }
            result = extract_part_ast(&result, idx)?;
          }
          if let Expr::Part { .. } = &result {
            part_too_deep = true;
          }
          if part_too_deep {
            result = base_val;
            for idx in &indices {
              result = Expr::Part {
                expr: Box::new(result),
                index: Box::new(idx.clone()),
              };
            }
          }
        }

        result
      };
      PART_DEPTH.with(|d| *d.borrow_mut() -= 1);
      // Part::partd / Part::pspec1: warn only at the outermost Part level
      let at_outermost = PART_DEPTH.with(|d| *d.borrow() == 0);
      if at_outermost && let Expr::Part { .. } = &result {
        // A string part spec on a non-association is never applicable,
        // regardless of the base (wolframscript: Part::pspec1).
        if let Some(Expr::String(s)) =
          indices.iter().find(|i| matches!(i, Expr::String(_)))
        {
          crate::emit_message(&format!(
            "Part::pspec1: Part specification {} is not applicable.",
            s
          ));
          return Ok(result);
        }
        let base = get_part_base(&result);
        if hit_atom_mid
          || matches!(
            base,
            Expr::Identifier(_)
              | Expr::Integer(_)
              | Expr::BigInteger(_)
              | Expr::Real(_)
              | Expr::String(_)
          )
        {
          let part_str = crate::syntax::format_expr(
            &result,
            crate::syntax::ExprForm::Output,
          );
          crate::emit_message(&format!(
            "Part::partd: Part specification {} is longer than depth of object.",
            part_str
          ));
        }
      }
      Ok(result)
    }
    Expr::Function { body } => {
      // Function has HoldAll, but Evaluate[...] inside the body forces
      // evaluation of its argument before the function is held. Match
      // wolframscript: Evaluate[expr]& becomes Function[expr_evaluated].
      let new_body = unwrap_evaluate_in_body(body)?;
      Ok(Expr::Function {
        body: Box::new(new_body),
      })
    }
    Expr::NamedFunction {
      params,
      body,
      bracketed,
    } => {
      // Return named function as-is
      Ok(Expr::NamedFunction {
        params: params.clone(),
        body: body.clone(),
        bracketed: *bracketed,
      })
    }
    Expr::Pattern {
      name,
      head,
      blank_type,
    } => Ok(Expr::Pattern {
      name: name.clone(),
      head: head.clone(),
      blank_type: *blank_type,
    }),
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Ok(Expr::PatternOptional {
      name: name.clone(),
      head: head.clone(),
      default: default.clone(),
    }),
    Expr::PatternTest {
      name,
      head,
      blank_type,
      test,
    } => Ok(Expr::PatternTest {
      name: name.clone(),
      head: head.clone(),
      blank_type: *blank_type,
      test: test.clone(),
    }),
    Expr::Raw(s) => {
      // Fallback: parse and evaluate the raw string
      let parsed = string_to_expr(s)?;
      Err(InterpreterError::TailCall(Box::new(parsed)))
    }
    Expr::Image { .. } => Ok(expr.clone()),
    Expr::Graphics { .. } => Ok(expr.clone()),
    Expr::CurriedCall { func, args } => {
      // Evaluate the curried call: f[a][b] -> apply f[a] to args.
      // Honour Function[{params}, body, attrs] hold attributes so that
      // Function[..., HoldAll][1+1] sees its arg unevaluated.
      let evaluated_func = evaluate_expr_to_expr(func)?;
      let hold_attrs = function_hold_attributes(&evaluated_func);
      let mut prepared: Vec<Expr> = Vec::with_capacity(args.len());
      for (i, a) in args.iter().enumerate() {
        let hold =
          hold_attrs.0 || (hold_attrs.1 && i == 0) || (hold_attrs.2 && i > 0);
        if hold {
          prepared.push(a.clone());
        } else {
          prepared.push(evaluate_expr_to_expr(a)?);
        }
      }
      apply_curried_call(&evaluated_func, &prepared)
    }
  }
}

/// Inspect a stored Function value for Hold attributes.
/// Returns (hold_all, hold_first, hold_rest).
fn function_hold_attributes(expr: &Expr) -> (bool, bool, bool) {
  let attrs = match expr {
    Expr::FunctionCall { name, args }
      if name == "Function" && args.len() >= 3 =>
    {
      Some(&args[2])
    }
    _ => None,
  };
  let has = |needle: &str| -> bool {
    match attrs {
      Some(Expr::Identifier(s)) => s == needle,
      Some(Expr::List(items)) => items
        .iter()
        .any(|e| matches!(e, Expr::Identifier(s) if s == needle)),
      _ => false,
    }
  };
  (
    has("HoldAll") || has("HoldAllComplete"),
    has("HoldFirst"),
    has("HoldRest"),
  )
}

/// Recursively unwrap `Evaluate[expr]` subexpressions inside a Function body
/// (or any held expression), evaluating their arguments. This implements
/// Wolfram's rule that Evaluate forces evaluation through HoldAll wrappers.
fn unwrap_evaluate_in_body(body: &Expr) -> Result<Expr, InterpreterError> {
  if let Expr::FunctionCall { name, args } = body
    && name == "Evaluate"
    && args.len() == 1
  {
    return evaluate_expr_to_expr(&args[0]);
  }
  // Recurse into common compound forms so nested Evaluate gets evaluated too.
  match body {
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> = args
        .iter()
        .map(unwrap_evaluate_in_body)
        .collect::<Result<_, _>>()?;
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      })
    }
    Expr::List(items) => {
      let new_items: Vec<Expr> = items
        .iter()
        .map(unwrap_evaluate_in_body)
        .collect::<Result<_, _>>()?;
      Ok(Expr::List(new_items.into()))
    }
    Expr::BinaryOp { op, left, right } => Ok(Expr::BinaryOp {
      op: *op,
      left: Box::new(unwrap_evaluate_in_body(left)?),
      right: Box::new(unwrap_evaluate_in_body(right)?),
    }),
    Expr::UnaryOp { op, operand } => Ok(Expr::UnaryOp {
      op: *op,
      operand: Box::new(unwrap_evaluate_in_body(operand)?),
    }),
    other => Ok(other.clone()),
  }
}

/// Determines whether a comparison chain should stay as a single node or
/// split into pairwise `&&`. wolframscript's rules:
///   - Homogeneous operators → keep chain.
///   - Presence of any NotEqual/UnsameQ in a mixed chain → split.
///   - Mixing a Less-direction op (`<`, `<=`) with a Greater-direction op
///     (`>`, `>=`) → split (pairwise makes the directionality explicit).
///   - Otherwise (Equal/SameQ mixed with one direction of inequalities,
///     or only Less-direction / only Greater-direction ops) → keep as a
///     single chain.  These print either as `a == b == c` or `Inequality[…]`.
/// Numeric range [lo, hi] of an expression for interval comparison: an
/// `Interval[{a,b}, …]` collapses to its overall min/max, and a scalar number
/// `s` becomes the degenerate range [s, s]. Returns None if any bound is not a
/// finite real (e.g. a symbol).
fn interval_numeric_bounds(expr: &Expr) -> Option<(f64, f64)> {
  use crate::functions::math_ast::try_eval_to_f64_with_infinity as to_f64;
  if let Some(subs) = crate::functions::interval_ast::is_interval(expr) {
    if subs.is_empty() {
      return None;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (l, h) in &subs {
      lo = lo.min(to_f64(l)?);
      hi = hi.max(to_f64(h)?);
    }
    Some((lo, hi))
  } else {
    let v = to_f64(expr)?;
    Some((v, v))
  }
}

/// Compare two values when at least one is an Interval, following Wolfram:
/// `X < Y` is True when X lies entirely below Y, False when entirely at/above,
/// and indeterminate (None) when the ranges overlap. `==`/`!=` resolve only for
/// disjoint ranges. None lets the caller leave the comparison unevaluated.
fn interval_compare(
  op: &ComparisonOp,
  left: &Expr,
  right: &Expr,
) -> Option<bool> {
  use ComparisonOp::*;
  let (xlo, xhi) = interval_numeric_bounds(left)?;
  let (ylo, yhi) = interval_numeric_bounds(right)?;
  let disjoint = xhi < ylo || yhi < xlo;
  match op {
    Less => {
      if xhi < ylo {
        Some(true)
      } else if xlo >= yhi {
        Some(false)
      } else {
        None
      }
    }
    LessEqual => {
      if xhi <= ylo {
        Some(true)
      } else if xlo > yhi {
        Some(false)
      } else {
        None
      }
    }
    Greater => {
      if xlo > yhi {
        Some(true)
      } else if xhi <= ylo {
        Some(false)
      } else {
        None
      }
    }
    GreaterEqual => {
      if xlo >= yhi {
        Some(true)
      } else if xhi < ylo {
        Some(false)
      } else {
        None
      }
    }
    // Equal/Unequal resolve only when the ranges cannot overlap; identical
    // intervals are still caught by the structural string-equality fallback.
    Equal if disjoint => Some(false),
    NotEqual if disjoint => Some(true),
    _ => None,
  }
}

fn all_same_comparison_family(ops: &[ComparisonOp]) -> bool {
  use ComparisonOp::*;
  let mixed = ops.iter().skip(1).any(|op| op != &ops[0]);
  if !mixed {
    return true;
  }
  let mut has_unequal = false;
  let mut has_less = false;
  let mut has_greater = false;
  for op in ops {
    match op {
      NotEqual | UnsameQ => has_unequal = true,
      Less | LessEqual => has_less = true,
      Greater | GreaterEqual => has_greater = true,
      Equal | SameQ => {}
    }
  }
  if has_unequal {
    return false;
  }
  if has_less && has_greater {
    return false;
  }
  true
}

/// Check if an expression contains free symbols (unbound identifiers).
/// Known constants (True, False, I, Pi, etc.) are NOT free symbols.
pub fn has_free_symbols(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(name) => !matches!(
      name.as_str(),
      "True"
        | "False"
        | "Null"
        | "I"
        | "Pi"
        | "E"
        | "Degree"
        | "Infinity"
        | "ComplexInfinity"
        | "Indeterminate"
        | "Nothing"
    ),
    Expr::Integer(_)
    | Expr::BigInteger(_)
    | Expr::Real(_)
    | Expr::BigFloat(_, _)
    | Expr::String(_)
    | Expr::Constant(_)
    | Expr::Slot(_)
    | Expr::SlotSequence(_) => false,
    Expr::List(items) => items.iter().any(has_free_symbols),
    Expr::BinaryOp { left, right, .. } => {
      has_free_symbols(left) || has_free_symbols(right)
    }
    Expr::UnaryOp { operand, .. } => has_free_symbols(operand),
    Expr::FunctionCall { name, args } => {
      has_free_symbols(&Expr::Identifier(name.clone()))
        || args.iter().any(has_free_symbols)
    }
    Expr::Comparison { operands, .. } => operands.iter().any(has_free_symbols),
    Expr::Rule {
      pattern,
      replacement,
    } => has_free_symbols(pattern) || has_free_symbols(replacement),
    Expr::Association(pairs) => pairs
      .iter()
      .any(|(k, v)| has_free_symbols(k) || has_free_symbols(v)),
    // Patterns like a_ / a_Integer / a_. are symbolic — they can match
    // anything, so Unequal[a_, b_] must stay unevaluated (matches Wolfram).
    Expr::Pattern { .. }
    | Expr::PatternOptional { .. }
    | Expr::PatternTest { .. } => true,
    _ => false,
  }
}
