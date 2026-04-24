#[allow(unused_imports)]
use super::*;

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
fn evaluate_args_with_hold(
  name: &str,
  args: &[Expr],
) -> Result<Vec<Expr>, InterpreterError> {
  let hold_all = has_hold_attribute(name, "HoldAll")
    || has_hold_attribute(name, "HoldAllComplete");
  let hold_first = has_hold_attribute(name, "HoldFirst");
  let hold_rest = has_hold_attribute(name, "HoldRest");

  if hold_all {
    Ok(args.to_vec())
  } else if hold_first && !args.is_empty() {
    let mut result = vec![args[0].clone()];
    for arg in &args[1..] {
      result.push(evaluate_expr_to_expr(arg)?);
    }
    Ok(result)
  } else if hold_rest && !args.is_empty() {
    let mut result = vec![evaluate_expr_to_expr(&args[0])?];
    result.extend(args[1..].to_vec());
    Ok(result)
  } else {
    args
      .iter()
      .map(evaluate_expr_to_expr)
      .collect::<Result<_, _>>()
  }
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
    args: vec![num(r), num(g), num(b)],
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
    args: vec![num(g)],
  }
}

/// Map named color identifiers to their Wolfram Language evaluation result.
/// Basic colors → RGBColor[r, g, b] or GrayLevel[g]
/// Light* colors → RGBColor[r, g, b] or GrayLevel[g]
pub fn named_color_expr_pub(name: &str) -> Option<Expr> {
  named_color_expr(name)
}

fn named_color_expr(name: &str) -> Option<Expr> {
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
pub fn prepare_iterating_function_args(
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
      result.push(Expr::List(new_items));
    } else {
      result.push(evaluate_expr_to_expr(arg)?);
    }
  }

  Ok(result)
}

/// Early dispatch for FunctionCall in evaluate_expr_to_expr — handles held functions
/// before argument evaluation. Returns Some(result) if handled, None otherwise.
#[inline(never)]
pub fn evaluate_expr_to_expr_early_dispatch(
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
            Ok(val) => result = val,
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
      return Ok(Some(result));
    }
    "Pattern" if args.len() == 2 => {
      return Ok(Some(evaluate_pattern_function_ast(args)?));
    }
    "RuleDelayed" if args.len() == 2 => {
      return Ok(Some(evaluate_rule_delayed_ast(args)?));
    }
    "AbsoluteTiming" | "Timing" if args.len() == 1 => {
      let start = std::time::Instant::now();
      let result = evaluate_expr_to_expr(&args[0])?;
      let elapsed = start.elapsed().as_secs_f64();
      return Ok(Some(Expr::List(vec![Expr::Real(elapsed), result])));
    }
    "RepeatedTiming" if args.len() == 1 => {
      let mut times = Vec::new();
      let mut last_result = Expr::Identifier("Null".to_string());
      let overall_start = std::time::Instant::now();
      for _ in 0..100 {
        let start = std::time::Instant::now();
        last_result = evaluate_expr_to_expr(&args[0])?;
        times.push(start.elapsed().as_secs_f64());
        if times.len() >= 3 && overall_start.elapsed().as_secs_f64() > 0.5 {
          break;
        }
      }
      times.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let median = times[times.len() / 2];
      return Ok(Some(Expr::List(vec![Expr::Real(median), last_result])));
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

/// Evaluate an Expr AST directly and return a string representation.
/// Delegates to `evaluate_expr_to_expr` and converts the result to a string.
pub fn evaluate_expr(expr: &Expr) -> Result<String, InterpreterError> {
  Ok(expr_to_string(&evaluate_expr_to_expr(expr)?))
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
  let mut current = expr.clone();
  loop {
    match evaluate_expr_to_expr_inner(&current) {
      Err(InterpreterError::TailCall(next)) => {
        current = *next;
      }
      result => return result,
    }
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
          StoredValue::ExprVal(e) => Ok(e),
          StoredValue::Raw(val) => {
            // Parse the stored value back to Expr
            string_to_expr(&val)
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
              Expr::List(vec![
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
              ]),
              Expr::String("Instant".to_string()),
              Expr::String("Gregorian".to_string()),
              Expr::Real(tz_offset_hours),
            ],
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
              Expr::List(vec![
                Expr::Integer(now.get_full_year() as i128),
                Expr::Integer((now.get_month() + 1) as i128),
                Expr::Integer(now.get_date() as i128),
                Expr::Integer(now.get_hours() as i128),
                Expr::Integer(now.get_minutes() as i128),
                Expr::Real(seconds),
              ]),
              Expr::String("Instant".to_string()),
              Expr::String("Gregorian".to_string()),
              Expr::Real(tz_offset_hours),
            ],
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
              Expr::List(vec![
                Expr::Integer(
                  date.format("%Y").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Integer(
                  date.format("%m").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Integer(
                  date.format("%d").to_string().parse::<i128>().unwrap(),
                ),
              ]),
              Expr::String("Day".to_string()),
            ],
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
              Expr::List(vec![
                Expr::Integer(d.get_full_year() as i128),
                Expr::Integer((d.get_month() + 1) as i128),
                Expr::Integer(d.get_date() as i128),
              ]),
              Expr::String("Day".to_string()),
            ],
          });
        }
        // Handle system $ variables
        if name.starts_with('$')
          && let Some(val) = get_system_variable(name)
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
            args: vec![Expr::Identifier("Large".to_string())],
          });
        }
        // Thin → Thickness[Tiny]
        if name == "Thin" {
          return Ok(Expr::FunctionCall {
            name: "Thickness".to_string(),
            args: vec![Expr::Identifier("Tiny".to_string())],
          });
        }
        // Dashed → Dashing[{Small, Small}]
        if name == "Dashed" {
          return Ok(Expr::FunctionCall {
            name: "Dashing".to_string(),
            args: vec![Expr::List(vec![
              Expr::Identifier("Small".to_string()),
              Expr::Identifier("Small".to_string()),
            ])],
          });
        }
        // Dotted → Dashing[{0, Small}]
        if name == "Dotted" {
          return Ok(Expr::FunctionCall {
            name: "Dashing".to_string(),
            args: vec![Expr::List(vec![
              Expr::Integer(0),
              Expr::Identifier("Small".to_string()),
            ])],
          });
        }
        // DotDashed → Dashing[{0, Small, Small, Small}]
        if name == "DotDashed" {
          return Ok(Expr::FunctionCall {
            name: "Dashing".to_string(),
            args: vec![Expr::List(vec![
              Expr::Integer(0),
              Expr::Identifier("Small".to_string()),
              Expr::Identifier("Small".to_string()),
              Expr::Identifier("Small".to_string()),
            ])],
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
    Expr::Constant(name) => Ok(Expr::Constant(name.clone())),
    Expr::List(items) => {
      let mut evaluated: Vec<Expr> = Vec::with_capacity(items.len());
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
      Ok(Expr::List(evaluated))
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
              args: new_args,
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
              op: crate::syntax::BinaryOperator::Plus,
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
              op: crate::syntax::BinaryOperator::Plus,
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
                args: vec![it.clone()],
              })?;
              results.push(r);
            }
            return Ok(Expr::List(results));
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
            // At least one rule exists for this head — Woxi has no per-pattern
            // removal yet, so clear all of them for now (matches the common
            // 'f[x_] =. after f[x_] := ...' case where exactly one rule was
            // defined). Emit no warning, return Null.
            let _ = lhs_args;
            crate::FUNC_DEFS.with(|m| {
              m.borrow_mut().remove(head);
            });
            return Ok(Expr::Identifier("Null".to_string()));
          }
          return Ok(Expr::Identifier("Null".to_string()));
        }
        // Special handling for Information/Definition/FullDefinition - hold argument unevaluated
        if (name == "Information"
          || name == "Definition"
          || name == "FullDefinition")
          && args.len() == 1
        {
          return dispatch::evaluate_function_call_ast(name, args);
        }
        // Special handling for AddTo, SubtractFrom, TimesBy, DivideBy - x += y, x -= y, etc.
        if (name == "AddTo"
          || name == "SubtractFrom"
          || name == "TimesBy"
          || name == "DivideBy")
          && args.len() == 2
        {
          let op = match name.as_str() {
            "AddTo" => crate::syntax::BinaryOperator::Plus,
            "SubtractFrom" => crate::syntax::BinaryOperator::Minus,
            "TimesBy" => crate::syntax::BinaryOperator::Times,
            "DivideBy" => crate::syntax::BinaryOperator::Divide,
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
        // Special handling for AppendTo, PrependTo - x = Append[x, elem]
        if (name == "AppendTo" || name == "PrependTo")
          && args.len() == 2
          && let Expr::Identifier(var_name) = &args[0]
        {
          let elem = evaluate_expr_to_expr(&args[1])?;
          let current = ENV.with(|e| e.borrow().get(var_name).cloned());
          let mut current_val = match current {
            Some(StoredValue::ExprVal(e)) => e,
            Some(StoredValue::Raw(s)) => {
              crate::syntax::string_to_expr(&s).unwrap_or(Expr::List(vec![]))
            }
            _ => {
              return Err(InterpreterError::EvaluationError(format!(
                "{} requires a variable with a list value",
                name
              )));
            }
          };
          let new_val = match &mut current_val {
            Expr::List(items) => {
              let mut items = std::mem::take(items);
              if name == "AppendTo" {
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
              if name == "AppendTo" {
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
          ENV.with(|e| {
            e.borrow_mut().insert(
              var_name.clone(),
              StoredValue::Raw(crate::syntax::expr_to_string(&new_val)),
            );
          });
          return Ok(new_val);
        }
        // Special handling for AssociateTo - x = Append[x, key -> val]
        if name == "AssociateTo"
          && args.len() == 2
          && let Expr::Identifier(var_name) = &args[0]
        {
          let rule = evaluate_expr_to_expr(&args[1])?;
          let (key, val) = match &rule {
            Expr::Rule {
              pattern,
              replacement,
            } => (pattern.as_ref().clone(), replacement.as_ref().clone()),
            _ => {
              // Not a rule — return unevaluated
              let mut eval_args = vec![args[0].clone()];
              eval_args.push(rule);
              return Ok(Expr::FunctionCall {
                name: "AssociateTo".to_string(),
                args: eval_args,
              });
            }
          };
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
                  let mut eval_args = vec![args[0].clone()];
                  eval_args.push(Expr::Rule {
                    pattern: Box::new(key),
                    replacement: Box::new(val),
                  });
                  return Ok(Expr::FunctionCall {
                    name: "AssociateTo".to_string(),
                    args: eval_args,
                  });
                }
              }
            }
            _ => {
              let mut eval_args = vec![args[0].clone()];
              eval_args.push(Expr::Rule {
                pattern: Box::new(key),
                replacement: Box::new(val),
              });
              return Ok(Expr::FunctionCall {
                name: "AssociateTo".to_string(),
                args: eval_args,
              });
            }
          };
          if let Some(pos) = items.iter().position(|(k, _)| {
            crate::evaluator::pattern_matching::expr_equal(k, &key)
          }) {
            items[pos].1 = val;
          } else {
            items.push((key, val));
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
                    args: vec![args[0].clone(), key],
                  });
                }
              }
            }
            _ => {
              return Ok(Expr::FunctionCall {
                name: "KeyDropFrom".to_string(),
                args: vec![args[0].clone(), key],
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
        // Special handling for Catch[expr] and Catch[expr, form]
        if name == "Catch" && !args.is_empty() && args.len() <= 2 {
          let tag_pattern = if args.len() == 2 {
            Some(evaluate_expr_to_expr(&args[1])?)
          } else {
            None
          };
          match evaluate_expr_to_expr(&args[0]) {
            Ok(result) => return Ok(result),
            Err(InterpreterError::ThrowValue(val, thrown_tag)) => {
              // If Catch has a tag pattern, check if it matches
              if let Some(ref pattern) = tag_pattern {
                if let Some(ref tag) = thrown_tag
                  && crate::syntax::expr_to_string(pattern)
                    == crate::syntax::expr_to_string(tag)
                {
                  return Ok(*val);
                }
                // Tag doesn't match - re-throw
                return Err(InterpreterError::ThrowValue(val, thrown_tag));
              }
              // No tag pattern - catch everything
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
        // Pause[n] - sleep for n seconds and return Null
        if name == "Pause" && args.len() == 1 {
          if let Ok(val) = evaluate_expr_to_expr(&args[0])
            && let Some(secs) =
              crate::functions::math_ast::try_eval_to_f64(&val)
            && secs > 0.0
          {
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
          }
          return Ok(Expr::Identifier("Null".to_string()));
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
        // Special handling for Check[expr, failexpr]
        if name == "Check" && args.len() == 2 {
          let warnings_before = crate::get_captured_warnings().len();
          match evaluate_expr_to_expr(&args[0]) {
            Ok(result) => {
              let warnings_after = crate::get_captured_warnings().len();
              if warnings_after > warnings_before {
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
        // Special handling for Switch - lazy evaluation of branches
        if name == "Switch" && args.len() >= 3 {
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
        || name == "BooleanTable"
        // Manipulate holds its body and variable specs so that controls
        // like Plot[Sin[a*x], {x, 0, 6}] aren't evaluated with `a` free
        // (matching Wolfram Language notebook behavior).
        || name == "Manipulate"
        {
          // Flatten Sequence even in held args (unless SequenceHold)
          let args = flatten_sequences(name, args);
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
              let evaluated_args: Vec<Expr> = args
                .iter()
                .map(evaluate_expr_to_expr)
                .collect::<Result<_, _>>()?;
              return apply_curried_call(stored_expr, &evaluated_args);
            }
            _ => {}
          }
        }
        // Variable holds a symbol name (e.g. t = Flatten) — re-dispatch as that function
        if let Some(resolved_name) = resolve_identifier_to_func_name(name) {
          return Err(InterpreterError::TailCall(Box::new(
            Expr::FunctionCall {
              name: resolved_name,
              args: args.to_vec(),
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
        _ => {}
      }

      let left_val = evaluate_expr_to_expr(left)?;
      let right_val = evaluate_expr_to_expr(right)?;

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
          // Alternatives stays symbolic (used in pattern matching)
          Ok(Expr::BinaryOp {
            op: BinaryOperator::Alternatives,
            left: Box::new(left_val),
            right: Box::new(right_val),
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
          args: terms,
        };
        return evaluate_expr_to_expr(&and_expr);
      }

      // UnsameQ with 3+ operands is NOT transitive: it requires ALL pairs
      // to be distinct, not just adjacent ones. Delegate to unsame_q_ast,
      // which implements the correct all-pairs check.
      if operators.len() >= 2
        && operators.iter().all(|op| *op == ComparisonOp::UnsameQ)
      {
        return crate::functions::boolean_ast::unsame_q_ast(&values);
      }

      // Evaluate comparison chain
      // Use try_eval_to_f64_with_infinity for numeric comparisons (handles symbolic Pi, E, Degree, Sin[...], Infinity, etc.)
      use crate::functions::math_ast::try_eval_to_f64_with_infinity as try_eval_to_f64;
      for i in 0..operators.len() {
        let left = &values[i];
        let right = &values[i + 1];
        let op = &operators[i];

        let result = match op {
          ComparisonOp::SameQ => {
            // SameQ tests structural identity, not numeric equality.
            // Integer[1] !== Real[1.] even though they are numerically equal.
            expr_to_string(left) == expr_to_string(right)
          }
          ComparisonOp::Equal => {
            if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              // Machine-precision Reals are compared up to the last ~7 bits
              // (matches wolframscript, which treats the f64 guard bits as
              // "insignificant"). Exact-vs-exact comparisons stay strict.
              let involves_real =
                matches!(left, Expr::Real(_) | Expr::BigFloat(_, _))
                  || matches!(right, Expr::Real(_) | Expr::BigFloat(_, _))
                  || matches!(left, Expr::UnaryOp { operand, .. }
                if matches!(operand.as_ref(), Expr::Real(_)))
                  || matches!(right, Expr::UnaryOp { operand, .. }
                  if matches!(operand.as_ref(), Expr::Real(_)));
              if involves_real {
                let tol = f64::max(l.abs(), r.abs()) * (2.0_f64).powi(-46);
                (l - r).abs() <= tol
              } else {
                l == r
              }
            } else if expr_to_string(left) == expr_to_string(right) {
              true
            } else if let Some(ord) =
              crate::functions::quantity_ast::try_quantity_compare(left, right)
            {
              ord == std::cmp::Ordering::Equal
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
          ComparisonOp::UnsameQ => {
            // UnsameQ tests structural non-identity, not numeric inequality.
            expr_to_string(left) != expr_to_string(right)
          }
          ComparisonOp::NotEqual => {
            if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l != r
            } else if expr_to_string(left) == expr_to_string(right) {
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
            if let (Some(l), Some(r)) =
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
            if let (Some(l), Some(r)) =
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
            if let (Some(l), Some(r)) =
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
            if let (Some(l), Some(r)) =
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
      // Delayed rules don't evaluate the replacement
      Ok(Expr::RuleDelayed {
        pattern: pattern.clone(),
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
      apply_replace_repeated_ast(&evaluated_expr, unwrapped)
    }
    Expr::Map { func, list } => {
      let evaluated_list = evaluate_expr_to_expr(list)?;
      apply_map_ast(func, &evaluated_list)
    }
    Expr::Apply { func, list } => {
      let evaluated_list = evaluate_expr_to_expr(list)?;
      apply_apply_ast(func, &evaluated_list)
    }
    Expr::MapApply { func, list } => {
      let evaluated_list = evaluate_expr_to_expr(list)?;
      apply_map_apply_ast(func, &evaluated_list)
    }
    Expr::PrefixApply { func, arg } => {
      // f @ x is equivalent to f[x]
      let evaluated_arg = evaluate_expr_to_expr(arg)?;
      apply_function_to_arg(func, &evaluated_arg)
    }
    Expr::Postfix { expr: e, func } => {
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      apply_postfix_ast(&evaluated_expr, func)
    }
    Expr::Part { expr: e, index } => {
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
            args: evaluated_args,
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
      // Part::partd: warn only at the outermost Part level (depth == 0)
      let at_outermost = PART_DEPTH.with(|d| *d.borrow() == 0);
      if at_outermost && let Expr::Part { .. } = &result {
        let base = get_part_base(&result);
        if matches!(
          base,
          Expr::Identifier(_)
            | Expr::Integer(_)
            | Expr::BigInteger(_)
            | Expr::Real(_)
        ) {
          let original = Expr::Part {
            expr: e.clone(),
            index: index.clone(),
          };
          let part_str = crate::syntax::expr_to_string(&original);
          crate::emit_message(&format!(
            "Part::partd: Part specification {} is longer than depth of object.",
            part_str
          ));
        }
      }
      Ok(result)
    }
    Expr::Function { body } => {
      // Return anonymous function as-is
      Ok(Expr::Function { body: body.clone() })
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
      // Evaluate the curried call: f[a][b] -> apply f[a] to args
      let evaluated_func = evaluate_expr_to_expr(func)?;
      let evaluated_args: Vec<Expr> = args
        .iter()
        .map(evaluate_expr_to_expr)
        .collect::<Result<_, _>>()?;
      apply_curried_call(&evaluated_func, &evaluated_args)
    }
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
    _ => false,
  }
}
