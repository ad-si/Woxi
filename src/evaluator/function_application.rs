#[allow(unused_imports)]
use super::*;

/// MapAll[f, expr] - apply f to every subexpression in expr (bottom-up)
pub fn map_all_ast(f: &Expr, expr: &Expr) -> Result<Expr, InterpreterError> {
  // First, recursively apply to subexpressions
  let mapped = match expr {
    Expr::FunctionCall { name, args } => {
      // Map over the arguments
      let new_args: Vec<Expr> = args
        .iter()
        .map(|a| map_all_ast(f, a))
        .collect::<Result<Vec<_>, _>>()?;
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    Expr::List(items) => {
      let new_items: Vec<Expr> = items
        .iter()
        .map(|a| map_all_ast(f, a))
        .collect::<Result<Vec<_>, _>>()?;
      Expr::List(new_items.into())
    }
    // Atoms: just return as-is (will be wrapped by f below)
    _ => expr.clone(),
  };

  // Then apply f to the result
  apply_function_to_arg(f, &mapped)
}

/// AST-based Distribute implementation
/// Distribute[f[x1, x2, ...]] distributes f over Plus in the xi
/// Distribute[expr, g] distributes over g instead of Plus
/// Distribute[expr, g, f] only distributes if outer head is f
pub fn distribute_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let expr = &args[0];

  // Determine the inner head to distribute over (default: Plus/addition)
  let distribute_over = if args.len() >= 2 {
    match &args[1] {
      Expr::Identifier(name) => name.clone(),
      _ => "Plus".to_string(),
    }
  } else {
    "Plus".to_string()
  };

  // Get the outer function call (or List)
  let (outer_name, outer_args) = match expr {
    Expr::FunctionCall { name, args } => (name.clone(), args.clone()),
    Expr::List(items) => ("List".to_string(), items.clone()),
    _ => {
      // Not a function call - return as-is
      return Ok(expr.clone());
    }
  };

  // If 3 args provided, check that outer head matches
  if args.len() == 3 {
    let required_head = match &args[2] {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Distribute".to_string(),
          args: args.to_vec().into(),
        });
      }
    };
    if outer_name != required_head {
      return Ok(expr.clone());
    }
  }

  // Split each argument into its parts based on distribute_over head
  let mut arg_lists: Vec<Vec<Expr>> = Vec::new();
  for arg in &outer_args {
    let parts = split_by_head(arg, &distribute_over);
    arg_lists.push(parts);
  }

  // Compute cartesian product
  let mut combinations: Vec<Vec<Expr>> = vec![vec![]];
  for parts in &arg_lists {
    let mut new_combinations = Vec::new();
    for combo in &combinations {
      for part in parts {
        let mut new_combo = combo.clone();
        new_combo.push(part.clone());
        new_combinations.push(new_combo);
      }
    }
    combinations = new_combinations;
  }

  // Build result: wrap each combination in outer_name, then combine with distribute_over
  let terms: Vec<Expr> = combinations
    .into_iter()
    .map(|combo| {
      if outer_name == "List" {
        Expr::List(combo.into())
      } else {
        Expr::FunctionCall {
          name: outer_name.clone(),
          args: combo.into(),
        }
      }
    })
    .collect();

  let result = if distribute_over == "List" {
    Expr::List(terms.into())
  } else {
    Expr::FunctionCall {
      name: distribute_over,
      args: terms.into(),
    }
  };

  evaluate_expr_to_expr(&result)
}

/// Largest slot index `n` referenced by `#n`/`##n` in `expr`. Returns 0 when
/// the body uses no slots. SlotSequence `##` (== `##1`) counts as slot 1.
pub fn max_slot_index(expr: &Expr) -> usize {
  fn walk(e: &Expr, max: &mut usize) {
    match e {
      Expr::Slot(n) | Expr::SlotSequence(n) if *n > *max => {
        *max = *n;
      }
      Expr::List(items) => items.iter().for_each(|i| walk(i, max)),
      Expr::FunctionCall { args, .. } => args.iter().for_each(|i| walk(i, max)),
      Expr::BinaryOp { left, right, .. } => {
        walk(left, max);
        walk(right, max);
      }
      Expr::UnaryOp { operand, .. } => walk(operand, max),
      Expr::Function { body } => walk(body, max),
      Expr::NamedFunction { body, .. } => walk(body, max),
      Expr::CurriedCall { func, args } => {
        walk(func, max);
        args.iter().for_each(|i| walk(i, max));
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        walk(pattern, max);
        walk(replacement, max);
      }
      Expr::Association(items) => items.iter().for_each(|(k, v)| {
        walk(k, max);
        walk(v, max);
      }),
      Expr::CompoundExpr(items) => items.iter().for_each(|i| walk(i, max)),
      _ => {}
    }
  }
  let mut m = 0;
  walk(expr, &mut m);
  m
}

/// Emit Function::slot1 / Function::slota messages for named slots (`#key`)
/// in an anonymous-function body that cannot be filled from `args`, matching
/// wolframscript: one message per unfillable named-slot occurrence.
fn emit_named_slot_messages(body: &Expr, args: &[Expr]) {
  let mut keys = Vec::new();
  crate::syntax::collect_named_slot_keys(body, &mut keys);
  if keys.is_empty() {
    return;
  }
  let body_str = crate::syntax::expr_to_message_form(body);
  if let Some(Expr::Association(items)) = args.first() {
    let assoc_str = crate::syntax::format_expr(
      args.first().unwrap(),
      crate::syntax::ExprForm::Output,
    );
    for key in keys {
      let filled = items
        .iter()
        .any(|(k, _)| matches!(k, Expr::String(s) if s == &key));
      if !filled {
        crate::emit_message(&format!(
          "Function::slota: Named slot {} in {} &  cannot be filled from {}.",
          key, body_str, assoc_str
        ));
      }
    }
  } else {
    let args_str: Vec<String> = args
      .iter()
      .map(crate::syntax::expr_to_message_form)
      .collect();
    for _ in keys {
      crate::emit_message(&format!(
        "Function::slot1: ({} & )[{}] is expected to have an Association as the first argument.",
        body_str,
        args_str.join(", ")
      ));
    }
  }
}

/// Apply `Derivative[n1, …, nk]` to a pure function body symbolically.
///
/// For each slot index `i` with `n_i > 0` we substitute `Slot(i)` for a fresh
/// dummy and try to factor the current expression as `c * dummy_i^p` where
/// `c` is constant in `dummy_i`. If the factorisation succeeds we replace
/// the `dummy_i^p` part with the wolframscript-style right-nested chain
/// `p*((p-1)*…*dummy_i^(p-n_i))` (or `0` when `n_i > p`). When the body
/// can't be peeled apart that way we fall back to `differentiate_expr`
/// repeated `n_i` times, which yields the simplified derivative.
///
/// Returns `None` if a fallback differentiation step fails (e.g. unknown
/// function head), letting the caller keep the unevaluated form.
fn differentiate_function_body(body: &Expr, orders: &[i128]) -> Option<Expr> {
  use crate::evaluator::dispatch::calculus_functions::{
    build_var_power_derivative_chain, extract_var_power_factor,
  };

  let dummies: Vec<String> = (0..orders.len())
    .map(|i| format!("__d_slot_{}__", i + 1))
    .collect();
  let dummy_exprs: Vec<Expr> = dummies
    .iter()
    .map(|d| Expr::Identifier(d.clone()))
    .collect();
  let mut current = crate::syntax::substitute_slots(body, &dummy_exprs);

  for (i, &n_i) in orders.iter().enumerate() {
    if n_i <= 0 {
      continue;
    }
    let dummy = &dummies[i];
    let dummy_expr = &dummy_exprs[i];
    if let Some((factor, p)) = extract_var_power_factor(&current, dummy)
      && let Some(chain) = build_var_power_derivative_chain(dummy_expr, p, n_i)
    {
      // Wolframscript keeps the literal `1` factor that appears when an
      // earlier slot's chain reduced to `1` (e.g.
      // `Derivative[1,2][#1*#2^3 &]` → `1*(3*(2*#2)) &`), so preserve `factor`
      // even when it's `Integer(1)`.
      current = if matches!(chain, Expr::Integer(0)) {
        Expr::Integer(0)
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(factor),
          right: Box::new(chain),
        }
      };
      continue;
    }
    for _ in 0..n_i {
      current = match crate::functions::calculus_ast::differentiate_expr(
        &current, dummy,
      ) {
        Ok(v) => v,
        Err(_) => return None,
      };
    }
  }

  for (i, dummy) in dummies.iter().enumerate() {
    current =
      crate::syntax::substitute_variable(&current, dummy, &Expr::Slot(i + 1));
  }
  Some(current)
}

/// Split an expression by its head. E.g., split_by_head(a + b, "Plus") = [a, b]
pub fn split_by_head(expr: &Expr, head: &str) -> Vec<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == head => args.to_vec(),
    Expr::List(items) if head == "List" => items.to_vec(),
    _ => vec![expr.clone()],
  }
}

/// Apply Map operation on AST (func /@ list)
pub fn apply_map_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_function_to_arg(func, item))
        .collect();
      Ok(Expr::List(results?.into()))
    }
    Expr::Association(items) => {
      // Map over association applies function to values only
      let results: Result<Vec<(Expr, Expr)>, InterpreterError> = items
        .iter()
        .map(|(key, val)| {
          let new_val = apply_function_to_arg(func, val)?;
          Ok((key.clone(), new_val))
        })
        .collect();
      Ok(Expr::Association(results?))
    }
    _ => {
      // Not a list or association, return unevaluated
      Ok(Expr::Map {
        func: Box::new(func.clone()),
        list: Box::new(list.clone()),
      })
    }
  }
}

/// Apply Apply operation on AST (func @@ list)
pub fn apply_apply_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    // Apply replaces the head of any expression: f @@ Plus[a, b, c] → f[a, b, c]
    Expr::FunctionCall { args, .. } => args.clone(),
    _ => {
      // Atoms have no children; Apply on an atom returns the atom unchanged
      return Ok(list.clone());
    }
  };

  // Apply converts List[a, b, c] to func[a, b, c]
  match func {
    Expr::Identifier(func_name) => {
      // Resolve variable holding a function name: f = Plus; f @@ {1,2} → 3
      let name = resolve_identifier_to_func_name(func_name)
        .unwrap_or_else(|| func_name.clone());
      // Build `name[items]` and evaluate it through the full pipeline so the
      // new head's attributes govern argument evaluation: a non-holding head
      // evaluates items the source head kept unevaluated
      // (List @@ Hold[1 + 1] -> {2}), while a holding head leaves them
      // untouched (Hold @@ Hold[1 + 1] -> Hold[1 + 1]).
      evaluate_expr_to_expr(&Expr::FunctionCall {
        name,
        args: items,
      })
    }
    Expr::Function { body } => {
      // Anonymous function applied to a list
      // For single-arg anonymous functions, apply to first element
      if items.len() == 1 {
        apply_function_to_arg(func, &items[0])
      } else {
        // Multiple args - substitute each slot
        let substituted = crate::syntax::substitute_slots(body, &items);
        evaluate_expr_to_expr(&substituted)
      }
    }
    Expr::NamedFunction { params, body, .. } => {
      // Named-parameter function applied to list items
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(items.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      evaluate_expr_to_expr(&substituted)
    }
    // Apply replaces the head of `list` with the whole `func`, whatever it
    // is: `g[a] @@ {1, 2}` → `g[a][1, 2]`, `Composition[f, g] @@ {1, 2}` →
    // `f[g[1, 2]]`. Build the curried application and evaluate it so
    // applicable heads reduce while inert heads stay symbolic.
    _ => apply_curried_call(func, &items),
  }
}

/// Apply MapApply operation on AST (f @@@ {{a, b}, {c, d}} -> {f[a, b], f[c, d]})
pub fn apply_map_apply_ast(
  func: &Expr,
  list: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      // Not a list, return unevaluated
      return Ok(Expr::MapApply {
        func: Box::new(func.clone()),
        list: Box::new(list.clone()),
      });
    }
  };

  // MapApply applies func to each sublist
  let results: Result<Vec<Expr>, InterpreterError> = items
    .iter()
    .map(|item| apply_apply_ast(func, item))
    .collect();

  Ok(Expr::List(results?.into()))
}

/// Apply Postfix operation on AST (expr // func)
pub fn apply_postfix_ast(
  expr: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  apply_function_to_arg(func, expr)
}

/// Apply a function to an argument (helper for Map, Postfix, etc.)
pub fn apply_function_to_arg(
  func: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      // Check if this identifier is a variable holding another function/value
      let resolved = ENV.with(|e| e.borrow().get(name).cloned());
      match &resolved {
        Some(StoredValue::ExprVal(expr)) if !matches!(expr, Expr::Identifier(n) if n == name) =>
        {
          return apply_function_to_arg(expr, arg);
        }
        _ => {}
      }
      // Resolve variable holding a function name: t = Flatten; t @ x → Flatten[x]
      if let Some(resolved_name) = resolve_identifier_to_func_name(name) {
        return evaluate_function_call_ast(&resolved_name, &[arg.clone()]);
      }
      // Simple function name: f applied to arg
      evaluate_function_call_ast(name, &[arg.clone()])
    }
    Expr::Function { body } => {
      // Anonymous function: first substitute #0 with the whole function (to
      // support recursion like If[#1<=1, 1, #1 #0[#1-1]]&), then substitute
      // the remaining numeric slots with arg. Skip the self-substitution
      // pass when #0 is absent — avoids a full tree clone per call.
      emit_named_slot_messages(body, std::slice::from_ref(arg));
      let substituted = if crate::syntax::contains_slot_zero(body) {
        let self_substituted =
          crate::syntax::substitute_slot_zero_with_self(body, func);
        crate::syntax::substitute_slots(&self_substituted, &[arg.clone()])
      } else {
        crate::syntax::substitute_slots(body, &[arg.clone()])
      };
      evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      // Named-parameter function: substitute params with arg
      if params.len() > 1 {
        // Too many parameters for a single argument — return unevaluated
        crate::emit_message(&format!(
          "Function::fpct: Too many parameters in {{{}}} to be filled from Function[{{{}}}, {}][{}].",
          params.join(", "),
          params.join(", "),
          crate::syntax::expr_to_string(body),
          crate::syntax::expr_to_string(arg),
        ));
        return Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: vec![arg.clone()],
        });
      }
      let mut substituted = (**body).clone();
      if let Some(param) = params.first() {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args } => {
      // Operator forms that evaluate to an actual function: evaluate the head
      // first, then apply the result. `FindSequenceFunction[seq]` becomes a
      // pure function, so `FindSequenceFunction[seq][k]` must apply it rather
      // than flatten to the (erroring) 2-arg `FindSequenceFunction[seq, k]`.
      if name == "FindSequenceFunction" && args.len() == 1 {
        let evaluated = evaluate_function_call_ast(name, args)?;
        if !matches!(&evaluated, Expr::FunctionCall { name: n, .. } if n == name)
        {
          return apply_function_to_arg(&evaluated, arg);
        }
      }
      // A `Function[params, body]` written as a FunctionCall (e.g. the named
      // form `Function[i, body]`) must be *applied* to the argument — binding
      // its parameters — not have the argument appended as another part.
      // This is what makes `Function[i, body] /@ list` work (Map applies the
      // function to each element).
      if name == "Function" && args.len() >= 2 {
        return apply_curried_call(func, std::slice::from_ref(arg));
      }
      // Composition[f, g, …][x] = f[g[…[x]]] and the left-to-right
      // RightComposition variant. apply_curried_call already reduces these;
      // without this, the generic fallback below would append the argument
      // (producing the unreduced Composition[f, g, x]), so e.g.
      // `Map[Length@*Union, …]` or `MaximalBy[…, Length@*Union]` failed.
      if (name == "Composition" || name == "RightComposition")
        && !args.is_empty()
      {
        return apply_curried_call(func, std::slice::from_ref(arg));
      }
      // MapAt[f, pos] operator form: the applied expression goes in the
      // middle
      if name == "MapAt" && args.len() == 2 {
        let new_args = vec![args[0].clone(), arg.clone(), args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // Insert[elem, pos] operator form: prepend the applied expression
      if name == "Insert" && args.len() == 2 {
        let new_args = vec![arg.clone(), args[0].clone(), args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // Nest[f, n][x] -> Nest[f, x, n]: the applied expression goes in the
      // middle.
      if name == "Nest" && args.len() == 2 {
        let new_args = vec![args[0].clone(), arg.clone(), args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // NearestTo[x][data] -> Nearest[data, x] (operator form of Nearest),
      // so `Map[NearestTo[x], lists]` finds the nearest in each list.
      if name == "NearestTo" && (args.len() == 1 || args.len() == 2) {
        let mut new_args = vec![arg.clone()];
        new_args.extend(args.iter().cloned());
        return evaluate_function_call_ast("Nearest", &new_args);
      }
      // Curried function: f[a] applied to b becomes f[a, b]
      // Special case: operator forms where f[x][y] becomes f[y, x]
      // (the applied argument becomes the first parameter)
      if matches!(
        name.as_str(),
        "ReplaceAll"
          | "ReplaceRepeated"
          | "StringStartsQ"
          | "StringEndsQ"
          | "StringContainsQ"
          | "StringFreeQ"
          | "StringMatchQ"
          | "StringReplace"
          | "StringCases"
          | "MemberQ"
          | "Select"
          | "KeyTake"
          | "KeyDrop"
          | "KeySelect"
          | "Lookup"
          | "TakeLargest"
          | "TakeSmallest"
          | "AllMatch"
          | "AnyMatch"
          | "FlattenAt"
          | "Delete"
          | "ReplacePart"
          | "Extract"
          | "Append"
          | "Prepend"
          | "FirstCase"
      ) && args.len() == 1
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![arg.clone(), args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if name == "ConstantFunction" {
        // ConstantFunction is not a built-in in wolframscript's Global
        // context, so f[c][x] should remain as the curried unevaluated form
        // ConstantFunction[c][x], not flatten into ConstantFunction[c, x].
        Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: vec![arg.clone()],
        })
      } else {
        // A composite head h[…] applied to an argument is a curried call
        // h[…][arg] (e.g. `g[a] /@ {1, 2}` → {g[a][1], g[a][2]}), not the
        // flattened h[…, arg]. apply_curried_call reduces operator heads
        // (Composition, OperatorApplied, …) and leaves inert heads symbolic.
        apply_curried_call(func, std::slice::from_ref(arg))
      }
    }
    Expr::Association(_) => {
      // An association used as a function is a key lookup: `assoc[key]`.
      // Needed when an association is the function in Map etc.
      // (`assoc /@ {k1, k2}` → {assoc[k1], assoc[k2]}). apply_curried_call
      // already implements the lookup (and Missing[…] for absent keys).
      apply_curried_call(func, std::slice::from_ref(arg))
    }
    Expr::CurriedCall { .. } => {
      // A curried operator used as a mapping function, e.g.
      // `Map[Curry[Power][2], list]` or `OperatorApplied[f][a] /@ list`.
      // Route through apply_curried_call so a partially-applied curry
      // operator fires once its final argument arrives, and any other
      // curried head accumulates the extra argument.
      apply_curried_call(func, std::slice::from_ref(arg))
    }
    // A compound arithmetic/relational head stays an inert curried call,
    // e.g. `(f + g) @ x` → `(f + g)[x]`. Stringifying it would mistake
    // "f + g" for a function name.
    Expr::BinaryOp { .. } | Expr::UnaryOp { .. } | Expr::Comparison { .. } => {
      Ok(Expr::CurriedCall {
        func: Box::new(func.clone()),
        args: vec![arg.clone()],
      })
    }
    _ => {
      // Fallback: create a function call expression
      let func_str = expr_to_string(func);
      if let Some(name) = func_str.strip_suffix('&') {
        // It's an anonymous function like "#^2&"
        let body = string_to_expr(name)?;
        let substituted =
          crate::syntax::substitute_slots(&body, &[arg.clone()]);
        evaluate_expr_to_expr(&substituted)
      } else {
        // Treat as a function name
        evaluate_function_call_ast(&func_str, &[arg.clone()])
      }
    }
  }
}

/// Parse a `Curry[...]` operator form — possibly mid-accumulation — into its
/// target function `f`, the slot permutation, and any already-collected args.
/// Returns `None` when `func` is not a Curry form.
///
/// The permutation is 1-indexed in application order: output position `i`
/// receives the `perm[i]`-th argument supplied (so `arranged[i] =
/// collected[perm[i] - 1]`). A bare `Curry[f]` reverses two arguments
/// (`{2, 1}`); `Curry[f, n]` collects `n` arguments in order (`{1, …, n}`);
/// `Curry[f, {p1, …}]` uses the explicit permutation.
fn parse_curry_form(func: &Expr) -> Option<(Expr, Vec<usize>, Vec<Expr>)> {
  let parse_base = |name: &str, cargs: &[Expr]| -> Option<(Expr, Vec<usize>)> {
    match cargs.len() {
      // A bare `Curry[f]` / `OperatorApplied[f]` reverses two arguments, but
      // `CurryApplied[f]` (no count) does not curry — it stays inert.
      1 if name == "CurryApplied" => None,
      1 => Some((cargs[0].clone(), vec![2, 1])),
      2 => {
        let perm = match &cargs[1] {
          Expr::Integer(n) if *n >= 1 => (1..=*n as usize).collect(),
          Expr::List(items) if !items.is_empty() => {
            let mut p = Vec::with_capacity(items.len());
            for it in items {
              match it {
                Expr::Integer(k) if *k >= 1 => p.push(*k as usize),
                _ => return None,
              }
            }
            p
          }
          _ => return None,
        };
        Some((cargs[0].clone(), perm))
      }
      _ => None,
    }
  };
  // `OperatorApplied` is the public-facing spelling of the same operator:
  // `OperatorApplied[f]` reverses two args, `OperatorApplied[f, n]` collects n
  // in order, `OperatorApplied[f, {perm}]` uses the explicit permutation.
  let is_curry =
    |n: &str| n == "Curry" || n == "OperatorApplied" || n == "CurryApplied";
  match func {
    Expr::FunctionCall { name, args } if is_curry(name) => {
      parse_base(name, args).map(|(f, perm)| (f, perm, Vec::new()))
    }
    Expr::CurriedCall { func: inner, args } => match inner.as_ref() {
      Expr::FunctionCall { name, args: cargs } if is_curry(name) => {
        parse_base(name, cargs).map(|(f, perm)| (f, perm, args.clone()))
      }
      _ => None,
    },
    _ => None,
  }
}

/// Handle application of a `Curry[...]` operator form to `new_args`.
/// Returns `None` when `func` is not a Curry form (so normal dispatch runs).
fn try_apply_curry(
  func: &Expr,
  new_args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let (f, perm, acc) = parse_curry_form(func)?;
  let n = perm.len();
  let mut collected = acc;
  collected.extend_from_slice(new_args);

  if collected.len() < n {
    // Not enough arguments yet: keep accumulating under the bare `Curry[…]`
    // head (so `Head[Curry[f][a]]` stays `Curry[f]`).
    let base = match func {
      Expr::CurriedCall { func: inner, .. } => (**inner).clone(),
      other => other.clone(),
    };
    return Some(Ok(Expr::CurriedCall {
      func: Box::new(base),
      args: collected,
    }));
  }

  // Arrange the first `n` collected arguments by the permutation.
  let mut head_args = Vec::with_capacity(n);
  for &src in &perm {
    if src < 1 || src > collected.len() {
      // Out-of-range slot: leave unevaluated.
      let base = match func {
        Expr::CurriedCall { func: inner, .. } => (**inner).clone(),
        other => other.clone(),
      };
      return Some(Ok(Expr::CurriedCall {
        func: Box::new(base),
        args: collected,
      }));
    }
    head_args.push(collected[src - 1].clone());
  }
  let leftover = collected[n..].to_vec();

  let applied = match apply_curried_call(&f, &head_args) {
    Ok(e) => e,
    Err(e) => return Some(Err(e)),
  };
  if leftover.is_empty() {
    Some(Ok(applied))
  } else {
    Some(apply_curried_call(&applied, &leftover))
  }
}

/// Apply a curried call: f[a][b, c] applies function result f[a] to args [b, c]
/// Handle application of a `ReverseApplied[f]` / `ReverseApplied[f, n]`
/// operator to `new_args`. `ReverseApplied[f]` reverses all supplied
/// arguments (`ReverseApplied[f][x1, …, xn]` = `f[xn, …, x1]`); the `n` form
/// reverses only the first `n`. Unlike Curry it does not accumulate — it
/// fires on the first application with whatever arguments are given.
fn try_apply_reverse_applied(
  func: &Expr,
  new_args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let Expr::FunctionCall { name, args } = func else {
    return None;
  };
  if name != "ReverseApplied" {
    return None;
  }
  let (f, n) = match args.len() {
    1 => (&args[0], None),
    2 => match &args[1] {
      Expr::Integer(k) if *k >= 0 => (&args[0], Some(*k as usize)),
      _ => return None,
    },
    _ => return None,
  };
  let mut head_args = new_args.to_vec();
  match n {
    None => head_args.reverse(),
    Some(n) => {
      let k = n.min(head_args.len());
      head_args[..k].reverse();
    }
  }
  Some(apply_curried_call(f, &head_args))
}

pub fn apply_curried_call(
  func: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if let Some(result) = try_apply_curry(func, args) {
    return result;
  }
  if let Some(result) = try_apply_reverse_applied(func, args) {
    return result;
  }
  match func {
    Expr::Identifier(name) => {
      // Simple function name applied to args
      evaluate_function_call_ast(name, args)
    }
    Expr::Function { body } => {
      // Anonymous function: substitute #0 with the whole function first,
      // then # with args and evaluate. Skip the self-substitution pass
      // when #0 is absent — avoids a full tree clone per call.
      emit_named_slot_messages(body, args);
      let substituted = if crate::syntax::contains_slot_zero(body) {
        let self_substituted =
          crate::syntax::substitute_slot_zero_with_self(body, func);
        crate::syntax::substitute_slots(&self_substituted, args)
      } else {
        crate::syntax::substitute_slots(body, args)
      };
      evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      // Named-parameter function: substitute each param with corresponding arg
      if params.len() > args.len() {
        // Too many parameters for the given arguments — return unevaluated
        let args_str: Vec<String> =
          args.iter().map(crate::syntax::expr_to_string).collect();
        crate::emit_message(&format!(
          "Function::fpct: Too many parameters in {{{}}} to be filled from Function[{{{}}}, {}][{}].",
          params.join(", "),
          params.join(", "),
          crate::syntax::expr_to_string(body),
          args_str.join(", "),
        ));
        return Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        });
      }
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(args.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "StringTemplate" && func_args.len() == 1 => {
      // StringTemplate[template][args…] — fill the template's slots.
      crate::functions::string_ast::apply_string_template(&func_args[0], args)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Query" && args.len() == 1 => {
      // Query[ops...][data] — successive-level query application
      crate::functions::query_ast::apply_query(func_args, &args[0])
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Entity" && func_args.len() == 2 => {
      // Entity["type", "name"]["property"] — property access on entities
      crate::functions::entity_ast::entity_property_access(func_args, args)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Interpreter" && func_args.len() == 1 => {
      // Interpreter["Country"][input] — resolve input to an Entity.
      crate::functions::country_data::apply_interpreter(&func_args[0], args)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "EntityStore" && func_args.len() == 1 => {
      // EntityStore[...][Entity["type", "name"], "property"] — callable store form
      crate::functions::entity_ast::entity_store_property_access(
        func_args, args,
      )
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "InterpolatingFunction"
      && (func_args.len() == 2 || func_args.len() == 3) =>
    {
      // InterpolatingFunction[domain, data][x] — interpolate at x
      crate::functions::ode_ast::evaluate_interpolating_function(
        func_args, args,
      )
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "FittedModel" && func_args.len() == 1 => {
      // FittedModel[assoc][x] — evaluate model or query property
      crate::functions::linear_algebra_ast::evaluate_fitted_model(
        func_args, args,
      )
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Dataset" && func_args.len() == 3 => {
      // Dataset[data, type, meta][args...] — dataset querying
      crate::functions::dataset_ast::dataset_query(func_args, args)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "BezierFunction" && func_args.len() == 1 => {
      // BezierFunction[{{p1}, {p2}, ...}][t] — evaluate Bezier curve at t
      evaluate_bezier_function(func_args, args)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "BezierFunction" && func_args.len() == 7 => {
      // Structured form: BezierFunction[degree, knots, {n}, {points, {}}, ...]
      // The control points live in args[3][0]. Wrap them in a 1-arg
      // BezierFunction and reuse the standard evaluator.
      if let Expr::List(slot) = &func_args[3]
        && !slot.is_empty()
      {
        let pts = vec![slot[0].clone()];
        evaluate_bezier_function(&pts, args)
      } else {
        Err(InterpreterError::EvaluationError(
          "BezierFunction: invalid structured form".into(),
        ))
      }
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "TransformationFunction" && func_args.len() == 1 => {
      // TransformationFunction[matrix][{x, y, ...}] — apply affine transformation
      apply_transformation_function(&func_args[0], args)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "CompiledFunction" && func_args.len() == 2 => {
      // CompiledFunction[{x, y, ...}, body][args...] — substitute and evaluate numerically
      let params: Vec<String> = match &func_args[0] {
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(n) = item {
              Some(n.clone())
            } else {
              None
            }
          })
          .collect(),
        Expr::Identifier(n) => vec![n.clone()],
        _ => vec![],
      };
      let body = &func_args[1];
      // Convert all arguments to Real (compiled functions work numerically).
      // Rationals also coerce to Real so `cf[1/2]` matches `cf[0.5]`.
      let num_args: Vec<Expr> = args
        .iter()
        .map(|a| match a {
          Expr::Integer(n) => Expr::Real(*n as f64),
          Expr::BigInteger(n) => {
            use std::str::FromStr;
            Expr::Real(f64::from_str(&n.to_string()).unwrap_or(f64::NAN))
          }
          Expr::FunctionCall { name, args: ra }
            if name == "Rational" && ra.len() == 2 =>
          {
            if let (Expr::Integer(n), Expr::Integer(d)) = (&ra[0], &ra[1]) {
              Expr::Real(*n as f64 / *d as f64)
            } else {
              a.clone()
            }
          }
          other => other.clone(),
        })
        .collect();
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(num_args.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      let result = evaluate_expr_to_expr(&substituted)?;
      // Ensure the result is numerical
      match &result {
        Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
        _ => Ok(result),
      }
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } => {
      // Curried function: f[a][b] becomes f[a, b]
      // Special case: operator forms where f[x][y] becomes f[y, x]
      if name == "MapAt" && func_args.len() == 2 && args.len() == 1 {
        let new_args =
          vec![func_args[0].clone(), args[0].clone(), func_args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      if name == "Insert" && func_args.len() == 2 && args.len() == 1 {
        let new_args =
          vec![args[0].clone(), func_args[0].clone(), func_args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // TakeLargestBy[f, n][list] -> TakeLargestBy[list, f, n] (and likewise
      // TakeSmallestBy): the applied list becomes the first argument.
      if (name == "TakeLargestBy" || name == "TakeSmallestBy")
        && func_args.len() == 2
        && args.len() == 1
      {
        let new_args =
          vec![args[0].clone(), func_args[0].clone(), func_args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // NearestTo[x][data] -> Nearest[data, x];
      // NearestTo[x, n][data] -> Nearest[data, x, n].
      if name == "NearestTo" && (func_args.len() == 1 || func_args.len() == 2) {
        let mut new_args = args.to_vec();
        new_args.extend(func_args.iter().cloned());
        return evaluate_function_call_ast("Nearest", &new_args);
      }
      if matches!(
        name.as_str(),
        "ReplaceAll"
          | "ReplaceRepeated"
          | "StringStartsQ"
          | "StringEndsQ"
          | "StringContainsQ"
          | "StringFreeQ"
          | "StringMatchQ"
          | "StringReplace"
          | "StringCases"
          | "MemberQ"
          | "Select"
          | "KeyTake"
          | "KeyDrop"
          | "KeySelect"
          | "Lookup"
          | "TakeLargest"
          | "TakeSmallest"
          | "SortBy"
          | "GroupBy"
          | "CountsBy"
          | "MaximalBy"
          | "MinimalBy"
          | "Cases"
          | "DeleteCases"
          | "Position"
          | "FreeQ"
          | "MatchQ"
          | "Count"
          | "AllMatch"
          | "AnyMatch"
          | "FlattenAt"
          | "Delete"
          | "ReplacePart"
          | "Extract"
          | "Append"
          | "Prepend"
          | "FirstCase"
      ) && func_args.len() == 1
        && args.len() == 1
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![args[0].clone(), func_args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if name == "Nest" && func_args.len() == 2 && args.len() == 1 {
        // Nest[f, n][x] -> Nest[f, x, n]
        let new_args =
          vec![func_args[0].clone(), args[0].clone(), func_args[1].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if name == "Composition" && !func_args.is_empty() {
        // Composition[f, g, h][x] applies functions right-to-left: f[g[h[x]]]
        let mut result = args.to_vec();
        for f in func_args.iter().rev() {
          let intermediate = apply_curried_call(f, &result)?;
          result = vec![intermediate];
        }
        Ok(result.into_iter().next().unwrap())
      } else if name == "RightComposition" && !func_args.is_empty() {
        // RightComposition[f, g, h][x] applies functions left-to-right: h[g[f[x]]]
        let mut result = args.to_vec();
        for f in func_args.iter() {
          let intermediate = apply_curried_call(f, &result)?;
          result = vec![intermediate];
        }
        Ok(result.into_iter().next().unwrap())
      } else if name == "MapAt" && func_args.len() == 2 && args.len() == 1 {
        // MapAt[f, pos][expr] -> MapAt[f, expr, pos]
        let new_args =
          vec![func_args[0].clone(), args[0].clone(), func_args[1].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if name == "Key" && func_args.len() == 1 && args.len() == 1 {
        // Key[k][assoc] — extract value for key k from association
        let key = &func_args[0];
        let key_str = expr_to_string(key);
        match &args[0] {
          Expr::Association(pairs) => {
            for (k, v) in pairs {
              if expr_to_string(k) == key_str {
                return Ok(v.clone());
              }
            }
            // Key not found: return Missing["KeyAbsent", k]
            Ok(Expr::FunctionCall {
              name: "Missing".to_string(),
              args: vec![Expr::String("KeyAbsent".to_string()), key.clone()]
                .into(),
            })
          }
          _ => {
            // Not an association: return unevaluated
            Ok(Expr::CurriedCall {
              func: Box::new(func.clone()),
              args: args.to_vec(),
            })
          }
        }
      } else if name == "Derivative"
        && func_args.len() > 1
        && func_args.iter().all(|a| matches!(a, Expr::Integer(_)))
      {
        // If every derivative order is 0, the derivative is the identity:
        // Derivative[0, 0, ..., 0][f] simplifies to f (or f applied to args,
        // if this is Derivative[0, ...][f][x, y, ...]).
        if func_args.iter().all(|a| matches!(a, Expr::Integer(0))) {
          if args.is_empty() {
            return Ok(args.first().cloned().unwrap_or_else(|| func.clone()));
          }
          if args.len() == 1 {
            return Ok(args[0].clone());
          }
          // Multi-arg: apply the body to the args.
          return evaluate_function_call_ast("CompoundExpression", args);
        }
        // `Derivative[n1, ..., nk][List]` — fold to a Function that
        // returns a `List` of length `k`. Each entry is `D[#i, #1^n1, …,
        // #k^nk]`: 1 when exactly that position has order 1 and every
        // other position has order 0, otherwise 0.
        if args.len() == 1
          && matches!(&args[0], Expr::Identifier(s) if s == "List")
        {
          let orders: Vec<i128> = func_args
            .iter()
            .map(|a| match a {
              Expr::Integer(n) => *n,
              _ => 0,
            })
            .collect();
          let any_high = orders.iter().any(|n| *n > 1 || *n < 0);
          let ones: Vec<usize> = orders
            .iter()
            .enumerate()
            .filter_map(|(i, n)| if *n == 1 { Some(i) } else { None })
            .collect();
          let body: Expr = if any_high || ones.len() > 1 {
            Expr::List(orders.iter().map(|_| Expr::Integer(0)).collect())
          } else if ones.len() == 1 {
            let target = ones[0];
            Expr::List(
              (0..orders.len())
                .map(|i| {
                  if i == target {
                    Expr::Integer(1)
                  } else {
                    Expr::Integer(0)
                  }
                })
                .collect(),
            )
          } else {
            // Every order is zero — this branch is unreachable because
            // the all-zeros guard above caught it, but keep the fallback
            // for completeness.
            Expr::List((0..orders.len()).map(|i| Expr::Slot(i + 1)).collect())
          };
          return Ok(Expr::Function {
            body: Box::new(body),
          });
        }
        // `Derivative[n1, …, nk][body &]` where the body only uses slots
        // up to index `k_max`: any non-zero order beyond `k_max`
        // differentiates with respect to a slot that doesn't appear, so
        // the result is `0 &`. Matches wolframscript: `Derivative[0, 1][# &]`
        // → `0 &`.
        if args.len() == 1
          && let Expr::Function { body } = &args[0]
        {
          let max_slot = max_slot_index(body);
          let orders: Vec<i128> = func_args
            .iter()
            .map(|a| match a {
              Expr::Integer(n) => *n,
              _ => 0,
            })
            .collect();
          let beyond_zero = orders
            .iter()
            .enumerate()
            .any(|(i, n)| (i + 1) > max_slot && *n > 0);
          if beyond_zero {
            return Ok(Expr::Function {
              body: Box::new(Expr::Integer(0)),
            });
          }
          // Try to evaluate `Derivative[n1, …, nk][body &]` symbolically.
          // For each slot k with order n_k > 0 we either factor the current
          // expression as `c * dummy_k^p` (constant `c` × pure power) and
          // splice in the right-nested chain `p*((p-1)*…*dummy_k^(p-n_k))`,
          // or fall back to ordinary `differentiate_expr` repeated n_k
          // times. The chain branch is what makes wolframscript's preserved
          // multiplication structure (e.g. `Cos[#1]*(3*(2*#2)) &`) come
          // through unsimplified.
          if let Some(result) = differentiate_function_body(body, &orders) {
            return Ok(Expr::Function {
              body: Box::new(result),
            });
          }
        }
        // Multi-index derivative: Derivative[n1, n2, ...][f] — keep as
        // CurriedCall since the flattened form is ambiguous with
        // Derivative[n, f, x] (nth derivative of f at x).
        Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        })
      } else if matches!(
        name.as_str(),
        "Derivative"
          | "Apply"
          | "Map"
          | "MapIndexed"
          | "MapThread"
          | "Scan"
          | "Append"
          | "Prepend"
          | "Take"
          | "Drop"
          | "Between"
          | "Comap"
          | "ComapApply"
          | "KeyMap"
          | "AssociationMap"
          | "KeyValueMap"
      ) {
        // `Derivative[n][const]` → `0&` for n ≥ 1, `const&` for n == 0.
        // Caught here (before flattening) so the multi-index `Derivative[1, 0]`
        // form, which goes through evaluate_function_call_ast directly, stays
        // symbolic. Constants are atomic numerics that can't be a derivative
        // order interpretation: Real, Rational, Constant, BigFloat, Complex,
        // and Integer (since `Derivative[1, 0][f]` already routed through the
        // multi-index branch above and never reaches here).
        if name == "Derivative" && func_args.len() == 1 && args.len() == 1 {
          let arg0 = &args[0];
          let is_constant_arg = matches!(
            arg0,
            Expr::Integer(_)
              | Expr::Real(_)
              | Expr::BigFloat(_, _)
              | Expr::BigInteger(_)
              | Expr::Constant(_)
          ) || matches!(
            arg0,
            Expr::FunctionCall { name: rn, .. }
              if rn == "Rational" || rn == "Complex"
          ) || matches!(
            arg0,
            Expr::Identifier(s) if s == "I"
          );
          if is_constant_arg {
            // Integer order: 0 -> const&, otherwise 0&. Non-integer order:
            // wolframscript also folds to 0& (treating the order as nonzero).
            let body = match &func_args[0] {
              Expr::Integer(0) => arg0.clone(),
              _ => Expr::Integer(0),
            };
            return Ok(Expr::Function {
              body: Box::new(body),
            });
          }
        }
        // Known operator-form functions: flatten curried call
        let mut new_args = func_args.clone();
        new_args.extend(args.iter().cloned());
        evaluate_function_call_ast(name, &new_args)
      } else if matches!(
        name.as_str(),
        "Replace" | "ReplaceAll" | "ReplaceRepeated"
      ) && args.len() == 1
      {
        // Curried replacement: Replace[rules][expr] = Replace[expr, rules].
        // The expr argument comes FIRST in the uncurried form — the opposite
        // of Map/Apply-style operator forms.
        let mut new_args = args.to_vec();
        new_args.extend(func_args.iter().cloned());
        evaluate_function_call_ast(name, &new_args)
      } else if name == "Function" && func_args.len() >= 2 {
        // Function[{params...}, body, attrs?][args...] — substitute params
        // with args in body and evaluate. Hold attributes on the 3rd arg
        // are honoured by the caller (see function_hold_attributes); here
        // we just bind and evaluate.
        let params: Vec<String> = match &func_args[0] {
          Expr::List(items) => items
            .iter()
            .filter_map(|e| match e {
              Expr::Identifier(n) => Some(n.clone()),
              _ => None,
            })
            .collect(),
          Expr::Identifier(n) => vec![n.clone()],
          _ => Vec::new(),
        };
        let body = &func_args[1];
        if params.is_empty() {
          let substituted = crate::syntax::substitute_slots(body, args);
          evaluate_expr_to_expr(&substituted)
        } else {
          let bindings: Vec<(&str, &Expr)> = params
            .iter()
            .zip(args.iter())
            .map(|(p, a)| (p.as_str(), a))
            .collect();
          let substituted =
            crate::syntax::substitute_variables(body, &bindings);
          evaluate_expr_to_expr(&substituted)
        }
      } else {
        // Try SubValue rules registered via `f[a][b] := …`. Reconstruct the
        // full curried call and match it against each stored rule keyed by the
        // outermost head; on a match, substitute the bindings into the body.
        let outer_head = {
          let mut inner: &Expr = func;
          loop {
            match inner {
              Expr::CurriedCall { func: f2, .. } => inner = f2.as_ref(),
              Expr::FunctionCall { name, .. } => break Some(name.clone()),
              _ => break None,
            }
          }
        };
        if let Some(head) = outer_head {
          let rules = crate::evaluator::assignment::SUB_VALUES
            .with(|m| m.borrow().get(&head).cloned());
          if let Some(rules) = rules {
            let actual = Expr::CurriedCall {
              func: Box::new(func.clone()),
              args: args.to_vec(),
            };
            for (lhs, body) in &rules {
              if let Some(bindings) =
                crate::evaluator::pattern_matching::match_pattern(&actual, lhs)
              {
                return crate::evaluator::pattern_matching::apply_bindings(
                  body, &bindings,
                );
              }
            }
          }
        }
        // Unknown/symbolic curried call: preserve the CurriedCall form
        // e.g. f[g][x] stays as f[g][x], not f[g, x]
        Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        })
      }
    }
    Expr::Association(pairs) => {
      // assoc["key"] — association lookup
      if args.len() == 1 {
        let key_str = expr_to_string(&args[0]);
        for (k, v) in pairs {
          if expr_to_string(k) == key_str {
            return Ok(v.clone());
          }
        }
        // Key not found: return Missing["KeyAbsent", key]
        Ok(Expr::FunctionCall {
          name: "Missing".to_string(),
          args: vec![Expr::String("KeyAbsent".to_string()), args[0].clone()]
            .into(),
        })
      } else {
        Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        })
      }
    }
    Expr::List(_) => {
      // {f, g}[x] stays unevaluated as a CurriedCall in Wolfram Language
      Ok(Expr::CurriedCall {
        func: Box::new(func.clone()),
        args: args.to_vec(),
      })
    }
    Expr::CurriedCall { .. } => {
      // Nested curried call: `s[a][b][c]` arrives here as
      // `apply_curried_call(CurriedCall{s[a], [b]}, [c])`. Preserve the
      // structure rather than collapsing the head to a string-named
      // FunctionCall (which loses the AST shape that pattern-matchers
      // rely on, e.g. `s[a][b][c] /. s[x_][y_][z_] -> …`).
      Ok(Expr::CurriedCall {
        func: Box::new(func.clone()),
        args: args.to_vec(),
      })
    }
    // A compound arithmetic/relational head stays an inert curried call,
    // e.g. `(f + g)[x]`, rather than being stringified to a function name.
    Expr::BinaryOp { .. } | Expr::UnaryOp { .. } | Expr::Comparison { .. } => {
      Ok(Expr::CurriedCall {
        func: Box::new(func.clone()),
        args: args.to_vec(),
      })
    }
    _ => {
      // Fallback: try to convert to string and evaluate
      let func_str = expr_to_string(func);
      if let Some(name) = func_str.strip_suffix('&') {
        // It's an anonymous function like "#^2&"
        let body = string_to_expr(name)?;
        let substituted = crate::syntax::substitute_slots(&body, args);
        evaluate_expr_to_expr(&substituted)
      } else if args.len() == 1 {
        // Treat as a function name with single arg
        evaluate_function_call_ast(&func_str, args)
      } else {
        // Multiple args - treat as curried
        evaluate_function_call_ast(&func_str, args)
      }
    }
  }
}

/// Evaluate BezierFunction[{control_points}][t] using de Casteljau's algorithm.
fn evaluate_bezier_function(
  func_args: &[Expr],
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "BezierFunction".to_string(),
      args: func_args.to_vec().into(),
    });
  }

  // Extract the parameter t as f64
  let t = match &args[0] {
    Expr::Real(f) => *f,
    Expr::Integer(n) => *n as f64,
    Expr::FunctionCall { name: rn, args: ra }
      if rn == "Rational" && ra.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&ra[0], &ra[1]) {
        *n as f64 / *d as f64
      } else {
        return Ok(Expr::FunctionCall {
          name: "BezierFunction".to_string(),
          args: func_args.to_vec().into(),
        });
      }
    }
    _ => {
      // Symbolic t: return unevaluated
      return Ok(Expr::FunctionCall {
        name: "BezierFunction".to_string(),
        args: func_args.to_vec().into(),
      });
    }
  };

  // Extract control points from func_args[0] which should be a list of points
  let points = match &func_args[0] {
    Expr::List(pts) => pts,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BezierFunction".to_string(),
        args: func_args.to_vec().into(),
      });
    }
  };

  if points.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "BezierFunction".to_string(),
      args: func_args.to_vec().into(),
    });
  }

  // Convert control points to Vec<Vec<f64>>
  let mut ctrl_pts: Vec<Vec<f64>> = Vec::new();
  for pt in points {
    match pt {
      Expr::List(coords) => {
        let mut fcoords = Vec::new();
        for c in coords {
          match c {
            Expr::Real(f) => fcoords.push(*f),
            Expr::Integer(n) => fcoords.push(*n as f64),
            Expr::FunctionCall { name: rn, args: ra }
              if rn == "Rational" && ra.len() == 2 =>
            {
              if let (Expr::Integer(n), Expr::Integer(d)) = (&ra[0], &ra[1]) {
                fcoords.push(*n as f64 / *d as f64);
              } else {
                return Ok(Expr::FunctionCall {
                  name: "BezierFunction".to_string(),
                  args: func_args.to_vec().into(),
                });
              }
            }
            _ => {
              return Ok(Expr::FunctionCall {
                name: "BezierFunction".to_string(),
                args: func_args.to_vec().into(),
              });
            }
          }
        }
        ctrl_pts.push(fcoords);
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "BezierFunction".to_string(),
          args: func_args.to_vec().into(),
        });
      }
    }
  }

  // De Casteljau's algorithm
  let n = ctrl_pts.len();
  let dim = ctrl_pts[0].len();
  let mut work = ctrl_pts;
  for r in 1..n {
    for i in 0..n - r {
      for d in 0..dim {
        work[i][d] = (1.0 - t) * work[i][d] + t * work[i + 1][d];
      }
    }
  }

  Ok(Expr::List(work[0].iter().map(|&v| Expr::Real(v)).collect()))
}

/// Apply TransformationFunction[matrix] to a point vector.
/// The matrix is an (n+1)x(n+1) augmented matrix for affine transformation.
/// Given point {x1, ..., xn}, computes matrix . {x1, ..., xn, 1} and returns
/// the first n components.
fn apply_transformation_function(
  matrix: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::CurriedCall {
      func: Box::new(Expr::FunctionCall {
        name: "TransformationFunction".to_string(),
        args: vec![matrix.clone()].into(),
      }),
      args: args.to_vec(),
    });
  }
  let point = &args[0];
  let coords = match point {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "TransformationFunction".to_string(),
          args: vec![matrix.clone()].into(),
        }),
        args: args.to_vec(),
      });
    }
  };

  let rows = match matrix {
    Expr::List(rows) => rows,
    _ => {
      return Ok(Expr::CurriedCall {
        func: Box::new(Expr::FunctionCall {
          name: "TransformationFunction".to_string(),
          args: vec![matrix.clone()].into(),
        }),
        args: args.to_vec(),
      });
    }
  };

  let n = coords.len();
  // Build homogeneous coordinate vector: {x1, ..., xn, 1}
  let mut hom = coords.clone();
  hom.push(Expr::Integer(1));

  // Multiply: take first n rows, dot with homogeneous vector
  let mut result = Vec::with_capacity(n);
  for i in 0..n {
    let row = match &rows[i] {
      Expr::List(r) => r,
      _ => {
        return Ok(Expr::CurriedCall {
          func: Box::new(Expr::FunctionCall {
            name: "TransformationFunction".to_string(),
            args: vec![matrix.clone()].into(),
          }),
          args: args.to_vec(),
        });
      }
    };
    // Dot product of row with homogeneous vector
    let dot = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: row
        .iter()
        .zip(hom.iter())
        .map(|(a, b)| Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![a.clone(), b.clone()].into(),
        })
        .collect(),
    };
    result.push(evaluate_expr_to_expr(&dot)?);
  }

  Ok(Expr::List(result.into()))
}
