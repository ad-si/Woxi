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
        return Ok(unevaluated("Distribute", args));
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
/// The name of one `Compile` argument spec and whether its declared element
/// type is `_Real`. A bare name (`x`) defaults to `_Real`, matching the
/// Wolfram Language; `{s, _Integer, 0}` binds an integer instead, which a
/// `Nest` count or a `Range` bound needs.
fn compile_arg_spec(spec: &Expr) -> Option<(String, bool)> {
  match spec {
    Expr::Identifier(name) => Some((name.clone(), true)),
    Expr::List(items) if !items.is_empty() => {
      let Expr::Identifier(name) = &items[0] else {
        return None;
      };
      let real = match items.get(1) {
        Some(Expr::Pattern { head, .. }) => head.as_deref() != Some("Integer"),
        // No declared type — the default is `_Real`.
        _ => true,
      };
      Some((name.clone(), real))
    }
    _ => None,
  }
}

/// Convert exact numbers to machine reals, descending into lists so a
/// rank-n `_Real` argument arrives numeric throughout — the sample grid a
/// compiled Mandelbrot iteration is handed is full of exact rationals.
fn coerce_to_real(expr: &Expr) -> Expr {
  match expr {
    Expr::Integer(n) => Expr::Real(*n as f64),
    Expr::BigInteger(n) => {
      use std::str::FromStr;
      Expr::Real(f64::from_str(&n.to_string()).unwrap_or(f64::NAN))
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Expr::Real(*n as f64 / *d as f64)
      } else {
        expr.clone()
      }
    }
    Expr::List(items) => {
      Expr::List(items.iter().map(coerce_to_real).collect::<Vec<_>>().into())
    }
    other => other.clone(),
  }
}

fn max_slot_index(expr: &Expr) -> usize {
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
        for i in args {
          walk(i, max);
        }
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
          "Function::slota: Named slot {key} in {body_str} &  cannot be filled from {assoc_str}."
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
        times2(factor, chain)
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
fn split_by_head(expr: &Expr, head: &str) -> Vec<Expr> {
  // Operators like `|` (Alternatives) or `+` (Plus) are stored as (possibly
  // nested) BinaryOp nodes rather than FunctionCall nodes. Map the operator to
  // its head name so Distribute can split e.g. `a | b | c` into {a, b, c}.
  let binop_head = |op: &BinaryOperator| -> Option<&'static str> {
    match op {
      BinaryOperator::Plus => Some("Plus"),
      BinaryOperator::Times => Some("Times"),
      BinaryOperator::And => Some("And"),
      BinaryOperator::Or => Some("Or"),
      BinaryOperator::StringJoin => Some("StringJoin"),
      BinaryOperator::Alternatives => Some("Alternatives"),
      _ => None,
    }
  };
  match expr {
    Expr::FunctionCall { name, args } if name == head => args.to_vec(),
    Expr::List(items) if head == "List" => items.to_vec(),
    Expr::BinaryOp { op, left, right } if binop_head(op) == Some(head) => {
      let mut parts = split_by_head(left, head);
      parts.extend(split_by_head(right, head));
      parts
    }
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
    // Any other expression maps over its arguments, keeping its head:
    // `f /@ g[a, b]` is `g[f[a], f[b]]`, and an atom maps to itself.
    // Delegating keeps the `/@` operator form and `Map[…]` from drifting.
    _ => crate::functions::list_helpers_ast::map_ast(func, list),
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
    // Some heads stay as (possibly nested) BinaryOp nodes rather than
    // FunctionCall nodes — notably Alternatives (`a | b | c`). Flatten
    // same-operator chains so `List @@ (a | b | c)` → {a, b, c}, matching WS.
    Expr::BinaryOp { op, left, right } => {
      fn flatten(op: BinaryOperator, e: &Expr, out: &mut Vec<Expr>) {
        if let Expr::BinaryOp {
          op: inner,
          left,
          right,
        } = e
          && *inner == op
          && !matches!(op, BinaryOperator::Power)
        {
          flatten(op, left, out);
          flatten(op, right, out);
          return;
        }
        out.push(e.clone());
      }
      let mut parts = Vec::new();
      flatten(*op, left, &mut parts);
      flatten(*op, right, &mut parts);
      parts.into()
    }
    // Comparison chains: `a == b == c` is Equal[a, b, c]; `a < b <= c` is
    // Inequality[a, Less, b, LessEqual, c]. Matches wolframscript.
    Expr::Comparison {
      operands,
      operators,
    } => {
      let (_, args) =
        crate::syntax::comparison_head_and_args(operands, operators);
      args.into()
    }
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
      evaluate_expr_to_expr(&Expr::FunctionCall { name, args: items })
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
    // `f @@@ expr` is `Apply[f, expr, {1}]`: it replaces the head of each
    // element of any expression, not just of a list.
    _ => {
      return crate::functions::list_helpers_ast::apply_at_level_ast(
        func,
        list,
        &Expr::List(vec![Expr::Integer(1)].into()),
      );
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
  // `expr // f` is `f[expr]`, so a `Sequence` argument spreads into f's
  // arguments the way it would if the call had been written out.
  if let Expr::FunctionCall { name, args } = expr
    && name == "Sequence"
    && let Expr::Identifier(head) = func
  {
    return evaluate_function_call_ast(
      head,
      &args.iter().cloned().collect::<Vec<_>>(),
    );
  }
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
        return evaluate_function_call_ast(
          &resolved_name,
          std::slice::from_ref(arg),
        );
      }
      // Simple function name: f applied to arg
      evaluate_function_call_ast(name, std::slice::from_ref(arg))
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
        crate::syntax::substitute_slots(
          &self_substituted,
          std::slice::from_ref(arg),
        )
      } else {
        crate::syntax::substitute_slots(body, std::slice::from_ref(arg))
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
      // MapAt[f, pos] / SubsetMap[f, pos] operator form: the applied
      // expression goes in the middle.
      if (name == "MapAt" || name == "SubsetMap") && args.len() == 2 {
        let new_args = vec![args[0].clone(), arg.clone(), args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // Insert[elem, pos] operator form: prepend the applied expression
      if name == "Insert" && args.len() == 2 {
        let new_args = vec![arg.clone(), args[0].clone(), args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // Nest[f, n][x] -> Nest[f, x, n]: the applied expression goes in the
      // middle (likewise NestList[f, n][x] -> NestList[f, x, n]).
      if (name == "Nest" || name == "NestList") && args.len() == 2 {
        let new_args = vec![args[0].clone(), arg.clone(), args[1].clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // Fold[f][list] -> Fold[f, list] (and likewise FoldList): the operator
      // form appends the applied list after the accumulating function.
      if (name == "Fold" || name == "FoldList") && args.len() == 1 {
        let new_args = vec![args[0].clone(), arg.clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // NearestTo[x][data] -> Nearest[data, x] (operator form of Nearest),
      // so `Map[NearestTo[x], lists]` finds the nearest in each list.
      if name == "NearestTo" && (args.len() == 1 || args.len() == 2) {
        let mut new_args = vec![arg.clone()];
        new_args.extend(args.iter().cloned());
        return evaluate_function_call_ast("Nearest", &new_args);
      }
      // Nearest[data] is a NearestFunction; applying it forwards to the direct
      // form: Nearest[data][x] -> Nearest[data, x].
      if name == "Nearest" && args.len() == 1 {
        let mut new_args = args.to_vec();
        new_args.push(arg.clone());
        return evaluate_function_call_ast("Nearest", &new_args);
      }
      // Distribution operator forms: CDF[dist][x] -> CDF[dist, x], and the
      // same for PDF / Quantile / InverseCDF / SurvivalFunction /
      // HazardFunction. The applied argument is appended. Guarded so the head
      // already holds a distribution object (a FunctionCall like
      // NormalDistribution[0, 1]).
      if matches!(
        name.as_str(),
        "CDF"
          | "PDF"
          | "Quantile"
          | "InverseCDF"
          | "SurvivalFunction"
          | "HazardFunction"
      ) && args.len() == 1
        && matches!(&args[0], Expr::FunctionCall { .. })
      {
        let new_args = vec![args[0].clone(), arg.clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // LinearSolve[m][b] -> LinearSolve[m, b] (operator form).
      if name == "LinearSolve"
        && args.len() == 1
        && matches!(&args[0], Expr::List(_))
      {
        let new_args = vec![args[0].clone(), arg.clone()];
        return evaluate_function_call_ast(name, &new_args);
      }
      // Curried function: f[a] applied to b becomes f[a, b]
      // Special case: operator forms where f[x][y] becomes f[y, x]
      // (the applied argument becomes the first parameter)
      if is_subject_first_operator(name)
        && args.len() == 1
        && operator_form_accepts_subject(name, arg)
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![arg.clone(), args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
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
          crate::syntax::substitute_slots(&body, std::slice::from_ref(arg));
        evaluate_expr_to_expr(&substituted)
      } else {
        // Treat as a function name
        evaluate_function_call_ast(&func_str, std::slice::from_ref(arg))
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

/// Whether an operator form may fire on this subject. `Merge`'s only does on a
/// list of associations or rules; anything else reports `Merge::list1` and
/// stays a curried call, so `Merge[{<|a -> 1|>}][Total]` is not rewritten to
/// `Merge[Total, {<|a -> 1|>}]`.
fn operator_form_accepts_subject(name: &str, subject: &Expr) -> bool {
  if name != "Merge" {
    return true;
  }
  let is_rule =
    |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let is_assoc_like = |e: &Expr| {
    matches!(e, Expr::Association(_))
      || is_rule(e)
      || matches!(e, Expr::List(items) if items.iter().all(&is_rule))
  };
  let accepted =
    matches!(subject, Expr::List(items) if items.iter().all(is_assoc_like));
  if !accepted {
    crate::emit_message(&format!(
      "Merge::list1: The argument {} is not a valid list of Associations or rules or lists of rules.",
      crate::syntax::expr_to_output(subject)
    ));
  }
  accepted
}

/// Functions whose one-argument call is an operator form that takes the
/// subject *first* once applied: `f[spec][expr]` is `f[expr, spec]`
/// (`Select[EvenQ][list]` → `Select[list, EvenQ]`). Both application paths
/// consult this one list so they cannot drift apart.
fn is_subject_first_operator(name: &str) -> bool {
  matches!(
    name,
    "ReplaceAll"
      | "ReplaceRepeated"
      | "StringStartsQ"
      | "StringEndsQ"
      | "StringContainsQ"
      | "StringFreeQ"
      | "StringMatchQ"
      | "StringReplace"
      | "StringCases"
      | "StringPosition"
      | "StringDelete"
      | "SequenceReplace"
      | "MemberQ"
      | "Select"
      | "SelectFirst"
      | "AllTrue"
      | "AnyTrue"
      | "NoneTrue"
      | "Discard"
      | "KeyTake"
      | "KeyDrop"
      | "KeySelect"
      | "KeySortBy"
      | "Merge"
      | "Lookup"
      | "TakeLargest"
      | "TakeSmallest"
      | "SortBy"
      | "OrderingBy"
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
      | "SubsetReplace"
      | "ContainsAll"
      | "ContainsAny"
      | "ContainsNone"
      | "ContainsOnly"
      | "ContainsExactly"
      | "GatherBy"
      | "SplitBy"
      | "DeleteDuplicatesBy"
  )
}

/// `DateObject[…]["Granularity"]`, and the clock components of a
/// `TimeObject[{h, m, s}, …]`. `None` for anything the caller should resolve
/// through `DateValue` instead.
fn date_object_tag_property(
  head: &str,
  obj_args: &[Expr],
  property: &str,
) -> Option<Expr> {
  if property == "Granularity" {
    return Some(match obj_args.get(1) {
      Some(Expr::String(tag)) => Expr::String(tag.clone()),
      // An untagged DateObject list is as fine-grained as its length.
      _ => Expr::String(
        match obj_args.first() {
          Some(Expr::List(items)) => match items.len() {
            1 => "Year",
            2 => "Month",
            3 => "Day",
            4 => "Hour",
            5 => "Minute",
            _ => "Instant",
          },
          _ => return None,
        }
        .to_string(),
      ),
    });
  }
  if head != "TimeObject" {
    return None;
  }
  let Some(Expr::List(components)) = obj_args.first() else {
    return None;
  };
  let index = match property {
    "Hour" => 0,
    "Minute" => 1,
    "Second" => 2,
    _ => return None,
  };
  components.get(index).cloned()
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
      // Query[ops...][data] — successive-level query application. A Dataset
      // subject goes through the Dataset query so the answer keeps (or drops)
      // the wrapper the same way `ds[ops…]` does.
      if let Expr::FunctionCall {
        name: ds_name,
        args: ds_args,
      } = &args[0]
        && ds_name == "Dataset"
        && !ds_args.is_empty()
      {
        return crate::functions::dataset_ast::dataset_query(
          ds_args, func_args,
        );
      }
      crate::functions::query_ast::apply_query(func_args, &args[0])
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Entity" && func_args.len() == 2 => {
      // Entity["type", "name"]["property"] — property access on entities
      crate::functions::entity_ast::entity_property_access(func_args, args)
    }
    // GeometricScene[{sym -> value, ...}, {primitives...}]["Graphics"] —
    // substitute point symbols into the primitives and return an ordinary
    // Graphics[{...}] expression. ["Elements"] returns the resolved point
    // definitions. Unknown properties keep the curried form unevaluated.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "GeometricScene"
      && (func_args.len() == 2 || func_args.len() == 3)
      && args.len() == 1
      && matches!(&args[0], Expr::String(_)) =>
    {
      let Expr::String(prop) = &args[0] else {
        unreachable!();
      };
      match prop.as_str() {
        "Graphics" => {
          crate::functions::graphics::geometric_scene_graphics(func_args)
        }
        "Elements" => {
          crate::functions::graphics::geometric_scene_elements(func_args)
        }
        _ => Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        }),
      }
    }
    // Around[x, δ]["Value"] / ["Uncertainty"] — property extraction on an
    // uncertain value. Unknown properties keep the curried form unevaluated.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Around"
      && func_args.len() == 2
      && args.len() == 1
      && matches!(&args[0], Expr::String(_)) =>
    {
      let Expr::String(prop) = &args[0] else {
        unreachable!();
      };
      match prop.as_str() {
        "Value" => Ok(func_args[0].clone()),
        "Uncertainty" => Ok(func_args[1].clone()),
        _ => Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        }),
      }
    }
    // EchoFunction[f][expr] — print ">> f[expr]" (or ">> label f[expr]"
    // for EchoFunction[label, f], or the expression itself for
    // EchoFunction[]) and return expr unchanged, like Echo.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "EchoFunction" && func_args.len() <= 2 && args.len() == 1 => {
      let expr = &args[0];
      let f = match func_args.len() {
        0 => None,
        1 => Some(&func_args[0]),
        _ => Some(&func_args[1]),
      };
      let display = match f {
        None => expr.clone(),
        Some(Expr::Identifier(f_name)) => {
          evaluate_function_call_ast(f_name, std::slice::from_ref(expr))?
        }
        Some(other) => apply_curried_call(other, std::slice::from_ref(expr))?,
      };
      let line = if func_args.len() == 2 {
        format!(
          ">> {} {}",
          crate::syntax::expr_to_output(&func_args[0]),
          crate::syntax::expr_to_output(&display)
        )
      } else {
        format!(">> {}", crate::syntax::expr_to_output(&display))
      };
      println!("{line}");
      crate::capture_stdout(&line);
      Ok(expr.clone())
    }
    // EchoLabel[label][expr] — print ">> label expr" and return expr,
    // exactly Echo[expr, label] in operator form.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "EchoLabel" && func_args.len() == 1 && args.len() == 1 => {
      let line = format!(
        ">> {} {}",
        crate::syntax::expr_to_output(&func_args[0]),
        crate::syntax::expr_to_output(&args[0])
      );
      println!("{line}");
      crate::capture_stdout(&line);
      Ok(args[0].clone())
    }
    // DateObject[…]["Granularity"] / TimeObject[…]["Granularity"] report the
    // granularity tag itself, and a TimeObject's components come from its
    // {hour, minute, second} list.
    Expr::FunctionCall {
      name,
      args: obj_args,
    } if (name == "DateObject" || name == "TimeObject")
      && args.len() == 1
      && matches!(&args[0], Expr::String(p)
        if date_object_tag_property(name, obj_args, p).is_some()) =>
    {
      let Expr::String(property) = &args[0] else {
        unreachable!();
      };
      Ok(
        date_object_tag_property(name, obj_args, property)
          .expect("guarded above"),
      )
    }
    // DateObject[...]["property"] — extract a date component (e.g. "Day",
    // "DayName", "Week"). Delegates to DateValue, which already resolves every
    // such property; if it cannot, the curried form is kept unevaluated.
    Expr::FunctionCall { name, .. }
      if name == "DateObject"
        && args.len() == 1
        && matches!(&args[0], Expr::String(_)) =>
    {
      let result = evaluate_function_call_ast(
        "DateValue",
        &[func.clone(), args[0].clone()],
      )?;
      if matches!(&result, Expr::FunctionCall { name, .. } if name == "DateValue")
      {
        Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        })
      } else {
        Ok(result)
      }
    }
    // HTTPRequest[…][prop] — property access on the symbolic HTTP request
    // object: a property string ("Method", "URL", …), the Method symbol, or
    // a list of those (yielding an association). Unknown properties emit
    // HTTPRequest::notprop and keep the curried form unevaluated.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "HTTPRequest"
      && args.len() == 1
      && matches!(
        &args[0],
        Expr::String(_) | Expr::Identifier(_) | Expr::List(_)
      ) =>
    {
      match crate::functions::http_ast::http_request_extract(
        func_args, &args[0],
      ) {
        Some(result) => Ok(result),
        None => Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        }),
      }
    }
    // Success[tag, assoc]["prop"] / Exception[tags, assoc]["prop"] — plain
    // association lookups. They differ from Failure (and from each other) in
    // what an absent key reports.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if (name == "Success" || name == "Exception")
      && args.len() == 1
      && matches!(&args[0], Expr::String(_)) =>
    {
      let Expr::String(prop) = &args[0] else {
        unreachable!()
      };
      let looked_up = if name == "Success" {
        crate::functions::confirm_ast::success_property(func_args, prop)
      } else {
        crate::functions::confirm_ast::exception_property(func_args, prop)
      };
      match looked_up {
        Some(result) => Ok(result),
        None => Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        }),
      }
    }
    // Failure[tag, assoc]["property"] — the tag, the message with its
    // parameters filled in, the standard property list, or any key of the
    // association. An unknown property gives Missing["NotAvailable", prop].
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Failure"
      && args.len() == 1
      && matches!(&args[0], Expr::String(_)) =>
    {
      let Expr::String(prop) = &args[0] else {
        unreachable!()
      };
      match crate::functions::confirm_ast::failure_property(func_args, prop) {
        Some(result) => Ok(result),
        None => Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        }),
      }
    }
    // Molecule[…]["property"] — property access on a molecule object
    // (e.g. "AtomCount", "MolecularFormula"). Unsupported properties and
    // invalid molecules keep the curried form unevaluated.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Molecule"
      && args.len() == 1
      && matches!(&args[0], Expr::String(_)) =>
    {
      let Expr::String(prop) = &args[0] else {
        unreachable!()
      };
      match crate::functions::molecule_ast::molecule_property(func_args, prop) {
        Some(result) => Ok(result),
        None => Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        }),
      }
    }
    // TimeSeries[…][t] — value lookup at time `t` (a date or number) or, for a
    // string argument, a path-property accessor (e.g. ts["Values"], ts["Path"]).
    Expr::FunctionCall { name, .. }
      if (name == "TimeSeries" || name == "EventSeries") && args.len() == 1 =>
    {
      crate::functions::timeseries_ast::apply_time_series(func, &args[0])
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "Interpreter" && func_args.len() == 1 => {
      // Interpreter["Country"][input] — resolve input to an Entity.
      crate::functions::country_data::apply_interpreter(&func_args[0], args)
    }
    // DiscreteWaveletData[…][wind | All | Automatic | "prop"(, form)] —
    // wavelet coefficient and property access.
    Expr::FunctionCall { name, .. }
      if name == "DiscreteWaveletData"
        && !args.is_empty()
        && args.len() <= 2 =>
    {
      crate::functions::wavelet_ast::data::apply_dwd(func, args)
    }
    // ContinuousWaveletData[…][{oct,voc} | All | "prop"(, form)]
    Expr::FunctionCall { name, .. }
      if name == "ContinuousWaveletData"
        && !args.is_empty()
        && args.len() <= 2 =>
    {
      crate::functions::wavelet_ast::continuous::apply_cwd(func, args)
    }
    // LiftingFilterData[wave]["prop"] — filter properties of the wavelet
    // the lifting data was derived from.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "LiftingFilterData"
      && func_args.len() == 1
      && args.len() == 1
      && matches!(&args[0], Expr::String(_)) =>
    {
      let Expr::String(prop) = &args[0] else {
        unreachable!()
      };
      match prop.as_str() {
        "Wavelet" => Ok(func_args[0].clone()),
        "PrimalLowpass" | "PrimalHighpass" | "DualLowpass" | "DualHighpass" => {
          Ok(crate::functions::wavelet_ast::filters::wavelet_filter_coefficients_ast(&[
            func_args[0].clone(),
            args[0].clone(),
          ]))
        }
        _ => Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        }),
      }
    }
    // ShortTimeFourierData[…]["property"] — property access on the data
    // object produced by ShortTimeFourier (e.g. "Data", "SampleRate").
    Expr::FunctionCall { name, .. }
      if name == "ShortTimeFourierData"
        && args.len() == 1
        && crate::functions::audio_ast::spectral::stfd_property(
          func, &args[0],
        )
        .is_some() =>
    {
      Ok(
        crate::functions::audio_ast::spectral::stfd_property(func, &args[0])
          .unwrap(),
      )
    }
    // AssessmentFunction[spec][answer] — grade the answer, returning an
    // AssessmentResultObject.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "AssessmentFunction" && args.len() == 1 => {
      crate::functions::assessment_ast::apply_assessment_function(
        func_args, &args[0],
      )
    }
    // QuestionObject[q, assess][answer] — grade the answer through the
    // embedded assessment.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "QuestionObject" && args.len() == 1 => {
      crate::functions::assessment_ast::apply_question_object(
        func_args, &args[0],
      )
    }
    // AssessmentResultObject[<|…|>]["property"] — property access
    // (e.g. "Score", "AnswerCorrect", or All for the whole association).
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "AssessmentResultObject" && args.len() == 1 => {
      crate::functions::assessment_ast::apply_result_object(func_args, &args[0])
    }
    // BooleanFunction[n, k][b1, …, bk] — the integer-indexed boolean function.
    // The result is bit `v` of `n`, where `v` is the k arguments read as a
    // binary number (first argument most significant, True/1 → 1, False/0 → 0).
    // e.g. BooleanFunction[7, 2][True, False]: v = 10b = 2, (7 >> 2) & 1 = 1 →
    // True. Negative `n` is two's-complement (BooleanFunction[-1, k] is all
    // True). Non-boolean args or a wrong count leave the call unevaluated.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "BooleanFunction"
      && func_args.len() == 2
      && matches!(&func_args[0], Expr::Integer(_))
      && matches!(&func_args[1], Expr::Integer(k) if *k >= 0 && *k as usize == args.len()) =>
    {
      let Expr::Integer(n) = &func_args[0] else {
        unreachable!()
      };
      let mut v: i128 = 0;
      let mut ok = true;
      for a in args {
        let bit = match a {
          Expr::Identifier(s) if s == "True" => 1,
          Expr::Identifier(s) if s == "False" => 0,
          Expr::Integer(1) => 1,
          Expr::Integer(0) => 0,
          _ => {
            ok = false;
            break;
          }
        };
        v = (v << 1) | bit;
      }
      if ok {
        return Ok(bool_expr((n >> v) & 1 == 1));
      }
      Ok(Expr::CurriedCall {
        func: Box::new(func.clone()),
        args: args.to_vec(),
      })
    }
    // BooleanCountingFunction[spec, n][b1, …] — True when the number of True
    // arguments is one of the counts the spec names. Only literal arguments
    // decide it; a symbolic one leaves the call as it stands.
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "BooleanCountingFunction"
      && func_args.len() == 2
      && matches!(&func_args[1], Expr::Integer(k) if *k >= 0) =>
    {
      let Expr::Integer(n) = &func_args[1] else {
        unreachable!()
      };
      let mut true_count = 0usize;
      let mut literal = true;
      for a in args {
        match a {
          Expr::Identifier(s) if s == "True" => true_count += 1,
          Expr::Identifier(s) if s == "False" => {}
          _ => {
            literal = false;
            break;
          }
        }
      }
      if literal
        && let Some(counts) =
          crate::functions::boolean_ast::counting_spec_counts(
            &func_args[0],
            *n as usize,
          )
      {
        return Ok(bool_expr(counts.contains(&true_count)));
      }
      Ok(Expr::CurriedCall {
        func: Box::new(func.clone()),
        args: args.to_vec(),
      })
    }
    // Distribution operator forms: CDF[dist][x] -> CDF[dist, x] (and PDF,
    // Quantile, InverseCDF, SurvivalFunction, HazardFunction). Guarded so the
    // head holds a distribution object (a FunctionCall).
    Expr::FunctionCall {
      name,
      args: func_args,
    } if matches!(
      name.as_str(),
      "CDF"
        | "PDF"
        | "Quantile"
        | "InverseCDF"
        | "SurvivalFunction"
        | "HazardFunction"
    ) && func_args.len() == 1
      && matches!(&func_args[0], Expr::FunctionCall { .. }) =>
    {
      let mut new_args = func_args.to_vec();
      new_args.extend(args.iter().cloned());
      evaluate_function_call_ast(name, &new_args)
    }
    // Region operator forms: RegionMember[reg][pt] -> RegionMember[reg, pt]
    // (and the analogous RegionDistance / RegionNearest / SignedRegionDistance
    // one-argument operators). The head holds a region object (a FunctionCall).
    Expr::FunctionCall {
      name,
      args: func_args,
    } if matches!(
      name.as_str(),
      "RegionMember"
        | "RegionDistance"
        | "RegionNearest"
        | "SignedRegionDistance"
    ) && func_args.len() == 1
      && matches!(&func_args[0], Expr::FunctionCall { .. }) =>
    {
      let mut new_args = func_args.to_vec();
      new_args.extend(args.iter().cloned());
      evaluate_function_call_ast(name, &new_args)
    }
    // LinearSolve operator form: LinearSolve[m][b] -> LinearSolve[m, b]. (The
    // bare LinearSolve[m] is wolframscript's opaque LinearSolveFunction, which
    // Woxi keeps unevaluated, but the application solves the system.)
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "LinearSolve"
      && func_args.len() == 1
      && matches!(&func_args[0], Expr::List(_)) =>
    {
      let mut new_args = func_args.to_vec();
      new_args.extend(args.iter().cloned());
      evaluate_function_call_ast(name, &new_args)
    }
    // TreeFold/TreeMap operator form: T[f][tree] -> T[f, tree].
    Expr::FunctionCall {
      name,
      args: func_args,
    } if (name == "TreeFold" || name == "TreeMap") && func_args.len() == 1 => {
      let mut new_args = func_args.to_vec();
      new_args.extend(args.iter().cloned());
      evaluate_function_call_ast(name, &new_args)
    }
    // TreeReplacePart / TreeSelect operator form: T[spec][tree] ->
    // T[tree, spec] (the tree argument comes first).
    Expr::FunctionCall {
      name,
      args: func_args,
    } if (name == "TreeReplacePart" || name == "TreeSelect")
      && func_args.len() == 1 =>
    {
      let mut new_args = args.to_vec();
      new_args.extend(func_args.iter().cloned());
      evaluate_function_call_ast(name, &new_args)
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
    // SparseArray[…]["property"] — grid metadata queries.
    Expr::FunctionCall {
      name,
      args: sa_args,
    } if name == "SparseArray"
      && args.len() == 1
      && matches!(&args[0], Expr::String(_)) =>
    {
      let Expr::String(prop) = &args[0] else {
        unreachable!()
      };
      if let Some(result) =
        crate::functions::list_helpers_ast::sparse_array_property(sa_args, prop)
      {
        Ok(result)
      } else {
        crate::emit_message(&format!(
          "SparseArray::nomthd: There is no method {prop} for SparseArray objects."
        ));
        Ok(Expr::CurriedCall {
          func: Box::new(func.clone()),
          args: args.to_vec(),
        })
      }
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "InterpolatingFunction"
      && (2..=4).contains(&func_args.len()) =>
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
    } if (name == "FoldWhile" || name == "FoldWhileList")
      && func_args.len() == 2
      && args.len() == 1 =>
    {
      // FoldWhile[f, test][list] — the operator form folds the given list,
      // taking its first element as the initial value.
      evaluate_function_call_ast(
        name,
        &[func_args[0].clone(), args[0].clone(), func_args[1].clone()],
      )
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "TuringMachine" && func_args.len() == 1 && args.len() == 1 => {
      // TuringMachine[rule][init] — the operator form is one step.
      crate::functions::turing_machine_ast::turing_machine_ast(&[
        func_args[0].clone(),
        args[0].clone(),
      ])
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "SubstitutionSystem"
      && func_args.len() == 1
      && args.len() == 1 =>
    {
      // SubstitutionSystem[rules][init] — the operator form is one step.
      evaluate_function_call_ast(
        "SubstitutionSystem",
        &[func_args[0].clone(), args[0].clone()],
      )
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "CellularAutomaton"
      && func_args.len() == 1
      && args.len() == 1 =>
    {
      // CellularAutomaton[rule][init] — the operator form is one step.
      crate::functions::cellular_automaton_ast::cellular_automaton_ast(&[
        func_args[0].clone(),
        args[0].clone(),
      ])
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
    } if name == "ColorDataFunction" && func_args.len() == 4 => {
      // ColorDataFunction[scheme, "Gradients", range, blend][t] — apply the
      // stored blend function (the structured form ColorData["scheme"]
      // evaluates to) at t.
      apply_curried_call(&func_args[3], args)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "BezierFunction" && func_args.len() == 1 => {
      // BezierFunction[{{p1}, {p2}, ...}][t] — evaluate Bezier curve at t
      Ok(evaluate_bezier_function(func_args, args))
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
        Ok(evaluate_bezier_function(&pts, args))
      } else {
        Err(InterpreterError::EvaluationError(
          "BezierFunction: invalid structured form".into(),
        ))
      }
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } if name == "BSplineFunction" && func_args.len() == 9 => {
      // BSplineFunction[dim, ranges, degrees, closed, {net, Automatic},
      // knots, ...][params] — evaluate the spline at the given parameters.
      Ok(evaluate_bspline_function(func_args, args))
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
      // CompiledFunction[specs, body][args…] — bind each argument at its
      // declared type and evaluate the body.
      let specs: Vec<&Expr> = match &func_args[0] {
        Expr::List(items) => items.iter().collect(),
        other => vec![other],
      };
      let body = &func_args[1];
      let bound: Vec<(String, Expr)> = specs
        .iter()
        .zip(args.iter())
        .filter_map(|(spec, arg)| {
          let (name, real) = compile_arg_spec(spec)?;
          Some((
            name,
            if real {
              coerce_to_real(arg)
            } else {
              arg.clone()
            },
          ))
        })
        .collect();
      let bindings: Vec<(&str, &Expr)> =
        bound.iter().map(|(p, a)| (p.as_str(), a)).collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      let result = evaluate_expr_to_expr(&substituted)?;
      // A compiled function whose arguments include a real works in machine
      // reals throughout, so exact numbers that leaked in from literals come
      // back inexact: `Compile[{{x, _Real, 0}}, Clip[x, {-4, 4}]][-5.]` is
      // `-4.`, not the exact bound. An all-integer signature keeps its exact
      // result.
      let any_real = specs
        .iter()
        .zip(args.iter())
        .any(|(spec, _)| matches!(compile_arg_spec(spec), Some((_, true))));
      Ok(if any_real {
        coerce_to_real(&result)
      } else {
        result
      })
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } => {
      // Property access on a canonical music object, e.g.
      // MusicNote[<|…|>]["Pitch"] or MusicChord[<|…|>]["PitchList"].
      if let Some(result) = crate::functions::music_ast::music_property_access(
        name, func_args, args,
      ) {
        return Ok(result);
      }
      // Curried function: f[a][b] becomes f[a, b]
      // Special case: operator forms where f[x][y] becomes f[y, x]
      if (name == "MapAt" || name == "SubsetMap")
        && func_args.len() == 2
        && args.len() == 1
      {
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
      // Nearest[data] is a NearestFunction; applying it forwards to the direct
      // form: Nearest[data][x] -> Nearest[data, x], and
      // Nearest[data][x, n] -> Nearest[data, x, n].
      if name == "Nearest"
        && func_args.len() == 1
        && (1..=2).contains(&args.len())
      {
        let mut new_args = func_args.to_vec();
        new_args.extend(args.iter().cloned());
        return evaluate_function_call_ast("Nearest", &new_args);
      }
      if is_subject_first_operator(name)
        && func_args.len() == 1
        && args.len() == 1
        && operator_form_accepts_subject(name, &args[0])
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![args[0].clone(), func_args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if (name == "Nest" || name == "NestList")
        && func_args.len() == 2
        && args.len() == 1
      {
        // Nest[f, n][x] -> Nest[f, x, n] (likewise NestList)
        let new_args =
          vec![func_args[0].clone(), args[0].clone(), func_args[1].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if (name == "Fold" || name == "FoldList")
        && func_args.len() == 1
        && args.len() == 1
      {
        // Fold[f][list] -> Fold[f, list] (and likewise FoldList)
        let new_args = vec![func_args[0].clone(), args[0].clone()];
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
        for f in func_args {
          let intermediate = apply_curried_call(f, &result)?;
          result = vec![intermediate];
        }
        Ok(result.into_iter().next().unwrap())
      } else if (name == "MapAt" || name == "SubsetMap")
        && func_args.len() == 2
        && args.len() == 1
      {
        // MapAt[f, pos][expr] -> MapAt[f, expr, pos] (likewise SubsetMap)
        let new_args =
          vec![func_args[0].clone(), args[0].clone(), func_args[1].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if name == "ReplaceAt" && func_args.len() == 2 && args.len() == 1 {
        // ReplaceAt[rules, pos][expr] -> ReplaceAt[expr, rules, pos]. Unlike
        // MapAt, ReplaceAt takes the subject as its FIRST argument.
        let new_args =
          vec![args[0].clone(), func_args[0].clone(), func_args[1].clone()];
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
            Ok(call(
              "Missing",
              vec![Expr::String("KeyAbsent".to_string()), key.clone()],
            ))
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
        // `Derivative[n][InterpolatingFunction[…]]` is another
        // InterpolatingFunction that carries the derivative order, so `f'`
        // stays an interpolating function the way wolframscript reports it.
        if name == "Derivative"
          && func_args.len() == 1
          && args.len() == 1
          && let Expr::Integer(extra) = &func_args[0]
          && *extra >= 0
          && let Expr::FunctionCall {
            name: interp_name,
            args: interp_args,
          } = &args[0]
          && interp_name == "InterpolatingFunction"
          && (2..=4).contains(&interp_args.len())
          && !matches!(interp_args.get(2), Some(Expr::List(_)))
        {
          let mut new_args: Vec<Expr> = interp_args.to_vec();
          while new_args.len() < 3 {
            new_args.push(Expr::Integer(1));
          }
          let previous = match new_args.get(3) {
            Some(Expr::Integer(n)) => *n,
            _ => 0,
          };
          let total = previous + extra;
          if new_args.len() == 4 {
            new_args[3] = Expr::Integer(total);
          } else {
            new_args.push(Expr::Integer(total));
          }
          return Ok(call("InterpolatingFunction", new_args));
        }
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
        Ok(call(
          "Missing",
          vec![Expr::String("KeyAbsent".to_string()), args[0].clone()],
        ))
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
    // Rules and patterns are heads in the same sense: `(u -> v)[x]` has head
    // `u -> v`, so collapsing it to a symbol named "u -> v" both misprints it
    // and makes `Head[Head[…]]` report Symbol instead of Rule.
    Expr::BinaryOp { .. }
    | Expr::UnaryOp { .. }
    | Expr::Comparison { .. }
    | Expr::Rule { .. }
    | Expr::RuleDelayed { .. }
    | Expr::Pattern { .. }
    | Expr::PatternOptional { .. } => Ok(Expr::CurriedCall {
      func: Box::new(func.clone()),
      args: args.to_vec(),
    }),
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
) -> crate::syntax::Expr {
  if args.len() != 1 {
    return unevaluated("BezierFunction", func_args);
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
        return unevaluated("BezierFunction", func_args);
      }
    }
    _ => {
      // Symbolic t: return unevaluated
      return unevaluated("BezierFunction", func_args);
    }
  };

  // Extract control points from func_args[0] which should be a list of points
  let Expr::List(points) = &func_args[0] else {
    return unevaluated("BezierFunction", func_args);
  };

  if points.is_empty() {
    return unevaluated("BezierFunction", func_args);
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
                return unevaluated("BezierFunction", func_args);
              }
            }
            _ => {
              return unevaluated("BezierFunction", func_args);
            }
          }
        }
        ctrl_pts.push(fcoords);
      }
      _ => {
        return unevaluated("BezierFunction", func_args);
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

  Expr::List(work[0].iter().map(|&v| Expr::Real(v)).collect())
}

/// Knot span index `k` with `knots[k] <= u < knots[k+1]` (clamped at the
/// right end), for `n+1` control points of degree `p`. Standard binary
/// search from The NURBS Book.
fn bspline_find_span(n: usize, p: usize, u: f64, knots: &[f64]) -> usize {
  if u >= knots[n + 1] {
    return n;
  }
  if u <= knots[p] {
    return p;
  }
  let (mut low, mut high) = (p, n + 1);
  let mut mid = usize::midpoint(low, high);
  while u < knots[mid] || u >= knots[mid + 1] {
    if u < knots[mid] {
      high = mid;
    } else {
      low = mid;
    }
    mid = usize::midpoint(low, high);
  }
  mid
}

/// Evaluate a B-spline of degree `p` with the given `knots` and `ctrl` points
/// at parameter `u`, using de Boor's algorithm.
fn de_boor(p: usize, knots: &[f64], ctrl: &[Vec<f64>], u: f64) -> Vec<f64> {
  let n = ctrl.len() - 1;
  let dim = ctrl[0].len();
  let span = bspline_find_span(n, p, u, knots);
  let mut d: Vec<Vec<f64>> =
    (0..=p).map(|j| ctrl[span - p + j].clone()).collect();
  for r in 1..=p {
    for j in (r..=p).rev() {
      let i = j + span - p;
      let denom = knots[i + 1 + p - r] - knots[i];
      let alpha = if denom.abs() < 1e-12 {
        0.0
      } else {
        (u - knots[i]) / denom
      };
      for c in 0..dim {
        d[j][c] = (1.0 - alpha) * d[j - 1][c] + alpha * d[j][c];
      }
    }
  }
  d[p].clone()
}

/// Extract a numeric coordinate vector from a list expression.
fn bspline_point_coords(e: &Expr) -> Option<Vec<f64>> {
  match e {
    Expr::List(coords) => coords
      .iter()
      .map(crate::functions::math_ast::try_eval_to_f64)
      .collect(),
    _ => None,
  }
}

/// Evaluate the structured `BSplineFunction[...]` object at the supplied
/// parameters (one per spline dimension). Curves take a single parameter and
/// surfaces take two; the control points are evaluated via (tensor-product)
/// de Boor. Non-numeric parameters leave the application unevaluated.
fn evaluate_bspline_function(
  func_args: &[Expr],
  args: &[Expr],
) -> crate::syntax::Expr {
  // When the spline can't be sampled (symbolic parameters, malformed form),
  // echo the object applied to its parameters: `BSplineFunction[...][params]`.
  let unevaluated = || Expr::CurriedCall {
    func: Box::new(unevaluated("BSplineFunction", func_args)),
    args: args.to_vec(),
  };

  let dim = match &func_args[0] {
    Expr::Integer(d) => *d as usize,
    _ => return unevaluated(),
  };
  if args.len() != dim {
    return unevaluated();
  }

  // Parameters must be numeric to sample the spline.
  let params: Option<Vec<f64>> = args
    .iter()
    .map(crate::functions::math_ast::try_eval_to_f64)
    .collect();
  let Some(params) = params else {
    return unevaluated();
  };

  // Degrees, control net and knot vectors from the structured form.
  let degrees: Vec<usize> = match &func_args[2] {
    Expr::List(ds) => ds
      .iter()
      .filter_map(|d| match d {
        Expr::Integer(n) => Some(*n as usize),
        _ => None,
      })
      .collect(),
    _ => return unevaluated(),
  };
  let net = match &func_args[4] {
    Expr::List(slot) if !slot.is_empty() => &slot[0],
    _ => return unevaluated(),
  };
  let knot_sets: Vec<Vec<f64>> = match &func_args[5] {
    Expr::List(ks) => ks
      .iter()
      .map(|k| match k {
        Expr::List(vs) => vs
          .iter()
          .map(crate::functions::math_ast::try_eval_to_f64)
          .collect::<Option<Vec<f64>>>(),
        _ => None,
      })
      .collect::<Option<Vec<_>>>()
      .unwrap_or_default(),
    _ => return unevaluated(),
  };

  if degrees.len() != dim || knot_sets.len() != dim {
    return unevaluated();
  }

  let result = match dim {
    1 => {
      let ctrl: Option<Vec<Vec<f64>>> = match net {
        Expr::List(pts) => pts.iter().map(bspline_point_coords).collect(),
        _ => None,
      };
      let ctrl = match ctrl {
        Some(c) if !c.is_empty() => c,
        _ => return unevaluated(),
      };
      de_boor(degrees[0], &knot_sets[0], &ctrl, params[0])
    }
    2 => {
      // Tensor-product surface: collapse the v-direction per row, then the
      // u-direction across the resulting points.
      let rows: Option<Vec<Vec<Vec<f64>>>> = match net {
        Expr::List(rows) => rows
          .iter()
          .map(|row| match row {
            Expr::List(pts) => {
              pts.iter().map(bspline_point_coords).collect::<Option<_>>()
            }
            _ => None,
          })
          .collect(),
        _ => None,
      };
      let rows = match rows {
        Some(r) if !r.is_empty() && r.iter().all(|row| !row.is_empty()) => r,
        _ => return unevaluated(),
      };
      let collapsed: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| de_boor(degrees[1], &knot_sets[1], row, params[1]))
        .collect();
      de_boor(degrees[0], &knot_sets[0], &collapsed, params[0])
    }
    _ => return unevaluated(),
  };

  Expr::List(result.into_iter().map(Expr::Real).collect())
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
      func: Box::new(call1("TransformationFunction", matrix.clone())),
      args: args.to_vec(),
    });
  }
  let point = &args[0];
  let Expr::List(coords) = point else {
    return Ok(Expr::CurriedCall {
      func: Box::new(call1("TransformationFunction", matrix.clone())),
      args: args.to_vec(),
    });
  };

  let Expr::List(rows) = matrix else {
    return Ok(Expr::CurriedCall {
      func: Box::new(call1("TransformationFunction", matrix.clone())),
      args: args.to_vec(),
    });
  };

  let n = coords.len();
  // Build homogeneous coordinate vector: {x1, ..., xn, 1}
  let mut hom = coords.clone();
  hom.push(Expr::Integer(1));

  // Multiply: take first n rows, dot with homogeneous vector
  let mut result = Vec::with_capacity(n);
  for i in 0..n {
    let Expr::List(row) = &rows[i] else {
      return Ok(Expr::CurriedCall {
        func: Box::new(call1("TransformationFunction", matrix.clone())),
        args: args.to_vec(),
      });
    };
    // Dot product of row with homogeneous vector
    let dot = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: row
        .iter()
        .zip(hom.iter())
        .map(|(a, b)| call("Times", vec![a.clone(), b.clone()]))
        .collect(),
    };
    result.push(evaluate_expr_to_expr(&dot)?);
  }

  // Homogeneous coordinate from the last row (rows[n]). For affine
  // transforms this row is {0, ..., 0, 1}, so h = 1 and the division below is
  // a no-op. For projective transforms (e.g. LinearFractionalTransform) the
  // last row is {w1, ..., wn, b}, so h = w.point + b rescales the result.
  if rows.len() > n
    && let Expr::List(last) = &rows[n]
    && last.len() == hom.len()
  {
    let h = evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: last
        .iter()
        .zip(hom.iter())
        .map(|(a, b)| call("Times", vec![a.clone(), b.clone()]))
        .collect(),
    })?;
    // Only rescale when the homogeneous coordinate is not the constant 1.
    if !matches!(&h, Expr::Integer(1)) {
      for comp in &mut result {
        *comp = evaluate_expr_to_expr(&div2(comp.clone(), h.clone()))?;
      }
    }
  }

  Ok(Expr::List(result.into()))
}
