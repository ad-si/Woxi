#[allow(unused_imports)]
use super::*;

use crate::functions::math_ast::{gcd_i128, rat_reduce};
use crate::syntax::bool_expr;

// Re-export crate types/functions for submodules (used by submodules via `use super::*`)
#[allow(unused_imports)]
pub(crate) use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
  unevaluated,
};
#[allow(unused_imports)]
pub(crate) use crate::{
  ENV, InterpreterError, PART_DEPTH, StoredValue, interpret,
};

pub mod arg_count;
mod association_functions;
mod audio_functions;
mod boolean_functions;
pub(crate) mod calculus_functions;
pub mod complex_and_special;
pub use complex_and_special::builtin_default_value;
pub use complex_and_special::builtin_default_value_at_position;
mod datetime_functions;
mod evaluation_control;
mod image_functions;
mod interval_functions;
mod io_functions;
mod linear_algebra_functions;
pub(crate) mod list_operations;
mod math_functions;
mod music_functions;
mod plotting;
mod polynomial_functions;
mod predicate_functions;
mod quantity_functions;
mod string_functions;
mod structural;
mod timeseries_functions;
mod wavelet_functions;

pub use association_functions::*;
pub use boolean_functions::*;
pub use calculus_functions::*;
pub use complex_and_special::*;
pub use datetime_functions::*;
pub use evaluation_control::*;
pub use image_functions::*;
pub use interval_functions::*;
pub use io_functions::*;
pub use linear_algebra_functions::*;
pub use list_operations::*;
pub use math_functions::*;
pub use plotting::*;
pub use polynomial_functions::*;
pub use predicate_functions::*;
pub use quantity_functions::*;
pub use string_functions::*;
pub use structural::*;

/// Transform a Dataset type expression for DeleteMissing:
/// Replace the fixed-length count in Vector[elem_type, N] with TypeSystem`AnyLength.
fn delete_missing_type(type_expr: &Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = type_expr
    && name == "TypeSystem`Vector"
    && args.len() == 2
  {
    return Expr::FunctionCall {
      name: "TypeSystem`Vector".to_string(),
      args: vec![
        args[0].clone(),
        Expr::Identifier("TypeSystem`AnyLength".to_string()),
      ]
      .into(),
    };
  }
  type_expr.clone()
}

/// Generate next lexicographic permutation in-place. Returns false when done.
fn next_permutation(arr: &mut [usize]) -> bool {
  let n = arr.len();
  if n <= 1 {
    return false;
  }
  // Find largest i such that arr[i] < arr[i+1]
  let mut i = n - 2;
  loop {
    if arr[i] < arr[i + 1] {
      break;
    }
    if i == 0 {
      return false;
    }
    i -= 1;
  }
  // Find largest j such that arr[i] < arr[j]
  let mut j = n - 1;
  while arr[j] <= arr[i] {
    j -= 1;
  }
  arr.swap(i, j);
  arr[i + 1..].reverse();
  true
}

pub fn evaluate_function_call_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Track recursion depth to prevent stack overflow from mutually-recursive
  // rules (e.g. ArcSec ↔ ArcCos cycles through built-in + user rules).
  // This catches cycles that bypass the depth guard in evaluate_expr_to_expr_impl
  // because they stay entirely within evaluate_function_call_ast_inner.
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
    return Ok(unevaluated(name, args));
  }

  stacker::maybe_grow(2 * 1024 * 1024, 4 * 1024 * 1024, || {
    evaluate_function_call_ast_inner(name, args)
  })
}

/// Promote any large `Expr::List` value in `values` from the Vec-backed
/// representation to the Tree-backed (imbl::Vector) representation.
///
/// Used before substituting function-call args into a user-defined body:
/// `substitute_variables` clones the bound value at every position the
/// parameter is referenced, so a 1500-element Vec list referenced twice
/// would otherwise cost two full O(N) clones. The Tree variant clones
/// in O(1) (Arc::clone of the imbl backbone), turning the substitution
/// cost from O(N × occurrences) into O(N + occurrences) for that arg.
///
/// Threshold of 16 keeps the Vec path for small argument lists where the
/// one-time conversion would dominate.
fn promote_lists_for_substitution(values: &mut [Expr]) {
  for v in values {
    if let Expr::List(items) = v
      && items.len() > 16
      && !items.is_tree()
    {
      items.upgrade_to_tree();
    }
  }
}

/// Distribute function call args to params using backtracking,
/// handling BlankSequence/BlankNullSequence params that consume variable numbers of args.
fn distribute_args_to_params(
  args: &[Expr],
  blank_types: &[u8],
  param_heads: &[Option<String>],
  param_defaults: &[Option<Expr>],
  param_idx: usize,
) -> Option<Vec<Vec<Expr>>> {
  stacker::maybe_grow(2 * 1024 * 1024, 4 * 1024 * 1024, || {
    distribute_args_to_params_impl(
      args,
      blank_types,
      param_heads,
      param_defaults,
      param_idx,
    )
  })
}

fn distribute_args_to_params_impl(
  args: &[Expr],
  blank_types: &[u8],
  param_heads: &[Option<String>],
  param_defaults: &[Option<Expr>],
  param_idx: usize,
) -> Option<Vec<Vec<Expr>>> {
  if param_idx >= blank_types.len() {
    return if args.is_empty() { Some(vec![]) } else { None };
  }

  let bt = blank_types[param_idx];
  let head = &param_heads[param_idx];

  if bt >= 2 {
    // Sequence param: consume min..max args
    let min_count: usize = if bt == 2 { 1 } else { 0 };
    let rest_min: usize = blank_types[param_idx + 1..]
      .iter()
      .zip(param_defaults[param_idx + 1..].iter())
      .map(|(&t, d)| {
        if d.is_some() {
          0
        } else if t >= 2 {
          if t == 2 { 1 } else { 0 }
        } else {
          1
        }
      })
      .sum();
    let max_count = if args.len() >= rest_min {
      args.len() - rest_min
    } else {
      return None;
    };

    for count in min_count..=max_count {
      let seq_args = &args[..count];
      // Check head constraint
      if let Some(h) = head
        && !seq_args
          .iter()
          .all(|a| crate::evaluator::pattern_matching::get_expr_head(a) == *h)
      {
        continue;
      }
      if let Some(mut rest) = distribute_args_to_params(
        &args[count..],
        blank_types,
        param_heads,
        param_defaults,
        param_idx + 1,
      ) {
        rest.insert(0, seq_args.to_vec());
        return Some(rest);
      }
    }
    None
  } else {
    // Regular param: consume 0 or 1 args
    if args.is_empty() {
      // Try using default
      if param_defaults[param_idx].is_some()
        && let Some(mut rest) = distribute_args_to_params(
          args,
          blank_types,
          param_heads,
          param_defaults,
          param_idx + 1,
        )
      {
        rest.insert(0, vec![]);
        return Some(rest);
      }
      return None;
    }
    // Check head constraint
    if let Some(h) = head
      && crate::evaluator::pattern_matching::get_expr_head(&args[0]) != *h
    {
      return None;
    }
    if let Some(mut rest) = distribute_args_to_params(
      &args[1..],
      blank_types,
      param_heads,
      param_defaults,
      param_idx + 1,
    ) {
      rest.insert(0, vec![args[0].clone()]);
      return Some(rest);
    }
    None
  }
}

/// Extract option bindings from function arguments for OptionsPattern.
/// Merges explicit options with stored Options[func_name] defaults.
fn collect_option_bindings(
  func_name: &str,
  params: &[String],
  effective_args: &[Expr],
  inline_opts: Option<&Vec<Expr>>,
) -> Option<Vec<(String, Expr)>> {
  // Find the __opts parameter
  let opts_idx = params.iter().position(|p| p.starts_with("__opts"))?;
  let opts_arg = &effective_args[opts_idx];

  // Get defaults: inline OptionsPattern[{...}] defaults take priority over Options[func_name]
  let defaults: Vec<Expr> = if let Some(inline) = inline_opts {
    inline.clone()
  } else {
    crate::FUNC_OPTIONS
      .with(|m| m.borrow().get(func_name).cloned())
      .unwrap_or_default()
  };

  // Build default bindings
  let mut bindings: Vec<(String, Expr)> = Vec::new();
  for rule in &defaults {
    if let Some((key, val)) = extract_rule_pair(rule) {
      bindings.push((key, val));
    }
  }

  // Extract explicit options from the argument. Wolfram accepts a
  // single rule, a Sequence of rules, a list of rules, or arbitrarily-
  // nested lists/Sequences thereof — `f[x, {{{n -> 4}}}]` should still
  // pull `n -> 4` out for OptionValue lookups.
  fn collect_rules(expr: &Expr, out: &mut Vec<Expr>) {
    match expr {
      Expr::Rule { .. } | Expr::RuleDelayed { .. } => out.push(expr.clone()),
      Expr::List(items) => {
        for item in items {
          collect_rules(item, out);
        }
      }
      Expr::FunctionCall { name, args } if name == "Sequence" => {
        for arg in args {
          collect_rules(arg, out);
        }
      }
      _ => {}
    }
  }
  let mut explicit_rules: Vec<Expr> = Vec::new();
  collect_rules(opts_arg, &mut explicit_rules);

  // Override defaults with explicit options
  for rule in &explicit_rules {
    if let Some((key, val)) = extract_rule_pair(rule) {
      if let Some(existing) = bindings.iter_mut().find(|(k, _)| k == &key) {
        existing.1 = val;
      } else {
        bindings.push((key, val));
      }
    }
  }

  Some(bindings)
}

/// Extract the key-value pair from a Rule or RuleDelayed expression.
/// Both Symbol keys (`m -> 7`) and String keys (`"m" -> 7`) collapse to
/// the bare name `"m"` so that `OptionValue[m]` and `OptionValue["m"]`
/// both find the same binding (matching Wolfram).
fn extract_rule_pair(rule: &Expr) -> Option<(String, Expr)> {
  match rule {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      let key = match pattern.as_ref() {
        Expr::Identifier(s) => s.clone(),
        Expr::String(s) => s.clone(),
        _ => expr_to_string(pattern),
      };
      Some((key, *replacement.clone()))
    }
    _ => None,
  }
}

/// Unwrap a possibly Labeled-wrapped graph edge and return its two endpoints,
/// regardless of direction. Handles Rule (`a -> b`), DirectedEdge,
/// UndirectedEdge and TwoWayRule (`a <-> b`).
fn edge_endpoints(edge: &Expr) -> Option<(Expr, Expr)> {
  edge_endpoints_dir(edge).map(|(s, d, _)| (s, d))
}

/// Like `edge_endpoints` but also reports whether the edge is directed.
/// Rule and DirectedEdge are directed; TwoWayRule and UndirectedEdge are not.
fn edge_endpoints_dir(edge: &Expr) -> Option<(Expr, Expr, bool)> {
  // Peel a Labeled[edge, label] wrapper if present.
  let inner = match edge {
    Expr::FunctionCall { name, args }
      if name == "Labeled" && args.len() == 2 =>
    {
      &args[0]
    }
    _ => edge,
  };
  match inner {
    Expr::Rule {
      pattern,
      replacement,
    } => Some(((**pattern).clone(), (**replacement).clone(), true)),
    Expr::FunctionCall { name, args } if args.len() == 2 => match name.as_str()
    {
      "DirectedEdge" | "Rule" => Some((args[0].clone(), args[1].clone(), true)),
      "UndirectedEdge" | "TwoWayRule" => {
        Some((args[0].clone(), args[1].clone(), false))
      }
      _ => None,
    },
    _ => None,
  }
}

/// True if `s` is a plain Wolfram symbol name (letters/digits/`$`/context
/// backtick, not starting with a digit). A FunctionCall whose head is *not*
/// identifier-like is really a stringified expression (e.g. a pure function
/// `#1 > 0 &` used as a `?test`) that must be re-parsed before it can apply.
fn is_identifier_like(s: &str) -> bool {
  let mut chars = s.chars();
  match chars.next() {
    Some(c) if c.is_alphabetic() || c == '$' => s
      .chars()
      .all(|c| c.is_alphanumeric() || c == '$' || c == '`'),
    _ => false,
  }
}

fn evaluate_function_call_ast_inner(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::functions::list_helpers_ast;

  // Thread Listable functions over list arguments
  let is_listable = is_builtin_listable(name)
    || crate::FUNC_ATTRS.with(|m| {
      m.borrow()
        .get(name)
        .is_some_and(|attrs| attrs.contains(&"Listable".to_string()))
    });
  if is_listable && let Some(result) = thread_listable(name, args)? {
    return Ok(result);
  }

  // Propagate uncertainty through an elementary unary function applied to an
  // Around value: f[Around[a, δ]] = Around[f[a], |f'[a]|·δ].
  if let Some(result) = crate::functions::try_around_unary(name, args) {
    return result;
  }

  // A machine Real times a NESTED exact-times-constant product folds in
  // wolframscript's pre-flattening order (((outer * reals) * nested exact)
  // * constants), one ulp away from the flattened fold — decide before the
  // Flat attribute erases the nesting (differential fuzzer, seed
  // 4134943276941009607).
  if name == "Times"
    && let Some(result) =
      crate::functions::math_ast::nested_exact_const_machine_times(args)
  {
    return Ok(result);
  }

  // Apply Flat attribute: flatten nested calls of the same function
  let has_flat = is_builtin_flat(name)
    || crate::FUNC_ATTRS.with(|m| {
      m.borrow()
        .get(name)
        .is_some_and(|attrs| attrs.contains(&"Flat".to_string()))
    });
  let args_after_flat;
  let args = if has_flat {
    let mut flat_args: Vec<Expr> = Vec::new();
    for arg in args {
      match arg {
        Expr::FunctionCall {
          name: inner_name,
          args: inner_args,
        } if inner_name == name => {
          flat_args.extend(inner_args.clone());
        }
        // `Unevaluated[f[a, b]]` inside a Flat `f[…]` flattens to
        // `Unevaluated[a], Unevaluated[b]` — each inner arg is rewrapped
        // so the hold context is preserved per spliced element.
        Expr::FunctionCall {
          name: u_name,
          args: u_args,
        } if u_name == "Unevaluated"
          && u_args.len() == 1
          && matches!(&u_args[0],
            Expr::FunctionCall { name: inner, .. } if inner == name) =>
        {
          if let Expr::FunctionCall {
            args: inner_args, ..
          } = &u_args[0]
          {
            for inner in inner_args {
              flat_args.push(Expr::FunctionCall {
                name: "Unevaluated".to_string(),
                args: vec![inner.clone()].into(),
              });
            }
          }
        }
        _ => flat_args.push(arg.clone()),
      }
    }
    args_after_flat = flat_args;
    &args_after_flat[..]
  } else {
    args
  };

  // Apply Orderless attribute: sort arguments into canonical order.
  // Plus and Times are exempt: Wolfram's machine-precision fold is
  // input-order-sensitive (Plus[0.1, 0.7, 0.3] → 1.0999999999999999 but
  // Plus[0.1, 0.3, 0.7] → 1.1; Times[3, 1/7, 0.1, 0.2, 0.3] →
  // 0.002571428571428571 but Times[0.1, 0.2, 3, 1/7, 0.3] →
  // 0.0025714285714285717), so plus_ast/times_ast must see the original
  // argument order; both canonically sort their own symbolic output.
  let has_orderless = name != "Plus"
    && name != "Times"
    && (is_builtin_orderless(name)
      || crate::FUNC_ATTRS.with(|m| {
        m.borrow()
          .get(name)
          .is_some_and(|attrs| attrs.contains(&"Orderless".to_string()))
      }));
  let args_after_sort;
  let args = if has_orderless {
    let mut sorted_args = args.to_vec();
    sorted_args.sort_by(|a, b| {
      let ord = list_helpers_ast::compare_exprs(a, b);
      if ord > 0 {
        std::cmp::Ordering::Less
      } else if ord < 0 {
        std::cmp::Ordering::Greater
      } else {
        std::cmp::Ordering::Equal
      }
    });
    args_after_sort = sorted_args;
    &args_after_sort[..]
  } else {
    args
  };

  // Handle structural conversions early
  match name {
    "Rule" if args.len() == 2 => {
      return Ok(Expr::Rule {
        pattern: Box::new(args[0].clone()),
        replacement: Box::new(args[1].clone()),
      });
    }
    "PatternTest" if args.len() == 2 => return evaluate_pattern_test_ast(args),
    "Blank" => return evaluate_blank_ast(args),
    "BlankSequence" | "BlankNullSequence" if args.len() <= 1 => {
      return evaluate_blank_sequence_ast(name, args);
    }
    "Slot" if args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        return Ok(Expr::Slot(*n as usize));
      }
    }
    "SlotSequence" if args.len() == 1 => {
      if let Expr::Integer(n) = &args[0] {
        return Ok(Expr::SlotSequence(*n as usize));
      }
    }
    _ => {}
  }

  // MakeBoxes[head[args], form] consults user `Format[head[…], form]`
  // rules BEFORE user `MakeBoxes[head[…], fmt_]` patterns — Format
  // definitions take precedence over MakeBoxes downvalues for the same
  // head, matching wolframscript. When a Format rule applies, the
  // formatted result is fed back into MakeBoxes for recursive boxing.
  if name == "MakeBoxes"
    && (args.len() == 1 || args.len() == 2)
    && let Expr::FunctionCall { name: head, .. } = &args[0]
  {
    let target_form = if args.len() == 2 {
      if let Expr::Identifier(form) = &args[1] {
        Some(form.clone())
      } else {
        None
      }
    } else {
      None
    };
    let has_format_rule = crate::evaluator::assignment::FORMAT_VALUES
      .with(|m| m.borrow().contains_key(head));
    if has_format_rule {
      let mut format_args = vec![args[0].clone()];
      if let Some(form) = &target_form {
        format_args.push(Expr::Identifier(form.clone()));
      }
      let format_call = Expr::FunctionCall {
        name: "Format".to_string(),
        args: format_args.into(),
      };
      if let Ok(formatted) = evaluate_expr_to_expr(&format_call) {
        // The Format dispatcher returns `args[0].clone()` (or
        // `Format[args[0], …]` unevaluated) when no user rule matches.
        // Detect both shapes so we only recurse when Format actually
        // substituted something different.
        let is_format_wrapper = matches!(
          &formatted,
          Expr::FunctionCall { name: fname, args: fargs }
            if fname == "Format"
              && (fargs.len() == 1 || fargs.len() == 2)
              && crate::evaluator::pattern_matching::expr_equal(&fargs[0], &args[0])
        );
        let unchanged = is_format_wrapper
          || crate::evaluator::pattern_matching::expr_equal(
            &formatted, &args[0],
          );
        if !unchanged {
          // Box the formatted result through the standard MakeBoxes path.
          let mut new_args = vec![formatted];
          if let Some(form) = target_form {
            new_args.push(Expr::Identifier(form));
          }
          return evaluate_function_call_ast_inner("MakeBoxes", &new_args);
        }
      }
    }
  }

  // Fast path: memoized literal-argument value (`f[x_] := f[x] = …`). These
  // are stored in MEMO_VALUES keyed by the joined argument string instead of
  // as FUNC_DEFS overloads, so lookup is O(1) rather than a linear scan over
  // every accumulated value. Checked before pattern overloads so memoized
  // base cases take priority, matching the literal-before-pattern ordering.
  if let Some(v) = crate::MEMO_VALUES.with(|m| {
    let map = m.borrow();
    let cache = map.get(name)?;
    if cache.is_empty() {
      return None;
    }
    let key = args
      .iter()
      .map(crate::syntax::expr_to_string)
      .collect::<Vec<_>>()
      .join("\u{1}");
    cache.get(&key).map(|(_, v)| v.clone())
  }) {
    return Ok(v);
  }

  // Check for user-defined functions (before built-in dispatch, so user
  // overrides take precedence — matching Wolfram Language semantics)
  // Clone overloads to avoid holding the borrow across evaluate calls.
  //
  // HoldAllComplete suppresses UpValues lookup (matching Wolfram). Woxi
  // stores upvalues in FUNC_DEFS too (their params start with `_up`), so
  // when the head carries HoldAllComplete we drop those entries before
  // dispatch — DownValues for the same head are still tried as usual.
  let head_has_hold_all_complete = crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(name)
      .is_some_and(|attrs| attrs.contains(&"HoldAllComplete".to_string()))
  }) || get_builtin_attributes(name)
    .contains(&"HoldAllComplete");
  let overloads = crate::FUNC_DEFS.with(|m| {
    let defs = m.borrow();
    let raw = defs.get(name).cloned();
    if head_has_hold_all_complete {
      raw.map(|v| {
        v.into_iter()
          .filter(|(params, _, _, _, _, _)| {
            !params.iter().any(|p| p.starts_with("_up"))
          })
          .collect::<Vec<_>>()
      })
    } else {
      raw
    }
  });
  let inline_opts_overloads = crate::FUNC_OPTS_INLINE.with(|m| {
    let map = m.borrow();
    map.get(name).cloned()
  });

  // For user-defined-function matching only: strip top-level
  // `Unevaluated[expr]` wrappers from args. The wrapper holds its argument
  // from outer evaluation but is transparent to pattern matching and
  // binding. Wolfram: `f[Unevaluated[1+2]]` with rule `f[x_]:=List[x]`
  // binds x to `1+2` (held). If no rule matches, the wrapper is preserved
  // in the fall-through output, so we keep the original `args` and use a
  // separate `stripped_args` only inside the overload-matching block.
  let has_unevaluated = args.iter().any(|a| {
    matches!(a, Expr::FunctionCall { name, args: ua }
      if name == "Unevaluated" && ua.len() == 1)
  });
  let stripped_args_owned: Vec<Expr> = if has_unevaluated {
    args
      .iter()
      .map(|a| match a {
        Expr::FunctionCall { name, args: ua }
          if name == "Unevaluated" && ua.len() == 1 =>
        {
          ua[0].clone()
        }
        _ => a.clone(),
      })
      .collect()
  } else {
    Vec::new()
  };
  let match_args: &[Expr] = if has_unevaluated {
    &stripped_args_owned[..]
  } else {
    args
  };

  if let Some(overloads) = overloads {
    // Use stripped args for user-defined matching (Unevaluated wrapper is
    // transparent to pattern matching). Original `args` stays untouched
    // for the fall-through path so unmatched calls keep the wrapper.
    let args = match_args;
    for (
      overload_idx,
      (params, conditions, param_defaults, param_heads, blank_types, body_expr),
    ) in overloads.iter().enumerate()
    {
      // Check if any parameter is a sequence pattern (BlankSequence/BlankNullSequence)
      let has_sequence_param = blank_types.iter().any(|&bt| bt >= 2);

      if has_sequence_param {
        // Variable-length argument matching for BlankSequence/BlankNullSequence
        // Count minimum required arguments: Blank=1, BlankSequence=1, BlankNullSequence=0
        let min_args: usize = blank_types
          .iter()
          .zip(param_defaults.iter())
          .map(|(&bt, d)| {
            if d.is_some() {
              0
            }
            // optional params
            else if bt == 3 {
              0
            }
            // BlankNullSequence: 0 or more
            else {
              1
            } // Blank or BlankSequence: at least 1
          })
          .sum();
        if args.len() < min_args {
          continue;
        }

        // Distribute args to params using backtracking for sequence patterns
        let distribution = distribute_args_to_params(
          args,
          blank_types,
          param_heads,
          param_defaults,
          0,
        );
        let mut effective_args = match distribution {
          Some(dist) => {
            // Validate OptionsPattern slots: every arg landing in an
            // `__opts<i>` parameter must be a `Rule` / `RuleDelayed`, or
            // an arbitrarily nested `List` / `Sequence` of those (e.g.
            // `f[x, {n -> 4}]` and `f[x, {{{n -> 4}}}]` both bind a
            // single option). Without this check `r[arg_., OptionsPattern
            // [r]] := …` would accept `r[a, b]` (binding `__opts` to
            // `b`), but wolframscript leaves that call unevaluated since
            // `b` isn't an option container.
            fn is_option_container(e: &Expr) -> bool {
              match e {
                Expr::Rule { .. } | Expr::RuleDelayed { .. } => true,
                Expr::List(items) => items.iter().all(is_option_container),
                Expr::FunctionCall { name, args } if name == "Sequence" => {
                  args.iter().all(is_option_container)
                }
                _ => false,
              }
            }
            let mut opts_ok = true;
            for (i, param_args) in dist.iter().enumerate() {
              if params[i].starts_with("__opts")
                && !param_args.iter().all(is_option_container)
              {
                opts_ok = false;
                break;
              }
            }
            if !opts_ok {
              continue;
            }
            let mut eff = Vec::with_capacity(params.len());
            for (i, param_args) in dist.iter().enumerate() {
              if blank_types[i] >= 2 {
                if param_args.len() == 1 {
                  eff.push(param_args[0].clone());
                } else {
                  eff.push(Expr::FunctionCall {
                    name: "Sequence".to_string(),
                    args: param_args.clone().into(),
                  });
                }
              } else if param_args.is_empty() {
                // Must be a default param
                if let Some(default) = &param_defaults[i] {
                  eff.push(default.clone());
                } else {
                  break;
                }
              } else {
                eff.push(param_args[0].clone());
              }
            }
            if eff.len() != params.len() {
              continue;
            }
            eff
          }
          None => continue,
        };

        // Check conditions
        let mut conditions_met = true;
        let mut structural_bindings: Vec<(String, Expr)> = Vec::new();
        for cond_expr in conditions.iter().flatten() {
          if let Expr::FunctionCall {
            name: marker_name,
            args: marker_args,
          } = cond_expr
            && marker_name == "__StructuralPattern__"
            && marker_args.len() == 2
            && let Expr::Identifier(param_name) = &marker_args[0]
          {
            let pattern = &marker_args[1];
            if let Some(idx) = params.iter().position(|p| p == param_name) {
              if idx < effective_args.len() {
                // Canonicalize the expression to match the canonical pattern form
                let canonical_arg =
                  crate::evaluator::assignment::canonicalize_divide_in_expr(
                    &effective_args[idx],
                  );
                // Push positional parameter bindings as context so inner
                // Orderless matching can check compatibility (e.g. x_Symbol=r).
                let mut positional_ctx: Vec<(String, Expr)> = Vec::new();
                for (pi, param) in params.iter().enumerate() {
                  if pi == idx
                    || pi >= effective_args.len()
                    || param.starts_with("__sp")
                    || param.starts_with("_dv")
                  {
                    continue;
                  }
                  positional_ctx
                    .push((param.clone(), effective_args[pi].clone()));
                }
                crate::evaluator::pattern_matching::push_match_context(
                  &positional_ctx,
                );
                let match_result =
                  crate::evaluator::pattern_matching::match_pattern(
                    &canonical_arg,
                    pattern,
                  );
                crate::evaluator::pattern_matching::pop_match_context();
                if let Some(bindings) = match_result {
                  // Check consistency: structural bindings must not conflict
                  // with positional parameter bindings (skip the structural
                  // param itself and synthetic names)
                  let mut check = bindings.clone();
                  let mut consistent = true;
                  for (pi, param) in params.iter().enumerate() {
                    if pi == idx
                      || pi >= effective_args.len()
                      || param.starts_with("__sp")
                      || param.starts_with("_dv")
                    {
                      continue;
                    }
                    if !crate::evaluator::pattern_matching::merge_bindings(
                      &mut check,
                      vec![(param.clone(), effective_args[pi].clone())],
                    ) {
                      consistent = false;
                      break;
                    }
                  }
                  if !consistent {
                    conditions_met = false;
                    break;
                  }
                  structural_bindings.extend(bindings);
                } else {
                  conditions_met = false;
                  break;
                }
              } else {
                conditions_met = false;
                break;
              }
            }
            continue;
          }
          // Build combined bindings for simultaneous substitution
          let mut all_bindings: Vec<(&str, &Expr)> = Vec::new();
          for (param, arg) in params.iter().zip(effective_args.iter()) {
            all_bindings.push((param.as_str(), arg));
          }
          for (bind_name, bind_val) in &structural_bindings {
            all_bindings.push((bind_name.as_str(), bind_val));
          }
          let substituted_cond =
            crate::syntax::substitute_variables(cond_expr, &all_bindings);
          match evaluate_expr_to_expr(&substituted_cond) {
            Ok(Expr::Identifier(ref s)) if s == "True" => {}
            _ => {
              conditions_met = false;
              break;
            }
          }
        }
        if !conditions_met {
          continue;
        }
        // Substitute and evaluate body — use simultaneous substitution
        // to prevent parameter values from leaking into other arguments
        promote_lists_for_substitution(&mut effective_args);
        let substituted = {
          let mut all_bindings: Vec<(&str, &Expr)> = Vec::new();
          for (param, arg) in params.iter().zip(effective_args.iter()) {
            all_bindings.push((param.as_str(), arg));
          }
          for (bind_name, bind_val) in &structural_bindings {
            all_bindings.push((bind_name.as_str(), bind_val));
          }
          crate::syntax::substitute_variables(body_expr, &all_bindings)
        };
        // Push option context if this overload uses OptionsPattern
        let inline_opts = inline_opts_overloads
          .as_ref()
          .and_then(|v| v.get(overload_idx))
          .and_then(|o| o.as_ref());
        let opt_bindings =
          collect_option_bindings(name, params, &effective_args, inline_opts);
        if let Some(ref bindings) = opt_bindings {
          crate::OPTION_VALUE_CONTEXT.with(|ctx| {
            ctx.borrow_mut().push((name.to_string(), bindings.clone()));
          });
        }
        let mut body = substituted;
        let result = loop {
          match evaluate_expr_to_expr_inner(&body) {
            Err(InterpreterError::TailCall(next)) => body = *next,
            Err(InterpreterError::ReturnValue(val)) => break Ok(*val),
            result => break result,
          }
        };
        // Block/Module/While/For inside the body wrap an internal
        // Return[val] as a literal `Return[val]` Expr; unwrap that here
        // so user-defined functions still short-circuit to `val`,
        // matching wolframscript's `f[] := Block[{}, Return[42]]; f[]`.
        let result = match result {
          Ok(Expr::FunctionCall {
            name: ref n,
            args: ref a,
          }) if n == "Return" && a.len() == 1 => Ok(a[0].clone()),
          other => other,
        };
        // Pop option context
        if opt_bindings.is_some() {
          crate::OPTION_VALUE_CONTEXT.with(|ctx| {
            ctx.borrow_mut().pop();
          });
        }
        // If the body returned Condition[expr, test], evaluate the test
        // as a guard: True → return expr, otherwise this overload fails.
        match &result {
          Ok(Expr::FunctionCall {
            name: cond_name,
            args: cond_args,
          }) if cond_name == "Condition" && cond_args.len() == 2 => {
            match evaluate_expr_to_expr(&cond_args[1]) {
              Ok(Expr::Identifier(ref s)) if s == "True" => {
                return evaluate_expr_to_expr(&cond_args[0]);
              }
              _ => continue, // condition not met, try next overload
            }
          }
          _ => return result,
        }
      }

      // Count required params (those without defaults)
      let required_count =
        param_defaults.iter().filter(|d| d.is_none()).count();
      let total_count = params.len();

      // Accept if required_count <= args.len() <= total_count
      if args.len() < required_count || args.len() > total_count {
        continue;
      }

      // For Orderless functions, try different argument orderings when conditions
      // involve literal matching (SameQ). Generate permutations for small arg counts.
      let arg_orderings: Vec<Vec<Expr>> =
        if has_orderless && args.len() >= 2 && args.len() <= 6 {
          let mut perms = Vec::new();
          let mut indices: Vec<usize> = (0..args.len()).collect();
          loop {
            perms.push(indices.iter().map(|&i| args[i].clone()).collect());
            // Next permutation (Heap's algorithm inline)
            if !next_permutation(&mut indices) {
              break;
            }
          }
          perms
        } else {
          vec![args.to_vec()]
        };

      let mut overload_matched = false;
      for perm_args in &arg_orderings {
        // Build the effective argument list by matching provided args to params.
        // Optional params are filled left-to-right; when there are fewer args than params,
        // optional params use their defaults starting from the leftmost optional param.
        let mut effective_args = if perm_args.len() == total_count {
          // All params provided - check head constraints
          let mut head_ok = true;
          for (i, arg) in perm_args.iter().enumerate() {
            if let Some(head) = &param_heads[i]
              && get_expr_head(arg) != *head
            {
              head_ok = false;
              break;
            }
          }
          if !head_ok {
            continue;
          }
          perm_args.to_vec()
        } else {
          // Fewer args than params - fill optional params with defaults
          let num_optional_to_default = total_count - perm_args.len();
          let mut effective = Vec::with_capacity(total_count);
          let mut arg_idx = 0;
          let mut defaults_used = 0;

          for i in 0..total_count {
            if param_defaults[i].is_some()
              && defaults_used < num_optional_to_default
            {
              let remaining_args = perm_args.len() - arg_idx;
              let remaining_required: usize = param_defaults[i + 1..]
                .iter()
                .filter(|d| d.is_none())
                .count();
              let should_default = if remaining_args <= remaining_required {
                true
              } else if let Some(head) = &param_heads[i] {
                arg_idx < perm_args.len()
                  && get_expr_head(&perm_args[arg_idx]) != *head
              } else {
                false
              };

              if should_default {
                effective.push(param_defaults[i].clone().unwrap());
                defaults_used += 1;
              } else if arg_idx < perm_args.len() {
                if let Some(head) = &param_heads[i]
                  && get_expr_head(&perm_args[arg_idx]) != *head
                {
                  break;
                }
                effective.push(perm_args[arg_idx].clone());
                arg_idx += 1;
              }
            } else if arg_idx < perm_args.len() {
              if let Some(head) = &param_heads[i]
                && get_expr_head(&perm_args[arg_idx]) != *head
              {
                break;
              }
              effective.push(perm_args[arg_idx].clone());
              arg_idx += 1;
            }
          }

          if effective.len() != total_count {
            continue; // this permutation doesn't match
          }
          effective
        };

        // Check all conditions (if any) by substituting params with args and evaluating
        let mut conditions_met = true;
        // Collect bindings from structural pattern matches for body substitution
        let mut structural_bindings: Vec<(String, Expr)> = Vec::new();
        for cond_opt in conditions.iter() {
          if let Some(cond_expr) = cond_opt {
            // Check for __StructuralPattern__ marker — use match_pattern instead of eval
            if let Expr::FunctionCall {
              name: marker_name,
              args: marker_args,
            } = cond_expr
              && marker_name == "__StructuralPattern__"
              && marker_args.len() == 2
              && let Expr::Identifier(param_name) = &marker_args[0]
            {
              let pattern = &marker_args[1];
              // Find the effective arg for this structural param
              if let Some(idx) = params.iter().position(|p| p == param_name) {
                if idx < effective_args.len() {
                  // Canonicalize the expression to match the canonical pattern form
                  // (e.g., BinaryOp::Divide → Times[..., Power[..., -1]])
                  let canonical_arg =
                    crate::evaluator::assignment::canonicalize_divide_in_expr(
                      &effective_args[idx],
                    );
                  // Push positional parameter bindings as context so inner
                  // Orderless matching can check compatibility.
                  let mut positional_ctx: Vec<(String, Expr)> = Vec::new();
                  for (pi, param) in params.iter().enumerate() {
                    if pi == idx
                      || pi >= effective_args.len()
                      || param.starts_with("__sp")
                      || param.starts_with("_dv")
                    {
                      continue;
                    }
                    positional_ctx
                      .push((param.clone(), effective_args[pi].clone()));
                  }
                  crate::evaluator::pattern_matching::push_match_context(
                    &positional_ctx,
                  );
                  let match_result =
                    crate::evaluator::pattern_matching::match_pattern(
                      &canonical_arg,
                      pattern,
                    );
                  crate::evaluator::pattern_matching::pop_match_context();
                  if let Some(bindings) = match_result {
                    // Check consistency: structural bindings must not conflict
                    // with positional parameter bindings (skip the structural
                    // param itself and synthetic names)
                    let mut check = bindings.clone();
                    let mut consistent = true;
                    for (pi, param) in params.iter().enumerate() {
                      if pi == idx
                        || pi >= effective_args.len()
                        || param.starts_with("__sp")
                        || param.starts_with("_dv")
                      {
                        continue;
                      }
                      if !crate::evaluator::pattern_matching::merge_bindings(
                        &mut check,
                        vec![(param.clone(), effective_args[pi].clone())],
                      ) {
                        consistent = false;
                        break;
                      }
                    }
                    if !consistent {
                      conditions_met = false;
                      break;
                    }
                    structural_bindings.extend(bindings);
                  } else {
                    conditions_met = false;
                    break;
                  }
                } else {
                  conditions_met = false;
                  break;
                }
              }
              continue;
            }
            // Build combined bindings for simultaneous substitution
            let mut all_bindings: Vec<(&str, &Expr)> = Vec::new();
            for (param, arg) in params.iter().zip(effective_args.iter()) {
              all_bindings.push((param.as_str(), arg));
            }
            for (bind_name, bind_val) in &structural_bindings {
              all_bindings.push((bind_name.as_str(), bind_val));
            }
            let substituted_cond =
              crate::syntax::substitute_variables(cond_expr, &all_bindings);
            // Evaluate the condition - it must return True. A `?test` whose
            // test is a pure function can be stored as a FunctionCall whose
            // head is the function's *string form* (e.g. `#1 > 0 & [arg]`),
            // which doesn't reduce as a call; if the direct evaluation leaves
            // it unapplied, re-parse the string form (which yields a proper
            // application) and evaluate that — same approach MatchQ uses.
            let mut cond_ok = matches!(
              evaluate_expr_to_expr(&substituted_cond),
              Ok(Expr::Identifier(ref s)) if s == "True"
            );
            if !cond_ok
              && matches!(&substituted_cond, Expr::FunctionCall { name, .. }
                if !is_identifier_like(name))
            {
              cond_ok = matches!(
                crate::interpret(&crate::syntax::expr_to_string(
                  &substituted_cond
                )),
                Ok(ref s) if s == "True"
              );
            }
            if !cond_ok {
              conditions_met = false;
              break;
            }
          }
        }
        if !conditions_met {
          continue; // try next permutation (or next overload if no more permutations)
        }
        overload_matched = true;
        // All conditions met - substitute parameters with arguments and evaluate body
        // Use simultaneous substitution to prevent variable name leakage
        promote_lists_for_substitution(&mut effective_args);
        let substituted = {
          let mut all_bindings: Vec<(&str, &Expr)> = Vec::new();
          for (param, arg) in params.iter().zip(effective_args.iter()) {
            all_bindings.push((param.as_str(), arg));
          }
          for (bind_name, bind_val) in &structural_bindings {
            all_bindings.push((bind_name.as_str(), bind_val));
          }
          crate::syntax::substitute_variables(body_expr, &all_bindings)
        };
        // Push option context if this overload uses OptionsPattern
        let inline_opts2 = inline_opts_overloads
          .as_ref()
          .and_then(|v| v.get(overload_idx))
          .and_then(|o| o.as_ref());
        let opt_bindings =
          collect_option_bindings(name, params, &effective_args, inline_opts2);
        if let Some(ref bindings) = opt_bindings {
          crate::OPTION_VALUE_CONTEXT.with(|ctx| {
            ctx.borrow_mut().push((name.to_string(), bindings.clone()));
          });
        }
        // Tail-call: return body for the trampoline to evaluate.
        // Catch Return[] at the function call boundary via local trampoline.
        let mut body = substituted;
        let result = loop {
          match evaluate_expr_to_expr_inner(&body) {
            Err(InterpreterError::TailCall(next)) => body = *next,
            Err(InterpreterError::ReturnValue(val)) => break Ok(*val),
            result => break result,
          }
        };
        // Unwrap a Block/Module/While/For-produced literal `Return[val]`
        // when it surfaces as the body's result of a user-defined function,
        // matching wolframscript: `f[] := Block[{}, Return[42]]; f[]` ⇒ 42.
        let result = match result {
          Ok(Expr::FunctionCall {
            name: ref n,
            args: ref a,
          }) if n == "Return" && a.len() == 1 => Ok(a[0].clone()),
          other => other,
        };
        // Pop option context
        if opt_bindings.is_some() {
          crate::OPTION_VALUE_CONTEXT.with(|ctx| {
            ctx.borrow_mut().pop();
          });
        }
        // If the body returned Condition[expr, test], evaluate the test
        // as a guard: True → return expr, otherwise this overload fails.
        match &result {
          Ok(Expr::FunctionCall {
            name: cond_name,
            args: cond_args,
          }) if cond_name == "Condition" && cond_args.len() == 2 => {
            match evaluate_expr_to_expr(&cond_args[1]) {
              Ok(Expr::Identifier(ref s)) if s == "True" => {
                return evaluate_expr_to_expr(&cond_args[0]);
              }
              _ => continue, // condition not met, try next permutation
            }
          }
          _ => return result,
        }
      } // end permutation loop
      if !overload_matched {
        continue; // no permutation matched, try next overload
      }
    }
  }

  // Check argument count for known built-in functions
  if let Some(result) = arg_count::check_arg_count(name, args) {
    return result;
  }

  // Several statistics functions operate on an association's values: replace
  // an association first argument with its list of values, like wolframscript.
  // (Mean/Total/Max/Min already handle associations in their own routines.)
  if matches!(
    name,
    "Median"
      | "MinMax"
      | "Variance"
      | "StandardDeviation"
      | "Quantile"
      | "GeometricMean"
      | "HarmonicMean"
      | "RootMeanSquare"
      | "Kurtosis"
      | "Skewness"
      | "Quartiles"
      | "InterquartileRange"
      | "TrimmedMean"
      | "CentralMoment"
  ) && let Some(Expr::Association(pairs)) = args.first()
  {
    let values: Vec<Expr> = pairs.iter().map(|(_, v)| v.clone()).collect();
    let mut new_args = args.to_vec();
    new_args[0] = Expr::List(values.into());
    return evaluate_function_call_ast(name, &new_args);
  }

  // Unary numeric functions thread through a Quantity's magnitude (keeping the
  // unit), matching wolframscript: Abs[Quantity[-5, m]] = Quantity[5, m], while
  // Sign reduces to the bare sign of the magnitude.
  if args.len() == 1
    && matches!(
      name,
      "Abs"
        | "Floor"
        | "Ceiling"
        | "Round"
        | "IntegerPart"
        | "FractionalPart"
        | "Re"
        | "Im"
        | "Conjugate"
        | "Sign"
        | "Positive"
        | "Negative"
        | "NonNegative"
        | "NonPositive"
    )
    && let Some(result) =
      crate::functions::quantity_ast::try_quantity_unary(name, &args[0])
  {
    return result;
  }

  // Dispatch through built-in submodules (after user-defined functions)
  if let Some(result) = structural::dispatch_structural(name, args) {
    return result;
  }
  if let Some(result) = attributes::dispatch_attributes(name, args) {
    return result;
  }
  if let Some(result) =
    evaluation_control::dispatch_evaluation_control(name, args)
  {
    return result;
  }
  if let Some(result) =
    timeseries_functions::dispatch_timeseries_functions(name, args)
  {
    return result;
  }
  if let Some(result) = audio_functions::dispatch_audio_functions(name, args) {
    return result;
  }
  if let Some(result) =
    wavelet_functions::dispatch_wavelet_functions(name, args)
  {
    return result;
  }
  if let Some(result) = list_operations::dispatch_list_operations(name, args) {
    return result;
  }
  if let Some(result) = string_functions::dispatch_string_functions(name, args)
  {
    return result;
  }
  if let Some(result) = image_functions::dispatch_image_functions(name, args) {
    return result;
  }
  if let Some(result) = io_functions::dispatch_io_functions(name, args) {
    return result;
  }
  if let Some(result) =
    datetime_functions::dispatch_datetime_functions(name, args)
  {
    return result;
  }
  if let Some(result) = plotting::dispatch_plotting(name, args) {
    return result;
  }
  if let Some(result) =
    predicate_functions::dispatch_predicate_functions(name, args)
  {
    return result;
  }
  if let Some(result) = music_functions::dispatch_music_functions(name, args) {
    return result;
  }
  if let Some(result) =
    association_functions::dispatch_association_functions(name, args)
  {
    return result;
  }
  if let Some(result) =
    quantity_functions::dispatch_quantity_functions(name, args)
  {
    return result;
  }
  if let Some(result) =
    interval_functions::dispatch_interval_functions(name, args)
  {
    return result;
  }
  if let Some(result) = math_functions::dispatch_math_functions(name, args) {
    return result;
  }
  if let Some(result) =
    boolean_functions::dispatch_boolean_functions(name, args)
  {
    return result;
  }
  if let Some(result) =
    polynomial_functions::dispatch_polynomial_functions(name, args)
  {
    return result;
  }
  if let Some(result) =
    calculus_functions::dispatch_calculus_functions(name, args)
  {
    return result;
  }
  if let Some(result) =
    linear_algebra_functions::dispatch_linear_algebra_functions(name, args)
  {
    return result;
  }
  if let Some(result) =
    complex_and_special::dispatch_complex_and_special(name, args)
  {
    return result;
  }

  // Element data function
  if name == "ElementData" {
    return crate::functions::element_data::element_data_ast(args);
  }

  // Polyhedron data function
  if name == "PolyhedronData" && !args.is_empty() {
    return crate::functions::polyhedron_data::polyhedron_data_ast(args);
  }

  // Chemistry: molecules and their properties
  match name {
    "Molecule" => {
      return crate::functions::molecule_ast::molecule_ast(args);
    }
    "MoleculePlot" => {
      return crate::functions::molecule_render::molecule_plot_ast(args);
    }
    "MoleculeQ" => {
      return crate::functions::molecule_ast::molecule_q_ast(args);
    }
    "AtomList" => {
      return crate::functions::molecule_ast::atom_list_ast(args);
    }
    "BondList" => {
      return crate::functions::molecule_ast::bond_list_ast(args);
    }
    "MoleculeValue" => {
      return crate::functions::molecule_ast::molecule_value_ast(args);
    }
    _ => {}
  }

  // Entity store functions
  match name {
    "EntityStore" => {
      return crate::functions::entity_ast::entity_store_ast(args);
    }
    "EntityRegister" => {
      return crate::functions::entity_ast::entity_register_ast(args);
    }
    "EntityUnregister" => {
      return crate::functions::entity_ast::entity_unregister_ast(args);
    }
    "EntityStores" => {
      return crate::functions::entity_ast::entity_stores_ast(args);
    }
    "EntityValue" => {
      return crate::functions::entity_ast::entity_value_ast(args);
    }
    "EntityList" => return crate::functions::entity_ast::entity_list_ast(args),
    "EntityProperties" => {
      return crate::functions::entity_ast::entity_properties_ast(args);
    }
    "EntityClassList" => {
      return crate::functions::entity_ast::entity_class_list_ast(args);
    }
    "CountryData" => {
      return crate::functions::country_data::country_data_ast(args);
    }
    "ExternalIdentifier" => {
      return crate::functions::wikidata_ast::external_identifier_ast(args);
    }
    "WikidataData" => {
      return crate::functions::wikidata_ast::wikidata_data_ast(args);
    }
    "GeoDistance" => {
      return crate::functions::geo_math::geo_distance_ast(args);
    }
    "GeoDirection" => {
      return crate::functions::geo_math::geo_direction_ast(args);
    }
    "GeoDestination" => {
      return crate::functions::geo_math::geo_destination_ast(args);
    }
    "GeoLength" => {
      return crate::functions::geo_math::geo_length_ast(args);
    }
    "GeoBounds" => {
      return crate::functions::geo_math::geo_bounds_ast(args);
    }
    "GeoAntipode" => {
      return crate::functions::geo_math::geo_antipode_ast(args);
    }
    "GeoNearest" => {
      return crate::functions::geographics::geo_nearest_ast(args);
    }
    // Interpreter["Country"] stays symbolic until applied to an input via the
    // curried form Interpreter["Country"][name] (see apply_curried_call).
    // Returning here avoids the "not yet implemented" warning.
    "Interpreter" => {
      return Ok(unevaluated("Interpreter", args));
    }
    _ => {}
  }

  // Check if the variable stores a value that can be called as a function
  // (e.g., anonymous function stored in a variable: f = (# + 1) &; f[5])
  let stored_value = crate::ENV.with(|e| {
    let env = e.borrow();
    env.get(name).cloned()
  });
  if let Some(stored) = &stored_value {
    let parsed = match stored {
      crate::StoredValue::ExprVal(e) => Some(e.clone()),
      crate::StoredValue::Raw(val_str) => {
        crate::syntax::string_to_expr(val_str).ok()
      }
      _ => None,
    };
    if let Some(Expr::Function { body }) = &parsed {
      let substituted = crate::syntax::substitute_slots(body, args);
      // Tail-call: return body for the trampoline to evaluate.
      // Catch Return[] at the function call boundary via local trampoline.
      let mut body = substituted;
      loop {
        match evaluate_expr_to_expr_inner(&body) {
          Err(InterpreterError::TailCall(next)) => body = *next,
          Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
          result => return result,
        }
      }
    }
  }

  // Begin["ctx`"] / BeginPackage["ctx`"] push context and return the context string.
  // The 2-arg `BeginPackage["ctx`", {"Other`Pkg`", …}]` form takes a list
  // of packages to load alongside; Woxi just ignores the second arg
  // (those packages are no-ops here) and still pushes the requested
  // context, matching wolframscript's `$Context` value after the call.
  // A "relative" context argument that starts with a backtick is
  // resolved against the current context — `Begin["`Private`"]` from
  // inside `MyPackage`` pushes `MyPackage`Private``.
  if (name == "Begin" || name == "BeginPackage")
    && (args.len() == 1 || args.len() == 2)
    && let Expr::String(ctx) = &args[0]
  {
    let resolved = if let Some(rel) = ctx.strip_prefix('`') {
      let parent = crate::current_context();
      let parent_trimmed = parent.trim_end_matches('`');
      format!("{}`{}", parent_trimmed, rel)
    } else {
      ctx.clone()
    };
    crate::push_context(resolved.clone());
    // BeginPackage also re-points $ContextPath to
    // `[pkg`, extras…, "System`"]` so symbols defined inside the
    // package are read from the new context. Begin doesn't touch
    // $ContextPath.
    if name == "BeginPackage" {
      let mut path = vec![resolved.clone()];
      if args.len() == 2
        && let Expr::List(items) = &args[1]
      {
        for item in items {
          if let Expr::String(extra) = item {
            path.push(extra.clone());
            crate::register_package(extra.clone());
          }
        }
      }
      path.push("System`".to_string());
      crate::push_context_path(path);
      // Register the new context so `$Packages` reflects it.
      crate::register_package(resolved.clone());
    }
    return Ok(Expr::String(resolved));
  }
  // End[] pops the context stack and returns the ended context
  if name == "End" && args.is_empty() {
    let ctx = crate::pop_context().unwrap_or_else(|| "Global`".to_string());
    return Ok(Expr::String(ctx));
  }
  // EndPackage[] pops the context stack and returns Null. WL semantics:
  // the package context (and the extras passed to BeginPackage[]) are
  // prepended to the underlying $ContextPath, so the package's exports
  // remain reachable after EndPackage[].
  // Without a prior BeginPackage[], wolframscript emits
  // `EndPackage::noctx` and still returns Null.
  if name == "EndPackage" && args.is_empty() {
    if !crate::has_package_context() {
      crate::emit_message("EndPackage::noctx: No previous context defined.");
      return Ok(Expr::Identifier("Null".to_string()));
    }
    crate::pop_context();
    if let Some(active) = crate::pop_context_path() {
      // The pushed path is `[pkg, extras..., "System`"]`; the prepend
      // list is everything before that trailing baseline entry.
      let prepend: Vec<String> =
        if active.last().map(|s| s == "System`").unwrap_or(false) {
          active[..active.len() - 1].to_vec()
        } else {
          active.clone()
        };
      let base = crate::current_context_path();
      let mut merged: Vec<String> =
        Vec::with_capacity(prepend.len() + base.len());
      for entry in prepend.into_iter().chain(base) {
        if !merged.contains(&entry) {
          merged.push(entry);
        }
      }
      crate::push_context_path(merged);
    }
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Needs["pkg`"] returns $Failed (packages not supported in Woxi)
  if name == "Needs" && args.len() == 1 {
    return Ok(Expr::Identifier("$Failed".to_string()));
  }

  // Package/message/system functions - no-op in Woxi, returns Null
  if name == "Begin"
    || name == "End"
    || name == "BeginPackage"
    || name == "ClearAttributes"
  {
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Off[Head::tag, ...] / On[Head::tag, ...] — toggle message suppression.
  // Each argument is expected to be `MessageName[Head, "tag"]`.
  if name == "Off" || name == "On" {
    for a in args {
      if let Expr::FunctionCall { name: mn, args: ma } = a
        && mn == "MessageName"
        && ma.len() == 2
        && let Expr::Identifier(head) = &ma[0]
      {
        let tag_str = match &ma[1] {
          Expr::String(s) => s.clone(),
          Expr::Identifier(s) => s.clone(),
          _ => continue,
        };
        let key = format!("{}::{}", head, tag_str);
        if name == "Off" {
          crate::off_message(&key);
        } else {
          crate::on_message(&key);
        }
      }
    }
    return Ok(Expr::Identifier("Null".to_string()));
  }
  // Remove[syms...] - fully remove the named symbols (drop env, defs,
  // attrs, options). Unlike Clear, a Removed symbol is gone from Names.
  if name == "Remove" {
    for arg in args {
      if let Expr::Identifier(sym) = arg {
        crate::ENV.with(|e| e.borrow_mut().remove(sym));
        crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
        crate::MEMO_VALUES.with(|m| m.borrow_mut().remove(sym));
        crate::FUNC_ATTRS.with(|m| m.borrow_mut().remove(sym));
        crate::FUNC_OPTIONS.with(|m| m.borrow_mut().remove(sym));
        crate::UPVALUES.with(|m| m.borrow_mut().remove(sym));
      }
    }
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // SetOptions[f] (no rules) returns the current options of `f`, matching
  // wolframscript. The variant with option rules is not implemented yet —
  // fall through to the unevaluated form for that case.
  if name == "SetOptions" {
    if args.len() == 1 {
      return crate::evaluator::evaluate_expr_to_expr(&unevaluated(
        "Options", args,
      ));
    }
    return Ok(unevaluated("SetOptions", args));
  }

  // Circle[] defaults to Circle[{0, 0}]
  if name == "Circle" {
    let center = if args.is_empty() {
      Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into())
    } else {
      args[0].clone()
    };
    let mut new_args = vec![center];
    if args.len() > 1 {
      new_args.extend_from_slice(&args[1..]);
    }
    return Ok(Expr::FunctionCall {
      name: "Circle".to_string(),
      args: new_args.into(),
    });
  }

  // Darker[color] and Darker[color, amount] — darken a color
  if name == "Darker"
    && !args.is_empty()
    && args.len() <= 2
    && let Some(result) = evaluate_darker_lighter(args, true)
  {
    return Ok(result);
  }

  // Lighter[color] and Lighter[color, amount] — lighten a color
  if name == "Lighter"
    && !args.is_empty()
    && args.len() <= 2
    && let Some(result) = evaluate_darker_lighter(args, false)
  {
    return Ok(result);
  }

  // Blend[{c1, c2, ...}] and Blend[{c1, c2, ...}, t]
  if name == "Blend"
    && !args.is_empty()
    && args.len() <= 2
    && let Some(result) = evaluate_blend(args)
  {
    return Ok(result);
  }

  // DiffusionPDETerm[{u, x}, c] → 0 (PDE term evaluates to zero outside solver context)
  if name == "DiffusionPDETerm"
    && args.len() == 2
    && let Expr::List(_) = &args[0]
  {
    return Ok(Expr::Integer(0));
  }

  // NDEigenvalues[DiffusionPDETerm[{u[x], {x}}], u, Element[{x}, Line[{{a}, {b}}]], n]
  // → the first n Neumann eigenvalues of -d²/dx² on [a, b], i.e.
  //   λ_k = (k π / (b - a))² for k = 0, 1, …, n - 1.
  // Wolfram's FEM solver returns the same sequence with small
  // discretisation errors; matching the closed form is at least as
  // accurate.
  if name == "NDEigenvalues"
    && args.len() == 4
    && let Some(eigs) = nd_eigenvalues_diffusion_line(args)
  {
    return Ok(eigs);
  }

  // Disk[] → Disk[{0, 0}] (default center)
  if name == "Disk" && args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Disk".to_string(),
      args: vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into())]
        .into(),
    });
  }

  // Rectangle[] → Rectangle[{0, 0}] (default origin)
  if name == "Rectangle" && args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Rectangle".to_string(),
      args: vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into())]
        .into(),
    });
  }

  // MessageName[sym, tag] — fall back to built-in message templates when
  // the user has not installed their own DownValue. Wolfram returns the
  // template text; without this, e.g. `General::argr` would stay
  // unevaluated as `MessageName[General, argr]`.
  if name == "MessageName"
    && args.len() == 2
    && let Some(text) = builtin_message_text(&args[0], &args[1])
  {
    return Ok(Expr::String(text.to_string()));
  }

  // BezierFunction[{p1, p2, ...}] → wolframscript's structured 7-arg form
  // BezierFunction[degree, knots, {n}, {points, {}}, {0}, MachinePrecision, "Unevaluated"].
  // Already-structured (multi-arg) calls and `BezierFunction[points][t]` (handled
  // at function-application time) fall through unchanged.
  if name == "BezierFunction"
    && args.len() == 1
    && let Expr::List(points) = &args[0]
    && !points.is_empty()
  {
    let to_real = |e: &Expr| -> Expr {
      match e {
        Expr::Integer(n) => Expr::Real(*n as f64),
        Expr::Real(_) => e.clone(),
        _ => e.clone(),
      }
    };
    let points_real: Vec<Expr> = points
      .iter()
      .map(|p| match p {
        Expr::List(coords) => Expr::List(coords.iter().map(to_real).collect()),
        _ => to_real(p),
      })
      .collect();
    return Ok(Expr::FunctionCall {
      name: "BezierFunction".to_string(),
      args: vec![
        Expr::Integer(1),
        Expr::List(
          vec![Expr::List(vec![Expr::Real(0.0), Expr::Real(1.0)].into())]
            .into(),
        ),
        Expr::List(vec![Expr::Integer((points.len() as i128) - 1)].into()),
        Expr::List(
          vec![Expr::List(points_real.into()), Expr::List(vec![].into())]
            .into(),
        ),
        Expr::List(vec![Expr::Integer(0)].into()),
        Expr::Identifier("MachinePrecision".to_string()),
        Expr::String("Unevaluated".to_string()),
      ]
      .into(),
    });
  }

  // BSplineFunction[{p1, p2, ...}] (curve) or BSplineFunction[array]
  // (surface / higher-dimensional manifold) → wolframscript's structured form
  // BSplineFunction[dim, ranges, degrees, closed, {points, Automatic}, knots,
  // {0,...}, MachinePrecision, Unevaluated].
  // Already-structured (9-arg) calls and `BSplineFunction[pts][t]` (handled at
  // function-application time) fall through unchanged.
  if name == "BSplineFunction"
    && args.len() == 1
    && let Expr::List(top) = &args[0]
    && !top.is_empty()
    && let Some(structured) = build_bspline_function(top)
  {
    return Ok(structured);
  }

  // RGBColor["#rrggbb"] parses a CSS-style hex string into channel values.
  // Accepts `#` followed by exactly 3 (shorthand), 6 (RGB), or 8 (RGBA) hex
  // digits, case-insensitively; anything else is left symbolic.
  if name == "RGBColor"
    && args.len() == 1
    && let Expr::String(s) = &args[0]
    && let Some(channels) = parse_hex_color(s)
  {
    return Ok(Expr::FunctionCall {
      name: "RGBColor".to_string(),
      args: channels
        .into_iter()
        .map(Expr::Real)
        .collect::<Vec<_>>()
        .into(),
    });
  }

  // Graphics primitives and style directives: return as symbolic (unevaluated)
  match name {
    "RGBColor"
    | "Hue"
    | "GrayLevel"
    | "CMYKColor"
    | "Opacity"
    | "Thickness"
    | "PointSize"
    | "Dashing"
    | "EdgeForm"
    | "FaceForm"
    | "Haloing"
    | "Directive"
    | "Point"
    | "Line"
    | "Disk"
    | "Rectangle"
    | "Polygon"
    | "RegularPolygon"
    | "Arrow"
    | "BezierCurve"
    | "BezierFunction"
    | "BSplineCurve"
    | "PolarCurve"
    | "FilledPolarCurve"
    | "GraphicsComplex"
    | "Rotate"
    | "Translate"
    | "Scale"
    | "Arrowheads"
    | "AbsoluteThickness"
    | "Inset"
    | "Text"
    | "Style"
    | "Subscript"
    | "BaseForm"
    | "Out"
    | "Condition"
    | "MessageName"
    | "Plot3D"
    | "Integer"
    | "Optional"
    | "String"
    | "Scaled"
    | "NonCommutativeMultiply"
    | "Superscript"
    | "Repeated"
    | "RepeatedNull"
    | "NumberForm"
    | "PercentForm"
    | "DigitBlock"
    | "Cubics"
    | "PageWidth"
    | "Constant"
    | "Catalan"
    | "Placed"
    | "Alternatives"
    | "Offset"
    | "RowBox"
    | "Bond"
    | "Atom"
    | "Colon"
    | "Cap"
    | "Cup"
    | "Congruent"
    | "DirectedEdge"
    | "RightTee"
    | "DoubleRightTee"
    | "LeftTee"
    | "DoubleLeftTee"
    | "UndirectedEdge"
    | "Entity"
    | "InfiniteLine"
    | "Ball"
    | "PlusMinus"
    | "MinusPlus"
    | "CircleTimes"
    | "CircleDot"
    | "Wedge"
    | "Del"
    | "Dispatch"
    | "Cycles"
    | "Exists"
    | "ForAll"
    | "ProbabilityDistribution"
    | "PatternSequence"
    | "StartOfString"
    | "EndOfString"
    | "Whitespace"
    | "SphericalBesselJ"
    | "HoldAllComplete"
    | "CirclePlus"
    | "CircleMinus"
    | "Glow"
    | "PrincipalValue"
    | "LetterCharacter"
    | "Longest"
    | "Shortest"
    | "GenerateConditions"
    | "OverTilde"
    | "AngleBracket"
    | "Larger"
    | "ZetaZero"
    | "MixtureDistribution"
    | "PermutationGroup"
    | "Threaded"
    | "WeightedData"
    | "LightDarkSwitched"
    | "ThemeColor"
    | "SystemColor"
    | "LegendLabel" => {
      return Ok(unevaluated(name, args));
    }
    // UpTo stays symbolic but validates its argument: anything numeric
    // that is not a non-negative integer (or Infinity) emits ::innf.
    "UpTo" => {
      if args.len() == 1 {
        let invalid = match &args[0] {
          Expr::Integer(v) => *v < 0,
          Expr::BigInteger(v) => v.sign() == num_bigint::Sign::Minus,
          Expr::Real(_) | Expr::BigFloat(..) => true,
          Expr::FunctionCall { name: dn, args: da }
            if dn == "DirectedInfinity" =>
          {
            !(da.len() == 1 && matches!(&da[0], Expr::Integer(1)))
          }
          _ => false,
        };
        if invalid {
          crate::emit_message(&format!(
            "UpTo::innf: Non-negative integer or Infinity expected at position 1 in {}.",
            crate::syntax::format_expr(
              &unevaluated("UpTo", args),
              crate::syntax::ExprForm::Output
            )
          ));
        }
      }
      return Ok(unevaluated(name, args));
    }
    _ => {}
  }

  // Permute[list, perm] — permute list elements
  if name == "Permute" && args.len() == 2 {
    // Permute operates on any non-atomic expression, not just Lists; the
    // result keeps the original head (e.g. Permute[f[a,b,c], perm] -> f[...]).
    let head_and_items: Option<(Option<&String>, &[Expr])> = match &args[0] {
      Expr::List(list) => Some((None, list)),
      Expr::FunctionCall { name: h, args: a } => Some((Some(h), a)),
      _ => None,
    };
    if let Some((head, list)) = head_and_items {
      let rebuild = |v: Vec<Expr>| -> Expr {
        match head {
          None => Expr::List(v.into()),
          Some(h) => Expr::FunctionCall {
            name: h.clone(),
            args: v.into(),
          },
        }
      };
      // Permute[expr, {p1, p2, ...}] — permutation list form
      // Element at position i goes to position perm[i]
      if let Expr::List(perm) = &args[1] {
        if perm.len() != list.len() {
          return Ok(unevaluated(name, args));
        }
        let mut result = vec![Expr::Integer(0); list.len()];
        for (i, p) in perm.iter().enumerate() {
          if let Some(idx) = expr_to_i128(p) {
            if idx >= 1 && (idx as usize) <= list.len() {
              result[(idx - 1) as usize] = list[i].clone();
            } else {
              return Ok(unevaluated(name, args));
            }
          } else {
            return Ok(unevaluated(name, args));
          }
        }
        return Ok(rebuild(result));
      }
      // Permute[expr, Cycles[{...}]] — cycle notation
      if let Expr::FunctionCall {
        name: cname,
        args: cargs,
      } = &args[1]
        && cname == "Cycles"
        && cargs.len() == 1
        && let Expr::List(cycle_list) = &cargs[0]
      {
        let mut result = list.to_vec();
        for cycle in cycle_list {
          if let Expr::List(c) = cycle {
            let indices: Vec<usize> = c
              .iter()
              .filter_map(|e| {
                if let Expr::Integer(n) = e {
                  Some(*n as usize)
                } else {
                  None
                }
              })
              .collect();
            if indices.len() >= 2 {
              // Cycle (a,b,c): position a gets value from c, b from a, c from b
              let last_val = list[indices[indices.len() - 1] - 1].clone();
              for i in (1..indices.len()).rev() {
                result[indices[i] - 1] = list[indices[i - 1] - 1].clone();
              }
              result[indices[0] - 1] = last_val;
            }
          }
        }
        return Ok(rebuild(result));
      }
    }
    return Ok(unevaluated(name, args));
  }

  // PermutationList[Cycles[{cycle1, cycle2, ...}]] — convert cycle notation to list form
  if name == "PermutationList" && (args.len() == 1 || args.len() == 2) {
    if let Expr::FunctionCall {
      name: cname,
      args: cargs,
    } = &args[0]
      && cname == "Cycles"
      && cargs.len() == 1
      && let Expr::List(cycle_list) = &cargs[0]
    {
      // Find the maximum element to determine list length
      let mut max_elem: usize = 0;
      for cycle in cycle_list {
        if let Expr::List(c) = cycle {
          for elem in c {
            if let Expr::Integer(n) = elem
              && *n as usize > max_elem
            {
              max_elem = *n as usize;
            }
          }
        }
      }
      // Allow explicit length via second argument
      if args.len() == 2
        && let Expr::Integer(n) = &args[1]
      {
        max_elem = *n as usize;
      }
      // Build identity permutation
      let mut perm: Vec<usize> = (0..=max_elem).collect();
      // Apply each cycle
      for cycle in cycle_list {
        if let Expr::List(c) = cycle {
          let indices: Vec<usize> = c
            .iter()
            .filter_map(|e| {
              if let Expr::Integer(n) = e {
                Some(*n as usize)
              } else {
                None
              }
            })
            .collect();
          if indices.len() >= 2 {
            // Cycle (a, b, c) means a->b, b->c, c->a
            let first = indices[0];
            for i in 0..indices.len() - 1 {
              perm[indices[i]] = indices[i + 1];
            }
            perm[indices[indices.len() - 1]] = first;
          }
        }
      }
      let result: Vec<Expr> = (1..=max_elem)
        .map(|i| Expr::Integer(perm[i] as i128))
        .collect();
      return Ok(Expr::List(result.into()));
    }
    // Permutation-list input: validate it is a permutation of {1..len}, then
    // return it (optionally re-lengthed by the second argument, padding with
    // fixed points or trimming trailing ones).
    if let Expr::List(plist) = &args[0] {
      let mut vals: Vec<i128> = Vec::with_capacity(plist.len());
      let all_int = plist.iter().all(|e| {
        if let Expr::Integer(v) = e {
          vals.push(*v);
          true
        } else {
          false
        }
      });
      if all_int {
        let len = vals.len() as i128;
        let mut sorted = vals.clone();
        sorted.sort_unstable();
        let is_perm =
          sorted.iter().enumerate().all(|(i, &v)| v == i as i128 + 1);
        if !is_perm {
          crate::emit_message(&format!(
            "PermutationList::permlist: Invalid permutation list {}.",
            crate::syntax::expr_to_string(&args[0])
          ));
          return Ok(unevaluated(name, args));
        }
        // Largest moved position (0 for the identity).
        let support_max = (1..=len)
          .rev()
          .find(|&i| vals[(i - 1) as usize] != i)
          .unwrap_or(0);
        let n = match (args.len() == 2).then(|| &args[1]) {
          Some(Expr::Integer(n)) => {
            if *n < support_max {
              crate::emit_message(&format!(
                "PermutationList::lowlen: Required length {} is smaller than maximum {} of support of {}.",
                n,
                support_max,
                crate::syntax::expr_to_string(&args[0])
              ));
              return Ok(unevaluated(name, args));
            }
            *n
          }
          _ => len,
        };
        let result: Vec<Expr> = (1..=n)
          .map(|i| {
            let v = if i <= len { vals[(i - 1) as usize] } else { i };
            Expr::Integer(v)
          })
          .collect();
        return Ok(Expr::List(result.into()));
      }
    }
    return Ok(unevaluated(name, args));
  }

  // PermutationCycles[{p1, p2, ...}] — convert permutation list to cycle notation
  if name == "PermutationCycles" && args.len() == 1 {
    if let Expr::List(perm) = &args[0] {
      let n = perm.len();
      // Extract integer values
      let mut perm_vals: Vec<usize> = Vec::with_capacity(n);
      let mut valid = true;
      for p in perm {
        if let Expr::Integer(val) = p {
          let v = *val as usize;
          if v >= 1 && v <= n {
            perm_vals.push(v);
          } else {
            valid = false;
            break;
          }
        } else {
          valid = false;
          break;
        }
      }
      if valid && perm_vals.len() == n {
        let mut visited = vec![false; n + 1];
        let mut cycles: Vec<Expr> = Vec::new();
        for i in 1..=n {
          if !visited[i] && perm_vals[i - 1] != i {
            // Found start of a non-trivial cycle
            let mut cycle = Vec::new();
            let mut j = i;
            while !visited[j] {
              visited[j] = true;
              cycle.push(Expr::Integer(j as i128));
              j = perm_vals[j - 1];
            }
            if cycle.len() >= 2 {
              cycles.push(Expr::List(cycle.into()));
            }
          } else {
            visited[i] = true;
          }
        }
        return Ok(Expr::FunctionCall {
          name: "Cycles".to_string(),
          args: vec![Expr::List(cycles.into())].into(),
        });
      }
    }
    return Ok(unevaluated(name, args));
  }

  // PermutationReplace[expr, perm] — replace each point by its image under perm.
  // perm may be Cycles[{...}] or a permutation list {p1, p2, ...}.
  if name == "PermutationReplace" && args.len() == 2 {
    // A list of permutations {Cycles[…], …}: apply each to the point/list and
    // return the list of results (PermutationReplace[e, {p1, …}] threads over
    // the permutations). Integer permutation-lists are handled below as a
    // single permutation, so only an all-Cycles list takes this branch.
    if let Expr::List(perms) = &args[1]
      && !perms.is_empty()
      && perms.iter().all(
        |p| matches!(p, Expr::FunctionCall { name: h, .. } if h == "Cycles"),
      )
    {
      let results: Vec<Expr> = perms
        .iter()
        .map(|p| {
          evaluate_expr_to_expr(&Expr::FunctionCall {
            name: "PermutationReplace".to_string(),
            args: vec![args[0].clone(), p.clone()].into(),
          })
          .unwrap_or_else(|_| args[0].clone())
        })
        .collect();
      return Ok(Expr::List(results.into()));
    }
    // Build the image map: point -> image. Points absent from the map are fixed.
    let mut images: std::collections::HashMap<i128, i128> =
      std::collections::HashMap::new();
    let mut valid_perm = false;
    match &args[1] {
      Expr::FunctionCall {
        name: cname,
        args: cargs,
      } if cname == "Cycles"
        && cargs.len() == 1
        && matches!(&cargs[0], Expr::List(_)) =>
      {
        if let Expr::List(cycle_list) = &cargs[0] {
          valid_perm = true;
          for cycle in cycle_list.iter() {
            if let Expr::List(c) = cycle {
              let indices: Vec<i128> = c
                .iter()
                .filter_map(|e| {
                  if let Expr::Integer(n) = e {
                    Some(*n)
                  } else {
                    None
                  }
                })
                .collect();
              if indices.len() != c.len() {
                valid_perm = false;
                break;
              }
              if indices.len() >= 2 {
                for i in 0..indices.len() - 1 {
                  images.insert(indices[i], indices[i + 1]);
                }
                images.insert(indices[indices.len() - 1], indices[0]);
              }
            } else {
              valid_perm = false;
              break;
            }
          }
        }
      }
      Expr::List(perm) => {
        valid_perm = true;
        for (i, p) in perm.iter().enumerate() {
          if let Expr::Integer(val) = p {
            images.insert((i + 1) as i128, *val);
          } else {
            valid_perm = false;
            break;
          }
        }
      }
      _ => {}
    }

    if valid_perm {
      // Map a single expr: integers get their image (default: identity);
      // lists are mapped element-wise (recursively); other exprs unchanged.
      fn map_point(
        e: &Expr,
        images: &std::collections::HashMap<i128, i128>,
      ) -> Expr {
        match e {
          Expr::Integer(n) => Expr::Integer(*images.get(n).unwrap_or(n)),
          Expr::List(items) => {
            Expr::List(items.iter().map(|it| map_point(it, images)).collect())
          }
          other => other.clone(),
        }
      }
      return Ok(map_point(&args[0], &images));
    }
    return Ok(unevaluated(name, args));
  }

  // RelationGraph[f, {v1, ..., vn}, opts...] → Graph whose edge i->j exists
  // whenever f[vi, vj] evaluates to True. The graph is undirected when the
  // relation is symmetric over every pair (f[vi,vj] == f[vj,vi]), otherwise
  // directed. Self-loops are included when f[vi, vi] is True.
  if name == "RelationGraph" && args.len() >= 2 {
    if let Expr::List(verts) = &args[1] {
      let f = &args[0];
      let vertices: Vec<Expr> = verts.to_vec();
      let n = vertices.len();

      // rel[i][j] = whether f[vi, vj] is True.
      let mut rel = vec![vec![false; n]; n];
      for i in 0..n {
        for j in 0..n {
          let res = crate::functions::list_helpers_ast::apply_func_to_two_args(
            f,
            &vertices[i],
            &vertices[j],
          )?;
          rel[i][j] = matches!(&res, Expr::Identifier(s) if s == "True");
        }
      }

      // Symmetric iff rel[i][j] == rel[j][i] for all i < j.
      let mut symmetric = true;
      'sym: for i in 0..n {
        for j in (i + 1)..n {
          if rel[i][j] != rel[j][i] {
            symmetric = false;
            break 'sym;
          }
        }
      }

      let mut edges = Vec::new();
      if symmetric {
        // One undirected edge per unordered pair (i <= j) where the relation holds.
        for i in 0..n {
          for j in i..n {
            if rel[i][j] {
              edges.push(Expr::FunctionCall {
                name: "UndirectedEdge".to_string(),
                args: vec![vertices[i].clone(), vertices[j].clone()].into(),
              });
            }
          }
        }
      } else {
        // One directed edge per ordered pair (i, j) where the relation holds.
        for i in 0..n {
          for j in 0..n {
            if rel[i][j] {
              edges.push(Expr::FunctionCall {
                name: "DirectedEdge".to_string(),
                args: vec![vertices[i].clone(), vertices[j].clone()].into(),
              });
            }
          }
        }
      }

      let opts: Vec<Expr> = args[2..]
        .iter()
        .filter(|a| matches!(a, Expr::Rule { .. }))
        .cloned()
        .collect();
      let mut graph_args =
        vec![Expr::List(vertices.into()), Expr::List(edges.into())];
      if !opts.is_empty() {
        graph_args.push(Expr::List(opts.into()));
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: graph_args.into(),
      });
    }
    return Ok(unevaluated(name, args));
  }

  // CompleteGraph[n, opts...] → Graph[{1,...,n}, {UndirectedEdge[i,j] for all i<j}, {opts}]
  if name == "CompleteGraph" && !args.is_empty() {
    if let Expr::Integer(n) = &args[0] {
      let n = *n as usize;
      let vertices: Vec<Expr> =
        (1..=n).map(|i| Expr::Integer(i as i128)).collect();
      let mut edges = Vec::new();
      for i in 1..=n {
        for j in (i + 1)..=n {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![Expr::Integer(i as i128), Expr::Integer(j as i128)]
              .into(),
          });
        }
      }
      // Collect any trailing Rule options into a {opts} list as Graph's
      // third argument, matching wolframscript's representation.
      let opts: Vec<Expr> = args[1..]
        .iter()
        .filter(|a| matches!(a, Expr::Rule { .. }))
        .cloned()
        .collect();
      let mut graph_args =
        vec![Expr::List(vertices.into()), Expr::List(edges.into())];
      if !opts.is_empty() {
        graph_args.push(Expr::List(opts.into()));
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: graph_args.into(),
      });
    }
    // CompleteGraph[{n1, ..., nk}] → complete k-partite graph: vertices
    // partitioned into groups of the given sizes, with an edge between every
    // pair of vertices in *different* groups. A single-element list {n} is the
    // ordinary complete graph K_n (matching wolframscript).
    if let Expr::List(parts) = &args[0] {
      let sizes: Option<Vec<usize>> = parts
        .iter()
        .map(|p| match p {
          Expr::Integer(n) if *n >= 0 => Some(*n as usize),
          _ => None,
        })
        .collect();
      if let Some(sizes) = sizes
        && !sizes.is_empty()
      {
        let total: usize = sizes.iter().sum();
        let vertices: Vec<Expr> =
          (1..=total).map(|i| Expr::Integer(i as i128)).collect();
        // part_of[v] = index of the group containing 1-indexed vertex v.
        let single_part = sizes.len() == 1;
        let mut part_of = vec![0usize; total + 1];
        let mut v = 1;
        for (pi, &sz) in sizes.iter().enumerate() {
          for _ in 0..sz {
            part_of[v] = pi;
            v += 1;
          }
        }
        let mut edges = Vec::new();
        for i in 1..=total {
          for j in (i + 1)..=total {
            // K_n for a single part; otherwise only cross-part edges.
            if single_part || part_of[i] != part_of[j] {
              edges.push(Expr::FunctionCall {
                name: "UndirectedEdge".to_string(),
                args: vec![Expr::Integer(i as i128), Expr::Integer(j as i128)]
                  .into(),
              });
            }
          }
        }
        let opts: Vec<Expr> = args[1..]
          .iter()
          .filter(|a| matches!(a, Expr::Rule { .. }))
          .cloned()
          .collect();
        let mut graph_args =
          vec![Expr::List(vertices.into()), Expr::List(edges.into())];
        if !opts.is_empty() {
          graph_args.push(Expr::List(opts.into()));
        }
        return Ok(Expr::FunctionCall {
          name: "Graph".to_string(),
          args: graph_args.into(),
        });
      }
    }
    return Ok(unevaluated(name, args));
  }

  // CycleGraph[n, opts...] → cycle graph: vertices 1..n with edges 1-2, 2-3, ..., n-1.
  if name == "CycleGraph"
    && !args.is_empty()
    && let Expr::Integer(n) = &args[0]
  {
    let n = *n as usize;
    if n >= 1 {
      let vertices: Vec<Expr> =
        (1..=n).map(|i| Expr::Integer(i as i128)).collect();
      // wolframscript lists the cycle edges in sorted order:
      // {1-2, 1-n, 2-3, 3-4, ...}. The construction is literal, so
      // CycleGraph[1] has a self-loop and CycleGraph[2] a doubled edge.
      let mut pairs: Vec<(usize, usize)> = Vec::new();
      for i in 1..n {
        pairs.push((i, i + 1));
      }
      pairs.push((1, n));
      pairs.sort_unstable();
      let edges: Vec<Expr> = pairs
        .into_iter()
        .map(|(a, b)| Expr::FunctionCall {
          name: "UndirectedEdge".to_string(),
          args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)].into(),
        })
        .collect();
      let opts: Vec<Expr> = args[1..]
        .iter()
        .filter(|a| matches!(a, Expr::Rule { .. }))
        .cloned()
        .collect();
      let mut graph_args =
        vec![Expr::List(vertices.into()), Expr::List(edges.into())];
      if !opts.is_empty() {
        graph_args.push(Expr::List(opts.into()));
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: graph_args.into(),
      });
    }
  }

  // GridGraph[{m, n}] → m×n grid: vertices 1..m*n in row-major order.
  // For each v: if not in last column emit v—(v+1); if not in last row emit v—(v+m).
  if name == "GridGraph"
    && args.len() == 1
    && let Expr::List(dims) = &args[0]
    && dims.len() == 2
    && let (Expr::Integer(mc), Expr::Integer(nc)) = (&dims[0], &dims[1])
  {
    let m = *mc as usize;
    let n = *nc as usize;
    if m >= 1 && n >= 1 {
      let total = m * n;
      let vertices: Vec<Expr> =
        (1..=total).map(|i| Expr::Integer(i as i128)).collect();
      let mut edges = Vec::new();
      for v in 1..=total {
        let col = ((v - 1) % m) + 1;
        let row = ((v - 1) / m) + 1;
        if col < m {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![
              Expr::Integer(v as i128),
              Expr::Integer((v + 1) as i128),
            ]
            .into(),
          });
        }
        if row < n {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![
              Expr::Integer(v as i128),
              Expr::Integer((v + m) as i128),
            ]
            .into(),
          });
        }
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vertices.into()), Expr::List(edges.into())]
          .into(),
      });
    }
  }

  // StarGraph[n] → star graph with 1 center vertex connected to n-1 outer vertices
  if name == "StarGraph"
    && args.len() == 1
    && let Expr::Integer(n) = &args[0]
  {
    let n = *n as usize;
    if n >= 2 {
      let vertices: Vec<Expr> =
        (1..=n).map(|i| Expr::Integer(i as i128)).collect();
      let edges: Vec<Expr> = (2..=n)
        .map(|i| Expr::FunctionCall {
          name: "UndirectedEdge".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(i as i128)].into(),
        })
        .collect();
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vertices.into()), Expr::List(edges.into())]
          .into(),
      });
    } else if n == 1 {
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![
          Expr::List(vec![Expr::Integer(1)].into()),
          Expr::List(vec![].into()),
        ]
        .into(),
      });
    }
  }

  // WheelGraph[n] → wheel graph: hub vertex 1 connected to the rim
  // vertices 2..n, plus a cycle on the rim vertices 2..n.
  // Matches wolframscript's vertex/edge ordering, including the degenerate
  // small cases (n=2 self-loop, n=3 double edge).
  if name == "WheelGraph"
    && args.len() == 1
    && let Expr::Integer(n) = &args[0]
  {
    let n = *n as usize;
    if n >= 1 {
      let vertices: Vec<Expr> =
        (1..=n).map(|i| Expr::Integer(i as i128)).collect();
      let mk_edge = |a: usize, b: usize| Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)].into(),
      };
      let mut edges = Vec::new();
      // Hub (vertex 1) connected to every rim vertex.
      for i in 2..=n {
        edges.push(mk_edge(1, i));
      }
      // Cycle on the rim vertices 2..n. `m` rim vertices.
      let m = n.saturating_sub(1);
      if m == 1 {
        // Single rim vertex → self-loop (CycleGraph of one vertex).
        edges.push(mk_edge(2, 2));
      } else if m == 2 {
        // Two rim vertices → double edge (CycleGraph of two vertices).
        edges.push(mk_edge(2, 3));
        edges.push(mk_edge(2, 3));
      } else if m >= 3 {
        // Canonical cycle edges, sorted lexicographically by (min, max).
        let mut cyc: Vec<(usize, usize)> = Vec::new();
        for i in 2..n {
          cyc.push((i, i + 1));
        }
        // Closing edge between the first and last rim vertices.
        cyc.push((2, n));
        cyc.sort();
        for (a, b) in cyc {
          edges.push(mk_edge(a, b));
        }
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vertices.into()), Expr::List(edges.into())]
          .into(),
      });
    }
  }

  // HararyGraph[k, n] — the minimal k-connected graph on n vertices.
  // Even k = 2r: circulant graph with jumps 1..r. Odd k = 2r + 1 adds the
  // diameter jump n/2 (n even), or the (1, 1 + (n-1)/2) edge plus
  // (i, i + (n+1)/2) for i = 1..(n-1)/2 (n odd).
  if name == "HararyGraph" && args.len() >= 2 {
    // Arguments beyond position 2 must be options (rules); they only
    // affect rendering, so they are validated and ignored.
    for extra in &args[2..] {
      let is_opt = matches!(
        extra,
        Expr::Rule { .. } | Expr::RuleDelayed { .. } | Expr::List(_)
      );
      if !is_opt {
        crate::emit_message(&format!(
          "HararyGraph::nonopt: Options expected (instead of {}) beyond position 2 in {}. An option must be a rule or a list of rules.",
          crate::syntax::expr_to_string(extra),
          crate::syntax::expr_to_string(&unevaluated("HararyGraph", args))
        ));
        return Ok(unevaluated("HararyGraph", args));
      }
    }
    // Positive machine-sized integers required (intpm); reals, zero, and
    // negative integers warn, symbolic arguments stay silent.
    let call_str =
      || crate::syntax::expr_to_string(&unevaluated("HararyGraph", args));
    let classify = |e: &Expr| -> Option<Option<i128>> {
      // Some(Some(v)): positive integer; Some(None): invalid numeric
      // (emit intpm); None: symbolic (stay unevaluated silently)
      match e {
        Expr::Integer(v) if *v >= 1 => Some(Some(*v)),
        Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_) => Some(None),
        _ => None,
      }
    };
    let unevaluated = unevaluated("HararyGraph", args);
    let (k, n) = match (classify(&args[0]), classify(&args[1])) {
      (Some(None), _) => {
        crate::emit_message(&format!(
          "HararyGraph::intpm: Positive machine-sized integer expected at position 1 in {}.",
          call_str()
        ));
        return Ok(unevaluated);
      }
      (_, Some(None)) => {
        crate::emit_message(&format!(
          "HararyGraph::intpm: Positive machine-sized integer expected at position 2 in {}.",
          call_str()
        ));
        return Ok(unevaluated);
      }
      (Some(Some(k)), Some(Some(n))) => (k, n),
      _ => return Ok(unevaluated),
    };
    if k < 2 {
      crate::emit_message(&format!(
        "HararyGraph::intg: Integer greater than 1 expected at position 1 in {}.",
        call_str()
      ));
      return Ok(unevaluated);
    }
    if n <= k {
      crate::emit_message(&format!(
        "HararyGraph::intg: Integer greater than {} expected at position 2 in {}.",
        k,
        call_str()
      ));
      return Ok(unevaluated);
    }
    let (k, n) = (k as usize, n as usize);
    let mut pairs: std::collections::BTreeSet<(usize, usize)> =
      std::collections::BTreeSet::new();
    let mut add = |a: usize, b: usize| {
      if a != b {
        pairs.insert((a.min(b), a.max(b)));
      }
    };
    let r = k / 2;
    for i in 1..=n {
      for j in 1..=r {
        add(i, (i - 1 + j) % n + 1);
      }
    }
    if k % 2 == 1 {
      if n % 2 == 0 {
        for i in 1..=n {
          add(i, (i - 1 + n / 2) % n + 1);
        }
      } else {
        let half = (n - 1) / 2;
        add(1, 1 + half);
        for i in 1..=half {
          add(i, i + half + 1);
        }
      }
    }
    let vertices: Vec<Expr> =
      (1..=n).map(|i| Expr::Integer(i as i128)).collect();
    let edges: Vec<Expr> = pairs
      .into_iter()
      .map(|(a, b)| Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)].into(),
      })
      .collect();
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // CirculantGraph[n, {j1, j2, ...}] — circulant graph
  // CirculantGraph[n, j] or CirculantGraph[n, {j1, j2, ...}] — each vertex i
  // connects to i±j (mod n) for every jump j. Edges wrap around modularly and
  // are deduplicated by undirected pair, listed in lexicographic order.
  if name == "CirculantGraph"
    && args.len() == 2
    && let Expr::Integer(n) = &args[0]
    && matches!(&args[1], Expr::List(_) | Expr::Integer(_))
  {
    let n = *n as usize;
    let jump_vals: Vec<usize> = match &args[1] {
      Expr::List(jumps) => jumps
        .iter()
        .filter_map(|j| match j {
          Expr::Integer(v) => Some(*v as usize),
          _ => None,
        })
        .collect(),
      Expr::Integer(v) => vec![*v as usize],
      _ => unreachable!(),
    };
    let vertices: Vec<Expr> =
      (1..=n).map(|i| Expr::Integer(i as i128)).collect();
    // BTreeSet keeps edges deduplicated and in lexicographic (min, max) order.
    let mut edge_set = std::collections::BTreeSet::new();
    for i in 1..=n {
      for &j in &jump_vals {
        if n == 0 {
          continue;
        }
        let target = ((i - 1 + j) % n) + 1;
        if i != target {
          let (a, b) = if i < target { (i, target) } else { (target, i) };
          edge_set.insert((a, b));
        }
      }
    }
    let edges: Vec<Expr> = edge_set
      .into_iter()
      .map(|(a, b)| Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)].into(),
      })
      .collect();
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // KaryTree[n] or KaryTree[n, k] — k-ary tree with n vertices (default k=2)
  if name == "KaryTree"
    && (args.len() == 1 || args.len() == 2)
    && let Expr::Integer(n) = &args[0]
  {
    let n = *n as usize;
    let k = if args.len() == 2 {
      if let Expr::Integer(kv) = &args[1] {
        *kv as usize
      } else {
        2
      }
    } else {
      2
    };
    if n >= 1 {
      let vertices: Vec<Expr> =
        (1..=n).map(|i| Expr::Integer(i as i128)).collect();
      let mut edges = Vec::new();
      for i in 1..=n {
        for c in 0..k {
          let child = k * (i - 1) + c + 2;
          if child <= n {
            edges.push(Expr::FunctionCall {
              name: "UndirectedEdge".to_string(),
              args: vec![
                Expr::Integer(i as i128),
                Expr::Integer(child as i128),
              ]
              .into(),
            });
          }
        }
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vertices.into()), Expr::List(edges.into())]
          .into(),
      });
    }
  }

  // CompleteKaryTree[n] or CompleteKaryTree[n, k] — complete k-ary tree
  // with n levels (depth n-1, default branching factor k=2).
  if name == "CompleteKaryTree"
    && (args.len() == 1 || args.len() == 2)
    && let Expr::Integer(n) = &args[0]
  {
    let n = *n;
    let k = if args.len() == 2 {
      if let Expr::Integer(kv) = &args[1] {
        *kv
      } else {
        return Ok(unevaluated(name, args));
      }
    } else {
      2
    };
    // Both n (levels) and k (branching) must be positive integers.
    if n >= 1 && k >= 1 {
      let n = n as usize;
      let k = k as usize;
      // Total vertices for a complete k-ary tree with n levels.
      let num_vertices: usize = if k == 1 {
        n
      } else {
        // (k^n - 1) / (k - 1)
        let mut sum = 0usize;
        let mut pow = 1usize;
        for _ in 0..n {
          sum += pow;
          pow *= k;
        }
        sum
      };
      let vertices: Vec<Expr> = (1..=num_vertices)
        .map(|i| Expr::Integer(i as i128))
        .collect();
      let mut edges = Vec::new();
      for i in 1..=num_vertices {
        for c in 0..k {
          let child = k * (i - 1) + c + 2;
          if child <= num_vertices {
            edges.push(Expr::FunctionCall {
              name: "UndirectedEdge".to_string(),
              args: vec![
                Expr::Integer(i as i128),
                Expr::Integer(child as i128),
              ]
              .into(),
            });
          }
        }
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vertices.into()), Expr::List(edges.into())]
          .into(),
      });
    }
  }

  // HypercubeGraph[n] — n-dimensional hypercube graph
  if name == "HypercubeGraph"
    && args.len() == 1
    && let Expr::Integer(n) = &args[0]
  {
    let n = *n as usize;
    let num_vertices = 1usize << n; // 2^n
    let vertices: Vec<Expr> = (1..=num_vertices)
      .map(|i| Expr::Integer(i as i128))
      .collect();
    let mut edges = Vec::new();
    for i in 0..num_vertices {
      for bit in 0..n {
        let j = i ^ (1 << bit);
        if i < j {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![
              Expr::Integer((i + 1) as i128),
              Expr::Integer((j + 1) as i128),
            ]
            .into(),
          });
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // TuranGraph[n, k] — complete k-partite graph with n vertices, partitions as equal as possible
  if name == "TuranGraph"
    && args.len() == 2
    && let Expr::Integer(n) = &args[0]
    && let Expr::Integer(k) = &args[1]
  {
    let n = *n as usize;
    let k = *k as usize;
    let vertices: Vec<Expr> =
      (1..=n).map(|i| Expr::Integer(i as i128)).collect();

    // Assign each vertex to a partition
    // Partition sizes: n%k partitions of ceil(n/k), rest of floor(n/k)
    let mut partition = vec![0usize; n];
    let base = n / k;
    let extra = n % k;
    let mut idx = 0;
    for p in 0..k {
      let size = base + if p < extra { 1 } else { 0 };
      for _ in 0..size {
        if idx < n {
          partition[idx] = p;
          idx += 1;
        }
      }
    }

    // Add edges between vertices in different partitions
    let mut edges = Vec::new();
    for i in 0..n {
      for j in (i + 1)..n {
        if partition[i] != partition[j] {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![
              Expr::Integer((i + 1) as i128),
              Expr::Integer((j + 1) as i128),
            ]
            .into(),
          });
        }
      }
    }

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // DeBruijnGraph[m, n] — n-dimensional De Bruijn graph with m symbols
  if name == "DeBruijnGraph"
    && args.len() == 2
    && let Expr::Integer(m) = &args[0]
    && let Expr::Integer(n) = &args[1]
  {
    let m = *m as usize;
    let n = *n as usize;
    let num_vertices = m.pow(n as u32);
    let vertices: Vec<Expr> = (1..=num_vertices)
      .map(|i| Expr::Integer(i as i128))
      .collect();

    // For each vertex v (0-indexed), successors are: (v % m^(n-1)) * m + c for c in 0..m
    let shift = m.pow((n as u32).saturating_sub(1));
    let mut edges = Vec::new();
    for v in 0..num_vertices {
      let base = (v % shift) * m;
      for c in 0..m {
        let w = base + c;
        edges.push(Expr::FunctionCall {
          name: "DirectedEdge".to_string(),
          args: vec![
            Expr::Integer((v + 1) as i128),
            Expr::Integer((w + 1) as i128),
          ]
          .into(),
        });
      }
    }

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // RecurrenceFilter[{{a0, a1, ...}, {b0, b1, ...}}, data] → IIR filter
  if name == "RecurrenceFilter"
    && args.len() == 2
    && let Expr::List(coeffs) = &args[0]
    && coeffs.len() == 2
    && let (Expr::List(a_coeffs), Expr::List(b_coeffs)) =
      (&coeffs[0], &coeffs[1])
    && !a_coeffs.is_empty()
  {
    // Image input: apply the filter along rows, then along columns,
    // per channel. The result preserves dimensions, channels, and
    // image_type.
    if let Expr::Image {
      color_space: _,
      width,
      height,
      channels,
      data,
      image_type,
    } = &args[1]
    {
      let w = *width as usize;
      let h = *height as usize;
      let ch = *channels as usize;
      let a_floats: Vec<f64> = a_coeffs
        .iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
          InterpreterError::EvaluationError(
            "RecurrenceFilter: a-coefficients must be numeric for an Image input"
              .into(),
          )
        })?;
      let b_floats: Vec<f64> = b_coeffs
        .iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
          InterpreterError::EvaluationError(
            "RecurrenceFilter: b-coefficients must be numeric for an Image input"
              .into(),
          )
        })?;
      let filter_1d = |seq: &[f64]| -> Vec<f64> {
        let n = seq.len();
        let mut out = vec![0.0_f64; n];
        for i in 0..n {
          let mut s = 0.0;
          for (j, bj) in b_floats.iter().enumerate() {
            if i >= j {
              s += bj * seq[i - j];
            }
          }
          for (k, ak) in a_floats.iter().enumerate().skip(1) {
            if i >= k {
              s -= ak * out[i - k];
            }
          }
          out[i] = s / a_floats[0];
        }
        out
      };
      let mut tmp = data.as_ref().clone();
      // Row pass.
      for c_idx in 0..ch {
        let mut row_buf = vec![0.0; w];
        for y in 0..h {
          for x in 0..w {
            row_buf[x] = tmp[(y * w + x) * ch + c_idx];
          }
          let filtered = filter_1d(&row_buf);
          for x in 0..w {
            tmp[(y * w + x) * ch + c_idx] = filtered[x];
          }
        }
      }
      // Column pass.
      for c_idx in 0..ch {
        let mut col_buf = vec![0.0; h];
        for x in 0..w {
          for y in 0..h {
            col_buf[y] = tmp[(y * w + x) * ch + c_idx];
          }
          let filtered = filter_1d(&col_buf);
          for y in 0..h {
            tmp[(y * w + x) * ch + c_idx] = filtered[y];
          }
        }
      }
      return Ok(Expr::Image {
        color_space: None,
        width: *width,
        height: *height,
        channels: *channels,
        data: std::sync::Arc::new(tmp),
        image_type: *image_type,
      });
    }

    // List input: filter along the flat sequence.
    let Expr::List(data) = &args[1] else {
      return Ok(unevaluated("RecurrenceFilter", args));
    };
    // y[n] = (sum_j b[j]*x[n-j] - sum_i(i>0) a[i]*y[n-i]) / a[0]
    let n = data.len();
    let mut output: Vec<Expr> = Vec::with_capacity(n);
    for i in 0..n {
      let mut sum = Expr::Integer(0);
      for (j, bj) in b_coeffs.iter().enumerate() {
        if i >= j {
          let term = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(bj.clone()),
            right: Box::new(data[i - j].clone()),
          };
          sum = Expr::BinaryOp {
            op: BinaryOperator::Plus,
            left: Box::new(sum),
            right: Box::new(term),
          };
        }
      }
      for (k, ak) in a_coeffs.iter().enumerate().skip(1) {
        if i >= k {
          let term = Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(ak.clone()),
            right: Box::new(output[i - k].clone()),
          };
          sum = Expr::BinaryOp {
            op: BinaryOperator::Minus,
            left: Box::new(sum),
            right: Box::new(term),
          };
        }
      }
      let result = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(sum),
        right: Box::new(a_coeffs[0].clone()),
      };
      match evaluate_expr_to_expr(&result) {
        Ok(v) => output.push(v),
        Err(e) => return Err(e),
      }
    }
    return Ok(Expr::List(output.into()));
  }

  // CantorMesh[n]    → 1D Cantor set MeshRegion (Lines)
  // CantorMesh[n, d] → d-dimensional Cantor set MeshRegion
  if name == "CantorMesh"
    && (args.len() == 1 || args.len() == 2)
    && let Some(n) = crate::functions::math_ast::try_eval_to_f64(&args[0])
  {
    let n = n as usize;
    let d = if args.len() == 2 {
      crate::functions::math_ast::try_eval_to_f64(&args[1]).unwrap_or(1.0)
        as usize
    } else {
      1
    };

    // 1D Cantor intervals at step n.
    let mut segments: Vec<(f64, f64)> = vec![(0.0, 1.0)];
    for _ in 0..n {
      let mut new_segs = Vec::with_capacity(segments.len() * 2);
      for (a, b) in &segments {
        let third = (b - a) / 3.0;
        new_segs.push((*a, a + third));
        new_segs.push((a + 2.0 * third, *b));
      }
      segments = new_segs;
    }

    if d == 1 {
      let mut points: Vec<f64> = Vec::new();
      for (a, b) in &segments {
        points.push(*a);
        points.push(*b);
      }
      let vertex_exprs: Vec<Expr> = points
        .iter()
        .map(|x| Expr::List(vec![Expr::Real(*x)].into()))
        .collect();
      let line_pairs: Vec<Expr> = (0..segments.len())
        .map(|i| {
          Expr::List(
            vec![
              Expr::Integer((2 * i + 1) as i128),
              Expr::Integer((2 * i + 2) as i128),
            ]
            .into(),
          )
        })
        .collect();
      return Ok(Expr::FunctionCall {
        name: "MeshRegion".to_string(),
        args: vec![
          Expr::List(vertex_exprs.into()),
          Expr::List(
            vec![Expr::FunctionCall {
              name: "Line".to_string(),
              args: vec![Expr::List(line_pairs.into())].into(),
            }]
            .into(),
          ),
        ]
        .into(),
      });
    }

    if d == 2 {
      // 1D vertex coordinates: endpoints of all intervals, sorted dedup.
      let mut coords: Vec<f64> = Vec::with_capacity(segments.len() * 2);
      for (a, b) in &segments {
        coords.push(*a);
        coords.push(*b);
      }
      // segments are non-overlapping and given in order, but adjacent
      // ones don't share endpoints — dedupe to be safe.
      coords.sort_by(|x, y| x.partial_cmp(y).unwrap());
      coords.dedup_by(|x, y| (*x - *y).abs() < 1e-15);
      let nc = coords.len();

      // Vertex list: Cartesian product with x outer, y inner.
      let mut vertices: Vec<Expr> = Vec::with_capacity(nc * nc);
      for &x in &coords {
        for &y in &coords {
          vertices.push(Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()));
        }
      }
      // 1-based vertex position for grid cell (i_x, i_y).
      let pos = |ix: usize, iy: usize| -> i128 { (ix * nc + iy + 1) as i128 };

      // Find each segment's lo/hi index in the coords list.
      let seg_indices: Vec<(usize, usize)> = segments
        .iter()
        .map(|(a, b)| {
          let lo = coords.iter().position(|c| (*c - *a).abs() < 1e-15).unwrap();
          let hi = coords.iter().position(|c| (*c - *b).abs() < 1e-15).unwrap();
          (lo, hi)
        })
        .collect();

      // Polygons in cell order (i_x outer, i_y inner).
      let mut polygons: Vec<Expr> = Vec::with_capacity(segments.len().pow(2));
      for &(x_lo, x_hi) in &seg_indices {
        for &(y_lo, y_hi) in &seg_indices {
          polygons.push(Expr::List(
            vec![
              Expr::Integer(pos(x_lo, y_lo)),
              Expr::Integer(pos(x_hi, y_lo)),
              Expr::Integer(pos(x_hi, y_hi)),
              Expr::Integer(pos(x_lo, y_hi)),
            ]
            .into(),
          ));
        }
      }

      return Ok(Expr::FunctionCall {
        name: "MeshRegion".to_string(),
        args: vec![
          Expr::List(vertices.into()),
          Expr::List(
            vec![Expr::FunctionCall {
              name: "Polygon".to_string(),
              args: vec![Expr::List(polygons.into())].into(),
            }]
            .into(),
          ),
        ]
        .into(),
      });
    }
  }

  // ArrayMesh[vec] → MeshRegion from a binary 1D array (Lines)
  if name == "ArrayMesh"
    && args.len() == 1
    && let Expr::List(cells) = &args[0]
    && cells.iter().all(|c| matches!(c, Expr::Integer(_)))
  {
    let bits: Vec<bool> = cells
      .iter()
      .map(|c| !matches!(c, Expr::Integer(0)))
      .collect();
    if bits.iter().all(|b| !*b) {
      return Ok(Expr::FunctionCall {
        name: "EmptyRegion".to_string(),
        args: vec![Expr::Integer(1)].into(),
      });
    }
    // Collect needed vertex positions (endpoints of any non-zero cell).
    let mut needed: std::collections::BTreeSet<i64> =
      std::collections::BTreeSet::new();
    for (i, &b) in bits.iter().enumerate() {
      if b {
        needed.insert(i as i64);
        needed.insert(i as i64 + 1);
      }
    }
    let vertex_list: Vec<i64> = needed.into_iter().collect();
    let position_for: std::collections::HashMap<i64, usize> = vertex_list
      .iter()
      .enumerate()
      .map(|(i, &v)| (v, i + 1))
      .collect();
    let vertex_exprs: Vec<Expr> = vertex_list
      .iter()
      .map(|&v| Expr::List(vec![Expr::Real(v as f64)].into()))
      .collect();
    let mut line_pairs: Vec<Expr> = Vec::new();
    for (i, &b) in bits.iter().enumerate() {
      if b {
        let a = position_for[&(i as i64)];
        let c = position_for[&(i as i64 + 1)];
        line_pairs.push(Expr::List(
          vec![Expr::Integer(a as i128), Expr::Integer(c as i128)].into(),
        ));
      }
    }
    return Ok(Expr::FunctionCall {
      name: "MeshRegion".to_string(),
      args: vec![
        Expr::List(vertex_exprs.into()),
        Expr::List(
          vec![Expr::FunctionCall {
            name: "Line".to_string(),
            args: vec![Expr::List(line_pairs.into())].into(),
          }]
          .into(),
        ),
      ]
      .into(),
    });
  }

  // ArrayMesh[matrix] → MeshRegion from a binary 2D array
  if name == "ArrayMesh"
    && args.len() == 1
    && let Expr::List(rows) = &args[0]
  {
    let nrows = rows.len();
    if nrows == 0 {
      return Ok(unevaluated("ArrayMesh", args));
    }
    // Parse the matrix
    let mut matrix: Vec<Vec<bool>> = Vec::new();
    let mut ncols = 0usize;
    for row in rows {
      if let Expr::List(cols) = row {
        if ncols == 0 {
          ncols = cols.len();
        }
        let row_vals: Vec<bool> = cols
          .iter()
          .map(|e| !matches!(e, Expr::Integer(0)))
          .collect();
        matrix.push(row_vals);
      } else {
        return Ok(unevaluated("ArrayMesh", args));
      }
    }

    // Collect vertices: process columns left-to-right, within each column rows bottom-to-top
    // Each cell corner is (x_left/x_right, y_bottom/y_top) added as BL, TL, BR, TR
    let mut vertices: Vec<(f64, f64)> = Vec::new();
    let mut vertex_index =
      std::collections::HashMap::<(i64, i64), usize>::new();

    let add_vertex =
      |x: i64,
       y: i64,
       vertices: &mut Vec<(f64, f64)>,
       index: &mut std::collections::HashMap<(i64, i64), usize>|
       -> usize {
        if let Some(&idx) = index.get(&(x, y)) {
          idx
        } else {
          let idx = vertices.len() + 1; // 1-indexed
          vertices.push((x as f64, y as f64));
          index.insert((x, y), idx);
          idx
        }
      };

    // Phase 1: collect vertices in the right order
    for col in 0..ncols {
      for row in (0..nrows).rev() {
        if matrix[row][col] {
          let y_bottom = (nrows - 1 - row) as i64;
          let y_top = y_bottom + 1;
          let x_left = col as i64;
          let x_right = x_left + 1;
          // BL, TL, BR, TR order
          add_vertex(x_left, y_bottom, &mut vertices, &mut vertex_index);
          add_vertex(x_left, y_top, &mut vertices, &mut vertex_index);
          add_vertex(x_right, y_bottom, &mut vertices, &mut vertex_index);
          add_vertex(x_right, y_top, &mut vertices, &mut vertex_index);
        }
      }
    }

    // Phase 2: create polygons in row-major order (top-to-bottom, left-to-right)
    let mut polygons: Vec<Expr> = Vec::new();
    for row in 0..nrows {
      for col in 0..ncols {
        if matrix[row][col] {
          let y_bottom = (nrows - 1 - row) as i64;
          let y_top = y_bottom + 1;
          let x_left = col as i64;
          let x_right = x_left + 1;
          // CCW: BR, TR, TL, BL
          let br = vertex_index[&(x_right, y_bottom)];
          let tr = vertex_index[&(x_right, y_top)];
          let tl = vertex_index[&(x_left, y_top)];
          let bl = vertex_index[&(x_left, y_bottom)];
          polygons.push(Expr::List(
            vec![
              Expr::Integer(br as i128),
              Expr::Integer(tr as i128),
              Expr::Integer(tl as i128),
              Expr::Integer(bl as i128),
            ]
            .into(),
          ));
        }
      }
    }

    // Build MeshRegion[vertices, {Polygon[polygons]}]
    let vertex_exprs: Vec<Expr> = vertices
      .iter()
      .map(|(x, y)| Expr::List(vec![Expr::Real(*x), Expr::Real(*y)].into()))
      .collect();

    return Ok(Expr::FunctionCall {
      name: "MeshRegion".to_string(),
      args: vec![
        Expr::List(vertex_exprs.into()),
        Expr::List(
          vec![Expr::FunctionCall {
            name: "Polygon".to_string(),
            args: vec![Expr::List(polygons.into())].into(),
          }]
          .into(),
        ),
      ]
      .into(),
    });
  }

  // FindDistributionParameters[data, dist[var1, var2]]
  // Closed-form MLE for known distributions.
  if name == "FindDistributionParameters"
    && args.len() == 2
    && let Expr::List(_) = &args[0]
    && let Expr::FunctionCall {
      name: dist_name,
      args: dist_args,
    } = &args[1]
    && dist_args.len() == 2
  {
    let data = args[0].clone();
    let v1 = dist_args[0].clone();
    let v2 = dist_args[1].clone();
    // wolframscript always returns MachinePrecision results.
    let to_real = |v: Expr| -> Result<Expr, InterpreterError> {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "N".to_string(),
        args: vec![v].into(),
      })
    };
    let rule = |p: Expr, v: Expr| Expr::Rule {
      pattern: Box::new(p),
      replacement: Box::new(v),
    };
    match dist_name.as_str() {
      "LaplaceDistribution" => {
        let median = crate::functions::median_ast(&data)?;
        let median = crate::evaluator::evaluate_expr_to_expr(&median)?;
        // mean(|x - median|)
        let abs_diffs = if let Expr::List(items) = &data {
          let mut out = Vec::with_capacity(items.len());
          for x in items.iter() {
            let diff = Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(x.clone()),
              right: Box::new(median.clone()),
            };
            let abs_expr = Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![diff].into(),
            };
            out.push(crate::evaluator::evaluate_expr_to_expr(&abs_expr)?);
          }
          Expr::List(out.into())
        } else {
          unreachable!()
        };
        let scale = crate::functions::mean_ast(&[abs_diffs])?;
        let scale = crate::evaluator::evaluate_expr_to_expr(&scale)?;
        return Ok(Expr::List(
          vec![rule(v1, to_real(median)?), rule(v2, to_real(scale)?)].into(),
        ));
      }
      "NormalDistribution" => {
        let mean = crate::functions::mean_ast(&[data.clone()])?;
        let mean = crate::evaluator::evaluate_expr_to_expr(&mean)?;
        // population variance: sum((x - mean)^2) / n
        let (n, squared_diffs) = if let Expr::List(items) = &data {
          let mut out = Vec::with_capacity(items.len());
          for x in items.iter() {
            let diff = Expr::BinaryOp {
              op: BinaryOperator::Minus,
              left: Box::new(x.clone()),
              right: Box::new(mean.clone()),
            };
            let sq = Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: Box::new(diff),
              right: Box::new(Expr::Integer(2)),
            };
            out.push(crate::evaluator::evaluate_expr_to_expr(&sq)?);
          }
          (items.len() as i128, Expr::List(out.into()))
        } else {
          unreachable!()
        };
        let sum_sq = Expr::FunctionCall {
          name: "Total".to_string(),
          args: vec![squared_diffs].into(),
        };
        let variance = Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(sum_sq),
          right: Box::new(Expr::Integer(n)),
        };
        let std_dev = Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![variance].into(),
        };
        let std_dev = crate::evaluator::evaluate_expr_to_expr(&std_dev)?;
        return Ok(Expr::List(
          vec![rule(v1, to_real(mean)?), rule(v2, to_real(std_dev)?)].into(),
        ));
      }
      _ => {}
    }
  }

  // ParameterMixtureDistribution[
  //   BinomialDistribution[n, p],
  //   Distributed[p, BetaDistribution[α, β]]]
  // → BetaBinomialDistribution[α, β, n]
  if name == "ParameterMixtureDistribution"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: dist_name,
      args: dist_args,
    } = &args[0]
    && dist_name == "BinomialDistribution"
    && dist_args.len() == 2
    && let Expr::FunctionCall {
      name: mix_name,
      args: mix_args,
    } = &args[1]
    && mix_name == "Distributed"
    && mix_args.len() == 2
    && let Expr::FunctionCall {
      name: param_dist_name,
      args: param_dist_args,
    } = &mix_args[1]
    && param_dist_name == "BetaDistribution"
    && param_dist_args.len() == 2
    && crate::syntax::expr_to_string(&mix_args[0])
      == crate::syntax::expr_to_string(&dist_args[1])
  {
    return Ok(Expr::FunctionCall {
      name: "BetaBinomialDistribution".to_string(),
      args: vec![
        param_dist_args[0].clone(),
        param_dist_args[1].clone(),
        dist_args[0].clone(),
      ]
      .into(),
    });
  }

  // PetersenGraph[] / PetersenGraph[n, k] / PetersenGraph[n, k, opts…] —
  // the generalized Petersen graph GP(n, k). wolframscript labels the
  // k-jump ring 1..n, the plain cycle n+1..2n, with spokes i—(n+i), and
  // lists the edges in ascending order. Defaults are (n=5, k=2).
  if name == "PetersenGraph"
    && (args.is_empty()
      || matches!(&args[0], Expr::Integer(_))
        && (args.len() == 1
          || matches!(&args[1], Expr::Integer(_))
          || args[1..].iter().all(|a| matches!(a, Expr::Rule { .. }))))
  {
    let (n, k, opts_start) = match args.first() {
      None => (5usize, 2usize, 0),
      Some(Expr::Integer(n)) if *n > 0 => match args.get(1) {
        Some(Expr::Integer(k)) if *k > 0 && (*k as usize) < *n as usize => {
          (*n as usize, *k as usize, 2)
        }
        _ => (*n as usize, 2, 1),
      },
      _ => {
        return Ok(unevaluated(name, args));
      }
    };
    let und = |a: usize, b: usize| Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)].into(),
    };
    let mut pairs: std::collections::BTreeSet<(usize, usize)> =
      std::collections::BTreeSet::new();
    // Ring 1..n joined by k-step jumps (deduplicated for even n = 2k).
    for i in 0..n {
      let a = 1 + i;
      let b = 1 + (i + k) % n;
      if a != b {
        pairs.insert((a.min(b), a.max(b)));
      }
    }
    // Spokes i—(n+i).
    for i in 1..=n {
      pairs.insert((i, n + i));
    }
    // Plain cycle n+1—n+2—…—2n—n+1.
    for i in 0..n {
      let a = n + 1 + i;
      let b = n + 1 + (i + 1) % n;
      pairs.insert((a.min(b), a.max(b)));
    }
    let edges: Vec<Expr> = pairs.into_iter().map(|(a, b)| und(a, b)).collect();
    let vertices: Vec<Expr> =
      (1..=2 * n).map(|i| Expr::Integer(i as i128)).collect();
    let mut graph_args =
      vec![Expr::List(vertices.into()), Expr::List(edges.into())];
    let opts: Vec<Expr> = args[opts_start..]
      .iter()
      .filter(|a| matches!(a, Expr::Rule { .. }))
      .cloned()
      .collect();
    if !opts.is_empty() {
      graph_args.push(Expr::List(opts.into()));
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: graph_args.into(),
    });
  }

  // GraphPlot[edges] / GraphPlot[edges, opts…] / GraphPlot[adj_matrix] —
  // currently a thin alias for Graph that reuses Woxi's existing graph
  // rendering. LayeredGraphPlot and similar variants forward the same
  // way so they produce a `-Graphics-` placeholder matching wolframscript.
  if matches!(
    name,
    "GraphPlot" | "LayeredGraphPlot" | "GraphPlot3D" | "TreePlot"
  ) && !args.is_empty()
  {
    // Only forward when args[0] is a list of edges or an already-built
    // Graph[…]; otherwise keep the symbolic head (matches Woxi's
    // long-standing "stay unevaluated" wrapper behaviour for opaque
    // arguments like a bare symbol).
    let looks_like_graph_input = match &args[0] {
      Expr::List(_) => true,
      Expr::FunctionCall { name: n, .. } => n == "Graph" || n == "SparseArray",
      _ => false,
    };
    if looks_like_graph_input {
      let graph_expr = if let Expr::FunctionCall {
        name: gn,
        args: gargs,
      } = &args[0]
        && gn == "Graph"
      {
        Expr::FunctionCall {
          name: "Graph".to_string(),
          args: gargs.clone(),
        }
      } else {
        unevaluated("Graph", args)
      };
      // Evaluate the embedded Graph so any rules/edges are normalised
      // into the canonical {vertex_list, edge_list, opts…} form, then
      // explicitly render to Graphics. Unlike plain `Graph[…]`, the
      // GraphPlot family is a visualisation primitive and so should
      // print as the `-Graphics-` placeholder rather than the
      // `Graph[<n>, <m>]` data-structure summary.
      let evaluated = crate::evaluator::evaluate_expr_to_expr(&graph_expr)?;
      if let Expr::FunctionCall {
        name: en,
        args: eargs,
      } = &evaluated
        && en == "Graph"
        && eargs.len() >= 2
        && let Ok(rendered) = crate::functions::graph::graph_ast(eargs)
      {
        if let Expr::Graphics { svg, .. } = &rendered {
          crate::capture_graphics(svg);
        }
        return Ok(rendered);
      }
      return Ok(evaluated);
    }
  }

  // AngularGauge[value, {min, max}, opts...] → Graphics dial showing
  // `value` on a circular scale from `min` to `max`. The gauge sweeps
  // clockwise from -135° at `min` to 135° at `max`. Options are kept on
  // the resulting Graphics so wolframscript's `-Graphics-` placeholder
  // is produced for callers that just check the head.
  if name == "AngularGauge" && args.len() >= 2 {
    let value = crate::functions::math_ast::try_eval_to_f64(&args[0]);
    let (lo, hi) = match &args[1] {
      Expr::List(items) if items.len() == 2 => (
        crate::functions::math_ast::try_eval_to_f64(&items[0]),
        crate::functions::math_ast::try_eval_to_f64(&items[1]),
      ),
      _ => (None, None),
    };
    let mut primitives: Vec<Expr> = Vec::new();
    // Outer dial.
    primitives.push(Expr::FunctionCall {
      name: "Circle".to_string(),
      args: vec![
        Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
        Expr::Integer(1),
      ]
      .into(),
    });
    // Needle from center to the angle corresponding to `value`.
    if let (Some(v), Some(lo), Some(hi)) = (value, lo, hi)
      && hi != lo
    {
      let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
      let start_deg = -135.0f64;
      let end_deg = 135.0f64;
      let theta_deg = start_deg + t * (end_deg - start_deg);
      // Wolfram's angular convention: 0° at top, increasing clockwise.
      // For SVG-style Graphics axes we use math convention with the dial
      // rotated by 90°.
      let theta = (90.0 - theta_deg).to_radians();
      let nx = theta.cos();
      let ny = theta.sin();
      primitives.push(Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(
          vec![
            Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
            Expr::List(vec![Expr::Real(nx), Expr::Real(ny)].into()),
          ]
          .into(),
        )]
        .into(),
      });
    }
    let mut graphics_args = vec![Expr::List(primitives.into())];
    for opt in &args[2..] {
      if matches!(opt, Expr::Rule { .. }) {
        graphics_args.push(opt.clone());
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graphics".to_string(),
      args: graphics_args.into(),
    });
  }

  // VoronoiMesh[{{x1,y1},{x2,y2},...}] → Voronoi tessellation as MeshRegion
  if name == "VoronoiMesh" && args.len() == 1 {
    return crate::functions::voronoi::voronoi_mesh_ast(args);
  }

  // ExpressionGraph[expr] → Graph of the expression tree
  if name == "ExpressionGraph" && args.len() == 1 {
    let mut counter = 0u64;
    let mut vertices = Vec::new();
    let mut edges = Vec::new();

    fn walk_expr(
      expr: &Expr,
      counter: &mut u64,
      vertices: &mut Vec<Expr>,
      edges: &mut Vec<Expr>,
    ) -> u64 {
      *counter += 1;
      let my_id = *counter;
      vertices.push(Expr::Integer(my_id as i128));

      // Get children based on expression type
      let children: Option<&[Expr]> = match expr {
        Expr::FunctionCall { args, .. } => Some(args),
        Expr::List(items) => Some(items),
        Expr::BinaryOp { .. } => None, // handled separately
        _ => None,
      };

      if let Some(kids) = children {
        for child in kids {
          let child_id = *counter + 1;
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![
              Expr::Integer(my_id as i128),
              Expr::Integer(child_id as i128),
            ]
            .into(),
          });
          walk_expr(child, counter, vertices, edges);
        }
      } else if let Expr::BinaryOp { left, right, .. } = expr {
        let left_id = *counter + 1;
        edges.push(Expr::FunctionCall {
          name: "UndirectedEdge".to_string(),
          args: vec![
            Expr::Integer(my_id as i128),
            Expr::Integer(left_id as i128),
          ]
          .into(),
        });
        walk_expr(left, counter, vertices, edges);
        let right_id = *counter + 1;
        edges.push(Expr::FunctionCall {
          name: "UndirectedEdge".to_string(),
          args: vec![
            Expr::Integer(my_id as i128),
            Expr::Integer(right_id as i128),
          ]
          .into(),
        });
        walk_expr(right, counter, vertices, edges);
      }
      // Atoms (Integer, Real, Identifier, String, etc.) have no children

      my_id
    }

    walk_expr(&args[0], &mut counter, &mut vertices, &mut edges);

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // PlanarGraph[...] is treated as Graph[...] with GraphLayout -> "TutteEmbedding"
  if name == "PlanarGraph" {
    let mut graph = evaluate_function_call_ast("Graph", args)?;
    // Append {GraphLayout -> "TutteEmbedding"} option to the Graph args
    if let Expr::FunctionCall {
      name: ref gn,
      args: ref mut ga,
    } = graph
      && gn == "Graph"
    {
      ga.push(Expr::List(
        vec![Expr::Rule {
          pattern: Box::new(Expr::Identifier("GraphLayout".to_string())),
          replacement: Box::new(Expr::String("TutteEmbedding".to_string())),
        }]
        .into(),
      ));
    }
    return Ok(graph);
  }

  // Graph[{rule1, rule2, ...}] or Graph[{edge1, edge2, ...}]
  // → Graph[{sorted vertices}, {DirectedEdge/UndirectedEdge[...], ...}]
  if name == "Graph" {
    // Normalize TwoWayRule[a, b] (produced by the `<->` operator) to
    // UndirectedEdge[a, b] inside edge lists so all downstream edge
    // handling stays uniform. Recursively rewrites through Lists and
    // Labeled wrappers.
    fn rewrite_two_way_rules(e: &Expr) -> Expr {
      match e {
        Expr::FunctionCall { name, args } if name == "TwoWayRule" => {
          Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: args.clone(),
          }
        }
        Expr::FunctionCall { name, args }
          if name == "Labeled" && args.len() == 2 =>
        {
          Expr::FunctionCall {
            name: "Labeled".to_string(),
            args: vec![rewrite_two_way_rules(&args[0]), args[1].clone()].into(),
          }
        }
        Expr::List(items) => {
          Expr::List(items.iter().map(rewrite_two_way_rules).collect())
        }
        other => other.clone(),
      }
    }
    let args_vec: Vec<Expr> = args.iter().map(rewrite_two_way_rules).collect();
    let args: &[Expr] = &args_vec;

    // Helper: check if expr is a graph edge (DirectedEdge/UndirectedEdge,
    // possibly wrapped in Labeled)
    fn is_graph_edge(e: &Expr) -> bool {
      let inner = match e {
        Expr::FunctionCall { name, args }
          if name == "Labeled" && args.len() == 2 =>
        {
          &args[0]
        }
        _ => e,
      };
      matches!(inner, Expr::FunctionCall { name, args }
        if (name == "UndirectedEdge" || name == "DirectedEdge") && args.len() == 2)
    }

    // Helper: extract vertices from an edge (possibly Labeled)
    fn edge_vertices(e: &Expr) -> Option<(&Expr, &Expr)> {
      let inner = match e {
        Expr::FunctionCall { name, args }
          if name == "Labeled" && args.len() == 2 =>
        {
          &args[0]
        }
        _ => e,
      };
      if let Expr::FunctionCall { args: eargs, .. } = inner
        && eargs.len() == 2
      {
        Some((&eargs[0], &eargs[1]))
      } else {
        None
      }
    }

    // Collect options (Rule expressions) from args after the edge/vertex lists
    let options_start = if args.len() >= 2
      && matches!(&args[0], Expr::List(_))
      && matches!(&args[1], Expr::List(_))
    {
      2
    } else {
      1
    };
    let trailing_options: Vec<Expr> = args[options_start..]
      .iter()
      .filter(|a| matches!(a, Expr::Rule { .. }))
      .cloned()
      .collect();

    if !args.is_empty()
      && let Expr::List(edges) = &args[0]
    {
      // Graph[{item1, item2, ...}, opts...] where each item is either
      // a Rule (a -> b, becomes DirectedEdge), a DirectedEdge, or an
      // UndirectedEdge (possibly wrapped in Labeled). Mixed lists are
      // supported — Wolfram accepts them too.
      let all_edge_like = edges
        .iter()
        .all(|e| matches!(e, Expr::Rule { .. }) || is_graph_edge(e));
      if all_edge_like && !edges.is_empty() {
        let mut vertex_set: Vec<Expr> = Vec::new();
        let mut out_edges: Vec<Expr> = Vec::with_capacity(edges.len());
        let push_vertex = |v: &Expr, set: &mut Vec<Expr>| {
          if !set.iter().any(|existing| {
            crate::evaluator::pattern_matching::expr_equal(existing, v)
          }) {
            set.push(v.clone());
          }
        };
        for e in edges {
          match e {
            Expr::Rule {
              pattern,
              replacement,
            } => {
              let src = (**pattern).clone();
              let dst = (**replacement).clone();
              push_vertex(&src, &mut vertex_set);
              push_vertex(&dst, &mut vertex_set);
              out_edges.push(Expr::FunctionCall {
                name: "DirectedEdge".to_string(),
                args: vec![src, dst].into(),
              });
            }
            _ => {
              if let Some((src, dst)) = edge_vertices(e) {
                push_vertex(src, &mut vertex_set);
                push_vertex(dst, &mut vertex_set);
              }
              out_edges.push(e.clone());
            }
          }
        }
        // Vertices are kept in first-appearance order (matching
        // wolframscript's VertexList), not sorted.
        let mut result_args =
          vec![Expr::List(vertex_set.into()), Expr::List(out_edges.into())];
        result_args.extend(trailing_options);
        return Ok(Expr::FunctionCall {
          name: "Graph".to_string(),
          args: result_args.into(),
        });
      }
    }

    // Graph[{vertices}, {edges}, opts...] — already normalized, pass through
    // with options preserved
    if args.len() >= 2
      && matches!(&args[0], Expr::List(_))
      && matches!(&args[1], Expr::List(_))
    {
      return Ok(unevaluated(name, args));
    }

    // Fall through: return as inert
    return Ok(unevaluated(name, args));
  }

  // FindCycle[graph | edgeList, kspec, n] → list of cycles (each a list of edges)
  if name == "FindCycle" && (1..=3).contains(&args.len()) {
    return crate::functions::graph::find_cycle_ast(args);
  }

  // FindHamiltonianCycle[graph] → length-1 list with one Hamiltonian cycle, or {}
  if name == "FindHamiltonianCycle" && args.len() == 1 {
    return crate::functions::graph::find_hamiltonian_cycle_ast(args);
  }

  // FindFundamentalCycles[graph | edgeList] → BFS fundamental cycle basis
  if name == "FindFundamentalCycles" {
    return crate::functions::graph::find_fundamental_cycles_ast(args);
  }

  // TransitiveClosureGraph[graph | edgeList] → graph with reachability edges
  if name == "TransitiveClosureGraph" && args.len() == 1 {
    return crate::functions::graph::transitive_closure_graph_ast(args);
  }

  // TransitiveReductionGraph[graph | edgeList] → minimal same-reachability graph
  if name == "TransitiveReductionGraph" && args.len() == 1 {
    return crate::functions::graph::transitive_reduction_graph_ast(args);
  }

  // ReverseGraph[graph] → graph with all directed edges reversed
  if name == "ReverseGraph" && !args.is_empty() && args.len() <= 2 {
    return crate::functions::graph::reverse_graph_ast(args);
  }

  if name == "DirectedGraph" && args.len() == 1 {
    return crate::functions::graph::directed_graph_ast(args);
  }

  // IncidenceGraph[m] / IncidenceGraph[vertices, m] → graph from an
  // incidence matrix
  if name == "IncidenceGraph" && !args.is_empty() && args.len() <= 2 {
    return crate::functions::graph::incidence_graph_ast(args);
  }

  // KirchhoffGraph[m] / KirchhoffGraph[vertices, m] → graph from a
  // Kirchhoff (Laplacian) matrix; trailing Graph options pass through
  if name == "KirchhoffGraph" && !args.is_empty() {
    return crate::functions::graph::kirchhoff_graph_ast(args);
  }

  // FindIndependentVertexSet[graph] → {maximum independent vertex set}
  if name == "FindIndependentVertexSet" && args.len() == 1 {
    return crate::functions::graph::find_independent_vertex_set_ast(args);
  }

  // VertexComponent[graph, v] → vertices that can reach v
  if name == "VertexComponent" && args.len() == 2 {
    return crate::functions::graph::vertex_component_ast(args);
  }

  // WeightedAdjacencyGraph[wmat] → weighted graph (Infinity = no edge)
  if name == "WeightedAdjacencyGraph" && (args.len() == 1 || args.len() == 2) {
    return crate::functions::graph::weighted_adjacency_graph_ast(args);
  }

  // FindMinimumCostFlow[cmat, s, t] → minimum cost of a maximum flow
  if name == "FindMinimumCostFlow" && args.len() == 3 {
    return crate::functions::graph::find_minimum_cost_flow_ast(args);
  }

  // NearestNeighborGraph[points] → undirected k-nearest-neighbor graph
  if name == "NearestNeighborGraph" && (args.len() == 1 || args.len() == 2) {
    return crate::functions::graph::nearest_neighbor_graph_ast(args);
  }

  // FindShortestPath[graph, src, dst, opts...] → list of vertices on a
  // shortest weighted path (Dijkstra).
  if name == "FindShortestPath" && args.len() >= 3 {
    return crate::functions::graph::find_shortest_path_ast(args);
  }

  // EdgeList[Graph[vertices, edges]] → edges list
  if name == "EdgeList" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(_) = &gargs[1]
    {
      return Ok(gargs[1].clone());
    }
    // Return unevaluated for non-graph input
    return Ok(unevaluated(name, args));
  }

  // EdgeList[graph, patt] keeps the edges matching the pattern, i.e.
  // Cases[EdgeList[graph], patt].
  if name == "EdgeList"
    && args.len() == 2
    && matches!(&args[0], Expr::FunctionCall { name: g, args: ga }
      if g == "Graph" && ga.len() >= 2)
  {
    let edge_list =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "EdgeList".to_string(),
        args: vec![args[0].clone()].into(),
      })?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Cases".to_string(),
      args: vec![edge_list, args[1].clone()].into(),
    });
  }

  // VertexList[Graph[vertices, edges]] → vertices list
  if name == "VertexList" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(_) = &gargs[0]
    {
      return Ok(gargs[0].clone());
    }
    return Ok(unevaluated(name, args));
  }

  // VertexList[graph, patt] keeps the vertices matching the pattern, i.e.
  // Cases[VertexList[graph], patt] (an integer is a literal-value pattern).
  if name == "VertexList"
    && args.len() == 2
    && matches!(&args[0], Expr::FunctionCall { name: g, args: ga }
      if g == "Graph" && ga.len() >= 2)
  {
    let vertex_list =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "VertexList".to_string(),
        args: vec![args[0].clone()].into(),
      })?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Cases".to_string(),
      args: vec![vertex_list, args[1].clone()].into(),
    });
  }

  // VertexDelete[Graph[vertices, edges], v | {v1, v2, ...}] →
  //   remove the given vertices and every edge incident to them.
  if name == "VertexDelete" && args.len() == 2 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(vertices) = &gargs[0]
      && let Expr::List(edges) = &gargs[1]
    {
      use crate::evaluator::pattern_matching::expr_equal;
      // A List second argument denotes several vertices to delete; any
      // other expression denotes a single vertex (which may itself be a
      // list-valued vertex name).
      let to_delete: Vec<Expr> = match &args[1] {
        Expr::List(vs) => vs.iter().cloned().collect(),
        other => vec![other.clone()],
      };
      // Every requested vertex must be present, otherwise emit the
      // VertexDelete::inv message (showing the original second argument)
      // and leave the expression unevaluated.
      let all_valid = to_delete
        .iter()
        .all(|d| vertices.iter().any(|v| expr_equal(v, d)));
      if !all_valid {
        crate::emit_message(&format!(
          "VertexDelete::inv: The argument {} in {} is not a valid vertex.",
          expr_to_string(&args[1]),
          expr_to_string(&unevaluated(name, args))
        ));
        return Ok(unevaluated(name, args));
      }
      let is_deleted = |v: &Expr| to_delete.iter().any(|d| expr_equal(v, d));
      // Helper: extract the two endpoints of an edge (possibly Labeled).
      fn vd_edge_vertices(e: &Expr) -> Option<(&Expr, &Expr)> {
        let inner = match e {
          Expr::FunctionCall { name, args }
            if name == "Labeled" && args.len() == 2 =>
          {
            &args[0]
          }
          _ => e,
        };
        match inner {
          Expr::FunctionCall { args: eargs, .. } if eargs.len() == 2 => {
            Some((&eargs[0], &eargs[1]))
          }
          Expr::Rule {
            pattern,
            replacement,
          } => Some((pattern.as_ref(), replacement.as_ref())),
          _ => None,
        }
      }
      let new_vertices: Vec<Expr> = vertices
        .iter()
        .filter(|v| !is_deleted(v))
        .cloned()
        .collect();
      let new_edges: Vec<Expr> = edges
        .iter()
        .filter(|e| match vd_edge_vertices(e) {
          Some((a, b)) => !is_deleted(a) && !is_deleted(b),
          None => true,
        })
        .cloned()
        .collect();
      let mut result_args = vec![
        Expr::List(new_vertices.into()),
        Expr::List(new_edges.into()),
      ];
      // Preserve any trailing graph options.
      result_args.extend(gargs[2..].iter().cloned());
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: result_args.into(),
      });
    }
    return Ok(unevaluated(name, args));
  }

  // VertexContract[Graph[...], {v1, v2, ...}] → merge the listed vertices into
  // the first one (v1). If any listed vertex is absent, wolframscript leaves
  // the graph unchanged.
  if name == "VertexContract" && args.len() == 2 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
    {
      let to_contract: Vec<Expr> = match &args[1] {
        Expr::List(vs) => vs.iter().cloned().collect(),
        other => vec![other.clone()],
      };
      return Ok(
        crate::functions::graph::contract_vertices_in_graph(
          gargs,
          &to_contract,
        )
        .unwrap_or_else(|| args[0].clone()),
      );
    }
    return Ok(unevaluated(name, args));
  }

  // EdgeContract[Graph[...], u <-> v] → contract a single existing edge,
  // equivalent to VertexContract[graph, {u, v}] but only when that edge is
  // present (otherwise the graph is returned unchanged). A multi-edge list
  // argument is left unevaluated.
  if name == "EdgeContract" && args.len() == 2 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(edges) = &gargs[1]
      && let Some((u, v, directed)) =
        crate::functions::graph::edge_endpoints(&args[1])
    {
      use crate::evaluator::pattern_matching::expr_equal;
      let edge_present = edges.iter().any(|e| {
        match crate::functions::graph::edge_endpoints(e) {
          Some((a, b, ed)) if ed == directed => {
            if directed {
              expr_equal(&a, &u) && expr_equal(&b, &v)
            } else {
              (expr_equal(&a, &u) && expr_equal(&b, &v))
                || (expr_equal(&a, &v) && expr_equal(&b, &u))
            }
          }
          _ => false,
        }
      });
      if edge_present {
        return Ok(
          crate::functions::graph::contract_vertices_in_graph(gargs, &[u, v])
            .unwrap_or_else(|| args[0].clone()),
        );
      }
      return Ok(args[0].clone());
    }
    return Ok(unevaluated(name, args));
  }

  // GraphDisjointUnion[g1, g2, ...] → disjoint union with vertices relabeled to
  // consecutive integers 1..N (each graph shifted by the running vertex count).
  if name == "GraphDisjointUnion" && !args.is_empty() {
    let mut parsed: Vec<(&[Expr], &[Expr])> = Vec::new();
    let mut ok = true;
    for a in args {
      if let Expr::FunctionCall {
        name: gname,
        args: gargs,
      } = a
        && gname == "Graph"
        && gargs.len() >= 2
        && let (Expr::List(v), Expr::List(e)) = (&gargs[0], &gargs[1])
      {
        parsed.push((&v[..], &e[..]));
      } else {
        ok = false;
        break;
      }
    }
    if ok && !parsed.is_empty() {
      return Ok(crate::functions::graph::graph_disjoint_union(&parsed));
    }
    return Ok(unevaluated(name, args));
  }

  // GraphReciprocity[Graph[verts, edges]] → fraction of directed edges that are
  // reciprocated. Edgeless, mixed, and multigraphs stay unevaluated (without the
  // not-yet-implemented note), matching wolframscript.
  if name == "GraphReciprocity" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(edges) = &gargs[1]
      && let Some(result) = crate::functions::graph::graph_reciprocity(edges)
    {
      return Ok(result);
    }
    return Ok(unevaluated(name, args));
  }

  // EdgeAdd[Graph[vertices, edges], e | {e1, e2, ...}] →
  //   append the given edge(s) to the graph and add any new endpoint
  //   vertices (in order of first appearance). A directed edge a -> b
  //   (Rule / DirectedEdge) is stored as DirectedEdge[a, b]; an undirected
  //   edge a <-> b (TwoWayRule / UndirectedEdge) is stored as
  //   UndirectedEdge[a, b]. Existing edges are kept, so re-adding an edge
  //   yields a multigraph (matching Wolfram).
  if name == "EdgeAdd" && args.len() == 2 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(vertices) = &gargs[0]
      && let Expr::List(edges) = &gargs[1]
    {
      use crate::evaluator::pattern_matching::expr_equal;
      // Normalize an edge specification to its canonical stored form,
      // returning the canonical edge and its two endpoints. Returns None
      // for anything that is not a valid edge specification.
      fn ea_canonical(e: &Expr) -> Option<(Expr, Expr, Expr)> {
        match e {
          Expr::Rule {
            pattern,
            replacement,
          } => Some((
            Expr::FunctionCall {
              name: "DirectedEdge".to_string(),
              args: vec![(**pattern).clone(), (**replacement).clone()].into(),
            },
            (**pattern).clone(),
            (**replacement).clone(),
          )),
          Expr::FunctionCall { name, args }
            if name == "DirectedEdge" && args.len() == 2 =>
          {
            Some((e.clone(), args[0].clone(), args[1].clone()))
          }
          Expr::FunctionCall { name, args }
            if (name == "UndirectedEdge" || name == "TwoWayRule")
              && args.len() == 2 =>
          {
            Some((
              Expr::FunctionCall {
                name: "UndirectedEdge".to_string(),
                args: vec![args[0].clone(), args[1].clone()].into(),
              },
              args[0].clone(),
              args[1].clone(),
            ))
          }
          _ => None,
        }
      }
      // A List second argument denotes several edges to add; any other
      // expression denotes a single edge.
      let to_add: Vec<Expr> = match &args[1] {
        Expr::List(es) => es.iter().cloned().collect(),
        other => vec![other.clone()],
      };
      // Every requested edge must be a valid edge specification, otherwise
      // emit the EdgeAdd::inv message and leave the expression unevaluated.
      if to_add.iter().any(|e| ea_canonical(e).is_none()) {
        crate::emit_message(&format!(
          "EdgeAdd::inv: The argument {} in {} is not a valid edge.",
          expr_to_string(&args[1]),
          expr_to_string(&unevaluated(name, args))
        ));
        return Ok(unevaluated(name, args));
      }
      let mut new_vertices: Vec<Expr> = vertices.iter().cloned().collect();
      let mut new_edges: Vec<Expr> = edges.iter().cloned().collect();
      for e in &to_add {
        let (canon, a, b) = ea_canonical(e).unwrap();
        for v in [a, b] {
          if !new_vertices.iter().any(|x| expr_equal(x, &v)) {
            new_vertices.push(v);
          }
        }
        new_edges.push(canon);
      }
      let mut result_args = vec![
        Expr::List(new_vertices.into()),
        Expr::List(new_edges.into()),
      ];
      // Preserve any trailing graph options.
      result_args.extend(gargs[2..].iter().cloned());
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: result_args.into(),
      });
    }
    return Ok(unevaluated(name, args));
  }

  // EdgeDelete[Graph[vertices, edges], e | {e1, e2, ...}] →
  //   remove the given edges (one matching occurrence per requested edge).
  //   Vertices are preserved. A directed edge a -> b (or DirectedEdge[a, b])
  //   matches the stored edge with the same orientation; an undirected edge
  //   a <-> b (or UndirectedEdge[a, b]) matches regardless of endpoint order.
  if name == "EdgeDelete" && args.len() == 2 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(vertices) = &gargs[0]
      && let Expr::List(edges) = &gargs[1]
    {
      use crate::evaluator::pattern_matching::expr_equal;
      // Canonical representation of an edge for matching: (directed?, a, b).
      // For undirected edges the endpoints are stored sorted by string form
      // so that a <-> b and b <-> a compare equal.
      fn edge_key(e: &Expr) -> Option<(bool, Expr, Expr)> {
        match e {
          Expr::Rule {
            pattern,
            replacement,
          } => Some((true, (**pattern).clone(), (**replacement).clone())),
          Expr::FunctionCall { name, args }
            if name == "DirectedEdge" && args.len() == 2 =>
          {
            Some((true, args[0].clone(), args[1].clone()))
          }
          Expr::FunctionCall { name, args }
            if (name == "UndirectedEdge" || name == "TwoWayRule")
              && args.len() == 2 =>
          {
            Some((false, args[0].clone(), args[1].clone()))
          }
          _ => None,
        }
      }
      fn edge_matches(a: &Expr, b: &Expr) -> bool {
        match (edge_key(a), edge_key(b)) {
          (Some((da, a1, a2)), Some((db, b1, b2))) => {
            if da != db {
              return false;
            }
            if da {
              expr_equal(&a1, &b1) && expr_equal(&a2, &b2)
            } else {
              (expr_equal(&a1, &b1) && expr_equal(&a2, &b2))
                || (expr_equal(&a1, &b2) && expr_equal(&a2, &b1))
            }
          }
          _ => false,
        }
      }
      // A List second argument denotes several edges to delete; any other
      // expression denotes a single edge.
      let to_delete: Vec<Expr> = match &args[1] {
        Expr::List(es) => es.iter().cloned().collect(),
        other => vec![other.clone()],
      };
      // Every requested edge must occur (with enough multiplicity) in the
      // graph, otherwise emit the EdgeDelete::inv message and leave the
      // expression unevaluated.
      let mut remaining: Vec<Option<Expr>> =
        edges.iter().cloned().map(Some).collect();
      let mut deleted_indices: Vec<usize> = Vec::new();
      let mut all_valid = true;
      for d in &to_delete {
        let mut found = false;
        for (i, slot) in remaining.iter_mut().enumerate() {
          if let Some(stored) = slot
            && edge_matches(stored, d)
          {
            *slot = None;
            deleted_indices.push(i);
            found = true;
            break;
          }
        }
        if !found {
          all_valid = false;
          break;
        }
      }
      if !all_valid || to_delete.iter().any(|d| edge_key(d).is_none()) {
        crate::emit_message(&format!(
          "EdgeDelete::inv: The argument {} in {} is not a valid edge.",
          expr_to_string(&args[1]),
          expr_to_string(&unevaluated(name, args))
        ));
        return Ok(unevaluated(name, args));
      }
      let new_edges: Vec<Expr> = remaining.into_iter().flatten().collect();
      let mut result_args =
        vec![Expr::List(vertices.clone()), Expr::List(new_edges.into())];
      // Preserve any trailing graph options.
      result_args.extend(gargs[2..].iter().cloned());
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: result_args.into(),
      });
    }
    return Ok(unevaluated(name, args));
  }

  // VertexIndex[Graph[vertices, edges], v] → 1-based position of v in VertexList[g]
  if name == "VertexIndex" && args.len() == 2 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(vertices) = &gargs[0]
    {
      let target = expr_to_string(&args[1]);
      if let Some(pos) =
        vertices.iter().position(|v| expr_to_string(v) == target)
      {
        return Ok(Expr::Integer(pos as i128 + 1));
      }
      // Vertex not present: emit message and return unevaluated.
      crate::emit_message(&format!(
        "VertexIndex::inv: The argument {} in {} is not a valid vertex.",
        target,
        expr_to_string(&unevaluated(name, args))
      ));
    }
    return Ok(unevaluated(name, args));
  }

  // IndexGraph[graph] or IndexGraph[graph, r] — reindex vertices to integers
  if (name == "IndexGraph")
    && (args.len() == 1 || args.len() == 2)
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let Expr::List(vertices) = &gargs[0]
    && let Expr::List(edges) = &gargs[1]
  {
    let start: i128 = if args.len() == 2 {
      match &args[1] {
        Expr::Integer(n) => *n,
        _ => {
          return Ok(unevaluated(name, args));
        }
      }
    } else {
      1
    };

    // Build mapping from old vertex to new integer index
    let mut vertex_map = std::collections::HashMap::new();
    for (i, v) in vertices.iter().enumerate() {
      vertex_map.insert(expr_to_string(v), Expr::Integer(start + i as i128));
    }

    let new_vertices: Vec<Expr> = (0..vertices.len())
      .map(|i| Expr::Integer(start + i as i128))
      .collect();

    let new_edges: Vec<Expr> = edges
      .iter()
      .map(|e| {
        if let Expr::FunctionCall {
          name: ename,
          args: eargs,
        } = e
          && eargs.len() == 2
        {
          let new_args: Vec<Expr> = eargs
            .iter()
            .map(|a| {
              vertex_map
                .get(&expr_to_string(a))
                .cloned()
                .unwrap_or_else(|| a.clone())
            })
            .collect();
          Expr::FunctionCall {
            name: ename.clone(),
            args: new_args.into(),
          }
        } else {
          e.clone()
        }
      })
      .collect();

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![
        Expr::List(new_vertices.into()),
        Expr::List(new_edges.into()),
      ]
      .into(),
    });
  }

  // VertexAdd[graph, v] or VertexAdd[graph, {v1, v2, ...}] — add vertices to a graph
  if name == "VertexAdd"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let Expr::List(vertices) = &gargs[0]
    && let Expr::List(edges) = &gargs[1]
  {
    let existing: std::collections::HashSet<String> =
      vertices.iter().map(expr_to_string).collect();
    let mut new_vertices = vertices.clone();

    let to_add = match &args[1] {
      Expr::List(vs) => vs.clone(),
      other => vec![other.clone()].into(),
    };

    for v in to_add {
      if !existing.contains(&expr_to_string(&v)) {
        new_vertices.push(v);
      }
    }

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(new_vertices), Expr::List(edges.clone())].into(),
    });
  }

  // GraphIntersection[g1, g2, ...] — intersection of graphs (union vertices, intersect edges)
  if name == "GraphIntersection" && args.len() >= 2 {
    // Extract all graphs
    let mut graphs = Vec::new();
    for arg in args {
      if let Expr::FunctionCall { name: gn, args: ga } = arg
        && gn == "Graph"
        && ga.len() >= 2
        && let (Expr::List(vs), Expr::List(es)) = (&ga[0], &ga[1])
      {
        graphs.push((vs, es));
      } else {
        // Not a graph, return unevaluated
        return Ok(unevaluated(name, args));
      }
    }

    // Union of all vertices (preserving order, no duplicates)
    let mut seen = std::collections::HashSet::new();
    let mut all_vertices = Vec::new();
    for (vs, _) in &graphs {
      for v in *vs {
        let key = expr_to_string(v);
        if seen.insert(key) {
          all_vertices.push(v.clone());
        }
      }
    }

    // Intersection of edges: keep only edges present in all graphs
    let (_, first_edges) = &graphs[0];
    let edge_sets: Vec<std::collections::HashSet<String>> = graphs[1..]
      .iter()
      .map(|(_, es)| es.iter().map(expr_to_string).collect())
      .collect();

    let common_edges: Vec<Expr> = first_edges
      .iter()
      .filter(|e| {
        let es = expr_to_string(e);
        edge_sets.iter().all(|set| set.contains(&es))
      })
      .cloned()
      .collect();

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![
        Expr::List(all_vertices.into()),
        Expr::List(common_edges.into()),
      ]
      .into(),
    });
  }

  // EdgeQ[graph, edge] — True if edge exists in graph
  if name == "EdgeQ"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let Expr::List(edges) = &gargs[1]
  {
    let edge_str = expr_to_string(&args[1]);
    let found = edges.iter().any(|e| expr_to_string(e) == edge_str);
    return Ok(bool_expr(found));
  }

  // EdgeRules[graph] — the edges as a list of rules (undirected edges
  // become rules too), in edge-list order. Non-graphs emit ::graph.
  if name == "EdgeRules" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(edges) = &gargs[1]
    {
      let mut rules: Vec<Expr> = Vec::with_capacity(edges.len());
      for edge in edges.iter() {
        let (from, to) = match edge {
          Expr::FunctionCall { name, args: eargs }
            if (name == "DirectedEdge" || name == "UndirectedEdge")
              && eargs.len() == 2 =>
          {
            (eargs[0].clone(), eargs[1].clone())
          }
          Expr::Rule {
            pattern,
            replacement,
          } => ((**pattern).clone(), (**replacement).clone()),
          _ => {
            return Ok(unevaluated("EdgeRules", args));
          }
        };
        rules.push(Expr::Rule {
          pattern: Box::new(from),
          replacement: Box::new(to),
        });
      }
      return Ok(Expr::List(rules.into()));
    }
    crate::emit_message(&format!(
      "EdgeRules::graph: A graph object is expected at position 1 in EdgeRules[{}].",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("EdgeRules", args));
  }

  // VertexQ[graph, vertex] — True if vertex is in graph. Anything that is
  // not a Graph in the first slot gives False (no message), like
  // wolframscript.
  if name == "VertexQ" && args.len() == 2 {
    let found = if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(vertices) = &gargs[0]
    {
      let vertex_str = expr_to_string(&args[1]);
      vertices.iter().any(|v| expr_to_string(v) == vertex_str)
    } else {
      false
    };
    return Ok(bool_expr(found));
  }

  // GraphQ[expr] — True iff expr is a valid Graph object
  if name == "GraphQ" && args.len() == 1 {
    // A valid edge is a 2-argument UndirectedEdge / DirectedEdge / Rule.
    fn is_valid_edge(e: &Expr) -> bool {
      match e {
        Expr::FunctionCall { name, args }
          if (name == "UndirectedEdge"
            || name == "DirectedEdge"
            || name == "Rule")
            && args.len() == 2 =>
        {
          true
        }
        // `a -> b` (parsed as a Rule node) denotes a directed edge.
        Expr::Rule { .. } => true,
        _ => false,
      }
    }
    let is_graph = if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    {
      gname == "Graph"
        && match gargs.as_slice() {
          // Graph[edges] — single list of valid edges
          [Expr::List(edges)] => edges.iter().all(is_valid_edge),
          // Graph[vertices, edges, ...] — vertices and edges lists,
          // every edge must be a valid edge expression
          [Expr::List(_vertices), Expr::List(edges), ..] => {
            edges.iter().all(is_valid_edge)
          }
          _ => false,
        }
    } else {
      false
    };
    return Ok(bool_expr(is_graph));
  }

  // Graph analysis helper: extract adjacency list from graph
  // UndirectedGraphQ[graph] — True if all edges are undirected
  if name == "UndirectedGraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(edges) = &gargs[1]
    {
      let all_undirected = edges.iter().all(|e| {
        matches!(e, Expr::FunctionCall { name, .. } if name == "UndirectedEdge")
      });
      return Ok(bool_expr(all_undirected));
    }
    return Ok(bool_expr(false));
  }

  // DirectedGraphQ[graph] — True if all edges are directed
  if name == "DirectedGraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(edges) = &gargs[1]
    {
      let all_directed = edges.iter().all(|e| {
        matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedEdge")
      });
      return Ok(bool_expr(all_directed));
    }
    return Ok(bool_expr(false));
  }

  // MixedGraphQ[graph] — True if the graph has both directed and undirected
  // edges.
  if name == "MixedGraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(edges) = &gargs[1]
    {
      let has_directed = edges.iter().any(|e| {
        matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedEdge")
      });
      let has_undirected = edges.iter().any(|e| {
        matches!(e, Expr::FunctionCall { name, .. } if name == "UndirectedEdge")
      });
      return Ok(Expr::Identifier(
        if has_directed && has_undirected {
          "True"
        } else {
          "False"
        }
        .to_string(),
      ));
    }
    return Ok(bool_expr(false));
  }

  // MultigraphQ[graph] — True if the graph has parallel edges (two edges with
  // the same endpoints and orientation, including repeated self-loops).
  // Directed u->v and v->u are distinct; undirected u<->v and v<->u are not.
  if name == "MultigraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(edges) = &gargs[1]
    {
      let mut seen: std::collections::HashSet<String> =
        std::collections::HashSet::new();
      let mut multi = false;
      for e in edges.iter() {
        if let Expr::FunctionCall { name: en, args: ea } = e
          && ea.len() == 2
        {
          let a = expr_to_string(&ea[0]);
          let b = expr_to_string(&ea[1]);
          let key = match en.as_str() {
            "DirectedEdge" => format!("D\0{a}\0{b}"),
            "UndirectedEdge" => {
              let (x, y) = if a <= b { (&a, &b) } else { (&b, &a) };
              format!("U\0{x}\0{y}")
            }
            _ => continue,
          };
          if !seen.insert(key) {
            multi = true;
            break;
          }
        }
      }
      return Ok(bool_expr(multi));
    }
    return Ok(bool_expr(false));
  }

  // WeightedGraphQ[graph] — True if the graph carries explicit edge or vertex
  // weights (an EdgeWeight/VertexWeight option set to something other than
  // Automatic).
  if name == "WeightedGraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
    {
      let has_weight = gargs.iter().any(|a| match a {
        Expr::Rule {
          pattern,
          replacement,
        } if matches!(pattern.as_ref(),
          Expr::Identifier(s) if s == "EdgeWeight" || s == "VertexWeight") =>
        {
          !matches!(replacement.as_ref(),
            Expr::Identifier(s) if s == "Automatic")
        }
        _ => false,
      });
      return Ok(bool_expr(has_weight));
    }
    return Ok(bool_expr(false));
  }

  // TreeGraphQ[graph] — True if graph is a tree (connected, n-1 edges for n vertices)
  if name == "TreeGraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      let n = vertices.len();
      // A tree has exactly n-1 edges and is connected
      if edges.len() != n - 1 {
        return Ok(bool_expr(false));
      }
      // Check connectivity via BFS/DFS
      let (pg_graph, _pg_idx) = build_undirected_graph(vertices, edges);
      let connected = is_connected_pg(&pg_graph);
      return Ok(bool_expr(connected));
    }
    return Ok(bool_expr(false));
  }

  // AcyclicGraphQ[graph] — True if graph has no cycles
  if name == "AcyclicGraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      let n = vertices.len();
      // For undirected: acyclic iff edges < n and connected components have tree structure
      // Simple check: edges < n (forest)
      let is_acyclic = edges.len() < n;
      return Ok(bool_expr(is_acyclic));
    }
    return Ok(bool_expr(false));
  }

  // EulerianGraphQ[graph] — True if graph has an Eulerian circuit
  if name == "EulerianGraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      let n = vertices.len();
      if n == 0 || edges.is_empty() {
        return Ok(bool_expr(true));
      }

      let is_directed = edges.iter().any(|e| {
        matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedEdge")
      });

      if is_directed {
        // Directed: connected + every vertex has equal in-degree and out-degree
        let vertex_index: std::collections::HashMap<String, usize> = vertices
          .iter()
          .enumerate()
          .map(|(i, v)| (expr_to_string(v), i))
          .collect();
        let mut in_deg = vec![0usize; n];
        let mut out_deg = vec![0usize; n];
        for edge in edges {
          if let Expr::FunctionCall { args: eargs, .. } = edge
            && eargs.len() == 2
          {
            let from_str = expr_to_string(&eargs[0]);
            let to_str = expr_to_string(&eargs[1]);
            if let (Some(&fi), Some(&ti)) =
              (vertex_index.get(&from_str), vertex_index.get(&to_str))
            {
              out_deg[fi] += 1;
              in_deg[ti] += 1;
            }
          }
        }
        let balanced = (0..n).all(|i| in_deg[i] == out_deg[i]);
        let connected = is_connected_pg(&{
          let (g, _) = build_undirected_graph(vertices, edges);
          g
        });
        return Ok(Expr::Identifier(
          if balanced && connected {
            "True"
          } else {
            "False"
          }
          .to_string(),
        ));
      } else {
        // Undirected: connected + all vertices have even degree
        let (pg_graph, _pg_idx) = build_undirected_graph(vertices, edges);
        let all_even = pg_graph
          .node_indices()
          .all(|ni| pg_graph.neighbors(ni).count() % 2 == 0);
        let connected = is_connected_pg(&pg_graph);
        return Ok(Expr::Identifier(
          if all_even && connected {
            "True"
          } else {
            "False"
          }
          .to_string(),
        ));
      }
    }
    return Ok(bool_expr(false));
  }

  // GraphDistance[g, s, t] — length of the shortest path from s to t.
  // GraphDistance[g, s] — list of distances from s to every vertex,
  //   ordered as in VertexList[g].
  // Unreachable vertices yield Infinity. Directed edges are honoured;
  // undirected edges are traversable in both directions. If any vertex
  // argument is not a vertex of the graph, the call stays unevaluated.
  if name == "GraphDistance"
    && (args.len() == 2 || args.len() == 3)
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    let vertex_strs: Vec<String> =
      vertices.iter().map(expr_to_string).collect();
    let index_of = |v: &Expr| -> Option<usize> {
      let s = expr_to_string(v);
      vertex_strs.iter().position(|x| x == &s)
    };

    // Build a directed adjacency list. Undirected edges (UndirectedEdge or
    // a 2-arg edge that is not DirectedEdge/Rule) go both ways; directed
    // edges (DirectedEdge or Rule) go one way only.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in edges.iter() {
      let (directed, src, dst) = match edge {
        Expr::FunctionCall {
          name: ename,
          args: eargs,
        } if eargs.len() == 2 => {
          (ename == "DirectedEdge", &eargs[0], &eargs[1])
        }
        Expr::Rule {
          pattern,
          replacement,
        } => (true, pattern.as_ref(), replacement.as_ref()),
        _ => continue,
      };
      if let (Some(si), Some(di)) = (index_of(src), index_of(dst)) {
        adj[si].push(di);
        if !directed {
          adj[di].push(si);
        }
      }
    }

    // Graphs carrying an EdgeWeight option use Dijkstra and return
    // machine reals, as wolframscript does ("7." for integer weights)
    let edge_weights: Option<Vec<f64>> = gargs[2..].iter().find_map(|g| {
      if let Expr::Rule {
        pattern,
        replacement,
      } = g
        && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "EdgeWeight")
        && let Expr::List(ws) = replacement.as_ref()
        && ws.len() == edges.len()
      {
        ws.iter().map(crate::functions::try_eval_to_f64).collect()
      } else {
        None
      }
    });
    if let Some(weights) = &edge_weights {
      let mut wadj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
      for (edge, &w) in edges.iter().zip(weights.iter()) {
        let (directed, src, dst) = match edge {
          Expr::FunctionCall {
            name: ename,
            args: eargs,
          } if eargs.len() == 2 => {
            (ename == "DirectedEdge", &eargs[0], &eargs[1])
          }
          Expr::Rule {
            pattern,
            replacement,
          } => (true, pattern.as_ref(), replacement.as_ref()),
          _ => continue,
        };
        if let (Some(si), Some(di)) = (index_of(src), index_of(dst)) {
          wadj[si].push((di, w));
          if !directed {
            wadj[di].push((si, w));
          }
        }
      }
      // O(n^2) Dijkstra (graphs here are small)
      let dijkstra = |start: usize| -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; n];
        let mut done = vec![false; n];
        dist[start] = 0.0;
        for _ in 0..n {
          let mut u = usize::MAX;
          let mut best = f64::INFINITY;
          for i in 0..n {
            if !done[i] && dist[i] < best {
              best = dist[i];
              u = i;
            }
          }
          if u == usize::MAX {
            break;
          }
          done[u] = true;
          for &(v, w) in &wadj[u] {
            if dist[u] + w < dist[v] {
              dist[v] = dist[u] + w;
            }
          }
        }
        dist
      };
      let wdist_to_expr = |d: f64| -> Expr {
        if d.is_infinite() {
          Expr::Identifier("Infinity".to_string())
        } else {
          Expr::Real(d)
        }
      };
      if let Some(s) = index_of(&args[1]) {
        let dist = dijkstra(s);
        if args.len() == 3 {
          if let Some(t) = index_of(&args[2]) {
            return Ok(wdist_to_expr(dist[t]));
          }
        } else {
          return Ok(Expr::List(dist.into_iter().map(wdist_to_expr).collect()));
        }
      }
      return Ok(unevaluated(name, args));
    }

    // BFS over unit-weight edges from `start`, returning distances
    // (-1 == unreachable).
    let bfs = |start: usize| -> Vec<i128> {
      let mut dist = vec![-1i128; n];
      let mut queue = std::collections::VecDeque::new();
      dist[start] = 0;
      queue.push_back(start);
      while let Some(u) = queue.pop_front() {
        for &w in &adj[u] {
          if dist[w] == -1 {
            dist[w] = dist[u] + 1;
            queue.push_back(w);
          }
        }
      }
      dist
    };

    let dist_to_expr = |d: i128| -> Expr {
      if d < 0 {
        Expr::Identifier("Infinity".to_string())
      } else {
        Expr::Integer(d)
      }
    };

    let start_idx = index_of(&args[1]);
    if args.len() == 3 {
      let target_idx = index_of(&args[2]);
      if let (Some(s), Some(t)) = (start_idx, target_idx) {
        let dist = bfs(s);
        return Ok(dist_to_expr(dist[t]));
      }
    } else if let Some(s) = start_idx {
      let dist = bfs(s);
      return Ok(Expr::List(dist.into_iter().map(dist_to_expr).collect()));
    }
    // Fall through to unevaluated when a vertex argument is invalid.
  }

  // GraphDistanceMatrix[g] — matrix whose (i, j) entry is the shortest-path
  // distance from vertex i to vertex j (both ordered as in VertexList[g]).
  // Unreachable pairs yield Infinity; the diagonal is 0. Directed edges are
  // honoured; undirected edges are traversable in both directions. An empty
  // graph stays unevaluated (matching wolframscript).
  if name == "GraphDistanceMatrix"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    && !vertices.is_empty()
  {
    let n = vertices.len();
    let vertex_strs: Vec<String> =
      vertices.iter().map(expr_to_string).collect();
    let index_of = |v: &Expr| -> Option<usize> {
      let s = expr_to_string(v);
      vertex_strs.iter().position(|x| x == &s)
    };

    // Build a directed adjacency list. Undirected edges go both ways;
    // directed edges (DirectedEdge or Rule) go one way only.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in edges.iter() {
      let (directed, src, dst) = match edge {
        Expr::FunctionCall {
          name: ename,
          args: eargs,
        } if eargs.len() == 2 => {
          (ename == "DirectedEdge", &eargs[0], &eargs[1])
        }
        Expr::Rule {
          pattern,
          replacement,
        } => (true, pattern.as_ref(), replacement.as_ref()),
        _ => continue,
      };
      if let (Some(si), Some(di)) = (index_of(src), index_of(dst)) {
        adj[si].push(di);
        if !directed {
          adj[di].push(si);
        }
      }
    }

    // BFS over unit-weight edges from `start` (-1 == unreachable).
    let bfs = |start: usize| -> Vec<i128> {
      let mut dist = vec![-1i128; n];
      let mut queue = std::collections::VecDeque::new();
      dist[start] = 0;
      queue.push_back(start);
      while let Some(u) = queue.pop_front() {
        for &w in &adj[u] {
          if dist[w] == -1 {
            dist[w] = dist[u] + 1;
            queue.push_back(w);
          }
        }
      }
      dist
    };

    let dist_to_expr = |d: i128| -> Expr {
      if d < 0 {
        Expr::Identifier("Infinity".to_string())
      } else {
        Expr::Integer(d)
      }
    };

    let rows = (0..n)
      .map(|s| Expr::List(bfs(s).into_iter().map(dist_to_expr).collect()))
      .collect();
    return Ok(Expr::List(rows));
  }

  // GraphDiameter, VertexEccentricity, GraphCenter, GraphPeriphery, GraphRadius
  if name == "GraphAssortativity" && args.len() == 1 {
    return crate::functions::graph::graph_assortativity_ast(args);
  }

  if matches!(name, "AdjacencyList" | "IncidenceList" | "EdgeIndex")
    && (args.len() == 1 || args.len() == 2)
  {
    return crate::functions::graph::graph_accessor_ast(name, args);
  }

  if name == "HermiteDecomposition" && args.len() == 1 {
    return crate::functions::linear_algebra_ast::hermite_decomposition_ast(
      args,
    );
  }

  if matches!(
    name,
    "GlobalClusteringCoefficient"
      | "MeanClusteringCoefficient"
      | "GraphDensity"
      | "MeanGraphDistance"
      | "MeanDegreeConnectivity"
      | "GraphLinkEfficiency"
  ) && args.len() == 1
  {
    return crate::functions::graph::graph_metric_ast(name, args);
  }

  if name == "PlanarGraphQ" && args.len() == 1 {
    return crate::functions::graph::planar_graph_q_ast(args);
  }

  if matches!(
    name,
    "HamiltonianGraphQ"
      | "BipartiteGraphQ"
      | "CompleteGraphQ"
      | "LoopFreeGraphQ"
      | "PathGraphQ"
      | "EmptyGraphQ"
      | "SimpleGraphQ"
  ) && args.len() == 1
  {
    return crate::functions::graph::graph_predicate_ast(name, args);
  }

  if name == "NeighborhoodGraph" && (args.len() == 2 || args.len() == 3) {
    return crate::functions::graph::neighborhood_graph_ast(args);
  }

  if name == "Subgraph" && args.len() == 2 {
    return crate::functions::graph::subgraph_ast(args);
  }

  if name == "LineGraph" && args.len() == 1 {
    return crate::functions::graph::line_graph_ast(args);
  }

  if name == "FindClique" && (1..=3).contains(&args.len()) {
    return crate::functions::graph::find_clique_ast(args);
  }

  if name == "KCoreComponents" && (args.len() == 2 || args.len() == 3) {
    return crate::functions::graph::k_core_components_ast(args);
  }

  if (name == "EdgeConnectivity" || name == "VertexConnectivity")
    && (args.len() == 1 || args.len() == 3)
  {
    return crate::functions::graph::connectivity_ast(name, args);
  }

  if (name == "GraphDiameter"
    || name == "VertexEccentricity"
    || name == "GraphCenter"
    || name == "GraphPeriphery"
    || name == "GraphRadius"
    || name == "EccentricityCentrality")
    && (args.len() == 1 || (name == "VertexEccentricity" && args.len() == 2))
  {
    let graph = if name == "VertexEccentricity" && args.len() == 2 {
      &args[0]
    } else {
      &args[0]
    };
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = graph
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      let n = vertices.len();
      let key = expr_to_string;
      let index: std::collections::HashMap<String, usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (key(v), i))
        .collect();
      // Directed adjacency: undirected edges point both ways.
      let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
      for edge in edges.iter() {
        if let Expr::FunctionCall { name: en, args: ea } = edge
          && ea.len() == 2
          && let (Some(&u), Some(&v)) =
            (index.get(&key(&ea[0])), index.get(&key(&ea[1])))
        {
          adj[u].push(v);
          if en == "UndirectedEdge" {
            adj[v].push(u);
          }
        }
      }
      // Shortest-path distances from every source (BFS, -1 = unreachable).
      let all_dists: Vec<Vec<i128>> = (0..n)
        .map(|s| {
          let mut dist = vec![-1i128; n];
          dist[s] = 0;
          let mut q = std::collections::VecDeque::from([s]);
          while let Some(u) = q.pop_front() {
            for &w in &adj[u] {
              if dist[w] < 0 {
                dist[w] = dist[u] + 1;
                q.push_back(w);
              }
            }
          }
          dist
        })
        .collect();
      // Eccentricity = greatest distance to a reachable vertex (0 for a sink).
      let ecc: Vec<i128> = all_dists
        .iter()
        .map(|d| d.iter().filter(|&&x| x >= 0).copied().max().unwrap_or(0))
        .collect();
      // Radius/diameter/centre/periphery are only finite/non-empty when the
      // graph is strongly connected (every vertex reaches every other).
      let strongly_connected =
        all_dists.iter().all(|d| d.iter().all(|&x| x >= 0));
      let infinity = || Expr::Identifier("Infinity".to_string());

      match name {
        "GraphDiameter" => {
          return Ok(if strongly_connected {
            Expr::Integer(ecc.iter().copied().max().unwrap_or(0))
          } else {
            infinity()
          });
        }
        "GraphRadius" => {
          return Ok(if strongly_connected {
            Expr::Integer(ecc.iter().copied().min().unwrap_or(0))
          } else {
            infinity()
          });
        }
        "VertexEccentricity" if args.len() == 2 => {
          let target = key(&args[1]);
          if let Some(&idx) = index.get(&target) {
            return Ok(Expr::Integer(ecc[idx]));
          }
        }
        "GraphCenter" => {
          if !strongly_connected {
            return Ok(Expr::List(vec![].into()));
          }
          let min_ecc = ecc.iter().copied().min().unwrap_or(0);
          let center: Vec<Expr> = vertices
            .iter()
            .enumerate()
            .filter(|(i, _)| ecc[*i] == min_ecc)
            .map(|(_, v)| v.clone())
            .collect();
          return Ok(Expr::List(center.into()));
        }
        "GraphPeriphery" => {
          if !strongly_connected {
            return Ok(Expr::List(vec![].into()));
          }
          let max_ecc = ecc.iter().copied().max().unwrap_or(0);
          let periphery: Vec<Expr> = vertices
            .iter()
            .enumerate()
            .filter(|(i, _)| ecc[*i] == max_ecc)
            .map(|(_, v)| v.clone())
            .collect();
          return Ok(Expr::List(periphery.into()));
        }
        "EccentricityCentrality" => {
          // 1 / eccentricity as a machine real (0 for a sink), vertex order.
          let result: Vec<Expr> = ecc
            .iter()
            .map(|&e| Expr::Real(if e == 0 { 0.0 } else { 1.0 / e as f64 }))
            .collect();
          return Ok(Expr::List(result.into()));
        }
        _ => {}
      }
    }
  }

  // DegreeCentrality[graph] — degree centrality of each vertex
  if name == "DegreeCentrality"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let (pg_graph, _) = build_undirected_graph(vertices, edges);
    // DegreeCentrality returns raw vertex degree (not normalized)
    let centralities: Vec<Expr> = pg_graph
      .node_indices()
      .map(|ni| Expr::Integer(pg_graph.neighbors(ni).count() as i128))
      .collect();
    return Ok(Expr::List(centralities.into()));
  }

  // GraphComplement[graph] — complement of a graph
  if name == "GraphComplement"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    // Build set of existing edges
    let vertex_strs: Vec<String> =
      vertices.iter().map(expr_to_string).collect();
    let mut edge_set = std::collections::HashSet::new();
    for edge in edges {
      if let Expr::FunctionCall { args: eargs, .. } = edge
        && eargs.len() == 2
      {
        let a = expr_to_string(&eargs[0]);
        let b = expr_to_string(&eargs[1]);
        edge_set.insert((a.clone(), b.clone()));
        edge_set.insert((b, a));
      }
    }
    // Build complement edges
    let mut comp_edges = Vec::new();
    for i in 0..n {
      for j in (i + 1)..n {
        if !edge_set.contains(&(vertex_strs[i].clone(), vertex_strs[j].clone()))
        {
          comp_edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![vertices[i].clone(), vertices[j].clone()].into(),
          });
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.clone()), Expr::List(comp_edges.into())]
        .into(),
    });
  }

  // GraphUnion[g1, g2, ...] — graph whose vertices are the (sorted) union of
  // the inputs' vertices and whose edges are the union of their edges, kept in
  // first-seen order and deduplicated by undirected endpoint pair.
  if name == "GraphUnion"
    && args.len() >= 2
    && args.iter().all(|a| {
      matches!(a, Expr::FunctionCall { name: gn, args: ga }
        if gn == "Graph" && ga.len() >= 2
          && matches!(&ga[0], Expr::List(_))
          && matches!(&ga[1], Expr::List(_)))
    })
  {
    let mut vertices: Vec<Expr> = Vec::new();
    let mut vertex_seen = std::collections::HashSet::new();
    let mut edges: Vec<Expr> = Vec::new();
    let mut edge_seen = std::collections::HashSet::new();
    for g in args.iter() {
      let Expr::FunctionCall { args: ga, .. } = g else {
        unreachable!()
      };
      if let Expr::List(vs) = &ga[0] {
        for v in vs.iter() {
          if vertex_seen.insert(expr_to_string(v)) {
            vertices.push(v.clone());
          }
        }
      }
      if let Expr::List(es) = &ga[1] {
        for edge in es.iter() {
          if let Expr::FunctionCall { args: ea, .. } = edge
            && ea.len() == 2
          {
            let (sa, sb) = (expr_to_string(&ea[0]), expr_to_string(&ea[1]));
            // Undirected: normalize the endpoint pair for deduplication.
            let key = if sa <= sb {
              (sa.clone(), sb.clone())
            } else {
              (sb.clone(), sa.clone())
            };
            if edge_seen.insert(key) {
              edges.push(edge.clone());
            }
          }
        }
      }
    }
    vertices.sort_by(crate::functions::list_helpers_ast::canonical_cmp);
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // GraphDifference[g1, g2] — g1 with the edges of g2 removed. Vertices and
  // surviving edges keep g1's order; edge matching is by undirected pair.
  if name == "GraphDifference"
    && args.len() == 2
    && let (
      Expr::FunctionCall { name: n1, args: a1 },
      Expr::FunctionCall { name: n2, args: a2 },
    ) = (&args[0], &args[1])
    && n1 == "Graph"
    && n2 == "Graph"
    && a1.len() >= 2
    && a2.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges1)) = (&a1[0], &a1[1])
    && let Expr::List(edges2) = &a2[1]
  {
    let edge_key = |edge: &Expr| -> Option<(String, String)> {
      if let Expr::FunctionCall { args: ea, .. } = edge
        && ea.len() == 2
      {
        let (sa, sb) = (expr_to_string(&ea[0]), expr_to_string(&ea[1]));
        Some(if sa <= sb { (sa, sb) } else { (sb, sa) })
      } else {
        None
      }
    };
    let remove: std::collections::HashSet<(String, String)> =
      edges2.iter().filter_map(edge_key).collect();
    let kept: Vec<Expr> = edges1
      .iter()
      .filter(|e| edge_key(e).is_none_or(|k| !remove.contains(&k)))
      .cloned()
      .collect();
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.clone()), Expr::List(kept.into())].into(),
    });
  }

  // GraphPower[graph, k] — connect every pair of distinct vertices whose
  // graph distance is at most k. Edges are listed in vertex-index order.
  if name == "GraphPower"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    && let Some(k) = expr_to_i128(&args[1])
    && k >= 1
  {
    let n = vertices.len();
    let vertex_strs: Vec<String> =
      vertices.iter().map(expr_to_string).collect();
    // Index-based adjacency from the edge list.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in edges {
      if let Expr::FunctionCall { args: eargs, .. } = edge
        && eargs.len() == 2
        && let (Some(a), Some(b)) = (
          vertex_strs
            .iter()
            .position(|v| v == &expr_to_string(&eargs[0])),
          vertex_strs
            .iter()
            .position(|v| v == &expr_to_string(&eargs[1])),
        )
      {
        adj[a].push(b);
        adj[b].push(a);
      }
    }
    // BFS distance from each source, then connect pairs within distance k.
    let mut power_edges = Vec::new();
    for i in 0..n {
      let mut dist = vec![usize::MAX; n];
      dist[i] = 0;
      let mut queue = std::collections::VecDeque::from([i]);
      while let Some(u) = queue.pop_front() {
        for &w in &adj[u] {
          if dist[w] == usize::MAX {
            dist[w] = dist[u] + 1;
            queue.push_back(w);
          }
        }
      }
      for j in (i + 1)..n {
        if dist[j] != usize::MAX && (dist[j] as i128) <= k {
          power_edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![vertices[i].clone(), vertices[j].clone()].into(),
          });
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.clone()), Expr::List(power_edges.into())]
        .into(),
    });
  }

  // VertexOutComponent[graph, v(, k)] — vertices reachable from v (following
  // directed edges forwards). VertexInComponent[graph, v(, k)] — vertices
  // that can reach v (following them backwards).
  if (name == "VertexOutComponent" || name == "VertexInComponent")
    && (args.len() == 2 || args.len() == 3)
  {
    return crate::functions::graph::vertex_reach_component_ast(
      name,
      args,
      name == "VertexOutComponent",
    );
  }

  // ClosenessCentrality[graph] — closeness centrality for each vertex
  if name == "ClosenessCentrality"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    // Closeness centrality of v = (number of vertices reachable from v) /
    // (sum of directed distances to them); 0 when nothing is reachable. For a
    // connected undirected graph this reduces to (n-1) / total distance.
    let all_dists = graph_directed_distances(vertices, edges);
    let centralities: Vec<Expr> = all_dists
      .iter()
      .map(|dists| {
        let reachable: Vec<i128> =
          dists.iter().filter(|&&d| d > 0).copied().collect();
        let count = reachable.len();
        let sum: i128 = reachable.iter().sum();
        if count > 0 {
          Expr::Real(count as f64 / sum as f64)
        } else {
          Expr::Real(0.0)
        }
      })
      .collect();
    return Ok(Expr::List(centralities.into()));
  }

  // RadialityCentrality[graph] — for each vertex v,
  //   (sum over reachable w of (D + 1 - d(v, w))) / ((n - 1) * D)
  // where D is the largest finite directed shortest-path distance. Distances
  // honour edge direction (undirected edges point both ways).
  if name == "RadialityCentrality"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    let all_dists = graph_directed_distances(vertices, edges);
    // Diameter = largest finite (positive) shortest-path distance.
    let diameter = all_dists
      .iter()
      .flat_map(|d| d.iter().copied())
      .filter(|&d| d > 0)
      .max()
      .unwrap_or(0);
    let result: Vec<Expr> = if n <= 1 || diameter == 0 {
      (0..n).map(|_| Expr::Real(0.0)).collect()
    } else {
      all_dists
        .iter()
        .map(|dists| {
          let sum: i128 = dists
            .iter()
            .filter(|&&d| d > 0)
            .map(|&d| diameter + 1 - d)
            .sum();
          Expr::Real(sum as f64 / ((n as f64 - 1.0) * diameter as f64))
        })
        .collect()
    };
    return Ok(Expr::List(result.into()));
  }

  // EigenvectorCentrality[graph] — principal eigenvector of the (undirected)
  // adjacency matrix, normalized so the entries sum to 1.
  if name == "EigenvectorCentrality"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    if n == 0 {
      return Ok(Expr::List(vec![].into()));
    }
    let mut idx: std::collections::HashMap<String, usize> =
      std::collections::HashMap::new();
    for (i, v) in vertices.iter().enumerate() {
      idx.insert(expr_to_string(v), i);
    }
    let is_directed = edges.iter().any(|e| {
      matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedEdge")
    });

    if is_directed {
      // Directed eigenvector centrality is the Perron eigenvector of the
      // in-adjacency (a vertex is central if central vertices point to it),
      // supported only on the dominant strongly-connected component. Vertices
      // outside it — and every vertex of an acyclic graph — get 0.
      let mut in_adj = vec![vec![0.0_f64; n]; n];
      let mut has_self = vec![false; n];
      for e in edges.iter() {
        if let Some((u, v, _)) = crate::functions::graph::edge_endpoints(e)
          && let (Some(&i), Some(&j)) =
            (idx.get(&expr_to_string(&u)), idx.get(&expr_to_string(&v)))
        {
          in_adj[j][i] += 1.0; // edge i -> j is an in-edge to j
          if i == j {
            has_self[i] = true;
          }
        }
      }
      // Strongly-connected components come from ConnectedComponents.
      let comps =
        evaluate_function_call_ast_inner("ConnectedComponents", args)?;
      let sccs: Vec<Vec<usize>> = match &comps {
        Expr::List(cs) => cs
          .iter()
          .filter_map(|c| match c {
            Expr::List(vs) => Some(
              vs.iter()
                .filter_map(|v| idx.get(&expr_to_string(v)).copied())
                .collect(),
            ),
            _ => None,
          })
          .collect(),
        _ => vec![],
      };
      // Perron eigenpair of each cyclic SCC; keep those of maximal eigenvalue.
      let mut best_lambda = 0.0_f64;
      let mut best: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
      for scc in &sccs {
        let cyclic = scc.len() > 1 || (scc.len() == 1 && has_self[scc[0]]);
        if !cyclic {
          continue;
        }
        let sub: Vec<Vec<f64>> = scc
          .iter()
          .map(|&ga| scc.iter().map(|&gb| in_adj[ga][gb]).collect())
          .collect();
        let (lambda, vec) = crate::functions::graph::perron_eigenpair(&sub);
        if lambda > best_lambda + 1e-9 {
          best_lambda = lambda;
          best = vec![(scc.clone(), vec)];
        } else if (lambda - best_lambda).abs() <= 1e-9 {
          best.push((scc.clone(), vec));
        }
      }
      let mut result = vec![0.0_f64; n];
      if best_lambda > 1e-9 {
        for (scc, vec) in &best {
          for (a, &g) in scc.iter().enumerate() {
            result[g] = vec[a];
          }
        }
        let total: f64 = result.iter().sum();
        if total != 0.0 {
          for x in result.iter_mut() {
            *x /= total;
          }
        }
      }
      return Ok(Expr::List(
        result
          .into_iter()
          .map(Expr::Real)
          .collect::<Vec<_>>()
          .into(),
      ));
    }

    let mut adj = vec![vec![0.0_f64; n]; n];
    for e in edges.iter() {
      if let Some((u, v, _directed)) =
        crate::functions::graph::edge_endpoints(e)
        && let (Some(&i), Some(&j)) =
          (idx.get(&expr_to_string(&u)), idx.get(&expr_to_string(&v)))
      {
        adj[i][j] += 1.0;
        if i != j {
          adj[j][i] += 1.0;
        }
      }
    }
    let centrality = crate::functions::graph::eigenvector_centrality(&adj);
    return Ok(Expr::List(
      centrality
        .into_iter()
        .map(Expr::Real)
        .collect::<Vec<_>>()
        .into(),
    ));
  }

  // KatzCentrality[graph, alpha] / KatzCentrality[graph, alpha, beta] —
  // solves (I - alpha A) x = beta * 1 (beta defaults to 1).
  if name == "KatzCentrality"
    && (args.len() == 2 || args.len() == 3)
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    && let Some(alpha) = crate::functions::math_ast::expr_to_f64(&args[1])
  {
    let n = vertices.len();
    // beta defaults to 1 for every vertex; a scalar third argument is used for
    // all vertices, and a length-n list gives a per-vertex bias vector.
    let beta: Vec<f64> = if args.len() == 3 {
      match &args[2] {
        Expr::List(bs) if bs.len() == n => {
          let vals: Option<Vec<f64>> = bs
            .iter()
            .map(crate::functions::math_ast::expr_to_f64)
            .collect();
          match vals {
            Some(v) => v,
            None => vec![1.0; n],
          }
        }
        other => {
          vec![crate::functions::math_ast::expr_to_f64(other).unwrap_or(1.0); n]
        }
      }
    } else {
      vec![1.0; n]
    };
    let mut idx: std::collections::HashMap<String, usize> =
      std::collections::HashMap::new();
    for (i, v) in vertices.iter().enumerate() {
      idx.insert(expr_to_string(v), i);
    }
    // Katz centrality solves x = alpha A^T x + beta, i.e. a vertex gains
    // centrality from its in-neighbours. Edge u -> v therefore contributes to
    // adj[v][u]; undirected edges contribute both ways.
    let mut adj = vec![vec![0.0_f64; n]; n];
    for e in edges.iter() {
      if let Some((u, v, directed)) = crate::functions::graph::edge_endpoints(e)
        && let (Some(&i), Some(&j)) =
          (idx.get(&expr_to_string(&u)), idx.get(&expr_to_string(&v)))
      {
        adj[j][i] += 1.0;
        if !directed && i != j {
          adj[i][j] += 1.0;
        }
      }
    }
    if let Some(centrality) =
      crate::functions::graph::katz_centrality(&adj, alpha, &beta)
    {
      return Ok(Expr::List(
        centrality
          .into_iter()
          .map(Expr::Real)
          .collect::<Vec<_>>()
          .into(),
      ));
    }
  }

  // PageRankCentrality[graph] / PageRankCentrality[graph, alpha] — PageRank
  // with damping factor alpha (default 0.85).
  if name == "PageRankCentrality"
    && (args.len() == 1 || args.len() == 2)
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let alpha = if args.len() == 2 {
      crate::functions::math_ast::expr_to_f64(&args[1]).unwrap_or(0.85)
    } else {
      0.85
    };
    let n = vertices.len();
    let mut idx: std::collections::HashMap<String, usize> =
      std::collections::HashMap::new();
    for (i, v) in vertices.iter().enumerate() {
      idx.insert(expr_to_string(v), i);
    }
    let mut adj = vec![vec![0.0_f64; n]; n];
    for e in edges.iter() {
      if let Some((u, v, directed)) = crate::functions::graph::edge_endpoints(e)
        && let (Some(&i), Some(&j)) =
          (idx.get(&expr_to_string(&u)), idx.get(&expr_to_string(&v)))
      {
        adj[i][j] += 1.0;
        // Only undirected edges contribute a reverse transition; directed
        // edges let PageRank flow one way.
        if !directed && i != j {
          adj[j][i] += 1.0;
        }
      }
    }
    if let Some(centrality) =
      crate::functions::graph::pagerank_centrality(&adj, alpha)
    {
      return Ok(Expr::List(
        centrality
          .into_iter()
          .map(Expr::Real)
          .collect::<Vec<_>>()
          .into(),
      ));
    }
  }

  // BetweennessCentrality[graph] — betweenness centrality for each vertex
  if name == "BetweennessCentrality"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    let (pg_graph, _pg_idx) = build_undirected_graph(vertices, edges);
    let mut betweenness = vec![0.0_f64; n];

    for s in 0..n {
      // BFS from s to compute shortest path counts and distances
      let mut dist = vec![-1i128; n];
      let mut sigma = vec![0.0_f64; n]; // number of shortest paths
      let mut pred: Vec<Vec<usize>> = vec![vec![]; n]; // predecessors
      let mut queue = std::collections::VecDeque::new();
      let mut stack = Vec::new();

      dist[s] = 0;
      sigma[s] = 1.0;
      queue.push_back(s);

      while let Some(v) = queue.pop_front() {
        stack.push(v);
        for w in pg_graph
          .neighbors(petgraph::graph::NodeIndex::new(v))
          .map(|ni| ni.index())
        {
          if dist[w] < 0 {
            dist[w] = dist[v] + 1;
            queue.push_back(w);
          }
          if dist[w] == dist[v] + 1 {
            sigma[w] += sigma[v];
            pred[w].push(v);
          }
        }
      }

      // Accumulate dependency
      let mut delta = vec![0.0_f64; n];
      while let Some(w) = stack.pop() {
        for &v in &pred[w] {
          delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
        }
        if w != s {
          betweenness[w] += delta[w];
        }
      }
    }

    // Normalize: divide by 2 for undirected graphs
    let centralities: Vec<Expr> =
      betweenness.iter().map(|&b| Expr::Real(b / 2.0)).collect();
    return Ok(Expr::List(centralities.into()));
  }

  // EdgeBetweennessCentrality[graph] — betweenness for each edge (in EdgeList
  // order): the number of shortest paths over all vertex pairs through it.
  if name == "EdgeBetweennessCentrality"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    let mut idx: std::collections::HashMap<String, usize> =
      std::collections::HashMap::new();
    for (i, v) in vertices.iter().enumerate() {
      idx.insert(expr_to_string(v), i);
    }
    let mut edge_pairs: Vec<(usize, usize)> = Vec::new();
    for e in edges.iter() {
      if let Some((u, v, _directed)) =
        crate::functions::graph::edge_endpoints(e)
        && let (Some(&i), Some(&j)) =
          (idx.get(&expr_to_string(&u)), idx.get(&expr_to_string(&v)))
      {
        edge_pairs.push((i, j));
      }
    }
    let eb =
      crate::functions::graph::edge_betweenness_centrality(n, &edge_pairs);
    return Ok(Expr::List(
      eb.into_iter().map(Expr::Real).collect::<Vec<_>>().into(),
    ));
  }

  // LocalClusteringCoefficient[graph] — local clustering coefficient for each vertex
  if name == "LocalClusteringCoefficient"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    // Directed graphs: fraction of in -> v -> out paths closed by a back arc.
    if let Some(local) =
      crate::functions::graph::directed_local_clustering(&args[0])
    {
      let coefficients: Vec<Expr> = local
        .into_iter()
        .map(|(closed, total)| {
          if total == 0 {
            Expr::Integer(0)
          } else {
            crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(closed), Expr::Integer(total)].into(),
            })
            .unwrap_or(Expr::Integer(0))
          }
        })
        .collect();
      return Ok(Expr::List(coefficients.into()));
    }
    let (pg_graph, _pg_idx) = build_undirected_graph(vertices, edges);
    // Build neighbor sets for quick lookup
    let neighbor_sets: Vec<std::collections::HashSet<usize>> = (0..n)
      .map(|v| {
        pg_graph
          .neighbors(petgraph::graph::NodeIndex::new(v))
          .map(|ni| ni.index())
          .collect()
      })
      .collect();

    let coefficients: Vec<Expr> = (0..n)
      .map(|v| {
        let k = pg_graph
          .neighbors(petgraph::graph::NodeIndex::new(v))
          .count();
        if k < 2 {
          return Expr::Integer(0);
        }
        let mut triangles = 0i128;
        let neighbors: Vec<usize> = pg_graph
          .neighbors(petgraph::graph::NodeIndex::new(v))
          .map(|ni| ni.index())
          .collect();
        for i in 0..neighbors.len() {
          for j in (i + 1)..neighbors.len() {
            if neighbor_sets[neighbors[i]].contains(&neighbors[j]) {
              triangles += 1;
            }
          }
        }
        let possible = (k * (k - 1) / 2) as i128;
        if triangles == possible {
          Expr::Integer(1)
        } else if triangles == 0 {
          Expr::Integer(0)
        } else {
          let g = gcd_i128(triangles, possible);
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![
              Expr::Integer(triangles / g),
              Expr::Integer(possible / g),
            ]
            .into(),
          }
        }
      })
      .collect();
    return Ok(Expr::List(coefficients.into()));
  }

  // ChromaticPolynomial[graph, k] — chromatic polynomial of a graph
  if name == "ChromaticPolynomial"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let k = &args[1];
    let n = vertices.len();

    // Build edge set as pairs of vertex indices
    let vertex_index: std::collections::HashMap<String, usize> = vertices
      .iter()
      .enumerate()
      .map(|(i, v)| (expr_to_string(v), i))
      .collect();
    let mut edge_pairs: Vec<(usize, usize)> = Vec::new();
    for edge in edges {
      if let Expr::FunctionCall { args: eargs, .. } = edge
        && eargs.len() == 2
      {
        let from_str = expr_to_string(&eargs[0]);
        let to_str = expr_to_string(&eargs[1]);
        if let (Some(&fi), Some(&ti)) =
          (vertex_index.get(&from_str), vertex_index.get(&to_str))
        {
          let (a, b) = if fi < ti { (fi, ti) } else { (ti, fi) };
          edge_pairs.push((a, b));
        }
      }
    }
    // Deduplicate edges
    edge_pairs.sort();
    edge_pairs.dedup();

    // Compute chromatic polynomial coefficients using deletion-contraction
    let coeffs = chromatic_poly_coeffs(n, &edge_pairs);

    // Build polynomial expression in k
    let poly = poly_to_expr(&coeffs, k);
    return Ok(poly);
  }

  // ButterflyGraph[n] — butterfly network graph with (n+1)*2^n vertices
  // Vertices are labeled 1..(n+1)*2^n, mapping from (level, index) where
  // level ∈ {0,...,n} and index ∈ {0,...,2^n-1}.
  // Vertex number = level * 2^n + index + 1
  // Edges: (level, index) -- (level+1, index) and (level, index) -- (level+1, index XOR 2^level)
  if name == "ButterflyGraph"
    && args.len() == 1
    && let Expr::Integer(n) = &args[0]
  {
    let n = *n as usize;
    let width = 1usize << n; // 2^n
    let total = (n + 1) * width;
    let vertices: Vec<Expr> =
      (1..=total).map(|i| Expr::Integer(i as i128)).collect();
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for level in 0..n {
      for idx in 0..width {
        let v1 = level * width + idx + 1; // current vertex
        // Straight edge: (level, idx) -- (level+1, idx)
        let v2 = (level + 1) * width + idx + 1;
        let (a, b) = if v1 < v2 { (v1, v2) } else { (v2, v1) };
        if seen.insert((a, b)) {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)]
              .into(),
          });
        }
        // Cross edge: (level, idx) -- (level+1, idx XOR 2^level)
        let cross_idx = idx ^ (1 << level);
        let v3 = (level + 1) * width + cross_idx + 1;
        let (a, b) = if v1 < v3 { (v1, v3) } else { (v3, v1) };
        if seen.insert((a, b)) {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)]
              .into(),
          });
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // KnightTourGraph[m, n] — graph of knight moves on an m×n chessboard
  if name == "KnightTourGraph"
    && args.len() == 2
    && let Expr::Integer(m) = &args[0]
    && let Expr::Integer(n) = &args[1]
  {
    let m = *m as usize;
    let n = *n as usize;
    let total = m * n;
    let vertices: Vec<Expr> =
      (1..=total).map(|i| Expr::Integer(i as i128)).collect();
    let knight_moves: [(i32, i32); 8] = [
      (1, 2),
      (1, -2),
      (-1, 2),
      (-1, -2),
      (2, 1),
      (2, -1),
      (-2, 1),
      (-2, -1),
    ];
    let mut edges = Vec::new();
    for r in 0..m {
      for c in 0..n {
        let from = r * n + c + 1; // 1-indexed
        for &(dr, dc) in &knight_moves {
          let nr = r as i32 + dr;
          let nc = c as i32 + dc;
          if nr >= 0 && nr < m as i32 && nc >= 0 && nc < n as i32 {
            let to = nr as usize * n + nc as usize + 1;
            if from < to {
              edges.push(Expr::FunctionCall {
                name: "UndirectedEdge".to_string(),
                args: vec![
                  Expr::Integer(from as i128),
                  Expr::Integer(to as i128),
                ]
                .into(),
              });
            }
          }
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.into()), Expr::List(edges.into())].into(),
    });
  }

  // AdjacencyMatrix[Graph[{vertices}, {edges}]] — build adjacency matrix
  if name == "AdjacencyMatrix" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      let n = vertices.len();
      // Build vertex-to-index mapping
      let vertex_index: std::collections::HashMap<String, usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (expr_to_string(v), i))
        .collect();
      // Initialize n×n zero matrix
      let mut matrix = vec![vec![Expr::Integer(0); n]; n];
      for edge in edges {
        if let Expr::FunctionCall {
          name: ename,
          args: eargs,
        } = edge
          && eargs.len() == 2
        {
          let from_str = expr_to_string(&eargs[0]);
          let to_str = expr_to_string(&eargs[1]);
          if let (Some(&fi), Some(&ti)) =
            (vertex_index.get(&from_str), vertex_index.get(&to_str))
          {
            matrix[fi][ti] = Expr::Integer(1);
            if ename == "UndirectedEdge" {
              matrix[ti][fi] = Expr::Integer(1);
            }
          }
        }
      }
      return Ok(Expr::List(
        matrix.into_iter().map(|v| Expr::List(v.into())).collect(),
      ));
    }
    return Ok(unevaluated(name, args));
  }

  // WeightedAdjacencyMatrix[Graph[{vertices}, {edges}, EdgeWeight -> {w...}]]
  // — n×n matrix whose (i,j) entry is the sum of the weights of all edges
  // from vertex i to vertex j (0 when no edge connects them). Directed edges
  // (Rule u->v or DirectedEdge[u,v]) contribute only to entry (u,v);
  // undirected edges (UndirectedEdge[u,v]) contribute to both (u,v) and (v,u),
  // except self-loops which contribute once. When EdgeWeight is absent (or has
  // fewer entries than edges) the missing weights default to 1, matching
  // wolframscript. Weights may be numbers or symbolic expressions; parallel
  // edges have their weights summed. wolframscript returns a SparseArray here;
  // Woxi mirrors AdjacencyMatrix and returns the dense (Normal) list of lists.
  if name == "WeightedAdjacencyMatrix" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      let n = vertices.len();
      let vertex_index: std::collections::HashMap<String, usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (expr_to_string(v), i))
        .collect();

      // Extract the EdgeWeight option from the Graph options (gargs[2..]).
      let mut weights: Vec<Expr> = Vec::new();
      for opt in &gargs[2..] {
        if let Expr::Rule {
          pattern,
          replacement,
        } = opt
          && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "EdgeWeight")
          && let Expr::List(ws) = replacement.as_ref()
        {
          weights = ws.to_vec();
        }
      }

      // Accumulate the list of weight expressions contributed to each cell.
      let mut cells: Vec<Vec<Vec<Expr>>> = vec![vec![Vec::new(); n]; n];
      for (ei, edge) in edges.iter().enumerate() {
        // Default weight is 1 when EdgeWeight is absent or too short.
        let w = weights.get(ei).cloned().unwrap_or(Expr::Integer(1));
        // Determine (from, to, directed?) for both edge spellings.
        let endpoints = match edge {
          Expr::FunctionCall {
            name: ename,
            args: eargs,
          } if eargs.len() == 2 => Some((
            expr_to_string(&eargs[0]),
            expr_to_string(&eargs[1]),
            ename == "DirectedEdge",
          )),
          Expr::Rule {
            pattern,
            replacement,
          } => Some((
            expr_to_string(pattern),
            expr_to_string(replacement),
            true, // Rules are directed
          )),
          _ => None,
        };
        if let Some((from_str, to_str, directed)) = endpoints
          && let (Some(&fi), Some(&ti)) =
            (vertex_index.get(&from_str), vertex_index.get(&to_str))
        {
          cells[fi][ti].push(w.clone());
          if !directed && fi != ti {
            cells[ti][fi].push(w);
          }
        }
      }

      // Collapse each cell's weight list into a single Expr: 0 when empty,
      // the lone weight when one edge, or a summed Plus[...] otherwise.
      let mut matrix: Vec<Expr> = Vec::with_capacity(n);
      for row in cells {
        let mut out_row: Vec<Expr> = Vec::with_capacity(n);
        for ws in row {
          let entry = match ws.len() {
            0 => Expr::Integer(0),
            1 => ws.into_iter().next().unwrap(),
            _ => {
              crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
                name: "Plus".to_string(),
                args: ws.into(),
              })?
            }
          };
          out_row.push(entry);
        }
        matrix.push(Expr::List(out_row.into()));
      }
      return Ok(Expr::List(matrix.into()));
    }
    return Ok(unevaluated(name, args));
  }

  // IncidenceMatrix[Graph[{vertices}, {edges}]] — vertex-by-edge incidence matrix.
  // Rows are vertices (in vertex order), columns are edges (in edge order).
  // Undirected edge {u,v}: both endpoints get +1 (a self-loop yields 2).
  // Directed edge u->v: source gets -1, target gets +1; a directed self-loop
  // yields -2 (matching Wolfram's convention).
  if name == "IncidenceMatrix" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      let n = vertices.len();
      let e = edges.len();
      let vertex_index: std::collections::HashMap<String, usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (expr_to_string(v), i))
        .collect();
      // n×e zero matrix (rows = vertices, cols = edges)
      let mut matrix = vec![vec![0i128; e]; n];
      for (col, edge) in edges.iter().enumerate() {
        // Extract (from, to, directed?) for both edge spellings:
        // DirectedEdge/UndirectedEdge[u,v] and the directed Rule u->v.
        let endpoints = match edge {
          Expr::FunctionCall {
            name: ename,
            args: eargs,
          } if eargs.len() == 2 => Some((
            expr_to_string(&eargs[0]),
            expr_to_string(&eargs[1]),
            ename == "DirectedEdge",
          )),
          Expr::Rule {
            pattern,
            replacement,
          } => Some((
            expr_to_string(pattern),
            expr_to_string(replacement),
            true, // Rules are directed
          )),
          _ => None,
        };
        if let Some((from_str, to_str, directed)) = endpoints
          && let (Some(&fi), Some(&ti)) =
            (vertex_index.get(&from_str), vertex_index.get(&to_str))
        {
          if directed {
            if fi == ti {
              // directed self-loop
              matrix[fi][col] = -2;
            } else {
              matrix[fi][col] = -1;
              matrix[ti][col] = 1;
            }
          } else {
            // undirected edge (a self-loop naturally yields 2)
            matrix[fi][col] += 1;
            matrix[ti][col] += 1;
          }
        }
      }
      return Ok(Expr::List(
        matrix
          .into_iter()
          .map(|row| Expr::List(row.into_iter().map(Expr::Integer).collect()))
          .collect(),
      ));
    }
    return Ok(unevaluated(name, args));
  }

  // KirchhoffMatrix[Graph[{vertices}, {edges}]] — the Laplacian (Kirchhoff)
  // matrix L = D - A, where D is the diagonal degree matrix (VertexDegree, i.e.
  // in-degree + out-degree for directed graphs) and A is the (directed)
  // adjacency matrix. For undirected edges {u,v} both endpoints contribute to
  // the degree and the adjacency entries A[u][v] and A[v][u]; for directed
  // edges u->v only the source's out-degree and A[u][v] are incremented while
  // the target still gains an in-degree count on the diagonal. wolframscript
  // returns a SparseArray here; Woxi mirrors AdjacencyMatrix and returns the
  // dense (Normal) list of lists.
  if name == "KirchhoffMatrix" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      let n = vertices.len();
      let vertex_index: std::collections::HashMap<String, usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (expr_to_string(v), i))
        .collect();
      // adjacency[i][j] = 1 if an edge connects i to j (symmetric for
      // undirected edges). Parallel edges are collapsed, mirroring Woxi's
      // AdjacencyMatrix and wolframscript's KirchhoffMatrix on multigraphs.
      let mut adjacency = vec![vec![0i128; n]; n];
      let mut degree = vec![0i128; n];
      for edge in edges {
        // Extract (from, to, directed?) for both edge spellings:
        // DirectedEdge/UndirectedEdge[u,v] and the directed Rule u->v.
        let endpoints = match edge {
          Expr::FunctionCall {
            name: ename,
            args: eargs,
          } if eargs.len() == 2 => Some((
            expr_to_string(&eargs[0]),
            expr_to_string(&eargs[1]),
            ename == "DirectedEdge",
          )),
          Expr::Rule {
            pattern,
            replacement,
          } => Some((
            expr_to_string(pattern),
            expr_to_string(replacement),
            true, // Rules are directed
          )),
          _ => None,
        };
        if let Some((from_str, to_str, directed)) = endpoints
          && let (Some(&fi), Some(&ti)) =
            (vertex_index.get(&from_str), vertex_index.get(&to_str))
        {
          // Self-loops do not contribute to the Kirchhoff matrix at all
          // (wolframscript ignores them for both degree and adjacency).
          if fi == ti {
            continue;
          }
          // Collapse parallel edges: only the first edge between an ordered
          // pair contributes to the degree, mirroring wolframscript's
          // KirchhoffMatrix on multigraphs.
          if directed {
            if adjacency[fi][ti] == 0 {
              adjacency[fi][ti] = 1;
              // A directed edge adds out-degree to the source and
              // in-degree to the target.
              degree[fi] += 1;
              degree[ti] += 1;
            }
          } else if adjacency[fi][ti] == 0 {
            adjacency[fi][ti] = 1;
            adjacency[ti][fi] = 1;
            // An undirected edge contributes one to each endpoint's degree.
            degree[fi] += 1;
            degree[ti] += 1;
          }
        }
      }
      let mut matrix = vec![vec![0i128; n]; n];
      for i in 0..n {
        for j in 0..n {
          if i == j {
            matrix[i][j] = degree[i];
          } else {
            matrix[i][j] = -adjacency[i][j];
          }
        }
      }
      return Ok(Expr::List(
        matrix
          .into_iter()
          .map(|row| Expr::List(row.into_iter().map(Expr::Integer).collect()))
          .collect(),
      ));
    }
    return Ok(unevaluated(name, args));
  }

  // ConnectedGraphQ[graph] — True if the graph is connected.
  // For undirected graphs this means a single connected component;
  // for directed graphs it means a single strongly connected component.
  // An empty graph (no vertices) is considered connected. Non-graph
  // arguments yield False.
  if name == "ConnectedGraphQ" && args.len() == 1 {
    let is_graph = matches!(&args[0], Expr::FunctionCall { name: gname, args: gargs }
      if gname == "Graph"
        && gargs.len() >= 2
        && matches!(gargs[0], Expr::List(_))
        && matches!(gargs[1], Expr::List(_)));
    if is_graph {
      // A graph is connected iff ConnectedComponents returns one component
      // (or zero components for the empty graph).
      let comps =
        evaluate_function_call_ast_inner("ConnectedComponents", args)?;
      if let Expr::List(comp_lists) = &comps {
        let connected = comp_lists.len() <= 1;
        return Ok(bool_expr(connected));
      }
    }
    return Ok(bool_expr(false));
  }

  // WeaklyConnectedGraphQ[g] — True iff the underlying undirected graph is
  // connected (edge directions ignored). Mirrors ConnectedGraphQ but counts
  // weakly-connected components. A non-graph argument is False.
  if name == "WeaklyConnectedGraphQ" && args.len() == 1 {
    let is_graph = matches!(&args[0], Expr::FunctionCall { name: gname, args: gargs }
      if gname == "Graph"
        && gargs.len() >= 2
        && matches!(gargs[0], Expr::List(_))
        && matches!(gargs[1], Expr::List(_)));
    if is_graph {
      let comps =
        evaluate_function_call_ast_inner("WeaklyConnectedComponents", args)?;
      if let Expr::List(comp_lists) = &comps {
        let connected = comp_lists.len() <= 1;
        return Ok(bool_expr(connected));
      }
    }
    return Ok(bool_expr(false));
  }

  // ConnectedComponents[Graph[{vertices}, {edges}]]
  // For undirected graphs: finds connected components (union-find)
  // For directed graphs: finds strongly connected components (Tarjan's)
  if name == "ConnectedComponents" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      // Check if graph is directed or undirected. A Rule-form edge (a -> b)
      // is a directed edge in wolframscript, just like DirectedEdge[a, b];
      // only an explicit vertex list leaves edges in Rule form, so this must
      // be recognised or such graphs misroute to the undirected branch.
      let is_directed = edges.iter().any(|e| {
        matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedEdge")
          || matches!(e, Expr::Rule { .. })
      });

      let comp_list = if is_directed {
        // Strongly connected components. Kosaraju gives the correct *within*-
        // component vertex order (matching wolframscript), while Tarjan visits
        // nodes in index order and returns components in reverse topological
        // order (sink component first) — wolframscript's *inter*-component
        // order. Combine them: take Kosaraju's components, then reorder them
        // to follow Tarjan's ordering. Never sort by size.
        let (pg_digraph, _) =
          crate::functions::graph::build_digraph(vertices, edges);
        let kosaraju = petgraph::algo::kosaraju_scc(&pg_digraph);
        let tarjan = petgraph::algo::tarjan_scc(&pg_digraph);
        // Rank each node by the position of its component in Tarjan's output.
        let mut tarjan_rank: std::collections::HashMap<usize, usize> =
          std::collections::HashMap::new();
        for (rank, comp) in tarjan.iter().enumerate() {
          for ni in comp {
            tarjan_rank.insert(ni.index(), rank);
          }
        }
        let mut components: Vec<(usize, Vec<Expr>)> = kosaraju
          .into_iter()
          .map(|comp| {
            let rank =
              comp.first().map(|ni| tarjan_rank[&ni.index()]).unwrap_or(0);
            let verts = comp
              .into_iter()
              .map(|ni| vertices[ni.index()].clone())
              .collect();
            (rank, verts)
          })
          .collect();
        components.sort_by_key(|(rank, _)| *rank);
        components.into_iter().map(|(_, verts)| verts).collect()
      } else {
        // Undirected: use petgraph connected_components
        let (pg_graph, _) = build_undirected_graph(vertices, edges);
        let n = vertices.len();
        let num_comps = petgraph::algo::connected_components(&pg_graph);
        // Map each node to its component
        use petgraph::algo::dijkstra;
        let mut comp_id = vec![usize::MAX; n];
        let mut next_comp = 0;
        for i in 0..n {
          if comp_id[i] == usize::MAX {
            let cid = next_comp;
            next_comp += 1;
            let dists = dijkstra(
              &pg_graph,
              petgraph::graph::NodeIndex::new(i),
              None,
              |_| 1u32,
            );
            for (ni, _) in dists {
              comp_id[ni.index()] = cid;
            }
          }
        }
        let _ = num_comps;
        let mut components: Vec<Vec<Expr>> = vec![vec![]; next_comp];
        for (i, v) in vertices.iter().enumerate() {
          if comp_id[i] < next_comp {
            components[comp_id[i]].push(v.clone());
          }
        }
        components.retain(|c| !c.is_empty());
        // Sort components by size (largest first)
        components.sort_by(|a, b| b.len().cmp(&a.len()));
        components
      };

      return Ok(Expr::List(
        comp_list
          .into_iter()
          .map(|v| Expr::List(v.into()))
          .collect(),
      ));
    }
    return Ok(unevaluated(name, args));
  }

  // ConnectedGraphComponents[graph] — returns subgraphs for each connected component
  if name == "ConnectedGraphComponents" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let (Expr::List(_vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
    {
      // Get connected components as vertex lists by delegating to ConnectedComponents
      let comp_result =
        evaluate_function_call_ast_inner("ConnectedComponents", args)?;
      if let Expr::List(comp_lists) = &comp_result {
        let mut subgraphs = Vec::new();
        for comp in comp_lists {
          if let Expr::List(comp_verts) = comp {
            let vert_set: std::collections::HashSet<String> =
              comp_verts.iter().map(expr_to_string).collect();
            // Collect edges where both endpoints are in this component
            let sub_edges: Vec<Expr> = edges
              .iter()
              .filter(|e| {
                if let Expr::FunctionCall { args: eargs, .. } = e
                  && eargs.len() == 2
                {
                  let from = expr_to_string(&eargs[0]);
                  let to = expr_to_string(&eargs[1]);
                  vert_set.contains(&from) && vert_set.contains(&to)
                } else {
                  false
                }
              })
              .cloned()
              .collect();
            subgraphs.push(Expr::FunctionCall {
              name: "Graph".to_string(),
              args: vec![
                Expr::List(comp_verts.clone()),
                Expr::List(sub_edges.into()),
              ]
              .into(),
            });
          }
        }
        return Ok(Expr::List(subgraphs.into()));
      }
    }
    return Ok(unevaluated(name, args));
  }

  // WeaklyConnectedComponents[graph] — connected components ignoring edge direction
  if name == "WeaklyConnectedComponents"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    // Treat all edges as undirected via build_ungraph
    let (pg_graph, _) = build_undirected_graph(vertices, edges);

    // Use petgraph dijkstra to find connected components
    let mut comp_id = vec![usize::MAX; n];
    let mut next_comp = 0;
    for i in 0..n {
      if comp_id[i] == usize::MAX {
        let cid = next_comp;
        next_comp += 1;
        let dists = petgraph::algo::dijkstra(
          &pg_graph,
          petgraph::graph::NodeIndex::new(i),
          None,
          |_| 1u32,
        );
        for (ni, _) in dists {
          comp_id[ni.index()] = cid;
        }
      }
    }
    let mut components: Vec<Vec<Expr>> = vec![vec![]; next_comp];
    for (i, v) in vertices.iter().enumerate() {
      if comp_id[i] < next_comp {
        components[comp_id[i]].push(v.clone());
      }
    }
    components.retain(|c| !c.is_empty());

    // Wolfram keeps undirected-graph component vertices in ascending vertex
    // order (`WeaklyConnectedComponents[CycleGraph[3]]` → `{{1, 2, 3}}`) but
    // reverses them for directed graphs (`{1 -> 2, 3 -> 4}` → `{{2, 1},
    // {4, 3}}`). Only reverse when at least one edge is directed.
    let directed = edges.iter().any(|e| {
      matches!(e, Expr::Rule { .. })
        || matches!(e, Expr::FunctionCall { name, .. }
          if name == "DirectedEdge" || name == "Rule")
    });
    if directed {
      for comp in &mut components {
        comp.reverse();
      }
    }
    components.sort_by(|a, b| b.len().cmp(&a.len()));
    return Ok(Expr::List(
      components
        .into_iter()
        .map(|v| Expr::List(v.into()))
        .collect(),
    ));
  }

  // StringExpression[...]: when all args are string literals, concatenate them
  if name == "StringExpression" {
    if args.iter().all(|a| matches!(a, Expr::String(_))) {
      let concatenated: String = args
        .iter()
        .map(|a| {
          if let Expr::String(s) = a {
            s.clone()
          } else {
            String::new()
          }
        })
        .collect();
      return Ok(Expr::String(concatenated));
    }
    return Ok(unevaluated(name, args));
  }

  // Triangle[] defaults to Triangle[{{0,0},{1,0},{0,1}}]
  if name == "Triangle" {
    if args.is_empty() {
      return Ok(Expr::FunctionCall {
        name: "Triangle".to_string(),
        args: vec![Expr::List(
          vec![
            Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
            Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()),
            Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into()),
          ]
          .into(),
        )]
        .into(),
      });
    }
    return Ok(unevaluated(name, args));
  }

  // PathGraph[{v1, v2, ...}] — create a path graph connecting sequential vertices
  if name == "PathGraph"
    && args.len() == 1
    && let Expr::List(verts) = &args[0]
    && verts.len() >= 2
  {
    let mut edges = Vec::new();
    for i in 0..verts.len() - 1 {
      edges.push(Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![verts[i].clone(), verts[i + 1].clone()].into(),
      });
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(verts.clone()), Expr::List(edges.into())].into(),
    });
  }

  // VertexCount[Graph[verts, edges]] — number of vertices
  if name == "VertexCount"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && !gargs.is_empty()
    && let Expr::List(verts) = &gargs[0]
  {
    return Ok(Expr::Integer(verts.len() as i128));
  }

  // VertexCount[graph, patt] counts the vertices matching the pattern, i.e.
  // Count[VertexList[graph], patt].
  if name == "VertexCount"
    && args.len() == 2
    && matches!(&args[0], Expr::FunctionCall { name: g, args: ga }
      if g == "Graph" && !ga.is_empty())
  {
    let vertex_list =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "VertexList".to_string(),
        args: vec![args[0].clone()].into(),
      })?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Count".to_string(),
      args: vec![vertex_list, args[1].clone()].into(),
    });
  }

  // EdgeCount[Graph[verts, edges]] — number of edges
  if name == "EdgeCount"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let Expr::List(edges) = &gargs[1]
  {
    return Ok(Expr::Integer(edges.len() as i128));
  }

  // EdgeCount[graph, patt] counts the edges matching the pattern `patt`,
  // i.e. Count[EdgeList[graph], patt]. A bare head like `UndirectedEdge`
  // matches no edge (use `_UndirectedEdge` or `UndirectedEdge[_, _]`).
  if name == "EdgeCount"
    && args.len() == 2
    && matches!(&args[0], Expr::FunctionCall { name: g, args: ga }
      if g == "Graph" && ga.len() >= 2)
  {
    let edge_list =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "EdgeList".to_string(),
        args: vec![args[0].clone()].into(),
      })?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Count".to_string(),
      args: vec![edge_list, args[1].clone()].into(),
    });
  }

  // FindMaximumFlow[graph, source, sink] — maximum flow value
  if name == "FindMaximumFlow"
    && args.len() == 3
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let Expr::List(verts) = &gargs[0]
    && let Expr::List(edges) = &gargs[1]
  {
    return find_maximum_flow_impl(verts, edges, &args[1], &args[2], args);
  }

  // VertexDegree[graph] or VertexDegree[graph, vertex] — vertex degree(s)
  if name == "VertexDegree"
    && (args.len() == 1 || args.len() == 2)
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(verts), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    // Count degree for each vertex
    let mut degrees = vec![0i128; verts.len()];
    for edge in edges {
      if let Expr::FunctionCall {
        name: _ename,
        args: eargs,
      } = edge
        && eargs.len() == 2
      {
        for (idx, v) in verts.iter().enumerate() {
          if expr_to_string(&eargs[0]) == expr_to_string(v)
            || expr_to_string(&eargs[1]) == expr_to_string(v)
          {
            degrees[idx] += 1;
          }
        }
      }
    }
    if args.len() == 2 {
      // Single vertex
      let target = expr_to_string(&args[1]);
      for (idx, v) in verts.iter().enumerate() {
        if expr_to_string(v) == target {
          return Ok(Expr::Integer(degrees[idx]));
        }
      }
    } else {
      return Ok(Expr::List(degrees.into_iter().map(Expr::Integer).collect()));
    }
  }

  // VertexInDegree[g] or VertexInDegree[g, v] — number of incoming edges per
  // vertex. Accepts either a Graph[...] or a raw edge list {a -> b, ...}.
  // Directed edges (Rule / DirectedEdge) contribute to the in-degree of their
  // target only (a self-loop a -> a counts once). Undirected edges
  // (TwoWayRule / UndirectedEdge) contribute to the in-degree of both
  // endpoints (an undirected self-loop a <-> a counts twice). Vertices are
  // ordered as in VertexList[g].
  if name == "VertexInDegree" && (args.len() == 1 || args.len() == 2) {
    // Resolve (ordered vertices, edges) from a Graph[...] or a raw edge list.
    let resolved: Option<(Vec<Expr>, Vec<Expr>)> = match &args[0] {
      Expr::FunctionCall {
        name: gname,
        args: gargs,
      } if gname == "Graph"
        && gargs.len() >= 2
        && matches!(&gargs[0], Expr::List(_))
        && matches!(&gargs[1], Expr::List(_)) =>
      {
        if let (Expr::List(verts), Expr::List(edges)) = (&gargs[0], &gargs[1]) {
          Some((verts.to_vec(), edges.to_vec()))
        } else {
          None
        }
      }
      Expr::List(edges) => {
        // Raw edge list: derive vertices in first-appearance order.
        let mut verts: Vec<Expr> = Vec::new();
        let push = |v: &Expr, set: &mut Vec<Expr>| {
          if !set
            .iter()
            .any(|e| crate::evaluator::pattern_matching::expr_equal(e, v))
          {
            set.push(v.clone());
          }
        };
        let mut ok = !edges.is_empty();
        for e in edges.iter() {
          match edge_endpoints(e) {
            Some((src, dst)) => {
              push(&src, &mut verts);
              push(&dst, &mut verts);
            }
            None => {
              ok = false;
              break;
            }
          }
        }
        if ok {
          Some((verts, edges.to_vec()))
        } else {
          None
        }
      }
      _ => None,
    };

    if let Some((verts, edges)) = resolved {
      let mut indeg = vec![0i128; verts.len()];
      let vidx = |v: &Expr| -> Option<usize> {
        verts
          .iter()
          .position(|x| crate::evaluator::pattern_matching::expr_equal(x, v))
      };
      // In a mixed/directed graph (at least one directed edge) only directed
      // edges contribute to in-degree; undirected edges contribute nothing.
      // In a purely undirected graph every undirected edge contributes to the
      // in-degree of both endpoints (a self-loop counts twice).
      let has_directed = edges
        .iter()
        .any(|e| matches!(edge_endpoints_dir(e), Some((_, _, true))));
      for e in &edges {
        if let Some((src, dst, directed)) = edge_endpoints_dir(e) {
          if directed {
            if let Some(i) = vidx(&dst) {
              indeg[i] += 1;
            }
          } else if !has_directed {
            if crate::evaluator::pattern_matching::expr_equal(&src, &dst) {
              if let Some(i) = vidx(&src) {
                indeg[i] += 2;
              }
            } else {
              if let Some(i) = vidx(&src) {
                indeg[i] += 1;
              }
              if let Some(i) = vidx(&dst) {
                indeg[i] += 1;
              }
            }
          }
        }
      }
      if args.len() == 2 {
        if let Some(i) = vidx(&args[1]) {
          return Ok(Expr::Integer(indeg[i]));
        }
        // Vertex not in graph — leave unevaluated, like wolframscript.
      } else {
        return Ok(Expr::List(indeg.into_iter().map(Expr::Integer).collect()));
      }
    }
  }

  // VertexOutDegree[g] or VertexOutDegree[g, v] — number of outgoing edges per
  // vertex. Accepts either a Graph[...] or a raw edge list {a -> b, ...}.
  // Directed edges (Rule / DirectedEdge) contribute to the out-degree of their
  // source only (a self-loop a -> a counts once). Undirected edges
  // (TwoWayRule / UndirectedEdge) contribute to the out-degree of both
  // endpoints (an undirected self-loop a <-> a counts twice). Vertices are
  // ordered as in VertexList[g].
  if name == "VertexOutDegree" && (args.len() == 1 || args.len() == 2) {
    // Resolve (ordered vertices, edges) from a Graph[...] or a raw edge list.
    let resolved: Option<(Vec<Expr>, Vec<Expr>)> = match &args[0] {
      Expr::FunctionCall {
        name: gname,
        args: gargs,
      } if gname == "Graph"
        && gargs.len() >= 2
        && matches!(&gargs[0], Expr::List(_))
        && matches!(&gargs[1], Expr::List(_)) =>
      {
        if let (Expr::List(verts), Expr::List(edges)) = (&gargs[0], &gargs[1]) {
          Some((verts.to_vec(), edges.to_vec()))
        } else {
          None
        }
      }
      Expr::List(edges) => {
        // Raw edge list: derive vertices in first-appearance order.
        let mut verts: Vec<Expr> = Vec::new();
        let push = |v: &Expr, set: &mut Vec<Expr>| {
          if !set
            .iter()
            .any(|e| crate::evaluator::pattern_matching::expr_equal(e, v))
          {
            set.push(v.clone());
          }
        };
        let mut ok = !edges.is_empty();
        for e in edges.iter() {
          match edge_endpoints(e) {
            Some((src, dst)) => {
              push(&src, &mut verts);
              push(&dst, &mut verts);
            }
            None => {
              ok = false;
              break;
            }
          }
        }
        if ok {
          Some((verts, edges.to_vec()))
        } else {
          None
        }
      }
      _ => None,
    };

    if let Some((verts, edges)) = resolved {
      let mut outdeg = vec![0i128; verts.len()];
      let vidx = |v: &Expr| -> Option<usize> {
        verts
          .iter()
          .position(|x| crate::evaluator::pattern_matching::expr_equal(x, v))
      };
      // In a mixed/directed graph (at least one directed edge) only directed
      // edges contribute to out-degree; undirected edges contribute nothing.
      // In a purely undirected graph every undirected edge contributes to the
      // out-degree of both endpoints (a self-loop counts twice).
      let has_directed = edges
        .iter()
        .any(|e| matches!(edge_endpoints_dir(e), Some((_, _, true))));
      for e in &edges {
        if let Some((src, dst, directed)) = edge_endpoints_dir(e) {
          if directed {
            if let Some(i) = vidx(&src) {
              outdeg[i] += 1;
            }
          } else if !has_directed {
            if crate::evaluator::pattern_matching::expr_equal(&src, &dst) {
              if let Some(i) = vidx(&src) {
                outdeg[i] += 2;
              }
            } else {
              if let Some(i) = vidx(&src) {
                outdeg[i] += 1;
              }
              if let Some(i) = vidx(&dst) {
                outdeg[i] += 1;
              }
            }
          }
        }
      }
      if args.len() == 2 {
        if let Some(i) = vidx(&args[1]) {
          return Ok(Expr::Integer(outdeg[i]));
        }
        // Vertex not in graph — leave unevaluated, like wolframscript.
      } else {
        return Ok(Expr::List(outdeg.into_iter().map(Expr::Integer).collect()));
      }
    }
  }

  // FindGraphIsomorphism[g1, g2] — find vertex mapping between isomorphic graphs
  if name == "FindGraphIsomorphism"
    && (args.len() == 2 || args.len() == 3)
    && let Expr::FunctionCall {
      name: gname1,
      args: gargs1,
    } = &args[0]
    && gname1 == "Graph"
    && gargs1.len() >= 2
    && let (Expr::List(verts1), Expr::List(edges1)) = (&gargs1[0], &gargs1[1])
    && let Expr::FunctionCall {
      name: gname2,
      args: gargs2,
    } = &args[1]
    && gname2 == "Graph"
    && gargs2.len() >= 2
    && let (Expr::List(verts2), Expr::List(edges2)) = (&gargs2[0], &gargs2[1])
  {
    return find_graph_isomorphism_impl(verts1, edges1, verts2, edges2, args);
  }

  // IsomorphicGraphQ[g1, g2] — True iff a graph isomorphism exists, i.e. when
  // FindGraphIsomorphism returns a non-empty mapping.
  if name == "IsomorphicGraphQ"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: gname1,
      args: gargs1,
    } = &args[0]
    && gname1 == "Graph"
    && gargs1.len() >= 2
    && let (Expr::List(verts1), Expr::List(edges1)) = (&gargs1[0], &gargs1[1])
    && let Expr::FunctionCall {
      name: gname2,
      args: gargs2,
    } = &args[1]
    && gname2 == "Graph"
    && gargs2.len() >= 2
    && let (Expr::List(verts2), Expr::List(edges2)) = (&gargs2[0], &gargs2[1])
  {
    let iso =
      find_graph_isomorphism_impl(verts1, edges1, verts2, edges2, args)?;
    let found = matches!(&iso, Expr::List(maps) if !maps.is_empty());
    return Ok(bool_expr(found));
  }
  // Any non-graph argument makes IsomorphicGraphQ False (matching wolframscript).
  if name == "IsomorphicGraphQ" && args.len() == 2 {
    return Ok(bool_expr(false));
  }

  // FindSpanningTree[Graph[verts, edges]] — minimum spanning tree (Kruskal's)
  if name == "FindSpanningTree"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let (Expr::List(verts), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    return find_spanning_tree_impl(verts, edges);
  }

  // GraphPropertyDistribution[property[g], Distributed[g, graphDist]]
  // Returns the probability distribution of a graph property over a graph distribution.
  if name == "GraphPropertyDistribution"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: dname,
      args: dargs,
    } = &args[1]
    && dname == "Distributed"
    && dargs.len() == 2
  {
    let var = &dargs[0];
    let graph_dist = &dargs[1];
    let property = &args[0];

    // Extract graph distribution name and params
    if let Expr::FunctionCall {
      name: dist_name,
      args: dist_args,
    } = graph_dist
    {
      // Extract property name and check if the graph variable matches
      if let Expr::FunctionCall {
        name: prop_name,
        args: prop_args,
      } = property
      {
        let var_str = expr_to_string(var);
        let prop_var_matches =
          !prop_args.is_empty() && expr_to_string(&prop_args[0]) == var_str;

        match (prop_name.as_str(), dist_name.as_str()) {
          // EdgeCount[g] with BernoulliGraphDistribution[n, p]
          // → BinomialDistribution[((-1 + n)*n)/2, p]
          ("EdgeCount", "BernoulliGraphDistribution")
            if prop_var_matches
              && prop_args.len() == 1
              && dist_args.len() == 2 =>
          {
            let n = &dist_args[0];
            let p = &dist_args[1];
            // Build ((-1 + n)*n)/2 — flatten Times to get correct parenthesization
            let n_minus_1 = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Integer(-1), n.clone()].into(),
            };
            let half = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
                },
                n_minus_1,
                n.clone(),
              ]
              .into(),
            };
            return Ok(Expr::FunctionCall {
              name: "BinomialDistribution".to_string(),
              args: vec![half, p.clone()].into(),
            });
          }

          // VertexCount[g] with BernoulliGraphDistribution[n, p]
          // → DiscreteUniformDistribution[{n, n}]
          ("VertexCount", "BernoulliGraphDistribution")
            if prop_var_matches
              && prop_args.len() == 1
              && dist_args.len() == 2 =>
          {
            let n = &dist_args[0];
            return Ok(Expr::FunctionCall {
              name: "DiscreteUniformDistribution".to_string(),
              args: vec![Expr::List(vec![n.clone(), n.clone()].into())].into(),
            });
          }

          // VertexDegree[g, v] with BernoulliGraphDistribution[n, p]
          // → BinomialDistribution[-1 + n, p]
          ("VertexDegree", "BernoulliGraphDistribution")
            if prop_var_matches
              && prop_args.len() == 2
              && dist_args.len() == 2 =>
          {
            let n = &dist_args[0];
            let p = &dist_args[1];
            let n_minus_1 = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Integer(-1), n.clone()].into(),
            };
            return Ok(Expr::FunctionCall {
              name: "BinomialDistribution".to_string(),
              args: vec![n_minus_1, p.clone()].into(),
            });
          }

          // EdgeCount[g] with UniformGraphDistribution[n, m]
          // → DiscreteUniformDistribution[{m, m}]
          ("EdgeCount", "UniformGraphDistribution")
            if prop_var_matches
              && prop_args.len() == 1
              && dist_args.len() == 2 =>
          {
            let m = &dist_args[1];
            return Ok(Expr::FunctionCall {
              name: "DiscreteUniformDistribution".to_string(),
              args: vec![Expr::List(vec![m.clone(), m.clone()].into())].into(),
            });
          }

          // VertexCount[g] with UniformGraphDistribution[n, m]
          // → DiscreteUniformDistribution[{n, n}]
          ("VertexCount", "UniformGraphDistribution")
            if prop_var_matches
              && prop_args.len() == 1
              && dist_args.len() == 2 =>
          {
            let n = &dist_args[0];
            return Ok(Expr::FunctionCall {
              name: "DiscreteUniformDistribution".to_string(),
              args: vec![Expr::List(vec![n.clone(), n.clone()].into())].into(),
            });
          }

          // VertexDegree[g, v] with UniformGraphDistribution[n, m]
          // → HypergeometricDistribution[m, -1 + n, ((-1 + n)*n)/2]
          ("VertexDegree", "UniformGraphDistribution")
            if prop_var_matches
              && prop_args.len() == 2
              && dist_args.len() == 2 =>
          {
            let n = &dist_args[0];
            let m = &dist_args[1];
            let n_minus_1 = Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Integer(-1), n.clone()].into(),
            };
            // Flatten Times to get correct parenthesization: ((-1 + n)*n)/2
            let half = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
                },
                n_minus_1.clone(),
                n.clone(),
              ]
              .into(),
            };
            return Ok(Expr::FunctionCall {
              name: "HypergeometricDistribution".to_string(),
              args: vec![m.clone(), n_minus_1, half].into(),
            });
          }

          _ => {
            // Unknown property: return unevaluated with formal variable
            // e.g., g → \[FormalG] (Wolfram convention for placeholder variables)
            if let Expr::Identifier(var_name) = var {
              let formal_name = if var_name.len() == 1 {
                let ch = var_name.chars().next().unwrap();
                let formal_ch = ch.to_uppercase().next().unwrap_or(ch);
                format!("\\[Formal{}]", formal_ch)
              } else {
                var_name.clone()
              };
              if formal_name != *var_name {
                let formal_var = Expr::Identifier(formal_name);
                // Replace var in the property and Distributed args
                fn replace_var(
                  expr: &Expr,
                  old: &str,
                  new_expr: &Expr,
                ) -> Expr {
                  match expr {
                    Expr::Identifier(name) if name == old => new_expr.clone(),
                    Expr::FunctionCall { name, args } => Expr::FunctionCall {
                      name: name.clone(),
                      args: args
                        .iter()
                        .map(|a| replace_var(a, old, new_expr))
                        .collect(),
                    },
                    Expr::List(items) => Expr::List(
                      items
                        .iter()
                        .map(|a| replace_var(a, old, new_expr))
                        .collect(),
                    ),
                    _ => expr.clone(),
                  }
                }
                let new_property = replace_var(property, var_name, &formal_var);
                let new_distributed = Expr::FunctionCall {
                  name: "Distributed".to_string(),
                  args: vec![formal_var, graph_dist.clone()].into(),
                };
                return Ok(Expr::FunctionCall {
                  name: "GraphPropertyDistribution".to_string(),
                  args: vec![new_property, new_distributed].into(),
                });
              }
            }
          }
        }
      }
    }
  }

  // TuttePolynomial[graph, {x, y}] — apply the 1-arg function form to {x, y}.
  if name == "TuttePolynomial"
    && args.len() == 2
    && let Expr::List(xy) = &args[1]
    && xy.len() == 2
  {
    let func =
      evaluate_function_call_ast("TuttePolynomial", &[args[0].clone()])?;
    let applied = Expr::CurriedCall {
      func: Box::new(func),
      args: vec![xy[0].clone(), xy[1].clone()],
    };
    return evaluate_expr_to_expr(&applied);
  }

  // TuttePolynomial[graph] — compute the Tutte polynomial via deletion-contraction
  if name == "TuttePolynomial"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() >= 2
    && let Expr::List(verts) = &gargs[0]
    && let Expr::List(edges) = &gargs[1]
  {
    use std::collections::HashMap;

    type Poly = HashMap<(i32, i32), i128>;

    fn poly_add(a: &Poly, b: &Poly) -> Poly {
      let mut result = a.clone();
      for (k, v) in b {
        *result.entry(*k).or_insert(0) += v;
      }
      result.retain(|_, v| *v != 0);
      result
    }

    fn poly_mul_var(p: &Poly, var: u8) -> Poly {
      // var: 0 = x, 1 = y
      p.iter()
        .map(|(&(xp, yp), &c)| {
          if var == 0 {
            ((xp + 1, yp), c)
          } else {
            ((xp, yp + 1), c)
          }
        })
        .collect()
    }

    // Compute connected components count using BFS
    fn count_components(
      vertex_ids: &[usize],
      adj: &HashMap<usize, Vec<usize>>,
    ) -> usize {
      let mut visited = std::collections::HashSet::new();
      let mut components = 0;
      for &v in vertex_ids {
        if !visited.contains(&v) {
          components += 1;
          let mut stack = vec![v];
          while let Some(u) = stack.pop() {
            if visited.insert(u)
              && let Some(neighbors) = adj.get(&u)
            {
              for &n in neighbors {
                if !visited.contains(&n) {
                  stack.push(n);
                }
              }
            }
          }
        }
      }
      components
    }

    // Build adjacency for given edge set
    fn build_adj(edges: &[(usize, usize)]) -> HashMap<usize, Vec<usize>> {
      let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
      for &(u, v) in edges {
        adj.entry(u).or_default().push(v);
        adj.entry(v).or_default().push(u);
      }
      adj
    }

    fn tutte_poly(vertex_ids: &[usize], edges: &[(usize, usize)]) -> Poly {
      if edges.is_empty() {
        let mut p = Poly::new();
        p.insert((0, 0), 1);
        return p;
      }

      let (u, v) = edges[0];
      let rest = &edges[1..];

      // Check if it's a loop
      if u == v {
        // Loop: T(G) = y * T(G - e)
        let sub = tutte_poly(vertex_ids, rest);
        return poly_mul_var(&sub, 1);
      }

      // Check if it's a bridge: removing it increases components
      let adj_with = build_adj(edges);
      let comp_with = count_components(vertex_ids, &adj_with);
      let adj_without = build_adj(rest);
      let comp_without = count_components(vertex_ids, &adj_without);

      if comp_without > comp_with {
        // Bridge: T(G) = x * T(G / e)
        // Contract: merge v into u in remaining edges
        let contracted_edges: Vec<(usize, usize)> = rest
          .iter()
          .map(|&(a, b)| {
            let a2 = if a == v { u } else { a };
            let b2 = if b == v { u } else { b };
            (a2, b2)
          })
          .collect();
        let contracted_verts: Vec<usize> =
          vertex_ids.iter().copied().filter(|&x| x != v).collect();
        let sub = tutte_poly(&contracted_verts, &contracted_edges);
        poly_mul_var(&sub, 0)
      } else {
        // Regular edge: T(G) = T(G - e) + T(G / e)
        // Deletion
        let del = tutte_poly(vertex_ids, rest);
        // Contraction: merge v into u, keep self-loops (they contribute y terms)
        let contracted_edges: Vec<(usize, usize)> = rest
          .iter()
          .map(|&(a, b)| {
            let a2 = if a == v { u } else { a };
            let b2 = if b == v { u } else { b };
            (a2, b2)
          })
          .collect();
        let contracted_verts: Vec<usize> =
          vertex_ids.iter().copied().filter(|&x| x != v).collect();
        let con = tutte_poly(&contracted_verts, &contracted_edges);
        poly_add(&del, &con)
      }
    }

    // Convert graph vertices to indices
    let vert_strs: Vec<String> = verts.iter().map(expr_to_string).collect();
    let vertex_ids: Vec<usize> = (0..verts.len()).collect();
    let edge_pairs: Vec<(usize, usize)> = edges
      .iter()
      .filter_map(|e| {
        if let Expr::FunctionCall { args: eargs, .. } = e
          && eargs.len() == 2
        {
          let a_str = expr_to_string(&eargs[0]);
          let b_str = expr_to_string(&eargs[1]);
          let a_idx = vert_strs.iter().position(|s| *s == a_str)?;
          let b_idx = vert_strs.iter().position(|s| *s == b_str)?;
          Some((a_idx, b_idx))
        } else {
          None
        }
      })
      .collect();

    let poly = tutte_poly(&vertex_ids, &edge_pairs);

    // Convert polynomial to Expr using Slot[1] for x, Slot[2] for y
    // Return as Function[{x, y}, poly_expr] using formal params
    let slot_x = Expr::Slot(1);
    let slot_y = Expr::Slot(2);

    // Build polynomial expression
    let mut terms: Vec<((i32, i32), i128)> = poly.into_iter().collect();
    terms.sort_by_key(|a| a.0);

    if terms.is_empty() {
      return Ok(Expr::Function {
        body: Box::new(Expr::Integer(0)),
      });
    }

    let mut term_exprs: Vec<Expr> = Vec::new();
    for &((xp, yp), coeff) in &terms {
      let mut factors: Vec<Expr> = Vec::new();
      if coeff != 1 || (xp == 0 && yp == 0) {
        factors.push(Expr::Integer(coeff));
      }
      if xp == 1 {
        factors.push(slot_x.clone());
      } else if xp > 1 {
        factors.push(Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![slot_x.clone(), Expr::Integer(xp as i128)].into(),
        });
      }
      if yp == 1 {
        factors.push(slot_y.clone());
      } else if yp > 1 {
        factors.push(Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![slot_y.clone(), Expr::Integer(yp as i128)].into(),
        });
      }
      let term = if factors.len() == 1 {
        factors.pop().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: factors.into(),
        }
      };
      term_exprs.push(term);
    }

    let poly_expr = if term_exprs.len() == 1 {
      term_exprs.pop().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: term_exprs.into(),
      }
    };

    return Ok(Expr::Function {
      body: Box::new(poly_expr),
    });
  }

  // BoundedRegionQ[region] — test if a geometric region is bounded
  if name == "BoundedRegionQ"
    && args.len() == 1
    && let Expr::FunctionCall {
      name: rname,
      args: _,
    } = &args[0]
  {
    let result = match rname.as_str() {
      // Bounded regions
      "Disk" | "Ball" | "Rectangle" | "Cuboid" | "Polygon" | "Triangle"
      | "Line" | "BezierCurve" | "BSplineCurve" | "Circle" | "Sphere"
      | "Ellipsoid" | "Cone" | "Cylinder" | "Tetrahedron" | "Hexahedron"
      | "Prism" | "Pyramid" | "Point" | "Interval" | "Simplex"
      | "Parallelepiped" | "Annulus" | "StadiumShape" | "DiskSegment"
      | "SphericalShell" | "CapsuleShape" | "Torus" | "FilledTorus"
      | "Parallelogram" => Some(true),
      // Unbounded regions
      "HalfPlane" | "HalfSpace" | "InfiniteLine" | "InfinitePlane"
      | "HalfLine" | "ConicHullRegion" | "AffineHalfSpace" | "AffineSpace" => {
        Some(false)
      }
      _ => None,
    };
    if let Some(b) = result {
      return Ok(bool_expr(b));
    }
  }

  // FunctionContinuous[f, x] or FunctionContinuous[{f, cond}, x] or FunctionContinuous[f, {x, y, ...}]
  if name == "FunctionContinuous" && (args.len() == 2 || args.len() == 3) {
    // Helper: check if an expression is continuous over all reals w.r.t. given variables
    fn is_continuous_on_reals(expr: &Expr, vars: &[String]) -> Option<bool> {
      match expr {
        Expr::Integer(_)
        | Expr::Real(_)
        | Expr::BigInteger(_)
        | Expr::BigFloat(_, _)
        | Expr::Constant(_) => Some(true),
        Expr::Identifier(s) => {
          if vars.contains(s) {
            Some(true) // Variable itself is continuous (identity)
          } else {
            Some(true) // Constant w.r.t. the variable
          }
        }
        Expr::FunctionCall { name, args } => {
          match name.as_str() {
            // Everywhere-continuous elementary functions
            "Sin" | "Cos" | "Sinh" | "Cosh" | "Exp" | "Tanh" | "Sech"
            | "ArcTan" | "ArcSinh" | "Erf" | "Erfc" | "Sinc" => {
              if args.len() == 1 {
                is_continuous_on_reals(&args[0], vars)
              } else {
                None
              }
            }
            "Abs" | "RealAbs" => {
              if args.len() == 1 {
                is_continuous_on_reals(&args[0], vars)
              } else {
                None
              }
            }
            // Functions not continuous on all reals
            "Tan" | "Cot" | "Sec" | "Csc" | "Log" | "Sqrt" | "Floor"
            | "Ceiling" | "Round" | "Sign" | "Coth" | "Csch" => {
              if args.iter().any(|a| expr_mentions_vars(a, vars)) {
                Some(false)
              } else {
                Some(true)
              }
            }
            "Plus" | "Times" => {
              let mut all_true = true;
              for arg in args {
                match is_continuous_on_reals(arg, vars) {
                  Some(true) => {}
                  Some(false) => return Some(false),
                  None => all_true = false,
                }
              }
              if all_true { Some(true) } else { None }
            }
            "Power" if args.len() == 2 => {
              let base = &args[0];
              let exp = &args[1];
              let base_has_var = expr_mentions_vars(base, vars);
              let exp_has_var = expr_mentions_vars(exp, vars);

              if !base_has_var && !exp_has_var {
                return Some(true);
              }

              match exp {
                Expr::Integer(n) if *n >= 0 => {
                  is_continuous_on_reals(base, vars)
                }
                Expr::Integer(_) => {
                  if base_has_var {
                    Some(false)
                  } else {
                    Some(true)
                  }
                }
                Expr::FunctionCall {
                  name: rname,
                  args: rargs,
                } if rname == "Rational" && rargs.len() == 2 => {
                  if base_has_var {
                    Some(false)
                  } else {
                    Some(true)
                  }
                }
                _ => {
                  if !exp_has_var && !base_has_var {
                    Some(true)
                  } else {
                    None
                  }
                }
              }
            }
            "Rational" if args.len() == 2 => Some(true),
            "Gamma" if args.len() == 1 => {
              if expr_mentions_vars(&args[0], vars) {
                Some(false)
              } else {
                Some(true)
              }
            }
            "Gamma" if args.len() == 2 => {
              is_continuous_on_reals(&args[1], vars)
            }
            _ => None,
          }
        }
        Expr::BinaryOp { op, left, right } => match op {
          BinaryOperator::Plus
          | BinaryOperator::Minus
          | BinaryOperator::Times => {
            match (
              is_continuous_on_reals(left, vars),
              is_continuous_on_reals(right, vars),
            ) {
              (Some(true), Some(true)) => Some(true),
              (Some(false), _) | (_, Some(false)) => Some(false),
              _ => None,
            }
          }
          BinaryOperator::Divide => {
            let right_has_var = expr_mentions_vars(right, vars);
            if right_has_var {
              Some(false) // Division by expression with variable
            } else {
              is_continuous_on_reals(left, vars)
            }
          }
          BinaryOperator::Power => {
            let base_has_var = expr_mentions_vars(left, vars);
            let exp_has_var = expr_mentions_vars(right, vars);
            if !base_has_var && !exp_has_var {
              return Some(true);
            }
            if !base_has_var && exp_has_var {
              // constant^f(x) — continuous if f(x) is continuous (e.g. E^x)
              return is_continuous_on_reals(right, vars);
            }
            match right.as_ref() {
              Expr::Integer(n) if *n >= 0 => is_continuous_on_reals(left, vars),
              Expr::Integer(_) => {
                if base_has_var {
                  Some(false)
                } else {
                  Some(true)
                }
              }
              Expr::FunctionCall { name: rn, args: ra }
                if rn == "Rational" && ra.len() == 2 =>
              {
                if base_has_var {
                  Some(false)
                } else {
                  Some(true)
                }
              }
              _ => None,
            }
          }
          _ => None,
        },
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } => is_continuous_on_reals(operand, vars),
        _ => None,
      }
    }

    fn expr_mentions_vars(expr: &Expr, vars: &[String]) -> bool {
      match expr {
        Expr::Identifier(s) => vars.contains(s),
        Expr::Integer(_)
        | Expr::Real(_)
        | Expr::BigInteger(_)
        | Expr::BigFloat(_, _)
        | Expr::Constant(_) => false,
        Expr::FunctionCall { args, .. } => {
          args.iter().any(|a| expr_mentions_vars(a, vars))
        }
        Expr::List(items) => items.iter().any(|a| expr_mentions_vars(a, vars)),
        Expr::BinaryOp { left, right, .. } => {
          expr_mentions_vars(left, vars) || expr_mentions_vars(right, vars)
        }
        Expr::UnaryOp { operand, .. } => expr_mentions_vars(operand, vars),
        _ => true,
      }
    }

    fn is_continuous_on_positive(expr: &Expr, vars: &[String]) -> Option<bool> {
      match expr {
        Expr::Integer(_)
        | Expr::Real(_)
        | Expr::BigInteger(_)
        | Expr::BigFloat(_, _)
        | Expr::Constant(_) => Some(true),
        Expr::Identifier(_) => Some(true),
        Expr::FunctionCall { name, args } => match name.as_str() {
          "Sin" | "Cos" | "Sinh" | "Cosh" | "Exp" | "Tanh" | "Sech"
          | "ArcTan" | "ArcSinh" | "Erf" | "Erfc" | "Sinc" | "Abs"
          | "RealAbs" | "Log" | "Sqrt" => {
            if args.len() == 1 {
              is_continuous_on_positive(&args[0], vars)
            } else {
              None
            }
          }
          "Power" if args.len() == 2 => match &args[1] {
            Expr::Integer(n) if *n >= 0 => {
              is_continuous_on_positive(&args[0], vars)
            }
            Expr::Integer(_) => is_continuous_on_positive(&args[0], vars),
            Expr::FunctionCall { name: rn, args: ra }
              if rn == "Rational" && ra.len() == 2 =>
            {
              is_continuous_on_positive(&args[0], vars)
            }
            _ => None,
          },
          "Plus" | "Times" => {
            let mut all = true;
            for a in args {
              match is_continuous_on_positive(a, vars) {
                Some(true) => {}
                Some(false) => return Some(false),
                None => all = false,
              }
            }
            if all { Some(true) } else { None }
          }
          "Rational" if args.len() == 2 => Some(true),
          "Floor" | "Ceiling" | "Round" | "Sign" | "Tan" | "Cot" | "Sec"
          | "Csc" => {
            if args.iter().any(|a| expr_mentions_vars(a, vars)) {
              Some(false)
            } else {
              Some(true)
            }
          }
          _ => None,
        },
        Expr::BinaryOp { op, left, right } => match op {
          BinaryOperator::Plus
          | BinaryOperator::Minus
          | BinaryOperator::Times => {
            match (
              is_continuous_on_positive(left, vars),
              is_continuous_on_positive(right, vars),
            ) {
              (Some(true), Some(true)) => Some(true),
              (Some(false), _) | (_, Some(false)) => Some(false),
              _ => None,
            }
          }
          BinaryOperator::Divide => {
            // On positive domain, division is fine as long as both parts are continuous
            match (
              is_continuous_on_positive(left, vars),
              is_continuous_on_positive(right, vars),
            ) {
              (Some(true), Some(true)) => Some(true),
              (Some(false), _) | (_, Some(false)) => Some(false),
              _ => None,
            }
          }
          BinaryOperator::Power => {
            // On positive domain, all powers are continuous
            match (
              is_continuous_on_positive(left, vars),
              is_continuous_on_positive(right, vars),
            ) {
              (Some(true), Some(true)) => Some(true),
              (Some(false), _) | (_, Some(false)) => Some(false),
              _ => None,
            }
          }
          _ => None,
        },
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand,
        } => is_continuous_on_positive(operand, vars),
        _ => None,
      }
    }

    let (func_expr, condition) = match &args[0] {
      Expr::List(items) if items.len() == 2 => (&items[0], Some(&items[1])),
      other => (other, None),
    };

    let var_names: Vec<String> = match &args[1] {
      Expr::Identifier(s) => vec![s.clone()],
      Expr::List(items) => items
        .iter()
        .filter_map(|e| {
          if let Expr::Identifier(s) = e {
            Some(s.clone())
          } else {
            None
          }
        })
        .collect(),
      _ => vec![],
    };

    if !var_names.is_empty() {
      let is_positive_domain = condition.is_some_and(|cond| {
        // Check Comparison { operands: [x, 0], operators: [Greater] }
        if let Expr::Comparison {
          operands,
          operators,
        } = cond
          && operands.len() == 2
            && operators.len() == 1
            && matches!(operators[0], ComparisonOp::Greater | ComparisonOp::GreaterEqual)
            && matches!(&operands[0], Expr::Identifier(s) if var_names.contains(s))
            && matches!(&operands[1], Expr::Integer(0))
          {
            return true;
          }
        // Also check FunctionCall form (Greater[x, 0])
        if let Expr::FunctionCall { name: op, args: cargs } = cond
          && (op == "Greater" || op == "GreaterEqual")
            && cargs.len() == 2
            && matches!(&cargs[0], Expr::Identifier(s) if var_names.contains(s))
            && matches!(&cargs[1], Expr::Integer(0))
          {
            return true;
          }
        false
      });

      let result = if is_positive_domain {
        is_continuous_on_positive(func_expr, &var_names)
      } else if condition.is_none() {
        is_continuous_on_reals(func_expr, &var_names)
      } else {
        None
      };

      if let Some(b) = result {
        return Ok(bool_expr(b));
      }
    }
  }

  // GraphEmbedding[graph] or GraphEmbedding[graph, method] — vertex coordinates
  if name == "GraphEmbedding" && (args.len() == 1 || args.len() == 2) {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() >= 2
      && let Expr::List(verts) = &gargs[0]
    {
      let n = verts.len();
      if n == 0 {
        return Ok(Expr::List(vec![].into()));
      }
      // Circular embedding: angle_k = π/2 + k * 2π/n for k = 1..n
      // Snap coordinates near simple rational values (0, ±0.5, ±1) to
      // eliminate platform-dependent ULP differences in f64 trig.
      use crate::functions::graph::snap_coord;
      let coords: Vec<Expr> = (1..=n)
        .map(|k| {
          let angle = std::f64::consts::FRAC_PI_2
            + (k as f64) * 2.0 * std::f64::consts::PI / (n as f64);
          Expr::List(
            vec![
              Expr::Real(snap_coord(angle.cos())),
              Expr::Real(snap_coord(angle.sin())),
            ]
            .into(),
          )
        })
        .collect();
      return Ok(Expr::List(coords.into()));
    }
    // Return unevaluated for non-graph input
    return Ok(unevaluated(name, args));
  }

  // SubstitutionSystem[rules, init, n] — apply substitution rules iteratively
  if name == "SubstitutionSystem"
    && args.len() == 3
    && let Expr::Integer(n_steps) = &args[2]
  {
    let n = *n_steps as usize;
    // Extract rules from either Expr::Rule or Expr::FunctionCall{name:"Rule",...}
    let extract_rules = |rule_list: &[Expr]| -> Vec<(Expr, Expr)> {
      let mut rules = Vec::new();
      for rule in rule_list {
        match rule {
          Expr::Rule {
            pattern,
            replacement,
          } => {
            rules.push((*pattern.clone(), *replacement.clone()));
          }
          Expr::FunctionCall {
            name: rn,
            args: rargs,
          } if rn == "Rule" && rargs.len() == 2 => {
            rules.push((rargs[0].clone(), rargs[1].clone()));
          }
          _ => {}
        }
      }
      rules
    };

    // Check if string mode or list mode
    if let Expr::String(init_str) = &args[1] {
      // String mode: rules map single chars to strings
      if let Expr::List(rule_list) = &args[0] {
        let raw_rules = extract_rules(rule_list);
        let mut rules: Vec<(char, String)> = Vec::new();
        for (from, to) in &raw_rules {
          if let (Expr::String(from_s), Expr::String(to_s)) = (from, to)
            && let Some(ch) = from_s.chars().next()
          {
            rules.push((ch, to_s.clone()));
          }
        }
        let mut history: Vec<Expr> = vec![Expr::String(init_str.clone())];
        let mut current = init_str.clone();
        for _ in 0..n {
          let mut next = String::new();
          for ch in current.chars() {
            if let Some((_from, to)) = rules.iter().find(|(f, _)| *f == ch) {
              next.push_str(to);
            } else {
              next.push(ch);
            }
          }
          current = next;
          history.push(Expr::String(current.clone()));
        }
        return Ok(Expr::List(history.into()));
      }
    } else if let Expr::List(init_list) = &args[1] {
      // List mode: rules map elements to lists
      if let Expr::List(rule_list) = &args[0] {
        let raw_rules = extract_rules(rule_list);
        let mut rules: Vec<(String, Vec<Expr>)> = Vec::new();
        for (from, to) in &raw_rules {
          if let Expr::List(to_list) = to {
            rules.push((expr_to_string(from), to_list.to_vec()));
          }
        }
        let mut history: Vec<Expr> = vec![Expr::List(init_list.clone())];
        let mut current = init_list.clone();
        for _ in 0..n {
          let mut next: Vec<Expr> = Vec::new();
          for elem in &current {
            let elem_str = expr_to_string(elem);
            if let Some((_from, to)) =
              rules.iter().find(|(f, _)| *f == elem_str)
            {
              next.extend(to.clone());
            } else {
              next.push(elem.clone());
            }
          }
          current = next.into();
          history.push(Expr::List(current.clone()));
        }
        return Ok(Expr::List(history.into()));
      }
    }
  }

  // FileBaseName[path] — get file name without extension
  if name == "FileBaseName"
    && args.len() == 1
    && let Expr::String(path) = &args[0]
  {
    let base = std::path::Path::new(path)
      .file_stem()
      .and_then(|s| s.to_str())
      .unwrap_or("")
      .to_string();
    return Ok(Expr::String(base));
  }

  // FileExtension[path] — get file extension
  if name == "FileExtension"
    && args.len() == 1
    && let Expr::String(path) = &args[0]
  {
    let ext = std::path::Path::new(path)
      .extension()
      .and_then(|e| e.to_str())
      .unwrap_or("")
      .to_string();
    return Ok(Expr::String(ext));
  }

  // FileExistsQ[path] — check if file/directory exists
  if name == "FileExistsQ"
    && args.len() == 1
    && let Expr::String(path) = &args[0]
  {
    let exists = std::path::Path::new(path).exists();
    return Ok(bool_expr(exists));
  }

  // FileInformation[path] — wolframscript returns `{}` when the file does
  // not exist; a populated list of properties when it does. Match the
  // not-found behavior; defer the populated-list shape until needed.
  if name == "FileInformation"
    && args.len() == 1
    && let Expr::String(path) = &args[0]
    && !std::path::Path::new(path).exists()
  {
    return Ok(Expr::List(vec![].into()));
  }

  // DeleteFile[path] or DeleteFile[{path1, path2, …}] — delete files.
  // wolframscript emits `DeleteFile::fdnfnd: Directory or file "<name>"
  // not found.` on the first missing entry and returns the `$Failed`
  // symbol (not `$Failed[]`).
  if name == "DeleteFile" && args.len() == 1 {
    let paths: Option<Vec<String>> = match &args[0] {
      Expr::String(s) => Some(vec![s.clone()]),
      Expr::List(items) => items
        .iter()
        .map(|it| match it {
          Expr::String(s) => Some(s.clone()),
          _ => None,
        })
        .collect(),
      _ => None,
    };
    if let Some(paths) = paths {
      for path in &paths {
        if std::fs::remove_file(path).is_err() {
          crate::emit_message(&format!(
            "DeleteFile::fdnfnd: Directory or file \"{}\" not found.",
            path
          ));
          return Ok(Expr::Identifier("$Failed".to_string()));
        }
      }
      return Ok(Expr::Identifier("Null".to_string()));
    }
    // Non-string / non-list-of-strings argument: emit the
    // wolframscript-style type-error message and leave the call
    // unevaluated.
    let arg_str = crate::syntax::expr_to_string(&args[0]);
    crate::emit_message(&format!(
      "DeleteFile::strs: A string or nonempty list of strings is expected at position 1 in DeleteFile[{}].",
      arg_str
    ));
    return Ok(unevaluated("DeleteFile", args));
  }

  // RenameFile[source, dest] — rename/move a file
  if name == "RenameFile"
    && args.len() == 2
    && let Expr::String(source) = &args[0]
    && let Expr::String(dest) = &args[1]
  {
    match std::fs::rename(source, dest) {
      Ok(()) => return Ok(Expr::String(dest.clone())),
      Err(_e) => {
        // wolframscript reports the absolute path; resolve relative
        // paths against the current working directory so the message
        // matches exactly.
        let path = std::path::Path::new(source);
        let abs = if path.is_absolute() {
          source.clone()
        } else {
          std::env::current_dir()
            .map(|cwd| cwd.join(path).to_string_lossy().into_owned())
            .unwrap_or_else(|_| source.clone())
        };
        crate::emit_message(&format!(
          "RenameFile::fdnfnd: Directory or file \"{}\" not found.",
          abs
        ));
        return Ok(Expr::Identifier("$Failed".to_string()));
      }
    }
  }

  // RenameDirectory[source, dest] — rename/move a directory (wolframscript
  // also renames plain files with it). Returns the absolute destination
  // path; missing sources emit `fdnfnd` with the absolute path, existing
  // destinations emit `eexist` with the path as given, and non-string
  // arguments stay unevaluated without a message.
  if name == "RenameDirectory" && args.len() == 2 {
    if let (Expr::String(source), Expr::String(dest)) = (&args[0], &args[1]) {
      let to_abs = |p: &str| {
        let path = std::path::Path::new(p);
        if path.is_absolute() {
          p.to_string()
        } else {
          std::env::current_dir()
            .map(|cwd| cwd.join(path).to_string_lossy().into_owned())
            .unwrap_or_else(|_| p.to_string())
        }
      };
      if !std::path::Path::new(source).exists() {
        crate::emit_message(&format!(
          "RenameDirectory::fdnfnd: Directory or file \"{}\" not found.",
          to_abs(source)
        ));
        return Ok(Expr::Identifier("$Failed".to_string()));
      }
      if std::path::Path::new(dest).exists() {
        crate::emit_message(&format!(
          "RenameDirectory::eexist: {} already exists.",
          dest
        ));
        return Ok(Expr::Identifier("$Failed".to_string()));
      }
      match std::fs::rename(source, dest) {
        Ok(()) => return Ok(Expr::String(to_abs(dest))),
        Err(_) => return Ok(Expr::Identifier("$Failed".to_string())),
      }
    }
    return Ok(unevaluated("RenameDirectory", args));
  }

  // DeleteDirectory[path] — remove an empty directory. Mirror
  // wolframscript's error paths: non-string args emit `strs`, missing
  // directories emit `dirnf`, non-empty dirs emit `dirne`.
  if name == "DeleteDirectory" && args.len() == 1 {
    let path = match &args[0] {
      Expr::String(s) => s.clone(),
      _ => {
        let arg_str = crate::syntax::expr_to_string(&args[0]);
        crate::emit_message(&format!(
          "DeleteDirectory::strs: A string or nonempty list of strings is expected at position 1 in DeleteDirectory[{}].",
          arg_str
        ));
        return Ok(unevaluated("DeleteDirectory", args));
      }
    };
    let p = std::path::Path::new(&path);
    if !p.exists() {
      crate::emit_message(&format!(
        "DeleteDirectory::dirnf: Directory {} not found.",
        path
      ));
      return Ok(Expr::Identifier("$Failed".to_string()));
    }
    match std::fs::remove_dir(&path) {
      Ok(()) => return Ok(Expr::Identifier("Null".to_string())),
      Err(_) => {
        return Ok(Expr::Identifier("$Failed".to_string()));
      }
    }
  }

  // CopyFile[source, dest] — copy a file
  if name == "CopyFile"
    && args.len() == 2
    && let (Expr::String(source), Expr::String(dest)) = (&args[0], &args[1])
  {
    if !std::path::Path::new(source).exists() {
      // Match wolframscript's message, which reports the absolute path.
      let path = std::path::Path::new(source);
      let abs = if path.is_absolute() {
        source.clone()
      } else {
        std::env::current_dir()
          .map(|cwd| cwd.join(path).to_string_lossy().into_owned())
          .unwrap_or_else(|_| source.clone())
      };
      crate::emit_message(&format!(
        "CopyFile::fdnfnd: Directory or file \"{}\" not found.",
        abs
      ));
      return Ok(Expr::Identifier("$Failed".to_string()));
    }
    if std::path::Path::new(dest).exists() {
      crate::emit_message(&format!(
        "CopyFile::eexist: {} already exists.",
        dest
      ));
      return Ok(Expr::Identifier("$Failed".to_string()));
    }
    match std::fs::copy(source, dest) {
      Ok(_) => return Ok(Expr::String(dest.clone())),
      Err(e) => {
        crate::emit_message(&format!("CopyFile::failed: {}", e));
        return Ok(Expr::Identifier("$Failed".to_string()));
      }
    }
  }

  // CreateDirectory[path] — create a directory (including intermediate dirs)
  // CreateDirectory[] — create a temporary directory
  if name == "CreateDirectory" {
    if args.is_empty() {
      // Create a temporary directory
      match std::env::temp_dir()
        .to_str()
        .map(|t| format!("{}/woxi_{}", t, std::process::id()))
      {
        Some(tmp_path) => match std::fs::create_dir_all(&tmp_path) {
          Ok(()) => return Ok(Expr::String(tmp_path)),
          Err(e) => {
            crate::emit_message(&format!("CreateDirectory::failed: {}", e));
            return Ok(Expr::Identifier("$Failed".to_string()));
          }
        },
        None => {
          return Ok(Expr::Identifier("$Failed".to_string()));
        }
      }
    }
    if args.len() == 1 {
      if let Expr::String(path) = &args[0] {
        let p = std::path::Path::new(path);
        if p.exists() {
          crate::emit_message(&format!(
            "CreateDirectory::eexist: {} already exists.",
            path
          ));
          return Ok(Expr::Identifier("$Failed".to_string()));
        }
        match std::fs::create_dir_all(path) {
          Ok(()) => return Ok(Expr::String(path.clone())),
          Err(e) => {
            crate::emit_message(&format!("CreateDirectory::failed: {}", e));
            return Ok(Expr::Identifier("$Failed".to_string()));
          }
        }
      } else {
        // Non-string argument
        return Ok(Expr::Identifier("$Failed".to_string()));
      }
    }
  }

  // DeleteMissing[list] — remove Missing[] elements from a list
  if name == "DeleteMissing" && args.len() == 1 {
    match &args[0] {
      Expr::List(items) => {
        let filtered: Vec<Expr> = items
            .iter()
            .filter(|item| {
              !matches!(item, Expr::FunctionCall { name, .. } if name == "Missing")
            })
            .cloned()
            .collect();
        return Ok(Expr::List(filtered.into()));
      }
      // On an association, drop key->value pairs whose value is Missing[...].
      Expr::Association(pairs) => {
        let filtered: Vec<(Expr, Expr)> = pairs
            .iter()
            .filter(|(_k, v)| {
              !matches!(v, Expr::FunctionCall { name, .. } if name == "Missing")
            })
            .cloned()
            .collect();
        return Ok(Expr::Association(filtered));
      }
      Expr::FunctionCall {
        name: ds_name,
        args: ds_args,
      } if ds_name == "Dataset" && ds_args.len() == 3 => {
        // DeleteMissing[Dataset[data, type, meta]]
        if let Expr::List(items) = &ds_args[0] {
          let filtered: Vec<Expr> = items
              .iter()
              .filter(|item| {
                !matches!(item, Expr::FunctionCall { name, .. } if name == "Missing")
              })
              .cloned()
              .collect();
          // Recompute type with AnyLength instead of fixed count
          let type_expr = delete_missing_type(&ds_args[1]);
          return Ok(Expr::FunctionCall {
            name: "Dataset".to_string(),
            args: vec![
              Expr::List(filtered.into()),
              type_expr,
              ds_args[2].clone(),
            ]
            .into(),
          });
        }
        // Non-list data in Dataset: pass through
      }
      _ => {}
    }
  }

  // PiecewiseExpand — expand certain functions into Piecewise form
  if name == "PiecewiseExpand" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: fname,
      args: fargs,
    } = &args[0]
    {
      match fname.as_str() {
        "Min" if fargs.len() >= 2 => {
          let n = fargs.len();
          let mut cases = Vec::new();
          for i in 0..n - 1 {
            let mut conds = Vec::new();
            for j in 0..n {
              if i != j {
                conds.push(Expr::Comparison {
                  operands: vec![
                    Expr::BinaryOp {
                      op: BinaryOperator::Minus,
                      left: Box::new(fargs[i].clone()),
                      right: Box::new(fargs[j].clone()),
                    },
                    Expr::Integer(0),
                  ],
                  operators: vec![ComparisonOp::LessEqual],
                });
              }
            }
            let cond = if conds.len() == 1 {
              conds.pop().unwrap()
            } else {
              Expr::FunctionCall {
                name: "And".to_string(),
                args: conds.into(),
              }
            };
            cases.push((fargs[i].clone(), cond));
          }
          let default = fargs[n - 1].clone();
          let pw_cases = Expr::List(
            cases
              .into_iter()
              .map(|(val, cond)| Expr::List(vec![val, cond].into()))
              .collect(),
          );
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![pw_cases, default].into(),
          };
          return evaluate_expr_to_expr(&pw);
        }
        "Max" if fargs.len() >= 2 => {
          let n = fargs.len();
          let mut cases = Vec::new();
          for i in 0..n - 1 {
            let mut conds = Vec::new();
            for j in 0..n {
              if i != j {
                conds.push(Expr::Comparison {
                  operands: vec![
                    Expr::BinaryOp {
                      op: BinaryOperator::Minus,
                      left: Box::new(fargs[i].clone()),
                      right: Box::new(fargs[j].clone()),
                    },
                    Expr::Integer(0),
                  ],
                  operators: vec![ComparisonOp::GreaterEqual],
                });
              }
            }
            let cond = if conds.len() == 1 {
              conds.pop().unwrap()
            } else {
              Expr::FunctionCall {
                name: "And".to_string(),
                args: conds.into(),
              }
            };
            cases.push((fargs[i].clone(), cond));
          }
          let default = fargs[n - 1].clone();
          let pw_cases = Expr::List(
            cases
              .into_iter()
              .map(|(val, cond)| Expr::List(vec![val, cond].into()))
              .collect(),
          );
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![pw_cases, default].into(),
          };
          return evaluate_expr_to_expr(&pw);
        }
        "UnitStep" if fargs.len() == 1 => {
          let cond = Expr::Comparison {
            operands: vec![fargs[0].clone(), Expr::Integer(0)],
            operators: vec![ComparisonOp::GreaterEqual],
          };
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(
                vec![Expr::List(vec![Expr::Integer(1), cond].into())].into(),
              ),
              Expr::Integer(0),
            ]
            .into(),
          };
          return evaluate_expr_to_expr(&pw);
        }
        // Ramp[x] = x for x >= 0, else 0.
        "Ramp" if fargs.len() == 1 => {
          let cond = Expr::Comparison {
            operands: vec![fargs[0].clone(), Expr::Integer(0)],
            operators: vec![ComparisonOp::GreaterEqual],
          };
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(
                vec![Expr::List(vec![fargs[0].clone(), cond].into())].into(),
              ),
              Expr::Integer(0),
            ]
            .into(),
          };
          return evaluate_expr_to_expr(&pw);
        }
        // Boole[cond] = 1 when cond is True, else 0.
        "Boole" if fargs.len() == 1 => {
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(
                vec![Expr::List(
                  vec![Expr::Integer(1), fargs[0].clone()].into(),
                )]
                .into(),
              ),
              Expr::Integer(0),
            ]
            .into(),
          };
          return evaluate_expr_to_expr(&pw);
        }
        "Clip" if !fargs.is_empty() => {
          let x = fargs[0].clone();
          let (lo, hi) = if fargs.len() >= 2 {
            if let Expr::List(bounds) = &fargs[1] {
              if bounds.len() == 2 {
                (bounds[0].clone(), bounds[1].clone())
              } else {
                (Expr::Integer(-1), Expr::Integer(1))
              }
            } else {
              (Expr::Integer(-1), Expr::Integer(1))
            }
          } else {
            (Expr::Integer(-1), Expr::Integer(1))
          };
          let cond_lo = Expr::Comparison {
            operands: vec![x.clone(), lo.clone()],
            operators: vec![ComparisonOp::Less],
          };
          let cond_hi = Expr::Comparison {
            operands: vec![x.clone(), hi.clone()],
            operators: vec![ComparisonOp::Greater],
          };
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![
              Expr::List(
                vec![
                  Expr::List(vec![lo, cond_lo].into()),
                  Expr::List(vec![hi, cond_hi].into()),
                ]
                .into(),
              ),
              x,
            ]
            .into(),
          };
          return evaluate_expr_to_expr(&pw);
        }
        _ => {}
      }
    }
    // For functions that can't be expanded into Piecewise form,
    // return the argument unchanged (not wrapped in PiecewiseExpand).
    return Ok(args[0].clone());
  }

  // AdjacencyGraph[matrix] or AdjacencyGraph[vertices, matrix] — create graph from adjacency matrix
  if name == "AdjacencyGraph" && (args.len() == 1 || args.len() == 2) {
    let (vertices, matrix) = if args.len() == 2 {
      if let Expr::List(verts) = &args[0] {
        (Some(verts.clone()), &args[1])
      } else {
        return Ok(unevaluated(name, args));
      }
    } else {
      (None, &args[0])
    };

    if let Expr::List(rows) = matrix {
      let n = rows.len();
      let verts: Vec<Expr> = vertices
        .unwrap_or_else(|| (1..=n).map(|i| Expr::Integer(i as i128)).collect())
        .to_vec();

      // Check if symmetric (undirected)
      let mut is_symmetric = true;
      let mut matrix_vals: Vec<Vec<i128>> = Vec::new();
      for row in rows {
        if let Expr::List(cols) = row {
          let vals: Vec<i128> = cols
            .iter()
            .map(|c| match c {
              Expr::Integer(v) => *v,
              _ => 0,
            })
            .collect();
          matrix_vals.push(vals);
        }
      }
      for i in 0..n {
        for j in 0..n {
          if i < matrix_vals.len()
            && j < matrix_vals[i].len()
            && j < matrix_vals.len()
            && i < matrix_vals[j].len()
            && matrix_vals[i][j] != matrix_vals[j][i]
          {
            is_symmetric = false;
          }
        }
      }

      let mut edges = Vec::new();
      let edge_name = if is_symmetric {
        "UndirectedEdge"
      } else {
        "DirectedEdge"
      };
      for i in 0..n {
        let start = if is_symmetric { i + 1 } else { 0 };
        for j in start..n {
          if i < matrix_vals.len()
            && j < matrix_vals[i].len()
            && matrix_vals[i][j] != 0
          {
            edges.push(Expr::FunctionCall {
              name: edge_name.to_string(),
              args: vec![verts[i].clone(), verts[j].clone()].into(),
            });
          }
        }
      }

      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(verts.into()), Expr::List(edges.into())].into(),
      });
    }
  }

  // MovingAverage[list, r] — simple moving average with window size r.
  // MovingAverage[list, {w1, ..., wr}] — weighted moving average with
  // weights wi over a sliding window; denominator is Sum[wi].
  if name == "MovingAverage"
    && args.len() == 2
    && let Expr::List(items) = &args[0]
  {
    let n = items.len();
    let (window, divisor) = match &args[1] {
      Expr::Integer(r) if *r >= 1 => {
        // Equivalent to weights {1, 1, ..., 1} of length r.
        (vec![Expr::Integer(1); *r as usize], Expr::Integer(*r))
      }
      Expr::List(weights) if !weights.is_empty() => {
        let sum = evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Plus".to_string(),
          args: weights.iter().cloned().collect::<Vec<_>>().into(),
        })?;
        (weights.iter().cloned().collect::<Vec<_>>(), sum)
      }
      _ => {
        return Ok(unevaluated(name, args));
      }
    };
    let r = window.len();
    if r > n {
      crate::emit_message(&format!(
        "MovingAverage::arg2: The second argument {} must be a positive integer less than or equal to the length {} of the first argument, or a vector of length less than or equal to the length of the first argument.",
        r, n
      ));
      return Ok(unevaluated(name, args));
    }
    let mut result = Vec::with_capacity(n - r + 1);
    for i in 0..=(n - r) {
      // Build Sum[w_j * items[i + j]] for j = 0..r, then divide by divisor.
      let mut terms: Vec<Expr> = Vec::with_capacity(r);
      for j in 0..r {
        terms.push(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(window[j].clone()),
          right: Box::new(items[i + j].clone()),
        });
      }
      let sum = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      };
      let avg = Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(sum),
        right: Box::new(divisor.clone()),
      };
      result.push(evaluate_expr_to_expr(&avg)?);
    }
    return Ok(Expr::List(result.into()));
  }

  // BSplineBasis[d, i, x] — the i-th uniform B-spline basis function of degree
  // d, and its 2-argument shorthand BSplineBasis[d, x] (i = 0). On the default
  // uniform knots this equals a shifted centered cardinal B-spline:
  //   BSplineBasis[d, i, x] = CardinalBSplineBasis[d, (d + 1) (x - 1/2) - i]
  // (verified against wolframscript). Delegating to CardinalBSplineBasis keeps
  // the exact-input path (BSplineBasis[3, 0, 1/2] = 2/3) and the exact integer
  // 0 outside the support. Index i must be a non-negative integer (i > d is
  // allowed and gives 0); a negative i emits invidx and stays unevaluated.
  if name == "BSplineBasis"
    && (args.len() == 2 || args.len() == 3)
    && let Expr::Integer(d) = &args[0]
    && *d >= 0
  {
    let x_arg = args.last().unwrap();
    let i = if args.len() == 3 {
      match crate::functions::math_ast::expr_to_i128(&args[1]) {
        Some(i) if i >= 0 => i,
        Some(i) => {
          crate::emit_message(&format!(
            "BSplineBasis::invidx: Index {i} should be a non-negative machine-sized integer."
          ));
          return Ok(unevaluated("BSplineBasis", args));
        }
        None => return Ok(unevaluated("BSplineBasis", args)),
      }
    } else {
      0
    };
    // A symbolic coordinate stays unevaluated.
    if crate::functions::math_ast::try_eval_to_f64(x_arg).is_none() {
      return Ok(unevaluated("BSplineBasis", args));
    }
    // arg = (d + 1) * (x - 1/2) - i
    let half = Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(-1), Expr::Integer(2)].into(),
    };
    let x_shift = crate::functions::math_ast::plus_ast(&[x_arg.clone(), half])?;
    let scaled =
      crate::functions::math_ast::times_ast(&[Expr::Integer(d + 1), x_shift])?;
    let arg =
      crate::functions::math_ast::plus_ast(&[scaled, Expr::Integer(-i)])?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "CardinalBSplineBasis".to_string(),
      args: vec![Expr::Integer(*d), arg].into(),
    });
  }

  // TimeValue[s, i, t] — time value of a present sum s at interest rate
  // i (per period) for t periods. Future value (t > 0) or present value
  // (t < 0): result = s * (1 + i)^t.
  if name == "TimeValue" && args.len() == 3 {
    let (s, i, t) = (&args[0], &args[1], &args[2]);
    // AnnuityDue pays at the start of each interval, so its value is the
    // annuity-immediate value grown by one payment interval:
    //   TimeValue[AnnuityDue[p, t, q], i, t0]
    //     = (1+i)^q * TimeValue[Annuity[p, t, q], i, t0]   (q defaults to 1)
    if let Expr::FunctionCall {
      name: ann_name,
      args: ann_args,
    } = s
      && ann_name == "AnnuityDue"
      && (ann_args.len() == 2 || ann_args.len() == 3)
    {
      let q = ann_args.get(2).cloned().unwrap_or(Expr::Integer(1));
      let ordinary = Expr::FunctionCall {
        name: "TimeValue".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Annuity".to_string(),
            args: ann_args.clone(),
          },
          args[1].clone(),
          args[2].clone(),
        ]
        .into(),
      };
      let due = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![Expr::Integer(1), args[1].clone()].into(),
              },
              q,
            ]
            .into(),
          },
          ordinary,
        ]
        .into(),
      };
      let result = evaluate_expr_to_expr(&due)?;
      // Only commit when the inner annuity actually evaluated.
      if !expr_to_string(&result).contains("Annuity") {
        return Ok(result);
      }
      return Ok(unevaluated("TimeValue", args));
    }
    let s_scalar =
      matches!(s, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
        || matches!(s, Expr::FunctionCall { name, .. } if name == "Rational");
    let i_scalar =
      matches!(i, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
        || matches!(i, Expr::FunctionCall { name, .. } if name == "Rational");
    let t_scalar =
      matches!(t, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
        || matches!(t, Expr::FunctionCall { name, .. } if name == "Rational");
    // TimeValue[s, i, {date_later, date_earlier}] — date_later/date_earlier
    // are 3-element {y,m,d} lists. Convert to (date_later - date_earlier)
    // in years (using a simple 365.25-day proxy) and fall through to the
    // scalar formula.
    let date_diff_periods: Option<Expr> = if let Expr::List(dates) = t
      && dates.len() == 2
    {
      fn date_to_days(d: &Expr) -> Option<f64> {
        let Expr::List(parts) = d else {
          return None;
        };
        if parts.is_empty() || parts.len() > 6 {
          return None;
        }
        let to_f = |e: &Expr| -> Option<f64> {
          match e {
            Expr::Integer(n) => Some(*n as f64),
            Expr::Real(f) => Some(*f),
            _ => None,
          }
        };
        let y = to_f(&parts[0])?;
        let m = if parts.len() >= 2 {
          to_f(&parts[1])?
        } else {
          1.0
        };
        let dd = if parts.len() >= 3 {
          to_f(&parts[2])?
        } else {
          1.0
        };
        // Days from year 0 (rough; only the difference matters).
        Some(y * 365.25 + (m - 1.0) * 30.4375 + (dd - 1.0))
      }
      let d1 = date_to_days(&dates[0]);
      let d2 = date_to_days(&dates[1]);
      if let (Some(a), Some(b)) = (d1, d2) {
        Some(Expr::Real((a - b) / 365.25))
      } else {
        None
      }
    } else {
      None
    };
    let t_for_formula = date_diff_periods.as_ref().unwrap_or(t);
    let t_is_usable = matches!(
      t_for_formula,
      Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_)
    ) || matches!(
      t_for_formula,
      Expr::FunctionCall { name, .. } if name == "Rational"
    );
    if s_scalar && i_scalar && t_is_usable {
      let value = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          s.clone(),
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![Expr::Integer(1), i.clone()].into(),
              },
              t_for_formula.clone(),
            ]
            .into(),
          },
        ]
        .into(),
      };
      return evaluate_expr_to_expr(&value);
    }

    // TimeValue[Annuity[{p, ip, fp}, tspan (, q)], i, t] — an annuity whose
    // first argument is a list carries, in addition to the level payment p,
    // an initial payment `ip` made at time 0 and a final payment `fp` made at
    // time tspan (a length-2 list omits `fp`). Its value at time t is
    //   V_t = (PV_annuity + ip + fp*(1+i)^-tspan) * (1+i)^t
    // where PV_annuity is the level-annuity present value using interval q
    // (q defaults to 1). Lists of length other than 2 or 3 stay unevaluated
    // (matching wolframscript). Handled before the scalar-payment blocks so
    // the list isn't threaded elementwise through the annuity formula.
    if let Expr::FunctionCall {
      name: ann_name,
      args: ann_args,
    } = s
      && ann_name == "Annuity"
      && (ann_args.len() == 2 || ann_args.len() == 3)
      && i_scalar
      && t_scalar
      && let Expr::List(pay) = &ann_args[0]
      && (pay.len() == 2 || pay.len() == 3)
    {
      let p = pay[0].clone();
      let ip = pay[1].clone();
      let fp = pay.get(2).cloned().unwrap_or(Expr::Integer(0));
      let tspan = ann_args[1].clone();
      let q = ann_args.get(2).cloned().unwrap_or(Expr::Integer(1));
      let one_plus_i = || Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![Expr::Integer(1), i.clone()].into(),
      };
      // (1+i)^-tspan
      let pow_neg_tspan = || Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          one_plus_i(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), tspan.clone()].into(),
          },
        ]
        .into(),
      };
      // i_eff = (1+i)^q - 1
      let i_eff = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![one_plus_i(), q].into(),
          },
          Expr::Integer(-1),
        ]
        .into(),
      };
      // PV_annuity = p * (1 - (1+i)^-tspan) / i_eff
      let pv_annuity = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          p,
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![
              Expr::Integer(1),
              Expr::FunctionCall {
                name: "Times".to_string(),
                args: vec![Expr::Integer(-1), pow_neg_tspan()].into(),
              },
            ]
            .into(),
          },
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![i_eff, Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      };
      // fp * (1+i)^-tspan  (final payment discounted from time tspan)
      let fp_term = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![fp, pow_neg_tspan()].into(),
      };
      // PV_0 = PV_annuity + ip + fp_term
      let pv0 = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![pv_annuity, ip, fp_term].into(),
      };
      // V_t = PV_0 * (1+i)^t
      let result = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          pv0,
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![one_plus_i(), t.clone()].into(),
          },
        ]
        .into(),
      };
      return evaluate_expr_to_expr(&result);
    }

    // TimeValue[Annuity[pmt, n], i, t] — present value (t = 0) or general
    // time-shifted value of an n-period level annuity with payment pmt at
    // each period and interest rate i per period.
    //   PV  = pmt * (1 - (1+i)^-n) / i   (annuity-immediate)
    //   V_t = PV * (1+i)^t
    if let Expr::FunctionCall {
      name: ann_name,
      args: ann_args,
    } = s
      && ann_name == "Annuity"
      && ann_args.len() == 2
      && !matches!(&ann_args[0], Expr::List(_))
      && i_scalar
      && t_scalar
    {
      let pmt = ann_args[0].clone();
      let n = ann_args[1].clone();
      // (1 + i)^-n
      let pow_neg_n = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: vec![Expr::Integer(1), i.clone()].into(),
          },
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), n].into(),
          },
        ]
        .into(),
      };
      // 1 - (1+i)^-n
      let numer = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Integer(1),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), pow_neg_n].into(),
          },
        ]
        .into(),
      };
      // PV = pmt * numer / i
      let pv = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          pmt,
          numer,
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![i.clone(), Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      };
      // V_t = PV * (1+i)^t
      let result = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          pv,
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: vec![Expr::Integer(1), i.clone()].into(),
              },
              t.clone(),
            ]
            .into(),
          },
        ]
        .into(),
      };
      return evaluate_expr_to_expr(&result);
    }

    // TimeValue[Annuity[pmt, tspan, q], i, t] — a level annuity with payment
    // pmt made at the end of every payment interval q over a total time span
    // tspan, valued with interest rate i per unit time. There are n = tspan/q
    // payments and the effective rate over one interval is i_eff = (1+i)^q - 1.
    // Because q*n = tspan, (1+i_eff)^-n = (1+i)^-tspan, so:
    //   PV  = pmt * (1 - (1+i)^-tspan) / ((1+i)^q - 1)   (annuity-immediate)
    //   V_t = PV * (1+i)^t
    // The 2-arg case is the q = 1 specialization of this formula.
    if let Expr::FunctionCall {
      name: ann_name,
      args: ann_args,
    } = s
      && ann_name == "Annuity"
      && ann_args.len() == 3
      && !matches!(&ann_args[0], Expr::List(_))
      && i_scalar
      && t_scalar
    {
      let pmt = ann_args[0].clone();
      let tspan = ann_args[1].clone();
      let q = ann_args[2].clone();
      let one_plus_i = || Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![Expr::Integer(1), i.clone()].into(),
      };
      // (1+i)^-tspan
      let pow_neg_tspan = Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![
          one_plus_i(),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), tspan].into(),
          },
        ]
        .into(),
      };
      // 1 - (1+i)^-tspan
      let numer = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Integer(1),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), pow_neg_tspan].into(),
          },
        ]
        .into(),
      };
      // i_eff = (1+i)^q - 1
      let i_eff = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![one_plus_i(), q].into(),
          },
          Expr::Integer(-1),
        ]
        .into(),
      };
      // PV = pmt * numer / i_eff
      let pv = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          pmt,
          numer,
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![i_eff, Expr::Integer(-1)].into(),
          },
        ]
        .into(),
      };
      // V_t = PV * (1+i)^t
      let result = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          pv,
          Expr::FunctionCall {
            name: "Power".to_string(),
            args: vec![one_plus_i(), t.clone()].into(),
          },
        ]
        .into(),
      };
      return evaluate_expr_to_expr(&result);
    }

    // TimeValue[Cashflow[list], i, t].
    //   Cashflow[{c0, c1, ...}]          — amount c_k at time k, so the value
    //                                      is Sum[c_k * (1+i)^(t-k)].
    //   Cashflow[{{t0,c0}, {t1,c1}, ...}] — amount c_k at explicit time t_k,
    //                                      so the value is Sum[c_k * (1+i)^(t-t_k)].
    // A list whose elements are 2-element {time, amount} pairs must use the
    // explicit-time form; feeding a pair into the scalar formula would thread
    // it elementwise and return a list of values.
    if let Expr::FunctionCall {
      name: cf_name,
      args: cf_args,
    } = s
      && cf_name == "Cashflow"
      && cf_args.len() == 1
      && i_scalar
      && t_scalar
      && let Expr::List(flows) = &cf_args[0]
    {
      let is_pair = |e: &Expr| matches!(e, Expr::List(p) if p.len() == 2);
      let all_pairs = !flows.is_empty() && flows.iter().all(is_pair);
      let any_list = flows.iter().any(|e| matches!(e, Expr::List(_)));
      // Mixed or malformed sublist shapes are not valid cashflow specs; leave
      // the call unevaluated rather than producing a spurious list.
      if any_list && !all_pairs {
        return Ok(unevaluated("TimeValue", args));
      }
      let mut terms: Vec<Expr> = Vec::with_capacity(flows.len());
      for (k, c) in flows.iter().enumerate() {
        // time_k and amount_k depend on whether this is a {time, amount} pair.
        let (time_k, amount) = if all_pairs {
          match c {
            Expr::List(p) => (p[0].clone(), p[1].clone()),
            _ => unreachable!(),
          }
        } else {
          (Expr::Integer(k as i128), c.clone())
        };
        // exponent = t - time_k
        let exp = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            t.clone(),
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![Expr::Integer(-1), time_k].into(),
            },
          ]
          .into(),
        };
        terms.push(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            amount,
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![Expr::Integer(1), i.clone()].into(),
                },
                exp,
              ]
              .into(),
            },
          ]
          .into(),
        });
      }
      let sum = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms.into(),
      };
      return evaluate_expr_to_expr(&sum);
    }
    // TimeValue[s, {r1, r2, ..., rn}, t] with non-negative integer t and a
    // flat list of per-period rates: result = s * Prod_{k=1..t} (1 + r_k)
    // where r_k = rates[min(k, n) - 1] (the last rate is repeated when t > n).
    if s_scalar
      && let Expr::List(rates) = i
      && !rates.is_empty()
      && rates.iter().all(|r| {
        matches!(r, Expr::Integer(_) | Expr::Real(_))
          || matches!(r, Expr::FunctionCall { name, .. } if name == "Rational")
      })
      && let Expr::Integer(t_int) = t
      && *t_int >= 0
    {
      let t_usize = *t_int as usize;
      let n = rates.len();
      let mut factors: Vec<Expr> = Vec::with_capacity(t_usize + 1);
      factors.push(s.clone());
      for k in 1..=t_usize {
        let idx = (k - 1).min(n - 1);
        factors.push(Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![Expr::Integer(1), rates[idx].clone()].into(),
        });
      }
      let prod = Expr::FunctionCall {
        name: "Times".to_string(),
        args: factors.into(),
      };
      return evaluate_expr_to_expr(&prod);
    }

    // TimeValue[s, {{t_1, r_1}, {t_2, r_2}, ...}, t] — list of per-period
    // {time, rate} pairs. Each pair {t_k, r_k} says the rate r_k applies
    // for the period ending at time t_k. Going from t=0 backward to t<0:
    //   value(t) = s / Product[(1+r_k) for k where t_k > t]
    if s_scalar
      && let Expr::List(rates) = i
      && !rates.is_empty()
      && rates.iter().all(|r| {
        if let Expr::List(pair) = r
          && pair.len() == 2
        {
          let ok_t = matches!(&pair[0], Expr::Integer(_) | Expr::Real(_))
            || matches!(&pair[0], Expr::FunctionCall { name, .. } if name == "Rational");
          let ok_r = matches!(&pair[1], Expr::Integer(_) | Expr::Real(_))
            || matches!(&pair[1], Expr::FunctionCall { name, .. } if name == "Rational");
          ok_t && ok_r
        } else {
          false
        }
      })
      && let Some(t_f) = crate::functions::math_ast::try_eval_to_f64(t)
    {
      let mut factors: Vec<Expr> = vec![s.clone()];
      for r in rates.iter() {
        if let Expr::List(pair) = r {
          let tk =
            crate::functions::math_ast::try_eval_to_f64(&pair[0]).unwrap();
          // Apply this rate's discount factor for every pair whose
          // period boundary lies within [t, 0], i.e. tk >= t (with
          // the rate pair {tk, rk} representing the rate active for
          // the period ending at time tk).
          if tk >= t_f {
            factors.push(Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![Expr::Integer(1), pair[1].clone()].into(),
                },
                Expr::Integer(-1),
              ]
              .into(),
            });
          }
        }
      }
      let prod = Expr::FunctionCall {
        name: "Times".to_string(),
        args: factors.into(),
      };
      return evaluate_expr_to_expr(&prod);
    }

    // TimeValue[s, {t_1 -> r_1, ...}, {t_start, t_end}] — yield curve as
    // a rule list (r_k = spot rate at maturity t_k). Result =
    //   s * (1 + r(t_end - t_start))^-(t_end - t_start)
    // where r(maturity) is linearly interpolated between adjacent nodes.
    if s_scalar
      && let Expr::List(rules) = i
      && !rules.is_empty()
      && let Expr::List(span) = t
      && span.len() == 2
    {
      let mut nodes: Vec<(f64, f64)> = Vec::new();
      let mut ok = true;
      for r in rules.iter() {
        match r {
          Expr::Rule {
            pattern,
            replacement,
          } => {
            let tk = crate::functions::math_ast::try_eval_to_f64(pattern);
            let rk = crate::functions::math_ast::try_eval_to_f64(replacement);
            if let (Some(t_v), Some(r_v)) = (tk, rk) {
              nodes.push((t_v, r_v));
            } else {
              ok = false;
              break;
            }
          }
          _ => {
            ok = false;
            break;
          }
        }
      }
      let t_start = crate::functions::math_ast::try_eval_to_f64(&span[0]);
      let t_end = crate::functions::math_ast::try_eval_to_f64(&span[1]);
      if ok && let (Some(t0), Some(t1)) = (t_start, t_end) {
        nodes.sort_by(|a, b| {
          a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        let maturity = t1 - t0;
        // Linearly interpolate the spot rate at `maturity` from the nodes.
        let rate = if let Some(exact) =
          nodes.iter().find(|n| (n.0 - maturity).abs() < 1e-12)
        {
          exact.1
        } else if maturity <= nodes[0].0 {
          nodes[0].1
        } else if maturity >= nodes.last().unwrap().0 {
          nodes.last().unwrap().1
        } else {
          let mut interp = nodes[0].1;
          for w in nodes.windows(2) {
            let (ta, ra) = w[0];
            let (tb, rb) = w[1];
            if maturity >= ta && maturity <= tb {
              let frac = (maturity - ta) / (tb - ta);
              interp = ra + frac * (rb - ra);
              break;
            }
          }
          interp
        };
        let value = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            s.clone(),
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Plus".to_string(),
                  args: vec![Expr::Integer(1), Expr::Real(rate)].into(),
                },
                Expr::Real(-maturity),
              ]
              .into(),
            },
          ]
          .into(),
        };
        return evaluate_expr_to_expr(&value);
      }
    }
  }

  // CirclePoints[n] — n equally spaced points on the unit circle
  // AngleVector[theta] → {Cos[theta], Sin[theta]}
  // AngleVector[{r, theta}] → {r*Cos[theta], r*Sin[theta]}
  // AngleVector[{x, y}, theta] → {x + Cos[theta], y + Sin[theta]}
  // AngleVector[{x, y}, {r, theta}] → {x + r*Cos[theta], y + r*Sin[theta]}
  if name == "AngleVector" && (args.len() == 1 || args.len() == 2) {
    let (center, r, theta) = if args.len() == 1 {
      match &args[0] {
        Expr::List(items) if items.len() == 2 => {
          // AngleVector[{r, theta}]
          (None, Some(items[0].clone()), items[1].clone())
        }
        other => {
          // AngleVector[theta]
          (None, None, other.clone())
        }
      }
    } else {
      // args.len() == 2
      let center = match &args[0] {
        Expr::List(items) if items.len() == 2 => {
          Some((items[0].clone(), items[1].clone()))
        }
        _ => {
          return Ok(unevaluated(name, args));
        }
      };
      let (r, theta) = match &args[1] {
        Expr::List(items) if items.len() == 2 => {
          (Some(items[0].clone()), items[1].clone())
        }
        other => (None, other.clone()),
      };
      (center, r, theta)
    };

    let cos_expr = evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![theta.clone()].into(),
    })?;
    let sin_expr = evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![theta].into(),
    })?;

    // Apply radius if present
    let (x_comp, y_comp) = if let Some(r) = r {
      let rx = evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(r.clone()),
        right: Box::new(cos_expr),
      })?;
      let ry = evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(r),
        right: Box::new(sin_expr),
      })?;
      (rx, ry)
    } else {
      (cos_expr, sin_expr)
    };

    // Apply center offset if present
    let (final_x, final_y) = if let Some((cx, cy)) = center {
      let fx = evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(cx),
        right: Box::new(x_comp),
      })?;
      let fy = evaluate_expr_to_expr(&Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(cy),
        right: Box::new(y_comp),
      })?;
      (fx, fy)
    } else {
      (x_comp, y_comp)
    };

    return Ok(Expr::List(vec![final_x, final_y].into()));
  }

  // CirclePoints in its several forms:
  //   CirclePoints[n]                 — n points on the unit circle
  //   CirclePoints[r, n]              — radius r, default starting angle
  //   CirclePoints[{r, theta}, n]     — radius r, explicit starting angle
  //   CirclePoints[{cx, cy}, ..., n]  — the above translated to a center
  if name == "CirclePoints" && (1..=3).contains(&args.len()) {
    let unevaluated = || Ok(unevaluated("CirclePoints", args));
    // n is always the last argument and must be a positive integer.
    let n = match args.last() {
      Some(Expr::Integer(n)) if *n >= 1 => *n as usize,
      _ => return unevaluated(),
    };
    // Center (defaults to the origin) when the first arg is a coordinate pair
    // and there are 3 arguments total.
    let (center_x, center_y, radius_spec): (Expr, Expr, Option<&Expr>) =
      if args.len() == 3 {
        match &args[0] {
          Expr::List(c) if c.len() == 2 => {
            (c[0].clone(), c[1].clone(), Some(&args[1]))
          }
          _ => return unevaluated(),
        }
      } else {
        (Expr::Integer(0), Expr::Integer(0), args.first())
      };
    // Resolve radius and (optional explicit) starting angle.
    let (radius, theta): (Expr, Option<Expr>) = match radius_spec {
      // {r, theta}
      Some(Expr::List(rt)) if rt.len() == 2 => {
        (rt[0].clone(), Some(rt[1].clone()))
      }
      // scalar radius, default angle (only valid when given alongside n,
      // i.e. 2- or 3-arg forms)
      Some(r) if args.len() >= 2 && !matches!(r, Expr::List(_)) => {
        (r.clone(), None)
      }
      // bare CirclePoints[n]: unit radius, default angle
      _ if args.len() == 1 => (Expr::Integer(1), None),
      _ => return unevaluated(),
    };
    // Base angle theta0: explicit, or the default Pi/2 - (n-1)*Pi/n.
    let pi = || Expr::Identifier("Pi".to_string());
    let base = theta.unwrap_or_else(|| Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(pi()),
        right: Box::new(Expr::Integer(2)),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer((n - 1) as i128)),
          right: Box::new(pi()),
        }),
        right: Box::new(Expr::Integer(n as i128)),
      }),
    });
    let mut points = Vec::with_capacity(n);
    for k in 0..n {
      // angle_k = base + 2*k*Pi/n
      let angle = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(base.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(k as i128 * 2)),
            right: Box::new(pi()),
          }),
          right: Box::new(Expr::Integer(n as i128)),
        }),
      };
      // coordinate = center + radius * trig(angle)
      let coord = |trig: &str, c: &Expr| Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(c.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(radius.clone()),
          right: Box::new(Expr::FunctionCall {
            name: trig.to_string(),
            args: vec![angle.clone()].into(),
          }),
        }),
      };
      let x = evaluate_expr_to_expr(&coord("Cos", &center_x))?;
      let y = evaluate_expr_to_expr(&coord("Sin", &center_y))?;
      points.push(Expr::List(vec![x, y].into()));
    }
    return Ok(Expr::List(points.into()));
  }

  // Key[k] is an operator form — return unevaluated (applied via CurriedCall)
  if name == "Key" {
    return Ok(unevaluated(name, args));
  }

  // Darker/Lighter fallback: return unevaluated if color couldn't be resolved
  if name == "Darker" || name == "Lighter" {
    return Ok(unevaluated(name, args));
  }

  // FunctionInterpolation[expr, {x, xmin, xmax}]
  if name == "FunctionInterpolation" && args.len() >= 2 {
    return function_interpolation_ast(args);
  }

  // PermutationProduct[Cycles[…], Cycles[…], …]
  // Compose permutations left-to-right (σ_1 applied first).
  if name == "PermutationProduct" {
    if args.len() == 1 {
      return Ok(args[0].clone());
    }
    // Permutation-list arguments compose to a permutation list; Cycles
    // arguments compose to a Cycles object.
    if let Some(result) = compose_permutation_lists(args) {
      return Ok(result);
    }
    if let Some(result) = compose_cycles_args(args) {
      return Ok(result);
    }
  }

  // Functions that return empty list
  if matches!(name, "SyntaxInformation" | "LaunchKernels") {
    return Ok(Expr::List(vec![].into()));
  }

  // FilePrint[path] — print the contents of a file
  if name == "FilePrint"
    && args.len() == 1
    && let Expr::String(path) = &args[0]
  {
    match std::fs::read_to_string(path) {
      Ok(contents) => {
        if !crate::is_quiet_print() {
          print!("{}", contents);
        }
        crate::capture_stdout(contents.trim_end_matches('\n'));
        return Ok(Expr::Identifier("Null".to_string()));
      }
      Err(_) => {
        crate::emit_message(&format!("General::noopen: Cannot open {}.", path));
        return Ok(unevaluated("FilePrint", args));
      }
    }
  }

  // FileType[path] — return the type of a file/directory
  if name == "FileType"
    && args.len() == 1
    && let Expr::String(path) = &args[0]
  {
    let p = std::path::Path::new(path);
    let result = if !p.exists() {
      "None"
    } else if p.is_dir() {
      "Directory"
    } else if p.is_file() {
      "File"
    } else {
      "Special"
    };
    return Ok(Expr::Identifier(result.to_string()));
  }

  // DirectoryQ[path] — check if path is a directory
  if name == "DirectoryQ" && args.len() == 1 {
    if let Expr::String(path) = &args[0] {
      let is_dir = std::path::Path::new(path).is_dir();
      return Ok(bool_expr(is_dir));
    }
    return Ok(bool_expr(false));
  }

  // FirstCase[x, y] returns Missing["NotFound"] when x is not a list
  if name == "FirstCase"
    && args.len() == 2
    && !matches!(&args[0], Expr::List(_))
  {
    return Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![Expr::String("NotFound".to_string())].into(),
    });
  }

  // Neural network layer/model functions return $Failed for invalid arguments
  if matches!(name, "TotalLayer" | "NetEncoder") {
    return Ok(Expr::Identifier("$Failed".to_string()));
  }

  // Morphological operations: Opening, Closing, Erosion, Dilation —
  // require a kernel argument; emit Wolfram-style `argr` when missing.
  if matches!(name, "Opening" | "Closing" | "Erosion" | "Dilation")
    && args.len() == 1
  {
    crate::emit_message(&format!(
      "{}::argr: {} called with 1 argument; 2 arguments are expected.",
      name, name
    ));
    return Ok(unevaluated(name, args));
  }
  if matches!(name, "Opening" | "Closing" | "Erosion" | "Dilation")
    && args.len() == 2
  {
    if let Some(result) = morphological_op(name, &args[0], &args[1]) {
      return result;
    }
    // First arg isn't a list or image — match wolframscript's arg1 warning.
    if !matches!(&args[0], Expr::List(_) | Expr::Image { .. }) {
      crate::emit_message(&format!(
        "{}::arg1: The first argument {} should be a rectangular array, image or video.",
        name,
        crate::syntax::expr_to_string(&args[0])
      ));
      return Ok(unevaluated(name, args));
    }
  }

  // Formatting wrappers and symbolic heads that stay unevaluated
  if matches!(
    name,
    "DecimalForm"
      | "Proportional"
      | "NetGraph"
      | "AccountingForm"
      | "Before"
      | "ComplexityFunction"
      | "CompilationOptions"
      | "AbsoluteDashing"
      | "DataReversed"
      | "AxesEdge"
      | "TaggingRules"
      | "Rationals"
      | "File"
      | "BodePlot"
      | "DelaunayMesh"
      | "ComplexRegionPlot"
      | "LongRightArrow"
      | "GeoGridPosition"
      | "OpenerView"
      | "Ellipsoid"
      | "MaxStepSize"
      | "RadioButtonBar"
      | "StepMonitor"
      | "Thumbnail"
      | "TransformedRegion"
      | "HypoexponentialDistribution"
      | "FormulaLookup"
      | "HalfLine"
      | "LocatorAutoCreate"
      | "LegendFunction"
      | "RasterSize"
      | "TransformedField"
      | "GeoProjection"
      | "SoundVolume"
      | "GradientFilter"
      | "RegionNearest"
      | "TildeTilde"
      | "NotebookClose"
      | "Failure"
      | "Annuity"
      | "AnnuityDue"
      | "Cashflow"
      | "LineIndent"
      | "LayeredGraphPlot"
      | "WordCharacter"
      | "ReflectionTransform"
      | "ParameterMixtureDistribution"
      | "FindDistributionParameters"
      | "FindPath"
      | "FindPeaks"
      | "NProbability"
      | "Weights"
      | "WhitespaceCharacter"
      | "VerticalSlider"
      | "CycleGraph"
      | "OverDot"
      | "MaxPlotPoints"
      | "AnimationRepetitions"
      | "ARMAProcess"
      | "UndoTrackedVariables"
      | "VectorColorFunction"
      | "NotebookGet"
      | "Visible"
      | "TruncatedDistribution"
      | "CoefficientRules"
      | "PowersRepresentations"
      | "BarnesG"
      | "SequenceCases"
      | "EndOfLine"
      | "RowLines"
      | "DeleteContents"
      | "ColumnSpacings"
      | "CriterionFunction"
      | "IntervalMarkers"
      | "AnyOrder"
      | "IntervalMarkersStyle"
      | "HatchFilling"
      | "IncludeConstantBasis"
      | "HeaderLines"
      | "SelfLoopStyle"
      | "ScaleDivisions"
      | "ColumnAlignments"
      | "ExtentElementFunction"
      | "Subset"
      | "TargetUnits"
      | "RowSpacings"
      | "PassEventsUp"
      | "NormalsFunction"
      | "StartOfLine"
      | "LeftArrow"
      | "DotEqual"
      | "NumberMarks"
      | "MixedRadix"
      | "MixedRadixQuantity"
      | "XMLObject"
      | "UnderoverscriptBox"
      | "ForwardBackward"
  ) {
    return Ok(unevaluated(name, args));
  }

  // ClearSystemCache[] - no-op, returns Null
  if name == "ClearSystemCache" {
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // XML`Parser`XMLGetString[xml] — minimal stub: return an expression
  // whose head is `XMLObject["Document"]`, so `Head[XML`Parser`XMLGetString[…]]`
  // matches the documented `XMLObject["Document"]`. The inner XML is
  // wrapped opaquely; full XML parsing is out of scope here. Well-
  // formedness is validated (balanced tags, no trailing content): if
  // the input is malformed, return `$Failed` to match wolframscript.
  if name == "XML`Parser`XMLGetString" && args.len() == 1 {
    if let Expr::String(s) = &args[0]
      && !is_well_formed_xml(s)
    {
      return Ok(Expr::Identifier("$Failed".to_string()));
    }
    return Ok(Expr::CurriedCall {
      func: Box::new(Expr::FunctionCall {
        name: "XMLObject".to_string(),
        args: vec![Expr::String("Document".to_string())].into(),
      }),
      args: vec![Expr::List(vec![].into()), args[0].clone()],
    });
  }

  // Play[f, {t, tmin, tmax}] builds a sound object whose amplitude is given by
  // the function f of the time variable t (in seconds) over [tmin, tmax].
  // wolframscript compiles f into a SampledSoundFunction; Woxi cannot sample
  // audio, so it wraps the inert Play expression in a Sound object. That makes
  // the result render as -Sound- and report Head -> Sound, matching the REPL.
  // The wrapped argument keeps the head `Play` (already evaluated, so it stays
  // inert inside Sound), which avoids re-dispatching into an endless loop.
  if name == "Play"
    && args.len() == 2
    && matches!(&args[1], Expr::List(items) if items.len() == 3)
  {
    return Ok(Expr::FunctionCall {
      name: "Sound".to_string(),
      args: vec![unevaluated("Play", args)].into(),
    });
  }

  // ListPlay[{a1, a2, …}, opts…] plays a list of amplitude levels as a sound.
  // wolframscript normalizes the amplitudes into [-1, 1] and wraps them in a
  // Sound[SampledSoundList[…]] object (sampled at 8000 Hz by default). Building
  // that same object makes the result render as -Sound-, report Head -> Sound,
  // and — in the visual hosts — play the normalized waveform.
  if name == "ListPlay"
    && !args.is_empty()
    && let Some(sound) = crate::functions::sound::list_play(args)
  {
    return Ok(sound);
  }

  // Check if the function is a known but unimplemented Wolfram Language function.
  // Some symbolic CAS objects (Root, RootSum, …) intentionally stay
  // unevaluated as their canonical form, so flagging them as
  // "unimplemented" is a false positive. Exclude that small set. Image3D
  // is similar: ImageQ classifies it without full Image3D support.
  if is_known_wolfram_function(name)
    && !matches!(
      name,
      "Root"
        | "RootSum"
        | "RootApproximant"
        | "Image3D"
        | "CenteredInterval"
        | "BernoulliGraphDistribution"
        // TemplateObject/TemplateSlot/TemplateExpression are symbolic template
        // objects produced by StringTemplate/FileTemplate/XMLTemplate. They stay
        // unevaluated as their canonical form (consumed by TemplateApply), so
        // they are not "unimplemented".
        | "TemplateObject"
        | "TemplateSlot"
        | "TemplateExpression"
        // SymmetricGroup[n] is a symbolic group object whose canonical form
        // stays unevaluated (matching wolframscript). It is consumed by
        // GroupOrder/GroupGenerators/etc., so it is not "unimplemented".
        | "SymmetricGroup"
        // DihedralGroup[n] is likewise a symbolic group object consumed by
        // GroupOrder/GroupGenerators/GroupElements.
        | "DihedralGroup"
        // CyclicGroup[n] is likewise a symbolic group object consumed by
        // GroupOrder/GroupGenerators/GroupElements.
        | "CyclicGroup"
        // URL["…"] is a symbolic URL wrapper whose canonical form stays
        // unevaluated (matching wolframscript). It is consumed by
        // HTTPRequest/URLBuild-style functions, so it is not "unimplemented".
        | "URL"
        // HTTPResponse[body, meta, …] is the symbolic response object
        // produced by URLRead; it stays unevaluated as its canonical form,
        // so it is not "unimplemented".
        | "HTTPResponse"
        // KeyValuePattern[…] is a symbolic pattern object whose canonical form
        // stays unevaluated (matching wolframscript). It is consumed by the
        // pattern matcher (MatchQ/Cases/Replace/…), so it is not "unimplemented".
        | "KeyValuePattern"
        // Simplex/Annulus/Parallelepiped are symbolic geometric region
        // objects whose canonical form stays unevaluated (matching
        // wolframscript). They are consumed by region functions
        // (RegionDimension/RegionMeasure/…), so they are not "unimplemented".
        | "Simplex"
        | "Annulus"
        | "Parallelepiped"
        // Torus/FilledTorus/Parallelogram/HalfPlane/InfinitePlane are likewise
        // symbolic geometric regions consumed by the region functions.
        | "Torus"
        | "FilledTorus"
        | "Parallelogram"
        | "HalfPlane"
        | "InfinitePlane"
        // JoinedCurve/BSplineSurface/Raster3D/AxisObject are graphics
        // primitives: they stay unevaluated as their canonical form and are
        // consumed by the Graphics/Graphics3D renderers.
        | "JoinedCurve"
        | "BSplineSurface"
        | "Raster3D"
        | "AxisObject"
        // OperatorApplied[f, …] is a symbolic operator object that stays
        // unevaluated until applied to arguments via the curried form
        // OperatorApplied[f][x][y] (handled in parse_curry_form), so it is
        // not "unimplemented".
        | "OperatorApplied"
        // CurryApplied[f, n] is likewise a symbolic operator object applied
        // as CurryApplied[f, n][x1]…[xn] (handled in parse_curry_form).
        | "CurryApplied"
        // NearestTo[x, …] is the operator form of Nearest: it stays
        // unevaluated until applied as NearestTo[x][data] (handled in the
        // curried-call machinery), so it is not "unimplemented".
        | "NearestTo"
        // BooleanFunction[n, k] is a symbolic boolean-function object that
        // stays unevaluated until applied as BooleanFunction[n, k][b1, …, bk]
        // (handled in apply_curried_call), so it is not "unimplemented".
        | "BooleanFunction"
        // Notation/display wrapper heads stay unevaluated as their canonical
        // form in wolframscript (like Subscript/Superscript/Framed already do):
        // they describe how something is laid out, not a value to compute, so
        // flagging them as "unimplemented" is a false positive.
        | "Overscript"
        | "Underscript"
        | "Underoverscript"
        | "Underlined"
        | "Highlighted"
        | "Mouseover"
        | "Magnify"
        // Geographic primitives consumed by GeoGraphics (rendered there);
        // they stay symbolic on their own, matching wolframscript, so they
        // are not "unimplemented".
        | "GeoMarker"
        | "GeoPath"
        | "GeoPolygon"
        | "GeoCircle"
        | "GeoDisk"
        // Ket/Bra are Dirac bra-ket notation objects; they stay symbolic until
        // consumed by the quantum framework, so they are not "unimplemented".
        | "Ket"
        | "Bra"
        // SwatchLegend[colors, labels, …] is a symbolic legend description
        // consumed by Legended (e.g. produced by PeriodicTablePlot["Phase"]);
        // it stays unevaluated as its canonical form, so it is not
        // "unimplemented".
        | "SwatchLegend"
        // Audio[data, …] is a symbolic audio object constructor. It stays
        // unevaluated as its canonical form and is consumed by AudioPlot,
        // so it is not "unimplemented". ShortTimeFourierData[…] is the
        // canonical data object produced by ShortTimeFourier.
        | "Audio"
        | "ShortTimeFourierData"
        // Wavelet family heads are symbolic constructor objects consumed by
        // the wavelet transforms, WaveletFilterCoefficients, and
        // WaveletPhi/WaveletPsi; the data objects and LiftingFilterData are
        // canonical forms produced by those functions.
        | "HaarWavelet"
        | "DaubechiesWavelet"
        | "SymletWavelet"
        | "CoifletWavelet"
        | "BattleLemarieWavelet"
        | "BiorthogonalSplineWavelet"
        | "ReverseBiorthogonalSplineWavelet"
        | "CDFWavelet"
        | "MeyerWavelet"
        | "ShannonWavelet"
        | "MexicanHatWavelet"
        | "GaborWavelet"
        | "DGaussianWavelet"
        | "MorletWavelet"
        | "PaulWavelet"
        | "DiscreteWaveletData"
        | "ContinuousWaveletData"
        | "LiftingFilterData"
        // Quiz/assessment objects. AssessmentFunction[spec] and
        // QuestionObject[q, assess] are symbolic constructor objects that stay
        // unevaluated until applied to a candidate answer (handled in
        // apply_curried_call), and AssessmentResultObject[<|…|>] is the graded
        // result they produce, so none of these are "unimplemented".
        | "AssessmentFunction"
        | "QuestionObject"
        | "AssessmentResultObject"
        // More notation/display wrapper heads. Like Subscript/Framed, these
        // describe layout rather than a value to compute, so wolframscript
        // leaves them unevaluated as their canonical form.
        | "Subsuperscript"
        | "Tooltip"
        | "Interpretation"
        | "Invisible"
        | "MouseAppearance"
        | "Editable"
        | "Selectable"
        // Interactive control / view wrapper heads. In script mode (the
        // wolframscript -code reference) these stay unevaluated rather than
        // producing an interactive object, so they are not "unimplemented".
        | "Button"
        | "ActionMenu"
        | "Deploy"
        | "Dynamic"
        | "DynamicWrapper"
        | "Setter"
        | "Slider"
        | "Toggler"
        | "Manipulator"
        | "ColorSlider"
        | "Opener"
        | "TabView"
        | "MenuView"
        | "SlideView"
        | "FlipView"
        // Further interactive control / animation / view heads from the
        // InteractiveManipulation guide. Like Slider/Toggler above, these
        // stay unevaluated as their canonical form in wolframscript's script
        // mode (they produce an interactive object only inside a notebook),
        // so flagging them as "unimplemented" is a false positive.
        | "Animate"
        | "Animator"
        | "ListAnimate"
        | "ControllerManipulate"
        | "Trigger"
        | "SetterBar"
        | "CheckboxBar"
        | "TogglerBar"
        | "RadioButton"
        | "ProgressIndicator"
        | "ClickPane"
        | "LocatorPane"
        | "PaneSelector"
        | "PopupView"
        | "ColorSetter"
        | "IntervalSlider"
        | "Slider2D"
        // Structured-array display wrappers produced by LUDecomposition (and
        // the other structured-matrix functions). Like the notation wrappers
        // above, these are canonical inert forms wrapping a
        // `StructuredArray`StructuredData[…]` object rather than a value to
        // compute, so flagging them as "unimplemented" is a false positive.
        | "LowerTriangularMatrix"
        | "UpperTriangularMatrix"
        | "PermutationMatrix"
    )
  {
    let args_str = args
      .iter()
      .map(expr_to_string)
      .collect::<Vec<_>>()
      .join(", ");
    let call_str = format!("{}[{}]", name, args_str);
    crate::capture_unimplemented_call(&call_str);
  }

  // Unknown function - return as symbolic function call
  Ok(unevaluated(name, args))
}

/// Extract RGB components from a color expression as (numerator, denominator) pairs.
/// Returns None if the expression is not a recognized color.
/// Lightweight XML well-formedness check used by `XML`Parser`XMLGetString`.
/// Accepts a single root element with balanced nested tags. Rejects
/// inputs with text outside the root element (such as trailing junk
/// after the closing tag), unbalanced tags, or empty content. Comments,
/// CDATA sections, processing instructions, and XML declarations are
/// recognized only at the document level. Not a conformant XML parser —
/// just enough to flag the obviously broken cases that should yield
/// `$Failed` to match wolframscript.
fn is_well_formed_xml(input: &str) -> bool {
  let bytes = input.as_bytes();
  let mut i = 0;
  let n = bytes.len();

  fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
      i += 1;
    }
    i
  }
  // Skip prolog: whitespace, XML declaration, processing instructions,
  // comments. Returns the new index, or None on malformed prolog.
  fn skip_prolog(b: &[u8], mut i: usize) -> Option<usize> {
    loop {
      i = skip_ws(b, i);
      if i >= b.len() || b[i] != b'<' {
        return Some(i);
      }
      // <?...?> declaration or PI
      if i + 1 < b.len() && b[i + 1] == b'?' {
        let end = b
          .windows(2)
          .enumerate()
          .skip(i + 2)
          .find(|(_, w)| w == b"?>");
        i = match end {
          Some((pos, _)) => pos + 2,
          None => return None,
        };
        continue;
      }
      // <!-- ... --> comment
      if b[i..].starts_with(b"<!--") {
        let end = b
          .windows(3)
          .enumerate()
          .skip(i + 4)
          .find(|(_, w)| w == b"-->");
        i = match end {
          Some((pos, _)) => pos + 3,
          None => return None,
        };
        continue;
      }
      // <!DOCTYPE...> — skip until matching '>'
      if b[i..].starts_with(b"<!") {
        let mut depth = 1;
        let mut j = i + 2;
        while j < b.len() && depth > 0 {
          match b[j] {
            b'<' => depth += 1,
            b'>' => depth -= 1,
            _ => {}
          }
          j += 1;
        }
        if depth != 0 {
          return None;
        }
        i = j;
        continue;
      }
      return Some(i);
    }
  }

  i = match skip_prolog(bytes, i) {
    Some(j) => j,
    None => return false,
  };
  if i >= n || bytes[i] != b'<' {
    return false;
  }

  // Parse a tag starting at `i` (which points to `<`). Returns
  // (end_index_after_>, kind) where kind is Open/Close/SelfClose/Other.
  // Open/Close also return the tag name.
  #[derive(PartialEq)]
  enum TagKind<'a> {
    Open(&'a [u8]),
    Close(&'a [u8]),
    SelfClose,
    Skip, // comment / PI / CDATA — no impact on tag stack
  }
  fn parse_tag<'a>(b: &'a [u8], i: usize) -> Option<(usize, TagKind<'a>)> {
    if i >= b.len() || b[i] != b'<' {
      return None;
    }
    // Comments
    if b[i..].starts_with(b"<!--") {
      let end = b
        .windows(3)
        .enumerate()
        .skip(i + 4)
        .find(|(_, w)| w == b"-->");
      return end.map(|(pos, _)| (pos + 3, TagKind::Skip));
    }
    // CDATA
    if b[i..].starts_with(b"<![CDATA[") {
      let end = b
        .windows(3)
        .enumerate()
        .skip(i + 9)
        .find(|(_, w)| w == b"]]>");
      return end.map(|(pos, _)| (pos + 3, TagKind::Skip));
    }
    // PI: <? ... ?>
    if i + 1 < b.len() && b[i + 1] == b'?' {
      let end = b
        .windows(2)
        .enumerate()
        .skip(i + 2)
        .find(|(_, w)| w == b"?>");
      return end.map(|(pos, _)| (pos + 2, TagKind::Skip));
    }
    // Closing tag </name>
    if i + 1 < b.len() && b[i + 1] == b'/' {
      let name_start = i + 2;
      let mut j = name_start;
      while j < b.len()
        && (b[j].is_ascii_alphanumeric()
          || b[j] == b'_'
          || b[j] == b':'
          || b[j] == b'-'
          || b[j] == b'.')
      {
        j += 1;
      }
      let name = &b[name_start..j];
      if name.is_empty() {
        return None;
      }
      // Skip whitespace then expect '>'
      let k = skip_ws(b, j);
      if k >= b.len() || b[k] != b'>' {
        return None;
      }
      return Some((k + 1, TagKind::Close(name)));
    }
    // Opening or self-closing tag: <name attrs...> or <name attrs.../>
    let name_start = i + 1;
    let mut j = name_start;
    while j < b.len()
      && (b[j].is_ascii_alphanumeric()
        || b[j] == b'_'
        || b[j] == b':'
        || b[j] == b'-'
        || b[j] == b'.')
    {
      j += 1;
    }
    let name = &b[name_start..j];
    if name.is_empty() {
      return None;
    }
    // Scan to '>' or '/>'. Track string quoting to skip '>' inside attribute values.
    let mut quote: Option<u8> = None;
    while j < b.len() {
      let c = b[j];
      match (quote, c) {
        (Some(q), x) if x == q => quote = None,
        (None, b'"') | (None, b'\'') => quote = Some(c),
        (None, b'/') if j + 1 < b.len() && b[j + 1] == b'>' => {
          return Some((j + 2, TagKind::SelfClose));
        }
        (None, b'>') => {
          return Some((j + 1, TagKind::Open(name)));
        }
        _ => {}
      }
      j += 1;
    }
    None
  }

  // Read the root element (and everything inside it).
  let mut stack: Vec<&[u8]> = Vec::new();
  let (mut next, kind) = match parse_tag(bytes, i) {
    Some(v) => v,
    None => return false,
  };
  match kind {
    TagKind::Open(name) => stack.push(name),
    TagKind::SelfClose => {
      // Single self-closing root tag is well-formed; jump past trailing
      // prolog-like content.
      i = next;
      return matches!(skip_prolog(bytes, i), Some(j) if j >= n);
    }
    _ => return false,
  }
  i = next;

  while !stack.is_empty() {
    if i >= n {
      return false; // unclosed tag
    }
    if bytes[i] != b'<' {
      // Text content — fine; advance to next '<'
      while i < n && bytes[i] != b'<' {
        i += 1;
      }
      continue;
    }
    let (j, k) = match parse_tag(bytes, i) {
      Some(v) => v,
      None => return false,
    };
    match k {
      TagKind::Open(name) => {
        stack.push(name);
      }
      TagKind::Close(name) => {
        let top = match stack.last() {
          Some(t) => *t,
          None => return false,
        };
        if top != name {
          return false;
        }
        stack.pop();
      }
      TagKind::SelfClose | TagKind::Skip => {}
    }
    i = j;
    next = j;
    let _ = next;
  }

  // After the root element closes, only prolog-like trailing content
  // (whitespace, comments, PIs) is allowed.
  match skip_prolog(bytes, i) {
    Some(j) => j >= n,
    None => false,
  }
}

fn extract_rgb_rational(color: &Expr) -> Option<[(i128, i128); 3]> {
  match color {
    Expr::FunctionCall { name, args }
      if name == "RGBColor" && args.len() >= 3 =>
    {
      let mut rgb = [(0i128, 1i128); 3];
      for (i, arg) in args[..3].iter().enumerate() {
        rgb[i] = expr_to_rational(arg)?;
      }
      Some(rgb)
    }
    Expr::FunctionCall { name, args }
      if name == "GrayLevel" && !args.is_empty() =>
    {
      let g = expr_to_rational(&args[0])?;
      Some([g, g, g])
    }
    _ => None,
  }
}

/// Convert an expression to a rational (numerator, denominator) pair.
fn expr_to_rational(expr: &Expr) -> Option<(i128, i128)> {
  if let Some(pair) = crate::functions::math_ast::expr_to_rational(expr) {
    return Some(pair);
  }
  match expr {
    Expr::Real(f) => {
      // Try to convert common float values to exact rationals
      if *f == 0.0 {
        Some((0, 1))
      } else if *f == 1.0 {
        Some((1, 1))
      } else {
        // For other floats, use denominator 1000 and simplify
        let n = (*f * 1000.0).round() as i128;
        let g = gcd_i128(n.abs(), 1000);
        Some((n / g, 1000 / g))
      }
    }
    _ => None,
  }
}

/// Build a petgraph UnGraph from Wolfram Graph vertices and edges.
/// Delegates to graph module.
fn build_undirected_graph(
  vertices: &[Expr],
  edges: &[Expr],
) -> (
  petgraph::graph::UnGraph<usize, ()>,
  std::collections::HashMap<String, petgraph::graph::NodeIndex>,
) {
  crate::functions::graph::build_ungraph(vertices, edges)
}

/// Check if undirected graph is connected using petgraph.
fn is_connected_pg(graph: &petgraph::graph::UnGraph<usize, ()>) -> bool {
  if graph.node_count() == 0 {
    return true;
  }
  petgraph::algo::connected_components(graph) == 1
}

/// BFS from start vertex using petgraph, return all distances.
/// All-pairs shortest-path distances honouring edge direction: directed edges
/// only point forwards, undirected edges point both ways. `result[s][t]` is
/// the hop distance from `s` to `t`, `0` for `s == t`, and `-1` when `t` is
/// unreachable from `s`.
fn graph_directed_distances(
  vertices: &[Expr],
  edges: &[Expr],
) -> Vec<Vec<i128>> {
  let n = vertices.len();
  let index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (expr_to_string(v), i))
    .collect();
  let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
  for edge in edges.iter() {
    if let Expr::FunctionCall { name: en, args: ea } = edge
      && ea.len() == 2
      && let (Some(&u), Some(&v)) = (
        index.get(&expr_to_string(&ea[0])),
        index.get(&expr_to_string(&ea[1])),
      )
    {
      adj[u].push(v);
      if en == "UndirectedEdge" {
        adj[v].push(u);
      }
    }
  }
  (0..n)
    .map(|s| {
      let mut dist = vec![-1i128; n];
      dist[s] = 0;
      let mut q = std::collections::VecDeque::from([s]);
      while let Some(u) = q.pop_front() {
        for &w in &adj[u] {
          if dist[w] < 0 {
            dist[w] = dist[u] + 1;
            q.push_back(w);
          }
        }
      }
      dist
    })
    .collect()
}

/// Build a rational or integer Expr from (num, den).
fn rational_to_expr(num: i128, den: i128) -> Expr {
  if den == 1 {
    Expr::Integer(num)
  } else if num == 0 {
    Expr::Integer(0)
  } else {
    let (n, d) = rat_reduce(num, den);
    if d == 1 {
      Expr::Integer(n)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
      }
    }
  }
}

/// Evaluate Darker[color, amount] or Lighter[color, amount].
/// `is_darker` = true for Darker, false for Lighter.
fn evaluate_darker_lighter(args: &[Expr], is_darker: bool) -> Option<Expr> {
  if let Expr::Image {
    color_space: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = &args[0]
  {
    let amount: f64 = if args.len() >= 2 {
      match &args[1] {
        Expr::Real(f) => *f,
        Expr::Integer(n) => *n as f64,
        Expr::FunctionCall { name, args: ra }
          if name == "Rational" && ra.len() == 2 =>
        {
          match (&ra[0], &ra[1]) {
            (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => {
              *n as f64 / *d as f64
            }
            _ => return None,
          }
        }
        _ => return None,
      }
    } else {
      1.0 / 3.0
    };
    let ch = *channels as usize;
    let has_alpha = ch == 2 || ch == 4;
    let is_real32 = matches!(image_type, crate::syntax::ImageType::Real32);
    let new_data: Vec<f64> = data
      .iter()
      .enumerate()
      .map(|(i, &c)| {
        if has_alpha && (i % ch) == ch - 1 {
          return c;
        }
        if is_real32 {
          let c32 = (c as f32) as f64;
          let r = if is_darker {
            c32 * (1.0 - amount)
          } else {
            (c32 * (1.0 + amount)).min(1.0)
          };
          (r as f32) as f64
        } else if is_darker {
          c * (1.0 - amount)
        } else {
          (c * (1.0 + amount)).min(1.0)
        }
      })
      .collect();
    return Some(Expr::Image {
      color_space: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(new_data),
      image_type: *image_type,
    });
  }

  let rgb = extract_rgb_rational(&args[0])?;

  // Default amount is 1/3
  let (amt_num, amt_den) = if args.len() >= 2 {
    expr_to_rational(&args[1])?
  } else {
    (1, 3)
  };

  // Per-component Real contagion (matches wolframscript): a component is
  // Real iff the corresponding input channel was Real or the amount is
  // Real. `Lighter[Orange, 1/4]` with `Orange = RGBColor[1, 0.5, 0]`
  // yields `RGBColor[1, 0.625, 1/4]` — only the middle channel participates
  // in Real contagion because only `0.5` was Real.
  let component_is_real: Vec<bool> = if let Expr::FunctionCall {
    name,
    args: rgb_args,
  } = &args[0]
    && (name == "RGBColor" || name == "GrayLevel")
  {
    // GrayLevel broadcasts: its single component controls all three outputs.
    if name == "GrayLevel" {
      let real = matches!(rgb_args.first(), Some(Expr::Real(_)));
      vec![real; 3]
    } else {
      rgb_args
        .iter()
        .take(3)
        .map(|a| matches!(a, Expr::Real(_)))
        .chain(std::iter::repeat(false))
        .take(3)
        .collect()
    }
  } else {
    vec![false; 3]
  };
  let amount_is_real = args.len() >= 2 && matches!(&args[1], Expr::Real(_));

  let mut result_rgb = [Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)];
  for (i, (c_num, c_den)) in rgb.iter().enumerate() {
    let force_real = component_is_real[i] || amount_is_real;
    if is_darker {
      // Darker: c * (1 - amount) = c * (den - num) / den
      let factor_num = amt_den - amt_num;
      let factor_den = amt_den;
      let new_num = c_num * factor_num;
      let new_den = c_den * factor_den;
      result_rgb[i] = if force_real {
        Expr::Real((new_num as f64) / (new_den as f64))
      } else {
        rational_to_expr(new_num, new_den)
      };
    } else {
      // Lighter: c + amount * (1 - c)
      // = c + amount - amount*c
      // = c*(1 - amount) + amount
      // = c_num/c_den * (amt_den - amt_num)/amt_den + amt_num/amt_den
      // = (c_num*(amt_den-amt_num) + c_den*amt_num) / (c_den * amt_den)
      let new_num = c_num * (amt_den - amt_num) + c_den * amt_num;
      let new_den = c_den * amt_den;
      result_rgb[i] = if force_real {
        Expr::Real((new_num as f64) / (new_den as f64))
      } else {
        rational_to_expr(new_num, new_den)
      };
    }
  }

  Some(Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![
      result_rgb[0].clone(),
      result_rgb[1].clone(),
      result_rgb[2].clone(),
    ]
    .into(),
  })
}

/// Check if an expression is a GrayLevel color.
fn is_graylevel(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, args } if name == "GrayLevel" && !args.is_empty())
}

/// Match `NDEigenvalues[DiffusionPDETerm[{u[x], {x}}], u, Element[{x}, Line[{{a}, {b}}]], n]`
/// and return the closed-form list of `n` Neumann eigenvalues of
/// `-d²/dx²` on `[a, b]`: `(k π / (b - a))²` for `k = 0, …, n - 1`.
fn nd_eigenvalues_diffusion_line(args: &[Expr]) -> Option<Expr> {
  // args[0]: DiffusionPDETerm[{u[x], {x}}]
  let (op_name, op_args) = match &args[0] {
    Expr::FunctionCall { name, args } => (name.as_str(), args),
    _ => return None,
  };
  if op_name != "DiffusionPDETerm" || op_args.len() != 1 {
    return None;
  }
  let vars = match &op_args[0] {
    Expr::List(items) if items.len() == 2 => items,
    _ => return None,
  };
  // Must be {u[x], {x}} — single dependent variable with single spatial var.
  let dep = match &vars[0] {
    Expr::FunctionCall { args, .. } if args.len() == 1 => Some(&args[0]),
    _ => None,
  }?;
  let spatial = match &vars[1] {
    Expr::List(xs) if xs.len() == 1 => &xs[0],
    _ => return None,
  };
  if !matches!(dep, Expr::Identifier(_))
    || !matches!(spatial, Expr::Identifier(_))
  {
    return None;
  }
  // args[2]: Element[{x}, Line[{{a}, {b}}]]
  let elem = match &args[2] {
    Expr::FunctionCall { name, args }
      if name == "Element" && args.len() == 2 =>
    {
      args
    }
    _ => return None,
  };
  let line = match &elem[1] {
    Expr::FunctionCall { name, args } if name == "Line" && args.len() == 1 => {
      &args[0]
    }
    _ => return None,
  };
  let endpoints = match line {
    Expr::List(items) if items.len() == 2 => items,
    _ => return None,
  };
  let to_pt = |e: &Expr| -> Option<Expr> {
    match e {
      Expr::List(xs) if xs.len() == 1 => Some(xs[0].clone()),
      _ => None,
    }
  };
  let a_expr = to_pt(&endpoints[0])?;
  let b_expr = to_pt(&endpoints[1])?;
  // args[3]: n (positive integer)
  let n = match &args[3] {
    Expr::Integer(n) if *n > 0 => *n as usize,
    _ => return None,
  };

  // Length L = b - a, evaluated to a number.
  let length_expr =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        b_expr,
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), a_expr].into(),
        },
      ]
      .into(),
    })
    .ok()?;
  let l = match crate::functions::math_ast::try_eval_to_f64(&length_expr) {
    Some(v) if v > 0.0 => v,
    _ => return None,
  };

  let pi = std::f64::consts::PI;
  let eigs: Vec<Expr> = (0..n)
    .map(|k| {
      let v = (k as f64 * pi / l).powi(2);
      Expr::Real(v)
    })
    .collect();
  Some(Expr::List(eigs.into()))
}

/// Evaluate Blend[{c1, c2, ...}] or Blend[{c1, c2, ...}, t].
fn evaluate_blend(args: &[Expr]) -> Option<Expr> {
  let colors = match &args[0] {
    Expr::List(items) if items.len() >= 2 => items,
    _ => return None,
  };

  // Explicit-position form: Blend[{{p1, c1}, {p2, c2}, …}, t] places the
  // colors at parameter positions p_i and interpolates piecewise-linearly at
  // t (clamped to [p1, p_last]). Detected when every entry is a {pos, color}
  // pair; falls through to the uniform form otherwise.
  if args.len() == 2
    && colors
      .iter()
      .all(|item| matches!(item, Expr::List(pair) if pair.len() == 2))
    && let Some(result) = evaluate_blend_positioned(colors, &args[1])
  {
    return Some(result);
  }

  // Image blend: linearly interpolate matching-shape Images per pixel.
  if colors.iter().all(|c| matches!(c, Expr::Image { .. })) {
    return blend_images(colors, args.get(1));
  }

  // Check if all colors are GrayLevel (to preserve output type)
  let all_graylevel = colors.iter().all(is_graylevel);

  // Extract RGB rationals for all colors
  let rgbs: Vec<[(i128, i128); 3]> = colors
    .iter()
    .map(extract_rgb_rational)
    .collect::<Option<Vec<_>>>()?;

  let n = rgbs.len();

  if args.len() == 1 {
    // Blend[{c1, c2, ...}] — equal blend (average)
    let mut sum = [(0i128, 1i128); 3];
    for rgb in &rgbs {
      for ch in 0..3 {
        // sum[ch] += rgb[ch]: a/b + c/d = (a*d + c*b) / (b*d)
        let (sn, sd) = sum[ch];
        let (cn, cd) = rgb[ch];
        sum[ch] = (sn * cd + cn * sd, sd * cd);
      }
    }
    // Divide by n
    let n_i128 = n as i128;
    if all_graylevel {
      let (num, den) = sum[0];
      Some(Expr::FunctionCall {
        name: "GrayLevel".to_string(),
        args: vec![rational_to_expr(num, den * n_i128)].into(),
      })
    } else {
      Some(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: (0..3)
          .map(|ch| {
            let (num, den) = sum[ch];
            rational_to_expr(num, den * n_i128)
          })
          .collect(),
      })
    }
  } else {
    // Blend[{c1, c2, ...}, weights] where weights is a list of the wrong
    // length: wolframscript emits Blend::argl and returns the call
    // unevaluated (with named colors expanded to RGBColor).
    if let Expr::List(weights) = &args[1]
      && weights.len() != n
    {
      let colors_expr = Expr::List(colors.clone());
      crate::emit_message(&format!(
        "Blend::argl: {} should be a real number or a list of non-negative numbers, which has the same length as {}.",
        crate::syntax::expr_to_string(&args[1]),
        crate::syntax::expr_to_string(&colors_expr),
      ));
      return None;
    }
    // Blend[{c1, c2, ...}, t] — interpolation along the color list.
    // If `t` is an inexact Real, fall back to float arithmetic so the
    // resulting RGBColor components are reals, matching wolframscript.
    let is_real_t = matches!(&args[1], Expr::Real(_));
    if is_real_t {
      let t = match &args[1] {
        Expr::Real(r) => *r,
        _ => return None,
      };
      return Some(blend_two_float(&rgbs, t, all_graylevel));
    }
    let (t_num, t_den) = expr_to_rational(&args[1])?;

    if n == 2 {
      // Simple case: c1*(1-t) + c2*t
      blend_two_rational(&rgbs[0], &rgbs[1], t_num, t_den, all_graylevel)
    } else {
      // Multi-color: map t in [0,1] to segments
      // t=0 → first color, t=1 → last color
      // segment_len = 1/(n-1), segment_index = floor(t * (n-1))
      let segments = (n - 1) as i128;
      // position = t * (n-1) = t_num * segments / t_den
      let pos_num = t_num * segments;
      let pos_den = t_den;

      // segment index = floor(pos_num / pos_den)
      let seg_idx = if pos_num <= 0 {
        0usize
      } else {
        let idx = (pos_num / pos_den) as usize;
        idx.min(n - 2)
      };

      // local t within the segment: local_t = pos - seg_idx
      // = pos_num/pos_den - seg_idx = (pos_num - seg_idx*pos_den) / pos_den
      let local_t_num = pos_num - (seg_idx as i128) * pos_den;
      let local_t_den = pos_den;

      blend_two_rational(
        &rgbs[seg_idx],
        &rgbs[seg_idx + 1],
        local_t_num,
        local_t_den,
        all_graylevel,
      )
    }
  }
}

/// Blend[{{p1, c1}, {p2, c2}, …}, t] — interpolate colors at explicit
/// parameter positions. Returns None (to fall through) if any entry is not a
/// numeric-position / color pair. Exact when t and all positions are exact
/// (rational output); float when any is inexact; endpoints returned unchanged
/// when t is outside [p1, p_last].
fn evaluate_blend_positioned(pairs: &[Expr], t_arg: &Expr) -> Option<Expr> {
  let n = pairs.len();
  let mut pos_f64: Vec<f64> = Vec::with_capacity(n);
  let mut rgbs: Vec<[(i128, i128); 3]> = Vec::with_capacity(n);
  let mut all_graylevel = true;
  let mut all_pos_exact = true;
  for item in pairs {
    let Expr::List(pair) = item else {
      return None;
    };
    if pair.len() != 2 {
      return None;
    }
    pos_f64.push(crate::functions::math_ast::expr_to_f64(&pair[0])?);
    if matches!(&pair[0], Expr::Real(_)) {
      all_pos_exact = false;
    }
    if !is_graylevel(&pair[1]) {
      all_graylevel = false;
    }
    rgbs.push(extract_rgb_rational(&pair[1])?);
  }

  let build_exact = |rgb: &[(i128, i128); 3]| -> Expr {
    if all_graylevel {
      Expr::FunctionCall {
        name: "GrayLevel".to_string(),
        args: vec![rational_to_expr(rgb[0].0, rgb[0].1)].into(),
      }
    } else {
      Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: (0..3)
          .map(|ch| rational_to_expr(rgb[ch].0, rgb[ch].1))
          .collect(),
      }
    }
  };

  let t_f64 = crate::functions::math_ast::expr_to_f64(t_arg)?;
  // Outside the parameter range: clamp to the endpoint color (unchanged).
  if t_f64 <= pos_f64[0] {
    return Some(build_exact(&rgbs[0]));
  }
  if t_f64 >= pos_f64[n - 1] {
    return Some(build_exact(&rgbs[n - 1]));
  }
  // Locate the segment [p_seg, p_{seg+1}] containing t.
  let mut seg = 0;
  for i in 0..n - 1 {
    if t_f64 >= pos_f64[i] && t_f64 <= pos_f64[i + 1] {
      seg = i;
      break;
    }
  }

  let t_exact = all_pos_exact && !matches!(t_arg, Expr::Real(_));
  if t_exact {
    let (tn, td) = expr_to_rational(t_arg)?;
    let Expr::List(lo) = &pairs[seg] else {
      return None;
    };
    let Expr::List(hi) = &pairs[seg + 1] else {
      return None;
    };
    let (an, ad) = expr_to_rational(&lo[0])?;
    let (bn, bd) = expr_to_rational(&hi[0])?;
    // local_t = (t - p_seg) / (p_{seg+1} - p_seg), as a rational.
    let num_diff = tn * ad - an * td; // over td*ad
    let den_diff = td * ad;
    let num_span = bn * ad - an * bd; // over bd*ad
    let den_span = bd * ad;
    if num_span == 0 {
      return None;
    }
    let lt_num = num_diff * den_span;
    let lt_den = den_diff * num_span;
    return blend_two_rational(
      &rgbs[seg],
      &rgbs[seg + 1],
      lt_num,
      lt_den,
      all_graylevel,
    );
  }

  // Inexact: interpolate the two segment colors in f64.
  let local = (t_f64 - pos_f64[seg]) / (pos_f64[seg + 1] - pos_f64[seg]);
  let c1 = &rgbs[seg];
  let c2 = &rgbs[seg + 1];
  let build_channel = |ch: usize| -> Expr {
    let v1 = c1[ch].0 as f64 / c1[ch].1 as f64;
    let v2 = c2[ch].0 as f64 / c2[ch].1 as f64;
    Expr::Real(v1 * (1.0 - local) + v2 * local)
  };
  if all_graylevel {
    Some(Expr::FunctionCall {
      name: "GrayLevel".to_string(),
      args: vec![build_channel(0)].into(),
    })
  } else {
    Some(Expr::FunctionCall {
      name: "RGBColor".to_string(),
      args: (0..3).map(build_channel).collect(),
    })
  }
}

/// Blend a list of Images (already known to be all images) per-pixel.
/// `weight` is the optional `t` argument: missing means equal average.
fn blend_images(colors: &[Expr], weight: Option<&Expr>) -> Option<Expr> {
  let n = colors.len();
  let (w, h, ch, image_type) = match &colors[0] {
    Expr::Image {
      width,
      height,
      channels,
      image_type,
      ..
    } => (*width, *height, *channels, *image_type),
    _ => return None,
  };
  for c in colors {
    let Expr::Image {
      width,
      height,
      channels,
      ..
    } = c
    else {
      return None;
    };
    if *width != w || *height != h || *channels != ch {
      return None;
    }
  }
  let len = (w as usize) * (h as usize) * (ch as usize);
  let datas: Vec<&[f64]> = colors
    .iter()
    .map(|c| match c {
      Expr::Image { data, .. } => data.as_slice(),
      _ => unreachable!(),
    })
    .collect();

  let t = match weight {
    None => None,
    Some(e) => Some(crate::functions::math_ast::try_eval_to_f64(e)?),
  };

  // For weight=None: equal average. For weight=t with two images:
  // (1-t)*img0 + t*img1. For n>2: map t in [0,1] to the segment in
  // [(i)/(n-1), (i+1)/(n-1)] and interpolate the two adjacent images.
  let mut new_data = Vec::with_capacity(len);
  for i in 0..len {
    let v = match t {
      None => datas.iter().map(|d| d[i]).sum::<f64>() / n as f64,
      Some(t) if n == 2 => (1.0 - t) * datas[0][i] + t * datas[1][i],
      Some(t) => {
        let pos = t * (n as f64 - 1.0);
        let seg = (pos.floor() as usize).min(n - 2);
        let local = pos - seg as f64;
        (1.0 - local) * datas[seg][i] + local * datas[seg + 1][i]
      }
    };
    new_data.push(v);
  }
  Some(Expr::Image {
    color_space: None,
    width: w,
    height: h,
    channels: ch,
    data: std::sync::Arc::new(new_data),
    image_type,
  })
}

/// Blend colors with a float weight t. Matches wolframscript's behaviour of
/// returning RGBColor components as reals when `t` is an inexact number.
fn blend_two_float(
  rgbs: &[[(i128, i128); 3]],
  t: f64,
  as_graylevel: bool,
) -> Expr {
  let t = t.clamp(0.0, 1.0);
  let n = rgbs.len();
  // Map t in [0, 1] to segments.
  let (c1, c2, local_t) = if n == 2 {
    (&rgbs[0], &rgbs[1], t)
  } else {
    let segments = (n - 1) as f64;
    let pos = t * segments;
    let mut seg_idx = pos.floor() as usize;
    if seg_idx >= n - 1 {
      seg_idx = n - 2;
    }
    let local = pos - seg_idx as f64;
    (&rgbs[seg_idx], &rgbs[seg_idx + 1], local)
  };

  let build_channel = |ch: usize| -> Expr {
    let (c1n, c1d) = c1[ch];
    let (c2n, c2d) = c2[ch];
    let v1 = c1n as f64 / c1d as f64;
    let v2 = c2n as f64 / c2d as f64;
    Expr::Real(v1 * (1.0 - local_t) + v2 * local_t)
  };

  if as_graylevel {
    Expr::FunctionCall {
      name: "GrayLevel".to_string(),
      args: vec![build_channel(0)].into(),
    }
  } else {
    Expr::FunctionCall {
      name: "RGBColor".to_string(),
      args: (0..3).map(build_channel).collect(),
    }
  }
}

/// Blend two colors with rational weight: c1*(1-t) + c2*t
fn blend_two_rational(
  c1: &[(i128, i128); 3],
  c2: &[(i128, i128); 3],
  t_num: i128,
  t_den: i128,
  as_graylevel: bool,
) -> Option<Expr> {
  // (1-t) = (t_den - t_num) / t_den
  let one_minus_t_num = t_den - t_num;

  let build_channel = |ch: usize| -> Expr {
    // c1[ch] * (1-t) + c2[ch] * t
    // = c1n/c1d * omt_n/t_den + c2n/c2d * t_num/t_den
    let (c1n, c1d) = c1[ch];
    let (c2n, c2d) = c2[ch];
    let num = c1n * one_minus_t_num * c2d + c2n * t_num * c1d;
    let den = c1d * c2d * t_den;
    rational_to_expr(num, den)
  };

  if as_graylevel {
    Some(Expr::FunctionCall {
      name: "GrayLevel".to_string(),
      args: vec![build_channel(0)].into(),
    })
  } else {
    Some(Expr::FunctionCall {
      name: "RGBColor".to_string(),
      args: (0..3).map(build_channel).collect(),
    })
  }
}

/// FunctionInterpolation[expr, {x, xmin, xmax}] — sample a function and build
/// an InterpolatingFunction with cubic spline interpolation.
fn function_interpolation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if let Expr::List(spec) = &args[1]
    && spec.len() == 3
    && let Expr::Identifier(var_name) = &spec[0]
  {
    // Force numeric evaluation of bounds
    let xmin_n = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![spec[1].clone()].into(),
    };
    let xmax_n = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![spec[2].clone()].into(),
    };
    let xmin_expr = crate::evaluator::evaluate_expr_to_expr(&xmin_n);
    let xmax_expr = crate::evaluator::evaluate_expr_to_expr(&xmax_n);
    if let (Ok(xmin_e), Ok(xmax_e)) = (&xmin_expr, &xmax_expr) {
      let xmin = match xmin_e {
        Expr::Integer(n) => Some(*n as f64),
        Expr::Real(f) => Some(*f),
        _ => None,
      };
      let xmax = match xmax_e {
        Expr::Integer(n) => Some(*n as f64),
        Expr::Real(f) => Some(*f),
        _ => None,
      };
      if let (Some(xmin), Some(xmax)) = (xmin, xmax) {
        let n_points = 65;
        let dx = (xmax - xmin) / (n_points - 1) as f64;
        let mut data_points = Vec::with_capacity(n_points);
        let body = &args[0];

        for i in 0..n_points {
          let x_val = xmin + i as f64 * dx;
          let x_expr = Expr::Real(x_val);
          let substituted =
            crate::syntax::substitute_variable(body, var_name, &x_expr);
          if let Ok(y_expr) =
            crate::evaluator::evaluate_expr_to_expr(&substituted)
          {
            let y_val = match &y_expr {
              Expr::Integer(n) => Some(*n as f64),
              Expr::Real(f) => Some(*f),
              _ => None,
            };
            if let Some(y) = y_val {
              data_points.push(Expr::List(
                vec![Expr::Real(x_val), Expr::Real(y)].into(),
              ));
            }
          }
        }

        if data_points.len() >= 2 {
          let domain = Expr::List(
            vec![Expr::List(vec![Expr::Real(xmin), Expr::Real(xmax)].into())]
              .into(),
          );
          return Ok(Expr::FunctionCall {
            name: "InterpolatingFunction".to_string(),
            args: vec![
              domain,
              Expr::List(data_points.into()),
              Expr::Integer(3),
            ]
            .into(),
          });
        }
      }
    }
  }
  Ok(unevaluated("FunctionInterpolation", args))
}

/// Compute chromatic polynomial coefficients via deletion-contraction.
/// Returns coefficients [c_0, c_1, ..., c_n] where P(k) = c_0 + c_1*k + ... + c_n*k^n.
fn chromatic_poly_coeffs(n: usize, edges: &[(usize, usize)]) -> Vec<i128> {
  if edges.is_empty() {
    // Empty graph (no edges): P(k) = k^n
    let mut coeffs = vec![0i128; n + 1];
    if n <= coeffs.len() {
      coeffs[n] = 1;
    }
    return coeffs;
  }

  // Pick an edge to delete/contract
  let (u, v) = edges[0];
  let remaining_edges: Vec<(usize, usize)> = edges[1..].to_vec();

  // Deletion: P(G-e, k) - remove the edge
  let del_coeffs = chromatic_poly_coeffs(n, &remaining_edges);

  // Contraction: P(G/e, k) - merge v into u, renumber vertices
  let mut contracted_edges: Vec<(usize, usize)> = Vec::new();
  for &(a, b) in &remaining_edges {
    let a2 = if a == v {
      u
    } else if a > v {
      a - 1
    } else {
      a
    };
    let b2 = if b == v {
      u
    } else if b > v {
      b - 1
    } else {
      b
    };
    if a2 != b2 {
      let (x, y) = if a2 < b2 { (a2, b2) } else { (b2, a2) };
      contracted_edges.push((x, y));
    }
  }
  contracted_edges.sort();
  contracted_edges.dedup();

  let con_coeffs = chromatic_poly_coeffs(n - 1, &contracted_edges);

  // P(G, k) = P(G-e, k) - P(G/e, k)
  let max_len = del_coeffs.len().max(con_coeffs.len());
  let mut result = vec![0i128; max_len];
  for (i, &c) in del_coeffs.iter().enumerate() {
    result[i] += c;
  }
  for (i, &c) in con_coeffs.iter().enumerate() {
    result[i] -= c;
  }
  result
}

/// Convert polynomial coefficients to an expression in variable k.
fn poly_to_expr(coeffs: &[i128], k: &Expr) -> Expr {
  let mut terms: Vec<Expr> = Vec::new();
  for (i, &c) in coeffs.iter().enumerate() {
    if c == 0 {
      continue;
    }
    let term = match i {
      0 => Expr::Integer(c),
      1 => {
        if c == 1 {
          k.clone()
        } else if c == -1 {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(k.clone()),
          }
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(c)),
            right: Box::new(k.clone()),
          }
        }
      }
      _ => {
        let k_pow = Expr::BinaryOp {
          op: BinaryOperator::Power,
          left: Box::new(k.clone()),
          right: Box::new(Expr::Integer(i as i128)),
        };
        if c == 1 {
          k_pow
        } else if c == -1 {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(k_pow),
          }
        } else {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(c)),
            right: Box::new(k_pow),
          }
        }
      }
    };
    terms.push(term);
  }

  if terms.is_empty() {
    return Expr::Integer(0);
  }

  // Build the polynomial via evaluation to get canonical form
  let sum = if terms.len() == 1 {
    terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms.into(),
    }
  };

  match crate::evaluator::evaluate_expr_to_expr(&sum) {
    Ok(result) => result,
    Err(_) => sum,
  }
}

/// 1D min/max filter with edge replication padding.
fn min_max_filter_1d(data: &[f64], radius: usize, use_min: bool) -> Vec<f64> {
  let n = data.len();
  let mut result = vec![0.0; n];
  for i in 0..n {
    let lo = i.saturating_sub(radius);
    let hi = if i + radius < n { i + radius } else { n - 1 };
    let mut val = data[lo];
    for j in (lo + 1)..=hi {
      if use_min {
        val = val.min(data[j]);
      } else {
        val = val.max(data[j]);
      }
    }
    result[i] = val;
  }
  result
}

/// 2D min/max filter with edge replication padding.
fn min_max_filter_2d(
  data: &[Vec<f64>],
  radius: usize,
  use_min: bool,
) -> Vec<Vec<f64>> {
  let nrows = data.len();
  if nrows == 0 {
    return vec![];
  }
  let ncols = data[0].len();
  let mut result = vec![vec![0.0; ncols]; nrows];
  for r in 0..nrows {
    for c in 0..ncols {
      let r_lo = r.saturating_sub(radius);
      let r_hi = if r + radius < nrows {
        r + radius
      } else {
        nrows - 1
      };
      let c_lo = c.saturating_sub(radius);
      let c_hi = if c + radius < ncols {
        c + radius
      } else {
        ncols - 1
      };
      let mut val = data[r_lo][c_lo];
      for ri in r_lo..=r_hi {
        for ci in c_lo..=c_hi {
          if use_min {
            val = val.min(data[ri][ci]);
          } else {
            val = val.max(data[ri][ci]);
          }
        }
      }
      result[r][c] = val;
    }
  }
  result
}

/// Helper to extract a numeric list from Expr::List
fn expr_list_to_f64(list: &[Expr]) -> Option<Vec<f64>> {
  list
    .iter()
    .map(crate::functions::math_ast::try_eval_to_f64)
    .collect()
}

/// Helper to extract a numeric 2D matrix from Expr::List(List(...))
fn expr_matrix_to_f64(rows: &[Expr]) -> Option<Vec<Vec<f64>>> {
  rows
    .iter()
    .map(|row| {
      if let Expr::List(cols) = row {
        expr_list_to_f64(cols)
      } else {
        None
      }
    })
    .collect()
}

/// Morphological operations: Opening, Closing, Erosion, Dilation
fn morphological_op(
  name: &str,
  data_expr: &Expr,
  radius_expr: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  // Structuring-element form: the second argument is a 0/1 kernel matrix
  // (e.g. CrossMatrix[1]) rather than a scalar radius. Apply the kernel-based
  // morphology to a 2D array.
  if let (Expr::List(kitems), Expr::List(items)) = (radius_expr, data_expr)
    && !kitems.is_empty()
    && kitems.iter().all(|r| matches!(r, Expr::List(_)))
    && !items.is_empty()
    && items.iter().all(|r| matches!(r, Expr::List(_)))
  {
    let matrix = expr_matrix_to_f64(items)?;
    let kernel = expr_matrix_to_f64(kitems)?;
    let result = apply_morphological_kernel_2d(name, &matrix, &kernel);
    let is_int = items.iter().all(|row| {
      matches!(row, Expr::List(cols)
        if cols.iter().all(|e| matches!(e, Expr::Integer(_))))
    });
    let result_expr = result
      .into_iter()
      .map(|row| {
        Expr::List(
          row
            .into_iter()
            .map(|v| {
              if is_int {
                Expr::Integer(v as i128)
              } else {
                Expr::Real(v)
              }
            })
            .collect(),
        )
      })
      .collect();
    return Some(Ok(Expr::List(result_expr)));
  }

  let radius =
    crate::functions::math_ast::try_eval_to_f64(radius_expr)? as usize;

  if let Expr::Image {
    color_space: _,
    width,
    height,
    channels,
    data,
    image_type,
  } = data_expr
  {
    let w = *width as usize;
    let h = *height as usize;
    let ch = *channels as usize;
    let mut new_data = vec![0.0; data.len()];
    for c_idx in 0..ch {
      let mut channel: Vec<Vec<f64>> = (0..h)
        .map(|y| (0..w).map(|x| data[(y * w + x) * ch + c_idx]).collect())
        .collect();
      channel = apply_morphological_2d(name, &channel, radius);
      for y in 0..h {
        for x in 0..w {
          new_data[(y * w + x) * ch + c_idx] = channel[y][x];
        }
      }
    }
    return Some(Ok(Expr::Image {
      color_space: None,
      width: *width,
      height: *height,
      channels: *channels,
      data: std::sync::Arc::new(new_data),
      image_type: *image_type,
    }));
  }

  match data_expr {
    Expr::List(items) if !items.is_empty() => {
      // Check if it's a 2D matrix or 1D list
      if matches!(&items[0], Expr::List(_)) {
        // 2D case
        let matrix = expr_matrix_to_f64(items)?;
        let result = apply_morphological_2d(name, &matrix, radius);
        let is_int = items.iter().all(|row| {
          if let Expr::List(cols) = row {
            cols.iter().all(|e| matches!(e, Expr::Integer(_)))
          } else {
            false
          }
        });
        let result_expr = result
          .into_iter()
          .map(|row| {
            Expr::List(
              row
                .into_iter()
                .map(|v| {
                  if is_int {
                    Expr::Integer(v as i128)
                  } else {
                    Expr::Real(v)
                  }
                })
                .collect(),
            )
          })
          .collect();
        Some(Ok(Expr::List(result_expr)))
      } else {
        // 1D case
        let data = expr_list_to_f64(items)?;
        let result = apply_morphological_1d(name, &data, radius);
        let is_int = items.iter().all(|e| matches!(e, Expr::Integer(_)));
        let result_expr = result
          .into_iter()
          .map(|v| {
            if is_int {
              Expr::Integer(v as i128)
            } else {
              Expr::Real(v)
            }
          })
          .collect();
        Some(Ok(Expr::List(result_expr)))
      }
    }
    _ => None,
  }
}

fn apply_morphological_1d(name: &str, data: &[f64], radius: usize) -> Vec<f64> {
  match name {
    "Erosion" => min_max_filter_1d(data, radius, true),
    "Dilation" => min_max_filter_1d(data, radius, false),
    "Opening" => {
      let eroded = min_max_filter_1d(data, radius, true);
      min_max_filter_1d(&eroded, radius, false)
    }
    "Closing" => {
      let dilated = min_max_filter_1d(data, radius, false);
      min_max_filter_1d(&dilated, radius, true)
    }
    _ => data.to_vec(),
  }
}

/// One pass of kernel-based morphology. `use_min` selects erosion (min, the
/// structuring element is placed directly) versus dilation (max, the element
/// is reflected through its center). The element is truncated at the image
/// boundary — out-of-bounds samples are skipped, not counted — so an erosion
/// does not eat away the border (matching the scalar-radius behavior).
fn kernel_morph_pass(
  data: &[Vec<f64>],
  kernel: &[Vec<f64>],
  use_min: bool,
) -> Vec<Vec<f64>> {
  let h = data.len();
  let w = if h > 0 { data[0].len() } else { 0 };
  let kh = kernel.len();
  let kw = if kh > 0 { kernel[0].len() } else { 0 };
  let (cr, cc) = ((kh as i64 - 1) / 2, (kw as i64 - 1) / 2);
  let get = |r: i64, c: i64| -> Option<f64> {
    if r >= 0 && (r as usize) < h && c >= 0 && (c as usize) < w {
      Some(data[r as usize][c as usize])
    } else {
      None
    }
  };
  let mut out = vec![vec![0.0; w]; h];
  for (i, out_row) in out.iter_mut().enumerate() {
    for (j, cell) in out_row.iter_mut().enumerate() {
      let mut acc: Option<f64> = None;
      for (kr, krow) in kernel.iter().enumerate() {
        for (kc, &kv) in krow.iter().enumerate() {
          if kv != 0.0 {
            let dr = kr as i64 - cr;
            let dc = kc as i64 - cc;
            let (r, c) = if use_min {
              (i as i64 + dr, j as i64 + dc)
            } else {
              (i as i64 - dr, j as i64 - dc)
            };
            if let Some(v) = get(r, c) {
              acc = Some(match acc {
                None => v,
                Some(a) if use_min => a.min(v),
                Some(a) => a.max(v),
              });
            }
          }
        }
      }
      *cell = acc.unwrap_or(0.0);
    }
  }
  out
}

/// Kernel-based morphology dispatch: Erosion/Dilation are single passes,
/// Opening = erode then dilate, Closing = dilate then erode.
fn apply_morphological_kernel_2d(
  name: &str,
  data: &[Vec<f64>],
  kernel: &[Vec<f64>],
) -> Vec<Vec<f64>> {
  match name {
    "Erosion" => kernel_morph_pass(data, kernel, true),
    "Dilation" => kernel_morph_pass(data, kernel, false),
    "Opening" => {
      let eroded = kernel_morph_pass(data, kernel, true);
      kernel_morph_pass(&eroded, kernel, false)
    }
    "Closing" => {
      let dilated = kernel_morph_pass(data, kernel, false);
      kernel_morph_pass(&dilated, kernel, true)
    }
    _ => data.to_vec(),
  }
}

fn apply_morphological_2d(
  name: &str,
  data: &[Vec<f64>],
  radius: usize,
) -> Vec<Vec<f64>> {
  match name {
    "Erosion" => min_max_filter_2d(data, radius, true),
    "Dilation" => min_max_filter_2d(data, radius, false),
    "Opening" => {
      let eroded = min_max_filter_2d(data, radius, true);
      min_max_filter_2d(&eroded, radius, false)
    }
    "Closing" => {
      let dilated = min_max_filter_2d(data, radius, false);
      min_max_filter_2d(&dilated, radius, true)
    }
    _ => data.to_vec(),
  }
}

/// Edmonds-Karp max flow implementation (extracted to avoid bloating dispatch stack frame)
/// FindMaximumFlow using petgraph's Ford-Fulkerson algorithm
fn find_maximum_flow_impl(
  verts: &[Expr],
  edges: &[Expr],
  source: &Expr,
  sink: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let vertex_idx = |v: &Expr| -> Option<usize> {
    verts
      .iter()
      .position(|vert| crate::evaluator::pattern_matching::expr_equal(vert, v))
  };

  let s = match vertex_idx(source) {
    Some(i) => i,
    None => {
      return Ok(unevaluated("FindMaximumFlow", args));
    }
  };
  let t = match vertex_idx(sink) {
    Some(i) => i,
    None => {
      return Ok(unevaluated("FindMaximumFlow", args));
    }
  };

  let (pg_graph, _) = crate::functions::graph::build_flow_graph(verts, edges);
  let source_idx = petgraph::graph::NodeIndex::new(s);
  let sink_idx = petgraph::graph::NodeIndex::new(t);

  let (max_flow, _flow_map) =
    petgraph::algo::ford_fulkerson(&pg_graph, source_idx, sink_idx);

  Ok(Expr::Integer(max_flow as i128))
}

/// FindGraphIsomorphism implementation using backtracking
/// FindGraphIsomorphism using petgraph's VF2 algorithm
fn find_graph_isomorphism_impl(
  verts1: &[Expr],
  edges1: &[Expr],
  verts2: &[Expr],
  edges2: &[Expr],
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let n1 = verts1.len();
  let n2 = verts2.len();

  if n1 != n2 {
    return Ok(Expr::List(vec![].into()));
  }
  let n = n1;

  let max_count = if args.len() == 3 {
    match &args[2] {
      Expr::Identifier(s) if s == "All" => usize::MAX,
      Expr::Integer(k) if *k > 0 => *k as usize,
      _ => 1,
    }
  } else {
    1
  };

  // Build petgraph graphs (use DiGraph to handle both directed and undirected)
  let (pg1, _) = crate::functions::graph::build_digraph(verts1, edges1);
  let (pg2, _) = crate::functions::graph::build_digraph(verts2, edges2);

  // Check if graphs have undirected edges — if so, add reverse edges
  let has_undirected1 = edges1.iter().any(|e| {
    matches!(e, Expr::FunctionCall { name, .. } if name == "UndirectedEdge")
  });
  let has_undirected2 = edges2.iter().any(|e| {
    matches!(e, Expr::FunctionCall { name, .. } if name == "UndirectedEdge")
  });

  // For undirected graphs, we need symmetric adjacency
  let mut g1 = pg1;
  let mut g2 = pg2;
  if has_undirected1 {
    let existing: Vec<_> = g1
      .edge_references()
      .map(|e| {
        use petgraph::visit::EdgeRef;
        (e.source(), e.target())
      })
      .collect();
    for (s, t) in existing {
      if !g1.contains_edge(t, s) {
        g1.add_edge(t, s, false);
      }
    }
  }
  if has_undirected2 {
    let existing: Vec<_> = g2
      .edge_references()
      .map(|e| {
        use petgraph::visit::EdgeRef;
        (e.source(), e.target())
      })
      .collect();
    for (s, t) in existing {
      if !g2.contains_edge(t, s) {
        g2.add_edge(t, s, false);
      }
    }
  }

  // Use petgraph's is_isomorphic for simple check, fall back to manual
  // backtracking for mapping extraction since subgraph_isomorphisms_iter
  // has complex lifetime requirements
  let mut results: Vec<Vec<usize>> = Vec::new();

  // Build adjacency matrices for backtracking search
  let mut adj1 = vec![vec![false; n]; n];
  let mut adj2 = vec![vec![false; n]; n];
  for edge_ref in g1.edge_references() {
    use petgraph::visit::EdgeRef;
    adj1[edge_ref.source().index()][edge_ref.target().index()] = true;
  }
  for edge_ref in g2.edge_references() {
    use petgraph::visit::EdgeRef;
    adj2[edge_ref.source().index()][edge_ref.target().index()] = true;
  }

  let deg1: Vec<usize> = (0..n)
    .map(|i| adj1[i].iter().filter(|&&b| b).count())
    .collect();
  let deg2: Vec<usize> = (0..n)
    .map(|i| adj2[i].iter().filter(|&&b| b).count())
    .collect();

  let mut mapping = vec![usize::MAX; n];
  let mut used = vec![false; n];

  fn backtrack(
    depth: usize,
    n: usize,
    adj1: &[Vec<bool>],
    adj2: &[Vec<bool>],
    deg1: &[usize],
    deg2: &[usize],
    mapping: &mut Vec<usize>,
    used: &mut Vec<bool>,
    results: &mut Vec<Vec<usize>>,
    max_count: usize,
  ) {
    if results.len() >= max_count {
      return;
    }
    if depth == n {
      results.push(mapping.clone());
      return;
    }
    for j in 0..n {
      if used[j] || deg1[depth] != deg2[j] {
        continue;
      }
      let mut consistent = true;
      for k in 0..depth {
        if adj1[depth][k] != adj2[j][mapping[k]]
          || adj1[k][depth] != adj2[mapping[k]][j]
        {
          consistent = false;
          break;
        }
      }
      if !consistent {
        continue;
      }
      mapping[depth] = j;
      used[j] = true;
      backtrack(
        depth + 1,
        n,
        adj1,
        adj2,
        deg1,
        deg2,
        mapping,
        used,
        results,
        max_count,
      );
      used[j] = false;
    }
  }

  backtrack(
    0,
    n,
    &adj1,
    &adj2,
    &deg1,
    &deg2,
    &mut mapping,
    &mut used,
    &mut results,
    max_count,
  );

  // Convert results to associations
  let assocs: Vec<Expr> = results
    .iter()
    .map(|m| {
      let rules: Vec<Expr> = (0..n)
        .map(|i| Expr::FunctionCall {
          name: "Rule".to_string(),
          args: vec![verts1[i].clone(), verts2[m[i]].clone()].into(),
        })
        .collect();
      Expr::FunctionCall {
        name: "Association".to_string(),
        args: rules.into(),
      }
    })
    .collect();

  let result = Expr::List(assocs.into());
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// FindSpanningTree using petgraph's min_spanning_tree (unit weights)
fn find_spanning_tree_impl(
  verts: &[Expr],
  edges: &[Expr],
) -> Result<Expr, InterpreterError> {
  if verts.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vec![].into()), Expr::List(vec![].into())].into(),
    });
  }

  let (pg_graph, _) = build_undirected_graph(verts, edges);

  // Use petgraph's min_spanning_tree (Kruskal's with unit weights)
  use petgraph::algo::min_spanning_tree;
  use petgraph::data::FromElements;

  let mst: petgraph::graph::UnGraph<usize, ()> =
    petgraph::graph::UnGraph::from_elements(min_spanning_tree(&pg_graph));

  // Convert MST edges back to Wolfram Expr edges
  let mut tree_edges: Vec<Expr> = Vec::new();
  for edge_ref in mst.edge_references() {
    use petgraph::visit::EdgeRef;
    let si = edge_ref.source().index();
    let di = edge_ref.target().index();
    // Find the corresponding original edge expression
    let src_str = expr_to_string(&verts[si]);
    let dst_str = expr_to_string(&verts[di]);
    // Look up in original edges to preserve edge type
    let original_edge = edges.iter().find(|e| {
      if let Expr::FunctionCall { args: eargs, .. } = e
        && eargs.len() == 2
      {
        let a = expr_to_string(&eargs[0]);
        let b = expr_to_string(&eargs[1]);
        (a == src_str && b == dst_str) || (a == dst_str && b == src_str)
      } else {
        false
      }
    });
    if let Some(edge) = original_edge {
      tree_edges.push(edge.clone());
    } else {
      // Fallback: create UndirectedEdge
      tree_edges.push(Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![verts[si].clone(), verts[di].clone()].into(),
      });
    }
  }

  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![
      Expr::List(verts.to_vec().into()),
      Expr::List(tree_edges.into()),
    ]
    .into(),
  })
}

/// Built-in message template lookup.
/// Returns the canonical text for `sym::tag` when no user definition has
/// been installed. The `\`1\``, `\`2\`` placeholders are filled in by the
/// caller (or printed verbatim when the message is referenced directly,
/// matching wolframscript's behavior).
/// Parse a `Cycles[{{...}, {...}, ...}]` expression into a map from
/// element to image under the permutation.
fn cycles_expr_to_map(
  e: &Expr,
) -> Option<std::collections::HashMap<i128, i128>> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "Cycles" || args.len() != 1 {
    return None;
  }
  let Expr::List(cycle_list) = &args[0] else {
    return None;
  };
  let mut map = std::collections::HashMap::new();
  for cycle in cycle_list.iter() {
    let Expr::List(c) = cycle else { return None };
    let mut ints: Vec<i128> = Vec::with_capacity(c.len());
    for entry in c.iter() {
      let Expr::Integer(n) = entry else { return None };
      ints.push(*n);
    }
    if ints.len() < 2 {
      continue;
    }
    let len = ints.len();
    for i in 0..len {
      map.insert(ints[i], ints[(i + 1) % len]);
    }
  }
  Some(map)
}

/// Convert a permutation (as a from→to map, omitting fixed points) into
/// canonical `Cycles[{{...}, ...}]` form: each cycle starts at its
/// smallest element, cycles are sorted by that smallest element, and
/// trivial cycles are dropped.
fn map_to_cycles_expr(map: &std::collections::HashMap<i128, i128>) -> Expr {
  let mut visited = std::collections::HashSet::<i128>::new();
  let mut cycles: Vec<Vec<i128>> = Vec::new();
  let mut keys: Vec<i128> = map.keys().copied().collect();
  keys.sort();
  for &start in &keys {
    if visited.contains(&start) {
      continue;
    }
    let mut cycle = Vec::new();
    let mut cur = start;
    loop {
      if visited.contains(&cur) {
        break;
      }
      visited.insert(cur);
      cycle.push(cur);
      let Some(&next) = map.get(&cur) else { break };
      if next == start {
        break;
      }
      cur = next;
    }
    if cycle.len() >= 2 {
      let min_idx = cycle
        .iter()
        .enumerate()
        .min_by_key(|(_, v)| *v)
        .map(|(i, _)| i)
        .unwrap_or(0);
      cycle.rotate_left(min_idx);
      cycles.push(cycle);
    }
  }
  cycles.sort_by_key(|c| c[0]);
  let cycle_exprs: Vec<Expr> = cycles
    .into_iter()
    .map(|c| Expr::List(c.into_iter().map(Expr::Integer).collect()))
    .collect();
  Expr::FunctionCall {
    name: "Cycles".to_string(),
    args: vec![Expr::List(cycle_exprs.into())].into(),
  }
}

/// Compose a series of `Cycles[...]` permutations left-to-right
/// (σ₁ applied first, then σ₂, …). Returns None when any argument
/// is not in `Cycles[…]` form.
/// Compose permutations given as permutation lists (image lists). Returns
/// `Some(list)` only when every argument is a valid permutation list (a flat
/// list whose entries are exactly {1, ..., len}); otherwise `None` so the
/// caller can fall back to the `Cycles` path. Permutations are applied left to
/// right: `result[i] = pk[...p2[p1[i]]]`. Lists of differing lengths are
/// extended by fixed points beyond their length.
fn compose_permutation_lists(args: &[Expr]) -> Option<Expr> {
  // Empty input is the identity; leave it to the Cycles path (-> Cycles[{}]).
  if args.is_empty() {
    return None;
  }
  let mut perms: Vec<Vec<i128>> = Vec::with_capacity(args.len());
  for a in args {
    let Expr::List(items) = a else { return None };
    let mut p: Vec<i128> = Vec::with_capacity(items.len());
    for it in items.iter() {
      match it {
        Expr::Integer(n) if *n >= 1 => p.push(*n),
        _ => return None,
      }
    }
    if p.is_empty() {
      return None;
    }
    // Must be a permutation of {1, ..., len}.
    let mut sorted = p.clone();
    sorted.sort_unstable();
    if sorted
      .iter()
      .enumerate()
      .any(|(i, &v)| v != (i as i128) + 1)
    {
      return None;
    }
    perms.push(p);
  }
  let n = perms.iter().map(|p| p.len()).max().unwrap_or(0);
  let mut result: Vec<Expr> = Vec::with_capacity(n);
  for i in 1..=n as i128 {
    let mut cur = i;
    for p in &perms {
      if cur >= 1 && (cur as usize) <= p.len() {
        cur = p[(cur - 1) as usize];
      }
      // Beyond this permutation's length the point is fixed.
    }
    result.push(Expr::Integer(cur));
  }
  Some(Expr::List(result.into()))
}

/// Parse a CSS-style hex color string into channel values in [0, 1].
/// Accepts `#` followed by exactly 3, 6, or 8 hex digits (case-insensitive):
/// 3 digits is the shorthand `#rgb` (each digit doubled), 6 is `#rrggbb`, and
/// 8 is `#rrggbbaa` with an alpha channel. Returns `None` for any other form.
fn parse_hex_color(s: &str) -> Option<Vec<f64>> {
  let hex = s.strip_prefix('#')?;
  if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
    return None;
  }
  // Expand the 3-digit shorthand by doubling each digit (#abc -> #aabbcc).
  let expanded: String = match hex.len() {
    3 => hex.chars().flat_map(|c| [c, c]).collect(),
    6 | 8 => hex.to_string(),
    _ => return None,
  };
  let channels = expanded
    .as_bytes()
    .chunks(2)
    .map(|pair| {
      let byte =
        u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).ok();
      byte.map(|b| b as f64 / 255.0)
    })
    .collect::<Option<Vec<f64>>>()?;
  Some(channels)
}

fn compose_cycles_args(args: &[Expr]) -> Option<Expr> {
  let mut maps: Vec<std::collections::HashMap<i128, i128>> =
    Vec::with_capacity(args.len());
  for a in args {
    maps.push(cycles_expr_to_map(a)?);
  }
  let mut domain = std::collections::HashSet::<i128>::new();
  for m in &maps {
    for &k in m.keys() {
      domain.insert(k);
    }
  }
  let mut result = std::collections::HashMap::<i128, i128>::new();
  for &x in &domain {
    let mut cur = x;
    for m in &maps {
      cur = *m.get(&cur).unwrap_or(&cur);
    }
    if cur != x {
      result.insert(x, cur);
    }
  }
  Some(map_to_cycles_expr(&result))
}

fn builtin_message_text(sym: &Expr, tag: &Expr) -> Option<&'static str> {
  let sym_name = match sym {
    Expr::Identifier(s) => s.as_str(),
    _ => return None,
  };
  let tag_str = match tag {
    Expr::String(s) => s.as_str(),
    Expr::Identifier(s) => s.as_str(),
    _ => return None,
  };
  match (sym_name, tag_str) {
    ("General", "argr") => {
      Some("`1` called with 1 argument; `2` arguments are expected.")
    }
    _ => None,
  }
}

// ─── BSplineFunction construction ──────────────────────────────────────
// Helpers that expand `BSplineFunction[points]` (a curve) or
// `BSplineFunction[array]` (a surface) into wolframscript's structured 9-arg
// representation. Evaluation of the resulting object lives in
// `function_application.rs`.

/// Is `e` a single control point, i.e. a list whose entries are all numbers?
fn bspline_is_point(e: &Expr) -> bool {
  matches!(e, Expr::List(coords)
    if !coords.is_empty()
      && coords
        .iter()
        .all(|c| crate::functions::math_ast::try_eval_to_f64(c).is_some()))
}

/// Convert a control point to a list of `Real` coordinates (for display).
fn bspline_point_to_real(e: &Expr) -> Expr {
  match e {
    Expr::List(coords) => Expr::List(
      coords
        .iter()
        .map(|c| {
          crate::functions::math_ast::try_eval_to_f64(c)
            .map(Expr::Real)
            .unwrap_or_else(|| c.clone())
        })
        .collect(),
    ),
    _ => e.clone(),
  }
}

/// Clamped, uniform knot vector for `n` control points of degree `p`:
/// `p + 1` leading zeros, evenly-spaced interior knots, `p + 1` trailing ones.
fn bspline_clamped_knots(n: usize, p: usize) -> Vec<f64> {
  let mut knots = vec![0.0; p + 1];
  let segments = n - p; // interior knots = segments - 1
  for i in 1..segments {
    knots.push(i as f64 / segments as f64);
  }
  knots.extend(std::iter::repeat_n(1.0, p + 1));
  knots
}

fn bspline_real_list(values: &[f64]) -> Expr {
  Expr::List(values.iter().map(|&v| Expr::Real(v)).collect())
}

/// Build the structured form for `BSplineFunction[top]`, detecting whether
/// `top` is a list of control points (curve) or a grid of points (surface).
/// Returns `None` if `top` is neither, so the caller leaves the call symbolic.
fn build_bspline_function(top: &[Expr]) -> Option<Expr> {
  // Curve: every element is a control point.
  if top.iter().all(bspline_is_point) {
    let n = top.len();
    let p = std::cmp::min(3, n - 1);
    let knots = bspline_clamped_knots(n, p);
    let points: Expr =
      Expr::List(top.iter().map(bspline_point_to_real).collect());
    return Some(bspline_structured(1, &[p], &[points], &[knots]));
  }

  // Surface: every element is a row of control points, all rows equal length.
  if top.iter().all(|row| {
    matches!(row, Expr::List(pts)
    if !pts.is_empty() && pts.iter().all(bspline_is_point))
  }) {
    let rows: Vec<&[Expr]> = top
      .iter()
      .filter_map(|row| match row {
        Expr::List(pts) => Some(pts.as_slice()),
        _ => None,
      })
      .collect();
    let n_u = rows.len();
    let n_v = rows[0].len();
    if rows.iter().any(|r| r.len() != n_v) {
      return None;
    }
    let p_u = std::cmp::min(3, n_u - 1);
    let p_v = std::cmp::min(3, n_v - 1);
    let grid: Expr = Expr::List(
      rows
        .iter()
        .map(|r| Expr::List(r.iter().map(bspline_point_to_real).collect()))
        .collect(),
    );
    return Some(bspline_structured(
      2,
      &[p_u, p_v],
      &[grid],
      &[
        bspline_clamped_knots(n_u, p_u),
        bspline_clamped_knots(n_v, p_v),
      ],
    ));
  }

  None
}

/// Assemble the 9-argument structured `BSplineFunction[...]` expression.
fn bspline_structured(
  dim: usize,
  degrees: &[usize],
  net: &[Expr],
  knots: &[Vec<f64>],
) -> Expr {
  let ranges: Expr = Expr::List(
    (0..dim)
      .map(|_| Expr::List(vec![Expr::Real(0.0), Expr::Real(1.0)].into()))
      .collect(),
  );
  let degree_list: Expr =
    Expr::List(degrees.iter().map(|&d| Expr::Integer(d as i128)).collect());
  let closed: Expr = Expr::List((0..dim).map(|_| bool_expr(false)).collect());
  let mut net_slot: Vec<Expr> = net.to_vec();
  net_slot.push(Expr::Identifier("Automatic".to_string()));
  let knot_lists: Expr =
    Expr::List(knots.iter().map(|k| bspline_real_list(k)).collect());
  let zeros: Expr = Expr::List((0..dim).map(|_| Expr::Integer(0)).collect());
  Expr::FunctionCall {
    name: "BSplineFunction".to_string(),
    args: vec![
      Expr::Integer(dim as i128),
      ranges,
      degree_list,
      closed,
      Expr::List(net_slot.into()),
      knot_lists,
      zeros,
      Expr::Identifier("MachinePrecision".to_string()),
      Expr::String("Unevaluated".to_string()),
    ]
    .into(),
  }
}
