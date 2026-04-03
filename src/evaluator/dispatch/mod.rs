#[allow(unused_imports)]
use super::*;

// Re-export crate types/functions for submodules (used by submodules via `use super::*`)
#[allow(unused_imports)]
pub(crate) use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
};
#[allow(unused_imports)]
pub(crate) use crate::{
  ENV, InterpreterError, PART_DEPTH, StoredValue, format_real_result, interpret,
};

pub mod arg_count;
mod association_functions;
mod attributes;
mod boolean_functions;
mod calculus_functions;
pub mod complex_and_special;
pub use complex_and_special::builtin_default_value;
pub use complex_and_special::builtin_default_value_at_position;
mod datetime_functions;
mod evaluation_control;
mod image_functions;
mod interval_functions;
mod io_functions;
mod linear_algebra_functions;
mod list_operations;
mod math_functions;
mod plotting;
mod polynomial_functions;
mod predicate_functions;
mod quantity_functions;
mod string_functions;
mod structural;

pub use association_functions::*;
pub use attributes::*;
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
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
  }

  stacker::maybe_grow(2 * 1024 * 1024, 4 * 1024 * 1024, || {
    evaluate_function_call_ast_inner(name, args)
  })
}

/// Helper for Read: read a single value of a given type from remaining stream content.
/// Returns (result_expr, bytes_consumed).
pub fn read_single_type(remaining: &str, read_type: &Expr) -> (Expr, usize) {
  let type_name = match read_type {
    Expr::Identifier(s) => s.as_str(),
    _ => "Expression",
  };

  if remaining.is_empty() {
    return (Expr::Identifier("EndOfFile".to_string()), 0);
  }

  match type_name {
    "Word" => {
      // Skip leading whitespace
      let trimmed = remaining.trim_start();
      let skipped = remaining.len() - trimmed.len();
      if trimmed.is_empty() {
        return (Expr::Identifier("EndOfFile".to_string()), remaining.len());
      }
      // Read until whitespace
      let end = trimmed
        .find(|c: char| c.is_whitespace())
        .unwrap_or(trimmed.len());
      let word = &trimmed[..end];
      (Expr::String(word.to_string()), skipped + end)
    }
    "Number" => {
      // Skip leading whitespace
      let trimmed = remaining.trim_start();
      let skipped = remaining.len() - trimmed.len();
      if trimmed.is_empty() {
        return (Expr::Identifier("EndOfFile".to_string()), remaining.len());
      }
      // Try to parse a number
      let mut end = 0;
      let chars: Vec<char> = trimmed.chars().collect();
      // Optional sign
      if end < chars.len() && (chars[end] == '+' || chars[end] == '-') {
        end += 1;
      }
      // Digits before decimal
      let start_digits = end;
      while end < chars.len() && chars[end].is_ascii_digit() {
        end += 1;
      }
      let has_int_part = end > start_digits;
      // Decimal point and more digits
      let mut is_real = false;
      if end < chars.len() && chars[end] == '.' {
        is_real = true;
        end += 1;
        while end < chars.len() && chars[end].is_ascii_digit() {
          end += 1;
        }
      }
      if end == 0 || (!has_int_part && !is_real) {
        return (Expr::Identifier("$Failed".to_string()), skipped);
      }
      let num_str = &trimmed[..end];
      if is_real {
        if let Ok(f) = num_str.parse::<f64>() {
          return (Expr::Real(f), skipped + end);
        }
      } else if let Ok(n) = num_str.parse::<i128>() {
        return (Expr::Integer(n), skipped + end);
      }
      (Expr::Identifier("$Failed".to_string()), skipped)
    }
    "String" => {
      // Read until newline
      let end = remaining.find('\n').unwrap_or(remaining.len());
      let line = &remaining[..end];
      let advance = if end < remaining.len() { end + 1 } else { end };
      (Expr::String(line.to_string()), advance)
    }
    "Character" => {
      let ch = remaining.chars().next().unwrap();
      (Expr::String(ch.to_string()), ch.len_utf8())
    }
    "Expression" | _ => {
      // Read until newline and try to interpret as expression
      let trimmed = remaining.trim_start();
      let skipped = remaining.len() - trimmed.len();
      if trimmed.is_empty() {
        return (Expr::Identifier("EndOfFile".to_string()), remaining.len());
      }
      let end = trimmed.find('\n').unwrap_or(trimmed.len());
      let line = &trimmed[..end];
      let advance = if skipped + end < remaining.len() {
        skipped + end + 1
      } else {
        remaining.len()
      };
      match crate::interpret(line) {
        Ok(result_str) => {
          let expr = crate::syntax::string_to_expr(&result_str)
            .unwrap_or(Expr::Identifier(result_str));
          (expr, advance)
        }
        Err(_) => (Expr::Identifier("$Failed".to_string()), advance),
      }
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

  // Extract explicit options from the argument (could be a Sequence of rules)
  let explicit_rules = match opts_arg {
    Expr::FunctionCall { name, args } if name == "Sequence" => args.clone(),
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => vec![opts_arg.clone()],
    _ => vec![],
  };

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

/// Extract the key-value pair from a Rule or RuleDelayed expression
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
        _ => expr_to_string(pattern),
      };
      Some((key, *replacement.clone()))
    }
    _ => None,
  }
}

pub fn evaluate_function_call_ast_inner(
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
        _ => flat_args.push(arg.clone()),
      }
    }
    args_after_flat = flat_args;
    &args_after_flat[..]
  } else {
    args
  };

  // Apply Orderless attribute: sort arguments into canonical order
  let has_orderless = is_builtin_orderless(name)
    || crate::FUNC_ATTRS.with(|m| {
      m.borrow()
        .get(name)
        .is_some_and(|attrs| attrs.contains(&"Orderless".to_string()))
    });
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

  // Check for user-defined functions (before built-in dispatch, so user
  // overrides take precedence — matching Wolfram Language semantics)
  // Clone overloads to avoid holding the borrow across evaluate calls
  let overloads = crate::FUNC_DEFS.with(|m| {
    let defs = m.borrow();
    defs.get(name).cloned()
  });
  let inline_opts_overloads = crate::FUNC_OPTS_INLINE.with(|m| {
    let map = m.borrow();
    map.get(name).cloned()
  });

  if let Some(overloads) = overloads {
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
        let effective_args = match distribution {
          Some(dist) => {
            let mut eff = Vec::with_capacity(params.len());
            for (i, param_args) in dist.iter().enumerate() {
              if blank_types[i] >= 2 {
                if param_args.len() == 1 {
                  eff.push(param_args[0].clone());
                } else {
                  eff.push(Expr::FunctionCall {
                    name: "Sequence".to_string(),
                    args: param_args.clone(),
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
                let mut positional_ctx: Vec<(String, crate::syntax::Expr)> =
                  Vec::new();
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
                crate::evaluator::pattern_matching::push_match_context_pub(
                  &positional_ctx,
                );
                let match_result =
                  crate::evaluator::pattern_matching::match_pattern(
                    &canonical_arg,
                    pattern,
                  );
                crate::evaluator::pattern_matching::pop_match_context_pub();
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
        let effective_args = if perm_args.len() == total_count {
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
                  let mut positional_ctx: Vec<(String, crate::syntax::Expr)> =
                    Vec::new();
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
                  crate::evaluator::pattern_matching::push_match_context_pub(
                    &positional_ctx,
                  );
                  let match_result =
                    crate::evaluator::pattern_matching::match_pattern(
                      &canonical_arg,
                      pattern,
                    );
                  crate::evaluator::pattern_matching::pop_match_context_pub();
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
            // Evaluate the condition - it must return True
            match evaluate_expr_to_expr(&substituted_cond) {
              Ok(Expr::Identifier(ref s)) if s == "True" => {} // condition met
              _ => {
                conditions_met = false;
                break;
              }
            }
          }
        }
        if !conditions_met {
          continue; // try next permutation (or next overload if no more permutations)
        }
        overload_matched = true;
        // All conditions met - substitute parameters with arguments and evaluate body
        // Use simultaneous substitution to prevent variable name leakage
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

  // Begin["ctx`"] / BeginPackage["ctx`"] push context and return the context string
  if (name == "Begin" || name == "BeginPackage")
    && args.len() == 1
    && let Expr::String(ctx) = &args[0]
  {
    crate::push_context(ctx.clone());
    return Ok(Expr::String(ctx.clone()));
  }
  // End[] pops the context stack and returns the ended context
  if name == "End" && args.is_empty() {
    let ctx = crate::pop_context().unwrap_or_else(|| "Global`".to_string());
    return Ok(Expr::String(ctx));
  }
  // EndPackage[] pops the context stack and returns Null
  if name == "EndPackage" && args.is_empty() {
    crate::pop_context();
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
    || name == "Off"
    || name == "On"
    || name == "Remove"
    || name == "ClearAttributes"
  {
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // SetOptions is not implemented - return unevaluated
  if name == "SetOptions" {
    return Ok(Expr::FunctionCall {
      name: "SetOptions".to_string(),
      args: args.to_vec(),
    });
  }

  // Circle[] defaults to Circle[{0, 0}]
  if name == "Circle" {
    let center = if args.is_empty() {
      Expr::List(vec![Expr::Integer(0), Expr::Integer(0)])
    } else {
      args[0].clone()
    };
    let mut new_args = vec![center];
    if args.len() > 1 {
      new_args.extend_from_slice(&args[1..]);
    }
    return Ok(Expr::FunctionCall {
      name: "Circle".to_string(),
      args: new_args,
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

  // Disk[] → Disk[{0, 0}] (default center)
  if name == "Disk" && args.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Disk".to_string(),
      args: vec![Expr::List(vec![Expr::Integer(0), Expr::Integer(0)])],
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
    | "Colon"
    | "Cap"
    | "Congruent"
    | "DirectedEdge"
    | "RightTee"
    | "UndirectedEdge"
    | "Entity"
    | "InfiniteLine"
    | "Ball"
    | "PlusMinus"
    | "CircleTimes"
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
    | "Glow"
    | "PrincipalValue"
    | "UpTo"
    | "LetterCharacter"
    | "ToRadicals"
    | "Longest"
    | "GenerateConditions"
    | "OverTilde"
    | "AngleBracket"
    | "Larger"
    | "ZetaZero"
    | "MixtureDistribution"
    | "Inactivate"
    | "LegendLabel" => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      });
    }
    _ => {}
  }

  // Permute[list, perm] — permute list elements
  if name == "Permute" && args.len() == 2 {
    if let Expr::List(list) = &args[0] {
      // Permute[list, {p1, p2, ...}] — permutation list form
      // Element at position i goes to position perm[i]
      if let Expr::List(perm) = &args[1] {
        if perm.len() != list.len() {
          return Ok(Expr::FunctionCall {
            name: name.to_string(),
            args: args.to_vec(),
          });
        }
        let mut result = vec![Expr::Integer(0); list.len()];
        for (i, p) in perm.iter().enumerate() {
          if let Some(idx) = expr_to_i128(p) {
            if idx >= 1 && (idx as usize) <= list.len() {
              result[(idx - 1) as usize] = list[i].clone();
            } else {
              return Ok(Expr::FunctionCall {
                name: name.to_string(),
                args: args.to_vec(),
              });
            }
          } else {
            return Ok(Expr::FunctionCall {
              name: name.to_string(),
              args: args.to_vec(),
            });
          }
        }
        return Ok(Expr::List(result));
      }
      // Permute[list, Cycles[{...}]] — cycle notation
      if let Expr::FunctionCall {
        name: cname,
        args: cargs,
      } = &args[1]
        && cname == "Cycles"
        && cargs.len() == 1
        && let Expr::List(cycle_list) = &cargs[0]
      {
        let mut result = list.clone();
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
        return Ok(Expr::List(result));
      }
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
      return Ok(Expr::List(result));
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
  }

  // CompleteGraph[n] → Graph[{1,...,n}, {UndirectedEdge[i,j] for all i<j}]
  if name == "CompleteGraph" && args.len() == 1 {
    if let Expr::Integer(n) = &args[0] {
      let n = *n as usize;
      let vertices: Vec<Expr> =
        (1..=n).map(|i| Expr::Integer(i as i128)).collect();
      let mut edges = Vec::new();
      for i in 1..=n {
        for j in (i + 1)..=n {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![Expr::Integer(i as i128), Expr::Integer(j as i128)],
          });
        }
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vertices), Expr::List(edges)],
      });
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
          args: vec![Expr::Integer(1), Expr::Integer(i as i128)],
        })
        .collect();
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vertices), Expr::List(edges)],
      });
    } else if n == 1 {
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vec![Expr::Integer(1)]), Expr::List(vec![])],
      });
    }
  }

  // CirculantGraph[n, {j1, j2, ...}] — circulant graph
  if name == "CirculantGraph"
    && args.len() == 2
    && let (Expr::Integer(n), Expr::List(jumps)) = (&args[0], &args[1])
  {
    let n = *n as usize;
    let vertices: Vec<Expr> =
      (1..=n).map(|i| Expr::Integer(i as i128)).collect();
    let mut edges = Vec::new();
    let jump_vals: Vec<usize> = jumps
      .iter()
      .filter_map(|j| {
        if let Expr::Integer(v) = j {
          Some(*v as usize)
        } else {
          None
        }
      })
      .collect();
    for i in 1..=n {
      for &j in &jump_vals {
        let target = ((i - 1 + j) % n) + 1;
        if i < target {
          edges.push(Expr::FunctionCall {
            name: "UndirectedEdge".to_string(),
            args: vec![Expr::Integer(i as i128), Expr::Integer(target as i128)],
          });
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices), Expr::List(edges)],
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
              ],
            });
          }
        }
      }
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(vertices), Expr::List(edges)],
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
            ],
          });
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices), Expr::List(edges)],
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
            ],
          });
        }
      }
    }

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices), Expr::List(edges)],
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
          ],
        });
      }
    }

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices), Expr::List(edges)],
    });
  }

  // RecurrenceFilter[{{a0, a1, ...}, {b0, b1, ...}}, data] → IIR filter
  if name == "RecurrenceFilter"
    && args.len() == 2
    && let (Expr::List(coeffs), Expr::List(data)) = (&args[0], &args[1])
    && coeffs.len() == 2
    && let (Expr::List(a_coeffs), Expr::List(b_coeffs)) =
      (&coeffs[0], &coeffs[1])
    && !a_coeffs.is_empty()
  {
    // y[n] = (sum_j b[j]*x[n-j] - sum_i(i>0) a[i]*y[n-i]) / a[0]
    let n = data.len();
    let mut output: Vec<Expr> = Vec::with_capacity(n);
    for i in 0..n {
      // Feedforward: sum b[j] * x[n-j]
      let mut sum = Expr::Integer(0);
      for (j, bj) in b_coeffs.iter().enumerate() {
        if i >= j {
          let term = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(bj.clone()),
            right: Box::new(data[i - j].clone()),
          };
          sum = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(sum),
            right: Box::new(term),
          };
        }
      }
      // Feedback: subtract a[k]*y[n-k] for k >= 1
      for (k, ak) in a_coeffs.iter().enumerate().skip(1) {
        if i >= k {
          let term = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(ak.clone()),
            right: Box::new(output[i - k].clone()),
          };
          sum = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Minus,
            left: Box::new(sum),
            right: Box::new(term),
          };
        }
      }
      // Divide by a[0]
      let result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(sum),
        right: Box::new(a_coeffs[0].clone()),
      };
      match evaluate_expr_to_expr(&result) {
        Ok(v) => output.push(v),
        Err(e) => return Err(e),
      }
    }
    return Ok(Expr::List(output));
  }

  // CantorMesh[n] → Cantor set at level n as a MeshRegion
  if name == "CantorMesh"
    && args.len() == 1
    && let Some(n) = crate::functions::math_ast::try_eval_to_f64(&args[0])
  {
    let n = n as usize;
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
    let mut points: Vec<f64> = Vec::new();
    for (a, b) in &segments {
      points.push(*a);
      points.push(*b);
    }
    let vertex_exprs: Vec<Expr> = points
      .iter()
      .map(|x| Expr::List(vec![Expr::Real(*x)]))
      .collect();
    let line_pairs: Vec<Expr> = (0..segments.len())
      .map(|i| {
        Expr::List(vec![
          Expr::Integer((2 * i + 1) as i128),
          Expr::Integer((2 * i + 2) as i128),
        ])
      })
      .collect();
    return Ok(Expr::FunctionCall {
      name: "MeshRegion".to_string(),
      args: vec![
        Expr::List(vertex_exprs),
        Expr::List(vec![Expr::FunctionCall {
          name: "Line".to_string(),
          args: vec![Expr::List(line_pairs)],
        }]),
      ],
    });
  }

  // ArrayMesh[matrix] → MeshRegion from a binary 2D array
  if name == "ArrayMesh"
    && args.len() == 1
    && let Expr::List(rows) = &args[0]
  {
    let nrows = rows.len();
    if nrows == 0 {
      return Ok(Expr::FunctionCall {
        name: "ArrayMesh".to_string(),
        args: args.to_vec(),
      });
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
        return Ok(Expr::FunctionCall {
          name: "ArrayMesh".to_string(),
          args: args.to_vec(),
        });
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
          polygons.push(Expr::List(vec![
            Expr::Integer(br as i128),
            Expr::Integer(tr as i128),
            Expr::Integer(tl as i128),
            Expr::Integer(bl as i128),
          ]));
        }
      }
    }

    // Build MeshRegion[vertices, {Polygon[polygons]}]
    let vertex_exprs: Vec<Expr> = vertices
      .iter()
      .map(|(x, y)| Expr::List(vec![Expr::Real(*x), Expr::Real(*y)]))
      .collect();

    return Ok(Expr::FunctionCall {
      name: "MeshRegion".to_string(),
      args: vec![
        Expr::List(vertex_exprs),
        Expr::List(vec![Expr::FunctionCall {
          name: "Polygon".to_string(),
          args: vec![Expr::List(polygons)],
        }]),
      ],
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
            ],
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
          ],
        });
        walk_expr(left, counter, vertices, edges);
        let right_id = *counter + 1;
        edges.push(Expr::FunctionCall {
          name: "UndirectedEdge".to_string(),
          args: vec![
            Expr::Integer(my_id as i128),
            Expr::Integer(right_id as i128),
          ],
        });
        walk_expr(right, counter, vertices, edges);
      }
      // Atoms (Integer, Real, Identifier, String, etc.) have no children

      my_id
    }

    walk_expr(&args[0], &mut counter, &mut vertices, &mut edges);

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices), Expr::List(edges)],
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
      ga.push(Expr::List(vec![Expr::Rule {
        pattern: Box::new(Expr::Identifier("GraphLayout".to_string())),
        replacement: Box::new(Expr::String("TutteEmbedding".to_string())),
      }]));
    }
    return Ok(graph);
  }

  // Graph[{rule1, rule2, ...}] or Graph[{edge1, edge2, ...}]
  // → Graph[{sorted vertices}, {DirectedEdge/UndirectedEdge[...], ...}]
  if name == "Graph" {
    if args.len() == 1
      && let Expr::List(edges) = &args[0]
    {
      // Check if all elements are Rule expressions
      let all_rules = edges.iter().all(|e| matches!(e, Expr::Rule { .. }));
      if all_rules && !edges.is_empty() {
        let mut vertex_set: Vec<Expr> = Vec::new();
        let mut directed_edges: Vec<Expr> = Vec::new();
        for e in edges {
          if let Expr::Rule {
            pattern,
            replacement,
          } = e
          {
            let src = (**pattern).clone();
            let dst = (**replacement).clone();
            if !vertex_set
              .iter()
              .any(|v| crate::evaluator::pattern_matching::expr_equal(v, &src))
            {
              vertex_set.push(src.clone());
            }
            if !vertex_set
              .iter()
              .any(|v| crate::evaluator::pattern_matching::expr_equal(v, &dst))
            {
              vertex_set.push(dst.clone());
            }
            directed_edges.push(Expr::FunctionCall {
              name: "DirectedEdge".to_string(),
              args: vec![src, dst],
            });
          }
        }
        vertex_set.sort_by(crate::functions::canonical_cmp);
        return Ok(Expr::FunctionCall {
          name: "Graph".to_string(),
          args: vec![Expr::List(vertex_set), Expr::List(directed_edges)],
        });
      }

      // Check if all elements are UndirectedEdge/DirectedEdge
      let all_edges = edges.iter().all(|e| {
        matches!(e, Expr::FunctionCall { name, args } if (name == "UndirectedEdge" || name == "DirectedEdge") && args.len() == 2)
      });
      if all_edges && !edges.is_empty() {
        let mut vertex_set: Vec<Expr> = Vec::new();
        for e in edges {
          if let Expr::FunctionCall { args: eargs, .. } = e {
            for v in eargs {
              if !vertex_set.iter().any(|existing| {
                crate::evaluator::pattern_matching::expr_equal(existing, v)
              }) {
                vertex_set.push(v.clone());
              }
            }
          }
        }
        vertex_set.sort_by(crate::functions::canonical_cmp);
        return Ok(Expr::FunctionCall {
          name: "Graph".to_string(),
          args: vec![Expr::List(vertex_set), Expr::List(edges.clone())],
        });
      }
    }
    // Fall through: return as inert
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
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
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
          return Ok(Expr::FunctionCall {
            name: name.to_string(),
            args: args.to_vec(),
          });
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
            args: new_args,
          }
        } else {
          e.clone()
        }
      })
      .collect();

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(new_vertices), Expr::List(new_edges)],
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
      other => vec![other.clone()],
    };

    for v in to_add {
      if !existing.contains(&expr_to_string(&v)) {
        new_vertices.push(v);
      }
    }

    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(new_vertices), Expr::List(edges.clone())],
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
        return Ok(Expr::FunctionCall {
          name: name.to_string(),
          args: args.to_vec(),
        });
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
      args: vec![Expr::List(all_vertices), Expr::List(common_edges)],
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
    return Ok(Expr::Identifier(
      if found { "True" } else { "False" }.to_string(),
    ));
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
      return Ok(Expr::Identifier(
        if all_undirected { "True" } else { "False" }.to_string(),
      ));
    }
    return Ok(Expr::Identifier("False".to_string()));
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
      return Ok(Expr::Identifier(
        if all_directed { "True" } else { "False" }.to_string(),
      ));
    }
    return Ok(Expr::Identifier("False".to_string()));
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
        return Ok(Expr::Identifier("False".to_string()));
      }
      // Check connectivity via BFS/DFS
      let adj = build_undirected_adj(vertices, edges);
      let connected = is_connected(&adj, n);
      return Ok(Expr::Identifier(
        if connected { "True" } else { "False" }.to_string(),
      ));
    }
    return Ok(Expr::Identifier("False".to_string()));
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
      return Ok(Expr::Identifier(
        if is_acyclic { "True" } else { "False" }.to_string(),
      ));
    }
    return Ok(Expr::Identifier("False".to_string()));
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
        return Ok(Expr::Identifier("True".to_string()));
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
        let mut adj = vec![vec![]; n];
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
              adj[fi].push(ti);
              adj[ti].push(fi);
            }
          }
        }
        let balanced = (0..n).all(|i| in_deg[i] == out_deg[i]);
        let connected = is_connected(&adj, n);
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
        let adj = build_undirected_adj(vertices, edges);
        let all_even = adj.iter().all(|neighbors| neighbors.len() % 2 == 0);
        let connected = is_connected(&adj, n);
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
    return Ok(Expr::Identifier("False".to_string()));
  }

  // GraphDiameter, VertexEccentricity, GraphCenter, GraphPeriphery, GraphRadius
  if (name == "GraphDiameter"
    || name == "VertexEccentricity"
    || name == "GraphCenter"
    || name == "GraphPeriphery"
    || name == "GraphRadius")
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
      let adj = build_undirected_adj(vertices, edges);
      let vertex_strs: Vec<String> =
        vertices.iter().map(expr_to_string).collect();

      // Compute eccentricities via BFS from each vertex
      let eccentricities: Vec<i128> =
        (0..n).map(|start| bfs_max_dist(&adj, start, n)).collect();

      match name {
        "GraphDiameter" => {
          let diameter = eccentricities.iter().copied().max().unwrap_or(0);
          return Ok(Expr::Integer(diameter));
        }
        "GraphRadius" => {
          let radius = eccentricities.iter().copied().min().unwrap_or(0);
          return Ok(Expr::Integer(radius));
        }
        "VertexEccentricity" if args.len() == 2 => {
          let target = expr_to_string(&args[1]);
          if let Some(idx) = vertex_strs.iter().position(|v| v == &target) {
            return Ok(Expr::Integer(eccentricities[idx]));
          }
        }
        "GraphCenter" => {
          let min_ecc = eccentricities.iter().copied().min().unwrap_or(0);
          let center: Vec<Expr> = vertices
            .iter()
            .enumerate()
            .filter(|(i, _)| eccentricities[*i] == min_ecc)
            .map(|(_, v)| v.clone())
            .collect();
          return Ok(Expr::List(center));
        }
        "GraphPeriphery" => {
          let max_ecc = eccentricities.iter().copied().max().unwrap_or(0);
          let periphery: Vec<Expr> = vertices
            .iter()
            .enumerate()
            .filter(|(i, _)| eccentricities[*i] == max_ecc)
            .map(|(_, v)| v.clone())
            .collect();
          return Ok(Expr::List(periphery));
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
    let adj = build_undirected_adj(vertices, edges);
    // DegreeCentrality returns raw vertex degree (not normalized)
    let centralities: Vec<Expr> = adj
      .iter()
      .map(|neighbors| Expr::Integer(neighbors.len() as i128))
      .collect();
    return Ok(Expr::List(centralities));
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
            args: vec![vertices[i].clone(), vertices[j].clone()],
          });
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices.clone()), Expr::List(comp_edges)],
    });
  }

  // VertexOutComponent[graph, v] — vertices reachable from v
  if name == "VertexOutComponent"
    && args.len() == 2
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
    let target = expr_to_string(&args[1]);
    if let Some(start) = vertex_strs.iter().position(|v| v == &target) {
      let adj = build_undirected_adj(vertices, edges);
      // BFS from start
      let mut visited = vec![false; n];
      let mut queue = std::collections::VecDeque::new();
      visited[start] = true;
      queue.push_back(start);
      let mut component = Vec::new();
      while let Some(v) = queue.pop_front() {
        component.push(vertices[v].clone());
        for &u in &adj[v] {
          if !visited[u] {
            visited[u] = true;
            queue.push_back(u);
          }
        }
      }
      return Ok(Expr::List(component));
    }
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
    let n = vertices.len();
    let adj = build_undirected_adj(vertices, edges);
    // Closeness centrality = (n-1) / sum of distances to all other vertices
    // ClosenessCentrality returns machine precision floats
    let centralities: Vec<Expr> = (0..n)
      .map(|start| {
        let dists = bfs_all_dists(&adj, start, n);
        let total_dist: i128 = dists.iter().filter(|&&d| d > 0).sum();
        if total_dist > 0 {
          let val = (n as f64 - 1.0) / (total_dist as f64);
          Expr::Real(val)
        } else {
          Expr::Real(0.0)
        }
      })
      .collect();
    return Ok(Expr::List(centralities));
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
    let adj = build_undirected_adj(vertices, edges);
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
        for &w in &adj[v] {
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
    return Ok(Expr::List(centralities));
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
    let adj = build_undirected_adj(vertices, edges);
    // Build neighbor sets for quick lookup
    let neighbor_sets: Vec<std::collections::HashSet<usize>> = adj
      .iter()
      .map(|neighbors| neighbors.iter().copied().collect())
      .collect();

    let coefficients: Vec<Expr> = (0..n)
      .map(|v| {
        let k = adj[v].len();
        if k < 2 {
          return Expr::Integer(0);
        }
        let mut triangles = 0i128;
        let neighbors = &adj[v];
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
            ],
          }
        }
      })
      .collect();
    return Ok(Expr::List(coefficients));
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

  // ButterflyGraph[n] — butterfly graph with 2n+1 vertices
  if name == "ButterflyGraph"
    && args.len() == 1
    && let Expr::Integer(n) = &args[0]
  {
    let n = *n as usize;
    // Butterfly graph: two cycles of size n sharing one vertex
    // Vertices: 1..=2n+1, center is vertex 1
    let total = 2 * n + 1;
    let vertices: Vec<Expr> =
      (1..=total).map(|i| Expr::Integer(i as i128)).collect();
    let mut edges = Vec::new();
    // First wing: vertices 1, 2, ..., n+1 form a cycle
    for i in 0..n {
      let a = i + 2; // 2, 3, ..., n+1
      let b = if i + 1 < n { i + 3 } else { 2 };
      edges.push(Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)],
      });
    }
    // Connect first wing to center
    edges.push(Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![Expr::Integer(1), Expr::Integer(2)],
    });
    edges.push(Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![Expr::Integer(1), Expr::Integer((n + 1) as i128)],
    });
    // Second wing: vertices 1, n+2, ..., 2n+1 form a cycle
    for i in 0..n {
      let a = n + 1 + i + 1; // n+2, n+3, ..., 2n+1
      let b = if i + 1 < n { n + 1 + i + 2 } else { n + 2 };
      edges.push(Expr::FunctionCall {
        name: "UndirectedEdge".to_string(),
        args: vec![Expr::Integer(a as i128), Expr::Integer(b as i128)],
      });
    }
    // Connect second wing to center
    edges.push(Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![Expr::Integer(1), Expr::Integer((n + 2) as i128)],
    });
    edges.push(Expr::FunctionCall {
      name: "UndirectedEdge".to_string(),
      args: vec![Expr::Integer(1), Expr::Integer((2 * n + 1) as i128)],
    });
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices), Expr::List(edges)],
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
                ],
              });
            }
          }
        }
      }
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vertices), Expr::List(edges)],
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
      return Ok(Expr::List(matrix.into_iter().map(Expr::List).collect()));
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
      let n = vertices.len();
      let vertex_index: std::collections::HashMap<String, usize> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (expr_to_string(v), i))
        .collect();

      // Check if graph is directed or undirected
      let is_directed = edges.iter().any(|e| {
            matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedEdge")
          });

      let comp_list = if is_directed {
        // Strongly connected components via Kosaraju's algorithm
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut radj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for edge in edges {
          if let Expr::FunctionCall { args: eargs, .. } = edge
            && eargs.len() == 2
          {
            let from_str = expr_to_string(&eargs[0]);
            let to_str = expr_to_string(&eargs[1]);
            if let (Some(&fi), Some(&ti)) =
              (vertex_index.get(&from_str), vertex_index.get(&to_str))
            {
              adj[fi].push(ti);
              radj[ti].push(fi);
            }
          }
        }

        // Pass 1: DFS on original graph to get finish order
        let mut visited = vec![false; n];
        let mut order: Vec<usize> = Vec::new();
        for i in 0..n {
          if !visited[i] {
            let mut stack = vec![(i, false)];
            while let Some((node, processed)) = stack.pop() {
              if processed {
                order.push(node);
                continue;
              }
              if visited[node] {
                continue;
              }
              visited[node] = true;
              stack.push((node, true));
              for &next in &adj[node] {
                if !visited[next] {
                  stack.push((next, false));
                }
              }
            }
          }
        }

        // Pass 2: DFS on reverse graph in reverse finish order
        let mut comp_id = vec![usize::MAX; n];
        let mut components: Vec<Vec<usize>> = Vec::new();
        for &node in order.iter().rev() {
          if comp_id[node] != usize::MAX {
            continue;
          }
          let cid = components.len();
          let mut component = Vec::new();
          let mut stack = vec![node];
          while let Some(v) = stack.pop() {
            if comp_id[v] != usize::MAX {
              continue;
            }
            comp_id[v] = cid;
            component.push(v);
            for &prev in &radj[v] {
              if comp_id[prev] == usize::MAX {
                stack.push(prev);
              }
            }
          }
          components.push(component);
        }

        // Sort components by size (largest first), matching undirected behavior
        components.sort_by(|a, b| b.len().cmp(&a.len()));

        // Convert to Expr lists
        components
          .into_iter()
          .map(|comp| comp.into_iter().map(|i| vertices[i].clone()).collect())
          .collect::<Vec<Vec<Expr>>>()
      } else {
        // Undirected: Union-Find
        let mut parent: Vec<usize> = (0..n).collect();
        let mut uf_rank = vec![0usize; n];

        fn find(parent: &mut Vec<usize>, i: usize) -> usize {
          if parent[i] != i {
            parent[i] = find(parent, parent[i]);
          }
          parent[i]
        }

        fn union(
          parent: &mut Vec<usize>,
          uf_rank: &mut Vec<usize>,
          a: usize,
          b: usize,
        ) {
          let ra = find(parent, a);
          let rb = find(parent, b);
          if ra == rb {
            return;
          }
          if uf_rank[ra] < uf_rank[rb] {
            parent[ra] = rb;
          } else if uf_rank[ra] > uf_rank[rb] {
            parent[rb] = ra;
          } else {
            parent[rb] = ra;
            uf_rank[ra] += 1;
          }
        }

        for edge in edges {
          if let Expr::FunctionCall { args: eargs, .. } = edge
            && eargs.len() == 2
          {
            let from_str = expr_to_string(&eargs[0]);
            let to_str = expr_to_string(&eargs[1]);
            if let (Some(&fi), Some(&ti)) =
              (vertex_index.get(&from_str), vertex_index.get(&to_str))
            {
              union(&mut parent, &mut uf_rank, fi, ti);
            }
          }
        }

        // Group vertices by their root, preserving insertion order
        let mut components: Vec<Vec<Expr>> = Vec::new();
        let mut root_to_idx: std::collections::HashMap<usize, usize> =
          std::collections::HashMap::new();
        for (i, v) in vertices.iter().enumerate() {
          let root = find(&mut parent, i);
          if let Some(&idx) = root_to_idx.get(&root) {
            components[idx].push(v.clone());
          } else {
            let idx = components.len();
            root_to_idx.insert(root, idx);
            components.push(vec![v.clone()]);
          }
        }

        // Sort components by size (largest first)
        components.sort_by(|a, b| b.len().cmp(&a.len()));
        components
      };

      return Ok(Expr::List(comp_list.into_iter().map(Expr::List).collect()));
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
              args: vec![Expr::List(comp_verts.clone()), Expr::List(sub_edges)],
            });
          }
        }
        return Ok(Expr::List(subgraphs));
      }
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
    let vertex_index: std::collections::HashMap<String, usize> = vertices
      .iter()
      .enumerate()
      .map(|(i, v)| (expr_to_string(v), i))
      .collect();

    // Union-Find (treat all edges as undirected)
    let mut parent: Vec<usize> = (0..n).collect();
    let mut uf_rank = vec![0usize; n];

    fn find_wcc(parent: &mut Vec<usize>, i: usize) -> usize {
      if parent[i] != i {
        parent[i] = find_wcc(parent, parent[i]);
      }
      parent[i]
    }
    fn union_wcc(
      parent: &mut Vec<usize>,
      rank: &mut Vec<usize>,
      a: usize,
      b: usize,
    ) {
      let ra = find_wcc(parent, a);
      let rb = find_wcc(parent, b);
      if ra == rb {
        return;
      }
      if rank[ra] < rank[rb] {
        parent[ra] = rb;
      } else if rank[ra] > rank[rb] {
        parent[rb] = ra;
      } else {
        parent[rb] = ra;
        rank[ra] += 1;
      }
    }

    for edge in edges {
      if let Expr::FunctionCall { args: eargs, .. } = edge
        && eargs.len() == 2
      {
        let from_str = expr_to_string(&eargs[0]);
        let to_str = expr_to_string(&eargs[1]);
        if let (Some(&fi), Some(&ti)) =
          (vertex_index.get(&from_str), vertex_index.get(&to_str))
        {
          union_wcc(&mut parent, &mut uf_rank, fi, ti);
        }
      }
    }

    let mut components: Vec<Vec<Expr>> = Vec::new();
    let mut root_to_idx: std::collections::HashMap<usize, usize> =
      std::collections::HashMap::new();
    for (i, v) in vertices.iter().enumerate() {
      let root = find_wcc(&mut parent, i);
      if let Some(&idx) = root_to_idx.get(&root) {
        components[idx].push(v.clone());
      } else {
        let idx = components.len();
        root_to_idx.insert(root, idx);
        components.push(vec![v.clone()]);
      }
    }

    // Reverse vertices within each component to match Wolfram's ordering
    for comp in &mut components {
      comp.reverse();
    }
    components.sort_by(|a, b| b.len().cmp(&a.len()));
    return Ok(Expr::List(components.into_iter().map(Expr::List).collect()));
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
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
  }

  // Triangle[] defaults to Triangle[{{0,0},{1,0},{0,1}}]
  if name == "Triangle" {
    if args.is_empty() {
      return Ok(Expr::FunctionCall {
        name: "Triangle".to_string(),
        args: vec![Expr::List(vec![
          Expr::List(vec![Expr::Integer(0), Expr::Integer(0)]),
          Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]),
          Expr::List(vec![Expr::Integer(0), Expr::Integer(1)]),
        ])],
      });
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
        args: vec![verts[i].clone(), verts[i + 1].clone()],
      });
    }
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(verts.clone()), Expr::List(edges)],
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
              args: vec![Expr::Integer(-1), n.clone()],
            };
            let half = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)],
                },
                n_minus_1,
                n.clone(),
              ],
            };
            return Ok(Expr::FunctionCall {
              name: "BinomialDistribution".to_string(),
              args: vec![half, p.clone()],
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
              args: vec![Expr::List(vec![n.clone(), n.clone()])],
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
              args: vec![Expr::Integer(-1), n.clone()],
            };
            return Ok(Expr::FunctionCall {
              name: "BinomialDistribution".to_string(),
              args: vec![n_minus_1, p.clone()],
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
              args: vec![Expr::List(vec![m.clone(), m.clone()])],
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
              args: vec![Expr::List(vec![n.clone(), n.clone()])],
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
              args: vec![Expr::Integer(-1), n.clone()],
            };
            // Flatten Times to get correct parenthesization: ((-1 + n)*n)/2
            let half = Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(1), Expr::Integer(2)],
                },
                n_minus_1.clone(),
                n.clone(),
              ],
            };
            return Ok(Expr::FunctionCall {
              name: "HypergeometricDistribution".to_string(),
              args: vec![m.clone(), n_minus_1, half],
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
                  args: vec![formal_var, graph_dist.clone()],
                };
                return Ok(Expr::FunctionCall {
                  name: "GraphPropertyDistribution".to_string(),
                  args: vec![new_property, new_distributed],
                });
              }
            }
          }
        }
      }
    }
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
    terms.sort_by(|a, b| a.0.cmp(&b.0));

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
          args: vec![slot_x.clone(), Expr::Integer(xp as i128)],
        });
      }
      if yp == 1 {
        factors.push(slot_y.clone());
      } else if yp > 1 {
        factors.push(Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![slot_y.clone(), Expr::Integer(yp as i128)],
        });
      }
      let term = if factors.len() == 1 {
        factors.pop().unwrap()
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: factors,
        }
      };
      term_exprs.push(term);
    }

    let poly_expr = if term_exprs.len() == 1 {
      term_exprs.pop().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: term_exprs,
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
      | "SphericalShell" | "CapsuleShape" => Some(true),
      // Unbounded regions
      "HalfPlane" | "HalfSpace" | "InfiniteLine" | "InfinitePlane"
      | "HalfLine" | "ConicHullRegion" | "AffineHalfSpace" | "AffineSpace" => {
        Some(false)
      }
      _ => None,
    };
    if let Some(b) = result {
      return Ok(Expr::Identifier(
        if b { "True" } else { "False" }.to_string(),
      ));
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
        return Ok(Expr::Identifier(
          if b { "True" } else { "False" }.to_string(),
        ));
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
        return Ok(Expr::List(vec![]));
      }
      // Circular embedding: angle_k = π/2 + k * 2π/n for k = 1..n
      // Snap coordinates near simple rational values (0, ±0.5, ±1) to
      // eliminate platform-dependent ULP differences in f64 trig.
      fn snap_coord(v: f64) -> f64 {
        for &target in &[0.0, 0.5, -0.5, 1.0, -1.0] {
          if (v - target).abs() < 1e-14 {
            return target;
          }
        }
        v
      }
      let coords: Vec<Expr> = (1..=n)
        .map(|k| {
          let angle = std::f64::consts::FRAC_PI_2
            + (k as f64) * 2.0 * std::f64::consts::PI / (n as f64);
          Expr::List(vec![
            Expr::Real(snap_coord(angle.cos())),
            Expr::Real(snap_coord(angle.sin())),
          ])
        })
        .collect();
      return Ok(Expr::List(coords));
    }
    // Return unevaluated for non-graph input
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
        return Ok(Expr::List(history));
      }
    } else if let Expr::List(init_list) = &args[1] {
      // List mode: rules map elements to lists
      if let Expr::List(rule_list) = &args[0] {
        let raw_rules = extract_rules(rule_list);
        let mut rules: Vec<(String, Vec<Expr>)> = Vec::new();
        for (from, to) in &raw_rules {
          if let Expr::List(to_list) = to {
            rules.push((expr_to_string(from), to_list.clone()));
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
          current = next;
          history.push(Expr::List(current.clone()));
        }
        return Ok(Expr::List(history));
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
    return Ok(Expr::Identifier(
      if exists { "True" } else { "False" }.to_string(),
    ));
  }

  // DeleteFile[path] — delete a file
  if name == "DeleteFile"
    && args.len() == 1
    && let Expr::String(path) = &args[0]
  {
    match std::fs::remove_file(path) {
      Ok(()) => return Ok(Expr::Identifier("Null".to_string())),
      Err(e) => {
        crate::emit_message(&format!(
          "DeleteFile::nffil: File not found during DeleteFile[{}]. {}",
          path, e
        ));
        return Ok(Expr::FunctionCall {
          name: "$Failed".to_string(),
          args: vec![],
        });
      }
    }
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
        crate::emit_message(&format!(
          "RenameFile::fdnfnd: Directory or file \"{}\" not found.",
          source
        ));
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
      crate::emit_message(&format!(
        "CopyFile::fdnfnd: Directory or file \"{}\" not found.",
        source
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
  if name == "DeleteMissing"
    && args.len() == 1
    && let Expr::List(items) = &args[0]
  {
    let filtered: Vec<Expr> = items
        .iter()
        .filter(|item| {
          !matches!(item, Expr::FunctionCall { name, .. } if name == "Missing")
        })
        .cloned()
        .collect();
    return Ok(Expr::List(filtered));
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
                      op: crate::syntax::BinaryOperator::Minus,
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
                args: conds,
              }
            };
            cases.push((fargs[i].clone(), cond));
          }
          let default = fargs[n - 1].clone();
          let pw_cases = Expr::List(
            cases
              .into_iter()
              .map(|(val, cond)| Expr::List(vec![val, cond]))
              .collect(),
          );
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![pw_cases, default],
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
                      op: crate::syntax::BinaryOperator::Minus,
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
                args: conds,
              }
            };
            cases.push((fargs[i].clone(), cond));
          }
          let default = fargs[n - 1].clone();
          let pw_cases = Expr::List(
            cases
              .into_iter()
              .map(|(val, cond)| Expr::List(vec![val, cond]))
              .collect(),
          );
          let pw = Expr::FunctionCall {
            name: "Piecewise".to_string(),
            args: vec![pw_cases, default],
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
              Expr::List(vec![Expr::List(vec![Expr::Integer(1), cond])]),
              Expr::Integer(0),
            ],
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
              Expr::List(vec![
                Expr::List(vec![lo, cond_lo]),
                Expr::List(vec![hi, cond_hi]),
              ]),
              x,
            ],
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
        return Ok(Expr::FunctionCall {
          name: name.to_string(),
          args: args.to_vec(),
        });
      }
    } else {
      (None, &args[0])
    };

    if let Expr::List(rows) = matrix {
      let n = rows.len();
      let verts: Vec<Expr> = vertices
        .unwrap_or_else(|| (1..=n).map(|i| Expr::Integer(i as i128)).collect());

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
              args: vec![verts[i].clone(), verts[j].clone()],
            });
          }
        }
      }

      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: vec![Expr::List(verts), Expr::List(edges)],
      });
    }
  }

  // MovingAverage[list, r] — simple moving average with window size r
  if name == "MovingAverage"
    && args.len() == 2
    && let (Expr::List(items), Some(r)) = (
      &args[0],
      match &args[1] {
        Expr::Integer(n) if *n >= 1 => Some(*n as usize),
        _ => None,
      },
    )
  {
    let n = items.len();
    if r > n {
      crate::emit_message(&format!(
        "MovingAverage::arg2: The second argument {} must be a positive integer less than or equal to the length {} of the first argument, or a vector of length less than or equal to the length of the first argument.",
        r, n
      ));
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      });
    }
    let mut result = Vec::with_capacity(n - r + 1);
    for i in 0..=(n - r) {
      let window: Vec<Expr> = items[i..i + r].to_vec();
      let sum = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: window,
      };
      let avg = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(sum),
        right: Box::new(Expr::Integer(r as i128)),
      };
      result.push(evaluate_expr_to_expr(&avg)?);
    }
    return Ok(Expr::List(result));
  }

  // CirclePoints[n] — n equally spaced points on the unit circle
  if name == "CirclePoints" && args.len() == 1 {
    if let Some(n) = match &args[0] {
      Expr::Integer(n) if *n >= 1 => Some(*n as usize),
      _ => None,
    } {
      let mut points = Vec::with_capacity(n);
      for k in 0..n {
        // angle_k = Pi/2 - (n-1)*Pi/n + k*2*Pi/n
        let angle = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Minus,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(Expr::Identifier("Pi".to_string())),
              right: Box::new(Expr::Integer(2)),
            }),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Divide,
              left: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::Integer((n - 1) as i128)),
                right: Box::new(Expr::Identifier("Pi".to_string())),
              }),
              right: Box::new(Expr::Integer(n as i128)),
            }),
          }),
          right: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(Expr::Integer(k as i128 * 2)),
              right: Box::new(Expr::Identifier("Pi".to_string())),
            }),
            right: Box::new(Expr::Integer(n as i128)),
          }),
        };
        let cos_expr = Expr::FunctionCall {
          name: "Cos".to_string(),
          args: vec![angle.clone()],
        };
        let sin_expr = Expr::FunctionCall {
          name: "Sin".to_string(),
          args: vec![angle],
        };
        let cos_val = evaluate_expr_to_expr(&cos_expr)?;
        let sin_val = evaluate_expr_to_expr(&sin_expr)?;
        points.push(Expr::List(vec![cos_val, sin_val]));
      }
      return Ok(Expr::List(points));
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
  }

  // Key[k] is an operator form — return unevaluated (applied via CurriedCall)
  if name == "Key" {
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
  }

  // Darker/Lighter fallback: return unevaluated if color couldn't be resolved
  if name == "Darker" || name == "Lighter" {
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
  }

  // FunctionInterpolation[expr, {x, xmin, xmax}]
  if name == "FunctionInterpolation" && args.len() >= 2 {
    return function_interpolation_ast(args);
  }

  // Functions that return their single argument unchanged (identity/pass-through)
  if matches!(
    name,
    "PermutationProduct"
      | "HornerForm"
      | "Parallelize"
      | "Setting"
      | "TrigFactor"
  ) && args.len() == 1
  {
    return Ok(args[0].clone());
  }

  // Functions that return empty list
  if matches!(
    name,
    "SyntaxInformation"
      | "SystemOptions"
      | "LaunchKernels"
      | "DistributeDefinitions"
  ) {
    return Ok(Expr::List(vec![]));
  }

  // Parallel/system functions that return Null (no-op in single-process mode)
  if matches!(name, "ParallelDo" | "SetSharedVariable" | "PrintTemporary") {
    return Ok(Expr::Identifier("Null".to_string()));
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
        return Ok(Expr::FunctionCall {
          name: "FilePrint".to_string(),
          args: args.to_vec(),
        });
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
      return Ok(Expr::Identifier(
        if is_dir { "True" } else { "False" }.to_string(),
      ));
    }
    return Ok(Expr::Identifier("False".to_string()));
  }

  // Vectors[x] defaults to Vectors[x, Complexes]
  if name == "Vectors" && args.len() == 1 {
    return Ok(Expr::FunctionCall {
      name: "Vectors".to_string(),
      args: vec![args[0].clone(), Expr::Identifier("Complexes".to_string())],
    });
  }

  // AbsArg[x] → {Abs[x], Arg[x]}
  if name == "AbsArg" && args.len() == 1 {
    return Ok(Expr::List(vec![
      Expr::FunctionCall {
        name: "Abs".to_string(),
        args: vec![args[0].clone()],
      },
      Expr::FunctionCall {
        name: "Arg".to_string(),
        args: vec![args[0].clone()],
      },
    ]));
  }

  // FirstCase[x, y] returns Missing["NotFound"] when x is not a list
  if name == "FirstCase"
    && args.len() == 2
    && !matches!(&args[0], Expr::List(_))
  {
    return Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![Expr::String("NotFound".to_string())],
    });
  }

  // Neural network layer/model functions return $Failed for invalid arguments
  if matches!(
    name,
    "TotalLayer"
      | "NetEncoder"
      | "ElementwiseLayer"
      | "SoftmaxLayer"
      | "PoolingLayer"
      | "BatchNormalizationLayer"
      | "NetModel"
      | "NetDecoder"
  ) {
    return Ok(Expr::Identifier("$Failed".to_string()));
  }

  // System/UI functions that return $Failed for unsupported operations
  if matches!(
    name,
    "CreatePalette" | "DialogInput" | "CopyToClipboard" | "ResourceObject"
  ) {
    return Ok(Expr::Identifier("$Failed".to_string()));
  }

  // Morphological operations: Opening, Closing, Erosion, Dilation
  if matches!(name, "Opening" | "Closing" | "Erosion" | "Dilation")
    && args.len() == 2
    && let Some(result) = morphological_op(name, &args[0], &args[1])
  {
    return result;
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
      | "GroupOrbits"
      | "RegionBounds"
      | "TensorContract"
      | "LocatorAutoCreate"
      | "LegendFunction"
      | "RasterSize"
      | "TransformedField"
      | "Conditioned"
      | "GeoProjection"
      | "SoundVolume"
      | "GradientFilter"
      | "RegionNearest"
      | "TildeTilde"
      | "NotebookClose"
      | "Failure"
      | "TimeValue"
      | "LineIndent"
      | "LayeredGraphPlot"
      | "WordCharacter"
      | "ReflectionTransform"
      | "BSplineBasis"
      | "ParameterMixtureDistribution"
      | "BinaryReadList"
      | "FindDistributionParameters"
      | "FindPath"
      | "FindPeaks"
      | "NProbability"
      | "DedekindEta"
      | "PixelValuePositions"
      | "Weights"
      | "WhitespaceCharacter"
      | "BarChart3D"
      | "VerticalSlider"
      | "CycleGraph"
      | "OverDot"
      | "MaxPlotPoints"
      | "PermutationCycles"
      | "AnimationRepetitions"
      | "ARMAProcess"
      | "FileNameTake"
      | "UndoTrackedVariables"
      | "VectorColorFunction"
      | "NotebookGet"
      | "Visible"
      | "TruncatedDistribution"
      | "NotebookFind"
      | "ClassifierMeasurements"
      | "EstimatedProcess"
      | "HighlightMesh"
      | "Animator"
      | "AutoScroll"
      | "ConfidenceLevel"
      | "CoefficientRules"
      | "Thinning"
      | "Tolerance"
      | "NetInitialize"
      | "BoundaryMeshRegion"
      | "GeometricBrownianMotionProcess"
      | "SelectComponents"
      | "MeshCellStyle"
      | "NotebookPut"
      | "TextSentences"
      | "PolynomialReduce"
      | "Cumulant"
      | "ThreeJSymbol"

      | "Magnify"
      | "ScriptBaselineShifts"
      | "LineSpacing"
      | "FunctionRange"
      | "SectorOrigin"
      | "MaxTrainingRounds"
      | "PolarAxes"
      | "SystemDialogInput"
      | "ARProcess"
      | "DiscreteWaveletTransform"
      | "RelationGraph"
      | "ImagePartition"
      | "PetersenGraph"
      | "RSolveValue"
      | "FeatureExtraction"
      | "GraphDistance"
      | "CellStyle"
      | "ImageIdentify"
      | "Asymptotic"
      | "CoordinateTransform"
      | "WindowMargins"
      | "AffineTransform"
      | "RadioButton"
      | "LegendMarkers"
      | "PowersRepresentations"
      | "ShowStringCharacters"
      | "NDEigensystem"
      | "TextureCoordinateFunction"
      | "FindDistribution"
      | "TextCases"
      | "Multicolumn"
      | "Record"
      | "WhittakerM"
      | "InterpretationBox"
      | "IncludePods"
      | "RulePlot"
      | "MathieuGroupM11"
      | "Trig"
      | "Overlaps"
      | "ItoProcess"
      | "RotationAction"
      | "Ket"
      | "DiscreteMarkovProcess"
      | "BoundaryDiscretizeGraphics"
      | "TradingChart"
      | "FindMaxValue"
      | "FormPage"
      | "NearestNeighborGraph"

      | "RiemannSiegelZ"
      | "ChartBaseStyle"
      | "MoonPhase"
      | "HazardFunction"
      | "ContentSize"
      | "WordBoundary"
      | "NExpectation"
      | "Mouseover"
      | "RectangleChart"
      | "AffineStateSpaceModel"
      | "LogLikelihood"
      | "SpanFromAbove"
      | "MinValue"
      | "SubPlus"
      | "Extension"
      | "WeightedAdjacencyGraph"
      | "CellFrame"
      | "Compiled"
      | "AudioGenerator"
      | "Underlined"
      | "FourierCoefficient"
      | "Overscript"
      | "Primes"
      | "CommunityGraphPlot"
      | "RandomPrime"
      | "SuperDagger"
      | "ReImPlot"
      | "ExponentFunction"
      | "ProductDistribution"
      | "TogglerBar"
      | "RegionDimension"
      | "FeatureExtractor"
      | "ArgMax"
      | "VertexNormals"
      | "CorrelationFunction"
      | "BellY"
      | "BarnesG"
      | "URL"
      | "FindGeometricTransform"
      | "Deployed"
      | "DirichletDistribution"
      | "RiemannSiegelTheta"
      | "RandomInstance"
      | "NotebookDelete"
      | "FindFormula"
      | "Graph3D"
      | "WhittakerW"
      | "MaxDetect"
      | "GeometricScene"
      | "ClusteringComponents"
      | "BernoulliGraphDistribution"
      | "MandelbrotSetPlot"
      | "Language"
      | "SequenceCases"
      | "TimeConstraint"
      | "DoubleRightTee"
      | "Matrices"
      | "JoinedCurve"
      | "RunProcess"
      | "StartingStepSize"
      | "DefaultButton"
      | "Trigger"
      | "GeoMarker"
      | "ContentSelectable"
      | "ExportForm"
      | "ParallelSubmit"
      | "Application"
      | "FindFile"
      | "DistanceTransform"
      | "TimelinePlot"
      | "PassEventsDown"
      | "CircleDot"
      | "VectorScaling"
      | "FindGeneratingFunction"
      | "AssociateTo"
      | "HistogramDistribution"
      | "GaussianMatrix"
      | "TextRecognize"
      | "NumberSigns"
      | "WeierstrassZeta"
      | "ListSurfacePlot3D"
      | "FRatioDistribution"
      | "DateValue"
      | "DensityPlot3D"
      | "GeoRegionValuePlot"
      | "MaxExtraConditions"
      | "TimeSeriesModelFit"
      | "PaneSelector"
      | "URLExecute"

      | "CoordinatesToolOptions"
      | "ColorCombine"
      | "Highlighted"
      | "TextGrid"
      | "NumericFunction"
      | "Scrollbars"
      | "ColorSetter"
      | "DistanceMatrix"
      | "InverseWaveletTransform"
      // TreeGraph is now implemented in plotting.rs
      | "PadeApproximant"
      | "FillingTransform"
      | "SamplingPeriod"
      | "FindCycle"
      | "TimeSeriesForecast"
      | "Cube"
      | "CharacteristicFunction"
      | "PermutationReplace"
      | "DiscreteVariables"
      | "StripOnInput"
      | "SubMinus"
      | "CornerNeighbors"
      | "TriangularDistribution"
      | "RealExponent"
      | "ColorQuantize"
      | "BinaryWrite"
      | "CheckboxBar"
      | "TooltipDelay"
      | "RandomPermutation"
      | "WatershedComponents"
      | "FactorialMoment"
      | "ViewCenter"
      | "QuantilePlot"
      | "FourierSinSeries"
      | "MathieuCharacteristicA"

      | "StieltjesGamma"
      | "PolarTicks"
      | "BeckmannDistribution"
      | "WeierstrassSigma"
      | "MathieuC"
      | "MetaInformation"
      | "NotebookSave"
      | "ListContourPlot3D"
      | "ResamplingMethod"
      | "AngularGauge"
      | "ColorReplace"
      | "GraphPlot3D"
      | "ButtonFunction"
      | "Sunday"
      | "FrobeniusSolve"
      | "ImageValue"
      | "GeneratedParameters"
      | "PlotRegion"
      | "MatrixLog"
      | "DensityHistogram"
      | "DistributionChart"
      | "InverseZTransform"
      | "IncidenceMatrix"
      | "Notebooks"
      | "ZTransform"
      | "LeastSquares"
      | "FeatureTypes"
      | "CovarianceFunction"
      | "XYZColor"
      | "GraphHighlightStyle"
      | "ImageTrim"
      | "BSplineSurface"
      | "SingularValueList"
      | "MorphologicalBinarize"
      | "VertexWeight"
      | "SingleLetterItalics"
      | "PolarGridLines"
      | "RootApproximant"
      | "Interpretation"
      | "SymmetricGroup"
      | "DihedralGroup"
      | "AlternatingGroup"
      | "WattsStrogatzGraphDistribution"
      | "BarabasiAlbertGraphDistribution"
      | "SpatialGraphDistribution"
      | "UniformGraphDistribution"
      | "Databin"
      | "SmoothDensityHistogram"
      | "NetExtract"
      | "HankelH1"
      | "Friday"
      | "CloudImport"
      | "Temporary"
      | "ServiceConnect"
      | "NonlinearStateSpaceModel"
      | "DefaultDuration"
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
      | "XMLObject"
      | "UnderoverscriptBox"
      | "ForwardBackward"
  ) {
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
  }

  // ClearSystemCache[] - no-op, returns Null
  if name == "ClearSystemCache" {
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Check if the function is a known but unimplemented Wolfram Language function
  if is_known_wolfram_function(name) {
    let args_str = args
      .iter()
      .map(expr_to_string)
      .collect::<Vec<_>>()
      .join(", ");
    let call_str = format!("{}[{}]", name, args_str);
    crate::capture_unimplemented_call(&call_str);
  }

  // Unknown function - return as symbolic function call
  Ok(Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec(),
  })
}

/// Extract RGB components from a color expression as (numerator, denominator) pairs.
/// Returns None if the expression is not a recognized color.
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
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
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
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

fn gcd_i128(a: i128, b: i128) -> i128 {
  if b == 0 { a } else { gcd_i128(b, a % b) }
}

/// Build undirected adjacency list from graph vertices and edges
fn build_undirected_adj(vertices: &[Expr], edges: &[Expr]) -> Vec<Vec<usize>> {
  let n = vertices.len();
  let vertex_index: std::collections::HashMap<String, usize> = vertices
    .iter()
    .enumerate()
    .map(|(i, v)| (expr_to_string(v), i))
    .collect();
  let mut adj = vec![vec![]; n];
  for edge in edges {
    if let Expr::FunctionCall { args: eargs, .. } = edge
      && eargs.len() == 2
    {
      let from_str = expr_to_string(&eargs[0]);
      let to_str = expr_to_string(&eargs[1]);
      if let (Some(&fi), Some(&ti)) =
        (vertex_index.get(&from_str), vertex_index.get(&to_str))
      {
        adj[fi].push(ti);
        adj[ti].push(fi);
      }
    }
  }
  adj
}

/// Check if undirected graph is connected via BFS
fn is_connected(adj: &[Vec<usize>], n: usize) -> bool {
  if n == 0 {
    return true;
  }
  let mut visited = vec![false; n];
  let mut queue = std::collections::VecDeque::new();
  visited[0] = true;
  queue.push_back(0);
  let mut count = 1;
  while let Some(v) = queue.pop_front() {
    for &u in &adj[v] {
      if !visited[u] {
        visited[u] = true;
        count += 1;
        queue.push_back(u);
      }
    }
  }
  count == n
}

/// BFS from start vertex, return all distances
fn bfs_all_dists(adj: &[Vec<usize>], start: usize, n: usize) -> Vec<i128> {
  let mut dist = vec![-1i128; n];
  let mut queue = std::collections::VecDeque::new();
  dist[start] = 0;
  queue.push_back(start);
  while let Some(v) = queue.pop_front() {
    for &u in &adj[v] {
      if dist[u] == -1 {
        dist[u] = dist[v] + 1;
        queue.push_back(u);
      }
    }
  }
  dist
}

/// BFS from start vertex, return max distance (eccentricity)
fn bfs_max_dist(adj: &[Vec<usize>], start: usize, n: usize) -> i128 {
  let mut dist = vec![-1i128; n];
  let mut queue = std::collections::VecDeque::new();
  dist[start] = 0;
  queue.push_back(start);
  while let Some(v) = queue.pop_front() {
    for &u in &adj[v] {
      if dist[u] == -1 {
        dist[u] = dist[v] + 1;
        queue.push_back(u);
      }
    }
  }
  dist.into_iter().filter(|&d| d >= 0).max().unwrap_or(0)
}

/// Build a rational or integer Expr from (num, den).
fn rational_to_expr(num: i128, den: i128) -> Expr {
  if den == 1 {
    Expr::Integer(num)
  } else if num == 0 {
    Expr::Integer(0)
  } else {
    let g = gcd_i128(num.abs(), den.abs());
    let (n, d) = (num / g, den / g);
    if d == 1 {
      Expr::Integer(n)
    } else {
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(n), Expr::Integer(d)],
      }
    }
  }
}

/// Evaluate Darker[color, amount] or Lighter[color, amount].
/// `is_darker` = true for Darker, false for Lighter.
fn evaluate_darker_lighter(args: &[Expr], is_darker: bool) -> Option<Expr> {
  let rgb = extract_rgb_rational(&args[0])?;

  // Default amount is 1/3
  let (amt_num, amt_den) = if args.len() >= 2 {
    expr_to_rational(&args[1])?
  } else {
    (1, 3)
  };

  let mut result_rgb = [Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)];
  for (i, (c_num, c_den)) in rgb.iter().enumerate() {
    if is_darker {
      // Darker: c * (1 - amount) = c * (den - num) / den
      let factor_num = amt_den - amt_num;
      let factor_den = amt_den;
      let new_num = c_num * factor_num;
      let new_den = c_den * factor_den;
      result_rgb[i] = rational_to_expr(new_num, new_den);
    } else {
      // Lighter: c + amount * (1 - c)
      // = c + amount - amount*c
      // = c*(1 - amount) + amount
      // = c_num/c_den * (amt_den - amt_num)/amt_den + amt_num/amt_den
      // = (c_num*(amt_den-amt_num) + c_den*amt_num) / (c_den * amt_den)
      let new_num = c_num * (amt_den - amt_num) + c_den * amt_num;
      let new_den = c_den * amt_den;
      result_rgb[i] = rational_to_expr(new_num, new_den);
    }
  }

  Some(Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![
      result_rgb[0].clone(),
      result_rgb[1].clone(),
      result_rgb[2].clone(),
    ],
  })
}

/// Check if an expression is a GrayLevel color.
fn is_graylevel(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, args } if name == "GrayLevel" && !args.is_empty())
}

/// Evaluate Blend[{c1, c2, ...}] or Blend[{c1, c2, ...}, t].
fn evaluate_blend(args: &[Expr]) -> Option<Expr> {
  let colors = match &args[0] {
    Expr::List(items) if items.len() >= 2 => items,
    _ => return None,
  };

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
        args: vec![rational_to_expr(num, den * n_i128)],
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
    // Blend[{c1, c2, ...}, t] — interpolation along the color list
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
      args: vec![build_channel(0)],
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
      args: vec![spec[1].clone()],
    };
    let xmax_n = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![spec[2].clone()],
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
              data_points
                .push(Expr::List(vec![Expr::Real(x_val), Expr::Real(y)]));
            }
          }
        }

        if data_points.len() >= 2 {
          let domain = Expr::List(vec![Expr::List(vec![
            Expr::Real(xmin),
            Expr::Real(xmax),
          ])]);
          return Ok(Expr::FunctionCall {
            name: "InterpolatingFunction".to_string(),
            args: vec![domain, Expr::List(data_points), Expr::Integer(3)],
          });
        }
      }
    }
  }
  Ok(Expr::FunctionCall {
    name: "FunctionInterpolation".to_string(),
    args: args.to_vec(),
  })
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
  use crate::syntax::BinaryOperator;

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
      args: terms,
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
  let radius =
    crate::functions::math_ast::try_eval_to_f64(radius_expr)? as usize;

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
      return Ok(Expr::FunctionCall {
        name: "FindMaximumFlow".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let t = match vertex_idx(sink) {
    Some(i) => i,
    None => {
      return Ok(Expr::FunctionCall {
        name: "FindMaximumFlow".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = verts.len();
  let mut cap = vec![vec![0i128; n]; n];
  for edge in edges {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = edge
      && eargs.len() == 2
      && let (Some(u), Some(v)) = (vertex_idx(&eargs[0]), vertex_idx(&eargs[1]))
    {
      cap[u][v] += 1;
      if ename == "UndirectedEdge" {
        cap[v][u] += 1;
      }
    }
  }

  let mut flow = vec![vec![0i128; n]; n];
  let mut max_flow: i128 = 0;

  loop {
    let mut parent = vec![None::<usize>; n];
    let mut visited = vec![false; n];
    visited[s] = true;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(s);

    while let Some(u) = queue.pop_front() {
      if u == t {
        break;
      }
      for v in 0..n {
        if !visited[v] && cap[u][v] - flow[u][v] > 0 {
          visited[v] = true;
          parent[v] = Some(u);
          queue.push_back(v);
        }
      }
    }

    if !visited[t] {
      break;
    }

    let mut bottleneck = i128::MAX;
    let mut v = t;
    while let Some(u) = parent[v] {
      bottleneck = bottleneck.min(cap[u][v] - flow[u][v]);
      v = u;
    }

    v = t;
    while let Some(u) = parent[v] {
      flow[u][v] += bottleneck;
      flow[v][u] -= bottleneck;
      v = u;
    }

    max_flow += bottleneck;
  }

  Ok(Expr::Integer(max_flow))
}

/// FindGraphIsomorphism implementation using backtracking
fn find_graph_isomorphism_impl(
  verts1: &[Expr],
  edges1: &[Expr],
  verts2: &[Expr],
  edges2: &[Expr],
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let n1 = verts1.len();
  let n2 = verts2.len();

  // Graphs must have same number of vertices for isomorphism
  if n1 != n2 {
    return Ok(Expr::List(vec![]));
  }
  let n = n1;

  // Determine how many isomorphisms to find
  let max_count = if args.len() == 3 {
    match &args[2] {
      Expr::Identifier(s) if s == "All" => usize::MAX,
      Expr::Integer(k) if *k > 0 => *k as usize,
      _ => 1,
    }
  } else {
    1
  };

  // Build adjacency matrices using string comparison
  let v1_strs: Vec<String> = verts1.iter().map(expr_to_string).collect();
  let v2_strs: Vec<String> = verts2.iter().map(expr_to_string).collect();

  let mut adj1 = vec![vec![false; n]; n];
  let mut adj2 = vec![vec![false; n]; n];

  // Helper to extract edge endpoints and whether edge is undirected
  let edge_endpoints = |edge: &Expr| -> Option<(String, String, bool)> {
    match edge {
      Expr::Rule {
        pattern,
        replacement,
      } => Some((expr_to_string(pattern), expr_to_string(replacement), false)),
      Expr::FunctionCall { name, args: eargs } if eargs.len() == 2 => {
        let undirected = name == "UndirectedEdge";
        Some((
          expr_to_string(&eargs[0]),
          expr_to_string(&eargs[1]),
          undirected,
        ))
      }
      _ => None,
    }
  };

  for edge in edges1 {
    if let Some((s, t, undirected)) = edge_endpoints(edge)
      && let (Some(i), Some(j)) = (
        v1_strs.iter().position(|v| *v == s),
        v1_strs.iter().position(|v| *v == t),
      )
    {
      adj1[i][j] = true;
      if undirected {
        adj1[j][i] = true;
      }
    }
  }

  for edge in edges2 {
    if let Some((s, t, undirected)) = edge_endpoints(edge)
      && let (Some(i), Some(j)) = (
        v2_strs.iter().position(|v| *v == s),
        v2_strs.iter().position(|v| *v == t),
      )
    {
      adj2[i][j] = true;
      if undirected {
        adj2[j][i] = true;
      }
    }
  }

  // Compute degree sequences for pruning
  let deg1: Vec<usize> = (0..n)
    .map(|i| adj1[i].iter().filter(|&&b| b).count())
    .collect();
  let deg2: Vec<usize> = (0..n)
    .map(|i| adj2[i].iter().filter(|&&b| b).count())
    .collect();

  // Backtracking search
  let mut results: Vec<Vec<usize>> = Vec::new();
  let mut mapping = vec![usize::MAX; n]; // mapping[i] = j means v1[i] -> v2[j]
  let mut used = vec![false; n]; // which v2 vertices are used

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
      if used[j] {
        continue;
      }
      // Degree pruning
      if deg1[depth] != deg2[j] {
        continue;
      }
      // Check adjacency consistency with already mapped vertices (both directions)
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

  // Convert results to associations: {<| v1[0] -> v2[mapping[0]], ... |>}
  let assocs: Vec<Expr> = results
    .iter()
    .map(|m| {
      let rules: Vec<Expr> = (0..n)
        .map(|i| Expr::FunctionCall {
          name: "Rule".to_string(),
          args: vec![verts1[i].clone(), verts2[m[i]].clone()],
        })
        .collect();
      Expr::FunctionCall {
        name: "Association".to_string(),
        args: rules,
      }
    })
    .collect();

  let result = Expr::List(assocs);
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// FindSpanningTree using Kruskal's algorithm (unit weights)
fn find_spanning_tree_impl(
  verts: &[Expr],
  edges: &[Expr],
) -> Result<Expr, InterpreterError> {
  let n = verts.len();
  if n == 0 {
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: vec![Expr::List(vec![]), Expr::List(vec![])],
    });
  }

  let v_strs: Vec<String> = verts.iter().map(expr_to_string).collect();

  // Helper to extract edge endpoints
  let edge_endpoints = |edge: &Expr| -> Option<(usize, usize)> {
    let (s, t) = match edge {
      Expr::Rule {
        pattern,
        replacement,
      } => (expr_to_string(pattern), expr_to_string(replacement)),
      Expr::FunctionCall { args: eargs, .. } if eargs.len() == 2 => {
        (expr_to_string(&eargs[0]), expr_to_string(&eargs[1]))
      }
      _ => return None,
    };
    let i = v_strs.iter().position(|v| *v == s)?;
    let j = v_strs.iter().position(|v| *v == t)?;
    Some((i, j))
  };

  // Union-Find
  let mut parent: Vec<usize> = (0..n).collect();
  let mut rank = vec![0usize; n];

  fn find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
      parent[x] = find(parent, parent[x]);
    }
    parent[x]
  }

  fn union(
    parent: &mut [usize],
    rank: &mut [usize],
    x: usize,
    y: usize,
  ) -> bool {
    let rx = find(parent, x);
    let ry = find(parent, y);
    if rx == ry {
      return false;
    }
    if rank[rx] < rank[ry] {
      parent[rx] = ry;
    } else if rank[rx] > rank[ry] {
      parent[ry] = rx;
    } else {
      parent[ry] = rx;
      rank[rx] += 1;
    }
    true
  }

  // Kruskal's: add edges that don't form cycles
  let mut tree_edges: Vec<Expr> = Vec::new();
  for edge in edges {
    if let Some((i, j)) = edge_endpoints(edge)
      && union(&mut parent, &mut rank, i, j)
    {
      tree_edges.push(edge.clone());
      if tree_edges.len() == n - 1 {
        break;
      }
    }
  }

  Ok(Expr::FunctionCall {
    name: "Graph".to_string(),
    args: vec![Expr::List(verts.to_vec()), Expr::List(tree_edges)],
  })
}
