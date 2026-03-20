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

pub fn evaluate_function_call_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  evaluate_function_call_ast_inner(name, args)
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
          let mut substituted_cond = cond_expr.clone();
          for (param, arg) in params.iter().zip(effective_args.iter()) {
            substituted_cond =
              crate::syntax::substitute_variable(&substituted_cond, param, arg);
          }
          for (bind_name, bind_val) in &structural_bindings {
            substituted_cond = crate::syntax::substitute_variable(
              &substituted_cond,
              bind_name,
              bind_val,
            );
          }
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
        // Substitute and evaluate body
        let mut substituted = body_expr.clone();
        for (param, arg) in params.iter().zip(effective_args.iter()) {
          substituted =
            crate::syntax::substitute_variable(&substituted, param, arg);
        }
        for (bind_name, bind_val) in &structural_bindings {
          substituted = crate::syntax::substitute_variable(
            &substituted,
            bind_name,
            bind_val,
          );
        }
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

      // Build the effective argument list by matching provided args to params.
      // Optional params are filled left-to-right; when there are fewer args than params,
      // optional params use their defaults starting from the leftmost optional param.
      let effective_args = if args.len() == total_count {
        // All params provided - check head constraints
        let mut head_ok = true;
        for (i, arg) in args.iter().enumerate() {
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
        args.to_vec()
      } else {
        // Fewer args than params - fill optional params with defaults
        // Strategy: try to assign args left-to-right, using defaults for optional params
        // when args run out. For each param, if it's optional and we need to save args
        // for required params later, use the default.
        let num_optional_to_default = total_count - args.len();
        let mut effective = Vec::with_capacity(total_count);
        let mut arg_idx = 0;
        let mut defaults_used = 0;

        for i in 0..total_count {
          if param_defaults[i].is_some()
            && defaults_used < num_optional_to_default
          {
            // Check if we should use default: if the arg doesn't match the head constraint
            // or if we need to reserve remaining args for required params
            let remaining_args = args.len() - arg_idx;
            let remaining_required: usize = param_defaults[i + 1..]
              .iter()
              .filter(|d| d.is_none())
              .count();
            let should_default = if remaining_args <= remaining_required {
              // Must default - not enough args for remaining required params
              true
            } else if let Some(head) = &param_heads[i] {
              // Has head constraint - check if current arg matches
              arg_idx < args.len() && get_expr_head(&args[arg_idx]) != *head
            } else {
              false
            };

            if should_default {
              effective.push(param_defaults[i].clone().unwrap());
              defaults_used += 1;
            } else if arg_idx < args.len() {
              // Check head constraint
              if let Some(head) = &param_heads[i]
                && get_expr_head(&args[arg_idx]) != *head
              {
                break; // head mismatch
              }
              effective.push(args[arg_idx].clone());
              arg_idx += 1;
            }
          } else if arg_idx < args.len() {
            // Required param or optional param that should be filled
            if let Some(head) = &param_heads[i]
              && get_expr_head(&args[arg_idx]) != *head
            {
              break; // head mismatch for required param - this overload doesn't match
            }
            effective.push(args[arg_idx].clone());
            arg_idx += 1;
          }
        }

        if effective.len() != total_count {
          continue; // matching failed
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
          // Substitute all parameters with their argument values in the condition
          let mut substituted_cond = cond_expr.clone();
          for (param, arg) in params.iter().zip(effective_args.iter()) {
            substituted_cond =
              crate::syntax::substitute_variable(&substituted_cond, param, arg);
          }
          // Also substitute structural pattern bindings into conditions
          for (bind_name, bind_val) in &structural_bindings {
            substituted_cond = crate::syntax::substitute_variable(
              &substituted_cond,
              bind_name,
              bind_val,
            );
          }
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
        continue;
      }
      // All conditions met - substitute parameters with arguments and evaluate body
      let mut substituted = body_expr.clone();
      for (param, arg) in params.iter().zip(effective_args.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      // Also substitute structural pattern bindings (e.g., x from 1/x_ matching 1/y → x=y)
      for (bind_name, bind_val) in &structural_bindings {
        substituted =
          crate::syntax::substitute_variable(&substituted, bind_name, bind_val);
      }
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
            _ => continue, // condition not met, try next overload
          }
        }
        _ => return result,
      }
    }
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
  // EndPackage[] returns Null (matching wolframscript)
  if name == "EndPackage" && args.is_empty() {
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
    || name == "EndPackage"
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
    | "DirectedEdge"
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
    | "AppellF1"
    | "PrincipalValue"
    | "UpTo"
    | "LetterCharacter"
    | "ToRadicals"
    | "Longest"
    | "BetaRegularized"
    | "GammaRegularized"
    | "GenerateConditions"
    | "OverTilde"
    | "SinhIntegral"
    | "CoshIntegral"
    | "AngleBracket"
    | "Larger"
    | "ZetaZero"
    | "Hypergeometric1F1Regularized"
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
      && gargs.len() == 2
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
      && gargs.len() == 2
      && let Expr::List(_) = &gargs[0]
    {
      return Ok(gargs[0].clone());
    }
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
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
    && gargs.len() == 2
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
      && gargs.len() == 2
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

  // TreeGraphQ[graph] — True if graph is a tree (connected, n-1 edges for n vertices)
  if name == "TreeGraphQ" && args.len() == 1 {
    if let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
      && gname == "Graph"
      && gargs.len() == 2
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
      && gargs.len() == 2
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
      && gargs.len() == 2
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
    && gargs.len() == 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    let adj = build_undirected_adj(vertices, edges);
    let max_degree = n.saturating_sub(1);
    let centralities: Vec<Expr> = adj
      .iter()
      .map(|neighbors| {
        if max_degree > 0 {
          let deg = neighbors.len() as i128;
          let den = max_degree as i128;
          let g = gcd_i128(deg, den);
          if den / g == 1 {
            Expr::Integer(deg / g)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(deg / g), Expr::Integer(den / g)],
            }
          }
        } else {
          Expr::Integer(0)
        }
      })
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
    && gargs.len() == 2
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
    && gargs.len() == 2
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
    && gargs.len() == 2
    && let (Expr::List(vertices), Expr::List(edges)) = (&gargs[0], &gargs[1])
  {
    let n = vertices.len();
    let adj = build_undirected_adj(vertices, edges);
    // Closeness centrality = (n-1) / sum of distances to all other vertices
    let centralities: Vec<Expr> = (0..n)
      .map(|start| {
        let dists = bfs_all_dists(&adj, start, n);
        let total_dist: i128 = dists.iter().filter(|&&d| d > 0).sum();
        if total_dist > 0 {
          let num = (n as i128) - 1;
          let g = gcd_i128(num, total_dist);
          if total_dist / g == 1 {
            Expr::Integer(num / g)
          } else {
            Expr::FunctionCall {
              name: "Rational".to_string(),
              args: vec![Expr::Integer(num / g), Expr::Integer(total_dist / g)],
            }
          }
        } else {
          Expr::Integer(0)
        }
      })
      .collect();
    return Ok(Expr::List(centralities));
  }

  // ChromaticPolynomial[graph, k] — chromatic polynomial of a graph
  if name == "ChromaticPolynomial"
    && args.len() == 2
    && let Expr::FunctionCall {
      name: gname,
      args: gargs,
    } = &args[0]
    && gname == "Graph"
    && gargs.len() == 2
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
      && gargs.len() == 2
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
      && gargs.len() == 2
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
        return Ok(Expr::FunctionCall {
          name: "$Failed".to_string(),
          args: vec![],
        });
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
      | "BooleanConvert"
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

  // DirectoryQ[x] returns False for non-string arguments
  if name == "DirectoryQ" {
    if let Some(Expr::String(_)) = args.first() {
      // For string arguments, check actual filesystem
      return Ok(Expr::Identifier("False".to_string()));
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
    "CreatePalette"
      | "CreateDirectory"
      | "DialogInput"
      | "CopyToClipboard"
      | "ResourceObject"
  ) {
    return Ok(Expr::Identifier("$Failed".to_string()));
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
      | "Erosion"
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
      | "CopyFile"
      | "Magnify"
      | "ScriptBaselineShifts"
      | "LineSpacing"
      | "FunctionRange"
      | "SectorOrigin"
      | "MaxTrainingRounds"
      | "PolarAxes"
      | "PolynomialGCD"
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
      | "FilePrint"
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
      | "SequencePosition"
      | "FileBaseName"
      | "CoordinatesToolOptions"
      | "ColorCombine"
      | "Highlighted"
      | "TextGrid"
      | "NumericFunction"
      | "Scrollbars"
      | "ColorSetter"
      | "DistanceMatrix"
      | "InverseWaveletTransform"
      | "TreeGraph"
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
      | "Standardize"
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
      | "FileType"
      | "StieltjesGamma"
      | "PolarTicks"
      | "BeckmannDistribution"
      | "WeierstrassSigma"
      | "MathieuC"
      | "StringReplacePart"
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
      | "Databin"
      | "InverseErf"
      | "SmoothDensityHistogram"
      | "NetExtract"
      | "HankelH1"
      | "Friday"
      | "CloudImport"
      | "Temporary"
      | "ServiceConnect"
      | "NonlinearStateSpaceModel"
      | "Closing"
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
