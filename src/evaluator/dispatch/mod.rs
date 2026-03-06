#[allow(unused_imports)]
use super::*;

// Re-export crate types/functions for submodules (used by submodules via `use super::*`)
#[allow(unused_imports)]
pub(crate) use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
};
#[allow(unused_imports)]
pub(crate) use crate::{
  ENV, InterpreterError, PART_DEPTH, StoredValue, format_real_result,
  format_result, interpret,
};

mod association_functions;
mod attributes;
mod boolean_functions;
mod calculus_functions;
mod complex_and_special;
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
) -> Option<Vec<(String, Expr)>> {
  // Find the __opts parameter
  let opts_idx = params.iter().position(|p| p.starts_with("__opts"))?;
  let opts_arg = &effective_args[opts_idx];

  // Get stored defaults from Options[func_name]
  let defaults: Vec<Expr> = crate::FUNC_OPTIONS
    .with(|m| m.borrow().get(func_name).cloned())
    .unwrap_or_default();

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

  if let Some(overloads) = overloads {
    for (
      params,
      conditions,
      param_defaults,
      param_heads,
      blank_types,
      body_expr,
    ) in &overloads
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
        let opt_bindings =
          collect_option_bindings(name, params, &effective_args);
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
      let opt_bindings = collect_option_bindings(name, params, &effective_args);
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
    | "Entity" => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      });
    }
    _ => {}
  }

  // Graph[{rule1, rule2, ...}] → Graph[{sorted vertices}, {DirectedEdge[...], ...}]
  if name == "Graph" {
    if args.len() == 1
      && let Expr::List(edges) = &args[0]
    {
      // Check if all elements are Rule expressions
      let all_rules = edges.iter().all(|e| matches!(e, Expr::Rule { .. }));
      if all_rules && !edges.is_empty() {
        // Extract unique vertices
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
            // Add to vertex set if not already present
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
        // Sort vertices canonically
        vertex_set.sort_by(crate::functions::canonical_cmp);
        return Ok(Expr::FunctionCall {
          name: "Graph".to_string(),
          args: vec![Expr::List(vertex_set), Expr::List(directed_edges)],
        });
      }
    }
    // Fall through: return as inert
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

  // Darker/Lighter fallback: return unevaluated if color couldn't be resolved
  if name == "Darker" || name == "Lighter" {
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
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
